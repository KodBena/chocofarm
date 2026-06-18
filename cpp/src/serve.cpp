// cpp/src/serve.cpp
// Purpose: the persistent --serve control loop (see serve.hpp) — the C++ side of the ActorTransport.
//   It holds the env + net + policy live across generations and dispatches the control_spec protocol
//   (configure / generate / ping / shutdown). configure adopts an ActorConfig: it builds the env ONCE
//   (first configure, from instance/faces), rebuilds nothing else; a later instance/faces change is a
//   loud instance_knob_changed reject, while the HOT search knobs are adopted live (the policy rebuilds
//   lazily on the next generate, since it needs the net). generate runs the two gates (config_epoch for
//   config adoption, version for weight reload — independent), reloads the net only when version
//   advances, rebuilds the policy when the net or the config changed, and replays via run_episodes. The
//   loop never throws: every boundary failure is a typed control_spec error reply (ADR-0002 / P9).
//
// Public Domain (The Unlicense).
#include "chocofarm/serve.hpp"

#include <cstdint>
#include <exception>
#include <istream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "chocofarm/actor_config.hpp"
#include "chocofarm/control_spec.hpp"
#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/instance.hpp"
#include "chocofarm/net.hpp"
#include "chocofarm/runner.hpp"
#include "chocofarm/runner_wire_batched.hpp"
#include "chocofarm/runtime_config.hpp"

namespace chocofarm {
namespace {

using json = nlohmann::json;
namespace ctl = chocofarm::control;

std::string key(std::string_view k) { return std::string(k); }  // string_view -> json key

void write_reply(std::ostream& out, const json& j) {
    out << j.dump() << "\n";
    out.flush();
}

json ok_reply() {
    json j;
    j[key(ctl::KEY_OK)] = true;
    return j;
}

json err_reply(std::string_view tag, std::string detail) {
    json j;
    j[key(ctl::KEY_OK)] = false;
    j[key(ctl::KEY_ERROR)] = std::string(tag);
    j[key(ctl::KEY_DETAIL)] = std::move(detail);
    return j;
}

// The persistent context the daemon win preserves: env+fb built ONCE (heap, so the env<-inst, fb<-env,
// policy<-net references stay valid for the process's life), the net version-gated, the policy rebuilt on
// a net reload or a HOT-config change. unique_ptr (not std::optional) for the held objects whose address
// must be stable across the run.
struct ServeState {
    std::string run;                          // the redis weight-key namespace (the --run startup arg)
    ServeOptions opts;                        // the wire-path startup knobs (endpoint + pool threads/batch)
    std::unique_ptr<Instance> inst;           // the loaded geometry (built once, first configure)
    std::unique_ptr<Environment> env;
    std::unique_ptr<FeatureBuilder> fb;
    std::string instance_path, faces_path;    // the adopted paths (for the instance-change reject)
    GumbelConfig gc{};                        // the current HOT search knobs
    std::optional<NetForward> net;            // the leaf net (version-gated reload)
    int loaded_version = -1;
    std::unique_ptr<GumbelAZPolicy> policy;   // rebuilt on a net reload or a HOT-config change
    int policy_built_for_epoch = -1;
    int epoch = 0;                            // increments per adopted configure

    [[nodiscard]] bool serving() const { return env != nullptr && epoch > 0; }
};

json handle_configure(ServeState& st, const json& msg) {
    if (!msg.contains(key(ctl::KEY_CONFIG)))
        return err_reply(ctl::ERR_MISSING_FIELD, "configure message missing 'config'");
    auto cfg = actor_config_from_json(msg.at(key(ctl::KEY_CONFIG)));
    if (!cfg)
        return err_reply(ctl::ERR_INVALID_CONFIG, cfg.error().message);

    // INSTANCE knobs: build the env ONCE; a later change is a NEW experiment, a loud reject.
    if (st.env == nullptr) {
        auto inst = load_instance(cfg->instance_path, cfg->faces_path);
        if (!inst)
            return err_reply(ctl::ERR_INVALID_CONFIG, "instance load failed: " + inst.error().message);
        try {
            st.inst = std::make_unique<Instance>(std::move(*inst));
            st.env = std::make_unique<Environment>(*st.inst);
            st.fb = std::make_unique<FeatureBuilder>(*st.env);
        } catch (const std::exception& e) {
            // building the env/feature geometry is the one throw surface here (e.g. std::bad_alloc on the
            // world-set / distance arrays); translate it to a typed reply, never an abort (P9 / ADR-0002).
            st.inst.reset(); st.env.reset(); st.fb.reset();  // leave no half-built context (serving() stays false)
            return err_reply(ctl::ERR_INVALID_CONFIG, std::string("env build raised: ") + e.what());
        }
        st.instance_path = cfg->instance_path;
        st.faces_path = cfg->faces_path;
    } else if (cfg->instance_path != st.instance_path || cfg->faces_path != st.faces_path) {
        return err_reply(ctl::ERR_INSTANCE_KNOB,
                         "instance/faces changed live (" + st.instance_path + " / " + st.faces_path +
                         " -> " + cfg->instance_path + " / " + cfg->faces_path +
                         ") — an INSTANCE change is a NEW experiment; restart the runner");
    }

    // HOT knobs: adopt the new GumbelConfig and advance the epoch. The policy rebuilds lazily on the next
    // generate (it needs the net, which is version-gated there).
    st.gc = cfg->gumbel;
    st.epoch += 1;
    json j = ok_reply();
    j[key(ctl::KEY_CONFIG_EPOCH)] = st.epoch;
    return j;
}

json handle_generate(ServeState& st, RedisClient& redis, const json& msg) {
    if (!st.serving())
        return err_reply(ctl::ERR_NOT_CONFIGURED, "generate before a successful configure");

    int req_epoch = 0, version = 0, episodes = 0, max_steps = 0;
    std::uint64_t seed = 0;
    double lam = 0.0;
    std::string res_token;
    try {
        req_epoch = msg.at(key(ctl::KEY_CONFIG_EPOCH)).get<int>();
        version = msg.at(key(ctl::KEY_VERSION)).get<int>();
        seed = msg.at(key(ctl::KEY_SEED)).get<std::uint64_t>();
        lam = msg.at(key(ctl::KEY_LAM)).get<double>();
        episodes = msg.at(key(ctl::KEY_EPISODES)).get<int>();
        max_steps = msg.at(key(ctl::KEY_MAX_STEPS)).get<int>();
        res_token = msg.at(key(ctl::KEY_RES_TOKEN)).get<std::string>();
    } catch (const json::exception& e) {
        return err_reply(ctl::ERR_MISSING_FIELD, std::string("generate field error: ") + e.what());
    }

    // gate 1 — config_epoch (config adoption): refuse to generate under a config the client did not
    // think was live. The common case (new weights, unchanged config) is new-version / SAME-epoch and
    // passes this gate (the version gate below is independent).
    if (req_epoch != st.epoch)
        return err_reply(ctl::ERR_EPOCH_MISMATCH,
                         "config_epoch " + std::to_string(req_epoch) + " != live epoch " +
                         std::to_string(st.epoch));

    // gate 2 + the compute, wrapped so a search-internal throw (e.g. std::bad_alloc from the unbounded
    // _Node arena) becomes a typed ERR_GENERATE_FAILED reply, NOT a process abort — the throw-free-loop
    // contract: a failed generate is a REPLY, not a std::terminate (ADR-0002 / P9). run_episodes's OWN
    // boundary failures (a missing weight payload, a failed redis write) are already typed std::expected
    // returns; this catch additionally covers the recoverable exhaustion surface the search can raise.
    try {
        // the per-generation scalars ride here (P4); both dispatch arms share this config.
        RunnerConfig rcfg;
        rcfg.run = st.run;
        rcfg.phase = "gen";  // the C++ actor only generates; eval stays in-process Python (ADR-0008)
        rcfg.version = version;
        rcfg.episodes = episodes;
        rcfg.lam = lam;
        rcfg.max_steps = max_steps;
        rcfg.seed = seed;
        rcfg.res_token = res_token;

        std::expected<int, Error> written = 0;
        if (!st.opts.infer_endpoint.empty()) {
            // ---- the WIRE path (BINARY dispatch on --infer-endpoint, Override O-2) ----
            // The leaf is resolved REMOTELY on the JAX InferenceServer — so the local NetForward reload
            // (gate 2's local reload) and the local GumbelAZPolicy build are BOTH SKIPPED; the wire driver
            // holds the YieldingNetEvaluator per TreeState. The two-gate version discipline is preserved on
            // the CONTROL channel (the version is still echoed below), and the server reloads its own
            // weights between generates (publish-then-bump, cpp_executor.py). No second weight read here.
            WireRunnerConfig wcfg;
            wcfg.endpoint = st.opts.infer_endpoint;
            // the pool knobs: the --serve startup args override RuntimeConfig::from_env's host-sized
            // defaults (the ONE home derivation; 0 means "take the env/default"). fibers_per_thread is
            // derived in run_episodes_wire_batched from these two (RuntimeConfig).
            RuntimeConfig rc = RuntimeConfig::from_env();
            wcfg.pool_threads = (st.opts.pool_threads > 0) ? st.opts.pool_threads : rc.thread_pool_size;
            wcfg.pool_batch = (st.opts.pool_batch > 0) ? st.opts.pool_batch : rc.batch_size;
            written = run_episodes_wire_batched(*st.env, *st.fb, st.gc, redis, rcfg, wcfg, nullptr);
        } else {
            // ---- the SERIAL path (no endpoint): local NetForward leaf, the original behaviour ----
            // gate 2 — version (weight reload), INDEPENDENT of the epoch: reload the net only when version
            // advances. The policy holds the OLD net by reference, so destroy it BEFORE replacing the net.
            if (version != st.loaded_version) {
                auto wp = redis.read_weights(st.run, "gen", version);
                if (!wp)
                    return err_reply(ctl::ERR_WEIGHT_READ, wp.error().message);
                auto nf = NetForward::create(*wp);
                if (!nf)
                    return err_reply(ctl::ERR_WEIGHT_READ, "net build failed: " + nf.error().message);
                st.policy.reset();
                st.net.emplace(std::move(*nf));
                st.loaded_version = version;
                st.policy_built_for_epoch = -1;  // force a rebuild against the new net
            }

            // (re)build the policy if the net reloaded or the HOT config changed since the last build.
            if (st.policy == nullptr || st.policy_built_for_epoch != st.epoch) {
                st.policy = std::make_unique<GumbelAZPolicy>(st.gc, *st.net, *st.env);
                st.policy_built_for_epoch = st.epoch;
            }

            // replay via the SHARED episode loop (no second weight read — the net is already loaded; P1).
            written = run_episodes(*st.env, *st.fb, *st.policy, redis, rcfg, nullptr);
        }
        if (!written)
            return err_reply(ctl::ERR_GENERATE_FAILED, written.error().message);

        json j = ok_reply();
        j[key(ctl::KEY_WRITTEN)] = *written;
        j[key(ctl::KEY_CONFIG_EPOCH)] = st.epoch;     // echo — the client asserts the round-trip matched
        j[key(ctl::KEY_VERSION)] = version;
        return j;
    } catch (const std::exception& e) {
        return err_reply(ctl::ERR_GENERATE_FAILED, std::string("generate raised: ") + e.what());
    }
}

json handle_ping(const ServeState& st) {
    json j = ok_reply();
    j[key(ctl::KEY_SERVING)] = st.serving();
    j[key(ctl::KEY_CONFIG_EPOCH)] = st.epoch;
    return j;
}

}  // namespace

int serve(RedisClient& redis, const std::string& run, const ServeOptions& opts, std::istream& in,
          std::ostream& out) {
    ServeState st;
    st.run = run;
    st.opts = opts;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty())
            continue;
        json msg;
        try {
            msg = json::parse(line);
        } catch (const json::exception& e) {
            write_reply(out, err_reply(ctl::ERR_BAD_JSON, std::string("invalid JSON: ") + e.what()));
            continue;
        }
        if (!msg.is_object() || !msg.contains(key(ctl::KEY_TYPE)) ||
            !msg.at(key(ctl::KEY_TYPE)).is_string()) {
            write_reply(out, err_reply(ctl::ERR_MISSING_FIELD, "message missing string 'type'"));
            continue;
        }
        std::string type = msg.at(key(ctl::KEY_TYPE)).get<std::string>();
        std::string_view tv{type};
        if (tv == ctl::MSG_CONFIGURE) {
            write_reply(out, handle_configure(st, msg));
        } else if (tv == ctl::MSG_GENERATE) {
            write_reply(out, handle_generate(st, redis, msg));
        } else if (tv == ctl::MSG_PING) {
            write_reply(out, handle_ping(st));
        } else if (tv == ctl::MSG_SHUTDOWN) {
            write_reply(out, ok_reply());
            return 0;  // graceful exit
        } else {
            write_reply(out, err_reply(ctl::ERR_UNKNOWN_TYPE, "unknown message type: " + type));
        }
    }
    return 0;  // stdin closed (EOF) — treat as shutdown
}

}  // namespace chocofarm
