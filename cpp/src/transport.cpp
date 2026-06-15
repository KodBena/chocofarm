// cpp/src/transport.cpp
// Purpose: the C++ redis wire client (hiredis) — the wire as the ONLY contract (ADR-0012 P7). See
//   transport.hpp. It mirrors chocofarm/az/transport.py + config.py byte-for-byte: the key
//   namespace, the manifest-driven float64 weight read (no hardcoded offsets — P1), the four
//   float32 result blocks (no second encoder — P7), and the CHOCO_REDIS_* connection contract
//   (default 127.0.0.1:6379 db0 — no hardcoded port; mirrors config.redis_params).
//
// Public Domain (The Unlicense).
#include "chocofarm/transport.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <hiredis/hiredis.h>
#include <nlohmann/json.hpp>

namespace chocofarm {

using nlohmann::json;

// ---- env contract (mirrors config.redis_params / config._result_ttl) ----
static std::string env_str(const char* name, const std::string& dflt) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : dflt;
}
static int env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : dflt;
}
int result_ttl() { return env_int("CHOCO_RESULT_TTL", 3600); }  // mirrors transport._result_ttl

// ---- key namespace (byte-identical to transport.weight_keys / result_keys) ----
std::pair<std::string, std::string> weight_keys(const std::string& run, const std::string& phase,
                                                 int version) {
    std::string base = "az:w:" + run + ":" + phase + ":" + std::to_string(version);
    return {base + ":m", base + ":b"};
}
ResultKeys result_keys(const std::string& res_token, int idx) {
    std::string base = "az:res:" + res_token + ":" + std::to_string(idx);
    return ResultKeys{base + ":X", base + ":PI", base + ":M", base + ":Y"};
}

// ---- connection ----
RedisClient::RedisClient() {
    // CHOCO_REDIS_HOST / CHOCO_REDIS_PORT / CHOCO_REDIS_DB — the SAME env contract as config.py
    // (defaults 127.0.0.1 / 6379 / 0). No hardcoded port (ADR-0012 P7 / P1: config.py is the one
    // owner of "which redis"; this just reads the same env vars so it lands on the same instance).
    std::string host = env_str("CHOCO_REDIS_HOST", "127.0.0.1");
    int port = env_int("CHOCO_REDIS_PORT", 6379);
    int db = env_int("CHOCO_REDIS_DB", 0);

    ctx_ = redisConnect(host.c_str(), port);
    if (ctx_ == nullptr || ctx_->err) {
        std::string msg = ctx_ ? ctx_->errstr : "allocation failure";
        if (ctx_) { redisFree(ctx_); ctx_ = nullptr; }
        // ADR-0002 / P5: fail loud NOW if redis is unreachable, not mid-run (mirrors connect()'s ping).
        throw std::runtime_error("chocofarm: redis connect failed (" + host + ":" +
                                 std::to_string(port) + "): " + msg);
    }
    if (db != 0) {
        redisReply* r = static_cast<redisReply*>(redisCommand(ctx_, "SELECT %d", db));
        if (r == nullptr) throw std::runtime_error("chocofarm: redis SELECT failed (no reply)");
        bool ok = (r->type != REDIS_REPLY_ERROR);
        freeReplyObject(r);
        if (!ok) throw std::runtime_error("chocofarm: redis SELECT db failed");
    }
    // fail loud now if unreachable (ADR-0002), mirroring r.ping() in transport.connect()
    redisReply* pong = static_cast<redisReply*>(redisCommand(ctx_, "PING"));
    if (pong == nullptr) throw std::runtime_error("chocofarm: redis PING failed (no reply)");
    freeReplyObject(pong);
}

RedisClient::~RedisClient() {
    if (ctx_) redisFree(ctx_);
}

// Fetch a raw-bytes value; returns false (and leaves `out` empty) on a nil reply (a missing key).
static bool redis_get_bytes(redisContext* ctx, const std::string& key, std::string& out) {
    redisReply* r = static_cast<redisReply*>(redisCommand(ctx, "GET %b", key.data(), key.size()));
    if (r == nullptr) throw std::runtime_error("chocofarm: redis GET failed (no reply) for " + key);
    bool present = (r->type == REDIS_REPLY_STRING);
    if (present) out.assign(r->str, r->len);   // raw bytes (may contain NULs — use len, not strlen)
    bool nil = (r->type == REDIS_REPLY_NIL);
    freeReplyObject(r);
    if (!present && !nil) throw std::runtime_error("chocofarm: redis GET unexpected reply for " + key);
    return present;
}

WeightPayload RedisClient::read_weights(const std::string& run, const std::string& phase,
                                        int version) {
    auto [mk, bk] = weight_keys(run, phase, version);
    std::string manifest, blob;
    bool have_m = redis_get_bytes(ctx_, mk, manifest);
    bool have_b = redis_get_bytes(ctx_, bk, blob);
    if (!have_m || !have_b) {
        // mirrors read_weights' RuntimeError (ADR-0002 / P5: never a silent stale-net serve).
        throw std::runtime_error("weight payload az:w:" + run + ":" + phase + ":" +
                                 std::to_string(version) + " missing from redis");
    }

    json m = json::parse(manifest);
    WeightPayload p;
    p.in_dim = m.at("in_dim").get<int>();
    p.H = m.at("H").get<int>();
    // n_actions may be null in the manifest (a value-only net); coerce null -> -1.
    p.n_actions = m.at("n_actions").is_null() ? -1 : m.at("n_actions").get<int>();
    p.y_mean = m.at("y_mean").get<double>();
    p.y_std = m.at("y_std").get<double>();
    p.residual = m.value("residual", false);  // older manifests without residual -> OFF (P1)

    // Bind each weight by the MANIFEST's (off, len, shape, dtype) — NO hardcoded offsets (ADR-0012
    // P1: a hardcoded offset is the cross-language form of the three-writer feature-layout cancer).
    for (const auto& e : m.at("layout")) {
        WeightEntry we;
        we.name = e.at("name").get<std::string>();
        we.dtype = e.at("dtype").get<std::string>();
        we.off = e.at("off").get<int64_t>();
        we.len = e.at("len").get<int64_t>();
        for (const auto& s : e.at("shape")) we.shape.push_back(s.get<int64_t>());

        // The blob is contiguous float64 ('<f8') — match it exactly (weights are float64, results
        // float32). Reject anything else loudly rather than misread (Port/ACL: translate-and-
        // validate, do not coerce; ADR-0012 P2).
        if (we.dtype != "<f8") {
            throw std::runtime_error("chocofarm: unexpected weight dtype '" + we.dtype +
                                     "' (expected '<f8'); the weight blob is float64");
        }
        if (we.off < 0 || we.len < 0 || we.off + we.len > static_cast<int64_t>(blob.size())) {
            throw std::runtime_error("chocofarm: weight entry " + we.name +
                                     " (off,len) exceeds blob bounds");
        }
        int64_t count = we.len / static_cast<int64_t>(sizeof(double));
        std::vector<double> arr(static_cast<size_t>(count));
        std::memcpy(arr.data(), blob.data() + we.off, static_cast<size_t>(we.len));
        p.layout.push_back(std::move(we));
        p.weights.push_back(std::move(arr));
    }
    return p;
}

void RedisClient::write_results(const std::string& res_token, int idx,
                                const std::vector<float>& X, int n, int feat_dim,
                                const std::vector<float>& PI, const std::vector<float>& M,
                                const std::vector<float>& Y, int n_slots) {
    // Emit EXACTLY what np.frombuffer(...).reshape(...) decodes (no second encoder; ADR-0012 P7):
    //   X (n, feat_dim), PI (n, n_slots), M (n, n_slots), Y (n,) — contiguous little-endian float32,
    //   row-major. The float vectors are already laid out row-major, so tobytes == raw memcpy.
    ResultKeys k = result_keys(res_token, idx);
    int ttl = result_ttl();
    auto set_block = [&](const std::string& key, const float* data, size_t count) {
        const char* bytes = reinterpret_cast<const char*>(data);
        size_t nbytes = count * sizeof(float);
        // SET key <bytes> EX <ttl> — the TTL in the SAME round-trip (aborted-iteration self-clean).
        redisReply* r = static_cast<redisReply*>(redisCommand(
            ctx_, "SET %b %b EX %d", key.data(), key.size(), bytes, nbytes, ttl));
        if (r == nullptr) throw std::runtime_error("chocofarm: redis SET failed (no reply) for " + key);
        bool ok = (r->type == REDIS_REPLY_STATUS || r->type == REDIS_REPLY_STRING);
        freeReplyObject(r);
        if (!ok) throw std::runtime_error("chocofarm: redis SET failed for " + key);
    };
    // sanity: the blocks must be the row-major sizes the reshape expects (fail loud, ADR-0002).
    if (X.size() != static_cast<size_t>(n) * feat_dim ||
        PI.size() != static_cast<size_t>(n) * n_slots ||
        M.size() != static_cast<size_t>(n) * n_slots ||
        Y.size() != static_cast<size_t>(n)) {
        throw std::runtime_error("chocofarm: result block size mismatch (would corrupt reshape)");
    }
    set_block(k.X, X.data(), X.size());
    set_block(k.PI, PI.data(), PI.size());
    set_block(k.M, M.data(), M.size());
    set_block(k.Y, Y.data(), Y.size());
}

}  // namespace chocofarm
