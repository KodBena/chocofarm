// cpp/src/transport.cpp
// Purpose: the C++ redis wire client (hiredis) — the wire as the ONLY contract (ADR-0012 P7). See
//   transport.hpp. It mirrors chocofarm/az/transport.py + config.py byte-for-byte: the key
//   namespace, the manifest-driven float64 weight read (no hardcoded offsets — P1), the four
//   float32 result blocks (no second encoder — P7), and the CHOCO_TRANSPORT_REDIS_* connection
//   contract (default 127.0.0.1:6380 db0 — no hardcoded port; mirrors config.transport_redis_params).
//
//   ADR-0012 P9 (rule 5): every boundary failure is returned as std::expected<T, Error>, never
//   thrown. The connect path is the RedisClient::create() factory; the read/write paths return
//   std::expected. The byte/wire contract itself is unchanged (same keys, same blobs, same TTL).
//
// Public Domain (The Unlicense).
#include "chocofarm/transport.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

#include <hiredis/hiredis.h>
#include <nlohmann/json.hpp>

namespace chocofarm {

using nlohmann::json;

namespace {

// ---- env contract (mirrors config.transport_redis_params / config._result_ttl) ----
std::string env_str(const char* name, std::string_view dflt) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string(dflt);
}
int env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : dflt;
}

// RAII wrapper for a hiredis reply (manual freeReplyObject -> scoped cleanup; ADR-0012 modern
// idiom: no hand-balanced free on every error path). Null-reply (a dead connection) is a valid
// state the caller checks before using `r`.
class Reply {
  public:
    explicit Reply(void* r) noexcept : r_(static_cast<redisReply*>(r)) {}
    ~Reply() { if (r_) freeReplyObject(r_); }
    Reply(const Reply&) = delete;
    Reply& operator=(const Reply&) = delete;
    [[nodiscard]] redisReply* get() const noexcept { return r_; }
    [[nodiscard]] explicit operator bool() const noexcept { return r_ != nullptr; }
    redisReply* operator->() const noexcept { return r_; }

  private:
    redisReply* r_ = nullptr;
};

}  // namespace

int result_ttl() { return env_int("CHOCO_RESULT_TTL", 3600); }  // mirrors transport._result_ttl

// ---- key namespace (byte-identical to transport.weight_keys / result_keys) ----
std::pair<std::string, std::string> weight_keys(std::string_view run, std::string_view phase,
                                                int version) {
    std::string base = "az:w:" + std::string(run) + ":" + std::string(phase) + ":" +
                       std::to_string(version);
    return {base + ":m", base + ":b"};
}
ResultKeys result_keys(std::string_view res_token, int idx) {
    std::string base = "az:res:" + std::string(res_token) + ":" + std::to_string(idx);
    return ResultKeys{base + ":X", base + ":PI", base + ":M", base + ":Y"};
}

// ---- connection ----
std::expected<RedisClient, Error> RedisClient::create() {
    // CHOCO_TRANSPORT_REDIS_HOST / _PORT / _DB — the SAME env contract as config.transport_redis_params
    // (defaults 127.0.0.1 / 6380 / 0). No hardcoded port (ADR-0012 P7 / P1: config.py is the one
    // owner of "which redis"; this just reads the same transport-role env vars so it lands on the
    // same ephemeral instance). The C++ runner is a transport component, so it uses the transport role.
    std::string host = env_str("CHOCO_TRANSPORT_REDIS_HOST", "127.0.0.1");
    int port = env_int("CHOCO_TRANSPORT_REDIS_PORT", 6380);
    int db = env_int("CHOCO_TRANSPORT_REDIS_DB", 0);

    redisContext* ctx = redisConnect(host.c_str(), port);
    if (ctx == nullptr || ctx->err) {
        std::string msg = ctx ? ctx->errstr : "allocation failure";
        if (ctx) { redisFree(ctx); }
        // ADR-0002 / P5 + P9 rule 5: fail loud NOW if redis is unreachable, as a typed Error returned
        // to the shell (mirrors connect()'s ping) — never mid-run, never a throw.
        return std::unexpected(make_error("chocofarm: redis connect failed (" + host + ":" +
                                          std::to_string(port) + "): " + msg));
    }
    if (db != 0) {
        Reply r(redisCommand(ctx, "SELECT %d", db));
        if (!r) { redisFree(ctx); return std::unexpected(make_error("chocofarm: redis SELECT failed (no reply)")); }
        if (r->type == REDIS_REPLY_ERROR) {
            redisFree(ctx);
            return std::unexpected(make_error("chocofarm: redis SELECT db failed"));
        }
    }
    // fail loud now if unreachable (ADR-0002), mirroring r.ping() in transport.connect()
    Reply pong(redisCommand(ctx, "PING"));
    if (!pong) { redisFree(ctx); return std::unexpected(make_error("chocofarm: redis PING failed (no reply)")); }
    return RedisClient(ctx);  // private noexcept ctor takes ownership of the connected context
}

RedisClient& RedisClient::operator=(RedisClient&& o) noexcept {
    if (this != &o) {
        if (ctx_) redisFree(ctx_);
        ctx_ = std::exchange(o.ctx_, nullptr);
    }
    return *this;
}

RedisClient::~RedisClient() {
    if (ctx_) redisFree(ctx_);
}

namespace {

// Fetch a raw-bytes value. Returns `true` (and fills `out`) if present, `false` if the key is
// missing (a nil reply). A no-reply / unexpected-type is a typed Error (P9 rule 5).
std::expected<bool, Error> redis_get_bytes(redisContext* ctx, const std::string& key,
                                           std::string& out) {
    Reply r(redisCommand(ctx, "GET %b", key.data(), key.size()));
    if (!r) return std::unexpected(make_error("chocofarm: redis GET failed (no reply) for " + key));
    if (r->type == REDIS_REPLY_STRING) {
        out.assign(r->str, r->len);   // raw bytes (may contain NULs — use len, not strlen)
        return true;
    }
    if (r->type == REDIS_REPLY_NIL) return false;
    return std::unexpected(make_error("chocofarm: redis GET unexpected reply for " + key));
}

// Parse the manifest JSON + bind each weight out of the float64 blob (NO hardcoded offsets — ADR-0012
// P1). A malformed manifest is a typed boundary Error (P9 rule 5); the nlohmann accessor exceptions
// (a missing key / wrong type) are caught at THIS edge and translated, so read_weights is total.
std::expected<WeightPayload, Error> parse_manifest(const std::string& manifest,
                                                   const std::string& blob) {
    json m = json::parse(manifest, nullptr, /*allow_exceptions=*/false);
    if (m.is_discarded()) return std::unexpected(make_error("chocofarm: weight manifest is not valid JSON"));
    try {
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
                return std::unexpected(make_error("chocofarm: unexpected weight dtype '" + we.dtype +
                                                  "' (expected '<f8'); the weight blob is float64"));
            }
            if (we.off < 0 || we.len < 0 || we.off + we.len > static_cast<int64_t>(blob.size())) {
                return std::unexpected(make_error("chocofarm: weight entry " + we.name +
                                                  " (off,len) exceeds blob bounds"));
            }
            int64_t count = we.len / static_cast<int64_t>(sizeof(double));
            std::vector<double> arr(static_cast<size_t>(count));
            std::memcpy(arr.data(), blob.data() + we.off, static_cast<size_t>(we.len));
            p.layout.push_back(std::move(we));
            p.weights.push_back(std::move(arr));
        }
        return p;
    } catch (const json::exception& e) {
        return std::unexpected(make_error(std::string("chocofarm: malformed weight manifest: ") + e.what()));
    }
}

}  // namespace

std::expected<WeightPayload, Error> RedisClient::read_weights(std::string_view run,
                                                              std::string_view phase, int version) {
    auto [mk, bk] = weight_keys(run, phase, version);
    std::string manifest, blob;
    auto have_m = redis_get_bytes(ctx_, mk, manifest);
    if (!have_m) return std::unexpected(have_m.error());
    auto have_b = redis_get_bytes(ctx_, bk, blob);
    if (!have_b) return std::unexpected(have_b.error());
    if (!*have_m || !*have_b) {
        // mirrors read_weights' RuntimeError (ADR-0002 / P5: never a silent stale-net serve).
        return std::unexpected(make_error("weight payload az:w:" + std::string(run) + ":" +
                                          std::string(phase) + ":" + std::to_string(version) +
                                          " missing from redis"));
    }
    return parse_manifest(manifest, blob);
}

std::expected<void, Error> RedisClient::write_results(std::string_view res_token, int idx,
                                                      std::span<const float> X, int n, int feat_dim,
                                                      std::span<const float> PI, std::span<const float> M,
                                                      std::span<const float> Y, int n_slots) {
    // Emit EXACTLY what np.frombuffer(...).reshape(...) decodes (no second encoder; ADR-0012 P7):
    //   X (n, feat_dim), PI (n, n_slots), M (n, n_slots), Y (n,) — contiguous little-endian float32,
    //   row-major. The float spans are already laid out row-major, so tobytes == raw memcpy.
    // sanity: the blocks must be the row-major sizes the reshape expects (fail loud, ADR-0002).
    if (X.size() != static_cast<size_t>(n) * feat_dim ||
        PI.size() != static_cast<size_t>(n) * n_slots ||
        M.size() != static_cast<size_t>(n) * n_slots ||
        Y.size() != static_cast<size_t>(n)) {
        return std::unexpected(make_error("chocofarm: result block size mismatch (would corrupt reshape)"));
    }
    ResultKeys k = result_keys(res_token, idx);
    int ttl = result_ttl();
    auto set_block = [&](const std::string& key, std::span<const float> data) -> std::expected<void, Error> {
        const char* bytes = reinterpret_cast<const char*>(data.data());
        size_t nbytes = data.size() * sizeof(float);
        // SET key <bytes> EX <ttl> — the TTL in the SAME round-trip (aborted-iteration self-clean).
        Reply r(redisCommand(ctx_, "SET %b %b EX %d", key.data(), key.size(), bytes, nbytes, ttl));
        if (!r) return std::unexpected(make_error("chocofarm: redis SET failed (no reply) for " + key));
        if (r->type != REDIS_REPLY_STATUS && r->type != REDIS_REPLY_STRING)
            return std::unexpected(make_error("chocofarm: redis SET failed for " + key));
        return {};
    };
    if (auto rx = set_block(k.X, X); !rx) return rx;
    if (auto rp = set_block(k.PI, PI); !rp) return rp;
    if (auto rm = set_block(k.M, M); !rm) return rm;
    if (auto ry = set_block(k.Y, Y); !ry) return ry;
    return {};
}

}  // namespace chocofarm
