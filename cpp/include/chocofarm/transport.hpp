// cpp/include/chocofarm/transport.hpp
// Purpose: the C++ redis wire client (hiredis) — the wire as the ONLY contract (ADR-0012 P7). It
//   mirrors chocofarm/az/transport.py byte-for-byte: the SAME `az:w:<run>:<phase>:<version>:m|:b`
//   weight keys and `az:res:<token>:<idx>:X|PI|M|Y` result keys, the SAME manifest-driven weight
//   read (float64 blob, no hardcoded offsets — P1), and the SAME four float32 result blocks (no
//   second encoder — P7). Connection facts come from the SAME `CHOCO_TRANSPORT_REDIS_*` env contract
//   as chocofarm/config.transport_redis_params (default 127.0.0.1:6380 db0), so the C++ client lands
//   wherever the operator points the Python transport. No hardcoded port.
//
//   ADR-0012 P9 (rule 5): every boundary failure here — an unreachable redis, a missing weight
//   payload, a malformed manifest — is a [[nodiscard]] std::expected<T, Error> returned by value,
//   never a thrown exception. The throwing connect-ctor becomes the RedisClient::create() factory
//   over a private noexcept ctor (a throwing ctor cannot return a value). Inputs and outputs are
//   typed, bounds-carrying views (std::span<const float>) — no raw-pointer/length pairs (rule 1).
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "chocofarm/error.hpp"

struct redisContext;  // hiredis (opaque)

namespace chocofarm {

// ---- the weight manifest (mirrors WeightContainer.pack's manifest JSON) ----
// One bound weight: its name, shape, dtype string ('<f8' on the wire), and its (offset, byte-len)
// into the float64 blob. The C++ side reads (offset, len, shape, dtype) FROM the manifest — never a
// hardcoded offset (ADR-0012 P1: the cross-language form of the three-writer feature-layout cancer).
struct WeightEntry {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype;   // numpy dtype str, e.g. "<f8"
    int64_t off = 0;     // byte offset into the blob
    int64_t len = 0;     // byte length
};

// The reconstructed net payload: the scalar construction meta + the per-weight bound arrays (values
// copied out of the float64 blob at each entry's (off, len)). RandomPolicy ignores `weights`, but
// the runner reads this path anyway to EXERCISE the weight-read seam (P7).
struct WeightPayload {
    int in_dim = 0;
    int H = 0;
    int n_actions = 0;
    double y_mean = 0.0;
    double y_std = 1.0;
    bool residual = false;
    std::vector<WeightEntry> layout;
    std::vector<std::vector<double>> weights;  // weights[k] = the bound float64 array for layout[k]
};

// ---- key namespace (mirrors transport.weight_keys / result_keys exactly) ----
[[nodiscard]] std::pair<std::string, std::string> weight_keys(std::string_view run,
                                                              std::string_view phase,
                                                              int version);  // (manifest_key, blob_key)
struct ResultKeys { std::string X, PI, M, Y; };
[[nodiscard]] ResultKeys result_keys(std::string_view res_token, int idx);

// ---- connection (the CHOCO_TRANSPORT_REDIS_* env contract; mirrors config.transport_redis_params) ----
class RedisClient {
  public:
    // Factory (ADR-0012 P9 rule 5): reads CHOCO_TRANSPORT_REDIS_HOST/PORT/DB (defaults 127.0.0.1/
    // 6380/0), connects + PINGs, and returns a connected client OR an Error — never throws on a dead
    // redis (mirrors connect()'s ping as a typed boundary failure). A throwing ctor cannot return a
    // value, so construction is a static factory over a private noexcept ctor.
    [[nodiscard]] static std::expected<RedisClient, Error> create();

    ~RedisClient();
    RedisClient(const RedisClient&) = delete;
    RedisClient& operator=(const RedisClient&) = delete;
    RedisClient(RedisClient&& o) noexcept : ctx_(std::exchange(o.ctx_, nullptr)) {}
    RedisClient& operator=(RedisClient&& o) noexcept;

    // Worker-side weight READ: fetch (manifest, blob) for (run, phase, version), parse the manifest,
    // and bind each weight by its (off, len, shape, dtype). A missing payload or malformed manifest
    // is a typed Error (mirrors read_weights' RuntimeError; ADR-0002 / P5 — never a silent stale
    // serve, and ADR-0012 P9 — a returned value, not a throw). Parses ONLY by the manifest; no
    // hardcoded offsets (P1).
    [[nodiscard]] std::expected<WeightPayload, Error> read_weights(std::string_view run,
                                                                  std::string_view phase,
                                                                  int version);

    // Worker-side result WRITE: SET the four contiguous float32 blocks under the per-task result
    // keys, each with `ttl` seconds expiry in the SAME SET round-trip (CHOCO_RESULT_TTL, default
    // 3600). No second encoder: each block is the raw little-endian float32 bytes
    // np.frombuffer(...).reshape(...) decodes (X (n,feat_dim), PI/M (n,n_slots), Y (n,)). The blocks
    // are typed bounds-carrying views (std::span<const float>); a redis SET failure or a block-size
    // mismatch is a typed Error returned by value (P9 rule 5).
    [[nodiscard]] std::expected<void, Error> write_results(std::string_view res_token, int idx,
                                                          std::span<const float> X, int n, int feat_dim,
                                                          std::span<const float> PI,
                                                          std::span<const float> M,
                                                          std::span<const float> Y, int n_slots);

    // The four float32 result blocks read back (the symmetric counterpart to write_results) — the
    // raw little-endian float32 bytes of X/PI/M/Y under az:res:<token>:<idx>:{X,PI,M,Y}, decoded into
    // four float32 vectors. Used by the batched-runtime parity check to read back what the driver wrote
    // and byte-compare it to the serial reference. A missing key or a redis error is a typed Error (P9
    // rule 5). (Not on the runner's hot path — the parent reconciliation reads results in Python.)
    struct ResultBlocks {
        std::vector<float> X, PI, M, Y;
    };
    [[nodiscard]] std::expected<ResultBlocks, Error> read_results(std::string_view res_token, int idx);

  private:
    explicit RedisClient(redisContext* ctx) noexcept : ctx_(ctx) {}
    redisContext* ctx_ = nullptr;
};

// The result TTL contract (CHOCO_RESULT_TTL, default 3600s) — the same env override transport reads.
[[nodiscard]] int result_ttl();

}  // namespace chocofarm
