// cpp/include/chocofarm/transport.hpp
// Purpose: the C++ redis wire client (hiredis) — the wire as the ONLY contract (ADR-0012 P7). It
//   mirrors chocofarm/az/transport.py byte-for-byte: the SAME `az:w:<run>:<phase>:<version>:m|:b`
//   weight keys and `az:res:<token>:<idx>:X|PI|M|Y` result keys, the SAME manifest-driven weight
//   read (float64 blob, no hardcoded offsets — P1), and the SAME four float32 result blocks (no
//   second encoder — P7). Connection facts come from the SAME `CHOCO_TRANSPORT_REDIS_*` env contract
//   as chocofarm/config.transport_redis_params (default 127.0.0.1:6380 db0), so the C++ client lands
//   wherever the operator points the Python transport. No hardcoded port.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

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
std::pair<std::string, std::string> weight_keys(const std::string& run, const std::string& phase,
                                                 int version);  // (manifest_key, blob_key)
struct ResultKeys { std::string X, PI, M, Y; };
ResultKeys result_keys(const std::string& res_token, int idx);

// ---- connection (the CHOCO_TRANSPORT_REDIS_* env contract; mirrors config.transport_redis_params) ----
class RedisClient {
  public:
    RedisClient();   // reads CHOCO_TRANSPORT_REDIS_HOST/PORT/DB (defaults 127.0.0.1/6380/0), connects + PINGs
    ~RedisClient();
    RedisClient(const RedisClient&) = delete;
    RedisClient& operator=(const RedisClient&) = delete;

    // Worker-side weight READ: fetch (manifest, blob) for (run, phase, version), parse the manifest,
    // and bind each weight by its (off, len, shape, dtype). A missing payload is a LOUD
    // std::runtime_error (mirrors read_weights' RuntimeError; ADR-0002 / P5 — never a silent stale
    // serve). Parses ONLY by the manifest; no hardcoded offsets (P1).
    WeightPayload read_weights(const std::string& run, const std::string& phase, int version);

    // Worker-side result WRITE: SET the four contiguous float32 blocks under the per-task result
    // keys, each with `ttl` seconds expiry in the SAME SET round-trip (CHOCO_RESULT_TTL, default
    // 3600). No second encoder: each block is the raw little-endian float32 bytes
    // np.frombuffer(...).reshape(...) decodes (X (n,feat_dim), PI/M (n,n_slots), Y (n,)).
    void write_results(const std::string& res_token, int idx,
                       const std::vector<float>& X, int n, int feat_dim,
                       const std::vector<float>& PI, const std::vector<float>& M,
                       const std::vector<float>& Y, int n_slots);

  private:
    redisContext* ctx_ = nullptr;
};

// The result TTL contract (CHOCO_RESULT_TTL, default 3600s) — the same env override transport reads.
int result_ttl();

}  // namespace chocofarm
