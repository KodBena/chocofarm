// cpp/include/chocofarm/net.hpp
// Purpose: the C++ NetForward — the value+policy MLP leaf evaluator, reimplementing the ONE Python
//   forward chocofarm/az/forward.forward_core(params, X, xp) EXACTLY (ADR-0012 P7 / R11). It is the
//   composable component the future C++ Gumbel-AZ search will call as `predict(X) -> {value, logits}`
//   at each leaf; it is NOT wired into the RandomPolicy runner.
//
//   It mirrors forward_core's graph verbatim:
//       z1 = X@W1 + b1;  a1 = ReLU(z1)
//       z2 = a1@W2 + b2; a2 = ReLU(z2)
//       if residual: head_in = a2 + (ReLU(a2@Wr1+br1)@Wr2+br2)   // PRE-activation skip, NO outer ReLU
//       else:        head_in = a2
//       v_std  = head_in@Wv + bv          (the STANDARDIZED scalar value)
//       logits = head_in@Wp + bp          (iff the net carries a policy head)
//   plus the value de-standardization the Python callers apply (predict_value): v = v_std*y_std + y_mean.
//
//   EVERY dimension and the residual on/off are DERIVED from the manifest-bound WeightPayload (the
//   weight matrices' shapes + the `residual` toggle the manifest carries — ADR-0012 P1): no hardcoded
//   layer count, in_dim, hidden, or n_actions. The residual block is applied IFF the payload carries
//   `Wr1` (exactly forward_core's `"Wr1" in params` toggle), the policy head IFF the payload carries
//   `Wp`. Parity is the ADR-0012 P6 behavioral bar (float32-equivalence < 1e-4 vs forward_core), not
//   byte-identity.
//
//   ADR-0012 P9: the forward core is a pure value-function — `predict(std::span<const float>)`
//   takes a typed, bounds-carrying input and RETURNS its result by value (NetPrediction), no raw
//   pointers, no out-parameters. The throwing manifest-validating constructor becomes the static
//   factory `NetForward::create(const WeightPayload&) -> std::expected<NetForward, Error>` over a
//   private noexcept ctor (a throwing ctor cannot return a value; a malformed manifest is a
//   recoverable boundary failure, not a throw — rule 5).
//
// Public Domain (The Unlicense).
#pragma once

#include <expected>
#include <span>
#include <vector>

#include "chocofarm/error.hpp"
#include "chocofarm/transport.hpp"

namespace chocofarm {

// One forward result: the de-standardized leaf value + the policy logits over the action slots (empty
// when the net is value-only — `Wp` absent in the manifest, mirroring forward_core's `logits=None`).
struct NetPrediction {
    float value = 0.0f;             // de-standardized: v_std*y_std + y_mean (the λ-penalized return scale)
    std::vector<float> logits;      // raw policy logits over n_actions slots (NOT softmaxed; empty if none)
};

// The value+policy MLP forward, reconstructed FROM the manifest-bound weights. Holds float32 copies of
// each weight (the parametric hot-path precision the Python `_predict_both_f32` runs at) keyed by name,
// derives all dims/toggles from their shapes, and computes one forward per `predict`.
class NetForward {
  public:
    // Build from the manifest-bound payload (transport.read_weights). Validates that the required
    // params (W1/b1/W2/b2/Wv/bv) are present and shape-consistent, and derives in_dim, hidden,
    // n_actions, the residual toggle, and the policy-head presence from the layout/meta — never
    // hardcoded (ADR-0012 P1). A missing required param or a shape mismatch is a typed Error
    // (ADR-0002 / P5 + ADR-0012 P9 rule 5: translate-and-validate, do not coerce; a returned value,
    // not a throw). Construction is a factory over a private noexcept ctor.
    [[nodiscard]] static std::expected<NetForward, Error> create(const WeightPayload& payload);

    // Run forward_core's graph once on a length-`in_dim()` float32 feature vector and return the
    // de-standardized value + the policy logits (the search's leaf-evaluator entry point). The input
    // is a typed bounds-carrying view (std::span<const float>) — a std::vector<float> binds
    // implicitly — and the result is returned BY VALUE (P9 rules 1 & 2). The caller guarantees
    // x.size() == in_dim() (an invariant the manifest-validating ctor already reconciled); a mismatch
    // is a programmer bug (an assert), not a boundary Error.
    [[nodiscard]] NetPrediction predict(std::span<const float> x) const;

    [[nodiscard]] int in_dim() const { return in_dim_; }
    [[nodiscard]] int hidden() const { return hidden_; }
    [[nodiscard]] int n_actions() const { return n_actions_; }     // 0 when value-only (no policy head)
    [[nodiscard]] bool residual() const { return residual_; }
    [[nodiscard]] bool has_policy() const { return n_actions_ > 0; }

  private:
    NetForward() noexcept = default;  // the factory fills the fields; construction never throws

    // Row-major weight matrices/biases in float32 (the manifest blob is float64; cast once at build,
    // matching the Python f32 inference cache — float32-equivalence is the bar, not byte-identity).
    struct Mat { int rows = 0, cols = 0; std::vector<float> v; };  // row-major, v[r*cols + c]
    struct Vec { std::vector<float> v; };

    Mat W1_, W2_, Wv_;
    Vec b1_, b2_, bv_;
    Mat Wr1_, Wr2_;            // residual block (only populated iff residual_)
    Vec br1_, br2_;
    Mat Wp_;                   // policy head (only populated iff has_policy())
    Vec bp_;

    float y_mean_ = 0.0f;
    float y_std_ = 1.0f;

    int in_dim_ = 0;
    int hidden_ = 0;
    int n_actions_ = 0;
    bool residual_ = false;
};

}  // namespace chocofarm
