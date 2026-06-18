// cpp/src/net.cpp
// Purpose: the C++ NetForward implementation — the value+policy MLP leaf evaluator reimplementing the
//   ONE Python forward chocofarm/az/forward.forward_core(params, X, xp) EXACTLY (ADR-0012 P7 / R11).
//   See net.hpp. Every dimension + the residual on/off is DERIVED from the manifest-bound weights
//   (their shapes + the manifest's `residual` toggle — ADR-0012 P1): no hardcoded layout. The graph
//   is forward_core's verbatim, run in float32 (the parametric hot-path precision the Python
//   `_predict_both_f32` runs at), plus the value de-standardization the Python callers apply.
//
//   ADR-0012 P9: the matmul/relu/require helpers are PURE VALUE-FUNCTIONS — they take typed,
//   bounds-carrying std::span<const float> inputs and RETURN their result by value (no raw pointers,
//   no `void f(..., Out& out)` out-parameters). The manifest validators return a small typed result
//   OR a typed Error (rule 5). The arithmetic core (matvec_bias, relu, forward_one, predict) is total —
//   it neither throws nor returns expected; the only error surface is the manifest validation at
//   create(). predict() calls the one single-row core forward_one (P1); no Eigen/BLAS/-ffast-math.
//
// Public Domain (The Unlicense).
#include "chocofarm/net.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

namespace chocofarm {

namespace {

// Look up a named weight in the payload (the manifest's canonical order; we read BY NAME, not by
// position — P1: the manifest owns the layout, we never re-enumerate).
struct Bound { const WeightEntry* entry; const std::vector<double>* data; };

std::unordered_map<std::string, Bound> index_by_name(const WeightPayload& p) {
    std::unordered_map<std::string, Bound> m;
    for (size_t k = 0; k < p.layout.size(); ++k) {
        m[p.layout[k].name] = Bound{&p.layout[k], &p.weights[k]};
    }
    return m;
}

// A required 2-D matrix, resolved + cast to float32 (rows/cols read back when the caller pins -1).
struct RequiredMatrix { int rows = 0; int cols = 0; std::vector<float> v; };
// A required 1-D vector, resolved + cast to float32.
struct RequiredVector { int len = 0; std::vector<float> v; };

// Cast a float64 manifest array to float32 (the f64 blob -> f32 inference cache; behavioral bar).
std::vector<float> to_f32(const std::vector<double>& src) {
    std::vector<float> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) out[i] = static_cast<float>(src[i]);
    return out;
}

// Require a named matrix of shape (want_rows, want_cols) — pass -1 to read a dim back. Returns the
// resolved (rows, cols) + a float32 copy BY VALUE, OR a typed Error (P9 rule 5). A missing param or
// a rank/shape mismatch is a recoverable boundary failure (ADR-0002 / P5), not a throw.
std::expected<RequiredMatrix, Error> require_matrix(
    const std::unordered_map<std::string, Bound>& by_name, const std::string& name,
    int want_rows, int want_cols) {
    auto it = by_name.find(name);
    if (it == by_name.end())
        return std::unexpected(make_error("chocofarm NetForward: manifest missing required param '" + name + "'"));
    const WeightEntry& e = *it->second.entry;
    if (e.shape.size() != 2)
        return std::unexpected(make_error("chocofarm NetForward: param '" + name + "' is not a 2-D matrix"));
    int r = static_cast<int>(e.shape[0]);
    int c = static_cast<int>(e.shape[1]);
    if (want_rows >= 0 && want_rows != r)
        return std::unexpected(make_error("chocofarm NetForward: param '" + name + "' rows " +
                                          std::to_string(r) + " != expected " + std::to_string(want_rows)));
    if (want_cols >= 0 && want_cols != c)
        return std::unexpected(make_error("chocofarm NetForward: param '" + name + "' cols " +
                                          std::to_string(c) + " != expected " + std::to_string(want_cols)));
    const std::vector<double>& src = *it->second.data;
    if (static_cast<int64_t>(src.size()) != static_cast<int64_t>(r) * c)
        return std::unexpected(make_error("chocofarm NetForward: param '" + name + "' element count mismatch"));
    return RequiredMatrix{r, c, to_f32(src)};
}

// Require a named bias vector of length `want_len` (pass -1 to read it back). Returns the resolved
// length + a float32 copy BY VALUE, OR a typed Error (P9 rule 5). Loud on missing / shape mismatch.
std::expected<RequiredVector, Error> require_vector(
    const std::unordered_map<std::string, Bound>& by_name, const std::string& name, int want_len) {
    auto it = by_name.find(name);
    if (it == by_name.end())
        return std::unexpected(make_error("chocofarm NetForward: manifest missing required param '" + name + "'"));
    const WeightEntry& e = *it->second.entry;
    // biases are 1-D (b1/b2/br1/br2/bp shape (n,)); bv is shape (1,).
    if (e.shape.size() != 1)
        return std::unexpected(make_error("chocofarm NetForward: param '" + name + "' is not a 1-D vector"));
    int n = static_cast<int>(e.shape[0]);
    if (want_len >= 0 && want_len != n)
        return std::unexpected(make_error("chocofarm NetForward: param '" + name + "' len " +
                                          std::to_string(n) + " != expected " + std::to_string(want_len)));
    const std::vector<double>& src = *it->second.data;
    if (static_cast<int64_t>(src.size()) != n)
        return std::unexpected(make_error("chocofarm NetForward: param '" + name + "' element count mismatch"));
    return RequiredVector{n, to_f32(src)};
}

// out[c] = bias[c] + sum_r in[r] * W[r*cols + c]   — the (1×rows)·(rows×cols) matmul + bias of
// forward_core's `X @ W + b`, accumulated in float32 (the Python f32 hot-path precision). A pure
// value-function (P9 rule 2): RETURNS the result vector by value (free under NRVO), no out-param.
// `bias` may be an EMPTY span (then out[c] = sum_r in[r]*W[r,c]). `in`/`W`/`bias` are typed,
// bounds-carrying views (rule 1).
std::vector<float> matvec_bias(std::span<const float> in, std::span<const float> W, int rows, int cols,
                               std::span<const float> bias) {
    std::vector<float> out(static_cast<size_t>(cols), 0.0f);
    const bool has_bias = !bias.empty();
    for (int c = 0; c < cols; ++c) {
        float acc = has_bias ? bias[static_cast<size_t>(c)] : 0.0f;
        for (int r = 0; r < rows; ++r)
            acc += in[static_cast<size_t>(r)] * W[static_cast<size_t>(r) * cols + c];
        out[static_cast<size_t>(c)] = acc;
    }
    return out;
}

// ReLU applied to a vector, consumed and returned by value (P9 rule 2: xp.maximum(z, 0.0) as a
// value-function, not an in-place void mutation — NRVO makes the by-value return free).
std::vector<float> relu(std::vector<float> v) {
    for (float& x : v) if (x < 0.0f) x = 0.0f;
    return v;
}

}  // namespace

std::expected<NetForward, Error> NetForward::create(const WeightPayload& payload) {
    auto by_name = index_by_name(payload);
    NetForward nf;  // private noexcept ctor; the factory fills the fields and returns by value

    // ---- residual / policy toggles: DERIVED from the manifest, exactly forward_core's key test ----
    // forward_core applies the block iff `"Wr1" in params` and the policy head iff `"Wp" in params`.
    // The manifest carries `residual` (and the Wr* entries iff so). We honor BOTH and require them to
    // agree: a residual=true manifest must carry Wr1/.. , and the presence of Wr1 is the real toggle.
    bool has_wr1 = by_name.count("Wr1") > 0;
    nf.residual_ = has_wr1;  // the operative toggle is param presence (forward_core's `"Wr1" in params`)
    if (payload.residual != has_wr1) {
        // Port/ACL: translate-and-validate, do not coerce (ADR-0012 P2 / ADR-0002 / P9 rule 5).
        return std::unexpected(make_error("chocofarm NetForward: manifest residual flag (" +
                                          std::string(payload.residual ? "true" : "false") +
                                          ") disagrees with the presence of Wr1 in the layout"));
    }
    bool has_wp = by_name.count("Wp") > 0;

    // ---- trunk: W1 (in_dim, H), W2 (H, H), and the value head Wv (H, 1) — derive in_dim/H ----
    auto w1 = require_matrix(by_name, "W1", -1, -1);   // resolves in_dim_ and hidden_
    if (!w1) return std::unexpected(w1.error());
    nf.in_dim_ = w1->rows; nf.hidden_ = w1->cols;
    nf.W1_ = NetForward::Mat{w1->rows, w1->cols, std::move(w1->v)};

    auto b1 = require_vector(by_name, "b1", nf.hidden_);
    if (!b1) return std::unexpected(b1.error());
    nf.b1_ = NetForward::Vec{std::move(b1->v)};

    auto w2 = require_matrix(by_name, "W2", nf.hidden_, nf.hidden_);
    if (!w2) return std::unexpected(w2.error());
    nf.W2_ = NetForward::Mat{w2->rows, w2->cols, std::move(w2->v)};
    auto b2 = require_vector(by_name, "b2", nf.hidden_);
    if (!b2) return std::unexpected(b2.error());
    nf.b2_ = NetForward::Vec{std::move(b2->v)};

    auto wv = require_matrix(by_name, "Wv", nf.hidden_, 1);
    if (!wv) return std::unexpected(wv.error());
    nf.Wv_ = NetForward::Mat{wv->rows, wv->cols, std::move(wv->v)};
    auto bv = require_vector(by_name, "bv", 1);
    if (!bv) return std::unexpected(bv.error());
    nf.bv_ = NetForward::Vec{std::move(bv->v)};

    // ---- residual block (iff present): Wr1 (H, H), Wr2 (H, H) — same H as the trunk ----
    if (nf.residual_) {
        auto wr1 = require_matrix(by_name, "Wr1", nf.hidden_, nf.hidden_);
        if (!wr1) return std::unexpected(wr1.error());
        nf.Wr1_ = NetForward::Mat{wr1->rows, wr1->cols, std::move(wr1->v)};
        auto br1 = require_vector(by_name, "br1", nf.hidden_);
        if (!br1) return std::unexpected(br1.error());
        nf.br1_ = NetForward::Vec{std::move(br1->v)};
        auto wr2 = require_matrix(by_name, "Wr2", nf.hidden_, nf.hidden_);
        if (!wr2) return std::unexpected(wr2.error());
        nf.Wr2_ = NetForward::Mat{wr2->rows, wr2->cols, std::move(wr2->v)};
        auto br2 = require_vector(by_name, "br2", nf.hidden_);
        if (!br2) return std::unexpected(br2.error());
        nf.br2_ = NetForward::Vec{std::move(br2->v)};
    }

    // ---- policy head (iff present): Wp (H, n_actions) — derive n_actions from its cols ----
    if (has_wp) {
        auto wp = require_matrix(by_name, "Wp", nf.hidden_, -1);   // resolves n_actions_ = cols
        if (!wp) return std::unexpected(wp.error());
        nf.Wp_ = NetForward::Mat{wp->rows, wp->cols, std::move(wp->v)};
        nf.n_actions_ = wp->cols;
        auto bp = require_vector(by_name, "bp", nf.n_actions_);
        if (!bp) return std::unexpected(bp.error());
        nf.bp_ = NetForward::Vec{std::move(bp->v)};
    } else {
        nf.n_actions_ = 0;
    }

    // ---- value de-standardization scalars (the manifest's y_mean/y_std; predict_value applies them) ----
    nf.y_mean_ = static_cast<float>(payload.y_mean);
    // mirror ValueMLP's y_std guard (y_std <= 1e-8 -> 1.0) so a degenerate scale never blows up.
    nf.y_std_ = static_cast<float>(payload.y_std > 1e-8 ? payload.y_std : 1.0);
    return nf;
}

NetPrediction NetForward::forward_one(std::span<const float> x) const {
    // x.size() == in_dim() is the caller's invariant (the manifest-validating create() reconciled the
    // dims). A mismatch is a programmer bug, not a boundary failure — an assert, not an Error (P9).
    assert(static_cast<int>(x.size()) == in_dim_ && "NetForward::forward_one: x length != in_dim");

    // forward_core, verbatim, B=1, float32:
    //   z1 = X@W1 + b1; a1 = ReLU(z1)
    std::vector<float> a1 = relu(matvec_bias(x, W1_.v, in_dim_, hidden_, b1_.v));
    //   z2 = a1@W2 + b2; a2 = ReLU(z2)
    std::vector<float> a2 = relu(matvec_bias(a1, W2_.v, hidden_, hidden_, b2_.v));

    //   if residual: head_in = a2 + (ReLU(a2@Wr1+br1)@Wr2+br2)   -- PRE-activation skip, NO outer ReLU
    //   else:        head_in = a2
    std::vector<float> head_in = a2;  // copy; we add the residual delta in place when present
    if (residual_) {
        std::vector<float> zr1 = relu(matvec_bias(a2, Wr1_.v, hidden_, hidden_, br1_.v));  // ar1 = ReLU(a2@Wr1 + br1)
        std::vector<float> zr2 = matvec_bias(zr1, Wr2_.v, hidden_, hidden_, br2_.v);       // a2@Wr1.. @Wr2 + br2
        for (int j = 0; j < hidden_; ++j)
            head_in[static_cast<size_t>(j)] = a2[static_cast<size_t>(j)] + zr2[static_cast<size_t>(j)];  // NO outer ReLU
    }

    //   v_std = (head_in @ Wv + bv).ravel()  -- the standardized scalar value
    std::vector<float> v_out = matvec_bias(head_in, Wv_.v, hidden_, 1, bv_.v);
    float v_std = v_out[0];

    NetPrediction out;
    // de-standardize back to the λ-penalized return scale (predict_value: v_std*y_std + y_mean).
    out.value = v_std * y_std_ + y_mean_;

    //   logits = head_in @ Wp + bp  (iff the net carries a policy head; empty otherwise, == None)
    if (has_policy()) {
        out.logits = matvec_bias(head_in, Wp_.v, hidden_, n_actions_, bp_.v);
    }
    return out;
}

std::expected<NetPrediction, Error> NetForward::predict(std::span<const float> x) const {
    // This LOCAL forward is TOTAL: it always returns the VALUE arm. The std::expected return is the
    // NetEvaluator port shape (shared with the fallible remote ZmqNetClient), not a real error surface.
    // The compute is the shared single-row core (one home — P1).
    return forward_one(x);
}

}  // namespace chocofarm
