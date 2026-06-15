// cpp/src/net.cpp
// Purpose: the C++ NetForward implementation — the value+policy MLP leaf evaluator reimplementing the
//   ONE Python forward chocofarm/az/forward.forward_core(params, X, xp) EXACTLY (ADR-0012 P7 / R11).
//   See net.hpp. Every dimension + the residual on/off is DERIVED from the manifest-bound weights
//   (their shapes + the manifest's `residual` toggle — ADR-0012 P1): no hardcoded layout. The graph
//   is forward_core's verbatim, run in float32 (the parametric hot-path precision the Python
//   `_predict_both_f32` runs at), plus the value de-standardization the Python callers apply.
//
// Public Domain (The Unlicense).
#include "chocofarm/net.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace chocofarm {

namespace {

// Look up a named weight in the payload (the manifest's canonical order; we read BY NAME, not by
// position — P1: the manifest owns the layout, we never re-enumerate). Returns nullptr if absent.
struct Bound { const WeightEntry* entry; const std::vector<double>* data; };

std::unordered_map<std::string, Bound> index_by_name(const WeightPayload& p) {
    std::unordered_map<std::string, Bound> m;
    for (size_t k = 0; k < p.layout.size(); ++k) {
        m[p.layout[k].name] = Bound{&p.layout[k], &p.weights[k]};
    }
    return m;
}

// Require a named matrix of shape (rows, cols) — derive `rows`/`cols` from -1 wildcards so the caller
// can either pin a dim or read it back. Returns the resolved (rows, cols) and fills a float32 copy.
// A missing param or a rank/shape mismatch is a loud std::runtime_error (ADR-0002 / P5).
void require_matrix(const std::unordered_map<std::string, Bound>& by_name, const std::string& name,
                    int& rows, int& cols, std::vector<float>& out) {
    auto it = by_name.find(name);
    if (it == by_name.end())
        throw std::runtime_error("chocofarm NetForward: manifest missing required param '" + name + "'");
    const WeightEntry& e = *it->second.entry;
    if (e.shape.size() != 2)
        throw std::runtime_error("chocofarm NetForward: param '" + name + "' is not a 2-D matrix");
    int r = static_cast<int>(e.shape[0]);
    int c = static_cast<int>(e.shape[1]);
    if (rows >= 0 && rows != r)
        throw std::runtime_error("chocofarm NetForward: param '" + name + "' rows " + std::to_string(r) +
                                 " != expected " + std::to_string(rows));
    if (cols >= 0 && cols != c)
        throw std::runtime_error("chocofarm NetForward: param '" + name + "' cols " + std::to_string(c) +
                                 " != expected " + std::to_string(cols));
    rows = r;
    cols = c;
    const std::vector<double>& src = *it->second.data;
    if (static_cast<int64_t>(src.size()) != static_cast<int64_t>(r) * c)
        throw std::runtime_error("chocofarm NetForward: param '" + name + "' element count mismatch");
    out.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) out[i] = static_cast<float>(src[i]);  // f64 blob -> f32
}

// Require a named bias vector of length `len` (derive from -1). Loud on missing / shape mismatch.
void require_vector(const std::unordered_map<std::string, Bound>& by_name, const std::string& name,
                    int& len, std::vector<float>& out) {
    auto it = by_name.find(name);
    if (it == by_name.end())
        throw std::runtime_error("chocofarm NetForward: manifest missing required param '" + name + "'");
    const WeightEntry& e = *it->second.entry;
    // biases are 1-D (b1/b2/br1/br2/bp shape (n,)); bv is shape (1,).
    if (e.shape.size() != 1)
        throw std::runtime_error("chocofarm NetForward: param '" + name + "' is not a 1-D vector");
    int n = static_cast<int>(e.shape[0]);
    if (len >= 0 && len != n)
        throw std::runtime_error("chocofarm NetForward: param '" + name + "' len " + std::to_string(n) +
                                 " != expected " + std::to_string(len));
    len = n;
    const std::vector<double>& src = *it->second.data;
    if (static_cast<int64_t>(src.size()) != n)
        throw std::runtime_error("chocofarm NetForward: param '" + name + "' element count mismatch");
    out.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) out[i] = static_cast<float>(src[i]);
}

// out[c] = bias[c] + sum_r in[r] * W[r*cols + c]   — the (1×rows)·(rows×cols) matmul + bias of
// forward_core's `X @ W + b`, accumulated in float32 (the Python f32 hot-path precision). `bias` may
// be null (then out[c] = sum_r in[r]*W[r,c]).
void matvec_bias(const float* in, const std::vector<float>& W, int rows, int cols,
                 const float* bias, std::vector<float>& out) {
    out.assign(static_cast<size_t>(cols), 0.0f);
    for (int c = 0; c < cols; ++c) {
        float acc = bias ? bias[c] : 0.0f;
        for (int r = 0; r < rows; ++r) acc += in[r] * W[static_cast<size_t>(r) * cols + c];
        out[c] = acc;
    }
}

inline void relu_inplace(std::vector<float>& v) {
    for (float& x : v) if (x < 0.0f) x = 0.0f;  // xp.maximum(z, 0.0)
}

}  // namespace

NetForward::NetForward(const WeightPayload& payload) {
    auto by_name = index_by_name(payload);

    // ---- residual / policy toggles: DERIVED from the manifest, exactly forward_core's key test ----
    // forward_core applies the block iff `"Wr1" in params` and the policy head iff `"Wp" in params`.
    // The manifest carries `residual` (and the Wr* entries iff so). We honor BOTH and require them to
    // agree: a residual=true manifest must carry Wr1/.. , and the presence of Wr1 is the real toggle.
    bool has_wr1 = by_name.count("Wr1") > 0;
    residual_ = has_wr1;  // the operative toggle is param presence (forward_core's `"Wr1" in params`)
    if (payload.residual != has_wr1) {
        // Port/ACL: translate-and-validate, do not coerce (ADR-0012 P2 / ADR-0002).
        throw std::runtime_error("chocofarm NetForward: manifest residual flag (" +
                                 std::string(payload.residual ? "true" : "false") +
                                 ") disagrees with the presence of Wr1 in the layout");
    }
    bool has_wp = by_name.count("Wp") > 0;

    // ---- trunk: W1 (in_dim, H), W2 (H, H), and the value head Wv (H, 1) — derive in_dim/H ----
    in_dim_ = -1; hidden_ = -1;
    require_matrix(by_name, "W1", in_dim_, hidden_, W1_.v);   // resolves in_dim_ and hidden_
    W1_.rows = in_dim_; W1_.cols = hidden_;
    { int len = hidden_; require_vector(by_name, "b1", len, b1_.v); }

    { int r = hidden_, c = hidden_; require_matrix(by_name, "W2", r, c, W2_.v); W2_.rows = r; W2_.cols = c; }
    { int len = hidden_; require_vector(by_name, "b2", len, b2_.v); }

    { int r = hidden_, c = 1; require_matrix(by_name, "Wv", r, c, Wv_.v); Wv_.rows = r; Wv_.cols = c; }
    { int len = 1; require_vector(by_name, "bv", len, bv_.v); }

    // ---- residual block (iff present): Wr1 (H, H), Wr2 (H, H) — same H as the trunk ----
    if (residual_) {
        { int r = hidden_, c = hidden_; require_matrix(by_name, "Wr1", r, c, Wr1_.v); Wr1_.rows = r; Wr1_.cols = c; }
        { int len = hidden_; require_vector(by_name, "br1", len, br1_.v); }
        { int r = hidden_, c = hidden_; require_matrix(by_name, "Wr2", r, c, Wr2_.v); Wr2_.rows = r; Wr2_.cols = c; }
        { int len = hidden_; require_vector(by_name, "br2", len, br2_.v); }
    }

    // ---- policy head (iff present): Wp (H, n_actions) — derive n_actions from its cols ----
    if (has_wp) {
        int r = hidden_, c = -1;
        require_matrix(by_name, "Wp", r, c, Wp_.v);   // resolves n_actions_ = c
        Wp_.rows = r; Wp_.cols = c;
        n_actions_ = c;
        { int len = n_actions_; require_vector(by_name, "bp", len, bp_.v); }
    } else {
        n_actions_ = 0;
    }

    // ---- value de-standardization scalars (the manifest's y_mean/y_std; predict_value applies them) ----
    y_mean_ = static_cast<float>(payload.y_mean);
    // mirror ValueMLP's y_std guard (y_std <= 1e-8 -> 1.0) so a degenerate scale never blows up.
    y_std_ = static_cast<float>(payload.y_std > 1e-8 ? payload.y_std : 1.0);
}

NetPrediction NetForward::predict(const std::vector<float>& X) const {
    if (static_cast<int>(X.size()) != in_dim_)
        throw std::runtime_error("chocofarm NetForward: X has length " + std::to_string(X.size()) +
                                 " != in_dim " + std::to_string(in_dim_));
    return predict(X.data());
}

NetPrediction NetForward::predict(const float* X) const {
    // forward_core, verbatim, B=1, float32:
    //   z1 = X@W1 + b1; a1 = ReLU(z1)
    std::vector<float> a1;
    matvec_bias(X, W1_.v, in_dim_, hidden_, b1_.v.data(), a1);
    relu_inplace(a1);
    //   z2 = a1@W2 + b2; a2 = ReLU(z2)
    std::vector<float> a2;
    matvec_bias(a1.data(), W2_.v, hidden_, hidden_, b2_.v.data(), a2);
    relu_inplace(a2);

    //   if residual: head_in = a2 + (ReLU(a2@Wr1+br1)@Wr2+br2)   -- PRE-activation skip, NO outer ReLU
    //   else:        head_in = a2
    std::vector<float> head_in = a2;  // copy; we add the residual delta in place when present
    if (residual_) {
        std::vector<float> zr1;
        matvec_bias(a2.data(), Wr1_.v, hidden_, hidden_, br1_.v.data(), zr1);
        relu_inplace(zr1);  // ar1 = ReLU(a2@Wr1 + br1)
        std::vector<float> zr2;
        matvec_bias(zr1.data(), Wr2_.v, hidden_, hidden_, br2_.v.data(), zr2);  // a2@Wr1.. @Wr2 + br2
        for (int j = 0; j < hidden_; ++j) head_in[j] = a2[j] + zr2[j];  // a2 + zr2 (NO outer ReLU)
    }

    //   v_std = (head_in @ Wv + bv).ravel()  -- the standardized scalar value
    std::vector<float> v_out;
    matvec_bias(head_in.data(), Wv_.v, hidden_, 1, bv_.v.data(), v_out);
    float v_std = v_out[0];

    NetPrediction out;
    // de-standardize back to the λ-penalized return scale (predict_value: v_std*y_std + y_mean).
    out.value = v_std * y_std_ + y_mean_;

    //   logits = head_in @ Wp + bp  (iff the net carries a policy head; empty otherwise, == None)
    if (has_policy()) {
        matvec_bias(head_in.data(), Wp_.v, hidden_, n_actions_, bp_.v.data(), out.logits);
    }
    return out;
}

}  // namespace chocofarm
