// cpp/src/gumbel.cpp
// Purpose: the C++ Gumbel-AlphaZero search Policy implementation (see gumbel.hpp). A faithful
//   reimplementation of the DISCRETE STRUCTURE of chocofarm/az/gumbel_search.py against the C++ env +
//   the NetEvaluator leaf port (ADR-0012 P2/P7: behavioral parity, NOT byte-identity; the env/runner
//   core is untouched). PHASE 1a: the algorithm structure + selection logic, validated exact-action on
//   precision-insensitive (coarse, well-separated) scripted leaf inputs.
//
//   Parity-critical detail (the same hazard the ISMCTS strict-`>`/first-wins cleared): the PUCT scan
//   iterates node.legal_slots in env-order with a strict `>` first-wins tie (mirrors Python `if v >
//   best_v` over node.legal); the SH cut sorts by g+logit+σ·q̂ with a STABLE descending sort (mirrors
//   Python `sorted(..., reverse=True)`, which is stable so equal keys keep their relative order). The
//   maps are keyed by action SLOT, but every order-sensitive scan is over legal_slots (a list), never
//   the std::map's sorted-key order.
//
//   1b SEAM (TIGHTENED — this file is the byte-identity port of value_target.py's mixed precision):
//   the σ-transform / v_mix / PUCT / log-prior reproduce numpy's DELIBERATE float32-prior × float64-Q
//   promotion EXACTLY. The float32 enters at FOUR places the Python rule's comments flag
//   (value_target.py:209-249, gumbel_search.py:397-426/436-458), each localized below. ALL FOUR read
//   the prior through ONE invariant (`prior_value` below): every prior read is the FLOAT32 stored prior
//   (`node.prior`) — the precision Python's float32 `root.prior` carries. This is the byte-faithful,
//   production path.
//
//   DISCRIMINATION CONTROL RETIRED (experiment/drop-prior-d): an earlier draft carried a second prior,
//   `node.prior_d` (the full-float64 pre-narrowing masked-softmax), toggled by CHOCO_GUMBEL_UNIFORM
//   (`kUniform`) to read the genuine 1a all-`double` port at every site — the discrimination control
//   that PROVED the float32 prior precision (not the structure) decides the near-ties. Its
//   non-vacuousness was already established (the uniform arm diverged X/N while the mixed arm matched
//   N/N). The control and its `node.prior_d` member were removed wholesale on this branch; the mixed
//   144/144 parity (cpp/parity/gumbel_precision.py, MIXED leg) remains the standing float32-seam
//   regression guard. With the control gone the GumbelNode struct carries ONE prior, not two, and the
//   prior read is unconditionally the float32 stored prior.
//
//     1. `evaluate` stores `node.prior` (float32, the net's wire dtype the Python search side-reads as
//        root.prior). The masked softmax that BUILDS the prior runs in float64; only the STORED `prior`
//        is narrowed. The downstream LOG-PRIOR root logit (`run_search`: logits[s]=log(prior_value(s)))
//        is the DOMINANT float32 effect on the discrete output (~1e-7 on log(prior)): it feeds the
//        Gumbel-top-k `logit+g` AND the SH cut key `g+logit+σ·q̂`, so the float32 prior FLIPS the SH
//        survivor and the improved-π argmax on near-tie inputs (vs the retired double control). This is
//        the seam the discrimination control proved load-bearing on the DISCRETE output.
//     2. `v_mix_mixed` computes the prior-weighted blend in FLOAT32: numpy weak-promotes
//        `prior[s](f32) * q(pyfloat) → f32`, and `pyfloat += f32 → f32`, so `pw_num`/`pw_den`/`v_bar`
//        AND the whole `(v_net + ΣN·v̄)/(1+ΣN)` return are float32 (the v_mix result is np.float32).
//     3. `improved_policy` completes UNVISITED slots with `σ·v_mix` rounded to FLOAT32 (numpy
//        `pyfloat(σ) * f32(vm) → f32`), added to the float64 root logit (numpy `f64 + f32 → f64`);
//        VISITED slots use the full-float64 `σ·q` (q is a Python float there). The masked softmax /
//        argmax over `completed` then run in float64 (matching _masked_softmax / np.argmax).
//     4. `puct_select` scores `q + c_puct·p·√ΣN/(1+n)` in FLOAT32 (numpy `p(f32)` weak-promotes the
//        whole U-term and the `q +` to float32), so the interior near-tie argmax is decided in float32.
//   The SH cut key `g + logit + σ·q̂`'s σ·q̂ stays float64 on BOTH sides (g/logits/sigma float64, q̂ a
//   Python float); the float32 enters that key ONLY through `logits = log(prior)` (seam 1).
//
//   HONEST SCOPE OF SEAMS 2/3/4 (verified, see the 1b audit): the VALUE fidelity of v_mix / σ·v_mix /
//   PUCT is byte-faithful to numpy (the 1e-4 value bar, P6) and the mixed path implements them so the
//   completed-Q / improved-π LOGITS match Python bit-for-bit. But on the DISCRETE output they are
//   near-unobservable in THIS search: v_mix (seams 2/3) feeds ONLY the improved-π UNVISITED-slot
//   completion, and an unvisited slot never wins the argmax (σ amplifies any visited-q lead; v_mix is a
//   blend that never exceeds the best visited q); PUCT (seam 4) CAN flip the survivor via interior
//   child selection but only on a ~1e-8 interior tie that fine inputs essentially never hit. So the
//   DISCRETE discrimination below rests on seam 1 (the prior precision). We do NOT over-claim that all
//   four seams flip the discrete output — claiming so would be the hack the audit names.
//
//   DISCRIMINATION CONTROL (RETIRED on experiment/drop-prior-d): the former CHOCO_GUMBEL_UNIFORM=1
//   `kUniform` arm read the FULL-float64 prior (`node.prior_d`) at every site (the genuine 1a
//   uniform-`double` port). On the COARSE 1a inputs (no near-ties) uniform==mixed; on the FINE near-tie
//   inputs (cpp/parity/gumbel_precision.py) the uniform port DIVERGED from Python on a LARGE fraction
//   while the mixed port matched N/N — the load-bearing proof that the float32 PRIOR precision, not the
//   structure, is what 1b fixed. That proof stands; the control arm + its `node.prior_d` data were
//   removed (see the header note above). The mixed 144/144 leg remains the float32-seam regression guard.
//
// Public Domain (The Unlicense).
#include "chocofarm/gumbel.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <span>

namespace chocofarm {

namespace {
// The MUTATION seam (test-only, the logic-check's discrimination proof). Reading CHOCO_GUMBEL_MUTATE
// once at startup, the search can be made to break the SH budget accounting (drop the full-budget
// remainder loop) or the PUCT formula (flip the U-term sign) — a DELIBERATE structural break the
// gumbel_logic.py mutation control sets and asserts diverges from the UNMODIFIED Python reference,
// proving the harness catches a real port bug (not just that an equality fires). Default (unset) is the
// FAITHFUL search; no production path sets it. This is the honest control: it mutates the ACTUAL search
// artifact, not the reference side.
enum class Mutate { None, ShBudget, Puct };
[[nodiscard]] Mutate read_mutate() {
    const char* m = std::getenv("CHOCO_GUMBEL_MUTATE");
    if (m == nullptr) return Mutate::None;
    if (std::strcmp(m, "sh-budget") == 0) return Mutate::ShBudget;
    if (std::strcmp(m, "puct") == 0) return Mutate::Puct;
    return Mutate::None;
}
const Mutate kMutate = read_mutate();  // read once (a process-lifetime test seam)

// The ONE prior-precision rule shared by ALL FOUR float32 prior read sites (the log-prior logit build,
// v_mix, the σ·v_mix completion, PUCT): every read is the FLOAT32 stored prior (`node.prior`) — the
// precision Python's float32 root.prior carries, the byte-faithful production path. (The retired
// CHOCO_GUMBEL_UNIFORM=1 control once routed these through a full-float64 `node.prior_d`; that arm and
// its member were removed on experiment/drop-prior-d — see the header note.)
[[nodiscard]] double prior_value(const GumbelNode& node, SlotIndex s) {
    return static_cast<double>(node.prior[static_cast<size_t>(s.value())]);  // .value() = index->size_t ACL
}
}  // namespace

namespace {
// Reconstruct an Action from its slot (the inverse of action_to_slot). Slot 0..N-1 = ("t", i);
// N..N+nD-1 = ("d", j); N+nD = TERMINATE.
// ACL: Action.i is a raw int (env.hpp, out of scope) carrying a treasure/face id; env.N()/n_detectors()
// return raw int. slot.value() crosses SlotIndex->raw at the kind decision + the detector-offset subtract.
[[nodiscard]] Action action_of_slot(const Environment& env, SlotIndex slot) {
    const int s = static_cast<int>(slot.value());
    if (s < env.N()) return Action{ActionKind::Treasure, s};
    if (s < env.N() + env.n_detectors()) return Action{ActionKind::Detector, s - env.N()};
    return terminate_action();
}

// The σ-transform scale prefactor (mirrors value_target.sigma_scale): (c_visit + max_a N(a))·c_scale.
// max over visited legal slots. INTEGER max-reduction — robust, precision-independent.
[[nodiscard]] double sigma_scale_1a(const GumbelNode& node, double c_visit, double c_scale) {
    VisitCount max_n{0};
    for (SlotIndex s : node.legal_slots) {
        const VisitCount n = node.N[static_cast<size_t>(s.value())];  // 0 if unvisited (former N.find==end)
        if (n > max_n) max_n = n;
    }
    return (c_visit + static_cast<double>(max_n.value())) * c_scale;
}

// The Danihelka §3 value-completion v_mix for unvisited actions (mirrors value_target.v_mix):
//   v_mix = (v_net + ΣN·v̄)/(1+ΣN),  v̄ = Σ_{N>0} π(b)Q(b) / Σ_{N>0} π(b)   (PRIOR-weighted).
// Returns v_net unchanged when nothing was visited or all visited priors are 0.
//
// 1b SEAM (seam 2): the Python rule computes this ENTIRELY in float32 — `prior[s](f32) * q(pyfloat)`
// weak-promotes to f32; `pyfloat(0.0) += f32 → f32` so `pw_num`/`pw_den`/`v_bar` are f32; `sum_n(pyint)
// * v_bar(f32) → f32` and `root_value(pyfloat) + f32 → f32`, `/ (1+sum_n)(pyint) → f32`, so the WHOLE
// return is np.float32 (value_target.py:226-249, "the v_mix return is np.float32"). We mirror it with
// `float` arithmetic: narrow `prior[s]·q` to float (numpy-weak), accumulate/divide/blend in float, and
// widen the float32 result to double ONCE for the (lossless) return — exactly the value Python's f32
// v_mix carries when it flows into the f64 `logits[s] + σ·vm` add downstream. (The retired kUniform
// control once ran a full-`double` path here; that arm was removed on experiment/drop-prior-d.)
[[nodiscard]] double v_mix_mixed(const GumbelNode& node, double root_value) {
    VisitCount sum_n{0};  // ΣN over visited legal slots (the one-home for the former `long sum_n`)
    // mixed precision (production): float32 prior-weighted blend, byte-faithful to numpy's weak promotion.
    float pw_num = 0.0f, pw_den = 0.0f;
    for (SlotIndex s : node.legal_slots) {
        VisitCount n = node.N[static_cast<size_t>(s.value())];  // 0 if unvisited (dense; former N.find==end -> 0)
        if (n > VisitCount{0}) {
            sum_n += n;
            float p = node.prior[static_cast<size_t>(s.value())];      // the stored float32 prior
            // numpy `f32 * pyfloat → f32` casts the WEAK Python operand to float32 FIRST, then
            // multiplies in float32 (verified: cast-first, NOT a f64 multiply narrowed). Mirror it by
            // casting `q` to float before the multiply, so the product is computed in true float32.
            pw_num += p * static_cast<float>(node.q(s));                // f32 * f32(q) → f32
            pw_den += p;                                                // pyfloat(0) += f32 → f32
        }
    }
    if (sum_n > VisitCount{0} && pw_den > 0.0f) {
        float v_bar = pw_num / pw_den;                                  // f32 / f32 → f32
        // numpy weak-promotes the Python-float `root_value` to float32 BEFORE the add (verified:
        // `pyfloat + f32` casts the weak operand to f32 first, then adds in f32 — NOT f64-then-narrow).
        float rv = static_cast<float>(root_value);
        float sn = static_cast<float>(sum_n.value());                  // VisitCount->float (the count value)
        float vmix = (rv + sn * v_bar)                                 // f32 + (pyint*f32→f32) → f32
                     / (1.0f + sn);                                    // / (pyint) → f32
        return static_cast<double>(vmix);                             // lossless widen for the f64 add
    }
    return root_value;
}

// The masked softmax over legal slots (mirrors mlp.ValueMLP._masked_softmax): subtract the per-row
// legal max, exp, zero illegal, normalize. Inputs are slot-indexed; `legal_slots` selects the legal
// entries. Returns an (n_slots,) row, exactly 0.0 on illegal slots. Robust EXCEPT the per-row max
// argmax on a near-tie (the 1b hazard — coarse 1a inputs have no near-ties).
// The into-variant: write the (n_slots,) softmax row into the caller-owned `out` (resized then fully
// overwritten — every slot is set to 0.0 first, the legal ones then normalized) instead of allocating a
// fresh vector per call (ADR-0012 P9 hot-path exception). The WRITTEN VALUES are byte-identical to the
// value-returning masked_softmax_1a for the same inputs — same body, `out` as the destination. Route the
// per-leaf evaluate() prior build through here; the per-decision improved_policy keeps the value-returning
// form (one alloc per decision, not per leaf).
void masked_softmax_1a_into(const std::vector<double>& completed,
                            std::span<const SlotIndex> legal_slots, SlotCount n_slots,
                            std::vector<double>& out) {
    out.assign(static_cast<size_t>(n_slots.value()), 0.0);  // .value() = slot-count->size_t ACL
    if (legal_slots.empty()) return;
    double row_max = -std::numeric_limits<double>::infinity();
    for (SlotIndex s : legal_slots) row_max = std::max(row_max, completed[static_cast<size_t>(s.value())]);
    double denom = 0.0;
    for (SlotIndex s : legal_slots) {
        double e = std::exp(completed[static_cast<size_t>(s.value())] - row_max);
        out[static_cast<size_t>(s.value())] = e;
        denom += e;
    }
    if (denom <= 0.0) denom = 1.0;
    for (SlotIndex s : legal_slots) out[static_cast<size_t>(s.value())] /= denom;
}

[[nodiscard]] std::vector<double> masked_softmax_1a(const std::vector<double>& completed,
                                                    std::span<const SlotIndex> legal_slots,
                                                    SlotCount n_slots) {
    std::vector<double> out;
    masked_softmax_1a_into(completed, legal_slots, n_slots, out);
    return out;
}
}  // namespace

GumbelAZPolicy::GumbelAZPolicy(const GumbelConfig& cfg, const NetEvaluator& net,
                               const Environment& env)
    : cfg_(cfg), net_(net), env_(env), fb_(env), n_slots_(n_action_slots(env)),
      term_slot_(term_slot(env)) {}

// ---- net evaluation (one forward, cached on the node) (mirrors _evaluate) --------------------------
double GumbelAZPolicy::evaluate(GumbelNode& node, const Loc& loc, const Belief& bw,
                                const CollectedSet& collected) const {
    // build the feature vector + the legal mask, run one forward through the net port (the leaf seam).
    // The per-leaf work writes into the policy's reused FeatureWorkspace (ws_) rather than allocating fresh
    // vectors each leaf (ADR-0012 P9 hot-path exception): the buffers are per-policy == per-tree/per-fiber
    // (gumbel.hpp ws_), so the reuse is clobber-safe (one tree's leaves run sequentially; the parked fiber's
    // ch.features aliases ws_.feat32 only until the driver encodes it at submit, before the next leaf's
    // evaluate overwrites it). The WRITTEN VALUES are byte-identical to the value-returning forms (P6) —
    // feat32 is the SAME per-element float narrowing of feat64 the former `vector<float>(b,e)` copy produced.
    //
    // MEASURED (honest, ADR-0009): a before/after K=64 wire profile showed the FEATURE-triple reuse
    // (feat64/feat32/mask, below) is metric-NEUTRAL on the malloc bucket — a byte-identical steady-state
    // refactor, NOT the source of the ~20% bucket. The bucket the profile flagged is the per-leaf
    // temporaries further down THIS function (logits_d + the masked-softmax prior, each ~n_slots, freshly
    // heap-allocated every leaf); those now reuse ws_.logits_d / ws_.prior_scratch — the per-leaf win.
    fb_.build_into(loc.pt, bw, collected, ws_.feat64);
    ws_.feat32.assign(ws_.feat64.begin(), ws_.feat64.end());  // the wire dtype the port consumes (float32)
    std::span<const float> feat(ws_.feat32);
    fb_.legal_mask_into(feat, ws_.mask);  // reuse build's belief sweep (P1)
    const std::vector<float>& mask = ws_.mask;

    auto pred = net_.predict(feat);
    // The local NetForward / the scripted leaf always return the value arm; a remote leaf's failure is
    // a typed Error. In the search the leaf is on a TOTAL path (we hold a live net), so a failure here
    // is a programmer/operator boundary fault — fail loud (ADR-0002 / P9) rather than silently degrade.
    assert(pred.has_value() && "gumbel: net leaf evaluation failed (NetEvaluator returned an Error)");
    const NetPrediction& np = *pred;

    // the prior = masked softmax of the net logits over the legal slots (mirrors predict_both: the net
    // emits raw logits, the search softmaxes them under the mask). 1b SEAM 1: `node.prior` is the float32
    // prior array the Python search side-reads (root.prior, float32) — the precision every read site uses.
    // (The retired kUniform control once also kept a full-float64 `node.prior_d`; removed on this branch.)
    // Reuse the policy's per-leaf scratch for the net-logits-as-double row instead of allocating a fresh
    // vector each leaf (ADR-0012 P9; ws_ is per-policy == per-tree/per-fiber, clobber-safe — one tree's
    // leaves run sequentially, each parked fiber has its OWN ws_). `assign` re-fills the whole slot space
    // with the -1e30 illegal sentinel, byte-identical to the former fresh `vector(n_slots_, -1e30)`.
    std::vector<double>& logits_d = ws_.logits_d;
    logits_d.assign(static_cast<size_t>(n_slots_.value()), -1e30);  // .value() = slot-count->size_t ACL
    // collect the legal slots from the MASK (the available/informative blocks build() already produced
    // — no second env.legal_actions → marginals sweep). Treasure slots 0..N-1 then detector slots
    // N..N+nD-1 (id order), then TERMINATE — the SAME order env.legal_actions yields (available = the
    // collect test, informative = env.informative), so node.legal_slots is bit-identical.
    // ACL: env.N()/n_detectors() are raw int (env.hpp); the mask is a raw-slot-position float vector. The
    // treasure slot i and detector slot N+j cross raw->SlotIndex at the push (the slot bijection's legs).
    node.legal_slots.clear();
    const int N = env_.N(), nD = env_.n_detectors();
    for (int i = 0; i < N; ++i)
        if (mask[static_cast<size_t>(i)] != 0.0f) node.legal_slots.push_back(SlotIndex{static_cast<LayoutRep>(i)});
    for (int j = 0; j < nD; ++j)
        if (mask[static_cast<size_t>(N + j)] != 0.0f) node.legal_slots.push_back(SlotIndex{static_cast<LayoutRep>(N + j)});
    node.legal_slots.push_back(term_slot_);  // TERMINATE is always legal
    // build the masked-softmax prior from the net logits. The net carries n_slots logits (the policy
    // head emits over the full slot space); illegal slots are masked.
    for (SlotIndex s : node.legal_slots) {
        // np.logits may be empty (value-only net) — in 1a the scripted leaf always carries logits; a
        // production value-only net would need a uniform prior, but the AZ search requires a policy head
        // (mirrors predict_both's n_actions assert), so we read the logit directly.
        assert(!np.logits.empty() && "gumbel: net has no policy head (logits empty)");
        logits_d[static_cast<size_t>(s.value())] = static_cast<double>(np.logits[static_cast<size_t>(s.value())]);
    }
    // build the masked-softmax prior into the policy's per-leaf scratch (P9), then move it onto the node:
    // the value is byte-identical to the former fresh-vector masked_softmax_1a, but the per-leaf heap
    // allocation is amortized into ws_.prior_scratch (reused across this tree's sequential leaves).
    masked_softmax_1a_into(logits_d, node.legal_slots, n_slots_, ws_.prior_scratch);
    // store the float32 narrowing of the masked-softmax prior (the precision Python's root.prior holds).
    // The softmax that BUILDS it ran in float64 (ws_.prior_scratch); only the STORED node.prior is
    // narrowed. (The retired kUniform control once ALSO stored a full-float64 node.prior_d here — a
    // per-node 65-double pmr alloc + .assign that the production path never read, finding #32; that
    // member was removed wholesale on experiment/drop-prior-d, so this is now a single store.)
    const std::vector<double>& prior_f64 = ws_.prior_scratch;
    node.prior.assign(static_cast<size_t>(n_slots_.value()), 0.0f);
    for (size_t s = 0; s < static_cast<size_t>(n_slots_.value()); ++s)
        node.prior[s] = static_cast<float>(prior_f64[s]);  // dense per-slot float32 narrowing (raw row index)

    node.value = static_cast<double>(np.value);
    node.evaluated = true;
    return node.value;
}

// ---- OPTION B eval split (additive; evaluate() above is UNTOUCHED) --------------------------------
// The pre-predict half of evaluate(): build the feature row + legal mask + node.legal_slots, returning
// the float32 feature span the leaf must forward. BYTE-IDENTICAL to evaluate()'s body up to (not
// including) net_.predict — the SAME fb_.build_into / fb_.legal_mask_into / legal-slot collection. The
// cursor parks on the returned span, the driver predicts, then eval_finish() consumes the prediction.
// Uses the CALLER's FeatureWorkspace (the cursor's own — never ws_), so a TreeCursor's per-leaf scratch
// is isolated from run_search's / the fiber's ws_ (P9 rule 4, per-owner scratch).
std::span<const float> GumbelAZPolicy::eval_build_features(GumbelNode& node, const Loc& loc,
                                                           const Belief& bw, const CollectedSet& collected,
                                                           FeatureWorkspace& ws) const {
    fb_.build_into(loc.pt, bw, collected, ws.feat64);
    ws.feat32.assign(ws.feat64.begin(), ws.feat64.end());  // the wire dtype the port consumes (float32)
    std::span<const float> feat(ws.feat32);
    fb_.legal_mask_into(feat, ws.mask);
    const std::vector<float>& mask = ws.mask;
    node.legal_slots.clear();
    const int N = env_.N(), nD = env_.n_detectors();  // ACL: env cardinalities raw int (env.hpp)
    for (int i = 0; i < N; ++i)
        if (mask[static_cast<size_t>(i)] != 0.0f) node.legal_slots.push_back(SlotIndex{static_cast<LayoutRep>(i)});
    for (int j = 0; j < nD; ++j)
        if (mask[static_cast<size_t>(N + j)] != 0.0f) node.legal_slots.push_back(SlotIndex{static_cast<LayoutRep>(N + j)});
    node.legal_slots.push_back(term_slot_);  // TERMINATE is always legal
    return feat;
}

// The post-predict half of evaluate(): given the leaf NetPrediction, build node.prior (the float64
// masked softmax, narrowed to its float32 store — the 1b seam-1) + store node.value — BYTE-IDENTICAL to
// evaluate()'s tail (same masked_softmax_1a_into, same per-element float32 store). node.legal_slots must
// already be set (by eval_build_features). The masked softmax that BUILDS the prior runs in float64;
// only the stored node.prior is narrowed (the precision Python's float32 root.prior carries). Reuses the
// caller's FeatureWorkspace logits_d/prior_scratch (the cursor owns one per slot — the SAME ws eval_build_
// features wrote feat32/mask into; logits_d/prior_scratch are disjoint from those, and feat32 is already
// consumed by the time the driver returns the prediction, so there is no aliasing). This amortizes the
// per-leaf heap alloc the local vectors re-paid every leaf — the exact ws_ amortization evaluate() has
// (finding #32-sibling: the cursor's eval-half had re-introduced the per-leaf malloc bucket). BYTE-
// IDENTICAL: same masked_softmax_1a_into, same per-element float32 store; only the buffers are reused.
void GumbelAZPolicy::eval_finish(GumbelNode& node, const NetPrediction& np, FeatureWorkspace& ws) const {
    std::vector<double>& logits_d = ws.logits_d;
    logits_d.assign(static_cast<size_t>(n_slots_.value()), -1e30);
    for (SlotIndex s : node.legal_slots) {
        assert(!np.logits.empty() && "gumbel(cursor): net has no policy head (logits empty)");
        logits_d[static_cast<size_t>(s.value())] = static_cast<double>(np.logits[static_cast<size_t>(s.value())]);
    }
    masked_softmax_1a_into(logits_d, node.legal_slots, n_slots_, ws.prior_scratch);
    node.prior.assign(static_cast<size_t>(n_slots_.value()), 0.0f);
    for (size_t s = 0; s < static_cast<size_t>(n_slots_.value()); ++s)
        node.prior[s] = static_cast<float>(ws.prior_scratch[s]);  // dense per-slot float32 narrowing (raw row index)
    node.value = static_cast<double>(np.value);
    node.evaluated = true;
}

double GumbelAZPolicy::sh_cut_sigma(const GumbelNode& root) const {
    return sigma_scale_1a(root, cfg_.c_visit, cfg_.c_scale);
}

double GumbelAZPolicy::root_logit(const GumbelNode& root, SlotIndex s) const {
    return std::log(std::max(prior_value(root, s), 1e-12));
}

// ---- AlphaZero PUCT select (mirrors _puct_select) -------------------------------------------------
SlotIndex GumbelAZPolicy::puct_select(const GumbelNode& node) const {
    // total_n = ΣN over the slot space. The former std::map held ONLY visited slots; the dense vector holds
    // ALL slots with unvisited == 0, so summing the whole vector gives the IDENTICAL total (the 0 entries
    // add nothing). Order-independent either way (a sum). VisitCount is additive (a count + a count).
    VisitCount total_n{0};
    for (VisitCount n : node.N) total_n += n;
    double sqrt_total = (total_n > VisitCount{0}) ? std::sqrt(static_cast<double>(total_n.value())) : 1.0;
    double base_v = node.value;  // unvisited Q completed by the node's own net value
    std::optional<SlotIndex> best_a;  // typed absence over the -1 "no best slot" sentinel (ADR-0002)
    double best_v = -std::numeric_limits<double>::infinity();
    // iterate node.legal_slots (the env-order list), strict `>` first-wins — mirrors Python's loop over
    // node.legal with `if v > best_v`, never the std::map's sorted-key order.
    for (SlotIndex s : node.legal_slots) {
        VisitCount n = node.N[static_cast<size_t>(s.value())];  // 0 if unvisited (dense; former N.find==end -> 0)
        double q = (n > VisitCount{0})
                       ? (node.W[static_cast<size_t>(s.value())] / static_cast<double>(n.value()))
                       : base_v;
        // 1b SEAM (seam 4): Python computes `q + c_puct·p·√ΣN/(1+n)` with `p` the float32 prior scalar,
        // which numpy weak-promotes through the WHOLE U-term and the `q +` to float32 — so the interior
        // near-tie argmax is decided in FLOAT32 (gumbel_search.py:397-426). Mirror it: cast the Python-
        // float operands to float so each numpy op runs in true float32 (cast-first weak promotion).
        // (The retired kUniform control once ran a full-`double` path here; removed on this branch.)
        float p = node.prior[static_cast<size_t>(s.value())];               // stored float32 prior
        float u = static_cast<float>(cfg_.c_puct) * p                        // pyfloat·f32 → f32
                  * static_cast<float>(sqrt_total)                           // ·pyfloat → f32
                  / (1.0f + static_cast<float>(n.value()));                  // /(pyint) → f32
        float vf = (kMutate == Mutate::Puct)
                       ? (static_cast<float>(q) - u)                         // q(pyfloat) ± f32 → f32
                       : (static_cast<float>(q) + u);
        double v = static_cast<double>(vf);  // lossless widen; the float32 ordering drives the argmax
        // strict `>` first-wins over node.legal_slots (mirrors Python `if v > best_v`). best_v starts at
        // -inf (the first slot always wins); thereafter the comparison is between the float32-rounded
        // scores (widened to double, an order-preserving widen) — the near-tie side Python lands on.
        if (v > best_v) {
            best_v = v;
            best_a = s;
        }
    }
    assert(best_a.has_value() && "puct_select on a node with no legal slots");
    return *best_a;
}

// ---- interior PUCT descent; net value at the leaf (mirrors _descend) ------------------------------
double GumbelAZPolicy::descend(NodePool& nodes, int node, const Loc& loc,
                               const Belief& bw, const CollectedSet& collected,
                               World world, double lam, GumbelSource& src, PlyDepth depth) const {
    // cfg_.max_depth is now a PlyDepth directly (the config field carries its domain); no wrap at the cap.
    const PlyDepth max_depth = cfg_.max_depth;
    if (depth >= max_depth || env_.empty(bw)) {
        if (!nodes[static_cast<size_t>(node)].evaluated) {
            if (env_.empty(bw)) return -lam * env_.exit_cost(loc.pt);
            return evaluate(nodes[static_cast<size_t>(node)], loc, bw, collected);
        }
        return nodes[static_cast<size_t>(node)].value;
    }
    if (!nodes[static_cast<size_t>(node)].evaluated) {
        // first visit to this leaf: the net value IS the leaf estimate (no playout — the F4 cure)
        return evaluate(nodes[static_cast<size_t>(node)], loc, bw, collected);
    }

    SlotIndex a = puct_select(nodes[static_cast<size_t>(node)]);
    Action act = action_of_slot(env_, a);
    double ret;
    if (act.kind == ActionKind::Terminate) {
        ret = -lam * env_.exit_cost(loc.pt);  // stop now: only the exit toll remains
    } else {
        Loc nloc = loc;  // COPY: apply computes dt = dist(OLD loc, target) then moves nloc to target
        Belief nbw = bw;
        CollectedSet nc = collected;
        StepResult sr = env_.apply(nloc, nbw, nc, act, world);  // World == uint32_t (env.apply ACL)
        double step = sr.reward - lam * sr.dt;
        std::tuple<SlotIndex, GBeliefKey> ckey{a, gumbel_belief_key(env_, nbw)};
        auto& cur = nodes[static_cast<size_t>(node)];
        int child;
        auto cit = cur.children.find(ckey);
        if (cit == cur.children.end()) {
            nodes.emplace_back(n_slots_);  // dense W/N sized to the slot space, zero-initialized
            child = static_cast<int>(nodes.size()) - 1;
            nodes[static_cast<size_t>(node)].children[ckey] = child;  // re-index after possible realloc
        } else {
            child = cit->second;
        }
        double cont = descend(nodes, child, nloc, nbw, nc, world, lam, src,
                              depth + SearchRep{1});  // PlyDepth affine: depth + 1 ply
        ret = step + cont;
    }
    auto& cur = nodes[static_cast<size_t>(node)];
    cur.W[static_cast<size_t>(a.value())] += ret;       // dense, zero-init: += is the former (count?W[a]:0)+ret
    cur.N[static_cast<size_t>(a.value())] += VisitCount{1};  // dense, zero-init: += is the former (count?N[a]:0)+1
    return ret;
}

// ---- one sim of a root action (mirrors _simulate_root_action) -------------------------------------
double GumbelAZPolicy::simulate_root_action(NodePool& nodes, const Loc& loc,
                                            const Belief& bw,
                                            const CollectedSet& collected, SlotIndex slot, World world,
                                            double lam, GumbelSource& src) const {
    Action a = action_of_slot(env_, slot);
    if (a.kind == ActionKind::Terminate) return -lam * env_.exit_cost(loc.pt);
    // outcome-averaging over c_outcome determinizations of the IMMEDIATE outcome (k=0 reuses the
    // threaded world; k>0 draws a fresh world from the belief — mirrors _simulate_root_action). k is an
    // OutcomeIndex (the k==0 reuse vs k>0 redraw distinction); cfg_.c_outcome is now an OutcomeIndex
    // directly (the shared count/index domain — domains.hpp), the bound on the determinization loop.
    double total = 0.0;
    const OutcomeIndex c_outcome = cfg_.c_outcome;
    for (OutcomeIndex k{0}; k < c_outcome; k = k + SearchRep{1}) {
        World w = (k == OutcomeIndex{0}) ? world : src.sample_world(bw);
        Loc nloc = loc;  // COPY: apply computes dt = dist(OLD loc, target) then moves nloc to target
        Belief nbw = bw;
        CollectedSet nc = collected;
        StepResult sr = env_.apply(nloc, nbw, nc, a, w);
        double step = sr.reward - lam * sr.dt;
        std::tuple<SlotIndex, GBeliefKey> ckey{slot, gumbel_belief_key(env_, nbw)};
        int child;
        auto cit = nodes[0].children.find(ckey);
        if (cit == nodes[0].children.end()) {
            nodes.emplace_back(n_slots_);  // dense W/N sized to the slot space, zero-initialized
            child = static_cast<int>(nodes.size()) - 1;
            nodes[0].children[ckey] = child;
        } else {
            child = cit->second;
        }
        double cont = descend(nodes, child, nloc, nbw, nc, w, lam, src, PlyDepth{1});
        total += step + cont;
    }
    return total / static_cast<double>(cfg_.c_outcome.value());  // OutcomeIndex -> the double divisor ACL
}

// ---- run `count` sims of root action `slot` (mirrors _visit) --------------------------------------
void GumbelAZPolicy::visit(NodePool& nodes, const Loc& loc,
                           const Belief& bw, const CollectedSet& collected, SlotIndex slot,
                           double lam, GumbelSource& src, SimBudget count) const {
    for (SimBudget i{0}; i < count; i = i + SimBudget{1}) {  // SimBudget is additive (count accumulates)
        World w = src.sample_world(bw);
        double ret = simulate_root_action(nodes, loc, bw, collected, slot, w, lam, src);
        nodes[0].W[static_cast<size_t>(slot.value())] += ret;       // dense, zero-init: former (count?W:0)+ret
        nodes[0].N[static_cast<size_t>(slot.value())] += VisitCount{1};  // dense, zero-init: former (count?N:0)+1
    }
}

// ---- Sequential Halving (Danihelka §2) (mirrors _sequential_halving) ------------------------------
SlotIndex GumbelAZPolicy::sequential_halving(NodePool& nodes, const Loc& loc,
                                       const Belief& bw,
                                       const CollectedSet& collected, double lam, GumbelSource& src,
                                       std::vector<SlotIndex> considered, const std::vector<double>& g,
                                       const std::vector<double>& logits, SimBudget& n_spent) const {
    n_spent = SimBudget{0};
    // considered is non-empty by the caller's contract (run_search returns early on an empty belief, and a
    // non-empty belief always has >=1 legal slot => >=1 candidate). The former defensive `return -1` is a
    // typed-absence non-state now (SlotIndex carries no -1); assert the invariant (ADR-0002/P9 own-state).
    assert(!considered.empty() && "sequential_halving on an empty candidate set (caller contract)");
    // cfg_.n_sims is now a SimBudget directly (the config field carries its domain); no wrap at the seed.
    const SimBudget n_sims = cfg_.n_sims;
    if (considered.size() == 1) {
        visit(nodes, loc, bw, collected, considered[0], lam, src, n_sims);
        n_spent = n_sims;
        return considered[0];
    }

    // m / n_phases / keep are CandidateCount (set-cardinality); the budget family is SimBudget. The
    // std::max/std::min/division mix the raw ints the formulas demand, crossed at .value()/wrap.
    const CandidateCount m{static_cast<SearchRep>(considered.size())};
    int n_phases = std::max(1, static_cast<int>(std::ceil(std::log2(static_cast<double>(m.value())))));
    SimBudget per_phase{static_cast<SearchRep>(std::max(1, static_cast<int>(n_sims.value()) / n_phases))};  // paper's N/⌈log2 m⌉ (SimBudget.value() at the division ACL)
    SimBudget budget = n_sims;

    while (considered.size() > 1 && budget > SimBudget{0}) {
        SimBudget phase_budget{std::min(per_phase.value(), budget.value())};
        SimBudget per_action{static_cast<SearchRep>(
            std::max(1, static_cast<int>(phase_budget.value()) / static_cast<int>(considered.size())))};
        for (SlotIndex s : considered) {
            SimBudget v{std::min(per_action.value(), budget.value())};
            if (v == SimBudget{0}) break;  // unsigned: v<=0 is v==0 (the never-negative invariant, ADR-0000)
            visit(nodes, loc, bw, collected, s, lam, src, v);
            budget = SimBudget{budget.value() - v.value()};  // budget draws down (no quantity_sub trait)
            n_spent += v;
        }
        // drop the worst half by g + logit + σ·q̂ (σ recomputed each phase as max_a N(a) grows).
        double sigma = sigma_scale_1a(nodes[0], cfg_.c_visit, cfg_.c_scale);
        // STABLE descending sort by the cut key — mirrors Python `sorted(considered, key=..., reverse
        // =True)`, which is stable so equal keys keep their relative (pre-sort) order. std::stable_sort
        // with a strict-`>` comparator gives the SAME order (a `>`-comparator stable sort = a reversed
        // stable ascending sort on a total order; on the coarse precision-insensitive inputs the keys
        // are well-separated so no ties arise regardless).
        std::vector<std::pair<double, SlotIndex>> keyed;
        keyed.reserve(considered.size());
        for (SlotIndex s : considered) {
            double key = g[static_cast<size_t>(s.value())] + logits[static_cast<size_t>(s.value())] +
                         sigma * nodes[0].q(s);
            keyed.emplace_back(key, s);
        }
        std::stable_sort(keyed.begin(), keyed.end(),
                         [](const std::pair<double, SlotIndex>& a, const std::pair<double, SlotIndex>& b) {
                             return a.first > b.first;  // descending; stable keeps ties in input order
                         });
        int keep = std::max(1, static_cast<int>(keyed.size()) / 2);
        std::vector<SlotIndex> next;
        next.reserve(static_cast<size_t>(keep));
        for (int i = 0; i < keep; ++i) next.push_back(keyed[static_cast<size_t>(i)].second);
        considered = std::move(next);
    }

    // spend any rounding remainder on the survivor(s) so the FULL budget is used (round-robin).
    // MUTATION (test-only): a port that DROPS the remainder loop under-spends the budget (invariant ii
    // breaks) AND leaves the surviving Q-estimates with a different sim count than the faithful path,
    // which on the cases where the remainder is non-zero shifts the cross-language survivor/argmax. The
    // faithful path always spends the FULL budget here.
    if (kMutate != Mutate::ShBudget) {
        size_t i = 0;  // round-robin index over the survivors (a raw container cursor, read modulo size)
        while (budget > SimBudget{0} && !considered.empty()) {
            SlotIndex s = considered[i % considered.size()];
            visit(nodes, loc, bw, collected, s, lam, src, SimBudget{1});
            budget = SimBudget{budget.value() - 1};
            n_spent += SimBudget{1};
            ++i;
        }
    }
    return considered.front();
}

// ---- the improved-π target (mirrors _improved_policy -> value_target.improved_policy) -------------
std::vector<double> GumbelAZPolicy::improved_policy(const GumbelNode& root,
                                                    const std::vector<double>& logits) const {
    double sigma = sigma_scale_1a(root, cfg_.c_visit, cfg_.c_scale);
    double vm = v_mix_mixed(root, root.value);   // float32-faithful v_mix (returned widened to double)
    std::vector<double> completed(static_cast<size_t>(n_slots_.value()), -1e30);
    for (SlotIndex s : root.legal_slots) {
        VisitCount n = root.N[static_cast<size_t>(s.value())];  // 0 if unvisited (dense; former N.find==end -> 0)
        // 1b SEAM (seam 3): `completed[s] = logits[s] + σ·q`, where `logits[s]` is a float64 root logit
        // (an element of the float64 `logits` array) and σ is a Python float. For VISITED slots q is a
        // Python float (root.q), so `σ·q` is float64 → the whole term is float64. For UNVISITED slots q
        // is `vm`, an np.float32 (v_mix's float32 return), so numpy `pyfloat(σ)·f32(vm) → f32` and then
        // `f64(logits[s]) + f32 → f64` (the float32-rounded σ·vm added in float64). Mirror EXACTLY:
        // visited → full double; unvisited → round σ·vm to float (cast-first weak promotion), then add
        // in double. (The retired kUniform control once ran a full-`double` path here; removed on this
        // branch.)
        if (n > VisitCount{0}) {
            completed[static_cast<size_t>(s.value())] =
                logits[static_cast<size_t>(s.value())] + sigma * root.q(s);
        } else {
            float sigma_vm = static_cast<float>(sigma) * static_cast<float>(vm);  // pyfloat·f32 → f32
            completed[static_cast<size_t>(s.value())] =
                logits[static_cast<size_t>(s.value())] + static_cast<double>(sigma_vm);  // f64 + f32 → f64
        }
    }
    return masked_softmax_1a(completed, root.legal_slots, n_slots_);
}

// ---- the pure search core (mirrors _decide_root, temperature 0) -----------------------------------
GumbelAZPolicy::Decision GumbelAZPolicy::run_search(const Loc& loc, const Belief& bw,
                                                    const CollectedSet& collected, double lam,
                                                    GumbelSource& src) const {
    // Scope the FeatureBuilder belief memo to ONE decision/tree: the node cache already folds same-node
    // rebuilds, so the memo's increment is same-belief-different-slot reuse WITHIN the tree; beliefs
    // narrow across decisions (low cross-decision reuse), and this keeps the long-lived serve-path builder
    // from accumulating a process-lifetime cache. Correctness-safe (a hit is bit-identical), always sound.
    fb_.reset_belief_cache();
    // Recycle the per-decision node-pool arena: a monotonic_buffer_resource frees nothing until release(),
    // so release() here hands the whole PRIOR decision's node storage back to arena_buf_ for this decision
    // to reuse (ADR-0012 P9 rule 4). Same per-decision reset point as the belief cache; same per-policy /
    // per-fiber ownership (gumbel.hpp arena_). Correctness-safe: every node below is freshly constructed.
    arena_.release();
    Decision out;
    // empty-belief guard (mirrors decide_with_target's len(bw)==0): the only continuation is to exit.
    if (env_.empty(bw)) {
        out.action = terminate_action();
        out.improved.assign(static_cast<size_t>(n_slots_.value()), 0.0);
        out.improved[static_cast<size_t>(term_slot_.value())] = 1.0;
        out.survivor_slot = term_slot_;  // typed SlotIndex (the empty-belief Terminate survivor)
        out.n_spent = SimBudget{0};
        return out;
    }

    NodePool nodes{&arena_};       // the per-decision node pool, served from the per-policy pmr arena
    nodes.emplace_back(n_slots_);  // the root (arena index 0); dense W/N sized to the slot space
    evaluate(nodes[0], loc, bw, collected);

    // root logits = log(prior) over legal slots (the masked-softmax prior is the reference; its log is
    // the root logit). -1e30 on illegal (mirrors _decide_root's logits build).
    // 1b SEAM 1 (the DOMINANT float32 effect on the discrete output): Python reads `prior[s]` off the
    // float32 `root.prior` here, so `math.log(max(prior[s], 1e-12))` is a log of a float32 scalar — the
    // ~1e-7 float32 narrowing perturbs every root logit, and these logits feed BOTH the Gumbel-top-k
    // (logit+g) AND the SH cut key (g+logit+σ·q̂), so the float32 prior (vs the retired double control)
    // FLIPS the survivor and the improved-π argmax on near-tie inputs. `prior_value` reads the float32
    // stored prior (the production path; the uniform discrimination arm was removed on this branch).
    std::vector<double> logits(static_cast<size_t>(n_slots_.value()), -1e30);
    for (SlotIndex s : nodes[0].legal_slots) {
        double p = prior_value(nodes[0], s);
        logits[static_cast<size_t>(s.value())] = std::log(std::max(p, 1e-12));
    }

    // Gumbel-Top-k: one gumbel draw over the FULL slot space, sort logits+g, take top-m legal slots.
    // ACL: gumbel(int n) takes the slot-space draw length raw (the override-gated virtual) — n_slots_.value().
    std::vector<double> g = src.gumbel(static_cast<int>(n_slots_.value()));
    // score0 = where(logits > -1e29, logits+g, -inf) (mirrors _decide_root). Build over legal slots
    // only (illegal stay -inf), then take the top m = min(self.m, #legal).
    std::vector<std::pair<double, SlotIndex>> scored;  // (score, slot) over legal slots
    scored.reserve(nodes[0].legal_slots.size());
    for (SlotIndex s : nodes[0].legal_slots) {
        scored.emplace_back(logits[static_cast<size_t>(s.value())] + g[static_cast<size_t>(s.value())], s);
    }
    // top-m by descending score (mirrors np.argsort(score0)[::-1][:m]). On the coarse precision-
    // insensitive inputs the gumbel-perturbed scores are well-separated (no ties); a stable descending
    // sort fixes the order deterministically (Python's argsort is value-ordered for distinct scores).
    std::stable_sort(scored.begin(), scored.end(),
                     [](const std::pair<double, SlotIndex>& a, const std::pair<double, SlotIndex>& b) {
                         return a.first > b.first;
                     });
    // m = min(self.m, #legal): cfg_.m is now a CandidateCount; .value() at the std::min with the raw
    // legal-slot size (the hot top-m loop counter stays a raw int — the loop-mod carve-out).
    int m = std::min(static_cast<int>(cfg_.m.value()), static_cast<int>(nodes[0].legal_slots.size()));
    std::vector<SlotIndex> considered;
    considered.reserve(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i) considered.push_back(scored[static_cast<size_t>(i)].second);

    // Sequential Halving over n_sims; returns the surviving slot (the executed action at temperature 0).
    // n_spent is a SimBudget; Decision.n_spent is now the SAME domain (no cast at the store).
    SimBudget n_spent{0};
    SlotIndex survivor = sequential_halving(nodes, loc, bw, collected, lam, src, considered, g, logits,
                                            n_spent);
    out.n_spent = n_spent;

    // the improved-π target over the FULL legal set.
    out.improved = improved_policy(nodes[0], logits);

    // executed action = the SH survivor (Danihelka §2; temperature 0). The temperature>0 sampling path
    // is a 1b/production concern; 1a's logic check is temperature 0 (the eval policy's rule).
    out.survivor_slot = survivor;  // typed SlotIndex (the SH survivor; was a raw-int store)
    out.action = action_of_slot(env_, survivor);

    // HPO/BENCHMARK-ONLY no-early-exit substitution (cfg_.no_early_exit, default false → this whole block
    // is skipped and the decision is BYTE-UNCHANGED). The search above ran exactly as normal — Gumbel-Top-k,
    // Sequential Halving, the Terminate edge SAMPLED and BACKPROPPED, and out.improved (the improved-π
    // target) is the real, UNTOUCHED π. The ONLY change here: if the EXECUTED action came out Terminate and a
    // non-terminate legal action still exists (the empty-belief case returned above, so legal_slots may hold
    // Treasure/Detector slots besides term_slot_), substitute the executed action for the best non-terminate
    // option so the benchmark episode CONTINUES instead of early-exiting. We pick the non-terminate legal slot
    // with the highest out.improved (the search's own improved-π ranking, already computed) — the faithful
    // "best non-terminate" per the flag's doc comment in gumbel.hpp. NOTE (deliberate, not a bug): out.improved
    // still carries Terminate's weight while the executed action is the substitute. That is FINE — this path is
    // benchmark-only (the wire runner abandons episodes and writes NO training data), so no PI target is
    // corrupted; we intentionally leave out.improved and the backprop unchanged and substitute ONLY the
    // executed action (+ survivor_slot, kept consistent with it).
    if (cfg_.no_early_exit && out.action.kind == ActionKind::Terminate) {
        std::optional<SlotIndex> best_slot;  // typed absence over the -1 "no non-terminate option" sentinel
        double best_pi = -1.0;
        for (SlotIndex s : nodes[0].legal_slots) {
            if (s == term_slot_) continue;  // skip the Terminate slot; want a non-terminate legal action
            const double pi = out.improved[static_cast<size_t>(s.value())];
            if (pi > best_pi) {  // strict `>` first-wins, mirroring the search's argmax tie-break
                best_pi = pi;
                best_slot = s;
            }
        }
        if (best_slot.has_value()) {  // a non-terminate legal action exists → continue the episode on it
            out.survivor_slot = best_slot;  // typed SlotIndex (the substituted survivor)
            out.action = action_of_slot(env_, *best_slot);
        }
        // else: no non-terminate legal action (only term_slot_ legal) → leave Terminate; the episode
        // correctly ends, exactly as the empty-belief guard above does (there is nothing to substitute).
    }
    return out;
}

// RngGumbelSource (the production Gumbel source) now lives in gumbel.hpp (the ONE home, ADR-0012 P1):
// promoted out of this anonymous namespace so the LOCAL batched driver's per-slot TreeState constructs
// the SAME source decide_with_target does — byte-identical RNG draw order across the serial and batched
// paths. The class body is unchanged; decide_with_target below constructs the header class identically.

GumbelAZPolicy::Decision GumbelAZPolicy::decide_with_target(
    const Environment& env, const Loc& loc, const Belief& bw,
    const CollectedSet& collected, double lam, std::mt19937_64& rng) const {
    (void)env;  // the policy holds its own env_ ref (the seam passes env for the contract; same object)
    RngGumbelSource src(env_, rng);  // the env homes the uniform world draw (L1)
    return run_search(loc, bw, collected, lam, src);
}

ActionAndPi GumbelAZPolicy::decide_target(const Environment& env, const Loc& loc,
                                          const Belief& bw, const CollectedSet& collected,
                                          double lam, std::mt19937_64& rng) const {
    // The AZ runner's PI source: one search, the executed action + the REAL improved-π (float32).
    Decision dec = decide_with_target(env, loc, bw, collected, lam, rng);
    return ActionAndPi{dec.action, std::vector<float>(dec.improved.begin(), dec.improved.end())};
}

Action GumbelAZPolicy::decide(const Environment& env, const Loc& loc, const Belief& bw,
                              const CollectedSet& collected, double lam,
                              std::mt19937_64& rng) const {
    // decide() is decide_with_target() composed with `.action` (DRY — one search entry point).
    return decide_with_target(env, loc, bw, collected, lam, rng).action;
}

}  // namespace chocofarm
