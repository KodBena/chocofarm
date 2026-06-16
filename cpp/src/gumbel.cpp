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
//   the prior through ONE invariant (toggleable by CHOCO_GUMBEL_UNIFORM, the discrimination control):
//     * MIXED (default, faithful): every prior read is the FLOAT32 stored prior (`node.prior`) — the
//       precision Python's float32 `root.prior` carries. This is the byte-faithful path.
//     * UNIFORM (`kUniform`, CHOCO_GUMBEL_UNIFORM=1): every prior read is the FULL-FLOAT64 prior
//       (`node.prior_d`, the pre-narrowing masked-softmax) — the genuine 1a all-`double` port. The
//       toggle is ONE rule over ALL FOUR read sites (`prior_read` below), NOT a per-site gate (an
//       earlier draft gated only v_mix/PUCT but left the dominant log-prior path float32 in BOTH arms,
//       which made the control VACUOUS — the float32 effect leaked into both arms; see the audit).
//
//     1. `evaluate` stores BOTH `node.prior` (float32, the net's wire dtype the Python search
//        side-reads as root.prior) AND `node.prior_d` (the same masked-softmax in full float64). The
//        masked softmax that BUILDS the prior runs in float64 either way; only the STORED `prior` is
//        narrowed. The downstream LOG-PRIOR root logit (`run_search`: logits[s]=log(prior_read(s))) is
//        the DOMINANT float32 effect on the discrete output (~1e-7 on log(prior)): it feeds the
//        Gumbel-top-k `logit+g` AND the SH cut key `g+logit+σ·q̂`, so the float32-vs-double prior FLIPS
//        the SH survivor and the improved-π argmax on near-tie inputs. This is the seam the
//        discrimination control actually proves load-bearing on the DISCRETE output.
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
//   DISCRIMINATION CONTROL (CHOCO_GUMBEL_UNIFORM=1): `kUniform` reads the FULL-float64 prior at every
//   site (seams 1-4), i.e. the genuine 1a uniform-`double` port. On the COARSE 1a inputs (no near-ties)
//   uniform==mixed (the structure check is precision-insensitive); on the FINE near-tie inputs
//   (cpp/parity/gumbel_precision.py) the uniform port DIVERGES from Python on a LARGE fraction while
//   the mixed port matches N/N — the load-bearing proof that the float32 PRIOR precision, not the
//   structure, is what 1b fixed (the 1b analogue of the 1a mutation control).
//
// Public Domain (The Unlicense).
#include "chocofarm/gumbel.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

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

// The 1b DISCRIMINATION control (test-only): CHOCO_GUMBEL_UNIFORM=1 reverts the three float32 seams
// (v_mix, the unvisited σ·v_mix completion, the PUCT score) to the 1a uniform-`double` precision, so
// the precision parity harness can prove the float32 path — not the structure — is what decides the
// near-ties. Default (unset) is the FAITHFUL mixed precision (float32-prior × float64-Q) that matches
// Python's value_target.py byte-for-byte. No production path sets it.
[[nodiscard]] bool read_uniform() {
    const char* u = std::getenv("CHOCO_GUMBEL_UNIFORM");
    return u != nullptr && std::strcmp(u, "1") == 0;
}
const bool kUniform = read_uniform();  // read once (a process-lifetime test seam)

// The ONE prior-precision rule shared by ALL FOUR float32 prior read sites (the log-prior logit build,
// v_mix, the σ·v_mix completion, PUCT). MIXED (default) reads the float32 stored prior — the precision
// Python's float32 root.prior carries. UNIFORM (kUniform) reads the full-float64 pre-narrowing prior —
// the genuine 1a all-`double` port. Localizing the toggle HERE (one invariant over all readers, not a
// per-site gate) is what makes the discrimination control non-vacuous: under kUniform NO read site sees
// the float32 narrowing, so the dominant log-prior effect is double in the uniform arm and float32 in
// the mixed arm — the discrete divergence the control proves.
[[nodiscard]] double prior_read(const GumbelNode& node, int s) {
    return kUniform ? node.prior_d[static_cast<size_t>(s)]
                    : static_cast<double>(node.prior[static_cast<size_t>(s)]);
}
}  // namespace

// ---- belief key: the (count, first, last) fingerprint (mirrors _belief_key) -----------------------
GBeliefKey gumbel_belief_key(const std::vector<uint32_t>& bw) {
    if (bw.empty()) return GBeliefKey{0, 0u, 0u};
    return GBeliefKey{static_cast<int>(bw.size()), bw.front(), bw.back()};
}

namespace {
// The fixed slot for an action (the action<->slot bijection, mirrors action_to_slot).
[[nodiscard]] int slot_of(const Environment& env, const Action& a) { return action_to_slot(env, a); }

// Reconstruct an Action from its slot (the inverse of action_to_slot). Slot 0..N-1 = ("t", i);
// N..N+nD-1 = ("d", j); N+nD = TERMINATE.
[[nodiscard]] Action action_of_slot(const Environment& env, int slot) {
    if (slot < env.N()) return Action{ActionKind::Treasure, slot};
    if (slot < env.N() + env.n_detectors()) return Action{ActionKind::Detector, slot - env.N()};
    return terminate_action();
}

// The σ-transform scale prefactor (mirrors value_target.sigma_scale): (c_visit + max_a N(a))·c_scale.
// max over visited legal slots. INTEGER max-reduction — robust, precision-independent.
[[nodiscard]] double sigma_scale_1a(const GumbelNode& node, double c_visit, double c_scale) {
    int max_n = 0;
    for (int s : node.legal_slots) {
        auto it = node.N.find(s);
        if (it != node.N.end() && it->second > max_n) max_n = it->second;
    }
    return (c_visit + static_cast<double>(max_n)) * c_scale;
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
// v_mix carries when it flows into the f64 `logits[s] + σ·vm` add downstream. `kUniform` runs the 1a
// uniform-`double` path (the discrimination control: diverges from Python on the fine near-tie inputs).
[[nodiscard]] double v_mix_mixed(const GumbelNode& node, double root_value) {
    long sum_n = 0;
    if (kUniform) {
        // the genuine 1a all-`double` path: the FULL-float64 prior (prior_read -> prior_d) and double
        // arithmetic throughout (the discrimination control — diverges from Python on fine near-ties).
        double pw_num = 0.0, pw_den = 0.0;
        for (int s : node.legal_slots) {
            auto nit = node.N.find(s);
            int n = (nit != node.N.end()) ? nit->second : 0;
            if (n > 0) {
                sum_n += n;
                double p = prior_read(node, s);  // full-float64 prior (kUniform)
                pw_num += p * node.q(s);
                pw_den += p;
            }
        }
        if (sum_n > 0 && pw_den > 0.0) {
            double v_bar = pw_num / pw_den;
            return (root_value + static_cast<double>(sum_n) * v_bar)
                   / (1.0 + static_cast<double>(sum_n));
        }
        return root_value;
    }
    // mixed precision (default): float32 prior-weighted blend, byte-faithful to numpy's weak promotion.
    float pw_num = 0.0f, pw_den = 0.0f;
    for (int s : node.legal_slots) {
        auto nit = node.N.find(s);
        int n = (nit != node.N.end()) ? nit->second : 0;
        if (n > 0) {
            sum_n += n;
            float p = node.prior[static_cast<size_t>(s)];               // the stored float32 prior
            // numpy `f32 * pyfloat → f32` casts the WEAK Python operand to float32 FIRST, then
            // multiplies in float32 (verified: cast-first, NOT a f64 multiply narrowed). Mirror it by
            // casting `q` to float before the multiply, so the product is computed in true float32.
            pw_num += p * static_cast<float>(node.q(s));                // f32 * f32(q) → f32
            pw_den += p;                                                // pyfloat(0) += f32 → f32
        }
    }
    if (sum_n > 0 && pw_den > 0.0f) {
        float v_bar = pw_num / pw_den;                                  // f32 / f32 → f32
        // numpy weak-promotes the Python-float `root_value` to float32 BEFORE the add (verified:
        // `pyfloat + f32` casts the weak operand to f32 first, then adds in f32 — NOT f64-then-narrow).
        float rv = static_cast<float>(root_value);
        float vmix = (rv + static_cast<float>(sum_n) * v_bar)          // f32 + (pyint*f32→f32) → f32
                     / (1.0f + static_cast<float>(sum_n));             // / (pyint) → f32
        return static_cast<double>(vmix);                             // lossless widen for the f64 add
    }
    return root_value;
}

// The masked softmax over legal slots (mirrors mlp.ValueMLP._masked_softmax): subtract the per-row
// legal max, exp, zero illegal, normalize. Inputs are slot-indexed; `legal_slots` selects the legal
// entries. Returns an (n_slots,) row, exactly 0.0 on illegal slots. Robust EXCEPT the per-row max
// argmax on a near-tie (the 1b hazard — coarse 1a inputs have no near-ties).
[[nodiscard]] std::vector<double> masked_softmax_1a(const std::vector<double>& completed,
                                                    const std::vector<int>& legal_slots,
                                                    int n_slots) {
    std::vector<double> out(static_cast<size_t>(n_slots), 0.0);
    if (legal_slots.empty()) return out;
    double row_max = -std::numeric_limits<double>::infinity();
    for (int s : legal_slots) row_max = std::max(row_max, completed[static_cast<size_t>(s)]);
    double denom = 0.0;
    for (int s : legal_slots) {
        double e = std::exp(completed[static_cast<size_t>(s)] - row_max);
        out[static_cast<size_t>(s)] = e;
        denom += e;
    }
    if (denom <= 0.0) denom = 1.0;
    for (int s : legal_slots) out[static_cast<size_t>(s)] /= denom;
    return out;
}
}  // namespace

GumbelAZPolicy::GumbelAZPolicy(const GumbelConfig& cfg, const NetEvaluator& net,
                               const Environment& env)
    : cfg_(cfg), net_(net), env_(env), fb_(env), n_slots_(n_action_slots(env)),
      term_slot_(term_slot(env)) {}

// ---- net evaluation (one forward, cached on the node) (mirrors _evaluate) --------------------------
double GumbelAZPolicy::evaluate(GumbelNode& node, const Loc& loc, const std::vector<uint32_t>& bw,
                                const std::set<int>& collected) const {
    // build the feature vector + the legal mask, run one forward through the net port (the leaf seam).
    std::vector<double> feat64 = fb_.build(loc.pt, bw, collected);
    std::vector<float> feat(feat64.begin(), feat64.end());  // the wire dtype the port consumes (float32)
    std::vector<float> mask = legal_mask(env_, bw, collected);

    auto pred = net_.predict(feat);
    // The local NetForward / the scripted leaf always return the value arm; a remote leaf's failure is
    // a typed Error. In the search the leaf is on a TOTAL path (we hold a live net), so a failure here
    // is a programmer/operator boundary fault — fail loud (ADR-0002 / P9) rather than silently degrade.
    assert(pred.has_value() && "gumbel: net leaf evaluation failed (NetEvaluator returned an Error)");
    const NetPrediction& np = *pred;

    // the prior = masked softmax of the net logits over the legal slots (mirrors predict_both: the net
    // emits raw logits, the search softmaxes them under the mask). 1b SEAM 1: `node.prior` is the float32
    // prior array the Python search side-reads (root.prior, float32) — the precision the mixed path reads.
    // We ALSO keep `node.prior_d`, the SAME masked softmax in full float64, so the discrimination control
    // (kUniform) can read the genuine pre-narrowing double prior at every site (prior_read).
    std::vector<double> logits_d(static_cast<size_t>(n_slots_), -1e30);
    // collect the legal slots (env.legal_actions order, then TERMINATE) — the SAME order Python's
    // root.legal carries (legal_actions list, with TERMINATE always legal appended by the mask).
    node.legal_slots.clear();
    std::vector<Action> legal = env_.legal_actions(bw, collected);
    for (const Action& a : legal) node.legal_slots.push_back(slot_of(env_, a));
    node.legal_slots.push_back(term_slot_);  // TERMINATE is always legal
    // build the masked-softmax prior from the net logits. The net carries n_slots logits (the policy
    // head emits over the full slot space); illegal slots are masked.
    for (int s : node.legal_slots) {
        // np.logits may be empty (value-only net) — in 1a the scripted leaf always carries logits; a
        // production value-only net would need a uniform prior, but the AZ search requires a policy head
        // (mirrors predict_both's n_actions assert), so we read the logit directly.
        assert(!np.logits.empty() && "gumbel: net has no policy head (logits empty)");
        logits_d[static_cast<size_t>(s)] = static_cast<double>(np.logits[static_cast<size_t>(s)]);
    }
    std::vector<double> prior_d = masked_softmax_1a(logits_d, node.legal_slots, n_slots_);
    // store BOTH: the full-float64 prior (prior_d, read by the uniform discrimination arm) AND its
    // float32 narrowing (prior, read by the default mixed arm — the precision Python's root.prior holds).
    node.prior_d = prior_d;
    node.prior.assign(static_cast<size_t>(n_slots_), 0.0f);
    for (int s = 0; s < n_slots_; ++s)
        node.prior[static_cast<size_t>(s)] = static_cast<float>(prior_d[static_cast<size_t>(s)]);

    node.value = static_cast<double>(np.value);
    node.evaluated = true;
    return node.value;
}

// ---- AlphaZero PUCT select (mirrors _puct_select) -------------------------------------------------
int GumbelAZPolicy::puct_select(const GumbelNode& node) const {
    int total_n = 0;
    for (const auto& kv : node.N) total_n += kv.second;
    double sqrt_total = (total_n > 0) ? std::sqrt(static_cast<double>(total_n)) : 1.0;
    double base_v = node.value;  // unvisited Q completed by the node's own net value
    int best_a = -1;
    double best_v = -std::numeric_limits<double>::infinity();
    // iterate node.legal_slots (the env-order list), strict `>` first-wins — mirrors Python's loop over
    // node.legal with `if v > best_v`, never the std::map's sorted-key order.
    for (int s : node.legal_slots) {
        auto nit = node.N.find(s);
        int n = (nit != node.N.end()) ? nit->second : 0;
        double q = (n > 0) ? (node.W.at(s) / static_cast<double>(n)) : base_v;
        // 1b SEAM (seam 4): Python computes `q + c_puct·p·√ΣN/(1+n)` with `p` the float32 prior scalar,
        // which numpy weak-promotes through the WHOLE U-term and the `q +` to float32 — so the interior
        // near-tie argmax is decided in FLOAT32 (gumbel_search.py:397-426). Mirror it: cast the Python-
        // float operands to float so each numpy op runs in true float32 (cast-first weak promotion).
        // `kUniform` runs the 1a uniform-`double` path (the discrimination control).
        double v;
        if (kUniform) {
            double p = prior_read(node, s);  // full-float64 prior (the genuine 1a all-`double` path)
            double u = cfg_.c_puct * p * sqrt_total / (1.0 + static_cast<double>(n));
            v = (kMutate == Mutate::Puct) ? (q - u) : (q + u);
        } else {
            float p = node.prior[static_cast<size_t>(s)];                       // stored float32 prior
            float u = static_cast<float>(cfg_.c_puct) * p                        // pyfloat·f32 → f32
                      * static_cast<float>(sqrt_total)                           // ·pyfloat → f32
                      / (1.0f + static_cast<float>(n));                          // /(pyint) → f32
            float vf = (kMutate == Mutate::Puct)
                           ? (static_cast<float>(q) - u)                         // q(pyfloat) ± f32 → f32
                           : (static_cast<float>(q) + u);
            v = static_cast<double>(vf);  // lossless widen; the float32 ordering drives the argmax
        }
        // strict `>` first-wins over node.legal_slots (mirrors Python `if v > best_v`). best_v starts at
        // -inf (the first slot always wins); thereafter the comparison is between the float32-rounded
        // scores (widened to double, an order-preserving widen) — the near-tie side Python lands on.
        if (v > best_v) {
            best_v = v;
            best_a = s;
        }
    }
    assert(best_a != -1 && "puct_select on a node with no legal slots");
    return best_a;
}

// ---- interior PUCT descent; net value at the leaf (mirrors _descend) ------------------------------
double GumbelAZPolicy::descend(std::vector<GumbelNode>& nodes, int node, const Loc& loc,
                               const std::vector<uint32_t>& bw, const std::set<int>& collected,
                               uint32_t world, double lam, GumbelSource& src, int depth) const {
    if (depth >= cfg_.max_depth || bw.empty()) {
        if (!nodes[static_cast<size_t>(node)].evaluated) {
            if (bw.empty()) return -lam * env_.exit_cost(loc.pt);
            return evaluate(nodes[static_cast<size_t>(node)], loc, bw, collected);
        }
        return nodes[static_cast<size_t>(node)].value;
    }
    if (!nodes[static_cast<size_t>(node)].evaluated) {
        // first visit to this leaf: the net value IS the leaf estimate (no playout — the F4 cure)
        return evaluate(nodes[static_cast<size_t>(node)], loc, bw, collected);
    }

    int a = puct_select(nodes[static_cast<size_t>(node)]);
    Action act = action_of_slot(env_, a);
    double ret;
    if (act.kind == ActionKind::Terminate) {
        ret = -lam * env_.exit_cost(loc.pt);  // stop now: only the exit toll remains
    } else {
        Loc nloc = loc;  // COPY: apply computes dt = dist(OLD loc, target) then moves nloc to target
        std::vector<uint32_t> nbw = bw;
        std::set<int> nc = collected;
        StepResult sr = env_.apply(nloc, nbw, nc, act, world);
        double step = sr.reward - lam * sr.dt;
        std::tuple<int, GBeliefKey> ckey{a, gumbel_belief_key(nbw)};
        auto& cur = nodes[static_cast<size_t>(node)];
        int child;
        auto cit = cur.children.find(ckey);
        if (cit == cur.children.end()) {
            nodes.emplace_back();
            child = static_cast<int>(nodes.size()) - 1;
            nodes[static_cast<size_t>(node)].children[ckey] = child;  // re-index after possible realloc
        } else {
            child = cit->second;
        }
        double cont = descend(nodes, child, nloc, nbw, nc, world, lam, src, depth + 1);
        ret = step + cont;
    }
    auto& cur = nodes[static_cast<size_t>(node)];
    cur.W[a] = (cur.W.count(a) ? cur.W[a] : 0.0) + ret;
    cur.N[a] = (cur.N.count(a) ? cur.N[a] : 0) + 1;
    return ret;
}

// ---- one sim of a root action (mirrors _simulate_root_action) -------------------------------------
double GumbelAZPolicy::simulate_root_action(std::vector<GumbelNode>& nodes, const Loc& loc,
                                            const std::vector<uint32_t>& bw,
                                            const std::set<int>& collected, int slot, uint32_t world,
                                            double lam, GumbelSource& src) const {
    Action a = action_of_slot(env_, slot);
    if (a.kind == ActionKind::Terminate) return -lam * env_.exit_cost(loc.pt);
    // outcome-averaging over c_outcome determinizations of the IMMEDIATE outcome (k=0 reuses the
    // threaded world; k>0 draws a fresh world from the belief — mirrors _simulate_root_action).
    double total = 0.0;
    for (int k = 0; k < cfg_.c_outcome; ++k) {
        uint32_t w = (k == 0) ? world : src.sample_world(bw);
        Loc nloc = loc;  // COPY: apply computes dt = dist(OLD loc, target) then moves nloc to target
        std::vector<uint32_t> nbw = bw;
        std::set<int> nc = collected;
        StepResult sr = env_.apply(nloc, nbw, nc, a, w);
        double step = sr.reward - lam * sr.dt;
        std::tuple<int, GBeliefKey> ckey{slot, gumbel_belief_key(nbw)};
        int child;
        auto cit = nodes[0].children.find(ckey);
        if (cit == nodes[0].children.end()) {
            nodes.emplace_back();
            child = static_cast<int>(nodes.size()) - 1;
            nodes[0].children[ckey] = child;
        } else {
            child = cit->second;
        }
        double cont = descend(nodes, child, nloc, nbw, nc, w, lam, src, 1);
        total += step + cont;
    }
    return total / static_cast<double>(cfg_.c_outcome);
}

// ---- run `count` sims of root action `slot` (mirrors _visit) --------------------------------------
void GumbelAZPolicy::visit(std::vector<GumbelNode>& nodes, const Loc& loc,
                           const std::vector<uint32_t>& bw, const std::set<int>& collected, int slot,
                           double lam, GumbelSource& src, int count) const {
    for (int i = 0; i < count; ++i) {
        uint32_t w = src.sample_world(bw);
        double ret = simulate_root_action(nodes, loc, bw, collected, slot, w, lam, src);
        nodes[0].W[slot] = (nodes[0].W.count(slot) ? nodes[0].W[slot] : 0.0) + ret;
        nodes[0].N[slot] = (nodes[0].N.count(slot) ? nodes[0].N[slot] : 0) + 1;
    }
}

// ---- Sequential Halving (Danihelka §2) (mirrors _sequential_halving) ------------------------------
int GumbelAZPolicy::sequential_halving(std::vector<GumbelNode>& nodes, const Loc& loc,
                                       const std::vector<uint32_t>& bw,
                                       const std::set<int>& collected, double lam, GumbelSource& src,
                                       std::vector<int> considered, const std::vector<double>& g,
                                       const std::vector<double>& logits, int& n_spent) const {
    n_spent = 0;
    if (considered.empty()) return -1;
    if (considered.size() == 1) {
        visit(nodes, loc, bw, collected, considered[0], lam, src, cfg_.n_sims);
        n_spent = cfg_.n_sims;
        return considered[0];
    }

    int m = static_cast<int>(considered.size());
    int n_phases = std::max(1, static_cast<int>(std::ceil(std::log2(static_cast<double>(m)))));
    int per_phase = std::max(1, cfg_.n_sims / n_phases);  // paper's N/⌈log2 m⌉ phase budget
    int budget = cfg_.n_sims;

    while (considered.size() > 1 && budget > 0) {
        int phase_budget = std::min(per_phase, budget);
        int per_action = std::max(1, phase_budget / static_cast<int>(considered.size()));
        for (int s : considered) {
            int v = std::min(per_action, budget);
            if (v <= 0) break;
            visit(nodes, loc, bw, collected, s, lam, src, v);
            budget -= v;
            n_spent += v;
        }
        // drop the worst half by g + logit + σ·q̂ (σ recomputed each phase as max_a N(a) grows).
        double sigma = sigma_scale_1a(nodes[0], cfg_.c_visit, cfg_.c_scale);
        // STABLE descending sort by the cut key — mirrors Python `sorted(considered, key=..., reverse
        // =True)`, which is stable so equal keys keep their relative (pre-sort) order. std::stable_sort
        // with a strict-`>` comparator gives the SAME order (a `>`-comparator stable sort = a reversed
        // stable ascending sort on a total order; on the coarse precision-insensitive inputs the keys
        // are well-separated so no ties arise regardless).
        std::vector<std::pair<double, int>> keyed;
        keyed.reserve(considered.size());
        for (int s : considered) {
            double key = g[static_cast<size_t>(s)] + logits[static_cast<size_t>(s)] +
                         sigma * nodes[0].q(s);
            keyed.emplace_back(key, s);
        }
        std::stable_sort(keyed.begin(), keyed.end(),
                         [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                             return a.first > b.first;  // descending; stable keeps ties in input order
                         });
        int keep = std::max(1, static_cast<int>(keyed.size()) / 2);
        std::vector<int> next;
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
        size_t i = 0;
        while (budget > 0 && !considered.empty()) {
            int s = considered[i % considered.size()];
            visit(nodes, loc, bw, collected, s, lam, src, 1);
            budget -= 1;
            n_spent += 1;
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
    std::vector<double> completed(static_cast<size_t>(n_slots_), -1e30);
    for (int s : root.legal_slots) {
        auto nit = root.N.find(s);
        int n = (nit != root.N.end()) ? nit->second : 0;
        // 1b SEAM (seam 3): `completed[s] = logits[s] + σ·q`, where `logits[s]` is a float64 root logit
        // (an element of the float64 `logits` array) and σ is a Python float. For VISITED slots q is a
        // Python float (root.q), so `σ·q` is float64 → the whole term is float64. For UNVISITED slots q
        // is `vm`, an np.float32 (v_mix's float32 return), so numpy `pyfloat(σ)·f32(vm) → f32` and then
        // `f64(logits[s]) + f32 → f64` (the float32-rounded σ·vm added in float64). Mirror EXACTLY:
        // visited → full double; unvisited → round σ·vm to float (cast-first weak promotion), then add
        // in double. `kUniform` runs the 1a uniform-`double` path (the discrimination control).
        if (n > 0) {
            completed[static_cast<size_t>(s)] = logits[static_cast<size_t>(s)] + sigma * root.q(s);
        } else if (kUniform) {
            completed[static_cast<size_t>(s)] = logits[static_cast<size_t>(s)] + sigma * vm;
        } else {
            float sigma_vm = static_cast<float>(sigma) * static_cast<float>(vm);  // pyfloat·f32 → f32
            completed[static_cast<size_t>(s)] =
                logits[static_cast<size_t>(s)] + static_cast<double>(sigma_vm);   // f64 + f32 → f64
        }
    }
    return masked_softmax_1a(completed, root.legal_slots, n_slots_);
}

// ---- the pure search core (mirrors _decide_root, temperature 0) -----------------------------------
GumbelAZPolicy::Decision GumbelAZPolicy::run_search(const Loc& loc, const std::vector<uint32_t>& bw,
                                                    const std::set<int>& collected, double lam,
                                                    GumbelSource& src) const {
    Decision out;
    // empty-belief guard (mirrors decide_with_target's len(bw)==0): the only continuation is to exit.
    if (bw.empty()) {
        out.action = terminate_action();
        out.improved.assign(static_cast<size_t>(n_slots_), 0.0);
        out.improved[static_cast<size_t>(term_slot_)] = 1.0;
        out.survivor_slot = term_slot_;
        out.n_spent = 0;
        return out;
    }

    std::vector<GumbelNode> nodes;
    nodes.emplace_back();  // the root (arena index 0)
    evaluate(nodes[0], loc, bw, collected);

    // root logits = log(prior) over legal slots (the masked-softmax prior is the reference; its log is
    // the root logit). -1e30 on illegal (mirrors _decide_root's logits build).
    // 1b SEAM 1 (the DOMINANT float32 effect on the discrete output): Python reads `prior[s]` off the
    // float32 `root.prior` here, so `math.log(max(prior[s], 1e-12))` is a log of a float32 scalar — the
    // ~1e-7 float32 narrowing perturbs every root logit, and these logits feed BOTH the Gumbel-top-k
    // (logit+g) AND the SH cut key (g+logit+σ·q̂), so the float32-vs-double prior FLIPS the survivor and
    // the improved-π argmax on near-tie inputs. `prior_read` routes the precision: float32 in the
    // default mixed arm (matching Python), full-float64 in the uniform discrimination arm.
    std::vector<double> logits(static_cast<size_t>(n_slots_), -1e30);
    for (int s : nodes[0].legal_slots) {
        double p = prior_read(nodes[0], s);
        logits[static_cast<size_t>(s)] = std::log(std::max(p, 1e-12));
    }

    // Gumbel-Top-k: one gumbel draw over the FULL slot space, sort logits+g, take top-m legal slots.
    std::vector<double> g = src.gumbel(n_slots_);
    // score0 = where(logits > -1e29, logits+g, -inf) (mirrors _decide_root). Build over legal slots
    // only (illegal stay -inf), then take the top m = min(self.m, #legal).
    std::vector<std::pair<double, int>> scored;  // (score, slot) over legal slots
    scored.reserve(nodes[0].legal_slots.size());
    for (int s : nodes[0].legal_slots) {
        scored.emplace_back(logits[static_cast<size_t>(s)] + g[static_cast<size_t>(s)], s);
    }
    // top-m by descending score (mirrors np.argsort(score0)[::-1][:m]). On the coarse precision-
    // insensitive inputs the gumbel-perturbed scores are well-separated (no ties); a stable descending
    // sort fixes the order deterministically (Python's argsort is value-ordered for distinct scores).
    std::stable_sort(scored.begin(), scored.end(),
                     [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                         return a.first > b.first;
                     });
    int m = std::min(cfg_.m, static_cast<int>(nodes[0].legal_slots.size()));
    std::vector<int> considered;
    considered.reserve(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i) considered.push_back(scored[static_cast<size_t>(i)].second);

    // Sequential Halving over n_sims; returns the surviving slot (the executed action at temperature 0).
    int survivor = sequential_halving(nodes, loc, bw, collected, lam, src, considered, g, logits,
                                      out.n_spent);

    // the improved-π target over the FULL legal set.
    out.improved = improved_policy(nodes[0], logits);

    // executed action = the SH survivor (Danihelka §2; temperature 0). The temperature>0 sampling path
    // is a 1b/production concern; 1a's logic check is temperature 0 (the eval policy's rule).
    assert(survivor != -1 && "gumbel: SH returned no survivor on a non-empty belief");
    out.survivor_slot = survivor;
    out.action = action_of_slot(env_, survivor);
    return out;
}

namespace {
// The production Gumbel source: the generic uniform sample_world (reused from the shared
// RngWorldSource — ADR-0012 P1) + a real gumbel draw off the same std::mt19937_64. RNG note (P6):
// std::mt19937_64 / the std gumbel transform do NOT match numpy's stream, so production parity is the
// BEHAVIORAL bar; the discrete logic is validated RNG-free by the scripted source in gumbel_dump.cpp.
class RngGumbelSource final : public GumbelSource {
  public:
    explicit RngGumbelSource(std::mt19937_64& rng) : draw_(rng), rng_(rng) {}

    uint32_t sample_world(const std::vector<uint32_t>& bw) override { return draw_.sample_world(bw); }

    std::vector<double> gumbel(int n) override {
        // Gumbel(0,1) via the inverse-CDF transform -log(-log(U)), U in (0,1) (mirrors numpy's gumbel
        // family, NOT its exact stream — the behavioral bar). U is drawn off (0,1) open to avoid log(0).
        std::vector<double> out(static_cast<size_t>(n));
        std::uniform_real_distribution<double> unif(
            std::numeric_limits<double>::min(), 1.0);  // (0,1], min() avoids log(0)
        for (int i = 0; i < n; ++i) {
            double u = unif(rng_);
            out[static_cast<size_t>(i)] = -std::log(-std::log(u));
        }
        return out;
    }

  private:
    RngWorldSource draw_;   // the shared generic uniform-from-belief draw (ADR-0012 P1)
    std::mt19937_64& rng_;  // the SAME stream the draw uses, for the gumbel draw
};
}  // namespace

Action GumbelAZPolicy::decide(const Environment& env, const Loc& loc, const std::vector<uint32_t>& bw,
                              const std::set<int>& collected, double lam,
                              std::mt19937_64& rng) const {
    (void)env;  // the policy holds its own env_ ref (the seam passes env for the contract; same object)
    RngGumbelSource src(rng);
    return run_search(loc, bw, collected, lam, src).action;
}

}  // namespace chocofarm
