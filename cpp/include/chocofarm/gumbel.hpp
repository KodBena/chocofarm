// cpp/include/chocofarm/gumbel.hpp
// Purpose: the C++ Gumbel-AlphaZero search Policy — the discrete search STRUCTURE of
//   chocofarm/az/gumbel_search.py (Danihelka et al. 2022) ported behind the composable env<->Policy
//   seam (ADR-0012 P2/P7: derive from the ONE Python authority, reimplement, behavioral parity NOT
//   byte-identity). A drop-in `Policy` alongside RandomPolicy / NMCSPolicy / ISMCTSPolicy: the runner
//   takes `const Policy&` and never names this class, so adding it is ZERO edits to the search/env core.
//
//   *** PHASE 1a SCOPE (structure) ***
//   The port mirrors the DISCRETE ALGORITHM: Gumbel-Top-k root sampling, Sequential Halving (the
//   per-phase budget accounting + the full-budget remainder loop), PUCT descent, the _Node W/N arena,
//   the c_outcome outcome-averaging, the improved-π σ-transform, and the executed action = the SH
//   survivor (temperature 0). 1a validated this structure exact-action on COARSE, well-separated
//   scripted leaf inputs (NO near-ties), so the discrete outcome was identical float32-vs-float64 —
//   proving the SELECTION LOGIC is faithful WITHOUT yet chasing the precision.
//
//   *** PHASE 1b (DONE — the precision FIDELITY) ***
//   The Python search runs the σ-transform at a DELIBERATE float32-prior × float64-Q mixed precision
//   (value_target.py:209-249, the byte-identity seam) that a uniform-precision port diverges from on
//   NEAR-TIE inputs. 1b makes the C++ reproduce that promotion EXACTLY, localized in gumbel.cpp at four
//   spots (each toggleable by CHOCO_GUMBEL_UNIFORM, the discrimination control):
//     1. `evaluate` stores `node.prior` as float32 (the in-search masked-softmax prior; the softmax
//        that BUILDS it stays float64, only the stored prior is narrowed);
//     2. `v_mix_mixed` computes the prior-weighted blend ENTIRELY in float32 (numpy's f32×pyfloat weak
//        promotion — the v_mix RETURN is np.float32);
//     3. `improved_policy` completes UNVISITED slots with σ·v_mix rounded to float32 (numpy
//        pyfloat·f32→f32, added to the float64 root logit), VISITED slots in full float64;
//     4. `puct_select` scores `q + c_puct·p·√ΣN/(1+n)` in float32 (the float32 prior weak-promotes the
//        U-term), deciding the interior near-tie argmax at float32.
//   The SH cut key (g+logit+σ·q̂) and the Gumbel-top-k (logit+g) are float64 on BOTH sides (numpy:
//   g/logits/sigma float64, q̂ a Python float) — no float32 there. cpp/parity/gumbel_precision.py proves
//   the FINE near-tie parity (mixed N/N exact, uniform diverges X/N — the load-bearing discrimination).
//
//   The leaf goes through the injected NetEvaluator port (net_evaluator.hpp / design §1): the search
//   is its first real consumer. The production decide() wires a NetForward (or a remote ZmqNetClient);
//   the DETERMINISTIC logic check (cpp/parity/gumbel_logic.py) injects a SCRIPTED NetEvaluator (canned
//   coarse (value, logits) per call, RNG-free) + a scripted GumbelSource, exactly as ISMCTS's
//   logic check scripts its leaf + RNG draws.
//
//   DRY against the shared base (ADR-0012 P1): the world-sampling draw (the shared WorldSource
//   sample_world) lives in policy.{hpp,cpp} and is REUSED. This unit does NOT include nmcs.hpp /
//   ismcts.hpp; it shares only policy.hpp (the base) + net_evaluator.hpp (the leaf port) +
//   features.hpp (the slot bijection + the feature/mask builder). Gumbel's own pieces are the
//   info-set _Node (W/N over actions, children keyed by (action, belief_key)), the Gumbel-Top-k root
//   sampling, the Sequential-Halving bracket, the PUCT interior select, and the improved-π σ-transform.
//
//   RNG note (ADR-0012 P6): std::mt19937_64 / the gumbel draw do NOT match numpy's stream, so
//   production parity on the RNG-driven aggregates is the BEHAVIORAL bar. The discrete SELECTION logic
//   is validated RNG-free: BOTH the gumbel draw AND the world sampling route through the injectable
//   GumbelSource so the logic check feeds both languages identical scripted sequences and asserts the
//   SAME executed action + improved-π argmax.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/net_evaluator.hpp"
#include "chocofarm/policy.hpp"

namespace chocofarm {

// The frozen scalar hyperparameters (mirrors GumbelAZSearch.__init__ defaults). The net (a
// NetEvaluator, not a scalar) stays a separate construction param, exactly as Python keeps the net
// out of the scalar config. Defaults match the Python defaults: m=12, n_sims=48, c_puct=1.25,
// c_visit=50, c_scale=1.0, c_outcome=2, max_depth=24.
struct GumbelConfig {
    int m = 12;            // root actions sampled by Gumbel-Top-k
    int n_sims = 48;       // simulations spent by Sequential Halving (the full budget)
    double c_puct = 1.25;  // PUCT exploration constant
    double c_visit = 50.0; // the Danihelka σ-transform visit prefactor (c_visit + max_a N(a))
    double c_scale = 1.0;  // the Danihelka σ-transform scale
    int c_outcome = 2;     // immediate-outcome determinizations averaged per root-action sim
    int max_depth = 24;    // interior descent depth cap
};

// The information-set node identity: the (count, first, last) belief fingerprint (mirrors
// gumbel_search's _belief_key = (len, bw[0], bw[-1]) — the SAME triple ISMCTS uses, kept local here so
// this unit does not include ismcts.hpp). Beliefs reached by the same observations are the same set of
// worlds regardless of path; this triple fingerprints the modest number of distinct beliefs one search
// reaches.
using GBeliefKey = std::tuple<int, uint32_t, uint32_t>;
[[nodiscard]] GBeliefKey gumbel_belief_key(const std::vector<uint32_t>& bw);

// One information-set node (a belief). Per-action aggregate W (summed λ-penalized return) / N
// (selection count) over the info set (the ISMCTS/F7 contract), children keyed by (action-slot,
// belief_key). prior/value/legal are the net's cached evaluation at this belief (one forward, reused
// across the node's action loop), populated lazily by `evaluate` (mirrors _Node + _evaluate). The
// maps are keyed by action SLOT (a faithful stand-in for the Python Action-tuple keys; the mapping is
// a bijection); the legal-slot order is tracked in `legal_slots` so the PUCT scan + the improved-π are
// over the SAME deterministic order the Python `node.legal` list carries (insertion / id order).
struct GumbelNode {
    bool evaluated = false;                 // has `evaluate` populated this node?
    double value = 0.0;                     // scalar net value V at this belief
    std::vector<float> prior;               // (n_slots,) masked-softmax prior P(s,·) — FLOAT32 (1b seam 1:
                                            //   the in-search prior the Python search side-reads as
                                            //   root.prior, a float32 array; softmaxed in f64, stored f32)
    std::vector<double> prior_d;            // (n_slots,) the SAME masked-softmax prior in FULL float64 —
                                            //   the pre-narrowing double prior. The DISCRIMINATION control
                                            //   (kUniform) reads THIS at every site so the uniform arm is
                                            //   the genuine 1a all-float64 port; the mixed (default) arm
                                            //   reads the float32 `prior`. See gumbel.cpp seam map.
    std::vector<int> legal_slots;           // legal action slots, in env.legal_actions + TERMINATE order
    std::map<int, double> W;                // action-slot -> summed λ-penalized return
    std::map<int, int> N;                   // action-slot -> selection count
    std::map<std::tuple<int, GBeliefKey>, int> children;  // (action-slot, belief_key) -> child arena idx

    // Q(slot) = W/N, or 0.0 unvisited (mirrors _Node.q).
    [[nodiscard]] double q(int slot) const {
        auto it = N.find(slot);
        if (it == N.end() || it->second == 0) return 0.0;
        return W.at(slot) / static_cast<double>(it->second);
    }
};

// The Gumbel search's RNG seam. Two draws route through it so the deterministic logic check can script
// both RNG-free across languages: (a) `gumbel(n)` — one i.i.d. Gumbel draw per slot over the FULL slot
// space at the root (mirrors rng.gumbel(size=n_slots)); (b) the world sampling (inherited from the
// shared WorldSource sample_world — mirrors env.sample_world). Production wires the real RNG; the logic
// check wires a scripted, RNG-free source.
struct GumbelSource : public WorldSource {
    // One i.i.d. Gumbel(0,1) draw per slot, length `n` (mirrors rng.gumbel(size=n_slots)). Returned by
    // value (a length-n vector). The logic check returns a fixed scripted vector here.
    [[nodiscard]] virtual std::vector<double> gumbel(int n) = 0;
};

// Gumbel-AlphaZero search as a pluggable Policy. Construction takes the scalar config + the injected
// NetEvaluator leaf (the net port — design §1). It mirrors GumbelPolicy.decide (the eval wrapper):
// decide returns the EXECUTED action = the SH survivor at temperature 0.
class GumbelAZPolicy final : public Policy {
  public:
    GumbelAZPolicy(const GumbelConfig& cfg, const NetEvaluator& net, const Environment& env);

    // The Policy contract. Builds the production GumbelSource off `rng` + the injected net leaf and
    // runs the search from the current observed state, returning the executed action (the SH survivor,
    // temperature 0). λ is the live Dinkelbach penalty threaded through every score (P4).
    [[nodiscard]] Action decide(const Environment& env, const Loc& loc,
                                const std::vector<uint32_t>& bw, const std::set<int>& collected,
                                double lam, std::mt19937_64& rng) const override;

    // The pure search core, parameterized by an injected GumbelSource (the seam the logic check
    // exploits). Runs the full Gumbel search from a FIXED (loc, belief, collected) and returns
    // (executed_action, improved_pi). Mirrors GumbelAZSearch._decide_root (temperature 0). Exposed for
    // the logic-check fixture so the structure + selection logic is validated independent of RNG.
    struct Decision {
        Action action;                 // the executed action (the SH survivor at temperature 0)
        std::vector<double> improved;  // (n_slots,) improved-π target (softmax of completed logits)
        int n_spent = 0;               // total root-action sims spent (= n_sims when bw non-empty)
        int survivor_slot = -1;        // the SH survivor slot (the executed action's slot)
    };
    [[nodiscard]] Decision run_search(const Loc& loc, const std::vector<uint32_t>& bw,
                                      const std::set<int>& collected, double lam,
                                      GumbelSource& src) const;

    // Like decide(), but returns the FULL Decision (the executed action + the improved-π target +
    // n_spent) — the AZ actor's per-decision record (mirrors the Python GumbelPolicy.decide_with_target).
    // It builds the production RngGumbelSource off `rng` and runs the search, exactly as decide() does;
    // decide() is this composed with `.action`. The runtime / the runner use this to capture the
    // improved-π PI target the trainer consumes.
    [[nodiscard]] Decision decide_with_target(const Environment& env, const Loc& loc,
                                              const std::vector<uint32_t>& bw,
                                              const std::set<int>& collected, double lam,
                                              std::mt19937_64& rng) const;

    [[nodiscard]] const GumbelConfig& config() const { return cfg_; }

  private:
    // Populate node.value/prior/legal_slots from one net forward (mirrors _evaluate). The leaf goes
    // through the injected NetEvaluator: it returns (value, logits); the prior is the masked softmax of
    // logits over the legal slots (mirrors predict_both). 1b SEAM 1: the masked softmax runs in float64
    // (mlp._masked_softmax), then the STORED prior is narrowed to float32 — the precision the Python
    // search side-reads (root.prior). `kUniform` widens it back at every read site (discrimination ctl).
    double evaluate(GumbelNode& node, const Loc& loc, const std::vector<uint32_t>& bw,
                    const std::set<int>& collected) const;

    // Sequential Halving over n_sims (Danihelka §2): n_phases = ceil(log2 m), per-phase equal-share
    // budget, drop the worst half each phase by g+logit+σ·q̂, then a remainder loop spends the FULL
    // budget. Returns the surviving slot (the executed action). Mirrors _sequential_halving.
    [[nodiscard]] int sequential_halving(std::vector<GumbelNode>& nodes, const Loc& loc,
                                         const std::vector<uint32_t>& bw, const std::set<int>& collected,
                                         double lam, GumbelSource& src, std::vector<int> considered,
                                         const std::vector<double>& g, const std::vector<double>& logits,
                                         int& n_spent) const;

    // Run `count` sims of root action `slot`, accumulating W/N (mirrors _visit).
    void visit(std::vector<GumbelNode>& nodes, const Loc& loc, const std::vector<uint32_t>& bw,
               const std::set<int>& collected, int slot, double lam, GumbelSource& src,
               int count) const;

    // One sim of a root action: realize it, average the leaf over c_outcome immediate determinizations,
    // descend the interior with PUCT for the remaining depth (mirrors _simulate_root_action). Returns
    // the λ-penalized return.
    [[nodiscard]] double simulate_root_action(std::vector<GumbelNode>& nodes, const Loc& loc,
                                              const std::vector<uint32_t>& bw,
                                              const std::set<int>& collected, int slot, uint32_t world,
                                              double lam, GumbelSource& src) const;

    // Interior PUCT descent; net value at the leaf (mirrors _descend). `node` is an arena index.
    [[nodiscard]] double descend(std::vector<GumbelNode>& nodes, int node, const Loc& loc,
                                 const std::vector<uint32_t>& bw, const std::set<int>& collected,
                                 uint32_t world, double lam, GumbelSource& src, int depth) const;

    // AlphaZero PUCT select: argmax q + c_puct·p·√(ΣN)/(1+n) over node.legal_slots, strict-`>`
    // first-wins, unvisited Q completed by the node's own net value (mirrors _puct_select). Returns the
    // selected action slot. 1b SEAM 4: the score is computed in FLOAT32 (the float32 prior weak-promotes
    // the whole U-term + the `q +`), so the interior near-tie argmax matches Python; `kUniform` runs it
    // in `double` (discrimination control).
    [[nodiscard]] int puct_select(const GumbelNode& node) const;

    // The improved-π target over the legal set: π′ = softmax(logit + σ(completedQ)) (mirrors
    // _improved_policy → value_target.improved_policy). 1b SEAMS 2+3: v_mix (the unvisited completion)
    // runs the float32 prior-weighted blend, and the unvisited slots' σ·v_mix is rounded to float32 (the
    // visited slots use full-float64 σ·q); the masked softmax over the completed logits then runs in
    // float64. `kUniform` reverts both to `double`. Returns an (n_slots,) row (0 on illegal).
    [[nodiscard]] std::vector<double> improved_policy(const GumbelNode& root,
                                                      const std::vector<double>& logits) const;

    GumbelConfig cfg_;
    const NetEvaluator& net_;
    const Environment& env_;
    FeatureBuilder fb_;
    int n_slots_;
    int term_slot_;
};

}  // namespace chocofarm
