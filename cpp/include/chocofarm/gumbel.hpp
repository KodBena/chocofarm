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

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "chocofarm/belief_key.hpp"
#include "chocofarm/collected_set.hpp"
#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/net_evaluator.hpp"
#include "chocofarm/policy.hpp"
#include "chocofarm/releasing_arena.hpp"  // MmapUpstream — the OS-releasing arena overflow (ADR-0000 fix)

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
    // HPO/BENCHMARK-ONLY (default false → production search byte-unchanged). When true, the search runs
    // EXACTLY as normal (Gumbel-Top-k, Sequential Halving, the early-exit Terminate edge SAMPLED and
    // BACKPROPPED correctly) — but if the chosen EXECUTED action is Terminate AND a non-terminate legal
    // action still exists (places still to visit), the executed action is substituted for the best
    // non-terminate option so the episode CONTINUES. This keeps benchmark episodes on a faithful,
    // non-early-exiting trajectory (the warm-pool/HPO population must span the real belief distribution,
    // which early-exit otherwise truncates) while exercising the real search code paths + distributions.
    // The forced-terminate-on-empty-belief case (no non-terminate option) is NOT substituted. Behavior
    // implemented in gumbel.cpp's decide core (the executed-action substitution point).
    bool no_early_exit = false;
};

// The information-set node identity: the (count, first, last) belief fingerprint (mirrors
// gumbel_search's _belief_key = (len, bw[0], bw[-1]) — the SAME triple ISMCTS uses, kept local here so
// this unit does not include ismcts.hpp). Beliefs reached by the same observations are the same set of
// worlds regardless of path; this triple fingerprints the modest number of distinct beliefs one search
// reaches.
using GBeliefKey = BeliefKey;  // the ONE fingerprint (belief_key.hpp), now shared with FeatureBuilder's memo
// The fingerprint now lives on the env (Environment::belief_key, the seam owns the read of the belief's
// representation — L2); this thin helper delegates so the node-cache key build reads one call.
[[nodiscard]] inline GBeliefKey gumbel_belief_key(const Environment& env, const Belief& bw) {
    return env.belief_key(bw);
}

// Hash for the children transposition key (action-slot, belief_key) = tuple<int, tuple<int,u32,u32>>.
// `children` is a find/insert-only TRANSPOSITION TABLE (never iterated in key order — descend /
// simulate_root_action only `find` then insert), so swapping std::map -> std::unordered_map is bit-exact:
// the node graph is unchanged, only the lookup container differs (no ordered traversal anywhere). The mix
// is the boost hash_combine recurrence (h ^= v*0x9e3779b9 + (h<<6) + (h>>2)) folded over the FOUR scalar
// fields (the outer slot + the inner count/first/last) — NOT a bare XOR-fold (which would let (a,b) and
// (b,a) cancel); the shift+golden-ratio mixing breaks that symmetry, so distinct keys hash distinctly with
// the usual avalanche. Correctness rests on operator== of the tuple (the table compares keys exactly on a
// bucket collision), not on the hash being injective.
struct GBeliefChildKeyHash {
    [[nodiscard]] std::size_t operator()(const std::tuple<int, GBeliefKey>& k) const noexcept {
        auto mix = [](std::size_t& h, std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };
        std::size_t h = 0;
        const GBeliefKey& bk = std::get<1>(k);
        mix(h, static_cast<std::size_t>(static_cast<uint32_t>(std::get<0>(k))));   // action slot
        mix(h, static_cast<std::size_t>(static_cast<uint32_t>(std::get<0>(bk))));  // belief count
        mix(h, static_cast<std::size_t>(std::get<1>(bk)));                         // first world id
        mix(h, static_cast<std::size_t>(std::get<2>(bk)));                         // last world id
        return h;
    }
};

// One information-set node (a belief). Per-action aggregate W (summed λ-penalized return) / N
// (selection count) over the info set (the ISMCTS/F7 contract), children keyed by (action-slot,
// belief_key). prior/value/legal are the net's cached evaluation at this belief (one forward, reused
// across the node's action loop), populated lazily by `evaluate` (mirrors _Node + _evaluate). W/N are
// DENSE per-slot vectors (sized n_slots, zero-initialized at node creation), indexed by action SLOT — a
// faithful stand-in for the Python Action-tuple keys (the mapping is a bijection). They REPLACE the former
// std::map<int,.> (the per-action maps whose std::_Rb_tree_increment dominated the K=128 profile at ~10%):
// the search only ever LOOKS UP W/N by slot (over node.legal_slots, in env order) and SUMS N (order-
// independent), never iterates them in key order, so a dense vector is byte-identical (unvisited slot ->
// N[slot]==0 -> q==0.0, the SAME unvisited-completion the map's `find==end` gave; total_n sums all slots,
// and unvisited entries are 0 so the sum equals the map's visited-only sum). `legal_slots` tracks the env
// order the PUCT scan + the improved-π iterate (the Python `node.legal` list order).
// ALLOCATION (ADR-0012 P9 rule 4, MEASURED — ADR-0009): the node pool is rebuilt fresh PER DECISION
// and each GumbelNode owns FIVE std::vector + a std::unordered_map, with O(tree-size) nodes per
// decision all freed at decision end. The K=64 e2e wire profile (perf report on chocofarm-cpp-r:
// _int_malloc 4.06%, unlink_chunk 1.03%, _int_free ~1.5%, __memmove_avx 0.87% — ~6% of runner
// self-cycles) placed that allocator traffic in the SEARCH DESCENT here, NOT at the per-leaf feature
// path (already heap-reuse-clean via ws_) nor the per-decision feat/mask/pi vectors (std::move'd into
// EpisodeBuilder storage — >1-owner, no arena). So the node pool's per-decision malloc/free churn is
// the measured bucket, and it is served from a per-policy std::pmr::monotonic_buffer_resource (a typed
// pmr arena, geometric 2x upstream growth) RESET via release() per decision. To route the node's own
// inner containers through that arena the node is ALLOCATOR-AWARE (uses-allocator construction): its
// pmr containers and the allocator-extended ctors below let std::pmr::vector<GumbelNode>::emplace_back
// PROPAGATE the arena's allocator into each node's W/N/legal_slots/prior(_d)/children. The CONTENTS and
// the find/insert-only children operations are UNCHANGED — only the allocator differs, so the node
// graph and every lookup are byte-identical to the std::vector/std::unordered_map form (P6: an
// allocator swap perturbs no value, and children is read by .find() only, never iterated per decision).
struct GumbelNode {
    using allocator_type = std::pmr::polymorphic_allocator<std::byte>;

    bool evaluated = false;                 // has `evaluate` populated this node?
    double value = 0.0;                     // scalar net value V at this belief
    std::pmr::vector<float> prior;          // (n_slots,) masked-softmax prior P(s,·) — FLOAT32 (1b seam 1:
                                            //   the in-search prior the Python search side-reads as
                                            //   root.prior, a float32 array; softmaxed in f64, stored f32)
    std::pmr::vector<double> prior_d;       // (n_slots,) the SAME masked-softmax prior in FULL float64 —
                                            //   the pre-narrowing double prior. The DISCRIMINATION control
                                            //   (kUniform) reads THIS at every site so the uniform arm is
                                            //   the genuine 1a all-float64 port; the mixed (default) arm
                                            //   reads the float32 `prior`. See gumbel.cpp seam map.
    std::pmr::vector<int> legal_slots;      // legal action slots, in env.legal_actions + TERMINATE order
    std::pmr::vector<double> W;             // (n_slots,) action-slot -> summed λ-penalized return (0 unvisited)
    std::pmr::vector<int> N;                // (n_slots,) action-slot -> selection count (0 unvisited)
    // (action-slot, belief_key) -> child arena idx. A find/insert-only transposition table (never iterated
    // in key order), so a pmr::unordered_map is bit-exact with the former std::unordered_map / the older
    // std::map — same GBeliefChildKeyHash + tuple operator==, only the allocator differs (GBeliefChildKeyHash).
    std::pmr::unordered_map<std::tuple<int, GBeliefKey>, int, GBeliefChildKeyHash> children;

    // The allocator-aware ctors (uses-allocator construction): the no-arg + n_slots forms forward an
    // allocator (the pmr arena, supplied by std::pmr::vector<GumbelNode>::emplace_back) to EVERY inner
    // pmr container, so the node's storage is served from the per-policy monotonic_buffer_resource. The
    // n_slots ctor still sizes W/N to the slot space zero-initialized (the unvisited semantics N[slot]==0).
    explicit GumbelNode(const allocator_type& alloc = {})
        : prior(alloc), prior_d(alloc), legal_slots(alloc), W(alloc), N(alloc), children(alloc) {}
    GumbelNode(int n_slots, const allocator_type& alloc)
        : prior(alloc), prior_d(alloc), legal_slots(alloc),
          W(static_cast<size_t>(n_slots), 0.0, alloc), N(static_cast<size_t>(n_slots), 0, alloc),
          children(alloc) {}
    // The allocator-extended copy/move ctors uses-allocator construction REQUIRES so the NodePool can
    // grow (it move-constructs the existing nodes into the new arena block, propagating THIS arena's
    // allocator to each rebuilt inner container — without these, std::vector<GumbelNode>::emplace_back's
    // realloc cannot satisfy uses_allocator). Each inner container's (value, alloc) ctor copies/moves the
    // contents byte-identically and binds it to `alloc`. The non-allocator copy/move stay defaulted.
    GumbelNode(const GumbelNode& o, const allocator_type& alloc)
        : evaluated(o.evaluated), value(o.value), prior(o.prior, alloc), prior_d(o.prior_d, alloc),
          legal_slots(o.legal_slots, alloc), W(o.W, alloc), N(o.N, alloc), children(o.children, alloc) {}
    GumbelNode(GumbelNode&& o, const allocator_type& alloc)
        : evaluated(o.evaluated), value(o.value), prior(std::move(o.prior), alloc),
          prior_d(std::move(o.prior_d), alloc), legal_slots(std::move(o.legal_slots), alloc),
          W(std::move(o.W), alloc), N(std::move(o.N), alloc), children(std::move(o.children), alloc) {}
    GumbelNode(const GumbelNode&) = default;
    GumbelNode(GumbelNode&&) = default;
    GumbelNode& operator=(const GumbelNode&) = default;
    GumbelNode& operator=(GumbelNode&&) = default;

    // Q(slot) = W/N, or 0.0 unvisited (mirrors _Node.q). Dense: N[slot]==0 <=> unvisited (the former
    // `N.find==end` branch), so the unvisited->0.0 semantics is preserved exactly.
    [[nodiscard]] double q(int slot) const {
        const int n = N[static_cast<size_t>(slot)];
        if (n == 0) return 0.0;
        return W[static_cast<size_t>(slot)] / static_cast<double>(n);
    }
};

// The per-decision node pool (the search-tree arena). A std::pmr::vector<GumbelNode> so its element
// storage AND each node's inner containers draw from the SAME per-policy monotonic_buffer_resource;
// the resource is release()'d per decision (run_search), so the whole tree's allocations are recycled
// from the up-front buffer instead of churning the global allocator per decision (ADR-0012 P9 rule 4).
using NodePool = std::pmr::vector<GumbelNode>;

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

// The PRODUCTION Gumbel source (the ONE home, ADR-0012 P1): the generic uniform sample_world (reused
// from the shared RngWorldSource) + a real gumbel draw off the SAME std::mt19937_64. RNG note (P6):
// std::mt19937_64 / the std gumbel transform do NOT match numpy's stream, so production parity is the
// BEHAVIORAL bar; the discrete logic is validated RNG-free by the scripted source in gumbel_dump.cpp.
//
// It is declared HERE (promoted from gumbel.cpp's anonymous namespace) so the LOCAL batched driver's
// per-slot TreeState can host the SAME source decide_with_target builds — byte-identical RNG draw order
// across the serial and batched paths (the §1-tension-#4 per-slot-RNG-isolation invariant of
// docs/design/cpp-local-batched-runtime.md rests on the draw order being identical per tree). Header-
// only (its body uses only RngWorldSource + the std gumbel transform), so there is exactly one
// definition both gumbel.cpp's decide_with_target and the batched driver construct.
class RngGumbelSource final : public GumbelSource {
  public:
    RngGumbelSource(const Environment& env, std::mt19937_64& rng) : draw_(env, rng), rng_(rng) {}

    uint32_t sample_world(const Belief& bw) override { return draw_.sample_world(bw); }

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
                                const Belief& bw, const CollectedSet& collected,
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
    [[nodiscard]] Decision run_search(const Loc& loc, const Belief& bw,
                                      const CollectedSet& collected, double lam,
                                      GumbelSource& src) const;

    // Like decide(), but returns the FULL Decision (the executed action + the improved-π target +
    // n_spent) — the AZ actor's per-decision record (mirrors the Python GumbelPolicy.decide_with_target).
    // It builds the production RngGumbelSource off `rng` and runs the search, exactly as decide() does;
    // decide() is this composed with `.action`. The runtime / the runner use this to capture the
    // improved-π PI target the trainer consumes.
    [[nodiscard]] Decision decide_with_target(const Environment& env, const Loc& loc,
                                              const Belief& bw,
                                              const CollectedSet& collected, double lam,
                                              std::mt19937_64& rng) const;

    // The Policy::decide_target override (the AZ runner's PI source): the executed action + the Gumbel
    // search's REAL improved-π (not the search-free uniform default), narrowed to float32. One search,
    // via decide_with_target. This is what makes the C++ Gumbel actor emit a correct AZ PI target.
    [[nodiscard]] ActionAndPi decide_target(const Environment& env, const Loc& loc,
                                            const Belief& bw,
                                            const CollectedSet& collected, double lam,
                                            std::mt19937_64& rng) const override;

    [[nodiscard]] const GumbelConfig& config() const { return cfg_; }

  private:
    // Populate node.value/prior/legal_slots from one net forward (mirrors _evaluate). The leaf goes
    // through the injected NetEvaluator: it returns (value, logits); the prior is the masked softmax of
    // logits over the legal slots (mirrors predict_both). 1b SEAM 1: the masked softmax runs in float64
    // (mlp._masked_softmax), then the STORED prior is narrowed to float32 — the precision the Python
    // search side-reads (root.prior). `kUniform` widens it back at every read site (discrimination ctl).
    double evaluate(GumbelNode& node, const Loc& loc, const Belief& bw,
                    const CollectedSet& collected) const;

    // Sequential Halving over n_sims (Danihelka §2): n_phases = ceil(log2 m), per-phase equal-share
    // budget, drop the worst half each phase by g+logit+σ·q̂, then a remainder loop spends the FULL
    // budget. Returns the surviving slot (the executed action). Mirrors _sequential_halving.
    [[nodiscard]] int sequential_halving(NodePool& nodes, const Loc& loc,
                                         const Belief& bw, const CollectedSet& collected,
                                         double lam, GumbelSource& src, std::vector<int> considered,
                                         const std::vector<double>& g, const std::vector<double>& logits,
                                         int& n_spent) const;

    // Run `count` sims of root action `slot`, accumulating W/N (mirrors _visit).
    void visit(NodePool& nodes, const Loc& loc, const Belief& bw,
               const CollectedSet& collected, int slot, double lam, GumbelSource& src,
               int count) const;

    // One sim of a root action: realize it, average the leaf over c_outcome immediate determinizations,
    // descend the interior with PUCT for the remaining depth (mirrors _simulate_root_action). Returns
    // the λ-penalized return.
    [[nodiscard]] double simulate_root_action(NodePool& nodes, const Loc& loc,
                                              const Belief& bw,
                                              const CollectedSet& collected, int slot, uint32_t world,
                                              double lam, GumbelSource& src) const;

    // Interior PUCT descent; net value at the leaf (mirrors _descend). `node` is an arena index.
    [[nodiscard]] double descend(NodePool& nodes, int node, const Loc& loc,
                                 const Belief& bw, const CollectedSet& collected,
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

    // The per-leaf evaluate() scratch (ADR-0012 P9 hot-path exception): evaluate() reuses these buffers
    // across THIS policy's leaves — the feature triple via fb_.build_into / fb_.legal_mask_into, plus the
    // logits_d / prior_scratch the masked-softmax prior build now writes into — instead of allocating fresh
    // vectors per leaf. MEASURED (honest, ADR-0009): a before/after K=64 wire profile showed the FEATURE
    // triple (feat64/feat32/mask) reuse is metric-NEUTRAL on the malloc bucket (a byte-identical steady-
    // state refactor, NOT the source of the ~20% bucket); the bucket the profile flagged is the per-leaf
    // logits_d + masked-softmax prior temporaries, which ws_.logits_d / ws_.prior_scratch now amortize.
    // OWNERSHIP is per-policy: each TreeState holds its OWN GumbelAZPolicy (fiber_tree.hpp), so this is
    // per-tree / per-fiber, NOT shared — within ONE tree the leaves are sequential (the parked fiber holds
    // ch.features into ws_.feat32 only until the driver encodes it at submit, BEFORE the next resume drives
    // the next leaf's evaluate), and concurrently-parked fibers each have their OWN ws_, so the reuse is
    // clobber-safe by ownership (no thread_local / global scratch — the buffers are passed explicitly to
    // the _into signatures / read by name in evaluate, P9 rule). `mutable` for the same reason fb_'s memos
    // are: the observable value-for-input is invariant, only the storage is reused (logical-const, single-owner).
    mutable FeatureWorkspace ws_;

    // The per-decision node-pool arena (ADR-0012 P9 rule 4; the SAME per-policy == per-TreeState/per-fiber
    // ownership as ws_, clobber-safe across the K concurrently-parked wire fibers — each holds its OWN
    // GumbelAZPolicy, so its OWN arena). A std::pmr::monotonic_buffer_resource: it carves the per-decision
    // node pool (the NodePool vector + each node's inner pmr containers) from `arena_buf_` up front, and
    // when that is exhausted requests the next chunk from the upstream allocator GEOMETRICALLY (the standard
    // monotonic_buffer_resource grows the requested block size ~2x as it goes — the 2x-growth arena the
    // maintainer directed, allocated up front + in batches from the normal allocator, NOT brk/sbrk).
    // run_search() calls arena_.release() at the START of each decision, recycling the whole prior tree's
    // storage back to `arena_buf_` (a monotonic resource frees nothing until release()) and reusing it
    // across that fiber's decisions. MEASURED justification (ADR-0009, the P9-rule-4 obligation): the K=64
    // e2e wire profile's ~6% allocator bucket (_int_malloc 4.06% + unlink_chunk/_int_free/__memmove) in
    // the search descent (see GumbelNode). Explicit typed members read by name (no thread_local / global) —
    // the P9-rule-4 typed-arena form. `mutable`: logical-const, single-owner storage reuse, exactly as ws_.
    //
    // O(fibers)-RESIDENT DISSOLUTION (ADR-0000; RCA tlab_finding #23/#26, massif-attributed, ADR-0009).
    // The node-pool arena was the DOMINANT per-fiber resident term and the one that scaled UNBOUNDEDLY with
    // the fiber population: a plain monotonic_buffer_resource over a large INLINE buffer with the DEFAULT
    // (new_delete -> glibc) upstream NEVER returns its grown high-water to the OS (release() only rewinds its
    // own chain; glibc retains the large frees — the measured R4 null result, <4% from the trim env knobs).
    // With every one of K fibers' trees held live, each fiber settled at its deepest-EVER decision's tree
    // size and held it for life -> resident = threads*K*max_tree (~2.4 GiB/producer at K=1024/n_sims=256;
    // four coincident producers OOM the 8 GiB box). TWO structural changes break the scaling, at BYTE-
    // IDENTICAL search output (only the allocator's upstream + the inline floor size change; ADR-0012 P6 —
    // an allocator swap perturbs no value; the parity gates + the Option-A proof re-validate this):
    //   (1) The overflow UPSTREAM is now MmapUpstream (releasing_arena.hpp): it mmap()s each block the
    //       monotonic resource overflows into and munmap()s it on deallocate, so run_search's per-decision
    //       arena_.release() (gumbel.cpp) RETURNS the prior decision's deep chunks to the OS. A parked fiber
    //       therefore holds only its CURRENT decision's live tree (release() runs at the NEXT decision's
    //       start, after the fiber's prior search has fully unwound), not its lifetime maximum. Episodic
    //       fibers sit mostly at shallow belief depths, so the coincident per-fiber high-water collapses.
    //   (2) The inline floor is cut 256 KiB -> kArenaInlineBytes (a shallow-decision floor): the inline buffer
    //       is itself an O(K) member term (256 KiB x K = 256 MiB at K=1024), pure waste for a PARKED fiber.
    //       It is sized so the COMMON shallow decision still fits without an upstream syscall (the hot path is
    //       unchanged); only deep decisions reach the mmap upstream, and that block is returned at the next
    //       release(). MEASURED sizing (ADR-0009): see the structural-fix run; the shallow-decision tree fits
    //       within kArenaInlineBytes, so the per-decision upstream-touch rate stays low (deep-tree tail only).
    static constexpr std::size_t kArenaInlineBytes = 32 * 1024;  // shallow-decision inline floor (was 256 KiB)
    mutable MmapUpstream arena_upstream_;                        // overflow -> OS-releasing mmap blocks
    mutable std::array<std::byte, kArenaInlineBytes> arena_buf_;
    mutable std::pmr::monotonic_buffer_resource arena_{arena_buf_.data(), arena_buf_.size(),
                                                       &arena_upstream_};
};

}  // namespace chocofarm
