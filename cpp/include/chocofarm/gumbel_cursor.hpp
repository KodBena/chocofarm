// cpp/include/chocofarm/gumbel_cursor.hpp
// Purpose: OPTION B — the Gumbel-AZ search as an explicit, value-returning RESUMABLE STATE MACHINE
//   (the ONE home of the cursor type, ADR-0012 P1). It is the head-to-head alternative to Option A
//   (the stackful-fiber TreeState, fiber_tree.hpp): instead of running the UNCHANGED run_search inside
//   a boost.context fiber that yields at each leaf, TreeCursor REIFIES run_search's five-level recursion
//   (run_search -> sequential_halving -> visit -> simulate_root_action -> descend -> evaluate ->
//   predict) into an explicit reentry cursor that RETURNS a leaf request by value (advance()/resume(pred)
//   -> Step = variant<NeedsLeaf, Decided>) and runs STRAIGHT-LINE on the normal thread stack — no fiber,
//   no boost.context, no per-decision mmap'd stack, no hidden control-flow yield.
//
//   *** THE TYPE-DRIVEN ANSWER (ADR-0000) ***
//   Option A's parked-search state lives implicitly in a suspended C++ call stack — a value that is not
//   representable, only frozen. Option B makes that state a REPRESENTABLE TYPED VALUE: the SH
//   phase/considered/budget, the per-visit + per-c_outcome counters, and the descend recursion path are
//   each an explicit member or an entry on an explicit descend_stack_. The single suspension point (the
//   net predict) becomes a returned NeedsLeaf; re-entry resumes at exactly the draw the recursion was
//   about to make. This is the docs/design/cpp-search-runtime.md §3.2 advance/resume shape and the
//   ADR-0012 P9 functional-core form: advance/resume are total value-functions of typed inputs returning
//   a typed Step by value, no I/O, no throw, the which-arm carried in the variant (no sentinel).
//
//   *** BIT-IDENTITY BY CONSTRUCTION (the non-negotiable correctness gate, ADR-0009 two-tier) ***
//   TreeCursor is a friend of GumbelAZPolicy and REUSES its VALIDATED precision-critical helpers
//   VERBATIM — eval_build_features/eval_finish (the evaluate() split bracketing the leaf), puct_select,
//   improved_policy, sh_cut_sigma, root_logit — every one of which calls gumbel.cpp's file-local
//   prior_value / v_mix_mixed / sigma_scale_1a / masked_softmax_1a. So the FOUR 1b float32 seams, the
//   Danihelka invariants, and the per-tree RNG draw order are reproduced because the cursor REUSES the
//   exact same math the recursion runs, only re-sequenced through an explicit stack — it never
//   re-derives a seam. The RNG draw order is preserved by construction: the cursor draws the root
//   src.gumbel(n_slots) AFTER the root-eval leaf resolves (exactly where run_search does), then per-sim
//   sample_world and per-c_outcome (k>0) sample_world in the SAME order visit/simulate_root_action draw
//   them. cpp/parity/gumbel_logic.py (144/144) + gumbel_precision.py (144/144) + a B-vs-direct
//   bit-identity proof (gumbel_cursor_proto.cpp, analogous to fiber_proto.cpp) re-prove the crown jewels
//   against THIS path.
//
//   INVARIANT (the serial-per-tree exactness mechanism, structural): between an advance()/resume()
//   returning NeedsLeaf and the matching resume(), the cursor has EXACTLY ONE outstanding leaf and
//   CANNOT issue a second — the only way to get the next leaf is to resume() the previous one. Per-tree
//   in-flight == 1 falls out of the interface shape (cpp-search-runtime.md §6 cap a), not a runtime check.
//
// Public Domain (The Unlicense).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "chocofarm/collected_set.hpp"
#include "chocofarm/domains.hpp"  // SlotIndex/SlotCount/SimBudget/CandidateCount/PlyDepth/OutcomeIndex/ByteCount + World (P1)
#include "chocofarm/env.hpp"
#include "chocofarm/features.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/net_evaluator.hpp"
#include "chocofarm/releasing_arena.hpp"  // MmapUpstream — the SAME node-pool arena upstream run_search uses

namespace chocofarm {

// The cursor has parked at a leaf: it needs the net's forward over `features` (the length-in_dim()
// float32 feature row eval_build_features produced — valid until the next resume()). The driver predicts
// and feeds the value back via resume(). No corr_id here (this is the single-cursor primitive the bench
// + the proof drive directly; a multiplexing runtime would stamp routing in its own shell, design §4).
struct CursorNeedsLeaf {
    std::span<const float> features;  // OUT: the leaf feature row to forward (valid while parked)
};

// The cursor has finished the decision: the full GumbelAZPolicy::Decision (the executed action + the
// improved-π target + n_spent + survivor_slot), returned by value (P9 rule 2).
struct CursorDecided {
    GumbelAZPolicy::Decision decision;
};

// One advance()/resume() result: a typed sum carrying the which-arm in the TYPE (P9 rule 5 — no
// sentinel, no nullable pointer). Returned by value.
using Step = std::variant<CursorNeedsLeaf, CursorDecided>;

// The resumable Gumbel-AZ tree (Option B). It OWNS its _Node graph (its own NodePool), its per-leaf
// FeatureWorkspace, and the explicit reentry state (the SH bookkeeping + the descend stack); it does NOT
// own a net, a socket, a thread, or a fiber. The driver calls advance() to start and resume(pred) to feed
// each leaf back, alternating until a CursorDecided arm.
//
// LIFETIME (the same contract Option A's TreeState documents): start()/advance() capture (loc, bw,
// collected) BY REFERENCE; the cursor re-reads them across ALL resume() calls, so the caller must keep
// them alive (named lvalues, never temporaries) until a CursorDecided is returned. `lam` is captured by
// value. The GumbelSource is held by reference (the cursor draws off it at the same points run_search
// does). Move-deleted: the cursor holds a NodePool whose nodes carry pmr containers bound to its own
// arena member, plus spans into its workspace — a relocation would dangle (made a hard compile error,
// ADR-0002), so hold it as a never-moved local or behind a unique_ptr.
class TreeCursor {
  public:
    // Construct a cursor for ONE decision against `policy` (whose validated helpers it reuses), the live
    // (loc, bw, collected, lam), and the injected `src` (the SAME GumbelSource run_search would draw off).
    // Construction does no search work (the first advance() does the root-eval park); the references are
    // captured and must outlive the cursor (the lifetime contract above).
    TreeCursor(const GumbelAZPolicy& policy, const Loc& loc, const Belief& bw,
               const CollectedSet& collected, double lam, GumbelSource& src);

    TreeCursor(const TreeCursor&) = delete;
    TreeCursor& operator=(const TreeCursor&) = delete;
    TreeCursor(TreeCursor&&) = delete;
    TreeCursor& operator=(TreeCursor&&) = delete;

    // Advance the search forward from the current explicit state until it either PARKS at a leaf
    // (returns CursorNeedsLeaf) or FINISHES the decision (returns CursorDecided). The FIRST call parks at
    // the ROOT eval leaf (run_search's first net forward); thereafter advance() is the no-op entry the
    // driver may call once before the first resume(). Total value-function (no I/O, no throw).
    [[nodiscard]] Step advance();

    // Resume from a parked leaf with its prediction: finish the parked node's evaluate() (the
    // masked-softmax prior + value store, the 1b seam-1), then continue the search (the next park or the
    // decision). Resuming a cursor that is NOT parked at a leaf is an INVARIANT violation (a driver bug)
    // — asserted (P9: assert for one's own impossible state, expected for the world's boundary).
    [[nodiscard]] Step resume(const NetPrediction& prediction);

    // ---- ADDITIVE: the BatchPredict (lever #3) producer-compute A/B measurement seam ------------------
    // NONE of this is on the production path. The production producer/runtime drive advance()/resume() and
    // each cursor builds its OWN feature row inside eval_build_features (the PER-LEAF arm). To measure the
    // BATCHED arm (the in-process batched featurizer over the K multiplexed cursors' parked leaves), a
    // driver needs (a) to make the cursor SKIP its per-leaf feature build at the park, (b) to read the
    // parked (loc, bw, collected) triple so it can batch-featurize them together, and (c) to feed each
    // cursor its batch-built row back. These three additive members provide exactly that — the search
    // control flow + the RNG draw order + every precision seam are UNCHANGED (deferred featurization only
    // moves WHERE the identical feature row is computed; the legal-slots collection from the row is the
    // byte-identical tail eval_legal_from_features runs), so the BATCHED arm's decisions are bit-identical
    // to the PER-LEAF arm's. The harness's bit-identity gate proves it.

    // (a) Put this cursor in DEFERRED-FEATURIZE mode for the rest of its life: at each leaf park it sets
    // parked_ + records the parked node but does NOT call eval_build_features (no per-leaf belief sweep).
    // The driver must then resume via resume_with_features (NOT resume) so the legal-slots tail runs on the
    // externally-supplied row. Call before the first advance(). Idempotent.
    void enable_deferred_featurize() { defer_featurize_ = true; }

    // (b) The parked leaf's (loc, bw, collected) — valid ONLY while parked (between an advance()/resume that
    // returned CursorNeedsLeaf and the matching resume). For the root-eval park these are the decision-level
    // (loc_, bw_, collected_); for an interior/descent park they are the single live descent state
    // (descent_loc_, descent_bw_, descent_coll_) — exactly the triple eval_build_features would have swept.
    [[nodiscard]] const Loc& parked_loc() const { return *parked_loc_; }
    [[nodiscard]] const Belief& parked_belief() const { return *parked_bw_; }
    [[nodiscard]] const CollectedSet& parked_collected() const { return *parked_coll_; }

    // (c) Resume a DEFERRED-FEATURIZE park with the externally-built feature ROW + its prediction. It runs
    // eval_legal_from_features(parked_node, row) — the byte-identical legal-slots tail of eval_build_features
    // — to set node.legal_slots, then proceeds EXACTLY as resume(prediction). `row` MUST be the length-dim()
    // feature row for THIS cursor's parked (loc, bw, collected) (the batched featurizer's row for it). Only
    // valid in deferred mode; asserted.
    [[nodiscard]] Step resume_with_features(std::span<const double> row, const NetPrediction& prediction);

  private:
    // The reentry phase of the OUTER (root/SH) machine. The descend recursion is reified separately as
    // the descend_stack_ below; PHASE tracks where the OUTER loop is so resume() re-enters correctly.
    enum class Phase {
        RootEval,     // parked at (or about to do) the root-eval leaf — the decision's first forward
        Running,      // driving SH sims (drawing worlds, descending); the descend_stack_ owns the leaf
        Done          // the Decision is built (decision_ is final)
    };

    // One frame of the reified descend recursion (one level of gumbel.cpp's descend()). The descent is a
    // LINEAR chain (each descend recurses into exactly ONE child before unwinding), so descend_stack_ is
    // that chain made explicit; the cursor parks when the frame's node needs an eval leaf, and on unwind
    // applies the W/N backup exactly as descend()'s tail does.
    //
    // BY-REFERENCE BELIEF PARKING (the post-ref memory/CPU lever; ADR-0001 COW). A frame DELIBERATELY does
    // NOT hold the (loc, bw, collected, world) it descends from — those live in the SINGLE per-cursor
    // descent_* state below, narrowed IN PLACE down the chain. The earlier by-COPY draft stored a 2064-byte
    // Belief (the inline bitset arm, sizeof probed) in EVERY frame: a parked tree at max_depth=24 held ~24
    // belief copies (~48 KiB) and paid a 2 KiB memcpy on every descent step. This is sound because the
    // descent NEVER re-reads an ancestor's belief: a frame reads its (loc, bw, collected) only WHILE it is
    // descend_stack_.back() (to eval/park or to puct_select+apply the ONE child it steps into); once it has
    // pushed its child it is never back() again until unwind_with pops the WHOLE chain at once, and unwind
    // touches only W/N (never a belief). So one live narrowing belief per cursor suffices — the depth-d
    // belief lives in descent_bw_, reset from bw_ at each c_outcome determinization start (the one copy
    // that remains, see "needs a copy" below). The frame keeps ONLY what the W/N backup needs.
    struct DescendFrame {
        NodeIndex node{0};         // arena index of this frame's node (NodePool container index; the former
                                   //   -1 default is a typed 0 — `node` is always assigned a real index
                                   //   before it is read, by start_root/pump_descent push order)
        PlyDepth depth{0};         // descent depth (root-action child enters at depth 1, as descend does)
        SlotIndex action_slot{0};  // the puct_select'd action this frame stepped on (for the W/N backup);
                                   //   only read when `stepped` (the former -1 "not stepped" carried no
                                   //   slot value — `stepped` is the real guard, ADR-0002 typed presence)
        double step = 0.0;         // the immediate λ-penalized step reward of action_slot (added to cont)
        bool stepped = false;      // has this frame chosen+applied its action (vs still needing its eval)?
    };

    // ---- the outer SH bookkeeping (reifies sequential_halving's locals across resume() re-entries) ----
    // These mirror sequential_halving + visit + simulate_root_action's loop variables EXACTLY, so the
    // sims fire in the IDENTICAL order (the same world-draw order, the same per-action visit counts, the
    // same full-budget remainder loop). See gumbel_cursor.cpp for the step-by-step correspondence.
    void start_root();                 // build root features, park at the root-eval leaf
    Step drive();                      // the outer driver: pump SH sims / the descent until park or Done
    Step pump_descent();               // advance the descend_stack_ one node at a time until park or sim-done
    void unwind_with(double cont);     // back a child's `cont` up the descend chain (descend's W/N tail),
                                       //   folding the final descend(child,depth=1) return into sim_total_
    void on_sim_complete(double ret);  // a finished sim's return: backprop to root W/N (visit's tail)
    void finalize();                   // SH survivor + improved-π + no-early-exit substitution -> decision_

    const GumbelAZPolicy& p_;
    const Loc& loc_;
    const Belief& bw_;
    const CollectedSet& collected_;
    double lam_;
    GumbelSource& src_;

    SlotCount n_slots_;               // = n_action_slots(env) — the count that bounds SlotIndex
    SlotIndex term_slot_;             // = term_slot(env) — the always-legal TERMINATE slot
    FeatureWorkspace ws_;              // the cursor's OWN per-leaf scratch (P9 rule 4; never aliases p_.ws_)
    // The cursor's OWN per-decision node-pool arena — DELIBERATELY THE SAME SHAPE run_search uses
    // (gumbel.hpp): a std::pmr::monotonic_buffer_resource over a kArenaInlineBytes inline floor with an
    // MmapUpstream overflow. Matching the node ALLOCATOR is what makes the B-vs-A head-to-head attributable
    // to the SCHEDULING MECHANISM (fiber vs explicit cursor) and NOT to an allocator difference: if B's
    // NodePool used a plain std::vector (malloc/realloc) while run_search uses the mmap-overflow pmr arena,
    // a measured B<direct gap would conflate the cursor mechanism with the cheaper allocator (the confound
    // the first rough measure exposed). So the cursor mirrors the arena exactly; only the await-mechanism
    // differs. (One decision per cursor here, so no release() loop is needed — the arena frees at dtor.)
    // The inline floor as a typed ByteCount (the pmr/array sizing domain — domains.hpp); .value() is the
    // count->size_t ACL at the std::array non-type template arg + the monotonic_buffer_resource sizing.
    static constexpr ByteCount kArenaInlineBytes{32 * 1024};  // == GumbelAZPolicy::kArenaInlineBytes
    static_assert(kArenaInlineBytes == GumbelAZPolicy::kArenaInlineBytes,
                  "the cursor must mirror the policy's arena inline floor (the B-vs-A allocator parity)");
    MmapUpstream arena_upstream_;
    std::array<std::byte, kArenaInlineBytes.value()> arena_buf_;
    std::pmr::monotonic_buffer_resource arena_{arena_buf_.data(), arena_buf_.size(), &arena_upstream_};
    NodePool nodes_{&arena_};          // the cursor's OWN node pool, served from the SAME arena shape as A
    std::vector<double> root_logits_;  // root log-prior logits (seam 1) feeding top-k AND the SH cut
    std::vector<double> g_;            // the root Gumbel draw (src.gumbel(n_slots), drawn after root eval)

    Phase phase_ = Phase::RootEval;

    // SH state: the survivor-elimination bracket. `considered_` is the live candidate set (halved each
    // phase); per_phase_/budget_/n_spent_ track the budget exactly as sequential_halving does. The
    // "which candidate / which of its per_action sims / which c_outcome" cursor is below.
    std::vector<SlotIndex> considered_;  // the live candidate-slot set (halved each phase)
    SimBudget sh_per_phase_{0};
    SimBudget sh_budget_{0};
    SimBudget n_spent_{0};
    bool sh_single_ = false;           // the |considered|==1 fast path (visit n_sims on the lone candidate)
    bool sh_remainder_ = false;        // in the post-bracket full-budget remainder round-robin
    bool sh_phase_active_ = false;     // a phase has been ENTERED (its per-action loop + cut owe to run);
                                       //   the while-condition is re-checked only when this clears (so the
                                       //   cut ALWAYS fires for an entered phase, even if budget hit 0)
    bool sh_phase_broke_ = false;      // the per-action loop broke early on v<=0 (still cuts afterwards)
    OutcomeIndex sh_phase_idx_{0};     // position into considered_ within the phase's per-action loop (an
                                       //   affine cursor 0..size — the shared OutcomeIndex affine-index
                                       //   domain, domains.hpp; NOT a slot value — considered_[idx] IS one)
    SimBudget sh_per_action_{0};       // this phase's per-action sim count
    SimBudget sh_action_done_{0};      // sims already issued for considered_[sh_phase_idx_] this phase
    OutcomeIndex sh_rr_{0};            // round-robin cursor for the remainder loop (the OutcomeIndex affine
                                       //   domain domains.hpp explicitly assigns "the round-robin survivor
                                       //   cursor, read modulo considered.size()")

    // The current sim's root-action context (simulate_root_action's locals): the root action slot under
    // test, its c_outcome accumulation, and which determinization k we are on. `sim_total_` accumulates
    // the c_outcome sum; the visit-level return is sim_total_/c_outcome.
    std::optional<SlotIndex> cur_root_slot_;  // the root action under test, or empty = "no current sim"
                                       //   (the former -1 sentinel -> typed absence, ADR-0002)
    OutcomeIndex cur_k_{0};            // c_outcome determinization index (0..c_outcome-1)
    World cur_world_ = 0;              // the visit-drawn world for this sim (k==0 reuses it, k>0 redraws)
    double sim_total_ = 0.0;           // running Σ_k (step + descend cont) for the current root-action sim

    std::vector<DescendFrame> descend_stack_;  // the reified descend recursion (linear chain)

    // THE SINGLE LIVE DESCENT STATE (the by-reference belief parking — one narrowing belief per cursor,
    // NOT one per frame). At a c_outcome determinization start these are COPIED ONCE from the decision-
    // level (bw_, loc_, collected_) + the drawn world; thereafter env.apply narrows them IN PLACE as the
    // descent steps deeper (the COW filter returns a fresh belief, but apply writes it back into
    // descent_bw_ — one belief alive at a time, the deepest frame's). pump_descent reads/steps these; the
    // frame holds only node/depth/action_slot/step (the W/N-backup state). descent_world_ is constant down
    // a determinization (descend reuses `world`). Correct because no ancestor belief is re-read after its
    // child is produced (see DescendFrame's by-reference note + the unwind-touches-only-W/N invariant).
    Loc descent_loc_;
    Belief descent_bw_;
    CollectedSet descent_coll_;
    World descent_world_ = 0;

    GumbelAZPolicy::Decision decision_;         // built by finalize()

    bool parked_ = false;              // true iff a leaf is outstanding (resume() expected next)

    // ---- ADDITIVE: deferred-featurize state (the BatchPredict A/B seam; off on the production path) -----
    // In deferred mode the park sites SKIP eval_build_features and instead record the parked node index +
    // the parked (loc, bw, collected) views, so the driver can batch-featurize and feed the row back via
    // resume_with_features. parked_node_ identifies which node eval_legal_from_features must populate (the
    // same node resume() would finish — nodes_[0] at root-eval, descend_stack_.back().node at a descent).
    bool defer_featurize_ = false;
    NodeIndex parked_node_{0};                       // the node parked (deferred): the eval_finish target
    const Loc* parked_loc_ = nullptr;                // the parked leaf's loc (root-eval or descent), or null
    const Belief* parked_bw_ = nullptr;              // the parked leaf's belief
    const CollectedSet* parked_coll_ = nullptr;      // the parked leaf's collected set
};

}  // namespace chocofarm
