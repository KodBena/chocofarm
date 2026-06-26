// throughput-lab/cpp/cursor_slot.hpp
// Purpose: CursorSlot — the OPTION-B per-slot adapter that drives ONE Gumbel-AZ decision through the
//   explicit-state TreeCursor (chocofarm/gumbel_cursor.hpp) while exposing the SAME surface the real
//   producer's episodic drivers already use on the Option-A TreeState (fiber_tree.hpp): `.ch.features`
//   (the parked leaf row), `.running` (parked-at-a-leaf vs decision-done), `.decision`, `.start(loc, bw,
//   coll, lam)`, and `.resume_with(pred)`. Because the surface matches TreeState exactly, the producer's
//   coalescing send/recv (`drive_round`) + episode state machine (`advance(i)`) + both pipe shapes
//   (round-sync / greedy) are REUSED VERBATIM over a slot TYPE PARAMETER (ADR-0012 P1: one home for the
//   episode logic; the engine is the only thing that differs). The fiber engine stays byte-untouched.
//
//   This is the lab-side ACL (ADR-0012 P2): it TRANSLATES TreeCursor's value-returning advance()/resume()
//   (Step = variant<CursorNeedsLeaf, Decided>) into the imperative `.running`/`.ch.features`/`.resume_with`
//   shape the drivers expect — it does NOT change the search (the cursor's bit-identity to run_search is
//   proven by cpp/parity + gumbel_cursor_proto). One leaf in flight per slot is structural (the only way to
//   get the next leaf is resume()), exactly the per-tree-in-flight==1 invariant the drivers rely on.
//
//   LIFETIME (the SAME contract TreeState documents): start() captures (loc, bw, coll) BY REFERENCE into
//   the TreeCursor and re-reads them across every resume() until the decision finishes; the caller must
//   keep them alive (the producer's per-slot ep_loc/ep_bw/ep_coll vectors are RESERVED, never reallocated,
//   exactly as the fiber path requires). Move-deleted (it owns a TreeCursor which is move-deleted and holds
//   references into the slot's own source) — hold it behind a unique_ptr, as the fiber path holds TreeState.
//
//   MEMORY NOTE (the maintainer's explicit point, surfaced not buried): the cursor parks Loc/Belief/
//   CollectedSet BY COPY per descend frame (a DescendFrame in the cursor's descend_stack_), and a Belief is
//   the world-set — so a parked B-slot's resident set can EXCEED A's demand-paged fiber stack. This adapter
//   does NOT change that (the by-reference / park-belief-by-index redesign is the NEXT phase, localized to
//   the cursor's DescendFrame); it only makes B runnable end-to-end so the pre-redesign A-vs-B e2e
//   (throughput AND RSS) can be measured. The lever lives in gumbel_cursor.hpp, not here.
//
// Public Domain (The Unlicense).
#pragma once

#include <expected>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "chocofarm/collected_set.hpp"
#include "chocofarm/error.hpp"
#include "chocofarm/cyclic_gumbel.hpp"
#include "chocofarm/env.hpp"
#include "chocofarm/gumbel.hpp"
#include "chocofarm/gumbel_cursor.hpp"
#include "chocofarm/net_evaluator.hpp"

namespace tlab {

// The nested `ch` mirrors TreeState::ch's ONE field the drivers read (the parked leaf feature row), so a
// driver's `slot->ch.features` is identical across engines. (TreeState's ch also carries the fiber caller
// + the value slot; the cursor needs neither — the features live in the cursor's workspace, exposed here.)
struct CursorLeafView {
    std::span<const float> features;  // the parked leaf row (valid while `running`)
};

// One Option-B decision slot. It owns a persistent GumbelAZPolicy + CyclicGumbelSource (so the gumbel
// table cycles across the slot's decisions exactly as the fiber slot's source does) and reconstructs a
// TreeCursor per decision in start(). The decision is driven leaf-by-leaf: start() advances to the first
// parked leaf (or straight to a finished decision on an empty belief), resume_with() feeds the evaluated
// leaf back and advances to the next. `running` is true iff a leaf is parked (vs the decision being done).
struct CursorSlot {
    chocofarm::GumbelAZPolicy policy;
    chocofarm::CyclicGumbelSource src;
    std::optional<chocofarm::TreeCursor> cur;   // (re)constructed each decision (move-deleted -> optional)
    CursorLeafView ch;                          // OUT: the parked leaf row (mirrors TreeState::ch.features);
                                                //   per-leaf mode only — UNUSED in batched mode (see below)
    chocofarm::GumbelAZPolicy::Decision decision;
    bool running = false;                       // parked at a leaf (vs the decision finished)
    bool batched = false;                       // BatchPredict (lever #3): if set, the cursor runs in
                                                //   DEFERRED-featurize mode — it parks WITHOUT building its
                                                //   per-leaf row, and the driver featurizes the K parked
                                                //   leaves TOGETHER (one belief sweep per RTT instead of K)
                                                //   then installs the batch-built row here. After
                                                //   install_batched_row, ch.features points at row32_ — so
                                                //   the driver's wire send path is byte-IDENTICAL across
                                                //   engines (it always concats slot->ch.features).
    std::vector<double> row64;                  // batched: the float64 featurized row (the resume_with_features
                                                //   input — the cursor's eval_legal tail consumes float64,
                                                //   exactly as the per-leaf eval_build_features does)
    std::vector<float> row32;                   // batched: the float32 narrow the wire forwards (== what the
                                                //   per-leaf path narrows from ch.features before send)

    // The scripted (RNG-free) ctor — mirrors TreeState's scripted ctor: `table` scripts the
    // CyclicGumbelSource draws. The producer rotates the table per slot so the K trees differ, exactly as
    // the fiber path does. `deferred_featurize` arms BatchPredict (lever #3): the cursor skips its per-leaf
    // feature build at every park, and the driver builds the rows in one batched belief sweep per RTT (the
    // same seam the mux producer-compute bench proved bit-identical — cpp/src/multiplexed_producer_compute_bench.cpp).
    CursorSlot(const chocofarm::GumbelConfig& cfg, const chocofarm::Environment& env,
               std::vector<double> table, bool deferred_featurize = false)
        : policy(cfg, leaf_port(), env), src(env, std::move(table)), batched(deferred_featurize) {}

    // Move-deleted (owns a move-deleted TreeCursor that captures references into THIS slot's `src`).
    CursorSlot(const CursorSlot&) = delete;
    CursorSlot& operator=(const CursorSlot&) = delete;
    CursorSlot(CursorSlot&&) = delete;
    CursorSlot& operator=(CursorSlot&&) = delete;

    // Begin a decision from (loc, bw, coll, lam): construct the cursor (capturing the refs) and advance to
    // its first parked leaf — or, on an empty belief, straight to a finished decision. After this, `running`
    // == parked-at-a-leaf and `ch.features` is the leaf row to forward. LIFETIME: loc/bw/coll MUST outlive
    // the decision (captured by reference, re-read on every resume_with — the producer's reserved per-slot
    // vectors satisfy this, identical to TreeState::start).
    void start(const chocofarm::Loc& loc, const chocofarm::Belief& bw,
               const chocofarm::CollectedSet& coll, double lam) {
        cur.emplace(policy, loc, bw, coll, lam, src);
        if (batched) cur->enable_deferred_featurize();  // lever #3: skip the per-leaf build; the driver batches
        apply_step(cur->advance());
    }

    // Feed the driver-evaluated leaf back and advance to the next leaf (or finish). Mirrors
    // TreeState::resume_with — the driver calls it identically across engines (PER-LEAF mode).
    void resume_with(const chocofarm::NetPrediction& pred) { apply_step(cur->resume(pred)); }

    // ---- BatchPredict (lever #3) seam: the deferred-featurize triple + the row-resume -------------------
    // These mirror the TreeCursor's deferred-featurize members (gumbel_cursor.hpp) one-to-one, exposing them
    // through the slot ACL so the driver can collect the K parked leaves' (loc, bw, collected) into a
    // BatchLeaf vector, featurize them ONCE per RTT (BatchFeaturizer::featurize_batch), and resume each slot
    // with its batch-built row. Valid ONLY in batched mode (asserted by the cursor); the parked triple is
    // valid only while `running` (between this slot's park and its resume_with_features), exactly as
    // TreeCursor::parked_* documents. The row crossed back must be THIS slot's length-dim() featurized row.
    [[nodiscard]] const chocofarm::Loc& parked_loc() const { return cur->parked_loc(); }
    [[nodiscard]] const chocofarm::Belief& parked_belief() const { return cur->parked_belief(); }
    [[nodiscard]] const chocofarm::CollectedSet& parked_collected() const { return cur->parked_collected(); }

    // Install the batch-built float64 row for THIS slot's parked leaf (called once per RTT after
    // featurize_batch). Stores the float64 row (kept for resume_with_features) AND its float32 narrow, then
    // points ch.features at the narrow — so the driver's wire send concats slot->ch.features UNCHANGED across
    // engines (the float32 row is byte-identical to what the per-leaf cursor would have produced; the mux
    // bench proves it). The float64 row outlives the send→reply window (it lives in this slot, parked).
    void install_batched_row(std::span<const double> row) {
        row64.assign(row.begin(), row.end());
        row32.assign(row.begin(), row.end());                 // double -> float32 (the wire dtype)
        ch.features = std::span<const float>(row32.data(), row32.size());
    }

    // Resume a batched (deferred-featurize) park with the installed float64 row + its prediction. Runs the
    // byte-identical legal-slots tail (eval_legal_from_features) then proceeds exactly as resume_with — so the
    // BATCHED arm is bit-identical to PER-LEAF (the seam the mux bench proved bit-identical). Uses row64
    // (installed above), NOT row32: the legal-slots tail consumes the float64 row, exactly as the per-leaf
    // eval_build_features does (the float32 narrow is only the wire's; precision seams unchanged).
    void resume_with_batched(const chocofarm::NetPrediction& pred) {
        apply_step(cur->resume_with_features(std::span<const double>(row64.data(), row64.size()), pred));
    }

  private:
    // The leaf the policy holds is irrelevant for the cursor (TreeCursor never calls policy.net_ — it parks
    // at eval_build_features and the driver forwards the row). The policy ctor still requires a NetEvaluator
    // reference, so we bind a process-static no-op port: a single shared sentinel, never invoked (the cursor
    // bypasses it). This keeps the policy construction honest (a real port reference, not a dangling one)
    // without a per-slot dummy. ADR-0002: if it were ever called it returns a typed Error (loud), not a lie.
    static const chocofarm::NetEvaluator& leaf_port() {
        struct Unused final : chocofarm::NetEvaluator {
            [[nodiscard]] std::expected<chocofarm::NetPrediction, chocofarm::Error> predict(
                std::span<const float>) const override {
                return std::unexpected(chocofarm::make_error(
                    "CursorSlot::leaf_port: the cursor must park at leaves, never call predict()"));
            }
        };
        static const Unused kUnused;
        return kUnused;
    }

    // Translate a Step into the imperative surface: NeedsLeaf -> running + expose the features; Decided ->
    // not running + capture the decision (the driver then reads `decision` + steps the episode).
    //
    // BatchPredict (lever #3) — the install-before-read NET (ADR-0000 make-the-illegal-state-unrepresentable
    // + ADR-0002 fail-loud): in DEFERRED mode the cursor parks WITHOUT building its row, so the span the Step
    // carries (the cursor's ws_.feat32) is STALE — the previous leaf's row, or empty on the first park. If the
    // driver ever forwarded THAT, it would silently send a wrong row. So here, in batched mode, we POISON
    // ch.features to an EMPTY span: it is non-empty ONLY after install_batched_row writes the batch-built row.
    // A missed install therefore ships a zero-length row, which the wire/recv size-match guard rejects LOUDLY
    // (a B-vs-preds mismatch) rather than corrupting a decision — the illegal "send an uninstalled batched
    // row" state is made structurally visible, not left to the two drivers' hand-matched call ordering.
    void apply_step(const chocofarm::Step& st) {
        if (std::holds_alternative<chocofarm::CursorNeedsLeaf>(st)) {
            ch.features = batched ? std::span<const float>{}                               // poison: install required
                                  : std::get<chocofarm::CursorNeedsLeaf>(st).features;     // per-leaf: the cursor's own row
            running = true;
        } else {
            decision = std::get<chocofarm::CursorDecided>(st).decision;
            running = false;
        }
    }
};

}  // namespace tlab
