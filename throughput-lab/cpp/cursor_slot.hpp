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
    CursorLeafView ch;                          // OUT: the parked leaf row (mirrors TreeState::ch.features)
    chocofarm::GumbelAZPolicy::Decision decision;
    bool running = false;                       // parked at a leaf (vs the decision finished)

    // The scripted (RNG-free) ctor — mirrors TreeState's scripted ctor: `table` scripts the
    // CyclicGumbelSource draws. The producer rotates the table per slot so the K trees differ, exactly as
    // the fiber path does.
    CursorSlot(const chocofarm::GumbelConfig& cfg, const chocofarm::Environment& env,
               std::vector<double> table)
        : policy(cfg, leaf_port(), env), src(env, std::move(table)) {}

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
        apply_step(cur->advance());
    }

    // Feed the driver-evaluated leaf back and advance to the next leaf (or finish). Mirrors
    // TreeState::resume_with — the driver calls it identically across engines.
    void resume_with(const chocofarm::NetPrediction& pred) { apply_step(cur->resume(pred)); }

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
    void apply_step(const chocofarm::Step& st) {
        if (std::holds_alternative<chocofarm::CursorNeedsLeaf>(st)) {
            ch.features = std::get<chocofarm::CursorNeedsLeaf>(st).features;
            running = true;
        } else {
            decision = std::get<chocofarm::CursorDecided>(st).decision;
            running = false;
        }
    }
};

}  // namespace tlab
