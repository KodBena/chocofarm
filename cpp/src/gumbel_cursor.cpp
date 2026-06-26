// cpp/src/gumbel_cursor.cpp
// Purpose: OPTION B implementation — TreeCursor, the explicit-state resumable Gumbel-AZ search (see
//   gumbel_cursor.hpp). It reifies gumbel.cpp's five-level recursion (run_search -> sequential_halving ->
//   visit -> simulate_root_action -> descend -> evaluate -> predict) into an advance()/resume() state
//   machine that returns a leaf request by value and runs on the normal thread stack. Every
//   precision-critical computation is DELEGATED to the friend policy's validated helpers
//   (eval_build_features/eval_finish, puct_select, improved_policy, sh_cut_sigma, root_logit), so the four
//   1b float32 seams + the Danihelka invariants are bit-identical BY CONSTRUCTION (reuse, not re-derive).
//   The only re-expressed logic is the CONTROL FLOW (the SH loop, the visit/c_outcome counters, and the
//   descend recursion as an explicit stack) — and that is structured to fire the RNG draws + the leaves in
//   the IDENTICAL order the recursion does, so re-entry resumes at exactly the draw the recursion was
//   about to make.
//
//   CORRESPONDENCE MAP (each cursor member ↔ its recursion local; gumbel.cpp line refs are illustrative):
//     run_search          -> start_root() (root eval park) + the post-resume top-k + finalize()
//     sequential_halving  -> considered_/sh_per_phase_/sh_budget_/n_spent_/sh_single_/sh_remainder_ + the
//                            phase cursor (sh_phase_idx_/sh_per_action_/sh_action_done_/sh_rr_)
//     visit (×count)      -> the per-candidate sim issue in drive(); on_sim_complete() is visit's W/N tail
//     simulate_root_action-> cur_root_slot_/cur_k_/sim_total_ (the c_outcome loop) + the root apply+child
//     descend (recursion) -> descend_stack_ (the linear descent chain) + pump_descent() (one node/step)
//     evaluate -> predict -> the single park point (CursorNeedsLeaf), bracketed by eval_build_features /
//                            eval_finish (the friend split)
//
// Public Domain (The Unlicense).
#include "chocofarm/gumbel_cursor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

namespace chocofarm {

namespace {
// Reconstruct an Action from its slot (the inverse of action_to_slot; a copy of gumbel.cpp's file-local
// action_of_slot — pure slot->Action, precision-irrelevant, NOT a seam). Slot 0..N-1 = Treasure i;
// N..N+nD-1 = Detector j; term_slot = TERMINATE.
// ACL: Action.i is a raw int (env.hpp), env.N()/n_detectors() return raw int; slot.value() crosses
// SlotIndex->raw at the kind decision + the detector-offset subtract.
[[nodiscard]] Action action_of_slot(const Environment& env, SlotIndex slot) {
    const int s = static_cast<int>(slot.value());
    if (s < env.N()) return Action{ActionKind::Treasure, s};
    if (s < env.N() + env.n_detectors()) return Action{ActionKind::Detector, s - env.N()};
    return terminate_action();
}
}  // namespace

TreeCursor::TreeCursor(const GumbelAZPolicy& policy, const Loc& loc, const Belief& bw,
                       const CollectedSet& collected, double lam, GumbelSource& src)
    : p_(policy), loc_(loc), bw_(bw), collected_(collected), lam_(lam), src_(src),
      n_slots_(n_action_slots(policy.env_)), term_slot_(term_slot(policy.env_)) {
    // Reserve the node pool modestly; it grows as the tree does, served from the cursor's own pmr arena
    // (the SAME monotonic_buffer_resource + MmapUpstream shape run_search uses — see gumbel_cursor.hpp).
    // Matching the node allocator keeps the B-vs-A head-to-head attributable to the SCHEDULING mechanism,
    // not to an allocator difference (the confound a plain-std::vector NodePool would introduce).
    nodes_.reserve(64);
}

// ---- advance / resume (the public state machine) --------------------------------------------------
Step TreeCursor::advance() {
    if (phase_ == Phase::Done) return CursorDecided{decision_};
    if (phase_ == Phase::RootEval && !parked_) {
        // empty-belief guard (mirrors run_search's len(bw)==0 short-circuit): the only continuation is
        // to exit; NO leaf is issued.
        if (p_.env_.empty(bw_)) {
            decision_.action = terminate_action();
            decision_.improved.assign(static_cast<size_t>(n_slots_.value()), 0.0);
            decision_.improved[static_cast<size_t>(term_slot_.value())] = 1.0;
            decision_.survivor_slot = term_slot_;  // typed SlotIndex (empty-belief Terminate survivor)
            decision_.n_spent = SimBudget{0};
            phase_ = Phase::Done;
            return CursorDecided{decision_};
        }
        start_root();  // build the root features, park at the root-eval leaf
        return CursorNeedsLeaf{ws_.feat32};
    }
    if (parked_) {
        // advance() called while parked is a no-op re-report of the same request (the driver should
        // resume(); we tolerate a redundant advance() returning the same park).
        if (!descend_stack_.empty())
            return CursorNeedsLeaf{ws_.feat32};
        return CursorNeedsLeaf{ws_.feat32};
    }
    return drive();
}

Step TreeCursor::resume(const NetPrediction& prediction) {
    assert(parked_ && "TreeCursor::resume on a cursor not parked at a leaf (driver invariant violation)");
    parked_ = false;

    if (phase_ == Phase::RootEval) {
        // finish the ROOT eval (the masked-softmax prior + value store), then do run_search's post-root
        // setup: the root logits (seam 1), the Gumbel-top-k draw + sort, and the SH bracket init.
        p_.eval_finish(nodes_[0], prediction, ws_);
        // root logits = log(prior) over legal slots (seam 1: float32 prior precision -> log) — illegal -1e30.
        root_logits_.assign(static_cast<size_t>(n_slots_.value()), -1e30);
        for (SlotIndex s : nodes_[0].legal_slots)
            root_logits_[static_cast<size_t>(s.value())] = p_.root_logit(nodes_[0], s);
        // Gumbel-Top-k: ONE gumbel draw over the FULL slot space (drawn AFTER the root eval — exactly
        // where run_search draws it), sort logits+g, take the top-m legal slots.
        // ACL: gumbel(int n) takes the slot-space draw length raw (the override-gated virtual).
        g_ = src_.gumbel(static_cast<int>(n_slots_.value()));
        std::vector<std::pair<double, SlotIndex>> scored;
        scored.reserve(nodes_[0].legal_slots.size());
        for (SlotIndex s : nodes_[0].legal_slots)
            scored.emplace_back(root_logits_[static_cast<size_t>(s.value())] + g_[static_cast<size_t>(s.value())], s);
        std::stable_sort(scored.begin(), scored.end(),
                         [](const std::pair<double, SlotIndex>& a, const std::pair<double, SlotIndex>& b) {
                             return a.first > b.first;
                         });
        // m = min(self.m, #legal): a CandidateCount; cfg_.m is a raw int (GumbelConfig, CLI-set).
        int m = std::min(p_.cfg_.m, static_cast<int>(nodes_[0].legal_slots.size()));
        considered_.clear();
        considered_.reserve(static_cast<size_t>(m));
        for (int i = 0; i < m; ++i) considered_.push_back(scored[static_cast<size_t>(i)].second);

        // SH bracket init (mirrors sequential_halving's head).
        n_spent_ = SimBudget{0};
        if (considered_.empty()) {  // unreachable on a non-empty belief, kept for the contract
            decision_.survivor_slot = std::nullopt;  // typed absence (the unreachable "no survivor")
            finalize();
            phase_ = Phase::Done;
            return CursorDecided{decision_};
        }
        phase_ = Phase::Running;
        if (considered_.size() == 1) {
            // the lone-candidate fast path: visit n_sims sims on considered_[0].
            sh_single_ = true;
            cur_root_slot_ = considered_[0];
        } else {
            // ACL: cfg_.n_sims is a raw int (GumbelConfig). m_sz/n_phases are CandidateCount; the budget
            // family is SimBudget.
            int m_sz = static_cast<int>(considered_.size());
            int n_phases = std::max(1, static_cast<int>(std::ceil(std::log2(static_cast<double>(m_sz)))));
            sh_per_phase_ = SimBudget{static_cast<SearchRep>(std::max(1, p_.cfg_.n_sims / n_phases))};
            sh_budget_ = SimBudget{static_cast<SearchRep>(p_.cfg_.n_sims)};
            // start the first phase's per-action loop
            SimBudget phase_budget{std::min(sh_per_phase_.value(), sh_budget_.value())};
            sh_per_action_ = SimBudget{static_cast<SearchRep>(std::max(
                1, static_cast<int>(phase_budget.value()) / static_cast<int>(considered_.size())))};
            sh_phase_idx_ = 0;
            sh_action_done_ = SimBudget{0};
        }
        // begin the first sim of the first candidate
        return drive();
    }

    // phase_ == Running: the parked leaf is the TOP descend frame's node eval. Finish it, then unwind.
    assert(!descend_stack_.empty() && "resume(Running) with an empty descend stack");
    DescendFrame& f = descend_stack_.back();
    p_.eval_finish(nodes_[static_cast<size_t>(f.node)], prediction, ws_);
    // The eval branches of descend() all RETURN node.value immediately after evaluating (the leaf
    // estimate; no deeper recursion this call, NO W/N touch on the evaluated node). Pop this frame and
    // back its node.value up the chain — exactly descend()'s eval-branch return propagating up.
    double leaf_val = nodes_[static_cast<size_t>(f.node)].value;
    descend_stack_.pop_back();
    unwind_with(leaf_val);
    return drive();  // advance the c_outcome loop / finish the sim
}

// ---- start: build the root node + its eval features, park -----------------------------------------
void TreeCursor::start_root() {
    nodes_.clear();
    nodes_.emplace_back(n_slots_);  // root at arena index 0; dense W/N
    p_.fb_.reset_belief_cache();    // scope the belief memo to this decision (mirrors run_search)
    // build the root features into the cursor's OWN workspace and park (eval_finish runs on resume).
    (void)p_.eval_build_features(nodes_[0], loc_, bw_, collected_, ws_);
    parked_ = true;
}

// ---- the outer driver: pump SH sims until a park or Done ------------------------------------------
Step TreeCursor::drive() {
    for (;;) {
        // If a descent is in flight, pump it (it parks or completes the current c_outcome determinization).
        if (!descend_stack_.empty()) {
            Step s = pump_descent();
            if (std::holds_alternative<CursorNeedsLeaf>(s)) return s;
            // descent completed: pump_descent set sim_total_ via the unwind; fall through to c_outcome.
        }

        // Are we mid-sim (a root action under c_outcome determinization)?
        if (cur_root_slot_.has_value()) {
            Action a = action_of_slot(p_.env_, *cur_root_slot_);
            if (a.kind == ActionKind::Terminate) {
                // simulate_root_action terminate short-circuit: return -lam*exit_cost (NO c_outcome loop,
                // NO leaf, NO world draws). The whole sim's value is this.
                double ret = -lam_ * p_.env_.exit_cost(loc_.pt);
                on_sim_complete(ret);
                continue;
            }
            // simulate_root_action's c_outcome loop. cur_k_ is the NEXT determinization to start; each
            // started determinization either parks (returns up via resume) or completes its descent
            // synchronously (a no-leaf path), folding its (step + descend cont) into sim_total_.
            // ACL: cfg_.c_outcome is a raw int (GumbelConfig) -> OutcomeIndex at the loop bound.
            if (cur_k_ < OutcomeIndex{static_cast<SearchRep>(p_.cfg_.c_outcome)}) {
                // k==0 reuses the visit-drawn world (cur_world_), k>0 draws fresh — the IDENTICAL draw
                // order simulate_root_action uses (no draw for k==0, one sample_world per k>0).
                World w = (cur_k_ == OutcomeIndex{0}) ? cur_world_ : src_.sample_world(bw_);
                cur_k_ = cur_k_ + SearchRep{1};  // OutcomeIndex affine: this determinization is consumed
                // BY-REFERENCE: seed the SINGLE live descent state from the decision-level (loc_, bw_,
                // collected_) — the ONE belief copy that remains, per determinization (the decision belief
                // is reused across c_outcome, so it must be preserved; descent_bw_ is the mutable narrowing
                // copy). env.apply then narrows descent_* IN PLACE to the depth-1 child state.
                descent_loc_ = loc_;
                descent_bw_ = bw_;
                descent_coll_ = collected_;
                descent_world_ = w;
                StepResult sr = p_.env_.apply(descent_loc_, descent_bw_, descent_coll_, a, descent_world_);
                double step = sr.reward - lam_ * sr.dt;
                // find/create the root child node for (slot, belief_key) (simulate_root_action's body).
                std::tuple<SlotIndex, GBeliefKey> ckey{*cur_root_slot_, gumbel_belief_key(p_.env_, descent_bw_)};
                int child;
                auto cit = nodes_[0].children.find(ckey);
                if (cit == nodes_[0].children.end()) {
                    nodes_.emplace_back(n_slots_);
                    child = static_cast<int>(nodes_.size()) - 1;
                    nodes_[0].children[ckey] = child;
                } else {
                    child = cit->second;
                }
                // simulate_root_action: total += step + descend(child, ..., depth=1). Fold `step` now;
                // the descent's `cont` is added by unwind_with when the descent unwinds.
                sim_total_ += step;
                DescendFrame fr;
                fr.node = child;
                fr.depth = PlyDepth{1};
                descend_stack_.push_back(fr);  // the depth-1 child state lives in descent_* (not the frame)
                continue;  // loop: pump_descent drives the new frame (park or complete)
            }
            // c_outcome loop finished for this root action: the sim value is sim_total_/c_outcome.
            double ret = sim_total_ / static_cast<double>(p_.cfg_.c_outcome);
            on_sim_complete(ret);
            continue;
        }

        // Not mid-sim: pull the NEXT sim to run from the SH schedule. Returns false-equivalent (Done) when
        // the budget is exhausted.
        // --- the lone-candidate fast path: visit n_sims on considered_[0] ---
        if (sh_single_) {
            if (n_spent_ < SimBudget{static_cast<SearchRep>(p_.cfg_.n_sims)}) {  // ACL: cfg_.n_sims raw int
                cur_root_slot_ = considered_[0];
                cur_world_ = src_.sample_world(bw_);  // visit(): w = src.sample_world(bw)
                cur_k_ = OutcomeIndex{0};
                sim_total_ = 0.0;
                continue;
            }
            // done: survivor is considered_[0].
            decision_.survivor_slot = considered_[0];  // typed SlotIndex (the lone-survivor fast path)
            finalize();
            phase_ = Phase::Done;
            return CursorDecided{decision_};
        }

        // --- the multi-candidate SH bracket ---
        // CORRESPONDENCE (the load-bearing structure): the original is
        //   while (considered.size() > 1 && budget > 0) { <per-action loop>; <cut>; }
        // so the condition is checked ONCE at the TOP of each phase, and the CUT runs at the END of EVERY
        // entered phase — EVEN IF the per-action loop drained budget to 0 (the condition is re-checked
        // only at the NEXT top). The cursor mirrors this with sh_phase_active_: a phase is ENTERED (the
        // top-of-loop check), its per-action loop runs to completion, then the cut ALWAYS fires; only
        // THEN is the while condition re-checked. (The earlier draft re-checked budget>0 BEFORE the cut,
        // skipping the final cut when a phase exactly drained the budget — the m≫1/small-n_sims survivor
        // divergence. ADR-0000: the fix is the structural one — enter-then-always-cut, not a guard.)
        if (!sh_remainder_) {
            if (!sh_phase_active_) {
                // top-of-while check: enter a phase iff size>1 && budget>0.
                if (considered_.size() > 1 && sh_budget_ > SimBudget{0}) {
                    sh_phase_active_ = true;
                    SimBudget phase_budget{std::min(sh_per_phase_.value(), sh_budget_.value())};
                    sh_per_action_ = SimBudget{static_cast<SearchRep>(std::max(
                        1, static_cast<int>(phase_budget.value()) / static_cast<int>(considered_.size())))};
                    sh_phase_idx_ = 0;
                    sh_action_done_ = SimBudget{0};
                    sh_phase_broke_ = false;
                    continue;
                }
                // bracket finished (size==1 or budget==0): enter the remainder round-robin.
                sh_remainder_ = true;
                sh_rr_ = 0;
                continue;
            }
            // a phase is active: run its per-action loop, then ALWAYS cut.
            if (!sh_phase_broke_ && sh_phase_idx_ < static_cast<int>(considered_.size())) {
                SlotIndex s = considered_[static_cast<size_t>(sh_phase_idx_)];
                SimBudget v{std::min(sh_per_action_.value(), sh_budget_.value())};
                if (v == SimBudget{0}) {  // unsigned: v<=0 is v==0 (the never-negative invariant, ADR-0000)
                    // sequential_halving: `if (v <= 0) break;` -> end this phase's per-action loop early
                    // (but STILL cut afterwards — the break only exits the per-action for-loop).
                    sh_phase_broke_ = true;
                    continue;
                }
                if (sh_action_done_ < v) {
                    // issue one more sim of candidate s (visit's per-iter body). The budget is charged in
                    // one lump (budget-=v) after the v sims are scheduled — matching sequential_halving's
                    // `visit(s, v); budget -= v;` (n_spent is incremented per sim in on_sim_complete).
                    cur_root_slot_ = s;
                    cur_world_ = src_.sample_world(bw_);  // visit(): w = src.sample_world(bw)
                    cur_k_ = OutcomeIndex{0};
                    sim_total_ = 0.0;
                    sh_action_done_ += SimBudget{1};
                    if (sh_action_done_ == v) {
                        sh_budget_ = SimBudget{sh_budget_.value() - v.value()};  // budget draws down
                        ++sh_phase_idx_;
                        sh_action_done_ = SimBudget{0};
                    }
                    continue;
                }
                ++sh_phase_idx_;  // defensive; action_done resets at v
                sh_action_done_ = SimBudget{0};
                continue;
            }
            // the per-action loop is done (ran out of candidates or broke on v<=0): ALWAYS cut the worst
            // half by g+logit+σ·q̂ (sequential_halving's per-phase cut, runs regardless of budget).
            double sigma = p_.sh_cut_sigma(nodes_[0]);
            std::vector<std::pair<double, SlotIndex>> keyed;
            keyed.reserve(considered_.size());
            for (SlotIndex s : considered_) {
                double key = g_[static_cast<size_t>(s.value())] + root_logits_[static_cast<size_t>(s.value())] +
                             sigma * nodes_[0].q(s);
                keyed.emplace_back(key, s);
            }
            std::stable_sort(keyed.begin(), keyed.end(),
                             [](const std::pair<double, SlotIndex>& a, const std::pair<double, SlotIndex>& b) {
                                 return a.first > b.first;
                             });
            int keep = std::max(1, static_cast<int>(keyed.size()) / 2);  // keep is a CandidateCount (survivors)
            std::vector<SlotIndex> next;
            next.reserve(static_cast<size_t>(keep));
            for (int i = 0; i < keep; ++i) next.push_back(keyed[static_cast<size_t>(i)].second);
            considered_ = std::move(next);
            sh_phase_active_ = false;  // re-check the while condition at the top next iteration
            continue;
        }

        // --- the full-budget remainder loop (round-robin on the survivors) ---
        if (sh_budget_ > SimBudget{0} && !considered_.empty()) {
            SlotIndex s = considered_[sh_rr_ % considered_.size()];
            cur_root_slot_ = s;
            cur_world_ = src_.sample_world(bw_);
            cur_k_ = OutcomeIndex{0};
            sim_total_ = 0.0;
            sh_budget_ = SimBudget{sh_budget_.value() - 1};  // budget draws down
            ++sh_rr_;
            continue;
        }
        // SH complete: survivor is considered_.front().
        decision_.survivor_slot = considered_.front();  // typed SlotIndex (the SH survivor)
        finalize();
        phase_ = Phase::Done;
        return CursorDecided{decision_};
    }
}

// ---- pump the descend stack one node at a time until a park or the descent completes --------------
Step TreeCursor::pump_descent() {
    for (;;) {
        assert(!descend_stack_.empty());
        DescendFrame& f = descend_stack_.back();
        GumbelNode& node = nodes_[static_cast<size_t>(f.node)];
        // BY-REFERENCE: the deepest frame's (loc, bw, collected, world) live in the single descent_*
        // members (NOT the frame). f is the back() frame, so descent_* IS its state (the descent narrowed
        // them in place to this depth). No per-frame belief copy is read here.

        // descend()'s top: depth>=max_depth || empty(bw). ACL: cfg_.max_depth is a raw int (GumbelConfig).
        if (f.depth >= PlyDepth{static_cast<SearchRep>(p_.cfg_.max_depth)} || p_.env_.empty(descent_bw_)) {
            if (!node.evaluated) {
                if (p_.env_.empty(descent_bw_)) {
                    // empty belief: return -lam*exit_cost (NO leaf, NO eval).
                    double val = -lam_ * p_.env_.exit_cost(descent_loc_.pt);
                    descend_stack_.pop_back();
                    unwind_with(val);
                    return CursorDecided{};  // descent done (no-leaf return); drive() handles c_outcome
                }
                // depth-cap leaf eval: park.
                (void)p_.eval_build_features(node, descent_loc_, descent_bw_, descent_coll_, ws_);
                parked_ = true;
                return CursorNeedsLeaf{ws_.feat32};
            }
            // evaluated already: return node.value (NO leaf).
            double val = node.value;
            descend_stack_.pop_back();
            unwind_with(val);
            return CursorDecided{};
        }

        // not at the boundary: if unevaluated, this is the first visit -> eval leaf (park). descend()
        // returns node.value immediately after this eval (the leaf estimate; no W/N touch here).
        if (!node.evaluated) {
            (void)p_.eval_build_features(node, descent_loc_, descent_bw_, descent_coll_, ws_);
            parked_ = true;
            return CursorNeedsLeaf{ws_.feat32};
        }

        // evaluated interior node: puct_select an action, apply it, descend into the child (push a frame),
        // OR (terminate) compute the return and unwind. This is descend()'s interior body.
        SlotIndex a = p_.puct_select(node);
        Action act = action_of_slot(p_.env_, a);
        if (act.kind == ActionKind::Terminate) {
            double ret = -lam_ * p_.env_.exit_cost(descent_loc_.pt);  // stop now: only the exit toll
            // descend()'s tail on THIS node: cur.W[a]+=ret; cur.N[a]+=1; return ret. (The terminate edge's
            // own W/N is on this node, then the value backs up to the parents as their step+cont.)
            node.W[static_cast<size_t>(a.value())] += ret;
            node.N[static_cast<size_t>(a.value())] += VisitCount{1};
            descend_stack_.pop_back();
            unwind_with(ret);
            return CursorDecided{};
        }
        // step the action: NARROW descent_* IN PLACE to the child state (the parent's state is never
        // re-read — see the by-reference note). env.apply mutates descent_loc_/bw_/coll_ to the child;
        // descent_world_ is unchanged (descend reuses `world` down the chain). This is the per-frame 2 KiB
        // belief COPY the by-reference design removes: apply writes the narrowed belief back in place.
        StepResult sr = p_.env_.apply(descent_loc_, descent_bw_, descent_coll_, act, descent_world_);
        double step = sr.reward - lam_ * sr.dt;
        std::tuple<SlotIndex, GBeliefKey> ckey{a, gumbel_belief_key(p_.env_, descent_bw_)};
        int child;
        auto cit = node.children.find(ckey);
        if (cit == node.children.end()) {
            nodes_.emplace_back(n_slots_);
            child = static_cast<int>(nodes_.size()) - 1;
            // emplace_back may reallocate nodes_; `node`/`f` references are now DANGLING. Re-index the
            // parent by f.node (a stable arena index) — do NOT touch `node`/`f` after this point.
            nodes_[static_cast<size_t>(f.node)].children[ckey] = child;
        } else {
            child = cit->second;
        }
        // record this frame's chosen action + step for the W/N backup on unwind (descend's tail).
        descend_stack_.back().action_slot = a;
        descend_stack_.back().step = step;
        descend_stack_.back().stepped = true;
        // push the child frame at depth+1; its (loc, bw, collected, world) ARE the just-narrowed descent_*.
        DescendFrame cf;
        cf.node = child;
        cf.depth = descend_stack_.back().depth + SearchRep{1};  // PlyDepth affine: depth + 1 ply
        descend_stack_.push_back(cf);
        // loop: pump the new top frame.
    }
}

// ---- back a child's `cont` up the descend chain (descend's W/N tail) -------------------------------
// The just-popped frame returned `cont` (a leaf value, an empty-belief exit cost, a terminate cost, or a
// cached node.value). Each parent frame on the stack chose an action (action_slot) and applied a step
// (descend's interior body), and awaits ret = step + cont, applying cur.W[a]+=ret; cur.N[a]+=1 (descend's
// tail). When the stack empties, `cont` is the descend(child, depth=1) return for the current c_outcome
// determinization, which simulate_root_action adds to its running total (the root step was already folded
// into sim_total_ when the root-child frame was pushed).
void TreeCursor::unwind_with(double cont) {
    while (!descend_stack_.empty()) {
        DescendFrame& parent = descend_stack_.back();
        double ret = parent.step + cont;  // descend(): ret = step + cont
        nodes_[static_cast<size_t>(parent.node)].W[static_cast<size_t>(parent.action_slot.value())] += ret;
        nodes_[static_cast<size_t>(parent.node)].N[static_cast<size_t>(parent.action_slot.value())] += VisitCount{1};
        descend_stack_.pop_back();
        cont = ret;  // this frame's ret becomes its parent's cont
    }
    sim_total_ += cont;  // the descend(child,depth=1) return for this determinization
}

// ---- visit's W/N tail: a finished sim's return backs up to the root --------------------------------
void TreeCursor::on_sim_complete(double ret) {
    // visit(): nodes[0].W[slot] += ret; nodes[0].N[slot] += 1. cur_root_slot_ is set while mid-sim.
    assert(cur_root_slot_.has_value() && "on_sim_complete with no current root slot (driver invariant)");
    const SlotIndex slot = *cur_root_slot_;
    nodes_[0].W[static_cast<size_t>(slot.value())] += ret;
    nodes_[0].N[static_cast<size_t>(slot.value())] += VisitCount{1};
    n_spent_ += SimBudget{1};
    // clear the per-sim cursor; the SH schedule pulls the next sim (typed absence over the -1 sentinel).
    cur_root_slot_.reset();
    cur_k_ = OutcomeIndex{0};
    sim_total_ = 0.0;
}

// ---- finalize: improved-π + executed action + the no-early-exit substitution ----------------------
void TreeCursor::finalize() {
    decision_.n_spent = n_spent_;  // SimBudget -> the SAME domain (no cast at the store)
    decision_.improved = p_.improved_policy(nodes_[0], root_logits_);
    // survivor_slot is the typed-SlotIndex Decision field set by drive()/resume(); it is never nullopt
    // here on a non-empty belief (drive() always sets a real survivor — typed presence over the -1 sentinel).
    assert(decision_.survivor_slot.has_value() &&
           "gumbel(cursor): SH returned no survivor on a non-empty belief");
    decision_.action = action_of_slot(p_.env_, *decision_.survivor_slot);

    // HPO/BENCHMARK-ONLY no-early-exit substitution (cfg_.no_early_exit; default false -> skipped, the
    // decision byte-unchanged). Identical to run_search's tail.
    if (p_.cfg_.no_early_exit && decision_.action.kind == ActionKind::Terminate) {
        std::optional<SlotIndex> best_slot;  // typed absence over the -1 "no non-terminate option" sentinel
        double best_pi = -1.0;
        for (SlotIndex s : nodes_[0].legal_slots) {
            if (s == term_slot_) continue;
            const double pi = decision_.improved[static_cast<size_t>(s.value())];
            if (pi > best_pi) {
                best_pi = pi;
                best_slot = s;
            }
        }
        if (best_slot.has_value()) {
            decision_.survivor_slot = best_slot;  // typed SlotIndex (the substituted survivor)
            decision_.action = action_of_slot(p_.env_, *best_slot);
        }
    }
}

}  // namespace chocofarm
