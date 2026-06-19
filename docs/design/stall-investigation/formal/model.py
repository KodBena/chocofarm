#!/usr/bin/env python3
"""
~/w/vdc/chocobo/runs/formal-stall/model.py

FORMAL MODEL (bounded model checking in Z3) of the C++<->Python leaf-evaluation
flow-control protocol:

  PRODUCER  = cpp/src/runner_wire_batched.cpp :: run_episodes_wire_pipelined
              (the non-blocking, D-in-flight pipelined driver) + the strict
              barrier variant for contrast.
  TRANSPORT = cpp/include/chocofarm/wire_leaf_pool.hpp :: submit_batch / recv_batch
              (DEALER socket, corr-id -> ordered-slot map, RCVTIMEO, NO SNDTIMEO/HWM).
  SERVER    = cpp/stage_a/stage_a_server.py :: StageAServer._serve_batch (group/leaf
              wakeup, bucket/padmax E-policy) over the base InferenceServer._drain
              (greedy drain up to max_batch ROWS, ONE forward per drained group).

We model the CONTROL / flow state machine only. The leaf features, the NN
forward's numeric values, and the search internals are ABSTRACTED to "a leaf
eval is requested / a result returns". We KEEP: each slot's lifecycle, the
per-thread driver loop, the in-flight MESSAGE cap D, the corr-id<->slot-set map
(as message objects), the producer->server request channel and server->producer
reply channel (bounded ZMQ pipes), and the server's accumulate -> one-forward ->
scatter.

This file builds the transition relation and unrolls it to depth K, asking Z3
for a reachable STUCK state (deadlock) where every process is blocked but work
remains. See report for the abstraction ledger.
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass

from z3 import (
    And, Bool, If, Implies, Int, Not, Or, Solver, Sum, sat, unsat,
)


# ----------------------------------------------------------------------------
# Parameters of a configuration. (N drives the slot count K via the SAME
# derivation the C++ uses: K = ceil(pool_batch / pool_threads); here we model
# ONE producer thread with K slots directly and let the test sweep K, plus the
# in-flight message cap D, plus the server max_batch cap in ROWS.)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    K: int            # slots in the (single modelled) producer thread
    D: int            # per-thread in-flight MESSAGE cap (--inflight-msgs)
    plies: int        # how many leaf evals each slot needs before it finalizes
    pipe_cap: int     # ZMQ default-HWM abstraction: request-channel capacity
                      # (messages the producer can have queued at the server +
                      #  in transit). Large => effectively unbounded send.


# ----------------------------------------------------------------------------
# STATE VARIABLES (per step t). We model ONE producer thread and ONE server,
# which is sufficient to expose a producer<->server mutual-wait: more threads
# only add senders, which can only HELP the server make progress, never create
# a wait the single-thread case lacks. (Documented assumption A6.)
#
# Producer slot lifecycle, per slot s in 0..K-1:
#   sl_remaining[s] : leaf evals still owed before the slot's episode finalizes
#                     (>0 => slot still has work; 0 => exhausted/idle).
#   sl_state[s]     : 0 = PARKED (ready: parked at a leaf, not submitted)
#                     1 = SUBMITTED (its leaf is outstanding to the server)
#                     2 = IDLE (remaining==0; no work; never re-parks)
#   A slot is "ready" iff state==PARKED and remaining>0 (mirrors is_ready()).
#
# Producer message accounting:
#   inflight : number of messages the producer has outstanding (== inflight_msgs).
#
# Channels (corr-id messages carry an ordered slot-set; we abstract the set to a
# COUNT of leaves it carries, which is all the flow-control depends on):
#   req_pending  : leaf-eval messages sitting in the request channel, not yet
#                  drained by the server (each is one corr-id / one forward-group
#                  member). Modelled as a list of leaf-counts per message.
#   rep_pending  : reply messages sitting in the reply channel, not yet recv'd
#                  by the producer. Each reply answers exactly one request msg.
#
# Server:
#   srv_state : 0 = BLOCKED (in _drain's bounded poll, waiting for >=1 request)
#               1 = (transient drain/forward — collapsed into one atomic step)
#
# The producer's recv_batch BLOCKS on RCVTIMEO; the server's _drain BLOCKS on
# its bounded poll. The DEADLOCK predicate: producer blocked in recv_batch AND
# server blocked in _drain AND unresolved work remains (req or rep pending, or a
# submitted/parked slot with remaining>0).
# ----------------------------------------------------------------------------

# Producer program counter (which blocking statement the single producer thread
# is at). We model the driver as: it issues messages up to D, then blocks in
# recv_batch until a reply arrives. The atomic "step" granularity is one driver
# action. PC values:
PC_ISSUE = 0   # about to run issue_one() refill loop (can send if ready & inflight<D)
PC_RECV  = 1   # blocked in recv_batch() (inflight>0, waiting for a reply)
PC_DONE  = 2   # loop exited (inflight==0 and nothing ready) — producer finished

# We encode the request channel as a fixed-capacity array of "leaf counts"
# (0 = empty slot in the array). Same for the reply channel. pipe_cap bounds it.


def build_and_check(cfg: Config, max_steps: int, want_deadlock: bool = True):
    """Unroll the transition relation to `max_steps` and SAT-check for a
    reachable state satisfying the DEADLOCK predicate. Returns (status, model,
    solver) where status in {'deadlock','safe-bounded'}."""
    K, D, P, CAP = cfg.K, cfg.D, cfg.plies, cfg.pipe_cap
    s = Solver()

    # state[t] dictionaries
    def mk(t):
        st = {}
        for j in range(K):
            st[f"rem{j}"] = Int(f"rem_{t}_{j}")     # remaining leaves for slot j
            st[f"ss{j}"]  = Int(f"ss_{t}_{j}")      # slot state 0/1/2
        st["inflight"] = Int(f"inflight_{t}")
        st["pc"]       = Int(f"pc_{t}")
        st["srv"]      = Int(f"srv_{t}")
        # request channel: CAP message-slots, each an int leaf-count (0=empty)
        for c in range(CAP):
            st[f"req{c}"] = Int(f"req_{t}_{c}")
        # reply channel: at most D replies outstanding; CAP slots is plenty
        for c in range(CAP):
            st[f"rep{c}"] = Int(f"rep_{t}_{c}")
        return st

    S = [mk(t) for t in range(max_steps + 1)]

    # ---- INITIAL STATE (mirrors the prime: fill K slots, each parked with P
    # leaves owed; issue_one has NOT run yet; channels empty; server blocked) ----
    s0 = S[0]
    for j in range(K):
        s.add(s0[f"rem{j}"] == P)
        s.add(s0[f"ss{j}"] == 0)        # PARKED (ready)
    s.add(s0["inflight"] == 0)
    s.add(s0["pc"] == PC_ISSUE)
    s.add(s0["srv"] == 0)               # BLOCKED in _drain
    for c in range(CAP):
        s.add(s0[f"req{c}"] == 0)
        s.add(s0[f"rep{c}"] == 0)

    # domain constraints for every step
    for t in range(max_steps + 1):
        st = S[t]
        for j in range(K):
            s.add(st[f"rem{j}"] >= 0, st[f"rem{j}"] <= P)
            s.add(st[f"ss{j}"] >= 0, st[f"ss{j}"] <= 2)
        s.add(st["inflight"] >= 0, st["inflight"] <= CAP)
        s.add(st["pc"] >= 0, st["pc"] <= 2)
        s.add(st["srv"] >= 0, st["srv"] <= 1)
        for c in range(CAP):
            s.add(st[f"req{c}"] >= 0, st[f"rep{c}"] >= 0)

    # helper predicates over a state dict ------------------------------------
    def ready_count(st):
        # slots that are PARKED and have remaining>0 (is_ready())
        return Sum([If(And(st[f"ss{j}"] == 0, st[f"rem{j}"] > 0), 1, 0)
                    for j in range(K)])

    def req_used(st):
        return Sum([If(st[f"req{c}"] > 0, 1, 0) for c in range(CAP)])

    def rep_used(st):
        return Sum([If(st[f"rep{c}"] > 0, 1, 0) for c in range(CAP)])

    def any_req(st):
        return Or([st[f"req{c}"] > 0 for c in range(CAP)])

    def any_rep(st):
        return Or([st[f"rep{c}"] > 0 for c in range(CAP)])

    # ---- TRANSITION RELATION -------------------------------------------------
    # At each step EXACTLY ONE of the following actions fires (an interleaving
    # semantics). Each action is guarded; the conjunction of "no action's guard
    # holds" is the STUCK condition we will separately assert reachable.
    #
    # Actions:
    #  (PI) PRODUCER_ISSUE   : pc==ISSUE, ready>0, inflight<D, room in req channel
    #                          -> coalesce ALL ready slots into ONE message
    #                             (count = ready_count), mark them SUBMITTED,
    #                             inflight += 1, append one req message.
    #                             Mirrors issue_one() + the refill while-loop.
    #  (PD) PRODUCER_ISSUE_DONE: pc==ISSUE, (ready==0 OR inflight==D)
    #                          -> if inflight>0: pc=RECV ; else pc=DONE.
    #                             Mirrors falling out of the refill loop into the
    #                             recv (or exiting the outer while when inflight==0).
    #  (PR) PRODUCER_RECV    : pc==RECV, a reply is available (any_rep)
    #                          -> consume ONE reply (the oldest), decrement
    #                             inflight, RESUME its slots: each answered slot's
    #                             remaining -=1; if now 0 -> IDLE, else -> PARKED.
    #                             pc=ISSUE (re-enter refill). Mirrors recv_batch +
    #                             the for-Completion resume + advance/fill, then the
    #                             "refill the pipe" while-loop.
    #  (SD) SERVER_DRAIN     : srv==0 (blocked), any_req
    #                          -> drain ALL queued req messages up to max_batch
    #                             ROWS (we model max_batch >= K so the cap never
    #                             binds for these small K — documented A4), run ONE
    #                             forward, scatter: for EACH drained req message
    #                             append ONE reply (same leaf-count) to rep channel.
    #                             srv stays 0 (loops back to _drain).
    #
    # The server is BLOCKED (cannot fire SD) exactly when no req is pending. The
    # producer is BLOCKED in recv (cannot fire PR) exactly when pc==RECV and no
    # rep is pending. THAT mutual condition, with work remaining, is the deadlock.

    trans = []
    for t in range(max_steps):
        a, b = S[t], S[t + 1]

        def frame_except(changed):
            """all vars not in `changed` are copied a->b"""
            cons = []
            allkeys = list(a.keys())
            for k in allkeys:
                if k not in changed:
                    cons.append(b[k] == a[k])
            return cons

        # --- (PI) PRODUCER_ISSUE ---
        rc = ready_count(a)
        # find first empty req slot index expression: we append into the lowest
        # empty channel slot. Encode via a chain of If.
        pi_changed = {f"req{c}" for c in range(CAP)} | {f"ss{j}" for j in range(K)} | {"inflight"}
        pi_guard = And(a["pc"] == PC_ISSUE, rc > 0, a["inflight"] < D,
                       req_used(a) < CAP)
        pi_body = []
        # mark all ready slots submitted
        for j in range(K):
            pi_body.append(
                b[f"ss{j}"] == If(And(a[f"ss{j}"] == 0, a[f"rem{j}"] > 0), 1, a[f"ss{j}"]))
        pi_body.append(b["inflight"] == a["inflight"] + 1)
        # append message with leaf-count rc into the first empty req slot
        for c in range(CAP):
            earlier_full = And(*[a[f"req{cc}"] > 0 for cc in range(c)]) if c > 0 else True
            is_target = And(earlier_full, a[f"req{c}"] == 0)
            pi_body.append(b[f"req{c}"] == If(is_target, rc, a[f"req{c}"]))
        PI = And(pi_guard, And(*pi_body), And(*frame_except(pi_changed)))

        # --- (PD) PRODUCER_ISSUE_DONE ---
        pd_changed = {"pc"}
        pd_guard = And(a["pc"] == PC_ISSUE, Or(rc == 0, a["inflight"] == D))
        pd_body = [b["pc"] == If(a["inflight"] > 0, PC_RECV, PC_DONE)]
        PD = And(pd_guard, And(*pd_body), And(*frame_except(pd_changed)))

        # --- (PR) PRODUCER_RECV --- consume the OLDEST reply (lowest non-empty)
        pr_changed = {f"rep{c}" for c in range(CAP)} | {f"rem{j}" for j in range(K)} \
            | {f"ss{j}" for j in range(K)} | {"inflight", "pc"}
        pr_guard = And(a["pc"] == PC_RECV, any_rep(a))
        # The reply consumed is the first non-empty rep slot; its leaf-count tells
        # how many slots it answers. We abstract WHICH slots: a reply answers the
        # SUBMITTED slots. Since all ready slots coalesce into one msg per issue,
        # and replies are consumed in order, the count matches the #submitted at
        # the time. For the flow-control deadlock we only need: consuming a reply
        # decrements inflight by 1 and resumes (count) submitted slots. We resume
        # ALL currently-SUBMITTED slots whose reply this is. To keep the model
        # sound w.r.t. counts we resume min(count, #submitted) slots — modelled by
        # resuming every submitted slot when this is the only outstanding message,
        # which is the regime the deadlock lives in (inflight collapses to 1; see
        # report A5). We encode the common, faithful case: resume ALL submitted.
        pr_body = []
        for j in range(K):
            was_sub = a[f"ss{j}"] == 1
            new_rem = If(was_sub, a[f"rem{j}"] - 1, a[f"rem{j}"])
            pr_body.append(b[f"rem{j}"] == new_rem)
            # submitted -> if remaining now 0: IDLE(2) else PARKED(0); else unchanged
            pr_body.append(
                b[f"ss{j}"] == If(was_sub, If(new_rem > 0, 0, 2), a[f"ss{j}"]))
        pr_body.append(b["inflight"] == a["inflight"] - 1)
        pr_body.append(b["pc"] == PC_ISSUE)
        # remove the oldest reply: shift is unnecessary; just clear the first
        # non-empty slot.
        for c in range(CAP):
            earlier_empty = And(*[a[f"rep{cc}"] == 0 for cc in range(c)]) if c > 0 else True
            is_oldest = And(earlier_empty, a[f"rep{c}"] > 0)
            pr_body.append(b[f"rep{c}"] == If(is_oldest, 0, a[f"rep{c}"]))
        PR = And(pr_guard, And(*pr_body), And(*frame_except(pr_changed)))

        # --- (SD) SERVER_DRAIN --- drain ALL req messages, emit one reply each
        sd_changed = {f"req{c}" for c in range(CAP)} | {f"rep{c}" for c in range(CAP)}
        sd_guard = And(a["srv"] == 0, any_req(a))
        sd_body = []
        # clear all req slots; for each previously-nonempty req, append a reply.
        # We append replies preserving order into the reply channel after existing
        # replies. Number of new replies = req_used(a). To keep it simple and
        # bounded we require the reply channel has room (req_used <= free rep slots);
        # CAP is sized so this holds (A4).
        # Compute, for each rep slot, whether it receives a drained req.
        # Strategy: replies are placed into rep slots starting at the first empty
        # rep slot, in req order. We unroll this placement.
        # Gather req leaf-counts in order:
        req_counts = [a[f"req{c}"] for c in range(CAP)]
        # number of existing replies = rep_used(a). We append after them.
        # For each output rep position p, its value = either existing rep, or the
        # (p - rep_used)-th nonzero req count. This is complex to encode directly;
        # instead we use the regime where the reply channel is EMPTY whenever the
        # server drains (true here: the producer consumes replies before it can
        # issue again, and the server only has reqs to drain after the producer
        # issued — documented A7). So we place replies at positions 0..; the value
        # at rep position p equals the p-th nonzero req count (compacted).
        # Compact req_counts -> for output position p, the p-th positive value.
        def pth_positive(p):
            # returns an Int expr: the value of the p-th (0-indexed) positive entry
            # in req_counts, or 0 if fewer than p+1 positives.
            expr = 0
            # build: count positives before each index
            for c in range(CAP):
                # number of positives strictly before c
                before = Sum([If(req_counts[cc] > 0, 1, 0) for cc in range(c)])
                expr = If(And(req_counts[c] > 0, before == p), req_counts[c], expr)
            return expr
        for c in range(CAP):
            sd_body.append(b[f"req{c}"] == 0)
        for p in range(CAP):
            sd_body.append(b[f"rep{p}"] == If(a[f"rep{p}"] > 0, a[f"rep{p}"], pth_positive(p)))
        SD = And(sd_guard, And(*sd_body), And(*frame_except(sd_changed)))

        trans.append(Or(PI, PD, PR, SD))

    for t in range(max_steps):
        s.add(trans[t])

    # ---- DEADLOCK predicate at the FINAL step (we ask: is the final state a
    # genuine stuck state with work remaining?). Because EXACTLY ONE action must
    # fire at each step, if we instead require that NO action can fire from a
    # state, that state is terminal. We assert reachability of such a state with
    # work remaining. Encode by: at step max_steps, no guard holds AND work
    # remains. ----
    last = S[max_steps]
    rc_last = ready_count(last)
    no_PI = Not(And(last["pc"] == PC_ISSUE, rc_last > 0, last["inflight"] < D,
                    req_used(last) < CAP))
    no_PD = Not(And(last["pc"] == PC_ISSUE, Or(rc_last == 0, last["inflight"] == D)))
    no_PR = Not(And(last["pc"] == PC_RECV, any_rep(last)))
    no_SD = Not(And(last["srv"] == 0, any_req(last)))
    stuck = And(no_PI, no_PD, no_PR, no_SD)

    # "work remains": some slot still owes leaves, OR a message is in a channel,
    # OR a slot is submitted/parked with remaining>0.
    work_remains = Or(
        any_req(last), any_rep(last), last["inflight"] > 0,
        Or([And(last[f"rem{j}"] > 0, last[f"ss{j}"] != 2) for j in range(K)]),
    )

    # We are NOT counting PC_DONE-with-no-work as a deadlock: that is clean
    # termination. The deadlock is stuck AND work_remains AND not(clean done).
    s.add(stuck)
    s.add(work_remains)
    s.add(Not(last["pc"] == PC_DONE))  # PC_DONE with inflight==0 is clean exit

    res = s.check()
    return res, (s.model() if res == sat else None), S


def render_trace(model, S, cfg):
    K, CAP = cfg.K, cfg.pipe_cap
    pcname = {0: "ISSUE", 1: "RECV", 2: "DONE"}
    ssname = {0: "PARK", 1: "SUBMIT", 2: "IDLE"}
    lines = []
    for t, st in enumerate(S):
        def g(k):
            v = model.evaluate(st[k], model_completion=True)
            return v.as_long()
        slots = " ".join(f"s{j}[{ssname[g(f'ss{j}')]},rem={g(f'rem{j}')}]" for j in range(K))
        reqs = [g(f"req{c}") for c in range(CAP)]
        reps = [g(f"rep{c}") for c in range(CAP)]
        reqs = [x for x in reqs if x > 0]
        reps = [x for x in reps if x > 0]
        lines.append(
            f"  t={t:2d} pc={pcname[g('pc')]:5s} inflight={g('inflight')} "
            f"srv={'BLOCKED' if g('srv')==0 else 'drain'} | {slots} | "
            f"req={reqs} rep={reps}")
    return "\n".join(lines)


if __name__ == "__main__":
    # default single run; the sweep is in sweep.py
    cfg = Config(K=int(sys.argv[1]) if len(sys.argv) > 1 else 2,
                 D=int(sys.argv[2]) if len(sys.argv) > 2 else 8,
                 plies=int(sys.argv[3]) if len(sys.argv) > 3 else 2,
                 pipe_cap=int(sys.argv[4]) if len(sys.argv) > 4 else 6)
    depth = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    res, model, S = build_and_check(cfg, depth)
    print(f"config={cfg} depth={depth} -> {res}")
    if res == sat:
        print("DEADLOCK reachable. Counterexample trace:")
        print(render_trace(model, S, cfg))
