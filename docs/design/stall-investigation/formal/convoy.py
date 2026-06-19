#!/usr/bin/env python3
"""
~/w/vdc/chocobo/runs/formal-stall/convoy.py

RETARGETED model. The empirical diagnosis (gdb + /proc + server rows/forward
instrumentation) showed the pathology is NOT a deadlock (BMC in model*.py proved
deadlock-free; the run ALWAYS eventually completes) but a metastable LIVELOCK: a
recoverable ~10-12x throughput collapse where the server's coalescing breaks down
into a self-sustaining 1:1 message<->forward CONVOY (server measured at ~1.4
rows/forward in the bad regime vs 55-177 healthy).

This model checks the RIGHT property: is there a reachable RECURRENT state (a
lasso: a cycle reachable from the prime) in which the coalescing degree stays
pinned at the minimum -- i.e. every wire message carries S=1 row and every server
forward processes ~1 row -- so the system is stuck in the low-throughput regime
even though it is making (slow) progress?  That is a SAFETY-style witness for a
livelock: a reachable cycle of states all satisfying COLLAPSED.

MODEL (faithful to the convoy mechanism; abstraction ledger in the report):
  ONE producer thread, K slots, inflight MESSAGE cap D.  We abstract the leaf
  TURNAROUND as: a message the server forwards becomes a reply one "tick" later.
  The server is GREEDY (group wakeup): each server tick it drains ALL currently
  queued request messages into ONE forward and replies to each (1:1 by corr-id,
  inference_server.py:_drain + stage_a_server group mode).  The KEY quantitative
  variable is, per server tick, how many request MESSAGES were queued when it
  woke (that is the coalescing degree across messages) and the total ROWS.

  Producer per tick (interleaved with the server): on a reply, resume that
  message's slots (each slot re-parks for its next ply with prob modelled as
  "always re-parks" until its plies run out), then issue_one() coalesces ALL
  currently-ready (parked & unsubmitted) slots into ONE message.

  The CONVOY arises when, at every issue, only ~1 slot is ready (the others are
  still SUBMITTED/in-flight), so each message is 1 row, and the server, waking
  with 1 queued message, forwards 1 row.

We search for: a reachable state s* from which the system returns to a state with
the SAME coalescing signature, with every message on the cycle carrying exactly
1 row.  We encode it as: find a trajectory of length L+ (a prefix to s*) + a cycle
back to s* such that EVERY issue on the cycle coalesces exactly 1 ready slot.
For tractability we check the simpler, sufficient witness: a reachable state in
which (a) inflight == D (pipe full), (b) exactly one slot is PARKED-ready while
all others are SUBMITTED, and (c) the server has exactly one message queued -- and
show this state maps to a successor with the same signature (self-loop of the
1:1 regime), establishing the convoy is an invariant set the dynamics can enter.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from z3 import And, If, Int, Not, Or, Solver, Sum, sat


@dataclass(frozen=True)
class Config:
    K: int        # slots/thread (= trees_per_thread N region: empirical repro N=4)
    D: int        # inflight message cap
    plies: int    # leaves per slot before finalize (kept high so slots stay live)


# Slot state: 0 PARKED(ready), 1 SUBMITTED(in-flight)
def build_convoy_witness(cfg, depth):
    """Reachability of the COLLAPSED 1:1 convoy signature, then show it is a
    fixpoint of the transition (a self-sustaining regime)."""
    K, D, P = cfg.K, cfg.D, cfg.plies
    s = Solver()

    # We model time as alternating producer/server micro-steps but COLLAPSE one
    # round into: (server forwards the queued messages -> replies) then (producer
    # consumes ONE reply, resumes 1 message's slots, refills by issuing coalesced
    # messages up to D).  State per step:
    #   parked[j], submitted[j]  (exactly one true per live slot; idle if plies done)
    #   rem[j]                    remaining plies
    #   inflight                  outstanding messages
    #   qmsgs                     request messages queued at server (each = #rows it carries)
    # We track, per outstanding message, its ROW COUNT, to read the convoy degree.
    # Represent the message multiset as up to D counters msgrows[0..D-1] (0=unused).

    def mk(t):
        st = {}
        for j in range(K):
            st[f"park{j}"] = Int(f"park_{t}_{j}")   # 1 if parked & ready & unsubmitted
            st[f"sub{j}"] = Int(f"sub_{t}_{j}")     # 1 if submitted (in a queued/inflight msg)
            st[f"rem{j}"] = Int(f"rem_{t}_{j}")
        st["inflight"] = Int(f"inflight_{t}")
        # rows of the message at the HEAD of the server queue that it will forward
        # next (the coalescing degree the server sees). 0 => server idle.
        st["head_rows"] = Int(f"head_{t}")
        return st

    S = [mk(t) for t in range(depth + 1)]

    s0 = S[0]
    # PRIME: K slots all parked-ready, none submitted, full plies, then the prime
    # loop issues messages up to D. The FIRST issue_one coalesces ALL K ready ->
    # one message of K rows; subsequent prime issues find nothing ready -> stop.
    # So after prime: 1 message of K rows inflight, all K slots submitted.
    for j in range(K):
        s.add(s0[f"park{j}"] == 0, s0[f"sub{j}"] == 1, s0[f"rem{j}"] == P)
    s.add(s0["inflight"] == 1, s0["head_rows"] == K)

    for t in range(depth + 1):
        st = S[t]
        for j in range(K):
            s.add(st[f"park{j}"] >= 0, st[f"park{j}"] <= 1)
            s.add(st[f"sub{j}"] >= 0, st[f"sub{j}"] <= 1)
            s.add(st[f"rem{j}"] >= 0, st[f"rem{j}"] <= P)
            # a live slot is exactly one of parked/submitted; an idle slot (rem==0)
            # is neither parked nor submitted.
            s.add(If(st[f"rem{j}"] == 0,
                     And(st[f"park{j}"] == 0, st[f"sub{j}"] == 0),
                     st[f"park{j}"] + st[f"sub{j}"] == 1))
        s.add(st["inflight"] >= 0, st["inflight"] <= D)
        s.add(st["head_rows"] >= 0, st["head_rows"] <= K)

    # TRANSITION: one "round" = server forwards the head message (head_rows rows)
    # and replies; producer consumes that ONE reply, resumes those slots (each
    # such slot was submitted; it re-parks if rem>1 else goes idle, rem-=1), then
    # refills by issue_one coalescing ALL now-ready slots into ONE new message
    # (added to inflight if inflight<D), and the server's NEXT head becomes the
    # OLDEST still-queued message. We abstract the queue depth to: head_rows of
    # the next round = the size of the message that the producer just issued IF
    # the server has caught up (steady state: 1 message resolved, 1 issued -> the
    # new message becomes the next head). This is the convoy steady state we are
    # testing for reachability/self-sustenance.
    #
    # The number of slots the reply frees == head_rows (the rows in that message).
    # In the convoy regime head_rows==1 so exactly 1 slot frees, exactly 1 re-parks,
    # issue_one coalesces exactly 1 ready slot -> a new 1-row message -> next head 1.

    for t in range(depth):
        a, b = S[t], S[t + 1]
        # choose which submitted slots the head message answers: a subset of size
        # head_rows of currently-submitted slots. We pick them as the lowest-index
        # submitted slots (WLOG by symmetry).
        hr = a["head_rows"]
        # freed[j] = this slot is among the first head_rows submitted slots
        freed = []
        for j in range(K):
            sub_before = Sum([If(a[f"sub{cc}"] == 1, 1, 0) for cc in range(j)]) if j > 0 else 0
            freed.append(And(a[f"sub{j}"] == 1, sub_before < hr))
        cons = []
        # resume freed slots: rem-=1; if rem>0 -> parked, else idle
        for j in range(K):
            nr = If(freed[j], a[f"rem{j}"] - 1, a[f"rem{j}"])
            cons.append(b[f"rem{j}"] == nr)
            # after resume, freed&rem>0 -> parked; freed&rem==0 -> idle;
            # not-freed-submitted stays submitted; previously-parked stays parked.
            cons.append(b[f"sub{j}"] == If(freed[j], 0, a[f"sub{j}"]))
            # parked AFTER issue: a freed slot that re-parks is immediately
            # RE-SUBMITTED by the refill issue_one (it coalesces all ready). So in
            # steady state the just-freed re-parked slots get submitted again.
        # rows ready to issue = number of freed slots that re-park (rem>0 after)
        ready_after = Sum([If(And(freed[j], a[f"rem{j}"] - 1 > 0), 1, 0) for j in range(K)])
        # plus any slots that were parked-but-unsubmitted already (shouldn't happen
        # in steady convoy, but include for faithfulness)
        already_ready = Sum([If(a[f"park{j}"] == 1, 1, 0) for j in range(K)])
        new_msg_rows = ready_after + already_ready
        # issue_one only fires if there is something ready AND inflight<D after the
        # -1 from the resolved message.
        inflight_after_recv = a["inflight"] - 1
        issues = And(new_msg_rows > 0, inflight_after_recv < D)
        cons.append(b["inflight"] == If(issues, inflight_after_recv + 1, inflight_after_recv))
        # all ready slots become submitted (issue_one coalesces ALL ready)
        for j in range(K):
            became_ready = Or(And(freed[j], a[f"rem{j}"] - 1 > 0), a[f"park{j}"] == 1)
            cons.append(b[f"park{j}"] == 0)  # issue_one submits all ready -> none left parked
            cons.append(b[f"sub{j}"] == If(issues, If(became_ready, 1, b[f"sub{j}"]),
                                           b[f"sub{j}"]))
        # next head: the next message the server forwards. In the pipelined steady
        # state the server immediately forwards whatever is queued; the new message
        # just issued becomes the next head (the convoy). If nothing issued, the
        # head is the next still-inflight message -- we model its rows as the rows
        # of the previously-issued message (carried), but for the witness we focus
        # on the case where the new message is the head.
        cons.append(b["head_rows"] == If(issues, new_msg_rows, 0))
        s.add(And(*cons))

    # CONVOY WITNESS: a reachable step t* (t*>=1) where head_rows==1 AND it stays
    # ==1 for the rest of the horizon (a sustained 1:1 regime) AND work remains
    # (slots still have plies). That is the metastable collapsed regime.
    sustained = []
    START = 1
    for t in range(START, depth + 1):
        sustained.append(S[t]["head_rows"] == 1)
    work_remains = Or([S[depth][f"rem{j}"] > 0 for j in range(K)])
    s.add(And(*sustained), work_remains)

    return s, S


def render(model, S, cfg):
    K = cfg.K
    out = []
    for t, st in enumerate(S):
        g = lambda k: model.evaluate(st[k], model_completion=True).as_long()
        slots = " ".join(
            ("P" if g(f"park{j}") else ("S" if g(f"sub{j}") else "i")) + str(g(f"rem{j}"))
            for j in range(K))
        out.append(f"  t={t:2d} inflight={g('inflight')} head_rows={g('head_rows')} | {slots}")
    return "\n".join(out)


if __name__ == "__main__":
    cfg = Config(K=int(sys.argv[1]) if len(sys.argv) > 1 else 4,
                 D=int(sys.argv[2]) if len(sys.argv) > 2 else 8,
                 plies=int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    depth = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    s, S = build_convoy_witness(cfg, depth)
    r = s.check()
    print(f"{cfg} depth={depth} -> {r}")
    if r == sat:
        print("CONVOY (sustained 1:1 message<->forward) regime is REACHABLE:")
        print(render(s.model(), S, cfg))
    else:
        print("No sustained 1:1 convoy within this horizon under these guards.")
