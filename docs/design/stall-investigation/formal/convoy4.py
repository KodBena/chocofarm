#!/usr/bin/env python3
"""
~/w/vdc/chocobo/runs/formal-stall/convoy4.py  --  staggered-arrival convoy witness.

convoy3 pinned inflight at 1 because issue_one coalesced all K slots into ONE
message and one reply freed them all together (lockstep). The REAL staggering
(empirically observed) comes from D distinct outstanding messages whose replies
interleave with re-issues, so issue_one frequently sees only ~1 ready slot.

To express that WITHOUT modelling each slot's distinct search latency (which is
the true source of desync), we model the essential consequence directly and
minimally: the system can be in a state with up to D OUTSTANDING 1-ROW messages,
and the schedule (chosen by Z3 = the OS/ZMQ timing) decides, at each server wake,
how many of the queued messages have ARRIVED. The convoy = a schedule where every
wake sees exactly ONE arrived message (rows/forward == 1) yet the pipe stays full
(inflight == D) -- the metastable lockstep.

State (abstract, sufficient for the rows/forward metric):
  inflight      : # outstanding 1-row messages (0..D)
  queued        : # of those that have ARRIVED at the server and await forward
  work          : total leaf-evals still owed across all slots (drives liveness)
Transitions (interleaving; Z3 picks):
  ARRIVE   : inflight>queued -> queued+=1            (a message reaches the server)
  FORWARD  : queued>=1 -> rows/forward = queued; queued=0; those become replies;
             (greedy drain: forwards ALL ARRIVED at the wake instant)
  RECV+ISSUE: a reply pending -> consume it (free 1 slot, work-=1), and issue_one
             ships 1 new 1-row message if work remains and inflight<D (refill).
The CONVOY: a schedule where ARRIVE and FORWARD alternate so each FORWARD sees
queued==1 (rows/forward==1) while inflight stays at D and work remains. The
HEALTHY schedule: let several ARRIVEs accumulate before a FORWARD (queued large).
Both are admissible -> the greedy drain does not FORCE the healthy one. That is
the root cause, stated as a reachability contrast.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from z3 import And, If, Int, Or, Solver, Sum


@dataclass(frozen=True)
class Config:
    D: int          # inflight message cap (per thread)
    work0: int      # total leaves owed (>= a few D so the regime can sustain)


def build(cfg, depth, mode):
    D, W0 = cfg.D, cfg.work0
    s = Solver()
    inflight = [Int(f"inf_{t}") for t in range(depth + 1)]
    queued = [Int(f"q_{t}") for t in range(depth + 1)]
    replies = [Int(f"rep_{t}") for t in range(depth + 1)]
    work = [Int(f"w_{t}") for t in range(depth + 1)]
    fwd_rows = [Int(f"fr_{t}") for t in range(depth + 1)]

    # prime: pipe filled to D outstanding 1-row messages, none arrived yet.
    s.add(inflight[0] == D, queued[0] == 0, replies[0] == 0,
          work[0] == W0, fwd_rows[0] == 0)
    for t in range(depth + 1):
        s.add(inflight[t] >= 0, inflight[t] <= D)
        s.add(queued[t] >= 0, queued[t] <= D)
        s.add(replies[t] >= 0, replies[t] <= D)
        s.add(work[t] >= 0, fwd_rows[t] >= 0)
        s.add(queued[t] <= inflight[t])  # arrived <= outstanding

    for t in range(depth):
        a = (inflight[t], queued[t], replies[t], work[t])
        # ARRIVE
        arrive = And(inflight[t] > queued[t],
                     inflight[t + 1] == inflight[t], queued[t + 1] == queued[t] + 1,
                     replies[t + 1] == replies[t], work[t + 1] == work[t],
                     fwd_rows[t + 1] == 0)
        # FORWARD: forward all currently-arrived (queued) -> they become replies
        forward = And(queued[t] >= 1,
                      inflight[t + 1] == inflight[t], queued[t + 1] == 0,
                      replies[t + 1] == replies[t] + queued[t], work[t + 1] == work[t],
                      fwd_rows[t + 1] == queued[t])
        # RECV+ISSUE: consume one reply, free a slot (work-=1); refill one 1-row
        # message iff work remains after AND inflight<D after the -1.
        issues = And(work[t] - 1 > 0, inflight[t] - 1 < D)
        recv = And(replies[t] >= 1,
                   replies[t + 1] == replies[t] - 1,
                   work[t + 1] == work[t] - 1,
                   inflight[t + 1] == If(issues, inflight[t], inflight[t] - 1),
                   queued[t + 1] == queued[t],
                   fwd_rows[t + 1] == 0)
        s.add(Or(arrive, forward, recv))

    if mode == "convoy":
        # every FORWARD has rows/forward == 1, sustained, pipe stays full, work left
        nfwd1 = Sum([If(fwd_rows[t] == 1, 1, 0) for t in range(1, depth + 1)])
        s.add(nfwd1 >= depth // 3)
        # no forward ever exceeds 1 row (pure convoy)
        for t in range(1, depth + 1):
            s.add(fwd_rows[t] <= 1)
        s.add(work[depth] > 1)
        s.add(inflight[depth] == D)   # the pipe stayed full the whole time
    elif mode == "healthy":
        s.add(Or([fwd_rows[t] >= max(2, D // 2) for t in range(1, depth + 1)]))
    return s, (inflight, queued, replies, work, fwd_rows)


def render(model, vs, depth):
    inflight, queued, replies, work, fwd_rows = vs
    g = lambda x: model.evaluate(x, model_completion=True).as_long()
    out = []
    for t in range(depth + 1):
        out.append(f"  t={t:2d} inflight={g(inflight[t])} arrived(queued)={g(queued[t])} "
                   f"replies_pending={g(replies[t])} work_left={g(work[t])} "
                   f"rows/forward={g(fwd_rows[t])}")
    return "\n".join(out)


if __name__ == "__main__":
    cfg = Config(D=int(sys.argv[1]) if len(sys.argv) > 1 else 8,
                 work0=int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 18
    print(f"=== CONVOY (sustained rows/forward==1, pipe full, work remains)? {cfg} depth={depth} ===")
    s, vs = build(cfg, depth, "convoy")
    r = s.check()
    print(f"  -> {r}")
    if str(r) == "sat":
        print("  COUNTEREXAMPLE schedule (the metastable 1-row/forward convoy):")
        print(render(s.model(), vs, depth))
    print(f"=== HEALTHY (a high rows/forward) reachable under the SAME protocol? ===")
    s2, vs2 = build(cfg, depth, "healthy")
    r2 = s2.check()
    print(f"  -> {r2}")
    if str(r2) == "sat":
        print("  (a coalescing schedule the SAME greedy drain also permits:)")
        print(render(s2.model(), vs2, depth))
