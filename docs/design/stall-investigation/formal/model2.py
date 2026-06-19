#!/usr/bin/env python3
"""
~/w/vdc/chocobo/runs/formal-stall/model2.py

RICHER formal model adding the two things model.py abstracted away:
  (1) the SERVER's greedy drain has a max_batch ROW cap and a request is
      INDIVISIBLE — a request whose rows exceed the remaining budget is still
      drained whole, but a *subsequent* queued request is LEFT in the channel
      until the next _drain wakeup.  (inference_server.py:348-363)
  (2) MULTIPLE producer threads, each its own DEALER / inflight_msgs / slot set,
      all fan-in to the ONE server ROUTER.  The server answers each request msg
      1:1 by corr-id but its single forward couples them.

This is still the pipelined driver (run_episodes_wire_pipelined). We are hunting
a reachable state where EVERY producer thread is blocked in recv_batch AND the
server is blocked in _drain (no req queued) AND work remains.

KEY MODELLING of the server <-> producer wakeup coupling (the place a real ZMQ
deadlock can hide):
  - The server, after draining a group and scattering replies, loops back to
    _drain and BLOCKS if no request is currently queued.
  - A producer only puts a request in the channel by issue_one(), which it can
    only do at pc==ISSUE with a ready slot and inflight<D.
  - A producer reaches pc==ISSUE either initially or after consuming a reply.

The question this model answers: can the system reach a global state where the
server has emptied the request channel (so it blocks) WHILE every thread is
parked in recv waiting for a reply that the server has *already sent* but that
is still required to wake them -- i.e. is there a lost/!missing wakeup, or a
state where replies are stranded?  ZMQ pipes do NOT lose messages, so we model
the reply channel as reliable.  The deadlock, if any, must be an ACCOUNTING one
(inflight desync) or a structural starvation under the row cap.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from z3 import And, If, Int, Not, Or, Solver, Sum, sat


@dataclass(frozen=True)
class Config:
    T: int            # producer threads
    K: int            # slots per thread
    D: int            # per-thread inflight MESSAGE cap
    plies: int        # leaves each slot owes
    max_rows: int     # server greedy-drain ROW cap (max_batch)
    cap: int          # channel array capacity (messages) -- bound


PC_ISSUE, PC_RECV, PC_DONE = 0, 1, 2


def build(cfg: Config, steps: int):
    T, K, D, P, MR, CAP = cfg.T, cfg.K, cfg.D, cfg.plies, cfg.max_rows, cfg.cap
    s = Solver()

    def mk(t):
        st = {}
        for i in range(T):
            for j in range(K):
                st[f"rem_{i}_{j}"] = Int(f"rem_{t}_{i}_{j}")
                st[f"ss_{i}_{j}"] = Int(f"ss_{t}_{i}_{j}")
            st[f"inflight_{i}"] = Int(f"inflight_{t}_{i}")
            st[f"pc_{i}"] = Int(f"pc_{t}_{i}")
        st["srv"] = Int(f"srv_{t}")
        # request channel: each entry (rows, owner-thread). encode two arrays.
        for c in range(CAP):
            st[f"reqrows_{c}"] = Int(f"reqrows_{t}_{c}")
            st[f"reqown_{c}"] = Int(f"reqown_{t}_{c}")   # -1 empty else thread id
        # reply channel: (rows, owner-thread)
        for c in range(CAP):
            st[f"reprows_{c}"] = Int(f"reprows_{t}_{c}")
            st[f"repown_{c}"] = Int(f"repown_{t}_{c}")
        return st

    S = [mk(t) for t in range(steps + 1)]

    s0 = S[0]
    for i in range(T):
        for j in range(K):
            s.add(s0[f"rem_{i}_{j}"] == P, s0[f"ss_{i}_{j}"] == 0)
        s.add(s0[f"inflight_{i}"] == 0, s0[f"pc_{i}"] == PC_ISSUE)
    s.add(s0["srv"] == 0)
    for c in range(CAP):
        s.add(s0[f"reqown_{c}"] == -1, s0[f"reqrows_{c}"] == 0)
        s.add(s0[f"repown_{c}"] == -1, s0[f"reprows_{c}"] == 0)

    for t in range(steps + 1):
        st = S[t]
        for i in range(T):
            for j in range(K):
                s.add(st[f"rem_{i}_{j}"] >= 0, st[f"rem_{i}_{j}"] <= P)
                s.add(st[f"ss_{i}_{j}"] >= 0, st[f"ss_{i}_{j}"] <= 2)
            s.add(st[f"inflight_{i}"] >= 0, st[f"inflight_{i}"] <= CAP)
            s.add(st[f"pc_{i}"] >= 0, st[f"pc_{i}"] <= 2)
        s.add(st["srv"] >= 0, st["srv"] <= 1)
        for c in range(CAP):
            s.add(st[f"reqown_{c}"] >= -1, st[f"reqown_{c}"] < T)
            s.add(st[f"repown_{c}"] >= -1, st[f"repown_{c}"] < T)
            s.add(st[f"reqrows_{c}"] >= 0, st[f"reprows_{c}"] >= 0)

    def ready_count(st, i):
        return Sum([If(And(st[f"ss_{i}_{j}"] == 0, st[f"rem_{i}_{j}"] > 0), 1, 0)
                    for j in range(K)])

    def req_used(st):
        return Sum([If(st[f"reqown_{c}"] >= 0, 1, 0) for c in range(CAP)])

    def any_req(st):
        return Or([st[f"reqown_{c}"] >= 0 for c in range(CAP)])

    def thread_has_rep(st, i):
        return Or([And(st[f"repown_{c}"] == i) for c in range(CAP)])

    def frame_except(a, b, changed):
        return [b[k] == a[k] for k in a if k not in changed]

    for t in range(steps):
        a, b = S[t], S[t + 1]
        opts = []

        for i in range(T):
            rc = ready_count(a, i)
            # PRODUCER_ISSUE (thread i): coalesce all ready slots -> one msg
            chg = {f"reqrows_{c}" for c in range(CAP)} | {f"reqown_{c}" for c in range(CAP)} \
                | {f"ss_{i}_{j}" for j in range(K)} | {f"inflight_{i}"}
            guard = And(a[f"pc_{i}"] == PC_ISSUE, rc > 0, a[f"inflight_{i}"] < D,
                        req_used(a) < CAP)
            body = []
            for j in range(K):
                body.append(b[f"ss_{i}_{j}"] ==
                            If(And(a[f"ss_{i}_{j}"] == 0, a[f"rem_{i}_{j}"] > 0), 1, a[f"ss_{i}_{j}"]))
            body.append(b[f"inflight_{i}"] == a[f"inflight_{i}"] + 1)
            for c in range(CAP):
                earlier_full = And(*[a[f"reqown_{cc}"] >= 0 for cc in range(c)]) if c > 0 else True
                tgt = And(earlier_full, a[f"reqown_{c}"] == -1)
                body.append(b[f"reqrows_{c}"] == If(tgt, rc, a[f"reqrows_{c}"]))
                body.append(b[f"reqown_{c}"] == If(tgt, i, a[f"reqown_{c}"]))
            opts.append(And(guard, And(*body), And(*frame_except(a, b, chg))))

            # PRODUCER_ISSUE_DONE (thread i)
            chg = {f"pc_{i}"}
            guard = And(a[f"pc_{i}"] == PC_ISSUE, Or(rc == 0, a[f"inflight_{i}"] == D))
            body = [b[f"pc_{i}"] == If(a[f"inflight_{i}"] > 0, PC_RECV, PC_DONE)]
            opts.append(And(guard, And(*body), And(*frame_except(a, b, chg))))

            # PRODUCER_RECV (thread i): consume oldest reply owned by i
            chg = {f"reprows_{c}" for c in range(CAP)} | {f"repown_{c}" for c in range(CAP)} \
                | {f"rem_{i}_{j}" for j in range(K)} | {f"ss_{i}_{j}" for j in range(K)} \
                | {f"inflight_{i}", f"pc_{i}"}
            guard = And(a[f"pc_{i}"] == PC_RECV, thread_has_rep(a, i))
            body = []
            for j in range(K):
                was_sub = a[f"ss_{i}_{j}"] == 1
                new_rem = If(was_sub, a[f"rem_{i}_{j}"] - 1, a[f"rem_{i}_{j}"])
                body.append(b[f"rem_{i}_{j}"] == new_rem)
                body.append(b[f"ss_{i}_{j}"] == If(was_sub, If(new_rem > 0, 0, 2), a[f"ss_{i}_{j}"]))
            body.append(b[f"inflight_{i}"] == a[f"inflight_{i}"] - 1)
            body.append(b[f"pc_{i}"] == PC_ISSUE)
            for c in range(CAP):
                earlier_no = And(*[a[f"repown_{cc}"] != i for cc in range(c)]) if c > 0 else True
                oldest = And(earlier_no, a[f"repown_{c}"] == i)
                body.append(b[f"repown_{c}"] == If(oldest, -1, a[f"repown_{c}"]))
                body.append(b[f"reprows_{c}"] == If(oldest, 0, a[f"reprows_{c}"]))
            opts.append(And(guard, And(*body), And(*frame_except(a, b, chg))))

        # SERVER_DRAIN: drain requests up to MR rows (indivisible), one reply each.
        # We model drain order = channel order; stop accumulating once rows>=MR
        # AFTER appending the request that crossed the cap (mirrors the while:
        # check total_rows<MR at TOP, so it drains until the running total
        # reaches/exceeds MR, leaving the rest queued).
        chg = {f"reqrows_{c}" for c in range(CAP)} | {f"reqown_{c}" for c in range(CAP)} \
            | {f"reprows_{c}" for c in range(CAP)} | {f"repown_{c}" for c in range(CAP)}
        guard = And(a["srv"] == 0, any_req(a))
        body = []
        # Determine which req indices get drained: walk channel in order, keep a
        # running prefix-rows; a req at position c is drained iff the prefix BEFORE
        # it is < MR (the while-guard tests total_rows<MR before recv).
        drained = []
        for c in range(CAP):
            prefix = Sum([If(And(a[f"reqown_{cc}"] >= 0, _is_drained_expr := True),
                             a[f"reqrows_{cc}"], 0) for cc in range(c)]) if c > 0 else 0
            # NOTE: exact prefix-of-drained is circular; we APPROXIMATE with the
            # prefix of ALL queued rows before c, which equals the drained prefix
            # because draining is a contiguous prefix of the channel (FIFO, no
            # skips). So position c is drained iff sum of rows strictly before c
            # (over queued entries) < MR.
            prefix_rows = Sum([If(a[f"reqown_{cc}"] >= 0, a[f"reqrows_{cc}"], 0) for cc in range(c)]) if c > 0 else 0
            drained.append(And(a[f"reqown_{c}"] >= 0, prefix_rows < MR))
        # clear drained req entries; keep undrained
        for c in range(CAP):
            body.append(b[f"reqown_{c}"] == If(drained[c], -1, a[f"reqown_{c}"]))
            body.append(b[f"reqrows_{c}"] == If(drained[c], 0, a[f"reqrows_{c}"]))
        # append one reply per drained req, preserving owner+rows, after existing
        # replies. Compact: for output rep position p, find the p-th drained req.
        def pth_drained_field(p, field):
            expr = -1 if field == "own" else 0
            for c in range(CAP):
                before = Sum([If(drained[cc], 1, 0) for cc in range(c)]) if c > 0 else 0
                val = a[f"reqown_{c}"] if field == "own" else a[f"reqrows_{c}"]
                expr = If(And(drained[c], before == p), val, expr)
            return expr
        rep_used_a = Sum([If(a[f"repown_{c}"] >= 0, 1, 0) for c in range(CAP)])
        for p in range(CAP):
            existing = a[f"repown_{p}"] >= 0
            # new replies appended starting at first empty rep slot. Index among
            # empties = p - (#existing before p). Use compaction over empties.
            empties_before = Sum([If(a[f"repown_{cc}"] == -1, 1, 0) for cc in range(p)]) if p > 0 else 0
            is_empty = a[f"repown_{p}"] == -1
            newown = pth_drained_field(empties_before, "own")
            newrows = pth_drained_field(empties_before, "rows")
            body.append(b[f"repown_{p}"] == If(existing, a[f"repown_{p}"],
                                               If(is_empty, newown, a[f"repown_{p}"])))
            body.append(b[f"reprows_{p}"] == If(existing, a[f"reprows_{p}"],
                                                If(is_empty, newrows, a[f"reprows_{p}"])))
        opts.append(And(guard, And(*body), And(*frame_except(a, b, chg))))

        s.add(Or(*opts))

    # DEADLOCK at final step
    last = S[steps]
    no_act = []
    for i in range(T):
        rc = ready_count(last, i)
        no_act.append(Not(And(last[f"pc_{i}"] == PC_ISSUE, rc > 0,
                              last[f"inflight_{i}"] < D, req_used(last) < CAP)))
        no_act.append(Not(And(last[f"pc_{i}"] == PC_ISSUE,
                              Or(rc == 0, last[f"inflight_{i}"] == D))))
        no_act.append(Not(And(last[f"pc_{i}"] == PC_RECV, thread_has_rep(last, i))))
    no_act.append(Not(And(last["srv"] == 0, any_req(last))))
    stuck = And(*no_act)

    work = Or(
        any_req(last),
        Or([last[f"repown_{c}"] >= 0 for c in range(CAP)]),
        Or([last[f"inflight_{i}"] > 0 for i in range(T)]),
        Or([And(last[f"rem_{i}_{j}"] > 0, last[f"ss_{i}_{j}"] != 2)
            for i in range(T) for j in range(K)]),
    )
    not_all_done = Or([last[f"pc_{i}"] != PC_DONE for i in range(T)])

    s.add(stuck, work, not_all_done)
    return s, S


def render(model, S, cfg):
    T, K, CAP = cfg.T, cfg.K, cfg.cap
    pc = {0: "ISSUE", 1: "RECV", 2: "DONE"}
    ss = {0: "PARK", 1: "SUB", 2: "IDLE"}
    out = []
    for t, st in enumerate(S):
        g = lambda k: model.evaluate(st[k], model_completion=True).as_long()
        threads = []
        for i in range(T):
            slots = ",".join(f"{ss[g(f'ss_{i}_{j}')]}{g(f'rem_{i}_{j}')}" for j in range(K))
            threads.append(f"T{i}[pc={pc[g(f'pc_{i}')]},inf={g(f'inflight_{i}')},{slots}]")
        reqs = [(g(f"reqown_{c}"), g(f"reqrows_{c}")) for c in range(CAP) if g(f"reqown_{c}") >= 0]
        reps = [(g(f"repown_{c}"), g(f"reprows_{c}")) for c in range(CAP) if g(f"repown_{c}") >= 0]
        out.append(f"  t={t:2d} srv={'BLK' if g('srv')==0 else 'drn'} {' '.join(threads)} "
                   f"req={reqs} rep={reps}")
    return "\n".join(out)


if __name__ == "__main__":
    cfg = Config(T=int(sys.argv[1]) if len(sys.argv) > 1 else 2,
                 K=int(sys.argv[2]) if len(sys.argv) > 2 else 2,
                 D=int(sys.argv[3]) if len(sys.argv) > 3 else 8,
                 plies=int(sys.argv[4]) if len(sys.argv) > 4 else 2,
                 max_rows=int(sys.argv[5]) if len(sys.argv) > 5 else 2,
                 cap=int(sys.argv[6]) if len(sys.argv) > 6 else 8)
    depth = int(sys.argv[7]) if len(sys.argv) > 7 else 14
    s, S = build(cfg, depth)
    r = s.check()
    print(f"{cfg} depth={depth} -> {r}")
    if r == sat:
        print("DEADLOCK reachable:")
        print(render(s.model(), S, cfg))
