#!/usr/bin/env python3
"""
~/w/vdc/chocobo/runs/formal-stall/convoy2.py

Faithful CONVOY model. The 1:1 message<->forward collapse is a STAGGERING /
queue-depth phenomenon across D distinct outstanding messages, NOT a single-
message effect. convoy.py collapsed the message queue to one head and could not
represent it. Here we model:

  * A producer thread with K slots and inflight MESSAGE cap D. Each outstanding
    message carries some rows (a subset of slots). issue_one() coalesces ALL
    currently-READY slots into ONE message (runner_wire_batched.cpp:551-569).
  * A request QUEUE at the server: a FIFO of messages (each = its row count).
  * The server (group wakeup, greedy _drain): when it runs a forward it takes
    EVERYTHING currently in the queue -> ONE forward, replies 1:1 by corr-id
    (inference_server.py:_drain / stage_a_server group). The COALESCING DEGREE of
    a forward = how many messages were in the queue at that instant.

  The pathology: an interleaving where the server forwards the queue while it
  holds only ONE message (because the producer re-issues exactly one 1-row
  message per reply it consumes, and the server is fast enough to forward it
  before the next arrives). The result is a forward-per-message convoy at S=1.

We model TIME as an interleaving (event semantics) and let Z3 pick the schedule:
  ACTIONS each step (exactly one fires):
    SERVER_FWD : queue non-empty -> forward ALL queued msgs in ONE batch (degree
                 = current queue length, rows = sum). Move them to the reply set;
                 each becomes a pending reply for the producer. Record the degree.
    PROD_RECV  : a reply is pending AND producer is waiting -> consume one reply,
                 resume its slots (rem-=1, re-park if rem>0), then issue_one:
                 coalesce ALL ready slots into ONE new request msg, append to queue
                 (if inflight<D).
  The CONVOY is a reachable schedule on which EVERY SERVER_FWD has degree 1 while
  work remains -- i.e. the bad regime is an admissible interleaving (a livelock
  the scheduler CAN sustain), which is exactly what "metastable / sticky" means.

We ALSO check the contrast: the HEALTHY schedule (server waits until many msgs
queue -> a high-degree forward) is admissible too -- the protocol PERMITS both,
so nothing forces the good one. That non-forcing is the root cause.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from z3 import And, If, Int, Not, Or, Solver, Sum


@dataclass(frozen=True)
class Config:
    K: int
    D: int
    plies: int
    qcap: int   # queue array capacity (>= D)


def build(cfg, depth, force_degree=None):
    K, D, P, QC = cfg.K, cfg.D, cfg.plies, cfg.qcap
    s = Solver()

    def mk(t):
        st = {}
        for j in range(K):
            st[f"rem{j}"] = Int(f"rem_{t}_{j}")
            st[f"st{j}"] = Int(f"st_{t}_{j}")   # 0 parked-ready, 1 submitted, 2 idle
        st["inflight"] = Int(f"inf_{t}")
        # request queue: QC msg-slots, each a row count (0 empty)
        for c in range(QC):
            st[f"q{c}"] = Int(f"q_{t}_{c}")
        # pending replies: QC slots, each a row count (0 empty)
        for c in range(QC):
            st[f"r{c}"] = Int(f"r_{t}_{c}")
        st["last_degree"] = Int(f"deg_{t}")   # degree of the most recent forward
        return st

    S = [mk(t) for t in range(depth + 1)]
    s0 = S[0]
    # prime: K ready slots, first issue_one coalesces all -> ONE K-row msg queued,
    # all submitted, inflight=1. (prime loop's later issues find nothing ready.)
    for j in range(K):
        s.add(s0[f"rem{j}"] == P, s0[f"st{j}"] == 1)
    s.add(s0["inflight"] == 1, s0["last_degree"] == 0)
    s.add(s0["q0"] == K)
    for c in range(1, QC):
        s.add(s0[f"q{c}"] == 0)
    for c in range(QC):
        s.add(s0[f"r{c}"] == 0)

    for t in range(depth + 1):
        st = S[t]
        for j in range(K):
            s.add(st[f"rem{j}"] >= 0, st[f"rem{j}"] <= P, st[f"st{j}"] >= 0, st[f"st{j}"] <= 2)
        s.add(st["inflight"] >= 0, st["inflight"] <= D)
        s.add(st["last_degree"] >= 0)
        for c in range(QC):
            s.add(st[f"q{c}"] >= 0, st[f"r{c}"] >= 0)

    def qlen(st):
        return Sum([If(st[f"q{c}"] > 0, 1, 0) for c in range(QC)])

    def any_q(st):
        return Or([st[f"q{c}"] > 0 for c in range(QC)])

    def any_r(st):
        return Or([st[f"r{c}"] > 0 for c in range(QC)])

    def ready_count(st):
        return Sum([If(st[f"st{j}"] == 0, 1, 0) for j in range(K)])

    def fr(a, b, chg):
        return [b[k] == a[k] for k in a if k not in chg]

    for t in range(depth):
        a, b = S[t], S[t + 1]
        opts = []

        # SERVER_FWD: forward ALL queued msgs (degree = qlen), each -> a reply.
        chg = {f"q{c}" for c in range(QC)} | {f"r{c}" for c in range(QC)} | {"last_degree"}
        guard = And(any_q(a))
        body = []
        deg = qlen(a)
        # move each queued msg (in order) to the first empty reply slot.
        # clear queue
        for c in range(QC):
            body.append(b[f"q{c}"] == 0)
        # append queued rows to reply channel after existing replies
        def pth_q(p):
            expr = 0
            for c in range(QC):
                before = Sum([If(a[f"q{cc}"] > 0, 1, 0) for cc in range(c)]) if c > 0 else 0
                expr = If(And(a[f"q{c}"] > 0, before == p), a[f"q{c}"], expr)
            return expr
        for p in range(QC):
            existing = a[f"r{p}"] > 0
            empties_before = Sum([If(a[f"r{cc}"] == 0, 1, 0) for cc in range(p)]) if p > 0 else 0
            is_empty = a[f"r{p}"] == 0
            body.append(b[f"r{p}"] == If(existing, a[f"r{p}"], If(is_empty, pth_q(empties_before), a[f"r{p}"])))
        body.append(b["last_degree"] == deg)
        sf = And(guard, And(*body), And(*fr(a, b, chg)))
        if force_degree is not None:
            # optionally force the server to only fire at a chosen degree (to test
            # whether a given regime is admissible)
            sf = And(sf, deg == force_degree)
        opts.append(sf)

        # PROD_RECV: consume oldest reply, resume its slots, issue_one refill.
        chg = {f"r{c}" for c in range(QC)} | {f"rem{j}" for j in range(K)} \
            | {f"st{j}" for j in range(K)} | {"inflight"} | {f"q{c}" for c in range(QC)}
        guard = any_r(a)
        # oldest reply rows = how many slots it frees
        # identify oldest reply slot
        body = []
        # free the first `freed_rows` submitted slots (lowest index) -- WLOG.
        # oldest reply rows:
        def oldest_r():
            expr = 0
            for c in range(QC):
                ee = And(*[a[f"r{cc}"] == 0 for cc in range(c)]) if c > 0 else True
                expr = If(And(ee, a[f"r{c}"] > 0), a[f"r{c}"], expr)
            return expr
        fr_rows = oldest_r()
        freed = []
        for j in range(K):
            sub_before = Sum([If(a[f"st{cc}"] == 1, 1, 0) for cc in range(j)]) if j > 0 else 0
            freed.append(And(a[f"st{j}"] == 1, sub_before < fr_rows))
        # resume freed: rem-=1; rem>0 -> parked(0) else idle(2)
        for j in range(K):
            nr = If(freed[j], a[f"rem{j}"] - 1, a[f"rem{j}"])
            body.append(b[f"rem{j}"] == nr)
        # now issue_one over ALL ready (freshly parked + any previously parked).
        # ready_after[j] = (freed & nr>0) OR (was parked & not freed)
        ready_after = []
        for j in range(K):
            nr = If(freed[j], a[f"rem{j}"] - 1, a[f"rem{j}"])
            ready_after.append(Or(And(freed[j], nr > 0), a[f"st{j}"] == 0))
        nready = Sum([If(ready_after[j], 1, 0) for j in range(K)])
        inflight_after = a["inflight"] - 1
        issues = And(nready > 0, inflight_after < D)
        body.append(b["inflight"] == If(issues, inflight_after + 1, inflight_after))
        # set states: if issues, all ready_after -> submitted(1); else ready stay parked(0)
        for j in range(K):
            nr = If(freed[j], a[f"rem{j}"] - 1, a[f"rem{j}"])
            idle = nr == 0
            new_state = If(idle, 2, If(issues, 1, If(ready_after[j], 0, a[f"st{j}"])))
            # careful: a not-freed submitted slot (st==1, not ready) stays submitted
            new_state = If(And(a[f"st{j}"] == 1, Not(freed[j])), 1, new_state)
            body.append(b[f"st{j}"] == new_state)
        # append the new request msg (rows = nready) to the queue if issues
        for c in range(QC):
            ef = And(*[a[f"q{cc}"] > 0 for cc in range(c)]) if c > 0 else True
            tgt = And(issues, ef, a[f"q{c}"] == 0)
            body.append(b[f"q{c}"] == If(tgt, nready, a[f"q{c}"]))
        # clear the consumed oldest reply
        for c in range(QC):
            ee = And(*[a[f"r{cc}"] == 0 for cc in range(c)]) if c > 0 else True
            old = And(ee, a[f"r{c}"] > 0)
            body.append(b[f"r{c}"] == If(old, 0, a[f"r{c}"]))
        opts.append(And(guard, And(*body), And(*fr(a, b, chg))))

        s.add(Or(*opts))

    return s, S


def reachable_convoy(cfg, depth):
    """SAT iff there is an admissible schedule on which EVERY server forward from
    step 1 on has degree exactly 1 while work remains (the sustained 1:1 convoy)."""
    s, S = build(cfg, depth)
    # require: every forward that happens has degree 1. We can't index 'which steps
    # are forwards' directly, so we require last_degree in {0,1} at every step AND
    # at least (depth//2) forwards occurred at degree 1 (progress in the bad regime)
    # AND work remains at the end.
    forwards_deg1 = []
    for t in range(1, depth + 1):
        # a forward happened into step t iff last_degree changed to >0
        s.add(Or(S[t]["last_degree"] == 0, S[t]["last_degree"] == 1))
    num_fwd = Sum([If(S[t]["last_degree"] == 1, 1, 0) for t in range(1, depth + 1)])
    s.add(num_fwd >= depth // 3)   # the convoy is actually running (many 1-degree fwds)
    s.add(Or([S[depth][f"rem{j}"] > 0 for j in range(cfg.K)]))  # work remains
    return s, S


def healthy_high_degree(cfg, depth, deg):
    """SAT iff a schedule exists where a forward of degree `deg`>1 happens -- shows
    the protocol ALSO permits good coalescing (so nothing forces the bad regime)."""
    s, S = build(cfg, depth)
    s.add(Or([S[t]["last_degree"] >= deg for t in range(1, depth + 1)]))
    return s, S


def render(model, S, cfg):
    K, QC = cfg.K, cfg.qcap
    nm = {0: "P", 1: "S", 2: "i"}
    out = []
    for t, st in enumerate(S):
        g = lambda k: model.evaluate(st[k], model_completion=True).as_long()
        slots = " ".join(nm[g(f"st{j}")] + str(g(f"rem{j}")) for j in range(K))
        q = [g(f"q{c}") for c in range(QC) if g(f"q{c}") > 0]
        r = [g(f"r{c}") for c in range(QC) if g(f"r{c}") > 0]
        out.append(f"  t={t:2d} inflight={g('inflight')} fwd_degree={g('last_degree')} | {slots} | q={q} rep={r}")
    return "\n".join(out)


if __name__ == "__main__":
    cfg = Config(K=int(sys.argv[1]) if len(sys.argv) > 1 else 4,
                 D=int(sys.argv[2]) if len(sys.argv) > 2 else 8,
                 plies=int(sys.argv[3]) if len(sys.argv) > 3 else 4,
                 qcap=int(sys.argv[4]) if len(sys.argv) > 4 else 10)
    depth = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    print(f"=== CONVOY (sustained degree-1 forwards) reachable? {cfg} depth={depth} ===")
    s, S = reachable_convoy(cfg, depth)
    r = s.check()
    print(f"  -> {r}")
    if str(r) == "sat":
        print("  COUNTEREXAMPLE schedule (the 1:1 message<->forward convoy):")
        print(render(s.model(), S, cfg))
    print(f"=== HEALTHY (a degree>=2 forward) also reachable? ===")
    s2, S2 = healthy_high_degree(cfg, depth, 2)
    r2 = s2.check()
    print(f"  -> {r2}  (both reachable => the protocol does not FORCE coalescing)")
