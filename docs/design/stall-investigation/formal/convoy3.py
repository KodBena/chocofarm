#!/usr/bin/env python3
"""
~/w/vdc/chocobo/runs/formal-stall/convoy3.py  --  the DECISIVE convoy witness.

Empirical metric is ROWS per forward (healthy 55-177, collapsed ~1.4), NOT
messages per forward. convoy2 marched all K slots in lockstep so every message
carried K rows (healthy by construction) -- it could not express the staggered
regime where each wire message carries ~1 row.

The collapse requires: slots free ONE AT A TIME (staggered across D distinct
outstanding messages), so issue_one() finds only ~1 ready slot and ships a 1-ROW
message; the server, fast, forwards that 1-row message before the next arrives.

Faithful model (event/interleaving semantics; Z3 picks the schedule = the OS/ZMQ
timing the empirical run cannot control):
  * K slots. Each slot is PARKED(ready) / SUBMITTED(its 1-row leaf is in some
    outstanding message) / IDLE. A leaf is ALWAYS one row (one slot = one leaf
    per ply -- abstraction A-leaf).
  * inflight = set of outstanding messages, each carrying a SUBSET of submitted
    slots; modelled as a per-message row count. We bound to D messages.
  * issue_one(): coalesce ALL currently-ready slots into ONE message (rows =
    #ready), append to the request queue, mark them submitted, inflight+=1, iff
    inflight<D (runner_wire_batched.cpp:551-569,578,596).
  * SERVER_FWD: forward a NON-EMPTY PREFIX of the request queue that has
    'arrived' (Z3 chooses how many messages have arrived -> models staggering).
    rows/forward = sum of those messages' rows. Replies 1:1.  The greedy drain
    forwards whatever is queued AT ITS WAKE; staggered arrival => prefix of 1.
  * PROD_RECV: consume oldest reply, resume EXACTLY the slots that message
    carried (rem-=1, re-park if rem>0), then issue_one refill.

We ask Z3 for a schedule where SEVERAL forwards in a row have rows/forward == 1
while >=2 slots have work -- the metastable 1:1 convoy that the design's
greedy-drain PERMITS.  We also confirm a high-rows/forward schedule exists (the
healthy regime the SAME protocol permits) -- so nothing FORCES coalescing: that
non-forcing is the root cause.
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
    qcap: int


def build(cfg, depth, convoy=False, healthy=False):
    K, D, P, QC = cfg.K, cfg.D, cfg.plies, cfg.qcap
    s = Solver()

    def mk(t):
        st = {}
        for j in range(K):
            st[f"rem{j}"] = Int(f"rem_{t}_{j}")
            st[f"st{j}"] = Int(f"st_{t}_{j}")  # 0 parked,1 submitted,2 idle
        st["inflight"] = Int(f"inf_{t}")
        for c in range(QC):
            st[f"q{c}"] = Int(f"q_{t}_{c}")    # request queue msg rows (FIFO, 0 empty)
            st[f"r{c}"] = Int(f"r_{t}_{c}")    # reply FIFO msg rows
        st["fwd_rows"] = Int(f"fwdrows_{t}")   # rows in the forward that produced this state
        return st

    S = [mk(t) for t in range(depth + 1)]
    s0 = S[0]
    # prime: each slot parked then issue_one coalesces ALL K -> but to allow the
    # staggered regime we let the prime be: the producer issued some messages.
    # Faithful prime = ONE K-row message (issue_one coalesces all ready at prime).
    # To let the convoy be reachable we DO NOT force lockstep afterwards: the
    # transition lets messages carry subsets via staggered replies.
    for j in range(K):
        s.add(s0[f"rem{j}"] == P, s0[f"st{j}"] == 1)
    s.add(s0["inflight"] == 1, s0["fwd_rows"] == 0)
    s.add(s0["q0"] == K)
    for c in range(1, QC):
        s.add(s0[f"q{c}"] == 0)
    for c in range(QC):
        s.add(s0[f"r{c}"] == 0)

    for t in range(depth + 1):
        st = S[t]
        for j in range(K):
            s.add(st[f"rem{j}"] >= 0, st[f"rem{j}"] <= P, st[f"st{j}"] >= 0, st[f"st{j}"] <= 2)
        s.add(st["inflight"] >= 0, st["inflight"] <= D, st["fwd_rows"] >= 0)
        for c in range(QC):
            s.add(st[f"q{c}"] >= 0, st[f"r{c}"] >= 0)

    def any_q(st): return Or([st[f"q{c}"] > 0 for c in range(QC)])
    def any_r(st): return Or([st[f"r{c}"] > 0 for c in range(QC)])
    def fr(a, b, chg): return [b[k] == a[k] for k in a if k not in chg]

    # message-arrival prefix variable: at each forward, how many queued messages
    # have 'arrived' -- Z3 picks it (1..qlen). This is the staggering knob.
    arr = [Int(f"arr_{t}") for t in range(depth)]

    for t in range(depth):
        a, b = S[t], S[t + 1]
        opts = []
        qlen = Sum([If(a[f"q{c}"] > 0, 1, 0) for c in range(QC)])

        # SERVER_FWD: forward the first `arr[t]` queued messages (1<=arr<=qlen).
        chg = {f"q{c}" for c in range(QC)} | {f"r{c}" for c in range(QC)} | {"fwd_rows"}
        s.add(arr[t] >= 1, arr[t] <= QC)
        guard = And(any_q(a), arr[t] <= qlen)
        body = []
        fwd_rows = Sum([If(And(a[f"q{c}"] > 0,
                              Sum([If(a[f"q{cc}"] > 0, 1, 0) for cc in range(c)]) < arr[t]),
                          a[f"q{c}"], 0) for c in range(QC)])
        # forwarded[c] = q[c] nonempty and among first arr[t]
        forwarded = []
        for c in range(QC):
            before = Sum([If(a[f"q{cc}"] > 0, 1, 0) for cc in range(c)]) if c > 0 else 0
            forwarded.append(And(a[f"q{c}"] > 0, before < arr[t]))
        # remove forwarded from queue (compact remaining toward front)
        remaining = []
        for c in range(QC):
            remaining.append(If(And(a[f"q{c}"] > 0, Not(forwarded[c])), a[f"q{c}"], 0))
        def pth_pos(vals, p):
            expr = 0
            for c in range(QC):
                before = Sum([If(vals[cc] > 0, 1, 0) for cc in range(c)]) if c > 0 else 0
                expr = If(And(vals[c] > 0, before == p), vals[c], expr)
            return expr
        for c in range(QC):
            body.append(b[f"q{c}"] == pth_pos(remaining, c))
        # append forwarded msgs as replies after existing replies
        fwd_list = [If(forwarded[c], a[f"q{c}"], 0) for c in range(QC)]
        rcount = Sum([If(a[f"r{c}"] > 0, 1, 0) for c in range(QC)])
        for p in range(QC):
            existing = a[f"r{p}"] > 0
            eb = Sum([If(a[f"r{cc}"] == 0, 1, 0) for cc in range(p)]) if p > 0 else 0
            body.append(b[f"r{p}"] == If(existing, a[f"r{p}"],
                                         If(a[f"r{p}"] == 0, pth_pos(fwd_list, eb), a[f"r{p}"])))
        body.append(b["fwd_rows"] == fwd_rows)
        opts.append(And(guard, And(*body), And(*fr(a, b, chg))))

        # PROD_RECV: consume oldest reply (rows = how many slots it frees), resume
        # those slots, issue_one refill.
        chg = {f"r{c}" for c in range(QC)} | {f"rem{j}" for j in range(K)} \
            | {f"st{j}" for j in range(K)} | {"inflight"} | {f"q{c}" for c in range(QC)} | {"fwd_rows"}
        guard = any_r(a)
        body = []
        def oldest_r():
            expr = 0
            for c in range(QC):
                ee = And(*[a[f"r{cc}"] == 0 for cc in range(c)]) if c > 0 else True
                expr = If(And(ee, a[f"r{c}"] > 0), a[f"r{c}"], expr)
            return expr
        fr_rows = oldest_r()
        freed = []
        for j in range(K):
            sb = Sum([If(a[f"st{cc}"] == 1, 1, 0) for cc in range(j)]) if j > 0 else 0
            freed.append(And(a[f"st{j}"] == 1, sb < fr_rows))
        for j in range(K):
            nr = If(freed[j], a[f"rem{j}"] - 1, a[f"rem{j}"])
            body.append(b[f"rem{j}"] == nr)
        ready_after = []
        for j in range(K):
            nr = If(freed[j], a[f"rem{j}"] - 1, a[f"rem{j}"])
            ready_after.append(Or(And(freed[j], nr > 0), a[f"st{j}"] == 0))
        nready = Sum([If(ready_after[j], 1, 0) for j in range(K)])
        inflight_after = a["inflight"] - 1
        issues = And(nready > 0, inflight_after < D)
        body.append(b["inflight"] == If(issues, inflight_after + 1, inflight_after))
        for j in range(K):
            nr = If(freed[j], a[f"rem{j}"] - 1, a[f"rem{j}"])
            idle = nr == 0
            base = If(idle, 2, If(And(a[f"st{j}"] == 1, Not(freed[j])), 1,
                       If(issues, If(ready_after[j], 1, a[f"st{j}"]),
                          If(ready_after[j], 0, a[f"st{j}"]))))
            body.append(b[f"st{j}"] == base)
        for c in range(QC):
            ef = And(*[a[f"q{cc}"] > 0 for cc in range(c)]) if c > 0 else True
            tgt = And(issues, ef, a[f"q{c}"] == 0)
            body.append(b[f"q{c}"] == If(tgt, nready, a[f"q{c}"]))
        for c in range(QC):
            ee = And(*[a[f"r{cc}"] == 0 for cc in range(c)]) if c > 0 else True
            old = And(ee, a[f"r{c}"] > 0)
            body.append(b[f"r{c}"] == If(old, 0, a[f"r{c}"]))
        body.append(b["fwd_rows"] == a["fwd_rows"])  # unchanged on a recv
        opts.append(And(guard, And(*body), And(*fr(a, b, chg))))

        s.add(Or(*opts))

    # WITNESS predicates -------------------------------------------------------
    if convoy:
        # several forwards at rows/forward == 1 while >=2 slots still have work
        nfwd1 = Sum([If(S[t]["fwd_rows"] == 1, 1, 0) for t in range(1, depth + 1)])
        s.add(nfwd1 >= max(2, depth // 4))
        s.add(Sum([If(S[depth][f"rem{j}"] > 0, 1, 0) for j in range(K)]) >= 2)
    if healthy:
        s.add(Or([S[t]["fwd_rows"] >= max(2, cfg.K) for t in range(1, depth + 1)]))
    return s, S, arr


def render(model, S, arr, cfg):
    K, QC = cfg.K, cfg.qcap
    nm = {0: "P", 1: "S", 2: "i"}
    out = []
    for t, st in enumerate(S):
        g = lambda k: model.evaluate(st[k], model_completion=True).as_long()
        slots = " ".join(nm[g(f"st{j}")] + str(g(f"rem{j}")) for j in range(K))
        q = [g(f"q{c}") for c in range(QC) if g(f"q{c}") > 0]
        r = [g(f"r{c}") for c in range(QC) if g(f"r{c}") > 0]
        out.append(f"  t={t:2d} inflight={g('inflight'):>1} fwd_rows={g('fwd_rows')} | {slots} | q={q} rep={r}")
    return "\n".join(out)


if __name__ == "__main__":
    cfg = Config(K=int(sys.argv[1]) if len(sys.argv) > 1 else 4,
                 D=int(sys.argv[2]) if len(sys.argv) > 2 else 8,
                 plies=int(sys.argv[3]) if len(sys.argv) > 3 else 6,
                 qcap=int(sys.argv[4]) if len(sys.argv) > 4 else 10)
    depth = int(sys.argv[5]) if len(sys.argv) > 5 else 14
    print(f"=== CONVOY (>=2 forwards at rows/forward==1, work remains)? {cfg} depth={depth} ===")
    s, S, arr = build(cfg, depth, convoy=True)
    r = s.check()
    print(f"  -> {r}")
    if str(r) == "sat":
        print("  COUNTEREXAMPLE schedule (the metastable 1-row/forward convoy):")
        print(render(s.model(), S, arr, cfg))
    print(f"=== HEALTHY (a rows/forward>={max(2,cfg.K)} forward) also reachable? ===")
    s2, S2, arr2 = build(cfg, depth, healthy=True)
    print(f"  -> {s2.check()}")
