#!/usr/bin/env python3
"""
~/w/vdc/chocobo/runs/formal-stall/model3.py

Model3 = model.py (single producer thread, pipelined driver) PLUS the transport
realism model.py idealized away:

  * The DEALER socket sets RCVTIMEO but NO SNDTIMEO and NO SNDHWM override, so it
    runs the ZMQ default SNDHWM (a FINITE send-buffer, ~1000 msgs upstream, but
    the QUEUE the producer fills toward the ROUTER is what matters). When that
    buffer is full, zmq_send BLOCKS (a DEALER with no SNDTIMEO blocks on a full
    pipe -- it does not drop, it does not error).  wire_leaf_pool.hpp:139-144.

  * The ROUTER's RECEIVE side has a finite RCVHWM; once the server is mid-forward
    (not recv'ing) and the pipe between DEALER and ROUTER fills, the DEALER's
    send blocks.

  * The producer's recv_batch BLOCKS on RCVTIMEO -- a true wait.

We DELIBERATELY shrink the request-channel capacity REQ_CAP to model the HWM as
a small bound, and we add the missing producer state PC_SEND_BLOCKED: a producer
that wants to issue but the request channel is full WAITS in send.

We then ask: is there a reachable state where the producer is blocked (in RECV or
in SEND) AND the server is blocked (in _drain, no req queued) AND work remains?

Crucially this also lets the producer be stuck in SEND while the REPLY channel is
backed up -- the cross-coupling a single-socket-pair half-duplex stall needs.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from z3 import And, If, Int, Not, Or, Solver, Sum, sat


@dataclass(frozen=True)
class Config:
    K: int
    D: int
    plies: int
    req_cap: int      # request-channel capacity (HWM abstraction) -- SMALL
    rep_cap: int      # reply-channel capacity (HWM abstraction) -- SMALL
    max_rows: int     # server drain row cap


PC_ISSUE, PC_RECV, PC_DONE, PC_SEND = 0, 1, 2, 3


def build(cfg, steps):
    K, D, P, RQ, RP, MR = cfg.K, cfg.D, cfg.plies, cfg.req_cap, cfg.rep_cap, cfg.max_rows
    s = Solver()

    def mk(t):
        st = {}
        for j in range(K):
            st[f"rem{j}"] = Int(f"rem_{t}_{j}")
            st[f"ss{j}"] = Int(f"ss_{t}_{j}")
        st["inflight"] = Int(f"inflight_{t}")
        st["pc"] = Int(f"pc_{t}")
        st["srv"] = Int(f"srv_{t}")
        st["pend_rows"] = Int(f"pendrows_{t}")  # rows the blocked SEND wants to push
        for c in range(RQ):
            st[f"req{c}"] = Int(f"req_{t}_{c}")
        for c in range(RP):
            st[f"rep{c}"] = Int(f"rep_{t}_{c}")
        return st

    S = [mk(t) for t in range(steps + 1)]
    s0 = S[0]
    for j in range(K):
        s.add(s0[f"rem{j}"] == P, s0[f"ss{j}"] == 0)
    s.add(s0["inflight"] == 0, s0["pc"] == PC_ISSUE, s0["srv"] == 0, s0["pend_rows"] == 0)
    for c in range(RQ):
        s.add(s0[f"req{c}"] == 0)
    for c in range(RP):
        s.add(s0[f"rep{c}"] == 0)

    for t in range(steps + 1):
        st = S[t]
        for j in range(K):
            s.add(st[f"rem{j}"] >= 0, st[f"rem{j}"] <= P, st[f"ss{j}"] >= 0, st[f"ss{j}"] <= 2)
        s.add(st["inflight"] >= 0, st["pc"] >= 0, st["pc"] <= 3, st["srv"] >= 0, st["srv"] <= 1)
        s.add(st["pend_rows"] >= 0)
        for c in range(RQ):
            s.add(st[f"req{c}"] >= 0)
        for c in range(RP):
            s.add(st[f"rep{c}"] >= 0)

    def ready_count(st):
        return Sum([If(And(st[f"ss{j}"] == 0, st[f"rem{j}"] > 0), 1, 0) for j in range(K)])

    def req_used(st):
        return Sum([If(st[f"req{c}"] > 0, 1, 0) for c in range(RQ)])

    def rep_used(st):
        return Sum([If(st[f"rep{c}"] > 0, 1, 0) for c in range(RP)])

    def any_req(st):
        return Or([st[f"req{c}"] > 0 for c in range(RQ)])

    def any_rep(st):
        return Or([st[f"rep{c}"] > 0 for c in range(RP)])

    def fr(a, b, chg):
        return [b[k] == a[k] for k in a if k not in chg]

    for t in range(steps):
        a, b = S[t], S[t + 1]
        opts = []
        rc = ready_count(a)

        # PRODUCER_ISSUE: ready, inflight<D, AND request channel has room.
        # Marks ready slots submitted, +1 inflight, append msg.
        chg = {f"req{c}" for c in range(RQ)} | {f"ss{j}" for j in range(K)} | {"inflight"}
        guard = And(a["pc"] == PC_ISSUE, rc > 0, a["inflight"] < D, req_used(a) < RQ)
        body = [b[f"ss{j}"] == If(And(a[f"ss{j}"] == 0, a[f"rem{j}"] > 0), 1, a[f"ss{j}"]) for j in range(K)]
        body.append(b["inflight"] == a["inflight"] + 1)
        for c in range(RQ):
            ef = And(*[a[f"req{cc}"] > 0 for cc in range(c)]) if c > 0 else True
            tgt = And(ef, a[f"req{c}"] == 0)
            body.append(b[f"req{c}"] == If(tgt, rc, a[f"req{c}"]))
        opts.append(And(guard, And(*body), And(*fr(a, b, chg))))

        # PRODUCER_ISSUE_WANTS_SEND_BUT_FULL: ready, inflight<D, but req channel
        # FULL -> the DEALER send BLOCKS. Mark slots submitted (the send call is
        # committed to these rows) and go to PC_SEND holding pend_rows. This is the
        # missing blocking-send state.
        chg = {f"ss{j}" for j in range(K)} | {"pc", "pend_rows", "inflight"}
        guard = And(a["pc"] == PC_ISSUE, rc > 0, a["inflight"] < D, req_used(a) == RQ)
        body = [b[f"ss{j}"] == If(And(a[f"ss{j}"] == 0, a[f"rem{j}"] > 0), 1, a[f"ss{j}"]) for j in range(K)]
        body += [b["pc"] == PC_SEND, b["pend_rows"] == rc, b["inflight"] == a["inflight"] + 1]
        opts.append(And(guard, And(*body), And(*fr(a, b, chg))))

        # PRODUCER_SEND_COMPLETES: in PC_SEND, req channel now has room -> the
        # blocked send lands its message; go to PC_RECV (it had inflight>0).
        chg = {f"req{c}" for c in range(RQ)} | {"pc", "pend_rows"}
        guard = And(a["pc"] == PC_SEND, req_used(a) < RQ)
        body = [b["pc"] == PC_RECV, b["pend_rows"] == 0]
        for c in range(RQ):
            ef = And(*[a[f"req{cc}"] > 0 for cc in range(c)]) if c > 0 else True
            tgt = And(ef, a[f"req{c}"] == 0)
            body.append(b[f"req{c}"] == If(tgt, a["pend_rows"], a[f"req{c}"]))
        opts.append(And(guard, And(*body), And(*fr(a, b, chg))))

        # PRODUCER_ISSUE_DONE: ready==0 or inflight==D -> RECV (inflight>0) / DONE.
        chg = {"pc"}
        guard = And(a["pc"] == PC_ISSUE, Or(rc == 0, a["inflight"] == D))
        opts.append(And(guard, b["pc"] == If(a["inflight"] > 0, PC_RECV, PC_DONE), And(*fr(a, b, chg))))

        # PRODUCER_RECV: reply available -> consume oldest, -1 inflight, resume
        # submitted slots, back to PC_ISSUE.
        chg = {f"rep{c}" for c in range(RP)} | {f"rem{j}" for j in range(K)} \
            | {f"ss{j}" for j in range(K)} | {"inflight", "pc"}
        guard = And(a["pc"] == PC_RECV, any_rep(a))
        body = []
        for j in range(K):
            ws = a[f"ss{j}"] == 1
            nr = If(ws, a[f"rem{j}"] - 1, a[f"rem{j}"])
            body.append(b[f"rem{j}"] == nr)
            body.append(b[f"ss{j}"] == If(ws, If(nr > 0, 0, 2), a[f"ss{j}"]))
        body += [b["inflight"] == a["inflight"] - 1, b["pc"] == PC_ISSUE]
        for c in range(RP):
            ee = And(*[a[f"rep{cc}"] == 0 for cc in range(c)]) if c > 0 else True
            old = And(ee, a[f"rep{c}"] > 0)
            body.append(b[f"rep{c}"] == If(old, 0, a[f"rep{c}"]))
        opts.append(And(guard, And(*body), And(*fr(a, b, chg))))

        # SERVER_DRAIN: drain contiguous prefix up to MR rows; emit one reply each,
        # BUT the reply channel is finite (RP). If draining would overflow the reply
        # channel, the server's send_multipart BLOCKS partway. Model: the server can
        # only place as many replies as there is room; if it cannot place all, the
        # remaining drained requests' replies are STUCK (server blocked in send).
        # We model the server send as ATOMIC per group but capacity-limited: it
        # drains and replies only if rep channel has room for all; otherwise it
        # drains what fits the reply channel. To stay faithful & simple we require
        # rep room for the whole drained group; if not enough room, the server
        # drains FEWER (only those whose replies fit). A req left undrained stays
        # queued; a reply that cannot be placed leaves the server effectively
        # blocked (we encode by limiting drained count to rep free slots).
        chg = {f"req{c}" for c in range(RQ)} | {f"rep{c}" for c in range(RP)}
        guard = And(a["srv"] == 0, any_req(a))
        rep_free = RP - rep_used(a)
        # drained[c]: position c is drained iff prefix-rows-before < MR AND the
        # number of drained-before < rep_free (reply must have a home).
        drained = []
        for c in range(RQ):
            pref_rows = Sum([If(a[f"req{cc}"] > 0, a[f"req{cc}"], 0) for cc in range(c)]) if c > 0 else 0
            dbefore = Sum([If(drained[cc], 1, 0) for cc in range(c)]) if c > 0 else 0
            drained.append(And(a[f"req{c}"] > 0, pref_rows < MR, dbefore < rep_free))
        body = []
        for c in range(RQ):
            body.append(b[f"req{c}"] == If(drained[c], 0, a[f"req{c}"]))
        def pth_drained(p):
            expr = 0
            for c in range(RQ):
                before = Sum([If(drained[cc], 1, 0) for cc in range(c)]) if c > 0 else 0
                expr = If(And(drained[c], before == p), a[f"req{c}"], expr)
            return expr
        for p in range(RP):
            existing = a[f"rep{p}"] > 0
            empties_before = Sum([If(a[f"rep{cc}"] == 0, 1, 0) for cc in range(p)]) if p > 0 else 0
            is_empty = a[f"rep{p}"] == 0
            body.append(b[f"rep{p}"] == If(existing, a[f"rep{p}"], If(is_empty, pth_drained(empties_before), a[f"rep{p}"])))
        # guard the server only fires if it makes progress (drains >=1)
        drains_one = Or(*drained)
        opts.append(And(guard, drains_one, And(*body), And(*fr(a, b, chg))))

        s.add(Or(*opts))

    last = S[steps]
    rc = ready_count(last)
    no_PI = Not(And(last["pc"] == PC_ISSUE, rc > 0, last["inflight"] < D, req_used(last) < RQ))
    no_PISF = Not(And(last["pc"] == PC_ISSUE, rc > 0, last["inflight"] < D, req_used(last) == RQ))
    no_PSC = Not(And(last["pc"] == PC_SEND, req_used(last) < RQ))
    no_PD = Not(And(last["pc"] == PC_ISSUE, Or(rc == 0, last["inflight"] == D)))
    no_PR = Not(And(last["pc"] == PC_RECV, any_rep(last)))
    # server can fire iff any_req and it can drain >=1 with rep room
    rep_free_l = RP - rep_used(last)
    srv_can = And(last["srv"] == 0, any_req(last), rep_free_l > 0)
    no_SD = Not(srv_can)
    stuck = And(no_PI, no_PISF, no_PSC, no_PD, no_PR, no_SD)

    work = Or(any_req(last), any_rep(last), last["inflight"] > 0,
              Or([And(last[f"rem{j}"] > 0, last[f"ss{j}"] != 2) for j in range(K)]))
    s.add(stuck, work, last["pc"] != PC_DONE)
    return s, S


def render(model, S, cfg):
    K, RQ, RP = cfg.K, cfg.req_cap, cfg.rep_cap
    pc = {0: "ISSUE", 1: "RECV", 2: "DONE", 3: "SEND-BLK"}
    ss = {0: "PARK", 1: "SUB", 2: "IDLE"}
    out = []
    for t, st in enumerate(S):
        g = lambda k: model.evaluate(st[k], model_completion=True).as_long()
        slots = " ".join(f"s{j}[{ss[g(f'ss{j}')]},r={g(f'rem{j}')}]" for j in range(K))
        reqs = [g(f"req{c}") for c in range(RQ) if g(f"req{c}") > 0]
        reps = [g(f"rep{c}") for c in range(RP) if g(f"rep{c}") > 0]
        out.append(f"  t={t:2d} pc={pc[g('pc')]:8s} inf={g('inflight')} "
                   f"srv={'BLK' if g('srv')==0 else 'drn'} pend={g('pend_rows')} | {slots} | req={reqs} rep={reps}")
    return "\n".join(out)


if __name__ == "__main__":
    cfg = Config(K=int(sys.argv[1]) if len(sys.argv) > 1 else 3,
                 D=int(sys.argv[2]) if len(sys.argv) > 2 else 8,
                 plies=int(sys.argv[3]) if len(sys.argv) > 3 else 2,
                 req_cap=int(sys.argv[4]) if len(sys.argv) > 4 else 1,
                 rep_cap=int(sys.argv[5]) if len(sys.argv) > 5 else 1,
                 max_rows=int(sys.argv[6]) if len(sys.argv) > 6 else 2)
    depth = int(sys.argv[7]) if len(sys.argv) > 7 else 16
    s, S = build(cfg, depth)
    r = s.check()
    print(f"{cfg} depth={depth} -> {r}")
    if r == sat:
        print("DEADLOCK reachable:")
        print(render(s.model(), S, cfg))
