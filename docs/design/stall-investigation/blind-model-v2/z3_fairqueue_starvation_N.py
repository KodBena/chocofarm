# Confirmation (NOT the source of trust): the N-parametric ROUTER fair-queue starvation question.
#
# Question: at large N each producer thread's single message is fatter (B up to K = N*base rows),
# so a server _drain can hit `total_rows >= max_batch` (inference_server.py:171) after FEWER peers'
# messages, DEFERRING the rest to the next drain. Under sustained saturation as N grows, can this
# starve one slow producer thread of forward slots -- i.e. defer ITS one queued message forever?
#
# We encode the server-side drain over T peers as the libzmq ROUTER fair-queue (fq.cpp) round-robin:
#   - a per-peer input queue holds at most ONE message (proven: inflight_msgs in {0,1} per thread,
#     runner_wire_batched.cpp issue-gathers-ALL + recv-only-re-parks; inflight_le1_check.py UNSAT).
#   - _drain pulls in round-robin pointer order, skipping empty pipes, appending whole messages,
#     until total_rows >= max_batch (loop-top cap, :171) OR no active pipe remains (zmq.Again, :174).
#   - the rotation pointer PERSISTS across drains (it is the socket's fq state, not reset per drain).
#   - EVERY drained peer is answered this cycle (serve_forever :219-225 forward+scatter, G5), after
#     which it may (adversarially, immediately) re-park and re-enqueue a NEW fat message.
#
# Saturation/adversary: we let every served peer instantly re-enqueue a max_batch-sized message
# (worst case for the cap). We ask: is there a schedule where one fixed peer p* is NEVER drained
# over the bounded horizon while it always has a message queued? If the fair-queue is faithful,
# this must be UNSAT for a horizon long enough to force a full rotation: p* is reached within one
# rotation regardless of how fat the others' messages are.
#
# This is the DECISIVE distinction the question turns on: deferral (yes, grows with N) vs.
# starvation (no -- fair-queue rotation bounds the wait to one rotation).

from z3 import (Solver, Int, Bool, Array, IntSort, BoolSort, Function, Or, And, Not,
                Implies, If, Sum, sat, unsat)

# ---- parameters (kept tiny; the argument is parameter-shape-independent) ----
T   = 4     # producer peers (= pool_threads). p* = peer 0 is the "slow" one we test for starvation.
M   = 4     # max_batch in "message units": with N large, ONE peer message can be >= M, so a single
            #   peer message can fill the cap. We model B_peer = M for EVERY peer (worst-case fatness,
            #   the large-N saturated regime). So each drain admits exactly ONE peer then stops (:171).
H   = 2 * T # horizon in drains; one full fair rotation over T active pipes takes <= T drains.

s = Solver()

# rr[t] = the fair-queue round-robin pointer at the START of drain t (a peer index in [0,T)).
rr   = [Int(f"rr_{t}") for t in range(H + 1)]
# queued[t][p] = peer p has its (single) message sitting in the ROUTER input queue at start of drain t.
queued = [[Bool(f"q_{t}_{p}") for p in range(T)] for t in range(H + 1)]
# served[t] = the peer index drained at drain t (each drain admits exactly one peer because B=M=cap).
served = [Int(f"served_{t}") for t in range(H)]

# Initial: pointer somewhere; ALL peers queued (sustained saturation: everyone always has work).
s.add(rr[0] >= 0, rr[0] < T)
for p in range(T):
    s.add(queued[0][p] == True)

def faircheck(t):
    # served[t] is the FIRST active (queued) peer at or after rr[t] in cyclic order -- exactly the
    # libzmq fair-queue: advance the pointer over inactive pipes, deliver from the first active one.
    # Enumerate the concrete pointer value r0 so all indices become Python ints.
    cons = []
    for r0 in range(T):
        for off in range(T):
            picked = (r0 + off) % T
            earlier_empty = And(*[Not(queued[t][(r0 + j) % T]) for j in range(off)]) if off > 0 else True
            cons.append(Implies(And(rr[t] == r0, earlier_empty, queued[t][picked]),
                                served[t] == picked))
    return And(*cons)

for t in range(H):
    s.add(rr[t] >= 0, rr[t] < T)
    s.add(served[t] >= 0, served[t] < T)
    s.add(Or(*[queued[t][p] for p in range(T)]))        # saturation: always >=1 active pipe
    s.add(faircheck(t))
    # (faircheck already forces served[t] to be a queued peer, so "serve only queued" is implied.)

    # pointer advances PAST the served peer (fq advances after delivery): rr[t+1] = (served+1)%T.
    for sv in range(T):
        s.add(Implies(served[t] == sv, rr[t + 1] == (sv + 1) % T))

    # queue update: the served peer is retired this cycle (forward+scatter), then ADVERSARIALLY
    # re-enqueues immediately (worst case for starvation). All others keep their queued message.
    # We let the served peer's re-enqueue be free (True or False); to maximize starvation pressure
    # we allow it to come right back. Either way p*'s message (if queued and not served) persists.
    for p in range(T):
        reenq = Bool(f"reenq_{t}_{p}")
        s.add(Implies(served[t] == p, queued[t + 1][p] == reenq))   # served peer: re-enqueue free
        s.add(Implies(served[t] != p, queued[t + 1][p] == queued[t][p]))  # others unchanged

# STARVATION GOAL: peer 0 (p*) is queued at every step but NEVER served over the whole horizon.
s.add(*[queued[t][0] for t in range(H + 1)])      # p* always has a pending message
s.add(*[served[t] != 0 for t in range(H)])        # p* is never drained

print(f"T={T} M(cap in msg-units)={M} horizon={H} drains")
print("asking: can the slow peer p*=0 be queued the whole time yet NEVER drained (starvation)?")
r = s.check()
print("result:", r)
if r == unsat:
    print("UNSAT: under fair-queue rotation + inflight<=1, a continuously-queued peer is drained")
    print("       within one rotation (<= T drains). DEFERRAL is bounded; STARVATION is impossible.")
else:
    print("SAT: a starvation schedule exists (would refute the fair-queue anti-starvation claim)")
    m = s.model()
    print("served sequence:", [m.evaluate(served[t]) for t in range(H)])
