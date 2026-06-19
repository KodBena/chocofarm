"""Bounded admissibility check for the C++ producer-pacing model.

Confirmation only (NOT the source of trust). Encodes one representative execution of
run_episodes_wire_pipelined (cleanroom cpp/src/runner_wire_batched.cpp) and asks Z3 whether the
model admits it. Public Domain (The Unlicense).

Scenario (parametric instance: T=1 thread, K=2 slots, D=1 in-flight-message cap):
  - PRIMING coalesces both eligible first-leaf parks into ONE message of B=2 rows
    (runner_wire_batched.cpp:437-444 gathers all is_ready slots; D=1 so only one issue before recv).
    Wait: with D=1 the prime issues exactly one message; that message coalesces ALL currently
    eligible slots (both), so B=2. Confirms DOF-2 coalescing across slots even at D=1.
  - The single message carries slots {0,1} under one corr c0.
  - The reply to c0 arrives (recv_batch), scatters to both slots.
  - Slot 1's reply is consumed before slot 0's within the same reply vector (out-of-order-by-corr is
    trivial with one corr; we instead model a 2-message variant to exercise reorder).

We model a 2-message variant to exercise DOF-4 (reply order != send order) with D=2:
  send msg c0 (slot0, leaf k=0), send msg c1 (slot1, leaf k=0); server replies c1 THEN c0.
  Constraints checked admissible:
    (A) positivity of park intervals (C1): every park duration > 0
    (B) reply-causality (C2/G4): slot s's leaf-1 request emitted only after its leaf-0 reply received
    (C) forward-causality: a reply's recv time > the forward's start > the request's send time
    (D) out-of-order: recv(c1) < recv(c0) although send(c0) < send(c1)
    (E) in-flight cap D=2 never exceeded
"""
from __future__ import annotations

import z3

s = z3.Solver()

# Times (reals). One worker thread, two slots, D=2.
# Events per slot per leaf: park (eligible), send (in a coalesced msg), forward, recv (reply), resume.
# Leaf indices: slot s, leaf k in {0,1}.

def R(name):
    return z3.Real(name)

# park[s][k] : wall time slot s parks at leaf k (becomes ELIGIBLE)
park = {(sl, k): R(f"park_{sl}_{k}") for sl in (0, 1) for k in (0, 1)}
# send[s][k] : time the message carrying slot s leaf k is sent (submit_batch)
send = {(sl, k): R(f"send_{sl}_{k}") for sl in (0, 1) for k in (0, 1)}
# fwd[c] : server forward-start time for the message carrying corr c
# recv[s][k]: time the producer receives the reply covering slot s leaf k
recv = {(sl, k): R(f"recv_{sl}_{k}") for sl in (0, 1) for k in (0, 1)}
# resume[s][k]: time resume_with fed the reply back into the coroutine
resume = {(sl, k): R(f"resume_{sl}_{k}") for sl in (0, 1) for k in (0, 1)}

# Two corr ids for the two leaf-0 messages (send order c0 < c1), then leaf-1 messages.
fwd = {c: R(f"fwd_{c}") for c in ("c0", "c1", "c10", "c11")}
# c0 carries (slot0,leaf0); c1 carries (slot1,leaf0); c10 carries (slot0,leaf1); c11 carries (slot1,leaf1)
carry = {"c0": (0, 0), "c1": (1, 0), "c10": (0, 1), "c11": (1, 1)}

# (A) positivity: a park can only follow its predecessor by a strictly positive search interval.
# Leaf-0 parks happen at episode start (fill->spawn_ply). Give them positive absolute times.
s.add(park[(0, 0)] > 0, park[(1, 0)] > 0)
# Leaf-1 park follows resume of leaf-0 by a strictly positive interval (C1 positivity + search work).
for sl in (0, 1):
    s.add(park[(sl, 1)] > resume[(sl, 0)])

# A slot is sent only after it has parked (issue_one gathers is_ready = parked & !submitted).
for sl in (0, 1):
    for k in (0, 1):
        s.add(send[(sl, k)] >= park[(sl, k)])

# (B) reply-causality (C2/G4): leaf-1 of slot s cannot even PARK before leaf-0's reply is resumed,
# hence cannot be SENT before. Already encoded via park[sl,1] > resume[sl,0] and send>=park.
# resume happens at/after recv.
for sl in (0, 1):
    for k in (0, 1):
        s.add(resume[(sl, k)] >= recv[(sl, k)])

# (C) forward-causality: a forward starts after its message is sent; a reply is received after the
# forward that produced it. (Reply cannot precede the forward.)
for c, (sl, k) in carry.items():
    s.add(fwd[c] >= send[(sl, k)])
    s.add(recv[(sl, k)] > fwd[c])

# Single-threaded server: forwards are serialized. Send order is c0 < c1; server may forward in any
# order but one at a time. We pick the out-of-order witness: c1 forwarded before c0.
s.add(send[(0, 0)] < send[(1, 0)])      # send order c0 < c1 (corr_seq monotone, line 84)
s.add(fwd["c1"] < fwd["c0"])            # server chooses to forward c1 first
# serialization: no two forwards overlap trivially encoded by distinct ordered starts among the two.
s.add(fwd["c0"] != fwd["c1"])

# (D) out-of-order reply: recv(slot1,leaf0) < recv(slot0,leaf0) although send(c0)<send(c1).
s.add(recv[(1, 0)] < recv[(0, 0)])

# (E) in-flight message cap D=2: at most 2 messages outstanding. With 2 leaf-0 messages sent before
# any recv, inflight reaches 2 (== D) and the prime loop stops issuing (line 456). The leaf-1 messages
# are issued only on refill after a recv decrements inflight (line 460,474). Encode: each leaf-1 send
# happens after at least one leaf-0 recv.
s.add(send[(0, 1)] > z3.If(recv[(0, 0)] < recv[(1, 0)], recv[(0, 0)], recv[(1, 0)]))
s.add(send[(1, 1)] > z3.If(recv[(0, 0)] < recv[(1, 0)], recv[(0, 0)], recv[(1, 0)]))

res = s.check()
print("admissible:", res)
if res == z3.sat:
    m = s.model()
    keys = sorted((str(d), m[d]) for d in m.decls())
    for name, val in keys:
        print(f"  {name} = {val}")
else:
    print("UNSAT core would indicate an over-constrained model bug:")
    print(s.unsat_core())
