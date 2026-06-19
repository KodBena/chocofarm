#!/usr/bin/env python3
"""
producer_transport_check.smt2.py — a MINIMAL Z3 confirmation that ONE representative execution of the
C++ producer-side transport model (model-producer-transport.md, E1+E3) is admissible. Confirmation of
the derivation, never its source (per the assignment). Bounded; runs in <1s.

Scenario: one worker thread, D=2 in-flight cap, two coalesced messages outstanding (corr c1 over slots
{2,5}, corr c2 over slot {3}); the single-threaded sink batches both queued requests into ONE forward and
replies OUT OF ORDER (c2 before c1). We assert the causal laws the code/transport impose and check SAT:
  - positivity of every search interval and the forward duration,
  - request-before-reply (a reply cannot precede its request's arrival),
  - reply-after-forward (a reply cannot precede the forward that produced it),
  - resume-after-reply (a slot's next request is gated on its previous reply),
  - inflight depth never exceeds D and is non-negative,
  - the producer's recv order is whatever the sink chose (here c2 then c1) — order-agnostic matching.
A SAT result confirms the model admits this execution; the model is the source of truth, this only checks
the constraints are mutually consistent (i.e. the derivation did not over-constrain itself into vacuity).

Public Domain (The Unlicense).
"""
from z3 import Real, Bool, Int, Solver, And, Or, sat

s = Solver()

# --- event times (seconds, real) ---
# submit times of the two messages (issue_one calls), reply send times, reply recv times.
t_sub1 = Real("t_sub1")   # producer sends c1 (slots {2,5})
t_sub2 = Real("t_sub2")   # producer sends c2 (slot {3})
t_arr1 = Real("t_arr1")   # c1 arrives at sink ROUTER
t_arr2 = Real("t_arr2")   # c2 arrives at sink ROUTER
t_drain = Real("t_drain") # sink drains both (greedy-drain snapshot)
tau_fwd = Real("tau_fwd") # ONE padded forward over the co-batched rows
t_fdone = Real("t_fdone") # forward completes (np.asarray pull done)
t_snd2 = Real("t_snd2")   # sink send_multipart for c2 (replies c2 FIRST — out of order)
t_snd1 = Real("t_snd1")   # sink send_multipart for c1
t_rcv2 = Real("t_rcv2")   # producer recv_batch returns c2
t_rcv1 = Real("t_rcv1")   # producer recv_batch returns c1
RCVTIMEO = 15.0           # default timeout_ms=15000 -> 15 s

# search intervals: first leaves are reply-gated for the NEXT ply (resume-after-reply).
delta_resume3 = Real("delta_resume3")  # slot 3's next-leaf search after its reply (resume_with)

# --- causal necessities (each a named law in the model) ---
# positivity
s.add(tau_fwd > 0, delta_resume3 > 0)
# request-before-reply: arrival after submit (one-way network delay > 0)
s.add(t_arr1 > t_sub1, t_arr2 > t_sub2)
# the sink drains both only after both have arrived (greedy drain snapshots the queue)
s.add(t_drain >= t_arr1, t_drain >= t_arr2)
# reply-after-forward: forward starts at/after drain, completes after tau_fwd, replies after completion
s.add(t_fdone == t_drain + tau_fwd)
s.add(t_snd2 >= t_fdone, t_snd1 >= t_fdone)
# OUT-OF-ORDER reply choice (the sink's free latitude, R5/DOF-4): c2 sent before c1
s.add(t_snd2 < t_snd1)
# producer recv after the corresponding send (network delay > 0)
s.add(t_rcv2 > t_snd2, t_rcv1 > t_snd1)
# producer recv_batch returns c2 first then c1 (it blocks for whichever lands first; out-of-order ok)
s.add(t_rcv2 < t_rcv1)
# resume-after-reply: slot 3's next request (a NEW submit) is gated on its reply t_rcv2
t_sub3_next = Real("t_sub3_next")
s.add(t_sub3_next == t_rcv2 + delta_resume3)
# both round-trips are within RCVTIMEO (the common, reply-not-timeout case)
s.add(t_rcv1 - t_sub1 < RCVTIMEO, t_rcv2 - t_sub2 < RCVTIMEO)

# --- in-flight depth invariant: 0 <= inflight_msgs <= D=2 over the trace ---
D = 2
# inflight goes 0 ->(sub1) 1 ->(sub2) 2 ->(rcv2) 1 ->(rcv1) 0 ; encode the peak constraint.
inflight_after_sub2 = Int("inflight_after_sub2")
s.add(inflight_after_sub2 == 2, inflight_after_sub2 <= D, inflight_after_sub2 >= 0)
# the second submit happens only while inflight < D (issue guard inflight_msgs < D)
s.add(t_sub2 > t_sub1)  # c2 issued after c1 while depth 1 < D

print("checking representative execution E1+E3 (out-of-order reply, D=2, causal laws)...")
r = s.check()
print("result:", r)
if r == sat:
    m = s.model()
    print("SAT — execution is ADMISSIBLE. One witness model:")
    for d in sorted(m.decls(), key=lambda d: d.name()):
        print(f"  {d.name()} = {m[d]}")
else:
    print("UNSAT/unknown — the model over-constrained this execution (would be a fidelity bug).")
