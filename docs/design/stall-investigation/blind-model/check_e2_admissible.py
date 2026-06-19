#!/usr/bin/env python3
"""
check_e2_admissible.py — a minimal bounded Z3 confirmation that representative execution E2
(staggered arrivals, D>1 pipeline, ONE out-of-order reply) is ADMISSIBLE under the producer-side
causal constraints derived in model-producer-pacing.md. Confirmation of the theory, NOT its source.

Setup (T=1 thread, K=2 slots, D=2): two slots park (slot 0 then slot 1), each its own message
(corr c0, c1), both outstanding (D=2). The server (single-threaded) runs forwards serially; we
assert the producer CAN observe c1's reply BEFORE c0's (out-of-order, DOF-3) while every causal
necessity from the model holds:
  - durations positive (delta_i > 0, sigma_k > 0)
  - submit precedes its own recv (round-trip)
  - a reply follows the forward that produced it; forwards are totally ordered (one server thread)
  - per-slot reply-dependence is not exercised here (single leaf per slot) — kept minimal.

SAT => the out-of-order pipelined interleaving is representable. (We also assert it is NOT forced:
a FIFO interleaving is separately SAT, so the model leaves the order free, not pinned.)

Public Domain (The Unlicense).
"""
from z3 import Real, Solver, And, sat

def admissible(out_of_order: bool) -> bool:
    s = Solver()
    # think-times (slot parks) — positive; slot 0 parks before slot 1 (staggered, DOF-2)
    d0, d1 = Real('d0'), Real('d1')
    # submit times of the two coalesced messages (one slot each here)
    sub0, sub1 = Real('sub0'), Real('sub1')
    # the two forwards the server runs (serial, single-threaded): start/complete
    fstart_a, fcomp_a = Real('fstart_a'), Real('fcomp_a')   # forward that produces c1's reply (lands first if OOO)
    fstart_b, fcomp_b = Real('fstart_b'), Real('fcomp_b')   # forward that produces c0's reply
    rec_first, rec_second = Real('rec_first'), Real('rec_second')  # producer recv events
    sig_a, sig_b = Real('sig_a'), Real('sig_b')  # service durations

    s.add(d0 > 0, d1 > 0, sig_a > 0, sig_b > 0)
    # staggered arrival -> submit order: slot0 parks+submits, then slot1
    s.add(sub0 == d0, sub1 == d0 + d1, sub0 < sub1)
    # service durations
    s.add(fcomp_a == fstart_a + sig_a, fcomp_b == fstart_b + sig_b)
    # a forward starts only after the rows it drains were submitted
    if out_of_order:
        # forward A drains c1 (the later submit), B drains c0 (the earlier submit)
        s.add(fstart_a >= sub1, fstart_b >= sub0)
        # the producer observes A's reply (c1) first, then B's (c0)
        s.add(rec_first >= fcomp_a, rec_second >= fcomp_b)
        s.add(rec_first < rec_second)              # OUT OF ORDER: c1 before c0
    else:
        s.add(fstart_a >= sub0, fstart_b >= sub1)  # FIFO: forward A drains c0, B drains c1
        s.add(rec_first >= fcomp_a, rec_second >= fcomp_b)
        s.add(rec_first < rec_second)
    # forwards are totally ordered on the single server thread (one at a time)
    s.add(fcomp_a <= fstart_b)
    # round-trip: every recv follows its corresponding submit (already implied transitively)
    s.add(rec_first > sub0, rec_second > sub0)
    return s.check() == sat

ooo = admissible(out_of_order=True)
fifo = admissible(out_of_order=False)
print(f"E2 out-of-order pipelined interleaving admissible: {ooo}")
print(f"FIFO interleaving also admissible (order is free, not pinned): {fifo}")
print("RESULT:", "PASS" if (ooo and fifo) else "FAIL")
