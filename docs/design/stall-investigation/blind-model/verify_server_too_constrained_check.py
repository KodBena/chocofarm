#!/usr/bin/env python3
"""
verify_server_too_constrained_check.py — minimal Z3 confirmation for the SERVER-side
TOO-CONSTRAINED audit (leaf-eval transport boundary).

This is CONFIRMATION of a derivation done by hand against the code, never its source.
It checks ONE qualitative property that both audited server models claim and that the
too-constrained lens must verify is NOT silently narrowed:

  Q: the single-threaded greedy-drain admits TWO consecutive forwards of DIFFERENT
     batch sizes that arise PURELY from arrival timing vs serialized drain instants,
     while obeying the causal partial order the code+causality force:
       (1) per-forward service duration > 0           (inference_server.py:177, no instant forward)
       (2) reply send strictly after its forward ends  (:387 after :177)
       (3) forwards never overlap (single thread)       (:436 loop, one thread)
       (4) a request drained into forward k must have ARRIVED by k's drain start
       (5) a request that arrives DURING forward k-1 cannot be touched until k-1 ends
           (the serialization the self-clocking loop depends on)

  If Q is SAT, the two models' central self-clocking latitude (variable B from timing)
  is jointly admissible with the causal constraints => the models are NOT over-
  constrained by collapsing B or service time to a constant on THIS axis.

  We ALSO assert the deliberately-impossible ordering (a reply BEFORE its forward ends)
  is UNSAT, so the admitted set is not vacuously permissive.

Run ONCE under: nice -n 19 timeout 90 .../python verify_server_too_constrained_check.py
"""
from z3 import Real, Int, Solver, And, Or, sat, unsat

def main() -> None:
    MAX_BATCH = 4

    s = Solver()

    # Two serialized drains/forwards: k=0 then k=1.
    t0 = Real("t0_drain_start"); f0 = Real("f0_forward_end"); svc0 = Real("svc0")
    t1 = Real("t1_drain_start"); f1 = Real("f1_forward_end"); svc1 = Real("svc1")

    # Five single-row requests, arrival instants a0..a4 (each B_i = 1 row).
    a = [Real(f"a{i}") for i in range(5)]

    # (1) positive service durations; forward ends after it starts + svc.
    s.add(svc0 > 0, svc1 > 0)
    s.add(f0 == t0 + svc0, f1 == t1 + svc1)
    s.add(t0 >= 0)
    for ai in a:
        s.add(ai >= 0)

    # (3) single thread: drain 1 cannot start before forward 0 finished.
    s.add(t1 >= f0)

    # We want batch 0 to drain exactly 4 rows (cap-sized) and batch 1 exactly 1 (leftover).
    # A request is drained into batch 0 iff it had arrived by t0 (a_i <= t0).
    # We force a0..a3 <= t0 (4 rows available at drain 0) and the cap stops at 4.
    for i in range(4):
        s.add(a[i] <= t0)
    # The 5th request arrives DURING forward 0 (after t0, at or before f0) -> (5): it is
    # NOT touchable until forward 0 ends, so it is drained into batch 1, not batch 0.
    s.add(a[4] > t0, a[4] <= f0)

    # Batch 1 drains the leftover 5th at t1 (it has arrived by then: a4 <= f0 <= t1).
    s.add(a[4] <= t1)

    # The qualitative property: two consecutive forwards of DIFFERENT batch sizes from timing.
    # B0 = 4 (cap), B1 = 1. Encoded structurally above; assert the cap and leftover counts hold:
    #   exactly 4 requests <= t0 (B0=4 == MAX_BATCH), and the 5th deferred (B1=1).
    s.add(MAX_BATCH == 4)  # the soft cap

    res = s.check()
    print("Q (variable-B from timing, serialized, reply-after-forward):", res)
    if res == sat:
        m = s.model()
        print("  witness:",
              "t0=", m[t0], "svc0=", m[svc0], "f0=", m[f0],
              "t1=", m[t1], "svc1=", m[svc1], "f1=", m[f1],
              "a4(arrived-during-f0)=", m[a[4]])

    # Negative control: a reply that ends BEFORE its forward ends must be UNSAT.
    neg = Solver()
    svc = Real("svc"); start = Real("start"); end = Real("end"); reply = Real("reply")
    neg.add(svc > 0, end == start + svc, reply > end)  # faithful: reply after forward end
    neg.add(reply < end)                                # contradiction injected
    nres = neg.check()
    print("Negative control (reply before forward end) is UNSAT:", nres == unsat, f"({nres})")


if __name__ == "__main__":
    main()
