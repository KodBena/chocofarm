#!/usr/bin/env python3
"""
check_composed_admissible.py — bounded Z3 confirmation of the COMPOSED faithful model
(SYNTHESIS.md). Confirmation of the derivation, NOT its source. Run ONE small bounded check.

Two facts the composed model asserts that prior per-side checks did NOT jointly encode:

  (A) ADMISSIBLE — the cross-thread coalescing regime under per-thread depth-1.
      T=2 producer threads, each at depth EXACTLY 1 (one coalesced message outstanding).
      During the server's forward f0 (serving thread 0's message), thread 1's single
      message arrives and is buffered; the next drain coalesces ... but with only thread 1
      ready (thread 0 is blocked awaiting f0's reply), so B1 = thread1's rows. Then f0's
      reply resumes thread 0, it re-issues, and during f1 thread 0's new message buffers.
      We assert: each thread holds at most ONE message outstanding at every instant, the
      server forwards are totally ordered and non-overlapping, replies follow their forward,
      and a thread's re-issue follows its reply (reply-gating). SAT => the composed model
      admits this real execution.

  (B) UNSAT — the per-thread depth-2 execution the audits proved impossible.
      A SINGLE producer thread with TWO messages outstanding simultaneously. The driver's
      coalesce-all issue_one + synchronous recv->resume->refill makes a slot ready only
      AFTER the recv that decremented the one outstanding message, so a single thread's
      second submit cannot precede its first reply. Encoding that control-flow law and
      asking for two simultaneously-outstanding messages on one thread => UNSAT.

The point: the composed model admits the real cross-thread overlap (A) while forbidding the
phantom per-thread pipeline (B) — neither too permissive nor too constrained at the seam.
"""
import z3


def check_A_admissible() -> str:
    s = z3.Solver()
    R = z3.RealSort()

    # Thread 0's message m0: submit, arrive, forward f0 [start,end], reply.
    sub0 = z3.Const("sub0", R); arr0 = z3.Const("arr0", R)
    f0s = z3.Const("f0s", R); f0e = z3.Const("f0e", R); rep0 = z3.Const("rep0", R)
    # Thread 1's message m1.
    sub1 = z3.Const("sub1", R); arr1 = z3.Const("arr1", R)
    f1s = z3.Const("f1s", R); f1e = z3.Const("f1e", R); rep1 = z3.Const("rep1", R)
    # Thread 0's SECOND message m0b (issued after m0's reply — reply-gated).
    sub0b = z3.Const("sub0b", R); arr0b = z3.Const("arr0b", R)
    f2s = z3.Const("f2s", R); f2e = z3.Const("f2e", R); rep2 = z3.Const("rep2", R)

    # positivity: emit precedes arrival; forwards have positive duration.
    for sub, arr in [(sub0, arr0), (sub1, arr1), (sub0b, arr0b)]:
        s.add(sub >= 0, arr > sub)                       # transit > 0
    for fs, fe in [(f0s, f0e), (f1s, f1e), (f2s, f2e)]:
        s.add(fe > fs)                                   # service > 0 (no instant forward)

    # drained-before-forward: a message is in a forward only after it arrived.
    s.add(f0s >= arr0)                                   # f0 serves m0
    s.add(f1s >= arr1)                                   # f1 serves m1
    s.add(f2s >= arr0b)                                  # f2 serves m0b

    # single-threaded server: forwards totally ordered, non-overlapping.
    s.add(f0e <= f1s, f1e <= f2s)

    # reply-after-forward.
    s.add(rep0 >= f0e, rep1 >= f1e, rep2 >= f2e)

    # reply-gating (per producer thread): thread 0's SECOND submit follows its FIRST reply.
    s.add(sub0b > rep0)

    # The cross-thread overlap we want to witness: thread 1's message arrives DURING f0
    # (while thread 0 is blocked on f0's reply) — both threads' single messages coexist
    # outstanding, but each thread holds exactly one.
    s.add(arr1 >= f0s, arr1 <= f0e)                      # m1 buffered during f0
    # depth-1 per thread: m0 is replied (rep0) before m0b is submitted (sub0b) — already above.
    # m1 is the only outstanding message of thread 1 in [arr1, rep1].

    res = s.solve() if hasattr(s, "solve") else s.check()
    if s.check() == z3.sat:
        return "A: SAT (cross-thread depth-1 coalescing overlap is admissible)"
    return "A: " + str(s.check())


def check_B_unsat() -> str:
    s = z3.Solver()
    R = z3.RealSort()
    # ONE producer thread, two submits sub1<sub2, recvs rcv1,rcv2.
    sub1 = z3.Const("sub1", R); rcv1 = z3.Const("rcv1", R)
    sub2 = z3.Const("sub2", R); rcv2 = z3.Const("rcv2", R)
    # round-trip: a submit precedes its own reply.
    s.add(rcv1 > sub1, rcv2 > sub2)
    # The driver's coalesce-all + synchronous recv->resume->refill control-flow LAW:
    # a single thread's SECOND message is issued only AFTER the FIRST message's reply was
    # received (the slot it re-gathers becomes ready only inside the resume that the recv
    # of message 1 triggered). Encode: sub2 > rcv1.
    s.add(sub2 > rcv1)
    # Ask for two messages SIMULTANEOUSLY outstanding on the one thread: there is an instant t
    # with sub1 <= t < rcv1 AND sub2 <= t < rcv2 (both un-replied at once).
    t = z3.Const("t", R)
    s.add(sub1 <= t, t < rcv1, sub2 <= t, t < rcv2)
    if s.check() == z3.unsat:
        return "B: UNSAT (single-thread two-outstanding is impossible — depth identically 1)"
    return "B: " + str(s.check()) + "  (UNEXPECTED — model would be too permissive)"


if __name__ == "__main__":
    print(check_A_admissible())
    print(check_B_unsat())
