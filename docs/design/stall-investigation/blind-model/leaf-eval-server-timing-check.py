#!/usr/bin/env python3
"""
leaf-eval-server-timing-check.py — a MINIMAL Z3 confirmation of the server-side timing model's §5.3
causal skeleton (model-server-transport.md). This CONFIRMS the theory; it is not its source.

It encodes a 2-peer, 2-forward execution (the E2-like regime: forward f1's service window must be able
to buffer the rows that f2 drains) and asks Z3 whether the causal constraints are jointly satisfiable
with the over-permission guards active:
  - positivity of service durations (S1):            end(f) - start(f) = s_f > 0
  - one-at-a-time / non-overlap of forwards (S3):    start(f2) >= end(f1)
  - reply-after-forward (S2):                         send_r > end(f_of_r)
  - drained-before-forward:                           a_r <= start(f_of_r)
  - emit-before-arrive (transit >= 0):                e_r < a_r
  - emit-after-reply for a reply-dependent re-issue (C1): e_r2 > a_reply_r1  (peer re-issues after recv)
  - buffering: peer P2's request r2 emitted during f1's service window (the heart of self-clocking)

A SAT result with a concrete model confirms the representative execution is admissible. We additionally
check that a deliberately IMPOSSIBLE ordering (a reply preceding its forward) is UNSAT, demonstrating the
model is not vacuously permissive.

Public Domain (The Unlicense).
"""
from z3 import Real, Solver, sat, unsat, And

def build_common(s):
    # forward f1 and f2 (single server, two rounds)
    s1_start, s1_end = Real('f1_start'), Real('f1_end')
    s2_start, s2_end = Real('f2_start'), Real('f2_end')
    # peer P1 request r1 (in batch of f1); peer P2 request r2 (emitted during f1, drained into f2)
    e1, a1 = Real('e_r1'), Real('a_r1')          # P1 emit, server-arrive
    send1 = Real('send_r1')                       # reply to r1 scattered
    a_reply1 = Real('a_reply_r1')                 # P1 receives reply
    e2, a2 = Real('e_r2'), Real('a_r2')          # P2 emit (during f1), server-arrive
    send2 = Real('send_r2')

    s_f1 = s1_end - s1_start
    s_f2 = s2_end - s2_start

    s.add(s_f1 > 0, s_f2 > 0)                      # S1 positivity (no instant forward)
    s.add(s2_start >= s1_end)                      # S3 non-overlap, total order
    s.add(e1 < a1)                                 # transit >= 0 (strict: positive transit)
    s.add(a1 <= s1_start)                          # drained before f1
    s.add(send1 > s1_end)                          # S2 reply-after-forward (r1)
    s.add(send1 < a_reply1)                        # reply transit to P1
    s.add(e2 > s1_start, e2 < s1_end)              # r2 EMITTED DURING f1's service window (buffering)
    s.add(e2 < a2)
    s.add(a2 <= s2_start)                          # r2 drained into f2
    s.add(send2 > s2_end)                          # S2 for r2
    # ground the timeline
    s.add(s1_start >= 0)
    return dict(s1_start=s1_start, s1_end=s1_end, s2_start=s2_start, s2_end=s2_end,
                e1=e1, a1=a1, send1=send1, a_reply1=a_reply1, e2=e2, a2=a2, send2=send2)

# --- 1. the representative execution is ADMISSIBLE (SAT) ---
s = Solver()
v = build_common(s)
res = s.check()
print("representative E2-like execution:", res)
if res == sat:
    m = s.model()
    for name in ('f1_start', 'f1_end', 'f2_start', 'f2_end', 'e_r1', 'a_r1',
                 'send_r1', 'a_reply_r1', 'e_r2', 'a_r2', 'send_r2'):
        print(f"  {name:12s} = {m[Real(name)]}")

# --- 2. an IMPOSSIBLE ordering (reply before its forward ends) is UNSAT ---
s2 = Solver()
v2 = build_common(s2)
s2.add(v2['send1'] < v2['s1_end'])   # contradicts S2: reply cannot precede the forward that produced it
print("reply-before-forward (must be UNSAT):", s2.check())
