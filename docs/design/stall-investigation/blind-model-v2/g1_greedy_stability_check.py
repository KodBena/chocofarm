# G-1 confirmation (NOT the source of trust; the derivation in derive-G1-greedy-stability.md is).
#
# Claim under test (the representative execution + its negation): under the production greedy drain's
# batch recurrence as derived from the cleanroom code, with a deliberately injected positive σ→arrivals
# coupling (a "spike" cycle that lets MORE rows accumulate), the offered batch can NEVER exceed the
# absorbing ceiling  ceil = min(T*K, max_batch + K - 1)  with K = N*base. I.e. "B_{i+1} > ceil" is UNSAT,
# at several N. This is the precise non-divergence fact the question asks about.
#
# Code grounding of each constraint (cleanroom):
#  - DRAIN transfer: B_{i+1} = min(A_i pulled up to cap-with-overshoot).  inference_server.py:171,184-185
#       * B_{i+1} <= A_i                       (non-blocking pull stops on Again, :172-174)
#       * B_{i+1} <= max_batch + (m_i - 1)     (loop-top cap + one message past, :171,184-185)
#       * m_i <= K (RELY-B, runner_wire_batched.cpp:286,437-444)
#  - OL ceiling: A_i <= T*K, INDEPENDENT of S_i  (RELY-A depth-1 + RELY-B; runner 447,456,458,474)
#       We model the adversary's freedom over A_i as: A_i may be ANY value in [0, T*K], i.e. a slow/
#       spiky forward can drive arrivals up to BUT NOT BEYOND T*K. This is strictly LOOSER than the
#       code (the code couples A_i to actual timing); looser-unsat => code-unsat a fortiori.
#  - The "spike": we let the adversary choose S_i freely (>0) and choose A_i adversarially high; the
#    point is that NO choice of S_i / A_i breaks the ceiling, because the ceiling does not depend on them.
#
# We check two things at each N:
#   (1) DIVERGENCE is UNSAT:  exists a reachable cycle with B_{i+1} > ceil  -> must be unsat.
#   (2) The representative execution is SAT (admissibility witness): a sub-cap small batch (B_i<<max_batch)
#       can be followed by an overshoot batch B_{i+1} in (max_batch, max_batch+K-1] when N is large enough
#       that K>1 over the cap -> must be sat (shows the model is not vacuously constrained / the overshoot
#       regime is genuinely reachable, matching CF-7).

from z3 import Solver, Int, Function, IntSort, BoolSort, ForAll, Implies, And, Or, sat, unsat

def ceiling(T, K, max_batch):
    return min(T * K, max_batch + K - 1)

def check(N, base, T, max_batch, STEPS=6):
    K = N * base
    ceil = ceiling(T, K, max_batch)

    # ---- (1) divergence is UNSAT ----
    s = Solver()
    B = [Int(f"B_{i}") for i in range(STEPS + 1)]
    A = [Int(f"A_{i}") for i in range(STEPS)]      # arrivals available before drain i+1
    m = [Int(f"m_{i}") for i in range(STEPS)]      # largest available message rows at cycle i
    s.add(B[0] >= 1, B[0] <= ceil)                 # start admissible
    for i in range(STEPS):
        # adversary: arrivals anywhere in [0, T*K] (looser than code: S-independent upper bound)
        s.add(A[i] >= 0, A[i] <= T * K)
        # largest available message: 1..K (RELY-B); if A[i]==0 there is no message, B_{i+1}=0-or-poll,
        # but the server only forwards when drained>=1, so model the FORWARD cycles (A[i]>=1).
        s.add(m[i] >= 1, m[i] <= K)
        s.add(A[i] >= 1)                           # a forward cycle has >=1 row drained
        # cap-with-overshoot bound and arrival bound (DRAIN):
        cap_i = max_batch + m[i] - 1
        # B_{i+1} = min(A[i], cap_i)  -- but we only need the UPPER bounds for non-divergence:
        s.add(B[i + 1] <= A[i])
        s.add(B[i + 1] <= cap_i)
        s.add(B[i + 1] >= 1)
    # negation of the invariant: some cycle exceeds the absorbing ceiling
    s.add(Or(*[B[i] > ceil for i in range(1, STEPS + 1)]))
    r1 = s.check()

    # ---- (2) representative overshoot execution is SAT (reachability witness) ----
    s2 = Solver()
    b0 = Int("b0"); b1 = Int("b1"); a0 = Int("a0"); m0 = Int("m0")
    s2.add(b0 >= 1, b0 <= max_batch)               # cycle 0 sub-cap (small)
    s2.add(b0 <= 3)                                 # genuinely small (a lone-ish early batch)
    s2.add(a0 >= 1, a0 <= T * K)
    s2.add(m0 >= 1, m0 <= K)
    s2.add(b1 == If_min(a0, max_batch + m0 - 1))
    # Overshoot (b1 > max_batch) is reachable iff the cap-with-overshoot ceiling exceeds max_batch
    # (needs K>1) AND the arrival ceiling can actually present > max_batch rows (needs T*K > max_batch):
    # b1 = min(a0, max_batch+m0-1) with a0<=T*K, m0<=K. To get b1>max_batch you need both
    # a0>max_batch (=> T*K>max_batch) and max_batch+m0-1>max_batch (=> m0>1 => K>1).
    overshoot_possible = (T * K > max_batch) and (K > 1)
    if overshoot_possible:
        s2.add(b1 > max_batch)                      # demand an overshoot in cycle 1
        s2.add(b1 <= max_batch + K - 1)
    r2 = s2.check()
    return K, ceil, r1, r2, overshoot_possible

# small helper: min as an Int expression
from z3 import If
def If_min(x, y):
    return If(x <= y, x, y)

CONFIGS = [
    # (N, base, T, max_batch)   base = ceil(pool_batch/T)
    (1,  8, 4, 256),   # prior baseline: K=8 << 256, overshoot UNREACHABLE
    (8,  8, 4, 256),   # K=64,  still < max_batch
    (33, 8, 4, 256),   # K=264 > 256: overshoot reachable (N > max_batch/base = 32)
    (75, 8, 4, 512),   # bench max_batch=512, K=600 > 512: the question's K>max_batch overshoot
    (200, 2, 4, 256),  # large N, small base: K=400 > 256
]

print("G-1 bounded confirmation (z3) — production greedy drain batch boundedness, parametric in N")
print(f"{'N':>4} {'base':>4} {'T':>3} {'max_batch':>9} {'K=N*base':>9} {'ceiling':>8} "
      f"{'divergence':>11} {'overshoot-reachable':>20}")
all_ok = True
for (N, base, T, mb) in CONFIGS:
    K, ceil, r1, r2, reachable = check(N, base, T, mb)
    div = "UNSAT" if r1 == unsat else f"SAT(!!)"
    if reachable:
        reach = "SAT" if r2 == sat else "unsat(!!)"
        reach_ok = (r2 == sat)
    else:
        reach = "n/a (K<=max_batch)"
        reach_ok = True
    ok = (r1 == unsat) and reach_ok
    all_ok = all_ok and ok
    print(f"{N:>4} {base:>4} {T:>3} {mb:>9} {K:>9} {ceil:>8} {div:>11} {reach:>20}")

print()
print("Expected: divergence UNSAT at EVERY config (no batch exceeds the absorbing ceiling at any N),")
print("and the overshoot regime SAT exactly where K>max_batch (reachable as N grows, per CF-7).")
print("RESULT:", "ALL CHECKS PASS" if all_ok else "A CHECK FAILED — investigate")
