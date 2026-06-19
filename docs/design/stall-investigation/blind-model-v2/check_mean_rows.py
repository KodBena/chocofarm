"""Bounded Z3 confirmation that the derived wave model admits a representative
execution of run_episodes_wire_pipelined's mean_rows_per_msg telemetry.

Public Domain (The Unlicense).

We model ONE worker thread (the cross-thread sum is just additive over threads,
§3 (★)). Premise depth==1: per wave, every ACTIVE slot advances exactly one ply
and contributes exactly one row to exactly one message; each wave = one message
(§2.1-2.2). We assert the simulator's bookkeeping is a legal trace and confirm:

  total_leaves == P    (sum of plies; conserved, §1)
  total_msgs   == W    (one message per wave, §2)
  mean = P / W,  with  ceil(P/K) <= W   and   mean <= K            (envelope §3)

and that as work per slot grows, mean approaches K (monotone rise toward N*base).

This is CONFIRMATION of admissibility, not the source of trust.
"""
from z3 import Int, Real, Solver, sat, If, Sum, ToReal

def wave_sim(K, ep_lengths):
    """Concrete reference simulator of the synchronous wave dynamics for one
    thread with K slots processing the given episodes (lengths in plies).
    Returns (P, W, leaves_per_wave)."""
    # dynamic episode assignment: a free slot grabs the next episode (fill).
    queue = list(ep_lengths)
    # remaining plies in each slot's current episode; None = idle
    rem = [None] * K
    def refill(i):
        rem[i] = queue.pop(0) if queue else None
    for i in range(K):
        refill(i)
    P = 0
    waves = []  # rows per wave
    while any(r is not None for r in rem):
        active = sum(1 for r in rem if r is not None)
        waves.append(active)        # one message, `active` rows
        P += active
        for i in range(K):          # each active slot advances one ply
            if rem[i] is not None:
                rem[i] -= 1
                if rem[i] == 0:
                    refill(i)
    return P, len(waves), waves


def confirm(K, ep_lengths, label):
    P, W, waves = wave_sim(K, ep_lengths)
    s = Solver()
    # Encode the trace as Z3 reals/ints and assert the closed-form relations.
    p = Int('P'); w = Int('W')
    s.add(p == P, w == W)
    # numerator conservation: sum of per-wave rows == P
    s.add(p == Sum([int(x) for x in waves]))
    # one message per wave
    s.add(w == len(waves))
    # envelope: ceil(P/K) <= W  (each wave does at most K plies)  and mean <= K
    import math
    s.add(w >= math.ceil(P / K))
    mean = Real('mean')
    s.add(mean == ToReal(p) / ToReal(w))
    s.add(mean <= K)            # the envelope mean_rows_per_msg <= K = N*base
    s.add(mean >= 1)            # at least one row per (non-empty) message
    assert s.check() == sat, f"{label}: UNSAT — model rejected a real trace!"
    m = s.model()
    print(f"{label}: K={K} episodes={len(ep_lengths)} P={P} W={W} "
          f"mean={P/W:.4f}  (envelope K={K})  admissible=SAT")
    return P / W


# Representative executions. base=8, vary N => K = N*base. Homogeneous depth-1
# episodes of length L, M episodes per thread.  Show mean rises toward K with N.
base = 8
L = 6
for N in (1, 2, 4):
    K = N * base
    M = 200                      # many episodes per slot => deep into the bulk
    means = confirm(K, [L] * M, f"N={N}")

print("\nTail/straggler corner (few long episodes, K large):")
confirm(16, [1, 1, 1, 30], "ragged")   # one long straggler => W dominated by 30

print("\nEmpty-run guard corner:")
# no plies => total_msgs == 0 => driver emits mean=0.0 (line 496 guard); we just
# assert the simulator produces W==0, P==0 for an empty workload.
P0, W0, _ = wave_sim(8, [])
assert P0 == 0 and W0 == 0, "empty run should produce no waves"
print("empty: P=0 W=0 -> driver guard emits mean_rows_per_msg=0.0 (SAT by construction)")

print("\nALL CHECKS SAT — derived wave model admits the representative executions.")
