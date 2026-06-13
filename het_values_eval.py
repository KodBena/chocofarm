#!/usr/bin/env python3
"""
Heterogeneous-value test, UNBIASED policy-rate Monte-Carlo.

The contention: adaptivity underperformed under unit values because (a) unit values mute
the margin and (b) the sparse root value was maximization-biased.  This patches a synthetic
heterogeneous value vector (a few rare-valuable treasures) and measures each policy's ACTUAL
rate by simulation (eval_rate is unbiased: it runs the policy and reports E[R]/E[T]).  If the
contention holds, the adaptive (rollout) policy should now clear a value-aware static route.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chocobo_stage2_approx as M

HIGH = {3, 9, 12, 17}                      # synthetic: a few rare-valuable treasures
M.value = [10.0 if i in HIGH else 1.0 for i in range(M.N)]
print(f"synthetic values: high(=10) {sorted(HIGH)}, rest=1\n", flush=True)


def value_aware_static():
    """Greedy value-per-distance route from CSNE, best prefix by value-weighted rate.
    Stronger than pure nearest-neighbour, still a static (no rerouting on observations)."""
    loc, unv, route, t, best = ("w", M.ENTRY), set(range(M.N)), [], 0.0, (-1.0, None)
    while unv:
        i = max(unv, key=lambda j: M.value[j] / (M.d(loc, ("t", j)) + 1e-9))
        t += M.d(loc, ("t", i)); loc = ("t", i); route.append(i); unv.discard(i)
        ER = (M.K / M.N) * sum(M.value[r] for r in route)
        T = t + M.exit_cost(loc)
        if ER / T > best[0]:
            best = (ER / T, list(route))
    return best


def fixed_point(decision, lam0, iters, runs, seed):
    lam = lam0
    for _ in range(iters):
        lam = M.eval_rate(decision, lam, runs, seed)[0]
    return lam


s_rate, s_route = value_aware_static()
print(f"value-aware static: rate={s_rate:.4f}  ({len(s_route)} treasures: {s_route})", flush=True)

lam_g = fixed_point(M.greedy_decision, 0.0, 4, 800, 1)
g_rate, gR, gT, g_ex, _ = M.eval_rate(M.greedy_decision, lam_g, 2000, 7)
print(f"greedy : lambda*={lam_g:.4f}  rate={g_rate:.4f}  (E[R]={gR:.2f}, E[T]={gT:.2f})", flush=True)

lam_r = lam_g
for _ in range(3):
    lam_r = M.eval_rate(M.rollout_decision, lam_r, 30, 3)[0]
r_rate, rR, rT, r_ex, r_det = M.eval_rate(M.rollout_decision, lam_r, 120, 7)
print(f"rollout: lambda*={lam_r:.4f}  rate={r_rate:.4f}  (E[R]={rR:.2f}, E[T]={rT:.2f})  "
      f"det/run={r_det:.2f}", flush=True)

print(f"\nrollout vs static : {(r_rate-s_rate)/s_rate*100:+.1f}%   (unbiased measured rates)", flush=True)
print(f"greedy  vs static : {(g_rate-s_rate)/s_rate*100:+.1f}%", flush=True)
print(f"tau_4 exits: greedy {g_ex.get('tau_4',0)}/2000, rollout {r_ex.get('tau_4',0)}/120", flush=True)
