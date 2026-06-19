# RECONCILER confirmation (not source of trust):
# Re-run the depth<=1 argument at SEVERAL values of K = N*base to make the N-INDEPENDENCE of
# the per-thread inflight bound explicit. The prior model derived inflight in {0,1} at N=1
# (K=base). The central differential question is whether that survives as N grows (K = N*base).
#
# The control-flow argument (runner_wire_batched.cpp): issue_one (434-452) gathers EVERY is_ready
# slot into ONE message and sets submitted[s]=1 for all of them (447); a slot becomes is_ready
# again (429: active & running & !submitted) ONLY inside a recv_batch completion loop
# (462-472: resume_with / advance / fill). So between two consecutive issue_one calls with NO
# intervening recv, no slot becomes newly ready -> the second issue_one finds gathered.empty()
# -> returns false (444). Hence PRIME (456) and every REFILL (474) issue exactly ONE message,
# and inflight_msgs in {0,1}. This is independent of K, hence of N.
#
# We confirm: for K in {2 (base, N=1), 8 (N=4 at base=2 / N=1 at base=8), 24 (large N)},
# inflight==2 is UNSAT under a model STRICTLY LOOSER than the code (RECV may re-park any subset,
# readiness of cleared slots left free). Looser-unsat => code-unsat a fortiori.

from z3 import Solver, Int, Bool, Or, And, Implies, If, sat

def depth2_reachable(K: int, STEPS: int = 8, D: int = 8) -> str:
    s = Solver()
    ready = [[Bool(f"r_{t}_{k}") for k in range(K)] for t in range(STEPS + 1)]
    subm  = [[Bool(f"s_{t}_{k}") for k in range(K)] for t in range(STEPS + 1)]
    infl  = [Int(f"i_{t}") for t in range(STEPS + 1)]
    act   = [Int(f"a_{t}") for t in range(STEPS)]
    for k in range(K):
        s.add(ready[0][k] == True, subm[0][k] == False)
    s.add(infl[0] == 0)
    for t in range(STEPS):
        s.add(Or(act[t] == 0, act[t] == 1))
        is_issue = act[t] == 0
        is_recv  = act[t] == 1
        elig = [And(ready[t][k], subm[t][k] == False) for k in range(K)]
        s.add(Implies(is_issue, Or(*elig)))
        s.add(Implies(is_issue, infl[t] < D))
        for k in range(K):
            s.add(Implies(is_issue, subm[t + 1][k] == Or(subm[t][k], elig[k])))
            s.add(Implies(is_issue, ready[t + 1][k] == ready[t][k]))
        s.add(Implies(is_issue, infl[t + 1] == infl[t] + 1))
        cleared = [Bool(f"c_{t}_{k}") for k in range(K)]
        s.add(Implies(is_recv, infl[t] > 0))
        for k in range(K):
            s.add(Implies(cleared[k], subm[t][k] == True))
            s.add(Implies(is_recv, subm[t + 1][k] == If(cleared[k], False, subm[t][k])))
            s.add(Implies(And(is_recv, cleared[k] == False), ready[t + 1][k] == ready[t][k]))
        s.add(Implies(is_recv, Or(*cleared)))
        s.add(Implies(is_recv, infl[t + 1] == infl[t] - 1))
    s.add(Or(*[infl[t] == 2 for t in range(STEPS + 1)]))
    return str(s.check())

for K in (2, 8, 24):
    r = depth2_reachable(K)
    print(f"K={K:>3} (stand-in for N*base): inflight==2 -> {r}  "
          f"({'depth<=1 holds (N-independent)' if r == 'unsat' else 'DEPTH GROWS -- refutes prior'})")
