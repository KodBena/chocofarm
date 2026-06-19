# Confirmation (not source of trust): the C++ pipelined driver's gather-all semantics force
# inflight_msgs in {0,1}, so the depth-2..D pipeline the models claim at large N is unreachable.
#
# We model ONE worker thread, K slots, a bounded sequence of driver "macro-steps". Each macro-step
# is either an ISSUE (issue_one) or a RECV (recv_batch+completion loop). We encode the ACTUAL code
# semantics:
#   ISSUE: enabled only if some slot is ready (active & running & !submitted). It sets submitted=1
#          for EVERY ready slot at once (gather-all, runner_wire_batched.cpp:437-447) and
#          inflight += 1.
#   RECV : enabled only if inflight>0. It picks the one outstanding message, clears submitted for
#          its slots, re-parks (running stays true) some subset, inflight -= 1.
# Crucially the driver NEVER re-parks a slot except inside a RECV's completion loop, and ISSUE
# gathers ALL ready slots. So two ISSUEs cannot occur back-to-back without a RECV adding readiness.
#
# We ask z3: is there ANY admissible schedule (any park/reply nondeterminism) that reaches
# inflight_msgs == 2? If unsat -> the depth>=2 pipeline is unreachable -> models P1/Q1 confirmed
# too-constrained (they require depth toward D at large N).

from z3 import Solver, Int, Bool, Or, And, Implies, If, Sum, sat

K = 3          # slots (stand-in for K = N*base; result is K-independent by the argument)
STEPS = 6      # bounded macro-steps
D = 3          # the cap the models claim inflight climbs toward

s = Solver()

# State arrays over time: ready[t][k] (slot k is_ready), submitted[t][k], inflight[t].
ready = [[Bool(f"ready_{t}_{k}") for k in range(K)] for t in range(STEPS + 1)]
subm  = [[Bool(f"subm_{t}_{k}")  for k in range(K)] for t in range(STEPS + 1)]
infl  = [Int(f"infl_{t}") for t in range(STEPS + 1)]

# action[t] in {0=ISSUE, 1=RECV}
act = [Int(f"act_{t}") for t in range(STEPS)]

# Initial: post-FILL, every slot parked-and-ready, none submitted, inflight 0.
for k in range(K):
    s.add(ready[0][k] == True, subm[0][k] == False)
s.add(infl[0] == 0)

for t in range(STEPS):
    s.add(Or(act[t] == 0, act[t] == 1))
    is_issue = act[t] == 0
    is_recv  = act[t] == 1

    # eligible(k) = ready & !submitted  (is_ready, runner_wire_batched.cpp:427-430)
    elig = [And(ready[t][k], subm[t][k] == False) for k in range(K)]
    some_elig = Or(*elig)

    # ---- ISSUE enabledness & effect (gather-ALL eligible into one message) ----
    s.add(Implies(is_issue, some_elig))            # issue_one only fires if some slot eligible
    s.add(Implies(is_issue, infl[t] < D))          # gate inflight<D (456/474)
    for k in range(K):
        # submitted becomes 1 for EVERY eligible slot; others unchanged. ready unchanged by ISSUE.
        s.add(Implies(is_issue, subm[t + 1][k] == Or(subm[t][k], elig[k])))
        s.add(Implies(is_issue, ready[t + 1][k] == ready[t][k]))
    s.add(Implies(is_issue, infl[t + 1] == infl[t] + 1))

    # ---- RECV enabledness & effect (drain one msg; re-park an arbitrary subset of its slots) ----
    s.add(Implies(is_recv, infl[t] > 0))
    # The outstanding message's slots are exactly the currently-submitted ones IF inflight==1
    # (the only reachable case). To avoid presupposing that, we allow RECV to clear submitted for a
    # nonempty subset of submitted slots and (nondeterministically) keep them ready (re-park) or
    # not. This is STRICTLY MORE permissive than the code, so an unsat for inflight==2 here implies
    # unsat for the code too.
    cleared = [Bool(f"cleared_{t}_{k}") for k in range(K)]
    for k in range(K):
        s.add(Implies(cleared[k], subm[t][k] == True))         # can only clear a submitted slot
        s.add(Implies(is_recv, subm[t + 1][k] == If(cleared[k], False, subm[t][k])))
        # ready after recv: cleared slots may re-park (free) or go empty; uncleared unchanged.
        # (we leave ready[t+1][k] free for cleared k -> maximal permissiveness)
        s.add(Implies(And(is_recv, cleared[k] == False), ready[t + 1][k] == ready[t][k]))
    s.add(Implies(is_recv, Or(*cleared)))                       # a reply clears >=1 slot
    s.add(Implies(is_recv, infl[t + 1] == infl[t] - 1))

# Goal: reach inflight == 2 at some time (the depth the models require at large N).
s.add(Or(*[infl[t] == 2 for t in range(STEPS + 1)]))

print("checking reachability of inflight_msgs == 2 under code semantics...")
r = s.check()
print("result:", r)
if r == sat:
    print("REACHABLE (would refute the too-constrained finding)")
else:
    print("UNSAT: inflight stays <= 1. Depth-2..D pipeline unreachable -> P1/Q1 confirmed.")
