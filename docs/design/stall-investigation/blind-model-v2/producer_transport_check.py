"""Minimal bounded Z3 confirmation of ONE representative producer-transport execution.

CONFIRMATION ONLY — not the source of trust. Encodes a few send/recv steps of one thread's
pipelined pump and asks Z3 whether the steady-state interleaving (representative execution #2,
"staggered waves fill D") is admissible under the modeled constraints:

  * inflight_msgs in [0, D]                              (cap D; runner_wire_batched.cpp:456/457/474)
  * |inflight set| == inflight_msgs (one map entry/msg)  (wire_leaf_pool.hpp:92,115,120; :448,460)
  * reply-after-request causality (a RECV consumes a corr SENT at a strictly earlier step)
  * monotone-unique corr ids (corr_seq fetch_add; wire_leaf_pool.hpp:84)
  * first action is a SEND (nothing to recv yet)

SAT => the representative trace is admissible under the model.
"""
from z3 import Int, Bool, Solver, And, Or, Implies, If, Sum, sat

D = 3        # in-flight MESSAGE cap (small, bounded)
STEPS = 8    # number of transport actions in the trace

s = Solver()

act  = [Int(f"act_{t}")  for t in range(STEPS)]   # 0=SEND, 1=RECV
corr = [Int(f"corr_{t}") for t in range(STEPS)]   # corr id touched at step t
infl = [Int(f"infl_{t}") for t in range(STEPS)]   # inflight_msgs AFTER step t
nextc = [Int(f"nextc_{t}") for t in range(STEPS)] # next monotone corr id available before step t

for t in range(STEPS):
    s.add(Or(act[t] == 0, act[t] == 1))
    s.add(corr[t] >= 0)

sent = [[Bool(f"sent_{t}_{c}") for c in range(STEPS)] for t in range(STEPS)]
done = [[Bool(f"done_{t}_{c}") for c in range(STEPS)] for t in range(STEPS)]

# step 0
s.add(nextc[0] == 0)
s.add(act[0] == 0)  # first action must be a SEND (recv with empty inflight impossible)
for c in range(STEPS):
    s.add(sent[0][c] == And(act[0] == 0, corr[0] == c))
    s.add(done[0][c] == False)  # cannot complete anything at the very first step

for t in range(1, STEPS):
    s.add(nextc[t] == nextc[t-1] + If(act[t-1] == 0, 1, 0))
    s.add(Implies(act[t] == 0, corr[t] == nextc[t]))   # SEND uses the current fresh id
    for c in range(STEPS):
        s.add(sent[t][c] == Or(sent[t-1][c], And(act[t] == 0, corr[t] == c)))
        # reply-after-request: can only RECV a corr already SENT at a strictly earlier step, not yet done
        s.add(done[t][c] == Or(done[t-1][c],
                               And(act[t] == 1, corr[t] == c, sent[t-1][c], done[t-1][c] == False)))
    # a RECV must actually consume some outstanding corr
    s.add(Implies(act[t] == 1, Or([And(corr[t] == c, sent[t-1][c], done[t-1][c] == False)
                                   for c in range(STEPS)])))

# first SEND uses id 0
s.add(corr[0] == nextc[0])

for t in range(STEPS):
    outstanding = Sum([If(And(sent[t][c], done[t][c] == False), 1, 0) for c in range(STEPS)])
    s.add(infl[t] == outstanding)
    s.add(infl[t] >= 0)
    s.add(infl[t] <= D)          # cap D

# Representative-trace shape: reach full pipeline depth D, end fully drained, with genuine pipelining
s.add(Or([infl[t] == D for t in range(STEPS)]))
s.add(infl[STEPS-1] == 0)
s.add(Or([And(act[t] == 1, Or([act[u] == 0 for u in range(t+1, STEPS)])) for t in range(STEPS)]))

res = s.check()
print("RESULT:", res)
if res == sat:
    m = s.model()
    print(f"Admissible representative trace (D={D}):")
    for t in range(STEPS):
        a = "SEND" if m.evaluate(act[t]).as_long() == 0 else "RECV"
        print(f"  step {t}: {a:4s} corr={m.evaluate(corr[t]).as_long()} inflight={m.evaluate(infl[t]).as_long()}")
    print("CONFIRMS: a pipelined send/recv interleaving up to depth D, fully drained, respecting")
    print("  cap-D, |inflight|==inflight_msgs, monotone-unique corr, and reply-after-request, is admissible.")
else:
    print("UNSAT/UNKNOWN — re-examine the derivation.")
