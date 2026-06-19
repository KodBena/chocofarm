"""
/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/out/check_server_greedy_drain.py

Public Domain (The Unlicense).

Bounded admissibility check for ONE representative execution of the production
greedy-drain inference server (chocofarm/az/inference_server.py), as modeled in
model-server-greedy-drain.md.

This is CONFIRMATION, not the source of trust. It encodes the server's
single-threaded serial cycle:

    DRAIN(block until >=1)  ->  pull all currently-queued up to max_batch rows
                            ->  ONE forward (service time S, function of the
                                FIXED padded shape, not real rows)
                            ->  scatter replies
                            ->  next cycle

and the causal-coalescing claim we rely on: requests that ARRIVE on the wire
while a forward is in flight are NOT visible to the in-flight drain; they sit in
the ROUTER's incoming queue and are all pulled by the NEXT drain. So the batch
size of cycle k+1 grows with the number of streams that arrived during cycle k's
service. We check that an execution exists where:
  - cycle 0 drains exactly 1 request (a lone early arrival), and
  - cycle 1 drains a coalesced batch of several requests that all arrived
    during cycle 0's service time,
which is the central "service time shapes batch size" latitude.

Run:
  nice -n 19 timeout 90 /home/bork/w/vdc/venvs/generic/bin/python check_server_greedy_drain.py
"""

from z3 import *

# ---- Parameters (kept symbolic-ish but bounded; values are illustrative, NOT fixed by the model) ----
MAX_BATCH = 256          # server max_batch (inference_server.py:171 cap on drained rows)
N_REQ = 5                # number of producer requests in this bounded horizon

s = Solver()

# Arrival time of each request on the server's ROUTER incoming queue (wire-visible).
# These are SET BY the producer's search progress => nondeterministic, positive, ordered only by causality.
arr = [Real(f"arr_{i}") for i in range(N_REQ)]
# Rows carried by each request (B_i >= 1; producer batches its ready slots, wire_leaf_pool submit_batch).
rows = [Int(f"rows_{i}") for i in range(N_REQ)]

for i in range(N_REQ):
    s.add(arr[i] >= 0)
    s.add(rows[i] >= 1, rows[i] <= 64)   # a single producer message: 1..K rows; bounded here

# Two drain cycles. Each cycle: t_start_drain (first poll wakeup), then forward of duration S, then scatter.
# Cycle 0
c0_wake = Real("c0_wake")     # instant the blocking poll returns (>=1 request queued)
c0_dur  = Real("c0_dur")      # service time S of cycle-0 forward (positive, depends on padded shape)
# Cycle 1
c1_wake = Real("c1_wake")
c1_dur  = Real("c1_dur")

# Service times are POSITIVE and bounded; both forwards pad to max_batch (production E-policy = pad-to-max),
# so the *compiled shape* is identical => S is drawn from the SAME nondeterministic band regardless of real B.
S_LO, S_HI = RealVal(1), RealVal(10)
for d in (c0_dur, c1_dur):
    s.add(d >= S_LO, d <= S_HI)

# Drain set membership: request i is drained in cycle 0 iff it had ARRIVED by the cycle-0 wake instant.
# (the drain is non-blocking recv of everything currently queued; an arrival strictly after the wake is NOT
#  seen by this drain — it is left for the next one. We model "queued at wake" as arr_i <= c0_wake.)
in_c0 = [Bool(f"in_c0_{i}") for i in range(N_REQ)]
in_c1 = [Bool(f"in_c1_{i}") for i in range(N_REQ)]

# The blocking poll returns as soon as >=1 request is queued; the wake cannot precede the earliest arrival,
# and there must be at least one request queued at the wake (the poll only returns on POLLIN).
s.add(Or([arr[i] <= c0_wake for i in range(N_REQ)]))

for i in range(N_REQ):
    # drained in cycle 0  <=>  arrived at or before the cycle-0 wake
    s.add(in_c0[i] == (arr[i] <= c0_wake))
    # cycle 1 drains what arrived after the cycle-0 wake but by the cycle-1 wake (i.e. during/after c0 service)
    s.add(in_c1[i] == And(arr[i] > c0_wake, arr[i] <= c1_wake))

# max_batch cap: total rows drained in a cycle <= MAX_BATCH (inference_server.py:171 `while total_rows < max_batch`)
def rows_in(flags):
    return Sum([If(flags[i], rows[i], 0) for i in range(N_REQ)])
s.add(rows_in(in_c0) <= MAX_BATCH)
s.add(rows_in(in_c1) <= MAX_BATCH)

# Single-threaded serialization: cycle 1 cannot begin its drain until cycle 0 has finished its forward+scatter.
# The forward starts right after the drain (instantaneous gather is fine) and finishes c0_dur later.
c0_done = c0_wake + c0_dur
s.add(c1_wake >= c0_done)            # serialization: no overlap of forwards
# cycle 1 also blocks until >=1 request is queued for it
s.add(Or([And(arr[i] > c0_wake, arr[i] <= c1_wake) for i in range(N_REQ)]))

# ---- The representative latitude we assert is ADMISSIBLE ----
# (A) cycle 0 drains exactly ONE request (a lone early arrival): coalescing did NOT happen for it.
s.add(Sum([If(in_c0[i], 1, 0) for i in range(N_REQ)]) == 1)
# (B) cycle 1 drains a COALESCED batch of >=3 requests, all of which arrived DURING cycle-0 service
#     (arr_i in (c0_wake, c0_done]) -- service time shaped the next batch size.
s.add(Sum([If(in_c1[i], 1, 0) for i in range(N_REQ)]) >= 3)
s.add(And([Implies(in_c1[i], And(arr[i] > c0_wake, arr[i] <= c0_done)) for i in range(N_REQ)]))

print("solving...")
r = s.check()
print("result:", r)
if r == sat:
    m = s.model()
    print("c0_wake =", m[c0_wake], " c0_dur =", m[c0_dur], " c0_done =", m.eval(c0_done))
    print("c1_wake =", m[c1_wake], " c1_dur =", m[c1_dur])
    for i in range(N_REQ):
        print(f"  req {i}: arr={m[arr[i]]} rows={m[rows[i]]} in_c0={m[in_c0[i]]} in_c1={m[in_c1[i]]}")
    b0 = sum(int(m[rows[i]].as_long()) for i in range(N_REQ) if is_true(m[in_c0[i]]))
    b1 = sum(int(m[rows[i]].as_long()) for i in range(N_REQ) if is_true(m[in_c1[i]]))
    print(f"  realized batch rows: cycle0={b0}  cycle1={b1}  (both padded to max_batch={MAX_BATCH})")
    print("ADMISSIBLE: greedy-drain coalescing-under-service-time execution exists.")
else:
    print("UNSAT or UNKNOWN -- model would need revisiting.")
