"""
Minimal bounded Z3 confirmation that the server-side self-batching execution
(Exec C in model-server-transport.md) is ADMISSIBLE under the derived timing/causal
constraints. This is confirmation only, never the source of trust.

We encode one server thread and T=2 peers over a few message rounds. The server
alternates POLL/DRAIN/FORWARD/SCATTER (single-thread mutual exclusion). We assert the
causal constraints from section 3.3 and ask Z3 to find a schedule in which a forward
running over [f0_start, f0_end] causes peer messages emitted during that window to be
batched together in the NEXT drain (B grows from 1 to >=2) -- the self-batching effect.

We do NOT model the NN math or socket internals; durations are positive reals.
"""
from z3 import Real, Int, Solver, And, Or, sat, Sum, If

s = Solver()

# --- server timeline: 2 serve iterations, each drain->forward->scatter ---
# iteration 0
p0_e0 = Real('p0_e0')      # peer0 emits msg0 at this time
p1_e0 = Real('p1_e0')      # peer1 emits msg0
dwire = Real('dwire')      # positive wire delay (single symbol, bounded below)
drain0_t = Real('drain0_t')   # time the server's drain0 recv-snapshot happens
f0_start = Real('f0_start')
S0 = Real('S0')            # sink service time of forward0 (positive)
f0_end = Real('f0_end')
scatter0_t = Real('scatter0_t')

# peers emit their NEXT msg only after receiving reply0 (D-cap with D such that
# they were saturated) -- causal constraint 3.3-3
p0_e1 = Real('p0_e1')
p1_e1 = Real('p1_e1')
reply0_seen_p0 = Real('reply0_seen_p0')
reply0_seen_p1 = Real('reply0_seen_p1')

drain1_t = Real('drain1_t')   # server's drain1 snapshot
B1 = Int('B1')                # number of messages batched in drain1

POS = lambda x: x > 0

s.add(POS(dwire), POS(S0))
s.add(p0_e0 >= 0, p1_e0 >= 0)

# constraint 3.3-4: a frame emitted at e is visible to a recv at r iff e+dwire <= r.
# drain0 sees ONLY peer0's msg0 (peer1's msg0 arrives just after the snapshot).
s.add(p0_e0 + dwire <= drain0_t)           # peer0 msg0 visible at drain0
s.add(p1_e0 + dwire > drain0_t)            # peer1 msg0 NOT yet visible at drain0
# so drain0 batches B=1 (only peer0).

# 3.3-5 single-thread: forward starts after drain snapshot; 3.3-2 reply after forward.
s.add(f0_start >= drain0_t)
s.add(f0_end == f0_start + S0)
s.add(scatter0_t >= f0_end)                # reply cannot precede its forward

# during the forward window, peer1's msg0 becomes visible AND peers emit msg1.
# (the server is NOT recv-ing while forwarding => these accumulate)
s.add(p1_e0 + dwire <= drain1_t)           # peer1 msg0 now visible at next drain

# reply0 reaches the peers no earlier than scatter, plus wire delay
s.add(reply0_seen_p0 >= scatter0_t + dwire)
s.add(reply0_seen_p1 >= scatter0_t + dwire)
# D-cap: peers' msg1 emitted only after their reply0 seen
s.add(p0_e1 >= reply0_seen_p0)
s.add(p1_e1 >= reply0_seen_p1)

# drain1 happens after scatter0 (next serve iteration, single thread)
s.add(drain1_t >= scatter0_t)

# what is visible at drain1: count peer1.msg0 (definitely) + peer0.msg1/peer1.msg1 if
# they became visible in time. Self-batching: B1 >= 2 means the next drain is bigger
# than drain0's B=1.
visible1 = Sum([
    If(p1_e0 + dwire <= drain1_t, 1, 0),   # peer1 msg0 (accumulated during forward)
    If(p0_e1 + dwire <= drain1_t, 1, 0),   # peer0 msg1
    If(p1_e1 + dwire <= drain1_t, 1, 0),   # peer1 msg1
])
s.add(B1 == visible1)
s.add(B1 >= 2)   # the self-batching claim: next drain coalesces >= 2 messages

r = s.check()
print("self-batching execution admissible:", r)
if r == sat:
    m = s.model()
    for v in [p0_e0, p1_e0, dwire, drain0_t, f0_start, S0, f0_end, scatter0_t,
              reply0_seen_p0, p0_e1, p1_e1, drain1_t]:
        print(f"  {v} = {m[v]}")
    print(f"  B1 (coalesced msgs in drain1) = {m[B1]}")
