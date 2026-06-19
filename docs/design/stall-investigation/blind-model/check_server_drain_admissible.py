#!/usr/bin/env python3
"""
check_server_drain_admissible.py — a MINIMAL bounded Z3 confirmation that one representative
execution of the SERVER-side drain model (model-server-drain.md) is admissible. This is
CONFIRMATION of the theory, never its source: it encodes the server's operational constraints
(block-until-1, drain-all-currently-queued up to max_batch, single-threaded forward serializes,
arrivals accumulate during a forward) and asks Z3 for ONE concrete schedule that exhibits the
qualitatively distinct latitude — two batches whose sizes differ because different numbers of
requests happened to be queued at the two drain instants.

Public Domain (The Unlicense).
"""
from z3 import Int, Real, Solver, And, Or, sat, Sum, If

s = Solver()

# --- parameters fixed for this tiny bounded instance ---
MAX_BATCH = 4          # the server's max_batch cap (pad_to); drain caps total_rows < MAX_BATCH+1
N = 5                  # number of requests (each B_i = 1 row, the degenerate single-leaf case)

# request i arrives at the ROUTER receive-buffer at time a[i] >= 0 (causal: positive arrival times)
a = [Real(f"a_{i}") for i in range(N)]
# the server runs exactly two drains in this instance; drain k STARTS at time t_k and the forward it
# launches FINISHES at f_k. Service time SVC is a positive duration; here the forward runs over the
# padded (MAX_BATCH, in_dim) shape, so SVC is a single positive constant per drain (>0) — we leave it
# as a free positive real per drain to honor the "service time is positive, not pinned" latitude.
t = [Real(f"t_{k}") for k in range(2)]
f = [Real(f"f_{k}") for k in range(2)]
svc = [Real(f"svc_{k}") for k in range(2)]

# membership: in_batch[k][i] == 1 iff request i is drained into batch k
inb = [[Int(f"inb_{k}_{i}") for i in range(N)] for k in range(2)]

for i in range(N):
    s.add(a[i] >= 0)
for k in range(2):
    s.add(svc[k] > 0)                 # CAUSAL: durations are positive (no instantaneous forward)
    s.add(f[k] == t[k] + svc[k])      # CAUSAL: reply cannot precede the forward that produced it
    for i in range(N):
        s.add(Or(inb[k][i] == 0, inb[k][i] == 1))

# drain 0 BLOCKS until >=1 request is queued: t[0] >= the earliest arrival, and >=1 in batch 0.
# (We model "block until >=1" as: t[0] is no earlier than some arrival, and batch 0 is non-empty.)
s.add(t[0] >= a[0])  # without loss of generality a[0] is the earliest (we pin an order below)
# the single-threaded loop: drain 1 cannot START until drain 0's forward FINISHED (serialization).
s.add(t[1] >= f[0])

# arrivals are ordered a[0] <= a[1] <= ... (WLOG label requests by arrival order)
for i in range(N - 1):
    s.add(a[i] <= a[i + 1])

# DRAIN SEMANTICS: a request is in batch k iff it had arrived by the drain instant t[k] AND was not
# already taken by an earlier batch AND the batch's running total stayed < MAX_BATCH when it was taken.
# We encode the essential observable consequences rather than the exact recv loop order:
#   (1) every request is in at most one batch;
#   (2) a request can only be in batch k if it arrived at or before t[k] (can't drain an un-arrived req);
#   (3) batch k's size <= MAX_BATCH (the cap);
#   (4) a request arrived-by-t[0] is taken by batch 0 unless batch 0 is full — i.e. the drain is GREEDY
#       (it takes everything currently queued up to the cap), it does not leave a queued request behind
#       except by the cap. This is the faithful "drain ALL currently-queued up to max_batch" rule.
for i in range(N):
    s.add(inb[0][i] + inb[1][i] <= 1)                       # (1)
    s.add(Or(inb[0][i] == 0, a[i] <= t[0]))                 # (2a)
    s.add(Or(inb[1][i] == 0, a[i] <= t[1]))                 # (2b)

for k in range(2):
    s.add(Sum([inb[k][i] for i in range(N)]) <= MAX_BATCH)  # (3)
    s.add(Sum([inb[k][i] for i in range(N)]) >= 1)          # each modeled drain is non-empty

# (4) GREEDINESS for batch 0: any request queued by t[0] is taken by batch 0 UNLESS batch 0 is at the cap.
size0 = Sum([inb[0][i] for i in range(N)])
for i in range(N):
    s.add(Or(a[i] > t[0], inb[0][i] == 1, size0 == MAX_BATCH))
# a request NOT taken by batch 0 and arrived-by-t[1] is taken by batch 1 (greedy, second drain).
size1 = Sum([inb[1][i] for i in range(N)])
for i in range(N):
    s.add(Or(a[i] > t[1], inb[0][i] == 1, inb[1][i] == 1, size1 == MAX_BATCH))

# THE QUALITATIVE LATITUDE WE WANT TO WITNESS: the two batch sizes DIFFER because different numbers of
# requests were queued at the two drain instants (batch-composition is a function of arrival timing).
s.add(size0 != size1)
# and make it concrete/non-trivial: batch 0 is the cap-sized burst, batch 1 is the leftover.
s.add(size0 == MAX_BATCH)
s.add(size1 == N - MAX_BATCH)

r = s.check()
print("RESULT:", r)
if r == sat:
    m = s.model()
    print("arrivals a   =", [m.eval(a[i]) for i in range(N)])
    print("drain starts t=", [m.eval(t[k]) for k in range(2)])
    print("svc          =", [m.eval(svc[k]) for k in range(2)])
    print("forward fin f=", [m.eval(f[k]) for k in range(2)])
    print("batch0 size  =", m.eval(size0), " members:", [i for i in range(N) if str(m.eval(inb[0][i])) == "1"])
    print("batch1 size  =", m.eval(size1), " members:", [i for i in range(N) if str(m.eval(inb[1][i])) == "1"])
    print("ADMISSIBLE: two distinct batch sizes arise purely from arrival timing vs serialized drain instants.")
