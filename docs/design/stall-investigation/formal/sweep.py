#!/usr/bin/env python3
"""~/w/vdc/chocobo/runs/formal-stall/sweep.py
N-dependence sweep: run the BMC deadlock query over a grid of (threads, slots K,
inflight D, plies, max_rows) at a fixed unroll depth, report SAT (deadlock) vs
UNSAT (deadlock-free within depth) per config. Bounded runs only.

K = ceil(pool_batch/pool_threads) is the C++ slot derivation; here we sweep K
DIRECTLY (the per-thread slot count) which IS that derived quantity, and report
against it so the N-dependence is read off K and D and the row cap.
"""
import itertools, time, sys
from model2 import Config, build

DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 16

grid = []
for T in (1, 2):
    for K in range(1, 6):          # slots/thread (the derived N-dependent count)
        for D in (1, 2, 4, 8):     # inflight message cap
            for P in (1, 2):       # plies (leaves per slot)
                for MR in (1, 2, 4):  # server row cap
                    grid.append(Config(T=T, K=K, D=D, plies=P, max_rows=MR, cap=10))

print(f"sweeping {len(grid)} configs at depth={DEPTH}")
any_dl = False
t0 = time.time()
for cfg in grid:
    s, S = build(cfg, DEPTH)
    r = s.check()
    if str(r) == "sat":
        any_dl = True
        print(f"  DEADLOCK: {cfg}")
print(f"done in {time.time()-t0:.1f}s; deadlock_found={any_dl}")
