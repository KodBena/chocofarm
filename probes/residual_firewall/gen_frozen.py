#!/usr/bin/env python3
"""Firewall probe — generate ONE frozen supervised dataset for the residual A/B.

Self-play generation from a converged baseline net (the "expert"); NO training. Freezes
(X, PI, M, Y) to npz so residual-ON vs residual-OFF train on the SAME data — removing the
non-stationary-target confound of the ExIt loop. Run pinned + bounded:

    CHOCO_AZ_DTYPE=float64 timeout 900 taskset -c 0,1,2,3 \
        python probes/residual_firewall/gen_frozen.py <expert.npz> <out.npz> <n_episodes>
"""
import sys, time
import numpy as np
from chocofarm.az.mlp import ValueMLP
from chocofarm.model.env import Environment
from chocofarm.az.features import FeatureBuilder
from chocofarm.az.gumbel_search import GumbelAZSearch
from chocofarm.az.exit_loop import generate_episode

expert, out, E = sys.argv[1], sys.argv[2], int(sys.argv[3])
env = Environment(); fb = FeatureBuilder(env)
net = ValueMLP.load(expert)
search = GumbelAZSearch(net, env, m=12, n_sims=128)
gen_rng = np.random.default_rng(999)
worlds = [int(gen_rng.choice(env.worlds)) for _ in range(E)]
recs = []
t0 = time.time()
for i, w in enumerate(worlds):
    recs.extend(generate_episode(env, search, fb, w, 0.0855, gen_rng, n_explore_plies=4,
                                 lam_blend=0.6, n_step=None))
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{E} eps, {len(recs)} dec, {time.time()-t0:.0f}s", flush=True)
X = np.asarray([r[0] for r in recs], dtype=np.float64)
PI = np.asarray([r[1] for r in recs], dtype=np.float64)
M = np.asarray([r[2] for r in recs], dtype=np.float64)
Y = np.asarray([r[3] for r in recs], dtype=np.float64)
print("dataset:", X.shape, "Y mean/std", float(Y.mean()), float(Y.std()), flush=True)
np.savez(out, X=X, PI=PI, M=M, Y=Y)
print(f"saved {out} in {time.time()-t0:.0f}s", flush=True)
