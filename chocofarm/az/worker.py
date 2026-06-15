#!/usr/bin/env python3
"""
chocofarm/az/worker.py — the per-worker unit of work for the AZ parallel ExIt loop (audit item K, the
Task third of the Transport ⊥ Pool ⊥ Task split out of `parallel.py`).

This module owns what ONE worker does per episode and nothing about the redis wire protocol (that is
`transport.py`) or the process pool (that is `worker_pool.py`). Concretely it owns: the per-worker
module global `_W` (env/feature-builder/net/search/version/redis — item L will promote this to a
`Worker` object; here it stays a global, routed through the transport collaborator for the wire), the
spawn-pool initializer `_worker_init` (single-thread env pinning, faulthandler+SIGUSR1, core pinning,
env/FeatureBuilder/numba-warmup build), the version-gated net reload `_ensure_net`, the
worker-count-invariant seed fold `_task_rng`, and the two work-kind entrypoints `_gen_task` /
`_eval_task`.

The two work-kinds (generate, evaluate) are captured as DATA in `TaskSpec` rather than two unrelated
code paths: the kind tag for the rng fold and the module-level worker callable. `_gen_task` /
`_eval_task` share the `_ensure_net` + rng-fold plumbing through one `_prepare(...)` helper and the
`TASK_SPECS` table; they stay as two small module-level entrypoints (NOT a metaprogrammed dispatch)
because the spawn pool resolves the worker callable by qualified name — a bound method or a closure
would not survive the spawn re-import. That picklability constraint is exactly why the split keeps two
named entrypoints over a single clever generic one.

These functions are module-level so the `spawn` pool can pickle them by qualified name and re-import
them in the child; `_W` lives in the same module they read from.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
from collections import namedtuple

import numpy as np

from chocofarm.az import transport


# ---- per-worker module-global state (one set per process, built lazily) ----
_W = {
    "env": None, "fb": None, "net": None, "search": None,
    "version": -1, "m": None, "n_sims": None, "base_seed": None, "redis": None,
}


def _worker_init(core_list, base_seed, m, n_sims):
    """Process-pool initializer: pin THIS worker to its core, build env/feature-builder, open the
    redis connection, warm the numba kernel. The net is loaded lazily on the first task (when the
    first weight version arrives over redis)."""
    # Pin native thread counts to 1 DETERMINISTICALLY, before numba/numpy/BLAS import inside this
    # child (deadlock fix H1a / Fix C). The worker is core-pinned and wants exactly one native
    # thread per math runtime; relying on the parent's JAX import to have `setdefault`'d these
    # into the inherited environment makes the worker's threading config depend on parent import
    # order and on JAX being present. Setting them here makes the child's native-threading
    # configuration independent of the parent — the same single-thread pin the (clean) numpy
    # runs had, severing the JAX→spawn-child residue the RCA fingers as the most likely
    # worker-wedge trigger. setdefault so an operator override still wins.
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(_var, "1")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

    # WORKER-SIDE FAULTHANDLER (the discriminating instrument). The deadlock is intermittent and
    # the prime-suspect cause (a worker wedged in a native-runtime lock vs a redis socket recv)
    # cannot be PROVEN without a dump of the WORKER process — ptrace_scope blocked py-spy, and the
    # parent's faulthandler only dumps the parent. Registering faulthandler here, with a SIGUSR1
    # handler, makes the next recurrence debuggable on the worker side: a watcher sends SIGUSR1 to
    # the wedged worker PID and gets an all-thread Python traceback on stderr (→ the run log),
    # which discriminates H1a (numba/native threading-init lock) from H2 (timeout-less socket
    # recv). This is the cheap, low-risk confirming step the bounding fixes (A/B) cannot supply on
    # their own — without it, Fix C ships un-falsifiable. faulthandler writes a C-level traceback
    # from a signal handler, so it is safe to call even when the GIL is contended (the exact hang
    # state). Fail soft if signals aren't available (e.g. a non-POSIX host).
    try:
        import faulthandler, signal
        faulthandler.enable()
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except (ImportError, OSError, ValueError):
        pass   # diagnostic only — never block worker startup on it

    import multiprocessing as mp
    name = mp.current_process().name
    try:
        widx = int(name.rsplit("-", 1)[-1]) - 1     # PoolWorker-1 -> 0
    except (ValueError, IndexError):
        widx = 0
    core = core_list[widx % len(core_list)]
    try:
        os.sched_setaffinity(0, {core})
    except (AttributeError, OSError):
        pass   # affinity not available — fail soft (a perf knob, not correctness)

    from chocofarm.model.env import Environment
    from chocofarm.az.features import FeatureBuilder
    from chocofarm.az.kernels import warmup as _kwarm

    env = Environment()
    _W.update(env=env, fb=FeatureBuilder(env), m=m, n_sims=n_sims,
              base_seed=base_seed, core=core, widx=widx, redis=transport.connect())
    _kwarm(env.N, len(env.detectors))   # compile the belief kernel once per process


def _ensure_net(run, version, hot_search=None):
    """Load the net into the worker iff the weight version changed — reading the raw-bytes payload
    from redis `az:w:<run>:<version>` (no pickle) through the transport collaborator. Rebuilds the
    GumbelAZSearch on the fresh net.

    `hot_search` (hp-registry §3.4): the live HOT search knobs (c_puct/c_visit/c_scale/c_outcome/
    max_depth) for THIS iteration, sized by the parent's per-iteration snapshot. The version bumps
    every iteration (the parent re-publishes weights each iter), so the worker rebuilds its search
    on the version change and the live HOT knobs are picked up at that natural refresh point — the
    seam §3.2 names ('the worker's natural refresh coincides with the parent's poll'). `m`/`n_sims`
    are RESTART (set once in `_worker_init`)."""
    if _W["version"] == version and _W["net"] is not None:
        return
    from chocofarm.az.gumbel_search import GumbelAZSearch
    manifest, blob = transport.read_weights(_W["redis"], run, version)
    net = transport.unpack_net(manifest, blob)
    _W["net"] = net
    hs = dict(hot_search) if hot_search else {}
    _W["search"] = GumbelAZSearch(net, _W["env"], m=_W["m"], n_sims=_W["n_sims"], **hs)
    _W["version"] = version


def _task_rng(version, kind, idx):
    """Independent, reproducible per-(iteration, kind, episode) RNG (folds base seed + version +
    kind tag + idx). Same logical episode → same stream under any worker count (parallel≈serial).

    The kind tag comes from `TASK_SPECS[kind]` so the gen/eval fold is ONE table, not two literals —
    but the fold ARITHMETIC is byte-for-byte the pre-split fold (the parallel≈serial determinism
    contract: the np.uint64 multipliers and the `version+1` term are unchanged)."""
    kind_tag = TASK_SPECS[kind].kind_tag
    seed = (np.uint64(_W["base_seed"])
            ^ (np.uint64(version + 1) * np.uint64(2_654_435_761))
            ^ (np.uint64(kind_tag) * np.uint64(40_503))
            ^ (np.uint64(idx) * np.uint64(2_246_822_519)))
    return np.random.default_rng(int(seed))


def _prepare(run, version, kind, idx, hot_search):
    """Shared task plumbing both work-kinds run: version-gated net reload (through the transport) +
    the worker-count-invariant rng fold. Returns the per-episode rng. Keeping the gen/eval entrypoints
    sharing this (rather than each re-reaching into `_W`) is the point of the Task split."""
    _ensure_net(run, version, hot_search=hot_search)
    return _task_rng(version, kind, idx)


def _gen_task(args):
    """Worker task: generate ONE training episode, write its records to redis as raw bytes, return
    only (idx, n_rows, feat_dim, n_slots). `args` = (run, version, world, lam, explore_plies,
    lam_blend, n_step, idx, res_token, hot_search, max_steps). `hot_search`/`max_steps` are the live
    HOT knobs for this iteration (hp-registry §3.4)."""
    (run, version, world, lam, explore_plies, lam_blend, n_step, idx, res_token,
     hot_search, max_steps) = args
    rng = _prepare(run, version, "gen", idx, hot_search)
    from chocofarm.az.exit_loop import generate_episode
    recs = generate_episode(_W["env"], _W["search"], _W["fb"], world, lam, rng,
                            explore_plies, max_steps=max_steps,
                            lam_blend=lam_blend, n_step=n_step)
    n = len(recs)
    if n == 0:
        return (idx, 0, 0, 0)
    feat_dim = recs[0][0].shape[0]
    n_slots = recs[0][1].shape[0]
    # stack into contiguous float32 blocks, write raw bytes (no pickle)
    X = np.empty((n, feat_dim), dtype=np.float32)
    PI = np.empty((n, n_slots), dtype=np.float32)
    M = np.empty((n, n_slots), dtype=np.float32)
    Y = np.empty(n, dtype=np.float32)
    for i, (f, pi, mask, g) in enumerate(recs):
        X[i] = f; PI[i] = pi; M[i] = mask; Y[i] = g
    transport.write_results(_W["redis"], res_token, idx, X, PI, M, Y)
    return (idx, n, feat_dim, n_slots)


def _eval_task(args):
    """Worker task: run ONE greedy-eval episode against a held-out world; return (R, T) (scalars —
    no array payload, so they ride the pipe cheaply). `args` = (run, version, world, lam, idx,
    hot_search, max_steps)."""
    run, version, world, lam, idx, hot_search, max_steps = args
    rng = _prepare(run, version, "eval", idx, hot_search)
    from chocofarm.az.gumbel_search import GumbelPolicy
    hs = dict(hot_search) if hot_search else {}
    pol = GumbelPolicy(_W["net"], _W["env"], m=_W["m"], n_sims=_W["n_sims"], **hs)
    R, T, _ = _W["env"].simulate(pol, world, lam, rng, max_steps=max_steps)
    return float(R), float(T)


# ---- the two work-kinds as DATA (not two parallel ad-hoc code paths) ----
# A TaskSpec captures what differs between a generation episode and an eval episode: the kind tag the
# rng fold uses, and the module-level worker callable the pool fans out. The shared plumbing (net
# reload + rng fold) is `_prepare`; what is per-kind is exactly these two fields. Kept module-level
# and as a small declarative record — NOT a metaprogrammed dispatch — because the spawn pool must
# resolve `callable` by qualified name (a closure/bound method would not survive the re-import).
TaskSpec = namedtuple("TaskSpec", ["kind", "kind_tag", "callable"])

TASK_SPECS = {
    "gen": TaskSpec("gen", 1_000_003, _gen_task),
    "eval": TaskSpec("eval", 7_000_037, _eval_task),
}
