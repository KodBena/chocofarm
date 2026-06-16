#!/usr/bin/env python3
"""
chocofarm/az/worker.py — the per-worker unit of work for the AZ parallel ExIt loop (audit item K's
Task third of the Transport ⊥ Pool ⊥ Task split out of `parallel.py`; promoted by item L / R14 from a
module-global dict to a `Worker` object).

NUMPY-ONLY CONTRACT (R14 — the deadlock ROOT CAUSE).  The spawn worker CHILD is a clean numerical
host: numpy + numba ONLY, never `jax` / `jaxlib` / `optax`.  The jaxtrain-deadlock RCA
(`docs/notes/jaxtrain-deadlock-rca.md`) fingered "JAX was imported in the parent before the spawn
Pool existed" as the discriminating change that made the loop wedge under JAX but not under numpy —
the child inheriting a second native-threading runtime's environment/allocator residue (H1a).  R14
removes that ROOT CAUSE rather than only bounding its symptom: the worker's whole import graph is
numpy/numba (env, FeatureBuilder, the belief kernel, `GumbelAZSearch`/`GumbelPolicy`, the numpy
`ValueMLP` via `transport.unpack_net`, `generate_episode` — all reach `forward.forward_core` over
numpy, never the jax forward).  The two jax entry points the worker is one import-edge away from —
`exit_loop`'s `JaxTrainer` and `gumbel_search`'s `MlpJaxForward` — are BOTH lazy (function-local
imports), so importing `generate_episode` / the search pulls no jax.  MEASURED 2026-06-15: a real
1-worker `generate` reported `jax`/`jaxlib`/`optax`/`mlp_jax*`/`optimizer` ALL absent from the
child's `sys.modules`.  `_worker_init` LOCKS this with a fail-loud guard (`Worker._assert_jax_free`,
ADR-0002) run in the SPAWN CHILD after the initializer's imports: a jax leak into the child is a
worker-startup RuntimeError, not a latent re-opened deadlock.  (The guard is in `_worker_init`, which
runs ONLY in the spawn child — not in `Worker.__init__`, which a jax-importing test harness would
otherwise trip when constructing a Worker in-process.)  The worker's net uses the numpy forward
(`unpack_net` builds `ValueMLP` without `use_jax_mlp`; the float32-numpy `predict_both` is the leaf
eval) — confirmed and kept that way.

This module owns what ONE worker does per episode and nothing about the redis wire protocol (that is
`transport.py`) or the process pool (that is `worker_pool.py`).  Concretely it owns: the `Worker`
object (env / feature-builder / net / search / current (phase, version) / redis connection /
base_seed — the search budget m/n_sims now rides the per-iteration hot_search (HOT), not the ctor)
held as the per-process module singleton `_WORKER`; the spawn-pool
initializer `_worker_init` (single-thread env pinning, the numpy-only guard, faulthandler+SIGUSR1,
core pinning, env/FeatureBuilder/numba-warmup build, `_WORKER` construction); the `(phase, version)`
reload-gate `Worker.ensure_net`; the worker-count-invariant seed fold `Worker.task_rng` (built on the
pure `Worker._fold_seed` arithmetic); and the two work-kind entrypoints `_gen_task` / `_eval_task`.

The two work-kinds (generate, evaluate) are captured as DATA in `TaskSpec` rather than two unrelated
code paths: the kind tag for the rng fold and the module-level worker callable.  `_gen_task` /
`_eval_task` are thin module-level entrypoints that delegate to `_WORKER.generate_episode(...)` /
`_WORKER.eval_episode(...)`; they share the `ensure_net` + rng-fold plumbing through the Worker's
`prepare(...)` and the `TASK_SPECS` table.  They stay as two named module-level functions (NOT a
metaprogrammed dispatch, NOT bound methods) because the spawn pool resolves the worker callable by
qualified name — a bound method or a closure would not survive the spawn re-import.  That
picklability constraint is exactly why the split keeps two named entrypoints over a single clever
generic one, AND why `_WORKER` is a module singleton (each child builds its own in `_worker_init`)
rather than a pickled-across argument.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import numpy as np

from chocofarm.az import transport

if TYPE_CHECKING:
    from chocofarm.model.env import Environment
    from chocofarm.az.features import FeatureBuilder
    from chocofarm.az.gumbel_search import GumbelAZSearch
    from chocofarm.az.mlp import ValueMLP


class Worker:
    """The per-process unit of work (item L / R14 — promoted from the `_W` module-global dict).  ONE
    instance per worker process, built in `_worker_init` and held as the module singleton `_WORKER`.
    It owns the env / feature-builder / net / search, the current `(phase, version)` (the reload
    gate), the redis connection, and the RESTART-fixed `base_seed`. The search budget `m`/`n_sims`
    is now HOT (it rides the per-iteration `hot_search`, applied on each `(phase, version)` rebuild),
    so it is no longer ctor state — exactly as `c_puct`/… already were.

    The two module-level task entrypoints (`_gen_task` / `_eval_task`, resolved by the spawn pool by
    qualified name) delegate to `generate_episode` / `eval_episode`; the shared net-reload + rng-fold
    plumbing is `prepare`.  The seed fold lives here too (`task_rng` over the pure `_fold_seed`
    arithmetic) — byte-for-byte the pre-split fold; PHASE namespaces the weight KEY only and never
    enters the rng (so parallel≈serial bit-identity is preserved by construction)."""

    def __init__(self, env: "Environment", fb: "FeatureBuilder", redis: Any, base_seed: int,
                 core: int | None = None, widx: int = 0) -> None:
        self.env = env
        self.fb = fb
        self.redis = redis
        self.base_seed = base_seed
        self.core = core
        self.widx = widx
        self.net: "ValueMLP | None" = None
        self.search: "GumbelAZSearch | None" = None
        # The reload gate keys on (phase, version) — NOT version alone (R14).  Within one iteration
        # `it`, gen and eval publish to DISTINCT weight keys (phase ∈ {"gen","eval"}); the eval
        # phase must reload the POST-TRAIN weights even though `version == it` is unchanged from
        # gen.  Sentinel (None, -1) forces the first reload.
        self.phase: str | None = None
        self.version = -1

    # ---- numpy-only contract enforcement (the deadlock ROOT-CAUSE lock) ----
    @staticmethod
    def _assert_jax_free() -> None:
        """Fail loud (ADR-0002) if `jax`/`jaxlib` is in `sys.modules`.  Called from `_worker_init`
        AFTER the worker's imports — and ONLY there, because `_worker_init` runs solely in the SPAWN
        CHILD (a clean fresh interpreter), where this property must hold.  It is deliberately NOT run
        from `Worker.__init__`: an in-process `Worker(...)` constructed by a test harness that has
        imported jax for an unrelated test would trip a guard that is about the CHILD's import graph,
        not the harness's.  The spawn child is the surface the numpy-only contract protects; a leak
        there — a new top-level `import jax` in a worker-reachable module, a once-lazy jax import made
        eager, or a fork copying the parent's XLA state — re-opens the JAX→spawn-child wedge mode the
        RCA traced.  `optax` rides on `jax`, so guarding `jax`/`jaxlib` covers it."""
        import sys as _sys
        leaked = [m_ for m_ in ("jax", "jaxlib") if m_ in _sys.modules]
        if leaked:
            raise RuntimeError(
                f"R14: the AZ worker child must be JAX-free (the numpy-only contract that removes "
                f"the jaxtrain-deadlock ROOT CAUSE), but {leaked} leaked into the spawn child's "
                f"sys.modules. A jax import reached a worker-imported module — find the leaking "
                f"import edge (a new top-level `import jax`, or a once-lazy jax import made eager) "
                f"and sever it (keep it function-local), or the spawn child re-inherits the "
                f"XLA-threading residue the RCA fingered (docs/notes/jaxtrain-deadlock-rca.md)."
            )

    # ---- the seed fold (worker-count-invariant; byte-for-byte the pre-split arithmetic) ----
    @staticmethod
    def _fold_seed(base_seed: int, version: int, kind: str, idx: int) -> np.uint64:
        """The np.uint64 seed fold — the parallel≈serial determinism contract.  PURE in
        `base_seed` (no `self`/global reach) so the arithmetic is testable in isolation, but the
        BODY is byte-for-byte the pre-split fold: the kind tag from `TASK_SPECS[kind]`, the
        multipliers `2_654_435_761` / `40_503` / `2_246_822_519`, and the `version+1` term are
        unchanged.  PHASE DOES NOT ENTER THE FOLD — it namespaces the weight key only; the same
        logical (iteration, kind, episode) draws the same stream under any worker count and either
        phase, so parallel≈serial bit-identity holds by construction."""
        kind_tag = TASK_SPECS[kind].kind_tag
        return (np.uint64(base_seed)
                ^ (np.uint64(version + 1) * np.uint64(2_654_435_761))
                ^ (np.uint64(kind_tag) * np.uint64(40_503))
                ^ (np.uint64(idx) * np.uint64(2_246_822_519)))

    def task_rng(self, version: int, kind: str, idx: int) -> np.random.Generator:
        """Independent, reproducible per-(iteration, kind, episode) RNG for THIS worker (folds the
        worker's `base_seed` + version + kind tag + idx via `_fold_seed`).  Same logical episode →
        same stream under any worker count (parallel≈serial)."""
        return np.random.default_rng(int(self._fold_seed(self.base_seed, version, kind, idx)))

    # ---- the (phase, version)-gated net reload ----
    def ensure_net(self, run: str, phase: str, version: int,
                   hot_search: dict[str, Any] | None = None) -> None:
        """Load the net into the worker iff the weight `(phase, version)` changed — reading the
        raw-bytes payload from redis `az:w:<run>:<phase>:<version>` (no pickle) through the transport
        collaborator.  Rebuilds the GumbelAZSearch on the fresh net.

        The gate keys on `(phase, version)`, not version alone (R14): within one iteration `it` the
        gen phase publishes at `("gen", it)` and the eval phase republishes the POST-TRAIN weights at
        `("eval", it)` — same `version`, distinct phase — so the worker MUST reload at the
        gen→eval transition (a version-only gate would serve the stale pre-train weights at eval).
        The it→it+1 transition still reloads because `version` bumps.

        `hot_search` (hp-registry §3.4): the live HOT search knobs (m/n_sims/c_puct/c_visit/c_scale/
        c_outcome/max_depth) for THIS iteration, sized by the parent's per-iteration snapshot.  The
        `(phase, version)` pair changes every phase/iteration (the parent re-publishes weights each
        phase), so the worker rebuilds its search on the change and the live HOT knobs are picked up
        at that natural refresh point — the seam §3.2 names ('the worker's natural refresh coincides
        with the parent's poll').  `m`/`n_sims` ride `hot_search` too now (HOT — the SH bracket is
        recomputed per decide), so they are applied at the same rebuild as the `c_*` knobs."""
        if self.phase == phase and self.version == version and self.net is not None:
            return
        from chocofarm.az.gumbel_search import GumbelAZSearch
        manifest, blob = transport.read_weights(self.redis, run, phase, version)
        net = transport.unpack_net(manifest, blob)
        self.net = net
        hs = dict(hot_search) if hot_search else {}
        self.search = GumbelAZSearch(net, self.env, **hs)
        self.phase = phase
        self.version = version

    def prepare(self, run: str, phase: str, version: int, kind: str, idx: int,
                hot_search: dict[str, Any] | None) -> np.random.Generator:
        """Shared task plumbing both work-kinds run: the `(phase, version)`-gated net reload (through
        the transport) + the worker-count-invariant rng fold.  Returns the per-episode rng.  Keeping
        the gen/eval entrypoints sharing this (rather than each re-reaching into worker state) is the
        point of the Task split."""
        self.ensure_net(run, phase, version, hot_search=hot_search)
        return self.task_rng(version, kind, idx)

    # ---- the two work-kinds ----
    def generate_episode(self, args: tuple[Any, ...]) -> tuple[int, int, int, int]:
        """Generate ONE training episode, write its records to redis as raw bytes, return only
        (idx, n_rows, feat_dim, n_slots).  `args` = (run, version, world, lam, explore_plies,
        lam_blend, n_step, idx, res_token, hot_search, max_steps).  Phase is "gen" (this entrypoint's
        kind)."""
        (run, version, world, lam, explore_plies, lam_blend, n_step, idx, res_token,
         hot_search, max_steps) = args
        rng = self.prepare(run, "gen", version, "gen", idx, hot_search)
        # `prepare` ran `ensure_net`, so the search is built (ADR-0002 fail-loud, not a None pass).
        assert self.search is not None
        from chocofarm.az.exit_loop import generate_episode
        recs = generate_episode(self.env, self.search, self.fb, world, lam, rng,
                                explore_plies, max_steps=max_steps,
                                lam_blend=lam_blend, n_step=n_step)
        n = len(recs)
        if n == 0:
            return (idx, 0, 0, 0)
        feat_dim = recs[0][0].shape[0]
        n_slots = recs[0][1].shape[0]
        # stack into contiguous float32 blocks, write raw bytes (no pickle). The block dtype is the
        # result_spec SSOT (ADR-0012 P1) — the ONE home of "the result blocks are float32", shared
        # with transport.read_and_delete_results and the C++ write_results; write_results re-validates
        # it at the boundary (ADR-0002 fail-loud).
        from chocofarm.az.result_spec import RESULT_DTYPE
        X = np.empty((n, feat_dim), dtype=RESULT_DTYPE)
        PI = np.empty((n, n_slots), dtype=RESULT_DTYPE)
        M = np.empty((n, n_slots), dtype=RESULT_DTYPE)
        Y = np.empty(n, dtype=RESULT_DTYPE)
        for i, (f, pi, mask, g) in enumerate(recs):
            X[i] = f; PI[i] = pi; M[i] = mask; Y[i] = g
        transport.write_results(self.redis, res_token, idx, X, PI, M, Y)
        return (idx, n, feat_dim, n_slots)

    def eval_episode(self, args: tuple[Any, ...]) -> tuple[float, float]:
        """Run ONE greedy-eval episode against a held-out world; return (R, T) (scalars — no array
        payload, so they ride the pipe cheaply).  `args` = (run, version, world, lam, idx,
        hot_search, max_steps).  Phase is "eval" (this entrypoint's kind)."""
        run, version, world, lam, idx, hot_search, max_steps = args
        rng = self.prepare(run, "eval", version, "eval", idx, hot_search)
        # `prepare` ran `ensure_net`, so the net is loaded (ADR-0002 fail-loud, not a None pass).
        assert self.net is not None
        from chocofarm.az.gumbel_search import GumbelPolicy
        hs = dict(hot_search) if hot_search else {}
        pol = GumbelPolicy(self.net, self.env, **hs)
        R, T, _ = self.env.simulate(pol, world, lam, rng, max_steps=max_steps)
        return float(R), float(T)


# ---- per-worker module singleton (one Worker per process, built in `_worker_init`) ----
_WORKER: "Worker | None" = None


def _worker_init(core_list: list[int], base_seed: int) -> None:
    """Process-pool initializer: pin THIS worker to its core, build env/feature-builder, open the
    redis connection, warm the numba kernel, and construct the per-process `Worker` singleton (whose
    constructor LOCKS the numpy-only contract).  The net is loaded lazily on the first task (when the
    first weight `(phase, version)` arrives over redis)."""
    # Pin native thread counts to 1 DETERMINISTICALLY, before numba/numpy/BLAS import inside this
    # child.  CORRECTNESS/PERF for the core-pinned numba+BLAS child, INDEPENDENT of JAX (R14): the
    # worker is core-pinned and wants exactly one native thread per math runtime, so a multi-thread
    # BLAS/OpenMP/numba pool inside a single-core-pinned worker would oversubscribe and thrash.
    # Setting them here (not relying on the parent's import to have `setdefault`'d them into the
    # inherited environment) makes the worker's threading config independent of parent import order.
    # `setdefault` so an operator override still wins.  (Pre-R14 this also carried an XLA_FLAGS pin;
    # that knob was RETIRED — XLA is now absent from the JAX-free child, so the flag was dead.  See
    # the band-aid ledger in docs/notes/jaxtrain-deadlock-rca.md's R14 amendment.)
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(_var, "1")

    # WORKER-SIDE FAULTHANDLER (the cheap diagnostic; KEPT, re-justified on orthogonal grounds —
    # R14).  Removing the JAX root cause does NOT make this moot: a numba threading-init lock or a
    # redis socket stall is STILL reachable in the numpy/numba child, and either wedges silently
    # without an instrument.  Registering faulthandler with a SIGUSR1 handler makes the next
    # recurrence debuggable on the worker side: a watcher sends SIGUSR1 to the wedged worker PID and
    # gets an all-thread Python traceback on stderr (→ the run log), discriminating a native
    # threading-init lock from a timeout-less socket recv.  faulthandler writes a C-level traceback
    # from a signal handler, so it is safe even when the GIL is contended (the exact hang state).
    # Fail soft if signals aren't available (e.g. a non-POSIX host).
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
    _kwarm(env.N, len(env.detectors))   # compile the belief kernel once per process

    # NUMPY-ONLY GUARD (R14 — fail loud, ADR-0002), run HERE in the spawn child AFTER every worker
    # import this initializer pulls (env / FeatureBuilder / the belief kernel + numba threading
    # layer) has executed.  This is the surface the numpy-only contract protects — a fresh spawn
    # child — so the guard lives in `_worker_init`, not in `Worker.__init__` (which a jax-importing
    # test harness would trip in-process).  The search (`GumbelAZSearch`) and the numpy net
    # (`ValueMLP` via `unpack_net`) are pulled lazily on the first `ensure_net`; both route only
    # through the numpy forward, so the child stays JAX-free past this point too (MEASURED 2026-06-15:
    # a full 1-worker generate leaves jax/jaxlib/optax absent — see the module header + the RCA R14
    # amendment).  If a future edit leaks a jax import into the child, this raises at worker startup.
    Worker._assert_jax_free()

    global _WORKER
    _WORKER = Worker(env, FeatureBuilder(env), transport.connect(), base_seed,
                     core=core, widx=widx)


def _gen_task(args: tuple[Any, ...]) -> tuple[int, int, int, int]:
    """Worker task entrypoint (module-level so the spawn pool resolves it by qualified name):
    delegate to the per-process `Worker` singleton's `generate_episode`.  `hot_search`/`max_steps`
    in `args` are the live HOT knobs for this iteration (hp-registry §3.4)."""
    # the singleton is built by `_worker_init` before any task runs (ADR-0002 fail-loud).
    assert _WORKER is not None
    return _WORKER.generate_episode(args)


def _eval_task(args: tuple[Any, ...]) -> tuple[float, float]:
    """Worker task entrypoint (module-level so the spawn pool resolves it by qualified name):
    delegate to the per-process `Worker` singleton's `eval_episode`."""
    assert _WORKER is not None     # built by `_worker_init` before any task runs (ADR-0002)
    return _WORKER.eval_episode(args)


# ---- the two work-kinds as DATA (not two parallel ad-hoc code paths) ----
# A TaskSpec captures what differs between a generation episode and an eval episode: the kind tag the
# rng fold uses, and the module-level worker callable the pool fans out.  The shared plumbing (net
# reload + rng fold) is `Worker.prepare`; what is per-kind is exactly these two fields.  Kept
# module-level and as a small declarative record — NOT a metaprogrammed dispatch — because the spawn
# pool must resolve `callable` by qualified name (a closure / bound method would not survive the
# re-import; this is also why the work runs through the module-singleton `_WORKER`, not a pickled
# Worker argument).
class TaskSpec(NamedTuple):
    kind: str
    kind_tag: int
    # the module-level worker callable (gen -> (idx,n,fd,ns) ints / eval -> (R,T) floats), resolved
    # by the spawn pool by qualified name; typed Callable[..., Any] over the two distinct returns.
    callable: Callable[..., Any]

TASK_SPECS = {
    "gen": TaskSpec("gen", 1_000_003, _gen_task),
    "eval": TaskSpec("eval", 7_000_037, _eval_task),
}
