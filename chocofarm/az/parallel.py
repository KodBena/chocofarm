#!/usr/bin/env python3
"""
chocofarm AZ — Part A: 4-core actor/learner episode parallelism for the ExIt loop (redis transport).

Each outer iteration is generate → train → eval → checkpoint. The two episode fan-outs (E generation
episodes, N eval episodes) are embarrassingly parallel — each an independent rollout under a frozen
net — while TRAIN stays central in the parent (one Adam pass over the gathered transitions). This
module parallelises the two fan-outs across a persistent process pool of `workers` processes, each
PINNED to a distinct core (`os.sched_setaffinity` in the initializer, cores 0..workers-1) with an
INDEPENDENT per-worker + per-episode RNG seed so episodes are uncorrelated.

Why processes, not threads: the search is pure-Python tree control flow (the GIL-bound 216ms/ep
floor, az-jax-perf.md) — threads would serialise on the GIL. Processes give true 4-core parallelism.
The pool is stdlib `multiprocessing`; the DATA TRANSPORT is **redis**, not pickle.

Three orthogonal collaborators (audit item K — Transport ⊥ Pool ⊥ Task)
----------------------------------------------------------------------
`ParallelExecutor` is the THIN orchestrator that composes three single-responsibility collaborators,
each the sole owner of one concern that used to be fused into this god-object:
  * `transport.RedisTransport` (`chocofarm/az/transport.py`) — the SOLE owner of the redis raw-bytes
    protocol: the bounded-timeout connection + fail-loud ping, the ONE place the `az:w:...` weight
    keys and the `az:res:...` result keys are spelled (`weight_keys`/`result_keys`), the weight
    publish, the worker-side weight read, the result-blob write/read+delete, and the TTLs. Weight
    (un)packing stays delegated to `WeightContainer` (audit J).
  * `worker_pool.WorkerPool` (`chocofarm/az/worker_pool.py`) — the SOLE owner of the multiprocessing
    lifecycle: the spawn pool built with `worker._worker_init`, the per-result bounded drain, the
    `imap_unordered` fan-out under that drain (`map`), and the bounded `close()` teardown.
  * `worker.Worker` + `worker.TaskSpec` (`chocofarm/az/worker.py`) — the unit of work: the `Worker`
    object (item L / R14 — promoted from the per-process `_W` dict) owning env/feature-builder/net/
    search/current-(phase,version)/redis/base_seed, with `_gen_task`/`_eval_task` as the two
    module-level work-kind entrypoints (DATA in `TASK_SPECS`) that delegate to the per-process
    singleton `_WORKER`, the `(phase, version)`-gated `Worker.ensure_net`, and the seed fold
    `Worker.task_rng`.  The child is JAX-free (the numpy-only contract that removes the deadlock root
    cause), LOCKED by a fail-loud guard in `Worker.__init__`.

For back-compat (and because the test suite + the weights/registry docstrings reference these names
on `parallel`), the collaborators' public callables are re-exported here: `pack_net` / `unpack_net` /
`_connect` / `_drain_imap` / `_worker_init` / `_gen_task` / `_eval_task` / `Worker`.

Why redis transport, not the pool pipe (pickle)
-----------------------------------------------
multiprocessing's default result return pickles the worker's Python objects (the per-episode
transition records — lists of float32 numpy arrays) back through ONE pipe to the parent. For
E=300 × ~30 decisions × 241 floats that is a large pickle payload funnelled through a single pipe —
both a CPU cost (pickle is slow on many small arrays) and a serialization point that caps scaling.

So neither weights nor results travel as pickle:
  * WEIGHTS broadcast — the parent packs the net's arrays as RAW bytes (`ndarray.tobytes()`, no
    pickle) plus a tiny JSON manifest (shapes/dtypes/scalars) into a single redis key
    `az:w:<run>:<phase>:<version>`. Workers reconstruct via `np.frombuffer` and rebuild the net only
    when the `(phase, version)` changes (the actor/learner contract; the R14 phase segment lets gen
    and eval of one iteration use distinct keys at the same real `version`). No disk, no pickle.
  * RESULTS return — each worker packs its episode's records into CONTIGUOUS raw-byte blocks (feats,
    pis, masks, targets stacked into float32 arrays via `tobytes()`) under a per-task redis key; the
    task returns only the small (idx, n_rows) tuple through the pipe. The parent reads the raw bytes
    back with `np.frombuffer` and reshapes — zero pickle of array data. Keys are namespaced by a
    per-call run-token and deleted after read so redis doesn't accumulate.

Connection facts come from `chocofarm/config.py` (`redis_params()`, env-overridable via
`CHOCO_REDIS_HOST`/`CHOCO_REDIS_PORT`/`CHOCO_REDIS_DB`), defaulting to 127.0.0.1:6379 db 0 — the
disk-persisted instance (`noeviction`, no `maxmemory` cap). Redis being unreachable is a loud
failure (ADR-0002) — the loop must not silently fall back to a slow path the operator didn't ask
for. Weights carry a 1h TTL and are read-validated (a missing payload is a loud RuntimeError, never
a silent stale-net serve), and result blobs are read + deleted in the same iteration they're
written, so transport keys never accumulate.

Determinism / parallel≈serial: a task's seed is folded from `base_seed`, the weight version, a kind
tag, and the episode index, so the SAME logical (iteration, kind, episode) draws the SAME stream
regardless of worker count or scheduling — the aggregate transition multiset is invariant to worker
count (verified: workers=1 and workers=4 produce bit-identical gathered data). `workers=1` runs the
pool with one pinned worker (the parallel code path, for the A/B); the loop's in-process path is the
true serial baseline.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import uuid

from chocofarm.az import transport
from chocofarm.az import worker as _worker
from chocofarm.az import worker_pool
from chocofarm.az.transport import connect as _connect, pack_net, unpack_net
from chocofarm.az.worker import Worker, _eval_task, _gen_task, _worker_init
from chocofarm.az.worker_pool import _RESULT_TIMEOUT_S, _drain_imap

__all__ = [
    "ParallelExecutor", "pack_net", "unpack_net", "_connect", "_drain_imap",
    "_worker_init", "_gen_task", "_eval_task", "Worker",
]


class ParallelExecutor:
    """Persistent process pool of `workers` core-pinned workers with a redis raw-bytes transport for
    the ExIt loop's two fan-outs. Construct once before the iteration loop; call `generate(...)` /
    `evaluate(...)` each iteration. Close at the end (or use as a context manager).

    A THIN orchestrator (audit item K): it composes a `RedisTransport` (the wire protocol), a
    `WorkerPool` (the multiprocessing lifecycle), and the `worker.TASK_SPECS` work-kinds. It owns no
    redis key strings and no pool internals — only the per-iteration choreography (publish → fan-out →
    gather) and the public surface the loop depends on."""

    def __init__(self, n_workers, cores, base_seed, m, n_sims):
        self.n_workers = int(n_workers)
        self.cores = list(cores)[:self.n_workers] if cores else list(range(self.n_workers))
        self.run = uuid.uuid4().hex[:12]            # namespace this run's redis keys
        self.transport = transport.RedisTransport(_connect())   # parent connection (publish + read)
        self.pool = worker_pool.WorkerPool(self.n_workers, self.cores, base_seed, m, n_sims)

    @property
    def r(self):
        """The parent's redis connection — preserved as a public attribute (it was `self.r` before
        the split). It lives on the transport collaborator now; this property keeps the name."""
        return self.transport.r

    def publish_weights(self, net, version, phase="gen"):
        """Pack the net to raw bytes and publish to redis `az:w:<run>:<phase>:<version>` (no pickle,
        no disk).  Workers reload it when the worker-side `(phase, version)` gate changes.  `phase`
        (R14) namespaces gen vs eval within one iteration; it defaults to "gen" so a bare
        `publish_weights(net, version)` (back-compat) still publishes the gen-phase key."""
        self.transport.publish_weights(net, phase, version, self.run)

    def generate(self, net, version, worlds, lam, explore_plies, lam_blend, n_step,
                 hot_search=None, max_steps=40):
        """Publish `net` at `("gen", version)`, fan E generation episodes across the pool, read the
        raw-byte results back from redis and reshape into one flat list of (feat, pi, mask, g)
        records.  The parent draws `worlds` so the world sequence is reproducible regardless of worker
        count.

        Phase is set internally to "gen" (the public signature stays `(net, version, ...)` — R14): the
        gen weights land at `az:w:<run>:gen:<version>`, distinct from the eval phase's key for the
        SAME `version`, so the worker reloads correctly at the gen→eval transition.

        `hot_search`/`max_steps` (hp-registry §3.4): the live HOT search knobs + rollout cap for
        this iteration, threaded into each task so the worker rebuilds its search with the live
        values on the `(phase, version)` change (which happens every phase/iteration)."""
        self.publish_weights(net, version, phase="gen")
        res_token = uuid.uuid4().hex[:12]
        hs = dict(hot_search) if hot_search else {}
        tasks = [(self.run, version, int(w), lam, explore_plies, lam_blend, n_step, i, res_token,
                  hs, max_steps)
                 for i, w in enumerate(worlds)]
        # bounded drain (Fix A): per-result timeout so a wedged worker aborts loud, not deadlocks
        metas = self.pool.map(_worker.TASK_SPECS["gen"].callable, tasks, "generate", self.run)
        return self.transport.read_and_delete_results(res_token, metas)

    def evaluate(self, net, version, worlds, lam, hot_search=None, max_steps=40):
        """Publish `net` at `("eval", version)`, fan N eval episodes across the pool; return (totR,
        totT, list_of_T).  Eval results are scalars (R, T) so they ride the pipe directly — no redis
        blob needed (the array transport is only for the large generation records).

        Phase is set internally to "eval" (the public signature stays `(net, version, ...)` — R14):
        the POST-TRAIN eval weights land at `az:w:<run>:eval:<version>` at the SAME real `version` the
        gen phase used, NOT a faked `version + 1_000_000`.  The worker's `(phase, version)` gate
        reloads the trained weights at the gen→eval transition because the phase changed.

        `hot_search`/`max_steps` (hp-registry §3.4): the live HOT search knobs + rollout cap,
        threaded as in `generate`."""
        self.publish_weights(net, version, phase="eval")
        hs = dict(hot_search) if hot_search else {}
        tasks = [(self.run, version, int(w), lam, i, hs, max_steps) for i, w in enumerate(worlds)]
        totR = totT = 0.0
        ets = []
        # bounded drain (Fix A): per-result timeout so a wedged worker aborts loud, not deadlocks
        for R, T in self.pool.map(_worker.TASK_SPECS["eval"].callable, tasks, "evaluate", self.run):
            totR += R; totT += T; ets.append(T)
        return totR, totT, ets

    def close(self):
        # Bounded teardown (the "parent never waits unbounded" invariant — Fix A) lives on the
        # WorkerPool; the parent's redis connection is closed here after the pool is reaped.
        self.pool.close()
        try:
            self.transport.r.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
