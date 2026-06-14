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

Why redis transport, not the pool pipe (pickle)
-----------------------------------------------
multiprocessing's default result return pickles the worker's Python objects (the per-episode
transition records — lists of float32 numpy arrays) back through ONE pipe to the parent. For
E=300 × ~30 decisions × 241 floats that is a large pickle payload funnelled through a single pipe —
both a CPU cost (pickle is slow on many small arrays) and a serialization point that caps scaling.

So neither weights nor results travel as pickle:
  * WEIGHTS broadcast — the parent packs the net's arrays as RAW bytes (`ndarray.tobytes()`, no
    pickle) plus a tiny JSON manifest (shapes/dtypes/scalars) into a single redis key
    `az:w:<run>:<version>`. Workers reconstruct via `np.frombuffer` and rebuild the net only when
    the version changes (the actor/learner contract). No disk, no pickle.
  * RESULTS return — each worker packs its episode's records into CONTIGUOUS raw-byte blocks (feats,
    pis, masks, targets stacked into float32 arrays via `tobytes()`) under a per-task redis key; the
    task returns only the small (idx, n_rows) tuple through the pipe. The parent reads the raw bytes
    back with `np.frombuffer` and reshapes — zero pickle of array data. Keys are namespaced by a
    per-call run-token and deleted after read so redis doesn't accumulate.

Connection facts come from env (`CHOCO_REDIS_HOST`/`CHOCO_REDIS_PORT`/`CHOCO_REDIS_DB`, defaulting
to 127.0.0.1:6380 db 0 — the memory-cache instance, a 1GB allkeys-lru store). Redis being
unreachable is a loud failure (ADR-0002) — the loop must not silently fall back to a slow path the
operator didn't ask for. The instance's `allkeys-lru` eviction can in principle drop a key under
memory pressure; weights carry a 1h TTL and are read-validated (a missing payload is a loud
RuntimeError, never a silent stale-net serve), and result blobs are read + deleted in the same
iteration they're written, so the eviction window is small (≤ a few MB live at once vs 1GB).

Determinism / parallel≈serial: a task's seed is folded from `base_seed`, the weight version, a kind
tag, and the episode index, so the SAME logical (iteration, kind, episode) draws the SAME stream
regardless of worker count or scheduling — the aggregate transition multiset is invariant to worker
count (verified: workers=1 and workers=4 produce bit-identical gathered data). `workers=1` runs the
pool with one pinned worker (the parallel code path, for the A/B); the loop's in-process path is the
true serial baseline.
"""
from __future__ import annotations

import json
import os
import uuid

import numpy as np


# ---- redis connection (raw-bytes transport; no pickle) ----
def _redis_params():
    return dict(
        host=os.environ.get("CHOCO_REDIS_HOST", "127.0.0.1"),
        port=int(os.environ.get("CHOCO_REDIS_PORT", "6380")),   # the memory-cache instance
        db=int(os.environ.get("CHOCO_REDIS_DB", "0")),
    )


def _connect():
    import redis  # local import so a serial (workers=0) run needs no redis at all
    # Bound EVERY socket op (ADR-0002 / deadlock fix H2). The default `socket_timeout=None`
    # makes every r.get / pipe.execute block FOREVER if the TCP socket stalls — a stalled
    # worker read is then indistinguishable, from the parent's imap fan-out, from a wedged
    # worker, and the loop sits at futex_do_wait at ~1% CPU with no way out. A bounded timeout
    # turns a stall into a loud redis.TimeoutError (retryable / restart-recoverable; checkpoints
    # are per-iteration) instead of a silent permanent hang. Loopback redis under no memory
    # pressure never trips 60s, so this is a safety net, not a happy-path behavior change.
    r = redis.Redis(
        socket_timeout=float(os.environ.get("CHOCO_REDIS_SOCKET_TIMEOUT", "60")),
        socket_connect_timeout=float(os.environ.get("CHOCO_REDIS_CONNECT_TIMEOUT", "10")),
        **_redis_params(),
    )
    r.ping()   # fail loud now if redis is unreachable (ADR-0002), not mid-iteration
    return r


# ---- net (de)serialization as raw bytes (the broadcast payload) ----
# The weight set is enumerated from the net's OWN param registry (`net._params()`), not a hardcoded
# tuple — so an optional block (e.g. the residual Wr*/br*) is transported without a second edit
# site. The `residual` flag rides the manifest so `unpack_net` rebuilds the block before binding
# its arrays (the params can only be set on a net that built the block).


def pack_net(net):
    """Pack a ValueMLP into (manifest_json: str, blob: bytes) — raw `tobytes()` of each weight
    concatenated, with a JSON manifest of (name, shape, dtype, byte-length) + the scalar meta. No
    pickle: the blob is contiguous float64 weight bytes. The weight set is whatever the net's
    param registry reports, so optional params (residual block) ride along automatically."""
    parts = []
    layout = []
    off = 0
    for k in net._params().keys():
        a = np.ascontiguousarray(getattr(net, k))
        b = a.tobytes()
        layout.append({"name": k, "shape": list(a.shape), "dtype": a.dtype.str,
                       "off": off, "len": len(b)})
        parts.append(b)
        off += len(b)
    manifest = {
        "in_dim": net.in_dim, "H": net.H, "n_actions": net.n_actions,
        "y_mean": net.y_mean, "y_std": net.y_std, "residual": net.residual, "layout": layout,
    }
    return json.dumps(manifest), b"".join(parts)


def unpack_net(manifest_json, blob):
    """Reconstruct a ValueMLP from `pack_net`'s (manifest, blob). `np.frombuffer` views, copied so
    the net owns writable arrays. No pickle. The `residual` flag rebuilds the block so the Wr*/br*
    layout entries have a slot to bind to (older manifests without the flag → block OFF)."""
    from chocofarm.az.mlp import ValueMLP
    m = json.loads(manifest_json)
    net = ValueMLP(m["in_dim"], hidden=m["H"],
                   n_actions=m["n_actions"], y_mean=m["y_mean"], y_std=m["y_std"],
                   residual=bool(m.get("residual", False)))
    for e in m["layout"]:
        a = np.frombuffer(blob, dtype=np.dtype(e["dtype"]),
                          count=int(np.prod(e["shape"])) if e["shape"] else 1,
                          offset=e["off"]).reshape(e["shape"]).copy()
        setattr(net, e["name"], a)
    return net


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
              base_seed=base_seed, core=core, widx=widx, redis=_connect())
    _kwarm(env.N, len(env.detectors))   # compile the belief kernel once per process


def _ensure_net(run, version):
    """Load the net into the worker iff the weight version changed — reading the raw-bytes payload
    from redis `az:w:<run>:<version>` (no pickle). Rebuilds the GumbelAZSearch on the fresh net."""
    if _W["version"] == version and _W["net"] is not None:
        return
    from chocofarm.az.gumbel_search import GumbelAZSearch
    r = _W["redis"]
    manifest = r.get(f"az:w:{run}:{version}:m")
    blob = r.get(f"az:w:{run}:{version}:b")
    if manifest is None or blob is None:
        raise RuntimeError(f"weight payload az:w:{run}:{version} missing from redis")
    net = unpack_net(manifest.decode("utf-8"), blob)
    _W["net"] = net
    _W["search"] = GumbelAZSearch(net, _W["env"], m=_W["m"], n_sims=_W["n_sims"])
    _W["version"] = version


def _task_rng(version, kind, idx):
    """Independent, reproducible per-(iteration, kind, episode) RNG (folds base seed + version +
    kind tag + idx). Same logical episode → same stream under any worker count (parallel≈serial)."""
    kind_tag = {"gen": 1_000_003, "eval": 7_000_037}[kind]
    seed = (np.uint64(_W["base_seed"])
            ^ (np.uint64(version + 1) * np.uint64(2_654_435_761))
            ^ (np.uint64(kind_tag) * np.uint64(40_503))
            ^ (np.uint64(idx) * np.uint64(2_246_822_519)))
    return np.random.default_rng(int(seed))


def _gen_task(args):
    """Worker task: generate ONE training episode, write its records to redis as raw bytes, return
    only (idx, n_rows, feat_dim, n_slots). `args` = (run, version, world, lam, explore_plies,
    lam_blend, n_step, idx, res_token)."""
    run, version, world, lam, explore_plies, lam_blend, n_step, idx, res_token = args
    _ensure_net(run, version)
    from chocofarm.az.exit_loop import generate_episode
    rng = _task_rng(version, "gen", idx)
    recs = generate_episode(_W["env"], _W["search"], _W["fb"], world, lam, rng,
                            explore_plies, lam_blend=lam_blend, n_step=n_step)
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
    r = _W["redis"]
    base = f"az:res:{res_token}:{idx}"
    pipe = r.pipeline(transaction=False)
    # Result blobs carry a TTL so an ABORTED iteration self-cleans. The happy path deletes them in
    # `_collect_results` the same iteration; but if the fan-out is aborted (Fix A's loud timeout)
    # the parent never reaches the delete, and a bare SET leaves the blob with no expiry forever
    # (the post-mortem found ~980 such leaked az:res:* keys, TTL=-1). A 1h TTL bounds that leak
    # without affecting the happy path (read+deleted within seconds). `ex=` sets the expiry in the
    # same SET round-trip (no extra command).
    _res_ttl = int(os.environ.get("CHOCO_RESULT_TTL", "3600"))
    pipe.set(base + ":X", X.tobytes(), ex=_res_ttl)
    pipe.set(base + ":PI", PI.tobytes(), ex=_res_ttl)
    pipe.set(base + ":M", M.tobytes(), ex=_res_ttl)
    pipe.set(base + ":Y", Y.tobytes(), ex=_res_ttl)
    pipe.execute()
    return (idx, n, feat_dim, n_slots)


def _eval_task(args):
    """Worker task: run ONE greedy-eval episode against a held-out world; return (R, T) (scalars —
    no array payload, so they ride the pipe cheaply). `args` = (run, version, world, lam, idx)."""
    run, version, world, lam, idx = args
    _ensure_net(run, version)
    from chocofarm.az.gumbel_search import GumbelPolicy
    pol = GumbelPolicy(_W["net"], _W["env"], m=_W["m"], n_sims=_W["n_sims"])
    rng = _task_rng(version, "eval", idx)
    R, T, _ = _W["env"].simulate(pol, world, lam, rng)
    return float(R), float(T)


# Per-result timeout for the fan-out drain (deadlock fix H1 / Fix A). An episode is ~0.2–0.4s of
# search × ~30 plies; 600s is ~1000× headroom and only trips on a TRUE wedge (a worker stuck in a
# native-runtime lock or a timeout-less socket read). Env-overridable.
_RESULT_TIMEOUT_S = float(os.environ.get("CHOCO_RESULT_TIMEOUT", "600"))


def _drain_imap(it, n_expected, phase, run):
    """Drive a Pool `imap_unordered` iterator to exhaustion with a PER-RESULT timeout, instead of
    the unbounded `list(imap_unordered(...))` that blocks forever on a wedged worker.

    `list(...)` parks the parent's main thread on the result-queue condition until EVERY task
    reports back; a worker that is alive-but-hung (stuck in a native-threading-init lock, or in a
    timeout-less redis recv) produces no Pool event, so the parent waits at futex_do_wait at ~1%
    CPU forever — the observed deadlock. Pulling each result with a timeout converts that silent
    permanent hang into a LOUD, diagnosable RuntimeError (ADR-0002) naming the phase, the run, and
    how many of the expected results were collected before the stall. A per-iteration checkpoint
    means a restart loses nothing."""
    import multiprocessing
    out = []
    while True:
        try:
            out.append(it.next(_RESULT_TIMEOUT_S))
        except StopIteration:
            break
        except multiprocessing.TimeoutError as e:
            raise RuntimeError(
                f"parallel {phase} fan-out (run={run}) stalled: collected {len(out)}/{n_expected} "
                f"results, then NO further result arrived within {_RESULT_TIMEOUT_S:.0f}s of the "
                f"previous one (per-result wait, not a whole-fan-out bound) — likely a wedged "
                f"worker (native-runtime lock or redis socket stall). To get a worker-side "
                f"traceback, send SIGUSR1 to the stuck worker PID (faulthandler is registered in "
                f"_worker_init). Aborting loud rather than deadlocking at futex_do_wait "
                f"(ADR-0002). Restart from the last checkpoint."
            ) from e
    return out


class ParallelExecutor:
    """Persistent process pool of `workers` core-pinned workers with a redis raw-bytes transport for
    the ExIt loop's two fan-outs. Construct once before the iteration loop; call `generate(...)` /
    `evaluate(...)` each iteration. Close at the end (or use as a context manager)."""

    def __init__(self, n_workers, cores, base_seed, m, n_sims):
        import multiprocessing as mp
        self.n_workers = int(n_workers)
        self.cores = list(cores)[:self.n_workers] if cores else list(range(self.n_workers))
        self.run = uuid.uuid4().hex[:12]            # namespace this run's redis keys
        self.r = _connect()                         # parent connection (weight publish + result read)
        ctx = mp.get_context("spawn")               # spawn: clean process, no inherited numba/jax state
        self.pool = ctx.Pool(
            processes=self.n_workers,
            initializer=_worker_init,
            initargs=(self.cores, base_seed, m, n_sims),
        )

    def publish_weights(self, net, version):
        """Pack the net to raw bytes and publish to redis `az:w:<run>:<version>` (no pickle, no
        disk). Workers `_ensure_net` read it when the version changes."""
        manifest, blob = pack_net(net)
        pipe = self.r.pipeline(transaction=False)
        pipe.set(f"az:w:{self.run}:{version}:m", manifest)
        pipe.set(f"az:w:{self.run}:{version}:b", blob)
        # weights expire after an hour so a long run doesn't leak old versions
        pipe.expire(f"az:w:{self.run}:{version}:m", 3600)
        pipe.expire(f"az:w:{self.run}:{version}:b", 3600)
        pipe.execute()

    def generate(self, net, version, worlds, lam, explore_plies, lam_blend, n_step):
        """Publish `net` at `version`, fan E generation episodes across the pool, read the raw-byte
        results back from redis and reshape into one flat list of (feat, pi, mask, g) records. The
        parent draws `worlds` so the world sequence is reproducible regardless of worker count."""
        self.publish_weights(net, version)
        res_token = uuid.uuid4().hex[:12]
        tasks = [(self.run, version, int(w), lam, explore_plies, lam_blend, n_step, i, res_token)
                 for i, w in enumerate(worlds)]
        # bounded drain (Fix A): per-result timeout so a wedged worker aborts loud, not deadlocks
        metas = _drain_imap(self.pool.imap_unordered(_gen_task, tasks, chunksize=1),
                            len(tasks), "generate", self.run)
        return self._collect_results(res_token, metas)

    def _collect_results(self, res_token, metas):
        out = []
        pipe = self.r.pipeline(transaction=False)
        order = []
        for (idx, n, fd, ns) in metas:
            if n == 0:
                continue
            base = f"az:res:{res_token}:{idx}"
            pipe.get(base + ":X"); pipe.get(base + ":PI")
            pipe.get(base + ":M"); pipe.get(base + ":Y")
            order.append((idx, n, fd, ns, base))
        if not order:
            return out
        blobs = pipe.execute()
        # delete the result keys (raw bytes can be large; don't leak across iterations)
        dpipe = self.r.pipeline(transaction=False)
        for k, (idx, n, fd, ns, base) in enumerate(order):
            xb, pib, mb, yb = blobs[4 * k:4 * k + 4]
            X = np.frombuffer(xb, dtype=np.float32).reshape(n, fd)
            PI = np.frombuffer(pib, dtype=np.float32).reshape(n, ns)
            M = np.frombuffer(mb, dtype=np.float32).reshape(n, ns)
            Y = np.frombuffer(yb, dtype=np.float32)
            for i in range(n):
                out.append((X[i], PI[i], M[i], float(Y[i])))
            dpipe.delete(base + ":X", base + ":PI", base + ":M", base + ":Y")
        dpipe.execute()
        return out

    def evaluate(self, net, version, worlds, lam):
        """Publish `net` at `version`, fan N eval episodes across the pool; return (totR, totT,
        list_of_T). Eval results are scalars (R, T) so they ride the pipe directly — no redis blob
        needed (the array transport is only for the large generation records)."""
        self.publish_weights(net, version)
        tasks = [(self.run, version, int(w), lam, i) for i, w in enumerate(worlds)]
        totR = totT = 0.0
        ets = []
        # bounded drain (Fix A): per-result timeout so a wedged worker aborts loud, not deadlocks
        for R, T in _drain_imap(self.pool.imap_unordered(_eval_task, tasks, chunksize=1),
                                len(tasks), "evaluate", self.run):
            totR += R; totT += T; ets.append(T)
        return totR, totT, ets

    def close(self):
        # Bounded teardown (completes the "parent never waits unbounded" invariant — see Fix A).
        # `Pool.join()` takes NO timeout, so a worker wedged at end-of-run would hang close()
        # forever exactly as the hot loop could. Instead: close (no new tasks), then join each
        # worker process with a timeout; any worker still alive after the grace period is
        # terminate()'d (SIGTERM) so teardown is bounded. Low-severity vs the hot loop (runs once
        # at end-of-run) but it makes the no-unbounded-wait property hold everywhere.
        self.pool.close()
        grace = float(os.environ.get("CHOCO_POOL_JOIN_TIMEOUT", "30"))
        procs = list(getattr(self.pool, "_pool", []))
        for p in procs:
            p.join(grace)
        for p in procs:
            if p.is_alive():
                p.terminate()      # bounded: don't let a wedged worker hang teardown
        try:
            self.pool.join()       # reap; all workers are now exited or terminated
        except Exception:
            pass
        try:
            self.r.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
