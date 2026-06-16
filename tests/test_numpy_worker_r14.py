#!/usr/bin/env python3
"""
test_numpy_worker_r14.py — the R14 gate: the numpy-only JAX-free worker child, the `(run, phase,
version)` weight-key namespacing, and the parallel≈serial bit-identity it must preserve.

These spin up a REAL 1-worker `ParallelExecutor` against the disk-persisted redis (127.0.0.1:6379)
and assert the three load-bearing R14 properties the unit pins in `test_transport_split.py` cannot
reach (they need a live spawn child + redis):

  * NO-JAX-IN-CHILD (the deadlock ROOT-CAUSE removal): a probe task run INSIDE the worker process,
    after `_worker_init` + all worker imports, reports `jax`/`jaxlib`/`optax` ABSENT from the child's
    `sys.modules`.  (The `_worker_init` → `Worker._assert_jax_free` fail-loud guard is the in-band
    enforcement; this test is the out-of-band proof that the guard's premise holds on the real import
    graph, plus a unit test that the guard FIRES when jax is present.)
  * NAMESPACING CORRECTNESS: gen and eval for the SAME iteration `it` publish to DISTINCT weight keys
    (`az:w:<run>:gen:<it>` vs `az:w:<run>:eval:<it>`, no collision), and the worker RELOADS the
    POST-TRAIN weights at the eval phase within that same `it` (a mutated-net eval blob differs from
    the gen blob; the reload gate keys on (phase, version), not version alone).
  * PARALLEL≈SERIAL BIT-IDENTITY: a 1-worker `generate` produces transitions byte-identical on the
    float32 wire to the in-process serial path (item K's invariant — phase namespacing must not
    perturb the rng or the wire bytes; phase does NOT enter the seed fold).

REDIS SAFETY (the disk-persisted redis at 127.0.0.1:6379 is SHARED with a live training run):
  * the executor namespaces its weight keys under a per-run uuid (`az:w:<run-uuid>:...`); each test
    deletes its own keys (try/finally), and the result blobs carry the self-clean TTL.
  * NO FLUSHALL / FLUSHDB, NO CONFIG SET, NO touching another run's az:* keys.
  * skips cleanly if redis is unreachable.

Run pinned + bounded, e.g.:
    PYTHONPATH=. taskset -c 0,1 timeout 240 /home/bork/w/vdc/venvs/generic/bin/python -m pytest \
        tests/test_numpy_worker_r14.py -q

Public Domain (The Unlicense).
"""
import os
import sys

os.environ.setdefault("CHOCO_AZ_DTYPE", "float32")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pytest

from chocofarm.az import worker as W
from chocofarm.az import transport as T
from chocofarm.model.env import Environment
from chocofarm.az.features import FeatureBuilder, feature_dim
from chocofarm.az.actions import n_action_slots
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.gumbel_search import GumbelAZSearch
from chocofarm.az.exit_loop import generate_episode


# fixed small search budget so the gate runs in seconds, pinned to the test cores
_BASE_SEED, _M, _NS = 7777, 4, 8
_EXPLORE, _MAXST = 2, 8


# ---- spawn-child jax probe (module-level so the spawn Pool resolves it by qualified name) ----
# The pool uses `spawn`, so the worker CHILD re-imports modules by qualified name — a monkeypatch in
# a parent test-function body never reaches the child.  Instead we run the EXACT production
# `worker._worker_init` initializer in a real spawn Pool, then fan out THIS module-level probe as the
# task: it reports the child's `sys.modules` membership for jax/jaxlib/optax, observed AFTER
# `_worker_init` (which builds the Worker — running its numpy-only guard — and warms the numba
# kernel) and after the worker's own imports.  The probe is importable in the child as
# `tests.test_numpy_worker_r14._child_jax_probe` (the repo root is on sys.path).
def _child_jax_probe(_ignored):
    import sys as _sys
    return {m_: (m_ in _sys.modules) for m_ in ("jax", "jaxlib", "optax")}

    W.Worker.ensure_net = _ensure_with_report


def _redis_or_skip():
    from chocofarm.az import transport
    try:
        return transport.connect()
    except Exception as e:                      # redis down / unreachable
        pytest.skip(f"redis unreachable: {e}")


def _make_net():
    env = Environment()
    in_dim, na = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=32, n_actions=na, seed=0)
    net.set_value_scale(0.0, 1.0)
    return env, net


def _to_wire(recs):
    """Stack a flat list of (feat, pi, mask, g) records into the float32 wire blocks the worker
    writes (X|PI|M|Y, contiguous, in the records' order) — the byte representation that travels over
    redis.  Comparing these bytes is the float32-wire bit-identity check (item K's bar)."""
    if not recs:
        return b""
    n = len(recs); fd = recs[0][0].shape[0]; ns = recs[0][1].shape[0]
    X = np.empty((n, fd), np.float32); PI = np.empty((n, ns), np.float32)
    M = np.empty((n, ns), np.float32); Y = np.empty(n, np.float32)
    for i, (f, pi, mask, g) in enumerate(recs):
        X[i] = f; PI[i] = pi; M[i] = mask; Y[i] = g
    return X.tobytes() + PI.tobytes() + M.tobytes() + Y.tobytes()


def _serial_generate(env, net, worlds, lam, version=0):
    """The in-process serial baseline, mirroring exit_loop's else-branch AND the worker's per-episode
    rng fold (the worker folds rng per (version, "gen", idx) with base_seed via Worker._fold_seed —
    PHASE does not enter it, so the serial path reproduces the worker's stream exactly)."""
    fb = FeatureBuilder(env)
    search = GumbelAZSearch(net, env, m=_M, n_sims=_NS)
    recs_all = []
    for i, w in enumerate(worlds):
        seed = int(W.Worker._fold_seed(_BASE_SEED, version, "gen", i))
        rng = np.random.default_rng(seed)
        recs_all.extend(generate_episode(env, search, fb, int(w), lam, rng, _EXPLORE,
                                         max_steps=_MAXST, lam_blend=1.0, n_step=None))
    return recs_all


def test_assert_jax_free_guard_fires_and_passes():
    """GATE 2 (the guard itself, deterministic) — `Worker._assert_jax_free` PASSES when jax is absent
    and RAISES a loud RuntimeError when `jax`/`jaxlib` is in `sys.modules`.  We don't actually import
    jax (it stays out of this pytest process for the in-process tests below); we inject a sentinel
    into `sys.modules` and assert the guard catches it, then remove it and assert the guard passes."""
    import sys as _sys
    # passes when absent (this assumes no jax loaded yet in-process — the suite orders this file's
    # in-process tests before any jax-importing test; if jax IS loaded, skip the pass-branch).
    if "jax" not in _sys.modules and "jaxlib" not in _sys.modules:
        W.Worker._assert_jax_free()   # must NOT raise
    # fires when present
    injected = "jax" not in _sys.modules
    if injected:
        _sys.modules["jax"] = object()
    try:
        with pytest.raises(RuntimeError) as ei:
            W.Worker._assert_jax_free()
        assert "JAX-free" in str(ei.value) and "R14" in str(ei.value)
    finally:
        if injected:
            del _sys.modules["jax"]


def test_worker_child_is_jax_free():
    """GATE 2 — the deadlock ROOT-CAUSE removal.  Run the EXACT production `worker._worker_init`
    initializer in a real spawn Pool, then probe the child's `sys.modules` AFTER init.
    `jax`/`jaxlib`/`optax` must ALL be absent — the out-of-band proof of the numpy-only contract the
    `_worker_init` → `Worker._assert_jax_free` guard locks.  (If a jax import HAD leaked into the
    child's import graph, the guard would already have raised inside `_worker_init`, so the Pool would
    fail to start; this test additionally pins the positive truth — the child is clean.)"""
    _redis_or_skip()     # _worker_init opens a redis connection; skip cleanly if redis is down
    import multiprocessing as mp
    from chocofarm.az.worker import _worker_init

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=1, initializer=_worker_init,
                    initargs=([0], _BASE_SEED))
    try:
        ans = pool.apply(_child_jax_probe, ((),))
    finally:
        pool.close()
        pool.join()

    assert ans == {"jax": False, "jaxlib": False, "optax": False}, ans


def test_gen_eval_namespace_no_collision_real_redis():
    """GATE 6 (namespacing, integration half) — gen and eval for the SAME `it` publish to DISTINCT
    weight keys in real redis (no collision), and the eval phase publishes the POST-TRAIN net.  We run
    a real 1-worker gen, MUTATE the net (simulating a train step), then a real 1-worker eval at the
    SAME `it=0`, and assert: both phase keys exist, they are different keys, and the eval blob differs
    from the gen blob.  The eval fan-out completing without a loud `weight payload az:w:...:eval:0
    missing` RuntimeError (read_weights is fail-loud, ADR-0002) proves the worker READ the eval key.
    The deterministic proof that the worker's (phase,version) gate RELOADS (rather than serving the
    stale gen net) is the redis-free `test_ensure_net_reload_gate` below."""
    _redis_or_skip()
    from chocofarm.az.parallel import ParallelExecutor

    env, net = _make_net()
    gen_worlds = [int(env.worlds[k]) for k in (10, 200)]
    eval_worlds = [int(env.worlds[k]) for k in (5, 6)]

    ex = ParallelExecutor(n_workers=1, cores=[0], base_seed=_BASE_SEED)
    r = ex.r
    run = ex.run
    try:
        ex.generate(net, 0, gen_worlds, 0.0855, _EXPLORE, lam_blend=1.0, n_step=None,
                    hot_search={"m": _M, "n_sims": _NS}, max_steps=_MAXST)
        gen_blob = r.get(f"az:w:{run}:gen:0:b")

        # simulate a train step: mutate the net so the eval (post-train) weights differ from gen's
        rng = np.random.default_rng(123)
        net.W1 = net.W1 + 0.5 * rng.standard_normal(net.W1.shape)
        net.Wv = net.Wv + 0.5 * rng.standard_normal(net.Wv.shape)

        # eval fan-out at the SAME it=0 — completes only if the worker reads az:w:<run>:eval:0
        ex.evaluate(net, 0, eval_worlds, 0.0855, hot_search={"m": _M, "n_sims": _NS},
                    max_steps=_MAXST)
        eval_blob = r.get(f"az:w:{run}:eval:0:b")
    finally:
        for ph in ("gen", "eval"):
            try:
                r.delete(f"az:w:{run}:{ph}:0:m", f"az:w:{run}:{ph}:0:b")
            except Exception:
                pass
        ex.close()

    assert gen_blob is not None and eval_blob is not None
    assert f"az:w:{run}:gen:0:b" != f"az:w:{run}:eval:0:b"     # distinct keys, no collision
    assert gen_blob != eval_blob, "eval phase did not publish the post-train weights"


class _FakeRedis:
    """A minimal in-memory stand-in for the worker's redis connection: only `get` is exercised by
    `Worker.ensure_net` (via `transport.read_weights`).  Counts gets so the reload-gate's
    short-circuit vs reload behavior is observable without a real pool or server."""

    def __init__(self):
        self.store = {}
        self.get_calls = 0

    def set_blob(self, key, val):
        self.store[key] = val

    def get(self, key):
        self.get_calls += 1
        return self.store.get(key)


def test_ensure_net_reload_gate():
    """GATE 6 (reload gate, deterministic half) — the worker's `(phase, version)` reload gate fires on
    BOTH the gen→eval transition (same `it`, phase changes) AND the it→it+1 transition (version
    changes), and short-circuits ONLY when BOTH are unchanged.  This is the byte-level proof the eval
    phase reloads the POST-TRAIN weights within one `it` — a version-only gate would short-circuit at
    eval (version==it unchanged) and serve the stale gen net.  Redis-free (a fake conn serves distinct
    blobs per (phase, version)); we assert on the LOADED net's identity, not on episode noise."""
    env = Environment()
    in_dim, na = feature_dim(env), n_action_slots(env)

    # three distinct nets: gen@0 (pre-train), eval@0 (post-train, SAME it), gen@1 (next it).
    nets = {}
    fake = _FakeRedis()
    for tag, (phase, version, seed) in {
        "gen0": ("gen", 0, 11), "eval0": ("eval", 0, 22), "gen1": ("gen", 1, 33),
    }.items():
        n = ValueMLP(in_dim, hidden=16, n_actions=na, seed=seed)
        n.set_value_scale(0.0, 1.0)
        nets[tag] = n
        manifest, blob = T.pack_net(n)
        mk, bk = T.weight_keys("runX", phase, version)
        fake.set_blob(mk, manifest.encode("utf-8")); fake.set_blob(bk, blob)

    wk = W.Worker(env=env, fb=FeatureBuilder(env), redis=fake, base_seed=_BASE_SEED)

    def loaded_w1sum():
        return float(np.asarray(wk.net.W1, np.float64).sum())

    s_gen0 = float(np.asarray(nets["gen0"].W1, np.float64).sum())
    s_eval0 = float(np.asarray(nets["eval0"].W1, np.float64).sum())
    s_gen1 = float(np.asarray(nets["gen1"].W1, np.float64).sum())
    # the three source nets are genuinely distinct (so a stale serve is observable)
    assert len({round(s_gen0, 6), round(s_eval0, 6), round(s_gen1, 6)}) == 3

    # (1) first load at (gen, 0) — reloads (was sentinel)
    wk.ensure_net("runX", "gen", 0)
    assert abs(loaded_w1sum() - s_gen0) < 1e-9
    gets_after_gen0 = fake.get_calls
    assert gets_after_gen0 > 0

    # (2) SAME (gen, 0) again — short-circuits (no new gets)
    wk.ensure_net("runX", "gen", 0)
    assert fake.get_calls == gets_after_gen0, "reloaded on an unchanged (phase, version)"
    assert abs(loaded_w1sum() - s_gen0) < 1e-9

    # (3) gen→eval at the SAME version=0 — MUST reload (phase changed) → the POST-TRAIN net
    wk.ensure_net("runX", "eval", 0)
    assert fake.get_calls > gets_after_gen0, "did NOT reload at gen→eval (version-only gate bug!)"
    assert abs(loaded_w1sum() - s_eval0) < 1e-9, "eval phase served the stale gen net"
    gets_after_eval0 = fake.get_calls

    # (4) it→it+1 (gen, 1) — MUST reload (version changed)
    wk.ensure_net("runX", "gen", 1)
    assert fake.get_calls > gets_after_eval0, "did NOT reload at it→it+1"
    assert abs(loaded_w1sum() - s_gen1) < 1e-9


def test_parallel_one_worker_bit_identical_to_serial():
    """GATE 7 — parallel≈serial bit-identity on the float32 wire.  A 1-worker `generate` must produce
    transitions byte-identical to the in-process serial path (item K's invariant): the `(phase,
    version)` namespacing changes the KEY shape only, never the rng (phase is absent from the seed
    fold) nor the wire bytes.  We compare the stacked X|PI|M|Y float32 bytes in idx order."""
    _redis_or_skip()
    from chocofarm.az.parallel import ParallelExecutor

    env, net = _make_net()
    worlds = [int(env.worlds[k]) for k in (10, 200, 999)]

    serial_recs = _serial_generate(env, net, worlds, 0.0855, version=0)
    serial_wire = _to_wire(serial_recs)
    assert len(serial_recs) > 0, "serial path produced no transitions (test is vacuous)"

    ex = ParallelExecutor(n_workers=1, cores=[0], base_seed=_BASE_SEED)
    try:
        par_recs = ex.generate(net, 0, worlds, 0.0855, _EXPLORE, lam_blend=1.0, n_step=None,
                               hot_search={"m": _M, "n_sims": _NS}, max_steps=_MAXST)
    finally:
        try:
            ex.r.delete(f"az:w:{ex.run}:gen:0:m", f"az:w:{ex.run}:gen:0:b")
        except Exception:
            pass
        ex.close()

    par_wire = _to_wire(par_recs)
    assert len(par_recs) == len(serial_recs), (len(par_recs), len(serial_recs))
    assert par_wire == serial_wire, "1-worker parallel transitions differ from serial on the float32 wire"


def test_fold_seed_grid_snapshot():
    """The `Worker._fold_seed` arithmetic across a grid of (kind, version, idx) — a snapshot of the
    EXACT pre/post-R14 fold (phase namespacing must not have perturbed it).  This is the local,
    redis-free companion to the bit-identity test: it pins the seed values themselves, so a future
    edit to the fold (a multiplier typo, a dropped `+1`) is caught even if no integration test runs.
    The values are computed from the documented arithmetic, NOT copied from a run, so the test fails
    loud if the live fold drifts from the spec."""
    def spec(base, version, kind, idx):
        kind_tag = {"gen": 1_000_003, "eval": 7_000_037}[kind]
        return int(np.uint64(base)
                   ^ (np.uint64(version + 1) * np.uint64(2_654_435_761))
                   ^ (np.uint64(kind_tag) * np.uint64(40_503))
                   ^ (np.uint64(idx) * np.uint64(2_246_822_519)))

    for base in (0, 7777, 2 ** 40):
        for kind in ("gen", "eval"):
            for version in (0, 1, 17, 1_000_000):
                for idx in (0, 3, 41):
                    got = int(W.Worker._fold_seed(base, version, kind, idx))
                    assert got == spec(base, version, kind, idx), (base, kind, version, idx)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all R14 numpy-worker checks passed")
