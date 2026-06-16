#!/usr/bin/env python3
"""
test_transport_split.py — collaborator-unit pins for the Transport ⊥ Pool ⊥ Task split (audit item K,
`chocofarm/az/parallel.py` → `transport.py` + `worker_pool.py` + `worker.py`).

These assert the load-bearing INVARIANTS of the split without spinning up a real Pool or redis:
  * the redis weight KEY now carries the R14 `phase` segment (`az:w:<run>:<phase>:<version>`) and the
    result key is unchanged (`az:res:<token>:<idx>:X|PI|M|Y`), both spelled in ONE place
    (`transport.weight_keys` / `transport.result_keys`);
  * the `Worker.task_rng` seed fold is byte-for-byte the pre-split fold (the parallel≈serial
    determinism contract), driven by the `TASK_SPECS` kind tags (gen=1_000_003 / eval=7_000_037), and
    PHASE does NOT enter it (it namespaces the weight key only);
  * the result TTL + weight TTL constants are preserved (the aborted-iteration self-clean band-aids);
  * the public re-exports on `parallel` survive the split + the item-L `Worker` promotion.

Run pinned + bounded, e.g.:
    taskset -c 3 timeout 60 /home/bork/w/vdc/venvs/generic/bin/python -m pytest \
        tests/test_transport_split.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chocofarm.az import transport as T
from chocofarm.az import worker as W
from chocofarm.az import parallel as P


def test_weight_keys_carry_phase_segment():
    """`az:w:<run>:<phase>:<version>:m` / `:b` — the R14 phase-namespaced weight key.  The phase
    segment (gen/eval) sits between run and version so the two phases of one iteration `it` publish to
    DISTINCT keys at the SAME real `version=it` (replacing the `it + 1_000_000` version hack)."""
    mk, bk = T.weight_keys("abcdef123456", "gen", 42)
    assert mk == "az:w:abcdef123456:gen:42:m"
    assert bk == "az:w:abcdef123456:gen:42:b"
    emk, ebk = T.weight_keys("abcdef123456", "eval", 42)
    assert emk == "az:w:abcdef123456:eval:42:m"
    assert ebk == "az:w:abcdef123456:eval:42:b"


def test_gen_eval_weight_keys_distinct_at_same_version():
    """The R14 namespacing correctness invariant at the KEY level: gen and eval for the SAME (run,
    version) map to DISTINCT keys (no collision), so the eval phase's POST-TRAIN weights never
    overwrite or alias the gen phase's at the same iteration."""
    gmk, gbk = T.weight_keys("run0", "gen", 7)
    emk, ebk = T.weight_keys("run0", "eval", 7)
    assert gmk != emk and gbk != ebk
    assert {gmk, gbk, emk, ebk} == {"az:w:run0:gen:7:m", "az:w:run0:gen:7:b",
                                    "az:w:run0:eval:7:m", "az:w:run0:eval:7:b"}


def test_result_keys_byte_identical_to_protocol():
    """`az:res:<token>:<idx>:X|PI|M|Y` — byte-identical to the pre-split f-strings (invariant 10)."""
    xk, pik, mk, yk = T.result_keys("fedcba654321", 3)
    assert (xk, pik, mk, yk) == ("az:res:fedcba654321:3:X", "az:res:fedcba654321:3:PI",
                                 "az:res:fedcba654321:3:M", "az:res:fedcba654321:3:Y")


def test_key_namespace_is_transports_sole_concern():
    """Transport is the ONE owner of the wire protocol: the worker side builds its read/write keys
    through the SAME `transport` helpers, so parent and child can never spell a key differently."""
    # the worker module reaches the protocol through `transport`, not its own key f-strings: it must
    # not CONSTRUCT a key (an f-string interpolating run/version/token/idx into the `az:` namespace).
    # (A docstring that NAMES the key as `az:w:<run>:<version>` is a reference, not a builder — so we
    # forbid only the f-string forms `f"az:w:` / `f"az:res:`, which are the actual drift risk.)
    here = os.path.dirname(T.__file__)
    for mod in ("worker.py", "worker_pool.py"):
        src = open(os.path.join(here, mod)).read()
        assert 'f"az:w:' not in src and 'f"az:res:' not in src, f"{mod} re-spells a redis key (drift)"
    # and the parent orchestrator likewise routes keys through the transport (no key f-strings)
    par_src = open(os.path.join(here, "parallel.py")).read()
    assert 'f"az:w:' not in par_src and 'f"az:res:' not in par_src
    # the transport IS where the key f-strings live (the sole owner)
    t_src = open(T.__file__).read()
    assert 'f"az:w:' in t_src and 'f"az:res:' in t_src


def test_task_rng_fold_is_byte_for_byte_preserved():
    """The seed fold (invariant 8): kind tags gen=1_000_003 / eval=7_000_037, the np.uint64
    multipliers, the version+1 term — reproduced here independently and asserted equal to the live
    `Worker.task_rng`. A drift in any term breaks the parallel≈serial determinism contract.  The fold
    is now a Worker method over the pure `Worker._fold_seed` arithmetic (item L / R14 — promoted from
    the `_W` module global); the BODY is byte-for-byte the pre-promotion fold."""
    # build a Worker WITHOUT _worker_init (no env/redis/numba): only the seed fold is under test, and
    # __init__'s numpy-only guard passes in this jax-free test process.
    wk = W.Worker(env=None, fb=None, redis=None, base_seed=4242)

    def reference(version, kind, idx):
        kind_tag = {"gen": 1_000_003, "eval": 7_000_037}[kind]
        seed = (np.uint64(4242)
                ^ (np.uint64(version + 1) * np.uint64(2_654_435_761))
                ^ (np.uint64(kind_tag) * np.uint64(40_503))
                ^ (np.uint64(idx) * np.uint64(2_246_822_519)))
        return np.random.default_rng(int(seed)).integers(0, 2 ** 31, size=5).tolist()

    for kind in ("gen", "eval"):
        for version in (0, 1, 17, 1_000_000):
            for idx in (0, 3, 41):
                got = wk.task_rng(version, kind, idx).integers(0, 2 ** 31, size=5).tolist()
                assert got == reference(version, kind, idx), (kind, version, idx)


def test_phase_does_not_enter_the_rng_fold():
    """PHASE namespaces the weight KEY, NEVER the rng (R14 invariant).  `Worker.task_rng` /
    `_fold_seed` take (version, kind, idx) and have no phase parameter — so the SAME logical
    (version, kind, idx) draws the SAME stream regardless of which phase published the weights.  This
    is what preserves parallel≈serial bit-identity by construction when phase enters the key shape."""
    import inspect
    fold_params = list(inspect.signature(W.Worker._fold_seed).parameters)
    rng_params = list(inspect.signature(W.Worker.task_rng).parameters)
    assert "phase" not in fold_params, fold_params
    assert "phase" not in rng_params, rng_params
    # and operationally: the gen fold for (version, idx) is unchanged whether or not an eval phase
    # ever published — the fold has no phase input to perturb it.
    wk = W.Worker(env=None, fb=None, redis=None, base_seed=99)
    a = wk.task_rng(5, "gen", 2).integers(0, 2 ** 31, size=4).tolist()
    b = wk.task_rng(5, "gen", 2).integers(0, 2 ** 31, size=4).tolist()
    assert a == b


def test_task_specs_table_is_the_kind_authority():
    """The two work-kinds are DATA: `TASK_SPECS` carries the kind tag + the module-level callable, and
    `Worker._fold_seed` reads the tag from the table (one place, not two literals)."""
    assert W.TASK_SPECS["gen"].kind_tag == 1_000_003
    assert W.TASK_SPECS["eval"].kind_tag == 7_000_037
    assert W.TASK_SPECS["gen"].callable is W._gen_task
    assert W.TASK_SPECS["eval"].callable is W._eval_task
    # the callables are module-level (so the spawn pool resolves them by qualified name) and delegate
    # to the per-process Worker singleton (item L / R14)
    assert W._gen_task.__module__ == "chocofarm.az.worker"
    assert W._eval_task.__module__ == "chocofarm.az.worker"


def test_ttls_preserved():
    """The result TTL (`CHOCO_RESULT_TTL` default 3600 — aborted-iteration self-clean) and the weight
    TTL (3600 — old-version self-clean) band-aids survive the split (invariants 5, 6)."""
    os.environ.pop("CHOCO_RESULT_TTL", None)
    assert T._result_ttl() == 3600
    assert T._WEIGHT_TTL_S == 3600


def test_result_ttl_env_overridable():
    old = os.environ.get("CHOCO_RESULT_TTL")
    try:
        os.environ["CHOCO_RESULT_TTL"] = "120"
        assert T._result_ttl() == 120
    finally:
        if old is None:
            os.environ.pop("CHOCO_RESULT_TTL", None)
        else:
            os.environ["CHOCO_RESULT_TTL"] = old


def test_parallel_reexports_survive_split():
    """The public names the loop + the weights/registry docstrings reference on `parallel` survive
    the split (back-compat): the collaborators are re-exported, resolving to their new homes.  After
    the item-L / R14 `Worker` promotion the per-process state is the `Worker` object (re-exported),
    NOT the retired `_W` dict / `_ensure_net` / `_task_rng` module globals (those are now Worker
    methods)."""
    for name in ("ParallelExecutor", "pack_net", "unpack_net", "_connect", "_drain_imap",
                 "_worker_init", "_gen_task", "_eval_task", "Worker"):
        assert hasattr(P, name), name
    assert P._connect.__module__ == "chocofarm.az.transport"
    assert P._drain_imap.__module__ == "chocofarm.az.worker_pool"
    assert P._gen_task.__module__ == "chocofarm.az.worker"
    assert P.Worker.__module__ == "chocofarm.az.worker"
    # the retired module globals are GONE from worker.py (promoted to Worker state/methods)
    assert not hasattr(W, "_W")
    assert not hasattr(W, "_ensure_net")
    assert not hasattr(W, "_task_rng")
