#!/usr/bin/env python3
"""
tests/test_mypy_strict.py — the ENFORCED mypy --strict gate (ADR-0011 mechanism; typing rollout
Stage 0/1/2).

Runs `mypy` (which reads the `[tool.mypy]` config in pyproject.toml — global `strict` + the four
documented stub-gap overrides) on the SET OF MODULES that are fully `--strict`-clean, and asserts
ZERO errors. This converts "typed signatures" from review-only prose into a CI-enforced contract
(the assessment §8 / ADR-0012 P8 draft). Later stages EXTEND `STRICT_CLEAN` as they type more
modules — keep the list explicit so the next stage just appends.

How the gate isolates the set
-----------------------------
mypy is invoked with `--follow-imports=silent`: imported modules are followed for their TYPES (so a
strict-clean module is checked against the real types of what it imports, not `Any`), but errors in
modules OUTSIDE this list are suppressed. So the assertion is "these listed modules are themselves
strict-clean," not "the whole tree is" — exactly the monotonic Stage-1 core the rollout enforces
first (assessment §5, Stage 0: "enforcing the Stage-1 set first," not gating the whole tree red).

What is in / out
----------------
IN (100% --strict-clean now):
  * Stage 1 core: config.py, az/dtypes.py, model/instance.py, hp/schema.py, every `__init__.py`.
  * Stage 2 — the env↔Policy SEAM (the keystone): model/env.py + solvers/base.py, with the seam
    aliases (Loc / MoveAction / Action / WorldSet / Collected) introduced in env.py and imported by
    every downstream solver/feature/bound. model/facemodel.py joins too — env delegates its detector
    dynamics to facemodel.SenseAction (filter/observe/informative), so the seam is only strict-clean
    once that single face-carrier is typed. With env.py typed, references.py and hp/registry.py lose
    their `no-untyped-call` into the seam and are now strict-clean too (the assessment's "vanishes as
    callees are annotated", §2). registry.py also needed two non-seam residuals fixed under ADR-0004
    minimal-touch (the redis-8 py.typed kwargs, the reflective facet-walk's group-union) — signatures
    only, no body rewrite.

  * Stage 3 — the medium bulk, now gated (the env↔Policy seam aliases are in place, so each
    landed as its callees did): model/arrangement.py; all of solvers/ (uct/ismcts/nmcs/decomp,
    base already above); all of bounds/ (vhats/vhats_decomp/vhats_exact/eval_bound/info_relaxation);
    the strict-clean az leaves (actions/dataset/features/transport/worker_pool); and the
    strict-clean eval/ scripts (eval_decomp/eval_faces/eval_ismcts/eval_nmcs/eval_uct + the
    harness/report/tb_runner reporting layer). `solvers/__init__.py`'s `SOLVERS` registry was typed
    `dict[str, type[Policy]]` (was the bare dict's `type[ABCMeta]`) so `SOLVERS[name](**kw)` is a
    `Policy`, not `Any` — the registry contract made honest at its one home (P8/P1). The
    `dinkelbach_rate` heterogeneous-dict field narrowing lives once in `eval/harness.dink_float`.

OUT, deliberately:
  * az/optimizer.py / az/mlp_jax_train.py — HARD modules (jax/optax seam, Stage 4). Only their
    standalone `AdamHParams` contract was made honest in Stage 1; the module bodies are not clean.
  * The 5 jax/numba kernel-boundary az modules, HELD for maintainer review of the boundary crossing
    (Stage 4): az/mlp.py, az/exit_loop.py, az/train_value.py, az/worker.py, az/gumbel_search.py.
    Each fails strict on a `no-untyped-call` into the kernel seam (forward_core / kernels.warmup /
    JaxTrainer / MlpJaxForward). They are fully annotated but NOT gated — the maintainer wants to
    contemplate the jax/numba boundary before committing the contract.
  * The az/eval modules that HARD-LOAD one of those 5 boundary modules at import time, even though
    they currently pass strict (mlp.py/worker.py are annotated, so the call into them is typed, not
    a no-untyped-call): az/feature_response.py, az/netvalue_ismcts.py, az/value_target.py
    (top-level `ValueMLP` import + call), az/parallel.py (top-level `import worker`), and
    eval/eval_az.py (transitively, via NetValueISMCTS). They are kept out so the gate's enforced
    contract does not reach across the deferred boundary; az/transport.py and az/worker_pool.py, by
    contrast, touch the held-out modules only under TYPE_CHECKING / lazily and so ARE gated.
  * the remaining hard az modules (az/kernels.py numba, az/forward.py, az/mlp_jax.py) — Stage 4.

Skips gracefully (does not fail) if mypy is not importable, mirroring how `tests/test_cpp_runner.py`
skips without its binary.

Public Domain (The Unlicense).
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The enforced strict-clean SET. Extend this list as later stages type more modules.
STRICT_CLEAN = [
    # --- Stage 1 core (easy_strict) ---
    "chocofarm/config.py",
    "chocofarm/az/dtypes.py",
    "chocofarm/model/instance.py",
    "chocofarm/hp/schema.py",
    "chocofarm/__init__.py",
    "chocofarm/az/__init__.py",
    "chocofarm/az/bench/__init__.py",
    "chocofarm/bounds/__init__.py",
    "chocofarm/eval/__init__.py",
    "chocofarm/hp/__init__.py",
    "chocofarm/model/__init__.py",
    "chocofarm/solvers/__init__.py",
    # --- Stage 2 — the env↔Policy seam (the keystone) + the downstream that depended on it ---
    "chocofarm/model/env.py",        # the seam: Environment + the Loc/Action/WorldSet aliases
    "chocofarm/model/facemodel.py",  # the SenseAction the env's detector dynamics delegate to
    "chocofarm/solvers/base.py",     # the Policy ABC + Policy.decide contract (every solver's seam)
    "chocofarm/references.py",       # was blocked only by no-untyped-call into the seam
    "chocofarm/hp/registry.py",      # likewise (+ two ADR-0004 minimal-touch non-seam residuals)
    # --- Stage 3 — medium bulk (solvers / bounds / az leaves / arrangement / eval) ---
    # Now that 3a (az leaves) and 3b (solvers/bounds) are both on main, their cross-subtree
    # no-untyped-calls resolve and each medium module that does NOT cross the deferred jax/numba
    # boundary enters the gate. Membership was determined EMPIRICALLY (mypy --strict, zero errors).
    "chocofarm/model/arrangement.py",   # the missed medium module decomp.py needs typed (via env)
    # solvers/ (base.py already above)
    "chocofarm/solvers/uct.py",
    "chocofarm/solvers/ismcts.py",
    "chocofarm/solvers/nmcs.py",
    "chocofarm/solvers/decomp.py",      # unblocked once arrangement.py was typed
    # bounds/
    "chocofarm/bounds/vhats.py",
    "chocofarm/bounds/vhats_decomp.py",
    "chocofarm/bounds/vhats_exact.py",
    "chocofarm/bounds/eval_bound.py",
    "chocofarm/bounds/info_relaxation.py",
    # az/ leaves that do NOT hard-load a held-out jax/numba boundary module (the 5 that do are OUT —
    # see the docstring): the in-set touches mlp/worker only under TYPE_CHECKING or lazily, if at all.
    "chocofarm/az/actions.py",
    "chocofarm/az/dataset.py",
    "chocofarm/az/features.py",         # imports az/kernels (numba, config-ignored), not a held-out
    "chocofarm/az/transport.py",        # ValueMLP only under TYPE_CHECKING / lazy
    "chocofarm/az/worker_pool.py",      # az/worker imported only lazily inside __init__
    # eval/ — the classical (non-AZ) eval wave; eval_az.py is OUT (hard-loads mlp via NetValueISMCTS)
    "chocofarm/eval/harness.py",        # owns dink_float, the dinkelbach_rate field-narrowing SSOT
    "chocofarm/eval/report.py",
    "chocofarm/eval/eval_uct.py",
    "chocofarm/eval/eval_ismcts.py",
    "chocofarm/eval/eval_nmcs.py",
    "chocofarm/eval/eval_faces.py",
    "chocofarm/eval/eval_decomp.py",
    "chocofarm/eval/tb_runner.py",
]


def _mypy_importable():
    try:
        import mypy  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _mypy_importable(),
                    reason="mypy not importable in this interpreter (pip install mypy)")
def test_strict_clean_modules_have_zero_mypy_errors():
    """The Stage-1 strict-clean SET passes `mypy --strict` (the pyproject config) with ZERO errors.
    The enforced gate (ADR-0011): a regression in any listed module's annotations fails CI here."""
    # --follow-imports=silent: check the listed modules against the real types of their imports,
    # but report errors ONLY for the listed modules (the Stage-1 isolation — see the module docstring).
    cmd = [sys.executable, "-m", "mypy", "--follow-imports=silent", *STRICT_CLEAN]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    combined = out.stdout + out.stderr
    assert out.returncode == 0, (
        "mypy --strict reported errors on the Stage-1 strict-clean set (the gate). "
        f"Fix the annotation regression below:\n{combined}")
    assert "Success" in combined or "no issues found" in combined, combined
