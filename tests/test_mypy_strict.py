#!/usr/bin/env python3
"""
tests/test_mypy_strict.py — the ENFORCED mypy --strict gate (ADR-0011 mechanism; typing rollout
Stage 0/1).

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
IN (100% --strict-clean now): config.py, az/dtypes.py, model/instance.py, hp/schema.py, and every
`__init__.py`. These have NO unresolved cross-module dependency on the still-untyped env↔Policy seam.

OUT, deliberately, though annotated this stage:
  * references.py / hp/registry.py — their OWN signatures are fully annotated (honest + complete),
    but they call into the still-untyped env.py seam (`env.d`/`env.exit_cost`/`Environment()` →
    `no-untyped-call`). That is the Stage-2 downstream backlog the assessment predicts "vanishes as
    callees are annotated" (§2). Including them now would require suppressing `no-untyped-call`, a
    HIDDEN relaxation — refused (ADR-0002). They join `STRICT_CLEAN` when Stage 2 types env.py.
  * az/optimizer.py / az/mlp_jax_train.py — HARD modules (jax/optax seam, Stage 4). Only their
    standalone `AdamHParams` contract was made honest this stage; the module bodies are not clean.

Skips gracefully (does not fail) if mypy is not importable, mirroring how `tests/test_cpp_runner.py`
skips without its binary.

Public Domain (The Unlicense).
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The enforced strict-clean SET (Stage 1 core). Extend this list as later stages type more modules.
STRICT_CLEAN = [
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
