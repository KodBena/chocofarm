#!/usr/bin/env python3
"""
test_references.py — the neutral-module isolation gate (roadmap item F).

`chocofarm/references.py` is the NEUTRAL home for the env-derived %VoI reference
lines (floor / ceiling / anchor + the `BeliefRefs` SSOT), moved out of the eval
harness so `az` (training) can depend on them WITHOUT reaching backwards into
`eval` (a consumer of training). These checks pin that:

  - importing `chocofarm.references` does NOT pull in `chocofarm.eval` or
    `chocofarm.az` (the cycle the move exists to break), verified in a fresh
    subprocess so prior imports cannot mask a residual dependency;
  - the names stay importable from BOTH `chocofarm.references` and (via the
    back-compat re-export) `chocofarm.eval.harness`, and are the SAME objects.

Run pinned + bounded, e.g.:
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_references.py -q
"""
import os
import subprocess
import sys

# Repo root on sys.path (the maintainer's run convention; mirrors test_smoke.py)
# so the package resolves both under pytest and as a bare script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_references_does_not_import_eval_or_az():
    """A fresh interpreter that imports only chocofarm.references must not have
    pulled chocofarm.eval or chocofarm.az into sys.modules — references is a
    foundation, not a consumer of either."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import sys; import chocofarm.references; "
        "leaked = [m for m in sys.modules "
        "if m == 'chocofarm.eval' or m.startswith('chocofarm.eval.') "
        "or m == 'chocofarm.az' or m.startswith('chocofarm.az.')]; "
        "assert not leaked, leaked; print('OK')"
    )
    env = dict(os.environ, PYTHONPATH=repo_root)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=repo_root,
    )
    assert out.returncode == 0, f"stdout={out.stdout!r} stderr={out.stderr!r}"
    assert out.stdout.strip().endswith("OK"), out.stdout


def test_names_importable_from_both_and_identical():
    """The re-export is a genuine alias, not a copy: harness re-exports the SAME
    objects references defines, so existing `eval.harness` importers and the new
    `references` importers see one canonical definition (no drift)."""
    from chocofarm import references as ref
    from chocofarm.eval import harness as harn

    assert harn.BeliefRefs is ref.BeliefRefs
    assert harn.realizable_static is ref.realizable_static
    assert harn.clairvoyant_rate is ref.clairvoyant_rate
    assert harn.DECOMP_ANCHOR == ref.DECOMP_ANCHOR == 0.0941


if __name__ == "__main__":
    # plain-runnable (no pytest needed) — mirrors test_smoke.py's bare-script path.
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all reference-isolation checks passed")
