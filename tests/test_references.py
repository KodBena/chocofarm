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


class _FakeRedis:
    """A dict-backed stand-in for the registry redis (get→bytes, set→stores) so the cache's fetch/store/
    garbage arms are tested WITHOUT a live redis and without the expensive rate compute."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def get(self, k: str) -> "bytes | None":
        return self.store.get(k)

    def set(self, k: str, v: object) -> None:
        self.store[k] = v if isinstance(v, bytes) else str(v).encode()


def test_clairvoyant_cache_round_trips_and_is_loud_on_garbage():
    """The optimistic cache: a MISS computes + stores; a HIT returns the stored value WITHOUT recomputing;
    a present-but-malformed blob is a LOUD ValueError (never serve a garbage ceiling — ADR-0002/P2)."""
    import pytest

    from chocofarm import references as ref
    from chocofarm.model.env import Environment
    env = Environment()
    fake = _FakeRedis()
    orig_redis, orig_rate = ref._cache_redis, ref.clairvoyant_rate
    try:
        ref._cache_redis = lambda: fake                       # type: ignore[assignment]
        ref.clairvoyant_rate = lambda e: 0.1234               # type: ignore[assignment] # stub the brute force
        assert ref.cached_clairvoyant_rate(env) == 0.1234     # miss → compute(stub) + store
        assert fake.store                                     # it was stored

        def _boom(e: object) -> float:
            raise AssertionError("recomputed on a cache hit")
        ref.clairvoyant_rate = _boom                          # type: ignore[assignment]
        assert ref.cached_clairvoyant_rate(env) == 0.1234     # hit → stored value, no recompute

        key = ref._CLAIRVOYANT_KEY_PREFIX + ref._env_clairvoyant_fingerprint(env)
        fake.store[key] = b"not-a-float"
        with pytest.raises(ValueError):
            ref.cached_clairvoyant_rate(env)
    finally:
        ref._cache_redis, ref.clairvoyant_rate = orig_redis, orig_rate   # type: ignore[assignment]


def test_clairvoyant_cache_degrades_to_compute_when_redis_down():
    """A redis OUTAGE (no client) degrades to a DIRECT compute — the cache is optimistic, it never fails
    the run for being unable to reach redis."""
    from chocofarm import references as ref
    from chocofarm.model.env import Environment
    orig_redis, orig_rate = ref._cache_redis, ref.clairvoyant_rate
    try:
        ref._cache_redis = lambda: None                       # type: ignore[assignment]
        ref.clairvoyant_rate = lambda e: 0.5                  # type: ignore[assignment]
        assert ref.cached_clairvoyant_rate(Environment()) == 0.5
    finally:
        ref._cache_redis, ref.clairvoyant_rate = orig_redis, orig_rate   # type: ignore[assignment]


if __name__ == "__main__":
    # plain-runnable (no pytest needed) — mirrors test_smoke.py's bare-script path.
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all reference-isolation checks passed")
