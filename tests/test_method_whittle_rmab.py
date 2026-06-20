#!/usr/bin/env python3
"""
test_method_whittle_rmab.py — unit gate for the Whittle-index restless-bandit controller
(cpp/stage_a/control_lab/methods/whittle_rmab.py), a STATIC candidate for the issue-gate control lab.

Imports the method's OWN submodule directly (NOT the methods package, and WITHOUT load_all()): sibling
method files are authored in parallel, so importing only `control_lab.methods.whittle_rmab` keeps this test
isolated from a half-written neighbour. Pins the FROZEN adapter.Controller contract (reset / observe / act /
metrics shape) plus the index gate's defining behavior:

  - act() returns a length-T list of values in {0,1};
  - observe() is a safe no-op (static — nothing learned);
  - the CROSS-THREAD COORDINATION (the mechanism the per-thread methods lack): with an activation budget of
    ceil(p*T) < T and every arm carrying positive backlog + headroom, only the HIGHEST-index arms are
    allowed and the lowest are denied — the gate ranks arms against each other and spends a shared budget;
  - the headroom factor: a SATURATED arm (inflight == D) prices to index 0 and is denied (when it still has
    work in flight), while a zero-backlog arm (ready == 0) also prices to 0;
  - the liveness override: inflight == 0 force-allows (a deny is a no-op there).

Run pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_whittle_rmab.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys

import pytest

# cpp/stage_a on sys.path so `control_lab.*` resolves both under pytest and as a bare script (mirrors the
# maintainer's PYTHONPATH=cpp/stage_a run convention for the lab).
_STAGE_A = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp", "stage_a"
)
if _STAGE_A not in sys.path:
    sys.path.insert(0, _STAGE_A)

from control_lab.adapter import Observation, TrialContext  # noqa: E402
from control_lab.methods import whittle_rmab as wr  # noqa: E402  (own submodule, NOT the package)


def _ctx(n_threads: int = 4, d: int = 4, k: int = 8) -> TrialContext:
    return TrialContext(
        n_threads=n_threads, d_ceiling=d, k_per_thread=k, s_min=2,
        chunk_floor=True, seed=0,
    )


def _obs(*, n_threads: int, inflight: list[int], ready: list[int], served: list[int] | None = None,
         t: float = 0.0) -> Observation:
    """Minimal synthetic Observation carrying the two length-T gauges the index reads (inflight, ready);
    the other feature slots are default-safe in act()."""
    if served is None:
        served = list(range(n_threads))
    features = {
        "n_threads": n_threads,
        "d_ceiling": 4,
        "server_rows_per_forward": float(sum(ready)),
        "inflight": inflight,
        "ready": ready,
        "msgs": [0] * n_threads,
        "leaves": [0] * n_threads,
        "rtt_us": [0] * n_threads,
    }
    return Observation(features=features, served=served, forward_rows=sum(ready), t_monotonic=t)


def test_reset_and_metrics_shape():
    """reset() sizes per-run state to T and metrics() exposes the dashboard scalars (index mean/max, n_active)."""
    c = wr.WhittleIndexGate(p=0.6)
    c.reset(_ctx(n_threads=4))
    m = c.metrics()
    for key in ("index_mean", "index_max", "n_active"):
        assert key in m
    # before any act(), the last-index vector is all zeros.
    assert m["index_mean"] == 0.0


def test_act_returns_length_t_binary():
    """act() returns a length-T list whose every entry is 0 or 1 (the per-thread allow bits)."""
    T = 4
    c = wr.WhittleIndexGate(p=0.6)
    c.reset(_ctx(n_threads=T))
    out = c.act(_obs(n_threads=T, inflight=[1] * T, ready=[3] * T))
    assert isinstance(out, list)
    assert len(out) == T
    assert all(v in (0, 1) for v in out)


def test_observe_is_safe_noop():
    """observe() is a no-op for the static index gate: it must not raise and must not perturb the gate."""
    T = 3
    c = wr.WhittleIndexGate(p=0.6)
    c.reset(_ctx(n_threads=T))
    before = c.act(_obs(n_threads=T, inflight=[1] * T, ready=[2] * T))
    c.observe(99.0, {"forward_rows": 7})
    after = c.act(_obs(n_threads=T, inflight=[1] * T, ready=[2] * T))
    assert before == after  # observe() leaves the (memoryless) gate identical


def test_cross_thread_coordination_top_p():
    """The defining behavior: the COORDINATED activation budget the per-thread methods lack. T=4, p=0.5 ->
    budget=2. All four arms have positive backlog AND headroom (so a per-thread rule would allow all four),
    but the index ranks them by ready backlog (inflight equal): the two HIGHEST-ready arms are allowed and
    the two lowest are denied. The shared budget is spent on the most valuable arms."""
    T = 4
    c = wr.WhittleIndexGate(p=0.5)
    c.reset(_ctx(n_threads=T, d=4, k=8))
    # equal inflight=1 (equal headroom); ready strictly increasing -> index strictly increasing in t.
    out = c.act(_obs(n_threads=T, inflight=[1, 1, 1, 1], ready=[1, 2, 3, 4]))
    assert sum(out) == 2, "exactly ceil(0.5*4)=2 arms activated (the shared budget)"
    assert out == [0, 0, 1, 1], "the two highest-index (highest-ready) arms win the budget; the lowest lose"


def test_headroom_saturation_and_zero_backlog_deny():
    """The index factors. A SATURATED arm (inflight == D) has zero headroom -> index 0 -> denied (it still
    has work in flight, so the liveness override does not fire). A zero-backlog arm (ready == 0) also prices
    to 0. With theta=0.0... use a positive theta so a 0-index arm is below threshold and denied."""
    T = 3
    c = wr.WhittleIndexGate(p=None, theta=0.01)  # threshold mode: allow iff index >= theta
    c.reset(_ctx(n_threads=T, d=4, k=8))
    # arm 0: saturated (inflight==D==4) with backlog -> headroom 0 -> index 0 -> deny.
    # arm 1: zero backlog (ready==0) with headroom -> index 0 -> deny.
    # arm 2: healthy (room + backlog) -> positive index -> allow.
    out = c.act(_obs(n_threads=T, inflight=[4, 1, 1], ready=[5, 0, 5]))
    assert out[0] == 0, "saturated arm (inflight==D) has zero headroom -> index 0 -> denied"
    assert out[1] == 0, "zero-backlog arm (ready==0) has index 0 -> denied"
    assert out[2] == 1, "healthy arm (room + backlog) prices above theta -> allowed"


def test_liveness_override_forces_allow_at_zero_inflight():
    """inflight == 0 force-allows regardless of the index (a deny is a no-op there — the flush is ungated).
    Use threshold mode so the low-index arm would otherwise be denied."""
    T = 3
    c = wr.WhittleIndexGate(p=None, theta=0.5)  # a high threshold so a small-backlog arm is denied on index
    c.reset(_ctx(n_threads=T, d=4, k=8))
    # arm 0: tiny backlog + has work in flight -> index below theta -> would deny.
    # arm 1: SAME tiny backlog but inflight==0 -> liveness override force-allows.
    # arm 2: large backlog -> index above theta -> allow on its own merit.
    out = c.act(_obs(n_threads=T, inflight=[1, 0, 1], ready=[1, 1, 8]))
    assert out[0] == 0, "small-backlog arm with work in flight stays denied (index below theta)"
    assert out[1] == 1, "small-backlog arm with inflight==0 is force-allowed (liveness override)"
    assert out[2] == 1, "large-backlog arm clears theta on its own"


def test_invalid_config_fails_loud():
    """ADR-0002: a degenerate selection config is a construction error (both rules, neither rule, or a p out
    of (0,1]) — raised at the ctor, not a per-forward surprise."""
    with pytest.raises(ValueError):
        wr.WhittleIndexGate(p=0.6, theta=0.5)   # both rules
    with pytest.raises(ValueError):
        wr.WhittleIndexGate(p=None, theta=None)  # neither rule
    with pytest.raises(ValueError):
        wr.WhittleIndexGate(p=1.5)               # p out of range


if __name__ == "__main__":
    # plain-runnable (no pytest needed), mirroring the repo's bare-script test convention. pytest.raises is
    # used above, so a bare run still imports pytest (present in the lab venv).
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
    print("all whittle_rmab method checks passed")
