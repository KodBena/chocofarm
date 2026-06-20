#!/usr/bin/env python3
"""
test_method_vegas.py — unit gate for the RTT-driven delay-based (TCP-Vegas style) congestion controller
(cpp/stage_a/control_lab/methods/vegas.py), a STATIC candidate for the issue-gate control lab.

Imports the method's OWN submodule directly (NOT the methods package, and WITHOUT load_all()): sibling
method files are authored in parallel, so importing only `control_lab.methods.vegas` keeps this test
isolated from a half-written neighbour. Pins the FROZEN adapter.Controller contract (reset / observe / act /
metrics shape) plus Vegas's defining behavior:

  - act() returns a length-T list of values in {0,1} (the per-thread allow bits);
  - observe() is a safe no-op (static: no reward learning, no gate perturbation);
  - after a thread establishes its zero-queue baseline RTT_min from a window of low samples, a forward whose
    rtt_us sits WELL ABOVE that baseline (queueing q > beta) is DENIED, while a thread reading AT its baseline
    (q <= alpha) is ALLOWED;
  - the RTT_min baseline is ROBUST to a one-time cold-compile high spike (a single huge sample does not lift
    the low-percentile floor enough to mask real queueing);
  - the liveness overrides hold: inflight==0 force-allows (a deny is a no-op there), and the un-warmed
    rtt_us==0 sentinel is never decided on (the thread holds its prior gate, defaulting to allow) and never
    poisons the baseline window.

Run pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_vegas.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys

# cpp/stage_a on sys.path so `control_lab.*` resolves both under pytest and as a bare script (mirrors the
# maintainer's PYTHONPATH=cpp/stage_a run convention for the lab).
_STAGE_A = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp", "stage_a"
)
if _STAGE_A not in sys.path:
    sys.path.insert(0, _STAGE_A)

from control_lab.adapter import Observation, TrialContext  # noqa: E402
from control_lab.methods import vegas as vg  # noqa: E402  (own submodule, NOT the package)


def _ctx(n_threads: int = 4) -> TrialContext:
    return TrialContext(
        n_threads=n_threads, d_ceiling=3, k_per_thread=8, s_min=2,
        chunk_floor=True, seed=0,
    )


def _obs(
    *, n_threads: int, rtt_us: list[float], inflight: list[int], served: list[int], t: float = 0.0
) -> Observation:
    """Minimal synthetic Observation with the length-T vectors the Vegas gate reads (rtt_us, inflight). The
    served set is informational here (the gate keys off the rtt_us==0 sentinel directly); other feature slots
    default-safe in act()."""
    features = {
        "n_threads": n_threads,
        "d_ceiling": 3,
        "server_rows_per_forward": 0.0,
        "inflight": inflight,
        "ready": [0] * n_threads,
        "msgs": [0] * n_threads,
        "leaves": [0] * n_threads,
        "rtt_us": rtt_us,
    }
    return Observation(features=features, served=served, forward_rows=0, t_monotonic=t)


def test_reset_and_metrics_shape():
    """reset() sizes per-run state to T and metrics() exposes the dashboard scalars (band + baselines)."""
    c = vg.VegasDelayGate(alpha=150.0, beta=600.0)
    c.reset(_ctx(n_threads=4))
    m = c.metrics()
    assert m["alpha"] == 150.0
    assert m["beta"] == 600.0
    assert "mean_rtt_min" in m
    assert "mean_queueing" in m
    # a fresh trial has no warmed samples yet -> baseline + queueing default to 0.
    assert m["mean_rtt_min"] == 0.0
    assert m["mean_queueing"] == 0.0


def test_bad_band_raises():
    """fail loud (ADR-0002): an inverted / degenerate band is a construction error, not a runtime surprise."""
    for bad in (dict(alpha=600.0, beta=150.0), dict(alpha=-1.0, beta=10.0), dict(alpha=10.0, beta=10.0)):
        raised = False
        try:
            vg.VegasDelayGate(**bad)
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for {bad}"
    # a degenerate window / quantile is likewise a build-time error.
    for bad in (dict(window=0), dict(quantile=1.5), dict(quantile=-0.1)):
        raised = False
        try:
            vg.VegasDelayGate(**bad)
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for {bad}"


def test_act_returns_length_t_binary():
    """act() returns a length-T list whose every entry is 0 or 1 (the per-thread allow bits)."""
    T = 4
    c = vg.VegasDelayGate()
    c.reset(_ctx(n_threads=T))
    out = c.act(_obs(n_threads=T, rtt_us=[100.0] * T, inflight=[1] * T, served=list(range(T))))
    assert isinstance(out, list)
    assert len(out) == T
    assert all(v in (0, 1) for v in out)


def test_observe_is_safe_noop():
    """observe() is a no-op for the static delay gate: it must not raise and must not perturb the gate."""
    T = 3
    c = vg.VegasDelayGate()
    c.reset(_ctx(n_threads=T))
    c.observe(123.4, {"forward_rows": 5})
    out = c.act(_obs(n_threads=T, rtt_us=[100.0] * T, inflight=[1] * T, served=[0, 1, 2]))
    assert all(v in (0, 1) for v in out)
    assert len(out) == 3


def test_unwarmed_sentinel_is_force_allowed_and_does_not_poison_baseline():
    """A thread reading rtt_us==0 (un-warmed/absent sentinel) is NEVER decided on (holds its prior allow) and
    its 0 is NEVER sampled into the RTT_min window — so the baseline stays empty/0 for that thread."""
    T = 2
    c = vg.VegasDelayGate(alpha=50.0, beta=200.0)
    c.reset(_ctx(n_threads=T))
    # thread 0 warmed at a huge RTT; thread 1 un-warmed (rtt_us==0). Thread 1 must allow regardless, and its
    # window must stay empty (baseline 0), proving the sentinel never entered the percentile.
    out = c.act(_obs(n_threads=T, rtt_us=[5000.0, 0.0], inflight=[2, 2], served=[0]))
    assert out[1] == 1, "un-warmed thread (rtt_us==0) holds its prior gate (allow), never decided on"
    assert c._rtt_win[1] == [], "the un-warmed sentinel 0 must not be sampled into the RTT_min window"


def test_inflight_zero_force_allows():
    """inflight==0 is an UNGATED forced flush: a deny is a no-op there, so the gate force-allows even a
    thread whose queueing would otherwise throttle it."""
    T = 1
    c = vg.VegasDelayGate(alpha=50.0, beta=200.0, window=8, quantile=0.0)
    c.reset(_ctx(n_threads=T))
    # establish a low baseline (~100us) over several warmed forwards WITH inflight, so q would be huge next.
    for _ in range(6):
        c.act(_obs(n_threads=T, rtt_us=[100.0], inflight=[2], served=[0]))
    # now a massive RTT but inflight==0 -> the liveness override force-allows despite q >> beta.
    out = c.act(_obs(n_threads=T, rtt_us=[9000.0], inflight=[0], served=[0]))
    assert out == [1], "inflight==0 force-allows regardless of the (huge) queueing estimate"


def test_vegas_denies_high_rtt_allows_at_baseline():
    """THE defining behavior. Establish a robust zero-queue baseline RTT_min from a window of low samples,
    then: a forward whose rtt_us sits WELL ABOVE the baseline (q > beta) is DENIED (throttle), while a thread
    reading AT its baseline (q <= alpha) is ALLOWED. The baseline is also robust to a single cold-compile
    high spike."""
    T = 2
    # band [alpha=50, beta=300] microseconds; a 10th-percentile baseline over a 32-sample window.
    c = vg.VegasDelayGate(alpha=50.0, beta=300.0, window=32, quantile=0.10)
    c.reset(_ctx(n_threads=T))

    # --- establish the baseline: feed BOTH threads many low (~100us) warmed samples, with one cold-compile
    # spike injected to prove the low-percentile floor ignores a high outlier (does NOT lift to mask queueing).
    base_rtt = 100.0
    for k in range(20):
        rtt0 = 5000.0 if k == 3 else base_rtt    # one-time cold-compile spike on thread 0
        c.act(_obs(n_threads=T, rtt_us=[rtt0, base_rtt], inflight=[2, 2], served=[0, 1]))

    # the robust baseline should have settled near the low ~100us floor on BOTH threads (the 5000us spike is a
    # high outlier the 10th percentile ignores), NOT been dragged up by the spike.
    assert c._rtt_min[0] < base_rtt * 2.0, "cold-compile spike must not lift the robust RTT_min baseline"
    assert c._rtt_min[1] < base_rtt * 2.0

    # --- the discriminating forward: thread 0 reads WELL ABOVE baseline (queueing q = 1000-100 = 900 > beta)
    # -> DENY; thread 1 reads AT baseline (q ~= 0 <= alpha) -> ALLOW. Both have inflight>0 so the band decides.
    out = c.act(_obs(n_threads=T, rtt_us=[base_rtt + 1000.0, base_rtt], inflight=[2, 2], served=[0, 1]))
    assert out[0] == 0, "thread 0: rtt_us well above its established RTT_min (q > beta) -> deny (throttle)"
    assert out[1] == 1, "thread 1: rtt_us at its established RTT_min (q <= alpha) -> allow"

    # metrics reflect a real (warmed) queueing estimate and the band.
    m = c.metrics()
    assert m["alpha"] == 50.0 and m["beta"] == 300.0
    assert m["mean_rtt_min"] > 0.0      # a baseline was established
    import math
    assert math.isfinite(m["mean_queueing"])


if __name__ == "__main__":
    # plain-runnable (no pytest needed), mirroring the repo's bare-script test convention.
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
    print("all vegas method checks passed")
