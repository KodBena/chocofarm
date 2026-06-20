#!/usr/bin/env python3
"""
tests/test_method_bang_bang.py — unit test for the bang_bang (s,S) hysteresis issue-gate controller.

Imports the method's OWN submodule directly (control_lab.methods.bang_bang), NOT the methods package and
NOT load_all() — so this test is isolated from sibling method files being authored in parallel (per the
methods/ package docstring: discovery is explicit, a single-submodule import pulls in no siblings).

Asserts the FROZEN adapter.Controller contract surface (reset / act shape+domain / observe safety) plus the
mechanism-specific hysteresis behavior: low-backlog deny, high-backlog allow, deadband stickiness, the
inflight==0 forced-flush override, and the hold-timeout forced allow.

Run: PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_bang_bang.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Sequence

from control_lab.adapter import Observation, TrialContext
from control_lab.methods.bang_bang import BangBangHysteresisGate


def _ctx(n_threads: int = 4, k: int = 10) -> TrialContext:
    return TrialContext(
        n_threads=n_threads,
        d_ceiling=8,
        k_per_thread=k,
        s_min=4,
        chunk_floor=True,
        seed=0,
    )


def _obs(ready: Sequence[float], inflight: Sequence[float], t_monotonic: float = 0.0) -> Observation:
    n = len(ready)
    return Observation(
        features={
            "n_threads": n,
            "d_ceiling": 8,
            "server_rows_per_forward": n,
            "ready": list(ready),
            "inflight": list(inflight),
            "msgs": [0] * n,
            "leaves": [0] * n,
            "rtt_us": [100.0] * n,
        },
        served=list(range(n)),
        forward_rows=n,
        t_monotonic=t_monotonic,
    )


def _assert_gate(out: Sequence[int], t: int) -> None:
    assert isinstance(out, list)
    assert len(out) == t
    assert all(v in (0, 1) for v in out), f"gate must be binary, got {out}"


def test_reset_and_act_shape_domain() -> None:
    """reset() works and act() returns a length-T list of {0,1} on a synthetic Observation."""
    t, k = 4, 10
    c = BangBangHysteresisGate()
    c.reset(_ctx(n_threads=t, k=k))
    # mid-deadband ready (r=0.4 for K=10) with inflight present -> a valid binary gate (here: stay allow).
    out = c.act(_obs(ready=[4, 4, 4, 4], inflight=[2, 2, 2, 2]))
    _assert_gate(out, t)


def test_observe_is_safe() -> None:
    """observe() is a no-op for the static family and must not throw."""
    c = BangBangHysteresisGate()
    c.reset(_ctx())
    c.observe(1.5, {})
    c.observe(-0.25, {"anything": 7})
    out = c.act(_obs(ready=[5, 5, 5, 5], inflight=[1, 1, 1, 1]))
    _assert_gate(out, 4)


def test_low_backlog_denies_high_backlog_allows() -> None:
    """Mechanism: r<=r_lo -> deny (hold to accumulate); r>=r_hi -> allow (let the fat batch fly).

    K=10, r_lo=0.25, r_hi=0.60. ready=1 -> r=0.1<=r_lo -> deny; ready=8 -> r=0.8>=r_hi -> allow. inflight>0
    so neither liveness override fires."""
    c = BangBangHysteresisGate(r_lo=0.25, r_hi=0.60)
    c.reset(_ctx(n_threads=2, k=10))
    out = c.act(_obs(ready=[1, 8], inflight=[3, 3]))
    assert out == [0, 1], out
    assert c.metrics()["denied"] == 1.0


def test_deadband_holds_prior_state() -> None:
    """Mechanism: inside (r_lo, r_hi) the per-thread state is sticky (the chatter-killing hysteresis).

    Start=allow. Drive thread 0 to deny with a low r, then feed an in-band r (0.4): thread 0 STAYS denied,
    while thread 1 (driven high, then in-band) STAYS allowed. inflight kept >0 so overrides don't intrude."""
    c = BangBangHysteresisGate(r_lo=0.25, r_hi=0.60)
    c.reset(_ctx(n_threads=2, k=10))
    first = c.act(_obs(ready=[1, 8], inflight=[3, 3]))   # thread0 -> deny, thread1 -> allow
    assert first == [0, 1], first
    # now both in the deadband (r=0.4): each holds its prior state.
    second = c.act(_obs(ready=[4, 4], inflight=[3, 3]))
    assert second == [0, 1], second


def test_inflight_zero_forces_allow() -> None:
    """Liveness override (i): inflight==0 is an UNGATED forced flush, so a deny is overridden to allow.

    ready=1 (r=0.1<=r_lo) would deny, but inflight==0 forces allow regardless."""
    c = BangBangHysteresisGate(r_lo=0.25, r_hi=0.60)
    c.reset(_ctx(n_threads=2, k=10))
    out = c.act(_obs(ready=[1, 1], inflight=[0, 0]))
    assert out == [1, 1], out


def test_hold_timeout_forces_allow() -> None:
    """Liveness override (ii): after > H consecutive denies a thread is force-allowed once.

    H=3, persistent low backlog (r=0.1) with inflight>0 so only the timeout can break the hold. Decisions
    1..3 deny (streak 1,2,3 — not yet > H); decision 4 fires the timeout (streak would be 4 > 3) -> allow;
    decision 5 denies again (streak reset to 1)."""
    c = BangBangHysteresisGate(r_lo=0.25, r_hi=0.60, hold_timeout=3)
    c.reset(_ctx(n_threads=1, k=10))
    seq = [c.act(_obs(ready=[1], inflight=[5]))[0] for _ in range(5)]
    assert seq == [0, 0, 0, 1, 0], seq


def test_metrics_reports_thresholds() -> None:
    """metrics() exposes r_lo, r_hi, and a denied count for the dashboard."""
    c = BangBangHysteresisGate(r_lo=0.3, r_hi=0.7)
    c.reset(_ctx())
    m = c.metrics()
    assert m["r_lo"] == 0.3
    assert m["r_hi"] == 0.7
    assert "denied" in m
