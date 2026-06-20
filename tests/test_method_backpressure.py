#!/usr/bin/env python3
"""
tests/test_method_backpressure.py — unit test for the backpressure (Lyapunov drift-plus-penalty) online
issue-gate controller.

Imports the method's OWN submodule directly (control_lab.methods.backpressure), NOT the methods package and
NOT load_all() — so this test is isolated from sibling method files being authored in parallel (per the
methods/ package docstring: discovery is explicit, a single-submodule import pulls in no siblings).

Asserts the FROZEN adapter.Controller contract surface (reset / act shape+domain / observe safety) plus the
mechanism-specific backpressure behavior: the MaxWeight allow/deny differential V*mu_hat vs the queue term
q[t], the inflight==0 forced-flush override, the per-served-thread reward normalization, and the ONLINE
LEARNING assertion — a clear reward gradient drives mu_hat the right way, flipping a denied thread to allowed.

Run: PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_backpressure.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from control_lab.adapter import Observation, TrialContext
from control_lab.methods.backpressure import BackpressureGate


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


def _info(n_served: int) -> Mapping[str, Any]:
    """The harness rides the served-tid count on the observe() info mapping (the un-sentineled cardinality)."""
    return {"n_served": n_served}


def _assert_gate(out: Sequence[int], t: int) -> None:
    assert isinstance(out, list)
    assert len(out) == t
    assert all(v in (0, 1) for v in out), f"gate must be binary, got {out}"


def test_reset_and_act_shape_domain() -> None:
    """reset() works and act() returns a length-T list of {0,1} on a synthetic Observation."""
    t, k = 4, 10
    c = BackpressureGate()
    c.reset(_ctx(n_threads=t, k=k))
    out = c.act(_obs(ready=[4, 4, 4, 4], inflight=[2, 2, 2, 2]))
    _assert_gate(out, t)


def test_observe_is_safe() -> None:
    """observe() must not throw — including a missing/garbage served count and an empty info mapping."""
    c = BackpressureGate()
    c.reset(_ctx())
    c.observe(1.5, {})                       # no served count: falls back to the raw reward, no throw
    c.observe(-0.25, {"anything": 7})        # irrelevant info key
    c.observe(3.0, {"n_served": "bad"})      # un-castable served count: falls back, no throw
    c.observe(2.0, _info(4))                 # the normal path
    out = c.act(_obs(ready=[5, 5, 5, 5], inflight=[1, 1, 1, 1]))
    _assert_gate(out, 4)


def test_inflight_zero_forces_allow() -> None:
    """Liveness override: inflight==0 is an UNGATED forced flush, so a would-be deny is overridden to allow.

    A large queue with mu_hat==0 (serve_gain=0 < q) would deny every thread, but inflight==0 forces allow."""
    c = BackpressureGate(v=4.0)
    c.reset(_ctx(n_threads=2, k=10))
    out = c.act(_obs(ready=[9, 9], inflight=[0, 0]))
    assert out == [1, 1], out


def test_maxweight_differential_denies_long_queue_at_zero_throughput() -> None:
    """Mechanism: with mu_hat unseeded (==0) the serve-gain V*mu_hat is 0, so any positive queue DENIES
    (hold to accumulate) while a zero queue ALLOWS (nothing to hold). inflight>0 so no liveness override."""
    c = BackpressureGate(v=4.0)
    c.reset(_ctx(n_threads=2, k=10))
    out = c.act(_obs(ready=[5, 0], inflight=[3, 3]))
    assert out == [0, 1], out          # thread0 (q=0.5) denied; thread1 (q=0) allowed
    assert c.metrics()["n_allowed"] == 1.0


def test_reward_normalized_per_served_thread() -> None:
    """The reward is normalized to a per-served-thread RATE: reward=4 over 4 served threads yields the same
    mu_hat as reward=1 over 1 served thread (so V trades one thread's backlog against one thread's
    throughput, never the whole-forward aggregate)."""
    a = BackpressureGate(v=4.0, beta=0.5)
    a.reset(_ctx(n_threads=4, k=10))
    a.observe(4.0, _info(4))           # rate = 4/4 = 1.0
    b = BackpressureGate(v=4.0, beta=0.5)
    b.reset(_ctx(n_threads=1, k=10))
    b.observe(1.0, _info(1))           # rate = 1/1 = 1.0
    assert a.metrics()["mu_hat"] == b.metrics()["mu_hat"] == 1.0


def test_online_learning_reward_gradient_flips_gate() -> None:
    """ONLINE LEARNING (the method-specific assertion): a clear reward gradient moves the learned mu_hat the
    right way and flips a denied thread to allowed.

    Setup T=1, K=10, V=4, beta=0.5 -> norm=10, queue q = ready/10. Fix ready=5 (q=0.5), inflight=3 (>0 so no
    liveness override). The allow condition is V*mu_hat >= q, i.e. 4*mu_hat >= 0.5, i.e. mu_hat >= 0.125.

    LOW-throughput regime: observe(reward=0) repeatedly drives mu_hat -> 0, so 4*0 = 0 < 0.5 -> DENY.
    HIGH-throughput regime: observe(reward=1, n_served=1) repeatedly drives mu_hat -> 1.0 (and STRICTLY UP
    each step), so 4*1.0 = 4.0 >= 0.5 -> ALLOW. The learner admitting more aggressively as throughput
    evidence accrues is exactly the backpressure / drift-plus-penalty behavior (larger realized throughput
    => the V-weighted term wins => serve)."""
    c = BackpressureGate(v=4.0, beta=0.5)
    c.reset(_ctx(n_threads=1, k=10))

    held = _obs(ready=[5], inflight=[3])     # q = 0.5; the same queue throughout, only mu_hat changes

    # LOW regime: zero throughput -> mu_hat collapses to ~0 -> the long queue is held (deny).
    for _ in range(8):
        c.observe(0.0, _info(1))
    assert c.act(held) == [0], "at zero learned throughput a backlogged thread must be held (deny)"
    assert c.metrics()["mu_hat"] < 0.125, c.metrics()["mu_hat"]

    # HIGH regime: feed a clear positive throughput gradient; mu_hat must rise monotonically toward the rate.
    prev = c.metrics()["mu_hat"]
    rose_every_step = True
    for _ in range(8):
        c.observe(1.0, _info(1))             # rate = 1.0 each step
        now = c.metrics()["mu_hat"]
        rose_every_step = rose_every_step and (now > prev)
        prev = now
    assert rose_every_step, "mu_hat must increase at every step under a strictly higher reward"
    assert c.metrics()["mu_hat"] >= 0.125, c.metrics()["mu_hat"]
    # the learned state now crosses the serve threshold -> the SAME backlogged thread flips to allow.
    assert c.act(held) == [1], "after learning high throughput the gate must admit the backlogged thread"


def test_metrics_exposes_learned_state() -> None:
    """metrics() exposes the learned mu_hat, the V knob, the EWMA window (the reward-hold W), and tallies."""
    c = BackpressureGate(v=3.0, beta=0.25)
    c.reset(_ctx())
    m = c.metrics()
    assert m["V"] == 3.0
    assert m["ewma_window"] == 4.0           # 1 / beta
    assert "mu_hat" in m
    assert "mean_q" in m
    assert "n_allowed" in m
    assert "denied" in m


def test_construction_rejects_bad_knobs() -> None:
    """fail loud (ADR-0002): a non-positive V or a degenerate EWMA step is a construction error, not a run
    surprise."""
    import pytest

    with pytest.raises(ValueError):
        BackpressureGate(v=0.0)
    with pytest.raises(ValueError):
        BackpressureGate(v=-1.0)
    with pytest.raises(ValueError):
        BackpressureGate(beta=0.0)
    with pytest.raises(ValueError):
        BackpressureGate(beta=1.5)
