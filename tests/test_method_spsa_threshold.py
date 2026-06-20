#!/usr/bin/env python3
"""
tests/test_method_spsa_threshold.py — unit test for the spsa_threshold (online-learning) issue-gate.

Imports the method's OWN submodule directly (control_lab.methods.spsa_threshold), NOT the methods package and
NOT load_all() — so this test is isolated from sibling method files being authored in parallel (per the
methods/ package docstring: discovery is explicit, a single-submodule import pulls in no siblings).

Asserts the FROZEN adapter.Controller contract surface (reset / act shape+domain / observe safety) plus the
mechanism-specific online behavior: the submit-pressure threshold gate (allow iff x[t] >= theta) shared across
threads, the theta-box extremes (theta=0 == all-allow, a large theta == deny-until-forced), the inflight==0
forced-flush override, and — the LEARNING assertions — (a) the SPSA finite-difference step moves theta UPHILL
on a concave objective in the deterministic single-cycle case, and (b) closed-loop under a clear reward
gradient the running theta converges toward the rewarded threshold.

Run: PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_spsa_threshold.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Sequence

import pytest

from control_lab.adapter import Observation, TrialContext
from control_lab.methods.spsa_threshold import SpsaThresholdGate


def _ctx(n_threads: int = 4, d: int = 8, seed: int = 0) -> TrialContext:
    return TrialContext(
        n_threads=n_threads,
        d_ceiling=d,
        k_per_thread=10,
        s_min=4,
        chunk_floor=True,
        seed=seed,
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


# --- contract surface ---------------------------------------------------------------------------------------


def test_reset_and_act_shape_domain() -> None:
    """reset() works and act() returns a length-T list of {0,1} on a synthetic Observation."""
    t = 4
    c = SpsaThresholdGate()
    c.reset(_ctx(n_threads=t))
    out = c.act(_obs(ready=[2, 4, 6, 8], inflight=[1, 2, 3, 4]))
    _assert_gate(out, t)


def test_observe_is_safe() -> None:
    """observe() must never throw — including on a non-finite reward (ignored, not poisoning the estimate)."""
    c = SpsaThresholdGate()
    c.reset(_ctx())
    c.observe(12.0, {})
    c.observe(-3.0, {"anything": 7})
    c.observe(float("nan"), {})   # non-finite reward is dropped, not folded in
    c.observe(float("inf"), {})
    out = c.act(_obs(ready=[5, 5, 5, 5], inflight=[1, 1, 1, 1]))
    _assert_gate(out, 4)


def test_construction_validates_knobs() -> None:
    """fail loud (ADR-0002): degenerate gains / window / box raise at construction."""
    with pytest.raises(ValueError):
        SpsaThresholdGate(a=0.0)
    with pytest.raises(ValueError):
        SpsaThresholdGate(c=0.0)
    with pytest.raises(ValueError):
        SpsaThresholdGate(A=-1.0)
    with pytest.raises(ValueError):
        SpsaThresholdGate(alpha=0.0)
    with pytest.raises(ValueError):
        SpsaThresholdGate(alpha=1.5)
    with pytest.raises(ValueError):
        SpsaThresholdGate(gamma=0.0)
    with pytest.raises(ValueError):
        SpsaThresholdGate(hold_window=0)
    with pytest.raises(ValueError):
        SpsaThresholdGate(theta_min=2.0, theta_max=1.0)   # inverted box (theta_min > theta_max) is refused
    with pytest.raises(ValueError):
        SpsaThresholdGate(theta0=99.0, theta_min=0.0, theta_max=8.0)   # start outside the box
    # a ZERO-WIDTH box (theta_min == theta_max) is the legitimate frozen-theta degenerate, NOT an error.
    SpsaThresholdGate(theta0=1.0, theta_min=1.0, theta_max=1.0)


# --- mechanism: the submit-pressure threshold gate and its box extremes -------------------------------------


def test_zero_threshold_is_all_allow() -> None:
    """theta=0 makes x[t] >= 0 trivially true everywhere -> byte-identical to AllAllow.

    Pin a zero-width box at 0: theta0 == theta_min == theta_max == 0, so the perturbation clips to 0 and the
    applied threshold is exactly 0 — even a low-pressure frame with inflight>0 allows every thread."""
    c = SpsaThresholdGate(theta0=0.0, theta_min=0.0, theta_max=0.0, c=0.5)
    c.reset(_ctx(n_threads=3))
    out = c.act(_obs(ready=[0, 1, 2], inflight=[4, 4, 4]))  # tiny pressure, but theta~0 -> all allow
    assert out == [1, 1, 1], out


def test_large_threshold_denies_until_forced() -> None:
    """A large applied threshold denies everything except the inflight==0 forced-flush override.

    Pin theta high (a degenerate box at theta_max): with inflight>0 the whole gate is deny; the one thread at
    inflight==0 is force-allowed (DENY-ONLY semantics — the forced flush is UNGATED, a deny is a no-op there)."""
    big = 1e6
    c = SpsaThresholdGate(theta0=big, theta_min=big, theta_max=big, c=0.5)
    c.reset(_ctx(n_threads=3))
    out = c.act(_obs(ready=[9, 9, 9], inflight=[2, 0, 5]))  # huge backlog, but a huge theta denies the warmed
    assert out == [0, 1, 0], out


def test_threshold_gates_on_submit_pressure() -> None:
    """Mechanism: allow iff x[t] = ready[t]/max(1, D-inflight) >= theta, the SAME theta for all threads.

    Pin a degenerate box at theta=1.0, D=8. Construct three threads with known pressure:
      t0: ready=1, inflight=0 -> forced-allow regardless (inflight==0 liveness override);
      t1: ready=2, inflight=6 -> headroom max(1, 8-6)=2 -> x=1.0 >= 1.0 -> allow;
      t2: ready=1, inflight=4 -> headroom 4 -> x=0.25 < 1.0 -> deny."""
    c = SpsaThresholdGate(theta0=1.0, theta_min=1.0, theta_max=1.0, c=0.5)
    c.reset(_ctx(n_threads=3, d=8))
    out = c.act(_obs(ready=[1, 2, 1], inflight=[0, 6, 4]))
    assert out == [1, 1, 0], out


def test_saturated_thread_clamps_headroom_to_high_pressure() -> None:
    """At inflight >= D the headroom divisor clamps to 1, so x = ready — a large pressure -> allow under a
    finite theta. (The issue is a no-op under inflight<D, but the signal still says 'wants to fly'.)"""
    c = SpsaThresholdGate(theta0=1.0, theta_min=1.0, theta_max=1.0, c=0.5)
    c.reset(_ctx(n_threads=1, d=8))
    out = c.act(_obs(ready=[3], inflight=[8]))   # headroom max(1, 8-8)=1 -> x=3.0 >= 1.0 -> allow
    assert out == [1], out


# --- the SPSA state machine ---------------------------------------------------------------------------------


def test_phase_machine_advances_plus_minus_step() -> None:
    """observe() over 2*W forwards completes one SPSA cycle: PLUS window -> MINUS window -> step (k advances).

    With W small, the first W observes fill the PLUS window (phase stays PLUS until close), the next W fill the
    MINUS window and trigger the step. After 2*W observes the iteration index k has advanced from 0 to 1 and
    the phase is back to PLUS for the next cycle."""
    w = 3
    c = SpsaThresholdGate(hold_window=w)
    c.reset(_ctx(n_threads=2))
    assert c.metrics()["k"] == 0.0 and c.metrics()["phase"] == 0.0   # start: cycle 0, PLUS
    for _ in range(w):
        c.observe(1.0, {})
    assert c.metrics()["phase"] == 1.0, "after the PLUS window closes the machine measures MINUS"
    assert c.metrics()["k"] == 0.0, "no step yet — only PLUS has closed"
    for _ in range(w):
        c.observe(1.0, {})
    assert c.metrics()["k"] == 1.0, "after the MINUS window the SPSA step advances the iteration index"
    assert c.metrics()["phase"] == 0.0, "the next cycle reopens at PLUS"


# --- the LEARNING assertions --------------------------------------------------------------------------------


def test_single_cycle_step_ascends_uphill() -> None:
    """LEARNING (deterministic single cycle): the SPSA finite-difference step moves theta in the ASCENT
    direction of a known reward gradient.

    Construct a controller whose objective rises with theta (J increases in theta over the box), so the
    gradient is positive and the step must INCREASE theta. To make the one-cycle direction deterministic
    regardless of the drawn sign delta, we feed observe() a reward equal to the APPLIED theta read from
    metrics() — J(theta_applied) = theta_applied, strictly increasing. Then for delta=+1: J(theta+c) > J(theta-c)
    -> g>0 -> theta up; for delta=-1: theta+ = theta-c (lower J), theta- = theta+c (higher J), and the 1/delta
    factor flips the sign back, so g>0 -> theta up either way. We assert theta strictly increased after one
    full PLUS/MINUS/step cycle."""
    w = 2
    c = SpsaThresholdGate(a=0.5, c=0.5, A=0.0, hold_window=w, theta0=2.0, theta_min=0.0, theta_max=8.0)
    c.reset(_ctx(n_threads=2))
    theta_before = c.metrics()["theta"]
    # one full cycle: PLUS window (W forwards) then MINUS window (W forwards), each scored by the APPLIED theta.
    for _ in range(2 * w):
        applied = c.metrics()["theta_applied"]   # the perturbed theta the gate is using this forward
        c.observe(applied, {})                   # reward strictly increasing in theta -> positive gradient
        c.act(_obs(ready=[4, 4], inflight=[2, 2]))
    assert c.metrics()["k"] == 1.0, "exactly one SPSA iteration completed"
    assert c.metrics()["theta"] > theta_before, (
        f"on a reward strictly increasing in theta the SPSA step must ascend theta; "
        f"before={theta_before}, after={c.metrics()['theta']}, grad={c.metrics()['last_grad']}"
    )
    assert c.metrics()["last_grad"] > 0.0, "the finite-difference gradient estimate must be positive uphill"


def _run_spsa_with_concave_objective(
    c: SpsaThresholdGate,
    target: float,
    scale: float,
    n_forwards: int,
) -> None:
    """Drive the controller closed-loop for n_forwards against a concave objective J(theta) = -scale*(theta-
    target)^2 peaked at `target`: each forward read the APPLIED theta from metrics(), feed observe() its
    objective value (the per-forward reward this run assigns to that gate), then call act() once.

    A concave objective with a unique interior maximizer is the canonical SPSA convergence setting: the
    finite-difference gradient points toward `target`, so the running theta must march toward it. Reading the
    APPLIED (perturbed) theta is faithful to the real harness — the reward credits the gate actually in force."""
    obs = _obs(ready=[4, 4], inflight=[2, 2])
    for _ in range(n_forwards):
        applied = c.metrics()["theta_applied"]
        reward = -scale * (applied - target) ** 2
        c.observe(reward, {})
        c.act(obs)


def test_spsa_converges_toward_the_rewarded_threshold() -> None:
    """LEARNING (closed-loop stochastic): under a clear concave reward gradient peaked at a target threshold,
    SPSA moves the running theta a substantial distance TOWARD that target from a far start.

    Objective J(theta) = -(theta - 4.0)^2 is maximized at theta=4.0; we start theta at 0.5 (far below). After
    many SPSA iterations (each costs 2*W forwards) the running theta must close most of the gap to 4.0 — we
    assert it ends well above the start and within a tolerance of the target. SPSA is stochastic in the
    perturbation sign, so we assert a generous band, not bit-convergence; the seed is pinned for reproducibility."""
    target = 4.0
    c = SpsaThresholdGate(
        a=0.5, c=0.6, A=20.0, hold_window=4, theta0=0.5, theta_min=0.0, theta_max=8.0, seed=0
    )
    c.reset(_ctx(n_threads=2, seed=0))
    theta_start = c.metrics()["theta"]

    # plenty of iterations: ~250 SPSA steps at W=4 -> 2000 forwards, the run's order of magnitude.
    _run_spsa_with_concave_objective(c, target=target, scale=1.0, n_forwards=2000)

    theta_end = c.metrics()["theta"]
    assert theta_end > theta_start + 1.5, (
        f"SPSA should climb from theta0={theta_start} toward the optimum {target}; ended at {theta_end}"
    )
    assert abs(theta_end - target) < 1.5, (
        f"SPSA should converge near the rewarded threshold {target}; ended at {theta_end}"
    )
    assert c.metrics()["k"] > 0.0, "at least one SPSA iteration must have completed"


def test_spsa_descends_toward_a_lower_optimum() -> None:
    """LEARNING (the other direction): when the objective is peaked at a LOW threshold, SPSA moves theta DOWN.

    Mirror of the convergence test with the optimum BELOW the start: J(theta) = -(theta - 1.0)^2 peaked at 1.0,
    start theta at 6.0. The running theta must descend toward 1.0 — proving the gradient sign is honored in both
    directions, not just upward."""
    target = 1.0
    c = SpsaThresholdGate(
        a=0.5, c=0.6, A=20.0, hold_window=4, theta0=6.0, theta_min=0.0, theta_max=8.0, seed=1
    )
    c.reset(_ctx(n_threads=2, seed=1))
    theta_start = c.metrics()["theta"]

    _run_spsa_with_concave_objective(c, target=target, scale=1.0, n_forwards=2000)

    theta_end = c.metrics()["theta"]
    assert theta_end < theta_start - 1.5, (
        f"SPSA should descend from theta0={theta_start} toward the lower optimum {target}; ended at {theta_end}"
    )
    assert abs(theta_end - target) < 1.5, (
        f"SPSA should converge near the rewarded threshold {target}; ended at {theta_end}"
    )
