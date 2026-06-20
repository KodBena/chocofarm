#!/usr/bin/env python3
"""
tests/test_method_threshold_bandit.py — unit test for the threshold_bandit (online-learning) issue-gate.

Imports the method's OWN submodule directly (control_lab.methods.threshold_bandit), NOT the methods package
and NOT load_all() — so this test is isolated from sibling method files being authored in parallel (per the
methods/ package docstring: discovery is explicit, a single-submodule import pulls in no siblings).

Asserts the FROZEN adapter.Controller contract surface (reset / act shape+domain / observe safety) plus the
mechanism-specific online behavior: the submit-pressure threshold gate (allow iff x[t] >= theta), the two
arm-set extremes (theta=0 == all-allow, theta=+inf == deny-until-forced), the inflight==0 forced-flush
override, and — the LEARNING assertion — that under a clear reward gradient the discounted-UCB bandit converges
its learned theta onto the rewarded arm.

Run: PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_threshold_bandit.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Sequence

import pytest

from control_lab.adapter import Observation, TrialContext
from control_lab.methods.threshold_bandit import ThresholdBanditGate


def _ctx(n_threads: int = 4, d: int = 8) -> TrialContext:
    return TrialContext(
        n_threads=n_threads,
        d_ceiling=d,
        k_per_thread=10,
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


# --- contract surface ---------------------------------------------------------------------------------------


def test_reset_and_act_shape_domain() -> None:
    """reset() works and act() returns a length-T list of {0,1} on a synthetic Observation."""
    t = 4
    c = ThresholdBanditGate()
    c.reset(_ctx(n_threads=t))
    out = c.act(_obs(ready=[2, 4, 6, 8], inflight=[1, 2, 3, 4]))
    _assert_gate(out, t)


def test_observe_is_safe() -> None:
    """observe() must never throw — including on a non-finite reward (ignored, not poisoning the learner)."""
    c = ThresholdBanditGate()
    c.reset(_ctx())
    c.observe(12.0, {})
    c.observe(-3.0, {"anything": 7})
    c.observe(float("nan"), {})   # non-finite reward is dropped, not folded in
    c.observe(float("inf"), {})
    out = c.act(_obs(ready=[5, 5, 5, 5], inflight=[1, 1, 1, 1]))
    _assert_gate(out, 4)


def test_construction_validates_knobs() -> None:
    """fail loud (ADR-0002): degenerate arm set / exploration / window / discount raise at construction."""
    with pytest.raises(ValueError):
        ThresholdBanditGate(arms=())
    with pytest.raises(ValueError):
        ThresholdBanditGate(arms=(0.0, -1.0))
    with pytest.raises(ValueError):
        ThresholdBanditGate(arms=(0.0, float("nan")))
    with pytest.raises(ValueError):
        ThresholdBanditGate(c=-0.1)
    with pytest.raises(ValueError):
        ThresholdBanditGate(hold_window=0)
    with pytest.raises(ValueError):
        ThresholdBanditGate(gamma=0.0)
    with pytest.raises(ValueError):
        ThresholdBanditGate(gamma=1.5)


# --- mechanism: the submit-pressure threshold gate and its arm-set extremes ---------------------------------


def test_zero_threshold_arm_is_all_allow() -> None:
    """The theta=0 arm makes x[t] >= 0 trivially true everywhere -> byte-identical to AllAllow.

    Force the current arm to theta=0 (the first/most-permissive arm at reset is the smallest threshold = 0 in
    the default set) and check a low-pressure frame with inflight>0 still allows every thread."""
    c = ThresholdBanditGate()                 # default arms start with 0.0
    c.reset(_ctx(n_threads=3))
    assert c.metrics()["theta"] == 0.0        # reset starts on the smallest threshold
    out = c.act(_obs(ready=[0, 1, 2], inflight=[4, 4, 4]))  # tiny pressure, but theta=0 -> all allow
    assert out == [1, 1, 1], out


def test_inf_threshold_arm_denies_until_forced() -> None:
    """The theta=+inf arm denies everything except the inflight==0 forced-flush override.

    A single-arm bandit pinned at +inf: with inflight>0 the whole gate is deny; the one thread at inflight==0
    is force-allowed (DENY-ONLY semantics — the forced flush is UNGATED, a deny is a no-op there)."""
    c = ThresholdBanditGate(arms=(float("inf"),))   # the only arm is deny-until-forced
    c.reset(_ctx(n_threads=3))
    out = c.act(_obs(ready=[9, 9, 9], inflight=[2, 0, 5]))  # huge backlog, but +inf denies the warmed ones
    assert out == [0, 1, 0], out


def test_threshold_gates_on_submit_pressure() -> None:
    """Mechanism: allow iff x[t] = ready[t]/max(1, D-inflight) >= theta, the same theta for all threads.

    Pin a single arm at theta=1.0, D=8. Construct three threads with known pressure:
      t0: ready=1, inflight=0 -> forced-allow regardless (inflight==0 liveness override);
      t1: ready=2, inflight=6 -> headroom max(1, 8-6)=2 -> x=1.0 >= 1.0 -> allow;
      t2: ready=1, inflight=4 -> headroom 4 -> x=0.25 < 1.0 -> deny."""
    c = ThresholdBanditGate(arms=(1.0,))
    c.reset(_ctx(n_threads=3, d=8))
    out = c.act(_obs(ready=[1, 2, 1], inflight=[0, 6, 4]))
    assert out == [1, 1, 0], out


def test_saturated_thread_clamps_headroom_to_high_pressure() -> None:
    """At inflight >= D the headroom divisor clamps to 1, so x = ready — a large pressure -> allow under a
    finite theta. (The issue is a no-op under inflight<D, but the signal still says 'wants to fly'.)"""
    c = ThresholdBanditGate(arms=(1.0,))
    c.reset(_ctx(n_threads=1, d=8))
    out = c.act(_obs(ready=[3], inflight=[8]))   # headroom max(1, 8-8)=1 -> x=3.0 >= 1.0 -> allow
    assert out == [1], out


# --- the LEARNING assertion ---------------------------------------------------------------------------------


def _run_bandit_with_reward_table(
    c: ThresholdBanditGate,
    arms: Sequence[float],
    reward_for_theta: dict[float, float],
    n_forwards: int,
) -> None:
    """Drive the bandit closed-loop for n_forwards: each forward, read the CURRENTLY-held theta from metrics(),
    feed observe() the reward this run assigns to that theta (the arm's true value), then call act() once.

    This faithfully exercises the observe()->window-close->D-UCB-select machinery: the reward fed at each
    forward is the value of whichever arm the learner currently holds, so a clear gradient (one arm rich, the
    rest poor) is exactly the per-forward throughput signal the harness would supply for that gate."""
    obs = _obs(ready=[4, 4], inflight=[2, 2])
    for _ in range(n_forwards):
        theta = c.metrics()["theta"]
        c.observe(reward_for_theta[theta], {})
        c.act(obs)


def test_bandit_converges_to_the_rewarded_arm() -> None:
    """LEARNING: under a clear reward gradient the discounted-UCB bandit moves its learned theta to the arm
    with the highest reward.

    Three arms {0.0, 0.5, 2.0}. The reward table makes theta=0.5 the unique winner (10.0) and the others poor
    (1.0). With a short hold window the learner first pulls each arm once (D-UCB optimism: an unpulled arm has
    +inf index), then exploits — after enough epochs the held theta (and the best-mean arm) is 0.5. We also
    assert the bandit's own bookkeeping moved the right way: the winning arm's discounted mean dominates."""
    arms = (0.0, 0.5, 2.0)
    reward = {0.0: 1.0, 0.5: 10.0, 2.0: 1.0}
    c = ThresholdBanditGate(arms=arms, c=0.3, hold_window=4, gamma=0.97)
    c.reset(_ctx(n_threads=2))

    # plenty of forwards for D-UCB to pull each of 3 arms (W=4 each) then exploit for many epochs.
    _run_bandit_with_reward_table(c, arms, reward, n_forwards=400)

    m = c.metrics()
    assert m["theta"] == 0.5, f"bandit should converge to the rewarded threshold 0.5, got theta={m['theta']}"
    # the winning arm's learned discounted mean must dominate the losing arms (the credit went the right way).
    win = m["arm1_mean"]   # arm index 1 is theta=0.5
    assert win > m["arm0_mean"], (win, m["arm0_mean"])
    assert win > m["arm2_mean"], (win, m["arm2_mean"])
    assert m["epochs"] > 0


def test_bandit_reselects_after_regime_shift() -> None:
    """LEARNING (non-stationarity): the DISCOUNTED bandit abandons a once-best arm when a later regime makes a
    different arm best — a stationary bandit would stay stuck on the early winner.

    Phase 1 rewards theta=0.0; phase 2 (longer, so the gamma-discounted Phase-1 evidence decays) rewards
    theta=2.0. The converged theta after phase 2 must be 2.0, proving the discount re-opened the dominated arm."""
    arms = (0.0, 0.5, 2.0)
    c = ThresholdBanditGate(arms=arms, c=0.3, hold_window=4, gamma=0.9)
    c.reset(_ctx(n_threads=2))

    # phase 1: theta=0.0 is rich.
    _run_bandit_with_reward_table(c, arms, {0.0: 10.0, 0.5: 1.0, 2.0: 1.0}, n_forwards=120)
    assert c.metrics()["theta"] == 0.0, c.metrics()["theta"]

    # phase 2: the regime flips — theta=2.0 is now rich. The aggressive discount (gamma=0.9) fades the stale
    # phase-1 credit so the bandit re-selects the new winner.
    _run_bandit_with_reward_table(c, arms, {0.0: 1.0, 0.5: 1.0, 2.0: 10.0}, n_forwards=400)
    assert c.metrics()["theta"] == 2.0, f"discounted bandit should follow the regime shift, got {c.metrics()['theta']}"
