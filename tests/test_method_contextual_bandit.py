#!/usr/bin/env python3
"""
test_method_contextual_bandit.py — unit gate for the homogeneous contextual-bandit controller
(cpp/stage_a/control_lab/methods/contextual_bandit.py), an ONLINE candidate for the issue-gate control lab.

Imports the method's OWN submodule directly (NOT the methods package, and WITHOUT load_all()): sibling
method files are authored in parallel, so importing only `control_lab.methods.contextual_bandit` keeps this
test isolated from a half-written neighbour. Pins the FROZEN adapter.Controller contract (reset / observe /
act / metrics shape) plus the bandit's defining ONLINE-LEARNING behavior:

  - act() returns a length-T list of values in {0,1};
  - observe() is safe (it must not raise; a non-finite reward is dropped; a reward before the first choice is
    ignored — nothing to credit);
  - the cold model reproduces the all-allow baseline (theta==0, equal posterior width -> tie -> allow);
  - the served-thread first-difference for the coalescence feature honors the wire subtlety (an ABSENT thread
    is never differenced; its baseline is untouched);
  - THE LEARNING ASSERTION (method-specific): under a CLOSED-LOOP reward (the realized pool reward is a
    monotone function of the gate's own decision — the lab's actual semantics, and the only way a reward can
    distinguish one arm from the other), a reward that pays MORE the more threads DENY drives the bandit off
    its all-allow cold start to learn deny (the shared model's allow-value falls below baseline, the UCB
    bonus on the unsampled deny arm lets it explore, and the high deny reward reinforces it) — while a reward
    that pays more the more threads ALLOW keeps the policy at all-allow. The learned weights move OFF their
    zero initialization (proof the closed-form ridge update fired).

Run pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_contextual_bandit.py -q

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
from control_lab.methods import contextual_bandit as cb  # noqa: E402  (own submodule, NOT the package)


def _ctx(n_threads: int = 4, d: int = 4, k: int = 8) -> TrialContext:
    return TrialContext(
        n_threads=n_threads, d_ceiling=d, k_per_thread=k, s_min=2,
        chunk_floor=True, seed=0,
    )


def _obs(
    *,
    n_threads: int,
    inflight: list[int],
    ready: list[int],
    msgs: list[int] | None = None,
    leaves: list[int] | None = None,
    served: list[int] | None = None,
    t: float = 0.0,
) -> Observation:
    """Minimal synthetic Observation carrying the length-T gauges the context reads (inflight, ready, and the
    cumulative msgs/leaves the coalescence feature first-differences); other slots are default-safe in act()."""
    if served is None:
        served = list(range(n_threads))
    if msgs is None:
        msgs = [0] * n_threads
    if leaves is None:
        leaves = [0] * n_threads
    features = {
        "n_threads": n_threads,
        "d_ceiling": 4,
        "server_rows_per_forward": float(sum(ready)),
        "inflight": inflight,
        "ready": ready,
        "msgs": msgs,
        "leaves": leaves,
        "rtt_us": [0] * n_threads,
    }
    return Observation(features=features, served=served, forward_rows=sum(ready), t_monotonic=t)


def _drive_closed_loop(c, obs, reward_of_decision, n_forwards: int) -> list[int]:
    """Drive the public observe/act loop for n_forwards, mirroring the harness interleave (observe the PREVIOUS
    act's reward, then act) with a CLOSED-LOOP reward: the pool reward fed at forward i is a function of the
    decision the controller actually emitted at forward i-1 — exactly the lab's semantics (the realized
    throughput depends on the gate). This gives a genuine per-arm gradient (the open-loop schedule cannot,
    since a reward independent of the action distinguishes no arm). Returns the last emitted decision.
    The first forward has no previous act, so observe is skipped there (as the harness's first epoch does)."""
    prev: list[int] | None = None
    last: list[int] = []
    for i in range(n_forwards):
        if i > 0 and prev is not None:
            c.observe(reward_of_decision(prev), {})
        last = list(c.act(obs))
        prev = last
    return last


# ----------------------------------------------------------------------------- contract


def test_reset_and_metrics_shape():
    """reset() sizes per-run state to T and metrics() exposes the learned-state dashboard scalars."""
    c = cb.ContextualBanditGate()
    c.reset(_ctx(n_threads=4))
    m = c.metrics()
    for key in ("w_norm_deny", "w_norm_allow", "exploration_alpha", "reward_baseline", "updates", "allow_frac"):
        assert key in m
    # a fresh model has zero learned weights and zero updates.
    assert m["w_norm_deny"] == 0.0
    assert m["w_norm_allow"] == 0.0
    assert m["updates"] == 0.0


def test_act_returns_length_t_binary():
    """act() returns a length-T list whose every entry is 0 or 1 (the per-thread allow bits)."""
    T = 4
    c = cb.ContextualBanditGate()
    c.reset(_ctx(n_threads=T))
    out = c.act(_obs(n_threads=T, inflight=[1] * T, ready=[3] * T))
    assert isinstance(out, list)
    assert len(out) == T
    assert all(v in (0, 1) for v in out)


def test_cold_start_is_all_allow():
    """The cold model (theta==0, identical per-arm posterior width) ties on every thread and breaks toward
    ALLOW, reproducing the all-allow baseline before any reward is learned."""
    T = 5
    c = cb.ContextualBanditGate()
    c.reset(_ctx(n_threads=T))
    out = c.act(_obs(n_threads=T, inflight=[2] * T, ready=[3, 0, 7, 1, 4]))
    assert out == [1] * T


def test_observe_is_safe():
    """observe() never raises: a reward before the first choice is ignored (nothing to credit), and a
    non-finite reward is dropped rather than poisoning the model."""
    T = 3
    c = cb.ContextualBanditGate()
    c.reset(_ctx(n_threads=T))
    c.observe(123.4, {})                 # before any act(): no held choice -> ignored, must not raise.
    assert c.metrics()["updates"] == 0.0
    c.act(_obs(n_threads=T, inflight=[1] * T, ready=[2] * T))
    c.observe(float("nan"), {})          # non-finite -> dropped, must not raise.
    c.observe(float("inf"), {})
    out = c.act(_obs(n_threads=T, inflight=[1] * T, ready=[2] * T))
    assert len(out) == T and all(v in (0, 1) for v in out)


def test_invalid_config_fails_loud():
    """ADR-0002: degenerate hyperparameters / an unknown context feature are CONSTRUCTION errors, raised at
    the ctor, never a per-forward surprise."""
    with pytest.raises(ValueError):
        cb.ContextualBanditGate(ridge_lambda=0.0)        # A would be singular
    with pytest.raises(ValueError):
        cb.ContextualBanditGate(alpha=-1.0)              # negative exploration scale
    with pytest.raises(ValueError):
        cb.ContextualBanditGate(gamma=1.5)               # forgetting out of (0, 1]
    with pytest.raises(ValueError):
        cb.ContextualBanditGate(window=0)                # hold window must be >= 1
    with pytest.raises(ValueError):
        cb.ContextualBanditGate(context=["not_a_feature"])  # unknown context column (ADR-0008: no fuzzy match)


# ----------------------------------------------------------------------------- wire subtlety


def test_absent_thread_is_not_first_differenced():
    """The wire subtlety: a thread ABSENT from a forward reads a sentinel-0 cumulative counter, so it must NOT
    be first-differenced (its 0 would manufacture a spurious delta) and its baseline must NOT advance. Seed a
    thread's msgs/leaves baseline, then exclude it from `served` with the wire's sentinel-0 readings; its
    baseline must be unchanged, while a served thread's baseline tracks its true reading."""
    T = 2
    c = cb.ContextualBanditGate(window=1)
    c.reset(_ctx(n_threads=T))
    # forward 1: both served, seed baselines at their true cumulative counts.
    c.act(_obs(n_threads=T, inflight=[1, 1], ready=[1, 1], msgs=[10, 20], leaves=[40, 80], served=[0, 1]))
    assert int(c._msgs_prev[0]) == 10 and int(c._msgs_prev[1]) == 20
    # forward 2: only thread 0 served; the WIRE reports thread 1 as the sentinel 0 in the length-T lists.
    c.act(_obs(n_threads=T, inflight=[1, 1], ready=[1, 1], msgs=[15, 0], leaves=[60, 0], served=[0]))
    assert int(c._msgs_prev[0]) == 15, "served thread's baseline tracks its true cumulative reading"
    assert int(c._msgs_prev[1]) == 20, "absent thread's baseline is untouched (the sentinel-0 is never differenced)"
    assert int(c._leaves_prev[1]) == 80, "absent thread's leaves baseline likewise untouched"


# ----------------------------------------------------------------------------- the learning assertion


def test_learns_the_higher_reward_arm_closed_loop():
    """ONLINE LEARNING (the defining behavior), under a CLOSED-LOOP reward (the realized throughput depends on
    the gate — the lab's actual semantics). The pool reward is a monotone function of how many threads the
    gate DENIES, so one arm genuinely dominates and the shared model has a real gradient to learn.

    Reward-favors-DENY arm: the cold bandit holds ALLOW (0 denies -> the LOW reward); the centered target for
    allow goes negative, allow's learned value falls, and the exploration bonus on the unsampled deny arm lets
    the policy try deny; deny then collects the HIGH reward and is reinforced. The learned policy must move
    toward MORE deny than the all-allow cold start (sum < T on a neutral context, inflight>0 so no liveness
    override masks it), and the learned weights must be off zero (the ridge update fired).

    Reward-favors-ALLOW control: the same machinery, reward decreasing in denies, must keep the policy at
    all-allow (allow is both the cold start AND the rewarded arm — no reason to ever explore deny)."""
    T = 4
    # inflight strictly positive so the liveness override never masks the learned policy; a steady,
    # non-saturating context so phi is a fixed positive vector across windows (the gradient is in the reward).
    state = _obs(n_threads=T, inflight=[1] * T, ready=[4] * T)

    # --- reward-favors-DENY: the more threads deny, the higher the realized pool reward ---
    def reward_favors_deny(decision: list[int]) -> float:
        n_deny = sum(1 for v in decision if v == 0)
        return 10.0 + 30.0 * n_deny    # all-allow -> 10 (low); all-deny -> 130 (high)

    deny_learner = cb.ContextualBanditGate(window=6, alpha=1.0, ridge_lambda=1.0, gamma=1.0)
    deny_learner.reset(_ctx(n_threads=T))
    out_deny = _drive_closed_loop(deny_learner, state, reward_favors_deny, n_forwards=6 * 30)

    assert deny_learner.metrics()["updates"] > 0.0, "the ridge model updated at least once (learning fired)"
    nonzero = deny_learner.metrics()["w_norm_deny"] > 0.0 or deny_learner.metrics()["w_norm_allow"] > 0.0
    assert nonzero, "the shared model's learned weights moved off their zero initialization"
    # inflight is positive, so every 0 in the output is a LEARNED deny (not the liveness override). The bandit
    # started all-allow (sum==T); with deny the better arm it must have learned to deny at least one thread.
    assert sum(out_deny) < T, "reward favors deny -> the bandit learned to deny at least one thread"

    # --- reward-favors-ALLOW control: the more threads ALLOW, the higher the reward -> stay all-allow ---
    def reward_favors_allow(decision: list[int]) -> float:
        n_allow = sum(1 for v in decision if v == 1)
        return 10.0 + 30.0 * n_allow    # all-deny -> 10 (low); all-allow -> 130 (high)

    allow_learner = cb.ContextualBanditGate(window=6, alpha=1.0, ridge_lambda=1.0, gamma=1.0)
    allow_learner.reset(_ctx(n_threads=T))
    out_allow = _drive_closed_loop(allow_learner, state, reward_favors_allow, n_forwards=6 * 30)
    assert sum(out_allow) == T, "reward favors allow (== the cold start) -> the policy stays all-allow"


if __name__ == "__main__":
    # plain-runnable (no pytest needed for the non-raises checks), mirroring the repo's bare-script convention.
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn) and _name != "test_invalid_config_fails_loud":
            _fn()
            print(f"PASS {_name}")
    print("all contextual_bandit method checks passed (run via pytest for the fail-loud config test)")
