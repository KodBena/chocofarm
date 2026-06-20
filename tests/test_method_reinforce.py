#!/usr/bin/env python3
"""
test_method_reinforce.py — unit gate for the REINFORCE policy-gradient controller
(cpp/stage_a/control_lab/methods/reinforce.py), a REINFORCEMENT-LEARNING candidate for the issue-gate control lab.

Imports the method's OWN submodule directly (NOT the methods package, and WITHOUT load_all()): sibling method
files are authored in parallel, so importing only `control_lab.methods.reinforce` keeps this test isolated from
a half-written neighbour. Pins the FROZEN adapter.Controller contract (reset / observe / act / metrics shape)
plus REINFORCE's defining RL behavior:

  - act() returns a length-T list of values in {0,1} (the per-thread allow bits, a Bernoulli sample);
  - observe() is safe (it must not raise; a reward before the first sampled act is ignored — nothing to credit;
    a non-finite reward is dropped, never poisoning the gradient);
  - reset() cold-starts (re-initializes the policy + adam moment state; the buffer/baseline/pending clear);
  - the served-thread first-difference for the coalescence feature honors the wire subtlety (an ABSENT thread
    is never differenced; its baseline is untouched);
  - THE LEARNING ASSERTION (method-specific): under a CLOSED-LOOP reward (the realized pool reward is a monotone
    function of the gate's own decision — the lab's actual semantics, the only way a reward can distinguish a
    policy direction), a reward that pays MORE the more threads DENY drives the policy-gradient to LOWER the
    shared allow probability (the policy moves toward deny), while a reward that pays more the more threads
    ALLOW keeps the allow probability high. The two regimes separate, the optax step fired (updates>0, a
    non-zero gradient norm was seen), and the deny regime moved the policy OFF its allow-leaning cold start.

Run pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_reinforce.py -q

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
from control_lab.methods import reinforce as rf  # noqa: E402  (own submodule, NOT the package)


def _ctx(n_threads: int = 4, d: int = 4, k: int = 8, seed: int = 0) -> TrialContext:
    return TrialContext(
        n_threads=n_threads, d_ceiling=d, k_per_thread=k, s_min=2,
        chunk_floor=True, seed=seed,
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
    """Minimal synthetic Observation carrying the length-T gauges the features read (inflight, ready, and the
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


def _drive_closed_loop(c, obs, reward_of_decision, n_forwards: int) -> None:
    """Drive the public observe/act loop for n_forwards, mirroring the harness interleave (observe the PREVIOUS
    act's reward, then act) with a CLOSED-LOOP reward: the pool reward fed at forward i is a function of the
    decision the controller actually emitted at forward i-1 — exactly the lab's semantics (the realized
    throughput depends on the gate). This gives a genuine policy gradient (an open-loop reward independent of
    the action distinguishes no direction). The first forward has no previous act, so observe is skipped there
    (as the harness's first epoch does)."""
    prev: list[int] | None = None
    for i in range(n_forwards):
        if i > 0 and prev is not None:
            c.observe(reward_of_decision(prev), {})
        prev = list(c.act(obs))


# ----------------------------------------------------------------------------- contract


def test_reset_and_metrics_shape():
    """reset() sizes per-run state to T and metrics() exposes the learned-state dashboard scalars."""
    c = rf.ReinforceGate()
    c.reset(_ctx(n_threads=4))
    m = c.metrics()
    for key in ("mean_allow_prob", "grad_norm", "baseline", "updates", "buffer"):
        assert key in m
    # a fresh learner has taken no updates and emptied its buffer/baseline.
    assert m["updates"] == 0.0
    assert m["buffer"] == 0.0
    assert m["baseline"] == 0.0
    # cold-start allow probability is the allow-leaning init (sigmoid(init_allow_logit) > 0.5).
    assert m["mean_allow_prob"] > 0.5


def test_act_returns_length_t_binary():
    """act() returns a length-T list whose every entry is 0 or 1 (the per-thread allow bits, a Bernoulli sample)."""
    T = 4
    c = rf.ReinforceGate()
    c.reset(_ctx(n_threads=T))
    out = c.act(_obs(n_threads=T, inflight=[1] * T, ready=[3] * T))
    assert isinstance(out, list)
    assert len(out) == T
    assert all(v in (0, 1) for v in out)


def test_liveness_override_forces_allow_at_zero_inflight():
    """DENY-ONLY semantics: a thread with inflight==0 is an UNGATED forced flush, so the gate must force-allow
    it regardless of what the policy sampled. With every thread at inflight==0, the gate is all-allow."""
    T = 5
    c = rf.ReinforceGate()
    c.reset(_ctx(n_threads=T))
    out = c.act(_obs(n_threads=T, inflight=[0] * T, ready=[2, 0, 5, 1, 3]))
    assert out == [1] * T


def test_observe_is_safe():
    """observe() never raises: a reward before the first sampled act is ignored (nothing to credit), and a
    non-finite reward is dropped rather than poisoning the gradient."""
    T = 3
    c = rf.ReinforceGate()
    c.reset(_ctx(n_threads=T))
    c.observe(123.4, {})                 # before any act(): no pending transition -> ignored, must not raise.
    assert c.metrics()["updates"] == 0.0
    assert c.metrics()["buffer"] == 0.0
    c.act(_obs(n_threads=T, inflight=[1] * T, ready=[2] * T))
    c.observe(float("nan"), {})          # non-finite -> the pending transition is dropped, must not raise.
    c.observe(float("inf"), {})          # nothing pending now -> ignored, must not raise.
    out = c.act(_obs(n_threads=T, inflight=[1] * T, ready=[2] * T))
    assert len(out) == T and all(v in (0, 1) for v in out)
    assert c.metrics()["buffer"] == 0.0  # the nan-dropped transition never entered the buffer.


def test_invalid_config_fails_loud():
    """ADR-0002: degenerate hyperparameters are CONSTRUCTION errors, raised at the ctor, never a per-forward
    surprise."""
    with pytest.raises(ValueError):
        rf.ReinforceGate(lr=0.0)                 # non-positive learning rate
    with pytest.raises(ValueError):
        rf.ReinforceGate(update_period=0)        # batch period must be >= 1
    with pytest.raises(ValueError):
        rf.ReinforceGate(hidden=-1)              # hidden width must be >= 0
    with pytest.raises(ValueError):
        rf.ReinforceGate(init_allow_logit=float("inf"))  # non-finite cold-start logit
    with pytest.raises(ValueError):
        rf.ReinforceGate(max_batch=0)            # buffer cap must be >= 1


# ----------------------------------------------------------------------------- wire subtlety


def test_absent_thread_is_not_first_differenced():
    """The wire subtlety: a thread ABSENT from a forward reads a sentinel-0 cumulative counter, so it must NOT
    be first-differenced (its 0 would manufacture a spurious delta) and its baseline must NOT advance. Seed a
    thread's msgs/leaves baseline, then exclude it from `served` with the wire's sentinel-0 readings; its
    baseline must be unchanged, while a served thread's baseline tracks its true reading."""
    T = 2
    c = rf.ReinforceGate()
    c.reset(_ctx(n_threads=T))
    # forward 1: both served, seed baselines at their true cumulative counts.
    c.act(_obs(n_threads=T, inflight=[1, 1], ready=[1, 1], msgs=[10, 20], leaves=[40, 80], served=[0, 1]))
    assert int(c._msgs_prev[0]) == 10 and int(c._msgs_prev[1]) == 20
    # forward 2: only thread 0 served; the WIRE reports thread 1 as the sentinel 0 in the length-T lists.
    c.act(_obs(n_threads=T, inflight=[1, 1], ready=[1, 1], msgs=[15, 0], leaves=[60, 0], served=[0]))
    assert int(c._msgs_prev[0]) == 15, "served thread's baseline tracks its true cumulative reading"
    assert int(c._msgs_prev[1]) == 20, "absent thread's baseline is untouched (the sentinel-0 is never differenced)"
    assert int(c._leaves_prev[1]) == 80, "absent thread's leaves baseline likewise untouched"


def test_periodic_update_fires_on_period():
    """The optax step is BATCHED: no update before N completed forwards, exactly one once the buffer reaches N,
    and the buffer clears after the step (Monte-Carlo on the just-collected trajectory)."""
    T = 3
    N = 8
    c = rf.ReinforceGate(update_period=N)
    c.reset(_ctx(n_threads=T))
    obs = _obs(n_threads=T, inflight=[1] * T, ready=[2] * T)
    # the first act has no previous reward; thereafter observe(prev)->act completes one transition per loop.
    prev = list(c.act(obs))
    for _ in range(N - 1):                       # complete N-1 transitions: still below the period.
        c.observe(5.0, {})
        prev = list(c.act(obs))
    assert c.metrics()["updates"] == 0.0, "no update before the period is reached"
    c.observe(5.0, {})                           # the N-th completed transition -> the batched step fires.
    assert c.metrics()["updates"] == 1.0, "exactly one optax step once the buffer reaches N"
    assert c.metrics()["buffer"] == 0.0, "the trajectory buffer clears after the step"


# ----------------------------------------------------------------------------- the learning assertion


def test_policy_gradient_moves_toward_the_rewarded_direction_closed_loop():
    """REINFORCE LEARNING (the defining RL behavior), under a CLOSED-LOOP reward (the realized throughput
    depends on the gate — the lab's actual semantics). The pool reward is a monotone function of how many
    threads the gate DENIES, so one policy direction genuinely dominates and the shared policy has a real
    gradient to climb.

    Reward-favors-DENY: forwards whose Bernoulli sample happened to DENY more threads earn a HIGHER reward, so
    their advantage R_f - b is positive and the policy-gradient RAISES the deny probability — the mean allow
    probability falls below its allow-leaning cold start. Reward-favors-ALLOW (control): the same machinery,
    reward increasing in allows, keeps the allow probability high (allow is both the cold start AND the
    rewarded direction). The two regimes must SEPARATE, the optax step must have fired (updates>0, a non-zero
    gradient norm was seen), and the deny regime must have moved OFF the cold start.

    inflight is strictly positive so the liveness override never masks the learned policy (every sampled action
    is a real act that carries credit); a steady non-saturating context keeps phi a fixed positive vector so
    the gradient lives in the reward, not in a drifting state."""
    T = 4
    NF = 480           # ~60 optax steps at N=8 — enough to converge the tiny policy in-budget.
    state = _obs(n_threads=T, inflight=[1] * T, ready=[4] * T)

    # --- reward-favors-DENY: the more threads deny, the higher the realized pool reward ---
    def reward_favors_deny(decision: list[int]) -> float:
        n_deny = sum(1 for v in decision if v == 0)
        return 10.0 + 30.0 * n_deny    # all-allow -> 10 (low); all-deny -> 130 (high)

    deny_learner = rf.ReinforceGate(lr=0.1, update_period=8, init_allow_logit=2.0)
    deny_learner.reset(_ctx(n_threads=T, seed=1))
    cold_prob = deny_learner.metrics()["mean_allow_prob"]   # the allow-leaning cold start (sigmoid(2.0)~0.88)
    _drive_closed_loop(deny_learner, state, reward_favors_deny, n_forwards=NF)
    deny_m = deny_learner.metrics()

    assert deny_m["updates"] > 0.0, "the optax policy-gradient step fired (learning happened)"
    assert deny_m["grad_norm"] > 0.0, "a non-zero gradient was applied (the policy actually moved)"
    deny_prob = deny_m["mean_allow_prob"]
    assert deny_prob < cold_prob, "reward favors deny -> the policy lowered its allow probability off the cold start"

    # --- reward-favors-ALLOW control: the more threads ALLOW, the higher the reward -> stay allow-leaning ---
    def reward_favors_allow(decision: list[int]) -> float:
        n_allow = sum(1 for v in decision if v == 1)
        return 10.0 + 30.0 * n_allow    # all-deny -> 10 (low); all-allow -> 130 (high)

    allow_learner = rf.ReinforceGate(lr=0.1, update_period=8, init_allow_logit=2.0)
    allow_learner.reset(_ctx(n_threads=T, seed=1))
    _drive_closed_loop(allow_learner, state, reward_favors_allow, n_forwards=NF)
    allow_prob = allow_learner.metrics()["mean_allow_prob"]

    # the two regimes must SEPARATE: the deny-rewarded policy allows much less than the allow-rewarded one.
    assert deny_prob < allow_prob, "the policy gradient separated the two reward regimes"
    assert allow_prob > 0.5, "reward favors allow -> the policy stays allow-leaning (above the indifference line)"


if __name__ == "__main__":
    # plain-runnable (no pytest needed for the non-raises checks), mirroring the repo's bare-script convention.
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn) and _name != "test_invalid_config_fails_loud":
            _fn()
            print(f"PASS {_name}")
    print("all reinforce method checks passed (run via pytest for the fail-loud config test)")
