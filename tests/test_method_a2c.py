#!/usr/bin/env python3
"""
test_method_a2c.py — unit gate for the advantage actor-critic (A2C) controller
(cpp/stage_a/control_lab/methods/a2c.py), a REINFORCEMENT-LEARNING candidate for the issue-gate control lab.

Imports the method's OWN submodule directly (NOT the methods package, and WITHOUT load_all()): sibling method
files are authored in parallel, so importing only `control_lab.methods.a2c` keeps this test isolated from a
half-written neighbour. Pins the FROZEN adapter.Controller contract (reset / observe / act / metrics shape)
plus A2C's defining RL behavior:

  - act() returns a length-T list of values in {0,1} (the per-thread allow bits, a Bernoulli sample of the
    shared actor);
  - observe() is safe (it must not raise; a reward before the first sampled act is ignored — nothing to credit;
    a non-finite reward is dropped, never poisoning the gradient);
  - reset() cold-starts (re-initializes BOTH actor + critic params and their adam moment states; the
    buffer/baseline/pending/awaiting clear);
  - the served-thread first-difference for the coalescence feature honors the wire subtlety (an ABSENT thread
    is never differenced; its baseline is untouched);
  - the bootstrapped transition needs BOTH a reward AND a next-state phi', so a transition lands one forward
    after its reward (its next-state's forward) — the batched optax step fires on that schedule and clears the
    buffer;
  - THE LEARNING ASSERTIONS (method-specific, two of them — the actor AND the critic move the right way):
      (a) actor: under a CLOSED-LOOP reward (the realized pool reward is a monotone function of the gate's own
          decision — the lab's actual semantics, the only way a reward can distinguish a policy direction), a
          reward that pays MORE the more threads DENY drives the actor's policy-gradient to LOWER the shared
          allow probability (the policy moves toward deny, off its allow-leaning cold start), while a reward
          that pays more the more threads ALLOW keeps the allow probability high; the two regimes separate.
      (b) critic: the SHARED CRITIC (cold value 0) learns a NON-TRIVIAL baseline a policy-only method has no
          organ for — under a strongly positive closed-loop reward (production gamma=0.9) its mean value rises
          well above 0; and, isolated at gamma=0 (where the one-step advantage A = r - V has no geometric
          bootstrap tail), the critic drives V -> E[r] so the realized advantage magnitude COLLAPSES from its
          cold-start value (|A| = the full reward) toward 0 — the precise 'critic baselines the advantage'
          mechanism. (For gamma in (0,1) the advantage does NOT vanish in-budget — the geometric fixed point
          V* = r/(1-gamma) is unreachable in ~60 single-step TD updates — so the clean collapse is asserted only
          in the gamma=0 isolation.)

Run pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_a2c.py -q

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
from control_lab.methods import a2c as a2c  # noqa: E402  (own submodule, NOT the package)


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
    """reset() sizes per-run state to T and metrics() exposes the learned-state dashboard scalars — the brief's
    three (mean_value, policy_entropy, mean_advantage) plus learning health."""
    c = a2c.A2CGate()
    c.reset(_ctx(n_threads=4))
    m = c.metrics()
    for key in (
        "mean_value", "policy_entropy", "mean_advantage",
        "grad_norm", "updates", "mean_allow_prob", "baseline", "buffer",
    ):
        assert key in m
    # a fresh learner has taken no updates and emptied its buffer/baseline.
    assert m["updates"] == 0.0
    assert m["buffer"] == 0.0
    assert m["baseline"] == 0.0
    # cold critic value is exactly 0 (zero-initialized value head).
    assert m["mean_value"] == 0.0
    # cold-start allow probability is the allow-leaning init (sigmoid(init_allow_logit) > 0.5).
    assert m["mean_allow_prob"] > 0.5
    # cold policy entropy is the Bernoulli entropy at the allow-leaning init prob (> 0, the policy is stochastic).
    assert m["policy_entropy"] > 0.0


def test_act_returns_length_t_binary():
    """act() returns a length-T list whose every entry is 0 or 1 (the per-thread allow bits, a Bernoulli sample)."""
    T = 4
    c = a2c.A2CGate()
    c.reset(_ctx(n_threads=T))
    out = c.act(_obs(n_threads=T, inflight=[1] * T, ready=[3] * T))
    assert isinstance(out, list)
    assert len(out) == T
    assert all(v in (0, 1) for v in out)


def test_liveness_override_forces_allow_at_zero_inflight():
    """DENY-ONLY semantics: a thread with inflight==0 is an UNGATED forced flush, so the gate must force-allow
    it regardless of what the actor sampled. With every thread at inflight==0, the gate is all-allow."""
    T = 5
    c = a2c.A2CGate()
    c.reset(_ctx(n_threads=T))
    out = c.act(_obs(n_threads=T, inflight=[0] * T, ready=[2, 0, 5, 1, 3]))
    assert out == [1] * T


def test_observe_is_safe():
    """observe() never raises: a reward before the first sampled act is ignored (nothing to credit), and a
    non-finite reward is dropped rather than poisoning the gradient (it never reaches the buffer)."""
    T = 3
    c = a2c.A2CGate()
    c.reset(_ctx(n_threads=T))
    c.observe(123.4, {})                 # before any act(): no pending transition -> ignored, must not raise.
    assert c.metrics()["updates"] == 0.0
    assert c.metrics()["buffer"] == 0.0
    c.act(_obs(n_threads=T, inflight=[1] * T, ready=[2] * T))
    c.observe(float("nan"), {})          # non-finite -> the pending transition is dropped, must not raise.
    c.observe(float("inf"), {})          # nothing pending now -> ignored, must not raise.
    out = c.act(_obs(n_threads=T, inflight=[1] * T, ready=[2] * T))
    assert len(out) == T and all(v in (0, 1) for v in out)
    assert c.metrics()["buffer"] == 0.0  # the nan-dropped transition never became a completed (phi') transition.


def test_invalid_config_fails_loud():
    """ADR-0002: degenerate hyperparameters are CONSTRUCTION errors, raised at the ctor, never a per-forward
    surprise."""
    with pytest.raises(ValueError):
        a2c.A2CGate(lr_actor=0.0)              # non-positive actor learning rate
    with pytest.raises(ValueError):
        a2c.A2CGate(lr_critic=0.0)             # non-positive critic learning rate
    with pytest.raises(ValueError):
        a2c.A2CGate(gamma=1.5)                 # discount out of [0, 1]
    with pytest.raises(ValueError):
        a2c.A2CGate(entropy_coef=-0.1)         # entropy bonus must be >= 0
    with pytest.raises(ValueError):
        a2c.A2CGate(update_period=0)           # batch period must be >= 1
    with pytest.raises(ValueError):
        a2c.A2CGate(hidden=-1)                 # hidden width must be >= 0
    with pytest.raises(ValueError):
        a2c.A2CGate(value_coef=-1.0)           # value loss weight must be >= 0
    with pytest.raises(ValueError):
        a2c.A2CGate(init_allow_logit=float("inf"))  # non-finite cold-start logit
    with pytest.raises(ValueError):
        a2c.A2CGate(max_batch=0)               # buffer cap must be >= 1


# ----------------------------------------------------------------------------- wire subtlety


def test_absent_thread_is_not_first_differenced():
    """The wire subtlety: a thread ABSENT from a forward reads a sentinel-0 cumulative counter, so it must NOT
    be first-differenced (its 0 would manufacture a spurious delta) and its baseline must NOT advance. Seed a
    thread's msgs/leaves baseline, then exclude it from `served` with the wire's sentinel-0 readings; its
    baseline must be unchanged, while a served thread's baseline tracks its true reading."""
    T = 2
    c = a2c.A2CGate()
    c.reset(_ctx(n_threads=T))
    # forward 1: both served, seed baselines at their true cumulative counts.
    c.act(_obs(n_threads=T, inflight=[1, 1], ready=[1, 1], msgs=[10, 20], leaves=[40, 80], served=[0, 1]))
    assert int(c._msgs_prev[0]) == 10 and int(c._msgs_prev[1]) == 20
    # forward 2: only thread 0 served; the WIRE reports thread 1 as the sentinel 0 in the length-T lists.
    c.act(_obs(n_threads=T, inflight=[1, 1], ready=[1, 1], msgs=[15, 0], leaves=[60, 0], served=[0]))
    assert int(c._msgs_prev[0]) == 15, "served thread's baseline tracks its true cumulative reading"
    assert int(c._msgs_prev[1]) == 20, "absent thread's baseline is untouched (the sentinel-0 is never differenced)"
    assert int(c._leaves_prev[1]) == 80, "absent thread's leaves baseline likewise untouched"


def test_periodic_update_fires_on_bootstrapped_schedule():
    """The optax step is BATCHED on the BOOTSTRAPPED schedule: a transition needs BOTH a reward AND a next-state
    phi', so it only completes when the NEXT act supplies phi'. No update fires before N transitions complete;
    exactly one fires once the buffer reaches N; the buffer clears after the step (on-policy)."""
    T = 3
    N = 8
    c = a2c.A2CGate(update_period=N)
    c.reset(_ctx(n_threads=T))
    obs = _obs(n_threads=T, inflight=[1] * T, ready=[2] * T)

    prev = list(c.act(obs))              # act_0: stashes pending_0; no completed transition yet.
    updates_seen = []
    acts = 1
    # drive observe(prev)->act; each act after the first finalizes ONE awaiting transition into the buffer.
    while c.metrics()["updates"] == 0.0 and acts < 4 * N:
        c.observe(5.0, {})               # attach reward to the pending transition (-> awaiting next-state).
        prev = list(c.act(obs))          # finalize the awaiting transition (phi' = this forward); maybe update.
        acts += 1
        updates_seen.append(c.metrics()["updates"])

    assert c.metrics()["updates"] == 1.0, "exactly one optax step once N transitions complete"
    assert c.metrics()["buffer"] == 0.0, "the trajectory buffer clears after the step"
    # the update fired only once the buffer reached N; before that no update was recorded. The bootstrap lag
    # means the completing act is one beyond reinforce's (a transition needs its next-state forward).
    assert acts >= N, "no update could fire before N transitions had a chance to complete (the bootstrap lag)"


# ----------------------------------------------------------------------------- the learning assertions


def test_actor_policy_gradient_separates_reward_regimes_closed_loop():
    """A2C ACTOR LEARNING (the defining RL behavior), under a CLOSED-LOOP reward (the realized throughput
    depends on the gate — the lab's actual semantics). The pool reward is a monotone function of how many
    threads the gate DENIES, so one policy direction genuinely dominates and the shared actor has a real
    advantage-weighted gradient to climb.

    Reward-favors-DENY: forwards whose Bernoulli sample happened to DENY more threads earn a HIGHER reward, so
    the critic-baselined advantage of their (deny) actions is positive and the policy-gradient RAISES the deny
    probability — the mean allow probability falls below its allow-leaning cold start. Reward-favors-ALLOW
    (control): the same machinery, reward increasing in allows, keeps the allow probability high. The two
    regimes must SEPARATE, the optax step must have fired (updates>0, a non-zero gradient norm was seen), and
    the deny regime must have moved OFF the cold start.

    inflight is strictly positive so the liveness override never masks the learned policy (every sampled action
    is a real act that carries credit); a steady non-saturating context keeps phi a fixed positive vector so the
    gradient lives in the reward, not in a drifting state."""
    T = 4
    NF = 480           # ~60 optax steps at N=8 — enough to converge the tiny actor in-budget.
    state = _obs(n_threads=T, inflight=[1] * T, ready=[4] * T)

    def reward_favors_deny(decision: list[int]) -> float:
        n_deny = sum(1 for v in decision if v == 0)
        return 10.0 + 30.0 * n_deny    # all-allow -> 10 (low); all-deny -> 130 (high)

    deny = a2c.A2CGate(lr_actor=0.1, lr_critic=0.2, gamma=0.9, update_period=8, init_allow_logit=2.0)
    deny.reset(_ctx(n_threads=T, seed=1))
    cold_prob = deny.metrics()["mean_allow_prob"]   # the allow-leaning cold start (sigmoid(2.0) ~ 0.88)
    _drive_closed_loop(deny, state, reward_favors_deny, n_forwards=NF)
    deny_m = deny.metrics()

    assert deny_m["updates"] > 0.0, "the optax actor-critic step fired (learning happened)"
    assert deny_m["grad_norm"] > 0.0, "a non-zero gradient was applied (the nets actually moved)"
    deny_prob = deny_m["mean_allow_prob"]
    assert deny_prob < cold_prob, "reward favors deny -> the actor lowered its allow probability off the cold start"

    def reward_favors_allow(decision: list[int]) -> float:
        n_allow = sum(1 for v in decision if v == 1)
        return 10.0 + 30.0 * n_allow    # all-deny -> 10 (low); all-allow -> 130 (high)

    allow = a2c.A2CGate(lr_actor=0.1, lr_critic=0.2, gamma=0.9, update_period=8, init_allow_logit=2.0)
    allow.reset(_ctx(n_threads=T, seed=1))
    _drive_closed_loop(allow, state, reward_favors_allow, n_forwards=NF)
    allow_prob = allow.metrics()["mean_allow_prob"]

    assert deny_prob < allow_prob, "the actor's advantage-weighted gradient separated the two reward regimes"
    assert allow_prob > 0.5, "reward favors allow -> the actor stays allow-leaning (above the indifference line)"


def test_critic_learns_a_nontrivial_value_baseline_closed_loop():
    """A2C CRITIC LEARNING under the REAL config (the family's defining upgrade over REINFORCE's scalar
    baseline). The shared critic starts at value 0 everywhere (zero-initialized read-out). Under a strongly
    POSITIVE closed-loop reward and the production gamma=0.9, the critic must regress its value toward the
    bootstrapped return r + gamma*V(phi'), so its mean value rises well ABOVE 0 — a learned, state-conditioned
    baseline a policy-only method has no organ for.

    Advantage standardization is OFF here so the critic regresses toward the RAW reward scale and `mean_value` is
    a direct, interpretable readout of the learned baseline (with standardization the scale is normalized away
    and the assertion would be against a unit-variance target, not the plant's reward). NOTE the bootstrapped
    advantage A = r + gamma*V' - V does NOT vanish at the in-budget operating point for gamma in (0,1): the
    geometric fixed point V* = r/(1-gamma) is far above what ~60 single-step TD updates reach, so a positive
    residual advantage is EXPECTED and correct (not a divergence) — the clean A-collapse is asserted separately
    below in the gamma=0 isolation, where the geometric tail is absent."""
    T = 4
    NF = 480
    state = _obs(n_threads=T, inflight=[1] * T, ready=[4] * T)

    # a strongly positive reward that still depends on the action (closed-loop), so the critic has a real,
    # large, non-zero return to regress toward (cold value 0 -> the baseline must climb).
    def positive_reward(decision: list[int]) -> float:
        n_allow = sum(1 for v in decision if v == 1)
        return 50.0 + 10.0 * n_allow   # in [50, 90] over the run — strictly, substantially positive.

    c = a2c.A2CGate(
        lr_actor=0.05, lr_critic=0.3, gamma=0.9, update_period=8,
        init_allow_logit=2.0, standardize_adv=False,   # regress toward the RAW reward scale (see docstring)
    )
    c.reset(_ctx(n_threads=T, seed=2))
    assert c.metrics()["mean_value"] == 0.0, "the critic's cold value is exactly 0"
    _drive_closed_loop(c, state, positive_reward, n_forwards=NF)
    m = c.metrics()

    assert m["updates"] > 0.0, "the optax actor-critic step fired"
    # the critic learned a substantial positive baseline tracking the (>= 50) reward — well off its cold 0. A
    # REINFORCE policy with no critic has no value to report at all; this is the A2C-defining organ working.
    assert m["mean_value"] > 20.0, (
        f"the shared critic learned a non-trivial positive value baseline (mean_value={m['mean_value']:.3f}); "
        "a critic-less policy-gradient method has no value organ to show this at all"
    )


def test_critic_collapses_the_advantage_under_gamma_zero_isolation():
    """A2C CRITIC LEARNING — the CLEAN 'baseline the advantage' demonstration, isolated at gamma=0 so the
    one-step advantage A = r + gamma*V' - V reduces to the pure regression residual A = r - V with NO geometric
    bootstrap tail. The critic's whole job is then to drive V -> E[r]; as it does, the realized advantage
    magnitude must COLLAPSE from its cold-start value (where V=0, so |A| = the full reward) toward 0. This is the
    precise sense in which the critic 'baselines the advantage' the brief names — and it is the mechanism a
    scalar running-mean baseline only approximates and a policy-only method lacks entirely.

    The reward here is a CONSTANT (action-independent) positive value so the regression target is unambiguous and
    the actor is held near-frozen (tiny lr_actor) to isolate the critic; the result is deterministic across seeds
    (the value-head dynamics on the fixed phi do not depend on the sampling RNG)."""
    T = 4
    NF = 480
    R = 80.0  # constant, action-independent reward -> the critic's regression target E[r] = 80; cold |A| = 80.
    state = _obs(n_threads=T, inflight=[1] * T, ready=[4] * T)

    c = a2c.A2CGate(
        lr_actor=1e-6,          # actor effectively frozen: isolate the critic's value regression
        lr_critic=0.5, gamma=0.0,  # gamma=0 -> A = r - V exactly (no bootstrap); a fast critic for a clean collapse
        update_period=8, init_allow_logit=2.0, standardize_adv=False,
    )
    c.reset(_ctx(n_threads=T, seed=3))
    assert c.metrics()["mean_value"] == 0.0, "cold value 0 -> the cold-start advantage equals the full reward R"
    _drive_closed_loop(c, state, lambda _decision: R, n_forwards=NF)
    m = c.metrics()

    assert m["updates"] > 0.0, "the optax actor-critic step fired"
    # the critic drove its value UP toward the reward R=80 (well past half) ...
    assert m["mean_value"] > 0.5 * R, (
        f"the critic regressed its value toward the immediate reward (mean_value={m['mean_value']:.3f}, R={R})"
    )
    # ... and so the advantage magnitude COLLAPSED far below the cold-start |A| = R: the critic baselined it.
    assert abs(m["mean_advantage"]) < 0.5 * R, (
        f"the critic baselined the advantage: |A| fell from the cold-start R={R} to "
        f"{abs(m['mean_advantage']):.3f} as V -> E[r] (gamma=0, no bootstrap tail)"
    )


if __name__ == "__main__":
    # plain-runnable (no pytest needed for the non-raises checks), mirroring the repo's bare-script convention.
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn) and _name != "test_invalid_config_fails_loud":
            _fn()
            print(f"PASS {_name}")
    print("all a2c method checks passed (run via pytest for the fail-loud config test)")
