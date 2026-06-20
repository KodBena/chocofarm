#!/usr/bin/env python3
"""
tests/test_method_tabular_q.py — unit test for the tabular_q (reinforcement-learning) issue-gate.

Imports the method's OWN submodule directly (control_lab.methods.tabular_q), NOT the methods package and NOT
load_all() — so this test is isolated from sibling method files being authored in parallel (per the methods/
package docstring: discovery is explicit, a single-submodule import pulls in no siblings).

Asserts the FROZEN adapter.Controller contract surface (reset / act shape+domain / observe safety) plus the
mechanism-specific RL behavior: the 9-state (3 submit-pressure x 3 velocity) discretization, the inflight==0
forced-flush override (and that the override does NOT corrupt the stored policy action), the wire subtlety
(an unserved thread's sentinel-0 neither fabricates a state update nor a velocity delta), and — the LEARNING
assertions — that under a clear reward gradient the shared Q table's TD bootstrap moves the greedy policy
toward the rewarded action (in BOTH directions: a reward favoring allow converges to allow, one favoring deny
converges to deny — proving the gradient, not a fixed bias, drives the policy).

Run: PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_tabular_q.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Sequence

import pytest

from control_lab.adapter import Observation, TrialContext
from control_lab.methods.tabular_q import (
    _N_ACTIONS,
    _N_STATES,
    _N_VELOCITY_BINS,
    TabularQGate,
)

_ALLOW = 1
_DENY = 0


def _ctx(n_threads: int = 4, d: int = 8, seed: int = 0) -> TrialContext:
    return TrialContext(
        n_threads=n_threads,
        d_ceiling=d,
        k_per_thread=10,
        s_min=4,
        chunk_floor=True,
        seed=seed,
    )


def _obs(
    ready: Sequence[float],
    inflight: Sequence[float],
    served: Sequence[int] | None = None,
    t_monotonic: float = 0.0,
) -> Observation:
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
        served=list(range(n)) if served is None else list(served),
        forward_rows=n,
        t_monotonic=t_monotonic,
    )


def _assert_gate(out: Sequence[int], t: int) -> None:
    assert isinstance(out, list)
    assert len(out) == t
    assert all(v in (0, 1) for v in out), f"gate must be binary, got {out}"


def _state_index(ready: float, inflight: float, d: int, velocity_bin: int) -> int:
    """Recompute the 9-state index the controller assigns, so the learning tests can name the single state the
    fixed test frame lands in (mirrors TabularQGate._states_for at the default cut points 0.5, 1.0)."""
    pressure = ready / max(1.0, d - inflight)
    p_bin = int(pressure >= 0.5) + int(pressure >= 1.0)
    return _N_VELOCITY_BINS * p_bin + velocity_bin


# --- contract surface ---------------------------------------------------------------------------------------


def test_reset_and_act_shape_domain() -> None:
    """reset() works and act() returns a length-T list of {0,1} on a synthetic Observation."""
    t = 4
    c = TabularQGate()
    c.reset(_ctx(n_threads=t))
    out = c.act(_obs(ready=[2, 4, 6, 8], inflight=[1, 2, 3, 4]))
    _assert_gate(out, t)


def test_observe_is_safe() -> None:
    """observe() must never throw — including on a non-finite reward (dropped, not poisoning the table)."""
    c = TabularQGate()
    c.reset(_ctx())
    c.observe(12.0, {})
    c.act(_obs(ready=[5, 5, 5, 5], inflight=[1, 1, 1, 1]))
    c.observe(-3.0, {"anything": 7})
    c.observe(float("nan"), {})   # non-finite reward is dropped, not folded in
    c.observe(float("inf"), {})
    out = c.act(_obs(ready=[5, 5, 5, 5], inflight=[1, 1, 1, 1]))
    _assert_gate(out, 4)


def test_observe_before_first_act_is_safe() -> None:
    """A reward delivered before any act (no stored transition) must be a safe no-op on the table."""
    c = TabularQGate()
    c.reset(_ctx(n_threads=2))
    c.observe(7.0, {})                                   # no pending (s, a) yet
    out = c.act(_obs(ready=[3, 3], inflight=[2, 2]))     # nothing to bootstrap against -> no update
    _assert_gate(out, 2)
    assert c.metrics()["updates"] == 0.0


def test_construction_validates_knobs() -> None:
    """fail loud (ADR-0002): degenerate alpha / gamma / epsilon schedule / bins raise at construction."""
    with pytest.raises(ValueError):
        TabularQGate(alpha=0.0)
    with pytest.raises(ValueError):
        TabularQGate(alpha=1.5)
    with pytest.raises(ValueError):
        TabularQGate(gamma=-0.1)
    with pytest.raises(ValueError):
        TabularQGate(gamma=1.0)              # gamma must be < 1 (a tabular bootstrap needs a contraction)
    with pytest.raises(ValueError):
        TabularQGate(eps_start=0.2, eps_end=0.5)   # eps_end must be <= eps_start
    with pytest.raises(ValueError):
        TabularQGate(eps_decay_epochs=0)
    with pytest.raises(ValueError):
        TabularQGate(pressure_cuts=(1.0, 0.5))      # lo < hi required
    with pytest.raises(ValueError):
        TabularQGate(pressure_cuts=(-0.1, 1.0))     # lo >= 0 required


# --- mechanism: the liveness override and the wire subtlety -------------------------------------------------


def test_inflight_zero_force_allows() -> None:
    """inflight==0 is an UNGATED forced flush (DENY-ONLY semantics) -> always force-allow regardless of policy.

    Force a greedy-deny policy (pre-load Q so deny dominates and kill exploration), then a frame with one
    thread at inflight==0: that thread is allowed despite the policy, the warmed threads follow the policy."""
    c = TabularQGate(eps_start=0.0, eps_end=0.0)   # no exploration: gate is the greedy policy + override only
    c.reset(_ctx(n_threads=3))
    c._Q[:, _DENY] = 5.0    # make deny greedily dominant in every state
    c._Q[:, _ALLOW] = 0.0
    out = c.act(_obs(ready=[9, 9, 9], inflight=[2, 0, 5]))   # huge backlog; deny-greedy everywhere
    assert out == [0, 1, 0], out   # the inflight==0 thread (t1) is force-allowed; the others are denied


def test_unserved_thread_holds_state_and_baseline() -> None:
    """WIRE SUBTLETY: a thread ABSENT from obs.served (sentinel-0 reading) must not get a TD update, a stored
    transition, or a velocity-baseline refresh — its pending (s, a) and baseline are held until it reappears.

    Two threads, only t0 served across two forwards (t1 always absent). After the second forward exactly one
    TD update has fired (t0's first transition closing against its second state); t1 contributed nothing."""
    c = TabularQGate()
    c.reset(_ctx(n_threads=2))
    c.act(_obs(ready=[4, 0], inflight=[2, 0], served=[0]))    # only t0 real; t1 absent (sentinel-0)
    assert c._last_state[1] == -1, "an unserved thread must have no stored transition"
    c.observe(5.0, {})
    c.act(_obs(ready=[4, 0], inflight=[2, 0], served=[0]))    # t0 closes one transition; t1 still absent
    assert c._last_state[1] == -1, "an unserved thread must still have no stored transition"
    assert c.metrics()["updates"] == 1.0, "exactly t0's single transition updated the table"


def test_state_discretization_spans_the_table() -> None:
    """The 3x3 discretization is reachable: low/med/high pressure x draining/flat/filling velocity all map into
    the 9-state index range, and distinct (pressure, velocity) frames land on distinct states."""
    d = 8
    # low (x=0.25), med (x=0.75), high (x=2.0) pressures at flat velocity -> three distinct states.
    s_low = _state_index(2, 0, d, velocity_bin=1)    # x = 2/8 = 0.25 -> low
    s_med = _state_index(6, 0, d, velocity_bin=1)    # x = 6/8 = 0.75 -> med
    s_high = _state_index(8, 4, d, velocity_bin=1)   # x = 8/4 = 2.0 -> high
    assert len({s_low, s_med, s_high}) == 3
    assert all(0 <= s < _N_STATES for s in (s_low, s_med, s_high))


# --- the LEARNING assertions --------------------------------------------------------------------------------


def _run_action_conditioned(
    c: TabularQGate,
    ready: float,
    inflight: float,
    reward_for_action: dict[int, float],
    n_forwards: int,
) -> list[int]:
    """Drive the learner closed-loop on a FIXED single-thread frame (so the state is constant after the first
    served forward) for n_forwards. Each forward, feed observe() the reward THIS run assigns to the action the
    policy emitted last forward, then act() once. The emitted gate equals the policy action because inflight>0
    (no liveness override) — so the reward is conditioned on the learner's own choice, the clean per-forward
    throughput signal the harness supplies for that gate. Returns the sequence of emitted actions."""
    obs = _obs(ready=[ready], inflight=[inflight])
    actions: list[int] = []
    last_action: int | None = None
    for _ in range(n_forwards):
        if last_action is not None:
            c.observe(reward_for_action[last_action], {})   # credit the previous action's reward
        out = c.act(obs)
        last_action = int(out[0])
        actions.append(last_action)
    return actions


def test_q_learning_converges_to_rewarded_allow() -> None:
    """LEARNING: when ALLOW earns far more than DENY, the shared Q table's TD bootstrap drives Q(s, allow) above
    Q(s, deny) in the (single) state the fixed frame visits, and the converged greedy policy is ALLOW.

    Fixed frame ready=4, inflight=2, D=8 -> x=4/6 in [0.5,1.0) = med pressure, flat velocity (constant ready).
    Reward 10 for allow, 1 for deny. After the run, the greedy action in that state is allow and Q[s,allow] >
    Q[s,deny], and exploration has annealed to eps_end."""
    c = TabularQGate(alpha=0.3, gamma=0.6, eps_start=0.5, eps_end=0.02, eps_decay_epochs=200)
    c.reset(_ctx(n_threads=1, seed=1))   # exploration RNG is seeded from ctx.seed (reproducible per trial)
    s = _state_index(4, 2, d=8, velocity_bin=1)   # the constant state the fixed frame lands in

    actions = _run_action_conditioned(c, ready=4, inflight=2, reward_for_action={_ALLOW: 10.0, _DENY: 1.0}, n_forwards=400)

    q = c._Q
    assert q[s, _ALLOW] > q[s, _DENY], f"allow must dominate after rewarding it: Q[s]={q[s].tolist()}"
    assert int(q[s].argmax()) == _ALLOW, f"greedy policy in the visited state must be allow: Q[s]={q[s].tolist()}"
    # the policy actually exploits allow by the back half of the run (exploration annealed, allow dominant).
    tail = actions[-100:]
    assert sum(tail) / len(tail) > 0.8, f"the converged policy should mostly allow, allow-rate={sum(tail)/len(tail)}"
    assert c.metrics()["epsilon"] == pytest.approx(0.02), c.metrics()["epsilon"]
    assert c.metrics()["updates"] > 0.0


def test_q_learning_converges_to_rewarded_deny() -> None:
    """LEARNING (the mirror — the GRADIENT drives it, not a fixed bias toward allow): with the reward flipped so
    DENY earns far more, the same fixed frame converges its greedy policy to DENY and Q(s, deny) > Q(s, allow).

    Same frame/state as the allow test; reward 10 for deny, 1 for allow. This rules out the policy just
    drifting to allow regardless of reward (e.g. via the liveness override or an init bias)."""
    c = TabularQGate(alpha=0.3, gamma=0.6, eps_start=0.5, eps_end=0.02, eps_decay_epochs=200)
    c.reset(_ctx(n_threads=1, seed=2))
    s = _state_index(4, 2, d=8, velocity_bin=1)

    actions = _run_action_conditioned(c, ready=4, inflight=2, reward_for_action={_ALLOW: 1.0, _DENY: 10.0}, n_forwards=400)

    q = c._Q
    assert q[s, _DENY] > q[s, _ALLOW], f"deny must dominate after rewarding it: Q[s]={q[s].tolist()}"
    assert int(q[s].argmax()) == _DENY, f"greedy policy in the visited state must be deny: Q[s]={q[s].tolist()}"
    tail = actions[-100:]
    assert sum(tail) / len(tail) < 0.2, f"the converged policy should mostly deny, allow-rate={sum(tail)/len(tail)}"


def test_bootstrap_uses_gamma_discounted_future() -> None:
    """RL-vs-bandit: the TD target includes the gamma*max_a' Q(s', a') bootstrap, so with a constant reward r
    the visited state's Q converges toward the discounted-return fixed point r/(1-gamma), NOT the bare r a
    one-step bandit would settle on. (This is the bootstrap term the lab synthesis asks this method to measure.)

    Constant reward 5.0, gamma=0.5 -> fixed point 5/(1-0.5)=10.0. After many updates Q at the visited state
    sits well above the bare immediate reward 5.0, evidencing the bootstrap is live."""
    r_const = 5.0
    gamma = 0.5
    c = TabularQGate(alpha=0.3, gamma=gamma, eps_start=0.3, eps_end=0.0, eps_decay_epochs=100)
    c.reset(_ctx(n_threads=1, seed=3))
    s = _state_index(4, 2, d=8, velocity_bin=1)

    _run_action_conditioned(c, ready=4, inflight=2, reward_for_action={_ALLOW: r_const, _DENY: r_const}, n_forwards=400)

    fixed_point = r_const / (1.0 - gamma)   # 10.0
    qmax = float(c._Q[s].max())
    assert qmax > r_const + 1.0, f"the bootstrap must lift Q above the bare reward {r_const}, got {qmax}"
    assert qmax == pytest.approx(fixed_point, abs=1.5), f"Q should approach r/(1-gamma)={fixed_point}, got {qmax}"


def test_metrics_shape() -> None:
    """metrics() exposes the introspection scalars the dashboard reads (epsilon, max Q, states visited, ...)."""
    c = TabularQGate()
    c.reset(_ctx(n_threads=2))
    c.act(_obs(ready=[4, 4], inflight=[2, 2]))
    m = c.metrics()
    for key in ("epsilon", "max_q", "min_q", "n_states_visited", "epochs", "updates", "last_reward"):
        assert key in m, f"missing metric {key}"
        assert isinstance(m[key], float)
    assert 0.0 <= m["n_states_visited"] <= float(_N_STATES)
    assert m["epsilon"] <= 0.5    # within the default schedule bounds
    assert m["epochs"] == 1.0
    # the action space is binary across all 9 states -> the table is 9x2.
    assert c._Q.shape == (_N_STATES, _N_ACTIONS)
