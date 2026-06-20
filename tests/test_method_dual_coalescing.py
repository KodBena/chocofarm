#!/usr/bin/env python3
"""
tests/test_method_dual_coalescing.py — unit test for the dual / Lagrangian coalescing issue-gate
(online family), control_lab.methods.dual_coalescing.

Imports the method module DIRECTLY (control_lab.methods.dual_coalescing), NOT the methods package and
NOT load_all() — sibling candidate methods are written in parallel and may be half-finished, so this test
must not pull them in (the package's explicit-discovery contract, methods/__init__.py).

Run:
  PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_dual_coalescing.py -q

Asserts the FROZEN adapter.Controller contract (reset / observe / act / metrics), then the method's
defining online-learning behaviors:
  * the SLOW outer loop hill-climbs the setpoint S* on the pool-reward gradient — improving reward keeps
    climbing, regressing reward reverses direction and backs S* off (the mandated learning assertion);
  * the FAST inner loop's dual price p ascends under a sustained coalescing shortfall (S_inst < S*) and the
    gate then DENIES the under-coalescing threads;
  * the liveness override (inflight==0 -> force allow) holds even with the price engaged;
  * the wire first-difference subtlety — an ABSENT thread (sentinel 0 in the length-T feature lists) is
    never differenced against its stale cumulative baseline (no phantom negative delta).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Sequence

import pytest

from control_lab.adapter import REGISTRY, Observation, TrialContext
from control_lab.methods.dual_coalescing import DualCoalescingGate

T = 4
CTX = TrialContext(n_threads=T, d_ceiling=3, k_per_thread=50, s_min=2, chunk_floor=True, seed=0)


def _obs(
    served: Sequence[int],
    inflight: Sequence[int],
    leaves: Sequence[int],
    msgs: Sequence[int],
    t_monotonic: float = 0.0,
) -> Observation:
    """Build a synthetic Observation. The harness fills length-T lists fresh as [0]*T and writes only the
    served tids, so absent threads carry sentinel 0 in these lists — the caller models that explicitly."""
    feats: dict[str, Any] = {
        "n_threads": T,
        "d_ceiling": 3,
        "server_rows_per_forward": int(sum(inflight)),
        "inflight": list(inflight),
        "ready": [0] * T,
        "msgs": list(msgs),
        "leaves": list(leaves),
        "rtt_us": [0] * T,
    }
    return Observation(
        features=feats, served=list(served), forward_rows=int(sum(inflight)), t_monotonic=t_monotonic
    )


def test_registered() -> None:
    """The method self-registers into the FROZEN REGISTRY under its name, and the factory builds a
    same-named controller in the online family."""
    assert "dual_coalescing" in REGISTRY
    g = REGISTRY["dual_coalescing"]()
    assert g.family == "online"
    assert g.name.startswith("dual_coalescing")


def test_reset_and_act_contract() -> None:
    """reset() sizes state to T and seeds S* from the runner's S_min; act() returns a length-T list of
    {0,1}; the first decision is the all-allow baseline (no prior counters / clock)."""
    g = DualCoalescingGate()
    g.reset(CTX)
    # S* seeds from the coalescing floor S_min when no explicit init is given.
    assert g.metrics()["s_star"] == float(CTX.s_min)

    out = g.act(_obs([0, 1, 2, 3], [5, 5, 5, 5], [10, 10, 10, 10], [5, 5, 5, 5]))
    assert isinstance(out, list)
    assert len(out) == T
    assert all(v in (0, 1) for v in out)
    assert out == [1, 1, 1, 1]   # first decision is the AllAllow baseline


def test_observe_safe_on_degenerate_inputs() -> None:
    """observe() must be total on the per-forward path: a non-finite reward is ignored, not propagated, and
    repeated observe()s never throw (the watchdog owns loudness; act/observe stay cheap and non-throwing)."""
    g = DualCoalescingGate(window=2)
    g.reset(CTX)
    g.observe(float("nan"), {})    # non-finite -> ignored
    g.observe(float("inf"), {})    # non-finite -> ignored
    for _ in range(10):
        g.observe(1.0, {})
    assert isinstance(g.metrics()["s_star"], float)


def test_act_never_throws_on_short_or_empty_features() -> None:
    """A malformed/short/absent feature frame is tolerated (defaulted reads) so act() is total on the hot
    path — it still returns a valid length-T binary vector."""
    g = DualCoalescingGate()
    g.reset(CTX)
    g.act(_obs([0, 1, 2, 3], [5, 5, 5, 5], [10, 10, 10, 10], [5, 5, 5, 5]))  # seed
    # empty / missing feature lists
    bad = Observation(features={}, served=[0, 1], forward_rows=0, t_monotonic=1.0)
    out = g.act(bad)
    assert len(out) == T and all(v in (0, 1) for v in out)


def test_slow_loop_hill_climbs_setpoint_on_reward_gradient() -> None:
    """THE LEARNING ASSERTION (slow outer loop). The setpoint S* is a coordinate hill-climb driven by the
    pool reward over a hold window W. Feed: window-1 a reward baseline; window-2 IMPROVING reward -> S*
    keeps climbing in the same (upward) direction; window-3 REGRESSING reward -> the search reverses
    direction (and damps the step) and S* backs off. This is the reward-gradient -> setpoint coupling that
    makes the method online-learning."""
    g = DualCoalescingGate(window=2, s_star_step=0.5, s_star_min=0.0, s_star_max=8.0)
    g.reset(CTX)
    s0 = g.metrics()["s_star"]
    step0 = g.metrics()["s_star_step"]
    assert step0 > 0.0   # starts climbing upward

    # window 1 closes -> records the baseline mean, S* does not move yet.
    g.observe(100.0, {})
    g.observe(100.0, {})
    assert g.metrics()["s_star"] == s0

    # window 2 closes with a HIGHER mean (improving) -> keep the upward direction, S* rises.
    g.observe(120.0, {})
    g.observe(120.0, {})
    s_after_improve = g.metrics()["s_star"]
    step_after_improve = g.metrics()["s_star_step"]
    assert step_after_improve > 0.0
    assert s_after_improve > s0, "improving reward must drive the setpoint S* upward"

    # window 3 closes with a LOWER mean (regression) -> reverse direction + shrink the step, S* backs off.
    g.observe(80.0, {})
    g.observe(80.0, {})
    s_after_regress = g.metrics()["s_star"]
    step_after_regress = g.metrics()["s_star_step"]
    assert step_after_regress < 0.0, "a reward regression must reverse the setpoint search direction"
    assert abs(step_after_regress) < abs(step_after_improve), "the reversed step is damped (shrunk)"
    assert s_after_regress < s_after_improve, "S* backs off after the throughput regression"


def test_fast_price_ascends_and_gate_denies_under_coalescing_shortfall() -> None:
    """FAST inner loop. Under a SUSTAINED coalescing shortfall (S_inst < S*), the dual price p does
    projected ascent past the cutoff and the gate then DENIES the under-coalescing threads (to fatten their
    next batch). With s_min=2 the seeded S* is 2.0; we drive S_inst=0.5 (leaves +2 per msgs +4)."""
    g = DualCoalescingGate(eta=0.5, cutoff=0.2, window=999)  # window huge: isolate the fast loop
    g.reset(CTX)
    assert g.metrics()["price"] == 0.0

    # seed the baselines (first decision is all-allow).
    g.act(_obs([0, 1, 2, 3], [5, 5, 5, 5], [100, 100, 100, 100], [10, 10, 10, 10]))

    last: Sequence[int] = [1, 1, 1, 1]
    leaves = 100
    msgs = 10
    for _ in range(6):
        leaves += 2   # +2 leaves ...
        msgs += 4     # ... per +4 msgs -> S_inst = 0.5, well under S*=2.0
        last = g.act(_obs([0, 1, 2, 3], [5, 5, 5, 5], [leaves] * 4, [msgs] * 4))

    assert g.metrics()["mean_s_inst"] == pytest.approx(0.5)
    assert g.metrics()["price"] > 0.2, "price must ascend above the cutoff under a sustained shortfall"
    assert sum(last) < T, "with the price engaged and S_inst < S*, the gate must deny at least one thread"


def test_liveness_override_forces_allow_at_inflight_zero() -> None:
    """DENY-ONLY gate semantics: inflight==0 is an UNGATED forced flush, so a deny is a no-op there. Even
    with the price engaged and the thread under-coalescing, inflight==0 must force allow."""
    g = DualCoalescingGate(eta=0.5, cutoff=0.2, window=999)
    g.reset(CTX)
    g.act(_obs([0, 1, 2, 3], [5, 5, 5, 5], [100, 100, 100, 100], [10, 10, 10, 10]))  # seed
    # drive the price up with a shortfall while inflight>0 ...
    leaves, msgs = 100, 10
    for _ in range(5):
        leaves += 2
        msgs += 4
        g.act(_obs([0, 1, 2, 3], [5, 5, 5, 5], [leaves] * 4, [msgs] * 4))
    assert g.metrics()["price"] > 0.2
    # ... now inflight==0 for everyone: the gate must force-allow despite the engaged price.
    leaves += 2
    msgs += 4
    out = g.act(_obs([0, 1, 2, 3], [0, 0, 0, 0], [leaves] * 4, [msgs] * 4))
    assert out == [1, 1, 1, 1], "inflight==0 must force allow (a deny is a no-op at the forced flush)"


def test_wire_subtlety_absent_thread_not_first_differenced() -> None:
    """Honor the wire first-difference subtlety: lab_server fills length-T lists fresh as [0]*T and writes
    only served tids, so an ABSENT thread reads sentinel 0. A first-difference of leaves/msgs must run ONLY
    for served threads against a per-thread baseline; an absent thread's baseline must stay put (a sentinel
    0 would manufacture a spurious negative delta / phantom coalescing reading)."""
    g = DualCoalescingGate(eta=0.5, cutoff=0.2, window=999)
    g.reset(CTX)
    # seed all four at cumulative leaves=100, msgs=10.
    g.act(_obs([0, 1, 2, 3], [5, 5, 5, 5], [100, 100, 100, 100], [10, 10, 10, 10]))

    # serve ONLY thread 0: leaves 100->108 (+8), msgs 10->14 (+4) => S_inst[0]=2.0. Threads 1..3 absent
    # carry sentinel 0 in the wire lists.
    g.act(_obs([0], [5, 0, 0, 0], [108, 0, 0, 0], [14, 0, 0, 0]))
    assert g._s_inst[0] == pytest.approx(2.0)             # served thread differenced correctly
    # absent threads' baselines are UNTOUCHED by the sentinel-0 reads.
    for j in (1, 2, 3):
        assert g._leaves_prev[j] == 100
        assert g._msgs_prev[j] == 10
        assert g._s_inst[j] == 0.0                        # never updated from a sentinel

    # now serve thread 1 for its first post-seed reading: 100->104 (+4), 10->12 (+2) => 2.0, NOT a phantom
    # negative delta off a stale baseline.
    g.act(_obs([1], [0, 5, 0, 0], [0, 104, 0, 0], [0, 12, 0, 0]))
    assert g._s_inst[1] == pytest.approx(2.0)
