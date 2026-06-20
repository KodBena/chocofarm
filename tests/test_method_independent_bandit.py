#!/usr/bin/env python3
"""
tests/test_method_independent_bandit.py — unit gate for the independent_bandit control-lab method
(cpp/stage_a/control_lab/methods/independent_bandit.py), the T-INDEPENDENT per-thread 2-arm bandits
(ONLINE-LEARNING family) decoupled ablation.

Imports the method module DIRECTLY (control_lab.methods.independent_bandit), NOT the methods package — its
siblings are written in parallel, so a package import / load_all() would pull in mid-write modules. Run pinned:

    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_independent_bandit.py -q

Asserts the FROZEN adapter.Controller contract (reset/observe/act/metrics; length-T {0,1} act; observe safe),
the WIRE SUBTLETY (an absent thread is never first-differenced into a spurious own-rate), the DENY-ONLY
liveness override (inflight==0 force-allows even on a learned deny arm), the deterministic registration, and
— the method-specific LEARNING assertion — that the per-thread independent learners move the RIGHT way and
DECOUPLE: under opposite per-thread own-rate gradients, the productive thread learns to prefer ALLOW while
the unproductive thread does not, and the two threads reach DIFFERENT learned values (the decoupling a
parameter-sharing method structurally cannot exhibit).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from control_lab.adapter import REGISTRY, Observation, TrialContext
from control_lab.methods.independent_bandit import (
    _ALLOW,
    _DENY,
    IndependentBanditGate,
    _argmax_allow_tiebreak,
)

# the arm-table axes the learning assertions index into the private state by (thread, bin, arm).
_BIN0 = 0


def _ctx(n_threads: int = 4, d: int = 8, k: int = 16, seed: int = 7) -> TrialContext:
    return TrialContext(
        n_threads=n_threads, d_ceiling=d, k_per_thread=k, s_min=4, chunk_floor=True, seed=seed
    )


def _obs(
    *,
    t: int,
    inflight: Sequence[int],
    ready: Sequence[int],
    leaves: Sequence[int],
    served: Sequence[int],
    now: float,
    forward_rows: int = 10,
    msgs: Sequence[int] | None = None,
    rtt_us: Sequence[int] | None = None,
) -> Observation:
    """Build a synthetic Observation the way lab_server does: length-T lists, the served tids carrying real
    values. (The caller is responsible for zeroing absent tids if it wants to model the [0]*T sentinel.)"""
    feats = {
        "n_threads": t,
        "d_ceiling": 8,
        "server_rows_per_forward": float(forward_rows),
        "inflight": list(inflight),
        "ready": list(ready),
        "msgs": list(msgs) if msgs is not None else [0] * t,
        "leaves": list(leaves),
        "rtt_us": list(rtt_us) if rtt_us is not None else [0] * t,
    }
    return Observation(features=feats, served=list(served), forward_rows=forward_rows, t_monotonic=now)


# --------------------------------------------------------------------------- contract


def test_registered_under_canonical_name() -> None:
    """The method registers additively under its canonical name and the factory yields the right class/family."""
    assert "independent_bandit" in REGISTRY
    g = REGISTRY["independent_bandit"]()
    assert isinstance(g, IndependentBanditGate)
    assert g.family == "online"
    assert isinstance(g.name, str) and g.name.startswith("independent_bandit")


def test_reset_sizes_state_and_clears() -> None:
    """reset() sizes every per-run table to T and starts every thread on the ALLOW arm (baseline-by-default),
    with empty learner statistics and empty hold-window accumulators."""
    g = IndependentBanditGate()
    g.reset(_ctx(n_threads=5))
    assert g._t == 5
    assert g._Q.shape == (5, 1, 2) and g._N.shape == (5, 1, 2)
    assert np.all(g._arm == _ALLOW)              # every thread starts allowing -> reproduces baseline cold
    assert not np.any(g._N)                      # no pulls yet
    assert g._epochs == 0 and g._pool_n == 0 and g._win_forwards == 0
    assert np.all(np.isnan(g._leaf_base))        # no leaves baseline until a thread is first served


def test_act_returns_length_t_binary() -> None:
    """act() returns a length-T sequence of {0,1} on a synthetic Observation (the FROZEN contract)."""
    g = IndependentBanditGate()
    t = 4
    g.reset(_ctx(n_threads=t))
    out = g.act(_obs(t=t, inflight=[2, 2, 2, 2], ready=[3, 1, 0, 5], leaves=[5, 5, 5, 5],
                     served=[0, 1, 2, 3], now=100.0))
    assert isinstance(out, list) and len(out) == t
    assert set(out) <= {0, 1}


def test_observe_is_safe_including_nonfinite() -> None:
    """observe() accepts a finite pool reward and is total on a non-finite one (ignored, never poisons state)."""
    g = IndependentBanditGate()
    g.reset(_ctx(n_threads=3))
    g.observe(12.0, {"forward_rows": 12})
    assert g._pool_n == 1 and g._pool_sum == pytest.approx(12.0)
    g.observe(float("nan"), {})                  # must be ignored, not folded
    g.observe(float("inf"), {})
    assert g._pool_n == 1 and g._pool_sum == pytest.approx(12.0)  # unchanged by the non-finite rewards


def test_construction_rejects_degenerate_knobs() -> None:
    """fail loud (ADR-0002): degenerate knobs are construction errors on the strongest surface (the ctor)."""
    with pytest.raises(ValueError):
        IndependentBanditGate(c=-1.0)
    with pytest.raises(ValueError):
        IndependentBanditGate(hold_window=0)
    with pytest.raises(ValueError):
        IndependentBanditGate(gamma=0.0)
    with pytest.raises(ValueError):
        IndependentBanditGate(gamma=1.5)
    with pytest.raises(ValueError):
        IndependentBanditGate(explore="bogus")   # type: ignore[arg-type]
    with pytest.raises(ValueError):
        IndependentBanditGate(epsilon=2.0)
    with pytest.raises(ValueError):
        IndependentBanditGate(context_bins=0)
    with pytest.raises(ValueError):
        IndependentBanditGate(beta=float("nan"))


def test_argmax_allow_tiebreak() -> None:
    """The cold-tie helper breaks toward ALLOW (arm 1): allow wins on a tie or when it is strictly larger;
    deny wins only when strictly larger."""
    score = np.array([[1.0, 1.0],   # tie         -> allow
                      [2.0, 1.0],   # deny larger -> deny
                      [0.5, 0.9]])  # allow larger-> allow
    assert _argmax_allow_tiebreak(score).tolist() == [_ALLOW, _DENY, _ALLOW]


# --------------------------------------------------------------------------- liveness / wire subtlety


def test_liveness_override_forces_allow_on_a_learned_deny() -> None:
    """DENY-ONLY semantics: inflight[t]==0 is an UNGATED forced flush, so a thread is force-allowed there even
    if its bandit has learned the DENY arm. Set one thread's held arm to DENY and give it zero in-flight."""
    g = IndependentBanditGate()
    t = 3
    g.reset(_ctx(n_threads=t))
    g._arm[1] = _DENY                            # pretend thread 1 learned to deny
    out = g.act(_obs(t=t, inflight=[2, 0, 2], ready=[1, 1, 1], leaves=[0, 0, 0],
                     served=[0, 1, 2], now=10.0))
    assert out[1] == 1                           # inflight==0 force-allow overrides the learned deny
    # a thread with the deny arm AND in-flight work is genuinely denied (the gate is real, not a no-op).
    g._arm[0] = _DENY
    out2 = g.act(_obs(t=t, inflight=[5, 0, 2], ready=[1, 1, 1], leaves=[0, 0, 0],
                      served=[0, 1, 2], now=11.0))
    assert out2[0] == 0


def test_absent_thread_is_never_first_differenced() -> None:
    """WIRE SUBTLETY: a thread ABSENT from a forward reads the [0]*T sentinel for the CUMULATIVE leaves
    counter, not its true value. The own-rate must first-difference ONLY served threads against their
    baseline; an absent thread must NOT be differenced (its sentinel-0 would manufacture a spurious negative
    delta, clamped to a fabricated own-rate of 0). Assert the absent thread accumulates NO own-rate sample and
    keeps its real baseline, while the served thread does difference."""
    g = IndependentBanditGate(hold_window=1000)  # huge window: keep all samples in one open window to inspect
    t = 2
    g.reset(_ctx(n_threads=t))
    # forward 1: both served, seeds both baselines (no first-difference on the seeding forward).
    g.act(_obs(t=t, inflight=[2, 2], ready=[1, 1], leaves=[100, 200], served=[0, 1], now=1.0))
    assert g._own_n.tolist() == [0, 0]           # seeding forward differences nothing
    assert g._leaf_base.tolist() == [100.0, 200.0]
    # forward 2: ONLY thread 0 served. thread 1 is absent -> its leaves slot is the sentinel 0 in the wire.
    g.act(_obs(t=t, inflight=[2, 0], ready=[1, 1], leaves=[140, 0], served=[0], now=2.0))
    # thread 0 differenced (140-100=40 over dt=1 -> own-rate 40, one sample); thread 1 NOT differenced.
    assert g._own_n.tolist() == [1, 0]
    assert g._own_sum[0] == pytest.approx(40.0)  # 40 leaves / 1.0s
    assert g._own_sum[1] == pytest.approx(0.0)   # absent thread: no spurious (0 - 200) delta manufactured
    assert g._leaf_base[1] == pytest.approx(200.0)  # absent thread keeps its REAL baseline, not the sentinel 0


# --------------------------------------------------------------------------- the LEARNING assertion


def _run_plant(
    g: IndependentBanditGate,
    *,
    t: int,
    productivity: Sequence[float],
    n_forwards: int,
    dt: float = 0.01,
) -> None:
    """Drive the bandit against a synthetic plant for n_forwards. The plant models the per-thread own-leaf
    signal the controller learns from: a thread's cumulative `leaves` grows by `productivity[i]` units per
    forward WHEN the controller ALLOWED it last forward, and by ~0 when it DENIED it. So a productive thread
    earns a high own-leaf-rate exactly on the forwards its bandit chose ALLOW, and ~0 on the forwards it chose
    DENY — a clean per-thread reward gradient that should drive a productive thread's ALLOW value above its
    DENY value, and leave an unproductive thread indifferent. The pool reward is the total leaves issued this
    forward (the blend term). All threads are always served (so the wire-subtlety masking is exercised
    elsewhere, not here). Deterministic: no rng (explore='ucb')."""
    leaves = [1000.0] * t                        # start cumulative leaves well above 0
    now = 0.0
    prev_allow = [1] * t                          # cold start = all-allow (matches the controller's reset arm)
    pending_pool: float | None = None
    for _ in range(n_forwards):
        # the plant advances cumulative leaves from the PREVIOUS forward's gate (the work that gate admitted).
        issued = [productivity[i] if prev_allow[i] == 1 else 0.0 for i in range(t)]
        for i in range(t):
            leaves[i] += issued[i]
        now += dt
        pool = float(sum(issued))                 # this forward's realized rows (the pool reward; HIGHER better)
        # harness order: observe(reward_of_previous_act) THEN act(obs).
        if pending_pool is not None:
            g.observe(pending_pool, {"forward_rows": int(pending_pool)})
        obs = _obs(
            t=t,
            inflight=[2] * t,                     # always in-flight>0 so the liveness override never fires here
            ready=[3] * t,
            leaves=[int(x) for x in leaves],
            served=list(range(t)),
            now=now,
            forward_rows=int(pool),
        )
        out = g.act(obs)
        prev_allow = list(out)
        pending_pool = pool


def test_learns_per_thread_and_decouples() -> None:
    """METHOD-SPECIFIC LEARNING ASSERTION (the decoupled-bandit contract).

    Two threads with OPPOSITE own-rate gradients: thread 0 is highly productive (big leaf growth while
    allowed), thread 1 produces nothing whether allowed or not. After running the plant long enough for both
    per-thread bandits to pull both arms several times:

      (a) the productive thread (0) has learned ALLOW > DENY in value and ends on the ALLOW arm — its own
          per-thread leaf-rate credit moved its bandit the right way;
      (b) the two threads reach DIFFERENT learned ALLOW values — the DECOUPLING: independent learners draw
          different conclusions from per-thread evidence, which a single shared-parameter learner structurally
          cannot. (thread 0's ALLOW value is well above thread 1's, since only thread 0 earned own-rate.)

    Uses beta=0 (the PURE decoupled bandit, no pool blend) so the assertion isolates the per-thread own-rate
    signal, and explore='ucb' (deterministic; both arms get an initial pull) so the test needs no rng seed."""
    g = IndependentBanditGate(beta=0.0, hold_window=4, gamma=0.97, explore="ucb", c=0.5)
    t = 2
    g.reset(_ctx(n_threads=t))
    _run_plant(g, t=t, productivity=[50.0, 0.0], n_forwards=400)

    q0_allow = g._Q[0, _BIN0, _ALLOW]
    q0_deny = g._Q[0, _BIN0, _DENY]
    q1_allow = g._Q[1, _BIN0, _ALLOW]

    # both arms of the productive thread were actually pulled (D-UCB forces the initial DENY pull, then the
    # learning is real, not a never-explored default).
    assert g._N[0, _BIN0, _ALLOW] > 0.0 and g._N[0, _BIN0, _DENY] > 0.0

    # (a) the productive thread learned ALLOW is worth more than DENY, and ends holding ALLOW.
    assert q0_allow > q0_deny
    assert g._arm[0] == _ALLOW

    # (b) decoupling: the productive thread's ALLOW value is well above the unproductive thread's ALLOW value
    # (independent learners reached different conclusions from per-thread evidence). A shared-parameter method
    # would have ONE allow value; these differ by the productivity gap the per-thread credit captured.
    assert q0_allow > q1_allow + 1.0


def test_contextual_bins_partition_submit_pressure() -> None:
    """The optional tabular-contextual mode partitions submit_pressure into context_bins coarse bins and keeps
    an INDEPENDENT 2-arm table per (thread, bin). A low-pressure forward and a high-pressure forward land a
    thread in DIFFERENT bins, so the bandit can learn a regime-specific arm. Assert the bin axis is sized and
    that the live submit_pressure drives the recorded bin (low pressure -> bin 0, high pressure -> a higher
    bin)."""
    g = IndependentBanditGate(context_bins=3)
    t = 2
    g.reset(_ctx(n_threads=t, d=8))
    assert g._Q.shape == (t, 3, 2)               # per-(thread,bin) arm table
    # low submit_pressure: ready small vs headroom -> bin 0.
    g.act(_obs(t=t, inflight=[0, 0], ready=[0, 0], leaves=[0, 0], served=[0, 1], now=1.0))
    assert g._arm_bin.tolist() == [0, 0]
    # high submit_pressure: ready large and headroom small (near the ceiling) -> a higher bin than 0.
    g.act(_obs(t=t, inflight=[7, 7], ready=[40, 40], leaves=[0, 0], served=[0, 1], now=2.0))
    assert int(g._arm_bin[0]) > 0 and int(g._arm_bin[1]) > 0
