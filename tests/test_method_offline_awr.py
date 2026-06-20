#!/usr/bin/env python3
"""
test_method_offline_awr.py — unit gate for the AWR OFFLINE-RL controller
(cpp/stage_a/control_lab/methods/offline_awr.py), a REINFORCEMENT-LEARNING candidate for the issue-gate
control lab.

Imports the method's OWN submodule directly (NOT the methods package, and WITHOUT load_all()): sibling
method files are authored in parallel, so importing only `control_lab.methods.offline_awr` keeps this test
isolated. Pins the FROZEN adapter.Controller contract (reset / observe / act / metrics shape) plus AWR's
defining OFFLINE-RL behavior:

  - AWRGate.act() returns a length-T list in {0,1} (the per-thread allow bits, DETERMINISTIC — argmax, not a
    sample);
  - observe() is a no-op (offline — nothing learned at runtime) and never raises;
  - reset() captures the trial geometry + re-zeros the served-diff baselines WITHOUT touching the frozen
    policy params (the fitted weights survive a reset);
  - the inflight==0 liveness override force-allows (DENY-ONLY semantics);
  - the served-thread first-difference for the coalescence feature honors the wire subtlety (an ABSENT thread
    is never differenced; its baseline is untouched) — and matches the trainer's offline reconstruction;
  - a checkpoint round-trips (save -> load reproduces the same gating);
  - THE OFFLINE-LEARNING ASSERTION (method-specific): AWRRecipe.fit on a synthetic corpus where DENY earns a
    higher return than ALLOW learns a policy that DENIES (the learned allow fraction drops well below the raw
    observed allow fraction), and the control corpus (ALLOW earns more) keeps the policy allowing — the two
    regimes separate, which is the convoy-taming signal the real corpus carries.

Run pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_offline_awr.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

_STAGE_A = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp", "stage_a"
)
if _STAGE_A not in sys.path:
    sys.path.insert(0, _STAGE_A)

from control_lab.adapter import Observation, TrialContext  # noqa: E402
from control_lab.methods import offline_awr as awr  # noqa: E402  (own submodule, NOT the package)


def _ctx(n_threads: int = 4, d: int = 8, k: int = 64, seed: int = 0) -> TrialContext:
    return TrialContext(n_threads=n_threads, d_ceiling=d, k_per_thread=k, s_min=1,
                        chunk_floor=True, seed=seed)


def _obs(*, n_threads: int, inflight, ready, msgs=None, leaves=None, served=None, t: float = 0.0) -> Observation:
    if served is None:
        served = list(range(n_threads))
    if msgs is None:
        msgs = [0] * n_threads
    if leaves is None:
        leaves = [0] * n_threads
    features = {
        "n_threads": n_threads, "d_ceiling": 8, "server_rows_per_forward": float(sum(ready)),
        "inflight": inflight, "ready": ready, "msgs": msgs, "leaves": leaves, "rtt_us": [0] * n_threads,
    }
    return Observation(features=features, served=served, forward_rows=sum(ready), t_monotonic=t)


def _tiny_gate(hidden: int = 0) -> awr.AWRGate:
    """An AWRGate built from explicit allow-leaning params (no fit) for the pure-contract tests."""
    if hidden <= 0:
        params = {"w": np.zeros(awr._D_IN, dtype=np.float32), "b": np.float32(2.0)}  # allow-leaning constant
    else:
        params = {"w1": np.zeros((awr._D_IN, hidden), np.float32), "b1": np.zeros(hidden, np.float32),
                  "w2": np.zeros(hidden, np.float32), "b2": np.float32(2.0)}
    return awr.AWRGate(params=params, hidden=hidden)


# ----------------------------------------------------------------------------- contract


def test_reset_and_metrics_shape():
    c = _tiny_gate()
    c.reset(_ctx(n_threads=4))
    m = c.metrics()
    for key in ("mean_allow_prob", "allow_frac"):
        assert key in m


def test_act_returns_length_t_binary_and_deterministic():
    """act() returns a length-T list of {0,1}, and is DETERMINISTIC (offline policy argmax — same input, same
    output across calls)."""
    T = 4
    c = _tiny_gate()
    c.reset(_ctx(n_threads=T))
    o = _obs(n_threads=T, inflight=[1] * T, ready=[3] * T)
    out1 = c.act(o)
    out2 = c.act(o)
    assert isinstance(out1, list) and len(out1) == T
    assert all(v in (0, 1) for v in out1)
    assert out1 == out2, "the deployed offline policy is deterministic (argmax, not a sample)"


def test_liveness_override_forces_allow_at_zero_inflight():
    """A thread with inflight==0 is an UNGATED forced flush -> force-allow regardless of the policy. Use a
    deny-leaning policy so we KNOW the override is what produced the allow."""
    T = 5
    params = {"w": np.zeros(awr._D_IN, np.float32), "b": np.float32(-5.0)}  # deny-leaning
    c = awr.AWRGate(params=params, hidden=0)
    c.reset(_ctx(n_threads=T))
    out = c.act(_obs(n_threads=T, inflight=[0] * T, ready=[2, 0, 5, 1, 3]))
    assert out == [1] * T, "inflight==0 forces allow on every thread (liveness)"
    # and with inflight>0 the deny-leaning policy denies (so we know the override, not the policy, did it above)
    out2 = c.act(_obs(n_threads=T, inflight=[1] * T, ready=[2] * T))
    assert out2 == [0] * T


def test_observe_is_noop_and_safe():
    """observe() is a no-op (offline) and never raises — including non-finite rewards and rewards before act."""
    T = 3
    c = _tiny_gate()
    c.reset(_ctx(n_threads=T))
    c.observe(123.4, {})
    c.observe(float("nan"), {})
    c.observe(float("inf"), {})
    out = c.act(_obs(n_threads=T, inflight=[1] * T, ready=[2] * T))
    assert len(out) == T and all(v in (0, 1) for v in out)


def test_reset_does_not_touch_frozen_params():
    """reset() re-zeros the per-thread baselines but MUST NOT alter the fitted policy params (the deployed
    weights survive a trial swap)."""
    c = _tiny_gate(hidden=4)
    before = {k: np.array(v, copy=True) for k, v in c._np_params.items()}
    c.reset(_ctx(n_threads=7))
    for k, v in before.items():
        assert np.allclose(c._np_params[k], v), f"param {k} changed across reset (must be frozen)"


def test_invalid_config_fails_loud():
    """ADR-0002: degenerate recipe hyperparameters are CONSTRUCTION errors at the ctor."""
    with pytest.raises(ValueError):
        awr.AWRRecipe(temp=0.0)
    with pytest.raises(ValueError):
        awr.AWRRecipe(w_max=0.0)
    with pytest.raises(ValueError):
        awr.AWRRecipe(value_steps=0)
    with pytest.raises(ValueError):
        awr.AWRRecipe(hidden=-1)
    with pytest.raises(ValueError):
        awr.AWRRecipe(lr_policy=0.0)


def test_empty_corpus_fails_loud():
    with pytest.raises(ValueError):
        awr.AWRCorpus(phi=np.zeros((0, awr._D_IN), np.float32),
                      action=np.zeros((0,), np.float32), ret=np.zeros((0,), np.float32))


def test_missing_checkpoint_fails_loud():
    with pytest.raises(FileNotFoundError):
        awr.load_checkpoint("/nonexistent/dir/awr_nope.npz")


# ----------------------------------------------------------------------------- wire subtlety


def test_absent_thread_is_not_first_differenced():
    """A thread ABSENT from a forward reads a sentinel-0 cumulative counter, so it must NOT be first-differenced
    and its baseline must NOT advance (identical to reinforce/a2c — and to the trainer's reconstruction)."""
    T = 2
    c = _tiny_gate()
    c.reset(_ctx(n_threads=T))
    c.act(_obs(n_threads=T, inflight=[1, 1], ready=[1, 1], msgs=[10, 20], leaves=[40, 80], served=[0, 1]))
    assert int(c._msgs_prev[0]) == 10 and int(c._msgs_prev[1]) == 20
    c.act(_obs(n_threads=T, inflight=[1, 1], ready=[1, 1], msgs=[15, 0], leaves=[60, 0], served=[0]))
    assert int(c._msgs_prev[0]) == 15, "served thread's baseline tracks its true reading"
    assert int(c._msgs_prev[1]) == 20, "absent thread's baseline untouched (sentinel-0 never differenced)"
    assert int(c._leaves_prev[1]) == 80


def test_grown_T_does_not_throw():
    """Live-T robustness: act() entered with a longer feature frame than reset's T must not throw (the lab
    server can grow T past reset)."""
    c = _tiny_gate()
    c.reset(_ctx(n_threads=2))
    # the server may grow the gate vector; act() must grow its baselines on demand and stay total.
    c._t = 4   # simulate the server having grown T
    out = c.act(_obs(n_threads=4, inflight=[1, 1, 1, 1], ready=[2, 2, 2, 2]))
    assert len(out) == 4 and all(v in (0, 1) for v in out)


# ----------------------------------------------------------------------------- checkpoint round-trip


def test_checkpoint_round_trips():
    """save_checkpoint -> load_checkpoint reproduces the same gating (the deploy path the factory uses)."""
    rng = np.random.default_rng(0)
    params = {"w1": rng.standard_normal((awr._D_IN, 4)).astype(np.float32),
              "b1": rng.standard_normal(4).astype(np.float32),
              "w2": rng.standard_normal(4).astype(np.float32), "b2": np.float32(0.3)}
    gate = awr.AWRGate(params=params, hidden=4, fit_diag={"adv_mean": 1.0, "learned_active_allow_frac": 0.2})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.npz")
        awr.save_checkpoint(path, gate)
        reloaded = awr.load_checkpoint(path)
    probe = rng.standard_normal((256, awr._D_IN)).astype(np.float32)
    p0 = awr._np_policy_probs(gate._np_params, probe, 4)
    p1 = awr._np_policy_probs(reloaded._np_params, probe, 4)
    assert np.allclose(p0, p1), "reloaded policy reproduces the fitted gating"
    assert reloaded.metrics()["fit_learned_active_allow_frac"] == pytest.approx(0.2)


# ----------------------------------------------------------------------------- the offline-learning assertion


def _synthetic_corpus(deny_is_good: bool, n: int = 6000, seed: int = 0) -> awr.AWRCorpus:
    """A synthetic per-thread corpus that mimics the convoy structure: feature 2 (inflight_saturation) is the
    discriminating feature; the OBSERVED action mixes deny/allow ~50/50, and the RETURN is high when the action
    matches the good direction on active (high-inflight) rows. AWR should learn to take the good action."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 1.0, size=(n, awr._D_IN)).astype(np.float32)
    phi[:, -1] = 1.0   # bias column
    act = (rng.random(n) < 0.5).astype(np.float32)   # ~50/50 observed gates (the off-policy mix)
    # return: a base + a bonus when the action equals the good action. deny_is_good -> deny (a==0) pays.
    good_action = 0.0 if deny_is_good else 1.0
    matches = (act == good_action).astype(np.float32)
    ret = (20.0 + 40.0 * matches + rng.normal(0.0, 3.0, size=n)).astype(np.float32)
    return awr.AWRCorpus(phi=phi, action=act, ret=ret)


def test_awr_learns_to_deny_when_deny_is_rewarded():
    """AWR OFFLINE LEARNING (the defining behavior): on a corpus where DENY earns the higher return, the fitted
    policy DENIES (learned allow fraction well below the ~0.5 observed), and on the control corpus (ALLOW
    rewarded) it ALLOWS. The two regimes separate — the convoy-taming signal."""
    recipe_deny = awr.AWRRecipe(hidden=8, temp=1.0, w_max=20.0, value_steps=800, policy_steps=800, seed=1)
    gate_deny = recipe_deny.fit(_synthetic_corpus(deny_is_good=True, seed=1))
    laf_deny = gate_deny.metrics()["fit_learned_active_allow_frac"]

    recipe_allow = awr.AWRRecipe(hidden=8, temp=1.0, w_max=20.0, value_steps=800, policy_steps=800, seed=1)
    gate_allow = recipe_allow.fit(_synthetic_corpus(deny_is_good=False, seed=1))
    laf_allow = gate_allow.metrics()["fit_learned_active_allow_frac"]

    # the deny-rewarded fit must gate hard toward deny; the allow-rewarded fit toward allow.
    assert laf_deny < 0.4, f"deny-rewarded corpus -> policy denies (allow_frac={laf_deny:.3f} should be low)"
    assert laf_allow > 0.6, f"allow-rewarded corpus -> policy allows (allow_frac={laf_allow:.3f} should be high)"
    assert laf_deny < laf_allow, "the two reward regimes separated (AWR learned the rewarded direction)"
    # the value baseline reduced its MSE (the fit actually trained).
    assert recipe_deny.curves["value_mse"][-1] < recipe_deny.curves["value_mse"][0]


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn) and "fails_loud" not in _name:
            _fn()
            print(f"PASS {_name}")
    print("all offline_awr method checks passed (run via pytest for the fail-loud config tests)")
