#!/usr/bin/env python3
"""
tests/test_optimizer_hp_roundtrip.py — pins the two §3 lying-signature / contract fixes.

Two annotation-only fixes are pinned here (behavior-preserving — these assert the runtime contract
the now-honest annotations describe):

  * `JaxTrainer.train_step` / `train_step_value` accept `hp=None` (→ uses the construction-time
    `self._default_hp`) AND an explicit `AdamHParams`. The old `hp: AdamHParams = None` annotation
    lied (a non-Optional param defaulted to None); the body always handled None. These round-trips
    prove both forms are accepted (ADR-0002 — the signature now matches the body).

  * `AdamHParams` declares `lr/b1/b2/eps` as `float | jax.Array`: the python-float default-construct
    path holds floats, the `_hp_arrays` `jnp.asarray(...)` path holds traced jax Arrays. This pins
    that `_hp_arrays` genuinely produces an `AdamHParams` whose four fields are jax Arrays (the
    second construction form the widened annotation now admits).

Skips gracefully (does not fail) if jax / the jax-trainer stack is not importable, mirroring how
`tests/test_cpp_runner.py` skips without its binary.

Public Domain (The Unlicense).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from chocofarm.az.optimizer import AdamHParams
from chocofarm.az.mlp_jax_train import JaxTrainer
from chocofarm.az.mlp import ValueMLP
from chocofarm.model.env import Environment
from chocofarm.az.features import feature_dim
from chocofarm.az.actions import n_action_slots


def _policy_trainer(seed=0):
    env = Environment()
    in_dim, na = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=32, n_actions=na, seed=seed, residual=False)
    net.set_value_scale(0.0, 1.0)
    return JaxTrainer(net, lr=1e-3), in_dim, na


def _value_trainer(seed=0):
    env = Environment()
    in_dim = feature_dim(env)
    net = ValueMLP(in_dim, hidden=32, n_actions=None, seed=seed, residual=False)
    net.set_value_scale(0.0, 1.0)
    return JaxTrainer(net, lr=1e-3), in_dim


def _policy_batch(in_dim, na, B=8, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((B, in_dim)).astype(np.float32)
    target_pi = np.full((B, na), 1.0 / na, dtype=np.float32)
    legal_mask = np.ones((B, na), dtype=np.float32)
    target_v = rng.standard_normal(B).astype(np.float32)
    return X, target_pi, legal_mask, target_v


def test_hp_arrays_produces_adamhparams_of_jax_arrays():
    """`_hp_arrays(AdamHParams of floats)` → an `AdamHParams` whose four fields are jax Arrays — the
    second construction form the `float | jax.Array` widening (the §3 P1 contract fix) admits."""
    trainer, _, _ = _policy_trainer()
    hp_f = AdamHParams(lr=1e-3, b1=0.9, b2=0.999, eps=1e-8)
    hp_a = trainer._hp_arrays(hp_f)
    assert isinstance(hp_a, AdamHParams)
    for field in ("lr", "b1", "b2", "eps"):
        v = getattr(hp_a, field)
        assert isinstance(v, jax.Array), f"{field} is {type(v)!r}, expected a jax.Array"
    # the values round-trip (the cast is value-preserving to the optax dtype)
    assert float(hp_a.lr) == pytest.approx(1e-3)
    assert float(hp_a.eps) == pytest.approx(1e-8)


def test_train_step_accepts_none_and_explicit_hp():
    """`train_step(hp=None)` uses `_default_hp`; an explicit `AdamHParams` is also accepted (the §3 P0
    honest-None fix — both forms round-trip and return the (ce, vmse) float pair)."""
    trainer, in_dim, na = _policy_trainer()
    X, target_pi, legal_mask, target_v = _policy_batch(in_dim, na)

    # hp=None → the construction-time _default_hp
    ce0, vmse0 = trainer.train_step(X, target_pi, legal_mask, target_v, hp=None)
    assert isinstance(ce0, float) and isinstance(vmse0, float)
    assert np.isfinite(ce0) and np.isfinite(vmse0)

    # explicit AdamHParams (of python floats) — the live-hp path
    hp = AdamHParams(lr=5e-4, b1=0.9, b2=0.999, eps=1e-8)
    ce1, vmse1 = trainer.train_step(X, target_pi, legal_mask, target_v, hp=hp)
    assert isinstance(ce1, float) and isinstance(vmse1, float)
    assert np.isfinite(ce1) and np.isfinite(vmse1)


def test_train_step_value_accepts_none_and_explicit_hp():
    """`train_step_value(hp=None)` uses `_default_hp`; an explicit `AdamHParams` is also accepted (the
    §3 P0 honest-None fix on the value-only path)."""
    trainer, in_dim = _value_trainer()
    rng = np.random.default_rng(2)
    X = rng.standard_normal((8, in_dim)).astype(np.float32)
    target_v = rng.standard_normal(8).astype(np.float32)

    vmse0 = trainer.train_step_value(X, target_v, hp=None)
    assert isinstance(vmse0, float) and np.isfinite(vmse0)

    hp = AdamHParams(lr=5e-4, b1=0.9, b2=0.999, eps=1e-8)
    vmse1 = trainer.train_step_value(X, target_v, hp=hp)
    assert isinstance(vmse1, float) and np.isfinite(vmse1)
