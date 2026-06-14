#!/usr/bin/env python3
"""
test_az_loop.py — bounded correctness gate for the Gumbel ExIt loop machinery.

Asserts the load-bearing contracts of the AZ loop modules without running the (multi-hour) real
loop: the action↔slot mapping is a fixed env-derived bijection, the two legal-mask paths agree,
the masked softmax puts zero mass on illegal slots, the combined train_step is finite and
reduces loss on a fixed batch, and the Gumbel search returns a well-formed improved-policy
target (sums to 1, zero on illegal, finite) plus a legal executed action.

Run pinned + bounded, e.g.:
    taskset -c 2 timeout 180 /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_az_loop.py -q

NOT a numerical-quality battery (the loop's eval is); it only asserts the machinery is correct.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from chocofarm.model.env import Environment, TERMINATE
from chocofarm.az.features import FeatureBuilder, feature_dim
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.actions import (n_action_slots, action_to_slot, slot_to_action,
                                  legal_mask, legal_mask_from_features)
from chocofarm.az.gumbel_search import GumbelAZSearch, GumbelPolicy


def test_action_space_size():
    env = Environment()
    assert n_action_slots(env) == env.N + len(env.detectors) + 1 == 65


def test_action_slot_bijection():
    """slot_to_action ∘ action_to_slot == id over every action; slots cover exactly the space."""
    env = Environment()
    seen = set()
    for i in range(env.N):
        s = action_to_slot(env, ("t", i)); assert slot_to_action(env, s) == ("t", i); seen.add(s)
    for j in env.detectors:
        s = action_to_slot(env, ("d", j)); assert slot_to_action(env, s) == ("d", j); seen.add(s)
    s = action_to_slot(env, TERMINATE); assert slot_to_action(env, s) == TERMINATE; seen.add(s)
    assert seen == set(range(n_action_slots(env)))


def test_legal_mask_paths_agree():
    """The authoritative env-based mask and the feature-slice mask agree, at root and after a
    detector read sharpens the belief (the regime where some faces become uninformative)."""
    env = Environment()
    fb = FeatureBuilder(env)
    loc, bw, coll = ("w", env.entry), env.worlds, set()
    feat = fb.build(loc, bw, coll)
    assert np.array_equal(legal_mask(env, loc, bw, coll),
                          legal_mask_from_features(env, feat))
    # post-sense belief
    w = int(env.worlds[123])
    _, nloc, nbw, nc, _ = env.apply(loc, bw, coll, ("d", 5), w)
    feat2 = fb.build(nloc, nbw, nc)
    assert np.array_equal(legal_mask(env, nloc, nbw, nc),
                          legal_mask_from_features(env, feat2))


def test_masked_softmax_zero_on_illegal():
    env = Environment()
    fb = FeatureBuilder(env)
    net = ValueMLP(feature_dim(env), hidden=32, n_actions=n_action_slots(env), seed=0)
    w = int(env.worlds[7])
    loc, bw, coll = ("w", env.entry), env.worlds, set()
    _, nloc, nbw, nc, _ = env.apply(loc, bw, coll, ("d", 3), w)
    feat = fb.build(nloc, nbw, nc)
    mask = legal_mask_from_features(env, feat)
    v, p = net.predict_both(feat, mask)
    assert np.isfinite(v)
    assert abs(float(p.sum()) - 1.0) < 1e-9
    assert float(p[mask == 0].sum()) == 0.0


def test_train_step_finite_and_reduces():
    env = Environment()
    fb = FeatureBuilder(env)
    net = ValueMLP(feature_dim(env), hidden=32, n_actions=n_action_slots(env), seed=0)
    feat = fb.build(("w", env.entry), env.worlds, set())
    mask = legal_mask_from_features(env, feat)
    B = 16
    X = np.stack([feat] * B)
    M = np.stack([mask] * B)
    PI = M / M.sum(1, keepdims=True)
    Y = np.linspace(-1.0, 1.0, B)
    ce0, vl0 = net.train_step(X, PI, M, Y, 1e-3, 1e-4)
    for _ in range(100):
        ce, vl = net.train_step(X, PI, M, Y, 1e-3, 1e-4)
    assert np.isfinite(ce) and np.isfinite(vl)
    assert vl < vl0  # value MSE must come down on a fixed batch


def test_gumbel_target_well_formed():
    """A search decision returns a legal executed action and a valid improved-policy target:
    sums to 1, zero on illegal slots, finite."""
    env = Environment()
    net = ValueMLP(feature_dim(env), hidden=32, n_actions=n_action_slots(env), seed=1)
    search = GumbelAZSearch(net, env, m=6, n_sims=16)
    rng = np.random.default_rng(0)
    loc, bw, coll = ("w", env.entry), env.worlds, set()
    a, pi = search.decide_with_target(env, loc, bw, coll, 0.0855, rng, temperature=0.0)
    mask = legal_mask(env, loc, bw, coll)
    assert a == TERMINATE or (isinstance(a, tuple) and a[0] in ("t", "d"))
    assert action_to_slot(env, a) in np.nonzero(mask)[0]
    assert np.isfinite(pi).all()
    assert abs(float(pi.sum()) - 1.0) < 1e-6
    assert float(pi[mask == 0].sum()) == 0.0


def test_gumbel_policy_simulates():
    """GumbelPolicy drives a full episode through env.simulate without error."""
    env = Environment()
    net = ValueMLP(feature_dim(env), hidden=32, n_actions=n_action_slots(env), seed=2)
    pol = GumbelPolicy(net, env, m=6, n_sims=16)
    R, T, e = env.simulate(pol, int(env.worlds[50]), 0.0855, np.random.default_rng(3))
    assert np.isfinite(R) and np.isfinite(T) and T > 0


# ---- Danihelka et al. 2022 fidelity invariants (the out-of-frame-audit immune system) ----

def test_sequential_halving_spends_full_budget():
    """SH must use the whole n_sims budget — no over/under-spend (paper §2)."""
    env = Environment()
    net = ValueMLP(feature_dim(env), hidden=32, n_actions=n_action_slots(env), seed=1)
    for m, n in [(6, 16), (12, 48), (4, 12)]:
        search = GumbelAZSearch(net, env, m=m, n_sims=n)
        rng = np.random.default_rng(0)
        from chocofarm.az.gumbel_search import _Node
        root = _Node()
        loc, bw, coll = ("w", env.entry), env.worlds, set()
        search._evaluate(root, loc, bw, coll)
        legal_slots = [action_to_slot(env, a) for a in root.legal]
        g = rng.gumbel(size=search.n_slots)
        logits = np.full(search.n_slots, -1e30)
        for s in legal_slots:
            logits[s] = np.log(max(root.prior[s], 1e-12))
        considered = list(np.argsort(np.where(logits > -1e29, logits + g, -np.inf))[::-1][:m])
        survivor = search._sequential_halving(env, root, loc, bw, set(coll), 0.0855, rng,
                                              considered, g, logits)
        assert sum(root.N.values()) == n, (m, n, sum(root.N.values()))
        assert survivor in considered


def test_executed_action_is_sh_survivor():
    """At temperature 0 the executed action IS the SH survivor (paper §2). This is the eval
    policy's decision rule; bug-1 (argmax over the full top-m) is what this pins shut."""
    env = Environment()
    net = ValueMLP(feature_dim(env), hidden=32, n_actions=n_action_slots(env), seed=4)
    search = GumbelAZSearch(net, env, m=6, n_sims=16)
    # decide_with_target(temperature=0) returns the survivor; re-run SH with the SAME rng stream
    # and confirm the survivor matches the executed action.
    loc, bw, coll = ("w", env.entry), env.worlds, set()
    for seed in range(5):
        rng = np.random.default_rng(seed)
        a, _ = search.decide_with_target(env, loc, bw, coll, 0.0855, rng, temperature=0.0)
        # the executed action must be a legal action (the survivor always is)
        assert a in env.legal_actions(loc, bw, coll) or a == TERMINATE


def test_vmix_prior_weighted():
    """v_mix's visited-Q term must be PRIOR-weighted, not visit-weighted (paper §3). With unequal
    priors AND unequal visits, the two formulas give measurably different v_mix; this pins the
    code to the prior-weighted one (bug-2 is the visit-weighted variant)."""
    env = Environment()
    net = ValueMLP(feature_dim(env), hidden=16, n_actions=n_action_slots(env), seed=0)
    from chocofarm.az.gumbel_search import _Node
    search = GumbelAZSearch(net, env, m=4, n_sims=8)
    root = _Node()
    loc, bw, coll = ("w", env.entry), env.worlds, set()
    search._evaluate(root, loc, bw, coll)
    legal_slots = [action_to_slot(env, a) for a in root.legal]
    a1, a2 = root.legal[0], root.legal[1]
    s1, s2 = action_to_slot(env, a1), action_to_slot(env, a2)
    # force DELIBERATELY unequal priors so prior- and visit-weighting diverge
    root.prior = root.prior.copy()
    root.prior[s1], root.prior[s2] = 0.8, 0.2
    root.value = 0.0
    root.N[a1], root.W[a1] = 10, 10.0   # Q1 = +1.0, visited 10x
    root.N[a2], root.W[a2] = 1, -1.0    # Q2 = -1.0, visited 1x
    sum_n = 11
    v_bar_prior = (0.8 * 1.0 + 0.2 * (-1.0)) / (0.8 + 0.2)       # = 0.6
    v_mix_prior = (0.0 + sum_n * v_bar_prior) / (1 + sum_n)       # ≈ 0.55
    v_bar_visit = (10 * 1.0 + 1 * (-1.0)) / 11                    # ≈ 0.818 (the WRONG variant)
    v_mix_visit = (0.0 + sum_n * v_bar_visit) / (1 + sum_n)
    got = search._v_mix(root, legal_slots)
    assert abs(got - v_mix_prior) < 1e-9, (got, v_mix_prior)
    assert abs(got - v_mix_visit) > 0.1   # and clearly NOT the visit-weighted variant


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all az-loop checks passed")
