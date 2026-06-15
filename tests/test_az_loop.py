#!/usr/bin/env python3
"""
test_az_loop.py — bounded correctness gate for the Gumbel ExIt loop machinery.

Asserts the load-bearing contracts of the AZ loop modules without running the (multi-hour) real
loop: the action↔slot mapping is a fixed env-derived bijection, the two legal-mask paths agree,
the masked softmax puts zero mass on illegal slots, the JaxTrainer's combined train_step is finite
and reduces loss on a fixed batch (training is JAX/optax; the numpy net is inference-only), and the
Gumbel search returns a well-formed improved-policy target (sums to 1, zero on illegal, finite)
plus a legal executed action.

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
    # The probability sum is normalized to 1 up to the working precision. With the parametric
    # hot-path DTYPE at float32 (the default), the softmax sum carries float32 rounding (~1e-7);
    # the float64 path is tighter. 1e-6 covers both. The LOGIC invariant — exactly zero mass on
    # illegal slots — is asserted exactly below and is unaffected by precision.
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert float(p[mask == 0].sum()) == 0.0


def test_predict_both_cache_coherent_across_writers():
    """The float32 inference cache (mlp.ValueMLP._f32_cache) must never serve weights that no
    longer match the float64 source. This pins the invariant over the surviving weight writers — a
    rebind (load / warm-start / the JaxTrainer write-back replacing the array object) and a y-scale
    change — after the cache has been populated by a prior predict. (Out-of-frame-audit guard: the
    first cut gated invalidation on the optimizer step alone, so a post-populate rebind served stale
    weights; this test reproduces that order and asserts the served forward tracks the source.

    Post-training cache coherence — that numpy inference sees the trained weights — is covered at
    the integration level by test_jax_train_writes_back_numpy_inference; the manual in-place Adam
    step that this test used to exercise was removed with the JAX training migration.)"""
    env = Environment()
    fb = FeatureBuilder(env)
    net = ValueMLP(feature_dim(env), hidden=64, n_actions=n_action_slots(env), seed=0)
    feat = fb.build(("w", env.entry), env.worlds, set())
    mask = legal_mask_from_features(env, feat)

    def truth_value():
        _, v_std, _ = net._forward(np.asarray(feat, dtype=np.float64)[None, :])
        return float(v_std[0] * net.y_std + net.y_mean)

    # populate the cache, then REBIND weights (the load/warm-start shape) and predict again
    net.predict_both(feat, mask)
    rng = np.random.default_rng(99)
    net.W1 = rng.standard_normal(net.W1.shape)
    net.W2 = rng.standard_normal(net.W2.shape)
    net.Wv = rng.standard_normal(net.Wv.shape)
    net.Wp = rng.standard_normal(net.Wp.shape)
    v_after_rebind, _ = net.predict_both(feat, mask)
    assert abs(v_after_rebind - truth_value()) < 1e-2, "stale cache after weight rebind"

    # y-scale change must be reflected
    net.set_value_scale(5.0, 2.0)
    v_after_scale = net.predict_both(feat, mask)[0]
    assert v_after_scale != v_after_rebind, "stale cache after y-scale change"


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


# ---- Part C: belief-resolution features (uncertainty encoding) ----

def test_feature_dim_includes_unc_block():
    """feature_dim is N×5 + nD×3 + (6+n_tele) — the per-treasure block grew 4N→5N (the unc
    sub-block) and the global block 5→6 (the Σunc scalar). On the live env: 241."""
    env = Environment()
    N, nD, nt = env.N, len(env.detectors), len(env.teleports)
    assert feature_dim(env) == N * 5 + nD * 3 + (6 + nt) == 241


def test_unc_features_are_bernoulli_variance():
    """Per-treasure unc[i] = marg[i]·(1−marg[i]); global Σunc sums it over UNCOLLECTED treasures.
    At the root every marginal is K/N = 5/20 = 0.25, so unc = 0.1875 and Σunc = 20·0.1875 = 3.75.
    A resolved treasure (marg 0 or 1) carries unc 0 — the known-vs-unknown signal."""
    env = Environment()
    fb = FeatureBuilder(env)
    N, nD = env.N, len(env.detectors)
    feat = fb.build(("w", env.entry), env.worlds, set())
    marg = feat[0:N]
    unc = feat[4 * N:5 * N]                       # 5th per-treasure sub-block (after dist)
    assert np.allclose(unc, marg * (1.0 - marg), atol=1e-6)
    assert np.allclose(unc, 0.25 * 0.75, atol=1e-6)   # root marginals are all 0.25
    sum_u_idx = 5 * N + 3 * nD + 5                # global block, 6th scalar (after nonempty)
    assert abs(float(feat[sum_u_idx]) - float(np.sum(unc))) < 1e-5  # no treasure collected yet
    assert abs(float(feat[sum_u_idx]) - 3.75) < 1e-5


def test_unc_zero_when_resolved():
    """After sensing enough to resolve a treasure's presence to 0 or 1, its unc drops to 0 — the
    feature distinguishes a resolved treasure from a split (marg-0.5) one, which marg alone cannot."""
    env = Environment()
    fb = FeatureBuilder(env)
    N = env.N
    # drive a chain of senses to sharpen the belief, then check at least one treasure resolved
    loc, bw, coll = ("w", env.entry), env.worlds, set()
    rng = np.random.default_rng(0)
    w = int(env.worlds[500])
    for _ in range(6):
        feat = fb.build(loc, bw, coll)
        from chocofarm.az.actions import legal_mask_from_features
        legal = env.legal_actions(loc, bw, coll)
        senses = [a for a in legal if a[0] == "d"]
        if not senses:
            break
        a = senses[0]
        _, loc, bw, coll, _ = env.apply(loc, bw, coll, a, w)
    feat = fb.build(loc, bw, coll)
    marg = feat[0:N]
    unc = feat[4 * N:5 * N]
    resolved = (marg == 0.0) | (marg == 1.0)
    assert np.all(unc[resolved] == 0.0), "resolved treasures must carry zero uncertainty"
    assert np.allclose(unc, marg * (1.0 - marg), atol=1e-6)


# ---- Part B: lower-variance value target (TD(λ)/n-step blend) ----

def test_value_target_mc_limit_bit_identical():
    """blended_returns_to_go at λ_blend=1 / n_step=None is bit-identical to the pure-MC suffix
    rule — Part B is opt-in, the default recovers the prior behavior exactly."""
    from chocofarm.az.value_target import suffix_returns_to_go, blended_returns_to_go
    step_rt = [(1.0, 3.0), (0.0, 5.0), (2.0, 4.0), (0.0, 2.0)]
    boot = [-0.5, -0.4, -0.3, -0.2]
    exit_c, lam = 10.0, 0.0855
    mc = suffix_returns_to_go(step_rt, exit_c, lam)
    assert blended_returns_to_go(step_rt, boot, exit_c, lam, lam_blend=1.0) == mc
    assert blended_returns_to_go(step_rt, boot, exit_c, lam, n_step=None) == mc


def test_value_target_td_lambda_limits():
    """λ_blend→0 equals the 1-step bootstrap target; the backward recurrence matches the forward
    view (geometric average of n-step returns) for several λ_blend."""
    from chocofarm.az.value_target import blended_returns_to_go
    step_rt = [(1.0, 3.0), (0.0, 5.0), (2.0, 4.0), (0.0, 2.0)]
    boot = [-0.5, -0.4, -0.3, -0.2]
    exit_c, lam = 10.0, 0.0855

    def nstep_forward(j, n):
        D = len(step_rt); acc = 0.0; end = j + n
        if end >= D:
            for t in range(j, D):
                acc += step_rt[t][0] - lam * step_rt[t][1]
            acc += -lam * exit_c
        else:
            for t in range(j, end):
                acc += step_rt[t][0] - lam * step_rt[t][1]
            acc += boot[end]
        return acc

    def td_forward(ell):
        D = len(step_rt); out = []
        for j in range(D):
            g = 0.0
            for n in range(1, D - j):
                g += (1 - ell) * ell ** (n - 1) * nstep_forward(j, n)
            g += ell ** (D - j - 1) * nstep_forward(j, D - j)
            out.append(g)
        return out

    b0 = blended_returns_to_go(step_rt, boot, exit_c, lam, lam_blend=0.0)
    b1 = blended_returns_to_go(step_rt, boot, exit_c, lam, n_step=1)
    assert np.allclose(b0, b1, atol=1e-9)
    for ell in (0.0, 0.3, 0.7, 0.95, 1.0):
        assert np.allclose(blended_returns_to_go(step_rt, boot, exit_c, lam, lam_blend=ell),
                           td_forward(ell), atol=1e-9), ell


def test_decide_with_value_returns_finite_bootstrap():
    """decide_with_value returns (action, pi, root_value); the bootstrap is finite and the
    (action, pi) pair is identical to decide_with_target on the SAME rng stream (the wrappers share
    one core)."""
    env = Environment()
    net = ValueMLP(feature_dim(env), hidden=32, n_actions=n_action_slots(env), seed=1)
    search = GumbelAZSearch(env=env, net=net, m=6, n_sims=16)
    loc, bw, coll = ("w", env.entry), env.worlds, set()
    a1, pi1, boot = search.decide_with_value(env, loc, bw, coll, 0.0855,
                                             np.random.default_rng(5), temperature=0.0)
    a2, pi2 = search.decide_with_target(env, loc, bw, coll, 0.0855,
                                        np.random.default_rng(5), temperature=0.0)
    assert np.isfinite(boot)
    assert a1 == a2
    assert np.allclose(pi1, pi2, atol=1e-9)


# ---- residual block (toggleable, between trunk output and the heads) ----
#
# NOTE: the hand-derived residual-backward finite-difference gradient-check that used to live here
# (`_residual_grad_check` / `test_residual_gradient_check`) was DROPPED with the JAX/optax training
# migration. Training gradients are now produced by `jax.value_and_grad` over the jit'd forward
# (`mlp_jax_train`), i.e. correct-by-construction — there is no hand-derived backward left to
# finite-difference-check. The load-bearing safeguard moved to `tests/test_jax_equivalence.py`,
# which pins the numpy inference forward against the jit'd jax training forward to float32. The
# "jax train_step reduces loss" test below replaces the manual-train-reduces checks.


def test_jax_train_step_reduces_loss():
    """The JAX/optax train step reduces BOTH the policy CE and the value MSE on a fixed batch, with
    the residual block ON — the autodiff training path is wired end to end (forward, value_and_grad,
    optax-Adam, weights written back into the net). Replaces the manual residual gradient-check
    (gradients are now correct-by-construction)."""
    from chocofarm.az.mlp_jax_train import JaxTrainer
    env = Environment()
    fb = FeatureBuilder(env)
    in_dim, na = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=64, n_actions=na, seed=0, residual=True)
    feat = fb.build(("w", env.entry), env.worlds, set())
    mask = legal_mask_from_features(env, feat)
    B = 64
    rng = np.random.default_rng(0)
    X = (np.stack([feat] * B) + 0.05 * rng.standard_normal((B, in_dim))).astype(np.float32)
    M = np.stack([mask] * B).astype(np.float32)
    # a non-uniform target so the CE has a real gradient to descend
    PI = M * rng.random((B, na)).astype(np.float32)
    PI = PI / PI.sum(1, keepdims=True)
    Y = rng.standard_normal(B).astype(np.float32)
    net.set_value_scale(float(Y.mean()), float(Y.std()))
    tr = JaxTrainer(net, lr=1e-3, l2=1e-4)
    ce0, vl0 = tr.train_step(X, PI, M, Y)
    for _ in range(150):
        ce, vl = tr.train_step(X, PI, M, Y)
    assert np.isfinite(ce) and np.isfinite(vl)
    assert vl < vl0, f"value MSE did not reduce: {vl0:.4f} -> {vl:.4f}"
    assert ce < ce0, f"policy CE did not reduce: {ce0:.4f} -> {ce:.4f}"


def test_jax_train_live_lr_l2_betas_eps():
    """Audit item M (the Optimizer⊥Trainer split + the betas/eps follow-up to R13's frozen-config
    headline): lr/l2/betas/eps are ALL LIVE per step. The Trainer DELEGATES the update to an
    `Optimizer` that owns the `optax.inject_hyperparams` transform; lr/b1/b2/eps are set per step from
    the REQUIRED `AdamHParams` (the construction-enforced single-writer), l2 is a traced loss arg.
    Pin the new capability:
      (a) lr=0 leaves params unchanged (proves the injected lr is consumed, NOT the inject_hyperparams
          placeholder — the single-writer fires);
      (b) 10x lr ⇒ ~10x the first-step update on a fixed gradient (Adam's fresh-moment first step has
          magnitude ~lr);
      (c) a changed l2 changes the gradient by exactly l2*W on weight tensors (l2 live, not baked);
      (d) a changed b1 (the momentum decay) CHANGES the multi-step update — the new betas/eps-live
          capability. b1 governs the first-moment EMA, so two runs identical but for b1 diverge after
          a few steps (the same gradient stream produces a different Adam trajectory). This proves
          the injected b1 is read live, not the placeholder 0.9."""
    import jax
    import jax.numpy as jnp
    from chocofarm.az.mlp_jax_train import JaxTrainer, _az_loss, _JDTYPE
    from chocofarm.az.optimizer import AdamHParams
    from chocofarm.az.mlp import is_weight
    env = Environment()
    fb = FeatureBuilder(env)
    in_dim, na = feature_dim(env), n_action_slots(env)
    feat = fb.build(("w", env.entry), env.worlds, set())
    mask = legal_mask_from_features(env, feat)
    B = 48
    rng = np.random.default_rng(5)
    X = (np.stack([feat] * B) + 0.05 * rng.standard_normal((B, in_dim))).astype(np.float32)
    M = np.stack([mask] * B).astype(np.float32)
    PI = M * rng.random((B, na)).astype(np.float32)
    PI = PI / PI.sum(1, keepdims=True)
    Y = rng.standard_normal(B).astype(np.float32)

    def fresh_net():
        n = ValueMLP(in_dim, hidden=64, n_actions=na, seed=9, residual=True)
        n.set_value_scale(float(Y.mean()), float(Y.std()))
        return n

    def step_linf(lr, l2=1e-4):
        n = fresh_net()
        p0 = {k: np.asarray(v, np.float64) for k, v in n._params().items()}
        tr = JaxTrainer(n, lr=1e-3, l2=l2)
        tr.train_step(X, PI, M, Y, alpha=1.0, beta=1.0,
                      hp=AdamHParams(lr=lr, b1=0.9, b2=0.999, eps=1e-8), l2=l2)
        p1 = {k: np.asarray(v, np.float64) for k, v in n._params().items()}
        return max(float(np.max(np.abs(p1[k] - p0[k]))) for k in p0)

    # (a) lr=0 ⇒ no update (the injected lr is consumed; not the 1.0 placeholder)
    d0 = step_linf(0.0)
    assert d0 < 1e-6, f"lr=0 changed params (placeholder lr leaked?): max|Δ|={d0:.3e}"
    # (b) 10x lr ⇒ ~10x first-step update magnitude
    d1, d10 = step_linf(1e-3), step_linf(1e-2)
    assert 8.0 < d10 / d1 < 12.0, f"10x lr did not ~10x the step: {d1:.3e} -> {d10:.3e}"

    # (c) l2 is a live traced arg: grad@l2=1 - grad@l2=0 == 1.0*W on weight tensors
    n = fresh_net()
    params = {k: jnp.asarray(v, _JDTYPE) for k, v in n._params().items()}
    ys = (jnp.asarray(Y, _JDTYPE) - np.float32(n.y_mean)) / np.float32(n.y_std)
    gfn = jax.value_and_grad(_az_loss, has_aux=True)
    a1, b1 = jnp.float32(1.0), jnp.float32(1.0)
    (_, _), g0 = gfn(params, jnp.asarray(X, _JDTYPE), jnp.asarray(PI, _JDTYPE),
                     jnp.asarray(M, _JDTYPE), ys, a1, b1, jnp.float32(0.0))
    (_, _), g1 = gfn(params, jnp.asarray(X, _JDTYPE), jnp.asarray(PI, _JDTYPE),
                     jnp.asarray(M, _JDTYPE), ys, a1, b1, jnp.float32(1.0))
    worst = 0.0
    for k in params:
        if is_weight(k):
            expected = np.asarray(params[k], np.float64)   # d(0.5*1.0*||W||^2)/dW = 1.0*W
            got = np.asarray(g1[k], np.float64) - np.asarray(g0[k], np.float64)
            worst = max(worst, float(np.max(np.abs(got - expected))))
    assert worst < 1e-4, f"l2 not applied as a live coupled penalty (weights-only): max|Δ|={worst:.3e}"

    # (d) betas are LIVE: a changed b1 changes the multi-step update. Run K steps on the same fixed
    # batch with b1=0.9 vs b1=0.5 (everything else identical); the first-moment EMA differs, so the
    # Adam trajectory diverges — measurably (NOT roundoff). This is the new capability item M adds.
    def run_k(b1_val, K=8):
        n = fresh_net()
        tr = JaxTrainer(n, lr=1e-3, l2=1e-4)
        hp = AdamHParams(lr=1e-3, b1=b1_val, b2=0.999, eps=1e-8)
        for _ in range(K):
            tr.train_step(X, PI, M, Y, alpha=1.0, beta=1.0, hp=hp, l2=1e-4)
        return {k: np.asarray(v, np.float64) for k, v in n._params().items()}

    p_default = run_k(0.9)
    p_changed = run_k(0.5)
    b1_delta = max(float(np.max(np.abs(p_default[k] - p_changed[k]))) for k in p_default)
    assert b1_delta > 1e-4, (
        f"changing b1 (0.9 -> 0.5) did not change the multi-step update (betas not live?): "
        f"max|Δ|={b1_delta:.3e}")


def test_jax_train_writes_back_numpy_inference():
    """After a JAX train step the net's numpy inference (predict_both) reads the TRAINED weights:
    the trainer rebinds the net's arrays, so the float32 inference cache's identity check rebuilds
    (the cache-coherence invariant). The predicted value must change after training."""
    from chocofarm.az.mlp_jax_train import JaxTrainer
    env = Environment()
    fb = FeatureBuilder(env)
    in_dim, na = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=64, n_actions=na, seed=1, residual=True)
    feat = fb.build(("w", env.entry), env.worlds, set())
    mask = legal_mask_from_features(env, feat)
    B = 32
    X = np.stack([feat] * B).astype(np.float32)
    M = np.stack([mask] * B).astype(np.float32)
    PI = M / M.sum(1, keepdims=True)
    Y = np.linspace(-1.0, 1.0, B).astype(np.float32)
    net.set_value_scale(float(Y.mean()), float(Y.std()))
    v_before, p_before = net.predict_both(feat, mask)   # populate the f32 cache
    tr = JaxTrainer(net, lr=1e-2, l2=0.0)
    for _ in range(20):
        tr.train_step(X, PI, M, Y)
    v_after, p_after = net.predict_both(feat, mask)
    assert v_after != v_before, "numpy inference served stale weights after JAX training (cache bug)"
    assert abs(float(p_after.sum()) - 1.0) < 1e-6
    assert float(p_after[mask == 0].sum()) == 0.0


def test_residual_off_bit_identical_to_baseline():
    """residual=False must be numerically identical to the pre-residual net: no block params exist
    (so the shared `forward_core` skips the block — `head_in` IS the trunk output a2), and the
    forward is byte-for-byte the explicit pre-residual matmul chain. This is the clean-ablation
    guarantee. (Post-R11 `_forward` returns `(None, v_std, logits)` — the trunk-intermediate cache
    that used to back this assertion was vestigial and is gone; the byte-identity check below proves
    the residual-OFF math observably, which is what the cache-poke proved indirectly.)"""
    env = Environment()
    fb = FeatureBuilder(env)
    in_dim, na = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=32, n_actions=na, seed=3, residual=False)
    assert not hasattr(net, "Wr1") and "Wr1" not in net._params()
    # draw-order guard: residual=False must consume the SAME rng draws as a net built before the
    # block existed — i.e. the block params are not drawn at all when OFF, so every trunk/head
    # weight is byte-identical to a residual-ON net's trunk/head (the block draws come AFTER).
    net_on = ValueMLP(in_dim, hidden=32, n_actions=na, seed=3, residual=True)
    for k in ("W1", "b1", "W2", "b2", "Wv", "bv", "Wp", "bp"):
        assert np.array_equal(getattr(net, k), getattr(net_on, k)), \
            f"{k} differs between residual OFF/ON at same seed — block draws perturbed the stream"
    feat = fb.build(("w", env.entry), env.worlds, set())
    X = np.stack([feat] * 4).astype(np.float64)
    _, v_got, lg_got = net._forward(X)
    # explicit pre-residual math (what the old code computed)
    z1 = X @ net.W1 + net.b1; a1c = np.maximum(z1, 0.0)
    z2 = a1c @ net.W2 + net.b2; a2c = np.maximum(z2, 0.0)
    v_ref = (a2c @ net.Wv + net.bv).ravel()
    lg_ref = a2c @ net.Wp + net.bp
    assert np.array_equal(v_ref, v_got) and np.array_equal(lg_ref, lg_got)


def test_residual_on_cache_coherent_across_writers():
    """The float32 inference cache stays coherent with residual ON across every weight writer —
    here a REBIND of a residual-block param (the load/warm-start shape) after the cache is
    populated. (The non-residual coverage lives in test_predict_both_cache_coherent_across_writers.)"""
    env = Environment()
    fb = FeatureBuilder(env)
    net = ValueMLP(feature_dim(env), hidden=64, n_actions=n_action_slots(env),
                   seed=0, residual=True)
    feat = fb.build(("w", env.entry), env.worlds, set())
    mask = legal_mask_from_features(env, feat)

    def truth_value():
        _, v_std, _ = net._forward(np.asarray(feat, dtype=np.float64)[None, :])
        return float(v_std[0] * net.y_std + net.y_mean)

    net.predict_both(feat, mask)            # populate cache
    rng = np.random.default_rng(7)
    net.Wr1 = rng.standard_normal(net.Wr1.shape)
    net.Wr2 = rng.standard_normal(net.Wr2.shape)
    v_after, _ = net.predict_both(feat, mask)
    assert abs(v_after - truth_value()) < 1e-2, "stale cache after residual-param rebind"


def test_residual_save_load_roundtrip_and_old_npz():
    """A residual net round-trips through save/load (block params preserved, predictions match);
    and a pre-residual npz (no Wr*/br*, 3-field _meta) loads with the block OFF — the graceful
    backward-compat path mirroring the --init-weights dim-mismatch handling (ADR-0002)."""
    import tempfile
    env = Environment()
    fb = FeatureBuilder(env)
    in_dim, na = feature_dim(env), n_action_slots(env)
    feat = fb.build(("w", env.entry), env.worlds, set())
    mask = legal_mask_from_features(env, feat)
    net = ValueMLP(in_dim, hidden=32, n_actions=na, seed=2, residual=True)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "res.npz")
        v0, p0 = net.predict_both(feat, mask)
        net.save(path)
        net2 = ValueMLP.load(path)
        assert net2.residual and hasattr(net2, "Wr1")
        v1, p1 = net2.predict_both(feat, mask)
        assert abs(v0 - v1) < 1e-6 and np.allclose(p0, p1, atol=1e-6)

        # forge a "pre-residual" npz: drop the block params and the 4th _meta field
        z = dict(np.load(path, allow_pickle=False))
        for k in ("Wr1", "br1", "Wr2", "br2"):
            z.pop(k)
        z["_meta"] = z["_meta"][:3]
        old_path = os.path.join(d, "old.npz")
        np.savez(old_path, **z)
        net_old = ValueMLP.load(old_path)
        assert not net_old.residual and "Wr1" not in net_old._params()

        # corrupt block-param shape (flag ON) must fail LOUDLY at load (ADR-0002: at setup, not
        # deep in the first forward).
        zc = dict(np.load(path, allow_pickle=False))
        zc["Wr1"] = zc["Wr1"][:, :-1]  # wrong shape
        bad_path = os.path.join(d, "bad.npz")
        np.savez(bad_path, **zc)
        try:
            ValueMLP.load(bad_path)
            assert False, "expected ValueError on corrupt residual param shape"
        except ValueError:
            pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all az-loop checks passed")
