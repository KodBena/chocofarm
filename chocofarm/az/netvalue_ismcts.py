#!/usr/bin/env python3
"""
chocofarm AZ — NetValueISMCTS: SO-ISMCTS with the LEARNED value at the leaf (design §9 Stage-2).

This is the E-DECIDE probe's whole point (design §1 H-calibrate, §7 ablation #1, §9): take the
existing `ISMCTSPolicy` UNCHANGED and swap ONLY the leaf evaluation — the determinized base
playout (`_base_value`, the optimistically-biased F4 leaf) is replaced by the trained value net
`V_λ(features(belief))`. Same iteration budget, same UCB selection, same tree machinery,
same TERMINATE handling — byte-identical except the one leaf call. That isolates the single
claim: calibrated learned value vs optimistic playout, at matched budget.

Implementation is minimal-touch (ADR-0004 register): we subclass `ISMCTSPolicy` and override
exactly the method that computes the leaf continuation. `ISMCTSPolicy._expand_and_simulate`
applies the freshly-expanded action, registers the successor child, then scores the leaf as
`_base_value(... post-action belief ...)`. We reproduce that method verbatim EXCEPT the final
continuation line, which becomes `self._leaf_value(env, nloc, nbw, ncoll, lam)`.

The net was trained (dataset.py / train_value.py) to predict the realized λ-penalized
return-to-go FROM the post-action belief — the same quantity `_base_value` estimates by rollout
— so the substitution is dimensionally and semantically matched. The net's predicted value is
on the λ-penalized-return scale (de-standardized by `ValueMLP.predict_value`).

Loads weights from an npz (the trained value net). Pin any eval to core 3 under timeout.
"""
from __future__ import annotations

from chocofarm.solvers.ismcts import ISMCTSPolicy, _Node, _belief_key
from chocofarm.model.env import TERMINATE
from chocofarm.az.features import FeatureBuilder
from chocofarm.az.mlp import ValueMLP


class NetValueISMCTS(ISMCTSPolicy):
    """ISMCTS with the learned value net at the leaf, in place of the determinized playout.

    `weights_path` — npz from `train_value.py`. `iterations`, `c`, `max_depth` carry the SAME
    meaning and defaults as `ISMCTSPolicy`; pass the matched budget so the comparison is fair.
    `env` is needed at construction to build the FeatureBuilder (feature dims are env-derived).
    """

    def __init__(self, env, weights_path, iterations=200, c=0.7, max_depth=24):
        # base= is irrelevant (the playout leaf is never used), but ISMCTSPolicy builds a
        # default GreedyStopBase; harmless and keeps the parent contract intact.
        super().__init__(iterations=iterations, c=c, base=None, max_depth=max_depth)
        self.net = ValueMLP.load(weights_path)
        self.fb = FeatureBuilder(env)

    def _leaf_value(self, env, loc, bw, collected, lam):
        """The F4 cure: learned λ-penalized return-to-go from this (post-action) belief,
        replacing the determinized base playout. One cached marginals call per leaf (F7)."""
        if len(bw) == 0:
            # empty belief: no continuation value beyond the exit toll (mirrors the base case).
            return -lam * env.exit_cost(loc)
        marg = env.marginals(bw)
        feat = self.fb.build(loc, bw, collected, marg=marg)
        return self.net.predict_value(feat)

    def _expand_and_simulate(self, env, node, loc, bw, collected, a, world, lam, rng, depth):
        """Identical to ISMCTSPolicy._expand_and_simulate EXCEPT the leaf continuation, which
        is the LEARNED value instead of `_base_value`'s determinized playout."""
        if a == TERMINATE:
            return -lam * env.exit_cost(loc)
        r, nloc, nbw, ncoll, dt = env.apply(loc, bw, collected, a, world)
        step = r - lam * dt
        ckey = (a, _belief_key(nbw))
        if ckey not in node.children:
            node.children[ckey] = _Node()
        # --- the ONLY substantive change vs the parent: learned value at the leaf ---
        cont = self._leaf_value(env, nloc, nbw, ncoll, lam)
        return step + cont
