#!/usr/bin/env python3
"""
chocofarm policies — the SOLVERS, pluggable behind one interface.

A Policy maps the observable state (location, belief, collected) + the rate target λ to an
action. Everything else (dynamics, simulation, evaluation) lives in env.py. To add a new
method — NMCS, ISMCTS, a learned policy — subclass Policy and implement `decide`; nothing in
env.py changes. The env is passed in, so a policy may freely query dynamics/belief primitives
(legal_actions, marginals, apply, filter_*, sample_world, d, exit_cost, route_time).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from chocofarm.model.env import TERMINATE

# The UCB1 exploration constant, held fixed across UCT/ISMCTS/NetValueISMCTS for a fair comparison — one home.
UCB_C = 0.7


# ---------------------------------------------------------------------------
# Per-solver SearchConfig dataclasses (audit item I / R10's deferred slice).
#
# Each classical solver freezes its hyperparameters as scalar `self.X` in
# `__init__`, so a sweep reconstructs one policy per budget. The audit
# (§3.5, appendix C) prescribes "a SearchConfig dataclass PER SOLVER FAMILY":
# the knobs genuinely differ between families (UCT has c/horizon, NMCS has
# level/sample-budgets/candidate-widths, Rollout has n_samples/near_*), so a
# single shared dataclass would be a union of disjoint fields, not a real SSOT.
# We therefore give each family its own frozen dataclass grouping exactly that
# family's current scalar __init__ knobs. The NON-scalar knobs (the rollout /
# base Policy *object*) stay as ordinary __init__ params — a frozen scalar
# config is the wrong home for a live Policy instance, and the audit's scope is
# the scalar hyperparameter set.
#
# Each solver's __init__ accepts EITHER `cfg=<Config>` OR the current individual
# kwargs (back-compat: `UCTPolicy(iterations=200)` still works — the kwargs build
# the config). Defaults are unchanged, so behaviour is preserved.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutConfig:
    """Frozen scalar hyperparameters for `RolloutPolicy` (the base Policy is passed separately)."""
    n_samples: int = 10
    near_det: int = 3
    near_tre: int = 3


@dataclass(frozen=True)
class SparseSamplingConfig:
    """Frozen scalar hyperparameters for `SparseSamplingPolicy` (the leaf Policy is passed separately)."""
    depth: int = 2
    width: int = 3


class Policy(ABC):
    @abstractmethod
    def decide(self, env, loc, bw, collected, lam, rng):
        """Return an action ('t', i) / ('d', i) / TERMINATE."""


class RandomPolicy(Policy):
    """Uniform-random over the legal action set (collects + informative senses + TERMINATE).

    The trivial composable Policy — the env↔Policy seam's simplest non-trivial instance, and the
    parity baseline for the C++ runner's `RandomPolicy` (ADR-0012 P2/P7: a new capability is a new
    `Policy` subclass with zero env edits; the C++ runner mirrors THIS behavioral contract behind
    the wire, not its bytes). It uses ONLY the env's own dynamics primitives — `legal_actions`
    (which already excludes uninformative senses and collected/absent treasures) plus the
    always-legal TERMINATE slot — and draws one uniformly with the injected `rng`. λ does not enter
    the choice (a dumb-random runner ignores the rate target); it is still threaded through the
    seam unchanged (P4), so a value-aware policy is a drop-in replacement with no signature change.

    Determinism note: the choice index is drawn with `rng.integers(len(acts) + 1)` over the legal
    actions in `env.legal_actions` order with TERMINATE appended last — a single integer draw per
    decision, so a reproducing harness can match the action-TYPE distribution exactly under matched
    seeds, while the cross-language float-sensitive aggregates (E[T], λ-return) are held to the
    ADR-0012 P6 behavioral-equivalence bar, not byte-identity."""
    def decide(self, env, loc, bw, collected, lam, rng):
        acts = env.legal_actions(loc, bw, collected)
        acts = acts + [TERMINATE]          # TERMINATE is always legal (matches actions.term_slot)
        return acts[int(rng.integers(len(acts)))]


class GreedyPolicy(Policy):
    """Myopic: go to the treasure with best expected λ-adjusted value; else terminate.
    Belief-responsive only through collect-reveals; detector-blind (a deliberately weak base)."""
    def decide(self, env, loc, bw, collected, lam, rng=None):
        marg = env.marginals(bw)
        best, act = 0.0, TERMINATE
        for i in range(env.N):
            if i in collected or marg[i] <= 0:
                continue
            s = marg[i] * env.value[i] - lam * env.d(loc, ("t", i))
            if s > best:
                best, act = s, ("t", i)
        return act


class GreedyStopBase(Policy):
    """Default ISMCTS/UCT playout policy: a λ-rational greedy that stops cleanly.

    Plain `GreedyPolicy` (the obvious base) over-collects under a renewal-reward penalty — it
    keeps a treasure as long as `marg·value − λ·travel > 0`, ignoring that reaching it also
    *relocates the exit*, so it sweeps low-marginal treasures across the map and the playout
    return understates the rate (the over-collection signature in docs/results). This base nets
    the exit relocation into the step value: move to the best treasure only when

        marg·value − λ·(go_there + exit(there) − exit(here)) > 0,

    else TERMINATE. That single correction turns the playout into a tighter renewal cycle, so
    leaf estimates reward banking a reachable basket and exiting — the behaviour the clairvoyant
    ceiling rewards — rather than an exhaustive sweep."""
    def decide(self, env, loc, bw, collected, lam, rng=None):
        marg = env.marginals(bw)
        cur_exit = env.exit_cost(loc)
        best, act = 0.0, TERMINATE
        for i in range(env.N):
            if i in collected or marg[i] <= 0:
                continue
            go = env.d(loc, ("t", i))
            net = marg[i] * env.value[i] - lam * (go + env.exit_cost(("t", i)) - cur_exit)
            if net > best:
                best, act = net, ("t", i)
        return act


class CertaintyEquivalentPolicy(Policy):
    """Collapse the belief to its most-likely scenario (the ~E[#present] treasures with the
    highest posterior marginal), plan the rate-optimal route over that set, take the first
    step; re-plan each step as the belief sharpens. A strong, belief-using base for rollout —
    after a detector splits the belief, the re-planned route changes, so rollout-over-CE
    *values* information that rollout-over-greedy cannot act on."""
    def decide(self, env, loc, bw, collected, lam, rng=None):
        marg = env.marginals(bw)
        rem = [i for i in range(env.N) if i not in collected and marg[i] > 0]
        if not rem:
            return TERMINATE
        m = max(1, round(sum(marg[i] for i in rem)))               # expected # still present
        map_set = set(sorted(rem, key=lambda i: -marg[i])[:m])      # most-likely-present set
        cur, t, route, best = loc, 0.0, [], (-lam * env.exit_cost(loc), None)
        while map_set:                                              # greedy NN route over it
            j = min(map_set, key=lambda x: env.d(cur, ("t", x)))
            t += env.d(cur, ("t", j)); cur = ("t", j); route.append(j); map_set.discard(j)
            v = sum(env.value[x] for x in route) - lam * (t + env.exit_cost(cur))
            if v > best[0]:
                best = (v, route[0])
        return ("t", best[1]) if best[1] is not None else TERMINATE


class RolloutPolicy(Policy):
    """One-step policy improvement over a base policy: for each candidate action, sample
    worlds from the current belief, apply the action, play the base to the end, average the
    λ-value, take the argmax. Candidates pruned to the nearest few detectors/treasures + exit."""
    def __init__(self, base, n_samples=10, near_det=3, near_tre=3, *, cfg=None):
        # cfg=RolloutConfig(...) supplies the scalar knobs in one frozen object; the individual
        # kwargs remain the back-compat path and build the config when no cfg is passed (ADR-0004).
        self.cfg = cfg if cfg is not None else RolloutConfig(n_samples, near_det, near_tre)
        self.base = base
        self.S, self.nd, self.nt = self.cfg.n_samples, self.cfg.near_det, self.cfg.near_tre

    def decide(self, env, loc, bw, collected, lam, rng):
        # nearest-few detectors/treasures via the shared pruner; Rollout handles exit through its
        # `best_q = -lam*exit_cost(loc)` init, so it does NOT include TERMINATE in the candidate set.
        cands = candidate_actions(env, loc, bw, collected, self.nd, self.nt)
        sample = rng.choice(bw, size=min(self.S, len(bw)), replace=len(bw) < self.S)
        best_q, best_a = -lam * env.exit_cost(loc), TERMINATE
        for a in cands:
            tot = 0.0
            for w in sample:
                w = int(w)
                r, nloc, nbw, nc, dt = env.apply(loc, bw, collected, a, w)
                tot += (r - lam * dt) + _base_value(env, self.base, nloc, nbw, nc, w, lam)
            q = tot / len(sample)
            if q > best_q:
                best_q, best_a = q, a
        return best_a


class SparseSamplingPolicy(Policy):
    """Sparse-sampling expectimax (Kearns–Mansour–Ng): the dumb convergent anchor. Full legal
    action set, sample `width` worlds per node, recurse to `depth`, base-policy rollout at the
    leaf. More width/depth → provably nearer optimal."""
    def __init__(self, depth=None, width=None, leaf=None, *, cfg=None):
        # cfg=SparseSamplingConfig(...) supplies (depth, width); the positional/kwarg form
        # SparseSamplingPolicy(depth, width, leaf) remains the back-compat path (ADR-0004). The
        # leaf base Policy is always passed separately (not a frozen-config scalar).
        if cfg is not None:
            self.cfg = cfg
        else:
            if depth is None or width is None:
                raise ValueError("SparseSamplingPolicy needs (depth, width) or cfg=SparseSamplingConfig(...)")
            self.cfg = SparseSamplingConfig(depth, width)
        self.depth, self.width = self.cfg.depth, self.cfg.width
        self.leaf = leaf

    def decide(self, env, loc, bw, collected, lam, rng):
        best_q, best_a = -lam * env.exit_cost(loc), TERMINATE
        for a in env.legal_actions(loc, bw, collected):
            q = self._q(env, loc, bw, collected, a, lam, self.depth, rng)
            if q > best_q:
                best_q, best_a = q, a
        return best_a

    def _q(self, env, loc, bw, collected, a, lam, depth, rng):
        sample = rng.choice(bw, size=min(self.width, len(bw)), replace=len(bw) < self.width)
        tot = 0.0
        for w in sample:
            w = int(w)
            r, nloc, nbw, nc, dt = env.apply(loc, bw, collected, a, w)
            step = r - lam * dt
            if depth <= 1:
                tot += step + _base_value(env, self.leaf, nloc, nbw, nc, w, lam)
            else:
                bv = -lam * env.exit_cost(nloc)
                for a2 in env.legal_actions(nloc, nbw, nc):
                    bv = max(bv, self._q(env, nloc, nbw, nc, a2, lam, depth - 1, rng))
                tot += step + bv
        return tot / len(sample)


def candidate_actions(env, loc, bw, collected, n_det, n_tre, include_terminate=False):
    """Nearest-n_det informative detectors + nearest-n_tre uncollected-possible treasures
    (by env.d from loc), optionally + TERMINATE. The shared bounded-branching pruner."""
    marg = env.marginals(bw)
    dets = sorted((i for i in env.detectors
                   if np.any((bw & env.cover_mask[i]) != 0) and np.any((bw & env.cover_mask[i]) == 0)),
                  key=lambda i: env.d(loc, ("d", i)))[:n_det]
    tres = sorted((i for i in range(env.N) if i not in collected and marg[i] > 0),
                  key=lambda i: env.d(loc, ("t", i)))[:n_tre]
    cands = [("d", i) for i in dets] + [("t", i) for i in tres]
    if include_terminate:
        cands.append(TERMINATE)
    return cands


def _base_value(env, base, loc, bw, collected, world, lam):
    """Play a (deterministic) base policy to the end in a fixed world; return its λ-value."""
    R = T = 0.0
    collected = set(collected)
    for _ in range(env.max_steps):              # the single episode-horizon home (env.py)
        a = base.decide(env, loc, bw, collected, lam, None)
        if a == TERMINATE:
            break
        r, loc, bw, collected, dt = env.apply(loc, bw, collected, a, world)
        R += r; T += dt
    return R - lam * (T + env.exit_cost(loc))
