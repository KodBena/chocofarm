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
import numpy as np
from chocofarm.model.env import TERMINATE

# The UCB1 exploration constant, held fixed across UCT/ISMCTS/NetValueISMCTS for a fair comparison — one home.
UCB_C = 0.7


class Policy(ABC):
    @abstractmethod
    def decide(self, env, loc, bw, collected, lam, rng):
        """Return an action ('t', i) / ('d', i) / TERMINATE."""


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
    def __init__(self, base, n_samples=10, near_det=3, near_tre=3):
        self.base, self.S, self.nd, self.nt = base, n_samples, near_det, near_tre

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
    def __init__(self, depth, width, leaf):
        self.depth, self.width, self.leaf = depth, width, leaf

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
