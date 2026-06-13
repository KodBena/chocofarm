#!/usr/bin/env python3
"""
chocofarm policy — Single-Observer Information Set Monte-Carlo Tree Search (SO-ISMCTS).

Faithful implementation of the single-observer variant of Cowling, Powley & Whitehouse,
"Information Set Monte Carlo Tree Search", IEEE TCIAIG 4(2), 2012 (Algorithm 1, §IV-E),
with the subset-armed-bandit UCB modification of §IV-B.

Why ISMCTS fits this problem. The chocofarm task is a single-observer belief-MDP: there is
no adversary, only an "environment" that resolves each action's observation (a treasure is
present/absent; a disjunctive detector reads positive/negative). The information set the
searching player occupies is *exactly* the belief `bw` — the numpy set of latent worlds still
consistent with everything observed. So the central ISMCTS idea maps directly:

  - A node in the tree is an information set (a belief), reached by a sequence of actions whose
    *observations* came out a particular way. Edges are actions.
  - Each iteration DETERMINIZES: sample one concrete world `w ~ bw`. Because the belief already
    encodes consistency (exactly-5-of-20, all past observations applied), sampling from `bw` is
    an unbiased draw of "which world am I really in" — no particle filter needed (env note).
  - The determinization `w` RESOLVES every action's observation outcome, hence which child you
    descend into. Two visits to the same node under different determinizations branch to
    different children of the same action edge — and the action statistics aggregate over the
    whole information set, which is the property determinized-UCT lacks and ISMCTS restores.

Subset-armed bandit (§IV-B). The set of legal actions at a node SHRINKS as the belief sharpens:
a detector drops out once its outcome is certain under `bw` (env.legal_actions enforces this),
and a treasure drops out once revealed absent. So an action is "available" only on some
iterations. UCB1 (eq. 7) is therefore computed with the parent's *availability* count n'(parent)
in the log term — incremented for every action that was legal on a given visit, whether or not
it was the one selected — rather than the raw parent visit count. Without this, rarely-legal
actions are over-explored. c defaults to 0.7, the value the paper used across all experiments.

Objective coupling. The playout return is the λ-penalized renewal reward Σ value − λ·(Σ travel +
exit), i.e. the same Dinkelbach surrogate `env.dinkelbach_rate` drives the whole harness with.
At a node the agent may also choose to TERMINATE (stop and exit now); we treat TERMINATE as an
ordinary edge whose value is the bare −λ·exit_cost continuation, so the search can *learn* the
early-exit option that the clairvoyant ceiling shows is where most of the +70% lives.
"""
import math
import numpy as np
from chocofarm.solvers.base import Policy, _base_value
from chocofarm.model.env import TERMINATE


class GreedyStopBase(Policy):
    """Default ISMCTS playout policy: a λ-rational greedy that stops cleanly.

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


def _belief_key(bw):
    """A cheap, order-insensitive identity for an information set (a belief world-set).

    Beliefs reached by the same observations are the same set of worlds regardless of the path;
    the smallest + largest + count triple is a collision-resistant fingerprint for the modest
    number of distinct beliefs a single search reaches (full equality is verified on collision).
    """
    n = len(bw)
    if n == 0:
        return (0, 0, 0)
    return (n, int(bw[0]), int(bw[-1]))


class _Node:
    """One information-set node. Children are keyed by action; an action's observation outcome
    under the active determinization selects which *successor belief* (sub-child) we descend to.

    We keep per-action aggregate statistics (reward sum, selection count, availability count)
    aggregated over the whole information set — that is the ISMCTS contract. The observation
    outcome only routes *which child node* the simulation continues from; it does not split the
    action's bandit statistics."""
    __slots__ = ("reward", "visits", "avail", "children")

    def __init__(self):
        self.reward = {}     # action -> summed playout return over selections of this action
        self.visits = {}     # action -> times this action was selected from here   (n_j)
        self.avail = {}      # action -> times this action was legal/available here  (n'_j)
        self.children = {}   # (action, belief_key) -> child _Node


class ISMCTSPolicy(Policy):
    """Single-Observer Information Set MCTS. `iterations` determinized tree-walks per decision;
    `c` the UCB1 exploration constant (paper default 0.7). `base` is the simulation (playout)
    policy used at leaves and for the rollout to the end of the episode; `GreedyStopBase` by
    default (pass a `policies.GreedyPolicy()` to use the plainer greedy) — cheap, and the search
    supplies the contingent depth the base lacks."""

    def __init__(self, iterations=300, c=0.7, base=None, max_depth=24):
        self.iterations = int(iterations)
        self.c = float(c)
        self.base = base if base is not None else GreedyStopBase()
        self.max_depth = int(max_depth)

    # ---- public API ----
    def decide(self, env, loc, bw, collected, lam, rng):
        if len(bw) == 0:
            return TERMINATE
        root = _Node()
        for _ in range(self.iterations):
            w = env.sample_world(bw, rng)              # (1) determinize: one world ~ belief
            self._iterate(env, root, loc, bw, set(collected), w, lam, rng, 0)
        # (final) return the most-visited action from the root; TERMINATE if nothing was tried.
        if not root.visits:
            return TERMINATE
        return max(root.visits, key=lambda a: root.visits[a])

    # ---- one determinized iteration: selection + expansion + simulation + backprop ----
    def _iterate(self, env, node, loc, bw, collected, world, lam, rng, depth):
        """Recursive descent for one iteration in fixed determinization `world`. Returns the
        λ-penalized return from `node` onward, which the caller backpropagates into the edge it
        took. `bw`/`collected` are the information-set + collected-set AT this node."""
        if depth >= self.max_depth:
            return -lam * env.exit_cost(loc)

        # Actions compatible with the determinization at this node: every legal action is
        # compatible (its observation is simply resolved by `world`); TERMINATE is always legal.
        legal = env.legal_actions(loc, bw, collected)
        actions = list(legal) + [TERMINATE]

        # Bump availability for every action legal on this visit (subset-armed bandit, §IV-B).
        for a in actions:
            node.avail[a] = node.avail.get(a, 0) + 1

        # (3) Expansion: if any compatible action is untried here, expand one at random.
        untried = [a for a in actions if a not in node.visits]
        if untried:
            a = untried[int(rng.integers(len(untried)))]
            ret = self._expand_and_simulate(env, node, loc, bw, collected, a, world, lam, rng, depth)
            self._update(node, a, ret)
            return ret

        # (2) Selection: UCB1 with availability count in the exploration term.
        a = self._ucb_select(node)
        if a == TERMINATE:
            ret = -lam * env.exit_cost(loc)            # stop now: only the exit toll remains
            self._update(node, a, ret)
            return ret

        r, nloc, nbw, ncoll, dt = env.apply(loc, bw, collected, a, world)
        step = r - lam * dt
        ckey = (a, _belief_key(nbw))
        child = node.children.get(ckey)
        if child is None:
            # The action edge exists, but this determinization routes to a successor belief not
            # yet seen — create that child node (still part of the same edge's statistics).
            child = _Node()
            node.children[ckey] = child
        cont = self._iterate(env, child, nloc, nbw, ncoll, world, lam, rng, depth + 1)
        ret = step + cont
        self._update(node, a, ret)
        return ret

    def _expand_and_simulate(self, env, node, loc, bw, collected, a, world, lam, rng, depth):
        """Realise a freshly expanded action under `world`, register its successor child, then
        (4) play the base policy to the end of the episode for the leaf estimate."""
        if a == TERMINATE:
            return -lam * env.exit_cost(loc)
        r, nloc, nbw, ncoll, dt = env.apply(loc, bw, collected, a, world)
        step = r - lam * dt
        ckey = (a, _belief_key(nbw))
        if ckey not in node.children:
            node.children[ckey] = _Node()
        # base playout from the post-action belief, scored as the λ-penalized return-to-go.
        cont = _base_value(env, self.base, nloc, nbw, ncoll, world, lam)
        return step + cont

    # ---- bandit + bookkeeping ----
    def _ucb_select(self, node):
        """UCB1 (eq. 7) with the subset-armed-bandit denominator: the log term uses the action's
        own availability count n'_j (times the action was legal here), per §IV-B."""
        best_a, best_v = None, -math.inf
        c = self.c
        for a, n_j in node.visits.items():
            if n_j == 0:
                return a
            exploit = node.reward[a] / n_j
            navail = node.avail.get(a, n_j)
            explore = c * math.sqrt(math.log(navail) / n_j) if navail > 1 else c
            v = exploit + explore
            if v > best_v:
                best_v, best_a = v, a
        return best_a

    @staticmethod
    def _update(node, a, ret):
        node.visits[a] = node.visits.get(a, 0) + 1
        node.reward[a] = node.reward.get(a, 0.0) + ret
