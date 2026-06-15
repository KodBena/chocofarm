#!/usr/bin/env python3
"""
chocofarm policy — Vanilla single-tree UCT on the belief-MDP (the no-determinization baseline).

This is plain UCT (Kocsis & Szepesvári, "Bandit Based Monte-Carlo Planning", ECML 2006)
applied directly to the chocofarm belief-MDP, deliberately WITHOUT any of SO-ISMCTS's
information-set machinery. It exists as the apples-to-apples control: same Policy interface,
same `iterations` budget knob, same λ-penalised differential return, same rollout base — so a
matched-budget comparison isolates exactly what ISMCTS's determinization buys (if anything).

How this DIFFERS from `ismcts.py` (the distinction is the whole point):

  - **Node identity.** Here a tree node is an action–OBSERVATION history: the exact sequence of
    actions taken AND the observation outcome each produced. Each node carries the exact belief
    that history reaches, obtained by applying `env.filter_*` along the path (the belief is the
    sufficient statistic of the history). SO-ISMCTS instead collapses every history that reaches
    the same information set into ONE node and aggregates each action's bandit statistics over the
    whole information set. We do NOT aggregate: two histories that reach the same belief by
    different observation sequences are different nodes with independent statistics.

  - **No per-iteration determinization of the tree.** SO-ISMCTS samples one concrete world
    `w ~ bw` at the root of each iteration and lets that single world RESOLVE every observation
    along the descent (so the same action edge routes to different sub-children on different
    iterations, but its stats are shared). Vanilla UCT has no such global determinization: at each
    action we treat the observation as an explicit CHANCE node and sample ONE outcome at a time,
    weighted by the belief-conditioned outcome probability at THAT node, then descend into the
    child belief that outcome produces. The tree therefore branches explicitly on observations and
    keeps per-(action,observation) statistics — the structure ISMCTS was designed to avoid because
    it fragments the information set's samples across observation-distinguished children.

  Both act only on information actually available (the belief, via the exact `filter_*`), so the
  information model is identical and the comparison is fair. The ONLY difference is the
  information-set grouping / determinization that ISMCTS adds on top.

Chance handling (binary outcomes, so no progressive widening). Every action has a binary
observation:
  - a treasure probe ('t', i): present / absent, with P(present) = the belief marginal at i;
  - a detector ('d', i): the disjunction over its cover reads positive / negative, with
    P(positive) = fraction of surviving worlds that intersect the cover mask.
We sample one outcome per traversal from that belief-conditioned distribution and descend into
the corresponding filtered belief. Sampling-one-outcome-weighted-by-belief gives each child the
right visit frequency in expectation, so its `Q̄` is an unbiased estimate of the outcome-averaged
continuation value — the chance-node analogue of UCB1's mean backup.

Rollout. From a freshly expanded leaf we play a cheap default base policy to the end of the
episode in a single sampled world and score the λ-penalised differential return
`Σ (reward − λ·dt)` to the exit (`_base_value`, shared with the other solvers). Default base is
`GreedyStopBase` (the shared rollout base in `base.py`) — the SAME base ISMCTS uses by default, chosen so
the rollout estimator is held FIXED across the two solvers and any rate difference is attributable
to the tree, not the leaf heuristic. (`GreedyPolicy` is available as a weaker alternative; pass it
to study sensitivity.)

Backup. The differential return flows back up the descent path; each decision node's `Q̄(a)` is
the running mean of returns observed through action `a`, and each chance node's value is the
running mean over the outcomes it sampled. UCB1 at a decision node selects
`argmax_a Q̄(a) + c·sqrt(ln N(node) / N(a))`. The exploration constant `c` defaults to 0.7, the
same value ISMCTS uses, again to hold everything but the tree machinery fixed (the reward scale
here is O(1) λ-differential return, for which c≈0.7 is a standard, well-behaved choice).

Decision. `decide` runs `iterations` simulations from the root then returns the **most-visited**
root action (the robust-child rule, matching ISMCTS's final selection, so the two report on the
same statistic). TERMINATE is an ordinary decision-node action valued at the bare `−λ·exit_cost`.
"""
import math
import numpy as np

from chocofarm.solvers.base import Policy, GreedyPolicy, _base_value, UCB_C, GreedyStopBase
from chocofarm.model.env import TERMINATE


class _Decision:
    """A decision node: an action–observation history with its exact belief. Holds UCB1 bandit
    statistics over the actions legal here, and a chance-node child per action taken."""
    __slots__ = ("reward", "visits", "n", "chance")

    def __init__(self):
        self.reward = {}     # action -> summed differential return backed up through it
        self.visits = {}     # action -> times this action was selected here          (N(a))
        self.n = 0           # total selections from this node                         (N(node))
        self.chance = {}     # action -> _Chance node reached by taking that action


class _Chance:
    """A chance node hanging off one action edge: it owns the action's post-action immediate step
    value and routes the binary observation outcome to one of two child decision nodes (the
    belief filtered to that outcome). Keeps its own visit/return mean for backup."""
    __slots__ = ("reward", "visits", "outcomes")

    def __init__(self):
        self.reward = 0.0    # summed (step + continuation) return through this chance node
        self.visits = 0      # times this chance node was traversed
        self.outcomes = {}   # outcome-key (bool) -> _Decision child for that observation


class UCTPolicy(Policy):
    """Vanilla single-tree UCT on the belief-MDP — the no-determinization baseline beside ISMCTS.

    Parameters
    ----------
    iterations : int
        Per-decision simulation budget (the matched knob vs ISMCTS). `decide` runs this many
        root-to-leaf simulations, then returns the most-visited root action.
    c : float
        UCB1 exploration constant. Default 0.7 (ISMCTS's value; standard for O(1)-scale return).
    rollout : str | Policy
        Leaf rollout base. "greedy_stop" (default) uses `GreedyStopBase`, the same λ-rational
        bank-and-exit greedy ISMCTS rolls out with; "greedy" uses the plainer `GreedyPolicy`; a
        Policy instance is used directly. Held fixed vs ISMCTS so only the tree machinery varies.
    horizon : int
        Hard cap on tree depth per simulation; the rollout base also self-caps inside `_base_value`.
    """

    def __init__(self, iterations=300, c=UCB_C, rollout="greedy_stop", horizon=24):
        self.iterations = int(iterations)
        self.c = float(c)
        self.horizon = int(horizon)
        if isinstance(rollout, Policy):
            self.base = rollout
        elif rollout == "greedy":
            self.base = GreedyPolicy()
        elif rollout == "greedy_stop":
            self.base = GreedyStopBase()
        else:
            raise ValueError(f"unknown rollout base {rollout!r}")

    # ---- public Policy API -------------------------------------------------------------
    def decide(self, env, loc, bw, collected, lam, rng):
        if len(bw) == 0:
            return TERMINATE
        root = _Decision()
        for _ in range(self.iterations):
            self._simulate(env, root, loc, bw, set(collected), lam, rng, 0)
        if not root.visits:
            return TERMINATE
        return max(root.visits, key=lambda a: root.visits[a])      # robust child (most-visited)

    # ---- one simulation: select -> expand -> rollout -> backup -------------------------
    def _simulate(self, env, node, loc, bw, collected, lam, rng, depth):
        """Recursive descent for one simulation. Returns the λ-penalised differential return from
        `node` onward, which the caller backs up into the chance node it descended through."""
        if depth >= self.horizon:
            return -lam * env.exit_cost(loc)

        actions = list(env.legal_actions(loc, bw, collected)) + [TERMINATE]

        # Expansion: if an action here is untried, expand one at random and rollout from it.
        untried = [a for a in actions if a not in node.visits]
        if untried:
            a = untried[int(rng.integers(len(untried)))]
            ret = self._expand(env, node, loc, bw, collected, a, lam, rng)
            self._backup_decision(node, a, ret)
            return ret

        # Selection: UCB1 over the (fully expanded) action set.
        a = self._ucb_select(node, actions)
        ret = self._descend(env, node, loc, bw, collected, a, lam, rng, depth)
        self._backup_decision(node, a, ret)
        return ret

    def _expand(self, env, node, loc, bw, collected, a, lam, rng):
        """First visit to action `a` from this node: create its chance node, sample one outcome,
        register the resulting child belief, and rollout the base policy from it for the leaf
        estimate. Returns the differential return (step + rollout-to-go)."""
        if a == TERMINATE:
            return -lam * env.exit_cost(loc)                       # stop now: only the exit toll
        chance = _Chance()
        node.chance[a] = chance
        world = env.sample_world(bw, rng)                          # one world -> resolves outcome
        r, nloc, nbw, ncoll, dt = env.apply(loc, bw, collected, a, world)
        step = r - lam * dt
        okey = self._outcome_key(a, world, env)
        child = _Decision()
        chance.outcomes[okey] = child
        cont = _base_value(env, self.base, nloc, nbw, ncoll, world, lam)   # rollout to episode end
        ret = step + cont
        self._backup_chance(chance, ret)
        return ret

    def _descend(self, env, node, loc, bw, collected, a, lam, rng, depth):
        """A previously-tried action selected by UCB1: realise it through its chance node, sample
        one observation outcome weighted by the current belief, and recurse into (or expand) the
        decision child for that outcome."""
        if a == TERMINATE:
            return -lam * env.exit_cost(loc)
        chance = node.chance[a]
        world = env.sample_world(bw, rng)                          # belief-weighted outcome draw
        r, nloc, nbw, ncoll, dt = env.apply(loc, bw, collected, a, world)
        step = r - lam * dt
        okey = self._outcome_key(a, world, env)
        child = chance.outcomes.get(okey)
        if child is None:                                          # outcome not yet seen this edge
            child = _Decision()
            chance.outcomes[okey] = child
            cont = _base_value(env, self.base, nloc, nbw, ncoll, world, lam)
        else:
            cont = self._simulate(env, child, nloc, nbw, ncoll, lam, rng, depth + 1)
        ret = step + cont
        self._backup_chance(chance, ret)
        return ret

    # ---- bandit + outcome keying -------------------------------------------------------
    def _ucb_select(self, node, actions):
        """UCB1: argmax_a Q̄(a) + c·sqrt(ln N(node) / N(a)). All `actions` are tried here (the
        expansion phase guarantees it), so every one has N(a) ≥ 1."""
        logN = math.log(node.n) if node.n > 0 else 0.0
        best_a, best_v = None, -math.inf
        c = self.c
        for a in actions:
            n_a = node.visits[a]
            exploit = node.reward[a] / n_a
            explore = c * math.sqrt(logN / n_a)
            v = exploit + explore
            if v > best_v:
                best_v, best_a = v, a
        return best_a

    @staticmethod
    def _outcome_key(a, world, env):
        """The binary observation outcome of action `a` in `world`: treasure present/absent, or
        detector positive/negative. This is the chance-node branch the traversal descends into."""
        kind, i = a
        if kind == "t":
            return bool((world >> i) & 1)
        return bool(world & env.cover_mask[i])

    # ---- backup ------------------------------------------------------------------------
    @staticmethod
    def _backup_decision(node, a, ret):
        node.visits[a] = node.visits.get(a, 0) + 1
        node.reward[a] = node.reward.get(a, 0.0) + ret
        node.n += 1

    @staticmethod
    def _backup_chance(chance, ret):
        chance.visits += 1
        chance.reward += ret
