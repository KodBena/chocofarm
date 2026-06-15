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
from dataclasses import dataclass
import numpy as np
from chocofarm.solvers.base import Policy, _base_value, UCB_C, GreedyStopBase
from chocofarm.model.env import (
    Action, Collected, Environment, Loc, TERMINATE, WorldSet, is_terminate,
)

# The information-set node identity (_belief_key): a (count, first, last) fingerprint, and
# a child edge key pairs the action taken with the successor belief's fingerprint.
BeliefKey = tuple[int, int, int]
ChildKey = tuple[Action, BeliefKey]


@dataclass(frozen=True)
class ISMCTSConfig:
    """Frozen scalar hyperparameters for `ISMCTSPolicy` (audit item I). The simulation `base`
    (a Policy, not a scalar) stays a separate __init__ param. Defaults match
    `ISMCTSPolicy.__init__` so a config built from the defaults is behaviour-identical. The config
    is the single coercion home: `__post_init__` applies the same `int()`/`float()` the old
    `__init__` did, so a config-built and a kwargs-built policy hold identical typed fields."""
    iterations: int = 300
    c: float = UCB_C
    max_depth: int = 24

    def __post_init__(self) -> None:
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "c", float(self.c))
        object.__setattr__(self, "max_depth", int(self.max_depth))


def _belief_key(bw: WorldSet) -> BeliefKey:
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

    def __init__(self) -> None:
        self.reward: dict[Action, float] = {}   # action -> summed playout return over selections
        self.visits: dict[Action, int] = {}     # action -> times this action was selected   (n_j)
        self.avail: dict[Action, int] = {}      # action -> times this action was available  (n'_j)
        self.children: dict[ChildKey, _Node] = {}   # (action, belief_key) -> child _Node


class ISMCTSPolicy(Policy):
    """Single-Observer Information Set MCTS. `iterations` determinized tree-walks per decision;
    `c` the UCB1 exploration constant (paper default 0.7). `base` is the simulation (playout)
    policy used at leaves and for the rollout to the end of the episode; `GreedyStopBase` by
    default (pass a `policies.GreedyPolicy()` to use the plainer greedy) — cheap, and the search
    supplies the contingent depth the base lacks."""

    def __init__(self, iterations: int = 300, c: float = UCB_C, base: Policy | None = None,
                 max_depth: int = 24, *, cfg: "ISMCTSConfig | None" = None) -> None:
        # cfg=ISMCTSConfig(...) supplies (iterations, c, max_depth) in one frozen object; the
        # individual kwargs remain the back-compat path and build the config when no cfg is passed
        # (ADR-0004). `base` (a Policy, not a scalar) is always a separate __init__ param. The config
        # is the single typed home (its __post_init__ does the int()/float()); the scalars decide()
        # reads are projected straight off it, so config-built and kwargs-built policies are equal.
        self.cfg = cfg if cfg is not None else ISMCTSConfig(iterations, c, max_depth)
        self.iterations = self.cfg.iterations
        self.c = self.cfg.c
        self.base = base if base is not None else GreedyStopBase()
        self.max_depth = self.cfg.max_depth

    # ---- public API ----
    def decide(self, env: Environment, loc: Loc, bw: WorldSet, collected: Collected,
               lam: float, rng: np.random.Generator | None = None) -> Action:
        # ADR-0002 fail-loud: ISMCTS is stochastic (it determinizes a world per iteration), so
        # it requires a real Generator — matches the seam's Optional-rng contract (base.py).
        assert rng is not None, "ISMCTSPolicy.decide requires a Generator (it determinizes worlds)"
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
    def _iterate(self, env: Environment, node: _Node, loc: Loc, bw: WorldSet,
                 collected: Collected, world: int, lam: float,
                 rng: np.random.Generator, depth: int) -> float:
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
        if is_terminate(a):
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

    def _expand_and_simulate(self, env: Environment, node: _Node, loc: Loc, bw: WorldSet,
                             collected: Collected, a: Action, world: int, lam: float,
                             rng: np.random.Generator, depth: int) -> float:
        """Realise a freshly expanded action under `world`, register its successor child, then
        (4) play the base policy to the end of the episode for the leaf estimate."""
        if is_terminate(a):
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
    def _ucb_select(self, node: _Node) -> Action:
        """UCB1 (eq. 7) with the subset-armed-bandit denominator: the log term uses the action's
        own availability count n'_j (times the action was legal here), per §IV-B."""
        best_a: Action | None = None
        best_v = -math.inf
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
        # `_ucb_select` is only called on a fully-expanded node (visits non-empty), so a
        # best is always found; ADR-0002 fail-loud rather than a None the caller can't use.
        assert best_a is not None
        return best_a

    @staticmethod
    def _update(node: _Node, a: Action, ret: float) -> None:
        node.visits[a] = node.visits.get(a, 0) + 1
        node.reward[a] = node.reward.get(a, 0.0) + ret
