#!/usr/bin/env python3
"""
chocofarm policy — Nested Monte-Carlo Search (Cazenave, IJCAI 2009), adapted to a
finite-horizon, stochastic, partially-observed belief-MDP.

NMCS in its original single-player form (Cazenave 2009, "Nested Monte-Carlo Search"):
a level-n search, at each step, tries every legal move, runs a level-(n-1) search from
each resulting state, keeps the move whose continuation scored best, plays it, and
recurses; level-0 is a base playout. The algorithm memorizes the best full sequence
found so far and *replays* it whenever a fresh nested search regresses below it
(Cazenave's "memorize the best sequence" rule — without it a nested search can wander
off a good line it had already found).

Three adaptations to this environment (see env.py for the contract; this file does NOT
modify it):

  (a) FINITE-HORIZON / EPISODIC. A line ends at TERMINATE, when no legal action remains,
      or at a hard step cap. TERMINATE is always a candidate move, so the search can
      choose to bank-and-exit early — the contingent "exit once a good basket is in hand"
      behaviour the project's clairvoyant ceiling shows is worth ~+70%.

  (b) STOCHASTIC, PARTIALLY-OBSERVED outcomes. The latent world is hidden, so a move's
      result (reward, next belief) is not deterministic. We DETERMINIZE by sampling a
      concrete world from the current belief `bw` (env.sample_world / rng.choice(bw) is
      unbiased — the belief already encodes consistency). A *playout* samples a world and
      plays a base policy to the end in it, scored by the lambda-penalized return
      `sum(value) - lam*(travel + exit)`. To cut determinization variance, both the
      level-0 playout score and the per-move evaluation inside a level-n step are averaged
      over a few sampled worlds.

  (c) The lambda passed to `decide` is the Dinkelbach penalty; every score in the search is
      the lambda-penalized return, so maximizing it is maximizing the renewal-reward
      objective at the current rate target.

The env drives policies one action at a time via `decide`. NMCS is an episode planner, so
`decide` runs a bounded NMCS search from the *current* observed state and returns only the
FIRST action of the best sequence it finds. Re-running per real step is the natural fit:
the real belief shrinks as the true episode observes, and re-planning over the sharpened
belief is exactly where the value-of-information is captured. Memory stays flat — the
search only ever holds one path and a handful of sampled worlds; it never enumerates or
caches the belief space (per the project's bounded-safety rule).
"""
from dataclasses import dataclass

import numpy as np

from chocofarm.model.env import (
    Action, Collected, Environment, Loc, MoveAction, TERMINATE, WorldSet, is_terminate,
)
from chocofarm.solvers.base import Policy, GreedyPolicy, _base_value, candidate_actions


@dataclass(frozen=True)
class NMCSConfig:
    """Frozen scalar hyperparameters for `NMCSPolicy` (audit item I). The level-0 `base`
    (a Policy, not a scalar) stays a separate __init__ param. Defaults match
    `NMCSPolicy.__init__` so a config built from the defaults is behaviour-identical."""
    level: int = 1
    playout_samples: int = 3
    step_samples: int = 2
    cand_det: int = 4
    cand_tre: int = 4
    max_steps: int = 24


class NMCSPolicy(Policy):
    """Nested Monte-Carlo Search as a pluggable Policy.

    Parameters
    ----------
    level : int
        NMCS nesting level (1 or 2 in this project's tests). Level 0 collapses to a
        single base playout from the root.
    base : Policy
        Base policy played at the leaf of a determinized playout (level-0). Defaults to
        GreedyPolicy. Played deterministically in a fixed sampled world.
    playout_samples : int
        Worlds sampled per playout score (variance reduction). The playout return is the
        mean lambda-value over this many independently sampled worlds.
    step_samples : int
        Worlds sampled when evaluating a candidate move at a level-n step. Each candidate's
        score is the mean over this many determinizations of (immediate lambda-step +
        nested level-(n-1) search from the resulting state).
    cand_det : int
    cand_tre : int
        Candidate pruning at each search step: the nearest `cand_det` still-informative
        detectors and nearest `cand_tre` still-uncollected treasures by travel distance,
        plus TERMINATE. Keeps the per-step branching bounded (the full legal set is ~36 at
        the root); the nearest-first restriction is the standard rate-aware pruning the
        existing RolloutPolicy already uses.
    max_steps : int
        Hard cap on the length of any search line (matches env.simulate's horizon).
    """

    def __init__(self, level: int = 1, base: Policy | None = None,
                 playout_samples: int = 3, step_samples: int = 2,
                 cand_det: int = 4, cand_tre: int = 4, max_steps: int = 24,
                 *, cfg: "NMCSConfig | None" = None) -> None:
        # cfg=NMCSConfig(...) supplies the scalar knobs in one frozen object; the individual
        # kwargs remain the back-compat path and build the config when no cfg is passed (ADR-0004).
        # `base` (the level-0 Policy, not a scalar) is always a separate __init__ param. The config
        # is the single home; the scalars decide() reads are projected straight off it. NB: the old
        # __init__ stored these knobs AS-PASSED (no int() coercion), so the config does NOT coerce —
        # behaviour-preserving means matching that, not adding a cast.
        self.cfg = cfg if cfg is not None else NMCSConfig(
            level, playout_samples, step_samples, cand_det, cand_tre, max_steps)
        self.base = base if base is not None else GreedyPolicy()
        self.level = self.cfg.level
        self.ps = self.cfg.playout_samples
        self.ss = self.cfg.step_samples
        self.cand_det = self.cfg.cand_det
        self.cand_tre = self.cfg.cand_tre
        self.max_steps = self.cfg.max_steps

    # ---- candidate generation (bounded branching) -------------------------------------
    def _candidates(self, env: Environment, loc: Loc, bw: WorldSet,
                    collected: Collected) -> list[Action]:
        # shared bounded-branching pruner; NMCS appends TERMINATE (bank-and-exit is always an option).
        return candidate_actions(env, loc, bw, collected, self.cand_det, self.cand_tre,
                                 include_terminate=True)

    # ---- level-0: determinized base playout from a state ------------------------------
    def _playout(self, env: Environment, loc: Loc, bw: WorldSet, collected: Collected,
                 lam: float, rng: np.random.Generator) -> float:
        """Mean lambda-value of a base-policy playout, averaged over `ps` sampled worlds.

        Each sampled world determinizes the whole rollout. `_base_value` (from policies.py)
        plays the base deterministically to the end in that fixed world and returns
        sum(value) - lam*(travel + exit). If the belief is empty (fully determined and
        nothing left) the only value is the exit penalty.
        """
        if len(bw) == 0:
            return -lam * env.exit_cost(loc)
        tot = 0.0
        for _ in range(self.ps):
            w = env.sample_world(bw, rng)
            tot += _base_value(env, self.base, loc, bw, collected, w, lam)
        return tot / self.ps

    # ---- level-n search from a state: returns (score, first_action) -------------------
    def _search(self, env: Environment, loc: Loc, bw: WorldSet, collected: Collected,
                lam: float, level: int, rng: np.random.Generator
                ) -> tuple[float, Action]:
        """Run an NMCS level-`level` search from (loc, bw, collected).

        Returns (score_of_best_line, first_action_of_best_line). For the env's `decide`
        contract only the first action of the root search is used, but the recursion needs
        the score of each child line to pick the best move at each step.

        Faithful to Cazenave's structure: walk the line forward; at each step evaluate
        every candidate by a level-(level-1) search of its result, take the best, but keep
        a memorized best-sequence and replay its move if this step's best regresses below
        the score already banked along the memorized line. Level 1's per-move evaluation is
        a base playout (the level-0 case).
        """
        cur_loc: Loc = loc
        cur_bw: WorldSet = bw
        cur_coll: Collected = set(collected)
        acc = 0.0                      # lambda-penalized reward accumulated along this line
        first_action: Action | None = None
        best_seq: list[Action] | None = None   # memorized best full continuation: list of moves
        best_seq_score = -np.inf       # its total score (from the start of this search)
        best_seq_first: Action | None = None

        for step in range(self.max_steps):
            cands = self._candidates(env, cur_loc, cur_bw, cur_coll)
            # If only TERMINATE is available, the line ends here.
            if cands == [TERMINATE]:
                if first_action is None:
                    first_action = TERMINATE
                line_score = acc - lam * env.exit_cost(cur_loc)
                if line_score > best_seq_score:
                    best_seq_score, best_seq_first = line_score, first_action
                return best_seq_score, (best_seq_first if best_seq_first is not None else TERMINATE)

            # Evaluate each candidate by a nested (level-1) search / playout of its result.
            best_q = -np.inf
            best_a: Action | None = None
            for a in cands:
                if is_terminate(a):
                    q = acc - lam * env.exit_cost(cur_loc)
                else:
                    q = acc + self._eval_move(env, cur_loc, cur_bw, cur_coll, a, lam, level, rng)
                if q > best_q:
                    best_q, best_a = q, a

            # cands is non-empty (the [TERMINATE]-only case returned above) and every q is a
            # real float > -inf, so best_a is always assigned; ADR-0002 fail-loud otherwise.
            assert best_a is not None
            if first_action is None:
                first_action = best_a

            # Memorize-and-replay: if this whole line so far is the best complete line we
            # have seen (i.e. taking best_a and stopping is better than the memorized one),
            # remember its first action. TERMINATE closes the line.
            if is_terminate(best_a):
                line_score = acc - lam * env.exit_cost(cur_loc)
                if line_score > best_seq_score:
                    best_seq_score, best_seq_first = line_score, first_action
                return best_seq_score, (best_seq_first if best_seq_first is not None else TERMINATE)

            # Otherwise play best_a forward in a determinized world and continue the line.
            w = env.sample_world(cur_bw, rng) if len(cur_bw) else None
            if w is None:
                line_score = acc - lam * env.exit_cost(cur_loc)
                if line_score > best_seq_score:
                    best_seq_score, best_seq_first = line_score, first_action
                return best_seq_score, (best_seq_first if best_seq_first is not None else TERMINATE)
            r, cur_loc, cur_bw, cur_coll, dt = env.apply(cur_loc, cur_bw, cur_coll, best_a, w)
            acc += r - lam * dt

            # Track the best complete line: closing here (exit now) gives this score.
            line_score = acc - lam * env.exit_cost(cur_loc)
            if line_score > best_seq_score:
                best_seq_score, best_seq_first = line_score, first_action

        # Horizon hit: close out.
        line_score = acc - lam * env.exit_cost(cur_loc)
        if line_score > best_seq_score:
            best_seq_score, best_seq_first = line_score, first_action
        return best_seq_score, (best_seq_first if best_seq_first is not None else TERMINATE)

    def _eval_move(self, env: Environment, loc: Loc, bw: WorldSet, collected: Collected,
                   a: MoveAction, lam: float, level: int,
                   rng: np.random.Generator) -> float:
        """Mean over `ss` determinizations of: immediate lambda-step of `a` + the value of
        a nested level-(level-1) search from the resulting state. At level 1 the nested
        value is a base playout (level-0)."""
        tot = 0.0
        for _ in range(self.ss):
            w = env.sample_world(bw, rng) if len(bw) else None
            if w is None:
                tot += -lam * env.exit_cost(loc)
                continue
            r, nloc, nbw, nc, dt = env.apply(loc, bw, collected, a, w)
            step = r - lam * dt
            if level <= 1:
                cont = self._playout(env, nloc, nbw, nc, lam, rng)
            else:
                cont, _ = self._search(env, nloc, nbw, nc, lam, level - 1, rng)
            tot += step + cont
        return tot / self.ss

    # ---- Policy interface --------------------------------------------------------------
    def decide(self, env: Environment, loc: Loc, bw: WorldSet, collected: Collected,
               lam: float, rng: np.random.Generator | None = None) -> Action:
        # ADR-0002 fail-loud: NMCS is stochastic (it determinizes sampled worlds per playout),
        # so it requires a real Generator — matches the seam's Optional-rng contract (base.py).
        assert rng is not None, "NMCSPolicy.decide requires a Generator (it samples worlds)"
        cands = self._candidates(env, loc, bw, collected)
        if cands == [TERMINATE]:
            return TERMINATE
        _, first_action = self._search(env, loc, bw, collected, lam,
                                       max(1, self.level), rng)
        return first_action
