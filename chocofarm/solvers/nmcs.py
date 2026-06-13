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
import numpy as np

from chocofarm.model.env import TERMINATE
from chocofarm.solvers.base import Policy, GreedyPolicy, _base_value


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

    def __init__(self, level=1, base=None, playout_samples=3, step_samples=2,
                 cand_det=4, cand_tre=4, max_steps=24):
        self.level = level
        self.base = base if base is not None else GreedyPolicy()
        self.ps = playout_samples
        self.ss = step_samples
        self.cand_det = cand_det
        self.cand_tre = cand_tre
        self.max_steps = max_steps

    # ---- candidate generation (bounded branching) -------------------------------------
    def _candidates(self, env, loc, bw, collected):
        marg = env.marginals(bw)
        dets = sorted(
            (i for i in env.detectors
             if np.any((bw & env.cover_mask[i]) != 0) and np.any((bw & env.cover_mask[i]) == 0)),
            key=lambda i: env.d(loc, ("d", i)))[:self.cand_det]
        tres = sorted(
            (i for i in range(env.N) if i not in collected and marg[i] > 0),
            key=lambda i: env.d(loc, ("t", i)))[:self.cand_tre]
        cands = [("d", i) for i in dets] + [("t", i) for i in tres]
        cands.append(TERMINATE)            # bank-and-exit is always an option
        return cands

    # ---- level-0: determinized base playout from a state ------------------------------
    def _playout(self, env, loc, bw, collected, lam, rng):
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
    def _search(self, env, loc, bw, collected, lam, level, rng):
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
        cur_loc, cur_bw, cur_coll = loc, bw, set(collected)
        acc = 0.0                      # lambda-penalized reward accumulated along this line
        first_action = None
        best_seq = None                # memorized best full continuation: list of moves
        best_seq_score = -np.inf       # its total score (from the start of this search)
        best_seq_first = None

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
            best_q, best_a = -np.inf, None
            for a in cands:
                if a == TERMINATE:
                    q = acc - lam * env.exit_cost(cur_loc)
                else:
                    q = acc + self._eval_move(env, cur_loc, cur_bw, cur_coll, a, lam, level, rng)
                if q > best_q:
                    best_q, best_a = q, a

            if first_action is None:
                first_action = best_a

            # Memorize-and-replay: if this whole line so far is the best complete line we
            # have seen (i.e. taking best_a and stopping is better than the memorized one),
            # remember its first action. TERMINATE closes the line.
            if best_a == TERMINATE:
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

    def _eval_move(self, env, loc, bw, collected, a, lam, level, rng):
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
    def decide(self, env, loc, bw, collected, lam, rng):
        cands = self._candidates(env, loc, bw, collected)
        if cands == [TERMINATE]:
            return TERMINATE
        _, first_action = self._search(env, loc, bw, collected, lam,
                                       max(1, self.level), rng)
        return first_action if first_action is not None else TERMINATE
