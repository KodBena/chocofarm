#!/usr/bin/env python3
"""
chocofarm AZ — Gumbel-AlphaZero low-simulation search over chance nodes (design §5).

Public Domain (The Unlicense).

The guided "expert" of the ExIt loop (design §6). It layers Gumbel-AlphaZero root selection
(Danihelka et al. 2022) on the SO-ISMCTS information-set tree (Cowling et al. 2012; the same
belief-as-node / chance-node-as-observation scaffold `solvers/ismcts.py` implements), with the
net's masked policy as the interior PUCT prior and the net's VALUE at the leaf in place of the
determinized playout (the F4 cure, design §5.2). It produces, per decision:

  * the EXECUTED action, and
  * the IMPROVED-POLICY target π′ = softmax(completed_logits) over the fixed slot space
    (design §4.4 / §5.1), where completed_logits = logit + σ(completedQ),
    σ(q) = (c_visit + max_a N(a)) · c_scale · q  (Danihelka monotone transform, c_visit≈50),
    and unvisited root actions have their Q "completed" by the value-net mix v_mix.

Why Gumbel and not PUCT-at-root: PUCT needs many sims to be a sound policy improvement; at our
tiny budget (m=12, n=48) it can DEGRADE the prior. Gumbel guarantees improvement at low sim
counts (design §5.1), which is what makes an ExIt loop affordable on one CPU core (design §8).

Chance nodes (design §5.2): each action is followed by an observation outcome resolved by a
determinization w ~ bw; the successor belief is the child node. The action's statistics aggregate
over the information set (SO-ISMCTS contract). For leaf-value VARIANCE we average each leaf over
`c_outcome` (=2) determinizations of the IMMEDIATE outcome — light progressive widening over the
binary outcome (design §5.2). Interior selection is PUCT with net prior+value (design §5.2).

Search params (design §5.4): m=12 root actions, n=48 sims (Sequential Halving over
⌈log2 m⌉ rounds), c_puct=1.25, c_visit=50, c_outcome=2, c_scale=1.0. All overridable.

Faithfulness to Danihelka et al. 2022 (verified by tests/test_az_loop.py):
  * Sequential Halving runs ⌈log2 m⌉ phases, each phase splitting an equal N/⌈log2 m⌉ share among
    the current survivors, halving by `g + logit + σ(q̂)` (paper §2). The full n_sims budget is
    spent (rounding remainder goes to the last phase's survivors).
  * The executed action at temperature 0 IS the Sequential-Halving survivor (paper §2).
  * v_mix completes unvisited Q by the PRIOR-weighted mean of visited Q (paper §3), not the
    visit-weighted mean — the two differ sharply because SH makes visit counts unequal.

Honest simplifications (also in docs/results/az-exit-loop.md §caveats):
  * The interior tree uses one determinization PER SIMULATION (the ISMCTS contract: a single
    world drawn at the root threads the whole descent), while the LEAF outcome-averaging draws
    `c_outcome` immediate successors. We do NOT do full progressive widening at every interior
    chance node — the design only calls for it at the immediate (leaf) outcome (§5.2), and the
    info-set tree already aggregates over determinizations across sims.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from chocofarm.model.env import (
    Action, Collected, Environment, Loc, TERMINATE, WorldSet, is_terminate,
)
from chocofarm.az.features import FeatureBuilder
from chocofarm.az.actions import (n_action_slots, action_to_slot, slot_to_action,
                                  legal_mask_from_features, slot_action_tables)
from chocofarm.az.mlp import ValueMLP
from chocofarm.solvers.ismcts import _belief_key
from chocofarm.solvers.base import Policy
from chocofarm.az.value_target import (improved_policy as _improved_policy_rule,
                                       v_mix as _v_mix_rule,
                                       sigma_scale as _sigma_scale_rule)


class _Node:
    """One information-set node (belief). Per-action aggregate Q-statistics over the info set
    (ISMCTS contract); children keyed by (action, successor-belief-key). `prior`/`value`/`feat`/
    `mask` are the net's cached evaluation at this belief (one forward pass, reused across the
    node's action loop — the F7 amortization)."""
    __slots__ = ("W", "N", "children", "prior", "value", "feat", "mask", "legal")

    def __init__(self) -> None:
        self.W: dict[Action, float] = {}          # action -> summed λ-penalized return over selections
        self.N: dict[Action, int] = {}            # action -> selection count
        self.children: dict[tuple[Action, Any], _Node] = {}   # (action, belief_key) -> _Node
        self.prior: npt.NDArray[Any] | None = None    # (n_slots,) masked-softmax prior P(s,·)
        self.value: float | None = None    # scalar net value V_λ(belief)
        self.feat: npt.NDArray[Any] | None = None     # cached feature vector
        self.mask: npt.NDArray[Any] | None = None     # (n_slots,) legal mask
        self.legal: list[Action] | None = None    # list of legal actions at this node

    def q(self, a: Action) -> float:
        n = self.N.get(a, 0)
        return self.W[a] / n if n else 0.0


class GumbelAZSearch:
    """Gumbel-AlphaZero search bound to a value+policy net. Stateless across decisions except the
    FeatureBuilder cache. `decide_with_target(env, loc, bw, collected, lam, rng)` returns
    `(executed_action, improved_pi)` where improved_pi is the (n_slots,) probability target.

    `temperature` controls the EXECUTED action during generation: >0 samples from improved_pi
    (exploration), 0 takes argmax (eval). The improved-policy TARGET is unaffected by temperature
    (it is always the full distribution — the apprentice learns the improved policy, design §4.4).
    """

    def __init__(self, net: ValueMLP, env: Environment, m: int = 12, n_sims: int = 48,
                 c_puct: float = 1.25, c_visit: float = 50.0, c_scale: float = 1.0,
                 c_outcome: int = 2, max_depth: int = 24, use_jax_mlp: bool = False) -> None:
        self.net = net
        self.env = env
        self.fb = FeatureBuilder(env)
        self.m = int(m)
        self.n_sims = int(n_sims)
        self.c_puct = float(c_puct)
        self.c_visit = float(c_visit)
        self.c_scale = float(c_scale)
        self.c_outcome = int(c_outcome)
        self.max_depth = int(max_depth)
        self.n_slots = n_action_slots(env)
        self.term_slot = env.N + len(env.detectors)
        # hoist the slot<->action bijection tables once (the search converts millions of times in
        # its edge loops; per-call function dispatch was a measurable hot-path cost). Index these
        # directly in the inner loops instead of calling slot_to_action / action_to_slot.
        self._s2a, self._a2s = slot_action_tables(env)
        # Leaf-eval forward. DEFAULT is net.predict_both — the float32-numpy fast path (sgemm,
        # ~1.8× the float64 path at single-row dispatch). JAX-CPU LOST here: jit looks fast in a
        # hot microbench (same on-device array reused, ~34µs) but the search calls predict_both
        # one leaf at a time with FRESH numpy arrays, paying ~500µs/call of host↔device transfer
        # + dispatch — ~13× SLOWER than f32-numpy in the real loop (the maintainer's flagged
        # single-eval-dispatch trap; see docs/results/az-jax-perf.md). JAX only wins batched
        # (~4µs/item at batch 48), which the sequential tree descent doesn't expose without a
        # bigger restructure. `use_jax_mlp=True` keeps the jit path selectable for the bench.
        # the leaf-eval forward: net.predict_both (default numpy fast path) or the jax MlpJaxForward
        # (held-hard Stage-4 module, Any-typed). Both take (feat, mask) and return (value, policy);
        # typed as the value/policy callable so callers see the shared contract, not the union.
        self._predict_both: Callable[
            [npt.NDArray[Any], npt.NDArray[Any]],
            tuple[float | npt.NDArray[Any], npt.NDArray[Any]]]
        self._mlp_fwd: Any
        if use_jax_mlp and net.n_actions is not None:
            from chocofarm.az.mlp_jax import MlpJaxForward
            # mlp_jax is the held-hard Stage-4 module (C1); MlpJaxForward.__init__/__warmup are
            # untyped until C1 merges. Cast to Any so the constructor + warmup calls are Any-typed
            # (not no-untyped-call errors) — the honest escapes at this seam, matching _mlp_fwd: Any.
            _fwd_cls: Any = MlpJaxForward  # Any-escape: mlp_jax stub-gap (resolved when C1 types mlp_jax.py)
            fwd: Any = _fwd_cls(net)
            fwd.warmup(net.in_dim, net.n_actions)
            self._predict_both = fwd.predict_both
            self._mlp_fwd = fwd
        else:
            self._predict_both = net.predict_both
            self._mlp_fwd = None

    # ---- net evaluation (one forward, cached on the node) ----
    def _evaluate(self, node: _Node, loc: Loc, bw: WorldSet, collected: Collected) -> float:
        """Populate node.feat/mask/prior/value/legal from a single net forward pass.

        The marginals call is left to `FeatureBuilder.build`, which serves the belief-derived
        block (marginals included) from its per-belief cache when this belief was already seen in
        the episode — so we do NOT pre-compute `env.marginals` here (it would be discarded on a
        cache hit, the ~3.5× common case). On a cache miss `build` computes marginals once."""
        env = self.env
        feat = self.fb.build(loc, bw, collected)
        mask = legal_mask_from_features(env, feat)
        # `feat` is a 1-D vector, so predict_both returns the scalar-value arm (float, policy-row);
        # float() pins the value to the node's `float` slot (no runtime change — already a float).
        v_raw, p = self._predict_both(feat, mask)
        v = float(v_raw)
        node.feat = feat
        node.mask = mask
        node.prior = p
        node.value = v
        s2a = self._s2a
        node.legal = [s2a[s] for s in np.nonzero(mask)[0]]
        return v

    # ---- root-value bootstrap (Part B: the lower-variance value-target seam) ----
    @staticmethod
    def _root_search_value(root: _Node) -> float:
        """The search's ~n_sims-averaged estimate of the ROOT belief's value: the visit-weighted
        mean of the root actions' aggregate λ-penalized returns, Σ_a W[a] / Σ_a N[a].

        This is exactly the empirical MCTS root value — it averages ALL n_sims simulated returns
        (each `_visit` adds the realized return of one simulation to W and 1 to N), so it is a
        ~48-sample average of the same λ-penalized-return quantity the MC target measures, at no
        extra rollout cost (the sims were run for action selection anyway). It is the bootstrap
        used by the TD(λ)/n-step value target (Part B). Falls back to the net leaf value `root.value`
        when no sims landed (degenerate single-action / empty-considered case)."""
        sum_n = sum(root.N.values())
        if sum_n <= 0:
            return float(root.value) if root.value is not None else 0.0
        sum_w = sum(root.W.values())
        return sum_w / sum_n

    # ---- public API ----
    def decide_with_value(self, env: Environment, loc: Loc, bw: WorldSet, collected: Collected,
                          lam: float, rng: np.random.Generator, temperature: float = 0.0
                          ) -> tuple[Action, npt.NDArray[Any], float]:
        """Like `decide_with_target` but ALSO returns the search's root-value bootstrap
        `_root_search_value(root)` — the ~n_sims-averaged λ-penalized root value the Part B
        lower-variance value target bootstraps from. Returns `(executed_action, improved_pi,
        root_value)`. `decide_with_target` is the thin (action, pi) wrapper over this."""
        n_slots = self.n_slots
        if len(bw) == 0:
            pi = np.zeros(n_slots)
            pi[self.term_slot] = 1.0
            # the only continuation from an empty belief is to exit; its value is the exit toll
            return TERMINATE, pi, -lam * env.exit_cost(loc)
        action, pi, root = self._decide_root(env, loc, bw, collected, lam, rng, temperature)
        return action, pi, self._root_search_value(root)

    def decide_with_target(self, env: Environment, loc: Loc, bw: WorldSet, collected: Collected,
                           lam: float, rng: np.random.Generator, temperature: float = 0.0
                           ) -> tuple[Action, npt.NDArray[Any]]:
        n_slots = self.n_slots
        if len(bw) == 0:
            pi = np.zeros(n_slots)
            pi[self.term_slot] = 1.0
            return TERMINATE, pi
        action, pi, _root = self._decide_root(env, loc, bw, collected, lam, rng, temperature)
        return action, pi

    def _decide_root(self, env: Environment, loc: Loc, bw: WorldSet, collected: Collected,
                     lam: float, rng: np.random.Generator, temperature: float
                     ) -> tuple[Action, npt.NDArray[Any], _Node]:
        """Shared core: run the Gumbel search at the root and return (executed_action, improved_pi,
        root_node). The root node carries the per-action W/N stats the root-value bootstrap reads.
        `decide_with_target` and `decide_with_value` are thin wrappers selecting what to expose."""
        n_slots = self.n_slots

        # The belief-feature cache (FeatureBuilder._belief_cache) is scoped to ONE EPISODE: the
        # same belief recurs ~3.5× across an episode's decisions (and ~2.6× within one decision's
        # search tree), so episode-wide reuse beats per-decision. The episode boundary is detected
        # caller-agnostically: an episode's FIRST decision is the only one whose root belief is
        # the full world-set (`len(bw) == len(env.worlds)`) — every action filters to a strict
        # subset, so the full size never recurs mid-episode (asserted by the belief mechanics).
        # Resetting there bounds the cache to one episode's distinct beliefs (hundreds, small),
        # not the whole iteration. Correctness is cache-independent — a hit only ever returns
        # features of a belief that compared equal — so the reset is purely a memory bound.
        if len(bw) == len(env.worlds):
            self.fb.reset_belief_cache()

        root = _Node()
        self._evaluate(root, loc, bw, set(collected))
        # _evaluate just populated the root, so its legal-action list and prior are cached — assert
        # to narrow honestly (ADR-0002 fail-loud, not a None-deref on a freshly-evaluated node).
        legal = root.legal
        assert legal is not None and root.prior is not None
        legal_slots = [action_to_slot(env, a) for a in legal]

        # --- root logits = log(prior) over legal slots (Danihelka works in logit space; the
        #     masked-softmax prior is the reference distribution, so its log is the root logit). ---
        prior = root.prior
        logits = np.full(n_slots, -1e30)
        for s in legal_slots:
            logits[s] = math.log(max(prior[s], 1e-12))

        # --- Gumbel-Top-k: sample m root actions without replacement on logit + g ---
        g = rng.gumbel(size=n_slots)
        score0 = np.where(logits > -1e29, logits + g, -np.inf)
        m = min(self.m, len(legal_slots))
        considered = list(np.argsort(score0)[::-1][:m])   # top-m slots by logit+g

        # --- Sequential Halving over n_sims, dropping worst half each round by g+logit+σ(q̂);
        #     returns the SURVIVING slot (the action Danihelka selects to execute). ---
        survivor = self._sequential_halving(env, root, loc, bw, set(collected), lam, rng,
                                            considered, g, logits)

        # --- improved policy π′ = softmax(completed_logits) over the FULL legal set ---
        improved = self._improved_policy(root, logits, legal_slots)

        # --- executed action: the Sequential-Halving survivor (Danihelka §2 — the action that
        #     wins the bracket IS the executed action at temperature 0). When exploring, sample
        #     from π′ instead to diversify trajectories (the TARGET stays the raw π′). ---
        if temperature > 0:
            probs = improved.copy()
            if temperature != 1.0:
                with np.errstate(divide="ignore"):
                    lp = np.where(probs > 0, np.log(probs) / temperature, -np.inf)
                probs = self.net._masked_softmax(lp[None, :], (improved > 0)[None, :].astype(float))[0]
            chosen = int(rng.choice(self.n_slots, p=probs))
            exec_action = slot_to_action(env, chosen)
        else:
            # legal_slots is non-empty here (the root has at least TERMINATE), so SH returns a real
            # survivor slot, never None — assert it loudly (ADR-0002) before the slot lookup.
            assert survivor is not None
            exec_action = slot_to_action(env, survivor)
        return exec_action, improved, root

    # ---- Sequential Halving (Danihelka et al. 2022 §2) ----
    def _sequential_halving(self, env: Environment, root: _Node, loc: Loc, bw: WorldSet,
                            collected: Collected, lam: float, rng: np.random.Generator,
                            considered: list[int], g: npt.NDArray[Any],
                            logits: npt.NDArray[Any]) -> int | None:
        """Allocate `n_sims` across the `considered` root actions in `⌈log2 m⌉` phases, the
        paper's schedule: in each phase the surviving set each gets an equal share of the phase's
        budget, then the top half by `g + logit + σ(q̂)` survives. Returns the final survivor slot
        (the action executed at temperature 0). Any rounding remainder is spent on the survivors
        of the last phase so the full budget is used (no over/under-spend)."""
        considered = list(considered)
        if not considered:
            return None
        if len(considered) == 1:
            self._visit(env, root, loc, bw, collected, considered[0], lam, rng, self.n_sims)
            return considered[0]

        m = len(considered)
        n_phases = max(1, math.ceil(math.log2(m)))
        per_phase = max(1, self.n_sims // n_phases)   # paper's N/⌈log2 m⌉ phase budget
        budget = self.n_sims

        while len(considered) > 1 and budget > 0:
            # equal share of THIS phase's budget across the current survivors (paper §2)
            phase_budget = min(per_phase, budget)
            per_action = max(1, phase_budget // len(considered))
            for s in considered:
                v = min(per_action, budget)
                if v <= 0:
                    break
                self._visit(env, root, loc, bw, collected, s, lam, rng, v)
                budget -= v
            # drop the worst half by g + logit + σ(q̂)
            sigma = self._sigma_scale(root)
            scored = sorted(considered,
                            key=lambda s: g[s] + logits[s] + sigma * root.q(slot_to_action(env, s)),
                            reverse=True)
            considered = scored[:max(1, len(scored) // 2)]

        # spend any rounding remainder on the survivor(s) so the full budget is used
        i = 0
        while budget > 0 and considered:
            s = considered[i % len(considered)]
            self._visit(env, root, loc, bw, collected, s, lam, rng, 1)
            budget -= 1
            i += 1
        return considered[0]

    def _visit(self, env: Environment, root: _Node, loc: Loc, bw: WorldSet, collected: Collected,
               slot: int, lam: float, rng: np.random.Generator, count: int) -> None:
        """Run `count` simulations of root action `slot`, accumulating its aggregate stats."""
        a = slot_to_action(env, slot)
        for _ in range(count):
            w = env.sample_world(bw, rng)
            ret = self._simulate_root_action(env, root, loc, bw, collected, a, w, lam, rng)
            root.W[a] = root.W.get(a, 0.0) + ret
            root.N[a] = root.N.get(a, 0) + 1

    def _simulate_root_action(self, env: Environment, root: _Node, loc: Loc, bw: WorldSet,
                              collected: Collected, a: Action, world: int, lam: float,
                              rng: np.random.Generator) -> float:
        """One simulation of a chosen root action: realize it (chance outcome from `world`),
        average the leaf over c_outcome immediate determinizations (design §5.2), descend the
        interior with PUCT for the remaining depth. Returns the λ-penalized return."""
        if is_terminate(a):     # the seam's TypeIs guard narrows `a` to MoveAction for `apply` below
            return -lam * env.exit_cost(loc)
        # outcome-averaging over c_outcome determinizations of the IMMEDIATE outcome
        total = 0.0
        for k in range(self.c_outcome):
            w = world if k == 0 else env.sample_world(bw, rng)
            r, nloc, nbw, ncoll, dt = env.apply(loc, bw, collected, a, w)
            step = r - lam * dt
            ckey = (a, _belief_key(nbw))
            child = root.children.get(ckey)
            if child is None:
                child = _Node()
                root.children[ckey] = child
            cont = self._descend(env, child, nloc, nbw, ncoll, w, lam, rng, depth=1)
            total += step + cont
        return total / self.c_outcome

    # ---- interior PUCT descent; net value at the leaf (design §5.2) ----
    def _descend(self, env: Environment, node: _Node, loc: Loc, bw: WorldSet, collected: Collected,
                 world: int, lam: float, rng: np.random.Generator, depth: int) -> float:
        if depth >= self.max_depth or len(bw) == 0:
            if node.value is None:
                if len(bw) == 0:
                    return -lam * env.exit_cost(loc)
                return self._evaluate(node, loc, bw, collected)   # the value it just cached
            return node.value
        if node.value is None:
            # first visit to this leaf: net value IS the leaf estimate (no playout — the F4 cure)
            return self._evaluate(node, loc, bw, collected)       # the value it just cached

        a = self._puct_select(env, node)
        # the interior node is evaluated (node.value not None), so it has legal actions and
        # _puct_select returns one — assert it (ADR-0002 fail-loud, not a None action below).
        assert a is not None
        if is_terminate(a):     # the seam's TypeIs guard narrows `a` to MoveAction for `apply` below
            ret = -lam * env.exit_cost(loc)
            node.W[a] = node.W.get(a, 0.0) + ret
            node.N[a] = node.N.get(a, 0) + 1
            return ret
        r, nloc, nbw, ncoll, dt = env.apply(loc, bw, collected, a, world)
        step = r - lam * dt
        ckey = (a, _belief_key(nbw))
        child = node.children.get(ckey)
        if child is None:
            child = _Node()
            node.children[ckey] = child
        cont = self._descend(env, child, nloc, nbw, ncoll, world, lam, rng, depth + 1)
        ret = step + cont
        node.W[a] = node.W.get(a, 0.0) + ret
        node.N[a] = node.N.get(a, 0) + 1
        return ret

    def _puct_select(self, env: Environment, node: _Node) -> Action | None:
        """AlphaZero PUCT (Silver et al. 2017): argmax Q + c_puct·P·√(ΣN)/(1+N), over the legal
        actions, with the net prior P and Q the running mean (net value for unvisited via the
        node's own value as the optimistic-free baseline).

        Hot-loop form of the SAME formula, kept bit-identical: the slot lookup is the hoisted
        bijection dict (not the per-call wrapper), Q is inlined as `W[a]/n` (exactly `node.q(a)`)
        instead of via `node.q` (which re-did the `N.get`), and the U term keeps the original
        `c_puct * p * sqrt_total / (1 + n)` operation order with `p` still the numpy float64
        prior scalar — so the arithmetic, the result, and the strict-`>` argmax are unchanged."""
        N_map, W_map = node.N, node.W
        total_n = sum(N_map.values())
        sqrt_total = math.sqrt(total_n) if total_n > 0 else 1.0
        c_puct = self.c_puct
        prior = node.prior
        a2s = self._a2s
        # _puct_select is reached only after _evaluate populated this node (in _descend), so the
        # cached prior/legal are set — assert it to narrow honestly (ADR-0002, not a None-deref).
        assert prior is not None and node.legal is not None
        base_v = node.value if node.value is not None else 0.0
        best_a: Action | None = None
        best_v = -np.inf
        for a in node.legal:
            n = N_map.get(a, 0)
            q = (W_map[a] / n) if n else base_v   # unvisited Q completed by the node value (= node.q(a))
            p = prior[a2s[a]]
            v = q + c_puct * p * sqrt_total / (1 + n)
            if v > best_v:
                best_v, best_a = v, a
        return best_a

    # ---- improved policy + σ transform (Danihelka et al. 2022) ----
    #
    # The RULE lives in `value_target.improved_policy`/`v_mix`/`sigma_scale` (audit item C — the AZ
    # policy-target rule as pure, unit-testable functions of explicit inputs). The methods below are
    # thin adapters: they gather the live node's per-slot statistics (root.prior / root.value / N /
    # Q over the legal slots) into slot-indexed containers and delegate. The math and call order are
    # unchanged — byte-identical outputs (verified by the audit-item-C byte-identity check).

    def _node_visited_lists(self, root: _Node, legal_slots: list[int]
                            ) -> tuple[list[float], list[int]]:
        """Project the node's per-action W/N stats onto slot-indexed (visited_q, visited_n) lists
        over the legal slots — the explicit per-root-action inputs the value_target rule consumes.
        `visited_n[s]` = N(slot s) (0 if unvisited); `visited_q[s]` = Q(slot s) = root.q(action).

        These are PLAIN PYTHON lists of Python float / Python int — deliberately NOT numpy arrays.
        The welded rule arithmetised `root.q(a)` (a Python float, W[a]/n) and `root.N[a]` (a Python
        int) against the float32 `root.prior[s]`; numpy treats a Python float as a WEAK operand, so
        `prior[s] * root.q(a)` stays float32 while keeping Q's full float64 magnitude in the σ·q
        completion. A numpy float64 array would force the prior-weighted product to float64, and a
        float32 array would truncate Q — either way diverging from the welded rule. Python floats
        reproduce both exactly. This is the byte-identity seam (audit item C)."""
        env = self.env
        visited_q = [0.0] * self.n_slots
        visited_n = [0] * self.n_slots
        for s in legal_slots:
            a = slot_to_action(env, s)
            n = root.N.get(a, 0)
            if n > 0:
                visited_n[s] = n
                visited_q[s] = root.q(a)
        return visited_q, visited_n

    def _sigma_scale(self, root: _Node) -> float:
        # max_a N(a) over the visited root actions; the legal slots cover every visited action
        # (root.N keys are always legal), so the max over legal slots equals max(root.N.values()).
        assert root.legal is not None     # the root is evaluated before this adapter runs (ADR-0002)
        legal_slots = [action_to_slot(self.env, a) for a in root.legal]
        _, visited_n = self._node_visited_lists(root, legal_slots)
        return _sigma_scale_rule(visited_n, legal_slots, self.c_visit, self.c_scale)

    def _v_mix(self, root: _Node, legal_slots: list[int]) -> float:
        """Thin adapter: gather the node stats and call `value_target.v_mix` (Danihelka §3 value
        completion — the PRIOR-weighted blend the search uses for unvisited actions)."""
        visited_q, visited_n = self._node_visited_lists(root, legal_slots)
        # the root is evaluated before this adapter runs, so value/prior are cached (ADR-0002).
        assert root.value is not None and root.prior is not None
        return _v_mix_rule(root.value, visited_q, visited_n, root.prior, legal_slots)

    def _improved_policy(self, root: _Node, logits: npt.NDArray[Any],
                         legal_slots: list[int]) -> npt.NDArray[np.float64]:
        """Thin adapter: gather the node stats and call `value_target.improved_policy`
        (π′ = softmax(logit + σ(completedQ)) over the legal slots, Danihelka et al. 2022 §3).
        Returns a (n_slots,) probability row (zero on illegal)."""
        visited_q, visited_n = self._node_visited_lists(root, legal_slots)
        # the root is evaluated before this adapter runs, so value/prior are cached (ADR-0002).
        assert root.value is not None and root.prior is not None
        return _improved_policy_rule(logits, visited_q, visited_n, root.value, root.prior,
                                     legal_slots, self.c_visit, self.c_scale)


class GumbelPolicy(Policy):
    """Eval wrapper: a `Policy` that decides by argmax of the Gumbel improved policy (temperature
    0). Used to measure the apprentice's greedy rate via `env.dinkelbach_rate` (design §6 step 3).
    Construct with an already-loaded net; reuses one GumbelAZSearch across decisions."""

    def __init__(self, net: ValueMLP, env: Environment, m: int = 12, n_sims: int = 48,
                 **kw: Any) -> None:
        self.search = GumbelAZSearch(net, env, m=m, n_sims=n_sims, **kw)

    def decide(self, env: Environment, loc: Loc, bw: WorldSet, collected: Collected,
               lam: float, rng: np.random.Generator | None = None) -> Action:
        # GumbelPolicy is a STOCHASTIC search and requires a real Generator (the seam's Optional is
        # for the deterministic playout bases); assert it loudly rather than silently default-deref.
        assert rng is not None, "GumbelPolicy.decide requires a numpy Generator (ADR-0002)"
        action, _ = self.search.decide_with_target(env, loc, bw, collected, lam, rng,
                                                    temperature=0.0)
        return action
