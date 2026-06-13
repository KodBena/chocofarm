#!/usr/bin/env python3
"""
decomp_solver.py — the EXACT HIERARCHICAL DECOMPOSITION solver for chocofarm.

The flat belief-MDP over C(20,5)=15,504 worlds is intractable to solve exactly,
and the approximate-search pack (NMCS / ISMCTS / rollout / sparse-sampling) sits
*below* the static floor — the value-of-information is gated behind face-read
CHAINS too deep for shallow search to pay for (docs/design/static-analysis-faces.md).

But the problem FACTORS (analyzer.py / the faces note):

  * Treasures partition into co-coverage CLUSTERS — small, sense-isolated blobs:
        SE+mid {0,1,2,13,14,15}   NW {8,9,10,11,12}   N {5,6,7}   S {17,18}
    plus four sense-isolated δ-singletons {3},{4},{16},{19} (observe == collect).
  * Conditioned on a cluster's OCCUPANCY k (how many of the 5 present treasures
    lie in it), the worlds factor as a uniform draw of k of the cluster's members
    (`#worlds = ∏_c C(size_c, k_c)`, the DET-IND keystone, exact).  The sole global
    coupling is Σ_c k_c = K = 5.
  * Each occupancy-conditioned per-cluster belief-MDP is microscopic — the
    occupancy-conditioned reachable-belief counts top out at 558 (SE+mid, k=3) —
    so EXACT backward induction is tractable.

This module builds the two layers the faces note (§6) recommends:

  MICRO  (build_cluster_micro):  for each (cluster, occupancy k) an EXACT
         λ-penalised belief-MDP solved by backward induction over the
         occupancy-conditioned local belief lattice.  State =
         (local-position, support of surviving size-k local present-sets,
         locally-collected set).  Actions = in-cluster face reads (every face id
         whose cover ⊆ cluster), in-cluster collects, and LEAVE (boundary action).
         Output: V*(state), the optimal action at each state, and the exact
         (E[reward | k], E[time | k]) of entering the cluster and following π*
         to the boundary.

  MACRO  (MacroPlanner):  a receding-horizon exact expectimax over the
         cluster-visit sequence of ONE excursion (entry → ordered subset of
         clusters → exit), with the occupancy posterior tracked EXACTLY as a
         multivariate hypergeometric over Σ k_c = 5.  Entering a cluster yields
         the micro layer's (E[R|k], E[T|k]) for the realised k and reveals k
         (the micro's own reads resolve it).  The macro decides which cluster to
         enter, when to bank-and-exit, and — via env.nearest_exit at TERMINATE —
         which teleport to leave by.

  POLICY (DecompPolicy):  wraps both layers behind the env's `Policy` interface.
         Precomputes the micro tables for the live λ (cached per λ), then at each
         env step decodes the global state to the active cluster's local state,
         replays π* (translated back to the env's ('d', face_id) / ('t', tre_id)
         actions), and consults the macro at cluster boundaries.

HONEST CAVEATS (also in docs/results/decomp-rate.md):
  * Micro within-occupancy uniformity is EXACT (the ∏ C(size,k) factorization), and
    the JOINT (runtime) micro carries per-latent completion-count weights
    C(N−size, K−j) so it is exact under the true env prior across occupancies too.
  * Macro occupancy posterior is recomputed EXACTLY from the live belief bw at every
    macro decision (project worlds onto cells, group-count) — it reflects every
    reveal so far with no incremental bookkeeping.  The cluster-ENTER look-ahead
    conditions the full joint and is exact at every depth.
  * Macro horizon: the SHIPPED default is horizon=1 (myopic — enter the single best
    cluster, re-evaluate at its boundary; it re-plans every boundary so the realised
    policy is not horizon-truncated).  At horizon=1 the macro value is exact.  The
    deeper-horizon expectimax is kept for inspection; its δ-DIP branch at horizon≥2
    holds the δ-pool occupancy at its prior across a chain of δ dips — an
    independence/staleness approximation confined to that look-ahead path, which
    never biases the measured rate (the env is ground truth) but degrades a deeper
    plan's δ valuation.  See MacroPlanner.value for the scoped comment.
  * Macro re-anchoring is a decision-only geometric approximation (the planner scores
    a cluster's travel as entry→anchor; the runtime corrects the executed action
    exactly from the live loc).  It can pick a slightly sub-optimal next cluster; it
    does not bias the rate.

Boundedness: every table is built under hard caps; nothing enumerates the global
belief.  Pin any run to CPU core 3 under `timeout` (see eval_decomp.py).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from functools import lru_cache

from chocofarm.model.env import Environment, TERMINATE


# ===========================================================================
# Cluster structure (read off the env's faces; matches analyzer.clusters)
# ===========================================================================

def _cocoverage_clusters(env: Environment):
    """Connected components of the treasure co-coverage hypergraph — exactly
    analyzer.clusters, recomputed here from the env's own face cover_masks so the
    solver and the env agree on the partition by construction."""
    parent = {t: t for t in range(env.N)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for fid in env.detectors:
        cover = [t for t in range(env.N) if (env.cover_mask[fid] >> t) & 1]
        for a, b in itertools.combinations(cover, 2):
            parent[find(a)] = find(b)
    comp = {}
    for t in range(env.N):
        comp.setdefault(find(t), set()).add(t)
    return sorted((sorted(s) for s in comp.values()), key=lambda s: (-len(s), s))


@dataclass(frozen=True)
class Cluster:
    """A sense-cluster (size>1) or a δ-singleton (size==1).  `faces` are the env
    face ids whose cover lies entirely inside the cluster (the in-cluster sense
    actions); `tres` the member treasure ids (sorted, the local bit order)."""
    name: str
    tres: tuple
    faces: tuple          # env face ids with cover ⊆ tres

    @property
    def size(self):
        return len(self.tres)


def discover_clusters(env: Environment):
    parts = _cocoverage_clusters(env)
    out = []
    for c in parts:
        cset = set(c)
        fids = tuple(fid for fid in env.detectors
                     if env.cover_mask[fid] and all(
                         ((env.cover_mask[fid] >> t) & 1) == 0 or t in cset
                         for t in range(env.N)))
        # name by centroid quadrant (cosmetic) / δ for singletons
        if len(c) == 1:
            name = f"d{c[0]}"
        else:
            name = "+".join(str(t) for t in c)
        out.append(Cluster(name=name, tres=tuple(c), faces=fids))
    return out


# ===========================================================================
# MICRO — exact occupancy-conditioned per-cluster belief-MDP
# ===========================================================================

@dataclass
class MicroSolution:
    """The exact solve of ONE (cluster, occupancy k) sub-problem at a fixed λ,
    entered from a fixed boundary point.

    `value[state]`      : V*(state) = max over local plans of (Σ value − λ·time)
                          from `state` to the boundary, EXCLUDING the cost already
                          paid to reach `state`.
    `act[state]`        : the optimal env action at `state` (('d',fid)/('t',tre)/LEAVE).
    `enter_ev`          : (E[reward], E[time]) of entering the cluster at occupancy
                          k from the boundary and following π* to LEAVE — exact,
                          averaged over the C(size,k) equiprobable present-sets.
    `enter_value`       : enter_ev[0] − λ·enter_ev[1] (the λ-value the macro chains).
    """
    cluster: str
    k: int
    n_states: int
    value: dict            # (loc, support, collected) -> V*
    act: dict              # (loc, support, collected) -> optimal action
    enter_ev: tuple
    enter_value: float
    face_localmask: dict = field(default_factory=dict)   # env face id -> local cover mask
    wmap: dict = field(default_factory=dict)             # local present-set -> prior weight
    bit: dict = field(default_factory=dict)              # treasure id -> local bit
    solve: object = None                                 # the memoised exact solver closure


LEAVE = ("leave", None)


def build_cluster_micro(env: Environment, cluster: Cluster, k, lam: float,
                        entry_loc, max_states: int = 200_000) -> MicroSolution:
    """EXACT backward induction for a per-cluster belief-MDP under λ.

    Two construction modes, both EXACT:
      * k an int → the OCCUPANCY-CONDITIONED MDP: the initial belief is the support
        of the C(size,k) local present-sets of size exactly k.  By the occupancy
        factorization the env prior is UNIFORM over these (exact), so a support of
        cardinality m carries 1/m — no weights.  This yields the exact
        (E[R|k], E[T|k]) the MACRO chains, per the task's "conditioned on the
        cluster's occupancy" framing.
      * k is None → the JOINT (occupancy-marginal) MDP: the initial belief is ALL
        2**size local subsets (every present-set, every k), the unconditioned local
        belief lattice the analyzer sized (745 / 1448 reachable beliefs).  This is
        what the RUNTIME executes, because a priori the cluster's occupancy is not
        known — the policy must act on the mixed-k belief, and the joint VI handles
        it exactly (no per-k mixture approximation).

    Belief = the SUPPORT (a frozenset) of surviving local present-set bitmasks over
    `cluster.tres`'s order.  State = (loc, belief_support, collected_local).  Actions:

      ('d', fid)  read in-cluster face fid (cost d(loc, face)); splits the support
                  by the disjunction over its cover; expectimax over the two
                  polarities weighted by support fractions.
      ('t', tre)  collect member `tre` (cost d(loc, tre)); reward = value on the
                  present branch; splits support by presence; collected grows there.
      LEAVE       stop the sub-episode (no extra cost — the exit cost is the MACRO's,
                  charged once when the excursion ends).

    Backward induction is well-founded: every face/collect shrinks the support or
    grows `collected` (both monotone), so the state graph is a memoised DAG.
    Boundedness: per-k reachable supports ≤558, joint ≤~34k states (analyzer +
    measured); `max_states` aborts loudly (ADR-0002) on an over-cap synthetic blob.
    """
    tres = cluster.tres
    size = cluster.size
    bit = {t: b for b, t in enumerate(tres)}           # treasure id -> local bit
    if k is None:
        latents = frozenset(range(1 << size))          # JOINT: every local subset
    else:
        latents = frozenset(sum(1 << bit[tres[i]] for i in combo)
                            for combo in itertools.combinations(range(size), k))
    # Per-latent prior WEIGHT.  The env's posterior over a cluster's local config is
    # NOT uniform across occupancies: a local config with j present is completed by
    # C(N−size, K−j) global worlds.  Within a fixed k all j are equal so the per-k
    # MDP is uniform (weight 1 each — exact).  The JOINT MDP must carry these
    # completion-count weights to be exact under the true env prior.
    def _completion(j):
        r = env.K - j                                  # remaining present outside the cluster
        if r < 0 or r > env.N - size:
            return 0                                    # infeasible occupancy under Σ=K
        return math.comb(env.N - size, r)
    wmap = {s: _completion(bin(s).count("1")) for s in latents}
    wmap = {s: w for s, w in wmap.items() if w > 0}    # drop K-infeasible configs
    latents = frozenset(wmap)

    # in-cluster face cover as a LOCAL bitmask
    face_localmask = {}
    for fid in cluster.faces:
        lm = 0
        for t in tres:
            if (env.cover_mask[fid] >> t) & 1:
                lm |= (1 << bit[t])
        face_localmask[fid] = lm

    def wsum(belief):
        return sum(wmap[s] for s in belief)

    state_count = [0]
    act_map = {}                                       # (loc, belief, collected) -> action
    val_map = {}                                       # (loc, belief, collected) -> V*

    @lru_cache(maxsize=None)
    def solve(loc, belief, collected):
        """belief: frozenset of local present-set masks (the support), carrying the
        prior weights `wmap`.  collected: treasure ids already collected here.
        Returns (V, best_action).  Expectimax branches weighted by `wsum`."""
        state_count[0] += 1
        if state_count[0] > max_states:
            raise RuntimeError(
                f"micro {cluster.name} k={k}: state cap {max_states} exceeded "
                f"(cluster too large to solve flat — sub-decompose)")
        W = wsum(belief)
        best_v, best_a = 0.0, LEAVE          # LEAVE now → value 0 (exit is the macro's)

        # --- collect actions: only members still possibly-present & uncollected ---
        for t in tres:
            if t in collected:
                continue
            b = bit[t]
            present = frozenset(s for s in belief if (s >> b) & 1)
            absent = frozenset(s for s in belief if not ((s >> b) & 1))
            if not present:
                continue                                  # certainly absent — pointless
            cost = env.d(loc, ("t", t))
            wp, wa = wsum(present), wsum(absent)
            vp = env.value[t] + solve(("t", t), present, collected | {t})[0]
            va = solve(("t", t), absent, collected)[0] if absent else 0.0
            q = (wp * vp + wa * va) / W - lam * cost
            if q > best_v + 1e-12:
                best_v, best_a = q, ("t", t)

        # --- face-read actions: only faces whose outcome is still uncertain ---
        for fid, lm in face_localmask.items():
            if lm == 0:
                continue
            hit = frozenset(s for s in belief if (s & lm))
            miss = frozenset(s for s in belief if not (s & lm))
            if not hit or not miss:
                continue                                  # uninformative on this support
            cost = env.d(loc, ("d", fid))
            wh, wm = wsum(hit), wsum(miss)
            vh = solve(("d", fid), hit, collected)[0]
            vm = solve(("d", fid), miss, collected)[0]
            q = (wh * vh + wm * vm) / W - lam * cost
            if q > best_v + 1e-12:
                best_v, best_a = q, ("d", fid)

        act_map[(loc, belief, collected)] = best_a
        val_map[(loc, belief, collected)] = best_v
        return best_v, best_a

    # roll up the exact (E[R], E[T]) under π* by replaying it as a weighted expectation.
    def expectation(loc, belief, collected):
        """Exact (E[R], E[T]) of following π* from this state to LEAVE, over the
        weighted support."""
        W = wsum(belief)
        _, a = solve(loc, belief, collected)
        if a == LEAVE:
            return 0.0, 0.0
        kind, i = a
        cost = env.d(loc, (kind, i))
        if kind == "t":
            b = bit[i]
            present = frozenset(s for s in belief if (s >> b) & 1)
            absent = frozenset(s for s in belief if not ((s >> b) & 1))
            wp, wa = wsum(present), wsum(absent)
            er = et = 0.0
            if wp:
                rp, tp = expectation(("t", i), present, collected | {i})
                er += wp * (env.value[i] + rp); et += wp * tp
            if wa:
                ra, ta = expectation(("t", i), absent, collected)
                er += wa * ra; et += wa * ta
            return er / W, cost + et / W
        # face read
        lm = face_localmask[i]
        hit = frozenset(s for s in belief if (s & lm))
        miss = frozenset(s for s in belief if not (s & lm))
        wh, wm = wsum(hit), wsum(miss)
        er = et = 0.0
        if wh:
            rh, th = expectation(("d", i), hit, collected)
            er += wh * rh; et += wh * th
        if wm:
            rm, tm = expectation(("d", i), miss, collected)
            er += wm * rm; et += wm * tm
        return er / W, cost + et / W

    init = (entry_loc, latents, frozenset())
    if k == 0 or not latents:
        # empty cluster: nothing to do
        return MicroSolution(cluster.name, k, 1, {}, {init: LEAVE}, (0.0, 0.0), 0.0,
                             face_localmask=face_localmask, wmap=wmap, bit=bit, solve=solve)

    v0, _ = solve(*init)
    eR, eT = expectation(*init)
    return MicroSolution(
        cluster=cluster.name, k=k, n_states=state_count[0],
        value=val_map, act=act_map, enter_ev=(eR, eT),
        enter_value=eR - lam * eT, face_localmask=face_localmask, wmap=wmap, bit=bit,
        solve=solve,
    )


# ===========================================================================
# Occupancy posterior — exact multivariate-hypergeometric bookkeeping
# ===========================================================================

def _occupancy_posterior(cell_sizes, budget):
    """[exact] Analytic PRIOR over occupancy vectors (k_0,…,k_{n-1}) with Σ k_c =
    budget and 0≤k_c≤cell_sizes[c], each world equiprobable: P(k) ∝ ∏ C(size_c, k_c).
    Returns {tuple(k): probability} — the analyzer's multivariate hypergeometric over
    ≤320 feasible vectors.  At RUNTIME the macro does not use this; it recomputes the
    posterior from the live belief (`_live_occupancy_posterior`), which reduces to
    exactly this on the full world set.  Kept as the analytic reference / sanity check
    (the prior the live recompute must match before any reveal)."""
    sizes = list(cell_sizes)
    vecs = []

    def rec(i, rem, acc):
        if i == len(sizes):
            if rem == 0:
                vecs.append(tuple(acc))
            return
        for kc in range(0, min(sizes[i], rem) + 1):
            rec(i + 1, rem - kc, acc + [kc])

    rec(0, budget, [])
    weights = {}
    tot = 0.0
    for v in vecs:
        w = 1
        for s, kc in zip(sizes, v):
            w *= math.comb(s, kc)
        weights[v] = w
        tot += w
    return {v: w / tot for v, w in weights.items()} if tot else {}


def _marginal_k(posterior, idx):
    """Marginal P(k_idx = j) from a joint posterior dict."""
    out = {}
    for v, p in posterior.items():
        out[v[idx]] = out.get(v[idx], 0.0) + p
    return out


# ===========================================================================
# MACRO — receding-horizon exact expectimax over the cluster-visit sequence
# ===========================================================================

class MacroPlanner:
    """The macro layer.  Given the live occupancy posterior and current location,
    decides the next macro move: enter a particular cluster, or bank-and-exit.

    The latent is the cluster-occupancy vector (multivariate hypergeometric over
    Σ k_c = K).  Entering a cluster reveals its occupancy (the micro's reads
    resolve it) and yields the micro layer's EXACT (E[R|k], E[T|k]).  The planner
    is an expectimax to depth `horizon` over (which cluster next | exit); it is
    re-invoked every boundary, so the realised policy is not horizon-truncated —
    only each single look-ahead is.

    Cell convention: cells = [sense-clusters in `self.clusters` order] + [δ-cell],
    where the δ-cell aggregates the four sense-isolated singletons (symmetric, all
    observe==collect).  Visiting the δ-cell collects the single cheapest still-
    plausible δ-treasure (the macro treats δ as a pool drawn down one at a time)."""

    def __init__(self, env, clusters, micro, lam, horizon=1):
        self.env = env
        self.lam = lam
        self.horizon = horizon
        self.sense = [c for c in clusters if c.size > 1]
        self.delta = [c.tres[0] for c in clusters if c.size == 1]   # δ treasure ids
        self.micro = micro                          # {(cluster_name,k): MicroSolution}
        # cell sizes: one per sense cluster, then the δ pool
        self.cell_sizes = [c.size for c in self.sense] + [len(self.delta)]
        self.anchor = {c.name: min(c.tres, key=lambda t: env.d(("w", env.entry), ("t", t)))
                       for c in self.sense}

    # ---- micro lookups ----
    def _micro_ev(self, cname, k):
        sol = self.micro.get((cname, k))
        return sol.enter_ev if sol else (0.0, 0.0)

    # ---- expectimax value of a macro state ----
    def value(self, loc, posterior, visited, delta_done, depth):
        """V*(macro state) = max over {exit, enter unvisited cluster, dip δ} of the
        expected λ-value (Σ reward − λ·(travel + exit)).  posterior: joint over the
        n_cells occupancy vector.  visited: set of sense-cluster indices done.
        delta_done: frozenset of δ-treasure ids already collected.  Pure λ-value
        (the exit cost is included so 'exit now' is comparable)."""
        # base option: exit now
        best = -self.lam * self.env.exit_cost(loc)
        best_move = ("exit", None)
        if depth <= 0:
            return best, best_move

        # enter an unvisited sense cluster
        for ci, c in enumerate(self.sense):
            if ci in visited:
                continue
            mk = _marginal_k(posterior, ci)
            travel = self.env.d(loc, ("t", self.anchor[c.name]))
            ev = -self.lam * travel
            for k, pk in mk.items():
                eR, eT = self._micro_ev(c.name, k)
                # condition posterior on k_ci = k, then recurse from the anchor.
                # The recursive value() ends in exactly one 'exit' leaf, which is the
                # single place the final exit cost is charged — no add-back here.
                cond = {v: p for v, p in posterior.items() if v[ci] == k}
                tot = sum(cond.values())
                cond = {v: p / tot for v, p in cond.items()} if tot else {}
                cont, _ = self.value(("t", self.anchor[c.name]), cond,
                                     visited | {ci}, delta_done, depth - 1)
                ev += pk * (eR - self.lam * eT + cont)
            if ev > best:
                best, best_move = ev, ("enter", ci)

        # dip the δ pool (collect the cheapest still-uncollected δ).  The δ cell's
        # marginal occupancy gives P(a given uncollected δ present) = E[k_δ]/|δ pool|.
        #
        # EXACTNESS SCOPE (honest): at the default horizon=1 this branch recurses to
        # depth 0 (exit), so the δ posterior is used for exactly one decision and the
        # value is exact.  At horizon≥2 this branch recurses with the *unconditioned*
        # `posterior` and a `p_present` denominator that does NOT shrink as δ are
        # collected — i.e. the δ-cell occupancy is held at its prior across a chain of
        # δ dips.  That is an independence/staleness APPROXIMATION confined to the
        # horizon≥2 δ-dip look-ahead (the cluster-enter branch above conditions the
        # full joint and is exact at every depth).  It never biases the measured rate
        # (the env recomputes the live posterior from bw each macro decision and
        # charges exact travel); it only degrades a deeper plan's δ valuation.  The
        # shipped horizon=1 policy does not exercise it.
        remaining = [t for t in self.delta if t not in delta_done]
        if remaining:
            di = len(self.sense)                       # δ cell index
            mk = _marginal_k(posterior, di)
            exp_kd = sum(j * p for j, p in mk.items())
            p_present = exp_kd / max(1, len(self.delta))
            cheapest = min(remaining, key=lambda t: self.env.d(loc, ("t", t)))
            travel = self.env.d(loc, ("t", cheapest))
            cont, _ = self.value(("t", cheapest), posterior, visited,
                                 delta_done | {cheapest}, depth - 1)
            ev = -self.lam * travel + p_present * 1.0 + cont
            if ev > best:
                best, best_move = ev, ("delta", cheapest)

        return best, best_move

    def decide_macro(self, loc, posterior, visited, delta_done):
        """Top-level macro move from the current boundary state."""
        _, move = self.value(loc, posterior, visited, delta_done, self.horizon)
        return move


# ===========================================================================
# POLICY — wrap micro + macro behind the env's Policy interface
# ===========================================================================

class DecompPolicy:
    """Exact hierarchical-decomposition policy, behind the env's Policy interface
    (policies.Policy.decide(env, loc, bw, collected, lam, rng)).

    Precomputes (lazily, cached per λ) the per-(cluster,k) micro VALUE functions
    and the macro planner, then executes them on the honest env by EXACT one-step
    lookahead:

      * Episode start (entry teleport, nothing collected, full belief) → reset the
        per-episode visit/δ bookkeeping, then consult the macro (which recomputes the
        exact occupancy posterior from the live belief) for the first cluster / δ /
        exit.
      * Inside a cluster → decode the live local belief support and query the
        cluster's JOINT (occupancy-marginal) exact belief-MDP solver for π* at the
        live (loc, support, collected).  The solver is memoised and exact for ANY
        support — including the ones the global Σ=K coupling conditions when other
        clusters' occupancies are already revealed — so there is no occupancy
        mixture and no fallback.
      * Macro 'exit' → TERMINATE; env charges the single exit cost.

    `horizon` is the macro look-ahead depth.  The SHIPPED default is horizon=1 (the
    myopic macro: enter the single best cluster, re-evaluate at its boundary) — it is
    the simplest, and at horizon=1 the macro value is EXACT (no recursion, so the
    δ-dip staleness approximation that the deeper look-ahead carries is never
    exercised; see MacroPlanner.value).  Empirically (eval_decomp.py horizon sweep)
    horizon=2 is within the N=2000 standard error of horizon=1 — they are
    statistically indistinguishable — so the simplest, exact one is the default; the
    deeper expectimax is kept for inspection.

    The env is ground truth: every distance the measured rate sees is exact env.d.
    The micro/macro tables only steer decisions; the rate is unbiased."""

    def __init__(self, horizon=1, verbose=False):
        self.horizon = horizon
        self.verbose = verbose
        self._cache = {}                 # lam(rounded) -> built tables
        self._ep = None                  # per-episode mutable state
        self.fallbacks = 0               # states solved on-demand (conditioned supports)

    # ---- precompute (per λ) ----
    def _build(self, env, lam):
        key = round(lam, 6)
        if key in self._cache:
            return self._cache[key]
        clusters = discover_clusters(env)
        sense = [c for c in clusters if c.size > 1]
        anchors = {c.name: min(c.tres, key=lambda t: env.d(("w", env.entry), ("t", t)))
                   for c in sense}
        micro = {}        # per-(cluster,k) occupancy-conditioned — for the MACRO
        joint = {}        # per-cluster joint (all-k) — for the RUNTIME execution
        for c in sense:
            entry = ("t", anchors[c.name])
            for k in range(1, c.size + 1):
                micro[(c.name, k)] = build_cluster_micro(env, c, k, lam, entry)
            joint[c.name] = build_cluster_micro(env, c, None, lam, entry)
        macro = MacroPlanner(env, clusters, micro, lam, horizon=self.horizon)
        built = {"clusters": clusters, "sense": sense, "anchors": anchors,
                 "micro": micro, "joint": joint, "macro": macro,
                 "bit": {c.name: {t: b for b, t in enumerate(c.tres)} for c in sense}}
        self._cache[key] = built
        return built

    # ---- per-episode reset ----
    def _reset(self, env, lam):
        self._build(env, lam)
        # NB: the occupancy posterior is NOT stored here — `decide` recomputes it
        # exactly from the live belief bw at every macro decision
        # (_live_occupancy_posterior), so there is no incremental-posterior slot to
        # keep in sync.  Episode state is the visit/δ bookkeeping and the active cluster.
        self._ep = {"lam": round(lam, 6), "visited": set(),
                    "delta_done": frozenset(), "active": None}

    # ---- live local belief decode (vectorised projection onto the cluster bits) ----
    @staticmethod
    def _local_support(bw, cluster, bm):
        cmask = sum(1 << t for t in cluster.tres)
        proj = bw & cmask
        local = proj * 0
        for t in cluster.tres:
            local = local | (((proj >> t) & 1) << bm[t])
        return frozenset(int(x) for x in set(local.tolist()))

    # ---- the Policy interface ----
    def decide(self, env, loc, bw, collected, lam, rng=None):
        fresh = (loc == ("w", env.entry) and not collected and len(bw) == len(env.worlds))
        if self._ep is None or self._ep["lam"] != round(lam, 6) or fresh:
            self._reset(env, lam)
        built = self._build(env, lam)
        macro, sense = built["macro"], built["sense"]
        ep = self._ep

        # Outside any cluster → consult the macro.  The occupancy posterior is
        # recomputed EXACTLY from the live belief bw (project each surviving world
        # onto the cells and count) — so every reveal so far (cluster chains, δ
        # collects) is reflected without incremental bookkeeping.
        if ep["active"] is None:
            post = _live_occupancy_posterior(env, bw, macro)
            move = macro.decide_macro(loc, post, ep["visited"], ep["delta_done"])
            kind = move[0]
            if kind == "exit":
                return TERMINATE
            if kind == "delta":
                ep["delta_done"] = ep["delta_done"] | {move[1]}
                return ("t", move[1])
            ep["active"] = move[1]                       # 'enter' — fall through

        # In-cluster → exact one-step lookahead over the JOINT micro value function
        # (the unconditioned per-cluster belief-MDP; handles the mixed-k live belief
        # exactly, no occupancy mixture).
        ci = ep["active"]
        c = sense[ci]
        bm = built["bit"][c.name]
        col_local = frozenset(t for t in c.tres if t in collected)
        sup = self._local_support(bw, c, bm)
        sol = built["joint"][c.name]
        # intersect the live support with the micro's representable latents (the live
        # support may be CONDITIONED by other clusters' revealed occupancy via the
        # global Σ=K coupling — a subset of the marginal latents).
        sup = frozenset(s for s in sup if s in sol.wmap)
        if not sup:
            action = LEAVE
        else:
            # The memoised exact solver returns π* for ANY support — including the
            # globally-conditioned ones — measuring travel from the live loc.  A
            # conditioned support that the marginal build never enumerated (Σ=K
            # coupling) is solved on demand and cached (bounded by the reachable-
            # belief count).  Count those as on-demand solves for the eval report.
            if (loc, sup, col_local) not in sol.value:
                self.fallbacks += 1
            _, action = sol.solve(loc, sup, col_local)

        if action == LEAVE:
            ep["visited"].add(ci)
            ep["active"] = None
            return self.decide(env, loc, bw, collected, lam, rng)
        return action


def _live_occupancy_posterior(env, bw, macro):
    """[exact] The occupancy-vector posterior given the live belief `bw`: project
    each surviving world onto the macro cells (sense clusters + the δ pool), count
    the resulting occupancy vectors, normalise.  Exact — it reflects every reveal
    encoded in `bw` (cluster chains, δ collects) with no incremental bookkeeping,
    and it reduces to the prior `∏ C(size,k)` when `bw` is the full world set.

    Vectorised: per cell, popcount the masked worlds (sum of extracted bits) and
    pack the per-cell occupancies into one signature integer, then group-count."""
    import numpy as np
    cells = [c.tres for c in macro.sense] + [tuple(macro.delta)]
    n = len(bw)
    occ = np.zeros((len(cells), n), dtype=np.int64)
    for ci, tres in enumerate(cells):
        acc = np.zeros(n, dtype=np.int64)
        for t in tres:
            acc += (bw >> t) & 1
        occ[ci] = acc
    # pack occupancies (each ≤ K=5, fits in 3 bits) into a signature
    sig = np.zeros(n, dtype=np.int64)
    for ci in range(len(cells)):
        sig = (sig << 3) | occ[ci]
    vals, counts = np.unique(sig, return_counts=True)
    out, tot = {}, int(counts.sum())
    for s, c in zip(vals.tolist(), counts.tolist()):
        vec = []
        s = int(s)
        for _ in range(len(cells)):
            vec.append(s & 0b111)
            s >>= 3
        out[tuple(reversed(vec))] = c / tot
    return out
