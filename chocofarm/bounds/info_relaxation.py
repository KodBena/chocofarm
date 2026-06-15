#!/usr/bin/env python3
"""
info_relaxation.py — a PROVABLE upper bound on the optimal long-run rate ρ* via the
Brown–Smith–Sun (2010) information-relaxation dual, sharpening the loose clairvoyant
ceiling 0.1454 (docs/results/voi-ceiling-2026-06-13.md).

Design + proofs: docs/design/dual-bound.md. The one-paragraph recap:

  * The fixed-λ problem g(λ) = sup_π E[ΣR − λΣT] is a finite-horizon belief-MDP.
    Dinkelbach: g is strictly decreasing with unique zero g(ρ*)=0 (every run costs
    time ≥ exit toll > 0).
  * PERFECT-INFORMATION RELAXATION + a dual-feasible PENALTY z built from an
    approximate value function V̂ gives the BSS weak-duality bound
        g(λ) ≤ B(λ, z) := E_w[ sup_a ( r(a,w) − z(a,w) ) ]          (★)
    where the sup is the EXACT per-world inner optimization (a deterministic
    sequencing DP in the fully-revealed world).
  * z is a sum of martingale differences w_t − E[w_t | F_t], w_t = r_t + V̂_{t+1},
    so E[z | F-adapted] = 0 (DUAL FEASIBILITY) for ANY V̂ — only TIGHTNESS depends
    on V̂'s quality, never VALIDITY.
  * Telescoping (the BSS penalized-reward identity, proven in dual-bound.md §4):
        Σ_t (r_t − z_t) = Σ_t [ E[r_t + V̂_{t+1} | F_t, a_t] − V̂_{t+1}(x_{t+1}) ]
    so the inner objective is a path sum of PENALIZED per-step rewards
        ρ̃(x_t, a_t, w) = E[r_t + V̂_{t+1} | F_t, a_t] − V̂_{t+1}(x_{t+1}),
    where the conditional expectation averages V̂ over the action's observation
    outcome under the belief b_t, and V̂_{t+1}(x_{t+1}) is at the REALIZED successor
    in world w.
  * The Dinkelbach composition: λ̄ = root of B(·, z) satisfies ρ* ≤ λ̄ (proof
    dual-bound.md §3, monotonicity direction pinned by time > 0).

THE KEY CORRECTNESS HAZARD (dual-bound.md §4): (★) is an upper bound ONLY if the
inner sup is an exact supremum or an OVER-estimate. A lower bound on the inner max
(single-path heuristic / truncated search) silently breaks the bound. This module's
inner solver is an EXACT memoized deterministic DP over a provable SUPERSET of useful
actions; it ABORTS LOUDLY on the state cap rather than truncating (never returns a
lower bound silently).

z ≡ 0 reproduces the existing clairvoyant ceiling 0.1454 — the regression check
(dual-bound.md §4.2): with z=0 in a known world, face reads and absent-treasure
visits are strictly dominated, so the inner sup collapses to clairvoyant_rate's
subset×permutation enumeration.

Pin any run to CPU core 3 under `timeout` (a live AZ job holds cores 0–3). The full
15,504-world computation is DEFERRED — validate on small sub-instances only.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from chocofarm.model.env import Environment, TERMINATE


# ===========================================================================
# Approximate value functions V̂(loc, belief, collected) at a fixed λ
# ===========================================================================
#
# A V̂ estimates the fixed-λ value-to-go E[ΣR − λΣT | state] of near-optimal
# continuation. ANY V̂ yields a VALID bound (dual feasibility is automatic); a good V̂
# yields a TIGHT one. The penalty / inner solve treat V̂ as an injected callable
# `vhat(env, loc, bw, collected, lam) -> float`.


def vhat_zero(env, loc, bw, collected, lam):
    """V̂ ≡ 0 — but NOTE this is NOT the z≡0 clairvoyant baseline. With V̂≡0 the
    value-function penalty is z_t = r_t − E[r_t | F_t, a_t] (the REWARD-DEVIATION
    martingale), which is dual-feasible and nonzero. It is a (mild) valid penalty, not
    the pure relaxation. The TRUE z≡0 regression baseline is `vhat=None` (the
    no-penalty mode in PenalizedClairvoyant), which uses the realized r − λ·dt and
    reproduces clairvoyant_rate exactly. Kept only as a curiosity / extra valid V̂."""
    return 0.0


def vhat_analytic(env, loc, bw, collected, lam):
    """Trivial analytic V̂₀ (sanity baseline, dual-bound.md §2.4(1)): expected
    still-collectable reward if grabbable for free, minus the cost to leave.

        V̂₀ = Σ_i marginals(b)[i]·value[i]·1[i∉c]  −  λ·exit_cost(loc)

    Crude but a genuine value estimate; it makes the penalty CHARGE for resolving
    marginals, so B(λ, V̂₀) is a valid bound that should sit modestly below 0.1454."""
    if len(bw) == 0:
        return -lam * env.exit_cost(loc)
    marg = env.marginals(bw)
    er = sum(marg[i] * env.value[i] for i in range(env.N) if i not in collected)
    return er - lam * env.exit_cost(loc)


class DecompVhat:
    """Decomp belief value function V̂_D (dual-bound.md §2.4(2)): the macro's λ-value
    of the live state, reusing chocofarm.solvers.decomp's exact per-cluster
    continuation values + the live occupancy posterior. This is the SAME object the
    decomp policy acts on (the 0.094-achievable belief value), reused as the penalty's
    value approximation — the strongest TRUSTED V̂ here.

    V̂_D(loc, bw, collected) = MacroPlanner.value(loc, live_posterior, visited∅,
    delta_done(from collected), horizon)[0], i.e. the expectimax λ-value of the
    macro state, which already includes the exit toll. Built lazily per λ and cached.

    Note: this is a DECISION value function (it steers the decomp policy), reused as a
    state-value estimate. It is accurate but sub-optimal, so the resulting bound is
    tight-ish, not exact (dual-bound.md §6)."""

    def __init__(self, horizon=1):
        self.horizon = horizon
        self._built = {}   # round(lam,6) -> (macro, sense, delta_ids)

    def _build(self, env, lam):
        key = round(lam, 6)
        if key in self._built:
            return self._built[key]
        # import here to keep the bounds module importable without decomp on hand
        from chocofarm.solvers import decomp as D
        clusters = D.discover_clusters(env)
        sense = [c for c in clusters if c.size > 1]
        anchors = {c.name: min(c.tres, key=lambda t: env.d(("w", env.entry), ("t", t)))
                   for c in sense}
        micro = {}
        for c in sense:
            entry = ("t", anchors[c.name])
            for k in range(1, c.size + 1):
                micro[(c.name, k)] = D.build_cluster_micro(env, c, k, lam, entry)
        macro = D.MacroPlanner(env, clusters, micro, lam, horizon=self.horizon)
        delta_ids = [c.tres[0] for c in clusters if c.size == 1]
        built = (macro, sense, delta_ids, D)
        self._built[key] = built
        return built

    def __call__(self, env, loc, bw, collected, lam):
        if len(bw) == 0:
            return -lam * env.exit_cost(loc)
        macro, sense, delta_ids, D = self._build(env, lam)
        post = D._live_occupancy_posterior(env, bw, macro)
        # visited: clusters already fully collected (all members collected) count as
        # visited so the macro does not re-enter them; conservative — an unvisited but
        # partly-collected cluster is left enterable (the macro re-values it).
        visited = set()
        for ci, c in enumerate(sense):
            if all(t in collected for t in c.tres):
                visited.add(ci)
        delta_done = frozenset(t for t in delta_ids if t in collected)
        v, _ = macro.value(loc, post, visited, delta_done, self.horizon)
        # macro.value returns the λ-value of CONTINUING (it includes the exit toll on
        # its 'exit' leaf). Add the already-collected reward? No: V̂ is value-TO-GO,
        # the continuation value from this state, which is exactly what macro.value
        # returns. Reward already banked is not part of value-to-go.
        return v


class ExactBeliefVhat:
    """The EXACT optimal value-to-go V*(loc, belief, collected) of the (small) belief-
    MDP at a fixed λ, by backward induction over the belief semilattice. Tractable ONLY
    on small sub-instances (`env.restrict(keep, k_local)`) — it enumerates reachable
    beliefs, which is the full 15,504-world intractability on the real env (do NOT use on
    the full env).

    Its purpose is the DEFINITIVE tightening test (dual-bound.md §2.3 / §6): BSS
    strong duality (Thm 3.4) says V̂ = V* makes the penalty OPTIMAL and the bound TIGHT
    — λ̄ = ρ*_subinstance exactly. So on a restricted sub-instance this V̂ should drive the dual bound
    down to the sub-instance's achievable optimum, well below its clairvoyant value —
    a direct demonstration that the machinery TIGHTENS when handed a good V̂ (the
    decomp / analytic V̂ are merely weaker approximations, not a failure of the
    construction)."""

    def __init__(self):
        self._memo = {}     # (lam, loc, belief, collected) -> V*

    def __call__(self, env, loc, bw, collected, lam):
        return self._solve(env, lam, loc,
                           tuple(int(x) for x in bw), frozenset(collected))

    def _solve(self, env, lam, loc, bw_key, collected):
        key = (round(lam, 9), loc, bw_key, collected)
        if key in self._memo:
            return self._memo[key]
        bw = np.array(bw_key, dtype=np.int64)
        if len(bw) == 0:
            self._memo[key] = -lam * env.exit_cost(loc)
            return self._memo[key]
        best = -lam * env.exit_cost(loc)                   # TERMINATE
        marg = env.marginals(bw)
        # collect a possibly-present uncollected treasure
        for i in range(env.N):
            if i in collected or marg[i] <= 0:
                continue
            dt = env.d(loc, ("t", i))
            q = float(marg[i])
            pres_b = env.filter_treasure(bw, i, True)
            abs_b = env.filter_treasure(bw, i, False)
            vp = env.value[i] + self._solve(env, lam, ("t", i),
                                            tuple(int(x) for x in pres_b),
                                            collected | {i}) if len(pres_b) else 0.0
            va = self._solve(env, lam, ("t", i), tuple(int(x) for x in abs_b),
                             collected) if len(abs_b) else 0.0
            q_val = -lam * dt + q * vp + (1.0 - q) * va
            best = max(best, q_val)
        # read an informative face
        for j in env.detectors:
            cm = env.cover_mask[j]
            hit = (bw & cm) != 0
            if not (hit.any() and (~hit).any()):
                continue
            dt = env.d(loc, ("d", j))
            p = float(hit.mean())
            vpos = self._solve(env, lam, ("d", j),
                               tuple(int(x) for x in bw[hit]), collected)
            vneg = self._solve(env, lam, ("d", j),
                               tuple(int(x) for x in bw[~hit]), collected)
            q_val = -lam * dt + p * vpos + (1.0 - p) * vneg
            best = max(best, q_val)
        self._memo[key] = best
        return best


# ===========================================================================
# The penalized inner per-world optimization — EXACT memoized deterministic DP
# ===========================================================================


class PenalizedClairvoyant:
    """Computes B(λ, z) = E_w[ sup_a (r(a,w) − z(a,w)) ], the penalized clairvoyant
    value at a fixed λ, by EXACT per-world inner optimization.

    The inner solve (dual-bound.md §4): in a fully-revealed world w every observation
    is determined, so the inner problem is a DETERMINISTIC sequencing DP. The state is
    (loc, collected, belief) — belief is a deterministic function of the action prefix
    given w. The penalized per-step reward (the telescoped BSS form) is

        ρ̃(x, a, w) = E[r + V̂' | F, a] − V̂'(x')   at the realized successor x' in w.

    V̂ is fixed at a REFERENCE λ* (Route A, dual-bound.md §3) so B(λ) is exactly
    strictly decreasing in the scanned λ and its root is well-defined; pass
    `vhat_lam=λ*` to fix it, or leave None to rebuild V̂ at the scanned λ (Route B —
    still valid, monotonicity not guaranteed).

    EXACTNESS: the DP enumerates the full legal action set (collects of
    possibly-present treasures, INFORMATIVE faces, TERMINATE) at every state and takes
    the true max — so it is an exact sup over the admitted action set. The admitted set
    is a SUPERSET of every action that can appear in the true optimum (a face/treasure
    that is uninformative/certainly-absent cannot improve any path), so the enumeration
    is exact. `max_inner_states` ABORTS LOUDLY (never truncates) — a truncated search
    could miss the argmax and return a LOWER bound, silently breaking (★)."""

    def __init__(self, env: Environment, vhat=None, vhat_lam=None,
                 max_inner_states=2_000_000, restrict_faces=True):
        self.env = env
        self.vhat = vhat
        self.vhat_lam = vhat_lam          # fix V̂ at this λ (Route A); None → rebuild
        self.max_inner_states = max_inner_states
        self.restrict_faces = restrict_faces

    # ---- penalized per-step reward (telescoped BSS form) ----
    def _vh(self, loc, bw, collected, lam):
        """V̂ at the reference λ (Route A) — or 0 in the NO-PENALTY mode (vhat=None),
        which makes the inner objective the pure realized r − λ·dt (the z≡0 clairvoyant
        regression baseline)."""
        if self.vhat is None:
            return 0.0
        vl = self.vhat_lam if self.vhat_lam is not None else lam
        return self.vhat(self.env, loc, bw, collected, vl)

    def _penalized_step(self, loc, bw, collected, action, world, lam):
        """The realized step contribution `r_t − z_t` of the inner objective, plus the
        realized (r, x') so the caller can recurse. Returns (contribution, r, loc',
        bw', collected', dt).

        z_t = (r_t + V̂_{t+1}(x_{t+1})) − E[r_t + V̂_{t+1} | F_t, a_t] is the BSS
        value-function martingale-difference penalty, so

            r_t − z_t = E[r_t + V̂_{t+1} | F_t, a_t] − V̂_{t+1}(x_{t+1})

        (the telescoped form). In the NO-PENALTY mode (vhat=None) every V̂ term is 0
        and z_t = r_t − E[r_t | F_t, a_t]; we then return the PURE realized r_t
        (= r − λ·dt) instead, so vhat=None reproduces the clairvoyant inner solve
        EXACTLY (dual-bound.md §4.2 regression). With a real V̂ we return the telescoped
        r_t − z_t."""
        env = self.env
        kind, i = action
        dt = env.d(loc, (kind, i))
        no_penalty = (self.vhat is None)
        if kind == "t":
            # conditional expectation over presence under the belief
            q = float(env.marginals(bw)[i]) if len(bw) else 0.0
            pres_b = env.filter_treasure(bw, i, True)
            abs_b = env.filter_treasure(bw, i, False)
            r_pres = env.value[i] if i not in collected else 0.0
            c_pres = collected | {i}
            # realized
            pres = bool((world >> i) & 1)
            r = env.value[i] if (pres and i not in collected) else 0.0
            nbw = pres_b if pres else abs_b
            nc = c_pres if pres else collected
            nloc = ("t", i)
            if no_penalty:
                return (r - lam * dt), r, nloc, nbw, nc, dt
            vh_pres = self._vh(("t", i), pres_b, c_pres, lam) if len(pres_b) else 0.0
            vh_abs = self._vh(("t", i), abs_b, collected, lam) if len(abs_b) else 0.0
            # E[r + V̂' | belief, a]  (r's travel part −λ·dt is deterministic; pull out)
            exp_rv = -lam * dt + q * (r_pres + vh_pres) + (1.0 - q) * (0.0 + vh_abs)
            vh_real = self._vh(nloc, nbw, nc, lam) if len(nbw) else 0.0
            return (exp_rv - vh_real), r, nloc, nbw, nc, dt
        else:  # face read
            cm = env.cover_mask[i]
            hit = (bw & cm) != 0
            pos_b = bw[hit]
            neg_b = bw[~hit]
            pos = bool(world & cm)
            nbw = pos_b if pos else neg_b
            nloc = ("d", i)
            if no_penalty:
                # a face read in a known world: no reward, just −λ·dt (strictly
                # dominated, so it is never chosen — collapsing to clairvoyant)
                return (-lam * dt), 0.0, nloc, nbw, collected, dt
            p_pos = float(hit.mean()) if len(bw) else 0.0
            vh_pos = self._vh(("d", i), pos_b, collected, lam) if len(pos_b) else 0.0
            vh_neg = self._vh(("d", i), neg_b, collected, lam) if len(neg_b) else 0.0
            exp_rv = -lam * dt + p_pos * vh_pos + (1.0 - p_pos) * vh_neg
            vh_real = self._vh(nloc, nbw, collected, lam) if len(nbw) else 0.0
            return (exp_rv - vh_real), 0.0, nloc, nbw, collected, dt

    # ---- exact inner DP for one world ----
    def inner_value(self, world, lam):
        """sup_a Σ_t ρ̃(x_t, a_t, w) over finite action sequences ending in TERMINATE,
        in the fully-revealed `world`. EXACT memoized DP. Returns (value, R, T): the
        penalized inner objective AND the realized (reward, time) along the OPTIMIZING
        path, so B's Dinkelbach root can be found from ΣR/ΣT exactly as
        clairvoyant_rate does (the realized R, T of the inner-optimal path).

        Termination at a state contributes the penalized exit step:
            ρ̃_exit = E[−λ·exit_cost + V̂(after-exit≡terminal 0)] − 0  = −λ·exit_cost,
        i.e. TERMINATE is deterministic, penalty increment 0, value −λ·exit_cost."""
        env = self.env
        state_count = [0]

        @lru_cache(maxsize=None)
        def solve(loc, bw_key, collected):
            state_count[0] += 1
            if state_count[0] > self.max_inner_states:
                raise RuntimeError(
                    f"inner DP state cap {self.max_inner_states} exceeded for world "
                    f"{world:#x} — REFUSING to truncate (a truncated inner search can "
                    f"return a LOWER bound and silently break the upper-bound property; "
                    f"dual-bound.md §4). Widen the cap or use the documented "
                    f"over-approximation, never a partial search.")
            bw = np.array(bw_key, dtype=np.int64)
            collected_set = set(collected)
            # base option: TERMINATE now
            best_v = -lam * env.exit_cost(loc)
            best_R, best_T = 0.0, env.exit_cost(loc)

            # candidate actions: collects of possibly-present uncollected treasures,
            # and informative faces. This is the legal set (env.legal_actions), a
            # SUPERSET of every action that can appear in the true optimum.
            acts = env.legal_actions(loc, bw, collected_set)
            if self.restrict_faces:
                # faces matter ONLY via their penalty rebate (no reward in a known
                # world); keeping all informative faces is exact. (restrict_faces is a
                # hook for an over-approximating prune; default True keeps all → exact.)
                pass
            for a in acts:
                pen, r, nloc, nbw, nc, dt = self._penalized_step(
                    loc, bw, collected_set, a, world, lam)
                sub_v, sub_R, sub_T = solve(nloc, tuple(int(x) for x in nbw),
                                            frozenset(nc))
                v = pen + sub_v
                if v > best_v + 1e-12:
                    best_v = v
                    best_R = r + sub_R
                    best_T = dt + sub_T
            return best_v, best_R, best_T

        return solve(loc=("w", env.entry),
                     bw_key=tuple(int(x) for x in self._world_belief()),
                     collected=frozenset())

    def _world_belief(self):
        """The initial belief = the full world-set (the sub-instance's worlds)."""
        return self.env.worlds

    # ---- B(λ) = E_w[ inner_value(λ) ] ----
    def B_value(self, lam, worlds, weights=None):
        """B(λ) = mean over `worlds` of the per-world inner sup value. This is the BSS
        penalized-clairvoyant value (★). Also returns the realized (ΣR, ΣT) along the
        inner-optimal paths (for diagnostics / the no-penalty ratio shortcut).
        Returns (B, totR, totT)."""
        totV = totR = totT = 0.0
        wsum = 0.0
        for idx, w in enumerate(worlds):
            wt = 1.0 if weights is None else float(weights[idx])
            v, R, T = self.inner_value(int(w), lam)
            totV += wt * v
            totR += wt * R
            totT += wt * T
            wsum += wt
        return totV / wsum, totR, totT


# ===========================================================================
# Dinkelbach driver — find λ̄ = root of B(·, z)  ⇒  ρ* ≤ λ̄
# ===========================================================================


def dual_bound_rate(pc: PenalizedClairvoyant, worlds, weights=None,
                    lo=0.0, hi=0.30, tol=1e-4, max_iter=40):
    """The dual upper bound λ̄ = root of B(·, z) (dual-bound.md §3), by BISECTION.

    B(λ) is strictly decreasing in λ when V̂ is fixed at vhat_lam (Route A): the inner
    sup of (R − λT − z) over a fixed action/penalty set is a sup of affine-in-λ
    functions with slopes −T < 0 (time > 0 on every run), so B is convex and strictly
    decreasing; its unique root λ̄ satisfies g(λ̄) ≤ B(λ̄) = 0 ⇒ λ̄ ≥ ρ* (weak duality +
    g decreasing). We bracket [lo, hi] with B(lo) > 0 > B(hi) and bisect.

    NB: we find the root of B(λ) DIRECTLY (mean inner value = 0), NOT via a ΣR/ΣT ratio
    fixed point — the ratio shortcut is valid ONLY for the no-penalty case (where
    inner_value = R − λT exactly); with a penalty the inner value carries V̂ terms and
    is not R − λT, so its λ-root is the correct Dinkelbach object and bisection on B is
    the right driver.

    `worlds`: the world-set to average over (sub-instance for validation, full for
    headline). Returns {'lambda': λ̄, 'B_at_root': B(λ̄), 'bracket': (lo,hi), 'log': [...]}.
    """
    log = []

    def B(lam):
        b, totR, totT = pc.B_value(lam, worlds, weights=weights)
        log.append({"lam": lam, "B": b, "totR": totR, "totT": totT})
        return b

    b_lo, b_hi = B(lo), B(hi)
    # widen hi if the bracket does not straddle zero (B(hi) should be < 0)
    tries = 0
    while b_hi > 0 and tries < 8:
        hi *= 1.5
        b_hi = B(hi)
        tries += 1
    if not (b_lo > 0 >= b_hi):
        # degenerate (e.g. B already ≤ 0 at lo): return lo as the conservative root
        return {"lambda": lo if b_lo <= 0 else hi, "B_at_root": min(b_lo, b_hi),
                "bracket": (lo, hi), "log": log, "warning": "bracket did not straddle 0"}
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        bm = B(mid)
        if bm > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    lam_bar = 0.5 * (lo + hi)
    return {"lambda": lam_bar, "B_at_root": B(lam_bar), "bracket": (lo, hi), "log": log}


# ===========================================================================
# Dual-feasibility (mean-zero) empirical check — deliverable §5(iv)
# ===========================================================================


def empirical_penalty_mean(env: Environment, pc: PenalizedClairvoyant, policy, lam,
                           runs=200, seed=11):
    """Direct dual-feasibility check (dual-bound.md §2.3 / §5 deliverable (iv)): run an
    F-adapted `policy`, accumulate the realized per-step penalty increments
        z_t = (r_t + V̂_{t+1}(x_{t+1})) − E[r_t + V̂_{t+1} | F_t, a_t]
    under the belief filtration, and average Σ_t z_t over runs. Should be ≈ 0
    (martingale, mean-zero) within Monte-Carlo error.

    Returns (mean_total_z, stderr). The increment is exactly the NEGATIVE of the
    telescoped penalized reward's deviation:  z_t = (r_t + V̂') − E[r_t + V̂' | F,a].
    We recompute it directly here (not via _penalized_step) to keep the check
    independent of the inner-solve code path."""
    rng = np.random.default_rng(seed)
    totals = []
    vl = pc.vhat_lam if pc.vhat_lam is not None else lam
    for _ in range(runs):
        world = int(rng.choice(env.worlds))
        loc, bw, collected = ("w", env.entry), env.worlds, set()
        zsum = 0.0
        for _step in range(env.max_steps):           # the single episode-horizon home (env.py)
            a = policy.decide(env, loc, bw, collected, lam, rng)
            if a == TERMINATE:
                break
            kind, i = a
            dt = env.d(loc, (kind, i))
            if kind == "t":
                q = float(env.marginals(bw)[i]) if len(bw) else 0.0
                pres_b = env.filter_treasure(bw, i, True)
                abs_b = env.filter_treasure(bw, i, False)
                r_pres = env.value[i] if i not in collected else 0.0
                vh_pres = pc.vhat(env, ("t", i), pres_b, collected | {i}, vl) if len(pres_b) else 0.0
                vh_abs = pc.vhat(env, ("t", i), abs_b, collected, vl) if len(abs_b) else 0.0
                exp_rv = q * (r_pres + vh_pres) + (1.0 - q) * vh_abs   # −λ·dt cancels
                pres = bool((world >> i) & 1)
                nbw = pres_b if pres else abs_b
                nc = (collected | {i}) if pres else collected
                r_real = env.value[i] if (pres and i not in collected) else 0.0
                vh_real = pc.vhat(env, ("t", i), nbw, nc, vl) if len(nbw) else 0.0
                z_t = (r_real + vh_real) - exp_rv
                loc, bw, collected = ("t", i), nbw, nc
            else:
                cm = env.cover_mask[i]
                hit = (bw & cm) != 0
                pos_b, neg_b = bw[hit], bw[~hit]
                p_pos = float(hit.mean()) if len(bw) else 0.0
                vh_pos = pc.vhat(env, ("d", i), pos_b, collected, vl) if len(pos_b) else 0.0
                vh_neg = pc.vhat(env, ("d", i), neg_b, collected, vl) if len(neg_b) else 0.0
                exp_rv = p_pos * vh_pos + (1.0 - p_pos) * vh_neg
                pos = bool(world & cm)
                nbw = pos_b if pos else neg_b
                vh_real = pc.vhat(env, ("d", i), nbw, collected, vl) if len(nbw) else 0.0
                z_t = vh_real - exp_rv
                loc, bw = ("d", i), nbw
            zsum += z_t
        totals.append(zsum)
    arr = np.array(totals)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))
