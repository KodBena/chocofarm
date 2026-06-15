#!/usr/bin/env python3
"""
chocofarm environment — the SIMULATION MODEL, decoupled from any solver.

Owns: the instance (treasures, arrangement-face sense actions, teleports, travel, values), the exact
belief mechanics (numpy world-set + filtering), the dynamics (legal actions, apply), and the
unbiased simulation/evaluation (simulate one episode; Monte-Carlo rate; Dinkelbach fixed
point). It knows nothing about HOW a decision is made — that is a `Policy` (see
chocofarm/solvers/base.py), passed in. New solution methods (NMCS, ISMCTS, …) are new
Policy subclasses; this file does not change.
"""
import copy
import math
import numpy as np

from chocofarm.model import arrangement as A
from chocofarm.model import facemodel
from chocofarm.model.instance import load_instance, world_array

TERMINATE = ("term", None)


class Environment:
    def __init__(self, instance_path=None, value=None, teleport_overhead=12.0, entry="CSNE"):
        inst = load_instance(instance_path)
        self.treasures = inst.treasures
        self.teleports = inst.teleports
        self.N, self.K = inst.N, inst.K
        # COPY-ON-WRITE CONTRACT (with_scenario): the scenario knobs `value`/`entry`/`tp`
        # are the ONLY construction state that may depend on the scenario. Nothing below may
        # cache a structure DERIVED from value/entry/tp (e.g. a value-weighted precompute) —
        # `with_scenario` shallow-copies the env and overrides only these three, so any such
        # derived cache would be shared stale across scenarios (silent divergence). Keep
        # value/entry/tp-derived quantities computed at point-of-use (apply/simulate/exit_cost).
        self.value = list(value) if value is not None else [1.0] * self.N
        self.entry, self.tp = entry, float(teleport_overhead)

        # The single episode-horizon home: a named safety-net cap on episode/rollout length
        # (episodes normally terminate on TERMINATE well before it). Every episode loop that
        # ran a bare `40` (simulate, _base_value, info-relaxation, generate_episode) references
        # this one attribute so the horizon has exactly one source of truth.
        self.max_steps = 40

        # detectors: arrangement faces (docs/consults/consult-002-detector-misspec-report.md §(4)
        # "The correct model and remedy"). A sense action is "stand at face F's representative
        # point and read the disjunction over F's cover" — cover and position are consistent BY
        # CONSTRUCTION (the face is the single carrier of both). This replaces the old
        # `cover_mask[i] = {i} ∪ overlap-neighbours`, which read the union over every face in Δ_i
        # (a k=5 semantics) at one face's rep-point (a k≤2 position) — an over-approximation that
        # handed out information no real sensor could.
        #
        # SINGLE FACE-CARRIER (audit item E). `facemodel.SenseAction` is now the env's ONE carrier
        # of a face's position+cover+observe/filter/informative — THE ENV no longer reimplements
        # those four methods inline beside a dead copy. `filter_detector`/`legal_actions`/`apply`
        # below all DELEGATE to the SenseAction (see those methods). The ('d', id) action shape is
        # UNCHANGED — only the underlying carrier is now the SenseAction object, not three loose
        # dicts kept in sync by nobody.
        #   SCOPE: this ends the env's OWN dead-vs-live duplication (the item-E target). `det_pt` /
        #   `cover_mask` remain public, so a handful of EXTERNAL readers (solvers/, bounds/, az/)
        #   still index `cover_mask[i]` and re-derive the disjunction inline; routing those onto
        #   `senses[i]` too is a larger cross-file follow-up beyond this byte-identity-bounded item,
        #   not regressed by it (those dicts are now DERIVED from the senses, so they cannot drift).
        #
        # GEOMETRIC DERIVABILITY (maintainer's binding constraint — preserved). A face is a DERIVED
        # object, never a frozen opaque table: it is the intersection-refinement of the atomic
        # detectors, computed from the geometric data and reproducible end-to-end via the pipeline
        #   scripts/chocobo_geometry.py   (parse chocobo.ggb -> the atomic detector regions Δ_j,
        #                                  data/instance.json's regions_wkt)
        #   arrangement.arrangement(...)  (polygonize(unary_union({∂Δ_j})) -> the atomic faces;
        #                                  each face's cover = {j : Δ_j ⊇ face}, read at an interior
        #                                  rep-point — a REFINEMENT of the Δ_j, not a change:
        #                                  per region, ⋃(covers of its faces) == the old cover_mask)
        #     -> arrangement.persist()  ->  data/faces.json  ->  arrangement.load()  (loaded below)
        #   scripts/{build_faces_ggb,verify_faces}.py  (the visual round-trip + the self-check that
        #                                  the per-region union equality holds; see
        #                                  docs/design/face-model-verification.md).
        # `sense_actions(faces)` merely WRAPS that derived face — it freezes nothing the geometry
        # does not already determine; re-running the pipeline regenerates faces.json and the senses
        # follow.
        faces = A.load()                                          # 44 atomic arrangement faces (derived; see above)
        self.senses = facemodel.sense_actions(faces)             # the single face-carrier, sense[k] wraps faces[k]
        self.detectors = list(range(len(self.senses)))           # face ids 0..43 are the sense actions
        # det_pt / cover_mask are now SERVED FROM the senses (the face is the one carrier): the
        # bitmask/rep_point come from the same SenseAction `filter_detector`/`apply` read below, so
        # the position-vs-semantics inconsistency consult-002 caught cannot recur. They stay public
        # attributes because external readers (bounds/, solvers/, az/) index them directly.
        self.det_pt = {k: self.senses[k].rep_point for k in self.detectors}
        self.cover_mask = {k: self.senses[k].bitmask for k in self.detectors}

        self.coord = {}
        for i, xy in self.treasures.items():
            self.coord[("t", i)] = xy
        for i, xy in self.det_pt.items():
            self.coord[("d", i)] = xy
        for k, xy in self.teleports.items():
            self.coord[("w", k)] = xy

        self.worlds = world_array(self.N, self.K)

        # The treasure ids `legal_actions` iterates as candidate ('t', i) pickups. For the
        # FULL env this is `range(self.N)` (every treasure); `restrict` overrides it with the
        # kept-subset for a sub-instance (a perf specialization — restricted worlds never set
        # non-keep bits, so marg[i]=0 there and range(N)+(marg>0) already excludes them).
        self._treasure_ids = range(self.N)

        # Precomputed inter-node distance table (perf). The coordinate set is STATIC for an
        # instance, so `d(a, b)` is a static function of the two coord keys; recomputing
        # `math.hypot` per call cost ~2M hypot calls per generated episode (the hot-path
        # profile). The table is built from the SAME `math.hypot(x1-x2, y1-y2)` inputs, so it is
        # bit-identical to the on-the-fly computation — a structural memoization, not an
        # approximation. 67 coord keys -> ~4.5k entries, a few ms to build, negligible memory.
        self._dist = {}
        coord_items = list(self.coord.items())
        for ka, (x1, y1) in coord_items:
            for kb, (x2, y2) in coord_items:
                self._dist[(ka, kb)] = math.hypot(x1 - x2, y1 - y2)

    # ---- public read of the legal-action treasure-id hook ----
    @property
    def keep(self):
        """The treasure ids this env proposes as ('t', i) collects: the kept subset for a
        restricted sub-instance (Environment.restrict), or all N for a full env. Public read
        of the legal-action treasure-id hook (_treasure_ids).

        For a full env this is tuple(range(N)) = every treasure; for a restricted env it is the
        sorted `keep` tuple `restrict` already stored. `_treasure_ids` stays the internal hook
        `legal_actions` iterates — this is the read-only public name for cross-module readers
        (bounds/eval_bound.py) so the private name is not reached across the module boundary.
        """
        return tuple(self._treasure_ids)

    # ---- scenario (copy-on-write) ----
    def with_scenario(self, scenario):
        """Return a NEW Environment that SHARES this env's immutable Tier-1 geometry
        by reference (copy-on-write) and overrides only the Tier-2 scenario knobs
        `value`/`entry`/`tp` from `scenario`.

        The expensive geometry — `_dist` (the ~4.5k-entry distance table), `coord`,
        `worlds`, `senses` (the face-carriers), `detectors`, `det_pt`, `cover_mask`,
        `treasures`, `teleports`, `N`, `K` — depends ONLY on the instance, NOT on the
        scenario knobs (`value` is read only in `apply`, `entry` only in `simulate`,
        `tp` only in `exit_cost`). So a `copy.copy(self)` (shallow — those attributes are aliased
        to the original, NOT rebuilt) plus the three overrides is exactly equivalent
        to a fresh `Environment(value=…, entry=…, teleport_overhead=…)`.

        A value/entry/teleport sweep is therefore
        `[env.with_scenario(s) for s in scenarios]` — one geometry build, N shallow
        copies — not N full rebuilds. `self` is NOT mutated.
        """
        if scenario.value is not None and len(scenario.value) != self.N:
            # ADR-0002 fail-loud: a wrong-length value vector is a config error, not
            # something to silently broadcast or truncate to N.
            raise ValueError(
                f"Scenario.value has length {len(scenario.value)}, "
                f"expected N={self.N} (one reward per treasure)."
            )
        new = copy.copy(self)
        new.value = list(scenario.value) if scenario.value is not None else [1.0] * new.N
        new.entry = scenario.entry
        new.tp = float(scenario.teleport_overhead)
        return new

    # ---- restriction (copy-on-write) ----
    def restrict(self, keep, k_local):
        """Return a restriction VIEW of this env: a genuinely small sub-instance over a
        SUBSET of treasures `keep` with a reduced present-count `k_local`, sharing this
        env's geometry AND belief mechanics by reference (copy-on-write) and overriding
        ONLY the world-set, the detector subset, K, and the treasure-id hook.

        Because the view reuses the parent's `marginals`/`filter_treasure`/`filter_detector`/
        `legal_actions`/`apply`, the information-relaxation dual bound certifies against the
        EXACT same dynamics the learner uses — there is one belief-mechanics implementation,
        not a copy (audit R8). This is DUAL-BOUND-CRITICAL: a silent divergence between the
        bound's inner solve and the real env would corrupt the trusted check with no test
        failure, so this MUST reproduce the dynamics byte-identically.

        Restriction:
          * treasures restricted to `keep` (a sorted tuple of ORIGINAL treasure ids); bit
            positions stay the original ids, so `cover_mask`/`d`/`value`/presence bits line
            up with the parent unchanged;
          * K = `k_local` present among them (worlds = C(|keep|, k_local) bitmasks over the
            ORIGINAL bit positions);
          * `_treasure_ids` = `keep`, so `legal_actions` only proposes kept ('t', i) — a perf
            specialization (restricted worlds never set non-keep bits, so range(N)+(marg>0)
            would already exclude them; iterating `keep` just skips the dead candidates);
          * detectors restricted to faces whose cover is non-empty and ⊆ keep.

        The shared geometry (`_dist`/`coord`/`value`/`entry`/`tp`/`treasures`/`teleports`/`N`)
        is aliased by a `copy.copy` (copy-on-write), NOT rebuilt; `self` is NOT mutated.
        """
        keep = tuple(sorted(keep))
        keepset = set(keep)
        # ADR-0002 fail-loud: an empty / over-restricted / out-of-range keep is a config
        # error, not something to silently clamp into a degenerate sub-instance.
        if not keep:
            raise ValueError("restrict: keep is empty (need at least one treasure id).")
        if any(t < 0 or t >= self.N for t in keep):
            raise ValueError(
                f"restrict: keep={keep} has ids outside [0, N={self.N}).")
        if k_local > len(keep):
            raise ValueError(
                f"restrict: k_local={k_local} exceeds |keep|={len(keep)} "
                f"(cannot have more present than kept).")
        new = copy.copy(self)
        new.K = int(k_local)
        new._treasure_ids = keep
        # worlds: k_local-of-keep present-sets, as bitmasks over ORIGINAL bit positions
        new.worlds = world_array(new.N, new.K, support=keep)
        # detectors: faces whose cover is non-empty and ⊆ keep (rebuilt EXACTLY as the old
        # bounds/minienv.py MiniEnv.__init__ did, folded in here by audit R8 — same filter,
        # same iteration order, same det_pt/cover_mask values, so the bound is unchanged).
        # `new.senses` stays the parent's full list by alias (copy.copy): a SenseAction is a
        # derived, immutable face-carrier keyed by face id, and `new.detectors` only ever holds
        # kept face ids, so `new.senses[fid]` is always the right (and identical) carrier — the
        # restricted dynamics delegate to the SAME face objects as the parent (audit item E + R8).
        new.detectors = []
        new.cover_mask = {}
        new.det_pt = {}
        for fid in self.detectors:
            cm = self.cover_mask[fid]
            cover = [t for t in range(self.N) if (cm >> t) & 1]
            if cover and set(cover) <= keepset:
                new.detectors.append(fid)
                new.cover_mask[fid] = cm
                new.det_pt[fid] = self.det_pt[fid]
        return new

    # ---- geometry ----
    def d(self, a, b):
        """Distance between two coord keys. Served from the precomputed static table built at
        construction (same `math.hypot` inputs -> bit-identical); falls back to a live compute
        for any key pair not in the table (none arise in normal use, but keeps the contract
        total)."""
        v = self._dist.get((a, b))
        if v is not None:
            return v
        (x1, y1), (x2, y2) = self.coord[a], self.coord[b]
        return math.hypot(x1 - x2, y1 - y2)

    def exit_cost(self, loc):
        return min(self.d(loc, ("w", k)) for k in self.teleports) + self.tp

    def nearest_exit(self, loc):
        return min(self.teleports, key=lambda k: self.d(loc, ("w", k)))

    def route_time(self, start, seq):
        if not seq:
            return self.exit_cost(start)
        t = self.d(start, ("t", seq[0]))
        for a, b in zip(seq, seq[1:]):
            t += self.d(("t", a), ("t", b))
        return t + self.exit_cost(("t", seq[-1]))

    # ---- belief ----
    def marginals(self, bw):
        if len(bw) == 0:
            return np.zeros(self.N)
        return ((bw[:, None] >> np.arange(self.N)) & 1).mean(0)

    def filter_treasure(self, bw, i, present):
        bit = (bw >> i) & 1
        return bw[bit == (1 if present else 0)]

    def filter_detector(self, bw, i, pos):
        # Delegates to the face's single carrier (audit item E): SenseAction.filter is the same
        # `bw[(bw & bitmask)!=0]` disjunction, now owned in one place rather than re-inlined here.
        return self.senses[i].filter(bw, pos)

    def sample_world(self, bw, rng):
        return int(rng.choice(bw))

    # ---- dynamics ----
    def legal_actions(self, loc, bw, collected):
        marg = self.marginals(bw)
        acts = [("t", i) for i in self._treasure_ids if i not in collected and marg[i] > 0]
        for i in self.detectors:
            if self.senses[i].informative(bw):     # outcome still uncertain (both polarities live)
                acts.append(("d", i))              # delegated to the face's single carrier (item E)
        return acts

    def apply(self, loc, bw, collected, action, world):
        """Realise `action` against the true `world`. Returns (reward, loc', bw', collected', dt)."""
        kind, i = action
        dt = self.d(loc, (kind, i))
        if kind == "t":
            pres = bool((world >> i) & 1)
            r = self.value[i] if (pres and i not in collected) else 0.0
            nc = collected | {i} if pres else collected
            return r, (kind, i), self.filter_treasure(bw, i, pres), nc, dt
        pos = self.senses[i].observe(world)        # the face's reading at this world (item E carrier)
        return 0.0, (kind, i), self.filter_detector(bw, i, pos), collected, dt

    # ---- simulation / evaluation (solver-agnostic) ----
    def simulate(self, policy, world, lam, rng, max_steps=None):
        if max_steps is None:
            max_steps = self.max_steps         # the single episode-horizon home (see __init__)
        loc, bw, collected, R, T = ("w", self.entry), self.worlds, set(), 0.0, 0.0
        for _ in range(max_steps):
            a = policy.decide(self, loc, bw, collected, lam, rng)
            if a == TERMINATE:
                break
            r, loc, bw, collected, dt = self.apply(loc, bw, collected, a, world)
            R += r; T += dt
        return R, T + self.exit_cost(loc), self.nearest_exit(loc)

    def rate(self, policy, lam, runs, seed):
        rng = np.random.default_rng(seed)
        totR = totT = 0.0
        exits = {}
        for _ in range(runs):
            w = int(rng.choice(self.worlds))
            R, T, e = self.simulate(policy, w, lam, rng)
            totR += R; totT += T
            exits[e] = exits.get(e, 0) + 1
        return totR / totT, totR / runs, totT / runs, exits

    def dinkelbach_rate(self, policy, iters=4, warm_runs=600, final_runs=3000, seed=7, lam0=0.0):
        """A policy's own long-run rate = its Dinkelbach fixed point (lambda <- achieved rate)."""
        lam = lam0
        for _ in range(iters):
            lam = self.rate(policy, lam, warm_runs, seed=1)[0]
        rate, ER, ET, exits = self.rate(policy, lam, final_runs, seed)
        return {"lambda": lam, "rate": rate, "ER": ER, "ET": ET, "exits": exits}
