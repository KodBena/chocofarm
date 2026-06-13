#!/usr/bin/env python3
"""
analyzer.py — the structural decomposition of a chocofarm instance, as a
*program* over the abstract instance, not a one-off hand-analysis.

WHY THIS FILE EXISTS
--------------------
The original hand-analysis (`docs/design/static-shortcuts.md`) was computed on
the broken `cover_mask` over-approximation — the union-over-faces masquerading
as a simultaneous disjunction (consult-002).  Its detector-coupled conclusions
("a detector covers {8,9,10,11,12}", "5.16× single-read collapse", "{8,9}/{11,12}
indistinguishable", "120.7-world floor") are therefore WRONG.  This module
re-derives every structural quantity under the CORRECTED face model
(`arrangement.Face`: each atomic face carries the exact disjunction `cover` read
at its `rep_point`), and does so as a set of small, independently-auditable
functions over an abstract `Instance`, so the same analysis runs on the real map
AND on synthetic ones (`synthetic.py`).

ABSTRACTION CONTRACT
--------------------
`analyze(instance) -> StructuralReport` consumes only the abstract instance:

    Instance = (treasures, faces, teleports, N, K)
        treasures : {id -> (x, y)}                      point per treasure
        faces     : [arrangement.Face]                  each: cover ⊆ ids, rep_point, area
        teleports : {name -> (x, y)}
        N, K      : the exactly-K-of-N prior (here 5 of 20 -> C(20,5)=15504 worlds)

Each structural quantity is one named function with a stated definition.  Every
function is tagged in its docstring:

    [DET-DEP]  — re-derived under the face model; SUPERSEDES the contaminated note.
    [DET-IND]  — a property of the prior/geometry; ports verbatim from the old note.

BOUNDEDNESS
-----------
The only enumeration touched is the C(N,K) world array (15,504 ints — cheap to
build and filter).  No solver is run; no reachable-belief enumeration of the
GLOBAL problem is attempted.  Per-cluster reachable-belief BFS is bounded by the
cluster's 2**size latent subsets and is gated by `max_cluster_bits` so a
pathological synthetic instance cannot blow up.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from math import comb

import numpy as np

import arrangement as A


# ===========================================================================
# The abstract instance
# ===========================================================================

@dataclass(frozen=True)
class Instance:
    """The abstract object `analyze` operates on.  Detector-agnostic in shape:
    faces are the only sensing carrier, exactly as the corrected model intends."""
    treasures: dict          # {id -> (x, y)}
    faces: list              # [arrangement.Face]
    teleports: dict          # {name -> (x, y)}
    N: int
    K: int

    @property
    def ids(self):
        return sorted(self.treasures)

    @property
    def covered(self):
        """Treasure ids that appear in at least one face's cover (sense-reachable)."""
        s = set()
        for f in self.faces:
            s |= set(f.cover)
        return s

    @property
    def delta(self):
        """δ-treasures: sense-isolated (no face covers them) — observe == collect."""
        return set(self.ids) - self.covered


def real_instance(faces_path: str = A.FACES, instance_path: str = A.INSTANCE,
                  N: int = 20, K: int = 5) -> Instance:
    """The frozen real chocofarm map, as an abstract Instance."""
    import json
    data = json.load(open(instance_path))
    treasures = {int(i): tuple(xy) for i, xy in data["treasures"].items()}
    teleports = {k: tuple(v) for k, v in data["teleports"].items()}
    faces = A.load(faces_path)
    return Instance(treasures, faces, teleports, N, K)


# ===========================================================================
# World prior (DET-IND — a property of the exactly-K-of-N prior)
# ===========================================================================

def world_array(inst: Instance) -> np.ndarray:
    """[DET-IND] The C(N,K) equiprobable worlds as a bitmask array (bit t = τ_t
    present).  Detector-independent: it is the prior, ports verbatim."""
    return np.array(
        [sum(1 << t for t in c) for c in itertools.combinations(range(inst.N), inst.K)],
        dtype=np.int64)


def bitmask(cover) -> int:
    return sum(1 << j for j in cover)


# ===========================================================================
# 1. Clusters — connected components of the treasure co-coverage hypergraph
# ===========================================================================

def cocoverage_edges(inst: Instance) -> set:
    """[DET-DEP] {(a,b) : some face's cover contains both a and b}.

    DEFINITION (corrected): treasures are linked iff a single position can read a
    disjunction mentioning both — i.e. some atomic face covers both.  Under the
    OLD model the link was "the regions Δ_a, Δ_b overlap"; that is an existential
    over the pair, whereas this is the genuine arrangement co-occurrence.  (On the
    real map the two edge sets happen to COINCIDE — every 2-D pairwise overlap
    contains a face covering both — so the cluster PARTITION ports unchanged; but
    the definition is now the honest one and on synthetic instances they can
    differ.)"""
    edges = set()
    for f in inst.faces:
        for a, b in itertools.combinations(sorted(f.cover), 2):
            edges.add((a, b))
    return edges


def clusters(inst: Instance) -> list:
    """[DET-DEP] Connected components of the co-coverage hypergraph, as a partition
    of ALL treasures (δ-treasures are isolated singletons).  Sorted by size desc."""
    parent = {t: t for t in inst.ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for a, b in cocoverage_edges(inst):
        union(a, b)
    comp = {}
    for t in inst.ids:
        comp.setdefault(find(t), set()).add(t)
    return sorted((sorted(s) for s in comp.values()), key=lambda s: (-len(s), s))


def cluster_geography(inst: Instance) -> list:
    """[DET-IND] Per-cluster centroid and nearest teleport (Euclidean).  Pure
    coordinates — ports verbatim; the teleport association is geometry, not
    sensor."""
    out = []
    for c in clusters(inst):
        cx = sum(inst.treasures[t][0] for t in c) / len(c)
        cy = sum(inst.treasures[t][1] for t in c) / len(c)
        ranked = sorted(inst.teleports.items(),
                        key=lambda kv: math.hypot(kv[1][0] - cx, kv[1][1] - cy))
        out.append({
            "treasures": c,
            "centroid": (round(cx, 2), round(cy, 2)),
            "nearest_tp": (ranked[0][0], round(math.hypot(ranked[0][1][0] - cx,
                                                          ranked[0][1][1] - cy), 2)),
            "second_tp": (ranked[1][0], round(math.hypot(ranked[1][1][0] - cx,
                                                        ranked[1][1][1] - cy), 2)) if len(ranked) > 1 else None,
        })
    return out


# ===========================================================================
# 2. Belief collapse — honest single-face leverage and face-read chains
# ===========================================================================

def face_collapse(inst: Instance, worlds: np.ndarray = None) -> list:
    """[DET-DEP] For each DISTINCT cover-set realized by some face, the honest
    single-read prune factors.

    DEFINITION: standing on a face with cover C reads (world & bitmask(C)) != 0.
    A positive read keeps the `hit` worlds; a negative keeps the rest.  prune =
    |before| / |after|.  This SUPERSEDES the §2.1 table built on cover_mask: the
    old "5.16× negative collapse for {8,9,10,11,12}" is unrealizable because no
    single face covers all five — the largest single-face negative collapse here
    is whatever the richest cover yields (on the real map the lone k=4 face, 3.55×)."""
    if worlds is None:
        worlds = world_array(inst)
    seen = {}
    for f in inst.faces:
        seen[frozenset(f.cover)] = None
    out = []
    n = len(worlds)
    for cover in sorted(seen, key=lambda c: (len(c), sorted(c))):
        m = bitmask(cover)
        hit = (worlds & m) != 0
        npos = int(hit.sum())
        nneg = n - npos
        out.append({
            "cover": sorted(cover),
            "k": len(cover),
            "n_pos": npos, "n_neg": nneg,
            "prune_pos": (n / npos) if npos else float("inf"),
            "prune_neg": (n / nneg) if nneg else float("inf"),
        })
    return out


def best_single_face_collapse(inst: Instance, worlds: np.ndarray = None) -> dict:
    """[DET-DEP] The single face with the largest NEGATIVE-read collapse — the
    honest replacement for the headline "5.16× free at entry".  Negative collapse
    is monotone in cover size (ruling out more treasures), so this is the richest
    cover."""
    tbl = face_collapse(inst, worlds)
    return max(tbl, key=lambda r: r["prune_neg"])


def cluster_resolution_chain(inst: Instance, cluster: list,
                             worlds: np.ndarray = None) -> dict:
    """[DET-DEP] A GREEDY chain of face-reads that resolves a cluster's occupancy
    as far as its faces allow, and its world-collapse.  The length is an UPPER
    BOUND on the reads a chain needs (greedy-maximal informative set, not a proven
    minimum) — the point is the order of magnitude: a cluster takes MANY reads, not
    one.

    WHY THIS MATTERS: the old note claimed a cluster collapses in ONE read because
    "a detector covers the cluster".  Under faces no single face covers a whole
    cluster (the max real face is k=4 over a 5-treasure cluster), so resolving a
    cluster needs a CHAIN of geographically-distinct face reads.  We build the
    chain greedily: repeatedly add the available cluster-face that most reduces the
    expected surviving-world count E[|belief|] = Σ n_i² / Σ n_i over the joint
    outcome partition, until adding any further cluster-face no longer shrinks it.
    (The terminal belief equals reading EVERY distinct cluster face — greedy just
    orders them by marginal leverage and drops the ones that add nothing.)

    Returns the chain (cover-sets in order), the per-step expected |belief|, and
    the final prune.  Bounded: the partition is over ≤ 2**(#cluster faces) cells,
    and the chain is capped at the number of distinct cluster faces."""
    if worlds is None:
        worlds = world_array(inst)
    cset = set(cluster)
    # distinct cover-sets that live entirely inside this cluster
    cand = sorted({frozenset(f.cover) for f in inst.faces if set(f.cover) <= cset},
                  key=lambda c: (-len(c), sorted(c)))
    masks = [bitmask(c) for c in cand]
    n = len(worlds)

    def expected_belief(chain_masks):
        if not chain_masks:
            return float(n)
        # outcome signature per world over the chosen faces
        sig = np.zeros(n, dtype=np.int64)
        for b, m in enumerate(chain_masks):
            sig |= (((worlds & m) != 0).astype(np.int64) << b)
        _, counts = np.unique(sig, return_counts=True)
        return float((counts.astype(float) ** 2).sum() / n)

    chosen, chosen_masks, steps = [], [], []
    cur = float(n)
    remaining = list(range(len(cand)))
    while remaining:
        best_i, best_e = None, cur
        for i in remaining:
            e = expected_belief(chosen_masks + [masks[i]])
            if e < best_e - 1e-9:
                best_e, best_i = e, i
        if best_i is None:
            break
        chosen.append(sorted(cand[best_i]))
        chosen_masks.append(masks[best_i])
        steps.append({"added": sorted(cand[best_i]), "E_belief": round(best_e, 1)})
        cur = best_e
        remaining.remove(best_i)
    return {
        "cluster": cluster,
        "chain": chosen,
        "chain_len": len(chosen),
        "steps": steps,
        "E_belief_final": round(cur, 1),
        "prune_final": (n / cur) if cur else float("inf"),
        "n_distinct_cluster_faces": len(cand),
    }


def sweep_collapse(inst: Instance, covers: list, worlds: np.ndarray = None) -> dict:
    """[DET-DEP] Expected surviving worlds after jointly reading a SET of faces
    (one signature per world; E[|belief|] = Σ n_i²/N).  Generalizes the §2.2
    multi-step chain table to any face set."""
    if worlds is None:
        worlds = world_array(inst)
    n = len(worlds)
    sig = np.zeros(n, dtype=np.int64)
    for b, cover in enumerate(covers):
        sig |= (((worlds & bitmask(cover)) != 0).astype(np.int64) << b)
    vals, counts = np.unique(sig, return_counts=True)
    e = float((counts.astype(float) ** 2).sum() / n)
    return {"faces": [sorted(c) for c in covers], "E_belief": round(e, 1),
            "prune": round(n / e, 2), "n_patterns": int(len(vals))}


# ===========================================================================
# 3. Occupancy factorization (DET-IND — a property of the prior; ports verbatim)
# ===========================================================================

def occupancy_factorization(inst: Instance, partition: list,
                            sample_vector: list = None,
                            worlds: np.ndarray = None) -> dict:
    """[DET-IND] Verify #worlds with occupancy (k_c) = ∏_c C(|cell_c|, k_c) for a
    given partition of treasures into cells.

    This is the keystone the old §3.1 names, GENERALIZED to any partition (not just
    the cluster partition): conditioned on the per-cell counts, the worlds factor
    as independent within-cell uniform subsets; the sole coupling is Σ k_c = K.
    Detector-INDEPENDENT — a property of the exactly-K prior, ports verbatim.

    With no sample_vector given, picks a feasible occupancy vector and checks the
    product identity by direct count over the world array."""
    if worlds is None:
        worlds = world_array(inst)
    sizes = [len(cell) for cell in partition]
    if sample_vector is None:
        # a feasible NON-degenerate vector: spread the budget across as many cells
        # as possible (so the check exercises a real product, not C(s,K)·1·1·…).
        sample_vector, budget = [0] * len(sizes), inst.K
        while budget > 0:
            spread = False
            for i, s in enumerate(sizes):
                if budget > 0 and sample_vector[i] < s:
                    sample_vector[i] += 1
                    budget -= 1
                    spread = True
            if not spread:
                break
    # direct count
    n = len(worlds)
    keep = np.ones(n, dtype=bool)
    for cell, kc in zip(partition, sample_vector):
        m = bitmask(cell)
        popcount = np.array([bin(int(w & m)).count("1") for w in worlds])
        keep &= (popcount == kc)
    counted = int(keep.sum())
    product = 1
    for s, kc in zip(sizes, sample_vector):
        product *= comb(s, kc)
    return {
        "partition_sizes": sizes,
        "occupancy_vector": sample_vector,
        "counted_worlds": counted,
        "product_formula": product,
        "exact_match": counted == product,
    }


def n_occupancy_partitions(inst: Instance, partition: list) -> int:
    """[DET-IND] Count occupancy vectors (k_c) with Σ k_c = K and 0 ≤ k_c ≤ |cell_c|
    — the size of the macro state space.  Ports the §4 "613 partitions" computation,
    generalized to any partition."""
    sizes = [len(cell) for cell in partition]

    def rec(i, budget):
        if i == len(sizes):
            return 1 if budget == 0 else 0
        return sum(rec(i + 1, budget - k) for k in range(min(sizes[i], budget) + 1))

    return rec(0, inst.K)


# ===========================================================================
# 4. Exact-solvable sub-problems — per-cluster reachable-belief sizing
# ===========================================================================

def reachable_local_beliefs(inst: Instance, cluster: list,
                            max_cluster_bits: int = 8,
                            max_states: int = 200_000) -> dict:
    """[DET-DEP] BFS over a cluster's LOCAL belief lattice: states are frozensets
    of in-cluster latent subsets (≤ 2**size of them).  Transitions: every distinct
    in-cluster face read (both polarities) and every in-cluster collect (both
    presence outcomes).  Counts distinct reachable beliefs — the size of the exact
    backward-induction table for that cluster.

    SUPERSEDES the §4 per-cluster sizing: those used cover_mask faces; here the
    in-cluster faces are the true arrangement faces (more of them, because a
    cluster now exposes singletons + asymmetric covers), so the reachable count
    differs.

    BOUNDEDNESS (load-bearing — ADR-0002 fail-loud, never hang/OOM): the reachable
    belief count grows much FASTER than 2**size (the lattice of subsets of the
    2**size latent subsets), so a size cap alone is not enough.  Two hard guards:
      * `max_cluster_bits` refuses clusters whose latent space 2**size is too big
        to enumerate at all (default 8 → ≤256 latent subsets);
      * `max_states` aborts the BFS the moment the reachable set exceeds the cap,
        returning a `truncated` marker rather than running unbounded.
    The honest reading: "exactly backward-induction-solvable" is a property of
    SMALL clusters only.  A large fused cluster (the real map's 6-cluster, or a
    dense synthetic blob) is over the cap and must be sub-decomposed, not solved
    flat — which the report states plainly."""
    size = len(cluster)
    if size > max_cluster_bits:
        return {"cluster": cluster, "size": size, "latent_subsets": 1 << size,
                "reachable_beliefs": None,
                "skipped": f"2**{size} latent subsets exceeds cap 2**{max_cluster_bits}"}
    idx = {t: b for b, t in enumerate(cluster)}
    full = frozenset(range(1 << size))   # all latent subsets, as the initial belief

    # in-cluster faces, as local bitmasks over the cluster's own bit positions
    local_face_masks = set()
    cset = set(cluster)
    for f in inst.faces:
        if set(f.cover) <= cset and f.cover:
            local_face_masks.add(sum(1 << idx[t] for t in f.cover))

    def split_face(belief, fm):
        pos = frozenset(s for s in belief if (s & fm))
        neg = frozenset(s for s in belief if not (s & fm))
        return [b for b in (pos, neg) if b]

    def split_collect(belief, bit):
        present = frozenset(s for s in belief if (s >> bit) & 1)
        absent = frozenset(s for s in belief if not ((s >> bit) & 1))
        return [b for b in (present, absent) if b]

    seen = {full}
    frontier = [full]
    while frontier:
        nxt = []
        for belief in frontier:
            children = []
            for fm in local_face_masks:
                children += split_face(belief, fm)
            for bit in range(size):
                children += split_collect(belief, bit)
            for ch in children:
                if ch not in seen:
                    seen.add(ch)
                    if len(ch) > 1:                  # singleton beliefs are leaves
                        nxt.append(ch)
                    if len(seen) > max_states:       # hard abort — never run unbounded
                        return {"cluster": cluster, "size": size,
                                "latent_subsets": 1 << size,
                                "n_local_faces": len(local_face_masks),
                                "reachable_beliefs": None,
                                "truncated": f">{max_states} reachable beliefs (over cap)"}
        frontier = nxt
    return {"cluster": cluster, "size": size,
            "latent_subsets": 1 << size,
            "n_local_faces": len(local_face_masks),
            "reachable_beliefs": len(seen)}


# ===========================================================================
# 5. Indistinguishability under the face model
# ===========================================================================

def face_signature(inst: Instance, t: int) -> frozenset:
    """[DET-DEP] The set of DISTINCT cover-sets that mention treasure t.  Two
    treasures are face-indistinguishable iff every face covers both or neither —
    equivalently iff they have the same set of (distinct) covers mentioning them.

    Re-DEFINED under faces.  The old §1.4 used the cover_mask signature and found
    {8,9},{11,12},{13,14},{17,18} indistinguishable; under faces those pairs are
    SEPARATED (a singleton or asymmetric face covers exactly one of each), so they
    collapse to distinguishable singletons.  Only the δ-treasures (no faces) stay
    mutually indistinguishable."""
    distinct = sorted({frozenset(f.cover) for f in inst.faces}, key=lambda c: (len(c), sorted(c)))
    return frozenset(i for i, c in enumerate(distinct) if t in c)


def indistinguishability_classes(inst: Instance) -> list:
    """[DET-DEP] Group treasures by identical face-signature.  Classes of size > 1
    are the genuine sensing floor — pairs no face chain can separate (a collect is
    required)."""
    groups = {}
    for t in inst.ids:
        groups.setdefault(face_signature(inst, t), []).append(t)
    return sorted((sorted(ts) for ts in groups.values()), key=lambda ts: (-len(ts), ts))


def full_sense_floor(inst: Instance, worlds: np.ndarray = None) -> dict:
    """[DET-DEP] Reading EVERY distinct face: the residual belief.  E[|belief|] =
    Σ n_i²/N over the joint outcome partition.  The honest replacement for the old
    "120.7-world / 319-class floor" — under faces the singleton covers pin every
    region-covered treasure, so the floor is driven only by the δ-treasures and any
    surviving size>1 indistinguishability class."""
    if worlds is None:
        worlds = world_array(inst)
    distinct = sorted({frozenset(f.cover) for f in inst.faces}, key=lambda c: (len(c), sorted(c)))
    return sweep_collapse(inst, distinct, worlds) | {"n_distinct_faces": len(distinct)}


# ===========================================================================
# 6. Recommended decomposition — operational reachability of per-cluster solving
# ===========================================================================

def decomposition_assessment(inst: Instance, worlds: np.ndarray = None) -> dict:
    """[DET-DEP] Is per-cluster decomposition still OPERATIONALLY reachable now
    that a cluster needs a face-read CHAIN (not one read)?  Quantifies, per
    cluster: the chain length to resolve it, the chain's world-collapse, and the
    exact-solvable local-belief count.  Honest verdict: the *factorization*
    (DET-IND) survives, but the *one-read cluster probe* the old hierarchy leaned
    on does not — each cluster's deep VoI is behind a multi-face chain."""
    if worlds is None:
        worlds = world_array(inst)
    cls = clusters(inst)
    rows = []
    for c in cls:
        if len(c) == 1:
            continue   # δ singletons: observe == collect, no chain
        chain = cluster_resolution_chain(inst, c, worlds)
        rb = reachable_local_beliefs(inst, c)
        rows.append({
            "cluster": c,
            "resolution_chain_len": chain["chain_len"],
            "chain_prune": round(chain["prune_final"], 2),
            "reachable_local_beliefs": rb.get("reachable_beliefs"),
        })
    return {"clusters": rows,
            "delta_treasures": sorted(inst.delta),
            "n_macro_partitions": n_occupancy_partitions(inst, cls)}


# ===========================================================================
# The report
# ===========================================================================

@dataclass
class StructuralReport:
    n_faces: int
    n_treasures: int
    K: int
    n_worlds: int
    clusters: list = field(default_factory=list)
    cluster_geography: list = field(default_factory=list)
    delta_treasures: list = field(default_factory=list)
    face_collapse: list = field(default_factory=list)
    best_single_face: dict = field(default_factory=dict)
    cluster_chains: list = field(default_factory=list)
    occupancy_check: dict = field(default_factory=dict)
    n_macro_partitions: int = 0
    reachable_beliefs: list = field(default_factory=list)
    indistinguishability: list = field(default_factory=list)
    sense_floor: dict = field(default_factory=dict)
    decomposition: dict = field(default_factory=dict)


def analyze(inst: Instance) -> StructuralReport:
    """The whole structural decomposition, composed from the named functions
    above.  Every field is reproducible by calling its function directly."""
    worlds = world_array(inst)
    cls = clusters(inst)
    rep = StructuralReport(
        n_faces=len(inst.faces),
        n_treasures=inst.N,
        K=inst.K,
        n_worlds=len(worlds),
        clusters=cls,
        cluster_geography=cluster_geography(inst),
        delta_treasures=sorted(inst.delta),
        face_collapse=face_collapse(inst, worlds),
        best_single_face=best_single_face_collapse(inst, worlds),
        cluster_chains=[cluster_resolution_chain(inst, c, worlds)
                        for c in cls if len(c) > 1],
        occupancy_check=occupancy_factorization(inst, cls, worlds=worlds),
        n_macro_partitions=n_occupancy_partitions(inst, cls),
        reachable_beliefs=[reachable_local_beliefs(inst, c) for c in cls if len(c) > 1],
        indistinguishability=indistinguishability_classes(inst),
        sense_floor=full_sense_floor(inst, worlds),
        decomposition=decomposition_assessment(inst, worlds),
    )
    return rep


def _print_report(rep: StructuralReport) -> None:
    print(f"faces={rep.n_faces}  treasures={rep.n_treasures}  K={rep.K}  worlds={rep.n_worlds}")
    print(f"\nclusters: {rep.clusters}")
    print(f"δ-treasures (sense-isolated): {rep.delta_treasures}")
    print(f"\nbest single-face NEG collapse: cover {rep.best_single_face['cover']} "
          f"-> {rep.best_single_face['prune_neg']:.2f}× (n_neg={rep.best_single_face['n_neg']})")
    print("\nsingle-face collapse (distinct covers):")
    for r in rep.face_collapse:
        print(f"  {r['cover']!s:18} k={r['k']}  neg {r['prune_neg']:.2f}× ({r['n_neg']})  "
              f"pos {r['prune_pos']:.2f}× ({r['n_pos']})")
    print("\ncluster resolution chains (min face-read chain to resolve a cluster):")
    for ch in rep.cluster_chains:
        print(f"  {ch['cluster']}: {ch['chain_len']} reads {ch['chain']} "
              f"-> E[belief]={ch['E_belief_final']} prune={ch['prune_final']:.2f}×")
    print(f"\noccupancy factorization check: {rep.occupancy_check}")
    print(f"macro occupancy partitions (Σk=K): {rep.n_macro_partitions}")
    print("\nper-cluster reachable local beliefs (exact-solvable sizing):")
    for rb in rep.reachable_beliefs:
        print(f"  {rb}")
    print(f"\nindistinguishability classes (face model): {rep.indistinguishability}")
    print(f"full-sense floor: {rep.sense_floor}")
    print(f"\ndecomposition assessment: {rep.decomposition}")


if __name__ == "__main__":
    _print_report(analyze(real_instance()))
