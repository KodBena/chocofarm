#!/usr/bin/env python3
"""
throughput-lab/harness/topology_enum.py — the SINGLE HOME of the throughput lab's process-topology
config space (ADR-0012 P1: one home; the sweep harness consumes this, it never re-authors the space).

WHAT THIS IS
------------
A declarative OR-Tools CP-SAT model that places the lab's process/thread classes onto the box's
vCPUs AND assigns each a scheduling policy, subject to "reasonableness" constraints, then ENUMERATES
ALL distinct feasible placements (CP-SAT `enumerate_all_solutions`). The output is a typed, runnable
config record per solution: a cpu-list (for `taskset -c`) and a sched policy (+nice / latency-nice)
per class. This replaces ad-hoc "try this pin and that pin" with a generative, single-homed,
symmetry-reduced candidate set the A/B sweep harness reads as its source of truth.

THE CLASSES PLACED (the lab's processes)
----------------------------------------
  - SERVER   : 1 INFERENCE consumer. Bursty + mildly latency-sensitive (matmul forwards interleaved
               with socket I/O; ~0.58 core busy). Wants prompt scheduling of its bursts.
  - GEN_k    : G GENERATOR producers (default 3). CPU-bound tree-search, always runnable. The work.
  - SURPLUS  : 0 or 1 extra generator — the low-priority slack-filler that soaks fragmented idle on
               a core it SHARES. Present iff it is the only class permitted to co-locate.

THE PLACEMENT SUBSTRATE (a HYPOTHESIS, not a hard truth)
--------------------------------------------------------
Host boots `rcu_nocbs=1-3 isolcpus=managed_irq,nohz,1-3 irqaffinity=0`:
  - core 0  : housekeeping — IRQs affined here, RCU callbacks offloaded ONTO it, general system tasks
              → contended.
  - cores 1-3: isolated — no managed IRQs, nohz_full, RCU-offloaded → clean compute.
CAVEAT (do NOT hard-code as truth): the lab runs in a guest that may not fully see host isolation, and
a prior in-guest A/B refuted a core-3 benefit. So core 0's contention is treated as a HYPOTHESIS the
enumeration must let us TEST — the config set deliberately SPANS "server on housekeeping core 0" and
"server on an isolated core with a generator/surplus on 0", and the co-locations between. We do not
prune by the isolation assumption; we span it.

THE SCHED-POLICY VOCABULARY (per kernel consult; EEVDF, 6.19)
-------------------------------------------------------------
Per-task knobs, carried as a typed attribute of each placement (we model WHICH policy each class may
take; we do NOT model the kernel):
  - OTHER(nice)    : SCHED_OTHER at a given nice. The generators' default (nice 0).
  - BATCH          : SCHED_BATCH — throughput-friendly, no wakeup preemption. A generator alt.
  - IDLE           : SCHED_IDLE — runs only in true idle, yields instantly. The textbook surplus fit.
  - OTHER_LATNICE  : SCHED_OTHER with EEVDF latency-nice (-20) — the latency-sensitive server fit.

OUTPUT CONTRACT (the SSOT the harness consumes)
-----------------------------------------------
`python topology_enum.py` writes a JSON array (default ./configs.json) AND prints a human table. Each
record is a typed self-describing `TopologyConfig` (ADR-0012: typed signature is the contract) — a
stable id/name, a one-line tag, and a per-class `Placement` = (cpus: list[int] for `taskset -c`,
policy, nice, latency_nice). Directly translatable to "how to launch this run".

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ortools.sat.python import cp_model


# ----------------------------------------------------------------------------------------------------
# Typed config records (ADR-0012: the typed signature IS the contract the harness reads).
# ----------------------------------------------------------------------------------------------------
class SchedPolicy(str, Enum):
    """The per-task scheduling-policy choices from the kernel consult vocabulary."""
    OTHER = "SCHED_OTHER"          # plain CFS/EEVDF, weighted by nice
    BATCH = "SCHED_BATCH"          # throughput-friendly, suppresses wakeup preemption
    IDLE = "SCHED_IDLE"            # runs only in true idle, yields instantly (slack-filler)
    OTHER_LATNICE = "SCHED_OTHER_LATNICE"  # SCHED_OTHER + EEVDF latency-nice (low-latency wakeups)


@dataclass(frozen=True)
class Placement:
    """One class pinned to a cpu-list with a scheduling policy. `cpus` feeds `taskset -c`."""
    role: str                 # "server" | "gen" | "surplus"
    cpus: tuple[int, ...]     # taskset -c cpu list
    policy: SchedPolicy
    nice: Optional[int] = None          # for OTHER / BATCH
    latency_nice: Optional[int] = None  # for OTHER_LATNICE

    def to_record(self) -> dict:
        return {
            "role": self.role,
            "cpus": list(self.cpus),
            "taskset": ",".join(str(c) for c in self.cpus),
            "policy": self.policy.value,
            "nice": self.nice,
            "latency_nice": self.latency_nice,
        }


@dataclass(frozen=True)
class TopologyConfig:
    """A full enumerated candidate: every class placed. The unit the sweep harness A/B-tests."""
    config_id: str
    tag: str
    placements: tuple[Placement, ...]

    def to_record(self) -> dict:
        return {
            "config_id": self.config_id,
            "tag": self.tag,
            "placements": [p.to_record() for p in self.placements],
        }


# ----------------------------------------------------------------------------------------------------
# The CP-SAT model. Decision: for each class, which core it sits on + which sched policy it carries.
# We model placement as a single-core pin per class (the lab pins each producer/consumer to one core;
# a class never spans cores here — that is the regime we are A/B-testing). Policy is a per-class index
# into the small vocabulary admissible for that class's ROLE.
# ----------------------------------------------------------------------------------------------------

# Policy vocabularies admissible PER ROLE (the typed reasonableness on policy):
#   server  : latency-nice (the fit) OR plain nice-0 OTHER (the null/baseline, to test the latency-nice
#             knob actually buys anything). Not BATCH/IDLE — a latency-sensitive consumer wants wakeups.
#   gen     : OTHER nice-0 (default) OR BATCH (throughput-friendly alt). The two honest generator modes.
#   surplus : IDLE (textbook) OR BATCH-at-positive-nice (graceful-but-not-starvable alt). Low priority
#             by definition — never nice-0 OTHER, or it is just another generator.
SERVER_POLICIES = [
    (SchedPolicy.OTHER_LATNICE, None, -20),
    (SchedPolicy.OTHER, 0, None),
]
GEN_POLICIES = [
    (SchedPolicy.OTHER, 0, None),
    (SchedPolicy.BATCH, 0, None),
]
SURPLUS_POLICIES = [
    (SchedPolicy.IDLE, None, None),
    (SchedPolicy.BATCH, 10, None),
]


def _policy_is_low_prio(pol: SchedPolicy) -> bool:
    """A policy that yields to nice-0 CPU-bound work — safe to co-locate with the server."""
    return pol in (SchedPolicy.IDLE,)


@dataclass
class ModelParams:
    n_cores: int = 4
    n_gens: int = 3
    housekeeping_core: int = 0   # the contended IRQ/RCU core; cores 1..n-1 are the isolated set


def build_and_enumerate(p: ModelParams) -> list[TopologyConfig]:
    cores = list(range(p.n_cores))
    isolated = [c for c in cores if c != p.housekeeping_core]
    hk = p.housekeeping_core

    m = cp_model.CpModel()

    # --- placement vars: core[role] in 0..n_cores-1 -------------------------------------------------
    server_core = m.new_int_var(0, p.n_cores - 1, "server_core")
    gen_core = [m.new_int_var(0, p.n_cores - 1, f"gen{g}_core") for g in range(p.n_gens)]
    surplus_present = m.new_bool_var("surplus_present")
    # surplus core only meaningful when present; pin to hk when absent to keep enumeration canonical.
    surplus_core = m.new_int_var(0, p.n_cores - 1, "surplus_core")

    # --- policy vars: an index into the per-role vocabulary -----------------------------------------
    # The generator policy is a SINGLE uniform knob applied to ALL generators (gen_pol), not a
    # per-generator choice. Rationale: the generators are interchangeable CPU-bound workers; "which
    # specific generator runs BATCH" carries no experimental signal — only the mode (all-OTHER vs
    # all-BATCH) is a knob worth A/B-testing. Letting each of G generators pick independently would
    # multiply the space by len(GEN_POLICIES)**G (the 2**3 = 8x explosion that bloated an early
    # version to 240 configs) with zero added information. One knob, one home.
    server_pol = m.new_int_var(0, len(SERVER_POLICIES) - 1, "server_pol")
    gen_pol_uniform = m.new_int_var(0, len(GEN_POLICIES) - 1, "gen_pol")
    gen_pol = [gen_pol_uniform for _ in range(p.n_gens)]  # same var → uniform across generators
    surplus_pol = m.new_int_var(0, len(SURPLUS_POLICIES) - 1, "surplus_pol")

    # Per-core occupancy booleans (who is on core c), to write co-location constraints cleanly.
    # server_on[c], gen_on[g][c], surplus_on[c]
    server_on = {c: m.new_bool_var(f"server_on_{c}") for c in cores}
    for c in cores:
        m.add(server_core == c).only_enforce_if(server_on[c])
        m.add(server_core != c).only_enforce_if(server_on[c].Not())
    gen_on = [{c: m.new_bool_var(f"gen{g}_on_{c}") for c in cores} for g in range(p.n_gens)]
    for g in range(p.n_gens):
        for c in cores:
            m.add(gen_core[g] == c).only_enforce_if(gen_on[g][c])
            m.add(gen_core[g] != c).only_enforce_if(gen_on[g][c].Not())
    # reify (surplus_core == c) per core, then AND with presence.
    surplus_at = {c: m.new_bool_var(f"surplus_at_{c}") for c in cores}
    for c in cores:
        m.add(surplus_core == c).only_enforce_if(surplus_at[c])
        m.add(surplus_core != c).only_enforce_if(surplus_at[c].Not())
    surplus_on = {c: m.new_bool_var(f"surplus_on_{c}") for c in cores}
    for c in cores:
        # surplus occupies c iff present AND placed there
        m.add_bool_and([surplus_present, surplus_at[c]]).only_enforce_if(surplus_on[c])
        m.add_bool_or([surplus_present.Not(), surplus_at[c].Not()]).only_enforce_if(surplus_on[c].Not())

    # =================================================================================================
    # REASONABLENESS CONSTRAINTS  (the artifact's value — each is a judgment call, tagged below)
    # =================================================================================================

    # R1. The n_gens generators each get a CLEAN, EXCLUSIVE core: no two nice-0 CPU-bound generators
    #     contend on one core (that is the contention the whole lab exists to avoid), and a generator
    #     never shares with the server. Generators are the bottleneck work; they want clean compute.
    #     => generator cores are pairwise distinct, and no server/surplus on a generator's core
    #        EXCEPT the surplus, which may co-locate (handled in R4).
    m.add_all_different(gen_core)
    for g in range(p.n_gens):
        for c in cores:
            # server never co-locates with a generator
            m.add_bool_or([gen_on[g][c].Not(), server_on[c].Not()])

    # R2. No clean isolated core sits idle while generators contend. With G generators and (n-1)
    #     isolated cores, if G >= (n-1) every isolated core must carry a generator; more generally we
    #     forbid an isolated core being EMPTY when any core is doubly-occupied. We encode the strong,
    #     simple form the reference motivates: every generator should be on an isolated core when
    #     there are enough isolated cores to hold them all (the housekeeping core is the contended one,
    #     so a generator only lands on core 0 when forced — i.e. when we deliberately put the server on
    #     an isolated core and push a generator onto 0, the spanning case the brief wants).
    #     We DON'T force generators off core 0 (that would prune the very "generator-on-housekeeping"
    #     span). Instead: at most ONE class-bearing core may be the housekeeping core's "demotion",
    #     and no isolated core may be left bare while two classes share another. See R3.

    # R3. Exactly one core hosts the server. The server sits on SOME core (always true by var domain).
    #     Span requirement: we want BOTH "server on housekeeping (0)" and "server on an isolated core".
    #     Both are feasible already; no constraint needed — they fall out of enumeration. We only
    #     prevent NONSENSE: an isolated core may not be left completely empty while the server is on
    #     the housekeeping core AND a generator could have used that isolated core. With all_different
    #     generators + server-not-on-gen-core, the only empties arise when n_cores > n_gens+1; for the
    #     default (4 cores, 3 gens) every core is used, so no empties. We add the general guard:
    #     no core is empty unless n_cores > n_gens + 1 forces slack.
    occupied = {}
    for c in cores:
        occ = m.new_bool_var(f"occupied_{c}")
        members = [server_on[c]] + [gen_on[g][c] for g in range(p.n_gens)] + [surplus_on[c]]
        m.add_max_equality(occ, members)
        occupied[c] = occ
    # Number of distinct classes (excluding surplus) is n_gens + 1 (server). They need n_gens+1 cores.
    # If n_cores == n_gens + 1, every core MUST be occupied (no clean core idle while work waits, R2).
    if p.n_cores == p.n_gens + 1:
        for c in cores:
            m.add(occupied[c] == 1)

    # R4. CO-LOCATION is the surplus's exclusive privilege, and only via a low-priority policy.
    #     - Any core carrying two classes: one of them is the surplus, and the surplus's policy is the
    #       low-priority (IDLE) one when it shares with the latency-sensitive server.
    #     - The server may share ONLY with a SCHED_IDLE surplus (never with a nice-0 generator: R1
    #       already forbids server+gen co-location).
    #     - Generators never double up (R1 all_different + server exclusion). So the ONLY co-location
    #       is surplus-with-{server or a generator}.
    # Surplus, when present, MUST co-locate (it is the slack-filler; a surplus on its own private core
    # is just a 4th generator — not what "surplus" means). So surplus_core ∈ {a core already used}.
    for c in cores:
        # if surplus is on c, some other class is also on c
        others_on_c = [server_on[c]] + [gen_on[g][c] for g in range(p.n_gens)]
        # surplus_on[c] implies at least one other occupant
        m.add(sum(others_on_c) >= 1).only_enforce_if(surplus_on[c])

    # If the surplus shares the SERVER's core, its policy must be the low-priority IDLE (index 0),
    # so it cannot steal the latency-sensitive consumer's wakeups. Sharing a generator's core, BATCH
    # at positive nice is also allowed (the generator is nice-0 OTHER/BATCH and out-weighs it).
    # surplus_with_server <=> surplus_present AND (surplus_core == server_core). We build it as the
    # conjunction of presence and a reified core-equality so it is FALSE (not contradictory) when the
    # surplus is absent — even if the canonicalized absent surplus_core happens to equal server_core.
    surplus_eq_server = m.new_bool_var("surplus_eq_server")  # raw core equality, presence-agnostic
    m.add(surplus_core == server_core).only_enforce_if(surplus_eq_server)
    m.add(surplus_core != server_core).only_enforce_if(surplus_eq_server.Not())
    surplus_with_server = m.new_bool_var("surplus_with_server")
    m.add_bool_and([surplus_present, surplus_eq_server]).only_enforce_if(surplus_with_server)
    m.add_bool_or([surplus_present.Not(), surplus_eq_server.Not()]).only_enforce_if(surplus_with_server.Not())
    # IDLE is index 0 in SURPLUS_POLICIES: sharing the server's core demands the yield-instantly policy.
    m.add(surplus_pol == 0).only_enforce_if(surplus_with_server)

    # When surplus is ABSENT, canonicalize its unused vars so enumeration does not multiply phantom
    # solutions: pin surplus_core to housekeeping and surplus_pol to 0. (surplus_with_server is now
    # gated on presence, so pinning surplus_core==hk cannot collide with a server also on hk.)
    m.add(surplus_core == hk).only_enforce_if(surplus_present.Not())
    m.add(surplus_pol == 0).only_enforce_if(surplus_present.Not())

    # =================================================================================================
    # SYMMETRY BREAKING  (so relabelings do not multiply the output)
    # =================================================================================================
    # S1. The G generators are interchangeable. Order them by (core, policy) to fix one representative
    #     per orbit. Strict lexicographic increase on a packed key core*K + pol.
    K = len(GEN_POLICIES)
    gen_key = []
    for g in range(p.n_gens):
        key = m.new_int_var(0, (p.n_cores - 1) * K + (K - 1), f"gen{g}_key")
        m.add(key == gen_core[g] * K + gen_pol[g])
        gen_key.append(key)
    for g in range(p.n_gens - 1):
        m.add(gen_key[g] < gen_key[g + 1])  # strict: distinct cores (R1) make this always satisfiable

    # The remaining symmetry — the permutation of the INTERCHANGEABLE ISOLATED CORES {1..n-1} acting
    # jointly with the generator relabeling — is NOT broken in the model. It is collapsed AFTER
    # enumeration by a provably-correct canonical-key dedup (`_canonical_key` + the dedup loop below).
    # We deliberately chose enumerate-then-canonicalize over in-model lex-leader constraints for the
    # ISOLATED-CORE symmetry: the canonical key IS the orbit invariant, so the reduction cannot
    # under-collapse (miss a true relabeling) or over-collapse (merge genuinely distinct configs) by
    # construction, for ANY (n_cores, n_gens). In-model lex constraints over a joint group are exactly
    # where naive symmetry-breaking silently errs; making the orbit boundary STRUCTURAL avoids that.
    # S1 (a within-class generator-index ordering) is kept because it is trivially correct and shrinks
    # the raw enumerated set; the ISOLATED-CORE orbit is handled solely by the canonical key.

    # =================================================================================================
    # ENUMERATE
    # =================================================================================================
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True

    collected: list[dict] = []

    class _Collector(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()

        def on_solution_callback(self):
            rec = {
                "server_core": self.value(server_core),
                "server_pol": self.value(server_pol),
                "gen_core": [self.value(x) for x in gen_core],
                "gen_pol": [self.value(x) for x in gen_pol],
                "surplus_present": bool(self.value(surplus_present)),
                "surplus_core": self.value(surplus_core),
                "surplus_pol": self.value(surplus_pol),
            }
            collected.append(rec)

    solver.solve(m, _Collector())

    # --- canonical-key dedup over the JOINT symmetry group ------------------------------------------
    # Collapse the orbit of (isolated-core permutations) x (generator permutations) by mapping each
    # raw solution to its canonical key (the lex-MIN image under the group) and keeping the FIRST
    # representative of each distinct key. Because the key is the orbit's lex-min, two raw solutions
    # share a key IFF they lie in the same orbit — exactly the merge the maintainer specified.
    by_key: dict[tuple, dict] = {}
    for rec in collected:
        key = _canonical_key(rec, p)
        if key not in by_key:
            by_key[key] = rec
    # Materialize in a stable order (by canonical key) so config ids/order are reproducible run-to-run.
    return [_materialize(by_key[k], p) for k in sorted(by_key.keys())]


def _canonical_key(rec: dict, p: ModelParams) -> tuple:
    """The orbit invariant: the lexicographically smallest image of `rec` under the JOINT group
    G = Sym(isolated cores {1..n-1}) x Sym(generators). Core 0 (housekeeping) is NEVER permuted (it is
    an asymmetric resource). Two raw solutions are isomorphic IFF they share this key.

    We build the key as a *relabeling-invariant fingerprint*: for each candidate isolated-core
    permutation, relabel every class's core, then describe the placement in a generator-anonymized,
    sorted form (which quotients out the generator permutation automatically). The min over all
    isolated-core permutations is the canonical form.
    """
    hk = p.housekeeping_core
    isolated = [c for c in range(p.n_cores) if c != hk]

    # The placement as (role-class, core, policy-index) triples; generators are anonymized to the
    # class label "g" so any generator permutation maps to the same multiset.
    base: list[tuple] = [("s", rec["server_core"], rec["server_pol"])]
    for g in range(p.n_gens):
        base.append(("g", rec["gen_core"][g], rec["gen_pol"][g]))
    if rec["surplus_present"]:
        base.append(("u", rec["surplus_core"], rec["surplus_pol"]))

    best = None
    for sigma in itertools.permutations(isolated):
        relabel = dict(zip(isolated, sigma))  # bijection on isolated cores; hk fixed
        imaged = tuple(sorted(
            (role, relabel.get(core, core), pol) for (role, core, pol) in base
        ))
        if best is None or imaged < best:
            best = imaged
    return best


def _materialize(rec: dict, p: ModelParams) -> TopologyConfig:
    """Turn a raw solver assignment into the typed, tagged, runnable TopologyConfig."""
    hk = p.housekeeping_core
    placements: list[Placement] = []

    s_pol, s_nice, s_lat = SERVER_POLICIES[rec["server_pol"]]
    placements.append(Placement("server", (rec["server_core"],), s_pol, s_nice, s_lat))

    for g in range(p.n_gens):
        g_pol, g_nice, g_lat = GEN_POLICIES[rec["gen_pol"][g]]
        placements.append(Placement(f"gen{g}", (rec["gen_core"][g],), g_pol, g_nice, g_lat))

    if rec["surplus_present"]:
        u_pol, u_nice, u_lat = SURPLUS_POLICIES[rec["surplus_pol"]]
        placements.append(Placement("surplus", (rec["surplus_core"],), u_pol, u_nice, u_lat))

    # --- tag / rationale ----------------------------------------------------------------------------
    server_on_hk = rec["server_core"] == hk
    parts = []
    parts.append("server-on-housekeeping" if server_on_hk else "server-isolated")
    if not server_on_hk:
        # who is on the housekeeping core 0?
        on_hk = [g for g in range(p.n_gens) if rec["gen_core"][g] == hk]
        if rec["surplus_present"] and rec["surplus_core"] == hk:
            parts.append("surplus-on-0")
        elif on_hk:
            parts.append("generator-on-0")
    if rec["surplus_present"]:
        if rec["surplus_core"] == rec["server_core"]:
            parts.append("surplus-shares-server")
        elif rec["surplus_core"] == hk:
            # sharing the generator on the HOUSEKEEPING core — distinct from sharing an isolated
            # generator (core 0 is an asymmetric resource: the orbit boundary the dedup respects).
            parts.append("surplus-shares-gen-on-0")
        else:
            parts.append("surplus-shares-gen-isolated")
    else:
        parts.append("no-surplus")
    # policy flavor
    if SERVER_POLICIES[rec["server_pol"]][0] == SchedPolicy.OTHER_LATNICE:
        parts.append("latnice")
    else:
        parts.append("plain-server")
    if any(GEN_POLICIES[rec["gen_pol"][g]][0] == SchedPolicy.BATCH for g in range(p.n_gens)):
        parts.append("gen-batch")
    tag = "+".join(parts)

    # --- stable id ----------------------------------------------------------------------------------
    sk = f"s{rec['server_core']}p{rec['server_pol']}"
    gk = "g" + "-".join(f"{rec['gen_core'][g]}.{rec['gen_pol'][g]}" for g in range(p.n_gens))
    uk = (f"u{rec['surplus_core']}p{rec['surplus_pol']}" if rec["surplus_present"] else "u_none")
    config_id = f"{sk}_{gk}_{uk}"

    return TopologyConfig(config_id=config_id, tag=tag, placements=tuple(placements))


# ----------------------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------------------
def _fmt_placement(pl: Placement) -> str:
    pol = pl.policy.value.replace("SCHED_", "")
    extra = ""
    if pl.nice is not None:
        extra = f" nice={pl.nice}"
    if pl.latency_nice is not None:
        extra = f" latnice={pl.latency_nice}"
    return f"{pl.role}@cpu[{','.join(map(str, pl.cpus))}] {pol}{extra}"


def render_table(configs: list[TopologyConfig]) -> str:
    lines = []
    lines.append(f"{len(configs)} feasible topology configs\n")
    for i, cfg in enumerate(configs):
        lines.append(f"[{i:>3}] {cfg.config_id}")
        lines.append(f"      tag: {cfg.tag}")
        for pl in cfg.placements:
            lines.append(f"        {_fmt_placement(pl)}")
        lines.append("")
    return "\n".join(lines)


def verify_orbits(configs: list[TopologyConfig], p: ModelParams) -> tuple[bool, str]:
    """Self-check the orbit reduction with an INDEPENDENT oracle: confirm the emitted configs are
    pairwise non-isomorphic under the JOINT group G = Sym(isolated cores) x Sym(generators). This
    recomputes the orbit by brute force over isolated-core permutations (generator permutation is
    quotiented by the anonymized-and-sorted fingerprint), so it does not trust `_canonical_key`'s
    construction — it cross-checks the RESULT. Returns (ok, message)."""
    hk = p.housekeeping_core
    isolated = [c for c in range(p.n_cores) if c != hk]

    def fingerprint(cfg: TopologyConfig) -> tuple:
        items = []
        for pl in cfg.placements:
            role = "g" if pl.role.startswith("gen") else pl.role
            items.append((role, pl.cpus[0], pl.policy.value, pl.nice, pl.latency_nice))
        return tuple(sorted(items))

    def orbit(fp: tuple) -> set:
        out = set()
        for sigma in itertools.permutations(isolated):
            relabel = dict(zip(isolated, sigma))
            out.add(tuple(sorted(
                (role, relabel.get(core, core), pol, nice, lat) for (role, core, pol, nice, lat) in fp
            )))
        return out

    seen_orbit_rep: dict[tuple, str] = {}
    collisions = []
    for cfg in configs:
        fp = fingerprint(cfg)
        hit = next((seen_orbit_rep[img] for img in orbit(fp) if img in seen_orbit_rep), None)
        if hit is not None:
            collisions.append((hit, cfg.config_id))
        else:
            rep = min(orbit(fp))
            seen_orbit_rep[rep] = cfg.config_id
    n_orbits = len(set(seen_orbit_rep.values()))
    ok = not collisions and n_orbits == len(configs)
    msg = (f"orbit self-check: {len(configs)} configs, {n_orbits} distinct orbits, "
           f"{len(collisions)} under-collapse collision(s)")
    if collisions:
        msg += " -> FAIL: " + ", ".join(f"{a}~{b}" for a, b in collisions[:5])
    return ok, msg


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=int, default=4, help="number of vCPUs (default 4)")
    ap.add_argument("--gens", type=int, default=3, help="number of generators (default 3)")
    ap.add_argument("--housekeeping-core", type=int, default=0,
                    help="the contended IRQ/RCU core (default 0); cores != this are isolated")
    ap.add_argument("--json", default="configs.json", help="output JSON path (default ./configs.json)")
    ap.add_argument("--no-table", action="store_true", help="suppress the human-readable table")
    ap.add_argument("--verify", action="store_true",
                    help="run the independent orbit self-check and fail loudly if the reduction is "
                         "not exactly one-representative-per-orbit (ADR-0002)")
    args = ap.parse_args(argv)

    if args.gens + 1 > args.cores:
        print(f"error: {args.gens} generators + 1 server need >= {args.gens + 1} cores, have {args.cores}",
              file=sys.stderr)
        return 2

    params = ModelParams(n_cores=args.cores, n_gens=args.gens, housekeeping_core=args.housekeeping_core)
    configs = build_and_enumerate(params)

    if args.verify:
        ok, msg = verify_orbits(configs, params)
        print(msg, file=sys.stderr)
        if not ok:
            print("ERROR: orbit reduction is unsound — refusing to emit (ADR-0002)", file=sys.stderr)
            return 3

    payload = {
        "schema": "throughput-lab/topology_enum/v1",
        "params": dataclasses.asdict(params),
        "n_configs": len(configs),
        "configs": [c.to_record() for c in configs],
    }
    with open(args.json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {len(configs)} configs -> {args.json}", file=sys.stderr)

    if not args.no_table:
        print(render_table(configs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
