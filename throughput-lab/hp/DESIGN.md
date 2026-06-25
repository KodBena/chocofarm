<!--
throughput-lab/hp/DESIGN.md — design spec for the throughput-lab hyperparameter SSOT and
its symmetry-reduced configuration-space compiler ("hpdsl"). The single home (ADR-0012 P1)
of every throughput-affecting hyperparameter's metadata, and the plan for the compiler that
generates the feasible, symmetry-reduced candidate set a constraint solver enumerates — the
generalization of throughput-lab/harness/topology_enum.py to the whole HP space.

Status: design spec (adjudicated synthesis of three independent proposals). No implementation
in this file; it is the contract the implementer follows.

Public Domain (The Unlicense).
-->

# `hpdsl` — a single-homed HP SSOT and its config-space compiler

## 0. The decision, up front

This spec adjudicates three independent design proposals (a CP-SAT lens, a Z3/SMT lens, and
a DSL/feature-model lens). They **converged** on almost everything that matters and disagreed
only on backend framing. The adjudicated decisions:

1. **Backend: CP-SAT is the primary enumerator. Z3/SMT is NOT used as the engine, and is
   declined for v1 even as an oracle.** All three proposals — *including the one written from
   the SMT lens* — concluded CP-SAT is the right enumeration engine for this problem shape and
   that SMT gives **no free symmetry reduction**. The SMT lens established this on the
   literature (Z3 has no first-class AllSMT mode; symmetry in SAT/SMT is *added* predicates,
   never automatic; the blocking-clause enumeration loop is the documented pathology). The
   decisive in-repo evidence is that `topology_enum.py` already does symmetry reduction
   *outside* the solver (enumerate-then-canonicalize) and leans on no solver symmetry feature
   at all — so the backend choice reduces to "which engine enumerates a finite-domain CSP with
   permutation symmetry most conveniently and reliably," and that is CP-SAT (native
   `enumerate_all_solutions`, a validated reference already in the tree). **The independent
   oracle is `itertools` brute force, not Z3** (see §4): a second *solver* re-encoding buys a
   differential check only if the IR→Z3 lowering is itself trusted, whereas `itertools` is
   trivially auditable and shares no code path with CP-SAT. The Z3-as-second-enumerator idea
   (Proposals 2 and 3) is recorded as a **filed deferral** (§9), not adopted: it adds a Z3
   lowering and the blocking-clause harness for a marginal gain over the brute oracle, and YAGNI
   here is a *measured* call (the spaces are small), not the scale-excuse ADR-0012 P7/P8 forbid.

2. **SSOT shape: a typed Python registry of frozen `HParam` descriptors** (one module), each
   pointing at the *real* config home rather than copying its default (ADR-0012 P1/P7). The
   descriptor algebra makes illegal configs unrepresentable (ADR-0000).

3. **DSL: an embedded Python eDSL** (the registry literals + a small constraint/symmetry
   mini-language), not an external grammar. Typed, `mypy --strict`-checkable (ADR-0012 P8); no
   new Port/ACL to re-validate (which would itself re-author the type system).

4. **IR: a backend-neutral `ConfigSpace`** (typed vars + reified constraints + a symbolic
   symmetry group + a projection set), with **two** lowerings consuming it: `to_cpsat` (the
   enumerator) and `to_grid` (the `itertools` oracle). The two-lowering design *is* the
   verification architecture, not an add-on — you cannot check CP-SAT against CP-SAT.

5. **Symmetry: enumerate-then-canonicalize** over a declared group (generalizing
   `topology_enum._canonical_key`), with **three** symmetry mechanisms — (a) trivially-correct
   in-model lex breaks for within-class orderings, (b) the post-hoc canonical-key orbit dedup
   for joint permutation groups, and (c) **`CanonInert`**, an IR node that pins a conditionally-
   inert HP to its default so inert flag combinations never multiply phantom configs. Mechanism
   (c) is the single most important generalization beyond the template, because the *dominant*
   symmetry in the overcommit surface is the activation-gate collapse, not the permutation
   group.

6. **Verification: two independent oracles, fail-loud (ADR-0002).** Oracle A (the
   `verify_orbits` pattern) catches under-collapse; Oracle B (the `to_grid` cross-check)
   catches over-collapse *and* any feasibility-encoding mismatch between the declarative CP-SAT
   model and the imperative filter. The template has only Oracle A; adding Oracle B is a genuine
   strengthening and it is cheap because the spaces are small.

7. **Selection: a typed `Target`** (surface and/or explicit HP subset and/or pinned values)
   that projects the SSOT to a sub-space and compiles only that. `Target(surface=OVERCOMMIT)`
   reproduces `overcommit_sweep`'s space; `Target(surface=TOPOLOGY)` must reproduce
   `topology_enum.py`'s output **bit-for-bit** (the migration acceptance test, §6).

Two non-negotiable rules that cut across all of the above:

- **Effects annotate; constraints prune.** A `MEASURED`/`HYPOTHESIZED`/`UNKNOWN` effect
  (ADR-0009) is *metadata*, never a feasibility filter. The compiler must never drop a
  candidate because a measured effect was "bad" (e.g. θ>0 "neutral-to-harmful at N=9" is
  regime-specific; pruning by it would bake an interpretation into the generative set — the
  `claims-measured-vs-interpreted` / `model-bound-is-conjecture-not-witness` error). `Effect`
  and `Constraint` are *different types* in the SSOT.
- **One home, derived not copied.** The SSOT's biggest execution risk (named by all three
  proposals) is that it becomes the (N+1)th re-author of a default that already lives in a C++
  struct or an argparse block. The descriptor's `home` is a `SourceRef`; agreement is
  *mechanically enforced* (§1.4). Copying a literal "for now" is the exact cancer (ADR-0012
  cancer B) this exercise exists to cure and is forbidden.

---

## 1. The SSOT — `throughput-lab/hp/spec.py` (the registry)

One Python module is the single home (ADR-0012 P1) of every throughput-affecting HP's
*metadata*. It does **not** re-author domains/defaults that have a code home; it references
the home and a drift check enforces agreement.

### 1.1 The descriptor algebra (make the illegal unrepresentable — ADR-0000)

```python
# Domains — a closed union; construction validates (default ∈ domain, lo ≤ hi, etc.; ADR-0002).
Domain = IntRange(lo, hi)            # contiguous int
       | IntSet([...])               # an explicit swept ladder (e.g. fibers {0,1,8,32,64,128,256})
       | FloatRange(lo, hi)
       | EnumSet[str]                # e.g. {"round-sync","greedy"}, {"strict-barrier","pipelined-bucket"}
       | Bool
       | Categorical([...])          # e.g. cpu-lists; bucket tuples (order-insensitive, stored sorted)
       | DerivedFrom(callable, *dep_names)   # e.g. K = ceil(pool_batch/pool_threads) — NEVER a free var

# Where the authoritative definition lives. The SSOT derives from this, never copies it.
SourceRef = CppField(file, symbol)       # a C++ struct field initializer (WireRunnerConfig::min_coalesce)
          | CppFlag(file, flag)          # a C++ --flag parse site with its inline default
          | PyArg(file, dest)            # an argparse default (e.g. inference_server / a harness)
          | PyField(file, symbol)        # a Python dataclass/ServerConfig field
          | NoCodeHome(reason)           # the ONLY case where a literal default is permitted in the SSOT

# Evidence — ADR-0009 made a construction-time type. MEASURED REQUIRES an evidence ref.
Effect = Measured(sign, note, evidence: EvidenceRef)   # sign ∈ {+,-,0}; evidence is mandatory
       | Hypothesized(rationale)
       | Unknown

# Symmetry class of this HP.
SymmetryClass = Free
              | Interchangeable(group_id)      # this HP is one of a permutable set (replicas, threads)
              | Asymmetric(reason)             # never permuted (housekeeping core 0)
              | OrderInsensitive               # a set/tuple HP; stored sorted (bucket set)

@dataclass(frozen=True)
class HParam:
    name: str                 # the ONE canonical key — uniqueness is enforced (P1)
    concept: str              # the cross-surface concept this binds (e.g. "coalesce_degree")
    surfaces: frozenset[Surface]
    kind: Kind                # PRODUCER_FLAG | SERVER_FLAG | SCHEDULING | BUILD | HARNESS_MEASURE
    home: SourceRef
    domain: Domain
    default: object | None    # None iff home is a SourceRef that supplies it (derived; see 1.4)
    symmetry: SymmetryClass
    effect: Effect
    activation: Guard | None  # the conditional-feature gate; None ⇒ ALWAYS live
    bindings: tuple[Binding, ...]   # per-surface flag name + parse home for ONE concept (1.3)
```

`Guard` is a predicate over *other HPs'* values, e.g. `eq("wire_mode","pipelined-bucket")`,
`and_(eq("wire_mode","pipelined-bucket"), is_true("chunk_floor"))`, `eq("driver","greedy")`.
Guards are the conditional-feature ("staged configuration") structure: a deselected parent's
children are not free dimensions.

### 1.2 Why typed Python, not YAML/TOML/a new grammar

An external grammar is a second home for the type system and a Port/ACL whose strict decoder
re-authors the domains — the very ADR-0012 P1 cancer. The embedded eDSL inherits `mypy
--strict` (P8's gate), dataclass construction-time validation (ADR-0002), and lets `home`
references *call into* the real homes (a build step reading a C++ struct default). The
maintainer authors specs; the consumers are Python harnesses; both favor typed Python.

### 1.3 One concept, many bindings (the deepest one-home payoff)

The inventory's cross-references are **one concept each**, with a per-surface binding — not one
fact per surface:

| concept | static-lab binding | overcommit binding | placement binding |
|---|---|---|---|
| `coalesce_degree` (producer rows/message) | `--msg-rows` | `min_coalesce` / synthetic `S` | — |
| `eval_shape_policy` (server eval shape) | `--max-batch` + warmup ladder | `e_policy` + `buckets` | — |
| `cpu_placement` (process → vCPU) | `taskset -c` per process | `server_core`/`producer_cores` | `topology_enum.Placement.cpus` |
| `fibers_per_thread` (concurrency width) | `--fibers` | `pool_batch` → K (derived) | — |
| `inflight_depth` / driver schedule | `--driver` (round-sync/greedy) | `wire_mode` (strict-barrier/pipelined-bucket) | — |
| `server_drain_wait` | server `--poll-timeout-ms` | `max_queue_delay_ms` | — |

The SSOT declares each concept once; selecting a surface picks the binding. `cpu_placement`'s
home is `topology_enum`'s typed `Placement` (cpus + policy + nice + latency_nice); the
overcommit `server_core`/`producer_cores` is a **degenerate binding** (taskset only,
sched-policy unset). This is also where the **producer-variant axis** lives: the real vs
synthetic producer disagree on the *same* logical HP (threads 3 vs 1; recv-timeout
10000-hardcoded vs 5000-flag). These are *named per-variant divergences* on a `variant`
binding axis, not silent disagreements — the drift check (1.4) understands the variant axis so
it does not false-positive on a legitimate per-variant default.

### 1.4 Drift enforcement — derive, don't copy (ADR-0012 P1/P7)

The descriptor's `default`/`domain` must agree with the cited `home`. ADR-0012 P7's enforcement
ladder, strongest first:

- **Strongest (extract / generate):** a build step parses the C++ struct/argparse defaults and
  populates the descriptor's `default`/`domain` so the SSOT *derives* from the one home. This is
  the P7-preferred "generate-or-compile-from-one-source."
- **Floor (build-time lint):** `tests/test_ssot_drift.py` reads each `SourceRef`'s cited line
  and asserts the descriptor's value equals it (modulo the named producer-variant axis),
  failing the build on drift. Same discipline as the cited `test_wire_drift.py`.

**Ship the lint floor first** (auditable today, no C++ parser to maintain); **file the
extractor as the P7-strongest follow-on** (§9). Naming this as a floor-with-filed-deferral is
the honest disposition ADR-0012 P7 demands — *not* a "for now" scale-excuse. The only HPs that
may carry a literal default in the SSOT are those with `home = NoCodeHome(reason)` (a genuine
runtime-only fact with no static line); those get a runtime parity backstop, the weakest P7
surface, named not buried.

### 1.5 Cross-HP structure — `throughput-lab/hp/relations.py`

Structure that is not per-HP (the topology orbit group, the 1:3 pin partition, the feasibility
predicates) lives in a sibling module as declarative objects, each carrying its **prose
rationale** (preserving `topology_enum`'s R1–R4 "every constraint carries its why"):

```python
SymmetryGroup(generators=[ReplicaPermutation(...), IsolatedCorePermutation(...)],
              anchors=["core0"], rationale="housekeeping core 0 is asymmetric (IRQ/RCU)")
Partition(whole="cores", parts=["server_core","producer_cores"], rationale="the 1:3 split")
Feasibility(expr="n_gens + 1 <= n_cores", rationale="server needs a core")
```

---

## 2. The DSL surface a human writes

Adding an HP = writing one `HParam(...)` literal (and, if it gates others, one `activation`
guard). Illustrative shape (final API is the implementer's, kept faithful to these types):

```python
COALESCE_DEGREE = HParam(
    name="coalesce_degree", concept="coalesce_degree",
    surfaces={Surface.STATIC_LAB, Surface.OVERCOMMIT}, kind=Kind.PRODUCER_FLAG,
    home=CppFlag("throughput-lab/cpp/real_producer.cpp", "--msg-rows"),
    domain=IntSet([1, 4, 16, 64, 128, 256, 512]), default=None,  # derived from home (=1)
    symmetry=Free,
    effect=Measured("+", "~2.9x, flat optimum 64-256",
                    EvidenceRef("docs/notes/tlab-performance-journey-2026-06-24.md", "1h/2")),
    activation=None,
    bindings=(
        Binding(Surface.STATIC_LAB, flag="--msg-rows",     home=CppFlag("real_producer.cpp", "--msg-rows")),
        Binding(Surface.OVERCOMMIT, flag="--min-coalesce", home=CppField("runner_wire_batched.hpp",
                "WireRunnerConfig::min_coalesce"),
                clamp=Clamp(lo=1, hi=Ref("fibers_per_thread"))),   # the [1,K] clamp, typed
    ),
)

CHUNK_FLOOR = HParam(
    name="chunk_floor", concept="chunk_floor", surfaces={Surface.OVERCOMMIT},
    kind=Kind.PRODUCER_FLAG, home=CppField("runner_wire_batched.hpp", "WireRunnerConfig::chunk_floor"),
    domain=Bool, default=None,                                 # derived (=false)
    symmetry=Free,
    activation=eq("wire_mode", "pipelined-bucket"),           # inert under strict-barrier
    effect=Measured("0", "winning region gen=ON & theta=0 at N=9 (regime-specific)",
                    EvidenceRef("cpp/stage_a/server_gen_floor_grid.py", "refine_configs")))
```

Note `default=None` everywhere a code home supplies it — the descriptor *cannot* silently
disagree with the struct because it carries no copy to disagree with (the extractor/lint fills
it). A `Measured` effect cannot be constructed without an `EvidenceRef` (ADR-0009 as a type).

---

## 3. The compiler IR — `throughput-lab/hp/ir.py`

A deliberately small, backend-neutral intermediate between the selected SSOT descriptors and a
solver invocation. A typed dataclass tree (not a string format).

```python
IRVar = Var(id: str, kind: Int|Bool|EnumIdx, lo: int, hi: int)   # enums lowered to 0..n-1

IRConstr = Linear(coeffs: dict[str,int], op: "=="|"!="|"<="|">="|"<", rhs: int)
         | Reify(boolvar: str, body: IRConstr)          # CP-SAT only_enforce_if
         | Implies(lit: str, body: IRConstr)
         | AllDifferent(vars: list[str])
         | MaxEquality(target: str, members: list[str]) # occupancy booleans (R2/R3)
         | Clamp(var: str, lo: Expr, hi: Expr)          # min_coalesce ∈ [1, K]
         | Derived(var: str, expr: Expr)                # K = ceil(pool_batch/pool_threads); never free
         | CanonInert(var: str, default_value: int, when: lit)  # pin inert HP to default (THE key node)
         | Table(vars: list[str], allowed: list[tuple]) # escape hatch for awkward feasibility

IRSym = SymmetryGroup(generators: list[Permutation],   # acts on var-tuples by value
                      anchors: list[str])              # vars never permuted (core 0)

@dataclass(frozen=True)
class ConfigSpace:
    vars: list[IRVar]
    constrs: list[IRConstr]
    sym: IRSym
    projection: list[str]      # the CONFIG-DEFINING vars; aux reifications are NOT distinct-config keys
    provenance: dict           # name -> Effect, carried through for the candidate-set ledger
```

Three IR commitments, each falsifiable:

- **Everything lowers to finite-domain integers.** Enums → indices (as `topology_enum` already
  does: `server_pol ∈ 0..1`). The whole HP space is finite-domain after `IntSet` ladders are
  materialized; nothing needs real arithmetic or quantifiers. *Falsifier:* one throughput HP
  whose feasible set genuinely needs nonlinear/quantified/real constraints. None found in the
  inventory (the only floats — `secs`, `delay_ms`, `rate` — are measurement/operating-point
  knobs with no inter-HP arithmetic constraint). This is also *why SMT's expressive power is
  unused* and CP-SAT is the right fit.
- **`projection` is explicit and load-bearing.** Both the solver and the oracle would otherwise
  multiply solutions by internal reification vars (`*_on[c]`, `surplus_eq_server`). The IR names
  the config-defining vars; enumeration is deduped onto the projection.
- **`CanonInert` is in the IR, verified, not a post-filter.** It is *both* a feasibility-
  correctness mechanism (an inert flag set to a non-default is not a real config) and the
  dominant symmetry-reduction mechanism (under strict-barrier, all `{D,N,S_min,chunk_floor}`
  combos collapse to one). Putting it in the IR means the oracle (§4) verifies the collapse
  actually happened.

`Derived` and `Clamp` encode the inventory's "K is derived, never a free flag" and the `[1,K]`
clamp structurally — the solver can never enumerate a `K` that disagrees with its definition.

---

## 4. Symmetry reduction and the two independent oracles

Adopt `topology_enum`'s proven pattern wholesale, generalized; add the second oracle.

### 4.1 Three symmetry mechanisms

1. **In-model lex break — only for trivially-correct within-class orderings.** Keep
   `topology_enum`'s S1 (generator index strictly increasing on `core*K+pol`; distinct gen
   cores make it always-satisfiable and it shrinks the *raw* enumerated set). Generalize to any
   `Interchangeable` set whose ordering is trivially sound, and to `OrderInsensitive` set HPs
   (the bucket tuple stored sorted). **Do NOT** add joint-group lex-leader constraints in-model
   — that is exactly where naive symmetry breaking silently under/over-collapses, and the
   template's comment (and the SMT lens's literature) confirm it.
2. **Enumerate-then-canonicalize — for the joint permutation group.** Generalize
   `_canonical_key`: map each emitted projection to the lex-min image under the declared
   `IRSym`. The canonical key *is* the orbit invariant, so the reduction cannot under- or
   over-collapse *by construction, for any group*. The canonicalizer is built as a
   **composition of per-factor canonicalizers** (sort the order-insensitive set; sort
   interchangeable replicas by packed key; relabel isolated cores by the lex-min permutation),
   applied factor-by-factor — polynomial per factor, avoiding `topology_enum`'s O(k!) full-
   product brute force at larger scale. **Caveat (load-bearing):** per-factor composition is
   sound only if the factor groups act on *disjoint* variable sets and commute; if two factors
   interact (a replica permutation that also permutes cores), per-factor canonicalization can
   under-collapse. Oracle B (§4.2) is the backstop that catches exactly this, which is why the
   verification layer must be built *before* the reduction is trusted.
3. **`CanonInert` — the conditional-feature collapse.** Pin each inert HP to its default when
   its guard is unsatisfied, in-IR, so the inert combinations are never enumerated. This is the
   *largest* symmetry in the overcommit surface.

### 4.2 Two oracles, fail-loud (ADR-0002), run under `--verify`

- **Oracle A — orbit non-isomorphism (the `verify_orbits` pattern, generalized).** Brute-force
  recompute each emitted config's orbit under `IRSym` *without trusting the canonical-key
  function*; assert the emitted set is pairwise non-isomorphic and orbit-disjoint. Catches
  under-collapse (two emitted configs in one orbit). This is what `topology_enum` already does.
- **Oracle B — the grid cross-check (`to_grid`, the genuine strengthening).** Independently
  generate the *full* feasible set via `itertools.product` + an imperative feasibility filter
  (a different implementation of the same `constrs`), quotient *it* by the same canonicalizer,
  and assert the canonical-rep set is **identical** to CP-SAT's emitted set. Catches (i)
  over-collapse — CP-SAT emitted *fewer* than the true orbit count, a real candidate silently
  dropped; (ii) the per-factor-canonicalizer interaction bug from §4.1; (iii) any mismatch
  between the declarative CP-SAT encoding and the imperative filter. `topology_enum` lacks this.
  Cheap because the spaces are small.
- **Inertness self-check.** Assert that configs differing only in a guard-false axis
  canonicalize to the same key — i.e. the `CanonInert` collapses the inventory specifies
  (strict-barrier eats D/N/S_min/chunk_floor; chunk_floor=0 eats D/S_min) actually happen. A
  symmetry the SSOT *generates from the activation types*, verified, not narrated.

Any divergence on any oracle ⇒ exit-nonzero, refuse-to-emit, naming the offending config(s) —
exactly `topology_enum`'s `--verify` returning 3.

---

## 5. The selection affordance — `throughput-lab/hp/compile.py`

```python
Target = { surfaces: set[Surface] | None,        # None ⇒ all
           include:  set[str] | ALL,              # explicit HP names
           pin:      dict[str, value],            # finalize some axes (staged configuration)
           variant:  str | None }                 # real | synthetic (the producer-variant axis)

compile(spec, target) -> ConfigSpace
```

`compile` selects the descriptors whose `surfaces` intersect the target (or are explicitly
included) plus the `relations.py` entries whose referenced vars are all in scope, resolves
per-surface/per-variant bindings, applies pins as IR equalities, projects away the rest, and
lowers to `ConfigSpace`. Worked selections (each a regression test):

- `Target(surfaces={OVERCOMMIT})` → the `overcommit_sweep` space (`wire_mode × N × D × S_min ×
  chunk_floor × θ × placement`), with `CanonInert` killing the strict-barrier and chunk_floor=0
  phantoms for free.
- `Target(surfaces={TOPOLOGY})` → **must reproduce `topology_enum.py --gens 3 --cores 4`
  bit-for-bit** (the migration acceptance test, §6).
- `Target(surfaces={STATIC_LAB, TOPOLOGY})` → composed; because PINNING/COALESCING/
  BUCKETING/CONCURRENCY are one concept each, the composition does not double-count.
- `Target(surfaces={STATIC_LAB}, pin={"episodic": True, "n_sims": 256, "m": 24})` → the
  production-shape episodic sub-space.

**Fail-loud selection rule (ADR-0002, the selection register):** a `Target` that names an HP
whose `activation` depends on an *unselected* HP must **refuse**, not silently pin — e.g.
selecting `min_coalesce` without `chunk_floor` in scope ("min_coalesce is inert unless
chunk_floor is selected and =1; include it or acknowledge the pin"). Silently emitting a
one-value axis the user thought was live is a config the receiver cannot honor.

---

## 6. Migration — hoist `topology_enum` into the SSOT, safely

The brief mandates the topology config space be hoisted INTO the SSOT (it is one HP surface,
not a separate thing). Do it as a *verified* refactor (ADR-0004 minimal-touch, ADR-0013 verify-
the-artifact):

1. Re-express `topology_enum`'s `SchedPolicy`, `Placement`, the per-role policy vocabularies,
   and R1–R4 (with their rationale) as SSOT `HParam`s under `kind=SCHEDULING` + a
   `PlacementConstraints` block in `relations.py`.
2. Implement `compile(Target(surfaces={TOPOLOGY}))` → `to_cpsat` → enumerate → canonicalize.
3. **Acceptance gate:** assert the emitted canonical set equals the standalone
   `topology_enum.py` output exactly (same N configs, same `config_id`s). Until they agree, do
   **not** replace the original. A divergence means the hoist changed the space — a P1
   violation — and fails loud.
4. Only after the gate passes, refactor `topology_enum.py` to *consume* the SSOT (it stops
   being a second author). Keep `verify_orbits` as Oracle A.

---

## 7. Component list (the modules the implementer creates)

All under `throughput-lab/hp/`, each with an ADR-0006 header (path + purpose + Public Domain):

| module | responsibility |
|---|---|
| `spec.py` | the SSOT registry: `HParam` descriptors + the descriptor algebra (Domain/SourceRef/Effect/SymmetryClass/Guard/Binding). The single home. |
| `relations.py` | cross-HP structure: `SymmetryGroup`, `Partition`, `Feasibility`, `PlacementConstraints` (R1–R4 with rationale). |
| `ir.py` | the backend-neutral `ConfigSpace` / `IRVar` / `IRConstr` / `IRSym`. |
| `compile.py` | `Target` + `compile(spec, target) -> ConfigSpace`; selection, binding/variant resolution, projection. |
| `backends/cpsat.py` | `to_cpsat(ConfigSpace) -> (CpModel, var_handles)` + the `enumerate_all_solutions` driver + the canonicalizer (composition of per-factor canonicalizers). |
| `backends/grid.py` | `to_grid(ConfigSpace) -> Iterator[dict]` — the `itertools` oracle (Oracle B), a *different implementation* of the same constraints. |
| `verify.py` | Oracle A (orbit non-isomorphism, generalized `verify_orbits`), Oracle B (grid cross-check), the inertness self-check; all fail-loud. |
| `cli.py` | `python -m throughput_lab.hp --select <surface|subset> [--pin ...] [--variant ...] --verify --json out.json` — writes the candidate set + the provenance ledger (which axes are measured/hypothesized/unknown). |
| `tests/test_ssot_drift.py` | the build-time lint (§1.4): every descriptor `default`/`domain` agrees with its `SourceRef`. |
| `tests/test_topology_parity.py` | the migration acceptance gate (§6): SSOT TOPOLOGY == standalone `topology_enum.py`. |
| `tests/test_oracles.py` | both oracles pass on every shipped `Target`; an injected fault fails loud. |

`DESIGN.md` (this file) is the eleventh component.

---

## 8. Build plan (step-by-step; each step is a coherent, committable checkpoint)

Per the maintainer's phase-checkpoint discipline, each step is its own coherent phase — do not
mix phases. **The verification layer is built before the reduction is trusted** (§4.1 caveat).

1. **Skeleton + types (no solver).** `spec.py` descriptor algebra, `ir.py`, `relations.py`
   types, with ADR-0006 headers and `mypy --strict` clean. No HPs declared yet. *Gate:* mypy
   passes; the types reject an illegal descriptor (default ∉ domain) at construction.
2. **Declare the TOPOLOGY surface only.** Re-express `topology_enum`'s placement space as SSOT
   descriptors + `PlacementConstraints`. *Gate:* `compile(Target(TOPOLOGY))` produces a
   `ConfigSpace`.
3. **`to_grid` + `to_cpsat` + canonicalizer for TOPOLOGY.** *Gate:* the migration acceptance
   test (§6) — SSOT TOPOLOGY canonical set == standalone `topology_enum.py` bit-for-bit. Build
   Oracle A + Oracle B here and prove both pass on TOPOLOGY *before* trusting the canonicalizer
   anywhere else.
4. **`CanonInert` + the activation machinery.** Add guards; verify the inertness self-check on a
   tiny synthetic surface (two gated flags) where the collapse is hand-countable. *Gate:* the
   inertness self-check passes; Oracle B confirms no over-collapse.
5. **Declare the OVERCOMMIT surface.** All surface-2 HPs (`wire_mode`, `trees_per_thread`,
   `max_inflight_msgs`, `min_coalesce`, `chunk_floor`, `pool_threads`/`pool_batch`→K,
   `min_forward_rows`, the 1:3 partition), with `home` refs and effect annotations transcribed
   from the inventory (marked measured/hypothesized/unknown faithfully). *Gate:*
   `compile(Target(OVERCOMMIT))` enumerates; both oracles pass; the strict-barrier and
   chunk_floor=0 collapses are present.
6. **Declare the STATIC_LAB surface** (producer + server + scheduling + build HPs), with the
   producer-`variant` axis. *Gate:* both oracles pass; `Target(STATIC_LAB, variant=real)`
   resolves the real/synthetic divergences correctly.
7. **The drift lint** (`test_ssot_drift.py`). *Gate:* it reads each `SourceRef` line and asserts
   agreement (variant-aware); it fails loud on an injected drift.
8. **The CLI + provenance ledger.** `python -m throughput_lab.hp ...`. *Gate:* emits the
   candidate set JSON + the measured/hypothesized/unknown ledger; `--verify` returns nonzero on
   an injected oracle fault.
9. **Refactor `topology_enum.py` to consume the SSOT** (it stops re-authoring the space). *Gate:*
   the parity test still passes; the standalone entry point still works (or is repointed).
10. **Documentation pass** (ADR-0005): update `docs/STATUS.md` if this changes an orientation
    surface; record any ADR "Revisit when…" trigger this fires (notably ADR-0012's self-
    application list — this is a P1 consolidation); retrofit ADR-0006 headers on every touched
    file. Code-only delivery with doc implications is incomplete.

---

## 9. Filed deferrals (named, not buried — ADR-0012 P7/P8 forbid the "for now" excuse)

- **The C++/argparse default *extractor*** (the P7-strongest "generate-from-one-source"). v1
  ships the build-time *lint* floor (§1.4); the extractor is the strict upgrade. Named here so
  settling for the lint is a *disclosed* floor, not a silent stopping point.
- **Z3 as a second differential enumerator.** Recorded by Proposals 2 and 3; declined for v1
  (§0.1) because the `itertools` Oracle B already brackets the failure modes and a Z3 lowering +
  blocking-clause harness is engineering for a marginal gain at this scale. *Revisit trigger:*
  if a future surface grows genuinely **nonlinear or quantified** constraints (none today), Z3
  becomes the right escape hatch — the IR is backend-neutral precisely so this swap is local.
- **A throughput *model* for ranking** (`dps ≈ f(K, M, θ, …)`) to order the candidate set.
  Explicitly out of scope: the brief is *enumeration* of the feasible, symmetry-reduced space,
  not optimization. A model would move this into OMT territory (where SMT/MILP + symmetry
  breaking is the literature's choice) and would risk pruning by *interpreted* effects — which
  §0's "effects annotate; constraints prune" rule forbids. The provenance ledger (effects as
  metadata) is the sanctioned, honest substitute.

---

## 10. The falsifiable claims this design rests on

| # | claim | falsifier |
|---|---|---|
| 1 | the whole HP space is finite-domain; SMT's theory power is unused → CP-SAT is the fit | one throughput HP needs nonlinear/quantified/real constraints |
| 2 | `itertools` Oracle B brackets the failure modes; a second *solver* oracle adds little at this scale | Oracle B misses a real over/under-collapse a Z3 cross-check would catch |
| 3 | the SSOT references homes (no copied defaults) → cancer B unrepresentable | a default appears as a bare literal in the SSOT that also lives in a struct/argparse with no derivation linking them |
| 4 | per-factor canonicalizer composition is sound | two symmetry factors interact (shared var set) → Oracle B reports over-collapse |
| 5 | `CanonInert` collapses the inert phantoms correctly | the inertness self-check / Oracle B finds it merging genuinely-distinct configs (an "inert" flag is not actually inert) |
| 6 | hoisted TOPOLOGY == standalone `topology_enum.py` | the parity gate (§6) diverges → the hoist changed the space (P1 violation) |
| 7 | the spaces are small enough for CP-SAT to enumerate comfortably | a single selection exceeds ~minutes / ~10^5 configs after symmetry reduction on the 4-vCPU box |

Claims 1–7 are **conjectures until the implementation runs them** (per
`model-bound-is-conjecture-not-witness`): the operational witness is the compiler enumerating
each `Target`, both oracles agreeing, and the parity gate passing. None of these is to be
recorded as a proven fact before that witness exists.

---

## 11. Amendment 2026-06-25 — STATIC_LAB populated, the `OperatingPoint` concept, the `CppFlag` lint

*(ADR-0005 Rule 8 amend-by-append; the §0–§10 body above is the adjudicated point-in-time record
and is left intact. This section records SSOT changes landed 2026-06-25; commits `8de75c6`,
`74afccf`, `6ab0113`. ADR-0012 P1 is the law these descend from.)*

### 11.1 STATIC_LAB is now POPULATED (closes the §8 step-6 / §9 follow-on deferral)

The STATIC_LAB surface — scaffolded but empty in the §0–§10 body — now declares **9 `HParam`
descriptors** in `spec.py::_STATIC_LAB`: `fibers`, `msg_rows`, `inflight_msgs`, `driver`,
`seconds`, `n_sims`, `m`, `max_batch`, `warmup_ladder`. Each is homed on its **real code default**
(not a copied literal), per the §1.4 one-home rule:

- the producer flags (`fibers`/`msg_rows`/`inflight_msgs`/`driver`/`seconds`) home on
  `throughput-lab/cpp/real_producer.cpp` via `CppFlag`;
- `n_sims`/`m` home on `cpp/include/chocofarm/gumbel.hpp` `GumbelConfig` via `CppField`;
- `max_batch` homes on `throughput-lab/server/__main__.py` argparse via `PyArg`;
- `warmup_ladder` is the one `NoCodeHome` list literal (its list-valued extractor is the deferral
  named in 11.3 / §9).

All nine carry an `Effect` (`Measured` with an `EvidenceRef`, or `Hypothesized`/`Unknown`),
preserving §0's "effects annotate; constraints prune." The eight scalar homes are drift-linted
(11.3).

### 11.2 NEW concept — `OperatingPoint` (a selected point, distinct from a sweep-axis `HParam`)

The §0–§10 body modelled only **axes** (`HParam`: a domain the compiler enumerates over). 2026-06-25
adds a second, orthogonal SSOT concept: an **`OperatingPoint`** — a *selected* point **within** an
enumerated surface (the tuned, banked winner of a sweep). It is deliberately **not** an `HParam`:
forcing a banked joint config onto a surface axis would add an enumeration variable and break the
§6 bit-for-bit parity gate. It is the **`pool_threads`/`pool_batch` pattern** generalized — a tuned
value diverging from the code default — and is owned by the SSOT as a `NoCodeHome` literal (the
sanctioned literal-with-reason of §1.4).

Two instances are declared in `spec.py`:

- **`BANKED_TOPOLOGY`** — a `config_id` *into* the TOPOLOGY surface. Resolved to placements by
  `harness/topology_enum.py::config_by_id`, which **validates membership against the live
  enumeration** (ADR-0002 fail-loud on an unknown id). Replaces the hand-pinned `taskset` literals
  formerly smeared across `episodic_dps.sh`. Its effect is recorded honestly as a **low-regret**
  adoption (server off the housekeeping core 0: +0.68%, paired one-sided p=0.045, bootstrap CI
  straddling 0; `tlab_finding` #20, status provisional), **not** a clean win.
- **`BANKED_STATIC`** — the tuned producer/server scalars (`fibers=1024`/`msg_rows=256`/
  `driver=greedy`/`max_batch=256`/`seconds=10`/…). Accessors: `banked_static()` returns the dict
  **validated against each value's `HParam` domain** at call time (ADR-0000: a banked value outside
  its axis domain is a loud error, not a silent bad default); `banked_static_env()` emits the
  banked point as shell `eval`-able `BANKED_*` assignments.

**One home for the banked launch shape.** Harnesses derive their **defaults** from these emitters
(`hp.cli --banked-static-env`; `harness/topology_enum.py --banked-env` for the topology side), with
override args still winning for sweeps. Because the banked `--seconds=10` now has a single home, the
per-harness run-length drift that confounded a run-length comparison (`tlab_finding` #21 — episodic
ran 14s, topology_sweep ran 5/10s) **can no longer recur**: it is derived, not re-authored.

### 11.3 The drift lint gained a `CppFlag` extractor (§9 follow-on statuses, by dated append)

`tests/test_ssot_drift.py` gained `_cpp_flag_default`, which parses the producer's
`opt(args,"--flag") ? <conv> : DEFAULT` ternary (the `' : '` is space-delimited, so it never
collides with a C++ `::` in the true-branch conversion) and recovers int/double/string/bool. The
producer `CppFlag` homes (the five STATIC_LAB producer flags) are therefore now **drift-guarded**,
alongside the existing `CppField`/`PyArg`/`PyField` coverage. Only the `warmup_ladder` **list-valued**
literal remains un-extracted.

**§9 deferral statuses (amended by append; the §9 list above is left as its point-in-time record):**

- *The STATIC_LAB follow-on (the §8 step-6 surface)* → **DONE** 2026-06-25 (11.1).
- *The `CppFlag` producer-flag lint extractor* → **DONE** 2026-06-25 (this section).
- *The P7-strongest "generate-from-one-source" extractor* → **still DEFERRED** (the lint *checks*
  drift; it does not yet *generate* descriptors from source — §1.4 / §9 unchanged).
- *The list-valued home extractor* (the `warmup_ladder` ladder literal) → **still DEFERRED** (the
  named list-valued deferral; the ladder homes as a `NoCodeHome` literal until it exists).
