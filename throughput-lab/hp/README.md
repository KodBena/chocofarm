<!--
throughput-lab/hp/README.md — the inspectable surface for the throughput-lab hyperparameter
SSOT, its embedded DSL, and the symmetry-reduced config-space compiler ("hpdsl"). What it is,
how to run it, how to add a hyperparameter under the one-home rule, how selection works, and an
honest WHAT WORKS / WHAT DOES NOT / WHY folding in the adversarial verdicts (failing checks are
recorded as known limitations, not hidden).

Companion documents: DESIGN.md (the adjudicated design spec — the contract this implements).
Authoritative homes are the code files each HParam descriptor points at, never this README.

Public Domain (The Unlicense).
-->

# `hp/` — a single-homed HP SSOT and its config-space compiler

## What this is

The throughput lab has many hyperparameters that affect throughput — especially binary flags —
scattered across C++ `--flag` parsers, Python argparse blocks, shell harnesses, and
scheduling/affinity choices. This package is the **single surface (SSOT)** that documents every
throughput-affecting hyperparameter once, and a **compiler** that turns that surface into the
*feasible, symmetry-reduced configuration space* a constraint solver enumerates — generalizing
what `harness/topology_enum.py` already did for the process/scheduling topology to the whole HP
space.

The point (ADR-0012 P1, one-home): "try this flag and that flag" becomes a generative, auditable
candidate set, and a single place from which "what works, what doesn't, and why" can be read off
(the provenance ledger).

Three layers:

1. **The SSOT (`spec.py`)** — a typed Python registry of frozen `HParam` descriptors. Each
   descriptor points at the *real* config home (a C++ struct field, an argparse default) rather
   than copying its value. The descriptor algebra makes an illegal config unrepresentable
   (ADR-0000): a `Measured` effect cannot be constructed without an evidence reference (ADR-0009);
   a default outside its domain is rejected at construction (ADR-0002).
2. **The DSL** — the registry literals plus a small constraint/symmetry mini-language
   (`relations.py`, `ir.py`). It is an *embedded* Python eDSL, not an external grammar, so it
   inherits Python typing and adds no second Port/ACL that would re-author the type system.
3. **The compiler (`compile.py` + `backends/`)** — `compile(spec, target) -> ConfigSpace`, then
   two independent lowerings of that one IR: `to_cpsat` (the enumerator) and `to_grid` (the
   `itertools` oracle). The two-lowering design *is* the verification architecture: you cannot
   check CP-SAT against CP-SAT.

### Backend decision (and why)

**CP-SAT is the sole solver backend; the second "backend" is an `itertools` brute-force oracle.
Z3/SMT is deliberately declined for v1.** This was an adjudicated call across three independent
design proposals (DESIGN.md §0), not a tooling gap — z3 4.16.0 *is* importable in the venv. The
reasons, substantiated against the actual compiled IR:

- The whole HP space is **finite-domain integer** after the `IntSet` ladders are materialized —
  every IR var is `Int`/`Bool`/`EnumIdx`, every constraint is linear/reify/all-different/table/
  derived/clamp/canon-inert. SMT's theory power (reals, quantifiers, nonlinear) is genuinely
  unused, so it buys nothing here.
- Symmetry reduction is done **outside** the solver (enumerate-then-canonicalize), exactly as
  `topology_enum.py` does — neither CP-SAT nor Z3 gives free symmetry reduction, so the backend
  choice reduces to "which engine enumerates a finite-domain CSP most conveniently," and that is
  CP-SAT (native `enumerate_all_solutions`, a validated reference already in the tree).
- The independent oracle is `itertools`, not a second solver: it shares no code path with CP-SAT
  and is trivially auditable, whereas a Z3 differential check is only as trustworthy as its own
  IR→Z3 lowering.

Z3 as a second differential enumerator is a **filed deferral** (DESIGN.md §9), to be revisited
only if a future surface grows genuinely nonlinear/quantified constraints — the IR is
backend-neutral precisely so that swap is local.

---

## How to run it

Interpreter (has ortools 9.15.6755 and z3 4.16.0):
`/home/bork/w/vdc/venvs/generic/bin/python`. Run from the lab root with `PYTHONPATH=.`:

```bash
cd /home/bork/w/vdc/1/chocofarm-hpdsl/throughput-lab
PY=/home/bork/w/vdc/venvs/generic/bin/python

# Enumerate the topology surface (the hoisted topology_enum space) and self-verify both oracles.
PYTHONPATH=. $PY -m hp.cli --select topology --gens 3 --cores 4 --verify
#   -> 40 feasible configs; Oracle A 40/40, Oracle B cpsat=40 grid=40, inertness 2 nodes 0 viol; exit 0

# Enumerate the older overcommit_sweep parameter region through the SAME compiler, self-verify.
PYTHONPATH=. $PY -m hp.cli --select overcommit --verify
#   -> 96 feasible configs; Oracle A 96/96, Oracle B cpsat=96 grid=96, inertness 4 nodes 0 viol; exit 0

# Write the candidate set + provenance ledger to JSON (no console table).
PYTHONPATH=. $PY -m hp.cli --select overcommit --json /tmp/candidates.json --no-table

# Pin some axes (staged configuration); compose surfaces with a comma.
PYTHONPATH=. $PY -m hp.cli --select topology,overcommit --no-table

# `python -m hp` is equivalent to `python -m hp.cli` (the package __main__ delegates).
```

The full test suite (run from the repo root, note the PYTHONPATH):

```bash
cd /home/bork/w/vdc/1/chocofarm-hpdsl
PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python -m pytest throughput-lab/hp/tests/ -q
#   -> 31 passed
```

`--verify` runs both oracles plus the inertness self-check and **returns non-zero (exit 3),
refusing to emit, on any divergence** — mirroring `topology_enum.py --verify` returning 3
(ADR-0002, fail-loud). This is real, not theater (see WHAT WORKS below).

---

## How to add a hyperparameter (one home)

You add **one `HParam(...)` literal** to `spec.py` (and, if it gates other HPs, one `activation`
guard). You do **not** copy a default: set `default=None` and point `home` at the real config
site; the value is derived from / checked against that home. Shape:

```python
HParam(
    name="coalesce_degree", concept="coalesce_degree",
    surfaces=frozenset({Surface.STATIC_LAB, Surface.OVERCOMMIT}), kind=Kind.PRODUCER_FLAG,
    home=CppFlag("throughput-lab/cpp/real_producer.cpp", "--msg-rows"),
    domain=IntSet([1, 4, 16, 64, 128, 256, 512]), default=None,   # derived from the home (=1)
    symmetry=Free,
    effect=Measured("+", "~2.9x, flat optimum 64-256",
                    EvidenceRef("docs/notes/tlab-performance-journey-2026-06-24.md", "1h/2")),
    activation=None,                          # None => always live; else a Guard over other HPs
    bindings=(...),                           # one concept, per-surface flag-name + parse home
)
```

Rules the types enforce for you:

- **`default=None` whenever a code home supplies the value.** The descriptor carries no copy, so
  it *cannot* silently disagree with the struct/argparse home. The only HPs permitted a literal
  default are those whose `home` is `NoCodeHome(reason)` (a genuine runtime-only fact).
- **`Measured(...)` requires an `EvidenceRef`** (ADR-0009 as a construction-time type). Use
  `Hypothesized(rationale)` or `Unknown()` when there is no isolating measurement. **Effects
  annotate; constraints prune** — an effect is metadata that flows into the provenance ledger, it
  is *never* a feasibility filter (a "neutral-to-harmful at N=9" effect is regime-specific; pruning
  by it would bake an interpretation into the generative set).
- **One concept, many bindings.** Cross-surface duplicates (e.g. producer rows/message =
  `--msg-rows` in the static lab = `min_coalesce`/synthetic `S` in overcommit) are ONE `HParam`
  with a per-surface `Binding`, not one fact per surface.
- **Inter-HP structure goes in `relations.py`** (the topology orbit group, the 1:3 pin partition,
  feasibility predicates), each carrying its prose rationale.

The drift lint (`tests/test_ssot_drift.py`) reads each descriptor's cited home line and asserts
agreement, failing the build on drift. Caveats on its current coverage are in WHAT DOES NOT below.

---

## How selection works

`compile(spec, Target)` projects the SSOT to a sub-space and compiles only that:

```python
Target = { surfaces: frozenset[Surface] | None,   # None => all
           include:  frozenset[str] | None,        # explicit HP names to add
           pin:      dict[str, value],             # finalize some axes
           variant:  str | None,                   # real | synthetic (producer-variant axis)
           topo:     TopologyParams }              # n_cores / n_gens / housekeeping_core
```

CLI: `--select <surface[,surface]>`, `--include name,name`, `--pin k=v` (repeatable),
`--variant`, `--gens/--cores/--housekeeping-core`.

The three surfaces:

- `topology` — the process/scheduling topology hoisted from `topology_enum.py` (server/gen/
  surplus scheduling-policy vocabulary, surplus presence, the R1–R4 placement constraints, the
  isolated-core × generator permutation symmetry).
- `overcommit` — the older `cpp/stage_a/overcommit_sweep.py` parameter region (`wire_mode × N ×
  S_min × D × chunk_floor × θ`, with the 1:3 pin and the `min_coalesce ≤ K` clamp).
- `static_lab` — the throughput-lab producer + server + scheduling + build HPs. **Scaffolded
  only** (see below): it selects and verifies but compiles to 1 config because no HParams are
  declared on it yet.

**Fail-loud selection rule (ADR-0002):** a `Target` naming an HP whose `activation` depends on an
*unselected* HP is **refused** (exit 2), not silently pinned — selecting `min_coalesce` without
`chunk_floor`/`wire_mode` in scope returns "selection refused", because emitting a one-value axis
the user thought was live is a config the receiver cannot honor.

---

## WHAT WORKS, WHAT DOES NOT, AND WHY

This section folds in the five adversarial verdicts verbatim in substance. Four passed; one
failed (severity major) and is recorded here as a known limitation, not hidden.

### What works (independently verified by running the artifacts)

1. **Reproduces `topology_enum.py` bit-for-bit on the default region (PASS).** Both tools emit
   exactly **40** configs at 4 cores / 3 gens, and a set comparison of the two JSON artifacts is
   an EXACT match on `config_id`s, tags, AND placements (cpus/policy/nice/latency_nice). Both
   self-verifiers are green. This is the migration acceptance gate, satisfied.
   - *Nuance the maintainer should know (not a failure):* `config_id` equality is **not**
     orbit-invariant. At 4c/3g the two tools happen to pick the same orbit representative, so even
     raw `config_id`s match. At other regions (verified 4c/2g, 5c/3g) the **counts** still match
     (56 == 56) but the two tools choose *different representatives* of the *same orbit partition*,
     so a naive `config_id`-set diff would falsely report divergence. The contract that holds
     universally is **orbit-set equality** (confirmed at 4c/3g, 4c/2g, 5c/3g with no
     under-collapse); at 4c/3g the stronger id-bit-equality also holds. Anything downstream that
     asserts `config_id`-set equality *off* the 4c/3g default will spuriously fail; assert orbit-set
     equality instead.

2. **Self-verification is real, fail-loud, not theater (PASS).** There are two genuinely
   independent enumerators — `backends/cpsat.py` (ortools `CpModel`, `enumerate_all_solutions`)
   and `backends/grid.py` (imperative `itertools.product` + a fixpoint feasibility filter, sharing
   no code path with CP-SAT). Every perturbation that genuinely changed the feasible/canonical set
   screamed and refused to emit:
   - a bogus `server_core==1` → Oracle B "cpsat=24 grid=40 → over-collapse".
   - an identity canonical key (no symmetry reduction) → Oracle A "128 configs, 40 distinct
     orbits, 88 under-collapse collisions"; the CLI returned **exit 3** with "refusing to emit
     (ADR-0002)".
   - dropping `CanonInert` (topology and overcommit) → Oracle B + inertness self-check both fired.
   - dropping the R4 surplus co-location *where it binds* (gens=2/cores=4) → Oracle B "24 configs
     CP-SAT emitted but the grid rejects".
   - Two perturbations stayed silent and were *measured* to be exact no-ops (delta=0 raw configs,
     the dropped constraint redundant in that topology); re-running the same drop on a topology
     where the constraint binds confirmed the oracle is not blind to that class.
   - *Disclosed residual (minor, named in DESIGN.md §4.1):* both oracles route through the SAME
     `cpsat.canonical_key`, so a canonicalizer that *over-collapses a symmetric surface* would pass
     both silently (demonstrated by injection: forcing `server_pol=0` collapsed 40→20 with
     `ok=True`). This is mitigated where it matters: the only symmetric surface (TOPOLOGY) is
     guarded by the independent `topology_enum.py` parity referee (`test_topology_parity.py`), and
     the only other enumerated surface (OVERCOMMIT) has an identity canonicalizer where the gap
     cannot exist. The honest gap-closer (Z3 as a second differential enumerator) is the filed
     §9 deferral.

3. **Selection affordance: the overcommit region compiles through the same compiler and the
   configs are feasible (PASS).** `--select overcommit --verify` emits **96** symmetry-reduced
   feasible configs (both oracles green). Feasibility against the surface's modeled constraints was
   confirmed four ways: the dual-oracle `--verify`, an injected-fault test (oracles fail loud), a
   from-scratch hand-count = 96 sharing no compiler code, and direct constraint checks —
   `min_coalesce ≤ K=ceil(64/3)=22` (0 violations; 32/64/128 dropped by the clamp), `pool_threads=3`
   everywhere (the 1:3 / one-core-per-generator pin), `trees ∈ {1,2,3}` with the documented N=4
   stall-bug correctly absent, and the strict-barrier / chunk_floor=0 inert collapses witnessed.
   - *Scope caveat (minor):* "top bucket ≤ max-batch", named in the task as an example constraint,
     is **not modeled** on this surface — buckets/max_batch are *fixed* (non-swept) operating points
     in `overcommit_sweep.py` (BUCKETS=(64,256,512), `--max-batch` default 512), so the invariant
     512≤512 holds trivially and is outside the enumerable region. The DSL models swept axes only;
     it deliberately does not represent fixed-invariant axes — worth knowing, not a feasibility
     violation.

4. **Backend honesty (PASS).** Exactly one solver backend (CP-SAT) plus the `itertools` oracle;
   Z3 declined deliberately and the justification is substantiated by the actual compiled IR
   (wholly finite-domain integer on every surface — no real/quantified constraints). CP-SAT and
   the grid oracle agree on both raw and reduced counts (TOPOLOGY 128→40, OVERCOMMIT 96=96), and
   the TOPOLOGY reduction is independently witnessed bit-for-bit by the standalone parity gate. No
   unsubstantiated symmetry/perf claim was found. (The shared-canonicalizer residual from item 2
   applies here too, minor, disclosed.)

5. **All new source/test files carry the ADR-0006 header** (path + purpose + Public Domain),
   verified per-file. 31/31 tests pass.

### What does NOT work / known limitations

1. **RESOLVED 2026-06-24 — `topology_enum.py` now *consumes* the SSOT (the topology config space is
   single-homed).** This entry previously recorded the ADR-0012 one-home MAJOR failure; it has been
   closed (DESIGN.md §8 step 9). The closure, for the record:
   - The `(policy, nice, latency_nice)` triples now have ONE home: `hp/relations.py`
     (`SERVER/GEN/SURPLUS_POLICIES`). `harness/topology_enum.py` no longer authors them — it
     imports the tables (and the model) from `hp/`. The standalone tool deleted its own CP-SAT
     model and policy tables; its `build_and_enumerate` now calls
     `compile(Target(TOPOLOGY))` → `cpsat.enumerate_configs` → `topology_materialize.materialize`
     (the same path the hp CLI uses), re-sorting to its historical emission order so `configs.json`
     stays bit-for-bit identical.
   - `spec.py`'s `server/gen/surplus_policy` descriptors now cite the real home
     (`PyField(POLICY_HOME="throughput-lab/hp/relations.py", "SERVER_POLICIES")`), not the tool —
     so the cited home is the one production reads (DESIGN.md §10 falsifiable-claim #3 satisfied).
     `surplus_present` was reclassified to `NoCodeHome` (it is an enumeration axis, not a code-home
     literal — the prior `PyField(topology_enum, "surplus_present")` citation pointed at a CP-SAT
     var that no longer exists).
   - The drift lint (`test_ssot_drift.py`) was extended to guard the **FULL** triples (policy +
     nice + latency_nice) of the single home against a canonical in-test reference, plus a
     descriptor-domain↔home agreement check. Demonstrated fail-loud: injecting `nice 10→99` into
     `relations.py SURPLUS_POLICIES` now FAILS the lint (2 failed); reverted, 19 pass.
   - `topology_enum.py`'s docstring no longer claims to be the single home (it states it is a
     consumer). `verify_orbits` is kept as the tool's Oracle A; the pure `_canonical_key` is kept
     only as the parity test's neutral referee.
   - Parity is preserved bit-for-bit at 4c/3g (40 configs) and as an orbit-partition at 5c/3g; both
     oracles green via `hp.cli --verify` (topology 40/40, overcommit 96/96).

2. **The STATIC_LAB surface is scaffolded but empty.** The `variant` axis, bindings, and
   `Surface.STATIC_LAB` exist, but no per-HP descriptors are declared, so `--select static_lab`
   compiles to **1 config** and its `--verify` is trivially vacuous (1 vs 1). The producer + server
   + scheduling + build HPs from the inventory are a mechanical follow-on using the same types.
   The two surfaces the brief named as acceptance targets (topology reproduction + overcommit
   selection) are complete and verified.

3. **`mypy --strict` (DESIGN.md §8 step 1) is not clean across the package.** *(Corrected 2026-06-24:
   the earlier "mypy is not installed" was wrong — mypy 2.1.0 IS in the venv and runs.)* Running
   `MYPYPATH=. mypy --strict hp/` reports **69 errors in 10 files**, predominantly missing annotations
   in `hp/tests/` (`no-untyped-def`, `type-arg`). The value/structure core (`spec.py`, `relations.py`)
   is strict-friendly (frozen dataclasses, typed unions), but the package as a whole does not yet pass
   the step-1 gate — a follow-on cleanup, not a correctness issue (runtime + the 31-test suite are green).

4. **The C++/argparse default *extractor* (the P7-strongest "generate-from-one-source") is the
   filed deferral; v1 ships the P7 lint floor** (`test_ssot_drift.py`). This is the disclosed floor
   the design sanctions (DESIGN.md §1.4 / §9), not a silent stopping point — but combined with
   limitation 1 it means the topology policy tables are neither extracted nor lint-guarded today.

5. **A resolved one-home tension, disclosed:** `pool_threads`/`pool_batch` C++ struct defaults
   (4/32) differ from the OVERCOMMIT operating point (3/64), which lives in `overcommit_sweep.py`'s
   argparse. Those two descriptors' `home` was pointed at the harness argparse (where the operating
   value actually lives) rather than copying a literal — a named binding divergence, per DESIGN.md
   §1.3/§1.4, not a silent disagreement.

---

## File map

| module | responsibility |
|---|---|
| `spec.py` | the SSOT registry: `HParam` descriptors + the descriptor algebra (Domain/SourceRef/Effect/SymmetryClass/Guard/Binding). The single home. Declares TOPOLOGY + OVERCOMMIT; STATIC_LAB scaffolded. |
| `relations.py` | cross-HP structure: TopologyParams, the R1–R4 placement constraints with rationale, the 1:3 partition, the policy vocabularies. (Currently a second home for the policy tables — see limitation 1.) |
| `ir.py` | the backend-neutral `ConfigSpace` / `IRVar` / `IRConstr` / `IRSym` (incl. `CanonInert`), construction-validated. |
| `compile.py` | `Target` + `compile(spec, target) -> ConfigSpace`; selection, binding/variant resolution, projection, the fail-loud selection rule. |
| `backends/cpsat.py` | `to_cpsat` + `enumerate_configs` + the orbit `canonical_key` (the shared invariant). |
| `backends/grid.py` | `to_grid` (Oracle B): an independent `itertools` feasibility re-derivation, no CP-SAT code path. |
| `verify.py` | Oracle A (orbit non-isomorphism), Oracle B (grid cross-check), the inertness self-check; all fail-loud. |
| `topology_materialize.py` | projection → `topology_enum`-compatible record (config_id/tag/placements) for the parity gate. |
| `cli.py` / `__main__.py` | `python -m hp.cli --select … [--pin …] [--variant …] --verify --json out.json` + the provenance ledger. |
| `tests/test_topology_parity.py` | the migration acceptance gate: SSOT TOPOLOGY == standalone `topology_enum.py`. |
| `tests/test_oracles.py` | both oracles pass on every shipped target; an injected fault fails loud. |
| `tests/test_ssot_drift.py` | the P7 drift lint floor (see limitation 1 for its current coverage gap). |
| `DESIGN.md` | the adjudicated design spec this implements. |
