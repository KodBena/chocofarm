<!-- docs/notes/leaf-eval-refactor-audit-2026-06-22/01-plan-vs-result.md — Public Domain (The Unlicense) -->

# 01 — Plan vs. result

[← README](README.md) · [02 — misnomer →](02-misnomer-adr-analysis.md)

## The ratified plan

`docs/design/leaf-eval-bound-responsibility-refactor.md` was authored as an advisory
and ratified as written ("looks good to me" on the **plan**, not on per-commit
discretion). It proposed two intertwined things:

**(A) A package relocation (§3)** — make the flat directory a real package along
responsibility seams:

```
tools/analysis/leaf_eval_bound/
  __init__.py            # NEW: a real package; kills the sys.path preamble
  contract/   estimate.py · grounding.py · grounded_types.py · references.py
  store/      bench_store.py · manifest.py · reconstruct.py
  alloc/      driver.py · kink.py · report.py · gradient.py
  models/     base.py · capacity.py · cycletime.py · transport/{...}
  benchmarks/ estimators.py · pools.py · harness.py · scaffold.py · bench_*.py · register.py
  runners/    support.py · throughput_bound.py · transport_sweep.py · untrusted_drive.py
  examples/   demo_msgpass.py
```

**(B) Seven "load-bearing moves" (§3)** — the seam-closing dedup/interface subset,
ranked by leverage.

**Plus §4 renames** (OpenTURNS→leaf_eval_bound; `neyman_driver`→`alloc/driver`;
`leaf_eval_grounding`→`contract/grounding` + the references/grounded_types split) and
**§5**, which framed the JAX swap as an adjacent arc the decomposition eases — and
advised unifying the model `f` to a single JAX-traceable callable, *retiring* the
numpy twin.

Two binding qualities of the plan: moves 1–5 are **behavior-preserving** and must
carry the **ADR-0009 bar** (the bound numbers unchanged on the existing tests), and
"each box can land on its own commit."

## The execution arc

Twenty commits, `1261f73` (the advisory) → `9cff51a`. Mapped to the plan, the
**numbered moves executed wildly out of leverage order**, with the JAX migration
interleaved (full map + diffstats in [04](04-evidence-log.md)):

```
move 4 (ec070fa, unlabeled in subject) → move 5 (d3914b2, 944606f) → hack-audit z-fix
  (8d0c764) → move 3a/b/c (shim deletions) → JAX J1–J4 → rename OpenTURNS→leaf_eval_bound
  → move 1 (075147f) → move 2 (8d34957) → move 3 typed-contract (7ad7ae7) → move 7 (9cff51a)
```

Two consequences of that ordering are visible in the history itself:

- **Move 5's numpy half was transient.** `944606f` single-homed the numpy
  delta-method fallback into `runner_support.py`; then the JAX migration `fc1c8be`
  (J4) **retired the numpy fallback wholesale**, deleting `runner_support.py`, its
  tests, *and* the hack-audit z-divergence pin (`8d0c764`) that had been parked in
  that test file. The plan's §5 had explicitly foretold that the JAX swap retires the
  numpy twin — so a JAX-first ordering avoids this churn. (Move 5's *gradient* half,
  `alloc/gradient.py`, survived and became the JAX seam — so only the numpy half was
  wasted.)
- **Move 3 straddled the whole migration:** the shim deletions (3a/b/c) ran *before*
  JAX, the typed-Protocol contract (move 3) *after* — a single plan move split across
  ten unrelated commits.

## Move-by-move conformance

| Plan item | Ratified intent | What landed | Verdict |
| --- | --- | --- | --- |
| Move 1 | split `bench_common` → `estimators`/`pools`/`harness` | done (`075147f`); bodies sliced verbatim; bound numbers byte-identical | ✅ |
| Move 2 | reconstruct glue → **`store/reconstruct.py`** | done (`8d34957`) but **top-level** `reconstruct.py` (no `store/`) | ✅ intent / ⚠ location |
| Move 3 | one model base + delete runner shims | re-scoped: shims already gone; delivered a typed `TransportModel` Protocol + conformance net as **`model_base.py`** (not `models/base.py`); gated on the §6 "growing family?" question | ✅ re-scoped |
| Move 4 | Clark kink → `alloc/kink.py` | done (`ec070fa`) | ✅ |
| Move 5 | one `runners/support.py` (gradient + numpy bound) | gradient half → `alloc/gradient.py` (survives); numpy half → `runner_support.py` then **deleted by J4** | ⚠ half transient |
| Move 6 | bench `scaffold.py` | **not done** | ❌ |
| Move 7 | discovery-driven `register.py` | done (`9cff51a`) as `register_benches.py` | ✅ |
| §3 package tree | `contract/ store/ models/ runners/` + top-level `__init__.py`; kill the `sys.path` preamble | only `alloc/` + `benchmarks/` exist; **no `__init__.py`**; preamble in **48 files** | ❌ largely unbuilt |
| §4 rename | OpenTURNS → leaf_eval_bound | done (`c1d954f`) | ✅ |
| §4 rename | `neyman_driver` → `alloc/driver` (maintainer-flagged) | **not done** — `neyman_driver.py` flat, `alloc/driver.py` absent | ❌ |
| §4 split | grounding → grounding/grounded_types/references | **not done** | ❌ |
| §5 | JAX swap; **retire** the numpy `f` twin | swap done (J1–J4, clean); but the **`throughput_numpy`⊕`throughput_jax` dual-write survives** | ⚠ half-done |

## The structural gaps, in priority order

1. **The §3 package skeleton — the plan's centerpiece — is unbuilt.** No
   `contract/store/models/runners`; no top-level `__init__.py`; the `sys.path.insert`
   preamble persists in **48** files; `neyman_driver.py` is flat and un-renamed;
   `leaf_eval_grounding.py` is unsplit. The deferral was flagged *inline* (a
   `944606f` commit section, an `alloc/__init__.py` note) but **is not recorded in
   `BACKLOG.md`** — the project's own home for consciously-deferred work — so the
   headline of a ratified plan sits in undocumented limbo.

2. **The driver god-object is intact (1051 lines, 2.6× the ADR-0007 ceiling).** Move 4
   lifted the kink and gradient; the §2.3 split of the `Recommendation` formatter
   (concern "E") into `alloc/report.py` and of `run()`/SOCP into `alloc/driver.py`
   did not happen. `report`/`where_to_spend`/`run`/`_socp_allocation`/`_assemble_sigma`
   are all still co-resident (verified line numbers in [04](04-evidence-log.md)).

3. **The model-`f` dual-write survives** — the §5 "single strongest" simplification.
   Every model still hand-writes `throughput_numpy` *and* `throughput_jax`; the
   muParser string went away but a second hand-written home remained. Enforcement is a
   runtime pinning test (`test_jax_f_equivalence.py`), the weakest tier — not the
   derivation §5 wanted.

4. **Move 6 and two renames remain open** — defensibly (move 6 is prophylactic; the
   renames are coupled to gap 1), but against ratified scope.

## On conduct

The deviations were, to the agent's credit, **flagged inline** — `944606f` carries a
section headed "STRUCTURAL DEVIATION FROM THE DESIGN NOTE — flagged for scrutiny";
move 3's commit says "RE-SCOPED honestly"; every move cites the ADR-0009 bar with the
actual bound numbers (E[f] 419.8/428.8, CI 98.2/53.0, "byte-identical"). That is the
honest register, and the *code that landed* is competent.

But disclosure is not authorization, and a flagged deferral is still a deferral. The
delivery — declared done at ~half the ratified plan, a flagship engine still bearing a
name its own docstring refutes (see [02](02-misnomer-adr-analysis.md)), the deferral
unfiled in `BACKLOG.md` — is where the work reads as amateur, even though the
individual commits do not.

[← README](README.md) · [02 — misnomer →](02-misnomer-adr-analysis.md)

*Public Domain (The Unlicense).*
