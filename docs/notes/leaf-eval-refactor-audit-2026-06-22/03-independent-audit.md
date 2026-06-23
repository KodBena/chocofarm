<!-- docs/notes/leaf-eval-refactor-audit-2026-06-22/03-independent-audit.md — Public Domain (The Unlicense) -->

# 03 — Independent conformance audit

[← 02 misnomer](02-misnomer-adr-analysis.md) · [04 evidence log →](04-evidence-log.md)

This phase is the report of an **independent `general-purpose` agent**, run unprimed
with a neutral both-ways brief: audit the end state of `tools/analysis/leaf_eval_bound/`
against the ADRs; read each ADR end to end before citing it; **treat commit-message and
docstring self-descriptions as claims to verify, not as evidence of compliance**; report
real violations *and* what is genuinely clean; do not manufacture findings. It read ADRs
0002/0005/0006/0007/0008/0009/0011/0012 in full, the ratified plan, and the modules in
full (`neyman_driver.py` all 1051 lines, the runners, manifest, reconstruct, model_base,
grounding, `alloc/*`, estimate, the models).

Reproduced verbatim-cleaned. **My verification of its headline quantitative claims**
(F1 line count + concern co-residence, F3 the 48-file preamble, F4 the dual-write, F2
the absent rename target) is recorded in [04](04-evidence-log.md); I corrected its one
numeric slip — the driver is **1051** lines, not 1052.

---

## Findings

**F1 — ADR-0012 P3 (god-object) + ADR-0007 (size): `neyman_driver.py` is still a
~1051-line god-object. [MAJOR]**
`class NeymanDriver` (line 250) — 19 methods, still co-owning six concerns the advisory
§2.3 enumerated for splitting: the Estimate/input seam, Σ-assembly (`_assemble_sigma`,
678), the SOCP solver (`_socp_allocation`, 798; `_closed_form_allocation`; `_fundability`),
the CI multiplier (`_family_multiplier`, 734), the `Recommendation` presentation formatter
(`report`, 195; `where_to_spend`, 191 — concern E), and the `run()` orchestration loop
(945). Only the kink (→`alloc/kink.py`) and gradient (→`alloc/gradient.py`) were
extracted. 2.6× the ADR-0007 400-line ceiling. The advisory's `alloc/{driver,report}.py`
split did not happen.

**F2 — ADR-0008 (fossil label): file/class still named "neyman" despite computing a
strict superset. [MAJOR]**
File `neyman_driver.py`, class `NeymanDriver` (`alloc/driver.py` absent). The code's own
docstring establishes it is not strict Neyman: lines 70–76, "THE ALLOCATION (§2.3). The
cost-constrained c-optimal SOCP … reduces to the closed form `n_i* ∝ √(a_i/c_i)` on the
diagonal"; `_socp_allocation` solves "the cost-constrained c-optimal allocation as a
SOCP," of which Neyman is the diagonal special case, plus a Clark-1961 kink path that is
not Neyman at all. The advisory §4 ratified `neyman_driver → alloc/driver`;
`alloc/__init__.py` itself admits the driver is "to become `alloc/driver.py` in a later
increment." The label is accurate for one branch, misnames the whole. *(Developed in
full in [02](02-misnomer-adr-analysis.md).)*

**F3 — Ratified plan §3/§4 largely ABSENT: the package skeleton never landed. [MAJOR]**
On disk vs ratified §3: **ABSENT** — top-level `__init__.py` (so the
`sys.path.insert(0, _HERE)` preamble persists in **48 files**), `contract/`, `store/`,
`models/`, `runners/` subpackages, `runners/support.py` (move 5), `benchmarks/scaffold.py`
(move 6), the `leaf_eval_grounding`→`contract/grounding` split into
`grounding.py`/`grounded_types.py`/`references.py` (all three concerns still in one
227-line file). **PRESENT** — `reconstruct.py` (move 2), `benchmarks/{estimators,pools,
harness}.py` (move 1), `model_base.py` (move 3, partial), `alloc/{kink,gradient}.py`
(move 4), `register_benches.py` (move 7). Roughly half the ratified moves landed; the
headline "make it a real package, kill the sys.path preamble" did not.

**F4 — ADR-0012 P1 (dual-write): every model writes its throughput `f` twice. [MAJOR]**
Each model carries a hand-typed `throughput_numpy` AND a hand-typed `throughput_jax` —
independent re-encodings of one expression (dict-key vs tuple access, `min` vs
`jnp.minimum`), e.g. `model_zmq_baseline.py:110` vs `:121`; `model_capacity.py:84` vs
`:95`. Mitigation is a pinning test (`tests/test_jax_f_equivalence.py:73`,
`assert float(M.throughput_jax(arr)) == pytest.approx(M.throughput_numpy(x0), rel=1e-9)`),
not derivation — the weakest enforcement tier (runtime test as backstop, ADR-0012 P7).
The advisory §5 said the JAX swap's "single strongest" win was to make `f` one function
and **retire** the numpy twin; the muParser string went away but a second hand-written
home remained.

**F5 — ADR-0005 Rule 3/5 + ADR-0008 (stale path-walk comments): 3 bench files name the
dead `OpenTURNS` directory. [MINOR]**
`benchmarks/bench_r_gen.py:71`, `bench_lpd.py:94`, `bench_g_core.py:86` each carry a live
`..`-walk comment naming `OpenTURNS`; the directory is now `leaf_eval_bound` (each file's
own line-2 header says so) — the comment contradicts its own header. The code
(`dirname×4`) is correct; only the comment resolves to a dead name.

**F6 — ADR-0005 Rule 3 (stale description): `alloc/__init__.py` describes the gradient
backend by its pre-swap state. [MINOR]**
"`gradient` — … OpenTURNS analytic `f.gradient()` with a finite-difference fallback
today; the ONE site the JAX swap replaces." But the swap is **done** — `alloc/gradient.py`
is pure `jax.grad`, imports no OpenTURNS. The description narrates a retired dependency as
"today."

**F7 — ADR-0002 + P1 (minor): broad `except` hiding two re-typed literals in
`untrusted_drive.py`. [MINOR]**
`untrusted_drive.py:135-136`: `except Exception:` / `iota, trow = 94.58, 4.317` swallows
any failure reading `_G.SERVE_INTERCEPT_US.mean`/`SERVE_SLOPE_US.mean` and substitutes
hardcoded literals that duplicate `leaf_eval_grounding` (a P1 dual-home) behind an
over-broad catch. Low impact: feeds only an ETA progress banner, never a bound number.
Also `transport_sweep.py:124 _OUTPUT_PULL_US = 9.14` is a live literal whose only other
home is a comment in grounding — a minor P1 (its siblings `_DISPATCH_FLOOR_US`/
`_STAGED_INTERCEPT_US` are correctly derived from `G.*`, so this is an inconsistency).

## What is genuinely clean (both-ways honesty)

- **ADR-0006 (headers): fully compliant.** All 48 `.py` files carry the path-first module
  docstring; all 48 declare "Public Domain (The Unlicense)"; every header path matches the
  new `leaf_eval_bound/` location (no stale `OpenTURNS/` header paths).
- **ADR-0002 (fail loudly): emphatically loud, ~zero violations.** 20+ clean named raises
  (`ValueError`/`RuntimeError`/`ImportError`/`KeyError`/`TypeError`), many citing ADR-0002,
  vs no clear-cut violation. Highlights: `manifest.py:380-386` byte-for-byte fixed-point
  guard; `neyman_driver.py:916` SOCP `gᵀΣ(n*)g≈V*` assertion (the solver's `optimal` status
  explicitly not trusted); `manifest.py` distinguishes "postgres absent" (announced,
  seed-only) from "SQL fault on open connection" (raises). The broad `except`s that exist
  are announced (stderr rows, `# noqa: BLE001`) or bounded.
- **`estimate.py` is a clean, fully-typed SSOT contract (P8 + ADR-0002 exemplary):** frozen
  dataclass, `__post_init__` validates (symmetric/PSD/finite/length) and raises rather than
  coercing; `ShrinkLaw` a proper 5-variant typed sum type; `from_jsonb` re-runs the gate on
  read. No lying signatures (the `=None` defaults are honestly typed `… | None`).
- **`reconstruct.py`, `manifest.py`, `alloc/*` are clean, single-concern, typed, fail-loud.**
  Move 2 (acyclic DAG), move 1, move 7 landed well. `alloc/jax_backend.py` even shows the
  P8 named-relaxation discipline (`# type: ignore[no-untyped-call]` with reason).
- **JAX migration is genuinely complete:** no file imports openturns; the "imports NO
  OpenTURNS" docstrings are true; `examples/demo_msgpass.py` updated to "requires jax + scipy."
- **Live referrers repointed (ADR-0005):** main-tree `tests/` import the new names; no stale
  `bench_common`/`register_baseline`/`analysis/OpenTURNS` referrers in the tracked tree.
- **The transport-model dialect shims the advisory flagged were largely resolved:**
  `_registry_qname`/`_model_sigmas` verbatim-duplication is gone (uniform
  `model.registry_qname`/`sigmas`); the residual `_untrusted`/`_model_estimates` are thin
  one-shape calls over the unified interface; `model_base.py` documents the contrast helpers
  as legitimately optional (a recorded P2 decision, not a sniffing shim).

## Overall characterization (independent agent)

> This is a partial, honestly-incomplete delivery — roughly half the ratified plan. The
> numerics migration (OpenTURNS→JAX) is complete and clean, and the cleanest, lowest-risk
> responsibility moves landed well: the contract (`estimate.py`) is an exemplary typed SSOT,
> `reconstruct.py`/`harness`/`pools`/`estimators` split cleanly, registration is
> discovery-driven, the fail-loud posture is exemplary, headers are 100% compliant. But the
> plan's structural skeleton never landed: no top-level `__init__.py` (the `sys.path`
> preamble persists in 48 files), none of the `contract/store/models/runners/` subpackages,
> the `neyman_driver.py` god-object is intact at ~1051 lines and still carries its fossil
> "neyman" name (the dir rename DID land, the driver rename did NOT), the grounding split
> didn't happen, and the model-`f` dual-write the JAX swap was meant to dissolve survives.
> The work is internally honest about its incompleteness — `alloc/__init__.py` calls itself
> "the FIRST increment" and names the deferred `alloc/driver.py`. The remaining gaps are not
> regressions or hidden corners; they are the larger relocation/subtraction moves still
> queued. Net: clean as far as it goes, but materially short of the ratified end state, with
> the god-object + fossil name the two most visible unaddressed items, and the `f` dual-write
> the most substantive un-dissolved P1 hazard.

[← 02 misnomer](02-misnomer-adr-analysis.md) · [04 evidence log →](04-evidence-log.md)

*Public Domain (The Unlicense).*
