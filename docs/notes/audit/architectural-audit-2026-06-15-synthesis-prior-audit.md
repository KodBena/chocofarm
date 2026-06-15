# Cross-Audit Synthesis — reconciling the prior architecture-refactor audit (2026-06-15)

A synthesis of the prior whole-project audit (`docs/design/architecture-refactor-audit.md`, branch `docs/architecture-refactor-audit`) against this session's audit (`architectural-audit-2026-06-15.md`). Notional **Appendix E** to the latter. The maintainer's instruction was explicit and shaped the method: *treat the prior document as untrusted* — read it for transferable lessons and net-new findings, but verify its claims against the running code rather than absorb its framing, and compose its conclusions into the verified record only where they survive that check.

**Status posture.** OPEN — advisory. No code changed. This document quarantines the prior audit's unverified claims from this audit's verified record and states explicitly what folds in, what corrects this audit, and what is held at arm's length.

**Method.** The prior note was read in full at byte fidelity (via `git show`, not a summarizer). Its load-bearing and net-new claims were then checked against the actual branches with `git ls-remote`/`git ls-tree`/`git show`/`grep`. Evidence-class tags as in the parent audit: *(runtime-verified)*, *(byte-verified)*, *(grep-verified)*, *(cited-not-rerun)*.

**One-line verdict.** The prior audit is *more* trustworthy than the "untrusted" label implied — on the design-architecture axis it is genuinely good — but it is **design-only and self-admittedly uncertified** (its own §6: "No code was run"), and a few of its specific cites do not resolve on `main`. Its single highest-value contribution is a **material correction to this audit's scope**: it audited a different branch, and that branch carries a configuration layer this audit never saw.

---

## §1 — The headline: this audit was branch-blind, and the prior one wasn't

The most important thing reading the prior audit changed: **it ran against a different branch, and that branch contains a central configuration layer that falsifies one of this audit's strongest universal claims.**

- This audit ran against `main@cfce276`. The prior audit's commission (its Appendix A) explicitly placed it on **`feat/hp-registry`** — "the latest state, including the in-flight hp registry + `config.py`."
- The repo has **10 branches on origin** *(grep-verified: `git ls-remote --heads`)*: `main`, `feat/{hp-registry, az-jax-train, az-residual}`, `firewall/residual-loss`, the design-note branches `docs/{hyperparam-registry-spec, sim-parallelization-viability, training-optimization-refactor}`, and the two competing audits (`docs/architecture-refactor-audit`, `docs/architectural-audit-2026-06-15`).
- **`feat/hp-registry` carries `chocofarm/config.py` and `chocofarm/hp/{__init__,registry,schema}.py`; `main` carries none of them** *(verified: `git ls-tree -r origin/feat/hp-registry` lists all four; `origin/main` lists none; they are absent from this audit's working tree)*.
- I read `hp/registry.py` and `exit_loop.py` on that branch *(byte-verified)*: the registry is a redis-backed, strict-decode, **live-read** codec (`load_snapshot` → `ConfigSnapshot` with a `launched_with` shadow; the loop refreshes the snapshot once per outer-iteration boundary, reads HOT fields off it, and fires `assert_no_restart_drift` loudly if a RESTART/INSTANCE field moved). `hp/schema.py` is a typed dataclass SSOT with HOT/RESTART/INSTANCE facets.

**Consequence for this audit.** §2.A and §3.10 state "There is no central configuration object anywhere in `chocofarm`." That is **true for `main` and false for the in-flight `feat/hp-registry`**, where a typed-schema SSOT + live-read registry is a *partial cure for exactly the disease this audit named central*. This audit's diagnosis is not wrong — it is **scoped to the wrong branch**. The correction is recorded in §7.

The transferable lesson is methodological and sharp: **a "whole-project" audit that reads one branch is not whole-project.** Before asserting a universal absence ("there is no X anywhere"), enumerate the branches and know what is in flight. The prior audit got this right only because its commission handed it the branch; left to its own devices it would have had the same blind spot. The discipline this audit must adopt: *enumerate `git ls-remote` first, and state which ref every universal claim is scoped to.*

**A second-order finding falls out of the same fact.** The project's architecture — code *and* design — is fragmented across ~10 unmerged branches: the registry on `feat/hp-registry`, the optimizer-split spec on `docs/training-optimization-refactor`, the sim-parallelization characterization on `docs/sim-parallelization-viability`, two independent audits, plus `feat/az-*` and `firewall/*`. This compounds this audit's §2.G (knowledge offloaded to prose the code cannot enforce): the knowledge is not only in prose, it is in *prose on branches that have not landed*. The single highest-leverage process fix the two audits jointly imply is **land the in-flight work to `main`** so that "the codebase" is a single resolvable thing.

---

## §2 — Trust calibration (the distrust mandate, discharged)

Every net-new or load-bearing claim of the prior audit I relied on was checked. The non-resolving ones are recorded so they are quarantined, not silently inherited.

| Prior-audit claim | Check | Verdict |
| — | — | — |
| `feat/hp-registry` carries a typed-schema SSOT + live-read registry + `config.py` | `git ls-tree`, read `registry.py`/`schema.py` | **confirmed** *(byte-verified)* — and absent from `main` |
| Registry reads HOT fields live each iteration; the loop rebuilds at the boundary | read `exit_loop.py` on `feat/hp-registry` | **confirmed** *(byte-verified)*; `lr/l2` remain RESTART (baked into `optax.adam`) — its own docstring says so |
| `JaxTrainer` conflates Trainer ⊕ Optimizer; `lr` baked at construction | matches this audit's §2.A / R13 | **confirmed**, independently |
| `GumbelAZSearch` fuses search ⊕ Policy ⊕ value-target rule; `_v_mix`/`_improved_policy` belong in `value_target.py` | grep `gumbel_search.py` + `value_target.py` | **confirmed** *(grep-verified)* — net-new (§3 N2) |
| Reference constants hardcoded in ~10 sites, already disagree (`0.094` vs `0.0941`) | matches this audit's §4 | **confirmed**, independently — both audits caught the same drift |
| `MiniEnv` reimplements env belief mechanics by hand | matches this audit's §2.B / §3.7 | **confirmed**, independently |
| `analysis/` is orphaned (used only by itself) | grep importers tree-wide | **confirmed** *(grep-verified)* — net-new (§3 N3) |
| `facemodel.py` un-wired; env references it only in a comment | matches this audit's §2.E / §3.1 | **confirmed**, independently |
| `clairvoyant_rate` "implemented twice (`harness.py` + `eval_bound.py:52-77`)" | grep `def clairvoyant_rate` on `main` | **does NOT resolve by name on `main`** (only `harness.py:28`); this audit's own bounds reader flagged a clairvoyant *solve* in `eval_bound`, so directionally right but the cite is a branch artifact |
| Line numbers throughout (`exit_loop.py:317-336`, `gumbel_search.py:104`, etc.) | spot-checked vs `main` | **model-layer cites match** (`env.py` `marginals`/`apply` at 99/125 on both); **`az/`/`eval/` cites do not** — they are `feat/hp-registry` line numbers and must be re-resolved per branch |

**Calibration verdict.** The prior audit's *structural* claims hold up under verification with one near-miss (the clairvoyant duplication is real in substance but mis-cited by name/branch). Its weakness is the one it declares itself: it is an argument from reading, never executed, so its "behavior-identical"/"bit-reproducible" claims for the refactor steps are *arguments*, not certified diffs — exactly the gap this audit's adversarial-verification pass exists to close. **Distrust discharged: trustworthy on architecture, uncertified on behavior, branch-scoped on line cites.**

---

## §3 — Net-new findings it surfaced that this audit missed

Verified, and worth folding in.

- **N1 — the `hp/` registry + `config.py` layer (branch divergence).** Covered in §1. Corrects this audit's §2.A/§3.10. *(byte-verified)*
- **N2 — the AZ target is split across two files; the policy-target rule is welded into the search engine.** `value_target.py` owns the *value*-target rules (`suffix_returns_to_go`, `blended_returns_to_go`), but the *policy*-target rule — the Danihelka `softmax(logit + σ(completed_q))` improved policy — lives inside `gumbel_search.py` as `_v_mix`/`_improved_policy` (`:231/:383`), side-reading node statistics so it cannot be called outside a live tree *(grep-verified)*. This audit caught that `GumbelAZSearch` is config-frozen and carries two `env` sources, but **missed the cleaner decomposition**: the pure, unit-testable, research-frontier target rule should be an extractable function in `value_target.py`, reusable by any search. The prior audit's framing — "search emits stats; `value_target.py` owns the rule" — is the right abstraction and this audit did not name it. **Fold in.**
- **N3 — `analysis/` is orphaned from the live pipeline, and decomp's clusters are hardcoded rather than read from the analyzer.** Nothing outside `analysis/` imports it *(grep-verified)*. This audit *praised* `analysis/` as the most disciplined code in the tree (true — A.8) but did not note that it is disconnected, nor the coupling implication: the solver's cluster definitions and the analyzer's are two unrelated implementations of one intent. Praise and orphaning are both true; the parent audit recorded only the first. **Fold in as a qualification to A.8.**
- **N4 — the V̂-as-Strategy framing, and the lazy import that breaks a self-inflicted cycle.** This audit flagged the bounds debt (MiniEnv duplication, `DecompVhat` copy-paste) but the prior audit's diagnosis is sharper: `info_relaxation.py` bundles ~five V̂ strategies in one module, forcing a lazy import (`:113` on its branch) to break a dependency cycle *the bundling itself created* — and V̂ is precisely the Strategy Port the calibration agenda needs (plug a trained `V̂_AZ` into the dual bound). The "lazy import as a symptom of a misplaced boundary" is a better lens than this audit's "copy-paste." **Fold the framing in.** *(cited-not-rerun — the five-strategy count not independently recounted)*
- **N5 — implicit bit-width contracts.** `kernels.py`'s int64 bitmask reduction silently assumes `N < 64`; true on the live env (N=20), unguarded. This audit did not note it. Minor. *(cited-not-rerun)*
- **N6 — the C++-sim swappability lens (§5 of the prior audit).** Out of this audit's commission, but a genuine architectural property: the env↔Policy seam is language-agnostic (scalars + raw bytes cross it), and the transport/MiniEnv consolidations make it *legible and singular*. Useful as a forcing-function test for "is this boundary clean" even absent a C++ plan. **Note, do not fold as a finding.**
- **N7 — the meta-finding: fragmentation across ~10 branches.** Covered in §1; this audit's §2.G undercounted it (prose-as-spec is worse than stated — it is prose-as-spec *on unlanded branches*).

---

## §4 — Where the two audits independently converge (corroboration)

Two audits run by different agents under different framings, against different branches, converging on the same finding is stronger evidence than either alone. They agree on:

- **The env↔Policy seam is the one clean boundary and the template for everything else** (prior §0; this audit §1/§6 seam-preservation). Both make it the organizing fact.
- **`JaxTrainer` freezes `lr` at construction; the queued lr-anneal is forced through `--resume`** (prior §2.1; this audit §2.A, R13). Both trace it to the same handoff experiment.
- **The feature/action layout has no single owner** (prior §2.3 "five files"; this audit §2.B "three writers"). The count differs because the prior audit includes `mlp.py`'s unvalidated `n_actions` and `gumbel_search.py:104`'s `term_slot` re-derivation as layout-knowledge sites; this audit counted only the three that *encode block offsets*. Both land on the same fix (a `FeatureLayout`/`FEATURE_LAYOUT` value object). Not a contradiction — different boundary-drawing of the same defect.
- **`ParallelExecutor` is a god-object fusing pool + transport + task** (prior §2.5; this audit §2.A/3.5).
- **`MiniEnv` is a hand-copied second Environment the dual bound certifies against** (prior §2.7; this audit §2.B).
- **The reference constants are hardcoded in ~10 sites and the `0.094`/`0.0941` copies already disagree** (prior §2.8; this audit §4). *Both audits independently caught the exact same drift* — the strongest convergence point.
- **`facemodel.py` is un-wired** (prior §2.9; this audit §2.E).
- **The `_JDTYPE`/forward-selection decision is scattered** (prior §2.6; this audit A.3 minor).

This convergence is itself a deliverable: the items above are now *doubly* confirmed, and the roadmap can treat them as settled.

---

## §5 — Where this audit is stronger, and where the prior one is (compositional honesty)

Neither audit dominates; they are different epistemic objects and should be composed, not ranked.

**This audit has what the prior lacks:**
- **Verification.** The prior audit ran no code and certifies nothing (its §6 says so plainly). This audit's adversarial pass *executed the env* and refuted two of its own framings — including the runtime check that the reference literals are **not actually stale** (`realizable_static=0.08553`, `clairvoyant=0.14537`; floor/ceiling are detector-independent). The prior audit calls the constant drift "the SSOT-violation-as-bug live in the tree" — true that the *drift exists*, but it never checked whether the frozen values currently *misreport*, and the answer is no (yet). This audit's §5 deflation is exactly the precision the prior audit's design-only posture cannot reach.
- **An evidence ledger** tracing every conclusion to a re-opened line, and a **frozen-config rollup** enumerating the symptom.

**The prior audit has what this one lacks:**
- **A cleaner target architecture.** Its **boundary/Ports inventory** (§3.1) and its decomposition lens — "right abstraction/boundary," god-objects split by *lifetime and reuse profile* (Optimizer⊥Trainer = two lifetimes; Transport⊥Pool⊥Task = three change-reasons) — is in places sharper than this audit's four-tier config model. The "two lifetimes in one constructor" framing of `JaxTrainer` and the "three responsibilities fused" framing of `GumbelAZSearch` are better abstractions than "frozen at construction."
- **A sequenced refactor plan with per-step verification and dependency ordering** (its §4), pitched at implementation. This audit's §7 roadmap is leverage-ordered but less granular on the per-step *verification* (which existing test pins each step). The prior audit ties each step to `test_jax_equivalence`/`test_az_loop`/`test_parallel_deadlock` and a per-step assertion — a discipline this audit should borrow.
- **The C++-seam analysis** (its §5), absent here.

The right composition: **the prior audit's boundary inventory + sequenced plan, sitting atop this audit's verified findings and runtime corrections.** Its design layer, this audit's evidence layer.

---

## §6 — Transferable lessons (the core ask)

Distilled for this analyst's own future work; each is a discipline, not a restatement.

| # | Lesson | Why it transfers |
| — | — | — |
| X1 | **Enumerate the branches before claiming "whole project."** `git ls-remote` first; scope every universal claim ("there is no X anywhere") to a named ref. | This audit asserted "no central config object anywhere" and was falsified by one `git ls-tree` on an in-flight branch. The cheapest possible miss, and the most embarrassing. |
| X2 | **"What is the right boundary here?" is a sharper organizing question than "what is frozen/duplicated?"** SSOT/DRY/frozen-config are *symptoms that locate a misplaced boundary*, not the headline. | The prior audit's lens produced cleaner decompositions (two-lifetimes, three-responsibilities) than this audit's tier model for the same defects. Lead with the boundary; let the symptom point at it. |
| X3 | **Split a god-object by the *independent reasons its parts change*, not by "it does too much."** Optimizer⊥Trainer (different lifetimes), Transport⊥Pool⊥Task (different change-drivers), search⊥target-rule (different reuse profiles). | "Too much" is a smell; "these two have different lifetimes / change-reasons" is the actual cut line. The second is actionable; the first is a complaint. |
| X4 | **An untrusted peer audit is a free adversarial check.** Converging findings are doubly-confirmed; *diverging* findings (the clairvoyant cite, the 3-vs-5 layout-site count) are precisely where to look — they mark a boundary one of you drew differently. | Distrust was the instruction, but the document's value was highest exactly where it disagreed or added — not where it echoed. Mine the deltas. |
| X5 | **A design-only audit and a verified audit are non-substitutable; compose them.** Reading produces the cleaner target architecture; execution produces the trustworthy present-tense claims (e.g. "the literals are not stale *yet*"). | The prior audit's "constant drift is a live bug" and this audit's "drift is latent, not realized" are both correct at different epistemic layers. Neither dominates; the synthesis needs both. |
| X6 | **Borrow the per-step verification discipline.** Tie each refactor step to the *specific existing test* that pins it + one new per-step assertion, and name inter-step dependencies. | This audit's roadmap is leverage-ordered but under-specifies *how each step is proven safe*. The prior audit's "Step N — verify with `test_X` + assertion Y; depends on Step M" is the standard to adopt. |
| X7 | **Use the C++-/FFI-swappability question as a boundary-legibility test, even with no port planned.** "Could a foreign implementation drop in here without a Python type leaking across?" is a fast forcing function for "is this seam clean." | The env↔Policy seam passes it; the god-object transport fails it not because the seam is wrong but because it is *illegible*. The question separates "true boundary" from "legible boundary." |

---

## §7 — Disposition: corrections, folds, and quarantine

**Corrections this audit needs (high priority):**
- **C1 — branch-scope the central-config claim.** §2.A and §3.10 must be amended: "*On `main@cfce276`* there is no central config object; an in-flight `feat/hp-registry` branch introduces a typed-schema SSOT (`hp/schema.py`) + live-read registry (`hp/registry.py`) + `config.py` that is a partial cure — it makes `n_step`/`td_lambda`-class fields HOT but **leaves `lr/l2`/search-width RESTART** (still baked at construction), so the §2.A core and R13 stand for the levers that matter most." This both corrects the overclaim and *sharpens* the finding: the cure exists and is incomplete in exactly the way that matters.
- **C2 — qualify the `analysis/` praise (A.8).** Add: disciplined *and* orphaned (no live importer); decomp's clusters are hardcoded rather than read from it.

**Folds (verified net-new, compose into the verified record):**
- **F1 — the AZ-target split (N2):** add as a finding under §2.E/A.4 and a roadmap item (extract `improved_policy(...)`/`v_mix(...)` as pure functions in `value_target.py`; `GumbelPolicy` becomes a thin adapter). Verified.
- **F2 — the V̂-Strategy-Port framing (N4):** fold into §3.7/bounds as the principled replacement (a `Vhat` Protocol with impls split by dependency), superseding this audit's looser "route through a public entry point."
- **F3 — adopt the per-step verification discipline (X6)** into the §7 roadmap.
- **F4 — the C++-seam legibility test (N6/X7)** as a design note, not a finding.
- **F5 — the meta-finding (N7/§1):** "land the in-flight branches to `main`" becomes the highest-leverage *process* recommendation, above any single refactor — because every other recommendation is ambiguous until "the codebase" is one resolvable ref.

**Quarantine (their claim, not certified here — do not promote to verified):**
- The "`clairvoyant_rate` implemented twice" cite (does not resolve by name on `main`; treat as "a clairvoyant solve is duplicated," which this audit's own bounds reader independently asserted, rather than as the prior audit's specific line cite).
- All `az/`/`eval/` line numbers in the prior audit (they are `feat/hp-registry` coordinates; re-resolve per branch before acting).
- The "five V̂ strategies" count and the `kernels.py` N<64 claim (plausible, not independently recounted — `cited-not-rerun`).

---

## §8 — Net verdict

The prior audit deserves more credit than "untrusted" suggests: its architecture map, boundary inventory, and sequenced plan are disciplined and largely correct, and on two points — the **`hp/` registry branch divergence** and the **AZ-target-rule decomposition** — it is materially ahead of this audit because it saw a branch this audit did not and drew a boundary this audit missed. It is weaker in exactly one way, the way it admits: it ran no code, so its present-tense "this is a live bug" claims are uncertified, and one of them (the constant drift "misreporting") this audit's runtime check downgrades to latent.

The two compose cleanly and should be merged, not ranked: **the prior audit's target architecture and sequenced plan, laid over this audit's verified findings, runtime corrections, and evidence ledger, with the central-config claim re-scoped to `main` and the AZ-target split folded in.** The first action on the merged record is not a refactor — it is **landing the in-flight branches so that "the codebase" is one thing**, after which the doubly-confirmed items (env↔Policy as the template, the constant SSOT, the MiniEnv duplication, the JaxTrainer split, the feature-layout owner) can be executed against a single resolvable tree.

---

*Companion to `architectural-audit-2026-06-15.md` and its appendix. The prior audit being synthesized is `docs/design/architecture-refactor-audit.md` on branch `docs/architecture-refactor-audit`. Point-in-time; not retro-edited. Public Domain / The Unlicense.*
