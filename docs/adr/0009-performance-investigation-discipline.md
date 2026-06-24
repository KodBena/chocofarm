# ADR-0009: Performance Investigation Discipline

- **Status:** Accepted
- **Genre:** Tenet (cross-cutting authoring discipline) — the seventh tenet,
  after ADR-0002 (fail loudly), ADR-0004 (minimal-touch), ADR-0005
  (documentation discipline), ADR-0006 (source-file headers), ADR-0007 (file
  size and information density), and ADR-0008 (classification discipline).
  Sibling of ADR-0008: same shape of unsubstantiated-claim failure, different
  domain — classification discipline forbids fuzzy vocabulary-fit; this tenet
  forbids unsubstantiated perf-fit ("this is faster," "this regressed," "no
  change," "behaviorally equivalent") against the closed vocabulary of perf
  and equivalence claims.
- **Date:** 2026-06-15
- **Provenance:** Transferred from the LengYue ADR corpus. The tenet (a perf
  claim is honest only when its investigation is captured and reproducible)
  is universal. LengYue's tool surface was browser DevTools / Firefox-profiler
  for a Vue SPA; chocofarm's perf surface is an ML/search hot path
  (numpy/JAX/numba), so the tool surface, the metric vocabulary, and the
  equivalence bar are re-instanced around chocofarm's real discipline. The
  tenet survives the substitution; only the tooling changes.
- **Scope:** All authoring work that asserts a performance or
  equivalence property — a speedup, a regression, a null result, or "the
  optimized path matches the baseline" — across the `chocofarm/` package.
  Applies to the perf write-ups (`docs/results/az-perf.md`,
  `az-jax-perf.md`), the bench harnesses, and any PR or note that lands a
  hot-path change.

## Context

chocofarm's hot path is the AlphaZero/Gumbel search and the JAX/numba forward,
not a browser render loop. The project already has a real, disciplined perf
posture — it just hasn't been named as a tenet. Three artifacts carry it:

- **`docs/results/az-perf.md` / `az-jax-perf.md`** — the captured perf
  write-ups. They report before/after numbers under a pinned, reproducible
  scenario (e.g. "cold seeded net hidden=256, m=12 n_sims=48, λ₀=0.0855, 20
  episodes, warmed"), not author intuition.
- **`chocofarm/az/bench/bench_hotpath.py`** — the per-component
  micro-benchmark of the search hot path. Each component (`env.marginals`,
  `features.build`, the forward) is timed **in isolation on representative
  captured states** (`states.npz`, beliefs from 15,504 worlds down to 1) so
  "an optimization is validated one component at a time, before/after, and a
  speedup claim is never an artifact of episode-level noise."
- **`chocofarm/az/bench/bench_equivalence.py`** — the behavioral-equivalence
  harness for the float32 + numba optimization.

The thing that makes chocofarm's perf register distinctive is the **two-tier
equivalence bar**, because a perf optimization here changes floating-point
results:

1. **Bit / exact, for the logic invariant.** Where an invariant is a logic
   fact rather than a numeric one — "exactly zero mass on illegal action
   slots" — it is asserted exactly (`== 0.0`). float32 may not perturb it.
2. **Aggregate behavioral, for the numerics.** float32 + numba *will* change
   floats and flip near-tied argmax / Sequential-Halving choices, so the bar
   is **not** bit- or per-decision equality (it can't be). It is **aggregate
   behavioral equivalence**: the optimized policy's fixed-λ₀ rate, mean E[T],
   and action distribution must be statistically indistinguishable from the
   float64 baseline over N≥300 episodes across ≥2 seeds, within Monte-Carlo
   CI — and the MC standard error is reported so "indistinguishable" is a
   number, not an eyeball.
3. **A tighter bar where it is achievable.** The jax/numpy forward
   equivalence test (`tests/test_jax_equivalence.py`) holds the f64 / f32 /
   jax forwards to `ABS_TOL = 1e-4` (comfortably above observed float error,
   below any behaviorally-meaningful difference) — the bit-near-identity
   contract that makes consolidating the four forward implementations (the
   audit's R11 `ForwardSpec`) *safe to attempt*.

The failure modes this tenet exists to forbid are the chocofarm analogs of
the ones any perf register has:

- **A perf claim without a captured before/after.** "The numba kernel is
  faster" with no `bench_hotpath` run attached is the closest-match selection
  ADR-0008's positive register forbids — a defensible-looking classification
  picked without verification.
- **An "equivalent" claim without the equivalence harness.** "The float32
  path matches" without `bench_equivalence`'s CI-overlap evidence is the same
  failure in the equivalence register. The audit's `max|Δp| = 0.0082`
  finding — a reproduced stale-weight divergence in the f32 cache — is exactly
  the kind of silent divergence an unrun equivalence check would miss
  (and which the rebind-not-mutate invariant of ADR-0001 closes).
- **Per-investigation tool re-derivation.** Each ad-hoc perf check
  re-deriving its own timing scaffolding and its own scenario. The cost
  compounds; comparability across investigations drops. The bench harnesses
  exist precisely so the scaffolding is shared.

The structural root is the same one ADR-0002 names at the runtime level and
ADR-0008 at the classification level: when a closed-vocabulary claim is being
made, the claim is honest only when its substantiation is attached.

## Decision

We adopt **Performance Investigation Discipline** as a codebase-wide tenet. A
perf-property or equivalence claim — speedup, regression, null result, or
"matches the baseline" — is honest only when the investigation behind it is
captured in a form the next reader can reproduce.

### Triggers — when to capture

1. **Before claiming a speedup landed.** A perf write-up or PR that asserts a
   hot-path change made something faster attaches the before/after numbers
   from `bench_hotpath` (per-component, on the captured states) or an
   episode-level capture under a pinned scenario. Without the pair, the claim
   reduces to author intuition.
2. **Before claiming the optimized path is equivalent to the baseline.** A
   float32 / numba / forward-consolidation change attaches the equivalence
   evidence: the `== 0.0` logic invariant for the exact part, and
   `bench_equivalence`'s CI-overlapping rate / E[T] / action-distribution for
   the behavioral part, or `tests/test_jax_equivalence.py`'s `ABS_TOL = 1e-4`
   for the forward. "Equivalent" is a claim against the equivalence
   vocabulary and needs the same substantiation a speedup does.
3. **Before/after a structural refactor of the hot path** (the forward
   consolidation, the parallel substrate, the feature builder). A baseline
   capture before the refactor lands gives a reference point if a felt or
   measured regression surfaces later.

### Tools — canonical surface

The canonical chocofarm perf surface (swap-the-tool, keep-the-tenet from
LengYue's browser surface):

- **`bench_hotpath.py`** — per-component timing on representative captured
  states. The regression guard: one component at a time, before/after.
- **`bench_equivalence.py`** — the behavioral-equivalence harness (fixed-λ₀
  rate, mean E[T], action histogram, MC standard error) over N≥300 episodes,
  ≥2 seeds, f64-vs-f32.
- **`bench_value_target.py`** — the value-target MC-limit identity check.
- **`tests/test_jax_equivalence.py`** — the f64/f32/jax forward bit-near-
  identity test at `ABS_TOL = 1e-4`.
- **`capture_states.py` → `states.npz`** — the representative-state corpus the
  benches run against (belief widths from full to singleton), so a capture is
  reproducible rather than dependent on ambient run state.

### Metric vocabulary

A canonical metric set lets investigations compare across time:

- **Per-component wall time** (before/after, on `states.npz`) — the speedup
  comparable.
- **Fixed-λ₀ rate `ΣR/ΣT`, mean E[T], action histogram, with MC standard
  error** — the behavioral-equivalence comparable.
- **Forward absolute difference** (value, logits) against `ABS_TOL = 1e-4` —
  the forward-numerics comparable.
- **Logic invariants asserted exactly** (`== 0.0` illegal-slot mass) — the
  bit-exact comparable, distinct from the behavioral one and never relaxed to
  a tolerance.

Additions to the vocabulary go here, not in per-investigation write-ups (the
per-investigation scatter the audit's §2.G / SSOT discipline forbids).

### Acceptance criteria for perf/equivalence-claimed changes

- **Speedup claims** attach before/after numbers under the same reproducible
  scenario (same belief corpus, same net, same seeds), via `bench_hotpath` or
  a pinned episode capture.
- **Equivalence claims** attach the two-tier evidence: the `== 0.0` exact
  invariant *and* the CI-overlapping behavioral metrics (or the forward
  `ABS_TOL` test).
- **Refactor PRs touching the hot path** attach the pre-refactor baseline.

The absence of substantiation does not block a change from landing — perf work
proceeds at the author's judgment — but the write-up **states the absence
explicitly** rather than carrying an unsubstantiated claim. The honest shape:
*"defensively sound but not substantiated by a bench pair; the speculative win
is X, the cost is Y."* This is the loudly-marked unsubstantiated case
(parallel to ADR-0008's deliberately-imprecise tags), not a fuzzy-fit claim
against the closed vocabulary.

## Calibration on the two-tier bar

The bit-vs-behavioral distinction is the chocofarm-specific calibration and
must not collapse:

- A **logic invariant** (illegal-slot mass, a legality mask) is a bit fact;
  asserting it with a tolerance would be a category error — it is `== 0.0` or
  it is a bug.
- A **numeric result** (a rate, a forward value under float32) is a
  behavioral fact; asserting it bit-exactly would be a category error in the
  other direction — float32 + numba legitimately move the float, so the honest
  bar is statistical indistinguishability within MC CI.

Confusing the two is the failure: pinning a float-sensitive rate bit-exactly
forbids a legitimate optimization (the ADR-0008 fossil/fuzzy failure in the
perf register), while relaxing a logic invariant to a tolerance admits a real
bug. The discipline is to apply the bar the quantity's *kind* demands.

## Consequences

### Positive

- **Perf and equivalence claims are legible across time.** A write-up backed
  by a bench reference is one a future investigator can extend; an unbacked
  claim is a dead end they must re-derive.
- **Substantiation cost is paid up-front.** Capturing the bench during the
  work is cheaper than reconstructing the scenario later — composes with
  ADR-0005's author-as-you-decide.
- **The forward consolidation is safe.** The `ABS_TOL = 1e-4` test and the
  bit-exact invariants are the contract that makes collapsing four forwards
  (audit R11) safe to attempt — the perf discipline directly enables a
  structural refactor.

### Negative

- **Per-claim authoring overhead.** Each perf/equivalence assertion carries
  "is this substantiated?" The answer must be "yes, bench attached" or "no,
  explicitly marked unsubstantiated."
- **Discipline is policy, not mechanism.** No automated check verifies that a
  write-up's perf claim is backed by a referenced bench (ADR-0011 Rule 1: a
  declared review-only surface; a claim-token scanner would be the
  mechanization trigger).
- **The behavioral bar needs enough episodes to be meaningful.** N≥300, ≥2
  seeds is the floor; a cheaper check is not a substantiation.

### Neutral

- **No retroactive sweep.** Existing write-ups whose claims lack the full
  bench evidence are not targeted for rewrite (ADR-0004's incremental-retrofit
  posture). The discipline operates at the moment of new authoring.
- **No mandate on a fixed benchmark beyond the existing harnesses.** New
  hot-path classes may need new benches; the tenet names the existing ones and
  the metric vocabulary, not a frozen suite.

## Exceptions

### Structural-by-inspection wins

A change whose perf effect is provable by inspection — an O(N²) replaced by
O(N) at a hot path, a redundant full-marginals pass removed (the audit's dead
`env.marginals()` at `netvalue_ismcts.py:54`) — does not require a bench pair.
The structural argument substantiates the claim; the write-up still names the
argument. This parallels ADR-0002's structurally-provable exception.

### Exploratory observations

A write-up may include exploratory perf observations made during investigation
without elevating them to the authoritative register ("I noticed X; needs a
bench before it's load-bearing"). The discipline applies to the authoritative
register, not the exploratory.

## Revisit when…

1. **A specific rule introduces its own failure mode.** Flag as the revisit
   trigger.
2. **The canonical tool surface needs replacement.** If the project's hot path
   moves (a C++ search seam — see `docs/design/scaling-and-cpp-seam.md` — or a
   different profiling stack), the tool surface updates; the discipline
   survives the substitution. Extensions go here by dated amendment.
3. **The metric vocabulary stops covering the perf-relevant axes.** A new
   investigation class (a multi-instance scenario sweep, a GPU forward) that
   the existing metric set doesn't fit warrants extending the vocabulary here,
   not in a per-investigation write-up.
4. **A check can mechanize the substantiation requirement.** A scanner that
   verifies a perf-claim write-up references a bench would partially mechanize
   the discipline (ADR-0011 Rule 1's enforcement-surface trigger).

## Related

- **ADR-0002 (fail loudly).** The reactive ancestor. An unsubstantiated perf
  or equivalence claim is the silent-failure shape ADR-0002 names, in the
  write-up authoring register.
- **ADR-0001 (immutability / rebind-not-mutate).** The `max|Δp| = 0.0082`
  stale-weight divergence the audit reproduced is exactly the equivalence
  failure this tenet's harness catches and ADR-0001's rebind invariant
  prevents.
- **ADR-0005 (documentation discipline).** Perf write-ups are documentation
  events; author-as-you-decide (Rule 6) is the temporal posture this tenet
  relies on for capturing the bench during the investigation, not after.
- **ADR-0008 (classification discipline).** The proactive sibling. "Faster" /
  "regression" / "equivalent" are closed-vocabulary claims; substantiation is
  the vocabulary-fit verification ADR-0008 implies for the perf register.
- **ADR-0010 (this corpus's render-locality entry).** Has no chocofarm
  applicability (no UI); its Related note points back here as the nearest
  chocofarm perf concern.

## Amendments

### 2026-06-24 — The throughput-lab perf surface + a captured-investigation DB (Revisit #2 + #4 fired)

The hot path moved exactly as Revisit #2 anticipated — the **C++ search seam**
(`docs/design/scaling-and-cpp-seam.md`), realized as the `throughput-lab/`
producer→boundary→inference-server testbed. The tool surface extends accordingly:

- **Canonical tools (extending §Tools):** `throughput-lab/harness/` — `episodic_dps.sh`
  (the production-shape DPS baseline), `coalesce_sweep.py`, `topology_enum.py` +
  `topology_sweep.py` (the CP-SAT-enumerated config space), and the forward
  microbench. These are this register's `bench_hotpath` analogs for transport
  throughput.
- **Metric vocabulary (extending §Metric vocabulary):** the throughput register —
  **leaf-rows/s** (the comparable, server-feed-rate), **DPS** (decisions/s), **real
  rows/forward** (batch fill), **server util %**, **forwards/s**, **LPD**. These are
  the transport-throughput comparables, distinct from the per-component wall-time and
  the equivalence comparables already named; a throughput claim attaches them.

And the substantiation requirement is now **mechanized** (Revisit #4, partially): a
captured throughput number is no longer prose. **`throughput-lab/harness/exp_db.py`**
(the `throughput_research` Postgres store) persists every reading with its code stamp
(commit/tree), HP config, exact command, and replicates — so a perf claim is
*code-addressable* and aggregated (median/IQR), never a single eyeballed number
(`robust-benchmark-statistics`). Crucially, this register now distinguishes the
**measurement** from the **interpretation** at the schema level: a reading
(`tlab_reading`) is an immutable fact; a perf *claim about* it ("X is faster", "the
+31% was an artifact") is an authored, supersedable **finding** (`tlab_finding`) —
the measured-vs-interpreted separation this tenet's honesty depends on, made
structural so an overturned claim is auditable, not lost. (The belief-layer mechanism
is recorded in ADR-0011's 2026-06-24 amendment; this is its perf-register face.)

## License

Public Domain (The Unlicense).
