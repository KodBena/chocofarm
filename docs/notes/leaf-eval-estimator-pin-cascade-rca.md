<!--
docs/notes/leaf-eval-estimator-pin-cascade-rca.md — RCA of the recurring
"measured-quantity-wired-as-an-un-shrinkable-Fixed-pin" defect class in the
leaf-eval Neyman-allocation estimator suite (tools/analysis/OpenTURNS/), and
the single structural change + guards that would have prevented the cascade.

A postmortem note (point-in-time; ADR-0005 Rule 8 — not retro-edited once
written). Sibling of docs/notes/jaxtrain-deadlock-rca.md and
docs/notes/lab-staging-divergence-rca.md (the house RCA style). Read-only
commission: no code or existing doc was edited in producing this.

Public Domain (The Unlicense).
-->

# RCA — the leaf-eval estimator "measured-but-pinned" cascade

**Branch analysed:** `feat/issue-control-lab` (the branch carrying the arc; its
HEAD when this RCA was written was `7fbc352`). All file/line citations below are
read from that branch's working tree (the leaf-eval `tools/analysis/OpenTURNS/`
suite). **Provenance note (a context discrepancy, surfaced per ADR-0004):** this
RCA's own git worktree was cut from `71594a5`, which **predates the entire
`tools/analysis/OpenTURNS/` leaf-eval suite** (the suite's 49 files / +12 408
lines are *introduced* in `71594a5..22cc56f`), so the worktree's checked-out tree
does not contain the analysed files. The arc's commit objects are reachable from
`feat/issue-control-lab` and were read there; nothing was inferred from a tree the
worktree could not show.
**Scope of the arc:** the harmonized-`Estimate`-interface migration and its
follow-on fixes — commits `5eb1f8e` (Phase-4 migration complete) through
`22cc56f` (the sizing-kwarg single-home), all landed **2026-06-21** in a single
~14-hour push, plus `7fbc352` (the dated amendment recording the shm/mpsc tmsg
reclassification *and naming the still-open shm wakeup crash*), plus the runtime
failure captured in `~/shm_spin_poll_fail`.
**Status:** code-grounded and history-grounded; every claim is anchored to a
file/line or a commit. One runtime failure (`~/shm_spin_poll_fail`) was read
directly; the rest is read from the `feat/issue-control-lab` tree and the git
log. I did **not** re-run the benches (read-only commission; the benches are
timing-sensitive sole-workload).

**ADRs read end-to-end for this RCA** (per the root `CLAUDE.md`/ADR-0002
read-before-citing rule): the **adr-synopsis**, **ADR-0002** (fail loudly),
**ADR-0005** (documentation discipline), **ADR-0008** (classification
discipline), **ADR-0011** (mechanization discipline), and **ADR-0012**
(compositional & structural hygiene, all 1329 lines). The
**`docs/design/harmonized-estimator-interface.md`** design note was read in full
(all 1619 lines — §0 is the dated changelog spine; §1–§8 the contract and its
executed verifications). `docs/notes/jaxtrain-deadlock-rca.md` was read in full
for the house style.

---

## 0. The verdict (one paragraph)

The cascade is **real and is primarily an ADR-0011 failure** (a recurrence never
converted to a mechanism — fixed one instance at a time as prose-and-patch),
**sourced in two ADR-0012 violations that the Phase-4 migration *manufactured
wholesale*: a P1 single-source violation (≈30 hand/fleet-templated copies of one
`_measure_raw`/`_estimate_from_raw`/`pin_estimate` shape, so the "is this quantity
measured-and-fundable?" decision has no single home and was re-made, wrongly, ≈30
times) and a P8 lying-signature violation (a bench labelled `needs_measurement=
True` on its `Grounded` while its body returns a `Fixed` pin — the signature says
"a sole-workload run will tighten this," the body says "no budget reduces it").**
The maintainer's hypothesis — "a massive ADR-0011 + ADR-0012 violation, primarily
ADR-0011 but sourced in ADR-0012" — is **confirmed in shape and largely
correct**, with one sharpening: the *originating* defect is not a single
mis-classification but the **migration's choice to propagate a per-bench template
instead of deriving from one home** (the audit's own cancer **D**, "copy-paste
programs," recurring inside the very migration that was fixing cancer-G prose),
and the *snowball mechanism* is precisely **ADR-0011 Rule 4** (every guard written
for the cascade quantifies over the *instance* or a *hand-enumerated list*, never
over the *class* — so each net fails open at the next instance, which is why
R_gen, then g_core/LPD/tmsg×5, then the sizing-kwarg, then a fresh `shm_spin_poll`
runtime crash each had to be discovered separately). The deepest root cause, one
level under both ADRs, is that **the bench↔driver contract carries load-bearing
invariants that live nowhere checkable** (shrinkable ⇒ sizable;
`needs_measurement` ⇒ not-`Fixed`; a sizing budget ⇒ a pool the estimator can
consume) — the call-boundary form of cancer **G**.

---

## 1. The symptom, stated mechanically

The leaf-eval suite computes a throughput lower bound by a delta-method Neyman
allocation: a driver (`tools/analysis/OpenTURNS/neyman_driver.py`) holds one
**`Estimate`** per model input, reads each input's already-divided sampling
variance off `Estimate.cov`, and allocates the next measurement batch to the
inputs whose extra samples most tighten the bound's CI. How an input's variance
responds to effort is carried by its `ShrinkLaw` (`estimate.py`): a `Poolwise`
mean shrinks `−V/n²`, a `QuantileLaw` median shrinks `−cov/n`, a `RegressionLaw`
fit is leverage-floored, and a **`Fixed` pin has marginal `0`** — *no finite
budget reduces it*.

The recurring defect: a quantity that **is actually measured** (it has a runnable
benchmark) was wired so its `measure()` returned a **`Fixed`** `Estimate` — built
by wrapping the v1 seed in `pin_estimate(get_seed())`. Because a `Fixed` law's
marginal is `0`, the allocator computes `A_i = −marginal·n² = 0`, the input is
**un-fundable**, and if that input sits on the **binding** arm the loop has
nothing to sample and **stalls** (spins identical rounds, `+samples=0`
everywhere). The same generative shape recurred one quantity at a time: `R_gen`,
then `g_core`, then `LPD`, then `tmsg` and its five transport variants, then a
*second* layer (the driver could not even *size* the now-shrinkable quantity),
then a *third* (a fresh `shm_spin_poll` runtime crash). Each fix was correct in
isolation; the class was never closed as a class.

The two shapes are visible side-by-side in `bench_common.py` — the three Phase-3
helpers each bench's `_estimate_from_raw` calls:

- `median_estimate(pool, …)` (`bench_common.py:221`) → a **shrinkable**
  `QuantileLaw` with a bootstrap median SE (fundable: marginal `< 0`).
- `fit_estimate(rows, medians, …)` (`bench_common.py:116`) → a **shrinkable**
  `RegressionLaw` (leverage-floored, funded by widening the x-design).
- `pin_estimate(value, sigma, …)` (`bench_common.py:293`) → an **un-shrinkable**
  `Fixed` (marginal `0`; the punt).

The bug class is: a bench called `pin_estimate(get_seed())` where it should have
run its measurement and called `median_estimate`/`fit_estimate`.

---

## 2. The originating defect — what created the class (ADR-0012 P1 + P8)

### 2.1 The migration templated the punt into every bench (P1 / cancer D)

Commit `5eb1f8e` ("Phase 4 — delete the coercion, measure()->Estimate, migrate
the pilot") is the **origin**. Its own message records the method:

> measure() -> Estimate (all 30 benches) … _measure_raw (the raw-pool producer)
> + _estimate_from_raw (the ONE home of the Estimate construction) + measure() =
> thin _estimate_from_raw(_measure_raw()). **(4 reference benches hand-written;
> Sonnet fleet applied the template to 26; Opus-verified all 30.)**

So the same three-function shape was **stamped into ≈30 files** (verified:
`grep -l "_estimate_from_raw\|_measure_raw" benchmarks/` returns all 30 bench
modules). For the genuinely-measured quantities whose live bench was not yet
wired (`R_gen`, `g_core`, `LPD`, the five `tmsg` variants), the template's
`_estimate_from_raw` was filled with `pin_estimate(get_seed())` — a placeholder
that compiles, validates, and *looks* migrated. The decision "is this quantity
measured-and-therefore-shrinkable, or a true pin?" was thereby **re-made ≈30
times, independently, by hand/fleet** — and made wrong wherever the live bench
lagged the template.

This is **ADR-0012 P1** (single source of truth) violated at the structural
level, and it is the **audit's cancer D** (copy-paste programs instead of one
parameterized runner) recurring **inside the very migration that was curing the
heuristic** — Phase 4's headline was deleting `untrusted_drive._per_sample`, the
longest-numeric-list heuristic that grabbed `t_row`'s `[32..512]` row-count axis
as a "pool" and cratered the bound to `E[f]≈11.9` (`untrusted_drive.py:22-28`
preserves the post-mortem of that symptom). The migration fixed one cancer (the
guessing heuristic, cancer G) by spreading another (the templated punt, cancer
D). The contract's intent was right — "the bench **declares** its Estimate, the
driver never guesses" (P2 reject-don't-guess) — but a *declaration copy-pasted
≈30 times with no single owner of the measured/pinned axis* is a P1 violation
that re-opens the same wound one bench at a time.

### 2.2 The lying signature: `needs_measurement=True` ⊕ `Fixed` body (P8)

`leaf_eval_grounding.py` is the genuine SSOT for the grounded physical
quantities (`Grounded`, line 22), and it does the classification honestly: every
measured-but-not-yet-instrumented quantity carries **`needs_measurement=True`**
— `R_gen` (line 142), `g_core` (line 137), `LPD` (line 124), `tmsg_us_leaf`
(line 184), `B_op` (line 163), `tau_io` (line 111), `iota`/`slope` (lines
81/87). The flag's own docstring (lines 27–36) says it is exactly "whether it
still needs a fresh SOLE-WORKLOAD measurement (the Neyman loop ranks these)."

The defect is the **contradiction between that flag and the bench body**: the
bench `measure()` returned a `Fixed` (`pin_estimate`) for a `Grounded` whose
`needs_measurement` is `True`. The annotation said "a sole-workload run will
tighten this"; the body said "no budget reduces it." That is **ADR-0012 P8's
lying signature** verbatim (and ADR-0002's "a config field the receiver cannot
honor must not be silently accepted") — the `needs_measurement=True` contract
the *driver* relies on (it ranks these for funding) is **not honored** by the
bench that returns an un-fundable `Fixed`. Critically, the two facts had **no
shared home**: `Grounded.needs_measurement` lived in `leaf_eval_grounding.py`,
the `Fixed`-vs-shrinkable decision lived independently in each bench's
`_estimate_from_raw`, and **nothing checked one against the other**. The bench's
`pin_estimate(constant=…)` derives from `Grounded.constant` (a P1 single-home the
suite *did* get right — `d5f84b7` correction 3, the N_gen flag), but **nothing
derives the shrinkable-vs-pinned axis from `needs_measurement`**. That missing
derivation is the P1/P8 leak through which every instance of the class escaped.

### 2.3 Why ADR-0008's "a prior with no bench is correctly a pin" is *not* the excuse

ADR-0008's neutrality result — a genuine engineering-judgement prior with no
runnable bench is *correctly* a `Fixed` pin (the design note's PIN-declared-spread
row, `harmonized-estimator-interface.md:711-713`) — is real and is *not* what
these were. The distinguishing fact, stated by each fix commit, is that the
**bench already existed and already ran**: `R_gen`'s `bench_r_gen._measure_raw`
runs the **built** C++ gen-ceiling binary (`bench_r_gen.py:114-173`) and the seed
itself is *labelled MEASURED* (`leaf_eval_grounding.py:140-142`, "adapter.md §2
line 93 'MEASURED gen 152 dps/core'"); the `tmsg` benches **already timed the
live codec/ring** and *discarded* it for the seed (the design note's blow-by-blow,
`harmonized-estimator-interface.md:166-169`, "did a REAL codec measurement but
`_estimate_from_raw()` wrapped the **seed** 1.0us in `pin_estimate`"). So these
were the **positive-register ADR-0008 failure** — a fuzzy match against an
inadequate vocabulary (calling a *measured* quantity a *pin* because the pin
encoding was the closest-fitting template slot), not the honest pin the
neutrality result protects. `B_op` is the genuine exemption and is treated below
(§5) — it is the one quantity whose faithful measurement (`bench_b_op.py:11-18`,
a saturated end-to-end rows/forward histogram) is an **un-built** harness
artifact, so its `Fixed` is honest *today*. The class is "measured quantity
mis-pinned," and `B_op` is correctly *not* in it.

---

## 3. The snowball — why it recurred one instance at a time (ADR-0011 Rule 4)

The defect was authored ≈30 times at once (§2.1); the *recurrence* — discovering
and fixing it one quantity at a time across seven commits — is the ADR-0011
failure. Every step diagnosed the **instance** and built a net (or no net) that
quantifies over the **instance**, never the **class**. ADR-0011 Rule 4 names
exactly this: *"enumerations of instances fail open at the next instance; a net
keys on a structural slot, a name/shape predicate, or a derived-from-one-source
invariant."* The arc is a near-perfect demonstration of the rule's negative case.

### 3.1 The cadence (commit timeline, all 2026-06-21)

| time | commit | what it fixed | what it left (the next instance) |
| — | — | — | — |
| 14:43 | `5eb1f8e` | Phase-4 migration: `measure()->Estimate` ×30, delete `_per_sample` | **created the templated `pin_estimate(get_seed())` punt** ×8 measured-but-pinned benches |
| 14:51 | `21f0497` | `untrusted_drive` `AttributeError` on `lockfree_mpsc` (two model-map shapes) | "the existing test covered only `zmq_baseline`, **so the gap was invisible**" — an enumeration-fails-open instance, 8 min later |
| 15:52 | `faf16cc` | the allocator's `A_i = Σ_ii·len(pools)` conflation → typed `ShrinkLaw.marginal` | surfaced (its own message) that **every real fit de-funds** (no `per_point_var`) — a related honesty gap |
| 16:09 | `573ca88` | per-step progress + ETA (the minutes-long silent waits) | (UX for the stall, not the stall) |
| 16:35 | `a14452b` | `run()` halts loud when nothing is fundable — the stall **made loud** | the *cause* of the empty fund-set (a binding pin) still un-addressed |
| 18:19 | `d5f84b7` | **first reclassification**: `R_gen` `Fixed`→`QuantileLaw`; N_gen DEGENERATE; prior floor; `Grounded.constant` single-homed | "**Flagged, NOT fixed (same measured-but-punted pattern off the same binary): g_core …**" |
| 20:54 | `b60b29a` | **the class audit**: `g_core`, `LPD`, `tmsg`×5 flipped; `iota/slope` flag | even here a *new* sub-shape: the `iota/slope` `needs_measurement` **double-home**, and the tmsg **sizing** not yet wired |
| 21:?? | `22cc56f` | the driver can **size** tmsg: single-home `SIZING_KWARGS`, add `budget`/`leaves` | a *second-layer* class (shrinkable-but-un-sizable) — and `~/shm_spin_poll_fail` shows a *third* |
| 21:35 | `7fbc352` | doc amend: shm/mpsc tmsg reclassified; **names the still-open shm wakeup crash** | the wakeup pool-floor crash (§4) — explicitly "tracked separately," still open |

Seven distinct code fix-turns plus a documented-but-open runtime crash, for
**one** generative defect. `a14452b` made the stall *loud* (good ADR-0002), but
loud-on-the-symptom is not the mechanism that prevents the next instance — it is
the ADR-0011 Rule 2 "more prose / another patch" register, applied seven times.

### 3.2 The shape repeated, and each "class fix" found a new layer

- `d5f84b7` fixed **R_gen** and *named* `g_core`/`LPD` as the same shape but did
  not fix them ("Flagged, NOT fixed"). That is the recurrence-not-mechanized
  admission in the commit message itself.
- `b60b29a` *did* audit the class ("found the class is systemic") and flipped
  `g_core`, `LPD`, `tmsg`×5 — but **inside the class fix a new sub-register
  surfaced**: `iota`/`slope` were already shrinkable `RegressionLaw`s; their
  defect was a *different* P1 leak — `needs_measurement` **double-homed**
  (`leaf_eval_grounding.py:66-76` records the fix: a literal-default `False` on
  `Grounded` vs `not trusted` on the manifest, so the static path printed
  "grounded" while the manifest path printed "needs-measurement" for the **same**
  physics). One class fix, a fresh P1 home-count bug.
- `22cc56f` then found the **second-layer class**: the tmsg benches were now
  shrinkable but the driver could not **size** them — `_make_measurer`
  introspects `measure()` for a recognized sizing kwarg
  (`untrusted_drive.py:170`), and `budget`/`leaves` were **absent** from the
  recognized set, *and the recognized set was itself duplicated* (an inline tuple
  in `bench_common.warm()` and `untrusted_drive._ITERS_KW`). The design note
  calls this exactly: *"the silent-de-fund shape one layer over, at the driver's
  introspection seam rather than the bench's `_estimate_from_raw`"*
  (`harmonized-estimator-interface.md:307-326`). The fix single-homed the list
  as `bench_common.SIZING_KWARGS` (`bench_common.py:387`), aliased by
  `untrusted_drive._ITERS_KW` (`untrusted_drive.py:78`). A P1 duplicate, fixed —
  again — only after it bit.

The through-line: **the class was attacked with per-instance flips and
per-instance/per-list nets, so each layer (pin → sizing → window-granularity)
revealed itself only when the next confounded run hit it.** That is ADR-0011
Rule 4's "fails open at the next instance," seven layers deep.

### 3.3 The guards that exist today are *still* per-instance (the smoking gun)

The most telling evidence that the class was never closed: the regression guards
the arc *did* mint quantify over the **enumerated instance**, not the class.

- `tests/test_untrusted_drive_phase4.py:247`
  (`test_reclassified_tmsg_benches_expose_a_recognized_sizing_kwarg`) iterates a
  **hand-listed `_TMSG_MODNAMES`** — the six known tmsg benches. A *new*
  shrinkable bench with an unrecognized knob fails open. (It does assert the P1
  single-home `U._ITERS_KW is BC.SIZING_KWARGS`, line 242 — good — but the
  coverage is the instance list, not "every shrinkable bench in the corpus.")
- The "FUNDABLE / marginal < 0" guards are **one test file per fixed instance**:
  `test_bench_tmsg_codec_framing.py`, `test_bench_futex_wake_tmsg_reclassification.py`,
  `test_bench_mpsc_shm_tmsg_reclassification.py`,
  `test_bench_iota_t_row_grounding_classification.py`,
  `test_bench_r_gen_cpp_gen_ceiling.py`, `test_bench_g_core_cpp_gen_ceiling.py`,
  `test_bench_lpd_cpp_leaf_count.py`. Eight per-instance test modules; **zero**
  class-level invariants over the bench corpus.
- There is **no** test that, by *discovery*, asserts "every bench whose
  `Grounded.needs_measurement` is `True` returns a non-`Fixed` Estimate" or
  "every shrinkable Estimate's bench exposes a recognized sizing kwarg." Exactly
  the two nets ADR-0011 Rule 4 demands and the arc never built.

---

## 4. The fresh runtime failure (`~/shm_spin_poll_fail`) — same root, *new* layer

The captured failure is **not** the `Fixed`-pin stall, and it is **not** the
sizing-kwarg gap. It is a **third instance of the same root cause** — the
bench↔driver contract carrying an un-stated invariant — surfacing in a sub-family
the per-instance fixes never audited. The maintainer's `7fbc352` message
independently reaches the same conclusion, calling it "a DISTINCT still-open
shm_spin_poll defect (the wakeup-funding crash: an allocator budget below the
median_estimate >=2-reading floor) tracked separately." This RCA is the
root-cause analysis of exactly that defect.

**What happened (read from `~/shm_spin_poll_fail`):** the un-trusted drive on
`model_shm_spin_poll` pilots fine, allocates `+samples=6` to `wakeup`, then on
the re-measure `bench_shm_spin_poll_wakeup.measure(trials=6)` **raises** inside
`median_estimate`:

```
ValueError: median_estimate('shm_spin_poll_wakeup_us'): need >= 2 readings for a
bootstrap median SE; got n=1 (ADR-0002: a 1-sample 'median' has no defensible
variance — it raises, it is not padded).
```

**Why it is the same root cause, not a new one:** the `wakeup` bench *is*
reclassified-shrinkable (it returns `median_estimate`, `bench_shm_spin_poll_wakeup.py:128`)
and *is* sizable (`trials` is in `SIZING_KWARGS`). So the *first two* layers are
fixed here. The crash is a **third** un-stated invariant of the same contract:
**the sizing knob a bench advertises is not guaranteed to map to ≥2 realized
pooled readings.** `bench_shm_spin_poll_wakeup._measure_raw(trials=6)`
(`bench_shm_spin_poll_wakeup.py:74-120`) spins a producer thread that bumps a
shared counter and a server thread that records `(observe − bump)` *only when it
catches a new counter value* (line 107) *and* the read is untorn (line 111). At a
tiny `trials`, the producer races to completion and the spinning server observes
**one** distinct value before `done.is_set()` breaks the loop — so
`per_trial_us` has **`n=1`** (the pool length is `len(per_trial_us)`, line 117,
the *realized* count, not `trials`). `median_estimate` then correctly fail-louds
(ADR-0002) on a 1-sample pool.

**Why the canonical fix did not cover it:** the reference benches solved "produce
≥2 readings for any budget" *structurally but inconsistently*:

- `bench_r_gen._measure_raw` guards the **budget**: `n = max(2, int(reps))`
  (`bench_r_gen.py:186`) — and r_gen's pool is deterministic (one reading per
  rep), so `max(2,…)` suffices.
- the six **tmsg** benches guard the **window count**: `n_windows = max(2, iters
  // _WINDOW)` (e.g. `bench_shm_spin_poll_tmsg.py:130`,
  `bench_lockfree_mpsc_tmsg.py:138`) — a window loop *deterministically* emits one
  reading per window, so ≥2 windows ⇒ ≥2 readings.
- the **race-based wakeup** benches (`bench_shm_spin_poll_wakeup`, and by the
  same shape `bench_futex_wake_wakeup`, `bench_zmq_baseline_wakeup`,
  `bench_cpp_inproc_port_wakeup`, `bench_lockfree_mpsc_wakeup`) have **no such
  guard**: their realized pool is the *race-dependent* count of caught wakeups,
  which underflows below the `trials` they were asked for. There is **no `max(2,
  len(pool))`** anywhere in `bench_shm_spin_poll_wakeup.py` (confirmed).

Contrast `bench_tau_io._measure_raw` (`bench_tau_io.py:96-113`): it appends
exactly one reading per `cycles` iteration with **no race**, so `cycles=6`
yields 6 readings — it never underflows. So the failing sub-family is precisely
the *race-based* benches, the one shape the deterministic reference fixes
(`r_gen`'s `max(2,reps)`, the tmsg window loop) do not generalize to. This is the
**same class** — "a bench↔driver contract invariant that lives nowhere
checkable" — and the *same recurrence mechanism* (a fix that quantified over the
shapes it had in front of it, not the class), surfacing one layer further out.

**Severity:** the crash is a fail-loud `ValueError` (ADR-0002 working as
intended — far better than a silent stall), not a corrupted number. But it
**halts a binding-arm measurement on the shm_spin_poll model**, so the
shm_spin_poll bound cannot be driven to convergence at small budgets today. It
fits the root-cause story exactly: not a new disease, a new *instance* of the
un-homed-contract disease.

---

## 5. The systemic fix — close the class, not the instances

### 5.1 The one structural change (the originating P1)

**Make the shrinkable-vs-pinned axis derive from one source, the way
`Grounded.constant` already does.** Today `Grounded.constant` is single-homed and
both the bench's `pin_estimate(constant=…)` and the manifest's seed-Estimate
derive from it (`leaf_eval_grounding.py:27-36`, the fix landed in `d5f84b7`
correction 3). `Grounded.needs_measurement` already exists and already carries
the exact bit — "this quantity has (or wants) a real measurement." The structural
move is to **make `needs_measurement` *generative*, not just descriptive**: a
bench whose `Grounded.needs_measurement is True` must route through a *runnable*
measurement path (`median_estimate`/`fit_estimate`), and a bench whose
`needs_measurement is False` *and* whose measurement is un-built routes through
`pin_estimate`. The decision then has **one home** (the `Grounded` flag plus a
single registry fact "does a runnable bench exist") instead of ≈30
independently-edited `_estimate_from_raw` bodies. This is P1 applied to the axis
the migration left un-homed — and it dissolves the originating defect, because
the punt cannot be *authored* once the classification is derived rather than
hand-typed per bench.

The complementary structural move (lower leverage, but it removes the cancer-D
duplication the cascade rode on): **factor the three repeated idioms into
`bench_common`** — (a) the window-loop pool builder (six near-identical copies:
two with `_FRAMES_PER_WINDOW=200`, four with `_WINDOW=1000`, the loop body
duplicated verbatim — `bench_tmsg.py:120`, `bench_zmq_baseline_tmsg_us_leaf.py:113`,
`bench_cpp_inproc_port_tmsg_us_leaf.py:91`, `bench_futex_wake_tmsg_us_leaf.py:117`,
`bench_shm_spin_poll_tmsg.py:132`, `bench_lockfree_mpsc_tmsg.py:140`), and (b) the
`next((k for k in SIZING_KWARGS if k in params), None)` introspection predicate,
copy-pasted at `untrusted_drive.py:170`, `bench_common.py:407`, and
`tests/test_untrusted_drive_phase4.py:262`. A single
`window_pool(measure_fn, *, units, window, budget)` helper that owns the
`max(2, …)`-minimum-pool guarantee is the **one place** the §5.2 invariant below
(c) would be enforced for the whole family — which is exactly where the
`shm_spin_poll_wakeup` crash would have been prevented, because a shared
pool-builder owns "always return a fundable pool."

### 5.2 The guards (over the class, per ADR-0011 Rule 4)

Three test/CI-gate invariants, each **quantified over the discovered bench
corpus** (not a hand-list), each landing at the strongest feasible surface
(ADR-0011 Rule 1):

- **(a) Single-home the classification, gate that no double-home re-forms.** A
  discovery test that, for every registered quantity, asserts the bench's
  `Estimate.shrink` kind is consistent with `Grounded.needs_measurement` **and**
  the run-built `cpp`/runnable-bench fact — i.e. the same fact never disagrees
  across the `Grounded` flag, the bench body, and the manifest seed path. This is
  the class form of the per-instance
  `test_bench_iota_t_row_grounding_classification.py` and the generative version
  of the structural fix in §5.1.

- **(b) shrinkable ⇒ sizable, over the corpus.** Generalize
  `test_reclassified_tmsg_benches_expose_a_recognized_sizing_kwarg`
  (`tests/test_untrusted_drive_phase4.py:247`) from its hand-listed
  `_TMSG_MODNAMES` to **every bench discovered via the manifest** whose
  `measure()` returns a non-`Fixed` Estimate: assert it exposes a member of
  `SIZING_KWARGS`. Keyed on the *predicate* "shrinkable," not the *list* "tmsg."
  This is the net `22cc56f` should have shipped instead of the per-tmsg list.

- **(c) a sizing budget ⇒ a fundable pool (the `shm_spin_poll` crash).** This is
  the missing third invariant. Two complementary forms, pick by feasibility:
  *the structural one* — route every pool-producing bench through the shared
  `window_pool`/pool-builder of §5.1 that **guarantees** `len(pool) >= 2` for any
  budget the driver may pass (so the estimator's precondition is satisfied by
  construction, not by each bench remembering to guard); *the gate one* — a
  smoke test that drives each model with a **small** budget (the realistic
  allocation `+samples`) and asserts no `measure()` raises a pool-underflow — the
  thing the `~/shm_spin_poll_fail` run would have caught pre-merge. The current
  `r_gen`'s `max(2, reps)` and the tmsg `max(2, iters//_WINDOW)` are the *right
  instinct applied per-bench*; the guard's job is to make it a *property of the
  contract*, so the race-based benches cannot omit it.

### 5.3 Critique of the two pre-floated guards (do not adopt as-stated)

The brief floats two guards. Both are directionally right and both **mis-fire as
literally stated**:

- **"shrinkable Estimate + no recognized sizing kwarg ⇒ loud failure."**
  *Correct and adoptable* — it is guard (b) above, and it under/over-fires only
  on the `B_op` question, which it handles correctly: `B_op` is `Fixed`, so the
  guard never triggers on it. The one refinement: state it over the **discovered
  corpus** (any non-`Fixed` bench), not over a list, or it inherits the very
  Rule-4 fail-open it is meant to close. As literally floated it is sound; the
  risk is implementing it as another enumeration.

- **"needs_measurement=True + Fixed bench ⇒ loud failure."** *Over-fires — do not
  adopt as stated.* `B_op` is the counterexample: `SERVE_FULL_BUCKET` carries
  `needs_measurement=True` (`leaf_eval_grounding.py:163`) **and**
  `bench_b_op.measure()` returns `pin_estimate` (a `Fixed`,
  `bench_b_op.py:66-72`). This guard would **false-fire on `B_op`** — yet `B_op`
  is the *honest* pin (ADR-0008's neutrality result): its faithful measurement is
  an **un-built** saturated end-to-end rows/forward histogram
  (`bench_b_op.py:11-18`), so a `Fixed` is correct *today*. **`B_op` is the
  correct exemption** — and it shows the guard's discriminator is wrong. The
  honest predicate is not `needs_measurement` (which conflates "wants a
  measurement someday" with "has a runnable bench now") but **"a runnable bench
  exists for this quantity"**: `needs_measurement=True` AND *a runnable bench
  exists* AND the bench returns `Fixed` ⇒ loud. That is the lying-signature the
  cascade actually was (`R_gen`/`g_core`/`LPD`/`tmsg` all had built/runnable
  benches), and it cleanly exempts `B_op` (no runnable bench yet). Equivalently:
  the guard must key on the *existence of the measurement path*, which is exactly
  the single-home fact §5.1 proposes to materialize. So the better framing is to
  *derive* the classification (§5.1) and let the guard assert the derivation held
  — rather than assert a flag-vs-body pair that legitimately disagrees for `B_op`.

---

## 6. Honest scope read — what is fixed, what is still pinned, what is broken

A current inventory of the suite (read from the `feat/issue-control-lab` tree;
all 30 benches surveyed):

- **The `Fixed`-pin class is fixed for the measured quantities that have runnable
  benches.** `R_gen`, `g_core`, `LPD` → shrinkable `QuantileLaw`
  (`bench_r_gen.py:209`, `bench_g_core.py`, `bench_lpd.py`). `iota`/`slope`/
  `t_row`/`t_disp` → shrinkable `RegressionLaw` (`fit_estimate`). All **six**
  `tmsg` benches → shrinkable `QuantileLaw` (none remain `Fixed`):
  `bench_tmsg`, `bench_zmq_baseline_tmsg_us_leaf`, `bench_cpp_inproc_port_tmsg_us_leaf`,
  `bench_futex_wake_tmsg_us_leaf`, `bench_shm_spin_poll_tmsg`,
  `bench_lockfree_mpsc_tmsg`. So **`shm_spin_poll`'s and `lockfree_mpsc`'s
  *tmsg* quantities are reclassified, not pinned** — their `_estimate_from_raw`
  returns `median_estimate` (`bench_shm_spin_poll_tmsg.py:156`,
  `bench_lockfree_mpsc_tmsg.py:165`) and they carry the `max(2, iters//_WINDOW)`
  pool guard, so they do **not** hit the §4 underflow. `7fbc352` independently
  records this reclassification.

- **The two remaining `Fixed` pins are correct.** `bench_b_op` (declared-spread
  prior, un-built faithful measurement — the honest ADR-0008 exemption) and
  `bench_n_gen` (a true DEGENERATE constant, `constant=True`). Neither is in the
  class.

- **`shm_spin_poll` is runtime-broken at small budgets** — but via its **wakeup**
  bench, not its tmsg bench (§4). The same latent underflow exists in the other
  **race-based wakeup** benches (`futex_wake`, `zmq_baseline`,
  `cpp_inproc_port`, `lockfree_mpsc` wakeups) by identical construction; they
  have not been *observed* to crash only because no captured small-budget drive
  has hit them yet. This is the unfixed tail of the class: the pin layer and the
  sizing layer are closed; the **pool-cardinality layer** is open for the
  race-based sub-family — and `7fbc352` confirms the maintainer is tracking
  exactly this defect as still-open.

- **`lockfree_mpsc`** is, on the evidence read, **not runtime-broken in the
  captured material** — its tmsg is reclassified and window-guarded, and no
  `lockfree_mpsc` failure file was provided. But its **wakeup** bench
  (`bench_lockfree_mpsc_wakeup`) shares the race-based shape, so it carries the
  same latent §4 underflow; treat it as *latently* exposed, not *confirmed*
  broken (claims-measured-vs-interpreted: I observed the shm crash, I infer the
  mpsc/futex/zmq/cpp wakeup exposure from the shared construction, and mark the
  inference provisional pending a run).

This all fits the root-cause story: the cascade is **one** un-homed contract
(measured⇒shrinkable⇒sizable⇒fundable-pool), peeled one layer at a time by
per-instance fixes, with the deepest layer (fundable-pool for race-based benches)
still open.

---

## 7. Where I agree and disagree with the framings

- **The maintainer's hypothesis ("massive ADR-0011 + ADR-0012, primarily
  ADR-0011, sourced in ADR-0012"): confirmed, refined.** Confirmed: the
  *recurrence* is the ADR-0011 failure (no class mechanism; Rule 4 fail-open
  guards), and it is *sourced* in ADR-0012 (P1 the un-homed classification, P8
  the lying `needs_measurement`/`Fixed` signature). Refined on two points: (1)
  the ADR-0012 violation is not abstract — it is concretely the **migration
  templating ≈30 copies** (cancer D) *inside* the commit curing cancer G, so the
  P1 violation was *manufactured*, not inherited; (2) "primarily ADR-0011" is
  right for the *snowball* but the *originating* commit `5eb1f8e` is primarily an
  ADR-0012 P1 event — the ADR-0011 failure is what turned one bad commit into a
  seven-commit cascade.

- **The brief's framing ("fixed reactively, one quantity at a time; the class was
  never addressed as a class"): confirmed verbatim by the guard landscape**
  (§3.3) — eight per-instance test modules, a hand-listed tmsg sizing test, zero
  corpus-level class invariants. The brief's instinct that this is one class is
  exactly right; the evidence is that even the *tests written to prevent
  recurrence* quantify over the instance.

- **A process observation (not an ADR violation, but causal):** the
  Opus-plan/Sonnet-fleet/Opus-verify method that "applied the template to 26"
  (`5eb1f8e`) is the mechanism by which the punt reached ≈30 files in one commit.
  The fleet faithfully *propagated a shape*; propagating a shape is precisely what
  P1 forbids when the shape encodes a per-instance decision (measured vs pinned).
  The lesson is not "don't fleet" — it is "fleet the *derivation*, never the
  *decision*": had the classification been derived from one home (§5.1) before the
  fleet ran, there would have been no per-bench punt for the fleet to copy.

---

## 8. Recommendations, ranked

1. **Single-home the measured-vs-pinned classification** (§5.1) — the one change
   that dissolves the originating P1/P8 defect. Highest leverage; everything else
   is a backstop for it.
2. **Guard (c): a sizing budget ⇒ a fundable pool**, via a shared window/pool
   builder owning the `len(pool) >= 2` guarantee (§5.2c) — the change that closes
   the open `shm_spin_poll`/race-wakeup tail (§4, §6) and the only guard that
   addresses the layer still broken now.
3. **Guard (b): shrinkable ⇒ sizable over the discovered corpus** (§5.2b) —
   adopt the brief's first floated guard, but keyed on the predicate, not a list.
4. **Do not adopt the second floated guard as stated** — re-key it from
   `needs_measurement` to "a runnable bench exists," so it stops false-firing on
   `B_op` (§5.3), or fold it into guard (a) as an assertion that §5.1's derivation
   held.
5. **Factor the duplicated idioms** (the six window loops, the three
   sizing-kwarg-introspection copies) into `bench_common` (§5.1) — removes the
   cancer-D surface the cascade rode on, and is where guard (c) naturally lives.

---

## 9. Caveats (claims-measured-vs-interpreted)

- The `~/shm_spin_poll_fail` crash was **observed**; the identical latent
  underflow in the `futex_wake`/`zmq_baseline`/`cpp_inproc_port`/`lockfree_mpsc`
  **wakeup** benches is **inferred** from shared construction (§4, §6), not run —
  marked provisional.
- I did not re-run any bench or the driver (read-only commission;
  timing-sensitive sole-workload benches). All "what `measure()` returns" claims
  are read from the bench source on `feat/issue-control-lab`, cross-checked
  against the design note's §0 executed-number changelog and the `7fbc352`
  amendment.
- This RCA's git worktree predates the analysed suite (see the header provenance
  note); the analysis was performed against the `feat/issue-control-lab` tree, to
  which all line citations are anchored.
- The cadence/timeline (§3.1) is from the git log on `feat/issue-control-lab`;
  the times are commit timestamps, which order the work but not necessarily the
  wall-clock of discovery.
- This note is a point-in-time RCA (ADR-0005 Rule 8): it is not retro-edited; a
  later fix that closes guard (c) should append, not rewrite.

---

## Addendum — 2026-06-22: recommendation #5 + guard (c) landed (ADR-0005 Rule 8 append)

The fixes the original RCA recommended have since landed on `feat/issue-control-lab`;
recorded here by dated append (not a rewrite of the point-in-time analysis above). Two
arcs, both implementing §5.1's structural move ("factor the duplicated idioms" + a shared
pool builder that owns `len(pool) >= 2`):

- **The race family (the CRASH half of guard (c)) — `4f81bac`, `eb760ad`.** `bench_common.collect_pool`
  floors the 4 race-based wakeup collectors (shm_spin_poll, futex_wake, lockfree_mpsc,
  cpp_inproc_port) at `>= min_readings` by RE-RUNNING the batch at growing effort (the floor
  binds on readings COLLECTED, the count a race collector cannot promise — §4). This closes
  the open `~/shm_spin_poll_fail` tail (§4, §6) for the whole race sub-family at once, and the
  class-level discovery guard `test_every_race_based_collector_bench_uses_the_pool_floor`
  (keyed on the `Thread`-in-`_measure_raw` predicate, ADR-0011 Rule 4) is the structural net.

- **The deterministic family (recommendation #5, the DRY half) — `453411f`, `ee9cfe0`.**
  `bench_common.window_pool(measure_window, *, name, count, min_windows=2)` is the deterministic
  COUNTERPART to `collect_pool` (§5.1's `window_pool` proposal, materialized): the
  `for _ in range(N): pool.append(measure_one_window())` idiom — the audit's cancer D — now has
  ONE home, the per-window measurement injected as a closure. Because a window loop's reading
  count is KNOWN (= the budget, one deterministic reading per window), there is nothing to
  retry; the helper instead owns the `>= 2` floor STRUCTURALLY (`len(pool) >= min_windows` by
  construction), making each deterministic bench explicitly safe at a tiny budget rather than
  leaning on the driver's `max(2,..)` (untrusted_drive `_make_measurer`). 12 deterministic
  window benches were migrated: the single-counter loops (tau_io, cpp_inproc_port_gather,
  futex_wake/lockfree_mpsc/shm_spin_poll req_drain/gather, zmq_baseline_tau_io,
  zmq_baseline_wakeup) and 5 of the 6 tmsg loops (tmsg, zmq_baseline/futex_wake/lockfree_mpsc/
  shm_spin_poll tmsg). At `count >= 2` the migration is a pure refactor (ADR-0009 behavioral
  equivalence: same closure body, same dict); the only change is the floor at a tiny budget.

- **What §5.1's "six window loops" list did NOT fully cover (honest scope).** Two
  deterministic benches were deliberately left un-migrated, each a documented quirk (not a
  silent skip): (1) the **4 two-pool tau_io benches** (cpp_inproc_port/futex_wake/lockfree_mpsc/
  shm_spin_poll `_tau_io`) time TWO arms in lockstep per cycle (>1 reading/window sharing one
  cache state) — `window_pool` returns one list, and forcing it would break the lockstep or
  distort the contract; (2) **`bench_cpp_inproc_port_tmsg_us_leaf`** indexes its per-window
  body by the window number (`slot = (w*window+j) % 1024`) and uses `max(1,..)` not `max(2,..)`,
  so it does not fit the no-arg `measure_window` closure without mutable state. These keep their
  own loops; they floor via their own `max(2,..)`/`max(1,..)` or the driver's `max(2,..)`.

- **The guards (§5.2), as landed.** Guard (b) shrinkable⇒sizable over the corpus is
  `test_every_shrinkable_bench_is_sizable_by_the_driver` (`ceb233b`, superseding the tmsg
  hand-list). Guard (a)/§5.1's single-home is RCA fix #1 (`0cfae7c`): `leaf_eval_grounding.Estimability`
  (CONSTANT/MEASURED/PRIOR) is the single home, `test_grounded_estimability_agrees_with_the_bench_body`
  the net — re-keyed from `needs_measurement` to the MEASURED-vs-PRIOR split exactly as §5.3
  prescribed (so it does not over-fire on `B_op`). Guard (c)'s STRUCTURAL form (a shared pool
  builder guaranteeing `len(pool) >= 2`) is now realized for BOTH families (`collect_pool` for
  the race sub-family, `window_pool` for the deterministic sub-family), each with a run-free
  unit test (`test_bench_common_collect_pool`, `test_bench_common_window_pool`). A separate
  "every deterministic window bench must call `window_pool`" discovery guard was deliberately
  NOT minted: the 2 legitimately-un-migrated deterministic benches above have no clean
  structural predicate separating them from the migrated ones, so such a guard would require an
  exemption ENUMERATION — the very ADR-0011 Rule 4 fail-open this RCA names as the smoking gun.
  The honest ADR-0011 Rule 1 level there is the helper-owns-the-floor structural guarantee plus
  its unit test, not a decaying enumeration.
