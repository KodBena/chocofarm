# The throughput-lab pre-registration layer: criterion-before-data, mechanized

**Date:** 2026-06-24. **Branch:** `worktree-agent-a469026fa52c93e30` (off `feat/tlab-real-generators`).
**Status:** a design note (ADR-0005 — a dated, slowly-aging account of *why this shape*, so the rationale
survives the code). The implementation is `throughput-lab/harness/exp_db.py` (the `tlab_prereg` /
`tlab_prereg_conclusion` tables + the `Criterion`/`Bin`/`PreReg` types + the `record_prereg` / `conclude_prereg`
/ `abandon_prereg` / `preregs` API + the `--record-prereg` / `--conclude-prereg` / `--abandon-prereg` /
`--preregs` CLI). This note is the *why*; the docstrings are the *what*.

---

## 0. The discipline being mechanized

Across the performance journey (`tlab-performance-journey-2026-06-24.md`) the project was repeatedly burned
by the same shape of error: a verdict criterion invented — or quietly bent — **after** the data was seen, so
any outcome could be narrated as decisive. Lessons 1 and 3 of that journey ("don't *infer* a regime, measure
it"; "separate measured from interpreted, mark conjecture") are the human-discipline statements of it. The
`tlab_finding` layer already mechanizes *measurement ⊥ interpretation* (a reading is fact; an interpretation
is an authored, supersedable belief). But a finding can still *assert* "this result was decisive" as prose a
reader must simply believe.

This session adopted, manually, a **pre-registration** discipline (witnessed in `tlab_finding` #10 and #11):
before running an experiment, register the question, the single decision metric, the **quantified**
decisiveness criterion (outcome bins with numeric thresholds), the plan, and the arithmetic that justifies why
those thresholds discriminate; after running, judge the measured value against the *pre-registered* criterion
— which bin did it land in, with what margin, or "criterion not met → escalate (ADR-0014)".

The accountability property we want made **structural rather than rhetorical**: the criterion is fixed and
immutable *before any data exists*, so a result can never be retro-fitted to a narrative, and "this was
decisive" becomes a **checkable claim** (did the value land in a pre-declared terminal bin, with margin?)
rather than an assertion.

---

## 1. The core design decision: the criterion is a *typed structure the code evaluates*, not prose

This is the ADR-0012 move (the typed signature is the single source of truth) applied to the verdict rule. The
defect-class "a result was retro-fitted to a narrative" is, at root, a *type not yet encoded*: as long as the
criterion lives as a sentence a human interprets, "did the result meet it?" is a human judgement made *after*
seeing the result — exactly where the bending happens.

So the criterion is a `Criterion`: an **ordered partition of one decision metric's value-line into named
outcome `Bin`s**, each `{name, lo, hi, verdict, decisive}` over a half-open interval `[lo, hi)`. The code
*evaluates* it:

```
criterion.evaluate(measured_value) -> BinHit(bin_name, verdict, decisive, margin)
```

`evaluate` returns which pre-declared bin the value fell in, the **margin** to the nearest finite threshold
(how decisively it landed), and whether that bin is **terminal-decisive**. "Did the result meet the criterion?"
is then literally `criterion.evaluate(v).decisive` — a mechanical check, not a reading.

### 1.1 The partition invariant is the whole point

`Criterion.__post_init__` validates that the bins **tile `(-inf, +inf)` edge-to-edge with no gap and no
overlap** (the sorted bins' `lo`/`hi` must chain: first `lo == -inf`, last `hi == +inf`, each `a.hi == b.lo`).
This is not bookkeeping — it is the invariant that makes `evaluate` *total* and forbids the silent ambiguity
this layer exists to close:

- A **gap** between two thresholds would let a measured value fall through to *no verdict* — the silent hole.
  By forcing a partition, a "not-decisive → escalate" outcome must be an **explicit `decisive=False` bin**
  (e.g. the real `[78, 90)` "escalate" band), never an accidental hole. The ambiguous zone is *declared*, with
  its own name and verdict prose, before any data — so "we couldn't decide" is itself a pre-registered,
  honoured outcome, not an after-the-fact shrug.
- An **overlap** would make the verdict order-dependent (which bin "wins"?). Forbidden at construction.

A gap or overlap is an `ADR-0002` loud `ValueError` at *construction* time — i.e. before the criterion can ever
be stored, let alone judged against. The bad criterion is unrepresentable (ADR-0000), not caught downstream.

### 1.2 The convenience builder mirrors how humans state it

Humans state these as cut points ("≤78 ⇒ A; 78–90 ⇒ escalate; ≥90 ⇒ B"). `thresholds_criterion(metric, cuts)`
takes that shape — a sorted list of `(upper_bound, name, verdict, decisive)` — and builds the contiguous
partition `[-inf, cut0), [cut0, cut1), …, [cut_last, +inf)`. It cannot produce a gap by construction (each
bin's `lo` is the previous bin's `hi`), so the common case is gap-safe *by shape*, and the general `Criterion`
constructor catches the rest.

### 1.3 Worked instance (the real finding-#10 criterion)

The session's real A/B (`tlab_finding` #10) used: `util ≥90 ⇒ producer-bound`; `≤78 ⇒ server-loop ceiling`;
`78–90 ⇒ not decisive → escalate`. Encoded and evaluated at the measured `68.9`:

```
68.9 -> bin 'server-loop-ceiling'  decisive=True   margin=9.1   => DECISIVE
84.0 -> bin 'escalate'             decisive=False  margin=6     => AMBIGUOUS (escalate, ADR-0014)
95.0 -> bin 'producer-bound'       decisive=True   margin=5     => DECISIVE
```

`68.9 → server-loop-ceiling, margin 9.1` reproduces finding #10's hand-narrated "bin B with 9pt margin"
exactly — now computed, stored, and queryable rather than asserted.

---

## 2. The table shape: two tables mirroring `reading ⊥ finding`

The immutability/accountability guarantee turns on the criterion being fixed *before* the data and *never
rewritten* by the result. So, exactly as `tlab_finding` is append-only (a corrected belief is a *new* row that
supersedes, never a mutation), the pre-registration is split across two tables:

### `tlab_prereg` — the immutable registration (criterion-before-data)

One row per registered experiment. Columns: `prereg_key` (a stable human slug, **UNIQUE**), the code stamp
(`git_commit`/`git_tree`/`host` — ADR-0011, the state the criterion was *declared* against, necessarily before
the result's commit), `question`, `metric` (the single decision metric), `criterion` (the typed `Criterion` as
jsonb), `rationale` (the justifying arithmetic), `method` (the plan), `refs`, `notes`.

Two structural locks make it immutable-by-design:

1. **`status` is pinned to `'registered'` by a CHECK** (`status IN ('registered')`). The row physically cannot
   carry a verdict status. The transition to concluded/abandoned lives in a *different* table. There is no
   column on this row a result could write into.
2. **`prereg_key` is UNIQUE, and `record_prereg` plain-`INSERT`s** (no `ON CONFLICT DO NOTHING`). Re-registering
   the same slug is a loud `UniqueViolation`, not a silent second criterion swapped in after seeing data. One
   experiment, one criterion.

### `tlab_prereg_conclusion` — the separate, later verdict

One row per *concluded* pre-registration, linked by `prereg_id` (**UNIQUE** — see §3). Columns: the code stamp
(the state the *verdict* was reached against — a second, later stamp), `outcome` (closed vocabulary
`{decisive, ambiguous, abandoned}`, ADR-0008), `observed` (the measured value), `bin_name` / `bin_verdict` /
`margin` (**computed by `Criterion.evaluate`** at conclusion time and stored, so the verdict is queryable, not
re-derived by every reader), and `resolved_by_reading` / `resolved_by_finding` (FKs closing the loop, §4).

The verdict is reached by `conclude_prereg(conn, prereg_id, observed)`: it **loads the frozen criterion, runs
`evaluate`, and writes a new conclusion row**. It never touches the `tlab_prereg` row. The criterion is read
and judged, never rewritten — the same amend-by-append discipline as a finding supersede.

### Why not one table with nullable verdict columns?

A single table would mean the verdict columns sit on the same row as the criterion — and "immutable" would then
be a *convention* (don't UPDATE these columns) rather than a *structure*. The two-table split makes the
criterion row have **no verdict column to write**, so criterion-before-data is enforced by the schema, not by
discipline. This is the same reason `tlab_reading` and `tlab_finding` are separate tables and not one table
with a nullable `interpretation` — the conflation must be *unrepresentable*, not merely discouraged.

### Why an additive layer in `exp_db.py`, not a sibling module?

`exp_db.py`'s own header argues it is "the ONE owner (ADR-0012 P3 one-owner) of the lab's experiment-DB I/O."
A sibling module would split that ownership — two homes for the `throughput_research` wire, two `ensure_schema`
seams a harness must remember to call, two connection idioms to keep in step. The pre-registration layer is the
*same* lab's persistence; it belongs behind the same one owner, sharing `connect`/`ensure_schema`/`code_stamp`/
the `_safe` fallback dir. The addition is purely additive (`CREATE TABLE IF NOT EXISTS`, new functions, new CLI
flags); it touches no `tlab_reading`/`tlab_finding` semantics and no existing CLI behaviour.

---

## 3. The status lifecycle

```
                          conclude_prereg(observed)
                         ┌──────────────────────────► decisive   (value in a terminal bin)
   record_prereg         │                          └ ambiguous  (value in a decisive=False bin → escalate)
   ─────────────► registered
                         │  abandon_prereg(note)
                         └──────────────────────────► abandoned   (called off, no measured verdict)
```

- `registered` is the only status ever written to `tlab_prereg` (CHECK-pinned). It is the live state — the
  experiment awaits a verdict.
- The three terminal states are `outcome` values on the *conclusion* row (CHECK-pinned closed vocabulary,
  ADR-0008). `decisive`/`ambiguous` are the two data-bearing verdicts (which one is **decided mechanically** by
  `evaluate` — the author does not get to *choose* "decisive"; the criterion and the value decide it).
  `abandoned` is the no-data leg (the instrument proved infeasible, the design was superseded), recorded with a
  `note` saying why (an unexplained abandonment is suspect).
- **A pre-registration concludes AT MOST ONCE.** `tlab_prereg_conclusion.prereg_id` is UNIQUE. A second verdict
  on one immutable criterion is a *contradiction, not an amendment* — if a re-run is warranted, that is a
  **new** pre-registration with a **new** criterion (which may, honestly, be a *different* criterion — but then
  the change of criterion is itself a visible, separately-registered act, not a silent re-judgement of the old
  one). This is the single most important guard: it is what makes "you cannot move the goalposts after seeing
  the data" a database constraint rather than a hope.

"Is this experiment still open?" is `LEFT JOIN tlab_prereg_conclusion … WHERE conclusion_id IS NULL` — the
`preregs(open_only=True)` query. No status field to drift out of sync; the open/closed fact is *derived* from
the presence of a conclusion row (the same derive-don't-duplicate posture as `tlab_finding`'s "current = nothing
supersedes it").

---

## 4. How it composes with `tlab_reading` and `tlab_finding`

The three layers form a chain, each a different *kind* of thing:

```
tlab_prereg        (a QUESTION + a frozen verdict CRITERION, before data)
      │  conclude_prereg(observed)
      ▼
tlab_prereg_conclusion  ──resolved_by_reading──►  tlab_reading   (the MEASUREMENT the verdict rests on)
      │                 ──resolved_by_finding──►  tlab_finding   (the INTERPRETATION of the verdict)
      ▼
   outcome ∈ {decisive, ambiguous, abandoned}, computed by Criterion.evaluate
```

- A conclusion **links the resolving `tlab_reading`** (the measurement the `observed` value came from) and/or
  **the resolving `tlab_finding`** (the authored interpretation that the verdict motivates). Both are nullable
  FKs, verified to exist at conclusion time (a dangling link is a loud error, ADR-0002), closing the loop
  *prereg → reading → finding* that the journey doc narrated by hand.
- The layers stay orthogonal: a **reading** is immutable fact (auto-recorded); a **finding** is a supersedable
  belief (deliberately authored); a **pre-registration** is an immutable *commitment to a verdict rule* whose
  conclusion is mechanically computed. The pre-registration does not replace the finding — finding #11 (a
  pre-registered experiment, "2 of 3 predictions VIOLATED") shows the natural workflow: pre-register the
  criterion → run → record the reading(s) → `conclude_prereg` for the mechanical verdict → author a `finding`
  that *interprets* the verdict (and links back via `resolved_by_finding`). The finding still carries the
  nuance and the prose; the pre-registration carries the *accountable, falsifiable core*.

---

## 5. The immutability / accountability guarantee, and how the schema enforces it

The claim "this result was decisive" is accountable iff three things hold, each a *structure* here, not a
convention:

1. **The criterion existed before the data.** Enforced by: the criterion lives on its own row, stamped with the
   `git_commit`/`git_tree` it was declared against (ADR-0011); that row carries no verdict column
   (`status` CHECK-pinned to `registered`); and the conclusion is a *separate* row written later, with its own
   (necessarily later) stamp. A reviewer can read the two stamps and see the criterion's commit precedes the
   verdict's.
2. **The criterion was not bent.** Enforced by: `prereg_key` UNIQUE (no second criterion for one experiment) +
   `record_prereg` plain-INSERT (a duplicate slug raises, never silently overwrites). The criterion jsonb is
   never UPDATEd by any code path here — `conclude_prereg` only `SELECT`s it.
3. **"Decisive" is the criterion-and-value's verdict, not the author's.** Enforced by: `outcome` is *computed*
   by `Criterion.evaluate`, not passed in. The author supplies the measured value; the partition decides the
   bin; the bin's `decisive` flag decides `decisive` vs `ambiguous`. The author cannot type "decisive" for a
   value that landed in the escalate band.

All three were verified against the real `throughput_research` DB (see §7): a second conclusion on one prereg
raises `UniqueViolation`; a re-register of one slug raises `UniqueViolation` (and, through the `_safe` CLI door,
dumps the unsaved criterion under `~/w/vdc` rather than losing it); a value in the escalate band concludes
`ambiguous` regardless of what the author wants it to be.

**Honest limit (read-path immutability).** The guarantees above are guarantees of the *write API and the schema
constraints*. They are not a guarantee against a privileged operator issuing a raw `UPDATE`/`DELETE` on the
table (Postgres has no append-only table type, and the lab DB is TRUST-auth with no row-level security). The
same caveat already applies to `tlab_reading`/`tlab_finding` — the discipline is "no code path mutates; the
constraints catch the *accidental* second criterion/verdict." A determined hand editing SQL directly is out of
scope (and would be visible as a stamp/`created_at` inconsistency). This is named, not papered over.

---

## 6. ADR alignment

- **ADR-0000 (make the bad state unrepresentable).** A gap/overlap criterion, an inverted bin (`lo ≥ hi`), an
  empty bin name/verdict, a non-`Criterion` criterion, and a verdict status on the criterion row are all
  unrepresentable — rejected at construction or by a CHECK, not caught downstream.
- **ADR-0002 (fail loudly).** Empty `prereg_key`/`question`/`rationale`/`metric`, a malformed criterion, a
  dangling `resolved_by_*` FK, a duplicate slug, a second conclusion — all loud errors. The `_safe` variants
  (`record_prereg_safe`) are the *named* exception: loud-but-non-fatal + dump-under-`~/w/vdc`, for the harness
  that pre-registers at run start and must not lose the criterion to a DB blip — exactly mirroring
  `record_reading_safe`'s weighing of ADR-0002 against a long run.
- **ADR-0005 (documentation discipline).** This dated note records the *why*; it is amend-by-append (a later
  design change is a new dated note/amendment, never a silent rewrite). The journey doc's findings #10/#11 are
  left un-retro-edited.
- **ADR-0006 (source-file headers).** No new source file (additive into `exp_db.py`, which already carries its
  header). This note is documentation, not source.
- **ADR-0008 (closed vocabulary).** Both enums — the (degenerate, single-member by design) `tlab_prereg.status`
  and the `tlab_prereg_conclusion.outcome` `{decisive, ambiguous, abandoned}` — are CHECK-pinned in the schema
  and mirrored by the `PREREG_OUTCOMES` tuple in code, the same SSOT-echo pattern as `DRIVERS`/`STATUSES`.
- **ADR-0011 (provenance).** Both the registration and the conclusion carry the `code_stamp` (commit/tree/host);
  a DIRTY tree is recorded as-is, never silently 'clean'. The two stamps witness criterion-before-data.
- **ADR-0012 (the typed signature is the SSOT).** The whole design: the criterion is a typed `Criterion` the
  code evaluates, not prose; `exp_db.py` remains the one owner of the lab's DB I/O; the schema CHECK echoes the
  code's closed vocabulary rather than re-authoring it.
- **ADR-0014 (second opinion when stumped).** The `ambiguous` outcome *is* the structural escalation hook: a
  value in a `decisive=False` band concludes `ambiguous`, and the CLI prints "criterion not met → escalate
  (ADR-0014)". The escalation is a pre-declared, honoured outcome, not an after-the-fact retreat.

---

## 7. Verification (run against the real `throughput_research` DB, 2026-06-24)

- `--ensure-schema` created `tlab_prereg` + `tlab_prereg_conclusion` idempotently (re-runs no-op; existing
  `tlab_config`/`tlab_reading`/`tlab_finding` untouched, `tlab_reading` 45 rows / `tlab_finding` 11 rows
  unchanged after).
- **Decisive round-trip:** registered the real finding-#10 criterion via the CLI, concluded against real
  `tlab_reading` #39 (`server_util_pct=68.9`): `→ bin 'server-loop-ceiling' (margin 9.1) → DECISIVE`.
- **Ambiguous round-trip:** a value of `84.0` concluded `→ bin 'escalate' (margin 6) → AMBIGUOUS (criterion not
  met → escalate, ADR-0014)` — the author could not make it "decisive."
- **Abandoned round-trip:** `--abandon-prereg` with a note recorded `outcome='abandoned'`, observed/bin/margin
  NULL.
- **Accountability guards (all fired):** a second `conclude_prereg` on one prereg → `UniqueViolation`; a
  re-register of one slug → `UniqueViolation` (and the `_safe` CLI door dumped the unsaved criterion under
  `~/w/vdc`, returning non-fatally).
- **Type-level:** `Criterion` rejects gaps and overlaps at construction; `to_json`/`from_json` round-trips the
  ±inf-bounded partition losslessly; `PreReg` rejects an empty rationale.

(All demo rows were deleted after verification; the empty tables remain, ready for real use.)

---

## 8. Tensions and limitations left open honestly

- **One metric per criterion.** A criterion partitions *one* metric's value-line. A genuinely multi-metric
  decisiveness rule (e.g. "decisive iff util ≥90 *and* latency ≤3 ms") is not expressible as a single
  `Criterion`. This is deliberate — the journey's discipline is "the *single* metric the verdict rests on" —
  but it is a real limit: a multi-metric experiment must either pick the one dominant metric or register
  multiple pre-registrations. Encoding a conjunctive/typed multi-metric criterion is a possible future
  extension (a `Criterion` over a tuple metric), left unbuilt rather than speculatively designed.
- **The criterion's *rationale* is still prose.** The arithmetic that justifies *why* the thresholds
  discriminate (`rationale`) is free text — it cannot be mechanically checked (we cannot verify that "78%" is
  the *right* cut). What *is* mechanized is that the cut, once declared, is immutable and the value is judged
  against it without bending. The honesty of the threshold choice still rests on the author and review; the
  layer removes the *retro-fit*, not the *mis-specification*. (Mis-specification is, correctly, the
  `tlab_finding` layer's job — finding #11 superseding #10 is precisely a recorded "the criterion/prediction
  was wrong," done by supersede, not by editing the prereg.)
- **Read-path immutability** is constraint-enforced against accidental double-writes, not against a privileged
  raw `UPDATE`/`DELETE` (§5). Named, not solved.
- **No automatic linkage from a reading to its prereg.** The harness must *choose* to call `conclude_prereg`;
  nothing forces a registered experiment to ever be concluded (it can sit `open` forever). `preregs(open_only)`
  surfaces the stragglers, but closing them is a human act — by design (a conclusion is a deliberate judgement,
  like authoring a finding), but it means an abandoned-in-spirit experiment shows as `open`, not `abandoned`,
  until someone says so.
- **A degenerate single-member `status` CHECK** (`status IN ('registered')`) is, frankly, a slightly unusual
  way to express "this column is always this value." It is chosen over a bare DEFAULT-without-CHECK precisely so
  the *schema* forbids a verdict status ever appearing on the criterion row — making the two-table immutability
  visible at the table definition. An alternative (drop the column entirely, derive "registered" from "no
  conclusion row") is arguably cleaner; the explicit column is kept for read-legibility and for a future where
  a pre-conclusion lifecycle state (e.g. `running`) might be wanted. Flagged as a judgement call a reviewer may
  reasonably revisit.
