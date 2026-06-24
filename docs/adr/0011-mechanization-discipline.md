# ADR-0011: Mechanization Discipline

- **Status:** Proposed
- **Genre:** Tenet (cross-cutting corrective-design discipline) — the eighth
  tenet. Rule 1 is the enforcement register of the
  ADR-0002 / ADR-0008 / ADR-0009 unsubstantiated-claim family (an enforcement
  level is a claim about a discipline, and it must be declared, not implied);
  Rules 2–4 are corrective-design protocol adjacent to that family.
- **Date:** 2026-06-15
- **Provenance:** Transferred from the LengYue ADR corpus. The tenet
  (disciplines declare their enforcement surface; recurrence converts to a
  mechanism, not more prose; nets quantify over the class, not the instance)
  is universal. LengYue's substrate was its own RCA and lint-adoption history;
  chocofarm's substrate is the 2026-06-15 architectural audit, whose entire
  diagnosis is the chocofarm form of LengYue's "prose disciplines decay,
  mechanisms stick." The mechanism instances are re-derived as chocofarm's
  real ones — the `FeatureLayout` descriptor, the equivalence tests, the
  hp-registry schema constraints.
- **Scope:** Corrective design and discipline authoring across the
  `chocofarm/` package and the docs corpus — the moments when a discipline is
  authored or amended, and when a corrective responds to a recurrence.

## Context

chocofarm's characteristic failure mode is the **invisible-at-authoring,
visible-only-in-aggregate defect**, against which policy enforced by one
person's attention and memory is structurally weak — only mechanical nets
help. The 2026-06-15 architectural audit (`docs/notes/audit/`) is the
chocofarm proof of exactly this, from both directions:

- **The audit's anti-pattern G ("load-bearing knowledge offloaded to prose
  the code cannot enforce").** `ADR-0002` was cited 16 times as a binding
  convention with no registry to look it up in; a design doc was the de-facto
  spec while three of its specifics were STALE in the code implementing their
  successors; `consult-002 §4` was a dangling pointer to the simulation's
  heart. Prose disciplines decayed exactly as LengYue's RCA found.
- **The audit's L3 ("duplicated knowledge is a time-bomb whose fuse is the
  next edit").** The reference-rate anchor *already drifted* (`0.0941` vs
  `0.094`) — the fuse already lit once. A prose "keep these in sync" note
  could not have caught it; only a single owner (`BeliefRefs(env)`) can.
- **The audit's L2 ("the proof a codebase *can* do it right is the indictment
  when it doesn't").** `feature_dim(env)` (derive, zero drift) sits in the
  same package as three hardcoded reference constants (duplicate, already
  drifted). The mechanism works where it is applied; the rot is where it is
  not.

The tenet+mechanism pairing — not the describing document alone — is what
arrests recurrence. chocofarm has a worked proof: the feature-layout
triplication (the audit's sharpest landmine, a three-writer SSOT violation
that a reorder would silently mislabel) was converted from a prose hazard into
a mechanism — `FeatureLayout` (`az/features.py`), a single ordered block table
that the three former writers now read **by name**, with a fail-loud
partition check (`ADR-0002`) that the blocks contiguously cover `[0, dim)`.
The describing record (the audit) named the hazard; the mechanism removed it.

## Decision

We adopt **Mechanization Discipline** as a codebase-wide tenet, in four rules.

### Rule 1 — Disciplines declare their enforcement surface

Every discipline-stating rule — an ADR rule, a CLAUDE.md convention, a
docstring contract — names how it is enforced, against this vocabulary
(related explicitly to ADR-0002's loudness hierarchy; the choice among them is
part of the rule's meaning):

- **construction/import-time** (a raise at setup; a strict schema decode — the
  hp registry's `decode_config`);
- **test/CI gate** (a test at `assert` strength — the jax/numpy equivalence
  test, the scenario-validation tests, the deadlock test);
- **write-time data constraint** (a schema/dataclass invariant that refuses a
  malformed write — `hp/schema.py`'s `check_invariants`);
- **run-time invariant** (a structural check at use — `FeatureLayout`'s
  contiguous-partition assertion; the f32-cache identity check of ADR-0001);
- **review-only**.

Review-only is legitimate but presumptively decaying — declaring it makes that
a visible, challengeable choice. Across this corpus, the "discipline is policy,
not mechanism" Negative bullets in ADR-0003 through ADR-0009 are this rule's
pre-existing instances: each declares review-only and names the trigger that
would mechanize it.

*Neutral scoping (no retroactive sweep):* declarations bind when a discipline
is authored or amended and at corrective-design moments; existing rules
retrofit on touch (ADR-0004 / ADR-0006).

### Rule 2 — Recurrence converts to mechanism, not more prose

When a failure shape recurs after its describing record exists, the
corrective names the mechanism it pairs with the rule — at the strongest
*feasible and proportionate* surface in Rule 1's vocabulary — or carries an
explicit policy-only admission and the trigger that would change it. "Tenet +
mechanism arrests recurrence; a describing-only document does not" is the
cited rationale (the `FeatureLayout` worked proof; the audit's L1/L2), not an
unconditional build-a-gate mandate. The audit's roadmap is the chocofarm
register of this rule: the reference-rate drift recurs → `BeliefRefs(env)`
(one owner, R3); the belief-mechanics duplication that the bound certifies
against → `env.restrict` sharing one implementation (R8); the `id(env)`-keyed
cache hazard → `env.slot_tables` owned on the object (R9). Each is a mechanism
the corrective names, not a prose "be careful."

### Rule 3 — Mechanisms adopt measure-first

A mechanism is adopted against a measured baseline, not an assumed one. Before
a check goes to `error`/`assert` strength, the existing tree is measured (the
audit ran the live env to find `realizable_static = 0.08553` /
`clairvoyant = 0.14537` and so demoted a "your metric is wrong now" claim to
"latent, one value-vector change away" — measure-first caught the
overstatement). A check at full strength lands only on a zero-or-fully-triaged
baseline; the equivalence tests are the worked instance — `ABS_TOL = 1e-4` was
chosen "comfortably above the observed" float error, a measured threshold, not
a guessed one. Where a paid-for defect exists, probe-verify the net fires on
its literal shape (the audit reproduced `max|Δp| = 0.0082` to confirm the
stale-weight hazard the f32-cache invariant guards).

### Rule 4 — Nets quantify over the class, not the instance

Enumerations of instances fail open at the next instance. A net keys on a
structural slot, a name/shape predicate, or a derived-from-one-source
invariant. The chocofarm worked instances:

- **`FeatureLayout`** keys on the ordered block *table* (the class of feature
  blocks), so a new block is one table entry and every consumer slices by name
  — a reorder edits one structure and cannot silently mislabel. The old
  three-writer enumeration failed open at exactly the next reorder.
- **The param-registry-driven net serializer** (`parallel.py`'s
  `pack_net`/`unpack_net`) enumerates the weight set from the net's own
  `_params()`, so an optional residual block transports with no second edit
  site — derive-don't-duplicate, a net over the class of params.
- **The contiguous-partition assertion** in `FeatureLayout` quantifies over
  *every* position (no gap, no overlap), not over a list of expected blocks.

Conversely, the audit's `it + 1_000_000` version offset is the failure this
rule names — a magic disambiguator that silently breaks at iters ≥ 1e6,
because it enumerated a case rather than namespacing the class (the fix:
namespace weight keys by `(run, phase, version)`, R14).

## Self-application

This tenet binds at corrective-design moments — audit recommendations, a new
mechanism's adoption, an ADR amendment — a handful of high-attention events,
not the per-edit regime where prose decays. Its own Rule-1 declaration:
**review-only, with the audit as the absence-detector.** chocofarm has no CI
sweep that detects a discipline-stating rule lacking an enforcement
declaration; the architectural audit is that detector, run on demand. The
protection this tenet offers is the mechanisms it mints (`FeatureLayout`,
`BeliefRefs`, the equivalence tests), not its own prose — the tenet expects its
own prose to be exactly as weak as Rule 1 says, which is why it names its
mechanisms rather than relying on the rule text.

## Consequences

**Positive.** Enforcement levels become legible per discipline — a reader (and
a future fork author, who inherits the tree's mechanisms but not the
maintainer's memory) can distinguish mechanism-policed from memory-policed
without archaeology. Correctives stop defaulting to the measured-decaying
prose form; the audit's R-series is overwhelmingly mechanisms, not notes.

**Negative.** Per-corrective authoring overhead (the assessment +
declaration); the risk of cargo-cult mechanisms is real — a check at full
strength on an un-triaged baseline is worse than none (Rule 3 is the
counterweight). chocofarm has no automated enforcement of this tenet itself;
it is review-and-audit-policed.

**Neutral.** No retroactive sweep (Rule 1's scoping clause); existing
mechanisms are not re-litigated.

## Revisit when…

1. A mechanism is retracted on false-positive economics — record the
   retraction here; Rule 3's calibration may need a rule.
2. A doc-side resolution check (does every cited path resolve?) matures — the
   advisory rung gains a member; reassess the vocabulary (this is also
   ADR-0005's Revisit #2).
3. A second OR/game instance adopts the corpus (ADR-0003's trigger) — the
   enforcement-surface declarations are the transfer manifest; check each
   discipline's mechanism survived the re-instantiation.

## Related

- **ADR-0002 (fail loudly).** The Rule-1 vocabulary maps onto ADR-0002's
  loudness hierarchy at the enforcement level. The `FeatureLayout` partition
  check and the registry strict decode are fail-loud mechanisms.
- **ADR-0008 (classification discipline).** Rule 1's vocabulary is a closed
  vocabulary under ADR-0008's care; extending it follows the
  revise-don't-fuzzy-match discipline.
- **ADR-0009 (perf investigation discipline).** The sibling per-domain
  instance of the unsubstantiated-claim family; Rule 3's measured baselines
  are the enforcement-domain analog of its captured benches.
- **The 2026-06-15 architectural audit** — the chocofarm RCA this tenet
  answers. Its R-series roadmap is the worked register of Rule 2
  (recurrence → mechanism); `FeatureLayout` (R6) is the worked proof of the
  tenet+mechanism pairing; its §5 measure-first deflation is Rule 3.

## Amendments

### 2026-06-24 — Empirical readings carry their code state (the commit-stamp net)

A "+31%" throughput win (the greedy vs round-sync episodic driver) banked from a
single session failed to reproduce, and could not be pinned to a commit, on a
controlled re-measurement — an attributed reading with no record of the code that
produced it is **unattributable by construction**. This is the
invisible-at-authoring/visible-only-in-aggregate defect this tenet names, in the
empirical-measurement domain: the per-reading provenance was offloaded to the
operator's memory (anti-pattern G), and it decayed across the very first session
boundary.

Per **Rule 2** (recurrence → mechanism, not more prose) and **Rule 4** (a net
quantifies over the *class*, not the instance): the measuring harness itself
emits, on **every** reading, the git commit short-hash + tree state
(`clean | DIRTY`) of the checkout that produced it. A `DIRTY` tree marks a
non-reproducible artifact — the producer binary / harness may not match `HEAD`,
so the number is provisional until committed. The net keys on the class of *all
readings* (the harness stamps unconditionally), not on an enumeration of the ones
someone remembered to label.

- **Enforcement surface (Rule 1):** run-time, at the harness — the stamp is
  emitted with the number, not left to review or recall.
- **Measure-first (Rule 3):** the trigger was itself a measurement failure (an
  unreproducible bench delta), and the mechanism is the lightest proportionate
  one (two `git` reads), not a heavier provenance system.
- **Worked instances (one home, ADR-0012 P1):** `throughput-lab/harness/code_stamp.py`
  is the single Python home, imported by `coalesce_sweep.py`, `topology_sweep.py`,
  and `cpp/stage_a/overcommit_sweep.py`; `throughput-lab/harness/episodic_dps.sh`
  mirrors the same two `git` invocations inline.
- **Pairs with ADR-0009** (perf-investigation discipline): a captured bench number
  is now code-addressable — the sibling of ADR-0009's captured-bench requirement.

Provenance: the maintainer's contribution, 2026-06-24, during the throughput-lab
driver-attribution work.

### 2026-06-24 — The interpretation/belief layer (the Witness chain, mechanized)

A measurement is an immutable fact; an **interpretation** of it (the reading that
motivates the next code change) is a different kind of thing — mutable, frequently
wrong, and spanning a *set* of readings. The throughput-lab journal recorded its
interpretations as prose **Witness/Correction** entries (Witness 1: "+31% clean
driver win" → retracted → Witness 2: "regime-dependent +15%" → Witness 3: the full
2× attribution). That prose chain is exactly the load-bearing knowledge offloaded to
a form the code cannot enforce (anti-pattern G), and the wrong belief that motivated
banking the wrong default is the cost.

Per **Rule 2** (recurrence → mechanism) and **Rule 4** (a net over the *class* of all
interpretations, not the remembered ones): the belief layer moves into a queryable,
append-only store — `tlab_finding` (`throughput-lab/harness/exp_db.py`), a SEPARATE
table from the `tlab_reading` measurements, so the conflation the project has been
burned by (a reading-*of* the data recorded as the data) is **structurally
unrepresentable** (composing with ADR-0000). A finding carries `motivation` +
`interpretation`, a `status` in the closed vocabulary `{provisional, confirmed,
retracted}` (ADR-0008), the commit the belief was formed against (the commit-stamp
amendment above), and a `supersedes` link to the finding it corrects — the
Witness→Correction step, append-only (ADR-0005: the prior belief is never rewritten;
the current belief on a scope is the one nothing supersedes).

- **Enforcement surface (Rule 1):** write-time data constraint (the `CHECK` enum, the
  NOT-NULL interpretation, the immutable supersede-chain) + the discipline that
  *measurements auto-record but findings are deliberately authored* — an
  interpretation is a conscious, attributable act, not a side effect of a run.
- **Worked instance:** the Witness 1→2→3 chain is backfilled into the store, so the
  retracted "+31%" is itself queryable (`exp_db.py --findings`).
- **Pairs with ADR-0009** (the measured-vs-interpreted bar, amended there same day).

## License

Public Domain (The Unlicense).
