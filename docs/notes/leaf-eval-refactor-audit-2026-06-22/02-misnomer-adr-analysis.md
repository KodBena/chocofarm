<!-- docs/notes/leaf-eval-refactor-audit-2026-06-22/02-misnomer-adr-analysis.md — Public Domain (The Unlicense) -->

# 02 — The misnomer as a multi-ADR violation

[← 01 plan vs. result](01-plan-vs-result.md) · [03 independent audit →](03-independent-audit.md)

The thesis under audit, in the maintainer's words: *the misnomer itself leaves a
truckload of ADR violations — ADR-0008 for example, probably ADR-0002 as well.* This
is correct, and the ADR corpus states it more strongly than a first pass did: a fossil
name is **one** defect that trips a documented sibling pair by construction.

## The artifact convicts itself

`neyman_driver.py`'s own module docstring refutes its own name:

- **Line 5:** "The GENERIC, model-agnostic **Neyman** optimal-allocation driver"
- **Lines 54–76 (§6 PHASE 2):** "the diagonal [Neyman] exposition above is now the
  **SPECIAL CASE** … THE ALLOCATION (§2.3). The cost-constrained **c-optimal SOCP** …
  reduces to the closed form `n_i* ∝ √(a_i/c_i)` on the diagonal."

The engine implemented is a **cost-constrained c-optimal experimental design solved as
an SOCP** (`_socp_allocation`, line 798), of which strict Neyman's `√(a_i/c_i)` is only
the diagonal special case, plus a **Clark-1961 kink path** (`alloc/kink.py`) that is
not Neyman at all. The file documents that its headline name is false and keeps the
name. `class NeymanDriver` (line 250) is imported by every model factory — **13
referrers** across the tool.

## ADR-0008 — the cause (squarely the negative register)

ADR-0008's **negative register** governs *extending* a vocabulary: "stale
categorisation left standing is as misleading as a fabricated one — the remedy is to
strip the fossils (mark them dead or remove them), not to leave the reader to guess."
Its worked instance is the `instance.json` fossil arrays: a stale categorisation "left
in the canonical vocabulary, which the next reader reads as authoritative."

`NeymanDriver` **is** that fossil — on the **highest-leverage surface in the tool**
(the core engine class). ADR-0008's **substitution test** (Rule 4) calibrates severity
not to the observed cost but to the worst-case surface the failure shape could land on;
a core engine's name is that worst case. The maintainer-flagged §4 rename to
`alloc/driver` was the *strip*; it did not happen.

**The agent's available defense — and why it is weak.** ADR-0008's
*scheduled-for-revision* exception permits a deferred misfit "if" it is filed visibly
with a named trigger (Rule 3 — "a consult record, an ADR amendment, or at minimum an
inline comment naming the misfit"). The agent partially invoked it: `alloc/__init__.py`
notes the driver is "to become `alloc/driver.py` in a later increment," and the §6
docstring addendum names the misfit. But the invocation is **partial on three counts**:

1. The deferral is filed in a module docstring + commit bodies, **not in `BACKLOG.md`**
   — where the project keeps deferred work, and where this audit confirms no such entry
   exists.
2. The **naming site itself still positively asserts "Neyman"** as the primary identity
   (the docstring *headline*, the class name) rather than carrying a `# TODO: misfit`
   marker — so the fossil is the first thing a reader keys on, the correction buried 50
   lines down.
3. The substitution test puts this on a surface where "defer with a note" is the wrong
   call; "strip" is.

Net: filed visibly enough to show good faith, not enough to satisfy the rule on a
load-bearing surface.

## ADR-0002 — the symptom ("probably," calibrated exactly right)

ADR-0008 names itself, in its own header, the **"Sibling of ADR-0002: same shape of
failure (a category error silently propagating), different intervention point."** So the
question is not *whether* 0002 attaches but *how*:

- **Rule 4 — the lying signature.** ADR-0002 Rule 4: "A config field that the receiver
  cannot honor must not be silently accepted … a seam that looks configured but is dead.
  Honor it or delete it." Its worked instance is the audit's "lying signature" finding
  (`train_epochs(lr, l2)` ignoring its args). A name asserting *Neyman* over an engine
  that runs *c-optimal SOCP + Clark kink* is a lying signature at the **name/type
  register**: a contributor who builds on Neyman semantics (diagonal independence,
  `√(a/c)` allocation) silently gets the general behavior. No channel surfaces the
  mismatch — the **bottom of ADR-0002's loudness hierarchy** ("silent fallback or
  default. Lowest.").

- **The sibling relationship is explicit, both ways.** ADR-0008: "A fuzzy classification
  that slips through becomes the silent symptom ADR-0002 surfaces; this tenet prevents
  the cause. The two compose at different intervention points." ADR-0002's Related
  section returns the reference. So one fossil name is, *by the corpus's own design*,
  an 0008 violation (cause) and an 0002 violation (symptom) at once.

It is "**probably** 0002" — the maintainer's exact hedge — because 0002's center of
gravity is the runtime-exception register; the misnomer attaches through Rule 4 and the
documented sibling relationship, not through a thrown exception. The weighting was right.

## The rest of the truckload — all radiating from the one name

- **ADR-0005 Rule 3 (stale description).** The self-refuting docstring headline is the
  flagship; three bench files (`bench_r_gen.py:71`, `bench_lpd.py:94`, `bench_g_core.py:86`)
  still name the dead `OpenTURNS/` dir in path-walk comments; `alloc/__init__.py`
  narrates the gradient backend as "OpenTURNS … today" after the swap retired it.
- **ADR-0007 / ADR-0012 P3 (god-object).** The rename did not happen *because* the
  driver decomposition did not. `neyman_driver.py` is **1051 lines** carrying six
  concerns (Σ-assembly, the SOCP, the CI multiplier, the `Recommendation` formatter,
  `run()`, the Estimate seam). The fossil name is the **visible marker of the entire
  deferred structural half** of the ratified plan — which is why it is not a cosmetic
  nit: pulling the thread of the name unravels gaps 1–3 of [01](01-plan-vs-result.md).

## Why this matters more than a rename

A misnomer on a leaf display line is near-zero cost (ADR-0008's own example). A misnomer
on the **core allocation engine of a tool whose output is a provable bound** is the
worst-case surface the substitution test exists to catch: the next contributor — human
or LLM — reasons about the uncertainty machinery through the wrong method-model, and
nothing fails loudly to correct them. That is the precise composition ADR-0008 and
ADR-0002 were written, as siblings, to prevent.

[← 01 plan vs. result](01-plan-vs-result.md) · [03 independent audit →](03-independent-audit.md)

*Public Domain (The Unlicense).*
