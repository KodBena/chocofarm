# CLAUDE.md — chocofarm

You are working on **chocofarm**, a single Python package: an
operations-research scratch project that computes optimal *gil* farming in
FFXIII, formalized as **adaptive stochastic orienteering under partial
observation** — a belief-state MDP. The measure of success is the quality of
the solver and the honesty of the mathematical proofs, not a game tool. It is
Public Domain (The Unlicense).

It is **not** a monorepo. There is no frontend, no backend, no proxy, no
service, no database of record, no submodule. It is one `chocofarm/` package
of numpy / JAX / numba Python, with a simulation `Environment` + `Policy`
seam (`model/`, `solvers/`), an AlphaZero/Gumbel stack (`az/`), a provable
dual bound (`bounds/`), eval (`eval/`), an hp registry (`hp/`), analysis
(`analysis/`), and an extensive `docs/` corpus.

For orientation, read `docs/STATUS.md` and the latest handoff
(`docs/handoff-2026-06-15.md`). For architectural decisions, the canonical
reference is `docs/adr/`; the condensed reference is `docs/adr-synopsis.md`.
**Read the synopsis before substantive work**; consult specific ADRs when the
synopsis points to one.

## Authoritative documents

The ADRs are load-bearing, not advisory. In particular:

- **ADR-0002 (fail loudly)** governs error handling everywhere — it is the
  registry the code's 16+ `ADR-0002` citations point at.
- **ADR-0004 (minimal-touch)** governs editing under partial visibility (the
  large files: `decomp.py`, `analyzer.py`, `registry.py`).
- **ADR-0005 (documentation discipline)** governs how documentation is
  authored and maintained.
- **ADR-0006 (source-file headers)** governs the per-file module-docstring
  header convention (path + purpose + Public Domain).

A contribution that fights any of these is wrong by default; if a specific
case appears to warrant deviation, name it explicitly and ask before
proceeding.

## ADR-0002 applies to documentation consumption (LLM collaborators)

ADR-0002 (fail loudly) applies with special force to LLM collaborators reading
documentation for orientation. **The single gravest sin against ADR-0002 is to
fail to read a piece of documentation from beginning to end, and then make any
statement that references any part within it, no matter how small.** Failing
loudly here means the user is never in the dark about whether the collaborator
has actually seen the document being referenced. An LLM collaborator must never
consume documentation partially.

Concretely:

- Orientation documents — `docs/STATUS.md`, the current handoff,
  `docs/adr-synopsis.md`, every ADR cited, this CLAUDE.md, every consult or
  design note relied on — are read end to end before any claim about them is
  made. The same applies to any further document those orientation documents
  point at, when that document is the one being relied on. (The handoff itself
  states this: "to cite a document one has merely skimmed constitutes a silent
  failure of duty.")
- A `grep` hit, a search-tool preview, or any partial render is a pointer to
  read the file, not a substitute for reading it. Acting on a fragment is the
  silent failure this section names.
- If a document is too long to read in full given the immediate budget, say
  so explicitly — name what was read, what was skipped, and what the skipped
  portion might affect — and ask how to proceed. Do not paper over the gap.
- A statement that cites a section, an ADR number, a heading, a filename, or a
  sentence from a document the collaborator has not read end to end is itself
  the silent failure ADR-0002 forbids. (The dangling `consult-002 §4`
  citation — a `§4` that never existed in the report — is the worked
  cautionary instance: citing a section nobody read end to end.)

This composes with ADR-0004 (minimal-touch under partial visibility): when
context is missing, ask for it; when context is present but unread, read it;
either way, do not bluff.

## Documentation is part of the work

Implementation is incomplete until the documentation reflects it. Before
declaring a task done, audit:

- Does `docs/STATUS.md` or the current handoff describe an orientation surface
  this change affects, and is it still accurate? **Status documents record
  slowly-aging decisions and rationale, not a live task queue** (ADR-0005
  Rule 6 — the 24-seconds-stale handoff is the cautionary instance: a
  "pending" item narrated in immutable prose that was done before the prose
  was read). The live queue belongs in the commit log and the branch state,
  not in immutable prose.
- Does any ADR's "Revisit when…" section name a trigger this change satisfies?
  If so, record the firing by dated amendment (ADR-0005 Rule 8: amend by
  append; never silently rewrite a point-in-time record).
- Does any cross-reference now describe its target inaccurately, or point at a
  path/section that no longer resolves? Repoint live referrers on a move
  (ADR-0005 Rule 3/5); leave point-in-time records (the architectural audit,
  the agent commissions) un-retro-edited.
- Per ADR-0006, if files were touched under full visibility and lack the
  standard module-docstring header (path + purpose + Public Domain), retrofit
  it.

If yes to any, propose the documentation edits as part of the same change.
Code-only changes that have documentation implications are incomplete
deliveries.

## Authoring posture

Roadmap before code. Contracts and types before implementation. Pure logic
before effectful glue. Explain the why, briefly, in the language of the
abstractions involved — the **env/Policy seam** (`Environment` owns dynamics,
belief, simulation; `Policy` is the injected `decide(env, loc, bw, collected,
lam, rng)` contract; a new method is a new `Policy` subclass with zero env
edits), **Port/ACL** (a boundary translates and validates rather than coerces
— the hp registry's strict decode), the **domain bands** (Band 1
solver-agnostic / Band 2 OR-general / Band 3 FFXIII-bound; ADR-0003). These
are the codebase's vocabulary; using them keeps reasoning load down. Provide
complete file contents when editing a fully-visible file; partial-file outputs
into a large file invite the silent failures ADR-0004 is shaped to prevent.

The tone is methodical and deferential to the existing structure. The codebase
has a coherent personality (recorded in the ADRs): the hardest architectural
decisions are made right (the env/Policy inversion of control, λ threaded as a
live per-call argument, derived dimensions never hardcoded) and the discipline
is to *extend* those, not impose a different personality. Honest and
mechanistic: where a value should be hot, make it hot; where a fact has one
home, derive from it; where a claim is a perf or equivalence claim, attach the
substantiation (ADR-0009).

## Asking before assuming

If the context needed to do the work correctly is not in view — a file's full
contents, a related module's interface, the state of a branch — ask for it
before proceeding. ADR-0004 makes this non-optional under partial visibility;
the same posture applies when context is simply missing rather than partially
visible. When investigating a hang or a wedge in the parallel substrate, ask
for runtime visibility (a faulthandler dump, a `kill -ABRT` traceback) rather
than inferring behavior from the symptom — the parallel path's failure modes
are exactly the ones a wire-only / outside-only read misdiagnoses (the
`docs/notes/jaxtrain-deadlock-rca.md` arc is the cautionary instance).

## Scope discipline

Sessions are scoped to the task. Do not expand a doc-only change into a code
refactor (or vice versa) without surfacing the cross-cutting nature first. The
2026-06-15 architectural audit (`docs/notes/audit/`) is the standing roadmap
of larger structural work (the R-series); a session does not silently execute
an audit recommendation without it being in scope.

## Operational facts (orientation)

These orient a contributor running the code; they are stable enough to record
here, but the handoff is the live source for the current run state.

- **Python interpreter:** `/home/bork/w/vdc/venvs/generic/bin/python` (JAX,
  optax, numba) — the shared scratch environment the scripts run from. Tests:
  `PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/ -q`.
- **Redis:** two instances. **`127.0.0.1:6380`** is the memory-cache
  (`volatile-lru`) instance used for **worker transport** (raw bytes, the
  parallel ExIt fan-out, `az/parallel.py`). **`127.0.0.1:6379`** is the
  **disk-persisted (`noeviction`) instance for the hp registry**
  (`hp/registry.py` — registry keys carry no TTL and survive a restart;
  connection facts in `chocofarm/config.py`).
- **Experiment records:** checkpoints and `.log` files under
  `~/w/vdc/chocobo/runs/`; TensorBoard under `~/w/vdc/chocobo/tb/az/` (the
  daemon serves `--logdir tb/az` on port 6006). Both are gitignored
  (`.gitignore`: `runs/`, `tb/`). **Never discard experiment output — preserve
  it under `~/w/vdc`, not `/tmp`.**
- **Host:** a 4-vCPU libvirt VM; pin execution with `--cores 0,1,2,3`
  (parallel ceiling ~1.9×). `ptrace_scope=1` means `py-spy` cannot attach to a
  running process — launch under `PYTHONFAULTHANDLER=1` and `kill -ABRT` for
  thread tracebacks, or start the process beneath `py-spy`.
- **Model tier:** "Fable" is unavailable in this environment; direct design
  and verification work to the Opus tier.
- **Git:** stage by explicit path (never `git add -A`). Commit messages end
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Pushing
  session-born branches to origin requires the human maintainer's explicit
  consent (an automated guardrail refuses it otherwise).

## License

Public Domain (The Unlicense). Per ADR-0006, source files declare this
individually in their module-docstring headers.
