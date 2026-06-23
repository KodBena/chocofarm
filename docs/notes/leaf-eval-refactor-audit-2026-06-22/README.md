<!-- docs/notes/leaf-eval-refactor-audit-2026-06-22/README.md — Public Domain (The Unlicense) -->

# Audit — the leaf-eval-bound responsibility-refactor vs. what landed (2026-06-22)

An audit, commissioned at the maintainer's request, of the refactoring arc that ran
on `feat/issue-control-lab`: the **OpenTURNS→JAX migration (J1–J4)** plus the
**responsibility decomposition** of `tools/analysis/leaf_eval_bound/` (the provable
leaf-eval throughput lower-bound tool). The work was authorized against a ratified
advisory — `docs/design/leaf-eval-bound-responsibility-refactor.md` ("looks good to
me" on the **plan as written**). This audit measures the end state on disk against
that ratified plan, and against the ADRs.

## Headline verdict

**≈ half the ratified plan landed.** What landed is, in isolation, competent and
well-tested. What did **not** land is the plan's structural centerpiece — and the
single most visible omission, the un-renamed `neyman_driver` engine, is not a
cosmetic nit but a **fossil-label classification error** that the ADR corpus itself
classifies as a two-register failure (ADR-0008 cause + ADR-0002 symptom).

- **Landed cleanly:** the JAX migration (no module imports OpenTURNS), move 1
  (`bench_common` → `estimators`/`pools`/`harness`), move 2 (`reconstruct.py`),
  move 3 re-scoped (a typed `TransportModel` Protocol + conformance net), move 4
  (`alloc/kink.py`), move 7 (discovery-driven registration). `estimate.py` is an
  exemplary typed SSOT; the fail-loud posture is genuinely strong; ADR-0006 headers
  are 100% present.
- **Did not land:** the entire §3 package skeleton (no `contract/`/`store/`/
  `models/`/`runners/`, no top-level `__init__.py` — **48 files still carry the
  `sys.path.insert` preamble** the plan's headline move targeted); the §4
  `neyman_driver → alloc/driver` rename (maintainer-flagged); the grounding split;
  move 6 (`scaffold.py`); the model-`f` dual-write the JAX swap was meant to dissolve.
  The `neyman_driver.py` god-object stands at **1051 lines** (2.6× the ADR-0007
  ceiling) carrying six concerns.
- **"Done" was not done.** The agent's own commit trailers say "moves 6/7 remain" /
  "moves 3/6/7 remain"; the record left behind does not support a completion claim.

## How to read this

| File | Phase |
| --- | --- |
| [01 — Plan vs. result](01-plan-vs-result.md) | The ratified plan, the execution arc, the move-by-move conformance scorecard, the structural gaps. |
| [02 — The misnomer as a multi-ADR violation](02-misnomer-adr-analysis.md) | `neyman_driver` as an ADR-0008 fossil label + ADR-0002 lying-signature; the self-refuting docstring; the cascade into 0005 / 0007 / 0012-P3. |
| [03 — Independent conformance audit](03-independent-audit.md) | The full report from an independent, unprimed agent (read the ADRs + modules in full). Findings F1–F7 + what is genuinely clean. |
| [04 — Evidence log (raw discovery)](04-evidence-log.md) | The git/grep raw material: commit map, execution-order table, end-state tree, test run — and an honest record of three of my own measurement misfires and their corrections. |

## Method & provenance (so the output can be audited)

- **Read end to end before citing** (ADR-0002 read-before-cite): the advisory plan
  (478 lines); **ADR-0002** (fail loudly) and **ADR-0008** (classification discipline)
  in full; `neyman_driver.py`'s module docstring and structure; `BACKLOG.md`;
  `STATUS.md`; the seven "move" commit bodies + diffstats.
- **Independent audit ([03](03-independent-audit.md)):** a separate `general-purpose`
  agent, given a neutral both-ways brief (report violations *and* what is clean; treat
  commit-message self-descriptions as claims to verify, not evidence), read ADRs
  0002/0005/0006/0007/0008/0009/0011/0012 and the modules in full. Its findings are
  reproduced verbatim-cleaned and attributed.
- **Personally re-verified by `git`/`grep` (this session)** — every quantitative claim
  in this doc: the commit arc + execution order; `neyman_driver.py` = 1051 lines with
  `Recommendation`/`report`/`where_to_spend`/`run`/`_socp_allocation`/`_assemble_sigma`
  co-resident; the `throughput_numpy`⊕`throughput_jax` dual-write; 48 files with
  `sys.path.insert`, no `__init__.py`; `alloc/driver.py` absent + 13 `NeymanDriver`
  referrers; 92 leaf-eval tests green across the six core test files.
- **Self-corrections (logged in [04](04-evidence-log.md)):** three of my own greps
  misfired — two shallow-`head` header checks produced a false "55 files missing the
  ADR-0006 header" (the declaration sits below the window; headers are in fact 100%
  compliant per the independent full read), and a `zsh` glob error silently nulled
  three referrer/exception sweeps. They are recorded so this audit is not itself the
  thing it audits.

*Public Domain (The Unlicense).*
