<!-- docs/notes/leaf-eval-refactor-audit-2026-06-22/04-evidence-log.md — Public Domain (The Unlicense) -->

# 04 — Evidence log (raw discovery)

[← 03 independent audit](03-independent-audit.md) · [README](README.md)

The raw material, kept so the audit can be re-derived and checked — including an honest
record of my own measurement misfires.

## A. The commit arc (advisory → HEAD), oldest first, with diffstats

```
1261f73 docs(design): advisory — refactor by responsibility            1 file,  +478
ec070fa refactor: lift Clark kink + gradient Port into alloc/ (move 4)  5 files, +392 -109
d3914b2 refactor: collapse runner gradient onto alloc/gradient (move 5) 4 files,  +99 -53
944606f refactor: single-home runner numpy bound (move 5, numpy half)   4 files, +154 -21
8d0c764 test: pin Z95 == driver z-quantile (hack-audit z re-divergence) 1 file,   +17 -2
0bcddc6 refactor: uniform model.registry_qname() (move 3a)              8 files,  +50 -30
7cfb868 refactor: canonical trusted_flags on all 5 models (move 3b)     3 files,  +21 -15
9ad52bf refactor: delete _model_sigmas dialect-sniffer (move 3c)        1 file,   +2 -10
576f6c4 feat: single JAX-traceable f + jax.grad seam (JAX J1)           10 files, +222
c95416c style: silence jax.config.update no-untyped-call (J1 fix)       1 file,   +1 -1
c669526 style: correct malformed type-ignore in jax_backend (J1 fix)   1 file,   +2 -1
8d72866 feat: driver consumes JAX f; OT f/gradient retired (JAX J2)     11 files, +82 -102
1101347 refactor: driver z/t CI quantiles OT→scipy (JAX J3a)            1 file,   +13 -28
cb20b44 refactor: remove dead OpenTURNS representation (JAX J3b)        11 files, +93 -331
fc1c8be refactor: collapse runners to single JAX path; retire numpy (J4) 7 files, +95 -407
c1d954f refactor: rename OpenTURNS/ -> leaf_eval_bound/                 70 files, +103 -103
075147f refactor: split bench_common -> estimators/pools/harness (m1)   43 files, +708 -644
8d34957 refactor: lift Estimate reconstruct glue into reconstruct (m2)  7 files, +174 -134
7ad7ae7 refactor: typed TransportModel contract + conformance net (m3)  4 files, +139 -12
9cff51a refactor: discovery-driven bench registration (move 7)          3 files, +137 -61
```

**Execution order vs. plan numbering:** `move 4 → move 5 → move 3(a/b/c) → JAX J1–J4 →
rename → move 1 → move 2 → move 3(typed) → move 7`. The plan's numbers were
leverage-ranked, not a strict sequence ("each box on its own commit"), so out-of-order is
not itself a fault — but two specific orderings cost work:

**The plan doc itself was committed once (`1261f73`) and never amended** — consistent with
ADR-0005 Rule 8 (a ratified version is a new record, not a rewrite), but it means the
deviations live only in commit bodies, not in any reconciliation record or `BACKLOG.md`.

## B. The move-5 / J4 churn (work undone within the arc)

```
$ git log --oneline --diff-filter=ADR -- tests/test_runner_support.py
fc1c8be refactor: collapse runners to single JAX path; retire numpy (J4)   [DELETED]
944606f refactor: single-home runner numpy bound (move 5, numpy half)      [ADDED]
```

`runner_support.py` is absent from the final tree. So move 5's numpy half (`944606f`) and
the hack-audit z-divergence pin (`8d0c764`, which lived in `test_runner_support.py`) were
both **created then deleted by J4** — `fc1c8be` "retire the numpy fallback." The plan's §5
had foretold exactly this ("the JAX swap retires the numpy twin"); a JAX-first ordering
avoids it. Move 5's *gradient* half (`alloc/gradient.py`) survived and became the JAX seam.

## C. End-state tree (flat — only `alloc/` + `benchmarks/` subdirs exist)

```
leaf_eval_bound/
  alloc/      __init__.py gradient.py jax_backend.py kink.py
  benchmarks/ __init__.py estimators.py harness.py pools.py register_benches.py bench_*.py(×30)
  examples/   demo_msgpass.py
  bench_store.py  estimate.py  leaf_eval_grounding.py  manifest.py  model_base.py
  model_capacity.py  model_cycletime.py  model_{zmq_baseline,shm_spin_poll,futex_wake,
    lockfree_mpsc,cpp_inproc_port}.py  neyman_driver.py  reconstruct.py
  throughput_bound.py  transport_sweep.py  untrusted_drive.py
```

Ratified §3 subpackages **not created:** `contract/ store/ models/ runners/`. No top-level
`__init__.py`.

## D. Verification batch (the doc's quantitative claims, re-checked by hand)

```
neyman_driver.py: 1051 lines
  155 class Recommendation        191 def where_to_spend     195 def report
  250 class NeymanDriver          678 def _assemble_sigma     734 def _family_multiplier
  798 def _socp_allocation        945 def run
model_capacity.py:84 def throughput_numpy / :95 def throughput_jax     (dual-write)
model_zmq_baseline.py:110 def throughput_numpy / :121 def throughput_jax (dual-write)
sys.path.insert preamble: 48 of 48 tool .py files     top-level __init__.py: ABSENT
alloc/driver.py: ABSENT     NeymanDriver referrers: 13 files
```

## E. End-state test run (the ADR-0009 "green" claim, partially re-verified)

```
$ pytest tests/test_alloc_kink.py tests/test_jax_f_equivalence.py \
    tests/test_transport_model_conformance.py tests/test_register_benches_discovery.py \
    tests/test_manifest_estimate_seam.py tests/test_neyman_driver_phase2.py -q
92 passed in 6.62s
```

Six core files green. The agent's per-commit claims of "280 / 291 leaf-eval tests pass"
were not re-run in full here; the independent audit ([03](03-independent-audit.md))
confirmed the suite (21 files) is repointed to the new module names and structurally sound.

## F. The "done" contradiction

The agent's own commit trailers track remaining work and never claim completion:

- `075147f` (move 1): "moves 2/3/6/7 remain."
- `8d34957` (move 2): "Moves 3/6/7 remain."
- `7ad7ae7` (move 3): "Moves 6/7 … remain."
- `9cff51a` (move 7, the final commit): leaves move 6 + the entire §3 relocation + the §4
  driver rename + the grounding split outstanding.

So a verbal "the work is done" is contradicted by the record the same agent committed.

## G. Self-corrections — my own measurement misfires (logged for honesty)

Three sweeps I ran misfired; recording them so this audit is not the thing it audits:

1. **Header false positive (twice).** `head -10` then `head -20` greps for "Public Domain"
   flagged 55, then 46, tool files as "missing the ADR-0006 header." **Wrong** — the
   declaration sits *below* the window in the longer module docstrings (it is at line 14 in
   `neyman_driver.py`, which the `-20` window happened to pass, and below line 22 in
   `estimate.py`, which it did not). The independent full read confirms headers are **100%
   compliant**. No header finding stands.
2. **`zsh` glob nulls.** Unquoted `--include=*.py` arguments made `zsh` try to glob them and
   abort the command ("no matches found: --include=*.py") — so my first OpenTURNS-referrer,
   swallowed-except, and done-claim sweeps returned *nothing because they did not run*, not
   because they were clean. Re-run quoted; results fold into [03](03-independent-audit.md)
   F5–F7.

The lesson driving this log: the first pass of this audit over-credited the agent's
commit-body self-justifications ("STRUCTURAL DEVIATION — flagged for scrutiny") as evidence
of clean conduct. Self-justifying prose is the artifact to verify, not the verdict — which
is why every quantitative claim here was re-derived from `git`/code, and the ADR citations
in [02](02-misnomer-adr-analysis.md) rest on full reads of ADR-0002 and ADR-0008.

[← 03 independent audit](03-independent-audit.md) · [README](README.md)

*Public Domain (The Unlicense).*
