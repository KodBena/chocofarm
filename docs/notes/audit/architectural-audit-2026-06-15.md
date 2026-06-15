# Architectural Audit — `chocofarm` whole-codebase, top-down (2026-06-15)

A top-down architectural audit of the entire `chocofarm` codebase (≈9,600 LOC of Python across `model/`, `solvers/`, `az/`, `bounds/`, `analysis/`, `eval/`, `probes/`, `scripts/`, `tests/`), conducted against the working tree at `main@cfce276`. The commission was deliberately broad: find architectural malpractice wherever it lives — unnecessary coupling, configuration frozen at construction time that should be live, broken separation of concerns, single-source-of-truth erosion, and band-aids standing in for sound abstractions. The hyperparameter/registry friction that prompted the review was treated as one representative symptom of a systemic pattern, not as the scope of it.

**Status posture.** OPEN — advisory. Nothing here is executed; this is a record of findings, their verification, a target architecture, and a sequenced roadmap, for the maintainer to act on or overrule. No code was modified by this audit. The repository's own prior audit notes under `docs/notes/` are explicitly **not** treated as authority for posture (they predate this discipline and are, per the maintainer, suspect); this document follows the battle-tested audit posture exemplified by the LengYue `docs/notes/audit/` corpus.

**Provenance.** Conducted by an orchestrated subagent workflow (`chocofarm-arch-audit`, run `wf_5ee2051c-33d`, 35 subagents) under a single human-issued commission, with the orchestrating reviewer performing independent first-hand grounding reads before and after the fan-out. Every worker's raw output is reproduced verbatim in the companion file `architectural-audit-2026-06-15-appendix.md`. Point-in-time; **not retro-edited** — where a worker overstated or mis-cited, the claim stands uncorrected in the appendix and is corrected here in §5.

**Evidence-class tags** used throughout: *(runtime-verified)* — a claim a verifier confirmed by executing code; *(byte-verified)* — confirmed by reading the exact line; *(grep-verified)* — confirmed by a tree-wide search the reviewer ran; *(cited-not-rerun)* — asserted by a worker, consistent with the reviewer's reading, but not independently re-executed.

---

## Orientation — what the system is

For the future reader who arrives at this record cold (and so that no conclusion here depends on prose that may have moved): `chocofarm` is an operations-research scratch project that computes optimal *gil* farming in FFXIII, formalized as **adaptive stochastic orienteering under partial observation** — a belief-state MDP. The instance: 20 treasures at fixed 2D coordinates; exactly 5 are present each run (drawn uniformly, re-rolled each run → `C(20,5)=15,504` equiprobable latent worlds); 44 arrangement-face sense actions yield disjunctive "is ≥1 covered treasure present?" observations that shrink the belief. The objective is long-run **rate = treasures / time**, solved by **Dinkelbach**: for a rate `λ`, maximize `E[Σ value − λ·Σ time]`; the fixed point `λ*` is the rate. That `λ` is the penalty whose live-threading the audit repeatedly praises.

The code is laid out as: `model/` (the `Environment` simulator + belief mechanics), `solvers/` (pluggable `Policy` subclasses — greedy, certainty-equivalent, rollout, sparse-sampling, UCT, ISMCTS, NMCS, and an exact cluster-decomposition), a full **AlphaZero/Gumbel Expert-Iteration** stack in `az/` (net, JAX/optax training, Gumbel search, features, action-slot mapping, a redis-backed parallel executor, and the ExIt loop), `bounds/` (a provable information-relaxation dual ceiling), `analysis/` (instance/geometry tooling), `eval/` (per-solver measurement scripts), plus `tests/`, `probes/`, `scripts/`, and an extensive `docs/` corpus. The research frontier at audit time: a static floor of ~0.0855, decomposition ~0.094, AlphaZero ~0.097–0.10, against a (loose) clairvoyant ceiling of 0.1454. The maintainer's stated next lever is **heterogeneous gil values** — which, as §11 notes, is precisely the experiment the current architecture is least ready for.

The six commission deliverables map to sections as: *executive summary* → §1; *systemic anti-patterns* → §2; *component audit* → §3; *target architecture* → §6; *prioritized roadmap* → §7; *closing verdict* → §14. The remaining sections (§4 representative-symptom trace, §5 verification ledger, §8 decision log, §9 serendipity, §10 self-critique, §11 maintainer points, §12 coverage, §13 lessons) are the auditability scaffolding the maintainer asked this record to carry.

---

## Method

**The harness.** The audit ran as a three-phase deterministic workflow rather than a single pass, because a one-shot read of 9,600 LOC produces confident-but-unverifiable architectural claims, which is precisely the failure mode an architecture review must not have. The phases were chosen so that *verification precedes synthesis* — the conclusions are built only on claims that survived an adversarial check.

- **Phase 1 — Component deep-reads (11 agents, parallel).** One `general-purpose` subagent per subsystem, each handed an explicit file list and instructed to read those files *in full* (not skim), under a shared charter naming the malpractice classes to hunt and demanding `file:line` evidence with a verbatim snippet for every finding. Subsystems: model, solvers, az-net+train, az-search+features, az-orchestration, eval, bounds, analysis+probes+scripts, a cross-cutting config-flow trace, tests, and docs↔code coupling. The eleventh (config-flow) was a deliberate cross-cut — not a directory — to catch the disease that hides *between* modules, which no directory-scoped reader can see.
- **Phase 2 — Adversarial verification (22 agents, parallel).** The strongest claims (every `critical`/`major` finding plus every frozen-at-construction item; 98 in total, deduplicated, ordered critical-first, capped at 22) were each handed to a fresh skeptic instructed to **refute** them against the running code and to default to `refuted`/`overstated` unless the code unambiguously supported the claim. This is the load-bearing discipline of the whole exercise: it converts plausible architectural assertions into either confirmed-with-line-proof or demoted.
- **Phase 3 — Synthesis (2 agents).** One agent distilled the 8 cross-subsystem anti-patterns from the 11 reports + 22 verdicts (instructed to discount anything verification deflated and to require recurrence across subsystems); one designed the target architecture grounded in the actual modules, instructed explicitly to preserve the seams the audit found sound.

**Read regime.** Phase-1 agents read end-to-end; phase-2 agents read the cited locus and its neighborhood and were free to *run* the code; the orchestrating reviewer independently read `env.py`, `solvers/base.py`, `az/exit_loop.py`, `az/parallel.py`, `az/features.py`, `az/actions.py`, `az/mlp_jax_train.py`, `eval/eval_az.py`, and `eval/eval_uct.py` in full before designing the fan-out, and ran tree-wide greps for the duplicated reference constants and config seams *(grep-verified)*. The reviewer's independent reads agreed with the agents on every overlapping point; no agent finding contradicted a reviewer read.

**Disciplines applied.** Verbatim-record (every worker output preserved exactly); refute-by-default verification; cross-subsystem-recurrence test for systemic claims (a finding in one file is local; the same shape in three files is a disease); seam-preservation (the audit is *required* to name and protect the parts that are correct, not only to flay the parts that are not — an audit that cannot name what is right cannot protect it during remediation).

**Scope boundaries / coverage limits** (full inventory in §12): `attic/` was read only for corroborating context (it is explicitly dead); `az/bench/` and `probes/` were covered by the analysis and config agents but not given a dedicated reader; the `docs/` corpus was sampled, not read exhaustively; 76 of the 98 extracted claims were **not** independently verified and are carried as *(cited-not-rerun)*, and are not load-bearing for any conclusion in §1–§7.

---

## §1 — Verdict

**The bones are sound; the connective tissue is rotting.** This is not a structural abomination and it does not need a core rewrite — a conclusion that matters, because the cheap reflex on a research codebase this dense is to reach for one. The hardest, most expensive-to-retrofit decision in the system is **correct**: simulation and solver are cleanly inverted. `Environment` owns dynamics, belief, and simulation; `Policy` is a thin injected `decide(env, loc, bw, collected, lam, rng)` seam; `env.py` imports no solver and the contract is honored to the letter (`env.py:8-10`, `base.py:16-19`) *(byte-verified)*. The numeric dimensions a lesser codebase hardcodes are **derived from the instance** (`feature_dim(env)`, `n_action_slots(env)`), and the design docs explicitly forbid hardcoding them *(byte-verified)*.

The sharpening that organizes the entire diagnosis: **the team has already proven it knows how to make a value hot and own it in one place.** The Dinkelbach penalty λ — the single hardest configuration axis in the system — is threaded as a live per-call argument to ~100 call sites, owned by exactly one fixed-point loop, and `DecompPolicy` even rebuilds its per-λ tables when it moves (`base.py:18`, `env.py:141/159-165`, `decomp.py:546`) *(byte-verified)*. That is the gold standard, and it was applied **exactly once and then abandoned.** Search width, learning rate, the value vector, the episode horizon, the headline reference rates, and the feature-vector layout are all welded shut at construction or smeared across N files — not from ignorance, but from a discipline that was understood and then not extended under research velocity.

This reframes the whole audit. The malpractice here is almost never *the wrong idea*; it is *the right idea applied once and not propagated*. `feature_dim(env)` (derive, never hardcode) sits in the same package as three hardcoded reference constants. The live-λ seam sits one argument away from a live-budget seam that was never added. That is an unusually tractable kind of rot, because the template for fixing it already exists in the tree.

**Rating, on a scale from "elegant" to "structural abomination":** *sound core, sclerotic periphery* — roughly the middle of the spectrum, materially better than its surface friction suggests. The correct mental model is a good chassis carrying a research project's worth of scar tissue in its wiring harness. The remediation is overwhelmingly *subtraction and relocation* (delete the second copy; move the value into the tier it belongs in), with a small number of genuinely structural moves and exactly one high-risk one. **Salvageable, and worth salvaging.**

---

## §2 — Systemic Anti-Patterns

Eight diseases recur across subsystems. The hyperparameter friction that prompted this review is **A**, and it is the smallest visible tip of it. Full verbatim text and the complete cross-subsystem `file:line` example sets are in **Appendix C**; the disposition table is the index, and the prose below explains the *mechanism* of each — why it is cancerous rather than merely untidy, and what the principled replacement is.

| # | Anti-pattern | Severity | Recurs in | Disposition |
| — | — | — | — | — |
| A | Config frozen at construction; ownership lives nowhere | critical | model, solvers, az-net, az-search, az-orch, config-xcut | **holds**, runtime-grounded |
| B | SSOT dissolved — same knowledge re-encoded in N places | critical | model, bounds, analysis, az-search, eval, docs | **holds**, one drift already realized |
| C | Hidden global state keyed by object identity | major | az-search, az-orch | **holds** |
| D | Copy-paste programs instead of one parameterized runner | major | eval, az-net, tests, probes | **holds** |
| E | Abstractions built then abandoned beside a live inline copy | major | model, az-search, az-net, az-orch, bounds | **holds**, one sub-claim deflated to *latent* |
| F | Magic constants strewn as bare literals | major | model, solvers, bounds, az | **holds** |
| G | Load-bearing knowledge offloaded to unenforceable prose | major | whole tree (111 `design §` / 16 `ADR-0002`) | **holds** |
| H | Defensive band-aids stacked against a hostile substrate | major | az-orch (parallel), az-net (cache coherence) | **holds** |

### A. Config frozen at construction; the missing config-ownership layer `[critical]`

**The disease.** There is no `Config` object anywhere in `chocofarm`. Every tunable is owned by whichever `__init__` or argparse `Namespace` happened to capture it, and the only configuration seam in the entire project is a CLI flag. The pattern in the small: a tuning knob is captured once in `__init__`, stored as `self.X`, read as a constant for the object's life, with no setter, no schedule hook, no per-call override.

**Why it is cancerous, not spartan.** The proof it is *wrong* rather than merely minimal is internal: the architecture demonstrably knows the right answer for λ and extended it to nothing else. The damage is concrete and recurring:
- `gumbel_search.py:91` — `c_puct/c_visit/c_scale/c_outcome/max_depth` frozen in `__init__`, with **no CLI flag and no path through `ParallelExecutor`** (whose signature carries only `m/n_sims`). Sweeping them requires a source edit *(byte-verified)*.
- `mlp_jax_train.py:215` — `self.opt = optax.adam(learning_rate=self.lr)` and `:219` `self._az_update = _make_az_update(self.opt, self.l2)` bake `lr`/`l2` into the jit'd update closure at construction. **No LR schedule, warmup, or anneal is possible without rebuilding the trainer and resetting Adam's moments** — which is exactly why the handoff's *queued* LR-anneal experiment must kill the process and `--resume` from a checkpoint. The frozen-at-construction failure is biting the project in production, on its own roadmap *(byte-verified)*.
- `env.py:31-33` — `K/value/entry/teleport_overhead` frozen; every one of ~40 `Environment(` call sites is zero-arg, so the `value=/entry=/teleport_overhead=` kwargs are dead theatre *(grep-verified)*.
- `parallel.py:328` — `initargs=(self.cores, base_seed, m, n_sims)` freezes the search budget into pool initargs consumed once at `_worker_init`; the per-iteration `generate()`/`evaluate()` entry points carry no budget argument, so the parallel path **cannot vary search width the serial path can** (a divergent capability between two paths that should be interchangeable).
- `decomp.py:248` — `@lru_cache(maxsize=None)` closing over `lam`, keyed per `round(lam,6)` table, never evicted; the Dinkelbach loop discards and recomputes the exact backward induction every iteration, and the cache grows without bound across a λ sweep.

The damage *compounds* with B and C: because the env bakes its action space at construction, downstream caches key on env *identity* (anti-pattern C), so a reconfigured env silently aliases a stale cache. Frozen config is the root that makes the identity-cache hazard load-bearing.

**Principled replacement.** Introduce frozen config dataclasses grouped by concern (`SearchConfig`, `TrainConfig`, `ScenarioConfig`, `ParallelConfig`) constructed once; argparse becomes a thin `from_args()` adapter so a sweep or notebook constructs configs directly instead of faking a `Namespace`. Crucially, *separate frozen-for-the-object config from hot config that experiments vary*: make search budget and lr per-call arguments (or callables of live state) exactly as λ already is, and thread a `SearchConfig` through the `ParallelExecutor` initargs so workers pick up budget changes via the same version-gated rebuild that already reloads the net. (Appendix C.1.)

### B. SSOT dissolved — the same knowledge re-encoded in N independent places `[critical]`

**The disease.** The most correctness-critical knowledge in the codebase — the belief/dynamics semantics, the C(N,K) prior, the feature-vector layout, the `K` constant, and the headline reference rates — each lives in multiple hand-maintained copies with nothing forcing them to agree.

**Why it is cancerous.** Two reasons, both demonstrated rather than hypothetical. First, *the copies have already drifted*: `DECOMP_ANCHOR=0.0941` (`exit_loop.py:51`) versus `ref/decomp=0.094` (`eval_az.py:79`) versus `vhat_lam=0.094` (`eval_bound.py:173`, where it is a numerical input to a provable bound, not a display line) *(grep-verified)*. Second, *the copies validate each other*: the dual-bound machinery whose entire purpose is to **certify correctness** re-implements the env's belief math in `MiniEnv`, so it would certify against stale dynamics the moment `apply`'s semantics are revised in `env.py` and forgotten in `minienv.py`.
- **Belief mechanics:** `minienv.py:86-115` `filter_treasure/filter_detector/sample_world/apply` are byte-for-byte copies of `env.py:104-135`, down to `(1 if present else 0)` — and they have *already* silently diverged in one method (`MiniEnv.legal_actions` iterates `self.keep`; `Environment.legal_actions` iterates `range(N)`) *(byte-verified)*.
- **The C(N,K) prior** is built three times: `env.py:57`, `analyzer.py:106` (whose docstring admits "it is the prior, ports verbatim"), `minienv.py:52`.
- **The feature layout has three writers:** `features.py:208` authors the block order positionally (`out[o:o+N]=marg; o+=N` … `=unc; o+=N`); `actions.py:112,116` re-encodes the offsets as the literals `feat[2*N:3*N]`/`feat[5*N:5*N+nD]`; `feature_response.py:44` lists the order a third time — and the third writer has **zero test coverage**. The only guard (`test_legal_mask_paths_agree`) is order-blind to `feature_response` entirely. Reorder a sub-block and you get no error and *silently mislabeled feature-importance rows*. This is the sharpest landmine in the codebase: invisible, consequential, and one refactor away from firing.
- **`K`** is hardcoded in 3+ places (`env.py:31`, `analyzer.py:88`, `decomp.py:18/661`) while `instance.json` — the supposed instance SSOT — carries neither `K` nor `N` *(byte-verified)*.
- **The reference rates** duplicate what `harness.py:17-50` computes live (full trace in §4).

**Principled replacement.** Each fact gets exactly one owner: a shared `BeliefMechanics` base (or free functions over `cover_mask/value/N`) that `Environment` and `MiniEnv` both call, with `MiniEnv` overriding only the world set and the legal-actions treasure-id hook; one `world_array(N, K, support=None)` in the model package; a single `FeatureLayout(env)` descriptor of ordered named blocks that `build()` writes through and `actions.py`/`feature_response.py`/tests slice **by name** (a reorder then edits one structure and cannot silently mislabel); `K` in `instance.json`; one `ReferenceLines(env)` computed once and imported everywhere. (Appendix C.2; deflation in §5.)

### C. Hidden global state keyed by object identity `[major]`

**The disease.** Because the env freezes its action space at construction and owns no first-class action-space object, downstream code bolts on module-global caches keyed by `id(env)` — the least value-stable key possible, since CPython reuses memory addresses after GC and `Environment` defines no `__eq__`/`__hash__`.

**Why it is cancerous.** Today the bug is masked (all envs are layout-identical), but the moment two envs differ in N or detector count — exactly the multi-instance / variant-detector experimentation this codebase exists to enable — the cache silently hands back the **wrong bijection with no error**, and in any long-lived process it leaks one never-evicted entry per env constructed (the test suite alone builds 20+).
- `actions.py:48,55` — `_SLOT_TABLES[id(env)]`, never evicted, identity-as-address the only key.
- `parallel.py:132-135` — the worker's entire state (`env/net/search/version/redis/budget`) rides a module-level `_W` dict mutated by `_worker_init`/`_ensure_net`/`_gen_task` with no owning object. This is precisely what forces the `it + 1_000_000` version hack at `exit_loop.py:308` to disambiguate gen-vs-eval weight publishes through the single global `version` slot — a magic offset that silently breaks at iters ≥ 1e6.

**Principled replacement.** The env owns its action-space mapping as an attribute computed in `__init__` (`env.slot_tables`), so consumers read it directly — lifetime tracks the object, no module global, no `id()` key, no leak. For the worker, wrap state in a `Worker` class instantiated once per process, and namespace redis weight keys by `(run, phase, version)` so the offset disappears. **Never key a cache on `id()` of an object that defines no value-equality.** (Appendix C.3.)

### D. Copy-paste programs instead of one parameterized runner `[major]`

**The disease.** The same orchestration is re-typed across many files that differ only in one literal (which policy, which budget).

**Why it is cancerous.** The headline `%VoI` formula `(r - static)/(ceil - static)*100` — the single most important number the project tracks — is hand-typed in 8+ files (`harness.py:74`, `eval_uct.py:81`, `eval_ismcts.py:41`, `eval_az.py:45`, …) with no shared definition, so one transcription slip yields a plausible wrong column and changing the metric means editing eight files. `r2_score` is defined verbatim twice (`train_value.py:30`, `exit_loop.py:54`); both then implement the same permute-into-batches epoch loop. The eval suite speaks three incompatible CLI dialects (hand-rolled `for tok in args` at `eval_uct.py:54`, `sys.argv[1:]` int-parse at `eval_nmcs.py:45`, argparse at `eval_az.py`) for one family of measurements. No `conftest.py` exists; the `sys.path` hack is copied into all four test files. This makes experimentation painful (drive the whole eval matrix and you learn three CLI dialects) and production fragile (the empty-batch guard, the standardization order, every band-aid must be fixed in N places).

**Principled replacement.** One `eval/runner.py` with `references(env)`, `voi_pct(rate, refs)`, and `run_plan(env, plan, *, seed)` so each script shrinks to a `PLAN` literal plus one call and the metric has a single definition and a single test; a `SOLVERS` registry consumed by a uniform argparse so adding a solver is one entry, not N driver edits; one shared `train_epochs` used by both the value-only and full-AZ paths; a `conftest.py` owning the path seam. (Appendix C.4.)

### E. Abstractions built, then abandoned beside a live inline copy `[major]`

**The disease.** The codebase repeatedly contains the *right* abstraction, fully built and even documented, sitting unused beside a hand-inlined copy that is the actually-live path — the worst of both worlds, because a reader cannot tell which encoding is authoritative and the two drift in silence.

**Why it is cancerous.** Each is a *lying signature*: a seam that looks configured or consumed but is dead.
- `facemodel.SenseAction` (`facemodel.py:29-56`) implements `cost/filter/observe/informative` fully, with a 25-line `ENV_ADOPTION` comment (`:67-92`) describing how the env "would" consume it — adoption that never happened. The env never imports it and reimplements all four methods inline (`env.py:108`); the `('f', k)` action shape the comment prescribes is also drifted (live env uses `('d', i)`). Only one smoke test exercises it.
- `build()`'s `marg` parameter is documented "a supplied marg is not consumed" (`features.py:183`) — yet `netvalue_ismcts.py:54` computes `env.marginals(bw)` *specifically to pass it*, a wasted full marginals pass on every ISMCTS leaf.
- `train_epochs` accepts `lr/l2` and the docstring admits "the trainer's configured lr/l2 are authoritative" (`exit_loop.py:147`) — a phantom seam.
- `info_relaxation.py:249,354` stores `restrict_faces` whose only use is `if self.restrict_faces: … pass` — a no-op flag shipped as public API.

**Principled replacement.** Adopt or delete — never keep a parallel ideal beside the live inline copy. Either `env.__init__` builds `self.senses = facemodel.sense_actions(faces)` and delegates (the documented `ENV_ADOPTION`), or `facemodel.py` is deleted and its insight folded in. Remove the `marg` parameter and the dead `env.marginals(bw)` call. Delete `lr/l2` from `train_epochs` or make `set_lr` a real schedule. **A parameter the receiver cannot honor must not be in the signature.** (Appendix C.5; one sub-claim — the `mlp_jax` residual-drop — deflated to *latent* in §5.)

### F. Magic constants strewn as bare literals `[major]`

**The disease.** Properties of the problem instance and shared algorithmic invariants are typed as bare literals at each use site rather than owned once and referenced.

**Why it is cancerous.** The episode horizon — which **must agree** across the simulator, the base-policy rollout, and the tree search for a value estimate to be unbiased — appears as `40` in `env.simulate` (`env.py:138`), `base.py:125`, `info_relaxation.py:476`, and `exit_loop.py:60`, but as `24` in the tree solvers (`uct.py:114`, `ismcts.py:110`, `nmcs.py:79`). A depth-24 ISMCTS tree runs 40-step rollouts at its leaves, and the "horizon" the search reasons about *silently disagrees* with the horizon the evaluation rolls to *(byte-verified)*. The UCB `c=0.7` "held fixed across UCT and ISMCTS for a fair comparison" is implemented by writing `0.7` three times (`uct.py:114`, `ismcts.py:110`, `netvalue_ismcts.py:41`) and trusting nobody edits one — the invariant asserted in prose, not code. The λ-rounding tolerance is `round(lam,6)` (decomp) versus `round(lam,9)` (`info_relaxation.py:172`) — a silent SSOT for "what counts as the same penalty."

**Principled replacement.** One model-level horizon owned by the env (`env.max_steps`, or derived from N/K) that `_base_value` and every solver read by default; one shared `UCB_C = 0.7` imported by both tree solvers so "c held fixed" is enforced by reference; one named λ-tolerance shared by decomp and the bounds drivers. (Appendix C.6.)

### G. Load-bearing knowledge offloaded to prose the code cannot enforce `[major]`

**The disease.** The codebase treats external prose documents as binding specifications and architectural authority, but the documents cannot be enforced by any test, several do not resolve, and the ones meant to cure staleness are themselves stale.

**Why it is cancerous.** `ADR-0002 "fail-loud"` is cited as a binding convention 16 times across seven modules, and the handoff instructs readers to "refer to the ADR-0002 registry" — but **no such registry exists anywhere in the repo** *(grep-verified: no `adr/` directory, no decision-record file)*, so a new contributor cannot look up what the rule requires and the numbering falsely implies a recorded sequence. 111 `design §N` citations make a design doc the de-facto spec, yet at least three of its load-bearing specifics (`37-slot` space, `90-float` vector, `ISMCTS` teacher) are marked STALE in the very code implementing their successors — the doc has the authority of a spec and the reliability of a scratch note. `consult-002 §4`, the authority for the env's core face model (`env.py:35`), is filed in the wrong directory and has no §4 anchor — a dangling pointer for the simulation's heart. Nine modules explain their own behavior by reference to ephemeral experiment-session tags ("Part A/B/C") from one results write-up, undecodable standalone.

**Principled replacement.** Encode conventions in code, not in disconnected comments: replace the phantom ADR with either a real `docs/adr/` registry the comments reference by stable id, or a shared `fail_loud()` helper / lint check so "fail-loud" is testable. Stop citing volatile specifics (slot counts, feature dims) by §N — cite the derivation (`feature_dim(env)`). Status docs should record slowly-aging decisions, never a live task queue mirroring git. (Appendix C.7; serendipitous instances in §9.)

### H. Defensive band-aids stacked against a self-chosen hostile substrate `[major]`

**The disease.** Rather than removing a substrate conflict, the codebase accretes layer after layer of defensive patch, each fixing a symptom of the previous layer's fight, until the reliability strategy *is* the stack of band-aids.

**Why it is cancerous.** `ParallelExecutor` combined three hostile substrates (spawn-pool multiprocessing + raw-redis transport + JAX/XLA + numba) inside one pool and now carries at least seven named defensive layers (H1a/H1/H2/Fix-A/Fix-C plus TTL leak-bounds and a bounded close): per-result imap timeouts to escape unbounded-`list` hangs (`parallel.py:282-310`), bounded socket timeouts, 1h result-blob TTLs because aborted iterations leaked ~980 redis keys (`:255-260`), worker-side `faulthandler`+SIGUSR1 *solely to diagnose the wedge* (`:166-172`), `setdefault` of six native-thread env vars to sever the JAX→spawn-child residue (`:150-153`), and core assignment by **scraping the CPython process-name string** (`:175`, fail-soft collapsing 4-core parallelism onto one core with no error). Each patch is individually reasonable, but together they wall off a design whose own correctness test (`test_parallel_deadlock`) can only assert the abort path fires **loud** — it cannot reproduce the wedge or assert end-to-end liveness. *A subsystem whose correctness test can only prove "fails loud" rather than "works" is fragile by construction.* The same disease appears in `mlp.py:166-176`'s f32 cache-coherence invariant ("every writer must REBIND not mutate"), an unenforceable action-at-a-distance contract the docstrings narrate as *having already failed once* (served stale weights), now babysat by three tests.

**Principled replacement.** Stop fighting the substrate at the root. Give workers an entrypoint that imports **no jax/XLA** (pure-numpy search + redis; only the parent touches JAX), which eliminates the fork/spawn/XLA interaction that caused the wedge and makes the thread-pin, faulthandler, and most timeout bands unnecessary. Defensive timeouts should be a thin safety net, not the load-bearing reliability strategy. For the cache, make weights an immutable params object swapped atomically so coherence is an invariant of one object identity, not a per-writer obligation. (Appendix C.8.)

---

## §3 — Component Audit

Per-subsystem treatment. Each block states what is fundamentally broken and why, the right abstraction, the concrete high-leverage changes with file references, and the earned praise (because the seams worth protecting are as important to record as the rot). Full agent reports — every finding with `file:line` and the frozen-config inventory — are in **Appendix A**, indexed A.1–A.11.

| Subsystem | Health | One-line verdict |
| — | — | — |
| `model/` (A.1) | messy | Clean seam, monolithic build, duplicated belief math |
| `solvers/` (A.2) | messy | Good ABC, frozen knobs, duck-typed decomp |
| `az/` net+train (A.3) | messy | Defensible split, four hand-synced forwards |
| `az/` search+feat (A.4) | messy | Faithful math, three-writer feature layout |
| `az/` orchestration (A.5) | messy | Clean pipeline, god-object + band-aid sediment |
| `eval/` (A.6) | messy | One good seam, eight copy-paste mains |
| `bounds/` (A.7) | **sound** | Careful math, parallel sim-model |
| `analysis/` (A.8) | **sound** | Disciplined, K/N pipeline leak |
| tests (A.9) | messy | Pins debt, not behavior |
| config-xcut (A.10) | messy | No central config anywhere |
| docs↔code (A.11) | messy | Prose is load-bearing and rotting |

**Earned praise — the load-bearing good parts (protect these across every refactor).** An audit that cannot name what is correct cannot protect it during remediation. These are not consolation prizes; they are the *template* the §6 target generalizes, and a refactor that breaks one of them is a regression no matter what else it fixes:

- **The env/Policy inversion of control** (`env.py:8-10`, `base.py:16-19`) — `env` imports no solver; a new method is a new `Policy` subclass with zero env edits. The single hardest decision in the system, made right.
- **λ as a live per-call cell** (`base.py:18`, `env.py:141/159-165`) — owned by one fixed-point loop, threaded to ~100 sites; the gold standard the rest of the config story must reach.
- **Derived dimensions, never hardcoded** (`feature_dim(env)`, `n_action_slots(env)`) — the discipline whose absence is every SSOT finding; where it is applied, there is zero drift.
- **The param-registry-driven net serializer** (`parallel.py:92-128`) — the weight set is enumerated from the net's own `_params()`, so the optional residual block transports with no second edit site. Derive-don't-duplicate, done right.
- **Three bit-exactness contracts** — the distance memo (`env.py:66-82`), the jax/numpy float32 equivalence test, and the value-target MC-limit identity. These make the consolidations in §7 (R11 `ForwardSpec`, R2 `world_array`) *safe to attempt*.
- **`harness.py`'s live-computed floor/ceiling** (`:17-50`) — the one correct SSOT decision in the config story; the remediation of §2.B is "make the tree do what `harness` already does."
- **`analysis/`'s reuse discipline** — synthetic geometry flows through the *same* `arrangement()` the real map uses; proof the team can share an abstraction when it chooses to.

### 3.1 — `model/` (env, arrangement, facemodel, instance/faces data) — *messy*

**Broken.**
- `Environment` is a build-time monolith: instance, value vector, `K`, entry, teleport overhead, and the entire 4.5k-entry distance table are baked in `__init__` (`env.py:24-70`); every one of ~40 call sites is zero-arg, so the `value=/entry=/teleport_overhead=` kwargs are dead theatre.
- `K=5` is a bare literal (`env.py:31`) divorced from `instance.json`, which carries no `K` — the instance's defining parameter split between data and code.
- The belief mechanics are copy-pasted into `MiniEnv` `[critical]` and the world-prior into `analyzer` — the most correctness-critical code in the model, duplicated where the dual bound certifies against it.
- `facemodel.SenseAction` is fully built but dead; the env reimplements its four methods inline and references it only in a comment (`env.py:35`).
- `instance.json` still carries the superseded 16-region detector arrays (`overlaps`, `delta_treasures`) the face arrangement replaced — fossils a reader cannot distinguish from live fields, an edit to which silently does nothing.

**Right abstraction.** Split the immutable geometry/instance (treasures, teleports, faces, distance table — Tier 1) from the mutable scenario knobs (value, `K`, entry — Tier 2). Build the expensive geometry once; expose the scenario via a frozen `Scenario` and copy-on-write `env.with_scenario(s)` that shares `_dist`. The belief math becomes a `BeliefMechanics` base both `Environment` and `MiniEnv` call. The face becomes the single carrier of position+cover+semantics via `facemodel.sense_actions(faces)` — or `facemodel` is deleted.

**High-leverage changes.** `model/instance.py::load_instance() -> Instance` (one path resolution, one treasure-dict parse, replacing the three `__file__`-relative joins at `env.py:26`/`arrangement.py:28`/`analyzer.py:90`); move `K` and `entry` into `instance.json`; `world_array(N,K)` hoisted into the model package; `MiniEnv = Environment.restrict(keep, k_local)` inheriting belief math; resolve `facemodel` adopt-or-delete; strip the fossil arrays from `instance.json`.

**Praise (earned).** The env/Policy contract (`env.py:8-10`) is honored to the letter — `simulate` calls `policy.decide(self, …)` and `env.py` imports no solver; adding a method is a new `Policy` subclass with no env edit. The distance memoization (`env.py:66-82`) is honest and well-documented: built from the same `math.hypot` inputs as the live path with a total fallback, so it is bit-identical, a structural memo correctly labeled as such. `arrangement.py` is a clean effect-free value module: `Face` is a frozen dataclass, the cover-at-representative-point logic correctly handles non-convex faces, and it "knows nothing about solvers, beliefs, or travel" and lives up to it.

### 3.2 — `solvers/` (Policy ABC, greedy/CE/rollout/sparse, UCT, ISMCTS, NMCS, decomp) — *messy*

**Broken.**
- `DecompPolicy` duck-types the `Policy` interface instead of subclassing the ABC (`decomp.py:505`), silently defeating the `isinstance(base, Policy)` check at `uct.py:118`.
- Every solver hyperparameter (`iterations`, `c`, `horizon`, `width`, `depth`, `n_samples`, the Gumbel `c_*`/`max_depth`) is frozen as a flat `self.X` in `__init__`, so the eval scripts sweep by reconstructing one policy per budget.
- `decomp` freezes λ inside never-evicted `lru_cache` closures (`decomp.py:248`).
- Candidate-pruning logic (informative-detector predicate + nearest-k) is copy-pasted between `RolloutPolicy` (`base.py:70`) and `NMCSPolicy` (`nmcs.py:93`); `GreedyStopBase` (a shared rollout base) is mislocated *inside* `ismcts.py:45` and cross-imported by `uct.py`; `MacroPlanner` re-derives clusters/sense-filter/entry-anchors that `DecompPolicy._build` already computed.

**Right abstraction.** A `SearchConfig` dataclass per solver family, passed once *and* accepted as an optional per-`decide()` override so budget can anneal with belief width and `c_puct` across iterations; a `SOLVERS` registry mapping name → `(factory, primary-knob)`; one `candidate_actions()` pruner and the base policies (`GreedyStopBase`, `GreedyPolicy`, `CertaintyEquivalentPolicy`) collected in `solvers.base`.

**High-leverage changes.** One-token `class DecompPolicy(Policy)` (restores `isinstance` honesty); shared `UCB_C = 0.7` constant; lift `candidate_actions()` and `GreedyStopBase` into `solvers.base`; `SearchConfig` + `SOLVERS` registry (couples to the eval-runner work in 3.6).

**Praise (earned).** The `Policy` ABC is a genuinely good seam: a single `decide(env, loc, bw, collected, lam, rng)` contract with the env owning all dynamics, and most solvers honor it cleanly. `decomp.py` is *not* the god-object its 674 LOC implies — it is three honest layers (cluster decomposition, micro-solve, macro-plan); its sin is freezing λ and re-deriving env-owned state, not sprawl.

### 3.3 — `az/` net + training (mlp, mlp_jax, mlp_jax_train, train_value, dtypes, kernels, dataset) — *messy*

**Broken.**
- The split into a numpy inference object (`ValueMLP`) and a JAX/optax trainer (`JaxTrainer`) is a defensible boundary — but the forward graph is now copy-pasted across **four** implementations kept bit-compatible by hand (`mlp.py:131`, `mlp.py:216`, `mlp_jax_train.py:75`, and the residual-*less* `mlp_jax.py:46`), and the equivalence test pins only three.
- The weight-layout knowledge is split-brained between `ValueMLP` and `JaxTrainer`, which re-derives `name.startswith('W')` (`:118`) and `setattr`-by-key (`:238`).
- `lr/l2` are frozen at `JaxTrainer` construction while the loop still threads them per-call to a function that ignores them.
- The value standardization (`y_mean/y_std`) is the one piece of state mutated *live* through a side-channel (`exit_loop.py:150` → `net.set_value_scale(...)`) that all forwards must re-read.
- `_JDTYPE` is recomputed and frozen-at-import in two jax modules; `train_value.train()` and `exit_loop.train_epochs()` are near-duplicate epoch loops.

**Right abstraction.** One precision-agnostic `ForwardSpec` (a single op-list) yielding numpy-f64 / numpy-f32 / jax forwards, so the equivalence test guards *numerics, not transcription*; a `WeightContainer` owning the params dict + L2 mask + residual flag + npz layout; value standardization folded into the value-head weights or owned by the dataset pipeline, not smeared across four forwards; live `lr/l2` via `optax.inject_hyperparams`.

**High-leverage changes.** Build `ForwardSpec` *behind the existing equivalence test* (the contract that makes the consolidation safe); `WeightContainer.absorb/save_npz/load_npz`; resolve `mlp_jax.MlpJaxForward` (delete the `use_jax_mlp` seam or route it through the spec — see the §5 deflation, it silently drops the residual block); live `lr/l2` to unblock the queued LR-anneal.

**Praise (earned).** Moving training to `jax.value_and_grad` to retire the hand-derived residual backward + finite-diff gradient-check was the correct call, well-reasoned in the docstring; the manual backprop was genuinely fragile and an architecture change is now a one-line forward edit. The coupled-L2-in-the-loss decision (reproducing the numpy `g + l2·W` exactly rather than optax's decoupled decay) is carefully argued and correct.

### 3.4 — `az/` search + features + actions (gumbel_search, features, actions, feature_response, value_target, netvalue_ismcts) — *messy*

**Broken.**
- The feature layout is a three-writer SSOT violation `[critical]` (the §2.B landmine): `features.py:208` authors it, `actions.py:112/116` and `feature_response.py:44` re-derive the offsets independently, the latter untested.
- Search hyperparameters are frozen at construction with **no config seam past `m/n_sims`** `[critical]` — the parallel executor cannot reach `c_puct` et al.
- The `id(env)`-keyed `_SLOT_TABLES` module global is an unbounded leak + address-reuse hazard.
- `build()`'s `marg` parameter is accepted-but-ignored while `netvalue_ismcts.py:54` pays to compute it.
- `GumbelAZSearch` carries two sources of truth for `env` (stored at construction *and* threaded through every method, never asserted equal); `decide_with_value`/`decide_with_target` duplicate the empty-belief special case; `feature_response.py:11` hardcodes the stale dim `220` (live is 241).

**Right abstraction.** A single `FEATURE_LAYOUT(env)` descriptor of ordered named `(name, width, start)` blocks computed from env; `build()` writes through it and every consumer slices by name. `env.slot_tables` replacing the module global. One env source in the search. `SearchConfig` reaching the executor's per-iteration task.

**High-leverage changes.** `FEATURE_LAYOUT(env)` + the missing `feature_names` test (this is the single highest-leverage silent-failure fix in the codebase); delete the dead `marg=` param and the wasted `env.marginals()` at `netvalue_ismcts.py:54`; pick one `env` source in `GumbelAZSearch`.

**Praise (earned).** The math is faithful and the hot-path engineering is genuinely careful: the fused numba kernel (`belief_marg_cover`) does marginals and the detector reduction in one pass and is documented bit-exact; the per-belief cache verifies key collisions with `np.array_equal` so a hit always returns the same belief's features (`features.py:152`) — a correct, well-labeled structural memo. `GumbelPolicy` is actually the *better* config seam (`**kw`); the loop just pins it to `m/n_sims`.

### 3.5 — `az/` orchestration (exit_loop, parallel) — *messy*

**Broken.**
- The argparse `Namespace` is a 26-flag god-object threaded as `args.*` through `run`/`generate`/`train` (`exit_loop.py:175`).
- `train_epochs` accepts `lr/l2` then ignores them (`exit_loop.py:147`); the search config (`m/n_sims/base_seed`) is frozen into pool initargs and cannot change mid-run (`parallel.py:328`).
- `_W` is a hidden mutable worker global (`parallel.py:132`); `version` is overloaded with `it + 1_000_000` to disambiguate gen vs eval (`exit_loop.py:308`).
- The reference constants are hardcoded (`exit_loop.py:49-51`), duplicating the env-derived SSOT, and `:318` computes the headline `%VoI` from the frozen ceiling.
- The parallel substrate carries five-plus stacked deadlock band-aids (§2.H); redis-as-transport partially reinvents a job queue with cleanup bolted on after the fact; worker-index parsing scrapes the process-name string (`parallel.py:175`).

**Right abstraction.** A `RunConfig` (nested `Net/Search/Train/Parallel/Eval/IO` frozen dataclasses) built once, with `run()` taking the config object, not 25 `args.*` reads; CLI becomes `RunConfig.from_args`. The per-iteration parallel task carries `(SearchConfig, lam, seed)` — the worker already rebuilds the search per weight-version, so the live path exists; only the net flows through it today. Workers import no JAX. Weight keys namespaced by `(run, phase, version)`. Worker state in a `Worker` object.

**High-leverage changes.** `BeliefRefs(env)` + fix `:318` to divide by the computed ceiling (immediate); thread `SearchConfig` through the existing per-version rebuild (medium); the numpy-only worker entrypoint (long, high-risk — §7 R14).

**Praise (earned).** The ExIt loop is a clean linear `generate→train→eval→checkpoint` pipeline that checkpoints every iteration (a timeout/restart loses nothing). The redis raw-bytes transport boundary is a genuinely good idea, and the param-registry-driven `pack_net`/`unpack_net` is well-factored: the weight set is enumerated from the net's own `_params()` so an optional block (the residual `Wr*`) is transported without a second edit site — exactly the derive-don't-duplicate discipline the rest of the codebase needs.

### 3.6 — `eval/` (harness + 8 eval scripts + tb_runner) — *messy*

**Broken.**
- Every solver gets a bespoke copy-paste `main()` that re-instantiates the env, re-prints the floor/ceiling header, re-derives the `%VoI`-claw formula, and re-rolls its own arg parsing and `PLAN` literal.
- The eval budgets / iteration sweeps are frozen as module-level `PLAN` literals — the thing you most want to vary is the least reachable.
- `exit_loop.py` and `eval_az.py` freeze the very numbers `harness` computes (`0.0855/0.1454/0.0941`) as module literals `[critical]`.
- Half the scripts pass `seed=7`, half rely on the env default; `tb_runner` logroot defaults to a hardcoded personal absolute path.

**Right abstraction.** One shared eval driver: `references(env)`, `voi_pct(rate, refs)`, `run_plan(env, plan, *, seed)`, so each script shrinks to a `PLAN` plus one call and the metric has a single definition and a single test. Reference constants come from `BeliefRefs(env)`, never module literals.

**High-leverage changes.** `eval/report.run_plan` collapsing the 8 mains; `BeliefRefs(env)` (shared with 3.5); a uniform argparse over the `SOLVERS` registry.

**Praise (earned).** `harness.py` recomputing the static floor and clairvoyant ceiling **live** from the env (`:17-50`), with six of eight scripts importing those functions rather than hardcoding, is the *one correct SSOT decision in the entire config story*. The whole remediation of §2.B is "make the rest of the tree do what `harness.py` already does." Honor that win; do not let the frozen literals undercut it.

### 3.7 — `bounds/` (info_relaxation, eval_bound, minienv) — *sound*

**Broken.** `MiniEnv` is a hand-rewritten parallel copy of `Environment`'s belief/dynamics mechanics — a second source of truth the dual-bound validation certifies against, which already drifted in `legal_actions`. `DecompVhat` copy-pastes `decomp.py`'s per-λ `_build` setup and calls a private `_live_occupancy_posterior` (a leaky boundary into solver internals). The clairvoyant solve is implemented three times (harness, eval_bound, the no-penalty inner DP), each claiming to be "the" z≡0 ground truth. Reference λ values and per-`V̂` config are scattered magic literals. `restrict_faces` is an accepted-but-ignored constructor parameter.

**Right abstraction.** `MiniEnv` becomes a restriction *view* of `Environment` inheriting belief math; the injected-callable `V̂` seam stays (it is correct); one shared `clairvoyant_solve`.

**High-leverage changes.** `MiniEnv = Environment.restrict(...)` (depends on the `BeliefMechanics` extraction in 3.1); drop `restrict_faces` until the prune exists; route `DecompVhat` through a public decomp entry point.

**Praise (earned).** The subsystem is mathematically careful and unusually well-documented. The injected-callable `V̂` is genuinely the right abstraction for the dual penalty — a clean function seam that lets a trained AZ value network or a decomp decision-value serve interchangeably as the bound's generator. The information-relaxation duality argument (z≡0 reproduces the clairvoyant ceiling as a regression check) is sound and the code states its own validity conditions.

### 3.8 — `analysis/` + probes + scripts (analyzer, synthetic, residual_firewall, scripts) — *sound*

**Broken.** The exactly-K-of-N prior is hardcoded independently in 3+ places (`env.py`, `analyzer.py`, `kernels.py`) with `K`/`N` absent from `instance.json`. `world_array` re-implements `env.worlds` combinatorics verbatim. `probes/residual_firewall/ab_train.py` forks the entire training loop and monkeypatches private net internals with hyperparameters that silently diverge from the loop defaults. Scripts hardcode absolute `/home/bork` paths and a cwd-dependent artifact name. `analyzer` mixes presentation (`_print_report`) into the analysis module, and `real_instance` can silently mis-set N. `verify_faces.py` re-implements the old `cover_mask` model to cross-check the new one.

**Right abstraction.** Shared `world_array` and `K` from data (couples to 3.1); probes that *reuse* `exit_loop.train_epochs` rather than forking it; scripts that resolve paths via `importlib.resources`, not absolute literals; separate analysis from presentation.

**High-leverage changes.** Point `analyzer` at `model/instance.py` and the shared `world_array`; de-fork `ab_train.py` onto the real loop.

**Praise (earned).** `analyzer.py` and `synthetic.py` are the most disciplined code in the tree: a clean abstract `Instance` boundary, one-function-per-quantity decomposition, bounded enumeration, and *genuine reuse* of the model's `arrangement()` so synthetic geometry flows through the same planar-arrangement code the real map uses. This is the reuse discipline the rest of the codebase needs — proof the team can do it.

### 3.9 — tests — *messy*

**Broken.**
- The smoke layer pins genuine behavior cleanly, but the AZ layer is dominated by tests that exist *only to police architectural debt*: float-equivalence among three hand-maintained forwards (entrenching the leaky split rather than removing it) `[critical]`; the action-slot bijection and `test_legal_mask_paths_agree`, which exist *because* the feature layout is triplicated `[critical]`.
- White-box re-implementations of the search reach into private `root.prior/N/W`; the cache-coherence tests babysit the `id`-identity invalidation band-aid.
- `test_smoke.py:94` pins the literal `0.0855` so a legitimate model retune *fails a test* instead of recomputing.
- No `conftest.py`; the `sys.path` hack is copied into all four files.

**Right abstraction.** Tests should pin *behavior* and *invariants*, not the existence of duplication. Once `ForwardSpec` and `FEATURE_LAYOUT` land, the equivalence test guards numerics (legitimate) and the bijection test becomes a name-sliced layout assertion (legitimate). A `conftest.py` owns the path seam and shared fixtures. The smoke test asserts the *recompute is sane*, not a frozen literal.

**High-leverage changes.** `conftest.py`; convert `test_smoke.py:94` from literal-pin to recompute-sanity once `BeliefRefs` exists; keep the deadlock test (it is honest and valuable) but recognize it documents a substrate held together by patches.

**Praise (earned).** The smoke tests are a clean, bounded gate (reference lines, env shape, solver wiring). The deadlock test is honest about what it can and cannot prove. The suite is small and runs without a network where it can.

### 3.10 — config cross-cut — *messy*

This is not a directory but the cross-cutting trace, and it is the spine of the whole diagnosis. **There is no central configuration object anywhere in `chocofarm`.** Every entrypoint owns its config via either an ad-hoc argparse `Namespace` threaded as `args` or an inline `PLAN`/`build_plan` tuple-literal; the three load-bearing reference constants are hand-copied across ~10 modules plus tests and docs, encoded inconsistently (`0.0941` vs `0.094`); the instance shape (`N=20`, `K=5`, `entry='CSNE'`) is hardcoded in code rather than read from `instance.json`. Against all of that, the one dynamic variable that genuinely needs to be live — λ — is threaded correctly as a function argument everywhere and owned by one fixed-point loop. The result: the *runtime MDP is reconfigurable* but the *experimental scaffolding* (search widths, learning rate, reference lines, instance shape) is frozen or duplicated, so changing the instance or the operating point means editing many files in lockstep. The fix is the `RunConfig` of §6 — the cross-cut owner that does not exist today.

### 3.11 — docs ↔ code coupling — *messy*

Critical design knowledge lives in prose, not code structure: 111 `design §N` citations and 16 `ADR-0002` invocations point outward to documents the code cannot enforce or resolve (§2.G). The docs drift faster than the code — the handoff lists a "pending" docstring fix committed 24 seconds later (§9); the design doc's `37-slot`/`90-float`/`ISMCTS-teacher` specifics are STALE in the code implementing their successors; `consult-002 §4` is a dangling pointer to the env's core face model. Against this, the code has one genuinely sound knowledge pattern — the env-computed floor/ceiling — which makes the *parallel hardcoding* of those same numbers a pure, self-inflicted SSOT violation. The remediation (§7 R15) is to make conventions testable code and cite derivations, not volatile prose.

### 3.12 — Frozen-at-construction inventory (rollup)

The representative symptom, enumerated. Every item below is a tuning knob captured once in a constructor (or a module global) and closed over for the object's life, with no setter, schedule hook, or per-call override — the concrete instances behind §2.A. Tier in the §6 model is noted where it clarifies the fix.

- **`model/` — the scenario knobs (Tier 2; should be copy-on-write):** `value` reward vector (`env.py:32`, read only at `:131`); `K` present-count (`env.py:31`, a bare literal); `entry` teleport (`env.py:33`, read only at `:139`); `teleport_overhead=12.0` (`env.py:24/33`, feeds only `exit_cost`). Every `Environment(` call site is zero-arg, so all four are dead kwargs.
- **`solvers/` — search budget (Tier 3; should be per-call `cfg`):** `iterations/width/depth/sample-counts` frozen in every solver `__init__` (`uct.py:114-117`, `ismcts.py:110-114`, `base.py:64-65/93-94`, `nmcs.py:78-86`); UCB `c` (`uct.py:115`, `ismcts.py:111`); λ frozen inside `decomp`'s never-evicted `lru_cache` (`decomp.py:248/540/546`); episode horizon as a literal `40`/`24` in three places.
- **`az/` net+train — the optimizer (Tier 3):** `lr` baked into `optax.adam` (`mlp_jax_train.py:215`); `l2` closed into the jit'd update kernels (`:219-220`); Adam `betas/eps` constructor defaults with no seam (`:206`); `_JDTYPE` bound once at import in two modules (`mlp_jax.py:37`, `mlp_jax_train.py:54`).
- **`az/` search+feat:** the Gumbel `c_puct/c_visit/c_scale/c_outcome/max_depth` (`gumbel_search.py:91-102`) with no path past `m/n_sims`; the `id(env)`-keyed `_SLOT_TABLES` module global (`actions.py:48-64`).
- **`az/` orchestration:** `m/n_sims` frozen into pool initargs, consumed once (`parallel.py:328` → `:191`), the per-iteration `generate()` carrying no budget arg (`:343`); `lr/l2` (`mlp_jax_train.py:215`) while `train_epochs` accepts-and-ignores them (`exit_loop.py:147-148`); `STATIC_FLOOR/CLAIRVOYANT_CEIL` literals feeding the `%VoI` metric (`exit_loop.py:49-51/318`).
- **`eval/`:** Dinkelbach schedules + iteration ladders frozen as module-level `PLAN` tuples (`eval_uct.py:45-50`, `eval_ismcts.py:31-34`, `eval_nmcs.py:33-40`, `eval_faces.py:46-70`); `LAM0=0.0855` operating point (`eval_az.py:34`); `tb_runner` logroot a personal absolute path (`tb_runner.py:57`).
- **`bounds/`:** `vhat/vhat_lam/max_inner_states` frozen in `PenalizedClairvoyant.__init__` (`info_relaxation.py:243-249`); `DecompVhat.horizon` (`:104-105`); the dual-root bisection bracket `[lo,hi]`/`tol`/`max_iter` as positional defaults (`:402-403`, `eval_bound.py:123/156`).
- **`analysis/` + probes:** `K=5` duplicated (`analyzer.py:88`); probe hyperparameters frozen inline diverging from the loop defaults (`gen_frozen.py:22/28-29`, `ab_train.py:125`).
- **The `it + 1_000_000` artefact:** not a frozen value but its consequence — the single global `version` slot (a hidden-state symptom, §2.C) forces a magic offset to disambiguate gen-vs-eval weight publishes (`exit_loop.py:308` vs `:278`).

The pattern is total: of the project's experimentation levers, exactly one — λ — is live. Every other dial in this list is welded shut.

---

## §4 — The Representative Symptom, Traced End-to-End: the three reference constants

The commission named the hyperparameter/registry mess as the visible symptom; the most instructive single trace is the project's three load-bearing reference rates, because it exhibits the SSOT disease, the frozen-vs-derived tier confusion, and a verification deflation all at once.

The three numbers are the **static floor** `0.0855`, the **clairvoyant ceiling** `0.1454`, and the **decomp anchor** `0.094`. They are the axes of the headline metric the whole project optimizes: `%VoI = (rate − floor) / (ceiling − floor)`.

There are **three different source-of-truth strategies** for these three numbers, in one codebase:
1. **Derived (correct):** `eval/harness.py:17-50` computes `realizable_static(env)` and `clairvoyant_rate(env)` live from the env; six of eight eval scripts import them *(byte-verified)*.
2. **Frozen module constants:** `exit_loop.py:49-51` hardcodes `STATIC_FLOOR = 0.0855 / CLAIRVOYANT_CEIL = 0.1454 / DECOMP_ANCHOR = 0.0941`, and `:318` computes the apprentice loop's headline `%VoI` **by dividing by the frozen ceiling, importing neither authoritative function** *(byte-verified)*.
3. **Re-typed literals scattered across the tree:** the same numbers recur as hardcoded defaults, prose, and TensorBoard reference lines in ≥10 modules plus tests *(grep-verified: `exit_loop.py, eval_az.py, eval_faces.py, dataset.py, eval_bound.py, info_relaxation.py, bench_equivalence.py, bench_value_target.py, capture_states.py, probes/residual_firewall/gen_frozen.py`, + 2 test files)*.

The drift is **already realized** for the anchor: `0.0941` in `exit_loop.py:51`, `0.094` in `eval_az.py:79`, and — most consequentially — `vhat_lam=0.094` at `eval_bound.py:173`, where it is not a display reference but a **numerical input to a provable-bound computation** *(grep-verified)*.

**This is the disease in miniature.** The three reference lines are a *derived* quantity (Tier 4) — they must move when the env's value vector or geometry moves — and the malpractice is that someone froze a derived value into a literal in the one place (`exit_loop.py:318`) that feeds it into the primary success metric. The fix is not "make the constant configurable"; it is `BeliefRefs(env)` computed once and imported everywhere, and a smoke test that asserts the *recompute is sane* rather than pinning the literal. (See §5 for the deflation that this is, today, latent rather than realized.)

---

## §5 — Deflation Record (adversarial verification)

22 of the 98 strongest claims were handed to refute-by-default skeptics. **Outcome: 20 confirmed, 2 partial, 0 refuted, 0 overstated.** A zero-refutation rate on a refute-by-default pass is a strong (not conclusive — see §10) signal the findings are real; the value of the pass is concentrated in the two partials, which corrected severity and attribution the synthesis would otherwise have inherited.

**Partial 1 — the `mlp_jax.MlpJaxForward` residual-drop.** *Claim:* a "rejected" fourth forward is LIVE, silently drops the residual block, and is uncovered by the equivalence test. *Verified:* every mechanical sub-claim holds — `_forward_both` (`mlp_jax.py:44-56`) reads `a2` into both heads with no residual branch, `refresh` (`:71-79`) never copies the `Wr*` arrays, `gumbel_search.py:117-121` wires it in on `use_jax_mlp=True`, and it is the only one of four forwards `test_jax_equivalence.py` never constructs. **Deflation:** the word *LIVE* overstates exposure. `use_jax_mlp` defaults `False` (`gumbel_search.py:92`) and **grep finds zero production call sites passing `True`** — only `bench_hotpath.py` drives it. **Disposition: holds as a *latent opt-in landmine*, demoted from *live production wrong-net*.** Real, dormant behind a flag nobody flips; the residual-omission is genuinely undocumented and the "rejected" framing contradicts `az-jax-perf.md:80-83`, which calls the path "kept and selectable."

**Partial 2 — the reference-constant SSOT (the §4 trace).** *Claim:* the floor/ceiling/anchor literals are hardcoded across ~10 files while computed elsewhere, and a `%VoI` headline from a frozen ceiling *silently misreports the moment the env moves*. *Verified:* the SSOT core is exact — the literals duplicate `harness.py`'s computed values, `exit_loop.py:318` genuinely computes `%VoI` from the frozen ceiling and imports neither authoritative function, and `test_smoke.py:94` pins the literal so a legitimate retune breaks the only guard. **Three deflations, all material:** (1) the verifier **ran the live env** — `realizable_static(env)=0.08553`, `clairvoyant_rate(env)=0.14537` — so the literals are **not currently stale**; the floor and ceiling are *detector-independent* and the 16-region→44-face model change did not move them. The risk is **latent**, not realized misreporting *(runtime-verified)*. (2) The auditor mis-attributed the worst offender: `eval_az.py` computes its `%VoI` from the *env-derived* static/ceil (it is the **good** citizen on the metric); its `LAM0=0.0855` is a λ operating-point passed to `env.rate()`, not a floor in a VoI formula. **The file that computes `%VoI` from a frozen literal is `exit_loop.py`, not `eval_az.py`.** (3) `DECOMP_ANCHOR=0.0941` at `exit_loop.py:51` is used *only* as a TensorBoard display line (`:340`), never in computation — calling it a computed-quantity duplication there overstates it (though `eval_bound.py:173`'s `vhat_lam=0.094` genuinely is load-bearing). **Disposition: holds, with severity corrected from "stale-and-misreporting-now" to "latent SSOT landmine," and the canonical example relocated to `exit_loop.py:318`.**

Both partials make the audit *stronger*, not weaker: they replace a dramatic-but-false "your metric is wrong right now" with a precise-and-true "your metric is one detector-dependent value-vector change away from silently wrong, and the guard that should catch it instead forbids the change." That is the claim that belongs in a record someone will read in a year.

**Evidence ledger — the 22 verified claims.** Every conclusion in §1–§4 traces to a row here; the full verifier text (with the exact lines each skeptic re-opened) is Appendix B. Abbreviated for the index; `partial` rows carry the correction made above.

| # | Verdict | Claim (abbrev.) | Verified code-fact (abbrev.) |
| — | — | — | — |
| 1 | `confirmed` | MiniEnv copy-pastes the entire belief/dynamics surface of Environment | `minienv.py:25` is a standalone class (no subclass); filter/marginals/apply byte-identical to `env.py:104-135` |
| 2 | `partial` | `MlpJaxForward` is a "rejected" 4th forward that silently drops the residual block, untested | Mechanics all hold; **`use_jax_mlp` defaults False, zero prod call sites** → dormant landmine, not live |
| 3 | `confirmed` | Feature layout is a three-writer SSOT violation (offsets re-derived in actions/feature_response) | author `features.py:208-212`; re-sliced `actions.py:112,116`; re-listed `feature_response.py:44` — no shared owner |
| 4 | `confirmed` | Gumbel search hyperparams (`c_puct…max_depth`) frozen at ctor, no seam past `m/n_sims` | `gumbel_search.py:91-92` frozen `self.X`; `ParallelExecutor` carries only `m/n_sims` |
| 5 | `confirmed` | Reference constants frozen as literals in `exit_loop`/`eval_az`, duplicating `harness` | `exit_loop.py:49-51` literals; `harness.py:17-50` computes them live |
| 6 | `confirmed` | Exactly-K-of-N prior encoded in 3+ places; K/N absent from the data file | `env.py:31` `K=5` literal; re-hardcoded `analyzer.py:88`, `decomp.py`; `instance.json` has no K |
| 7 | `confirmed` | The three reference rates hand-copied across ~10 modules + tests, encoded inconsistently | `0.0941` vs `0.094` drift confirmed across `exit_loop`/`eval_az`/`eval_bound` (grep) |
| 8 | `confirmed` | Tests' load-bearing safeguard is float-equivalence among THREE hand-maintained forwards | `mlp.py:117`, `mlp.py:214`, `mlp_jax_train.py:60` — three transcriptions, equivalence test pins them |
| 9 | `confirmed` | Feature layout in three places; bijection/mask tests exist only to police that duplication | repo-wide grep: no shared layout constant; the guard tests exist because of the triplication |
| 10 | `confirmed` | ADR-0002/0004 are comment-only aspirations: ~16 invocations, zero defining registry | 12 `ADR-0002` + 1 `ADR-0004` in code, 3 in tests; **no `adr/` registry exists** |
| 11 | `partial` | A `%VoI` headline from the frozen `0.1454` silently misreports the moment the env moves | SSOT real & `exit_loop.py:318` divides by frozen ceiling; **but ran env: 0.08553/0.14537 → latent, not stale-now; offender is `exit_loop` not `eval_az`** |
| 12 | `confirmed` | Environment is a build-time monolith; the `value=/entry=/tp=` kwargs are dead | `env.py:24` kwargs; every `Environment(` call site zero-arg |
| 13 | `confirmed` | `K=5` hardcoded in `env.__init__`, divorced from `instance.json` | `env.py:31`; `instance.json` carries neither K nor N |
| 14 | `confirmed` | `facemodel.SenseAction` is fully built but dead; env reimplements it inline | "attempted to refute and could not"; env never imports facemodel, dup at `env.py:108` |
| 15 | `confirmed` | The C(N,K) world-set built verbatim in three files | `env.py:57`, `analyzer.py:106` ("ports verbatim"), `minienv.py:52` — no shared helper |
| 16 | `confirmed` | FROZEN: the value reward vector frozen in `__init__`, every site zero-arg | `env.py:32`; read only at `:131`; het-values experiment must rebuild or monkeypatch |
| 17 | `confirmed` | FROZEN: `K` hardcoded literal `5`, not data | `env.py:31` verbatim |
| 18 | `confirmed` | FROZEN: `entry` closed over at ctor, defaults `CSNE` | `env.py:24`; read only at `:139` |
| 19 | `confirmed` | FROZEN: `teleport_overhead=12.0` frozen, only feeds `exit_cost` | `env.py:24/33/85`; all call sites zero-arg |
| 20 | `confirmed` | `DecompPolicy` duck-types `Policy` instead of subclassing the ABC | `decomp.py:505` no base class; `base.py:16` `class Policy(ABC)`, subclassed elsewhere |
| 21 | `confirmed` | Every solver hyperparameter frozen in `__init__`, sweeping requires re-instantiation | verified all six solver files; cited lines exact |
| 22 | `confirmed` | `decomp` freezes λ in `lru_cache` closures, never evicts the per-λ table | `decomp.py` read in full (674 lines); λ frozen in memoised solver, no eviction |

The 76 unverified claims are carried as *(cited-not-rerun)* and are **not** load-bearing for any §2 anti-pattern (each anti-pattern rests on at least one verified example); they are recorded in Appendix A for completeness and traceability, not relied upon.

---

## §6 — Target Architecture

The end-state is organized by a single rule: **every configuration value belongs to exactly one of four tiers, and the tier is a structural property of where the value lives — not a discipline each call site must remember.** Full diagram, component interfaces, config-flow, and init strategy are in **Appendix D**; the essentials follow.

```
 data/instance.json  (SSOT: treasures, teleports, K, entry)
        │  load_instance() ─► Instance            [one parse, one owner]
        ▼
 ┌──────── MODEL (immutable geometry) ────────┐    TIER 1 — INVARIANT (ctor arg)
 │ Environment(instance)                       │      distance table, worlds, faces
 │   belief math · simulate · dinkelbach       │    TIER 4 — DERIVED (never stored twice)
 │   derived: feature_dim · slot_tables · refs │      N, feature_dim, horizon, refs
 │   .with_scenario(Scenario)  copy-on-write   │    TIER 2 — SCENARIO (copy-on-write)
 │   MiniEnv = restrict(keep,k): overrides     │      value, entry, K
 │     worlds+detectors only, INHERITS belief  │
 └─────────────────────────────────────────────┘
   ▲ env per-call            ▲ env read-only
 ┌──────────┐   ┌─────────────────────┐
 │ Policy   │   │ BeliefRefs(env)     │   SearchConfig (frozen, cheap-to-replace)
 │ decide(  │   │  floor/ceil/anchor  │   lam + budget = TIER 3 LIVE CELL (per-decide)
 │  …,lam,  │   │  = f(env), derived  │
 │  cfg)    │   └─────────────────────┘   [SOLVERS registry: name→factory]
 └──────────┘
 ┌──────── AZ STACK ──────────────────────────────────┐
 │ ForwardSpec (ONE op-list ─► numpy-f64/f32/jax)      │
 │ WeightContainer OWNS params+L2 mask+residual+npz    │
 │ JaxTrainer  lr/l2 ─► inject_hyperparams (LIVE)      │
 │ FeatureBuilder uses FEATURE_LAYOUT(env) (one owner) │
 │ ParallelExecutor: per-iter task carries             │
 │   (SearchConfig, lam, seed); workers import NO jax   │
 └──────────────────────────────────────────────────────┘
        ▲ RunConfig (Net/Search/Train/Parallel/Eval/IO) built once in main()
```

**The four tiers (the config-ownership rule).** The litmus test for each: *if the value changes during a run or across a sweep, it is a live cell, not a constructor invariant.* Under that test, `m/n_sims/lr` fail today (frozen) and λ passes (live) — which is the whole bug.
- **Tier 1 — true constructor invariant:** the geometry (distance table ~4.5k entries, the C(N,K) world array, parsed faces). Expensive, never changes within a run. Lives in `Environment.__init__` and nowhere else; `instance.json`/`faces.json` are its on-disk SSOT.
- **Tier 2 — scenario (copy-on-write on the env):** `value`, `entry`, `teleport_overhead`, `K`. None enter the distance table *(verified: `value` read only at `env.py:131`, `entry` at `:139`, `tp` at `:85`)*; `env.with_scenario(s)` returns a new env sharing `_dist`, so a value/K sweep is a comprehension over `Scenario`s, not 40 rebuilds — and because it yields a fresh object, no `id(env)` cache can silently alias the old config. The attic het-values monkeypatch becomes first-class.
- **Tier 3 — live cell:** λ *(already correct)*, search budget `(m, n_sims, c_*)`, learning rate `(lr, l2)`. Made live two ways: a `SearchConfig` dataclass cheap to replace, *and* an optional per-`decide()` / per-iteration-task override so budget anneals with belief width and `c_puct` across iterations; `lr/l2` via `optax.inject_hyperparams`.
- **Tier 4 — derived:** `N`, `feature_dim`, `n_action_slots`, `slot_tables`, the episode horizon, the three reference lines. Computed from source on demand (or cached on the object that owns the source); **never frozen into a literal.** The codebase already nails `feature_dim`/`n_action_slots`; the target extends that exact discipline to the three places that violate it (refs, horizon, layout).

**Worked signatures** (the shape of the seams, grounded in the modules they replace):
- `load_instance(path=DEFAULT) -> Instance(treasures, teleports, K, faces, regions_wkt)`, `N = len(treasures)` — replaces the 3× parse and the `K=5`/`N=20` triple-hardcode.
- `Environment(instance)`; `.with_scenario(Scenario) -> Environment` (shares `_dist`); `.max_steps`; `.slot_tables`; `.restrict(keep, k_local) -> MiniEnv` — the existing `d/exit_cost/route_time/marginals/filter_*/apply/legal_actions/simulate/rate/dinkelbach_rate` are **unchanged** (the seam is already correct).
- `Policy.decide(env, loc, bw, collected, lam, rng, cfg=None)` — the optional `cfg` is the only addition; the env-passed contract stays.
- `FEATURE_LAYOUT(env) -> {name: slice}`; `FeatureBuilder.build(loc, bw, collected)` (no `marg`); `feature_dim(env)` kept.
- `ForwardSpec.eval(params, X, *, backend, dtype) -> (v, logits)`; `WeightContainer.absorb/save_npz/load_npz`.
- `RunConfig.from_args(argv) -> RunConfig`; `eval.report.run_plan(env, plan, *, seed)`; `BeliefRefs(env).static_floor/.clairvoyant_ceiling/.decomp_anchor`.

**Before / after — the three highest-leverage moves**, sketched against the actual code they replace.

*R3 — reference rates (the §4 symptom), from frozen-literal to derived:*
```python
# BEFORE — exit_loop.py:49-51,318: a derived value frozen into the success metric
STATIC_FLOOR = 0.0855; CLAIRVOYANT_CEIL = 0.1454; DECOMP_ANCHOR = 0.0941
voi = (rate - STATIC_FLOOR) / (CLAIRVOYANT_CEIL - STATIC_FLOOR) * 100   # divides by a literal

# AFTER — one owner, computed from the env that the metric is about
refs = BeliefRefs(env)                       # wraps the existing harness.realizable_static/clairvoyant_rate
voi = refs.voi_pct(rate)                      # (rate - refs.static_floor)/(refs.ceiling - refs.static_floor)*100
# test asserts refs.static_floor is *sane*, never pins a literal that a retune would break
```

*R6 — feature layout (the sharpest landmine), from three hand-synced writers to one descriptor:*
```python
# BEFORE — three independent encodings, one untested; reorder → silent mislabel
# features.py:208   out[o:o+N]=marg; o+=N ... out[o:o+N]=unc; o+=N      (author, positional)
# actions.py:112    avail = feat[2*N:3*N]                               (re-sliced by literal)
# feature_response.py:44  re-lists the block order                      (third writer, 0 tests)

# AFTER — one descriptor; build() writes through it, everyone slices by NAME
LAYOUT = FEATURE_LAYOUT(env)                  # ordered named (name, width, start) blocks
avail  = feat[LAYOUT['available']]            # actions.py
unc    = out[LAYOUT['unc']] = marg*(1-marg)   # features.py writes the same names it slices
# a reorder edits ONE structure; a name typo is an error, not a silent wrong row
```

*R7 — scenario knobs, from frozen monolith to copy-on-write (unblocks the het-values experiment):*
```python
# BEFORE — value frozen in __init__; a sweep rebuilds the 4.5k-entry distance table per config,
# and the attic experiment monkeypatches a module global:  M.value = [10.0 if i in HIGH else 1.0 ...]

# AFTER — geometry built once, scenario is copy-on-write sharing _dist
base = Environment(load_instance())           # Tier-1 geometry, one build
sweep = [base.with_scenario(Scenario(value=v)) for v in value_vectors]   # Tier-2, shallow copies
# no id(env) cache aliases the old config — with_scenario yields a fresh object
```

**Why "hot where it should be hot" stops being heroic.** Heat is determined by tier, and tier is determined by where the value lives. (1) The per-call seam that already makes λ hot is the same seam that carries a `cfg` override — making budget hot is mechanical and local, not a global rewrite; the contract that makes λ hot is the contract that makes search budget hot. (2) Copy-on-write makes the cold scenario knobs *cheap to re-bind* so nobody is tempted to freeze them; the geometry stays a true invariant (correctly cold), the scenario becomes trivially re-bindable (appropriately warm). (3) Derived-from-source eliminates the stale-literal class entirely — and the codebase already proves this is free, because the only places with zero drift today (`feature_dim`, `n_action_slots`) are exactly the derived ones, while all the drift sits where someone froze a derivative. **Nothing is "kept in sync by nobody" because nothing is kept in sync at all — it is derived.**

**Seams to preserve (non-negotiable across every refactor):** the env/Policy boundary (`env.py:8-10`, `base.py:16-19`); the per-call λ threading; the derived-dimension discipline (`feature_dim`/`n_action_slots`); and three bit-exactness contracts that are load-bearing — the distance memo (`env.py:66-82`), the jax/numpy float32 equivalence test, and the value-target MC-limit identity. These are the *template* for the remediation, not targets of it.

---

## §7 — Prioritized Refactoring Roadmap

Recommendations with outcome columns. Risk is the chance of a behavior regression; leverage is rot retired per unit risk. Full verbatim roadmap in Appendix D.6.

| # | Recommendation | Tier of work | Risk | Disposition |
| — | — | — | — | — |
| R1 | `model/instance.py::load_instance`; `K`/`entry` into `instance.json`; repoint 3 parse sites | immediate | low | pure dedup, no behavior change |
| R2 | Hoist `world_array(N,K)` into the model package (3 call sites) | immediate | low | analyzer's "ports verbatim" is the confession |
| R3 | `BeliefRefs(env)`; delete `exit_loop.py:49-51` literals; **fix `:318` to divide by the computed ceiling**; reconcile `0.094`/`0.0941` | immediate | low | retires the §4 symptom |
| R4 | `env.max_steps` single horizon; reconcile the four `40`s and three `24`s | immediate | low-med | unbiases the value estimate |
| R5 | `class DecompPolicy(Policy)`; shared `UCB_C`; shared `candidate_actions()`; delete dead `marg=` + the wasted `env.marginals()` | immediate | low | restores `isinstance` honesty |
| R6 | `FEATURE_LAYOUT(env)` descriptor; slice by name; add the missing `feature_names` test | medium | med | closes the highest-leverage silent-failure landmine |
| R7 | `Scenario` + `Environment.with_scenario` (copy-on-write) | medium | med | makes value/K/entry first-class sweepable |
| R8 | `MiniEnv = Environment.restrict(keep, k_local)` inheriting belief math | medium | med | one source of truth for the certified dynamics |
| R9 | `env.slot_tables` attribute; delete `_SLOT_TABLES` global | medium | low | kills the `id(env)` aliasing hazard |
| R10 | `SearchConfig` + `SOLVERS` registry; collapse 8 eval mains into `eval/report.run_plan`; thread `SearchConfig` through the per-iteration task | medium | med | one config seam, one eval driver |
| R11 | `ForwardSpec` + `WeightContainer` behind the equivalence test; resolve `mlp_jax`'s residual-less forward | medium | med-high | equivalence test guards numerics, not transcription |
| R12 | `RunConfig` nested dataclasses; `run()` takes the config object; CLI → `from_args` | long | med | enables programmatic sweeps/notebooks |
| R13 | Live `lr/l2` via `optax.inject_hyperparams`; honor or delete the per-call `lr/l2` | long | med | unblocks the queued LR-anneal |
| R14 | Numpy-only worker entrypoint (no JAX in child); namespace weight keys; `Worker` object replaces `_W`; retire band-aids | long | **high** | removes the substrate the deadlock RCA fought |
| R15 | Adopt-or-delete `facemodel.SenseAction`; write/retire the ADR registry; fix the dangling `consult-002 §4` | long | low | ends the dead-vs-live dual encodings |

**Sequencing (the dependency DAG).** Arrows mean "must land before". The immediate tier is a free-standing front that unblocks everything; the two structural spines are the `BeliefMechanics`/`instance` extraction and the `FEATURE_LAYOUT` guard.

```
  R1 load_instance ─┬─► R7 Scenario ──────┬─► R10 SearchConfig+registry ─► R12 RunConfig
  R2 world_array  ──┘                      │
                     └─► R8 MiniEnv.restrict┘ (needs BeliefMechanics from R1/3.1)
  R3 BeliefRefs ─────────────────────────────► (feeds R10 eval-runner, R4 metric sanity)
  R6 FEATURE_LAYOUT ─► [GATE: any feature-block reorder]      R9 slot_tables (indep.)
  R11 ForwardSpec ─[behind equivalence test]─► (indep. of the model spine)
  R13 live lr/l2 ─► (needs R12 RunConfig for the schedule seam)
  R14 numpy-only worker ─[GATE: test_parallel_deadlock; remove band-aids in ONE step]
  R5, R15 ─ independent, do anytime
```

**Dependencies & risks (full set in Appendix D.6):**
- **Ordering is load-bearing.** R1–R2 must land before R7–R8 (Scenario/restrict); R7–R8 before R10/R12 (config threading). **R6 (`FEATURE_LAYOUT`) must land before any feature-block reorder is attempted** — it is the guard that makes a reorder safe.
- **R14 is the single high-risk item.** It alters the parallel substrate the deadlock RCA fought; gate it behind `test_parallel_deadlock`, keep the band-aids until the no-JAX-in-child path is proven, then remove them in **one deliberate step**, not incrementally.
- **R8 must reproduce the `legal_actions` specialization** (iterate `keep`, not `range(N)`, `minienv.py:99`) — the one method that legitimately differs; the wrong hook silently offers treasures outside the sub-instance and **breaks the provable bound**.
- **R11 must run behind the existing equivalence test**, not around it — the bit-exactness contract is what makes consolidating four forwards safe.
- **R7 copy-on-write** must guarantee no consumer caches a value/K-derived structure on a shared env; audit `FeatureBuilder`'s env snapshot (`features.py:100`) and every `id(env)` consumer before enabling scenario sweeps in the AZ stack.

---

## §8 — Decisions Taken Along the Way (methodological log)

Recorded so the shape of the conclusion can be traced to the shape of the inquiry.

- **Workflow over single-pass.** A 9,600-LOC architecture review delivered as one model pass yields confident, unverifiable claims. The decision to fan out and *then verify* is the reason this document can label findings `confirmed`/`partial` with line proof rather than asserting them. The cost (35 agents, ≈2M tokens) was accepted because the commission was a comprehensive audit, not a spot-check.
- **A dedicated cross-cut agent, not just per-directory agents.** Config-ownership rot lives *between* modules; a reader scoped to one directory cannot see that `0.0855` appears in ten files. The config-flow agent (A.10) existed specifically to catch the disease the directory agents structurally could not, and it produced the SSOT map that organizes §2.B.
- **Refute-by-default verification, capped at 22.** Skeptics were told to default to `refuted` — the asymmetry is deliberate; a verification pass that defaults to "confirm" verifies nothing. The cap at 22 (of 98) was a cost decision, recorded honestly as a coverage limit (§12) rather than hidden; critical findings were ordered first so the cap fell on the weakest-prioritized claims.
- **Verification *before* synthesis, not after.** The two synthesis agents received the verdicts, with instruction to discount anything deflated. This is why the anti-pattern set and target architecture do not inherit the two overstatements the partials caught — the correction propagated forward instead of needing a retraction.
- **`general-purpose` agents (full tool access), not `Explore`.** `Explore` reads excerpts and locates code; it does not read whole files or run them. The audit required full reads and the ability to *execute the env* (which is exactly what caught Partial 2's "not actually stale"). The agent-type choice is the reason a runtime fact entered an architecture review.
- **Preserve-the-good-parts as an explicit charter clause.** Agents were required to earn and cite praise. This is not politeness; an audit that cannot name what is correct cannot protect it during remediation, and the §6 seam-preservation list is the direct product of that clause.
- **Partial-handling: correct in the main record, preserve in the appendix.** When verification deflated a claim, the appendix kept the agent's original (over)statement verbatim and the correction was made only here, in §5. Silently editing the worker record to match the verdict would destroy the traceability the whole exercise exists for.
- **Reviewer grounding reads before *and* designing the fan-out.** Reading nine core files first is what let the phase-1 prompts name specific suspect sites (sharper recall) — and is also the confirmation-pressure risk §10 records. The trade was made deliberately and the adversarial pass was the counterweight.

---

## §9 — Serendipitous Findings

Surfaced incidentally, not targeted by the commission, and worth the record.

- **A doc written to cure staleness that was stale in 24 seconds.** The `2026-06-15` handoff lists as "pending" the `train_value.py` "manual-Adam SGD" docstring fix — git shows that exact fix committed 24 seconds *after* the handoff (`train_value.py:7` now reads "optax-Adam … via JaxTrainer"). A live task queue narrated in immutable prose is stale before it is read *(byte-verified against `git log`)*.
- **The literals are not stale — yet.** The most useful negative result of the audit: running the env proved `realizable_static=0.08553` / `clairvoyant=0.14537` match the frozen `0.0855`/`0.1454`, because floor and ceiling are detector-independent. The SSOT violation is real but its consequence is *latent*, a sharper and more defensible claim than "your numbers are wrong" *(runtime-verified)*.
- **A binding convention with no definition.** `ADR-0002 "fail-loud"` is invoked 16 times and the handoff points readers to "the ADR-0002 registry" — which does not exist. A new contributor cannot look up what the rule requires; the numbering falsely implies a recorded sequence *(grep-verified)*.
- **A dangling pointer to the simulation's heart.** `consult-002 §4`, cited in `env.py:35` and `facemodel.py:5` as the authority for the corrected face-detector model, is misfiled (under `docs/agents/`, not `docs/consults/`) and its report has no §4 anchor. The most correctness-critical modeling decision in the project is justified by a reference that does not resolve.
- **Core-pinning by scraping a CPython-internal string.** `parallel.py:175` assigns each worker's core by parsing the `multiprocessing` process *name* (`PoolWorker-1` → 0) with a fail-soft `except: widx = 0` that silently collapses 4-core parallelism onto one core if the name convention changes across Python versions.
- **A no-op flag shipped as public API.** `info_relaxation.py`'s `restrict_faces` constructor parameter gates a body that is literally `pass` — a configuration knob that configures nothing.
- **A hot-path kernel that is dead in production.** `kernels.marginals_kernel` claims to be the `env.marginals` hot path, but `env.marginals` uses the numpy path; the kernel is not on the live route *(cited-not-rerun, A.10)*.
- **A probe that forks the training loop.** `probes/residual_firewall/ab_train.py` reimplements the entire ExIt training loop and monkeypatches private net internals with hyperparameters that silently diverge from the loop's defaults — an experiment that no longer measures the system it claims to.
- **`Part A/B/C` as load-bearing identifiers.** Nine modules explain their present behavior by reference to ephemeral experiment-session tags ("Part C", "Part B") from one results write-up — code indexed by *when it was written* rather than *what it does*.

---

## §10 — Coordinator Self-Critique

The audit's own failure modes, recorded with the same candor demanded of the code. Eight risks; catch-origin credited.

1. **Confirmation pressure in the charter.** The phase-1 prompts named the malpractice classes to hunt and, for some subsystems, pre-named specific suspect sites (the reviewer had already read those files). This sharpens recall but risks priming agents to confirm the reviewer's priors. *Mitigation that fired:* the independent refute-by-default pass, which deflated two of the reviewer's own framings (the `mlp_jax` "live", the `eval_az` mis-attribution). *Residual risk:* the 76 unverified claims did not get that adversarial check. — *caught by: the verification phase, by construction.*
2. **The 22/98 cap.** Only 22 of 98 critical/major/frozen claims were verified. The cap was a cost decision; the unverified 76 are carried *(cited-not-rerun)* and explicitly **not** load-bearing for any anti-pattern, but a reader must not treat the appendix's unverified findings as having the same evidentiary weight as the 22. — *self-disclosed.*
3. **Single synthesizer per synthesis artifact.** The anti-patterns and the target architecture each came from one agent. A panel would have given an independent cross-check on the *framing* (not just the facts). The reviewer's own reading served as the second opinion, weaker than an independent agent. — *self-disclosed.*
4. **Severity is a judgment, not a measurement.** `critical`/`major`/`minor` were assigned by the agents under a one-line rubric. The two `critical`s that survived verification (MiniEnv duplication, feature-layout 3-writer) are defensible, but the line between `critical` and `major` is softer than the line between `confirmed` and `refuted`. — *self-disclosed.*
5. **Agents could have hallucinated a `file:line`.** The verification pass checked the 22 strongest; the reviewer independently confirmed a sample during grounding reads; not every one of the ~90 findings' citations was re-opened. Spot-checks found the cited lines accurate (verifiers repeatedly noted "all cited line numbers verified EXACT"). — *partially caught by: verifiers + reviewer spot-checks.*
6. **The docs corpus was sampled, not exhausted.** The docs↔code agent read the named design/handoff/consult docs and grepped for citation tokens; it did not read all 40+ docs. The "111 design § / 16 ADR-0002" counts are grep counts, reliable; any *content* claim about an unread doc is not made. — *self-disclosed.*
7. **No dynamic/test-run baseline.** The audit did not run the test suite or a training iteration end-to-end (one verifier ran `realizable_static`/`clairvoyant_rate`; nothing more). Claims about the deadlock substrate (§2.H) rest on reading the code and the RCA, not on reproducing a wedge — appropriately, since the wedge is intermittent and the point is structural, but "the worker can still wedge" is an architectural inference, not a demonstrated event. — *self-disclosed.*
8. **The reviewer authored the target architecture's framing.** The four-tier model is a clean lens, but it is *a* lens; a different principal might weight the parallel-substrate rewrite (R14) or the config object (R12) differently. The roadmap's risk column is honest about R14 being high-risk, but the sequencing reflects one engineer's judgment of leverage. — *self-disclosed.*

**Systemic synthesis of the self-critique:** the audit's evidentiary spine (the 22 verified claims, the seam-preservation list, the §4 trace) is strong and line-grounded; its softer edges are *severity calibration*, *the unverified tail*, and *single-perspective synthesis*. None of those undermine the central diagnosis (§1), which rests entirely on verified facts.

---

## §11 — Maintainer Decision Points

Open questions that require human judgment, not a default.

- **`K` in `instance.json` (R1).** Moving `K` into the data file is the clean fix, but it changes the file's schema and every reader. Confirm no external tooling parses `instance.json` with a fixed key set before migrating.
- **Adopt vs delete `facemodel.SenseAction` (R15).** The audit is agnostic on *which* — both end the dead-vs-live dual encoding. Adoption makes the face the single carrier (the documented `ENV_ADOPTION`); deletion is cheaper. This is a taste-and-roadmap call: is a richer sense-action abstraction wanted for the heterogeneous-value work, or not?
- **Is R14 (numpy-only worker) worth the risk now?** It removes the substrate that spawned seven band-aids, but it is the highest-risk item and touches the one subsystem with intermittent failures. The alternative is to leave the band-aids and treat the parallel path as "works, don't touch." Decide against the project's appetite for parallel-throughput work.
- **The het-values experiment is the forcing function.** The handoff names heterogeneous gil values as the key next lever. That experiment is *exactly* the one the frozen-`value` monolith (§2.A) and the `id(env)` caches (§2.C) make painful. Recommend R7 (Scenario) land *before* the het-values work, so the experiment is a comprehension over `Scenario`s rather than a monkeypatch. Confirm the sequencing.
- **ADR registry: write it or retire the convention (R15).** Either create `docs/adr/` so the 16 `ADR-0002` cites resolve, or demote "fail-loud" to a real testable `fail_loud()` helper. Shipping a cited-but-nonexistent registry is the one option not on the table.

---

## §12 — Coverage & Limits

What was read, against what authority, and what was not.

- **Read in full, by a dedicated agent, end-to-end:** `model/{env,arrangement,facemodel}.py`, `data/{instance,faces}.json`; `solvers/{base,uct,ismcts,nmcs,decomp}.py`; `az/{mlp,mlp_jax,mlp_jax_train,train_value,dtypes,kernels,dataset}.py`; `az/{gumbel_search,features,actions,feature_response,value_target,netvalue_ismcts}.py`; `az/{exit_loop,parallel}.py`; `eval/*.py` (all 8); `bounds/{info_relaxation,eval_bound,minienv}.py`; `analysis/{analyzer,synthetic}.py`; `tests/*.py` (all 4). — *binding evidence: full-read authority.*
- **Read in full, independently, by the orchestrating reviewer** (cross-check): `env.py`, `solvers/base.py`, `exit_loop.py`, `parallel.py`, `features.py`, `actions.py`, `mlp_jax_train.py`, `eval_az.py`, `eval_uct.py`. — *binding evidence: reviewer corroboration.*
- **Read partially / by cross-cut:** `probes/residual_firewall/*.py`, `scripts/*.py`, `az/bench/*.py` — covered by the analysis and config agents, not given a dedicated reader; claims about them are *(cited-not-rerun)*.
- **Sampled, not exhausted:** the `docs/` corpus — the named design/handoff/consult docs were read; citation tokens were grep-counted tree-wide; unread docs carry no content claims.
- **Excluded:** `attic/` (explicitly dead; read only for corroborating context, e.g. the het-values monkeypatch precedent).
- **Verified by execution:** exactly the `realizable_static`/`clairvoyant_rate` recompute (Partial 2). No test suite run, no training iteration, no deadlock reproduction.
- **Verification coverage:** 22 of 98 extracted critical/major/frozen claims adversarially checked; 76 carried *(cited-not-rerun)* and not load-bearing for any §1–§7 conclusion.
- **Git authority:** the 24-second handoff-rot finding is verified against `git log` *(byte-verified)*; the working tree was clean at audit time (`main@cfce276`).

---

## §13 — Lessons for the Record

Durable, transferable lessons distilled from this audit — the things worth carrying to the next subsystem or the next project, each with a one-line disposition. These are *why the rot formed*, not restatements of *where it is* (that is §2–§4).

| # | Lesson | Disposition |
| — | — | — |
| L1 | **A value's "heat" is decided by where it lives, not by intentions.** A knob assigned to `self.X` in `__init__` is cold no matter how often you mean to sweep it; the same knob arriving as an argument is hot for free. | The four-tier model (§6) makes heat structural. λ proves it works; everything frozen is a value placed in the wrong tier. |
| L2 | **The proof a codebase *can* do it right is the indictment when it doesn't.** `feature_dim(env)` (derive) and frozen `STATIC_FLOOR` (duplicate) sit in the same package. | Generalize the existing good pattern outward; you are not introducing a discipline, you are finishing one. |
| L3 | **Duplicated knowledge is a time-bomb whose fuse is the next edit, not today.** Every byte-identical copy (belief math, world prior, forward graph, reference rates) is correct until the first one-sided change. | One owner per fact (§2.B). The decomp anchor already drifted (`0.0941`/`0.094`) — the fuse already lit once. |
| L4 | **A test that pins a derived literal forbids the legitimate change it should permit.** `test_smoke.py:94` breaks on any honest model retune. | Assert the *recompute is sane*, never the frozen number. A guard that punishes correctness is an anti-guard. |
| L5 | **The worst duplication is the one that validates the original.** `MiniEnv` re-implements the dynamics the dual bound *certifies* — so the certifier and the certified can silently disagree. | Share the mechanics (§7 R8); never let a correctness oracle carry its own copy of the thing it checks. |
| L6 | **A parameter the receiver cannot honor is a lie in the signature.** `train_epochs(lr, l2)` ignored; `build(marg)` ignored; `restrict_faces` gates `pass`. | Delete it or make it real. A dead seam costs more than no seam — it invites wasted work (`netvalue_ismcts.py:54`) and stale-config bugs. |
| L7 | **Keeping the right abstraction *next to* its inline copy is worse than not building it.** `facemodel.SenseAction` is the clean object; the env reimplements it inline. | Adopt or delete (§7 R15). A reader cannot tell which encoding is authoritative, and they drift in silence. |
| L8 | **When the reliability strategy becomes a stack of patches, the substrate is the bug.** Seven deadlock band-aids fight one JAX-in-the-worker decision. | Remove the conflict at the root (numpy-only worker, R14); a correctness test that can only prove "fails loud" is not proving "works". |
| L9 | **Conventions that live only in prose are not conventions.** `ADR-0002` is cited 16× and its registry does not exist. | Make it testable code (`fail_loud()`) or a real registry. An unenforceable rule is a decoration. |
| L10 | **Docs that mirror a live task queue rot faster than anyone reads them.** The handoff's "pending" item was done 24 seconds later. | Status docs record slowly-aging decisions and rationale; the live queue belongs in the issue tracker, not immutable prose. |
| L11 | **`id()` is the wrong key for any object without value-equality.** `_SLOT_TABLES[id(env)]` is masked only because every env is identical today. | Own the derived data on the object whose lifetime it shares (`env.slot_tables`); never key a cache on an address. |
| L12 | **Verify before you synthesize, and keep the worker's error visible.** The refute-by-default pass deflated two of the reviewer's own framings; the appendix preserves the originals. | Auditability is the product. A conclusion you cannot trace to a re-opened line is a guess wearing a citation. |

---

## §14 — Closing Thoughts (bird's-eye)

Step back from the `file:line` and the shape is simple. This is a codebase that **made every hard architectural decision correctly and then, under the relentless churn of research, stopped extending the cheap discipline that follows from those decisions.** The inversion of control between simulation and solver — the thing that is genuinely hard to retrofit and genuinely catastrophic to get wrong — is right. The derived-dimension discipline is right. The single hardest live value, λ, is owned and threaded right. These are not accidents; they are the work of someone who understands the craft.

And then the same hands froze the value vector at construction, copied the belief mechanics into a second file the bound certifies against, wrote the headline metric's divisor as a literal in ten places, transcribed one forward graph four times, and offloaded the explanation for all of it into prose documents that drift in seconds and reference registries that do not exist. **None of this is incompetence. All of it is the entropy of a fast-moving research project whose abstractions grew slower than its features.** The `feature_dim(env)` discipline — derive, never hardcode — is sitting *right there* in the same package as the three hardcoded reference constants. The team knows the answer. They wrote it down once and then ran out of time to apply it the other forty times.

That is why the verdict is "salvageable without a rewrite" and why I would resist anyone who reaches for one. A rewrite throws away the correct hard decisions to fix the entropy in the easy ones. The actual work is overwhelmingly *subtraction and relocation*: delete the second copy of the belief math; move the value vector into a Scenario; route ten files through one `BeliefRefs(env)`; collapse four forwards behind the equivalence test that already exists to make that safe. The single structural risk worth taking deliberately — the numpy-only worker — is isolated, gated by an existing test, and optional. Everything else is the codebase finishing a sentence it started correctly.

Do the immediate tier first; it is pure subtraction and cannot regress behavior. Then fix the feature layout before it bites someone during the exact heterogeneous-value experiment the whole project is building toward — because that experiment is the one the current architecture is least ready for, and it is next on the maintainer's own list. Stop apologizing for the good decisions in comments and let the structure carry the knowledge instead of the prose. The bones are good. Finish the skeleton.

---

**Companion record.** Full verbatim worker outputs — 11 component deep-reads, 22 verification verdicts, 8 anti-patterns, and the target architecture — are reproduced in `architectural-audit-2026-06-15-appendix.md`, assembled mechanically per the verbatim-record discipline. This document and its appendix are point-in-time and not retro-edited.

*Public Domain / The Unlicense, per repository convention.*
