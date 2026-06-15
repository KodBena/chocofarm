# Handoff — architectural refactor session (2026-06-15)

Prepared for the next contributor. Read end to end before relying on it (ADR-0002).

**This handoff supersedes the "Condition of the Repository" section of
`docs/handoff-2026-06-15.md`.** That prior memorandum is a point-in-time record of the
JAX-training-migration session and is left un-retro-edited (ADR-0005 Rule 8); but its
repository picture (`main` at `2cbbbd8`, an unmerged `feat/az-jax-train →
fix/jaxtrain-deadlock → cleanup/remove-numpy-adam` branch stack, "merge the stack" as
pending item 1) is now obsolete. That stack is long merged; `main` is at `f262fe6`; and
the architectural audit's structural roadmap has since been executed in full. Everything
else in the prior handoff — the rate frontier, consult-003, the dual bound — still stands.

## What this session did

Executed the 2026-06-15 architectural audit (`docs/notes/audit/`) — its R-series of
structural defects — to completion. The audit and its prior-audit synthesis are the
point-in-time roadmap; this is the dated record that the roadmap landed. The work was
orchestrated as file-disjoint, worktree-isolated subagents, each gated on
behavioural-equivalence (float32-roundoff, not byte-identity — this is machine learning)
and merged only after an independent verify-before-merge pass.

## Repository condition

- `main` at **`f262fe6`** (pushed to origin `github.com/KodBena/chocofarm.git`).
- Test suite: **138 passing** (was ~39 at audit time; the growth is the new SSOT/collaborator
  unit tests + the parallel-substrate determinism pins, not behavioural change).
- No session-born branch is left unmerged. The per-item worktrees were pruned at close.

## The roadmap, executed (newest last)

Each item is a merge commit citing its audit item; the audit (`docs/notes/audit/`) carries
the rationale. Behavioural equivalence (or byte-identity where it was free) was the gate
on every training-path change.

- **R4/R5** `7e0fff0` — one episode-horizon + solver dedup; drop dead `marg`.
- **R6** `70f7c1b` — single-source the feature layout via `FeatureLayout`.
- **R7** `2750c3a` — `Scenario` + `Environment.with_scenario` copy-on-write.
- **R8** `50e6349` — fold `MiniEnv` into `Environment.restrict`; one belief-mechanics impl.
- **R9** `2e43642` — `WeakKeyDictionary` env caches; kill the `id(env)` leak + aliasing.
- **R10** `440e9c1` — one `BeliefRefs`-based eval runner + `SOLVERS` registry.
- **R11** `157bef4` — one `ForwardSpec` core for all four forwards; resolve the residual-drop.
- **R13** `846d5fe` — live `lr`/`l2` via `optax.inject_hyperparams` (the frozen-config headline).
- **A+B** `5c23e75` — ADR corpus + `CLAUDE.md` adapted from LengYue; resolve the `consult-002` pointer.
- **D** `c7249ec` — V̂ Strategy Port (`bounds/vhats*.py`); split by dependency, dissolve the lazy import.
- **G** `38aa6ab` — pin `FeatureConfig` per-group counts to the `FeatureLayout` SSOT.
- **C** `bef298e` — extract the AZ policy-target rule into `az/value_target.py`.
- **E** `3e0b871` — adopt `facemodel.SenseAction` as the env's single face-carrier (geometric
  derivability preserved: worlds remain computable from the atomic detectors' intersection refinement).
- **M** `009454c` — Optimizer⊥Trainer split + live `betas`/`eps` (`az/optimizer.py`).
- **I** `f83e224` — per-solver `SearchConfig` dataclasses, back-compatible.
- **H** `eae0ec4` — public `Environment.keep` accessor; `eval_bound` stops reading `_treasure_ids`.
- **J** `108e0ad` — `WeightContainer` owns the weight layout (`az/weights.py`): params/L2-mask/npz/transport.
- **F** `127b4f7` — relocate belief-reference machinery to neutral `chocofarm/references.py` (cuts the az→eval edge).
- **K** `bc3f2da` — split `ParallelExecutor` into `RedisTransport`/`WorkerPool`/`worker`+`TaskSpec`.
- **L** `f262fe6` — numpy-only JAX-free worker + `Worker` object + `(run,phase,version)` namespacing
  (retires the `it+1_000_000` hack; removes the deadlock root cause — see the dated R14 amendment in
  `docs/notes/jaxtrain-deadlock-rca.md`).

Net structural effect: the god-objects are split (the parallel substrate; the weight layout;
the optimizer), the single-source-of-truth violations are closed (feature layout, weight layout,
belief references, the `keep` set), the `az→eval` back-edge is severed, and the worker is a clean
numpy/numba host with the deadlock root cause removed rather than band-aided. The one clean seam
(`env`↔`Policy`) was the template throughout; no item fought it.

## Deliberately deferred (forward-looking, not malpractice — documented, not built)

These are new capabilities the architecture should make *fall out*, not existing defects to
excise. They were consciously left for later and have design records:

- **R12 `RunConfig`/`ExperimentConfig`** + a typed `sweep()` helper — the Python-native config
  story. `docs/design/experiment-config.md` (with the verdict that Dhall is over-engineering here).
- **A `Net` port** (`predict_both`/`predict_value`) with `local-numpy | zmq-client` impls — unblocks
  a batched ZeroMQ inference service. `docs/design/scaling-and-cpp-seam.md`.
- **The sync→async actor-learner loop** — the only genuinely new structure; the seams now compose
  straight into it. Same note, §3/§4.

The K/L split was chosen to make the C++/numba worker and the continuous actor-learner shapes fall
out of existing seams; `scaling-and-cpp-seam.md` is the target to check those against.

## The actual frontier is unchanged

The architecture work is plumbing; it does not move the rate. The research priority remains exactly
as `docs/handoff-2026-06-15.md` and `docs/consults/consult-003-marry-static-az-katago.md` state it:
**value-function calibration** (drive the belief state-value Bellman residual to zero), which by
strong duality simultaneously tightens the certified ceiling (`docs/design/dual-bound.md`) and
improves the policy. The rate plateau (~0.10, %VoI ~+20–27%) and the loose clairvoyant ceiling
(0.1454) are unchanged. Start there, on the now-clean code.

## Operational facts

Unchanged; the canonical source is `CLAUDE.md` and the prior handoff's "Operational Particulars".
Interpreter `/home/bork/w/vdc/venvs/generic/bin/python`; redis `6380` (transport) / `6379` (registry);
4-vCPU host, pin `--cores 0,1,2,3`; pushing session-born branches needs the maintainer's consent.

*Public Domain (The Unlicense).*
