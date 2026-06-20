<!--
docs/notes/lab-staging-divergence-rca.md — postmortem RCA: the params-staging
consolidation (b5df1e2) that did not propagate to the control lab, the interim
re-align that crashed because the structural fix was deferred, and the server
topology that reframed the severity. Point-in-time record (ADR-0005 Rule 8 —
amend by append, never silently rewrite).
Public Domain (The Unlicense).
-->

# RCA — the params-staging consolidation that did not reach the control lab

**Date:** 2026-06-20 · **Branch:** `feat/issue-control-lab` · **Genre:** postmortem
(the `docs/notes/jaxtrain-deadlock-rca.md` convention)

## Summary

The params-staging consolidation (`b5df1e2`) routed `InferenceServer`'s forward
through a new `_effective_forward` seam (weights staged device-resident,
~50–80 µs/forward). `StageAServer` and `LabServer` **override `_serve_batch` and
hand-copy the forward dispatch**, so the staging silently did not propagate to
them — a lineage of **three** hand-copies of one dispatch (base + StageA + Lab),
of which the consolidation upgraded exactly one. The **deepest root is P3**: a
god-method (`_serve_batch`) that welds the forward dispatch to the serve/control
boundary, so a subclass that needs to extend the boundary is *compelled* to copy
the dispatch.

The proper fix (a template-method split) was **deferred** in favour of a one-line
interim re-align. The interim re-align **crashed on its first bucketed forward** —
the single-shape staged handle (compiled for `pad_to=max_batch`) is incompatible
with the lab's multi-bucket pad policy — **confirming the RCA's own verdict that
the band-aid was insufficient, within minutes of landing it.**

A topology check then **reframed the severity downward**: the lab is a
*deliberately separate bench* (`LabServer`'s docstring: "does NOT touch the
production eval path"), `StageAServer` is "a THROWAWAY bench-scoped" server, and
production is `InferenceServer` (which *did* get the staging). So the feared harm
— a trajectory corpus poisoned by a regime shift — **never materialised**: the
consolidation never reached the lab, the lab's regime is unchanged, and the
corpus is valid. The structural smell is real and is being fixed; the disaster we
both took it for was not one.

## 1. What happened

- `b5df1e2` made `InferenceServer._serve_batch` resolve its forward via
  `forward_fn = self._effective_forward(params, y_mean, y_std)` — the staged,
  device-resident-params `LowLatencyFn` for the default `jit_forward_core` path.
- `cpp/stage_a/stage_a_server.py:121` (`StageAServer._serve_batch`) and
  `cpp/stage_a/control_lab/lab_server.py:230` (`LabServer._serve_batch`) **override
  `_serve_batch` and call `run_microbatch(self._forward_fn, …)` directly** — the
  raw un-staged forward, fetching params via `self._params_source.current()` and
  never touching `_effective_forward`. (Two throwaway repro servers under
  `docs/design/stall-investigation/empirical/standalone_server{2,3}.py` carry the
  same shape — five bypass sites total.)
- So the lab measured its issue-gate controllers against the un-staged forward,
  ~50–80 µs/forward slower than the staged base — a silent offset no test or gate
  flagged.

## 2. The topology that reframed it

| server | role | pad policy | staged? |
| — | — | — | — |
| `InferenceServer` (via `CppActorExecutor`, `cpp_executor.py:256`) | **production** serve actor | pad-to-max (`B=pool_batch`) | **yes** (`b5df1e2`) |
| `StageAServer` | "THROWAWAY bench-scoped" (its docstring) | bucket-E `(64,256,512)` | no (bypass) |
| `LabServer ← StageAServer` | the control lab — "does NOT touch the production eval path" | bucket-E | no (bypass) |

The decisive facts: **production is `InferenceServer`, and it does get the
staging** (so the consolidation was a real production win, not a theoretical one);
and **the lab is a separate bench that never shared production's pad policy**
(bucket-E vs pad-to-max), independent of staging. The consolidation never reached
the lab, so there was no regime shift to invalidate the corpus. The premise that
sent us re-aligning — "the lab is now stale relative to production" — was a
misunderstanding: the lab was *never* a mirror of production.

## 3. Root cause — P3 (no god-objects)

`_serve_batch` owns **two orthogonal concerns welded into one overridable
method**: (a) the *forward dispatch* (which params, staged or not, the pad shape,
`run_microbatch`) and (b) the *serve/scatter/controller boundary* (decode
envelopes, run the Controller, tag gate frames, `send_multipart`). The lab's
legitimate need is to extend (b). But because (a) and (b) are the same method, the
only way to touch (b) is to override the whole method — which **mechanically
compels** copying (a). That compelled copy is then a **P1** violation (the forward
dispatch has three homes that must agree and silently don't) and **cancer E** in
textbook form (`_effective_forward` sits fully built while the live lab path
hand-inlines `run_microbatch` beside it). Collapse the P3 conflation and the
subclass has nothing to copy — P1 and E cannot be authored. **P3 is the single
deepest root; P1 and E are its symptoms.**

## 4. The operational lapse

- **No override audit when changing a base method.** The consolidation's
  correctness depended on *every* `_serve_batch` routing through
  `_effective_forward`; a one-line `grep _serve_batch` returns all three. The
  change was authored, tested, and reviewed as a closed in-module edit — the
  question *"who else overrides the method I just changed?"* was never asked.
- **Tests covered the base seam in isolation.** `b5df1e2`'s new tests drive
  `_effective_forward` / `build_staged_forward` directly and never instantiate
  `StageAServer` or `LabServer` — nothing asserted "the lab's forward == the
  base's staged forward."
- **The nets are blind to `cpp/`.** Both the mypy `--strict` gate
  (`files=["chocofarm"]`) and the host-device lint (`PACKAGE_ROOT=chocofarm`)
  scope to `chocofarm/` — the lab lives in `cpp/`, outside every guard. The
  "throwaway bench" framing in the lab's own docstring is what put it there, while
  it was in fact load-bearing for a real decision.
- **The documentation tell.** ADR-0012's own 2026-06-20 amendment documenting
  this consolidation never once mentions a subclass or an override — the override
  surface was not in view even in the author's point-in-time record.

## 5. Could a lint or mypy have caught it?

- **mypy — no.** It is a semantic divergence in *type-identical* code:
  `run_microbatch(self._forward_fn, …)` and `run_microbatch(self._effective_forward(…), …)`
  are both well-typed and return the same type. mypy checks signatures, not which
  callee a body invokes — and `cpp/` is out of the gate's scope regardless.
- **A structural AST lint — yes,** and the host-device lint is the exact template:
  flag a `_serve_batch` override whose body calls `run_microbatch` with
  `self._forward_fn` rather than delegating to `_effective_forward`. Low
  false-positive (private names); **must be scoped to include `cpp/`**. A
  build-time net over the *shape*, not unrepresentability.
- **A subclass parity test — yes,** the behavioural backstop: stand up a real
  `LabServer`/`StageAServer` and assert its served output `allclose` to the base on
  matched input (+ a spy that the staged seam was hit). The test that was missing.

## 6. The fix (ADR-0011 strongest-feasible hierarchy)

1. **The P3 template-method split (unrepresentable-by-construction).** A *sealed*
   forward seam (`_run_forward` — owns `poll`/`current`, `_effective_forward`, the
   pad shape, `run_microbatch`) that no subclass overrides, and an *overridable*
   boundary hook the lab/StageA override for their Controller/scatter. The subclass
   then **cannot** re-author the forward — it never sees `run_microbatch`. (The
   sealed seam must return the real-row-count + pad-bucket metadata the lab's
   reward/observe needs — a small contract, not a reason to keep the methods
   fused.)
2. **A subclass parity test** — the P6 behavioural backstop.
3. **Extend the host-device lint (and, in time, the mypy gate) to `cpp/`** — close
   the blind spot the lab lives in.
4. **Lab alignment** — if the lab's gating findings are to *transfer to
   production*, the bench should mirror production's pad-to-max + staged regime
   (which also dissolves the single-shape/bucket crash). Tracked separately.

## 7. Epilogue — deferring the structural fix bit immediately

The RCA's verdict was explicit: *"removing the duplication fixes the instance but
not the class; the P3 split is needed."* The structural split was nonetheless
deferred in favour of a one-line interim re-align (`12b27bf`: `LabServer`
delegating its forward to `_effective_forward`). It **crashed on its first
bucketed forward**:

```
TypeError: Argument 'x' compiled with float32[512,241] and called with float32[64,241]
```

`build_staged_forward` AOT-compiles for one fixed shape (`pad_to=max_batch`=512);
the lab's bucket-E policy feeds `[64,241]`/`[256,241]`; the single-shape handle
rejects them. `warmup` did not catch it (it probes at max only). The re-measure
agent surfaced it with a real traceback and correctly refused to paper over it
(`padmax` would compile-match but change the throughput regime, contaminating the
comparison). `12b27bf` was reverted (`5df7a45`).

Two failures compounded:

- **The structural fix was deferred** (the executive decision below). Had the
  forward dispatch been a sealed seam reconciling the pad policy with the staging,
  the bucket/single-shape incompatibility would have been confronted in the seam's
  design rather than discovered as a live crash. The band-aid re-created the very
  hazard the RCA named.
- **The interim fix was verified too weakly.** It was committed and pushed behind
  `py_compile` + an import smoke — which cannot catch a runtime shape mismatch on a
  `cpp/` path that the mypy and lint gates do not cover. The "behaviour-preserving"
  claim was asserted, not earned.

## 8. Ownership

- **Executive (the maintainer):** deferring the *proper* P3 fix in favour of the
  interim re-align — a lapse of ADR-0012 discipline (structure born clean is
  upstream of any band-aid; a fix that re-creates the hazard is not a fix). Owned
  fully and recorded here at the maintainer's explicit request, so the lesson is
  not lost: *when an RCA names the structural fix as load-bearing, do not ship the
  band-aid first.*
- **Execution (the collaborator):** committing and pushing a crashing change
  (`12b27bf`) behind verification (`py_compile` + import) categorically too weak to
  substantiate the behaviour-preserving claim on an un-gated `cpp/` path — the same
  fail-loud-vs-verify discipline (ADR-0002 / ADR-0009) the codebase holds elsewhere,
  not applied here.

The honest composite: an under-treated structural hazard, shipped as a band-aid,
behind verification that could not see the failure. The corpus survived only
because the lab turned out to be a separate bench — luck the process did not earn.

## License

Public Domain (The Unlicense).
