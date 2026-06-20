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

## 9. Resolution — the P3 split landed (appended 2026-06-20, ADR-0005 Rule 8)

*Additive amendment; §1–8 stand unedited. This records that the load-bearing fix
§6.1 named — deferred in §7 — has now landed, with the iron-clad verification the
band-aid lacked.*

**Fix #1 (the P3 template-method split — unrepresentable-by-construction) landed.**
`InferenceServer._serve_batch` is now a thin `final`: it calls a **SEALED**
`_run_forward(drained) -> (responses, forwards)` that owns the *whole* forward
dispatch (poll/current params, the fixed-pad-gated `_effective_forward`, the pad
shape, `run_microbatch`), then an **OVERRIDABLE** `_scatter(drained, responses,
forwards)` boundary hook. The two focused hooks the dispatch reads — `_pad_shape`
(per-forward pad policy) and `_forward_groups` (the drained→forwards partition) —
let a subclass vary the pad/grouping without touching the dispatch. `StageAServer`
now overrides `_pad_shape` (bucket-E) + `_forward_groups` (group/leaf) + `_scatter`
(counters); `LabServer` overrides `_scatter` (the Controller call + gate-frame
tagging) and inherits the rest. **Neither overrides `_serve_batch`/`_run_forward`**,
and the only `run_microbatch(...)` serve-path call-site is inside the sealed
`_run_forward` — so the subclass *cannot* re-author (and silently diverge) the
forward (the override-divergence bug class §3 named is now unrepresentable). The
hand-copies in both subclasses (and the lab's dead `_bucket_for_server`) are gone.

**The single-shape/bucket incompatibility (§7's crash) is confronted in the seam,
not as a live crash.** The staged single-shape AOT handle is valid only for a
*fixed* pad; the seam encodes this as an explicit `_uses_fixed_pad` predicate
(`InferenceServer` True → staged pad-to-max; `StageAServer`/`LabServer` False →
**un-staged** `self._forward_fn` at their bucket shapes), which
`_effective_forward` checks. So a bucketing server can never feed the single-shape
handle a non-max bucket — the lab serves its un-staged bucket-E forward exactly as
before, and the lab-alignment step (mirror production's pad-to-max + staged regime)
becomes a **clean flip** of that one flag (fixed-pad ⇒ auto-staged, no crash),
tracked separately as fix #4. **Behaviour-preserving:** each server's current
forward is byte-identical (InferenceServer staged pad-to-max; StageA/Lab un-staged
bucket-E) — a pure structural refactor, no pad/staging policy changed here.

**Fix #2 (the subclass parity test — the P6 backstop §5/§6.2 said was missing)
landed.** `tests/test_zmq_inference.py::test_subclass_servers_parity_with_base`
stands up a real `StageAServer` and a real `LabServer` (the latter driven over its
lab FEATURE/GATE envelope) alongside the base and asserts each one's served
`(value, logits)` is `allclose(1e-4)` to the base's on a matched `(params, X)` —
the exact test that would have caught a subclass serving a different forward. It
passes (max|Δ| ≈ 1.2e-7, residual ON/OFF).

**The verification this time was iron-clad** (the §7/§8 lesson applied — not
`py_compile`+import on the un-gated `cpp/` path): (a) a SHORT lab sweep was **run
end-to-end** through `lab_harness.py` (`all_allow`, `ready_threshold2`, 2.0s
windows, server pinned core 0 / producers 1,2,3) and **did not crash** — the warm
pool primed, the C++ producer streamed `pipelined-bucket` with **zero** errors, the
bucketed forwards (rows/fwd ≈ 44) served and gate bits rode back (`ready_threshold2`
diverged from `all_allow`, `method_metrics{threshold:2.0}`), **0 malfunctions / 0
flags** both trials (output under `~/w/vdc/chocobo/runs/control_lab/p3-split-verify-*`);
(b) the subclass parity test above passes; (c) the full `tests/` suite + the opt-in
`CHOCO_RUN_ZMQ=1` socket parity + `mypy --strict` (`inference_server.py` is in the
gate's `STRICT_CLEAN` set) are green, and `tests/test_no_gratuitous_transfers.py`
still holds (the grandfathered `run_microbatch::np.asarray` pull is untouched —
`run_microbatch` itself was not edited).

**Still open (unchanged scope):** fix #3 (extend the host-device lint + the mypy
gate to `cpp/` — the blind spot the lab lives in) and fix #4 (the lab-alignment
pad-to-max + staged flip) remain forward work, tracked separately. The P3 split
makes a *forward-dispatch* lint largely redundant (there is no second dispatch to
drift), but the `cpp/`-coverage gap that hid this is worth closing on its own.

## License

Public Domain (The Unlicense).
