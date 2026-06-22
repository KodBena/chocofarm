<!--
docs/notes/leaf-eval-witness-lowering-review-2026-06-22.md
Purpose: a RECORD of an external review of the impl->model diagnostic-loop consultation
  (docs/design/leaf-eval-impl-to-model-diagnostic-loop.md), which the maintainer interrogated
  specifically on WITNESS-LOWERING (how to build the witness). It captures the review verbatim
  (the maintainer notes they deliberately biased it toward out-of-the-box thinking) plus a synthesis
  of its contributions and the one caveat the bias understates. Input for the maintainer's eventual
  "build the witness vs run degraded" decision (consultation §11.1); no action is taken by this note.
ADR-0005 (a point-in-time record — amend by append, never rewrite); ADR-0006 header.
Public Domain (The Unlicense).
-->

# Witness-lowering — an external review (recorded 2026-06-22)

The implementation→model consultation named the **witness** as its single biggest open question
(§5.1, §11.1): the witness is the one object legitimately comparable to *both* `f(μ̂)` and the
implementation, but it does not exist yet, and building it is the largest piece of net-new work. The
maintainer interrogated an external reviewer specifically on *witness-lowering* — **deliberately
biasing toward out-of-the-box thinking** (the maintainer's own note: *"I did bias it, but that's
because I needed reflection on out-of-the-box thinking"*) — and recorded the result here. It is
**input**, not a decision; the build-vs-degraded call (consultation §11.1) remains the maintainer's.

## The review, verbatim

> *Noting with appreciation* the two-layer distinction you draw, which names the design's actual hinge rather than its surface,
> *Recalling* the Sovereign Repositories' own stricture that the witness is "a composition, not a rewrite" (MANUAL §2.1) — it must reuse the implementation's real stages, never reimplement them,
> *Having considered* how a refactor-and-port mandate could supply that composition with minimal confounding,
>
> 1. *Affirms* the two-layer reading and sharpens it to the type level: Layer 1 is the *stage semantics* — the benchmarks, each factor denoting a cost `Estimate(mean, CI, ShrinkLaw)` in isolation; Layer 2 is *composition*, and the decisive point is that it carries **two** maps — the denotational `f`, which composes the *means* (total, cheap, coordination-free by construction because a function has no time), and the operational **witness**, which composes the *executions* (partial, expensive, coordination-paying); on this reading `gap_A = f(μ̂) − witness` is *literally the gap between the two composition semantics*, the witness is the **adequacy witness** for `f`, and the question "is `f` faithful?" is exactly "does the denotation agree with the operation, up to the declared coordination budget?" — your instinct that the witness "defines the meta-semantics" is precisely correct;
>
> 2. *Endorses* refactor-plus-throughput-ports as the right *enabling* mechanism, but conditions it on aim: the refactor's real purpose is to **carve the cycle at the model's own joints** — one callable unit per model term, the typed signature as SSOT (ADR-0012) — so that witness and implementation *share the identical stage code* and differ only in composition glue, which is the entire difference between a witness and a second model; a "port" is then the instrumented seam *at a carved joint*, never a wrapper bolted onto an unrefactored blob, because the wrapper reintroduces exactly the mock and boundary-mismatch confounds the witness exists to escape (and the carve should land port-per-term, so the port-set is isomorphic to `f`'s factorization by construction — the cleanest possible anti-boundary-mismatch design);
>
> 3. *Recommends* that the ports be **passive** — counters and ring-buffered timestamps parsed offline, in the established shape of `CHOCO_EVENTLOG` / `mean_rows_per_msg`, never synchronous logging in the hot path — since a probe that takes a lock or blocks on a write will *serialize otherwise-overlapped work* and thereby manufacture the very coordination cost you are trying to measure;
>
> 4. *Urges*, as the centerpiece, that the witness be built not as one artifact but as a **ladder from the idealization downward**, each rung replacing exactly one *modelled idealization* with its *measured reality*: rung 0 is `f(μ̂)` itself; rung 1 is the *real stages under an idealized in-process composition* (perfect overlap, zero transport) — which should land near `f(μ̂)` if the stages were benched faithfully, so **rung-1-vs-rung-0 is an operational stage-fidelity check** that surfaces a lying benchmark before any coordination analysis; rung 2 swaps in the real ZMQ framing (the delta = transport cost); rung 3 the real parking/drain timing (the delta = convoy / RTT-idle); the implementation is the last rung (the delta = redis-stall and unmodelled-source cost); and because consecutive rungs differ by *one controlled swap* under a *constant* probe, each throughput delta is cleanly attributable **and the probe's overhead cancels in the delta** — so the design target is not a zero-overhead port (unattainable) but a *consistent* one (sufficient), which is what makes confounding tractable; this also turns even the B↔S feedback into a *named rung* (fixed-`B` composition minus fed-back-`B` composition = the coupling's throughput cost), and is plainly §8's one-change / pre-registered-prediction discipline applied to *construction* rather than diagnosis;
>
> 5. *Cautions* that the enabling refactor is itself a confound unless **proven throughput-neutral at the ports**: a refactor that silently repairs or introduces a stage cost while "just extracting" builds the witness on a moved target, so every extraction step must show pre/post port readings agreeing within CI (ADR-0009 substantiation) — the diagnostic loop's own gate turned recursively on the phase that constructs its instrument;
>
> 6. *Notes* the irreducible boundary, so the mandate does not over-promise: ports reveal stage *costs* but not *couplings*, because a coupling lives in the composition, not at a seam; the ladder converts *known* couplings into rungs and *bounds* the unknown ones (the residual surviving all named rungs), but it cannot *name* a coupling no one has identified — that residual is §7.3's qualitative-physics reasoning, and a residual that persists across the whole ladder is precisely the §7a *tool-inadequate* verdict (the separable DSL cannot carry a coupling that has not been found);
>
> 7. *Observes*, as a dividend that pays for the ports twice over, that throughput-seams operationalize Purpose 2: the gradient `a_i` *predicts* which stage binds, the ports *measure* whether it binds in the running cycle, and a disagreement — the in-situ binding seam differs from the model's `min()`-kink — is a first-order form finding read directly off the hardware rather than inferred, so the same instrument that assembles the witness also adjudicates the model's bottleneck.

## Synthesis — what it contributes, and the one caveat the bias understates

**The deepest contribution is the type-level reframe (point 1): the witness is the *operational
semantics* to `f`'s *denotational semantics*.** `f` composes the stage *means* — total and
coordination-free, "because a function has no time"; the witness composes the stage *executions* —
partial and coordination-paying. So `gap_A = f(μ̂) − witness` is not a vague "optimism budget" but
*literally the adequacy gap between two composition maps over the same stage semantics*, and
"is `f` faithful?" becomes the precise, checkable "does the denotation agree with the operation up to
the declared coordination budget?" This elevates the witness from the MANUAL's "a runnable cycle"
to **the adequacy witness for `f`** — and it is the right name for what the consultation's §5.1
three-point comparison was reaching for.

**The most actionable contribution is the ladder (point 4).** Rung 0 = `f(μ̂)` → rung 1 (real stages,
idealized composition — the *operational* stage-fidelity check) → +transport → +parking/drain →
implementation; each delta is one named coordination cost; the probe overhead **cancels in the
delta** (a *consistent* port, not a zero-overhead one, is the achievable target). This is the
constructive form of the consultation's three-point comparison: it turns `gap_A` into a *sequence* of
attributable deltas, and — crucially — **rung-1-vs-rung-0 catches a lying benchmark *operationally***,
complementing the consultation's §7 `bench`-vs-`impl` reading with a second, composition-side fidelity
check. It even folds the B↔S coupling (the §7a tool-inadequacy worked example) into a named rung.

**Carve-at-the-joints (point 2)** makes the MANUAL §2.1 "composition, not rewrite" stricture concrete:
refactor so each model term is *one callable unit* (typed signature as SSOT, ADR-0012), so witness and
implementation *share the identical stage code* and differ only in glue — the port-set isomorphic to
`f`'s factorization. Passive ports (3) and the throughput-neutral-refactor proof (5) are sound
instrument discipline (the observer effect; the loop's own gate turned recursively on the phase that
*builds* its instrument). Points 6-7 honestly bound the method (ports measure costs, not couplings;
the residual across the whole ladder *is* the §7a tool-inadequate verdict) and surface a real dividend
(the same seams that assemble the witness adjudicate the model's binding stage against the gradient's
`a_i` prediction — a form finding read off the hardware, not inferred).

**The one caveat the bias understates — the build cost.** Carve-at-the-joints (point 2) entails
refactoring the *implementation itself* — the C++ serve/runner path and the Python `inference_server`
— so its units align with the model's terms, *proven throughput-neutral at every seam* (point 5).
That is substantially more than "lower the model": it is a production-code refactor with its own
parity / throughput-neutrality obligation (it is, in fact, the framework↔instance alignment the
backlog already flags, now made concrete on the *implementation* side). The review — biased toward
ambition by construction — presents this as the cleanest design, which it is; but it **raises** the
"build" arm of the consultation's build-vs-degraded decision (§11.1), it does not lower it. The honest
reading: the ladder + carve is the *right* way to build the witness *if it is built*, and the review
has made "build" both more powerful and more expensive; the decision of whether to pay for it is
unchanged in kind, and now far better-costed.

**Net:** a genuine, load-bearing contribution to the witness design. If/when the witness is built,
adopt the adequacy-witness framing and the descending ladder; and weigh the implementation-refactor
cost — not just a model-lowering — in the build-vs-degraded call.

---

## Note (2026-06-23) — `control_lab` is a partial witness, which lowers the build cost

Step 0's re-grounding (production = `control_lab`, the closed-loop issue-gate control lab —
`docs/notes/leaf-eval-loop/step-0-synthesis-and-path-forward.md`) softens the "build cost" caveat above.
`control_lab` is already an operational, instrumented, **clocked** end-to-end cycle running the *real* stages
(the `StageAServer` real-net forward + the `wire-ab-bench` producer over the pipelined wire), with **passive
Postgres egress** (the `lab_trial` / `metrics_series` blobs) — i.e. exactly the "passive ports parsed offline"
this review's point 3 prescribes, **already built**. So a substantial *down-payment on the witness exists*:
the ladder's lower rungs (real stages, real composition, real coordination) are partly in place, and `gap_B`
is partly *measured already* (the `AllAllow` → best-controller delta). This **lowers** the build arm of the
consultation's §11.1 decision that the review (biased toward ambition) had raised — though the per-stage
decomposition (a `tau_io` port; the carve-at-joints) and the *adequacy* framing remain to be built. The
descending-ladder design is unaffected; control_lab is its partial realization, not a substitute.
