<!--
docs/notes/leaf-eval-loop/static-reconciliation.md
Purpose: the re-scoping of the leaf-eval impl->model loop to THE STATIC THROUGHPUT / SYSTEMS-INTEGRATION
  problem (ours), isolating the control-theoretic problem (control_lab's), plus the FIRST quantified static
  result: the model evaluated at the realized operating point, splitting the gap into its operating-point
  part vs the systems-integration part. Autonomous session the maintainer authorized; input for review.
ADR-0005 (point-in-time record); ADR-0006 header; claims-measured-vs-interpreted (every number tagged).
Public Domain (The Unlicense).
-->

# Static reconciliation — the re-scoping + the first result (2026-06-23)

## The re-scoping (maintainer-clarified)

This yak-shave exists to **isolate the static throughput / systems-integration problem (ours) from the
control-theoretic problem (`control_lab`'s)**. So the prior synthesis note's "**Finding 1 — the model does
not model control**" is **re-framed: that is the intentional SCOPE, not a deficiency.**

- **Ours:** the *static* throughput problem — at a *fixed* operating point, how much throughput does a
  well-integrated producer(N)-consumer(1) system achieve, and where is the integration loss?
- **Not ours:** the dynamic queue control (the per-forward issue-gate policies — BangBang, contextual
  bandit, REINFORCE, …). Modeling dynamically-driven queues is intractable in practice (REINFORCE in the
  stack); the model *shouldn't* try. control_lab is our **entry point** into a system of exactly our shape
  (it is the parent project that birthed ours) — we use it at a *frozen* operating point, control isolated.
- **`overcommit_sweep.py` is vestigial** — git arbitrates: first added 2026-06-19 (`dbf256f`), last touched
  06-20 01:24, all *before* `control_lab/` first appears (06-20 08:40, `b57476f`). It predates control_lab;
  it misled the grounding and is superseded.
- (The "if throughput is affine, BangBang is provably correct" thread is real but secondary — our static
  work stands regardless of the queuing flavor.)

## The static reconciliation: split the gap

The model's optimistic ceiling (≈456, or ≈429 on the Design-B serialized-cycle model) is at the **idealized**
operating point (`B_op=256`, full bucket). Realized operating points run lower `B`. To isolate *our* gap,
evaluate the model **at the realized `B`**, then split:

- **operating-point gap** = `model@idealized − model@realized-B` — the model's loss from the lower batch (not ours).
- **integration gap** = `model@realized-B − realized dps` — the **systems-integration loss (ours)**.

`model_cycletime` (Design-B): `cycle_us = T_disp + T_io + B·t_row`; `serve = 1e6·B/(cycle_us·L)`;
`producer = N_gen·R_gen`; `f = min(serve, producer)`. Grounded inputs `{N_gen:3, R_gen:152, T_disp:68.84,
T_io:20, t_row:4.32, L:500}` (verified from `initial_point()`).

| operating point | B | model f (dps) | bind | model cyc µs | realized dps | **integration gap** | realized cyc µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| idealized full bucket | 256.0 | 428.8 | serve | 1194 | — | — | — |
| AllAllow drain-all (pb=192,padmax,30s) | 95.7 | 381.3 | serve | 502 | 95.9 | **+285.4** | ~2000 |
| convoy s_min=1, AllAllow | 95.7 | 381.3 | serve | 502 | 11.0 | +370.3 | ~17400 |
| convoy s_min=1, contextual_bandit | 95.7 | 381.3 | serve | 502 | 57.0 | +324.3 | ~3358 |

(`realized cyc µs` is the per-forward wall the realized dps implies: `decisions/forward = B/L`, so
`forwards/s = dps·L/B`, `cyc_real = 1e6/forwards_s`.)

## The result: the integration loss DOMINATES, and it is largely control-independent

- **Operating-point gap is small** (~48 dps: 428.8 → 381.3 as `B` drops 256 → 95.7) — the serve stage is
  near-flat in `B` over this range (the "fast region"). So the low batch width is **not** the story.
- **The integration gap is the story** (~285 dps at the best operating point): at the realized `B=95.7` the
  model says ~381 dps; the implementation achieves ~96. The **realized serve forward takes ~4× the modeled
  cycle (~2000 vs ~500 µs)** — that ~1500 µs/forward is coordination / idle / wire RTT the static model omits.
- **Control does not close it.** Under the convoy regime the controller recovers AllAllow's collapse
  (11 → 57) but stays far below the model (a ~324 dps integration gap *with* control). So the systems-integration
  loss is the big lever, independent of the issue-gate control — confirming the scope: this is worth solving
  regardless of the queuing aspects.

## Honest accounting (claims-measured-vs-interpreted)

- **Measured (M):** the realized `(B, dps)` pairs — `(95.7, 95.9)` AllAllow flagship
  (`corpus-collect/lab_session-20260620-181752.json`); the convoy dps `(11.0, 57.0)`
  (`corpus-depthN/smin1`). Per investigation `acfa2641`.
- **Computed (M):** `model@B` — `model_cycletime.throughput_numpy` over the grounded inputs.
- **Inferred / provisional (I):** the realized cycle (~2000 µs) rests on `L=500` (a *design-pin*, self-labeled
  a tautology in `grounding.py`); `model@B` rests on the grounded inputs, of which `T_io=20` is an **UNMEASURED**
  prior and `T_disp`/`t_row` are fit read-offs. **The direction (a large integration gap, ~4× forward cycle)
  is robust; the exact magnitude is provisional** and is the thing Step 2 firms by instrumenting the forward.
- **Git (M):** `overcommit_sweep.py` predates `control_lab` (vestigial).

## The path forward (re-scoped to the static problem)

0. **The SSOT prerequisite stands** — one home for the operating point (synthesis note §"path forward (0)"),
   so the realized `B`/config the model is evaluated at is authoritative, not assembled ad-hoc.
1. **Instrument the realized serve forward (the loop's Step 2).** Decompose the ~2000 µs into the modeled
   compute (~500 µs: `T_disp + T_io + B·t_row`) vs the ~1500 µs coordination/idle. The instruments largely
   exist: `CHOCO_EVENTLOG` (`FWD`/`DRAIN` events on the server) + a producer-side wire-RTT timer; control_lab
   logs the forward stream already. This is a *static* measurement at a frozen operating point — control isolated.
2. **Form-vs-fidelity on the cycle (Step 3).** Is the ~1500 µs a real coordination cost the model omits
   (**FORM** — add a coordination/idle term to `cycle_us`), or are the grounded compute costs wrong at this
   operating point (**FIDELITY** — `T_disp`/`T_io`/`t_row` mis-grounded)? The discriminator is the realized
   per-stage timing vs the grounded values (consultation §7).
3. **Control stays isolated.** Study the static gap at a *fixed* operating point (`AllAllow` or a frozen gate);
   the issue-gate methods are control_lab's domain. The static integration loss is the lever this project owns.
