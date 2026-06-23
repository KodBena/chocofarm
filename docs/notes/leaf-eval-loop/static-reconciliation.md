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

## Step-2 result (2026-06-23) — the coordination hypothesis is REFUTED; the gap is a model FORM fault

I ran the instrumented forward (one `AllAllow` sweep, `--secs 8 --pool-batch 192 --no-postgres
--no-trajectory`, `CHOCO_EVENTLOG` → `step2-static/eventlog-allallow.log`; 2691 steady-state forwards) and
decomposed each forward into its `dt_us` (compute) and the inter-forward `gap` (idle). **The hypothesis this
note staked above — that the ~1500 µs is "coordination / idle / wire RTT the static model omits" — is
REFUTED by the measurement.**

| quantity | measured | model @ B=95.4 |
| --- | --- | --- |
| `dt_us` — forward COMPUTE | **2204 µs**, stable (p10–p90 2126–2294) | — |
| `gap` — IDLE between forwards | **583 µs** (21% of the wall) | — |
| `period` — WALL | 2787 µs | modeled cycle **501 µs** |
| `width` — padded width the forward computes on | **512, constant (all 2691)** | model uses **B=95.4** |
| `B` — useful rows/forward | 95.4 | 95.4 |

**The forward computes on the PADDED width (`max_batch=512`), not B.** `dt_us = 2204 ≈ 512·t_row` (2212 at
`t_row=4.32` — **`t_row` is faithful to 0.3%**), and it is *constant and independent of B* (B swings 64→128,
`dt_us` holds ~2200). The idle is only 21% of the wall. So:

- **The mechanism is a model FORM fault, not a coordination loss.** `cycle = T_disp + T_io + B·t_row` uses the
  *useful* width B (right only if the forward computes on B rows). Under **padmax** (the production-aligned
  regime the lab runs) the forward pads every batch to `max_batch=512` and computes on 512 rows regardless of
  B. The model conflates two quantities that are **equal in the bucket regime but diverge under padmax**: the
  **compute width** W (=512, drives the cost) and the **useful width** B (=95, drives the throughput).
- **Corrected serve form** (padmax-aware, W=512): `serve = 1e6·B / ((W·t_row + T_io)·L)` = **83 dps** — vs this
  note's 381, vs realized **99.5**. The form correction closes the gap from ~4× to ~0.84×.
- **The "+285 dps integration gap" was a MODELING ARTIFACT, not a static integration loss.** Once the form is
  padmax-correct, the model (~83) is *below* the realized (~99.5) — the implementation slightly *beats* the
  corrected serve bound here. **There is no large static integration loss at the drain-all/AllAllow operating
  point.** This **reverses this note's headline** ("the integration loss dominates") for the static point: the
  apparent 4× gap was the model grounded for the wrong regime (bucket, W=B) while the lab runs padmax (W=512).

**Secondary (provisional — inferred from the dps identity, not directly measured):** the residual (corrected
model 83 vs realized 99.5) implies `L=500` (LPD, the self-labeled design-pin "TAUTOLOGY") is ~16% high for the
lab's actual search (`n_sims=256` ⇒ effective leaves/decision ≈ 431). And the realized *exceeding* the
corrected serve bound means serve is not strictly binding at this point — a small residual within grounding
uncertainty (`L`, `t_row`), not a coordination loss. The primary form fault dwarfs it.

**Form-vs-fidelity verdict: FORM** (the compute-width structure), with `t_row` *faithful* (the bench did not
lie — it was applied to the wrong width). This is also a facet of **Finding 2 (the SSOT)**: under padmax the
operating point carries **two** widths (compute 512, useful 95) and the model/grounding record one — the
operating-point-not-well-defined disease, surfaced at the cost term. The padmax-aware serve form needs the
*compute width* as a first-class operating-point field (`max_batch` under padmax, the bucket under `bucket`).

**What it says about the project's question:** at the static drain-all point the throughput is well-explained
by a *regime-correct* model — the apparent 4× gap was the bucket-regime compute width. The large *unexplained*
losses live in the **convoy** regime (AllAllow → 11), which is `control_lab`'s control domain, out of our
static scope. Artifacts preserved under `~/w/vdc/chocobo/runs/control_lab/step2-static/` (eventlog, session
JSON, `parse_eventlog.py`). **Open confirm (cheap):** a `--e-policy bucket` A/B should show `width` tracking
the bucket (snapping with B) and `dt_us = T_disp + width·t_row` across multiple widths — proving `dt_us`
tracks the *compute width* (so `t_row` faithful, the fault purely the width variable).
