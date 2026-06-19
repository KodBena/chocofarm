# Adversarial fidelity audit — SERVER side, TOO-CONSTRAINED lens

Subject: the two `side: server` models of the leaf-eval transport boundary
(`InferenceServer` ROUTER + `run_microbatch` gather/pad/forward/scatter + the `forward_core` MLP sink).

Lens: **too-constrained** — find executions the real code CAN produce that a model FORBIDS.

Author posture: I read all six required files end to end (ADR-0002), plus `wire_spec.py` and
`config.py`, and verified the installed libzmq (4.3.5) / pyzmq (27.1) ROUTER socket defaults
directly. Every finding maps to a specific code line.

---

## 0. What I verified against the source (not the models' citations)

- `inference_server.py` read 1–457. The ROUTER has **no `setsockopt` calls** — only `bind()`
  (:316), `Poller.register(POLLIN)` (:318), and `close(linger=0)` (:454). Confirmed by grep: the
  only `setsockopt` in `chocofarm/az/` are in `zmq_net_client.py` (the *client*, RCVTIMEO+LINGER),
  not the server. So every ROUTER option is at its libzmq-4.3.5 default.
- Live socket probe (a constant read, not a system run): a fresh `zmq.ROUTER` reports
  `SNDHWM=1000, RCVHWM=1000, SNDTIMEO=-1, RCVTIMEO=-1, LINGER=-1, ROUTER_MANDATORY` unset.
- `forward.py` 1–64: `forward_core` is a row-independent matmul chain; the service time is the
  wall-clock of `np.asarray(forward_fn(...))` at `inference_server.py:177`.
- `config.py:41-42`: `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, `OMP_NUM_THREADS=1` — the
  single-thread compute pin both models cite. `DEFAULT_INFERENCE_BATCH=64` (:47).
- The peer: `wire_leaf_pool.hpp` (LINGER=0, RCVTIMEO=timeout_ms, corr-id leading frame, loud abort
  on count/corr-id mismatch) and `runner_wire_batched.cpp` (strict barrier D=1 default :66-67,343;
  pipelined D=max_inflight_msgs :392,578-596). The RELYs both models state are grounded.
- The Z3 confirmation (`verify_server_too_constrained_check.py`, run once under `nice -n 19
  timeout 90`): the variable-B-from-timing self-clocking is SAT under positivity + serialization +
  reply-after-forward; the reply-before-forward ordering is UNSAT. So the central self-clocking
  axis is NOT over-constrained by either model.

The decisive socket fact for this audit: **`ZMQ_ROUTER` with `ROUTER_MANDATORY` off (the default
here) DROPS on a full per-peer high-water mark — it does NOT block.** libzmq-4.3.5 ROUTER manual:
a message routed to a peer whose individual HWM is reached "shall be dropped" unless
`ZMQ_ROUTER_MANDATORY` is set (then it returns `EAGAIN`). Blocking-at-HWM is DEALER/PUSH/PUB
behavior, never ROUTER-with-MANDATORY-off. This single fact resolves the two models' disagreement
about `send_multipart` and is the spine of finding S-2 / M2-2.

---

## 1. Findings on MODEL 1 (`model-server-drain.md`)

### M1-1 — server-thread death on a ragged-`in_dim` / bad-forward-shape batch is not a state (forbids-too-much, minor)

`run_microbatch` raises an **uncaught** `ValueError` at `inference_server.py:162` (mixed `in_dim`
across drained requests — "ragged batch") and again at `:179` (forward returns the wrong shape).
These are raised INSIDE `_serve_batch` (called at :385), which has no `try/except`, so they
propagate through `serve_forever`'s loop (:438-439) and **terminate the server thread**.

Model 1's T-REJECT transition (DRAINING→DRAINING, code_ref :356-360) covers ONLY the per-frame
`decode_request` `WireError`, which IS caught at :358. The ragged-batch and bad-forward-shape
ValueErrors are a DIFFERENT, uncaught path; Model 1's state machine has no terminal
"uncaught-exception / thread-dies" state reachable out of DRAINING/FORWARD. So Model 1 FORBIDS the
execution "two requests with different `in_dim` are drained into one batch → server crashes."

Reachability: under RELY R1 (one net, one feature dim by construction) this never fires with the
documented peer. But Model 1 already MODELS the equally-RELY-gated per-frame WireError path
(T-REJECT) and the all-malformed empty drain (DOF-8) — so excluding this same-class but
server-FATAL path is an inconsistency, not a principled scoping. Severity minor (RELY-gated), but
it is a real forbidden execution: the code's failure mode here is a hard crash, not a graceful
drop, and that asymmetry (crash vs reject) is itself behavior the model erases.

Correction: add a terminal CRASH state reachable from RELOAD_CHK/FORWARD on (a) `run_microbatch`
ragged-`in_dim` (:162), (b) `:179` shape guard, and (c) a `RedisParamsSource.poll()` /
`read_weights` raise at :284 — all uncaught, all killing `serve_forever`. Gate them on RELY
violations, exactly as DOF-8 is gated.

### M1-2 — DOF-7 silent-drop framing is faithful; the SNDHWM outcome is correctly NOT a block (no defect)

Model 1's DOF-7 says a `send_multipart` to a full/unroutable peer with `ROUTER_MANDATORY` off is
"silently dropped by libzmq." This is the CORRECT ROUTER-default behavior for BOTH the unknown-peer
and the known-peer-at-HWM case (the drop, never a block). Model 1 does NOT admit a phantom
indefinite-block state — which is the faithful choice (contrast Model 2, M2-2). No finding;
recorded because it is the one place the two models diverge and Model 1 is the faithful one.

### M1-3 — DOF-5 "no added per-request latency under load" is right but states the poll-phase claim slightly too strongly (none / verified)

Model 1 DOF-5: the 100ms poll is level-triggered on POLLIN so an arriving request returns the poll
immediately (`:341-342`). Verified: `poller.poll(timeout=100)` returns as soon as the ROUTER has a
readable frame; the 100ms only bounds the IDLE `_stop` re-check. A request that arrives during a
forward is deferred by serialization, not by the poll — Model 1 attributes that to DOF-3 (service
duration), correctly. Not over-constrained. (Confirmed admissible by the Z3 witness: the deferred
5th request.)

### M1-4 — timing left genuinely free (no defect)

Model 1 leaves the arrival timeline (C-arr-1..4), service duration (DOF-3), service regime/JIT
spike (DOF-4), batch composition (DOF-1), and soft-cap overrun (DOF-2) all free. It does NOT
collapse service time to a constant; C-svc-3's "near-constancy ACROSS B≤max_batch" is correctly
DERIVED from the single compiled padded shape (:171-172, jit one-executable :95-115) and each
forward still carries a free positive duration. This is the faithful reading of the pad discipline.
No over-constraint on the timing axis.

**Model 1 verdict: faithful, with one minor forbids-too-much hole (M1-1, the uncaught-crash class).**

---

## 2. Findings on MODEL 2 (`model-server-transport.md`)

### M2-1 — same uncaught-crash hole (forbids-too-much, minor)

Identical to M1-1. Model 2's DRAINING→DRAINING reject transition (code_ref :356-360) covers only
the caught `decode_request` `WireError`. The ragged-`in_dim` (:162) and bad-forward-shape (:179)
`ValueError`s, and a `RedisParamsSource` reload raise (:284), are uncaught and kill `serve_forever`
— Model 2 has no STOPPED-by-exception state (its only STOPPED is the clean `_stop` path). So Model
2 FORBIDS the same crash execution. Severity minor (RELY-gated), same correction as M1-1: add an
exceptional-termination terminal gated on the RELY violation.

### M2-2 — the SNDHWM "block" state (E7/DOF-7) forbids the real known-peer DROP (forbids-too-much via a wrong transition; major within its scope)

This is the sharpest too-constrained finding. Model 2 adds a transition SCATTER→SCATTER
"send to a KNOWN identity whose outbound pipe is at SNDHWM → send blocks indefinitely
(SNDTIMEO=-1)" (state-machine last transition; DOF-7; E7), and treats the silent drop as occurring
ONLY for an "unknown/disconnected identity."

The libzmq-4.3.5 ROUTER reality (verified above): with `ROUTER_MANDATORY` OFF, a send to a
**known** peer whose per-peer HWM is reached is **DROPPED, not blocked**. The ROUTER never blocks
at HWM under the default options; only `ROUTER_MANDATORY=1` turns a full pipe into `EAGAIN` (which,
with `SNDTIMEO=-1`, would then block). Since the server sets NO options (:315-318), the real
behavior for a known-but-full peer is the SAME silent drop as for an unknown peer.

Consequence for the too-constrained lens: Model 2 FORBIDS the genuine execution "a reply to a
known, alive-but-slow peer at HWM is silently dropped, then surfaces at that peer as an `RCVTIMEO`
loud abort (R6)." Model 2 routes that case into an indefinite-block wedge (E7) that the ROUTER
cannot produce. So Model 2 simultaneously (a) admits a phantom block-forever execution
[too-permissive] and (b) erases the real drop-to-known-full-peer execution [too-constrained]. Half
(b) is squarely in this lens. The drop is the actual liveness-relevant behavior — the server keeps
serving other peers and the dropped peer self-detects via RCVTIMEO — whereas Model 2's block wedges
the entire single-threaded server, an outcome the code does not produce.

expected_code_ref: `inference_server.py:387` + the absence of any `setsockopt`/`ROUTER_MANDATORY`
on the ROUTER (`:315-318`), against libzmq-4.3.5 ROUTER-drop-at-HWM semantics.

Correction: DOF-7 / E7 should model the SNDHWM outcome as a **drop** for BOTH known and unknown
peers (the ROUTER-default mute = discard), removing the indefinite-block transition entirely. The
peer's `RCVTIMEO` (R6) then surfaces every dropped reply as a loud abort — which is the system-level
safety Model 2's own R6 already states. (The block would only be reachable under a hypothetical
`ROUTER_MANDATORY=1`, which the code does not set; modeling it as the default is the defect.)

### M2-3 — Model 2 otherwise leaves the timing free (no defect)

Model 2's source_emission (e_r positive, post-reply, D-bounded, not gridded) and sink_service (s_f
positive, shape-invariant steady distribution + a one-time cold-compile tail S5, never collapsed)
are faithful. S4 (shape-invariance from the pad to a single executable, :171-172) is derived, and
s_f is deliberately kept bounded-nondeterministic rather than constant — the correct call. The
self-clocking buffering law is admissible (Z3 SAT). No over-constraint on the timing axis.

### M2-4 — B>max_batch handling is faithful (no defect; recorded for completeness)

Model 2's self-audit notes that a single request whose `B_i` pushes the total over `max_batch`
fires `pad_to > B` false (:171) → an UNPADDED larger-shape forward (a second executable). Verified
at :348 (cap tested on the pre-request total) + :362 (whole `X.shape[0]` added) + :171. Model 2
ADMITS this (not forbids) — faithful, and correctly NOT over-constrained.

**Model 2 verdict: mixed — faithful on the timing/self-clocking core, but M2-2 forbids a real
execution (the known-peer HWM drop) by asserting a phantom block, and M2-1 forbids the
uncaught-crash class.**

---

## 3. Trace admissibility (both models)

All representative executions in both models are genuine schedules the code admits, with ONE
exception:

- **Model 1**: E1–E6 are all admissible. E2's "4-then-1 from arrival timing vs serialized drain"
  is exactly the Z3-witnessed schedule (the 5th request arrives during forward 0, deferred to batch
  1). E4's overrun (70 rows, unpadded, first-sight JIT) follows :348/:362/:171. E5's
  reject-and-empty-drain follows :358/:438. All steps enabled.
- **Model 2**: E1–E6 are admissible by the same reasoning. **E7 is NOT admissible** — its step-1
  "send to a HWM-full KNOWN peer pipe → BLOCK" is a transition the default-option ROUTER cannot
  take (it drops). E7 is a phantom schedule (a too-permissive artifact whose flip side, M2-2,
  forbids the real drop). Every OTHER Model 2 trace is a real schedule.

So: Model 1 traces_admissible = true. Model 2 traces_admissible = false (solely E7).

---

## 4. Timing fidelity (both models)

Neither model collapsed source-emission or sink-service timing to a constant or an instant. Both
left the arrival timeline free (positive, reply-gated, D-bounded) and the service duration a bounded
positive nondeterministic value, with the pad-to-one-shape near-constancy correctly DERIVED (not
imposed) and the cold-compile / overrun regimes kept distinct. The Z3 confirmation shows the
variable-B-from-timing latitude is jointly admissible with the causal partial order. Timing is
faithful in both models. The fidelity gaps are at the failure-edge (uncaught crash; ROUTER HWM
semantics), not in the timing model.

---

## 5. Cross-model note

The two models agree on the entire self-clocking core (greedy drain, soft cap, one padded forward,
serialized service, scatter in drained order, free timing) and both got it right. They DISAGREE on
exactly one load-bearing point: the `send_multipart` SNDHWM outcome.

- **Model 1** (faithful here): "silently dropped" for full/unroutable — the correct ROUTER-default.
- **Model 2** (unfaithful here): "blocks indefinitely" for a known-full peer (E7/DOF-7), with the
  drop reserved for unknown peers only.

Model 1 is the faithful one: libzmq-4.3.5 ROUTER with `ROUTER_MANDATORY` off DROPS at HWM for known
AND unknown peers; it never blocks. Model 2's block-wedge is a phantom that, by its presence, also
erases the real known-peer drop — a single error that is both too-permissive (admits the block) and
too-constrained (forbids the drop). Both models share the minor uncaught-crash hole (ragged-`in_dim`
:162 / bad-shape :179 / reload raise :284 all kill `serve_forever`), which neither models as a
state.
