# Focused derivation — is uncaught-ValueError server-thread-death reachable under the conforming C++ peer, and distinct from `_reject`?

All paths are under
`/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom/`. Line numbers are
the real source's; I derive forward from the code's operational semantics only.

## The question (verbatim)

Is the uncaught-ValueError server-thread-death terminal (ragged in_dim / bad
forward shape) reachable under the conforming C++ peer at any N, and is it
distinct from the `_reject` drop-one-reply path the fresh models modeled?

## Summary of finding

1. **DISTINCT — yes, mechanically.** `_reject` and the uncaught ValueError are
   two different control paths with two different observable effects:
   - `_reject` is *caught* (a `try/except` around `decode_request` inside the
     drain), prints, and `continue`s. The malformed request is silently dropped
     — **no reply** is ever sent for that one correlation id; the server keeps
     serving. Effect on peers: exactly one DEALER's `recv_batch` times out
     (RCVTIMEO) → that one worker's `set_error`. The other in-flight messages
     and all other threads are unaffected.
   - The uncaught ValueError is raised *inside* `run_microbatch`, which is
     called from `_serve_batch`, called from `serve_forever`, with **no
     enclosing `try/except` anywhere on that call chain**. It unwinds out of
     `serve_forever` and **kills the server thread**. Effect on peers: the
     ROUTER stops draining forever; **every** worker's next/pending
     `recv_batch` blocks until RCVTIMEO and fails → ALL peers go to `set_error`.
     This is the prior's EXCEPTIONAL_TERMINATION terminal.

   So the two are not the same node; modeling only `_reject` under-derives the
   thread-death and mislabels its blast radius (one peer vs. all peers).

2. **REACHABLE under a *conforming* C++ peer at any N — NO (for the
   peer-controlled raise sites).** Of the three `raise ValueError` sites in
   `run_microbatch`, the two the question names (ragged `in_dim`, line 53; bad
   forward shape, line 63) are **unreachable** when the C++ peer conforms,
   for *all* N, T, max_batch, D, and drain variant. The ragged-batch raise is
   blocked by a peer invariant: every message the peer emits carries the same
   `in_dim`. The bad-forward-shape raise is blocked by row-count algebra
   (`pad_to ≥ B`, forward preserves rows). The empty-batch raise (line 45) is
   blocked by the drain/serve guard.

   The thread-death terminal is therefore reachable **only** by breaching a
   RELY the peer is responsible for — i.e. by a **non-conforming** peer (or a
   net/feature-width *misconfiguration* that is a deployment fact, not a
   per-message peer act). It exists in the state space, but no conforming-peer
   execution reaches it. A faithful model keeps EXCEPTIONAL_TERMINATION as a
   reachable terminal **of the open system** (peer can misbehave), guarded by
   the named RELY, and marks it **unreachable in the closed system** (this
   conforming peer). It is *not* an interleaving the conforming-peer composition
   can produce.

## The three raise sites and where each is caught

`run_microbatch` (`chocofarm/az/inference_server.py:40-73`) raises `ValueError`
in three places:

- **L45** `if not requests: raise ValueError("empty batch …")`.
- **L51-53** `if m.ndim != 2 or m.shape[1] != in_dim: raise ValueError("… ragged batch")`
  with `in_dim = mats[0].shape[1]` (L48).
- **L62-63** `if out_arr.ndim != 2 or out_arr.shape[0] < B: raise ValueError("forward returned shape …")`.

Catch boundary, traced on the call chain:

- The drain wraps ONLY `decode_request` in `try/except`
  (`inference_server.py:179-183`): `except Exception as exc: self._reject(ident, exc); continue`.
  `_reject` (L188-190) merely `print`s. So a *decode* failure is the caught,
  drop-one-reply path — this is what the fresh models modeled.
- `run_microbatch` is called from `_serve_batch`
  (`inference_server.py:197-199`, production) and from the bench
  `StageAServer._serve_batch` (`cpp/stage_a/stage_a_server.py:65`). **Neither
  `_serve_batch` is wrapped in try/except**, and `serve_forever`
  (`inference_server.py:219-225`) calls `_serve_batch` with no guard. Hence any
  `ValueError` from L45/L53/L63 propagates out of `serve_forever` and
  terminates the thread that runs it (`stage_a_server.py:97`
  `threading.Thread(target=server.serve_forever, …)`; production deployment is
  analogous). **The death terminal is real and the call chain is unguarded.**

## Why L53 (ragged in_dim) is unreachable under the conforming peer, ∀N

`in_dim` at L48 is the in_dim of the *first* drained request; L51 raises if any
other request in the *same drained batch* has a different `shape[1]`. A drained
batch is whatever the single-threaded ROUTER `recv_multipart`-loop pulled in
one `_drain` (`inference_server.py:171-185`) — i.e. messages from possibly many
DEALER peers, interleaved in arbitrary order. So L53 fires iff **two messages in
one drain carry different `in_dim`**.

Trace the peer's `in_dim`:

- `run_episodes_wire_pipelined` (`cpp/src/runner_wire_batched.cpp`):
  `feat_dim = fb.dim()` (L275), a per-run constant; per worker
  `const wire::count_t in_dim = static_cast<wire::count_t>(feat_dim);` (L325),
  declared `const`, never reassigned.
- Every submit goes through `pool.submit_batch(gathered, gather, in_dim)`
  (L445), passing that same constant.
- `WireLeafPool::submit_batch` forwards it unchanged to
  `wire::encode_request(flat, B, in_dim)`
  (`cpp/include/chocofarm/wire_leaf_pool.hpp:76-80`); the header it writes is
  exactly that `in_dim` (`cpp/include/chocofarm/inference_wire.hpp:65-67`).
- The other (non-pipelined) runner is identical:
  `feat_dim = fb.dim()` (L48), `in_dim = feat_dim` (L99), `submit_batch(…, in_dim)`
  (L235).

Both runners are instantiated from one `Environment`/`FeatureBuilder`, so
`feat_dim` is a single value shared across **all T threads** and constant for the
whole run. Therefore **every message from every conforming peer carries the
identical `in_dim = feat_dim`**, for any N (N only sets `K = N·ceil(pool_batch/T)`
slots per thread, `runner_wire_batched.cpp:286`; it changes how many rows/slots a
message gathers, never the per-row width). Server-side, `decode_request`
(`inference_wire.py:42-61`) returns `(B_i, in_dim_i)` with `in_dim_i` read from
that message's header — uniformly `feat_dim`. Hence in `run_microbatch`
`mats[0].shape[1] = feat_dim` and every `m.shape[1] = feat_dim = in_dim`; the
L51 guard is always false. `m.ndim != 2` is also always false: `decode_request`
always `reshape(B, in_dim)` (2-D), and `np.atleast_2d` (L47) keeps it 2-D.
**L53 cannot fire under the conforming peer at any N.** ∎

(Counterpoint making it *reachable*: a non-conforming peer that writes a header
`in_dim ≠ feat_dim` on some message. The server-side `decode_request` does **not**
cross-check `in_dim` against any expected width — it accepts any `in_dim ≥ 1`
whose body length matches `B·in_dim·4` — so a single rogue message with a
different but self-consistent `in_dim`, landing in the same drain as a normal
one, trips L53 and kills the thread. That is the breach of the
`RELY(uniform in_dim)` below.)

## Why L63 (bad forward shape) is unreachable under the conforming forward, ∀N

`B = Xb.shape[0]` after `concatenate(mats)` (L55-56); each `B_i ≥ 1`
(`decode_request` rejects `B==0`, `inference_wire.py:49-50`), so `B ≥ 1`. The
pad (L58-59) only ever *adds* rows: `if pad_to is not None and pad_to > B`,
making `Xb` exactly `pad_to ≥ B` rows; otherwise `Xb` stays `B` rows. Both
`_serve_batch` callers pass a `pad_to` (production `pad_to=self._max_batch`,
`inference_server.py:198`; bench `pad_to ∈ {max_batch} ∪ {64,256,512}`,
`stage_a_server.py:61-64`), and `_bucket_for` returns ≥ real-rows by
construction (`stage_a_server.py:32-37`). So **input rows to the forward are
`max(B, pad_to) ≥ B`**.

`forward_fn` is `jit_forward_core` (`stage_a_server.py:80`,
`inference_server.py:145` default), whose `_fwd` (L29-32) returns
`concatenate([v(rows,1), logits(rows, n_actions)], axis=1)` — 2-D, row count =
input row count ≥ B. So `out_arr.ndim == 2` and `out_arr.shape[0] ≥ B`; the L62
guard is always false. **L63 cannot fire under the conforming forward at any N.**
∎

(Counterpoint: L63 *would* fire if `forward_fn` returned a 1-D array or fewer
rows than B — but that is a `forward_fn` defect, not a peer act, so it is outside
the transport-boundary RELY entirely. The width-mismatch case — `X.shape[1] ≠
W1.shape[0]` — does not reach L63: it raises *earlier*, inside the JAX matmul
`X @ params["W1"]` in `forward_core` (`chocofarm/az/forward.py:5`). That is *also*
an uncaught exception on the same unguarded chain, so it produces the *same*
thread-death terminal, but it is provoked by a **deployment** mismatch between the
peer's `feat_dim` and the net's `W1` width, an across-boundary agreement (a RELY),
not a per-message peer act. Under matched configuration — `stage_a_server.py:73-77`
builds the net's `in_dim` from the *same* `Environment` the C++ peer's `fb.dim()`
reads — `X.shape[1] == feat_dim == W1.shape[0]` and it does not fire.)

## Why L45 (empty batch) is unreachable

`_serve_batch` is only called when `drained` is truthy
(`serve_forever` L224 `if not self._stop and drained:`), and the bench
`_serve_batch` builds `rows` from a non-empty group. `run_microbatch`'s own L45
is a redundant guard; it never fires from the server path. (Also each request
has `B_i ≥ 1`, so a drained list is never of empty matrices.)

## Assume-guarantee statement (server side, the modeled party)

RELY (on the C++ DEALER peer over the wire; each checkable against the cited peer code):
- **R1 — uniform in_dim.** Every request frame carries the same `in_dim` value,
  equal to the net's input width. *Checkable:* `runner_wire_batched.cpp:325,445`
  (const `in_dim = feat_dim`, passed unchanged), `wire_leaf_pool.hpp:80`,
  `inference_wire.hpp:65-67`. Breaching R1 reaches L53 → thread death.
- **R2 — width agreement.** That common `in_dim` equals `W1.shape[0]`.
  *Checkable across the boundary:* both sides derive width from one
  `Environment` (`stage_a_server.py:73-77` net side; `fb.dim()` peer side).
  Breaching R2 raises inside `forward_core` (`forward.py:5`) → same thread death.
- **R3 — well-formed frames.** Header version = 2, body length = `B·in_dim·4`,
  finite floats, `B ≥ 1` (`inference_wire.py:44-61`). Breaching R3 hits the
  *caught* `_reject` path (drop-one-reply), **not** thread death.

GUARANTEE (server provides, given R1∧R2∧R3 hold for all messages in every drain):
- It performs exactly one forward per drained group and scatters one reply per
  request, echoing each request's identity+envelope (`inference_server.py:197-200`,
  bench `stage_a_server.py:69-70`), and **does not terminate**: no `ValueError`
  from L45/L53/L63 and no matmul width error can arise, so `serve_forever`
  loops indefinitely. The death terminal is gated entirely behind ¬R1 ∨ ¬R2
  (or a `forward_fn` defect, outside the wire RELY).

## Mapping the two terminals for the synthesizer

| terminal | trigger | caught? | reply emitted? | blast radius |
|---|---|---|---|---|
| `_reject` (drop-one-reply) | decode failure: bad header / length / non-finite / B=0 (¬R3) | yes (`inference_server.py:181-183`) | no — one corr-id silently unanswered | exactly the one peer that sent it → its single `recv_batch` RCVTIMEO |
| EXCEPTIONAL_TERMINATION (kill-the-server) | L53 ragged (¬R1), or matmul width (¬R2), or L63 / L45 | **no** — propagates past `serve_forever` | no — server stops forever | **all** peers → every pending/next `recv_batch` RCVTIMEO |

Under the conforming peer (R1∧R2∧R3), only the *no-error* GUARANTEE branch is
reachable; `_reject` is reachable only via ¬R3 and EXCEPTIONAL_TERMINATION only
via ¬R1∨¬R2. The prior is right to carry EXCEPTIONAL_TERMINATION as a *terminal
of the open system*; the fresh server models are right that the *closed*
conforming-peer system never reaches it — but they under-derive by collapsing
its distinct path/effect into `_reject`. Both nodes belong in the model, gated
by the RELYs above.

## Confidence

**High** for the mechanical distinctness (two call paths, one caught one not,
different reply/blast-radius effects — all on cited lines) and for L53/L63/L45
being unreachable under the conforming peer at any N (the in_dim-uniformity and
row-count arguments are closed-form over the cited code). The one residual is
that the cleanroom does not include `forward_core`'s caller's exception behavior
beyond `forward.py`/the JIT wrapper, but the matmul-raise reasoning needs only
that `X @ W1` raises on width mismatch, which is numpy/JAX semantics, not
project code. RCVTIMEO blast-radius rests on `wire_leaf_pool.hpp:41`
(`ZMQ_RCVTIMEO` set) — confirmed set, not OS default.
