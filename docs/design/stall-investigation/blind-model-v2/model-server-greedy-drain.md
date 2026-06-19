# Server side of the leaf-eval transport boundary — a faithful parametric model

**Path:** `/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/out/model-server-greedy-drain.md`
**Purpose:** Forward-derived operational model of the **Python single-threaded ROUTER inference server**
as one party of the leaf-evaluation transport protocol, covering BOTH drain variants in scope: the
**production greedy drain** (`chocofarm/az/inference_server.py`) and the **bench bucketed/group drain**
(`cpp/stage_a/stage_a_server.py`). Parametric in N = `trees_per_thread`, T = `pool_threads`,
`max_batch`, D = `max_inflight_msgs`, and the drain variant. Public Domain (The Unlicense).

All `file:line` references are to the cleanroom under
`/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/cleanroom`; line numbers correspond to the real source.

---

## 0. Orientation: what the server is, mechanically

The server is **one Python thread** (`StageAServer` runs `serve_forever` on a daemon thread —
`stage_a_server.py:97`; the production server is normally the main thread). It owns **one ZeroMQ ROUTER
socket** (`inference_server.py:153–154`) and **one `zmq.Poller`** registered for `POLLIN`
(`:155–156`). Its entire life is the loop `serve_forever` (`:219–225`):

```
while not stop:
    drained = self._drain()                 # block until >=1 req, pull all queued up to max_batch rows
    if not stop and drained:
        self._serve_batch(drained)          # ONE forward over the whole batch, then scatter
```

There is **no concurrency inside the server**: drain, forward, and scatter are strictly serialized on the
single thread. This single fact is the spine of the entire model — it is why arrivals *coalesce* (§3, §5)
and why service time *shapes* the next batch (§7).

The compute is pinned single-threaded by `config.py:5–6`:
`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, `OMP_NUM_THREADS=1` (set via `setdefault`, so an
environment override is possible — a degree of freedom, DOF-7). `config.py` is imported at the top of
`stage_a_server.py:14` and transitively by `inference_server.py:25` inside `jit_forward_core`, so the pin
is in force before the first forward compiles.

The forward itself (`forward.py:3–18`) is a 2-layer ReLU MLP with an **optional** residual block
(`if "Wr1" in params`, `:9–13`) and an **optional** policy-logit head (`if "Wp" in params`, `:17`). It is
pure dense linear algebra: service time is a function of the **matrix shape fed to `forward_fn`**, and that
shape is the *padded* shape (§7), not the real row count.

---

## 1. Parameters and the derived quantities the server sees

The server's own config is small; most parameters are *peer* parameters that the server only sees through
**arrival timing and per-message row counts**. Stating the full set keeps the model parametric.

| Symbol | Meaning | Source | Server's relationship |
|---|---|---|---|
| `max_batch` | row cap per drain / pad target | `inference_server.py:149`, default 256 (`:145`); bench default 512 (`stage_a_server.py:89`) | **Directly owned.** Caps drained rows (`:171`) and is the production pad target (`:198`). |
| `forward_fn` | the JIT'd forward | `:145`, default `jit_forward_core` (`:22–34`) | Owned; service-time generator. |
| `_POLL_INTERVAL_MS` = 100 | blocking-poll quantum | `:142,165` | Owned; the only timeout the server sets. |
| `e_policy` ∈ {padmax, bucket} | bench pad shape | `stage_a_server.py:41,48,61–64` | Owned (bench only). |
| `wakeup` ∈ {group, leaf} | bench forwards-per-drain | `stage_a_server.py:41,49,57` | Owned (bench only). |
| T = `pool_threads` | # producer threads = # DEALER peers | `runner_wire_batched.cpp:283` | **Peer.** = number of independent arriving streams. |
| N = `trees_per_thread` | trees per producer thread | `runner_wire_batched.cpp:285` | **Peer.** Inflates per-stream concurrency: K = N·base. |
| base = `ceil(pool_batch/T)` | slots per tree | `runtime_config.hpp:12–15` | **Peer.** `fibers_per_thread()`. |
| K = N·base | slots per producer thread | `runner_wire_batched.cpp:286` | **Peer.** Max rows one thread can have parked at once. |
| D = `max_inflight_msgs` | per-thread in-flight message cap | `runner_wire_batched.cpp:287` | **Peer.** Caps each stream to ≤ D un-replied messages. |

**Aggregate offered load.** Across the whole system, at most **T·D messages** and at most **T·K rows** can
be in flight (queued at the ROUTER or being served) at any instant: each of T threads holds ≤ D unanswered
messages (`runner_wire_batched.cpp:456,474`) and ≤ K parked slots (`:324`). The server cannot exceed these;
they are hard upper bounds on what any single drain can ever pull. This is the load envelope every part of
the model lives inside, and **every quantity scales in N through K = N·base** (and through D, T).

---

## 2. The wire framing the server must speak (grounded in both peers)

DEALER→ROUTER. The producer sends, per message, two ZeroMQ frames
(`wire_leaf_pool.hpp:86,89`):

1. `corr` — an 8-byte `uint64` correlation id (`submit_batch`, `:84–88`), `SNDMORE`;
2. `payload` — `encode_request` bytes: 9-byte header (version `u8`=2, B `u32`, in_dim `u32` —
   `wire_spec.hpp:8,18`; `inference_wire.hpp:65–67`) followed by `B·in_dim` little-endian f32
   (`inference_wire.hpp:68`).

The ROUTER prepends the peer **identity** frame. So `recv_multipart` (`inference_server.py:173`) yields
`[ident, corr, payload]`, and the server splits it as `ident = frames[0]`, `envelope = frames[1:-1]`
(= `[corr]`), `payload = frames[-1]` (`:176–178`). It **decodes only the payload** (`decode_request`,
`:180`) and treats `corr` as an opaque envelope frame it echoes back verbatim (`:200` sends
`[ident, *envelope, resp]` = `[ident, corr, resp]`). The server never parses or matches the correlation
id — **correlation-id matching is entirely the producer's job** (`wire_leaf_pool.hpp:115` `inflight_.find(corr)`);
the server's guarantee is only *frame-order preservation of the envelope*. The ROUTER strips `ident` on
send, so the DEALER receives `[corr, resp]`, matching the producer's `recv_corr_payload`
(`wire_leaf_pool.hpp:157–163`: `frames.front()` = corr, `frames.back()` = payload).

`decode_request` (`inference_wire.py:42–61`) validates: version byte == 2 (`:47`), B≠0 (`:49`), in_dim≠0
(`:51`), exact payload length `B·in_dim·4` (`:54–56`), and **all-finite** (`:59`). The Python `wire_spec`
module is not in the cleanroom, but the C++ `wire_spec.hpp` fixes the format the Python `struct` formats
must mirror (version `u8`, two `u32` counts), and the C++ encoder (`inference_wire.hpp:65–68`) confirms
the exact byte layout the Python decoder consumes. (RELY R1.)

---

## 3. ZeroMQ socket options — exactly what is set, and the blocking consequences

These determine every blocking/queueing behavior; defaults matter as much as explicit sets.

**Server ROUTER** (`inference_server.py`):
- `zmq.ROUTER` (`:153`), `bind` (`:154`). The server is the **bind** side; producers `connect`
  (`wire_leaf_pool.hpp:42`).
- **`LINGER` is set to 0 only at `close()`** (`:236` `self._sock.close(linger=0)`). During normal operation
  LINGER is the ZeroMQ default (−1, infinite) — irrelevant while running; on shutdown, 0 means pending
  outbound replies are dropped immediately.
- **`RCVTIMEO`, `SNDTIMEO`, `SNDHWM`, `RCVHWM`, `ROUTER_MANDATORY`, `ROUTER_NOTSOCKET` — NONE set.**
  They take ZeroMQ defaults: RCVTIMEO=−1, SNDTIMEO=−1, SNDHWM=1000, RCVHWM=1000, ROUTER_MANDATORY=0.
- The server never does a *blocking* recv: every `recv_multipart` uses `flags=zmq.NOBLOCK` (`:173`).
  Blocking is done **only** through the poller (`:165` `self._poller.poll(timeout=100ms)`).
- Sends use plain `send_multipart` (`:200`) — blocking-capable. With ROUTER_MANDATORY=0, a send to an
  **unroutable** identity is **silently dropped** (no error); with the default SNDHWM=1000, a send to a
  *known-but-full* peer pipe would block. In practice the reply is small and the DEALER's RCVHWM is the
  default 1000, so blocking on send is not normally reachable, but the model admits it (DOF-6).

**Producer DEALER** (`wire_leaf_pool.hpp`): `ZMQ_DEALER` (`:35`), `ZMQ_LINGER=0` (`:39–40`),
`ZMQ_RCVTIMEO=timeout_ms` (`:41`, default 15000 — `runner_wire_batched.hpp:22`). SNDHWM/RCVHWM default 1000.
`connect` (`:42`). This grounds RELY R3 (the producer can time out a recv, but never blocks a send beyond
its SNDHWM).

**Consequence for coalescing (the load-bearing inference).** The ROUTER has a per-peer incoming queue
(bounded by RCVHWM=1000 messages/peer). While the server thread is inside a forward (§7), it is **not**
calling `recv`. Producer messages that arrive during that window accumulate in the ROUTER's incoming
queues. The *next* `_drain` (`:160`) then pulls **all of them** in one non-blocking loop (`:171–186`). This
is why the realized batch of cycle *k+1* grows with the number of streams that emitted during cycle *k*'s
service — confirmed admissible by the Z3 check (§9, `out/check_server_greedy_drain.py`).

---

## 4. Operational state machine (production greedy drain)

States are the server thread's control location across one `serve_forever` iteration.

### States

| State | Meaning |
|---|---|
| `S_POLL` | Inside `_drain`'s blocking-poll loop (`:163–166`): waiting up to 100 ms for ≥1 POLLIN; loops while idle. |
| `S_PULL` | Inside `_drain`'s non-blocking pull loop (`:171–186`): repeatedly `recv_multipart(NOBLOCK)` until `Again` or `total_rows ≥ max_batch`, decoding each and rejecting malformed ones. |
| `S_RELOAD` | `_serve_batch` start (`:194–195`): `params_source.poll()` for a weight update, else `current()`. (Production only; bench uses `current()` unconditionally — `stage_a_server.py:56`.) |
| `S_FORWARD` | Inside `run_microbatch` (`:40–73`): concat, pad to `max_batch` (`:58–59`), `forward_fn`, and the **`np.asarray(...)` that blocks until XLA finishes** (`:61`) — this is the **service-time** state. |
| `S_SCATTER` | The `for … send_multipart` reply loop (`:197–200`) echoing each `[ident, corr, resp]`. |
| `S_STOP` | `_stop` observed true (`:163,167,222,224`); drain returns `[]`, loop exits. |

### Transitions

| # | from→to | guard | action | code_ref | free? |
|---|---|---|---|---|---|
| T1 | `S_POLL`→`S_POLL` | poll() returns falsy (no POLLIN within 100 ms) **and** not stop | re-enter blocking poll | `:163–166` | **free** (depends on peer arrival timing, RELY R2) |
| T2 | `S_POLL`→`S_PULL` | poll() truthy (≥1 queued) **and** not stop | begin non-blocking pull, `total_rows=0` | `:165–169` | **free** (which/when arrivals are queued) |
| T3 | `S_POLL`→`S_STOP` | `_stop` true | return `[]` | `:167–168` | no (external stop) |
| T4 | `S_PULL`→`S_PULL` | `recv` succeeds, decode ok, `total_rows < max_batch` after add | append `(ident,[corr],X)`, `total_rows += X.shape[0]` | `:171–185` | **free** (how many more are already queued) |
| T5 | `S_PULL`→`S_PULL` | `recv` succeeds, decode **raises** | `_reject` (print), `continue` — **not appended, row not counted** | `:179–183,188–190` | **free** (whether a malformed/partial frame is present) |
| T6 | `S_PULL`→`S_RELOAD` | `recv` raises `zmq.Again` (queue momentarily empty) **or** `total_rows ≥ max_batch` | break out of pull with `drained` (≥1) | `:171,174–175` | **free** (queue-drain boundary) |
| T7 | `S_RELOAD`→`S_FORWARD` | always | pick params (reloaded or current) | `:194–196` | no |
| T8 | `S_FORWARD`→`S_SCATTER` | `asarray` returns (XLA done); shape check passes | build per-ident responses | `:61–73` | **free** (service-time duration; §7) |
| T9 | `S_FORWARD`→`S_FORWARD`(abort) | `out_arr.ndim≠2` or `rows<B` | `raise ValueError` → thread dies | `:62–63` | no (peer/contract violation; ADR-0002 fail-loud) |
| T10 | `S_SCATTER`→`S_POLL` | reply loop exhausted | next `serve_forever` iteration | `:200,222–225` | no |
| T11 | `S_PULL`→`S_RELOAD` | `total_rows≥max_batch` reached mid-pull | break with a **possibly-truncated** drain (queue still has more) | `:171` | **free** (overflow split, §6) |

**Note on T6/T11 and the cap.** The cap is checked at the **top** of the loop (`while total_rows < max_batch`,
`:171`), and a single message can carry up to K rows. So the *last* admitted message may push `total_rows`
**above** `max_batch` (e.g. `max_batch=256`, drained 250, next message carries K=64 → `total_rows=314`).
The drained row count is bounded above by `max_batch + (K−1)`, not `max_batch`. This matters for padding
(§6) and is a genuine over-fill the code permits.

---

## 5. State machine (bench variant: `stage_a_server.py`)

The bench subclass overrides **only** `_serve_batch` (`stage_a_server.py:54–70`); `_drain`,
`serve_forever`, sockets, and poll are inherited unchanged. So states `S_POLL`/`S_PULL`/`S_STOP` and
transitions T1–T6, T10–T11 are **identical**. What changes is the forward/scatter sub-machine, governed by
two knobs.

- **`wakeup`** (`:57`): `groups = [drained]` (one group = whole drain) if `group`, else `[[d] for d in
  drained]` (one group **per request**). With `leaf`, the server does **one forward per queued request**:
  the inner loop (`:58–70`) runs `len(drained)` forwards, **serially**, each its own service time. This
  *destroys* coalescing within a drain even though the drain itself still pulled them together.
- **`e_policy`** (`:61–64`): `pad_to = max_batch` (padmax) or `pad_to = _bucket_for(real)` (bucket), where
  `_bucket_for` (`:32–37`) snaps `real` up to the smallest of {64,256,512} that is ≥ real, capping at 512.
  This sets the **compiled shape** and therefore the service-time band (§7).

### Bench states (replace `S_RELOAD`/`S_FORWARD`/`S_SCATTER`)

| State | Meaning |
|---|---|
| `B_PLAN` | `_serve_batch` start: `params=current()` (`:56`), split `drained` into `groups` per `wakeup` (`:57`). |
| `B_GROUP_FWD` | For one group: compute `real = Σ X.shape[0]` (`:60`), pick `pad_to` (`:61–64`), `run_microbatch` → ONE forward (`:65`), bump stats (`:66–68`). Service-time state for this group. |
| `B_GROUP_SCAT` | Scatter that group's replies (`:69–70`). |
| `B_NEXT` | If more groups, back to `B_GROUP_FWD`; else `S_POLL`. |

### Bench transitions (in addition to inherited T1–T6, T10–T11)

| # | from→to | guard | action | code_ref | free? |
|---|---|---|---|---|---|
| BT1 | `S_RELOAD`-slot→`B_PLAN` | always | `current()`, build `groups` | `:56–57` | no |
| BT2 | `B_PLAN`→`B_GROUP_FWD` | ≥1 group | enter first group | `:58` | no |
| BT3 | `B_GROUP_FWD`→`B_GROUP_SCAT` | `asarray` returns | per-ident responses for group | `:65,69` | **free** (service time; §7) |
| BT4 | `B_GROUP_SCAT`→`B_GROUP_FWD` | more groups remain | next group, fresh forward | `:58,69` | no |
| BT5 | `B_GROUP_SCAT`→`S_POLL` | groups exhausted | next iteration | `:58→69→serve_forever` | no |
| BT6 | `B_GROUP_FWD`→`B_GROUP_FWD`(bucket cap) | `real > 512` and bucket policy | `pad_to = 512 < real` → **pad smaller than real** → run_microbatch pads nothing (`:58` guard `pad_to > B` false) → forward at real shape > 512 | `:37,58,64` | **free** (overflow past largest bucket) |

**Bench-specific fidelity points.**
- With `wakeup=leaf`, the number of **forwards per drain equals the number of queued requests** — so a drain
  that coalesced G producer messages produces G serial forwards, each padded independently. As N grows
  (more ready slots per thread → more/larger messages, §8), `leaf` multiplies forward count by the message
  multiplicity, the opposite of the production coalescing regime.
- `_bucket_for` (`:32–37`) returns 512 for any `real > 256`, including `real > 512`; in that last case
  `pad_to=512 < real`, so `run_microbatch`'s pad guard (`inference_server.py:58` `pad_to > B`) is **false**
  and **no padding happens** — the forward runs at the true `real` shape (BT6). The bench never *truncates*
  rows; only the production drain's `max_batch` cap (T11) can split a drain.
- Bench `_serve_batch` does **not** call `params_source.poll()` (`:56` uses `current()` only) — so the
  S_RELOAD weight-reload latitude (DOF-5) is **absent** in the bench; weights are frozen
  (`StaticParamsSource`, `stage_a_server.py:78`, whose `poll()` returns None anyway — `:110–111`).

---

## 6. The pad/cap arithmetic (production), exactly

Let the drain admit messages with real row counts `r_1,…,r_g` (each `r_j = X_j.shape[0] ≥ 1`,
`decode_request` guarantees `B≥1`). Then:

- **Drained real rows** `R = Σ r_j`. The pull loop stops when `R ≥ max_batch` *checked at loop top*
  (`:171`), so `max_batch ≤ R ≤ max_batch + (max(r_j)−1) ≤ max_batch + (K−1)` when it stops on the cap, or
  `R < max_batch` when it stops on `Again` (queue emptied).
- **Concatenated matrix** `Xb`, `B = R` rows (`run_microbatch:55–56`).
- **Padding** (`:58–59`): if `max_batch > B`, append `max_batch − B` zero rows → fed shape =
  `max_batch × in_dim`. If `B ≥ max_batch` (the over-fill case), **no padding**, fed shape = `B × in_dim`
  which may **exceed** `max_batch`. So the production fed-shape is `max(max_batch, B) × in_dim`.
- **Forward output** must have `≥ B` rows (`:62–63`), else fail-loud (ADR-0002).
- **Scatter** (`:67–73`): slice `out_arr[off:off+r_j]` per ident in drain order, `encode_response`
  (`inference_wire.py:63–86`), echo `[ident, corr, resp]`. Padding rows are discarded (never indexed past
  `off=Σr_j`).

**Production E-policy is fixed pad-to-`max_batch`** (`:198`). Thus, except in the over-fill case, **every
production forward runs at the identical `max_batch × in_dim` shape** regardless of real B. This is the key
service-time fact for the production server (§7): one compiled shape, one service-time band, independent of
how many requests coalesced.

---

## 7. Timing model

### 7.1 Source emission (what the server sees arriving) — nondeterministic, derived from the peer

Per RELY R2/R4, a producer thread emits a message when ≥1 of its K slots is *ready* (parked at a leaf,
`is_ready`, `runner_wire_batched.cpp:427–430`) and it has in-flight headroom (`inflight_msgs < D`,
`:456,474`). The **interval between a slot becoming ready and the next** is set by the search's internal
work (`TreeState`/fiber advance — `fiber_tree.hpp`, `fiber_leaf.hpp`), which the code **does not fix**.
Therefore arrivals are modeled as a **bounded nondeterministic point process per stream**:

- Each stream *i* (∈ T threads) emits messages at times `a_{i,1} < a_{i,2} < …`, intervals **positive and
  otherwise free** (no fixed rate, no upper bound except the system makes progress).
- Each message carries `r ∈ [1, K]` rows: the producer's `issue_one` (`:434–452`) gathers **all** currently
  ready slots into one message (`:437–443`), so `r` is the count of slots that became ready since the last
  issue — itself nondeterministic, but **bounded by K = N·base**.
- A stream has at most **D** messages outstanding at once and at most **K** slots parked total
  (RELY R3); after the cap it **cannot emit** until a reply frees a slot (`recv_batch`,
  `:458–460`, then `issue_one` again `:474`) — a **back-pressure coupling** from the server's own replies.

**N-dependence of emission.** As N grows: K = N·base grows linearly, so (a) the max rows per message grows
linearly, (b) each thread can have more slots ready simultaneously, fattening messages and raising the
arrival burstiness during any fixed window, and (c) the D-cap bites *later* relative to slot supply (more
ready slots per outstanding message), so a thread sustains its emission longer before blocking on replies.
Net: **larger and burstier arrivals as N↑**, which (via §3 coalescing) **inflates the realized drain batch
size as N↑**.

### 7.2 Sink service time — nondeterministic band, function of the compiled shape (NOT an instant)

The forward's service time is the wall time of `np.asarray(forward_fn(params, Xb, …))`
(`inference_server.py:61`): `forward_fn` returns a lazy JAX array; `np.asarray` **blocks until XLA finishes
computing it** (the device-to-host materialization). This is the service time and it is **not collapsed to
an instant**. Model it as a positive nondeterministic duration `S(shape)` drawn from a band that depends on
the **compiled input shape**:

- The forward is dense matmuls (`forward.py:5–17`); cost grows monotonically with row count. With XLA
  single-thread pinned (`config.py:5–6`), it is roughly affine in rows for a fixed `in_dim`/hidden, but the
  model only commits to **monotone non-decreasing in the padded row count and positive**.
- **`jax.jit` compiles once per distinct input shape** (`jit_forward_core:24–34` caches the jitted fn; XLA
  caches the compiled executable per shape). The **first** forward at a new shape pays a large
  **compile cost** (one-time, can be seconds); subsequent forwards at that shape pay only execution.
  `warmup` (`inference_server.py:202–217`; called for buckets ∪ {max_batch} at `stage_a_server.py:82`)
  pre-pays the compile for the shapes it expects, **removing the first-call spike from steady state** — but
  any shape never warmed up (e.g. a production over-fill `B > max_batch`, §6; or a bench bucket transition)
  pays a fresh compile on first occurrence. So `S(shape)` has **two bands per shape**: a one-time
  *compile+exec* band and a steady *exec-only* band.
- **Production (pad-to-`max_batch`)**: essentially **one steady shape** → one steady-exec service band for
  all non-over-fill drains. Service time is **independent of real coalesced batch size** — a 1-row drain
  and a 256-row drain (both padded to 256) cost the **same** steady `S`. (This is the production server's
  defining timing property and the reason coalescing is "free throughput": more rows per forward at
  constant per-forward cost.)
- **Bench `bucket`**: up to **three** steady shapes {64,256,512} (plus over-512 unpadded, BT6) → a
  **piecewise** service band that steps up at bucket boundaries; `real ≤ 64` costs the 64-band even at 1
  row, etc.
- **Bench `padmax`**: one shape = `max_batch` (default 512), like production but at 512.
- **Bench `leaf`**: G serial forwards per drain, each its own `S(bucket_for(r_j))` (group) — the per-drain
  service time is the **sum** over queued requests, not one forward.

### 7.3 Causal constraints (the only constraints on the free timing)

1. **Service positivity:** every `S > 0` (a forward takes time; `:61` blocks).
2. **No reply before its forward:** a reply frame for drain *k* is sent (`:200`) strictly after
   `S_FORWARD`(k) completes (`:61`). A producer cannot observe a prediction before the forward that produced
   it.
3. **Single-thread serialization:** drain *k+1* cannot begin (`:223`) until scatter *k* completes
   (`:200,224`). Forwards never overlap. This is the coalescing engine.
4. **No reply-dependent request before its reply:** a producer slot that is `submitted` cannot re-emit until
   its reply arrives and clears `submitted` (`runner_wire_batched.cpp:447,466`), and the D-cap couples emission to
   replies. So the server's *output* timing feeds back into its *input* timing (a closed loop, not an open
   arrival process).
5. **Poll quantum:** when idle, the server wakes at most every 100 ms (`:165`) to re-check `_stop`; an
   arrival during a poll wakes it immediately (POLLIN), so the 100 ms is a *liveness/stop* bound, not added
   latency to a waiting request.

**Nothing was collapsed to a constant.** Both the source inter-arrival interval and the sink service time
are kept as bounded nondeterministic durations; the only equalities asserted are causal (positivity,
ordering, serialization) and the structural fact that the **production fed-shape is constant** (which makes
the *service band* shape-constant, not the *duration* constant).

---

## 8. How realized batch size and throughput depend on timing — parametric in N, T

Let steady service time be `S` (production: one band). Consider the closed loop. Define the **coalescing
window** as one service interval `S`. During `S`, every stream that becomes ready emits into the ROUTER
queue (up to its D-cap). The next drain pulls all of them. So:

- **Realized batch (production)** `B_{k+1} ≈ Σ_i (rows stream i emitted during cycle k's S)`, capped at
  `max_batch (+K−1 over-fill)`. With T streams each able to emit up to K rows but capped by D messages and
  by the cap, the steady realized batch sits between **1** (one lone arrival, light load) and
  **`min(max_batch+K−1, T·K)`** (saturated).
- **Two regimes:**
  - *Service-bound (heavy load / large N,T):* arrivals during `S` exceed `max_batch` → every drain fills to
    `max_batch`, throughput = `max_batch / S` rows/s, and **excess load queues** (latency ↑, bounded by
    T·D total messages). Batch size pinned at the cap.
  - *Arrival-bound (light load / small N,T):* fewer than `max_batch` rows arrive per `S` → drain pulls
    whatever is queued (often 1), throughput = (rows offered)/s < `max_batch/S`, padding fraction high.
- **N-dependence (the crux).** Increasing N scales K = N·base linearly. This (i) raises the per-stream
  row supply per window, (ii) fattens each message (more ready slots gathered per `issue_one`), and (iii)
  raises the saturation ceiling T·K. So **as N↑, the system slides from arrival-bound toward service-bound**:
  realized batches grow toward `max_batch`, **padding fraction falls**, and throughput in *useful rows/s*
  rises toward `max_batch/S` — the **whole point of N>1 overcommit** (it manufactures concurrent ready
  slots to coalesce). Past the point where T·K ≳ `max_batch`, further N buys only deeper queueing (latency),
  not batch size, because the cap binds. In the bench `bucket` policy the same slide *also* climbs the
  bucket ladder 64→256→512, stepping service time up at each threshold while cutting pad fraction.
- **T-dependence.** T is the number of independent streams; more streams means more arrivals per window
  *and* more identities to scatter to, but the server cost is per-row (one forward) + per-message (scatter
  loop), so T mainly raises offered load toward saturation, like N but without fattening a single message.

---

## 9. Assume–Guarantee contract (server is one party)

### RELY (assumptions about the producer peer, each checkable against `wire_leaf_pool.hpp` /
`runner_wire_batched.cpp`)

- **R1 (codec):** every request is `[corr(8B), payload]` with payload = 9-byte header (ver=2, B≥1,
  in_dim≥1) + `B·in_dim` finite f32; identical `in_dim` is **not** guaranteed across messages — the server
  rejects ragged batches loudly (`run_microbatch:51–53`). *Check:* `wire_leaf_pool.hpp:86,89`,
  `inference_wire.hpp:65–68`, `wire_spec.hpp:8`.
- **R2 (arrival timing):** inter-emission intervals are positive and otherwise unconstrained (set by search
  progress, `runner_wire_batched.cpp:170–179,389–397`); no rate guarantee. *Check:* fiber advance is the
  only thing that makes a slot ready (`fiber_leaf.hpp:24–28`).
- **R3 (flow caps):** each stream keeps ≤ D messages in flight and ≤ K slots parked; it **will not emit**
  once at the D-cap until a reply frees headroom (`runner_wire_batched.cpp:456,474`). Producer never blocks a
  send beyond its SNDHWM=1000 (LINGER=0, RCVTIMEO set — `wire_leaf_pool.hpp:39–41`).
- **R4 (corr matching is the peer's job):** the producer owns the `corr→slots` map and validates reply
  cardinality and identity itself (`wire_leaf_pool.hpp:115–124`). The server need not (and does not) match.
- **R5 (reply consumption):** the producer eventually `recv_batch`es every reply (`:458`), so the ROUTER's
  outbound pipe to that peer does not stay full (server send won't wedge under SNDHWM long-term).
- **R6 (one-to-one drain↔request):** the producer expects the reply's prediction count to equal the request's
  row count (`wire_leaf_pool.hpp:121–124`). The server's per-ident slice `out_arr[off:off+n]`
  (`run_microbatch:68–72`) preserves exactly this — GUARANTEE G2.

### GUARANTEE (what the server provides)

- **G1 (envelope fidelity):** the server echoes the request's `corr` frame **verbatim** and routes the reply
  to the originating `ident` (`inference_server.py:200`). It never reorders frames within a message.
- **G2 (per-request cardinality & order):** each ident receives exactly its `r_j` predictions, sliced in
  request order from the concatenated output (`run_microbatch:67–73`); response framing per
  `inference_wire.py:63–86`.
- **G3 (one forward per drained group):** production = exactly one forward per non-empty drain
  (`:198,219–225`); bench `group` = one per drain, bench `leaf` = one per request (`stage_a_server.py:57`).
  No request is served twice; no request in a non-empty drain is dropped (except malformed → G5).
- **G4 (no spurious empty forward):** `_drain` returns only `drained` with ≥1 request before
  `_serve_batch` runs (`:224`), and `run_microbatch` fail-loud-rejects an empty batch (`:44–45`).
- **G5 (malformed isolation):** a request that fails `decode_request` is rejected in isolation (`:182,188`)
  and excluded from the batch — it does **not** poison sibling requests in the same drain. *But* a malformed
  request gets **no reply at all** (only a server-side `print`, `:190`), so the producer's `corr` for it
  stays in `inflight_` forever → that slot will eventually hit the DEALER's RCVTIMEO (`wire_leaf_pool.hpp:41`)
  and the producer fails loud. (This is a real liveness coupling: G5 is "don't poison siblings", **not**
  "always reply".)
- **G6 (fail-loud on contract violation):** ragged in_dim (`:51–53`), too-few output rows (`:62–63`), empty
  batch (`:44–45`) all `raise` and kill the server thread (ADR-0002). The server does **not** silently
  degrade.

---

## 10. Degrees of freedom (each with code_ref, behaviors admitted, N-dependence)

- **DOF-1 — Drain boundary (how many messages one drain pulls).** `_drain` pulls until `Again` or cap
  (`:171,174`). *Admits:* any drain size from 1 message up to "all currently queued (≤ T·D), capped at
  `max_batch` rows (+K−1 over-fill)". *N-dependence:* the **upper** reachable drain size scales with K=N·base
  (fatter messages) and the saturation ceiling T·K, so as N↑ drains skew larger and pad fraction falls.

- **DOF-2 — Realized batch size B given a drain.** `B = Σ r_j` (`run_microbatch:55`). *Admits:* `1 ≤ B ≤
  max_batch+(K−1)`. *N-dependence:* both the per-message max (K) and the achievable sum grow with N; the
  steady B rises toward `max_batch` as N↑ (service-bound slide, §8).

- **DOF-3 — Service-time duration S.** `np.asarray(forward_fn(...))` blocks (`:61`). *Admits:* any positive
  duration in the band for the compiled shape; a one-time compile spike on a never-warmed shape. *N-dependence:*
  in **production**, S is **independent of N** (constant fed shape `max_batch`); N changes *how often* the
  cap is hit, not S. In **bench bucket**, larger N climbs to higher buckets → S steps up with N.

- **DOF-4 — Over-fill past `max_batch`.** Cap checked at loop top (`:171`); last message may carry K rows.
  *Admits:* `B ∈ (max_batch, max_batch+K−1]`, an **unwarmed, larger-than-`max_batch`** forward shape → a
  fresh compile spike + no padding (`:58` guard false). *N-dependence:* the over-fill magnitude is `K−1 =
  N·base − 1`, so the over-fill window **grows linearly with N** — larger N makes bigger, more frequent
  unwarmed over-fills (a real cost that N introduces).

- **DOF-5 — Weight reload mid-stream (production only).** `params_source.poll()` (`:194`) may return new
  weights between any two drains. *Admits:* consecutive forwards using *different* params; a `RedisParamsSource`
  re-read (`:129–138`). *N-dependence:* none directly; reload cadence is external. **Absent in the bench**
  (`stage_a_server.py:56` uses `current()` only). 

- **DOF-6 — Send blocking / silent drop on scatter.** `send_multipart` (`:200`) with default SNDHWM=1000,
  ROUTER_MANDATORY=0. *Admits:* (a) block if a peer's outbound pipe is full (R5 makes this transient); (b)
  **silent drop** if the peer identity is unknown/disconnected (no error). *N-dependence:* more in-flight
  replies (T·D) per window as N,T↑ raises pressure on SNDHWM, making (a) marginally more reachable; (b) is
  driven by peer disconnect, N-independent.

- **DOF-7 — Compute-thread pin override.** `config.py:5–6` uses `os.environ.setdefault` → an externally-set
  `XLA_FLAGS`/`OMP_NUM_THREADS` survives. *Admits:* a multi-threaded XLA forward (different S band, possibly
  contending with the host's 4 vCPUs). *N-dependence:* none; an environment fact. (Default = single-thread.)

- **DOF-8 (bench) — `wakeup` ∈ {group, leaf}.** `stage_a_server.py:57`. *Admits:* one forward per drain
  (group) **or** one forward per queued request (leaf). *N-dependence:* under `leaf`, forwards-per-drain =
  message multiplicity, which **rises with N** (more ready slots → more/fatter messages per drain), so `leaf`
  multiplies forward count with N — the inverse of production coalescing.

- **DOF-9 (bench) — `e_policy` ∈ {padmax, bucket}.** `stage_a_server.py:61–64`, `_bucket_for:32–37`.
  *Admits:* fed shape = `max_batch` (padmax) **or** the smallest bucket ≥ real, ≤512, **or real itself when
  real>512** (no pad). *N-dependence:* as N↑ realized `real` climbs the bucket ladder 64→256→512→(unpadded),
  stepping S up and pad-fraction down at each threshold.

---

## 11. Representative executions (each a concrete trace; stability; N-dependence)

### E1 — Production coalescing under service time (the central latitude; Z3-confirmed)

Light-to-moderate load; one lone early request, then a burst during its service.

| step | state | transition | code_ref |
|---|---|---|---|
| 0 | `S_POLL` | T2: one request from stream A queued → poll truthy | `:165` |
| 1 | `S_PULL` | T4 then T6: pull the 1 request (1 row), then `Again` | `:173,174` |
| 2 | `S_FORWARD` | T7→T8: pad 1→256, one forward, `asarray` blocks for `S` | `:58,61` |
| 3 | (during step 2) | streams B,C,D emit (3 messages, 45 rows total) → queue in ROUTER | (causal §3) |
| 4 | `S_SCATTER` | T8→T10: reply to A; loop ends | `:200` |
| 5 | `S_POLL`→`S_PULL` | T2,T4×3,T6: pull all 3 coalesced (45 rows) | `:171–185` |
| 6 | `S_FORWARD` | pad 45→256, **one** forward at the **same** steady shape/S | `:58,61` |

- *Stability:* **self-reinforcing** under steady offered load — once a service window is busy, the arrivals
  that pile up during it form the next batch, perpetuating coalescing. It is the attractor of the closed
  loop at moderate load.
- *N-dependence:* more reachable and **larger** as N↑ — K=N·base grows the rows-per-message, so the cycle-1
  coalesced batch climbs toward `max_batch`; past T·K ≳ max_batch it pins at the cap (E2).

### E2 — Production saturation (service-bound, heavy load / large N·T)

| step | state | transition | code_ref |
|---|---|---|---|
| 0 | `S_PULL` | T4 repeated until `total_rows ≥ max_batch` | `:171,185` |
| 1 | `S_PULL` | T11: cap hit; break with B ∈ [max_batch, max_batch+K−1] | `:171` |
| 2 | `S_FORWARD` | forward at `max(max_batch,B)`; if B>max_batch, **unwarmed** shape → compile spike (DOF-4) | `:58,61` |
| 3 | `S_SCATTER`→`S_POLL` | scatter; queue still has ≥1 → next drain immediately POLLIN | `:200,165` |

- *Stability:* **self-reinforcing** while offered load > `max_batch/S`; queue depth grows (latency ↑, bounded
  by T·D messages), batch pinned at cap.
- *N-dependence:* **more reachable as N↑** (N is exactly the knob that manufactures the ready slots to
  saturate); the over-fill window `K−1 = N·base−1` widens with N, so the compile-spike sub-case (step 2)
  grows with N.

### E3 — Arrival-bound idle/poll (light load, small N)

| step | state | transition | code_ref |
|---|---|---|---|
| 0 | `S_POLL` | T1: poll times out (no arrival in 100 ms), loops, re-checks `_stop` | `:163–166` |
| 1 | `S_POLL` | T2: a single arrival → truthy | `:165` |
| 2 | `S_PULL`→`S_FORWARD`→`S_SCATTER` | drain=1 row, pad 1→256, forward, reply | `:171,58,200` |

- *Stability:* **transient** per-cycle but the *regime* is stable at low load; padding fraction ≈
  `(max_batch−1)/max_batch` (wasteful).
- *N-dependence:* **less reachable as N↑** — N manufactures concurrency that fills windows, pushing the
  system out of arrival-bound toward E1/E2. At fixed small total load, larger N still empties the idle poll
  because each thread holds more ready slots.

### E4 — Bench `leaf` fan-out (de-coalescing)

| step | state | transition | code_ref |
|---|---|---|---|
| 0 | `S_PULL` | drain pulls G=3 messages together (45 rows) | `:171–185` |
| 1 | `B_PLAN` | `wakeup=leaf` → `groups = [[d0],[d1],[d2]]` | `stage_a_server.py:57` |
| 2 | `B_GROUP_FWD`×3 | **three serial forwards**, each padded by `e_policy` (bucket: r_j→64) | `:58,65` |
| 3 | `B_GROUP_SCAT`×3 | reply per group | `:69–70` |

- *Stability:* **self-reinforcing** under load — `leaf` structurally refuses to coalesce; per-drain service
  time = Σ over G of `S(bucket(r_j))`.
- *N-dependence:* **worsens with N** — larger N → fatter/more messages per drain → larger G → more serial
  forwards; throughput in rows/s falls relative to `group`. This is the explicit anti-coalescing baseline.

### E5 — Bench `bucket` ladder climb

| step | state | transition | code_ref |
|---|---|---|---|
| 0 | `B_GROUP_FWD` | real=40 → `_bucket_for=64` → forward at 64-shape | `:32–37,64` |
| 1 | (load rises with N) | real=200 → bucket 256 → forward at 256-shape (S steps up) | `:35,64` |
| 2 | (load rises) | real=400 → bucket 512; real=600 → `pad_to=512<600` → **unpadded** 600-forward (BT6) | `:37,58` |

- *Stability:* **self-reinforcing** at each plateau; transitions between buckets are transient.
- *N-dependence:* **monotone climb with N** — N drives `real` up the 64→256→512→(unpadded) ladder; S steps
  up and pad-fraction steps down at each boundary; past 512 the over-bucket forward is unwarmed unless
  `max_batch>512` warmed it (`warmup` covers buckets ∪ {max_batch}, `stage_a_server.py:82`).

---

## 12. n_dependence_summary

N (`trees_per_thread`) enters the server **only** through the peer: it scales K = N·base (`runner_wire_batched.cpp:286`),
the per-thread parked-slot and per-message row ceiling. Increasing N manufactures more concurrently-ready
slots per stream, which (a) **fattens and burstifies arrivals** within any service window, (b) via the
single-thread serialization + non-blocking drain (§3) **inflates the coalesced batch of the next drain**,
sliding the server from arrival-bound (E3, high padding, throughput < max_batch/S) toward service-bound (E2,
batch pinned at `max_batch`, padding → 0, throughput → max_batch/S). In the **production** server the forward
**shape is constant** (pad-to-max_batch), so N changes batch *occupancy* and *queue depth/latency*, **not**
the per-forward service time — except it widens the over-fill window `K−1 = N·base−1`, introducing larger,
unwarmed `B>max_batch` forwards (a compile spike) more often. In the **bench** server N additionally climbs
the bucket ladder (stepping service time up) under `bucket`, and multiplies serial forward count under `leaf`.

---

## 13. DOF-control notes (what each constraint forbids if removed)

- **DOF-1/2 controlled by:** the `total_rows < max_batch` cap (`:171`) + non-blocking `Again` break (`:174`).
  *Remove the cap →* admits unbounded single-forward batches (up to T·K), executions the code forbids
  (forward shape would exceed `max_batch` arbitrarily, no warmed shape). *Remove the `Again` break →* the
  drain would block waiting for more, collapsing coalescing into a never-returning recv — forbidden (the code
  is strictly non-blocking inside the drain).
- **DOF-3 controlled by:** the `asarray` blocking barrier (`:61`) + the single steady shape (pad, `:58,198`).
  *Collapse S to an instant →* admits zero-latency replies and unbounded throughput, violating causal
  constraint C1 (positivity) and erasing the coalescing mechanism (no service window for arrivals to pile up
  in). *Make S depend on real B (drop the pad) →* would admit production executions where a 1-row drain is
  cheaper than a 256-row drain; the pad-to-max forbids that, so the model must keep S shape-constant in
  production.
- **DOF-4 controlled by:** cap-checked-at-loop-top semantics (`:171`). *Forbid over-fill (cap per row) →*
  would remove the `B>max_batch` unwarmed-forward executions the code genuinely produces — over-constrained.
- **DOF-5 controlled by:** `poll()` call site (`:194`, production only). *Force `current()` always →*
  forbids the mid-stream weight-change executions the production server permits (and that the bench, by
  construction, does not have).
- **DOF-6 controlled by:** ROUTER_MANDATORY=0 + default SNDHWM (unset in `inference_server.py`). *Set
  ROUTER_MANDATORY=1 →* the silent-drop scatter (DOF-6b) becomes a raise, removing the silent-drop execution
  — i.e. the unset option is exactly what admits the silent drop; the model must keep it.
- **DOF-8/9 controlled by:** the bench knobs (`stage_a_server.py:41,57,61`). *Fix to (group,padmax) →*
  collapses the bench to the production server at the bench's `max_batch`; the {leaf, bucket} executions
  (E4, E5) become unrepresentable — correctly so, those are bench-only latitudes.

---

## 14. Fidelity self-audit

**Possible over-permissions (admitting executions the code may not produce):**

- The model lets the production drain over-fill to `max_batch+K−1`. This requires the *last* admitted message
  to carry up to K rows. Whether a real producer ever emits a K-row message depends on search dynamics
  (all K slots ready at one `issue_one`); the code permits it (`runner_wire_batched.cpp:437–443` gathers all ready),
  so I keep it, but if in practice messages are usually small the large over-fill is rare. Kept as admitted
  (code-permitted), flagged as load-dependent.
- DOF-6b (silent scatter drop) is admitted because ROUTER_MANDATORY is unset; it is only *reachable* if a
  peer disconnects mid-flight. Under RELY R5 (producers always consume) it may be practically unreachable,
  but the socket option genuinely permits it, so the model keeps it rather than forbidding it.
- DOF-7 (XLA thread override) is admitted via `setdefault`; in the normal run the env is not pre-set, so the
  single-thread pin holds. Admitting the multi-thread band is faithful to the *code* (the override path
  exists) even though the deployment fixes it.

**Possible over-constraints (forbidding executions the code can produce):**

- I model arrivals as a per-stream **ordered** point process (a_{i,1}<a_{i,2}<…). The producer is strictly
  sequential per thread (one `issue_one`/`recv_batch` loop, `runner_wire_batched.cpp:457–475`), so per-stream
  ordering is real; **cross-stream** order is fully free, which I preserve. I do not believe this
  over-constrains.
- I treat the production fed-shape as a single constant `max_batch×in_dim`. If `in_dim` varied across runs
  (different feature dim), there would be more than one steady shape; within a single run `in_dim` is fixed
  (`fb.dim()`, `runner_wire_batched.cpp:275`), so this is faithful per-run. Across runs it is a parameter, not a
  per-execution freedom — correctly not modeled as in-run latitude.
- The malformed-request path (G5/T5) is modeled as "no reply, sibling-safe". I did **not** model the
  downstream producer RCVTIMEO firing as a server transition (it is the peer's state machine), only noted it
  as a liveness coupling. This is a boundary of scope, not an over-constraint of the server.

**Timing not collapsed:** confirmed — source inter-arrival and sink service time are both bounded
nondeterministic durations; only causal equalities (positivity, no-reply-before-forward, serialization,
reply-coupled emission) and the structural shape-constancy of the production pad are asserted.

---

## 15. Z3 confirmation (confirmation only, not the source of trust)

`/home/bork/w/vdc/chocobo/runs/leaf-eval-model-2/out/check_server_greedy_drain.py` encodes the central
latitude E1: a lone early request drained alone in cycle 0, and a coalesced batch of ≥3 requests in cycle 1,
**all of which arrived strictly during cycle 0's service window** `(c0_wake, c0_wake+c0_dur]`, with the
single-thread serialization `c1_wake ≥ c0_wake + c0_dur`, the `max_batch` row cap, and positive service
times. Result (run under `nice -n 19 timeout 90`):

```
result: sat
c0_wake = 0  c0_dur = 1  c0_done = 1
c1_wake = 1  c1_dur = 1
  req 1: arr=0   rows=1  in_c0=True   in_c1=False
  req 2: arr=1/2 rows=2  in_c0=False  in_c1=True
  req 3: arr=1   rows=1  in_c0=False  in_c1=True
  req 4: arr=1   rows=42 in_c0=False  in_c1=True
  realized batch rows: cycle0=1  cycle1=45  (both padded to max_batch=256)
ADMISSIBLE
```

This confirms the model admits the "service time shapes the next batch size" execution (cycle-0 batch = 1
row, cycle-1 coalesced batch = 45 rows, both padded to the same `max_batch=256` shape → same service band),
which is the production server's defining behavior and the basis of its N-dependence (§8, §12).
