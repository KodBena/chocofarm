# A faithful model of the Python server side of the leaf-evaluation transport boundary

**Subject.** `InferenceServer` (the ZeroMQ ROUTER) + the MLP forward sink, as the *server* party of a
two-party transport protocol whose peer is the C++ `WireLeafPool` DEALER driver.

**Method.** Derived FORWARD from the code's operational semantics. Every state, transition, guard, free
choice and timing latitude is mapped to a specific code line or a named causal necessity. No outside
expectation of how the system "ought" to behave is introduced. Timing is modeled as bounded
nondeterminism, constrained only as the code and causality require.

**Files read end to end (ADR-0002):**
- `chocofarm/az/inference_server.py` (the whole file, lines 1–457)
- `chocofarm/az/forward.py` (1–64)
- `chocofarm/az/inference_wire.py` (1–185)
- `chocofarm/az/wire_spec.py` (1–84) — the codec's SSOT, pulled in because `inference_wire` derives from it
- `cpp/include/chocofarm/wire_leaf_pool.hpp` (1–243) — the peer DEALER wrapper (RELY grounding)
- `cpp/src/runner_wire_batched.cpp` (1–630) — the peer driver, BOTH the strict-barrier and pipelined arms
- `docs/design/zmq-inference-service.md` (1–367) — protocol intent

**Environment facts checked against the running stack (not run):** pyzmq 27.1.0, libzmq 4.3.5 (so the
libzmq 4.3.x defaults below apply: SNDHWM=RCVHWM=1000, ROUTER without `ROUTER_MANDATORY` silently drops
unroutable frames and otherwise blocks-or-drops on a full per-peer pipe).

---

## 0. The socket-option census (the blocking surface is a function of exactly these)

A socket's blocking behavior is determined by which options are set. I enumerate them from the code.

### 0.1 The server's ROUTER socket — `inference_server.py`

```
L315:  self._sock = self._ctx.socket(zmq.ROUTER)
L316:  self._sock.bind(bind)                       # default "tcp://127.0.0.1:5599"
L317-318: self._poller.register(self._sock, zmq.POLLIN)
L454:  self._sock.close(linger=0)                  # teardown only
```

The ROUTER socket has **NO `setsockopt` call anywhere** (grep over the file is empty except the
`close(linger=0)` at L454). Therefore every option is at its libzmq default:

| option | value | consequence (derived) |
|---|---|---|
| `ZMQ_SNDHWM` | 1000 (default) | the ROUTER's *outbound* per-peer pipe holds ≤1000 queued messages before back-pressure |
| `ZMQ_RCVHWM` | 1000 (default) | each peer's *inbound* pipe into the ROUTER holds ≤1000 before back-pressure to that DEALER |
| `ZMQ_RCVTIMEO` | -1 (default, infinite) | but recv is ALWAYS called `flags=zmq.NOBLOCK` (L350), so RCVTIMEO is moot |
| `ZMQ_SNDTIMEO` | -1 (default, infinite) | `send_multipart` (L387) is BLOCKING — relevant below |
| `ZMQ_ROUTER_MANDATORY` | 0 (default, OFF) | a send to an **unknown/disconnected** identity is **silently dropped**, not an error and not a block |
| `ZMQ_LINGER` | -1 at runtime; forced 0 only at `close` (L454) | irrelevant to the serve loop |
| context options | none | `zmq.Context()` (L314) is plain — default 1 I/O thread, default max sockets |

The bind is `tcp://127.0.0.1:5599` by default (L306); the peer connects `wcfg.endpoint` (the design note
and `wire_leaf_pool.hpp` L70 mention an `ipc://` variant too — the transport is parameterized, the
model is transport-agnostic over tcp/ipc).

### 0.2 The peer's DEALER socket — `wire_leaf_pool.hpp` (for the RELY, §6)

```
L81-82:  int linger = 0; zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof);
L83:     zmq_setsockopt(sock, ZMQ_RCVTIMEO, &timeout_ms, sizeof);   // == wcfg.timeout_ms
L84:     zmq_connect(sock, endpoint)
```

The DEALER sets exactly LINGER=0 and RCVTIMEO=`timeout_ms`. It sets **NO** SNDHWM, RCVHWM, or SNDTIMEO,
so those are libzmq defaults (SNDHWM=RCVHWM=1000, SNDTIMEO=-1 infinite). The C++ context
(`runner_wire_batched.cpp` L87/L394) is a plain `zmq_ctx_new()` with no `zmq_ctx_set`. The DEALER's
sends (`zmq_send`, L139/L142) are blocking with no SNDTIMEO, and its recv (`zmq_msg_recv`, L217) blocks
up to RCVTIMEO. These ground the RELY in §6.

---

## 1. The decoded vocabulary of the server loop

The server is **single-threaded** by construction (module docstring L33-35; class docstring L292-294).
There is exactly one thread executing the loop; JAX/XLA owns the forward. So the model is a single
sequential automaton — the *concurrency* it must capture is the concurrency **between the server and the
DEALER peers over the wire**, plus the concurrency **between the forward-in-progress and requests
buffering in the ROUTER's receive queue** (these buffer in the kernel/libzmq pipes while the single
thread is busy, L429-432 docstring; this is the whole point of greedy-drain).

The loop body (`serve_forever`, L428-439):

```
L435  self._params_source.current()          # standup assertion (params exist)
L436  while not self._stop:
L437      drained = self._drain()            # BLOCK for ≥1, then non-blocking drain to cap
L438      if not self._stop and drained:
L439          self._serve_batch(drained)     # poll-reload, ONE forward, scatter
```

`_drain` (L322-363):

```
L339  while not self._stop:
L342      if self._poller.poll(timeout=_POLL_INTERVAL_MS):   # 100 ms bounded poll
L343          break
L344  if self._stop: return []                                # woke on stop
L348  while total_rows < self._max_batch:
L350      frames = self._sock.recv_multipart(flags=zmq.NOBLOCK)
L351      except zmq.Again: break                             # nothing more queued → run the batch
L353      ident = frames[0]; envelope = frames[1:-1]; payload = frames[-1]
L357      X = decode_request(payload)                         # (B_i, in_dim); WireError on malformed
L358-360  except: self._reject(ident, exc); continue          # loud drop of THIS frame, batch unaffected
L361      drained.append((ident, envelope, X)); total_rows += X.shape[0]
```

`_serve_batch` (L372-387):

```
L381  reloaded = self._params_source.poll()                   # between-batch version-gated reload
L382  params, y_mean, y_std = reloaded or self._params_source.current()
L383  rows = [(ident, X) for ident, _env, X in drained]
L385  run_microbatch(forward_fn, params, y_mean, y_std, rows, pad_to=self._max_batch)   # ONE forward
L387  self._sock.send_multipart([ident, *envelope, resp])     # scatter, per drained request
```

`run_microbatch` (L134-189): concatenate the `(B_i, in_dim)` matrices into `(N_total, in_dim)` (L164),
**pad up to `pad_to=max_batch`** with zero rows (L171-172), call `forward_fn` ONCE (L177), pull back the
`(>=B, 1+n_actions)` block (L177-178), scatter each request's own `counts[i]` rows back as one response
frame (L184-188).

---

## 2. The operational STATE MACHINE

States are the server thread's control point plus the relevant queue/forward status. The "wire" — the
ROUTER's inbound pipe — is an environment the model reasons about through the RELY, but its *occupancy*
is a state variable the server observes only through `poll`/`recv`.

### 2.1 States

| name | meaning |
|---|---|
| `STANDUP` | constructed; `serve_forever` entered; `current()` asserted (L435). One-shot. |
| `POLL_WAIT` | inside `_drain`'s bounded-poll loop (L339-343): blocked in `poll(timeout=100ms)`, waiting for the FIRST request or a `_stop` re-check. CPU ~0 between wakeups. |
| `DRAINING` | poll returned readable; executing the non-blocking `recv_multipart` drain loop (L348-362), accumulating ≤`max_batch` rows. |
| `RELOAD_CHECK` | `_serve_batch` entered; calling `params_source.poll()` (L381) — may do a redis read (RedisParamsSource) or be a no-op (StaticParamsSource L249). |
| `FORWARD` | inside `run_microbatch` → `forward_fn` (L177): XLA executes ONE padded forward over `(max_batch, in_dim)`. Consumes SERVICE TIME. While here, the single thread is NOT servicing the socket; inbound requests buffer in the ROUTER pipe. |
| `SCATTER` | the `send_multipart` loop (L384-387): pushing each response back to its identity+envelope. |
| `STOPPED` | `_stop` observed; loop exited (L436/L344/L438). Terminal w.r.t. serving. |

`WARMUP` is a distinct pre-loop phase (L389-426) that runs the SAME forward path with no socket; it is a
proper sub-machine: for each `b` in `batch_sizes` it builds a `(b,in_dim)` dummy and calls
`run_microbatch(..., pad_to=max_batch)` (L426), forcing XLA to compile the single `(max_batch,in_dim)`
executable. Modeled as a sequence of FORWARD steps with no SCATTER and no socket interaction.

### 2.2 Transitions (guard / action / code_ref / free-vs-determined)

| # | from → to | guard | action | code_ref | free? |
|---|---|---|---|---|---|
| T0 | — → STANDUP | server constructed | bind ROUTER, register poller | L315-318 | determined |
| T1 | STANDUP → POLL_WAIT | `current()` returns (params present) | enter `_drain` | L435→L437→L339 | determined |
| T2 | POLL_WAIT → POLL_WAIT | `poll(100ms)` returned 0 (no readable) AND `not _stop` | re-issue poll | L339,L342 | **free (timing)**: whether the 100 ms window elapses with nothing queued depends on peer emission timing |
| T3 | POLL_WAIT → STOPPED | `_stop` true at loop head | `return []`, loop sees empty → re-checks `_stop` | L339/L344, L438 | determined by `_stop` (set externally, L445) |
| T4 | POLL_WAIT → DRAINING | `poll(100ms)` returned readable (≥1 frame queued) | `break` out of poll loop | L342-343 | **free (timing)**: WHEN ≥1 arrives is peer-set |
| T5 | DRAINING → DRAINING | `total_rows < max_batch` AND `recv_multipart(NOBLOCK)` succeeded | decode payload; on success append `(ident,env,X)`, `total_rows += B_i` | L348-351,L357,L361-362 | **free (arrival count)**: how many frames are *currently queued at this instant* is peer/scheduling-set |
| T6 | DRAINING → DRAINING | a drained frame's `decode_request` raised | `_reject(ident)` logs; `continue` (frame dropped, batch unaffected, `total_rows` unchanged) | L356-360, L365-370 | determined by frame content (RELY: peer never sends malformed; §6) |
| T7 | DRAINING → RELOAD_CHECK | `recv_multipart(NOBLOCK)` raised `zmq.Again` (nothing more queued) OR `total_rows >= max_batch` | `break`; return `drained`; `_serve_batch` entered iff `drained` non-empty | L348,L351-352, L438-439 | **free (timing)**: the cap-vs-Again boundary is exactly the instantaneous queue depth |
| T8 | DRAINING → POLL_WAIT | drain produced an EMPTY list (every queued frame was malformed-and-rejected, so `drained==[]`) | `_serve_batch` skipped (L438 `and drained`); loop re-enters `_drain` | L438, L437 | determined (RELY makes this unreachable in practice; structurally present) |
| T9 | RELOAD_CHECK → FORWARD | always (after `poll()`/`current()` resolve params) | concat+pad rows; call `forward_fn` | L381-385, L164-177 | determined |
| T10 | FORWARD → SCATTER | the forward's `np.asarray` returns (XLA done) | read back `(>=B,1+n_actions)`; build responses | L177-189 | **free (timing)**: SERVICE TIME is nondeterministic (§5) |
| T11 | SCATTER → SCATTER | more drained requests to answer | `send_multipart([ident,*env,resp])` for next | L384-387 | determined (1:1 with drained, in order) |
| T12 | SCATTER → POLL_WAIT | all responses sent | `_serve_batch` returns; loop head; `_drain` again | L387→L439→L436→L437 | determined |
| T13 | any non-FORWARD head → STOPPED | `_stop` observed at L436 or L339 or L438 | exit loop | L436,L339,L438 | determined by external `stop()` |
| W0 | STANDUP → WARMUP → POLL_WAIT | `warmup(batch_sizes)` called before `serve_forever` | per b: build dummy, `run_microbatch(pad_to=max_batch)` (forces compile) | L412-426 | determined (caller-driven; the set `batch_sizes` is a free input) |

**Key determinacy facts the code fixes (not free):**
- The drain is **greedy and non-blocking after the first**: it takes *whatever is queued now*, never
  waits for more (L350 `NOBLOCK`, L351-352 `Again`→break). So B is a pure function of the instantaneous
  queue, not a tunable window. (design §3 L122-125.)
- The forward runs **exactly once per drained batch** (L177; design §3 L117). Never zero, never twice.
- The batch is **always padded to `max_batch`** before the forward (L171-172), so XLA sees ONE shape and
  the service-time's batch-size dependence is the dependence of a **fixed `(max_batch,in_dim)`** forward
  — see §5.2.
- Responses scatter **1:1 with drained, in drained order** (L184-188 align by `counts`; L384 `zip`).
- Malformed frames are **dropped from the batch, never zero-filled** (L358-360, design §5 L165).

### 2.3 The non-obvious blocking points (the "blocking surface", captured exactly)

There are exactly **four** places the single thread can wait or stall. I name each and its nature.

1. **`poll(timeout=100ms)` (L342) — bounded wait.** The only *intended* idle wait. Re-issued until
   readable or `_stop` (L339). Wakes every ≤100 ms to re-check `_stop` (L341 comment; `stop()`
   docstring L442-444). NOT a spin — parks at ~0 CPU (L336-337). This is the ONLY place a request's
   arrival is awaited. **Latitude:** the number of POLL_WAIT self-loops before T4 is unbounded (idle
   server) and peer-timing-determined.

2. **`recv_multipart(flags=zmq.NOBLOCK)` (L350) — never blocks.** By construction non-blocking; raises
   `zmq.Again` when the pipe is momentarily empty (L351). So the drain CANNOT stall mid-batch; it
   returns the current queue. (This is why B self-clocks.)

3. **`send_multipart([...])` (L387) — CAN block.** This is the subtle one. `SNDTIMEO` is default -1
   (§0.1), so a `send` that cannot enqueue **blocks the thread indefinitely**. ROUTER send semantics
   (libzmq 4.3, `ROUTER_MANDATORY` OFF — §0.1):
   - If the destination identity is **unknown/disconnected**, the message is **silently dropped**, send
     returns immediately (no block, no error). A reply to a peer that has gone away is lost — and because
     the server holds no per-request timeout/bookkeeping, it simply moves on (the peer's RCVTIMEO surfaces
     it, §6 RELY).
   - If the destination is **known but its outbound pipe is at SNDHWM (1000 queued)**, the send
     **blocks** until the pipe drains (because SNDTIMEO=-1). This is a real stall surface: a peer whose
     DEALER stops receiving (e.g. its RCVHWM full because the search isn't consuming) could in principle
     back-pressure the ROUTER's send. In practice the peer issues a *blocking* request/await cycle and
     always recvs its reply promptly (RELY §6: `recv_batch` is called right after each `submit_batch`,
     L321-324/L580 of the driver), so the per-peer reply queue depth is ≤ D (the in-flight cap, default
     `wire_pipelined` `max_inflight_msgs`, or 1 for the strict barrier). D ≪ 1000, so the SNDHWM block is
     **never reached** under the RELY — but the model *admits* it as a reachable state if the RELY is
     violated (DOF-7).

4. **`params_source.poll()` (L381) — CAN block (RedisParamsSource only).** For `RedisParamsSource`,
   `poll()` calls `version_supplier()` and, on a version change, `transport.read_weights` (a redis read,
   L283-285) — a synchronous network/IPC call that takes time and could stall if redis is slow. For
   `StaticParamsSource` (the test/parity path) `poll()` returns `None` immediately (L249-250). This is a
   between-batch stall, never mid-forward (single-threaded, L381 is before L385).

**There is no other wait.** Notably the FORWARD (L177) is not a *wait* in the OS sense — it is CPU/XLA
compute consuming SERVICE TIME — but it is the dominant time sink and the reason requests buffer.

---

## 3. The wire / request buffering model (how requests accumulate while a forward runs)

This is the heart of the concurrency the server must capture, and it is entirely an emergent property of
single-threaded greedy-drain + ZeroMQ's pipes.

- The ROUTER has, per connected DEALER peer, an inbound pipe of capacity `RCVHWM`=1000 messages
  (§0.1). Requests sent by a peer while the server is in FORWARD/SCATTER/RELOAD_CHECK are **buffered in
  these pipes by libzmq's I/O thread**, not lost (until 1000 deep per peer, which the RELY's D≪1000
  never approaches).
- When the server returns to POLL_WAIT/DRAINING, `poll` reports readable and the drain pulls *all*
  currently-buffered frames (across all peers, fair-queued by ROUTER) up to `max_batch` rows.
- Therefore **batch size B (in rows) at drain time = the number of request-rows that arrived during the
  previous FORWARD+SCATTER+RELOAD window, capped at `max_batch`** (L348). Under heavy load B rises to the
  cap; under light load B≈1 (design §3 L122-125). This is the self-clocking the design claims, and it is
  a *determined* function of arrival timing — the timing being the free variable (§5.1).
- **Cross-peer coalescing.** A single forward can mix rows from *different DEALER peers* (different C++
  worker threads) and from *different correlation ids of the same peer* (the pipelined driver holds D
  messages outstanding, L578-596). The server does not distinguish — it concatenates by drain order
  (L164) and scatters back by identity+envelope (L387). The envelope (`frames[1:-1]`, L354) carries the
  8-byte corr-id opaquely (L128-130 docstring) and is echoed verbatim (L387), so the peer re-pairs by
  corr-id (RELY §6).
- **Partial-message safety.** ZeroMQ delivers whole multipart messages atomically; `recv_multipart`
  (L350) never returns a half-frame. So the drain never sees a torn `[identity][corr][payload]`.

---

## 4. The codec boundary (value-level, determined)

`decode_request(payload)` (L357 → inference_wire L105-127): validates the protocol byte (==2, wire_spec
L49), `B≥1`, `in_dim≥1`, exact byte count `B·in_dim·4`, and all-finite. Any violation → `WireError`,
caught at L358 → `_reject` → the frame is **dropped from the batch** (L360 `continue`), `total_rows`
unchanged. This is **determined by frame content**; under the RELY (§6) the peer's `encode_request`
(wire_leaf_pool L133, inference_wire L95-102) always produces a well-formed v2 frame, so T6/T8 are
RELY-unreachable but structurally present (fidelity: the code CAN reject, so the model admits it).

`encode_response(v_rows, l_rows)` (L187 → inference_wire L130-158) packs `[ver][B][n_actions][records]`.
Value-only nets (`logits=None`, forward_core L62) emit `n_actions=0`. Determined by the forward output
shape (L181 `has_logits = out_arr.shape[1] > 1`).

This layer adds **no timing latitude** and **no concurrency** — it is a pure function at the boundary.
Its only role in the state machine is T6 (reject) vs T5 (accept).

---

## 5. The TIMING MODEL (the two nondeterministic clocks, as bounded nondeterminism)

The task is explicit: model source-emission and sink-service timing as bounded nondeterminism — pin
nothing the code leaves free, forbid nothing causality forbids.

### 5.1 SOURCE emission timing (peer request pacing) — nondeterministic interval

**What the code says.** The server NEVER sets a peer's emission time; it only *observes* arrivals via
`poll`/`recv` (L342/L350). On the peer side the emission time is set by the search's own progress:
`run_episodes_wire_pipelined` issues a coalesced message whenever slots are ready (L551-569 `issue_one`),
which happens after `resume_with`+`advance` walk the search forward by an amount the code does not fix
(`advance` L502-510 loops the per-ply state machine; the number of plies and the search work per ply is
data-dependent). The strict-barrier arm (L310-337) emits one gathered message per round, again paced by
how long the resumed searches take to re-park.

**Representation.** For each request `r` emitted by a peer, an emission instant `e_r ∈ ℝ_{≥0}`,
constrained ONLY by:
- **(C1) positivity / monotonic per-peer issue:** within one peer (one DEALER, one inflight map), the
  *next* reply-dependent message cannot be issued before the prior reply that unparks its slots is
  received: `e_{r'} > a_{reply(slot of r')}` where the slot was outstanding. This is the
  `submitted[]` guard (driver L437,L543,L564,L588): a slot in flight is not re-gathered until its reply
  resumes it. (For the strict barrier, D=1: the next gather strictly follows the prior reply, L323-326.)
- **(C2) in-flight cap:** at most D messages per peer are outstanding (`inflight_msgs < D`, driver
  L578/L596; D=1 for strict barrier). So the count of un-replied requests from one peer is ≤ D.
- **(C3) otherwise free:** the *interval* between a peer's successive eligible emissions is an arbitrary
  positive duration (the search's variable work). NOT a constant, NOT zero, NOT bounded above by the
  code. This is the latitude DOF-1 names.

**Crucially NOT collapsed to a constant.** The design note itself (§4 L154-159) records that *which leaves
land in which batch depends on arrival timing* and is run-to-run nondeterministic. Pinning `e_r` to a
grid would forbid exactly the executions §4 says occur. So emission is genuinely free bounded
nondeterminism.

### 5.2 SINK service timing (the forward) — nondeterministic positive duration with a derived batch-shape law

**What the code says about service time:**

1. **One forward per batch, one-at-a-time.** Single-threaded (L33-35); `forward_fn` is called once per
   `_serve_batch` (L177) and the result is *pulled back* with `np.asarray` (L177), which **blocks the
   thread until XLA has finished** (run_microbatch docstring L173-176; warmup docstring L405-407 says the
   asarray "forces XLA compilation to COMPLETE, not merely enqueue"). So FORWARD is a single
   non-overlapping compute interval — no two forwards run concurrently.

2. **Fixed input shape via padding.** Every batch is padded to `(max_batch, in_dim)` before the forward
   (L171-172). So the forward's matmul shape is **constant** regardless of the real B. The jitted
   executable is compiled once for that one shape (jit_forward_core docstring L100-104: "pads every batch
   to one shape, so it compiles a SINGLE executable"; warmup L389-411 forces this compile up front).

3. **Compile vs steady-state.** A COLD forward for a shape not yet compiled includes the XLA compile cost
   (warmup docstring L394-400: the per-B recompile "dominated the server profile"; with padding there is
   exactly ONE shape, so at most ONE cold compile). `warmup` (L389-426) moves that cold compile *before*
   the loop, so under warmup the very first served forward is already steady-state. WITHOUT warmup, the
   first FORWARD pays the compile cost (a one-time tail), every subsequent FORWARD is steady-state.

**Representation.** For each forward instance `f` (a FORWARD step) a service duration `s_f ∈ ℝ_{>0}`,
constrained by:
- **(S1) positivity:** `s_f > 0` (compute takes time; the task forbids collapsing to an instant).
- **(S2) reply-after-forward:** every response scattered in the SCATTER following `f` has its
  send-instant `> end(f)` (the reply cannot precede the forward that produced it). Causally necessary;
  also code-enforced (SCATTER L387 is strictly after the forward returns L177).
- **(S3) one-at-a-time / non-overlap:** for the single server, FORWARD intervals are pairwise disjoint
  and totally ordered (single thread). A second forward's start `> first forward's end`.
- **(S4) shape-invariance of the steady-state law:** because the input is padded to a constant shape
  (L171-172), `s_f` does **NOT** depend on the real B (number of live rows). The steady-state service
  time is drawn from one distribution `D_steady` parameterized only by `(max_batch, in_dim, net
  architecture, host load)` — none of which the drain varies. **This is the one place the code JUSTIFIES
  near-constancy** (a fixed-shape matmul on a fixed host), but I do NOT collapse it to a literal constant:
  host scheduling jitter on the shared 4-vCPU VM, XLA runtime variation, and OS preemption make `s_f` a
  *bounded nondeterministic* duration around a shape-determined central value. So: free within
  `D_steady`, not pinned.
- **(S5) cold-compile tail:** the FIRST forward of a given executable (no warmup, or a shape never seen)
  has `s_f = s_compile + s_steady` with `s_compile ≫ s_steady` (warmup docstring L394-400). At most once
  per executable; with padding, at most once total. After warmup, S5 never fires in the loop.

**Why batch-size dependence is *modeled* but resolves to *padded-shape* dependence.** The task says model
"any dependence on batch size." The honest derivation: the *unpadded* concatenation is `(N_total,in_dim)`
(L164) with `N_total ≤ max_batch` (the drain cap L348), but it is **immediately padded to `max_batch`**
(L171-172) before the forward. So the forward never sees a variable row count — the batch-size dependence
is *erased by the padding by design* (that is the padding's stated purpose, L168-170). The model captures
this faithfully: `s_f` is independent of live B (S4), and the *only* batch-shaped cost a real B incurs is
the host-side `np.concatenate`/`astype`/`np.asarray` marshalling (L164,L172,L177), which scales with
`max_batch·in_dim` regardless of B and is dominated by the matmul. **Collapsing `s_f` to depend on B would
be UNFAITHFUL** (it would forbid the padded-shape executions the code actually runs). This is the
sharpest faithfulness point in the timing model.

### 5.3 Composed causal constraints (the full timing skeleton)

For any execution, the partial order on instants is:
```
e_r (peer emits)  <  a_r (server's recv at drain, L350)               # transit + buffering ≥ 0
a_r               ≤  start(f) where r ∈ batch(f)                       # drained before forward, L361,L177
start(f)          <  end(f)        with end-start = s_f > 0            # S1
end(f)            <  send_r' for every r' ∈ batch(f)                   # S2, scatter after forward, L387
send_r'           <  a'_{r'} (peer recv of reply)                     # transit ≥ 0
a'_{r'}           <  e_{r''} for any r'' whose slot was r''s in-flight # C1, peer re-issue after reply
```
No instant is pinned; only these inequalities (plus C2/C3/S3/S4) hold. This is exactly the latitude the
code leaves — a family of executions parameterized by the free `e_r` and `s_f`, quotient by these orders.

---

## 6. ASSUME-GUARANTEE contract

I am the SERVER. My peer is the C++ `WireLeafPool` DEALER driver(s). Each clause is checkable against the
named peer code.

### RELY (what the server assumes about the peer, over the wire)

- **R1 (well-formed v2 frames).** Every request frame's payload is a valid v2 `encode_request` output:
  `[ver=2][B≥1][in_dim≥1][B·in_dim finite f32]`. *Check:* peer builds it via `wire::encode_request`
  (wire_leaf_pool L133) from the shared `wire_spec.hpp` SSOT (drift-tested, design amendment 2026-06-16).
  Consequence if violated: T6 reject (server stays live; that one frame is dropped + logged, L365-370).
- **R2 (three-frame envelope `[corr-id:8B][payload]` after the ROUTER identity).** The peer sends
  `[corr-id (8 bytes, SNDMORE)][payload]` (wire_leaf_pool L139-142). The ROUTER prepends the peer
  identity, so the server sees `[identity][corr-id][payload]`, and `envelope=frames[1:-1]=[corr-id]`
  (L354). *Check:* L138-144. The server treats the envelope OPAQUELY and echoes it (L387) — it never
  parses the corr-id, so any envelope shape is round-tripped; R2 is only needed so the peer can match.
- **R3 (the peer eventually recvs its reply).** After each `submit_batch`, the peer calls `recv_batch`
  (driver L321-324 strict; L580 pipelined) and blocks up to RCVTIMEO (wire_leaf_pool L170-196). So the
  ROUTER's outbound per-peer pipe is drained promptly; its depth stays ≤ D ≪ SNDHWM(1000). *Check:* the
  driver's drain loops always pair a submit with a recv; `submitted[]`/`inflight_msgs` cap outstanding at
  D (driver L437/L578/L596). Consequence: the server's `send_multipart` (L387) never reaches the SNDHWM
  block (blocking-surface point 3). If R3 is violated (peer wedged not recving), the server *would*
  eventually block in send — DOF-7.
- **R4 (bounded outstanding per peer = D).** A peer holds at most D un-replied messages (D=1 strict, D=
  `max_inflight_msgs` pipelined). *Check:* `inflight_msgs < D` guard (driver L578/L596). This bounds how
  many rows one peer contributes to one drain.
- **R5 (the peer tolerates batch-composition roundoff and out-of-order corr-id replies).** The peer
  routes replies by echoed corr-id, not by arrival order (wire_leaf_pool L179-188), and accepts a
  reply whose B equals the submitted slot count (L185). *Check:* L179-196. So the server is free to
  coalesce/reorder across peers and corr-ids (it does — §3). It MUST return each corr-id's reply with B
  exactly equal to that request's B (server does: counts align 1:1, L184-188).
- **R6 (the peer survives a dropped reply only as a loud timeout, not a hang).** If the server drops a
  reply (e.g. ROUTER drops to a since-disconnected identity, point 3), the peer's `recv_batch` times out
  (RCVTIMEO, wire_leaf_pool L83,L217-220) → a loud Error → whole-pass abort (driver L324/L581). *Check:*
  L217-220. So a server-side drop is never a silent corruption on the peer; the server may rely on the
  peer to fail loudly rather than wedge.

### GUARANTEE (what the server guarantees the peer, grounded in server code)

- **G1 (every accepted request gets exactly one correctly-shaped reply, echoing its envelope).** For each
  drained request `(ident, envelope=[corr-id], X with B_i rows)`, the server sends exactly one
  `[ident][corr-id][encode_response(B_i records)]` (L184-188 build B_i-record response; L387 send with
  the verbatim envelope). The reply's B equals the request's B (L184 `counts`). *Ground:* L177-189,L387.
- **G2 (no silent coercion of a bad request).** A malformed frame is dropped + logged, never zero-filled
  into the forward (L358-360,L365-370; design §5 L165). The peer sees no reply for that corr-id and times
  out loudly (R6). *Ground:* L356-360.
- **G3 (one consistent net version per batch).** All rows of one forward use one `(params,y_mean,y_std)`
  snapshot taken once at the top of `_serve_batch` (L381-382); a reload happens only *between* batches
  (L381 before L385). So every leaf in a batch sees one net version (design §3 L136-137). *Ground:*
  L381-385.
- **G4 (the forward is row-independent / order-preserving).** The padded forward's real-row outputs are
  byte-identical to the unpadded forward (zero pad rows, row-independent matmul; L168-170 comment,
  forward_core L50-63), and outputs scatter back in drained order (L184-188). So coalescing a peer's
  leaf with others' does not change its prediction beyond accepted f32 roundoff (design §4 L150-153).
  *Ground:* L164-188, forward_core.
- **G5 (liveness under the RELY).** Given R1-R4, the server never blocks except in the bounded 100 ms
  poll (point 1) and the between-batch reload (point 4, fast under StaticParamsSource; a redis read under
  RedisParamsSource). It makes progress: every drain with ≥1 valid frame runs a forward and scatters
  (L437-439). *Ground:* the only unbounded wait is `send_multipart` at SNDHWM, excluded by R3.
- **G6 (clean shutdown without racing the socket).** `stop()` (L441-445) flips a flag the bounded poll
  observes within ≤100 ms (L339-344); the socket is closed only after the loop is between polls (L447-456
  docstring). The peer is not affected (its RCVTIMEO surfaces the silence). *Ground:* L339-344,L441-456.

---

## 7. DEGREES OF FREEDOM the code leaves (each with code_ref and admitted behaviors)

- **DOF-1 — source emission interval (peer pacing).** `e_r` is an arbitrary positive duration after the
  prior reply (C3, §5.1). *code_ref:* no server line fixes it; peer side driver L551-569 (`issue_one`
  fires on readiness, search-paced). *Admits:* B at any drain ∈ {1..max_batch}; an idle server looping in
  POLL_WAIT indefinitely (T2); a saturated server draining exactly `max_batch` every round (T7 via cap).

- **DOF-2 — instantaneous queue depth at drain (batch size B).** How many frames are buffered when the
  drain runs is the count that arrived during the prior FORWARD+SCATTER window (§3). *code_ref:* L348-352
  (drain until Again or cap). *Admits:* every B from 1 (light load) to `max_batch` (cap), and every mix
  of per-peer / per-corr-id rows summing to that B.

- **DOF-3 — sink service duration `s_f`.** A positive duration drawn from the shape-determined steady-
  state distribution (plus a one-time cold tail). *code_ref:* L177 (the forward + asarray block), pad
  L171-172, warmup L389-426. *Admits:* fast forwards (warm, idle host) and slow forwards (cold compile
  S5, or host contention) — and therefore, via the buffering law (§3), small or large *next* batches. The
  service time and the next batch size are causally coupled (a slow forward → more buffered → bigger next
  B), and the model preserves that coupling rather than treating them independently.

- **DOF-4 — cross-peer / cross-corr-id coalescing.** The drain mixes rows from any peers and any of a
  peer's D outstanding messages into one forward. *code_ref:* L164 concat by drain order; L354 opaque
  envelope; L387 scatter by identity+envelope. *Admits:* a single forward serving 1..T peers'
  leaves; the exact membership is arrival-timing-determined (design §4 L154-159 batch-composition
  nondeterminism) — so a given leaf's f32 value can vary run-to-run within the 1e-4 bar.

- **DOF-5 — reject-vs-accept of a frame.** A frame either decodes (accepted) or raises (rejected and
  dropped). *code_ref:* L356-360. *Admits:* a batch with some frames dropped (RELY-unreachable but
  structurally enabled); an entirely-rejected drain producing `drained==[]` → T8 (no forward, re-poll).

- **DOF-6 — reload timing / outcome (RedisParamsSource).** `poll()` may return new params (version
  advanced) or `None`. *code_ref:* L279-288, L381-382. *Admits:* a batch served on version v then the
  next on v+1; a reload that takes time (redis read L284); the version supplier can advance at any
  between-batch point — never mid-batch (G3).

- **DOF-7 — `send_multipart` stall reachability.** With SNDTIMEO=-1 and SNDHWM=1000, a send to a
  HWM-full peer pipe blocks. *code_ref:* L387, §0.1. *Admits:* (only if RELY R3 is violated) the server
  parked indefinitely in SCATTER. Under the RELY this is unreachable, but it is a genuine code-permitted
  state — the model includes it and pins it to the RELY violation.

- **DOF-8 — warmup batch-size set.** `warmup(batch_sizes)` (L389-426) is caller-supplied. *code_ref:*
  L418-426. *Admits:* a server entering the loop fully warm (every reachable shape compiled — though with
  padding there is one shape) or cold (no warmup → first forward pays S5).

- **DOF-9 — `_POLL_INTERVAL_MS` granularity.** The 100 ms poll window (L304) bounds the shutdown latency
  and the idle re-check cadence, NOT the batch latency (the drain runs the instant ≥1 frame arrives,
  T4). *code_ref:* L304,L342. *Admits:* `_stop` observed up to 100 ms late; an idle server waking 10×/s.

---

## 8. REPRESENTATIVE EXECUTIONS (concrete traces of genuinely-enabled transitions)

Each trace lists `step: state —transition→ state [code_ref]`. Stability = whether the regime is
self-reinforcing or transient.

### E1 — Light load, B≈1 (the low-latency regime). Exercises DOF-1, DOF-2.
```
1  POLL_WAIT  —T2 (poll 100ms, nothing)→ POLL_WAIT          [L342]
2  POLL_WAIT  —T2→ POLL_WAIT  (idle, several windows)        [L342]   # peer's search still working
3  POLL_WAIT  —T4 (one peer emits one corr-id msg, B_i=1)→ DRAINING   [L342-343]
4  DRAINING   —T5 (recv that one frame, decode ok)→ DRAINING [L350,L361]
5  DRAINING   —T7 (next recv → Again)→ RELOAD_CHECK          [L351-352]
6  RELOAD_CHECK —T9 (StaticParamsSource.poll→None)→ FORWARD  [L381-385]
7  FORWARD    —T10 (s_f steady, padded to max_batch)→ SCATTER[L177]
8  SCATTER    —T12 (one send to that identity+corr-id)→ POLL_WAIT [L387]
9  POLL_WAIT  —T2→ ...                                        [L342]
```
**Stability: self-reinforcing while load stays light.** A single-leaf forward is fast (S4); by the time
it finishes only ~1 new leaf has arrived (the peer is blocked awaiting its reply, R3/C2 with D small), so
the next B is again ≈1. The regime persists until emission rate rises (DOF-1). The design names this the
B≈1 low-latency regime (§3 L122-125).

### E2 — Heavy load, B saturates at `max_batch` (the high-throughput regime). Exercises DOF-2, DOF-3, DOF-4.
```
1  POLL_WAIT  —T4 (many peers/corr-ids already queued)→ DRAINING     [L342]
2  DRAINING   —T5 ×k (recv+decode rows, accumulating)→ DRAINING      [L350,L361]
   ...        (total_rows climbs toward max_batch)
3  DRAINING   —T7 (total_rows ≥ max_batch)→ RELOAD_CHECK             [L348 cap]
4  RELOAD_CHECK —T9→ FORWARD                                          [L385]
5  FORWARD    —T10 (s_f steady; same padded shape as E1!)→ SCATTER    [L177,L171-172]
6  SCATTER    —T11 ×B_msgs (send each request's reply)→ ... → POLL_WAIT[L384-387]
7  POLL_WAIT  —T4 (a full new batch already buffered during step 5)→ DRAINING [L342]
```
**Stability: self-reinforcing under sustained demand.** While B is capped, the forward takes the *same*
service time as E1 (padding, S4), but during that time ≥`max_batch` new rows buffer (DOF-3 coupling), so
the next drain again hits the cap (T7 via L348). The pipe stays full; B stays at `max_batch`. This is the
high-throughput attractor. (Note: because S4 makes the forward time *independent* of real B, light and
heavy load have the SAME per-forward cost — the throughput win is purely amortization, exactly the design
claim §0 L47, and the model reproduces it without assuming it.)

### E3 — Cold first forward then warm steady state. Exercises DOF-3 (S5), DOF-8.
```
0  STANDUP    —(no warmup called)→ POLL_WAIT                          [L435-437]
1  POLL_WAIT  —T4→ DRAINING —T5→ DRAINING —T7→ RELOAD_CHECK           [...]
2  RELOAD_CHECK —T9→ FORWARD                                          [L385]
3  FORWARD    —T10 (s_f = s_compile + s_steady, COLD)→ SCATTER         [L177; warmup doc L394-400]
   ...(during this long s_f, many rows buffer — DOF-3 coupling)
4  SCATTER → POLL_WAIT → T4 (large buffered batch) → DRAINING → ... → FORWARD
5  FORWARD    —T10 (s_f = s_steady, WARM, same executable)→ SCATTER    [L177]
```
**Stability: the cold tail is TRANSIENT (fires at most once — one padded shape, one compile); the warm
regime that follows is self-reinforcing.** This is precisely the confound `warmup` (L389-426) removes:
with `warmup([...])` called, step 3's S5 is paid before the loop and step 5's regime holds from the first
served request. Without it, the first generation's throughput is depressed and (design's measure-honesty
concern, warmup doc L398-400) must not be read as steady state.

### E4 — Between-batch version reload (RedisParamsSource). Exercises DOF-6.
```
1  ...→ RELOAD_CHECK —poll(): version_supplier advanced→ read_weights (redis read, takes time) [L283-285]
2  RELOAD_CHECK —T9 (new params,y_mean,y_std swapped)→ FORWARD (on net v+1)                      [L286-288,L385]
3  FORWARD → SCATTER → POLL_WAIT
4  next RELOAD_CHECK —poll(): want==loaded→ None→ current() (net v+1)                            [L281-282]
```
**Stability: transient per reload (one version step), then self-reinforcing on the new version** until
the supplier advances again. G3 holds: the reload is strictly between batches, so no batch straddles two
versions.

### E5 — Malformed frame interleaved with good ones (RELY-violation probe). Exercises DOF-5.
```
1  POLL_WAIT —T4→ DRAINING
2  DRAINING —T5 (good frame A, decode ok, appended)→ DRAINING            [L361]
3  DRAINING —T6 (frame X: decode_request raises WireError)→ DRAINING      [L358-360]  # _reject logs, continue
4  DRAINING —T5 (good frame B)→ DRAINING                                  [L361]
5  DRAINING —T7 (Again)→ RELOAD_CHECK → FORWARD over {A,B} (X excluded)   [L177]
6  SCATTER: replies to A and B only; X's corr-id gets NO reply           [L387]
   → X's peer recv_batch times out (RCVTIMEO) → loud abort (R6)          [peer L217-220,L324]
```
**Stability: transient (one bad frame), and under the RELY (R1) it never occurs — the peer's
`encode_request` cannot emit a malformed v2 frame.** Included to show the model admits exactly what the
code admits (loud drop, batch unaffected) and no more (no zero-fill).

### E6 — Clean shutdown. Exercises DOF-9, T3/T13.
```
1  POLL_WAIT —T2 (idle)→ POLL_WAIT          [L342]
   (another thread calls stop(): _stop=True)  [L445]
2  POLL_WAIT —(poll window elapses, ≤100ms)→ loop head sees _stop → T3→ STOPPED  [L339,L344]
3  serve_forever loop head: not _stop is False → exits                            [L436]
   later: close() — socket closed, ctx term                                       [L447-456]
```
**Stability: terminal (self-reinforcing trivially — STOPPED is absorbing).** Shutdown latency ≤
`_POLL_INTERVAL_MS` (100 ms); the socket is never closed under a polling thread (G6).

### E7 — `send_multipart` stall under RELY violation. Exercises DOF-7.
```
1  ...→ FORWARD → SCATTER
2  SCATTER —send to peer P whose RCVHWM-bound outbound pipe is full (P stopped recving, R3 violated)→ BLOCK [L387]
   (SNDTIMEO=-1, ROUTER_MANDATORY off but identity is KNOWN → block, not drop) [§0.1]
   server parked in SCATTER indefinitely; other peers starve (single thread)
```
**Stability: self-reinforcing wedge (absorbing under the violation) — the server cannot make progress
until P resumes recving.** This state is code-reachable but RELY-excluded (R3). Its presence in the model
is the fidelity requirement: the code DOES permit it (no SNDTIMEO, no ROUTER_MANDATORY), so a faithful
model must not forbid it; it is gated precisely on the RELY clause it violates.

---

## 9. DOF-CONTROL notes (what design constraint removes each latitude, and what becomes unrepresentable)

- **DOF-1 (emission interval):** *Constraint:* a fixed-rate / barrier-synchronized emitter (every peer
  emits on a global clock tick). *Removes:* E1's idle POLL_WAIT loops and the B-tracks-demand coupling;
  every drain would see a fixed B. Executions with bursty/variable B (E1↔E2 transitions) become
  unrepresentable.

- **DOF-2 (queue depth / B):** *Constraint:* a fixed-B barrier drain (block until exactly B rows queued,
  the design's "deterministic drain", §4 L157-159). *Removes:* the self-clocking; B≠B_fixed drains
  become unrepresentable, including the B=1 low-latency regime (E1) and partial-cap drains.

- **DOF-3 (service duration):** *Constraint:* a real-time deadline scheduler pinning `s_f` to a constant
  (impossible on a shared VM, but as a model constraint). *Removes:* the cold-tail (E3) and the
  service-time↔next-batch coupling; slow-forward-then-big-batch executions become unrepresentable.

- **DOF-4 (coalescing):** *Constraint:* a per-peer dedicated forward (no cross-peer mixing) or a
  single-peer deployment. *Removes:* batch-composition nondeterminism (design §4); the run-to-run f32
  variation of a given leaf's value becomes unrepresentable (values become per-leaf-deterministic, at the
  throughput cost the design notes, §4 L157-159).

- **DOF-5 (reject/accept):** *Constraint:* a typed error-reply frame instead of a silent drop (the
  protocol "carries no error frame", L369). *Removes:* the silent-drop+peer-timeout path (E5 tail);
  the peer would get an explicit reject, so the "no reply → RCVTIMEO" execution becomes unrepresentable.

- **DOF-6 (reload):** *Constraint:* immutable weights (StaticParamsSource only, `poll()`≡None, L249).
  *Removes:* E4 entirely; version-straddling-across-batches is *already* impossible (G3), so only the
  inter-batch version step disappears.

- **DOF-7 (send stall):** *Constraint:* set `ZMQ_SNDTIMEO` to a finite value + handle the EAGAIN (or
  `ZMQ_ROUTER_MANDATORY=1` to turn unroutable into an error). *Removes:* E7's indefinite wedge; the
  server would instead error/skip on a stuck peer, so the absorbing-wedge execution becomes
  unrepresentable (replaced by a loud-fail or drop). **This is the one DOF whose control would change a
  real failure mode**, and the code today deliberately leaves it (relying on R3) — recorded honestly.

- **DOF-8 (warmup set):** *Constraint:* always call `warmup([max_batch])` at standup. *Removes:* the
  cold first forward (E3); every loop forward is steady-state, so the cold-tail execution is
  unrepresentable.

- **DOF-9 (poll granularity):** *Constraint:* an interruptible/`zmq_socket`-monitor shutdown signal
  instead of a polled flag. *Removes:* the ≤100 ms shutdown latency and the idle wakeup cadence; the
  bounded-poll-self-loop (T2) executions disappear.

---

## 10. FIDELITY SELF-AUDIT

### Possible over-permissions (does the model admit executions the code cannot produce?)
- **Concurrent forwards.** Excluded by S3 (single thread, L33-35). The model does not allow two FORWARD
  intervals to overlap. ✓ not over-permitted.
- **A forward without a preceding non-empty drain.** Excluded: `_serve_batch` runs only if
  `drained` is non-empty (L438 `and drained`), and `run_microbatch` refuses an empty batch (L152-153). ✓
- **A reply preceding its forward.** Excluded by S2 + L387-after-L177. ✓
- **B exceeding `max_batch`.** The cap is on `total_rows` BEFORE appending (L348 checks at loop top), so a
  single request with `B_i` rows can push `total_rows` past `max_batch` *by one request's overshoot*
  (the check is `total_rows < max_batch` at L348, then a whole `X.shape[0]` is added at L362). The model
  reflects this: B can be up to `max_batch - 1 + max(B_i)`. `pad_to=max_batch` (L385) then assumes
  `pad_to ≥ B` (L171 only pads when `pad_to > B`); if a single request's `B_i` makes `B > max_batch`,
  **no padding happens** and the forward runs at the larger shape (a second compiled shape!). The model
  ADMITS this (it is what the code does); flagging it as a real, code-faithful edge — the padding's
  "only ever pads UP" claim (L169-170) holds only if every single request's `B_i ≤ max_batch`. Under the
  RELY the peer's per-message S can be up to K (fibers/thread), so if K > max_batch this fires. Faithful
  to include; would be over-constrained to forbid.
- **Envelope parsing.** The model treats the envelope opaquely (echo only), matching L354/L387 (never
  parsed). It does NOT assume a corr-id is present — a bare DEALER with `envelope==[]` is admitted
  (L128-130). ✓ not over-constrained there.

### Possible over-constraints (does the model forbid executions the code can produce?)
- **`send_multipart` partial-stall.** I modeled send as either immediate (drop to unknown identity, or
  enqueue) or a full block (HWM). libzmq's ROUTER send of a multipart message is atomic at the API level;
  a partial-frame stall is not exposed. So modeling it as block-or-complete is faithful, not
  over-constrained. ✓
- **Reject path B-accounting.** I assert `total_rows` is unchanged on reject (L360 `continue` before
  L362). Verified: the `continue` skips the `total_rows += X.shape[0]`. ✓ not over-constrained.
- **The 100 ms poll as the only idle wait.** I did not collapse the poll to instantaneous; T2 self-loops
  are explicitly admitted (E1, E6). ✓
- **Service time near-constancy (S4).** I deliberately did NOT pin `s_f` to a constant despite the padding
  making the shape constant — I kept it a bounded nondeterministic duration around a shape-determined
  center (host jitter on the shared VM). Pinning it would have been over-constrained; the code does not
  guarantee a constant wall-clock. ✓ This is the timing clause most at risk of over-constraint and I
  resolved it toward latitude.

### The one judgment call, stated openly
The deepest faithfulness question was §5.2: the code pads to a fixed shape (L171-172), which *erases*
batch-size dependence from the forward. The task asks to model "any dependence on batch size and any
effect of fixed-shape/padded compilation." The faithful answer derived from the code is: **the padding
makes service time independent of live B (S4), so the only B-dependence is the host-side marshalling
(concat/asarray, L164/L177) which is `O(max_batch·in_dim)` regardless of B.** I did not invent a B-scaling
service law (that would forbid the padded executions the code runs), and I did not collapse service time
to a constant (host jitter is real). This is the single most load-bearing derivation and it is grounded
entirely in L168-172 + the warmup docstring's compile-once claim (L100-104).

---

## 11. Code-derivation attestation

Every state, transition, guard, free choice, timing constraint, RELY, GUARANTEE, and DOF above is
derived purely from the cited code: `inference_server.py` (the loop, the drain, the scatter, the socket
options absent, the pad, the warmup, the poll interval), `forward.py` (the single row-independent forward
graph → service-time shape-invariance), `inference_wire.py` + `wire_spec.py` (the codec boundary → the
reject/accept transition and the value-only n_actions=0 case), `wire_leaf_pool.hpp` and
`runner_wire_batched.cpp` (the peer's socket options, the corr-id envelope, the submit/recv pairing, the
D-bounded in-flight cap → every RELY clause), and `docs/design/zmq-inference-service.md` (the design
intent, used only to confirm — never to source — the operational claims, e.g. §4's explicit
batch-composition-nondeterminism record). The libzmq 4.3.5 defaults (SNDHWM/RCVHWM=1000, ROUTER drop-vs-
block, SNDTIMEO=-1) are the documented semantics of the version present, applied to the *absence* of any
`setsockopt` in the server code. No outside expectation of intended behavior was introduced; where the
code leaves latitude (DOF-1..9) the model leaves exactly that latitude, and where the code determines a
choice (single forward per batch, scatter-after-forward, pad-to-fixed-shape) the model determines it.

---

## 12. Bounded confirmation (Z3) — theory-confirming only

A minimal Z3 encoding of the §5.3 causal skeleton for a 2-peer, 2-round execution confirms a
representative interleaving (E2-like: a forward whose service window buffers the next batch) is
admissible under the inequalities, and that the over-permission guards (no overlapping forwards;
reply-after-forward; emission-after-reply) are simultaneously satisfiable. See
`leaf-eval-server-timing-check.py` in this directory.

**Result (one run, `nice -n 19 timeout 90`, z3 4.16.0):**
- The representative E2-like execution is **SAT**. A witness: `f1=[0,2]`, `f2=[2,3]` (non-overlapping,
  S3); peer P2's request `r2` is emitted at `e_r2=1` — *inside* `f1`'s service window `[0,2]` — buffered,
  and drained into `f2` (`a_r2=2 ≤ f2_start=2`). This is exactly the §3 buffering law: rows arriving
  during a forward are served by the next forward. Replies follow their forwards (`send_r1=3 > f1_end=2`).
- The deliberately-impossible ordering (a reply preceding the forward that produced it, contradicting S2)
  is **UNSAT** — confirming the model is not vacuously permissive: it forbids the causally-impossible
  interleaving while admitting the real one.

This confirms the §5.3 causal skeleton is consistent and that its two key faithfulness guards (S2
reply-after-forward, S3 non-overlap) are simultaneously active. It is a confirmation of the derivation,
not its source.
