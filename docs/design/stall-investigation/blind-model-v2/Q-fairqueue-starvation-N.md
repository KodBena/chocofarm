# Focused derivation — Can ROUTER fair-queue starvation interact with N?

**Question (reconciliation open-Q G-5).** At large N each producer thread's single message is fatter
(more rows), so a server `_drain` may hit the `max_batch` cap after FEWER peers' messages, *deferring*
the rest to the next drain. Under sustained saturation as N grows, could this **starve** a slow producer
thread of forward slots — i.e. defer its one queued message indefinitely?

**Answer (high confidence).** N genuinely fattens each peer's per-drain message and so genuinely reduces
**how many peers fit under the cap per drain** — at large N a single peer's message can alone reach
`max_batch`, so a drain can admit *one* peer and stop, deferring the other T−1. That much is real and
**N-monotone**. But this **deferral cannot escalate into starvation**. Three code-grounded facts close
the gap, and none of them weakens as N grows:

1. **Per-thread wire depth is ≤ 1**, not D — so a slow thread has at most ONE message queued, and cannot
   bury its own message behind a backlog.
2. **libzmq ROUTER receive is a fair-queue (round-robin over active pipes)** — a continuously-pending
   peer is reached within **one rotation (≤ T drains)** regardless of how fat the other peers' messages
   are.
3. **The server spine is strictly serial and answers every drained message in the same cycle** (no
   well-formed drained message is ever dropped), and a served peer must complete a full reply round-trip
   before it can re-enqueue — so it cannot repeatedly jump the rotation ahead of a waiting peer.

The bounded wait *does* grow with N (a deferred peer waits up to one rotation, and each drain in that
rotation now runs a fatter, slower forward), so **latency** for a deferred peer is N-sensitive. But the
wait is bounded by one rotation for all N: **deferral scales with N; starvation does not occur at any N.**

---

## 0. Scope and method

Derived purely from the cleanroom. Files read end-to-end:
`chocofarm/az/inference_server.py`, `chocofarm/az/inference_wire.py`,
`cpp/src/runner_wire_batched.cpp`, `cpp/include/chocofarm/wire_leaf_pool.hpp`,
`cpp/include/chocofarm/runtime_config.hpp`, `cpp/include/chocofarm/runner_wire_batched.hpp`,
`cpp/include/chocofarm/wire_spec.hpp`, `cpp/include/chocofarm/inference_wire.hpp`,
`cpp/include/chocofarm/fiber_tree.hpp`, `cpp/include/chocofarm/fiber_leaf.hpp`,
`cpp/include/chocofarm/net_evaluator.hpp`, `cpp/stage_a/stage_a_server.py`, `chocofarm/config.py`,
`chocofarm/az/forward.py`. The one external fact used is the documented behavior of libzmq 4.3.5's
ROUTER receive path (fair-queue round-robin), stated as a RELY and justified below.

Confirmation: a bounded Z3 check (`z3_fairqueue_starvation_N.py`, UNSAT) plus a positive control
(adversarial scheduler, SAT) and the prior `inflight_le1_check.py` (UNSAT). Z3 is confirmation only; the
trust is in the derivation.

Parameters: N = `trees_per_thread`, T = `pool_threads`, `max_batch` (server cap, M below), D =
`max_inflight_msgs`, drain variant ∈ {production greedy, bench group, bench leaf}. Derived:
`base = ceil(pool_batch / T)` (`runtime_config.hpp:12-15`), `K = N · base` per-thread slot count
(`runner_wire_batched.cpp:286`).

---

## 1. The deferral mechanism, derived (the half of the question that is YES)

### 1.1 N fattens each peer's single per-drain message

The N>1 path is `run_episodes_wire_pipelined`. Its `issue_one` (`runner_wire_batched.cpp:434-452`)
gathers **every** ready slot across this thread's K slots into ONE flat buffer
(`gather.insert(... feats ...)`, `:439-441`) and submits it as ONE message
(`pool.submit_batch(gathered, gather, in_dim)`, `:445`). `submit_batch` (`wire_leaf_pool.hpp:76-94`)
sends a 2-frame ZMQ message: an 8-byte `corr` frame (`zmq_send(&corr, ..., ZMQ_SNDMORE)`, `:86`) and the
payload frame `[ver=2][B][in_dim][B·in_dim f32]` (`:89`, encoded by `wire/encode_request`,
`inference_wire.hpp:51-70`), where **B = gathered.size() ≤ K = N·base**.

So one producer message carries up to **K = N·base** feature rows. **B grows linearly with N.** This is
the source/sink interface fact the question rests on.

### 1.2 The server's row cap counts rows, and is checked at the loop top

`_drain` (`inference_server.py:160-186`):

```
while total_rows < self._max_batch:          # :171  (loop-TOP guard)
    frames = self._sock.recv_multipart(NOBLOCK)   # :173  (zmq.Again -> break, :174)
    ...
    X = decode_request(payload)              # :180
    drained.append((ident, envelope, X))     # :184
    total_rows += X.shape[0]                  # :185  (B rows of THIS one peer message)
```

`X.shape[0]` is B = the rows in that one peer's message (`decode_request`, `inference_wire.py:42-61`,
returns the `(B, in_dim)` matrix). The guard is checked **before** pulling the next message
(`:171`). Therefore: the drain keeps admitting whole peer messages until the *cumulative* row count
reaches `max_batch`, then stops — leaving any still-queued peer messages for the next drain.

### 1.3 N-monotone consequence: fewer peers fit per drain

Let each peer's pending message carry B rows. The number of peers a single drain admits before the cap
fires is roughly `ceil(max_batch / B)` (plus one straddling message, the over-fill of `:171`). As N
grows, B rises toward K = N·base, so:

| regime | B per peer | peers admitted before cap |
|---|---|---|
| small N (B ≪ max_batch) | small | many — often all T peers coalesce into one drain |
| large N (B ≥ max_batch) | ≥ max_batch | **one** — the first peer fills the cap; the rest defer |

So the question's premise is **correct and N-monotone**: at large N a drain can admit a single fat peer
and defer the other T−1 to the next cycle. (This is the same N-driven shift the prior server models
record as DOF-1/DOF-2 "g saturates" and the saturated regime, `model-server-transport.md:365-369`.)

The open question is whether this *deferral* can become *starvation* — a peer deferred forever.

---

## 2. Fact A — per-thread wire depth is ≤ 1 (a slow thread cannot back up its own queue)

This is the decisive structural fact and it is N-independent.

`issue_one` gathers **all** ready slots (`is_ready(s) = active && running && !submitted`,
`runner_wire_batched.cpp:427-430`) and marks each gathered slot `submitted[s]=1` (`:447`). Readiness
(`running` true after a park) is created in exactly one place: inside the `recv_batch` completion loop,
where `resume_with(c.pred)` re-runs the search until it parks again (`:466-468`,
`fiber_tree.hpp:58-62`). After an `issue_one`, **every** ready slot is now `submitted`, so a second
`issue_one` finds `gathered.empty()` and returns false (`:444`).

Hence both issue loops — the priming `while (inflight_msgs < D && issue_one())` (`:456`) and the
post-recv `while (inflight_msgs < D && !failed && issue_one())` (`:474`) — fire `issue_one` **at most
once** per pass: the first call consumes all readiness, the second is a no-op. The D-cap guard
(`inflight_msgs < D`, `:456,474`) **never binds**: `inflight_msgs` oscillates 1 → (recv, `:460`) 0 →
(issue) 1. So:

> **Per producer thread, `inflight_msgs ∈ {0,1}`, for all D and all N.**

Confirmed UNSAT by `inflight_le1_check.py` (re-run: `result: unsat`) under a strictly-more-permissive
encoding. (The prior verifier flagged this as the "inflight≤1" pivot,
`verify-server-too-constrained.md:40-54`.)

**Consequence for starvation.** A given producer thread p has at most ONE message resident in the
ROUTER's per-peer input queue at any instant. It emits the next message only *after* `recv_batch`
(`:458`) returns its reply — which requires the server to have served and scattered it (G5 below). So a
slow thread **cannot accumulate a backlog of its own messages**, and a deferred message is never buried
behind a newer one from the same thread. The competition for cap space is strictly **across** the ≤ T
distinct peers, at most one message each. N changes the *width* of each of those ≤ T messages, never
their *count* per peer.

---

## 3. Fact B — libzmq ROUTER receive is a fair-queue (RELY, justified)

### RELY R-FQ (about the peer = libzmq, checkable against documented 4.3.5 semantics)

> The server's `recv_multipart` (`inference_server.py:173`) draws from the ROUTER's connected DEALER
> peers by **fair-queuing**: libzmq maintains a round-robin pointer over the *active* input pipes (pipes
> that currently have a message); each receive delivers from the pipe at the pointer, advancing past
> inactive pipes, and advances the pointer past the delivered pipe. A pipe that continuously has a
> message pending is therefore delivered from **within one full rotation** — at most after every other
> active pipe has been delivered from once.

Justification this is the real behavior, not an assumption I am free to vary:
- The server sets **no** receive-side socket options that would change pipe scheduling. A whole-tree grep
  finds only two `setsockopt` calls and both are on the **producer** DEALER
  (`wire_leaf_pool.hpp:40-41`: `ZMQ_LINGER=0`, `ZMQ_RCVTIMEO=timeout_ms`). The server binds ROUTER
  (`inference_server.py:153-154`) and otherwise only `close(linger=0)` at shutdown (`:236`). So
  `ZMQ_RCVHWM` = default 1000, `ZMQ_SNDHWM` = default 1000, `ZMQ_ROUTER_MANDATORY` = 0, and the receive
  discipline is the stock ROUTER fair-queue.
- Fair-queuing is the defining receive discipline of ROUTER/XSUB-family sockets in libzmq (the `fq_t`
  load balancer); round-robin over active pipes is exactly its anti-starvation contract. This is the same
  fact the prior verifiers relied on to correct the "strict FIFO" mischaracterizations
  (`verify-server-too-constrained.md:32-36, 88-93, 102-110`).

What R-FQ does **not** assume: it does not assume any global arrival-time ordering, nor that the pointer
resets per drain. The pointer is socket state and **persists across `_drain` calls** — the drain that
hits the cap stops mid-rotation, and the next drain resumes the rotation from where it left off. This
persistence is what makes deferral *progress* rather than *repeat*.

### Why fatness (N) does not defeat fairness

The fair-queue rotates over **pipes**, not over **rows**. The row cap (`:171`) only decides *when a drain
stops*; it does not re-order the pipe rotation. A fat message from peer p (large N) makes the drain stop
*sooner* (after fewer pipes), but the **pointer has still advanced past every pipe it delivered from**,
and on the next drain it continues to the *next* pipe — which is precisely the deferred peer. Fatness
changes the **batching granularity**, never the **visitation order**. So the rotation guarantee — every
continuously-active pipe is reached within one rotation — is **independent of N**.

---

## 4. Fact C — serial server, every drained message answered, no re-jump

`serve_forever` (`inference_server.py:219-225`) is a single thread doing `_drain` → (if drained)
`_serve_batch` → `_drain` …. `_serve_batch` (`:192-200`; bench `stage_a_server.py:54-70`) runs the
forward and `send_multipart`s a reply for **every** drained request, zipping responses with the drained
list (`:197-200`). No well-formed drained message is ever held back or dropped (G5,
`model-server-transport.md:416-422`); the only non-answers are a malformed frame (`_reject`, `:182`,
unreachable under the peer's `encode_request`) or a scatter to a dead peer (ROUTER_MANDATORY-off drop) —
neither relevant to a live, well-formed slow thread.

**No re-jump.** A peer served in drain c gets its reply scattered (`:200`). Its producer thread is
blocked in `recv_batch` (`:458`) until that reply arrives; only then does it `resume_with` → re-park →
`issue_one` a NEW message (`:466-474`). So a just-served peer cannot deposit a fresh message and be
re-visited by the rotation *before* the rotation reaches a peer that was already waiting: the fair-queue
pointer has moved past the just-served peer, and a continuously-pending peer sits ahead of it in the
rotation. A served peer therefore cannot repeatedly cut ahead of a waiting peer; at worst it re-joins the
rotation behind the pointer and is visited next time around.

---

## 5. Composition — deferral is bounded by one rotation; starvation is impossible

Fix any producer thread p* whose message is continuously pending (the "slow producer" of the question —
slow in that its search rarely reaches a forward, but whenever it has parked it has exactly one message
queued, Fact A). Consider the sequence of server drains while p*'s message stays queued.

- By Fact A, every other peer also has ≤ 1 message queued — at most T−1 competitors.
- By Fact B (R-FQ), each drain delivers from the next active pipe in rotation and advances the pointer;
  the pointer persists across drains.
- A drain stops early only on the row cap (Fact 1) or `zmq.Again` (queue empty). In either case the
  pointer has advanced past every pipe it delivered from this drain.
- By Fact C, every peer the rotation delivers from is fully served (reply scattered) before it can
  re-enqueue, so it cannot be re-visited ahead of p*.

Therefore the rotation pointer advances monotonically toward p* across drains, visiting each of the ≤ T−1
other pipes **at most once** before reaching p*. p*'s message is drained within **one rotation ≤ T
drains**, for every N. Deferral is bounded; **no execution defers p* forever.**

### Bounded-wait magnitude (the N-sensitive part — latency, not starvation)

The *number of drains* p* waits is ≤ T (N-independent). The *wall-clock* wait is the sum of those ≤ T
drains' service times, and **each drain's forward is fatter and slower at large N**: under production /
bench-padmax the forward shape is fixed at `max_batch` (`pad_to=max_batch`, `inference_server.py:198`;
`stage_a_server.py:61-62`), so the per-drain service S is N-invariant at its ceiling; under bench-bucket
the shape steps up the buckets {64,256,512} with the real row count (`stage_a_server.py:30-37,63-64`), so
S rises with N until it saturates at the top bucket. Either way, p*'s bounded wait is ≤ T forwards each
at S(N) — an N-monotone but **bounded** latency, not starvation.

Two second-order N effects, both latency-only:
- **Over-fill (DOF-7).** Because the cap is checked at the loop top (`:171`), the straddling message can
  push a drain's row count to `max_batch + (B−1) ≤ max_batch + K − 1 = max_batch + N·base − 1`. Under
  padmax this can feed an **unwarmed shape B > max_batch** (no padding when `pad_to ≤ B`, `:58`), risking
  a one-time cold-compile spike (`jit_forward_core:22-34`) — a latency spike on p*'s wait, still finite,
  still one rotation.
- **Bench-leaf wakeup (DOF-5).** `wakeup="leaf"` runs one forward per drained MESSAGE
  (`stage_a_server.py:57`), so a fat-N drain of g messages pays g forwards — lengthening each drain in
  p*'s ≤ T-drain wait. Still bounded; the *count* of drains p* waits is unchanged.

---

## 6. The fragile assumption, named honestly

The entire no-starvation conclusion rests on **R-FQ** (ROUTER receive is round-robin fair across active
pipes). This is not in the cleanroom source — it is libzmq behavior. The cleanroom *guarantees* the
server adds nothing that would override it (no `setsockopt` on the ROUTER, §3), but the round-robin itself
is the library's contract, not chocofarm's code. If a future libzmq changed ROUTER receive to a
non-rotating discipline (e.g. drain a pipe greedily), the §5 argument breaks and N-driven starvation would
become reachable — the positive control in §7 shows starvation IS satisfiable the instant fairness is
removed. So the honest statement is: **starvation is impossible given the libzmq fair-queue, and chocofarm
does nothing to disable it; the residual risk lives entirely in that library invariant.**

A related code-level caveat: R-FQ guarantees *fairness across pipes*, not *equal throughput*. A slow p*
that parks rarely will be *served promptly whenever it parks* (≤ one rotation), but it will simply have
fewer parks per unit time — that is the search's own pacing (source timing, `fiber_tree.hpp`), not a
transport starvation. The transport never withholds a forward slot from a parked p*.

---

## 7. Z3 confirmation (confirmation only, not the source of trust)

`z3_fairqueue_starvation_N.py`: encodes T peers, each with ≤ 1 queued message (Fact A), the fair-queue
rotation with a persistent pointer (Fact B), the worst-case large-N fatness (each peer's message alone =
the cap, so each drain admits exactly one peer then stops), and an **adversarial** immediate re-enqueue
of every served peer (worst case for starvation). Goal: peer p*=0 queued at every step yet never served
over a 2T-drain horizon.

- Result: **UNSAT** — under fair-queue rotation a continuously-queued peer is drained within ≤ T drains,
  for any message fatness. Deferral bounded; starvation impossible.
- Positive control (same state model, fairness REMOVED — serve any queued peer): **SAT** — p*=0 starves.
  This proves the UNSAT is *caused by the fair-queue*, not by an over-constrained/vacuous encoding.
- Supporting: `inflight_le1_check.py` **UNSAT** (Fact A: per-thread depth ≤ 1 for all D, N).

```
T=4 M(cap in msg-units)=4 horizon=8 drains
asking: can the slow peer p*=0 be queued the whole time yet NEVER drained (starvation)?
result: unsat
... DEFERRAL is bounded; STARVATION is impossible.

adversarial (no fair-queue) starvation of p*=0: sat
```

---

## 8. Verdict

- **Does N interact with the drain's peer-count-per-cycle?** YES, monotonically: fatter messages (B up to
  K = N·base) fill the row cap (`inference_server.py:171,185`) after fewer peers, so a large-N drain
  admits as few as one peer and defers the rest. The prior left this open at N=1; the parametric answer is
  that the *deferral effect strengthens with N*.
- **Can that deferral starve a slow producer thread?** NO, at any N. Per-thread wire depth ≤ 1
  (`runner_wire_batched.cpp:427-447,456,474`, no self-backlog), libzmq ROUTER fair-queue rotation
  (RELY R-FQ; no overriding `setsockopt`, `wire_leaf_pool.hpp:40-41` are the only ones and are on the
  producer), and a serial server that answers every drained message and forces a full round-trip before
  re-enqueue (`inference_server.py:200,219-225`; `:458,466-474`) together bound any deferred peer's wait
  to **one rotation ≤ T drains** — independent of N.
- **What DOES scale with N** is the *wall-clock* of that bounded wait: each of the ≤ T drains runs a
  fatter, slower forward (bucket steps up; padmax pinned at ceiling; over-fill/leaf add finite spikes).
  So N raises a deferred peer's **latency**, never turning deferral into **starvation**.

**Confidence: high** for the no-starvation conclusion and its N-independence, conditional on R-FQ (the
libzmq ROUTER fair-queue), which is external to the cleanroom but which the cleanroom provably does not
disable. The one genuine residual risk is exactly that library invariant, named in §6.
