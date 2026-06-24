#!/usr/bin/env python3
"""
throughput-lab/server/server.py — the clean-room receive/serve loop (the SERVER) for the
producer->boundary->server throughput testbed. Re-implemented FRESH (not lifted): the ZMQ ROUTER
transport, the drain, the per-request scatter. The ONLY lifted dependency is the compute
(server/lifted/mlp_forward.py — the MLP forward + the phantom-typed jax/numpy ACL) and the wire view
(server/wire.py).

THE HARD-WON DESIGN (decouple the RECEIVER from the COMPUTE)
-----------------------------------------------------------
A serial drain->forward->scatter loop forfeits overlap: while the GIL-free XLA forward runs, the
socket is not being drained, so the producer's in-flight requests queue at the transport instead of
being gathered into the NEXT batch. The finding the parent server records: in a FREE-RUNNING producer
a DECOUPLED receiver does NOT deadlock the way a coupled one would (the producer is not blocked on the
server, so the receiver can run ahead and gather). So this server SPLITS into two threads + two
in-process queues:

  * the IO THREAD (owns the ZMQ ROUTER socket EXCLUSIVELY — a ZMQ socket is not thread-safe, so exactly
    one thread ever touches it). Its loop polls (the ROUTER + a wake pipe; see THE REPLY WAKE) with a
    bounded timeout and, each iteration:
    (a) drains EVERY immediately-available incoming request off the ROUTER, decodes+narrows each payload
        at the boundary to a wire.BoundedBatch (decode_bounded — an oversize / wrong-width / wrong-dtype
        / malformed frame is a loud per-identity reject, never zero-filled or passed downstream to
        detonate — ADR-0002), and pushes a DrainedRequest onto `req_q`;
    (b) drains EVERY ready reply off `resp_q` and send_multipart()s it back addressed to its identity
        with its opaque envelope echoed verbatim.
    It NEVER blocks on the forward, so the socket is always being drained.
  * the COMPUTE THREAD (touches NO ROUTER socket — only the queues, the lifted forward, and the WRITE
    end of the wake pipe). It blocking-gets one DrainedRequest, then GATHERS every other request
    currently queued (up to a max_batch row cap), concatenates their feature matrices into ONE
    (N_total, in_dim) matrix, PACKS that into a warmed shape (forward.pack -> a PaddedBatch; see THE
    BUCKET LADDER), runs ONE forward (server/lifted/mlp_forward.forward_batch) which returns the real
    rows with the pad sliced off, pushes each request's response onto `resp_q`, and POKES the wake pipe
    so the IO thread flushes the replies immediately. Any exception in this loop is FATAL and LOUD (see
    THE FAIL-LOUD COMPUTE BOUNDARY): it must never silently kill this thread.

THE BOUNDARY REFINEMENT (decode_bounded -> a request that is legal BY CONSTRUCTION)
----------------------------------------------------------------------------------
The drain does not stop at decode_request (which validates only a frame's INTERNAL self-consistency —
its body byte count matches its OWN header). It calls server.wire.decode_bounded(payload, max_batch,
in_dim), which ALSO enforces the SERVER's law — 1 <= rows <= max_batch, cols == in_dim, float32 —
returning a wire.BoundedBatch whose shape is legal BY CONSTRUCTION. That refinement is what keeps each
gathered request's row count <= max_batch (so no LONE request can exceed the cap and outrun the bucket
ladder) and each request's columns == in_dim (so the per-batch np.concatenate can never blow up on a
column mismatch and poison its co-batched neighbours). The law is stated ONCE, at the boundary in
wire.py (ADR-0012); the gather, the concat, and pack consume a BoundedBatch without re-checking it.

THE BUCKET LADDER (why the forward never sees a raw row count) — a TYPE, not a check
-----------------------------------------------------------------------------------
The jitted forward (server/lifted/mlp_forward) compiles ONE XLA kernel per (rows, in_dim, dtype) shape.
The gather produces an ARBITRARY row count (1 .. max_batch), so forwarding the raw count would hit a
never-compiled shape on nearly every batch and pay a full ~50ms XLA compile INSIDE the timed window —
mis-read as compute. The guard is not a check at the call site — it is a TYPE (ADR-0012): the forward,
after warmup, OWNS its warmed set; `forward.pack(X)` rounds a raw row count up to the next warmed bucket
(max_batch is always on the ladder, so every gathered batch — capped at max_batch — has a covering
bucket) and zero-pads, returning a PaddedBatch whose row count is warmed BY CONSTRUCTION. `forward_batch`
accepts ONLY a PaddedBatch, so a raw arbitrary row count cannot reach the recompiling jit — the illegal
state is unrepresentable. Because the BoundedBatch boundary caps each request at max_batch and the gather
caps the SUM at max_batch (== the top warmed bucket), pack is now TOTAL on the serving path — it is never
handed an oversize matrix, and its loud oversize reject is a backstop that can no longer fire on a real
request. Every bucket is compiled BEFORE serving (ADR-0009). This mirrors chocofarm's production
InferenceServer (a fixed padded-bucket ladder, not a per-batch shape).

THE REPLY WAKE (decouple reply latency from the poll cadence) — clause 6 of the ratified resolution
---------------------------------------------------------------------------------------------------
The IO thread sleeps in poller.poll(poll_timeout_ms). A ZMQ poller wakes on inbound ROUTER traffic but
NOT on a reply landing in resp_q — so, unaided, a reply the compute thread produces while the IO thread
is parked would idle up to poll_timeout_ms before being sent. Under a CLOSED-LOOP (coupled) producer
(which sends nothing new until it holds its reply) that idle is on the critical path and serializes the
system at the poll cadence. The fix is a WAKE PIPE: an inproc ZMQ PAIR whose READER the IO thread
registers in its poller and whose WRITER the compute thread pokes after enqueuing replies. The poke makes
poll() return immediately, so a reply is flushed the instant it is ready, at ANY poll_timeout_ms. Each
end is owned by exactly ONE thread (writer = compute, reader = IO) — the correct reading of the
single-socket rule, not a breach of it. Consequence: poll_timeout_ms NO LONGER floors coupled RTT; it
only bounds how quickly the loop notices stop() and the idle wake cadence.

THE FAIL-LOUD COMPUTE BOUNDARY (a dying thread must die NOISILY) — ratification finding
---------------------------------------------------------------------------------------
The compute thread is a daemon. An uncaught exception in its loop (a pack / forward / concatenate
failure) would SILENTLY kill it, after which the IO thread would drain forever into a req_q nobody
reads — the whole server WEDGED, with no process crash. So the compute loop wraps its work: any exception
is announced on stderr, sets the stop flag (so the IO thread tears down), and is RE-RAISED so the thread
dies with a visible traceback rather than vanishing (ADR-0002 — fail loud, never a silent wedge).

THE TEARDOWN (a true join, and no dropped request) — clauses 7 + 8 of the ratified resolution
---------------------------------------------------------------------------------------------
On stop(), the compute loop exits at its head test, but req_q may still hold drained-but-uncomputed
requests. Its `finally` DRAINS those (answers them, so no corr-id is left forever unanswered — clause 7),
then signals the IO thread via a one-shot `stats_done` queue. That signal — not a timed join — is the
real join point (clause 8): the IO thread BLOCKS on it before reading the stats, so it never iterates a
counter the compute thread is still mutating (the old `join(timeout=1.0)` proceeded regardless and raced
summary() against note_batch — "dictionary changed size during iteration"). A LOUD escape bounds a
genuinely hung compute thread: if the signal does not arrive within a few seconds, the IO thread says so
and refuses to print torn stats, rather than blocking forever or quietly lying.

THE ZMQ ENVELOPE (DEALER producer <-> ROUTER server) — see server/wire.py Layer 2 for the bytes
---------------------------------------------------------------------------------------------------
The server binds a ZMQ_ROUTER. recv_multipart() yields, per request:

    [ identity ] [ corr-id ] [ <Layer-1 request payload> ]
     frames[0]    frames[1]    frames[-1]

  * identity  = frames[0]  — the ROUTER-assigned producer connection id; the address to reply to.
  * envelope  = frames[1:-1] — OPAQUE transport-routing frames the server ECHOES BACK VERBATIM,
                NEVER parsing them. Here that is exactly the single [corr-id] frame (the producer's
                u64). Capturing it as `frames[1:-1]` (not `frames[1]`) keeps the server agnostic to
                how many envelope frames a future producer uses (ADR-0012 P7 serialization⊥transport).
  * payload   = frames[-1] — the Layer-1 request, decoded+narrowed by server.wire.decode_bounded to a
                BoundedBatch (B_i, in_dim).

The reply is send_multipart([identity, *envelope, response]) — the identity routes it back to the
producer's DEALER, the echoed envelope lets the producer match the reply to its outstanding corr-id,
and the response is server.wire.encode_response(values, logits) for THAT request's B_i rows.

FAIL LOUD (ADR-0002): a malformed / oversize / wrong-width / wrong-dtype payload is rejected for THAT
identity at the boundary (counted in the summary, never zero-filled or passed downstream); a too-short
envelope is a loud reject; a reply to a vanished identity is a counted peer-gone event while any other
ZMQ error is re-raised loud; an exception inside the compute thread is fatal-and-loud, not a silent
wedge.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import zmq

from server.lifted.mlp_forward import MlpForward, NumpyMlpForward, ProdMlpForward, StagedMlpForward
from server.wire import STAGE_A_IN_DIM, WireError, decode_bounded, encode_response


# A drained request the IO thread hands the compute thread: the identity to scatter back to, the
# opaque envelope frames to echo verbatim (here the single [corr-id]), and the decoded (B_i, in_dim)
# matrix. `recv_mono` is the steady-clock timestamp the request was drained, so the compute thread can
# attribute an in-server latency (drain -> reply-ready) — see ServerStats.note_latency.
@dataclass
class DrainedRequest:
    identity: bytes
    envelope: "list[bytes]"        # frames[1:-1] — opaque, echoed back unchanged
    X: "npt.NDArray[np.float32]"   # the VALIDATED (B_i, in_dim) matrix from decode_bounded's BoundedBatch:
                                   #   B_i <= max_batch, cols == in_dim, float32 — assumed downstream (ADR-0012)
    recv_mono: float


# A reply the compute thread hands back to the IO thread to send: the identity, the echoed envelope,
# and the already-encoded Layer-1 response frame. The compute thread does the encode (off the IO
# thread); the IO thread only does the socket send (keeping the socket single-threaded).
@dataclass
class PendingReply:
    identity: bytes
    envelope: "list[bytes]"
    frame: bytes


@dataclass
class ServerConfig:
    bind: str = "ipc:///tmp/tlab-infer.sock"   # the ZMQ ROUTER bind endpoint (match the producer's)
    max_batch: int = 4096                       # cap on N_total rows per forward (bounds the matmul shape)
    in_dim: int = STAGE_A_IN_DIM                # feature width (241 = Stage-A); the net's first weight dim
    n_actions: int = 0                          # policy width (0 = value-only); sizes the response logits
    hidden: int = 256                           # MLP hidden width (the live trunk is 256x256)
    residual: bool = False                      # add the keyed residual block (matches a residual net)
    seed: int = 0                               # RNG seed for the throwaway random weights
    warmup_batch_sizes: "list[int]" = field(default_factory=lambda: [1, 8, 64, 512, 4096])
    poll_timeout_ms: int = 50                   # IO-thread idle poll timeout: bounds stop()-detection and
                                                # the idle wake cadence. It NO LONGER floors coupled RTT —
                                                # the wake pipe flushes a reply the instant it is ready
                                                # (see THE REPLY WAKE in the module docstring).
    verbose: bool = True                        # print a ready line + a teardown stats summary
    single_thread: bool = False                 # serve on ONE thread (drain->forward->scatter inline, no
                                                # IO/compute split, no wake pipe) -- the production
                                                # InferenceServer model. The A/B arm for the two-thread-
                                                # contention hypothesis (tlab_finding #4): on a single pinned
                                                # core the decoupled receiver's IO thread steals compute
                                                # cycles; here the OS socket buffer queues the next batch
                                                # DURING the forward instead. Same forward + ladder (one home).
    forward_impl: str = "jax"                   # "jax" = MlpForward (XLA-jit + bucket ladder); "numpy" =
                                                # NumpyMlpForward (forward_core in numpy, NO XLA dispatch, NO
                                                # pad) -- the A/B arm for the XLA-per-call-overhead hypothesis.
    profile_forward: bool = False               # (single-thread only) split each forward into h2d|jit|d2h
                                                # sub-phase timers -- localizes the in-serve forward inflation.
                                                # SERIALISES the XLA pipeline (blocks between phases) so it
                                                # slows the run; a diagnostic mode, not for banked numbers.


@dataclass
class ServerStats:
    """Server-side throughput / utilization counters (ADR-0009: every claim is measured). All counts
    are cumulative over the served window; `wall_s` is the serve duration the rates are over."""
    requests_recv: int = 0          # Layer-2 messages drained off the ROUTER (one per producer submit)
    rows_recv: int = 0              # total leaf rows across all drained requests (sum of B_i)
    rejects: int = 0               # malformed/oversize/wrong-width frames rejected at the boundary (ADR-0002)
    forwards: int = 0              # number of batched forward() calls (a forward gathers >= 1 request)
    rows_forwarded: int = 0        # total rows pushed through the forward (== rows_recv minus in-flight)
    replies_sent: int = 0          # Layer-2 reply messages send_multipart()'d back
    undeliverable: int = 0         # replies whose producer identity had vanished (ROUTER_MANDATORY EHOSTUNREACH)
    dropped_replies: int = 0       # replies dropped because the producer's recv buffer was full (decoupled flood)
    drained_on_stop: int = 0       # queued requests answered during teardown (clause 7) rather than dropped
    batch_hist: "dict[int, int]" = field(default_factory=dict)   # forward batch-row-count -> occurrences
    compute_s: float = 0.0          # wall seconds spent inside forward_batch (the compute-busy time)
    prep_s: float = 0.0             # wall seconds in host-side prep (concat the gather + pack/pad) before the forward
    scatter_s: float = 0.0          # wall seconds in host-side scatter (encode_response + send) after the forward
    h2d_s: float = 0.0              # (profile_forward) wall in the forward's host->device transfer (jnp.asarray)
    jit_s: float = 0.0              # (profile_forward) wall in the jitted forward (_forward_both)
    d2h_s: float = 0.0              # (profile_forward) wall in the forward's device->host pulls (np.asarray x2)
    in_server_lat_sum_s: float = 0.0   # sum of per-request drain->reply-ready latencies (the in-server time)
    in_server_lat_max_s: float = 0.0   # worst single drain->reply-ready latency
    lat_count: int = 0              # number of replies the latency was measured over
    serve_start_mono: float = 0.0
    serve_end_mono: float = 0.0

    @property
    def wall_s(self) -> float:
        return max(self.serve_end_mono - self.serve_start_mono, 1e-9)

    def note_batch(self, n_rows: int) -> None:
        self.forwards += 1
        self.rows_forwarded += n_rows
        self.batch_hist[n_rows] = self.batch_hist.get(n_rows, 0) + 1

    def note_latency(self, lat_s: float) -> None:
        """Attribute one request's in-server latency (drain -> reply-ready). Honors the ADR-0009 promise
        the carried-but-unread recv_mono used to make without a measurement behind it (clause 9a)."""
        self.in_server_lat_sum_s += lat_s
        self.lat_count += 1
        if lat_s > self.in_server_lat_max_s:
            self.in_server_lat_max_s = lat_s

    def summary(self) -> str:
        w = self.wall_s
        mean_batch = (self.rows_forwarded / self.forwards) if self.forwards else 0.0
        util = self.compute_s / w if w > 0 else 0.0
        mean_lat_ms = (self.in_server_lat_sum_s / self.lat_count * 1e3) if self.lat_count else 0.0
        max_lat_ms = self.in_server_lat_max_s * 1e3
        nf = self.forwards or 1   # guard the per-forward divisions (0 forwards => a no-op run)
        prep_us, comp_us, scat_us = (self.prep_s / nf * 1e6, self.compute_s / nf * 1e6, self.scatter_s / nf * 1e6)
        overhead_pct = (self.prep_s + self.scatter_s) / w * 100 if w > 0 else 0.0
        fwd_split = (f"\n              forward split (us/fwd, SERIALISED — relative only): "
                     f"h2d {self.h2d_s / nf * 1e6:.0f} | jit {self.jit_s / nf * 1e6:.0f} | "
                     f"d2h {self.d2h_s / nf * 1e6:.0f}") if self.h2d_s > 0 else ""
        return (
            f"[tlab-server] served {self.requests_recv} requests / {self.rows_recv} rows in {w:.3f}s\n"
            f"              throughput: {self.requests_recv / w:,.0f} req/s  |  {self.rows_recv / w:,.0f} rows/s\n"
            f"              forwards: {self.forwards}  (mean batch {mean_batch:.1f} rows, "
            f"max {max(self.batch_hist) if self.batch_hist else 0})\n"
            f"              compute-busy: {self.compute_s:.3f}s  ({util * 100:.1f}% of wall)\n"
            f"              per-forward (us): prep {prep_us:.0f} | compute {comp_us:.0f} | scatter {scat_us:.0f}  "
            f"(host serve-overhead prep+scatter = {overhead_pct:.1f}% of wall){fwd_split}\n"
            f"              in-server latency: mean {mean_lat_ms:.2f} ms  max {max_lat_ms:.2f} ms  "
            f"(drain -> reply-ready, {self.lat_count} replies)\n"
            f"              reply ledger: sent {self.replies_sent}  |  rejects {self.rejects}  |  "
            f"undeliverable {self.undeliverable}  |  dropped {self.dropped_replies}  |  "
            f"drained-on-stop {self.drained_on_stop}\n"
            f"              batch-size histogram (rows->count): "
            f"{dict(sorted(self.batch_hist.items()))}"
        )


class ThroughputServer:
    """The clean-room receive/serve loop. Holds a ZMQ ROUTER bound to cfg.bind, an MlpForward (the
    lifted compute) built from a random net of the configured shapes, the IO/compute split, and the
    inproc reply-wake pipe. The compute is warmed up (every reachable batch size compiled) BEFORE
    serve_forever() handles a single real request (ADR-0009)."""

    # A loud escape if the compute thread hangs at teardown: rather than block the IO thread forever on
    # the done-signal (or proceed and print torn stats), say so after this many seconds and stop.
    _TEARDOWN_TIMEOUT_S: float = 5.0

    # Max requests drained from the ROUTER per IO pass. Bounds the time the IO thread spends draining
    # before it returns to _io_step to flush replies and recheck stop(). An UNBOUNDED drain (while True
    # until EAGAIN) MONOPOLISES the IO thread under a sustained flood: on a shared core it starves the
    # compute thread (0 forwards, 0% util) and never notices stop() — the server then wedges and is
    # SIGKILLed at teardown (observed: a rows=1 producer flood killed 11/18 server-pinned cells). Capping
    # the drain time-slices IO against compute and keeps stop() responsive (ADR-0002 — never wedge).
    _MAX_DRAIN_PER_PASS: int = 1024

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self.stats = ServerStats()

        # -- transport: bind the ROUTER (the IO thread is the ONLY toucher of this socket) -----------
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.ROUTER)
        # ROUTER drops a reply to an unknown/departed peer silently by default; in the lab we WANT a
        # loud failure if we ever address a vanished identity, so set ROUTER_MANDATORY (a send to an
        # unroutable identity then raises EHOSTUNREACH rather than silently dropping — ADR-0002).
        self._sock.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._sock.bind(cfg.bind)

        # -- the reply WAKE pipe (clause 6): an inproc PAIR. The IO thread polls the READER; the compute
        # thread pokes the WRITER after enqueuing replies, so a reply flushes the instant it is ready
        # rather than idling up to poll_timeout_ms (see THE REPLY WAKE). Bind the reader HERE, on the
        # IO/main thread that owns it (inproc needs bind-before-connect); the compute thread creates and
        # connects its writer when it starts, so each end is touched by exactly one thread. The endpoint
        # is per-instance (id) so two servers in one process do not collide on it.
        self._wake_endpoint = f"inproc://tlab-wake-{id(self):x}"
        self._wake_r = self._ctx.socket(zmq.PAIR)
        self._wake_r.bind(self._wake_endpoint)

        # -- compute: a random net of the live geometry (throughput is a property of the SHAPES) -----
        # forward_impl selects the backend: "jax" = XLA-jit (the bucket-ladder forward), "numpy" = the same
        # forward_core in numpy (no XLA dispatch, no pad). "prod"/"staged" are DIAGNOSTIC cross-boundary arms
        # (the REAL production jit_forward_core / build_staged_forward — the apples-to-apples attribution of
        # the tlab/overcommit forward-envelope gap; never shipped). All share one param builder (one home).
        _FORWARDS = {"jax": MlpForward, "numpy": NumpyMlpForward,
                     "prod": ProdMlpForward, "staged": StagedMlpForward}
        if cfg.forward_impl not in _FORWARDS:
            raise ValueError(f"forward_impl must be one of {sorted(_FORWARDS)}, got {cfg.forward_impl!r}")
        _forward_cls = _FORWARDS[cfg.forward_impl]
        self._forward = _forward_cls.random_net(
            in_dim=cfg.in_dim, hidden=cfg.hidden, n_actions=cfg.n_actions,
            residual=cfg.residual, seed=cfg.seed,
        )

        # -- the two-thread split: req_q (IO -> compute), resp_q (compute -> IO); stats_done is the
        # teardown done-signal (clause 8: the real join, replacing the timed-join fiction) ------------
        self._req_q: "queue.Queue[DrainedRequest]" = queue.Queue()
        self._resp_q: "queue.Queue[PendingReply]" = queue.Queue()
        self._stats_done: "queue.Queue[bool]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._compute_thread = threading.Thread(target=self._compute_loop, name="tlab-compute", daemon=True)

        # The BUCKET LADDER (see the module docstring). The server's only job here is POLICY: choose
        # which batch sizes to warm and ensure max_batch is among them (so every gathered batch — capped
        # at max_batch rows by the gather — has a covering bucket; a warmup beyond the cap compiles a
        # kernel that can never be hit, hence min(b, max_batch)). ENFORCEMENT lives in the forward: after
        # warmup it owns the warmed set, `pack` rounds a raw row count into it, and the typed PaddedBatch
        # makes an unwarmed forward shape unrepresentable (server/lifted/mlp_forward, ADR-0012). Every
        # bucket is compiled BEFORE serving a single real request (ADR-0009). The server keeps NO second
        # copy of the ladder — it reads the warmed set back from the forward (one source of truth).
        warm = sorted({min(b, cfg.max_batch) for b in cfg.warmup_batch_sizes if b > 0} | {cfg.max_batch})
        self._forward.warmup(warm, cfg.in_dim)
        self._warmup_sizes = self._forward.warmed_sizes

    # -- lifecycle ------------------------------------------------------------------------------------

    def serve_forever(self) -> None:
        """Run the receive/serve loop until stop(): the compute thread starts, then THIS (the main)
        thread becomes the IO thread — it owns the socket, drains requests, and sends replies, polling
        the ROUTER + the wake pipe with a bounded timeout so stop() takes effect within poll_timeout_ms."""
        self.stats.serve_start_mono = time.monotonic()
        if self.cfg.single_thread:
            self._serve_forever_single()   # its own READY + loop + teardown; no compute thread / wake pipe
            return
        self._compute_thread.start()
        if self.cfg.verbose:
            # The harness waits on this READY line before launching the producer (do NOT start the
            # producer before warmup or the first batches pay XLA compile — ADR-0009). Flush so the
            # line is visible immediately to a line-buffered reader (a subprocess pipe).
            print(f"[tlab-server] READY bind={self.cfg.bind} in_dim={self.cfg.in_dim} "
                  f"hidden={self.cfg.hidden} n_actions={self.cfg.n_actions} "
                  f"max_batch={self.cfg.max_batch} warmup={self._warmup_sizes}", flush=True)
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        poller.register(self._wake_r, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                self._io_step(poller)
        finally:
            self._teardown()

    def _serve_forever_single(self) -> None:
        """SINGLE-THREADED serve loop (cfg.single_thread) — the production InferenceServer model: ONE thread
        does drain -> forward -> scatter, with NO IO/compute split and NO wake pipe. The A/B arm for the
        two-thread-contention hypothesis (tlab_finding #4): on a single pinned core the decoupled receiver's
        IO thread steals the compute thread's cycles (inflating the in-server forward to ~1.75ms vs ~1.16ms
        isolated); here the OS socket buffer queues the next batch DURING the forward instead — exactly what
        the production server relies on. The forward + the bucket ladder are IDENTICAL to the two-thread path
        (one home: self._forward); only the IO plumbing differs. Decode/reject (decode_bounded), the cap, and
        the per-request counters mirror the two-thread path so the two arms are comparable."""
        cap = self.cfg.max_batch
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        if self.cfg.verbose:
            print(f"[tlab-server] READY (single-thread) bind={self.cfg.bind} in_dim={self.cfg.in_dim} "
                  f"hidden={self.cfg.hidden} n_actions={self.cfg.n_actions} "
                  f"max_batch={self.cfg.max_batch} warmup={self._warmup_sizes}", flush=True)
        # A decoded request deferred because adding it would exceed the cap — it heads the NEXT forward (the
        # single-thread analog of _gather putting the overshoot request back on req_q; a destructive recv
        # can't un-read, so we carry it). Counted in rows_recv when first drained, processed once.
        pending: "tuple[bytes, list[bytes], npt.NDArray[np.float32], float] | None" = None
        try:
            while not self._stop.is_set():
                # poll(0) when work is already held; else the idle timeout so stop() is noticed promptly.
                socks = dict(poller.poll(0 if pending is not None else self.cfg.poll_timeout_ms))
                if pending is None and socks.get(self._sock) != zmq.POLLIN:
                    continue
                now = time.monotonic()
                batch: "list[tuple[bytes, list[bytes], npt.NDArray[np.float32], float]]" = []
                n_rows = 0
                if pending is not None:
                    batch.append(pending)
                    n_rows += int(pending[2].shape[0])
                    pending = None
                while n_rows < cap:
                    try:
                        frames = self._sock.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if len(frames) < 2:
                        self.stats.rejects += 1
                        continue
                    identity, envelope, payload = frames[0], frames[1:-1], frames[-1]
                    try:
                        bounded = decode_bounded(payload, max_batch=self.cfg.max_batch, in_dim=self.cfg.in_dim)
                    except WireError:
                        self.stats.rejects += 1
                        continue
                    rows = int(bounded.X.shape[0])
                    self.stats.requests_recv += 1
                    self.stats.rows_recv += rows
                    if n_rows + rows > cap and n_rows > 0:
                        pending = (identity, envelope, bounded.X, now)   # defer; heads the next forward
                        break
                    batch.append((identity, envelope, bounded.X, now))
                    n_rows += rows
                if not batch:
                    continue
                tp = time.monotonic()
                X = batch[0][2] if len(batch) == 1 else np.concatenate([b[2] for b in batch], axis=0)
                packed = self._forward.pack(X)
                t0 = time.monotonic()
                self.stats.prep_s += t0 - tp           # host-side concat + pad (before the forward)
                if self.cfg.profile_forward:
                    values, logits, (h2d, jit_t, d2h) = self._forward.forward_batch_timed(packed)
                    self.stats.h2d_s += h2d; self.stats.jit_s += jit_t; self.stats.d2h_s += d2h
                else:
                    values, logits = self._forward.forward_batch(packed)
                done = time.monotonic()
                self.stats.compute_s += done - t0
                self.stats.note_batch(n_rows)
                off = 0
                for identity, envelope, Xi, recv_mono in batch:
                    b_i = int(Xi.shape[0])
                    v_slice = values[off:off + b_i]
                    l_slice = logits[off:off + b_i] if logits is not None else None
                    off += b_i
                    frame = encode_response(v_slice, l_slice)
                    try:
                        self._sock.send_multipart([identity, *envelope, frame], flags=zmq.NOBLOCK)
                        self.stats.replies_sent += 1
                    except zmq.Again:
                        self.stats.dropped_replies += 1          # producer recv buffer full — bounded, never wedge
                    except zmq.ZMQError as e:
                        if e.errno == zmq.EHOSTUNREACH:
                            self.stats.undeliverable += 1
                        else:
                            raise
                    self.stats.note_latency(done - recv_mono)
                self.stats.scatter_s += time.monotonic() - done   # host-side encode + send (after the forward)
        finally:
            self.stats.serve_end_mono = time.monotonic()
            self._wake_r.close(linger=0)   # bound in __init__ though unused in single-thread mode
            self._sock.close(linger=0)
            if self.cfg.verbose:
                print(self.stats.summary(), file=sys.stderr, flush=True)

    def stop(self) -> None:
        """Signal the receive/serve loop to stop. Sets the stop flag; the IO thread notices within
        poll_timeout_ms (or sooner on a wake) and exits its loop, then tears down — blocking on the
        compute thread's done-signal (a true join, not a timed race) before reading stats and closing
        the sockets on the IO thread (ADR-0002; clauses 7+8 of the ratified resolution)."""
        self._stop.set()

    def _teardown(self) -> None:
        """Teardown on the IO thread (clauses 7+8). Block on the compute thread's done-signal — the REAL
        join, not a timed fiction — so the stats are quiescent before we read them; a loud escape bounds a
        hung compute thread. Then flush the straggler replies the compute thread queued while draining its
        leftover requests, close BOTH sockets on the thread that bound them, and print the summary."""
        finished = True
        try:
            self._stats_done.get(timeout=self._TEARDOWN_TIMEOUT_S)
        except queue.Empty:
            finished = False
            print(f"[tlab-server] WARNING: compute thread did not finish within "
                  f"{self._TEARDOWN_TIMEOUT_S:.0f}s — stats may be torn; refusing to print a summary "
                  f"(ADR-0002: a loud gap, not a quiet lie)", file=sys.stderr, flush=True)
        if finished:
            self._compute_thread.join()   # returns at once — the signal already means it is done
            self._flush_replies()         # send the replies the compute thread queued during its drain
        self.stats.serve_end_mono = time.monotonic()
        self._wake_r.close(linger=0)
        self._sock.close(linger=0)
        if self.cfg.verbose and finished:
            print(self.stats.summary(), file=sys.stderr, flush=True)

    # -- the IO thread (owns the socket) -------------------------------------------------------------

    def _io_step(self, poller: "zmq.Poller") -> None:
        """One IO iteration: poll the ROUTER + the wake pipe (bounded timeout), drain whatever is ready,
        then flush every ready reply. The wake pipe makes poll() return the instant the compute thread
        has a reply, so a reply is never held for the poll timeout (THE REPLY WAKE)."""
        socks = dict(poller.poll(self.cfg.poll_timeout_ms))
        if socks.get(self._wake_r) == zmq.POLLIN:
            self._drain_wake()
        if socks.get(self._sock) == zmq.POLLIN:
            self._drain_socket()
        # Flush AFTER the poll so a reply produced during the parked poll (which poked the wake) goes out
        # THIS iteration. A reply produced after this flush re-pokes the wake -> the next poll returns at
        # once -> it is flushed then. No reply is ever held for poll_timeout_ms.
        self._flush_replies()

    def _drain_wake(self) -> None:
        """Consume the compute thread's wake pokes (coalesced — N pokes mean 'replies are ready', and one
        _flush_replies sends them all). Non-blocking drain so a burst of pokes does not keep re-waking."""
        while True:
            try:
                self._wake_r.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return

    def _drain_socket(self) -> None:
        """Drain UP TO _MAX_DRAIN_PER_PASS requests from the ROUTER (non-blocking) into req_q, then return
        so the IO loop flushes replies and rechecks stop(). If more is pending at the cap, the next poll()
        returns immediately (the socket is still readable), so nothing is lost — the drain is time-sliced,
        not truncated. An UNBOUNDED drain monopolises the IO thread under a flood, starving compute on a
        shared core and wedging stop() (see _MAX_DRAIN_PER_PASS). Each payload is decoded AND narrowed to
        the server's geometry at the boundary (decode_bounded): an oversize / wrong-width / wrong-dtype /
        malformed frame is a loud per-identity reject (counted, surfaced in summary), never passed downstream to
        detonate at the forward, the ladder, or inside np.concatenate (ADR-0002)."""
        now = time.monotonic()
        drained = 0
        while drained < self._MAX_DRAIN_PER_PASS:
            try:
                frames = self._sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            drained += 1
            if len(frames) < 2:
                # A ROUTER message is at minimum [identity][payload]; fewer is a framing violation. The
                # `rejects` counter is the SSOT (surfaced once in summary()); NO per-event flush'd print on
                # the hot path — an unbounded log can block the main thread on a full pipe (see _flush_replies).
                self.stats.rejects += 1
                continue
            identity = frames[0]
            envelope = frames[1:-1]      # opaque echoed frames (here the single [corr-id])
            payload = frames[-1]
            try:
                bounded = decode_bounded(payload, max_batch=self.cfg.max_batch, in_dim=self.cfg.in_dim)
            except WireError:
                # ADR-0002: reject THIS identity's request; do not poison the batch. decode_bounded folds
                # oversize / wrong-width / wrong-dtype rejects in with the malformed-frame ones, so they
                # cannot reach the forward, the ladder, or the concat. Counted (`rejects` is the SSOT,
                # surfaced in summary()); NO per-event flush'd print on the hot path (it can block the main
                # thread on a full pipe and wedge teardown — see _flush_replies).
                self.stats.rejects += 1
                continue
            self.stats.requests_recv += 1
            self.stats.rows_recv += int(bounded.X.shape[0])
            self._req_q.put(DrainedRequest(identity=identity, envelope=envelope,
                                           X=bounded.X, recv_mono=now))

    def _flush_replies(self) -> None:
        """Send every reply currently ready on resp_q (non-blocking get). The send blocks only if the OS
        socket buffer is full — acceptable back-pressure; the producer is reading replies. A vanished
        producer identity (ROUTER_MANDATORY -> EHOSTUNREACH) is a counted peer-gone reply (a
        torn-down producer at end-of-run); ANY OTHER ZMQError is NOT benign and is re-raised LOUD —
        ETERM/EFSM must never be silently reclassified as peer-gone (ADR-0002; clause 9b/c)."""
        while True:
            try:
                rep = self._resp_q.get_nowait()
            except queue.Empty:
                return
            try:
                self._sock.send_multipart([rep.identity, *rep.envelope, rep.frame], flags=zmq.NOBLOCK)
                self.stats.replies_sent += 1
            except zmq.Again:
                # The producer's recv buffer is full — it is not reading its replies fast enough (common in
                # a DECOUPLED flood, where it free-runs sending and abandons most replies). Do NOT block: a
                # blocking reply send wedges the IO thread (it can no longer flush, drain, or notice stop()
                # -> the server is SIGKILLed at teardown, losing its stats). DROP the reply and count it; the
                # producer's own recv count is the honest served number (ADR-0002 — bounded, never wedge).
                # NB: zmq.Again is a zmq.ZMQError subclass, so this arm must precede the general one below.
                self.stats.dropped_replies += 1
            except zmq.ZMQError as e:
                if e.errno == zmq.EHOSTUNREACH:
                    # A vanished producer identity (a torn-down producer at end-of-run). COUNT it — the
                    # per-event print is deliberately REMOVED: under a flood it emitted thousands of
                    # flush'd writes to stderr, and once the harness's (undrained) pipe filled at 64 KB the
                    # blocking write held the GIL on the main thread, so the SIGINT handler (stop()) never
                    # ran and the server wedged at teardown. `undeliverable` is the SSOT, surfaced once in
                    # summary() (ADR-0000: the serve loop performs NO unbounded, downstream-gated blocking
                    # write — a per-event flush'd log is exactly that).
                    self.stats.undeliverable += 1
                else:
                    raise

    # -- the compute thread (touches NO ROUTER socket) -----------------------------------------------

    def _compute_loop(self) -> None:
        """Gather -> forward -> scatter, on a thread that touches NO ROUTER socket. Blocking-get one
        request (so an idle server does not spin), gather every other queued request up to the row cap,
        run ONE forward over the concatenation, scatter per-request replies onto resp_q, and poke the
        wake pipe. Owns the WRITE end of the wake pipe (created+connected HERE so exactly one thread
        touches it).

        Two disciplines live in this method's structure (ratified resolution): every exception is FATAL
        and LOUD (the except — a silent death would wedge the server), and the `finally` answers any
        drained-but-uncomputed request and hands a done-signal to the IO thread (the true join)."""
        wake_w = self._ctx.socket(zmq.PAIR)
        wake_w.connect(self._wake_endpoint)
        cap = self.cfg.max_batch
        try:
            while not self._stop.is_set():
                try:
                    first = self._req_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                batch, n_rows = self._gather(first, cap)
                self._run_forward(batch, n_rows, wake_w)
        except BaseException as e:
            # FAIL LOUD (ADR-0002 + ratification finding): an uncaught exception here would silently kill
            # this daemon thread, after which the IO thread drains forever into a queue nobody reads — the
            # server WEDGED with no crash. Announce it, signal stop so the IO thread tears down, and
            # RE-RAISE so the thread dies with a visible traceback instead of vanishing.
            print(f"[tlab-server] FATAL in compute thread: {type(e).__name__}: {e} — aborting server",
                  file=sys.stderr, flush=True)
            self._stop.set()
            raise
        finally:
            # Teardown (clauses 7+8): answer every request already drained but not yet computed (so no
            # corr-id is left forever unanswered), close the wake writer, and signal the IO thread we are
            # TRULY done. The inner try/finally guarantees the signal is sent even if the drain raises, so
            # the IO thread never blocks forever waiting for a join that will not come.
            try:
                self._drain_remaining(wake_w)
            finally:
                wake_w.close(linger=0)
                self._stats_done.put(True)

    def _gather(self, first: "DrainedRequest", cap: int) -> "tuple[list[DrainedRequest], int]":
        """Greedily gather `first` plus every other currently-queued request, up to the max_batch row cap
        (a request is admitted WHOLE — never split across two forwards). Returns (batch, n_rows). Because
        each request is a decode_bounded'd BoundedBatch with rows <= max_batch, n_rows starts <= cap, so
        the cap is a true bound on the gathered total — no lone oversize request can blow past it."""
        batch = [first]
        n_rows = int(first.X.shape[0])
        while n_rows < cap:
            try:
                nxt = self._req_q.get_nowait()
            except queue.Empty:
                break
            rows = int(nxt.X.shape[0])
            if n_rows + rows > cap and n_rows > 0:
                # Would exceed the cap: put it back (it heads the next forward) and stop. queue.Queue has
                # no push-front, so this re-queues at the tail — an acceptable reorder in a throughput lab
                # where requests are interchangeable. (n_rows > 0 always holds now: the boundary rejects a
                # 0-row request, so every gathered request has >= 1 row — the guard is defensive.)
                self._req_q.put(nxt)
                break
            batch.append(nxt)
            n_rows += rows
        return batch, n_rows

    def _run_forward(self, batch: "list[DrainedRequest]", n_rows: int, wake_w: "zmq.Socket") -> None:
        """Concatenate the batch's (validated) feature matrices, run ONE forward, scatter per-request
        replies onto resp_q, record each request's in-server latency, and poke the wake pipe so the IO
        thread flushes immediately. Every X is a decode_bounded'd matrix (cols == in_dim), so the concat
        can never blow up on a column mismatch (clause 4)."""
        if len(batch) == 1:
            X = batch[0].X
        else:
            X = np.concatenate([r.X for r in batch], axis=0)
        # Narrow the arbitrary gathered row count to a WARMED shape: pack() rounds it up to a bucket on
        # the forward's ladder and zero-pads, returning a PaddedBatch whose row count is warmed BY
        # CONSTRUCTION. forward_batch accepts only that type — so a raw row count can never reach the
        # recompiling jit (the illegal state is unrepresentable; mlp_forward / ADR-0012) — and it returns
        # the real rows with the pad tail already sliced off.
        packed = self._forward.pack(X)
        t0 = time.monotonic()
        values, logits = self._forward.forward_batch(packed)
        done = time.monotonic()
        self.stats.compute_s += done - t0
        self.stats.note_batch(n_rows)

        # Scatter: each request owns a contiguous row-slice of the concatenated result. Re-encode that
        # slice into a Layer-1 response, hand it to the IO thread, and attribute its in-server latency.
        off = 0
        for r in batch:
            b_i = int(r.X.shape[0])
            v_slice = values[off:off + b_i]
            l_slice = logits[off:off + b_i] if logits is not None else None
            off += b_i
            frame = encode_response(v_slice, l_slice)
            self._resp_q.put(PendingReply(identity=r.identity, envelope=r.envelope, frame=frame))
            self.stats.note_latency(done - r.recv_mono)
        self._wake(wake_w)

    def _drain_remaining(self, wake_w: "zmq.Socket") -> None:
        """Teardown (clause 7): on stop, the gather loop has exited but req_q may still hold
        drained-but-uncomputed requests. Answer them (so no corr-id is left forever unanswered) in
        cap-sized forwards, exactly as the steady loop would have; the replies land on resp_q and the IO
        thread flushes them post-join. A forward failure during the drain is logged, not propagated, so
        it cannot break the done-signal handoff."""
        cap = self.cfg.max_batch
        while True:
            try:
                first = self._req_q.get_nowait()
            except queue.Empty:
                return
            batch, n_rows = self._gather(first, cap)
            try:
                self._run_forward(batch, n_rows, wake_w)
                self.stats.drained_on_stop += len(batch)
            except Exception as e:
                print(f"[tlab-server] teardown: could not answer {len(batch)} queued request(s): "
                      f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
                return

    def _wake(self, wake_w: "zmq.Socket") -> None:
        """Poke the IO thread's poller so it returns NOW and flushes the replies just enqueued, instead
        of waiting out poll_timeout_ms (THE REPLY WAKE). NOBLOCK: if the wake buffer already holds an
        unconsumed poke, the poller will wake anyway — the pokes coalesce, so dropping one is harmless."""
        try:
            wake_w.send(b"", flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
