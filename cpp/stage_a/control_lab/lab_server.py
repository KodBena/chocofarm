#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/lab_server.py — the eval-server side of the issue-gate control lab's PER-FORWARD
on-wire decision transport. A bench-scoped subclass of the Stage-A StageAServer (which itself subclasses
the production InferenceServer); it does NOT touch the production eval path.

THE DECISION EPOCH IS THE FORWARD (synchronous). The producer rides each thread's feature snapshot in
the request's LAB-CONTROL envelope frame (lab_wire.py / lab_control_wire.hpp — frame[1], between the
corr-id and the value payload); after the forward produces predictions this server:
  1. decodes the served threads' FEATURE frames off their envelopes,
  2. computes the per-forward REWARD of the PREVIOUS act (a single swappable function — default the
     forward's real row count = coalescing achieved),
  3. calls the ACTIVE Controller — observe(reward, info) then act(obs) — to get the per-thread gate bits,
  4. tags each served reply's envelope with that thread's GATE frame (so the bit rides back on the wire).
Threads NOT in this forward keep their last-decided bit (the Controller holds a length-T gate vector;
each served thread's reply carries its CURRENT bit, refreshed whenever the thread next appears).

ONE-OWNER (ADR-0012 P3): the SERVER owns serving + the policy call (observe/act) on the forward boundary;
the HARNESS (lab_harness.py) owns orchestration + scoring + the watchdog; the CODEC has one home
(lab_control_wire.hpp, derived by lab_wire.py). The Controller is INJECTED (set_controller) and swapped
between trials by the harness over the SAME warm pool, so warmup is paid once.

WATCHDOG (ADR-0002 fail-loud, never tear down the fixture on a bad method): act()/observe() run under a
per-decision wall deadline + an exception guard; a slow/hung/throwing/malformed decision FALLS BACK to
all-allow for that decision and is FLAGGED loudly on the shared malfunction record — the forward still
serves, the fixture survives, the harness moves on. A degenerate-but-valid all-zeros method just scores
low (the box ends it). The fallback is the depth-1 liveness floor the producer already guarantees.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

REPO = "/home/bork/w/vdc/1/chocofarm"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
_STAGE_A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STAGE_A not in sys.path:
    sys.path.insert(0, _STAGE_A)

from control_lab.adapter import AllAllow, Controller, Observation, TrialContext  # noqa: E402
from control_lab.lab_wire import (  # noqa: E402
    LabFeature,
    decode_feature,
    encode_gate,
)
from stage_a_server import StageAServer  # noqa: E402


# ---- the per-forward REWARD function (the learning signal, NOT the score) ---------------------------
# A SINGLE, clearly-named, swappable function: given the just-completed forward's served-thread features +
# its real row count, return one scalar reward fed to the Controller's observe() for the PREVIOUS act. The
# default is the coalescing achieved (real rows in the forward) — more rows/forward is the lab's lever. The
# SCORING metric stays the bench's dps over the trial window (lab_harness.py); this is the per-step signal.
RewardFn = Callable[[int, "Sequence[LabFeature]"], float]


def reward_forward_rows(forward_rows: int, served: "Sequence[LabFeature]") -> float:
    """Default reward = the forward's REAL row count (the coalescing the server achieved this forward).
    A bigger coalesced forward amortizes the fixed per-forward cost — the throughput lever — so the
    Controller is rewarded for gate policies that fatten forwards."""
    return float(forward_rows)


@dataclass
class MalfunctionRecord:
    """Loud, structured record of a method's misbehaviour on the decision path (ADR-0002). The harness
    reads `flags` into the trial record and marks the method; the fixture is never torn down."""
    slow: int = 0           # act() exceeded the per-decision deadline (fell back to all-allow)
    raised: int = 0         # act()/observe() threw (fell back to all-allow)
    malformed: int = 0      # act() returned a non-length-T or non-binary vector (rejected -> all-allow)
    last_error: str = ""    # the most recent diagnostic string (for the harness log)
    flags: list[str] = field(default_factory=list)   # ordered, de-duplicated human-readable flags

    def note(self, flag: str, err: str = "") -> None:
        if err:
            self.last_error = err
        if flag not in self.flags:
            self.flags.append(flag)

    def total(self) -> int:
        return self.slow + self.raised + self.malformed


class LabServer(StageAServer):
    """The lab eval server: a StageAServer that runs the injected Controller on each forward boundary and
    rides the per-thread gate bit back on the reply envelope. Group-wakeup only (one forward over all
    drained rows — the lab measures coalesced forwards). The Controller + reward + watchdog deadline are
    swapped per trial by the harness; the warm pool persists across the whole session.

    REGIME FLIP (lab-staging-divergence-rca.md §6 fix #4 — the deliberate alignment the RCA named). Unlike
    its StageAServer parent (which pins `_uses_fixed_pad = False` because the Stage-A throughput corpus is
    ALWAYS the un-staged bench regime), the LAB chooses its forward regime FROM its pad policy: it OVERRIDES
    `_uses_fixed_pad` as a PROPERTY that DERIVES `self._e_policy == "padmax"`. A `padmax` lab pads every
    forward to the ONE fixed `max_batch` shape — exactly the shape `build_staged_forward`'s single-shape AOT
    handle compiled for — so the fixed-pad predicate is True and `InferenceServer._effective_forward` hands
    the lab production's DEVICE-RESIDENT STAGED forward (the post-staging regime production runs under). A
    `bucket` lab keeps the historical un-staged bench (per-forward width varies — the single-shape handle
    cannot serve it). ONE HOME (ADR-0012 P1): pad policy -> fixed-pad -> staging -> the regime stamp the
    harness egresses; the lab adopts or refuses production's staging by the SAME flag, never a second knob.
    (This shadows the inherited `_uses_fixed_pad = False` class attribute via the MRO; the only runtime
    reader, `_effective_forward`, runs on a serve/warmup — long after `__init__` set `self._e_policy` — so
    the property always sees a constructed policy. It is LAB-SPECIFIC: StageAServer's default is untouched,
    so the other Stage-A benchmarks keep their un-staged regime.)"""

    @property
    def _uses_fixed_pad(self) -> bool:  # type: ignore[override]
        """DERIVE the fixed-pad staging predicate from the lab's pad policy (the deliberate regime flip,
        lab-staging-divergence-rca.md §6 #4): `padmax` => the lab pads every forward to the ONE fixed
        max_batch shape the staged AOT handle compiled for => fixed-pad True => `_effective_forward` adopts
        production's device-resident STAGED forward (post-staging regime). `bucket` => per-forward width
        varies => fixed-pad False => the un-staged `self._forward_fn` (the historical bench regime), exactly
        as the single-shape handle requires. One home: the pad policy decides the regime."""
        return self._e_policy == "padmax"

    def __init__(self, *args: Any, controller: Controller | None = None,
                 reward_fn: RewardFn = reward_forward_rows,
                 decision_deadline_s: float = 0.050, **kwargs: Any) -> None:
        # Force group wakeup (the lab's coalesced-forward decision epoch); bucket-E is fine (the Stage-A
        # default). A caller passing wakeup="leaf" is a misuse for the lab (per-leaf forwards have no
        # cross-thread coalescing to reward) — fail loud.
        if kwargs.get("wakeup", "group") != "group":
            raise ValueError("LabServer requires wakeup='group' (the per-forward decision epoch)")
        kwargs["wakeup"] = "group"
        super().__init__(*args, **kwargs)
        self._reward_fn = reward_fn
        self._decision_deadline_s = float(decision_deadline_s)
        # The live gate vector (length T, all-allow until the Controller acts). Persists across forwards so
        # a thread absent from one forward still carries its last-decided bit on its next reply.
        self._gates: list[int] = []
        self._n_threads = 0
        self._trial_ctx: TrialContext | None = None   # NB: NOT self._ctx (that is the base zmq Context)
        self._controller: Controller = controller if controller is not None else AllAllow()
        self._trial_active = False   # True once set_trial has run; before that the gates are plain all-allow
        self._pending_reward: float | None = None   # reward of the PREVIOUS act, delivered before the next
        self._malfunction = MalfunctionRecord()
        # Per-trial telemetry the harness samples (lab_harness writes the time-series). Cheap counters
        # updated on the serve thread; read under the lock by reset()/snapshot().
        self._lock = threading.Lock()
        self._decisions = 0
        self._forward_rows_acc = 0
        self._forwards_in_trial = 0
        # Per-thread LATEST cumulative producer-side decision count (monotone over the whole continuous run)
        # + their running sum. The harness deltas `_lab_decisions_total` over each wall window to score true
        # dps (completed Gumbel searches/sec) WITHOUT stopping the continuous producer (the warm pool
        # persists across the session). Indexed by tid; grown lazily as threads appear.
        self._thread_decisions: dict[int, int] = {}
        self._lab_decisions_total = 0
        # The optional (s,a,r) trajectory sink (the RL-loop data). When the harness sets a TrajectoryBuffer
        # for a trial, each forward's COMMITTED decision (obs, action-actually-sent, reward) is appended on
        # the hot path (the codec's cheap allocation-free append, ~0.17% at this config). None -> no logging
        # (a pure-timing run). Encoded between trials by the harness; never on a forward. Read/written on the
        # serve thread under self._lock at swap time only (the append itself is on the serve thread).
        self._traj: Any | None = None

    # ---- trial lifecycle (the harness calls these between trials, over the SAME warm pool) ----
    def set_trial(self, controller: Controller, ctx: TrialContext,
                  reward_fn: RewardFn | None = None, decision_deadline_s: float | None = None,
                  trajectory: Any | None = None) -> None:
        """Begin a fresh method-trial: reset the Controller, the gate vector, the per-trial telemetry, and
        the malfunction record. `trajectory` is an OPTIONAL fresh TrajectoryBuffer (trajectory_codec) the
        server appends each forward's (obs, action, reward) into for this trial — None disables the sink
        (a pure-timing run). The harness builds ONE buffer per trial and encodes it after the wall box.
        Called between trials WITHOUT tearing down the server (warm pool persists). A reset() that throws is
        the method's failure, surfaced loudly to the harness (ADR-0002).

        ORDER (ADR-0002, the un-reset-controller race): the new controller is RESET FIRST — outside the lock,
        while the PRIOR controller is still the active one serving forwards — and only then ATOMICALLY swapped
        in + activated under the lock. The previous code activated (self._controller=…, self._trial_active=True)
        BEFORE reset(), so the serve thread could grab the brand-new, UN-RESET instance and drive its raw
        __init__ state (e.g. an RL learner's empty param pytree) for many forwards before reset() ran — a
        per-decision malfunction storm (act_raised/slow_act) that was the method's only on-wire failure. Resetting
        before activation keeps reset() off the lock (a non-trivial reset must not freeze the snapshot path) AND
        guarantees the serve thread only ever sees a fully-reset controller."""
        controller.reset(ctx)   # FIRST, outside the lock: the method's own reset (may be non-trivial / slow)
        with self._lock:
            self._controller = controller   # swap in the now-RESET controller atomically
            self._trial_ctx = ctx
            self._trial_active = True
            self._n_threads = int(ctx.n_threads)
            self._gates = [1] * self._n_threads          # all-allow until the first act
            if reward_fn is not None:
                self._reward_fn = reward_fn
            if decision_deadline_s is not None:
                self._decision_deadline_s = float(decision_deadline_s)
            self._pending_reward = None
            self._malfunction = MalfunctionRecord()
            self._decisions = 0
            self._forward_rows_acc = 0
            self._forwards_in_trial = 0
            self._traj = trajectory   # swap the per-trial sink (None -> no (s,a,r) logging this trial)

    def detach_trajectory(self) -> Any | None:
        """Atomically detach the per-trial trajectory sink: swap self._traj to None under the lock and
        return the old buffer (or None). The harness calls this AFTER the wall box ends and BEFORE it
        encodes the buffer, so the serve thread stops appending first — encode() then runs on a quiesced
        buffer (the codec's between-trials contract), never racing a concurrent append/_grow. Idempotent."""
        with self._lock:
            traj = self._traj
            self._traj = None
            return traj

    def snapshot(self) -> dict[str, Any]:
        """A thread-safe sample of the trial telemetry + the Controller's metrics() for the dashboard time
        series. Cheap; called by the harness sampler. metrics() runs under the watchdog (a throwing
        metrics() is flagged, not fatal)."""
        with self._lock:
            decisions = self._decisions
            rows = self._forward_rows_acc
            forwards = self._forwards_in_trial
            flags = list(self._malfunction.flags)
            malfunctions = self._malfunction.total()
            lab_decisions_total = self._lab_decisions_total
        try:
            method_metrics = {str(k): float(v) for k, v in self._controller.metrics().items()}
        except Exception as exc:   # noqa: BLE001 — a bad metrics() is a flag, never fatal (ADR-0002)
            with self._lock:
                self._malfunction.note("metrics_raised", f"metrics() raised: {exc!r}")
            method_metrics = {}
        return {
            "t_monotonic": time.monotonic(),
            "decisions": decisions,
            "forward_rows_acc": rows,
            "forwards": forwards,
            "mean_forward_rows": (rows / forwards) if forwards else 0.0,
            "lab_decisions_total": lab_decisions_total,   # session-cumulative; harness deltas it for dps
            "malfunctions": malfunctions,
            "flags": flags,
            "method_metrics": method_metrics,
        }

    def malfunction_record(self) -> MalfunctionRecord:
        return self._malfunction

    # ---- the decision boundary (override of the SEALED dispatch's _scatter hook, ADR-0012 P3) ----
    def _scatter(self, drained: list, responses: list, forwards: list) -> None:  # type: ignore[override]
        """The lab's serve/scatter BOUNDARY — run the Controller on the just-evaluated forward and tag each
        served reply with its thread's gate bit. After the ADR-0012 P3 template-method split
        (lab-staging-divergence-rca.md §6), this is an override of the OVERRIDABLE `_scatter` hook, NOT of the
        welded `_serve_batch`/dispatch: the SEALED `InferenceServer._run_forward` already ran the ONE group
        forward (the lab is group-wakeup; bucket-E via the inherited `_pad_shape`) and handed back the encoded
        `responses` + the per-forward `(real, pad)` in `forwards` — so the lab's Controller call + gate-frame
        tagging live ENTIRELY here, and the forward dispatch (`run_microbatch`) is the base's one home, never
        a hand-copy that can silently diverge (the bug class this split removes). The envelope is
        `[corr_id, feature_frame]` (frames[1:-1]); the reply re-uses corr_id verbatim and REPLACES
        feature_frame with the gate frame. A request with no/garbled FEATURE frame is served normally but
        contributes no observation + gets no gate frame (its reply is the 2-frame non-lab envelope), so a
        mixed lab/non-lab stream is safe."""
        # The lab is group-wakeup, so the sealed dispatch ran EXACTLY one forward (one `(real, pad)` entry);
        # `real` is this forward's coalescing — the reward signal + the counters' basis.
        real, pad = forwards[0]
        self.n_forwards += 1
        self.n_real_rows += real
        self.n_padded_rows += max(0, pad - real)

        # Decode the served threads' FEATURE frames (envelope = frames[1:-1]; for a lab request the single
        # middle frame is the feature snapshot). Build the per-thread observation surface.
        served_feats: list[LabFeature] = []
        served_envelopes: list[list[bytes]] = [env for _i, env, _X in drained]
        for env in served_envelopes:
            if len(env) >= 2:   # [corr_id, feature_frame, ...]; the lab feature is env[1]
                try:
                    served_feats.append(decode_feature(env[1]))
                except Exception as exc:   # noqa: BLE001 — a garbled feature is dropped from the obs, not fatal
                    with self._lock:
                        self._malfunction.note("bad_feature_frame", f"FEATURE decode: {exc!r}")
        # Update the session-cumulative decision count (the dps numerator) from the served threads' monotone
        # per-thread counters — the harness deltas this over each wall window. Under the lock (cheap).
        if served_feats:
            with self._lock:
                for f in served_feats:
                    prev = self._thread_decisions.get(f.tid, 0)
                    if f.decisions > prev:
                        self._lab_decisions_total += (f.decisions - prev)
                        self._thread_decisions[f.tid] = f.decisions

        # Run the Controller (under the watchdog) iff at least one served thread carried a feature snapshot
        # (a non-lab stream leaves the gates untouched — all-allow). Updates self._gates in place. If no
        # trial is active yet (pre-first-set_trial, during the warm-pool prime), LAZILY size the gate vector
        # to the largest served tid so every lab request still gets a valid (all-allow) GATE frame back —
        # the producer is in lab mode and STRICTLY expects a gate when it sent a feature (never an echo).
        if served_feats:
            with self._lock:
                need = max(f.tid for f in served_feats) + 1
                if len(self._gates) < need:
                    self._gates.extend([1] * (need - len(self._gates)))
                if self._n_threads < need:
                    self._n_threads = need
                run_ctl = self._trial_active   # pre-trial: keep the plain all-allow gates (no policy call)
            if run_ctl:
                self._run_controller(served_feats, real)

        # Scatter. A reply to a request that carried a FEATURE frame ALWAYS gets a GATE frame back (the
        # producer's recv_batch_lab requires it); the gate is the Controller's bit for that tid, defaulting
        # to allow=1 when no decision exists yet. ONLY a genuine non-lab request (no feature frame, tid None)
        # is echoed verbatim as the 2-frame envelope. A feature frame is NEVER echoed back (that would be a
        # 49-byte middle frame the producer would reject as a malformed GATE — the bug this guards).
        with self._lock:
            gates = list(self._gates)
        for (ident, resp), env in zip(responses, served_envelopes):
            tid = _tid_of(env)
            if tid is not None:
                allow = bool(gates[tid]) if 0 <= tid < len(gates) else True
                gate_frame = encode_gate(tid, allow)
                self._sock.send_multipart([ident, env[0], gate_frame, resp])   # [id][corr][gate][resp]
            else:
                self._sock.send_multipart([ident, *env, resp])   # non-lab: echo the envelope verbatim

    def _run_controller(self, served_feats: "Sequence[LabFeature]", forward_rows: int) -> None:
        """observe(reward_of_previous_act) then act(obs) under the per-decision watchdog. Any slowness,
        exception, or malformed return FALLS BACK to all-allow for this decision + FLAGS the method
        (ADR-0002 loud); the forward already served, the fixture survives. Holds self._lock only for the
        cheap state writes (NOT across the policy call — a hung policy must not freeze the snapshot path)."""
        T = self._n_threads
        # Build the Observation (the decoded feature surface for this forward). Length-T vectors, indexed
        # by tid; threads absent from this forward keep their prior (sentinel) feature slot.
        inflight = [0] * T
        ready = [0] * T
        msgs = [0] * T
        leaves = [0] * T
        rtt_us = [0] * T
        served_ids: list[int] = []
        for f in served_feats:
            if 0 <= f.tid < T:
                inflight[f.tid] = f.inflight
                ready[f.tid] = f.ready
                msgs[f.tid] = f.msgs
                leaves[f.tid] = f.leaves
                rtt_us[f.tid] = f.rtt_us
                served_ids.append(f.tid)
        features: dict[str, Any] = {
            "n_threads": T,
            "d_ceiling": self._trial_ctx.d_ceiling if self._trial_ctx is not None else 1,
            "server_rows_per_forward": float(forward_rows),
            "inflight": inflight, "ready": ready, "msgs": msgs, "leaves": leaves, "rtt_us": rtt_us,
        }
        obs = Observation(features=features, served=served_ids, forward_rows=int(forward_rows),
                          t_monotonic=time.monotonic())

        reward = self._reward_fn(forward_rows, served_feats)
        deadline = self._decision_deadline_s
        t0 = time.monotonic()
        try:
            # observe the PREVIOUS act's outcome (skip on the very first decision of the trial).
            if self._pending_reward is not None:
                self._controller.observe(self._pending_reward, {"forward_rows": forward_rows})
            new_gates = self._controller.act(obs)
            elapsed = time.monotonic() - t0
            if elapsed > deadline:
                # A slow (but eventually-returning) act still rode the forward latency; flag it and fall
                # back to all-allow for THIS decision so a slow method cannot wedge the throughput meter.
                with self._lock:
                    self._malfunction.note("slow_act",
                                           f"act() took {elapsed*1e3:.1f}ms > {deadline*1e3:.1f}ms deadline")
                    self._malfunction.slow += 1
                    self._gates = [1] * T
                    self._append_traj(obs, self._gates, reward)   # the COMMITTED (fallback) action
                    self._pending_reward = reward
                    self._decisions += 1
                    self._forward_rows_acc += forward_rows
                    self._forwards_in_trial += 1
                return
            gates = _validate_gates(new_gates, T)
            if gates is None:
                with self._lock:
                    self._malfunction.note("malformed_gates",
                                           f"act() returned a non-length-{T}/non-binary vector: {new_gates!r}")
                    self._malfunction.malformed += 1
                    self._gates = [1] * T
                    self._append_traj(obs, self._gates, reward)   # the COMMITTED (fallback) action
                    self._pending_reward = reward
                    self._decisions += 1
                    self._forward_rows_acc += forward_rows
                    self._forwards_in_trial += 1
                return
            with self._lock:
                self._gates = gates
                self._append_traj(obs, gates, reward)   # the COMMITTED policy action (the (s,a,r) RL tuple)
                self._pending_reward = reward
                self._decisions += 1
                self._forward_rows_acc += forward_rows
                self._forwards_in_trial += 1
        except Exception as exc:   # noqa: BLE001 — a throwing method is flagged + all-allow, never fatal
            with self._lock:
                self._malfunction.note("act_raised", f"act()/observe() raised: {exc!r}")
                self._malfunction.raised += 1
                self._gates = [1] * T
                self._append_traj(obs, self._gates, reward)   # the COMMITTED (fallback) action
                self._pending_reward = reward
                self._decisions += 1
                self._forward_rows_acc += forward_rows
                self._forwards_in_trial += 1

    def _append_traj(self, obs: Observation, action: "Sequence[int]", reward: float) -> None:
        """Append ONE committed decision (obs, action-actually-sent, reward) to the per-trial trajectory
        sink, iff one is set. The codec's cheap allocation-free hot append (~0.17% at this config); a None
        sink is a single branch (a pure-timing run pays nothing). MUST be called with self._lock held (the
        four _run_controller commit branches already hold it) — the buffer is swapped under the lock at
        set_trial, so this never races the encode (which the harness runs between trials)."""
        traj = self._traj
        if traj is not None:
            traj.append(obs, action, reward)


# ---- small pure helpers ----
def _validate_gates(gates: Any, T: int) -> "list[int] | None":
    """Validate a Controller's act() return: a length-T sequence of {0,1}. Returns the coerced list, or
    None on any shape/value violation (the caller then falls back to all-allow + flags)."""
    try:
        g = list(gates)
    except TypeError:
        return None
    if len(g) != T:
        return None
    out: list[int] = []
    for v in g:
        if v == 0:
            out.append(0)
        elif v == 1:
            out.append(1)
        else:
            return None
    return out


def _tid_of(envelope: "list[bytes]") -> "int | None":
    """Recover the producer thread id from a request envelope by decoding its FEATURE frame (env[1]).
    Returns None for a non-lab (no feature frame) or garbled envelope — the reply is then echoed verbatim
    (no gate). Cheap: the feature frame is 41 bytes."""
    if len(envelope) < 2:
        return None
    try:
        return decode_feature(envelope[1]).tid
    except Exception:   # noqa: BLE001 — a non-lab/garbled middle frame just means "no gate for this reply"
        return None
