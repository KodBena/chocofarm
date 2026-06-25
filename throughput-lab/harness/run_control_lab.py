#!/usr/bin/env python3
"""
throughput-lab/harness/run_control_lab.py — the TLAB CONTROL-LAB harness: run the issue-gate control
lab's 16 controller METHODS on tlab-real-producer's existing Gate-A control plane, score each, and
egress every trial to the control_research PostgreSQL via the lab's own lab_store.

WHAT THIS IS (a REWIRE, not a reimplementation — ADR-0004 minimal-touch). The control-lab machinery
(adapter.Controller contract, the 16 self-registering methods, lab_store's postgres egress) was built
against a DIFFERENT producer (chocofarm-wire-ab-bench --lab-decision) over a DIFFERENT control wire
(lab_control_wire.hpp — a per-forward decision that rides ALONGSIDE the value request inside the eval
server's reply envelope, synchronous on the forward). tlab-real-producer does NOT speak that wire. It
carries the OTHER, pre-existing control plane (Gate A, tlab_finding #13): the ASYNC issue-control bridge
(issue_control_bridge.hpp) that, on a slow cadence, REQs an EXTERNAL Python POLICY ENGINE
(issue_engine.py) with a BATCHED snapshot of ALL T threads' features and applies the per-thread allow
bits the engine replies. So NO producer change is needed (ADR-0000: the producer already exposes the
seam) — and NO new C++ is written.

THE TWO PROTOCOLS DIFFER (the pivotal design question, resolved by reading both wires):
  * issue_control_bridge.hpp  (what tlab-real-producer speaks): batched ZMQ REQ/REP, one frame = ALL T
    threads' {inflight, ready, msgs, leaves, rtt_us}; the engine BINDS a REP socket and replies T allow
    bits. Async, slow cadence, dedicated control thread. The Python peer is issue_engine.IssueEngine.
  * lab_control_wire.hpp      (what LabServer speaks): per-thread, per-FORWARD, rode in the value reply
    envelope; synchronous. Its own header states it SUPERSEDES the async bridge FOR THE LAB.
They are deliberately different transports onto the SAME actuation hub (IssueController). Because they
differ, the integration is a THIN protocol ADAPTER (a Port/ACL — translate-and-validate, never coerce;
ADR-0002): the engine's batched FEATURES dict -> a per-thread adapter.Observation -> the active
Controller.act() -> the per-thread allow list the engine encodes back as GATES. The adapter authors NO
new policy and NO new wire; it only translates between the issue_engine dict the producer already speaks
and the FROZEN adapter.Controller contract the 16 methods already implement (ADR-0012 P8 typed-signature
SSOT — the Controller protocol is the contract; this harness honors it, it does not re-state it).

WHAT IS REUSED (maximally; the new code is the wiring + the thin adapter):
  * issue_engine.IssueEngine — the REP-socket serve loop the producer's bridge already targets. We inject
    a `policy` closure (the adapter) and read `on_features` to score.
  * control_lab.adapter — Controller / REGISTRY / Decimate / TrialContext / Observation.
  * control_lab.watchdog — the SHARED method-watchdog contract (MalfunctionRecord + validate_gates): one
    home both control wires import, NOT re-authored (ADR-0012 P1). A slow/throwing/malformed method FALLS
    BACK to all-allow and is FLAGGED; the fixture — the continuous producer — survives (ADR-0002).
  * control_lab.methods.load_all() + reference_methods — the 16 controllers self-register into REGISTRY.
  * control_lab.lab_store — the one-owner postgres egress (psycopg3 ONLY; control_research; connection
    facts in chocofarm/config.py). One lab_session row per harness run, one lab_trial per method, the
    per-sample metrics_series blob. (No (s,a,r) trajectory blob here: that codec is bound to the
    per-forward LabServer path; this async path scores from the producer's leaves telemetry instead.)

WHY ONLY --episodic --driver greedy. The Gate-A hook (may_issue + publish) is live ONLY in
run_thread_fiber_episodic's greedy pipe (real_producer.cpp ~378-405); the other three drivers neither
gate nor publish. So the producer is launched --episodic --driver greedy (the banked driver) — the one
config where the controllers actuate AND their telemetry flows.

THE MEMORY CONSTRAINT (load-bearing — tlab_finding #23). The control harness runs ONE producer
CONTINUOUSLY for minutes; at the banked 1024 fibers x 3 threads that OOMs an 8 GiB box and the producer's
admission guard REFUSES it loudly. The operating point is DERIVED from the banked SSOT
(hp.spec.banked_static / `python -m hp.cli --banked-static-env`) but OVERRIDES fibers to a memory-safe
value (default 256; threads=3 -> ~2.3 GiB estimated resident, comfortably under the guard's 50%-of-
MemAvailable bar — confirmed admitted before any long run). Everything else (n_sims, m, msg_rows,
inflight, driver) follows the bank.

THE SCORE. The async wire carries no per-thread DECISION count (the producer publishes cumulative
`leaves` per thread, not decisions). So the throughput meter this harness scores is LEAVES/SEC (lps):
Δ(sum of per-thread cumulative leaves) / Δt over each method's wall window — the right comparator for
gating policies (a gate that throttles issuance moves leaves/s directly). The producer ALSO prints its
own true decisions/s and leaves/s at teardown; those land in the producer log for cross-checking. lps is
mapped onto the lab_trial.dps_* columns (the store's throughput slot) and the run is stamped in
lab_session.notes so a reader never confounds this async-wire lps with the per-forward LabServer dps.

ARTIFACTS under ~/w/vdc (NEVER /tmp): a JSON session record + the postgres rows. The producer + server
logs are preserved under --out.

Usage:
    PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python harness/run_control_lab.py \
        --methods all_allow,ready_threshold2,token_bucket --secs 6 --fibers 256

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

# ---- locations (self-contained; resolve relative to this file) --------------------------------------
HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_ROOT = os.path.dirname(HARNESS_DIR)                       # throughput-lab/
REPO = os.path.dirname(LAB_ROOT)                              # repo root /home/bork/w/vdc/1/chocofarm
PYTHON = "/home/bork/w/vdc/venvs/generic/bin/python"
PRODUCER_BIN = os.path.join(LAB_ROOT, "cpp", "build", "tlab-real-producer")
INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")
STAGE_A = os.path.join(REPO, "cpp", "stage_a")               # where control_lab + issue_engine live

# Make the reused control-lab + issue-engine packages importable (the same path lab_harness adds).
for p in (REPO, LAB_ROOT, STAGE_A):
    if p not in sys.path:
        sys.path.insert(0, p)

import issue_engine  # noqa: E402 — the REP serve loop the producer's bridge targets (REUSED verbatim)
from control_lab import lab_store  # noqa: E402 — the one-owner postgres egress (control_research, psycopg3)
from control_lab import methods as _lab_methods  # noqa: E402 — the 14-method fan-out package
from control_lab import reference_methods  # noqa: F401,E402 — registers ready_threshold2 + malfunctioning
from control_lab.adapter import (  # noqa: E402 — the FROZEN Controller contract (ADR-0012 P8 SSOT)
    REGISTRY,
    Controller,
    Decimate,
    Observation,
    TrialContext,
)
# The method-watchdog contract (gate-shape validator + malfunction tally) has ONE shared home both control
# wires import (ADR-0012 P1) — NOT re-authored here. watchdog.py is dependency-light (no StageAServer/JAX),
# so the async policy peer reuses it without dragging the inference stack in (the concrete reason it is a
# separate module from lab_server, which imports StageAServer at module load).
from control_lab.watchdog import MalfunctionRecord, validate_gates  # noqa: E402

# Register the 14 candidate methods (+ the 2 references above + the all_allow baseline = the 16).
_lab_methods.load_all()

# Default core split: server core 0, producer cores 1,2,3 (the canonical 4-vCPU layout; run_real_best.sh).
SERVER_CORE = "0"
PRODUCER_CORES = "1,2,3"
DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "w", "vdc", "chocobo", "runs", "control_lab_tlab")
READY_RE = re.compile(r"\[tlab-server\] READY\b")


# ============================================================================================
# BANKED OPERATING POINT — the SSOT (hp.spec.banked_static), fibers OVERRIDDEN memory-safe.
# ============================================================================================
def banked_config() -> dict[str, int | str]:
    """The banked-static config (the SSOT), read from hp.spec. fibers is OVERRIDDEN below to a memory-
    safe value by the CLI; this returns the bank's own values for the rest (n_sims, m, msg_rows, inflight,
    driver, seconds). Fail loud if the bank module is absent (ADR-0002 — never silently guess a config)."""
    try:
        from hp import spec as _spec  # the tlab hp package (throughput-lab/hp)
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "run_control_lab: cannot import hp.spec for the banked operating point — ensure "
            "PYTHONPATH includes throughput-lab (the bank is the config SSOT; ADR-0002)") from exc
    env = _spec.banked_static()   # the banked-static SSOT: native dict (fibers/msg_rows/inflight_msgs/...).
    return {
        "fibers": int(env["fibers"]),
        "msg_rows": int(env["msg_rows"]),
        "inflight": int(env["inflight_msgs"]),
        "driver": str(env["driver"]),
        "seconds": int(env["seconds"]),
        "n_sims": int(env["n_sims"]),
        "m": int(env["m"]),
        "max_batch": int(env["max_batch"]),
    }


# The producer's own admission-guard estimate (real_producer.cpp est_fiber_resident_bytes), mirrored here
# so the harness can CONFIRM a config is guard-admitted BEFORE a long run (ADR-0013 verify the artifact —
# do not launch a producer that will SIGKILL minutes in). Derived from the SAME field widths the C++ uses.
def est_resident_mib(threads: int, fibers: int, n_sims: int) -> int:
    per_fiber = (512 * 1024) + (256 * 1024) + n_sims * 9 * 1024   # stack + arena floor + grown nodes
    live = threads * max(1, fibers)                              # fiber path keeps EVERY tree live at once
    return (live * per_fiber) >> 20


def mem_available_mib() -> "int | None":
    """MemAvailable from /proc/meminfo, in MiB (the kernel's own allocatable-without-swap estimate). None
    if unreadable (a typed absence — the guard then skips, never guesses; mirrors the C++)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


# ============================================================================================
# THE THIN PROTOCOL ADAPTER (a Port/ACL) — issue_engine FEATURES dict <-> adapter.Controller.
# Enforces the SHARED watchdog discipline (control_lab.watchdog: slow/throwing/malformed -> all-allow +
# FLAG; never wedge the producer) — imported, NOT re-authored (ADR-0012 P1, one home). The Controller
# decides the per-thread gate; the engine encodes it back as the GATES frame.
# ============================================================================================
class ControllerPolicy:
    """The Port/ACL between the issue_engine batched FEATURES dict and the FROZEN adapter.Controller.

    issue_engine calls this as its `policy(features_dict) -> allow_list`. We translate the dict (the
    decoded async FEATURES frame — n_threads, d_ceiling, server_rows_per_forward + length-T inflight/
    ready/msgs/leaves/rtt_us) into an adapter.Observation, run observe()+act() under the per-decision
    watchdog, and return the per-thread allow list. A slow/throwing/malformed decision FALLS BACK to
    all-allow for that tick and FLAGS the method (ADR-0002). The active controller is HOT-SWAPPED between
    trials (set_trial); the producer + the engine + the REP socket persist across the whole session.

    Scoring: each tick also accumulates the session-cumulative leaves total (sum of the per-thread
    cumulative leaves the producer published) under the lock — the harness deltas it per wall window to
    score leaves/sec. The per-thread cumulative is monotone; we track the running sum the same way
    lab_server tracks _lab_decisions_total (one home for the throughput numerator)."""

    def __init__(self, decision_deadline_s: float) -> None:
        self._deadline = float(decision_deadline_s)
        self._lock = threading.Lock()
        self._controller: Controller | None = None
        self._ctx: TrialContext | None = None
        self._T = 0
        self._gates: list[int] = []
        self._pending_reward: float | None = None
        self._mal = MalfunctionRecord()
        self._active = False
        # throughput numerator: session-cumulative leaves (monotone per thread -> running sum).
        self._thread_leaves: dict[int, int] = {}
        self._leaves_total = 0
        self._ticks = 0

    # ---- trial lifecycle (the harness calls these between trials, over the SAME producer/engine) ----
    def set_trial(self, controller: Controller, ctx: TrialContext) -> None:
        """Begin a fresh method-trial. RESET the controller FIRST (outside the lock, while the prior one is
        still serving — the un-reset-controller race lab_server documents), then swap it in atomically. The
        per-trial telemetry/malfunction record reset here; the session-cumulative leaves total persists (it
        is deltaed per window, not reset). A reset() that throws is the method's failure, surfaced loud."""
        controller.reset(ctx)
        with self._lock:
            self._controller = controller
            self._ctx = ctx
            self._T = int(ctx.n_threads)
            self._gates = [1] * self._T
            self._pending_reward = None
            self._mal = MalfunctionRecord()
            self._active = True

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe sample of the session-cumulative leaves + this trial's malfunction state + the
        controller's metrics() (under the watchdog — a throwing metrics() is flagged, not fatal)."""
        with self._lock:
            leaves_total = self._leaves_total
            ticks = self._ticks
            flags = list(self._mal.flags)
            malfunctions = self._mal.total()
            controller = self._controller
        method_metrics: dict[str, float] = {}
        if controller is not None:
            try:
                method_metrics = {str(k): float(v) for k, v in controller.metrics().items()}
            except Exception as exc:  # noqa: BLE001 — a bad metrics() is a flag, never fatal (ADR-0002)
                with self._lock:
                    self._mal.note("metrics_raised", f"metrics() raised: {exc!r}")
        return {
            "leaves_total": leaves_total,
            "ticks": ticks,
            "malfunctions": malfunctions,
            "flags": flags,
            "method_metrics": method_metrics,
        }

    # ---- the policy entry point issue_engine calls each control tick ----
    def __call__(self, f: dict) -> list[int]:
        T = int(f["n_threads"])
        # Update the session-cumulative leaves (the throughput numerator) from the published per-thread
        # cumulative counters — monotone, so we add only the positive delta (mirrors lab_server's decisions).
        leaves = f.get("leaves") or []
        with self._lock:
            for tid in range(min(T, len(leaves))):
                prev = self._thread_leaves.get(tid, 0)
                cur = int(leaves[tid])
                if cur > prev:
                    self._leaves_total += (cur - prev)
                    self._thread_leaves[tid] = cur
            self._ticks += 1
            active = self._active
            controller = self._controller
            if len(self._gates) != T:
                self._gates = [1] * T   # producer thread count is authoritative; re-size the gate vector
        if not active or controller is None:
            return [1] * T   # pre-first-trial (warm-up): plain all-allow, no policy call

        obs = self._build_obs(f, T)
        reward = float(f.get("server_rows_per_forward", 0.0)) or float(sum(int(x) for x in (f.get("ready") or [])))
        return self._decide(controller, obs, reward, T)

    def _build_obs(self, f: dict, T: int) -> Observation:
        """Translate the engine's batched FEATURES dict into the adapter.Observation the methods consume.
        The async wire carries ALL T threads every tick (not a per-forward served subset), so `served` is
        every thread and there is no absent-thread sentinel — a cleaner observation than the per-forward
        path. The features mapping carries the SAME keys the methods read (inflight/ready/msgs/leaves/
        rtt_us + n_threads/d_ceiling/server_rows_per_forward)."""
        features: dict[str, Any] = {
            "n_threads": T,
            "d_ceiling": int(f.get("d_ceiling", self._ctx.d_ceiling if self._ctx else 1)),
            "server_rows_per_forward": float(f.get("server_rows_per_forward", 0.0)),
            "inflight": list(f.get("inflight", []))[:T],
            "ready": list(f.get("ready", []))[:T],
            "msgs": list(f.get("msgs", []))[:T],
            "leaves": list(f.get("leaves", []))[:T],
            "rtt_us": list(f.get("rtt_us", []))[:T],
        }
        return Observation(features=features, served=list(range(T)),
                           forward_rows=int(features["server_rows_per_forward"]),
                           t_monotonic=time.monotonic())

    def _decide(self, controller: Controller, obs: Observation, reward: float, T: int) -> list[int]:
        """observe(previous reward) then act(obs) under the per-decision watchdog. Slowness / exception /
        malformed return FALL BACK to all-allow + FLAG the method (ADR-0002 loud). Holds the lock only for
        the cheap state writes, NOT across the policy call (a hung policy must not freeze the snapshot)."""
        t0 = time.monotonic()
        try:
            if self._pending_reward is not None:
                controller.observe(self._pending_reward, {"server_rows_per_forward": reward})
            new_gates = controller.act(obs)
            elapsed = time.monotonic() - t0
            if elapsed > self._deadline:
                with self._lock:
                    self._mal.note("slow_act", f"act() took {elapsed*1e3:.1f}ms > "
                                               f"{self._deadline*1e3:.1f}ms deadline")
                    self._mal.slow += 1
                    self._gates = [1] * T
                    self._pending_reward = reward
                    return list(self._gates)
            gates = validate_gates(new_gates, T)
            if gates is None:
                with self._lock:
                    self._mal.note("malformed_gates",
                                   f"act() returned a non-length-{T}/non-binary vector: {new_gates!r}")
                    self._mal.malformed += 1
                    self._gates = [1] * T
                    self._pending_reward = reward
                    return list(self._gates)
            with self._lock:
                self._gates = gates
                self._pending_reward = reward
                return list(self._gates)
        except Exception as exc:  # noqa: BLE001 — a throwing method is flagged + all-allow, never fatal
            with self._lock:
                self._mal.note("act_raised", f"act()/observe() raised: {exc!r}")
                self._mal.raised += 1
                self._gates = [1] * T
                self._pending_reward = reward
                return list(self._gates)


# ============================================================================================
# Orchestration: launch the tlab server + the policy engine + ONE continuous producer, hot-swap per
# method, score lps over a wall window, egress each trial.
# ============================================================================================
def _wait_for_ready(log_path: str, proc: subprocess.Popen, timeout_s: float) -> bool:
    """Tail the server's redirect FILE (never a pipe — the run_lab pipe-wedge discipline) until its READY
    line appears, it dies, or timeout. Returns True iff READY was seen."""
    deadline = time.monotonic() + timeout_s
    pos = 0
    while time.monotonic() < deadline:
        try:
            with open(log_path) as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except OSError:
            chunk = ""
        if chunk and any(READY_RE.search(ln) for ln in chunk.splitlines()):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.05)
    return False


def start_server(endpoint: str, in_dim: int, n_actions: int, hidden: int, max_batch: int,
                 server_core: str, out_dir: str, stamp: str) -> "tuple[subprocess.Popen, str]":
    """Launch the tlab inference server (`python -m server`) pinned to `server_core`, stdout/stderr to a
    FILE (the pipe-wedge defense). Returns (proc, log_path). The caller waits for READY before the producer."""
    env = dict(os.environ)
    env["PYTHONPATH"] = LAB_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = ["taskset", "-c", server_core, PYTHON, "-m", "server",
           "--bind", endpoint, "--in-dim", str(in_dim), "--n-actions", str(n_actions),
           "--hidden", str(hidden), "--max-batch", str(max_batch), "--poll-timeout-ms", "50"]
    log_path = os.path.join(out_dir, f"server-{stamp}.log")
    fh = open(log_path, "w")
    fh.write("# server cmd: " + " ".join(cmd) + "\n")
    fh.flush()
    proc = subprocess.Popen(cmd, cwd=LAB_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    fh.close()
    return proc, log_path


def launch_producer(endpoint: str, control_ep: str, threads: int, fibers: int, cfg: dict,
                    cadence_ms: float, producer_cores: str, out_dir: str, stamp: str) -> "tuple[subprocess.Popen, str]":
    """Launch ONE CONTINUOUS tlab-real-producer under taskset, --episodic --driver greedy (the ONLY driver
    where Gate A's may_issue+publish are live), pointed at the policy engine via --control-endpoint. A huge
    --seconds keeps it streaming for the whole session; the harness terminates it at the end. Memory-safe
    fibers (guard-admitted; checked by the caller before launch). stdout -> a FILE under --out."""
    cmd = ["taskset", "-c", producer_cores, PRODUCER_BIN,
           "--instance", INSTANCE, "--faces", FACES, "--endpoint", endpoint,
           "--threads", str(threads), "--fibers", str(fibers),
           "--msg-rows", str(cfg["msg_rows"]), "--driver", "greedy", "--episodic",
           "--inflight-msgs", str(cfg["inflight"]), "--n-sims", str(cfg["n_sims"]), "--m", str(cfg["m"]),
           "--seconds", "1000000",
           "--control-endpoint", control_ep, "--controller-cadence-ms", str(cadence_ms)]
    log_path = os.path.join(out_dir, f"producer-{stamp}.log")
    fh = open(log_path, "w")
    fh.write("# producer cmd: " + " ".join(cmd) + "\n")
    fh.flush()
    proc = subprocess.Popen(cmd, cwd=REPO, env=os.environ.copy(), stdout=fh, stderr=subprocess.STDOUT)
    fh.close()
    return proc, log_path


def resolve_method(spec: str) -> "tuple[str, int]":
    """Parse `name` or `name@k` into (registry_name, decimate_k). Fail loud on an unknown method
    (ADR-0002 — never a silent skip)."""
    name, _, ktok = spec.partition("@")
    k = int(ktok) if ktok else 1
    if name not in REGISTRY:
        raise KeyError(f"run_control_lab: method {name!r} not in REGISTRY (known: {sorted(REGISTRY)}) "
                       f"— refusing to guess (ADR-0002)")
    return name, k


def make_controller(name: str, k: int) -> "tuple[Controller, str, int]":
    obj = REGISTRY[name]()
    if not (hasattr(obj, "act") and hasattr(obj, "reset")):
        raise TypeError(f"run_control_lab: method {name!r} factory returned a non-Controller "
                        f"({type(obj).__name__}); supervised TrainableRecipe.fit is out of scope here")
    inner: Controller = obj  # type: ignore[assignment]
    if k > 1:
        wrapped = Decimate(inner, k)
        return wrapped, wrapped.family, k
    return inner, inner.family, 1


def agg(vals: "list[float]") -> dict:
    if not vals:
        return {"mean": 0.0, "pstdev": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {"mean": statistics.mean(vals), "pstdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals), "n": len(vals)}


def git_sha() -> "str | None":
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=5.0)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except Exception:  # noqa: BLE001
        return None


def run_trial(policy: ControllerPolicy, spec: str, ctx: TrialContext, secs: float,
              sample_hz: float, ts_file: Any) -> "tuple[dict, list[dict]]":
    """Run ONE method-trial over the persistent producer/engine stream: hot-swap the Controller (reset, NOT
    the fixture), box a `secs` wall window, sample at ~sample_hz, score lps from the policy's session-
    cumulative leaves delta. Returns (TRIAL_RECORD, samples). The fixture survives any malfunction."""
    name, k = resolve_method(spec)
    controller, family, k = make_controller(name, k)
    policy.set_trial(controller, ctx)
    label = name if k == 1 else f"decimate{k}:{name}"

    dt = 1.0 / max(1e-6, sample_hz)
    samples: list[dict] = []
    t0 = time.monotonic()
    base = policy.snapshot()
    base_leaves = int(base["leaves_total"])
    prev = base
    prev_t = t0
    lps_samples: list[float] = []
    while True:
        now = time.monotonic()
        t_rel = now - t0
        if t_rel >= secs:
            break
        time.sleep(min(dt, max(0.0, secs - t_rel)))
        snap = policy.snapshot()
        s_now = time.monotonic()
        d_lv = int(snap["leaves_total"]) - int(prev["leaves_total"])
        d_t = s_now - prev_t
        lps_inst = (d_lv / d_t) if d_t > 0 else 0.0
        if d_t > 0 and d_lv > 0:
            lps_samples.append(lps_inst)
        rec = {"method": label, "t_rel": round(t_rel, 4), "lps_inst": lps_inst,
               "leaves_total": snap["leaves_total"], "ticks": snap["ticks"],
               "malfunctions": snap["malfunctions"], "flags": snap["flags"],
               "method_metrics": snap["method_metrics"]}
        samples.append(rec)
        ts_file.write(json.dumps(rec) + "\n")
        ts_file.flush()
        prev = snap
        prev_t = s_now

    final = policy.snapshot()
    window_s = time.monotonic() - t0
    d_lv = int(final["leaves_total"]) - base_leaves
    lps_window = (d_lv / window_s) if window_s > 0 else 0.0
    flags = list(final["flags"])
    malfunctions = int(final["malfunctions"])
    rec = {"method": label, "family": family, "decimate_k": k, "window_s": round(window_s, 4),
           "lps": agg(lps_samples), "lps_window": lps_window, "leaves": d_lv,
           "malfunctions": malfunctions, "flags": flags, "ok": malfunctions == 0}
    print(f"[tlab-ctl] {label:>22} ({family:>9}) k={k}: lps_window={lps_window:9.1f} "
          f"lps_samp={rec['lps']['mean']:9.1f}+/-{rec['lps']['pstdev']:.1f} "
          f"leaves={d_lv} ticks={final['ticks']} malfunctions={malfunctions} "
          f"flags={flags if flags else '-'}", flush=True)
    return rec, samples


def _egress_trial(conn: Any, session_id: str, ctx: TrialContext, a: argparse.Namespace,
                  rec: dict, samples: "list[dict]") -> None:
    """BETWEEN-TRIAL postgres flush (off the control-tick path): map this trial onto a lab_trial row +
    a metrics_series blob, via lab_store (the one-owner SQL). lps is mapped onto the dps_* throughput
    slot (the store's columns); lab_session.notes records the wire so a reader never confounds it with the
    per-forward LabServer dps. A DB/SQL error propagates loud (ADR-0002)."""
    lps = rec["lps"]
    trow = lab_store.TrialRow(
        session_id=session_id, method=rec["method"], family=rec["family"], decimate_k=rec["decimate_k"],
        n_threads=ctx.n_threads, d_ceiling=ctx.d_ceiling, k_per_thread=ctx.k_per_thread, s_min=ctx.s_min,
        chunk_floor=ctx.chunk_floor, seed=ctx.seed, inflight_msgs=a.inflight_override or None,
        pool_batch=None, secs=a.secs,
        dps_window=rec["lps_window"], dps_mean=lps["mean"], dps_pstdev=lps["pstdev"],
        dps_min=lps["min"], dps_max=lps["max"],
        rows_per_fwd=None, forwards=None, n_decisions=rec["leaves"],
        malfunctions=rec["malfunctions"], flags=rec["flags"], ok=rec["ok"],
        started_at=lab_store.stamp_to_timestamp(time.strftime("%Y%m%d-%H%M%S")),
        duration_s=rec["window_s"])
    trial_id = lab_store.insert_trial(conn, trow)
    if samples:
        payload, raw_n, comp_n = lab_store.compress_metrics_series(samples)
        lab_store.insert_blob(conn, trial_id, kind="metrics_series", codec="zstd-json",
                              payload=payload, raw_bytes=raw_n, compressed_bytes=comp_n, n_decisions=None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="all_allow,ready_threshold2,token_bucket",
                    help="comma-separated registry names; suffix @k applies Decimate(k). "
                         "'ALL' = every registered method.")
    ap.add_argument("--secs", type=float, default=6.0, help="wall-time box per method (default 6.0s)")
    ap.add_argument("--threads", type=int, default=3, help="producer threads T (the gate vector length)")
    ap.add_argument("--fibers", type=int, default=256,
                    help="fibers/thread — the MEMORY-SAFE override of the banked 1024 (tlab_finding #23). "
                         "256 is verified-safe (~97k leaves/s) and guard-admitted at threads=3. Default %(default)s.")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--cadence-ms", type=float, default=5.0, help="control-loop tick period (producer bridge)")
    ap.add_argument("--decision-deadline-ms", type=float, default=50.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--server-core", default=SERVER_CORE)
    ap.add_argument("--producer-cores", default=PRODUCER_CORES)
    ap.add_argument("--warmup-grace-s", type=float, default=120.0,
                    help="max wall to wait for the producer's pool build + first published leaves")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-postgres", action="store_true", help="skip the control_research egress (JSON only)")
    a = ap.parse_args()
    a.inflight_override = 0  # filled below from the bank (so the egress records the real D)

    if not os.path.exists(PRODUCER_BIN):
        print(f"run_control_lab: producer binary not found at {PRODUCER_BIN} (build with "
              f"-DTLAB_REAL_GENERATOR=ON)", file=sys.stderr)
        return 2

    bank = banked_config()
    a.inflight_override = bank["inflight"]
    if bank["driver"] != "greedy":
        # The Gate-A hook is live only in the greedy-episodic pipe; a non-greedy bank would silently
        # disable the control plane. Fail loud (ADR-0002) rather than run a producer that ignores the gate.
        print(f"run_control_lab: banked driver is {bank['driver']!r}, but the Gate-A control hook is live "
              f"ONLY under --driver greedy --episodic — refusing to run an un-actuated producer (ADR-0002)",
              file=sys.stderr)
        return 2

    # CONFIRM the config is guard-admitted BEFORE a long run (ADR-0013 — verify, do not risk a mid-run
    # SIGKILL). The producer's own guard refuses > 50% of MemAvailable; we check the same bar up front.
    est = est_resident_mib(a.threads, a.fibers, bank["n_sims"])
    avail = mem_available_mib()
    print(f"[tlab-ctl] est_resident={est} MiB ({a.threads} threads x {a.fibers} fibers x n_sims="
          f"{bank['n_sims']}); MemAvailable={avail} MiB", flush=True)
    if avail is not None and est > avail // 2:
        print(f"run_control_lab: estimated resident {est} MiB exceeds 50% of MemAvailable {avail} MiB — "
              f"the producer's admission guard would REFUSE this. Reduce --fibers/--threads (ADR-0002).",
              file=sys.stderr)
        return 1

    specs = ([m for m in sorted(REGISTRY) if m != "malfunctioning"] + ["malfunctioning"]
             if a.methods.strip().upper() == "ALL"
             else [s.strip() for s in a.methods.split(",") if s.strip()])
    for s in specs:
        resolve_method(s)   # fail loud before standing anything up

    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = f"tlabctl-{stamp}"
    endpoint = f"ipc:///tmp/tlab-ctl-{os.getpid()}.sock"
    control_ep = f"ipc:///tmp/tlab-ctl-engine-{os.getpid()}.sock"
    T = a.threads

    # n_actions = the env's action-slot count (the producer drives real Gumbel-AZ; the net must match).
    from chocofarm.az.actions import n_action_slots  # noqa: E402
    from chocofarm.model.env import Environment  # noqa: E402
    n_actions = n_action_slots(Environment())
    in_dim = 241

    # The policy peer (the REUSED IssueEngine REP loop) + the thin adapter as its policy. on_features is
    # unused (the adapter itself accumulates the leaves total under its lock).
    policy = ControllerPolicy(decision_deadline_s=a.decision_deadline_ms / 1000.0)
    engine = issue_engine.IssueEngine(control_ep, policy=policy)

    # --- postgres egress: connect ONCE up front + ensure schema + insert the session row, so a DB outage
    # fails LOUD before a long run (ADR-0002). control_research, psycopg3, via lab_store (one-owner SQL).
    pg_conn = None
    if not a.no_postgres:
        pg_conn = lab_store.connect()
        lab_store.ensure_schema(pg_conn)

    ctx = TrialContext(n_threads=T, d_ceiling=bank["inflight"],
                       k_per_thread=a.fibers,   # the per-thread fiber count = the capacity normalizer here
                       s_min=bank["msg_rows"], chunk_floor=True, seed=7919)

    server_proc = None
    producer_proc = None
    records: list[dict] = []
    rc = 0
    try:
        server_proc, server_log = start_server(endpoint, in_dim, n_actions, a.hidden, bank["max_batch"],
                                                a.server_core, a.out, stamp)
        print(f"[tlab-ctl] server launched (pid={server_proc.pid}) core {a.server_core}; waiting for READY "
              f"(in_dim={in_dim} n_actions={n_actions} hidden={a.hidden} max_batch={bank['max_batch']})",
              flush=True)
        if not _wait_for_ready(server_log, server_proc, a.warmup_grace_s):
            raise RuntimeError(f"tlab server never reported READY within {a.warmup_grace_s}s — see {server_log}")
        print("[tlab-ctl] server READY", flush=True)

        engine.start()
        print(f"[tlab-ctl] policy engine (IssueEngine REP) bound {control_ep}; the 16 methods registered "
              f"({len(REGISTRY)} in REGISTRY)", flush=True)

        if pg_conn is not None:
            lab_store.insert_session(pg_conn, lab_store.SessionRow(
                session_id=run, started_at=lab_store.stamp_to_timestamp(stamp), git_sha=git_sha(),
                host=socket.gethostname(), reward_fn="server_rows_per_forward",
                regime=lab_store.REGIME_POST,
                net={"in_dim": in_dim, "n_actions": n_actions, "hidden": a.hidden,
                     "n_sims": bank["n_sims"], "m": bank["m"]},
                warm_pool={"threads": T, "fibers": a.fibers, "msg_rows": bank["msg_rows"],
                           "inflight_msgs": bank["inflight"], "driver": "greedy", "episodic": True,
                           "cadence_ms": a.cadence_ms, "server_core": a.server_core,
                           "producer_cores": a.producer_cores, "max_batch": bank["max_batch"]},
                notes=("TLAB ASYNC issue_control_bridge wire (issue_engine REP/REQ), NOT the per-forward "
                       "LabServer wire. SCORE = leaves/sec (lps) mapped onto the dps_* columns "
                       "(n_decisions column holds the window leaf delta). banked-static config, "
                       f"fibers OVERRIDDEN {bank['fibers']}->{a.fibers} memory-safe (tlab_finding #23).")))
            print(f"[tlab-ctl] control_research egress up: session {run} -> "
                  f"{lab_store.lab_pg_params().get('dbname')}@{lab_store.lab_pg_params().get('host')}",
                  flush=True)

        producer_proc, producer_log = launch_producer(endpoint, control_ep, T, a.fibers, bank,
                                                       a.cadence_ms, a.producer_cores, a.out, stamp)
        print(f"[tlab-ctl] producer launched (pid={producer_proc.pid}) --episodic --driver greedy "
              f"--fibers {a.fibers}; log={producer_log}", flush=True)

        # Wait for the producer to build its pool + start publishing leaves (the engine ticks climb). Fail
        # loud if it dies or never streams within the grace window (ADR-0002 — never a silent hang).
        t_wait0 = time.monotonic()
        last = 0
        while True:
            if producer_proc.poll() is not None:
                raise RuntimeError(f"producer exited early (rc={producer_proc.returncode}) before streaming "
                                   f"— see {producer_log}")
            snap = policy.snapshot()
            cur = int(snap["leaves_total"])
            if cur >= 200 and cur > last:
                break
            last = cur
            if time.monotonic() - t_wait0 > a.warmup_grace_s:
                raise RuntimeError(f"producer did not start publishing leaves within {a.warmup_grace_s}s "
                                   f"(leaves_total={cur}) — see {producer_log}")
            time.sleep(0.25)
        print(f"[tlab-ctl] producer streaming + control plane live after "
              f"{time.monotonic()-t_wait0:.1f}s (engine ticks={policy.snapshot()['ticks']})", flush=True)

        ts_path = os.path.join(a.out, f"tlabctl_timeseries-{stamp}.jsonl")
        with open(ts_path, "w") as ts_file:
            for spec in specs:
                rec, samples = run_trial(policy, spec, ctx, a.secs, a.sample_hz, ts_file)
                records.append(rec)
                if pg_conn is not None:
                    _egress_trial(pg_conn, run, ctx, a, rec, samples)
                if producer_proc.poll() is not None:
                    raise RuntimeError(f"producer DIED during method {spec!r} (rc={producer_proc.returncode}) "
                                       f"— the fixture did not survive; see {producer_log}")
    finally:
        for proc in (producer_proc, server_proc):
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10.0)
                except Exception:  # noqa: BLE001
                    try:
                        proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
        try:
            engine.stop()
        except Exception as exc:  # noqa: BLE001 — a failed engine is loud but must not mask a primary error
            print(f"[tlab-ctl] WARNING: engine stop raised (non-fatal): {exc!r}", flush=True)
        if pg_conn is not None:
            try:
                pg_conn.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[tlab-ctl] WARNING: postgres close raised (non-fatal): {exc!r}", flush=True)

    session = {"run": run, "stamp": stamp, "secs": a.secs, "threads": T, "fibers": a.fibers,
               "wire": "async-issue_control_bridge", "score": "leaves_per_sec",
               "banked": bank, "methods": specs, "n_threads": T}
    out_json = os.path.join(a.out, f"tlabctl_session-{stamp}.json")
    with open(out_json, "w") as f:
        json.dump({"schema_version": 1, "session": session, "trials": records}, f, indent=2)

    print("\n==== TLAB CONTROL LAB SUMMARY (leaves/s per method over a "
          f"{a.secs:.1f}s wall box, one continuous producer) ====", flush=True)
    base = next((r for r in records if r["method"] == "all_allow"), None)
    for r in records:
        rel = ""
        if base and base["lps_window"] > 0 and r["method"] != "all_allow":
            rel = f"  ({100.0 * r['lps_window'] / base['lps_window']:.0f}% of all_allow)"
        ok = "OK" if r["ok"] else f"MALFUNCTION{r['flags']}"
        print(f"  {r['method']:>22} ({r['family']:>9}): lps={r['lps_window']:9.1f}  [{ok}]{rel}", flush=True)
    print(f"\n[tlab-ctl] wrote {out_json}", flush=True)
    if not a.no_postgres:
        print(f"[tlab-ctl] control_research: session {run} + {len(records)} trial row(s) -> "
              f"{lab_store.lab_pg_params().get('dbname')}@{lab_store.lab_pg_params().get('host')}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
