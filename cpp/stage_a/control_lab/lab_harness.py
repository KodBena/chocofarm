#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/lab_harness.py — the FOUNDATIONAL (Batch-0) continuous, time-boxed CONTROL LAB
harness: the experimental platform that scores issue-gate controller METHODS against the leaf-eval
transport, over the PER-FORWARD on-wire decision path (lab_server.py + lab_control_wire.hpp).

THE ONE-OWNER SPLIT (ADR-0012 P3): this harness owns ORCHESTRATION + SCORING + the WATCHDOG-policy
wiring; the LabServer owns serving + the policy call (observe/act on the forward boundary) + the
per-decision watchdog mechanics; the codec has one home (lab_control_wire.hpp / lab_wire.py). The C++
producer (wire-ab-bench --lab-decision) owns the real Gumbel-AZ search + the issue actuation.

HOW IT RUNS (warmup paid ONCE, warm pool persists across the whole session):
  1. Build + publish ONE 241->H->65 net (the Stage-A geometry), stand up ONE LabServer pinned to core 0,
     warm every XLA bucket shape up front.
  2. Launch ONE CONTINUOUS C++ producer (wire-ab-bench --sweep-configs <one> --lab-decision 1) under
     taskset -c 1,2,3 with a huge measure budget, so it builds its warm pool ONCE and then streams real
     search forever — riding each thread's feature snapshot in, reading its gate bit back.
  3. Run a SEQUENCE of method-trials over that SAME stream: for each method, swap the server's Controller
     (server.set_trial — resets the method, NOT the fixture), run a wall-time-boxed window (default 4.0s),
     sample the time series, score dps from the server's cumulative-decision delta, then swap to the next.
     The warm pool + the XLA warm state + the producer all persist across every trial.

WATCHDOG (ADR-0002, never tear down the fixture on a bad method): the per-decision deadline + the
exception/malformed guards live in LabServer._run_controller (a slow/hung/throwing/malformed decision
FALLS BACK to all-allow for that decision and FLAGS the method); this harness reads those flags into the
trial record, marks the method, and CONTINUES. A degenerate-but-valid all-zeros method just scores low
and the wall box ends it. The producer's forced-flush stays the depth-1 liveness floor (a denied thread
never deadlocks), so a gate-everything method cannot wedge the producer.

ARTIFACTS under ~/w/vdc (NEVER /tmp): a JSON session record (the cross-batch schema below) + a JSONL
time-series the dashboard (a later batch) streams. The schema is DOCUMENTED in `SCHEMA_DOC` and is a
CROSS-BATCH CONTRACT the dashboard + data-collection batches reuse.

Usage:
    python lab_harness.py [--methods all_allow,ready_threshold2,malfunctioning]
                          [--secs 4.0 --hidden 256 --m 24 --n-sims 256 --max-batch 512
                           --pool-batch 192 --producer-threads 3 --inflight-msgs 8 --pool-plies 24
                           --decision-deadline-ms 50 --sample-hz 20 --out <dir>]

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

REPO = "/home/bork/w/vdc/1/chocofarm"
sys.path.insert(0, REPO)
_STAGE_A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STAGE_A)

import chocofarm.config  # noqa: F401,E402 — XLA/OMP single-thread pin BEFORE jax init (SSOT)

from chocofarm.az.actions import n_action_slots  # noqa: E402
from chocofarm.az.features import feature_dim  # noqa: E402
from chocofarm.az.inference_server import (  # noqa: E402
    StaticParamsSource,
    jit_forward_core,
    params_from_manifest_blob,
)
from chocofarm.az.mlp import ValueMLP  # noqa: E402
from chocofarm.az.transport import RedisTransport, connect, pack_net  # noqa: E402
from chocofarm.model.env import Environment  # noqa: E402

from control_lab import reference_methods  # noqa: F401,E402 — registers the reference methods into REGISTRY
from control_lab.adapter import REGISTRY, Controller, Decimate, TrialContext  # noqa: E402
from control_lab.lab_server import LabServer  # noqa: E402
from stage_a_server import BUCKETS  # noqa: E402

AB_BENCH = os.path.join(REPO, "cpp", "build", "chocofarm-wire-ab-bench")
INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")

SERVER_CORE = "0"
PRODUCER_CORES = "1,2,3"


# ============================================================================================
# THE RECORD / METRICS SCHEMA — the CROSS-BATCH CONTRACT (the dashboard + data-collection batches
# reuse this verbatim). Two artifacts, both under ~/w/vdc:
#
#   <out>/lab_session-<stamp>.json   — the session record:
#     { "schema_version": 1,
#       "session": { run, stamp, secs, hidden, m, n_sims, pool_batch, producer_threads, inflight_msgs,
#                    pool_plies, decision_deadline_ms, n_threads(T), server_core, producer_cores,
#                    reward_fn, methods:[...] },
#       "trials": [ TRIAL_RECORD, ... ] }
#
#     TRIAL_RECORD (one per method, the structured score + the malfunction flags):
#       { "method":     <registry name, e.g. "all_allow">,
#         "family":     <"static"|"online"|"supervised"|"rl">,
#         "decimate_k": <int, 1 if not decimated>,
#         "window_s":   <measured wall window, seconds>,
#         "dps":        { "mean", "pstdev", "min", "max", "n" }   — per-sample dps over the window
#                          (sample dps = Δ(server cumulative decisions) / Δt between consecutive samples);
#                          the SCORING metric (completed Gumbel searches / s),
#         "dps_window": <decisions over the whole window / window_s — the single headline number>,
#         "forwards":         <server forwards in the window>,
#         "forward_rows":     <server REAL rows forwarded in the window>,
#         "mean_forward_rows":<rows / forwards — the coalescing achieved (the reward signal's basis)>,
#         "malfunctions":     <count of watchdog hits (slow+raised+malformed)>,
#         "flags":            [<"slow_act"|"act_raised"|"malformed_gates"|"bad_feature_frame"|...>],
#         "ok":               <bool: malfunctions == 0> }
#
#   <out>/lab_timeseries-<stamp>.jsonl  — one JSON object per sample (the dashboard streams this live):
#       { "method", "t_rel": <s since window start>, "dps_inst": <instantaneous>,
#         "mean_forward_rows", "lab_decisions_total", "malfunctions", "flags", "method_metrics": {...} }
#     `method_metrics` is the Controller.metrics() snapshot (learned threshold / arm values / loss / ...),
#     so the dashboard plots a method's internal state over the window with zero schema change per method.
# ============================================================================================
SCHEMA_VERSION = 1
SCHEMA_DOC = "see the module header block — lab_session-*.json (schema_version=1) + lab_timeseries-*.jsonl"


def build_and_publish(hidden: int, run: str, version: int):
    """Build ONE 241->H->65 net (seed=17, residual=False), publish it to redis at (run,'gen',version) so
    the C++ producer's weight-read sanity passes, and return a StaticParamsSource over the SAME packed
    bytes so the in-process server serves the identical net. (The stage_b harness recipe.)"""
    env = Environment()
    in_dim, n_actions = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=hidden, n_actions=n_actions, seed=17,
                   y_mean=0.0, y_std=1.0, residual=False)
    manifest, blob = pack_net(net)
    rt = RedisTransport(connect())
    rt.publish_weights(net, phase="gen", version=version, run=run)
    params, y_mean, y_std = params_from_manifest_blob(manifest, blob)
    return StaticParamsSource(params, y_mean, y_std), in_dim, n_actions


def start_server(src, endpoint: str, max_batch: int, decision_deadline_s: float):
    """Stand up ONE LabServer (bucket-E + group-wakeup — the lab decision epoch) over `src`, warm every
    bucket shape + the max so a partial-drain forward never pays a cold JIT in a window, and spin the
    serve loop on a daemon thread. Returns (server, thread)."""
    server = LabServer(src, bind=endpoint, max_batch=max_batch, forward_fn=jit_forward_core,
                       e_policy="bucket", wakeup="group", decision_deadline_s=decision_deadline_s)
    server.warmup(sorted(set(BUCKETS) | {max_batch}))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, t


def launch_producer(endpoint: str, run: str, version: int, threads: int, pool_batch: int,
                    inflight: int, gc_m: int, n_sims: int, pool_plies: int, total_decisions: int,
                    log_path: str) -> subprocess.Popen:
    """Launch ONE CONTINUOUS C++ producer under taskset -c 1,2,3. --sweep-configs with a single config is
    the path that builds the warm pool ONCE and wires the IssueController; --lab-decision 1 rides features
    + actuates the gate off the reply. --measure-decisions is set huge so the single config streams for the
    whole session (the harness boxes by wall time on the server side and kills this at the end). Its stdout
    goes to `log_path` for diagnosis; the harness does NOT parse it (it scores from server counters)."""
    tok = f"lab-{uuid.uuid4().hex[:8]}"
    # One config: chunk_floor=0 (drain-all, the production depth-1 path), S_min=1, D=inflight.
    sweep = f"0:1:{inflight}"
    cmd = [
        "taskset", "-c", PRODUCER_CORES, AB_BENCH,
        "--instance", INSTANCE, "--faces", FACES, "--endpoint", endpoint,
        "--run", run, "--version", str(version), "--res-token", tok,
        "--wire-mode", "pipelined-bucket",
        "--m", str(gc_m), "--n-sims", str(n_sims),
        "--pool-threads", str(threads), "--pool-batch", str(pool_batch),
        "--inflight-msgs", str(inflight),
        "--sweep-configs", sweep,
        "--lab-decision", "1",
        "--measure-decisions", str(total_decisions),
        "--settle-decisions", "0",
        "--pool-plies", str(pool_plies),
        "--warmup-decisions", str(max(2000, pool_batch * 4)),
    ]
    logf = open(log_path, "w")
    logf.write(f"# producer cmd: {' '.join(cmd)}\n")
    logf.flush()
    return subprocess.Popen(cmd, cwd=REPO, text=True, stdout=logf, stderr=subprocess.STDOUT)


def agg(vals: "list[float]") -> dict:
    if not vals:
        return {"mean": 0.0, "pstdev": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": statistics.mean(vals),
        "pstdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def resolve_method(spec: str) -> "tuple[str, int]":
    """Parse a method spec into (registry_name, decimate_k). A `name` is k=1; `name@k` applies Decimate(k).
    The name must be in the REGISTRY (ADR-0002 — fail loud on an unknown method, never a silent skip)."""
    name, _, ktok = spec.partition("@")
    k = int(ktok) if ktok else 1
    if name not in REGISTRY:
        raise KeyError(f"lab_harness: method {name!r} is not in REGISTRY "
                       f"(known: {sorted(REGISTRY)}) — refusing to guess (ADR-0002)")
    return name, k


def make_controller(name: str, k: int) -> "tuple[Controller, str, int]":
    """Build a Controller from the registry factory, wrapping it in Decimate(k) when k>1. A factory that
    returns a TrainableRecipe (supervised, offline fit) is OUT of Batch-0 scope — fail loud (ADR-0002)."""
    obj = REGISTRY[name]()
    if not (hasattr(obj, "act") and hasattr(obj, "reset")):
        raise TypeError(f"lab_harness: method {name!r} factory returned a non-Controller "
                        f"({type(obj).__name__}); supervised TrainableRecipe.fit is out of Batch-0 scope")
    inner: Controller = obj  # type: ignore[assignment]
    if k > 1:
        wrapped = Decimate(inner, k)
        return wrapped, wrapped.family, k
    return inner, inner.family, 1


def run_trial(server: LabServer, method_spec: str, ctx: TrialContext, secs: float, sample_hz: float,
              ts_file: Any) -> dict:
    """Run ONE method-trial over the persistent stream: swap the Controller (reset, NOT the fixture), box a
    `secs` wall window, sample the time series at ~sample_hz, score dps from the server cumulative-decision
    delta. Returns the structured TRIAL_RECORD (the schema above). The fixture survives any method
    malfunction (the watchdog lives in the server; this reads its flags)."""
    name, k = resolve_method(method_spec)
    controller, family, k = make_controller(name, k)
    server.set_trial(controller, ctx)

    # Sample at the window edges + ~sample_hz between. The first sample is the window-start baseline (so the
    # first interval's dps is well-defined); the server counters are session-cumulative, so we delta them.
    dt = 1.0 / max(1e-6, sample_hz)
    samples: list[dict] = []
    t0 = time.monotonic()
    base = server.snapshot()
    base_dec = int(base["lab_decisions_total"])
    base_fwd = int(base["forwards"])
    base_rows = int(base["forward_rows_acc"])
    prev = base
    prev_t = t0
    dps_samples: list[float] = []
    while True:
        now = time.monotonic()
        t_rel = now - t0
        if t_rel >= secs:
            break
        time.sleep(min(dt, max(0.0, secs - t_rel)))
        snap = server.snapshot()
        s_now = time.monotonic()
        d_dec = int(snap["lab_decisions_total"]) - int(prev["lab_decisions_total"])
        d_t = s_now - prev_t
        dps_inst = (d_dec / d_t) if d_t > 0 else 0.0
        if d_t > 0 and snap["lab_decisions_total"] != prev["lab_decisions_total"]:
            dps_samples.append(dps_inst)
        rec = {
            "method": name if k == 1 else f"decimate{k}:{name}",
            "t_rel": round(t_rel, 4),
            "dps_inst": dps_inst,
            "mean_forward_rows": snap["mean_forward_rows"],
            "lab_decisions_total": snap["lab_decisions_total"],
            "malfunctions": snap["malfunctions"],
            "flags": snap["flags"],
            "method_metrics": snap["method_metrics"],
        }
        samples.append(rec)
        ts_file.write(json.dumps(rec) + "\n")
        ts_file.flush()
        prev = snap
        prev_t = s_now

    final = server.snapshot()
    window_s = time.monotonic() - t0
    d_dec = int(final["lab_decisions_total"]) - base_dec
    d_fwd = int(final["forwards"]) - base_fwd
    d_rows = int(final["forward_rows_acc"]) - base_rows
    dps_window = (d_dec / window_s) if window_s > 0 else 0.0
    mfwd = (d_rows / d_fwd) if d_fwd else 0.0
    flags = list(final["flags"])
    malfunctions = int(final["malfunctions"])
    rec = {
        "method": name if k == 1 else f"decimate{k}:{name}",
        "family": family,
        "decimate_k": k,
        "window_s": round(window_s, 4),
        "dps": agg(dps_samples),
        "dps_window": dps_window,
        "forwards": d_fwd,
        "forward_rows": d_rows,
        "mean_forward_rows": mfwd,
        "malfunctions": malfunctions,
        "flags": flags,
        "ok": malfunctions == 0,
    }
    print(f"[lab] {rec['method']:>22} ({family:>9}) {k=}: dps_window={dps_window:7.1f} "
          f"dps_samp={rec['dps']['mean']:7.1f}+/-{rec['dps']['pstdev']:.1f} "
          f"rows/fwd={mfwd:6.1f} fwds={d_fwd} malfunctions={malfunctions} "
          f"flags={flags if flags else '-'}", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=SCHEMA_DOC)
    ap.add_argument("--methods", default="all_allow,ready_threshold2,malfunctioning",
                    help="comma-separated registry names; suffix @k applies Decimate(k)")
    ap.add_argument("--secs", type=float, default=4.0, help="wall-time box per method (default 4.0s)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--n-sims", type=int, default=256)
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--pool-batch", type=int, default=192)
    ap.add_argument("--producer-threads", type=int, default=3)
    ap.add_argument("--inflight-msgs", type=int, default=8)
    ap.add_argument("--pool-plies", type=int, default=24)
    ap.add_argument("--decision-deadline-ms", type=float, default=50.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--warmup-grace-s", type=float, default=120.0,
                    help="max wall to wait for the producer's pool build + first lab forwards")
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "w", "vdc", "chocobo",
                                                  "runs", "control_lab"))
    a = ap.parse_args()

    # M3 pinning: this harness process hosts the in-process JAX server thread -> pin to the server core
    # only; the C++ producer runs under taskset -c 1,2,3 (off core 0). The 1:3 split the stage-b harness uses.
    os.sched_setaffinity(0, {int(SERVER_CORE)})
    affinity = sorted(os.sched_getaffinity(0))

    specs = [s.strip() for s in a.methods.split(",") if s.strip()]
    # Fail loud on an unknown method BEFORE standing anything up (ADR-0002).
    for s in specs:
        resolve_method(s)

    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = f"lab-{stamp}"
    version = 0
    endpoint = f"ipc:///tmp/choco-lab-{os.getpid()}.ipc"
    T = a.producer_threads  # the producer's pool_threads == the controller's n_threads (the gate vector len)

    src, in_dim, n_actions = build_and_publish(a.hidden, run, version)
    print(f"[lab] net published run={run} v={version} in_dim={in_dim} n_actions={n_actions} "
          f"hidden={a.hidden} | server pinned core {SERVER_CORE} (affinity={affinity}); "
          f"producers -> taskset -c {PRODUCER_CORES}, threads={T}", flush=True)

    server, server_thread = start_server(src, endpoint, a.max_batch, a.decision_deadline_ms / 1000.0)
    print(f"[lab] LabServer up (bucket+group, per-forward decision) endpoint={endpoint} "
          f"max_batch={a.max_batch} decision_deadline={a.decision_deadline_ms}ms", flush=True)

    # The producer streams for the whole session: a budget large enough that it never self-terminates
    # before the harness has run every wall window. (The harness kills it at the end.)
    est_session_s = len(specs) * a.secs + a.warmup_grace_s + 30.0
    huge_budget = 1_000_000_000
    prod_log = os.path.join(a.out, f"producer-{stamp}.log")
    producer = launch_producer(endpoint, run, version, T, a.pool_batch, a.inflight_msgs, a.m,
                               a.n_sims, a.pool_plies, huge_budget, prod_log)
    print(f"[lab] producer launched (pid={producer.pid}) -> streaming continuously over ONE warm pool; "
          f"log={prod_log}", flush=True)

    # Wait for the producer's pool build + the first LAB forwards to flow (warmup paid ONCE here). We detect
    # it by the server's cumulative-decision counter beginning to climb. Fail loud if the producer dies or
    # never streams within the grace window (ADR-0002 — never a silent hang).
    ctx = TrialContext(n_threads=T, d_ceiling=a.inflight_msgs,
                       k_per_thread=max(1, -(-a.pool_batch // T)), s_min=1, chunk_floor=False, seed=7919)
    records: list[dict] = []
    rc = 0
    try:
        t_wait0 = time.monotonic()
        last = 0
        while True:
            if producer.poll() is not None:
                raise RuntimeError(f"producer exited early (rc={producer.returncode}) before streaming — "
                                   f"see {prod_log}")
            snap = server.snapshot()
            cur = int(snap["lab_decisions_total"])
            if cur > 0 and cur > last + 50:   # streaming and climbing -> the pool is warm, lab forwards flow
                break
            last = cur
            if time.monotonic() - t_wait0 > a.warmup_grace_s:
                raise RuntimeError(f"producer did not start streaming lab forwards within "
                                   f"{a.warmup_grace_s}s (lab_decisions_total={cur}) — see {prod_log}")
            time.sleep(0.25)
        warm_wall = time.monotonic() - t_wait0
        print(f"[lab] warm pool primed + lab forwards flowing after {warm_wall:.1f}s "
              f"(paid ONCE; persists across all {len(specs)} method trials)", flush=True)

        ts_path = os.path.join(a.out, f"lab_timeseries-{stamp}.jsonl")
        with open(ts_path, "w") as ts_file:
            for spec in specs:
                rec = run_trial(server, spec, ctx, a.secs, a.sample_hz, ts_file)
                records.append(rec)
                # Fail loud if the fixture died under a method (it must NOT) — the producer is the canary.
                if producer.poll() is not None:
                    raise RuntimeError(f"producer DIED during method {spec!r} (rc={producer.returncode}) — "
                                       f"the fixture did not survive; see {prod_log}")
    finally:
        # Tear down: stop the producer first (it depends on the server), then the server. Each step is
        # guarded so a teardown hiccup never MASKS the real exception propagating out of the try (ADR-0002).
        try:
            producer.terminate()
            producer.wait(timeout=10.0)
        except Exception:   # noqa: BLE001 — a stubborn producer is killed; never block teardown
            try:
                producer.kill()
            except Exception:   # noqa: BLE001
                pass
        try:
            server.stop()
            server_thread.join(timeout=5.0)
            server.close()
        except Exception as exc:   # noqa: BLE001 — surface but do not mask the primary error
            print(f"[lab] WARNING: server teardown raised (non-fatal): {exc!r}", flush=True)

    # ---- the session record (the cross-batch schema) ----
    session = {
        "run": run, "stamp": stamp, "secs": a.secs, "hidden": a.hidden, "m": a.m, "n_sims": a.n_sims,
        "pool_batch": a.pool_batch, "producer_threads": T, "inflight_msgs": a.inflight_msgs,
        "pool_plies": a.pool_plies, "decision_deadline_ms": a.decision_deadline_ms, "n_threads": T,
        "server_core": SERVER_CORE, "producer_cores": PRODUCER_CORES, "reward_fn": "reward_forward_rows",
        "methods": specs,
    }
    out_json = os.path.join(a.out, f"lab_session-{stamp}.json")
    with open(out_json, "w") as f:
        json.dump({"schema_version": SCHEMA_VERSION, "session": session, "trials": records}, f, indent=2)

    print("\n==== CONTROL LAB SESSION SUMMARY (dps per method over a "
          f"{a.secs:.1f}s wall box, one warm pool) ====", flush=True)
    base = next((r for r in records if r["method"] == "all_allow"), None)
    for r in records:
        rel = ""
        if base and base["dps_window"] > 0 and r["method"] != "all_allow":
            rel = f"  ({100.0 * r['dps_window'] / base['dps_window']:.0f}% of all_allow)"
        ok = "OK" if r["ok"] else f"MALFUNCTION{r['flags']}"
        print(f"  {r['method']:>22} ({r['family']:>9}): dps={r['dps_window']:7.1f}  "
              f"rows/fwd={r['mean_forward_rows']:6.1f}  [{ok}]{rel}", flush=True)
    print(f"\n[lab] wrote {out_json}", flush=True)
    print(f"[lab] wrote {os.path.join(a.out, f'lab_timeseries-{stamp}.jsonl')}", flush=True)
    print(f"[lab] producer log {prod_log}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
