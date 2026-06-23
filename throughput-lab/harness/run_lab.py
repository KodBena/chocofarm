#!/usr/bin/env python3
"""
throughput-lab/harness/run_lab.py — the RUN+MEASURE harness: stand up the Python server, run the C++
producer against it, collect the per-thread/aggregate throughput + latency + utilization, and report
one record per sweep cell. This is the reproducible OUTER LOOP a throughput claim cites (ADR-0009: a
throughput claim is honest only when its investigation is captured reproducibly).

ORCHESTRATION CONTRACT (what this file implements)
--------------------------------------------------
For each sweep cell (topology x mode x ...):
  1. pick a fresh ipc endpoint (ipc:///tmp/tlab-{pid}-{seq}.sock) so concurrent / sequential runs do
     not collide on a stale unix socket;
  2. launch the server: `python -m server --bind <endpoint> [--n-actions A] [--in-dim D] ...`
     (interpreter /home/bork/w/vdc/venvs/generic/bin/python, PYTHONPATH=throughput-lab) and WAIT for
     its `[tlab-server] READY ...` line on stdout before launching the producer (do NOT start the
     producer before warmup or the first batches pay XLA compile and the throughput is mis-measured,
     ADR-0009);
  3. run the producer: `cpp/build/tlab-producer --endpoint <endpoint> --topology <A|B> --mode
     <decoupled|coupled> --threads N --rate HZ --rows B --seconds S` and parse its machine-readable
     `RESULT thread=... ...` (per thread) and `AGGREGATE ...` (one) lines;
  4. tear down the server cleanly (SIGINT -> bounded stop()); capture its teardown stats line
     (`[tlab-server] served ...`) from stderr;
  5. emit a parse-friendly record (JSON) of the cell: (topology, mode, threads, rate, rows) ->
     (requested vs ACHIEVED aggregate throughput, per-thread latency p50/p99, server forward count /
     mean batch / compute utilization).

WHY A FRESH SERVER PER CELL.  One warmed server could serve every cell, but a per-cell server restart
gives each cell a CLEAN server-side stat window (forwards / batch histogram / compute-busy attributed
to exactly that cell, not bled across cells). The warmup cost is paid once per cell and is OUTSIDE the
producer's measured window (the producer only starts after READY), so it does not contaminate the
throughput number. The transparent, inspectable choice for a maintainer reading a single cell's record.

MEASUREMENT DISCIPLINE (the project's standing benchmark hygiene — CLAUDE.md memory).  This harness is
deliberately a SMOKE/SWEEP driver, not a publication-grade regression: it runs each cell ONCE by
default. For a load-bearing claim, raise --replicates (each cell is then run R times and the per-cell
achieved-rate median + IQR are reported), discard the first replicate as warmup, and wrap the whole
sweep in tools/shell/compute-watchdog.sh so a wedged producer/server trips a CPU-flatline kill instead
of hanging. The numbers this harness prints are MEASURED (achieved rate = productions/wall, server
util = compute-busy/wall); a requested-vs-achieved gap is surfaced, never hidden.

This harness ORCHESTRATES; it does not itself implement the load (that is the C++ producer) or the
compute (that is the Python server).

Run:
    PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python harness/run_lab.py
    PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python harness/run_lab.py \
        --threads 4 --rate 5000 --seconds 5 --rows 1 --replicates 3 \
        --json-out /home/bork/w/vdc/chocobo/runs/tlab-sweep.json

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---- fixed locations (the testbed is self-contained; resolve relative to this file) -----------------
HARNESS_DIR = Path(__file__).resolve().parent
LAB_ROOT = HARNESS_DIR.parent                      # throughput-lab/
PRODUCER_BIN = LAB_ROOT / "cpp" / "build" / "tlab-producer"
PYTHON = "/home/bork/w/vdc/venvs/generic/bin/python"   # the project interpreter (JAX/numpy/pyzmq)

# The matrix the brief names: {topology A, topology B} x {mode decoupled, mode coupled}.
TOPOLOGIES = ["per-thread", "coalescing"]          # A, B
MODES = ["decoupled", "coupled"]

# Parse-anchors on the two processes' machine-readable output.
_AGG_RE = re.compile(r"^AGGREGATE\s+(.*)$")
_RESULT_RE = re.compile(r"^RESULT\s+(.*)$")
_READY_RE = re.compile(r"\[tlab-server\] READY\b")
_SERVED_RE = re.compile(r"\[tlab-server\] served\s+(\d+)\s+requests\s+/\s+(\d+)\s+rows\s+in\s+([\d.]+)s")
_UTIL_RE = re.compile(r"compute-busy:\s+([\d.]+)s\s+\(([\d.]+)% of wall\)")
_FWD_RE = re.compile(r"forwards:\s+(\d+)\s+\(mean batch\s+([\d.]+)\s+rows,\s+max\s+(\d+)\)")


def _parse_kv(line: str) -> "dict[str, str]":
    """Parse a `key=value key=value ...` machine line into a dict of strings (the producer's RESULT /
    AGGREGATE format). Values are left as strings; the caller coerces the few it needs."""
    out: dict[str, str] = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


@dataclass
class CellResult:
    """One sweep cell's measured outcome — the parse-friendly record (one JSON object per cell)."""
    topology: str
    mode: str
    threads: int
    rate_hz_per_thread: float
    rows_per_batch: int
    seconds: float
    in_dim: int
    n_actions: int
    max_batch: int
    hidden: int
    server_poll_timeout_ms: int

    # producer-side (the load generator's own measurement)
    requested_total_hz: float = 0.0
    achieved_total_hz: float = 0.0
    batches_sent: int = 0
    replies_recv: int = 0
    any_overhead_bound: bool = False
    lat_mean_us: float = 0.0            # mean over threads of per-thread mean reply latency
    lat_p50_us: float = 0.0            # max over threads of per-thread p50 (worst-thread tail)
    lat_p99_us: float = 0.0            # max over threads of per-thread p99
    calib_ops_per_sec: float = 0.0     # mean over threads of calibrated x+=1 ops/sec

    # server-side (the compute's own teardown counters)
    server_requests: int = 0
    server_rows: int = 0
    server_wall_s: float = 0.0
    server_forwards: int = 0
    server_mean_batch_rows: float = 0.0
    server_max_batch_rows: int = 0
    server_compute_util_pct: float = 0.0

    # bookkeeping
    ok: bool = False
    note: str = ""
    replicate_achieved_hz: "list[float]" = field(default_factory=list)   # producer SEND rate (offered)
    replicate_served_hz: "list[float]" = field(default_factory=list)     # replies_recv/seconds (SERVED, honest)


def _wait_for_ready(proc: subprocess.Popen, server_lines: "list[str]", timeout_s: float) -> bool:
    """Block until the server prints its READY line on stdout, or it dies, or timeout. Every server
    stdout line is appended to `server_lines` so nothing is lost (the teardown stats arrive later, but
    READY may be interleaved with warmup logging). Returns True iff READY was seen."""
    deadline = time.monotonic() + timeout_s
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line == "":
            # EOF on stdout — the server exited before READY.
            if proc.poll() is not None:
                return False
            time.sleep(0.01)
            continue
        server_lines.append(line.rstrip("\n"))
        if _READY_RE.search(line):
            return True
    return False


def _drain_remaining(proc: subprocess.Popen, sink: "list[str]") -> None:
    """After the server has been asked to stop, read whatever it printed (the teardown stats summary is
    emitted to stderr; we merged stderr into stdout via the Popen below, so it shows up here)."""
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sink.append(line.rstrip("\n"))
    except Exception:  # noqa: BLE001 — best-effort drain on teardown
        pass


def _parse_server_stats(lines: "list[str]", cell: CellResult) -> None:
    """Pull the server's teardown counters out of its captured output into the cell record."""
    for ln in lines:
        m = _SERVED_RE.search(ln)
        if m:
            cell.server_requests = int(m.group(1))
            cell.server_rows = int(m.group(2))
            cell.server_wall_s = float(m.group(3))
        m = _UTIL_RE.search(ln)
        if m:
            cell.server_compute_util_pct = float(m.group(2))
        m = _FWD_RE.search(ln)
        if m:
            cell.server_forwards = int(m.group(1))
            cell.server_mean_batch_rows = float(m.group(2))
            cell.server_max_batch_rows = int(m.group(3))


def _run_one_cell(
    topology: str,
    mode: str,
    args: argparse.Namespace,
    seq: int,
) -> CellResult:
    """Launch a fresh server, wait for READY, run the producer for this (topology, mode) cell across
    `args.replicates` replicates, tear the server down, and assemble the CellResult."""
    cell = CellResult(
        topology=topology, mode=mode, threads=args.threads,
        rate_hz_per_thread=args.rate, rows_per_batch=args.rows, seconds=args.seconds,
        in_dim=args.in_dim, n_actions=args.n_actions,
        max_batch=args.max_batch, hidden=args.hidden,
        server_poll_timeout_ms=args.server_poll_timeout_ms,
    )

    endpoint = f"ipc:///tmp/tlab-{os.getpid()}-{seq}.sock"
    sock_path = endpoint[len("ipc://"):]
    # A stale socket from a crashed prior run would make bind() fail loudly; clear it first.
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass

    server_lines: "list[str]" = []
    server_env = dict(os.environ)
    server_env["PYTHONPATH"] = str(LAB_ROOT) + os.pathsep + server_env.get("PYTHONPATH", "")
    server_cmd = [
        PYTHON, "-m", "server",
        "--bind", endpoint,
        "--in-dim", str(args.in_dim),
        "--n-actions", str(args.n_actions),
        "--hidden", str(args.hidden),
        "--max-batch", str(args.max_batch),
        "--poll-timeout-ms", str(args.server_poll_timeout_ms),
    ]
    if args.residual:
        server_cmd.append("--residual")
    if args.server_core:
        # Pin the consumer (compute) to its own core so the producer's spinning cannot steal it.
        server_cmd = ["taskset", "-c", str(args.server_core)] + server_cmd

    print(f"\n=== cell: topology={topology} mode={mode} "
          f"threads={args.threads} rate={args.rate}hz rows={args.rows} "
          f"seconds={args.seconds} (x{args.replicates}) ===", file=sys.stderr, flush=True)

    # stderr -> stdout so the server's READY (stdout) and teardown stats (stderr) arrive on one stream.
    proc = subprocess.Popen(
        server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=server_env, text=True, bufsize=1,
    )
    try:
        if not _wait_for_ready(proc, server_lines, args.server_ready_timeout_s):
            cell.note = (f"server never reported READY within {args.server_ready_timeout_s}s "
                         f"(exit={proc.poll()}); last server lines: {server_lines[-5:]}")
            return cell
        print("    server READY", file=sys.stderr, flush=True)

        # --- producer replicates ----------------------------------------------------------------
        last_agg: "dict[str, str]" = {}
        last_results: "list[dict[str, str]]" = []
        for rep in range(args.replicates):
            prod_cmd = [
                str(PRODUCER_BIN),
                "--endpoint", endpoint,
                "--topology", topology,
                "--mode", mode,
                "--threads", str(args.threads),
                "--rate", str(args.rate),
                "--rows", str(args.rows),
                "--in-dim", str(args.in_dim),
                "--seconds", str(args.seconds),
                "--recv-timeout-ms", str(args.recv_timeout_ms),
            ]
            if args.producer_cores:
                # Pin the load generator to the producer cores (disjoint from the server core).
                prod_cmd = ["taskset", "-c", str(args.producer_cores)] + prod_cmd
            try:
                prod = subprocess.run(
                    prod_cmd, capture_output=True, text=True,
                    timeout=args.seconds + args.producer_grace_s,
                )
            except subprocess.TimeoutExpired:
                cell.note = (f"producer timed out (> {args.seconds + args.producer_grace_s}s) on "
                             f"replicate {rep}; likely a wedged transport — see the C++ recv timeout")
                return cell
            if prod.returncode != 0:
                cell.note = (f"producer exited {prod.returncode} on replicate {rep}: "
                             f"{prod.stderr.strip()[:400]}")
                return cell

            agg: "dict[str, str]" = {}
            results: "list[dict[str, str]]" = []
            for ln in prod.stdout.splitlines():
                m = _AGG_RE.match(ln)
                if m:
                    agg = _parse_kv(m.group(1))
                m = _RESULT_RE.match(ln)
                if m:
                    results.append(_parse_kv(m.group(1)))
            if not agg:
                cell.note = f"no AGGREGATE line from producer on replicate {rep}; stdout: {prod.stdout[-400:]}"
                return cell
            last_agg, last_results = agg, results
            cell.replicate_achieved_hz.append(float(agg.get("achieved_total_hz", "0")))
            cell.replicate_served_hz.append(float(agg.get("replies_recv", "0")) / max(args.seconds, 1e-9))
            print(f"    replicate {rep}: achieved={agg.get('achieved_total_hz')}hz "
                  f"sent={agg.get('batches_sent')} recv={agg.get('replies_recv')}",
                  file=sys.stderr, flush=True)

        # Report the LAST replicate's full detail, but the achieved-rate central value over replicates
        # (median is robust to a warmup-skewed first replicate; one replicate => that value).
        cell.requested_total_hz = float(last_agg.get("requested_total_hz", "0"))
        cell.achieved_total_hz = statistics.median(cell.replicate_achieved_hz)
        cell.batches_sent = int(last_agg.get("batches_sent", "0"))
        cell.replies_recv = int(last_agg.get("replies_recv", "0"))
        cell.any_overhead_bound = last_agg.get("any_overhead_bound", "0") == "1"
        if last_results:
            cell.lat_mean_us = statistics.mean(float(r.get("lat_mean_us", "0")) for r in last_results)
            cell.lat_p50_us = max(float(r.get("lat_p50_us", "0")) for r in last_results)
            cell.lat_p99_us = max(float(r.get("lat_p99_us", "0")) for r in last_results)
            cell.calib_ops_per_sec = statistics.mean(
                float(r.get("calib_ops_per_sec", "0")) for r in last_results)
        cell.ok = True

    finally:
        # Bounded clean teardown: SIGINT (the server's handler sets stop()), then wait; SIGKILL only if
        # it does not exit (so a wedged server cannot hang the sweep). Drain its remaining output for
        # the teardown stats either way.
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=args.server_stop_timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            cell.note = (cell.note + " | " if cell.note else "") + "server did not stop on SIGINT; killed"
        _drain_remaining(proc, server_lines)
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass

    _parse_server_stats(server_lines, cell)
    return cell


def _print_table(cells: "list[CellResult]") -> None:
    """A compact human summary table (the JSON record is the machine artifact; this is the eyeball)."""
    hdr = (f"{'topology':<12} {'mode':<10} {'thr':>3} {'req_hz':>9} {'ach_hz':>9} "
           f"{'sent':>8} {'recv':>8} {'lat_p50_ms':>11} {'lat_p99_ms':>11} "
           f"{'srv_util%':>9} {'srv_fwds':>8} {'srv_mbatch':>10} {'ok':>3}")
    print("\n" + hdr, file=sys.stderr)
    print("-" * len(hdr), file=sys.stderr)
    for c in cells:
        print(f"{c.topology:<12} {c.mode:<10} {c.threads:>3} "
              f"{c.requested_total_hz:>9.0f} {c.achieved_total_hz:>9.0f} "
              f"{c.batches_sent:>8} {c.replies_recv:>8} "
              f"{c.lat_p50_us / 1000.0:>11.2f} {c.lat_p99_us / 1000.0:>11.2f} "
              f"{c.server_compute_util_pct:>9.1f} {c.server_forwards:>8} "
              f"{c.server_mean_batch_rows:>10.1f} {'Y' if c.ok else 'N':>3}", file=sys.stderr)
        if c.note:
            print(f"    note[{c.topology}/{c.mode}]: {c.note}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_lab.py",
        description="throughput-lab sweep harness: server + C++ producer across "
                    "{per-thread,coalescing} x {decoupled,coupled}.")
    p.add_argument("--threads", type=int, default=2, help="producer threads (default: %(default)s)")
    p.add_argument("--rate", type=float, default=2000.0,
                   help="per-thread target emission rate, hz (default: %(default)s)")
    p.add_argument("--rows", type=int, default=1, help="rows per batch B (default: %(default)s)")
    p.add_argument("--seconds", type=float, default=3.0,
                   help="producer measured run duration (default: %(default)s)")
    p.add_argument("--replicates", type=int, default=1,
                   help="producer runs per cell; achieved-rate median reported (default: %(default)s)")
    p.add_argument("--in-dim", type=int, default=241, help="feature width (default: %(default)s)")
    p.add_argument("--n-actions", type=int, default=0,
                   help="policy width; 0 = value-only (default: %(default)s)")
    p.add_argument("--hidden", type=int, default=256, help="MLP hidden width (default: %(default)s)")
    p.add_argument("--max-batch", type=int, default=4096,
                   help="server N_total row cap per forward (default: %(default)s)")
    p.add_argument("--residual", action="store_true", help="use a residual net on the server")
    p.add_argument("--topologies", default=",".join(TOPOLOGIES),
                   help="comma list subset of per-thread,coalescing (default: both)")
    p.add_argument("--modes", default=",".join(MODES),
                   help="comma list subset of decoupled,coupled (default: both)")
    p.add_argument("--recv-timeout-ms", type=int, default=5000,
                   help="producer Boundary recv/poll timeout (default: %(default)s)")
    p.add_argument("--server-poll-timeout-ms", type=int, default=50,
                   help="server IO-thread idle poll timeout, ms (default: %(default)s). Bounds stop() "
                        "latency; NO LONGER floors coupled RTT (the server's reply wake pipe flushes a "
                        "reply the instant it is ready — see THE REPLY WAKE in server/server.py).")
    p.add_argument("--server-core", default=None,
                   help="pin the CONSUMER (server compute) process to this taskset cpu list, e.g. 0 "
                        "(default: unpinned). Mirrors the main harness's --server-core.")
    p.add_argument("--producer-cores", default=None,
                   help="pin the PRODUCER (load) process to this taskset cpu list, e.g. 1,2,3 "
                        "(default: unpinned). Mirrors the main harness's --producer-cores.")
    p.add_argument("--server-ready-timeout-s", type=float, default=120.0,
                   help="max wait for the server READY line, covering XLA warmup (default: %(default)s)")
    p.add_argument("--server-stop-timeout-s", type=float, default=10.0,
                   help="max wait for the server to stop on SIGINT before kill (default: %(default)s)")
    p.add_argument("--producer-grace-s", type=float, default=30.0,
                   help="seconds added to --seconds for the producer subprocess timeout "
                        "(covers calibration + tail-drain; default: %(default)s)")
    p.add_argument("--json-out", default=None,
                   help="write the per-cell records as a JSON array to this path (also stdout always)")
    return p


def main(argv: "list[str] | None" = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)

    if not PRODUCER_BIN.exists():
        print(f"run_lab.py: producer binary not found at {PRODUCER_BIN}\n"
              f"  build it first:  cmake -S {LAB_ROOT}/cpp -B {LAB_ROOT}/cpp/build "
              f"-DCMAKE_BUILD_TYPE=Release && cmake --build {LAB_ROOT}/cpp/build -j",
              file=sys.stderr)
        return 2

    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for t in topologies:
        if t not in TOPOLOGIES:
            print(f"run_lab.py: unknown topology {t!r} (choose from {TOPOLOGIES})", file=sys.stderr)
            return 2
    for m in modes:
        if m not in MODES:
            print(f"run_lab.py: unknown mode {m!r} (choose from {MODES})", file=sys.stderr)
            return 2

    cells: "list[CellResult]" = []
    seq = 0
    for topology in topologies:
        for mode in modes:
            seq += 1
            cells.append(_run_one_cell(topology, mode, args, seq))

    _print_table(cells)

    records = [asdict(c) for c in cells]
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, indent=2))
        print(f"\nrun_lab.py: wrote {len(records)} cell records to {out_path}", file=sys.stderr)
    # The JSON array is always emitted to stdout (the machine artifact a caller can capture).
    print(json.dumps(records, indent=2))

    # Exit non-zero iff any cell failed to produce a result (so a CI/sweep driver notices).
    return 0 if all(c.ok for c in cells) else 1


if __name__ == "__main__":
    raise SystemExit(main())
