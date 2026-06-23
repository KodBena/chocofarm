#!/usr/bin/env bash
# throughput-lab/harness/sweep.sh — hyperparameter sweep driver for MAX SERVING THROUGHPUT. Drives
# run_lab.py across a grid of SERVING-STRATEGY knobs (threads x rate x rows x max_batch, each over the
# topology x mode inner matrix run_lab already owns) via GNU parallel, then aggregates with
# sweep_report.py into a ranked report.
#
# WHY -j1 (SERIAL) BY DEFAULT — a throughput measurement needs the cores to ITSELF. This is a 4-vCPU
# guest (vCPUs pinned 1:1 to host cores; guest throughput verified ~= host for default params), so the
# 4 cores ARE the machine. Running cells CONCURRENTLY would make them contend for those 4 cores, and
# the measured throughput would then reflect contention, not the config — actively MISLEADING the
# best-config search. GNU parallel is used here purely as a clean, RESUMABLE cartesian-product driver:
#   * --joblog + --resume : re-running with the SAME OUTDIR skips grid points already completed;
#   * --results           : per-grid-point stdout/stderr captured for post-hoc inspection;
#   * --timeout           : a wedged grid point is reaped (wall-clock stall guard) instead of hanging.
# Override JOBS=N at your own risk (a loud 5s warning fires first); only sensible where each cell can
# own >= the full core set. (compute-watchdog.sh is NOT used to wrap a cell: run_lab is an IO-bound
# orchestrator — its own CPU flatlines while it waits on the server/producer, so a CPU-flatline kill
# would false-trip; parallel's wall-clock --timeout is the right stall guard for this layer.)
#
# CORE LAYOUT (the realistic chocofarm split): the CONSUMER (the server's compute) is pinned to core 0
# and the 3 PRODUCER (load) threads to cores 1,2,3 — one inference server vs N search workers, the
# isolation that stops the producer's spinning from stealing the server's matmul core. run_lab applies
# this PER-PROCESS via taskset (--server-core / --producer-cores); the sweep does NOT blanket-taskset.
#
# Held FIXED across the grid: producer threads=3 (pinned to the load cores) and the live workload
# geometry (in_dim=241, hidden=256, n_actions=0). The sweep varies the serving STRATEGY only — rows
# (producer leaves/message), max_batch (server rows/forward), topology, mode — so every cell measures
# the SAME work.
#
# Usage:
#   bash throughput-lab/harness/sweep.sh                      # full default grid -> ~/w/vdc/chocobo/runs/tlab/
#   THREADS="2 4" RATES=30000 ROWS=1 REPLICATES=3 bash .../sweep.sh   # override any grid axis
#   OUTDIR=<existing-sweep-dir> bash .../sweep.sh             # RESUME a partial sweep (skips done points)
#
# Public Domain (The Unlicense).
set -uo pipefail

LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/bork/w/vdc/venvs/generic/bin/python

# ---- the grid (env-overridable). Each is a space-separated list (the parallel ::: inputs). ----------
THREADS="${THREADS:-3}"                  # producer threads — FIXED at 3 (one per producer/load core)
RATES="${RATES:-100000}"                # per-thread target hz, deliberately ABOVE any config's ceiling so the
                                        # producer over-offers and back-pressure throttles it to the server's
                                        # serve rate => achieved-rate reads the true CEILING (now safe: the
                                        # byte-budgeted send HWM bounds memory, no OOM). sat% << 100 confirms it.
ROWS="${ROWS:-1 16 64}"                 # producer rows per WIRE MESSAGE B (leaves the producer packs per send)
MAXBATCH="${MAXBATCH:-512 4096 16384}"  # server gather row cap = rows per MLP forward (the consumer's batch)
# ---- inner matrix (run_lab handles these; comma lists) + measurement depth ----
MODES="${MODES:-decoupled,coupled}"     # decoupled = the throughput mode; coupled = RTT-bound (reported apart)
TOPOS="${TOPOS:-per-thread,coalescing}"
REPLICATES="${REPLICATES:-3}"           # per cell; achieved-rate median + min-max reported (robust to warmup)
SECONDS_PER="${SECONDS_PER:-3}"
SERVER_POLL_MS="${SERVER_POLL_MS:-10}"  # throughput-irrelevant since the wake pipe; small => low stop latency
# ---- execution ----
JOBS="${JOBS:-1}"                       # KEEP 1 on this host (see the -j1 rationale above)
SERVER_CORE="${SERVER_CORE:-0}"         # pin the CONSUMER (server compute) to this core
PRODUCER_CORES="${PRODUCER_CORES:-1,2,3}"  # pin the 3 PRODUCER (load) threads to these cores (disjoint)
JOB_TIMEOUT="${JOB_TIMEOUT:-900}"       # wall seconds per grid point before parallel reaps it
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="${OUTDIR:-$HOME/w/vdc/chocobo/runs/tlab/sweep-${STAMP}}"

# ---- preflight -------------------------------------------------------------------------------------
if [ ! -x "$LAB/cpp/build/tlab-producer" ]; then
    echo "sweep.sh: producer not built at $LAB/cpp/build/tlab-producer" >&2
    echo "  build:  cmake -S $LAB/cpp -B $LAB/cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build $LAB/cpp/build -j" >&2
    exit 2
fi
command -v parallel >/dev/null || { echo "sweep.sh: GNU parallel not found on PATH" >&2; exit 2; }
mkdir -p "$OUTDIR/cells" "$OUTDIR/logs"
mkdir -p "$HOME/.parallel" && touch "$HOME/.parallel/will-cite"  # silence parallel's citation reminder (scripted use)
PROGRESS="$OUTDIR/progress.log"   # the live, human-readable, `tail -f`-able feed (cells stream in as they finish)

if [ "${JOBS}" -gt 1 ]; then
    {
        echo "############################################################################"
        echo "WARNING: JOBS=${JOBS} (>1) on a $(nproc)-core host. A THROUGHPUT measurement"
        echo "needs the cores to itself — concurrent cells CONTEND and the numbers become"
        echo "contention-confounded, MISLEADING the best-config search. Use JOBS=1 unless"
        echo "each cell can own the full core set. Proceeding in 5s (Ctrl-C to abort)..."
        echo "############################################################################"
    } >&2
    sleep 5
fi

# ---- manifest (the record describes itself; ADR-0009) ----------------------------------------------
GITREV="$(git -C "$LAB" rev-parse --short HEAD 2>/dev/null || echo unknown)"
n_rate=$(wc -w <<<"$RATES"); n_rows=$(wc -w <<<"$ROWS"); n_mb=$(wc -w <<<"$MAXBATCH")
n_topo=$(tr ',' ' ' <<<"$TOPOS" | wc -w); n_mode=$(tr ',' ' ' <<<"$MODES" | wc -w)
NPOINTS=$(( n_rate * n_rows * n_mb ))
NCELLS=$(( NPOINTS * n_topo * n_mode ))
{
    echo "# throughput-lab sweep manifest"
    echo "stamp:    $STAMP"
    echo "git:      $GITREV"
    echo "host:     $(nproc) vCPU pinned 1:1 to host cores; JOBS=$JOBS (1 = serial)"
    echo "pinning:  consumer (server) -> core $SERVER_CORE  |  producer ($THREADS threads) -> cores $PRODUCER_CORES"
    echo "workload: FIXED threads=$THREADS in_dim=241 hidden=256 n_actions=0 (the live geometry)"
    echo "grid:     rate_hz/thr={$RATES}  rows={$ROWS}  max_batch={$MAXBATCH}"
    echo "inner:    topologies={$TOPOS}  modes={$MODES}"
    echo "depth:    replicates=$REPLICATES  seconds=$SECONDS_PER  server_poll_ms=$SERVER_POLL_MS"
    echo "size:     $NPOINTS grid points (parallel jobs) x ($n_topo topo x $n_mode mode) = $NCELLS cells"
} | tee "$OUTDIR/MANIFEST.txt" "$PROGRESS" >&2

{
    echo ""
    echo "=== sweep started $(date '+%F %T') — tail this file to watch cells stream in ==="
    echo "    columns: ach=achieved req/s  sat=achieved/requested  util=server compute%  batch=mean rows/forward  p50=latency"
    echo ""
} | tee -a "$PROGRESS" >&2

echo "[sweep] $NPOINTS grid points / $NCELLS cells -> $OUTDIR  (serial; warmup dominates wall time)" >&2
echo "[sweep] watch live:  tail -f $PROGRESS" >&2

# ---- drive the grid: one parallel job per (threads, rate, rows, max_batch) point --------------------
# NOTE: NO `--bar` and NO outer `taskset` here. `--bar` writes to /dev/tty, which does not exist in a
# background run (it spews "/dev/tty: No such device"); progress.log + joblog.tsv are the progress feed.
# Core pinning is per-process inside run_lab (--server-core / --producer-cores), not a blanket taskset.
# threads is FIXED ("$THREADS"=3, a literal arg); the grid axes are rate x rows x max_batch.
parallel -j"$JOBS" --timeout "$JOB_TIMEOUT" \
    --joblog "$OUTDIR/joblog.tsv" --resume --results "$OUTDIR/logs" \
    "$LAB/harness/sweep_cell.sh" "$THREADS" {1} {2} {3} "$OUTDIR" \
        --replicates "$REPLICATES" --seconds "$SECONDS_PER" \
        --modes "$MODES" --topologies "$TOPOS" --server-poll-timeout-ms "$SERVER_POLL_MS" \
        --server-core "$SERVER_CORE" --producer-cores "$PRODUCER_CORES" \
    ::: $RATES ::: $ROWS ::: $MAXBATCH
rc=$?

echo "[sweep] grid finished (parallel rc=$rc); aggregating + analyzing..." >&2
PYTHONPATH="$LAB" "$PY" "$LAB/harness/sweep_report.py"  "$OUTDIR" > "$OUTDIR/REPORT.md"
PYTHONPATH="$LAB" "$PY" "$LAB/harness/sweep_analyze.py" "$OUTDIR" > "$OUTDIR/ANALYSIS.md"

# Fold the final human artifacts into the tail of the live log, so one `tail -f progress.log` shows the
# whole arc: cells streaming in -> the ranked leaderboard -> the regression.
{
    echo ""
    echo "############################ GRID COMPLETE $(date '+%F %T') ############################"
    echo ""
    cat "$OUTDIR/REPORT.md"
    echo ""
    echo "##############################################################################################"
    echo ""
    cat "$OUTDIR/ANALYSIS.md"
} >> "$PROGRESS"

echo "" >&2
echo "[sweep] artifacts in $OUTDIR :" >&2
echo "         progress.log   — the live human feed you tailed (now also holds the final report+analysis)" >&2
echo "         REPORT.md      — the ranked throughput leaderboard (markdown)" >&2
echo "         ANALYSIS.md    — the OLS regression + marginal means + plain-language takeaway" >&2
echo "         MANIFEST.txt   — the grid + git rev this run measured" >&2
echo "         cells/*.json   — per-grid-point raw records (run_lab output)" >&2
echo "         joblog.tsv     — parallel job log (re-run with OUTDIR=$OUTDIR to --resume)" >&2
echo "         logs/          — per-grid-point stdout/stderr" >&2
echo "[sweep] DONE -> $OUTDIR/REPORT.md  and  $OUTDIR/ANALYSIS.md" >&2
