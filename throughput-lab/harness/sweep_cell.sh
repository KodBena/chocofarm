#!/usr/bin/env bash
# throughput-lab/harness/sweep_cell.sh — ONE sweep grid point: run run_lab.py for a single
# (threads, rate, rows, max_batch) point across ALL topologies x modes, writing the JSON cell records
# to the sweep's cells/ dir. Invoked once per grid point by sweep.sh via GNU parallel.
#
# The fixed WORKLOAD (the live net geometry: in_dim=241, hidden=256, n_actions=0) is held constant so
# "throughput" is COMPARABLE across grid points — the sweep varies the SERVING STRATEGY (how the load
# is shaped and gathered), never the work itself (shrinking the net would game the number, not answer
# the question). A grid point that produces no JSON at all (a hard run_lab failure) exits non-zero so
# GNU parallel marks it for re-run on --resume; a partial result (some cells ok=false) still wrote its
# JSON and is considered done (the failures are recorded in the record, not silently dropped).
#
# Public Domain (The Unlicense).
set -uo pipefail

threads="$1"; rate="$2"; rows="$3"; max_batch="$4"; outdir="$5"
shift 5   # remaining args are passed through verbatim to run_lab.py (replicates, seconds, modes, ...)

LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/bork/w/vdc/venvs/generic/bin/python

tag="t${threads}_r${rate}_b${rows}_mb${max_batch}"
out="${outdir}/cells/cell_${tag}.json"
mkdir -p "${outdir}/cells"
echo "[sweep-cell] ${tag} -> ${out}"

PYTHONPATH="$LAB" JAX_PLATFORMS=cpu "$PY" "$LAB/harness/run_lab.py" \
    --threads "$threads" --rate "$rate" --rows "$rows" --max-batch "$max_batch" \
    --in-dim 241 --hidden 256 --n-actions 0 \
    --json-out "$out" "$@" >/dev/null || true   # run_lab exits 1 on a PARTIAL failure but still writes JSON

if [ ! -f "$out" ]; then
    echo "[$(date +%T)] point ${tag}: produced NO JSON (hard failure) — will retry on --resume" >> "${outdir}/progress.log"
    echo "[sweep-cell] ${tag}: produced NO json (hard failure) — parallel will retry on --resume" >&2
    exit 1
fi

# Append this grid point's per-cell results to the live human feed (serial -j1 => ordered, no lock
# needed). This is what `tail -f ${outdir}/progress.log` shows as the sweep runs.
PYTHONPATH="$LAB" "$PY" "$LAB/harness/sweep_report.py" --cell "$out" >> "${outdir}/progress.log" 2>/dev/null || true
