#!/usr/bin/env bash
# throughput-lab/harness/topology_screen.sh — QUIET-BOX TOPOLOGY SCREEN (finding #18's open half; run it
#   like infpair.sh — box quiet, NO concurrent Claude). Screens all 40 canonical process-topology configs
#   at the BANKED producer/server operating point, 3 interleaved reps each, median leaf-rows/s, ranked.
#
#   The operating point (fibers / msg-rows / max-batch / driver / inflight / seconds / n-sims / m / warmup)
#   is resolved at run time from the hp SSOT (hp/spec.banked_static() via `-m hp.cli --banked-static-env`),
#   NOT hardcoded here — one home, the episodic_dps.sh derive-with-override pattern. Each BANKED_* var below
#   can be overridden by the matching environment variable (FIBERS=... etc.) for a sweep arm.
#   max-batch is LOAD-BEARING: the prior screen ran at the topology_sweep legacy default 4096, which put the
#   server in the big-batch/overcommit regime (mean ~1530-row batches padding to 4096, ~37% util) -> 74-92k
#   for ALL configs, uninterpretable. The banked single-pull point caps batches at MSG_ROWS (256).
#
#   The INCUMBENT is the banked topology config_id (hp/spec.banked_topology_config_id(); resolved + echoed
#   below) -- find that row in the ranked REPORT.md; anything that tops it by more than the ~0.5-1% screen
#   noise floor is a candidate worth a follow-up PAIRED test (topology_pair.py). Expectation: null (no
#   topology beats the incumbent), but this is the "be sure" sweep.
#
#   Output (attributable, never /tmp): ~/w/vdc/chocobo/runs/tlab/topo-screen-<UTC>/ {configs.json,
#   REPORT.md, results.json, server-*.log}. Runtime ~35-45 min (40 cfgs x 3 reps x (~10s warmup + 10s run)).
#   Public Domain (The Unlicense).
set -uo pipefail
cd /home/bork/w/vdc/1/chocofarm

PYBIN=/home/bork/w/vdc/venvs/generic/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$HOME/w/vdc/chocobo/runs/tlab/topo-screen-$STAMP"
mkdir -p "$OUT"

COMMIT=$(git rev-parse --short HEAD)
[ -z "$(git status --porcelain)" ] && TREE=clean || TREE=DIRTY
echo "=== topology screen @ $COMMIT ($TREE) -> $OUT ==="

# Banked PRODUCER/SERVER OPERATING POINT — resolved from the hp SSOT (hp/spec.banked_static()), NOT pinned
# here. One home for fibers/msg-rows/inflight/driver/seconds/n-sims/m/max-batch/warmup; the flag values
# below DERIVE from these BANKED_* vars (the matching env-var overrides still win, for sweeps). This kills
# the cross-harness defaults drift (notably --seconds; finding #21, banked at 10). Assign-then-eval so the
# resolver's non-zero exit propagates (`eval "$(...)"` would mask it — ADR-0002).
_BANKED_STATIC="$(PYTHONPATH=throughput-lab "$PYBIN" -m hp.cli --banked-static-env)" \
  || { echo "FATAL: could not resolve the banked static operating point from the hp SSOT" >&2; exit 1; }
eval "$_BANKED_STATIC"
FIBERS="${FIBERS:-$BANKED_FIBERS}"; SECONDS_="${SECONDS_:-$BANKED_SECONDS}"; NSIMS="${NSIMS:-$BANKED_N_SIMS}"
M="${M:-$BANKED_M}"; MSG_ROWS="${MSG_ROWS:-$BANKED_MSG_ROWS}"; INFLIGHT="${INFLIGHT:-$BANKED_INFLIGHT}"
DRIVER="${DRIVER:-$BANKED_DRIVER}"; WARMUP="${WARMUP:-$BANKED_WARMUP}"; MAXBATCH="${MAXBATCH:-$BANKED_MAX_BATCH}"

# The banked INCUMBENT topology config_id, resolved from the hp SSOT (echoed so the operator knows the row
# to compare against in REPORT.md). topology_enum --banked-env emits it as TOPOLOGY_STR among others.
_BANKED_TOPO_ENV="$(PYTHONPATH=throughput-lab:throughput-lab/harness "$PYBIN" \
  throughput-lab/harness/topology_enum.py --banked-env)" \
  || { echo "FATAL: could not resolve the banked topology from the hp SSOT" >&2; exit 1; }
eval "$_BANKED_TOPO_ENV"

# 1. Regenerate the config space into the run dir (self-contained + attributable; the enumerator is the
#    single home of WHICH configs exist -- ADR-0012 P1).
PYTHONPATH=throughput-lab:throughput-lab/harness "$PYBIN" \
  throughput-lab/harness/topology_enum.py --json "$OUT/configs.json"

# 2. Run the sweep at the banked operating point. topology_sweep.py interleaves reps, takes the median,
#    ranks, and writes REPORT.md + results.json. single-thread is a SERVER flag (orthogonal to topology).
PYTHONPATH=throughput-lab:throughput-lab/harness "$PYBIN" \
  throughput-lab/harness/topology_sweep.py \
    --configs "$OUT/configs.json" \
    --fibers "$FIBERS" --seconds "$SECONDS_" --n-sims "$NSIMS" --m "$M" \
    --msg-rows "$MSG_ROWS" --inflight "$INFLIGHT" --driver "$DRIVER" --episodic --single-thread \
    --warmup "$WARMUP" --max-batch "$MAXBATCH" --reps 3 \
    --outdir "$OUT"

echo
echo "=== DONE -> $OUT/REPORT.md ==="
echo "incumbent row to compare against: $BANKED_CONFIG_ID ($TOPOLOGY_STR)"
