#!/usr/bin/env bash
# throughput-lab/harness/run_real_best.sh — the canonical, optimal real-generator load run for this
#   4-vCPU host, so the +18% scheduling win is not rediscovered by hand (ADR-0011: mechanize the finding;
#   see docs/notes/tlab-real-generators-2026-06-24.md). The winning topology (measured, IQR ~0.3%):
#     server         -> core 0
#     3 generators   -> cores 1, 2, 3  (one each, clean)
#     surplus gen    -> core 0, SCHED_IDLE  (soaks the server core's ~42% idle slack, yields to forwards)
#   All UNPRIVILEGED: SCHED_IDLE is self-settable; no cap, no root. Reports aggregate leaf-rows/s.
#
#   Usage:  run_real_best.sh [K] [SECONDS] [N_SIMS]   (defaults: 64 5 24)
#   Public Domain (The Unlicense).
set -euo pipefail

K="${1:-64}"; SECONDS_RUN="${2:-5}"; NSIMS="${3:-24}"
ROOT="/home/bork/w/vdc/1/chocofarm"
PY="/home/bork/w/vdc/venvs/generic/bin/python"
PROD="$ROOT/throughput-lab/cpp/build/tlab-real-producer"
WRAP="$ROOT/throughput-lab/cpp/build/sched_wrap"
INST="$ROOT/chocofarm/data/instance.json"
FACES="$ROOT/chocofarm/data/faces.json"
EP="ipc:///tmp/tlab-real-best-$$.sock"
LOG="$(mktemp -t tlab-real-best-XXXX.log)"

for b in "$PROD" "$WRAP"; do
  [ -x "$b" ] || { echo "missing binary: $b (build with -DTLAB_REAL_GENERATOR=ON)"; exit 2; }
done

# server on core 0 (file-redirected so a flooding write never wedges its SIGINT thread — see run_lab.py)
PYTHONPATH="$ROOT/throughput-lab" PYTHONUNBUFFERED=1 taskset -c 0 "$PY" -m server \
  --bind "$EP" --in-dim 241 --n-actions 65 --hidden 256 --max-batch 4096 --poll-timeout-ms 50 \
  >"$LOG" 2>&1 &
SRV=$!
cleanup(){ kill -INT "$SRV" 2>/dev/null || true; sleep 1; kill "$SRV" 2>/dev/null || true;
           rm -f "${EP#ipc://}"; }
trap cleanup EXIT
for _ in $(seq 1 240); do grep -q READY "$LOG" && break; sleep 0.5; done
grep -q READY "$LOG" || { echo "server never READY"; cat "$LOG"; exit 1; }
echo "server READY (core 0); launching 3 generators + 1 SCHED_IDLE surplus, K=$K, ${SECONDS_RUN}s"

gen(){ taskset -c "$1" "$PROD" --instance "$INST" --faces "$FACES" --endpoint "$EP" \
         --threads 1 --fibers "$K" --driver round-sync --seconds "$SECONDS_RUN" --n-sims "$NSIMS"; }

TMP="$(mktemp -d)"
gen 1 >"$TMP/g1" 2>&1 & p1=$!
gen 2 >"$TMP/g2" 2>&1 & p2=$!
gen 3 >"$TMP/g3" 2>&1 & p3=$!
# the surplus on core 0, SCHED_IDLE (via sched_wrap; runs only in the server core's idle slack)
taskset -c 0 "$WRAP" --policy idle -- "$PROD" --instance "$INST" --faces "$FACES" --endpoint "$EP" \
  --threads 1 --fibers "$K" --driver round-sync --seconds "$SECONDS_RUN" --n-sims "$NSIMS" \
  >"$TMP/gs" 2>&1 & ps=$!
# wait ONLY for the generators — a bare `wait` would also wait on the server (SRV), which runs until the
# EXIT-trap SIGINT, deadlocking. The generators self-terminate at --seconds; then we tear the server down.
wait "$p1" "$p2" "$p3" "$ps"

total=0
for f in "$TMP"/g1 "$TMP"/g2 "$TMP"/g3 "$TMP"/gs; do
  l=$(grep -oE 'leaves=[0-9]+' "$f" | head -1 | cut -d= -f2 || echo 0)
  total=$(( total + ${l:-0} ))
done
rm -rf "$TMP"
echo "AGGREGATE leaves=$total  leaf-rows/s=$(awk "BEGIN{printf \"%.0f\", $total/$SECONDS_RUN}")"
