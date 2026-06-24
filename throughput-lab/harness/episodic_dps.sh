#!/usr/bin/env bash
# throughput-lab/harness/episodic_dps.sh — the EPISODIC-static DPS baseline (the number any dynamic-control
#   attempt must beat; ADR-0009 measured). Runs the production-shape workload: real episodes (env stepped by
#   each executed action, --episodic) at the production search config (sims256/m24), --no-early-exit so
#   episodes run full-length, over the banked static optimum (server@0, 3 gens@1,2,3, SCHED_IDLE surplus@0,
#   --msg-rows from $4). Sums decisions across the 4 producers -> aggregate DPS; reports leaf-rows/s, LPD,
#   and the server serve-path breakdown. DPS is eval-limited here, so msg-rows (the coalescing floor) moves
#   it directly: run with MSG_ROWS=1 vs 64 for the auditable coalescing-translates-to-DPS comparison.
#
#   DRIVER selects the producer pipe shape, the only axis of the within-stack overlap A/B (ADR-0013):
#   round-sync (the committed barrier baseline) vs greedy (keep INFLIGHT coalesced msgs continuously in
#   flight so producer compute overlaps the server forward). Coalescing (MSG_ROWS) is held fixed across both.
#
#   Usage:  episodic_dps.sh [K] [SECONDS] [N_SIMS] [MSG_ROWS] [DRIVER] [INFLIGHT]
#           (defaults: 128 14 256 64 round-sync 8)
#   Public Domain (The Unlicense).
set -uo pipefail
cd /home/bork/w/vdc/1/chocofarm
PYBIN=/home/bork/w/vdc/venvs/generic/bin/python
PROD=throughput-lab/cpp/build/tlab-real-producer; W=throughput-lab/cpp/build/sched_wrap
INST=chocofarm/data/instance.json; FACES=chocofarm/data/faces.json
EP="ipc:///tmp/tlab-edps-$$.sock"; rm -f "${EP#ipc://}"; LOG="$(mktemp -t tlab-edps-XXXX.log)"
# DRIVER defaults to greedy: the within-stack A/B (3 interleaved replicates, msg-rows=64) banked it as a
# clean ADR-0009 win over round-sync -- greedy ~93 DPS median vs ~71 (greedy MIN 90 > round-sync MAX 77),
# +31%, AND tighter variance (round-sync's whole-round barrier is jitter-sensitive; greedy's continuous
# overlap smooths it). Pass `round-sync` as $5 to reproduce the baseline arm.
K="${1:-128}"; S="${2:-14}"; NSIMS="${3:-256}"; MSG_ROWS="${4:-64}"; DRIVER="${5:-greedy}"; INFLIGHT="${6:-8}"
for b in "$PROD" "$W"; do [ -x "$b" ] || { echo "missing $b (build -DTLAB_REAL_GENERATOR=ON)"; exit 2; }; done

PYTHONPATH=throughput-lab PYTHONUNBUFFERED=1 taskset -c 0 "$PYBIN" -m server --bind "$EP" \
  --in-dim 241 --n-actions 65 --hidden 256 --max-batch 4096 --poll-timeout-ms 50 >"$LOG" 2>&1 & SRV=$!
cleanup(){ kill -INT "$SRV" 2>/dev/null||true; sleep 1; kill "$SRV" 2>/dev/null||true; rm -f "${EP#ipc://}"; }
trap cleanup EXIT
for _ in $(seq 1 240); do grep -q READY "$LOG" && break; sleep 0.5; done
grep -q READY "$LOG" || { echo "server never READY"; cat "$LOG"; exit 1; }

G(){ taskset -c "$1" ${2:-} "$PROD" --instance "$INST" --faces "$FACES" --endpoint "$EP" \
     --threads 1 --fibers "$K" --msg-rows "$MSG_ROWS" --inflight-msgs "$INFLIGHT" \
     --episodic --no-early-exit --driver "$DRIVER" \
     --seconds "$S" --n-sims "$NSIMS" --m 24; }
TMP="$(mktemp -d)"
G 1 "" >"$TMP/e1" 2>&1 & a=$!; G 2 "" >"$TMP/e2" 2>&1 & b=$!; G 3 "" >"$TMP/e3" 2>&1 & c=$!
G 0 "$W --policy idle --" >"$TMP/es" 2>&1 & d=$!
wait "$a" "$b" "$c" "$d"

dec=0; lv=0
for f in "$TMP"/e1 "$TMP"/e2 "$TMP"/e3 "$TMP"/es; do
  D=$(grep -oE 'decisions=[0-9]+' "$f"|head -1|cut -d= -f2); L=$(grep -oE ' leaves=[0-9]+' "$f"|head -1|cut -d= -f2)
  dec=$((dec+${D:-0})); lv=$((lv+${L:-0}))
done
rm -rf "$TMP"
echo "EPISODIC-STATIC (sims${NSIMS}/m24, no-early-exit, 3 gens + IDLE surplus, driver=$DRIVER inflight=$INFLIGHT, msg-rows=$MSG_ROWS, K=$K, ${S}s)"
echo "  decisions=$dec  ->  DPS = $((dec/S))"
echo "  leaves=$lv  ->  leaf-rows/s = $((lv/S))   LPD ~= $((lv/(dec>0?dec:1)))"
grep -E 'served|forwards|compute-busy|latency' "$LOG"
