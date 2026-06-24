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
# DRIVER defaults to greedy. NOTE: the earlier "+31% clean win" was RETRACTED (an unstamped single-session
# reading; journey doc Witness 2, commit 2ac1cef). The stamped quiet-box 2x2 shows the driver is
# REGIME-DEPENDENT: ~+4.5% and NOT clean at the pad-tax 4096 ladder (server compute-bound, no idle to
# overlap), but a CLEAN +15% at the lean ladder below (greedy MIN > round-sync MAX) where the fast server
# is coupling/RTT-limited. Greedy still wins-or-ties everywhere, so it stays the default. Pass `round-sync`
# as $5 for the baseline arm.
# DEFAULT K=256 (banked, journey doc Witness 3): the bridge-the-2x attribution found the tlab/overcommit
# leaf-rows/s gap was producer batch-FILL, not compute. K=128 underfilled the 256 bucket (147 real, 58%);
# K=256 fills it (210 real, 82%) -> +27% (74k->95k leaf-rows/s), landing in overcommit's efficient regime
# at HIGHER util (78.8% vs ~71%). Pass K=128 as $1 for the old underfilled arm.
K="${1:-256}"; S="${2:-14}"; NSIMS="${3:-256}"; MSG_ROWS="${4:-128}"; DRIVER="${5:-greedy}"; INFLIGHT="${6:-8}"
# Server bucket ladder (the snap-up policy; server reads the warmed set back from the forward -> one home).
# DEFAULT = {64,256}/max-256 (banked, Witness 3): with K=256 the producer fills the 256 bucket, so CAPPING
# at 256 (no 512) forbids the wasteful 512-spill (a ~210-row gather padded to 512 = 41% fill, a slower
# forward for no gain). max-256 + K=256 = the well-filled efficient regime; +27% over the old K=128 banked
# config, every replicate. (Earlier Witness 2 banked the lean {64,256,512}/512 ladder over the old
# [1,8,64,512,4096] pad-tax ladder -- +25-37%; this refines it.) Override WARMUP/MAXBATCH for other arms.
WARMUP="${WARMUP:-64,256}"; MAXBATCH="${MAXBATCH:-256}"
# ADR-0011 (mechanize the finding): stamp EVERY reading with the exact code state so an attributed number
# is always time-travellable. commit = HEAD short hash; tree = clean|dirty (dirty => the producer binary /
# harness may not match HEAD, so the reading is NOT a reproducible artifact until committed). The maintainer's
# rule: never record an attributed reading without its commit hash (a session-to-session discrepancy you
# cannot pin to a code state is unattributable by construction). See robust-benchmark-statistics.
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_TREE="$(test -z "$(git status --porcelain 2>/dev/null)" && echo clean || echo DIRTY)"
for b in "$PROD" "$W"; do [ -x "$b" ] || { echo "missing $b (build -DTLAB_REAL_GENERATOR=ON)"; exit 2; }; done

# SINGLE_THREAD (env, non-empty) serves on ONE thread (the production InferenceServer model) instead of the
# two-thread IO/compute split -- the A/B arm for the two-thread-on-one-core contention test (tlab_finding #4).
PYTHONPATH=throughput-lab PYTHONUNBUFFERED=1 taskset -c 0 "$PYBIN" -m server --bind "$EP" \
  --in-dim 241 --n-actions 65 --hidden 256 --max-batch "$MAXBATCH" --warmup "$WARMUP" \
  --poll-timeout-ms 50 ${SINGLE_THREAD:+--single-thread} >"$LOG" 2>&1 & SRV=$!
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
# SIGINT the server NOW and wait for its teardown summary to flush, so the server-side stats (mean batch,
# util) are available to the report AND the auto-record. (The EXIT trap still hard-kills it afterwards.)
kill -INT "$SRV" 2>/dev/null
for _ in $(seq 1 24); do grep -q 'compute-busy' "$LOG" && break; sleep 0.5; done

dec=0; lv=0
for f in "$TMP"/e1 "$TMP"/e2 "$TMP"/e3 "$TMP"/es; do
  D=$(grep -oE 'decisions=[0-9]+' "$f"|head -1|cut -d= -f2); L=$(grep -oE ' leaves=[0-9]+' "$f"|head -1|cut -d= -f2)
  dec=$((dec+${D:-0})); lv=$((lv+${L:-0}))
done
rm -rf "$TMP"
RPF=$(grep -oE 'mean batch [0-9.]+' "$LOG"|grep -oE '[0-9.]+'|head -1)
UTIL=$(grep -oE '\([0-9.]+% of wall\)' "$LOG"|grep -oE '[0-9.]+'|head -1)
# server_impl carries the threading ARM (single-thread vs two-thread). This is an ARTIFACT/treatment facet,
# NOT a hyperparameter: it is an A/B between server designs resolved by deleting the loser, not a knob tuned
# to a shipped optimum -- so it lives in provenance (server_impl), never in the hp/ SSOT (ADR-0008 classify).
SERVER_IMPL="tlab-server.py:$([ -n "${SINGLE_THREAD:-}" ] && echo single-thread || echo two-thread)"
echo "EPISODIC-STATIC [commit=$GIT_COMMIT tree=$GIT_TREE] (sims${NSIMS}/m24, no-early-exit, 3 gens + IDLE surplus, driver=$DRIVER inflight=$INFLIGHT, msg-rows=$MSG_ROWS, ladder=[$WARMUP] max-batch=$MAXBATCH, K=$K, server=$SERVER_IMPL, ${S}s)"
echo "  decisions=$dec  ->  DPS = $((dec/S))"
echo "  leaves=$lv  ->  leaf-rows/s = $((lv/S))   LPD ~= $((lv/(dec>0?dec:1)))"
grep -E 'served|forwards|compute-busy|latency' "$LOG"
# Auto-persist this reading to throughput_research (the measurement side of "commit as we go"). Best-effort:
# exp_db --record is loud-but-non-fatal (a DB blip dumps the reading under ~/w/vdc, never fails the run).
# Set TLAB_NO_DB=1 to skip; TLAB_TAG to label a cohort.
if [ "${TLAB_NO_DB:-0}" != "1" ]; then
  RID=$(printf '{"config":{"driver":"%s","server_impl":"%s","producer_bin":"tlab-real-producer","msg_rows":%s,"fibers":%s,"inflight_msgs":%s,"n_sims":%s,"m":24,"max_batch":%s,"warmup_ladder":[%s],"topology":"srv@0,gens@1,2,3,surplus@0(idle)"},"reading":{"command":"episodic_dps.sh %s","tool":"episodic_dps.sh","tag":"%s","leaf_rows_s":%s,"dps":%s,"decisions":%s,"leaves":%s,"wall_s":%s,"real_rows_per_fwd":%s,"server_util_pct":%s},"stamp":{"commit":"%s","tree":"%s"}}' \
    "$DRIVER" "$SERVER_IMPL" "$MSG_ROWS" "$K" "$INFLIGHT" "$NSIMS" "$MAXBATCH" "$WARMUP" "$*" \
    "${TLAB_TAG:-episodic_dps}" "$((lv/S))" "$((dec/S))" "$dec" "$lv" "$S" "${RPF:-null}" "${UTIL:-null}" \
    "$GIT_COMMIT" "$GIT_TREE" \
    | PYTHONPATH=throughput-lab "$PYBIN" throughput-lab/harness/exp_db.py --record 2>>"$LOG")
  [ -n "$RID" ] && echo "  [exp_db] recorded reading_id=$RID (tag=${TLAB_TAG:-episodic_dps})" \
                || echo "  [exp_db] not recorded — see $LOG"
fi
