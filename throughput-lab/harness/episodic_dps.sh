#!/usr/bin/env bash
# throughput-lab/harness/episodic_dps.sh — the EPISODIC-static DPS baseline (the number any dynamic-control
#   attempt must beat; ADR-0009 measured). Runs the production-shape workload: real episodes (env stepped by
#   each executed action, --episodic) at the production search config (sims256/m24), --no-early-exit so
#   episodes run full-length, over the banked static optimum. The PROCESS TOPOLOGY is resolved at run time
#   from the hp SSOT (hp/spec.BANKED_TOPOLOGY, currently s2p1_g0.0-1.0-3.0_u2p0 = server@2 + gens@0,1,3 +
#   SCHED_IDLE surplus@2) — NOT pinned here; see the --banked-env eval below. The server is placed OFF the
#   housekeeping core 0 (a plain generator takes core 0): a paired test (tlab_finding #20) found the server
#   benefits from a clean core by ~+0.68% (marginal, one-sided p=0.045) — the generalizable lever is "server
#   not on core 0", core 2 is one representative. --msg-rows from $4.
#   Sums decisions across the 4 producers -> aggregate DPS; reports leaf-rows/s, LPD,
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
# Banked PROCESS TOPOLOGY — resolved from the hp SSOT (hp/spec.BANKED_TOPOLOGY), NOT hand-pinned here.
# topology_enum --banked-env derives SRV_CORE / GEN_CORES / SURPLUS_CORE / *_WRAP_ARGS / TOPOLOGY_STR from
# the one banked config_id (validated against the live enumeration; fails loud on a stale id — ADR-0002).
# This kills the prior drift: the taskset literals + the duplicated provenance string had to be hand-synced.
# (Assign-then-eval so the resolver's non-zero exit propagates — `eval "$(...)"` would mask it.)
_BANKED_ENV="$(PYTHONPATH=throughput-lab:throughput-lab/harness "$PYBIN" \
  throughput-lab/harness/topology_enum.py --banked-env)" \
  || { echo "FATAL: could not resolve the banked topology from the hp SSOT" >&2; exit 1; }
eval "$_BANKED_ENV"
# Banked PRODUCER/SERVER OPERATING POINT — resolved from the hp SSOT (hp/spec.BANKED_STATIC), NOT pinned
# here. One home for fibers/msg-rows/inflight/driver/seconds/n-sims/m/max-batch/warmup; the arg defaults
# below DERIVE from these BANKED_* vars (the override args $1..$6 still win, for sweeps). This kills the
# cross-harness defaults drift — notably --seconds (episodic ran 14, topology_sweep 5/10), the gap that
# confounded the run-length comparison (finding #21; banked at 10).
_BANKED_STATIC="$(PYTHONPATH=throughput-lab "$PYBIN" -m hp.cli --banked-static-env)" \
  || { echo "FATAL: could not resolve the banked static operating point from the hp SSOT" >&2; exit 1; }
eval "$_BANKED_STATIC"
# DRIVER defaults to greedy. NOTE: the earlier "+31% clean win" was RETRACTED (an unstamped single-session
# reading; journey doc Witness 2, commit 2ac1cef). The stamped quiet-box 2x2 shows the driver is
# REGIME-DEPENDENT: ~+4.5% and NOT clean at the pad-tax 4096 ladder (server compute-bound, no idle to
# overlap), but a CLEAN +15% at the lean ladder below (greedy MIN > round-sync MAX) where the fast server
# is coupling/RTT-limited. Greedy still wins-or-ties everywhere, so it stays the default. Pass `round-sync`
# as $5 for the baseline arm.
# DEFAULT K=1024 MSG_ROWS=256 (banked by the HP SWEEP, DB finding #17): the producer-HP optimum on the single-
# thread one-pull stack. K is the big lever -- a bigger fiber pool fills bigger server batches; K=1024 is the
# KNEE (256->1024 = +~25%, ~100k->~125k; K=2048/4096 plateau). MSG_ROWS=256 (max coalescing -> fills the 256
# max-batch) beats 128, but ONLY with K>=1024 to supply it (K=256/MSG=256 STARVES the server, util 47%). The
# wall is the ~69% single-core serve-loop ceiling (findings #10/#11) -> ~125k is near the single-core max; more
# throughput needs a 2nd core, not HP. (Earlier K=256/MSG=128 was Witness 3's banked best; superseded.) Pass
# K/MSG_ROWS as $1/$4 for other arms; the 512-ladder (WARMUP/MAXBATCH) is +1-2% only.
K="${1:-$BANKED_FIBERS}"; S="${2:-$BANKED_SECONDS}"; NSIMS="${3:-$BANKED_N_SIMS}"; MSG_ROWS="${4:-$BANKED_MSG_ROWS}"; DRIVER="${5:-$BANKED_DRIVER}"; INFLIGHT="${6:-$BANKED_INFLIGHT}"
# SINGLE_THREAD defaults ON (banked, finding #5: single-thread serve path +7.8% vs the two-thread split on one
# pinned core). Unset => single-thread; pass SINGLE_THREAD= (explicit empty) for the two-thread arm.
SINGLE_THREAD="${SINGLE_THREAD-1}"
# Server bucket ladder (the snap-up policy; server reads the warmed set back from the forward -> one home).
# DEFAULT = {64,256}/max-256 (banked): with K=1024 + MSG_ROWS=256 the producer pins the 256 bucket every
# forward (rows/fwd=256), so CAPPING at 256 (no 512) forbids the wasteful 512-spill (the HP sweep, finding #17,
# confirmed the 512-ladder is +1-2% only — the maintainer's "last lever", minor). max-256 is the well-filled
# efficient regime. (Earlier Witness 2 banked the lean {64,256,512}/512 ladder over the old
# [1,8,64,512,4096] pad-tax ladder -- +25-37%; this refines it.) Override WARMUP/MAXBATCH for other arms.
WARMUP="${WARMUP:-$BANKED_WARMUP}"; MAXBATCH="${MAXBATCH:-$BANKED_MAX_BATCH}"
# FORWARD selects the server compute backend: jax (default; XLA-jit + bucket ladder) | numpy (forward_core
# in numpy -- NO XLA per-call dispatch overhead, NO pad). The A/B arm for the XLA-overhead hypothesis; numpy
# wants single-thread BLAS (OMP_NUM_THREADS=1) on the one pinned core.
FORWARD="${FORWARD:-jax}"
SRVENV=""; [ "$FORWARD" = numpy ] && SRVENV="OMP_NUM_THREADS=1"
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
# NOTE (load-bearing — do NOT revert to a bare `$SRVENV PYTHONPATH=... cmd` prefix): bash does assignment-
# prefix recognition at PARSE time on LITERAL tokens. A leading VARIABLE-expansion word (`$SRVENV`) is NOT a
# literal `name=value`, so it terminates the assignment scan — the following literal `PYTHONPATH=...` then
# lands in COMMAND-WORD position and bash runs it as a command ("PYTHONPATH=throughput-lab: command not found"
# -> the server never starts -> the READY-wait hangs). This silently broke every jax/numpy run after the
# $SRVENV prefix was added (and was the real cause of the "4-gen wedge" once mis-read as a sandbox kill).
# `env` fixes it: with env the words are ARGS (VAR=VAL pairs env applies), not a shell assignment prefix, so
# an empty or non-empty $SRVENV both work. Keep the `env`.
# NET (env, a path to a real AZ .npz checkpoint, jax only) serves the REAL trained net (Gate B) instead of a
# random one -> AZ-relevant metrics (the real net's LPD/throughput, ~+13% vs random; finding #15). Empty = random.
env $SRVENV PYTHONPATH=throughput-lab PYTHONUNBUFFERED=1 taskset -c "$SRV_CORE" "$PYBIN" -m server --bind "$EP" \
  --in-dim 241 --n-actions 65 --hidden 256 --max-batch "$MAXBATCH" --warmup "$WARMUP" \
  --poll-timeout-ms 50 ${SINGLE_THREAD:+--single-thread} --forward "$FORWARD" ${NET:+--net "$NET"} >"$LOG" 2>&1 & SRV=$!
cleanup(){ kill -INT "$SRV" 2>/dev/null||true; sleep 1; kill "$SRV" 2>/dev/null||true; rm -f "${EP#ipc://}"; }
trap cleanup EXIT
for _ in $(seq 1 240); do grep -q READY "$LOG" && break; sleep 0.5; done
grep -q READY "$LOG" || { echo "server never READY"; cat "$LOG"; exit 1; }

G(){ taskset -c "$1" ${2:-} "$PROD" --instance "$INST" --faces "$FACES" --endpoint "$EP" \
     --threads 1 --fibers "$K" --msg-rows "$MSG_ROWS" --inflight-msgs "$INFLIGHT" \
     --episodic --no-early-exit --driver "$DRIVER" \
     --seconds "$S" --n-sims "$NSIMS" --m "$BANKED_M"; }
TMP="$(mktemp -d)"
# banked topology resolved from the hp SSOT (the --banked-env eval above): server@$SRV_CORE off the
# housekeeping core, 3 working gens on $GEN_CORES, SCHED_IDLE surplus@$SURPLUS_CORE sharing the server core.
# The *_WRAP_ARGS carry the per-role sched_wrap policy; we prepend our own $W (sched_wrap path).
read -r g1 g2 g3 <<<"$GEN_CORES"
GEN_WRAP="${GEN_WRAP_ARGS:+$W $GEN_WRAP_ARGS --}"; SURPLUS_WRAP="${SURPLUS_WRAP_ARGS:+$W $SURPLUS_WRAP_ARGS --}"
G "$g1" "$GEN_WRAP" >"$TMP/e1" 2>&1 & a=$!; G "$g2" "$GEN_WRAP" >"$TMP/e2" 2>&1 & b=$!; G "$g3" "$GEN_WRAP" >"$TMP/e3" 2>&1 & c=$!
G "$SURPLUS_CORE" "$SURPLUS_WRAP" >"$TMP/es" 2>&1 & d=$!
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
SERVER_IMPL="tlab-server.py:$([ -n "${SINGLE_THREAD:-}" ] && echo single-thread || echo two-thread):$FORWARD$([ -n "${NET:-}" ] && echo :realnet)"
echo "EPISODIC-STATIC [commit=$GIT_COMMIT tree=$GIT_TREE] (sims${NSIMS}/m24, no-early-exit, 3 gens + IDLE surplus, driver=$DRIVER inflight=$INFLIGHT, msg-rows=$MSG_ROWS, ladder=[$WARMUP] max-batch=$MAXBATCH, K=$K, server=$SERVER_IMPL, ${S}s)"
echo "  decisions=$dec  ->  DPS = $((dec/S))"
echo "  leaves=$lv  ->  leaf-rows/s = $((lv/S))   LPD ~= $((lv/(dec>0?dec:1)))"
grep -E 'served|forwards|compute-busy|latency' "$LOG"
# Auto-persist this reading to throughput_research (the measurement side of "commit as we go"). Best-effort:
# exp_db --record is loud-but-non-fatal (a DB blip dumps the reading under ~/w/vdc, never fails the run).
# Set TLAB_NO_DB=1 to skip; TLAB_TAG to label a cohort.
if [ "${TLAB_NO_DB:-0}" != "1" ]; then
  RID=$(printf '{"config":{"driver":"%s","server_impl":"%s","producer_bin":"tlab-real-producer","msg_rows":%s,"fibers":%s,"inflight_msgs":%s,"n_sims":%s,"m":%s,"max_batch":%s,"warmup_ladder":[%s],"topology":"%s"},"reading":{"command":"episodic_dps.sh %s","tool":"episodic_dps.sh","tag":"%s","leaf_rows_s":%s,"dps":%s,"decisions":%s,"leaves":%s,"wall_s":%s,"real_rows_per_fwd":%s,"server_util_pct":%s},"stamp":{"commit":"%s","tree":"%s"}}' \
    "$DRIVER" "$SERVER_IMPL" "$MSG_ROWS" "$K" "$INFLIGHT" "$NSIMS" "$BANKED_M" "$MAXBATCH" "$WARMUP" "$TOPOLOGY_STR" "$*" \
    "${TLAB_TAG:-episodic_dps}" "$((lv/S))" "$((dec/S))" "$dec" "$lv" "$S" "${RPF:-null}" "${UTIL:-null}" \
    "$GIT_COMMIT" "$GIT_TREE" \
    | PYTHONPATH=throughput-lab "$PYBIN" throughput-lab/harness/exp_db.py --record 2>>"$LOG")
  [ -n "$RID" ] && echo "  [exp_db] recorded reading_id=$RID (tag=${TLAB_TAG:-episodic_dps})" \
                || echo "  [exp_db] not recorded — see $LOG"
fi
