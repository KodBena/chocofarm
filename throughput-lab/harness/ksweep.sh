#!/usr/bin/env bash
# throughput-lab/harness/ksweep.sh — controlled banked-producer fiber-count (K) saturation sweep.
#
#   PURPOSE (a MEASUREMENT, not a fix): find the SMALLEST per-thread fiber count K at which SERVER-SIDE
#   throughput plateaus at the banked 4-up operating point — the re-bank candidate (the smallest K that
#   holds full throughput, freeing the per-producer resident the larger K costs). Substantiates the
#   banked-default change with controlled data (ADR-0009); asserts nothing the data does not witness.
#
#   The banked point is resolved FROM THE hp SSOT (hp/spec.BANKED_TOPOLOGY + BANKED_STATIC), exactly as
#   episodic_dps.sh does — only --fibers (K) is overridden per cell. So the sweep moves K and holds
#   EVERYTHING ELSE at the bank (server@2 single-thread + gens@0,1,3 + SCHED_IDLE surplus@2; greedy/
#   episodic, msg-rows 256, inflight 8, n_sims 256, m 24, max-batch 256, warmup 64,256, 10s).
#
#   DENOMINATOR-HONEST METRIC: server-side rows/s = (server `served R rows in W.WWWs`) — the server's own
#   serve window, NOT the producer-side decisions. The single-thread jax server IS the bottleneck, so
#   server rows/s is the throughput K is being chosen against.
#
#   ROBUST DISCIPLINE (robust-benchmark-statistics): INTERLEAVED replicates with the K order ROTATED per
#   replicate, so slow drift cannot masquerade as a K-trend. Per-producer PEAK RSS captured (/usr/bin/time
#   -v on each producer) so the memory saved by a smaller K is quantified. Every cell stamped with the git
#   commit + clean/DIRTY tree (ADR-0011) — a reading unpinnable to a code state is unattributable.
#
#   Writes one JSONL row per cell to $OUTDIR/cells.jsonl (raw, immutable) + per-cell server/producer logs.
#   Does NOT auto-record to exp_db (these are sweep cells, aggregated post-hoc; the recommendation row is).
#
#   Usage:  KS="128 256 384 512 768 1024" REPS=5 SECONDS_PER=10 \
#             OUTDIR=~/w/vdc/chocobo/runs/tlab/ksweep-... bash throughput-lab/harness/ksweep.sh
#   Public Domain (The Unlicense).
set -uo pipefail
cd /home/bork/w/vdc/1/chocofarm
PYBIN=/home/bork/w/vdc/venvs/generic/bin/python
PROD=throughput-lab/cpp/build/tlab-real-producer; W=throughput-lab/cpp/build/sched_wrap
TIME=/usr/bin/time
INST=chocofarm/data/instance.json; FACES=chocofarm/data/faces.json

KS="${KS:-128 256 384 512 768 1024}"
REPS="${REPS:-5}"
SECONDS_PER="${SECONDS_PER:-10}"
OUTDIR="${OUTDIR:-$HOME/w/vdc/chocobo/runs/tlab/ksweep-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUTDIR/cells"
CELLS="$OUTDIR/cells.jsonl"

for b in "$PROD" "$W"; do [ -x "$b" ] || { echo "missing $b (build -DTLAB_REAL_GENERATOR=ON)"; exit 2; }; done
[ -x "$TIME" ] || { echo "missing $TIME (GNU time, for peak RSS)"; exit 2; }

# Banked PROCESS TOPOLOGY + STATIC operating point — resolved from the hp SSOT (one home; ADR-0012 P1).
_BANKED_ENV="$(PYTHONPATH=throughput-lab:throughput-lab/harness "$PYBIN" \
  throughput-lab/harness/topology_enum.py --banked-env)" \
  || { echo "FATAL: could not resolve the banked topology from the hp SSOT" >&2; exit 1; }
eval "$_BANKED_ENV"
_BANKED_STATIC="$(PYTHONPATH=throughput-lab "$PYBIN" -m hp.cli --banked-static-env)" \
  || { echo "FATAL: could not resolve the banked static operating point from the hp SSOT" >&2; exit 1; }
eval "$_BANKED_STATIC"

MSG_ROWS="$BANKED_MSG_ROWS"; INFLIGHT="$BANKED_INFLIGHT"; DRIVER="$BANKED_DRIVER"
NSIMS="$BANKED_N_SIMS"; MVAL="$BANKED_M"; WARMUP="$BANKED_WARMUP"; MAXBATCH="$BANKED_MAX_BATCH"
SINGLE_THREAD=1   # banked (finding #5)

GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_TREE="$(test -z "$(git status --porcelain 2>/dev/null)" && echo clean || echo DIRTY)"

{
  echo "# ksweep manifest"
  echo "stamp_utc:   $(date -u +%FT%TZ)"
  echo "commit:      $GIT_COMMIT  tree=$GIT_TREE"
  echo "host:        $(nproc) vCPU"
  echo "topology:    $TOPOLOGY_STR  (banked id=$BANKED_CONFIG_ID)"
  echo "static:      driver=$DRIVER msg_rows=$MSG_ROWS inflight=$INFLIGHT n_sims=$NSIMS m=$MVAL max_batch=$MAXBATCH warmup=$WARMUP single_thread=$SINGLE_THREAD"
  echo "sweep:       K in {$KS}  reps=$REPS  seconds_per=$SECONDS_PER (server-side rows/s metric)"
  echo "interleave:  K order ROTATED per replicate (drift cannot masquerade as a K-trend)"
} | tee "$OUTDIR/MANIFEST.txt"

read -r g1 g2 g3 <<<"$GEN_CORES"
GEN_WRAP="${GEN_WRAP_ARGS:+$W $GEN_WRAP_ARGS --}"; SURPLUS_WRAP="${SURPLUS_WRAP_ARGS:+$W $SURPLUS_WRAP_ARGS --}"

# One producer, wrapped in /usr/bin/time -v so its PEAK RSS is captured ($2 = an optional sched_wrap prefix).
# /usr/bin/time writes its -v report to its OWN stderr; we route per-producer stderr to a file, parse RSS.
G(){ local core="$1" wrap="${2:-}" rssf="$3"
     taskset -c "$core" $wrap "$TIME" -v "$PROD" --instance "$INST" --faces "$FACES" --endpoint "$EP" \
       --threads 1 --fibers "$K" --msg-rows "$MSG_ROWS" --inflight-msgs "$INFLIGHT" \
       --episodic --no-early-exit --driver "$DRIVER" \
       --seconds "$SECONDS_PER" --n-sims "$NSIMS" --m "$MVAL" 2>"$rssf"; }

run_cell(){    # args: K rep
  K="$1"; local rep="$2"
  local tag="K${K}_r${rep}"
  local cdir="$OUTDIR/cells/$tag"; mkdir -p "$cdir"
  EP="ipc:///tmp/tlab-ksweep-$$-$tag.sock"; rm -f "${EP#ipc://}"
  local LOG="$cdir/server.log"
  env PYTHONPATH=throughput-lab PYTHONUNBUFFERED=1 taskset -c "$SRV_CORE" "$PYBIN" -m server --bind "$EP" \
    --in-dim 241 --n-actions 65 --hidden 256 --max-batch "$MAXBATCH" --warmup "$WARMUP" \
    --poll-timeout-ms 50 ${SINGLE_THREAD:+--single-thread} --forward jax >"$LOG" 2>&1 & local SRV=$!
  local ok=0
  for _ in $(seq 1 240); do grep -q READY "$LOG" && { ok=1; break; }; sleep 0.5; done
  if [ "$ok" != 1 ]; then echo "  $tag: server never READY"; cat "$LOG"; kill "$SRV" 2>/dev/null; return 1; fi

  G "$g1" "$GEN_WRAP" "$cdir/rss_g1" >"$cdir/e1" 2>>"$cdir/e1" & local a=$!
  G "$g2" "$GEN_WRAP" "$cdir/rss_g2" >"$cdir/e2" 2>>"$cdir/e2" & local b=$!
  G "$g3" "$GEN_WRAP" "$cdir/rss_g3" >"$cdir/e3" 2>>"$cdir/e3" & local c=$!
  G "$SURPLUS_CORE" "$SURPLUS_WRAP" "$cdir/rss_es" >"$cdir/es" 2>>"$cdir/es" & local d=$!
  wait "$a" "$b" "$c" "$d"
  kill -INT "$SRV" 2>/dev/null
  for _ in $(seq 1 30); do grep -q 'compute-busy' "$LOG" && break; sleep 0.5; done
  kill "$SRV" 2>/dev/null; rm -f "${EP#ipc://}"

  # Server-side throughput: `served R rows in W.WWWs` -> rows/s = R/W (the denominator-honest metric).
  local line rows wall rps
  line="$(grep -oE 'served [0-9]+ requests / [0-9]+ rows in [0-9.]+s' "$LOG" | head -1)"
  rows="$(echo "$line" | grep -oE '/ [0-9]+ rows' | grep -oE '[0-9]+')"
  wall="$(echo "$line" | grep -oE 'in [0-9.]+s' | grep -oE '[0-9.]+')"
  rps="$(grep -oE '[0-9,]+ rows/s' "$LOG" | head -1 | tr -d ', ' | grep -oE '[0-9]+')"
  local util batch
  util="$(grep -oE '\([0-9.]+% of wall\)' "$LOG" | grep -oE '[0-9.]+' | head -1)"
  batch="$(grep -oE 'mean batch [0-9.]+' "$LOG" | grep -oE '[0-9.]+' | head -1)"

  # Per-producer peak RSS (KiB) from each /usr/bin/time -v report.
  rss(){ grep -oE 'Maximum resident set size \(kbytes\): [0-9]+' "$1" 2>/dev/null | grep -oE '[0-9]+$'; }
  local r1 r2 r3 rs
  r1="$(rss "$cdir/rss_g1")"; r2="$(rss "$cdir/rss_g2")"; r3="$(rss "$cdir/rss_g3")"; rs="$(rss "$cdir/rss_es")"

  # Producer-side decisions (secondary cross-check only).
  local dec=0
  for f in e1 e2 e3 es; do
    local D; D=$(grep -oE 'decisions=[0-9]+' "$cdir/$f" | head -1 | cut -d= -f2); dec=$((dec+${D:-0}))
  done
  # mean peak RSS across the 4 producers (KiB), guarded.
  local rmean=""
  if [ -n "$r1" ] && [ -n "$r2" ] && [ -n "$r3" ] && [ -n "$rs" ]; then
    rmean=$(( (r1 + r2 + r3 + rs) / 4 ))
  fi

  printf '{"k":%s,"rep":%s,"server_rows_s":%s,"server_rows":%s,"server_wall_s":%s,"util_pct":%s,"mean_batch":%s,"rss_kib":{"g1":%s,"g2":%s,"g3":%s,"surplus":%s},"rss_mean_kib":%s,"decisions":%s,"commit":"%s","tree":"%s"}\n' \
    "$K" "$rep" "${rps:-null}" "${rows:-null}" "${wall:-null}" "${util:-null}" "${batch:-null}" \
    "${r1:-null}" "${r2:-null}" "${r3:-null}" "${rs:-null}" "${rmean:-null}" "$dec" "$GIT_COMMIT" "$GIT_TREE" \
    | tee -a "$CELLS"
  echo "  $tag: server_rows/s=${rps:-?} util=${util:-?}% batch=${batch:-?} rss_mean=${rmean:-?}KiB dec=$dec"
}

# Interleaved: outer = replicate; inner = K, ROTATED left by `rep` each replicate.
ks_arr=($KS); n=${#ks_arr[@]}
echo "[ksweep] $n K-levels x $REPS reps = $((n*REPS)) cells -> $OUTDIR"
for rep in $(seq 1 "$REPS"); do
  echo "=== replicate $rep/$REPS ==="
  for i in $(seq 0 $((n-1))); do
    idx=$(( (i + rep - 1) % n ))
    run_cell "${ks_arr[$idx]}" "$rep"
  done
done
echo "[ksweep] done -> $CELLS"
