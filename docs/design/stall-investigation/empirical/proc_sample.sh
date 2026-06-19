#!/bin/bash
# Launch the bench bare (native speed). When it wedges, sample each thread's kernel
# state + wchan from /proc over several seconds (no ptrace). Distinguishes a steady
# recv-block (state S, wchan ep_poll/poll) from compute (state R). Bounded. Public Domain.
set -u
WT=/home/bork/w/vdc/1/chocofarm-wt-stall
BIN=$WT/cpp/build/chocofarm-wire-ab-bench   # RELEASE build (steady-state timing)
EP=$(cat /home/bork/w/vdc/chocobo/runs/stall-diag/endpoint2.txt)
OUT=/home/bork/w/vdc/chocobo/runs/stall-diag
REPS=${REPS:-15}
HANGAFTER=${HANGAFTER:-16}   # if no RESULT after this many secs, treat as wedged & sample

cd "$WT"
export CHOCO_FEATURE_LAYOUT=$WT/chocofarm/data/feature_layout.json
for rep in $(seq 1 $REPS); do
  LOG=$OUT/proc-rep${rep}.log
  : > "$LOG"
  taskset -c 1,2,3 "$BIN" \
    --instance "$WT/chocofarm/data/instance.json" --faces "$WT/chocofarm/data/faces.json" \
    --endpoint "$EP" --run stall-diag2 --version 0 --res-token "proc-r$rep" \
    --wire-mode pipelined-bucket --secs 1 --m 24 --n-sims 256 \
    --pool-threads 3 --pool-batch 64 --inflight-msgs 8 --trees-per-thread 4 > "$LOG" 2>&1 &
  PID=$!
  # wait up to HANGAFTER for RESULT; if it appears, normal.
  ok=0
  for i in $(seq 1 $HANGAFTER); do
    if grep -q "RESULT: PASS" "$LOG" 2>/dev/null; then ok=1; break; fi
    if ! kill -0 $PID 2>/dev/null; then ok=2; break; fi
    sleep 1
  done
  if [ $ok -eq 1 ]; then
    echo "rep=$rep normal"; wait $PID 2>/dev/null; continue
  fi
  # WEDGED: sample /proc 4x over ~6s
  SMP=$OUT/procsmp-rep${rep}.txt; : > "$SMP"
  echo "WEDGED pid=$PID — sampling /proc" | tee -a "$SMP"
  for s in 1 2 3 4; do
    echo "----- sample $s ($(date +%T)) -----" >> "$SMP"
    for taskdir in /proc/$PID/task/*; do
      t=$(basename "$taskdir")
      st=$(awk '{print $3}' "$taskdir/stat" 2>/dev/null)
      wc=$(cat "$taskdir/wchan" 2>/dev/null)
      cm=$(cat "$taskdir/comm" 2>/dev/null)
      echo "  tid=$t comm=$cm state=$st wchan=$wc" >> "$SMP"
    done
    sleep 2
  done
  echo "--- server2 tail ---" >> "$SMP"
  tail -3 "$OUT/server2.log" >> "$SMP"
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "STALL SAMPLED rep=$rep -> $SMP"
  exit 0
done
echo "no stall in $REPS reps"; exit 1
