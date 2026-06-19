#!/bin/bash
# Run the bench under gdb; if it does not finish within WATCH seconds, break in and
# dump all thread backtraces (that IS the stall). Loop reps until one wedges.
# Bounded: each gdb attempt wrapped in `timeout`. Public Domain.
set -u
WT=/home/bork/w/vdc/1/chocofarm-wt-stall
BIN=$WT/cpp/build/chocofarm-wire-ab-bench
EP=$(cat /home/bork/w/vdc/chocobo/runs/stall-diag/endpoint.txt)
OUT=/home/bork/w/vdc/chocobo/runs/stall-diag
WATCH=${WATCH:-40}
REPS=${REPS:-12}

for rep in $(seq 1 $REPS); do
  TOK="gdb-n4-r${rep}-$(date +%s)"
  BT=$OUT/bt-rep${rep}.txt
  : > "$BT"
  # gdb runs the inferior; a sidecar sends SIGINT to gdb after WATCH secs to break in.
  gdb -q -batch \
    -ex "set pagination off" \
    -ex "set confirm off" \
    -ex "cd $WT" \
    -ex "set environment CHOCO_FEATURE_LAYOUT=$WT/chocofarm/data/feature_layout.json" \
    -ex "handle SIGINT stop print nopass" \
    -ex "python import threading,os,signal,time
def _w():
    time.sleep($WATCH)
    os.kill(os.getpid(), signal.SIGINT)
threading.Thread(target=_w,daemon=True).start()" \
    -ex "run" \
    -ex "echo \n===== STOPPED (stall break-in or exit) =====\n" \
    -ex "info threads" \
    -ex "thread apply all bt" \
    -ex "kill" \
    --args "$BIN" \
      --instance "$WT/chocofarm/data/instance.json" --faces "$WT/chocofarm/data/faces.json" \
      --endpoint "$EP" --run stall-diag --version 0 --res-token "$TOK" \
      --wire-mode pipelined-bucket --secs 1 --m 24 --n-sims 256 \
      --pool-threads 3 --pool-batch 64 --inflight-msgs 8 --trees-per-thread 4 \
    > "$BT" 2>&1
  # Did it stall? Real stall = broke in on SIGINT (our watchdog), no RESULT, no SIGABRT.
  if grep -q "Program received signal SIGINT" "$BT" && ! grep -q "RESULT: PASS" "$BT" \
       && ! grep -q "SIGABRT" "$BT"; then
    echo "STALL CAUGHT at rep=$rep -> $BT"
    exit 0
  else
    echo "rep=$rep completed normally (no stall)"
  fi
done
echo "no stall in $REPS reps"
exit 1
