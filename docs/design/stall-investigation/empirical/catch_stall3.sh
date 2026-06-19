#!/bin/bash
# Native-speed stall catcher v3. gdb runs inferior in background (async). A SEPARATE
# watchdog process SIGINTs the gdb pid after WATCH secs (gdb turns that into a sync
# stop of the inferior), then gdb dumps all thread backtraces. Bounded. Public Domain.
set -u
WT=/home/bork/w/vdc/1/chocofarm-wt-stall
BIN=$WT/cpp/build/chocofarm-wire-ab-bench
EP=$(cat /home/bork/w/vdc/chocobo/runs/stall-diag/endpoint.txt)
OUT=/home/bork/w/vdc/chocobo/runs/stall-diag
WATCH=${WATCH:-22}
REPS=${REPS:-20}

for rep in $(seq 1 $REPS); do
  TOK="g3-n4-r${rep}-$(date +%s)"
  BT=$OUT/bt3-rep${rep}.txt
  # Run gdb; its Python watchdog sends SIGINT to gdb's OWN pid (interrupts inferior),
  # while gdb stays in synchronous mode (no run &) so the stop is observed in-order.
  gdb -q -batch \
    -ex "set pagination off" \
    -ex "set confirm off" \
    -ex "cd $WT" \
    -ex "set environment CHOCO_FEATURE_LAYOUT=$WT/chocofarm/data/feature_layout.json" \
    -ex "handle SIGINT stop print nopass" \
    -ex "python
import threading, os, signal, time
def _wd():
    time.sleep($WATCH)
    os.kill(os.getpid(), signal.SIGINT)
threading.Thread(target=_wd, daemon=True).start()
" \
    -ex "run" \
    -ex "echo \n===== BREAK-IN =====\n" \
    -ex "info threads" \
    -ex "thread apply all bt" \
    -ex "kill" \
    --args "$BIN" \
      --instance "$WT/chocofarm/data/instance.json" --faces "$WT/chocofarm/data/faces.json" \
      --endpoint "$EP" --run stall-diag --version 0 --res-token "$TOK" \
      --wire-mode pipelined-bucket --secs 1 --m 24 --n-sims 256 \
      --pool-threads 3 --pool-batch 64 --inflight-msgs 8 --trees-per-thread 4 \
    > "$BT" 2>&1
  if grep -q "RESULT: PASS" "$BT"; then
    echo "rep=$rep completed normally (no stall)"
  elif grep -q "===== BREAK-IN =====" "$BT" && grep -q "Thread 4" "$BT"; then
    echo "STALL CAUGHT at rep=$rep -> $BT"
    exit 0
  else
    echo "rep=$rep inconclusive (rc/other) -> $BT"
  fi
done
echo "no stall in $REPS reps"
exit 1
