#!/bin/bash
# Native-speed stall catcher: gdb runs the inferior in the BACKGROUND (run &) so it
# executes without per-event mediation, then after WATCH secs `interrupt`s and dumps
# all thread backtraces. Loop reps until one wedges. Bounded per attempt. Public Domain.
set -u
WT=/home/bork/w/vdc/1/chocofarm-wt-stall
BIN=$WT/cpp/build/chocofarm-wire-ab-bench
EP=$(cat /home/bork/w/vdc/chocobo/runs/stall-diag/endpoint.txt)
OUT=/home/bork/w/vdc/chocobo/runs/stall-diag
WATCH=${WATCH:-25}
REPS=${REPS:-20}

for rep in $(seq 1 $REPS); do
  TOK="g2-n4-r${rep}-$(date +%s)"
  BT=$OUT/bt2-rep${rep}.txt
  cat > /tmp/gdbcmds.$$ <<EOF
set pagination off
set confirm off
cd $WT
set environment CHOCO_FEATURE_LAYOUT=$WT/chocofarm/data/feature_layout.json
set target-async on
set non-stop off
run &
shell sleep $WATCH
interrupt
shell sleep 3
echo \n===== BREAK-IN (post-WATCH interrupt) =====\n
info threads
thread apply all bt
info threads
thread apply all bt
kill
quit
EOF
  gdb -q -batch -x /tmp/gdbcmds.$$ \
    --args "$BIN" \
      --instance "$WT/chocofarm/data/instance.json" --faces "$WT/chocofarm/data/faces.json" \
      --endpoint "$EP" --run stall-diag --version 0 --res-token "$TOK" \
      --wire-mode pipelined-bucket --secs 1 --m 24 --n-sims 256 \
      --pool-threads 3 --pool-batch 64 --inflight-msgs 8 --trees-per-thread 4 \
    > "$BT" 2>&1
  rm -f /tmp/gdbcmds.$$
  if grep -q "RESULT: PASS" "$BT"; then
    echo "rep=$rep completed normally (no stall)"
    rm -f "$BT"
  else
    echo "STALL CAUGHT at rep=$rep -> $BT"
    exit 0
  fi
done
echo "no stall in $REPS reps"
exit 1
