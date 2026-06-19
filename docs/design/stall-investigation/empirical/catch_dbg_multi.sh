#!/bin/bash
# Debug catcher that, once wedged, takes 4 spaced snapshots of all worker threads +
# inflight_msgs/pool.inflight_ to distinguish a steady recv-block from transient compute.
# Bounded. Public Domain.
set -u
WT=/home/bork/w/vdc/1/chocofarm-wt-stall
BIN=$WT/cpp/build-dbg/chocofarm-wire-ab-bench
EP=$(cat /home/bork/w/vdc/chocobo/runs/stall-diag/endpoint2.txt)
OUT=/home/bork/w/vdc/chocobo/runs/stall-diag
WATCH=${WATCH:-30}
REPS=${REPS:-30}

DUMP='python
import gdb
for th in gdb.selected_inferior().threads():
    nm=th.name or "?"
    if "ZMQ" in nm: continue
    th.switch()
    f=gdb.newest_frame(); target=None; topname=f.name() or "?"
    while f is not None:
        n=f.name() or ""
        if "run_episodes_wire_pipelined" in n and "operator()" in n and "int" in n: target=f
        f=f.older()
    gdb.write("\nLWP %s TOP=%s\n" % (str(th.ptid[1]), topname))
    if target is None: continue
    target.select()
    for v in ("tid","inflight_msgs"):
        try: gdb.write("  %s=%s" % (v, gdb.parse_and_eval(v)))
        except Exception as e: gdb.write("  %s:err" % v)
    try: gdb.write("  pool.inflight_=%s" % gdb.parse_and_eval("pool.inflight_.size()"))
    except Exception: gdb.write("  pool.inflight_:err")
    try:
        n=int(gdb.parse_and_eval("submitted.size()")); s=0
        for i in range(n): s+=int(gdb.parse_and_eval("submitted[%d]"%i))
        gdb.write("  submitted_sum=%d/%d" % (s,n))
    except Exception: gdb.write("  submitted:err")
    gdb.write("\n")
'

for rep in $(seq 1 $REPS); do
  TOK="dm-n4-r${rep}-$(date +%s)"
  BT=$OUT/btm-rep${rep}.txt
  gdb -q -batch \
    -ex "set pagination off" -ex "set confirm off" -ex "cd $WT" \
    -ex "set environment CHOCO_FEATURE_LAYOUT=$WT/chocofarm/data/feature_layout.json" \
    -ex "handle SIGINT stop print nopass" \
    -ex "python
import threading,os,signal,time
threading.Thread(target=lambda:(time.sleep($WATCH),os.kill(os.getpid(),signal.SIGINT)),daemon=True).start()
" \
    -ex "run" \
    -ex "echo \n=== SNAP 1 ===\n" -ex "$DUMP" \
    -ex "continue &" -ex "shell sleep 3" -ex "interrupt" -ex "shell sleep 1" \
    -ex "echo \n=== SNAP 2 ===\n" -ex "$DUMP" \
    -ex "continue &" -ex "shell sleep 3" -ex "interrupt" -ex "shell sleep 1" \
    -ex "echo \n=== SNAP 3 ===\n" -ex "$DUMP" \
    -ex "thread apply all bt" \
    -ex "kill" \
    --args "$BIN" \
      --instance "$WT/chocofarm/data/instance.json" --faces "$WT/chocofarm/data/faces.json" \
      --endpoint "$EP" --run stall-diag2 --version 0 --res-token "$TOK" \
      --wire-mode pipelined-bucket --secs 1 --m 24 --n-sims 256 \
      --pool-threads 3 --pool-batch 64 --inflight-msgs 8 --trees-per-thread 4 \
    > "$BT" 2>&1
  if grep -q "RESULT: PASS" "$BT"; then
    echo "rep=$rep normal"
  elif grep -q "=== SNAP 3 ===" "$BT"; then
    echo "STALL CAUGHT rep=$rep -> $BT"; exit 0
  else
    echo "rep=$rep inconclusive -> $BT"
  fi
done
echo "no stall in $REPS reps"; exit 1
