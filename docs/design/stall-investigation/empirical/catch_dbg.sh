#!/bin/bash
# Debug-build stall catcher with local-variable inspection (inflight_msgs, submitted sum,
# pool.inflight_.size). Bounded. Public Domain.
set -u
WT=/home/bork/w/vdc/1/chocofarm-wt-stall
BIN=$WT/cpp/build-dbg/chocofarm-wire-ab-bench
EP=$(cat /home/bork/w/vdc/chocobo/runs/stall-diag/endpoint2.txt)
OUT=/home/bork/w/vdc/chocobo/runs/stall-diag
WATCH=${WATCH:-30}
REPS=${REPS:-30}

for rep in $(seq 1 $REPS); do
  TOK="dbg-n4-r${rep}-$(date +%s)"
  BT=$OUT/btd-rep${rep}.txt
  gdb -q -batch \
    -ex "set pagination off" -ex "set confirm off" -ex "cd $WT" \
    -ex "set environment CHOCO_FEATURE_LAYOUT=$WT/chocofarm/data/feature_layout.json" \
    -ex "handle SIGINT stop print nopass" \
    -ex "python
import threading,os,signal,time
threading.Thread(target=lambda:(time.sleep($WATCH),os.kill(os.getpid(),signal.SIGINT)),daemon=True).start()
" \
    -ex "run" \
    -ex "echo \n===== BREAK-IN =====\n" \
    -ex "info threads" \
    -ex "python
import gdb
for th in gdb.selected_inferior().threads():
    nm = th.name or '?'
    if 'ZMQ' in nm: continue
    th.switch()
    f = gdb.newest_frame(); target=None
    while f is not None:
        n=f.name() or ''
        if 'run_episodes_wire_pipelined' in n and 'operator()' in n and 'int' in n: target=f
        f=f.older()
    gdb.write('\n--- LWP %s top=%s ---\n' % (str(th.ptid[1]), (gdb.newest_frame().name() or '?')))
    if target is None: gdb.write('  (no worker frame)\n'); continue
    target.select()
    for v in ('tid','inflight_msgs','D','K'):
        try: gdb.write('  %s = %s\n' % (v, gdb.parse_and_eval(v)))
        except Exception as e: gdb.write('  %s : %s\n' % (v,e))
    try:
        sub=gdb.parse_and_eval('submitted')
        n=int(gdb.parse_and_eval('submitted.size()'))
        s=0
        for i in range(n): s+= int(gdb.parse_and_eval('submitted[%d]'%i))
        gdb.write('  submitted.size=%d sum(submitted)=%d\n' % (n,s))
    except Exception as e: gdb.write('  submitted : %s\n' % e)
    try: gdb.write('  pool.inflight_.size = %s\n' % gdb.parse_and_eval('pool.inflight_.size()'))
    except Exception as e: gdb.write('  pool.inflight_ : %s\n' % e)
" \
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
  elif grep -q "recv_corr_payload" "$BT"; then
    echo "STALL CAUGHT (recv-blocked) rep=$rep -> $BT"; exit 0
  elif grep -q "===== BREAK-IN =====" "$BT"; then
    echo "rep=$rep wedged-but-compute-snapshot -> $BT (retrying for recv state)"
  else
    echo "rep=$rep inconclusive -> $BT"
  fi
done
echo "no stall in $REPS reps"; exit 1
