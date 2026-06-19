#!/bin/bash
# Catch the stall and dump, per worker thread, the inflight_msgs counter and the pool's
# inflight_ map size (how many corr-ids are outstanding) — to compare producer-believed
# outstanding vs server-received. Bounded. Public Domain.
set -u
WT=/home/bork/w/vdc/1/chocofarm-wt-stall
BIN=$WT/cpp/build/chocofarm-wire-ab-bench
EP=$(cat /home/bork/w/vdc/chocobo/runs/stall-diag/endpoint.txt)
OUT=/home/bork/w/vdc/chocobo/runs/stall-diag
WATCH=${WATCH:-22}
REPS=${REPS:-20}

for rep in $(seq 1 $REPS); do
  TOK="gv-n4-r${rep}-$(date +%s)"
  BT=$OUT/btv-rep${rep}.txt
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
inf = gdb.selected_inferior()
for th in inf.threads():
    try:
        nm = th.name or '?'
        if 'ZMQ' in nm: continue
        th.switch()
        gdb.write('\n--- LWP %s (%s) ---\n' % (str(th.ptid[1]), nm))
        f = gdb.newest_frame()
        target = None
        while f is not None:
            fn = f.name() or ''
            if 'run_episodes_wire_pipelined' in fn and 'operator()(int)' in fn:
                target = f
            f = f.older()
        if target is not None:
                target.select()
                for v in ('inflight_msgs','D','K','my_msgs','tid'):
                    try: gdb.write('  %s = %s\n' % (v, str(gdb.parse_and_eval(v))))
                    except Exception as e: gdb.write('  %s : %s\n' % (v, e))
                try: gdb.write('  pool.inflight_.size = %s\n' % str(gdb.parse_and_eval('pool.inflight_.size()')))
                except Exception as e: gdb.write('  pool.inflight_.size : %s\n' % e)
    except Exception as e:
        gdb.write('thread err: %s\n' % e)
" \
    -ex "thread apply all bt" \
    -ex "kill" \
    --args "$BIN" \
      --instance "$WT/chocofarm/data/instance.json" --faces "$WT/chocofarm/data/faces.json" \
      --endpoint "$EP" --run stall-diag --version 0 --res-token "$TOK" \
      --wire-mode pipelined-bucket --secs 1 --m 24 --n-sims 256 \
      --pool-threads 3 --pool-batch 64 --inflight-msgs 8 --trees-per-thread 4 \
    > "$BT" 2>&1
  if grep -q "RESULT: PASS" "$BT"; then
    echo "rep=$rep normal"
  elif grep -q "===== BREAK-IN =====" "$BT" && grep -q "recv_batch" "$BT"; then
    echo "STALL CAUGHT rep=$rep -> $BT"; exit 0
  else
    echo "rep=$rep inconclusive -> $BT"
  fi
done
echo "no stall in $REPS reps"; exit 1
