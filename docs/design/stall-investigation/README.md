# Stall investigation — artifacts (formal + empirical)

Reproducible code + key evidence behind the diagnosis docs (`../cpp-eval-wire-formal-diagnosis.md`; the
empirical verdict in `../cpp-eval-transport-adapter.md` §6/§7). The "stall" of the `pipelined-bucket` wire
driver at moderate overcommit was diagnosed by TWO independent arms as a metastable throughput **LIVELOCK**
(the 1:1 message↔forward convoy), **not** a deadlock. Recovered from the (since-reaped) investigator agents.
Bulky raw server logs (`server*.log`, ~230KB) remain under `~/w/vdc/chocobo/runs/stall-diag/`.

## formal/ — Z3 bounded model checking (agent ac9094fce3a8cf204)
- `model.py`/`model2.py`/`model3.py` — escalating-faithfulness control-flow models; the **deadlock query is
  UNSAT** across the sweep (proven deadlock-free).
- `convoy.py`..`convoy4.py` — retargeted to the throughput/liveness property; **`convoy4.py` is SAT** (the
  1:1 convoy is an admissible interleaving, as is a healthy high-coalescing schedule, under the same protocol).
- `sweep.py` — the T/K/D/plies/max_rows sweep. `HACK_AUDIT.md` — the model's self-audit.
- Run e.g.: `python convoy4.py 8 40 18`  → SAT, prints the convoy and a healthy schedule.

## empirical/ — gdb + /proc + instrumented server (agent a5e1df9bd28608a95)
- `standalone_server*.py` — the bucketed group-wakeup server with recv/sent/forward + rows/forward counters.
- `catch_*.sh`/`proc_sample.sh`/`launch4.sh` — bounded (timeout-wrapped) repro + stuck-stack capture.
- `bt*.txt`/`procsmp*.txt` — the captured stacks: all 3 producer threads parked in
  `recv_batch→zmq_msg_recv→poll`, server steady at ~1.4 rows/forward in the collapse. Verdict evidence:
  `recv==sent`, `route_errors=0`, always-eventually-completes, rows/forward **1.4 (collapsed) vs 55–177 (healthy)**.

## The fix (both arms agree)
Enforce a minimum coalescing degree independent of arrival timing: server-side `max_queue_delay`/
`preferred_batch_size` (a tuning surface) OR producer-side "never issue a sub-threshold message while
`inflight < D`" (a CLOSED structural fix — the one the abstraction should carry).
