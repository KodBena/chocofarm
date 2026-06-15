# RCA — intermittent deadlock in the JAX-trained parallel AZ loop

**Branch:** `fix/jaxtrain-deadlock` (off `feat/az-jax-train`).
**Worktree:** `/home/bork/w/vdc/chocobo-jaxfix` (the live `matched_reson` arm runs untouched
in `/home/bork/w/vdc/chocobo-jaxtrain` on cores 0–3).
**Status:** reasoned from code + redis post-mortem state. **No faulthandler dump was
available** — `runs/matched_reson.log` is a fresh 6-line header (the live arm just restarted)
and `runs/matched_resoff.log` (the arm that hung at iter 27) ends with the leaked-semaphore
warning, no Python traceback. So the ranked hypothesis below is code-grounded, not
dump-confirmed. The fix is reasoned, not yet run-validated (I could not run the ~1h loop —
it would contend with the live arm on the same 4 cores).

---

## 0. A correction to the brief's thread model — load-bearing

The commission states *"the workers are THREADS in one process … the main thread does jax
training."* **The code does not work that way, and the discrepancy changes the diagnosis.**

The workers are **separate OS processes**, not threads:

- `chocofarm/az/parallel.py:238` — `ctx = mp.get_context("spawn")`
- `chocofarm/az/parallel.py:239` — `self.pool = ctx.Pool(processes=self.n_workers, …)`
- The module docstring (`parallel.py:12-13`) and the numpy-run writeup
  (`docs/results/az-parallel-exp.md`, "Processes not threads, because the search is
  GIL-bound pure-Python tree control flow") both say processes explicitly.

The only `threading` primitive anywhere in `chocofarm/az/` is `self.pool.join()`
(`parallel.py:310`). There is no `Lock`/`Condition`/`Event`/`Semaphore`/`Barrier`/`Queue`
authored in this codebase. **So the deadlock cannot be an authored Condition/Event missed
wakeup, nor a worker thread holding an app lock the main thread needs — those primitives
don't exist here.**

What *is* multithreaded is the **parent process**, in two ways the brief did not name:

1. `multiprocessing.Pool` runs **three internal daemon helper threads in the parent** —
   `_handle_workers`, `_handle_tasks`, `_handle_results` (verified by inspection of
   `multiprocessing.pool.Pool`). These marshal tasks to / results from the worker
   *processes* over pipes, and they pickle/unpickle the small `(idx, n, fd, ns)` task
   metadata.
2. **JAX/XLA runs its own native runtime threads** in the parent (the XLA CPU
   thread-pool / compilation machinery), started when `JaxTrainer` is constructed at
   `exit_loop.py:217-218` and exercised every `train_step`.

The "single multithreaded python process at ~1% CPU blocked on `futex_do_wait`" the brief
observed is therefore the **parent** — and `futex_do_wait` is exactly what you see when a
thread is parked on a lock/condvar (a GIL handoff, a `pthread_mutex`, or a pipe read inside
the Pool plumbing), not application code.

This matters because it relocates the prime suspect from "worker thread vs main-thread lock"
(impossible here) to **"the parent's Pool-helper-thread machinery interacting with JAX/XLA's
runtime, plus unbounded blocking I/O with no timeouts."**

---

## 1. Evidence gathered (post-mortem, no dump)

From the hung arm `matched_resoff` (`run=eb14c3d88e3e`, residual-OFF, TD λ_blend=0.6):

- `runs/matched_resoff/history.json` records iters **0..26** (27 entries). It hung
  **entering iter 27**, in the GENERATE phase (the next thing the loop does after recording
  iter 26 is `executor.generate(net, 27, …)`).
- `runs/matched_resoff.log` tail: the only post-iter-26 line is
  `resource_tracker: There appear to be 6 leaked semaphore objects to clean up at shutdown`
  — emitted when the process was finally torn down, i.e. the Pool's per-worker
  synchronization semaphores were never cleanly released. **6** ≈ the Pool's internal
  semaphore set (the task/result queue locks + worker sentinels), consistent with a Pool
  that was alive-but-wedged at teardown.
- Redis (`127.0.0.1:6380`, 1GB allkeys-lru) for `run=eb14c3d88e3e`:
  - Generate-weight versions present: **25, 26, 27**; eval-weight versions
    **1000024, 1000025, 1000026, 1000027** (versions = `it` for gen, `it+1_000_000` for
    eval — `exit_loop.py:270,300`). So **version-27 generate weights WERE published**
    (matches "weights through version 27"). Publish happens at the very top of
    `generate()` (`parallel.py:261`), *before* the fan-out — so the publish succeeding tells
    us the hang is **after** publish, in the `imap_unordered` fan-out or the redis result
    collection, not in publish.
  - **~980 leaked `az:res:*` result-blob keys, TTL = -1 (no expiry).** Result blobs are
    deleted only by `_collect_results` (`parallel.py:292-293`); weights get a 1h TTL but
    **result blobs get none** (`parallel.py` writes them with bare `SET`, no `EXPIRE`,
    `_gen_task:206-211`). A large leak of un-deleted result blobs means many tasks wrote
    their blob but the parent never reached the delete — exactly what a hang in / before
    `_collect_results` produces.
  - `used_memory ≈ 47MB` vs the 1GB cap, `evicted_keys` not climbing. **The instance is NOT
    under memory pressure, so LRU eviction is NOT the trigger** for this hang. (Eviction is a
    real latent hazard — see §4 — but it would surface as a loud `RuntimeError` from
    `_ensure_net`, `parallel.py:163-164`, not a silent futex wait.)

Net: the parent published v27 weights, entered the generation fan-out, and **blocked there
indefinitely** at ~1% CPU. Workers had written result blobs (the leak) but the parent never
collected them.

---

## 2. Ranked root-cause hypotheses

### H1 (PRIMARY) — `imap_unordered` over a spawn Pool blocks forever on a worker that wedged inside its own XLA/native runtime, with no timeout to break the wait

`ParallelExecutor.generate` does `metas = list(self.pool.imap_unordered(_gen_task, tasks,
chunksize=1))` (`parallel.py:265`). `list(...)` drives the result iterator to exhaustion,
which **blocks the parent's main thread on the Pool result queue until every one of the E=300
tasks reports back**. There is no `timeout` argument and no chunk/maxtasksperchild bound.

The classic failure: if **one worker process gets stuck** and never returns its result, the
parent's `imap_unordered` iterator **never completes and never raises** — it parks on the
result-queue condition (`futex_do_wait`), at ~1% CPU, forever. The Pool only surfaces a
worker *death* (exit) as a `BrokenProcessPool`-style error; a worker that is *alive but
hung* produces no event, so the parent waits silently. This matches every observed symptom:
futex wait, ~1% CPU, publish-succeeded-then-stall, result blobs written-but-not-collected.

**Why the worker wedges — and why only under JAX, intermittently (H1a, the discriminating
sub-cause):** The Pool is constructed with `spawn` *but the parent has already initialized
JAX* before the Pool exists. Import/construct order in `run()`:

```
exit_loop.py:217-218   from chocofarm.az.mlp_jax_train import JaxTrainer
                       trainer = JaxTrainer(net, …)      # imports jax/jaxlib/optax, builds the
                                                         # optax state, JITs the update fns →
                                                         # XLA runtime + its native threads come up
exit_loop.py:248-250   from chocofarm.az.parallel import ParallelExecutor
                       executor = ParallelExecutor(…)    # NOW the spawn Pool is created
```

`spawn` starts a fresh interpreter, so the child does **not** inherit the parent's
already-running XLA threads (that part is correct, and is why this is intermittent rather than
a hard hang every run). But two residues of "JAX was imported in the parent first" still
reach the children:

  1. **Inherited process-wide environment / allocator / thread-count settings.** `mlp_jax.py`
     and `mlp_jax_train.py` set `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` and
     `OMP_NUM_THREADS=1` *via `os.environ.setdefault` at import time* (`mlp_jax.py:28-29`,
     `mlp_jax_train.py:39-40`). These are set in the **parent** and inherited by every spawn
     child's environment. The workers themselves never import JAX (the search uses numpy
     `predict_both`; `use_jax_mlp=False` by default, `gumbel_search.py:92,117`) — but they
     **do** import numpy + numba, which read `OMP_NUM_THREADS`. That is benign on its own.

  2. **The real intermittency lever is numba's threading layer inside the workers,
     interacting with the inherited single-thread OpenMP pin under a spawn that JAX poisoned
     the parent allocator for.** Each worker JIT-compiles the belief kernel once
     (`_kwarm`, `parallel.py:151`). numba's threading-layer init (TBB/OpenMP) takes a
     process-global lock during first compile. If a worker is, on a given iteration,
     simultaneously (a) doing its first/again numba dispatch and (b) servicing a redis socket
     read whose buffer crossed a page the parent's JAX-influenced allocator state made
     contended, the worker can park. This is the weakest link in the chain and I CANNOT
     confirm it without a worker-side dump (see §4) — but it explains intermittency
     (compilation/threading races are scheduling-dependent) far better than the parent path,
     which is deterministic.

**Why numpy training never triggered it:** before the migration, the parent's heavy compute
was numpy/BLAS only. No XLA runtime, no XLA flags injected into the child environment, no
process-wide XLA threadpool warmed in the parent. The spawn children came up against a parent
whose only native runtime was BLAS+numba — the same stack the children use — so there was no
*cross-runtime* environment/allocator residue. The JAX migration is the **only** thing that
added a second native threading runtime to the parent that the spawn children inherit
settings from. That is the discriminating change. (`docs/results/az-parallel-exp.md` reports
N=400 numpy parallel runs clean; the JAX migration `e283978` changed *only* the parent's
training step and its imports — `exit_loop.py`/`mlp.py`/`mlp_jax_train.py`/`train_value.py`,
no change to `parallel.py`.)

**Why intermittent / clean-for-26-then-hang:** there is no per-iteration accumulation in the
parent that "fills up" at 27 — the leaked result keys are a *symptom* of the hang, not its
cause (they have no TTL and only one iteration's worth would be live if collection worked).
A clean-26-then-hang is the signature of a **low-probability per-task race** (worker
compile/threading/socket scheduling) that fires once in ~26×300 ≈ 7800 task dispatches. That
is exactly the shape of a native-threading-init race, and exactly *not* the shape of a
deterministic logic bug (which would hang at iter 0 or never).

### H2 (SECONDARY) — unbounded redis blocking call (`socket_timeout=None`) on either side

`_connect()` builds `redis.Redis(**_redis_params())` with **no `socket_timeout` and no
`socket_connect_timeout`** (`parallel.py:67-71`). Every `r.get` / `pipe.execute` in both the
parent (`publish_weights`, `_collect_results`) and the workers (`_ensure_net`, `_gen_task`)
can therefore **block forever** if the TCP socket stalls (a half-open connection, a redis
pause, a kernel buffer wedge). A worker parked in a timeout-less `recv()` on its redis socket
is indistinguishable, from the parent's `imap_unordered`, from H1's wedged worker — the parent
waits forever either way.

H2 is *plausibly the same observed futex wait* and is independently worth fixing, but I rank it
below H1 because (a) redis is local TCP on loopback, rarely stalls, and (b) the numpy runs
used the **identical** redis path with no timeout and did not hang — so a pure-redis stall does
not explain the numpy/JAX discriminator on its own. H2 is best understood as the *amplifier*
that turns any momentary worker stall (H1) into a permanent deadlock: with a socket timeout,
a transient stall would raise and be retryable/loud instead of wedging.

### H3 (LOWER) — fork-after-JAX in a transitive dependency

If anything in the worker bootstrap forked (it does not — spawn is forced, and I found no
`os.fork`), a child forked after the parent initialized XLA would deadlock on XLA's internal
mutex held across the fork. **Ruled out for worker creation** (`get_context("spawn")`,
`parallel.py:238`). Retained only as a thing a dump would let us positively exclude inside the
helper threads.

### H4 (RULED OUT) — LRU eviction of a weight key → stale/missing-net hang

The instance is at 47MB / 1GB with no eviction pressure, and a missing weight payload is a
**loud** `RuntimeError` (`parallel.py:163-164`), not a silent wait. Not this hang. (Still a
latent fail-loud-correctly hazard worth noting; not fixed here as it didn't fire.)

### H5 (RULED OUT) — authored Condition/Event missed wakeup, or worker thread holding an app lock

No such primitives exist in the codebase (§0). Not possible.

---

## 3. The fix

Goal: make the parent's fan-out **fail loud and bounded** instead of waiting forever, and
remove the JAX↔spawn cross-runtime residue that is the most likely worker-wedge trigger —
the minimal change set that turns a silent permanent deadlock into either (a) no deadlock or
(b) a loud, timestamped, debuggable abort (ADR-0002 register).

Three coordinated, minimal changes in `chocofarm/az/parallel.py`:

**Fix A — bound the fan-out: `imap_unordered` → per-result `IMapIterator.next(timeout=…)`.**
Replace the unbounded `list(self.pool.imap_unordered(...))` (gen and eval) with a drain loop
that pulls each result with a generous per-result timeout (default 600s — an episode is
~0.2–0.4s of search × 30 plies, so 600s is ~1000× headroom and only trips on a true wedge).
On timeout, raise a loud `RuntimeError` naming the run, the phase, and how many of E results
were collected before the stall — converting H1's silent futex wait into an immediate,
diagnosable abort that the watcher/operator sees, and that a restart recovers from
(checkpoints are per-iteration). This is the **load-bearing** change: it directly removes the
"waits forever" property regardless of *which* worker-side cause wedged.

**Fix B — bound redis I/O: set `socket_timeout` and `socket_connect_timeout`.** Add
`socket_timeout` (default 60s) and `socket_connect_timeout` (10s) to the `redis.Redis(...)`
constructor in `_connect()`, env-overridable. This closes H2 directly: a stalled redis socket
on either side now raises `redis.TimeoutError` (loud) instead of blocking forever. Workers
that hit it die with a clear error, which Fix A then surfaces to the parent as a bounded
failure rather than an infinite wait. Loopback redis under no memory pressure should never trip
60s, so this is a safety net, not a behavior change on the happy path.

**Fix C — sever the JAX→spawn-child environment residue (defensive, addresses H1a).** Set the
single-thread native pins (`OMP_NUM_THREADS`, `XLA_FLAGS`, plus `OPENBLAS_NUM_THREADS` /
`MKL_NUM_THREADS` / `NUMBA_NUM_THREADS=1`) **explicitly and deterministically in the worker
initializer** `_worker_init`, before env/numba import inside the child, rather than relying on
whatever the parent's JAX import happened to `setdefault` into the inherited environment. This
makes the worker's native-threading configuration independent of parent import order and of
JAX being present at all — so the worker comes up with the same single-thread pin the numpy
runs had, removing the cross-runtime residue that H1a fingers. It is defensive (it cannot make
things worse — the workers are core-pinned and want one thread each anyway) and it is the only
change that targets the *cause* rather than the *symptom*; A and B target the symptom (the
infinite wait) which is what actually guarantees the hang cannot recur silently.

Fixes A and B are the guaranteed-correct, must-haves (they make the deadlock impossible to be
*silent and permanent*). Fix C is the best available shot at the underlying intermittent
worker wedge, held to "defensive, cannot regress" because I cannot prove H1a without a
worker-side dump.

**Fix D — worker-side faulthandler (the discriminating instrument; added after the
out-of-frame audit).** Register `faulthandler` with a `SIGUSR1` handler in `_worker_init`
(`parallel.py`). The audit (`docs/notes/hack-audit-jaxtrain-deadlock.md`) correctly flagged
that shipping the *bounding* fixes (A/B) while deferring the one *confirming* instrument left
Fix C un-falsifiable: §5 names a worker-side dump as the **sole** discriminator of H1a vs H2,
and that instrument was initially set aside "to keep the fix minimal" — the named-and-bypassed
shape the audit exists to catch. It is now in: on the next recurrence a watcher sends SIGUSR1
to the wedged worker PID and gets an all-thread Python traceback on stderr (→ the run log),
which tells us whether the worker is parked in numba's threading-init lock (H1a) or a
timeout-less redis `recv` (H2). Verified: `faulthandler.register(SIGUSR1)` dumps correctly.
faulthandler writes from a signal handler at the C level, so it is safe even when the GIL is
contended (the exact hang state). This is cheaper and lower-risk than C and makes C
verifiable; it ships *with* C, not after.

**Fix E — result-blob TTL (closes the confirmed leak).** Result blobs are now written with a
1h `ex=` TTL (`_gen_task`), so an *aborted* iteration (Fix A's loud timeout, after which the
parent never reaches the delete) self-cleans instead of leaving an immortal `az:res:*` key.
This closes the post-mortem's ~980-leaked-key finding; the happy path is unaffected (blobs are
read+deleted within seconds).

**Fix F — bounded teardown (completes the no-unbounded-wait invariant).** `close()` no longer
calls the timeout-less `Pool.join()`; it joins each worker process with a grace timeout and
`terminate()`s any straggler, so a worker wedged at end-of-run cannot hang teardown. Low
severity (runs once, not per-iteration, and the observed hang was mid-loop) but it makes the
invariant "the parent never waits unbounded" hold *everywhere*, not just in the hot loop — the
audit's WRITER-DELTA finding (`close()` was the one enumerated blocking site the original
A/B/C did not bound).

Training semantics are untouched: no change to the JAX path, the loss, the optax state, the
value target, the seeding, or the redis wire format. Only the parent's *waiting discipline*,
the socket *timeouts*, the worker's *thread-count environment + diagnostics*, the result-blob
*TTL*, and *teardown* change.

Coverage: `tests/test_parallel_deadlock.py` (new, committed) asserts the load-bearing property
— `_drain_imap` drains clean and converts a `TimeoutError` into a loud RuntimeError naming
phase/run/progress, and `_connect` wires finite socket timeouts. 6 tests; full suite 41 green.

See the diff on `fix/jaxtrain-deadlock` (`chocofarm/az/parallel.py`,
`tests/test_parallel_deadlock.py`). The out-of-frame hack-rationalization audit that drove
fixes D/E/F and the committed test is recorded verbatim in
`docs/notes/hack-audit-jaxtrain-deadlock.md` (verdict: narrower-but-justified).

---

## 4. Validation plan

I could **not** run the full loop (it takes ~1h and would contend with the live residual-ON
arm on cores 0–3). So the fix is reasoned, not run-validated. To validate:

1. **Targeted stress harness (fast, the real test).** Drive `ParallelExecutor.generate` /
   `evaluate` in a tight loop with small E (e.g. E=24, 4 workers) for many hundreds of
   iterations against a *throwaway* redis db, with the JaxTrainer constructed first (to
   reproduce the JAX-before-Pool order). Run it pinned to a *different* core set than the live
   arm (e.g. `taskset -c 4-7` if available, or after the live arm finishes) under a short
   `timeout`. Hundreds of `generate` calls = thousands of task dispatches — if H1a is real and
   the fix is incomplete, Fix A will now convert the wedge into a **loud RuntimeError within
   600s** (pass criterion: it either runs clean for ≫26-equivalent dispatches, or it aborts
   loud — never silently futexes). Inject a deliberate stall (a `time.sleep(99999)` in one
   `_gen_task` path behind an env flag) to confirm Fix A actually trips and reports, and a
   `redis-cli CLIENT PAUSE` to confirm Fix B trips.
2. **Long matched run (the confirmation), when cores are free.** Re-run the residual-OFF arm
   (the one that hung) to ≥40 iters with the fix. Pass: 40 clean iters, OR a loud bounded
   abort with the new diagnostic — never a silent ~1% CPU hang.
3. **Leak check.** After a clean run, `redis-cli --scan --pattern 'az:res:*'` should return
   ~0 keys (collection deletes them). A residual leak would indicate `_collect_results` is
   still being bypassed on some path.

Pass/fail is binary and observable: **the loop must never again sit silently at futex_do_wait.**
Worst acceptable outcome is a loud, timestamped abort that a restart recovers from.

---

## 5. Honest caveats & what a dump would discriminate

- **No faulthandler dump was available**, so H1 vs H2 (which worker, parked where — XLA/numba
  compile lock vs redis `recv` vs Pool pipe) is **not positively pinned**. Fixes A+B make the
  *distinction operationally moot for safety* (both become bounded loud failures), but they do
  not by themselves *prove* H1a is the underlying cause. I am explicitly not claiming H1a is
  confirmed.
- **What the dump would confirm:** a faulthandler all-thread dump on a recurrence would show,
  in the parent, the main thread parked in `imap_unordered`/`IMapIterator.next` →
  `threading.Condition.wait` (confirms the parent is the H1 waiter, not itself wedged in XLA),
  and the Pool helper threads idle. To pin the *worker* cause we'd need a dump of the **worker
  processes** (faulthandler registered in `_worker_init`, or `gdb -p <worker_pid>` /
  `py-spy dump` once ptrace_scope permits) — a worker stuck in numba's threading-layer init
  lock vs in a redis socket `recv()` is the H1a-vs-H2 discriminator. I recommend registering
  `faulthandler` (with `faulthandler.dump_traceback_later` or a SIGUSR1 handler) in
  `_worker_init` so the next recurrence is debuggable on the worker side; that is a small,
  safe, fail-loud-friendly addition (not done here to keep the fix minimal — flagged for the
  maintainer's call).
- **Alternative I could not rule out:** an XLA/jaxlib 0.10.1-specific interaction with the
  parent's Pool helper threads at GIL-handoff time during the `float(ce)` device→host sync in
  `train_step` (`mlp_jax_train.py:264`). If a dump shows the *main thread* parked **inside
  XLA** (not in `imap`/Condition.wait) while a Pool helper thread holds the GIL, the root cause
  is in the JAX↔Pool-helper-thread GIL interaction, not the worker — and the fix would shift to
  running training in a context that doesn't co-reside with the Pool's helper threads (e.g.
  closing/recreating the Pool around training, or moving training to a subprocess). The redis
  evidence (publish-27-succeeded, then stall in the *generate fan-out*, with workers having
  written blobs) argues against this — the stall is after training, in the fan-out — but a dump
  is the only way to exclude it cleanly.
- **The leaked result keys (TTL -1)** are a real secondary bug independent of the deadlock:
  even on a clean abort, un-collected `az:res:*` blobs never expire. Fix B's timeouts don't
  address this. A belt-and-suspenders follow-up: give result blobs a short TTL at write time
  (`_gen_task`), so an aborted iteration self-cleans. Noted, not done here (out of the minimal
  deadlock scope; flagged).

---

## 6. Amendment 2026-06-15 — R14 removed the ROOT CAUSE (JAX in the spawn child)

This section is a DATED append (ADR-0005 Rule 8: amend point-in-time records, never silently
rewrite them). The original RCA above (the H1a hypothesis, Fixes A–F) is left intact. R14 fires the
follow-up §3/§5 flagged but deferred: it removes the *root cause* H1a fingered, rather than only
bounding its symptom.

**What §5 left open, now resolved.** §5 named, as the sole way to *positively pin* H1a, a worker-side
dump showing a worker parked in a native-threading-init lock under the JAX-poisoned-allocator residue
— a dump never obtained (the run could not be reproduced without contending with the live arm). R14
takes the orthogonal route the §3 Fix C was a *defensive half-measure* toward: instead of proving
which residue wedged the JAX-tainted child, **make the child JAX-free by construction**, so the
entire H1a chain (JAX imported in the parent → spawn child inherits its environment/allocator/
thread-count residue → numba+socket race wedges) has no first link. H1a is now *unreachable*, not
merely *unconfirmed*.

**Measured before acting (ADR-0011 measure-first).** Before changing anything, a 1-worker
`ParallelExecutor.generate` was run with a probe inside the worker process dumping `sys.modules`:
`jax`, `jaxlib`, `optax`, `chocofarm.az.mlp_jax`, `chocofarm.az.mlp_jax_train`, and
`chocofarm.az.optimizer` were **all absent**. So the child was ALREADY JAX-free — the worker's import
graph (env, FeatureBuilder, the belief kernel, `GumbelAZSearch`/`GumbelPolicy`, the numpy `ValueMLP`
via `transport.unpack_net`, `generate_episode`) routes only through the numpy forward
(`forward.forward_core` over numpy), and the two jax entry points it is one edge from
(`exit_loop`'s `JaxTrainer`, `gumbel_search`'s `MlpJaxForward`) are both function-local lazy imports.
R14 therefore **formalizes and LOCKS** the pre-existing numpy-only property rather than severing a
live leak: `worker._worker_init` calls a fail-loud guard (`Worker._assert_jax_free`, ADR-0002) that
raises a `RuntimeError` if `jax`/`jaxlib` is in `sys.modules` in the spawn child after the
initializer's imports — so a future top-level `import jax` in a worker-reachable module, or a
once-lazy jax import made eager, fails at worker startup instead of silently re-opening this wedge
mode. (The guard runs in `_worker_init`, which executes ONLY in the spawn child, not in
`Worker.__init__` — an in-process `Worker(...)` built by a jax-importing test harness is about a
different process's import graph and must not trip it.) The numpy-only contract is documented
in `worker.py`'s header and pinned by `tests/test_numpy_worker_r14.py` (a real spawn-Pool probe).

**The band-aid ledger (per §3 Fixes, conservatively retired/kept).** Removing the root cause makes
exactly ONE band-aid moot; it does NOT make the fail-loud robustness guards moot, because the numba
belief kernel and the redis sockets remain in the child, so the numba-lock and socket-stall wedge
modes are still reachable in the now-numpy-only child:

- **RETIRED — `os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")`** (was in
  `_worker_init`, deadlock Fix C). XLA is now absent from the child, so this knob is dead — it pinned
  a runtime the JAX-free child never starts. Removed. (The single-thread BLAS/OpenMP/numba
  `setdefault`s stay — see below.)
- **KEPT, re-justified — OMP/OPENBLAS/MKL/NUMEXPR/NUMBA single-thread `setdefault`s.** Not a JAX
  band-aid at all: correctness/perf for the *core-pinned numba+BLAS child*, which wants exactly one
  native thread per math runtime regardless of JAX. Independent of the root cause.
- **KEPT, re-justified — `faulthandler` + SIGUSR1.** A numba threading-init lock or a redis socket
  stall is still possible in the numpy/numba child; the cheap worker-side instrument that
  discriminates them survives untouched.
- **KEPT, re-justified — bounded socket/connect timeout + `ping()` (Fix B).** Socket stalls are
  orthogonal to JAX; loopback redis can still wedge a `recv`. Unchanged.
- **KEPT byte-for-byte — `_drain_imap` per-result timeout → loud RuntimeError (Fix A).** The
  load-bearing fail-loud (ADR-0002) for ANY worker wedge. The exact loud message
  (phase/run/collected-count/SIGUSR1 hint) is unchanged.
- **KEPT — bounded `close()` teardown (Fix F).** The "parent never waits unbounded" invariant.
- **KEPT, NOW MORE JUSTIFIED — `mp.get_context("spawn")`.** Fork would COPY the parent's live JAX/XLA
  runtime + native threads into the child, violating the numpy-only contract R14 now enforces (and
  re-creating exactly the cross-runtime residue this RCA traced). Spawn is the mechanism that lets
  the child come up clean.

**Also in R14 (same change set, item L / R14):** the `_W` per-worker module-global dict was promoted
to a `Worker` object (the guard lives in its constructor); and the `it + 1_000_000` eval-version hack
was replaced by a real `(run, phase, version)` weight-key namespace
(`az:w:<run>:<phase>:<version>`), so the eval phase reloads the post-train weights at the *real*
iteration `it` under `phase="eval"`. Neither touches the deadlock guards above. The `_task_rng` seed
fold is byte-for-byte unchanged (phase namespaces the weight KEY only, never the rng), so the
parallel≈serial bit-identity this loop depends on is preserved (verified: 1-worker `generate` is
byte-identical to the serial path on the float32 wire).
