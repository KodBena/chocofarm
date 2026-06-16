#!/usr/bin/env python3
"""
chocofarm/az/worker_pool.py — WorkerPool: the SOLE owner of the AZ parallel-loop multiprocessing
lifecycle (audit item K, the Pool third of the Transport ⊥ Pool ⊥ Task split out of `parallel.py`).

This module owns the process-pool lifecycle and nothing about the redis wire protocol (that is
`transport.py`) or what one worker computes (that is `worker.py`). Concretely it owns: the
`mp.get_context("spawn")` pool built with `worker._worker_init` (NOT fork — a clean child; R14 makes
this MORE load-bearing, see below), the per-result bounded drain `_drain_imap` (the deadlock-RCA
load-bearing piece: a per-`it.next(timeout)` bound, NOT a whole-fan-out bound, → a LOUD diagnosable
RuntimeError), the `imap_unordered` fan-out under that drain (`map(...)`), and the bounded `close()`
teardown (close → per-proc `join(grace)` → `terminate()` stragglers → reap) plus the context-manager.

The deadlock-RCA band-aids owned HERE are KEPT byte-for-byte and re-justified on orthogonal grounds
(R14 — the root cause, JAX-in-the-spawn-child, was removed in `worker.py`, but these guards address
wedge modes that remain reachable in the numpy/numba child): spawn (not fork) — now MORE justified,
since fork would copy the parent's live JAX/XLA runtime state into the child and VIOLATE the
numpy-only contract `worker.py` now enforces; the per-result timeout with the exact loud message
naming phase/run/collected-count/SIGUSR1 hint — fail-loud (ADR-0002) for ANY worker wedge (a numba
lock or a socket stall, both still possible); and the bounded teardown so the parent never waits
unbounded anywhere.  (The `XLA_FLAGS` setdefault — the ONE band-aid R14 retired as moot — lived in
`worker.py`'s `_worker_init`, not here.)

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Literal


# Per-result timeout for the fan-out drain (deadlock fix H1 / Fix A). An episode is ~0.2–0.4s of
# search × ~30 plies; 600s is ~1000× headroom and only trips on a TRUE wedge (a worker stuck in a
# native-runtime lock or a timeout-less socket read). Env-overridable.
_RESULT_TIMEOUT_S = float(os.environ.get("CHOCO_RESULT_TIMEOUT", "600"))


def _drain_imap(it: Any, n_expected: int, phase: str, run: str) -> list[Any]:
    """Drive a Pool `imap_unordered` iterator to exhaustion with a PER-RESULT timeout, instead of
    the unbounded `list(imap_unordered(...))` that blocks forever on a wedged worker.

    `list(...)` parks the parent's main thread on the result-queue condition until EVERY task
    reports back; a worker that is alive-but-hung (stuck in a native-threading-init lock, or in a
    timeout-less redis recv) produces no Pool event, so the parent waits at futex_do_wait at ~1%
    CPU forever — the observed deadlock. Pulling each result with a timeout converts that silent
    permanent hang into a LOUD, diagnosable RuntimeError (ADR-0002) naming the phase, the run, and
    how many of the expected results were collected before the stall. A per-iteration checkpoint
    means a restart loses nothing."""
    import multiprocessing
    out: list[Any] = []
    while True:
        try:
            out.append(it.next(_RESULT_TIMEOUT_S))
        except StopIteration:
            break
        except multiprocessing.TimeoutError as e:
            raise RuntimeError(
                f"parallel {phase} fan-out (run={run}) stalled: collected {len(out)}/{n_expected} "
                f"results, then NO further result arrived within {_RESULT_TIMEOUT_S:.0f}s of the "
                f"previous one (per-result wait, not a whole-fan-out bound) — likely a wedged "
                f"worker (native-runtime lock or redis socket stall). To get a worker-side "
                f"traceback, send SIGUSR1 to the stuck worker PID (faulthandler is registered in "
                f"_worker_init). Aborting loud rather than deadlocking at futex_do_wait "
                f"(ADR-0002). Restart from the last checkpoint."
            ) from e
    return out


class WorkerPool:
    """The SOLE owner of the AZ parallel-loop multiprocessing lifecycle: a persistent pool of
    `n_workers` core-pinned spawn workers built with `worker._worker_init`. Build once; call
    `map(task_fn, tasks, phase, run)` each fan-out (it runs `imap_unordered` under the bounded
    per-result drain); `close()` at the end (bounded teardown) or use as a context manager."""

    def __init__(self, n_workers: int, cores: list[int], base_seed: int) -> None:
        import multiprocessing as mp
        from chocofarm.az.worker import _worker_init
        self.n_workers = int(n_workers)
        # spawn (KEPT, MORE justified by R14): a clean fresh-interpreter child. fork would COPY the
        # parent's live JAX/XLA runtime + its native threads into the child, violating the numpy-only
        # contract worker.py enforces (and re-creating the deadlock-prone cross-runtime residue).
        # The search budget m/n_sims is no longer an initarg — it is HOT and rides the per-iteration
        # hot_search into each task (worker.ensure_net applies it on the (phase,version) rebuild).
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(
            processes=self.n_workers,
            initializer=_worker_init,
            initargs=(list(cores), base_seed),
        )

    def map(self, task_fn: Callable[..., Any], tasks: list[Any], phase: str,
            run: str) -> list[Any]:
        """Fan `tasks` across the pool via `imap_unordered` and drain under the bounded per-result
        timeout (Fix A): a wedged worker aborts LOUD, never deadlocks. Returns the worker results in
        completion order (the caller reassembles by the idx the task carries)."""
        it = self.pool.imap_unordered(task_fn, tasks, chunksize=1)
        return _drain_imap(it, len(tasks), phase, run)

    def close(self) -> None:
        # Bounded teardown (completes the "parent never waits unbounded" invariant — see Fix A).
        # `Pool.join()` takes NO timeout, so a worker wedged at end-of-run would hang close()
        # forever exactly as the hot loop could. Instead: close (no new tasks), then join each
        # worker process with a timeout; any worker still alive after the grace period is
        # terminate()'d (SIGTERM) so teardown is bounded. Low-severity vs the hot loop (runs once
        # at end-of-run) but it makes the no-unbounded-wait property hold everywhere.
        self.pool.close()
        grace = float(os.environ.get("CHOCO_POOL_JOIN_TIMEOUT", "30"))
        procs = list(getattr(self.pool, "_pool", []))
        for p in procs:
            p.join(grace)
        for p in procs:
            if p.is_alive():
                p.terminate()      # bounded: don't let a wedged worker hang teardown
        try:
            self.pool.join()       # reap; all workers are now exited or terminated
        except Exception:
            pass

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        self.close()
        return False
