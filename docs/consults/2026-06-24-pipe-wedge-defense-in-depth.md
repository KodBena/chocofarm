# Consult — harness defense-in-depth for the throughput-lab pipe wedge

**Date:** 2026-06-24
**Commissioned under:** ADR-0014 (second opinion when a problem resists resolution) + ADR-0013 (verify the artifact).
**Question put to the agent:** the server-side wedge was fixed at its instance (per-event diagnostic prints on the serve hot path, removed → counters surfaced once in `summary()`). Is a *harness-side* defense-in-depth warranted to close the *class*, and if so what design — given that concurrency is outside the maintainer's expertise, assess the concurrency risk explicitly.

**Scope clarification (added by the executor, not the agent):** this concerns the server subprocess's `stdout`/`stderr` *diagnostic/log* channel, which the harness captures via `subprocess.PIPE` and does not drain during the producer run. It is entirely orthogonal to the ZMQ leaf-eval data path (untouched) and to the MLP forward (untouched). The fix lives in `harness/run_lab.py` only.

The agent's verdict is recorded verbatim below (ADR-0005 Rule 9 — commissioned-review artifacts recorded verbatim, in-tree).

---

Empirically confirmed:

- **Case A (pipe, parent not reading)**: the child **blocked** in `write()` after only ~50 KB (one pipe buffer) and never exited within 2.5s — exactly the wedge mechanism. It only finished once the parent drained.
- **Case B (temp file)**: the child wrote ~406 MB and exited cleanly with rc=0 — **a file write never back-pressures the writer.**

This is the decisive evidence. No scratch changes were made to committed files (only a throwaway script in scratchpad). I have everything for the deliverable.

---

# Second Opinion: Harness defense-in-depth for the throughput-lab pipe wedge

## (a) Recommendation — DO IT, and the right design is the temp-file redirect, not a reader thread

**Warranted: yes.** Implement it as a **redirect of the server subprocess's stdout/stderr to a real file** (`tempfile`/a per-cell run log), replacing the `subprocess.PIPE`. Drop the continuous-reader-thread option. Here is the reasoning, weighed honestly both ways.

**Why warranted at all (the "is this churn against a hypothetical?" question, answered against my own first instinct to say "the server fix is enough"):**

The server-side type-fix (per-event diagnostics → counters surfaced once in `summary()`) is correct and is the *primary* fix — it closes the instance that detonated. But it does **not** close the *class*, and the class is squarely an ADR-0000 Specimen-1-shaped defect. Look at what remains representable in the harness:

- The harness wires `stdout=PIPE, stderr=STDOUT` (line 234-237) and **does not read that pipe for the entire duration of the producer run** — the main thread is blocked inside `subprocess.run(prod_cmd, …)` (lines 265-268). The only reads are `_wait_for_ready` (*before* the run) and `_drain_remaining` (*after* SIGINT).
- The OS pipe buffer is ~64 KB. My empirical test above confirms an unread writer **blocks after one buffer (~50 KB) and stays blocked indefinitely**.
- The server's `main` thread *is* the IO thread *is* the only thread Python delivers SIGINT to. If that thread is blocked in a C-level `write()` to the full pipe, the `_on_sigint` handler **physically cannot run** (Python signal handlers run between bytecode ops on the main thread only). Worse: the handler itself does `print(..., flush=True)` to the same pipe (`__main__.py:125`) — so even SIGINT acknowledgement can block on the full pipe.

So the wedge is not a property of the one diagnostic that was removed; it is a property of **the harness offering an unbounded, downstream-gated sink that nobody drains during the measured window.** *Any* future server write on a hot path re-arms it: a re-added log line, a noisier `summary()` histogram under a pathological `batch_hist`, a third-party library (JAX/XLA, pyzmq) that prints a warning under load, a `DeprecationWarning`. The discriminator from ADR-0013 Rule 4 / ADR-0000 — "is a *class* at stake, or a true one-off?" — comes down on the side of *class*: this is a recurring shape ("a measured subprocess writes to a sink the measurer isn't draining"), and the cost of re-discovery is high (a SIGKILL that *destroys the very stats the harness exists to collect* — a silent loss of the measurement, the worst-case surface).

The honest case *against* (so it is weighed, not dismissed): the reader-thread version of this fix carries real concurrency hazard and would be churn that earns its keep only against the hypothetical. That argument defeats the *reader-thread* design — and is exactly why I reject it (below). It does **not** defeat the *file-redirect* design, which is strictly *less* machinery than today's PIPE plumbing, not more.

## (b) Concurrency-risk assessment (the explicit blind spot)

This is where the two candidate designs diverge sharply, and it is the whole reason for the recommendation.

### The continuous reader/drain thread — REJECTED on concurrency grounds

A reader thread reading `proc.stdout` for the cell's whole lifetime introduces several genuine hazards, and the maintainer is right to be wary:

1. **Race with `_wait_for_ready`.** Today `_wait_for_ready` *itself* does `proc.stdout.readline()` to find READY. If a background thread is also reading the same pipe, the two race for lines — READY could be consumed by the drain thread and never seen by `_wait_for_ready`, hanging the cell until timeout. You'd have to **invert** the design: the reader thread becomes the *sole* reader and READY-detection moves to a `threading.Event` the reader sets when it matches `_READY_RE`. That is a non-trivial refactor of the startup path.
2. **Race with `_drain_remaining`.** Same single-consumer rule: `_drain_remaining` *also* reads `proc.stdout`. With a background reader you must delete `_drain_remaining` and instead `join()` the reader thread after the process exits, then read its accumulated buffer. Two readers on one fd is a data race on line boundaries.
3. **Join/shutdown ordering.** The reader thread blocks in `readline()`; it unblocks only on EOF (child exit) or a line. After SIGKILL the fd hits EOF and the thread can join — but if you ever `proc.kill()` you must be sure to `join()` the reader *after* the wait, and guard against the thread outliving a cell and bleeding lines into the next cell's `server_lines`. A per-cell thread lifecycle with a clean join barrier is needed.
4. **The shared mutable `server_lines` list** would be appended from the reader thread and read by `_parse_server_stats` on the main thread — needs a join-before-read barrier (the GIL makes `list.append` atomic, but visibility/ordering of "is the drain complete?" still needs the join).

None of these is unsolvable, but together they reproduce *in the harness* the exact category of bug the whole exercise is trying to eliminate — and they're enforced by discipline, not structure. That fails the ADR-0000 test: it trades a representable wedge for a representable set of reader-thread races.

### The temp-file redirect — RECOMMENDED, essentially zero new concurrency surface

`subprocess.Popen(server_cmd, stdout=run_log_fh, stderr=subprocess.STDOUT, …)` where `run_log_fh` is an OS file handle.

- **No new thread.** The OS writes the child's fds to the file; the parent never participates in draining. There is no producer/consumer relationship to get wrong, no join ordering, no shared buffer across threads.
- **The writer cannot block.** Empirically confirmed above: 406 MB written, rc=0, never blocked. A regular-file `write()` is not gated on a downstream reader, so back-pressure on the measured process is **structurally impossible** — which is precisely the property we want (see (c)).
- **READY detection becomes a file tail.** `_wait_for_ready` polls the file: read new bytes appended since the last offset, scan for `_READY_RE`, sleep ~10 ms on no-match, bail on `proc.poll() is not None` past EOF with no match. This is a small, *single-threaded, single-reader* change to one function — no concurrency at all. (The file-tail poll is the same shape as `tools/shell/compute-watchdog.sh`'s polling, which the corpus already trusts.)
- **`_drain_remaining` collapses to a final file read.** After `proc.wait()`/kill, read the file's full contents into `server_lines` and run `_parse_server_stats` as today. No thread to join.
- **The residual risks are mundane and bounded, not concurrency races:**
  - *Temp-file lifecycle*: create per cell, delete in the `finally` (or, better, keep it under `~/w/vdc/chocobo/runs/` per the memory note "never discard experiment output" — a named per-cell `tlab-server-{pid}-{seq}.log` is *more* useful than a vanishing pipe). Use a deterministic path or `tempfile.mkstemp`; clean up alongside the existing `os.unlink(sock_path)`.
  - *Read/write ordering of a tail*: a tailer can read a partially-written line. Trivially handled by only acting on lines terminated by `\n` (keep a remainder buffer), and READY is a full `flush=True` line so it lands atomically for a line-sized write.
  - *Disk growth*: unbounded in pathological flooding — but a file at disk-write speed is a non-issue for a bounded run, and it degrades to "a large log" not "a wedged measurement." Strictly better failure mode.

**Verdict on (b):** the file redirect introduces **no inter-thread synchronization at all**; it removes the producer/consumer pipe relationship that is the root of the hazard. The reader thread would add four real races. This is the decisive separator.

### Option 3 (make the server's stdout non-blocking / drop-on-full at the fd level) — REJECTED

This pushes the fix back onto the *measured* process (set `O_NONBLOCK` on its stdout, or drop-on-`EAGAIN`). It's the wrong owner: it makes the server responsible for tolerating a measurer that won't read it, and it silently *drops* server output (including possibly the `summary()` stats line) under exactly the load where you most want it. It also can't help with third-party libraries that hold their own fd. Don't.

## (c) The ADR-0000 type/structural framing — yes, there is one, and it's exact

The brief already cites the server-side half of it: the in-code comment at `server.py:454` frames the fix as *"the serve loop performs NO unbounded, downstream-gated blocking write."* That is the **producer-side** invariant. The harness change supplies the **dual, consumer-side** invariant, and together they form a clean by-construction property:

> **A measured process cannot be back-pressured by the measurer's transport.**

The structural realization is: **make the server's output sink a type that is total on `write` — i.e. one whose `write` cannot block on a downstream reader.** A pipe is *not* total on `write` (it's gated on a consumer draining it — my Case A). A regular file *is* total on `write` for all practical purposes (Case B: 406 MB, no block). So choosing a file over a pipe is choosing a sink-type for which the illegal state ("measured process blocked in `write()` holding its own SIGINT-handling thread") is **unrepresentable**, rather than merely *avoided by the discipline* "remember to keep the server's writes bounded forever."

This is precisely ADR-0000 Rule 2(a) ("what type makes the class unrepresentable") composed with its Specimen-3/4 pattern (an unbounded sink → a structurally-bounded one). It also answers ADR-0000 Rule 2(b) ("what operational lapse let it recur"): the net that *was* review-only ("don't add hot-path prints") becomes structural ("the sink can't back-pressure the writer"). The server-side counter-fix and the harness-side file-redirect are the **two facets of one foreclosed class** — the server stops emitting unbounded writes (shape), and the harness stops offering a back-pressuring sink (the net that holds even if a future writer regresses). Doing only the server half leaves the class representable through any *other* writer; the file redirect is what actually closes it.

One honest caveat per the corpus's measured-vs-interpreted discipline: I have *measured* that a pipe blocks an unread writer and a file does not (the script output above), and I have *read* the harness to confirm the main thread is unread-blocked during the producer run and that SIGINT lands on that thread. I have **not** run the full producer flood against a deliberately re-armed server to reproduce the end-to-end SIGKILL under the new design — that's the operational witness. The mechanism is confirmed; the end-to-end "it now survives a re-armed flood" is a claim the maintainer should verify with one flood run after integration (ADR-0013 Rule 5 — verify the artifact).

## (d) Effort / risk estimate

- **Effort: small.** Roughly: change the one `Popen` call (PIPE→file handle, drop `bufsize=1`/`text=True` pipe semantics or open the file in text mode); rewrite `_wait_for_ready` as a file-tail poll (~15 lines); collapse `_drain_remaining` to a final file read (~5 lines); add per-cell log create/cleanup alongside the existing socket cleanup. One file (`run_lab.py`), no server change, no new dependency.
- **Risk: low.** No new threads, no new synchronization. The only behavioral changes are (i) READY is now found by tailing a file instead of `readline()` on a pipe — testable in isolation, and (ii) server output now lands in a file the maintainer can inspect after the run (a feature, and it aligns with the "never discard experiment output, keep it under `~/w/vdc`" memory). The change is also *reversible* and self-contained to the harness.
- **Documentation:** the ORCHESTRATION CONTRACT docstring in `run_lab.py` (step 2/4 mention the stdout/stderr pipe and the READY-line wait) must be repointed to describe the file redirect (ADR-0005 Rule 3/5, live referrer on a mechanism change). If a "Revisit when…" trigger in the relevant ADR names this, record the firing. That's part of the delivery, not optional (CLAUDE.md "Documentation is part of the work").

---

**Bottom line:** Do the defense-in-depth change. Use the **temp/run-file redirect**, not a reader thread — it is *less* machinery than the current pipe, adds *zero* concurrency surface (the maintainer's blind spot is thereby sidestepped entirely, not navigated), and it makes the ADR-0000 invariant "*a measured process cannot be back-pressured by the measurer's transport*" true **by construction** rather than by the standing discipline "keep the server's writes bounded forever." It is the consumer-side dual of the server fix already committed, and only the two together actually foreclose the class. Effort small, risk low; verify with one re-armed flood run post-integration.
