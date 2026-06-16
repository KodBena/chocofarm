<!-- docs/handoff-2026-06-16-zmq-async-gumbel.md -->

# Handoff — 2026-06-16: the ZmqNetClient async question + the Gumbel-AZ port

Point-in-time record (ADR-0005 — append, do not retro-edit). Supersedes the
repository-condition section of `docs/handoff-2026-06-15-architecture-refactor.md`
for current run-state. `main` = `51b13b9`.

## TL;DR for the next session

1. **The live open question — treat as the top priority.** The C++ `ZmqNetClient`
   (`cpp/src/zmq_net_client.cpp`, merged at `51b13b9`; the impl commit is `d06db93`)
   is **under scrutiny as a likely ADR-0012 P7 hack**, and the prior session (me)
   defended it badly. Two distinct problems:
   - It is a **blocking `ZMQ_REQ`** client — strict lock-step send→recv, one
     in-flight call per socket/thread — which is in direct tension with the
     **asynchronous work-stealing loop** that `docs/design/cpp-batched-search.md`
     is built around (M trees in flight, batched server-side).
   - It + the `NetEvaluator` port were built **ahead of any consumer**: no C++
     search dispatches through the port yet (NMCS/ISMCTS use the `WorldSource`
     seam; the Gumbel search that *would* consume it isn't ported/merged). So the
     port's polymorphism is unexercised — speculative generality — and a
     permanently-dead `Error` arm was added to `NetForward::predict` to host it.
   - **The prior session's defense of all this was the P7 violation in miniature:**
     "build Embodiment 1 (blocking REQ) now, Embodiment 2 (DEALER/fiber async) only
     if measurement shows thread count is the bottleneck" is the forbidden
     scale/proportionality argument shape almost verbatim. **Do not repeat it.**
   - The maintainer is running a **consult** on this themselves (the prompt:
     investigate `zmq_net_client.cpp` + its commit vs ADR-0012 and the async-loop
     goal; sunk cost vs rectification cost; whether all of it must be discarded;
     under the hack-rationalization-detector lens). **Await that verdict, then
     rectify or discard — do not pre-defend the client.** A reasonable rectification
     shape: keep the `NetEvaluator` port + the shared `inference_wire.hpp` codec +
     the de-std/decode logic; replace the blocking-REQ transport with a
     `DEALER` non-blocking submit/poll + completion routing + a fiber/coroutine
     `predict` that *looks* synchronous to the search but yields instead of
     blocking. Sunk cost is roughly the REQ transport layer of `zmq_net_client.cpp`;
     the port/codec/decode survive. Let the consult set the real scope.

2. **1a Gumbel-AZ structure port — DONE but UNMERGED.** Branch
   `worktree-agent-a48030e67b35f7dd3` @ `56611e5` (off `51b13b9`). `GumbelAZPolicy`
   ported (Gumbel-top-k → Sequential Halving → PUCT → tree), the first real intended
   consumer of the `NetEvaluator` port. Verified: **144/144** action + improved-π
   argmax parity vs Python on precision-insensitive scripted inputs; both Danihelka
   invariants (executed==SH-survivor; SH spends full budget); a **genuine** mutation
   control (mutates the real C++ search: sh-budget 120/144, puct 61/144 divergence —
   the agent caught its own first weaker control as an undischarged hack and fixed
   it); fresh build clean at `-std=c++23 -Wall -Wextra`; default suite **196 passed /
   15 skipped**. Review + merge (independently re-run `cpp/parity/gumbel_logic.py`
   first, per "workers can't blind-verify"). **Then 1b** (the deferred half).

3. **1b Gumbel mixed-precision — NOT started.** The known hazard: the in-search
   masked-softmax prior is float32 and `v_mix`/`improved_policy`/`sigma_scale` mix
   **float32-prior × float64-Q** (`chocofarm/az/value_target.py:~209-280`, which the
   code calls byte-identity-critical). A uniform-precision port diverges beyond 1e-4
   on near-tie inputs. 1a left `1b SEAM` comments at the three spots in
   `cpp/src/gumbel.cpp` (evaluate/prior, v_mix product, improved_policy/masked-softmax
   argmax). 1b makes the precision exact + adds a near-tie/fine-input parity.

## Merged on `main` (51b13b9) — the completed arcs

- **Typing:** `mypy --strict` gate enforces 40 modules (Stage 1 core + the env↔Policy
  seam + solvers/bounds + medium az leaves + arrangement + eval). The 5 jax/numba
  boundary az modules (`mlp`, `exit_loop`, `train_value`, `worker`, `gumbel_search`)
  are **Stage 4, HELD** for the maintainer to weigh the jax/numba boundary. Gate:
  `tests/test_mypy_strict.py`.
- **cpp ADR-0012 P9** compliance + modernization (C++20→23): `create()`/`std::expected`
  factories, `std::optional`/`std::span`, RAII, invariants as assert/abort.
- **#23** Py↔C++ wire/result-format **drift net**: SSOTs `chocofarm/az/{wire_spec,
  result_spec}.py` + C++ mirrors; codecs derive from them; `tests/test_wire_drift.py`
  fails the default suite on a constant OR codec-dtype drift (mutation-verified).
  Codegen top-rung (generate the mirror headers) deferred — recorded in BACKLOG with
  the concrete firing trigger (when the cpp build enters a gate).
- **#27** Shape B net port: Python `chocofarm/az/inference_server.py` (greedy-drain
  batched ZMQ server) + the C++ `ZmqNetClient` (the one under scrutiny).
- **ISMCTS** C++ port: reviewed `trustworthy-mergeable`
  (`docs/notes/ismcts-port-review-2026-06-16.md`); both verification coverage holes
  closed with executed discriminating tests (tie-break 128/128 + 14/128 mutant;
  multi-belief split `cpp/parity/ismcts_multiworld.py` 192/192 + 40/192 mutant).
- **NMCS parity retired** (skipped in `tests/test_cpp_runner.py`) — validated
  repeatedly, nmcs-init milestone far off.
- Design records: `docs/design/cpp-batched-search.md` (the async work-stealing loop),
  `docs/design/zmq-inference-service.md` (Shape B). `BACKLOG.md` (deferred items).

## The 3→2→1 plan (the current build sequence) and where it stands
- **#3** mechanize the wire contract — **DONE** (#23).
- **#2** C++ `ZmqNetClient` — **merged but UNDER SCRUTINY** (TL;DR #1).
- **#1** C++ Gumbel-AZ search port — **1a DONE/unmerged, 1b pending** (TL;DR #2/#3).

## consult-004 — EXPUNGED from history; do not resurrect
A proportionality-firewall consult ("consult-004") was run by the prior session with
a **biased, exonerating prompt** ("assume the corpus is structurally compromised, then
disprove it" = operationally "prove it sound") — self-review of docs the same engine
authored — and returned a scoped-wrong "0 violations / nothing to discard" verdict.
It was **expunged from git history** (main force-pushed `b9705e2`→`51b13b9`). Do not
restore it or treat its verdict as authority; its lens (prose mechanism-downgrade)
never even asked the speculative-generality question that the ZmqNetClient raises.

## Standing preferences + lessons (persisted in the memory dir)
`no-time-estimates` (frame by complexity/risk, never calendar); `reports-to-origin`
(push commissioned reports/designs for review); `behavioral-equivalence-gate` (float32
~1e-4, not byte-identity; RNG behavioral-only); `green-push-discipline` (exit-code-gate
every push, fresh cpp build before parity, opt-in cpp tests); `workers-cannot-blind-verify`
(subagents have no Agent/Task tool → run independent review from the main loop, not in
the worker's own prompt); `parity-tests-run-selectively` (glacially slow; run a
component's parity only when its code changed); `delegate-implementation-lean-audits`
(stay an orchestrator; keep verification proportionate — the cost is redundant audit
deliberation); `use-clangd-and-dmypy`. **New, unwritten-to-memory lesson from this
session:** the prior session defended a weak mechanism (the blocking ZMQ client) with
the exact P7-forbidden proportionality shape, and ran an ass-covering self-audit to
exonerate it — both are the failure modes to avoid; the ZmqNetClient is the live
instance to fix.

*Public Domain (The Unlicense).*
