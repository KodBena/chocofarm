# BACKLOG

Consciously-deferred work, recorded so it isn't lost. NOT a live task queue (that's the commit
log + branch state) — these are items postponed on purpose, with enough context to pick up cold.

## ISMCTS port — test hardening (deferred 2026-06-16)

The C++ ISMCTS port is merged and independently reviewed `trustworthy-mergeable`
(`docs/notes/ismcts-port-review-2026-06-16.md`). Both verification-coverage holes the review found
are already closed with discriminating *executed* tests — the `_ucb_select` tie-break (the review's
integer-leaf run: 128/128, 14/128 mutant control) and the multi-belief sub-child split
(`cpp/parity/ismcts_multiworld.py`, committed: 192/192 parity, 40/192 `belief_key`-collapse mutant
control vs 0/240 on the old `bw[0]` check). Remaining is test-only; production `ismcts.cpp` is untouched:

- **(a) Permanent integer-leaf tie-forcing fixture.** The `_ucb_select` insertion-order tie-break was
  verified by the review's *ad-hoc* run, not a committed test. Make it a permanent fixture under
  `cpp/parity/` (mirroring `ismcts_multiworld.py`: integer leaf FIFO + a `>`→`>=` / sorted-key mutant
  control) so it's reproducible and regression-gated.
- **(b) Soften the over-claiming docstrings.** `cpp/parity/ismcts_logic.py:22-23/:194-196` and the
  ISMCTS asserts in `tests/test_cpp_runner.py` claim "UCB select ... covered"; the float-leaf grid
  exercises the UCB arithmetic + availability denominator but NOT the insertion-order tie-break (that's
  (a) + the multiworld fixture). Narrow the claims to what each test actually proves.
- **(c) Wire the new opt-in fixtures into `tests/test_cpp_runner.py`** — `ismcts_multiworld.py` and the
  (a) tie-forcing fixture, behind `CHOCO_RUN_CPP`, mirroring the existing opt-in pattern.
- **(d) Systemic aggregate-methodology fix (shared `cpp/parity/parity.py`).** The aggregate parity
  discards raw per-episode data (`tempfile.mktemp` + `os.unlink`) and uses an uncorrected 6-statistic
  3σ gate (~1.6% family-wise false-fail, low sensitivity). Persist raw rows under `~/w/vdc` (CLAUDE.md
  "never discard experiment output"), and Bonferroni/Holm-correct the gate. Scope: ISMCTS + Random
  (NMCS parity is retired — below).

## #23 wire/result drift net — promote the floor to codegen when the C++ build lands (deferred 2026-06-16)

The Python↔C++ wire frame (`wire_spec.py`) and result blob (`result_spec.py`) are mechanized against
silent drift by `tests/test_wire_drift.py` (one SSOT per layout; always-on layout-agreement +
codec-derives-from-spec legs that fail `pytest tests/ -q` on a format-constant or codec drift; an
opt-in `CHOCO_RUN_CPP` C++ golden round-trip). Per ADR-0012 P7's hierarchy (generate/compile-from-one-
source > build-time lint > runtime parity), the always-on test is the **floor** (a lint failing the
default gate) and the golden is the **backstop** — the **top rung (codegen) is deferred for a concrete
reason: the C++ consumer doesn't exist yet** (the `ZmqNetClient` + the redis-client `cpp/` build are
deferred to the P9 `cpp/` pass; there is no `cpp/build/` in any gate). When that pass lands and the C++
side is built in a gate:

- **Generate `cpp/include/chocofarm/{wire_spec,result_spec}.hpp` from the Python SSOT** (a tiny
  build-step that emits the `constexpr` mirror from `wire_spec.py`/`result_spec.py`), so the mirror is
  *derived, not hand-written* — closing the residual gap that the headers are hand-authored today,
  joined to the SSOT only by the runtime test. The drift test stays as the backstop.
- **Add the one-line C++ cross-check** `prod(shape) * sizeof(double) == len` in `transport.cpp::
  parse_manifest` (today C++ derives a weight's element count from `len/sizeof(double)` and Python from
  `prod(shape)`; consistent only because one writer emits both — assert it).

## Possible cpp refactor (minor, non-blocking)

- A shared `Sampler` (just `sample_world`) under `WorldSource` (NMCS) and `ISMCTSSource` (ISMCTS),
  which currently each declare it. Review-clean as-is; extract only if a third search wants it.
- **Audit `using` type aliases for phantom/strong types.** Review where the cpp uses `using` for bare
  aliases — especially the many `int` indices (action slot, world index, `action.i`, `belief_key`
  fields) — and consider whether a phantom-like type template (a tagged newtype, e.g.
  `template<class Tag> struct Idx { int v; };`) is more appropriate, so semantically-distinct ints
  can't be silently mixed. Postponed for token-saving; revisit when the cpp type surface is next open.

## ZDD belief arm — close the per-leaf value-semantics gap (deferred 2026-06-17)

The opt-in ZDD belief arm (§B.4(b), `CHOCO_BELIEF_ZDD`; `docs/design/cpp-belief-zdd-onramp.md`) is
**landed and sound**: runtime-viable (no OOM), bit-exact (the flat-vs-ZDD FEATURE A/B + the
construction-order-invariance net in `belief_sweep_oracle_check.cpp`), free of O(nb) value ops. Two
value-semantics footguns the head-to-head exposed are already fixed — the per-descent full-arena
copy/OOM (`compact()` + transient hash-cons, commit `5391c59`) and `operator==`'s O(nb) `members()`
double-enumeration (canonical structural compare, commit `b826baa`). What remains is a **third layer**,
deferred on purpose:

- **Finding** (head-to-head vs bitset, K=512; profiles under `~/w/vdc/chocobo/profiles/h2h*`): ZDD costs
  **~10× the client CPU per leaf** of bitset (perf-sample volume at matched ~128K leaves: ~11 MB vs
  ~1.1 MB), and **~70% of it is allocation churn** (`_int_malloc`/`memset`/free) intrinsic to maintaining
  the diagram as a *copyable value* — `compact()`'s fresh+remap vectors (+ memset-init), `seed_unique()`'s
  per-restrict hashtable rebuild, the per-descent `nodes_` copy. The diagram *math* is cheap (~10%).
  Bitset sidesteps all of it (a ~2 KB inline `memcpy`, zero per-op alloc). The gap is **structural (the
  copyable-diagram machinery), not algorithmic**.
- **Deferred work:** a workspace / copy-on-write-arena refactor — reuse the `compact()`/`seed_unique()`
  scratch (the `remap` vector, the hash table) across mutations via a per-thread workspace, and avoid the
  per-mutation/per-copy `nodes_` reallocation (e.g. a shared immutable arena with COW on restrict).
- **Why deferred (low-ROI here):** on the live instance bitset wins decisively regardless (its
  masked-popcount is cheap + SIMD); ZDD's role is the **large-N hedge** (a non-enumerable universe where
  the bitset gate can't apply) — a regime this instance isn't in. The **alloc campaign** (the production
  bitset path's #1 cost, ~30%) is the higher-value next step, and its workspace/buffer-reuse techniques
  are exactly what this refactor would also need — so it lays the groundwork.
- **Acceptance when resumed:** bit-exact (the A/B + construction-order net stay green); the per-leaf
  client-CPU gap closes (the alloc-churn share collapses, the diagram math comes to dominate the ZDD
  self-time); no OOM at the production search config (`n_sims=256 max_depth=24`). Feeds the per-nb
  dispatch decision in `docs/design/cpp-belief-dynamic-rep-selection.md`.

## Belief/eval caches are EPHEMERAL — cross-search sharing was the intended semantics (deferred 2026-06-20)

**The fact (record so it isn't lost):** the search's belief-keyed caches are **per-search / ephemeral** —
the Gumbel transposition table (`gumbel.hpp` `children` keyed by `(action-slot, belief_key)`) and the
within-search net-eval reuse (`gumbel.hpp:127` "prior/value/legal are the net's cached evaluation at this
belief (one forward, reused)") are rebuilt **fresh every ply** (each decision is a new tree). The ISMCTS
table is the same. So a belief evaluated in ply *k*'s search is **re-evaluated** in ply *k+1*'s search,
even though the belief evolves slowly and consecutive plies' search trees overlap heavily.

**This was never intended.** The intended semantics is **cache SHARING ACROSS SEARCHES** — a belief's
(prior/value) evaluation, once paid, reused by every later ply's search that re-encounters that
`belief_key` — so the net is evaluated once per distinct belief per *episode*, not once per distinct
belief per *ply-search*. The current ephemeral form pays redundant remote leaf evals (the wire path's
transport-per-decision = unique `belief_key`s in *that* search, not across the episode).

**How it surfaced (2026-06-20):** the HPO warm-pool / decision-budget bench work. The measured throughput
turned out to depend on the belief-size distribution precisely *because* transport-per-decision is the
per-search unique-leaf count — a dependency a cross-search cache would dampen. (It also means a faithful
HPO measure must reproduce the policy's belief distribution — the faithful-warm-pool + no-early-exit
benchmark-search work this entry came out of.)

**Deferred because:** it is a correctness-preserving *efficiency* change to the live search hot path
(cross-search cache lifetime + invalidation on belief mutation) with its own parity obligation (bit-exact
search output, P6) — out of scope for the HPO tooling that exposed it. **Acceptance when resumed:** the
f64/f32/jax search-output parity stays green; a measured drop in remote leaf-evals per episode at the
production config; cache coherence held on the rebind-not-mutate invariant (ADR-0001).

## leaf-eval — the witness/port direction + the framework↔instance question (recorded 2026-06-22, hold for a disciplined resolution)

Context: the leaf-eval tool computes a model's throughput LOWER BOUND. The maintainer's framing
(2026-06-22), now the spine of `tools/analysis/leaf_eval_bound/MANUAL.md`: a model-bound is a
*denotational conjecture* contingent on the model being faithful — it MOTIVATES, it does not REFUTE an
empirical claim like "~200 DPS is the roof." The refutation is *operational*: a WITNESS — a real cycle
that runs and clears the number. The bridge is that the benchmarks ARE the operational semantics of the
model's primitives, so a model should *lower / port* into a runnable end-to-end cycle that witnesses the
floor (MANUAL §2.1). Two questions fall out, **worth answering but deliberately held** — the maintainer's
call is to move ad-hoc first (next: the mereological loop explaining WHY the current implementation
underperforms) and let that work inform a disciplined resolution, rather than pre-committing the
architecture now:

- **Q1 — what is "the DSL" / what does "port the models back into Python/C++" mean?** (a) a composition
  harness *behind today's `f`-functions* — compose the already-benchmarked stages per `f`'s structure
  into a runnable cycle and clock it; or (b) the stronger reading — a model authored **once** and lowered
  to *both* the bound and an executable (a real DSL with two backends, the full force of "the benchmarks
  entail operational semantics"). The reading chosen changes what gets built.
- **Q2 — the witness sequence.** Proposed (not started): first *assess the lowering* (read-only — does
  every model term have a *composable* operational bench, i.e. is the gap from "benches a stage in
  isolation" to "runs the stage end-to-end" mechanical, and where is the DSL short?); then build the first
  witness for one model as the proof; then generalize. "It should be easy" is a testable claim — true iff
  the DSL is expressive enough that lowering is composition, not rewrite.
- **The framework↔instance signal (the maintainer's corollary).** De-biasing the MANUAL (2026-06-22)
  exposed that the conceptual sections and the model skeleton had smuggled the specific producer/serve/
  transport cycle in as canon; the fix pushed the cycle-specific parts to placeholders, keeping only the
  framework boilerplate concrete. That the manual *can* be taught instance-free is good — but that every
  current model shares one spine, and a concrete model kept wanting to sit at the center, is the tell the
  maintainer flagged: the cycle-modeling **framework** and the specific cycle **instance** may still be
  entangled in the tool. If a future need forces the concrete model back into the framework's core, that
  is the trigger to continue the refactor in a *separate-framework-from-instance* direction — direction
  currently undetermined ("I know not where, at the moment").

## serve drain overshoots max_batch → AOT crash at high overcommit — FIXED 2026-06-23 (residual: oversized-single-request chunking)

`InferenceServer._drain` (`chocofarm/az/inference_server.py:545-562`) checks the cap at request-boundary
granularity — `while total_rows < self._max_batch` (`:548`) runs *before* the recv, then `total_rows +=
X.shape[0]` (`:562`) adds the whole next request — so a request straddling the cap overshoots (e.g. 480 + 96
= 576 > max_batch=512). `run_microbatch` then pads only `if pad_to > B` (`:271`), so a 576-row batch hits the
512-AOT-compiled forward → `TypeError: float32[512,241] called with [576,241]` (`:445`). This **violates the
invariant the code itself documents** (`:546-547` "up to the max_batch cap"; `:269` "the drain caps the total
at max_batch, so this only ever pads UP") and is an ADR-0002 fail-loud breach (a cryptic JAX crash, not a clear
config error). **It is a PYTHON serve-path bug** — the producer is C++, the drain/forward is Python.

- **Repro:** `lab_harness.py --trees-per-thread N` with slots (= threads·N·⌈pool_batch/threads⌉) > max_batch,
  drain-all + padmax. N≥3 at pool_batch=192 / 3 threads (576 slots) crashes; N=1,2 (192, 384) are safe.
  Artifacts: `~/w/vdc/chocobo/runs/control_lab/step3-nsweep/`. Detail: `docs/notes/leaf-eval-loop/step-3-nsweep.md`.
- **Why it blocks work:** it caps the overcommit fill lever below max_batch, so the model's full-fill ceiling
  (B→512) cannot be tested in the static drain-all path — the open question the N-sweep was meant to answer.
- **Fix (small, Python, fully-visible file):** partition the drained requests so each forward ≤ max_batch —
  either `_forward_groups` chunks each group to ≤ max_batch, or `_drain` defers a straddling request to the
  next forward (a 1-request lookahead buffer). `_scatter` (1:1 in drained order) is unaffected.
- **Residual:** a SINGLE request whose `B_i` alone > max_batch (per-thread issue > 512, e.g. N≥9 at 3 threads)
  needs the chunked-forward path (forward in ≤max_batch chunks, concatenate the real rows) OR a loud reject —
  handle or fail loud, never silently crash.
- **Status: FIXED 2026-06-23** (maintainer-authorized): `_drain` now caps at max_batch by deferring a
  straddling request WHOLE to the next drain (a 1-slot `self._pending` lookahead, keeping its 1:1 reply); a
  single request wider than max_batch is loud-rejected (ADR-0002). Regression test (default suite):
  `tests/test_zmq_inference.py::test_drain_caps_at_max_batch_and_defers_straddler` (+ the reject test).
  Verified: serve tests green; the N-sweep N=3,4 (576, 768 slots) now run clean (`step3-nsweep-fixed/`).
- **RESIDUAL (still deferred):** a SINGLE request whose `B_i` > max_batch is loud-rejected, NOT chunked. The
  chunked-forward path (forward an oversized request in ≤max_batch chunks, concatenate the real rows) is the
  remaining work — needed only at per-thread issue > max_batch (e.g. N≥~8 at 3 threads, pool_batch=192).

## producer pool-warmup fails at very high overcommit (zmq EAGAIN) (found 2026-06-23)

At `--trees-per-thread 6` (1152 in-flight slots; pool_batch=192 / 3 threads) the C++ producer's pool warmup
fails — `wire-ab-bench: FATAL: pool warmup failed: WireLeafPool::poll: zmq_msg_recv failed: Resource
temporarily unavailable` (a zmq EAGAIN during the warmup build) — the producer exits rc=1 before streaming and
`lab_harness` fails loud ("producer exited early"). N≤4 (≤768 slots) warm up fine. This is a **PRODUCER-side**
limit (the C++ `WireLeafPool` warmup), independent of the server-drain fix above. Likely a zmq HWM / recv
timeout during the large pool build, or a slot ceiling. It caps the static N-sweep at N≤4 (B≈277) for now.
Fix/investigate: the producer pool-warmup poll (raise the HWM / lengthen the warmup recv timeout / stage the
build) — or accept N≤4 as the static range. Artifacts: `~/w/vdc/chocobo/runs/control_lab/step3-nsweep-fixed/N6/`.

## `Windowed` measurement type at the source — the full foreclosure of the rate-window-mismatch class (found 2026-06-24)

The reference-140k faceplant (DB finding #12) was a rate computed as `whole-call-leaves / measure-window-wall`
— numerator and denominator from *different* intervals. The ADR-0000 disposition added a construction-time
guard in `throughput-lab/harness/exp_db.py` (`Reading.__post_init__`): a rate field (`leaf_rows_s`, `dps`,
`forwards_s`, `lpd`) is rejected unless its count + time-span are present and the rate equals their quotient.
That forecloses the **recorded** artifact (a provenance-less rate, the actual shape of readings 16/17) and
makes a cross-window rate *auditable* (both operands are stored). It does **not** yet make the cross-window
case *unrepresentable*: if a caller stores whole-call leaves *and* a measure-window wall *and* their (matching)
quotient, the guard passes. The full foreclosure (ADR-0012 illegal-states-unrepresentable) is a **`Windowed`
value type produced at the MEASUREMENT SITE** — `Windowed(count, elapsed_s, window_label)` capturing count and
span over **one** interval as a unit, with `.rate` the only way to a rate; `Reading` would store `Windowed`s,
so mixing two intervals needs two objects and there is no constructor that crosses them. Deferred because it
touches every measurement producer (the harness, the C++ bench RESULT parse, the ad-hoc probes), not just the
recorder. Until then: the guard + the discipline that **a baseline/target is a stamped reading, not a prose
number, and a finding cites a `reading_id` not a bare figure** (the executive-lapse half of finding #12).

## Fiber producer per-fiber resident footprint — let high fiber counts fit (found 2026-06-25)

`tlab-real-producer`'s greedy/round-sync fiber drivers keep EVERY `--fibers K` TreeState live (parked on a
leaf) for the whole run — `run_thread_fiber_episodic` starts all K up front (`ready.push_back(i)` over K) and
re-arms each on completion — so producer resident memory is `threads*K*per_fiber`. VALGRIND/massif (128 fibers,
n_sims 256, 323 MiB peak) attributes per_fiber to TWO per-fiber costs, both confirmed in the heap:
- **the boost.context fiber STACK** — `fixedsize_stack(512 * 1024)` per TreeState (`fiber_tree.hpp:91`),
  massif's largest bucket (`TreeState::start`, 67 MiB / 128 == exactly 512 KiB each). The Gumbel search
  recurses at most `max_depth` (24) deep with modest frames, so 512 KiB is almost certainly oversized;
- **the per-decision Gumbel node ARENA** — the per-policy `monotonic_buffer_resource` (`gumbel.hpp`), 256 KiB
  inline floor + ~9 KiB/sim grown (n_sims 16/64/256 -> 0.26/0.95/2.4 MiB/fiber, MEASURED RSS sweep).
At the banked `--fibers 1024 --n-sims 256` this is ~3 MiB/fiber == ~3 GiB per producer; four concurrent
producers' bursty peaks align past an 8 GiB box and the OOM killer SIGKILLs one mid-run (REPRODUCED 4-up:
rc=137, MemAvailable->19 MiB; slab/page-cache flat, so it is process heap not transport — refuting the
original finding-#22 "NOT memory" read, which was a single-producer/under-sampled observation that missed the
4-up spike alignment). An ADMISSION GUARD now FAILS LOUD (`real_producer.cpp` main, ADR-0002) instead of a
silent SIGKILL, but it REFUSES the banked config rather than making it run. The deeper fixes that would let
1024 fibers fit (each independent, mandate-aligned, deferred only because they touch the validated SHARED
fiber core `fiber_tree.hpp` used by three drivers + the Option-A proof, so out of scope for a producer-only
stability fix — ADR-0004 minimal-touch):
- **(a) right-size the fiber stack** (e.g. `protected_fixedsize_stack` or a measured smaller `fixedsize_stack`)
  — measure the search's true stack high-water first (n_sims 256, m 24, max_depth 24); a 64–128 KiB stack
  would cut the 1024-fiber footprint by ~400 MiB. Re-run the bench + Option-A parity after.
- **(b) bound the count of fibers with a LIVE arena** to the in-flight budget rather than K — only
  `inflight_msgs * msg_rows` leaves are ever mid-flight, so most of the K live arenas are idle parked state;
  decoupling "coalescing batch width K" from "K resident arenas" is the structural win (and the ADR-0000
  invariant that makes the OOM unrepresentable: resident is O(in-flight), not O(K)).

## Retired

- **NMCS parity tests** marked `skip` in `tests/test_cpp_runner.py` (2026-06-16): validated repeatedly,
  and the nmcs-init milestone (a 2-level NMCS to initialize an AZ run before switching to ISMCTS) is
  far off. Re-enable when that work resumes.

## tlab greedy-episodic Gate-A liveness floor — fixed in the greedy pipe; audit the siblings (2026-06-25)

Integrating the control-lab's 16 methods onto tlab-real-producer's async Gate-A control plane
(`throughput-lab/harness/run_control_lab.py`) surfaced a producer-side **liveness defect**: the greedy
episodic driver's discretionary issue gate (`real_producer.cpp` ~L381 `... && (!ctl || ctl->may_issue(idx))`)
had **no forced-flush floor**. A hard-gating controller (e.g. `ready_threshold2` at 256 fibers) denied
every issue, the in-flight count drained to 0, and `if (in_flight == 0) break;` **terminated the producer
thread mid-trial (rc=0)** — violating the lab's own stated contract (`lab_server.py` header: "the
producer's forced-flush stays the depth-1 liveness floor … a gate-everything method cannot wedge the
producer"). Fixed minimally + measure-verified: when `in_flight==0 && !ready.empty()`, push ONE group
regardless of the gate (the depth-1 floor `runner_wire_batched.cpp` already guarantees). Control-OFF is
byte-unchanged (the branch only fires when `ctl` denies). After the fix all methods run a full trial, the
watchdog correctly flags + survives `malfunctioning`, and trials egress to `control_research`.

Deferred / to audit (ADR-0011 Rule 4 — quantify over the class, not the instance): the OTHER drivers
(`run_thread_fiber` round-sync, `run_thread_fiber_greedy`, `run_thread` non-fiber) do NOT carry `ctl` at
all, so they cannot be gated today — but if the Gate-A hook is ever extended to them, the same forced-flush
floor must come with it. The C++ liveness invariant now lives in TWO hand-authored copies
(`real_producer.cpp` greedy-episodic + `runner_wire_batched.cpp` refill) with no shared home; hoisting "a
gated issue point always carries a depth-1 forced-flush floor" into one helper both call (so a new gated
driver cannot be born without it — the net, not the patch) is the durable fix. NOT done in this session
(it touches the shared `runner_wire_batched.cpp` hot path under partial visibility — ADR-0004; a C++ helper
hoist across two files + a hot-path edit is its own scoped change, not a producer-stability sub-task). The
per-instance floor IS in place and verified; this records the remaining class-net.

Done in this session (the watchdog twin, the Python analogue of the above): the method-watchdog safety
contract (gate-shape validator + malfunction tally) was being COPIED into `run_control_lab.py` from
`lab_server.py` — caught by an out-of-frame hack-rationalization review. It is now ONE shared home,
`control_lab/watchdog.py` (`MalfunctionRecord` + `validate_gates`), imported by BOTH control wires (ADR-0012
P1). It is a SEPARATE module from `lab_server` for a concrete reason: `lab_server` imports `StageAServer`
(JAX) at module load, so the async policy peer would otherwise drag the inference stack in just to validate
a gate vector.

Open schema smell (not pursued — out of scope; would alter the shared `lab_trial` table the per-forward
harness also writes): `run_control_lab` stores its leaves/sec score in the `dps_*` columns and the window
leaf-delta in `n_decisions`, disclosed only in `lab_session.notes` (free prose). A typed `metric_kind`
discriminator column on `lab_trial` (e.g. `dps` vs `lps`) would make "this throughput is leaves/s not
decisions/s" a queryable field rather than a sentence a reader must find. Deferred as a lab-store schema
change, surfaced here so it is not lost.

*Public Domain (The Unlicense).*
