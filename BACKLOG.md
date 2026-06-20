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

## Retired

- **NMCS parity tests** marked `skip` in `tests/test_cpp_runner.py` (2026-06-16): validated repeatedly,
  and the nmcs-init milestone (a 2-level NMCS to initialize an AZ run before switching to ISMCTS) is
  far off. Re-enable when that work resumes.

*Public Domain (The Unlicense).*
