<!-- docs/design/cpp-belief-rep-scoping.md
     Scoping report for the GATED multi-representation belief design (flat base + bitset fast path;
     ZDD a third arm per the 2026-06-17 reversal-of-the-reversal). Produced by the belief-rep-scoping
     workflow (3 mappers -> synthesis -> adversarial critique -> final), 2026-06-17, branch
     cpp-actor-online-reconfig. Phases preserved in docs/notes/belief-rep-scoping-2026-06-17-phases.md.
     Public Domain (The Unlicense). -->

All confirmed. Key additional finding: `base_value` takes `bw` **by value** (`std::vector<uint32_t> bw` at policy.cpp:114 / policy.hpp:129) — it's a copy taken to mutate during the playout. That's a copy site that becomes a `Belief` value copy (constant 1.9 KiB for bitset). And `gumbel.cpp:233` confirms `evaluate` routes through `fb_.build(loc.pt, bw, ...)` — through the seam.

The filter bench at belief_filter_bench.cpp:51 (`branchless_ref`), 104/106 (`filter_inplace` to build events), 137-138 (bit-exact net idiom vs branchless) is confirmed as flat-arm-only bench territory.

I have verified everything material. The critique's findings all check out against source:
- runner.cpp:27, :98 are genuinely omitted nb-traffic sites
- node maps keyed by fingerprint (gumbel.hpp:113, ismcts.hpp:90) guarantee no rep in node arena
- variant size blowup (1.9 KiB inline paid in flat fallback) is unaddressed by the report
- gate threshold conflates a derived quantity (mask bytes from N/K/nD) with a machine constant (cache budget)
- the `*inv` convention is pinned by the oracle (lines 59-63) and must be reproduced
- order-equivalence is documented in the fixtures themselves (ismcts_dump.cpp:11, nmcs_dump.cpp:10) — the report over-hedges
- the belief_key triple is bit-identical across reps, not merely collision-resistant
- nmcs.cpp:46 is a second descent-copy site (eval_move), in addition to :66

I also confirmed one critique-adjacent detail it didn't fully flag: `base_value` takes `bw` by value (policy.hpp:129), another copy site.

Now I'll produce the final scoping report folding in every valid critique point.

# SCOPING REPORT — Dense Bitset Belief Behind a Gated Two-Representation Seam

**Branch:** `cpp-actor-online-reconfig` · **Status:** read-only scoping, no code changed · Decision settled per `/home/bork/belief_bitset_decision_reversal.md` (read end-to-end). This report tells the implementer how to host BOTH the flat `std::vector<uint32_t>` (general base + fallback) and a dense bitset (gated fast path) under one search, with no rep leak into the search and no per-op overhead that erodes the win. Every `file:line` below was re-verified against source; the adversarial critique's valid points are folded in and its refutations applied.

---

## 0. The one fact that frames the whole design

The belief is **already cleanly seamed for reads and mutation**, and **never crosses the wire or the serve boundary**. Re-verified:

- The runner builds the belief from `env.worlds()` (`cpp/src/runner.cpp:24`), mutates it only via `env.apply` (`runner.cpp:68`), featurizes it via `fb.build` + `legal_mask` (`runner.cpp:49-50`), and `write_results` ships only the four float32 blocks X/PI/M/Y (`runner.cpp:177`) — the belief is never serialized. `serve.cpp` only calls `run_episodes` (`cpp/src/serve.cpp:187`); the sole belief mention there is a comment about world-set/distance-array allocation (`serve.cpp:100`). **A belief never crosses the serve/runner boundary.** This contains the blast radius: the rep change is a C++-internal substitution; the wire/result contracts (P7) are untouched.
- `SearchTask::bw` (`cpp/include/chocofarm/search_runtime.hpp:53`) is a by-value member field travelling to the runtime via `std::span<const SearchTask>` — but it never leaves the process; it is the in-process task description.

So the entire change is internal. The work is: define ONE belief value type, route the handful of raw-vector pokes through it, gate the rep at env construction, and re-key the caches.

---

## 1. TOUCHPOINT INVENTORY

Verified by a tree-wide grep for `bw[`, `.front()`, `.back()`, `.size()`, `.empty()`, range-for `for (… : bw)`, `.data()`, `.begin()`, `sample_world`, and `= bw` across env, policy, features, the three searches, runner, serve, and the fixtures. There are exactly **two categories**.

### 1A. GOES THROUGH THE SEAM (changes by a type substitution only)

These name `bw` only to hand it to an `env` method, a `WorldSource`, the fingerprint, or `FeatureBuilder::build`. They change by *retyping* the parameter `const std::vector<uint32_t>&` → `const Belief&` (and the by-value copies → `Belief`).

- **Env reader/mutator API** — `marginals` (`env.hpp:81`), `legal_actions` (`env.hpp:88`), `informative` (`env.hpp:112`), `filter_treasure`/`filter_detector` (`env.hpp:97-98`), `apply` (`env.hpp:93`). Bodies (`env.cpp:67-103,120-165`) move *inside* the seam (§2). Note `apply` mutates `bw` **in place** by reference (`env.hpp:93`, `env.cpp:133`) — the seam's filter ops must therefore take a *mutable* `Belief&` (§2, §6 risk 8).
- **Policy contracts** — `Policy::decide`/`decide_target` (`policy.hpp:60,70`), `RandomPolicy` (`policy.hpp:82`), `GreedyBase`/`GreedyStopBase` (`policy.hpp:99,110`), `candidate_actions` (`policy.hpp:120`), `base_value` (`policy.hpp:128`). Bodies call only env methods (`policy.cpp:39,61,83`). **`base_value` takes `bw` BY VALUE** (`policy.hpp:129`, `policy.cpp:114`) — a deliberate copy it mutates during the playout; this becomes a `Belief` value copy (constant 1.9 KiB for the bitset arm — a side benefit, §6 risk 3).
- **Search contracts + hot paths** — Gumbel `decide`/`run_search`/`decide_with_target`/`evaluate`/`descend`/`simulate_root_action` (`gumbel.hpp:144,158,167,188,208,214`; `gumbel.cpp:230-235,330-403`); ISMCTS `decide`/`run_search`/`iterate` (`ismcts.hpp:104,112,123`; `ismcts.cpp:120-197`); NMCS `decide`/`search`/`playout`/`eval_move` (`nmcs.hpp:80,85,91,100`; `nmcs.cpp:40-164`). `gumbel.cpp:233` confirms `evaluate` routes through `fb_.build(loc.pt, bw, …)` — through the seam. The node-local copies become `Belief` copies (§3 value-semantics finding): `nbw = bw` / `cur_bw = bw` at `gumbel.cpp:352,389`; `ismcts.cpp:134,159`; `nmcs.cpp:46,66` (note **two** NMCS copy sites — `:46` in `eval_move`, `:66` in `search` — the prior draft cited only `:66`).
- **Featurizer entry** — `FeatureBuilder::build` (`features.hpp:107`; `features.cpp:258`), `legal_mask` oracle (`features.hpp:50`; `features.cpp:60`), `legal_mask_from_features` (a pure float-span slice — `features.cpp:302`, unaffected).
- **Fingerprint call sites** — `gumbel_belief_key` (`gumbel.hpp:90`), the `belief_key(nbw)` calls in all three searches (`gumbel.cpp:356,393`; `ismcts.cpp:139,163`) and in `belief_feats_` (`features.cpp:332`) — these *call* the fingerprint; they don't compute it.
- **Init** — `bw = env.worlds()` (`runner.cpp:24`; `gumbel_dump.cpp:201`; `ismcts_dump.cpp:142`; `nmcs_dump.cpp:107`; `mask_dump.cpp:78`) becomes `Belief b = env.full_belief()`.

### 1B. POKES THE RAW VECTOR (the actual work)

Six poke kinds (the prior draft's five plus the omitted runner shrinkage site), each absorbed behind the seam:

| # | Site | Poke | Resolution |
|---|------|------|-----------|
| **L1** | `RngWorldSource::sample_world` `policy.hpp:152-154` | `bw.size()` + `bw[pick(rng_)]` | **Move sampling into `env`** as `env.sample_world(const Belief&, rng)` (note lines 132-148). Only *production* leak. |
| **L2** | `belief_key` `belief_key.hpp:22-25` | `bw.size()`, `bw.front()`, `bw.back()` | Becomes `env.belief_key(const Belief&)`. Bitset: `(popcount, world_at_rank(0), world_at_rank(nb-1))` via `countr_zero`/`countl_zero` (§6 risk 5). |
| **L3** | `belief_features_nonempty` `features.cpp:190-193` | `for (uint32_t w : bw)` — the O(nb·(N+nD)) sweep | The §A.4 sweep is **replaced** by masked-AND+popcount on the bitset arm (note 41-47); flat keeps the sweep. Must reproduce the `* inv` convention (§4, §6 risk 6). |
| **L4** | Fixture `sample_world` — `cyclic_gumbel.hpp:27` (`bw[0]`), `gumbel_dump.cpp:150-153`, `ismcts_dump.cpp:85-89`, `nmcs_dump.cpp:66` (`bw[0]` / `bw[raw%n]`) | scripted `bw[i]` | Need `env.world_at_rank(const Belief&, r)` (the r-th set bit → world). Mechanical (§5 step 6, §6 risk 1). |
| **L5** | `belief_cache_` value type — `features.hpp:164`; store/verify `features.cpp:335,341` | stores `std::vector<uint32_t>`; `std::ranges::equal(entry.first, bw)` | Swap stored type to `Belief`; equality → `Belief::operator==`. |
| **L6** | `runner.cpp:27,98` belief-shrinkage stat | `bw0 = bw.size()` and `1.0 − bw.size()/bw0` | Route through `env.nb(const Belief&)`. **The critique's caught omission** — a bare `belief.bits.size()` here would silently return the word count (243), not the popcount. |

**Order-equivalence note (settles the L1/L4 parity question — see §6 risk 1):** the flat vector starts ascending (`worlds_` built ascending, `env.cpp:21-38`) and `std::erase_if` **preserves order** (`env.cpp:121`); the bitset is inherently ascending (note 93-106). The kept SET *and its ascending ORDER* are therefore identical between reps. The existing fixtures already document and rely on this: `ismcts_dump.cpp:11` / `nmcs_dump.cpp:10` say "`bw[0]` (the lowest-bitmask world; itertools/combinations order is the same on both sides)" and the prefix-advance comments call `bw[0]` "the same deterministic world both languages advance by" (`gumbel_dump.cpp:205`, `ismcts_dump.cpp:146`, `nmcs_dump.cpp:111`). So the r-th flat element equals the r-th set bit for the same `uniform_int(0,nb-1)` draw: **sampling is byte-identical, established by construction.** This corrects the expert note's §Equivalence caveat (note lines 161-166), which assumed an arbitrary insertion order the flat vector does not have.

---

## 2. THE SEAM — the boundary that hosts both reps

**The seam is the `Environment`'s belief API, with sampling and fingerprinting pulled IN and the belief value type made opaque.** The env already owns world-space + masks as env-static derived data (`env.cpp:40-52`); the bitset's masks are homed there exactly like `face_masks_` (`env.cpp:42-46`, note 85-91). Op set the seam must expose, so **no caller pokes rep internals**:

**Belief value type** (opaque to callers): a `std::variant<FlatBelief, BitsetBelief>` (§3). `FlatBelief` wraps `std::vector<uint32_t>`; `BitsetBelief` wraps `std::array<uint64_t, kW64>` **plus a cached `int count_`** (the O(1)-nb obligation, §6 risk 2). Value-typed, copyable, `==`-comparable.

| Op | Replaces | flat arm | bitset arm |
|----|----------|----------|-----------|
| `env.full_belief() -> Belief` | `bw = env.worlds()` | copy `worlds_` | all-ones over nb worlds, `count_=nb` |
| `env.filter_treasure(Belief&, i, present)` | `env.hpp:97` | `erase_if` | `&= ±treasure_mask[i]`, recompute `count_` |
| `env.filter_detector(Belief&, j, positive)` | `env.hpp:98` | `erase_if` | `&= ±detector_mask[j]`, recompute `count_` |
| `env.apply(Loc&, Belief&, …)` | `env.hpp:93` | body unchanged | calls the two filters (in-place mutation through the visited `Belief&`) |
| `env.marginals(const Belief&)` | `env.hpp:81` | sweep | `popcount_and(b, treasure_mask[t]) · inv` |
| `env.legal_actions(const Belief&, collected)` | `env.hpp:88` | marginals + informative | same, via bitset bodies |
| `env.informative(j, const Belief&)` | `env.hpp:112` | two-polarity scan | `0 < popcount_and(b, detector_mask[j]) < nb(b)` |
| `env.nb(const Belief&) -> int` | `bw.size()`/`bw.empty()` (incl. **L6**) | `.size()` | return cached `count_` (NOT a recount) |
| `env.sample_world(const Belief&, rng) -> uint32_t` | **L1** | `worlds_vec[uniform]` | r-th set bit → `worlds_[idx]` (note 132-148) |
| `env.belief_key(const Belief&) -> BeliefKey` | **L2** | `(size, front, back)` | `(count_, world_at_rank(0), world_at_rank(nb-1))` |
| `env.world_at_rank(const Belief&, r) -> uint32_t` | **L4** | `bw[r]` | r-th set bit → world |
| `belief_features(const Belief&, masks, dims)` | **L3** | §A.4 sweep | masked-AND+popcount, `* inv` |
| `Belief::operator==` | **L5** | vector `==` | array `==` (count_ is derived, compare bits) |

**Critical seam principles:**
- *(P1/P7)* The bitset masks (`treasure_mask[t]`, `detector_mask[j]`) derive from the SAME enumeration + face masks the env owns (`env.cpp:41-46`). Build them in the ctor alongside `worlds_`/`face_masks_`. The identity `observe(j,w) == (w & face_masks()[j]) != 0` (`env.hpp:109`) is what makes `detector_mask[j]` derivable; the oracle (`belief_sweep_oracle_check.cpp:57`) pins it.
- *(perf, §6 risk 2)* `nb`/`empty` is the highest-traffic op — `gumbel.cpp:333,335,533`, `ismcts.cpp:186`, `nmcs.cpp:40,124,161`, `features.cpp:181,217`, and **`runner.cpp:27,37,98`** (the L6 sites). For the bitset arm this MUST read the cached `count_`, updated in `filter`, never a 243-word recount at each guard — otherwise the gate's win erodes at the guards.

---

## 3. DISPATCH MECHANISM — recommendation

**Recommend `std::variant<FlatBelief, BitsetBelief>` as the `Belief` value type, with dispatch *inside the env methods* (one coarse `std::visit`/`holds_alternative` per op), NOT at every call site.**

Substantiation (ADR-0009 — attach the reasoning, not just the conclusion):

1. **The rep cannot enter the node arena.** `GumbelNode::children` is `std::map<std::tuple<int, GBeliefKey>, int>` (`gumbel.hpp:113`) and `ISMCTSNode::children` is `std::map<std::tuple<int, BeliefKey>, int>` (`ismcts.hpp:90`) — the node graph is keyed by the **fingerprint**, never the belief value; the belief is copied descent-locally (`gumbel.cpp:352`, `ismcts.cpp:134`, `nmcs.cpp:46,66`) and passed by const-ref. That map type is what *guarantees* the variant-belief is a pure descent-local value. A `variant` is value-typed and copies its active arm cleanly, fitting the existing ownership exactly (P3 one-owner, P9 by-value).
2. **The rep is chosen ONCE per env and is invariant for the env's life**, so the variant's active alternative is constant across a whole run. The dispatch branch is perfectly predicted and paid **per env-method-call, not per world**; the expensive inner loops (the 243-word AND, the popcount) run inside the chosen arm with no per-iteration dispatch.
3. **Alternatives rejected for cited reasons:**
   - *Template the search on the rep* (`GumbelAZPolicy<BitsetBelief>`): zero dispatch but **2× search instantiation** (compile time + binary), and the rep type leaks into `SearchTask`/`SearchRuntime` as a type parameter — fighting `SearchTask` being a plain vector-storable struct (`search_runtime.hpp:51`) and violating "no rep leak into the search." The gate would become a compile-time fork.
   - *Virtual `Belief` base*: forces heap allocation + a vtable indirection per op and **breaks the cheap value-copy at nodes** (`base_value` takes `bw` by value, `policy.hpp:129`; descent copies are stack values). Worst fit.

**The `std::variant` size question (the critique's catch — now answered, not skipped):** a `variant<FlatBelief, BitsetBelief>` is sized to its largest alternative, so the 1.9 KiB bitset footprint is paid inline **even in the flat-fallback regime** (large/non-enumerable `|worlds|`, where the bitset alternative is never active). **Recommendation: accept the 1.9 KiB inline.** Rationale: beliefs are descent-local values, not held en masse (the node arena keys by fingerprint, not value — see point 1); the only live beliefs are the per-descent copies and `base_value`'s by-value parameter, a small set. A `std::unique_ptr<std::array<…>>` bitset arm would trade the inline footprint for an allocation **per belief copy** — and copies happen per descent step (`gumbel.cpp:352` etc.), so that allocation churn would erode the very win the bitset buys. The flat-fallback regime is the *feasibility* fallback, not the perf-critical path, so its 1.9 KiB-vs-24-byte inline cost is irrelevant there. (If a future profile of the flat-fallback regime ever shows this matters, the boxed-bitset arm is the drop-in mitigation — but do not pay its allocation cost speculatively.)

**Dispatch cost summary:** variant + visit-in-env = ~one predicted branch per env op, zero per-world dispatch, single search instantiation, gate stays a runtime value. **Keep the `std::visit` coarse** — dispatch once at the top of each env method, then run a pure rep-specific body; never `visit` inside the per-world loop.

---

## 4. THE GATE — where the threshold is computed and where selection happens

**The gate has TWO inputs — a derived quantity and a machine constant — and the critique is right that the prior draft blurred them. Home each honestly (P1):**

1. **Derived quantity (a pure function of N/K/nD):** the mask-storage bytes `(N + nD) · kW64 · 8`, where `kW64 = (|worlds| + 63) / 64` (note 93) — `kW64` is **derived, never the literal 243**. For the live instance (N=20, K=5, |worlds|=15504): `kW64=243`, masks ≈ 122 KiB (treasure ~38 KiB + detector ~85 KiB), per-belief 1.9 KiB (note 87-91). Computed in the `Environment` ctor alongside `worlds_`/`face_masks_` (`env.cpp:40-52`).
2. **Machine constant (a target-cache budget, NOT derivable from the dims):** a cache-residency ceiling the mask set must fit. **Home this explicitly as what it is** — a named `constexpr` carrying the target-cache figure (the i5-6600 Skylake L2, note line 64) and its rationale in the comment. Do NOT pass it off as "derived" or bury it as "122 KiB fits, ship it"; that is exactly the scattered-magic-number failure the gate exists to avoid.

**Gate predicate:**
```
use_bitset = (|worlds| enumerable)
           AND (mask_bytes(N,nD,kW64)  <=  kTargetMaskCacheBudgetBytes)   // derived ≤ machine-constant
```

**Commit the live instance lands on the bitset side:** with masks ≈ 122 KiB comfortably L2-resident and per-belief 1.9 KiB, `use_bitset_ = true` for N=20, K=5 (note 75). This is not left as an open question.

**Selection (one-time, behavior-neutral):** in the ctor, if `use_bitset_`, build `treasure_mask`/`detector_mask` and have `full_belief()`/the env belief ops take the bitset arm; else the flat arm. **No call site decides.** Because both reps are bit-exact (filters/counts/determinizations — §5), the gate is a pure feasibility/speed choice that cannot change outputs (P6/P7). For large/non-enumerable `|worlds|` (the future variant, note 179-185) the gate falls back to flat automatically — **flat is the necessary fallback, not legacy.** This is the same homing question already open for the distance matrix and detector masks (note 170-173); resolve them together.

---

## 5. CUTOVER PLAN — ordered, with bit-exact validation at each step

Introduce the seam first on the flat arm (a pure refactor, must be byte-identical), then add the bitset arm behind the gate, validating each rep against the other and the existing oracles.

**Step 0 — Confirm the order-equivalence (a CHECK, not a discovery).** Order-equivalence is established by construction (§1B note: ascending `worlds_` + order-preserving `erase_if` + ascending bitset; already the documented basis of the parity fixtures, which PASS). The A/B harness (Step 2/3) **confirms** it; it does not gate on an uncertain empirical result. Record the confirmation. The only way it goes behavioral is an implementer choosing a non-ascending flat rep (e.g. a hash-set) — out of scope, so the "N≥300 behavioral re-baseline" path fires only on such a deviation, not on this design.

**Step 1 — Introduce the `Belief` seam on the flat arm only (pure refactor).** Define `Belief = variant<FlatBelief, BitsetBelief>` with only the flat arm populated; move `sample_world` into `env` (L1); make `belief_key` an env op (L2); add `env.nb`/`world_at_rank` and route L6 (`runner.cpp:27,98`) + all guards through `env.nb`; retype all 1A signatures; swap `belief_cache_` value type (L5). *Validation:* the entire existing test + parity suite passes byte-identically (this step changes no math). Run `belief_sweep_oracle_check` (PASS unchanged), the gumbel/ismcts/nmcs parity dumps, `tests/`. This isolates the type-plumbing from the bitset math.

**Step 2 — Add the bitset arm + the gate (forced off first).** Build `treasure_mask`/`detector_mask` in the ctor; implement the bitset bodies (note 116-148) with the cached `count_`. *Validation:* extend `belief_sweep_oracle_check.cpp` (it already builds beliefs as prefixes + a strided subset, lines 104-114, and diffs `BeliefFeatures` field-by-field via `equal_features`, lines 73-83) into a **flat-vs-bitset A/B harness**: for each sampled belief, build BOTH reps, assert `marginals`/`informative`/`legal_actions`/`belief_features` are **byte-identical**. This is the strongest P6 tier (counts are exact integers, note 156-162) and the existing oracle structure transfers directly.

**Step 3 — Sampling A/B.** For a fixed RNG stream, assert `env.sample_world(flat, rng) == env.sample_world(bitset, rng)` over many draws on many beliefs. By Step 0 this is byte-identical; the harness confirms it. Also confirm `world_at_rank` agrees across reps for all `r` (the L4 fixture contract).

**Step 4 — Enable the gate; end-to-end parity.** Flip `use_bitset_` on for the live instance. *Validation:* the gumbel parity end-to-end (`cpp/parity/gumbel_*`) and the runner/serve episode traces (the `exec_slots`/`world` JSON, `runner.cpp:162-173`). These stay bit-exact; the dump fixtures' `bw[0]`/`bw[raw%n]` (L4) resolve to the same worlds via `world_at_rank`.

**Step 5 — Confirm cache behavior.** The bitset `belief_key` triple is bit-identical to the flat one (§6 risk 5), so the FeatureBuilder/gumbel/ismcts cache hit-rate and gumbel transposition behavior are preserved exactly. Confirm via the full-equality verify now routed through `Belief::operator==` (`features.cpp:335`).

**What this SUPERSEDES** (retire for the bitset arm when the gate is on):
- The §A.4 fused sweep `belief_features_nonempty` (`features.cpp:178-210`) — replaced by masked-AND+popcount. **Keep it as the flat arm's body.**
- The flat `filter_inplace` `erase_if` (`env.cpp:120-123`) — replaced by `&= mask`. **Keep as flat arm.**
- The branchless-filter and SIMD-compaction/pos-popcount rungs (note 78-81, 60-62) — **moot** for the bitset (nothing to compact); they remain the flat arm's deferred options.

**What this KEEPS unchanged:**
- The Gosper/`next_combination` march and enumeration ORDER (`env.cpp:21-38`; note 93-106) — the bitset *indexes* this same order.
- The offset cache / layout SSOT in `FeatureBuilder` (`features.cpp:131-140`) — feature *assembly* is rep-agnostic (it consumes `BeliefFeatures`, which both reps produce identically).
- `build`'s assembly (`features.cpp:258-300`), `legal_mask_from_features` (the float-span slice, `features.cpp:302`).
- The bit-exact **oracle** (`belief_sweep_oracle_check.cpp`) — extended to A/B both reps, the regression net for both.
- The gumbel/ismcts/nmcs **parity** structure — the scripted sources adapt their `sample_world` body (L4) to `world_at_rank` but feed the same worlds.
- `belief_filter_bench` (`belief_filter_bench.cpp`) — its `branchless_ref`/`filter_inplace` timing (lines 51,137-138) stays the flat arm's bench; add a bitset-`&=` timing only if a filter profile is wanted.

---

## 6. RISKS / OPEN NUMBERS

1. **The `bw[0]` / sampling parity — MECHANICAL, not a live risk (critique-corrected).** The prior draft called this "the single most important risk that could turn bit-exact into behavioral." It is not. `bw[0]` = lowest remaining combination = first set bit (rank 0); the seam exposes `env.world_at_rank(belief, r)` and the fixtures call it with `r=0` / `r=raw%nb`. Byte-exact by the same ascending order the fixtures already document (`ismcts_dump.cpp:11`, `nmcs_dump.cpp:10`). The only path to behavioral is a non-ascending flat rep — out of scope. **Downgrade to a mechanical consequence of Step 0.**

2. **`nb`/`empty` traffic — the genuine concrete obligation.** ~14 guard sites do `bw.empty()`/`.size()` (`gumbel.cpp:333,335,533`; `ismcts.cpp:186`; `nmcs.cpp:40,124,161`; `features.cpp:181,217`; **`runner.cpp:27,37,98`**). The bitset `Belief` MUST cache `count_` (updated in `filter`) so these stay O(1), not a 243-word recount. Left as a recount, the gate's win erodes at the guards.

3. **`sample_world` O(1) → O(243) (the note's open number).** Flat sampling is a single index (`policy.hpp:154`); bitset sampling scans up to 243 words for the r-th set bit (note 134-148; BMI2 `pdep`+`tzcnt` accelerates, early-exits on the containing word). **Not a profiled hotspot** — the profiles named the *belief sweep* at ~81% (`feature_compute.hpp:7`); sampling never appears. The 243-word scan is sub-µs (note 64-77); the bitset *eliminates* the 81% sweep, dwarfing any sampling regression. Frequency: per `c_outcome>0` determinization in gumbel (`gumbel.cpp:387`, c_outcome=2), per descent step in nmcs (`nmcs.cpp:44,125,164`), per ismcts iteration (`ismcts.cpp:190`). **Open: record the measured net** (almost certainly positive, P6).

4. **Node / `base_value` belief-copy cost.** Each descent step copies the belief (`gumbel.cpp:352`, `ismcts.cpp:134`, `nmcs.cpp:46,66`), and `base_value` takes one by value (`policy.hpp:129`). Bitset copy = fixed 1.9 KiB; flat copy = up to 62 KiB and variable (note 59). **Bitset is ~32× smaller and constant-size — a side benefit, not a risk.** The flat-arm vector-copy is unchanged from today.

5. **The belief-cache re-key — bit-IDENTICAL across reps (critique-strengthened).** Today `belief_key = (size, front, back)` (`belief_key.hpp:24`). Because order is ascending in BOTH reps, the bitset triple `(count_, world_at_rank(0), world_at_rank(nb-1))` is **bit-identical** to the flat triple for the same belief — not merely "collision-resistant." This is what preserves the cache hit-rate and gumbel transposition behavior exactly and lets the parity fixtures pass unchanged. Re-key in ONE place (the env op); FeatureBuilder, gumbel, and ismcts all derive it (`features.cpp:332`, `gumbel.hpp:90`, `ismcts.cpp:139` — P1). The full-equality verify (`Belief::operator==`) remains the safety net regardless.

6. **The `* inv` normalization is a load-bearing convention the bitset MUST reproduce (critique-added).** The bitset's popcount counts must feed the **settled `* inv`** normalization (`features.cpp:198-205`), NOT the original `/ nb`. The oracle pins `* inv` as THE reference (`belief_sweep_oracle_check.cpp:59-63,132`). The bitset is byte-identical to the current sweep *only because* both apply `* inv` to exact integer counts; an implementer who applies `/ nb` to bitset counts would break the oracle. Name this in the bitset `belief_features` body.

7. **`variant` exhaustiveness / fail-loud (ADR-0002, P9).** Every `std::visit` must handle both arms; a `valueless_by_exception` or unhandled alternative aborts loudly. Mirror the existing fail-loud pattern: the `sample_world` invariant (`r < nb` guarantees a hit; `std::abort()`, note 146) and `apply`'s TERMINATE abort (`env.cpp:144-146`).

8. **In-place mutation through the visited variant.** `apply` mutates `bw` in place (`env.hpp:93`, `env.cpp:133-165`). The seam's `filter_treasure`/`filter_detector` take a **mutable `Belief&`**; the `std::visit` dispatches to either `erase_if` (flat) or `&= mask` (bitset, **updating `count_` in the same call**). This is where the O(1)-nb obligation (risk 2) and the mutation seam intersect — the filter is the one place `count_` is written.

---

**Documentation obligation (ADR-0005):** this change makes `belief_features_and_decision_diagram_note.md`'s §A.4 sweep the *flat-arm* body (no longer the sole path) and realizes `/home/bork/belief_bitset_decision_reversal.md`. `docs/STATUS.md` and the current handoff likely describe the belief sweep as *the* hot path (the ~81% bucket); that orientation surface needs a **dated amendment** (amend by append, never silent rewrite) when the gate lands. The ZDD note stays valid as the future-variant hedge (note 179-185). Flag this to the implementer as part of the delivery, not after.

---

## DECISIONS STILL FOR THE HUMAN

These are genuine forks the implementer/maintainer should settle, not defaults the report can pick:

1. **The machine-constant cache budget value (`kTargetMaskCacheBudgetBytes`).** §4 establishes it must be homed honestly as a named target-cache fact, but the *number* is a maintainer call — the i5-6600 L2 is 256 KiB/core; the live mask set is ~122 KiB. How much headroom to demand (and whether to budget against L2 size, a fraction of it, or measured residency) is a judgment about the target machine, not derivable from the code. The report's recommendation is to set it from the i5-6600 L2 with stated headroom, but the exact figure is yours.

2. **Whether to compile the flat path in always, or behind a build flag.** The report assumes **both arms always compiled** (the variant carries both; the gate is a runtime value), which keeps the fallback genuine and the A/B harness possible in one binary. A build flag that drops the flat arm for a bitset-only build would shrink the binary but forfeits the fallback and the in-binary A/B check, and re-introduces the rep at a compile boundary. Recommendation: always compile both; flagged here as a real fork only if binary size on the target becomes a constraint.

3. **`world_at_rank` / `belief_key` first/last extraction method on the bitset.** `countr_zero(first non-zero word)` / `countl_zero(last non-zero word)` is the obvious O(kW64)-worst implementation; whether to also cache first/last rank on the `Belief` (like `count_`) is a micro-optimization the implementer can defer until a profile asks — it is not load-bearing for correctness. Flagged because it is the one spot where "cache more on the Belief" is a judgment call rather than an obligation (unlike `count_`, which risk 2 makes mandatory).

4. **The dispatch mechanism is NOT a live fork.** The report commits to `std::variant` + visit-in-env with substantiation (§3); template-on-rep and virtual-base are rejected for cited reasons. Listed here only to record that it was considered and is not left open.

---

**Files the implementer touches (all absolute):** `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/env.hpp` + `cpp/src/env.cpp` (Belief variant type, gate, masks, sampling, filters, marginals, informative, nb, world_at_rank, belief_key home); `cpp/include/chocofarm/policy.hpp` + `cpp/src/policy.cpp` (retype contracts, remove `RngWorldSource::sample_world`'s rep poke L1, `base_value` by-value param); `cpp/include/chocofarm/belief_key.hpp` (rep-aware fingerprint L2); `cpp/include/chocofarm/feature_compute.hpp` + `cpp/src/features.cpp` (sweep → masked-popcount on the bitset arm L3 with `* inv`, cache value type L5); `cpp/include/chocofarm/features.hpp` (`belief_cache_` type L5); `cpp/include/chocofarm/{gumbel,ismcts,nmcs}.hpp` + `cpp/src/{gumbel,ismcts,nmcs}.cpp` (retype signatures + local copies); `cpp/include/chocofarm/search_runtime.hpp` (`SearchTask::bw` type); `cpp/include/chocofarm/cyclic_gumbel.hpp` + `cpp/src/{gumbel_dump,ismcts_dump,nmcs_dump,mask_dump}.cpp` (fixture `sample_world` → `world_at_rank`, L4); `cpp/src/belief_sweep_oracle_check.cpp` (extend to the flat-vs-bitset A/B); `cpp/src/runner.cpp` (the `bw = env.worlds()` → `env.full_belief()` init and the L6 shrinkage/`nb` reads at lines 27,37,98). `cpp/src/serve.cpp` is untouched beyond inheriting the new types through `run_episodes`; **no wire/result change.**

---

## STEP-2 IMPLEMENTATION NOTE (2026-06-17 — amend-by-append, ADR-0005 Rule 8)

Step 2 (the bitset arm + the gate) is implemented and validated (the flat-vs-bitset A/B is byte-identical across every seam op + a full filter sequence; the end-to-end Python parity holds at 25 passed / 2 skipped UNCHANGED with the gate ON). Two records this point-in-time scoping report got wrong or under-specified, corrected here without silently rewriting the body above:

1. **§1B "Order-equivalence note" / §6 risk 1 — the world order is NOT numerically ascending.** The report repeatedly says "the flat vector starts ascending (`worlds_` built ascending)". This is **false**: `build_worlds` (env.cpp) emits `combinations(range(N), K)` order, which is NOT monotone in the bitmask value — e.g. world rank 15 is the high-bit combination `{0,1,2,3,19}` = 524303, and rank 16 is `{0,1,2,4,5}` = 55 (the first numeric inversion). The report's CONCLUSION still holds, for the corrected reason: the flat belief is a **rank-ordered (combinations-order) subsequence** of `worlds_` and the bitset indexes the **same `worlds_` by rank**, so `front()`/`back()` ARE the rank-0 / rank-(count-1) worlds, and `world_at_rank`/`belief_key`/`sample_world` are byte-identical across arms by **shared RANK order**, not a numeric one. The A/B harness (`belief_sweep_oracle_check.cpp`) builds the bitset by a value→rank **map**, NOT a binary search, precisely because `worlds_` is unsorted. **Caution for a future slice:** the decision-reversal note's `rank(uint32 -> index)` "binary search over `worlds`" (its line 109) would be WRONG on the unsorted `worlds_` — value→rank needs a map / linear scan, or `worlds_` must be re-sorted with the masks re-indexed. Production does not need value→rank in this slice (it only goes rank→value via `worlds_[idx]`), so this is a latent trap, not a live bug.

2. **`BitsetBelief::bits` is `std::vector<uint64_t>`, not the report's `std::array<uint64_t, kW64>` (§1/§3).** `kW64` is DERIVED from the env's world count at construction (the live instance gives 243), so it cannot be a compile-time template arg without hardcoding 243 into the TYPE — which fights ADR-0012 P1 / "derived dimensions never hardcoded." Resolved in favour of the runtime vector. This makes the variant SMALLER inline (24-byte vector vs the 1.9 KiB array), so §3's "1.9 KiB paid inline even in the flat fallback" concern only eases; correctness is identical (the A/B nets it).

Also delivered: the bitset kernels (`popcount_all` / `popcount_and` / `rth_set_bit_index`) are homed ONCE in `cpp/include/chocofarm/belief_bitset_ops.hpp` (P1, shared by `env.cpp` + `features.cpp` — no per-TU copy); the gate decision (kW64 / mask_bytes / budget / use_bitset) is printed by the oracle and the live instance is PINNED to the bitset arm by `tests/test_cpp_runner.py::test_cpp_belief_sweep_oracle` (it asserts the A/B PASS line, not SKIP).
