<!-- docs/design/cpp-belief-dynamic-rep-selection.md
     A RECORDED HYPOTHESIS (not a locked design): dynamically choosing the belief representation
     PER BELIEF, keyed on the current support size nb, rather than once-per-env. Records the cost model,
     the two candidate crossovers, the conceptual reframe (portfolio, not opt-in alternatives), the two
     real design constraints, and a PRE-REGISTERED decision rule to be settled by the head-to-head
     ZDD-vs-bitset profile + an nb-histogram. Produced 2026-06-17 on branch cpp-actor-online-reconfig,
     ahead of that measurement, so the profile lands against a stated hypothesis. Public Domain (The Unlicense). -->

# Dynamic per-belief representation selection (nb-keyed) — HYPOTHESIS + decision gate

**Status:** recorded hypothesis, **pending the deciding measurement** (the head-to-head ZDD-vs-bitset
profile + an nb-histogram over real search leaves). This is **not** an implementation contract. It is
written *before* the measurement deliberately — the gumbel 1a/1b mutation-control posture and the
zdd-onramp control arm (`cpp-belief-zdd-onramp.md` §7) both teach that a number is only trustworthy
against a hypothesis stated in advance. §6 pre-registers the decision rule; the probe/profile produces
the number; the human reads it.

**Builds on:** `cpp-belief-rep-scoping.md` (the gated two-rep seam — owns the static gate, the `Belief`
variant, the bit-exact A/B oracle, the rep-independent `belief_key`) and `cpp-belief-zdd-onramp.md` (the
ZDD engine + the `|Z|`-vs-`nb` probe). Driving note: `/home/bork/belief_features_and_decision_diagram_note.md`
(§A.4 the flat sweep, §B the diagram).

---

## 0. The question

Today the representation is chosen **once, at env construction, keyed on the *universe*** — the static
gate `use_bitset_ = enumerable && mask_bytes(N,nD,kW64) ≤ kTargetMaskCacheBudgetBytes && kW64 ≤
kBitsetMaxWords` (`cpp-belief-rep-scoping.md` §4). It picks one arm of `Belief =
std::variant<FlatBelief, BitsetBelief>` for the *whole* run, and ZDD is an all-or-nothing opt-in build
flag (`cpp-belief-zdd-onramp.md`) — there is **no** runtime ZDD heuristic at all.

The hypothesis: the *optimal* representation depends not on the universe but on the **current support
`nb`** (worlds still consistent with the observations so far), which varies by orders of magnitude across
a single search tree. A static per-env choice cannot be optimal everywhere; a **per-belief, runtime,
nb-keyed** choice can beat it.

This is a **new axis**, not a tweak to the existing gate: per-belief vs per-env, runtime vs
construction-time, nb (support) vs universe.

---

## 1. The cost model (why the axis exists)

Two different sizes are in play, and the static gate keys on the *other* one:

- **Universe** — fixed for an env instance (N=20 treasures, nD=44 detectors ⇒ |worlds|=C(20,5)=15504,
  `kW64=243` bitset words). Constant for the whole search. **This is what the static gate sees.**
- **Support `nb`** — the worlds consistent with the *current* observations. At the root `nb ≈ |worlds|`;
  every observation (a treasure opened, a detector read) only *adds* a constraint, so **`nb` is
  monotonically non-increasing along any root→leaf path**, and deep leaves can have `nb` in the single
  digits.

Per-op cost of the three reps (per `belief_features` evaluation — the unit, `feature_compute.hpp`):

| rep | belief_features cost | scaling |
|---|---|---|
| **flat** (§A.4 fused sweep over `nb` worlds) | ≈ `nb · (K + nD)` ≈ `nb · 49` | **O(nb)** |
| **bitset** (masked-AND + popcount over `kW64` words, per treasure/detector) | ≈ `(N + nD) · kW64` ≈ `64 · 243` ≈ 15.5K | **O(kW64), constant in nb** |
| **ZDD** (forward×backward sweep, non-constructing disjoint-count) | ≈ `(N + nD) · |Z|` | **O(\|Z\|)** |

The crux: bitset and ZDD pay a cost **independent of `nb`**, flat pays **O(nb)**. So:

- **Large `nb` (near the root):** flat's `nb·49` is huge; bitset's constant 15.5K wins (the ~64× the
  bitset cutover already bought). If the belief has structure, ZDD's `O(|Z|)` may win further.
- **Small `nb` (deep leaves):** flat does ≈ `nb·49` ops — at `nb=2`, ~100 ops. Bitset *still* does its
  constant ≈ 15.5K word-ops regardless. **Flat is ~2 orders of magnitude cheaper at the leaves, and the
  leaves are where `belief_features` (18.7% of the K=512 client profile) is spent.**

**Back-of-envelope crossover (to be pinned, not trusted):** flat ≈ bitset when `nb·49 ≈ (N+nD)·kW64`,
i.e. `nb* ≈ kW64 ≈ a few hundred` worlds. Both bodies are SIMD-vectorized under `-march=native`, so the
constants differ from this naive op-count — **the microbench (§5c) is the arbiter**, but the prediction
"`T_fb` is on the order of `kW64`" is the falsifiable hypothesis.

**The cache mutes — and possibly *concentrates* — the win.** `belief_features` is memoized on
`belief_key` (`FeatureBuilder::belief_cache_`), so the rep cost lands only on the **miss** (recompute)
path — the hit rate itself is **currently unmeasured** (§5a should capture it). But deep small-`nb`
beliefs accrue more observations ⇒ more *distinct* `belief_key`s ⇒ are **more likely to be misses**. So
the small-`nb` regime is plausibly *over-represented* among misses, and the dynamic win could exceed its
all-leaves share. The nb-histogram (§5a) must therefore be taken **over misses specifically**, not all
leaves.

---

## 2. The two candidate crossovers, ranked by how well-founded each is

### 2a. flat ↔ bitset — the WELL-FOUNDED one (ZDD-independent, no build flag)

"**Bitset near the root (large nb), flat at the leaves (small nb).**" This is the grounded win:

- **Both arms already exist, already proven byte-identical** across every seam op + a full filter
  sequence (`cpp-belief-rep-scoping.md` §5 Steps 2–3; the A/B oracle `belief_sweep_oracle_check.cpp`).
  No new representation, no build flag — both are always compiled today.
- **Conversion is cheap exactly when it fires.** bitset→flat = enumerate the set bits into a world list,
  O(nb) — and you only convert when `nb` has dropped below `T_fb` (a few hundred), so the enumeration is
  small by construction.
- **Monotone `nb` ⇒ a one-way ratchet ⇒ no thrash.** Down any path `nb` only shrinks, so it crosses
  `T_fb` *at most once* — at most one conversion per root→leaf path, and never back. (See §4 for why this
  is automatic under the "arm = pure function of nb" rule.)

The likely headline win is **flat-at-the-leaves**: turning off the bitset's wasted constant-cost popcount
exactly where `belief_features` spends its 18.7%.

### 2b. flat/bitset ↔ ZDD — the SPECULATIVE one (needs the head-to-head data)

On *this* instance the bitset already beat ZDD head-to-head (SIMD popcount > pointer-chasing) at the
sizes measured so far. ZDD-dynamic is worth *deciding* only after the head-to-head profile, and it has a
harder selection problem:

- **The ZDD crossover is not purely `nb`.** ZDD cost is `O(|Z|)`, and `|Z|` tracks *structure*, not `nb`
  (`cpp-belief-zdd-onramp.md` §7 carries a random-subset control precisely because `|Z|/nb` is a
  structure measure). So `nb` is only a **proxy** for "is ZDD cheaper here."
- **Chicken-and-egg.** You cannot know `|Z|` without building the ZDD — the very cost you'd skip for
  small beliefs. So the selector can only use *cheap* signals (`nb`, observation depth) to predict the
  ZDD win. §5/§6 measure whether `nb` predicts it well enough to use.

---

## 3. The conceptual reframe — a real dynamic win SUPERSEDES the build-flag framing

If 2a or 2b genuinely wins, the framing changes shape, and design-cost "the dynamic path fights the
opt-in build flag" **dissolves rather than being paid**:

Today flat/bitset/ZDD are conceived as **alternatives you pick between** (statically, or at build time
for ZDD). A runtime per-belief selector needs *all candidates compiled in at once* — which reads as a
cost against the "ZDD = opt-in build flag" decision (`cpp-belief-zdd-onramp.md`). But that decision only
exists *because* ZDD is a speculative alternative. **A demonstrated dynamic win reconceives the three
reps as a standing *portfolio* the runtime draws from per belief** — at which point ZddBelief being a
permanent variant arm is not a cost, it is the point. The opt-in framing is superseded, not violated.

This reframe is **contingent on the win being real**. Until the §5 measurement says so, ZDD stays the
opt-in build-flag arm exactly as `cpp-belief-zdd-onramp.md` specifies; the flat↔bitset ratchet (2a) needs
no flag regardless (both arms are always compiled).

---

## 4. The two design constraints the reframe does NOT dissolve

### 4a. Representation must be a pure function of *identity*, never of path history (ADR-0012 P1)

The collision guard in `belief_feats_` verifies a `belief_key` match with full equality `entry.first ==
bw` (`features.cpp`), and **`std::variant::operator==` is `false` across different arms**. The naive
fear is that the same world-set stored as different arms on different paths compares unequal and breaks
the cache.

It does not — **provided the arm is a pure function of the belief.** Identity already has a
representation-independent home: the `belief_key` triple `(count, world_at_rank(0),
world_at_rank(nb-1))` is bit-identical across arms (`cpp-belief-rep-scoping.md` §6 risk 5). So if we set
`arm = f(nb)` (or any pure function of the world-set), then:

1. The same belief always gets the same arm ⇒ a cache hit is a same-arm `==` ⇒ **correct and efficient**.
2. A genuine `belief_key` collision (different world-sets, same triple) compares `false` (same or
   different arm) ⇒ collision correctly detected ⇒ recompute. **Correct.**
3. Cross-path consistency is automatic — two paths reaching the same information set have the same `nb`,
   hence the same arm.

So 4a is **not a hazard to mitigate; it is a design rule to honor: the selector reads only the belief
(its `nb`), never the path that produced it.** Representation becomes a function *of* identity — the
P1-clean shape, stronger than "separate identity from representation." (The full-equality verify stays
the ADR-0011 net regardless.)

### 4b. The threshold is a measured crossover, not a guessed constant (ADR-0009)

`T_fb` (flat↔bitset) is predicted ≈ `O(kW64)` (§1) but must be pinned by the microbench (§5c) because
the SIMD constants are not the naive op-counts. `T_bz` (bitset↔ZDD), *if* 2b is in play, is harder:
`nb` is a proxy for `|Z|`, so §5b must first establish that `nb` predicts the ZDD winner at all. A
threshold that doesn't survive measurement is exactly the scattered-magic-number the static gate's
honest homing (`cpp-belief-rep-scoping.md` §4) exists to prevent.

---

## 5. The deciding measurement

Three measurements, all on the live instance at the K=512 sweetspot, all reusing existing harnesses:

- **(a) nb-histogram over `belief_features` MISSES.** Instrument the wire-pool / runner leaf path to log
  `nb` at each cache *miss* (the recompute path) during a real search — **and the hit/miss split itself,
  which is currently unmeasured** — then histogram the miss `nb`. **Decides whether flat-at-the-leaves
  pays** — if a large fraction of misses sit below `T_fb`, it does. (Take it over misses, not all
  leaves — §1.)
- **(b) head-to-head ZDD-vs-bitset profile + the `|Z|`-vs-`nb` table.** The profile is already planned
  (post-ZDD-land); the zdd-onramp probe (`cpp-belief-zdd-onramp.md` §7) already emits median `|Z|/nb` by
  depth with a random-subset control. **Decides whether any `nb` regime exists where ZDD beats bitset on
  real beliefs**, and whether `nb` (or depth) predicts it.
- **(c) per-rep cost vs `nb` microbench.** Extend `belief_sweep_bench.cpp` (which already times
  `belief_features`) to sweep `nb` across flat / bitset / (optionally) ZDD and find the crossovers
  `T_fb` and `T_bz`. **Pins the thresholds 4b leaves open.**

---

## 6. PRE-REGISTERED decision rule (stated before the numbers)

- **If** (a) shows ≥ a material fraction of `belief_features` misses at `nb < T_fb` (from (c)) **→ build
  the flat↔bitset ratchet (2a).** It pays, it is ZDD-independent, and it is bit-exact by the existing
  A/B oracle extended to a mixed-arm population. This is buildable on its own merits regardless of ZDD.
- **If** (b) shows ZDD beats bitset in some `nb`/depth regime **and** that regime is visited often enough
  to matter (per (a)'s histogram) **→ extend to the 3-way portfolio**, and the §3 reframe fires (ZddBelief
  becomes a permanent arm; the build-flag framing is superseded).
- **If neither** → **keep the static gate** and record the honest "no dynamic win" finding — the shelve
  outcome, exactly as the zdd-onramp control arm's "if they coincide, that is the honest shelve"
  (`cpp-belief-zdd-onramp.md` §7). A known-negative beats a guessed-positive.

The two crossovers are independent: 2a can be adopted while 2b is shelved.

---

## 7. Scope, sequencing, ADR hygiene

- **Out of the current ZDD-finalize scope.** That work lands the opt-in ZDD arm + its parity; this is the
  *next* design question, decided after the head-to-head profile. Do not fold it in.
- **2a is the cheap, well-founded first build** (no flag, both arms exist, bit-exact net ready). **2b
  waits for the data.**
- **ADR-0012:** P1 — representation as a pure function of identity (§4a); P3 — the selector is one-owner
  (the env filter ops decide the arm, no call site chooses, exactly as the static gate today); P6 — the
  flat↔bitset A/B oracle extends to a mixed-arm population (bit-exact across arms is already proven).
  **ADR-0009** — §5/§6 are the measure-first spine. **ADR-0002** — variant exhaustiveness / fail-loud is
  already pinned (`cpp-belief-rep-scoping.md` §6 risk 7); a new arm inherits it.
- **ADR-0005:** this note is a point-in-time hypothesis. When the §5 measurement fires, record the
  verdict by **dated amendment** here (append, never rewrite), and repoint the cross-references in
  `cpp-belief-rep-scoping.md` (the gate-owner) and `cpp-belief-zdd-onramp.md` (the ZDD-owner) if the
  portfolio reframe is adopted.

---

**Files a future implementation would touch (not now — recorded for scope):** the env filter ops
(`env.cpp` / `env.hpp` — the one place the arm is chosen, mirroring the static gate); the conversion
helper (bitset→flat enumerate, flat→bitset set-bits); the A/B oracle (`belief_sweep_oracle_check.cpp` —
extend to a mixed-arm population); the microbench (`belief_sweep_bench.cpp` — sweep `nb`); leaf-path
instrumentation for the nb-histogram. No wire/result change (the belief never crosses the boundary —
`cpp-belief-rep-scoping.md` §0).

---

## MEASUREMENT (b) FIRED + the per-nb dispatch mechanism (2026-06-17 — amend-by-append, ADR-0005 Rule 8)

The §5b head-to-head ran (profiles `~/w/vdc/chocobo/profiles/h2h*`). **Result:** after fixing the ZDD
arm's two value-semantics footguns (the per-descent-copy OOM, commit `5391c59`; `operator==`'s O(nb)
enumeration, `b826baa`), ZDD is sound + bit-exact but still costs **~10× the client CPU per leaf** of
bitset, **~70% of it allocation churn** from the copyable-diagram machinery (the diagram math is cheap,
~10%). The residual is *structural*, deferred to **`BACKLOG.md` → "ZDD belief arm — close the per-leaf
value-semantics gap."** So for input (b): **on this instance ZDD wins in no `nb` regime** — bitset's
masked-popcount is cheaper everywhere the universe is enumerable; ZDD is the large-N hedge, out of scope
here. The §6 decision rule resolves to its third branch for ZDD (honest shelve) and its first for
flat↔bitset (the well-founded 2a crossover stands).

**Consequence for the dispatch design:** the per-nb table **collapses to a flat↔bitset two-way table** on
this instance — the ZDD column is empty until the BACKLOG refactor closes its per-leaf gap.

**The mechanism (the §5c/§6 realization — the maintainer's proposal, recorded as the concrete design):**
build the dispatch as a **per-nb table computed once at startup** (a calibration step): micro-benchmark
`belief_features` per rep across the `nb` range (extend `belief_sweep_bench.cpp`) and set `arm[nb] =
argmin_rep cost`. Properties:
- It *is* the measure-first crossover — no hand-tuned `T`; it captures the real SIMD/cache constants on
  the actual hardware (ADR-0009 baked in).
- `arm = table[nb]` is a **pure function of `nb`** → satisfies §4a (the cache collision guard stays
  correct), and the monotone-`nb` ratchet bounds conversions to ≤1 per root→leaf path. The filter outputs
  `arm[nb]` directly (the stored rep *is* the compute rep; conversions only at table-boundary crossings).
- **Rigor caveat (the §4b issue, concretely):** the table is *faithful* for flat (O(nb)) and bitset
  (constant in `nb`) — both `nb`-determined — but only a **proxy for ZDD**, whose cost is `|Z|`-determined,
  not `nb` (the same `nb` has wildly different `|Z|`). Admit ZDD into the table only in `nb`-regimes where
  it *reliably* wins (low `|Z|`-variance), which on this instance is none.
- **Persist** the table keyed by (instance dims, CPU) — a stable function of hardware+dims, so a one-time
  calibration, not per-run startup cost.
