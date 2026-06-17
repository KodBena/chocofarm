# Belief-rep scoping — workflow phases (2026-06-17)

The 3 region maps + the draft synthesis + the adversarial critique behind `docs/design/cpp-belief-rep-scoping.md`. Preserved per the expensive-output convention. Public Domain.

## Region maps

```json
[
  {
    "region": "env+seam (belief mechanics + WorldSource seam)",
    "touchpoints": [
      {
        "location": "cpp/include/chocofarm/env.hpp:57",
        "how_used": "worlds() returns const vector<uint32_t>& — source of initial belief (full C(N,K) enumeration)",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/env.hpp:81",
        "how_used": "marginals(const vector<uint32_t>& bw) reads bw, iterates each w to count set bits",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/env.hpp:88",
        "how_used": "legal_actions(const vector<uint32_t>& bw, ...) calls marginals(bw) and informative(j, bw)",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/env.hpp:97",
        "how_used": "filter_treasure(vector<uint32_t>& bw, int i, bool present) mutates bw in-place via filter_inplace",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/env.hpp:98",
        "how_used": "filter_detector(vector<uint32_t>& bw, int i, bool positive) mutates bw in-place via filter_inplace",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/env.hpp:112",
        "how_used": "informative(int face_id, const vector<uint32_t>& bw) iterates bw to check both polarities",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/env.hpp:128",
        "how_used": "filter_inplace(vector<uint32_t>& bw, uint32_t mask, bool want) free function — the core belief filter",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/env.cpp:120-123",
        "how_used": "filter_inplace uses std::erase_if with predicate ((w & mask) != 0) != want",
        "through_seam": false
      },
      {
        "location": "cpp/include/chocofarm/env.cpp:133-165",
        "how_used": "apply(Loc&, vector<uint32_t>& bw, ...) mutates bw in-place by calling filter_treasure / filter_detector",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/policy.hpp:61",
        "how_used": "Policy::decide abstract signature has const vector<uint32_t>& bw parameter",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/policy.hpp:82-90",
        "how_used": "RandomPolicy::decide passes bw to env.legal_actions(bw, collected)",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/policy.hpp:143",
        "how_used": "WorldSource::sample_world(const vector<uint32_t>& bw) abstract contract",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/policy.hpp:152-154",
        "how_used": "RngWorldSource::sample_world direct index access: bw[pick(rng_)] — raw vector access",
        "through_seam": false
      },
      {
        "location": "cpp/src/policy.cpp:23-33",
        "how_used": "Policy::decide_target(bw) calls decide(bw) and env.legal_actions(bw, collected)",
        "through_seam": true
      },
      {
        "location": "cpp/src/policy.cpp:36-51",
        "how_used": "GreedyBase::decide(bw) calls env.marginals(bw)",
        "through_seam": true
      },
      {
        "location": "cpp/src/policy.cpp:57-76",
        "how_used": "GreedyStopBase::decide(bw) calls env.marginals(bw) and env.exit_cost",
        "through_seam": true
      },
      {
        "location": "cpp/src/policy.cpp:79-111",
        "how_used": "candidate_actions(bw) calls env.marginals(bw), env.informative(j, bw)",
        "through_seam": true
      },
      {
        "location": "cpp/src/policy.cpp:114-128",
        "how_used": "base_value(..., vector<uint32_t> bw copies by value, calls env.apply(loc, bw, collected, a, world)",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/nmcs.hpp:67-68",
        "how_used": "NMCSWorldSource::playout_value abstract has const vector<uint32_t>& bw",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/nmcs.hpp:80-81",
        "how_used": "NMCSPolicy::decide signature has const vector<uint32_t>& bw",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/nmcs.hpp:85-87",
        "how_used": "NMCSPolicy::search signature has const vector<uint32_t>& bw parameter",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/ismcts.hpp:76-77",
        "how_used": "ISMCTSSource::leaf_value has const vector<uint32_t>& bw",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/ismcts.hpp:104-105",
        "how_used": "ISMCTSPolicy::decide has const vector<uint32_t>& bw",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/gumbel.hpp:90",
        "how_used": "gumbel_belief_key(const vector<uint32_t>& bw) calls belief_key(bw) — fingerprints by (size, front, back)",
        "through_seam": false
      },
      {
        "location": "cpp/include/chocofarm/gumbel.hpp:144-146",
        "how_used": "GumbelAZPolicy::decide has const vector<uint32_t>& bw",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/gumbel.hpp:158-160",
        "how_used": "GumbelAZPolicy::run_search has const vector<uint32_t>& bw",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/gumbel.hpp:188-189",
        "how_used": "GumbelAZPolicy::evaluate(node, loc, vector<uint32_t>& bw, collected) calls env methods through features",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/features.hpp:50-51",
        "how_used": "legal_mask(env, vector<uint32_t>& bw, collected) oracle call",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/features.hpp:107-108",
        "how_used": "FeatureBuilder::build(loc, const vector<uint32_t>& bw, collected) reads bw via belief_feats_",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/features.hpp:169",
        "how_used": "belief_feats_(const vector<uint32_t>& bw) memoized by belief VALUE in belief_cache_",
        "through_seam": false
      },
      {
        "location": "cpp/include/chocofarm/features.hpp:164",
        "how_used": "belief_cache_ stores vector<pair<vector<uint32_t> copy, BeliefFeatures>> keyed by BeliefKey fingerprint",
        "through_seam": false
      },
      {
        "location": "cpp/include/chocofarm/belief_key.hpp:20",
        "how_used": "BeliefKey tuple (int, uint32_t, uint32_t) = (size(), front(), back())",
        "through_seam": false
      },
      {
        "location": "cpp/include/chocofarm/belief_key.hpp:22-25",
        "how_used": "belief_key(const vector<uint32_t>& bw) fingerprints belief by (size, front, back)",
        "through_seam": false
      },
      {
        "location": "cpp/include/chocofarm/search_runtime.hpp:53",
        "how_used": "SearchTask::bw is std::vector<uint32_t> member field — travels to runtime via span",
        "through_seam": false
      },
      {
        "location": "cpp/include/chocofarm/cyclic_gumbel.hpp:27",
        "how_used": "CyclicGumbelSource::sample_world(bw) returns bw.empty() ? 0u : bw[0] — direct index",
        "through_seam": false
      },
      {
        "location": "cpp/src/gumbel_dump.cpp:gumbel() (line varies)",
        "how_used": "GumbelSource::sample_world(bw) scripted to return bw[idx % |bw|] for fixtures",
        "through_seam": false
      },
      {
        "location": "cpp/src/nmcs_dump.cpp:sample_world(bw) (line varies)",
        "how_used": "NMCSWorldSource::sample_world scripted to return bw[0]",
        "through_seam": false
      },
      {
        "location": "cpp/src/features.cpp:190-193",
        "how_used": "belief_features_nonempty iterates over bw: for (uint32_t w : bw)",
        "through_seam": false
      }
    ],
    "seam_assessment": "The env belief API is CLEANLY SEAMED. The public boundary (worlds, marginals, legal_actions, informative, filter_treasure, filter_detector, apply) operates on LOGICAL ABSTRACTIONS: belief size, set membership tests, filtering via predicates. The ONE raw-vector leak is RngWorldSource::sample_world(bw)[pick(rng_)] — a direct index access that MUST route through a seam method.\n\nFor BITSET INTEGRATION, the seam design is:\n1. CREATE an abstract Belief type behind env.hpp (opaque to caller). The current std::vector<uint32_t> is ONE representation; the bitset is ANOTHER.\n2. ENCAPSULATE all representation-specific ops (sampling, iterating, size, filtering) in env methods or a dedicated belief-operations namespace. The existing filter_inplace, marginals, informative already route through the env; only sample_world escapes.\n3. MOVE sample_world FROM WorldSource (a policy mixin) INTO the env. It becomes env.sample_world(const Belief&, rng) with TWO implementations (vec_sample_world, bitset_sample_world), selected at env construction by a homed size/budget threshold (ADR-0012 P1).\n4. The belief-cache key (BeliefKey fingerprint) SURVIVES unchanged — (size, front, back) is representation-agnostic (only needs a contiguous bit order, guaranteed by env.worlds() enumeration).\n\nThe BELIEF-SWEEP COST: belief_features calls belief_feats_ which iterates O(nb·(N+nD)). With the bitset, this COLLAPSES to masked-AND + popcount over 243 uint64 (CONSTANT in nb). The cache itself (keyed by fingerprint, storing copies) can SWAP its value type from vector to bitset — a ONE-LINE change to the pair<vector<uint32_t>, ...> pair&lt;Belief, ...&gt;.\n\nRISK ZONE: Any code that INDEXES bw directly (bw[i]) or COPIES bw as a member field (SearchTask::bw) POKES THE RAW VECTOR. These MUST route through env methods (e.g., env.sample_world) or accept an env-provided belief accessor. The current RngWorldSource is the ONLY sampling site; gumbel_dump/nmcs_dump override it, so they must adapt to env.sample_world's new form.",
    "notes": "\n## Summary of Belief-Type Surface\n\n### ENV AS THE PRIMARY SEAM (env.hpp + env.cpp)\nThe Environment class owns ALL belief mechanics. Every operation on std::vector<uint32_t> bw passes through env methods (marginals, legal_actions, informative, apply). This is ADR-0012 P2 in action — the env is a closed box; a Policy never manipulates bw directly.\n\nKey methods that touch belief:\n- **worlds()**: returns full enumeration (source of initial belief)\n- **marginals(bw)**: READS bw, iterates to count bits per treasure\n- **legal_actions(bw, collected)**: READS bw via marginals & informative\n- **informative(face, bw)**: READS bw to check detector split\n- **filter_treasure/filter_detector(bw, ...)**: MUTATES bw in-place via filter_inplace (std::erase_if)\n- **apply(loc, bw, collected, action, world)**: MUTATES bw via filter_* during observe/collect\n- **filter_inplace(bw, mask, want)**: free function, the CORE compaction (uses std::erase_if)\n\n### POLICY SEAM BOUNDARY (policy.hpp)\nThe abstract Policy contract takes `const vector<uint32_t>& bw` as a decision input. Implementations (RandomPolicy, GreedyBase, GreedyStopBase, NMCSPolicy, ISMCTSPolicy, GumbelAZPolicy) NEVER INDEX bw directly — they call env.legal_actions(bw, collected), env.marginals(bw), etc.\n\nONE EXCEPTION: RngWorldSource::sample_world(const vector<uint32_t>& bw) returns bw[pick(rng_)]. This is a DIRECT INDEX into the raw vector, violating the seam — the ONLY sampling site that does so.\n\n### SEARCH POLICIES (nmcs, ismcts, gumbel)\nAll searches take const vector<uint32_t>& bw as input and pass it THROUGH the env seam. The only search-specific belief use is node caching via BeliefKey fingerprint (belief_key(bw) = (size, front, back)) — used by GumbelAZPolicy and ISMCTSPolicy info-set transposition tables, and by FeatureBuilder's belief cache.\n\n### FEATURE CACHING (features.hpp)\nFeatureBuilder::build(loc, bw, collected) memoizes belief-derived intermediates (marginals, informative status, coverage counts) by belief VALUE in belief_cache_. The cache:\n- KEYS: BeliefKey fingerprint (size, front, back)\n- VALUES: vector<pair<vector<uint32_t> copy, BeliefFeatures>>\n- COLLISIONS: verified by FULL bw equality (fingerprint is collision-resistant, not -free)\n\nThe belief_sweep that populates BeliefFeatures iterates over bw for O(nb·(N+nD)) ops. This is where the bitset buys the most — constant-cost masked-AND + popcount.\n\n### BELIEF-KEY FINGERPRINTING (belief_key.hpp)\nAll three caches (FeatureBuilder, Gumbel node, ISMCTS node) use THE SAME BeliefKey: (size, front, back). This is ADR-0012 P1 (one home, derive-don't-duplicate). The fingerprint is REPRESENTATION-AGNOSTIC — it only needs a contiguous ordering of worlds (guaranteed by env.worlds() enumeration order).\n\n### EXTERNAL BOUNDARIES\n- **SearchTask**: carries vector<uint32_t> bw as a member (travels through search_runtime.hpp)\n- **Runner**: sees bw only in SearchTask; does NOT serialize/wire beliefs\n- **Fixtures (gumbel_dump, nmcs_dump)**: override sample_world to index bw[0] or scripted indices\n\n## THE BITSET SEAM DESIGN (implementation path)\n\n**Step 1: Belief Type**\nDefine an opaque Belief type in env.hpp:\n```cpp\nstruct Belief { std::array<uint64_t, kW64> bits; };  // or variant<vector, bitset>\n```\nHomed in Environment (the ONE owner of world space + masks).\n\n**Step 2: Gate at Env Construction**\nCompute (N+nD)*2^N / cache_bytes budget; if bitset fits L1, USE_BITSET = true. Store this as an env-static derived datum (P1).\n\n**Step 3: Env Methods Return/Accept Belief**\n- worlds() returns vector<uint32_t> (external, enumeration source)\n- Initial belief: env.full_belief() -> Belief (ALL worlds as bitset or vector)\n- Filter methods: filter_treasure(Belief& b, ...) / filter_detector(Belief& b, ...)\n- Readers: marginals(const Belief& b) / informative(int, const Belief& b)\n- sample_world(const Belief& b, rng) -> uint32_t (MOVED from WorldSource into env)\n\n**Step 4: SearchTask & Policy Contracts**\n- SearchTask::bw → SearchTask::belief (Belief type, opaque)\n- Policy::decide(env, loc, const Belief& bw, ...) (one signature, works for both reps)\n- WorldSource REMOVED from policy.hpp (sample_world is now env.sample_world)\n\n**Step 5: Feature Cache Value Type**\nbelief_cache_: BeliefKey → vector<pair<Belief, BeliefFeatures>>\nA SINGLE value type, selected at build time (both reps fit the same cache interface).\n\n**Step 6: Fixtures**\ngumbel_dump, nmcs_dump, etc. ADAPT sample_world calls to env.sample_world(...).\n\n## Why This Design Satisfies ADR-0012\n\n- **P1 (one home)**: Belief type, masks, world enumeration ALL homed in env. Derived data (bitset masks, enumeration) built once in ctor.\n- **P2 (env<->Policy seam)**: Policy takes const Belief&, never names representation. env.sample_world is an env method, not a Policy mixin.\n- **P3 (one-owner)**: Belief is value-typed (not a pointer), copied by searches as needed, no dangling refs.\n- **P6 (equivalence)**: Vector and bitset filters produce BYTE-IDENTICAL kept sets (same mask predicates). Marginals/counts from bitset are INTEGER-EXACT (popcount), bit-identical to vector sweep.\n- **P7 (behavioral parity)**: Wire/runner contracts see Belief as opaque. No representation leaks to the outside.\n\n## Touchpoint Count Summary\n\n- **Env seam (clean)**: 8 methods (worlds, marginals, legal_actions, informative, filter_*, apply)\n- **Free function (clean)**: 1 (filter_inplace)\n- **WorldSource seam (LEAK)**: 1 (RngWorldSource::sample_world direct index) — MUST MOVE to env.sample_world\n- **Policy signatures (clean)**: ~15 (all take const vector/Belief&, pass to env)\n- **Search signatures (clean)**: ~12 (all thread const vector/Belief& through seam)\n- **Feature cache (hidden)**: 1 belief_cache_ entry point (belief_feats_), value type swappable\n- **Belief-key (representation-agnostic)**: 1 fingerprint (used by 3 caches)\n- **Fixtures (override points)**: 4 (gumbel_dump, nmcs_dump, cyclic_gumbel, ismcts_dump) — each has sample_world override\n- **Raw-vector access (LEAKS)**: 4 sites (RngWorldSource[pick], CyclicGumbelSource[0], fixture overrides x2)\n\nALL LEAKS are sampling-related and can be unified by moving sample_world into the env. No other code directly indexes or exposes the representation.\n"
  },
  {
    "region": "searches+cache (gumbel/ismcts/nmcs + belief cache)",
    "touchpoints": [
      {
        "location": "cpp/include/chocofarm/policy.hpp:152-154",
        "how_used": "RngWorldSource.sample_world pokes bw.size() and bw[pick(rng)]",
        "through_seam": false
      },
      {
        "location": "cpp/include/chocofarm/belief_key.hpp:24",
        "how_used": "belief_key pokes bw.front() and bw.back()",
        "through_seam": false
      },
      {
        "location": "cpp/src/features.cpp:190-192",
        "how_used": "belief_features_nonempty pokes for(uint32_t w : bw)",
        "through_seam": false
      },
      {
        "location": "cpp/src/gumbel.cpp:352",
        "how_used": "descend copies nbw = bw",
        "through_seam": false
      },
      {
        "location": "cpp/src/gumbel.cpp:356",
        "how_used": "descend calls gumbel_belief_key(nbw)",
        "through_seam": true
      },
      {
        "location": "cpp/src/gumbel.cpp:367",
        "how_used": "descend passes nbw const-ref recursively",
        "through_seam": true
      },
      {
        "location": "cpp/src/ismcts.cpp:139",
        "how_used": "iterate copies nbw = bw and calls belief_key(nbw)",
        "through_seam": false
      },
      {
        "location": "cpp/src/nmcs.cpp:73",
        "how_used": "search copies cur_bw = bw",
        "through_seam": false
      },
      {
        "location": "cpp/src/nmcs.cpp:128",
        "how_used": "search calls env.apply(cur_loc, cur_bw, ...)",
        "through_seam": true
      },
      {
        "location": "cpp/src/features.cpp:332-343",
        "how_used": "belief_cache_ uses belief_key lookup, equality verify, full copy store",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/gumbel.hpp:113",
        "how_used": "GumbelNode.children keyed by (int, GBeliefKey)",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/ismcts.hpp:90",
        "how_used": "ISMCTSNode.children keyed by (int, BeliefKey)",
        "through_seam": true
      }
    ],
    "seam_assessment": "Clean abstraction seam already in place. Three direct-poke sites (RngWorldSource::sample_world indexing bw[pick], belief_key calling bw.front()/back(), belief_features_nonempty iterating for(w:bw)) are isolated and safe to abstract. All 15+ other touches route through env methods (apply, filter_treasure/detector, legal_actions, informative), WorldSource interface, or fingerprinting. Beliefs are value-typed, copied at steps, never indexed in tree recursion, never held at nodes (only (slot, BeliefKey) children). Cache uses value-semantics: stores full vector copies (line 341) and equality-verifies by iteration (line 335)—both adapt if rep changes. To host bitset: define Belief abstract, FlatBelief/BitsetBelief implementations, gate once at Environment ctor (P1), wrap 3 pokers. No per-search refactor needed. Filters set-identical, features byte-identical, sampling deterministic+reordered (behavioral re-baseline).",
    "notes": "Belief representation scoping report for dense bitset integration. Three direct-poke sites poke raw vector<uint32_t>: (1) RngWorldSource.sample_world (policy.hpp:152-154) indexes bw.size()/bw[pick]—wrap with Belief::sample_world(rng). (2) belief_key (belief_key.hpp:24) calls bw.front()/bw.back()—optionally wrap as Belief::fingerprint(). (3) belief_features_nonempty (features.cpp:190-192) iterates for(w:bw)—use Belief iteration protocol. All 12+ other sites abstracted: env methods, WorldSource, fingerprint call, FeatureBuilder.build, belief_cache memo. Beliefs copied at steps (nmcs:73, ismcts:139, gumbel:352), never at nodes (only (slot,BeliefKey) children). Cache stores full vector copies (features.cpp:341) and equality-verifies (line 335) using value-semantics—both transparent to rep change. Plan: Belief abstract interface, FlatBelief<vector>/BitsetBelief<array<uint64_t,243>> implementations, gate at Environment ctor on |worlds|*(N+nD)*sizeof(mask) footprint vs cache/budget threshold (P1), wrap 3 pokers with Belief method calls. All tests transparent via gate. Filters set-identical, features byte-identical (exact counts + * inv), sampling deterministic but reordered (behavioral re-baseline, N≥300 verification)."
  },
  {
    "region": "features+wire+parity",
    "touchpoints": [
      {
        "location": "cpp/include/chocofarm/belief_key.hpp:22",
        "how_used": "belief_key(bw) fingerprint for cache lookup",
        "through_seam": true
      },
      {
        "location": "cpp/src/features.cpp:331-344",
        "how_used": "belief_feats_() memo: key lookup, full equality, stores bw copy",
        "through_seam": true
      },
      {
        "location": "cpp/src/features.cpp:215-219",
        "how_used": "belief_features() entry: dispatch on bw.empty()",
        "through_seam": true
      },
      {
        "location": "cpp/src/features.cpp:178-210",
        "how_used": "belief_features_nonempty(): fused sweep over bw",
        "through_seam": true
      },
      {
        "location": "cpp/src/features.cpp:258-300",
        "how_used": "FeatureBuilder::build() calls belief_feats_()",
        "through_seam": true
      },
      {
        "location": "cpp/src/features.cpp:60-68",
        "how_used": "legal_mask() oracle: env.legal_actions(bw, collected)",
        "through_seam": true
      },
      {
        "location": "cpp/src/features.cpp:302-320",
        "how_used": "legal_mask_from_features() slices blocks",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/features.hpp:107",
        "how_used": "build() signature: const bw reference",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/features.hpp:164",
        "how_used": "belief_cache_ member stores owned copies",
        "through_seam": true
      },
      {
        "location": "cpp/src/runner.cpp:24",
        "how_used": "run_episode: bw = env.worlds()",
        "through_seam": true
      },
      {
        "location": "cpp/src/runner.cpp:49-50",
        "how_used": "run_episode: fb.build(), legal_mask() per-decision",
        "through_seam": true
      },
      {
        "location": "cpp/src/runner.cpp:68",
        "how_used": "run_episode: env.apply() mutates bw in-place",
        "through_seam": true
      },
      {
        "location": "cpp/src/runner.cpp:177",
        "how_used": "write_results(): belief never serialized",
        "through_seam": true
      },
      {
        "location": "cpp/src/serve.cpp:187",
        "how_used": "serve path runs run_episodes",
        "through_seam": true
      },
      {
        "location": "cpp/src/gumbel.cpp:230-235",
        "how_used": "evaluate(): fb.build() + reuse",
        "through_seam": true
      },
      {
        "location": "cpp/src/gumbel.cpp:352-367",
        "how_used": "descend(): nbw copy, belief_key()",
        "through_seam": true
      },
      {
        "location": "cpp/src/gumbel.cpp:389",
        "how_used": "simulate_root_action(): nbw copy, belief_key()",
        "through_seam": true
      },
      {
        "location": "cpp/src/gumbel.cpp:530",
        "how_used": "run_search(): reset_belief_cache()",
        "through_seam": true
      },
      {
        "location": "cpp/src/ismcts.cpp:134-144",
        "how_used": "iterate() expansion: belief_key()",
        "through_seam": true
      },
      {
        "location": "cpp/src/ismcts.cpp:159-175",
        "how_used": "iterate() continuation: belief_key()",
        "through_seam": true
      },
      {
        "location": "cpp/src/gumbel_dump.cpp:201",
        "how_used": "bw = env.worlds() init",
        "through_seam": true
      },
      {
        "location": "cpp/src/gumbel_dump.cpp:149-154",
        "how_used": "ScriptedGumbelSource::sample_world(bw)",
        "through_seam": true
      },
      {
        "location": "cpp/src/ismcts_dump.cpp:142",
        "how_used": "bw = env.worlds() init",
        "through_seam": true
      },
      {
        "location": "cpp/src/nmcs_dump.cpp:107",
        "how_used": "bw = env.worlds() init",
        "through_seam": true
      },
      {
        "location": "cpp/src/belief_sweep_oracle_check.cpp:55-75",
        "how_used": "reference() independent iteration",
        "through_seam": true
      },
      {
        "location": "cpp/src/belief_filter_bench.cpp:85-95",
        "how_used": "branchless_ref() candidate filter",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/policy.hpp:143",
        "how_used": "WorldSource::sample_world(bw) contract",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/policy.hpp:152-154",
        "how_used": "RngWorldSource::sample_world(bw)",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/env.hpp:81",
        "how_used": "marginals(bw) iterate-accumulate",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/env.hpp:88-89",
        "how_used": "legal_actions(bw, collected)",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/env.hpp:93-94",
        "how_used": "apply(): bw reference",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/env.hpp:97-98",
        "how_used": "filter_treasure/detector() delegate",
        "through_seam": true
      },
      {
        "location": "cpp/include/chocofarm/env.hpp:112",
        "how_used": "informative(face, bw)",
        "through_seam": true
      },
      {
        "location": "cpp/src/env.cpp:120-123",
        "how_used": "filter_inplace(): std::erase_if()",
        "through_seam": true
      }
    ],
    "seam_assessment": "The belief representation (std::vector<uint32_t> bw) is held exclusively behind THREE clean seams. ENV SEAM: Environment owns all mutations via public methods. Callers never index bw directly. FEATURES SEAM: FeatureBuilder::build(loc,bw,collected) sole belief entry. belief_feats_() memoizes by BeliefKey, storing owned bw copies. belief_features(span<const uint32_t>) pure and env-free. Output crosses wire (float32 X/M); bw never does. POLICY SEAM: WorldSource::sample_world(const bw) abstract. RngWorldSource implements uniform sampling. Searches never index bw directly. FOR BITSET: Replace std::vector<uint32_t> with Belief. Env methods take &Belief. belief_features(const Belief&,...) value not span. belief_cache_ stores Belief copies. WorldSource::sample_world(const Belief&) adapts to bitset indexing. Searches copy Belief at nodes (1.9KiB vs 62KiB). No interface inflation—all TYPE SUBSTITUTIONS. Gate at env construction selects representation once (P1). Bitset operations O(1) in nb vs flat vector O(nb) sweeps.",
    "notes": "Belief NEVER CROSSES WIRE—only float32 features/masks/values. Belief constructed per-episode, mutated by env, consumed by featurizer/searches. Features.cpp: belief_features(span bw) pure kernel; belief_feats_() stores owned copies in cache (map<BeliefKey, pair<bw, BeliefFeatures>>); hit verifies full equality. Runner/Serve: bw local per-episode, env.apply() mutates in-place, write_results never sees it. Searches: nbw copies at nodes, belief_key(nbw) for child keys. Parity fixtures: gumbel_dump/ismcts_dump/nmcs_dump adapt sample_world bodies but RNG seeding/world samples identical. belief_sweep_oracle_check reference() verifies bit-exactness via independent env.observe() iteration. belief_filter_bench times filter_inplace on realistic beliefs. Bitset adaptation: belief_features(const Belief&) value parameter; belief_cache_ stores Belief copies; env methods take &Belief; WorldSource::sample_world(const Belief&) indexes via popcount+find_r-th_bit; searches copy Belief (constant 1.9KiB storage). Gate computed once at env construction from N/K/nD (fits L1 -> bitset else flat). No per-call overhead."
  }
]

```

## Draft synthesis

I have everything I need. I have read the decision note, env.hpp/cpp, policy.hpp, features.hpp/cpp, belief_key.hpp, feature_compute.hpp, gumbel.hpp + the hot paths of gumbel.cpp, ismcts.hpp + hot path, nmcs.hpp + hot path, runner.hpp/cpp, serve.cpp (belief touchpoints), search_runtime.hpp, cyclic_gumbel.hpp, the dump fixtures, and both bit-exact harnesses, plus the ADR-0012 synopsis. Now I'll produce the scoping report.

# SCOPING REPORT — Dense Bitset Belief Behind a Gated Two-Representation Seam

**Branch:** `cpp-actor-online-reconfig` · **Status:** read-only scoping, no code changed · **Decision settled** per `/home/bork/belief_bitset_decision_reversal.md` (read end-to-end). This report tells the implementer how to host BOTH the flat `std::vector<uint32_t>` (general base + fallback) and a dense bitset (gated fast path) under one search, with no rep leak into the search and no per-op overhead that erodes the win. It cites `file:line` throughout.

---

## 0. The one fact that frames the whole design

The belief is **already cleanly seamed for reads and mutation**, and **never crosses the wire**. Confirmed:

- The runner builds the belief from `env.worlds()` (`cpp/src/runner.cpp:24`), mutates it only via `env.apply` (`runner.cpp:68`), featurizes it via `fb.build` + `legal_mask` (`runner.cpp:49-50`), and `write_results` never serializes it — only the four float32 blocks (feat/pi/mask/value) cross. `serve.cpp` calls `run_episodes` (`cpp/src/serve.cpp:187`) and the only belief mention in serve is a comment (`serve.cpp:100`). **A belief never crosses the serve/runner boundary.** This is the load-bearing fact that contains the blast radius: the rep change is a C++-internal substitution, the wire/result contracts (P7) are untouched.
- `SearchTask::bw` (`cpp/include/chocofarm/search_runtime.hpp:53`) is a *member field* held by value and travels to the runtime via `std::span<const SearchTask>` — but it never leaves the process; it is the in-process task description.

So the entire change is internal. The work is: define ONE belief value type, route the handful of raw-vector pokes through it, gate the rep at env construction, and re-key the caches.

---

## 1. TOUCHPOINT INVENTORY (consolidated from the three maps + verified)

I verified the maps against the source with a tree-wide grep for `bw[`, `.front()`, `.back()`, `.size()`, `.empty()`, range-for `for (… : bw)`, `.data()`, `.begin()`. The maps are accurate. There are exactly **two categories**.

### 1A. GOES THROUGH THE SEAM (no work beyond a type substitution)

These name `bw` only to hand it to an `env` method, a `WorldSource`, the fingerprint, or `FeatureBuilder::build`. They do not inspect rep internals. They change by *retyping the parameter* `const std::vector<uint32_t>&` → `const Belief&` (and the by-value copies → `Belief`).

- **Env reader/mutator API** — `marginals` (`env.hpp:81`), `legal_actions` (`env.hpp:88`), `informative` (`env.hpp:112`), `filter_treasure`/`filter_detector` (`env.hpp:97-98`), `apply` (`env.hpp:93`). All bodies (`env.cpp:67-103,125-165`) move *inside* the seam (see §2).
- **Policy contracts** — `Policy::decide` / `decide_target` (`policy.hpp:60,70`), `RandomPolicy` (`policy.hpp:82`), `GreedyBase`/`GreedyStopBase` (`policy.hpp:99,110`), `candidate_actions` (`policy.hpp:120`), `base_value` (`policy.hpp:128`). Bodies call only env methods (`policy.cpp:23-128`).
- **Search contracts + hot paths** — Gumbel `decide`/`run_search`/`evaluate`/`descend`/`simulate_root_action` (`gumbel.hpp:144,158,188`; `gumbel.cpp:230-235,330-374,377-399`); ISMCTS `decide`/`leaf_value`/`iterate` (`ismcts.hpp:76,104`; `ismcts.cpp:120-180`); NMCS `decide`/`search`/`playout`/`eval_move`/`playout_value` (`nmcs.hpp:67,80,85,91,100`; `nmcs.cpp:61-140`). The node-local **copies** `nbw = bw` / `cur_bw = bw` (`gumbel.cpp:352,389`; `ismcts.cpp:134,159`; `nmcs.cpp:66`) become `Belief` copies — this is the value-semantics finding (§3): beliefs are copied **at descent steps, never held as node members**. Nodes hold only `(slot, BeliefKey)` children (`gumbel.hpp:113`, `ismcts.hpp:90`).
- **Featurizer entry** — `FeatureBuilder::build` (`features.hpp:107`; `features.cpp:258`), `legal_mask` oracle (`features.hpp:50`; `features.cpp:60`), `legal_mask_from_features` (already a pure float-span slice — `features.cpp:302`, unaffected).
- **Fingerprint call sites** — `gumbel_belief_key` (`gumbel.hpp:90`), the `belief_key(nbw)` calls in all three searches — these *call* the fingerprint, they don't compute it.
- **Runner/serve/dumps init** — `bw = env.worlds()` (`runner.cpp:24`; `gumbel_dump.cpp:201`; `ismcts_dump.cpp:142`; `nmcs_dump.cpp:107`; `mask_dump.cpp:78`) becomes `Belief b = env.full_belief()`.

### 1B. POKES THE RAW VECTOR (the actual work — each must be absorbed behind the seam)

Exactly **five distinct poke kinds**, all of which the seam must absorb or expose an equivalent op for:

| # | Site | Poke | Resolution |
|---|------|------|-----------|
| **L1** | `RngWorldSource::sample_world` `policy.hpp:152-154` | `bw.size()` + `bw[pick(rng_)]` | **Move sampling into `env`** as `env.sample_world(const Belief&, rng)` (note §"sample_world", lines 132-148). This is the *only production* leak. |
| **L2** | `belief_key` `belief_key.hpp:22-25` | `bw.size()`, `bw.front()`, `bw.back()` | Becomes `env.belief_key(const Belief&)` **or** a `Belief` method. For the bitset, `(nb, first-set-rank's world, last-set-rank's world)` (or just first/last set-bit index — see §6 re-key). Must stay collision-resistant; consumers still full-verify. |
| **L3** | `belief_features_nonempty` `features.cpp:190-193` | `for (uint32_t w : bw)` — the O(nb·(N+nD)) sweep | The §A.4 sweep is **replaced** by the bitset's masked-AND+popcount (note lines 41-47). This is the win. Flat keeps the sweep. |
| **L4** | Fixture `sample_world` overrides — `cyclic_gumbel.hpp:27` (`bw[0]`), `gumbel_dump.cpp:149-154`, `ismcts_dump.cpp:86-89`, `nmcs_dump.cpp:66` (`bw[0]` / `bw[idx%n]`) | scripted `bw[i]` | These must adapt to whatever `sample_world` becomes (§5 step 6). They need an **indexed/ranked accessor** on `Belief` (the r-th set bit → world). |
| **L5** | `belief_cache_` value type — `features.hpp:164`; store/verify `features.cpp:335,341` | stores `std::vector<uint32_t>` copy; `std::ranges::equal(entry.first, bw)` | Swap stored type to `Belief`; equality becomes `Belief::operator==` (bitset: array `==`; flat: vector `==`). One-line type change + the equality op. |

**Also note (L4-adjacent, a parity hazard):** the dump prefix-advance picks the true world as `bw[0]` (`gumbel_dump.cpp:208`; `ismcts_dump.cpp:149`; `nmcs_dump.cpp:114`) and the world-index FIFO indexes `bw[idx%n]`. `bw[0]` in flat order is the lowest-bitmask combination; in **rank-ordered bitset space `worlds[0]` is the same world** (both are `itertools.combinations`/Gosper ascending — `env.cpp:21-38` vs note lines 93-106 produce the **identical enumeration order**). So `bw[0]` → "the first set bit's world" is order-stable *for the initial full belief*, but for a **filtered** belief the "i-th element" differs between insertion-order (flat) and rank-order (bitset) only if the kept set is the same but the *positions* differ — which they are not, because both keep in ascending world order (flat `erase_if` preserves order; bitset is inherently ascending). **The kept SET and its ascending ORDER are identical between reps** (note §Equivalence, lines 156-166, claims order is *not* preserved — but that is about the bitset being rank-ordered vs an arbitrary insertion order; here the flat vector is *also* ascending because it starts ascending and `erase_if` preserves order). The implementer must confirm this: if true, the dump `bw[i]` fixtures are bit-exact across reps and the note's "sampling is behavioral re-baseline" caveat applies only to the *production RNG draw* `uniform_int(0,nb-1)` → r-th set bit, which **is** byte-identical (same r, same ascending r-th element). **This is the single most important thing to pin in step 1 of cutover** (§5).

---

## 2. THE SEAM — the exact boundary that hosts both reps

**The seam is the `Environment`'s belief API, with sampling and fingerprinting pulled IN, and the belief value type made opaque.** The env is already the closed box that owns world-space + masks (`env.hpp:50-119`); the bitset's masks are "env-static derived data homed once" exactly like `face_masks_` already is (`env.cpp:42-46`, note lines 85-91). The seam must expose this op set so **no caller pokes rep internals**:

**Belief value type** (opaque to callers):
```cpp
// flat arm: struct holds std::vector<uint32_t>;  bitset arm: struct holds std::array<uint64_t,kW64>
// dispatch via std::variant (see §3). Value-typed, copyable, ==-comparable.
```

**Op set the seam must expose** (each maps an existing touchpoint):

| Op | Replaces | Both reps |
|----|----------|-----------|
| `env.full_belief() -> Belief` | `bw = env.worlds()` (L1B init) | flat: copy `worlds_`; bitset: all-ones over nb worlds |
| `env.filter_treasure(Belief&, i, present)` | `env.hpp:97` | flat: `erase_if`; bitset: `&= ±treasure_mask[i]` |
| `env.filter_detector(Belief&, j, positive)` | `env.hpp:98` | flat: `erase_if`; bitset: `&= ±detector_mask[j]` |
| `env.apply(Loc&, Belief&, …)` | `env.hpp:93` | unchanged body, calls the two filters |
| `env.marginals(const Belief&)` | `env.hpp:81` | flat: sweep; bitset: `popcount_and(b, treasure_mask[t])·inv` |
| `env.legal_actions(const Belief&, collected)` | `env.hpp:88` | calls marginals + informative |
| `env.informative(j, const Belief&)` | `env.hpp:112` | flat: two-polarity scan; bitset: `0 < popcount_and(b,detector_mask[j]) < nb(b)` |
| `env.nb(const Belief&) -> int` | `bw.size()` / `bw.empty()` everywhere | flat: `.size()`; bitset: `Σ popcount` |
| `env.sample_world(const Belief&, rng) -> uint32_t` | **L1** `policy.hpp:152` | flat: `worlds_vec[uniform]`; bitset: r-th set bit → `worlds_[idx]` (note 132-148) |
| `env.belief_key(const Belief&) -> BeliefKey` | **L2** `belief_key.hpp:22` | rep-specific fingerprint (§6) |
| `belief_features(const Belief&, masks, dims)` | **L3** `feature_compute.hpp:29` | flat: §A.4 sweep; bitset: count lines |
| `Belief::operator==` | **L5** `features.cpp:335` | array `==` / vector `==` |
| `env.world_at_rank(const Belief&, r) -> uint32_t` | **L4** fixtures | the r-th set element (both: ascending) |

**Critical seam principle (P1/P7):** the bitset masks (`treasure_mask[t]`, `detector_mask[j]`) are derived from the SAME enumeration + face masks the env already owns (`env.cpp:41-46`). Build them in the ctor alongside `worlds_`/`face_masks_`. The identity `observe(j,w) == (w & face_masks()[j]) != 0` (`env.hpp:109`) is what makes `detector_mask[j]` derivable; the oracle (`belief_sweep_oracle_check.cpp:57`) already pins it.

**`nb`/`empty` is the highest-traffic op** — it appears at `gumbel.cpp:333,335,533`, `ismcts.cpp:186`, `nmcs.cpp:40,124,161`, `features.cpp:181,217`, `runner.cpp:37`. For flat it's `.size()==0`; for bitset it's a 243-word popcount-or-any. **Cache `nb` on the bitset belief** (a small `int count_` updated by `filter`) so `empty()`/`nb()` stay O(1) and do not become a hidden 243-word scan at every guard — otherwise the gate's win is partly eaten by the proliferation of empty-checks. This is a concrete P4/perf obligation for the implementer.

---

## 3. DISPATCH MECHANISM — recommendation

**Recommend `std::variant<FlatBelief, BitsetBelief>` as the `Belief` value type, with dispatch *inside the env methods* (a single `std::visit` or `if (holds_alternative)` per op), NOT at every call site.**

Rationale, grounded in the actual touchpoints:

1. **Beliefs are value-typed and copied at descent steps, never held polymorphically at nodes.** Verified: nodes store only `(slot, BeliefKey)` children (`gumbel.hpp:113`, `ismcts.hpp:90`); the belief is copied locally (`gumbel.cpp:352`, `ismcts.cpp:134`, `nmcs.cpp:66`) and passed down by const-ref. A `variant` is value-typed and copies cleanly (the bitset arm is 1.9 KiB, the flat arm a vector) — it fits the existing ownership exactly (P3 one-owner, P9 by-value). A **virtual `Belief` base would force heap allocation + a pointer**, breaking value semantics and the cheap node copy, and would add a vtable indirection to *every* op — rejected.
2. **The rep is chosen ONCE per env and is invariant for the env's life.** So the variant's active alternative is constant across a whole run. The branch predictor sees the same arm every time — the `std::visit`/`holds_alternative` dispatch is effectively free in the hot loop (a single perfectly-predicted branch), and it is paid **per env-method-call, not per world**. The expensive inner loops (the 243-word AND, the popcount) are *inside* the chosen arm with no per-iteration dispatch.
3. **Templating the search on the rep** (`GumbelAZPolicy<BitsetBelief>`) would give zero dispatch cost but **duplicates the entire search instantiation** for both arms, doubles compile time and binary size, and forces the *gate* to live at a type boundary the runner/serve/runtime would have to thread through `SearchTask`/`SearchRuntime` as a type parameter — leaking the rep into the scheduling layer (violates "no rep leak into the search," and fights `SearchTask` being a plain vector-storable struct, `search_runtime.hpp:51`). Rejected unless a profile later shows the variant branch is measurable (it will not be — it's a constant-arm branch).

**Cost summary:**
- `variant` + visit-in-env: **~one predicted branch per env op**; zero per-world dispatch; single search instantiation; gate stays a runtime value. Belief copy = active-arm copy.
- template-on-rep: zero dispatch but **2× code**, rep type leaks into `SearchTask`/runtime, gate becomes a compile-time fork.
- virtual base: heap + vtable per op + breaks value semantics at nodes. Worst fit.

**One caveat for the implementer:** keep the `std::visit` *coarse* — dispatch once at the top of each env method, then run a pure rep-specific body. Do not `visit` inside the per-world loop. (The note's inline functions `nb`/`popcount_and`/`filter`, lines 120-130, are the bitset bodies; the flat bodies are the current `env.cpp` loops.)

---

## 4. THE GATE — where the threshold is computed and where selection happens

**Threshold (derived, homed once — P1, ADR-0012 P4 "a value's heat is decided by where it lives"):**

Computed in the `Environment` ctor from `N`/`K`/`nD`, alongside `worlds_`/`face_masks_` (`env.cpp:40-52`). The feasibility/budget quantity is **`(N + nD) · |worlds| bits` of mask storage** (note lines 87-91: `treasure_mask` ~38 KiB + `detector_mask` ~85 KiB ≈ 122 KiB for the live instance, "single-digit µs, L2-bandwidth-bound"). The gate predicate:

```
use_bitset = ( |worlds| enumerable AND  kW64 = ceil(|worlds|/64)  fits the per-belief budget
               AND  (N+nD)*kW64*8 bytes  fits the mask-cache budget )   // L2-resident
```

The budget constant is a single named `constexpr` (e.g. a mask-bytes ceiling), NOT a scattered magic number — homed next to where it's read, derived from the same dims. `kW64` is derived (`(|worlds|+63)/64`, note line 93), never hardcoded as 243.

Store the result as an env-static derived datum: `bool use_bitset_` (plus the bitset masks, built only when `use_bitset_`). This is the "same homing question already open for the distance matrix and detector masks; resolve them together" the note flags (lines 170-173).

**Selection (one-time, behavior-neutral):**

In the ctor: if `use_bitset_`, build `treasure_mask`/`detector_mask` and have `full_belief()`/the env belief ops take the bitset arm; else the flat arm. **No call site decides** — the variant's active alternative is fixed here and every op visits it. Because both reps are bit-exact (filters/counts/determinizations — §5 validation), the gate is a pure feasibility/speed choice that cannot change outputs (P6/P7).

For large/non-enumerable `|worlds|` (a future variant, note lines 22-27, 179-185) the gate falls back to flat automatically — **flat is the necessary fallback, not legacy.**

---

## 5. CUTOVER PLAN — ordered, with bit-exact validation at each step

The principle: introduce the seam first on the *flat* arm (a pure refactor, must be byte-identical), then add the bitset arm behind the gate, validating each rep against the other and against the existing oracles.

**Step 0 — Pin the order-equivalence claim (BLOCKING).** Before any code: confirm the flat vector stays ascending-world-ordered through filter→featurize (it does: `worlds_` starts ascending `env.cpp:21-38`; `erase_if` preserves order `env.cpp:121`). Confirm the bitset's rank order == this ascending order (note 93-111). If equal, the r-th set bit == the r-th flat element for the SAME `uniform_int(0,nb-1)` draw ⇒ **sampling is byte-identical, not a re-baseline** (this contradicts the note's §Equivalence caveat lines 156-166, which the implementer must reconcile — the caveat assumed arbitrary insertion order). Record the finding; it decides whether the gumbel/nmcs/ismcts parity fixtures stay bit-exact or need an N≥300 behavioral re-baseline.

**Step 1 — Introduce the `Belief` seam on the flat arm only (pure refactor).** Define `Belief = variant<FlatBelief,...>` with only the flat arm populated; move `sample_world` into `env` (L1); make `belief_key` an env/Belief op (L2); retype all 1A signatures; swap `belief_cache_` value type (L5). *Validation:* the entire existing test + parity suite must pass byte-identically (this step changes no math). Run `belief_sweep_oracle_check` (`belief_sweep_oracle_check.cpp` — PASS unchanged), the gumbel/ismcts/nmcs parity dumps, and `tests/`. This isolates the type-plumbing from the bitset math.

**Step 2 — Add the bitset arm + the gate (off by default first).** Build `treasure_mask`/`detector_mask` in the ctor; implement the bitset bodies (note 116-148). Add the gate but force flat. *Validation:* an explicit **flat-vs-bitset A/B harness** (extend `belief_sweep_oracle_check.cpp` — it already builds beliefs as prefixes/strided subsets, lines 106-114): for each sampled belief, build BOTH reps, assert `marginals`/`informative`/`legal_actions`/`belief_features` are **byte-identical** (the note's strongest P6 tier, lines 156-162; counts are exact integers). This is the bit-exact bar that transfers directly.

**Step 3 — Sampling A/B.** For a fixed RNG stream, assert `env.sample_world(flat,rng)` == `env.sample_world(bitset,rng)` over many draws on many beliefs (per Step 0, expected byte-identical if order-equiv holds). If Step 0 found order does differ, re-baseline behaviorally (N≥300 episodes, ≥2 seeds, within MC CI — the note's bucket, lines 162-166).

**Step 4 — Enable the gate; end-to-end parity.** Flip the gate on for the live instance. *Validation:* the gumbel parity end-to-end (`cpp/parity/gumbel_*`) and the runner/serve episode traces (`runner.hpp:53-59` records `(world, slots)` for replay). With Step 0 holding, these stay bit-exact; the dumps' `bw[0]`/`bw[idx%n]` fixtures (L4) resolve to the same worlds.

**Step 5 — Re-key the caches (§6) and confirm hit-rate.** Tune the bitset `BeliefKey` (§6) and confirm cache correctness via the full-equality verify (`features.cpp:335`, now `Belief::operator==`).

**What this SUPERSEDES** (retire when the gate is on, for the bitset arm):
- The §A.4 fused sweep `belief_features_nonempty` (`features.cpp:178-210`) — replaced by masked-AND+popcount (note 41-47). **Keep it as the flat arm's body.**
- The flat `filter_inplace` `erase_if` (`env.cpp:120-123`) — replaced by `&= mask` for bitset. **Keep as flat arm.**
- The branchless-filter and SIMD-compaction/pos-popcount rungs (note lines 78-81, 60-62) — **moot** for the bitset (nothing to compact). They remain the flat arm's deferred options.

**What this KEEPS unchanged:**
- The Gosper/`next_combination` march and the enumeration ORDER (`env.cpp:21-38`; note 93-106) — the bitset *indexes* this same order.
- The offset cache / layout SSOT in `FeatureBuilder` (`features.cpp:131-140`) — feature *assembly* is rep-agnostic (it consumes `BeliefFeatures`, which both reps produce identically).
- `build`'s assembly (`features.cpp:258-300`), `legal_mask_from_features` (the float-span slice, `features.cpp:302`).
- The bit-exact **oracle** (`belief_sweep_oracle_check.cpp`) — extended to A/B both reps, the regression net for both.
- The gumbel/ismcts/nmcs **parity** structure — the scripted sources adapt their `sample_world` body (L4) but feed the same worlds.
- `belief_filter_bench` (`belief_filter_bench.cpp`) — its `branchless_ref`/`filter_inplace` timing stays the flat arm's bench; add a bitset-`&=` timing if a filter profile is wanted.

---

## 6. RISKS / OPEN NUMBERS

1. **`sample_world` O(1) → O(243) (the note's open number).** Flat sampling is a single index (`policy.hpp:154`); bitset sampling scans up to 243 words to find the r-th set bit (note 134-148, BMI2 `pdep`+`tzcnt` accelerates). **Frequency:** called per c_outcome>0 determinization in gumbel (`gumbel.cpp:387`, c_outcome=2 so once per root-action sim), per descent step in nmcs (`nmcs.cpp:125`), per ismcts step. It was **not** flagged as a profile hotspot (the profiles named the *belief sweep* at ~81% — `feature_compute.hpp:7` — and `FeatureLayoutSpec::start` at 2.3%; sampling never appears). The 243-word scan is sub-µs (note 64-77); the bitset *eliminates* the 81% sweep, which dwarfs any sampling regression. **Open: measure sample_world frequency vs the sweep savings** — almost certainly net-positive, but record it (P6). Mitigate by early-exit on the word containing the r-th bit (note's loop already does, line 138-143).

2. **`nb`/`empty` traffic (raised in §2).** ~12 guard sites do `bw.empty()`/`.size()`. For bitset these must be O(1) via a cached `count_`, not a 243-word recount each guard. If left as a recount, the gate's win erodes at the guards. **Concrete obligation:** the bitset `Belief` caches its popcount, updated in `filter`. (Open number from the note, lines 192-196: "how many beliefs are live/copied at once" — favors the 1.9 KiB compact encoding regardless.)

3. **Node belief-copy cost.** Each descent step copies the belief (`gumbel.cpp:352`, etc.). Bitset copy = fixed 1.9 KiB; flat copy = up to 62 KiB and variable (note line 59). **Bitset is ~32× smaller and constant-size at nodes** — a *side benefit*, not a risk. The risk is only if the variant's flat arm's vector-copy is somehow worse than today's — it is not (same vector).

4. **The belief-cache re-key (L5 + §6 fingerprint).** Today `belief_key` = `(size, front, back)` (`belief_key.hpp:24`) — order-insensitive, collision-resistant, full-verified on hit (`features.cpp:335`). For the bitset, `front`/`back` (first/last *world value*) require finding the first/last set bit (cheap: `countr_zero` of first non-zero word / `countl_zero` of last). **The fingerprint must remain consistent across reps if any cache outlives a rep switch** — but the rep is fixed per env, so the cache only ever sees one rep; the key just needs to be cheap + collision-resistant *within* the chosen rep. **Open:** confirm the bitset key's collision profile is no worse (same triple semantics, so equivalent). The full-equality verify (`Belief::operator==`) is the safety net regardless. Note the SAME key is shared by FeatureBuilder, gumbel nodes, ismcts nodes (`gumbel.hpp:89`, `ismcts.hpp:90`) — re-key in ONE place (the env/Belief op), all three derive it (P1).

5. **The `bw[0]`-as-true-world dump convention (L4 / §1B note).** The dump prefix-advance and the fixture `sample_world` index `bw[0]`/`bw[idx%n]` (`gumbel_dump.cpp:208`, `ismcts_dump.cpp:86`, `nmcs_dump.cpp:66`). These are parity-load-bearing. If Step-0 order-equivalence holds, they're byte-identical across reps; if not, the parity fixtures need the behavioral re-baseline. **This is the single risk that could turn a "bit-exact" claim into a "behavioral" one** — pin it first (Step 0).

6. **The note's §Equivalence "order not preserved" caveat (lines 156-166) vs the flat vector's actual ascending order.** The note assumed sampling becomes a behavioral re-baseline because a bitset is rank-ordered. But the flat vector is *also* ascending (Step 0). The implementer must reconcile these two claims before declaring the sampling bit-exact — they cannot both be assumed. This is the one place the expert note may be conservative; verify against the code, don't inherit the caveat.

7. **`variant` exhaustiveness / fail-loud (ADR-0002, P9).** Every `std::visit` must handle both arms; a `valueless_by_exception` or an unhandled alternative aborts loudly. The `sample_world` invariant (`r < nb` guarantees a hit, note line 146 `std::abort()`) and `apply`'s TERMINATE abort (`env.cpp:144-146`) are the existing fail-loud pattern to mirror.

---

**Documentation obligation (ADR-0005):** this change makes `belief_features_and_decision_diagram_note.md`'s §A.4 sweep the *flat-arm* body (no longer the sole path) and realizes the note `/home/bork/belief_bitset_decision_reversal.md` — `docs/STATUS.md` and the current handoff likely describe the belief sweep as *the* hot path; that orientation surface will need a dated amendment when the gate lands. The ZDD note stays valid as the future-variant hedge (note lines 179-185). Flag this to the implementer as part of the delivery, not after.

**Files the implementer touches (all absolute):** `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/env.hpp` + `cpp/src/env.cpp` (Belief type, gate, masks, sampling, filters, marginals, informative, belief_key home); `cpp/include/chocofarm/policy.hpp` (retype contracts, remove `WorldSource::sample_world`'s rep poke L1); `cpp/include/chocofarm/belief_key.hpp` (rep-aware fingerprint L2); `cpp/include/chocofarm/feature_compute.hpp` + `cpp/src/features.cpp` (sweep → masked-popcount on bitset arm L3, cache value type L5); `cpp/include/chocofarm/features.hpp` (`belief_cache_` type L5); `cpp/include/chocofarm/{gumbel,ismcts,nmcs}.hpp` + `cpp/src/{gumbel,ismcts,nmcs}.cpp` (retype, local copies); `cpp/include/chocofarm/search_runtime.hpp` (`SearchTask::bw` type); `cpp/include/chocofarm/cyclic_gumbel.hpp` + `cpp/src/{gumbel_dump,ismcts_dump,nmcs_dump}.cpp` (fixture sample_world L4); `cpp/src/belief_sweep_oracle_check.cpp` (extend to flat-vs-bitset A/B). `cpp/src/runner.cpp`/`serve.cpp` change only at `bw = env.worlds()` → `env.full_belief()` and the type; **no wire/result change.**


---

## Adversarial critique

I now have complete visibility. All the report's cited file:line references check out. The serve.cpp:100 mention is indeed a comment about world-set allocation, not a belief crossing the wire — confirming the wire-boundary claim. The `belief_shrinkage` at runner.cpp:98 reads `bw.size()` — a touchpoint the report's §2 nb-traffic list does NOT enumerate (it cited runner.cpp:37 but not 98 or 27). Minor.

I have everything needed for the adversarial review. Let me synthesize my findings against the five review axes.

---

ADVERSARIAL REVIEW OF THE SCOPING REPORT

Verdict: the report is substantially correct and implementable. Its central, contested claim — that sampling is bit-exact, contradicting the expert note's §Equivalence caveat — is RIGHT, and I verified it against the source. The seam, dispatch, and gate recommendations are sound. The defects are at the margin: a handful of understated/missed touchpoints, one place where it inherits a hazard it should have killed outright, and two spots where it hedges a claim that the code actually settles. Detail below, keyed to your (a)-(e).

(a) DOES THE SEAM HOST BOTH REPS WITHOUT A RAW-VECTOR LEAK? — Yes, with one missed poke and one mischaracterization.

The report's 1A/1B split is accurate. I independently grepped every `bw[`, `.front/.back/.size/.empty`, range-for, `.data/.begin` across the searches, env, policy, features, runner, serve, and the fixtures. The five poke-kinds (L1 sample, L2 belief_key, L3 sweep, L4 fixtures, L5 cache value) are the complete set of rep-internal accesses, and each maps to a seam op. Confirmed against source:
- L1 RngWorldSource (policy.hpp:152-154): the only production rep poke. Correctly routed to env.sample_world.
- L2 belief_key.hpp:22-25 front/back/size. Correct.
- L3 features.cpp:190 `for (uint32_t w : bw)`. Correct.
- L4 fixtures bw[0]/bw[raw%n] (gumbel_dump.cpp:150,153; ismcts_dump.cpp:86,89; nmcs_dump.cpp:66; cyclic_gumbel.hpp:27). Correct.
- L5 belief_cache_ value type + std::ranges::equal (features.hpp:164; features.cpp:335,341). Correct.

MISSED TOUCHPOINT: `runner.cpp:98` `out.belief_shrinkage = ... bw.size() / bw0` and `runner.cpp:27` `bw0 = bw.size()`. The report's §2 nb-traffic list cites "runner.cpp:37" but not 27/98. This is a real `bw.size()` read on a NON-empty, possibly-collapsed belief at episode end. It must go through `env.nb(belief)`. It is harmless (covered by the same `env.nb` op the report does define) but it is an enumerated-site omission in a report whose whole value is the exhaustive touchpoint list — and it is exactly the kind of site that, left as `belief.bits.size()`-style direct access, silently returns 243 (the word count) instead of the popcount. Flag it.

MISCHARACTERIZATION (minor): the report says nodes "hold only (slot, BeliefKey) children ... the belief is copied locally and passed down by const-ref." Correct, but it should state the stronger structural fact it relies on: `GumbelNode.children` is `std::map<std::tuple<int, GBeliefKey>, int>` (gumbel.hpp:113) and `ISMCTSNode.children` is `std::map<std::tuple<int, BeliefKey>, int>` (ismcts.hpp:90) — the node graph is keyed by the FINGERPRINT, never by the belief value, so the rep type literally cannot enter the node arena. That is what makes the variant-belief a pure descent-local value. The report asserts the conclusion but doesn't cite the map type that guarantees it. (Not a defect in the recommendation, only in the substantiation — which, per ADR-0009, the report is supposed to attach.)

(b) IS THE DISPATCH SOUND? — Yes. The std::variant<FlatBelief, BitsetBelief> recommendation with visit-inside-env is correct, and the report correctly rejects the two alternatives for the right reasons.

- Value semantics / node copies: verified. The descent-local copies are `std::vector<uint32_t> nbw = bw` (gumbel.cpp:352,389; ismcts.cpp:134,159; nmcs.cpp:46,66). A variant copies its active arm; this preserves the existing ownership (P3/P9). Correct.
- std::variant SIZE BLOWUP — the report addresses this implicitly but NOT explicitly, and your prompt asks specifically. A `std::variant<FlatBelief, BitsetBelief>` is sized to its largest alternative. BitsetBelief is `std::array<uint64_t,243>` = 1944 bytes (+ the cached count_ + discriminant). FlatBelief is a `std::vector` = 24 bytes. So EVERY belief value — including in the FLAT arm, the large-|worlds| fallback case where the bitset is INFEASIBLE — pays 1.9 KiB of stack/inline footprint. In the flat fallback regime (the case the bitset can't serve), you've bloated the base rep's value size ~80x for an alternative that is never active. This is the one dispatch concern the report leaves unaddressed despite naming "no std::variant size blowup problem unaddressed" as a review target. It is probably acceptable (beliefs are descent-local, not held en masse — the note's open-number #1), but the report should have surfaced it and noted the mitigation: the bitset arm can hold a `std::unique_ptr<std::array<...>>` or the array can be heap-backed, trading the 1.9 KiB inline for a pointer + an allocation per belief copy. That allocation-per-copy would erode the win (copies happen per descent step), so the better answer is to ACCEPT the 1.9 KiB inline variant and note that the flat-fallback regime is not the perf-critical one (it's the feasibility fallback). Either way: the report should have made the call, not skipped it.
- Per-op dispatch cost: the "constant active arm ⇒ perfectly-predicted branch" argument is correct and the "keep std::visit coarse, never inside the per-world loop" caveat is the right and load-bearing instruction.

(c) IS THE GATE A DERIVED-AND-HOMED-ONCE FEASIBILITY FACT? — Yes in design, but the report leaves the threshold UNDER-SPECIFIED in a way that risks reintroducing a magic number.

The report correctly frames the gate as: computed in the Environment ctor from N/K/nD, homed as `use_bitset_` + the mask tables, alongside `worlds_`/`face_masks_` (env.cpp:40-52), with kW64 derived as `(|worlds|+63)/64` (never the literal 243). It correctly frames it as FEASIBILITY (flat is the necessary base/fallback), satisfying ADR-0012 P1 and the "flat is not legacy" constraint. Good.

BUT: the gate predicate it gives — "kW64 fits the per-belief budget AND (N+nD)*kW64*8 fits the mask-cache budget" — depends on "a single named constexpr ... a mask-bytes ceiling." That ceiling IS a magic number unless it is itself derived from a named hardware fact. The report calls for it to be "homed next to where it's read, derived from the same dims" — but a cache-size ceiling is NOT derivable from N/K/nD; it is a property of the target machine (the note's i5-6600 L2). So the honest framing is: the gate has TWO inputs — a derived quantity (the mask bytes, a pure function of N/K/nD) and a machine constant (the cache budget). The report blurs these into "derived." For P1 fidelity the implementer must home the machine-constant ceiling explicitly as what it is (a named target-cache budget with the i5-6600 L2 figure and rationale in its comment), not pretend it falls out of the dims. The report understates this; left as written, an implementer could pick "122 KiB fits, ship it" and bury the actual number — exactly the scattered-magic-number failure the gate is meant to avoid. (Also: the report never states a concrete provisional threshold or which side the LIVE instance lands on beyond asserting it's feasible; the note gives 122 KiB masks / 1.9 KiB belief, comfortably L2-resident, so use_bitset_ = true for N=20,K=5. The report should commit that the live instance is on the bitset side, which it implies but doesn't pin.)

(d) BIT-EXACTNESS AND ADR-0012? — The bit-exactness verdict is CORRECT and is the report's strongest contribution; the ADR-0012 posture is sound. One overclaim to temper and one convention the report half-states.

The report's headline correction — that sampling is byte-identical, NOT the behavioral re-baseline the expert note's §Equivalence (note lines 161-166) declares — is RIGHT. I verified the chain:
1. build_worlds (env.cpp:21-38) emits worlds in ascending bitmask order (the next-combination walk).
2. filter_inplace uses std::erase_if (env.cpp:120-123), which preserves relative order — so the flat belief stays ascending through filter→featurize.
3. The bitset is rank-ordered = the same ascending enumeration (the note's own construction, lines 93-106).
4. Therefore the r-th flat element == the r-th set bit's world, for the same uniform_int(0,nb-1) r. So flat `bw[pick]` (policy.hpp:153) and the bitset "r-th set bit" return the identical world. Byte-identical determinizations.
The codebase CORROBORATES this against the note: the fixtures' own headers say "bw[0] (the lowest-bitmask world; itertools/combinations order is the same on both sides)" (ismcts_dump.cpp:11, nmcs_dump.cpp:10) and "bw[0] ... Gosper ascending." The note's caveat assumed arbitrary insertion order; this codebase's flat vector is ascending, so the caveat does not apply here. The report is correct to force the implementer to reconcile this (Step 0, BLOCKING) rather than inherit the note's behavioral bucket. This is the single most valuable judgment in the report.

HOWEVER — the report then HEDGES its own correct finding. It repeatedly says "if Step-0 holds" / "if true" / "almost certainly" and routes a fallback to "N≥300 behavioral re-baseline" (its Step 3, Risk 5, Risk 6). Given that the order-equivalence is not a guess but a provable property of erase_if + ascending enumeration (and is already the documented basis of the existing parity fixtures, which PASS), the hedge is too soft. If the bitset's r-th-set-bit sampler is implemented to the note's spec, sampling parity is a THEOREM, not an empirical question — and the existing gumbel/ismcts/nmcs parity fixtures already encode bw[0]-order determinism as their contract. The report should state that order-equivalence is established by construction and the A/B harness CONFIRMS rather than DISCOVERS it; the "behavioral re-baseline" fallback should be marked as the branch that only fires if the implementer DEVIATES from ascending order (e.g. a hash-set flat rep), which nothing here does. As written, the report leaves a reader thinking byte-exactness is in doubt when the code settles it.

THE `* inv` CONVENTION (under-stated): the report's §5 "What this SUPERSEDES" says the bitset replaces the §A.4 sweep, but does not flag that the bitset's popcount counts must feed the SETTLED `* inv` normalization (features.cpp:154-157, 198-205), not the original `/ nb`. The note says `* inv` (note line 152), and the oracle (belief_sweep_oracle_check.cpp:59-63) pins `* inv` as THE reference. The bitset is byte-identical to the current sweep ONLY because both apply `* inv` to exact integer counts. The report asserts "byte-identical features" without naming the convention that makes it so — an implementer who applies `/ nb` to the bitset counts would break the oracle. This is exactly the kind of load-bearing convention ADR-0002/ADR-0009 want named. Minor but concrete.

ADR-0012 posture: the report correctly keeps FLAT as Band-1/2 base, BITSET as a sanctioned specialization gated once (not premature-E, the abstraction is paid for by the measured 81% sweep cost), the seam minimal (env belief API), and the wire/result contracts untouched (P7) — I confirmed write_results ships only X/PI/M/Y (runner.cpp:177) and serve.cpp only calls run_episodes (serve.cpp:187), never a belief. The documentation-obligation call-out (ADR-0005, the belief-sweep-as-hot-path orientation surface needs a dated amendment) is correct and appropriately scoped.

(e) UNDERSTATED TOUCHPOINTS / RISKS:

1. THE FIXTURE bw[0] CONVENTION IS A HARD COUPLING, NOT A "hazard to pin." The report (Risk 5/6) treats the fixtures' bw[0]/bw[raw%n] as a possible parity-break to reconcile. But it is fully determined: bw[0] = lowest remaining combination = first set bit (rank 0). The seam needs `env.world_at_rank(belief, r)` and the fixtures call `world_at_rank(b, 0)` / `world_at_rank(b, raw%nb)`. That is mechanical and bit-exact by the same order-equivalence. The report's framing (could turn "bit-exact" into "behavioral") overstates the risk — the only way it goes behavioral is an implementer choosing a non-ascending flat rep, which is out of scope. The report should downgrade this from "single most important risk" to "mechanical, follows from Step 0."

2. nb O(1) CACHING — the report's strongest concrete catch, correctly raised (§2, Risk 2): the ~12 `bw.empty()`/`.size()` guards (verified: gumbel.cpp:333,335,533; ismcts.cpp:186; nmcs.cpp:40,124,161; runner.cpp:27,37,98; features.cpp:181,217) must hit an O(1) cached count_ on the bitset, not a 243-word recount. This is a genuine perf obligation and the report names it well. (It missed runner.cpp:27/98 from the enumerated list, per (a).)

3. THE belief_key COLLISION PROFILE — the report says the bitset key "(first/last set-bit world)" stays collision-resistant because the full-equality verify backstops it. True, BUT: there is a subtle correctness point it glosses. The current key is (size, bw.front(), bw.back()) (belief_key.hpp:24) — the actual first/last WORLD VALUES. The bitset's equivalent is (popcount, world_at_rank(0), world_at_rank(nb-1)). Because order is ascending in BOTH reps, these triples are IDENTICAL across reps for the same belief — which matters more than the report says: it means a bitset node-cache key equals the flat node-cache key for the same information set, so the cache hit-rate and the gumbel transposition behavior are preserved exactly, not just "collision-resistant." The report treats key-equivalence as a within-rep concern ("the rep is fixed per env, so the cache only sees one rep") — correct that it's fixed per env, but it undersells that the key is bit-IDENTICAL across reps, which is what lets the parity fixtures pass unchanged. A small strengthening, same direction as the §A.4 point.

4. `apply` SIGNATURE / IN-PLACE FILTER — the report routes filter_treasure/filter_detector and apply through the seam (env.cpp:125-165). One thing it doesn't call out: `apply` mutates `bw` IN PLACE (env.hpp:93, returns StepResult, belief mutated by reference). The variant must support in-place mutation through the visited filter — i.e. `env.filter_treasure(Belief&, ...)` takes a mutable Belief& and the visit dispatches to either erase_if (flat) or `&= mask` (bitset, updating count_). The report's op table shows this (`filter_treasure(Belief&, i, present)`) so it's covered, but the in-place-mutation-through-variant detail (the visit must be on a mutable reference, and the bitset arm must update its cached count_ in the same call) deserves an explicit line — it is where the O(1)-nb obligation and the mutation seam intersect.

5. THE TWO HARNESSES — the report says belief_sweep_oracle_check extends to an A/B harness (correct: it already builds prefixes + a strided subset, oracle_check.cpp:106-114, and diffs BeliefFeatures field-by-field via equal_features, lines 73-83 — a clean A/B target). It says belief_filter_bench's filter_inplace timing "stays the flat arm's bench" (correct: branchless_ref/filter_inplace, bench.cpp:51,106). Both claims check out. But note the oracle currently diffs `belief_features(flat) vs naive-reference(flat)` — extending it to `belief_features(bitset) vs naive-reference(flat)` requires the bitset to produce a BeliefFeatures with the SAME field set, which it does by design. Accurate.

SUMMARY: The report is implementable as-is and its load-bearing judgment (sampling is bit-exact, the note's caveat is void here) is correct and well-substantiated against the source. Required fixes before handing to an implementer:
- Add runner.cpp:27 and :98 (belief_shrinkage) to the nb-traffic touchpoint list (the one missed site).
- Address the variant size question explicitly (1.9 KiB inline paid even in the flat-fallback regime; recommend accepting it, with the reason).
- Split the gate threshold into its derived part (mask bytes from N/K/nD) and its machine-constant part (the target-cache ceiling), and home the latter honestly as a named hardware fact — do not pass it off as "derived." Commit that the live instance lands on the bitset side.
- Name the `* inv` normalization convention as the thing the bitset must reproduce (not `/ nb`), since the oracle pins it.
- Downgrade the bw[0]/sampling "behavioral re-baseline" framing from a live risk to a mechanical consequence of ascending order; state order-equivalence as established-by-construction (erase_if preserves order + ascending enumeration), confirmed-not-discovered by the A/B harness.
- Strengthen: the bitset belief_key triple is bit-IDENTICAL to the flat one (not merely collision-resistant), which is what preserves transposition/cache behavior and lets the parity fixtures pass unchanged.

None of these overturn the design. The seam hosts both reps without leaking the rep into the search; the dispatch is right; the gate is a genuine derived-feasibility fact (modulo the machine-constant honesty fix); bit-exactness and ADR-0012 are preserved.

Files I read end-to-end to verify: /home/bork/belief_bitset_decision_reversal.md; cpp/include/chocofarm/{policy,env,belief_key,feature_compute,features,gumbel,ismcts,nmcs,cyclic_gumbel,search_runtime}.hpp; cpp/src/{env,features,gumbel,ismcts,nmcs,runner,gumbel_dump,belief_sweep_oracle_check}.cpp; and the relevant spans of cpp/src/{serve,ismcts_dump,nmcs_dump,mask_dump,belief_filter_bench}.cpp.
