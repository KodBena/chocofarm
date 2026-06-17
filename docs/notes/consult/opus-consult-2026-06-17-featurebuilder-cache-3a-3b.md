<!-- docs/notes/consult/opus-consult-2026-06-17-featurebuilder-cache-3a-3b.md -->

# Consult: FeatureBuilder belief/loc cache (3a) + mask-from-features (3b) under ADR-0012

**Commissioned:** 2026-06-17, by the maintainer, before introducing the 3a memoization cache (a new
data source) — to verify it does not run afoul of ADR-0012, and to settle the structural shape
("memoization-wrappers" vs "free functions lying around").

**Method:** a 3-agent consultation — SUGGEST (proposing architect) -> CRITIQUE (adversarial reviewer,
default-skeptic) -> SYNTHESIS (consultant of record, adjudicating on the merits). Each agent read
ADR-0012 in full, the refactor dossier in full, and the post-3b `features.{hpp,cpp}` / `gumbel.cpp` /
`features.py` / `serve.cpp` / `runner.cpp` end to end, verifying every load-bearing claim against the
source. Phases 1-2 are preserved verbatim in the appendices (ADR-0005: keep the working record).

**Headline.** 3b is compliant (with two cheap follow-ups). 3a is admissible but **gated on a C++
belief-recurrence measurement that has not been taken** — the prior "the Python ~3.5x already meets the
gate" justification is an ADR-0009 dodge (a Python number transferred onto an unmeasured C++ artifact).
The likely-dominant recompute is the two-builder root duplication, whose P1-honest fix is sharing the
root vector — possibly obviating the cache. See the Implementer's summary (§7).

Public Domain (The Unlicense).

---

# Consultation of Record — 3a (belief/loc memoization) and 3b (legal-mask-from-features) under ADR-0012

## 1. Question & scope

Two changes to the C++ `FeatureBuilder` hot path, adjudicated against ADR-0012 (Compositional and Structural Hygiene) and its supporting tenets (ADR-0002 fail-loudly, ADR-0003 bands/extraction-on-second-instance, ADR-0009 measure-first, ADR-0011 runtime backstops):

- **3b (landed, `bc73a79`)** — `FeatureBuilder::legal_mask_from_features(std::span<const float>)` slices the already-built feature vector for the legal mask, replacing two fresh `env.legal_actions → marginals` sweeps per leaf in `GumbelAZPolicy::evaluate`. Reviewed for retroactive compliance.
- **3a (proposed)** — memoize the belief-derived intermediates (`BeliefFeatures`) and per-loc distances (`GeometryFeatures`) on `FeatureBuilder`, porting Python's `_belief_cache` / `_loc_cache`. The subject of the maintainer's two concerns: (1) is a new cache a P1/P9/P4/cancer-C/P6 violation; (2) what is the cleanest ADR-0012-aligned *shape*, resolving the "free functions lying around" discomfort.

I read end-to-end: `docs/adr-synopsis.md`; ADR-0012 in full; the dossier in full; `features.hpp`/`features.cpp` (POST-3b); `gumbel.cpp` (`evaluate`, `descend`, `run_search`, the `(slot, GBeliefKey)` transposition, `node.evaluated`) and `gumbel.hpp` (`GBeliefKey` :88, `GumbelNode`); `features.py` (the cache idiom, `reset_belief_cache`, `_LAYOUTS`); `serve.cpp` (`ServeState`, the build-once/instance-reject/version-reload gating); `runner.cpp` (`run_episode`/`run_episodes`); `instance.hpp` (`Point`). I had a SUGGEST proposal and an adversarial CRITIQUE in front of me; I verified every load-bearing factual claim either side made against the source rather than trusting the briefs. Five of the critique's findings are corroborated by the code; one of its sub-claims (the `bit_cast` width bug) is correct and damning of the SUGGEST's concrete keying code; the SUGGEST's narrow correctness defenses (not-cancer-C, not-a-second-marg-home, bit-exact-on-hit) survive intact.

## 2. Verdict

**3b: YES, compliant as landed.** It is a genuine P1 win (marg collapsed from three homes to one) and a clean P9 pure-slice. It carries **one residual debt**, not a violation: an *undeclared* `build()`→`legal_mask_from_features` layout coupling that the `std::span<const float>` signature erases into a comment (a P8/cancer-G call-boundary tension). The fix is a one-line contract assertion plus a documentation line, not a redesign. The SUGGEST's framing of 3b as "a *model* of P1 to cite" overstates by ignoring this seam; the CRITIQUE's framing of it as a rubber-stamped P8 violation overstates the severity. The truth is in between and I rule it below.

**3a: YES-WITH-CHANGES — but gated, and the gate is not yet met.** The maintainer's core fears are answerable: a member cache keyed by belief *value* with full-equality verification is **not** cancer-C, is **not** a second home for `marg`, and a hit is **bit-exact**. Those SUGGEST defenses survive. But two CRITIQUE findings are real and I uphold them as blocking the design *as proposed*:

1. **The justification inverts its own cited authority (ADR-0009 / dossier §4.4).** The dossier tags belief-feature memoization **HOLD** with the explicit gate *"needs real access-pattern data. Measure belief recurrence before building"* (dossier:176). SUGGEST claims "the measure-first gate 4.4 is already met" by transferring a *Python* recurrence number (~3.5×) onto an **unmeasured C++ artifact**. That is the precise dodge ADR-0009 and ADR-0012 P6 forbid. **3a may not land on this justification.** It is gated on a C++ recurrence measurement that has not been taken. This is the one finding that, left unaddressed, makes the proposal dishonest in the ADR-0012 sense.

2. **The homing argument is false for the serve path's process-scoped builder.** SUGGEST asserts every `FeatureBuilder` "dies with the builder, which dies with (or before) the env," and rules the concurrency objection SURVIVES because "the builder is never shared across threads." That is true for the per-task runtime builders. It is **false for `st.fb`**, which `serve.cpp:70/97` builds **once on the process heap and never rebuilds** for the daemon's entire life — across every episode of every generation, bounded only by the 50000 whole-cache-stomp, with **no** per-episode reset (the very mechanism Python relies on, `features.py:288-292`). The `const build()` + `mutable` cache is honest *as a value contract* but the SUGGEST's lifetime claim is factually wrong on the one builder where it matters most.

The eventual feature is admissible; its *reason to exist* and its *homing* must be re-grounded. The clean design is in §3.

## 3. Recommended design for 3a

The recommendation below is what 3a should be **if and only if** the §6 measurement clears the gate. It adopts the SUGGEST's directional structure (pure compute core + memo wrapper), corrects the over-build the CRITIQUE caught, and re-homes the cache to respect the lifetime facts.

### 3.1 Resolving the "free functions lying around" discomfort — decisively

The maintainer's instinct ("hoist into memoization-wrappers") is right. The discomfort ("free functions lying around in the source code") is misplaced about freeness but correct about *visibility*. The resolution:

**The two compute functions already exist and are already correctly shaped — keep them exactly where they are.** `belief_features(env, bw, N, nD, log_nworlds)` and `geometry_features(env, loc, N, nD, n_tel, diag)` live in the `features.cpp` anonymous namespace today (features.cpp:119–202), born of the already-landed step-1 decomposition, called from `build` at :210–211. They are pure value-functions returning by value, bounds-carrying (`std::span<const uint32_t>` / `const Point&`). **This is the functional core; a functional core *is* free pure functions** (P9). They are not clutter.

**Do NOT promote them to a public `feature_compute.hpp`.** This is where I overrule the SUGGEST and side with the CRITIQUE. The SUGGEST proposes moving them into a new public header to make them "named and unit-testable." That reopens the *already-landed* §2 decomposition and manufactures a **new compiled-component public contract** (every signature now a reviewed P8/P9 surface, a new TU, new `#include` edges from `gumbel.cpp`/`runner.cpp`) — a file-topology change neither the task nor the dossier asked for (dossier files §2 decomposition and §4.4 memo as *separate* avenues; §2 is done). CLAUDE.md scope discipline forbids expanding a memoization task into an API refactor without surfacing it. Keep them anonymous-namespace-private. If a unit test is wanted, a `feature_compute_testonly.hpp` exposing them to the parity TU *only* is the bounded move — but that is optional and out of 3a's scope.

So the honest answer to the discomfort is: **the free functions are the core, not a smell; leave them private and in place. The cache is a thin memo *method* that calls them. Nothing is "lying around" — a private pure function is the textbook functional core P9 endorses.**

### 3.2 Where the memo lives, and what type it is

Reject the SUGGEST's `template<class Compute> BeliefMemo::get(span, Compute&&)`. A single-instantiation generic with exactly one `compute` ever passed is preemptive abstraction — ADR-0003's "extract only when a second concrete instance exists," and the audit's cancer-E shape (an abstraction built ahead of its second user). The CRITIQUE is right here. The cache is **not** a reusable generic component; it is two private members and two private wrapper methods on `FeatureBuilder`, calling the two named-but-private compute functions directly:

```cpp
// in features.hpp, FeatureBuilder private section:
private:
  // ---- behaviour-preserving memos (derived data; bit-identical hit≡miss) ----
  mutable std::map<GBeliefKey,
                   std::vector<std::pair<std::vector<uint32_t>, BeliefFeatures>>> belief_cache_;
  mutable int belief_cache_n_ = 0;
  static constexpr int kBeliefCacheCap = 50000;   // mirrors features.py _belief_cache_cap
  mutable std::unordered_map<Point, GeometryFeatures, PointHash, PointEq> loc_cache_;

  // thin memo wrappers — call the private pure compute fns, store, return.
  const BeliefFeatures&   belief_feats_(std::span<const uint32_t> bw) const;
  const GeometryFeatures& geometry_feats_(const Point& loc) const;
```

`build()` (still `const`) calls `belief_feats_(bw)` / `geometry_feats_(loc)` instead of the bare compute fns — a two-line change at the call sites (features.cpp:210–211).

**`const` + `mutable`: kept, and honest.** `build()` stays `const`. The mutation is *logically const* — a hit returns bit-identical bytes to a miss (the §6 P6 guarantee), so the builder's observable value (vector-for-input) is invariant. This is `mutable`'s designed meaning (the std-lib blesses memoized-result + `mutable std::mutex`). Dropping `const` would be the *worse* lie — it would advertise "I mutate my logical value" to the per-task runtimes that legitimately hold `fb_` as a const collaborator. P9-rule-3's "can a reviewer name every mutated state from the call?" is satisfied: a reviewer sees `belief_feats_(bw)` and `geometry_feats_(loc)` and names the two caches.

**BUT the SUGGEST's concurrency defense is conditionally false and the design must enforce the condition (this is the upheld CRITIQUE (b)).** The `mutable`-cache-behind-`const` is data-race-safe *only* on a builder that is never shared across threads. That holds for the per-task runtime builders (`search_runtime.cpp` constructs a fresh policy+`fb_` per task). It does **not** hold structurally for serve's `st.fb`, which is process-scoped. Today serve's message loop is single-threaded so there is no *live* race — but `const`-on-a-shared-object is the canonical C++ "safe to share" advertisement, and the async actor-learner is a stated incoming multi-threaded body of code. **Therefore the design must, at minimum, document the single-thread-ownership precondition at the cache site and on `build()`'s contract, and the implementer must not silently rely on it surviving a future sharing.** The §6 remediation (share the root feature vector instead of double-building) is the structurally cleaner answer that sidesteps this entirely for the serve path.

### 3.3 Keying

**Belief: reuse `GBeliefKey` / `gumbel_belief_key` (gumbel.hpp:88–89). Do not re-author `(n, front, back)`.** Authoring a second belief fingerprint would itself be the P1/cancer-B violation. The belief cache and the node cache must read the *same* fingerprint authority. The map is **bucketed**: `GBeliefKey → vector<pair<vector<uint32_t>, BeliefFeatures>>`, and a hit walks the bucket testing **full `std::ranges::equal(stored_bw, bw)`** before returning — the exact mirror of Python's `np.array_equal` collision guard (features.py:329–331). Correctness rests *entirely* on the full-equality check; the fingerprint is only a pre-filter. The bucket must **own a copy** of `bw` (Python stored a reference safe under belief-immutability; in C++ a stored span would dangle). The copy is a per-*miss* allocation only — paid when we compute anyway, never on a hit.

**Loc: key on the discrete coordinate, not on raw `double` bits — and this is a correction to the SUGGEST.** `Point` is `{double x, double y}` with no operators (instance.hpp:28–31). The SUGGEST's proposed `std::bit_cast<uint64_t>` hash **does not compile** over a 16-byte `Point` — a concrete bug that reveals the loc-key facility was hand-waved. More importantly, the CRITIQUE's deeper point stands: keying on raw `double` bits makes correctness rest on the *unenforced prose* premise that `loc` is always a named coordinate (cancer-G). The clean key is the **named-coordinate identity the geometry is already indexed by** — every standing `loc` resolves from `env.coord` (a finite fixed set: treasure pts, face rep_points, teleport pts, entry). If a stable discrete id for that point is available or cheaply derivable, key on *it*; that is faster, immune to the float-key question, and aligns with §5's "matrix indexed by coordinate key." If no such id is exposed, the fallback is an exact `PointEq`/`PointHash` over **both** doubles' bit-patterns (`std::bit_cast<uint64_t>` on each of `x` and `y`, never on the whole struct; **never** an epsilon compare — epsilon would conflate distinct fixed coordinates), *plus* a `[[maybe_unused]]` assertion or comment documenting the named-coordinate premise. Exact-bits-of-fixed-data is correct because `unordered_map` resolves hash collisions by `operator==`, so two distinct points are never conflated; a computed `loc` would only cost hit-rate, not correctness.

### 3.4 Homing and lifecycle — re-grounded for the serve path

The SUGGEST's "member on the lifetime-shared builder = R9's literal intent, no reset needed" is right for the *per-task* builders and wrong for `st.fb`. Decision:

- **Per-task runtime builders:** member cache, no reset hook, the 50000 cap as pure memory bound. Correct and sufficient — the builder dies at task end, the cache with it.
- **The serve `st.fb`:** this builder lives for the whole process. The SUGGEST's claim that it "dies with the env" is the part the source refutes (serve.cpp:97 builds it once; nothing rebuilds it). Two acceptable dispositions, in order of preference:
  1. **Preferred — do not double-build at all (the real P1 fix the CRITIQUE surfaced).** The search root is already built by `policy->fb_` at gumbel.cpp:545; `run_episode` then *rebuilds the same root belief* with `st.fb` at runner.cpp:49. That is the single largest belief-recompute the serve path does, and a per-builder cache **cannot** fold it (two disjoint caches). The honest P1 move — the move 3b started — is to have the record path **consume the search's already-built root feature vector** rather than rebuild it. This likely captures more of the real duplication than the cross-leaf memo does, and it removes the process-scoped cache's growth and concurrency questions entirely. **This should be measured first and may obviate the serve-path belief cache.**
  2. **If a serve-path cache is still wanted after measurement:** give `st.fb` back the **per-episode reset** Python has (`reset_belief_cache`, features.py:295–300, driven per-episode by `GumbelAZSearch`). The SUGGEST dismissed this as "dead symmetry (ADR-0008)" — that is wrong: features.py:288–292 documents the reset as the live mechanism bounding the cache to *one episode's hundreds of beliefs* rather than letting a long-lived builder grow to the 50000 cap. Dropping it changes the serve builder's memory profile from O(hundreds) to O(50000) and turns the cap into a periodic cold-start recompute cliff on the hot path. Port `reset_belief_cache()` and call it per-episode in `run_episodes` for the serve builder.

So: a `clear()` method exists (mirrors `reset_belief_cache`), the cap is a memory backstop, and **the serve path either avoids the cache (preferred) or resets it per episode** — it does not silently run a never-reset process-lifetime cache.

## 4. 3b assessment

**Compliant as landed, with one documentary/contract follow-up.**

- **P1 — strengthened.** Before 3b, `marg` had three homes per leaf (build's inline sweep + two `env.legal_actions → marginals` recomputations in `evaluate`). 3b makes `legal_mask_from_features` consume build's `available`/`informative` blocks — collapsing to one home. This is the cancer-B remediation, the same move as `FeatureLayout`. The SUGGEST is right that this is a positive P1 instance. Confirmed against gumbel.cpp:241/257–266 and features.hpp:53–61.
- **P9 — clean.** `[[nodiscard]] std::vector<float> legal_mask_from_features(std::span<const float>) const`: bounds-carrying in, by-value out, mutates nothing, offsets from the `FeatureLayout` SSOT. Textbook pure slice. No `mutable`-cache question (3b adds no state).
- **P6 — right tier, verified.** The mask is a logic invariant, asserted bit-exact (illegal-slot mass `== 0.0`); "verified bit-exact (gumbel parity green)" is the correct substantiation at the correct tier. The legal-slot order (treasure ids, detector ids, TERMINATE) matches `env.legal_actions`, so the PUCT scan order is preserved (gumbel.cpp:257–266).
- **Keeping the free `legal_mask(env, bw, collected)` is NOT cancer-E.** It is the non-hot path (the per-step training-mask emission at runner.cpp:50 and the parity tool — neither has a built `feat` in hand) and the parity *oracle*. An oracle actively used to net the hot path is the opposite of an abandoned-abstraction-beside-a-live-copy. The SUGGEST's reading here survives the CRITIQUE's silence on it.

**The follow-up (upheld from CRITIQUE (g), severity downgraded from "violation" to "debt"):** `legal_mask_from_features(std::span<const float> feat)` takes a bare float span. Its real contract — "`feat` must be *this builder's* `build()` output, in *this* layout" — is carried by the header comment (features.hpp:56–60), not the type. A caller could pass any `dim()`-length float buffer; the signature cannot see the coupling to `build`. This is the call-boundary form of cancer-G (load-bearing knowledge in prose) / the P8 tension (the signature is not the SSOT of the contract). It is **mitigated** by both methods being members of the *same* `FeatureBuilder` (the coupling is object-scoped, not global), which is why this is a debt, not a violation. **Follow-up, both cheap:** (i) assert `feat.size() == static_cast<size_t>(dim_)` at the top of `legal_mask_from_features` (fail-loud per ADR-0002 if a mis-laid buffer ever arrives — this is the enforceable backstop P7 wants); (ii) one header line naming the `build()`↔`legal_mask_from_features` layout coupling and the `legal_mask`↔`legal_mask_from_features` oracle/hot-path relationship, so a future reader does not mistake the pair for E-style duplication. This is the ADR-0005 "documentation is part of the work" completion the SUGGEST itself gestured at but under-specified.

## 5. ADR-0012 conformance mapping

| Principle / cancer | How the recommended design honors it | Residual tension |
| --- | --- | --- |
| **P1 (SSOT)** | Memo *stores the output of the one author* (`belief_features`); contains no belief arithmetic, structurally cannot drift. Belief key *reuses* `gumbel_belief_key` (no second fingerprint). 3b collapsed marg to one home. | **Real, unresolved by 3a as scoped:** the serve path computes the *same root belief vector twice* in two builders (`policy->fb_` gumbel.cpp:545 vs `st.fb` runner.cpp:49). A per-builder memo cannot fold this. The P1 fix is to **share the root vector** (§4 disposition 1), not memoize two builders. SUGGEST's "categorically not a second SSOT" overstates by ignoring this. |
| **P2 (homing / lifetime-shared owner)** | Per-task builder: cache dies with the builder, value-stable belief/loc keys, no module global, no `id()` — R9's literal intent achieved by ownership. | **Real:** "dies with the builder which dies with the env" is **false for `st.fb`** (process-lifetime). Honored only after §4 disposition (avoid-or-reset). |
| **P3 (one axis per owner)** | `belief_feats_` / `geometry_feats_` each own one memo axis; compute fns each own one compute axis. No god-object. | Geometry homing is dossier-§5 **DEFERRED** (dossier:274). 3a must not resolve §5 by fiat; the member-on-builder home is acceptable *for the cache* but the distance-*matrix* home stays open. |
| **P4 (live-not-frozen)** | Belief/geometry features are functions of `(frozen-instance env, bw, loc)` only. HOT knobs (`m`,`n_sims`,`lam`,`max_steps`) and net version are **not featurizer inputs** (`lam` enters scoring at gumbel.cpp:341/361/398, never the featurizer). Instance change is a loud `instance_knob_changed` reject (serve.cpp:106–110). No reconfig can stale a value. | **Strong / survives** on the search-knob, net-version, and instance axes. **Residual** only on lifecycle: the dropped per-episode reset on `st.fb` is a deliberate lifecycle removal mislabeled "dead symmetry" — fixed in §4. |
| **P6 (bit-exactness)** | Hit returns the **stored bytes** unchanged — bit-identical to recompute (no arithmetic on the hit path). Belief collision guard is full `std::ranges::equal` (exact). Loc key is exact-bits / discrete id (never epsilon). Parity test must construct a forced fingerprint collision and assert per-belief correctness. | Survives. The loc-key correctness depends on the named-coordinate premise — enforce by discrete-id key (preferred) or assertion, not prose (cancer-G avoidance). |
| **P8 (signature is the contract)** | 3a memo wrappers are typed and named. | **3b debt:** `legal_mask_from_features(span<const float>)` erases the "must be my build output, same layout" contract into a comment. Fix: size assertion + header line (§4). |
| **P9 (functional core / honest signatures)** | Pure compute fns are the core (kept private, in place). Memo is a thin imperative shell; `build()` stays `const` (logical-const, honored because hit≡miss). Reject the templated generic memo (preemptive abstraction). | `const`+`mutable` is honest **as a value contract** but advertises thread-safety the process-scoped `st.fb` does not structurally guarantee. Document the single-thread-ownership precondition; the async-actor future owes the synchronization analysis *at the sharing site* if one is introduced. |
| **Cancer C (hidden state keyed by nothing)** | Member, not global; belief-*value* key + full-equality, not an address; no `id()`, no never-evicted cross-env container. | The *address* hazard is fully avoided. The *cross-episode unbounded accumulation* hazard is **live on `st.fb`** until §4 disposition. |
| **Cancer B (re-encoded fact)** | Memo holds no copy of the belief math; equality check is an identity test, not arithmetic. | None for the memo. The two-builder duplication (P1 row) is the B-*spirit* issue. |
| **Cancer E (abstraction beside a live copy)** | Reject the single-use `template Memo<Compute>` (would be premature E-shaped abstraction). Kept `legal_mask` is an *active oracle*, not abandoned. | None, once the generic is dropped. |
| **Cancer G (load-bearing prose)** | Loc-key premise and the 3b `build`→mask contract moved from prose to enforced form (discrete-id key / size assertion). | The named-coordinate premise needs the enforced key or an assertion to fully discharge. |
| **ADR-0009 / measure-first (P6, P9-rule-4)** | — | **FATAL as proposed:** the dossier gate for §4.4 is **HOLD / "measure belief recurrence before building"** (dossier:176). SUGGEST transferred a *Python* ~3.5× onto an unmeasured C++ artifact. **Gate not met.** Resolved only by §6's C++ measurement. |

## 6. Risks, staleness modes, and open questions — each flagged, none silently resolved

1. **[BLOCKING — ADR-0009] The justification gate is not met.** §4.4 is HOLD; the gate is a *C++* belief-recurrence measurement (dossier:176), not a Python number. **Open action:** before building 3a, instrument the serve path with a per-builder distinct-belief-builds vs total-builds count (`bench_hotpath`-style). The decisive quantity is the recurrence *incremental to the existing node cache* (`node.evaluated` + `(slot, GBeliefKey)` transposition, gumbel.cpp:340/362/399 — which already builds each `(slot, belief)` once per tree). I confirmed the node cache folds same-parent-slot repeats within a tree; what a builder cache additionally captures is cross-slot/cross-root-decision recurrence within an episode. That increment is **plausibly small and possibly near-1** — unproven either way, which is exactly why the gate governs. **Do not assume; measure.**

2. **[REAL — P1] The dominant recompute may be the two-builder duplication, not cross-leaf recurrence.** `policy->fb_` (search root) and `st.fb` (record) both build the same root belief (gumbel.cpp:545 / runner.cpp:49). A per-builder memo cannot fold across the two builders. The measurement in (1) should report this separately; if it dominates, the correct fix is **sharing the root feature vector** (the P1-honest continuation of 3b), which may make the serve-path belief cache unnecessary. Open design question: does the runner already have access to the search's root `feat`, or does the seam need a small extension to return it from `decide_target`?

3. **[REAL — P2/P9] Process-scoped `st.fb` + `const` cache + future threads.** Single-threaded today (serve loop), but the `const`-on-shared-object advertises thread-safety the async actor-learner will violate. **Open action:** document the single-thread precondition at the cache site; if 3a caches on `st.fb` at all, prefer §4 disposition 1 (no cache there) so the precondition is moot.

4. **[REAL — lifecycle] No per-episode reset on the long-lived serve builder.** Python resets per episode (features.py:288–300); the C++ port as proposed omits it, turning the 50000 cap into the only bound on a process-lifetime cache — an O(50000) memory profile and a periodic cold-start recompute cliff. **Resolved by §4** (avoid the cache or port `reset_belief_cache` + call per episode in `run_episodes`).

5. **[CORRECTNESS-SAFE, HIT-RATE risk — P6/cancer-G] Loc float key.** Correctness holds even under a computed `loc` (`unordered_map` falls through to exact `==`); only hit-rate degrades. But the SUGGEST's concrete `std::bit_cast<uint64_t>` over a 16-byte `Point` **does not compile**. **Resolved by §3.3:** key on a discrete coordinate id (preferred) or exact bits of *both* doubles with a documented/asserted named-coordinate premise; never epsilon.

6. **[GUARD — P6/P7] Belief fingerprint collisions are real.** `_belief_key` collisions (equal-size beliefs sharing min/max world ids) are documented; the full-`bw`-equality guard is load-bearing and must be exact and complete. **Open action:** the parity harness must *construct* a forced collision and assert each belief gets its own features (net the guard, don't trust it — ADR-0011).

7. **[DEBT — P8/cancer-G] 3b's `build`→mask coupling.** Carried by comment. **Resolved by §4:** size assertion + one header line. Low effort, do it regardless of 3a.

8. **[SCOPE] Do not reopen §2 or §5.** Promoting the compute fns to a public header (reopening the landed §2 decomposition) and homing the distance *matrix* (the DEFERRED §5 question) are both out of 3a's scope. Keep the compute fns private; the cache-member home is acceptable for the *cache* but does not settle §5's matrix home.

## 7. Implementer's summary (checklist)

**Before any 3a code (BLOCKING):**
- [ ] Measure C++ belief-feature recurrence in the **serve path, per builder**: distinct-belief builds vs total builds, *incremental to the existing node cache*. Report `policy->fb_` and `st.fb` separately, and report the two-builder root duplication as its own line. (ADR-0009; the dossier §4.4 HOLD gate.)
- [ ] If the two-builder duplication dominates, prototype **sharing the search-root feature vector** with the record path (the P1-honest fix) and compare against the cache before building the cache.

**3a, if the gate clears:**
- [ ] Keep `belief_features` / `geometry_features` **private in the anonymous namespace, in place** (functional core). Do **not** create a public `feature_compute.hpp`.
- [ ] Add two private memo wrappers `belief_feats_(span<const uint32_t>) const` / `geometry_feats_(const Point&) const` on `FeatureBuilder`; `build()` stays `const` and calls them. **No templated generic memo** — call the compute fns directly.
- [ ] Belief cache: **reuse `GBeliefKey` / `gumbel_belief_key`** (no second fingerprint). Bucketed map; full `std::ranges::equal` collision guard; **own a copy of `bw`** in the bucket (miss-path only).
- [ ] Loc cache: key on a **discrete coordinate id** if available; else exact bits of *both* `x` and `y` (`bit_cast` each `double` separately — never the whole `Point`), with a documented/asserted named-coordinate premise. Never epsilon.
- [ ] Cap mirrors `_belief_cache_cap = 50000` with a `clear()` method (mirrors `reset_belief_cache`).
- [ ] **Serve path:** prefer *not* caching on `st.fb` (share the root vector instead); if you do cache there, call `clear()` per episode in `run_episodes`. Document the single-thread-ownership precondition at the cache site and on `build()`.
- [ ] Parity: assert cached `build()` output is byte-identical (`== 0.0` delta) to uncached over a hit-exercising trajectory, **including a forced `GBeliefKey` collision** that proves each belief gets its own features.

**3b follow-up (do regardless):**
- [ ] Add `assert(feat.size() == static_cast<size_t>(dim_))` at the top of `legal_mask_from_features` (fail-loud, ADR-0002).
- [ ] Add one header line naming the `build()`→`legal_mask_from_features` layout coupling and the `legal_mask`↔`legal_mask_from_features` oracle/hot-path relationship (ADR-0005).

**Documentation:** 3a touches no orientation surface in `STATUS.md`/the handoff beyond a commit-log entry; record the measurement result (the gate evidence) in the commit message per ADR-0009. The `legal_mask` header line is the ADR-0005 completion 3b already owed.

---

## Appendix A — Phase 1: SUGGEST (proposing architect, raw)

This is the decisive fact for the P4 staleness analysis. In the **serve** (online-reconfig) path:

- The `env` and `fb` (FeatureBuilder) are built **ONCE** on first `configure`, on the heap via `unique_ptr`, and **a later instance/faces change is a loud `instance_knob_changed` reject** — never a mutation. The env is immutable for the process lifetime.
- The `policy` (which in the gumbel path holds its own `fb_` member built from the same env) is **rebuilt on a net reload or a HOT-config change** — but the env it's built from never changes.
- The net version reload swaps `st.net` but **does not touch the env or fb**.

So a FeatureBuilder's belief cache and loc cache depend only on the env (geometry, cover masks, worlds) and the belief `bw` / loc — none of which the net version or the HOT knobs (`m`, `n_sims`, `lam`, `max_steps`) can perturb. I now have everything I need. Let me write the analysis.

I have now read all required materials end to end: the ADR synopsis, ADR-0012 in full (all 1151 lines across two pages), the dossier in full, features.hpp, features.cpp, gumbel.cpp's evaluate + node-cache machinery (GBeliefKey, node.evaluated, the (slot, GBeliefKey) child transposition), and features.py (FeatureLayout, _belief_feats/_belief_cache, _loc_block/_loc_cache, reset_belief_cache, _LAYOUTS). I also verified the C++ lifetime facts (Point has no operators; the serve-path env/fb are built once on the heap and an instance change is a loud reject; the policy/fb is per-task in the runtimes).

---

# 3a (belief/loc cache) — the proposal; and 3b — retroactive compliance

## Preamble: the one fact that reframes the whole review

The maintainer's concerns are framed against the Python idiom, where `FeatureBuilder` is a single long-lived object the caller resets per episode. The C++ lifetime is **different in a way that resolves most of the hazard before we even key the cache**:

- In the **runtimes** (`search_runtime.cpp` `SerialRuntime::run` / `PoolRuntime::run`), `GumbelAZPolicy policy(task.cfg, counter, env)` — and therefore its `FeatureBuilder fb_` member — is constructed **fresh per task** and destroyed at task end. There is no cross-thread sharing of a builder; each worker thread owns its own policy+builder for the duration of one tree.
- In the **serve / online-reconfig** path (`serve.cpp`), `env` and `fb` are built **once**, on the heap, and held by `unique_ptr` for the process's life; **a later instance/faces change is a loud `instance_knob_changed` reject, never a mutation**; the `policy` (which holds its own `fb_`) is rebuilt on a net reload or HOT-config change, but always from the *same, immutable* env.

So a `FeatureBuilder`'s cache lifetime is **bounded by one of two owners, both lifetime-shared with the env and both immutable in `env` for the duration**: either a per-task policy (runtime) or a per-process held context whose env never changes (serve). This is exactly the R9 *intent* — derived data tied to a value-stable, lifetime-shared owner — achieved here by ownership structure rather than needing a WeakKeyDictionary, because C++ gives us deterministic lifetime that Python's GC did not.

With that established, the eight points:

---

## (a) HOMING — where the cache lives, lifetime ownership, why it is NOT cancer-C

**Home: the cache lives as mutable members on the `FeatureBuilder` instance itself** — `mutable BeliefCache belief_cache_;` and `mutable LocCache loc_cache_;` — exactly as the Python `_belief_cache` / `_loc_cache` live on the Python `FeatureBuilder`. This is the direct structural port and it is the *correct* home, for the following precise reasons.

**Why this is P2-compliant and specifically NOT cancer-C.** Cancer C is "hidden global state keyed by object identity" — the `_SLOT_TABLES[id(env)]` shape: a *module global* keyed by a *value-less identity* (a CPython address) that (1) leaks one never-evicted entry per env and (2) hands back the wrong value at a reused address. The proposed cache fails every clause of that indictment:

1. **It is not a module global.** It is an instance member of `FeatureBuilder`. There is no file-static, no thread-local, no anonymous-namespace mutable. The dossier (§4.1) is explicit that a module global for scratch is "cancer C (hidden state keyed by nothing)"; we do not author one. (This is also why the pure-compute functions in step-1 must stay *stateless* free functions — they hold no cache; the cache is the member, see (c).)

2. **The owner's lifetime is shared with the source.** The `FeatureBuilder` is constructed `fb_(env)` from the env and references it (`const Environment& env_`). Its cache is derived data whose source (`env_`'s geometry, cover masks, `worlds()`) shares the builder's lifetime exactly. This is P2's literal rule: "derived state owned on the object whose lifetime it shares." The builder *is* that object — it is the C++ analogue of "the env attribute" R9 wanted, without R9's import-cycle problem (there is no features→env→features cycle in C++; the builder already holds `env_`).

3. **The key is value-stable, not an address.** The whole address-reuse hazard of cancer C is that `id(env)` is a CPython address that can be recycled. Our keys (see (b)) are a value fingerprint `(n, bw.front(), bw.back())` plus a *full bw-equality* check on a hit, and the loc `Point(x,y)` by value. No address is keyed; a reused address cannot alias because no address is in the key. The R9 remediation re-keyed *from* `id()` *to* a value-stable owner; we are born on the value-stable side.

4. **It cannot leak across envs.** The pre-R9 `id(env)` dict leaked because it was a *global* outliving every env. Our cache is a *member* — it dies with the builder, which dies with (or before) the env. In the runtimes it dies at task end; in serve it dies when the policy is rebuilt or the process exits. There is no never-evicted accumulation across envs because there is no cross-env container at all.

**The one structural subtlety to name honestly (P2(c) under threading).** Because `PoolRuntime` constructs a *fresh* `FeatureBuilder` per task on each worker thread, **the belief cache is per-thread, per-task by construction** — there is no shared-mutable-cache-across-threads question to answer, and therefore no lock and no data race. This is the right answer, not an accident: it means the cache's hit-rate is scoped to one tree's belief recurrence (which is exactly where the dossier's ~3.5× incremental recurrence lives — see (e)), and it means the `mutable` members are touched only by the single thread that owns the builder. I flag this explicitly because a critic's first move will be "a mutable cache + PoolRuntime = data race"; the answer is that the builder is never shared across threads, so the mutation is single-threaded by ownership. (If a future restructure ever shared one builder across worker threads, *that* change — not this one — would owe the synchronization analysis; P2's check fires at the sharing site, and there is no sharing site today.)

---

## (b) KEYING — the fingerprint + full-equality check; the Point key; value-stability

### Belief cache key: fingerprint + full bw-equality on a hit

The key mirrors `_belief_key` exactly: `GBeliefKey = std::tuple<int, uint32_t, uint32_t>` = `(n, bw.front(), bw.back())` — the count, first-world-id, last-world-id fingerprint. This type *already exists in the tree* (`gumbel.hpp:88`, `gumbel_belief_key`), used as the child-transposition key in the node cache. **Reuse it; do not re-author a second belief-fingerprint** — re-authoring `(n, front, back)` a second time would itself be a P1/cancer-B violation (a second home for the belief-identity fact). The belief cache and the node cache must read the same `gumbel_belief_key(bw)`.

The fingerprint is **collision-resistant, not collision-free** — distinct equal-size beliefs can share min/max world ids (Python's docstring documents exactly this). So the cache is a **bucketed map**: `key -> std::vector<std::pair<std::vector<uint32_t>, BeliefFeatures>>`, and a hit walks the bucket testing **full `bw` equality** (`std::ranges::equal(stored_bw, bw)`) before returning. This is the precise mirror of Python's `for bw_ref, feats in bucket: if bw_ref is bw or np.array_equal(bw_ref, bw)`. The fingerprint is a fast pre-filter; *correctness rests entirely on the full-equality check*, never on the fingerprint.

**Why value-stable.** Beliefs in the model are immutable — every `filter_*` returns a fresh array, never an in-place edit (ADR-0001's first seam; mirrored in the C++ env's filters which produce fresh `bw` vectors). So a stored `bw` is a frozen value; its fingerprint never changes under it, and the equality check compares two immutable value-sequences. The key is a *function of the belief's value*, not of any address or mutable state — the antithesis of cancer C.

**One C++-specific honesty note the critic will probe:** storing the `bw` for the equality check means the bucket owns a `std::vector<uint32_t>` copy (or the `bw` is moved in on the miss path). Python stored a *reference* and relied on belief immutability ("a reference is safe"). In C++ a stored reference/`span` would dangle once the caller's `bw` goes out of scope; we must **own a copy** in the bucket. This is a real per-miss allocation (N≤32 fits `uint32_t`, beliefs are ≤15,504 worlds), and it is the honest cost of the equality check — it is paid only on a *miss* (when we compute anyway), never on a hit. I would not elide it; the equality check is the collision guard and the guard must compare against a value that outlives the call.

### Loc cache key: `Point(x, y)` by value

The Python `_loc_cache` is keyed by `Loc` (a coordinate tuple), value-hashable. The C++ `Point` is `{double x, double y}` with **no `operator==`, no `operator<`, no `std::hash`** (I checked `instance.hpp`). So the port must *add* a key facility. Two honest options:

- A transparent comparator / hash on the exact `(x, y)` bit-pattern, used in an `unordered_map<Point, GeometryFeatures, PointHash, PointEq>` (or a `std::map` with a `Point` `operator<`).

**Why value-stable, and the P6 caveat (deferred to (g)).** The coordinates are fixed instance geometry — `env.coord` values, never mutated. A `Loc` in this env is one of a *finite, fixed set of named coordinate points* (treasure pts, face rep_points, teleport pts), so the key space is small and the `x,y` are literally the same `double` bit-patterns every time the agent stands at that named point — distance is `std::hypot(Point, Point)` over those fixed inputs. The equality is therefore over **identical bit-patterns of fixed data**, not over a computed float that could vary. This is what makes the exact `==` on a `double` key *safe here specifically* (the general "never `==` a float" rule is about *computed* floats; these are *stored fixed* floats). I make this argument fully in (g) because it is the load-bearing one a critic attacks.

---

## (c) SHAPE — the concrete C++ structure that resolves the free-function discomfort

The maintainer's instinct ("hoist the calculations into memoization-wrappers") is **right**; the discomfort ("free functions lying around in the source code") is **misplaced about the wrong thing** — and naming why is the resolution.

### The free compute functions are already correct and already in the tree — keep them, but make them *real* pure functions

`belief_features(env, bw, N, nD, log_nworlds)` and `geometry_features(env, loc, N, nD, n_tel, diag)` already exist in `features.cpp`'s anonymous namespace, born of step-1 decomposition. They are **pure value-functions returning `BeliefFeatures`/`GeometryFeatures` by value** (P9-rule-2), taking `std::span<const uint32_t>` / `const Point&` (P9-rule-1, bounds-carrying). The dossier (§2) endorses exactly this decomposition on **hygiene grounds alone (P3 one-axis-per-function, P9), independent of any perf argument**. They are not "lying around" — they are the *functional core*, and a functional core *is* a set of free pure functions. P9's entire thesis is "functional core, imperative shell": the core's purest form is free functions you can unit-test and compose. So:

**The discomfort is not "free functions exist"; it is "free functions in an anonymous namespace are invisible to a unit test and read as orphaned."** Resolve that, not the freeness:

1. **Promote the two pure-compute functions to a named, testable seam** — a small free-function header, not the anonymous namespace. Concretely a `belief_compute.hpp` / `geometry_compute.hpp` (or one `feature_compute.hpp`) declaring:
   ```cpp
   [[nodiscard]] BeliefFeatures belief_features(const Environment&, std::span<const uint32_t> bw,
                                                int N, int nD, double log_nworlds);
   [[nodiscard]] GeometryFeatures geometry_features(const Environment&, const Point& loc,
                                                    int N, int nD, int n_tel, double diag);
   ```
   These become **nameable in one clause** ("compute belief intermediates from `bw`"; "compute the distance block from `loc`"), unit-testable in isolation (the dossier's P6 case-c bit-exact deltas can be asserted directly on them), and *visibly the core* rather than orphans. This is the honest fix for "lying around": a pure function is not a smell — an *un-named, un-testable, anonymous* pure function reads as one, so give it a name and a test surface.

2. **The cache is a memo *wrapper method* on `FeatureBuilder`, not a third free function and not inlined into `build`.** Add two private methods:
   ```cpp
   const BeliefFeatures&   belief_feats_(std::span<const uint32_t> bw) const;   // memoizes belief_features(...)
   const GeometryFeatures& geometry_feats_(const Point& loc) const;            // memoizes geometry_features(...)
   ```
   Each method: hash/fingerprint → bucket walk / map lookup → **on a hit, return the cached value; on a miss, call the pure free function, store, return**. `build()` then calls `belief_feats_(bw)` and `geometry_feats_(loc)` instead of the bare free functions — *one-line* change at the call sites in `build`. The pure core stays pure; the memo is a thin imperative shell method that *names its mutation in its home* (`this`, via the `mutable` members — see (d)).

### Should the caches be a dedicated typed `Cache` component instead of raw members?

**Yes — this is the cleanest answer and it directly dissolves the remaining discomfort.** Rather than scatter `mutable std::map<...> belief_cache_; mutable int belief_cache_n_; int belief_cache_cap_; mutable std::unordered_map<Point,...> loc_cache_;` as four loose members, introduce two small **one-owner typed components** (P3):

```cpp
// One axis: "memoize belief intermediates by belief value, collision-checked." Nameable in one clause.
class BeliefMemo {
  public:
    explicit BeliefMemo(int cap = 50000) : cap_(cap) {}
    // Returns the cached features for `bw`, computing+storing via `compute` on a miss.
    // `compute` is the pure core (belief_features); the memo never re-implements the math.
    template <class Compute>
    const BeliefFeatures& get(std::span<const uint32_t> bw, Compute&& compute);
    void clear() noexcept;                 // mirrors reset_belief_cache (memory-only, never correctness)
  private:
    std::map<GBeliefKey, std::vector<std::pair<std::vector<uint32_t>, BeliefFeatures>>> buckets_;
    int n_ = 0;
    int cap_;
};

class LocMemo {                            // one axis: "memoize the distance block by standing point."
  public:
    template <class Compute>
    const GeometryFeatures& get(const Point& loc, Compute&& compute);
  private:
    std::unordered_map<Point, GeometryFeatures, PointHash, PointEq> by_loc_;
};
```

and on the builder:
```cpp
mutable BeliefMemo belief_memo_;
mutable LocMemo    loc_memo_;
```

This is strictly better than raw members on three ADR-0012 counts:

- **P3 (one owner):** `BeliefMemo` owns *exactly* "belief-keyed memoization with collision verification and a memory cap"; `LocMemo` owns "loc-keyed distance memoization." Each is nameable in one clause; the cap, the bucket structure, the `n_` counter, the `clear()` are all *inside* the component, not leaking into the builder's surface. The builder's responsibility stays "assemble the §2.2 vector," undiluted.
- **P9 (the memo is the *only* mutation, and it is named):** the `get(bw, compute)` signature *names* its job — it takes the pure compute as a callable and memoizes it. The component cannot re-author the belief math (it has no access to it — it receives `compute`), so it is structurally incapable of becoming a *second home for marg* (see (e)). The mutation is `this` (the memo's own buckets), declared by the non-const method on a `mutable` member.
- **It makes the cap honest and local.** Python's `_belief_cache_cap = 50000` with a whole-cache clear lives *inside* `BeliefMemo` where it belongs, not as a stray builder field.

**Recommended shape, in one sentence:** *pure free compute functions in a named, testable header (the functional core); two small typed `BeliefMemo`/`LocMemo` components (one-owner memo, P3) held as `mutable` members of `FeatureBuilder`; thin `belief_feats_/geometry_feats_` wrapper methods (or direct `memo_.get(bw, [&]{ return belief_features(...); })` calls) inside `build()`.* The free functions are *kept* and *named* (they are the core, not clutter); the memo is a component, not loose state; `build()` stays `const` and reads through the memo.

---

## (d) P9 — `build()` const + a mutable cache: make the mutation HONEST

P9-rule-3 says a function mutates only what its signature names. `build()` is `const`, yet a memo mutates the builder. Is that a *lying* const?

**No — and the framing that makes it honest is the dossier's own Workspace / named-state framing, applied at the right granularity.** The argument:

1. **The mutation is behaviour-preserving and invisible at the value boundary.** A cache hit returns *bit-identical* values to a miss (this is the P6 guarantee, (g)). `build()`'s observable contract — "(loc, bw, collected) → the §2.2 vector by value" — is *unchanged*: it is still a pure function *of its inputs* at the value level. The `mutable` members carry no semantic state that any caller can observe; they are a pure *acceleration* structure. This is precisely C++'s designed meaning of `mutable`: "logically const, physically caches." Const-ness in C++ is **logical const**, and the standard library itself blesses this (a `mutable std::mutex`, a memoized result). So `build() const` with a `mutable` memo is *not* a lying signature in the P8/P2 sense — the lie P2 forbids is "a parameter the receiver cannot honor" / "an annotation the body breaks"; here the annotation (`const` = "I do not change my logical value") **is honored**: the builder's logical value (what vector it produces for any input) is identical before and after.

2. **But P9 demands the mutation be *named*, not merely blessed by `mutable`.** This is where the dossier's Workspace framing earns its keep — *with one sharpening*. The dossier (§4.1) reserves `Workspace&` for **measured hot-path *scratch*** (transient buffers reused per call), and §5 is explicit that a `Workspace` is for *mutable* scratch while read-only shared state crosses as `const&`. A memo is **neither**: it is *persistent* derived state, not per-call scratch, and it is *written* (a new entry on a miss), not read-only. So the honest C++ form is **not** to thread a `Workspace&` parameter through `build()` (that would misclassify a persistent cache as transient scratch, and would force the cache into the *caller's* ownership, which is wrong — the cache belongs to the builder's lifetime, (a)). The honest form is the **named typed component** of (c): the mutation is named by *the member's type* (`mutable BeliefMemo belief_memo_`) and *the method's name* (`belief_memo_.get(...)` says "memoized get"). A reviewer reading `build()` sees `belief_memo_.get(bw, …)` and `loc_memo_.get(loc, …)` and can name, from the call alone, the single mutated state (the two memos) — satisfying P9-rule-3's check ("can a reviewer name, from the signature/call, every piece of state mutated?") *better* than a hidden inline `static` ever could.

**So the precise honest framing is:** `build()` stays `const` because it is *logically* const (value-pure in its inputs, P6-bit-exact across hit/miss); the mutation is honest because it is (i) confined to two `mutable`, *named, typed* memo components owned by the builder, (ii) behaviour-preserving by the P6 contract, and (iii) *not* misrepresented as Workspace scratch — it is correctly typed as persistent derived state on the lifetime-shared owner. The Workspace framing is the right *vocabulary* ("the only sanctioned hidden mutation is a typed, named one") but the *wrong specific mechanism* here (Workspace = transient scratch ⊥ memo = persistent derived state); the named-component-on-the-owner is the memo's correct typed form.

I would additionally **mark the memo methods `const`-correct internally and keep `build` `const`** rather than dropping `const` from `build` — dropping `const` would be the *worse* lie, because it would advertise "I mutate my logical value" to every caller (including the per-task runtimes that legitimately treat `fb_` as a const collaborator) when in fact the logical value is invariant. Keeping `const` + `mutable` is the *more* honest signature, not a loophole.

---

## (e) P1 — is the cache a second SSOT for `marg`?

**No. It is a memo *of* the one computation, not a second *home* of the fact.** This is the sharpest point and it must be argued exactly, because it is the maintainer's lead concern.

The P1/cancer-B violation is **"the same knowledge re-encoded in N places"** — a *second hand-authored copy of a fact* that *must agree* with the first and *can drift* from it. The test (P1, verbatim): "grep the tree for the value; if it appears as an independent literal in two places that must agree, the rule is violated." The decisive question is therefore: **does the cache *re-author* the belief math, or does it *store the output of the one author*?**

- The belief math has **exactly one author**: the pure `belief_features(env, bw, …)` free function (the functional core). The marginal sweep `for w in bw: for t: marg[t] += bit` lives there and *only* there.
- The `BeliefMemo` receives `compute` as a callable and **stores its result**. It contains *no* belief arithmetic — no popcount, no cover reduction, no normalization. It is structurally incapable of drifting from the author because it does not *contain* a copy of the author to drift. A change to the belief math changes the one free function; the memo automatically stores the new values; *there is no second site to update in lockstep*. Grep for the marginal recurrence and you find it once.

This is the *same* relationship the in-tree precedents already establish and ADR-0012 already blesses:
- **`FeatureLayout`** (P1's worked proof) is "the SSOT made structural" — and `feature_layout()` / `_LAYOUTS` *memoize it per env*. The memo is not a second layout SSOT; it is a cached read of the one SSOT. The belief memo is the identical move for the belief-intermediates fact.
- **3b itself** (just landed) is the *positive* instance: `legal_mask_from_features` *consumes* `build`'s output instead of recomputing marg — collapsing marg's three homes to one. The 3a cache extends exactly this: it ensures that even the *one* home is *evaluated* once per distinct belief, not re-evaluated per leaf.

**The honest boundary I will state for the critic:** P1 *would* be violated if the memo re-implemented the marginal sweep "for speed" (a hand-inlined fast path beside the free function) — *that* would be cancer-B (two writers, the audit's facemodel.SenseAction / split-brain-encoder shape, E). The design forbids this: the memo's only contract is "call the one `compute` and store it." The collision-equality check (`std::ranges::equal`) is *not* belief math — it is an identity test on stored values, no second home of `marg`. So: **memo-of-the-one-computation, categorically not a second SSOT.**

---

## (f) P4 staleness — could a HOT reconfig / net reload / instance change make the cache stale?

**The belief features depend on exactly: `env`'s cover masks (`observe`), `env.worlds()` size (via `log_nworlds`), `N`/`nD`, and the belief value `bw`. The geometry features depend on exactly: `env`'s fixed coordinates (treasure/face/teleport pts, `exit_cost`), `diag`, and `loc`. They depend on *nothing else* — and crucially on *none of the things a reconfig changes*.** Walking the three reconfig vectors against `serve.cpp`'s actual gating:

1. **HOT reconfig (`m`, `n_sims`, `lam`, `max_steps`, the GumbelConfig knobs).** None of these enters `belief_features` or `geometry_features` — they are search-budget scalars (P4 live cells crossing the wire as numbers), consumed by the *search*, not the *featurizer*. `lam` scales rewards in `descend`/`simulate_root_action`, never the feature vector. A HOT-config change in serve **rebuilds the policy** (and thus a fresh `FeatureBuilder` with a fresh, empty cache) — but even if it *didn't*, the cached belief/geometry values would be **identical**, because they are not functions of any HOT knob. **No staleness.**

2. **Net reload (version advance).** `serve.cpp` reloads `st.net` on a version change and rebuilds the policy against the new net — but **does not touch `st.env` or `st.fb`**. The belief/geometry features are functions of the *belief and geometry*, not the *net weights*. The net consumes the feature vector; it does not parameterize it. A new net reads the *same* features for the same `(loc, bw)`. **No staleness** — and note the policy rebuild gives a fresh cache anyway, so even the per-process `fb`'s cache (if it were the one used) carries no cross-version contamination; but the point stands that there is no fact to contaminate.

3. **Instance / faces change.** This is the *only* input that *would* change `belief_features` (cover masks, worlds) or `geometry_features` (coordinates). And `serve.cpp` makes it a **loud `instance_knob_changed` reject** — the env is built *once* and an instance change is refused, never silently adopted (ADR-0002 / P4's RESTART-with-loud-drift-refusal facet, applied at the INSTANCE tier). So within a single env's life — the only life the cache shares — the instance is invariant by construction. A new instance is a new process / a new env / a new builder / a new (empty) cache. **No staleness possible**: the one input that could stale the cache cannot change under a live env.

**This is why Python's `reset_belief_cache` is memory-only, not correctness — and why the C++ port correctly omits a per-episode reset hook.** Python's own docstring says clearing "is always correctness-safe (a hit only ever returns equal-belief features)"; the cache returns features of a belief that compared *equal*, so a stale entry is impossible *by the equality check* — there is no belief whose features changed under a fixed env. The cap (clear-whole-cache past 50000) is therefore purely a memory bound, and clearing at any time is safe. The C++ `BeliefMemo::clear()` mirrors this: callable anytime, never required for correctness. **The cap is the right and sufficient hygiene; a per-episode reset hook would be dead symmetry** (ADR-0008: don't fabricate a lifecycle the correctness does not need) — though I'd note one could expose `clear()` for a caller that wants tighter memory, exactly as Python exposes `reset_belief_cache`.

**The P4 summary the critic should not be able to dent:** the cache's inputs are `(env-as-frozen-instance, bw, loc)`; of the reconfig vectors, two (HOT knobs, net version) are *not inputs to the featurizer at all*, and the third (instance) is *a loud reject under a live env*. The belief feat depends on the belief and the env's fixed geometry — nothing that breathes within an env's life. There is no live-not-frozen violation because there is nothing live to freeze: the cached fact is a function of immutable inputs only.

---

## (g) P6 bit-exactness — what guarantees it; what could break it

**The guarantee.** A cache hit must return values **bit-identical** to recomputing — this is the strongest P6 tier (case (c): bit-identity where free and proven), and it *is* free and proven here because the cache *stores the exact bytes the pure function produced* and returns them unchanged. No arithmetic happens on the hit path; there is no reordering, no re-association, no float drift. The dossier's parity table (§7) lists memoization (4.4) explicitly as **"bit-exact-provable"** for exactly this reason: "cached values are identical." The belief intermediates are exact integer counts (`marg_raw`, `cnt`) followed by a single deterministic `1/nb` divide — order-independent, so even a *recompute* is byte-identical, and a *cache hit* trivially is. So the equivalence claim 3a must carry (P6/ADR-0009) is the **bit-exact tier, asserted `== 0.0` on the delta** in the parity harness — not the behavioral tier. This is the cheapest possible thing to verify and the design preserves it.

**What could break it — two real hazards, both controllable:**

1. **The float `Point` key (the loc cache).** Keying a cache on a `double` is the classic "never `==` a float" trap *if the key is a computed float*. Here it is **not** — and the distinction is the whole safety argument: the `Point(x,y)` values are *fixed instance geometry* read straight from `env.coord` (treasure/face/teleport coordinates, and the agent's standing point, which is always one of those named points). They are the *same stored bit-patterns* every time, never the result of an arithmetic that could land on a different rounding. So `x == x'` is comparing identical bytes of fixed data, and a `std::hash` over the bit-pattern is stable. **The hazard would materialize only if `loc` ever became a *computed* point** (an interpolated position, a midpoint) — then two "equal" locations could differ in the last bit and either miss (harmless: recompute, still correct) or, worse, a near-equal-but-different loc could *collide in the hash and be treated equal* — but `unordered_map` resolves hash collisions by `operator==`, so a hash collision between two *different* `Point`s falls through to the equality check and is *not* conflated; only a true bit-equal `Point` hits. So even under computed locs the *correctness* holds (equality is exact); only the *hit rate* would suffer. **The honest mitigation:** assert/document that `loc` is always a named coordinate (it is — `Loc{Point}` is resolved from `env.coord`, per `env.hpp`'s comment "the resolved Point, as in env.coord"), so the key is over fixed data; and key on the *exact* bit-pattern (a transparent `PointEq` doing `a.x==b.x && a.y==b.y`, a `PointHash` over `std::bit_cast<uint64_t>`), never an epsilon-tolerant compare (an epsilon compare would be the bug — it could conflate two distinct fixed coordinates that happen to be close). Exact-bits-of-fixed-data is correct; epsilon is not.

2. **The collision bucket (the belief cache).** The fingerprint `(n, front, back)` is *not* unique; the correctness rests on the **full `bw`-equality** check inside the bucket. If that check were dropped or weakened (e.g. trusting the fingerprint, or comparing only a prefix), a fingerprint collision would return **another belief's features** — a silent wrong-number, the worst P6 failure (the audit's `max|Δp|` stale-feature shape). The guard is therefore **load-bearing and must be exact**: `std::ranges::equal(stored_bw, bw)` over the full vectors, mirroring `np.array_equal`. This is bit-exact identity on the belief value, not a numeric tolerance. As long as that check is present and full, a hit is *provably* the same belief and therefore *provably* bit-identical features. I would add a parity test that *constructs* a fingerprint collision (two equal-size beliefs sharing front/back) and asserts the cache returns each its own features — to net the guard, not trust it (P7/ADR-0011: a runtime backstop is a backstop, but the guard is the contract).

**Net P6 posture:** the cache is on the *bit-exact* side of the two-tier bar; the parity harness asserts the cached `build()` output is byte-identical to the uncached `build()` output (`== 0.0` delta over a belief/loc trajectory that exercises hits and a forced collision). The float-sensitive concern that P6 usually raises (float32 non-associativity flipping a near-tied argmax) **does not arise** because the cache changes *where the computation happens*, not *how* — the bytes are identical, so no argmax can flip.

---

## (h) PRECEDENT — consistency with `_LAYOUTS` / `_SLOT_TABLES` / the R9 WeakKeyDictionary

The design is the **faithful C++ continuation** of the in-tree homing precedent, with one principled divergence the C++ lifetime model *earns*:

- **`_LAYOUTS` (features.py) and `actions._SLOT_TABLES`** are the live R9 instances: derived-data caches re-keyed from `id(env)` to a **WeakKeyDictionary keyed on the env *object*** — lifetime-shared owner, value-stable key, GC-evicted, no address-reuse alias. The 3a cache replicates the *intent* (R9's stated goal): "key on the belief value, tie lifetime to its owner" (the dossier's §4.4 homing note cites R9 by name as the precedent to "replicate").

- **The principled divergence, named per ADR-0012's own deviation-recording posture (P2's R9 note does exactly this):** R9 used a WeakKeyDictionary *because Python forced a choice* — an env attribute would create a features→env→features import cycle, and a module-global `id(env)` dict leaked/aliased. **C++ has neither problem.** The builder *already* holds `env_` by reference (no cycle — C++ headers don't import-cycle the way the Python modules did), and the cache lives as a *member of the builder* (no module global — so no leak, no address-reuse, nothing to weak-reference). So the C++ design lands the cache *on the lifetime-shared owner directly* — which is the home R9's audit *literally wanted* ("the `env.slot_tables` attribute") and which R9 could only *approximate* with a WeakKeyDictionary because of the cycle. **The C++ port is thus *closer* to R9's literal intent than R9's own Python remediation could be**, and it is closer for a *named, structural* reason (deterministic C++ lifetime + the builder already owning `env_`), not a convenience one. This is exactly the kind of deviation ADR-0012 §P2 records: "a recorded deviation from the audit's literal env-attribute… the [mechanism] achieves R9's intent without [the cycle]."

- **Same memo idiom as `feature_layout()`:** `_LAYOUTS` memoizes the *layout* per env; the C++ builder already memoizes the *layout* (it reads `feature_layout.json` once in the ctor). The 3a belief/loc memos are the *same idiom one level in* — memoize the belief-derived intermediates and the per-loc distances, the next two env-derived-or-belief-derived facts down. The precedent is not merely "consistent with"; 3a is the *natural fourth instance* of the memo-on-the-lifetime-shared-owner pattern the codebase already runs three times (`_LAYOUTS`, `_SLOT_TABLES`, the C++ ctor's layout read).

---

## 3b (landed, bc73a79) — retroactive compliance assessment

**Verdict: compliant, and a *positive* instance of P1 — it should be cited as a worked example, not merely cleared.** Walking the maintainer's checklist:

- **P1 (SSOT) — strengthened, not violated.** Before 3b, the belief marginal sweep (`marg`) had **three homes per leaf**: `build`'s inline sweep, plus two `env.legal_actions → marginals` recomputations in `evaluate`. 3b makes `legal_mask_from_features` *consume* `build`'s already-written `available`/`informative` blocks (which *are* the collect-legal and sense-legal masks by construction) — collapsing marg to **one home (build's)**, the mask reading build's output. This is P1's "derived quantities are computed [once], never re-typed" applied exactly: the mask is *derived from* the feature vector, not *recomputed from* the belief. The header comment states this correctly ("marg has ONE home, build's; the mask consumes it"). **This is the cancer-B remediation, not a new B** — and it is the same move as the audit's `FeatureLayout` (collapse N writers to one).

- **P9 (functional core / honest signatures) — clean.** `legal_mask_from_features(std::span<const float> feat) const` takes a **bounds-carrying `std::span<const float>`** (P9-rule-1), **returns `std::vector<float>` by value** (P9-rule-2, free under NRVO), is **`[[nodiscard]]`** (P9-rule-5's nodiscard surface — a caller cannot silently drop the mask), mutates **nothing** (no out-param, no hidden state — P9-rule-3), and **derives its offsets from the FeatureLayout SSOT** (`layout_.start("available")` / `start("informative")`, not magic literals — P7/P1). It is a *textbook* P9 pure value-function: typed in, typed out, no effect. There is no `mutable`-cache question here at all — 3b adds *no* state; it is a pure slice of an existing buffer.

- **P4 (live-not-frozen) — N/A, no freezing.** The mask is computed per-leaf from the live feature vector; nothing is baked.

- **Cancer-C (hidden state keyed by nothing) — N/A.** 3b introduces no cache, no global, no identity key. It is stateless.

- **P6 (bit-exactness) — the right tier, and verified.** The legal mask is a **logic invariant** (P6 case (a) / P7: "the legality `M` mask is bit-identical to Python's for the same `(loc, belief)`… illegal-slot mass `== 0.0`"). 3b's claim "verified bit-exact (gumbel parity green)" is the *correct* substantiation at the *correct* tier — the mask is exactly the kind of quantity P6 says to assert bit-exactly, not behaviorally. The equivalence argument is airtight *because* `available == (marg>0 ∧ ¬collected) == env.legal_actions`'s collect test and `informative == (0<cnt<nb) == env.informative` — the header documents this identity, and it is a *logic* identity float cannot perturb. The `node.legal_slots` derived from the mask is therefore in the same env order `env.legal_actions` yields (treasure ids, then detector ids, then TERMINATE), so the downstream PUCT scan order is preserved — the comment in `evaluate` (gumbel.cpp:257-259) makes this explicit and it checks out.

- **The one thing to verify is named honestly and is correct:** 3b *keeps* the free `legal_mask(env, bw, collected)` for the per-step training-mask emission (runner.cpp) and the parity tool. Is keeping both a cancer-B "abstraction-beside-a-live-copy" (E)? **No** — and the distinction is exactly P5/E's: the two functions compute the *same logical mask* but from *different available inputs at different call sites*. `legal_mask_from_features` is the **hot-path** form (a leaf already has `feat` in hand — consuming it is free); `legal_mask(env, bw, collected)` is the **non-hot** form (the per-step training emission and the parity tool do *not* have a built `feat` and would otherwise have to build one just to slice it). This is *not* two writers of one truth that can drift — both ultimately rest on the *same* belief facts (`available`/`informative` ≡ `env.legal_actions`), and the parity tool's whole job is to *assert they agree*. The honest reading: `legal_mask` is the **reference/oracle** kept deliberately for the parity backstop (P7: the runtime parity test that nets the hot path against the reference), and `legal_mask_from_features` is the hot path it certifies. Keeping the oracle is *correct*, not E — E is an *abandoned* abstraction beside a live inline copy; here the "second" function is the *actively-used oracle that nets the first*. I would add one line to the header making this explicit ("`legal_mask` is retained as the parity oracle and the no-feat-in-hand path; `legal_mask_from_features` is the hot path it certifies") so a future reader does not mistake the pair for an E-style duplication — a small ADR-0005/P7 documentation completion, the kind CLAUDE.md's "documentation is part of the work" calls for.

**3b is clean. Its only debt is documentary** (name the `legal_mask` ↔ `legal_mask_from_features` relationship as oracle↔hot-path explicitly), and it is a *model* of the P1 "consume the one computation" move that 3a then extends one level deeper.

---

## Where a critic will attack this proposal (stated so the next reviewer aims true)

1. **"Mutable cache + PoolRuntime = data race."** Answered in (a): the builder is constructed *per task per thread*; it is never shared across threads, so the `mutable` mutation is single-threaded by ownership. The critic must show a *sharing site* — there is none today; if one is introduced, the synchronization obligation fires *there*.
2. **"Keying a cache on a `double` is the float-equality sin."** Answered in (g): the key is *fixed stored geometry*, not a computed float; exact-bits-of-fixed-data is correct and an epsilon compare would be the actual bug; `unordered_map` resolves hash collisions by exact `==`, so correctness holds even under computed locs (only hit-rate would suffer).
3. **"The cache is a second home for `marg` (P1/cancer-B)."** Answered in (e): the memo *stores the output of the one author* and contains no belief arithmetic; it is structurally incapable of drift because it holds no copy of the math to drift. Memo-of-the-one-computation ≠ second SSOT.
4. **"`build() const` + a mutating cache is a lying signature (P8/P2)."** Answered in (d): `const` is *logical* const, honored because hit≡miss bit-exactly (P6); the mutation is named by the typed `BeliefMemo`/`LocMemo` members and the `.get()` call, satisfying P9-rule-3's reviewer-can-name-every-mutation check better than dropping `const` would.
5. **"You re-authored the belief fingerprint."** Pre-empted in (b): *reuse* `gumbel_belief_key` / `GBeliefKey` from `gumbel.hpp`; authoring a second `(n,front,back)` would itself be the P1 violation.
6. **"The Workspace framing means thread a `Workspace&` through `build()`."** Answered in (d): a Workspace is *transient scratch*; a memo is *persistent derived state* on the lifetime-shared owner — different category, different mechanism. The named-component-on-the-builder is the memo's correct typed form, and the cache must *not* be pushed into the caller's ownership.

**Bottom line.** 3a should land as: (i) the step-1 pure free functions *promoted to a named, unit-testable compute header* (functional core); (ii) two one-owner typed memo components `BeliefMemo`/`LocMemo` (P3) held as `mutable` members of `FeatureBuilder`, the belief memo reusing `gumbel_belief_key`/`GBeliefKey` with a full-`bw`-equality collision guard and an internal cap, the loc memo keyed on the exact-bit `Point`; (iii) thin memo-wrapper calls inside `build()`, which stays `const`. It is not cancer-C (member, not global; value key, not address; lifetime-shared owner), not a second SSOT for marg (memo of the one author), not stale (its inputs are immutable under a live env — HOT knobs and net version are not featurizer inputs, and an instance change is a loud reject), and bit-exact (stored bytes returned unchanged, exact equality guards). It is the natural fourth instance of the `_LAYOUTS`/`_SLOT_TABLES`/ctor-layout memo idiom, and lands R9's *literal* intended home that Python could only approximate. 3b is already compliant — a positive P1 instance — with one small documentary completion owed (name `legal_mask` as the retained parity oracle).

Relevant files: `/home/bork/w/vdc/1/chocofarm/cpp/src/features.cpp`, `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/features.hpp`, `/home/bork/w/vdc/1/chocofarm/cpp/src/gumbel.cpp`, `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/gumbel.hpp` (GBeliefKey at :88), `/home/bork/w/vdc/1/chocofarm/cpp/src/search_runtime.cpp` (per-task builder lifetime), `/home/bork/w/vdc/1/chocofarm/cpp/src/serve.cpp` (build-once env/fb, instance-change reject, net-version reload), `/home/bork/w/vdc/1/chocofarm/cpp/include/chocofarm/instance.hpp` (Point has no operators), `/home/bork/w/vdc/1/chocofarm/chocofarm/az/features.py` (the idiom 3a ports; `_LAYOUTS` at :204), `/home/bork/w/vdc/1/chocofarm/chocofarm/solvers/ismcts.py` (`_belief_key` at :70).

---

## Appendix B — Phase 2: CRITIQUE (adversarial reviewer, raw)

I now have everything required. The decisive finding for ADR-0009 transferability: the Python `_belief_key` returns `(n, int(bw[0]), int(bw[-1]))` — and crucially `bw[0]`/`bw[-1]` are the FIRST and LAST elements of the world-set *array*, which is order-sensitive on the array's storage order. The C++ `gumbel_belief_key` uses `bw.front()` / `bw.back()`. SUGGEST claimed reusing `GBeliefKey` is correct; I must check whether the C++ `bw` ordering matches Python's `bw[0]/bw[-1]` semantics, since a belief-feature cache keyed on it requires the same array-order invariant the Python comment relies on. But that's already established by the existing node-cache reusing the same key. The genuinely decisive issue is elsewhere.

I have all the material. Let me write the critique.

---

# Adversarial review of the 3a/3b SUGGEST proposal

I have read end to end: `docs/adr-synopsis.md`; ADR-0012 in full (all 1151 lines, both pages); the dossier in full; `features.hpp` + `features.cpp` (POST-3b); `gumbel.cpp` evaluate + the node-cache machinery (`gumbel_belief_key` :120, `GBeliefKey` :88, `node.evaluated`, the `(slot, GBeliefKey)` child transposition :362/:399); `features.py` in full; and — because the proposal's P4 defense rests entirely on lifetime facts — `serve.cpp`'s `ServeState`, `runner.cpp`'s `run_episode`/`run_episodes`, `search_runtime.cpp`'s per-task builder, and `gumbel_search.py`'s episode-scoped reset. The decisive facts the proposal under-weighted are in `serve.cpp:70/97/173` and `runner.cpp:49` vs `gumbel.cpp:239`.

A general note first: SUGGEST is long, fluent, and pre-arms every attack — which is itself a tell. ADR-0012 names the *argument shape* "answered in (x), the critic must show a sharing site" as the move to distrust (P7/P8/P9 all reject "the discipline applied once is fine" pre-emptive dismissals). Several of its "SURVIVES" rulings are pre-emptive dismissals of the wrong attack. I go after the real ones.

---

## (a) P1 — is the cache REALLY not a second data source? Steelman the unease. **VERDICT: REAL (the proposal's "categorically not" overstates; a genuine duplication exists, just not the one it defended against).**

SUGGEST defends the easy version: "the memo stores the output of the one author and contains no belief arithmetic, so it cannot drift." That much is correct **for the marginal sweep**: `BeliefMemo` receiving `compute` as a callable and storing its result is not a second home for `marg`. Granted.

But the maintainer's unease has a sharper target the proposal walked past. **There is already a second consumer of the belief sweep on the hot path, and 3a multiplies the number of independent belief computations rather than collapsing them.** Look at what actually runs per ply in the serve/runner path:

- `gumbel.cpp:239` — the policy's `fb_.build(...)` runs the `O(nb·(N+nD))` belief sweep at every leaf inside the search (memoized per node by `node.evaluated`).
- `runner.cpp:49` — *after* the search returns, `run_episode` calls a **different** `FeatureBuilder` (`fb`, = `st.fb`, the daemon's, **not** `st.policy->fb_`) to `build` the **same** `(loc, bw, collected)` belief **again**, to produce the recorded `X` block.

So the root belief of every decision is sweep-computed at least twice by two *different* builder instances — once by `st.policy->fb_` (search root, `gumbel.cpp:545`) and once by `st.fb` (the record, `runner.cpp:49`). 3a's per-builder cache **cannot** collapse that: the two builders have two disjoint caches. P1's literal check is "grep the tree for the value; if it appears in two places that must agree" — here the *same belief features* are independently recomputed in two builders that must agree (and they do agree only because `build` is deterministic). That is not cancer-B (no hand-copied literal), but it is exactly the P1 *spirit* the dossier's §1 DAG and 3b's own "marg has ONE home" framing serve: **a belief feature vector with one logical home computed at N physical sites.** 3a, as scoped to a per-builder member, leaves this duplication standing and even sanctifies it (each builder gets its own cache). The honest P1 move — the one 3b started — would be for the record path to *consume the search root's already-built feature vector* (the search already built `X` for the root at `gumbel.cpp:545`), not rebuild it in a second builder. The proposal never notices the two-builder duplication because it asserts "each worker thread owns its own policy+builder" as though that were the whole story; in serve the *record* builder is a third instance neither the per-task nor the policy builder.

This does not sink 3a, but "memo-of-the-one-computation, **categorically** not a second SSOT" is an overclaim. The cache is honest *qua marg*; the surrounding structure has a real two-writers-of-one-belief-vector smell 3a should be scoped against, not declared immune to. **REAL.**

---

## (b) P9 rule 3 — is `build() const` + a mutable cache an honest signature? Is logical-const a defense or a dodge? **VERDICT: REAL → leaning FATAL-if-unfixed (the proposal's logical-const argument is half-right and half a dodge, and it dodges the part that actually bites).**

SUGGEST's logical-const argument is correct *as far as single-threaded value semantics go*: `mutable` for a memo whose hit ≡ miss bit-exactly is the textbook sanctioned use, the std-lib blesses it, and the typed `BeliefMemo`/`.get()` does name the mutation better than a hidden `static`. P9-rule-3's check — "can a reviewer name, from the call, every piece of state mutated?" — is met. Within one thread, this is SURVIVES.

But P9-rule-3 is not only about *nameability*; ADR-0012's P9 frames the whole compiled-component contract around *honest signatures the reviewer enforces from the signature alone*, and a `const` method that mutates shared state is dishonest **the moment the object is shared across threads** — and the proposal's own (a) tells you it will be. Here is the dodge: SUGGEST asserts "the builder is never shared across threads, so the mutation is single-threaded by ownership," and rules the data-race attack SURVIVES. That assertion is **true for the two search runtimes** (`search_runtime.cpp:42/71` construct a fresh `GumbelAZPolicy policy(task.cfg, counter, env)` — hence a fresh `fb_` — per task per thread). **It is false for serve.** In `serve.cpp`, `st.fb` (`:70`) and `st.policy` (`:75`, holding `fb_`) are **long-lived process-scoped objects**, and `run_episodes` is called with `*st.fb` across many `generate` calls. The serve daemon today is single-threaded per the message loop, so there is no race *today* — but the `const`-correctness of `build()` is precisely what advertises to a future maintainer "this is safe to call from multiple threads / safe to share." A `const` method silently mutating a `mutable` cache is the canonical C++ thread-safety trap: `const` is the language's standard-library contract for "safe for concurrent reads" (this is why `std::shared_ptr`'s const methods are atomic and why the standard requires const member functions of standard types to be data-race-free). 3a *breaks that contract* on a process-scoped object. The proposal even acknowledges "if a future restructure shared one builder across threads, *that* change would owe the synchronization analysis" — but that future is **already half-present** in serve's process-scoped `st.fb`, and the async actor-learner (ADR-0012's stated second incoming body of code) is exactly a multi-threaded sharing future.

The honest reading: logical-const is a *real* defense for the *value contract* and a *dodge* for the *concurrency contract*, and ADR-0012 cares about both (P9 is "the modern-C++ discipline"; data-race-freedom of const is a core modern-C++ invariant). The proposal should either (i) keep the cache strictly on per-task builders and *forbid* it on the process-scoped `st.fb` (i.e., do NOT memoize the serve daemon's record builder), or (ii) if it memoizes a shared builder, owe the synchronization analysis *now*, not defer it. Declaring SURVIVES while the live serve path holds a process-scoped builder is the dodge. **REAL, trending FATAL** because it is a silent-correctness landmine of exactly the cancer-C "masked today only because…" shape ADR-0012 P2 warns of ("masked today only because every env is layout-identical").

---

## (c) SHAPE — is the recommended structure clean, or still "free functions lying around"? **VERDICT: REAL (the proposal solves the maintainer's stated discomfort by relabeling it, and introduces a worse smell it doesn't price).**

The maintainer said "free functions lying around in the source code." SUGGEST's answer: "the discomfort is misplaced — a functional core *is* free functions; promote them to a named header and the discomfort dissolves." This is rhetorically clean and partly right (P9 genuinely does endorse free pure functions as the core). But it does two things the maintainer should reject:

1. **It expands a memoization task into a public-API/file-topology change the task did not ask for.** Promoting `belief_features`/`geometry_features` out of the `features.cpp` anonymous namespace into a new `feature_compute.hpp` public header is a real surface expansion: those functions become a *new compiled-component public contract* (P8/P9 — every signature now reviewed, `[[nodiscard]]`, span-typed, the works), a new translation unit, a new thing for `gumbel.cpp`/`runner.cpp` to depend on. ADR-0012's scope is "binds new structure at authoring time" and CLAUDE.md's scope discipline says *do not expand the change without surfacing the cross-cutting nature*. The dossier itself files the decomposition (§2) and the memo (§4.4) as **separate** avenues — §2 is DO-NOW hygiene already *landed* (the step-1 decomposition is in `features.cpp:119-200` today), §4.4 is HOLD. SUGGEST quietly re-opens the *already-landed* §2 (it wants to move the functions to a header) and bundles it with the *gated* §4.4. That is two refactors wearing one PR.

2. **The `template<class Compute> get(bw, compute)` shape is itself a smell the proposal mis-prices.** SUGGEST's `BeliefMemo::get(span, Compute&&)` "cannot re-author the belief math because it receives `compute`." True — but a *templated* memo taking a generic callable is *more* abstraction than the problem has (it is a one-instantiation generic; there is exactly one `compute` ever passed). ADR-0003's "abstractions are extracted only when a second concrete instance exists, not preemptively" and the audit's **cancer E** ("an abstraction built then abandoned beside a live inline copy") both bear here: a generic `Memo<Compute>` with a single call site is an over-built abstraction. The cleaner shape is the dossier's own: the memo is a plain method on `FeatureBuilder` that *directly* calls the (anonymous-namespace, kept-private) `belief_features`, with the cache as private members — no new header, no template, no public compute-function contract. That is *less* structure, not more, and it is what the maintainer's "hoist into memoization-wrappers" instinct actually described. SUGGEST overshoots it into a typed-component-with-injected-callable design and calls the overshoot "cleanest."

The uglier truth SUGGEST missed: the **distance/geometry home is unresolved** (dossier §5 "OPEN, deferred — where the matrix is homed," and §4.4 "same class as §5"). The proposal homes `loc_cache_` on the builder as a `mutable` member *without engaging the deferred §5 question at all*. The dossier explicitly flags that geometry homing is the same P1/P2 open question as the belief cache and is **DEFERRED**, not DO-NOW. SUGGEST resolves it by fiat (builder member) and never cites that §5 left it open. That is scope-jumping a deferred design decision. **REAL.**

---

## (d) cancer-C — does the keying/homing truly avoid it? **VERDICT: REAL (the address/global hazard is avoided, but the two-instances + cap-clear hazards the maintainer's hint points at are live).**

The proposal is right on the narrow cancer-C indictment: a `mutable` member keyed on a value fingerprint + full-`bw`-equality (belief) and on `Point` by value (loc) is *not* a module-global-keyed-by-`id()`. No address is keyed; no never-evicted global accumulates. That clause SURVIVES.

But cancer-C is "hidden state keyed by **nothing** — by a value-less identity," and the maintainer's parenthetical hint names the live version of *that*: **"the runner has a SEPARATE FeatureBuilder from the policy; the wire-bench builds a fresh FeatureBuilder per task."** This is the hazard the proposal's homing argument does not dispose of, because the danger isn't an address — it's **two builders disagreeing because they hold two caches with two cap-clear histories**. Concretely, in serve:

- `st.fb` (record builder, `serve.cpp:97`) and `st.policy->fb_` (search builder, rebuilt at `:173` on every net reload / HOT change) are **distinct instances with distinct caches**. Today `build` is pure so they agree. But once each holds a cap-clearing cache (`clear the whole cache past 50000`), the two builders evict at *different times* depending on how many distinct beliefs each saw — `fb_` is rebuilt fresh on every HOT/version change (so its cache is young), while `st.fb` is **never rebuilt for the process's life** (so its cache fills and clears on its own schedule). They never return *different* values for the same belief (the equality guard holds), so correctness survives — but the proposal's claim that the cache "dies with the builder, which dies with (or before) the env" is **false for `st.fb`**: `st.fb` lives for the *entire daemon lifetime*, accumulating beliefs across *every episode of every generation*, and is bounded only by the 50000-cap whole-cache-clear. That is precisely the unbounded-accumulation-across-episodes shape Python's `reset_belief_cache` exists to prevent (`gumbel_search.py:222-232`), and which the C++ port — by SUGGEST's explicit choice — **omits the reset hook for**. So the long-lived serve `st.fb` gets the worst of it: no per-episode reset, only a coarse 50000 whole-cache-stomp. The proposal *waved the reset off* as "dead symmetry (ADR-0008)" — but `gumbel_search.py:222` shows the reset is not symmetry; it is the live mechanism that bounds the cache to *one episode's hundreds of beliefs* rather than letting it grow to the cap. Dropping it changes the memory profile of the long-lived serve builder from O(hundreds) to O(50000), and the proposal presents that as a clean simplification. **REAL.**

(Sub-note, also REAL: SUGGEST proposes keying the loc cache on `Point` with a `std::bit_cast<uint64_t>` hash over the `double` bit-pattern. `Point` is `{double x, double y}` — that is **two** doubles, 128 bits, so `std::bit_cast<uint64_t>` does not even compile against a 16-byte `Point`; the proposal's own concrete code is wrong on the width. A minor implementation bug, but it reveals the loc-key facility was hand-waved, not designed.)

---

## (e) P4 — staleness modes the proposal dismissed. **VERDICT: mostly SURVIVES on the search-knob axis; REAL on the cap-clear/cross-episode axis (see (d)); SURVIVES on instance.**

The core P4 argument is sound and I will not pretend otherwise: the belief/geometry features are functions of `(env-as-frozen-instance, bw, loc)`; the HOT knobs (`m`, `n_sims`, `lam`, `max_steps`) and the net version are **not inputs to the featurizer** (confirmed: `lam` enters `descend`/`simulate_root_action` scoring at `gumbel.cpp:341/355/361/398`, never `belief_features`/`geometry_features`); and an instance/faces change is a loud `instance_knob_changed` reject (`serve.cpp:5-6`, the env built once at `:96`). So no HOT/version/instance reconfig can stale a cached value. That part **SURVIVES**, and it is the proposal's strongest section.

What it *dismissed* and should not have:

- **Cross-episode persistence on the long-lived `st.fb`** (detailed in (d)): the proposal's "no per-episode reset hook needed" is a real P4-adjacent change — Python deliberately *resets per episode* (`gumbel_search.py:231`), and the C++ serve path has no equivalent, so the daemon's record builder accumulates across the whole run. Not staleness-of-value, but a deliberate-lifecycle removal dressed as "dead symmetry." **REAL.**
- **The cap-clear is a silent behavior cliff.** Clearing the *whole* cache at 50000 means a cache that was serving hits drops to cold in one step. The proposal calls this "purely a memory bound." On the long-lived `st.fb` it is also a *latency* cliff (a periodic full-recompute storm), which on a hot path is exactly the kind of thing ADR-0009 says you measure before claiming benign. Minor, but the "purely a memory bound" framing is incomplete.

---

## (f) P6 — can the collision verification OR the float `Point` key break bit-exactness/correctness? **VERDICT: SURVIVES on correctness, REAL on one honesty gap the proposal buried.**

The belief collision guard is correct and load-bearing: fingerprint is collision-resistant (`_belief_key`'s own docstring, `ismcts.py:70-80`), and a hit walks the bucket testing **full `bw` equality** before returning, mirroring `np.array_equal` (`features.py:330`). With the full-equality guard present, a hit is provably the same belief, hence bit-identical features. **SURVIVES** — and the proposal's instinct to *add a parity test that constructs a fingerprint collision* is exactly right (P6/P7: net the guard, don't trust it).

The float-`Point`-key argument **survives on correctness** for the reason SUGGEST gives (it's exact `==` on stored fixed geometry, not a computed float; `unordered_map` falls through to `operator==` on a hash collision, so distinct points are never conflated). I agree the *correctness* holds.

But there is a buried honesty gap the proposal glossed: the loc key correctness rests on the **load-bearing premise that `loc` is always one of a finite set of named coordinate points**, and SUGGEST asserts this from a comment ("`Loc{Point}` is the resolved Point, as in env.coord"). That premise is *not enforced anywhere* — it is exactly the "load-bearing knowledge offloaded to unenforceable prose" of **cancer G**. If a future caller ever passes an interpolated/computed `Point` (the proposal admits this is possible), correctness still holds but the cache silently degrades to ~0% hit-rate — a *silent perf regression* with no fail-loud. ADR-0012 P5/G says encode the invariant or cite the enforced derivation, not the comment. The proposal cites the comment. A clean 3a would either (i) key the loc cache on a *discrete coordinate index* (the named-point id the geometry table is already indexed by — the dossier §5 distance-*matrix* is indexed by coordinate key, not by `Point`!) rather than on raw `double` bits, which is *both* faster and immune to the float-key question entirely, or (ii) assert the named-point premise. SUGGEST picked the raw-`double`-bits key, which is the one shape that *needs* the unenforceable premise — and §5 already says the geometry home is a **matrix indexed by coordinate key**, so the float-`Point` key is arguably re-authoring an index that already has a discrete home. **REAL.**

---

## (g) 3b — did the proposal rubber-stamp it? Is there an implicit/undeclared interface between `build()` and the mask? **VERDICT: REAL — the proposal pronounced 3b "a *model* of P1" while glossing the genuine P2/P8 seam concern the maintainer raised.**

SUGGEST gives 3b a glowing "compliant, a positive instance of P1, cite it as a worked example." Most of that is defensible: `legal_mask_from_features` takes `std::span<const float>`, returns by value, is `[[nodiscard]]`, derives offsets from the layout SSOT — textbook P9-rule-1/2/5. Collapsing marg's three homes to one is a real P1 win. Granted.

But the maintainer's concern (g) is precise and the proposal did not answer it: **deriving the mask from the built feature vector creates an implicit, undeclared interface between `build()` and `legal_mask_from_features()`.** This is a genuine P2/P8 issue SUGGEST waved past:

- `legal_mask_from_features(std::span<const float> feat)` takes a *bare float span*. Its contract — "this span must be exactly the vector `build()` produced, in the current layout, with `available` meaning `marg>0 ∧ ¬collected` and `informative` meaning `0<cnt<nb`" — is **nowhere in the type**. The signature says `span<const float> → vector<float>`. Nothing forces the caller to pass a `build()` output rather than an arbitrary float buffer, and nothing in the type couples the two functions to the *same* `FeatureLayout` instance. That is precisely **P8's "the typed signature is the SSOT of the contract"** failing: the real contract (must be a build-output of this builder) is carried by a comment (`features.hpp:56-60`), not the signature. ADR-0012 P8's check (a): "does the body honor each annotation — no value the annotation forbids reaches a consumer?" — here a wrongly-laid-out span *is* a value the annotation `span<const float>` permits but the body cannot honor. It is the **call-boundary form of G** (load-bearing knowledge in a comment) that P8 names verbatim.

- The mitigation is real and cheap and the proposal *should* have flagged it instead of celebrating: the two methods share the *implicit* invariant "same builder, same layout, `feat` is *my* output." Because both are members of the **same `FeatureBuilder`**, the coupling is at least *scoped* to one object (better than two free functions) — but the `span<const float>` input still erases it. A reviewer cannot, from the signature alone, see that `feat` must be `this->build(...)`'s return. SUGGEST's own framing in (d) ("a reviewer can name every mutated state from the call") cuts the other way here: a reviewer *cannot* name, from `legal_mask_from_features(feat)`'s signature, that `feat` is layout-coupled to `build`. This is a real, if modest, P2/P8 seam debt that 3b introduced and the proposal **rubber-stamped as a positive example**. The honest verdict on 3b is "compliant on P1/P9-r1/r2/r5, with an undeclared `build`↔`mask` layout coupling carried by comment (P8/G debt) that should be named" — not "a model to cite." **REAL.**

(The proposal's other 3b point — keeping the free `legal_mask(env,bw,collected)` as a "parity oracle" — is fine and I agree it's not cancer-E. That sub-ruling SURVIVES.)

---

## (h) ADR-0009 — is 3a's value established, or is "the gate is already met" a dodge? **VERDICT: FATAL (this is the proposal's worst move, and it is the one ADR-0009/P6 exist to forbid).**

This is the attack the proposal least survives. SUGGEST's justification for 3a's *existence* is: "dossier measure-first gate 4.4 already met: features.py records ~3.5× belief recurrence per episode; the C++ Gumbel search has the SAME node-level cache as Python, so the ~3.5× is the INCREMENTAL recurrence a belief cache captures." Every clause of this is a transfer of a **Python measurement onto a C++ artifact that has not been measured**, and ADR-0009/P6 name exactly that as the dodge.

1. **The dossier does NOT say the gate is met. It says the opposite.** Dossier §4.4 is tagged **HOLD**, and its gate reads: *"GATE: needs real access-pattern data. Measure belief recurrence before building."* The §6 table lists 4.4 as **HOLD**, enforcement "access-pattern data; value-stable key." The proposal claims "measure-first gate 4.4 already met" — the dossier it cites as authority says **not met, measure first**. SUGGEST inverts its own cited source. ADR-0011 Rule 3 (measure-first) and ADR-0009 ("a perf claim is honest only when its investigation is captured reproducibly") are both violated by asserting a HOLD-gated avenue's gate is cleared by pointing at a *different* program's number.

2. **The ~3.5× was measured on the Python ISMCTS/Gumbel search, not the C++ one — and the proposal admits the transfer is non-trivial, then makes it anyway.** The `features.py:254` ~3.5× ("the same belief is reached ~3.5× on average") is a Python-search access-pattern fact. The C++ search is a *different program* with a *different node cache* (`GBeliefKey` transposition, `node.evaluated`). SUGGEST argues the C++ node cache is "the SAME," so the ~3.5× is the *incremental* recurrence above the node cache. But that is an *assumption about the C++ access pattern*, and ADR-0009's whole point is that an access-pattern/perf claim is established by a *captured C++ measurement* (`bench_hotpath.py`, an allocation/recurrence profile), never by an author's argument that two programs *should* have the same locality. The proposal even gives the tell P7/P8/P9 reject in three different words: it argues the value from *plausibility* ("the same node-level cache, so the ~3.5× is incremental") with **no C++ recurrence count attached**. ADR-0012 P6: a claim "is honest only with its substantiation attached." There is no substantiation — there is a borrowed number and a syllogism.

3. **The incremental recurrence above the node cache could be near-1.** This is the substantive risk the borrowed number hides. The C++ node cache already evaluates each `(slot, GBeliefKey)` child *once* (`node.evaluated`, `gumbel.cpp:340/346`). The belief *features* are built inside `evaluate` (`gumbel.cpp:239`), which is *already gated by `node.evaluated`* — so within one tree the same belief's features are already built once per node. The cross-leaf recurrence a belief cache would additionally capture is only: (i) the *same belief reached at different `(action)` parents* (a genuine transposition the node cache *does* fold via `GBeliefKey`, so already captured), and (ii) the same belief across *different root decisions in one episode* — which is the part the *episode-scoped* Python cache captures and the C++ has no episode reset for. So the incremental win of a *builder-level* belief cache **on top of** the existing `node.evaluated` + `GBeliefKey` node cache is plausibly *much smaller* than 3.5× — and could be dominated by the **two-builder duplication** I identified in (a): the single largest belief-recompute the C++ does is `st.fb.build` (record) re-doing what `st.policy->fb_.build` (search root) just did. A cache *within* one builder doesn't fold that *across* builders. So the measured C++ win could be ~0 from the proposed cache while a *different* fix (share the root feature vector between search and record) captures the real duplication. The proposal optimizes the wrong recompute because it never measured which recompute dominates.

The honest disposition, per the dossier's own tag and ADR-0009: **3a is gated on a C++ access-pattern measurement (`bench_hotpath` recurrence count: distinct-belief builds vs total builds, per builder, in the serve path) that has not been taken.** "The gate is already met" is the exact `for now`/`it's faster`/borrowed-number dodge ADR-0012 P6/P9-rule-4/ADR-0009 are written to reject. Until that measurement exists, 3a is **HOLD**, not PROPOSED-and-justified. **FATAL** to the proposal's justification (not necessarily to the eventual feature — but the feature cannot land on this justification).

---

## Summary table

| Concern | Verdict | Core ADR-0012 citation |
| — | — | — |
| (a) P1 second data source | **REAL** | P1 / cancer-B spirit: the two-builder belief-vector duplication (`runner.cpp:49` vs `gumbel.cpp:545`) stands; "categorically not a second SSOT" overclaims |
| (b) P9-r3 `const`+mutable honesty | **REAL → FATAL if applied to `st.fb`** | P9 rule 3 + P2 "masked today only because"; logical-const defends the value contract, dodges the *concurrency* contract on serve's process-scoped builder |
| (c) SHAPE | **REAL** | ADR-0003 / cancer-E (premature generic `Memo<Compute>`); scope-creep over the already-landed §2 decomp + the *deferred* §5 geometry home; CLAUDE.md scope discipline |
| (d) cancer-C / two instances | **REAL** | P2(c) + cancer-C "keyed by nothing" in the *two-builders, two-cap-histories* form the maintainer flagged; "dies with the builder" false for `st.fb` |
| (e) P4 staleness | **SURVIVES** on search-knob/instance axes; **REAL** on cap-clear/cross-episode | P4: HOT/version/instance cannot stale a value (strong); but the dropped per-episode reset (`gumbel_search.py:231`) is a real lifecycle removal mislabeled "dead symmetry" |
| (f) P6 collision + float key | **SURVIVES** on correctness; **REAL** on the unenforced named-point premise | P6 (guard is load-bearing and exact — correct) + cancer-G (the loc-key correctness rests on a *comment*, not enforcement; §5's coordinate-index home is the cleaner key) |
| (g) 3b implicit `build`↔mask seam | **REAL** | P8 / cancer-G at the call boundary: the `span<const float>` input erases the "must be *my* build output, same layout" contract into a comment; 3b was rubber-stamped |
| (h) ADR-0009 transferability | **FATAL** | P6 / ADR-0009 / ADR-0011 Rule 3: the dossier §4.4 gate is **HOLD/measure-first**; the proposal inverts it and transfers a *Python* recurrence number onto an *unmeasured* C++ artifact with a syllogism, the precise dodge these tenets forbid |

**Bottom line for the maintainer.** The proposal's narrow cancer-C / P1-marg / P6-bit-exactness defenses are genuinely sound — the cache is *not* cancer-C, *not* a second home for the marginal sweep, and a hit *is* bit-exact under the full-`bw` guard. But it earns those wins by attacking the easy versions and pre-dismissing the hard ones. The two findings that should block 3a *as justified*: **(h)** its entire reason-to-exist is an inverted gate plus a borrowed measurement (FATAL — take the C++ recurrence profile first, in the serve path, per builder, and you will likely find the dominant recompute is the two-builder `st.fb`/`fb_` duplication of (a), not cross-leaf recurrence); and **(b)** homing a `mutable` cache behind a `const build()` on serve's **process-scoped** `st.fb` is a silent concurrency-contract break the proposal's "single-threaded by ownership" claim is simply false about (it is true only for the per-task runtime builders). The clean path: (1) measure C++ belief recurrence per builder before building anything (ADR-0009); (2) scope any cache to the *per-task* builders that are provably single-threaded and short-lived, and explicitly exclude the long-lived serve `st.fb` (or give it back the episode reset Python has); (3) consider that the real P1 win is sharing the search-root feature vector with the record path, not memoizing two builders independently; (4) for 3b, add one line of P8 honesty naming the `build`↔`legal_mask_from_features` layout coupling the `span` input erases. The structural recommendation (free pure compute + a typed memo component) is *directionally* right but over-built (drop the template; keep the compute functions private; do not re-open the §5 geometry home by fiat).
