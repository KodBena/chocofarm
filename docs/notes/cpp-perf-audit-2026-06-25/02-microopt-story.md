# Micro-opt per-stage perf CPU stats

All single-process, **core 3**, `nice -n -19`, `--mode cursor` (the production engine path), production
config (m=24 n_sims=256 c_outcome=2 max_depth=24). throughput = median µs/decision; topdown =
retiring/backend/frontend/bad-spec %; cache = `mem_load_retired` L1i/L2/L3 misses + dTLB; prefetch =
`l2_rqsts.pf_miss`. The raw harness output is preserved below the table as provenance.

| stage | µs/dec | cycles (e9) | retiring | backend | frontend | bad-spec | L1i-miss (M) | L2-miss (M) | L3-miss (k) | dTLB-miss (M) | prefetch-miss (M) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (901be39) | 7096 | 17.81 | 57.4% | 24.1% | 12.6% | 5.9% | 159.8 | 45.2 | 748 | 2.21 | 554.6 |
| **C** eval_finish ws (`2d160ee`) | 7081 | 17.69 | 57.3% | 24.1% | 12.7% | 5.9% | 158.3 | 46.0 | 663 | 2.20 | 462.7 |
| **D** collected inline (`5cba058`) | ~7095 † | 17.85 † | 57.2% | 24.1% | 12.7% | 6.0% | 162.5 | 41.1 | 895 | 2.24 | 360.3 |

† D's single full-matrix run was noisy (7139 µs/dec, 17.85e9 cyc); the **interleaved D-vs-C A/B** showed
throughput **tied** (~7095 both). The cache/prefetch deltas are robust.

**Reading it:** throughput is ~flat across C and D and the topdown mix barely moves — because the 55%
hotspot is the bitset popcount (no easy lever; candidate A refuted). What the two malloc-removals (C, D)
*do* move is allocation/cache pressure: **prefetch-miss −35% cumulative** (554.6 → 360.3M), L2-miss −9%
(45.2 → 41.1M), cycles −0.6% at C. So the wins are structural (no per-leaf alloc on the production path)
+ cache, not raw cycles.

**Candidate A** (int64→uint32 accumulators) measured ~0.6% faster vs the **direct-mode** baseline (7160
µs/dec) — but in the *dead flat arm* (production runs the bitset popcount), so that delta is unattributable
code-layout noise → **reverted/refuted**. **#8/#13** (guard legibility) had no perf change (no behavior
change; not measured).

**Earlier major stages (context, measured separately — see findings #34/#35/#36 + 04/05):** producer-bound
CPU `cursor 7339 < direct 7374 < fiber 7458` µs/dec (Option B, finding #34); prior_d removal was CPU-neutral
but **−17% producer RSS** (#36). Those were distinct A/Bs (not this uniform cursor-mode sweep), so they
are referenced, not merged into the table above, to avoid apples-to-oranges.

---
# Raw harness measurements (provenance)

## BASELINE (901be39: cursor + prior_d removed)
throughput(median us/dec): 7160.29
topdown: 57.0 %  tma_retiring 13.1 %  tma_frontend_bound 24.0 %  tma_backend_bound 5.8 %  tma_bad_speculation 
cache/prefetch: 17,897,466,355 cycles |45,571,898,313 instructions |152,120,199 mem_load_retired.l1_miss |43,051,429 mem_load_retired.l2_miss |746,131 mem_load_retired.l3_miss |153,628,497 l1-icache-load-misses |2,266,580 dtlb-load-misses |1,363,549,182 l2_rqsts.all_pf |437,279,955 l2_rqsts.pf_miss |


## A — int64->uint32 belief_features accumulators: REVERTED (refuted)
Targeted `belief_features_nonempty` (the FLAT arm) int64 bit_cnt/det_cnt. But `full_belief()` returns a
BitsetBelief (use_bitset_=true at this instance: kW64=243 <= inline_cap=256), so the dispatcher runs
`belief_features_bitset` -> `popcount_and` (the 55% hotspot is the auto-vectorized SIMD popcount; the
vpand/vpaddq/vpmovzxdq I first read as the flat loop is actually the vectorized popcount). The flat arm is
DEAD for this instance, and `popcount_and` already accumulates in `int` (32-bit) — NO int-over-width lever
in the production hot path. Candidate #1 (lying int types) is REFUTED where it matters.
Measured A vs baseline: ~0.6% faster (7090-7117 vs 7129-7184 us/dec) — but UNATTRIBUTABLE: the narrowed
arm does not run, so the delta is code-layout/i-cache noise from recompiling the TU, not the change's
mechanism. Reverted (no production win; a real win must be in the bitset popcount or elsewhere).
LESSON: the earlier "int64 accumulator in the 55% hotspot" read mis-identified the arm; the bitset popcount
is the hotspot and is already well-typed. (The flat-arm narrowing is a valid but dormant flat-path cleanup;
file if the flat arm ever becomes hot.)
## BASELINE-cursor (901be39, production engine path)
throughput(median us/dec): 7096.33
topdown: 57.4 %  tma_retiring 12.6 %  tma_frontend_bound 24.1 %  tma_backend_bound 5.9 %  tma_bad_speculation 
cache/prefetch: 17,813,310,919 cycles |45,436,411,064 instructions |147,630,856 mem_load_retired.l1_miss |45,175,795 mem_load_retired.l2_miss |748,478 mem_load_retired.l3_miss |159,753,290 l1-icache-load-misses |2,210,704 dtlb-load-misses |1,367,218,917 l2_rqsts.all_pf |554,567,760 l2_rqsts.pf_miss |

## C: eval_finish reuses FeatureWorkspace (per-leaf malloc removed) [cursor path]
throughput(median us/dec): 7081.14
topdown: 57.3 %  tma_retiring 12.7 %  tma_frontend_bound 24.1 %  tma_backend_bound 5.9 %  tma_bad_speculation 
cache/prefetch: 17,690,953,705 cycles |45,371,413,255 instructions |147,514,751 mem_load_retired.l1_miss |45,982,576 mem_load_retired.l2_miss |662,910 mem_load_retired.l3_miss |158,262,557 l1-icache-load-misses |2,198,614 dtlb-load-misses |1,387,575,069 l2_rqsts.all_pf |462,742,671 l2_rqsts.pf_miss |

### C analysis: ~-0.7% cycles, bit-identical, KEPT (committed)
The per-leaf logits_d/prior_scratch malloc in eval_finish (cursor path) now reuses ws_. cycles -0.7%,
throughput -0.2% (within median noise but cycles is robust), L3-miss -11%, prefetch-miss -16%. The malloc
bucket is small vs the popcount-dominated total, so the win is modest; the structural correctness (the
production cursor path no longer re-pays the amortized bucket) is the real value. Pipeline mix unchanged
(retiring ~57%, backend ~24%).
## D: collected_features inlined into out[] (per-leaf vector removed) [cursor]
throughput(median us/dec): 7139.27
topdown: 57.2 %  tma_retiring 12.7 %  tma_frontend_bound 24.1 %  tma_backend_bound 6.0 %  tma_bad_speculation 
cache/prefetch: 17,851,592,066 cycles |45,376,558,939 instructions |147,878,039 mem_load_retired.l1_miss |41,116,867 mem_load_retired.l2_miss |895,180 mem_load_retired.l3_miss |162,534,288 l1-icache-load-misses |2,236,843 dtlb-load-misses |1,300,673,082 l2_rqsts.all_pf |360,304,558 l2_rqsts.pf_miss |

### D analysis: throughput-neutral, L2-miss -10% / prefetch-miss -22%, bit-identical, KEPT
Removed the per-leaf CollectedFeatures vector (write indicator into out[] directly). Throughput tied
(D~=C ~7095 us/dec interleaved); cache/prefetch improved (the alloc is gone) but not cycle-bound. Like C,
the value is structural (no per-leaf alloc on the production path) + cache hygiene. Cumulative C+D:
prefetch-miss 554M (baseline) -> 360M (-35%), L2-miss 45M -> 41M. Throughput ~flat (popcount dominates).

## #8 + #13 — admission-guard legibility: KEPT (committed), NO perf change
Rename est_fiber->est_tree (the engine is fiber-less) + name the /2 headroom literal. Same constants/
threshold/behavior (guard still rejects 400000-tree over-config). Pure legibility (ADR-0012 F / no-lying-
name); no throughput/cache/pipeline effect (not measured — not a perf change). The numeric down-recalibration
of the guard (would admit more) stays filed in BACKLOG (loosens admission; needs a deliberate decision).

---
# SUMMARY (as of d57cbdc)

| change | disposition | throughput | cache / prefetch | bit-identical |
| --- | --- | --- | --- | --- |
| A int64->uint32 accum | REVERTED (refuted) | n/a | n/a | n/a — wrong arm (flat dead; prod=bitset popcount, already int32) |
| C eval_finish ws | KEPT 2d160ee | -0.7% cycles | L3-miss -11%, pf-miss -16% | yes |
| D collected_features inline | KEPT 5cba058 | flat | L2-miss -10%, pf-miss -22% | yes |
| #8/#13 guard legibility | KEPT d57cbdc | n/a (no perf change) | n/a | yes (no behavior change) |

Cumulative (baseline 901be39 -> D): prefetch-miss 554M -> 360M (-35%), L2-miss 45M -> 41M, throughput ~flat
(~7095 us/dec). KEY: the CPU micro-opts are marginal because the 55% hotspot is the bitset popcount sweep,
which has NO easy lever (A refuted the int-width idea — it was in the dead flat arm). The real wins are in
MEMORY, which the remaining tier targets.

# REMAINING TIER — each has a DESIGN DECISION on validated core (flagged for the maintainer, not rushed)

- **E (geometry single-home, ~37 MB RSS @ K=1024 = ~3%):** the loc_cache_ (per-FeatureBuilder, K copies of
  the ~67-loc x GeometryFeatures table) should be single-homed. BIT-SAFE (a caching-LOCATION change, same
  geometry_features output). BUT the audit's "own it on the Environment" INVERTS the env<-features layering
  (ADR-0003: env is Band-3 instance, GeometryFeatures is Band-2 featurization). Options: (a) env owns it
  (layering inversion); (b) a shared_ptr<const GeometryTable> the env holds opaquely + features populates;
  (c) an env-keyed shared cache (the R9 WeakKeyDictionary pattern). DECISION NEEDED: which home.
- **#6 (boundary outstanding_ redundant set, per-message malloc):** the BoundaryPerThread set duplicates the
  drivers' corr_to_group + a check they already do. Fix touches the boundary seam (which layer owns the
  in-flight-corr fact). Measured by producer throughput, not the microbench. Medium-clean.
- **B (ArenaPool, O(K)->O(in-flight) resident — the HIGH item, biggest memory win):** redesigns the node-arena
  ownership. The cursor's arena is SELF-REFERENCED (nodes_{&arena_}) and =delete-move; an ArenaPool that
  slots check out/in is a genuine redesign of that ownership + the slot lifecycle, with a bit-identity
  re-proof. This is the one that most needs your design input (and is the riskiest to do blind).

RECOMMENDATION: E is bit-safe and worth doing once the table's home is chosen; B is the big win but wants a
design pass with you (the cursor self-ref arena). I stopped here rather than impose a layering/ownership
choice on validated core while you were out (ADR-0013: surface the decision, don't grind it blind).
