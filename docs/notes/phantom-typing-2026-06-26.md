# Zero-cost phantom integer-domain types — 2026-06-26

Branch `feat/tlab-phantom-counts` (off `feat/tlab-real-generators` @ `37075c3`). This is **increment 1** of
the whole-codebase mandate: phantom-type every integer domain (no arbitrarily-signed ints; every bit-width
motivated at the use site; conversions only at motivated boundaries), zero-cost, illegal domain mixing made
unrepresentable (ADR-0000 / ADR-0012 / ADR-0008).

## What landed
- `7966d2f` — the `Quantity<Tag, Rep>` SSOT machinery (`quantity.hpp`: `QuantityRep` concept, opt-in
  concept-gated `additive`/`affine` arithmetic, explicit ctor + `.value()` as the only ACLs) + domain types
  (`WorldCount`/`WorldRank`, `FaceId`/`TeleportId`/`ActionSlot`, the counts `N`/`nD`/`kW64`, wire counts)
  propagated across the C++ tree. `rth_set_bit_index` → `std::optional<WorldRank>` (the `-1` sentinel killed).
  45 files, +2241/−842.
- `23cd2a4` — repair of the one regression the audit found (below).

## Bit-identity (independently re-run on the committed artifact)
belief-sweep oracle byte-for-byte + flat-vs-bitset byte-identical · cursor-proto PASS · `gumbel_logic` PASS ·
`gumbel_precision` 144/144 · build GREEN default + `-DCHOCO_BELIEF_ZDD=ON` + throughput-lab · `-Wall -Wextra` clean.

## Zero-cost is actually a SPEEDUP (statistically significant)
Interleaved A/B (12 reps, warmup discarded, bootstrap 95% CI, core 3, nice -19), BEFORE=`37075c3`, AFTER=phantom:

| path | BEFORE | AFTER | paired Δ | 95% CI | verdict |
| --- | --- | --- | --- | --- | --- |
| cursor | 7128.6 µs | 7002.7 µs | −1.84% | [−2.05%, −1.57%] | SIGNIFICANT |
| direct | 7150.2 µs | 7035.3 µs | −1.65% | [−1.92%, −1.27%] | SIGNIFICANT |

### Mechanism (attributed, not conjectured)
Instructions FLAT (45.39e9→45.47e9) → pure IPC win (2.53→2.59). Cycles −2.10%. Against headroom: the
workload is 56.9% retiring, so 43.1% non-retiring is all that's addressable; non-retiring (stall) cycles fell
7.72e9→7.33e9 = **−5.1% of the stall budget**, retiring cycles flat. topdown: retiring +1.3pp, backend
−0.8pp; L1i-miss −13.1% (smaller binary 502K→477K), dTLB −11.7%. Hot `belief_features` disasm:
cast/extension opcodes **24→6 (−75%)**, the `movslq` int→int64 sign-extends **20→2** — the lying-int64
widenings gone; they sat on the dependency path and forced 64-bit lanes, so removing them lifted IPC.
(Note: my earlier *half-measure* uint32 with casts at every seam was ~0.4% SLOWER — coherence is the win.)

Full data: `~/w/vdc/chocobo/runs/tlab/phantom-ab-20260626/RESULTS.md` (gitignored).

## The one audit finding (out-of-frame rationalization detector), repaired
`HIDDEN-BEHAVIOR`: the `--recv-timeout-ms` sentinel rewrite collapsed `-1` (block-forever) into a
non-blocking poll (the dealer's `: -1` arm went dead), contradicting its own "bit-faithful" comment.
Resolution (`23cd2a4`): the lab is strictly **always-bounded** — restoring block-forever would reintroduce
the permanent-hang-on-dead-server the lab's ADR-0002 discipline exists to prevent. Negative timeout now
loud-rejects; `0`=non-blocking; `recv_timeout_ms` returned to a plain `Milliseconds` (the dead absence state
removed). The audit found NO cast-arounds, NO domain weakening across ~2200 lines — verdict "overwhelmingly
clean", 1 violation.

## Still outstanding (toward the full mandate)
- Complete the scope: `count_`/`cached_count_` members + the inventory-catalogued domains not yet propagated
  (env public API still raw `int`). The **DMZ lint** (capstone) will measure completeness objectively.
- The **loop-modernization sweep**: generator-fed `for(T i=0;i<N;i++)` → range-for / `std::views`; manual
  index only where consumed, phantom-typed.
- The **DMZ lint**: clang-tidy/clang-query gate so raw int/long/char live only in a designated boundary DMZ.

Public Domain (The Unlicense).
