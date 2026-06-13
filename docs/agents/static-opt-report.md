# Static-optimization agent — report (verbatim)

> The implementation agent's complete final message, reproduced verbatim. Commission in
> `static-opt-commission.md`; full design doc at `docs/design/static-shortcuts.md`.
> Recorded per the verbatim-record discipline; not retro-edited.

---

Done. Committed `0e9b89b` on `feat/static-opt`, working tree clean, not pushed, and `env.py`/`policies.py`/`run.py` untouched (no diff). All computation ran pinned to core 3 under `timeout`, bounded counting/structure only — no solver, no global belief enumeration.

## Final report

I read the required files end to end (env.py, policies.py, run.py, chocobo_geometry.py, chocobo_instance.json, STATUS.md, all results/, both agent reports, plus the consult). The current `env.py` already uses the real 17-pair `overlaps` array for `cover_mask`, so the consult's flagged rep-point bug is fixed in this worktree — the cover sets I measured are the true disjunctive unions. Deliverable: `/home/bork/w/vdc/chocobo-staticopt/docs/design/static-shortcuts.md`.

### Structural map (all numbers filtered from the real 15,504-world array)

**Sensing clusters** (detector-cover connected components): NW {8,9,10,11,12}; the fused SE+mid {0,1,2,13,14,15} (bridged by D_1 covering {0,1,15} and D_15 covering {1,13,14,15}); N {5,6,7}; S {17,18}; δ-singletons {3},{4},{16},{19} (no region — visit-only).

**Teleport partition**: CSNE entry → {NW, N} (CSNE is 2.6 from NW centroid, 0.5 from τ_10); CSCE → {mid, SE, S, δ16, δ19}; **τ_4 serves only {3,4}** and is the most distant exit from everything else — dominated, as the project suspected.

**Indistinguishability floor**: {8,9},{11,12},{13,14},{17,18} have identical detector signatures and {3,4,16,19} have none — so detectors alone bottom out at E[120.7] surviving worlds (128× collapse); breaking a pair requires a collect.

### Ranked, quantified shortcuts

1. **Entry-NW negative collapse — 5.16× free at entry.** D_8/D_9 neg: 15,504 → 3,003 worlds (rules out all of {8,9,10,11,12}); the maintainer's lead, measured.
2. **Exact per-cluster sub-solvers — 445 reachable local beliefs total** (NW=332, others 10–36; merged SE+mid=1,320). Each is a tiny belief-MDP, exactly solvable by backward induction — trusted anchors that remove the determinization-optimism/max-bias that wrecked NMCS and ISMCTS.
3. **Exact factorization keystone**: given the per-cluster occupancy vector, worlds factor as independent within-cluster uniform subsets, sole coupling Σk=5 (verified #worlds == ∏C(size,k)). Cross-cluster coupling ≤2.8% deviation from independence. Macro state = 613 occupancy partitions.
4. **One-per-cluster sweep {D_8,D_13,D_0,D_7,D_17} — 18.7× to E[828] worlds** (near-clairvoyant from 5 cheap reads).

### Recommended decomposition

Two-level hierarchy: a **macro** layer over the multivariate-hypergeometric occupancy posterior (613 partitions, exact) choosing which cluster to probe / when to bank-and-exit, over **micro** exact per-cluster belief-MDP solvers (≤332 beliefs each), chained across the 3 teleport regions. This removes both failure modes the prior search agents hit — depth (the deep chain is now inside an exactly-solved cluster) and determinization noise (exact leaf, no sampling).

**Exactly solvable now**: every individual cluster (NW the largest at 332 local beliefs), the merged SE+mid supercluster (1,320), and the macro routing layer (613) — all in the low thousands of states where the flat global belief-MDP was unbounded.
