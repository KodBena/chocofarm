"""
tools/analysis/leaf_eval_bound/contract/references.py

The REF_* display/comparison ANCHORS (§2.2 split): NOT model inputs -- re-derive, do not anchor. The
empirical plateau, the serve ceilings, the bench over-reads. Consumed only by the runner headers and
the variant models' ref_plateau_dps(); never by the bound itself.

Public Domain (The Unlicense).
"""

# --- Reference points (NOT targets — re-derive, do not anchor) ------------------------
# The empirical ~203 dps plateau is a USER-supplied reference for ONE config family on
# the current harness; it is NOT grounded in any readable repo file (the only repo '203'
# hits are unrelated). The nearest MEASURED production-path numbers are below.
REF_PLATEAU_DPS = 203.0       # user-supplied empirical reference (one config family)
REF_PRIOR_MODEL_DPS = 456.0   # overcommit_sweep.py:307 BARE LITERAL model_optimistic_dps (overcommit_sweep
                              # MOLTED 2026-06-25 — git history); adapter.md §6 calls it "an upper bound"
                              # the bench fell short of
# MEASURED production-path anchors (analysis_clean.txt + adapter.md §5/§7):
REF_STRICT_BARRIER_DPS_PER_CORE = 49.0     # analysis_clean.txt strict-barrier ref
REF_GREEDY_ASYNC_DPS_PER_CORE = 37.0       # analysis_clean.txt greedy-async ref
REF_GLOBAL_MAX_DPS = 468.0                 # analysis_clean.txt GLOBAL MAX (full bucket, pad=0)
REF_SERVE_CEILING_DPS = (380.0, 528.0)     # 190k..264k leaves/s / 500
REF_HIGH_N_BENCH_DPS = 189.0               # adapter.md §7 N=9 (BENCH, over-reads e2e)
REF_STAGEB_1THREAD_DPS_PER_CORE = 72.65    # adapter.md §5 arm3 1-thread (e2e-ish)
