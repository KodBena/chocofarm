"""
tools/analysis/leaf_eval_bound/contract/grounding.py

Single-home (ADR-0012 P1) for the MEASURED Band-3 physical-quantity CONSTANTS the leaf-eval models draw
on. The vocabulary lives in `grounded_types` (Grounded/Estimability); the display anchors in `references`
(REF_*). Import-clean (stdlib only) -- the grounding SSOT every model + bench reads.

Public Domain (The Unlicense).
"""

from __future__ import annotations

from leaf_eval_bound.contract.grounded_types import Estimability, Grounded


# --- Server forward affine fit (the ONE measured serve cost model) -------------------
# ~/w/vdc/chocobo/bench/run_microbatch_staging/results_nopad.json, fits.staged:
#   intercept_us = 94.582 (R2=0.998), slope_us_per_row = 4.3156 (R2=0.998).
# This is the PRODUCTION run_microbatch STAGED path (params staged device-resident once,
# ADR-0012 2026-06-20 follow-on). pad_to=0 (no-pad) — so the fit is time = intercept +
# slope * REAL_ROWS, valid for a FULL bucket where pad ~= real (the optimum's operating
# point; see model docstrings). The 'current' (un-staged) intercept is 173.9us — using it
# would only LOWER the serve stage, the more pessimistic alt; the staged number is the
# right one for the optimized boundary.
#
# needs_measurement=True on BOTH (MEASURED quantities, NOT true constants — `constant` stays
# its default False): iota (the intercept) and slope/t_row are the SAME k=2 staged-forward
# OLS fit that bench_iota.py / bench_t_row.py now RUN LIVE (estimators.fit_estimate over the
# production `run_microbatch` forward at a width sweep -> a SHRINKABLE RegressionLaw Estimate
# with the −0.81 slope/intercept off-diagonal, the SE from resid_var + the x-design — NOT the
# 12.0us/0.5us hand-literals below, and NOT the stored JSON, which carries only intercept/slope/
# r2 with NO covariance). These mean/sigma are the v1 SEED — the DISTRUST/trust=False fallback
# prior (declared spread, NORMAL) bench_iota/bench_t_row.get_seed() return; the LIVE measure()
# path is the RegressionLaw whose variance authority is its own cov. So this is the same
# measured-but-punted reclassification the R_gen / g_core / LPD / tmsg flips landed (the design's
# §3 REGRESSION-fit row + the needs_measurement/manifest-trusted distinction): the flag now
# SINGLE-HOMES the "needs a sole-workload run" semantics across BOTH paths — the static models
# (model_capacity/model_cycletime, which read Grounded.needs_measurement) and the manifest models
# (model_zmq_baseline/model_cpp_inproc_port, which derive needs_measurement = not trusted) now
# both classify iota/slope as NEEDS-SOLE-WORKLOAD, instead of the prior path-dependent split that
# told the static-path operator NOT to measure a runnable fit (ADR-0008 classification; ADR-0012
# P1 derive-don't-duplicate / P8 typed-signature-is-SSOT). NB it is a RegressionLaw, NOT the
# QuantileLaw the R_gen flip used: a fit is leverage-floored (its marginal is ~0 absent a
# per_point_var weighted-LS SE — the design's §4.3/§7.E conservative posture), so funding it means
# WIDENING the x-design / RUNNING its bench to flip it trusted, not pouring iters into the median.
SERVE_INTERCEPT_US = Grounded(
    name="iota_us", mean=94.58, sigma=12.0, cost=6.0, unit="us",
    provenance="run_microbatch_staging/results_nopad.json fits.staged.intercept_us "
               "(v1 seed; MEASURED live by bench_iota as the k=2 staged-fit intercept)",
    estimability=Estimability.MEASURED, module="bench_iota",
)
SERVE_SLOPE_US = Grounded(
    name="slope_us", mean=4.317, sigma=0.5, cost=6.0, unit="us/row",
    provenance="run_microbatch_staging/results_nopad.json fits.staged.slope_us_per_row "
               "(v1 seed; MEASURED live by bench_t_row as the k=2 staged-fit slope)",
    estimability=Estimability.MEASURED, module="bench_t_row",
)

# Decomposition of the JAX-forward fixed cost (mlp_lowlatency/results.json decomposition):
#   dispatch_floor_us = 68.84 (irreducible pjit/XLA), params_transfer = 45.23 (consolidated
#   away by staging), input_transfer = 5.52, output_pull = 9.14, current_intercept = 128.72.
# CRITIQUE-A CORRECTION: the 94.58 staged intercept is JAX-FORWARD ONLY (dispatch floor +
# output pull + input + residual). It contains NO ZMQ drain/recv, NO scatter/send, NO poll
# wakeup — those run SERIALLY between forwards on the single-threaded server (SYNTHESIS
# v2 §3.3) and are in NO bench cited. They are the separate tau_io term below.
DISPATCH_FLOOR_US = 68.84    # mlp_lowlatency decomposition.dispatch_floor_us (informational)

# --- Server-side per-forward TRANSPORT/DRAIN serial cost (the missing Stage-4 term) ---
# NOT measured in any read bench (Critique A's "missing term"). The single-threaded server
# serializes _drain (recv_multipart x ~T) + decode (x T) + encode (x T) + _scatter
# (send_multipart x T) BETWEEN forwards (inference_server.py _drain/_scatter; SYNTHESIS
# §3.3 "drain k+1 cannot begin until scatter k completes"). At ~2-10us per ZMQ I/O over
# ~T coalesced messages, a plausible per-forward cost is ~10-40us. Held as a LIVE input
# with wide sigma and FLAGGED needs-measurement — it sits in the BINDING (serve) stage so
# the Neyman allocator ranks it high. A sole-workload microbench of the serve loop
# (recv x T + decode x T + encode x T + send x T, no forward) would measure it directly.
SERVE_IO_US = Grounded(
    name="tau_io_us", mean=20.0, sigma=12.0, cost=8.0, unit="us",
    provenance="UNMEASURED — SYNTHESIS v2 §3.3 serial drain/scatter; inference_server.py",
    estimability=Estimability.MEASURED, module="bench_tau_io",
)

# --- Leaves per recorded decision (the unit conversion to dps) -----------------------
# analysis_clean.txt reference block: "@ 500 leaves/decision" divisor; the gen-ceiling
# row is "152 dps/core (76k leaves/s)". CRITIQUE CORRECTION: 76000/152 = 500 is a
# TAUTOLOGY (stage_a_analyze defines 76000 := 152*500), NOT an independent cross-check.
# So LPD=500 is an explicitly-labelled DESIGN PIN (a sims256/m24 Gumbel tree's aggregate
# distinct-node count), not a measured per-decision histogram. A per-decision leaf-count
# histogram from one instrumented run would ground it; flagged needs-measurement.
LEAVES_PER_DECISION = Grounded(
    name="LPD", mean=500.0, sigma=25.0, cost=2.0, unit="leaves/decision",
    provenance="analysis_clean.txt '@ 500 leaves/decision' (DESIGN PIN, not a histogram)",
    estimability=Estimability.MEASURED, module="bench_lpd",
)

# --- Per-core generation rate (the producer ceiling input) ---------------------------
# analysis_clean.txt: "gen-ceiling: 152 dps/core (76k leaves/s)". cpp-eval-transport-
# adapter.md §2 line 93: "MEASURED — gen 152 dps/core (76k leaves/s), 4.0x linear core
# scaling". So the C++ gen core scales 4.0x-LINEAR (a clean 3x for 3 cores). The ~1.9x
# ceiling in CLAUDE.md is the Python-ExIt parallel substrate (a DIFFERENT subsystem),
# NOT the C++ transport path — so n_gen=3 deserves a clean 3x here. (A genuinely
# conservative bound also reports the 1.9x worst case; see model_capacity.py.)
GEN_PER_CORE_LEAVES = Grounded(
    name="g_core", mean=76000.0, sigma=9000.0, cost=3.0, unit="leaves/s/core",
    provenance="analysis_clean.txt gen-ceiling; adapter.md §2 line 93 (MEASURED, 4.0x linear)",
    estimability=Estimability.MEASURED, module="bench_g_core",   # a SOLE-WORKLOAD per-core read would tighten it
)
GEN_PER_CORE_DPS = Grounded(
    name="R_gen", mean=152.0, sigma=8.0, cost=30.0, unit="decisions/s/core",
    provenance="adapter.md §2 line 93 'MEASURED gen 152 dps/core, 4.0x linear'",
    estimability=Estimability.MEASURED, module="bench_r_gen",
)
N_GEN_CORES = Grounded(
    name="n_gen", mean=3.0, sigma=0.05, cost=0.5, unit="cores",
    provenance="adapter.md §6 M3 1:3 pinning; CLAUDE.md host (4-vCPU, isolcpus 1-3)",
    estimability=Estimability.CONSTANT, module="bench_n_gen",   # a TRUE CONSTANT: 3 generator cores is a
                     # layout/pinning fact, not a measured spread — DEGENERATE, ~0 bound contribution
                     # (§3). The σ=0.05 is a display placeholder on an integer core count, never CI-bearing.
)

# --- The server's sustained FULL-bucket operating point (rows/forward) ----------------
# The achievable optimum runs at a FULL bucket where pad ~= real (analysis_clean.txt
# GLOBAL MAX cell: 233825 leaves/s, rows/fwd=511.5, pad=0.00 -> 468 dps). The serve curve
# is a SAWTOOTH real/(iota+slope*bucket(real)) (Critique B), maximized at full buckets:
# full-64 -> 345 dps, full-256 -> 427 dps, full-512 -> 444 dps. B_op is the real row count
# at a full bucket. Default 256 (a value the sweeps reach: rows/fwd 54 -> 192 -> 511).
# FLAG needs-measurement: the steady-state full-bucket B the optimum sustains under a fed
# producer set is the single quantity that most moves the serve stage (top Neyman target).
SERVE_FULL_BUCKET = Grounded(
    name="B_op", mean=256.0, sigma=64.0, cost=4.0, unit="rows/forward",
    provenance="analysis_clean.txt GLOBAL MAX rows/fwd=511.5 pad=0; inference_server max_batch=256",
    estimability=Estimability.PRIOR, module="bench_b_op",
)

# --- Per-leaf-amortized message-passing cost (the transport stage; non-binding) -------
# cpp/include/chocofarm/inference_wire.hpp: pure-memcpy [ver][B][in_dim][f32xB*in_dim]
# codec (no parse-copy); request ~968B (241 f32), response ~264B (66 f32) per leaf. Over
# a coalesced S-leaf frame the per-leaf framing share is ~us, far below the per-forward
# budget, so transport never binds. The 1.0us/leaf here is the v1 SEED — a deliberate
# over-charge (still leaves transport >> the binding stages) on the DISTRUST/trust=False
# seed path. It is provably non-binding by a wide margin, so it ranks LAST for the Neyman
# allocator. needs_measurement=True (a MEASURED quantity, NOT a true constant — `constant`
# stays its default False): bench_tmsg.py / bench_zmq_baseline_tmsg_us_leaf.py now RUN the
# live inference_wire codec (encode_request + decode_response over a coalesced S-leaf frame,
# /S, windowed) and return a SHRINKABLE median QuantileLaw — the ADR-0008/ADR-0012 P8
# reclassification mirroring R_gen (this seed stays the seed-path fallback only; the
# same-quantity-class non-binding sibling bench_cpp_inproc_port_tmsg_us_leaf is constructed
# shrinkable too, so non-binding is a RANKING fact, not an un-measurability fact).
MSG_PER_LEAF_US = Grounded(
    name="tmsg_us_leaf", mean=1.0, sigma=0.5, cost=2.0, unit="us/leaf",
    provenance="inference_wire codec per-leaf framing (v1 seed 1.0us deliberate over-charge; "
               "MEASURED by bench_tmsg over the live codec; non-binding, ranks LAST)",
    estimability=Estimability.MEASURED, module="bench_tmsg",
)
