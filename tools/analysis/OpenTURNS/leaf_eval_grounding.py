"""
tools/analysis/OpenTURNS/leaf_eval_grounding.py
===============================================

Single-home (ADR-0012 P1) for the MEASURED physical quantities the two leaf-eval
throughput lower-bound models (`model_capacity.py`, `model_cycletime.py`) draw on, so
the grounded numbers have ONE definition both models import — never two hand-copied
literals that must agree. Each constant carries its provenance in a comment; a value
that is a DESIGN PIN (not a fresh measurement) or that needs a fresh sole-workload
benchmark is labelled as such, per the claims-measured-vs-interpreted discipline.

This module imports nothing but the standard library + numpy, so it is import-clean
with or without openturns (the numpy-only fallback path needs it).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grounded:
    """One grounded quantity: a mean, a 1-sigma spread, a relative per-sample benchmark
    cost, and whether it still needs a fresh SOLE-WORKLOAD measurement (the Neyman loop
    ranks these). `provenance` is the file the number was read from."""
    name: str
    mean: float
    sigma: float
    cost: float
    unit: str
    provenance: str
    needs_measurement: bool = False


# --- Server forward affine fit (the ONE measured serve cost model) -------------------
# ~/w/vdc/chocobo/bench/run_microbatch_staging/results_nopad.json, fits.staged:
#   intercept_us = 94.582 (R2=0.998), slope_us_per_row = 4.3156 (R2=0.998).
# This is the PRODUCTION run_microbatch STAGED path (params staged device-resident once,
# ADR-0012 2026-06-20 follow-on). pad_to=0 (no-pad) — so the fit is time = intercept +
# slope * REAL_ROWS, valid for a FULL bucket where pad ~= real (the optimum's operating
# point; see model docstrings). The 'current' (un-staged) intercept is 173.9us — using it
# would only LOWER the serve stage, the more pessimistic alt; the staged number is the
# right one for the optimized boundary.
SERVE_INTERCEPT_US = Grounded(
    name="iota_us", mean=94.58, sigma=12.0, cost=6.0, unit="us",
    provenance="run_microbatch_staging/results_nopad.json fits.staged.intercept_us",
)
SERVE_SLOPE_US = Grounded(
    name="slope_us", mean=4.317, sigma=0.5, cost=6.0, unit="us/row",
    provenance="run_microbatch_staging/results_nopad.json fits.staged.slope_us_per_row",
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
    needs_measurement=True,
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
    needs_measurement=True,
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
    needs_measurement=True,   # a fresh SOLE-WORKLOAD per-core read (eval mocked) would tighten it
)
GEN_PER_CORE_DPS = Grounded(
    name="R_gen", mean=152.0, sigma=8.0, cost=30.0, unit="decisions/s/core",
    provenance="adapter.md §2 line 93 'MEASURED gen 152 dps/core, 4.0x linear'",
    needs_measurement=True,
)
N_GEN_CORES = Grounded(
    name="n_gen", mean=3.0, sigma=0.05, cost=0.5, unit="cores",
    provenance="adapter.md §6 M3 1:3 pinning; CLAUDE.md host (4-vCPU, isolcpus 1-3)",
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
    needs_measurement=True,
)

# --- Per-leaf-amortized message-passing cost (the transport stage; non-binding) -------
# cpp/include/chocofarm/inference_wire.hpp: pure-memcpy [ver][B][in_dim][f32xB*in_dim]
# codec (no parse-copy); request ~968B (241 f32), response ~264B (66 f32) per leaf. Over
# a coalesced S-leaf frame the per-leaf framing share is ~us, far below the per-forward
# budget, so transport never binds. Placeholder 1.0us/leaf is a deliberate over-charge
# (still leaves transport >> the binding stages). Provably non-binding by a wide margin,
# so it ranks LAST for the Neyman allocator.
MSG_PER_LEAF_US = Grounded(
    name="tmsg_us_leaf", mean=1.0, sigma=0.5, cost=2.0, unit="us/leaf",
    provenance="inference_wire.hpp pure-memcpy codec (deliberate over-charge; non-binding)",
    needs_measurement=True,
)

# --- Reference points (NOT targets — re-derive, do not anchor) ------------------------
# The empirical ~203 dps plateau is a USER-supplied reference for ONE config family on
# the current harness; it is NOT grounded in any readable repo file (the only repo '203'
# hits are unrelated). The nearest MEASURED production-path numbers are below.
REF_PLATEAU_DPS = 203.0       # user-supplied empirical reference (one config family)
REF_PRIOR_MODEL_DPS = 456.0   # overcommit_sweep.py:307 BARE LITERAL model_optimistic_dps;
                              # adapter.md §6 calls it "an upper bound" the bench fell short of
# MEASURED production-path anchors (analysis_clean.txt + adapter.md §5/§7):
REF_STRICT_BARRIER_DPS_PER_CORE = 49.0     # analysis_clean.txt strict-barrier ref
REF_GREEDY_ASYNC_DPS_PER_CORE = 37.0       # analysis_clean.txt greedy-async ref
REF_GLOBAL_MAX_DPS = 468.0                 # analysis_clean.txt GLOBAL MAX (full bucket, pad=0)
REF_SERVE_CEILING_DPS = (380.0, 528.0)     # 190k..264k leaves/s / 500
REF_HIGH_N_BENCH_DPS = 189.0               # adapter.md §7 N=9 (BENCH, over-reads e2e)
REF_STAGEB_1THREAD_DPS_PER_CORE = 72.65    # adapter.md §5 arm3 1-thread (e2e-ish)
