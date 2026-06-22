"""
tools/analysis/OpenTURNS/model_capacity.py
==========================================

Design-A: the CAPACITY / BOTTLENECK lower bound on the achievable leaf-eval throughput
(dps), as one model module the generic `NeymanDriver` consumes (ADR-0012 P1/P2 — the
driver owns no model; this module owns this one).

FRAMING. The leaf-eval transport is a closed pipeline of stages; the achievable optimum
T* >= min over stages of each stage's FLAT-OUT, fully-fed capacity (the rate a stage
sustains in isolation with enough in-flight leaves that it never starves), with all
COORDINATION losses (RTT idle, convoy, single-thread serial stalls, cold-JIT) excluded —
because those are exactly the losses the optimum is permitted to engineer away (more
in-flight leaves, overlap, bucketing). The binding stage sets the bound. Three stages:

  1. GENERATION  : n_gen * g_core / LPD            (the 3 producer cores' aggregate)
  2. SERVE       : B_op / ((iota + slope*B_op + tau_io)*1e-6) / LPD   (the MLP forward)
  3. TRANSPORT   : 1 / (LPD * tmsg_us_leaf * 1e-6)  (ZMQ multipart, per-leaf amortized)

  f = min(GENERATION, SERVE, TRANSPORT)

REFINEMENTS over the first draft (per Critique A — fixing the loosening flaws):

  * iota_us is JAX-FORWARD ONLY. The 94.58us staged intercept is dispatch_floor (68.84) +
    output_pull (9.14) + input (5.52) + residual — it contains NO ZMQ drain/recv/scatter/
    poll (those are in NO bench cited; SYNTHESIS v2 §3.3 has them serial BETWEEN forwards
    on the single-threaded server). So a SEPARATE Stage-4 term `tau_io_us` (server-side
    per-forward drain+decode+encode+scatter) is ADDED into the serve denominator. It is
    UNMEASURED, sits in the binding stage, and is therefore the kind of term the Neyman
    allocator funds first. (Folding it into iota, as the first draft did, was wrong: the
    staging bench provably contains none of it.)

  * SERVE operates at a FULL BUCKET. The deployed servers PAD every forward; the
    production base pads to max_batch (pad-to-max, wasteful on partial drains), but the
    ADOPTED design (adapter.md §1, survey KataGo) BUCKETS, and the measured GLOBAL MAX
    (analysis_clean.txt: 233825 leaves/s, rows/fwd=511.5, pad=0.00 -> 468 dps) is reached
    at a FULL bucket where pad ~= real. So the serve capacity is computed at a FULL bucket
    B_op (real ~= pad), where cost = iota + slope*B_op is the right model — NOT the
    pad-to-max waste (which would UNDER-state throughput badly: real=64 padded to 256 is
    only ~107 dps), and NOT the B->inf asymptote (~463 dps, UNREACHABLE under the
    max_batch cap). Throughput vs B is a SAWTOOTH (Critique B): it peaks at full buckets
    (full-64 ~345, full-256 ~427, full-512 ~444 dps) and DROPS just past a bucket edge,
    so "larger B is always conservative" is false — B_op is held at a sustainable full
    bucket, not pushed past an edge.

  * GENERATION honesty. n_gen*g_core uses the MEASURED C++ gen rate (76k leaves/s/core,
    4.0x linear core scaling, adapter.md §2 line 93), giving a clean 3x = 456 dps. The
    ~1.9x ceiling in CLAUDE.md is the Python-ExIt substrate (a DIFFERENT subsystem), so
    it does NOT bind the C++ path; but `report_alt_producer_caps()` below also reports the
    1.9x worst case (289 dps) so the bound is honest about the unproven scaling.

WHY A LOWER BOUND. f = min of each stage's CONSERVATIVE flat-out capacity, and each
Cap_i is computed below the stage's true ceiling (gen with no overlap credit; serve at a
FIXED sustainable bucket rather than the unreachable asymptote, WITH the extra tau_io
cost added; transport over-charged per-leaf yet still non-binding). The load-bearing
discipline: NO coordination loss is put INTO any stage capacity — those are the losses
the optimum removes. The bound is therefore a floor on the OPTIMUM, contingent on the
optimum (a) reaching the full-bucket feed (the §7 measured curve climbs there only at
high N) and (b) removing the single-threaded drain/scatter serialization beyond the
tau_io already charged. It is a BENCH-dps bound: every input rate is a sole-workload
bench figure the project flags as reading higher than the closed-loop e2e (adapter.md
§7), so the production-e2e optimum is correspondingly lower.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import leaf_eval_grounding as G  # noqa: E402
from neyman_driver import NeymanDriver  # noqa: E402

# Input order is the single home of the model's signature (used by both `throughput_jax`
# and the numpy `throughput_numpy` so they cannot drift — P1).
INPUT_NAMES = ["g_core", "n_gen", "LPD", "iota_us", "slope_us", "tau_io_us", "B_op",
               "tmsg_us_leaf"]

def throughput_numpy(x: dict[str, float]) -> float:
    """The numpy-only evaluation of f (the fallback path, and the
    cross-check that the formula evaluates to the same number). SAME formula as
    throughput_jax — kept in lockstep by construction."""
    gen = x["n_gen"] * x["g_core"] / x["LPD"]
    fwd_us = x["iota_us"] + x["slope_us"] * x["B_op"] + x["tau_io_us"]
    serve = (x["B_op"] / (fwd_us * 1e-6)) / x["LPD"]
    transport = 1.0 / (x["LPD"] * x["tmsg_us_leaf"] * 1e-6)
    return float(min(gen, serve, transport))


def throughput_jax(x: Any) -> Any:
    """The single JAX-traceable throughput f (x ordered by INPUT_NAMES) — the OT→JAX migration's one home
    for f (§5): `jax.grad(throughput_jax)` is the gradient (analytic, exact-through-`min()`; the arm-tie is
    handled by alloc.kink, not the linearization), evaluating identically to `throughput_numpy` (pinned in
    tests/test_jax_f_equivalence.py). The model's single f the driver consumes (the OT string THROUGHPUT_EXPR is retired; the numpy twin throughput_numpy retires with the numpy fallback in migration J4)."""
    from alloc.jax_backend import jnp
    g_core, n_gen, LPD, iota_us, slope_us, tau_io_us, B_op, tmsg_us_leaf = x
    gen = n_gen * g_core / LPD
    fwd_us = iota_us + slope_us * B_op + tau_io_us
    serve = (B_op / (fwd_us * 1e-6)) / LPD
    transport = 1.0 / (LPD * tmsg_us_leaf * 1e-6)
    return jnp.minimum(jnp.minimum(gen, serve), transport)


# Per-input grounded (mean, sigma, cost) — single-homed in leaf_eval_grounding.
_INPUTS: list[G.Grounded] = [
    G.GEN_PER_CORE_LEAVES,   # g_core
    G.N_GEN_CORES,           # n_gen
    G.LEAVES_PER_DECISION,   # LPD
    G.SERVE_INTERCEPT_US,    # iota_us
    G.SERVE_SLOPE_US,        # slope_us
    G.SERVE_IO_US,           # tau_io_us  (the Stage-4 term, top Neyman priority)
    G.SERVE_FULL_BUCKET,     # B_op
    G.MSG_PER_LEAF_US,       # tmsg_us_leaf
]


# Per-input maps keyed on THIS model's INPUT_NAMES (here the grounded `.name`s already
# match INPUT_NAMES, but keying on the zip keeps it robust and uniform with Design-B — P1).
SIGMAS: dict[str, float] = {nm: g.sigma for nm, g in zip(INPUT_NAMES, _INPUTS)}
COSTS: dict[str, float] = {nm: g.cost for nm, g in zip(INPUT_NAMES, _INPUTS)}
NEEDS_MEASUREMENT: dict[str, bool] = {nm: g.needs_measurement
                                      for nm, g in zip(INPUT_NAMES, _INPUTS)}


def initial_point() -> dict[str, float]:
    """The grounded mean point f is first evaluated at (mu_hat before any sampling)."""
    return {nm: g.mean for nm, g in zip(INPUT_NAMES, _INPUTS)}


def build_driver(tolerance: float = 5.0) -> tuple[NeymanDriver, dict[str, float]]:
    """Factory: a configured `NeymanDriver` (over the JAX `f` (`throughput_jax`), grounded costs) + the
    initial point estimate. `tolerance` is the target CI half-width on E[f] in dps."""
    f = throughput_jax  # the driver consumes the JAX-traceable f directly (OT→JAX migration, §5)
    costs = [g.cost for g in _INPUTS]
    driver = NeymanDriver(
        f, costs=costs, tolerance=tolerance, names=INPUT_NAMES,
        confidence=0.95, growth_cap=3.0,
    )
    return driver, initial_point()


def stage_capacities(x: dict[str, float]) -> dict[str, float]:
    """The three stage capacities (dps) at point x — so a caller can see which binds and
    by how much. The binding stage is the min."""
    gen = x["n_gen"] * x["g_core"] / x["LPD"]
    fwd_us = x["iota_us"] + x["slope_us"] * x["B_op"] + x["tau_io_us"]
    serve = (x["B_op"] / (fwd_us * 1e-6)) / x["LPD"]
    transport = 1.0 / (x["LPD"] * x["tmsg_us_leaf"] * 1e-6)
    return {"GENERATION": gen, "SERVE": serve, "TRANSPORT": transport}


def report_alt_producer_caps() -> dict[str, float]:
    """The producer ceiling under both core-scaling assumptions (honesty about the 4.0x
    vs 1.9x question a lower bound must surface). 4.0x is the MEASURED C++ figure; 1.9x is
    the Python-ExIt worst case, reported in case the deployed path reintroduces that
    contention."""
    gc, lpd = G.GEN_PER_CORE_LEAVES.mean, G.LEAVES_PER_DECISION.mean
    return {
        "gen_4.0x_linear_dps": 3 * gc / lpd,        # = 456, the measured C++ ceiling
        "gen_1.9x_exit_dps": 1.9 * gc / lpd,        # = 289, the worst-case if ExIt binds
    }


def serve_sawtooth(real: int, buckets=(64, 256, 512), max_batch=512) -> float:
    """The serve dps at `real` rows under BUCKETING (snap up to the smallest bucket >=
    real; past the top bucket, run unpadded at width=real) — the SAWTOOTH Critique B
    named, INCLUDING the tau_io term. Lets the runner show the bound peaks at full buckets
    and drops just past an edge (so a larger B is not unconditionally conservative)."""
    pad = next((b for b in buckets if b >= real), real)
    if real > max_batch:
        pad = real
    iota, slope, tau = G.SERVE_INTERCEPT_US.mean, G.SERVE_SLOPE_US.mean, G.SERVE_IO_US.mean
    fwd_us = iota + slope * pad + tau
    return (real / (fwd_us * 1e-6)) / G.LEAVES_PER_DECISION.mean


if __name__ == "__main__":
    x0 = initial_point()
    print("Design-A capacity model — initial point:", x0)
    print("stage capacities (dps):", {k: round(v, 1) for k, v in stage_capacities(x0).items()})
    print("f(mu_hat) =", round(throughput_numpy(x0), 1), "dps")
    print("alt producer caps:", {k: round(v, 1) for k, v in report_alt_producer_caps().items()})
