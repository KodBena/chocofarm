"""
tools/analysis/leaf_eval_bound/model_cycletime.py
===========================================

Design-B: the CYCLE-TIME / LATENCY lower bound on the achievable leaf-eval throughput
(dps), as one model module the generic `AllocationDriver` consumes (ADR-0012 P1/P2). It
derives the SAME bound as Design-A by a different route — composing the single
serialized server's per-forward CYCLE from named latency terms rather than fitting a
leaves/s curve — so the two numbers agreeing is the cross-check, and every transport
cost (drain, msg, ctx) is a NAMED additive term the Neyman allocator can rank.

THE SPINE (SYNTHESIS v2 §0, §3.3, §6, formally verified): per-thread in-flight message
depth is identically 1 for all (N,T,D); the server is single-threaded so forward k+1
cannot begin until scatter k completes ("this is the coalescing engine"); at saturation
throughput is pinned at rows_per_forward / cycle_time (regime R2). So the server is ONE
serial worker turning a per-forward CYCLE:

    cycle_us(B) = T_disp + T_io + B_eff*t_row
                  where B_eff = pad(B) is the COMPILED (bucket) width, and at a FULL
                  bucket pad(B) ~= B (the optimum's operating point).
    dps = min( N_gen*R_gen ,  1e6 * B / (cycle_us(B) * L) )

REFINEMENTS over the first draft (per Critique B — fixing the INVALIDATING flaws):

  * PAD-AWARE service time. The first draft charged compute as B*t_row with B = real
    rows, but the deployed servers PAD: the production base pads to max_batch (constant
    in real B), the StageA bench snaps UP to a bucket {64,256,512}. So the forward cost is
    pad(B)*t_row, CONSTANT in real B for B <= pad. Throughput is therefore B/cycle, and
    vs B it is a SAWTOOTH (drops at each bucket edge), NOT monotone — so "B=192 is
    conservatively below 512" is unsound (real=224 into bucket-512 gives ~194 dps, WORSE
    than real=192 into bucket-256's ~320). This model evaluates at a FULL BUCKET (B at a
    bucket's own top, pad ~= B), the achievable peak, and exposes `serve_sawtooth()` so
    the runner can SHOW the non-monotonicity rather than assert monotonicity.

  * T_disp is the JAX-forward floor only; transport is SEPARATE. T_disp = 68.84us is the
    pure pjit/XLA dispatch floor (mlp_lowlatency). The server-side per-forward drain +
    decode + encode + scatter (recv/send over ~T coalesced messages) runs SERIALLY
    between forwards (SYNTHESIS §3.3) and is in NO bench — it is the named `T_io` term
    (Critique A's missing Stage-4; subsumes the first draft's separate msg+ctx, which
    Critique B flagged as double-counted/mis-scaled against the run_microbatch path).
    T_io is UNMEASURED, sits in the binding cycle, and is the top Neyman target.

  * PRODUCER ceiling takes the WORSE of both scalings. N_gen*R_gen uses gen-only 4.0x
    (3*152=456); but `producer_caps()` also reports the 1.9x Python-ExIt worst case
    (289). A lower bound must surface the worse case absent evidence the C++ path escapes
    the contention.

  * DROPPED claims the record refuted: the "~9us pipelining slack left on the table" was
    REFUTED as a local lever (ADR-0012 2026-06-20: io-crossing bench output_delta=-3.84us,
    donate=-1.43us — ~0 recoverable), so it is NOT credited as conservative margin. And
    the "independently re-derives ~456/457" framing is dropped: 456 is a BARE LITERAL in
    overcommit_sweep.py:307 that adapter.md §6 calls "an upper bound" the bench fell short
    of (the §7 measured high-N curve tops at ~189 bench dps); this model lands at the
    full-bucket cycle, and where it meets 456 that is the GENERATION ceiling coinciding,
    not a confirmation of the prior literal.

  * The B->inf asymptote (1e6/t_row/L ~= 463 dps) is reported SEPARATELY as UNREACHABLE
    (max_batch caps B at <=512), never as the bound.

CONTRAST VARIANT (computed, not advocated): the C++ in-process queue-PORT removes the
wire — T_io ~= 0 and t_row = the BARE-forward 3.092us/row (no run_microbatch concat) over
a NO-PAD full batch — isolating how much of the gap is transport vs irreducible XLA. See
`inproc_port_contrast()`.

WHY A LOWER BOUND. Every cost is the REAL staged run_microbatch fixed cost + slope (>=
the bare-forward floor), T_io is ADDED on top, B is a FULL bucket (not the unreachable
asymptote), and the producer ceiling is a hard min. It is a BENCH-dps bound (the input
rates read higher than the closed-loop e2e — adapter.md §7). It is contingent on the
optimum reaching full-bucket feed (high N) and on the single-threaded serialization being
no worse than the T_io charged.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from leaf_eval_bound.contract import grounding as G  # noqa: E402
from leaf_eval_bound.alloc.driver import AllocationDriver  # noqa: E402

# Single home of the model signature (`throughput_jax` + numpy `throughput_numpy` share it — P1).
INPUT_NAMES = ["N_gen", "R_gen", "B", "T_disp", "T_io", "t_row", "L"]

def throughput_numpy(x: dict[str, float]) -> float:
    """Dict-keyed convenience eval of f, DERIVED from the single JAX home `throughput_jax`
    (F4 / §5 -- one formula now, so the numpy view cannot drift from it). Orders x by
    INPUT_NAMES and evaluates the one f as a float."""
    return float(throughput_jax([x[nm] for nm in INPUT_NAMES]))


def throughput_jax(x: Any) -> Any:
    """The single JAX-traceable throughput f (x ordered by INPUT_NAMES) — the OT→JAX migration's one home
    for f (§5): `jax.grad(throughput_jax)` is the gradient (analytic, exact-through-`min()`; the arm-tie is
    handled by alloc.kink, not the linearization), evaluating identically to `throughput_numpy` (pinned in
    tests/test_jax_f_equivalence.py). The model's single f the driver consumes (the OT string THROUGHPUT_EXPR is retired; the numpy convenience `throughput_numpy` is DERIVED from this single home (F4/§5), not a hand-written twin)."""
    from leaf_eval_bound.alloc.jax_backend import jnp
    N_gen, R_gen, B, T_disp, T_io, t_row, L = x
    cycle_us = T_disp + T_io + B * t_row
    serve = 1e6 * B / (cycle_us * L)
    producer = N_gen * R_gen
    return jnp.minimum(producer, serve)


# Per-input grounded (mean, sigma, cost). T_disp/t_row are derived from the JSONs;
# R_gen/L/N_gen/B from the grounding module. T_io is the named, unmeasured Stage-4 term.
_T_DISP = G.Grounded(
    name="T_disp", mean=G.DISPATCH_FLOOR_US, sigma=2.0, cost=1.0, unit="us",
    provenance="mlp_lowlatency/results.json decomposition.dispatch_floor_us (68.84, R2~0.997)",
    # MEASURED (RCA fix #1): bench_t_disp runs a live RegressionLaw fit (a shrinkable body), so the
    # single-home estimability axis classifies it MEASURED — closing the same path-dependent split the
    # _T_ROW note below already hand-patched for t_row (it had been left at the defaulted, un-measured pin).
    estimability=G.Estimability.MEASURED, module="bench_t_disp",
)
# t_row IS the staged-fit slope (G.SERVE_SLOPE_US): its mean AND its estimability (hence the derived
# needs_measurement) DERIVE from that one SSOT (P1 single-home / ADR-0008; RCA fix #1), so the slope
# classifies IDENTICALLY here and on model_capacity (which reads G.SERVE_SLOPE_US directly). Re-homing it
# as a fresh Grounded with a wrong estimability would re-open the path-dependent split the iota/slope fix
# closes — the slope is a runnable RegressionLaw fit (bench_t_row), so it is MEASURED on every path.
# (sigma/cost stay this model's own engineering-judgement seed-path spread for the cycle term.)
_T_ROW = G.Grounded(
    name="t_row", mean=G.SERVE_SLOPE_US.mean, sigma=0.15, cost=1.0, unit="us/row",
    provenance="run_microbatch_staging/results_nopad.json fits.staged.slope_us_per_row (4.316)",
    estimability=G.SERVE_SLOPE_US.estimability, module="bench_t_row",
)
_INPUTS: list[G.Grounded] = [
    G.N_GEN_CORES,            # N_gen
    G.GEN_PER_CORE_DPS,       # R_gen  (152 dps/core, MEASURED)
    G.SERVE_FULL_BUCKET,      # B      (full-bucket width)
    _T_DISP,                  # T_disp
    G.SERVE_IO_US,            # T_io   (the named drain/scatter serial term, top priority)
    _T_ROW,                   # t_row
    G.LEAVES_PER_DECISION,    # L
]


# `_INPUTS` is in the SAME order as INPUT_NAMES, but some grounded quantities carry a
# different `.name` (tau_io_us vs this model's T_io, B_op vs B), so the initial point and
# the per-input sigma/cost maps key on THIS model's INPUT_NAMES (zipped), never on the
# grounded `.name` — keeping the model signature self-homed (P1).
SIGMAS: dict[str, float] = {nm: g.sigma for nm, g in zip(INPUT_NAMES, _INPUTS)}
COSTS: dict[str, float] = {nm: g.cost for nm, g in zip(INPUT_NAMES, _INPUTS)}
NEEDS_MEASUREMENT: dict[str, bool] = {nm: g.needs_measurement
                                      for nm, g in zip(INPUT_NAMES, _INPUTS)}


def initial_point() -> dict[str, float]:
    return {nm: g.mean for nm, g in zip(INPUT_NAMES, _INPUTS)}


def build_driver(tolerance: float = 5.0) -> tuple[AllocationDriver, dict[str, float]]:
    """Factory: a configured `AllocationDriver` + the initial point. `tolerance` = target CI
    half-width on E[f] in dps."""
    f = throughput_jax  # the driver consumes the JAX-traceable f directly (OT→JAX migration, §5)
    costs = [g.cost for g in _INPUTS]
    driver = AllocationDriver(
        f, costs=costs, tolerance=tolerance, names=INPUT_NAMES,
        confidence=0.95, growth_cap=3.0,
    )
    return driver, initial_point()


def cycle_breakdown(x: dict[str, float]) -> dict[str, float]:
    """The per-forward cycle decomposed into its named terms (us) + the serve/producer
    capacities (dps), so a caller can see what binds and which term dominates the cycle."""
    disp, io, comp = x["T_disp"], x["T_io"], x["B"] * x["t_row"]
    cycle = disp + io + comp
    return {
        "T_disp_us": disp, "T_io_us": io, "compute_us": comp, "cycle_us": cycle,
        "serve_dps": 1e6 * x["B"] / (cycle * x["L"]),
        "producer_dps": x["N_gen"] * x["R_gen"],
    }


def producer_caps() -> dict[str, float]:
    """Producer ceiling under both scalings (4.0x measured vs 1.9x Python-ExIt worst)."""
    rg = G.GEN_PER_CORE_DPS.mean
    return {"producer_4.0x_dps": 3 * rg, "producer_1.9x_dps": 1.9 * rg}


def serve_sawtooth(real: int, buckets=(64, 256, 512), max_batch=512) -> float:
    """Serve dps at `real` rows under BUCKETING + the T_io term — the SAWTOOTH (drops at
    bucket edges). Shared shape with model_capacity.serve_sawtooth; computed via the cycle
    decomposition here. Demonstrates the non-monotonicity Critique B established."""
    pad = next((b for b in buckets if b >= real), real)
    if real > max_batch:
        pad = real
    cycle = _T_DISP.mean + G.SERVE_IO_US.mean + pad * _T_ROW.mean
    return 1e6 * real / (cycle * G.LEAVES_PER_DECISION.mean)


def inproc_port_contrast(full_batch: int = 512) -> dict[str, float]:
    """The C++ in-process queue-PORT contrast (computed, NOT advocated): remove the wire —
    T_io ~= 0, t_row = the BARE-forward 3.092us/row (mlp_lowlatency fully_device slope, no
    run_microbatch concat), no pad over a full batch. Isolates transport vs irreducible
    XLA: the residual cycle is dispatch + bare compute only."""
    t_row_bare = 3.092   # mlp_lowlatency/results.json fully_device slope (bare forward)
    cycle = _T_DISP.mean + 0.0 + full_batch * t_row_bare
    asymptote = 1e6 / t_row_bare / G.LEAVES_PER_DECISION.mean   # B->inf, bare slope
    return {
        "full_batch": float(full_batch),
        "cycle_us": cycle,
        "dps_at_full_batch": 1e6 * full_batch / (cycle * G.LEAVES_PER_DECISION.mean),
        "asymptote_dps_bare_slope": asymptote,
    }


def asymptote_dps() -> float:
    """The B->inf serve asymptote (1e6/t_row/L) — reported as UNREACHABLE (max_batch caps
    B), never the bound."""
    return 1e6 / _T_ROW.mean / G.LEAVES_PER_DECISION.mean


if __name__ == "__main__":
    x0 = initial_point()
    print("Design-B cycle-time model — initial point:", x0)
    print("cycle breakdown:", {k: round(v, 1) for k, v in cycle_breakdown(x0).items()})
    print("f(mu_hat) =", round(throughput_numpy(x0), 1), "dps")
    print("producer caps:", {k: round(v, 1) for k, v in producer_caps().items()})
    print("B->inf asymptote (UNREACHABLE):", round(asymptote_dps(), 1), "dps")
    print("inproc-port contrast:", {k: round(v, 1) for k, v in inproc_port_contrast().items()})
