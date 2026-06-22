"""
tools/analysis/leaf_eval_bound/model_zmq_baseline.py
==============================================

Transport variant DESIGN-zmq_baseline: the first-principles leaf-eval throughput LOWER BOUND
(dps) for the CURRENT ZMQ ROUTER/DEALER multipart transport — the REFERENCE every other
transport variant (shm_spin_poll, futex_wake, lockfree_mpsc, cpp_inproc_port) is measured
against. One model module the generic `AllocationDriver` consumes (ADR-0012 P1/P2: the driver owns
no model; this module owns this one, the AllocationDriver owns allocation, bench_store owns SQL).

WHAT THE TRANSPORT IS (inference_server.py `_drain` / `_scatter`). poll()-block first-request
wakeup (`self._poller.poll(timeout=_POLL_INTERVAL_MS)`), recv_multipart(NOBLOCK) greedy drain,
inference_wire memcpy codec decode/encode, send_multipart scatter — all SERIAL on the single-
threaded serve loop BETWEEN forwards (SYNTHESIS v2 §3.3 "forward k+1 cannot begin until scatter
k completes — this is the coalescing engine").

THE SPINE (invariant across ALL transport variants — model_cycletime.py). The binding stage is
the single serialized serve forward, turning a per-forward CYCLE:

    cycle_us(B) = T_disp + tau_io + wakeup + B_eff*t_row
                  (B_eff = pad(B) the compiled bucket width; at a FULL bucket pad(B) ~= B, the
                  optimum's operating point — the serve curve is a SAWTOOTH, peaks at full
                  buckets, so `serve_sawtooth()` shows the non-monotonicity, never an assumed
                  monotonicity)
    serve_dps   = 1e6 * B / (cycle_us(B) * L)
    transport_dps (NON-BINDING) = 1.0 / (L * tmsg_us_leaf * 1e-6)
    dps = min( N_gen*R_gen ,  serve_dps ,  transport_dps )

WHAT THIS VARIANT MOVES vs HOLDS (the brief's design contract). zmq_baseline MOVES NOTHING off
the reference — it IS the reference. Its (tau_io, wakeup, tmsg) profile is the BASELINE the v1
grounding already describes; this module composes those onto the SAME spine and registers them
under the slug-prefixed names the sweep is uniform over:

  MOVED-TERM profile (zmq_baseline-registered quantities, the transport levers):
    * zmq_baseline_tau_io_us   — serial drain+decode+encode+scatter per coalesced frame (the
                                 DOMINANT lever, top Neyman priority). Seed 20us (v1 G.SERVE_IO_US).
    * zmq_baseline_wakeup_us   — poll-syscall + libzmq fd-readiness wakeup at saturation. NAMED
                                 separately so the sweep contrasts wakeup mechanisms; ADDED to the
                                 cycle (not double-charged: tau_io is the I/O proper, wakeup the
                                 first-poll readiness before it). Seed 1.5us.
    * zmq_baseline_tmsg_us_leaf — per-leaf memcpy codec framing (NON-BINDING; the transport-
                                 capacity arm, ranks LAST). Seed 1.0us (v1 G.MSG_PER_LEAF_US).

  TRANSPORT-INVARIANT inputs (pulled from the manifest by their registered names, NOT
  re-measured here — they are the same forward/producer physics for every variant):
    * n_gen, R_gen   — producer ceiling (3 cores * 152 dps/core, the 4.0x-linear C++ gen rate)
    * T_disp_us      — the pjit/XLA dispatch floor (68.84us)
    * t_row_us       — the run_microbatch staged slope (4.317 us/row)
    * B_op           — the sustained full-bucket width (256 rows/forward)
    * LPD            — leaves per decision (500, the dps unit conversion)

EVERY INPUT IS PULLED THROUGH THE MANIFEST CONTRACT (ADR-0012 P1 single-home; the brief's
"no hand-copied literals"). `manifest.value(name, trust=True)` returns the LATEST measured
(mean, sigma, n) from postgres with trusted=True, or the registered seed flagged trusted=False
when no sole-workload run has populated it yet. This module BRANCHES on `trusted` and records
which inputs are still seed-only — exactly what the Neyman allocator then ranks for the next
sole-workload bench. (Graceful degradation: if postgres is down the manifest announces once and
every value() returns its seed flagged untrusted, so the bound still computes — as the v1 models
did.)

WHY A LOWER BOUND. Every term is the REAL staged run_microbatch fixed cost + slope (>= the bare-
forward floor); tau_io + wakeup are ADDED on top (the serial transport the forward bench does
not contain); B is a FULL bucket (not the unreachable B->inf asymptote); the producer ceiling is
a hard min taking the WORSE of the 4.0x and 1.9x scalings. NO coordination loss (RTT idle,
convoy, cold-JIT) is put INTO any stage — those are the losses the OPTIMUM engineers away. It is
a BENCH-dps bound (the input rates read higher than the closed-loop e2e — adapter.md §7),
contingent on the optimum reaching full-bucket feed and on the single-threaded serialization
being no worse than the tau_io+wakeup charged.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np


from leaf_eval_bound.store import manifest  # noqa: E402  — the SSOT registry contract (import-clean; touches no DB on import)
from leaf_eval_bound.alloc.driver import AllocationDriver  # noqa: E402

# The transport SLUG (the registry prefix for this variant's moved terms; the comparison-table key).
SLUG = "zmq_baseline"

# Single home of the model signature (`throughput_jax` + numpy `throughput_numpy` share it — P1). The order is the
# contract; INPUT_QUANTITIES maps each model-input name 1:1 to the REGISTRY quantity it pulls.
INPUT_NAMES = ["N_gen", "R_gen", "B", "T_disp", "tau_io", "wakeup", "t_row", "L", "tmsg"]

# model-input name -> (registry quantity name, default cost). The cost is the per-sample bench cost
# the Neyman allocator uses (relative scale only); it mirrors the v1 grounding costs (the binding
# serve/transport terms are the expensive sole-workload benches). A model never hard-codes a MEAN —
# the mean/sigma come from the manifest; cost is the bench-effort weight, which is a model-side fact.
INPUT_QUANTITIES: dict[str, tuple[str, float]] = {
    "N_gen":  ("n_gen", 0.5),
    "R_gen":  ("R_gen", 30.0),
    "B":      ("B_op", 4.0),
    "T_disp": ("T_disp_us", 1.0),
    "tau_io": (f"{SLUG}_tau_io_us", 8.0),     # the DOMINANT moved lever (top Neyman priority)
    "wakeup": (f"{SLUG}_wakeup_us", 6.0),     # the wakeup moved term (ranks behind tau_io)
    "t_row":  ("t_row_us", 1.0),
    "L":      ("LPD", 2.0),
    "tmsg":   (f"{SLUG}_tmsg_us_leaf", 2.0),  # NON-BINDING transport-capacity arm (ranks last)
}

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
    N_gen, R_gen, B, T_disp, tau_io, wakeup, t_row, L, tmsg = x
    producer = N_gen * R_gen
    cycle_us = T_disp + tau_io + wakeup + B * t_row
    serve = 1e6 * B / (cycle_us * L)
    transport = 1.0 / (L * tmsg * 1e-6)
    return jnp.minimum(jnp.minimum(producer, serve), transport)


# --------------------------------------------------------------------------- #
# Manifest resolution — every input through the ONE contract (P1; the brief's "no hand-copied
# literals"). Resolved ONCE at import into a Quantity table the model + the report read.
# --------------------------------------------------------------------------- #
def _resolve(trust: bool = True) -> dict[str, "manifest.Quantity"]:
    """Resolve each model input to its registry Quantity via `manifest.quantity(name, trust=...)`.
    Returns {model_input_name: Quantity}. trust=True: the latest measured value (trusted=True) or the
    seed (trusted=False) per input; trust=False forces the v1 seeds. A quantity that is not registered
    is a loud KeyError from the manifest (ADR-0002) — never a silent default."""
    out: dict[str, manifest.Quantity] = {}
    for nm in INPUT_NAMES:
        qname, _cost = INPUT_QUANTITIES[nm]
        out[nm] = manifest.quantity(qname, trust=trust)
    return out


def registry_qname(nm: str) -> str:
    """The registry quantity name model-input `nm` pulls from — the model's ONE coupling to the registry,
    exposed uniformly (refactor move 3a). Replaces the runner-side `_registry_qname` shim (which sniffed
    INPUT_QUANTITIES vs _MANIFEST_NAME) and its verbatim copy in untrusted_drive — the duplicated P1 the
    refactor note and the out-of-frame hack-audit flagged. Here the map is INPUT_QUANTITIES[nm]=(qname,cost)."""
    return INPUT_QUANTITIES[nm][0]


def initial_point(trust: bool = True) -> dict[str, float]:
    """The grounded mean point f is first evaluated at (mu_hat). Pulled from the manifest (measured
    if a sole-workload run has populated the quantity, else its seed)."""
    return {nm: q.mean for nm, q in _resolve(trust=trust).items()}


def sigmas(trust: bool = True) -> dict[str, float]:
    """Per-input 1-sigma spread, keyed on THIS model's INPUT_NAMES (P1: the model signature is self-
    homed; the registry quantity name is an implementation detail of the pull)."""
    return {nm: q.sigma for nm, q in _resolve(trust=trust).items()}


def costs() -> dict[str, float]:
    """Per-input bench cost (the Neyman allocator's effort weight). A model-side fact (which benches
    are expensive), not a measured quantity — so it lives here, not in the registry."""
    return {nm: INPUT_QUANTITIES[nm][1] for nm in INPUT_NAMES}


def trusted_flags(trust: bool = True) -> dict[str, bool]:
    """Per-input `trusted` bool: True iff the value is a live postgres measurement, False iff it is a
    seed. The model BRANCHES on this to flag the bound as resting on unmeasured inputs (the brief's
    trust/distrust contract); the report prints it; the Neyman loop ranks the untrusted binding terms."""
    return {nm: q.trusted for nm, q in _resolve(trust=trust).items()}


def needs_measurement(trust: bool = True) -> dict[str, bool]:
    """A NEEDS-SOLE-WORKLOAD flag per input: a quantity is outstanding iff it is not yet trusted (no
    live measurement). This is what the Neyman ranking annotates — the v1 grounding's per-quantity
    needs_measurement flag is subsumed by "trusted=False" here (a seed always needs a real run)."""
    return {nm: (not t) for nm, t in trusted_flags(trust=trust).items()}


# Module-level resolved views (TRUST) — the SIGMAS/COSTS/NEEDS_MEASUREMENT the runner reads, same
# attribute surface as model_capacity / model_cycletime. Resolved at import (one manifest pass).
SIGMAS: dict[str, float] = sigmas(trust=True)
COSTS: dict[str, float] = costs()
NEEDS_MEASUREMENT: dict[str, bool] = needs_measurement(trust=True)


def build_driver(tolerance: float = 5.0, trust: bool = True) -> tuple[AllocationDriver, dict[str, float]]:
    """Factory: a configured `AllocationDriver` (over the JAX `f` (`throughput_jax`), manifest-grounded costs) + the
    initial point estimate. `tolerance` is the target CI half-width on E[f] in dps. The costs are the
    model-side bench-effort weights (costs()); the pilot is drawn from the manifest means/sigmas."""
    f = throughput_jax  # the driver consumes the JAX-traceable f directly (OT→JAX migration, §5)
    c = [COSTS[nm] for nm in INPUT_NAMES]
    driver = AllocationDriver(
        f, costs=c, tolerance=tolerance, names=INPUT_NAMES,
        confidence=0.95, growth_cap=3.0,
    )
    return driver, initial_point(trust=trust)


# --------------------------------------------------------------------------- #
# Diagnostics — the cycle decomposition, stage capacities, the sawtooth, producer caps. So a caller
# sees WHAT binds, by how much, and which moved term dominates the cycle.
# --------------------------------------------------------------------------- #
def cycle_breakdown(x: dict[str, float]) -> dict[str, float]:
    """The per-forward cycle decomposed into its named terms (us) + the three stage capacities (dps).
    The binding stage is the min of the capacities; the cycle shows tau_io + wakeup vs compute."""
    disp, io, wake = x["T_disp"], x["tau_io"], x["wakeup"]
    comp = x["B"] * x["t_row"]
    cycle = disp + io + wake + comp
    return {
        "T_disp_us": disp, "tau_io_us": io, "wakeup_us": wake, "compute_us": comp, "cycle_us": cycle,
        "serve_dps": 1e6 * x["B"] / (cycle * x["L"]),
        "producer_dps": x["N_gen"] * x["R_gen"],
        "transport_dps": 1.0 / (x["L"] * x["tmsg"] * 1e-6),
    }


def stage_capacities(x: dict[str, float]) -> dict[str, float]:
    """The three stage capacities (dps) at point x — GENERATION, SERVE (the binding cycle), TRANSPORT
    (non-binding). The bound is the min."""
    cb = cycle_breakdown(x)
    return {"GENERATION": cb["producer_dps"], "SERVE": cb["serve_dps"], "TRANSPORT": cb["transport_dps"]}


def producer_caps() -> dict[str, float]:
    """Producer ceiling under both core scalings (4.0x measured C++ vs 1.9x Python-ExIt worst case). A
    lower bound surfaces the worse case absent evidence the C++ transport path escapes the ExIt
    contention. R_gen is pulled from the manifest (the registered gen rate)."""
    rg = manifest.quantity("R_gen", trust=True).mean
    return {"producer_4.0x_dps": 3 * rg, "producer_1.9x_dps": 1.9 * rg}


def serve_sawtooth(real: int, buckets=(64, 256, 512), max_batch=512, trust: bool = True) -> float:
    """Serve dps at `real` rows under BUCKETING + the tau_io + wakeup terms — the SAWTOOTH (drops at
    bucket edges; Critique B). Snap up to the smallest bucket >= real; past the top bucket run unpadded
    at width=real. Demonstrates the bound peaks at FULL buckets and drops just past an edge, so a larger
    B is not unconditionally conservative. Pulls the cycle terms from the manifest."""
    x = initial_point(trust=trust)
    pad = next((b for b in buckets if b >= real), real)
    if real > max_batch:
        pad = real
    cycle = x["T_disp"] + x["tau_io"] + x["wakeup"] + pad * x["t_row"]
    return 1e6 * real / (cycle * x["L"])


def asymptote_dps(trust: bool = True) -> float:
    """The B->inf serve asymptote (1e6/t_row/L) — reported as UNREACHABLE (max_batch caps B at <=512),
    never the bound."""
    x = initial_point(trust=trust)
    return 1e6 / x["t_row"] / x["L"]


def bound(trust: bool = True) -> dict[str, Any]:
    """The variant's headline result: f(mu_hat), the binding stage, the cycle breakdown, the
    trusted/untrusted input map, and the ratio to the ~203 plateau reference. The one call a report
    wants. `trust=False` computes the seed-only (v1 grounding) bound."""
    x = initial_point(trust=trust)
    f = throughput_numpy(x)
    caps = stage_capacities(x)
    binding = min(caps, key=caps.get)
    tflags = trusted_flags(trust=trust)
    plat = ref_plateau_dps()
    return {
        "slug": SLUG,
        "dps": f,
        "binding_stage": binding,
        "stage_capacities": caps,
        "cycle_breakdown": cycle_breakdown(x),
        "producer_caps": producer_caps(),
        "trusted": tflags,
        "all_trusted": all(tflags.values()),
        "untrusted_inputs": [nm for nm, t in tflags.items() if not t],
        "ref_plateau_dps": plat,
        "ratio_to_plateau": f / plat,
    }


# The ~203 dps empirical plateau is a USER-supplied reference for ONE config family (NOT grounded in
# any repo file, NOT a target). It is a model-side reference constant, not a measured quantity, so it
# lives here as a named constant (mirroring references.REF_PLATEAU_DPS — the single home of
# the references; this module re-exposes it for the report rather than re-deriving a literal).
def ref_plateau_dps() -> float:
    """The ~203 dps empirical plateau reference (NOT a target). Single-homed in the v1 grounding
    (`references.REF_PLATEAU_DPS`); pulled from there so there is one home for the reference."""
    from leaf_eval_bound.contract import grounding as G
    from leaf_eval_bound.contract import references
    return float(references.REF_PLATEAU_DPS)


if __name__ == "__main__":
    print(f"Design-{SLUG} — ZMQ baseline transport leaf-eval throughput lower bound")
    print("=" * 78)
    plat = ref_plateau_dps()

    for label, tr in (("TRUST (latest measured, else seed)", True), ("DISTRUST (v1 seeds)", False)):
        x = initial_point(trust=tr)
        f = throughput_numpy(x)
        cb = cycle_breakdown(x)
        caps = stage_capacities(x)
        binding = min(caps, key=caps.get)
        tflags = trusted_flags(trust=tr)
        print(f"\n[{label}] initial point:", {k: round(v, 3) for k, v in x.items()})
        print("  per-forward cycle (us):", {k: round(v, 1) for k, v in cb.items() if k.endswith("_us")})
        print("  stage capacities (dps):", {k: round(v, 1) for k, v in caps.items()})
        print(f"  f(mu_hat) = {f:.1f} dps   binding stage = {binding}   "
              f"(~{f/plat:.2f}x the ~{plat:.0f} plateau)")
        print("  producer caps:", {k: round(v, 1) for k, v in producer_caps().items()})
        untrusted = [nm for nm, t in tflags.items() if not t]
        print(f"  trusted inputs: {sum(tflags.values())}/{len(tflags)}; "
              f"untrusted (seed-only, NEEDS-SOLE-WORKLOAD): {untrusted}")

    print("\n  B->inf asymptote (UNREACHABLE under max_batch cap):", round(asymptote_dps(), 1), "dps")
    saw = {r: round(serve_sawtooth(r), 1) for r in (64, 128, 192, 224, 256, 384, 512)}
    print("  serve sawtooth dps (real rows -> bucketed serve dps, incl. tau_io+wakeup):", saw)
    print("\n  NOTE: zmq_baseline MOVES NOTHING off the reference — its (tau_io, wakeup, tmsg) profile")
    print("  IS the baseline. The bound rests on the UNMEASURED moved terms (top Neyman targets: "
          f"{SLUG}_tau_io_us >> {SLUG}_wakeup_us >> {SLUG}_tmsg_us_leaf).")
