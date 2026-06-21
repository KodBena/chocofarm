"""
tools/analysis/OpenTURNS/model_futex_wake.py
============================================

Transport variant FUTEX-WAKE: the leaf-eval throughput LOWER BOUND (dps) for a SHARED-MEMORY
RING transport whose serve core FUTEX-WAITs on the ring's tail word when the ring is empty and
is FUTEX-WAKE'd by a producer on the empty->nonempty edge. It is the shm_spin_poll transport
with the BURNT-CORE busy-spin replaced by a parking wait: the ring drain is identical (zero-copy
request span + reply-ring memcpy + counter bookkeeping), only the WAKEUP mechanism differs —
one futex syscall on the edge (the group-wakeup the convoy work studied; here a single parked
consumer, so a 1-waiter wake) in exchange for NOT burning a core. It composes the SAME
serialized-serve-cycle structure the v1 cycle-time model (model_cycletime.py) uses —
`cycle_us = T_disp + wakeup + tau_io + B_eff*t_row`,
`dps = min(N_gen*R_gen, 1e6*B/(cycle*L))` evaluated at a FULL bucket (the serve sawtooth's
peak) — with THIS transport's (tau_io, wakeup, msg-cost) profile substituted. It is one model
module the generic `NeymanDriver` consumes (ADR-0012 P1/P2: the driver owns no model; this
module owns ITS math + reads every input through the manifest SSOT, never a hand-copied literal).

WHAT THIS TRANSPORT MOVES (vs the ZMQ baseline; the rest of the cycle is INVARIANT):
  * WAKEUP  -> `futex_wake_wakeup_us` (~2us seed): a FUTEX_WAKE of the one parked serve thread
    on the ring's empty->nonempty edge — one futex syscall + a scheduler context switch. This is
    the DISTINGUISHING lever: it sits BETWEEN shm_spin_poll's ~0.1us spin (which burns a core)
    and zmq_baseline's ~1.5us poll path (through the broker), and is the price the futex variant
    pays to NOT burn the serve core (it sleeps when the ring is empty). Reported as a SEPARATE
    additive cycle term (the brief names it distinctly).
        THE HONEST SATURATION SUBTLETY (the load-bearing modeling choice). At saturation (the
        regime R2 this bound models) the ring is rarely empty: the serve core finishes forward k
        and requests for k+1 are ALREADY queued, so it does NOT FUTEX_WAIT — it drains directly,
        paying ~0 wakeup. The futex syscall is paid ONLY on the empty->nonempty edge, a FRACTION
        of forwards. A strict LOWER bound takes the PESSIMISTIC arm: the headline charges the
        wakeup EVERY forward (the serve core parks after each forward and must be woken). The
        `saturation_wakeup_contrast()` then amortizes it by the empty-edge fraction -> ~0, shown
        honestly (NOT the headline) so the Neyman allocator can rank "how often does the ring go
        empty?" as its own question. At a full bucket the cycle is compute-bound (B*t_row ~=
        1105us at B_op=256), so even the pessimistic per-forward 2-3us wakeup moves the serve
        bound by <1 dps — so this transport's value is OPERATIONAL (no burnt core), not throughput.
  * tau_io  -> `futex_wake_tau_io_us`: the per-forward serial drain+scatter — IDENTICAL to the
    shm ring drain (a REPLY-RING memcpy + per-message counter bookkeeping, the request rows drained
    ZERO-COPY as a ring span; ~8.8us seed at B_op=256). The futex wakeup is the SEPARATE term
    above, NOT folded into tau_io. This is the DOMINANT lever the transport design controls and the
    binding-stage term — the top Neyman target while unmeasured.
  * msg-cost -> `futex_wake_tmsg_us_leaf` (~0.15us/leaf): a bare in-ring memcpy, no frame envelope
    (same ring copy as shm). Reported (non-binding by a wide margin), ranks LAST for the allocator.

The cycle's INVARIANT inputs are pulled from the manifest by their registered names —
`T_disp_us`, `t_row_us`, `B_op`, `LPD`, `R_gen`, `n_gen` — UNCHANGED by the transport (the
one exception the brief carves out, cpp_inproc_port moving t_row, does NOT apply here: the
futex ring feeds the SAME staged `run_microbatch` forward, so t_row is the staged slope).

THE COPY-BOTH CONTRAST (computed, honestly surfaced — NOT the headline). The futex tau_io seed
uses the ZERO-COPY request drain (the design's defining property, shared with shm). Whether that
elision is truly realized is a dominant uncertainty; `copy_both_contrast()` adds the SEPARATE
`futex_wake_req_drain_us` term (~31us) to show the bound IF the request drain is a real memcpy
(the pessimistic arm). The Neyman allocator then ranks "is zero-copy real?" as its own question.

WHY A LOWER BOUND. Every term is the REAL staged forward cost (T_disp the pjit/XLA dispatch
floor, t_row the staged run_microbatch slope) with the futex tau_io ADDED on top at a CONSERVATIVE
memcpy bandwidth (8 B/ns — a slower copy gives a LARGER tau_io, a LOWER throughput) and the futex
wakeup charged PER FORWARD (the pessimistic park-every-forward arm), B held at a FULL bucket (not
the unreachable B->inf asymptote), and the producer ceiling a hard min taking the worse of the
4.0x/1.9x core scalings. It is a BENCH-dps bound (the input rates read higher than the closed-loop
e2e), contingent on the optimum reaching the full-bucket feed and on the single-threaded
serialization being no worse than the tau_io charged. EVERY input is resolved through the
manifest's TRUST contract and the bound is FLAGGED when it rests on a seed (trusted=False) rather
than a live postgres measurement — ADR-0002: a seed is never silently reported as measured.

REFERENCE POINTS (NOT targets — workflow-brief-neutrality). The empirical ~203 dps plateau
(one config family) and the v1 ~429 (Design-B) / ~620 (the v1 inproc-port contrast) are
reference points the bound is compared against, never tuned toward.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import manifest  # noqa: E402  — the SSOT registry; every input is pulled through value()/quantity()

# The transport slug + the prefixed moved-term quantity names (single home of THIS variant's identity).
SLUG = "futex_wake"
TAU_IO_NAME = "futex_wake_tau_io_us"
WAKEUP_NAME = "futex_wake_wakeup_us"
TMSG_NAME = "futex_wake_tmsg_us_leaf"
REQ_DRAIN_NAME = "futex_wake_req_drain_us"     # the copy-both contrast term (not in the headline cycle)

# The model signature — single home (the symbolic function + the numpy fallback share it, P1). The cycle is
# T_disp + wakeup + tau_io + B*t_row; dps = min(producer, 1e6*B/(cycle*L)). WAKEUP is a NAMED additive term
# (the brief names it separately): unlike the spin variant it is NOT ~0 — the futex pays a syscall on the
# empty-edge, charged PER FORWARD in the pessimistic lower-bound arm.
INPUT_NAMES = ["N_gen", "R_gen", "B", "T_disp", "wakeup", "tau_io", "t_row", "L"]

# The throughput EXPRESSION, single-homed as a string (muParser grammar). SAME shape as model_cycletime,
# with the WAKEUP term added into the cycle and tau_io being the FUTEX (= shm ring) drain term.
THROUGHPUT_EXPR = (
    "min("
    "  N_gen*R_gen ,"
    "  1e6 * B / ((T_disp + wakeup + tau_io + B*t_row) * L)"
    ")"
)

# The manifest quantity name each model input is pulled from (1:1 with INPUT_NAMES). The invariant inputs
# carry the baseline registered names; the moved terms carry the prefixed futex names. This map is the
# model's ONLY coupling to the registry — there are no hand-copied literals.
_MANIFEST_NAME: dict[str, str] = {
    "N_gen": "n_gen",
    "R_gen": "R_gen",
    "B": "B_op",
    "T_disp": "T_disp_us",
    "wakeup": WAKEUP_NAME,
    "tau_io": TAU_IO_NAME,
    "t_row": "t_row_us",
    "L": "LPD",
}

# Per-input benchmark COST (relative; the Neyman allocator's c_i). The moved terms are cheap microbenches;
# B_op/LPD/R_gen are the expensive end-to-end/saturation reads (mirrors leaf_eval_grounding's costs). Homed
# here so the model carries its own allocation costs (the manifest stores physical quantities, not costs).
# The futex wakeup is a touch costlier than the spin's cache-line read (it needs a two-thread, two-core
# pinned futex handoff), but still cheap vs the saturation reads.
_COST: dict[str, float] = {
    "N_gen": 0.5, "R_gen": 30.0, "B": 4.0, "T_disp": 1.0,
    "wakeup": 3.0, "tau_io": 8.0, "t_row": 6.0, "L": 2.0,
}


def throughput_numpy(x: dict[str, float]) -> float:
    """numpy-only evaluation of f (the fallback path + the lockstep cross-check of the symbolic formula).
    SAME formula as THROUGHPUT_EXPR. B is the FULL-bucket width (pad ~= B). The wakeup is charged PER FORWARD
    (the pessimistic park-every-forward arm)."""
    cycle_us = x["T_disp"] + x["wakeup"] + x["tau_io"] + x["B"] * x["t_row"]
    serve = 1e6 * x["B"] / (cycle_us * x["L"])
    producer = x["N_gen"] * x["R_gen"]
    return float(min(producer, serve))


# --------------------------------------------------------------------------- #
# Manifest resolution — pull every input through the TRUST contract; branch on `trusted`.
# --------------------------------------------------------------------------- #
def resolve_inputs(trust: bool = True) -> dict[str, manifest.Quantity]:
    """Resolve every model input to a `manifest.Quantity` (mean/sigma/n/trusted/source). The model's ONE
    read of the registry: each INPUT_NAMES key maps to its manifest quantity name and is pulled through
    `manifest.quantity(..., trust=trust)`. trust=True returns the live measurement when one exists (and
    flags trusted=True), else falls back to the seed (trusted=False). NEVER passes rerun=True (timing-
    sensitive — an operator action)."""
    return {nm: manifest.quantity(_MANIFEST_NAME[nm], trust=trust) for nm in INPUT_NAMES}


def initial_point(trust: bool = True) -> dict[str, float]:
    """The resolved mean point f is first evaluated at. trust=True uses live measurements where they exist;
    trust=False forces the v1/first-principles seeds."""
    return {nm: q.mean for nm, q in resolve_inputs(trust=trust).items()}


def sigmas(trust: bool = True) -> dict[str, float]:
    """Per-input 1-sigma spread (the resolved measurement/seed sigma) keyed on INPUT_NAMES."""
    return {nm: q.sigma for nm, q in resolve_inputs(trust=trust).items()}


def costs() -> dict[str, float]:
    """Per-input benchmark cost keyed on INPUT_NAMES (the Neyman allocator's c_i)."""
    return dict(_COST)


def untrusted_inputs(trust: bool = True) -> list[str]:
    """The model inputs currently resting on a SEED (trusted=False) rather than a live postgres measurement
    — the ADR-0002 honesty surface: the bound is flagged as resting on these, and the Neyman allocator ranks
    them for the next sole-workload bench. Returns the INPUT_NAMES (model-side names)."""
    return [nm for nm, q in resolve_inputs(trust=trust).items() if not q.trusted]


# --------------------------------------------------------------------------- #
# OpenTURNS / driver surface (mirrors model_cycletime / model_shm_spin_poll).
# --------------------------------------------------------------------------- #
def build_symbolic_function() -> Any:
    import openturns as ot
    return ot.SymbolicFunction(INPUT_NAMES, [THROUGHPUT_EXPR])


def build_driver(tolerance: float = 5.0, trust: bool = True) -> tuple[Any, dict[str, float]]:
    """Factory: a configured `NeymanDriver` (over the symbolic f, the per-input costs) + the resolved
    initial point. `tolerance` is the target CI half-width on E[f] in dps. `trust` selects live-vs-seed
    inputs. Imported lazily (the driver pulls in openturns)."""
    from neyman_driver import NeymanDriver
    f = build_symbolic_function()
    cost_list = [_COST[nm] for nm in INPUT_NAMES]
    driver = NeymanDriver(
        f, costs=cost_list, tolerance=tolerance, names=INPUT_NAMES,
        confidence=0.95, growth_cap=3.0,
    )
    return driver, initial_point(trust=trust)


# --------------------------------------------------------------------------- #
# Decomposition / diagnostics.
# --------------------------------------------------------------------------- #
def cycle_breakdown(x: dict[str, float]) -> dict[str, float]:
    """The per-forward cycle decomposed into its named terms (us) + the serve/producer capacities (dps), so
    a caller sees what binds and which term dominates. The futex transport's wakeup + tau_io are the terms it
    moved; compute = B*t_row is the invariant forward work."""
    disp, wake, io = x["T_disp"], x["wakeup"], x["tau_io"]
    comp = x["B"] * x["t_row"]
    cycle = disp + wake + io + comp
    return {
        "T_disp_us": disp, "wakeup_us": wake, "tau_io_us": io, "compute_us": comp, "cycle_us": cycle,
        "serve_dps": 1e6 * x["B"] / (cycle * x["L"]),
        "producer_dps": x["N_gen"] * x["R_gen"],
    }


def producer_caps(trust: bool = True) -> dict[str, float]:
    """Producer ceiling under both core scalings (4.0x measured C++ vs 1.9x Python-ExIt worst), pulled from
    the manifest R_gen — a lower bound surfaces the worse case absent evidence the C++ path escapes the
    contention."""
    rg, _, _, _ = manifest.value("R_gen", trust=trust)
    return {"producer_4.0x_dps": 3 * rg, "producer_1.9x_dps": 1.9 * rg}


def serve_sawtooth(real: int, buckets=(64, 256, 512), max_batch=512, trust: bool = True) -> float:
    """Serve dps at `real` rows under BUCKETING + the futex tau_io + (per-forward) wakeup terms — the
    SAWTOOTH (drops at bucket edges; peaks at full buckets). Pulls T_disp/tau_io/wakeup/t_row/L from the
    manifest. Demonstrates the non-monotonicity (a larger B is not unconditionally conservative)."""
    pad = next((b for b in buckets if b >= real), real)
    if real > max_batch:
        pad = real
    t_disp, _, _, _ = manifest.value("T_disp_us", trust=trust)
    wake, _, _, _ = manifest.value(WAKEUP_NAME, trust=trust)
    tau, _, _, _ = manifest.value(TAU_IO_NAME, trust=trust)
    t_row, _, _, _ = manifest.value("t_row_us", trust=trust)
    lpd, _, _, _ = manifest.value("LPD", trust=trust)
    cycle = t_disp + wake + tau + pad * t_row
    return 1e6 * real / (cycle * lpd)


def saturation_wakeup_contrast(trust: bool = True, empty_edge_fraction: float = 0.05) -> dict[str, float]:
    """The HONEST saturation arm (computed, NOT the headline). The headline charges the futex wakeup EVERY
    forward (the pessimistic park-every-forward lower bound). At saturation the ring is rarely empty, so the
    futex syscall is paid only on the empty->nonempty EDGE — a fraction `empty_edge_fraction` of forwards.
    This amortizes the wakeup by that fraction (the OPTIMISTIC arm) so the operator sees what saturation buys
    and the Neyman allocator can rank "how often does the ring go empty?" as its own measurable question. The
    `empty_edge_fraction` default (0.05) is a FIRST-PRINCIPLES seed (at saturation the producer set keeps the
    ring fed, so the empty-edge is rare); it is NOT a manifest quantity (it is a regime assumption, surfaced
    here, not a physical measurement). Returns the per-forward-charged headline + the amortized arm + the
    delta (which is ~0 at a full bucket — the point: the wakeup mechanism barely moves throughput here)."""
    x = initial_point(trust=trust)
    bound_per_forward = throughput_numpy(x)                       # headline: wakeup every forward
    x_amort = dict(x)
    x_amort["wakeup"] = x["wakeup"] * empty_edge_fraction         # amortize over the empty-edge fraction
    bound_amortized = throughput_numpy(x_amort)
    return {
        "wakeup_per_forward_us": x["wakeup"],
        "empty_edge_fraction": empty_edge_fraction,
        "wakeup_amortized_us": x_amort["wakeup"],
        "bound_wakeup_per_forward_dps": bound_per_forward,
        "bound_wakeup_amortized_dps": bound_amortized,
        "delta_dps": bound_amortized - bound_per_forward,
    }


def copy_both_contrast(trust: bool = True) -> dict[str, float]:
    """The PESSIMISTIC tau_io arm: charge the request-drain memcpy (`futex_wake_req_drain_us`) on top of the
    zero-copy tau_io, i.e. the bound IF the design's zero-copy ring-span drain is NOT realized. Computed and
    surfaced honestly (NOT the headline) so the operator sees what zero-copy buys + the Neyman allocator can
    rank "is zero-copy real?" as its own measurable question. Returns the copy-both tau_io + the resulting
    bound at the operating point, alongside the zero-copy headline for contrast."""
    x = initial_point(trust=trust)
    req_drain, _, _, _ = manifest.value(REQ_DRAIN_NAME, trust=trust)
    x_cb = dict(x)
    x_cb["tau_io"] = x["tau_io"] + req_drain         # charge the request-drain copy on top
    return {
        "req_drain_us": req_drain,
        "tau_io_zerocopy_us": x["tau_io"],
        "tau_io_copyboth_us": x_cb["tau_io"],
        "bound_zerocopy_dps": throughput_numpy(x),
        "bound_copyboth_dps": throughput_numpy(x_cb),
    }


def tmsg_capacity(trust: bool = True) -> dict[str, float]:
    """The transport-stage per-leaf capacity (dps) from `futex_wake_tmsg_us_leaf`, to confirm it is
    NON-BINDING by a wide margin (1/(LPD*tmsg*1e-6)). Reported, ranks last for the allocator."""
    tmsg, _, _, _ = manifest.value(TMSG_NAME, trust=trust)
    lpd, _, _, _ = manifest.value("LPD", trust=trust)
    return {"tmsg_us_leaf": tmsg, "transport_capacity_dps": 1.0 / (lpd * tmsg * 1e-6)}


def asymptote_dps(trust: bool = True) -> float:
    """The B->inf serve asymptote (1e6/t_row/L) — reported as UNREACHABLE (max_batch caps B), never the
    bound."""
    t_row, _, _, _ = manifest.value("t_row_us", trust=trust)
    lpd, _, _, _ = manifest.value("LPD", trust=trust)
    return 1e6 / t_row / lpd


# A reference point (NOT a target) for the report — the empirical plateau, pulled from the grounding module
# so it has ONE home (it is a reference constant, not a manifest-measured quantity).
def _ref_plateau() -> float:
    import leaf_eval_grounding as G
    return G.REF_PLATEAU_DPS


if __name__ == "__main__":
    trust = True
    x0 = initial_point(trust=trust)
    cb = cycle_breakdown(x0)
    f0 = throughput_numpy(x0)
    plateau = _ref_plateau()
    untrusted = untrusted_inputs(trust=trust)
    print(f"FUTEX-WAKE transport — leaf-eval throughput LOWER BOUND")
    print(f"  postgres available: {manifest.postgres_available()}")
    print(f"  initial point (manifest-resolved): {{{', '.join(f'{k}={v:.3g}' for k,v in x0.items())}}}")
    print(f"  cycle breakdown (us): {{{', '.join(f'{k}={round(v,1)}' for k,v in cb.items() if k.endswith('_us'))}}}")
    print(f"  serve vs producer (dps): serve={cb['serve_dps']:.1f} producer={cb['producer_dps']:.0f}")
    print(f"  f(mu_hat) = {f0:.1f} dps  (~{f0/plateau:.2f}x the ~{plateau:.0f} plateau reference)")
    print(f"  producer caps: {{{', '.join(f'{k}={round(v,1)}' for k,v in producer_caps(trust=trust).items())}}}")
    print(f"  tmsg capacity (non-binding): {{{', '.join(f'{k}={round(v,1)}' for k,v in tmsg_capacity(trust=trust).items())}}}")
    print(f"  saturation-wakeup contrast (per-forward headline vs empty-edge amortized): "
          f"{{{', '.join(f'{k}={round(v,3)}' for k,v in saturation_wakeup_contrast(trust=trust).items())}}}")
    print(f"  copy-both contrast (zero-copy NOT realized): "
          f"{{{', '.join(f'{k}={round(v,1)}' for k,v in copy_both_contrast(trust=trust).items())}}}")
    print(f"  B->inf asymptote (UNREACHABLE): {asymptote_dps(trust=trust):.1f} dps")
    saw = {r: round(serve_sawtooth(r, trust=trust), 1) for r in (64, 192, 224, 256, 384, 512)}
    print(f"  serve sawtooth dps (real rows -> bucketed serve, incl. tau_io+wakeup): {saw}")
    if untrusted:
        print(f"  [ADR-0002] BOUND RESTS ON SEEDS (trusted=False): {untrusted} — flagged, NOT measured. "
              f"Run each bench pinned (taskset -c 0[,1]) + sole-workload to flip them to trusted.")
    else:
        print(f"  [ADR-0002] all inputs are live postgres measurements (trusted=True).")
