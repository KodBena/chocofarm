"""
tools/analysis/OpenTURNS/model_cpp_inproc_port.py
=================================================

Transport variant CPP-INPROC-PORT: the leaf-eval throughput LOWER BOUND (dps) for the C++
IN-PROCESS QUEUE-PORT transport — generation and serve run in ONE process, a leaf-eval is a DIRECT
FUNCTION CALL into the batched forward, with NO wire at all (no ZMQ ROUTER/DEALER, no broker, no
multipart recv/send syscall, no `inference_wire` codec, no corr-id envelope). It is the MOST
AGGRESSIVE endpoint of the transport-design space (the v1 `model_cycletime.inproc_port_contrast`
sketched it at ~620 dps; this is the FULL model on the shared spine). It composes the SAME
serialized-serve-cycle structure the v1 cycle-time model (model_cycletime.py) uses —
`cycle_us = T_disp + wakeup + tau_io + B_eff*t_row`, `dps = min(N_gen*R_gen, 1e6*B/(cycle*L))`
evaluated at a FULL bucket (the serve sawtooth's peak) — with THIS transport's (tau_io, wakeup,
msg-cost) profile substituted. It is one model module the generic `NeymanDriver` consumes (ADR-0012
P1/P2: the driver owns no model; this module owns ITS math + reads every input through the manifest
SSOT, never a hand-copied literal; the NeymanDriver owns allocation; bench_store owns SQL).

WHAT THIS TRANSPORT MOVES (vs the ZMQ baseline; the rest of the cycle is INVARIANT):

  * t_row  -> `cpp_inproc_port_t_row_bare_us` (the BARE-forward slope, 3.092 us/row, vs the staged
    run_microbatch slope 4.317). This is the ONE variant the brief carves out as ALSO moving the
    per-row XLA slope: an in-process direct call into a device-resident batched forward, reading the
    reply device-resident, is the `fully_device` geometry (input device-resident, output not pulled),
    whose slope is the bare-forward per-row cost — NO run_microbatch host-block concat, NO device->host
    pull-to-host-then-reframe. So this variant ISOLATES transport-removed (the concat + reframe the
    inproc port elides) from irreducible-XLA (the matmul slope no transport can move). The contrast is
    surfaced explicitly: `bare_vs_staged_t_row_contrast()` shows what moving to the bare slope buys.

  * tau_io -> `cpp_inproc_port_tau_io_us`: the per-forward serial transport collapses from the ZMQ
    multipart/syscall/per-message-codec (~20us seed) to the RESIDUAL INTRINSIC staging — the one
    host->device crossing of the gathered B-row input block (the transfer the bare/fully_device slope
    DELIBERATELY EXCLUDES, so charging it here makes t_row + tau_io carry the full per-forward cost with
    no double-count and no gap). Seed ~41us at B_op=256 (a conservative host->device bandwidth — a slower
    copy gives a LARGER tau_io, a LOWER throughput). This is the DOMINANT lever the transport design
    controls and the binding-stage term, the top Neyman target while unmeasured.

    THE ARENA / GATHER ASYMMETRY (the INVERSE of lockfree_mpsc). The MPSC queue's nodes are enqueued
    INDEPENDENTLY by N producers, so they are scattered and the MPSC headline CHARGES a gather (elision
    optimistic). The inproc port lives in ONE address space, so the producers can write their feature
    rows DIRECTLY into the consumer's contiguous staging ARENA — so the headline ELIDES the gather (no
    scattered materialization) and pays only the host->device crossing; the gather-CHARGED arm (scattered
    writes + a same-process gather) is the PESSIMISTIC contrast. `copy_contrast()` shows both arms; the
    Neyman allocator ranks "is the staging arena contiguous (gather elidable)?" (`cpp_inproc_port_gather_us`)
    as the dominant transport DESIGN question for this variant. (Reporting it the MPSC way — charging the
    gather by default — would UNDERSTATE the bound for a transport that can plausibly arrange the arena.)

  * WAKEUP -> `cpp_inproc_port_wakeup_us` (~0.1us): the consumer spins a shared-address-space
    ready-counter on its dedicated serve core. At SATURATION (regime R2 — the regime this bound models)
    a ready leaf is essentially always already published, so the consumer NEVER parks and pays the
    same-process cross-core cache-line coherence floor (~0.1us). Reported as a SEPARATE additive cycle
    term (the brief names it distinctly); negligible vs the ~900us full-bucket cycle, but it MAKES
    EXPLICIT that the in-regime consumer pays ~0 wakeup. The off-regime futex-park cost (~1-5us) is NOT
    folded in (provably not paid at saturation — see the wakeup bench).

  * msg-cost -> `cpp_inproc_port_tmsg_us_leaf` (~0.05us/leaf): a producer writes one feature row into
    its arena stripe + pushes a slot index onto a ready-queue (one relaxed-atomic SPSC/MPSC push), no
    frame, no codec, no corr-id, no syscall — the CHEAPEST tmsg of any variant. Reported (NON-BINDING by
    a wide margin), ranks LAST for the allocator (the wire request/reply CAPACITY never binds).

The cycle's INVARIANT inputs are pulled from the manifest by their registered names — `T_disp_us`,
`B_op`, `LPD`, `R_gen`, `n_gen` — UNCHANGED by the transport. (t_row is the ONE exception, moved to
the prefixed bare-forward quantity above.)

WHY A LOWER BOUND. The bare t_row is the REAL fully_device fit slope (>= no XLA floor a transport can
undercut); T_disp is the pjit/XLA dispatch floor; the inproc tau_io is ADDED on top at a CONSERVATIVE
host->device bandwidth with the gather ELIDED in the headline (the honest one-address-space default,
the gather-charged arm surfaced as the pessimistic contrast); B is held at a FULL bucket (not the
unreachable B->inf asymptote); and the producer ceiling is a hard min taking the WORSE of the
4.0x/1.9x core scalings. NO coordination loss (RTT idle, convoy, cold-JIT) is put INTO any stage —
those are the losses the OPTIMUM engineers away. It is a BENCH-dps bound (the input rates read higher
than the closed-loop e2e — adapter.md §7), contingent on the optimum reaching full-bucket feed and on
the single-threaded serialization being no worse than the tau_io+wakeup charged. EVERY input is
resolved through the manifest's TRUST contract and the bound is FLAGGED when it rests on a seed
(trusted=False) rather than a live postgres measurement — ADR-0002: a seed is never silently reported
as measured.

REFERENCE POINTS (NOT targets — workflow-brief-neutrality). The empirical ~203 dps plateau (one config
family) and the v1 ~429 (Design-B) / ~620 (the v1 inproc-port contrast) are reference points the bound
is compared against, never tuned toward. This variant landing well ABOVE the staged-slope variants
(because it ALSO moves t_row) is a real finding, reported as such; if the residual host->device tau_io
turns out to bind the serve cycle harder than the gather-elided seed assumes, the bound dropping is
equally a real finding.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import manifest  # noqa: E402  — the SSOT registry; every input is pulled through value()/quantity()

# The transport SLUG + the prefixed moved-term quantity names (the single home of THIS variant's identity).
SLUG = "cpp_inproc_port"
T_ROW_BARE_NAME = "cpp_inproc_port_t_row_bare_us"   # the ONE per-row term this variant moves (bare fully_device slope)
TAU_IO_NAME = "cpp_inproc_port_tau_io_us"
WAKEUP_NAME = "cpp_inproc_port_wakeup_us"
TMSG_NAME = "cpp_inproc_port_tmsg_us_leaf"
GATHER_NAME = "cpp_inproc_port_gather_us"           # the gather-elision contrast term (the dominant uncertainty; NOT a separate cycle term)

# The model signature — single home (the symbolic function + the numpy fallback share it, P1). The cycle is
# T_disp + wakeup + tau_io + B*t_row; dps = min(producer, serve, transport-capacity). WAKEUP is a NAMED
# additive term (the brief names it separately) even though it is ~0 — so the structure makes the in-regime
# zero-wakeup explicit. tmsg is the NON-BINDING transport-capacity arm.
INPUT_NAMES = ["N_gen", "R_gen", "B", "T_disp", "wakeup", "tau_io", "t_row", "L", "tmsg"]

# The throughput EXPRESSION, single-homed as a string (muParser grammar). SAME shape as the sibling variants
# (model_zmq_baseline / model_lockfree_mpsc), with the inproc tau_io + wakeup in the cycle and t_row pulled
# from the BARE-forward quantity (the one term this variant moves off the staged-slope baseline).
THROUGHPUT_EXPR = (
    "min("
    "  N_gen*R_gen ,"                                                   # GENERATION (producer ceiling)
    "  1e6 * B / ((T_disp + wakeup + tau_io + B*t_row) * L) ,"        # SERVE cycle (full bucket; bare t_row)
    "  1.0/(L*tmsg*1e-6)"                                              # TRANSPORT capacity (non-binding)
    ")"
)

# The manifest quantity name each model input is pulled from (1:1 with INPUT_NAMES). The invariant inputs carry
# the baseline registered names; the moved terms carry the prefixed inproc-port names — INCLUDING t_row, the
# one per-row exception. This map is the model's ONLY coupling to the registry — no hand-copied literals
# (ADR-0012 P1).
_MANIFEST_NAME: dict[str, str] = {
    "N_gen": "n_gen",
    "R_gen": "R_gen",
    "B": "B_op",
    "T_disp": "T_disp_us",
    "wakeup": WAKEUP_NAME,
    "tau_io": TAU_IO_NAME,
    "t_row": T_ROW_BARE_NAME,          # the BARE-forward slope (this variant's one per-row move)
    "L": "LPD",
    "tmsg": TMSG_NAME,
}

# Per-input benchmark COST (relative; the Neyman allocator's c_i). The moved terms are cheap microbenches;
# B_op/LPD/R_gen are the expensive end-to-end/saturation reads (mirrors the sibling models' costs). Homed here
# so the model carries its own allocation costs (the manifest stores physical quantities, not costs).
_COST: dict[str, float] = {
    "N_gen": 0.5, "R_gen": 30.0, "B": 4.0, "T_disp": 1.0,
    "wakeup": 6.0, "tau_io": 8.0, "t_row": 1.0, "L": 2.0, "tmsg": 2.0,
}


def throughput_numpy(x: dict[str, float]) -> float:
    """numpy-only evaluation of f (the fallback path + the lockstep cross-check of the symbolic formula).
    SAME formula as THROUGHPUT_EXPR. B is the FULL-bucket width (pad ~= B). The transport capacity is a
    separate non-binding min arm."""
    producer = x["N_gen"] * x["R_gen"]
    cycle_us = x["T_disp"] + x["wakeup"] + x["tau_io"] + x["B"] * x["t_row"]
    serve = 1e6 * x["B"] / (cycle_us * x["L"])
    transport = 1.0 / (x["L"] * x["tmsg"] * 1e-6)
    return float(min(producer, serve, transport))


# --------------------------------------------------------------------------- #
# Manifest resolution — pull every input through the TRUST contract; branch on `trusted`.
# --------------------------------------------------------------------------- #
def resolve_inputs(trust: bool = True) -> dict[str, manifest.Quantity]:
    """Resolve every model input to a `manifest.Quantity` (mean/sigma/n/trusted/source). The model's ONE read
    of the registry: each INPUT_NAMES key maps to its manifest quantity name and is pulled through
    `manifest.quantity(..., trust=trust)`. trust=True returns the live measurement when one exists (and flags
    trusted=True), else falls back to the seed (trusted=False). NEVER passes rerun=True (timing-sensitive — an
    operator action). A quantity that is not registered is a loud KeyError (ADR-0002)."""
    return {nm: manifest.quantity(_MANIFEST_NAME[nm], trust=trust) for nm in INPUT_NAMES}


def registry_qname(nm: str) -> str:
    """The registry quantity name model-input `nm` pulls from — the model's ONE coupling to the registry,
    exposed uniformly (refactor move 3a). Replaces the runner-side `_registry_qname` shim (which sniffed
    INPUT_QUANTITIES vs _MANIFEST_NAME) and its verbatim copy in untrusted_drive — the duplicated P1 the
    refactor note and the out-of-frame hack-audit flagged. Here the map is _MANIFEST_NAME[nm]=qname."""
    return _MANIFEST_NAME[nm]


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


def trusted_flags(trust: bool = True) -> dict[str, bool]:
    """Per-input `trusted` bool: True iff the value is a live postgres measurement, False iff it is a seed. The
    model BRANCHES on this to flag the bound as resting on unmeasured inputs; the Neyman loop ranks the
    untrusted binding terms."""
    return {nm: q.trusted for nm, q in resolve_inputs(trust=trust).items()}


def untrusted_inputs(trust: bool = True) -> list[str]:
    """The model inputs currently resting on a SEED (trusted=False) rather than a live postgres measurement —
    the ADR-0002 honesty surface: the bound is flagged as resting on these, and the Neyman allocator ranks them
    for the next sole-workload bench. Returns the INPUT_NAMES (model-side names)."""
    return [nm for nm, q in resolve_inputs(trust=trust).items() if not q.trusted]


# Module-level resolved views (TRUST) — the SIGMAS/COSTS the runner reads, same attribute surface as
# model_capacity / model_cycletime / model_zmq_baseline. Resolved at import (one manifest pass).
SIGMAS: dict[str, float] = sigmas(trust=True)
COSTS: dict[str, float] = costs()
NEEDS_MEASUREMENT: dict[str, bool] = {nm: (not t) for nm, t in trusted_flags(trust=True).items()}


# --------------------------------------------------------------------------- #
# OpenTURNS / driver surface (mirrors model_zmq_baseline / model_lockfree_mpsc).
# --------------------------------------------------------------------------- #
def build_symbolic_function() -> Any:
    """The model as an `ot.SymbolicFunction` over INPUT_NAMES (the form the driver consumes). Imported lazily so
    this module stays import-clean without openturns."""
    import openturns as ot
    return ot.SymbolicFunction(INPUT_NAMES, [THROUGHPUT_EXPR])


def build_driver(tolerance: float = 5.0, trust: bool = True) -> tuple[Any, dict[str, float]]:
    """Factory: a configured `NeymanDriver` (over the symbolic f, the per-input costs) + the resolved initial
    point. `tolerance` is the target CI half-width on E[f] in dps. `trust` selects live-vs-seed inputs. Imported
    lazily (the driver pulls in openturns)."""
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
    """The per-forward cycle decomposed into its named terms (us) + the three stage capacities (dps), so a
    caller sees what binds and which moved term dominates. The inproc-port's tau_io + wakeup are the terms it
    moved; compute = B*t_row is the forward work (at the BARE slope — the other term it moved)."""
    disp, wake, io = x["T_disp"], x["wakeup"], x["tau_io"]
    comp = x["B"] * x["t_row"]
    cycle = disp + wake + io + comp
    return {
        "T_disp_us": disp, "wakeup_us": wake, "tau_io_us": io, "compute_us": comp, "cycle_us": cycle,
        "serve_dps": 1e6 * x["B"] / (cycle * x["L"]),
        "producer_dps": x["N_gen"] * x["R_gen"],
        "transport_dps": 1.0 / (x["L"] * x["tmsg"] * 1e-6),
    }


def stage_capacities(x: dict[str, float]) -> dict[str, float]:
    """The three stage capacities (dps) at point x — GENERATION, SERVE (the binding cycle), TRANSPORT
    (non-binding). The bound is the min."""
    cb = cycle_breakdown(x)
    return {"GENERATION": cb["producer_dps"], "SERVE": cb["serve_dps"], "TRANSPORT": cb["transport_dps"]}


def producer_caps(trust: bool = True) -> dict[str, float]:
    """Producer ceiling under both core scalings (4.0x measured C++ vs 1.9x Python-ExIt worst), pulled from the
    manifest R_gen — a lower bound surfaces the worse case absent evidence the C++ path escapes the contention."""
    rg, _, _, _ = manifest.value("R_gen", trust=trust)
    return {"producer_4.0x_dps": 3 * rg, "producer_1.9x_dps": 1.9 * rg}


def serve_sawtooth(real: int, buckets=(64, 256, 512), max_batch=512, trust: bool = True) -> float:
    """Serve dps at `real` rows under BUCKETING + the inproc tau_io + wakeup terms, at the BARE t_row — the
    SAWTOOTH (drops at bucket edges; peaks at full buckets). Snap up to the smallest bucket >= real; past the top
    bucket run unpadded at width=real. Demonstrates the non-monotonicity (a larger B is not unconditionally
    conservative). Pulls the cycle terms from the manifest."""
    x = initial_point(trust=trust)
    pad = next((b for b in buckets if b >= real), real)
    if real > max_batch:
        pad = real
    cycle = x["T_disp"] + x["wakeup"] + x["tau_io"] + pad * x["t_row"]
    return 1e6 * real / (cycle * x["L"])


def copy_contrast(trust: bool = True) -> dict[str, float]:
    """The PESSIMISTIC arm: ADD the same-process arena GATHER (`cpp_inproc_port_gather_us`) onto the headline
    tau_io, i.e. the bound IF the staging arena is NOT contiguous (producers write scattered, so the B rows must
    be gathered contiguous before the host->device crossing). Computed and surfaced honestly (NOT replacing the
    headline) so the operator sees what gather-elision SAVES + the Neyman allocator can rank "is the staging
    arena contiguous?" as its own measurable question. This is the INVERSE of lockfree_mpsc's gather_contrast:
    the MPSC headline CHARGES the gather (scattered nodes; elision optimistic), the inproc-port headline ELIDES
    it (one address space; charging is pessimistic). Returns the gather term + both arms' bounds.

    The seed tau_io is the gather-ELIDED headline (see bench_cpp_inproc_port_tau_io_us.get_seed), so the charged
    arm ADDS the gather term."""
    x = initial_point(trust=trust)
    gather, _, _, _ = manifest.value(GATHER_NAME, trust=trust)
    x_charged = dict(x)
    x_charged["tau_io"] = x["tau_io"] + gather          # charge the gather (scattered-arena pessimistic arm)
    return {
        "gather_us": gather,
        "tau_io_headline_us": x["tau_io"],              # gather ELIDED (the honest one-address-space default)
        "tau_io_gather_charged_us": x_charged["tau_io"],  # gather CHARGED (scattered writes; pessimistic)
        "bound_headline_dps": throughput_numpy(x),
        "bound_gather_charged_dps": throughput_numpy(x_charged),
    }


def bare_vs_staged_t_row_contrast(trust: bool = True) -> dict[str, float]:
    """The contrast THIS variant exists to isolate (the brief's "transport-removed vs irreducible-XLA"): the
    bound at the BARE-forward t_row (`cpp_inproc_port_t_row_bare_us`, the inproc-port slope) vs the bound at the
    STAGED run_microbatch t_row (`t_row_us`, the wire-fed baseline slope), holding the rest of the inproc cycle
    FIXED. The gap is what moving the per-row XLA slope (the concat + device->host reframe the inproc port elides)
    buys ON TOP of removing the wire (tau_io/wakeup/tmsg). Surfaced honestly so the operator sees how much of
    this variant's lead over the staged-slope variants is the bare slope vs the removed wire. Returns both t_row
    values + both bounds + the irreducible-XLA asymptote at the bare slope."""
    x = initial_point(trust=trust)
    t_row_staged, _, _, _ = manifest.value("t_row_us", trust=trust)   # the baseline staged slope (transport-invariant)
    x_staged = dict(x)
    x_staged["t_row"] = t_row_staged
    return {
        "t_row_bare_us": x["t_row"],                    # the fully_device slope (this variant)
        "t_row_staged_us": t_row_staged,                # the run_microbatch slope (the wire-fed baseline)
        "bound_bare_t_row_dps": throughput_numpy(x),    # the headline (this variant)
        "bound_staged_t_row_dps": throughput_numpy(x_staged),  # what it would be on the staged slope (the wire removed but the concat kept)
        "asymptote_bare_dps": 1e6 / x["t_row"] / x["L"],       # B->inf at the bare slope (UNREACHABLE; the irreducible-XLA ceiling)
    }


def tmsg_capacity(trust: bool = True) -> dict[str, float]:
    """The transport-stage per-leaf capacity (dps) from `cpp_inproc_port_tmsg_us_leaf`, to confirm it is
    NON-BINDING by a wide margin (1/(LPD*tmsg*1e-6)). Reported, ranks last for the allocator."""
    tmsg, _, _, _ = manifest.value(TMSG_NAME, trust=trust)
    lpd, _, _, _ = manifest.value("LPD", trust=trust)
    return {"tmsg_us_leaf": tmsg, "transport_capacity_dps": 1.0 / (lpd * tmsg * 1e-6)}


def asymptote_dps(trust: bool = True) -> float:
    """The B->inf serve asymptote (1e6/t_row/L) at the BARE slope — reported as UNREACHABLE (max_batch caps B),
    never the bound. This is the irreducible-XLA ceiling for the inproc port (no transport can undercut the
    fully_device slope)."""
    t_row, _, _, _ = manifest.value(T_ROW_BARE_NAME, trust=trust)
    lpd, _, _, _ = manifest.value("LPD", trust=trust)
    return 1e6 / t_row / lpd


def bound(trust: bool = True) -> dict[str, Any]:
    """The variant's headline result: f(mu_hat), the binding stage, the cycle breakdown, the trusted/untrusted
    input map, the gather (copy) contrast, the bare-vs-staged t_row contrast, and the ratio to the ~203 plateau
    reference. The one call a report wants. `trust=False` computes the seed-only (first-principles) bound."""
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
        "producer_caps": producer_caps(trust=trust),
        "copy_contrast": copy_contrast(trust=trust),
        "bare_vs_staged_t_row_contrast": bare_vs_staged_t_row_contrast(trust=trust),
        "trusted": tflags,
        "all_trusted": all(tflags.values()),
        "untrusted_inputs": [nm for nm, t in tflags.items() if not t],
        "ref_plateau_dps": plat,
        "ratio_to_plateau": f / plat,
    }


# The ~203 dps empirical plateau is a USER-supplied reference for ONE config family (NOT grounded in any repo
# file, NOT a target). It is a model-side reference constant, not a measured quantity, so it is single-homed in
# the v1 grounding (`leaf_eval_grounding.REF_PLATEAU_DPS`); pulled from there so there is one home.
def ref_plateau_dps() -> float:
    """The ~203 dps empirical plateau reference (NOT a target). Single-homed in the v1 grounding; pulled from
    there so there is one home for the reference."""
    import leaf_eval_grounding as G
    return float(G.REF_PLATEAU_DPS)


if __name__ == "__main__":
    print(f"Design-{SLUG} — C++ IN-PROCESS queue-port transport leaf-eval throughput LOWER BOUND")
    print("=" * 78)
    plat = ref_plateau_dps()
    print(f"  postgres available: {manifest.postgres_available()}")

    for label, tr in (("TRUST (latest measured, else seed)", True), ("DISTRUST (v1/first-principles seeds)", False)):
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
        print("  producer caps:", {k: round(v, 1) for k, v in producer_caps(trust=tr).items()})
        cc = copy_contrast(trust=tr)
        print("  gather (copy) contrast (headline ELIDES the gather — contiguous arena; charged = scattered writes):",
              {k: round(v, 1) for k, v in cc.items()})
        bs = bare_vs_staged_t_row_contrast(trust=tr)
        print("  bare-vs-staged t_row contrast (transport-removed vs irreducible-XLA):",
              {k: round(v, 2) if "row" in k else round(v, 1) for k, v in bs.items()})
        print("  tmsg capacity (non-binding):",
              {k: round(v, 1) for k, v in tmsg_capacity(trust=tr).items()})
        untrusted = [nm for nm, t in tflags.items() if not t]
        print(f"  trusted inputs: {sum(tflags.values())}/{len(tflags)}; "
              f"untrusted (seed-only, NEEDS-SOLE-WORKLOAD): {untrusted}")

    print("\n  B->inf asymptote at the BARE slope (UNREACHABLE under max_batch cap; the irreducible-XLA ceiling):",
          round(asymptote_dps(), 1), "dps")
    saw = {r: round(serve_sawtooth(r), 1) for r in (64, 128, 192, 224, 256, 384, 512)}
    print("  serve sawtooth dps (real rows -> bucketed serve dps, bare t_row + inproc tau_io+wakeup):", saw)
    print(f"\n  NEYMAN RANKING — two complementary orderings (both honest):")
    print(f"    (a) DESIGN-PRIORITY (which transport DOF to characterize first): {SLUG}_tau_io_us "
          f"(the residual host->device staging — the dominant binding-stage lever) >> {SLUG}_gather_us "
          f"(is the staging arena contiguous? — the dominant transport uncertainty, the swing of tau_io's two "
          f"arms) >> {SLUG}_t_row_bare_us (the bare-vs-staged slope, the per-row XLA this variant moves) >> "
          f"{SLUG}_wakeup_us >> {SLUG}_tmsg_us_leaf.")
    print(f"    (b) VARIANCE-CONTRIBUTION (the allocator's data-driven c_i ranking): once SERVE-bound the cycle "
          f"is compute-dominated (B*t_row at the bare slope ~= {256*3.092:.0f}us of the ~{256*3.092+68.84+41.1:.0f}us "
          f"cycle), so t_row and L (then B) carry the largest a_i=(df/dx)^2*sigma^2 and the NeymanDriver funds "
          f"them first — surface BOTH: design-priority says measure tau_io + the gather to pin the transport; "
          f"variance says measure the bare t_row / L to pin E[f]. The two agree the transport terms are settled "
          f"once tau_io + the gather-elision are known; the residual CI is then a SERVE-physics (t_row/L) question.")
