"""
tools/analysis/OpenTURNS/model_lockfree_mpsc.py
===============================================

Transport variant LOCKFREE-MPSC: the leaf-eval throughput LOWER BOUND (dps) for a LOCK-FREE
MPSC (multi-producer / single-consumer) QUEUE transport — N=3 producer cores ENQUEUE leaf-eval
request nodes (a CAS on the tail, no per-message mutex, no broker), and the 1 serve core
BATCH-DEQUEUES all ready nodes into ONE forward (the coalescing moves from ZMQ's ROUTER into a
CAS-based queue). It composes the SAME serialized-serve-cycle structure the v1 cycle-time model
(model_cycletime.py) uses — `cycle_us = T_disp + wakeup + tau_io + B_eff*t_row`,
`dps = min(N_gen*R_gen, 1e6*B/(cycle*L))` evaluated at a FULL bucket (the serve sawtooth's
peak) — with THIS transport's (tau_io, wakeup, msg-cost) profile substituted. It is one model
module the generic `NeymanDriver` consumes (ADR-0012 P1/P2: the driver owns no model; this
module owns ITS math + reads every input through the manifest SSOT, never a hand-copied literal;
the NeymanDriver owns allocation; bench_store owns SQL).

WHAT THIS TRANSPORT MOVES (vs the ZMQ baseline; the rest of the cycle is INVARIANT):
  * tau_io  -> `lockfree_mpsc_tau_io_us`: the per-forward serial drain+scatter collapses from
    the ZMQ multipart/recv/send/per-message-codec (~20us seed) to a BATCH-DEQUEUE (CAS pops over
    T ready nodes) + the row GATHER + a reply-slot memcpy (~39.5us seed at B_op=256 WITH the
    gather charged). This is the DOMINANT lever the transport design controls and the binding-
    stage term — the top Neyman target while unmeasured.

    THE HONEST GATHER ASYMMETRY (vs shm_spin_poll). The MPSC queue's nodes are enqueued
    INDEPENDENTLY by N producers, so the B request rows are NOT contiguous in memory — a batched
    forward needs a GATHER (the existing C++ `WireLeafPool::submit_batch` already pays this exact
    "STRICT GATHER-BARRIER"; gather is INTRINSIC to coalescing, NOT what MPSC removes — MPSC
    removes the per-message ZMQ ENVELOPE). So unlike the shm ring (whose contiguous span lets the
    headline ELIDE the request copy and treats the copy as the pessimistic arm), the MPSC headline
    CHARGES the gather, and gather-ELISION (a staging path consuming a scatter/gather iovec list)
    is the OPTIMISTIC contrast. `gather_contrast()` shows both arms; the Neyman allocator ranks
    "is the gather elidable?" (`lockfree_mpsc_gather_us`) as its own dominant measurable question.

  * WAKEUP  -> `lockfree_mpsc_wakeup_us` (~0.1us): a HYBRID spin-then-park consumer at a chosen
    point on the spin<->futex axis. The consumer spins the atomic head for a bounded window and
    parks on a futex only if the window expires empty. At SATURATION (regime R2 — the regime this
    bound models) a node is essentially always already enqueued, so the consumer NEVER parks and
    pays the SPIN-phase cross-core cache-line coherence floor (~0.1us). Reported as a SEPARATE
    additive cycle term (the brief names it distinctly); negligible vs the ~1200us full-bucket
    cycle, but it MAKES EXPLICIT that the in-regime hybrid pays ~0 wakeup. The off-regime futex-
    park cost (~1-5us) is NOT folded in (provably not paid at saturation — see the wakeup bench).
  * msg-cost -> `lockfree_mpsc_tmsg_us_leaf` (~0.18us/leaf): a tail-CAS enqueue + slot write/read,
    no frame envelope, no corr-id. Reported (NON-BINDING by a wide margin), ranks LAST for the
    allocator (the wire request/reply CAPACITY never binds).

The cycle's INVARIANT inputs are pulled from the manifest by their registered names —
`T_disp_us`, `t_row_us`, `B_op`, `LPD`, `R_gen`, `n_gen` — UNCHANGED by the transport (the one
exception the brief carves out, cpp_inproc_port moving t_row, does NOT apply here: the MPSC queue
feeds the SAME staged `run_microbatch` forward, so t_row is the staged slope).

WHY A LOWER BOUND. Every term is the REAL staged forward cost (T_disp the pjit/XLA dispatch
floor, t_row the staged run_microbatch slope) with the MPSC tau_io ADDED on top at a CONSERVATIVE
memcpy bandwidth (8 B/ns — a slower copy gives a LARGER tau_io, a LOWER throughput), the gather
CHARGED in the headline (the honest scattered-node default), B held at a FULL bucket (not the
unreachable B->inf asymptote), and the producer ceiling a hard min taking the worse of the
4.0x/1.9x core scalings. It is a BENCH-dps bound (the input rates read higher than the closed-loop
e2e), contingent on the optimum reaching full-bucket feed and on the single-threaded serialization
being no worse than the tau_io charged. EVERY input is resolved through the manifest's TRUST
contract and the bound is FLAGGED when it rests on a seed (trusted=False) rather than a live
postgres measurement — ADR-0002: a seed is never silently reported as measured.

REFERENCE POINTS (NOT targets — workflow-brief-neutrality). The empirical ~203 dps plateau (one
config family) and the v1 ~429 (Design-B) / ~620 (the v1 inproc-port contrast) are reference
points the bound is compared against, never tuned toward. The MPSC headline landing serve-bound
just under the producer ceiling (gather charged) and the gather-elided arm landing producer-
ceiling-adjacent are real findings, reported as such.

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
SLUG = "lockfree_mpsc"
TAU_IO_NAME = "lockfree_mpsc_tau_io_us"
WAKEUP_NAME = "lockfree_mpsc_wakeup_us"
TMSG_NAME = "lockfree_mpsc_tmsg_us_leaf"
GATHER_NAME = "lockfree_mpsc_gather_us"     # the gather-elision contrast term (the dominant uncertainty; NOT a separate cycle term)

# The model signature — single home (`throughput_jax` + the numpy fallback `throughput_numpy` share it, P1). The cycle is
# T_disp + wakeup + tau_io + B*t_row; dps = min(producer, serve, transport-capacity). WAKEUP is a NAMED
# additive term (the brief names it separately) even though it is ~0 — so the structure makes the hybrid
# transport's saturation-regime zero-wakeup explicit. tmsg is the NON-BINDING transport-capacity arm.
INPUT_NAMES = ["N_gen", "R_gen", "B", "T_disp", "wakeup", "tau_io", "t_row", "L", "tmsg"]

# The manifest quantity name each model input is pulled from (1:1 with INPUT_NAMES). The invariant inputs
# carry the baseline registered names; the moved terms carry the prefixed mpsc names. This map is the model's
# ONLY coupling to the registry — there are no hand-copied literals (ADR-0012 P1).
_MANIFEST_NAME: dict[str, str] = {
    "N_gen": "n_gen",
    "R_gen": "R_gen",
    "B": "B_op",
    "T_disp": "T_disp_us",
    "wakeup": WAKEUP_NAME,
    "tau_io": TAU_IO_NAME,
    "t_row": "t_row_us",
    "L": "LPD",
    "tmsg": TMSG_NAME,
}

# Per-input benchmark COST (relative; the Neyman allocator's c_i). The moved terms are cheap microbenches;
# B_op/LPD/R_gen are the expensive end-to-end/saturation reads (mirrors leaf_eval_grounding's costs). Homed
# here so the model carries its own allocation costs (the manifest stores physical quantities, not costs).
_COST: dict[str, float] = {
    "N_gen": 0.5, "R_gen": 30.0, "B": 4.0, "T_disp": 1.0,
    "wakeup": 6.0, "tau_io": 8.0, "t_row": 1.0, "L": 2.0, "tmsg": 2.0,
}


def throughput_numpy(x: dict[str, float]) -> float:
    """numpy-only evaluation of f (the fallback path + the lockstep cross-check of
    throughput_jax). SAME formula as throughput_jax. B is the FULL-bucket width (pad ~= B). The transport capacity is a
    separate non-binding min arm."""
    producer = x["N_gen"] * x["R_gen"]
    cycle_us = x["T_disp"] + x["wakeup"] + x["tau_io"] + x["B"] * x["t_row"]
    serve = 1e6 * x["B"] / (cycle_us * x["L"])
    transport = 1.0 / (x["L"] * x["tmsg"] * 1e-6)
    return float(min(producer, serve, transport))


def throughput_jax(x: Any) -> Any:
    """The single JAX-traceable throughput f (x ordered by INPUT_NAMES) — the OT→JAX migration's one home
    for f (§5): `jax.grad(throughput_jax)` is the gradient (analytic, exact-through-`min()`; the arm-tie is
    handled by alloc.kink, not the linearization), evaluating identically to `throughput_numpy` (pinned in
    tests/test_jax_f_equivalence.py). The model's single f the driver consumes (the OT string THROUGHPUT_EXPR is retired; the numpy twin throughput_numpy retires with the numpy fallback in migration J4)."""
    from alloc.jax_backend import jnp
    N_gen, R_gen, B, T_disp, wakeup, tau_io, t_row, L, tmsg = x
    producer = N_gen * R_gen
    cycle_us = T_disp + wakeup + tau_io + B * t_row
    serve = 1e6 * B / (cycle_us * L)
    transport = 1.0 / (L * tmsg * 1e-6)
    return jnp.minimum(jnp.minimum(producer, serve), transport)


# --------------------------------------------------------------------------- #
# Manifest resolution — pull every input through the TRUST contract; branch on `trusted`.
# --------------------------------------------------------------------------- #
def resolve_inputs(trust: bool = True) -> dict[str, manifest.Quantity]:
    """Resolve every model input to a `manifest.Quantity` (mean/sigma/n/trusted/source). The model's ONE
    read of the registry: each INPUT_NAMES key maps to its manifest quantity name and is pulled through
    `manifest.quantity(..., trust=trust)`. trust=True returns the live measurement when one exists (and
    flags trusted=True), else falls back to the seed (trusted=False). NEVER passes rerun=True (timing-
    sensitive — an operator action). A quantity that is not registered is a loud KeyError (ADR-0002)."""
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
    """Per-input `trusted` bool: True iff the value is a live postgres measurement, False iff it is a seed.
    The model BRANCHES on this to flag the bound as resting on unmeasured inputs; the Neyman loop ranks the
    untrusted binding terms."""
    return {nm: q.trusted for nm, q in resolve_inputs(trust=trust).items()}


def untrusted_inputs(trust: bool = True) -> list[str]:
    """The model inputs currently resting on a SEED (trusted=False) rather than a live postgres measurement
    — the ADR-0002 honesty surface: the bound is flagged as resting on these, and the Neyman allocator ranks
    them for the next sole-workload bench. Returns the INPUT_NAMES (model-side names)."""
    return [nm for nm, q in resolve_inputs(trust=trust).items() if not q.trusted]


# Module-level resolved views (TRUST) — the SIGMAS/COSTS the runner reads, same attribute surface as
# model_capacity / model_cycletime / model_zmq_baseline. Resolved at import (one manifest pass).
SIGMAS: dict[str, float] = sigmas(trust=True)
COSTS: dict[str, float] = costs()


def build_driver(tolerance: float = 5.0, trust: bool = True) -> tuple[Any, dict[str, float]]:
    """Factory: a configured `NeymanDriver` (over the JAX `f` (`throughput_jax`), the per-input costs) + the resolved
    initial point. `tolerance` is the target CI half-width on E[f] in dps. `trust` selects live-vs-seed
    inputs. Imported lazily (deferred to keep this module import-cheap)."""
    from neyman_driver import NeymanDriver
    f = throughput_jax  # the driver consumes the JAX-traceable f directly (OT→JAX migration, §5)
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
    caller sees what binds and which moved term dominates. The MPSC transport's tau_io + wakeup are the terms
    it moved; compute = B*t_row is the invariant forward work."""
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
    """Producer ceiling under both core scalings (4.0x measured C++ vs 1.9x Python-ExIt worst), pulled from
    the manifest R_gen — a lower bound surfaces the worse case absent evidence the C++ path escapes the
    contention."""
    rg, _, _, _ = manifest.value("R_gen", trust=trust)
    return {"producer_4.0x_dps": 3 * rg, "producer_1.9x_dps": 1.9 * rg}


def serve_sawtooth(real: int, buckets=(64, 256, 512), max_batch=512, trust: bool = True) -> float:
    """Serve dps at `real` rows under BUCKETING + the MPSC tau_io + wakeup terms — the SAWTOOTH (drops at
    bucket edges; peaks at full buckets). Snap up to the smallest bucket >= real; past the top bucket run
    unpadded at width=real. Demonstrates the non-monotonicity (a larger B is not unconditionally
    conservative). Pulls the cycle terms from the manifest."""
    x = initial_point(trust=trust)
    pad = next((b for b in buckets if b >= real), real)
    if real > max_batch:
        pad = real
    cycle = x["T_disp"] + x["wakeup"] + x["tau_io"] + pad * x["t_row"]
    return 1e6 * real / (cycle * x["L"])


def gather_contrast(trust: bool = True) -> dict[str, float]:
    """The OPTIMISTIC arm: SUBTRACT the request-GATHER (`lockfree_mpsc_gather_us`) from the headline tau_io,
    i.e. the bound IF the host->device staging consumes a scatter/gather iovec list and the B rows are NOT
    materialized contiguous. Computed and surfaced honestly (NOT replacing the headline) so the operator sees
    what gather-elision buys + the Neyman allocator can rank "is the gather elidable?" as its own measurable
    question. This is the MPSC mirror of shm_spin_poll's copy_both_contrast, INVERTED: the shm headline
    elides its copy (the copy is the pessimistic arm); the MPSC headline CHARGES the gather (elision is the
    optimistic arm) — because scattered nodes make gather-elision LESS plausible than a contiguous ring's
    span. Returns the gather term + both arms' bounds.

    The seed tau_io already INCLUDES the gather (see bench_lockfree_mpsc_tau_io.get_seed), so the elided arm
    subtracts the gather term; clamped to a small positive floor so the cycle stays well-formed if a future
    measured gather exceeds the measured tau_io (which would itself be a loud inconsistency to investigate)."""
    x = initial_point(trust=trust)
    gather, _, _, _ = manifest.value(GATHER_NAME, trust=trust)
    x_el = dict(x)
    x_el["tau_io"] = max(x["tau_io"] - gather, 0.5)   # elide the gather (floor keeps the cycle well-formed)
    return {
        "gather_us": gather,
        "tau_io_headline_us": x["tau_io"],            # gather CHARGED (the honest default)
        "tau_io_gather_elided_us": x_el["tau_io"],    # gather ELIDED via iovec staging (optimistic)
        "bound_headline_dps": throughput_numpy(x),
        "bound_gather_elided_dps": throughput_numpy(x_el),
    }


def tmsg_capacity(trust: bool = True) -> dict[str, float]:
    """The transport-stage per-leaf capacity (dps) from `lockfree_mpsc_tmsg_us_leaf`, to confirm it is
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


def bound(trust: bool = True) -> dict[str, Any]:
    """The variant's headline result: f(mu_hat), the binding stage, the cycle breakdown, the
    trusted/untrusted input map, the gather contrast, and the ratio to the ~203 plateau reference. The one
    call a report wants. `trust=False` computes the seed-only (first-principles) bound."""
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
        "gather_contrast": gather_contrast(trust=trust),
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
    print(f"Design-{SLUG} — LOCK-FREE MPSC queue transport leaf-eval throughput LOWER BOUND")
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
        gc = gather_contrast(trust=tr)
        print("  gather contrast (headline CHARGES the gather; elided = scatter/gather staging):",
              {k: round(v, 1) for k, v in gc.items()})
        print("  tmsg capacity (non-binding):",
              {k: round(v, 1) for k, v in tmsg_capacity(trust=tr).items()})
        untrusted = [nm for nm, t in tflags.items() if not t]
        print(f"  trusted inputs: {sum(tflags.values())}/{len(tflags)}; "
              f"untrusted (seed-only, NEEDS-SOLE-WORKLOAD): {untrusted}")

    print("\n  B->inf asymptote (UNREACHABLE under max_batch cap):", round(asymptote_dps(), 1), "dps")
    saw = {r: round(serve_sawtooth(r), 1) for r in (64, 128, 192, 224, 256, 384, 512)}
    print("  serve sawtooth dps (real rows -> bucketed serve dps, incl. tau_io+wakeup):", saw)
    print(f"\n  NEYMAN RANKING — two complementary orderings (both honest):")
    print(f"    (a) DESIGN-PRIORITY (which transport DOF to characterize first): {SLUG}_gather_us "
          f"(is the gather elidable? — the dominant transport uncertainty, ~31us swing) >> "
          f"{SLUG}_tau_io_us >> {SLUG}_wakeup_us >> {SLUG}_tmsg_us_leaf.")
    print(f"    (b) VARIANCE-CONTRIBUTION (the allocator's data-driven c_i ranking, once the bound is "
          f"SERVE-bound): the cycle is compute-dominated (B*t_row ~= {256*4.317:.0f}us of the ~1214us "
          f"cycle), so t_row and L (then B) carry the largest a_i=(df/dx)^2*sigma^2 and the NeymanDriver "
          f"funds them first — surface BOTH: design-priority says measure the gather to pin the transport, "
          f"variance says measure t_row/L to pin E[f]. The two agree that the transport terms are settled "
          f"once gather is known; the residual CI on the bound is then a SERVE-physics (t_row/L) question.")
