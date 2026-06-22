"""
tools/analysis/leaf_eval_bound/transport_sweep.py
===========================================

The TRANSPORT-DESIGN-SPACE SWEEP: the synthesis runner that, for EACH leaf-eval transport
variant (`model_<slug>.py`), computes a first-principles throughput LOWER BOUND (dps, with a
delta-method CI), ranks the variants = the OPTIMUM-OVER-TRANSPORTS, and prints the top Neyman
live-benchmark targets per transport (what to run sole-workload next). The five variant models
are the things transported; this runner + the generic `AllocationDriver` are the transport
(ADR-0012 P2 separation — the runner owns no model's math; each model owns its own).

WHY A SWEEP, NOT "model the current system". Modelling ZMQ faithfully just regenerates ~200 dps
(a coordination artifact of one config). The DESIGN VARIABLE is the transport/wakeup mechanism
itself: the space spans ZMQ block/wakeup (the baseline) -> shared-memory spin-poll / futex wake
/ lock-free MPSC queue / C++ in-process queue-port. Each variant = a different (tau_io,
wakeup-latency, message-cost [, t_row for the inproc port]) profile feeding the SAME serialized-
serve-cycle spine (model_cycletime.py). The sweep holds the genuine per-component knowns fixed
and moves only the transport-attributable terms, so the optimum-over-transports is what a
WELL-DESIGNED control cycle achieves, and the gap to ~203 is how far one config sits below it.

WHAT THIS RUNNER ADDS OVER EACH MODEL'S OWN __main__ (the SYNTHESIS, single-homed here per
ADR-0012 P1 — these cross-variant corrections live in ONE place, not copied into five models):

  (1) THREE HONESTY LEVELS per variant, each computed through the model's OWN
      `throughput_numpy` / `build_driver` (the model is reported faithfully; this runner never
      silently rewrites a reviewed model's headline):

        * HEADLINE      — the variant's own f(mu_hat) exactly as the model authored it
                          (producer ceiling = N_gen*R_gen, the 4.0x-linear C++ gen arm; the
                          model's own host<->device transfer accounting).
        * CONSERVATIVE  — the strictly-defensible floor: the two cross-cutting corrections EVERY
                          per-variant critique demanded, applied uniformly:
                            (a) PRODUCER worst case. Every model docstring claims a "hard min
                                taking the WORSE of the 4.0x and 1.9x core scalings", but every
                                model's headline min() uses only the 4.0x arm (3*R_gen). This
                                runner takes the 1.9x arm (1.9*R_gen) as a hard min — making the
                                "worse case" the models CLAIM actually computed. (Numerically the
                                1.9x Python-ExIt contention may not apply to the C++ gen path; it
                                is reported as the conservative floor, NOT asserted as the truth.)
                            (b) HOST<->DEVICE transfer residual. The cycle's T_disp = 68.84us is
                                the fully_device DISPATCH FLOOR, which DELIBERATELY EXCLUDES the
                                input H2D crossing (5.52us) + output D2H pull (9.14us) that live
                                in the measured STAGED intercept (94.58us). A usable leaf-eval
                                MUST cross both (run_microbatch does one np.asarray pull to expand
                                the tree). This runner adds that residual back per a PER-VARIANT
                                policy (below), so the forward portion is >= the measured staged
                                forward — restoring a strict floor (the gap the lockfree/futex/cpp
                                critiques flagged; inherited from the v1 model_cycletime spine).
        * CI            — the grounded-uncertainty delta-method CI half-width on E[f], via the
                          model's `build_driver()` Neyman step (the JAX-driven AllocationDriver).

  (2) The PER-VARIANT TRANSFER POLICY (the per-variant fact the critiques established; declared
      here with provenance, NOT blanket-applied):
        * the four STAGED-slope variants (zmq_baseline, shm_spin_poll, futex_wake, lockfree_mpsc)
          feed the SAME run_microbatch forward whose T_disp omits BOTH crossings, so the full
          staged residual (94.58 - 68.84 = 25.74us) is added.
        * cpp_inproc_port ALREADY charges the H2D input crossing inside its own tau_io (its tau_io
          IS "the host->device crossing of the gathered B-row input block"), so adding the full
          residual would DOUBLE-COUNT the input transfer; it gets the D2H OUTPUT PULL only
          (9.14us) — the term its fully_device geometry genuinely omits (a usable consumer must
          read value+logits back to host to expand the tree).

  (3) The OPTIMUM-OVER-TRANSPORTS under both readings, the comparison table vs the references
      (~203 plateau; v1 Design-B ~429 / the v1 inproc-port contrast ~620), and the per-transport
      top Neyman targets in TWO complementary orderings:
        * VARIANCE-CONTRIBUTION — the allocator's data-driven c_i ranking (what most tightens the
          bound's CI). It funds whatever BINDS: the compute terms (t_row/L/B) when serve-bound,
          the producer terms (N_gen/R_gen) when generation-bound.
        * DESIGN-PRIORITY (transport-moved terms only) — which TRANSPORT DOF to characterize
          first (tau_io, wakeup, gather/req_drain, tmsg). These carry small a_i precisely because
          the cycle is compute-dominated at a full bucket, so the variance ranking alone would
          never surface them — but they are the levers the transport DESIGN moves, hence "what to
          run sole-workload next to pin the transport".

HONESTY (workflow-brief-neutrality; ADR-0002). ~203 and the v1 ~429/~620 are REFERENCE POINTS,
never targets. EVERY model input currently resolves trusted=False (a seed flagged NEEDS-SOLE-
WORKLOAD — no sole-workload bench has populated postgres yet), so EVERY bound below is a first-
principles SEED ESTIMATE, not a measured floor. The sweep prints the grounded-vs-unmeasured split
and the per-variant measurement targets so the reader knows exactly what a sample would support
vs what is still a seed. The benches are registered + written + left runnable (timing-sensitive —
an operator runs each pinned, taskset -c 0, sole-workload; the manifest's trusted flag then flips
to True automatically with no model edit).

HOW THE BOUND IS COMPUTED. The HEADLINE + CONSERVATIVE bounds and the binding-stage logic are pure
numpy over each model's diagnostics; the CI half-width + the VARIANCE ranking are the JAX-driven
`AllocationDriver` step (the OT→JAX migration — this runner imports no openturns, and the old numpy
delta-method fallback retired with the single-f collapse, J4). A driver failure RAISES loudly
(ADR-0002) — there is no silent fallback.

Run: /home/bork/w/vdc/venvs/generic/bin/python -m leaf_eval_bound.runners.transport_sweep   (from tools/analysis, or PYTHONPATH=tools/analysis)

Public Domain (The Unlicense).
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any, cast  # (Callable dropped with the runner _fd_gradient — move 5; Optional was already unused)


from leaf_eval_bound.contract import grounding as G  # noqa: E402  — the single home of the references + the transfer decomposition
from leaf_eval_bound.contract import references
from leaf_eval_bound.store import manifest  # noqa: E402   — the SSOT registry (import-clean; touches no DB on import)
from leaf_eval_bound.models.model_base import TransportModel  # noqa: E402  — the typed transport-variant contract (move 3)


# --------------------------------------------------------------------------- #
# The transport variants under sweep + their per-variant transfer policy. This list is the sweep's
# ONE coupling to the variant models; adding a variant is appending a TransportVariant row + its
# model_<slug>.py (no edit elsewhere). The transfer policy is the per-variant fact the critiques
# established — homed HERE (a cross-variant synthesis concern), with provenance, not in the models.
# --------------------------------------------------------------------------- #
# The host<->device residual the fully_device DISPATCH FLOOR (T_disp=68.84) omits, decomposed in the
# v1 grounding (leaf_eval_grounding docstring of SERVE_INTERCEPT_US / DISPATCH_FLOOR_US):
#   staged intercept (94.58) - dispatch floor (68.84) = 25.74us = input_transfer(5.52) + output_pull(9.14)
#   + residual(11.08). These are read from the grounding so there is ONE home for the numbers.
_STAGED_INTERCEPT_US = G.SERVE_INTERCEPT_US.mean          # 94.58 — staged run_microbatch intercept
_DISPATCH_FLOOR_US = G.DISPATCH_FLOOR_US                  # 68.84 — fully_device dispatch floor (the cycle's T_disp)
_H2D_D2H_RESIDUAL_US = _STAGED_INTERCEPT_US - _DISPATCH_FLOOR_US   # 25.74 — both crossings + residual
_OUTPUT_PULL_US = 9.14                                    # mlp_lowlatency decomposition.output_pull_us (D2H only)


@dataclass(frozen=True)
class TransportVariant:
    """One transport variant under sweep: its model module name, a human label of WHAT TERM the
    transport moves, and its per-variant H<->D transfer residual (the conservative-arm correction
    the critiques established). `transfer_residual_us` is ADDED to T_disp in the CONSERVATIVE arm
    only; `transfer_note` records why that residual (provenance for the per-variant policy)."""
    module: str
    moves: str                       # the term(s) this transport moves off the baseline (for the report)
    transfer_residual_us: float      # the H<->D crossing the model's T_disp omits (conservative-arm add)
    transfer_note: str               # provenance for the per-variant residual choice


# The DESIGN-PRIORITY transport-moved terms per variant slug (the levers the transport actually
# moves — what "run sole-workload next to pin the transport" means). The variance ranking funds the
# binding stage (compute/producer); this names the TRANSPORT DOF the variance ranking buries because
# they are small at a full bucket. Pulled by the variant's registered quantity names.
_TRANSPORT_MOVED_TERMS: dict[str, list[str]] = {
    "zmq_baseline": ["zmq_baseline_tau_io_us", "zmq_baseline_wakeup_us", "zmq_baseline_tmsg_us_leaf"],
    "shm_spin_poll": ["shm_spin_poll_tau_io_us", "shm_spin_poll_req_drain_us",
                      "shm_spin_poll_wakeup_us", "shm_spin_poll_tmsg_us_leaf"],
    "futex_wake": ["futex_wake_tau_io_us", "futex_wake_wakeup_us",
                   "futex_wake_req_drain_us", "futex_wake_tmsg_us_leaf"],
    "lockfree_mpsc": ["lockfree_mpsc_tau_io_us", "lockfree_mpsc_gather_us",
                      "lockfree_mpsc_wakeup_us", "lockfree_mpsc_tmsg_us_leaf"],
    "cpp_inproc_port": ["cpp_inproc_port_tau_io_us", "cpp_inproc_port_gather_us",
                        "cpp_inproc_port_t_row_bare_us", "cpp_inproc_port_wakeup_us",
                        "cpp_inproc_port_tmsg_us_leaf"],
}


VARIANTS: list[TransportVariant] = [
    TransportVariant(
        module="leaf_eval_bound.models.model_zmq_baseline",
        moves="nothing (the REFERENCE: ZMQ ROUTER/DEALER multipart, poll-block wakeup)",
        transfer_residual_us=_H2D_D2H_RESIDUAL_US,
        transfer_note="staged-slope forward; T_disp omits both H2D+D2H -> full staged residual",
    ),
    TransportVariant(
        module="leaf_eval_bound.models.model_shm_spin_poll",
        moves="tau_io (ring memcpy, zero-copy req drain) + wakeup (~0, burnt poll core)",
        transfer_residual_us=_H2D_D2H_RESIDUAL_US,
        transfer_note="staged-slope forward; T_disp omits both H2D+D2H -> full staged residual",
    ),
    TransportVariant(
        module="leaf_eval_bound.models.model_futex_wake",
        moves="wakeup (FUTEX_WAKE edge handoff, no burnt core) + tau_io (= shm ring drain)",
        transfer_residual_us=_H2D_D2H_RESIDUAL_US,
        transfer_note="staged-slope forward; T_disp omits both H2D+D2H -> full staged residual",
    ),
    TransportVariant(
        module="leaf_eval_bound.models.model_lockfree_mpsc",
        moves="tau_io (CAS batch-dequeue + row gather charged) + wakeup (hybrid spin-then-park)",
        transfer_residual_us=_H2D_D2H_RESIDUAL_US,
        transfer_note="staged-slope forward; T_disp omits both H2D+D2H -> full staged residual",
    ),
    TransportVariant(
        module="leaf_eval_bound.models.model_cpp_inproc_port",
        moves="t_row (BARE fully_device slope) + tau_io (residual H2D crossing) + wakeup (~0)",
        transfer_residual_us=_OUTPUT_PULL_US,
        transfer_note="tau_io ALREADY charges the H2D input crossing -> add D2H output pull ONLY "
                      "(avoid double-counting the input transfer)",
    ),
]


# --------------------------------------------------------------------------- #
# Per-variant evaluation — headline (model's own), conservative floor (the two corrections), CI.
# --------------------------------------------------------------------------- #
@dataclass
class VariantResult:
    slug: str
    moves: str
    headline_dps: float                       # the model's own f(mu_hat) (4.0x producer, model transfer accounting)
    headline_binding: str
    serve_floor_dps: float                    # headline + H<->D transfer residual, STILL 4.0x producer (isolates
    serve_floor_binding: str                  #   the missing-transfer fix the valid=false critiques demanded)
    conservative_dps: float                   # 1.9x producer min + H<->D transfer residual (the strict floor)
    conservative_binding: str
    producer_4x_dps: float
    producer_19x_dps: float
    transfer_residual_us: float
    transfer_note: str
    ci_halfwidth: float                       # delta-method CI half-width on E[f]
    ci_source: str                            # 'jax' (the AllocationDriver step; the numpy fallback retired, J4)
    all_trusted: bool
    untrusted_inputs: list[str]
    variance_targets: list[tuple[str, float, int]]   # (model-input name, a_i, +samples) ranked desc
    transport_targets: list[tuple[str, float, bool]]  # (registry quantity, seed-mean, trusted) — design-priority
    cycle_breakdown: dict[str, float] = field(default_factory=dict)
    contrasts: dict[str, Any] = field(default_factory=dict)


def _binding_stage(x: dict[str, float], producer_dps: float, model: TransportModel) -> tuple[float, str]:
    """The serve dps + which of {GENERATION, SERVE, TRANSPORT} binds, given a producer ceiling. Uses
    the model's own cycle_breakdown for the serve/transport capacities so the structure matches the
    model exactly; substitutes the supplied producer ceiling for the min (the conservative arm passes
    the 1.9x ceiling, the headline passes 3x via the model's own throughput_numpy)."""
    cb = model.cycle_breakdown(x)
    serve = cb["serve_dps"]
    transport = cb.get("transport_dps", float("inf"))
    caps = {"GENERATION": producer_dps, "SERVE": serve, "TRANSPORT": transport}
    binding = min(caps, key=caps.get)
    return caps[binding], binding


def _serve_floor_bound(model: TransportModel, variant: TransportVariant,
                       x: dict[str, float]) -> tuple[float, str]:
    """The TRANSFER-CORRECTED serve floor: add the per-variant H<->D transfer residual into T_disp but
    KEEP the 4.0x-linear producer ceiling (3*R_gen). This ISOLATES the one fix the two valid=false
    critiques (lockfree_mpsc, cpp_inproc_port) demanded — the host<->device crossing the fully_device
    T_disp omits — WITHOUT the (separate, far larger) 1.9x-producer correction swamping it. For the
    wire-fed SERVE-bound variants this lands ~9 dps below the headline (the missing-transfer term made
    visible); for the GENERATION-bound inproc port it is unchanged (transfer is off the binding stage,
    and its policy is output-pull-only anyway)."""
    x_c = dict(x)
    x_c["T_disp"] = x["T_disp"] + variant.transfer_residual_us
    producer_4x = x["N_gen"] * x["R_gen"]
    return _binding_stage(x_c, producer_4x, model)


def _conservative_bound(model: TransportModel, variant: TransportVariant,
                        x: dict[str, float]) -> tuple[float, str]:
    """The strictly-defensible floor: add the per-variant H<->D transfer residual into T_disp AND take
    the 1.9x producer ceiling as a hard min. Computed via the model's OWN cycle structure (so the serve
    cycle is identical bar the +residual on the dispatch term)."""
    x_c = dict(x)
    x_c["T_disp"] = x["T_disp"] + variant.transfer_residual_us
    rg = x["R_gen"]
    producer_19x = 1.9 * rg
    bound, binding = _binding_stage(x_c, producer_19x, model)
    return bound, binding


def _ci_via_driver(model: TransportModel) -> tuple[float, str]:
    """The delta-method CI half-width on E[f] via the model's `build_driver()` + each input fed as its
    harmonized `Estimate` (§6 Phase 4 — `driver.set_estimates_by_name`, REPLACING the fabricated 2-point
    `{mean±sigma}` pilot). The grounded inputs are `Fixed`/declared-spread Estimates (`cov=[[sigma^2]]`),
    so `g^T Σ g` is byte-for-byte the old pilot's bound (the `{mean±sigma}` set's std is √2·sigma, so
    a_i/n_i = grad^2·sigma^2 either way — no `/2` bug) and the grounded mean still anchors the gradient at
    the binding stage (the min() kink makes it point-sensitive). Returns (ci_halfwidth, 'jax'). A driver
    failure RAISES (ADR-0002 — there is no silent fallback; the numpy delta-method fallback retired with
    the single-f collapse, J4)."""
    driver, x0 = model.build_driver(tolerance=0.1, trust=True)
    driver.set_estimates_by_name(_model_estimates(model))
    rec = driver.step(second_order_check=False)
    return rec.ci_halfwidth, "jax"


def _variance_targets(model: TransportModel, top: int = 6) -> list[tuple[str, float, int]]:
    """The allocator's VARIANCE-CONTRIBUTION ranking (which input most tightens E[f]'s CI): (model-input
    name, a_i=(df/dx)^2*sigma^2, recommended +samples), ranked desc. Via the JAX-driven driver step (§6
    Phase 4 — each input fed as its harmonized `Estimate` via `set_estimates_by_name`). `+samples` is 0
    for a declared-spread prior (un-shrinkable — the §2.3 allocator funds none). This funds whatever BINDS
    (compute when serve-bound, producer when generation-bound). A driver failure RAISES (ADR-0002)."""
    driver, x0 = model.build_driver(tolerance=0.1, trust=True)
    driver.set_estimates_by_name(_model_estimates(model))
    rec = driver.step(second_order_check=False)
    ranked = sorted(rec.primitives, key=lambda p: p.a, reverse=True)
    return [(p.name, float(p.a), int(p.recommend)) for p in ranked[:top]]


def _transport_targets(slug: str) -> list[tuple[str, float, bool]]:
    """The DESIGN-PRIORITY ranking restricted to the transport-MOVED terms (the levers this transport
    moves; what to run sole-workload next to pin the transport). Returns (registry quantity name,
    current seed mean, trusted) for each moved term, ordered as declared (dominant lever first). These
    are exactly the quantities whose benches an operator runs pinned + sole-workload to flip the bound
    from seed-estimate to grounded floor."""
    out: list[tuple[str, float, bool]] = []
    for qname in _TRANSPORT_MOVED_TERMS.get(slug, []):
        try:
            q = manifest.quantity(qname, trust=True)
            out.append((qname, q.mean, q.trusted))
        except Exception as exc:  # noqa: BLE001 — a missing moved-term quantity is a loud gap, not silent
            print(f"[transport_sweep] moved-term {qname!r} unresolved "
                  f"({type(exc).__name__}: {exc}).", file=sys.stderr)
            out.append((qname, float("nan"), False))
    return out


def _model_estimates(model: TransportModel) -> dict[str, "Any"]:
    """The §6 Phase-4 input feed: `{model_input_name: manifest.Estimate}` for `driver.set_estimates_by_name`,
    REPLACING the fabricated 2-point `{mean±sigma}` pilot. Each input is resolved to its harmonized
    `Estimate` straight from the manifest (`manifest.estimate(qname, trust=True)`) — a `Fixed`/declared-spread
    seed today (the bound rests on the grounded prior, `cov=[[sigma^2]]` un-divided), automatically the real
    measured Estimate (Poolwise/QuantileLaw/fit) once a sole-workload bench flips the quantity to trusted (no
    code change). `g^T Σ g` over these is byte-for-byte the old 2-point pilot's `sum a_i/n_i` (the `{mean±sigma}`
    set's sample-std is √2·sigma, so a_i/n_i = grad^2·sigma^2 either way — the spec REFUTED a claimed `/2`
    bug). A declared-spread prior is un-shrinkable, so the §2.3 allocator funds none; the design-priority /
    variance rankings (which read a_i) are unchanged."""
    return {nm: manifest.estimate(model.registry_qname(nm), trust=True) for nm in model.INPUT_NAMES}


def _untrusted(model: TransportModel) -> tuple[bool, list[str]]:
    """(all_trusted, untrusted_input_names) — the ADR-0002 honesty surface. Every transport model now
    exposes the canonical `trusted_flags(trust)` (move 3b grew it on shm_spin_poll + futex_wake), so the
    old 3-fallback that sniffed untrusted_inputs / trusted_flags / resolve_inputs collapses to the one
    surface. (`trusted_flags` and the models' `untrusted_inputs` both derive from the same `resolve_inputs`
    pull, so this is byte-for-byte the old result — the untrusted list keeps its INPUT_NAMES order.)"""
    tf = model.trusted_flags(trust=True)
    return all(tf.values()), [nm for nm, t in tf.items() if not t]


def evaluate_variant(variant: TransportVariant) -> VariantResult:
    """Compute one variant's headline + conservative bounds, CI, trust state, and both Neyman rankings.
    The headline is the model's OWN f(mu_hat) (reported faithfully); the conservative arm applies the
    two cross-cutting corrections (1.9x producer + H<->D transfer residual)."""
    model: TransportModel = cast(TransportModel, importlib.import_module(variant.module))
    x = model.initial_point(trust=True)

    headline = model.throughput_numpy(x)
    producer_4x = x["N_gen"] * x["R_gen"]
    headline_bound, headline_binding = _binding_stage(x, producer_4x, model)

    serve_floor, serve_floor_binding = _serve_floor_bound(model, variant, x)
    conservative, conservative_binding = _conservative_bound(model, variant, x)

    ci, ci_src = _ci_via_driver(model)
    all_trusted, untrusted = _untrusted(model)
    variance_targets = _variance_targets(model)
    transport_targets = _transport_targets(model.SLUG)

    cb = model.cycle_breakdown(x)
    contrasts = _variant_contrasts(model)

    return VariantResult(
        slug=model.SLUG,
        moves=variant.moves,
        headline_dps=headline,
        headline_binding=headline_binding,
        serve_floor_dps=serve_floor,
        serve_floor_binding=serve_floor_binding,
        conservative_dps=conservative,
        conservative_binding=conservative_binding,
        producer_4x_dps=producer_4x,
        producer_19x_dps=1.9 * x["R_gen"],
        transfer_residual_us=variant.transfer_residual_us,
        transfer_note=variant.transfer_note,
        ci_halfwidth=ci,
        ci_source=ci_src,
        all_trusted=all_trusted,
        untrusted_inputs=untrusted,
        variance_targets=variance_targets,
        transport_targets=transport_targets,
        cycle_breakdown=cb,
        contrasts=contrasts,
    )


def _variant_contrasts(model: TransportModel) -> dict[str, Any]:
    """The variant's own design-question contrasts (zero-copy? gather elidable? bare-vs-staged slope?),
    each computed by the model's own helper when present. Reported so the sweep surfaces each variant's
    dominant transport uncertainty, not just its headline."""
    out: dict[str, Any] = {}
    for helper in ("copy_both_contrast", "copy_contrast", "gather_contrast",
                   "bare_vs_staged_t_row_contrast", "saturation_wakeup_contrast", "tmsg_capacity"):
        fn = getattr(model, helper, None)
        if callable(fn):
            try:
                out[helper] = fn(trust=True)
            except Exception as exc:  # noqa: BLE001 — a contrast that fails is shown, not swallowed
                out[helper] = f"<{type(exc).__name__}: {exc}>"
    return out


# --------------------------------------------------------------------------- #
# The sweep + the report.
# --------------------------------------------------------------------------- #
def run_sweep() -> list[VariantResult]:
    """Evaluate every variant. Returns the VariantResult list (the comparison data the report renders)."""
    return [evaluate_variant(v) for v in VARIANTS]


def _ref_v1() -> dict[str, float]:
    """The v1 reference bounds (NOT targets): Design-B cycle-time ~429, the v1 inproc-port contrast ~620
    (model_cycletime.inproc_port_contrast at full-512). Pulled live from the v1 model so the references
    track the v1 home rather than a hand-copied literal."""
    from leaf_eval_bound.models import model_cycletime as v1
    x0 = v1.initial_point()
    design_b = v1.throughput_numpy(x0)
    inproc = v1.inproc_port_contrast(full_batch=512)["dps_at_full_batch"]
    return {"design_b_dps": design_b, "inproc_port_contrast_dps": inproc}


def print_report(results: list[VariantResult]) -> None:
    plat = float(references.REF_PLATEAU_DPS)
    v1 = _ref_v1()
    pg = manifest.postgres_available()

    print("=" * 92)
    print("LEAF-EVAL TRANSPORT-DESIGN SWEEP — first-principles throughput LOWER BOUNDS per transport")
    print("=" * 92)
    print(f"postgres (metric store) available: {pg}")
    print(f"Reference points (NOT targets — workflow-brief-neutrality):")
    print(f"  empirical plateau ~{plat:.0f} dps (user-supplied, ONE config family on the current harness)")
    print(f"  v1 Design-B cycle-time bound ~{v1['design_b_dps']:.0f} dps   |   "
          f"v1 inproc-port contrast ~{v1['inproc_port_contrast_dps']:.0f} dps (full-512, serve-only)")
    print(f"  measured anchors: gen-ceiling 3*152={3*152} dps (4.0x linear) | "
          f"serve GLOBAL MAX {references.REF_GLOBAL_MAX_DPS:.0f} dps | high-N bench {references.REF_HIGH_N_BENCH_DPS:.0f} dps")

    # ---- the transport -> bound table -------------------------------------------------------
    print("\n" + "-" * 92)
    print("TRANSPORT -> BOUND   three honesty levels (all numpy; CI via the Neyman driver):")
    print("  HEADLINE   = the model's own f (4.0x-linear producer, the model's own transfer accounting)")
    print("  SERVE-FLOOR= HEADLINE + the H<->D transfer residual the fully_device T_disp omits, STILL")
    print("               4.0x producer (isolates the missing-transfer fix the valid=false critiques demanded)")
    print("  CONSERV    = SERVE-FLOOR + the 1.9x Python-ExIt producer worst case as a hard min (strict floor)")
    print("-" * 92)
    hdr = (f"  {'transport':<16}{'HEADLINE':>9}{'bd':>4}  {'SERVE-FLR':>9}{'bd':>4}  {'CONSERV':>8}{'bd':>4}  "
           f"{'CI+/-':>6}  {'xPlat(hl)':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sorted(results, key=lambda r: r.headline_dps, reverse=True):
        b_hl = r.headline_binding[:3]
        b_sf = r.serve_floor_binding[:3]
        b_cons = r.conservative_binding[:3]
        print(f"  {r.slug:<16}{r.headline_dps:>9.1f}{b_hl:>4}  {r.serve_floor_dps:>9.1f}{b_sf:>4}  "
              f"{r.conservative_dps:>8.1f}{b_cons:>4}  {r.ci_halfwidth:>6.1f}  {r.headline_dps/plat:>8.2f}x")

    # ---- the optimum-over-transports --------------------------------------------------------
    opt_hl = max(results, key=lambda r: r.headline_dps)
    opt_sf = max(results, key=lambda r: r.serve_floor_dps)
    opt_cons = max(results, key=lambda r: r.conservative_dps)
    print("\n" + "-" * 92)
    print("OPTIMUM OVER TRANSPORTS")
    print("-" * 92)
    print(f"  by HEADLINE     : {opt_hl.slug}  =  {opt_hl.headline_dps:.1f} dps "
          f"(~{opt_hl.headline_dps/plat:.2f}x the ~{plat:.0f} plateau)   binding = {opt_hl.headline_binding}")
    print(f"  by SERVE-FLOOR  : {opt_sf.slug}  =  {opt_sf.serve_floor_dps:.1f} dps "
          f"(~{opt_sf.serve_floor_dps/plat:.2f}x the ~{plat:.0f} plateau)   binding = {opt_sf.serve_floor_binding}")
    print(f"  by CONSERVATIVE : {opt_cons.slug}  =  {opt_cons.conservative_dps:.1f} dps "
          f"(~{opt_cons.conservative_dps/plat:.2f}x the ~{plat:.0f} plateau)   binding = {opt_cons.conservative_binding}")
    if opt_cons.conservative_binding == "GENERATION":
        print(f"  NOTE: under the 1.9x worst case ALL transports collapse to the same producer-capped "
              f"floor (~{opt_cons.conservative_dps:.0f} dps) — the transport stops mattering because")
        print(f"        GENERATION binds below every serve cycle. So the conservative optimum is a "
              f"PRODUCER question (does the C++ gen path hold 4.0x?), not a transport one.")
    hl_spread = max(r.headline_dps for r in results) - min(r.headline_dps for r in results)
    serve_variants = [r for r in results if r.headline_binding == "SERVE"]
    if serve_variants:
        serve_spread = (max(r.headline_dps for r in serve_variants)
                        - min(r.headline_dps for r in serve_variants))
        print(f"\n  THE FINDING (neutral): across the 4 staged-slope SERVE-bound variants the headline spans "
              f"only ~{serve_spread:.0f} dps")
        print(f"  (zmq -> shm/futex/mpsc), because at a FULL bucket the cycle is COMPUTE-dominated "
              f"(B*t_row dominates,")
        print(f"  tau_io is a small residual) — so among wire-fed transports the wakeup/queue MECHANISM "
              f"barely moves the")
        print(f"  bound; its value is OPERATIONAL (no burnt core / no broker), not throughput. The one "
              f"real jump is")
        print(f"  cpp_inproc_port (+{opt_hl.headline_dps - max(r.headline_dps for r in serve_variants):.0f} dps "
              f"over the best wire-fed), because it ALSO moves t_row (the bare fully_device")
        print(f"  slope) and removes the wire entirely — pushing the bottleneck OFF serve onto the producers "
              f"(GENERATION-bound).")
        print(f"  Full headline span across ALL transports: ~{hl_spread:.0f} dps.")

    # ---- which term each transport moves ----------------------------------------------------
    print("\n" + "-" * 92)
    print("WHICH TERM EACH TRANSPORT MOVES (the design variable), + its per-variant transfer policy")
    print("-" * 92)
    for r in results:
        cyc = r.cycle_breakdown
        print(f"  {r.slug:<16} moves: {r.moves}")
        print(f"  {'':<16} cycle(us): T_disp={cyc['T_disp_us']:.1f} + tau_io={cyc.get('tau_io_us', 0):.1f} "
              f"+ wakeup={cyc.get('wakeup_us', 0):.1f} + compute={cyc['compute_us']:.1f} "
              f"= {cyc['cycle_us']:.1f}")
        print(f"  {'':<16} transfer policy (conservative arm): +{r.transfer_residual_us:.2f}us  "
              f"[{r.transfer_note}]")

    # ---- the per-transport top measurement targets ------------------------------------------
    print("\n" + "-" * 92)
    print("PER-TRANSPORT TOP NEYMAN TARGETS  (what to run sole-workload next, pinned taskset -c 0)")
    print("-" * 92)
    print("  Two complementary orderings per variant (BOTH honest):")
    print("   * VARIANCE  — the allocator's data-driven a_i ranking (what most tightens the bound's CI);")
    print("                 funds the BINDING stage (compute t_row/L/B when serve-bound; producer when gen-bound).")
    print("   * TRANSPORT — the design-priority moved-term ranking (the levers THIS transport moves); these")
    print("                 carry small a_i at a full bucket, so the variance ranking buries them — but they")
    print("                 are 'what to run to PIN THE TRANSPORT'. [seed] = trusted=False (NEEDS-SOLE-WORKLOAD).")
    for r in results:
        print(f"\n  [{r.slug}]  (binding = {r.headline_binding})")
        vt = ", ".join(f"{nm}(a={a:.3g},+{s})" for nm, a, s in r.variance_targets[:4])
        print(f"     VARIANCE : {vt}")
        tt = ", ".join(f"{q.split(r.slug + '_')[-1]}={m:.3g}{'' if t else ' [seed]'}"
                       for q, m, t in r.transport_targets)
        print(f"     TRANSPORT: {tt}")

    # ---- the grounded-vs-unmeasured split (ADR-0002 honesty) --------------------------------
    print("\n" + "-" * 92)
    print("GROUNDED-vs-UNMEASURED SPLIT  (claims-measured-vs-interpreted — ADR-0002)")
    print("-" * 92)
    total_inputs = sum(len(importlib.import_module(v.module).INPUT_NAMES) for v in VARIANTS)
    total_untrusted = sum(len(r.untrusted_inputs) for r in results)
    print(f"  trusted (live postgres measurement) inputs across all variants: "
          f"{total_inputs - total_untrusted} / {total_inputs}")
    if total_untrusted == total_inputs:
        print(f"  EVERY input is a SEED (trusted=False). So EVERY bound above is a first-principles SEED")
        print(f"  ESTIMATE, NOT a measured floor — the honest current state. The benches are registered +")
        print(f"  written + runnable; an operator runs each pinned (taskset -c 0) sole-workload and the")
        print(f"  manifest flips trusted=True automatically (no model edit). Until then, what the SAMPLES")
        print(f"  support = nothing yet; what is INTERPRETED (seed/first-principles) = the entire table.")
    else:
        for r in results:
            tag = "ALL TRUSTED" if r.all_trusted else f"untrusted: {r.untrusted_inputs}"
            print(f"  {r.slug:<16} {tag}")
    print(f"\n  WHAT IS GROUNDED IN A FIT (not a fresh sole-workload bench, but a real measured JSON):")
    print(f"    T_disp=68.84us (mlp_lowlatency dispatch floor), t_row=4.317us/row (run_microbatch staged")
    print(f"    slope), the bare t_row=3.092us/row (fully_device slope) — these are REAL fit read-offs.")
    print(f"  WHAT IS A SEED/PIN (first-principles or design pin, flagged needs-measurement in the v1 grounding):")
    print(f"    every tau_io/wakeup/gather/req_drain/tmsg (the transport residuals), B_op=256 (full-bucket"
          f" pin),")
    print(f"    LPD=500 (design pin, not a per-decision histogram), R_gen=152 (the 4.0x-linear gen rate).")

    # ---- the caveats that make these LOWER bounds (and where they are still seeds) -----------
    print("\n" + "-" * 92)
    print("CAVEATS (what the bounds are contingent on — neither overclaimed nor hidden)")
    print("-" * 92)
    print("  * BENCH-dps bounds: the input rates read HIGHER than the closed-loop e2e (adapter.md §7), so")
    print("    the production-e2e optimum is correspondingly lower than these per-stage flat-out floors.")
    print("  * Contingent on the optimum reaching a FULL-bucket feed (the high-N regime) and on the single-")
    print("    threaded serialization being no worse than the tau_io+wakeup charged.")
    print("  * The HEADLINE uses the 4.0x-linear C++ gen producer ceiling (3*152=456); the CONSERVATIVE")
    print("    column takes the 1.9x Python-ExIt worst case (288.8) as a hard min — so where a headline is")
    print("    SERVE-bound the conservative floor is producer-capped at ~289 (the 'min taking the WORSE")
    print("    case' every model docstring claims but computes only in the conservative column here).")
    print("  * The CONSERVATIVE column also adds the H<->D transfer residual the fully_device T_disp omits")
    print("    (per-variant: full staged residual for the wire-fed variants; D2H output-pull only for the")
    print("    inproc port, which already charges the H2D input crossing in its tau_io).")


def main() -> None:
    results = run_sweep()
    print_report(results)


if __name__ == "__main__":
    main()
