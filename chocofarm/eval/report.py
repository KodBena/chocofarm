#!/usr/bin/env python3
"""
chocofarm/eval/report.py — the ONE BeliefRefs-based reporting layer for the classical
(non-AZ) solver evals.

Before this module each eval_*.py hand-typed the same three things: the floor/ceiling
reference pair (`realizable_static(env)` / `clairvoyant_rate(env)`), the %VoI formula
`(r - static) / (ceil - static) * 100`, and the floor/ceiling banner — a copy-paste SSOT
hazard the audit (R10) flags. `harness.BeliefRefs` already single-sources the metric for
exit_loop / eval_az (R3); this module finishes the propagation by giving the remaining
scripts ONE entry point each:

  - `references(env)`            -> a memoized `BeliefRefs` (the floor/ceiling/%VoI SSOT).
  - `print_reference_header(...)` -> the common floor/ceiling banner, byte-for-byte as the
                                     scripts printed it.
  - `run_plan(...)`             -> the shared Dinkelbach table runner for the common-format
                                     scripts (eval_uct / eval_ismcts / eval_nmcs / eval_faces).

This is a DEDUP, not a redesign: every banner line and every table column/width here
reproduces what the scripts already printed, and the metric flows through `BeliefRefs`
so it cannot drift. The bespoke evals (eval_decomp / eval_az) keep their own formatting and
only repoint the metric through `references()`.

DEFERRED (out of R10 scope, noted): a per-solver `SearchConfig` dataclass. The solvers'
frozen `__init__` is only truly made configurable by a per-call cfg object — a larger change
than this dedup, so the PLAN literals stay inline in each script for now.

Fail-loud (ADR-0002): unknown `columns` raises rather than silently printing a wrong table.

Public Domain (The Unlicense).
"""
import time

from chocofarm.eval.harness import BeliefRefs


def references(env):
    """The single entry the scripts call: the floor/ceiling/%VoI SSOT for this env."""
    return BeliefRefs(env)


def print_reference_header(refs, *, extra_lines=(), faces=False):
    """Print the common floor/ceiling banner, then a blank line.

    Reproduces the scripts' CURRENT banner byte-for-byte. The default (uct / ismcts / nmcs)
    is:

        static floor        : {static:.4f}
        clairvoyant ceiling : {ceil:.4f}   (VoI headroom +{headroom:.0f}%)
        <blank>

    `extra_lines` is a tuple of extra strings printed (un-prefixed) before the blank line —
    e.g. eval_decomp's "clairvoyant per-excursion: …" line.

    `faces=True` selects eval_faces's variant, which inlines the detector-independent hints
    ("expect ~0.0855" on the floor and "expect ~0.1454, VoI headroom +N%" on the ceiling) so
    that script's output is unchanged.
    """
    static = refs.static_floor
    ceil = refs.clairvoyant_ceiling
    headroom = (ceil - static) / static * 100
    if faces:
        print(f"static floor        : {static:.4f}   (detector-independent; expect ~0.0855)")
        print(f"clairvoyant ceiling : {ceil:.4f}   (detector-independent; expect ~0.1454, "
              f"VoI headroom +{headroom:.0f}%)\n", flush=True)
        return
    print(f"static floor        : {static:.4f}")
    if extra_lines:
        print(f"clairvoyant ceiling : {ceil:.4f}   (VoI headroom +{headroom:.0f}%)")
        for line in extra_lines[:-1]:
            print(line)
        print(extra_lines[-1] + "\n", flush=True)
    else:
        print(f"clairvoyant ceiling : {ceil:.4f}   (VoI headroom +{headroom:.0f}%)\n", flush=True)


# Column specs for the common-format Dinkelbach table. Each entry is
# (name_width, header_string, row_renderer). The row_renderer takes the per-row values and
# returns the formatted line, so each script's CURRENT column set + alignment widths are
# reproduced exactly. Add a new spec here rather than re-inlining a loop in a script.
def _hdr_base(name_w, extra):
    return (f"{'policy':>{name_w}} {'rate':>8} {'%ceiling':>9} {'VoI clawed':>11}"
            + extra)


_COLUMN_SPECS = {
    # eval_ismcts.py: policy(16) / rate / %ceiling / VoI clawed / sec(6)
    "ismcts": dict(
        name_w=16,
        header=_hdr_base(16, f" {'sec':>6}"),
        row=lambda name, res, claw, ceil, sec, budget:
            f"{name:>16} {res['rate']:>8.4f} {res['rate'] / ceil * 100:>8.0f}% "
            f"{claw:>10.0f}% {sec:>6.0f}",
    ),
    # eval_nmcs.py: policy(16) / rate / %ceiling / VoI clawed / runs(6) / sec(7)
    "nmcs": dict(
        name_w=16,
        header=_hdr_base(16, f" {'runs':>6} {'sec':>7}"),
        row=lambda name, res, claw, ceil, sec, budget:
            f"{name:>16} {res['rate']:>8.4f} {res['rate'] / ceil * 100:>8.0f}% "
            f"{claw:>10.0f}% {budget['final_runs']:>6} {sec:>7.0f}",
    ),
    # eval_faces.py: policy(20) / rate / %ceiling / VoI clawed / runs(6) / sec(7)
    "faces": dict(
        name_w=20,
        header=_hdr_base(20, f" {'runs':>6} {'sec':>7}"),
        row=lambda name, res, claw, ceil, sec, budget:
            f"{name:>20} {res['rate']:>8.4f} {res['rate'] / ceil * 100:>8.0f}% "
            f"{claw:>10.0f}% {budget['final_runs']:>6} {sec:>7.0f}",
    ),
    # eval_uct.py: policy(14) / rate / %ceiling / VoI clawed / E[R](6) / E[T](7) / runs(5)
    #              / sec/ep(7) / sec(6)
    "uct": dict(
        name_w=14,
        header=_hdr_base(14, f" {'E[R]':>6} {'E[T]':>7} {'runs':>5} {'sec/ep':>7} {'sec':>6}"),
        row=lambda name, res, claw, ceil, sec, budget:
            f"{name:>14} {res['rate']:>8.4f} {res['rate'] / ceil * 100:>8.0f}% "
            f"{claw:>10.0f}% {res['ER']:>6.2f} {res['ET']:>7.1f} "
            f"{budget['final_runs']:>5} {sec / budget['final_runs']:>7.1f} {sec:>6.0f}",
    ),
}


def run_plan(env, refs, plan, *, seed=7, columns="ismcts"):
    """The shared Dinkelbach table runner for the common-format scripts.

    `plan` is a list of `(name, policy, budget_dict)`. Prints the header row + one row per
    plan entry: runs `env.dinkelbach_rate(pol, seed=seed, **budget)`, computes the %VoI
    clawed via `refs.voi_pct(rate)`, and renders the row per the `columns` spec (one of
    "uct" / "ismcts" / "nmcs" / "faces"). Each script's CURRENT column set + alignment widths
    are reproduced by its named spec; new variants are added to `_COLUMN_SPECS`, not inlined.
    """
    spec = _COLUMN_SPECS.get(columns)
    if spec is None:                          # fail-loud (ADR-0002): no silent wrong table
        raise ValueError(
            f"run_plan: unknown columns={columns!r}; "
            f"known: {sorted(_COLUMN_SPECS)}")
    ceil = refs.clairvoyant_ceiling
    print(spec["header"], flush=True)
    for name, pol, budget in plan:
        t0 = time.time()
        res = env.dinkelbach_rate(pol, seed=seed, **budget)
        sec = time.time() - t0
        claw = refs.voi_pct(res["rate"])
        print(spec["row"](name, res, claw, ceil, sec, budget), flush=True)
