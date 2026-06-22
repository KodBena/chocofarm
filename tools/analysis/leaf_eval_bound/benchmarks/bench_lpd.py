"""
tools/analysis/leaf_eval_bound/benchmarks/bench_lpd.py
================================================

LIVE benchmark for `LPD` — leaves per recorded decision (leaves/decision): the unit-conversion
divisor from leaves/s to decisions/s (dps). Baseline, transport-invariant (a transport moves
I/O cost, not how many leaves a search expands per decision).

WHAT run()/measure() MEASURES (1:1 with the model input). LPD is a per-decision distinct-leaf
count — its faithful measurement is the PER-DECISION leaf-count distribution from an instrumented
search run (the count of net forwards a sims256/m24 Gumbel tree issues per recorded decision).
The tool is the SAME C++ gen-ceiling sole-workload bench that grounds R_gen / g_core —
`cpp/build/chocofarm-search-runtime-bench` (source cpp/src/search_runtime_bench.cpp) — which runs a
batch of independent Gumbel-AZ decisions through `SerialRuntime` over a LOCAL in-process `DetNet`
(a deterministic, stateless, weightless, RNG-free leaf evaluator — the eval mock by construction)
at the gen-ceiling config (`sims256/m24`, max_depth 24: the config whose per-decision distinct-node
count IS the LPD≈500 grounding). EACH task is ONE independent decision, so the bench's per-task
`leaf_requests` (printed `leaf_requests_per_task=<n0> <n1> ...`) IS that decision's leaf count — a
POOL of per-decision LPD readings, one physical run grounding LPD just as the SAME run grounds R_gen
and g_core (ADR-0012 P1 single-home: one instrumented run, all three rates; `LPD =
leaf_requests_total/n_tasks` is the aggregate cross-read the sibling benches already log).

WHY SHRINKABLE NOW (the ADR-0008 reclassification this module IS). LPD is a MEASURED quantity, not a
config pin (grounding `LEAVES_PER_DECISION.needs_measurement=True`, `constant` defaults False — a
measured quantity, the §3 PIN-now/measurable-later class, NOT a true constant). The prior version of
this module PUNTED — `_measure_raw()` returned the seed (500) and `_estimate_from_raw()` wrapped it
in `pin_estimate(...)` → an un-shrinkable `Fixed` Estimate — so the Neyman loop could not sample it
(a `Fixed` law's `marginal_dvar_deffort` is 0 → `A_i = 0` → never funded → the generation arm of
`untrusted_drive` STALLS — the SAME punt @d5f84b7 removed for R_gen). The bench's own v1 docstring
conceded the measurement is "a per-decision leaf-count HISTOGRAM from an instrumented search run …
a C++/search-harness artifact" — naming the runnable harness while still recording the pin. The
binary EXISTS and is BUILT, so the measurement is real: this module RUNS it and returns a SHRINKABLE
`QuantileLaw` (median) Estimate over the per-decision leaf-count pool (a real bootstrap median SE —
docs/design/harmonized-estimator-interface.md §7.A, §3 MEDIAN row and the PIN-now/measurable-later
row: `LPD` as a leaf-count histogram flips `Fixed` -> `QuantileLaw` once instrumented). The model
consumes LPD as a pure scalar divisor (model_capacity.py:99-102 `n_gen*g_core/LPD`, `serve/LPD`,
`1/(LPD*tmsg)`), so a shrinkable median over the per-decision counts is sufficient to make LPD
fundable — the "histogram" framing is a strictly-higher bar than the scalar contract needs. More
decisions sampled (a bigger `trials` budget) -> a bigger pool -> a tighter median SE -> a tighter
generation-arm CI, so the loop FUNDS LPD instead of stalling.

CLASSIFICATION (ADR-0008): LPD is a MEDIAN (a measured quantity whose variance responds to effort),
NOT a `pin_declared_spread` and NOT a true constant. It is deliberately NOT marked `constant=True`:
that would make it DEGENERATE (`a_i≈0`, un-fundable) — re-introducing the stall — and is the wrong
vocabulary (LPD is not a layout/pinning fact like n_gen; it is a measured per-decision count that
varies tree-to-tree, verified non-degenerate `[504,500,503,500,505,502,508,506]`). The honest
classification refuses the fuzzy match (revise to the fitting class, do not pick the closest).

`get_seed()` stays the DISTRUST fallback (the v1 500 leaves/decision DESIGN PIN, a `Fixed`
declared-spread prior on the SEED path — the manifest's `trust=False` / pg-down route via
`_estimate_from_seed`; only the MEASURED path is shrinkable). FAIL LOUD (ADR-0002): if the C++ bench
is absent/unbuilt at run time, or the run fails, or it does not report `RESULT: PASS`, or it yields
no per-decision readings, `_measure_raw()` RAISES — it NEVER silently falls back to the
seed-as-if-measured (that silent fallback is exactly the punt this module removes).

TIMING-NEUTRAL but a SOLE-WORKLOAD subprocess. LPD is a STRUCTURAL leaf count, deterministic per
seeded task (the bench asserts the serial and pool runtimes produce bit-identical per-task leaf
counts), so it is not timing-sensitive — but it still spawns the gen-ceiling subprocess, so pin it
(`taskset -c 0`, `_TASKSET`) and do NOT run it during the parallel fan-out (one binary run per
explicit operator `run()` / driver `measure()`, never amid the fan-out).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
import leaf_eval_grounding as G  # noqa: E402
from bench_common import logged_run, median_estimate  # noqa: E402

NAME = "LPD"
MODULE_PATH = "benchmarks.bench_lpd"
_DESC = ("Leaves per recorded decision (leaves/decision): the leaves/s -> dps divisor. LIVE = the C++ "
         "gen-ceiling sole-workload bench (search_runtime_bench, eval mocked, SerialRuntime; the per-task "
         "leaf_requests pool at sims256/m24 — each task is one decision). v1 seed 500 (a sims256/m24 "
         "Gumbel tree's distinct-node count). Baseline, transport-invariant. Same instrumented run as "
         "R_gen/g_core (one run grounds all three — ADR-0012 P1).")

# --- The C++ gen-ceiling bench: the binary, the gen-ceiling Gumbel config, the eval mock --------
# The SAME binary/config/geometry as bench_r_gen.py / bench_g_core.py (ADR-0012 P1: one instrumented
# run grounds R_gen, g_core, AND LPD — LPD is the per-decision leaf-count pool the SAME run prints).
# This file is tools/analysis/leaf_eval_bound/benchmarks/bench_lpd.py -> up 4 (benchmarks -> OpenTURNS ->
# analysis -> tools -> <repo root>).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
# The built binary + the instance/faces geometry (same convention + env names as bench_r_gen /
# bench_g_core / exit_loop.py: `cpp/build/chocofarm-*` + `chocofarm/data/{instance,faces}.json`).
# Env-overridable so an operator can point at an alternate build/geometry without editing the bench.
_BENCH_BIN = os.environ.get(
    "CHOCO_SEARCH_RUNTIME_BENCH",
    os.path.join(_REPO_ROOT, "cpp", "build", "chocofarm-search-runtime-bench"))
_INSTANCE = os.environ.get("CHOCO_BENCH_INSTANCE", os.path.join(_REPO_ROOT, "chocofarm", "data", "instance.json"))
_FACES = os.environ.get("CHOCO_BENCH_FACES", os.path.join(_REPO_ROOT, "chocofarm", "data", "faces.json"))
# The bench ALSO reads the feature-layout spec by a RELATIVE path (chocofarm/data/feature_layout.json —
# "run from the repo root"); --instance/--faces are passed absolute, but this one is looked up internally,
# so the subprocess is run with cwd=_REPO_ROOT AND CHOCO_FEATURE_LAYOUT set (belt + suspenders) so it
# resolves from ANY caller CWD. Without this the bench SIGABRTs (FATAL, exit 134) from e.g.
# tools/analysis/leaf_eval_bound/benchmarks/ — verified.
_FEATURE_LAYOUT = os.environ.get("CHOCO_FEATURE_LAYOUT", os.path.join(_REPO_ROOT, "chocofarm", "data", "feature_layout.json"))
# The gen-ceiling Gumbel config: sims256/m24, max_depth 24 — the tree whose per-decision distinct-node
# count IS the LPD≈500 grounding (verified: per-task leaf counts [504,500,503,500,505,502,508,506],
# leaf_requests_total/n_tasks ≈ 503.5). These are the config the seed's "500 leaves/decision" provenance
# is read at, and the IDENTICAL config bench_r_gen/bench_g_core time (one run grounds all three).
_GEN_N_SIMS = 256
_GEN_M = 24
_GEN_MAX_DEPTH = 24
_DEFAULT_TASKS = 32    # the per-decision pool size at the default measure(): 32 independent decisions,
                       # the SAME --tasks the sibling rate benches use (so the default run is literally
                       # the run that grounds R_gen/g_core). The `trials` budget sizes this up.
_REPS = 1             # LPD is STRUCTURAL (deterministic per seeded task), not timed — one rep suffices
                      # (the per-task leaf-count line is printed once, before the timing loop).
_TASKSET = ("taskset", "-c", "0")   # sole-workload single-core pin (TIMING-NEUTRAL but sole-workload)

# Parse `RESULT: PASS` (the run is usable), the per-decision pool `leaf_requests_per_task=<n0> <n1> ...`
# (one int per task — each task is ONE decision, so this IS the per-decision LPD pool), and the
# aggregate `leaf_requests_total=<n>` (the same cross-read R_gen/g_core log: LPD = total/n_tasks).
_RE_RESULT = re.compile(r"RESULT:\s*PASS\b")
_RE_PER_TASK = re.compile(r"^leaf_requests_per_task=([0-9 ]+?)\s*$", re.MULTILINE)
_RE_LEAF_TOTAL = re.compile(r"\bleaf_requests_total=(\d+)\b")


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): LPD=500 leaves/decision (DESIGN PIN — a sims256/m24 Gumbel tree's
    distinct-node count, provenance in the seed). Used by the SEED path (manifest `trust=False` / pg-down)
    as a `Fixed` declared-spread prior; the MEASURED path (measure()/run() running the C++ bench) is the
    shrinkable `QuantileLaw`. Per ADR-0008 the seed is a declared-spread prior on the seed path; only the
    measured per-decision pool flips it to a median."""
    return G.LEAVES_PER_DECISION


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="leaves_per_decision", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def _run_cpp_bench(n_tasks: int) -> tuple[list[int], int]:
    """Run the C++ gen-ceiling bench ONCE over `n_tasks` independent decisions (sole-workload, taskset
    -c 0, eval mocked by the bench's built-in DetNet), parse its output, and return
    `(leaf_per_task, leaf_requests_total)`:

      * `leaf_per_task`       : the per-task `leaf_requests` (one int per decision) off the
                                `leaf_requests_per_task=` line — the per-decision LPD POOL (`n_tasks`
                                readings). `n_tasks` IS the shrink budget — more decisions tighten the
                                bootstrap median SE.
      * `leaf_requests_total` : the aggregate distinct-leaf count over the batch (the same cross-read the
                                rate benches log: LPD = leaf_requests_total/n_tasks ≈ 503.5).

    FAIL LOUD (ADR-0002): the binary missing/unbuilt, a non-zero exit, no PASS, or no per-decision
    readings all RAISE — NEVER a silent seed fallback (the punt this module removes). The seed is the
    DISTRUST fallback path (get_seed()), not a measured-result substitute. Mirrors bench_r_gen /
    bench_g_core `_run_cpp_bench`."""
    if not (os.path.isfile(_BENCH_BIN) and os.access(_BENCH_BIN, os.X_OK)):
        raise FileNotFoundError(
            f"bench_lpd: the C++ gen-ceiling bench is not built at {_BENCH_BIN!r} (the live LPD "
            f"re-measure — the per-decision leaf-count pool). Build it (cmake --build cpp/build --target "
            f"chocofarm-search-runtime-bench) or set CHOCO_SEARCH_RUNTIME_BENCH. ADR-0002: a missing "
            f"measurement bench RAISES — it is NEVER a silent fall-back to the 500 design pin as if measured.")
    for p, what, ev in ((_INSTANCE, "instance", "CHOCO_BENCH_INSTANCE"),
                        (_FACES, "faces", "CHOCO_BENCH_FACES"),
                        (_FEATURE_LAYOUT, "feature-layout", "CHOCO_FEATURE_LAYOUT")):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"bench_lpd: the C++ bench {what} geometry is missing at {p!r} (set {ev}). "
                f"ADR-0002: cannot measure LPD without the geometry.")
    taskset = list(_TASKSET) if shutil.which("taskset") else []   # pin if available; degrade loud-noted
    cmd = [
        *taskset, _BENCH_BIN,
        "--instance", _INSTANCE, "--faces", _FACES,
        "--tasks", str(int(n_tasks)), "--workers", "1", "--reps", str(_REPS),
        "--n-sims", str(_GEN_N_SIMS), "--m", str(_GEN_M), "--max-depth", str(_GEN_MAX_DEPTH),
    ]
    # cwd=_REPO_ROOT so the bench's RELATIVE feature-layout lookup resolves (it SIGABRTs otherwise from a
    # non-root CWD — verified); CHOCO_FEATURE_LAYOUT set explicitly too. --instance/--faces are absolute.
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          cwd=_REPO_ROOT, env={**os.environ, "CHOCO_FEATURE_LAYOUT": _FEATURE_LAYOUT})
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench_lpd: the C++ gen-ceiling bench exited {proc.returncode} (cmd: {' '.join(cmd)}).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n"
            f"(ADR-0002: a failed measurement RAISES — never a silent seed substitute.)")
    out = proc.stdout
    if _RE_RESULT.search(out) is None:
        raise RuntimeError(
            f"bench_lpd: the C++ bench did not report `RESULT: PASS` (a FAIL is a serial/parallel "
            f"mismatch — the per-task leaf counts disagree, so they are not a usable per-decision pool). "
            f"Output:\n{out}\nstderr:\n{proc.stderr}")
    pm = _RE_PER_TASK.search(out)
    if pm is None:
        raise RuntimeError(
            f"bench_lpd: the C++ bench reported no `leaf_requests_per_task=<n0> <n1> ...` line — the "
            f"per-decision leaf-count pool is missing (is the bench rebuilt with the per-task print?). "
            f"Output:\n{out}  (ADR-0002: no per-decision readings is a loud fault, not a fabricated pool.)")
    leaf_per_task = [int(tok) for tok in pm.group(1).split()]
    if len(leaf_per_task) < 2:
        raise RuntimeError(
            f"bench_lpd: the C++ bench reported only {len(leaf_per_task)} per-decision reading(s) "
            f"(n_tasks={n_tasks}); need >= 2 for a bootstrap median SE. Output:\n{out}  (ADR-0002.)")
    lm = _RE_LEAF_TOTAL.search(out)
    leaf_total = int(lm.group(1)) if lm else sum(leaf_per_task)
    return leaf_per_task, leaf_total


def _measure_raw(trials: int = _DEFAULT_TASKS) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): RUN the C++ gen-ceiling bench
    (eval mocked, sole-workload, taskset -c 0) over `trials` independent decisions, and return the
    per-decision leaf-count pool plus provenance. EACH task is ONE decision, so its `leaf_requests` is
    that decision's leaf count (an LPD reading); the pool is the per-task counts. `trials` IS the shrink
    budget — more decisions -> a bigger pool -> a tighter median SE (the Neyman loop sizes it via the
    `trials` kwarg). Returns {'lpd' (the pool MEDIAN), 'per_decision_leaves' (the pool the Estimate is
    built over), 'leaf_requests_total', 'lpd_mean_cross_read' (leaf_requests_total/n_tasks, the same-bench
    aggregate cross-read R_gen/g_core log), 'n_tasks', config}. `measure()`/`run()` both consume this ONE
    measurement (P1). FAIL LOUD via `_run_cpp_bench` if the binary is absent/unbuilt or the run fails —
    never the seed-as-if-measured."""
    import numpy as np
    n = max(2, int(trials))   # >= 2 decisions so the bootstrap median SE is defined (a 1-reading pool has none)
    leaf_per_task, leaf_total = _run_cpp_bench(n)
    per_decision = [float(v) for v in leaf_per_task]
    return {
        "lpd": float(np.median(per_decision)),
        "per_decision_leaves": per_decision,
        "leaf_requests_total": leaf_total,
        "lpd_mean_cross_read": leaf_total / float(len(leaf_per_task)),
        "n_tasks": len(leaf_per_task),
        "config": f"search_runtime_bench leaf_requests_per_task; sims{_GEN_N_SIMS}/m{_GEN_M} "
                  f"depth{_GEN_MAX_DEPTH}; eval mocked (DetNet); taskset -c 0",
    }


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build LPD's harmonized SHRINKABLE `Estimate` — the SINGLE home of the Estimate construction (P1),
    called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)` with a BOOTSTRAP median SE
    over the per-decision leaf-count pool (§7.A — a real order-statistic SE, NOT a `Fixed` pin),
    `family=EMPIRICAL`, `kind='median'`, POSITIVE support. This is the ADR-0008 reclassification: LPD is
    a MEASURED quantity whose variance RESPONDS to effort (the median's `marginal_dvar_deffort` is
    `−cov/n < 0`), so the Neyman loop can FUND it (more decisions -> tighter SE), where the prior `Fixed`
    pin (marginal=0) made it un-fundable and stalled the generation arm. ADR-0012 P8: the Estimate's
    family/shrink IS the contract (typed-signature-is-SSOT) — flipping Fixed->QuantileLaw makes the
    `needs_measurement=True` signature honest (the body now honors it)."""
    return median_estimate(res["per_decision_leaves"], name=NAME)   # bootstrap median SE over the per-decision pool


def measure(trials: int = _DEFAULT_TASKS) -> "_est.Estimate":
    """Measure LPD (RUN the C++ gen-ceiling bench, eval mocked, sole-workload) and return its harmonized
    k=1 SHRINKABLE median `Estimate` (§6 Phase 4: `measure()` returns the `Estimate` the bench DECLARES —
    the driver/untrusted_drive `set_estimate`s it directly). `trials` sizes the per-decision measurement
    pool (the budget the Neyman loop passes — more decisions tightens LPD's SE -> the generation-arm CI).
    SOLE-WORKLOAD — pinned (taskset -c 0); never during the fan-out."""
    return _estimate_from_raw(_measure_raw(trials=trials))


def run(trials: int = _DEFAULT_TASKS) -> dict[str, Any]:
    """Measure LPD (RUN the C++ gen-ceiling bench) and LOG it to postgres as a harmonized k=1 SHRINKABLE
    median `Estimate` (§6 Phase 3): `QuantileLaw(p=0.5)` with a BOOTSTRAP median SE over the per-decision
    leaf-count pool (§7.A), `family=EMPIRICAL`, `kind='median'`. The per-decision readings are logged as
    raw PROVENANCE — the variance authority is `estimate.cov`, so the headline LPD is NOT double-logged as
    a sample row (§5.2 de-dup). Returns the raw provenance dict. SOLE-WORKLOAD — operator-invoked, pinned
    (taskset -c 0), never during the fan-out."""
    res = _measure_raw(trials=trials)        # ONE measurement (Est + provenance pool)
    est = _estimate_from_raw(res)            # the SAME Estimate measure() returns (P1)
    cfg = {"kind": "cpp_gen_ceiling_measured", "config": res["config"], "n_tasks": res["n_tasks"],
           "leaf_requests_total": res["leaf_requests_total"], "lpd_mean_cross_read": res["lpd_mean_cross_read"],
           "lpd_median": res["lpd"]}
    with logged_run(NAME, quantity="leaves_per_decision", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2): the raw per-decision readings. The headline LPD lives in
        # estimate.theta_hat[0] (the SSOT), the median SE in estimate.cov.
        log(res["per_decision_leaves"], sample_size=1)
    return res


if __name__ == "__main__":
    print(f"[bench_lpd] seed: {get_seed().mean} {get_seed().unit} (DESIGN PIN — {get_seed().provenance})")
    print(f"[bench_lpd] live bench: {_BENCH_BIN}")
    register_self()
    print("[bench_lpd] registered. measure()/run() RUN the C++ gen-ceiling bench (eval mocked, "
          "taskset -c 0) -> a SHRINKABLE median Estimate over the per-decision leaf-count pool. "
          "get_seed() is the DISTRUST fallback.")
