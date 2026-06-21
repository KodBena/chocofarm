"""
tools/analysis/OpenTURNS/benchmarks/bench_g_core.py
===================================================

LIVE benchmark for `g_core` — the per-core generation LEAF rate (leaves/s/core): the producer
ceiling input in LEAF units (the capacity model `model_capacity.py` uses leaves/s — `gen =
n_gen*g_core/LPD`; the cycle model uses dps/core). Baseline, transport-invariant (a generator
core's leaf rate is independent of which transport carries its leaf-eval requests; the transport
binds the SERVE stage, not generation). The aggregate producer cap is `N_gen * g_core / LPD`.

WHAT run()/measure() MEASURES (1:1 with the model input). g_core is the SAME physical generation
measurement as `R_gen`, expressed in leaves/s/core rather than decisions/s/core:
`g_core = R_gen * LPD` where `LPD = leaf_requests_total / n_tasks`. Equivalently, off the SAME
sole-workload run, `g_core = leaf_requests_total / (n_tasks * serial_time)` (leaf requests over the
single-core serial wall-time of the batch). The tool is the SAME C++ gen-ceiling sole-workload bench
that grounds R_gen — `cpp/build/chocofarm-search-runtime-bench` (source cpp/src/search_runtime_bench.cpp)
— which runs a batch of independent Gumbel-AZ decisions through `SerialRuntime` over a LOCAL in-process
`DetNet` (a deterministic, stateless, weightless, RNG-free leaf evaluator — the eval mock by
construction) at the gen-ceiling config (`sims256/m24`, max_depth 24: the config whose aggregate
distinct-node count IS the LPD≈500 grounding), and reports the per-rep `serial=<t>s` wall-times plus
the aggregate `leaf_requests_total`. From those two this module forms a per-rep leaves/s/core pool —
the leaf-unit twin of R_gen's per-rep dps/core pool, ONE physical run feeding both.

WHY SHRINKABLE NOW (the ADR-0008 reclassification this module IS). g_core is a MEASURED quantity, not
a config pin (grounding `GEN_PER_CORE_LEAVES.needs_measurement=True`, `constant` defaults False — a
measured quantity, the §3 PIN-now/measurable-later class, not a true constant). The prior version of
this module PUNTED — `_measure_raw()` returned the seed (76000) and `_estimate_from_raw()` wrapped it
in `pin_estimate(...)` → an un-shrinkable `Fixed` Estimate — so the Neyman loop could not sample it (a
`Fixed` law's `marginal_dvar_deffort` is 0 → `A_i = 0` → never funded → the generation arm of
`untrusted_drive` STALLS), and the dict labelled itself `is_cpp_bench=True` while running NO bench
(the lying signature — ADR-0012 P8/P1; ADR-0002 no-lying-signature). The binary EXISTS and is BUILT,
so the measurement is real: this module RUNS it and returns a SHRINKABLE `QuantileLaw` (median)
Estimate over a pool of per-rep leaves/s/core readings (a real bootstrap median SE —
docs/design/harmonized-estimator-interface.md §7.A, §3 MEDIAN row and the PIN-now/measurable-later
row: `R_gen`/`g_core` as a C++ rate flips `Fixed` -> `Poolwise`/`QuantileLaw` once instrumented; the
§0 status names `g_core` as the SAME measured-but-punted class as R_gen, queued behind it). More/longer
runs (a bigger `reps` budget) -> a tighter pool -> a tighter g_core SE -> a tighter generation-arm CI,
so the loop FUNDS g_core instead of stalling.

This mirrors the `R_gen` @d5f84b7 fix exactly — g_core is its leaf-unit twin, from the IDENTICAL
binary/config/run. (Per ADR-0012 P1 the single `_run_cpp_bench(...)` parse could be factored into a
shared helper both benches derive from; a standalone mirror is sufficient and minimal-touch — ADR-0004
— and keeps the already-landed `bench_r_gen.py` untouched. The cost is one extra binary run per
quantity, paid only by an explicit operator `run()`, never during the fan-out.)

`get_seed()` stays the DISTRUST fallback (the v1 76000 leaves/s/core seed, a `Fixed` declared-spread
prior on the SEED path — the manifest's `trust=False` / pg-down route; only the MEASURED path is
shrinkable). FAIL LOUD (ADR-0002): if the C++ bench is absent/unbuilt at run time, or the run fails,
or it does not report `RESULT: PASS`, or it yields no per-rep readings / no leaf count,
`_measure_raw()` RAISES — it NEVER silently falls back to the seed-as-if-measured (that silent
fallback is exactly the punt this module removes).

TIMING-SENSITIVE — DO NOT run() during the parallel fan-out; the bench is a sole-workload single-
core timing. Pinned with `taskset -c 0` (`_TASKSET`), single worker (`SerialRuntime`).

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

NAME = "g_core"
MODULE_PATH = "benchmarks.bench_g_core"
_DESC = ("Per-core generation leaf rate (leaves/s/core): the producer ceiling input in leaf units "
         "(= R_gen*LPD). LIVE = the C++ gen-ceiling sole-workload bench (search_runtime_bench, eval "
         "mocked, SerialRuntime; leaf_requests_total/(n_tasks*serial) at sims256/m24). v1 seed 76000 "
         "leaves/s/core. Same physical generation measurement as R_gen. Baseline, transport-invariant.")

# --- The C++ gen-ceiling bench: the binary, the gen-ceiling Gumbel config, the eval mock --------
# The SAME binary/config/geometry as bench_r_gen.py (g_core is its leaf-unit twin). This file is
# tools/analysis/OpenTURNS/benchmarks/bench_g_core.py -> up 4 (benchmarks -> OpenTURNS -> analysis ->
# tools -> <repo root>).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
# The built binary + the instance/faces geometry (same convention as bench_r_gen / exit_loop.py:
# `cpp/build/chocofarm-*` + `chocofarm/data/{instance,faces}.json`). Env-overridable so an operator can
# point at an alternate build/geometry without editing the bench (shared env names with bench_r_gen —
# one binary, one geometry).
_BENCH_BIN = os.environ.get(
    "CHOCO_SEARCH_RUNTIME_BENCH",
    os.path.join(_REPO_ROOT, "cpp", "build", "chocofarm-search-runtime-bench"))
_INSTANCE = os.environ.get("CHOCO_BENCH_INSTANCE", os.path.join(_REPO_ROOT, "chocofarm", "data", "instance.json"))
_FACES = os.environ.get("CHOCO_BENCH_FACES", os.path.join(_REPO_ROOT, "chocofarm", "data", "faces.json"))
# The bench ALSO reads the feature-layout spec by a RELATIVE path (chocofarm/data/feature_layout.json —
# "run from the repo root"); --instance/--faces are passed absolute, but this one is looked up internally,
# so the subprocess is run with cwd=_REPO_ROOT AND CHOCO_FEATURE_LAYOUT set (belt + suspenders) so it
# resolves from ANY caller CWD. Without this the bench SIGABRTs (FATAL, exit 134) from e.g.
# tools/analysis/OpenTURNS/benchmarks/ — verified.
_FEATURE_LAYOUT = os.environ.get("CHOCO_FEATURE_LAYOUT", os.path.join(_REPO_ROOT, "chocofarm", "data", "feature_layout.json"))
# The gen-ceiling Gumbel config: sims256/m24, max_depth 24 — the tree whose aggregate distinct-node
# count IS the LPD≈500 grounding (leaf_requests_total/n_tasks ≈ 504), so leaves/s/core ≈ 76k (verified:
# per-rep serial 0.209–0.212s over a 32-task batch, leaf_requests_total=16122 -> ~76k leaves/s/core).
# These are the config the seed's "76k leaves/s (152 dps/core * 500 LPD)" provenance is read at.
_GEN_N_SIMS = 256
_GEN_M = 24
_GEN_MAX_DEPTH = 24
_N_TASKS = 32          # the batch of independent root tasks the serial runtime times
_TASKSET = ("taskset", "-c", "0")   # sole-workload single-core pin (TIMING-SENSITIVE; CLAUDE.md core 0)

# Parse `RESULT: PASS` (the run is usable), `leaf_requests_total=<n>` (the aggregate distinct-leaf
# count), and `rep <r>: serial=<t>s parallel=<t>s` (the per-rep single-core wall-times for the pool).
_RE_RESULT = re.compile(r"RESULT:\s*PASS\b")
_RE_REP = re.compile(r"^rep\s+\d+:\s*serial=([0-9.eE+-]+)s\b", re.MULTILINE)
_RE_LEAF = re.compile(r"\bleaf_requests_total=(\d+)\b")


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): g_core=76000 leaves/s/core (MEASURED — 152 dps/core * 500 LPD).
    Used by the SEED path (manifest `trust=False` / pg-down) as a `Fixed` declared-spread prior; the
    MEASURED path (measure()/run() running the C++ bench) is the shrinkable `QuantileLaw`."""
    return G.GEN_PER_CORE_LEAVES


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="producer_leaves_per_core", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def _run_cpp_bench(reps: int) -> tuple[list[float], int]:
    """Run the C++ gen-ceiling bench ONCE at `reps` reps (sole-workload, taskset -c 0, eval mocked by
    the bench's built-in DetNet), parse its output, and return `(per_rep_serial_s, leaf_requests_total)`:

      * `per_rep_serial_s`    : the single-core `serial=<t>s` wall-time of EACH `rep <r>:` line — the
                                per-rep timing the leaves/s pool is built from (`leaf_requests_total /
                                serial` per rep, both whole-batch). `reps` IS the shrink budget — more
                                reps tightens the SE.
      * `leaf_requests_total` : the aggregate distinct-leaf count over the batch (the numerator of
                                leaves/s, and the LPD cross-read LPD = leaf_requests_total/_N_TASKS ≈ 504).

    FAIL LOUD (ADR-0002): the binary missing/unbuilt, a non-zero exit, no PASS, no per-rep readings, OR
    no `leaf_requests_total` (g_core is a LEAF rate — without the leaf count there is no measurement)
    all RAISE — NEVER a silent seed fallback (the punt this module removes). The seed is the DISTRUST
    fallback path (get_seed()), not a measured-result substitute."""
    if not (os.path.isfile(_BENCH_BIN) and os.access(_BENCH_BIN, os.X_OK)):
        raise FileNotFoundError(
            f"bench_g_core: the C++ gen-ceiling bench is not built at {_BENCH_BIN!r} (the live g_core "
            f"re-measure). Build it (cmake --build cpp/build --target chocofarm-search-runtime-bench) "
            f"or set CHOCO_SEARCH_RUNTIME_BENCH. ADR-0002: a missing measurement bench RAISES — it is "
            f"NEVER a silent fall-back to the 76000 seed as if measured.")
    for p, what, ev in ((_INSTANCE, "instance", "CHOCO_BENCH_INSTANCE"),
                        (_FACES, "faces", "CHOCO_BENCH_FACES"),
                        (_FEATURE_LAYOUT, "feature-layout", "CHOCO_FEATURE_LAYOUT")):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"bench_g_core: the C++ bench {what} geometry is missing at {p!r} (set {ev}). "
                f"ADR-0002: cannot measure g_core without the geometry.")
    taskset = list(_TASKSET) if shutil.which("taskset") else []   # pin if available; degrade loud-noted
    cmd = [
        *taskset, _BENCH_BIN,
        "--instance", _INSTANCE, "--faces", _FACES,
        "--tasks", str(_N_TASKS), "--workers", "1", "--reps", str(int(reps)),
        "--n-sims", str(_GEN_N_SIMS), "--m", str(_GEN_M), "--max-depth", str(_GEN_MAX_DEPTH),
    ]
    # cwd=_REPO_ROOT so the bench's RELATIVE feature-layout lookup resolves (it SIGABRTs otherwise from a
    # non-root CWD — verified); CHOCO_FEATURE_LAYOUT set explicitly too. --instance/--faces are absolute.
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          cwd=_REPO_ROOT, env={**os.environ, "CHOCO_FEATURE_LAYOUT": _FEATURE_LAYOUT})
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench_g_core: the C++ gen-ceiling bench exited {proc.returncode} (cmd: {' '.join(cmd)}).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n"
            f"(ADR-0002: a failed measurement RAISES — never a silent seed substitute.)")
    out = proc.stdout
    if _RE_RESULT.search(out) is None:
        raise RuntimeError(
            f"bench_g_core: the C++ bench did not report `RESULT: PASS` (a FAIL is a serial/parallel "
            f"mismatch — not a usable rate). Output:\n{out}\nstderr:\n{proc.stderr}")
    per_rep_serial_s = [float(t) for t in _RE_REP.findall(out) if float(t) > 0.0]
    if not per_rep_serial_s:
        raise RuntimeError(
            f"bench_g_core: the C++ bench reported no per-rep `serial=<t>s` readings to pool (reps="
            f"{reps}). Output:\n{out}  (ADR-0002: no readings is a loud fault, not a fabricated pool.)")
    lm = _RE_LEAF.search(out)
    if lm is None:
        raise RuntimeError(
            f"bench_g_core: the C++ bench reported no `leaf_requests_total=<n>` — g_core is a LEAF rate "
            f"and there is no leaf count to scale the per-rep timings by. Output:\n{out}  (ADR-0002: a "
            f"missing leaf numerator is a loud fault, not a seed substitute.)")
    return per_rep_serial_s, int(lm.group(1))


def _measure_raw(reps: int = 8) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): RUN the C++ gen-ceiling bench
    (eval mocked, sole-workload, taskset -c 0) sized by `reps`, and return the per-rep leaves/s/core pool
    plus provenance. g_core = leaf_requests_total/serial per rep (= per_rep_dps · LPD, the SAME physical
    generation run as R_gen in leaf units — the n_tasks cancels: (n_tasks/serial)·(leaf_total/n_tasks) =
    leaf_total/serial). `reps` IS the shrink budget — more reps -> a bigger
    pool -> a tighter median SE (the Neyman loop sizes it via the `reps` kwarg). Returns
    {'g_core_leaves_per_core' (the pool MEDIAN), 'per_rep_leaves_per_sec' (the pool the Estimate is built
    over), 'leaf_requests_total', 'lpd' (leaf_requests_total/n_tasks, the same-bench LPD cross-read),
    'reps', 'n_tasks', config}. `measure()`/`run()` both consume this ONE measurement (P1). FAIL LOUD via
    `_run_cpp_bench` if the binary is absent/unbuilt or the run fails — never the seed-as-if-measured."""
    import numpy as np
    n = max(2, int(reps))   # >= 2 readings so the bootstrap median SE is defined (a 1-rep pool has none)
    per_rep_serial_s, leaf_total = _run_cpp_bench(n)
    # leaves/s/core per rep = leaf_requests_total / serial_time. BOTH are WHOLE-batch quantities (the
    # single core ran all _N_TASKS tasks serially in `serial` seconds, doing `leaf_total` distinct leaf
    # requests over them), so their ratio is the per-core leaf RATE — the n_tasks cancels against R_gen's
    # n_tasks/serial: (n_tasks/serial)·(leaf_total/n_tasks) = leaf_total/serial = per_rep_dps · LPD. One
    # number per rep -> the shrink pool. (Verified: 16122/0.209s ≈ 77.2k, centered on the 76k seed.)
    per_rep_leaves_per_sec = [float(leaf_total) / t for t in per_rep_serial_s]
    return {
        "g_core_leaves_per_core": float(np.median(per_rep_leaves_per_sec)),
        "per_rep_leaves_per_sec": per_rep_leaves_per_sec,
        "leaf_requests_total": leaf_total,
        "lpd": leaf_total / float(_N_TASKS),
        "reps": n,
        "n_tasks": _N_TASKS,
        "config": f"search_runtime_bench leaf_requests_total/(n_tasks*serial); sims{_GEN_N_SIMS}/m{_GEN_M} "
                  f"depth{_GEN_MAX_DEPTH}; eval mocked (DetNet); taskset -c 0",
    }


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build g_core's harmonized SHRINKABLE `Estimate` — the SINGLE home of the Estimate construction
    (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)` with a BOOTSTRAP
    median SE over the per-rep leaves/s/core pool (§7.A — a real order-statistic SE, NOT a `Fixed` pin),
    `family=EMPIRICAL`, `kind='median'`, POSITIVE support. This is the ADR-0008 reclassification: g_core
    is a MEASURED quantity whose variance RESPONDS to effort (the median's `marginal_dvar_deffort` is
    `−cov/n < 0`), so the Neyman loop can FUND it (more reps -> tighter SE), where the prior `Fixed` pin
    (marginal=0) made it un-fundable and stalled the generation arm."""
    return median_estimate(res["per_rep_leaves_per_sec"], name=NAME)   # bootstrap median SE over the pool


def measure(reps: int = 8) -> "_est.Estimate":
    """Measure g_core (RUN the C++ gen-ceiling bench, eval mocked, sole-workload) and return its
    harmonized k=1 SHRINKABLE median `Estimate` (§6 Phase 4: `measure()` returns the `Estimate` the
    bench DECLARES — the driver/untrusted_drive `set_estimate`s it directly). `reps` sizes the
    measurement pool (the budget the Neyman loop passes — more reps tightens g_core's SE -> the
    generation-arm CI). TIMING-SENSITIVE — pinned (taskset -c 0); never during the fan-out."""
    return _estimate_from_raw(_measure_raw(reps=reps))


def run(reps: int = 8) -> dict[str, Any]:
    """Measure g_core (RUN the C++ gen-ceiling bench) and LOG it to postgres as a harmonized k=1
    SHRINKABLE median `Estimate` (§6 Phase 3): `QuantileLaw(p=0.5)` with a BOOTSTRAP median SE over the
    per-rep leaves/s/core pool (§7.A), `family=EMPIRICAL`, `kind='median'`. The per-rep readings are
    logged as raw PROVENANCE — the variance authority is `estimate.cov`, so the headline rate is NOT
    double-logged as a sample row (§5.2 de-dup). Returns the raw provenance dict. TIMING-SENSITIVE —
    operator-invoked, pinned (taskset -c 0), never during the fan-out."""
    res = _measure_raw(reps=reps)            # ONE measurement (Est + provenance pool)
    est = _estimate_from_raw(res)            # the SAME Estimate measure() returns (P1)
    cfg = {"kind": "cpp_gen_ceiling_measured", "config": res["config"], "reps": res["reps"],
           "n_tasks": res["n_tasks"], "leaf_requests_total": res["leaf_requests_total"],
           "lpd_cross_read": res["lpd"], "g_core_leaves_per_core_median": res["g_core_leaves_per_core"]}
    with logged_run(NAME, quantity="producer_leaves_per_core", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2): the raw per-rep readings. The headline rate lives in
        # estimate.theta_hat[0] (the SSOT), the median SE in estimate.cov.
        log(res["per_rep_leaves_per_sec"], sample_size=1)
    return res


if __name__ == "__main__":
    print(f"[bench_g_core] seed: {get_seed().mean} {get_seed().unit} (MEASURED — {get_seed().provenance})")
    print(f"[bench_g_core] live bench: {_BENCH_BIN}")
    register_self()
    print("[bench_g_core] registered. measure()/run() RUN the C++ gen-ceiling bench (eval mocked, "
          "taskset -c 0) -> a SHRINKABLE median Estimate. get_seed() is the DISTRUST fallback.")
