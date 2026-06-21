"""
tools/analysis/OpenTURNS/benchmarks/bench_r_gen.py
==================================================

LIVE benchmark for `R_gen` — the per-core single-generator decision rate (decisions/s/core):
the producer ceiling input. Baseline, transport-invariant (a generator core's search rate is
independent of which transport carries its leaf-eval requests; the transport binds the SERVE
stage, not generation). The aggregate producer cap is `N_gen * R_gen`.

WHAT run()/measure() MEASURES (1:1 with the model input). The decisions/s a SINGLE generator
core sustains with the leaf-eval MOCKED (so the read is the producer's own search throughput,
not gated on the inference server) — the C++ gen-ceiling SOLE-WORKLOAD bench
`cpp/build/chocofarm-search-runtime-bench` (source cpp/src/search_runtime_bench.cpp). That bench
runs a batch of independent Gumbel-AZ decisions through `SerialRuntime` over a LOCAL in-process
`DetNet` — a deterministic, stateless, weightless, RNG-free leaf evaluator (the eval mock by
construction) — at the gen-ceiling config (`sims256/m24`, max_depth 24: the config whose distinct-
node count IS the LPD=500 grounding, leaf_requests_total/n_tasks ≈ 500), and reports the SINGLE-
CORE `serial_dps = n_tasks / best_serial`. That serial-runtime rate IS R_gen.

WHY SHRINKABLE NOW (the ADR-0008 reclassification this module IS). R_gen is a MEASURED quantity,
not a config pin (the seed itself is labelled MEASURED — adapter.md §2 line 93). The prior version
of this module PUNTED — `_measure_raw()` returned the seed (152) and `_estimate_from_raw()` wrapped
it in `pin_estimate(...)` → an un-shrinkable `Fixed` Estimate — so the Neyman loop could not sample
it (a `Fixed` law's `marginal_dvar_deffort` is 0 → `A_i = 0` → never funded → the generation arm of
`untrusted_drive` STALLS). The binary EXISTS and is BUILT, so the measurement is real: this module
RUNS it and returns a SHRINKABLE `QuantileLaw` (median) Estimate over a pool of per-rep readings
(a real bootstrap median SE — docs/design/harmonized-estimator-interface.md §7.A, §3 MEDIAN row and
the PIN-now/measurable-later row: `R_gen`/`g_core` as a C++ rate flips `Fixed` -> `Poolwise`/
`QuantileLaw` once instrumented). More/longer runs (a bigger `reps` budget) -> a tighter pool ->
a tighter R_gen SE -> a tighter generation-arm CI, so the loop FUNDS R_gen instead of stalling.

`get_seed()` stays the DISTRUST fallback (the v1 152 dps/core seed, a `Fixed` declared-spread prior
on the SEED path — the manifest's `trust=False` / pg-down route; only the MEASURED path is
shrinkable). FAIL LOUD (ADR-0002): if the C++ bench is absent/unbuilt at run time, or the run fails,
or it does not report `RESULT: PASS`, `_measure_raw()` RAISES — it NEVER silently falls back to the
seed-as-if-measured (that silent fallback is exactly the punt this module removes).

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

NAME = "R_gen"
MODULE_PATH = "benchmarks.bench_r_gen"
_DESC = ("Per-core single-generator decision rate (decisions/s/core): the producer ceiling input "
         "(aggregate = N_gen*R_gen). LIVE = the C++ gen-ceiling sole-workload bench "
         "(search_runtime_bench, eval mocked, SerialRuntime serial_dps at sims256/m24). v1 seed 152 "
         "dps/core. Baseline, transport-invariant.")

# --- The C++ gen-ceiling bench: the binary, the gen-ceiling Gumbel config, the eval mock --------
# The repo root (this file is tools/analysis/OpenTURNS/benchmarks/bench_r_gen.py -> up 4:
# benchmarks -> OpenTURNS -> analysis -> tools -> <repo root>).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
# The built binary + the instance/faces geometry (the same convention exit_loop.py/cpp_actor_loop.py
# use: `cpp/build/chocofarm-*` + `chocofarm/data/{instance,faces}.json`). Env-overridable so an
# operator can point at an alternate build/geometry without editing the bench.
_BENCH_BIN = os.environ.get(
    "CHOCO_SEARCH_RUNTIME_BENCH",
    os.path.join(_REPO_ROOT, "cpp", "build", "chocofarm-search-runtime-bench"))
_INSTANCE = os.environ.get("CHOCO_BENCH_INSTANCE", os.path.join(_REPO_ROOT, "chocofarm", "data", "instance.json"))
_FACES = os.environ.get("CHOCO_BENCH_FACES", os.path.join(_REPO_ROOT, "chocofarm", "data", "faces.json"))
# The bench ALSO reads the feature-layout spec by a RELATIVE path (chocofarm/data/feature_layout.json —
# "run from the repo root"); --instance/--faces are passed absolute, but this one is looked up internally,
# so the subprocess is run with cwd=_REPO_ROOT AND CHOCO_FEATURE_LAYOUT set (belt + suspenders) so it
# resolves from ANY caller CWD. Without this the bench SIGABRTs from e.g. tools/analysis/OpenTURNS/.
_FEATURE_LAYOUT = os.environ.get("CHOCO_FEATURE_LAYOUT", os.path.join(_REPO_ROOT, "chocofarm", "data", "feature_layout.json"))
# The gen-ceiling Gumbel config: sims256/m24, max_depth 24 — the tree whose aggregate distinct-node
# count IS the LPD=500 grounding (leaf_requests_total/n_tasks ≈ 500), so serial_dps ≈ 152 dps/core
# (verified: 6 invocations 152.4–153.2 dps/core; leaf_requests_total/n_tasks ≈ 504). These are the
# config the seed's "152 dps/core (76k leaves/s), 4.0x linear" provenance is read at.
_GEN_N_SIMS = 256
_GEN_M = 24
_GEN_MAX_DEPTH = 24
_N_TASKS = 32          # the batch of independent root tasks the serial runtime times (each ~150–153 dps)
_TASKSET = ("taskset", "-c", "0")   # sole-workload single-core pin (TIMING-SENSITIVE; CLAUDE.md core 0)

# Parse `serial_dps=<n>` off the RESULT line, and `rep <r>: serial=<t>s parallel=<t>s` for the pool.
_RE_RESULT = re.compile(r"RESULT:\s*PASS\b.*?\bserial_dps=([0-9.eE+-]+)")
_RE_REP = re.compile(r"^rep\s+\d+:\s*serial=([0-9.eE+-]+)s\b", re.MULTILINE)
_RE_LEAF = re.compile(r"\bleaf_requests_total=(\d+)\b")


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): R_gen=152 dps/core (MEASURED, 4.0x linear). Used by the SEED
    path (manifest `trust=False` / pg-down) as a `Fixed` declared-spread prior; the MEASURED path
    (measure()/run() running the C++ bench) is the shrinkable `QuantileLaw`."""
    return G.GEN_PER_CORE_DPS


def register_self() -> Any:
    from bench_common import register_quantity
    return register_quantity(NAME, quantity="producer_decisions_per_core", units=get_seed().unit,
                             description=_DESC, module_path=MODULE_PATH)


def _run_cpp_bench(reps: int) -> tuple[list[float], float, int]:
    """Run the C++ gen-ceiling bench ONCE at `reps` reps (sole-workload, taskset -c 0, eval mocked by
    the bench's built-in DetNet), parse its output, and return `(per_rep_dps, serial_dps_headline,
    leaf_requests_total)`:

      * `per_rep_dps`         : `_N_TASKS / serial_time` for EACH `rep <r>: serial=<t>s` line — a pool
                                of `reps` decisions/s/core readings (the shrink budget — more reps
                                tightens the SE).
      * `serial_dps_headline` : the bench's own `serial_dps = _N_TASKS / best_serial` (best-of-reps;
                                the gen-ceiling headline ≈ 152).
      * `leaf_requests_total` : the aggregate distinct-leaf count over the batch (for the LPD/g_core
                                cross-read: LPD = leaf_requests_total/_N_TASKS ≈ 500).

    FAIL LOUD (ADR-0002): the binary missing/unbuilt, a non-zero exit, no PASS, or no per-rep readings
    all RAISE — NEVER a silent seed fallback (the punt this module removes). The seed is the DISTRUST
    fallback path (get_seed()), not a measured-result substitute."""
    if not (os.path.isfile(_BENCH_BIN) and os.access(_BENCH_BIN, os.X_OK)):
        raise FileNotFoundError(
            f"bench_r_gen: the C++ gen-ceiling bench is not built at {_BENCH_BIN!r} (the live R_gen "
            f"re-measure). Build it (cmake --build cpp/build --target chocofarm-search-runtime-bench) "
            f"or set CHOCO_SEARCH_RUNTIME_BENCH. ADR-0002: a missing measurement bench RAISES — it is "
            f"NEVER a silent fall-back to the 152 seed as if measured.")
    for p, what, ev in ((_INSTANCE, "instance", "CHOCO_BENCH_INSTANCE"),
                        (_FACES, "faces", "CHOCO_BENCH_FACES"),
                        (_FEATURE_LAYOUT, "feature-layout", "CHOCO_FEATURE_LAYOUT")):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"bench_r_gen: the C++ bench {what} geometry is missing at {p!r} (set {ev}). "
                f"ADR-0002: cannot measure R_gen without the geometry.")
    taskset = list(_TASKSET) if shutil.which("taskset") else []   # pin if available; degrade loud-noted
    cmd = [
        *taskset, _BENCH_BIN,
        "--instance", _INSTANCE, "--faces", _FACES,
        "--tasks", str(_N_TASKS), "--workers", "1", "--reps", str(int(reps)),
        "--n-sims", str(_GEN_N_SIMS), "--m", str(_GEN_M), "--max-depth", str(_GEN_MAX_DEPTH),
    ]
    # cwd=_REPO_ROOT so the bench's RELATIVE feature-layout lookup resolves (it SIGABRTs otherwise from a
    # non-root CWD); CHOCO_FEATURE_LAYOUT set explicitly too. --instance/--faces are already absolute.
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          cwd=_REPO_ROOT, env={**os.environ, "CHOCO_FEATURE_LAYOUT": _FEATURE_LAYOUT})
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench_r_gen: the C++ gen-ceiling bench exited {proc.returncode} (cmd: {' '.join(cmd)}).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n"
            f"(ADR-0002: a failed measurement RAISES — never a silent seed substitute.)")
    out = proc.stdout
    m = _RE_RESULT.search(out)
    if m is None:
        raise RuntimeError(
            f"bench_r_gen: the C++ bench did not report `RESULT: PASS serial_dps=...` (a FAIL is a "
            f"serial/parallel mismatch — not a usable rate). Output:\n{out}\nstderr:\n{proc.stderr}")
    serial_dps_headline = float(m.group(1))
    per_rep_dps = [float(_N_TASKS) / float(t) for t in _RE_REP.findall(out) if float(t) > 0.0]
    if not per_rep_dps:
        raise RuntimeError(
            f"bench_r_gen: the C++ bench reported no per-rep `serial=<t>s` readings to pool (reps="
            f"{reps}). Output:\n{out}  (ADR-0002: no readings is a loud fault, not a fabricated pool.)")
    lm = _RE_LEAF.search(out)
    leaf_total = int(lm.group(1)) if lm else 0
    return per_rep_dps, serial_dps_headline, leaf_total


def _measure_raw(reps: int = 8) -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): RUN the C++ gen-ceiling bench
    (eval mocked, sole-workload, taskset -c 0) sized by `reps`, and return the per-rep decisions/s/core
    pool plus provenance. `reps` IS the shrink budget — more reps -> a bigger pool -> a tighter median
    SE (the Neyman loop sizes it via the `reps` kwarg). Returns {'r_gen_dps_per_core' (the pool MEDIAN),
    'per_rep_dps' (the pool the Estimate is built over), 'serial_dps_headline', 'leaf_requests_total',
    'lpd' (leaf_requests_total/n_tasks, the same-bench LPD cross-read), 'reps', 'n_tasks', config}.
    `measure()`/`run()` both consume this ONE measurement (P1). FAIL LOUD via `_run_cpp_bench` if the
    binary is absent/unbuilt or the run fails — never the seed-as-if-measured."""
    import numpy as np
    n = max(2, int(reps))   # >= 2 readings so the bootstrap median SE is defined (a 1-rep pool has none)
    per_rep_dps, serial_dps_headline, leaf_total = _run_cpp_bench(n)
    return {
        "r_gen_dps_per_core": float(np.median(per_rep_dps)),
        "per_rep_dps": per_rep_dps,
        "serial_dps_headline": serial_dps_headline,
        "leaf_requests_total": leaf_total,
        "lpd": (leaf_total / float(_N_TASKS)) if leaf_total else None,
        "reps": n,
        "n_tasks": _N_TASKS,
        "config": f"search_runtime_bench serial_dps; sims{_GEN_N_SIMS}/m{_GEN_M} depth{_GEN_MAX_DEPTH}; "
                  f"eval mocked (DetNet); taskset -c 0",
    }


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build R_gen's harmonized SHRINKABLE `Estimate` — the SINGLE home of the Estimate construction
    (P1), called by BOTH `measure()` and `run()`. A k=1 median `QuantileLaw(p=0.5)` with a BOOTSTRAP
    median SE over the per-rep decisions/s/core pool (§7.A — a real order-statistic SE, NOT a `Fixed`
    pin), `family=EMPIRICAL`, `kind='median'`, POSITIVE support. This is the ADR-0008 reclassification:
    R_gen is a MEASURED quantity whose variance RESPONDS to effort (the median's `marginal_dvar_deffort`
    is `−cov/n < 0`), so the Neyman loop can FUND it (more reps -> tighter SE), where the prior `Fixed`
    pin (marginal=0) made it un-fundable and stalled the generation arm."""
    return median_estimate(res["per_rep_dps"], name=NAME)   # bootstrap median SE over the per-rep pool


def measure(reps: int = 8) -> "_est.Estimate":
    """Measure R_gen (RUN the C++ gen-ceiling bench, eval mocked, sole-workload) and return its
    harmonized k=1 SHRINKABLE median `Estimate` (§6 Phase 4: `measure()` returns the `Estimate` the
    bench DECLARES — the driver/untrusted_drive `set_estimate`s it directly). `reps` sizes the
    measurement pool (the budget the Neyman loop passes — more reps tightens R_gen's SE -> the
    generation-arm CI). TIMING-SENSITIVE — pinned (taskset -c 0); never during the fan-out."""
    return _estimate_from_raw(_measure_raw(reps=reps))


def run(reps: int = 8) -> dict[str, Any]:
    """Measure R_gen (RUN the C++ gen-ceiling bench) and LOG it to postgres as a harmonized k=1
    SHRINKABLE median `Estimate` (§6 Phase 3): `QuantileLaw(p=0.5)` with a BOOTSTRAP median SE over the
    per-rep decisions/s/core pool (§7.A), `family=EMPIRICAL`, `kind='median'`. The per-rep readings are
    logged as raw PROVENANCE — the variance authority is `estimate.cov`, so the headline rate is NOT
    double-logged as a sample row (§5.2 de-dup). Returns the raw provenance dict. TIMING-SENSITIVE —
    operator-invoked, pinned (taskset -c 0), never during the fan-out."""
    res = _measure_raw(reps=reps)            # ONE measurement (Est + provenance pool)
    est = _estimate_from_raw(res)            # the SAME Estimate measure() returns (P1)
    cfg = {"kind": "cpp_gen_ceiling_measured", "config": res["config"], "reps": res["reps"],
           "n_tasks": res["n_tasks"], "serial_dps_headline": res["serial_dps_headline"],
           "leaf_requests_total": res["leaf_requests_total"], "lpd_cross_read": res["lpd"],
           "r_gen_dps_per_core_median": res["r_gen_dps_per_core"]}
    with logged_run(NAME, quantity="producer_decisions_per_core", units=get_seed().unit, description=_DESC,
                    module_path=MODULE_PATH, config=cfg, estimate=est) as log:
        # PROVENANCE only (§5.2): the raw per-rep readings. The headline rate lives in
        # estimate.theta_hat[0] (the SSOT), the median SE in estimate.cov.
        log(res["per_rep_dps"], sample_size=1)
    return res


if __name__ == "__main__":
    print(f"[bench_r_gen] seed: {get_seed().mean} {get_seed().unit} (MEASURED — {get_seed().provenance})")
    print(f"[bench_r_gen] live bench: {_BENCH_BIN}")
    register_self()
    print("[bench_r_gen] registered. measure()/run() RUN the C++ gen-ceiling bench (eval mocked, "
          "taskset -c 0) -> a SHRINKABLE median Estimate. get_seed() is the DISTRUST fallback.")
