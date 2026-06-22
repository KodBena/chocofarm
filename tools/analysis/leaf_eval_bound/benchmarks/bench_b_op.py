"""
tools/analysis/leaf_eval_bound/benchmarks/bench_b_op.py
=================================================

LIVE benchmark for `B_op` — the server's sustained FULL-BUCKET operating point (rows/forward):
the real row count at a full bucket, the achievable serve peak (the serve curve is a sawtooth
real/(iota+slope*bucket(real)+tau_io), maximized at full buckets). It is the single quantity
that most moves the serve stage AT the binding point — a top Neyman target alongside tau_io.

WHAT run() MEASURES (1:1 with the model input). B_op is the steady-state full-bucket B the
OPTIMUM sustains under a fed producer set — its faithful measurement is the rows/forward
histogram of a saturated end-to-end run (the server fed by N producers at the full-bucket feed),
read off the server's per-forward batch-size counter. That is an end-to-end harness artifact, not
an isolated microbench; `run()` records the v1 estimate (256 rows/forward — analysis_clean.txt
GLOBAL MAX rows/fwd=511.5, server max_batch=256) with a config note that the saturated rows/fwd
histogram is the outstanding measurement, and keeps the quantity flagged needs-measurement.

A transport variant that changes the coalescing dynamics (a group-wakeup futex, a lock-free
queue) can change the sustained B_op, so a variant MAY register its own `<slug>_B_op`.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any


from leaf_eval_bound.contract import estimate as _est  # noqa: E402  — the harmonized Estimate contract (measure() returns one — §6 Phase 4)
from leaf_eval_bound.contract import grounding as G  # noqa: E402
from leaf_eval_bound.benchmarks.estimators import pin_estimate  # noqa: E402
from leaf_eval_bound.benchmarks.scaffold import bench as _scaffold  # noqa: E402  — move 6 wiring

NAME = "B_op"
MODULE_PATH = "leaf_eval_bound.benchmarks.bench_b_op"
_DESC = ("Server sustained FULL-BUCKET operating point (rows/forward): the achievable serve peak (the "
         "serve sawtooth is maximized at full buckets). v1 256 (GLOBAL MAX rows/fwd=511.5, max_batch=256). "
         "Real measurement = saturated end-to-end rows/forward histogram. Top serve-stage Neyman target.")


def get_seed() -> G.Grounded:
    """The v1 seed (DISTRUST fallback): B_op=256 rows/forward (full-bucket operating point)."""
    return G.SERVE_FULL_BUCKET


def _measure_raw() -> dict[str, Any]:
    """The raw-pool PROVENANCE producer (the §6 Phase-4 internal helper): the current B_op estimate
    (full-bucket rows/forward). A faithful measure is the saturated end-to-end rows/forward histogram (the
    server's per-forward batch-size counter under a fed producer set), an e2e harness artifact. Returns
    {'b_op_rows', 'note'}. `measure()` wraps the seed into a `Fixed` Estimate; `run()` uses this dict for the
    raw provenance row."""
    return {"b_op_rows": get_seed().mean,
            "note": "full-bucket operating point; saturated end-to-end rows/forward histogram outstanding"}


def _estimate_from_raw(res: dict[str, Any]) -> "_est.Estimate":
    """Build B_op's harmonized `Estimate` — the SINGLE home of the Estimate construction (P1), called by
    BOTH `measure()` and `run()`. A k=1 `Fixed` declared-spread Estimate recovering the declared spread
    UN-DIVIDED (`cov=[[σ²]]`, the §5 store-bug fix — B_op's σ=64 reaches the instance variance). The value
    is the seed pin (`res['b_op_rows']`); the spread is the declared σ (a pin has no sample n)."""
    return pin_estimate(res["b_op_rows"], get_seed().sigma, name=NAME)


# Move 6: the shared scaffold wires register_self / measure / run from the bench-specific parts above.
# B_op is a v1 operating-point pin (the saturated e2e rows/forward histogram is the outstanding real
# measurement); run() logs the pinned value (sample_size=None) flagged needs-measurement.
_B = _scaffold(
    name=NAME, quantity="serve_full_bucket_rows", module_path=MODULE_PATH, description=_DESC,
    seed=get_seed, measure_raw=_measure_raw, estimate_from_raw=_estimate_from_raw,
    run_config=lambda res, **kw: {"kind": "operating_point",
                            "needs_measurement": "saturated end-to-end rows/forward histogram (server batch-size counter)",
                            "note": res["note"]},
    run_log=lambda res, log, **kw: log(res["b_op_rows"], sample_size=None),
)
register_self, measure, run = _B.register_self, _B.measure, _B.run


if __name__ == "__main__":
    print(f"[bench_b_op] seed: {get_seed().mean} {get_seed().unit} (provenance: {get_seed().provenance})")
    register_self()
    print("[bench_b_op] registered. The real measurement is the saturated e2e rows/forward histogram.")
