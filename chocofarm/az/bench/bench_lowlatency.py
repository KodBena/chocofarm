#!/usr/bin/env python3
"""
chocofarm/az/bench/bench_lowlatency.py — the toy dispatch-bound benchmark for the low-overhead JAX
dispatcher (chocofarm/az/lowlatency.py).

THE FEASIBILITY GATE (ADR-0009 perf-substantiation; the falsifiable claim). The premise: a small jit'd
JAX computation at small batch is DISPATCH-BOUND — the FIXED per-call cost (the regression INTERCEPT of
time-vs-batch-rows) dominates the per-row compute SLOPE, and the abstraction's job is to LOWER THE
INTERCEPT without inflating the slope. This bench tests that on a toy 2-layer MLP value head (the shape
the search leaf-eval runs), fitting `time = intercept + slope * rows` for each dispatch path and
reporting intercept and slope with variance (median + IQR over repeats, best-of-N to exclude the
one-time compile + OS jitter).

The paths compared, in the SERVER'S TRUE call pattern (a fresh host numpy `x` in, the result pulled to
host — the inference server's `np.asarray(forward(...))`):
  * plain_jit_host    — `np.asarray(jax.jit(fn)(params, jnp.asarray(x_host)))`, the baseline.
  * robust_host       — the LowLatencyFn robust AOT Compiled call, host x in, host pull out.
  * unsafe_host       — the LowLatencyFn unsafe loaded-executable call, host x in, host pull out.
And a DEVICE-RESIDENT control (x pre-placed on device, no host transfer in; block-don't-pull out) that
ISOLATES the dispatch floor from the host<->device round-trip:
  * plain_jit_device  — `jax.jit(fn)(params, x_dev).block_until_ready()`.
  * robust_device     — the robust call on a device-resident x.
  * unsafe_device     — the unsafe call on a device-resident x.

WHY both regimes: the host-pattern is what the server actually pays; the device-control reveals WHERE the
intercept lives. The measured finding (recorded so the next reader does not re-derive it; warm, median±IQR,
R²>0.99 fits, jax 0.10.1 single-thread CPU, in_dim=80 hidden=256): the ROBUST AOT handle LOWERS the
host-pattern intercept ~47% — plain `jax.jit` intercept ≈121us vs robust ≈64us, a ~57us drop reproducible
to <1us — with the slope unchanged (≈2.04→≈1.96 us/row). The win is the handle staging the PARAMS
device-resident once (re-passing those device buffers), so each call transfers only `x`, eliminating the
per-call params flatten+transfer plain `jit(params, jnp.asarray(x))` repeats; the device-resident control
(params staged AND x pre-placed) confirms it — plain-jit ≈46us vs robust ≈49us converge. The UNSAFE
loaded-executable call is SLOWER here (intercept ≈224us, ~+103us vs plain jit: `ExecuteReplicated.__call__`
is heavier than pjit's C++ fast-path and the `Compiled` wrapper), so it is off by default. The further
orthogonal lever (device-resident x, donation, a bigger batch to amortize the host-pull) is the CALLER's,
not the AOT-vs-jit axis. The bench reports the numbers so the verdict is a measurement, not an assertion.

Run (pinned + bounded; the config XLA single-thread pin lands via the lowlatency import):
    PYTHONPATH=. taskset -c 0 /home/bork/w/vdc/venvs/generic/bin/python -m chocofarm.az.bench.bench_lowlatency \\
        --out ~/w/vdc/chocobo/bench/lowlatency/results.json

`--batches`, `--in-dim`, `--hidden`, `--iters`, `--repeat` tune the sweep. The JSON dump carries the raw
per-(path, batch) median us/call AND the fitted (intercept, slope, R²) per path, so a re-run is mechanical.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Callable

import numpy as np

# Importing the dispatcher applies the XLA/OMP single-thread pin (it side-effect-imports config), so the
# bench runs on the SAME single-threaded Eigen backend the inference server uses — the regime where
# dispatch dominates the tiny matmul (config.py's rationale).
from chocofarm.az.lowlatency import compile_lowlatency, run


def _mlp_value(p: dict[str, Any], x: Any) -> Any:
    """The toy dispatch-bound fn: a 2-layer MLP value head `(params, x) -> (B, 1)` — the shape the search
    leaf-eval forward runs, small enough that Python dispatch dominates XLA compute at small batch."""
    import jax.numpy as jnp
    a1 = jnp.maximum(x @ p["W1"] + p["b1"], 0.0)
    a2 = jnp.maximum(a1 @ p["W2"] + p["b2"], 0.0)
    return a2 @ p["Wv"] + p["bv"]


def _toy_params(in_dim: int, hidden: int, seed: int = 0) -> dict[str, Any]:
    import jax.numpy as jnp
    rng = np.random.default_rng(seed)

    def mk(a: int, b: int, s: float = 1.0) -> Any:
        return jnp.asarray((rng.standard_normal((a, b)) * s).astype(np.float32))
    return {"W1": mk(in_dim, hidden), "b1": jnp.zeros(hidden, jnp.float32),
            "W2": mk(hidden, hidden), "b2": jnp.zeros(hidden, jnp.float32),
            "Wv": mk(hidden, 1, 0.1), "bv": jnp.zeros(1, jnp.float32)}


def _median_iqr_us(fn: Callable[[], Any], iters: int, repeat: int) -> tuple[float, float]:
    """Median (and inter-quartile range) over `repeat` runs of the per-call wall time (microseconds) of
    `fn` executed `iters` times. The MEDIAN is the reported center (best-of robustness to a scheduler
    blip — ADR-0009 median+IQR), the IQR (q75 - q25) the spread reported alongside it as the variance."""
    per_call: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        per_call.append((time.perf_counter() - t0) / iters)
    arr = np.array(sorted(per_call))
    med = float(np.median(arr)) * 1e6
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25)) * 1e6
    return med, iqr


def _fit_line(rows: list[int], times_us: list[float]) -> tuple[float, float, float]:
    """Least-squares fit `time = intercept + slope * rows`. Returns (intercept_us, slope_us_per_row, R²).
    The INTERCEPT is the fixed per-call dispatch cost (the premise's target); the SLOPE is the per-row
    compute. R² reports how linear the relationship is (a poor fit means the model — and the
    intercept/slope read-off — should be distrusted)."""
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(times_us, dtype=np.float64)
    A = np.vstack([np.ones_like(x), x]).T          # columns: [1, rows] -> coeffs [intercept, slope]
    (intercept, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = intercept + slope * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(intercept), float(slope), r2


def bench(batches: list[int], in_dim: int, hidden: int, iters: int, repeat: int, warmup: int
          ) -> dict[str, Any]:
    """Run the sweep: for each path and each batch size, time `iters` calls `repeat` times and record the
    median+IQR per-call us; then fit `time = intercept + slope*rows` per path. Returns the full record."""
    import jax
    import jax.numpy as jnp

    params = _toy_params(in_dim, hidden)
    jfn = jax.jit(_mlp_value)

    # Per-batch handles + inputs. A fresh host matrix per batch (the server's reality); a device copy for
    # the device-resident control. Build a robust and an unsafe handle per batch (one AOT executable each
    # — the handle holds exactly one (shape,dtype) signature).
    per_batch: dict[int, dict[str, Any]] = {}
    for B in batches:
        x_host = np.random.default_rng(B + in_dim).standard_normal((B, in_dim)).astype(np.float32)
        x_dev = jax.device_put(jnp.asarray(x_host))
        h_robust = compile_lowlatency(_mlp_value, params, x_host)               # robust default
        h_unsafe = compile_lowlatency(_mlp_value, params, x_host, prefer_unsafe=True)
        per_batch[B] = {"x_host": x_host, "x_dev": x_dev, "robust": h_robust, "unsafe": h_unsafe}

    # The six timed closures per batch. host-pattern PULLS the result to host (np.asarray) — the server's
    # true cost; device-pattern blocks on the device array (no fresh host input, no host pull) — the
    # dispatch floor. (`run(..., unsafe=...)` selects the path; `__call__` would route via use_unsafe.)
    def closures(B: int) -> dict[str, Callable[[], Any]]:
        d = per_batch[B]
        xh, xd, hr, hu = d["x_host"], d["x_dev"], d["robust"], d["unsafe"]
        return {
            "plain_jit_host":   lambda: np.asarray(jfn(params, jnp.asarray(xh))),
            "robust_host":      lambda: np.asarray(run(hr, xh, unsafe=False)),
            "unsafe_host":      lambda: np.asarray(run(hu, xh, unsafe=True)),
            "plain_jit_device": lambda: jfn(params, xd).block_until_ready(),
            "robust_device":    lambda: run(hr, xd, unsafe=False).block_until_ready(),
            "unsafe_device":    lambda: run(hu, xd, unsafe=True).block_until_ready(),
        }

    paths = ["plain_jit_host", "robust_host", "unsafe_host",
             "plain_jit_device", "robust_device", "unsafe_device"]

    # WARM UP every (path, batch) so the one-time AOT compile + first-call XLA cost is excluded from the
    # timed region (ADR-0009 measure-honesty — the cold compile is the confound the inference server's
    # own warmup() also removes).
    for B in batches:
        cl = closures(B)
        for name in paths:
            for _ in range(warmup):
                cl[name]()

    # Timed sweep: median+IQR per (path, batch).
    results: dict[str, dict[int, dict[str, float]]] = {name: {} for name in paths}
    for B in batches:
        cl = closures(B)
        for name in paths:
            med, iqr = _median_iqr_us(cl[name], iters, repeat)
            results[name][B] = {"median_us": med, "iqr_us": iqr}

    # Linear fit per path: intercept (fixed dispatch cost) + slope (per-row compute), with R².
    fits: dict[str, dict[str, float]] = {}
    for name in paths:
        rows = list(batches)
        med_us = [results[name][B]["median_us"] for B in batches]
        intercept, slope, r2 = _fit_line(rows, med_us)
        fits[name] = {"intercept_us": intercept, "slope_us_per_row": slope, "r2": r2}

    return {
        "config": {"batches": batches, "in_dim": in_dim, "hidden": hidden,
                   "iters": iters, "repeat": repeat, "warmup": warmup,
                   "xla_flags": os.environ.get("XLA_FLAGS"), "omp_num_threads": os.environ.get("OMP_NUM_THREADS")},
        "per_batch_us": {name: {str(B): results[name][B] for B in batches} for name in paths},
        "fits": fits,
    }


def _verdict(fits: dict[str, dict[str, float]]) -> str:
    """The falsifiable claim, stated against the measured intercepts: does the abstraction LOWER the
    intercept (the fixed dispatch cost) below plain jit, WITHOUT inflating the slope? Reported for the
    host pattern (the server's reality) and noted against the device floor."""
    pj_h = fits["plain_jit_host"]
    rb_h = fits["robust_host"]
    un_h = fits["unsafe_host"]
    lines = []
    lines.append("VERDICT — does the abstraction lower the per-call intercept (host pattern)?")
    lines.append(f"  plain_jit_host : intercept {pj_h['intercept_us']:8.2f} us   slope {pj_h['slope_us_per_row']:7.3f} us/row")
    lines.append(f"  robust_host    : intercept {rb_h['intercept_us']:8.2f} us   slope {rb_h['slope_us_per_row']:7.3f} us/row")
    lines.append(f"  unsafe_host    : intercept {un_h['intercept_us']:8.2f} us   slope {un_h['slope_us_per_row']:7.3f} us/row")
    d_rb = rb_h["intercept_us"] - pj_h["intercept_us"]
    d_un = un_h["intercept_us"] - pj_h["intercept_us"]
    lines.append(f"  robust intercept delta vs plain jit: {d_rb:+8.2f} us  ({'LOWER' if d_rb < 0 else 'NOT lower'})")
    lines.append(f"  unsafe intercept delta vs plain jit: {d_un:+8.2f} us  ({'LOWER' if d_un < 0 else 'NOT lower'})")
    pj_d = fits["plain_jit_device"]
    lines.append("  (device-resident control — isolates the dispatch floor from the host<->device round-trip:)")
    lines.append(f"   plain_jit_device intercept {pj_d['intercept_us']:8.2f} us   slope {pj_d['slope_us_per_row']:7.3f} us/row")
    held = (d_rb < 0) or (d_un < 0)
    lines.append(f"  => premise's remedy holds (an AOT path lowers the host-pattern intercept): {held}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Toy dispatch-bound bench for the low-overhead JAX dispatcher.")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--in-dim", type=int, default=80)     # ~the ValueMLP feature dim
    ap.add_argument("--hidden", type=int, default=256)    # the ValueMLP hidden
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--out", type=str, default=None, help="JSON dump path (preserve under ~/w/vdc, not /tmp)")
    args = ap.parse_args()

    rec = bench(args.batches, args.in_dim, args.hidden, args.iters, args.repeat, args.warmup)

    print(f"toy MLP value head  in_dim={args.in_dim} hidden={args.hidden}  "
          f"iters={args.iters} repeat={args.repeat}  XLA_FLAGS={rec['config']['xla_flags']}")
    print()
    paths = list(rec["per_batch_us"].keys())
    hdr = "  batch  " + "".join(f"{p:>20}" for p in paths)
    print(hdr)
    for B in args.batches:
        row = f"  {B:5d}  "
        for p in paths:
            cell = rec["per_batch_us"][p][str(B)]
            row += f"{cell['median_us']:11.2f}±{cell['iqr_us']:<6.2f} "
        print(row)
    print()
    print("LINEAR FIT  time = intercept + slope * rows   (intercept = fixed dispatch cost, slope = per-row compute)")
    for p in paths:
        f = rec["fits"][p]
        print(f"  {p:<18} intercept {f['intercept_us']:8.2f} us   slope {f['slope_us_per_row']:8.4f} us/row   R²={f['r2']:.4f}")
    print()
    print(_verdict(rec["fits"]))

    if args.out:
        out = os.path.expanduser(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
