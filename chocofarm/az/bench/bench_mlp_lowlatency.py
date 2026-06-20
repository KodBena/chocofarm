#!/usr/bin/env python3
"""
chocofarm/az/bench/bench_mlp_lowlatency.py — the REAL-MLP dispatch-decomposition benchmark for the
low-overhead JAX dispatcher (chocofarm/az/lowlatency.py), at the inference server's production geometry.

WHAT THIS IS (and is not). The toy bench (chocofarm/az/bench/bench_lowlatency.py) established the
falsifiable claim on a *toy* value head (in_dim=80, value-only): the robust AOT handle lowers the
per-call dispatch INTERCEPT ~47% (≈121us→≈64us) by staging the PARAMS device-resident once, with the
slope unchanged. This bench re-runs that gate on the REAL net — the `forward.forward_core` graph at the
inference server's geometry (in_dim=241 → hidden=256 → value head + n_actions=65 policy head, float32,
residual OFF) — to (1) confirm the params-staging win REPLICATES on the production MLP, and (2) DECOMPOSE
the inference server's per-call fixed cost into its transfer components, sizing how much of the real MLP's
intercept is consolidatable host<->device transfer versus the irreducible dispatch floor. It is a
MEASUREMENT only: it imports the production forward read-only and does NOT rewire the server (the
consolidation of run_microbatch is a separate follow-on — ADR-0009 captures the number first).

THE DECOMPOSITION (the four variants, each a clean linear fit `time = intercept + slope·rows`). All four
run the SAME forward graph — the production `[v | logits]` block: `forward_core` then de-standardize the
value (`v = v_std·y_std + y_mean`) then `concatenate([v, logits])` → `(B, 1+n_actions)`, exactly the
graph `inference_server.jit_forward_core` jits — so the comparison is apples-to-apples (verified allclose,
ABS_TOL=1e-4, the project's forward bar). The y-scale scalars ride FOLDED INTO the params closure
(`p["_ym"]`/`p["_ys"]`) rather than as the production forward's two traced args, so the shared graph fits
the lowlatency handle's `fn(params, x)` contract while staying numerically the same de-standardize
(P1/P6 — the same forward, one transcription, behavioral-equivalence not byte-identity). The variants:

  1. `current` — the server's per-call path TODAY. A plain `jax.jit(fwd)` called as
     `np.asarray(jfn(params_HOST, Xb_host))`: the float32 weight dict passed as HOST numpy every call (the
     server hands `run_microbatch` the host params from `params_from_manifest_blob`), the host input cast
     FOLDED into the jit (no separate eager `jnp.asarray` — the production fold), and the result pulled to
     host (`np.asarray`, the server's one device→host pull). The full fixed per-call cost the server pays.
  2. `staged_params` — the lowlatency ROBUST handle: the params are staged DEVICE-RESIDENT ONCE at
     construction (re-passed as device buffers each call), the input still host (`Xb_host`, per-call
     host→device), the output still pulled (`np.asarray`). `current → staged_params` delta = the PARAMS
     host→device transfer the server repeats every call (the ~57us the toy bench isolated).
  3. `staged_params_input` — params staged + the input ALSO pre-placed device-resident (`Xb_dev`, no
     per-call `jnp.asarray`/transfer). `staged_params → this` delta = the INPUT host→device transfer.
  4. `fully_device` — params + input staged + the output kept device-resident (`block_until_ready()`, no
     `np.asarray` host pull). `staged_params_input → this` delta = the OUTPUT device→host pull; THIS
     variant's residual intercept = dispatch + framing, the irreducible floor.

So the CONSECUTIVE INTERCEPT DELTAS decompose the `current` fixed cost into {params transfer, input
transfer, output pull, dispatch floor}. The DELTAS are the levers consolidating run_microbatch would
claim (stage params via the handle; keep the input device-resident across the drain; batch the one
device→host pull); the `fully_device` intercept is the floor those levers cannot move.

METHODOLOGY (mirrors the toy bench — ADR-0009 rigor). For each variant and each batch size in a sweep of
rows: WARM every (variant, batch) so the one-time AOT compile + first-call XLA cost is excluded from the
timed region; then time `iters` calls `repeat` times and record the MEDIAN per-call us with its IQR
(q75−q25) as the spread; then a least-squares fit `time = intercept + slope·rows` per variant, reporting
intercept (the fixed dispatch cost — the premise's target), slope (per-row compute), and R² (the fit
quality — a poor fit means distrust the intercept read-off). Pinned single-thread CPU (the lowlatency
import applies config.py's XLA/OMP pin), best-of-`repeat` median to exclude scheduler blips.

Run (pinned + bounded; the config XLA single-thread pin lands via the lowlatency import):
    PYTHONPATH=. taskset -c 0 /home/bork/w/vdc/venvs/generic/bin/python -m chocofarm.az.bench.bench_mlp_lowlatency \\
        --out ~/w/vdc/chocobo/bench/mlp_lowlatency/results.json

`--batches`, `--in-dim`, `--hidden`, `--n-actions`, `--iters`, `--repeat`, `--warmup` tune the sweep
(defaults are the production geometry). The JSON dump carries the raw per-(variant, batch) median+IQR us,
the fitted (intercept, slope, R²) per variant, and the consecutive-delta decomposition, so a re-run is
mechanical.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Callable, cast

import numpy as np
import numpy.typing as npt

# Importing the dispatcher applies the XLA/OMP single-thread pin (it side-effect-imports config), so the
# bench runs on the SAME single-threaded Eigen backend the inference server uses — the regime where
# dispatch dominates the per-leaf matmul (config.py's rationale). The production forward GRAPH is imported
# read-only (forward_core) so this bench runs the real net, never a re-transcription (P1).
from chocofarm.az.forward import forward_core
from chocofarm.az.lowlatency import compile_lowlatency, run

# forward_core's OWN signature (params, X, xp) -> (v_std, logits|None) — the backend-polymorphic SSOT
# (forward.py carries no annotations: the documented `xp`-seam stub-gap). A typed alias so calling it in
# this --strict module is not a no-untyped-call — the SAME shape inference_server.py binds (P1: reuse the
# project's established adapter for the untyped forward, don't invent a second one).
ForwardCore = Callable[[dict[str, "npt.NDArray[Any]"], Any, Any], "tuple[Any, Any | None]"]
_FORWARD_CORE: ForwardCore = forward_core


# ---- the REAL forward, at the production [v | logits] contract ----------------------------------------
def _prod_forward(p: dict[str, Any], x: Any) -> Any:
    """The production forward graph as a pure `fn(params, x) -> (B, 1+n_actions)` the lowlatency handle
    can stage: run the ONE `forward.forward_core`, DE-STANDARDIZE the value on-device, and pack
    `[v | logits]` — byte-for-byte the graph `inference_server.jit_forward_core` jits (ADR-0012 P1/P6: the
    same `forward_core`, the same de-standardize, only the y-scale scalars are read off the params closure
    `p["_ym"]`/`p["_ys"]` instead of the production forward's two traced args, so the shape fits the
    handle's `fn(params, x)` contract while the numerics are identical). `p` is `dict[str, Any]` (the
    mixed weights+scalars closure forward_core consumes positionally — the same honest `Any` params seam
    the toy bench and forward_core ride; P8). A value-only net (no `Wp`) returns `(B, 1)`; the production
    net carries the policy head so it returns `(B, 1+n_actions)`. The jax output is the documented
    backend-`Any` seam, cast to the declared return (P8: a cast documents the assertion in-source, the
    same one lowlatency.run uses, not a silencing ignore)."""
    import jax.numpy as jnp
    v_std, logits = _FORWARD_CORE(p, x, jnp)
    v = jnp.reshape(v_std, (-1, 1)) * p["_ys"] + p["_ym"]      # de-standardize ON-device → (B, 1)
    return cast(Any, v if logits is None else jnp.concatenate([v, logits], axis=1))


def _prod_params(in_dim: int, hidden: int, n_actions: int, seed: int = 0,
                 y_mean: float = 1.5, y_std: float = 3.0) -> dict[str, Any]:
    """The production-shaped FLOAT32 params dict (residual OFF), keyed exactly like `ValueMLP._params()`
    plus the two folded y-scale scalars `_ym`/`_ys`. Typed `dict[str, Any]` because it holds a MIX of
    ndarray weights and the two `np.float32` y-scale scalars — the honest closure `forward_core` consumes
    positionally (P8: the same `Any` params seam forward_core itself carries, not a convenience). float32
    because the inference server casts the weights to f32 ONCE at load (`params_from_manifest_blob`) — the
    SSOT inference precision — so the staged handle and the `current` baseline both carry f32 weights,
    matching the server. He-init draws (`np.sqrt(2/fan_in)`) so the matmul magnitudes are realistic (the
    value/policy heads ×0.1, as `ValueMLP.__init__`); the exact draw is immaterial to the timing (dispatch
    + transfer + the same-shape matmul), only the shapes/dtypes are."""
    rng = np.random.default_rng(seed)

    def he(a: int, b: int, s: float = 1.0) -> npt.NDArray[np.float32]:
        # the cast states the f32 contract — numpy's standard_normal/`*`-broadcast return Any, so the
        # `.astype(np.float32)` result is Any under the stubs (same seam mlp._he_init casts; P8).
        return cast("npt.NDArray[np.float32]",
                    (rng.standard_normal((a, b)) * np.sqrt(2.0 / a) * s).astype(np.float32))

    return {
        "W1": he(in_dim, hidden), "b1": np.zeros(hidden, np.float32),
        "W2": he(hidden, hidden), "b2": np.zeros(hidden, np.float32),
        "Wv": he(hidden, 1, 0.1), "bv": np.zeros(1, np.float32),
        "Wp": he(hidden, n_actions, 0.1), "bp": np.zeros(n_actions, np.float32),
        "_ym": np.float32(y_mean), "_ys": np.float32(y_std),
    }


# ---- timing + fit helpers (identical methodology to the toy bench) ------------------------------------
def _median_iqr_us(fn: Callable[[], Any], iters: int, repeat: int) -> tuple[float, float]:
    """Median (and inter-quartile range) over `repeat` runs of the per-call wall time (microseconds) of
    `fn` executed `iters` times. The MEDIAN is the reported center (best-of robustness to a scheduler
    blip — ADR-0009 median+IQR), the IQR (q75 − q25) the spread reported alongside it as the variance."""
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
    """Least-squares fit `time = intercept + slope · rows`. Returns (intercept_us, slope_us_per_row, R²).
    The INTERCEPT is the fixed per-call dispatch cost (the premise's target — the quantity the consecutive
    deltas decompose); the SLOPE is the per-row compute. R² reports how linear the relationship is (a poor
    fit means the model — and the intercept/slope read-off — should be distrusted)."""
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(times_us, dtype=np.float64)
    A = np.vstack([np.ones_like(x), x]).T          # columns: [1, rows] -> coeffs [intercept, slope]
    (intercept, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = intercept + slope * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(intercept), float(slope), r2


# The four decomposition variants, in the consecutive-delta order the decomposition reads them.
VARIANTS = ["current", "staged_params", "staged_params_input", "fully_device"]


def bench(batches: list[int], in_dim: int, hidden: int, n_actions: int,
          iters: int, repeat: int, warmup: int) -> dict[str, Any]:
    """Run the sweep: for each variant and each batch size, time `iters` calls `repeat` times and record
    the median+IQR per-call us; then fit `time = intercept + slope·rows` per variant; then derive the
    consecutive-intercept-delta decomposition. Returns the full record. The four variants are VERIFIED to
    compute the same forward (allclose, ABS_TOL=1e-4) at every batch before timing — an apples-to-apples
    gate, ADR-0009's forward-equivalence bar."""
    import jax
    import jax.numpy as jnp

    params = _prod_params(in_dim, hidden, n_actions)
    jfn = jax.jit(_prod_forward)   # the `current` baseline jit (host params + host x, cast folded in)

    # Per-batch handle + inputs. A fresh host matrix per batch (the server's reality); a device copy for
    # the staged-input / fully-device variants. ONE robust lowlatency handle per batch (it stages exactly
    # one (shape, dtype) signature, with the params placed device-resident at construction).
    per_batch: dict[int, dict[str, Any]] = {}
    for B in batches:
        x_host = np.random.default_rng(B + in_dim).standard_normal((B, in_dim)).astype(np.float32)
        x_dev = jax.device_put(jnp.asarray(x_host))
        handle = compile_lowlatency(_prod_forward, params, x_host)   # robust: params staged device-resident
        per_batch[B] = {"x_host": x_host, "x_dev": x_dev, "handle": handle}

    # APPLES-TO-APPLES VERIFICATION (ADR-0009 forward bar): every variant must compute the SAME (B, 1+NA)
    # block — `current` (host params), `staged_params`/`staged_params_input`/`fully_device` (staged handle,
    # host vs device x). A drift here means the decomposition compares different forwards; fail loud
    # (ADR-0002) before any timing rather than report an invalid comparison.
    ABS_TOL = 1e-4
    for B in batches:
        d = per_batch[B]
        ref = np.asarray(jfn(params, d["x_host"]), dtype=np.float32)             # current
        o_sp = np.asarray(run(d["handle"], d["x_host"]), dtype=np.float32)       # staged_params
        o_si = np.asarray(run(d["handle"], d["x_dev"]), dtype=np.float32)        # staged_params_input
        o_fd = np.asarray(run(d["handle"], d["x_dev"]).block_until_ready(), dtype=np.float32)  # fully_device
        if ref.shape != (B, 1 + n_actions):
            raise ValueError(f"forward at B={B} returned {ref.shape}, expected {(B, 1 + n_actions)} "
                             f"(the production [v|logits] block) — geometry mismatch, ADR-0002")
        for name, o in (("staged_params", o_sp), ("staged_params_input", o_si), ("fully_device", o_fd)):
            if not np.allclose(ref, o, atol=ABS_TOL):
                md = float(np.max(np.abs(ref - o)))
                raise ValueError(
                    f"variant {name} disagrees with `current` at B={B}: max|Δ|={md:.3e} > {ABS_TOL} — the "
                    f"variants must compute the same forward for the decomposition to be apples-to-apples "
                    f"(ADR-0009 forward-equivalence bar; ADR-0002 fail-loud).")

    # The four timed closures per batch. `current` pulls the result to host AND passes HOST params (the
    # server's true cost — the params re-transfer every call). `staged_params` uses the staged handle but
    # host x + host pull. `staged_params_input` feeds device x. `fully_device` blocks on the device array
    # (no host pull). (`run(handle, x)` is the robust call; the params are the handle's staged closure.)
    def closures(B: int) -> dict[str, Callable[[], Any]]:
        d = per_batch[B]
        xh, xd, h = d["x_host"], d["x_dev"], d["handle"]
        return {
            "current":             lambda: np.asarray(jfn(params, xh)),
            "staged_params":       lambda: np.asarray(run(h, xh)),
            "staged_params_input": lambda: np.asarray(run(h, xd)),
            "fully_device":        lambda: run(h, xd).block_until_ready(),
        }

    # WARM UP every (variant, batch) so the one-time AOT compile + first-call XLA cost is excluded from the
    # timed region (ADR-0009 measure-honesty — the cold compile is the confound the server's warmup() also
    # removes).
    for B in batches:
        cl = closures(B)
        for name in VARIANTS:
            for _ in range(warmup):
                cl[name]()

    # Timed sweep: median+IQR per (variant, batch).
    results: dict[str, dict[int, dict[str, float]]] = {name: {} for name in VARIANTS}
    for B in batches:
        cl = closures(B)
        for name in VARIANTS:
            med, iqr = _median_iqr_us(cl[name], iters, repeat)
            results[name][B] = {"median_us": med, "iqr_us": iqr}

    # Linear fit per variant: intercept (fixed dispatch cost) + slope (per-row compute), with R².
    fits: dict[str, dict[str, float]] = {}
    for name in VARIANTS:
        rows = list(batches)
        med_us = [results[name][B]["median_us"] for B in batches]
        intercept, slope, r2 = _fit_line(rows, med_us)
        fits[name] = {"intercept_us": intercept, "slope_us_per_row": slope, "r2": r2}

    # The consecutive-intercept-delta DECOMPOSITION: each component is the drop in fixed cost from removing
    # one transfer. params = current − staged_params; input = staged_params − staged_params_input;
    # output = staged_params_input − fully_device; dispatch_floor = fully_device's residual intercept.
    ic = {name: fits[name]["intercept_us"] for name in VARIANTS}
    decomposition = {
        "params_transfer_us": ic["current"] - ic["staged_params"],
        "input_transfer_us": ic["staged_params"] - ic["staged_params_input"],
        "output_pull_us": ic["staged_params_input"] - ic["fully_device"],
        "dispatch_floor_us": ic["fully_device"],
        "current_intercept_us": ic["current"],
        "consolidatable_transfer_us": ic["current"] - ic["fully_device"],   # params+input+output
    }

    return {
        "config": {"batches": batches, "in_dim": in_dim, "hidden": hidden, "n_actions": n_actions,
                   "iters": iters, "repeat": repeat, "warmup": warmup, "abs_tol": ABS_TOL,
                   "xla_flags": os.environ.get("XLA_FLAGS"),
                   "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                   "jax_version": jax.__version__},
        "per_batch_us": {name: {str(B): results[name][B] for B in batches} for name in VARIANTS},
        "fits": fits,
        "decomposition": decomposition,
    }


def _report(rec: dict[str, Any]) -> str:
    """The decomposition report: the per-variant intercept/slope/R² table and the consecutive-delta
    transfer breakdown, with the verdict on whether the params-staging win replicates and how large the
    consolidatable transfer is versus the irreducible dispatch floor."""
    fits = rec["fits"]
    dec = rec["decomposition"]
    lines: list[str] = []
    lines.append("LINEAR FIT  time = intercept + slope * rows   (intercept = fixed dispatch cost, slope = per-row compute)")
    lines.append(f"  {'variant':<22} {'intercept us':>14} {'slope us/row':>14} {'R^2':>8}")
    for name in VARIANTS:
        f = fits[name]
        lines.append(f"  {name:<22} {f['intercept_us']:14.2f} {f['slope_us_per_row']:14.4f} {f['r2']:8.4f}")
    lines.append("")
    lines.append("DECOMPOSITION of the `current` per-call fixed cost (consecutive intercept deltas):")
    lines.append(f"  params transfer  (current -> staged_params)        : {dec['params_transfer_us']:8.2f} us")
    lines.append(f"  input  transfer  (staged_params -> +input)         : {dec['input_transfer_us']:8.2f} us")
    lines.append(f"  output pull      (+input -> fully_device)          : {dec['output_pull_us']:8.2f} us")
    lines.append(f"  dispatch floor   (fully_device residual intercept) : {dec['dispatch_floor_us']:8.2f} us")
    lines.append(f"  ----------------------------------------------------")
    lines.append(f"  current intercept (sum)                            : {dec['current_intercept_us']:8.2f} us")
    lines.append(f"  consolidatable transfer (params+input+output)      : {dec['consolidatable_transfer_us']:8.2f} us")
    lines.append("")
    params_drop = dec["params_transfer_us"]
    floor = dec["dispatch_floor_us"]
    cur = dec["current_intercept_us"]
    lines.append("VERDICT:")
    lines.append(f"  params-staging win on the REAL MLP: {params_drop:+.2f} us "
                 f"({'REPLICATES (lower)' if params_drop > 0 else 'does NOT replicate'}) "
                 f"— {100.0 * params_drop / cur:.1f}% of the current intercept.")
    lines.append(f"  consolidatable host<->device transfer: {dec['consolidatable_transfer_us']:.2f} us "
                 f"({100.0 * dec['consolidatable_transfer_us'] / cur:.1f}% of current) vs irreducible "
                 f"dispatch floor {floor:.2f} us ({100.0 * floor / cur:.1f}%).")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="REAL-MLP dispatch-decomposition bench for the low-overhead JAX dispatcher.")
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--in-dim", type=int, default=241)        # the production feature dim (features.py)
    ap.add_argument("--hidden", type=int, default=256)        # the production ValueMLP hidden
    ap.add_argument("--n-actions", type=int, default=65)      # 20 collect + 44 sense + 1 terminate (mlp.py)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--out", type=str, default=None, help="JSON dump path (preserve under ~/w/vdc, not /tmp)")
    args = ap.parse_args()

    rec = bench(args.batches, args.in_dim, args.hidden, args.n_actions,
                args.iters, args.repeat, args.warmup)

    cfg = rec["config"]
    print(f"REAL MLP forward  in_dim={cfg['in_dim']} hidden={cfg['hidden']} n_actions={cfg['n_actions']} "
          f"(value+policy, f32, residual OFF)")
    print(f"  iters={cfg['iters']} repeat={cfg['repeat']} warmup={cfg['warmup']}  "
          f"jax={cfg['jax_version']}  XLA_FLAGS={cfg['xla_flags']}  OMP_NUM_THREADS={cfg['omp_num_threads']}")
    print()
    hdr = "  batch  " + "".join(f"{p:>22}" for p in VARIANTS)
    print(hdr)
    for B in args.batches:
        row = f"  {B:5d}  "
        for p in VARIANTS:
            cell = rec["per_batch_us"][p][str(B)]
            row += f"{cell['median_us']:12.2f}+-{cell['iqr_us']:<7.2f} "
        print(row)
    print()
    print(_report(rec))

    if args.out:
        out = os.path.expanduser(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
