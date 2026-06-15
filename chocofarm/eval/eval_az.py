#!/usr/bin/env python3
"""
chocofarm AZ — E-DECIDE Stage-2 eval (design §9): does the LEARNED-VALUE leaf beat the
determinized-PLAYOUT leaf at MATCHED budget?

Mirrors `eval_ismcts.py`. Measures `NetValueISMCTS` (learned value at the leaf) against, at the
SAME iteration budget:
  - `ISMCTSPolicy` with the determinized-playout leaf (the F4 baseline),
  - the static floor 0.0855 and clairvoyant ceiling 0.1454 (reference lines, design F1),
  - the decomp policy 0.094 (the value teacher; the strong known anchor).

Two rate readings per searched policy (design §9 step 4):
  - the policy's own unbiased Dinkelbach fixed point (`env.dinkelbach_rate`),
  - the rate at FIXED λ₀=0.0855 (`env.rate` at λ₀) — the operating point the value was trained
    at, and the apples-to-apples row.

Also reports E[T] (mean episode time) — the over-collection signature: the GO read-out (design
§9) wants the learned leaf to BEAT the playout leaf with SHORTER E[T] (less over-collection).

CLI: python -m chocofarm.eval.eval_az --weights w.npz [--it 200] [--n 300] [--seed 7]
Pin to core 3 under timeout. For the full run use N≥300 (design's <2% SE rule).

Public Domain (The Unlicense).
"""
import argparse
import time
from typing import Any

import numpy as np

from chocofarm.model.env import Environment
from chocofarm.references import BeliefRefs
from chocofarm.solvers.base import Policy
from chocofarm.solvers.ismcts import ISMCTSPolicy
from chocofarm.solvers.decomp import DecompPolicy
from chocofarm.az.netvalue_ismcts import NetValueISMCTS
from chocofarm.eval.harness import DECOMP_ANCHOR, dink_float
from chocofarm.eval.report import references, print_reference_header

LAM0 = 0.0855  # static-floor rate; the fixed training/operating λ (design §4.1)


def measure(env: Environment, pol: Policy, refs: BeliefRefs, n: int, seed: int,
            dink_iters: int = 2, warm: int = 10) -> dict[str, Any]:
    """Returns (dinkelbach-rate row, fixed-λ₀ row) for `pol`, both with E[T] and %VoI."""
    # the policy's own Dinkelbach fixed point
    res = env.dinkelbach_rate(pol, iters=dink_iters, warm_runs=warm, final_runs=n, seed=seed)
    # the rate at fixed λ₀ (the operating point the value was trained at)
    r0, ER0, ET0, exits0 = env.rate(pol, LAM0, n, seed=seed)
    return {
        "dink_rate": dink_float(res, "rate"), "dink_lam": dink_float(res, "lambda"),
        "dink_ET": dink_float(res, "ET"),
        "dink_voi": refs.voi_pct(dink_float(res, "rate")),
        "fix_rate": r0, "fix_ET": ET0, "fix_ER": ER0,
        "fix_voi": refs.voi_pct(r0), "fix_exits": exits0,
    }


def stream_compare(env: Environment, net_pol: Policy, base_pol: Policy | None, refs: BeliefRefs,
                   n: int, chunk: int, seed: int,
                   logdir: str) -> dict[str, list[float]]:
    """Interleaved paired comparison at FIXED λ₀, streaming cumulative rate curves to TB.

    Net-value and playout-leaf are run on the SAME seed each chunk (paired → variance reduction),
    accumulating sumR/sumT so the logged rate is the cumulative fixed-λ₀ rate that TIGHTENS with
    episodes — the same cumulative-rate shape the solver runners stream. Reference lines
    (floor/ceiling/decomp) are logged alongside. A timeout mid-run still leaves usable curves.
    Returns the accumulators for the final read-out."""
    from tensorboardX import SummaryWriter
    static, ceil = refs.static_floor, refs.clairvoyant_ceiling   # the reference lines for TB/display
    w = SummaryWriter(logdir)
    acc: dict[str, list[float]] = {"net": [0.0, 0.0, 0], "playout": [0.0, 0.0, 0]}  # sumR, sumT, eps
    pols: dict[str, Policy | None] = {"net": net_pol, "playout": base_pol}
    done, s = 0, seed
    while done < n:
        c = min(chunk, n - done)
        for tag in ("net", "playout"):
            pol = pols[tag]
            if pol is None:
                continue
            _, ER, ET, _ = env.rate(pol, LAM0, c, seed=s)
            a = acc[tag]
            a[0] += ER * c; a[1] += ET * c; a[2] += c
            cumrate = a[0] / a[1] if a[1] > 0 else 0.0
            w.add_scalar(f"rate/{tag}", cumrate, a[2])
            w.add_scalar(f"E_time/{tag}", a[1] / a[2], a[2])
            w.add_scalar(f"voi_pct/{tag}", refs.voi_pct(cumrate), a[2])
        w.add_scalar("ref/floor", static, done + c)
        w.add_scalar("ref/ceiling", ceil, done + c)
        w.add_scalar("ref/decomp", DECOMP_ANCHOR, done + c)
        w.flush()
        s += 1; done += c
        rn = acc["net"]
        msg = f"  [{done}/{n}] net rate={rn[0]/rn[1] if rn[1] else 0:.4f} ET={rn[1]/rn[2]:.1f}"
        if base_pol is not None:
            rb = acc["playout"]
            msg += f" | playout rate={rb[0]/rb[1] if rb[1] else 0:.4f} ET={rb[1]/rb[2]:.1f}"
        print(msg, flush=True)
    w.close()

    rn = acc["net"]; net_rate, net_ET = rn[0] / rn[1], rn[1] / rn[2]
    print(f"\nFINAL fixed-λ₀={LAM0}  net rate={net_rate:.4f}  ET={net_ET:.1f}  "
          f"%VoI={refs.voi_pct(net_rate):.0f}", flush=True)
    if base_pol is not None:
        rb = acc["playout"]; b_rate, b_ET = rb[0] / rb[1], rb[1] / rb[2]
        print(f"            playout rate={b_rate:.4f}  ET={b_ET:.1f}  "
              f"%VoI={refs.voi_pct(b_rate):.0f}", flush=True)
        print(f"READ-OUT: net − playout Δrate={net_rate - b_rate:+.4f}, ΔET={net_ET - b_ET:+.1f} "
              f"({'shorter→less' if net_ET < b_ET else 'longer→more'} over-collection)", flush=True)
        print(f"  GO iff net beats playout (ideally clears floor {static:.4f}) with shorter ET.",
              flush=True)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description="E-DECIDE Stage-2: learned-value vs playout leaf.")
    ap.add_argument("--weights", type=str, required=True, help="trained value-net npz")
    ap.add_argument("--it", type=int, default=200, help="ISMCTS iteration budget (matched)")
    ap.add_argument("--n", type=int, default=300, help="final-eval episodes (N≥300 for <2% SE)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the playout-leaf ISMCTS baseline (faster smoke)")
    ap.add_argument("--tb-logdir", type=str, default=None,
                    help="if set, stream cumulative fixed-λ₀ rate curves (net + playout) to TB")
    ap.add_argument("--chunk", type=int, default=20, help="episodes per streamed TB point")
    args = ap.parse_args()

    env = Environment()
    refs = references(env)
    static = refs.static_floor                              # the floor, for the GO read-out display
    print_reference_header(
        refs, extra_lines=(f"decomp anchor       : {DECOMP_ANCHOR} (the value teacher)",))

    if args.tb_logdir:
        net_pol = NetValueISMCTS(env, args.weights, iterations=args.it)
        base_pol = None if args.no_baseline else ISMCTSPolicy(iterations=args.it)
        stream_compare(env, net_pol, base_pol, refs, args.n, args.chunk,
                       args.seed, args.tb_logdir)
        return

    hdr = (f"{'policy':>22} {'dink_rate':>9} {'dink_ET':>7} {'fixλ_rate':>9} "
           f"{'fixλ_ET':>7} {'fixλ_%VoI':>9} {'sec':>5}")
    print(hdr, flush=True)

    def row(name: str, m: dict[str, Any], sec: float) -> None:
        print(f"{name:>22} {m['dink_rate']:>9.4f} {m['dink_ET']:>7.1f} "
              f"{m['fix_rate']:>9.4f} {m['fix_ET']:>7.1f} {m['fix_voi']:>8.0f}% {sec:>5.0f}",
              flush=True)

    # 1) the probe: learned-value leaf
    net_pol = NetValueISMCTS(env, args.weights, iterations=args.it)
    t0 = time.time()
    m_net = measure(env, net_pol, refs, args.n, args.seed)
    row(f"net-value (it={args.it})", m_net, time.time() - t0)

    # 2) the F4 baseline: determinized-playout leaf at the SAME budget
    if not args.no_baseline:
        base_pol = ISMCTSPolicy(iterations=args.it)
        t0 = time.time()
        m_base = measure(env, base_pol, refs, args.n, args.seed)
        row(f"playout-leaf (it={args.it})", m_base, time.time() - t0)

        # the GO/NO-GO read-out (design §9)
        d_fix = m_net["fix_rate"] - m_base["fix_rate"]
        d_et = m_net["fix_ET"] - m_base["fix_ET"]
        print(f"\nREAD-OUT (fixed λ₀={LAM0}): net − playout rate Δ = {d_fix:+.4f}; "
              f"E[T] Δ = {d_et:+.1f} ({'shorter' if d_et < 0 else 'longer'} → "
              f"{'less' if d_et < 0 else 'more'} over-collection)", flush=True)
        print(f"  GO iff net leaf significantly beats playout leaf (ideally clears floor "
              f"{static:.4f}) with shorter E[T].", flush=True)


if __name__ == "__main__":
    main()
