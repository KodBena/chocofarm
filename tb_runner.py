#!/usr/bin/env python3
"""
Long-running, core-pinnable, memory-capped evaluator that streams a policy's converging
rate to TensorBoard. Bounded BY CONSTRUCTION: it holds only running sums + a writer (no
per-episode retention; the policies discard their search trees per decision), and it
self-guards on RSS — if its resident set exceeds --rss_cap_mb it logs and exits. Pin it to a
core with `taskset -c <n>` at launch.

Per config it logs the cumulative running rate (totR/totT) vs episodes, so the estimate
TIGHTENS indefinitely; multiple configs (e.g. ISMCTS iteration budgets) each get their own
curve, so the budget→rate relationship emerges. ref/static and ref/ceiling are the floor and
the clairvoyant ceiling for reference.
"""
import sys
import os
import time
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from env import Environment
from run import realizable_static, clairvoyant_rate
from nmcs import NMCSPolicy
from ismcts import ISMCTSPolicy
from tensorboardX import SummaryWriter


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def make_policy(method, cfg):
    if method == "nmcs":
        return NMCSPolicy(level=int(cfg))
    if method == "ismcts":
        return ISMCTSPolicy(iterations=int(cfg))
    raise SystemExit("unknown method " + method)


def label(method, cfg):
    return ("L%d" % int(cfg)) if method == "nmcs" else ("it%d" % int(cfg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)         # nmcs | ismcts
    ap.add_argument("--configs", required=True)         # comma list (levels or iteration budgets)
    ap.add_argument("--tag", required=True)             # logdir subtag
    ap.add_argument("--logroot", default="/home/bork/w/vdc/chocobo/tb")
    ap.add_argument("--batch", type=int, default=1)     # episodes per logged step (1 = snappy)
    ap.add_argument("--rss_cap_mb", type=float, default=1200.0)
    a = ap.parse_args()

    env = Environment()
    static = realizable_static(env)
    ceil = clairvoyant_rate(env)
    configs = [c.strip() for c in a.configs.split(",") if c.strip()]
    pols = {c: make_policy(a.method, c) for c in configs}
    writer = SummaryWriter(os.path.join(a.logroot, a.tag))
    state = {c: dict(sumR=0.0, sumT=0.0, n=0, lam=0.0) for c in configs}

    for c in configs:                                   # no warm-up (it cost minutes of silence);
        state[c]["lam"] = 0.08                           # init lambda near observed rates, nudge in-loop

    seed, total = 1000, 0
    while True:
        if rss_mb() > a.rss_cap_mb:
            writer.add_scalar("diag/aborted_rss_mb", rss_mb(), total)
            writer.flush()
            break
        for c in configs:
            s = state[c]
            t_ep = time.time()
            _, ER, ET, _ = env.rate(pols[c], s["lam"], a.batch, seed=seed)
            sec_ep = (time.time() - t_ep) / a.batch
            seed += 1
            s["sumR"] += ER * a.batch
            s["sumT"] += ET * a.batch
            s["n"] += a.batch
            total += a.batch
            run_rate = s["sumR"] / s["sumT"] if s["sumT"] > 0 else 0.0
            s["lam"] = 0.7 * s["lam"] + 0.3 * run_rate          # gentle Dinkelbach nudge
            lab = label(a.method, c)
            writer.add_scalar(f"rate/{lab}", run_rate, s["n"])
            writer.add_scalar(f"pct_ceiling/{lab}", run_rate / ceil, s["n"])
            writer.add_scalar(f"voi_clawed/{lab}", (run_rate - static) / (ceil - static), s["n"])
            writer.add_scalar(f"E_reward/{lab}", s["sumR"] / s["n"], s["n"])
            writer.add_scalar(f"E_time/{lab}", s["sumT"] / s["n"], s["n"])
            writer.add_scalar(f"lambda/{lab}", s["lam"], s["n"])
            writer.add_scalar(f"ref/static_{lab}", static, s["n"])
            writer.add_scalar(f"ref/ceiling_{lab}", ceil, s["n"])
            writer.add_scalar(f"sec_per_episode/{lab}", sec_ep, s["n"])
            writer.add_scalar("diag/rss_mb", rss_mb(), total)
            writer.flush()


if __name__ == "__main__":
    main()
