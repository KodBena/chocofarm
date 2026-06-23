#!/usr/bin/env python3
"""
tools/zmq-wire-bench/run-sweep.py — drive the isolated ZMQ wire benchmark with ROBUST statistics.

Sweeps message width B and producer-thread count P; for each (B,P) cell it runs R replicates in INTERLEAVED
(shuffled) order with a warmup pass discarded, so run-to-run drift does not confound the cell effect. Reports
per-cell MEDIAN + IQR (RTTs are right-skewed; the median is robust) and a linear regression of median RTT vs B
per P with a bootstrap 95% CI on the slope/intercept. The point: contrast the raw wire RTT against the
leaf-eval lab's per-forward gap (905-1864 us, step-4) to show the wire is a small fraction (the rest is the
producer's search-wait).

Public Domain (The Unlicense).
"""
import os
import random
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCER = os.path.join(HERE, "producer")
CONSUMER = os.path.join(HERE, "consumer.py")
ENDPOINT = "ipc:///tmp/zmqwirebench.ipc"
IN_DIM, OUT_DIM = 241, 66            # production request / reply widths
B_GRID = [32, 64, 128, 256, 512]
P_GRID = [1, 2, 3]
R = 8                                # replicates per cell
T = 2.0                              # seconds per run
SEED = 20260623

rng = random.Random(SEED)


def run_cell(B, P):
    out = subprocess.run([PRODUCER, ENDPOINT, str(B), str(IN_DIM), str(OUT_DIM), str(P), str(T), "0"],
                         capture_output=True, text=True, timeout=T + 30)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT"):
            kv = dict(tok.split("=") for tok in line.split()[1:])
            return {"throughput": float(kv["throughput_msgs_s"]),
                    "median_rtt": float(kv["median_rtt_us"]),
                    "mean_rtt": float(kv["mean_rtt_us"]),
                    "p90_rtt": float(kv["p90_rtt_us"])}
    raise RuntimeError(f"no RESULT line: stdout={out.stdout!r} stderr={out.stderr!r}")


def main():
    cons = subprocess.Popen([sys.executable, CONSUMER, ENDPOINT, str(IN_DIM), str(OUT_DIM)])
    time.sleep(1.0)                                  # let the ROUTER bind
    cells = [(B, P) for B in B_GRID for P in P_GRID]
    try:
        for c in cells:                              # warmup pass (discarded)
            run_cell(*c)
        data = {c: [] for c in cells}
        for _ in range(R):                           # R interleaved replicates
            order = cells[:]
            rng.shuffle(order)
            for c in order:
                data[c].append(run_cell(*c))
    finally:
        cons.send_signal(signal.SIGTERM)
        cons.wait()

    med = lambda xs: float(np.median(xs))
    iqr = lambda xs: float(np.percentile(xs, 75) - np.percentile(xs, 25))
    print(f"\n=== per-cell robust summary (R={R} interleaved replicates, T={T}s, usleep=0 saturated) ===")
    print(f"{'B':>5} {'P':>2} {'med_rtt_us':>11} {'IQR_us':>7} {'med_thru_msgs_s':>16} {'med_p90_us':>11}")
    for P in P_GRID:
        for B in B_GRID:
            rs = data[(B, P)]
            mr = [d["median_rtt"] for d in rs]
            th = [d["throughput"] for d in rs]
            p9 = [d["p90_rtt"] for d in rs]
            print(f"{B:>5} {P:>2} {med(mr):>11.1f} {iqr(mr):>7.1f} {med(th):>16.0f} {med(p9):>11.1f}")

    print("\n=== regression  median_rtt_us ~ a + b*B  (per P; OLS over all replicate readings; bootstrap 95% CI) ===")
    for P in P_GRID:
        Bs, Rt = [], []
        for B in B_GRID:
            for d in data[(B, P)]:
                Bs.append(B)
                Rt.append(d["median_rtt"])
        Bs, Rt = np.array(Bs, float), np.array(Rt, float)
        b, a = np.polyfit(Bs, Rt, 1)                 # slope, intercept
        pred = a + b * Bs
        ss_res = float(np.sum((Rt - pred) ** 2))
        ss_tot = float(np.sum((Rt - Rt.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        ba, bb = [], []
        n = len(Bs)
        for _ in range(3000):                        # bootstrap over the readings
            idx = [rng.randrange(n) for _ in range(n)]
            s, i = np.polyfit(Bs[idx], Rt[idx], 1)
            bb.append(s)
            ba.append(i)
        ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        cia, cib = ci(ba), ci(bb)
        print(f"  P={P}: intercept a={a:7.1f} us  CI[{cia[0]:.1f},{cia[1]:.1f}]   "
              f"slope b={b:.4f} us/row CI[{cib[0]:.4f},{cib[1]:.4f}]   R^2={r2:.3f}")

    print("\n=== contrast with the lab's step-4 per-forward GAP (the wire vs producer-wait split) ===")
    lab = {128: 905.0, 256: 1864.0}                  # lab gap us at B~128 (N=2), B~256 (N=4)
    for B, gap in lab.items():
        for P in P_GRID:
            w = med([d["median_rtt"] for d in data[(B, P)]])
            print(f"  B={B} P={P}: wire RTT(median)={w:6.1f}us  vs lab gap={gap:.0f}us  -> wire is {100*w/gap:4.1f}% "
                  f"(producer-wait ~ the remaining {100*(1-w/gap):4.1f}%)")
    print("\nNB: echo consumer = PURE ZMQ wire (a LOWER bound on the lab's real drain/decode/scatter); even so,"
          " it is a small fraction of the gap -> the gap is dominated by the producer's search-wait, not the wire.")


if __name__ == "__main__":
    main()
