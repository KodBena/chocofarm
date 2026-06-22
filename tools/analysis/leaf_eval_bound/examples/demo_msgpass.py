"""
tools/analysis/leaf_eval_bound/examples/demo_msgpass.py
=================================================

The synthetic message-passing throughput demo extracted out of `neyman_driver.py`
per the ADR-0012 purification (P1 single-home / P2 separation of the allocator from
the thing allocated). It is ONE example model the generic `NeymanDriver` consumes —
the driver owns no model; this module owns this model. It exists to exercise the
allocator on a deliberately heavy-tailed input (the messy context-switch cost) so the
report visibly funds the high-variance primitive.

Run: `python tools/analysis/leaf_eval_bound/examples/demo_msgpass.py` (requires jax + scipy).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Allow running as a bare script (examples/ is a sibling of neyman_driver.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neyman_driver import NeymanDriver  # noqa: E402


def build_demo() -> tuple[NeymanDriver, dict[int, object], list[str], list[float]]:
    """Return a configured `NeymanDriver` + the per-input samplers/names/costs for the
    toy 'crude throughput' model:

        throughput = 1e6 / (msg_lat_us + ctx_switch_us + 1e6/infer_tput + serialize_us)

    Three clean primitives plus one messy, heavy-tailed context-switch cost; the
    allocation should spend its budget overwhelmingly on the messy one.
    """
    rng = np.random.default_rng(0)

    def s_msg_lat(k):       # clean, tight
        return rng.normal(8.0, 0.5, k)

    def s_ctx_switch(k):    # messy: lognormal body + rare large spikes
        body = rng.lognormal(mean=np.log(3.0), sigma=0.6, size=k)
        spikes = (rng.random(k) < 0.03) * rng.normal(40.0, 8.0, k).clip(0)
        return body + spikes

    def s_infer_tput(k):    # clean-ish (msgs/sec), moderate spread
        return rng.normal(250_000.0, 6_000.0, k)

    def s_serialize(k):     # clean, tiny
        return rng.normal(1.2, 0.1, k)

    samplers = {0: s_msg_lat, 1: s_ctx_switch, 2: s_infer_tput, 3: s_serialize}
    names = ["msg_lat_us", "ctx_switch_us", "infer_tput", "serialize_us"]

    def f(x):  # JAX-traceable; x ordered by `names` (msg_lat, ctx, infer_tput, serialize)
        msg_lat, ctx, infer_tput, serialize = x
        return 1e6 / (msg_lat + ctx + 1e6 / infer_tput + serialize)
    costs = [1.0, 25.0, 2.0, 0.5]   # per-sample bench cost; the ctx-switch bench is dear

    driver = NeymanDriver(
        f, costs=costs, tolerance=2.0,   # want E[throughput] to +/- 2 msgs/sec
        names=names, confidence=0.95, growth_cap=3.0, max_batch=200_000,
    )
    return driver, samplers, names, costs


def main() -> None:
    driver, samplers, names, costs = build_demo()
    driver.run(samplers, pilot=512, max_rounds=20, verbose=True)
    print("Final pools:", {names[i]: len(driver.pools[i]) for i in range(len(names))})
    print("Total benchmark cost:",
          sum(len(driver.pools[i]) * costs[i] for i in range(len(names))))


if __name__ == "__main__":
    main()
