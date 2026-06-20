#!/usr/bin/env python3
"""
test_method_token_bucket.py — unit gate for the per-thread row-metered token-bucket controller
(cpp/stage_a/control_lab/methods/token_bucket.py), a STATIC candidate for the issue-gate control lab.

Imports the method's OWN submodule directly (NOT the methods package, and WITHOUT load_all()): sibling
method files are authored in parallel, so importing only `control_lab.methods.token_bucket` keeps this test
isolated from a half-written neighbour. Pins the FROZEN adapter.Controller contract (reset / observe / act /
metrics shape) plus the bucket's defining behavior:

  - the FIRST decision of a trial is all-allow (no dt to meter yet — the AllAllow baseline);
  - act() returns a length-T list of values in {0,1};
  - leaves is CUMULATIVE and ROW-metered: a thread that offers enough rows (leaves increment / s_min) to
    drain its bucket below one token, with no refill (clock held fixed), is DENIED — and it spends only the
    INCREMENT, not the absolute cumulative level (the first-difference);
  - the liveness override holds: a drained thread with inflight==0 still force-allows (a deny is a no-op
    there);
  - a thread that never issues stays allowed (its bucket never drains).

Run pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_token_bucket.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import sys

# cpp/stage_a on sys.path so `control_lab.*` resolves both under pytest and as a bare script (mirrors the
# maintainer's PYTHONPATH=cpp/stage_a run convention for the lab).
_STAGE_A = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp", "stage_a"
)
if _STAGE_A not in sys.path:
    sys.path.insert(0, _STAGE_A)

from control_lab.adapter import Observation, TrialContext  # noqa: E402
from control_lab.methods import token_bucket as tb  # noqa: E402  (own submodule, NOT the package)


def _ctx(n_threads: int = 4, s_min: int = 2, k: int = 8) -> TrialContext:
    return TrialContext(
        n_threads=n_threads, d_ceiling=3, k_per_thread=k, s_min=s_min,
        chunk_floor=True, seed=0,
    )


def _obs(
    *, n_threads: int, inflight: list[int], leaves: list[int], served: list[int], t: float
) -> Observation:
    """Minimal synthetic Observation with the length-T vectors the bucket reads (inflight, leaves) and the
    served set the metering keys off; other feature slots default-safe in act()."""
    features = {
        "n_threads": n_threads,
        "d_ceiling": 3,
        "server_rows_per_forward": float(sum(leaves)),
        "inflight": inflight,
        "ready": [0] * n_threads,
        "msgs": [0] * n_threads,
        "leaves": leaves,
        "rtt_us": [0] * n_threads,
    }
    return Observation(features=features, served=served, forward_rows=sum(leaves), t_monotonic=t)


def test_reset_and_metrics_shape():
    """reset() sizes per-run state to T and metrics() exposes the dashboard scalars (rho, mean_token)."""
    c = tb.TokenBucketGate(rho=8.0, c_burst=16.0)
    c.reset(_ctx(n_threads=4))
    m = c.metrics()
    assert m["rho"] == 8.0
    assert "mean_token" in m
    # a fresh trial starts every bucket at the burst cap.
    assert m["mean_token"] == 16.0


def test_act_returns_length_t_binary():
    """act() returns a length-T list whose every entry is 0 or 1 (the per-thread allow bits)."""
    T = 4
    c = tb.TokenBucketGate()
    c.reset(_ctx(n_threads=T))
    out = c.act(_obs(n_threads=T, inflight=[1] * T, leaves=[0] * T, served=list(range(T)), t=0.0))
    assert isinstance(out, list)
    assert len(out) == T
    assert all(v in (0, 1) for v in out)


def test_observe_is_safe_noop():
    """observe() is a no-op for the static bucket: it must not raise and must not perturb the gate."""
    T = 3
    c = tb.TokenBucketGate()
    c.reset(_ctx(n_threads=T))
    c.observe(123.4, {"forward_rows": 5})
    out = c.act(_obs(n_threads=T, inflight=[1] * T, leaves=[0] * T, served=[0, 1, 2], t=0.0))
    assert out == [1, 1, 1]  # first decision -> all-allow, undisturbed by observe()


def test_first_decision_is_all_allow():
    """The first decision of a trial has no dt to meter against, so it allows every thread (the baseline)."""
    T = 4
    c = tb.TokenBucketGate(rho=8.0, c_burst=16.0)
    c.reset(_ctx(n_threads=T))
    out = c.act(_obs(n_threads=T, inflight=[2] * T, leaves=[1000] * T, served=list(range(T)), t=5.0))
    assert out == [1, 1, 1, 1]


def test_row_metered_denial_and_liveness_and_first_difference():
    """The defining behavior. With the clock HELD FIXED (no refill), a thread that offers enough rows to
    drain its bucket below one token is DENIED; a quiet thread stays allowed; a drained thread with
    inflight==0 force-allows (liveness); and consumption is the leaves INCREMENT, not the absolute level."""
    T = 4
    # c_burst = 16, s_min = 2  ->  draining one bucket needs > 16 tokens = > 32 rows of leaves increment.
    c = tb.TokenBucketGate(rho=10.0, c_burst=16.0)
    c.reset(_ctx(n_threads=T, s_min=2))

    # Decision 1 (t=0): seed the per-thread leaves baselines. Thread 0/1/2/3 start already at nonzero
    # cumulative leaves so we can prove the bucket first-differences (spends the INCREMENT, not the level).
    base = [100, 100, 100, 100]
    out0 = c.act(_obs(n_threads=T, inflight=[2] * T, leaves=base, served=[0, 1, 2, 3], t=0.0))
    assert out0 == [1, 1, 1, 1]  # first decision -> all-allow regardless of the (large) absolute leaves

    # Decision 2 (t=0 still -> dt=0, no refill): thread 0 offers +40 rows (20 chunks, drains 16 fully and
    # goes negative) -> DENY. thread 1 offers +2 rows (1 chunk) -> still has ~15 tokens -> ALLOW. thread 2
    # offers the SAME huge absolute leaves but only +0 increment -> ALLOW (proves first-difference, not
    # level). thread 3 offers +40 rows AND has inflight==0 -> force-ALLOW (liveness override beats the empty
    # bucket).
    out1 = c.act(
        _obs(
            n_threads=T,
            inflight=[2, 2, 2, 0],
            leaves=[base[0] + 40, base[1] + 2, base[2] + 0, base[3] + 40],
            served=[0, 1, 2, 3],
            t=0.0,
        )
    )
    assert out1[0] == 0, "thread 0 drained its bucket via row-metered consumption -> deny"
    assert out1[1] == 1, "thread 1 spent only one chunk -> still above one token -> allow"
    assert out1[2] == 1, "thread 2 had zero leaves increment (despite large level) -> allow (first-diff)"
    assert out1[3] == 1, "thread 3 inflight==0 -> liveness override force-allows despite the drained bucket"

    # mean_token recorded for the dashboard; thread 0 went negative, so the mean is below the burst cap.
    assert c.metrics()["mean_token"] < 16.0


if __name__ == "__main__":
    # plain-runnable (no pytest needed), mirroring the repo's bare-script test convention.
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
    print("all token_bucket method checks passed")
