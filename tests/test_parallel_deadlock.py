#!/usr/bin/env python3
"""
test_parallel_deadlock.py — bounded gate for the parallel-loop deadlock fix
(fix/jaxtrain-deadlock; see docs/notes/jaxtrain-deadlock-rca.md).

These assert the LOAD-BEARING property of the fix without spinning up a real Pool, redis, or the
multi-hour AZ loop: the parent's fan-out drain must never wait unbounded — a worker that fails to
report must surface as a LOUD, diagnosable RuntimeError (ADR-0002), never a silent permanent park.

Covered:
  * `_drain_imap` drains a normal iterator to exhaustion, preserving order/values.
  * `_drain_imap` converts a `multiprocessing.TimeoutError` (the "worker never reported" signal)
    into a RuntimeError whose message names the phase, the run, and the collected/expected count.
  * `_connect` builds the redis client WITH bounded socket timeouts (no construction-time connect
    needed — we only assert the timeouts are wired, not that redis is up).

Run pinned + bounded, e.g.:
    taskset -c 3 timeout 60 /home/bork/w/vdc/venvs/generic/bin/python -m pytest \
        tests/test_parallel_deadlock.py -q
"""
import os
import sys
import multiprocessing

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chocofarm.az import parallel as P


class _FakeImap:
    """Stand-in for a Pool `IMapIterator`: yields `items` via `.next(timeout)` and, optionally,
    raises `multiprocessing.TimeoutError` at index `raise_timeout_at` (the "worker never reported"
    condition the real drain must survive). Exhaustion raises StopIteration, like the real one."""

    def __init__(self, items, raise_timeout_at=None):
        self.items = list(items)
        self.i = 0
        self.rt = raise_timeout_at

    def next(self, timeout=None):
        if self.rt is not None and self.i == self.rt:
            raise multiprocessing.TimeoutError()
        if self.i >= len(self.items):
            raise StopIteration
        v = self.items[self.i]
        self.i += 1
        return v


def test_drain_imap_clean_exhaustion_preserves_results():
    got = P._drain_imap(_FakeImap([1, 2, 3]), 3, "generate", "run0")
    assert got == [1, 2, 3]


def test_drain_imap_empty():
    assert P._drain_imap(_FakeImap([]), 0, "evaluate", "run0") == []


def test_drain_imap_timeout_raises_loud_runtimeerror_with_progress():
    # 5 expected, worker stalls after the 1st result -> loud RuntimeError naming progress/phase/run
    with pytest.raises(RuntimeError) as ei:
        P._drain_imap(_FakeImap([10, 20], raise_timeout_at=1), 5, "evaluate", "runX")
    msg = str(ei.value)
    assert "evaluate" in msg          # phase named
    assert "runX" in msg              # run named
    assert "1/5" in msg               # collected/expected progress
    # the original TimeoutError is chained (debuggability), not swallowed
    assert isinstance(ei.value.__cause__, multiprocessing.TimeoutError)


def test_drain_imap_timeout_on_first_result():
    with pytest.raises(RuntimeError) as ei:
        P._drain_imap(_FakeImap([], raise_timeout_at=0), 3, "generate", "runY")
    assert "0/3" in str(ei.value)


def test_connect_sets_bounded_socket_timeouts(monkeypatch):
    """`_connect` must wire socket_timeout + socket_connect_timeout (the H2 fix). We intercept the
    `redis.Redis` constructor so no live redis is needed — and assert the timeout kwargs are passed
    and finite (not None, which is block-forever)."""
    captured = {}

    class _FakeRedis:
        def __init__(self, **kw):
            captured.update(kw)

        def ping(self):
            return True

    import redis
    monkeypatch.setattr(redis, "Redis", _FakeRedis)
    P._connect()
    assert captured.get("socket_timeout") is not None
    assert captured.get("socket_connect_timeout") is not None
    assert float(captured["socket_timeout"]) > 0
    assert float(captured["socket_connect_timeout"]) > 0


def test_connect_socket_timeout_env_overridable(monkeypatch):
    captured = {}

    class _FakeRedis:
        def __init__(self, **kw):
            captured.update(kw)

        def ping(self):
            return True

    import redis
    monkeypatch.setattr(redis, "Redis", _FakeRedis)
    monkeypatch.setenv("CHOCO_REDIS_SOCKET_TIMEOUT", "12.5")
    P._connect()
    assert float(captured["socket_timeout"]) == 12.5
