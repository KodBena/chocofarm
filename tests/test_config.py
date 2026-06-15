#!/usr/bin/env python3
"""
test_config.py — the two-instance redis configuration surface (chocofarm/config.py).

chocofarm runs TWO redis instances by design, one per role, and `config.py` is the single owner of
"which redis" for each: the TRANSPORT role (the AZ parallel-loop weight/result blobs — transport.py)
addresses the ephemeral allkeys-lru 6380 instance; the REGISTRY role (the hp config blobs —
registry.py) addresses the disk-persisted noeviction 6379 instance. These two roles MUST stay
distinct, and their env-var families MUST be independent (an override on one role must not bleed into
the other) — that independence is the property this test pins.

These tests touch NO redis (they only read the param dicts and the env contract), so they run in any
environment. Run, e.g.:
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_config.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chocofarm import config


def test_transport_defaults_to_6380():
    """The transport role addresses the ephemeral memory-cache instance at 127.0.0.1:6380 db 0."""
    p = config.transport_redis_params()
    assert p == {"host": "127.0.0.1", "port": 6380, "db": 0}


def test_registry_defaults_to_6379():
    """The registry role addresses the disk-persisted noeviction instance at 127.0.0.1:6379 db 0."""
    p = config.registry_redis_params()
    assert p == {"host": "127.0.0.1", "port": 6379, "db": 0}


def test_roles_address_distinct_ports_by_default():
    """The whole point of the two-instance design: the two roles are NOT the same instance."""
    assert config.transport_redis_params()["port"] != config.registry_redis_params()["port"]


def test_transport_port_override_does_not_bleed_into_registry(monkeypatch):
    """A `CHOCO_TRANSPORT_REDIS_PORT` override moves the transport ONLY — the registry default holds."""
    monkeypatch.setenv("CHOCO_TRANSPORT_REDIS_PORT", "7000")
    assert config.transport_redis_params()["port"] == 7000
    assert config.registry_redis_params()["port"] == 6379  # untouched


def test_registry_port_override_does_not_bleed_into_transport(monkeypatch):
    """A `CHOCO_REGISTRY_REDIS_PORT` override moves the registry ONLY — the transport default holds."""
    monkeypatch.setenv("CHOCO_REGISTRY_REDIS_PORT", "7001")
    assert config.registry_redis_params()["port"] == 7001
    assert config.transport_redis_params()["port"] == 6380  # untouched


def test_both_overrides_are_independent(monkeypatch):
    """Both families set at once: each role lands on its OWN override, no cross-contamination."""
    monkeypatch.setenv("CHOCO_TRANSPORT_REDIS_HOST", "10.0.0.1")
    monkeypatch.setenv("CHOCO_TRANSPORT_REDIS_PORT", "7000")
    monkeypatch.setenv("CHOCO_TRANSPORT_REDIS_DB", "3")
    monkeypatch.setenv("CHOCO_REGISTRY_REDIS_HOST", "10.0.0.2")
    monkeypatch.setenv("CHOCO_REGISTRY_REDIS_PORT", "7001")
    monkeypatch.setenv("CHOCO_REGISTRY_REDIS_DB", "4")
    assert config.transport_redis_params() == {"host": "10.0.0.1", "port": 7000, "db": 3}
    assert config.registry_redis_params() == {"host": "10.0.0.2", "port": 7001, "db": 4}


def test_timeouts_are_shared_across_roles(monkeypatch):
    """The socket/connect timeouts are NOT role-specific — one env contract drives both roles."""
    monkeypatch.setenv("CHOCO_REDIS_SOCKET_TIMEOUT", "12.5")
    monkeypatch.setenv("CHOCO_REDIS_CONNECT_TIMEOUT", "3.5")
    assert config.redis_socket_timeout() == 12.5
    assert config.redis_connect_timeout() == 3.5


def test_old_role_agnostic_redis_params_is_gone():
    """The removed single `redis_params()` is the SSOT hazard the two-role split eliminates — it must
    not be re-introduced (a single 'which redis' function is ambiguous by construction)."""
    assert not hasattr(config, "redis_params")
