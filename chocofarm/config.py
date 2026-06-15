#!/usr/bin/env python3
"""
chocofarm/config.py — infrastructure configuration surface.

One place for the runtime connection facts the codebase needs. Today that is the
single redis instance shared by the AZ transport (`chocofarm/az/parallel.py`) and
the hyperparameter registry (`chocofarm/hp/registry.py`): both call
`redis_params()` here instead of re-reading `os.environ` themselves, so "which
redis" is decided once rather than drifting across modules.

The default is **127.0.0.1:6379 db 0** — the DISK-PERSISTED redis (RDB `save`
enabled, `maxmemory-policy noeviction`, no `maxmemory` cap). Registry blobs there
survive a restart and are never evicted, so no eviction workaround is needed (the
6380 memory-cache instance, an `allkeys-lru` store, was the reason the registry
once nudged `volatile-lru`; on 6379 that is moot). Each value is env-overridable
for an operator who wants a different instance.

Public Domain (Unlicense).
"""
from __future__ import annotations

import os

# Canonical defaults — the disk-persisted redis. Env vars override at runtime.
DEFAULT_REDIS_HOST = "127.0.0.1"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0
DEFAULT_REDIS_SOCKET_TIMEOUT = 60.0
DEFAULT_REDIS_CONNECT_TIMEOUT = 10.0


def redis_params() -> dict:
    """Connection facts (host/port/db) for the shared redis, env-overridable. The transport and the
    registry both call this so they address one instance by default."""
    return dict(
        host=os.environ.get("CHOCO_REDIS_HOST", DEFAULT_REDIS_HOST),
        port=int(os.environ.get("CHOCO_REDIS_PORT", str(DEFAULT_REDIS_PORT))),
        db=int(os.environ.get("CHOCO_REDIS_DB", str(DEFAULT_REDIS_DB))),
    )


def redis_socket_timeout() -> float:
    """Per-op socket timeout (ADR-0002: a stall becomes a loud error, not a silent hang)."""
    return float(os.environ.get("CHOCO_REDIS_SOCKET_TIMEOUT", str(DEFAULT_REDIS_SOCKET_TIMEOUT)))


def redis_connect_timeout() -> float:
    """Connection-establish timeout for the redis client."""
    return float(os.environ.get("CHOCO_REDIS_CONNECT_TIMEOUT", str(DEFAULT_REDIS_CONNECT_TIMEOUT)))
