#!/usr/bin/env python3
"""
chocofarm/config.py — infrastructure configuration surface.

One place for the runtime connection facts the codebase needs. chocofarm runs TWO redis instances by
design — one per role — and this module is the single owner of "which redis" for each, so the facts
are decided once here rather than drifting across modules:

  * TRANSPORT (the AZ parallel-loop weight broadcast + result blobs — `chocofarm/az/transport.py`,
    orchestrated by `chocofarm/az/parallel.py`) → **127.0.0.1:6380 db 0**, the EPHEMERAL memory-cache
    instance (`allkeys-lru`, a `maxmemory` cap). The transport's churn — versioned weight blobs and
    per-task result blobs — is short-lived; the keys carry 1h TTLs and are read+deleted within the
    iteration, and the LRU policy is the safety-net that evicts anything left behind. Nothing here
    needs to survive a restart.

  * REGISTRY (the hp config blobs — `chocofarm/hp/registry.py`) → **127.0.0.1:6379 db 0**, the
    DISK-PERSISTED instance (RDB `save` enabled, `maxmemory-policy noeviction`, no `maxmemory` cap).
    Registry blobs carry NO TTL: they must survive a restart and must never be evicted, so they live
    on the noeviction instance. The `allkeys-lru` eviction the 6380 transport instance applies would
    silently drop a registry blob — which is exactly why the two roles are deliberately distinct
    instances, not one shared store.

Each role's connection facts are env-overridable independently, via DISTINCT env-var families
(`CHOCO_TRANSPORT_REDIS_*` vs `CHOCO_REGISTRY_REDIS_*`) so an operator can point one role at a
different instance without touching the other. The socket/connect timeouts are NOT role-specific (a
stall is a stall on either instance), so they stay shared across both roles under one env contract.

Public Domain (Unlicense).
"""
from __future__ import annotations

import os

# --- inference / XLA process tuning (SSOT) — set at IMPORT so it lands BEFORE jax initializes ---
# XLA reads XLA_FLAGS (and its Eigen threadpool sizing) ONCE, at first jax use. The inference server's
# hot path imports jax.numpy directly and NOT mlp_jax, so the per-module XLA pins that used to live in
# mlp_jax.py / mlp_jax_train.py never reached the server's jax — leaving its forwards on a multi-threaded
# Eigen pool that spins / work-steals on the tiny per-leaf matmul (on a 4-vCPU host, single-threaded
# wins). Pinning it HERE — config is a leaf module imported before jax in every jax-using path — is the
# ONE home (ADR-0012 P1) and is what actually reaches the server. `setdefault` so an operator override wins.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# The fixed inference batch size — the JAX forward is padded to this ONE shape so XLA compiles a single
# executable, not one per drained B. DEFAULT only; the LIVE value rides --cpp-pool-batch (the one batch
# contract both the C++ runner's RuntimeConfig and the Python server's pad derive from — ADR-0012 P1/P7).
DEFAULT_INFERENCE_BATCH = 64

# Transport role — the EPHEMERAL memory-cache redis (allkeys-lru). Env vars override at runtime.
DEFAULT_TRANSPORT_REDIS_HOST = "127.0.0.1"
DEFAULT_TRANSPORT_REDIS_PORT = 6380
DEFAULT_TRANSPORT_REDIS_DB = 0

# Registry role — the DISK-PERSISTED redis (noeviction). Env vars override at runtime.
DEFAULT_REGISTRY_REDIS_HOST = "127.0.0.1"
DEFAULT_REGISTRY_REDIS_PORT = 6379
DEFAULT_REGISTRY_REDIS_DB = 0

# Shared across both roles — a stall is a stall on either instance, so the timeouts are not
# role-specific (one env contract for both).
DEFAULT_REDIS_SOCKET_TIMEOUT = 60.0
DEFAULT_REDIS_CONNECT_TIMEOUT = 10.0


def transport_redis_params() -> dict[str, str | int]:
    """Connection facts (host/port/db) for the TRANSPORT redis — the ephemeral allkeys-lru instance
    at 127.0.0.1:6380 db 0 that carries the AZ parallel-loop weight/result blobs. Env-overridable via
    `CHOCO_TRANSPORT_REDIS_HOST`/`CHOCO_TRANSPORT_REDIS_PORT`/`CHOCO_TRANSPORT_REDIS_DB`, independent
    of the registry's family."""
    return dict(
        host=os.environ.get("CHOCO_TRANSPORT_REDIS_HOST", DEFAULT_TRANSPORT_REDIS_HOST),
        port=int(os.environ.get("CHOCO_TRANSPORT_REDIS_PORT", str(DEFAULT_TRANSPORT_REDIS_PORT))),
        db=int(os.environ.get("CHOCO_TRANSPORT_REDIS_DB", str(DEFAULT_TRANSPORT_REDIS_DB))),
    )


def registry_redis_params() -> dict[str, str | int]:
    """Connection facts (host/port/db) for the REGISTRY redis — the disk-persisted noeviction instance
    at 127.0.0.1:6379 db 0 that holds the hp config blobs (no TTL, must survive a restart).
    Env-overridable via `CHOCO_REGISTRY_REDIS_HOST`/`CHOCO_REGISTRY_REDIS_PORT`/`CHOCO_REGISTRY_REDIS_DB`,
    independent of the transport's family."""
    return dict(
        host=os.environ.get("CHOCO_REGISTRY_REDIS_HOST", DEFAULT_REGISTRY_REDIS_HOST),
        port=int(os.environ.get("CHOCO_REGISTRY_REDIS_PORT", str(DEFAULT_REGISTRY_REDIS_PORT))),
        db=int(os.environ.get("CHOCO_REGISTRY_REDIS_DB", str(DEFAULT_REGISTRY_REDIS_DB))),
    )


def redis_socket_timeout() -> float:
    """Per-op socket timeout, shared across both roles (ADR-0002: a stall becomes a loud error, not a
    silent hang)."""
    return float(os.environ.get("CHOCO_REDIS_SOCKET_TIMEOUT", str(DEFAULT_REDIS_SOCKET_TIMEOUT)))


def redis_connect_timeout() -> float:
    """Connection-establish timeout for the redis client, shared across both roles."""
    return float(os.environ.get("CHOCO_REDIS_CONNECT_TIMEOUT", str(DEFAULT_REDIS_CONNECT_TIMEOUT)))
