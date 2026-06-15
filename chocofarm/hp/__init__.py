#!/usr/bin/env python3
"""
chocofarm — the centralized, live, redis-backed hyperparameter registry.

The implementation of `docs/design/hyperparameter-registry.md`. Three pieces:

  * `schema.py`  — the typed contract: a hierarchy of `@dataclass` groups
                   (`ExperimentConfig` over the ten per-axis groups), each field
                   carrying its `Mut` facet (HOT / RESTART / INSTANCE) and codec,
                   plus the strict, fail-loud (de)serialization (no default-coercion).
  * `registry.py`— the redis store (one JSON blob per experiment at
                   `choco:hp:<id>` + `:meta`, no TTL, namespaced), the
                   read-at-point-of-use snapshot with RESTART-refusal, the
                   seed-from-argparse bootstrap, and the operator CLI (get/set/init).

See the design doc for the rationale; this package implements it as written.
Public Domain (The Unlicense).
"""
from __future__ import annotations

from chocofarm.hp.schema import (
    Mut,
    hp,
    EnvConfig,
    SearchConfig,
    ValueTargetConfig,
    FeatureConfig,
    ArchConfig,
    TrainConfig,
    ExItLoopConfig,
    EvalConfig,
    ParallelConfig,
    BoundsConfig,
    ExperimentConfig,
    SCHEMA_VERSION,
    RegistryDecodeError,
    encode_config,
    decode_config,
)

__all__ = [
    "Mut",
    "hp",
    "EnvConfig",
    "SearchConfig",
    "ValueTargetConfig",
    "FeatureConfig",
    "ArchConfig",
    "TrainConfig",
    "ExItLoopConfig",
    "EvalConfig",
    "ParallelConfig",
    "BoundsConfig",
    "ExperimentConfig",
    "SCHEMA_VERSION",
    "RegistryDecodeError",
    "encode_config",
    "decode_config",
]
