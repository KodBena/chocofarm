#!/usr/bin/env python3
"""
chocofarm/az/control_spec.py — the SSOT for the C++ actor CONTROL PROTOCOL message contract: the message
type tags, the request/reply envelope field names, and the closed error-tag set the persistent Gumbel
actor and its Python ActorTransport client both speak (ADR-0012 P7/P1 — a cross-boundary fact has one
home; every side derives, never re-authors).

This is the transport-AGNOSTIC contract: the SAME messages travel whether the ActorTransport impl is a
subprocess pipe, a unix socket, or a ZeroMQ daemon (P7: separate the serialization contract from the
transport mechanism — the durable rule is mechanism-independent). The CONFIG payload of a `configure`
message is the ActorConfig (its field set + per-field Mut class are drift-netted in actor_config.py);
this module owns the ENVELOPE + the reply + the protocol vocabulary around it.

The protocol (lock-step, ONE in-flight request at a time — the synchronous loop's shape):

  configure  {type, config:{<ActorConfig>}}                 -> {ok, config_epoch}      | {ok:false, error, detail}
  generate   {type, config_epoch, version, seed, lam,       -> {ok, written,
              episodes, max_steps, res_token}                   config_epoch, version} | {ok:false, error, detail}
  ping       {type}                                         -> {ok, serving, config_epoch}
  shutdown   {type}                                         -> {ok}

Two independent gates:
  * config_epoch gates CONFIG ADOPTION. The runner increments an epoch each time it adopts a `configure`
    and returns it; a `generate` carries the epoch the client believes is live; a mismatch is a loud
    `config_epoch_mismatch` reject (the client learns the live epoch from the configure reply).
  * version gates WEIGHT RELOAD, independently. The runner reloads weights from redis when `version`
    advances, REGARDLESS of the epoch — the common case (new weights, unchanged config) is a
    new-version / same-epoch generate and MUST NOT be rejected.

Drift discipline. The C++ mirror cpp/include/chocofarm/control_spec.hpp declares the SAME message type
tags + error tags; tests/test_wire_drift.py drift-checks those two vocabularies, because a tag the client
BRANCHES on that silently drifts from the runner's spelling would mis-handle without a loud parse error.
The envelope/reply KEY names are fail-loud-at-parse on both sides (a missing/renamed JSON key is a
`missing_field` reject, not a silent reshape), so they are SSOT constants here + their C++ mirror, with
the round-trip integration test the backstop — the proportionate surface for a loud-failure contract
(ADR-0011: mechanize at the strongest feasible-and-proportionate level, the failure mode sets the bar).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Final

# ---- message type tags (the "type" field of every request) — the drift-netted protocol vocabulary ----
MSG_CONFIGURE: Final[str] = "configure"
MSG_GENERATE: Final[str] = "generate"
MSG_PING: Final[str] = "ping"
MSG_SHUTDOWN: Final[str] = "shutdown"

# the canonical ordered request-type set (drift-checked against the C++ mirror's CONTROL_MSG_TYPES).
MSG_TYPES: Final[tuple[str, ...]] = (MSG_CONFIGURE, MSG_GENERATE, MSG_PING, MSG_SHUTDOWN)

# ---- envelope / reply field names (the JSON keys both sides read/write; fail-loud-at-parse) ----
# request keys
KEY_TYPE: Final[str] = "type"
KEY_CONFIG: Final[str] = "config"
KEY_CONFIG_EPOCH: Final[str] = "config_epoch"
KEY_VERSION: Final[str] = "version"
KEY_SEED: Final[str] = "seed"
KEY_LAM: Final[str] = "lam"
KEY_EPISODES: Final[str] = "episodes"
KEY_MAX_STEPS: Final[str] = "max_steps"
KEY_RES_TOKEN: Final[str] = "res_token"
# reply keys
KEY_OK: Final[str] = "ok"
KEY_WRITTEN: Final[str] = "written"
KEY_SERVING: Final[str] = "serving"
KEY_ERROR: Final[str] = "error"
KEY_DETAIL: Final[str] = "detail"

# ---- the closed error-tag set (the machine tag a reply's "error" field carries; the client BRANCHES
#      on these, never on the human "detail" prose) — drift-netted, one home. There is no
#      `restart_knob_changed` tag: ActorConfig has NO RESTART field (use_jax_mlp, the only one, is
#      Python-side and not in ActorConfig — actor_config.py), so the only not-live class is INSTANCE. ----
ERR_BAD_JSON: Final[str] = "bad_json"                     # the request line was not valid JSON
ERR_UNKNOWN_TYPE: Final[str] = "unknown_type"             # "type" is not in MSG_TYPES
ERR_MISSING_FIELD: Final[str] = "missing_field"           # a required envelope/config field is absent/typed wrong
ERR_INVALID_CONFIG: Final[str] = "invalid_config"         # the config failed the schema/domain validation
ERR_INSTANCE_KNOB: Final[str] = "instance_knob_changed"   # instance/faces changed live (a NEW experiment)
ERR_EPOCH_MISMATCH: Final[str] = "config_epoch_mismatch"  # generate's epoch != the runner's current epoch
ERR_NOT_CONFIGURED: Final[str] = "not_configured"         # a generate before the first successful configure
ERR_WEIGHT_READ: Final[str] = "weight_read_failed"        # the redis weight payload was missing/unreadable
ERR_GENERATE_FAILED: Final[str] = "generate_failed"       # the episode run/write failed

# the canonical ordered error-tag set (drift-checked against the C++ mirror's CONTROL_ERROR_TAGS).
ERROR_TAGS: Final[tuple[str, ...]] = (
    ERR_BAD_JSON, ERR_UNKNOWN_TYPE, ERR_MISSING_FIELD, ERR_INVALID_CONFIG,
    ERR_INSTANCE_KNOB, ERR_EPOCH_MISMATCH, ERR_NOT_CONFIGURED, ERR_WEIGHT_READ,
    ERR_GENERATE_FAILED,
)
