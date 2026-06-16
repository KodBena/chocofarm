// cpp/include/chocofarm/control_spec.hpp
// Purpose: the C++ MIRROR of the actor CONTROL PROTOCOL message contract (the messages the persistent
//   Gumbel actor and its Python ActorTransport client exchange). The ONE authoritative declaration is
//   chocofarm/az/control_spec.py (ADR-0012 P1/P7: a cross-boundary fact has one home; every side
//   DERIVES its view). This header mirrors the message TYPE tags + the closed ERROR tag set as parseable
//   constexpr literal arrays, DRIFT-CHECKED against the Python SSOT in tests/test_wire_drift.py — because
//   a tag the client BRANCHES on that silently drifts from the runner's spelling would mis-handle
//   WITHOUT a loud parse error. The envelope/reply KEY names are fail-loud-at-parse (a missing/renamed
//   JSON key is a `missing_field` reject, not a silent reshape), so they are SSOT constants here + in
//   control_spec.py with the round-trip integration test as the backstop (the proportionate surface for
//   a loud-failure contract — the failure mode sets the bar, ADR-0011).
//
//   ── DERIVED FROM chocofarm/az/control_spec.py — DO NOT EDIT EITHER SIDE WITHOUT THE OTHER. ──
//
//   This contract is transport-AGNOSTIC (P7: serialization ⊥ transport): the SAME messages travel over a
//   subprocess pipe, a unix socket, or a ZeroMQ daemon. The CONFIG payload of a `configure` is the
//   ActorConfig (its field set is drift-netted in actor_config.hpp); this header owns the envelope.
//
// Public Domain (The Unlicense).
#pragma once

#include <array>
#include <string_view>

namespace chocofarm::control {

// ── message type tags (the "type" field of every request) — drift-netted vocabulary ─────────────────
inline constexpr std::string_view MSG_CONFIGURE = "configure";
inline constexpr std::string_view MSG_GENERATE = "generate";
inline constexpr std::string_view MSG_PING = "ping";
inline constexpr std::string_view MSG_SHUTDOWN = "shutdown";

// the canonical ordered request-type set (mirrors control_spec.MSG_TYPES; drift-checked as text).
inline constexpr std::array<std::string_view, 4> CONTROL_MSG_TYPES = {
    "configure", "generate", "ping", "shutdown"};

// ── envelope / reply field names (the JSON keys; fail-loud-at-parse, SSOT-mirrored not array-drifted) ─
// request keys
inline constexpr std::string_view KEY_TYPE = "type";
inline constexpr std::string_view KEY_CONFIG = "config";
inline constexpr std::string_view KEY_CONFIG_EPOCH = "config_epoch";
inline constexpr std::string_view KEY_VERSION = "version";
inline constexpr std::string_view KEY_SEED = "seed";
inline constexpr std::string_view KEY_LAM = "lam";
inline constexpr std::string_view KEY_EPISODES = "episodes";
inline constexpr std::string_view KEY_MAX_STEPS = "max_steps";
inline constexpr std::string_view KEY_RES_TOKEN = "res_token";
// reply keys
inline constexpr std::string_view KEY_OK = "ok";
inline constexpr std::string_view KEY_WRITTEN = "written";
inline constexpr std::string_view KEY_SERVING = "serving";
inline constexpr std::string_view KEY_ERROR = "error";
inline constexpr std::string_view KEY_DETAIL = "detail";

// ── the closed error-tag set (a reply's "error" field; the client branches on these) — drift-netted ──
inline constexpr std::string_view ERR_BAD_JSON = "bad_json";
inline constexpr std::string_view ERR_UNKNOWN_TYPE = "unknown_type";
inline constexpr std::string_view ERR_MISSING_FIELD = "missing_field";
inline constexpr std::string_view ERR_INVALID_CONFIG = "invalid_config";
inline constexpr std::string_view ERR_INSTANCE_KNOB = "instance_knob_changed";
inline constexpr std::string_view ERR_EPOCH_MISMATCH = "config_epoch_mismatch";
inline constexpr std::string_view ERR_NOT_CONFIGURED = "not_configured";
inline constexpr std::string_view ERR_WEIGHT_READ = "weight_read_failed";
inline constexpr std::string_view ERR_GENERATE_FAILED = "generate_failed";

// the canonical ordered error-tag set (mirrors control_spec.ERROR_TAGS; drift-checked as text). No
// `restart_knob_changed` — ActorConfig has no RESTART field, so INSTANCE is the only not-live class.
inline constexpr std::array<std::string_view, 9> CONTROL_ERROR_TAGS = {
    "bad_json", "unknown_type", "missing_field", "invalid_config",
    "instance_knob_changed", "config_epoch_mismatch", "not_configured",
    "weight_read_failed", "generate_failed"};

}  // namespace chocofarm::control
