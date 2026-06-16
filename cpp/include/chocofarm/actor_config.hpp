// cpp/include/chocofarm/actor_config.hpp
// Purpose: the C++ MIRROR of the actor control config (the knobs the persistent Gumbel actor
//   reconfigures live). The ONE authoritative declaration is chocofarm/az/actor_config.py
//   (ADR-0012 P1/P7: a cross-boundary fact has one home; every side DERIVES its view). This header
//   declares the SAME field set + per-field Mut class as parseable constexpr literal arrays,
//   DRIFT-CHECKED against the Python SSOT in the default Python suite (tests/test_wire_drift.py parses
//   ACTOR_CONFIG_FIELDS / ACTOR_CONFIG_MUT as TEXT and asserts equality with actor_config.FIELD_NAMES /
//   actor_config.MUT_CLASSES), so a field add/remove/rename, or a Mut-class flip, on one side reds the
//   default suite rather than silently desyncing the control config (ADR-0002 / ADR-0011 Rule 4: a net
//   over the field set + Mut class, not one field).
//
//   ── DERIVED FROM chocofarm/az/actor_config.py — DO NOT EDIT EITHER SIDE WITHOUT THE OTHER. ──
//
//   The config carries the geometry paths (instance/faces — INSTANCE: built once, a live change is a new
//   experiment) and the 7 GumbelConfig search knobs (m/n_sims/c_* — HOT: the SH bracket is recomputed
//   per decide, so they reconfigure live). It does NOT carry use_jax_mlp (a Python-side forward selector
//   the C++ runner does not consume) nor the parity knobs explore_plies/lam_blend/n_step (the C++ search
//   cannot honor them yet). The per-generation scalars (version/seed/lam/episodes/max_steps/res_token)
//   are NOT config — they ride each generate request.
//
//   The wire field set (ACTOR_CONFIG_FIELDS, the JSON keys) is FLAT; the struct nests the 7 search knobs
//   in a GumbelConfig so the runner consumes `cfg.gumbel` directly (P1 — reuse the one GumbelConfig
//   definition, do not re-transcribe the knob list). `from_json` (the Port/ACL: validate-don't-coerce)
//   lands with the persistent runner.
//
// Public Domain (The Unlicense).
#pragma once

#include <array>
#include <expected>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "chocofarm/error.hpp"   // Error — the boundary failure type from_json returns by value
#include "chocofarm/gumbel.hpp"  // GumbelConfig — the 7 search knobs ActorConfig carries (reused, P1)

namespace chocofarm {

// The actor control config: the geometry paths + the GumbelConfig search budget/knobs. A plain value
// struct (ADR-0012 P9 — typed fields, no raw pointers); built from the control message's JSON by the
// `from_json` factory (the Port/ACL: validate-don't-coerce), never hand-mutated field by field.
struct ActorConfig {
    std::string instance_path;  // INSTANCE — geometry source (a live change is a NEW experiment)
    std::string faces_path;     // INSTANCE — the DERIVED face cover alongside the instance
    GumbelConfig gumbel{};      // HOT — m / n_sims / c_puct / c_visit / c_scale / c_outcome / max_depth
};

// Parse a `configure` message's "config" object into a validated ActorConfig (the Port/ACL —
// validate-don't-coerce, ADR-0012 P2 / P9 rule 5). A missing/wrong-typed field, or an out-of-domain
// value (m<1, an empty path, ...), is a typed Error returned by value — never a throw, never a coerced
// default. The field set it reads is the one drift-netted against actor_config.py; the domain checks
// mirror the Python schema.check_invariants. Implemented in src/actor_config.cpp.
[[nodiscard]] std::expected<ActorConfig, Error> actor_config_from_json(const nlohmann::json& j);

// ── The drift-net manifest (parsed as TEXT by tests/test_wire_drift.py) ──────────────────────────────
// The FLAT wire field set (the JSON keys), in the SAME order as actor_config.FIELD_NAMES. A field
// added / removed / renamed on either side reds the field-set agreement leg.
inline constexpr std::array<std::string_view, 9> ACTOR_CONFIG_FIELDS = {
    "instance_path", "faces_path",
    "m", "n_sims", "c_puct", "c_visit", "c_scale", "c_outcome", "max_depth"};

// Each field's Mut class (the live-vs-reject classification), in field order, mirroring
// actor_config.MUT_CLASSES (which READS it from schema.py's metadata["mut"] — the one home). A field
// that changes its HOT/INSTANCE class on one side reds the Mut-class agreement leg — so "the geometry
// is INSTANCE, the search knobs are HOT" is a drift-protected fact, not a comment (it also tracks the
// m/n_sims RESTART->HOT classification: re-freezing them in the schema would red here).
inline constexpr std::array<std::string_view, 9> ACTOR_CONFIG_MUT = {
    "instance", "instance",
    "hot", "hot", "hot", "hot", "hot", "hot", "hot"};

static_assert(ACTOR_CONFIG_FIELDS.size() == ACTOR_CONFIG_MUT.size(),
              "ACTOR_CONFIG_FIELDS and ACTOR_CONFIG_MUT must be the same length (one Mut per field)");

}  // namespace chocofarm
