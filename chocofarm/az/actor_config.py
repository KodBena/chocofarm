#!/usr/bin/env python3
"""
chocofarm/az/actor_config.py — ActorConfig: the C++ Gumbel actor's control config, projected from the
hp-registry ExperimentConfig (ADR-0012 P1/P7 — derive-don't-duplicate across the language boundary).

This is the SSOT for the config knobs the C++ actor consumes: the instance/faces geometry paths and the
7 GumbelConfig search knobs the runner reads as --gumbel-* flags today. It is a DERIVED VIEW of the hp
schema — `from_experiment_config` reads the (already strict-decoded, domain-validated) SearchConfig and
holds NO defaults of its own (the defaults live in schema.py, the one config SSOT). The C++ mirror
`cpp/include/chocofarm/actor_config.hpp` declares the SAME field set + per-field Mut class as parseable
literal arrays, drift-checked against this module in tests/test_wire_drift.py (ADR-0012 P7: the
cross-boundary fact has one authoritative home; every side derives, the drift net is the backstop) —
so a field added/removed/renamed, or a Mut-class flip, on one side reds the default suite rather than
silently desyncing the control config. This module's P1/P7 honesty is CONTINGENT on that drift leg
existing and working (the same status `result_spec` has).

Scope (P2 honesty — only carry what the receiver consumes). ActorConfig carries the knobs the C++
runner ACTUALLY reads. It excludes:
  * use_jax_mlp — a Python-side knob (it selects the JAX vs numpy forward for the in-process eval); the
    C++ runner builds its own NetForward and has no --use-jax-mlp flag, so it never crosses this seam.
  * explore_plies / td_lambda (lam_blend) / n_step — the parity knobs the C++ search cannot honor yet
    (the executor refuses them loudly); they join ActorConfig when the parity thread makes the C++
    search consume them. Putting a field the receiver ignores into the config is the lying signature
    ADR-0002/P2 forbid.

The per-generation scalars (version, seed, lam, episodes, max_steps, res_token) are NOT config — they
ride each generate request (the determinism anchor lives in the per-call message, never sticky config).

Mut classes (read from schema.py's metadata["mut"], the one home — see _SCHEMA_SOURCE): the 7 search
knobs are HOT (the SH bracket is recomputed per decide — they flow live through reconfigure); the two
geometry paths are INSTANCE (built once; a live change is a NEW experiment, a loud reject).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from chocofarm.hp.schema import EnvConfig, ExperimentConfig, Mut, SearchConfig


@dataclass(frozen=True)
class ActorConfig:
    """The C++ Gumbel actor's control config — instance/faces geometry + the 7 GumbelConfig search
    knobs. A frozen, typed projection of ExperimentConfig (P8: the typed signature is the contract).
    Built via `from_experiment_config`; serialized to the control `configure` message in the transport.
    The fields are FLAT (matching the JSON wire keys); the C++ mirror nests the 7 search knobs in a
    `GumbelConfig` for the runner's use, but the wire field set is flat — see actor_config.hpp."""

    instance_path: str
    faces_path: str
    m: int
    n_sims: int
    c_puct: float
    c_visit: float
    c_scale: float
    c_outcome: int
    max_depth: int


# The field set + each field's home in the hp schema (group, field) — the ONE place ActorConfig's fields
# map back to the SSOT. The Mut class is READ from that schema field's metadata (never declared here), so
# the Mut classification has a single home (the hp schema). instance_path/faces_path both map to
# env.instance_path: faces is the DERIVED geometry of the same instance, so it shares instance's INSTANCE
# class (there is no separate schema field for it — the C++ runner takes the faces path directly).
_SCHEMA_SOURCE: tuple[tuple[str, str, str], ...] = (
    # (ActorConfig field, schema group, schema field)
    ("instance_path", "env", "instance_path"),
    ("faces_path", "env", "instance_path"),
    ("m", "search", "m"),
    ("n_sims", "search", "n_sims"),
    ("c_puct", "search", "c_puct"),
    ("c_visit", "search", "c_visit"),
    ("c_scale", "search", "c_scale"),
    ("c_outcome", "search", "c_outcome"),
    ("max_depth", "search", "max_depth"),
)

# the ordered field names (the drift net compares this against the C++ ACTOR_CONFIG_FIELDS literal).
FIELD_NAMES: tuple[str, ...] = tuple(f for f, _g, _sf in _SCHEMA_SOURCE)

# fail loud at import if the dataclass and the schema-source map disagree (a field added to one and not
# the other is a silent desync the projection must never start under — ADR-0002).
assert FIELD_NAMES == tuple(f.name for f in dataclasses.fields(ActorConfig)), (
    "ActorConfig dataclass fields and _SCHEMA_SOURCE disagree — "
    f"{tuple(f.name for f in dataclasses.fields(ActorConfig))!r} vs {FIELD_NAMES!r}")


_GROUPS: dict[str, type] = {"env": EnvConfig, "search": SearchConfig}


def _mut_of(group: str, field: str) -> Mut:
    """Read the Mut class of one schema (group, field) from its dataclass metadata — the one home. A
    field whose metadata carries no Mut is a loud failure (the schema contract: every leaf has a Mut)."""
    cls = _GROUPS[group]
    for f in dataclasses.fields(cls):
        if f.name == field:
            mut = f.metadata.get("mut")
            assert isinstance(mut, Mut), f"schema {group}.{field} has no Mut facet"
            return mut
    raise KeyError(f"{group}.{field} is not a field of {cls.__name__}")


# the per-field Mut class string ("hot"/"restart"/"instance"), in FIELD_NAMES order — the drift net
# compares this against the C++ ACTOR_CONFIG_MUT literal, so a Mut-class flip on either side reds. The
# strings are exactly the schema Mut enum values (no third vocabulary).
MUT_CLASSES: tuple[str, ...] = tuple(_mut_of(g, sf).value for _f, g, sf in _SCHEMA_SOURCE)


def from_experiment_config(cfg: ExperimentConfig, *, instance_path: str,
                           faces_path: str) -> ActorConfig:
    """Project a (strict-decoded, domain-validated) ExperimentConfig into the actor's ActorConfig.

    Reads the 7 search knobs off `cfg.search` (the hp SSOT — no defaults re-declared here) and takes the
    two geometry paths explicitly (they are the C++ runner's instance/faces, supplied by the executor's
    construction, not an hp schema field — there is no schema home for the C++ faces path). The values
    are already validated by `schema.check_invariants` at decode; this projection adds no new defaults
    and no second validation (Port/ACL: derive, don't re-author)."""
    s = cfg.search
    return ActorConfig(
        instance_path=instance_path,
        faces_path=faces_path,
        m=s.m,
        n_sims=s.n_sims,
        c_puct=s.c_puct,
        c_visit=s.c_visit,
        c_scale=s.c_scale,
        c_outcome=s.c_outcome,
        max_depth=s.max_depth,
    )
