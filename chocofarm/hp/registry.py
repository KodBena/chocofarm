#!/usr/bin/env python3
"""
chocofarm hp — the redis-backed registry: store, live read path, bootstrap, and operator CLI.

Implements design §2 / §3 / §5 / §6. The schema (the typed contract) is `schema.py`; this module is
the thin redis layer over it (the §8 verdict: stdlib dataclasses + a hand-built redis layer, no
framework).

Store (design §2, §5)
---------------------
One JSON blob per experiment, addressed by an operator-meaningful `experiment_id`:

    choco:hp:<experiment_id>        -> the serialized ExperimentConfig blob (§5.2)
    choco:hp:<experiment_id>:meta   -> {schema_version, created_at, last_write_at, writer}

The single-blob-per-experiment choice (over key-per-field) is design §5.2: a read is one GET + one
strict decode (atomic + whole-config-validated by construction); a write is a read-modify-write of
the one small (<4 KB) blob under a WATCH/MULTI/EXEC optimistic guard (§5.4). Registry keys carry
NO TTL (a bare SET — design §2.1), so they never sit on a clock; on the disk-persisted redis
(127.0.0.1:6379, `noeviction` — see `chocofarm/config.py`) they survive a restart and are never
evicted, so the §2.2/§2.3 `volatile-lru` eviction workaround the 6380 memory-cache instance needed
is moot here.

Namespacing (design §5.1): distinct ids → disjoint key prefixes → concurrent experiments never
clobber, mirroring the transport's per-run token (parallel.py) under a human name.

Read path (design §3)
---------------------
`load_snapshot` GETs the blob and strict-decodes it to an `ExperimentConfig` (fail-loud on any
mismatch — never coerce to a default). A `ConfigSnapshot` wraps it with the `launched_with` shadow
(the construction-time argparse truth) so the RESTART-refusal (§3.4) can tell "you changed a baked
field" from "you changed a hot one." The loop refreshes the snapshot once per outer-iteration
boundary (≤1-iteration staleness, atomic within an iteration — §3.1/§3.3), reads HOT fields off it,
and lets `assert_no_restart_drift` fire the loud refusal if a RESTART/INSTANCE field moved.

Bootstrap (design §6 — the consolidation)
-----------------------------------------
`from_argparse(args, experiment_id)` builds an `ExperimentConfig` from a parsed argparse namespace
(dataclass defaults = argparse defaults, one source). `seed_registry` writes it as the experiment's
blob IF IT DOES NOT EXIST (idempotent — a --resume re-binds to the existing blob rather than
clobbering operator overrides). So the existing CLI invocation IS the seed; the operator writes
nothing by hand to start, and the same path is the post-restart recovery (§2.3).

Failure modes (design §7), each fail-loud
-----------------------------------------
  * redis down/unreachable          -> RegistryUnavailable (a failed ping/GET, never silent reuse)
  * key missing (unseeded / wiped)  -> RegistryKeyMissing  (run init / the launch seed; never default)
  * malformed / drifted blob        -> RegistryDecodeError (schema.py; never decode-into-wrong)
  * RESTART/INSTANCE changed mid-run-> RestartRequired      (§3.4 loud refusal, name + both values)
  * two writers racing              -> bounded WATCH retry, then RegistryWriteConflict (loud)

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import time
from dataclasses import fields

from chocofarm.hp.schema import (
    Mut,
    ExperimentConfig,
    EnvConfig, SearchConfig, ValueTargetConfig, FeatureConfig, ArchConfig,
    TrainConfig, ExItLoopConfig, EvalConfig, ParallelConfig, BoundsConfig,
    SCHEMA_VERSION,
    RegistryDecodeError,
    encode_config,
    decode_config,
    check_invariants,
)

# ---------------------------------------------------------------------------
# Key namespace (design §5.1) and connection facts (mirror parallel.py's env-driven params)
# ---------------------------------------------------------------------------
KEY_PREFIX = "choco:hp:"

# bounded write-retry count for the WATCH/MULTI/EXEC optimistic guard (design §5.4)
_WRITE_RETRIES = int(os.environ.get("CHOCO_HP_WRITE_RETRIES", "8"))


def _key(experiment_id: str) -> str:
    return f"{KEY_PREFIX}{experiment_id}"


def _meta_key(experiment_id: str) -> str:
    return f"{KEY_PREFIX}{experiment_id}:meta"


def _redis_params() -> dict:
    """Shared connection facts from chocofarm/config.py (the transport in parallel.py uses the same),
    so the registry and the transport address one redis instance by default."""
    from chocofarm import config
    return config.redis_params()


# ---------------------------------------------------------------------------
# Errors (design §7) — each fail-loud, each distinct so the operator's mental model stays true
# ---------------------------------------------------------------------------
class RegistryUnavailable(RuntimeError):
    """redis is down / unreachable / stalled (design §7). The reader raises this rather than
    silently reusing the last snapshot forever — a silent reuse means an operator's change never
    lands and they are never told (the ADR-0002 silent-failure)."""


class RegistryKeyMissing(RuntimeError):
    """No registry blob for the experiment_id (unseeded, or wiped by a redis restart — design §7).
    Never coerce to defaults silently; the remediation is `init` (or the launch seed)."""


class RegistryWriteConflict(RuntimeError):
    """A write lost its optimistic WATCH race repeatedly (design §5.4) — bounded retries exhausted.
    Neither writer's change is silently dropped: the loser is told to retry."""


class RegistrySchemaDrift(RuntimeError):
    """A re-bound blob's recorded env-derived facts (feature dim / action-slot count / dtype /
    present_k) disagree with the running process (design §7) — the blob was seeded against a
    different env or precision. Refused loudly rather than running a mismatched net silently."""


class RestartRequired(RuntimeError):
    """A RESTART or INSTANCE field differs from the value the running process was CONSTRUCTED with
    (design §3.4). Raised LOUDLY naming the field, the construction-time value, and the new value.
    The operator adopts a RESTART change by restarting with --resume; an INSTANCE change is a NEW
    experiment (the running net is invalid against the changed env)."""


def _connect():
    """Open a bounded redis connection (mirrors parallel.py's bounded-socket discipline / ADR-0002:
    a stall becomes a loud error, not a silent hang). Pings now so an unreachable redis fails loud
    here rather than mid-read."""
    try:
        import redis
    except ImportError as e:  # pragma: no cover - environment guard
        raise RegistryUnavailable(
            "redis-py is not importable; the hp registry needs it (it is the transport's dep too)"
        ) from e
    from chocofarm import config
    r = redis.Redis(
        socket_timeout=config.redis_socket_timeout(),
        socket_connect_timeout=config.redis_connect_timeout(),
        **_redis_params(),
    )
    try:
        r.ping()
    except Exception as e:
        raise RegistryUnavailable(
            f"hp registry redis unreachable at {_redis_params()}: {e}") from e
    return r


# ---------------------------------------------------------------------------
# Store primitives (design §5.2 / §5.4)
# ---------------------------------------------------------------------------
def write_config(experiment_id: str, cfg: ExperimentConfig, writer: str | None = None, r=None) -> None:
    """Write the whole config blob (+ meta) for `experiment_id` with NO TTL (design §2.1: a bare
    SET leaves TTL=-1). Validates before writing (ADR-0002: refuse a malformed write at the source).
    Used by the bootstrap seed and the CLI `set`/`init` (which add the optimistic guard around it)."""
    check_invariants(cfg)  # never store a config that would fail its own read-time decode
    own = r is None
    if own:
        r = _connect()
    try:
        blob = json.dumps(encode_config(cfg), sort_keys=True)
        now = time.time()
        existing_meta = r.get(_meta_key(experiment_id))
        created = now
        if existing_meta is not None:
            try:
                created = json.loads(existing_meta).get("created_at", now)
            except (ValueError, TypeError):
                created = now
        meta = {
            "schema_version": SCHEMA_VERSION,
            "created_at": created,
            "last_write_at": now,
            "writer": writer or _default_writer(),
        }
        pipe = r.pipeline(transaction=False)
        pipe.set(_key(experiment_id), blob)               # NO ex= — no TTL (design §2.1)
        pipe.set(_meta_key(experiment_id), json.dumps(meta))
        pipe.execute()
    finally:
        if own:
            try:
                r.close()
            except Exception:
                pass


def read_config(experiment_id: str, r=None) -> ExperimentConfig:
    """GET + strict-decode the experiment's blob (design §3.6). Raises `RegistryKeyMissing` if
    unseeded (never coerce to defaults — design §7), `RegistryDecodeError` if malformed/drifted,
    `RegistryUnavailable` if redis stalls."""
    own = r is None
    if own:
        r = _connect()
    try:
        try:
            raw = r.get(_key(experiment_id))
        except Exception as e:
            raise RegistryUnavailable(
                f"hp registry GET failed for {experiment_id!r}: {e}") from e
        if raw is None:
            raise RegistryKeyMissing(
                f"no registry blob for experiment_id {experiment_id!r} (key {_key(experiment_id)!r}) "
                "— was it seeded? run `python -m chocofarm.hp.registry init --experiment-id "
                f"{experiment_id}`, or the launch seed failed / a redis restart wiped it (design §7)")
        try:
            data = json.loads(raw)
        except ValueError as e:
            raise RegistryDecodeError(
                f"registry blob for {experiment_id!r} is not valid JSON: {e}") from e
        return decode_config(data)
    finally:
        if own:
            try:
                r.close()
            except Exception:
                pass


def exists(experiment_id: str, r=None) -> bool:
    own = r is None
    if own:
        r = _connect()
    try:
        return bool(r.exists(_key(experiment_id)))
    finally:
        if own:
            try:
                r.close()
            except Exception:
                pass


def delete_experiment(experiment_id: str, r=None) -> int:
    """Delete an experiment's blob + meta (test cleanup / operator teardown). Returns the number
    of keys removed."""
    own = r is None
    if own:
        r = _connect()
    try:
        return int(r.delete(_key(experiment_id), _meta_key(experiment_id)))
    finally:
        if own:
            try:
                r.close()
            except Exception:
                pass


def _default_writer() -> str:
    """Writer attribution (design §5.5): CHOCO_HP_WRITER override, else the OS user (mirroring the
    transport's actor-awareness instinct)."""
    w = os.environ.get("CHOCO_HP_WRITER")
    if w:
        return w
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Field-path resolution (design §5.3 — `set group.field value`) over the typed schema
# ---------------------------------------------------------------------------
_GROUP_TYPES = {
    "env": EnvConfig, "search": SearchConfig, "value": ValueTargetConfig,
    "feat": FeatureConfig, "arch": ArchConfig, "train": TrainConfig,
    "loop": ExItLoopConfig, "eval": EvalConfig, "par": ParallelConfig,
    "bounds": BoundsConfig,
}


def _split_path(path: str):
    """`'train.lr'` -> ('train', 'lr'). Loud on a malformed path or an unknown group/field."""
    if "." not in path:
        raise RegistryDecodeError(
            f"field path {path!r} must be 'group.field' (e.g. train.lr); known groups: "
            f"{sorted(_GROUP_TYPES)}")
    group, _, leaf = path.partition(".")
    if group not in _GROUP_TYPES:
        raise RegistryDecodeError(
            f"unknown group {group!r} in path {path!r}; known: {sorted(_GROUP_TYPES)}")
    leaf_names = {f.name for f in fields(_GROUP_TYPES[group])}
    if leaf not in leaf_names:
        raise RegistryDecodeError(
            f"unknown field {leaf!r} in group {group!r}; known: {sorted(leaf_names)}")
    return group, leaf


def apply_field(cfg: ExperimentConfig, path: str, raw_value: str) -> ExperimentConfig:
    """Apply one `group.field=raw_value` to a COPY of `cfg`, parsing `raw_value` against the field's
    declared type via the same strict codec, then re-validating the whole config (design §5.3). The
    parse goes through `decode_config` so a bad value fails with the SAME strict checks the read
    path uses (no second, looser validator). Returns the mutated copy; raises `RegistryDecodeError`
    on a bad value/type/invariant. Does NOT write — the caller does, under the optimistic guard."""
    group, leaf = _split_path(path)
    # encode the current config, mutate the one leaf with a python-literal parse of raw_value, then
    # round-trip through the strict decoder so the new value is type-checked exactly like a read.
    data = encode_config(cfg)
    data[group][leaf] = _parse_literal(raw_value)
    return decode_config(data)


def _parse_literal(raw: str):
    """Parse a CLI string token into a JSON value (the codec then type-checks it against the field).
    Accepts JSON literals (numbers, true/false/null, quoted strings, lists) and falls back to a bare
    string. `1e-4` parses as a float, `true` as bool, `null` as None, `[1,2]` as a list."""
    s = raw.strip()
    try:
        return json.loads(s)
    except ValueError:
        return s  # a bare unquoted string (e.g. an entry name or a path)


# ---------------------------------------------------------------------------
# Atomic write under the optimistic WATCH/MULTI/EXEC guard (design §5.4)
# ---------------------------------------------------------------------------
def set_fields(experiment_id: str, updates: dict, writer: str | None = None, r=None):
    """Atomically apply a related SET of `group.field -> raw_value` updates (the motivating
    drop-lr-and-raise-l2 case — design §5.4). Read-modify-write of the one blob under WATCH; on a
    lost EXEC (another writer raced between WATCH and EXEC) retry a bounded number of times, then
    fail loud (`RegistryWriteConflict`). Validates the merged config in-memory BEFORE the write, so
    a bad value never touches redis. Returns (old_cfg, new_cfg) for the change log.

    `updates`: ordered dict / dict of `'train.lr' -> '1e-4'` (raw CLI strings)."""
    own = r is None
    if own:
        r = _connect()
    try:
        import redis as _redis
        key = _key(experiment_id)
        last_err = None
        for _attempt in range(_WRITE_RETRIES):
            with r.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    if raw is None:
                        raise RegistryKeyMissing(
                            f"cannot set on unseeded experiment_id {experiment_id!r} — run `init` first")
                    old_cfg = decode_config(json.loads(raw))
                    new_cfg = old_cfg
                    for path, value in updates.items():
                        new_cfg = apply_field(new_cfg, path, value)
                    check_invariants(new_cfg)  # belt-and-braces: the merged config must be valid
                    blob = json.dumps(encode_config(new_cfg), sort_keys=True)
                    now = time.time()
                    existing_meta = pipe.get(_meta_key(experiment_id))
                    created = now
                    if existing_meta is not None:
                        try:
                            created = json.loads(existing_meta).get("created_at", now)
                        except (ValueError, TypeError):
                            created = now
                    meta = {
                        "schema_version": SCHEMA_VERSION,
                        "created_at": created,
                        "last_write_at": now,
                        "writer": writer or _default_writer(),
                    }
                    pipe.multi()
                    pipe.set(key, blob)                       # NO ex= — no TTL (design §2.1)
                    pipe.set(_meta_key(experiment_id), json.dumps(meta))
                    pipe.execute()
                    return old_cfg, new_cfg
                except _redis.WatchError as e:
                    last_err = e
                    continue  # another writer raced; retry the read-modify-write
        raise RegistryWriteConflict(
            f"could not land the set on {experiment_id!r} after {_WRITE_RETRIES} optimistic "
            f"retries — repeated concurrent writes (design §5.4). Last: {last_err!r}")
    finally:
        if own:
            try:
                r.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Bootstrap from argparse (design §6 — the consolidation)
# ---------------------------------------------------------------------------
# The mapping from the exit_loop argparse namespace attribute names to the schema's nested fields.
# argparse remains the launch CLI; this adapter is the bridge that makes the dataclass defaults =
# the argparse defaults and seeds the registry identically to launching the CLI today (design §6).
# Each entry is (argparse_attr, group, field); a missing argparse_attr leaves the dataclass default.
_ARGPARSE_MAP = (
    # search
    ("m", "search", "m"),
    ("n_sims", "search", "n_sims"),
    # train (the jit-boundary knobs; lr/l2 are RESTART)
    ("lr", "train", "lr"),
    ("l2", "train", "l2"),
    ("alpha", "train", "alpha"),
    ("beta", "train", "beta"),
    ("epochs", "train", "epochs"),
    ("batch", "train", "batch"),
    # value-target
    ("td_lambda", "value", "td_lambda"),
    ("n_step", "value", "n_step"),
    # loop
    ("iters", "loop", "iters"),
    ("episodes", "loop", "episodes"),
    ("window", "loop", "window"),
    ("lam", "loop", "lam"),
    ("explore_plies", "loop", "explore_plies"),
    ("seed", "loop", "seed"),
    # arch
    ("hidden", "arch", "hidden"),
    ("residual", "arch", "residual"),
    ("seed", "arch", "init_seed"),  # exit_loop's --seed seeds the net He-init too
    # eval
    ("eval_n", "eval", "eval_n"),
    ("eval_seed", "eval", "eval_seed"),
    # parallel
    ("workers", "par", "workers"),
    ("cores", "par", "cores"),
)


def from_argparse(args: argparse.Namespace, experiment_id: str) -> ExperimentConfig:
    """Build an `ExperimentConfig` from a parsed argparse namespace (design §6). The dataclass
    defaults ARE the argparse defaults, so a namespace carrying its argparse defaults produces a
    config equal to the dataclass defaults — one source. Any arg the namespace does not carry
    leaves the schema default. Derived dims (arch.in_dim / arch.n_actions) and env INSTANCE facts
    are filled by `seed_registry` (which has the env in hand), not here."""
    cfg = ExperimentConfig(experiment_id=experiment_id)
    groups = {
        "env": cfg.env, "search": cfg.search, "value": cfg.value, "feat": cfg.feat,
        "arch": cfg.arch, "train": cfg.train, "loop": cfg.loop, "eval": cfg.eval,
        "par": cfg.par, "bounds": cfg.bounds,
    }
    for attr, group, fld in _ARGPARSE_MAP:
        if hasattr(args, attr):
            val = getattr(args, attr)
            setattr(groups[group], fld, val)
    check_invariants(cfg)
    return cfg


def seed_registry(experiment_id: str, cfg: ExperimentConfig, env=None,
                  overwrite: bool = False, r=None) -> ExperimentConfig:
    """Seed the experiment's blob from `cfg` IF IT DOES NOT ALREADY EXIST (idempotent — design §6).
    A --resume of an existing experiment re-binds to the existing blob (does NOT clobber operator
    overrides) unless `overwrite=True`. If `env` is given, the derived dims + INSTANCE facts are
    recorded for the drift check (design §7): arch.in_dim/n_actions and the env constants the net
    was fit to. On the disk-persisted 6379 redis (`noeviction`) registry keys never expire or evict,
    so no eviction-policy nudge is needed (the 6380 memory-cache instance once required one).

    Returns the config now in the registry (the freshly-seeded one, or the existing one on a
    re-bind). This is also the post-restart recovery path (design §2.3)."""
    own = r is None
    if own:
        r = _connect()
    try:
        if env is not None:
            _record_derived(cfg, env)
        if not overwrite and exists(experiment_id, r=r):
            existing = read_config(experiment_id, r=r)
            # §7 drift check on re-bind: the existing blob's RECORDED, env-derived facts (feature
            # dim / action-slot count / dtype) must match the running process's. A mismatch means
            # the blob was seeded against a different env or precision — re-binding to it would run a
            # net shaped for one env under another. Fail loud rather than decode-into-wrong.
            if env is not None:
                _assert_no_derived_drift(experiment_id, recorded=existing, live=cfg)
            print(f"[hp-registry] experiment {experiment_id!r} already seeded — re-binding to the "
                  f"existing blob (idempotent; operator overrides preserved)", flush=True)
            return existing
        write_config(experiment_id, cfg, r=r)
        print(f"[hp-registry] seeded experiment {experiment_id!r} from launch defaults "
              f"(schema v{SCHEMA_VERSION}, no TTL)", flush=True)
        return cfg
    finally:
        if own:
            try:
                r.close()
            except Exception:
                pass


def _record_derived(cfg: ExperimentConfig, env) -> None:
    """Fill the DERIVED, recorded-for-provenance fields from the env + the running precision (design
    §4.4 / §7): the feature dim and action-slot count that size the net, the env INSTANCE constants
    the net is fit to, and the live CHOCO_AZ_DTYPE (read once at import). These are recorded (the
    drift check reads them), not free knobs."""
    from chocofarm.az.features import feature_dim
    from chocofarm.az.actions import n_action_slots
    from chocofarm.az.dtypes import DTYPE
    import numpy as _np
    cfg.arch.in_dim = int(feature_dim(env))
    cfg.arch.n_actions = int(n_action_slots(env))
    cfg.arch.dtype = "float32" if _np.dtype(DTYPE) == _np.dtype(_np.float32) else "float64"
    cfg.env.present_k = int(env.K)
    cfg.env.teleport_overhead = float(env.tp)
    cfg.env.entry = str(env.entry)


def _assert_no_derived_drift(experiment_id: str, recorded: ExperimentConfig,
                             live: ExperimentConfig) -> None:
    """Fail loud (design §7) if a re-bound blob's recorded env-derived facts disagree with the
    running process's. These define the net shape / precision the blob was fit to; re-binding under
    a different env or dtype would run a mismatched net silently."""
    drifts = []
    if recorded.arch.in_dim is not None and recorded.arch.in_dim != live.arch.in_dim:
        drifts.append(f"arch.in_dim recorded={recorded.arch.in_dim} live={live.arch.in_dim}")
    if recorded.arch.n_actions is not None and recorded.arch.n_actions != live.arch.n_actions:
        drifts.append(f"arch.n_actions recorded={recorded.arch.n_actions} live={live.arch.n_actions}")
    # normalize the dtype aliases (f32/float32) before comparing
    _norm = {"f32": "float32", "f64": "float64"}
    rd = _norm.get(recorded.arch.dtype, recorded.arch.dtype)
    ld = _norm.get(live.arch.dtype, live.arch.dtype)
    if rd != ld:
        drifts.append(f"arch.dtype recorded={recorded.arch.dtype!r} live={live.arch.dtype!r}")
    if recorded.env.present_k != live.env.present_k:
        drifts.append(f"env.present_k recorded={recorded.env.present_k} live={live.env.present_k}")
    if drifts:
        raise RegistrySchemaDrift(
            f"env/precision drift on re-bind to experiment {experiment_id!r} (design §7): "
            + "; ".join(drifts)
            + ". The stored blob was seeded against a different env / precision than this process is "
            "running. Re-binding would run a mismatched net silently — refusing. Use a new "
            "experiment_id, or --overwrite if the blob is stale.")


# ---------------------------------------------------------------------------
# Live read snapshot + RESTART-refusal (design §3)
# ---------------------------------------------------------------------------
class ConfigSnapshot:
    """A per-process typed snapshot of an experiment's config (design §3.1), refreshed once per
    outer-iteration boundary. Wraps the decoded `ExperimentConfig` (read HOT fields off `.cfg`)
    with the `launched_with` shadow (the construction-time argparse truth — design §6) that the
    RESTART-refusal (§3.4) compares against.

    Within an iteration the snapshot is fixed (atomicity — §3.3): an episode never sees a
    half-applied multi-field change. `refresh()` re-reads at the boundary; `assert_no_restart_drift`
    fires the loud refusal if a RESTART/INSTANCE field moved off its construction value."""

    def __init__(self, experiment_id: str, cfg: ExperimentConfig, launched_with: ExperimentConfig):
        self.experiment_id = experiment_id
        self.cfg = cfg
        self.launched_with = launched_with
        self._iter = -1

    @classmethod
    def launch(cls, experiment_id: str, launched_with: ExperimentConfig, r=None) -> "ConfigSnapshot":
        """Bind a snapshot at launch: read the current registry blob (must be seeded) and capture
        the construction-time config as the `launched_with` shadow. Both are validated by the read
        path. Use this once after `seed_registry`, then `refresh()` per iteration."""
        cfg = read_config(experiment_id, r=r)
        return cls(experiment_id, cfg, launched_with)

    def refresh(self, iteration: int, r=None) -> "ConfigSnapshot":
        """Re-read the registry blob at an iteration boundary (design §3.1). Logs any applied HOT
        change loudly (§5.5 reader side), and fires the loud RESTART/INSTANCE refusal (§3.4) before
        returning if a baked field moved. Returns self (the snapshot now holds the fresh cfg)."""
        new_cfg = read_config(self.experiment_id, r=r)
        self._log_hot_changes(self.cfg, new_cfg, iteration)
        self.cfg = new_cfg
        self._iter = iteration
        self.assert_no_restart_drift()
        return self

    def assert_no_restart_drift(self) -> None:
        """Compare every RESTART / INSTANCE leaf against the `launched_with` shadow; raise
        `RestartRequired` loudly on the first mismatch (design §3.4). HOT fields are exempt (they
        are meant to change). This is the failure mode the mutability facet exists to make loud
        rather than silently-ineffective."""
        for group_name, mut, fld, live, launched in _iter_facet_diffs(self.cfg, self.launched_with):
            if mut is Mut.HOT:
                continue
            if live != launched:
                if mut is Mut.RESTART:
                    raise RestartRequired(
                        f"RESTART field {group_name}.{fld} changed on the running experiment "
                        f"{self.experiment_id!r}: constructed with {launched!r}, registry now says "
                        f"{live!r}. This value is baked into a constructed object / jit closure / "
                        f"array shape (design §3.4/§4.5) and is NOT live. To adopt it, restart with "
                        f"--resume <latest_net.npz> (the loop checkpoints every iteration, so this "
                        f"loses nothing). Refusing loudly rather than telling you a change took that "
                        f"did not (ADR-0002).")
                else:  # INSTANCE
                    raise RestartRequired(
                        f"INSTANCE field {group_name}.{fld} changed on the running experiment "
                        f"{self.experiment_id!r}: constructed with {launched!r}, registry now says "
                        f"{live!r}. This defines the belief-MDP itself (design C5) — the running net "
                        f"is INVALID against the changed env. This is not a re-tune: start a NEW "
                        f"experiment_id (do NOT --resume). Refusing loudly (ADR-0002).")

    def _log_hot_changes(self, old: ExperimentConfig, new: ExperimentConfig, iteration: int) -> None:
        """Log every applied HOT change loudly at the iteration boundary (design §5.5 reader side):
        'applied <field>: <old> -> <new> at iter N'."""
        for group_name, mut, fld, new_v, old_v in _iter_facet_diffs(new, old):
            if mut is Mut.HOT and new_v != old_v:
                print(f"[hp-registry] applied {group_name}.{fld}: {old_v!r} -> {new_v!r} "
                      f"at iter {iteration} (experiment {self.experiment_id!r})", flush=True)


def _iter_facet_diffs(a: ExperimentConfig, b: ExperimentConfig):
    """Yield (group_name, mut, field_name, a_value, b_value) over every leaf field of two configs,
    carrying each field's `Mut` facet (read from the schema metadata). The single place the read
    path walks the facet-tagged leaves."""
    groups = {
        "env": (a.env, b.env), "search": (a.search, b.search), "value": (a.value, b.value),
        "feat": (a.feat, b.feat), "arch": (a.arch, b.arch), "train": (a.train, b.train),
        "loop": (a.loop, b.loop), "eval": (a.eval, b.eval), "par": (a.par, b.par),
        "bounds": (a.bounds, b.bounds),
    }
    for group_name, (ga, gb) in groups.items():
        for f in fields(ga):
            mut = f.metadata.get("mut")
            yield group_name, mut, f.name, getattr(ga, f.name), getattr(gb, f.name)


# ---------------------------------------------------------------------------
# Operator CLI (design §5.3): get / set / init
# ---------------------------------------------------------------------------
def _cli_get(args) -> int:
    cfg = read_config(args.experiment_id)
    print(json.dumps(encode_config(cfg), indent=2, sort_keys=True))
    return 0


def _cli_set(args) -> int:
    # args.assignments is a flat [path, value, path, value, ...] list (the atomic multi-field case)
    toks = args.assignments
    if not toks or len(toks) % 2 != 0:
        raise SystemExit("set takes pairs: PATH VALUE [PATH VALUE ...] (e.g. train.lr 1e-4)")
    updates = {}
    order = []
    for i in range(0, len(toks), 2):
        path, value = toks[i], toks[i + 1]
        updates[path] = value
        order.append((path, value))
    # warn loudly (design §5.3 step iii) about RESTART/INSTANCE fields BEFORE writing, so the
    # operator is told the truth about when the change takes effect.
    for path, _value in order:
        group, leaf = _split_path(path)
        f = next(ff for ff in fields(_GROUP_TYPES[group]) if ff.name == leaf)
        mut = f.metadata.get("mut")
        if mut is Mut.RESTART:
            print(f"[hp-registry] NOTE: {path} is RESTART — a running process will REFUSE this "
                  f"change loudly and keep its constructed value; adopt it by restarting with "
                  f"--resume (design §3.4). The change is still recorded.", flush=True)
        elif mut is Mut.INSTANCE:
            print(f"[hp-registry] NOTE: {path} is INSTANCE — this defines the belief-MDP; a running "
                  f"process refuses it (the net is invalid against the new env). Start a NEW "
                  f"experiment, do not --resume (design §3.4). The change is still recorded.",
                  flush=True)
    old_cfg, new_cfg = set_fields(args.experiment_id, updates, writer=args.writer)
    # log every applied change loudly (design §5.5 write side)
    for path, _value in order:
        group, leaf = _split_path(path)
        ov = getattr(getattr(old_cfg, group), leaf)
        nv = getattr(getattr(new_cfg, group), leaf)
        print(f"[hp-registry] set {args.experiment_id} {path}: {ov!r} -> {nv!r} "
              f"by {args.writer or _default_writer()}", flush=True)
    return 0


def _cli_init(args) -> int:
    # seed from the dataclass defaults (design §6 bootstrap). The launch path normally seeds with an
    # env in hand (for the derived dims); the CLI `init` seeds from pure defaults and nudges policy.
    cfg = ExperimentConfig(experiment_id=args.experiment_id)
    env = None
    if not args.no_env:
        try:
            from chocofarm.model.env import Environment
            env = Environment()
        except Exception as e:
            print(f"[hp-registry] could not construct Environment for derived dims ({e}); "
                  f"seeding without them (--no-env to silence)", flush=True)
    seeded = seed_registry(args.experiment_id, cfg, env=env, overwrite=args.overwrite)
    print(json.dumps(encode_config(seeded), indent=2, sort_keys=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m chocofarm.hp.registry",
        description="chocofarm hyperparameter registry — operator CLI (design §5.3). "
                    "get / set / init against the redis-backed live registry.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("get", help="read + validate + pretty-print the whole config")
    g.add_argument("--experiment-id", required=True)
    g.set_defaults(func=_cli_get)

    s = sub.add_parser("set", help="set one or more group.field values atomically (validated first)")
    s.add_argument("--experiment-id", required=True)
    s.add_argument("--writer", default=None, help="writer attribution (default CHOCO_HP_WRITER/user)")
    s.add_argument("assignments", nargs="+",
                   help="PATH VALUE pairs, e.g. train.lr 1e-4 train.l2 5e-4 (atomic multi-field set)")
    s.set_defaults(func=_cli_set)

    i = sub.add_parser("init", help="seed a fresh experiment from the dataclass defaults (idempotent)")
    i.add_argument("--experiment-id", required=True)
    i.add_argument("--overwrite", action="store_true",
                   help="overwrite an existing blob (default: idempotent re-bind)")
    i.add_argument("--no-env", action="store_true",
                   help="do not construct an Environment for the derived dims")
    i.set_defaults(func=_cli_init)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
