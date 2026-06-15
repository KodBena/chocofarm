#!/usr/bin/env python3
"""
chocofarm hp — the typed dataclass schema (design §1) and the strict, fail-loud codec (design §3.6).

The schema is the single contract: the one place the hyperparameter taxonomy lives, the defaults
live (= the argparse defaults, so the registry seeds identically to launching the CLI today), the
per-field mutability facet lives, and the per-field (de)serialization lives. Nothing about a
hyperparameter is recorded in two places.

A `Mut` facet (design §3.4) is attached to every leaf field via `field(metadata=...)`:

  * HOT      — read fresh at point of use; safe to change on a running experiment (apply + log).
  * RESTART  — baked into a constructed object / jit closure / array shape; a mid-run change is
               REFUSED LOUDLY (design §3.4), the operator adopts it by restarting with --resume.
  * INSTANCE — defines the belief-MDP itself (env.py constants, C5); a change is a NEW experiment,
               not a re-tune — refused loudly with a stronger remediation.

The facet is a READING of where the code consumes each value (design §4 surveys it from the actual
code: the jax.jit and constructor boundaries draw the line, not taste). It lives next to the field
so it moves WITH the code.

The codec (design §3.6): redis stores bytes; the dataclass is typed. `encode_config` flattens the
nested dataclass to a JSON-serializable dict; `decode_config` validates and reconstructs it,
raising a loud `RegistryDecodeError` on ANY type / domain / unknown-key / missing-key /
cross-field-invariant mismatch. A malformed or missing value is NEVER coerced to a default —
that is the ADR-0002 silent-failure this codec exists to prevent (the dual-bound `vhat=None`-vs-
`vhat_zero` confusion in `dual-bound.md` §4.2 is the cautionary tale: a silently-defaulted
hyperparameter produced a wrong number that looked right).

The hand-written recursive decoder over `dataclasses.fields()` + the field annotations is used
rather than a dependency (`dacite`): the design's §8 verdict is "stdlib dataclasses + a ~50-line
decoder beats pulling a dependency for a schema this size," and the shared scratch venv carries no
`dacite`. The decoder is the strict, fail-loud half the verdict keeps.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field, fields
from enum import Enum

from chocofarm import config


# The schema-shape version. Bumped when the dataclass shape changes (a field added / removed /
# renamed / re-typed). The stored blob carries it; the reader checks it against the code's
# (design §7, schema-drift failure mode) so an upgraded code reading an old blob fails LOUDLY
# rather than decoding into a subtly-wrong config.
SCHEMA_VERSION = 1


class Mut(Enum):
    """Per-field mutability facet (design §3.4)."""

    HOT = "hot"            # read fresh at point of use; safe to change on a running experiment
    RESTART = "restart"    # baked into a constructed object / jit closure / array shape
    INSTANCE = "instance"  # defines the belief-MDP itself; a change is a NEW experiment


class RegistryDecodeError(ValueError):
    """A stored hyperparameter blob failed strict typed decode (design §3.6): a type, domain,
    unknown-key, missing-key, or cross-field-invariant mismatch. Raised LOUDLY — the reader never
    proceeds on a config it could not validate, and never coerces to a default (ADR-0002)."""


def hp(default, mut: Mut, doc: str, codec: str = "json"):
    """Declare a leaf hyperparameter field (design §1). Wraps `field(default=..., metadata=...)`
    so the per-field facet / doc / codec stay readable at the declaration site. The `default` is
    the argparse default (one source of defaults — design §6 consolidation)."""
    return field(default=default, metadata={"mut": mut, "doc": doc, "codec": codec})


# ---------------------------------------------------------------------------
# The ten per-axis groups (design §1 / §4 — each maps to one file / constructed object so a reader
# can trace a field back to the line that consumes it). Defaults = the argparse defaults.
# ---------------------------------------------------------------------------
@dataclass
class EnvConfig:
    """chocofarm/model/env.py — INSTANCE-defining (design C5). These define the belief-MDP itself;
    a net is fit against a specific env, so a mid-run change silently invalidates it (refuse loudly,
    new experiment). `max_steps` is the one HOT field — a per-call rollout cap, not instance shape."""

    instance_path: typing.Optional[str] = hp(None, Mut.INSTANCE, "geometry source (data/instance.json); None=package default")
    teleport_overhead: float = hp(12.0, Mut.INSTANCE, "TELE_OH; exit toll added to every run")
    present_k: int = hp(5, Mut.INSTANCE, "treasures present per world (env.K)")
    entry: str = hp("CSNE", Mut.INSTANCE, "entry teleport / start location")
    value_vector: typing.Optional[typing.List[float]] = hp(None, Mut.INSTANCE, "per-treasure reward; None=unit")
    max_steps: int = hp(40, Mut.HOT, "rollout horizon cap (per-call, not instance)")


@dataclass
class SearchConfig:
    """chocofarm/az/gumbel_search.py — `m`/`n_sims`/`use_jax_mlp` bake into the constructed search
    (RESTART); the `c_*`/`max_depth` knobs are read off `self` per selection (HOT)."""

    m: int = hp(12, Mut.RESTART, "Gumbel root actions; sizes the SH bracket")
    n_sims: int = hp(48, Mut.RESTART, "sim budget; baked into the SH phase loop")
    c_puct: float = hp(1.25, Mut.HOT, "PUCT exploration coeff (read per selection)")
    c_visit: float = hp(50.0, Mut.HOT, "Danihelka sigma additive const")
    c_scale: float = hp(1.0, Mut.HOT, "Danihelka sigma multiplicative scale")
    c_outcome: int = hp(2, Mut.HOT, "leaf outcome-averaging count (loop bound, read per sim)")
    max_depth: int = hp(24, Mut.HOT, "interior PUCT descent cutoff (soft, per recursion)")
    use_jax_mlp: bool = hp(False, Mut.RESTART, "jit forward vs numpy fast path; binds a fn")


@dataclass
class ValueTargetConfig:
    """chocofarm/az/value_target.py — all pure functions, all HOT (read as per-call args). The
    `td_lambda < 1.0` and `n_step is not None` combination is the mutually-exclusive cross-field
    invariant the codec enforces (design §3.6 / §4.2)."""

    td_lambda: float = hp(1.0, Mut.HOT, "TD(lambda) blend; 1.0=pure MC (mutually excl. n_step)")
    n_step: typing.Optional[int] = hp(None, Mut.HOT, "n-step bootstrap horizon; None=inf=pure MC")


@dataclass
class FeatureConfig:
    """chocofarm/az/features.py — the per-block multipliers ARE the input dimension (they size
    ValueMLP.W1), so RESTART. Exposed for provenance; the feature_dim itself is derived from env."""

    # NOTE (audit R6): these are PROVENANCE counts only — never read to slice/build the vector
    # (feature_dim is derived from env via FeatureLayout). The display strings below mirror the
    # canonical block-display tokens FeatureLayout.element_names emits (e.g. the belief-sharpness
    # scalar is named "log|bw|", its block KEY is "sharpness"); a future change to the layout's
    # group widths should update FeatureLayout, and these provenance counts alongside it.
    # SSOT (audit item G): FeatureLayout (chocofarm/az/features.py) is the single source of truth for
    # the feature layout; these counts are a provenance copy PINNED to it — registry derives the
    # per-group widths from FeatureLayout.blocks (in _feature_group_channels) and fails loud if these
    # disagree, on BOTH the fresh-seed path (_record_derived) and the re-bind drift check, so the
    # layout cannot drift away from this copy silently.
    per_treasure_channels: int = hp(5, Mut.RESTART, "marg,collected,available,dist,unc")
    per_detector_channels: int = hp(3, Mut.RESTART, "informative,p_pos,dist")
    global_channels: int = hp(6, Mut.RESTART, "log|bw|,n_collected,sum_marg,exit_cost,nonempty,sum_unc (+n_tele)")


@dataclass
class ArchConfig:
    """chocofarm/az/mlp.py + actions.py — all weight-matrix shapes, RESTART. `in_dim`/`n_actions`
    are DERIVED from env (feature_dim / n_action_slots) and recorded for the drift check (design
    §7), not free knobs — they default to None and the launch seed fills the recorded values."""

    hidden: int = hp(256, Mut.RESTART, "trunk width; sizes every weight matrix")
    residual: bool = hp(False, Mut.RESTART, "gates the HxH residual block params")
    init_seed: int = hp(0, Mut.RESTART, "He-init RNG; consumed only at construction")
    in_dim: typing.Optional[int] = hp(None, Mut.RESTART, "DERIVED feature_dim; recorded for drift check, not a free knob")
    n_actions: typing.Optional[int] = hp(None, Mut.RESTART, "DERIVED n_action_slots; recorded for drift check, not a free knob")
    dtype: str = hp("float32", Mut.RESTART, "CHOCO_AZ_DTYPE; read once at import")


@dataclass
class TrainConfig:
    """chocofarm/az/mlp_jax_train.py — the jit boundary (design C4). `lr`/`l2` are LIVE (HOT) as of
    audit R13 (training-optimization-refactor.md §4.1, the frozen-config headline): `lr` is injected
    via `optax.inject_hyperparams`, so it lives in `opt_state.hyperparams` as a traced value and is
    set per step from the live snapshot (no rebuild — Adam's moments persist across a live anneal);
    `l2` is a traced loss coefficient (joins `alpha`/`beta` as a `value_and_grad` arg). So a registry
    lr-drop / L2-retune now lands LIVE on the running experiment, not as a `--resume` adoption.
    `betas`/`eps` are STILL baked into `optax.adam` at construction (RESTART — R13 is the minimal
    lr/l2 slice; the betas/eps live-injection + the full Optimizer⊥Trainer object split are the
    deferred follow-up, design note §2.1). `alpha`/`beta` are traced call-args (HOT). `epochs`/
    `batch` are loop bounds read at iter start (HOT)."""

    lr: float = hp(1e-3, Mut.HOT, "Adam lr — LIVE via optax.inject_hyperparams, set per step (audit R13)")
    l2: float = hp(1e-4, Mut.HOT, "L2 — LIVE traced loss coefficient, read per step (audit R13)")
    beta1: float = hp(0.9, Mut.RESTART, "Adam b1 — baked into optax.adam (R13 defers betas/eps)")
    beta2: float = hp(0.999, Mut.RESTART, "Adam b2 — baked into optax.adam (R13 defers betas/eps)")
    eps: float = hp(1e-8, Mut.RESTART, "Adam eps — baked into optax.adam (R13 defers betas/eps)")
    alpha: float = hp(1.0, Mut.HOT, "policy CE weight — traced call-arg, read each step")
    beta: float = hp(1.0, Mut.HOT, "value MSE weight — traced call-arg, read each step")
    epochs: int = hp(2, Mut.HOT, "train epochs over the buffer per iter (loop bound)")
    batch: int = hp(256, Mut.HOT, "minibatch size (loop bound, read at iter start)")


@dataclass
class ExItLoopConfig:
    """chocofarm/az/exit_loop.py — the outer loop. The loop bounds (`iters`/`episodes`/`window`/
    `explore_plies`) and `lam` are read at iteration start (HOT). `seed` is the master RNG folded
    into worker/episode seeds at launch — changing it mid-run breaks the parallel≈serial
    determinism contract, so RESTART."""

    iters: int = hp(40, Mut.HOT, "outer ExIt iterations (loop bound)")
    episodes: int = hp(300, Mut.HOT, "self-play episodes/iter (read at iter start)")
    window: int = hp(5, Mut.HOT, "replay window in iterations")
    lam: float = hp(0.0855, Mut.HOT, "pinned lambda0 (static-floor rate)")
    explore_plies: int = hp(4, Mut.HOT, "plies sampling executed action from pi'")
    seed: int = hp(7, Mut.RESTART, "master RNG seed; folded into worker seeds at launch")


@dataclass
class EvalConfig:
    """exit_loop eval block + eval_az.py — MC held-out eval. Both HOT (sample size + draw seed)."""

    eval_n: int = hp(200, Mut.HOT, "held-out eval episodes/iter")
    eval_seed: int = hp(12345, Mut.HOT, "eval world draw seed")


@dataclass
class ParallelConfig:
    """chocofarm/az/parallel.py — the process pool is built once before the loop, workers are
    core-pinned in the initializer, the redis connection opens once. All RESTART."""

    workers: int = hp(4, Mut.RESTART, "process-pool size; pool built once before the loop")
    cores: str = hp("0,1,2,3", Mut.RESTART, "core-pin list; set in the pool initializer")
    redis_host: str = hp(config.DEFAULT_REDIS_HOST, Mut.RESTART, "CHOCO_REDIS_HOST")
    redis_port: int = hp(config.DEFAULT_REDIS_PORT, Mut.RESTART, "CHOCO_REDIS_PORT")
    redis_db: int = hp(config.DEFAULT_REDIS_DB, Mut.RESTART, "CHOCO_REDIS_DB")


@dataclass
class BoundsConfig:
    """chocofarm/bounds/{eval_bound,info_relaxation}.py — the dual-bound solver, reconstructed per
    invocation so nearly all HOT. `max_inner_states` carries a CORRECTNESS contract (design §7): the
    inner solve ABORTS LOUDLY on the cap, never truncates — the registry stores it as an ordinary
    HOT field but the consuming code keeps its loud abort; do NOT 'tune' it down as a free perf knob."""

    vhat: str = hp("none", Mut.HOT, "V-hat generator: none|zero|analytic|decomp|exact|az-ckpt")
    vhat_lam: typing.Optional[float] = hp(None, Mut.HOT, "reference lambda* fixing V-hat (Route A); None=Route B")
    max_inner_states: int = hp(2_000_000, Mut.HOT, "inner-DP cap; ABORTS LOUDLY, never truncates (design §7) — not a free perf knob")
    lam_lo: float = hp(0.0, Mut.HOT, "lambda-scan bracket low")
    lam_hi: float = hp(0.40, Mut.HOT, "lambda-scan bracket high")
    lam_tol: float = hp(1e-4, Mut.HOT, "bisection convergence tolerance")
    max_iter: int = hp(40, Mut.HOT, "bisection iteration cap")


@dataclass
class ExperimentConfig:
    """The top-level typed contract (design §1). One `ExperimentConfig` per experiment, addressed
    by its `experiment_id`, serialized as one JSON blob (design §5.2). The `schema_version` gates
    the drift check (design §7)."""

    experiment_id: str = ""
    env: EnvConfig = field(default_factory=EnvConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    value: ValueTargetConfig = field(default_factory=ValueTargetConfig)
    feat: FeatureConfig = field(default_factory=FeatureConfig)
    arch: ArchConfig = field(default_factory=ArchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loop: ExItLoopConfig = field(default_factory=ExItLoopConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    par: ParallelConfig = field(default_factory=ParallelConfig)
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    schema_version: int = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Cross-field invariants (design §3.6 step 4). Enforced at decode AND at the write-path's
# pre-write validation, so a bad COMBINATION fails as loudly as a bad single VALUE.
# ---------------------------------------------------------------------------
def check_invariants(cfg: ExperimentConfig) -> None:
    """Raise `RegistryDecodeError` if any cross-field invariant is violated. The one the loop
    already enforces (exit_loop.run, ADR-0002 fail-loud): `td_lambda < 1.0` and `n_step is not
    None` are mutually exclusive — at most one value-target blend may be set."""
    v = cfg.value
    if v.n_step is not None and v.td_lambda < 1.0:
        raise RegistryDecodeError(
            "value-target invariant violated: set at most one of n_step / td_lambda "
            f"(got n_step={v.n_step!r}, td_lambda={v.td_lambda!r}); the other must stay at its "
            "pure-MC default (n_step=None / td_lambda=1.0)")
    if v.n_step is not None and v.n_step < 1:
        raise RegistryDecodeError(f"value.n_step must be >= 1 or None (got {v.n_step!r})")
    if not (0.0 <= v.td_lambda <= 1.0):
        raise RegistryDecodeError(f"value.td_lambda must be in [0, 1] (got {v.td_lambda!r})")
    if cfg.par.workers < 0:
        raise RegistryDecodeError(f"par.workers must be >= 0 (got {cfg.par.workers!r})")
    if cfg.bounds.lam_lo > cfg.bounds.lam_hi:
        raise RegistryDecodeError(
            f"bounds.lam_lo ({cfg.bounds.lam_lo}) must be <= lam_hi ({cfg.bounds.lam_hi})")
    # Positive-count domains. These size loop bounds / weight matrices / sim budgets; a zero or
    # negative value is a config error, not a thing to store silently (a `range(-1)` empty loop, an
    # `m=0` SH bracket, a `hidden=0` net are the "wrong number that looks right" failures §3.6/§7
    # exist to catch). `max_inner_states` is the spec's named correctness-load-bearing knob (§7):
    # the inner DP ABORTS LOUDLY on it, so a <1 cap is meaningless and refused here.
    _positive = {
        "search.m": cfg.search.m, "search.n_sims": cfg.search.n_sims,
        "search.c_outcome": cfg.search.c_outcome,
        "arch.hidden": cfg.arch.hidden,
        "train.epochs": cfg.train.epochs, "train.batch": cfg.train.batch,
        "loop.iters": cfg.loop.iters, "loop.episodes": cfg.loop.episodes,
        "loop.window": cfg.loop.window,
        "eval.eval_n": cfg.eval.eval_n,
        "bounds.max_inner_states": cfg.bounds.max_inner_states, "bounds.max_iter": cfg.bounds.max_iter,
        "env.max_steps": cfg.env.max_steps, "env.present_k": cfg.env.present_k,
    }
    for name, val in _positive.items():
        if val < 1:
            raise RegistryDecodeError(f"{name} must be >= 1 (got {val!r})")
    if cfg.loop.explore_plies < 0:
        raise RegistryDecodeError(f"loop.explore_plies must be >= 0 (got {cfg.loop.explore_plies!r})")
    if cfg.search.max_depth < 1:
        raise RegistryDecodeError(f"search.max_depth must be >= 1 (got {cfg.search.max_depth!r})")
    allowed_vhat = {"none", "zero", "analytic", "decomp", "exact", "az-ckpt"}
    if cfg.bounds.vhat not in allowed_vhat:
        raise RegistryDecodeError(
            f"bounds.vhat={cfg.bounds.vhat!r} not in {sorted(allowed_vhat)}")
    allowed_dtype = {"float32", "float64", "f32", "f64"}
    if cfg.arch.dtype not in allowed_dtype:
        raise RegistryDecodeError(
            f"arch.dtype={cfg.arch.dtype!r} not in {sorted(allowed_dtype)}")


# ---------------------------------------------------------------------------
# Encode — nested dataclass -> JSON-serializable dict (design §5.2 single blob)
# ---------------------------------------------------------------------------
def encode_config(cfg: ExperimentConfig) -> dict:
    """Flatten the nested `ExperimentConfig` to a plain dict ready for `json.dumps`. `Mut` is not
    stored (it is code, not data — it lives in the schema, not the blob), so this is just the value
    tree. `dataclasses.asdict` recurses the groups; everything is already JSON-native (str / int /
    float / bool / None / list)."""
    if not isinstance(cfg, ExperimentConfig):
        raise TypeError(f"encode_config expects an ExperimentConfig, got {type(cfg).__name__}")
    return dataclasses.asdict(cfg)


# ---------------------------------------------------------------------------
# Decode — strict, fail-loud JSON dict -> typed ExperimentConfig (design §3.6)
# ---------------------------------------------------------------------------
def _is_optional(ann):
    """Return (is_optional, inner_type) for `Optional[T]` / `T | None`, else (False, ann)."""
    origin = typing.get_origin(ann)
    if origin in (typing.Union, getattr(types, "UnionType", None)):
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return True, args[0]
    return False, ann


def _check_scalar(value, ann, path: str):
    """Strict type-check + coerce a JSON scalar to the annotated type WITHOUT silent lossy
    coercion. JSON has one number type, so a float field legitimately receives an int (5 → 5.0);
    that widening is exact and allowed. Everything else that mismatches is a loud failure — no
    int←float truncation, no str←number, no bool←int (JSON `true`/`false` decode to bool already,
    and a Python bool is an int subclass so an explicit guard keeps `1` out of a bool field)."""
    is_opt, inner = _is_optional(ann)
    if value is None:
        if is_opt:
            return None
        raise RegistryDecodeError(f"{path}: got null but field is not Optional (type {ann})")
    target = inner

    # list[float] / List[float] etc.
    origin = typing.get_origin(target)
    if origin in (list, typing.List):
        if not isinstance(value, list):
            raise RegistryDecodeError(f"{path}: expected a list, got {type(value).__name__}")
        (elem_ann,) = typing.get_args(target) or (object,)
        return [_check_scalar(v, elem_ann, f"{path}[{i}]") for i, v in enumerate(value)]

    if target is bool:
        if not isinstance(value, bool):
            raise RegistryDecodeError(f"{path}: expected bool, got {type(value).__name__} ({value!r})")
        return value
    if target is int:
        # reject bool (int subclass) and float — only a genuine JSON integer is an int here
        if isinstance(value, bool) or not isinstance(value, int):
            raise RegistryDecodeError(f"{path}: expected int, got {type(value).__name__} ({value!r})")
        return value
    if target is float:
        # a JSON int into a float field is accepted ONLY when the widening is LOSSLESS — exact for
        # |n| <= 2**53 (the float53 mantissa), but an int beyond that loses bits (2**53+1 -> 2**53),
        # which would silently change the value and then slip past the refusal's `live != launched`
        # equality check. Reject the lossy case loudly (ADR-0002) rather than store a wrong number.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RegistryDecodeError(f"{path}: expected float, got {type(value).__name__} ({value!r})")
        if isinstance(value, int) and abs(value) > 2 ** 53:
            raise RegistryDecodeError(
                f"{path}: integer {value!r} cannot widen to float without loss (|n| > 2**53) — "
                "write it as a float literal if that magnitude is intended")
        return float(value)
    if target is str:
        if not isinstance(value, str):
            raise RegistryDecodeError(f"{path}: expected str, got {type(value).__name__} ({value!r})")
        return value
    if target is object:
        return value
    raise RegistryDecodeError(f"{path}: unsupported field annotation {target!r}")


def _resolved_hints(group_cls):
    """Resolve a group dataclass's annotations to real type objects. Under
    `from __future__ import annotations` a field's `.type` is the SOURCE STRING (e.g.
    'typing.Optional[str]'), so the codec must resolve it via `get_type_hints` to type-check
    against the actual annotation rather than parse the string by hand."""
    return typing.get_type_hints(group_cls)


def _decode_group(group_cls, data, path: str):
    """Reconstruct one dataclass group from a dict, strict: every key must be a known field
    (unknown key → loud), every present value is type-checked, missing keys take the dataclass
    default (the field's declared default, NOT a silent fallback — a missing key in a
    schema-versioned blob means 'this field had its default'). Returns an instance of `group_cls`."""
    if not isinstance(data, dict):
        raise RegistryDecodeError(f"{path}: expected an object, got {type(data).__name__}")
    flds = {f.name: f for f in fields(group_cls)}
    hints = _resolved_hints(group_cls)
    for k in data:
        if k not in flds:
            raise RegistryDecodeError(
                f"{path}.{k}: unknown field (not in {group_cls.__name__}; known: {sorted(flds)})")
    kwargs = {}
    for name in flds:
        if name in data:
            kwargs[name] = _check_scalar(data[name], hints[name], f"{path}.{name}")
        # else: leave it to the dataclass default (the declared hp() default)
    return group_cls(**kwargs)


def decode_config(data: dict) -> ExperimentConfig:
    """Strict, fail-loud decode (design §3.6): a plain dict (from `json.loads`) → a validated
    `ExperimentConfig`. Raises `RegistryDecodeError` on any type / domain / unknown-key /
    missing-required / schema-version / cross-field-invariant mismatch. NEVER coerces a malformed
    or missing value to a default silently (ADR-0002)."""
    if not isinstance(data, dict):
        raise RegistryDecodeError(f"top-level: expected an object, got {type(data).__name__}")

    top_fields = {f.name for f in fields(ExperimentConfig)}
    for k in data:
        if k not in top_fields:
            raise RegistryDecodeError(
                f"top-level key {k!r} unknown (known: {sorted(top_fields)}) — schema drift? "
                "the stored blob has a field the code does not (design §7)")

    if "experiment_id" not in data:
        raise RegistryDecodeError("missing required key 'experiment_id'")
    if not isinstance(data["experiment_id"], str) or not data["experiment_id"]:
        raise RegistryDecodeError(
            f"experiment_id must be a non-empty string (got {data['experiment_id']!r})")

    # schema-version drift check (design §7): a stored blob from an older/newer code shape is a
    # LOUD failure, not a forward-compatible default-fill.
    stored_ver = data.get("schema_version", None)
    if stored_ver is None:
        raise RegistryDecodeError("missing required key 'schema_version' (cannot verify schema drift)")
    if stored_ver != SCHEMA_VERSION:
        raise RegistryDecodeError(
            f"schema_version drift: stored blob is v{stored_ver}, code is v{SCHEMA_VERSION} "
            "(design §7) — the code shape changed; re-seed the experiment (init) rather than "
            "decode an old blob into a subtly-wrong config")

    groups = {
        "env": EnvConfig, "search": SearchConfig, "value": ValueTargetConfig,
        "feat": FeatureConfig, "arch": ArchConfig, "train": TrainConfig,
        "loop": ExItLoopConfig, "eval": EvalConfig, "par": ParallelConfig,
        "bounds": BoundsConfig,
    }
    kwargs = {"experiment_id": data["experiment_id"], "schema_version": stored_ver}
    for name, cls in groups.items():
        kwargs[name] = _decode_group(cls, data.get(name, {}), name)

    cfg = ExperimentConfig(**kwargs)
    check_invariants(cfg)
    return cfg
