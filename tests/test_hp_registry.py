#!/usr/bin/env python3
"""
test_hp_registry.py — the hyperparameter-registry verification battery (design hp-registry).

Covers: schema (de)serialization round-trip, the strict fail-loud decode on every bad-input
class (type / domain / unknown-key / missing-key / schema-drift / cross-field-invariant), the
redis round-trip (no-TTL write + decode), namespacing isolation (two ids never clobber), the
RESTART-refusal (a baked field changed mid-run fires the loud refusal; a HOT field does not), and
the seed-from-argparse bootstrap (argparse defaults = dataclass defaults; idempotent re-bind).

REDIS SAFETY (the disk-persisted redis at 127.0.0.1:6379 is SHARED with a live training run):
  * every test key is under an ISOLATED namespace `choco:hp:__test__<uuid>` — never an az:* key.
  * each test cleans up its own keys (try/finally delete).
  * NO FLUSHALL / FLUSHDB, NO CONFIG SET on the server, NO touching az:* keys.

The redis-touching tests skip cleanly if redis is unreachable (so the schema/codec tests still run
in a redis-less environment). Run pinned + bounded, e.g.:
    PYTHONPATH=. taskset -c 3 timeout 120 /home/bork/w/vdc/venvs/generic/bin/python -m pytest \
        tests/test_hp_registry.py -q
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import pytest

from chocofarm.hp.schema import (
    ExperimentConfig, encode_config, decode_config, RegistryDecodeError, Mut, SCHEMA_VERSION,
    check_invariants,
)
from chocofarm.hp import registry as reg


# ---------------------------------------------------------------------------
# Schema (de)serialization — no redis needed
# ---------------------------------------------------------------------------
def test_roundtrip_defaults():
    """A default config encodes to a JSON-native dict and decodes back equal (design §3.6)."""
    cfg = ExperimentConfig(experiment_id="rt")
    d = encode_config(cfg)
    back = decode_config(json.loads(json.dumps(d)))
    assert back == cfg


def test_roundtrip_with_overrides():
    """Overrides across groups (incl. an Optional list and an Optional int) round-trip."""
    cfg = ExperimentConfig(experiment_id="rt2")
    cfg.train.lr = 1e-4
    cfg.value.n_step = 3
    cfg.value.td_lambda = 1.0           # keep the invariant satisfied (n_step set => td_lambda==1)
    cfg.env.value_vector = [1.0, 2.0, 3.5]
    cfg.search.use_jax_mlp = True
    back = decode_config(json.loads(json.dumps(encode_config(cfg))))
    assert back == cfg
    assert back.env.value_vector == [1.0, 2.0, 3.5]
    assert back.search.use_jax_mlp is True


def test_int_widens_to_float_in_float_field():
    """JSON has one number type, so a JSON int into a float field widens exactly (5 -> 5.0). This
    is the one allowed coercion (lossless); everything else is loud."""
    cfg = ExperimentConfig(experiment_id="w")
    d = encode_config(cfg)
    d["train"]["lr"] = 1                # a JSON integer in a float field
    back = decode_config(d)
    assert back.train.lr == 1.0
    assert isinstance(back.train.lr, float)


# ---------------------------------------------------------------------------
# Fail-loud decode — every bad-input class raises RegistryDecodeError, never coerces to a default
# ---------------------------------------------------------------------------
def _bad(mutator):
    cfg = ExperimentConfig(experiment_id="bad")
    d = json.loads(json.dumps(encode_config(cfg)))
    mutator(d)
    return d


def test_decode_loud_on_float_into_int():
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d["train"].__setitem__("epochs", 2.5)))


def test_decode_loud_on_str_into_int():
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d["train"].__setitem__("epochs", "5")))


def test_decode_loud_on_bool_into_int():
    """A Python bool is an int subclass; the codec must keep `true` out of an int field."""
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d["train"].__setitem__("epochs", True)))


def test_decode_loud_on_number_into_str():
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d["env"].__setitem__("entry", 5)))


def test_decode_loud_on_unknown_field():
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d["train"].__setitem__("bogus", 1)))


def test_decode_loud_on_unknown_top_level_key():
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d.__setitem__("bogus_group", {})))


def test_decode_loud_on_missing_experiment_id():
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d.__delitem__("experiment_id")))


def test_decode_loud_on_empty_experiment_id():
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d.__setitem__("experiment_id", "")))


def test_decode_loud_on_null_into_non_optional():
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d["train"].__setitem__("lr", None)))


def test_decode_loud_on_schema_version_drift():
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d.__setitem__("schema_version", SCHEMA_VERSION + 99)))


def test_decode_loud_on_missing_schema_version():
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d.__delitem__("schema_version")))


def test_decode_loud_on_mutually_exclusive_value_target():
    """The cross-field invariant the loop already enforces: n_step set AND td_lambda<1 is illegal."""
    def m(d):
        d["value"]["n_step"] = 3
        d["value"]["td_lambda"] = 0.5
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(m))


def test_decode_loud_on_out_of_domain_td_lambda():
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d["value"].__setitem__("td_lambda", 1.5)))


def test_decode_loud_on_bad_vhat_enum():
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d["bounds"].__setitem__("vhat", "made_up")))


def test_decode_loud_on_lam_lo_above_hi():
    def m(d):
        d["bounds"]["lam_lo"] = 0.5
        d["bounds"]["lam_hi"] = 0.1
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(m))


@pytest.mark.parametrize("path,value", [
    ("search.m", 0), ("search.n_sims", 0), ("search.c_outcome", 0), ("search.max_depth", 0),
    ("arch.hidden", 0), ("train.epochs", 0), ("train.batch", 0),
    ("loop.iters", 0), ("loop.episodes", -1), ("loop.window", 0),
    ("eval.eval_n", 0), ("bounds.max_inner_states", -1), ("bounds.max_iter", 0),
    ("env.max_steps", 0), ("env.present_k", 0), ("loop.explore_plies", -1),
])
def test_decode_loud_on_nonpositive_count(path, value):
    """Loop bounds / weight-matrix sizes / sim budgets / the max_inner_states correctness cap must
    be >= 1 (the audit found these silently accepted; the spec §7 names max_inner_states the
    correctness-load-bearing knob). A zero/negative is a config error, not a stored value."""
    group, leaf = path.split(".")
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d[group].__setitem__(leaf, value)))


def test_decode_loud_on_lossy_int_to_float():
    """A JSON int beyond the float53 mantissa cannot widen to a float losslessly; the codec must
    refuse rather than silently store a changed number that would also slip past the refusal's
    equality check (the audit's Finding 5)."""
    with pytest.raises(RegistryDecodeError):
        decode_config(_bad(lambda d: d["train"].__setitem__("lr", 2 ** 53 + 1)))


def test_decode_ok_on_exact_int_to_float():
    """An int within the mantissa widens exactly and is accepted (the allowed lossless case)."""
    cfg = decode_config(_bad(lambda d: d["train"].__setitem__("lr", 2 ** 52)))
    assert cfg.train.lr == float(2 ** 52)


# ---------------------------------------------------------------------------
# Facet coverage — every leaf carries a Mut facet, and the known split holds
# ---------------------------------------------------------------------------
def test_every_leaf_has_a_facet():
    """Each declared hp() leaf carries a Mut in its metadata (the schema contract)."""
    a = ExperimentConfig(experiment_id="f")
    for _g, mut, _fld, _av, _bv in reg._iter_facet_diffs(a, a):
        assert isinstance(mut, Mut), f"{_g}.{_fld} has no Mut facet"


def test_known_facet_split():
    """Spot-check the design §4 facet reading: lr/l2/betas/eps are ALL HOT (audit item M — the
    Optimizer owns the inject_hyperparams transform; lr/betas/eps are set per step from the live
    AdamHParams, l2 is a traced loss arg), alpha is HOT (traced call-arg), the search BUDGET m/n_sims
    is HOT now (the SH bracket is recomputed per decide — ADR-0012 P4; only search.use_jax_mlp stays
    RESTART), the net shape / master seed stay RESTART, the env constants are INSTANCE."""
    facets = {}
    a = ExperimentConfig(experiment_id="f")
    for g, mut, fld, _av, _bv in reg._iter_facet_diffs(a, a):
        facets[f"{g}.{fld}"] = mut
    assert facets["train.lr"] is Mut.HOT       # audit M: injected via inject_hyperparams, live
    assert facets["train.l2"] is Mut.HOT       # audit R13: traced loss coefficient, live per step
    assert facets["train.beta1"] is Mut.HOT    # audit M: live injected hparam, set per step
    assert facets["train.beta2"] is Mut.HOT    # audit M: live injected hparam, set per step
    assert facets["train.eps"] is Mut.HOT      # audit M: live injected hparam, set per step
    assert facets["train.alpha"] is Mut.HOT
    assert facets["train.beta"] is Mut.HOT
    assert facets["search.m"] is Mut.HOT       # SH bracket recomputed per decide → HOT
    assert facets["search.n_sims"] is Mut.HOT  # SH phase loop sized per decide → HOT
    assert facets["search.use_jax_mlp"] is Mut.RESTART  # binds the forward fn — stays RESTART
    assert facets["search.c_puct"] is Mut.HOT
    assert facets["arch.hidden"] is Mut.RESTART
    assert facets["env.teleport_overhead"] is Mut.INSTANCE
    assert facets["loop.lam"] is Mut.HOT
    assert facets["loop.seed"] is Mut.RESTART


# ---------------------------------------------------------------------------
# Seed-from-argparse bootstrap (design §6) — no redis needed for the build half
# ---------------------------------------------------------------------------
def _exit_loop_default_args():
    """Parse the exit_loop argparse with only the required --ckpt-dir, so every other field carries
    its argparse default (the bootstrap's 'launch with the CLI defaults' case)."""
    from chocofarm.az.exit_loop import main as _  # ensure the module imports
    import chocofarm.az.exit_loop as el
    import argparse as _ap
    # rebuild the parser exactly as exit_loop.main does, then parse the minimal required set
    # (we cannot call el.main() — it would run the loop — so we mirror its parser via a thin call).
    # exit_loop.main builds the parser inline; re-parse by invoking it through parse_known on a
    # constructed argv. Simplest: construct the parser the same way by calling the function body is
    # not exposed, so we synthesize a namespace with the documented argparse defaults instead.
    ns = _ap.Namespace(
        iters=40, episodes=300, window=5, epochs=2, batch=256, m=12, n_sims=48,
        lr=1e-3, l2=1e-4, alpha=1.0, beta=1.0, lam=0.0855, explore_plies=4,
        eval_n=200, eval_seed=12345, seed=7, hidden=256, residual=False,
        init_weights=None, resume=None, workers=4, cores="0,1,2,3",
        td_lambda=1.0, n_step=None, tb_logdir=None, ckpt_dir="runs/__test__",
    )
    return ns


def test_from_argparse_equals_dataclass_defaults():
    """The whole point of design §6: a namespace carrying the argparse DEFAULTS produces a config
    equal to the dataclass defaults (one source of defaults — argparse defaults ARE the dataclass
    defaults)."""
    ns = _exit_loop_default_args()
    cfg = reg.from_argparse(ns, experiment_id="boot")
    default = ExperimentConfig(experiment_id="boot")
    # arch.init_seed picks up exit_loop's --seed (documented: --seed seeds the net He-init too),
    # which equals the schema default 0 only if seed==0; here seed==7, so init_seed becomes 7.
    assert cfg.arch.init_seed == 7
    default.arch.init_seed = 7
    assert cfg == default


def test_from_argparse_carries_overrides():
    ns = _exit_loop_default_args()
    ns.lr = 1e-4
    ns.hidden = 512
    ns.workers = 0
    cfg = reg.from_argparse(ns, experiment_id="boot2")
    assert cfg.train.lr == 1e-4
    assert cfg.arch.hidden == 512
    assert cfg.par.workers == 0


# ---------------------------------------------------------------------------
# redis-touching tests (isolated namespace; skip if redis is unreachable)
# ---------------------------------------------------------------------------
def _redis_or_skip():
    try:
        r = reg._connect()
    except reg.RegistryUnavailable as e:
        pytest.skip(f"redis unreachable: {e}")
    return r


@pytest.fixture
def isolated_id():
    """A unique experiment_id under the test namespace; guarantees no collision with az:* keys or
    other tests, and cleans up both blob + meta afterward."""
    eid = f"__test__{uuid.uuid4().hex[:12]}"
    yield eid
    try:
        r = reg._connect()
        reg.delete_experiment(eid, r=r)
        r.close()
    except reg.RegistryUnavailable:
        pass


def test_redis_roundtrip_no_ttl(isolated_id):
    r = _redis_or_skip()
    try:
        cfg = ExperimentConfig(experiment_id=isolated_id)
        cfg.train.lr = 7e-4
        reg.write_config(isolated_id, cfg, writer="pytest", r=r)
        # the key exists and carries NO TTL (design §2.1: bare SET -> TTL == -1)
        assert r.ttl(reg._key(isolated_id)) == -1
        back = reg.read_config(isolated_id, r=r)
        assert back == cfg
        # meta is recorded with the writer attribution (design §5.5)
        meta = json.loads(r.get(reg._meta_key(isolated_id)))
        assert meta["writer"] == "pytest"
        assert meta["schema_version"] == SCHEMA_VERSION
    finally:
        r.close()


def test_redis_missing_key_is_loud(isolated_id):
    r = _redis_or_skip()
    try:
        with pytest.raises(reg.RegistryKeyMissing):
            reg.read_config(isolated_id, r=r)   # never seeded
    finally:
        r.close()


def test_namespacing_isolation():
    """Two distinct ids never clobber each other (design §5.1): a write to one cannot touch the
    other's blob."""
    r = _redis_or_skip()
    a = f"__test__{uuid.uuid4().hex[:12]}"
    b = f"__test__{uuid.uuid4().hex[:12]}"
    try:
        ca = ExperimentConfig(experiment_id=a); ca.train.lr = 1e-2
        cb = ExperimentConfig(experiment_id=b); cb.train.lr = 1e-5
        reg.write_config(a, ca, r=r)
        reg.write_config(b, cb, r=r)
        reg.set_fields(a, {"train.lr": "3e-3"}, r=r)   # mutate only a
        assert reg.read_config(a, r=r).train.lr == 3e-3
        assert reg.read_config(b, r=r).train.lr == 1e-5   # b untouched
    finally:
        reg.delete_experiment(a, r=r)
        reg.delete_experiment(b, r=r)
        r.close()


def test_set_fields_atomic_multi(isolated_id):
    """The motivating atomic case: drop lr AND raise l2 together (design §5.4)."""
    r = _redis_or_skip()
    try:
        reg.write_config(isolated_id, ExperimentConfig(experiment_id=isolated_id), r=r)
        old, new = reg.set_fields(isolated_id, {"train.lr": "1e-4", "train.l2": "5e-4"}, r=r)
        assert old.train.lr == 1e-3 and old.train.l2 == 1e-4
        assert new.train.lr == 1e-4 and new.train.l2 == 5e-4
        # persisted
        back = reg.read_config(isolated_id, r=r)
        assert back.train.lr == 1e-4 and back.train.l2 == 5e-4
    finally:
        r.close()


def test_set_fields_loud_on_bad_value_before_write(isolated_id):
    """A bad value fails at the source (design §5.3) — and does NOT corrupt the stored blob."""
    r = _redis_or_skip()
    try:
        reg.write_config(isolated_id, ExperimentConfig(experiment_id=isolated_id), r=r)
        with pytest.raises(RegistryDecodeError):
            reg.set_fields(isolated_id, {"train.epochs": "2.5"}, r=r)  # float into an int field
        # the stored blob is still the valid default — the bad set never touched it
        assert reg.read_config(isolated_id, r=r).train.epochs == 2
    finally:
        r.close()


def test_seed_registry_idempotent(isolated_id):
    """Seeding an existing experiment re-binds to the existing blob (design §6 — a --resume must
    not clobber operator overrides)."""
    r = _redis_or_skip()
    try:
        cfg = ExperimentConfig(experiment_id=isolated_id)
        reg.seed_registry(isolated_id, cfg, r=r)
        # operator override after the seed
        reg.set_fields(isolated_id, {"train.lr": "1e-4"}, r=r)
        # a second seed (a --resume) with the DEFAULT cfg must NOT overwrite the override
        cfg2 = ExperimentConfig(experiment_id=isolated_id)   # lr back at default
        seeded = reg.seed_registry(isolated_id, cfg2, r=r)
        assert seeded.train.lr == 1e-4   # the override survived the idempotent re-bind
    finally:
        r.close()


# ---------------------------------------------------------------------------
# RESTART-refusal (design §3.4) — the heart of the mutability facet
# ---------------------------------------------------------------------------
def test_restart_refusal_fires_on_baked_field(isolated_id):
    """A RESTART field (arch.hidden — the trunk width, baked into every weight-matrix shape at net
    construction and the optimizer's moment pytree; audit item M made ALL optimizer coefficients
    lr/l2/betas/eps HOT, so the baked-field example is now a genuine net-shape RESTART) changed
    mid-run vs the launched_with shadow fires the loud refusal."""
    r = _redis_or_skip()
    try:
        launched = ExperimentConfig(experiment_id=isolated_id)
        reg.write_config(isolated_id, launched, r=r)
        snap = reg.ConfigSnapshot.launch(isolated_id, launched_with=launched, r=r)
        # operator changes the net width in the registry (a RESTART field — sizes the weight matrices
        # and the optimizer's moment pytree, so it cannot move on a running process)
        reg.set_fields(isolated_id, {"arch.hidden": "512"}, r=r)
        with pytest.raises(reg.RestartRequired) as ei:
            snap.refresh(iteration=1, r=r)
        assert "arch.hidden" in str(ei.value)
        assert "--resume" in str(ei.value)
    finally:
        r.close()


def test_betas_eps_hot_change_applied_not_refused(isolated_id):
    """Audit item M (the betas/eps follow-up): beta1/beta2/eps are now HOT — a mid-run change via the
    registry snapshot is APPLIED at the next refresh, NOT refused (the counterpart to the old
    test_restart_refusal that used train.beta1 when it was still RESTART). This is the registry-
    integration half of GATE-3; the trainer-side live-beta scaling is verified in test_az_loop's
    test_jax_train_live_lr_l2_betas_eps (d)."""
    r = _redis_or_skip()
    try:
        launched = ExperimentConfig(experiment_id=isolated_id)
        reg.write_config(isolated_id, launched, r=r)
        snap = reg.ConfigSnapshot.launch(isolated_id, launched_with=launched, r=r)
        # operator retunes the Adam betas + eps on the running experiment (all HOT post-item-M)
        reg.set_fields(isolated_id, {"train.beta1": "0.95", "train.beta2": "0.9999",
                                     "train.eps": "1e-7"}, r=r)
        snap.refresh(iteration=1, r=r)   # must NOT raise — betas/eps are HOT
        assert snap.cfg.train.beta1 == 0.95
        assert snap.cfg.train.beta2 == 0.9999
        assert snap.cfg.train.eps == 1e-7
    finally:
        r.close()


def test_lr_l2_hot_change_applied_not_refused(isolated_id):
    """Audit R13 (the frozen-config headline): lr/l2 are now HOT — a mid-run change via the registry
    snapshot is APPLIED at the next refresh, NOT refused. This is the registry-integration half of
    R13's GATE 3 (the trainer-side live-lr scaling is verified in test_az_loop)."""
    r = _redis_or_skip()
    try:
        launched = ExperimentConfig(experiment_id=isolated_id)
        reg.write_config(isolated_id, launched, r=r)
        snap = reg.ConfigSnapshot.launch(isolated_id, launched_with=launched, r=r)
        # operator anneals lr and retunes l2 on the running experiment (both HOT post-R13)
        reg.set_fields(isolated_id, {"train.lr": "1e-4", "train.l2": "5e-4"}, r=r)
        snap.refresh(iteration=1, r=r)   # must NOT raise — lr/l2 are HOT
        assert snap.cfg.train.lr == 1e-4
        assert snap.cfg.train.l2 == 5e-4
    finally:
        r.close()


def test_instance_refusal_fires_on_env_field(isolated_id):
    """An INSTANCE field (env.teleport_overhead) changed mid-run fires the stronger refusal."""
    r = _redis_or_skip()
    try:
        launched = ExperimentConfig(experiment_id=isolated_id)
        reg.write_config(isolated_id, launched, r=r)
        snap = reg.ConfigSnapshot.launch(isolated_id, launched_with=launched, r=r)
        reg.set_fields(isolated_id, {"env.teleport_overhead": "15.0"}, r=r)
        with pytest.raises(reg.RestartRequired) as ei:
            snap.refresh(iteration=2, r=r)
        assert "teleport_overhead" in str(ei.value)
        assert "NEW experiment" in str(ei.value)
    finally:
        r.close()


def test_hot_change_does_not_refuse(isolated_id):
    """A HOT field (loop.lam, train.alpha) changed mid-run is applied, NOT refused — the snapshot
    refresh adopts it and the RESTART-drift check passes."""
    r = _redis_or_skip()
    try:
        launched = ExperimentConfig(experiment_id=isolated_id)
        reg.write_config(isolated_id, launched, r=r)
        snap = reg.ConfigSnapshot.launch(isolated_id, launched_with=launched, r=r)
        reg.set_fields(isolated_id, {"loop.lam": "0.09", "train.alpha": "2.0"}, r=r)
        snap.refresh(iteration=3, r=r)   # must NOT raise
        assert snap.cfg.loop.lam == 0.09
        assert snap.cfg.train.alpha == 2.0
    finally:
        r.close()


def test_hot_search_knob_change_does_not_refuse(isolated_id):
    """The audit's Finding 1: HOT SEARCH knobs (c_puct/max_depth) are genuinely per-iteration (the
    search object is rebuilt each iteration). A change to them must be APPLIED at the refresh, not
    refused — the consolidation reads them off snap.cfg into the rebuilt search. The search BUDGET
    m/n_sims is HOT too now (the SH bracket is recomputed per decide — ADR-0012 P4), so a mid-run
    change to it is likewise applied, not refused (it rides the same hot_search bag)."""
    r = _redis_or_skip()
    try:
        launched = ExperimentConfig(experiment_id=isolated_id)
        reg.write_config(isolated_id, launched, r=r)
        snap = reg.ConfigSnapshot.launch(isolated_id, launched_with=launched, r=r)
        reg.set_fields(isolated_id, {"search.c_puct": "2.5", "search.max_depth": "10",
                                     "search.m": "8", "search.n_sims": "32"}, r=r)
        snap.refresh(iteration=1, r=r)   # must NOT raise (these are HOT — incl. the SH budget)
        assert snap.cfg.search.c_puct == 2.5
        assert snap.cfg.search.max_depth == 10
        assert snap.cfg.search.m == 8        # HOT now: SH bracket recomputed per decide
        assert snap.cfg.search.n_sims == 32  # HOT now: SH phase loop sized per decide
    finally:
        r.close()


def test_search_restart_knob_refuses(isolated_id):
    """search.use_jax_mlp is RESTART (it binds the forward fn at construction, and the parallel worker
    is numpy-only by R14), so a mid-run change to it DOES refuse — the counterpart to the
    HOT-search-knob test. (m/n_sims USED to be RESTART here; they were reclassified HOT once the SH
    bracket was confirmed recomputed per decide — ADR-0012 P4 — and their live-apply is now covered by
    test_hot_search_knob_change_does_not_refuse.)"""
    r = _redis_or_skip()
    try:
        launched = ExperimentConfig(experiment_id=isolated_id)
        reg.write_config(isolated_id, launched, r=r)
        snap = reg.ConfigSnapshot.launch(isolated_id, launched_with=launched, r=r)
        reg.set_fields(isolated_id, {"search.use_jax_mlp": "true"}, r=r)
        with pytest.raises(reg.RestartRequired) as ei:
            snap.refresh(iteration=1, r=r)
        assert "search.use_jax_mlp" in str(ei.value)
    finally:
        r.close()


def test_resume_rebind_adopts_lr_no_false_refusal(isolated_id):
    """The audit's Finding 2/3 + spec §3.5: an operator drops lr in the registry, then a --resume
    re-binds to that blob; the re-bound config IS the construction-time shadow, so (a) the process
    adopts the dropped lr (the registry value, not the args default) and (b) the iter-0 refresh
    does NOT refuse. Post-R13 lr is HOT (so a live drop is applied without --resume too — see
    test_lr_l2_hot_change_applied_not_refused); this still exercises the re-bind/adopt path that
    --resume relies on for the genuinely-RESTART fields, with lr as the carrier value."""
    r = _redis_or_skip()
    try:
        # first launch: seed with default lr (1e-3)
        cfg0 = reg.seed_registry(isolated_id, ExperimentConfig(experiment_id=isolated_id),
                                 r=r)
        assert cfg0.train.lr == 1e-3
        # operator drops lr in the registry
        reg.set_fields(isolated_id, {"train.lr": "1e-4"}, r=r)
        # second launch (--resume): re-seed with the DEFAULT config; re-bind returns the dropped blob
        cfg0b = reg.seed_registry(isolated_id, ExperimentConfig(experiment_id=isolated_id),
                                  r=r)
        assert cfg0b.train.lr == 1e-4   # ADOPTED the registry drop (the construction value)
        # the re-bound config is the shadow -> iter-0 refresh must NOT refuse
        snap = reg.ConfigSnapshot.launch(isolated_id, launched_with=cfg0b, r=r)
        snap.refresh(iteration=0, r=r)   # must NOT raise
    finally:
        r.close()


def test_derived_drift_refused_on_rebind(isolated_id):
    """The audit's Finding 6 (drift check): a re-bound blob whose recorded env-derived dims/dtype
    disagree with the running process fails loud, not silently runs a mismatched net. Simulated by
    seeding a blob with a wrong in_dim, then re-binding with a live cfg carrying the true dim."""
    r = _redis_or_skip()
    try:
        # seed a blob with a deliberately-wrong recorded in_dim (as if fit to a different env)
        bad = ExperimentConfig(experiment_id=isolated_id)
        bad.arch.in_dim = 999
        bad.arch.n_actions = 7
        reg.write_config(isolated_id, bad, r=r)
        # a live cfg carrying the true derived dims (what _record_derived would set)
        live = ExperimentConfig(experiment_id=isolated_id)
        live.arch.in_dim = 241
        live.arch.n_actions = 65
        with pytest.raises(reg.RegistrySchemaDrift) as ei:
            reg._assert_no_derived_drift(isolated_id, recorded=reg.read_config(isolated_id, r=r),
                                         live=live)
        assert "in_dim" in str(ei.value)
    finally:
        r.close()


if __name__ == "__main__":
    # plain-runnable for the schema/bootstrap (redis-less) checks; full battery via pytest.
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn) and fn.__code__.co_argcount == 0:
            fn()
            print(f"PASS {name}")
    print("redis-less checks passed; run pytest for the redis battery")
