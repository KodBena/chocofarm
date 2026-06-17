#!/usr/bin/env python3
"""
test_feature_layout.py — pins the §2.2 FeatureLayout descriptor (audit R6) against the writers.

The feature-vector layout used to be hand-kept in sync in three independent places (the builder's
positional writes, actions.py's literal slice offsets, and feature_response's name/tag list) — a
block reorder in one silently MISLABELED the others, and feature_response had ZERO coverage. R6
consolidated all three onto one owner, `FeatureLayout`. This test is the coverage that was missing:
it asserts the layout's dim/slices/names/tags match exactly what the consumers historically used,
so a future reorder can no longer drift silently.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from chocofarm.model.env import Environment
from chocofarm.az.features import FeatureLayout, feature_dim


def test_layout_dim_matches_feature_dim():
    """The layout is the single source of the vector length."""
    env = Environment()
    assert FeatureLayout(env).dim == feature_dim(env) == 241


def test_layout_slices_match_historical_offsets():
    """available at 2N..3N and informative at 5N..5N+nD — the literals actions.py used to hardcode
    (a regression here is exactly the silent-mislabel bug R6 closes)."""
    env = Environment()
    N, nD = env.N, len(env.detectors)
    layout = FeatureLayout(env)
    assert layout["available"] == slice(2 * N, 3 * N)
    assert layout["informative"] == slice(5 * N, 5 * N + nD)


def test_element_names_and_tags_cover_every_position():
    """One name and one tag per feature position, for all `dim` positions."""
    env = Environment()
    layout = FeatureLayout(env)
    assert len(layout.element_names()) == feature_dim(env)
    assert len(layout.block_tags()) == feature_dim(env)


def test_element_name_spot_checks():
    """Spot-check the element names at block boundaries — this is the output feature_response
    emits, which had no test of its own before R6."""
    env = Environment()
    N = env.N
    layout = FeatureLayout(env)
    names = layout.element_names()
    # first per-treasure element
    assert names[0] == "t0.marg"
    assert names[N - 1] == f"t{N - 1}.marg"
    # first per-detector element starts the 6th per-treasure-wide block (offset 5N)
    assert names[5 * N] == "d0.informative"
    # the six global scalars appear in order, immediately after the per-detector block (5N + 3nD)
    nD = len(env.detectors)
    g0 = 5 * N + 3 * nD
    assert names[g0:g0 + 6] == [
        "global.log|bw|", "global.n_collected", "global.sum_marg",
        "global.exit_cost", "global.nonempty", "global.sum_unc",
    ]
    # per-teleport names close the vector
    n_tel = len(env.teleports)
    assert names[-n_tel:] == [f"global.tele_dist{k}" for k in range(n_tel)]


def test_full_element_names_and_tags_pin_every_block():
    """FULL golden pin of every element name and tag (not just boundaries). Spot-checks at block
    edges miss an interior reorder (e.g. swapping the per-treasure `collected`↔`dist_t` channels
    keeps every boundary intact); this reconstructs the complete historical list from the canonical
    block table and asserts the layout reproduces it position-for-position, so ANY reorder — interior
    or boundary — fails here. This is the exact byte-labeling feature_response emits and never tested."""
    env = Environment()
    N, nD, n_tel = env.N, len(env.detectors), len(env.teleports)
    layout = FeatureLayout(env)

    # The canonical §2.2 element-name / tag list, rebuilt independently of FeatureLayout's internals
    # (the order here IS the contract feature_response historically produced).
    exp_names, exp_tags = [], []
    for disp in ["marg", "collected", "available", "dist", "unc"]:
        for i in range(N):
            exp_names.append(f"t{i}.{disp}"); exp_tags.append(f"treasure/{disp}")
    for disp in ["informative", "p_pos", "dist"]:
        for j in range(nD):
            exp_names.append(f"d{j}.{disp}"); exp_tags.append(f"detector/{disp}")
    for disp in ["log|bw|", "n_collected", "sum_marg", "exit_cost", "nonempty", "sum_unc"]:
        exp_names.append(f"global.{disp}"); exp_tags.append("global")
    for k in range(n_tel):
        exp_names.append(f"global.tele_dist{k}"); exp_tags.append("global")

    assert layout.element_names() == exp_names
    assert layout.block_tags() == exp_tags


# ---------------------------------------------------------------------------
# FeatureConfig per-group pins vs the FeatureLayout SSOT (audit item G)
# ---------------------------------------------------------------------------
# hp/schema.py's FeatureConfig holds a fourth, provenance-only copy of the per-group channel counts
# (per_treasure=5 / per_detector=3 / global=6). FeatureLayout (features.py) is the SSOT after R6;
# the registry's drift check pinned only the TOTAL in_dim (241), so a layout change that updated
# FeatureLayout but not FeatureConfig could keep the total equal while a per-group width drifted —
# silently. Item G derives the per-group widths from FeatureLayout.blocks (not a second hardcode of
# 5/3/6) and fails loud if FeatureConfig disagrees, on BOTH the fresh-seed path (_record_derived)
# and the re-bind drift check. These tests gate it.
def test_feature_group_channels_derived_from_layout_match_schema():
    """(a) The per-group widths derived from FeatureLayout.blocks equal both the FeatureConfig
    provenance counts AND the expected 5/3/6 — and the pin check passes on the real, in-sync config
    (returns no drift)."""
    from chocofarm.hp.schema import ExperimentConfig
    from chocofarm.hp import registry as reg
    env = Environment()
    derived = reg._feature_group_channels(env)
    assert derived == {
        "per_treasure_channels": 5, "per_detector_channels": 3, "global_channels": 6,
    }
    cfg = ExperimentConfig(experiment_id="g")
    # the provenance copy in the schema matches what the layout produces
    assert cfg.feat.per_treasure_channels == derived["per_treasure_channels"]
    assert cfg.feat.per_detector_channels == derived["per_detector_channels"]
    assert cfg.feat.global_channels == derived["global_channels"]
    # the pin check reports no drift on the in-sync config, and the re-bind drift assertion (env in
    # hand triggers the per-group check) passes without raising.
    assert reg._assert_feature_config_pins(cfg, env) == []
    reg._assert_no_derived_drift("g", recorded=cfg, live=cfg, env=env)


@pytest.mark.parametrize("field_name", [
    "per_treasure_channels", "per_detector_channels", "global_channels",
])
def test_feature_group_channel_drift_fires_loud_on_rebind(field_name):
    """(b) A FeatureConfig per-group count deliberately out of sync with the FeatureLayout SSOT
    fires the loud RegistrySchemaDrift (ADR-0002) on the re-bind drift check, naming the mismatched
    group — the silent drift the total-only in_dim check would miss."""
    from chocofarm.hp.schema import ExperimentConfig
    from chocofarm.hp import registry as reg
    env = Environment()
    good = ExperimentConfig(experiment_id="g")
    bad = ExperimentConfig(experiment_id="g")
    setattr(bad.feat, field_name, getattr(bad.feat, field_name) + 1)  # monkeypatch one count
    with pytest.raises(reg.RegistrySchemaDrift) as ei:
        reg._assert_no_derived_drift("g", recorded=good, live=bad, env=env)
    assert f"feat.{field_name}" in str(ei.value)


@pytest.mark.parametrize("field_name", [
    "per_treasure_channels", "per_detector_channels", "global_channels",
])
def test_feature_group_channel_drift_fires_loud_on_fresh_seed(field_name):
    """(c) The fresh-seed path: _record_derived (which runs on the FIRST seed of an experiment, not
    only on re-bind) ALSO fires the loud RegistrySchemaDrift when a FeatureConfig per-group pin is
    stale — so a layout change shipped against a stale schema is caught when the net is first bound
    to the env, not silently written and only caught on a later re-bind."""
    from chocofarm.hp.schema import ExperimentConfig
    from chocofarm.hp import registry as reg
    env = Environment()
    bad = ExperimentConfig(experiment_id="g")
    setattr(bad.feat, field_name, getattr(bad.feat, field_name) + 1)  # monkeypatch one count
    with pytest.raises(reg.RegistrySchemaDrift) as ei:
        reg._record_derived(bad, env)   # the fresh-seed (and re-bind) derived-fact recorder
    assert f"feat.{field_name}" in str(ei.value)


def test_feature_group_width_mislabel_fires_loud():
    """The block-count→channel-count derivation assumes each group's blocks carry that group's unit
    width (treasure=N, detector=nD, global=1). A block mislabeled with the wrong group (here a
    width-N block tagged global) would otherwise inflate the count and let a wrong pin satisfy the
    check silently. The derive asserts the width invariant and fails loud (ADR-0002) instead."""
    import chocofarm.az.features as feat
    from chocofarm.hp import registry as reg
    env = Environment()
    orig_init = feat.FeatureLayout.__init__

    def mislabel_init(self, e):
        orig_init(self, e)
        # append a width-N block but tag it group="global" — a per-treasure-sized quantity
        # masquerading as one global scalar; the count-only derive would report global_channels+1.
        self.blocks.append(("rogue", e.N, "global", "rogue"))

    feat.FeatureLayout.__init__ = mislabel_init
    feat._LAYOUTS.clear()   # the layout is memoized per-env; drop the cached good one
    try:
        with pytest.raises(reg.RegistrySchemaDrift) as ei:
            reg._feature_group_channels(env)
        assert "rogue" in str(ei.value) and "width" in str(ei.value)
    finally:
        feat.FeatureLayout.__init__ = orig_init
        feat._LAYOUTS.clear()   # evict the mislabeled layout so other tests see the real one


# ---------------------------------------------------------------------------
# Cross-language drift: the checked-in feature_layout.json the C++ runtime-reads
# ---------------------------------------------------------------------------
# The C++ FeatureBuilder runtime-reads chocofarm/data/feature_layout.json (the ordered (key, width)
# block table + dim) to build its named slices, instead of re-encoding the layout as a positional
# `o += N` offset ladder (ADR-0012 P7 — the C++ re-derives nothing; the single-source the runtime-read
# buys). That artifact is GENERATED from FeatureLayout.spec(), so it must not drift from the one owner;
# this is the same fail-loud SSOT-net idiom as tests/test_wire_drift.py. Regenerate after a layout
# change with:
#   PYTHONPATH=. python -c "import json; from chocofarm.model.env import Environment; \
#     from chocofarm.az.features import FeatureLayout; \
#     open('chocofarm/data/feature_layout.json','w').write( \
#       json.dumps(FeatureLayout(Environment()).spec(), indent=2) + '\n')"
_SPEC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chocofarm", "data", "feature_layout.json",
)


def test_checked_in_layout_spec_matches_owner():
    """The C++-read feature_layout.json equals FeatureLayout.spec() for the live env — so the artifact
    the C++ slices against cannot drift from the FeatureLayout SSOT (ADR-0002 / P7). A failure here
    after a layout change means the file is stale: regenerate it (command in the section header)."""
    import json
    env = Environment()
    with open(_SPEC_PATH) as f:
        on_disk = json.load(f)
    assert on_disk == FeatureLayout(env).spec(), (
        "feature_layout.json drifted from FeatureLayout — regenerate it (see test_feature_layout.py header)"
    )
