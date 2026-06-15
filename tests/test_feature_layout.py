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
