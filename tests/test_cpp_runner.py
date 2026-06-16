#!/usr/bin/env python3
"""
tests/test_cpp_runner.py — pins for the C++ runner seam (ADR-0012's C++ beachhead).

Two layers:
  * ALWAYS-ON (no C++ / no redis): the Python `RandomPolicy` contract the C++ runner mirrors — it is
    a `Policy` subclass (the env<->Policy seam, P2), draws only legal actions + the always-legal
    TERMINATE, and its action distribution sits ON the legality mask (illegal-slot mass is 0, the
    same logic invariant the C++ M carries). These are the parity baseline's invariants and run in
    every `pytest tests/ -q`.
  * OPT-IN (needs the built C++ binary + redis): the full ADR-0012 P6/P7 behavioral-parity harness
    (cpp/parity/parity.py) — aggregate-stat indistinguishability within MC CI, the bit-exact mask,
    the feature X-port equivalence, the format round-trip. SKIPPED (not failed) when the binary or
    redis is absent, so the default `pytest tests/ -q` stays green on a box without the C++ build.

The NMCS Policy (the nested Monte-Carlo search ported behind the SAME seam) adds:
  * ALWAYS-ON: NMCSPolicy is a Policy subclass registered in SOLVERS (the seam invariant).
  * OPT-IN (needs chocofarm-nmcs-dump): the DETERMINISTIC logic check (cpp/parity/nmcs_logic.py) —
    same selected action on identical scripted leaf returns, level-1 AND level-2. NO redis (pure
    env + scripted source).
  * OPT-IN (needs chocofarm-cpp-runner + redis): the AGGREGATE behavioral parity
    (cpp/parity/nmcs_parity.py) — NMCS aggregates within MC CI. Both SKIPPED when the fixture/redis
    is absent.

The ISMCTS Policy (single-observer Information Set MCTS, ported behind the SAME seam) adds the same
three layers:
  * ALWAYS-ON: ISMCTSPolicy is a Policy subclass registered in SOLVERS (the seam invariant).
  * OPT-IN (needs chocofarm-ismcts-dump): the DETERMINISTIC logic check (cpp/parity/ismcts_logic.py)
    — same selected action on identical scripted world/expansion/leaf draws, across iteration counts
    (expansion, UCB select, the availability denominator). NO redis (pure env + scripted source).
  * OPT-IN (needs chocofarm-cpp-runner + redis): the AGGREGATE behavioral parity
    (cpp/parity/ismcts_parity.py) — ISMCTS aggregates within MC CI. Both SKIPPED when the
    fixture/redis is absent.

The Gumbel-AZ Policy (the Danihelka Gumbel-AlphaZero search, ported behind the SAME seam — PHASE 1a,
the discrete STRUCTURE only) adds:
  * ALWAYS-ON: GumbelPolicy is a Policy subclass (the seam invariant the C++ GumbelAZPolicy mirrors —
    a drop-in alongside RandomPolicy / NMCSPolicy / ISMCTSPolicy with zero env/runner core edits).
  * OPT-IN (needs chocofarm-gumbel-dump): the DETERMINISTIC STRUCTURE logic check
    (cpp/parity/gumbel_logic.py) — same executed action AND improved-pi argmax on identical scripted
    gumbel/world/leaf draws, plus the two structural Danihelka invariants (executed==SH-survivor; SH
    spends the full n_sims budget). PRECISION-INSENSITIVE (coarse, well-separated scripted leaf, no
    near-ties) so the discrete outcome is float32-vs-float64 identical — this is the 1a structure
    check; the mixed-precision near-tie path is 1b (NOT covered). NO redis (the scripted leaf is
    in-process).

Public Domain (The Unlicense).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chocofarm.az.actions import action_to_slot, legal_mask, n_action_slots, term_slot
from chocofarm.model.env import TERMINATE, Environment
from chocofarm.solvers import SOLVERS
from chocofarm.solvers.base import Policy, RandomPolicy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-cpp-runner")
NET_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-net-dump")
NMCS_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-nmcs-dump")
ISMCTS_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-ismcts-dump")
PARITY = os.path.join(REPO, "cpp", "parity", "parity.py")
NET_PARITY = os.path.join(REPO, "cpp", "parity", "net_parity.py")
NMCS_LOGIC = os.path.join(REPO, "cpp", "parity", "nmcs_logic.py")
NMCS_PARITY = os.path.join(REPO, "cpp", "parity", "nmcs_parity.py")
ISMCTS_LOGIC = os.path.join(REPO, "cpp", "parity", "ismcts_logic.py")
ISMCTS_PARITY = os.path.join(REPO, "cpp", "parity", "ismcts_parity.py")
GUMBEL_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-gumbel-dump")
GUMBEL_LOGIC = os.path.join(REPO, "cpp", "parity", "gumbel_logic.py")

# OPT-IN gate. The binary-dependent cpp parity tests run only with CHOCO_RUN_CPP=1 (and a freshly
# built binary). They are slow integration checks driven by a MANUALLY-built C++ binary: a stale
# binary (one that predates a new --policy, say) silently fails rather than skipping, which can red
# the DEFAULT suite even though the code is sound. So they are opt-in; validate cpp explicitly with
#   cmake --build cpp/build && CHOCO_RUN_CPP=1 PYTHONPATH=. python -m pytest tests/test_cpp_runner.py
_RUN_CPP = bool(os.environ.get("CHOCO_RUN_CPP"))
_CPP_SKIP = "opt-in cpp parity: set CHOCO_RUN_CPP=1 and build the binary fresh (cmake --build cpp/build)"


# ---------------------------------------------------------------------------
# ALWAYS-ON: the Python RandomPolicy contract (the C++ parity baseline).
# ---------------------------------------------------------------------------
def test_random_policy_is_a_policy_subclass():
    """P2: a new capability is a new `Policy` subclass with zero env edits — RandomPolicy is one, and
    it is registered in the SOLVERS name table."""
    assert issubclass(RandomPolicy, Policy)
    assert SOLVERS["random"] is RandomPolicy


def test_nmcs_policy_is_a_policy_subclass():
    """P2: the NMCS search is a `Policy` subclass registered in SOLVERS — the SAME seam invariant the
    C++ NMCSPolicy mirrors (a drop-in alongside RandomPolicy with zero env/runner core edits)."""
    from chocofarm.solvers.nmcs import NMCSPolicy
    assert issubclass(NMCSPolicy, Policy)
    assert SOLVERS["nmcs"] is NMCSPolicy


def test_ismcts_policy_is_a_policy_subclass():
    """P2: the single-observer ISMCTS search is a `Policy` subclass registered in SOLVERS — the SAME
    seam invariant the C++ ISMCTSPolicy mirrors (a drop-in alongside RandomPolicy / NMCSPolicy with
    zero env/runner core edits)."""
    from chocofarm.solvers.ismcts import ISMCTSPolicy
    assert issubclass(ISMCTSPolicy, Policy)
    assert SOLVERS["ismcts"] is ISMCTSPolicy


def test_gumbel_policy_is_a_policy_subclass():
    """P2: the Gumbel-AlphaZero search is a `Policy` subclass — the SAME seam invariant the C++
    GumbelAZPolicy mirrors (a drop-in alongside RandomPolicy / NMCSPolicy / ISMCTSPolicy with zero
    env/runner core edits). GumbelPolicy is the eval wrapper (temperature 0 = the SH survivor) in
    chocofarm.az.gumbel_search; it is NOT in the classical SOLVERS registry (it is the net-using AZ
    self-play search, constructed with an already-loaded net), so we pin only the Policy-subclass
    contract — the seam the C++ port stands behind."""
    from chocofarm.az.gumbel_search import GumbelPolicy
    assert issubclass(GumbelPolicy, Policy)


def test_random_policy_only_picks_legal_actions():
    """Every action RandomPolicy returns is either legal (env.legal_actions) or TERMINATE (always
    legal). Drive many decisions over evolving beliefs and assert legality each time."""
    env = Environment()
    pol = RandomPolicy()
    rng = np.random.default_rng(0)
    for ep in range(40):
        loc, bw, collected = ("w", env.entry), env.worlds, set()
        for _ in range(env.max_steps):
            if len(bw) == 0:
                break
            legal = set(env.legal_actions(loc, bw, collected)) | {TERMINATE}
            a = pol.decide(env, loc, bw, collected, 0.1, rng)
            assert a in legal, (a, ep)
            if a == TERMINATE:
                break
            world = int(rng.choice(bw))
            _, loc, bw, collected, _ = env.apply(loc, bw, collected, a, world)


def test_random_policy_distribution_sits_on_the_legal_mask():
    """The empirical RandomPolicy action distribution puts mass ONLY on slots the legality mask marks
    legal — illegal-slot mass is exactly 0 (the same logic invariant the C++ M / PI carries)."""
    env = Environment()
    pol = RandomPolicy()
    rng = np.random.default_rng(1)
    loc, bw, collected = ("w", env.entry), env.worlds, set()
    mask = legal_mask(env, loc, bw, collected)
    counts = np.zeros(n_action_slots(env))
    for _ in range(5000):
        a = pol.decide(env, loc, bw, collected, 0.0, rng)
        counts[action_to_slot(env, a)] += 1
    # zero mass on any illegal slot (== 0.0, bit-exact logic fact)
    assert float(counts[mask == 0.0].sum()) == 0.0
    # the TERMINATE slot is always legal and is drawn
    assert mask[term_slot(env)] == 1.0


def test_random_policy_lambda_is_threaded_not_consumed():
    """P4: lam is threaded through the seam unchanged but RandomPolicy ignores it — the SAME rng
    state yields the SAME action regardless of lam (a value-aware policy would differ)."""
    env = Environment()
    pol = RandomPolicy()
    loc, bw, collected = ("w", env.entry), env.worlds, set()
    a1 = pol.decide(env, loc, bw, collected, 0.0, np.random.default_rng(7))
    a2 = pol.decide(env, loc, bw, collected, 9.9, np.random.default_rng(7))
    assert a1 == a2


# ---------------------------------------------------------------------------
# OPT-IN: the full C++ behavioral-parity harness (needs the binary + redis).
# ---------------------------------------------------------------------------
def _redis_up():
    try:
        from chocofarm.az import transport
        transport.connect()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(CPP_BIN)), reason=_CPP_SKIP)
def test_cpp_parity_harness():
    """Run the full ADR-0012 P6/P7 parity harness end-to-end. Skips (does not fail) when redis is
    down, so the default suite stays green without the worker-transport instance up."""
    if not _redis_up():
        pytest.skip("redis not reachable on the CHOCO_TRANSPORT_REDIS_* contract")
    out = subprocess.run([sys.executable, PARITY], cwd=REPO,
                         env={**os.environ, "PYTHONPATH": REPO},
                         capture_output=True, text=True, timeout=600)
    # the harness prints a verdict and returns 0 on PASS
    assert out.returncode == 0, f"parity harness FAILED:\n{out.stdout}\n{out.stderr}"
    assert "RESULT: PASS" in out.stdout, out.stdout


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(NET_BIN)), reason=_CPP_SKIP)
def test_cpp_net_forward_parity():
    """The C++ NetForward forward-parity harness (cpp/parity/net_parity.py): the C++ leaf evaluator
    reimplements the ONE Python `forward_core` to the test_jax_equivalence bar (max|Δvalue| AND
    max|Δlogit| < 1e-4) over N≥1000 random float32 feature vectors, residual ON and OFF — the same
    ADR-0012 P6 behavioral-equivalence bar (NOT byte-identity). Skips (does not fail) when redis is
    down, so the default suite stays green without the worker-transport instance up."""
    if not _redis_up():
        pytest.skip("redis not reachable on the CHOCO_TRANSPORT_REDIS_* contract")
    out = subprocess.run([sys.executable, NET_PARITY], cwd=REPO,
                         env={**os.environ, "PYTHONPATH": REPO},
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, f"net-forward parity harness FAILED:\n{out.stdout}\n{out.stderr}"
    assert "RESULT: PASS" in out.stdout, out.stdout


@pytest.mark.skip(reason="NMCS parity retired until nmcs-init work resumes (validated repeatedly; see BACKLOG.md)")
@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(NMCS_BIN)), reason=_CPP_SKIP)
def test_cpp_nmcs_logic_parity():
    """The DETERMINISTIC NMCS logic check (cpp/parity/nmcs_logic.py): with the RNG abstracted behind a
    scripted, RNG-free WorldSource (sample_world->bw[0]; playout_value->a fixed cycled table), the C++
    NMCS and the Python NMCS SELECT THE SAME ACTION on fixed (loc, belief) inputs — for level-1 AND
    level-2 (the milestone). This validates the nesting + selection logic, the part that MUST be exact,
    independent of RNG (ADR-0012 P6). No redis needed (pure env + scripted source)."""
    out = subprocess.run([sys.executable, NMCS_LOGIC], cwd=REPO,
                         env={**os.environ, "PYTHONPATH": REPO},
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, f"NMCS logic check FAILED:\n{out.stdout}\n{out.stderr}"
    assert "RESULT: PASS" in out.stdout, out.stdout


@pytest.mark.skip(reason="NMCS parity retired until nmcs-init work resumes (validated repeatedly; see BACKLOG.md)")
@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(CPP_BIN)), reason=_CPP_SKIP)
def test_cpp_nmcs_aggregate_parity():
    """The AGGREGATE NMCS behavioral parity (cpp/parity/nmcs_parity.py): the C++ NMCS runner
    (`--policy nmcs`) and the Python NMCSPolicy over matched-seed episodes agree on every aggregate
    (mean length, λ-return, action-type distribution, belief-shrinkage) within Monte-Carlo CI, with
    the MC SE reported — the ADR-0012 P6 behavioral bar (NOT byte-identity; the RNGs differ). NMCS is
    the slowest solver, so N is moderate. Skips (does not fail) when redis is down."""
    if not _redis_up():
        pytest.skip("redis not reachable on the CHOCO_TRANSPORT_REDIS_* contract")
    out = subprocess.run([sys.executable, NMCS_PARITY], cwd=REPO,
                         env={**os.environ, "PYTHONPATH": REPO},
                         capture_output=True, text=True, timeout=1200)
    assert out.returncode == 0, f"NMCS aggregate parity FAILED:\n{out.stdout}\n{out.stderr}"
    assert "RESULT: PASS" in out.stdout, out.stdout


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(ISMCTS_BIN)), reason=_CPP_SKIP)
def test_cpp_ismcts_logic_parity():
    """The DETERMINISTIC ISMCTS logic check (cpp/parity/ismcts_logic.py): with the RNG abstracted
    behind a scripted, RNG-free ISMCTSSource (sample_world->bw[0]; expand_index->a fixed FIFO mod n;
    leaf_value->a fixed cycled table), the C++ ISMCTS and the Python ISMCTS SELECT THE SAME ACTION on
    fixed (loc, belief) inputs — across iteration counts that cover pure expansion, UCB selection, and
    the availability denominator. This validates the selection + nesting logic, the part that MUST be
    exact, independent of RNG (ADR-0012 P6). No redis needed (pure env + scripted source)."""
    out = subprocess.run([sys.executable, ISMCTS_LOGIC], cwd=REPO,
                         env={**os.environ, "PYTHONPATH": REPO},
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, f"ISMCTS logic check FAILED:\n{out.stdout}\n{out.stderr}"
    assert "RESULT: PASS" in out.stdout, out.stdout


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(GUMBEL_BIN)), reason=_CPP_SKIP)
def test_cpp_gumbel_logic_parity():
    """The DETERMINISTIC Gumbel-AZ STRUCTURE logic check (cpp/parity/gumbel_logic.py): with the RNG +
    leaf abstracted behind a scripted, RNG-free seam (rng.gumbel->a fixed FIFO; sample_world->bw[0];
    the leaf (value, coarse logits)->a fixed cycled table), the C++ Gumbel-AZ search and the Python
    Gumbel-AZ search EXECUTE THE SAME ACTION and produce the SAME improved-pi argmax on fixed
    (loc, belief) inputs — across (n_sims, m, c_puct, max_depth, prefix). This validates the discrete
    structure + selection logic, the part that MUST be exact, independent of RNG AND of float32-vs-
    float64 precision (the scripted leaf is COARSE, well-separated -> NO near-ties, so the discrete
    outcome is precision-insensitive). It ALSO pins the two structural Danihelka invariants
    (executed==SH-survivor; SH spends the full n_sims budget). This is PHASE 1a (structure only); the
    mixed-precision near-tie path is PHASE 1b (NOT covered). NO redis (the scripted leaf is in-process).

    NOTE: the mutation control (--mutate sh-budget|puct, which mutates the C++ binary and asserts it
    diverges from the unmodified Python) is exercised by running gumbel_logic.py with those flags; this
    guard runs the FAITHFUL pass (must PASS)."""
    out = subprocess.run([sys.executable, GUMBEL_LOGIC], cwd=REPO,
                         env={**os.environ, "PYTHONPATH": REPO},
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, f"Gumbel logic check FAILED:\n{out.stdout}\n{out.stderr}"
    assert "RESULT: PASS" in out.stdout, out.stdout


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(CPP_BIN)), reason=_CPP_SKIP)
def test_cpp_ismcts_aggregate_parity():
    """The AGGREGATE ISMCTS behavioral parity (cpp/parity/ismcts_parity.py): the C++ ISMCTS runner
    (`--policy ismcts`) and the Python ISMCTSPolicy over matched-seed episodes agree on every
    aggregate (mean length, λ-return, action-type distribution, belief-shrinkage) within Monte-Carlo
    CI, with the MC SE reported — the ADR-0012 P6 behavioral bar (NOT byte-identity; the RNGs differ).
    ISMCTS runs many iterations per decision, so N is moderate. Skips (does not fail) when redis is
    down."""
    if not _redis_up():
        pytest.skip("redis not reachable on the CHOCO_TRANSPORT_REDIS_* contract")
    out = subprocess.run([sys.executable, ISMCTS_PARITY], cwd=REPO,
                         env={**os.environ, "PYTHONPATH": REPO},
                         capture_output=True, text=True, timeout=1200)
    assert out.returncode == 0, f"ISMCTS aggregate parity FAILED:\n{out.stdout}\n{out.stderr}"
    assert "RESULT: PASS" in out.stdout, out.stdout
