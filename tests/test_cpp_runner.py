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
    check. NO redis (the scripted leaf is in-process).
  * OPT-IN (needs chocofarm-gumbel-dump): the PHASE 1b mixed-precision NEAR-TIE parity
    (cpp/parity/gumbel_precision.py) — on FINE scripted leaf (value, full-precision per-slot logits)
    drawn so the DISCRETE output sits on the float32-prior knife-edge, the MIXED-precision C++ matches
    Python EXACTLY (N/N) while the all-float64 DISCRIMINATION control (CHOCO_GUMBEL_UNIFORM=1) diverges
    on a non-trivial fraction — proving the float32 PRIOR precision (not the structure) decides the
    near-ties (non-vacuous, the 1b analogue of the 1a mutation control). NO redis (in-process leaf).

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
GUMBEL_PRECISION = os.path.join(REPO, "cpp", "parity", "gumbel_precision.py")
SERIAL_CHECK_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-serial-runtime-check")
BENCH_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-search-runtime-bench")
WIRE_BENCH_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-wire-bench")
WIRE_BENCH = os.path.join(REPO, "cpp", "parity", "wire_bench.py")
FIBER_PROTO_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-fiber-proto")
BELIEF_CACHE_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-belief-cache-check")
BELIEF_ORACLE_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-belief-sweep-oracle-check")
DATA_INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
DATA_FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")

# OPT-IN gate. The binary-dependent cpp parity tests run only with CHOCO_RUN_CPP=1 (and a freshly
# built binary). They are slow integration checks driven by a MANUALLY-built C++ binary: a stale
# binary (one that predates a new --policy, say) silently fails rather than skipping, which can red
# the DEFAULT suite even though the code is sound. So they are opt-in; validate cpp explicitly with
#   cmake --build cpp/build && CHOCO_RUN_CPP=1 PYTHONPATH=. python -m pytest tests/test_cpp_runner.py
_RUN_CPP = bool(os.environ.get("CHOCO_RUN_CPP"))
_CPP_SKIP = "opt-in cpp parity: set CHOCO_RUN_CPP=1 and build the binary fresh (cmake --build cpp/build)"


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(BELIEF_CACHE_BIN)), reason=_CPP_SKIP)
def test_cpp_belief_cache_collision_guard():
    """The FeatureBuilder belief-memo keys by the (count, first, last) belief fingerprint, which is
    collision-RESISTANT, not collision-free; correctness rests on the full bw-equality guard walking a
    fingerprint bucket. chocofarm-belief-cache-check FORCES a collision (two distinct beliefs sharing a
    fingerprint) and asserts each gets its OWN features + that a cache hit is bit-identical to a recompute
    (ADR-0011: net the guard, don't trust it). Runs from REPO so FeatureBuilder finds feature_layout.json."""
    out = subprocess.run([BELIEF_CACHE_BIN, "--instance", DATA_INSTANCE, "--faces", DATA_FACES],
                         cwd=REPO, capture_output=True, text=True, timeout=60,
                         env={**os.environ, "PYTHONPATH": REPO})
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    assert out.returncode == 0 and "RESULT: PASS" in out.stdout


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(BELIEF_ORACLE_BIN)), reason=_CPP_SKIP)
def test_cpp_belief_sweep_oracle():
    """The belief sweep (chocofarm::belief_features) was rewritten to the §A.4 form — a fused branchless
    integer sweep over contiguous env.face_masks(), normalizing marg AND p_pos via `* inv` (the settled
    re-baseline). chocofarm-belief-sweep-oracle-check recomputes it two INDEPENDENT ways (production vs a
    naive env.observe count, same *inv spec) and asserts byte-equality over a sample of beliefs — the
    in-language bit-exact net for the rewrite and every later rung (SIMD, the Part B diagram). ADR-0011:
    net the rewrite, don't trust it. Pure compute (no FeatureBuilder, no layout file, no redis) so the
    binary is cwd-independent; cwd=REPO / PYTHONPATH are kept only for parity with the other cpp gates.

    STEP 2 (the bitset belief arm, docs/design/cpp-belief-rep-scoping.md §5) extended this binary with a
    flat-vs-bitset A/B that builds BOTH reps and asserts byte-identity across every env seam op (marginals,
    informative, legal_actions, belief_features, nb, belief_key, world_at_rank, sample_world) plus a full
    in-place filter sequence. This test PINS the live instance to the bitset arm (ADR-0011 Rule 1: the gate
    is enforced here, the strongest feasible surface): the A/B must print its OWN 'RESULT: PASS' line, NOT
    'RESULT: SKIP' — a SKIP means the gate silently fell back to flat (e.g. a dim change pushed mask_bytes
    past the cache budget), which the bare 'RESULT: PASS in stdout' check below would not catch."""
    out = subprocess.run([BELIEF_ORACLE_BIN, "--instance", DATA_INSTANCE, "--faces", DATA_FACES],
                         cwd=REPO, capture_output=True, text=True, timeout=60,
                         env={**os.environ, "PYTHONPATH": REPO})
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    assert out.returncode == 0 and "RESULT: PASS" in out.stdout
    # the live instance MUST gate ON: the flat-vs-bitset A/B ran (PASS) and did not silently fall back (SKIP)
    assert "RESULT: PASS flat-vs-bitset A/B" in out.stdout, out.stdout
    assert "RESULT: SKIP" not in out.stdout, out.stdout


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


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(GUMBEL_BIN)), reason=_CPP_SKIP)
def test_cpp_gumbel_precision_parity():
    """The PHASE 1b mixed-precision NEAR-TIE parity (cpp/parity/gumbel_precision.py): the numerical-
    FIDELITY twin of the 1a STRUCTURE check above. On REALISTIC FINE scripted leaf (value, full-
    precision per-slot logits) — drawn so the search's DISCRETE output (the SH survivor + the improved-pi
    argmax) sits on the float32-prior knife-edge — the MIXED-precision C++ search reproduces Python's
    DELIBERATE float32-prior x float64-Q precision EXACTLY (executed action AND improved-pi argmax,
    N/N), while the DISCRIMINATION control (CHOCO_GUMBEL_UNIFORM=1, the genuine 1a all-float64 port)
    DIVERGES from the SAME Python reference on a NON-TRIVIAL fraction (~34/144). The harness ASSERTS
    BOTH: mixed==Python N/N proves byte-faithfulness; uniform-diverges>=10 proves the parity is NOT
    vacuous (the 1b analogue of the 1a mutation control — the float32 PRIOR precision, not the structure,
    decides the near-ties). NO redis (the scripted leaf is in-process). RESULT: PASS gates both."""
    out = subprocess.run([sys.executable, GUMBEL_PRECISION], cwd=REPO,
                         env={**os.environ, "PYTHONPATH": REPO},
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, f"Gumbel precision check FAILED:\n{out.stdout}\n{out.stderr}"
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


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(SERIAL_CHECK_BIN)), reason=_CPP_SKIP)
def test_cpp_serial_runtime_seam_faithful():
    """The SearchRuntime SEAM-FAITHFULNESS check (cpp/src/serial_runtime_check.cpp): SerialRuntime.run
    over a batch of independent tasks produces, for each task, the SAME executed action as a direct
    GumbelAZPolicy::decide with the same RNG seed — proving the runtime seam does NOT perturb the search
    (the work-stealing pool's parity precondition, docs/design/cpp-search-runtime.md §7.1). The leaf is a
    DETERMINISTIC, STATELESS, in-process net (no redis, no weights, no RNG), so the check is fully
    reproducible. A C++-INTERNAL self-check (NOT a cross-language parity): the binary asserts and exits
    0/nonzero; the gate is exit 0 + 'PASS' in stdout."""
    out = subprocess.run([SERIAL_CHECK_BIN, "--instance", DATA_INSTANCE, "--faces", DATA_FACES],
                         cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, f"SerialRuntime seam check FAILED:\n{out.stdout}\n{out.stderr}"
    assert "PASS" in out.stdout, out.stdout


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(BENCH_BIN)), reason=_CPP_SKIP)
def test_cpp_pool_runtime_matches_serial():
    """PoolRuntime (the local task-parallel runtime) produces BIT-IDENTICAL per-task results to
    SerialRuntime — same executed action AND leaf-request count for every task — because the trees are
    independent and deterministic (seeded), so the parallelism is EXACT, not merely aggregate-equivalent
    (docs/design/cpp-search-runtime.md: the C++-native-MLP / local-parallel config). The benchmark binary
    asserts this before timing and exits nonzero on any mismatch; here we run a small/fast config and gate
    on exit 0 + 'RESULT: PASS' (this is the pool's correctness regression guard — the throughput numbers
    it also prints are not asserted, only the bit-identity)."""
    out = subprocess.run([BENCH_BIN, "--instance", DATA_INSTANCE, "--faces", DATA_FACES,
                          "--tasks", "6", "--n-sims", "8", "--max-depth", "4", "--workers", "4",
                          "--reps", "1"],
                         cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, f"PoolRuntime vs SerialRuntime check FAILED:\n{out.stdout}\n{out.stderr}"
    assert "RESULT: PASS" in out.stdout, out.stdout


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(WIRE_BENCH_BIN)), reason=_CPP_SKIP)
def test_cpp_wire_benchmark():
    """The over-the-wire benchmark, BOTH axes (cpp/parity/wire_bench.py): spin the Python InferenceServer
    in-process (StaticParamsSource, a dimension-matched ValueMLP — NO redis) and run, against that one
    server, (a) the SYNCHRONOUS bench (SerialRuntime + a blocking ZmqNetClient leaf — one in-flight at a
    time) and (b) the PARALLEL bench (K boost.context tree-fibers batch-submitting parked leaves over a
    DEALER so the server batches them) — the §6-Q5 sync-vs-parallel comparison. The driver SKIPS (returns
    0) if pyzmq / a binary is absent; here we gate on exit 0 (PASS or SKIP), not a throughput threshold
    (wall time is hardware-dependent; this guard pins that BOTH wire paths RUN end to end)."""
    out = subprocess.run([sys.executable, WIRE_BENCH], cwd=REPO,
                         env={**os.environ, "PYTHONPATH": REPO},
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, f"wire-sync benchmark FAILED:\n{out.stdout}\n{out.stderr}"
    assert ("RESULT: PASS" in out.stdout) or ("RESULT: SKIP" in out.stdout), out.stdout


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(FIBER_PROTO_BIN)), reason=_CPP_SKIP)
def test_cpp_fiber_proto_matches_direct():
    """Option A foundation proof (cpp/src/fiber_proto.cpp): the UNCHANGED GumbelAZPolicy::run_search,
    driven inside a boost.context stackful fiber with a YieldingNetEvaluator that yields at each leaf,
    produces a BIT-IDENTICAL result (executed action, improved-pi argmax, n_spent) to a direct synchronous
    run_search fed the same scripted leaves + RNG. This is the resumable-search mechanism the wire-parallel
    work-stealing pool needs, established WITHOUT touching the 1a/1b-validated search (the fiber preserves
    fidelity by construction — only WHEN predict returns changes, not WHAT). Gate on exit 0 + 'RESULT:
    PASS'."""
    out = subprocess.run([FIBER_PROTO_BIN, "--instance", DATA_INSTANCE, "--faces", DATA_FACES,
                          "--n-sims", "24", "--max-depth", "8"],
                         cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, f"fiber prototype FAILED:\n{out.stdout}\n{out.stderr}"
    assert "RESULT: PASS" in out.stdout, out.stdout


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(CPP_BIN)), reason=_CPP_SKIP)
def test_cpp_actor_loop_turns():
    """The goal-2 assembly (chocofarm/az/cpp_actor_loop.py): one ExIt iteration driven by the C++ Gumbel
    ACTOR — publish weights -> chocofarm-cpp-runner --policy gumbel generates transitions (improved-π PI)
    -> JaxTrainer.train_step -> repeat. Proves the full generate->train->publish cycle turns with the C++
    runtime as the actor (vs exit_loop's Python worker). Needs redis (the transport) + the runner built +
    jax; skips otherwise. Gate on exit 0 + 'DONE' (the loop completed an iteration)."""
    if not _redis_up():
        pytest.skip("redis not reachable on the CHOCO_TRANSPORT_REDIS_* contract")
    out = subprocess.run([sys.executable, "-m", "chocofarm.az.cpp_actor_loop", "--runner", CPP_BIN,
                          "--instance", DATA_INSTANCE, "--faces", DATA_FACES,
                          "--iters", "1", "--episodes", "4", "--n-sims", "8", "--gumbel-max-depth", "6",
                          "--hidden", "32", "--epochs", "1", "--run", "cpp-actor-loop-test"],
                         cwd=REPO, env={**os.environ, "PYTHONPATH": REPO},
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, f"C++-actor loop FAILED:\n{out.stdout}\n{out.stderr}"
    assert "DONE" in out.stdout, out.stdout


@pytest.mark.skipif(not _RUN_CPP, reason=_CPP_SKIP)
def test_cpp_actor_exit_loop_swap_turns():
    """The SWAP (chocofarm/az/cpp_executor.CppActorExecutor): the C++ Gumbel actor injected into the
    PRODUCTION exit_loop as the GENERATION executor — so the held-out eval, replay window, JAX training,
    checkpointing, and hp registry are all inherited unchanged while the C++ runner produces the
    transitions. Proves one full exit_loop iteration turns through the C++ actor (vs the Python pool) and
    writes a checkpoint. Needs redis + the runner + jax; gate on exit 0, 'DONE 1 iters', and a real ckpt."""
    if not _redis_up():
        pytest.skip("redis not reachable on the CHOCO_TRANSPORT_REDIS_* contract")
    import shutil
    import tempfile
    ckpt = tempfile.mkdtemp(prefix="cpp_exit_swap_")
    try:
        out = subprocess.run(
            [sys.executable, "-m", "chocofarm.az.exit_loop", "--cpp-runner", CPP_BIN,
             "--cpp-instance", DATA_INSTANCE, "--cpp-faces", DATA_FACES,
             "-I", "1", "-E", "4", "-W", "2", "--epochs", "1", "--m", "12", "--n-sims", "6",
             "--eval-n", "1", "--explore-plies", "0", "--hidden", "32", "--lam", "0.0855", "--seed", "7",
             "--ckpt-dir", ckpt],  # experiment-id defaults to the UNIQUE tmpdir basename — a fresh
            #                        registry seed each run (so --explore-plies 0 is not overridden by a
            #                        stale re-bound blob), and no cross-run registry pollution under a fixed id
            cwd=REPO, env={**os.environ, "PYTHONPATH": REPO},
            capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, f"C++-actor exit_loop SWAP FAILED:\n{out.stdout}\n{out.stderr}"
        assert "DONE 1 iters" in out.stdout, out.stdout
        assert "C++ Gumbel ACTOR generation" in out.stdout, out.stdout
        assert os.path.exists(os.path.join(ckpt, "net_iter000.npz")), "no checkpoint written"
        assert os.path.exists(os.path.join(ckpt, "history.json")), "no history written"
    finally:
        shutil.rmtree(ckpt, ignore_errors=True)


@pytest.mark.skipif(not _RUN_CPP, reason=_CPP_SKIP)
def test_cpp_actor_executor_partb_fails_loud():
    """ADR-0002: the C++ actor emits the pure-MC λ-return, so requesting a Part-B blend
    (td_lambda<1 or n_step) must FAIL LOUD at generate() — never silently train on a pure-MC target the
    operator did not ask for. The guard fires BEFORE any weight publish / subprocess, so a None net is
    fine. Needs redis only for the executor's connection in __init__."""
    if not _redis_up():
        pytest.skip("redis not reachable on the CHOCO_TRANSPORT_REDIS_* contract")
    from chocofarm.az.actions import n_action_slots
    from chocofarm.az.cpp_executor import CppActorExecutor
    from chocofarm.az.features import feature_dim
    from chocofarm.model.env import Environment
    env = Environment()
    ex = CppActorExecutor(CPP_BIN, DATA_INSTANCE, DATA_FACES, env, base_seed=7,
                          use_jax_mlp=False, in_dim=feature_dim(env), n_slots=n_action_slots(env))
    try:
        # explore_plies=0 isolates the Part-B guard (else the explore_plies guard could fire first).
        with pytest.raises(RuntimeError, match="pure-MC"):
            ex.generate(None, 0, [0, 1], 0.0855, 0, lam_blend=0.5, n_step=None)
        with pytest.raises(RuntimeError, match="pure-MC"):
            ex.generate(None, 0, [0, 1], 0.0855, 0, lam_blend=1.0, n_step=3)
    finally:
        ex.close()


@pytest.mark.skipif(not _RUN_CPP, reason=_CPP_SKIP)
def test_cpp_actor_executor_explore_plies_fails_loud():
    """ADR-0002: the C++ actor plays the temperature-0 SH survivor every ply, so it cannot honor the
    temperature-1 explore_plies exploration prefix both Python paths apply. Requesting explore_plies>0 must
    FAIL LOUD at generate() — never silently generate zero-exploration self-play (the same fail-loud
    standard the Part-B guard sets). The guard fires before any subprocess, so a None net is fine."""
    if not _redis_up():
        pytest.skip("redis not reachable on the CHOCO_TRANSPORT_REDIS_* contract")
    from chocofarm.az.actions import n_action_slots
    from chocofarm.az.cpp_executor import CppActorExecutor
    from chocofarm.az.features import feature_dim
    from chocofarm.model.env import Environment
    env = Environment()
    ex = CppActorExecutor(CPP_BIN, DATA_INSTANCE, DATA_FACES, env, base_seed=7,
                          use_jax_mlp=False, in_dim=feature_dim(env), n_slots=n_action_slots(env))
    try:
        with pytest.raises(RuntimeError, match="explore_plies"):
            ex.generate(None, 0, [0, 1], 0.0855, 4, lam_blend=1.0, n_step=None)
        # pure-MC + explore_plies=0 passes both guards (it would proceed to publish/subprocess) — assert
        # the two guards do NOT fire on the supported default-shape call by reaching past them is covered
        # by the swap-turns test; here we only assert the explore_plies>0 refusal.
    finally:
        ex.close()


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(CPP_BIN)), reason=_CPP_SKIP)
def test_cpp_serve_online_reconfiguration():
    """The persistent --serve runner (the ActorTransport's C++ side) drives the FULL online-
    reconfiguration capability end-to-end against redis: configure -> generate -> LIVE-retune n_sims on
    the SAME process (the env/policy context preserved, no respawn) -> generate at a new version (the
    weight-reload gate) -> read the records back, reconciling `written`. Plus the two loud rejects: a
    stale config_epoch (config_epoch_mismatch) and an instance/faces change (instance_knob_changed). This
    is the cross-language integration proof of the protocol the fake-runner unit tests
    (test_actor_transport.py) pin in pure Python — the m/n_sims-HOT online-reconfig the whole thread
    exists to deliver, exercised against the real C++ Gumbel actor."""
    if not _redis_up():
        pytest.skip("redis not reachable on the CHOCO_TRANSPORT_REDIS_* contract")
    import uuid
    from chocofarm.az import transport
    from chocofarm.az.actor_config import ActorConfig
    from chocofarm.az.actor_transport import ControlError, GenerateRequest, SubprocessActorTransport
    from chocofarm.az.features import feature_dim
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.az.result_spec import RESULT_DTYPE
    from chocofarm.az.transport import result_keys
    from chocofarm.model.env import Environment

    env = Environment()
    net = ValueMLP(feature_dim(env), hidden=32, n_actions=n_action_slots(env), seed=0)
    net.set_value_scale(0.0, 1.0)
    conn = transport.connect()
    tp = transport.RedisTransport(conn)
    run = uuid.uuid4().hex[:12]

    def read_back(tok, n):
        found = 0
        for idx in range(n):
            xk, pik, mk, yk = result_keys(tok, idx)
            if conn.get(yk):
                found += 1
                conn.delete(xk, pik, mk, yk)
        return found

    def cfg(n_sims):
        return ActorConfig(DATA_INSTANCE, DATA_FACES, m=4, n_sims=n_sims, c_puct=1.25, c_visit=50.0,
                           c_scale=1.0, c_outcome=2, max_depth=24)

    tp.publish_weights(net, "gen", 0, run)
    t = SubprocessActorTransport(CPP_BIN, extra_args=("--run", run))
    try:
        epoch = t.reconfigure(cfg(8))
        assert epoch == 1
        tok0 = run + "-gen-0"
        res = t.generate(GenerateRequest(config_epoch=epoch, version=0, seed=7, lam=0.0855,
                                         episodes=6, max_steps=12, res_token=tok0))
        assert res.written >= 1 and res.config_epoch == 1 and res.version == 0
        assert read_back(tok0, 6) == res.written, "written must reconcile with the redis read-back"

        # THE CAPABILITY: live-retune n_sims 8->16 with NO process/env teardown, then generate at a new
        # version (the weight-reload gate, independent of the config epoch).
        epoch2 = t.reconfigure(cfg(16))
        assert epoch2 == 2
        tp.publish_weights(net, "gen", 1, run)
        tok1 = run + "-gen-1"
        res2 = t.generate(GenerateRequest(config_epoch=epoch2, version=1, seed=8, lam=0.0855,
                                          episodes=6, max_steps=12, res_token=tok1))
        assert res2.written >= 1 and res2.config_epoch == 2 and res2.version == 1
        assert read_back(tok1, 6) == res2.written

        # the two loud rejects (the gates / the INSTANCE-knob ACL).
        with pytest.raises(ControlError) as ei:
            t.generate(GenerateRequest(config_epoch=1, version=1, seed=9, lam=0.0855, episodes=2,
                                       max_steps=12, res_token=run + "-bad"))
        assert ei.value.tag == "config_epoch_mismatch"
        with pytest.raises(ControlError) as ei2:
            t.reconfigure(ActorConfig("other.json", DATA_FACES, m=4, n_sims=16, c_puct=1.25,
                                      c_visit=50.0, c_scale=1.0, c_outcome=2, max_depth=24))
        assert ei2.value.tag == "instance_knob_changed"
        # the runner survived both rejects and still serves (ping after the errors).
        assert t.ping().serving is True
    finally:
        t.close()
        conn.close()


@pytest.mark.skipif(not (_RUN_CPP and os.path.exists(CPP_BIN)), reason=_CPP_SKIP)
def test_cpp_actor_executor_drives_persistent_runner():
    """CppActorExecutor (the exit_loop generation executor) drives the persistent --serve runner over the
    ActorTransport end-to-end: a generate produces real (X,PI,M,Y) records, and a SECOND generate with a
    changed hot_search (n_sims 8->16) live-reconfigures the SAME runner (no respawn) and still produces
    records. This is the exit_loop-facing proof of the transport switch — the generate/evaluate/close
    contract is unchanged, so the loop is oblivious to it."""
    if not _redis_up():
        pytest.skip("redis not reachable on the CHOCO_TRANSPORT_REDIS_* contract")
    from chocofarm.az.cpp_executor import CppActorExecutor
    from chocofarm.az.features import feature_dim
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.model.env import Environment

    env = Environment()
    in_dim, n_slots = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=32, n_actions=n_slots, seed=0)
    net.set_value_scale(0.0, 1.0)
    ex = CppActorExecutor(CPP_BIN, DATA_INSTANCE, DATA_FACES, env, base_seed=7, use_jax_mlp=False,
                          in_dim=in_dim, n_slots=n_slots)
    try:
        hs = dict(m=4, n_sims=8, c_puct=1.25, c_visit=50.0, c_scale=1.0, c_outcome=2, max_depth=24)
        recs0 = ex.generate(net, 0, [0] * 6, 0.0855, 0, 1.0, None, hot_search=hs, max_steps=12)
        assert len(recs0) > 0, "the persistent runner produced no transitions"
        # the SAME runner live-reconfigures (n_sims 8->16, no respawn) and generates again at a new version.
        recs1 = ex.generate(net, 1, [0] * 6, 0.0855, 0, 1.0, None,
                            hot_search=dict(hs, n_sims=16), max_steps=12)
        assert len(recs1) > 0
        # each record is the (feat, pi, mask, g) shape exit_loop consumes.
        feat, pi, mask, g = recs0[0]
        assert feat.shape == (in_dim,)
        assert pi.shape == (n_slots,) and mask.shape == (n_slots,)
        assert isinstance(g, float)
    finally:
        ex.close()
