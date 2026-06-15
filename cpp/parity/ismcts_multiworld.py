#!/usr/bin/env python3
"""
cpp/parity/ismcts_multiworld.py — the MULTI-DETERMINIZATION logic parity check for the C++ ISMCTS.

Closes the coverage hole the independent review (docs/notes/ismcts-port-review-2026-06-16.md) named the
collapsed-determinization: cpp/parity/ismcts_logic.py scripts sample_world -> bw[0], so EVERY iteration
determinizes to the SAME world and NO action edge ever accumulates >1 belief_key child — the
ISMCTS-DEFINING multi-belief information-set sub-child split (ismcts.py:175-179 / ismcts.cpp:172-180) is
never exercised, and a mutation collapsing _belief_key changes 0/240 selected actions (the logic check
is blind to it). Here we script sample_world to CYCLE distinct worlds of `bw` (bw[idx mod len(bw)]) off a
fixed world-index FIFO fed IDENTICALLY to both languages, so a single action at a node resolves to
DIFFERENT observation outcomes across iterations -> multiple (action, belief_key) children.

Two assertions, held to the SAME standard the tie-break verification used:
  (1) PARITY: the committed C++ ISMCTS selects the SAME action as the CORRECT Python ISMCTS on every
      fixed input across the grid (exact-action identity over the multi-belief routing).
  (2) DISCRIMINATION (the non-vacuity / sensitivity control): a Python MUTANT with _belief_key collapsed
      to a constant — exactly the multi-belief-routing bug — must SELECT A DIFFERENT action from correct
      Python on some inputs. If it does, these inputs genuinely depend on the belief_key sub-child
      routing, so (1)'s agreement is MEANINGFUL, not vacuous. If the mutant never diverged, the fixture
      would prove nothing — the exact failure mode of the bw[0] logic check, which we refuse to repeat.

The three RNG-free seams are fed identically to both sides: sample_world (the world-index FIFO mod
len(bw)), rng.integers (the expansion FIFO mod n), and the leaf value (the leaf FIFO). The world FIFO is
consumed once per top-level iteration (decide calls env.sample_world once per walk), the other two during
descent — the same call order on both sides.

Run (from repo root):
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python cpp/parity/ismcts_multiworld.py

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from chocofarm.model.env import TERMINATE, Environment
from chocofarm.az.actions import action_to_slot, term_slot
from chocofarm.solvers import ismcts as ismcts_mod
from chocofarm.solvers.ismcts import ISMCTSConfig, ISMCTSPolicy

ISMCTS_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-ismcts-dump")
INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")


class _ScriptedRng:
    """Delivers the next expansion-index off the FIFO, mod the requested bound (the C++
    ScriptedISMCTSSource::expand_index mirror)."""

    def __init__(self, idxs):
        self.idxs = idxs
        self.i = 0

    def integers(self, n):
        raw = int(self.idxs[self.i % len(self.idxs)])
        self.i += 1
        return raw % n


def _scripted_base_value_fifo(table):
    """_base_value replacement delivering the next leaf off `table` in call order, cycled."""
    state = {"i": 0}

    def _bv(env, base, loc, bw, collected, world, lam):
        i = state["i"]
        state["i"] += 1
        return table[i % len(table)]

    return _bv


def _cycled_sample_world(world_fifo):
    """sample_world replacement: cycle bw[world_fifo[i] mod len(bw)] in call order — the C++
    ScriptedISMCTSSource mirror with a non-empty world-index FIFO. The per-construction counter resets
    each decide call (one source per call), matching the fresh C++ process (widx_ = 0)."""
    state = {"i": 0}

    def _sw(self, bw, rng):
        raw = int(world_fifo[state["i"] % len(world_fifo)])
        state["i"] += 1
        n = len(bw)
        return int(bw[((raw % n) + n) % n])

    return _sw


def _advance_prefix(env, prefix_slots):
    """Advance (loc, bw, collected) by the prefix against the true world bw[0] — the same deterministic
    world both sides advance by (identical to ismcts_logic.py / ismcts_dump.cpp --prefix)."""
    loc, bw, collected = ("w", env.entry), env.worlds, set()
    world0 = int(bw[0])
    for slot in prefix_slots:
        if len(bw) == 0:
            break
        if slot >= env.N + len(env.detectors):
            break
        a = ("t", slot) if slot < env.N else ("d", slot - env.N)
        _, loc, bw, collected, _ = env.apply(loc, bw, collected, a, world0)
    return loc, bw, collected


def py_select(env, cfg, lam, prefix_slots, idx_fifo, leaf_fifo, world_fifo, mutate_belief_key=False):
    """Run Python ISMCTS with the cycled-world seam. `mutate_belief_key` collapses _belief_key to a
    constant (the multi-belief-routing bug) for the discrimination control."""
    loc, bw, collected = _advance_prefix(env, prefix_slots)
    pol = ISMCTSPolicy(cfg=cfg)
    bv_fn = _scripted_base_value_fifo(leaf_fifo)
    sw_fn = _cycled_sample_world(world_fifo)
    orig_sample = Environment.sample_world
    orig_bv = ismcts_mod._base_value
    orig_bk = ismcts_mod._belief_key
    Environment.sample_world = sw_fn
    ismcts_mod._base_value = bv_fn
    if mutate_belief_key:
        ismcts_mod._belief_key = lambda bw_: (0, 0, 0)  # collapse all determinizations to one child
    try:
        action = pol.decide(env, loc, bw, collected, lam, rng=_ScriptedRng(idx_fifo))
    finally:
        Environment.sample_world = orig_sample
        ismcts_mod._base_value = orig_bv
        ismcts_mod._belief_key = orig_bk
    return term_slot(env) if action == TERMINATE else action_to_slot(env, action)


def cpp_select(cfg, lam, prefix_slots, idx_fifo, leaf_fifo, world_fifo):
    """Run the committed C++ ismcts-dump with the SAME config / prefix / FIFOs incl. the world FIFO
    (stdin line 3). The C++ uses the CORRECT belief_key; we verify it matches CORRECT Python."""
    cmd = [ISMCTS_BIN, "--instance", INSTANCE, "--faces", FACES,
           "--iterations", str(cfg.iterations), "--max-depth", str(cfg.max_depth),
           "--c", repr(cfg.c), "--lam", repr(lam)]
    if prefix_slots:
        cmd += ["--prefix", " ".join(str(s) for s in prefix_slots)]
    stdin = (" ".join(str(int(v)) for v in idx_fifo) + "\n"
             + " ".join(repr(float(v)) for v in leaf_fifo) + "\n"
             + " ".join(str(int(v)) for v in world_fifo) + "\n")
    out = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ismcts-dump failed (rc={out.returncode}): {out.stderr}")
    return int(out.stdout.split()[0])


def main():
    if not os.path.exists(ISMCTS_BIN):
        print(f"FAIL: C++ ismcts-dump not built at {ISMCTS_BIN}\n"
              f"      build it: cmake -S cpp -B cpp/build && cmake --build cpp/build")
        return 1

    env = Environment()
    print("=== multi-determinization logic check: C++ ISMCTS vs Python ISMCTS (CYCLED worlds) ===")
    print(f"instance: N={env.N} K={env.K} nDet={len(env.detectors)} |worlds|={len(env.worlds)}")
    print("seam: sample_world -> bw[world_fifo[i] mod len(bw)] (NOT bw[0]); expand FIFO mod n; leaf FIFO")
    print("assert: (1) committed C++ == correct Python (exact action); (2) a belief_key-collapse mutant")
    print("        DIVERGES from correct Python on some inputs (proves the inputs exercise the routing)\n")

    rng = np.random.default_rng(20260616)
    idx_fifo = rng.integers(0, 97, size=2048).tolist()
    leaf_fifo = (rng.standard_normal(2048) * 1.7).tolist()
    world_fifo = rng.integers(0, 100003, size=2048).tolist()   # cycle distinct worlds (mod len(bw))

    cases = []
    for iters in (4, 16, 64, 300):
        for max_depth in (4, 24):
            for c in (0.0, 0.7):
                for lam in (0.0, 0.1, 0.35):
                    for prefix in ([], [25], [25, 27], [5]):
                        cases.append((iters, max_depth, c, prefix, lam))

    n_ok = 0
    n_disc = 0
    for iters, max_depth, c, prefix, lam in cases:
        cfg = ISMCTSConfig(iterations=iters, c=c, max_depth=max_depth)
        py = py_select(env, cfg, lam, prefix, idx_fifo, leaf_fifo, world_fifo)
        cp = cpp_select(cfg, lam, prefix, idx_fifo, leaf_fifo, world_fifo)
        if py == cp:
            n_ok += 1
        else:
            print(f"  PARITY MISMATCH iters={iters} md={max_depth} c={c} prefix={prefix} "
                  f"lam={lam}: py={py} cpp={cp}")
        mut = py_select(env, cfg, lam, prefix, idx_fifo, leaf_fifo, world_fifo, mutate_belief_key=True)
        if mut != py:
            n_disc += 1

    total = len(cases)
    print(f"\n[parity]         {n_ok}/{total} cases: committed C++ == correct Python (exact action)")
    print(f"[discrimination] {n_disc}/{total} cases: belief_key-collapse MUTANT diverges from correct Python")

    if n_disc == 0:
        print("\nRESULT: FAIL — VACUOUS: the belief_key-collapse mutant never diverged, so these inputs do")
        print("NOT exercise the multi-belief routing — the fixture would prove nothing (the bw[0] failure")
        print("mode). Widen the world FIFO / iterations until the routing is genuinely exercised.")
        return 1
    if n_ok != total:
        print("\nRESULT: FAIL — the committed C++ multi-belief routing DIVERGES from correct Python on the")
        print("multi-determinization path (a genuine behavioral defect, OR a bw-ordering mismatch between")
        print("the C++ env.worlds() and numpy env.worlds — investigate which before concluding).")
        return 1
    print("\nRESULT: PASS — the C++ multi-belief sub-child routing is action-identical to Python on inputs")
    print("PROVEN sensitive to belief_key routing (the mutant diverges), so the agreement is not vacuous.")
    print("The collapsed-determinization coverage hole is closed with an executed exact-action test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
