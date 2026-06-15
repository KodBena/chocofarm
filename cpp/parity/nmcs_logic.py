#!/usr/bin/env python3
"""
cpp/parity/nmcs_logic.py — the DETERMINISTIC logic parity check for the C++ NMCS (ADR-0012 P6).

This validates the NMCS NESTING + SELECTION logic — the part that MUST be exact — INDEPENDENT of
RNG. RNG enters NMCS only through world-sampling, so we make the search RNG-free on BOTH sides:

  * sample_world(bw) -> bw[0]      (the lowest-bitmask world; itertools/combinations order is the
                                    same in C++ build_worlds and numpy world_array, and both filters
                                    preserve order, so the forward-played world is identical);
  * the level-0 playout value      -> the next value off a fixed FIFO, consumed in CALL ORDER.

The recursion is structurally identical across the two implementations, so the FIFO is consumed in
the SAME order on both sides. Feeding BOTH the SAME FIFO + bw[0] sampler on a FIXED (loc, belief,
collected) input must therefore yield the SAME selected first action. We assert that for several
fixed inputs at level 1 AND level 2 (the milestone).

C++ side: cpp/build/chocofarm-nmcs-dump (the scripted-source fixture).
Python side: chocofarm.solvers.nmcs.NMCSPolicy._search, with `_playout` and `env.sample_world`
             monkeypatched to the SAME FIFO + bw[0] rule.

Run (from repo root):
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python cpp/parity/nmcs_logic.py

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
from chocofarm.solvers.nmcs import NMCSConfig, NMCSPolicy

NMCS_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-nmcs-dump")
INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")


def _scripted_playout_fifo(table):
    """Return a `_playout` replacement that delivers the next value off `table`, consumed in CALL
    ORDER and CYCLED modulo len(table) — mirroring the C++ ScriptedSource::playout_value exactly (a
    level-2 search consumes far more leaf values than the table holds, so both sides cycle the SAME
    table the SAME way). The signature matches NMCSPolicy._playout (self, env, loc, bw, collected,
    lam, rng)."""
    state = {"i": 0}

    def _playout(self, env, loc, bw, collected, lam, rng):
        i = state["i"]
        state["i"] += 1
        return table[i % len(table)]

    return _playout, state


def py_select(env, cfg, lam, prefix_slots, fifo):
    """Advance (loc, bw, collected) by `prefix_slots` against the true world bw[0], then run
    NMCSPolicy._search with the scripted playout FIFO + a bw[0] sampler, and return the selected
    first-action SLOT and the search score — mirroring the C++ fixture exactly."""
    # advance the real state by the prefix (the same deterministic world both sides advance by).
    loc, bw, collected = ("w", env.entry), env.worlds, set()
    world0 = int(bw[0])
    for slot in prefix_slots:
        if len(bw) == 0:
            break
        if slot >= env.N + len(env.detectors):
            break  # TERMINATE in prefix
        a = ("t", slot) if slot < env.N else ("d", slot - env.N)
        _, loc, bw, collected, _ = env.apply(loc, bw, collected, a, world0)

    pol = NMCSPolicy(cfg=cfg)

    # monkeypatch _playout (FIFO) and env.sample_world (bw[0]) — the RNG-free deterministic seam.
    play_fn, _state = _scripted_playout_fifo(fifo)
    orig_playout = NMCSPolicy._playout
    orig_sample = Environment.sample_world
    NMCSPolicy._playout = play_fn
    Environment.sample_world = lambda self, bw, rng: int(bw[0])
    try:
        level = max(1, cfg.level)
        score, action = pol._search(env, loc, bw, collected, lam, level, rng=None)
    finally:
        NMCSPolicy._playout = orig_playout
        Environment.sample_world = orig_sample

    slot = term_slot(env) if action == TERMINATE else action_to_slot(env, action)
    return slot, score


def cpp_select(cfg, lam, prefix_slots, fifo):
    """Run the C++ nmcs-dump fixture with the same config / prefix / FIFO; return (slot, score)."""
    cmd = [NMCS_BIN, "--instance", INSTANCE, "--faces", FACES,
           "--level", str(cfg.level), "--cand-det", str(cfg.cand_det),
           "--cand-tre", str(cfg.cand_tre), "--step-samples", str(cfg.step_samples),
           "--max-steps", str(cfg.max_steps), "--lam", repr(lam)]
    if prefix_slots:
        cmd += ["--prefix", " ".join(str(s) for s in prefix_slots)]
    fifo_str = " ".join(repr(float(v)) for v in fifo)
    out = subprocess.run(cmd, input=fifo_str + "\n", capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"nmcs-dump failed (rc={out.returncode}): {out.stderr}")
    parts = out.stdout.split()
    return int(parts[0]), float(parts[1])


def main():
    if not os.path.exists(NMCS_BIN):
        print(f"FAIL: C++ nmcs-dump not built at {NMCS_BIN}\n"
              f"      build it: cmake -S cpp -B cpp/build && cmake --build cpp/build")
        return 1

    env = Environment()
    print("=== ADR-0012 P6 deterministic logic check: C++ NMCS vs Python NMCS ===")
    print(f"instance: N={env.N} K={env.K} nDet={len(env.detectors)} |worlds|={len(env.worlds)}")
    print("RNG-free seam: sample_world -> bw[0]; playout_value -> a fixed FIFO (same on both sides)")
    print("assert: SAME selected first-action slot for fixed (loc, belief) inputs, level-1 AND level-2\n")

    # a deterministic, varied table of leaf values, CYCLED in call order by both sides (so it never
    # exhausts under a level-2 search). A fixed pseudo-random spread of positive/negative values so
    # the argmax actually discriminates between candidates and lines. The values round-trip exactly
    # through repr()/atof (full float64), so the C++ table is byte-identical to this one.
    rng = np.random.default_rng(20260615)
    fifo = (rng.standard_normal(4096) * 1.7).tolist()

    # test cases: (level, prefix_slots, lam, candidate widths). Prefixes advance the real state so the
    # fixed search input spans root AND mid-episode states; widths vary the branching.
    cases = []
    for level in (1, 2):
        for cand in ((4, 4), (2, 3), (3, 2)):
            for lam in (0.0, 0.1, 0.35):
                for prefix in ([], [25], [25, 27], [5]):  # detector/treasure prefixes (slot ids)
                    cases.append((level, prefix, lam, cand))

    n_ok = 0
    n_l1 = n_l2 = 0
    for level, prefix, lam, (cd, ct) in cases:
        cfg = NMCSConfig(level=level, playout_samples=3, step_samples=2,
                         cand_det=cd, cand_tre=ct, max_steps=24)
        py_slot, py_score = py_select(env, cfg, lam, prefix, fifo)
        cpp_slot, cpp_score = cpp_select(cfg, lam, prefix, fifo)
        same = (py_slot == cpp_slot)
        # the score is the SAME float64 math on identical inputs -> agree to roundoff
        score_close = abs(py_score - cpp_score) <= 1e-9 * max(1.0, abs(py_score))
        if same and score_close:
            n_ok += 1
            if level == 1:
                n_l1 += 1
            else:
                n_l2 += 1
        else:
            print(f"  MISMATCH level={level} prefix={prefix} lam={lam} cand=({cd},{ct}): "
                  f"py_slot={py_slot} cpp_slot={cpp_slot} py_score={py_score:.6f} "
                  f"cpp_score={cpp_score:.6f}")

    total = len(cases)
    print(f"[logic] {n_ok}/{total} fixed-input cases agree on the selected action AND the score "
          f"(level-1: {n_l1}, level-2: {n_l2})")
    if n_ok == total:
        print("\nRESULT: PASS — the C++ NMCS nesting + selection is action-identical to Python's on "
              "identical leaf returns, for level-1 and level-2 (the milestone)")
        return 0
    print("\nRESULT: FAIL — a fixed-input selection diverged (the nesting/selection logic differs)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
