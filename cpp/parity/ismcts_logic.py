#!/usr/bin/env python3
"""
cpp/parity/ismcts_logic.py — the DETERMINISTIC logic parity check for the C++ ISMCTS (ADR-0012 P6).

This validates the ISMCTS SELECTION + NESTING logic — the part that MUST be exact — INDEPENDENT of
RNG. RNG enters ISMCTS through THREE draws: (a) world-sampling per iteration, (b) the expansion-index
draw, and (c) the leaf base playout. We make all three RNG-free on BOTH sides:

  * sample_world(bw) -> bw[0]      (the lowest-bitmask world; itertools/combinations order is the
                                    same in C++ build_worlds and numpy world_array, and both filters
                                    preserve order, so each iteration's determinization is identical);
  * rng.integers(n)  -> the next value off a fixed EXPANSION-INDEX FIFO, taken mod n (so the scripted
                                    index is always a legal untried-list index; both sides apply the
                                    SAME mod n at the SAME call -> the SAME expanded action);
  * the leaf value   -> the next value off a fixed LEAF FIFO, consumed in CALL ORDER.

The descent is structurally identical across the two implementations, so the two FIFOs are consumed
in the SAME order on both sides. Feeding BOTH the SAME (sample_world=bw[0]) + (expand FIFO mod n) +
(leaf FIFO) on a FIXED (loc, belief, collected) input must therefore yield the SAME selected action
(the most-visited root action after `iterations` walks). We assert that for several fixed inputs at
a few (iterations, max_depth, c) settings — covering expansion, UCB selection, the availability
denominator (only reached when an action is re-selected after expansion, i.e. enough iterations),
the TERMINATE edge, and the most-visited final.

This is the SAME shape as the NMCS logic check (cpp/parity/nmcs_logic.py): an RNG-free seam fed
identically to both languages, asserting exact-action identity — NOT an aggregate stat, NOT redis.

C++ side: cpp/build/chocofarm-ismcts-dump (the scripted-source fixture).
Python side: chocofarm.solvers.ismcts.ISMCTSPolicy.decide, with env.sample_world (-> bw[0]),
             rng.integers (-> the expand FIFO mod n) and ismcts._base_value (-> the leaf FIFO) all
             scripted to the SAME tables the C++ ScriptedISMCTSSource uses.

Run (from repo root):
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python cpp/parity/ismcts_logic.py

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
    """A fake numpy.random.Generator-like object whose ONLY consumed method is `integers`. It
    delivers the next value off the expansion-index FIFO, taken mod the requested bound — exactly the
    C++ ScriptedISMCTSSource::expand_index (raw mod n, non-negative). ISMCTS's decide calls
    `int(rng.integers(len(untried)))`; this returns the scripted, mod-reduced index, so both sides
    expand the SAME untried action. `sample_world` is monkeypatched separately (it ignores rng), and
    the leaf value comes from the patched `_base_value`, so no other rng method is reached."""

    def __init__(self, idxs):
        self.idxs = idxs
        self.i = 0

    def integers(self, n):
        raw = int(self.idxs[self.i % len(self.idxs)])
        self.i += 1
        return raw % n  # n > 0 at every expansion (untried is non-empty); non-negative scripted idx


def _scripted_base_value_fifo(table):
    """Return a `_base_value` replacement delivering the next value off `table`, consumed in CALL
    ORDER and CYCLED modulo len(table) — mirroring the C++ ScriptedISMCTSSource::leaf_value exactly.
    The signature matches solvers.base._base_value(env, base, loc, bw, collected, world, lam)."""
    state = {"i": 0}

    def _bv(env, base, loc, bw, collected, world, lam):
        i = state["i"]
        state["i"] += 1
        return table[i % len(table)]

    return _bv, state


def py_select(env, cfg, lam, prefix_slots, idx_fifo, leaf_fifo):
    """Advance (loc, bw, collected) by `prefix_slots` against the true world bw[0], then run
    ISMCTSPolicy.decide with the scripted seam (sample_world=bw[0]; rng.integers=idx_fifo mod n;
    _base_value=leaf_fifo), and return the selected action SLOT — mirroring the C++ fixture exactly."""
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

    pol = ISMCTSPolicy(cfg=cfg)

    # monkeypatch the three RNG-free seams: env.sample_world (bw[0]), _base_value (leaf FIFO); the
    # expansion-index draw is fed via the _ScriptedRng passed AS the rng (decide calls rng.integers).
    bv_fn, _bv_state = _scripted_base_value_fifo(leaf_fifo)
    orig_sample = Environment.sample_world
    orig_bv = ismcts_mod._base_value
    Environment.sample_world = lambda self, bw, rng: int(bw[0])
    ismcts_mod._base_value = bv_fn
    try:
        action = pol.decide(env, loc, bw, collected, lam, rng=_ScriptedRng(idx_fifo))
    finally:
        Environment.sample_world = orig_sample
        ismcts_mod._base_value = orig_bv

    return term_slot(env) if action == TERMINATE else action_to_slot(env, action)


def cpp_select(cfg, lam, prefix_slots, idx_fifo, leaf_fifo):
    """Run the C++ ismcts-dump fixture with the same config / prefix / FIFOs; return the slot."""
    cmd = [ISMCTS_BIN, "--instance", INSTANCE, "--faces", FACES,
           "--iterations", str(cfg.iterations), "--max-depth", str(cfg.max_depth),
           "--c", repr(cfg.c), "--lam", repr(lam)]
    if prefix_slots:
        cmd += ["--prefix", " ".join(str(s) for s in prefix_slots)]
    idx_str = " ".join(str(int(v)) for v in idx_fifo)
    leaf_str = " ".join(repr(float(v)) for v in leaf_fifo)
    stdin = idx_str + "\n" + leaf_str + "\n"
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
    print("=== ADR-0012 P6 deterministic logic check: C++ ISMCTS vs Python ISMCTS ===")
    print(f"instance: N={env.N} K={env.K} nDet={len(env.detectors)} |worlds|={len(env.worlds)}")
    print("RNG-free seam: sample_world -> bw[0]; rng.integers -> a fixed FIFO (mod n); "
          "leaf -> a fixed FIFO (same tables on both sides)")
    print("assert: SAME most-visited action slot for fixed (loc, belief) inputs across "
          "(iterations, max_depth, c) settings\n")

    # Two fixed pseudo-random tables, CYCLED in call order by both sides (so they never exhaust under
    # a 300-iteration search). The expand-index FIFO is a spread of small non-negative ints (mod n at
    # each call selects an untried action); the leaf FIFO is a spread of positive/negative values so
    # the UCB exploit term actually discriminates between arms and the most-visited final is decided.
    # The values round-trip exactly through repr()/atof (full float64), so the C++ tables are
    # byte-identical to these.
    rng = np.random.default_rng(20260615)
    idx_fifo = rng.integers(0, 97, size=2048).tolist()       # scripted expansion indices (mod n at use)
    leaf_fifo = (rng.standard_normal(2048) * 1.7).tolist()   # scripted leaf returns

    # test cases: (iterations, max_depth, c, prefix_slots, lam). Iterations vary so we cover the pure
    # expansion regime (few iterations -> only fresh expansions) AND the selection+availability regime
    # (more iterations -> UCB select with the n'_j denominator on re-visited arms). Prefixes advance
    # the real state so the fixed input spans root AND mid-episode (sharper belief, fewer legal arms,
    # the subset-armed availability path). lam varies the TERMINATE-edge vs step trade-off.
    cases = []
    for iters in (1, 4, 16, 64, 300):
        for max_depth in (4, 24):
            for c in (0.0, 0.7):
                for lam in (0.0, 0.1, 0.35):
                    for prefix in ([], [25], [25, 27], [5]):  # detector/treasure prefixes (slot ids)
                        cases.append((iters, max_depth, c, prefix, lam))

    n_ok = 0
    by_iters = {}
    for iters, max_depth, c, prefix, lam in cases:
        cfg = ISMCTSConfig(iterations=iters, c=c, max_depth=max_depth)
        py_slot = py_select(env, cfg, lam, prefix, idx_fifo, leaf_fifo)
        cpp_slot = cpp_select(cfg, lam, prefix, idx_fifo, leaf_fifo)
        if py_slot == cpp_slot:
            n_ok += 1
            by_iters[iters] = by_iters.get(iters, 0) + 1
        else:
            print(f"  MISMATCH iters={iters} max_depth={max_depth} c={c} prefix={prefix} "
                  f"lam={lam}: py_slot={py_slot} cpp_slot={cpp_slot}")

    total = len(cases)
    coverage = " ".join(f"iters={k}:{by_iters.get(k, 0)}" for k in (1, 4, 16, 64, 300))
    print(f"[logic] {n_ok}/{total} fixed-input cases agree on the selected action  ({coverage})")
    if n_ok == total:
        print("\nRESULT: PASS — the C++ ISMCTS selection + nesting is action-identical to Python's on "
              "identical scripted world/expansion/leaf draws (expansion, UCB select, the availability "
              "denominator, the TERMINATE edge, and the most-visited final all covered)")
        return 0
    print("\nRESULT: FAIL — a fixed-input selection diverged (the selection/nesting logic differs)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
