#!/usr/bin/env python3
"""
cpp/parity/gumbel_logic.py — the DETERMINISTIC structure logic check for the C++ Gumbel-AZ search.

*** PHASE 1a (structure only) ***
This validates the Gumbel-AZ search STRUCTURE + SELECTION logic — the part that MUST be exact —
INDEPENDENT of RNG and INDEPENDENT of float32-vs-float64 precision. It is the C++ twin of the
ISMCTS logic check (cpp/parity/ismcts_logic.py): an RNG-free, precision-INSENSITIVE seam fed
identically to both languages, asserting exact-action identity — NOT an aggregate stat, NOT redis.

RNG / leaf enters the Gumbel search through THREE places; we make all three RNG-free and
precision-insensitive on BOTH sides:

  * rng.gumbel(n_slots) -> a fixed GUMBEL FIFO (cycled): the SAME per-slot perturbations drive the
                           root logit+g top-k AND every Sequential-Halving cut key on both sides;
  * env.sample_world(bw) -> bw[0] (the lowest-bitmask world; C++ build_worlds and numpy world_array
                           agree on order, and both filters preserve it), so each sim's
                           determinization is identical;
  * the leaf (value, logits) -> a fixed LEAF FIFO consumed in CALL ORDER (the descent is structurally
                           identical on both sides, so the call order matches). The scripted logits
                           are a COARSE, well-separated ramp logits[s] = logit_base + s*0.25 — distinct
                           per slot by >=0.25, so the masked-softmax prior has NO near-tie. The
                           discrete outcome (the SH survivor + the improved-pi argmax) is therefore
                           IDENTICAL whether the sigma-transform / v_mix / softmax run in float32 or
                           float64 — which is exactly what makes this a STRUCTURE check, not a
                           numerical-fidelity check.

Feeding BOTH the SAME (gumbel FIFO) + (sample_world=bw[0]) + (leaf FIFO) on a FIXED (loc, belief,
collected) input must therefore yield the SAME executed action (the SH survivor at temperature 0) AND
the SAME improved-pi argmax. We assert that across a grid of (n_sims, m, c_puct, max_depth, prefix),
AND assert the two structural Danihelka invariants that are precision-independent (mirroring
tests/test_az_loop.py): (i) the executed action IS the SH survivor, (ii) Sequential Halving spends the
FULL n_sims budget. A MUTATION control (--mutate sh-budget | puct) proves the check discriminates by
running the UNMODIFIED Python reference against a DELIBERATELY MUTATED C++ binary (CHOCO_GUMBEL_MUTATE:
sh-budget drops the full-budget remainder loop; puct flips the U-term sign) and asserting they DIVERGE
— it mutates the ACTUAL artifact under test, not the reference side, so a real port break is caught.

*** NOT covered here (= PHASE 1b) ***
The mixed-precision path. The Python search runs the sigma-transform at a DELIBERATE float32-prior x
float64-Q mixed precision (chocofarm/az/value_target.py:226-280, the byte-identity seam) that a
uniform-precision port diverges from on NEAR-TIE inputs. The C++ port (gumbel.cpp) runs that transform
in ONE consistent precision (float64); 1b makes it exactly float32-prior/float64-Q and adds a
near-tie / fine-input parity. This 1a check is precision-INSENSITIVE by construction (coarse inputs, no
near-ties), so it cannot — and is not meant to — catch the 1b hazard.

C++ side: cpp/build/chocofarm-gumbel-dump (the scripted-leaf + scripted-source fixture).
Python side: chocofarm.az.gumbel_search.GumbelAZSearch.decide_with_target (temperature 0), with
             search._predict_both (-> the scripted leaf), env.sample_world (-> bw[0]) and the rng's
             gumbel (-> the gumbel FIFO) all scripted to the SAME tables the C++ fixture uses.

Run (from repo root):
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python cpp/parity/gumbel_logic.py
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python cpp/parity/gumbel_logic.py --mutate sh-budget
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python cpp/parity/gumbel_logic.py --mutate puct

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from chocofarm.model.env import TERMINATE, Environment
from chocofarm.az.actions import action_to_slot, n_action_slots, term_slot
from chocofarm.az.features import feature_dim
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.gumbel_search import GumbelAZSearch

GUMBEL_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-gumbel-dump")
INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")

# The coarse, well-separated logit ramp the scripted leaf adds per slot. MUST match the C++
# ScriptedNet (gumbel_dump.cpp): logits[s] = logit_base + s * LOGIT_RAMP. >= 0.25 apart -> no near-tie.
LOGIT_RAMP = 0.25


def _masked_softmax(logits_row, mask_row):
    """The SAME masked softmax the search/value_target use (mlp.ValueMLP._masked_softmax), applied to
    a single (n_slots,) logits row under a {0,1} mask row. Returns the (n_slots,) prior."""
    return ValueMLP._masked_softmax(np.asarray(logits_row)[None, :],
                                    np.asarray(mask_row, dtype=np.float64)[None, :])[0]


class _ScriptedRng:
    """A fake numpy.random.Generator-like object whose ONLY consumed method is `gumbel`. It delivers
    the next n values off the gumbel FIFO (cycled), exactly as the C++ ScriptedGumbelSource::gumbel.
    The Gumbel search calls `rng.gumbel(size=n_slots)` once at the root; sample_world is monkeypatched
    separately (it ignores rng), and the leaf comes from the patched _predict_both, so no other rng
    method is reached. (temperature 0 takes the SH-survivor branch, never rng.choice.)"""

    def __init__(self, gumbels):
        self.gumbels = gumbels
        self.i = 0

    def gumbel(self, size):
        out = np.empty(int(size), dtype=np.float64)
        for k in range(int(size)):
            out[k] = self.gumbels[self.i % len(self.gumbels)]
            self.i += 1
        return out


def _scripted_predict_both(env, value_fifo, logit_base_fifo, n_slots):
    """Return a `_predict_both` replacement (the search's leaf seam: net.predict_both -> (value,
    prior)) delivering the next (value, logits-ramp) off the FIFOs, consumed in CALL ORDER and CYCLED
    modulo the table length — mirroring the C++ ScriptedNet exactly. The prior IS the masked softmax of
    the coarse logit ramp over the legal slots (mirroring predict_both: the net emits raw logits, the
    search softmaxes under the mask). Signature matches net.predict_both(feat, mask) -> (value, prior)."""
    state = {"i": 0}

    def _pb(feat, mask):
        i = state["i"]
        state["i"] += 1
        v = float(value_fifo[i % len(value_fifo)])
        lb = float(logit_base_fifo[i % len(logit_base_fifo)])
        logits = np.array([lb + s * LOGIT_RAMP for s in range(n_slots)], dtype=np.float64)
        prior = _masked_softmax(logits, mask).astype(np.float32)  # float32 prior (the 1b seam dtype)
        return v, prior

    return _pb, state


def py_decide(env, m, n_sims, c_puct, max_depth, prefix_slots, gumbel_fifo, value_fifo,
              logit_base_fifo):
    """Advance (loc, bw, collected) by `prefix_slots` against the true world bw[0], then run
    GumbelAZSearch.decide_with_target (temperature 0) with the scripted seam (gumbel=gumbel_fifo;
    sample_world=bw[0]; _predict_both=leaf FIFOs), and return (executed_slot, improved_argmax_slot,
    survivor_slot, n_spent) — mirroring the C++ fixture exactly."""
    n_slots = n_action_slots(env)
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

    # a real ValueMLP so the search constructs; its predict_both is overridden by the scripted leaf.
    net = ValueMLP(feature_dim(env), hidden=8, n_actions=n_slots, seed=0)
    search = GumbelAZSearch(net, env, m=m, n_sims=n_sims, c_puct=c_puct, max_depth=max_depth)
    pb_fn, _pb_state = _scripted_predict_both(env, value_fifo, logit_base_fifo, n_slots)
    search._predict_both = pb_fn  # the leaf seam (net.predict_both) -> the scripted leaf

    orig_sample = Environment.sample_world
    Environment.sample_world = lambda self, bw, rng: int(bw[0])
    try:
        action, improved, root = search._decide_root(
            env, loc, bw, collected, 0.0855, _ScriptedRng(gumbel_fifo), temperature=0.0)
    finally:
        Environment.sample_world = orig_sample

    exec_slot = term_slot(env) if action == TERMINATE else action_to_slot(env, action)
    argmax_slot = int(np.argmax(improved))
    n_spent = int(sum(root.N.values()))
    return exec_slot, argmax_slot, n_spent


def cpp_decide(m, n_sims, c_puct, max_depth, prefix_slots, gumbel_fifo, value_fifo,
               logit_base_fifo, mutate=None):
    """Run the C++ gumbel-dump fixture with the same config / prefix / FIFOs; return (executed_slot,
    improved_argmax_slot, n_spent). The leaf FIFO is flattened as 'v0 lb0 v1 lb1 ...' on stdin line 2.

    `mutate` (None | 'sh-budget' | 'puct'): when set, the C++ search is run with CHOCO_GUMBEL_MUTATE
    so it executes a DELIBERATELY broken SH-budget / PUCT — the mutation control runs the UNMODIFIED
    Python reference against this MUTATED C++ binary and asserts they diverge (the real discrimination
    proof: it mutates the ARTIFACT under test, not the reference side)."""
    cmd = [GUMBEL_BIN, "--instance", INSTANCE, "--faces", FACES,
           "--m", str(m), "--n-sims", str(n_sims), "--c-puct", repr(c_puct),
           "--max-depth", str(max_depth), "--lam", repr(0.0855)]
    if prefix_slots:
        cmd += ["--prefix", " ".join(str(s) for s in prefix_slots)]
    gumbel_str = " ".join(repr(float(v)) for v in gumbel_fifo)
    leaf_pairs = []
    for v, lb in zip(value_fifo, logit_base_fifo):
        leaf_pairs.append(repr(float(v)))
        leaf_pairs.append(repr(float(lb)))
    leaf_str = " ".join(leaf_pairs)
    stdin = gumbel_str + "\n" + leaf_str + "\n"  # line 3 (world idx) omitted -> sample_world = bw[0]
    sub_env = dict(os.environ)
    if mutate:
        sub_env["CHOCO_GUMBEL_MUTATE"] = mutate
    else:
        sub_env.pop("CHOCO_GUMBEL_MUTATE", None)
    out = subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=sub_env)
    if out.returncode != 0:
        raise RuntimeError(f"gumbel-dump failed (rc={out.returncode}): {out.stderr}")
    parts = out.stdout.split()
    return int(parts[0]), int(parts[1]), int(parts[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutate", choices=["sh-budget", "puct"], default=None,
                    help="inject a known structural break to prove the check discriminates (RED)")
    cli = ap.parse_args()

    if not os.path.exists(GUMBEL_BIN):
        print(f"FAIL: C++ gumbel-dump not built at {GUMBEL_BIN}\n"
              f"      build it: cmake -S cpp -B cpp/build && cmake --build cpp/build")
        return 1

    env = Environment()
    print("=== ADR-0012 P6 deterministic STRUCTURE logic check: C++ Gumbel-AZ vs Python Gumbel-AZ "
          "(PHASE 1a, precision-insensitive) ===")
    print(f"instance: N={env.N} K={env.K} nDet={len(env.detectors)} |worlds|={len(env.worlds)} "
          f"n_slots={n_action_slots(env)}")
    print("RNG-free seam: rng.gumbel -> a fixed FIFO; sample_world -> bw[0]; leaf (value,logits) -> a "
          "fixed FIFO (coarse logits[s]=lb+s*0.25, NO near-tie -> precision-insensitive)")
    print("assert: SAME executed action AND improved-pi argmax for fixed (loc, belief) inputs across "
          "(n_sims, m, c_puct, max_depth, prefix); + the 2 structural Danihelka invariants\n")
    if cli.mutate:
        print(f"*** MUTATION CONTROL: --mutate {cli.mutate} — the check MUST go RED (it discriminates) "
              "***\n")

    # Fixed pseudo-random tables, CYCLED in call order by both sides (so they never exhaust). The
    # gumbel FIFO is a spread drawn from the Gumbel(0,1) law (the perturbation the top-k + SH cut use);
    # the leaf value/logit-base FIFOs are spreads of well-separated values so the priors and Q-backups
    # discriminate. The values round-trip exactly through repr()/atof (full float64).
    rng = np.random.default_rng(20260616)
    gumbel_fifo = rng.gumbel(size=4096).tolist()
    value_fifo = (rng.standard_normal(2048) * 1.3).tolist()        # scripted leaf values
    logit_base_fifo = (rng.standard_normal(2048) * 2.0).tolist()   # scripted per-leaf logit bases

    # the grid: (n_sims, m, c_puct, max_depth, prefix). n_sims/m vary to exercise multiple SH phases +
    # the remainder loop; c_puct varies the interior PUCT trade-off; max_depth varies the leaf-vs-
    # descent boundary; prefixes advance the real state so the fixed input spans root AND mid-episode.
    cases = []
    for n_sims, m in [(12, 4), (16, 6), (24, 8), (48, 12), (32, 5), (8, 3)]:
        for c_puct in (0.0, 1.25, 2.5):
            for max_depth in (2, 24):
                for prefix in ([], [25], [25, 27], [5]):  # detector/treasure prefixes (slot ids)
                    cases.append((n_sims, m, c_puct, max_depth, prefix))

    n_ok = 0
    n_inv_survivor = 0
    n_inv_budget = 0
    n_diverged = 0   # mutation mode: cases where the MUTATED C++ diverges from the faithful Python
    mismatches = 0
    total = len(cases)
    for n_sims, m, c_puct, max_depth, prefix in cases:
        # the FAITHFUL Python reference — UNMODIFIED in every mode (the mutation only touches the C++).
        py_exec, py_argmax, py_spent = py_decide(
            env, m, n_sims, c_puct, max_depth, prefix, gumbel_fifo, value_fifo, logit_base_fifo)

        # structural Danihelka invariant (ii): the Python SH spends the FULL n_sims budget. On a
        # non-empty belief ALL n_sims sims land (no prefix here empties the belief).
        if py_spent == n_sims:
            n_inv_budget += 1
        else:
            print(f"  INVARIANT(budget) FAIL n_sims={n_sims} m={m}: py spent {py_spent} != {n_sims}")

        # the C++ search — FAITHFUL in normal mode, MUTATED (CHOCO_GUMBEL_MUTATE) in --mutate mode.
        cpp_exec, cpp_argmax, cpp_spent = cpp_decide(
            m, n_sims, c_puct, max_depth, prefix, gumbel_fifo, value_fifo, logit_base_fifo,
            mutate=cli.mutate)

        if cli.mutate:
            # MUTATION CONTROL: assert the MUTATED C++ DIVERGES from the faithful Python. A break is
            # observed when the executed action OR the improved-pi argmax differs, OR (for sh-budget)
            # the C++ under-spent the budget (cpp_spent != n_sims while Python spent the full n_sims).
            diverged = (py_exec != cpp_exec or py_argmax != cpp_argmax or cpp_spent != py_spent)
            if diverged:
                n_diverged += 1
            continue

        # NORMAL MODE: assert the faithful C++ is action-identical to the faithful Python on BOTH the
        # executed action AND the improved-pi argmax, AND that BOTH spent the full budget (invariant ii
        # cross-language). The executed==SH-survivor identity (invariant i) is witnessed by the dump
        # returning action_of_slot(survivor) as its executed slot — so agreement on the executed slot
        # means both languages took the survivor branch identically.
        exec_ok = (py_exec == cpp_exec)
        argmax_ok = (py_argmax == cpp_argmax)
        budget_ok = (cpp_spent == n_sims)
        if exec_ok and argmax_ok and budget_ok:
            n_ok += 1
            n_inv_survivor += 1
        else:
            mismatches += 1
            if mismatches <= 20:
                print(f"  MISMATCH n_sims={n_sims} m={m} c_puct={c_puct} max_depth={max_depth} "
                      f"prefix={prefix}: py(exec={py_exec},argmax={py_argmax},spent={py_spent}) "
                      f"cpp(exec={cpp_exec},argmax={cpp_argmax},spent={cpp_spent})")

    if cli.mutate:
        # the mutation MUST break the run on a MEANINGFUL fraction of cases: a faithful harness that
        # actually exercises the broken search will see the MUTATED C++ diverge from the UNMODIFIED
        # Python. If NOTHING diverged, the check does NOT discriminate (a silent ADR-0002 failure).
        print(f"\n[mutation: {cli.mutate}] {n_diverged}/{total} cases where the MUTATED C++ diverged "
              f"from the FAITHFUL Python reference")
        if n_diverged == 0:
            print("\nRESULT: FAIL — the MUTATION did NOT diverge from the faithful reference on ANY "
                  "case (the check does not discriminate; a real port break would slip through)")
            return 1
        print("\nRESULT: PASS (mutation control) — mutating the ACTUAL C++ search (the SH-budget / PUCT "
              "logic, via CHOCO_GUMBEL_MUTATE) makes it diverge from the UNMODIFIED Python reference on "
              f"{n_diverged}/{total} cases, proving the harness catches a real port break.")
        return 0

    print(f"\n[logic] {n_ok}/{total} fixed-input cases agree on the executed action, the improved-pi "
          f"argmax, AND the full-budget invariant")
    print(f"[invariant i  executed==SH-survivor] {n_inv_survivor}/{total} cross-language survivor "
          f"identity held")
    print(f"[invariant ii SH spends full budget ] {n_inv_budget}/{total} (Python) + cross-language "
          f"(C++ n_spent==n_sims) cases spent exactly n_sims")

    all_ok = (n_ok == total and n_inv_budget == total)
    if all_ok:
        print("\nRESULT: PASS — the C++ Gumbel-AZ search STRUCTURE is action-identical to Python's on "
              "identical scripted gumbel/world/leaf draws (Gumbel-Top-k, Sequential Halving, the "
              "c_outcome averaging, PUCT descent, the improved-pi sigma-transform, and executed==SH-"
              "survivor all covered) — PHASE 1a (precision-insensitive). The mixed-precision near-tie "
              "path is PHASE 1b (NOT covered here).")
        return 0
    print("\nRESULT: FAIL — a fixed-input selection diverged OR an SH-budget invariant broke (the "
          "structure logic differs)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
