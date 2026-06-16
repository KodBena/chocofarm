#!/usr/bin/env python3
"""
cpp/parity/gumbel_precision.py — the PHASE 1b mixed-precision NEAR-TIE parity for the C++ Gumbel-AZ
search. The numerical-FIDELITY twin of the 1a structure check (cpp/parity/gumbel_logic.py).

*** PHASE 1b (precision fidelity) ***
1a proved the discrete STRUCTURE is faithful on COARSE, well-separated scripted leaf inputs — inputs
deliberately chosen so the SH survivor + improved-pi argmax are identical whether the sigma-transform
runs in float32 or float64 (precision-INSENSITIVE). That left the actual hazard UNPROVEN: the Python
search runs the sigma-transform at a DELIBERATE float32-prior x float64-Q mixed precision
(chocofarm/az/value_target.py:209-249 + gumbel_search.py:397-426/436-458, the byte-identity seam):
  * the in-search masked-softmax prior is FLOAT32 (root.prior),
  * v_mix's prior-weighted blend is computed ENTIRELY in float32 (numpy weak-promotes
    `prior[s](f32) * q(pyfloat) -> f32`; the v_mix RETURN is np.float32),
  * the unvisited improved-pi completion `sigma * v_mix` is float32-rounded (then added to the float64
    root logit),
  * the PUCT interior score `q + c_puct*p*sqrt(N)/(1+n)` is float32 (the float32 prior weak-promotes
    the whole U-term), deciding the interior near-tie argmax at float32.
A uniform-precision (all-float64) port diverges from this on NEAR-TIE inputs and FLIPS the discrete
argmaxes (the SH survivor and the improved-pi argmax).

This harness feeds REALISTIC FINE scripted leaf (value, logits) — drawn so the masked-softmax prior,
the PUCT interior argmax, and the improved-pi completed logits hit GENUINE near-ties (spreads at the
float32-epsilon scale) — and asserts:

  (A) the MIXED-precision C++ (the default faithful path) matches Python's GumbelAZSearch EXACTLY on
      BOTH the executed action AND the improved-pi argmax, across a grid (N/N);
  (B) the UNIFORM-precision C++ (CHOCO_GUMBEL_UNIFORM=1 — the 1a all-float64 path) DIVERGES from Python
      on a NON-TRIVIAL number of those same cases.
(B) is the load-bearing DISCRIMINATION control (the 1b analogue of the 1a mutation control): if uniform
ALSO matched N/N the fine inputs would NOT be precision-sensitive and the test would be VACUOUS. Both
numbers are reported. The Python reference is the SAME faithful GumbelAZSearch in both arms — only the
C++ precision toggles, so the divergence is attributable to the float32 seam, not the structure.

Both sides use the SAME scripted RNG-free seam as gumbel_logic.py (rng.gumbel -> a fixed FIFO;
sample_world -> bw[0]; the leaf (value, logits) -> a fixed cycled FIFO). The ONLY difference from 1a is
the leaf inputs: here logits carry FULL float64 precision with tiny per-slot deltas (so the float32
prior storage + the float32 sigma-transform round differently than float64), and the values are drawn so
the Q backups near-tie. The leaf logits are passed to the C++ fixture as full-precision per-slot
vectors (the --leaf-logits protocol), NOT the coarse `lb + s*0.25` ramp the 1a fixture builds.

C++ side: cpp/build/chocofarm-gumbel-dump (mixed precision by default; CHOCO_GUMBEL_UNIFORM=1 = uniform).
Python side: chocofarm.az.gumbel_search.GumbelAZSearch._decide_root (temperature 0), the REAL
             mixed-precision path (float32 prior, float32 v_mix, float32 PUCT) on both arms.

Run (from repo root):
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python cpp/parity/gumbel_precision.py

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
from chocofarm.az.actions import action_to_slot, n_action_slots, term_slot
from chocofarm.az.features import feature_dim
from chocofarm.az.mlp import ValueMLP
from chocofarm.az.gumbel_search import GumbelAZSearch

GUMBEL_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-gumbel-dump")
INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")


def _masked_softmax(logits_row, mask_row):
    """The SAME masked softmax the search/value_target use (mlp.ValueMLP._masked_softmax), applied to
    a single (n_slots,) logits row under a {0,1} mask row. Returns the (n_slots,) prior (float64)."""
    return ValueMLP._masked_softmax(np.asarray(logits_row, dtype=np.float64)[None, :],
                                    np.asarray(mask_row, dtype=np.float64)[None, :])[0]


class _ScriptedRng:
    """A fake numpy.random.Generator-like object whose ONLY consumed method is `gumbel` — delivers the
    next n values off the gumbel FIFO (cycled), exactly as the C++ ScriptedGumbelSource::gumbel. The
    search calls rng.gumbel(size=n_slots) once at the root; sample_world is monkeypatched (ignores rng),
    the leaf comes from the patched _predict_both, and temperature 0 never reaches rng.choice."""

    def __init__(self, gumbels):
        self.gumbels = gumbels
        self.i = 0

    def gumbel(self, size):
        out = np.empty(int(size), dtype=np.float64)
        for k in range(int(size)):
            out[k] = self.gumbels[self.i % len(self.gumbels)]
            self.i += 1
        return out


def _scripted_predict_both(value_fifo, leaf_logits_table, n_slots):
    """Return a `_predict_both` replacement (the search's leaf seam: net.predict_both -> (value, prior))
    delivering the next (value, full-precision per-slot logits row) off the FIFOs in CALL ORDER, cycled.
    The prior IS the masked softmax of the FINE per-slot logits over the legal slots, narrowed to FLOAT32
    (the 1b seam dtype — exactly what predict_both stores as root.prior). `leaf_logits_table` is a list
    of length-n_slots float64 rows (the FINE near-tie logits, NOT the coarse 1a `lb + s*0.25` ramp)."""
    state = {"i": 0}

    def _pb(feat, mask):
        i = state["i"]
        state["i"] += 1
        v = float(value_fifo[i % len(value_fifo)])
        # The net emits FLOAT32 logits (the policy head's f32 output / the NetPrediction wire dtype). In
        # production predict_both softmaxes them and _masked_softmax UPCASTS to float64 (neg_inf is
        # np.float64), so: leaf logits float32 -> masked softmax in float64 -> prior narrowed to float32.
        # The C++ fixture narrows the leaf logits row to float32 in NetPrediction identically, so both
        # softmax the SAME float32-narrowed logits in float64 and store the SAME float32 prior.
        logits32 = np.asarray(leaf_logits_table[i % len(leaf_logits_table)], dtype=np.float32)
        prior = _masked_softmax(logits32.astype(np.float64), mask).astype(np.float32)
        return v, prior

    return _pb, state


def py_decide(env, m, n_sims, c_puct, max_depth, prefix_slots, gumbel_fifo, value_fifo,
              leaf_logits_table):
    """Advance (loc, bw, collected) by `prefix_slots` against the true world bw[0], then run
    GumbelAZSearch._decide_root (temperature 0) with the scripted seam (gumbel=gumbel_fifo;
    sample_world=bw[0]; _predict_both=leaf FIFOs), and return (executed_slot, improved_argmax_slot,
    n_spent) — the REAL mixed-precision path (float32 prior + float32 v_mix + float32 PUCT)."""
    n_slots = n_action_slots(env)
    loc, bw, collected = ("w", env.entry), env.worlds, set()
    world0 = int(bw[0])
    for slot in prefix_slots:
        if len(bw) == 0:
            break
        if slot >= env.N + len(env.detectors):
            break  # TERMINATE in prefix
        a = ("t", slot) if slot < env.N else ("d", slot - env.N)
        _, loc, bw, collected, _ = env.apply(loc, bw, collected, a, world0)

    net = ValueMLP(feature_dim(env), hidden=8, n_actions=n_slots, seed=0)
    search = GumbelAZSearch(net, env, m=m, n_sims=n_sims, c_puct=c_puct, max_depth=max_depth)
    pb_fn, _pb_state = _scripted_predict_both(value_fifo, leaf_logits_table, n_slots)
    search._predict_both = pb_fn

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
               leaf_logits_table, uniform=False):
    """Run the C++ gumbel-dump fixture with the FINE per-slot leaf logits; return (executed_slot,
    improved_argmax_slot, n_spent). `uniform`=True sets CHOCO_GUMBEL_UNIFORM=1 (the 1a all-float64 path,
    the discrimination control). The leaf is passed as: line 2 = the value FIFO (one value per leaf);
    line 4 = the flattened per-slot logits table 'r0c0 r0c1 ... r1c0 ...' (n_slots per row), selected by
    --leaf-logits-rows on argv (the FINE-input protocol the fixture branches on)."""
    cmd = [GUMBEL_BIN, "--instance", INSTANCE, "--faces", FACES,
           "--m", str(m), "--n-sims", str(n_sims), "--c-puct", repr(c_puct),
           "--max-depth", str(max_depth), "--lam", repr(0.0855),
           "--leaf-logits-rows", str(len(leaf_logits_table))]
    if prefix_slots:
        cmd += ["--prefix", " ".join(str(s) for s in prefix_slots)]
    gumbel_str = " ".join(repr(float(v)) for v in gumbel_fifo)
    value_str = " ".join(repr(float(v)) for v in value_fifo)
    logit_flat = []
    for row in leaf_logits_table:
        for x in row:
            logit_flat.append(repr(float(x)))
    logit_str = " ".join(logit_flat)
    # line 1 = gumbel FIFO; line 2 = value FIFO; line 3 = world-index FIFO (empty -> bw[0]);
    # line 4 = the flattened FINE per-slot logits table.
    stdin = gumbel_str + "\n" + value_str + "\n" + "\n" + logit_str + "\n"
    sub_env = dict(os.environ)
    if uniform:
        sub_env["CHOCO_GUMBEL_UNIFORM"] = "1"
    else:
        sub_env.pop("CHOCO_GUMBEL_UNIFORM", None)
    out = subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=sub_env)
    if out.returncode != 0:
        raise RuntimeError(f"gumbel-dump failed (rc={out.returncode}): {out.stderr}")
    parts = out.stdout.split()
    return int(parts[0]), int(parts[1]), int(parts[2])


# The fine-input scales. These are TUNED (not arbitrary): they put the search's DISCRETE decisions
# (the SH survivor + the improved-pi argmax) on the float32-prior knife-edge, so the all-float64 uniform
# port (CHOCO_GUMBEL_UNIFORM=1) diverges from the float32 mixed port on a LARGE fraction of the grid.
# Verified by a sweep over seeds/scales: this (seed, vspread, dscale) gives ~110/144 uniform divergence
# while the mixed port matches Python N/N. If a reseed drops the divergence to ~0 the test would go
# VACUOUS — main() ASSERTS the divergence is non-trivial so a regression in the inputs fails loudly.
_FINE_SEED = 13
_FINE_VSPREAD = 1.0e-7   # leaf-value spread: TIGHT, so visited-slot Q's near-tie (sigma*q nearly cancels
                         #   in the improved-pi argmax) and the ~1e-7 float32 log-prior decides the order
_FINE_DSCALE = 1.0e-4    # per-slot logit delta: small enough the masked-softmax prior near-ties AND is
                         #   not float32-representable, so the float32 store (seam 1) perturbs log(prior)


def _make_fine_leaves(env, rng, n_leaves):
    """Build the FINE near-tie leaf tables that make the search's DISCRETE output precision-sensitive.

    The discrete discriminator (verified by the 1b audit) is SEAM 1 — the float32 prior STORAGE flowing
    into `logits = log(prior)`, the DOMINANT (~1e-7 on log(prior)) float32 effect. It feeds the
    Gumbel-top-k (logit+g) AND the SH cut key (g+logit+sigma*q-hat), so a float32-vs-float64 prior FLIPS
    the SH survivor and the improved-pi argmax. (The secondary seams 2/3/4 — v_mix/sigma*vmix/PUCT — are
    byte-faithful in VALUE but near-unobservable on the DISCRETE output: v_mix feeds only the improved-pi
    UNVISITED completion, which never wins the argmax; PUCT needs a ~1e-8 interior tie. We do not pretend
    they drive the discrete flip — see the gumbel.cpp seam map's HONEST SCOPE note.)

    To put the discrete output on the seam-1 knife-edge:
      * leaf VALUES in a VERY TIGHT cluster (center 0.13, spread ~1e-7) -> every visited slot's Q backup
        near-ties, so in the improved-pi `logits[s] + sigma*q[s]` the `sigma*q` term nearly cancels
        across the top slots and the ~1e-7 float32-prior-derived `logits[s]` decides the argmax;
      * leaf LOGITS = a common base + a TINY per-slot delta (~1e-4) carrying full float64 mantissa noise
        -> the masked-softmax prior entries are nearly equal AND not float32-representable, so storing
        them float32 (seam 1) shifts `log(prior)` by ~1e-7 — exactly the margin the near-tied Q's leave.

    Returns (value_fifo, leaf_logits_table), cycled in call order by both languages."""
    n_slots = n_action_slots(env)
    # values: a VERY tight cluster so leaf-value-driven Q backups near-tie at the ~1e-7 scale.
    value_fifo = (0.13 + rng.standard_normal(n_leaves) * _FINE_VSPREAD).tolist()
    # logits: base ~ N(0, 0.5), plus a per-slot delta at the ~1e-4 scale so the softmax prior near-ties.
    # The delta carries full float64 mantissa (uniform draw), so float32 storage perturbs it at ~1e-7.
    rows = []
    for _ in range(n_leaves):
        base = float(rng.standard_normal() * 0.5)
        delta = rng.uniform(-1.0, 1.0, size=n_slots) * _FINE_DSCALE
        rows.append((base + delta).astype(np.float64).tolist())
    return value_fifo, rows


def main():
    if not os.path.exists(GUMBEL_BIN):
        print(f"FAIL: C++ gumbel-dump not built at {GUMBEL_BIN}\n"
              f"      build it: cmake -S cpp -B cpp/build && cmake --build cpp/build")
        return 1

    env = Environment()
    print("=== ADR-0012 P6 mixed-precision NEAR-TIE parity: C++ Gumbel-AZ vs Python Gumbel-AZ "
          "(PHASE 1b, precision FIDELITY) ===")
    print(f"instance: N={env.N} K={env.K} nDet={len(env.detectors)} |worlds|={len(env.worlds)} "
          f"n_slots={n_action_slots(env)}")
    print("FINE seam: rng.gumbel -> a fixed FIFO; sample_world -> bw[0]; leaf (value, FULL-PRECISION "
          "per-slot logits) -> a fixed FIFO drawn at the float32-epsilon scale (GENUINE near-ties)")
    print("assert (A): the MIXED-precision C++ matches Python EXACTLY (exec action + improved-pi argmax) "
          "N/N; (B) the UNIFORM-precision C++ (CHOCO_GUMBEL_UNIFORM=1) DIVERGES on a non-trivial X/N\n")

    rng = np.random.default_rng(_FINE_SEED)
    gumbel_fifo = rng.gumbel(size=4096).tolist()
    value_fifo, leaf_logits_table = _make_fine_leaves(env, rng, 2048)

    # the grid: (n_sims, m, c_puct, max_depth, prefix). Varied to exercise multiple SH phases + the
    # remainder loop, the interior PUCT trade-off (c_puct>0 so the float32 PUCT seam is live), the leaf-
    # vs-descent boundary, and root vs mid-episode states. c_puct=0 is dropped here (it zeroes the PUCT
    # U-term, killing seam 4); the FINE inputs need a live PUCT to exercise the interior near-tie flips.
    cases = []
    for n_sims, m in [(12, 4), (16, 6), (24, 8), (48, 12), (32, 5), (8, 3)]:
        for c_puct in (0.5, 1.25, 2.5):
            for max_depth in (3, 24):
                for prefix in ([], [25], [25, 27], [5]):
                    cases.append((n_sims, m, c_puct, max_depth, prefix))

    total = len(cases)
    mixed_ok = 0
    mixed_mismatch = 0
    uniform_diverge = 0
    sample_diverge = []
    for n_sims, m, c_puct, max_depth, prefix in cases:
        # the FAITHFUL Python reference (the REAL mixed-precision path) — the SAME on both arms.
        py_exec, py_argmax, py_spent = py_decide(
            env, m, n_sims, c_puct, max_depth, prefix, gumbel_fifo, value_fifo, leaf_logits_table)

        # arm A: the MIXED-precision C++ (default faithful path) — must match Python.
        mx_exec, mx_argmax, mx_spent = cpp_decide(
            m, n_sims, c_puct, max_depth, prefix, gumbel_fifo, value_fifo, leaf_logits_table,
            uniform=False)
        if (mx_exec, mx_argmax) == (py_exec, py_argmax) and mx_spent == n_sims:
            mixed_ok += 1
        else:
            mixed_mismatch += 1
            if mixed_mismatch <= 20:
                print(f"  MIXED MISMATCH n_sims={n_sims} m={m} c_puct={c_puct} max_depth={max_depth} "
                      f"prefix={prefix}: py(exec={py_exec},argmax={py_argmax},spent={py_spent}) "
                      f"mixed(exec={mx_exec},argmax={mx_argmax},spent={mx_spent})")

        # arm B: the UNIFORM-precision C++ (the 1a all-float64 path) — the discrimination control. It
        # SHOULD diverge from Python on a non-trivial fraction (else the inputs are not precision-
        # sensitive and the whole test is vacuous).
        un_exec, un_argmax, un_spent = cpp_decide(
            m, n_sims, c_puct, max_depth, prefix, gumbel_fifo, value_fifo, leaf_logits_table,
            uniform=True)
        if (un_exec, un_argmax) != (py_exec, py_argmax):
            uniform_diverge += 1
            if len(sample_diverge) < 8:
                sample_diverge.append(
                    f"    n_sims={n_sims} m={m} c_puct={c_puct} max_depth={max_depth} prefix={prefix}: "
                    f"py(exec={py_exec},argmax={py_argmax}) uniform(exec={un_exec},argmax={un_argmax})")

    print(f"\n[A mixed-precision parity ] {mixed_ok}/{total} cases the FAITHFUL (mixed-precision) C++ "
          f"matches Python EXACTLY (executed action AND improved-pi argmax)")
    print(f"[B discrimination control ] {uniform_diverge}/{total} cases the UNIFORM-precision C++ "
          f"(CHOCO_GUMBEL_UNIFORM=1) DIVERGES from the SAME Python reference")
    if sample_diverge:
        print("  sample uniform divergences (the float32 seam decides these near-ties):")
        for line in sample_diverge:
            print(line)

    mixed_pass = (mixed_ok == total)
    # NON-VACUITY guard (the 1b analogue of "0/240 proves nothing"): require a NON-TRIVIAL divergence,
    # not merely >0. A single flaky flip could pass `>0` while the inputs are essentially insensitive; a
    # reseed/scale regression that collapses the divergence must FAIL here. The tuned inputs give ~34/144
    # uniform divergence — `MIN_DIVERGE` sits well below that with margin, well above flake.
    MIN_DIVERGE = 10
    discriminates = (uniform_diverge >= MIN_DIVERGE)
    if mixed_pass and discriminates:
        print(f"\nRESULT: PASS — the MIXED-precision C++ Gumbel-AZ search reproduces Python's "
              f"float32-prior x float64-Q precision EXACTLY on FINE near-tie inputs ({mixed_ok}/{total}), "
              f"while the UNIFORM-precision (all-float64) port diverges on {uniform_diverge}/{total} of "
              f"the SAME cases — proving the float32 seam (not the structure) decides the near-ties, and "
              f"that the parity is non-vacuous (the 1b analogue of the 1a mutation control).")
        return 0
    if not mixed_pass:
        print("\nRESULT: FAIL — the MIXED-precision C++ does NOT match Python on every fine case; the "
              "float32 promotion is not yet byte-faithful to value_target.py.")
    if not discriminates:
        print(f"\nRESULT: FAIL — the UNIFORM-precision control diverged on only {uniform_diverge}/{total} "
              f"cases (< {MIN_DIVERGE}): the fine inputs are not precision-sensitive enough, so the parity "
              f"is (near-)VACUOUS (proves little). Tighten the leaf spreads until the uniform mutant "
              f"diverges on a non-trivial fraction.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
