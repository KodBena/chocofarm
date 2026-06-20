#!/usr/bin/env python3
"""
cpp/parity/gumbel_no_early_exit.py — the focused behavior check for GumbelConfig::no_early_exit (the
HPO/BENCHMARK-ONLY no-early-exit substitution in the C++ Gumbel-AZ decide core, gumbel.cpp run_search).

This is a C++-only behavior check (NOT a cross-language parity): it drives chocofarm-gumbel-dump with
the SAME scripted, RNG-free seam the 1a logic check uses (gumbel FIFO; sample_world->bw[0]; the leaf
(value, coarse logits)->a cycled table), toggling ONLY the --no-early-exit flag, and asserts the three
properties the flag's contract names (gumbel.hpp GumbelConfig::no_early_exit, gumbel.cpp substitution
block):

  (a) FLAG OFF == BASELINE (default-false). The dump's stdout (executed slot, improved-pi argmax, sims
      spent) with the flag ABSENT is byte-identical to the dump run with the flag absent again — and the
      executed slot IS Terminate on the discriminating input below (so OFF genuinely early-exits). The
      production search is byte-unchanged by default. (The cross-language 1a/1b parity scripts, which
      run the dump with the flag absent, independently confirm the default path is byte-faithful.)

  (b) FLAG ON, a non-terminate legal action exists => the EXECUTED action is NEVER Terminate, the
      improved-pi argmax (the real PI target) is UNCHANGED from OFF, and the sims spent are UNCHANGED.
      The substitution touches ONLY the executed action, leaving the search dynamics + the improved-pi
      target exactly as the unconstrained search produced them (gumbel.cpp leaves out.improved + the
      backprop untouched). The discriminating input is a leaf that makes the unconstrained search pick
      Terminate (a very-negative non-terminate leaf return vs Terminate's mild -lam*exit_cost), so (b)
      is non-vacuous: OFF executes Terminate, ON substitutes a non-terminate slot.

  (c) FLAG ON, NO non-terminate legal action exists => the executed action is STILL Terminate (the
      episode correctly ends; there is nothing to substitute). Realized by a prefix that collects ALL
      treasures so the state has a non-empty belief but ZERO legal non-terminate actions — exactly the
      `best_slot == -1` branch the substitution guards (the same end-state the empty-belief guard yields).

Run (from repo root, after `cmake --build cpp/build --target chocofarm-gumbel-dump`):
    PYTHONPATH=. /home/bork/w/vdc/venvs/generic/bin/python cpp/parity/gumbel_no_early_exit.py

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from chocofarm.model.env import Environment
from chocofarm.az.actions import n_action_slots, term_slot

GUMBEL_BIN = os.path.join(REPO, "cpp", "build", "chocofarm-gumbel-dump")
INSTANCE = os.path.join(REPO, "chocofarm", "data", "instance.json")
FACES = os.path.join(REPO, "chocofarm", "data", "faces.json")

# The discriminating scripted leaf: a VERY negative non-terminate leaf return (value=-50) makes every
# real-action backup low, while Terminate's Q = -lam*exit_cost is mild -> the unconstrained search picks
# Terminate. gumbel all-zero (deterministic on the coarse ramp). m/n_sims default-ish.
LAM = 0.0855
GUMBEL_LINE = "0.0 0.0 0.0 0.0"
LEAF_LINE = "-50.0 0.0"   # one (value, logit_base) pair: value=-50, logit_base=0 (ramp lb+s*0.25)


def run_dump(no_early_exit: bool, prefix: str | None = None) -> tuple[int, int, int]:
    """Drive chocofarm-gumbel-dump with the scripted FIFOs; return (exec_slot, argmax_slot, n_spent)."""
    cmd = [GUMBEL_BIN, "--instance", INSTANCE, "--faces", FACES,
           "--m", "12", "--n-sims", "48", "--c-puct", "1.25", "--max-depth", "24", "--lam", repr(LAM)]
    if prefix:
        cmd += ["--prefix", prefix]
    if no_early_exit:
        cmd += ["--no-early-exit"]
    stdin = GUMBEL_LINE + "\n" + LEAF_LINE + "\n"
    out = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"gumbel-dump failed (rc={out.returncode}): {out.stderr}")
    parts = out.stdout.split()
    return int(parts[0]), int(parts[1]), int(parts[2])


def main() -> int:
    if not os.path.exists(GUMBEL_BIN):
        print(f"FAIL: C++ gumbel-dump not built at {GUMBEL_BIN}\n"
              f"      build it: cmake --build cpp/build --target chocofarm-gumbel-dump")
        return 1

    env = Environment()
    ts = term_slot(env)
    print("=== GumbelConfig::no_early_exit behavior check (HPO/BENCHMARK-ONLY substitution) ===")
    print(f"instance: N={env.N} nDet={len(env.detectors)} n_slots={n_action_slots(env)} term_slot={ts}")
    print("scripted seam: gumbel->0 FIFO; sample_world->bw[0]; leaf value=-50 (exit favored) ramp "
          "logits[s]=s*0.25\n")

    ok = True

    # (a) FLAG OFF == BASELINE, and OFF genuinely early-exits (executes Terminate) on this input.
    off1 = run_dump(no_early_exit=False)
    off2 = run_dump(no_early_exit=False)
    print(f"(a) OFF run1={off1} run2={off2}")
    if off1 != off2:
        print("    FAIL: flag-off is not deterministic / byte-identical across runs")
        ok = False
    if off1[0] != ts:
        print(f"    FAIL: flag-off did NOT execute Terminate (exec={off1[0]} != term_slot={ts}); the "
              f"input is not discriminating, so (b) would be vacuous")
        ok = False
    else:
        print(f"    OK: flag-off executes Terminate (slot {ts}) — early-exit, baseline byte-stable")

    # (b) FLAG ON with a non-terminate legal action present: never Terminate; improved-pi + spend same.
    on = run_dump(no_early_exit=True)
    print(f"(b) ON ={on}  (OFF={off1})")
    if on[0] == ts:
        print(f"    FAIL: flag-on STILL executed Terminate (slot {ts}) though non-terminate slots exist")
        ok = False
    else:
        print(f"    OK: flag-on substituted a NON-terminate executed slot ({on[0]}) for Terminate")
    if on[1] != off1[1]:
        print(f"    FAIL: improved-pi argmax changed OFF={off1[1]} -> ON={on[1]} (the PI target must be "
              f"UNTOUCHED — the substitution touches ONLY the executed action)")
        ok = False
    else:
        print(f"    OK: improved-pi argmax UNCHANGED ({on[1]}) — the PI target is left intact")
    if on[2] != off1[2]:
        print(f"    FAIL: sims spent changed OFF={off1[2]} -> ON={on[2]} (the search dynamics must be "
              f"identical; only the executed action is substituted)")
        ok = False
    else:
        print(f"    OK: sims spent UNCHANGED ({on[2]}) — the search ran identically")

    # (c) FLAG ON but NO non-terminate legal action (all treasures collected): still Terminate.
    prefix_all_treasures = " ".join(str(s) for s in range(env.N))
    on_c = run_dump(no_early_exit=True, prefix=prefix_all_treasures)
    print(f"(c) ON, all-treasures-collected prefix => {on_c}")
    if on_c[0] != ts:
        print(f"    FAIL: with NO non-terminate legal action, flag-on must STILL execute Terminate "
              f"(got exec={on_c[0]} != term_slot={ts})")
        ok = False
    else:
        print(f"    OK: no non-terminate legal action => flag-on correctly STILL terminates (slot {ts})")

    if ok:
        print("\nRESULT: PASS — GumbelConfig::no_early_exit substitutes a non-terminate executed action "
              "for an early-exit Terminate when one exists (PI target + search dynamics untouched), is "
              "byte-stable when OFF, and correctly leaves Terminate when no non-terminate action remains.")
        return 0
    print("\nRESULT: FAIL — the no-early-exit substitution behavior deviated from its contract")
    return 1


if __name__ == "__main__":
    sys.exit(main())
