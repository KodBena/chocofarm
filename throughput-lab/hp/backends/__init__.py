"""
throughput-lab/hp/backends/__init__.py — backend lowerings for the HP config-space compiler.

Two lowerings consume a backend-neutral ir.ConfigSpace: `cpsat` (the CP-SAT enumerator + the
canonicalizer) and `grid` (the itertools oracle, an independent re-derivation). The two-lowering
design IS the verification architecture (DESIGN.md §4).

Public Domain (The Unlicense).
"""
