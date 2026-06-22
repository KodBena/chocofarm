"""
tools/analysis/leaf_eval_bound/__init__.py

the leaf-eval throughput lower-bound tool as a REAL package (the responsibility-refactor §3 layout). The per-module `sys.path.insert` preamble is retired: modules import each other via `leaf_eval_bound.<subpkg>.<module>`, and the package parent (tools/analysis) goes on sys.path (the tests insert it; standalone runs use `python -m leaf_eval_bound.runners.<runner>`).

Public Domain (The Unlicense).
"""
