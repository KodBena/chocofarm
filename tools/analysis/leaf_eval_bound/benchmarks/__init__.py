"""
tools/analysis/leaf_eval_bound/benchmarks/__init__.py
===============================================

The LIVE-benchmark package for the leaf-eval transport-design sweep: one `bench_<name>.py`
per measurable quantity (the model inputs), each exposing `get_seed()` (the v1 fallback),
`run()` (the live measurement that logs samples to postgres), and `register_self()` (the
benchmark_definition INSERT). The manifest resolves a quantity to its bench via the
`module_path` recorded in its definition row (the dotted `benchmarks.bench_<name>` form),
so registering a new quantity needs no manifest edit.

Public Domain (The Unlicense).
"""
