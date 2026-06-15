# Experiment configuration — Dhall vs Python-native, and the target shape (2026-06-15)

A forward-looking design record (decision-time, per the documentation discipline). The
question, raised by the maintainer: does it make sense to use **Dhall** (the typed,
functional configuration language) to author the AZ experiments we are about to run —
the calibration agenda, het-values, LR/anneal sweeps — or would that be too ad-hoc / a
poor fit? Verdict and the recommended shape below.

> **Dhall is over-engineering for chocofarm at its current scale. The pain it targets
> is real — experiments are 26-flag bash command lines, re-specified per variant, untyped
> at the authoring surface — but the typed-SSOT half is already solved by the `hp`
> registry, and the variant-DRY half is better filled in Python than by importing a new
> typed config language. Finish the Python-native config story (a `RunConfig`/
> `ExperimentConfig` authoring artifact + a typed `sweep()` helper that seeds the
> registry). Revisit Dhall/CUE/Jsonnet only if config becomes polyglot, multi-consumer,
> or team-authored.**

## What Dhall would buy

Dhall is a typed, total (non-Turing-complete), side-effect-free config language with
functions, `let`-bindings, and integrity-hashed imports, compiling to JSON. The genuine
wins for an experiment program: (a) type-checked configs caught *before* a run;
(b) DRY variant generation — `base // { lr = 1e-4 }`, or `map (\lr -> base // {lr})
[1e-3, 5e-4, 1e-4]` for a sweep; (c) configs as content-addressed, versioned,
reproducible artifacts. That maps onto the ad-hoc-ness the architecture audit flagged
(no central config; sweeps as bash flags; the handoff's experiments as multi-line
command lines).

## Why it is the wrong tool here

1. **The typed-SSOT half already exists in Python.** The `hp` registry is a typed
   dataclass SSOT (`hp/schema.py`) seeded from argparse and stamped as a versioned redis
   blob (writer + timestamp + schema_version). Dhall would not *replace* the registry —
   it would sit in front of it, generating JSON that seeds it. The only gap it fills is
   the *authoring / variant-generation surface*.
2. **That gap is Python-solvable, no new language.** The DRY-variant win is
   `dataclasses.replace(base, lr=lr)` over a sweep comprehension — i.e. `RunConfig`
   (audit R12) plus a small typed `sweep()` helper that constructs `ExperimentConfig`s
   and seeds the registry. That makes experiments first-class typed Python artifacts in
   git — ~90% of Dhall's value at zero new-language cost.
3. **The costs are real for this project.** `dhall-python` is a niche, Rust-backed
   (pyo3) third-party binding — actively maintained but small and not a mainstream
   ML-config path — plus a language for the maintainer/collaborators to learn and a
   Dhall→JSON build step, on a single-maintainer numpy/JAX codebase.
4. **Dhall's headline strengths do not apply.** Totality, language-agnostic output,
   untrusted-config safety, cryptographic import hashes — chocofarm's config is
   single-consumer (the parent AZ loop), single-maintainer, trusted, Python-only (the
   future C++ worker reads *weights*, not the experiment config — see
   `scaling-and-cpp-seam.md`). Reproducibility is already carried by the seed-fold
   determinism + the versioned registry blob + git, not by the config language.

## When to revisit

If the program grows into any of: non-Python collaborators authoring experiments; config
consumed by multiple tools/languages; a team wanting config-as-shared-content-addressed
artifacts — then a typed config layer (Dhall, or CUE/Jsonnet) earns its keep, because its
`// { override }` ergonomics genuinely beat ad-hoc YAML at that scale. None hold today.

## The target shape (Python-native)

The authoring artifact is a typed `ExperimentConfig`/`RunConfig` (the registry's
`hp/schema.py` is already this), constructed in a small in-tree experiment-definition
module or a `sweep()` helper rather than a bash command:

```python
base = ExperimentConfig(lr=1e-3, m=24, n_sims=128, lam=0.0855, residual=True, ...)
lr_anneal = [replace(base, lr=lr) for lr in (1e-3, 5e-4, 1e-4)]          # the sweep, typed, DRY
for cfg in lr_anneal:
    seed_registry(experiment_id(cfg), cfg)                               # seeds the same registry
```

This makes a sweep a typed, git-versioned comprehension; the registry stays the live-read
SSOT and the reproducible record; no second config language enters the stack. Depends on
audit R12 (`RunConfig`) + a thin `sweep()`/`experiment_id()` helper.

*Public Domain (The Unlicense).*
