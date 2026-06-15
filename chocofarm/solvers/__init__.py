#!/usr/bin/env python3
"""
chocofarm.solvers — the pluggable Policy implementations + the SOLVERS name registry.

`SOLVERS` maps a short solver NAME to its `Policy` class, so callers that pick a solver by
string (eval_*.py drivers, tb_runner) construct it from ONE table instead of a hand-kept
if/elif. Adding a solver is a single registry line, not an edit in every caller.

The registry holds CLASSES, not instances: each class has its own `__init__` signature —
`UCTPolicy(iterations=…)`, `NMCSPolicy(level=…)`, etc. — so a caller constructs
`SOLVERS[name](**kwargs)` with the kwargs that solver expects, OR
`SOLVERS[name](cfg=<family-config>)` (audit item I). Each classical search solver now has a
frozen per-family `SearchConfig` dataclass grouping its scalar hyperparameters — `UCTConfig`,
`ISMCTSConfig`, `NMCSConfig` (in their solver modules), `RolloutConfig`, `SparseSamplingConfig`
(in `solvers.base`). The config is accepted at construction; the individual kwargs remain the
back-compat path (`UCTPolicy(iterations=200)` still works and builds the config). A uniform
per-`decide()` cfg override (the Tier-3 live-cell shape) is deferred — see eval/report.py — to
keep this slice behaviour-preserving (the per-call override would touch every `decide` body).

Public Domain (The Unlicense).
"""
from chocofarm.solvers.uct import UCTPolicy, UCTConfig
from chocofarm.solvers.ismcts import ISMCTSPolicy, ISMCTSConfig
from chocofarm.solvers.nmcs import NMCSPolicy, NMCSConfig
from chocofarm.solvers.decomp import DecompPolicy
from chocofarm.solvers.base import RandomPolicy, RolloutConfig, SparseSamplingConfig

# name -> Policy class. The SINGLE source for "which class is this solver".
SOLVERS = {
    "random": RandomPolicy,
    "uct": UCTPolicy,
    "ismcts": ISMCTSPolicy,
    "nmcs": NMCSPolicy,
    "decomp": DecompPolicy,
}

# name -> frozen SearchConfig dataclass for the family (audit item I). Lets a caller build a
# solver from a config object (`SOLVERS[n](cfg=SOLVER_CONFIGS[n](...))`) where one exists. decomp
# carries a single `horizon` knob and is intentionally left config-free (minimal-touch, ADR-0004).
SOLVER_CONFIGS = {
    "uct": UCTConfig,
    "ismcts": ISMCTSConfig,
    "nmcs": NMCSConfig,
}
