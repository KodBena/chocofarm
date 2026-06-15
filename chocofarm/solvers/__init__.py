#!/usr/bin/env python3
"""
chocofarm.solvers — the pluggable Policy implementations + the SOLVERS name registry.

`SOLVERS` maps a short solver NAME to its `Policy` class, so callers that pick a solver by
string (eval_*.py drivers, tb_runner) construct it from ONE table instead of a hand-kept
if/elif. Adding a solver is a single registry line, not an edit in every caller.

The registry holds CLASSES, not instances: each class still has its own (currently frozen)
`__init__` signature — `UCTPolicy(iterations=…)`, `NMCSPolicy(level=…)`, etc. — so a caller
constructs `SOLVERS[name](**kwargs)` with the kwargs that solver expects. (A uniform per-call
`SearchConfig` dataclass that would let every solver be built the same way is the deferred
follow-up noted in eval/report.py; it is out of this step's scope.)

Public Domain (The Unlicense).
"""
from chocofarm.solvers.uct import UCTPolicy
from chocofarm.solvers.ismcts import ISMCTSPolicy
from chocofarm.solvers.nmcs import NMCSPolicy
from chocofarm.solvers.decomp import DecompPolicy

# name -> Policy class. The SINGLE source for "which class is this solver".
SOLVERS = {
    "uct": UCTPolicy,
    "ismcts": ISMCTSPolicy,
    "nmcs": NMCSPolicy,
    "decomp": DecompPolicy,
}
