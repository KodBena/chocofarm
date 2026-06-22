"""
tools/analysis/leaf_eval_bound/contract/grounded_types.py

The grounded-quantity VOCABULARY (§2.2 split): the `Estimability` enum (measured/constant/prior, the
single-home measured-vs-pinned axis) + the frozen `Grounded` dataclass. Pure types -- no data, no
references; `grounding` imports these to build its constant table.

Public Domain (The Unlicense).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Estimability(enum.Enum):
    """The single-home (ADR-0012 P1) measured-vs-pinned axis of a grounded quantity — RCA fix #1
    (docs/notes/leaf-eval-estimator-pin-cascade-rca.md). It is GENERATIVE: both the model flags
    (`Grounded.constant`/`needs_measurement` DERIVE from it) and the bench's pin-vs-shrinkable body answer
    to this ONE axis, so the measured-but-punted P8 lie — a quantity labelled measurable whose bench pins —
    cannot be authored (there is no second flag to disagree). The estimability-agreement guard in
    `tests/test_untrusted_drive_phase4.py` enforces body <=> this axis."""
    CONSTANT = "constant"   # a TRUE deployment/layout constant (n_gen=3 cores) -> DEGENERATE Fixed pin, a_i~0
    MEASURED = "measured"   # a RUNNABLE bench measures it live -> a SHRINKABLE Estimate (median / regression fit)
    PRIOR = "prior"         # an engineering-judgement prior, NO runnable bench yet (B_op) -> NORMAL Fixed pin


@dataclass(frozen=True)
class Grounded:
    """One grounded quantity: a mean, a 1-sigma spread, a relative per-sample benchmark cost, its
    `estimability` (the measured-vs-pinned axis), and the `module` of the bench that owns its live
    measurement. `provenance` is the file the number was read from.

    `estimability` (`Estimability`, ADR-0012 P1 single-home; RCA fix #1) is the SSOT of the
    DEGENERATE-vs-declared-spread-vs-measured classification (the harmonized-estimator-interface §3 PIN
    distinction). The two former flags now DERIVE from it (properties below), so they cannot disagree with
    the bench body: `constant` (a TRUE CONSTANT — a deployment/layout fact like `n_gen`=3 cores, set by the
    1:3 pinning, NOT CI-bearing) iff `CONSTANT`; `needs_measurement` (still needs a fresh SOLE-WORKLOAD run,
    the Neyman loop ranks these) iff NOT `CONSTANT`. The bench's `pin_estimate(constant=…)` and the
    manifest's seed Estimate (`family=DEGENERATE` vs `NORMAL`) read `constant`, so a true constant cannot
    leak its frozen-display σ into the bound on one path while dropping out on another. A `CONSTANT`
    quantity's `sigma` is a display/seed artifact (a placeholder on an integer/fixed value), not a real
    spread — the bound treats it as ~0 (the §3 'a_i ≈ 0' rule)."""
    name: str
    mean: float
    sigma: float
    cost: float
    unit: str
    provenance: str
    estimability: Estimability
    module: str

    @property
    def constant(self) -> bool:
        """A TRUE CONSTANT (DEGENERATE pin, ~0 bound contribution) iff `estimability is CONSTANT` — DERIVED
        from the single axis (P1), never an independent flag that could disagree with the bench body."""
        return self.estimability is Estimability.CONSTANT

    @property
    def needs_measurement(self) -> bool:
        """Needs a fresh SOLE-WORKLOAD measurement iff NOT a true constant — DERIVED from the single axis
        (both MEASURED and PRIOR would benefit; only MEASURED has a runnable bench today). The static models
        (model_capacity/model_cycletime) read this; it now single-homes off `estimability`."""
        return self.estimability is not Estimability.CONSTANT
