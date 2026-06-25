#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/watchdog.py — the ONE home for the issue-gate control lab's METHOD WATCHDOG
contract: the per-decision gate VALIDATION + the structured MALFUNCTION record, shared by every control
wire the lab drives.

WHY ONE HOME (ADR-0012 P1 single-source-of-truth). The lab has TWO control transports onto the same
IssueController actuation hub: the per-forward LabServer path (lab_server.py, lab_control_wire.hpp) and
the async issue-control-bridge path (run_control_lab.py, issue_control_bridge.hpp). BOTH must enforce the
IDENTICAL safety contract — a slow / throwing / malformed method FALLS BACK to all-allow and is FLAGGED,
never wedging the producer (ADR-0002 fail-loud-but-don't-tear-down). That contract is a cross-cutting fact
with one authoritative definition; each wire DERIVES its view, neither re-authors it. Living here (a
dependency-light module — no StageAServer / JAX import, unlike lab_server) lets the async harness reuse it
WITHOUT dragging the inference stack into its policy peer.

WHAT IS HERE (and only this — P3 one-owner): the gate-shape validator and the malfunction tally. The
per-wire DECISION DRIVER (build the Observation for that wire, hold the gate vector, append the trajectory)
stays in each wire's own server/harness — those differ by transport (per-forward served-subset vs async
all-T snapshot) and are correctly NOT shared.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MalfunctionRecord:
    """Loud, structured record of a method's misbehaviour on the decision path (ADR-0002). The harness/
    server reads `flags` + `total()` into the trial record and marks the method; the fixture (the producer)
    is never torn down — a bad gate just falls back to all-allow for that decision."""
    slow: int = 0           # act() exceeded the per-decision deadline (fell back to all-allow)
    raised: int = 0         # act()/observe() threw (fell back to all-allow)
    malformed: int = 0      # act() returned a non-length-T or non-binary vector (rejected -> all-allow)
    last_error: str = ""    # the most recent diagnostic string (for the harness log)
    flags: list[str] = field(default_factory=list)   # ordered, de-duplicated human-readable flags

    def note(self, flag: str, err: str = "") -> None:
        if err:
            self.last_error = err
        if flag not in self.flags:
            self.flags.append(flag)

    def total(self) -> int:
        return self.slow + self.raised + self.malformed


def validate_gates(gates: Any, T: int) -> "list[int] | None":
    """Validate a Controller's act() return: a length-T sequence of {0,1}. Returns the coerced list, or
    None on any shape/value violation (the caller then falls back to all-allow + flags). The ONE definition
    of the gate-shape contract both control wires enforce (P1) — tighten it HERE and both wires inherit it."""
    try:
        g = list(gates)
    except TypeError:
        return None
    if len(g) != T:
        return None
    out: list[int] = []
    for v in g:
        if v == 0:
            out.append(0)
        elif v == 1:
            out.append(1)
        else:
            return None
    return out
