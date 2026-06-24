#!/usr/bin/env python3
"""
throughput-lab/harness/code_stamp.py — the ONE home (ADR-0012 P1) for the ADR-0011 measurement-provenance
stamp: every attributed benchmark reading carries the git commit short-hash + tree state (clean|DIRTY) of
the checkout that produced it, emitted by the measuring harness itself.

WHY (ADR-0011 Rule 2/4, the recurrence→mechanism net over the CLASS of all readings): a reading not
pinnable to a code state is unattributable by construction — a "+31%" throughput win banked from one
session could not be reproduced or pinned to a commit across sessions (2026-06-24). A `DIRTY` tree marks a
NON-reproducible artifact: the producer binary / harness may not match HEAD, so the number is provisional
until committed. Pairs with ADR-0009 (a captured bench number is now code-addressable).

The Python sweeps (coalesce_sweep.py, topology_sweep.py, cpp/stage_a/overcommit_sweep.py) import this; the
shell harness (episodic_dps.sh) MIRRORS the same two git invocations inline (`git rev-parse --short HEAD`
and `git status --porcelain` → clean|DIRTY) — keep the two in step.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def code_stamp(cwd: "str | Path | None" = None) -> "dict[str, str]":
    """Return {'commit': <short hash | 'unknown'>, 'tree': 'clean' | 'DIRTY'} for the git checkout that
    contains the measuring harness (default: this file's directory, so it reads the right repo/worktree).
    Never raises — a missing git / non-repo degrades to {'unknown', 'DIRTY'} (an un-pinnable reading is, by
    the discipline, treated as non-reproducible rather than silently 'clean')."""
    where = str(cwd) if cwd is not None else str(Path(__file__).resolve().parent)

    def _git(*args: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(["git", "-C", where, *args], capture_output=True, text=True)

    try:
        h = _git("rev-parse", "--short", "HEAD")
        commit = h.stdout.strip() if (h.returncode == 0 and h.stdout.strip()) else "unknown"
        s = _git("status", "--porcelain")
        tree = "clean" if (s.returncode == 0 and not s.stdout.strip()) else "DIRTY"
    except OSError:
        commit, tree = "unknown", "DIRTY"
    return {"commit": commit, "tree": tree}


def code_stamp_str(cwd: "str | Path | None" = None) -> str:
    """The stamp as a single token pair for a header/print line: `commit=<short> tree=clean|DIRTY`."""
    st = code_stamp(cwd)
    return f"commit={st['commit']} tree={st['tree']}"
