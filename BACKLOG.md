# BACKLOG

Consciously-deferred work, recorded so it isn't lost. NOT a live task queue (that's the commit
log + branch state) — these are items postponed on purpose, with enough context to pick up cold.

## ISMCTS port — test hardening (deferred 2026-06-16)

The C++ ISMCTS port is merged and independently reviewed `trustworthy-mergeable`
(`docs/notes/ismcts-port-review-2026-06-16.md`). Both verification-coverage holes the review found
are already closed with discriminating *executed* tests — the `_ucb_select` tie-break (the review's
integer-leaf run: 128/128, 14/128 mutant control) and the multi-belief sub-child split
(`cpp/parity/ismcts_multiworld.py`, committed: 192/192 parity, 40/192 `belief_key`-collapse mutant
control vs 0/240 on the old `bw[0]` check). Remaining is test-only; production `ismcts.cpp` is untouched:

- **(a) Permanent integer-leaf tie-forcing fixture.** The `_ucb_select` insertion-order tie-break was
  verified by the review's *ad-hoc* run, not a committed test. Make it a permanent fixture under
  `cpp/parity/` (mirroring `ismcts_multiworld.py`: integer leaf FIFO + a `>`→`>=` / sorted-key mutant
  control) so it's reproducible and regression-gated.
- **(b) Soften the over-claiming docstrings.** `cpp/parity/ismcts_logic.py:22-23/:194-196` and the
  ISMCTS asserts in `tests/test_cpp_runner.py` claim "UCB select ... covered"; the float-leaf grid
  exercises the UCB arithmetic + availability denominator but NOT the insertion-order tie-break (that's
  (a) + the multiworld fixture). Narrow the claims to what each test actually proves.
- **(c) Wire the new opt-in fixtures into `tests/test_cpp_runner.py`** — `ismcts_multiworld.py` and the
  (a) tie-forcing fixture, behind `CHOCO_RUN_CPP`, mirroring the existing opt-in pattern.
- **(d) Systemic aggregate-methodology fix (shared `cpp/parity/parity.py`).** The aggregate parity
  discards raw per-episode data (`tempfile.mktemp` + `os.unlink`) and uses an uncorrected 6-statistic
  3σ gate (~1.6% family-wise false-fail, low sensitivity). Persist raw rows under `~/w/vdc` (CLAUDE.md
  "never discard experiment output"), and Bonferroni/Holm-correct the gate. Scope: ISMCTS + Random
  (NMCS parity is retired — below).

## Possible cpp refactor (minor, non-blocking)

- A shared `Sampler` (just `sample_world`) under `WorldSource` (NMCS) and `ISMCTSSource` (ISMCTS),
  which currently each declare it. Review-clean as-is; extract only if a third search wants it.

## Retired

- **NMCS parity tests** marked `skip` in `tests/test_cpp_runner.py` (2026-06-16): validated repeatedly,
  and the nmcs-init milestone (a 2-level NMCS to initialize an AZ run before switching to ISMCTS) is
  far off. Re-enable when that work resumes.

*Public Domain (The Unlicense).*
