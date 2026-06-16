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

## #23 wire/result drift net — promote the floor to codegen when the C++ build lands (deferred 2026-06-16)

The Python↔C++ wire frame (`wire_spec.py`) and result blob (`result_spec.py`) are mechanized against
silent drift by `tests/test_wire_drift.py` (one SSOT per layout; always-on layout-agreement +
codec-derives-from-spec legs that fail `pytest tests/ -q` on a format-constant or codec drift; an
opt-in `CHOCO_RUN_CPP` C++ golden round-trip). Per ADR-0012 P7's hierarchy (generate/compile-from-one-
source > build-time lint > runtime parity), the always-on test is the **floor** (a lint failing the
default gate) and the golden is the **backstop** — the **top rung (codegen) is deferred for a concrete
reason: the C++ consumer doesn't exist yet** (the `ZmqNetClient` + the redis-client `cpp/` build are
deferred to the P9 `cpp/` pass; there is no `cpp/build/` in any gate). When that pass lands and the C++
side is built in a gate:

- **Generate `cpp/include/chocofarm/{wire_spec,result_spec}.hpp` from the Python SSOT** (a tiny
  build-step that emits the `constexpr` mirror from `wire_spec.py`/`result_spec.py`), so the mirror is
  *derived, not hand-written* — closing the residual gap that the headers are hand-authored today,
  joined to the SSOT only by the runtime test. The drift test stays as the backstop.
- **Add the one-line C++ cross-check** `prod(shape) * sizeof(double) == len` in `transport.cpp::
  parse_manifest` (today C++ derives a weight's element count from `len/sizeof(double)` and Python from
  `prod(shape)`; consistent only because one writer emits both — assert it).

## Possible cpp refactor (minor, non-blocking)

- A shared `Sampler` (just `sample_world`) under `WorldSource` (NMCS) and `ISMCTSSource` (ISMCTS),
  which currently each declare it. Review-clean as-is; extract only if a third search wants it.
- **Audit `using` type aliases for phantom/strong types.** Review where the cpp uses `using` for bare
  aliases — especially the many `int` indices (action slot, world index, `action.i`, `belief_key`
  fields) — and consider whether a phantom-like type template (a tagged newtype, e.g.
  `template<class Tag> struct Idx { int v; };`) is more appropriate, so semantically-distinct ints
  can't be silently mixed. Postponed for token-saving; revisit when the cpp type surface is next open.

## Retired

- **NMCS parity tests** marked `skip` in `tests/test_cpp_runner.py` (2026-06-16): validated repeatedly,
  and the nmcs-init milestone (a 2-level NMCS to initialize an AZ run before switching to ISMCTS) is
  far off. Re-enable when that work resumes.

*Public Domain (The Unlicense).*
