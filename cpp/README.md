<!-- cpp/README.md — the C++ search/sim runner (the dumb-random seam-proof MVP). Public Domain (The Unlicense). -->
# chocofarm C++ runner — the seam-proof MVP

The first C++ component in chocofarm: a thin **search/sim runner** —
`scaling-and-cpp-seam.md`'s **Shape A** — but for this slice a **dumb-random**
runner. It proves the four already-clean seams compose across the language
boundary: the **env↔Policy** seam, the **redis raw-bytes wire**, the
**version-gated weight broadcast**, and the **derived dimensions** — *before*
any Gumbel search or MLP forward is ported (both deferred to a later slice).

It is governed by **ADR-0012** (Compositional and Structural Hygiene),
especially **P7** (cross-language wire discipline) and the *"Concrete guidance
for a new-language (C++) component"* section, and validated under the **P6/P7
behavioral-equivalence bar — NOT byte-identity**.

## Layout

```
cpp/
  CMakeLists.txt            C++20 + hiredis (system) + nlohmann/json (FetchContent)
  include/chocofarm/
    instance.hpp            instance geometry: treasures/teleports/K + the DERIVED faces (cover+rep_point)
    env.hpp                 the env port: belief world-set, legal actions, apply, filters, distances
    policy.hpp              the SHARED base unit (mirrors solvers/base.py): the Policy seam + RandomPolicy
                            + GreedyBase/GreedyStopBase + UCB_C + candidate_actions + base_value + the
                            generic WorldSource sample_world seam — the one home both searches include
    nmcs.hpp                the NMCS Policy (nested Monte-Carlo search) — a drop-in alongside RandomPolicy
    ismcts.hpp              the ISMCTS Policy (single-observer Information Set MCTS) — a drop-in too
    features.hpp            §2.2 featurization + the action↔slot legality mask (all dims DERIVED)
    transport.hpp           the redis wire client (the SOLE contract; manifest-driven weight read)
    runner.hpp              the runner: read weights → run E episodes → write (X,PI,M,Y)
  src/
    instance.cpp env.cpp features.cpp transport.cpp policy.cpp nmcs.cpp ismcts.cpp runner.cpp
    main.cpp                the runner entrypoint (live scalars as CLI args; --policy random|nmcs|ismcts)
    mask_dump.cpp           a tiny PARITY fixture (replay → dump mask/features); not the runner (P3)
    nmcs_dump.cpp           a tiny PARITY fixture (scripted-source NMCS search → selected action); not the runner (P3)
    ismcts_dump.cpp         a tiny PARITY fixture (scripted-source ISMCTS search → selected action); not the runner (P3)
  parity/
    parity.py              the ADR-0012 P6/P7 behavioral-parity harness (RandomPolicy)
    nmcs_logic.py          the NMCS deterministic logic check (same action on scripted leaf returns)
    nmcs_parity.py         the NMCS aggregate behavioral parity (aggregates within MC CI)
    ismcts_logic.py        the ISMCTS deterministic logic check (same action on scripted world/expand/leaf)
    ismcts_parity.py       the ISMCTS aggregate behavioral parity (aggregates within MC CI)
  README.md
```

The `policy.hpp` / `policy.cpp` unit is the **shared base** mirroring the Python
layout: `chocofarm/solvers/base.py` holds the search-agnostic primitives
(`GreedyPolicy`, `GreedyStopBase`, `candidate_actions`, `_base_value`, `UCB_C`)
and `nmcs.py` / `ismcts.py` each **import** from it. The C++ mirrors that exactly
— `nmcs.cpp` and `ismcts.cpp` both `#include "chocofarm/policy.hpp"` for those
primitives, neither re-authors a base/sampling/leaf, and `ismcts.cpp` does **not**
include `nmcs.hpp` (ADR-0012 P1: one home, derive-don't-duplicate).

## Build

The build is self-contained except for **two documented system dependencies**:
**hiredis** (Debian/Ubuntu `libhiredis-dev`; Fedora/openSUSE `hiredis-devel`) —
the redis wire — and **libzmq** (Debian/Ubuntu `libzmq3-dev`; Fedora/openSUSE
`zeromq-devel`) — the Shape B inference client's transport, used via the C API
`zmq.h` (cppzmq's `zmq.hpp` is **not** required). Both are located by
`pkg-config` (with a `find_path`/`find_library` fallback). nlohmann/json is
fetched by CMake `FetchContent` (pinned `v3.11.3`), or a system `nlohmann_json`
≥ 3.2 is used if installed.

```sh
cmake -S cpp -B cpp/build && cmake --build cpp/build
```

This builds `cpp/build/chocofarm-cpp-runner` (the runner) and
`cpp/build/chocofarm-mask-dump` (the parity fixture).

## Run

Connection is the **same `CHOCO_TRANSPORT_REDIS_*` env contract**
`chocofarm/config.transport_redis_params()` owns (default `127.0.0.1:6380` db 0,
the ephemeral transport instance) — no hardcoded port. The runner reads
weights for a `(run, phase, version)` (a published manifest+blob must exist),
then runs `E` random episodes and writes the four result blocks.

```sh
cpp/build/chocofarm-cpp-runner \
    --instance chocofarm/data/instance.json \
    --faces    chocofarm/data/faces.json \
    --run R --phase gen --version 0 \
    --episodes 300 --lam 0.1 --max-steps 40 --seed 42 \
    --res-token T
```

`lam` / `episodes` / `max-steps` are **live CLI scalars** (P4), never baked in.
A missing weight payload is a **loud abort** (non-zero exit + the same message
`read_weights` raises), never a silent stale serve.

## The NMCS Policy (nested Monte-Carlo search)

`--policy nmcs` selects the **NMCS** Policy (`nmcs.hpp` / `nmcs.cpp`), a faithful
port of `chocofarm/solvers/nmcs.py` behind the **same env↔Policy seam** —
**zero edits to the runner / env core** (the P2 seam test: `main.cpp` is the one
place a policy is chosen, a clean `--policy random|nmcs` strategy selection; the
runner takes `const Policy&` and never names a concrete subclass).

It mirrors `nmcs.py`'s three parts exactly:

- **The level-k nested recursion** (`NMCSPolicy::search`): walk the line forward;
  at each step evaluate every candidate by a level-`(k-1)` search of its result,
  take the **argmax** (strict `>`, first-wins on ties — matching Python's
  `if q > best_q`), play it in a determinized world, continue; **memorize-and-
  replay** the best complete line's first action (Cazenave's rule). 2-level is
  the milestone.
- **The level-0 determinized base playout** (`NMCSPolicy::playout`): mean over
  `playout_samples` sampled worlds of `GreedyBase` (the λ-rational `GreedyPolicy`)
  played deterministically to the end (`base_value`), scored by the λ-penalized
  return `Σvalue − λ(travel + exit)`.
- **The per-move evaluation** (`NMCSPolicy::eval_move`): mean over `step_samples`
  determinizations of (immediate λ-step + the nested level-`(k-1)` continuation).

Candidate pruning is the shared nearest-few-detectors/treasures + always
`TERMINATE` (mirrors `solvers.base.candidate_actions(..., include_terminate=True)`).
NMCS uses **NO net** (no `NetForward`). The knobs are live CLI scalars (P4),
defaulting to `NMCSConfig`'s:

```sh
cpp/build/chocofarm-cpp-runner --policy nmcs \
    --instance chocofarm/data/instance.json --faces chocofarm/data/faces.json \
    --run R --res-token T --episodes 150 --lam 0.1 --max-steps 24 \
    --nmcs-level 1 --nmcs-playouts 3 --nmcs-step-samples 2 \
    --nmcs-cand-det 4 --nmcs-cand-tre 4 --nmcs-max-steps 24
```

### NMCS parity (ADR-0012 P6 — behavioral, not byte-identity)

The C++ NMCS has its **own** RNG (`std::mt19937_64 ≠ numpy`), so parity is the
behavioral bar. Two harnesses:

- **Deterministic logic check** (`cpp/parity/nmcs_logic.py`, needs
  `chocofarm-nmcs-dump`, **no redis**). RNG enters NMCS only through
  world-sampling, so we make the search **RNG-free on both sides** — `sample_world
  → bw[0]` (the lowest-bitmask world, same combinations order both sides) and the
  level-0 playout value → **a fixed table cycled in call order**. The recursion is
  structurally identical, so the table is consumed identically; feeding **both**
  the **same** scripted leaf returns on fixed `(loc, belief, collected)` inputs,
  the two **select the same action** — asserted for **level-1 AND level-2** (the
  milestone) over a grid of states / λ / candidate widths. This validates the
  nesting + selection logic (the part that *must* be exact) independent of RNG.
- **Aggregate behavioral parity** (`cpp/parity/nmcs_parity.py`, needs the runner
  + redis). The C++ NMCS runner and the Python `NMCSPolicy` over matched-seed
  episodes agree on every aggregate — mean length, λ-return, action-type
  distribution, belief-shrinkage — within Monte-Carlo CI (every `|z| = |Δ|/SE <
  3`), with the MC SE **reported**. NMCS is the slowest solver, so N is moderate
  (150 episodes × 2 seeds, level 1); level-2 is covered exactly by the logic
  check above.

Both are gated in `tests/test_cpp_runner.py` (skip when the fixture / redis is
absent), and `NMCSPolicy is a Policy subclass registered in SOLVERS` is an
always-on pin.

## The ISMCTS Policy (single-observer Information Set MCTS)

`--policy ismcts` selects the **ISMCTS** Policy (`ismcts.hpp` / `ismcts.cpp`), a
faithful port of `chocofarm/solvers/ismcts.py` behind the **same env↔Policy
seam** — **zero edits to the runner / env core** (the same P2 seam: `main.cpp`'s
`--policy random|nmcs|ismcts` strategy selection is the one place a policy is
chosen). It is **DRY against the shared base** (`policy.hpp`): it reuses
`base_value` (the leaf utility), `GreedyStopBase` (its default leaf base),
`UCB_C`, and the generic `WorldSource` `sample_world` draw — exactly as
`ismcts.py` imports `_base_value` / `UCB_C` / `GreedyStopBase` from
`solvers.base`. It does **not** include `nmcs.hpp`.

It mirrors `ismcts.py` exactly:

- **Information-set node** (`ISMCTSNode`): per-action `reward[a]` / `visits[a]`
  (n_j) / `avail[a]` (n'_j) **aggregated over the info-set**, children keyed by
  `(action, belief_key)` where `belief_key = (count, bw[0], bw[-1])` (the
  ISMCTS-specific `_belief_key` fingerprint, kept local).
- **Per `decide()`**: `iterations` (default 300) determinized walks; each samples
  one world `w ~ bw` and recurses `iterate` in that fixed world.
- **`iterate`**: depth≥`max_depth` (24) → `−λ·exit_cost`; `actions = legal + [TERMINATE]`;
  **bump `avail[a]` for every action** (subset-armed bandit §IV-B); if any untried,
  expand one uniformly (the source's expansion-index draw), play the base to the
  end for the leaf (`base_value` with `GreedyStopBase`), `_update`, return; else
  **UCB1-select** (eq. 7, subset-armed denominator), route the determinization to
  the `(action, belief_key)` child, recurse, backprop.
- **UCB1** (eq. 7): `exploit = reward[a]/n_j`; `navail = avail.get(a, n_j)`;
  `explore = c·sqrt(log(navail)/n_j) if navail>1 else c`; **strict `>`, first-wins
  tie over INSERTION order** — the parity-critical detail (the same hazard the NMCS
  strict-`>`/first-wins cleared). The TERMINATE edge value is `−λ·exit_cost`; a
  step value is `r − λ·dt`.
- **Final**: the **most-visited** root action (first-wins tie over insertion
  order); TERMINATE if nothing was tried.

`ISMCTSConfig` (`iterations=300`, `c=UCB_C`, `max_depth=24`) is the frozen scalar
config; `base` (the `GreedyStopBase` leaf) is a separate construction param. The
knobs are live CLI scalars (P4):

```sh
cpp/build/chocofarm-cpp-runner --policy ismcts \
    --instance chocofarm/data/instance.json --faces chocofarm/data/faces.json \
    --run R --res-token T --episodes 120 --lam 0.1 --max-steps 24 \
    --ismcts-iterations 300 --ismcts-c 0.7 --ismcts-max-depth 24
```

### ISMCTS parity (ADR-0012 P6 — behavioral, not byte-identity)

The C++ ISMCTS has its **own** RNG (`std::mt19937_64 ≠ numpy`), so parity is the
behavioral bar. Two harnesses, mirroring the NMCS pair:

- **Deterministic logic check** (`cpp/parity/ismcts_logic.py`, needs
  `chocofarm-ismcts-dump`, **no redis**). ISMCTS's THREE RNG draws are scripted
  RNG-free on both sides — `sample_world → bw[0]`, `expand_index → a fixed FIFO
  mod n`, the leaf value → **a fixed table cycled in call order**. The descent is
  structurally identical, so the FIFOs are consumed identically; feeding **both**
  the **same** scripted draws on fixed `(loc, belief)` inputs, the two **select
  the same most-visited action** — asserted across iteration counts (1/4/16/64/300)
  that cover **pure expansion**, **UCB selection**, and **the availability
  denominator**, plus the TERMINATE edge and the most-visited final, over a grid
  of states / `c` / `max_depth` / λ. This validates the selection + nesting logic
  (the part that *must* be exact) independent of RNG.
- **Aggregate behavioral parity** (`cpp/parity/ismcts_parity.py`, needs the
  runner + redis). The C++ ISMCTS runner and the Python `ISMCTSPolicy` over
  matched-seed episodes agree on every aggregate — mean length, λ-return,
  action-type distribution, belief-shrinkage — within Monte-Carlo CI (every
  `|z| = |Δ|/SE < 3`), with the MC SE **reported**. ISMCTS runs many iterations
  per decision, so N is moderate (120 episodes × 2 seeds at iterations=80); the
  full default `iterations=300` selection is covered exactly by the logic check.

Both are gated in `tests/test_cpp_runner.py` (skip when the fixture / redis is
absent), and `ISMCTSPolicy is a Policy subclass registered in SOLVERS` is an
always-on pin.

## How the env port mirrors `env.py` / `facemodel.py`

- **Belief world-set:** the full `C(20,5)=15504` bitmask world-set, built in
  `itertools.combinations(range(N), K)` order (mirrors `instance.world_array`).
  `bw` is a `std::vector<uint32_t>` filtered in place — logic-exact, so
  bit-identical to the numpy env's belief.
- **Cover structure is DERIVED from geometry, not fossil arrays.** The
  disjunctive cover (each face's `bitmask` + `rep_point`) is read **only** from
  `data/faces.json` — the intersection-refinement of the atomic detectors the
  geometry pipeline (`scripts/chocobo_geometry.py` →
  `arrangement.arrangement()`) produces, exactly the derivation
  `facemodel.sense_actions` wraps. The loader **never reads** `instance.json`'s
  `overlaps` / `delta_treasures` arrays (the superseded per-region cover the
  face model replaced; their removal in the parallel cleanup does not touch
  this port).
- **Dynamics:** `legal_actions` (collects with `marg>0` & not collected, in
  treasure-id order; informative faces in face-id order; TERMINATE the
  always-legal extra slot), `apply` (move → observe → filter belief → collect),
  `filter_treasure` / `filter_detector` (the disjunction over a face's cover),
  `exit_cost`, distances via `std::hypot` — each mirrors its `env.py` method.
- **Featurization + mask:** the §2.2 layout (per-treasure `N×5`, per-detector
  `nD×3`, global `6+n_tel`) and the `N + nD + 1 = 65`-slot mask, every dimension
  **derived** from the env (`feat_dim` / `n_action_slots`), nothing hardcoded.

## Parity (ADR-0012 P6/P7 bar — behavioral, not byte-identity)

`PYTHONPATH=. <py> cpp/parity/parity.py` (needs the binary built + redis up):

- **Logic invariants → bit-exact.** The legality mask `M` is **byte-identical**
  to Python's `legal_mask` for the same `(loc, belief)` (matched-sequence
  replay through `chocofarm-mask-dump`); illegal-slot `PI` mass is `== 0.0`.
- **Wire-content cross-impl parity.** Each C++ episode emits its exact trace
  `(world, executed slots)`; the harness **replays the same episode in Python**
  and value-compares the **actual wire bytes** read back from redis against an
  **independent** Python computation — `PI` and `M` **bit-exact**, `X` and `Y`
  to `ABS_TOL=1e-4` — over **12,150 decisions × 2 seeds**. This compares the
  `PI`/`Y` *content* (not just illegal-mass + shape), mechanizing the
  manifest-round-trip / wire-content parity ADR-0012 P7's self-application flags
  as otherwise deferred.
- **Float-sensitive / RNG-driven → aggregate behavioral equivalence.** Mean
  episode length, mean λ-return (at a fixed λ₀), action-type distribution, and
  mean belief-shrinkage over **N=400 episodes × 2 seeds (800/side)** are
  statistically indistinguishable within Monte-Carlo CI (every `|z| = |Δ|/SE <
  3`), with the MC standard error reported. (Independent RNG draws per side —
  the comparison is a genuine Monte-Carlo distribution comparison, not a matched
  world.)
- **Feature X-port → forward-roundoff bar.** The §2.2 feature vector matches
  Python's to `max|Δ| < 1e-4` (the `test_jax_equivalence` `ABS_TOL`; observed
  `~1e-7`, pure float64 roundoff).
- **Format round-trip.** The four blocks read back via
  `np.frombuffer(...).reshape(...)` per `transport.py` at the exact shapes
  `X (n,feat_dim)`, `PI`/`M (n,n_slots)`, `Y (n,)`, all float32.

A pytest entry (`tests/test_cpp_runner.py`) pins the Python `RandomPolicy`
baseline always-on, and runs the full harness opt-in (skips if the binary or
redis is absent, so `pytest tests/ -q` stays green without the C++ build).

## ADR-0012 self-check (principle → where it lives, and any partial honor)

- **P1 — single source of truth / derive-don't-duplicate.** The weight read
  (`transport.cpp::read_weights`) binds each weight **by the manifest's**
  `(name, shape, dtype, off, len)` — **no hardcoded offset** anywhere; an
  unexpected dtype is rejected loudly. The result write emits **no second
  encoder**: the four float32 blocks are exactly what `np.frombuffer().reshape()`
  decodes. Every dimension (`feat_dim`, `n_action_slots`, the world count) is
  **derived from the env**, never typed as a literal. "Which redis" defers to
  `config.py`'s `CHOCO_TRANSPORT_REDIS_*` contract (the transport role, default
  6380 db0) — this client only reads the same env vars, it does not re-own the
  connection facts.
- **P2 — composable Policy seam.** `policy.hpp` defines the abstract `Policy`
  with the single `decide(env, loc, bw, collected, lam, rng)` method (mirroring
  Python's). The env owns all dynamics; the runner takes `const Policy&` and
  **never names a concrete subclass**, so a new capability (a search/MLP policy)
  is a **new `Policy` subclass with zero edits to the env core or the runner** —
  verified by construction (`main.cpp` is the only file that names
  `RandomPolicy`; swapping it is a one-line change there).
- **P4 — live, not frozen.** `lam`, `episodes`, `max_steps` arrive as live CLI
  scalars in `RunnerConfig` and are threaded to each decision; **nothing is
  baked into the policy or env object**. `RandomPolicy` ignores `lam` (a
  dumb-random runner) but it stays in the seam signature, so a value-aware
  policy is a drop-in with no signature change.
- **P7 — the wire is the only contract.** The C++ side shares **no types** with
  Python; it speaks only the redis raw-bytes protocol `transport.py` owns —
  the same `az:w:<run>:<phase>:<version>:m|:b` weight keys (float64 blob +
  manifest) and `az:res:<token>:<idx>:X|PI|M|Y` result keys (float32 blocks,
  `CHOCO_RESULT_TTL` expiry). A missing payload is a loud abort. Validated by
  the matched-seed aggregate parity above.

**Partial-honor caveats (this is a thin slice — stated explicitly per P6/P7):**

- **The weights are read but not consumed.** `RandomPolicy` is search-free, so
  the runner **exercises** the manifest-driven weight-read seam (parse by
  manifest, abort loud on missing) to prove it, but does not run a forward.
  P1's "no hardcoded offset / no second encoder" is fully honored on the read
  path; the *consume* path (the MLP forward `forward_core`) is **deferred** to
  the search slice.
- **The PI target is the policy's own action distribution**, not a Gumbel
  search-improved π′ (there is no search yet — `RandomPolicy` has no
  `decide_with_value`, so it structurally cannot traverse `generate_episode`).
  It is a valid normalized target **on the legality mask** (illegal-slot mass
  `== 0.0`), and the wire-content parity above confirms the emitted `PI` bytes
  are **bit-exact** to an independent Python computation of that same rule — so
  what *is* shipped is an honest mirror; only the search-improved π′ *semantics*
  is deferred (it arrives with the search).
- **The value target is the pure-MC λ-penalized return-to-go** (the
  `lam_blend=1 / n_step=None` limit `generate_episode` produces by default; the
  C++ formula is bit-equivalent to `value_target.suffix_returns_to_go`, and the
  wire `Y` bytes are confirmed against an independent Python replay above). The
  TD(λ)/n-step blend that needs the search's root-value bootstrap is **deferred**
  (it has no meaning without the search bootstrap that does not yet exist).
- **P3 (no god-objects)** is honored structurally (env / policy / features /
  transport / runner are one-owner collaborators; the mask-replay fixture is a
  separate executable from the runner), but it is the principle this slice
  exercises least deliberately — it is a small codebase. **P5** (fail loud) is
  honored on every wire/IO boundary (missing payload, unreachable redis,
  malformed instance, dtype/shape mismatch all raise/abort); there is no
  band-aid stack because there is no substrate fight yet.
