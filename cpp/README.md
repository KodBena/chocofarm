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
    policy.hpp              the composable Policy interface (P2) + RandomPolicy (the trivial drop-in)
    features.hpp            §2.2 featurization + the action↔slot legality mask (all dims DERIVED)
    transport.hpp           the redis wire client (the SOLE contract; manifest-driven weight read)
    runner.hpp              the runner: read weights → run E episodes → write (X,PI,M,Y)
  src/
    instance.cpp env.cpp features.cpp transport.cpp runner.cpp
    main.cpp                the runner entrypoint (live scalars as CLI args)
    mask_dump.cpp           a tiny PARITY fixture (replay → dump mask/features); not the runner (P3)
  parity/
    parity.py              the ADR-0012 P6/P7 behavioral-parity harness
  README.md
```

## Build

The build is self-contained except for **one documented system dependency,
hiredis** (Debian/Ubuntu `libhiredis-dev`; Fedora/openSUSE `hiredis-devel`).
nlohmann/json is fetched by CMake `FetchContent` (pinned `v3.11.3`), or a
system `nlohmann_json` ≥ 3.2 is used if installed.

```sh
cmake -S cpp -B cpp/build && cmake --build cpp/build
```

This builds `cpp/build/chocofarm-cpp-runner` (the runner) and
`cpp/build/chocofarm-mask-dump` (the parity fixture).

## Run

Connection is the **same `CHOCO_REDIS_*` env contract** `chocofarm/config.py`
owns (default `127.0.0.1:6379` db 0) — no hardcoded port. The runner reads
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
- **Float-sensitive / RNG-driven → aggregate behavioral equivalence.** Mean
  episode length, mean λ-return (at a fixed λ₀), action-type distribution, and
  mean belief-shrinkage over **N=400 episodes × 2 seeds (800/side)** are
  statistically indistinguishable within Monte-Carlo CI (every `|z| = |Δ|/SE <
  3`), with the MC standard error reported.
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
  `config.py`'s `CHOCO_REDIS_*` contract — this client only reads the same env
  vars, it does not re-own the connection facts.
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
  search-improved π′ (there is no search yet). It is a valid normalized target
  **on the legality mask** (illegal-slot mass `== 0.0`), and the format/round-
  trip it exercises is the real wire contract; the search-improved π′ arrives
  with the search.
- **The value target is the pure-MC λ-penalized return-to-go** (the
  `lam_blend=1 / n_step=None` limit `generate_episode` produces by default); the
  TD(λ)/n-step blend that needs the search's root-value bootstrap is **deferred**
  (it has no meaning without the search bootstrap).
- **P3 (no god-objects)** is honored structurally (env / policy / features /
  transport / runner are one-owner collaborators; the mask-replay fixture is a
  separate executable from the runner), but it is the principle this slice
  exercises least deliberately — it is a small codebase. **P5** (fail loud) is
  honored on every wire/IO boundary (missing payload, unreachable redis,
  malformed instance, dtype/shape mismatch all raise/abort); there is no
  band-aid stack because there is no substrate fight yet.
