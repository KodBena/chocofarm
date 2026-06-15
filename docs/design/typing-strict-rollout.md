# mypy --strict rollout — assessment & staged plan (2026-06-15)

A scoping record (authored at decision time, per the documentation discipline),
produced by an assessment workflow before any annotation was written. The
question, raised by the maintainer: *most of the code is untyped — why, and how
do we bring it to `mypy --strict`?* This note states the measured extent, the
genuine bugs `--strict` surfaces, the staged plan, the proposed config, and the
ADR-0012 principle the discipline becomes.

> **chocofarm is genuinely untyped — 9% of in-package functions (33/375) carry
> full param+return annotations. A from-scratch `mypy --strict` run emits 913
> errors, but 89% (816) are the pure annotation backlog (`no-untyped-def` + its
> downstream `no-untyped-call`); after reading every flagged site the genuine
> latent bugs reduce to exactly ONE cluster (3 errors, all in
> `az/mlp_jax_train.py`). Full `--strict` is achievable as the end state — no
> language-level blocker, only work — and the honest bar to gate on today is
> global `strict` plus four narrow, individually-justified stub-gap relaxations
> (numba, optax, tensorboardX, redis), with jax handled as a documented use-site
> `Any` (jax ships `py.typed`, so it is NOT blanket-ignored).**

## 1. Current state (measured)

- **mypy was not installed** in the project interpreter; installed `mypy 2.1.0`.
- **No mypy config exists** anywhere (no `pyproject.toml`, `mypy.ini`,
  `setup.cfg`, `tox.ini`); the project is also unpackaged (no build config).
- **Typed-def ratio (full param + return annotations), `chocofarm/**/*.py`:**

  | area | typed / total | % |
  | --- | --- | --- |
  | `az/` (the bulk) | 8 / 188 | 4% |
  | `model/` | 10 / 33 | 30% |
  | `hp/` | 11 / 37 | 30% |
  | `bounds/` | 0 / 25 | 0% |
  | `solvers/` | 0 / 60 | 0% |
  | `eval/` | 0 / 21 | 0% |
  | top-level (`config.py`, `references.py`, …) | 4 / 11 | 36% |
  | **chocofarm/ overall** | **33 / 375** | **9%** |
  | `tools/analysis/` (offline, outside the package) | 18 / 31 | 58% |

  91% of in-package functions lack full annotations — the maintainer's premise,
  confirmed.

## 2. `mypy --strict` result — 913 errors, small sharp signal

| code | count | meaning |
| --- | --- | --- |
| `no-untyped-call` | 468 | calling an unannotated fn — second-order; vanishes as callees are annotated |
| `no-untyped-def` | 348 | missing param/return annotations — the bulk, mechanical |
| `type-arg` | 28 | bare generics (`dict`/`tuple`/`list` sans `[...]`) — trivial |
| `var-annotated` | 19 | empty-collection locals need an annotation — noise |
| `assignment` | 10 | mostly sentinel/narrowing; **2 genuine** (mlp_jax_train default-None) |
| `index` | 9 | indexing `Any|None` — all None-sentinel-init, guarded before use |
| `import-untyped` | 9 | shapely (×3), numba, optax, tensorboardX — stub-gaps |
| `no-any-return` | 6 | numpy/jax/`asdict` `Any`-leakage — resolved by `cast`, not logic |
| `arg-type` | 6 | mostly numpy/jax leakage; **the mlp_jax_train pair is genuine** |
| `attr-defined` | 5 | 3 implicit-reexport FALSE positives + 2 narrowing noise |
| `union-attr` | 3 | None-sentinel-init (`_Node`, the `_WORKER` global) — guarded |
| `method-assign` | 2 | intentional test instrumentation (`capture_states`, restored in `finally`) |

**89% (816) is the annotation backlog.** Everything mypy shouts under
`index`/`union-attr`/`attr-defined` is None-sentinel-init or implicit-reexport
noise on untyped code — verified site-by-site, not runtime defects.

## 3. The genuine latent bugs — fix regardless of the rollout

Exactly one cluster, all in `chocofarm/az/mlp_jax_train.py` (the item-M optimizer
path). Both are ADR-0002 *lying-signature* / ADR-0012 P2 *"a parameter the
receiver cannot honor"* defects, now made type-visible:

- **P0 — the lying signature.** `hp: AdamHParams = None` at `train_step:260` and
  `train_step_value:290`: a non-Optional param defaulted to `None`, and the body
  (`hp = self._default_hp if hp is None else hp`) *proves* `None` is an accepted,
  documented value. The annotation lies. → `hp: AdamHParams | None = None`.
- **P1 — the contract violation.** `AdamHParams` (a `NamedTuple`) declares
  `lr/b1/b2/eps` as `float`, but `_hp_arrays` constructs it with
  `jnp.asarray(...)` — traced jax Arrays. Downstream consumers reading the fields
  as floats get traced scalars. → widen to `float | jax.Array` (they genuinely
  hold both forms across the two construction paths).

## 4. Module landscape (by typeability)

- **easy_strict (full `--strict` cheap):** `config.py`, `references.py`,
  `hp/schema.py`, `hp/registry.py`, `az/dtypes.py`, `model/instance.py`, the
  standalone dataclasses/NamedTuples/Protocols (`AdamHParams`,
  `RolloutConfig`/`SparseSamplingConfig`, the `Vhat` Protocol contract), all
  `__init__.py`. ~25 trivial errors total.
- **medium (38 modules — signatures typeable, array internals `NDArray[Any]`):**
  `model/env.py`, `solvers/base.py`, the `model/`/`az/` leaves, the solvers, the
  bounds, all of `eval/`. None touch jax/optax/numba — strict here is pure work.
- **hard (5 — real friction):** `az/kernels.py` (numba `@njit` erases
  signatures), `az/forward.py`, `az/mlp_jax.py`, `az/mlp_jax_train.py`,
  `az/optimizer.py` (the inject_hyperparams transform half).

## 5. Staged plan

- **Stage 0 — land the gate (mechanism first, ADR-0011).** Add `[tool.mypy]` to
  a new `pyproject.toml` (§6) with global `strict` + the four justified stub-gap
  overrides + the jax use-site `Any` posture; install `types-shapely`. Do *not*
  gate the whole tree red on day one — ratchet a monotonically-decreasing
  baseline, enforcing the Stage-1 set first.
- **Stage 1 — easy_strict → full strict** and make them the gate's enforced
  core. ~1–1.5 days.
- **Stage 2 — the keystone: type the `env↔Policy` seam** (`env.py` + `base.py`,
  with `Loc`/`Action` aliases). Every consumer imports `Environment`/
  `Policy.decide`; nailing this collapses a large share of the 468 downstream
  `no-untyped-call` for free. ~2–3 days, high-judgment, low-volume.
- **Stage 3 — the medium bulk** (38 modules, bottom-up so `no-untyped-call`
  dissolves as callees land). `decomp.py` (675L) and `registry.py` are ADR-0004
  minimal-touch heavies — signatures only, no rewrites. ~6–9 days, the bulk.
- **Stage 4 — the hard jax/optax/numba modules behind documented `Any` seams.**
  ~2–3 days; friction is concentration, not volume.
- **Stage 5 — fix the §3 genuine bugs** (independent; ship immediately).

**Total ~12–19 engineer-days to a fully-gated `--strict` end state; a usable
gate (Stages 0–2) in ~4–5, delivering most of the safety value, with Stages 3–4
ratcheting the baseline down under ADR-0004's on-touch posture.**

## 6. Proposed `[tool.mypy]` config

```toml
[tool.mypy]
# chocofarm has NO existing config and is unpackaged; this stands up the
# ADR-0011 CI-gate mechanism from scratch. Global bar = the maximal real
# strictness achievable now. Every relaxation below is a documented stub-gap,
# never a convenience.
python_version = "3.11"
files = ["chocofarm"]
strict = true                      # disallow_untyped_defs/calls/incomplete,
                                   # check_untyped_defs, no_implicit_optional,
                                   # warn_return_any, strict_equality, …
warn_unreachable = true
warn_redundant_casts = true
warn_unused_ignores = true         # keeps per-module ignores honest — a
                                   # relaxation that stops being needed fails CI
disallow_any_generics = true       # the 28 bare-generic noise
no_implicit_reexport = true        # KEEP strict; fix the 3 attr-defined FALSE
                                   # positives by adding __all__ (BeliefRefs/
                                   # DECOMP_ANCHOR/is_weight ARE re-exported)
show_error_codes = true

# === NARROW per-module relaxations — each a GENUINE stub-gap (the lib ships NO
# py.typed), scoped to the import name, NEVER a convenience. ===
[[tool.mypy.overrides]]
module = ["numba", "numba.*"]          # @njit erases the decorated signature
ignore_missing_imports = true
[[tool.mypy.overrides]]
module = ["optax", "optax.*"]          # GradientTransformation unstubbed
ignore_missing_imports = true
[[tool.mypy.overrides]]
module = ["tensorboardX", "tensorboardX.*"]   # logging sink, no type contract
ignore_missing_imports = true
[[tool.mypy.overrides]]
module = ["redis", "redis.*"]          # partial bundled types; weakest, first to drop
ignore_missing_imports = true
# shapely: NO override — install `types-shapely` (stubs beat an ignore).
# jax:     NO override — jax ships py.typed; its pytree/opt_state/backend-`xp`
#          friction is a COMMENTED `Any` at each seam site (visible in the diff),
#          confined to az/{forward,mlp_jax,mlp_jax_train,optimizer,kernels}.py.
```

## 7. Per-module relaxations — constraint, not excuse

Each is a *verified* stub-gap (the library ships no `py.typed` and no stubs),
documented at the narrowest scope:

- **numba** — no stubs; `@njit` erases the decorated signature. Hand-stubbing is
  disproportionate. `warn_unused_ignores` flags the ignore if numba ever ships
  `py.typed`.
- **optax** — no `py.typed`; `GradientTransformation`/`inject_hyperparams`/the
  `opt_state` dict are unstubbed.
- **tensorboardX** — no stubs; a logging sink, no type contract crosses into the
  numerics.
- **redis** — partial bundled types for the raw-bytes surface; the client is
  deliberately duck-typed (the ADR-0012 P7 bytes-store role). The weakest of the
  four — prefer dropping it once a typed redis surface covers the bytes API.
- **jax pytree / `opt_state` / backend-`xp` seam** — **NOT** a config ignore.
  jax ships `py.typed` and is checkable, so a blanket ignore would be a
  convenience-relaxation and is rejected. The friction (a flat `dict[str,
  jax.Array]` pytree, `value_and_grad`'s untyped return, `@jax.jit → Any`,
  `forward_core`'s backend-polymorphic `xp`) is an explicit, *commented* `Any` at
  each use site — a named relaxation visible in the source, distinguishing
  constraint from excuse in the diff itself.

## 8. The discipline becomes ADR-0012 P8 (draft)

To be folded into ADR-0012 as **P8 — Typed signatures are the single source of
truth of a function's contract** (the structural twin, at the call boundary, of
P1's derive-don't-duplicate and ADR-0002's no-lying-signature). The bar is
*strict-where-achievable*; the named-relaxation posture above is the rule; the
mypy CI gate is the ADR-0011 mechanism that converts "typed signatures" from
review-only prose into an enforced contract. It reuses the P7 reframe's
no-scale-excuse language verbatim — *never* justify a weaker bar with a scale /
"one maintainer" / "for now" / minimality argument; that argument shape is the
tell. (Folded in after the P7 reframe landed at `72f3c1d`.)

## 9. Status

- **Assessment only; no annotations written.** `mypy 2.1.0` installed into the
  project venv; full `--strict` output captured during the run. The two genuine
  bugs (§3) and the foundation (§5 Stages 0–1 + the principle) are the first
  slice; the medium/hard grind (Stages 2–4) is the ~10-day push.

*Public Domain (The Unlicense).*
