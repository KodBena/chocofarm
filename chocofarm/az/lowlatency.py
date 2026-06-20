#!/usr/bin/env python3
"""
chocofarm/az/lowlatency.py — a phantom-typed, SSOT-constructed, low-overhead JAX dispatcher.

A micro-library (one handle type, one constructor, one call) that adapts ANY pure jax function
`fn(params, x) -> y` into a minimal-dispatch callable, AOT-compiled once for a fixed input
shape/dtype. It is the intended SSOT builder for the codebase's latency-critical JAX paths — the
MLP inference forward (`az/inference_server.jit_forward_core`, `az/mlp_jax`) and the control-lab RL
policies — so a hot path declares "compile this fn for THESE example shapes, hand me back a typed
callable" in one place, with all validation at construction and the phantom type carrying the
input/output contract downstream (ADR-0012 P8: the typed signature is the contract's SSOT).

WHY AOT (the premise this attacks). A small jit'd JAX computation at small batch is DISPATCH-BOUND:
the per-call Python overhead (pytree flatten, jit cache lookup, host->device transfer, async
dispatch) can dominate the XLA compute, so the FIXED per-call cost (the regression INTERCEPT of
time-vs-batch) is large relative to the per-row SLOPE. JAX's AOT remedy is
`jax.jit(fn).lower(*example_args).compile()`, which returns a `Compiled` executable called directly
with no re-trace; the lowest-overhead "unsafe" path bypasses even the `Compiled.__call__` wrapper by
handing the already-flattened, already-device-placed leaves straight to the loaded executable.

WHAT THE BENCH FOUND (ADR-0009 honesty — measured by chocofarm/az/bench/bench_lowlatency.py, numbers
under ~/w/vdc/chocobo/bench/lowlatency/; warm, median±IQR, R²>0.99 linear fits, jax 0.10.1 single-thread
CPU, toy 2-layer MLP value head in_dim=80 hidden=256). The falsifiable claim — *the abstraction lowers
the per-call INTERCEPT (the fixed dispatch cost) without inflating the SLOPE (per-row compute)* — is the
gate, and it splits by path:

  * ROBUST AOT path: HOLDS, and is the load-bearing positive result. In the server's true call pattern
    (a fresh host numpy `x` in, the result pulled to host), fitting `time = intercept + slope·rows`:
    plain `jax.jit` intercept ≈ 121 us, robust handle intercept ≈ 64 us — a ~57 us (≈47%) DROP,
    reproducible to <1 us across runs — with the slope essentially UNCHANGED (≈2.04 → ≈1.96 us/row).
    The win is NOT "AOT skips a re-trace" (a warm jit's cache lookup is already cheap): it is that the
    handle stages the PARAMS device-resident ONCE at construction and re-passes those device buffers,
    so each call transfers only `x` — eliminating the per-call params pytree-flatten + host->device
    transfer that plain `jit(params, jnp.asarray(x))` repeats every call. The device-resident control
    (params staged AND `x` pre-placed) confirms it: plain-jit ≈ 46 us vs robust ≈ 49 us converge, so
    the ~57 us the robust handle saves in the host pattern is exactly that per-call params staging.
  * UNSAFE (loaded-executable) path: does NOT hold here — intercept ≈ 224 us, ~+103 us WORSE than plain
    jit (the `ExecuteReplicated.__call__` Python path is heavier than both pjit's C++ dispatch fast-path
    and the `Compiled` wrapper on this backend). So the unsafe path is built and MEASURED but is OFF by
    default, with the robust path as the documented fallback — kept for backends/versions where the
    pre-flattened call may win.

So the abstraction delivers a real, substantiated dispatch-intercept reduction via the robust AOT path
(by staging params on-device), AND a clean SSOT-constructed, fail-loud, phantom-typed handle. A further
orthogonal lever the bench isolates but this module does not itself apply (the CALLER's job) is keeping
`x` device-resident, donating a staged input buffer, and amortizing the host-pull over a larger batch —
which is why this is the micro-library + its own toy bench ONLY, not a rewire of the inference server.

Contract / failure semantics (ADR-0002, ADR-0012 P2 Port/ACL — translate-and-validate, never coerce;
P9 functional core): `compile_lowlatency` is the ONE constructor (the Port/ACL boundary). It lowers,
AOT-compiles, and validates the example args' shapes/dtypes AT CONSTRUCTION, raising loudly on a
malformed input; the returned `LowLatencyFn` is an immutable value (the only effect is construction).
`run(handle, *args)` then trusts the phantom-typed contract — a per-call shape mismatch is XLA's loud
error on the robust path, and is the documented unchecked precondition on the unsafe path.

Typing (ADR-0012 P8 / the mypy --strict gate): `LowLatencyFn[In, Out]` is `Generic` in two PHANTOM
type parameters — `In` (the pytree type of the input args) and `Out` (of the output). They exist only
for the static contract (`mypy --strict`); they carry ZERO runtime cost (no field stores them). The
jax internals (`Compiled`, the loaded executable, the flattened leaves) are the documented
backend-`Any` seam jax's stubs leave — the same honest use-site `Any` `mlp_jax`/`forward_core` ride,
visible in the source, not a blanket ignore (jax ships py.typed; per pyproject there is no jax
override).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar, cast

# The XLA/OMP single-thread pin lives in ONE home (chocofarm/config.py, ADR-0012 P1). Importing config
# BEFORE jax applies it (config sets the env at its import) — the same side-effect import mlp_jax uses,
# so a handle built here lands on the same single-threaded Eigen backend the inference server runs.
from chocofarm import config as _config  # noqa: F401 — side-effect import: applies the XLA/OMP pin pre-jax

import jax
import jax.numpy as jnp
import jax.tree_util as _tu

# The phantom type parameters. `In` is the pytree type of the input args tuple the handle was compiled
# for; `Out` is the type of the function's output. They are NEVER stored on the handle (zero runtime
# cost) — they exist only so a caller can write `LowLatencyFn[MyInput, MyOutput]` and have mypy track
# the contract end to end. Bound to nothing: any pytree-shaped value qualifies.
In = TypeVar("In")
Out = TypeVar("Out")

# A jax `Compiled` AOT executable (`jax.stages.Compiled`). jax's stubs type `.lower().compile()` loosely
# across the stack, so this is the documented backend-`Any` seam (P8 commented use-site Any, not a
# blanket ignore) — the same one mlp_jax rides for its jitted forward.
_Compiled = Any


@dataclass(frozen=True)
class AvalSpec:
    """The staged shape/dtype of one array leaf — an abstract value (jax `ShapedArray`) reduced to the
    two facts a fail-loud boundary checks: `shape` and `dtype` (a numpy dtype name, e.g. 'float32').

    Immutable (ADR-0012 P9 — a value, not a mutable record). Carries no jax object so the handle's
    metadata is plain-Python inspectable (the bench and a caller read `.in_avals` without touching the
    device)."""

    shape: tuple[int, ...]
    dtype: str

    @staticmethod
    def of(aval: Any) -> "AvalSpec":
        """Reduce a jax abstract value (a `ShapedArray`, as carried on `Compiled.in_avals`/`.out_info`)
        to the plain `(shape, dtype)` pair. `aval.shape` is a tuple of ints; `str(aval.dtype)` is the
        stable dtype-name string we compare on (a numpy dtype stringifies to its name, e.g. 'float32'),
        so two avals match iff their shape and dtype-name match — the exact contract the AOT executable
        was compiled for."""
        return AvalSpec(tuple(int(s) for s in aval.shape), str(aval.dtype))


@dataclass(frozen=True)
class LowLatencyFn(Generic[In, Out]):
    """An immutable, phantom-typed handle around ONE AOT-compiled jax executable (ADR-0012 P9: the
    handle is a value; `compile_lowlatency` is the only effect that builds it).

    The two type parameters `In`/`Out` are PHANTOM — they appear in no field, cost nothing at runtime,
    and exist only for the static `mypy --strict` contract (P8): a caller annotates
    `LowLatencyFn[InputArgs, Output]` and mypy threads that contract through `run`/`__call__`. The
    runtime carries only the jax executable plus its staged shape/dtype metadata and the
    flatten-once plumbing for the unsafe path.

    Build it through `compile_lowlatency` — the SSOT constructor (the Port/ACL boundary). Do not
    construct directly: the validated invariants (the executable matches `in_avals`, the static leaves
    are device-placed, the leaf order matches `in_tree`) are established there."""

    # The robust AOT executable — `jax.jit(fn).lower(*example_args).compile()`. Called directly with no
    # re-trace; jax flattens the pytree args against the compiled `in_tree` and validates avals.
    _compiled: _Compiled
    # The staged params (the fn's first arg), device-placed once at construction. `Compiled.__call__`
    # expects the SAME positional args it was lowered with — `(params, x)` — so the robust path re-passes
    # these staged DEVICE params each call (device_put on a device array is identity, so no re-transfer).
    # Held here, not re-passed by the caller, so `run(handle, x)` takes only the varying `x` (the params
    # are the handle's staged closure — P4: weights are RESTART-cold relative to the per-call hot `x`).
    _params: Any
    # Staged metadata (P1 derive-don't-duplicate: these are READ OFF the compiled executable at
    # construction, never re-typed). `in_avals` is one AvalSpec per flattened input leaf, in `in_tree`
    # leaf order; `out_aval`s the same for the output. The bench and a caller introspect these.
    in_avals: tuple[AvalSpec, ...]
    out_avals: tuple[AvalSpec, ...]
    # Whether the function returns a single array (out pytree is one leaf) — so the robust call can
    # return the array itself, matching the plain `jit(fn)(x)` ergonomics, not a 1-tuple.
    _single_output: bool
    # The unsafe (loaded-executable) call surface and its plumbing. `_unsafe_call` is the executable's
    # pre-flattened entry (`_executable.unsafe_call`) — see `run(..., unsafe=True)` for the contract it
    # assumes. `_static_leaves` are the params/closure leaves flattened+device-placed ONCE at
    # construction; `_x_leaf_index` is where the per-call varying leaf sits in the flat leaf list; the
    # full flat leaf order is `_static_leaves` with the call arg spliced in at `_x_leaf_index`.
    _unsafe_call: Callable[..., Any] | None
    _static_leaves: tuple[Any, ...]
    _x_leaf_index: int
    # Whether unsafe is the chosen default for THIS handle (prefer_unsafe at construction AND the path
    # was buildable). When False, `__call__`/`run` take the robust path; unsafe stays reachable via
    # `run(handle, x, unsafe=True)` for the bench/an explicit caller.
    use_unsafe: bool = field(default=False)

    # ---- the ONE low-overhead call ----
    def __call__(self, x: Any) -> Out:
        """Minimal-dispatch call: `handle(x) -> y`, same result as `jax.jit(fn)(params, x)` for the
        staged params. Takes the robust AOT path unless this handle was built `prefer_unsafe=True` and
        the unsafe path was constructible. `x` is the single varying argument (the params/closure were
        staged at construction). Returns the bare output array when the fn has a single-array output
        (matching plain-jit ergonomics), else the output pytree."""
        return run(self, x, unsafe=self.use_unsafe)


def _flatten_call(params: Any, x: Any) -> "tuple[list[Any], Any]":
    """Flatten the `(params, x)` call args into the canonical leaf list + treedef jit uses. jax jits a
    function of positional args, flattening `((args...), kwargs)`; here the args are `(params, x)` with
    no kwargs, so the flat leaf order is `tree_flatten(((params, x), {}))`. Single-homed here so the
    constructor and the unsafe call agree on leaf order by construction (P1)."""
    leaves, treedef = _tu.tree_flatten(((params, x), {}))
    return leaves, treedef


def compile_lowlatency(
    fn: Callable[[Any, Any], Any],
    params: Any,
    example_x: Any,
    *,
    donate_x: bool = False,
    prefer_unsafe: bool = False,
) -> "LowLatencyFn[Any, Any]":
    """THE SSOT constructor (the ONLY way to build a `LowLatencyFn`) — the Port/ACL boundary that
    lowers, AOT-compiles, validates, and stages, returning the immutable typed handle.

    `fn(params, x) -> y` is any PURE jax function (P9 functional core). `params` is the staged
    closure/weights (a pytree of jax arrays — held device-resident on the handle); `example_x` is a
    representative input whose shape/dtype pins the ONE compiled signature. The handle then dispatches
    `fn(params, ·)` for inputs of that exact shape/dtype.

    AOT (the jax 0.10.1 idiomatic path): `jax.jit(fn, donate_argnums=...).lower(params, example_x)
    .compile()` returns a `jax.stages.Compiled`. `donate_x=True` donates the input buffer
    (`donate_argnums=1`) — the documented "unsafe trades safety for speed" knob: it lets XLA reuse the
    input's device buffer for the output, so the caller must NOT reuse `x` after the call. Donation of
    a host (numpy) `x` is a no-op (jax can only donate device buffers); it bites only when the caller
    pre-stages `x` on-device (the real latency lever this module documents but does not itself apply).

    Validation AT CONSTRUCTION (ADR-0002 fail-loud, translate-and-validate not coerce): the example
    args are device-placed (`jax.device_put`) so the staged params and the unsafe static leaves are
    real device buffers, and the flattened input leaves are checked to be array-like (carry `.shape`
    and `.dtype`) — a non-array leaf in the call args (a Python scalar threaded as an arg, a `None`)
    is rejected HERE, where it is a loud standup abort, not a per-call surprise. Downstream the phantom
    type guarantees the contract; the robust path re-validates avals inside jax on every call (XLA
    raises loudly on a shape/dtype mismatch), the unsafe path does not (its assumptions are documented
    on `run`).

    The UNSAFE path is built iff it is constructible: the call args must flatten to exactly ONE
    varying leaf (the params are static, `x` is one array leaf) and the executable must expose the
    pre-flattened `unsafe_call`. When `x` is itself a pytree of several arrays the unsafe splice is not
    well-defined, so unsafe is left unavailable (the robust path always works) — a deliberate
    narrowing, not a silent fallback (`prefer_unsafe=True` with an unbuildable unsafe path raises, so
    a caller asking for unsafe is never silently downgraded)."""
    if not callable(fn):
        raise TypeError(f"compile_lowlatency: fn must be callable, got {type(fn).__name__}")

    # Validate the contract on the ORIGINAL args, BEFORE device_put (translate-and-validate at the
    # boundary, ADR-0002 / P2 — and the root-cause placement, P5: `jax.device_put` silently PROMOTES a
    # python scalar leaf to a 0-d array, so a post-device_put array-ness check is dead code that can
    # never fire; checking the original leaves is where the contract "every call-arg leaf is an array"
    # is actually enforceable). A leaf that is neither a numpy nor a jax array — a python scalar / a
    # plain object threaded as a call arg — is rejected HERE, loudly, not coerced into a 0-d array whose
    # shape no caller intended.
    import numpy as _np
    orig_leaves = _tu.tree_leaves((params, example_x))
    for i, leaf in enumerate(orig_leaves):
        if not (isinstance(leaf, (_np.ndarray, jax.Array)) or (hasattr(leaf, "shape") and hasattr(leaf, "dtype"))):
            raise TypeError(
                f"compile_lowlatency: call-arg leaf {i} is {type(leaf).__name__}, not an array. Every "
                f"leaf of (params, x) must be a jax/numpy array — a Python scalar or object threaded as "
                f"a call arg is not supported (ADR-0002, refusing to coerce it into a 0-d array).")

    # Place the example args on-device so the staged params (and the unsafe static leaves) are real
    # device buffers, and the compiled signature matches device arrays (P9: construction is the effect).
    # device_put itself fail-louds on a leaf jax cannot interpret (a string/object) — wrap that into a
    # clear construction abort rather than letting a bare jax TypeError surface from inside the ctor.
    try:
        params_dev = jax.device_put(params)
        x_dev = jax.device_put(example_x)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"compile_lowlatency: example args carry a leaf jax cannot place on-device "
            f"({type(exc).__name__}: {exc}) — every leaf must be a jax/numpy array (ADR-0002).") from exc

    # AOT lower+compile for THIS exact (params, x) signature — the idiomatic jax 0.10.1 path. Donation
    # is on argnum 1 (x) when requested; argnum 0 (params) is never donated (the weights persist across
    # calls). One executable per (shape, dtype) signature; the handle holds exactly one.
    donate_argnums: tuple[int, ...] = (1,) if donate_x else ()
    jitted = jax.jit(fn, donate_argnums=donate_argnums)
    try:
        compiled = jitted.lower(params_dev, x_dev).compile()
    except Exception as exc:  # a lowering/compile failure is a loud construction abort (ADR-0002)
        raise ValueError(
            f"compile_lowlatency: jax could not lower+compile fn for example shapes "
            f"params={_describe(params_dev)} x={_describe(x_dev)} — {type(exc).__name__}: {exc}"
        ) from exc

    # Read the staged metadata OFF the compiled executable (P1 — derive, don't re-type). `in_avals` is
    # the nested input pytree of avals `(({params}, x), {})` and `out_info` the output pytree of
    # ShapeDtypeStructs; flatten each to its array leaves (the same leaf order the call args flatten to)
    # and reduce to plain (shape, dtype). `AvalSpec.of` reads `.shape`/`.dtype`, which both a
    # `ShapedArray` (inputs) and a `ShapeDtypeStruct` (outputs) carry.
    in_avals = tuple(AvalSpec.of(a) for a in _tu.tree_leaves(compiled.in_avals))
    out_leaves = _tu.tree_leaves(compiled.out_info)
    out_avals = tuple(AvalSpec.of(a) for a in out_leaves)
    single_output = len(out_leaves) == 1

    # The canonical flattened leaf order jit uses (params leaves first, then x) — the device leaves the
    # unsafe splice indexes into. Array-ness was validated on the original args above (the device leaves
    # are all arrays by construction now).
    flat_leaves, _in_treedef = _flatten_call(params_dev, x_dev)

    # Build the unsafe plumbing IFF it is well-defined: exactly one varying leaf (x is a single array)
    # and the executable exposes the pre-flattened entry. The varying leaf is the LAST leaf (params
    # flattens first, then x) when x is a single array; if x is a pytree of arrays, the splice index is
    # ambiguous, so unsafe is left unavailable.
    x_leaves, _x_tree = _tu.tree_flatten(x_dev)
    unsafe_call: Callable[..., Any] | None = None
    static_leaves: tuple[Any, ...] = ()
    x_leaf_index = -1
    ex = getattr(compiled, "_executable", None)
    raw_unsafe = getattr(ex, "unsafe_call", None) if ex is not None else None
    if raw_unsafe is not None and len(x_leaves) == 1:
        # static leaves = all leaves EXCEPT the single x leaf; x sits last in canonical order.
        x_leaf_index = len(flat_leaves) - 1
        static_leaves = tuple(flat_leaves[:x_leaf_index])
        unsafe_call = raw_unsafe

    if prefer_unsafe and unsafe_call is None:
        # A caller that explicitly asked for the unsafe path must not be silently downgraded (ADR-0002):
        # either the executable lacks the pre-flattened entry (a jax version that removed it) or x is a
        # multi-array pytree (the splice is ill-defined). Name which, and refuse.
        why = ("x flattened to %d leaves (unsafe needs exactly 1)" % len(x_leaves)
               if len(x_leaves) != 1 else "the compiled executable exposes no pre-flattened unsafe_call")
        raise ValueError(
            f"compile_lowlatency(prefer_unsafe=True): the unsafe path is not constructible here ({why}). "
            f"Use prefer_unsafe=False for the robust AOT Compiled call, which always works.")

    return LowLatencyFn(
        _compiled=compiled,
        _params=params_dev,
        in_avals=in_avals,
        out_avals=out_avals,
        _single_output=single_output,
        _unsafe_call=unsafe_call,
        _static_leaves=static_leaves,
        _x_leaf_index=x_leaf_index,
        use_unsafe=bool(prefer_unsafe and unsafe_call is not None),
    )


def run(handle: "LowLatencyFn[In, Out]", x: Any, *, unsafe: bool = False) -> Out:
    """The single low-overhead call: `run(handle, x) -> y`, equal (allclose) to `jax.jit(fn)(params, x)`
    for the params staged at construction.

    PRIMARY (robust) PATH — `unsafe=False`: call the AOT `Compiled` executable directly. jax flattens
    `(params, x)` against the compiled `in_tree` and validates avals, then dispatches with no re-trace;
    a per-call shape/dtype mismatch raises loudly inside jax (ADR-0002 at the jax boundary). Returns the
    bare output array for a single-array fn (matching plain-jit ergonomics), else the output pytree.

    UNSAFE PATH — `unsafe=True` (only valid if the handle was built with the unsafe plumbing): hand the
    already-flattened, already-device-placed leaves straight to the loaded executable's pre-flattened
    entry (`_executable.unsafe_call`), skipping the per-call pytree flatten and the `Compiled.__call__`
    argument-validation wrapper. WHAT IT ASSUMES (documented, unchecked — the trade): (1) `x` has
    EXACTLY the staged shape and dtype — no per-call check is done, a mismatch is undefined behavior,
    not a loud error; (2) the staged params are unchanged (they are spliced from the construction-time
    device leaves); (3) if the handle was built `donate_x=True`, the input buffer may be consumed, so
    `x` must not be reused. It returns the executable's flat output list, unwrapped to match the robust
    path. The bench measures this path; on jax 0.10.1 CPU it is SLOWER than the robust/plain-jit path
    (the loaded-executable entry redoes per-call sharding work the cpp fast path elides), so it is OFF
    by default — kept for measurement and for backends/versions where it may win.

    `x` is the single varying argument; the params staged at construction are NOT re-passed."""
    if unsafe:
        uc = handle._unsafe_call
        if uc is None:
            # Asked for unsafe on a handle that has no unsafe plumbing — a loud refusal, never a silent
            # fall-through to robust (ADR-0002: the caller asked for a specific path; honor or refuse).
            raise ValueError(
                "run(unsafe=True): this handle was not built with the unsafe path (build it via "
                "compile_lowlatency(..., prefer_unsafe=True), or call run(handle, x) for the robust path).")
        # Place x on-device (a host numpy x is transferred; a device x is a no-op), splice it into the
        # canonical leaf order at the staged index, and call the pre-flattened executable entry.
        x_leaf = jax.device_put(x)
        leaves = (*handle._static_leaves, x_leaf)  # x is the last (single) leaf by construction
        out_flat = uc(*leaves)
        # The loaded executable returns a flat list of output leaves; unwrap to match the robust path.
        # The jax output IS the function's declared `Out` (a backend-`Any` jax value asserted to the
        # phantom contract — P8: a cast documents the assertion in-source, not a silencing ignore).
        if handle._single_output:
            return cast("Out", _single(out_flat))
        return cast("Out", out_flat)

    # ROBUST: the AOT Compiled call. `Compiled.__call__` expects the SAME positional args it was lowered
    # with — `(params, x)` — and flattens+validates them against the compiled `in_tree`/avals, then
    # dispatches with no re-trace. We re-pass the handle's staged DEVICE params (held since construction;
    # device_put-on-device is identity, so no re-transfer) and the per-call `x`. A shape/dtype mismatch
    # on `x` raises loudly inside jax (ADR-0002 at the jax boundary).
    out = handle._compiled(handle._params, x)
    return cast("Out", out)  # the jax output asserted to the phantom `Out` contract (P8)


def _single(out_flat: Any) -> Any:
    """Unwrap a single-output executable's flat result. `unsafe_call` returns a list of output leaves;
    a single-array fn yields a length-1 list, so return its one element to match `jit(fn)(x)`."""
    if isinstance(out_flat, (list, tuple)):
        if len(out_flat) != 1:
            raise ValueError(
                f"low-latency unsafe call: single-output fn returned {len(out_flat)} leaves, expected 1 "
                f"(the handle's _single_output flag disagrees with the executable — a construction bug).")
        return out_flat[0]
    return out_flat


def _describe(x: Any) -> str:
    """A compact shape/dtype description of a pytree of arrays for an error message (the flat leaves'
    `(shape, dtype)` list) — used only on the construction-failure path, so it is allowed to be slow."""
    try:
        leaves = _tu.tree_leaves(x)
        return "[" + ", ".join(
            f"{tuple(getattr(l, 'shape', ()))}:{getattr(l, 'dtype', type(l).__name__)}" for l in leaves
        ) + "]"
    except Exception:  # pragma: no cover - error-path formatting must never itself raise
        return repr(type(x).__name__)
