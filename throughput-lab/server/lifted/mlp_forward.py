#!/usr/bin/env python3
"""
throughput-lab/server/lifted/mlp_forward.py — the JAX-jit inference wrapper over `forward_core`, a
condensed copy of chocofarm/az/mlp_jax.py's `MlpJaxForward` / `_forward_both`. Part of the lifted
trusted core (the MLP forward + its phantom-typed jax/numpy ACL). It is the SERVER's compute: the
decoupled compute stage (server/server.py) holds one of these and calls `pack(X)` then
`forward_batch(packed)` to turn a drained (N_total, in_dim) request matrix into (N_total,)
de-standardized values + (N_total, n_actions) raw logits. `pack` narrows the arbitrary gathered row
count to a WARMED shape (a PaddedBatch) and `forward_batch` accepts only that — so the jit never
recompiles for an off-ladder shape (the bug that motivated the type; see PaddedBatch / ADR-0012).

PROVENANCE: condensed from chocofarm/az/mlp_jax.py @ d30fe8e. The forward graph is the lifted
`forward.forward_core` (the SAME body the numpy net and the jax trainer run). The wrapper:
  * holds the net weights as a device-array params dict at the pinned DTYPE (float32 default);
  * runs the jitted forward (XLA caches one compiled kernel per (shape, dtype) — constant across the
    run, so it does not recompile when weights change);
  * DE-STANDARDIZES the value on-device (v = v_std*y_std + y_mean) and returns RAW logits.

DELIBERATE DIVERGENCE FROM THE PARENT (documented, ADR-0002 honesty): the parent's `_forward_both`
applies a legal-mask + softmax on-device and returns the policy PROBABILITIES `p`. This testbed
returns the RAW logits and takes NO legal mask, because the testbed's wire response contract
(server/wire.py) carries raw logits and keeps masking client-side. So this wrapper drops the parent's
mask/softmax tail and the `lm` argument; everything upstream (the trunk + residual + value
de-standardization) is the same compute, so the throughput is the throughput of the same matmuls.
The parent does NOT lift cleanly with its softmax tail intact — that tail is the divergence, and it
is named here rather than silently transplanted.

The lab does not need a trained net: throughput is a property of the matmul SHAPES (in_dim=241,
hidden=256, n_actions), not the weights. `MlpForward.random_net(...)` builds a random net of the
configured geometry; the explicit `__init__(params, y_mean, y_std)` stays the typed seam the server
build agent constructs against (ADR-0012 — params is the SSOT for which heads exist: a residual block
iff "Wr1" is present, a policy head iff "Wp" is present).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import bisect
import time
from dataclasses import dataclass
from typing import Any

# The XLA/OMP single-thread pin is a chocofarm-internal config the testbed does NOT lift (it is not
# part of the forward graph). JAX is left at its environment default here; if the maintainer wants the
# parent's single-thread pin, set XLA_FLAGS / OMP_NUM_THREADS in the launch environment.
import deal
import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp

from server.lifted.dtypes import DTYPE
from server.lifted.forward import forward_core

# The jax float dtype mirroring the pinned numpy DTYPE (float32 default). One pin, derived — never a
# second float-precision literal that could disagree with dtypes.DTYPE (ADR-0012 P1, derived dims).
_JDTYPE = jnp.float32 if np.dtype(DTYPE) == np.dtype(np.float32) else jnp.float64


@dataclass(frozen=True)
class PaddedBatch:
    """A feature matrix whose row count is, BY CONSTRUCTION, a WARMED XLA shape — so it can be fed to
    the jitted forward without ever triggering a per-shape recompile in the timed window.

    Making the illegal state UNREPRESENTABLE (ADR-0012: the typed signature is the contract). The jit
    compiles one kernel per (rows, in_dim, dtype) shape, but the server's gather produces an arbitrary
    row count; feeding that raw count to the jit pays a ~50ms recompile mis-read as compute. The fix is
    not a check sprinkled at the call site — it is a TYPE: `MlpForward.forward_batch` accepts ONLY a
    PaddedBatch, and the ONLY sanctioned way to obtain one is `MlpForward.pack` (which rounds the real
    row count up to a warmed bucket on the forward's ladder and zero-pads). So a raw arbitrary row count
    simply cannot reach the recompiling jit — there is no path. A shape off the ladder is a loud reject
    at `pack` (ADR-0002), never a silent recompile.

    `x` is the (bucket, in_dim) padded matrix (bucket rows == a warmed shape); `n_real` is the count of
    real leading rows — the forward returns exactly `result[:n_real]`, the pad tail [n_real:] being
    forward-computed but meaningless. Construct ONLY via `MlpForward.pack`; building one by hand off the
    ladder defeats the invariant the type exists to carry (and `forward_batch` backstop-rejects it)."""
    x: "npt.NDArray[np.float32]"
    n_real: int


@jax.jit
def _forward_packed(params: dict[str, Any], x: Any, ym: Any, ys: Any) -> Any:
    """De-standardized value + RAW logits PACKED into ONE `(B, 1+n_actions)` device array (column 0 the
    value, columns 1.. the raw logits; `(B, 1)` value-only) over the ONE `forward.forward_core` (audit R11).

    TWO consolidations vs the old `_forward_both` two-output tuple (the in-serve A/B @6deb40c attributed
    ~13–17% of the forward envelope to them, mirroring the production `jit_forward_core`):
      * the host->device CAST is FOLDED IN — `x` is passed as the HOST numpy batch and cast INSIDE the jit
        (the traced-arg convert), eliminating the eager `jnp.asarray(x)` the caller used to do as a SEPARATE
        XLA dispatch before the forward;
      * the value + logits are `concatenate`d into ONE array so the caller makes ONE device->host pull, not
        two (`np.asarray(v)` then `np.asarray(logits)` were two device syncs).
    ADR-0012 P6: a numerically-equivalent reordering of the SAME `forward_core` (the wire-parity bar holds,
    not byte identity); P1/P7: still the one `forward_core`, only packed. `params` is the flat weight dict
    keyed like `ValueMLP._params()` (residual iff `"Wr1"`, policy iff `"Wp"`), passed as an ARGUMENT (not a
    closure) so XLA caches one kernel per (pytree, shape, dtype) and does not recompile when weights change.
    NO legal mask, NO softmax — the testbed wire carries raw logits, masking client-side (module docstring)."""
    v_std, logits = forward_core(params, x, jnp)
    v = jnp.reshape(v_std, (-1, 1)) * ys + ym
    return v if logits is None else jnp.concatenate([v, logits], axis=1)


def _random_params(in_dim: int, hidden: int, n_actions: int, residual: bool, seed: int) -> dict[str, Any]:
    """Build a RANDOM net's flat param dict of the live geometry (the ONE home for both the JAX and numpy
    forwards' random_net — ADR-0012 P1, never two param-builders that could drift). Glorot-ish small normals
    keep the forward finite under float32; the testbed reads only the SHAPES, never the values."""
    rng = np.random.default_rng(seed)

    def w(rows: int, cols: int) -> npt.NDArray[np.floating]:
        return (rng.standard_normal((rows, cols)) / np.sqrt(rows)).astype(DTYPE)

    def b(n: int) -> npt.NDArray[np.floating]:
        return np.zeros(n, dtype=DTYPE)

    params: dict[str, Any] = {"W1": w(in_dim, hidden), "b1": b(hidden),
                              "W2": w(hidden, hidden), "b2": b(hidden)}
    if residual:
        params.update({"Wr1": w(hidden, hidden), "br1": b(hidden),
                       "Wr2": w(hidden, hidden), "br2": b(hidden)})
    params["Wv"] = w(hidden, 1)
    params["bv"] = b(1)
    if n_actions > 0:
        params["Wp"] = w(hidden, n_actions)
        params["bp"] = b(n_actions)
    return params


class NullForward:
    """A ZERO-COMPUTE forward: returns correctly-SHAPED zeros instantly (no matmul, no XLA, no numpy stack). NOT
    a real forward — a MEASUREMENT instrument that isolates the SERVE-LOOP ceiling. With compute ~free, the
    server's max forwards/s (and rows/s) reveals the drain+pack+scatter+poll+wire overhead ALONE — the
    non-compute floor per serve cycle. Mirrors the forward seam (random_net/warmup/pack/forward_batch) so the
    server holds it interchangeably and the coalesce/pack path is exercised identically; only the matmul is
    nulled. `--forward null` is a probe, never shipped. No jax/numpy-forward import — it holds only the geometry
    the scatter needs to shape its zeros (the n_actions width)."""

    def __init__(self, params: dict[str, Any], y_mean: float, y_std: float) -> None:
        self._ym: float = float(y_mean)
        self._ys: float = float(y_std)
        self.in_dim = int(params["W1"].shape[0])
        self.hidden = int(params["W1"].shape[1])
        self.has_policy = "Wp" in params
        self.n_actions = int(params["Wp"].shape[1]) if self.has_policy else 0
        self.has_residual = "Wr1" in params
        self._warmed_ladder: tuple[int, ...] = ()
        self._warmed_set: frozenset[int] = frozenset()

    @classmethod
    def random_net(cls, *, in_dim: int = 241, hidden: int = 256, n_actions: int = 0,
                   residual: bool = False, seed: int = 0) -> "NullForward":
        return cls(_random_params(in_dim, hidden, n_actions, residual, seed), y_mean=0.0, y_std=1.0)

    def warmup(self, batch_sizes: "list[int]", in_dim: int) -> None:
        """No compile to warm (there is no kernel) — record the ladder so `pack` and the READY line behave like
        a real forward, and validate the geometry (uniform with the other forwards' warmup contract)."""
        if in_dim != self.in_dim:
            raise ValueError(f"warmup in_dim {in_dim} != net in_dim {self.in_dim} (geometry mismatch)")
        self._warmed_ladder = tuple(sorted(set(batch_sizes)))
        self._warmed_set = frozenset(self._warmed_ladder)

    @property
    def warmed_sizes(self) -> "list[int]":
        return list(self._warmed_ladder)

    def pack(self, X: "npt.NDArray[np.floating]") -> PaddedBatch:
        """Identity pack (the serve loop's gather already produced the rows; a null forward needs no warmed
        shape) — return the real rows wrapped in PaddedBatch (n_real == row count, no pad tail)."""
        a = np.ascontiguousarray(X, dtype=DTYPE)
        if a.ndim != 2 or a.shape[1] != self.in_dim:
            raise ValueError(f"pack expects (n, {self.in_dim}), got shape {a.shape}")
        if a.shape[0] == 0:
            raise ValueError("pack got an empty (0-row) matrix")
        return PaddedBatch(x=a, n_real=int(a.shape[0]))

    def forward_batch(self, batch: PaddedBatch) -> (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]):
        """Return zeros of the real shape — NO compute. (The scatter encodes whatever it gets; the throughput
        number is the point, not the values.)"""
        n = batch.n_real
        values = np.zeros(n, dtype=np.float32)
        logits = np.zeros((n, self.n_actions), dtype=np.float32) if self.has_policy else None
        return values, logits

    def forward_batch_timed(self, batch: PaddedBatch) -> (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None, tuple[float, float, float]]):
        """Null compute timed at ~0 (all in the 'jit' slot for the uniform (h2d, jit, d2h) contract)."""
        t0 = time.monotonic()
        v, l = self.forward_batch(batch)
        return v, l, (0.0, time.monotonic() - t0, 0.0)


class NumpyMlpForward:
    """A pure-NUMPY forward of `forward_core` (xp=np) — the SAME graph MlpForward runs on JAX, but with NO XLA
    dispatch and NO bucket/pad (numpy forwards the exact gathered row count directly). The A/B arm testing
    whether XLA-CPU's per-call dispatch overhead (measured ~1.35 ms in-serve for a ~20 M-flop MLP — pure
    launch latency, not arithmetic) is the serve-forward-envelope bottleneck. Mirrors MlpForward's
    construction/`pack`/`forward_batch` seam so the server (server.py) can hold either one interchangeably
    (cfg.forward_impl). `pack` is the identity (numpy needs no warmed shape); `warmup` only records the
    ladder for the READY line; `forward_batch` runs forward_core over the REAL rows — no pad tax at all."""

    def __init__(self, params: dict[str, Any], y_mean: float, y_std: float) -> None:
        self.params: dict[str, Any] = {k: np.asarray(v, dtype=DTYPE) for k, v in params.items()}
        self.ym = np.asarray(y_mean, dtype=DTYPE)
        self.ys = np.asarray(y_std, dtype=DTYPE)
        self.in_dim = int(params["W1"].shape[0])
        self.hidden = int(params["W1"].shape[1])
        self.has_policy = "Wp" in params
        self.n_actions = int(params["Wp"].shape[1]) if self.has_policy else 0
        self.has_residual = "Wr1" in params
        self._warmed_ladder: tuple[int, ...] = ()
        self._warmed_set: frozenset[int] = frozenset()

    @classmethod
    def random_net(cls, *, in_dim: int = 241, hidden: int = 256, n_actions: int = 0,
                   residual: bool = False, seed: int = 0) -> "NumpyMlpForward":
        return cls(_random_params(in_dim, hidden, n_actions, residual, seed), y_mean=0.0, y_std=1.0)

    def warmup(self, batch_sizes: "list[int]", in_dim: int) -> None:
        """No compile to warm (numpy has no per-shape kernel) — just record the ladder for the READY line and
        validate the geometry, so the server's warmup call is uniform across forward impls."""
        if in_dim != self.in_dim:
            raise ValueError(f"warmup in_dim {in_dim} != net in_dim {self.in_dim} (geometry mismatch)")
        self._warmed_ladder = tuple(sorted(set(batch_sizes)))
        self._warmed_set = frozenset(self._warmed_ladder)

    @property
    def warmed_sizes(self) -> "list[int]":
        return list(self._warmed_ladder)

    def pack(self, X: "npt.NDArray[np.floating]") -> PaddedBatch:
        """Identity-pack: numpy forwards any row count, so there is NO bucket/pad — return the REAL rows
        (wrapped in PaddedBatch for the server's uniform seam; n_real == the row count, no pad tail)."""
        a = np.ascontiguousarray(X, dtype=DTYPE)
        return PaddedBatch(x=a, n_real=int(a.shape[0]))

    def forward_batch(self, batch: PaddedBatch) -> (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]):
        """forward_core in numpy over the real rows (no pad). Returns (de-standardized values, raw logits)."""
        v_std, logits = forward_core(self.params, batch.x, np)
        v = (v_std * self.ys + self.ym).astype(np.float32)
        logits_np = logits.astype(np.float32) if logits is not None else None
        return v, logits_np

    def forward_batch_timed(self, batch: PaddedBatch) -> (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None, tuple[float, float, float]]):
        """Phase-split shim for cfg.profile_forward: numpy has no host<->device transfer, so it is all
        'compute' (h2d=0, d2h=0) — the matmul stack timed as one phase."""
        t0 = time.monotonic()
        v, logits_np = self.forward_batch(batch)
        return v, logits_np, (0.0, time.monotonic() - t0, 0.0)


class MlpForward:
    """JAX-jit inference wrapper over `forward_core` — the server's compute stage.

    The constructor takes the params dict + the de-standardization scalars (y_mean, y_std); `params`
    is the SSOT for which heads exist (residual iff "Wr1", policy iff "Wp"). The forward returns
    de-standardized values + RAW logits (the testbed wire response contract). For the testbed a random
    net of the live shapes (in_dim=241, hidden=256) is the intended default — use `MlpForward.random_net`
    — because throughput is a property of the matmul shapes, not the trained weights."""

    def __init__(self, params: dict[str, Any], y_mean: float, y_std: float) -> None:
        # Hold the weights as a device-array params dict at the pinned precision; donate the host
        # arrays to jax.device_put so the kernel reads device memory, not a per-call host->device copy.
        self.params: dict[str, Any] = {k: jnp.asarray(v, dtype=_JDTYPE) for k, v in params.items()}
        self.ym: Any = jnp.asarray(y_mean, dtype=_JDTYPE)
        self.ys: Any = jnp.asarray(y_std, dtype=_JDTYPE)
        # Surface the geometry the params imply (read-only facts, for the server's instrumentation).
        self.in_dim: int = int(params["W1"].shape[0])
        self.hidden: int = int(params["W1"].shape[1])
        self.has_policy: bool = "Wp" in params
        self.n_actions: int = int(params["Wp"].shape[1]) if self.has_policy else 0
        self.has_residual: bool = "Wr1" in params
        # The warmed bucket ladder — the SINGLE source of truth for which forward shapes are compiled.
        # Empty until warmup(); `pack` rounds into it (bisect needs it sorted) and `forward_batch`
        # backstop-checks membership (the frozenset is the O(1) view of the same fact). The forward owns
        # this — the server does not keep a second copy that could drift (ADR-0012 P1).
        self._warmed_ladder: tuple[int, ...] = ()
        self._warmed_set: frozenset[int] = frozenset()

    # -- construction ---------------------------------------------------------------------------------

    @classmethod
    def random_net(cls, *, in_dim: int = 241, hidden: int = 256, n_actions: int = 0,
                   residual: bool = False, seed: int = 0) -> "MlpForward":
        """Build a RANDOM net of the live Stage-A geometry. The weights are throwaway (Glorot-ish
        small-scale normals — they keep the forward numerically finite; the testbed never reads the
        VALUES, only times the SHAPES). `n_actions == 0` omits the policy head (value-only, "Wp"
        absent); `residual=True` adds the keyed residual block ("Wr1"..). y_mean/y_std are the
        identity de-standardization (0, 1) — the value is meaningless in the lab, the matmul is not."""
        return cls(_random_params(in_dim, hidden, n_actions, residual, seed), y_mean=0.0, y_std=1.0)

    @classmethod
    def from_npz(cls, path: str) -> "MlpForward":
        """Load a REAL (trained) net from an AZ checkpoint `.npz` (consolidation Gate B — give throughput-lab
        direct AZ-loop relevance: measure on the real net a worker serves, not a throwaway random one). The
        archive is `ValueMLP.save`'s layout: the flat weight keys (`W1,b1,W2,b2[,Wr1..],Wv,bv[,Wp,bp]`) the
        SAME `forward_core` consumes (so it slots straight into __init__, no transcription — ADR-0012 P1/P6),
        plus `_yscale = [y_mean, y_std]` (the TRAINED de-standardization, unlike random_net's identity (0,1))
        and `_meta` (geometry, derived from the weights themselves; not re-stated here). Underscore-prefixed
        keys are metadata, not weights. The geometry the producer's features must match (in_dim, n_actions)
        is validated downstream by warmup()'s in_dim check + the wire response width (ADR-0002, fail loud)."""
        z = np.load(path, allow_pickle=False)
        if "W1" not in z.files:
            raise ValueError(f"from_npz({path}): no 'W1' weight — not a ValueMLP checkpoint (ADR-0002)")
        params = {k: z[k] for k in z.files if not k.startswith("_")}
        if "_yscale" in z.files:
            ym, ys = float(z["_yscale"][0]), float(z["_yscale"][1])
        else:
            ym, ys = 0.0, 1.0   # a checkpoint without the de-standardization scalars => identity (loud-safe)
        return cls(params, y_mean=ym, y_std=ys)

    # -- compute --------------------------------------------------------------------------------------

    def warmup(self, batch_sizes: "list[int]", in_dim: int) -> None:
        """Pre-compile the XLA kernel for each batch size B the drain can produce, BEFORE the timed
        run (ADR-0009 — mirror chocofarm InferenceServer.warmup; a cold per-B JIT compile inside the
        timed window is mis-read as throughput jitter). XLA caches one kernel per (B, in_dim, dtype),
        so a forward over a zero matrix of each shape compiles + caches it. `block_until_ready` forces
        the async dispatch to actually finish the compile before we call the warmup done."""
        if in_dim != self.in_dim:
            raise ValueError(f"warmup in_dim {in_dim} != net in_dim {self.in_dim} (geometry mismatch)")
        for bsz in batch_sizes:
            if bsz <= 0:
                raise ValueError(f"warmup batch size must be >= 1, got {bsz}")
            # Warm with a HOST numpy zero matrix — the SAME input type the serving path passes (so the jit
            # caches the executable for the host-array signature; warming with a jnp array would compile a
            # DIFFERENT traced type and force a recompile on the first real, host-fed forward).
            x = np.zeros((bsz, in_dim), dtype=np.float32)
            _forward_packed(self.params, x, self.ym, self.ys).block_until_ready()
        # Record the warmed ladder as THIS forward's contract: from here on `pack` may only round into
        # these shapes and `forward_batch` accepts only a PaddedBatch built off them (ADR-0012 — the
        # warmed set is the single source of truth, derived once, never re-stated where it could drift).
        self._warmed_ladder = tuple(sorted(set(batch_sizes)))
        self._warmed_set = frozenset(self._warmed_ladder)

    @property
    def warmed_sizes(self) -> "list[int]":
        """The warmed bucket ladder (the shapes `pack` rounds into). Read-only; the server reports it on
        its READY line rather than keeping its own copy (this forward is the source of truth)."""
        return list(self._warmed_ladder)

    @deal.pre(lambda self, X: getattr(X, "ndim", 0) == 2 and int(X.shape[0]) >= 1
              and int(X.shape[1]) == self.in_dim
              and (not self._warmed_ladder or int(X.shape[0]) <= self._warmed_ladder[-1]))
    @deal.ensure(lambda self, X, result: result.n_real == int(X.shape[0])
                 and int(result.x.shape[0]) in self._warmed_set)
    def pack(self, X: "npt.NDArray[np.floating]") -> PaddedBatch:
        """Narrow an arbitrary (n, in_dim) feature matrix to a WARMED shape: round n up to the next
        bucket on this forward's ladder and zero-pad to it, returning a PaddedBatch (the only sanctioned
        constructor). This is the Port/ACL at the gather->forward seam — parse-and-narrow a raw row count
        into the closed warmed set, never pass the raw count to the jit. A matrix with MORE rows than the
        largest warmed bucket, a non-2-D matrix, an in_dim mismatch, or a 0-row matrix is a loud failure
        (ADR-0002), never a silent recompile. Requires warmup() first (else there is no ladder to pack
        into).

        The `deal` contract is the machine-checkable SPEC of pack's totality: the @pre is the caller's
        obligation (a non-empty 2-D in_dim-wide matrix no taller than the top warmed bucket — exactly
        what a wire.BoundedBatch under the gather's row cap guarantees), and the @ensure is pack's own
        guarantee (it returns a warmed bucket that covers the real rows). The property suite discharges
        both adversarially; on the serving hot-path `deal.disable()` strips them to nothing, leaving the
        body's own loud checks (which stay — they are the defense-in-depth guard, not the spec)."""
        if not self._warmed_ladder:
            raise RuntimeError("pack() before warmup(): no warmed buckets to pack into")
        a = np.ascontiguousarray(X)
        if a.ndim != 2:
            raise ValueError(f"pack expects a 2-D (n, in_dim) matrix, got shape {a.shape}")
        n_real, in_dim = a.shape
        if in_dim != self.in_dim:
            raise ValueError(f"pack in_dim {in_dim} != net in_dim {self.in_dim} (geometry mismatch)")
        if n_real == 0:
            raise ValueError("pack got an empty (0-row) matrix")
        idx = bisect.bisect_left(self._warmed_ladder, n_real)
        if idx == len(self._warmed_ladder):
            raise ValueError(
                f"batch of {n_real} rows exceeds the largest warmed bucket {self._warmed_ladder[-1]} — "
                f"no covering kernel (warm a larger bucket / raise the gather's row cap)")
        bucket = self._warmed_ladder[idx]
        if bucket == n_real:
            return PaddedBatch(x=a, n_real=n_real)
        pad = np.zeros((bucket - n_real, in_dim), dtype=a.dtype)
        return PaddedBatch(x=np.concatenate([a, pad], axis=0), n_real=n_real)

    def forward_batch(self, batch: PaddedBatch) -> (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]):
        """Run ONE forward over a PaddedBatch — a (bucket, in_dim) matrix whose row count is GUARANTEED
        warmed (obtain it via `pack`; this method accepts NOTHING else, which is the contract that makes
        an unwarmed forward shape unrepresentable — see PaddedBatch). Returns (values: (n_real,)
        DE-STANDARDIZED float32, logits: (n_real, n_actions) RAW float32 or None for the value-only net)
        with the pad tail already sliced off. This is the testbed's compute hotspot: one XLA-fused matmul
        stack, the SAME compute the production server runs, and always a warmed kernel (ADR-0009)."""
        if not isinstance(batch, PaddedBatch):
            raise TypeError(
                f"forward_batch accepts only a PaddedBatch (build it with .pack); got {type(batch).__name__}")
        rows = int(batch.x.shape[0])
        # Backstop the type's invariant (defense in depth): a PaddedBatch hand-built off the ladder would
        # silently recompile, so a row count that is not a warmed bucket fails LOUD here (ADR-0002)
        # instead of paying a hidden compile — the contract holds even if pack() is bypassed.
        if rows not in self._warmed_set:
            raise ValueError(
                f"PaddedBatch row count {rows} is not a warmed bucket {self._warmed_ladder} "
                f"(a PaddedBatch must come from .pack)")
        # ONE forward → ONE device→host pull of the packed `(rows, 1+n_actions)` block (the cast folded
        # inside the jit — batch.x is passed HOST, no eager jnp.asarray). np.asarray blocks until ready, so
        # the returned arrays are materialized. Slice off the pad tail and split column 0 (value) from
        # columns 1.. (raw logits); a value-only net returns a single column → logits None.
        out = np.asarray(_forward_packed(self.params, batch.x, self.ym, self.ys), dtype=np.float32)
        n = batch.n_real
        values = out[:n, 0]
        logits_np = out[:n, 1:] if out.shape[1] > 1 else None
        return values, logits_np

    def forward_batch_timed(self, batch: PaddedBatch) -> (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None, tuple[float, float, float]]):
        """forward_batch with per-phase timers (h2d | jit | d2h) for the serve-loop forward-envelope
        profiling (server cfg.profile_forward). It `block_until_ready()`s BETWEEN phases to attribute each —
        which SERIALISES the otherwise-async XLA pipeline, so the returned sum is an UPPER BOUND on the real
        (pipelined) forward_batch wall. Use it for the RELATIVE split (where the time goes), never as the
        absolute forward cost. Same contract/returns as forward_batch, plus the (h2d_s, jit_s, d2h_s) tuple."""
        if not isinstance(batch, PaddedBatch) or int(batch.x.shape[0]) not in self._warmed_set:
            raise ValueError("forward_batch_timed: build the batch with .pack (warmed PaddedBatch required)")
        # The cast is now FOLDED into the jit (no separate eager h2d to time) — so the split is (0, jit, d2h):
        # `jit` is cast+forward fused (block_until_ready forces it) and `d2h` is the one pull. Still SERIALISED
        # (block between the compute and the pull), so the sum is an UPPER BOUND on the pipelined wall.
        t0 = time.monotonic()
        out = _forward_packed(self.params, batch.x, self.ym, self.ys)
        out.block_until_ready()                                # force the fused cast+compute to complete
        t1 = time.monotonic()
        arr = np.asarray(out, dtype=np.float32)
        n = batch.n_real
        values = arr[:n, 0]
        logits_np = arr[:n, 1:] if arr.shape[1] > 1 else None
        t2 = time.monotonic()                                  # device->host pull + slice
        return values, logits_np, (0.0, t1 - t0, t2 - t1)


# ---- DIAGNOSTIC cross-boundary arms (measurement only — ADR-0013 verify-the-artifact) -----------------
# `ProdMlpForward`/`StagedMlpForward` run the REAL production forwards (chocofarm.az.inference_server) over
# the SAME batches the tlab server serves, so an in-serve A/B attributes the tlab/overcommit forward-envelope
# gap to the actual production forward (its [v|logits] ONE-pull packing, device-resident staging) rather than
# to a guess. They DELIBERATELY cross the clean-room boundary (import chocofarm) and ONLY for this attribution
# — never a shipped forward. The plain-jax MlpForward above stays the testbed's own clean-room compute. The
# import is LAZY (in warmup/_prod_fn) so server's module import stays chocofarm-free until an arm is selected.

class ProdMlpForward(MlpForward):
    """The REAL production UN-STAGED forward `jit_forward_core` (the `forward_core` jitted to pack
    `[v | logits]` into ONE device array → ONE device→host pull) run over MlpForward's bucket-ladder pack.
    Holds params HOST-resident, so jit_forward_core re-transfers the weight dict per call (the ~45–53 µs the
    staged path removes — see StagedMlpForward). The A/B that isolates the production forward GRAPH + the
    one-pull pack from MlpForward's two-pull `(v, logits)` tuple, threading/ladder held identical."""

    def __init__(self, params: dict[str, Any], y_mean: float, y_std: float) -> None:
        # HOST numpy params (jit_forward_core re-transfers them each call — the un-staged production path).
        self.params = {k: np.asarray(v, dtype=DTYPE) for k, v in params.items()}
        self._ym: float = float(y_mean)
        self._ys: float = float(y_std)
        self.in_dim = int(params["W1"].shape[0])
        self.hidden = int(params["W1"].shape[1])
        self.has_policy = "Wp" in params
        self.n_actions = int(params["Wp"].shape[1]) if self.has_policy else 0
        self.has_residual = "Wr1" in params
        self._warmed_ladder: tuple[int, ...] = ()
        self._warmed_set: frozenset[int] = frozenset()
        self._jfc: Any = None   # the bound production jit_forward_core (lazy import)

    def _prod_fn(self) -> Any:
        if self._jfc is None:
            from chocofarm.az.inference_server import jit_forward_core  # diagnostic cross-boundary import
            self._jfc = jit_forward_core
        return self._jfc

    def warmup(self, batch_sizes: "list[int]", in_dim: int) -> None:
        """Compile + cache jit_forward_core for each bucket shape (it jits one executable per shape, exactly
        like MlpForward's warmup), then record the ladder. `np.asarray(...)` forces the async compile."""
        if in_dim != self.in_dim:
            raise ValueError(f"warmup in_dim {in_dim} != net in_dim {self.in_dim} (geometry mismatch)")
        jfc = self._prod_fn()
        for bsz in sorted(set(batch_sizes)):
            if bsz <= 0:
                raise ValueError(f"warmup batch size must be >= 1, got {bsz}")
            x = np.zeros((bsz, in_dim), dtype=np.float32)
            np.asarray(jfc(self.params, x, self._ym, self._ys))
        self._warmed_ladder = tuple(sorted(set(batch_sizes)))
        self._warmed_set = frozenset(self._warmed_ladder)

    def forward_batch(self, batch: PaddedBatch) -> (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]):
        """ONE jit_forward_core call → ONE device→host pull of the `(rows, 1+n_actions)` block (col 0 the
        de-standardized value, cols 1.. the raw logits), sliced to the real rows. (Inherits MlpForward.pack.)"""
        out = np.asarray(self._prod_fn()(self.params, batch.x, self._ym, self._ys), dtype=np.float32)
        n = batch.n_real
        values = out[:n, 0]
        logits = out[:n, 1:] if out.shape[1] > 1 else None
        return values, logits

    def forward_batch_timed(self, batch: PaddedBatch) -> (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None, tuple[float, float, float]]):
        """One-pull forward; the production forward fuses cast+compute+pull into one compiled call, so there
        is no host-visible h2d/d2h split to attribute — it is all 'jit' (h2d=0, d2h=0)."""
        t0 = time.monotonic()
        values, logits = self.forward_batch(batch)
        return values, logits, (0.0, time.monotonic() - t0, 0.0)


class StagedMlpForward:
    """The REAL production STAGED forward (build_staged_forward via the lowlatency AOT dispatcher) — the
    ACTUAL forward overcommit_sweep's 140k baseline runs. Params are staged DEVICE-RESIDENT once (a forward
    re-transfers only Xb), `[v | logits]` packed (ONE device→host pull), and the graph is AOT-compiled for
    ONE fixed (pad_to, in_dim) shape (pad_to = the largest warmed bucket = the server's max_batch). So `pack`
    pads EVERY batch to that one shape (the production pad-to-max), NOT the bucket ladder. `--forward staged`
    is the apples-to-apples target: if it runs lean in the tlab serve loop, the tlab/overcommit gap is our
    MlpForward wrapper (one-pull + staging), not the serve loop."""

    def __init__(self, params: dict[str, Any], y_mean: float, y_std: float) -> None:
        self._params = {k: np.asarray(v, dtype=DTYPE) for k, v in params.items()}
        self._ym: float = float(y_mean)
        self._ys: float = float(y_std)
        self.in_dim = int(params["W1"].shape[0])
        self.hidden = int(params["W1"].shape[1])
        self.has_policy = "Wp" in params
        self.n_actions = int(params["Wp"].shape[1]) if self.has_policy else 0
        self.has_residual = "Wr1" in params
        self._pad_to: int = 0
        self._fn: Any = None
        self._warmed_ladder: tuple[int, ...] = ()
        self._warmed_set: frozenset[int] = frozenset()

    @classmethod
    def random_net(cls, *, in_dim: int = 241, hidden: int = 256, n_actions: int = 0,
                   residual: bool = False, seed: int = 0) -> "StagedMlpForward":
        return cls(_random_params(in_dim, hidden, n_actions, residual, seed), y_mean=0.0, y_std=1.0)

    def warmup(self, batch_sizes: "list[int]", in_dim: int) -> None:
        """Build the staged AOT handle for the ONE fixed shape (pad_to = max warmed bucket) and force its
        cold compile. The staged forward has a SINGLE warmed shape — pack pads everything to it."""
        if in_dim != self.in_dim:
            raise ValueError(f"warmup in_dim {in_dim} != net in_dim {self.in_dim} (geometry mismatch)")
        self._pad_to = max(batch_sizes)
        from chocofarm.az.inference_server import build_staged_forward  # diagnostic cross-boundary import
        self._fn = build_staged_forward(self._params, self._ym, self._ys, self._pad_to)
        np.asarray(self._fn(self._params, np.zeros((self._pad_to, in_dim), dtype=np.float32),
                            self._ym, self._ys))   # force the AOT compile + param staging
        self._warmed_ladder = (self._pad_to,)
        self._warmed_set = frozenset(self._warmed_ladder)

    @property
    def warmed_sizes(self) -> "list[int]":
        return list(self._warmed_ladder)

    def pack(self, X: "npt.NDArray[np.floating]") -> PaddedBatch:
        """Pad to the ONE staged shape (pad_to) — production pad-to-max, not a bucket ladder. A row count
        over pad_to, a non-2-D matrix, an in_dim mismatch, or 0 rows is a loud failure (ADR-0002)."""
        if not self._pad_to:
            raise RuntimeError("pack() before warmup(): no staged shape to pad into")
        a = np.ascontiguousarray(X, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(f"pack expects a 2-D (n, in_dim) matrix, got shape {a.shape}")
        n_real, in_dim = a.shape
        if in_dim != self.in_dim:
            raise ValueError(f"pack in_dim {in_dim} != net in_dim {self.in_dim} (geometry mismatch)")
        if n_real == 0:
            raise ValueError("pack got an empty (0-row) matrix")
        if n_real > self._pad_to:
            raise ValueError(f"batch of {n_real} rows exceeds the staged shape {self._pad_to} (raise max_batch)")
        if n_real == self._pad_to:
            return PaddedBatch(x=a, n_real=n_real)
        pad = np.zeros((self._pad_to - n_real, in_dim), dtype=np.float32)
        return PaddedBatch(x=np.concatenate([a, pad], axis=0), n_real=n_real)

    def forward_batch(self, batch: PaddedBatch) -> (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]):
        """ONE staged call (device-resident params, only Xb transferred) → ONE device→host pull of the
        `(pad_to, 1+n_actions)` block, sliced to the real rows."""
        out = np.asarray(self._fn(self._params, batch.x, self._ym, self._ys), dtype=np.float32)
        n = batch.n_real
        values = out[:n, 0]
        logits = out[:n, 1:] if out.shape[1] > 1 else None
        return values, logits

    def forward_batch_timed(self, batch: PaddedBatch) -> (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None, tuple[float, float, float]]):
        """One-pull staged forward; no host-visible h2d/d2h split (all 'jit')."""
        t0 = time.monotonic()
        values, logits = self.forward_batch(batch)
        return values, logits, (0.0, time.monotonic() - t0, 0.0)
