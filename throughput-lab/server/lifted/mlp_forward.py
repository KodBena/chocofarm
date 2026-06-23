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
def _forward_both(params: dict[str, Any], x: Any, ym: Any, ys: Any) -> tuple[Any, Any | None]:
    """De-standardized value + RAW logits over the ONE `forward.forward_core` (audit R11). `params`
    is the flat weight dict keyed like `ValueMLP._params()`, so the residual block is applied iff
    `"Wr1"` is present and the policy head iff `"Wp"` is present — exactly the toggles `forward_core`
    keys on. Weights are passed as an ARGUMENT (not a closure) so XLA caches one compiled kernel per
    (pytree-structure, shape, dtype) and does not recompile when weights change.

    Returns `(v, logits)` where `v = v_std*ys + ym` is the de-standardized scalar value (shape (B,))
    and `logits` is the (B, n_actions) RAW (NOT softmaxed, NOT masked) policy logits, or `None` for
    the value-only net (`"Wp"` absent). NO legal mask, NO softmax — the testbed wire carries raw
    logits and keeps masking client-side (see the module docstring's deliberate-divergence note)."""
    v_std, logits = forward_core(params, x, jnp)
    v = v_std * ys + ym
    return v, logits


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
        rng = np.random.default_rng(seed)

        def w(rows: int, cols: int) -> npt.NDArray[np.floating]:
            # small-scale normal so a deep-ish ReLU stack stays finite under float32 (not a trained
            # init — just a finite one); std = 1/sqrt(fan_in) is the usual stabilising scale.
            return (rng.standard_normal((rows, cols)) / np.sqrt(rows)).astype(DTYPE)

        def b(n: int) -> npt.NDArray[np.floating]:
            return np.zeros(n, dtype=DTYPE)

        params: dict[str, Any] = {
            "W1": w(in_dim, hidden), "b1": b(hidden),
            "W2": w(hidden, hidden), "b2": b(hidden),
        }
        if residual:
            params.update({
                "Wr1": w(hidden, hidden), "br1": b(hidden),
                "Wr2": w(hidden, hidden), "br2": b(hidden),
            })
        params["Wv"] = w(hidden, 1)
        params["bv"] = b(1)
        if n_actions > 0:
            params["Wp"] = w(hidden, n_actions)
            params["bp"] = b(n_actions)
        return cls(params, y_mean=0.0, y_std=1.0)

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
            x = jnp.zeros((bsz, in_dim), dtype=_JDTYPE)
            v, logits = _forward_both(self.params, x, self.ym, self.ys)
            v.block_until_ready()
            if logits is not None:
                logits.block_until_ready()
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
        x = jnp.asarray(batch.x, dtype=_JDTYPE)
        v, logits = _forward_both(self.params, x, self.ym, self.ys)
        # Pull back to host numpy float32 (the wire encoder's dtype) and slice off the pad tail. np.asarray
        # on a jax array blocks until ready, so the returned arrays are materialized — no dangling handle.
        n = batch.n_real
        values = np.asarray(v, dtype=np.float32)[:n]
        logits_np = np.asarray(logits, dtype=np.float32)[:n] if logits is not None else None
        return values, logits_np
