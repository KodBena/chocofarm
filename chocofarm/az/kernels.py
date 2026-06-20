#!/usr/bin/env python3
"""
chocofarm AZ — numba kernels for the belief reductions (the per-decision hot-path #1).

The belief-derived feature block is, per leaf, a reduction over the world-set `bw` (up to 15,504
int64 world bitmasks at the root, ~100 at the median leaf): the per-treasure marginals and the
per-detector positive-read counts. The numpy form materialises a `(nb × N)` and a `(nb × nD)`
boolean/int matrix and reduces — bandwidth-bound and allocation-heavy. A single numba loop fuses
BOTH reductions into one pass over `bw` with no large temporary, ~12× faster across the whole
`|bw|` distribution (full-set, 1k, 100, 10 — see docs/results/az-jax-perf.md).

This is the lever the maintainer named ("if jax is no good for `_belief_feats`, numba probably
is"): jax does not help a reduction over int64 bitmasks, but a numba scalar loop does. The kernel
is integer arithmetic (bit tests + integer counts), so it is bit-exact with the numpy reduction —
the float32/float64 choice does not touch it; only the downstream division-to-rates does.

`marginals` is also exposed as a standalone kernel for `env.marginals` callers that don't want
the detector block (the shared-client fast path — see env.marginals).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True, fastmath=False)  # type: ignore[untyped-decorator]  # numba stub-gap: @njit has no py.typed stubs
def belief_marg_cover(
    bw: npt.NDArray[np.int64],
    cover: npt.NDArray[np.int64],
    N: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Fused belief reduction. `bw`: (nb,) int64 world bitmasks; `cover`: (nD,) int64 detector
    cover masks; `N`: treasure count. Returns (marg float64[N], cnt int64[nD]) where
    `marg[t]` = fraction of worlds with treasure t present, `cnt[d]` = number of worlds the
    detector d reads positive. One pass over `bw`, no (nb×*) temporary.

    `cnt` carries exactly what the numpy `count_nonzero((bw[:,None]&cover[None,:])!=0, axis=0)`
    produced; the caller derives `p_pos = cnt/nb`, `informative = (cnt>0)&(cnt<nb)` from it
    (identical to the numpy boolean any/~any). Integer-exact regardless of feature dtype."""
    nb = bw.shape[0]
    nD = cover.shape[0]
    marg = np.zeros(N, dtype=np.float64)
    cnt = np.zeros(nD, dtype=np.int64)
    for w in range(nb):
        wv = bw[w]
        for t in range(N):
            if (wv >> t) & 1:
                marg[t] += 1.0
        for d in range(nD):
            if (wv & cover[d]) != 0:
                cnt[d] += 1
    inv = 1.0 / nb
    for t in range(N):
        marg[t] *= inv
    return marg, cnt


@njit(cache=True, fastmath=False)  # type: ignore[untyped-decorator]  # numba stub-gap: @njit has no py.typed stubs
def marginals_kernel(
    bw: npt.NDArray[np.int64],
    N: int,
) -> npt.NDArray[np.float64]:
    """Per-treasure marginals only (the `env.marginals` hot path). `bw`: (nb,) int64; returns
    float64[N]. Single pass, no (nb×N) temporary. Bit-exact with the numpy bit-extract reduction.
    Caller handles the empty-belief case (nb==0)."""
    nb = bw.shape[0]
    marg = np.zeros(N, dtype=np.float64)
    for w in range(nb):
        wv = bw[w]
        for t in range(N):
            if (wv >> t) & 1:
                marg[t] += 1.0
    inv = 1.0 / nb
    for t in range(N):
        marg[t] *= inv
    return marg


def warmup(N: int = 20, nD: int = 44) -> None:
    """Trigger AOT-cached compilation of both kernels on tiny inputs (so the first real call
    isn't paying the JIT cost). Idempotent; safe to call at import-adjacent setup time."""
    bw = np.array([1, 3, 5], dtype=np.int64)
    cover = np.ones(nD, dtype=np.int64)
    belief_marg_cover(bw, cover, N)
    marginals_kernel(bw, N)
