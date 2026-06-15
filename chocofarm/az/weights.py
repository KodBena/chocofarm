#!/usr/bin/env python3
"""
chocofarm AZ — WeightContainer: the ONE owner of the value+policy MLP's weight LAYOUT (audit item J).

Public Domain (The Unlicense).

What "the layout" is, and why it has one home now
-------------------------------------------------
The net's weights are a small registry — `W1 b1 W2 b2 Wv bv [Wr1 br1 Wr2 br2] [Wp bp]` (trunk, then
the value head, then the optional residual block, then the optional policy head — the exact order the
npz bytes and the redis manifest commit to) — plus three scalars (`residual`, `y_mean`, `y_std`) and
one predicate (`is_weight`: which params are L2-penalized weight MATRICES). Four facts travel together:
the key ORDER, the L2 MASK, the residual TOGGLE, and the y-scale META. They define exactly two things —
what the net's weights ARE and how they (de)serialize.

Before this module that definition was SPLIT-BRAINED (audit R11's deferred follow-up). `ValueMLP`
(mlp.py) owned the `_params()` registry, the `is_weight` mask, the residual flag, the y-scale, and the
npz `save`/`load`. The redis TRANSPORT (`parallel.pack_net`/`unpack_net`) INDEPENDENTLY re-enumerated
the same registry into a raw-bytes manifest + blob. Two encoders of one layout: an optional-block edit
(or a new head) had to land in BOTH, and a drift between them is exactly the silent-serialization bug
ADR-0002 forbids and ADR-0011 Rule 4 says to net structurally ("derive from one source, never an
enumeration that fails open at the next instance").

This module is that one source. It owns:
  * `param_order(residual, has_policy)` — the canonical key ORDER (the only place the sequence
    `W1 b1 W2 b2 Wv bv [Wr1 br1 Wr2 br2] [Wp bp]` is spelled — value head BEFORE the residual block,
    policy head last; this is exactly the order `ValueMLP._params()` historically produced).
  * `is_weight(name)` — the L2-scope predicate (a param is penalized iff its name starts "W").
  * `params(net)` — the registry dict {key: net's array}, read live off the net's attributes (so the
    arrays stay the net's, preserving the rebind-not-mutate float32-cache invariant, ADR-0001).
  * `save_npz(net, path)` / `load(ctor, path)` — the npz persistence (meta = [in_dim, H, n_actions,
    residual], yscale = [y_mean, y_std]; opens the npz exactly once; the residual-OFF / shape-mismatch
    fail-loud handling, ADR-0002; `ctor` is the net constructor so construction stays the net's).
  * `pack(net)` / `unpack_into(net, manifest, blob)` — the transport's raw-bytes (de)serialization
    (the JSON manifest of name/shape/dtype/offset/len + the concatenated `tobytes()` blob, rebuilt via
    `np.frombuffer`). NO pickle — contiguous weight bytes.

`ValueMLP` HOLDS no separate copy: it keeps the arrays as its own attributes (the f32 cache and the
trainer's `getattr`/`setattr` write-back read them there) and DELEGATES `_params`/`_is_weight`/`save`
(passing `self`) and `load` (passing its own `cls` as the constructor adapter) to the container.
`parallel.pack_net`/`unpack_net` likewise call `pack` / construct-then-`unpack_into`. The container
reads and writes a net through the SAME attribute names every
consumer already uses (`getattr(net, k)` / `setattr(net, k, ...)`), so the public `ValueMLP` API and
the trainer's `net._params()` access (audit M) are unchanged — and the serialization is BYTE-IDENTICAL
to the two pre-split encoders (the J-gate pins npz round-trip + pack_net manifest+blob bit-for-bit,
both residual settings).
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeVar, cast

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

# `load`/`unpack_into` are generic over the concrete net type: `ctor` builds it and the same object
# is returned, so the caller (`ValueMLP.load`) keeps its `ValueMLP` static type rather than widening
# to the `_Net` Protocol. Bound to `_Net` so the body may still read the layout-bearing scalars.
_NetT = TypeVar("_NetT", bound="_Net")


class _Net(Protocol):
    """The structural net surface `WeightContainer` reads/writes — the layout-bearing scalars
    (`residual`/`n_actions`/`in_dim`/`H`/`y_mean`/`y_std`) plus the weight arrays held as attributes
    (read live via `getattr`, rebound on load via direct assignment). `ValueMLP` satisfies it without
    an import cycle (mlp imports weights), and the stateless container takes the net it operates on as
    this Protocol. The optional residual-block / policy-head arrays are present only when the net built
    that block — `params`/`load` gate on `residual`/`n_actions` before touching them, so they are
    declared here (the layout the container owns) but accessed only on the right toggle."""

    residual: bool
    n_actions: int | None
    in_dim: int
    H: int
    y_mean: float
    y_std: float
    W1: "npt.NDArray[Any]"
    b1: "npt.NDArray[Any]"
    W2: "npt.NDArray[Any]"
    b2: "npt.NDArray[Any]"
    Wv: "npt.NDArray[Any]"
    bv: "npt.NDArray[Any]"
    Wr1: "npt.NDArray[Any]"
    br1: "npt.NDArray[Any]"
    Wr2: "npt.NDArray[Any]"
    br2: "npt.NDArray[Any]"
    Wp: "npt.NDArray[Any]"
    bp: "npt.NDArray[Any]"


def is_weight(name: str) -> bool:
    """THE L2-scope predicate (audit R11, single source): a param is L2-penalized iff it is a weight
    MATRIX — its registry name starts 'W' (W1/W2/Wr1/Wr2/Wv/Wp), not a bias 'b*'. The JaxTrainer's
    loss (`mlp_jax_train._l2_sumsq`) and `ValueMLP._is_weight` route here so there is ONE definition,
    not a re-derived `name.startswith('W')` per consumer."""
    return name.startswith("W")


class WeightContainer:
    """The single owner of the net's weight LAYOUT — key order, L2 mask, residual toggle, y-scale meta,
    npz persistence, and the transport's raw-bytes (de)serialization. Stateless: every method takes the
    `net` whose attributes ARE the weights, so the arrays stay the net's (rebind-not-mutate cache
    coherence, ADR-0001) and `getattr`/`setattr` stay the access path every consumer already uses."""

    # the L2-scope predicate, surfaced as a staticmethod AND a module function (single definition)
    is_weight = staticmethod(is_weight)

    @staticmethod
    def param_order(residual: bool, has_policy: bool) -> list[str]:
        """THE canonical param-key ORDER — the only place the sequence is spelled. Trunk, then the
        value head, then (iff `residual`) the residual block, then (iff `has_policy`) the policy head:

            W1 b1 W2 b2  Wv bv  [Wr1 br1 Wr2 br2]  [Wp bp]

        This is EXACTLY the order `ValueMLP._params()` historically produced (value head before the
        residual block, policy head last) — the byte order the npz save and the redis manifest both
        commit to, so the J-gate's byte-identity holds."""
        keys = ["W1", "b1", "W2", "b2", "Wv", "bv"]
        if residual:
            keys += ["Wr1", "br1", "Wr2", "br2"]
        if has_policy:
            keys += ["Wp", "bp"]
        return keys

    @classmethod
    def params(cls, net: _Net) -> dict[str, npt.NDArray[Any]]:
        """The param registry as an ORDERED dict {key: net's array}, read live off the net's attributes
        (drives L2; consumed by save/load, the transport, the trainer, and `forward_core`). The residual
        block is included iff `net.residual`; the policy head iff `net.n_actions is not None` — the same
        two toggles `param_order` keys on."""
        keys = cls.param_order(net.residual, net.n_actions is not None)
        # the weight arrays are held as net attributes reached by name; getattr's `Any` is the
        # array-internals `NDArray[Any]` the layout deals in (no runtime change — same arrays).
        return {k: getattr(net, k) for k in keys}

    # ---- npz persistence ----
    @classmethod
    def save_npz(cls, net: _Net, path: str) -> None:
        """Write the net's params + meta to an npz at `path`. `_meta` carries 4 fields ([in_dim, H,
        n_actions-or-(-1), residual-0/1]); `_yscale` carries [y_mean, y_std]. Old npz files have only
        3 meta fields — `load_into` handles that length explicitly (absent → residual OFF)."""
        # the splat into np.savez(**d) is the heterogeneous npz key->array map; typed `Any` (the
        # array-internals leakage the layout deals in) so the keyword-unpack matches savez's stub
        # (its `**kwds: ArrayLike` vs `allow_pickle: bool` overload rejects a narrowed array value).
        d: dict[str, Any] = {k: v for k, v in cls.params(net).items()}
        d["_meta"] = np.array([net.in_dim, net.H,
                               net.n_actions if net.n_actions is not None else -1,
                               1 if net.residual else 0],
                              dtype=np.int64)
        d["_yscale"] = np.array([net.y_mean, net.y_std], dtype=np.float64)
        np.savez(path, **d)

    @classmethod
    def load(cls, ctor: Callable[..., _NetT], path: Any) -> _NetT:
        """Reconstruct a net from an npz, opening the file EXACTLY ONCE (so a stream/BytesIO `path`
        round-trips — the original `ValueMLP.load` opened it once). `ctor(in_dim, hidden, n_actions,
        y_mean, y_std, residual) -> net` is the net constructor (the `ValueMLP` classmethod hands its
        own `cls`), so the container owns the LAYOUT (meta read, residual decision, bind + shape
        validation) while construction stays the net's.

        The residual block is built only if BOTH `_meta` says so AND the Wr*/br* arrays are present —
        mismatch → block OFF with a clear log line (ADR-0002: fail informative, not opaque). Binding
        REBINDS each attribute (new objects) so the f32 inference cache invalidates (ADR-0001); the
        block-param SHAPES are validated at bind time (ADR-0002), not deep in the first forward."""
        z = np.load(path, allow_pickle=False)
        meta = [int(x) for x in z["_meta"]]
        in_dim, H, na = meta[0], meta[1], meta[2]
        # 4th meta field is the residual flag; absent in pre-residual npz files (length 3).
        meta_residual = bool(meta[3]) if len(meta) >= 4 else False
        y_mean, y_std = (float(x) for x in z["_yscale"])
        n_actions = None if na < 0 else na
        have_res_params = all(k in z.files for k in ("Wr1", "br1", "Wr2", "br2"))
        residual = meta_residual and have_res_params
        if meta_residual and not have_res_params:
            print(f"[ValueMLP.load] {path}: _meta says residual=ON but block params "
                  f"(Wr1/br1/Wr2/br2) are absent — loading with residual OFF", flush=True)
        net = ctor(in_dim, H, n_actions, y_mean, y_std, residual)
        net.W1, net.b1 = z["W1"], z["b1"]
        net.W2, net.b2 = z["W2"], z["b2"]
        net.Wv, net.bv = z["Wv"], z["bv"]
        if residual:
            for k, want in (("Wr1", (H, H)), ("Wr2", (H, H)), ("br1", (H,)), ("br2", (H,))):
                if z[k].shape != want:
                    raise ValueError(
                        f"ValueMLP.load {path}: residual param {k} has shape {z[k].shape}, "
                        f"expected {want} (hidden={H}) — corrupt/incompatible npz")
            net.Wr1, net.br1 = z["Wr1"], z["br1"]
            net.Wr2, net.br2 = z["Wr2"], z["br2"]
        if n_actions is not None:
            net.Wp, net.bp = z["Wp"], z["bp"]
        return net

    # ---- transport: raw-bytes (de)serialization (the redis broadcast payload, no pickle) ----
    @classmethod
    def pack(cls, net: _Net) -> tuple[str, bytes]:
        """Pack `net` into (manifest_json: str, blob: bytes) — raw `tobytes()` of each weight in the
        canonical `params(net)` order, concatenated, plus a JSON manifest of (name, shape, dtype,
        offset, byte-length) and the scalar meta (in_dim, H, n_actions, y_mean, y_std, residual). NO
        pickle: contiguous weight bytes. Optional params (the residual block) ride along automatically
        because `params(net)` reports them iff the net built the block."""
        parts = []
        layout = []
        off = 0
        for k, arr in cls.params(net).items():
            a = np.ascontiguousarray(arr)
            b = a.tobytes()
            layout.append({"name": k, "shape": list(a.shape), "dtype": a.dtype.str,
                           "off": off, "len": len(b)})
            parts.append(b)
            off += len(b)
        manifest = {
            "in_dim": net.in_dim, "H": net.H, "n_actions": net.n_actions,
            "y_mean": net.y_mean, "y_std": net.y_std, "residual": net.residual, "layout": layout,
        }
        return json.dumps(manifest), b"".join(parts)

    @staticmethod
    def unpack_meta(manifest_json: str) -> dict[str, Any]:
        """The construction meta from a `pack`'d manifest (no array data touched): a dict with `in_dim`,
        `H`, `n_actions`, `y_mean`, `y_std`, `residual`. `parallel.unpack_net` uses this to construct the
        net (older manifests without `residual` → block OFF), then `unpack_into` binds the arrays."""
        # json.loads is Any; the pack'd manifest is always a JSON object — the cast states that
        # contract (no runtime change — same parsed dict).
        return cast("dict[str, Any]", json.loads(manifest_json))

    @classmethod
    def unpack_into(cls, net: _NetT, manifest_json: str, blob: bytes) -> _NetT:
        """Bind `pack`'s (manifest, blob) onto an already-constructed `net` (its residual/n_actions
        already match the manifest, so the Wr*/br* layout entries have a slot to bind to). `np.frombuffer`
        views, copied so the net owns writable arrays. REBINDS each attribute (f32-cache coherence,
        ADR-0001). No pickle."""
        m = json.loads(manifest_json)
        for e in m["layout"]:
            a = np.frombuffer(blob, dtype=np.dtype(e["dtype"]),
                              count=int(np.prod(e["shape"])) if e["shape"] else 1,
                              offset=e["off"]).reshape(e["shape"]).copy()
            setattr(net, e["name"], a)
        return net
