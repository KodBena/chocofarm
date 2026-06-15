#!/usr/bin/env python3
"""
chocofarm/az/transport.py — RedisTransport: the SOLE owner of the AZ parallel-loop redis raw-bytes
wire protocol (audit item K, the Transport ⊥ Pool ⊥ Task split out of `parallel.py`).

This module owns EVERYTHING about how weights and results travel over redis and nothing about the
process pool (that is `worker_pool.py`) or what one worker computes (that is `worker.py`). Concretely
it owns: the bounded-timeout connection construction + fail-loud ping (ADR-0002 / deadlock fix H2),
the ONE place the `az:w:<run>:<version>:m|:b` weight keys and the `az:res:<token>:<idx>:X|PI|M|Y`
result keys are spelled (`weight_keys()` / `result_keys()` — no f-strings scattered across
publish/ensure_net/gen_task/collect anymore), the weight publish/read, the result blob write, and the
result blob read+delete. The TTLs live here too (weights 3600s; results `CHOCO_RESULT_TTL`).

Weight (un)packing STAYS delegated to `WeightContainer` (audit item J): this module calls
`pack_net`/`unpack_net` and never re-encodes the layout. The key STRINGS and the on-the-wire bytes are
byte-identical to the pre-split encoder — the protocol is unchanged, only its ownership is.

Connection facts come from `chocofarm/config.py` (`redis_params()` / `redis_socket_timeout()` /
`redis_connect_timeout()`), defaulting to 127.0.0.1:6379 db 0 — DO NOT change which instance/db this
targets (the module is shared with the registry's `redis_params()`).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import os

import numpy as np


# ---- net (de)serialization as raw bytes — DELEGATED to WeightContainer (audit item J) ----
# The raw-bytes (de)serialization is the WeightContainer's (audit item J): the net's weight LAYOUT —
# the param-key order, the manifest's name/shape/dtype/offset/len entries, the scalar meta, and the
# `tobytes()`/`np.frombuffer` round-trip — has ONE owner there, the same owner the npz save/load and
# the L2-mask route through. The transport NO LONGER re-enumerates `net._params()` into its own
# manifest (that was the split-brained second encoder R11 deferred); it constructs the net from the
# manifest's meta and delegates the (un)packing. The blob/manifest bytes are byte-identical to the
# pre-J encoder (the param order is the container's canonical order, which is the historical order).


def pack_net(net):
    """Pack a ValueMLP into (manifest_json: str, blob: bytes) — DELEGATED to
    `WeightContainer.pack` (the one owner of the weight layout, audit item J). Raw `tobytes()` of each
    weight concatenated, a JSON manifest of (name, shape, dtype, offset, byte-length) + the scalar
    meta. No pickle: the blob is contiguous float64 weight bytes; optional params (residual block) ride
    along automatically because the container reports them iff the net built the block."""
    from chocofarm.az.weights import WeightContainer
    return WeightContainer.pack(net)


def unpack_net(manifest_json, blob):
    """Reconstruct a ValueMLP from `pack_net`'s (manifest, blob). The container reads the manifest's
    construction meta (so older manifests without `residual` → block OFF); this builds the net via the
    `ValueMLP` constructor (so the Wr*/br* layout entries have a slot to bind to), then the container
    binds the arrays (`np.frombuffer` views, copied so the net owns writable arrays). No pickle."""
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.az.weights import WeightContainer
    m = WeightContainer.unpack_meta(manifest_json)
    net = ValueMLP(m["in_dim"], hidden=m["H"],
                   n_actions=m["n_actions"], y_mean=m["y_mean"], y_std=m["y_std"],
                   residual=bool(m.get("residual", False)))
    return WeightContainer.unpack_into(net, manifest_json, blob)


# ---- the wire protocol's KEY NAMESPACE (the ONE place the key strings are spelled) ----
# These build the byte-identical keys the pre-split code had inlined as f-strings in publish_weights /
# _ensure_net / _gen_task / _collect_results. Putting them here makes RedisTransport the sole owner of
# the on-the-wire protocol: a key change is a one-site change, and the worker side (worker.py) builds
# its read/write keys through the SAME helpers, so the parent and child can never disagree.

def weight_keys(run, version):
    """The two weight keys for (run, version): (manifest_key, blob_key) =
    `az:w:<run>:<version>:m`, `az:w:<run>:<version>:b`. Byte-identical to the pre-split f-strings."""
    base = f"az:w:{run}:{version}"
    return base + ":m", base + ":b"


def result_keys(res_token, idx):
    """The four result-blob keys for (res_token, idx): (X, PI, M, Y) =
    `az:res:<token>:<idx>:X|PI|M|Y`. Byte-identical to the pre-split f-strings."""
    base = f"az:res:{res_token}:{idx}"
    return base + ":X", base + ":PI", base + ":M", base + ":Y"


# ---- redis connection (raw-bytes transport; no pickle) ----
def _redis_params():
    """Shared connection facts from chocofarm/config.py (the registry uses the same), so transport
    and registry address one redis instance by default."""
    from chocofarm import config
    return config.redis_params()


def connect():
    import redis  # local import so a serial (workers=0) run needs no redis at all
    # Bound EVERY socket op (ADR-0002 / deadlock fix H2). The default `socket_timeout=None`
    # makes every r.get / pipe.execute block FOREVER if the TCP socket stalls — a stalled
    # worker read is then indistinguishable, from the parent's imap fan-out, from a wedged
    # worker, and the loop sits at futex_do_wait at ~1% CPU with no way out. A bounded timeout
    # turns a stall into a loud redis.TimeoutError (retryable / restart-recoverable; checkpoints
    # are per-iteration) instead of a silent permanent hang. Loopback redis under no memory
    # pressure never trips 60s, so this is a safety net, not a happy-path behavior change.
    from chocofarm import config
    r = redis.Redis(
        socket_timeout=config.redis_socket_timeout(),
        socket_connect_timeout=config.redis_connect_timeout(),
        **_redis_params(),
    )
    r.ping()   # fail loud now if redis is unreachable (ADR-0002), not mid-iteration
    return r


# TTL for result blobs: a TTL so an ABORTED iteration self-cleans. The happy path deletes them in
# `read_and_delete_results` the same iteration; but if the fan-out is aborted (the loud drain timeout)
# the parent never reaches the delete, and a bare SET leaves the blob with no expiry forever (the
# post-mortem found ~980 such leaked az:res:* keys, TTL=-1). A 1h TTL bounds that leak without
# affecting the happy path (read+deleted within seconds). `ex=` sets the expiry in the same SET
# round-trip (no extra command). Env-overridable.
def _result_ttl():
    return int(os.environ.get("CHOCO_RESULT_TTL", "3600"))


# Weight keys carry a 1h expiry so a long run doesn't leak old versions.
_WEIGHT_TTL_S = 3600


class RedisTransport:
    """The SOLE owner of the AZ parallel-loop redis raw-bytes protocol. Construct on the parent (with
    a `connect()`'d client) for weight publish + result read; the worker side calls the module-level
    read/write functions with its OWN connection (kept in `_W`, item L) — but the key strings, the
    TTLs, and the (un)packing all route through this module either way, so there is exactly one wire
    protocol."""

    def __init__(self, conn):
        self.r = conn

    # ---- weights: parent publishes, worker reads ----
    def publish_weights(self, net, version, run):
        """Pack the net to raw bytes and publish to redis `az:w:<run>:<version>` (no pickle, no disk).
        Workers `read_weights` it when the version changes. Weight keys carry a 1h expiry."""
        manifest, blob = pack_net(net)
        mk, bk = weight_keys(run, version)
        pipe = self.r.pipeline(transaction=False)
        pipe.set(mk, manifest)
        pipe.set(bk, blob)
        pipe.expire(mk, _WEIGHT_TTL_S)
        pipe.expire(bk, _WEIGHT_TTL_S)
        pipe.execute()

    # ---- results: worker writes, parent reads+deletes ----
    def read_and_delete_results(self, res_token, metas):
        """Read the raw-byte result blobs the workers wrote for `metas` (a list of (idx, n, feat_dim,
        n_slots)) back into one flat list of (feat, pi, mask, g) records, then DELETE the keys (raw
        bytes can be large; don't leak across iterations). The happy-path cleanup that pairs with
        `write_results`' TTL safety net."""
        out = []
        pipe = self.r.pipeline(transaction=False)
        order = []
        for (idx, n, fd, ns) in metas:
            if n == 0:
                continue
            xk, pik, mk, yk = result_keys(res_token, idx)
            pipe.get(xk); pipe.get(pik); pipe.get(mk); pipe.get(yk)
            order.append((idx, n, fd, ns, xk, pik, mk, yk))
        if not order:
            return out
        blobs = pipe.execute()
        # delete the result keys (raw bytes can be large; don't leak across iterations)
        dpipe = self.r.pipeline(transaction=False)
        for k, (idx, n, fd, ns, xk, pik, mk, yk) in enumerate(order):
            xb, pib, mb, yb = blobs[4 * k:4 * k + 4]
            X = np.frombuffer(xb, dtype=np.float32).reshape(n, fd)
            PI = np.frombuffer(pib, dtype=np.float32).reshape(n, ns)
            M = np.frombuffer(mb, dtype=np.float32).reshape(n, ns)
            Y = np.frombuffer(yb, dtype=np.float32)
            for i in range(n):
                out.append((X[i], PI[i], M[i], float(Y[i])))
            dpipe.delete(xk, pik, mk, yk)
        dpipe.execute()
        return out


def read_weights(conn, run, version):
    """Worker-side weight READ: fetch the raw-bytes payload for (run, version) over `conn` and return
    (manifest_str, blob_bytes). A missing payload is a loud RuntimeError (ADR-0002: never a silent
    stale-net serve). The bytes feed `unpack_net`; the (un)packing stays the WeightContainer's."""
    mk, bk = weight_keys(run, version)
    manifest = conn.get(mk)
    blob = conn.get(bk)
    if manifest is None or blob is None:
        raise RuntimeError(f"weight payload az:w:{run}:{version} missing from redis")
    return manifest.decode("utf-8"), blob


def write_results(conn, res_token, idx, X, PI, M, Y):
    """Worker-side result WRITE: pipeline the four contiguous float32 blocks (X/PI/M/Y) under the
    per-task result keys, each with the result TTL set in the same SET round-trip (the aborted-iteration
    self-clean safety net). No pickle — `tobytes()` of contiguous arrays."""
    xk, pik, mk, yk = result_keys(res_token, idx)
    ttl = _result_ttl()
    pipe = conn.pipeline(transaction=False)
    pipe.set(xk, X.tobytes(), ex=ttl)
    pipe.set(pik, PI.tobytes(), ex=ttl)
    pipe.set(mk, M.tobytes(), ex=ttl)
    pipe.set(yk, Y.tobytes(), ex=ttl)
    pipe.execute()
