#!/usr/bin/env python3
"""
cpp/stage_a/stage_a_server.py — a THROWAWAY bench-scoped InferenceServer variant for the
eval-transport-adapter Stage A microbench (docs/design/cpp-eval-transport-adapter.md §4). NOT a
committed fixture, NOT the production server, and it does NOT modify the production eval path
(inference_server.py is imported unchanged; this only SUBCLASSES it for the bench).

What it adds over the production InferenceServer (all bench-scoped, behind a subclass — the production
drain stays exactly as shipped):

  * E-policy (the server eval SHAPE, decoupled from S — design §1a):
      - "padmax"  : pad every drained batch to a single fixed shape (pad_to = 512). This is the
                    production behaviour at max_batch=512 — the pad-to-max baseline (the half-batch
                    ~2x pad tax the design predicts loses on partial drains).
      - "bucket"  : snap the drained row count UP to the nearest of {64, 256, 512} (NEVER pad-to-max).
                    Three AOT-compiled bucket shapes; the drain biases to the largest the arrival
                    stream fills. This is the design's ADOPT lever.
    E is chosen SERVER-SIDE from however many rows the drain accumulated — it is independent of S (the
    rows-per-wire-message the C++ producer chose). The bench reports mean rows/forward to confirm that.

  * wakeup (design §1c):
      - "group"   : drain ALL currently-queued requests, ONE forward over their concatenated rows
                    (the production greedy-drain — one wake per drained group).
      - "leaf"    : process ONE queued request (one S-message) per forward — the per-leaf-condvar
                    degenerate where the server never coalesces across requests (one wake per leaf
                    cohort). Models the literal per-leaf signal the design predicts loses.

  * counters: total forwards run + total rows forwarded (REAL rows, not padded) + total padded rows,
    so the harness can report forwards/s and mean REAL rows/forward, and confirm E decouples from S.

The drain/serve overrides re-use the production `run_microbatch` (the one forward SSOT, P1/P7) — the
E-policy only changes the `pad_to` argument, the wakeup only changes how many requests one forward
covers. No second forward transcription, no codec edit, the corr-id stays a transport-envelope frame
the base server round-trips opaquely (frames[1:-1]; ADR-0012 P7).

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from typing import Any

REPO = "/home/bork/w/vdc/1/chocofarm"
sys.path.insert(0, REPO)

import chocofarm.config  # noqa: F401,E402 — XLA/OMP single-thread pin BEFORE any jax init (SSOT)
import numpy as np  # noqa: E402

from chocofarm.az.actions import n_action_slots  # noqa: E402
from chocofarm.az.features import feature_dim  # noqa: E402
from chocofarm.az.inference_server import (  # noqa: E402
    InferenceServer,
    StaticParamsSource,
    jit_forward_core,
    params_from_manifest_blob,
    run_microbatch,
)
from chocofarm.az.mlp import ValueMLP  # noqa: E402
from chocofarm.az.transport import pack_net  # noqa: E402
from chocofarm.model.env import Environment  # noqa: E402

BUCKETS = (64, 256, 512)


def _bucket_for(n_rows: int, buckets: "tuple[int, ...]" = BUCKETS) -> int:
    """Snap a real row count UP to the nearest bucket — never pad-to-max. A drain larger than the top
    bucket is capped at the bench max_batch upstream, so n_rows <= buckets[-1] here; clamp defensively."""
    for b in buckets:
        if n_rows <= b:
            return b
    return buckets[-1]


class StageAServer(InferenceServer):
    """A bench-scoped InferenceServer: same greedy ROUTER drain, but the E-policy (pad shape) and the
    wakeup granularity are knobs, and per-forward counters are kept. Production `_drain`/`_serve_batch`
    are untouched on the base class; this overrides only for the bench."""

    def __init__(self, *args: Any, e_policy: str = "bucket", wakeup: str = "group",
                 buckets: "tuple[int, ...] | None" = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if e_policy not in ("padmax", "bucket"):
            raise ValueError(f"e_policy must be padmax|bucket, got {e_policy!r}")
        if wakeup not in ("group", "leaf"):
            raise ValueError(f"wakeup must be group|leaf, got {wakeup!r}")
        # The AOT bucket set (snap real row count UP to one of these). Default = the module BUCKETS; a
        # sweep that pushes the server forward WIDTH higher passes a richer set (e.g. up to max_batch) so a
        # large accumulated batch lands on a compiled shape instead of forcing a cold per-width compile.
        self._buckets = tuple(sorted(buckets)) if buckets else BUCKETS
        if self._buckets[-1] > self._max_batch:
            raise ValueError(f"top bucket {self._buckets[-1]} exceeds max_batch {self._max_batch}")
        self._e_policy = e_policy
        self._wakeup = wakeup
        self.n_forwards = 0
        self.n_real_rows = 0
        self.n_padded_rows = 0

    def _serve_batch(self, drained: list) -> None:  # type: ignore[override]
        """Run ONE microbatch per the wakeup granularity, choosing the pad shape per the E-policy.

        per-group: all drained requests -> ONE forward (concatenated rows, one pad/bucket).
        per-leaf : EACH drained request -> its OWN forward (no cross-request coalescing).

        The E-policy decides pad_to from the REAL row count of the forward: padmax -> 512; bucket ->
        the nearest of {64,256,512}. run_microbatch is the one production forward (P1) — only its pad_to
        and the requests it covers change here."""
        params, y_mean, y_std = self._params_source.current()
        groups: list[list] = [drained] if self._wakeup == "group" else [[d] for d in drained]
        for group in groups:
            rows = [(ident, X) for ident, _envelope, X in group]
            real = int(sum(X.shape[0] for X in (X for _i, X in rows)))
            if self._e_policy == "padmax":
                pad_to = self._max_batch
            else:
                pad_to = _bucket_for(real, self._buckets)
            responses = run_microbatch(self._forward_fn, params, y_mean, y_std, rows, pad_to=pad_to)
            self.n_forwards += 1
            self.n_real_rows += real
            self.n_padded_rows += max(0, pad_to - real)
            for (ident, resp), (_ident, envelope, _X) in zip(responses, group):
                self._sock.send_multipart([ident, *envelope, resp])


def build(hidden: int, endpoint: str, max_batch: int, e_policy: str, wakeup: str,
          min_forward_rows: int = 0, max_queue_delay_ms: float = 0.0):
    env = Environment()
    in_dim, n_actions = feature_dim(env), n_action_slots(env)
    net = ValueMLP(in_dim, hidden=hidden, n_actions=n_actions, seed=17,
                   y_mean=0.0, y_std=1.0, residual=False)
    params, y_mean, y_std = params_from_manifest_blob(*pack_net(net))
    server = StageAServer(StaticParamsSource(params, y_mean, y_std), bind=endpoint,
                          max_batch=max_batch, forward_fn=jit_forward_core,
                          e_policy=e_policy, wakeup=wakeup,
                          min_forward_rows=min_forward_rows, max_queue_delay_ms=max_queue_delay_ms)
    # Warm EVERY bucket shape + the pad-to-max shape up front so a partial-drain forward never pays a
    # cold JIT-compile inside the timed window (ADR-0009 measure-honesty). The bucket forwards compile
    # at {64,256,512}; padmax always compiles at max_batch.
    server.warmup(sorted(set(BUCKETS) | {max_batch}))
    return server, in_dim, n_actions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--e-policy", choices=("padmax", "bucket"), default="bucket")
    ap.add_argument("--wakeup", choices=("group", "leaf"), default="group")
    # The increment-(ii) server floor (server-floor-design.md): default OFF (θ=0) = the greedy drain.
    ap.add_argument("--min-forward-rows", type=int, default=0)
    ap.add_argument("--max-queue-delay-ms", type=float, default=0.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd

    server, in_dim, n_actions = build(a.hidden, a.endpoint, a.max_batch, a.e_policy, a.wakeup,
                                      a.min_forward_rows, a.max_queue_delay_ms)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[stage_a_server] up hidden={a.hidden} in_dim={in_dim} n_actions={n_actions} "
          f"max_batch={a.max_batch} e_policy={a.e_policy} wakeup={a.wakeup} "
          f"floor(theta={a.min_forward_rows},delay_ms={a.max_queue_delay_ms}) endpoint={a.endpoint}",
          flush=True)

    rc = 0
    if cmd:
        import subprocess
        t_run0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=REPO, text=True, timeout=900)
        run_wall = time.perf_counter() - t_run0
        rc = proc.returncode
        server.stop()
        t.join(timeout=5.0)
        fwds, real, pad = server.n_forwards, server.n_real_rows, server.n_padded_rows
        mean_rows = (real / fwds) if fwds else 0.0
        pad_frac = (pad / (real + pad)) if (real + pad) else 0.0
        print(f"[stage_a_server] SERVER_STATS forwards={fwds} real_rows={real} padded_rows={pad} "
              f"mean_real_rows_per_fwd={mean_rows:.2f} pad_fraction={pad_frac:.4f} "
              f"server_fwd_per_s={fwds / run_wall:.1f} run_wall={run_wall:.3f}", flush=True)
        server.close()
    else:
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        stop.wait()
        server.stop()
        t.join(timeout=5.0)
        server.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
