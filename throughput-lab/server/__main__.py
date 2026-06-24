#!/usr/bin/env python3
"""
throughput-lab/server/__main__.py — the server entry point (the imperative shell): parse CLI flags
into a ServerConfig, stand up a ThroughputServer, and serve until interrupted. Run as:
    PYTHONPATH=throughput-lab python -m server [--bind ipc://...] [--n-actions N] ...

The CLI is an ACL (parse-and-validate argv ONCE into the typed ServerConfig; a bad flag is a loud
stderr error + non-zero exit via argparse, never a silent default — ADR-0002). The interpreter is
/home/bork/w/vdc/venvs/generic/bin/python (JAX, numpy, pyzmq), run with PYTHONPATH=throughput-lab so
`import server` resolves.

CLI surface:
    --bind        <ipc://...|tcp://...>   ROUTER bind endpoint (default ipc:///tmp/tlab-infer.sock)
    --max-batch   <N>                     cap on N_total rows per forward (default 4096)
    --in-dim      <D>                     feature width (default 241 = Stage-A)
    --n-actions   <A>                     policy width (default 0 = value-only)
    --hidden      <H>                     MLP hidden width (default 256)
    --residual                            add the keyed residual block (match a residual net)
    --seed        <S>                     RNG seed for the throwaway random weights (default 0)
    --warmup      <B,B,...>               batch sizes to pre-compile (default 1,8,64,512,4096)
    --poll-timeout-ms <ms>                IO-thread idle poll timeout (default 50); bounds stop()
                                          latency (NO LONGER floors COUPLED RTT — the reply wake pipe
                                          flushes a reply the instant it is ready)
    --quiet                               suppress the READY line + the teardown stats summary

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import signal
import sys

import deal

from server.server import ServerConfig, ThroughputServer


def _parse_warmup(s: str) -> "list[int]":
    """Parse a comma-separated batch-size list, validating loudly (a non-positive or non-integer entry
    is a configuration error, ADR-0002 — not a value to silently drop)."""
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            b = int(tok)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"warmup entry {tok!r} is not an integer") from e
        if b <= 0:
            raise argparse.ArgumentTypeError(f"warmup batch size must be >= 1, got {b}")
        out.append(b)
    if not out:
        raise argparse.ArgumentTypeError("warmup list is empty")
    return out


def _build_config(argv: "list[str]") -> ServerConfig:
    p = argparse.ArgumentParser(
        prog="python -m server",
        description="throughput-lab server: a clean-room decoupled ROUTER receive/serve loop.")
    p.add_argument("--bind", default="ipc:///tmp/tlab-infer.sock",
                   help="ZMQ ROUTER bind endpoint (default: %(default)s)")
    p.add_argument("--max-batch", type=int, default=4096,
                   help="cap on N_total rows per forward (default: %(default)s)")
    p.add_argument("--in-dim", type=int, default=241,
                   help="feature width per leaf row (default: %(default)s = Stage-A)")
    p.add_argument("--n-actions", type=int, default=0,
                   help="policy width; 0 = value-only (default: %(default)s)")
    p.add_argument("--hidden", type=int, default=256,
                   help="MLP hidden width (default: %(default)s)")
    p.add_argument("--residual", action="store_true",
                   help="add the keyed residual block (match a residual net)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the throwaway random weights (default: %(default)s)")
    p.add_argument("--warmup", type=_parse_warmup, default=None,
                   help="comma-separated batch sizes to pre-compile (default: 1,8,64,512,4096)")
    p.add_argument("--poll-timeout-ms", type=int, default=50,
                   help="IO-thread idle poll timeout, ms (default: %(default)s). Bounds how quickly the "
                        "loop notices stop() and the idle wake cadence. It does NOT floor coupled RTT: "
                        "the reply wake pipe pokes the poller the instant a reply is ready, so coupled "
                        "latency reflects compute + wire at any timeout (see THE REPLY WAKE in server.py).")
    p.add_argument("--quiet", action="store_true",
                   help="suppress the READY line and the teardown stats summary")
    p.add_argument("--single-thread", action="store_true",
                   help="serve on ONE thread (drain->forward->scatter, no IO/compute split) -- the production "
                        "InferenceServer model; the A/B arm for the two-thread-contention test")
    p.add_argument("--profile-forward", action="store_true",
                   help="(single-thread) split each forward into h2d|jit|d2h sub-phase timers (diagnostic; "
                        "serialises the XLA pipeline so it slows the run)")
    p.add_argument("--forward", choices=("jax", "numpy", "prod", "staged", "null"), default="jax",
                   help="forward backend: jax (XLA-jit + bucket ladder) | numpy (forward_core in numpy, no "
                        "XLA dispatch, no pad) | prod (DIAGNOSTIC: real production jit_forward_core) | staged "
                        "(DIAGNOSTIC: real build_staged_forward — the actual overcommit 140k forward) | null "
                        "(PROBE: zero-compute, returns shaped zeros — isolates the serve-loop ceiling). prod/"
                        "staged/null are measurement-only (never shipped)")
    p.add_argument("--net", default="",
                   help="Gate B: serve a REAL trained net from this AZ .npz checkpoint (MlpForward.from_npz) "
                        "instead of a random net — gives tlab measurements direct AZ-loop relevance. Empty = "
                        "random net. Only valid with --forward jax. Geometry is derived from the checkpoint.")
    ns = p.parse_args(argv)

    # Loud validation of the few invariants argparse's types do not catch (ADR-0002 — a bad geometry
    # is a configuration error surfaced now, not a confusing forward shape later).
    if ns.max_batch <= 0:
        p.error(f"--max-batch must be >= 1, got {ns.max_batch}")
    if ns.in_dim <= 0:
        p.error(f"--in-dim must be >= 1, got {ns.in_dim}")
    if ns.n_actions < 0:
        p.error(f"--n-actions must be >= 0, got {ns.n_actions}")
    if ns.hidden <= 0:
        p.error(f"--hidden must be >= 1, got {ns.hidden}")
    if ns.poll_timeout_ms < 0:
        p.error(f"--poll-timeout-ms must be >= 0, got {ns.poll_timeout_ms}")

    cfg = ServerConfig(
        bind=ns.bind, max_batch=ns.max_batch, in_dim=ns.in_dim, n_actions=ns.n_actions,
        hidden=ns.hidden, residual=ns.residual, seed=ns.seed, verbose=not ns.quiet,
        poll_timeout_ms=ns.poll_timeout_ms, single_thread=ns.single_thread,
        profile_forward=ns.profile_forward, forward_impl=ns.forward, net_path=ns.net,
    )
    if ns.warmup is not None:
        cfg.warmup_batch_sizes = ns.warmup
    return cfg


def main(argv: "list[str] | None" = None) -> int:
    # Strip the `deal` contracts on the serving hot-path (decode_bounded / pack): they are the
    # machine-checkable SPEC, discharged by the property suite (tests run with them ON) — here, at
    # measurement time, they are pure overhead the throughput number should not pay (clause 10: "the
    # rigorous test passed, the abstraction made free"). The `attrs` BoundedBatch validator stays ON
    # (it is NOT disabled): it is the actual ADR-0002 boundary guard that rejects a bad frame, not a
    # redundant spec — disabling it would un-fix the oversize/wrong-width rejects.
    deal.disable()
    cfg = _build_config(sys.argv[1:] if argv is None else argv)
    server = ThroughputServer(cfg)   # binds the ROUTER + warms XLA before we install the handler

    def _on_sigint(signum, frame):  # noqa: ANN001 — signal handler signature
        # A bounded stop: set the flag; the IO thread notices within poll_timeout_ms and tears down on
        # its own thread (no socket touched from the handler — ADR-0002/P5, not a mid-poll race).
        print("\n[tlab-server] SIGINT — stopping", file=sys.stderr, flush=True)
        server.stop()

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
