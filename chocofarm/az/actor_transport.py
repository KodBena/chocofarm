#!/usr/bin/env python3
"""
chocofarm/az/actor_transport.py — the ActorTransport Port (the env<->actor control seam, ADR-0012 P2)
and its persistent-subprocess impl, the vehicle for ONLINE RECONFIGURATION of the C++ Gumbel actor.

The seam (Port/ACL). `ActorTransport` is the boundary `CppActorExecutor` holds and never names a
concrete impl: it owns "how the actor is configured and driven," nothing about the wire mechanism. A
new transport (a unix socket, a ZeroMQ daemon for the async Shape C) is a new `ActorTransport` impl with
ZERO edits to the executor or to exit_loop (P2 — the seam makes pipe↔ZMQ swappable; P7 — the control
PROTOCOL is the SSOT, the transport is the swappable mechanism). The methods:

  * reconfigure(config) -> epoch   — adopt an ActorConfig on the LIVE runner (HOT knobs rebuild the
                                     policy live; an INSTANCE-knob change is a loud reject), returns the
                                     new config_epoch the runner assigned.
  * generate(req)        -> result — play req.episodes at the current config + the per-call scalars
                                     (version/seed/lam/episodes/max_steps/res_token); weights and the
                                     (X,PI,M,Y) blocks stay on the redis bytes-store, only the structured
                                     meta (written + the echoed epoch/version) rides the control channel.
  * ping()               -> ...    — readiness/liveness (the runner's serving state + active epoch).
  * close()                        — graceful-then-forceful reap.

The first impl, `SubprocessActorTransport`, runs the C++ runner as a PERSISTENT subprocess and speaks the
control_spec protocol as JSON lines over its stdin/stdout (weights/results on redis, unchanged). Its
non-hang discipline is a BOUNDED RECV: a reader thread drains stdout into a queue and `_recv` blocks only
on a `queue.get(timeout=...)` — the pipe analog of the C++ ZmqNetClient's ZMQ_RCVTIMEO and of
transport.connect's socket_timeout (a stall becomes a loud `ControlError`, never a forever-block). On a
recv timeout or a dead runner the executor raises loudly and the runner is reaped (ADR-0002 / P5).

Determinism (P6). The per-generation seed (base_seed + version) rides each `generate` message, never
sticky config — so two generates at one version reproduce, and a new version derives a new seed; the
runner re-seeds per generate and carries no RNG across generations (the subprocess already met this; the
persistent runner must preserve it). This module does not change the per-episode draw semantics.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import collections
import dataclasses
import json
import queue
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from chocofarm.az import control_spec as C
from chocofarm.az.actor_config import ActorConfig

# transport-level error tags (distinct from the protocol's control_spec.ERROR_TAGS, which are the
# RUNNER's machine tags): these name failures of the CHANNEL itself, surfaced by the client.
ERR_RECV_TIMEOUT = "recv_timeout"      # no reply within the bounded recv window (a wedged runner)
ERR_RUNNER_DIED = "runner_died"        # the runner process exited / closed its stdout (EOF)
ERR_BAD_REPLY = "bad_reply"            # the reply line was not valid JSON / missing "ok"
ERR_SEND_FAILED = "send_failed"        # the request could not be written to the runner's stdin


class ControlError(RuntimeError):
    """A control-protocol failure — either the RUNNER rejected the request (`tag` is one of
    control_spec.ERROR_TAGS, e.g. config_epoch_mismatch / instance_knob_changed) or the CHANNEL failed
    (`tag` is one of the ERR_* transport tags above). Carries the machine `tag` (so the caller branches
    on it, never on prose) and the human `detail`. Raised loudly — the executor never proceeds on a
    request the runner could not honor or a reply it could not read (ADR-0002)."""

    def __init__(self, tag: str, detail: str) -> None:
        super().__init__(f"{tag}: {detail}")
        self.tag = tag
        self.detail = detail


@dataclass(frozen=True)
class GenerateRequest:
    """The per-generation scalars that ride one `generate` message (P8: the typed contract). NOT config
    — these change every call (version/seed) or per iteration (lam/episodes/max_steps); the version->seed
    derivation lives HERE, in the message, never in sticky config (the determinism anchor, P6)."""

    config_epoch: int   # the epoch the client believes is live (loud config_epoch_mismatch on drift)
    version: int        # gates the runner's redis weight reload, INDEPENDENT of the epoch
    seed: int           # base_seed + version — re-derived per call, never cached
    lam: float          # the live Dinkelbach penalty for this generation
    episodes: int       # E — the number of episodes to play
    max_steps: int      # the live episode-horizon cap
    res_token: str      # the result-key namespace for this generation's (X,PI,M,Y) blocks


@dataclass(frozen=True)
class GenerateResult:
    """The structured meta a `generate` reply carries — the typed replacement for the stderr
    `wrote N episode(s)` scrape. The executor reconciles `written` against what it reads back from
    redis, and asserts the echoed epoch/version matched its request (the loud-on-desync check)."""

    written: int        # episodes the runner reports it wrote (reconciled against the redis read)
    config_epoch: int   # echoed — asserted == the request's config_epoch
    version: int        # echoed — asserted == the request's version


@dataclass(frozen=True)
class PingResult:
    """A `ping` reply: the runner's readiness (`serving` — env+policy built, ready to generate) and the
    active config_epoch (0 before the first successful configure)."""

    serving: bool
    config_epoch: int


@runtime_checkable
class ActorTransport(Protocol):
    """The control seam (P2). `CppActorExecutor` holds one and never names a concrete impl, so the
    transport mechanism (subprocess pipe today, a ZeroMQ daemon for the async Shape C later) is swappable
    with zero edits above this boundary. The control PROTOCOL (control_spec) is the SSOT both sides
    derive; this Port is the typed Python view of it."""

    def reconfigure(self, config: ActorConfig) -> int: ...
    def generate(self, req: GenerateRequest) -> GenerateResult: ...
    def ping(self) -> PingResult: ...
    def close(self) -> None: ...


_EOF = object()  # reader-thread sentinel: the runner closed its stdout (process exit)


class SubprocessActorTransport:
    """ActorTransport over a PERSISTENT C++ runner subprocess speaking the control_spec protocol as JSON
    lines on stdin/stdout (weights/results stay on redis). The simplest honest lock-step transport — one
    request in flight at a time, exactly the synchronous loop's shape — behind the ActorTransport seam so
    a ZeroMQ daemon drops in later unchanged. The non-hang safety net is the bounded `_recv` (a reader
    thread + `queue.get(timeout=...)`); a wedged or dead runner trips it and is reaped loudly (ADR-0002).
    """

    def __init__(self, runner_path: str, *, recv_timeout_s: float = 3600.0,
                 ready_timeout_s: float = 30.0, extra_args: tuple[str, ...] = (),
                 stderr_tail: int = 40) -> None:
        """Spawn the runner in serve mode and start the stdout/stderr reader threads. `recv_timeout_s`
        bounds a generate's reply (generous — a real E=300 generation is minutes — but FINITE, like the
        subprocess path's gen_timeout_s); `ready_timeout_s` bounds the lighter configure/ping replies.
        `extra_args` are appended after `--serve` (e.g. a redis-namespacing run id, if the runner takes
        one). Construction failure (the binary missing / not executable) raises loudly."""
        self.runner_path = runner_path
        self._recv_timeout_s = float(recv_timeout_s)
        self._ready_timeout_s = float(ready_timeout_s)
        self._closed = False
        try:
            self._proc = subprocess.Popen(
                [runner_path, "--serve", *extra_args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1)  # line-buffered text streams
        except OSError as e:
            raise ControlError(ERR_RUNNER_DIED,
                               f"could not spawn the actor runner {runner_path!r}: {e}") from e
        self._stdout_q: queue.Queue[Any] = queue.Queue()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=int(stderr_tail))
        self._readers = [
            threading.Thread(target=self._drain_stdout, daemon=True),
            threading.Thread(target=self._drain_stderr, daemon=True),
        ]
        for t in self._readers:
            t.start()

    # ---- reader threads (so _recv blocks only on a BOUNDED queue.get, never on a raw pipe read) ----
    def _drain_stdout(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:        # ends at EOF (the runner exited / closed stdout)
            self._stdout_q.put(line)
        self._stdout_q.put(_EOF)

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:        # keep the last N lines for diagnostics in errors
            self._stderr_tail.append(line.rstrip("\n"))

    def _stderr_diag(self) -> str:
        tail = list(self._stderr_tail)
        return (" | runner stderr tail: " + " ⏎ ".join(tail)) if tail else ""

    # ---- the bounded send/recv/request cycle ----
    def _send(self, msg: dict[str, Any]) -> None:
        if self._proc.stdin is None:
            raise ControlError(ERR_SEND_FAILED, "runner stdin is closed")
        try:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError) as e:  # ValueError: write to a closed stream
            raise ControlError(ERR_RUNNER_DIED,
                               f"runner stdin write failed ({e}){self._stderr_diag()}") from e

    def _recv(self, timeout: float) -> dict[str, Any]:
        try:
            line = self._stdout_q.get(timeout=timeout)
        except queue.Empty:
            self._reap()  # a wedged runner: reap it, then raise loudly (the §2.4 non-hang net)
            raise ControlError(
                ERR_RECV_TIMEOUT,
                f"no reply within {timeout:.0f}s — runner reaped{self._stderr_diag()}") from None
        if line is _EOF:
            self._reap()
            raise ControlError(ERR_RUNNER_DIED,
                               f"runner exited before replying{self._stderr_diag()}")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as e:
            raise ControlError(ERR_BAD_REPLY, f"reply was not valid JSON: {line!r} ({e})") from e
        if not isinstance(reply, dict):
            raise ControlError(ERR_BAD_REPLY, f"reply was not a JSON object: {reply!r}")
        return reply

    def _request(self, msg: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Send one request, await one reply (lock-step), and translate an error reply into a loud
        ControlError carrying the runner's machine tag (Port/ACL: validate, never silently proceed)."""
        self._send(msg)
        reply = self._recv(timeout)
        if C.KEY_OK not in reply:
            raise ControlError(ERR_BAD_REPLY, f"reply missing {C.KEY_OK!r}: {reply!r}")
        if not reply[C.KEY_OK]:
            tag = str(reply.get(C.KEY_ERROR, "unknown"))
            detail = str(reply.get(C.KEY_DETAIL, ""))
            raise ControlError(tag, detail)
        return reply

    # ---- the ActorTransport contract ----
    def reconfigure(self, config: ActorConfig) -> int:
        """Adopt `config` on the live runner; return the new config_epoch. The config crosses as a flat
        JSON object (the ActorConfig field set — drift-netted in actor_config/test_wire_drift); the
        runner rebuilds the policy live on a HOT change and loud-rejects an INSTANCE-knob change."""
        reply = self._request(
            {C.KEY_TYPE: C.MSG_CONFIGURE, C.KEY_CONFIG: dataclasses.asdict(config)},
            self._ready_timeout_s)
        return int(reply[C.KEY_CONFIG_EPOCH])

    def generate(self, req: GenerateRequest) -> GenerateResult:
        """Play `req.episodes` at the current config + the per-call scalars; return the structured meta.
        Asserts the runner echoed the SAME epoch/version (the loud-on-desync round-trip check, §11.4)."""
        reply = self._request({
            C.KEY_TYPE: C.MSG_GENERATE,
            C.KEY_CONFIG_EPOCH: req.config_epoch, C.KEY_VERSION: req.version,
            C.KEY_SEED: req.seed, C.KEY_LAM: req.lam, C.KEY_EPISODES: req.episodes,
            C.KEY_MAX_STEPS: req.max_steps, C.KEY_RES_TOKEN: req.res_token,
        }, self._recv_timeout_s)
        echoed_epoch = int(reply[C.KEY_CONFIG_EPOCH])
        echoed_version = int(reply[C.KEY_VERSION])
        if echoed_epoch != req.config_epoch or echoed_version != req.version:
            raise ControlError(
                ERR_BAD_REPLY,
                f"generate reply echoed (epoch={echoed_epoch}, version={echoed_version}) but the request "
                f"was (epoch={req.config_epoch}, version={req.version}) — control-channel desync")
        return GenerateResult(written=int(reply[C.KEY_WRITTEN]),
                              config_epoch=echoed_epoch, version=echoed_version)

    def ping(self) -> PingResult:
        reply = self._request({C.KEY_TYPE: C.MSG_PING}, self._ready_timeout_s)
        return PingResult(serving=bool(reply[C.KEY_SERVING]),
                          config_epoch=int(reply[C.KEY_CONFIG_EPOCH]))

    def close(self) -> None:
        """Graceful-then-forceful reap (idempotent — safe on every exit path). Send `shutdown` and await
        the ack on a bounded recv; then escalate to terminate/kill so a wedged runner is never waited on
        indefinitely. Mirrors the inference server's stop->join->close sequence (ADR-0002 / P5)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._send({C.KEY_TYPE: C.MSG_SHUTDOWN})
            self._recv(self._ready_timeout_s)  # best-effort ack; a timeout/EOF here just reaps below
        except ControlError:
            pass  # the runner may already be gone or wedged — the reap below is the real teardown
        self._reap()

    def _reap(self) -> None:
        """Bounded process teardown: close stdin, then SIGTERM, wait briefly, SIGKILL if still alive,
        then reap. The parent never waits unbounded (the §5.5 discipline)."""
        proc = self._proc
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass  # nothing more we can do without blocking the parent forever

    def __enter__(self) -> "SubprocessActorTransport":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
