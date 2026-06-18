#!/usr/bin/env python3
"""
chocofarm/az/cpp_executor.py — the C++ Gumbel ACTOR as an exit_loop generation executor.

`CppActorExecutor` satisfies the SAME executor contract `exit_loop.run` drives `ParallelExecutor`
through (`generate` / `evaluate` / `close`, plus `.run` / `.cores`), so the C++ Gumbel self-play actor
can be SWAPPED into exit_loop's GENERATION step with ZERO change to the loop's orchestration: the
held-out eval, the replay window, the JAX/optax training, the per-iteration checkpointing, the
TensorBoard streaming, and the hp registry are all inherited unchanged. This is the "swap the C++ actor
into exit_loop's generation" path; it SUPERSEDES the minimal standalone `cpp_actor_loop.py` (which
re-implemented a bare generate→train→publish loop) for the full ExIt run.

Division of labour — the C++ actor owns GENERATION, exit_loop owns the rest (the env<->actor seam):
  * generate(): publishes the frozen net to redis (the SAME `az:w:<run>:<phase>:<version>` weight seam
    the runner reads via `read_weights`/`NetForward`), drives a PERSISTENT `chocofarm-cpp-runner --serve`
    over the ActorTransport (a `configure` when the live search knobs changed — m/n_sims/c_* are HOT, so a
    retune lands WITHOUT a respawn — then a `generate` control message) to play E episodes against those
    weights, and reads the four (X, PI, M, Y) float32 result blocks back into exit_loop's flat
    `list[_Record]`. The value target is the actor's own pure-MC λ-return.

The WIRE generation path (docs/design/cpp-wire-generation-roadmap.md, Phase D+E): when `pool_batch>0`, the
executor ALSO stands up an in-process JAX `InferenceServer` daemon thread over the live net on an `ipc://`
endpoint, and passes `--infer-endpoint`/`--pool-threads`/`--pool-batch` to the `--serve` runner so its
generation resolves every Gumbel-AZ search LEAF REMOTELY on that batched server over a DEALER socket (the
T×K fiber-pool driver, `run_episodes_wire_batched`) instead of a serial local `NetForward`-per-leaf. The
server is the SSOT batched leaf evaluator (the SAME `forward_core` every Python path runs); it serves
GENERATION only (eval stays in-process Python, ADR-0008). The redis weight seam now feeds THIS in-process
server (via `RedisParamsSource`, version-gated), not a C++ local `NetForward`. Weights are published
publish-THEN-bump (publish the blob FIRST, then advance the version the server's reload poll wants — a
missing-blob reload-abort window otherwise, CRITIQUE B2). The pool knobs ride `--serve` STARTUP args, never
`ActorConfig` (their ONE home is `RuntimeConfig` — ADR-0012 P1 / Q6). When `pool_batch==0` (the default),
no server is stood up and the runner uses the serial local-`NetForward` path (binary dispatch).
  * evaluate(): runs exit_loop's OWN greedy `GumbelPolicy` eval on the trained net IN-PROCESS (Python).
    The eval measures the net's greedy rate — a language-agnostic quantity — and "swap into GENERATION"
    leaves eval to the loop. No subprocess, no redis: a pure-Python search over the passed net.

TWO generation-shaping knobs the loop threads into `generate()` are NOT yet wired across the C++ wire;
both FAIL LOUD (ADR-0002) rather than silently producing a different training distribution than the
operator asked for:

  * Part-B value-target blend (`--td-lambda<1` / `--n-step`): the runner emits the pure-MC λ-return only
    (no per-decision root-value bootstrap crosses the wire). The path to honor it is local: expose the
    search's `v_mix` on the C++ `Decision` -> the runner emits a per-decision `boot` block -> `generate()`
    blends via the ONE Python `blended_returns_to_go` (value_target.py — never a second blend transcribed
    in C++). Until then: run pure-MC (the default) with the C++ actor, or `--workers>0` for Part-B.
  * `explore_plies` (default 4): both Python paths sample the EXECUTED action from π′ for the first
    this-many plies (temperature 1) to diversify trajectories (design §6); the C++ Gumbel actor executes
    the Sequential-Halving survivor at temperature 0 EVERY ply, so it cannot honor the exploration prefix.
    `generate()` refuses `explore_plies>0`. The path to honor it: expose a temperature>0 executed-action
    sample on the C++ `Decision`/runner (mirroring `generate_episode`'s `temp = 1.0 if ply < n_explore_plies
    else 0.0`). Until then: pass `--explore-plies 0` to accept greedy (temperature-0) generation with the
    C++ actor, or `--workers>0` for the exploration prefix.

Connection facts: the transport redis (`CHOCO_TRANSPORT_REDIS_*`, default 127.0.0.1:6380 — `config.py`),
the same instance the Python worker pool and the C++ runner use.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import numpy as np

from chocofarm.az.actor_config import ActorConfig
from chocofarm.az.actor_transport import GenerateRequest, SubprocessActorTransport
from chocofarm.az.result_spec import RESULT_DTYPE
from chocofarm.az.transport import RedisTransport, connect, result_keys

if TYPE_CHECKING:
    from chocofarm.az.mlp import ValueMLP
    from chocofarm.az.transport import _Record
    from chocofarm.model.env import Environment

class CppActorExecutor:
    """An exit_loop executor whose GENERATION is the C++ Gumbel actor — a PERSISTENT
    `chocofarm-cpp-runner --serve` driven over the ActorTransport (online-reconfigured when the live HOT
    search knobs change, no respawn) — and whose EVALUATION is exit_loop's own in-process Python
    `GumbelPolicy`. A drop-in for `ParallelExecutor`: the same generate/evaluate/close surface, so
    `exit_loop.run` is oblivious both to which actor produced the transitions and to the transport."""

    def __init__(self, runner_path: str, instance: str, faces: str, env: "Environment",
                 base_seed: int, use_jax_mlp: bool, in_dim: int, n_slots: int,
                 gen_timeout_s: int = 3600, pool_threads: int = 0, pool_batch: int = 0) -> None:
        self.runner = runner_path
        self.instance = instance
        self.faces = faces
        self.env = env
        self.base_seed = int(base_seed)
        # m/n_sims are HOT (they ride the per-iteration hot_search into the ActorConfig the `configure`
        # control message carries — see _actor_config / reconfigure), so they are not frozen ctor state.
        # use_jax_mlp is RESTART (a Python-side forward selector the C++ actor never consumes).
        self.use_jax_mlp = bool(use_jax_mlp)
        self.in_dim = int(in_dim)
        self.n_slots = int(n_slots)
        self.gen_timeout_s = int(gen_timeout_s)
        self.run = uuid.uuid4().hex[:12]            # namespace this run's redis keys
        # the WIRE generation knobs (--serve STARTUP args, never ActorConfig — their ONE home is
        # RuntimeConfig, P1). pool_batch>0 turns the wire path ON: the executor stands up an in-process
        # JAX InferenceServer and the runner resolves every leaf remotely over it. pool_batch==0 (default)
        # = the serial local-NetForward path (no server). The endpoint is namespaced by this run.
        self.pool_threads = int(pool_threads)
        self.pool_batch = int(pool_batch)
        self.wire = self.pool_batch > 0
        self.infer_endpoint = f"ipc:///tmp/choco-infer-{self.run}.sock" if self.wire else ""
        # the in-process JAX InferenceServer (the SSOT batched leaf evaluator on the wire path): built in
        # _ensure_actor (lazily, after the first net publish so RedisParamsSource can load it), torn down in
        # close(). `_published_version` is the version supplier RedisParamsSource.poll() reads (publish-then-
        # bump, CRITIQUE B2); start it at -1 so the FIRST generate's publish-then-bump to its version is the
        # initial load. None until the server is up.
        self._server: Any = None
        self._server_thread: Any = None
        self._published_version: int = -1
        self.cores: list[int] = []                  # the runner is one subprocess running episodes
        #                                             serially; no per-worker core pin (unlike the pool)
        self._eval_seed = self.base_seed + 10_000   # fixed eval randomness (only the net changes/iter)
        # connect via the HARDENED transport path (bounded socket timeouts + a fail-loud ping at
        # construction), the same connection discipline ParallelExecutor uses — not a bare redis.Redis.
        self._conn = connect()
        self.transport = RedisTransport(self._conn)
        # the persistent C++ actor (the ActorTransport, actor_transport.py): spawned LAZILY on the first
        # generate (AFTER the fail-loud guards pass), so constructing the executor — and exercising those
        # guards — needs no built binary. Held across generations so a HOT reconfigure (m/n_sims/c_*)
        # rebuilds the runner's policy without a respawn. `_cur_config`/`_cur_epoch` track the live config
        # and the runner-assigned epoch, so `configure` is sent only when the projected ActorConfig changes.
        self._actor: SubprocessActorTransport | None = None
        self._cur_config: ActorConfig | None = None
        self._cur_epoch: int = 0

    def generate(self, net: "ValueMLP", version: int, worlds: list[int], lam: float,
                 explore_plies: int, lam_blend: float, n_step: int | None,
                 hot_search: dict[str, Any] | None = None,
                 max_steps: int = 40) -> list["_Record"]:
        """Publish `net` at `("gen", version)`, drive the persistent C++ Gumbel actor (configure-on-change
        + a generate control message) to play `len(worlds)` episodes against it, and read the (X, PI, M, Y)
        blocks back as a flat `list[_Record]`. The C++ runner draws its OWN per-episode worlds (seeded from
        `base_seed + version`), so `worlds` is used only for its COUNT — the actor's reproducibility rides
        its seed, not the parent's world list."""
        # Part-B guard (ADR-0002): the runner emits the pure-MC λ-return only — refuse a blend it cannot
        # honor rather than silently training on the wrong target. (td_lambda is a schema-validated float
        # in [0,1], never None — so `< 1.0` is the whole blend region; cf. exit_loop.py's sibling checks.)
        if (n_step is not None) or (lam_blend < 1.0):
            raise RuntimeError(
                "CppActorExecutor: the C++ actor emits the pure-MC value target; the Part-B blend "
                f"(td_lambda={lam_blend}, n_step={n_step}) does not yet cross the C++ wire. Use the "
                "Python pool (--workers>0) for Part-B, or pure-MC (the default) with the C++ actor.")
        # explore_plies guard (ADR-0002): the C++ actor plays the temperature-0 SH survivor every ply, so
        # it cannot honor the temperature-1 exploration prefix both Python paths apply. Refuse it loudly
        # rather than silently generating zero-exploration self-play (the SAME fail-loud standard Part-B
        # sets — see the module docstring's known-deferred list).
        if explore_plies and explore_plies > 0:
            raise RuntimeError(
                "CppActorExecutor: the C++ Gumbel actor executes the Sequential-Halving survivor at "
                f"temperature 0 every ply; explore_plies={explore_plies} (sample the executed action from "
                "π′ for the first this-many plies, design §6) does not cross the C++ wire. Pass "
                "--explore-plies 0 to accept greedy (temperature-0) generation with the C++ actor, or use "
                "the Python pool (--workers>0) for the exploration prefix.")
        hs = dict(hot_search) if hot_search else {}
        # PUBLISH-THEN-BUMP (CRITIQUE B2): write the net blob to redis FIRST, THEN advance the version the
        # wire server's RedisParamsSource.poll() wants. The reverse order opens a window where poll() wants a
        # version whose blob is not yet written -> read_weights raises -> a loud reload-abort mid-generate.
        # On the SERIAL path (no server) the bump is harmless bookkeeping. The server is built (lazily, in
        # _ensure_actor) AFTER this first publish-then-bump, so its RedisParamsSource initial-load finds the
        # blob; subsequent generates' bumps are picked up by the server's between-batch reload poll.
        self.transport.publish_weights(net, "gen", version, self.run)
        self._published_version = version
        tok = f"{self.run}-gen-{version}"
        n_eps = len(worlds)
        # Drive the persistent runner over the ActorTransport. Adopt the live config FIRST: project
        # hot_search + instance/faces into the ActorConfig and `configure` ONLY when it changed — the
        # runner rebuilds its policy live on a HOT change (m/n_sims/c_*) without tearing down the env; an
        # instance change is a loud reject. The runner assigns the config_epoch, carried back in the
        # generate reply (the two-gate desync check, §2.2). The per-generation scalars (version/seed/lam/
        # episodes/max_steps/res_token) ride the message; the version->seed derivation is the determinism
        # anchor (§9), never sticky config, and the runner reloads weights when `version` advances.
        actor = self._ensure_actor()
        cfg = self._actor_config(hs)
        if cfg != self._cur_config:
            self._cur_epoch = actor.reconfigure(cfg)
            self._cur_config = cfg
        result = actor.generate(GenerateRequest(
            config_epoch=self._cur_epoch, version=version, seed=self.base_seed + version, lam=lam,
            episodes=n_eps, max_steps=max_steps, res_token=tok))
        records, n_found = self._read_records(tok, n_eps)
        # Reconcile what we READ against what the runner reports it WROTE (the structured `written` reply —
        # the typed replacement for the old `wrote N episode` stderr scrape, always present now). The
        # transport redis is LRU-evicting, so a result blob can be evicted between the runner's write and
        # the parent's read under memory pressure; a non-empty-requested generation must never collapse to
        # a smaller buffer without a loud failure (ADR-0002 — the same guard the Python pool gets via its
        # structural meta channel).
        if n_found != result.written:
            raise RuntimeError(
                f"CppActorExecutor: read {n_found} non-empty episode block(s) but the runner reported "
                f"writing {result.written} at gen version {version} — result blob(s) went missing (LRU "
                "eviction under transport memory pressure?). Refusing to train on a silently-shrunk buffer.")
        return records

    def _ensure_actor(self) -> SubprocessActorTransport:
        """Spawn the persistent C++ `--serve` runner on first use (LAZY — so constructing the executor
        and firing the fail-loud guards above needs no built binary) and probe its readiness with a ping.
        The spawn + ping are the loud-at-first-generate readiness check: a missing binary, or a runner
        that died on startup (e.g. redis unreachable), raises a `ControlError` here, not a silent hang
        (ADR-0002 / P5). The runner is held across generations (the no-respawn win).

        On the WIRE path (self.wire), an in-process JAX `InferenceServer` daemon thread is stood up FIRST
        (BEFORE the runner's first ping) over the live net via a version-gated `RedisParamsSource` on the
        run-namespaced `ipc://` endpoint; the runner is then spawned with `--infer-endpoint`/`--pool-threads`
        /`--pool-batch` so its generation resolves every leaf remotely on that server. The server is built
        AFTER the first publish-then-bump (its RedisParamsSource initial-load needs the blob), so this is
        called from generate() only after `self.transport.publish_weights` + `self._published_version=...`."""
        if self._actor is None:
            extra = ["--run", self.run]
            if self.wire:
                self._start_server()
                extra += ["--infer-endpoint", self.infer_endpoint,
                          "--pool-threads", str(self.pool_threads),
                          "--pool-batch", str(self.pool_batch)]
            actor = SubprocessActorTransport(self.runner, recv_timeout_s=self.gen_timeout_s,
                                             extra_args=tuple(extra))
            actor.ping()  # readiness: the runner spawned + speaks the protocol (serving=False pre-configure)
            self._actor = actor
        return self._actor

    def _start_server(self) -> None:
        """Stand up the in-process JAX `InferenceServer` daemon thread (the SSOT batched leaf evaluator for
        the wire path) over the live net via `RedisParamsSource(self._conn, self.run, "gen", lambda:
        self._published_version, initial_version=self._published_version)` on `self.infer_endpoint`. Built
        AFTER the first publish-then-bump (so RedisParamsSource's initial load finds the blob). SINGLE-
        THREADED server (JAX/XLA owns the forward — the R14 / jaxtrain-deadlock invariant; no XLA in a worker
        thread). Asserts the OR-4 geometry invariant: the server's in_dim/n_actions (derived from the SAME
        self.env the C++ actor's instance/faces describe) MUST equal the actor's fb.dim()/n_slots — a
        mismatch is a ragged-batch loud reject downstream, so fail loud HERE at standup (ADR-0002)."""
        import threading

        from chocofarm.az.actions import n_action_slots
        from chocofarm.az.features import feature_dim
        from chocofarm.az.inference_server import InferenceServer, RedisParamsSource

        # OR-4 (CRITIQUE B3): the server's geometry must match the C++ actor's (same instance/faces -> same
        # feature dim / action-slot count). self.env is the SAME Environment the executor derived in_dim/
        # n_slots from at construction; assert it loudly before any leaf crosses the wire.
        srv_in_dim, srv_n_actions = feature_dim(self.env), n_action_slots(self.env)
        if srv_in_dim != self.in_dim or srv_n_actions != self.n_slots:
            raise RuntimeError(
                f"CppActorExecutor: inference-server geometry (in_dim={srv_in_dim}, n_actions="
                f"{srv_n_actions}) != actor geometry (in_dim={self.in_dim}, n_slots={self.n_slots}). The "
                "Python self.env and the C++ actor's instance/faces must describe the SAME env (OR-4) — a "
                "mismatch is a ragged-batch / dimension-corruption hazard. Refusing to serve.")
        src = RedisParamsSource(self._conn, self.run, "gen",
                                version_supplier=lambda: self._published_version,
                                initial_version=self._published_version)
        self._server = InferenceServer(src, bind=self.infer_endpoint, max_batch=256)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()

    def _actor_config(self, hs: dict[str, Any]) -> ActorConfig:
        """Project the live hot_search bag (the per-iteration HOT search knobs) + the runner's instance/
        faces into the ActorConfig the `configure` message carries (the Port/ACL — the executor is the
        boundary that projects the hp-derived knobs into the C++ actor's config). All 7 GumbelConfig knobs
        must be present (exit_loop's hot_search provides them); a missing one is a loud failure, never a
        silent default that would train under a different search than the operator set (ADR-0002)."""
        try:
            return ActorConfig(
                instance_path=self.instance, faces_path=self.faces,
                m=int(hs["m"]), n_sims=int(hs["n_sims"]), c_puct=float(hs["c_puct"]),
                c_visit=float(hs["c_visit"]), c_scale=float(hs["c_scale"]),
                c_outcome=int(hs["c_outcome"]), max_depth=int(hs["max_depth"]))
        except KeyError as e:
            raise RuntimeError(
                f"CppActorExecutor: hot_search is missing the search knob {e} — the C++ actor's ActorConfig "
                "needs all of m/n_sims/c_puct/c_visit/c_scale/c_outcome/max_depth (exit_loop's per-iteration "
                "hot_search provides them; a partial bag cannot configure the runner).") from e

    def _read_records(self, tok: str, n_eps: int) -> tuple[list["_Record"], int]:
        """Read each episode's four float32 blocks (deriving n from the Y block length — the runner
        skips empty episodes), unstack into per-decision (feat, pi, mask, g) records, and DELETE the
        keys (raw bytes can be large; don't leak across iterations). Returns (records, n_found) where
        n_found is the number of NON-EMPTY episodes actually read — the caller reconciles it against the
        runner's reported written count. Mirrors transport.read_and_delete_results' decode, but derives n
        per episode (the runner doesn't return per-episode metas through a pipe the way the pool does)."""
        records: list["_Record"] = []
        n_found = 0
        dt = RESULT_DTYPE
        conn = self._conn
        for idx in range(n_eps):
            xk, pik, mk, yk = result_keys(tok, idx)
            xb, pib, mb, yb = conn.get(xk), conn.get(pik), conn.get(mk), conn.get(yk)
            if yb is None or xb is None or pib is None or mb is None:
                # an empty episode (the runner wrote nothing) OR an evicted blob (any of the FOUR
                # independent TTL'd keys can LRU-evict) — counted as not-found and reconciled by the
                # caller's written-vs-n_found check. Guarding ALL four (not just X/Y) keeps that clean
                # diagnostic instead of an opaque np.frombuffer(None) TypeError on an evicted PI/M (ADR-0002).
                continue
            n = len(yb) // dt.itemsize
            if n == 0:
                continue
            X = np.frombuffer(xb, dtype=dt).reshape(n, self.in_dim)
            PI = np.frombuffer(pib, dtype=dt).reshape(n, self.n_slots)
            M = np.frombuffer(mb, dtype=dt).reshape(n, self.n_slots)
            Y = np.frombuffer(yb, dtype=dt)
            for i in range(n):
                records.append((X[i], PI[i], M[i], float(Y[i])))
            n_found += 1
            conn.delete(xk, pik, mk, yk)
        return records, n_found

    def evaluate(self, net: "ValueMLP", version: int, worlds: list[int], lam: float,
                 hot_search: dict[str, Any] | None = None,
                 max_steps: int = 40) -> tuple[float, float, list[float]]:
        """exit_loop's own held-out eval, run IN-PROCESS: the greedy argmax-π′ `GumbelPolicy` on the
        trained net over the held-out `worlds` (drawn by the parent from the HOT eval_seed and passed in),
        at fixed λ. Returns (totR, totT, list_of_T). The search RNG uses an INDEPENDENT, construction-fixed
        seed (base_seed+10000), fixed across iterations so only the net changes — a valid estimate of the
        net's greedy rate, but NOT bit-for-bit comparable to a serial/eval_seed-driven eval of the same
        checkpoint. (This matches the Python WORKER POOL's eval, which also folds base_seed and ignores the
        HOT eval_seed; the serial path is the one that tracks eval_seed exactly.)"""
        from chocofarm.az.gumbel_search import GumbelPolicy
        hs = dict(hot_search) if hot_search else {}
        pol = GumbelPolicy(net, self.env, use_jax_mlp=self.use_jax_mlp, **hs)
        rng = np.random.default_rng(self._eval_seed)
        totR = totT = 0.0
        ets: list[float] = []
        for w in worlds:
            R, T, _ = self.env.simulate(pol, int(w), lam, rng, max_steps=max_steps)
            totR += R
            totT += T
            ets.append(T)
        return totR, totT, ets

    def close(self) -> None:
        # reap the persistent actor (graceful shutdown then bounded SIGTERM/SIGKILL — actor_transport),
        # THEN tear down the in-process wire server (stop -> join -> close, the clean shutdown sequence the
        # InferenceServer's bounded-poll loop wants), THEN close redis. The actor is reaped first so no leaf
        # is in flight to the server when it stops (the control channel is lock-step — the generate reply is
        # already received, CRITIQUE B4 — so this ordering is tidy shutdown, not mid-generate-block
        # avoidance). All paths are best-effort + idempotent (close runs on every exit path).
        if self._actor is not None:
            try:
                self._actor.close()
            except Exception:
                pass
        if self._server is not None:
            try:
                self._server.stop()
                if self._server_thread is not None:
                    self._server_thread.join(timeout=5.0)
                self._server.close()
            except Exception:
                pass
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "CppActorExecutor":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False
