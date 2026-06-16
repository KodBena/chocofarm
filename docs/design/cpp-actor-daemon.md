<!-- docs/design/cpp-actor-daemon.md -->

# Daemonizing the C++ self-play actor: a ZeroMQ-controlled, JSON-configured runner

**Status:** Design record (forward-looking, contracts-first). No code is committed; this is the
artifact the maintainer reviews before any implementation begins. It is the synthesis of a proposer
draft and an adversarial critique against the actual interfaces — every load-bearing claim below is
cited to the file and line it was verified at, and where a draft claim was *wrong against the code* the
correction is called out inline so a later reader does not re-import the error. Read end to end before
implementation (ADR-0002 doc-consumption discipline).

Public Domain (The Unlicense).

---

## Maintainer addendum — decision & disposition (parked, with reap list)

> Appended by the maintainer (Claude Opus 4.8) after producing this note, recording the decision taken in
> session. **This note is design-on-file: the daemon itself is PARKED (implementation deferred), and the
> transport-agnostic portions are REAPED into the current roadmap.** The note was commissioned under a
> mandate to *implement* a daemon, so everything below §0 argues the daemon as a committed design; this
> addendum is the decision overlay that supersedes that framing on the points it names.
>
> **Decision: build the config SSOT + a transport SEAM (subprocess-first) + the parity work NOW; DEFER the
> daemon implementation** until the continuous async actor (Shape C, `scaling-and-cpp-seam.md`) is the
> actual consumer. Rationale (ADR-0012 P7, serialization ⊥ transport): the config schema is the durable
> SSOT; the transport (subprocess pipe vs ZMQ daemon) is a swappable detail behind a seam. The daemon's
> unique wins are marginal under the current synchronous per-iteration loop — the no-restart saving is a
> sub-second process spawn against minutes of generation (measure-first, ADR-0009), and a persistent
> eval/transposition cache gains ~nothing across generates because the net is retrained every iteration
> (the cache is valid only within one net version — intra-search / intra-generate, which the subprocess
> already gets). Building the daemon now is ahead of the consumer (the built-ahead-of-consumer /
> YAGNI-as-a-tell shape the ADRs name).
>
> **REAP NOW (the transport-agnostic portions kept for the current roadmap):**
> - **§3 — the JSON config SSOT.** The priority. Built as the `ActorConfig` projection of
>   `ExperimentConfig` (§3.2) so it is NOT a third transcription of the hp-schema / GumbelConfig /
>   RunnerConfig knobs; both sides derive + validate at the boundary (§3.3), drift-net'd. Delivered to the
>   **subprocess** first (a single `--config <redis-key>` / blob retiring the ~15 flags + the stderr
>   regex), NOT over a socket yet.
> - **The transport SEAM (an `ActorTransport` port), subprocess-FIRST.** §7.1's executor-as-client
>   inversion is reaped as the seam — but this **supersedes §7.2**: the subprocess path is **kept as the
>   first `ActorTransport` impl** (it is the active transport, not a silent fallback), with the ZMQ daemon
>   as the deferred second impl behind the same `run(config, token) -> result` interface. Getting that
>   interface shape right (stateless spawn and stateful daemon both satisfy it without leaking lifecycle
>   upward) is what lets the daemon drop in later with no churn to `exit_loop` / the executor.
> - **§8 — the parity work** (its own thread): `explore_plies` (§8.1) + the Part-B blend (§8.2). Heed
>   §8.2's corrected contract (critique B1): the BOOT block carries the **root search value**
>   (`_root_search_value`, the visit-weighted root return), **NOT** `v_mix`, and the C++ `Decision` exposes
>   *neither* today — so the parity work adds that exposure. Account for §4.1 / critique B2: the 5th BOOT
>   block is ~7-site fixed-arity surgery, not a free seam extension.
> - **The eval / transposition cache — built in the C++ search, decoupled from the daemon** (not
>   §-scoped here). It helps intra-search / intra-generate for *either* transport; it is not a daemon
>   justification.
> - **§4 — the redis raw-bytes weight/result seam stays** (unchanged for both transports); **§9's
>   determinism property** (`fold_seed` pure + fresh-per-episode, independent of process lifetime) is the
>   bar the subprocess already meets and the daemon must preserve.
>
> **PARKED (the daemon implementation, awaiting the Shape-C consumer) — this note is its spec:** §1
> (ROUTER/DEALER transport), §2 (the control protocol), §5 (daemon lifecycle / spawn / readiness /
> reaping), §6 (reconfigure-without-restart over the socket), and the §10 daemon-rollout steps. When Shape
> C lands, this note is the contracts-first starting point; revisit §11's open items (the first C++
> server-bind surface §11.1, the arity refactor §11.2) then.

---

## Build amendment — §2 control protocol + §6 reconfigure BUILT (2026-06-16)

> Appended by the implementation thread after building the reaped roadmap. Per ADR-0005 Rule 8 (amend
> point-in-time records by append, never silently rewrite), this records what was actually built — so a
> reader does not re-import the addendum's "parked" status for the pieces now in the tree. It does NOT
> rewrite the addendum above; it corrects its disposition on the points named.
>
> **Decision overlay (the maintainer's directive).** Online reconfiguration of the running actor —
> driving the HOT search knobs (and, by this directive, the now-HOT `m`/`n_sims`) via the hp interface
> WITHOUT destroying the runner context — is the **primary motivation**, not a deferred daemon nicety.
> So §2 (the control protocol) and §6 (reconfigure-without-restart) were **UN-PARKED and built** — over a
> **subprocess-pipe `ActorTransport`**, not the §1 ZMQ ROUTER/DEALER daemon, which remains the deferred
> Shape-C transport behind the same seam (P7: serialization ⊥ transport — the control protocol is the
> SSOT, the mechanism is swappable).
>
> **Corrections to the addendum's technical claims.** The §6/B3 "`m`/`n_sims` MUST be RESTART" claim was
> **wrong against the code**: the SH bracket is recomputed per `decide()` (read identically to the HOT
> `c_puct`), and ADR-0012's own C++ guidance already names `m`/`n_sims` live per-decision scalars (P4).
> They were reclassified **RESTART→HOT** in the hp schema. Consequently `ActorConfig` (§3) **excludes**
> `use_jax_mlp` (a Python-side forward selector the C++ runner never consumes) and carries **no RESTART
> field at all** — only INSTANCE (instance/faces) and HOT (the 7 GumbelConfig knobs); the §3.1 field set
> here is therefore narrower than drafted.
>
> **What landed** (branch `cpp-actor-online-reconfig`):
> - the hp schema flip + the per-iteration `hot_search` flow for `m`/`n_sims` across the serial / pool /
>   C++ generation paths (no frozen ctor copies — P4).
> - `chocofarm/az/actor_config.py` + `control_spec.py` (+ `cpp/include/chocofarm/actor_config.hpp`,
>   `control_spec.hpp`), drift-netted in `tests/test_wire_drift.py` (field-set, Mut-class read from
>   `schema.py`, message/error-tag vocabulary, each with a negative-mutation self-check).
> - `chocofarm/az/actor_transport.py` — the `ActorTransport` Port + `SubprocessActorTransport` (bounded
>   recv = the pipe analog of `ZMQ_RCVTIMEO`), tested against a fake runner (`tests/test_actor_transport.py`).
> - `cpp/src/serve.cpp` — the `--serve` control loop; `actor_config_from_json` (validate-don't-coerce);
>   `run_episodes` extracted from `run()` (P1). It reuses the proven episode loop; the two gates; the loud
>   `instance_knob_changed` reject.
> - `CppActorExecutor` rewired onto the Port (its `exit_loop` contract unchanged; `written` from the
>   structured reply, retiring the stderr scrape). Online reconfig verified end-to-end against the real
>   actor + redis (`test_cpp_serve_online_reconfiguration`, `test_cpp_actor_executor_drives_persistent_runner`).
>
> **Still deferred (unchanged):** §8.2 Part-B + §8.1 `explore_plies` (the executor still loud-refuses
> them — a genuine C++-search dependency, not transport); §1 (ROUTER/DEALER), §5 (daemon lifecycle), and
> the §10 daemon-rollout steps remain the Shape-C spec this note holds.

---

## 0. Problem statement — what is being replaced and why

`chocofarm/az/cpp_executor.py::CppActorExecutor` is an `exit_loop` generation executor whose self-play
is the C++ Gumbel actor. Per `generate()` it (i) publishes the frozen net to the transport redis
(`publish_weights`, `cpp_executor.py:124`), (ii) **subprocesses** `chocofarm-cpp-runner --policy gumbel`
with ~15 CLI flags (`cpp_executor.py:127-134`), (iii) blocks on `subprocess.run(..., timeout=
gen_timeout_s)` (`:135`), (iv) parses `"wrote (\d+) episode"` from the child's **stderr** (`:147`) to
reconcile against what it read back, and (v) reads the four (X, PI, M, Y) float32 result blocks out of
redis (`_read_records`, `:161-188`). The C++ side is `cpp/src/main.cpp` (the CLI ACL,
`main.cpp:101-224`) → `cpp/src/runner.cpp::run` (`runner.cpp:132`) → `GumbelAZPolicy`
(`cpp/include/chocofarm/gumbel.hpp:136`).

This shape is replaced for the maintainer's stated reasons:

- **A fresh process spawn per generation** (per ExIt iteration, E≈300 episodes) pays instance load
  (`load_instance`, `main.cpp:132`), redis connect (`:140`), net read + `NetForward::create`
  (`:181-191`) on **every** iteration, and orchestrates the child over pipes + a **stderr regex**.
- **Configuration is scattered across CLI flags.** The control surface is the argv list
  (`main.cpp:107-179`); a new knob is a new flag on both sides, with no single documented schema.
- **The runner cannot adopt a new configuration without restarting.** Even the live-retunable HOT search
  knobs are re-passed as fresh argv on a fresh process each generation.

The replacement: the C++ actor runs as a **daemon** listening on a ZeroMQ socket; the Python executor
becomes a **client** that sends it a JSON control message per generation. Configuration is consolidated
into a JSON schema with **one in-repo home** that both sides derive from and validate against (Port/ACL,
ADR-0012 P2). The `generate`/`evaluate`/`close` executor contract is unchanged, so `exit_loop.run` stays
oblivious to the transport switch.

**Scope fence (load-bearing).** This note daemonizes the **synchronous, lock-step** actor — one client,
one in-flight `generate` at a time, the same fan-out shape the subprocess had. It does **not** build the
async work-stealing / multi-tree-multiplexer machinery. `docs/design/cpp-search-runtime.md` establishes
that `{scheduler, transport}` is a **matched pair** (its §0: a blocking `predict` cannot run inside an
async continuation) and that the DEALER/`FiberMuxRuntime` embodiment is gated behind an explicit
ADR-0009 measure-first benchmark (its §6-Q5, §8.5). This daemon is the actor-control transport *below*
that question and must not pre-decide it. The relationship is recorded in §11.6; the async restructure
stays out of scope.

---

## 1. Transport shape

### 1.1 ROUTER (daemon) / DEALER (client), not REP/REQ

**Decision: the daemon binds a ZeroMQ `ROUTER`; the executor connects a `DEALER`.** This mirrors
`inference_server.py:245` (`self._ctx.socket(zmq.ROUTER)`), so the codebase has **one daemon idiom**,
not two. The justification is *not* speculative async (the project's handoff names the
"proportionate-to-future-work" justification shape as the tell ADR-0012 P7 forbids); it is two concrete
present-tense properties:

1. **Liveness during a long call.** A `generate` runs for minutes (E≈300 episodes). With REP/REQ the
   strict request→reply state machine forbids the daemon emitting anything between the request and its
   single reply. ROUTER/DEALER lets either side send a frame at any time, which the §5.3 readiness probe
   uses (an early `serving:false` / error reply).
2. **One idiom, already battle-tested.** The greedy-drain ROUTER loop and its bounded-poll shutdown
   (`inference_server.py:252-322`) are the project's proven non-spinning, cleanly-stoppable daemon loop.
   Reusing the socket type lets the actor daemon reuse that loop's *shape* — the bounded poll that lets a
   `stop` flag wake the loop without closing the socket from another thread
   (`inference_server.py:266-275`) — without inventing a second one.

The DEALER carries no correlation-id pool in *this* note — the client issues one request and awaits one
reply (lock-step). The corr-id machinery the async embodiment needs (`cpp-search-runtime.md` §4) is
deliberately not built here; the ROUTER's opaque-envelope echo (`inference_server.py:284-296`
round-trips `frames[1:-1]` verbatim) means a future corr-id is an envelope frame the daemon never has to
parse — the transport is *forward-compatible* with it without *committing* to it.

### 1.2 Endpoint: `ipc` default, `tcp` override

**Decision: `ipc:///tmp/chocofarm-actor-<run>.sock` by default; `tcp://127.0.0.1:<port>` available by
config.** `ipc` is faster (no TCP stack), needs no port allocation, and is collision-free by
construction because the `<run>` id namespaces the path (§5.4). `tcp` is retained for the case the daemon
and executor are on different hosts — not a present need, but a one-line config switch, not a redesign.

**Why `/tmp` is acceptable here (preempting the reviewer reflex).** CLAUDE.md's operational facts
distrust `/tmp` for **experiment output** ("Never discard experiment output — preserve it under
`~/w/vdc`, not `/tmp`"). The control socket is **not** experiment output: it is an ephemeral coordination
artifact with no record value, recreated every run, meaningless after the daemon exits. `/tmp` is the
correct home for exactly that class of file. (The experiment records — checkpoints, `.log`, TensorBoard —
continue to live under `~/w/vdc` per the unchanged executor/loop path.)

---

## 2. The control protocol

### 2.1 Message types

JSON request envelope, one type per message, dispatched on a `"type"` field:

| type | direction | purpose | built? |
| --- | --- | --- | --- |
| `configure` | client → daemon | adopt a new `ActorConfig` (§3); rebuild policy/net per the Mut class (§6) | yes |
| `generate` | client → daemon | play E episodes against the current config + a per-call `(version, seed, lam, episodes, res_token)`; reply the structured meta | yes |
| `ping` | client → daemon | readiness / liveness probe; reply carries the daemon's `serving` state + the active `config_epoch` (§5.3) | yes |
| `shutdown` | client → daemon | graceful exit (close sockets, return) | yes |
| `evaluate` | — | **reserved, not built** | no |

`evaluate` stays **reserved-not-built**: `CppActorExecutor.evaluate` runs exit_loop's own in-process
Python `GumbelPolicy` (`cpp_executor.py:190-212`) and never touches the C++ side. Eval measures the
net's greedy rate — language-agnostic — and "swap into GENERATION" leaves eval to the loop. The daemon
reserves the message name so a future C++ eval is an additive type, not a protocol break.

### 2.2 Request / reply shapes and the two independent gates

A `generate` request:

```json
{ "type": "generate", "config_epoch": 7, "version": 42, "seed": 49,
  "lam": 0.0855, "episodes": 300, "max_steps": 40, "res_token": "<run>-gen-42" }
```

A `generate` reply (the structured meta that **replaces the stderr regex**):

```json
{ "ok": true, "written": 300, "config_epoch": 7, "version": 42 }
```

**Two gates, independent — stated crisply to forbid a wording bug the draft risked (critique M5).**

- **`config_epoch` gates *config adoption*.** The daemon increments an epoch counter each time it
  successfully adopts a `configure`. A `generate` carries the epoch the executor believes is live; a
  mismatch is a **loud reject** (`{"ok": false, "error": "config_epoch_mismatch", ...}`) — the daemon
  refuses to generate under a config the client did not think was active. The executor sends `configure`
  only when its projected `ActorConfig` (§3) *changed*, so the epoch advances rarely.
- **`version` gates *weight reload*, and is independent of the epoch.** The net version is carried **per
  `generate`** (it changes almost every iteration — new trained weights), not in the config. The daemon
  reloads weights from redis whenever `version` advances, **regardless of the epoch**. The normal path —
  new weights, unchanged search knobs — is a **new-version / same-epoch** `generate`, and it **must not
  be rejected**. The epoch echo + version echo in the reply let the executor assert both round-trips
  matched (§11.4).

This is the explicit correction of the draft's epoch-gate language, which could be read as rejecting a
legitimate new-version/same-epoch generate. Epoch and version gate different things; the common case
advances only the version.

### 2.3 Error replies

Every reply carries `"ok": bool`. On failure: `{"ok": false, "error": "<machine_tag>", "detail":
"<human string>"}`. The boundary **validates, does not coerce** (ADR-0002 / ADR-0012 P5, Port/ACL): a
malformed control message — unknown `type`, missing required field, a config that fails the §3 schema, an
out-of-class live knob (§6), a `config_epoch_mismatch` — is a **loud structured rejection**, never a
silent default. The error tags are a closed set the drift net pins (§10.3) so the executor can branch on
them without string-matching prose. This is the daemon's analog of the inference server's loud reject
(`inference_server.py:294`, "does not coerce it into a zero-filled forward") and the runner CLI's loud
unknown-policy abort (`main.cpp:194-197`).

### 2.4 How a long `generate` is handled on the socket

The `generate` reply is **blocking from the executor's view** — it issues one DEALER send and one recv,
exactly as the lock-step subprocess blocked on `subprocess.run` (`cpp_executor.py:135`). The non-hang
discipline is a bounded mechanism, not an indefinite block:

**A bounded `ZMQ_RCVTIMEO` on the client recv**, mirroring `zmq_net_client.cpp:50-57` (the C++ client's
`ZMQ_RCVTIMEO` bound that turns a server-down into a typed timeout, not a forever-block) and
`transport.connect`'s `socket_timeout` discipline (`transport.py:128-147`, "a bounded timeout turns a
stall into a loud `redis.TimeoutError` … instead of a silent permanent hang"). The timeout is sized as
`gen_timeout_s` was (default 3600s, `cpp_executor.py:75`) — generous, because a legitimate E=300
generation is minutes, but **finite**. On expiry the executor raises loudly and reaps the daemon (§5).

**No redis heartbeat in v1 (correcting the draft's design, critique N3).** The draft put a heartbeat key
`az:actor:<run>:hb` on redis, but **the lock-step executor is blocked in the recv and cannot read it
concurrently** on the same thread — as drafted, nothing consumes it. **Resolution: the heartbeat is
dropped from v1.** The bounded `ZMQ_RCVTIMEO` *is* the safety net — the same mechanism the C++
`ZmqNetClient` relies on — and it is sufficient: a wedged daemon trips the bound and is reaped. A
heartbeat is reintroduced only if and when an out-of-band watcher exists to read it (e.g. the async
embodiment's separate poll thread); adding a key nothing reads is the complexity-for-a-diagnostic-
nobody-consumes the critique correctly flags. (Judgment call: the cost is that the only signal of a
slow-but-alive daemon is "the recv has not yet returned and the bound has not yet fired" — acceptable
under lock-step, where there is no concurrent reader anyway.)

There is **no progress stream** (no per-episode push). The episode count is reconciled once, in the
single reply's `"written"` field, against what the executor reads back from redis (§4) — the structured
replacement for the `wrote N episode` scrape.

---

## 3. The JSON config SSOT

This is the consolidation the maintainer asked for, and the section the critique found most over-sold.
The corrections (B3, M1, M2) are folded in below.

### 3.1 What the config carries — the full runner-on-par knob set

The control config must carry every knob the runner needs to reach **Python parity**, which is
`RunnerConfig` ∪ `GumbelConfig` ∪ the two parity knobs being added in the parallel thread:

| field | source-of-truth field | Mut class | crosses in |
| --- | --- | --- | --- |
| `instance_path` | `EnvConfig.instance_path` (`schema.py:90`) | **INSTANCE** | `configure` (build-once; live change → loud reject, §6) |
| `faces_path` | derived alongside instance | INSTANCE | `configure` |
| `m` | `SearchConfig.m` (`schema.py:103`) | **RESTART** | `configure` (live change → loud reject, §6) |
| `n_sims` | `SearchConfig.n_sims` (`schema.py:104`) | **RESTART** | `configure` (live change → loud reject) |
| `use_jax_mlp` | `SearchConfig.use_jax_mlp` (`schema.py:110`) | **RESTART** | `configure` (live change → loud reject) |
| `c_puct` | `SearchConfig.c_puct` (`schema.py:105`) | **HOT** | `configure` (live policy rebuild) |
| `c_visit` | `SearchConfig.c_visit` (`schema.py:106`) | HOT | `configure` |
| `c_scale` | `SearchConfig.c_scale` (`schema.py:107`) | HOT | `configure` |
| `c_outcome` | `SearchConfig.c_outcome` (`schema.py:108`) | HOT | `configure` |
| `max_depth` | `SearchConfig.max_depth` (`schema.py:109`) | HOT | `configure` |
| `explore_plies` | `ExItLoopConfig.explore_plies` (`schema.py:201`) | HOT | `configure` (§8.1) |
| `lam_blend` (`td_lambda`) | `ValueTargetConfig.td_lambda` (`schema.py:119`) | HOT | `configure` (§8.2) |
| `n_step` | `ValueTargetConfig.n_step` (`schema.py:120`) | HOT | `configure` (§8.2) |
| `version` | per-generation | — | **per `generate` message**, not the config (§9) |
| `seed` | `base_seed + version` (`cpp_executor.py:130`) | — | **per `generate` message**, not the config (§9) |
| `lam`, `episodes`, `max_steps`, `res_token` | per-generation | HOT/derived | **per `generate` message** |

The split between "config knobs" (carried by `configure`, sticky until changed) and "per-generation
knobs" (`version`, `seed`, `lam`, `episodes`, `res_token`, carried by every `generate`) is the
determinism anchor (§9): the version→seed derivation lives in the **message**, never in cached config.

### 3.2 Where it lives — one home, a projection of the hp schema

**Decision: a new `chocofarm/az/actor_config.py` declaring `ActorConfig`, constructed by a
`from_experiment_config(cfg: ExperimentConfig) -> ActorConfig` projection that holds NO defaults of its
own.** The defaults live exactly where they live today — the `hp(...)` declarations in
`chocofarm/hp/schema.py` (`SearchConfig`, `ValueTargetConfig`, `ExItLoopConfig`, `EnvConfig`). The
projection *reads* them; it never *re-declares* them. The hp registry stays the one config SSOT for the
loop; the daemon config is a **derived view**, not a third writer.

**The honest characterization (correcting the draft's §3.4, critique M2).** The draft claimed
`ActorConfig` "is NOT a third transcription." That defends only against a second *default* writer, which
the projection does handle. But ADR-0012 P7 is about the *layout/field-set*, and there are genuinely
**two hand-authored field-set declarations**: `actor_config.py` (Python) and `actor_config.hpp` (the C++
mirror, §3.3). That is the *same* status as `result_spec.py` / `result_spec.hpp` — which the project
accepts **only because a mechanized drift net pins them** (`result_spec.py:23-25`, the
`test_wire_drift.py` legs). So the honest claim is:

> `ActorConfig` is a **mirrored transcription pinned by the drift net** — the same status `result_spec`
> has — not "not a transcription." Its P1/P7 honesty is *contingent on the §3.3 drift leg existing and
> working.*

This is a weaker claim than the draft made, and it is the correct one.

### 3.3 How both sides derive + validate, and how it is drift-protected

- **Python side (the executor's Port/ACL).** `from_experiment_config` projects the validated
  `ExperimentConfig` (already strict-decoded by `schema.decode_config`, `schema.py:426`) into
  `ActorConfig`, then `json.dumps` it into the `configure` message. The values are already
  domain-validated by `check_invariants` (`schema.py:268`); the projection adds no new defaults.
- **C++ side (the daemon's Port/ACL).** `actor_config.hpp` declares the mirror struct; the daemon parses
  the incoming JSON with **`nlohmann::ordered_json`** — the library `instance.cpp:19,28` already uses
  (`#include <nlohmann/json.hpp>`, `using json = nlohmann::ordered_json`), so the config parser is **no
  new dependency**. The parse **validates, does not coerce** (ADR-0002): a missing required field, a
  wrong type, or an out-of-domain value is a typed `Error` reported as a structured error reply (§2.3) —
  the same boundary discipline `instance.cpp:101-108` uses to translate nlohmann's accessor exceptions
  into typed `Error` at the edge.

**The drift mechanism — corrected to be real, not vapor (critique M1).** The draft said the field-name
leg would parse `actor_config.hpp` "the same way the wire-drift test parses constexpr literals." That is
**false against the test**: `test_wire_drift.py`'s parsers (`_cpp_int_const`/`_cpp_str_const`/
`_cpp_str_array`/`_cpp_int_array`, `test_wire_drift.py:75-105`) extract **named scalar/array literals**
(`name = 42;`, `name = {"X","PI"};`). A C++ `struct ActorConfig { double lam; int m; ... }` is **not** a
literal those regexes can read. So the field-set leg requires the C++ header to **also** declare its field
set as a parseable literal array, exactly the way `result_spec.hpp` declares `BLOCK_ORDER` as a
`std::array<std::string_view, N>` (parsed by `_cpp_str_array`, `test_wire_drift.py:164`).

**Decision: `actor_config.hpp` carries explicit, parseable field-set + Mut-class manifest arrays**
alongside the struct — e.g.

```cpp
// cpp/include/chocofarm/actor_config.hpp  (forward-looking — not built)
inline constexpr std::array<std::string_view, /*N*/> ACTOR_CONFIG_FIELDS = {
    "instance_path", "faces_path", "m", "n_sims", "use_jax_mlp",
    "c_puct", "c_visit", "c_scale", "c_outcome", "max_depth",
    "explore_plies", "lam_blend", "n_step" };
// the live-vs-reject classification (§6), itself drift-checked, in field order:
inline constexpr std::array<std::string_view, /*N*/> ACTOR_CONFIG_MUT = {
    "instance",   "instance",  "restart", "restart", "restart",
    "hot",        "hot",       "hot",     "hot",     "hot",
    "hot",        "hot",       "hot" };
```

and `actor_config.py` exposes the same two ordered sequences — the field names, and each field's Mut
class **read from `schema.py`'s `metadata["mut"]`**, so the Mut classification has **one home** (the hp
schema). The drift net then gets new always-on legs asserting:

1. `_cpp_str_array(actor_config.hpp, "ACTOR_CONFIG_FIELDS") == list(actor_config.FIELD_NAMES)`;
2. `_cpp_str_array(actor_config.hpp, "ACTOR_CONFIG_MUT") == [mut_of(f) for f in FIELD_NAMES]`, where
   `mut_of` reads `schema.py`'s `metadata["mut"]` — so a field that changes Mut class on one side reds;
3. a negative mutation self-check (the `test_wire_drift.py:302-308` `_perturb_cpp_const` pattern,
   generalized to perturb one array entry) proving the leg catches a one-sided drift.

Without that explicit literal array the cross-language config SSOT is **not** drift-protected and P7 is
unmet — so it is part of the deliverable, not an afterthought. (This is also why §3.2's
no-third-transcription claim is *contingent* on this leg: the leg is what earns it.)

### 3.4 What the config does NOT carry

`ActorConfig` carries only the runner-on-par knob set (§3.1). It does **not** carry the learner-side
knobs (`TrainConfig`, `ArchConfig` weight shapes, `BoundsConfig`, `EvalConfig`, `ParallelConfig`) — those
never cross the actor boundary; the net architecture crosses as the **self-describing weight manifest**
over redis (§4), so the actor derives `in_dim`/`n_actions`/`hidden`/`residual` from the manifest bytes,
never from the config (the `transport.unpack_net` / `params_from_manifest_blob` discipline,
`transport.py:79-90`, `inference_server.py:131`). This is why the config schema stays small and is
genuinely a projection, not a re-pack of `ExperimentConfig`.

---

## 4. Weights and results: stay on the redis raw-bytes seam

**Decision: weights and result blocks stay on the transport redis (6380, raw float bytes, no pickle);
only the small structured meta moves onto the socket.** Justification:

- The redis weight seam (`publish_weights` / `read_weights`, `transport.py:177-240`) is the **one weight
  holder** shape the inference server already shares (`RedisParamsSource`, `inference_server.py:184-219`):
  one publish, one version-gated reload, every consumer reads the same bytes. Moving weights onto the
  control socket would *duplicate* that holder and re-pack the byte-identical `WeightContainer` layout a
  second way — exactly the second-transcription ADR-0012 P7 forbids.
- The result blocks are large raw float32 (`result_spec.py`, X/PI/M/Y); they belong on the LRU-evicting
  bytes store (`transport.py:30-36`), read+deleted within the iteration. Streaming megabytes of floats
  through a JSON control socket would be strictly worse.
- What moves onto the socket is the **tiny structured meta** that was previously a stderr scrape: the
  `"written"` count (replacing the `wrote (\d+) episode` regex, `cpp_executor.py:147`) plus the epoch and
  version echoes. This is a genuine improvement — the meta channel is now structured and typed, the way
  the Python worker pool already gets its `(idx, n, feat_dim, n_slots)` meta structurally rather than by
  scraping.

The reconciliation logic in `generate` (`cpp_executor.py:140-158`: read N back, compare to the reported
written count, fail loud on a shrunk buffer under LRU eviction) is **preserved unchanged** — it now reads
`written` from the structured reply instead of a regex, and the "could not parse the count" floor
(`cpp_executor.py:155`) disappears because the count is always structurally present.

### 4.1 A 5th result block (BOOT) would be fixed-arity surgery, not a free extension

The Part-B parity work (§8.2) needs a **fifth** result block. The draft called this "extend the existing
seam — `read_and_delete_results` reads it the same way it reads the other four." **That is wrong against
the code (critique B2).** The result transport is **hardcoded to exactly four blocks** at the following
sites, none of which derive arity from `BLOCK_ORDER`:

- `transport.result_keys()` returns a literal **4-tuple** `(X, PI, M, Y)` (`transport.py:108-112`).
- `RedisTransport.read_and_delete_results` issues 4 `pipe.get` per task and slices `blobs[4*k:4*k+4]`
  with a fixed X/PI/M/Y reshape (`transport.py:204-225`); its `metas` tuple is `(idx, n, feat_dim,
  n_slots)` (`transport.py:193`) and carries **no length for a 5th 1-D block**.
- `transport.write_results(..., X, PI, M, Y)` is a **fixed parameter list** validated block-by-block
  against `BLOCK_X/PI/M/Y` (`transport.py:243-268`).
- `cpp_executor._read_records` independently does the same 4-block decode (`cpp_executor.py:172-187`).
- C++ `transport.hpp`: `struct ResultKeys { X, PI, M, Y; }` and a fixed-arity `write_results` signature;
  `result_spec.hpp`: `std::array<std::string_view, 4> BLOCK_ORDER`, `std::array<int,4> BLOCK_RANK`, the
  literal `4` recurring.

So adding a 5th block is a **coordinated, fixed-arity edit across ~7 sites in two languages**, plus a new
`metas` length entry for the 1-D BOOT block. The drift net *will* red on a one-sided change
(`test_result_spec_block_order_and_ranks_agree`, `test_wire_drift.py:159`, reds when `BLOCK_ORDER` grows
on one side; the round-trip leg `:264` exercises the real reader) — which is good, but that is the net
**catching** the coordinated edit, not the edit being free. This note states the work honestly:
**§8.2's BOOT block is the §10 migration's largest single code change**, and it is gated on the §8.2
emission decision being made first (B4 below).

A cleaner alternative — *derive* the block arity from `BLOCK_ORDER` everywhere first, so a 5th block is
one SSOT edit — is a real refactor of `transport.py` + `transport.cpp` and is **out of scope** for this
note (§11.2); if taken it is sequenced *before* the BOOT block lands and is called out as its own step.
This note assumes the fixed-arity surgery unless that refactor is separately scheduled.

---

## 5. Daemon lifecycle and the fail-loud / timeout discipline

### 5.1 Who spawns it

**Decision: the executor spawns the daemon at `CppActorExecutor.__init__`** via `subprocess.Popen`
(detached, not `subprocess.run`), holds the handle, and reaps it in `close()`. This keeps the executor
**self-contained** — `exit_loop.run` constructs the executor and gets a working actor with no external
process to manage — exactly as it constructs `ParallelExecutor`'s pool today. An externally-run daemon is
**not** the default (it would require an out-of-band launch + addressing handshake); it is reachable via
the `tcp` endpoint config (§1.2) for a future remote/persistent-daemon case, but the in-repo path is
executor-spawned.

### 5.2 Addressing and port collisions across concurrent experiments

The `<run>` id (`self.run = uuid.uuid4().hex[:12]`, `cpp_executor.py:87`) namespaces the `ipc` socket
path (`ipc:///tmp/chocofarm-actor-<run>.sock`, §1.2) and the redis weight/result keys
(`transport.py:99-112`). Because the run id is a fresh uuid **per executor construction**, two concurrent
experiments get **structurally distinct** socket paths and key namespaces — collision-freedom by
construction, not by a port-allocation dance. (For the `tcp` override, the port is config; a collision
there is a loud bind failure, §5.4.)

### 5.3 Readiness: bounded-retry with a "serving" signal, not a single ping (critique M3)

The draft caught "a daemon that died on startup" with a single post-spawn `ping`. But the failure mode
the project is *wariest* of (CLAUDE.md, the `jaxtrain-deadlock-rca.md` arc) is the daemon that is **alive
but wedged** — bound the socket, then blocked in `load_instance` or a redis connect that hangs, never
reaching the serve loop. A single ping on a short timeout would then *time out*, and `Popen.poll()` would
show the process *alive* — indistinguishable from "slow startup." So:

**Decision: the readiness protocol is bounded-retry-with-ceiling + an explicit `serving` signal.**

- The daemon builds env + policy + net **before** it begins serving, and its **first `ping` only
  succeeds (`{"ok": true, "serving": true, ...}`) once construction is complete**. A `ping` that arrives
  before construction finishes either is not yet being recv'd (the daemon is pre-loop) or, if the daemon
  recvs early, replies `{"serving": false}`.
- The executor issues `ping` on a short per-attempt `ZMQ_RCVTIMEO`, **retrying up to a hard ceiling** (a
  bounded number of attempts × the per-attempt timeout = a total readiness budget, e.g. 30s). Each
  attempt also checks `Popen.poll()`: if the process **died**, that is an immediate loud failure (no
  point retrying). If the ceiling is reached with the process **alive but never `serving:true`**, the
  daemon is wedged-in-construction → the executor **SIGKILLs it and raises loudly at construction**.

This makes "bound but wedged in construction" a **loud timeout at `__init__`**, not a hang on the first
`generate`. The `serving` distinction is the piece the draft's single ping lacked.

### 5.4 Stale `ipc` socket: live-peer-loud unlink, and the run-id invariant (critique M4)

A crashed prior daemon can leave a stale `ipc` socket file. The draft asserted the unlink-and-bind is
race-free "by uuid construction." That argument covers **two different experiments** (distinct run ids →
distinct paths) but **not** a **restart of the same experiment**. The honest statement:

- **State the invariant explicitly:** `self.run` is a fresh uuid **per `CppActorExecutor.__init__`**
  (`cpp_executor.py:87`). A resumed run constructs a **fresh** executor → a **fresh** run id → a
  **fresh** socket path, so in the common case there is no same-id concurrency. **Any future change that
  pins the run id across `--resume`** would reintroduce the race and must revisit this section.
- **Make the unlink a live-peer-loud check, not an unconditional unlink.** Before binding, the daemon
  *connect-probes* the existing socket path: if a **live** peer answers (a still-shutting-down prior
  daemon holding the socket), the bind is a **loud failure**, not a silent unlink-over-a-live-peer. If the
  probe finds no peer (a truly stale file), it unlinks and binds. This turns the one genuine race (fast
  crash-respawn of the same path, were the id ever pinned) into a loud abort instead of a silent
  socket-steal.

### 5.5 Graceful shutdown, crash, and `close()` reaping

`CppActorExecutor.close()` (today `cpp_executor.py:214-218`, just a redis close) gains a
**graceful-then-forceful** daemon reap, mirroring the inference server's documented sequence (`stop()` →
join → `close()`, `inference_server.py:316-322`):

1. Send `shutdown` on the socket; await an ack on a **bounded** recv.
2. If the ack arrives, the daemon is exiting cleanly; `Popen.wait(timeout=...)` reaps it.
3. If the ack does **not** arrive within the bound, **escalate**: `SIGTERM`, then after a short grace
   `SIGKILL`, then `Popen.wait`. A wedged daemon is never waited on indefinitely.
4. Close the redis connection (the existing `cpp_executor.py:215-217` behavior) and unlink the `ipc`
   socket file.

A **crashed** daemon (between generations) is detected the next time the executor touches the socket: a
bounded-recv timeout on `generate` (§2.4) plus a `Popen.poll()` showing a dead process → loud
`RuntimeError` (the same loud-on-runner-failure posture as `cpp_executor.py:136-139`). No zombie: the
executor owns the `Popen` handle and `wait`s it on every exit path.

---

## 6. Reconfigure-without-restart — split by Mut class (critique B3)

This is the maintainer's headline win, and the section the draft over-claimed. The draft said a
`configure` with a "new `GumbelConfig`" rebuilds the policy live as "a cheap object reconstruction." But
the hp schema's **own mutability classes** forbid that blanket rule:

- `m: Mut.RESTART` (`schema.py:103`, "sizes the SH bracket"),
- `n_sims: Mut.RESTART` (`schema.py:104`, "baked into the SH phase loop"),
- `use_jax_mlp: Mut.RESTART` (`schema.py:110`, "binds a fn"),
- only `c_puct/c_visit/c_scale/c_outcome/max_depth` are `Mut.HOT` (`schema.py:105-109`).

This is exactly why the current executor's `_RUNNER_HOT_KNOBS` is **precisely those five HOT knobs and
not `m`/`n_sims`** (`cpp_executor.py:64`). The whole codebase treats `m`/`n_sims`/`use_jax_mlp` as
RESTART. A `configure` that silently rebuilt the policy on an `m` change would **silently elevate a
RESTART knob to HOT** — a P2 / fail-loud violation.

**Decision: `configure` dispatches on the field's Mut class (the §3.3 `ACTOR_CONFIG_MUT` array, sourced
from `schema.py`):**

- **HOT search knobs changed** (`c_puct/c_visit/c_scale/c_outcome/max_depth`, `explore_plies`,
  `lam_blend`/`n_step`) → **rebuild the `GumbelAZPolicy` live** off the current `GumbelConfig` (a cheap
  object reconstruction — the `GumbelAZPolicy(gc, net, env)` construction, `main.cpp:192`). The env is
  untouched; the net is untouched (its version is gated separately, §2.2). This is the genuine
  live-retune.
- **RESTART knob changed** (`m`, `n_sims`, `use_jax_mlp`) → a **loud `invalid_config` reject**
  (`{"ok": false, "error": "restart_knob_changed_live", "detail": "m: 12→16 is RESTART; restart the
  experiment with --resume"}`), the same remediation the hp registry gives (`schema.py:14`, "REFUSED
  LOUDLY … the operator adopts it by restarting with --resume"). The daemon does **not** rebuild.
- **INSTANCE knob changed** (`instance_path`/`faces_path`) → a **loud reject with the stronger
  remediation** (`schema.py:15`, "a change is a NEW experiment"). The env is built once at the first
  `configure` and never rebuilt.
- **New `version`** → handled by `generate`'s version gate (§2.2), a redis weight reload, **not** a
  `configure`.

**The honest scope of the win, restated.** The marginal win of the daemon over the subprocess is **"no
respawn"** — the subprocess path *already* live-retunes the five HOT knobs via `_RUNNER_HOT_KNOBS`
(`cpp_executor.py:132-134`) by re-passing them as argv each generation; the daemon does the same retune
without paying instance-load + redis-connect + net-read per iteration. It does **not** newly enable live
retuning of `m`/`n_sims` — those remain RESTART, refused loud. Claiming otherwise would be the
silent-RESTART-elevation the schema exists to prevent.

---

## 7. The executor refactor

### 7.1 `CppActorExecutor` becomes a daemon client; the contract is unchanged

`generate(net, version, worlds, lam, explore_plies, lam_blend, n_step, hot_search, max_steps)`,
`evaluate(...)`, and `close()` keep their **exact signatures** (`cpp_executor.py:96-99, 190-192, 214`),
so `exit_loop.run` is oblivious. Internally:

- `__init__` spawns the daemon (§5.1), runs the bounded-retry readiness probe (§5.3), and sends the first
  `configure` (the projected `ActorConfig`, §3.2). It computes and caches the live `config_epoch`.
- `generate` (i) publishes weights to redis as today (`publish_weights`, `cpp_executor.py:124`), (ii) if
  the projected `ActorConfig` **changed** since the last `configure`, sends a new `configure` and bumps
  the cached epoch, (iii) sends `generate` with the per-call `(config_epoch, version, seed, lam,
  episodes, max_steps, res_token)`, (iv) on the structured reply reconciles `written` against the blocks
  it reads back from redis (the preserved `cpp_executor.py:140-154` logic), (v) returns the flat
  `list[_Record]`.
- The **two current fail-loud guards** (`cpp_executor.py:104-122`, refusing the Part-B blend and
  `explore_plies>0` because the C++ wire cannot carry them) are **lifted only as §8 wires those knobs
  across** — not before. Until §8.1/§8.2 land, the daemon-client keeps the same loud refusals, so the
  executor never silently trains on a wrong target.
- `evaluate` is unchanged (in-process Python, no daemon contact, `cpp_executor.py:200-212`).
- `close` reaps the daemon (§5.5).

### 7.2 The subprocess path is retired, not kept as a silent fallback

**Decision: the subprocess `generate` is retired, not kept as a silent fallback** — matching the
project's no-silent-local-fallback precedent (`zmq_net_client.cpp:127`, "NOT falling back to a local net
(ADR-0002)"; `inference_server.py` §5). A silent fallback would mask the daemon path being broken, which
is the SSOT-path-down-masked failure ADR-0002 forbids.

**Migration window (the one concession).** A `--cpp-runner-subprocess` launch flag keeps the old path
available for **one transition window** (so a daemon regression does not block experiments). To prevent
the flag itself becoming a silent-degradation seam (critique N4), selecting it **warns loudly** at launch
that it is the **retiring** path (a stderr banner, not a silent acceptance). The flag is removed once the
daemon path is the proven default.

---

## 8. Parity integration

This is the section the critique found most broken. The two parity knobs being added in the parallel
thread flow through the config + the result seam — but the draft mis-identified the value-target quantity
(B1), under-stated the transport change (B2, handled in §4.1), and left the emission contract undecided
(B4). All three are resolved here.

### 8.1 `explore_plies` — config knob, HOT

`explore_plies` (`schema.py:201`, HOT) crosses in the `ActorConfig` (§3.1). Both Python paths sample the
**executed** action from π′ for the first this-many plies at temperature 1 (`cpp_executor.py:31-37`
documents the parity gap). The C++ side must mirror `generate_episode`'s `temp = 1.0 if ply <
n_explore_plies else 0.0`: expose a temperature>0 executed-action sample on the C++ `Decision`/runner. The
C++ `GumbelAZPolicy` currently executes the **SH survivor at temperature 0 every ply**
(`gumbel.hpp:135,152`), so this is a real addition to the search's executed-action selection — a
parity-thread change. The daemon merely **carries the knob** and rebuilds the policy live on a change
(§6, HOT). Until the parity thread exposes the temperature sample, the executor keeps the loud
`explore_plies>0` refusal (`cpp_executor.py:116-122`).

### 8.2 The Part-B value blend — the BOOT block carries the **root search value**, NOT `v_mix`

**The correctness fix (critique B1).** The draft repeatedly said the Part-B bootstrap is "`v_mix`
(already computed in the C++ search's `improved_policy`)" and proposed emitting `v_mix` as the 5th block.
**`v_mix` and the bootstrap are different quantities — emitting `v_mix` would train Part-B on the wrong
number.** Verified against `value_target.py`:

- The bootstrap `boot[j]` is the **visit-weighted mean of the root actions' simulated returns** —
  `GumbelAZSearch._root_search_value` (`value_target.py:29-32`): *"The search already produces, at every
  decision, a ~n_sims-averaged estimate of the CURRENT belief's λ-penalized value
  (`GumbelAZSearch._root_search_value` — the visit-weighted mean of the root actions' simulated returns).
  Call it `boot[j]`."* This is the scalar the value target bootstraps off (`blended_returns_to_go`,
  `value_target.py:120-190`).
- `v_mix` is the **value-completion for unvisited actions** — *"the net leaf value blended with the
  PRIOR-weighted (not visit-weighted) mean of the visited actions' Q"* (`value_target.py:69-72`,
  `v_mix(...)` at `:226-249`). It is a per-slot completion term feeding the improved-π softmax
  (`improved_policy`, `:252-280`), **not** the scalar root value the blend needs.

In the C++ search, `improved_policy` (`gumbel.hpp:229`) computes `v_mix` internally for the σ-softmax, and
the `Decision` struct exposes only `{action, improved, n_spent, survivor_slot}` (`gumbel.hpp:151-156`) —
**neither `v_mix` nor the root search value is currently exposed.** So the parity-thread change is:
**expose the C++ analog of `_root_search_value`** (the visit-weighted root return) on the `Decision`
struct, and the runner emits it as a per-decision **`BOOT` block** (the 5th result block, §4.1). The
Python side then blends via the **one** `blended_returns_to_go` (`value_target.py:120`) — never a second
blend transcribed in C++ (`cpp_executor.py:28-30` states this discipline). The drift net's `result_spec`
legs pin the 5-block layout (§4.1, §10.3).

**The Y-block / blend-ingredient contradiction — resolved, not punted (critique B4).** The draft honestly
flagged that `run_episode` emits the **reduced** suffix-MC Y (`runner.cpp:78-117`: `out.Y[j]` is
`g_steps[j]`, a suffix sum), while `blended_returns_to_go` needs **per-step `(r, dt)` + boot**
(`value_target.py:120-140`). You **cannot** reconstruct the per-step blend from the reduced Y — the
suffix sum has thrown away the per-step structure. The draft listed three incompatible "resolutions" and
deferred all of them — while *simultaneously* committing the BOOT transport, whose shape depends on which
resolution holds. That is incoherent. This note decides it:

> **Decision 8.2-D (the Part-B contract for when it lands): the C++ runner emits, in addition to BOOT,
> the per-step `(r, dt)` as the raw ingredients the Python blend consumes; Y becomes Python-derivable for
> the Part-B path, not C++-reduced.** Concretely, the Part-B path emits per-decision `r` and `dt` so
> `blended_returns_to_go(step_rt, boot, exit_c, lam, lam_blend, n_step)` runs in Python over real
> ingredients. This makes the `result_spec` change **concrete** — it is BOOT **plus** the `(r, dt)`
> carriage — and it compounds the §4.1 fixed-arity surgery (more blocks, or a reshaped layout), which
> §4.1 already says must be enumerated honestly.

**The v1 sequencing decision.** The cleaner near-term move is **scope Part-B out of the v1 daemon**: ship
the daemon + `explore_plies` carriage, keep the executor's loud Part-B refusal
(`cpp_executor.py:107-111`), and land the BOOT + `(r,dt)` emission as its own coordinated `result_spec`
step (§10.1 step 5) **once the parity thread settles what `_root_search_value` is in C++ and whether the
emission is per-step `(r,dt)` or a reshaped Y.** This note **recommends the v1 scope-out**: the daemon's
value — no respawn, consolidated config, structured meta — is fully realized without Part-B; and freezing
the BOOT/`(r,dt)` shape here would commit a transport to an unmade numerics decision (exactly the
incoherence the critique named). Decision 8.2-D is the contract for *when* Part-B lands; the v1 daemon
*carries the knob and refuses it loudly* until then. This is a genuine cross-thread dependency, recorded
honestly in §11.3.

---

## 9. Determinism / parity (ADR-0012 P6)

**Decision: the persistent daemon preserves the per-episode RNG / world-draw semantics of the subprocess
exactly.** The argument, verified against `runner.cpp`:

- Per-episode seeding is `fold_seed(cfg.seed, idx)` (`runner.cpp:125-130`): a pure splitmix-style fold
  over `(seed, idx)`, producing a fresh `std::mt19937_64` per episode (`runner.cpp:143`), from which the
  world is drawn (`runner.cpp:144-146`). This is a **pure function of `(seed, idx)`** — it reads no
  process-lifetime state, so a persistent daemon and a fresh-per-call subprocess produce **bit-identical**
  episode RNG streams for the same `(seed, idx)`. Process lifetime is irrelevant to the draw.
- The per-generation seed is `base_seed + version` (`cpp_executor.py:130`), carried in the **`generate`
  message** (§3.1), never cached in config. So **two `generate`s at the same version** (a retry)
  reproduce identically, and a new version derives a new seed deterministically. **This version→seed
  derivation living in the per-call message — not in sticky config — is the determinism anchor**; it is
  restated here (and in §3.1) precisely so a future change cannot quietly move it into cached config and
  break the property.
- **The daemon must re-seed per `generate` and never carry an RNG across generations.** The env/world
  list is held live (built once at `configure`), and `env.worlds()` (`runner.cpp:140`) is stable, so the
  world *set* is identical generation-to-generation; only the per-episode draw RNG re-folds from the fresh
  `(seed, idx)`. The daemon holds **no** RNG state between `generate` calls — each call's RNGs are
  reconstructed from the message's seed. This keeps the behavioral P6 bar (`gumbel.hpp` RNG note:
  "production parity on the RNG-driven aggregates is the BEHAVIORAL bar") exactly as the subprocess had
  it; the daemon changes nothing about the per-episode draw semantics.

(Cross-language byte-identity is **not** claimed — `fold_seed` is the C++ runner's own fold, distinct from
numpy's stream by design, `runner.cpp:121-124`. The P6 bar is behavioral, and the daemon does not move
it.)

---

## 10. Migration / rollout plan and test strategy

### 10.1 Incremental steps

1. **The config SSOT + drift net.** `actor_config.py` (the projection + `FIELD_NAMES` + Mut-class view),
   `actor_config.hpp` (the struct + the parseable `ACTOR_CONFIG_FIELDS` / `ACTOR_CONFIG_MUT` literal
   arrays, §3.3), and the new always-on drift legs (field-set agreement, Mut-class agreement, negative
   mutation self-check) in `test_wire_drift.py`. No daemon yet — this lands the SSOT and its net first
   (contracts before glue).
2. **The C++ daemon skeleton.** First C++ ZMQ **server bind** in the tree (§11.1): a ROUTER loop reusing
   the inference server's bounded-poll shutdown *shape*; the JSON config parse via
   `nlohmann::ordered_json` (no new dep, §3.3); `configure` (Mut-class dispatch, §6) / `ping` (the
   `serving` signal, §5.3) / `shutdown`. `generate` initially wraps the **existing** `runner.cpp::run`
   verbatim (same redis weight read + 4-block write) — so the daemon is a control wrapper around the
   proven episode loop, not a rewrite of it.
3. **The executor becomes a client.** `CppActorExecutor` spawns + readiness-probes + `configure`s + sends
   `generate`, reconciles `written` from the structured reply (§7). The subprocess path stays behind the
   warn-loud `--cpp-runner-subprocess` flag (§7.2) for the transition window. A behavioral parity gate
   (§10.2) runs both and compares.
4. **Retire the subprocess path.** Once the daemon is the proven default, remove the flag.
5. **(Deferred to the parity thread) the BOOT + `(r,dt)` result-seam change** for Part-B (§8.2-D), landed
   as its own coordinated fixed-arity step (§4.1) when the parity thread settles the emission. This is
   **not** in steps 1-4 (the v1 scope-out, §8.2).

### 10.2 Test strategy

- **Control-protocol drift net (§10.3)** — the always-on legs for `actor_config`, modeled on the
  `result_spec`/`wire_spec` legs.
- **Fake-ROUTER / fake-DEALER unit tests** for the executor client, the way `test_zmq_inference.py` pins
  the inference server's drain/scatter without a real socket: drive `configure`/`generate`/`ping` request
  encode + reply decode against a stub daemon, assert the epoch/version gates (§2.2), the loud rejects
  (§2.3), the Mut-class dispatch (§6), and the readiness retry/ceiling (§5.3).
- **Subprocess ↔ daemon behavioral parity gate** (during the step-3 window): run a fixed `(seed, version,
  episodes)` generation through both paths and assert the produced records match **bit-for-bit** — the
  determinism argument (§9) makes this a same-language episode-RNG bit-identity check, so the daemon must
  change nothing. The strongest available proof the transport switch is behavior-preserving.
- **Lifecycle tests:** readiness timeout on a wedged-in-construction daemon → loud `__init__` raise
  (§5.3); a crashed daemon mid-run → loud `generate` raise (§5.5); `close()` graceful-then-SIGKILL reap
  with no zombie (§5.5); stale-socket live-peer-loud bind (§5.4).

### 10.3 The control-protocol drift net (concretely)

New always-on legs in `tests/test_wire_drift.py` (no C++ build, no redis), modeled on the existing
layout-agreement + negative-mutation legs (`test_wire_drift.py:159-167, 311-346`):

- **Config field-set agreement:** `_cpp_str_array(actor_config.hpp, "ACTOR_CONFIG_FIELDS")` equals
  `actor_config.FIELD_NAMES`. A field added/removed/renamed on one side reds.
- **Mut-class agreement:** `_cpp_str_array(actor_config.hpp, "ACTOR_CONFIG_MUT")` equals `[schema mut of f
  for f in FIELD_NAMES]`. A field that changes its live/reject class on one side reds — this is what makes
  the §6 "RESTART knob → loud reject" classification a **drift-protected** fact, not a comment.
- **Negative mutation self-check:** perturb one array entry (the `test_wire_drift.py:302-308` pattern
  generalized) and assert the agreement raises — proving the leg catches drift.
- **(With the §10.1-step-5 BOOT change)** the `result_spec` block-order/rank legs already cover a 5-block
  layout (`test_wire_drift.py:159`); the round-trip leg (`:264`) exercises the real reader, so the
  fixed-arity surgery is netted.

---

## 11. Open questions, risks, and out-of-scope

### 11.1 First C++ server-bind in the tree (risk, honestly flagged)

The C++ side today has **clients only** — `ZMQ_REQ` in `zmq_net_client.cpp`, `DEALER` in the wire
benches. A daemon means the **first C++ `zmq_bind` / ROUTER** in the codebase: new server-side socket
lifecycle (RAII ctx/socket, `LINGER 0` as `zmq_net_client.cpp:58-66` sets it, bounded poll). It is a
genuinely new surface; step 2 (§10.1) isolates it behind a skeleton that wraps the proven `runner.cpp`
episode loop, so the *control* surface is the only new thing.

### 11.2 The §4.1 arity refactor vs. fixed-arity surgery (open)

Whether to *derive* result-block arity from `BLOCK_ORDER` (a `transport.py`/`transport.cpp` refactor,
making a 5th block one SSOT edit) **before** the BOOT block, or do the fixed-arity surgery across ~7
sites. This note defaults to the surgery and flags the refactor as a separately-scheduled option (§4.1).
Maintainer decision when step-5 is scheduled.

### 11.3 The §8.2 Part-B emission contract depends on the parity thread (genuine cross-thread dependency)

The BOOT block's exact content — the C++ analog of `_root_search_value`, and whether Part-B carries
per-step `(r, dt)` (§8.2-D) or a reshaped Y — is a **numerics decision owned by the parallel parity
thread**, not by this transport note. This note recommends scoping Part-B out of the v1 daemon (§8.2) so
the daemon ships without freezing that decision, and states 8.2-D as the contract for when it lands.
Surfaced, not silently resolved.

### 11.4 The two-gate design (resolved, recorded)

`config_epoch` (config adoption) and `version` (weight reload) gate different things (§2.2); the reply
echoes both and the executor asserts both round-trips. This is not redundant — it is the loud-on-desync
check that a new-version/same-epoch generate (the common case) is *not* mis-rejected and that a stale
config is *never* generated under. Recorded so a later reader does not "simplify" it into one gate.

### 11.5 Heartbeat dropped (resolved, recorded)

The redis heartbeat is **dropped** from v1 (§2.4): under lock-step the executor is blocked in the recv and
cannot read it, so it had no in-loop consumer. The bounded `ZMQ_RCVTIMEO` is the real safety net. A
heartbeat returns only with an out-of-band watcher (the async embodiment), not before.

### 11.6 Out of scope (and the relationship to the scaling docs)

**Explicitly out of scope:** the async work-stealing / multi-tree multiplexer (Shape C of
`docs/design/scaling-and-cpp-seam.md`; the `FiberMuxRuntime` / DEALER-submit-poll of
`docs/design/cpp-search-runtime.md`). That restructure is gated behind an ADR-0009 measure-first
benchmark (`cpp-search-runtime.md` §6-Q5, §8.5) and binds `{scheduler, transport}` as a matched pair
(`cpp-search-runtime.md` §0) — a blocking control call (this note's lock-step `generate`) cannot live
inside an async continuation. **The relationship, not the fold-in:** this daemon is the *control
transport* for the synchronous actor; it is forward-compatible with the async work (the ROUTER's
opaque-envelope echo, §1.1, leaves room for a future corr-id with no parse), but it does **not**
pre-decide it and must not be cited as having built it. The async restructure remains the separate,
benchmark-gated R-series work those notes own.

---

*Public Domain (The Unlicense).*
