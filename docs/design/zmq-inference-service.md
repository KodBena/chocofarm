# The batched ZeroMQ inference service (Shape B) — concrete design (2026-06-15)

A forward-looking design record (authored at decision time, per the documentation
discipline), not yet built. It makes **Shape B** of `docs/design/scaling-and-cpp-seam.md`
(§2 — "inference over a Python-hosted batched ZeroMQ service") concrete: the wire
contract, the server's batching discipline, the version-gated weight reload, the
honest fidelity trade, and the zero-cost ACL that lets the C++ search call it as a
drop-in for the local leaf evaluator.

It composes with — does not contradict — the seam note (§2 there is the parent
claim: "falls out behind seam 2"), `docs/design/simulation-parallelization-viability.md`
(this is **Axis A** cross-episode batching, the #1 throughput lever, NOT Axis C
within-search leaf batching), and the C++ `cpp/include/chocofarm/net.hpp` ACL the
client slots behind. Where it says "drop-in," it means "satisfies the
`NetEvaluator`/`Net` port the search already holds — zero search edits."

> **The MLP forward stays in Python by design.** `chocofarm/az/forward.forward_core`
> is the ONE forward graph (audit R11 collapsed four hand-transcriptions to one); a
> native C++ MLP is a *second transcription* of that SSOT — knowingly tolerated only
> as the interim `NetForward`, behind an ACL, so it is deletable without touching the
> search. Inference lives where the SSOT lives and where batched JAX is fast, and the
> net architecture stays free to change (residual on/off, depth, heads, a different
> trunk entirely) because the wire contract is just *float32-vector-in → (value,
> logits)-out* — no consumer recompiles when the net's shape changes.

---

## 0. Why Python-hosted (the SSOT argument), and what the C++ native MLP is for

`forward_core(params, X, xp)` is the single source of truth for the value+policy
graph — one body parameterized on the array module, serving the numpy-f64,
numpy-f32, and JAX paths at once (R11). The C++ `NetForward` (`cpp/src/net.cpp`)
reimplements that graph a *second* time. Two transcriptions of one graph is the
exact hazard R11 existed to kill (the fourth copy was once silently residual-dropping).

So the maintainer's standing decision: **the production leaf evaluator is the Python
service**, and `NetForward` is an *interim* — it lets the C++ runner evaluate leaves
with no service process up (useful for the search port's own parity work), and it is
held behind a zero-cost abstraction boundary (§2) so that when the service is the only
path, deleting `NetForward` touches one construction site, not the search. The C++
native MLP is therefore a *deliberately-tolerated SSOT violation with an ACL around
it*, not a competing forward. Keeping inference in Python buys three things at once:

1. **One forward graph.** The service runs `forward_core` — the same SSOT every Python
   path runs. No drift surface.
2. **Batched JAX is fast.** A server batching leaves from N independent workers runs
   one `forward_core` over a stacked `(B, in)` matrix; the per-leaf cost amortizes.
3. **Architecture flexibility for free.** Because the wire contract carries only the
   feature vector and the raw `(value, logits)`, the net's internal shape — derived
   end-to-end from the manifest-bound weights, never hardcoded — can change with zero
   change to any worker or to the C++ client.

## 1. The port and its two impls — the zero-cost ACL

The search holds the net as an injected dependency (seam 2) and calls only a leaf
evaluator. In C++ that port is, after the ADR-0012 P9 pass:

```cpp
// the NetEvaluator port (the search compiles against THIS, not a concrete net)
struct NetPrediction { float value; std::vector<float> logits; };  // de-std value, RAW logits
// predict(features) -> prediction, or a typed transport/validation failure (P9 rule 5)
[[nodiscard]] std::expected<NetPrediction, Error> predict(std::span<const float> X) const;
```

Two implementations satisfy it:

| impl | forward runs | when |
| --- | --- | --- |
| `NetForward` | locally, in-process C++ | interim; service not up (parity, smoke) |
| `ZmqNetClient` | remotely, on the Python service (batched) | the SSOT path |

"Zero-cost ACL" is precise: the port is a thin dispatch boundary; the real cost is the
matmul (`NetForward`) or the round-trip (`ZmqNetClient`), never the abstraction. Swapping
impls is a *construction-site* choice — the search, the NMCS/ISMCTS/Gumbel policies, and
the belief mechanics do not change a line. On the Python side the mirror is a `Net`
`Protocol` (`predict(X) -> (value, logits)`) that both `ValueMLP` (local) and a Python
`ZmqNetClient` satisfy, so a Python worker and the parity harness can use local or remote
interchangeably.

## 2. The wire contract — the `NetPrediction` shape, masking stays client-side

The service speaks **exactly** the `NetPrediction` contract `net.hpp` already defines:

- **Request:** one float32 feature vector `X`, length `in_dim` (the `FeatureBuilder`
  output). That is all — no legal mask, no weights, no version.
- **Response:** `value` (float32, **de-standardized**: `v = v_std·y_std + y_mean`, the
  λ-penalized-return scale) + `logits` (float32`[n_actions]`, **raw** — not softmaxed;
  empty when the net is value-only, mirroring `forward_core`'s `logits=None`).

The masked softmax (`ValueMLP._masked_softmax` / the C++ search's prior) is **NOT** on
the wire. The legal mask is per-node search state the server does not hold, and masking
is a pure function of `(raw logits, legal_mask)` — so it stays at the consumer. This is
why the request is minimal and why `ZmqNetClient` is a literal drop-in for `NetForward`:
both return raw `NetPrediction`; the search applies the mask either way. (`ValueMLP.predict_both`
remains the convenience that composes the raw forward with the mask for the Python local
path; the *service* exposes the raw forward beneath it.)

**Framing.** Length-prefixed little-endian float32, in the raw-bytes spirit of the redis
transport (seam 3 — language-agnostic by construction). A request is `[in_dim:u32][X:
f32×in_dim]`; a response is `[n_actions:u32][value:f32][logits:f32×n_actions]`. A tiny
fixed header byte carries a protocol version so a Python/C++ codec mismatch fails loudly
rather than silently misreading floats. This frame is a prime candidate for the #23
mechanized result-format contract (one schema → both codecs), so Python and C++ cannot
drift — recorded as the tie-in, designed but not yet generated.

## 3. The server — self-clocking microbatching, no latency timer

A single Python process, single-threaded around one ZeroMQ `ROUTER` socket; workers are
`REQ` (or `DEALER`). The loop is **greedy-drain**, which self-clocks the batch size to
the load with no time-window tuning:

```
loop:
  block until ≥1 request is queued on the ROUTER
  drain ALL currently-queued requests (up to a max batch cap)  -> identities[], X_rows[]
  Xb = stack(X_rows)                       # (B, in_dim) float32
  v_std, logits = forward_core(params, Xb, jnp)   # ONE forward, the SSOT
  v = v_std * y_std + y_mean
  scatter (v[i], logits[i]) back to identities[i]
```

While the forward of batch *K* runs, requests for batch *K+1* queue up; when it returns,
the loop drains whatever accumulated. Under light load B≈1 (low latency); under heavy
load B grows to the cap (high throughput) — the batch size tracks demand automatically,
no microbatch timer to tune. Single-threaded is deliberate: JAX/XLA owns the forward, no
shared-state concurrency, and **no XLA-in-a-worker-thread** — the failure mode the
`docs/notes/jaxtrain-deadlock-rca.md` arc fought. If one process saturates (it will not at
the current tiny-MLP scale — the sim-note ranks the C++ inner core as #2-conditional and
GPU leaf-batching as gold-plating), run N stateless instances behind a load balancer; the
design does not need it.

**Weight reload (seam 4).** The server is the *one* holder of the weights. It subscribes
to the version-gated weight broadcast (`transport.read_weights(run, phase, version)` — the
same `(phase, version)` key the workers' `_ensure_net` watches) and reloads `params` when
the published version changes, between batches. Workers never touch weights — a side win
over Shape A's per-worker reload: one reload serves all leaves, and every leaf in a batch
sees one consistent net version.

## 4. Fidelity — Axis A roundoff, honestly

This is **cross-episode** batching (sim-note Axis A): the B rows come from B independent
workers' leaves. A row of the batched `(B,in)@W` matmul is the same row-wise-independent
dot product as the single-row call — it carries only the **forward-roundoff** non-exactness
the project already accepts (`tests/test_jax_equivalence.py`, ABS_TOL=1e-4, float32/jit). It
does **NOT** touch any search's Sequential-Halving budget, RNG order, or the Danihelka
invariants — that is Axis C (within-search leaf batching), which the project rightly defers.

Two roundoff facts to record so a later reader does not mistake them for bugs (both inside
the accepted P6 behavioral envelope — float32-equivalence, NOT byte-identity):

- **Row-vs-single.** XLA may reduce a `(B,in)@(in,h)` matmul in a different order than a
  `(1,in)@(in,h)` one, so a leaf's value can differ from its standalone evaluation in the
  low bits. Bounded by the same 1e-4 the equivalence test pins.
- **Batch-composition nondeterminism.** Which leaves land in which batch depends on arrival
  timing, so the exact f32 a given leaf receives can vary run-to-run by roundoff. This is the
  determinism trade already recorded for Shape C (seam note §2.5): per-leaf values are
  reproducible only up to batch-composition roundoff. Acceptable under P6; a *deterministic*
  drain (e.g. fixed B with a barrier) is available if exact aggregate reproducibility is ever
  wanted, at a throughput cost — not the default.

## 5. Failure semantics (ADR-0002, P9)

The ACL boundary **validates, does not coerce** (Port/ACL: translate-and-validate):

- **Server.** A malformed request (wrong length, NaN, unknown protocol byte) is a loud
  rejection, never a zero-filled or truncated forward. A reload that yields a shape-
  inconsistent manifest is a loud abort of the reload, not a silent run on stale weights.
- **Client.** `ZmqNetClient::predict` returns `std::expected<NetPrediction, Error>` (P9 rule
  5) — a timeout or a server-down is a **typed failure** propagated to the caller, NOT a
  silent fallback to `NetForward` (that would mask the SSOT path being down) and NOT an
  exception (an untyped effect). The Python `ZmqNetClient` raises loudly (ADR-0002) on the
  same conditions; there is no degraded-quiet mode.

## 6. Implementation surface

Concrete pieces, contracts-first (none built yet):

- **`Net` Protocol** (Python) — extract `predict(X) -> (value, logits)` (raw) as the port
  `ValueMLP` and `ZmqNetClient` both satisfy; `predict_both`/`predict_value` stay as the
  masked/­destandardized conveniences composed on top.
- **`chocofarm/az/inference_server.py`** — the ROUTER + greedy-drain loop + the version-gated
  reload hook + the wire codec. Loads `params` via the existing `ValueMLP.load` / `read_weights`
  path; runs `forward_core` under JAX.
- **`chocofarm/az/zmq_net_client.py`** — the Python `Net` impl that RPCs the service (drop-in
  for `ValueMLP` at the leaf; also the reference the C++ client is checked against).
- **C++ `ZmqNetClient`** — the `NetEvaluator` impl RPCing the service; **deferred until the
  ADR-0012 P9 `cpp/` pass lands** so it is born on the `std::expected`/`std::span` port shape.
  Designed here at the wire level; its in-process signature conforms to whatever P9 port the
  cpp pass settles.
- **The wire codec** — one frame spec, two codecs; #23's mechanization target so they cannot
  drift.

## 7. Parity

Mirrors `cpp/parity/net_parity.py`: the `ZmqNetClient` (Python first, then C++) over the
running service returns `(value, logits)` within **1e-4** of the local `forward_core` over
N≥1000 random float32 feature vectors, residual ON and OFF — the same ADR-0012 P6 behavioral
bar (NOT byte-identity). The batched path's row-vs-single roundoff (§4) lives inside that bar;
the harness drives batches of varied B to exercise it. Skips (does not fail) when the service
or redis is absent, like the existing opt-in cpp parity.

## 8. Non-goals

- **NOT Axis C.** No within-search leaf batching, no virtual loss, no async machinery in the
  worker — workers stay dumb (one blocking `predict` RPC each); the *server* does the batching.
- **NOT a search host.** The service evaluates leaves; it never runs a policy or touches the
  tree.
- **NOT GPU-first.** At the tiny-MLP scale this is CPU-batched; GPU is gold-plating the sim-note
  ranks below the throughput lever this serves.
- **`NetForward` is retained, not promoted.** It stays the interim local path behind the ACL,
  deletable once the service is the sole leaf evaluator.

---

## 9. Status

- **Forward-looking, not built.** No code accompanies this note; it is the contract the #27
  implementation targets and the design the maintainer reviews before that implementation begins.
- **Sequencing.** The Python server + Python `ZmqNetClient` + the `Net` Protocol can be built
  now (Python-only, no cpp dependency). The C++ `ZmqNetClient` waits on the ADR-0012 P9 `cpp/`
  pass so it lands on the compliant `NetEvaluator` port.
- **Single instance, uncalibrated time model** — the standing chocofarm caveat; it does not bear
  on this structural design (about the boundary, not instance numbers).

### Amendment 2026-06-15 — the Python side landed (ADR-0005 Rule 8: append, don't rewrite)

The forward-looking record above is preserved as written. As of this amendment the **Python half is
built** (the C++ `ZmqNetClient` remains deferred to the P9 `cpp/` pass, exactly as §9 sequenced):

- `chocofarm/az/inference_wire.py` — the ONE wire codec (the §2 frame: a protocol-version byte,
  length-prefixed little-endian float32; request `[ver][in_dim][X]`, response
  `[ver][n_actions][value][logits]`, `n_actions=0` ⇒ value-only). Shared by server and client.
- `chocofarm/az/net_port.py` — the `Net` Protocol (`predict(X) -> (value, logits)`, the §1 raw
  forward) plus the local `ValueMLPNet` adapter; both it and `ZmqNetClient` satisfy it.
- `chocofarm/az/inference_server.py` — the ROUTER + self-clocking greedy-drain microbatch loop (§3),
  the pure `run_microbatch` core, the version-gated reload hook as a mockable `ParamsSource`
  (`StaticParamsSource` injects params with NO redis; `RedisParamsSource` is the seam-4 path), and a
  jax-free `params_from_manifest_blob` so the server runs `forward_core` without dragging the held-out
  jax/numba boundary into the mypy gate.
- `chocofarm/az/zmq_net_client.py` — the remote `Net` impl (`ZmqNetClient`), a blocking REQ RPC that
  raises loudly on timeout/server-down (§5, no silent local fallback).
- `tests/test_zmq_inference.py` — always-on codec/Protocol/greedy-drain pins + the opt-in
  (`CHOCO_RUN_ZMQ=1`) server+client parity harness (§7), which measured max|Δvalue|≈4.8e-7,
  max|Δlogit|≈2.4e-7 over N=1200 vectors across batch sizes B∈{1..64}, residual ON and OFF — far
  inside the 1e-4 ADR-0012 P6 bar.

The four modules are `mypy --strict`-clean and in `STRICT_CLEAN`; pyzmq (27.1, ships py.typed) needs
no `ignore_missing_imports` override. The new dependency is **pyzmq** on the shared scratch venv.

### Amendment 2026-06-16 — #23 mechanized: one SSOT per layout + a drift net (ADR-0005 Rule 8: append)

The forward-looking record above (§2 "designed but not yet generated", §6 "one frame spec, two codecs;
#23's mechanization target") and `docs/design/cpp-gumbel-search-port.md`'s "#23 result-format codegen"
phrasing are preserved as written. As of this amendment **#23 is mechanized.**

**Where this sits on P7's own enforcement hierarchy (stated honestly).** ADR-0012 P7 orders the
mechanisms *generate-or-compile-from-one-source > build-time lint > runtime parity test*, names the
static result format as "exactly what codegen/lint is warranted [for]," and **forbids justifying a
weaker bar with a scale / minimality / "proportionate" / "for now" argument** — that argument shape is
the tell P7 exists to reject. So this amendment does NOT claim "enforce-by-test instead of codegen
because codegen is heavy" (that would be exactly the forbidden shape). The honest mapping:

- **The always-on layout-agreement test IS P7's FLOOR.** P7's floor is "a build-time lint that fails
  the build on a Python/C++ format-constant disagreement." There is **no C++ build in any default
  gate** (the C++ side — `ZmqNetClient` and the redis-client `cpp/` build — is deferred; there is no
  `cpp/build/` in CI), so a literal compile-time lint has no build to attach to yet. `tests/
  test_wire_drift.py`'s always-on legs are the available form of that floor: they parse the C++ mirror
  headers' constants and **fail the standard `pytest tests/ -q` gate** on a format-constant
  disagreement, with no C++ build or fixtures required. This catches drift *unconditionally on every
  run*, which is strictly stronger than P7's "backstop" caveat ("catches drift only if it runs, with
  the right fixtures, after the drift exists").
- **The opt-in C++ golden round-trip IS P7's BACKSTOP** (the runtime parity test) — the stronger
  end-to-end check when a C++ toolchain is present.
- **The top rung (generate/compile-from-one-source) is deferred for a CONCRETE reason, not a
  minimality one: the consumer it would generate for does not exist yet.** Codegen emits a derived
  reader for a specific compiled consumer; the C++ codec is deferred to the P9 `cpp/` pass (§9). When
  that pass lands and the C++ side is built in a gate, the floor SHOULD be promoted to the top rung —
  generate `wire_spec.hpp` / `result_spec.hpp` from the Python SSOT so the mirror is *derived, not
  hand-written* (the residual gap today: the mirror headers are hand-authored, joined to the SSOT only
  by this test). **This is recorded as the open promotion**, per ADR-0011 Rule 1 (declare the
  enforcement surface so the weaker-than-top choice is challengeable), and BACKLOG'd.

What landed:

- **Two single-source-of-truth layout modules** (the "one home per layout", ADR-0012 P1/P7):
  `chocofarm/az/wire_spec.py` (the ZMQ wire frame's protocol version, byte order, field widths, f32
  dtype) and `chocofarm/az/result_spec.py` (the redis result blob's dtype, block order X/PI/M/Y, ranks).
  The Python codecs DERIVE from them (`inference_wire.py` builds its `struct.Struct` formats from
  `wire_spec`; `transport.py` reads `result_spec.RESULT_DTYPE` / `BLOCK_ORDER`, never a hardcoded
  `np.float32` / `<f4`). The C++ mirror headers (`cpp/include/chocofarm/{wire_spec,result_spec}.hpp`)
  declare the SAME constants for the deferred C++ side to derive from. (Provenance note: the SSOT
  modules + mirror headers are part of the same uncommitted working-tree batch as this drift net —
  earlier prose calling them "already landed" overstates a separate provenance the VCS does not
  corroborate; they and the net are one change.)
- **The drift net** — `tests/test_wire_drift.py`, several always-on legs in the default suite (no C++
  binary, no redis): a LAYOUT-AGREEMENT leg parses the C++ mirror headers' `constexpr` literals and
  asserts equality with the Python SSOT (version, widths, dtype, block order/ranks); a
  CODEC-DERIVES-FROM-SPEC leg that drives the ACTUAL codecs and asserts the bytes they emit/decode are
  exactly the spec's little-endian-f32 bytes (built from an independent spec-derived reference, and —
  for the result blob — round-tripped through `write_results`→`read_and_delete_results` over an
  in-memory fake redis), so a codec that drifted its OWN float interpretation away from the spec (e.g.
  read the payload as `>f4`) reds even though the mirror constants still agree; a NEGATIVE mutation
  self-check proving the agreement fails on a deliberate one-sided perturbation (so the net is
  demonstrated to catch drift, not merely pass); a WEIGHT-MANIFEST leg pinning the one shared `<f8`
  cross-language literal. Plus one OPT-IN leg (`CHOCO_RUN_CPP=1`): Python encodes golden vectors → a
  standalone C++ decoder (`cpp/parity/wire_golden.cpp`, compiled with a bare `g++ -std=c++23` over ONLY
  the mirror headers) round-trips them → byte-exact assertion. A deliberate one-sided drift of the real
  C++ header was confirmed to red the default suite, then reverted; and a codec-side float-dtype drift
  (`_F32`/the reader dtype → `>f4`) was confirmed to red the codec-derives-from-spec legs.
- **The weight manifest (contract #3) deliberately NOT over-mechanized.** It is a self-describing JSON
  manifest (each entry carries its own name/shape/dtype/off/len), so a reader derives the layout from
  the bytes — it is the **dynamic** layout P7 line 348-352 explicitly sanctions a runtime-read manifest
  for ("absorbed, not drifted"), and gets no separate spec module. The ONE cross-language fact the
  manifest does not make a reader re-derive — "the weight blob is float64 (`<f8`)" — is pinned by the
  drift net's weight-manifest leg. (Residual, recorded: the C++ `parse_manifest` derives a weight's
  element count from `len/sizeof(double)` while Python `unpack_into` uses `prod(shape)`; both are
  consistent today because one writer (`pack`) emits both, but C++ does not cross-check
  `prod(shape)*8 == len` — a one-line hardening for the `cpp/` pass.)

The two SSOT modules are `mypy --strict`-clean and added to `STRICT_CLEAN`. The default suite grew from
182 to 192 passed with 10 skipped (+2 opt-in C++ golden legs).

*Public Domain (The Unlicense).*
