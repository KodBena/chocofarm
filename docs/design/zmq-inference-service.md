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

*Public Domain (The Unlicense).*
