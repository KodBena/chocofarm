# throughput-lab

A **clean-room synthetic-load throughput testbed** that isolates the
producer → boundary → server throughput of the chocofarm leaf-eval serving
path **from the tree search**. It answers one question honestly: how many
leaf-batches per second can flow `producer → boundary → server → reply`, as a
function of the boundary topology, the producer mode, the thread count, and the
batch size — with the search removed so the number is a property of the
serving wire and the server's receive/compute overlap, not of the MCTS that
normally drives it.

It is **self-contained**. The only code lifted from chocofarm is the MLP
forward and its phantom-typed jax/numpy ACL (copied verbatim into
`server/lifted/`); everything else is re-implemented fresh. The **wire** is
matched byte-for-byte to chocofarm's live producer↔server path so the server is
comparable.

Public Domain (The Unlicense). Per ADR-0006 every source file opens with a
header (path + one-line purpose + Public Domain).

---

## Architecture — three components

```
   ┌─────────────────────┐        ┌──────────────────┐        ┌──────────────────────┐
   │   PRODUCER (C++)     │        │  BOUNDARY (C++)  │        │    SERVER (Python)   │
   │  N threads, each     │ submit │  the transport   │  ZMQ   │  decoupled receiver  │
   │  self-CALIBRATES its │──────► │  DEALER ↔ ROUTER │──────► │  + batched MLP fwd   │
   │  rate (x+=1 spin →   │        │  Topology A or B │  ipc   │  + per-request       │
   │  ops/sec), emits     │ ◄──────│  (one socket per │ ◄──────│  scatter             │
   │  241-float rows at a │ reply  │  thread, or one  │        │                      │
   │  dialable rate       │        │  coalescing thr) │        │                      │
   └─────────────────────┘        └──────────────────┘        └──────────────────────┘
        DECOUPLED / COUPLED            the typed seam               receiver ⊥ compute
```

### Producer (C++, `cpp/producer.hpp`, `cpp/main.cpp`)

`N` threads, one socket per thread by default. On start each thread
**calibrates its own compute rate** by spinning a timed `x += 1` loop and timing
it to recover ops/sec, then emits synthetic leaf-batches (rows of 241 float32 =
the Stage-A `in_dim`) at a **known, dialable rate** by doing a calibrated amount
of `x += 1` busy-work per production — the rate is hardware-calibrated, **not** a
fragile `usleep`. The calibration protocol is documented in
`cpp/producer.hpp` (STEP 1 calibrate ops/sec → STEP 2 derive busy-work per
production → STEP 3 emit at rate; every achieved rate is **measured** and
reported next to the requested rate).

**Producer mode is a plug** (ADR-0012 — not the only mode the seam admits), and
**both are built and selectable**:

- **DECOUPLED** — free-run at the calibrated rate; replies are received and
  their latency measured, but they do **not** gate production.
- **COUPLED** — wait for each batch's reply before producing more, emulating the
  real search's leaf-eval **on the critical path**.

### Boundary (C++, abstract, `cpp/boundary.hpp`)

The producer → server transport, behind a **typed seam** (ADR-0012 P8 — the
interface is the SSOT; the topologies are two impls of it, and the seam admits
more):

- **Topology A** — one outbound DEALER socket **per producer thread** (today's
  chocofarm shape).
- **Topology B** — the producer threads feed a **separate coalescing thread**
  that holds **one** socket and coalesces their production before sending.

The boundary owns the ZMQ transport (the DEALER socket(s), the correlation-id
stamping, the multipart framing); it is instrumented for throughput, latency,
and utilization. It never computes the forward and never parses the
correlation id (it round-trips it opaquely).

### Server (Python, `server/server.py`)

A **fresh** receive/serve loop with the **receiver decoupled from the
compute** — the hard-won finding: the serial drain→forward→scatter forfeits
overlap, and in a free-running producer a decoupled receiver does **not**
deadlock the way a coupled one does. The receiver drains the ROUTER socket
independently of the forward; the compute gathers the largest available concat,
runs **one** batched MLP forward, and scatters per-request replies. The MLP
forward and the phantom-typed jax/numpy ACL are the **only** parts lifted from
chocofarm (`server/lifted/`, copied verbatim); the loop, the transport, and the
scatter are re-implemented here.

---

## The wire (matched to chocofarm, byte-for-byte)

Two layers, kept apart (ADR-0012 P7: serialization ⊥ transport). The full
byte-level spec lives in **both** `cpp/wire.hpp` and `server/wire.py` (two views
of one truth, not two authors). Summary:

**Layer 1 — the value frame** (length-prefixed little-endian float32, fronted by
a one-byte protocol version; `PROTOCOL_VERSION = 2`; all multi-byte fields LE):

```
Request  : [ver:u8][B:u32 LE][in_dim:u32 LE][X : f32 × (B·in_dim) LE]   (X row-major)
Response : [ver:u8][B:u32 LE][n_actions:u32 LE][ B × (value:f32 LE, logits:f32 × n_actions LE) ]
```

Fixed header = 9 bytes (`1 + 4 + 4`). `B = 1` is the degenerate single-leaf
case (the batched frame subsumes single-leaf). `n_actions == 0` ⇒ value-only
(empty logits blocks). The response value is **de-standardized**
(`v = v_std·y_std + y_mean`); the logits are **raw** (not softmaxed) — masking
stays client-side.

**Layer 2 — the ZMQ transport envelope** (DEALER producer ↔ ROUTER server):

```
producer DEALER sends :  [ corr-id : u64 (8 raw native-endian bytes) ] [ <Layer-1 request> ]
server  ROUTER recv   :  [ identity ] [ corr-id ] [ <Layer-1 request> ]   (ZMQ prepends identity)
                          frames[0]    frames[1:-1] is the OPAQUE echoed envelope    frames[-1]
server  ROUTER sends  :  [ identity ] [ corr-id ] [ <Layer-1 response> ]
producer DEALER recv  :  [ corr-id ] [ <Layer-1 response> ]               (ZMQ strips identity)
```

The correlation id is a **transport concern** the producer stamps and the server
round-trips byte-for-byte without ever parsing — it stays out of the Layer-1
value codec. This matches chocofarm's `cpp/.../wire_leaf_pool.hpp` (DEALER,
`[corr-id][payload]`) and `az/inference_server.py` (`frames[1:-1]` opaque
envelope) exactly.

---

## Folder layout

```
throughput-lab/
├── README.md                     ← this file (architecture, contracts, build + run)
├── cpp/
│   ├── wire.hpp                   ← SSOT byte spec + Layer-1 codec (header-only, transport-free)
│   ├── boundary.hpp              ← the abstract Boundary seam (Topology A/B; the typed SSOT)
│   ├── producer.hpp              ← the producer seam: modes (coupled/decoupled) + calibration protocol
│   ├── zmq_context.hpp           ← the process-global ZMQ context the dealers share (header-only)
│   ├── zmq_dealer.hpp            ← the shared Layer-2 DEALER framing both topologies ride (header-only)
│   ├── boundary_per_thread.cpp   ← Topology A impl (one DEALER per producer thread)
│   ├── boundary_coalescing.cpp   ← Topology B impl (producer threads → one coalescing thread → one DEALER)
│   ├── boundary_factory.cpp      ← make_boundary() dispatch on BoundaryTopology
│   ├── producer.cpp              ← run_producer: calibration spin + the two ProducerModes + instrumentation
│   ├── main.cpp                   ← producer entry point (imperative shell; CLI ACL → run_producer → report)
│   └── CMakeLists.txt            ← build scaffolding (tlab-producer target, libzmq, C++23)
├── server/
│   ├── __init__.py
│   ├── __main__.py                ← server entry point (CLI ACL → ThroughputServer)
│   ├── wire.py                    ← Python view of the wire (Layer-1 codec + Layer-2 docs)
│   ├── server.py                  ← the decoupled receive/serve loop (ROUTER; receiver ⊥ compute)
│   └── lifted/                    ← the ONLY parts COPIED from chocofarm (verbatim)
│       ├── __init__.py
│       ├── forward.py             ← forward_core (the MLP graph) — copied verbatim
│       ├── dtypes.py              ← the float32/64 precision pin — copied
│       └── mlp_forward.py         ← the jax-jit wrapper over forward_core (server compute)
└── harness/
    └── run_lab.py                 ← run+measure orchestration (server + producer + sweep report)
```

All files are implemented (the skeletons the two build agents were handed are
filled); the testbed builds clean and runs the full
`{per-thread,coalescing} × {decoupled,coupled}` matrix end-to-end — see
**Verified state** below.

---

## Build

### C++ producer + boundary (CMake + libzmq, C++23)

```sh
cmake -S throughput-lab/cpp -B throughput-lab/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build throughput-lab/cpp/build -j
# → throughput-lab/cpp/build/tlab-producer
```

Requires `libzmq` (`libzmq3-dev` / `zeromq-devel`; the build finds it via
pkg-config, falling back to `find_library`). **C++ standard: C++23.** The brief
said "C++17", but the contracts use the modern types ADR-0012 P9 mandates —
`std::span` (C++20), `std::expected` (C++23), `std::optional` (C++17) — and P9
is exactly the discipline the brief invokes, so the standard that honors the
contracts is C++23 (also what chocofarm's own `cpp/` uses, so the testbed
matches the parent toolchain). The wire codec and the contract headers are
verified to compile clean under `-std=c++23 -Wall -Wextra`.

### Python server

No build step — pure Python. The interpreter is the project's shared scratch
venv (JAX, numpy, pyzmq):

```sh
/home/bork/w/vdc/venvs/generic/bin/python --version
```

---

## Run

Server (binds the ROUTER, builds a random net of the live shapes, warms up XLA
for every batch bucket it pads to, then serves; prints `[tlab-server] READY …`
on stdout once warm — the harness waits on that line):

```sh
PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python -m server \
    --bind ipc:///tmp/tlab-infer.sock --in-dim 241 --n-actions 0
```

Producer (against the same endpoint; prints one `RESULT thread=…` line per
thread and one `AGGREGATE …` line, both machine-parseable):

```sh
throughput-lab/cpp/build/tlab-producer \
    --endpoint ipc:///tmp/tlab-infer.sock \
    --topology per-thread --mode decoupled --threads 4 --rate 5000 --rows 1 --seconds 5
```

Or drive both through the harness — it stands up a **fresh server per cell**
(clean server-side stat window), waits for the `READY` line (so the producer
never pays XLA warmup), runs the producer across the matrix, tears the server
down bounded (SIGINT → kill fallback), and emits one JSON record per cell:

```sh
# full matrix {per-thread,coalescing} × {decoupled,coupled}, 2 threads @ 2000 hz, 3 s:
PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python harness/run_lab.py

# a load-bearing sweep: more threads/rate, replicates (achieved-rate median), a persisted artifact:
PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python harness/run_lab.py \
    --threads 4 --rate 5000 --seconds 5 --replicates 3 \
    --json-out ~/w/vdc/chocobo/runs/tlab/sweep.json

# a single cell (subset the matrix):
PYTHONPATH=throughput-lab /home/bork/w/vdc/venvs/generic/bin/python harness/run_lab.py \
    --topologies per-thread --modes coupled
```

The human summary table goes to **stderr**; the JSON record array goes to
**stdout** (and to `--json-out` if given). `run_lab.py --help` lists every knob.

> **Coupled-mode reply latency.** In COUPLED mode a thread sends one batch and
> blocks on its reply. The server's IO thread parks in
> `poller.poll(poll_timeout_ms)`, but a **reply wake pipe** (an inproc ZMQ PAIR
> the compute thread pokes the instant a reply is ready; see *THE REPLY WAKE* in
> `server/server.py`) makes that poll return immediately — so a coupled
> round-trip reflects **compute + wire**, not the poll granularity, at **any**
> `poll_timeout_ms`. Measured (1 thread, ipc, `--poll-timeout-ms 50`, small net):
> **~0.5 ms median RTT**. `poll_timeout_ms` now only bounds how fast the loop
> notices `stop()` and the idle wake cadence; it no longer floors coupled RTT.
> DECOUPLED mode is likewise unaffected.
>
> > **Update (2026-06-23, wake pipe).** Earlier this file documented a
> > `poll_timeout_ms` **floor** on coupled RTT (poll 50 ms → ~50.8 ms p50;
> > 5 ms → 5.4 ms; 1 ms → 1.3 ms), dialed around with `--poll-timeout-ms 1`.
> > That floor was clause 6 of the ratified resolution and is now **removed** by
> > the wake pipe — coupled RTT is low at any timeout, so the dial-down caveat no
> > longer applies.

---

## Verified state

Built and smoke-measured by the integrate phase (2026-06-23):

- **C++:** clean from-scratch CMake build, **zero warnings** under
  `-Wall -Wextra -std=c++23` (GCC 15.2, libzmq 4.3.5) → `cpp/build/tlab-producer`.
- **Python:** all 8 modules import + run under
  `/home/bork/w/vdc/venvs/generic/bin/python` (JAX 0.10.1, numpy 2.4.6,
  pyzmq/libzmq 4.3.5).
- **End-to-end:** the full matrix
  `{per-thread, coalescing} × {decoupled, coupled}` round-trips
  (producer → boundary → server → reply), `sent == recv`, **0 rejects**.
  Representative cell (2 threads @ 5000 hz/thread = 10 000 hz requested, 3 s,
  server `--poll-timeout-ms 1`, **4 replicates** — achieved-rate median;
  pinned `taskset -c 0-3`, watchdog-wrapped):

  | topology   | mode      | req hz | ach hz | p50 lat | srv util | mean batch |
  | ---------- | --------- | -----: | -----: | ------: | -------: | ---------: |
  | per-thread | decoupled | 10000  | ~8641  |  ~85 ms |   ~56 %  | ~24 rows   |
  | per-thread | coupled   | 10000  | ~1210  | ~1.5 ms |   ~25 %  | ~1.3 rows  |
  | coalescing | decoupled | 10000  | ~9520  | ~2.3 ms |   ~68 %  | ~5.9 rows  |
  | coalescing | coupled   | 10000  | ~695   | ~2.6 ms |   ~15 %  | ~1.0 rows  |

  Read: DECOUPLED free-runs ahead of the server, so a genuine request **backlog**
  forms (achieved < requested) and queue depth shows up as reply latency — worse
  for `per-thread` (~85 ms p50, many tiny Layer-2 messages) than for `coalescing`
  (~2 ms p50, fewer/larger messages the server keeps pace with). COUPLED is
  RTT-bound (batch ≈ 1, low util, low latency once the poll floor is lifted). The
  `srv util` is now genuine matmul time — see the correction note below.
  These are smoke numbers (a short window, on a 4-vCPU VM) — for a load-bearing
  claim raise `--replicates`, pin cores, and wrap in
  `tools/shell/compute-watchdog.sh`.

  > **Correction (2026-06-23, same-day).** The original decoupled rows here read
  > `~201–207 ms p50 / ~85 % util / ~860 rows` and attributed the latency to
  > transport backlog. That was a measurement **artifact**: the server jitted one
  > XLA kernel per `(rows, in_dim)` shape and warmed only a fixed set, but the
  > compute thread gathered an *arbitrary* row count — so nearly every decoupled
  > forward paid a ~50 ms XLA **recompile** inside the timed window (mis-read as
  > compute-busy), and the slow forwards backed requests into ever-larger
  > novel-shape batches (a self-reinforcing recompile storm). The server now
  > rounds each batch up to a warmed **bucket** and zero-pads to it (production
  > `InferenceServer`'s discipline); the rows above are the post-fix re-measure.

  > **Re-measure note (2026-06-23, ratified-resolution hardening).** The two
  > **coupled** rows above were taken at `--poll-timeout-ms 1` specifically to
  > dodge the old reply poll-floor. The reply **wake pipe** has since removed that
  > floor (coupled RTT is now compute+wire-bound at any timeout — see the
  > *Coupled-mode reply latency* note under **Run**), so the coupled `p50 lat`
  > cells should be **re-measured** without the `--poll-timeout-ms 1` caveat. The
  > **decoupled** rows are unaffected. The server's boundary, teardown, and
  > reply-error handling were hardened in the same change (a refined
  > `BoundedBatch` wire type that rejects oversize/wrong-width frames per-identity;
  > a fail-loud compute thread; a true-join teardown that drops no request); none
  > of these moves a number on a well-formed run (`sent == recv`, 0 rejects).

---

## Discipline

- **ADR-0012 P7** — the wire is one truth in two views (`cpp/wire.hpp`,
  `server/wire.py`); serialization (Layer 1) is kept apart from transport
  (Layer 2, the corr-id envelope).
- **ADR-0012 P8** — the typed signatures are the SSOT: the boundary topology
  and the producer mode are **plugs**, not the only impls the seams admit.
- **ADR-0012 P9** — honest C++ signatures: bounds-carrying `std::span`, typed
  `std::expected`/`std::optional` for failure/absence (no untyped-effectful-void,
  no nullable-pointer/sentinel), RAII, `create()` factories over throwing ctors.
- **ADR-0002** — fail loudly at the boundary: a malformed / oversize /
  wrong-width / wrong-dtype frame, a codec mismatch, an absent server (recv
  timeout) is a loud typed failure, never a zero-filled forward or a silent
  fallback. The server narrows each request to a refined `wire.BoundedBatch`
  (1 ≤ rows ≤ max_batch, cols == in_dim, float32) at the door, so an illegal
  shape is one per-identity reject — not a crash three layers down at the
  forward, the bucket ladder, or inside `np.concatenate`.
- **ADR-0006** — every source file opens with its header.
- **ADR-0009** — every throughput/latency claim is **measured** and reported
  next to what was requested; the server warms up XLA before the timed run AND
  rounds every gathered batch up to a warmed bucket (zero-padded) so no
  per-batch-size compile lands in the timed window and is mis-read as jitter (the
  fixed warmup set alone is not enough — the gather produces arbitrary row counts;
  see the same-day correction note under *Verified state*).

The maintainer inspects the code by hand before any commit — favor transparent,
well-documented, simple code over cleverness.
