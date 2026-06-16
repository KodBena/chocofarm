<!-- docs/notes/consult/opus-consult-2026-06-16-zmq-net-client-blocking-req.md -->

# Consult — the C++ `ZmqNetClient` (blocking `ZMQ_REQ`) vs. ADR-0012 and the async work-stealing goal

**Provenance / frame (read first).** This memo is authored **from primary
sources only**: ADR-0009, ADR-0011, ADR-0012 (each read end to end), the design
records `docs/design/cpp-batched-search.md` and `docs/design/zmq-inference-service.md`
(end to end), `docs/design/scaling-and-cpp-seam.md`, the C++ files
(`cpp/{src,include/chocofarm}/`), and the commit `d06db93` (merged at `51b13b9`).
It **deliberately excludes** `docs/handoff-2026-06-16-zmq-async-gumbel.md` as an
input — that record was identified by the maintainer as a biasing artifact, and
an earlier pass of this very investigation demonstrably inherited its framing.
This memo does **not** adjudicate the deleted "consult-004": there is no
primary evidence in the tree about its content or correctness, so it is neither
cited nor relied on, in either direction. The guard held throughout was *not* to
replace the handoff's bias with its mirror image — the verdict lands where the
ADRs and the code put it, and some of the criticism survives.

**Scope:** the maintainer's three questions, verbatim.
1. The client vs. ADR-0012 and the goal of an asynchronous work-stealing loop.
2. The sunk cost and the cost of rectifying it.
3. Whether all of it must be discarded.

---

## TL;DR

- **It is not an ADR-0012 P7 violation.** P7 governs the cross-language **wire**
  (the serialization SSOT), and on that axis the change is exemplary: one codec
  home (`inference_wire.hpp`), every field derived from the `wire_spec` SSOT,
  drift-net-checked, and it **deleted** a former second hand-codec. P7 *itself*
  separates the serialization contract from the transport mechanism and says in
  as many words "do not enshrine ZeroMQ as 'the one way'." Calling a blocking
  `REQ` socket "the P7 weaker mechanism" conflates the two axes P7 keeps apart.
- **The real, narrower finding is on two *different* axes.** (a) A blocking
  `ZMQ_REQ` is the transport for **Embodiment 1** (thread-per-inflight) of the
  work-stealing loop and is **structurally incompatible with Embodiment 2** (the
  single-thread DEALER/fiber multiplexer). Which embodiment a 4-vCPU host wants
  is an **explicitly benchmark-gated, not-yet-measured** question (the design's
  own ADR-0009 gate). (b) The port + client were merged **ahead of any
  consumer** — but that absence is **disclosed loudly in the commit**, which is
  ADR-0009's sanctioned shape, not a silent failure.
- **Run under the project's own hack-rationalization-detector, this is
  `narrower-but-justified`, not `UNDISCHARGED-HACK`.** The tells scanner is
  clean on the commit; the one deferral (the multiplexer) names a **concrete
  cost** and files a real follow-up, which is the justified-narrowing signature.
- **Sunk cost is small; "discard all of it" is unsupported.** The codec, the
  port concept, the decode/de-standardize logic, and the parity harness are
  keepers (the codec is a P7 *gain*). The only piece coupled to lock-step is the
  `REQ` socket lifecycle + the blocking `predict` body, and even that has
  **present** value (it is a working Shape-B synchronous client and a
  cross-language parity proof). The dominant cost of the async loop — the
  multiplexer and the search's continuation refactor — is **intrinsic to the
  goal and is not increased by this client's existence.**

The honest one-line answer to "is this a mishap to be discarded?": **No. It is a
correctly-built Shape-B synchronous leaf evaluator, sequenced one step ahead of
its consumer, whose transport happens to serve one of the two async embodiments
and not the other — a fact that becomes a problem only once a benchmark shows
Embodiment 1 inadequate, which has not been run.**

---

## §0 — The one load-bearing distinction (do not conflate these)

ADR-0012 P7 is titled *Cross-language **wire** discipline*. Its subject is a
**cross-boundary fact** — "a layout, a key, a byte format" — which "has exactly
**one authoritative definition**; every side **derives** its view." Its
enforcement hierarchy (`generate-or-compile-from-one-source > build-time lint >
runtime parity test`) and its famous no-excuse clause ("*Never* justify settling
for a weaker mechanism with a scale / minimality / 'one X' / 'for now' /
'unnecessary here' / YAGNI argument") both quantify over **that** mechanism — the
mechanism that keeps two codecs from drifting.

P7 then, in its own text, **draws a fence around the transport**:

> Separate the **serialization contract** from the **transport/coordination
> mechanism** — the durable rule is mechanism-independent. … Do not enshrine
> redis as "the contract," and do not enshrine ZeroMQ as "the one way" either —
> they are instances of (bytes-store) and (messaging fabric).

So the client must be judged on **three distinct axes**, and they return **three
different verdicts**. The failure mode this memo exists to avoid is collapsing
them — which yields either "it's a P7 hack, discard the transport" (the wire axis
laundered onto the transport axis) or "the wire is clean, so it's all fine" (the
transport axis laundered onto the wire axis). Both are wrong for the same reason.

| Axis | What it measures | Verdict |
| — | — | — |
| **1. Wire / SSOT** (P7's actual subject) | does a second hand-author of the byte format exist? | **Compliant, exemplary** (a P7 *gain*) |
| **2. Transport concurrency** (vs. the async goal) | can this socket serve the work-stealing loop? | **Serves Embodiment 1; incompatible with Embodiment 2** |
| **3. Earns-its-keep / consumer** (ADR-0009) | does a consumer exist; is the absence disclosed? | **Premature by one step, disclosed loudly** |

---

## §1 — Axis 1: the wire / SSOT axis (where the P7 accusation would have to land)

**Where it is sound.** There is exactly one C++ definition of the frame:
`cpp/include/chocofarm/inference_wire.hpp` is the sole site of `encode_request`,
`decode_request`, `encode_response`, `decode_response` (`:85`, `:97`, `:127`,
`:141`). It "**DERIVES** every byte width / version / order from the SSOT mirror
`chocofarm/wire_spec.hpp`" (`:3–5`), the Python codec derives the same layout
from `chocofarm/az/wire_spec.py`, and "the two specs are drift-checked in the
default Python suite (`tests/test_wire_drift.py`)" (`:6–7`). The client drives
that one codec (`zmq_net_client.cpp:129` `wire::encode_request`, `:166`
`wire::decode_response`); it re-authors nothing.

The commit went **further than non-violation**: `cpp/parity/wire_golden.cpp`'s
prior inline passthrough codec was deleted and "**re-pointed at the shared one**"
(commit `d06db93`; `wire_golden.cpp:93–103` now call `chocofarm::wire::…`). So
the net effect on the count of hand-authored codecs is **−1**. Under ADR-0011
Rule 4 (a net "keys on … a derived-from-one-source invariant"), the cross-boundary
fact is *mechanized*, not left to convention.

**Hack-detector writer delta, run honestly:** claimed = "one shared codec";
independently enumerated = one definition home + two derivers
(`zmq_net_client.cpp`, `wire_golden.cpp`) + a drift net. **Confirmed single-home.**
This is the *inverse* of the documented per-writer-gate hack (Case A): the change
*reduces* writers rather than gating them one at a time.

**Where it cannot reach.** A clean bill on Axis 1 says **nothing** about the
transport. P7's own fence forbids carrying this verdict over to Axis 2. The wire
being exemplary does not make the blocking transport right; it only forecloses the
specific accusation "this is a P7 wire violation," which is the accusation the
phrase "ADR-0012 P7 hack" most naturally implies.

---

## §2 — Axis 2: transport concurrency vs. the async work-stealing loop (the genuine collision)

This is where the substance is. The maintainer's stated goal
(`cpp-batched-search.md §0`) is explicit: a pool of workers, each advancing its
own tree, that "pushes the leaf over ZeroMQ, then goes on to fetch the next work
unit **until a maximum number of in-flight work units**." *Many outstanding
requests* is the defining property.

A blocking `ZMQ_REQ` socket has the opposite property by construction. The client
itself says so: "REQ is strict send→recv lock-step so one client is NOT
thread-safe (give each worker its own)" (`zmq_net_client.hpp:25–26`); `predict`
is `send` (`:130`) then `recv` (`:145`) with exactly **one** call outstanding.
The design record states the consequence directly: the built `ZmqNetClient` "is a
**blocking REQ** socket … **one in flight, not thread-safe** … It **cannot**
multiplex many parked trees on one socket" (`cpp-batched-search.md §3.4`).

The design then names the two embodiments, and this is the crux of Question 1:

- **Embodiment 1 — thread-per-inflight.** "One OS worker per tree; at the leaf
  the worker **blocks** in its own `ZmqNetClient.predict`. The server's greedy
  drain batches whatever REQ requests are concurrently in flight. `B` is bounded
  by thread count and a blocked worker is a parked core — acceptable **only when
  threads ≫ cores**." **The blocking `REQ` is precisely this embodiment's
  transport.** It is not useless against the async goal — it is one valid
  realization of it.
- **Embodiment 2 — DEALER submit/poll multiplexer.** One DEALER socket (many
  outstanding sends) + parked fibers, requiring "a **continuation refactor** of
  the synchronous `WorldSource::playout_value` / `Net::predict` in C++ — **the one
  structural gap**" (`§3.4`). **The blocking `REQ` cannot be this embodiment's
  transport; it is bypassed.**

**The false friend (name it, then disarm it).** The seductive reading is "the
`NetEvaluator` port makes the transport a drop-in swap, so going async is free
later." It is half-true and the dangerous half is false. The port
(`net_evaluator.hpp:49–58`) makes **local↔remote-blocking** swappable
(`NetForward` ↔ `ZmqNetClient`, same `predict(span)->expected<NetPrediction>`).
It does **not** make **blocking↔async** swappable: the async leaf call is
*submit-and-yield*, a shape the synchronous `predict()->value` signature cannot
express. That is exactly why Embodiment 2 needs the continuation refactor the
design calls "the one structural gap." So the port abstraction does **not**
absorb the blocking→async transition; the search's call site must change.

**The reality collision, stated precisely.** Against the async goal:
1. The blocking `REQ` **is** the Embodiment-1 transport and **cannot** be the
   Embodiment-2 transport.
2. Which embodiment a 4-vCPU host should run is **explicitly benchmark-gated and
   unmeasured**: `cpp-batched-search.md §6` open question 5 (an ADR-0009 gate)
   asks, in the maintainer's own words, "does the DEALER+fiber multiplexer
   actually beat M dumb blocking-REQ workers? **Benchmark before committing to
   the multiplexer's complexity.**"

So the premise that would make the blocking client a *mishap* — "Embodiment 1 is
inadequate, therefore the lock-step transport was the wrong build" — **is an
unmeasured claim.** The design itself declines to assert it. This is the honest
heart of Question 1: the blocking `REQ` is incompatible with Embodiment 2 (true),
a valid transport for Embodiment 1 (true), and the choice between them is an open
measured question (so "mishap" is not established from primary evidence).

**Where the criticism still bites.** Even granting Embodiment 1 is valid, the
blocking `REQ` is the embodiment the design flags as "acceptable **only** when
threads ≫ cores" — the *least* attractive shape on a 4-vCPU host. If the
maintainer's committed near-term target is specifically Embodiment 2, then the
`REQ` transport is effort spent on the path the target skips. That is a real
risk — but it is a **direction question for the maintainer**, not a discipline
violation in the artifact.

---

## §3 — Axis 3: earns-its-keep / the missing consumer (ADR-0009, honestly)

**Where it bites.** On `main` (`51b13b9`), the `NetEvaluator` port has **two
impls and zero polymorphic consumers.** Independently enumerated: the only
constructor of `ZmqNetClient` anywhere is the parity probe
(`zmq_net_probe.cpp:68`); `NetForward` is consumed only by a dump tool
(`net_dump.cpp`) and the parity harness (`cpp/parity/net_parity.py`); no C++
search dispatches through `NetEvaluator` (the ported searches — NMCS/ISMCTS — use
the `WorldSource` determinization seam, not the leaf-eval port). A concrete cost
rides along: `NetForward::predict` now returns `std::expected<…, Error>` whose
error arm is **permanently dead** — "always returns the VALUE arm … the
`std::expected` return is the NetEvaluator port shape … **not a real error
surface**" (`net.cpp:206–207`).

This is the shape of audit anti-pattern **E** ("abstraction built then abandoned")
— with the twist that there is no live inline copy beside it either; there is
simply **no consumer yet**. ADR-0012's preventing rule for E is **P5** ("adopt or
delete"). Right now the port is neither adopted (no caller) nor deleted.

**Where it does not bite (and this is decisive for the verdict).** The absence is
**disclosed loudly in the commit message**: "NetEvaluator has no polymorphic
consumer yet (the Gumbel search isn't ported) — the port is correct at the type
level, its swap-impls payoff lands when option 1 wires it." That is **exactly**
ADR-0009's sanctioned posture: *"The absence of substantiation does not block a
change from landing … but the write-up **states the absence explicitly** rather
than carrying an unsubstantiated claim."* A loudly-marked deferral is the
opposite of a silent failure.

Two further ADRs cut the same way and are worth stating because they invert the
intuitive "you should have built async-first" reflex:

- **ADR-0011 Rule 3 (measure-first)** *supports* not building the multiplexer
  yet: "a mechanism is adopted against a measured baseline, not an assumed one."
  Building the DEALER+fiber machinery speculatively, before the benchmark of
  §6-Q5, would itself violate measure-first.
- **The port is *over*-generalized, not under-generalized.** The
  hack-detector's "failed-to-generalize" fingerprint is N patches where one
  invariant would do. Here the single invariant (leaf-eval-behind-a-port) was
  built *before* it had even one consumer — the opposite failure. The detector's
  documented shapes do not fire; the honest label is **prematurity**, which the
  detector does not target.

A fair reading of Axis 3: **premature by one sequenced step, disclosed loudly.**
Whether one-step-ahead sequencing (land the remote impl on the P9-compliant port
during the P9 `cpp/` pass, per `zmq-inference-service.md §9`) is acceptable is a
judgment the maintainer owns — it is defensible, and it is honestly flagged.

---

## §4 — The hack-rationalization-detector artifact (verbatim)

Run out of frame (this memo did not author `d06db93`), with the commit message
treated as the object of suspicion.

```
## Hack-rationalization review: d06db93 (C++ NetEvaluator port + ZmqNetClient)

FRAME CHECK: out-of-frame — did not author the change; commit message treated as suspect.

GENERAL FIX:   "the C++ leaf evaluator is a port the search holds; its transport is
               swappable behind that port" — BUILT (the NetEvaluator port). The larger
               invariant the *async goal* needs ("a leaf call that submits and yields,
               not blocks") is a DIFFERENT, bigger generalization, not attempted here and
               explicitly deferred as Embodiment 2.
PATCH SHIPPED: a blocking-REQ client + a synchronous port + the shared codec + a parity
               probe/test, merged with zero polymorphic consumers.
DOWNGRADE:     the async mechanism the stated goal needs (the DEALER/fiber multiplexer)
               was deferred — but with a NAMED CONCRETE COST (the synchronous→continuation
               refactor, "the one structural gap", §3.4) and a MEASURE-FIRST GATE
               (ADR-0009, §6-Q5), filed as real follow-up (Embodiment 2). A cost, not a mood.
WRITER DELTA:  wire-format writers: claimed "one shared codec"; enumerated = 1 definition
               home (inference_wire.hpp) + 2 derivers (zmq_net_client.cpp, wire_golden.cpp)
               + a drift net. Confirmed single-home; the change DELETED a former hand-codec
               (writers −1). The inverse of the per-writer-gate hack.
RUNTIME:       wire+server+decode path observed faithful by tests/test_zmq_net_cpp.py
               (opt-in), max|Δ| < 1e-4. The async-loop FIT is not a runtime claim (the loop
               isn't built) — N/A, not "unverified-on-paper".

TELLS (Step 1): commit message — NONE (grep_tells clean). cpp-batched-search.md — 14
               minimality-terms / 14 named-fix cues / 0 co-occurrence; the multiplexer
               deferral pairs a minimality flavor with a NAMED cost = justified narrowing,
               not a named-and-bypassed better fix.

VERDICT: narrower-but-justified

WHY: On P7's own axis the change reduces hand-codecs to one SSOT-derived home — a gain, not
     a violation. The single deferral names a concrete cost and files the follow-up, which is
     the justified-narrowing signature, not the mood-based downgrade the hack shape requires.

FINDINGS BEYOND VERDICT (required):
  - The port is OVER-generalized (built before any consumer) — the inverse of the detector's
    "failed-to-generalize"; the detector's fingerprints don't fire, but prematurity is real.
  - A permanently-dead std::expected error arm on NetForward::predict (net.cpp:206-207) is a
    minor cost that exists ONLY because the fallible port was built before its remote consumer.
    (Note: a TOTAL impl of a FALLIBLE port is not itself a lying signature — P9 sanctions it.)
  - Nothing structurally prevents the NEXT reader from re-adopting the proportionality framing:
    the design's "build Embodiment 1 first" language (§3.4) reads as a blanket license unless
    re-anchored to its measure-first gate (§6-Q5). The gate is the load-bearing condition; the
    "first" is downstream of it. This is the one place a future session most easily launders a
    measured-deferral into an unconditional "blocking is fine."
  - The "mishap" premise (Embodiment 1 is inadequate → the lock-step build was wrong) is
    UNMEASURED. The benchmark that would settle it (§6-Q5) has not been run.
```

---

## §5 — Question 2: sunk cost and rectification cost

The commit is 952 insertions / 74 deletions across 13 files (`d06db93 --stat`).
Decompose it by **coupling to the lock-step model**, because that is the only
thing the async goal forces to change:

**Carries forward unchanged — keepers (transport-independent):**
- `inference_wire.hpp` (168 lines) — the shared codec. Encode/decode bytes are
  identical whether the socket is `REQ` or `DEALER`. A **P7 gain**, independently
  valuable.
- `net_evaluator.hpp` (61) — the port *concept* (injected leaf evaluator,
  swappable impl). Its synchronous *method shape* is the one caveat (§2): the
  concept transfers; an async path needs a submit/yield variant beside it.
- The decode → de-standardize → `NetPrediction` assembly in `zmq_net_client.cpp`
  (`:160–173`) and the boundary validation (`:121–127`).
- `zmq_net_probe.cpp` + `test_zmq_net_cpp.py` (100 + 217) — the parity
  *assertions* (value/logits within 1e-4 of `forward_core`) transfer to any
  transport; only the socket underneath would change.
- The libzmq C-API **RAII discipline** (ctx/socket ownership, `LINGER 0`,
  `RCVTIMEO`, `create()->expected` factory) — the *same* discipline a `DEALER`
  client needs; reference value, not loss.

**Superseded *behind the port*, if and when Embodiment 2 is built:**
- The `REQ` socket lifecycle + the blocking `send→recv` body of `predict`
  (`zmq_net_client.cpp:36–75`, `130–158`). This is the genuinely lock-step-coupled
  code — on the order of ~80–100 lines, behind a port, replaced not rewritten-in-place.

**Intrinsic to the async goal regardless of this commit (NOT a rectification cost
of the client):**
- The DEALER submit/poll multiplexer + completion routing + correlation
  (`cpp-batched-search.md §3.4`, Embodiment 2 — "the only genuinely new
  component").
- The search's continuation/fiber refactor so a leaf call can yield ("the one
  structural gap"). **This exists whether or not the blocking client was ever
  built.**

The load-bearing point for Question 2: **the dominant cost of the async loop is
the multiplexer + the continuation refactor, and the blocking client's existence
neither adds to it nor subtracts from it.** The blocking client did not buy down
that cost (it is bypassed), but it also did not inflate it. What it *did* buy,
already realized today, is (i) a working Shape-B synchronous client on the
P9-compliant port and (ii) an end-to-end cross-language parity proof of the wire +
server path. The rectification cost *attributable to having built it* is
therefore ~the one superseded transport layer — small, bounded, and behind a seam.

---

## §6 — Question 3: must all of it be discarded?

**No — and "discard" is the wrong verb for any of it.**

- The codec, the port concept, the decode/de-standardize logic, and the parity
  harness are **keepers** (the codec is a P7 improvement; deleting it would
  *re-introduce* a second hand-codec — a P7 regression).
- The only lock-step-coupled layer (the `REQ` socket + blocking `predict` body)
  is **superseded behind the port** *if* the maintainer commits to Embodiment 2 —
  a replacement of one bounded layer, not a discard, and conditional on a
  benchmark not yet run.
- Nothing is **wasted**, because the present value (the Shape-B synchronous
  client + the cross-language parity proof) is already realized, and the async
  loop is forward-looking / not built.

**What *would* be over-engineering (the symmetric warning):**
1. Discarding the codec or the port to "start clean" — that destroys a P7 gain
   and re-opens the drift surface the #23 net closed.
2. Building the DEALER+fiber multiplexer **now**, speculatively, before the
   Gumbel consumer exists and before the §6-Q5 benchmark — that violates
   ADR-0011 Rule 3 (measure-first) and ADR-0009 in the *opposite* direction, and
   would be the same un-measured leap the "mishap" framing accuses the blocking
   client of.
3. Treating "build Embodiment 1 first" as either a blanket virtue *or* a blanket
   sin. It is neither: it is a **measure-first deferral**, valid exactly as long
   as its gate (§6-Q5) is honored and re-stated. Detached from the gate it
   becomes the proportionality launder; attached to it, it is ADR-0009 discipline.

---

## §7 — The blunt verdict, in the maintainer's own terms

- **What is real:** the wire is exemplary (Axis 1); the blocking `REQ` is the
  Embodiment-1 transport and is structurally incompatible with the Embodiment-2
  multiplexer (Axis 2); the port + client were merged one sequenced step ahead of
  their consumer, with the absence disclosed loudly (Axis 3).
- **The two honest compromises that make the work sound:** (1) the blocking
  client's near-term value is *parity + a working synchronous Shape-B client*,
  not *progress toward Embodiment 2* — bank it as that, not as async-loop
  foundation; (2) its long-run fate (kept for a thread-per-inflight mode vs.
  superseded by a multiplexer) is **the §6-Q5 benchmark's call**, and that
  benchmark is the missing fact, not a missing rewrite.
- **What this consult will *not* certify:** that the blocking client is "the P7
  hack" (Axis 1 refutes it); and equally, that it is unconditionally fine (Axis 2
  shows it cannot be the Embodiment-2 transport, and Axis 3 shows it leads its
  consumer).
- **The honest bottom line:** keep the codec, the port, the decode logic, and the
  parity harness; treat the `REQ` transport as the Embodiment-1 layer behind the
  port; **run the §6-Q5 benchmark before committing to either embodiment**; and
  when the Gumbel consumer lands, decide Embodiment 1-vs-2 on that measurement.
  **Stop there.** Discarding more than the one transport layer, or building the
  multiplexer before the measurement, are the two ways to turn a sound,
  slightly-early piece of work into an actual mistake.

---

## Grounding files (every primary source this memo relies on)

- `cpp/src/zmq_net_client.cpp`, `cpp/include/chocofarm/zmq_net_client.hpp` — the
  blocking `REQ` client (lock-step `:130`/`:145`; `create()` factory; codec calls
  `:129`/`:166`).
- `cpp/include/chocofarm/net_evaluator.hpp` — the port (`predict(span)->expected`,
  `:58`); `cpp/src/net.cpp:203–208` — `NetForward`'s total predict / dead error arm.
- `cpp/include/chocofarm/inference_wire.hpp` — the single codec home
  (`:85/97/127/141`), SSOT-derived (`:3–7`); `cpp/parity/wire_golden.cpp:93–103` —
  re-pointed at it; `tests/test_wire_drift.py` — the drift net.
- `cpp/src/zmq_net_probe.cpp:68` — the *only* constructor of `ZmqNetClient`;
  `tests/test_zmq_net_cpp.py` — the parity round-trip.
- `git show --stat d06db93` — the commit scope (952/74).
- `docs/design/cpp-batched-search.md` — §0 (the async goal), §1.2 (serial-per-tree
  blocked-on leaf), §3.4 (Embodiments 1/2; "cannot multiplex"; "the one structural
  gap"), §6 Q2/Q5 (the deferred multiplexer + the ADR-0009 benchmark gate).
- `docs/design/zmq-inference-service.md` — §1 (zero-cost ACL / two impls), §8
  (workers stay dumb, one blocking predict each), §9 (C++ client sequenced to the
  P9 `cpp/` pass).
- `docs/design/scaling-and-cpp-seam.md` — Shapes A/B/C; the single-asterisk async
  restructure.
- ADR-0009 (earns-its-keep; "state the absence explicitly"), ADR-0011 (Rule 3
  measure-first; Rule 4 nets-over-the-class), ADR-0012 (P7 wire discipline +
  transport fence; anti-pattern E / P5 adopt-or-delete; P9 fallible-port).

## The single load-bearing finding, restated

**P7 governs the wire, not the transport concurrency model — and on the wire the
client is a model citizen that deletes a hand-codec. The blocking `REQ` is the
Embodiment-1 transport of the async loop and is incompatible only with
Embodiment 2; which embodiment to build is an unmeasured, benchmark-gated
question (§6-Q5). The defensible criticism is prematurity, disclosed loudly — not
a P7 hack. Discard nothing but, at most, supersede the one `REQ` transport layer
behind the port once the benchmark is run.**

## Appendix — the commission (the three questions)

1. Investigate `cpp/src/zmq_net_client.cpp` and the commit that led to it in
   light of ADR-0012 and the goal of running an asynchronous work-stealing loop
   (`docs/design/cpp-batched-search.md`).
2. Investigate the sunk cost and the cost of rectifying it.
3. Determine whether all of it needs to be discarded, given the project's stance
   to hacks (the hack-rationalization-detector).

*Public Domain (The Unlicense).*
