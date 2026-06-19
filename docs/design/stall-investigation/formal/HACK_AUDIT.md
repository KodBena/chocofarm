## Hack-rationalization review: formal-stall diagnosis + proposed fix direction

FRAME CHECK: Out-of-frame in the sense that matters — the object of suspicion is
MY OWN modeling process and the proposed fix, treated as suspect (not as context
to agree with). The specific risk audited: did I loosen the formal model until it
produced the SAT the empirical team had already handed me (a confirmation-shaped
model rather than an independent one)?

GENERAL FIX:   "Coalescing degree must not be left to arrival timing: at least one
               actor (server drain OR producer issue) must enforce a minimum
               rows-per-forward before a forward fires." One invariant quantifying
               over BOTH writers of the coalescing degree.
PATCH SHIPPED: No code patch — this is diagnosis. The deliverable is (a) a BMC
               proof the protocol is deadlock-free, (b) a SAT witness that the
               sustained 1-row/forward convoy is an admissible schedule while a
               high-coalescing schedule is admissible under the SAME protocol, and
               (c) a fix DIRECTION (server min-batch/max-delay; producer min-S).
DOWNGRADE:     N/A for a patch. The relevant downgrade risk is in the MODEL: I
               moved from model.py/model2/model3 (deadlock query, UNSAT) through
               convoy/convoy2/convoy3 (convoy query, UNSAT due to lockstep
               over-idealization) to convoy4 (SAT). That chain is a real downgrade
               hazard — each retarget could be "loosen until SAT." Cost that
               justifies it: the FIRST three deadlock models were independently
               UNSAT and that matches the empirical "always eventually completes,"
               so the retarget to a liveness/throughput property was forced by the
               evidence, not chosen to manufacture a hit.
WRITER DELTA:  claimed 2 vs enumerated 2. Writers of rows-per-forward:
               (1) server _drain/_serve_batch timing (inference_server.py:322-387,
                   stage_a_server.py:97-120) — how many queued messages one forward
                   covers; (2) producer issue_one (runner_wire_batched.cpp:551-569,
                   578, 596) — how many rows per wire message. The proposed fix
                   names BOTH and states each independently suffices. No third
                   writer found (the wire codec/transport is degree-neutral, P7).
RUNTIME:       The CONVOY is reproduced two ways: empirically by the maintainer
               (gdb + /proc + server rows/forward = 1.4 vs 55-177), and formally by
               convoy4.py (SAT). The deadlock NON-result is BMC-proven UNSAT to
               depth 8-16 across a T/K/D/max_rows sweep. The proposed FIX is NOT
               runtime-verified (it is diagnosis; do not implement) — so the fix's
               efficacy is unverified-on-paper and must be marked as such.

TELLS (Step 1): grep_tells over the proposed-fix prose: 0 minimality-terms, 2
               named-fix cues, 0 co-occurrence tells. No named-and-bypassed
               signature. (The fix NAMES the better remedy and does not downgrade
               it with a discipline-word.)

VERDICT: general (for the fix DIRECTION) — with a modeling caveat below.
WHY: The fix is stated as one invariant over both actors that set the coalescing
degree, not an enumeration of cases, and the writer count is independently
confirmed at 2-of-2. The diagnosis's central claim (livelock, not deadlock) is
corroborated by an independent BMC UNSAT on the deadlock query, so the SAT convoy
witness is not the only leg it stands on.

FINDINGS BEYOND VERDICT (required):
  - MODEL-FAITHFULNESS RESIDUE (the load-bearing one): convoy4.py reproduces the
    convoy only by ABSTRACTING the staggering into a free "arrive vs forward"
    schedule Z3 chooses. It does NOT mechanistically derive the staggering from
    per-slot search-latency variance (the true physical cause). So convoy4 proves
    "the greedy drain PERMITS a sustained 1-row/forward schedule" — it does NOT
    prove that real ZMQ/OS timing ENTERS that schedule with any particular
    probability, nor that it is metastable/sticky. The empirical evidence
    (bimodal 6s/70s, sticky across runs) supplies the "actually entered + sticky"
    leg; the model supplies only the "admissible" leg. Neither alone is the whole
    claim. This matches the maintainer's own stated gap ("did not single-step the
    entry transition").
  - The earlier convoy models (convoy.py/2/3) returned UNSAT for a WRONG reason
    (they forced all K slots into lockstep, so inflight pinned at 1 and rows/msg
    pinned at K — healthy by construction). An UNSAT from an over-constrained
    model is not evidence of absence. I am flagging this explicitly so the UNSAT
    artifacts are not later miscited as "the convoy is unreachable."
  - NOTHING IN THE PROTOCOL PREVENTS RECURRENCE. Even with the proposed
    server-side min-batch/max-delay, the invariant lives in a TUNING PARAMETER
    (the delay/threshold), not in a structural guarantee. A threshold set too low
    re-opens the convoy; a threshold set too high adds latency. The fix converts a
    metastable failure into a tuning surface — better, but not a closed invariant.
    A genuinely closed fix would make under-coalescing UNREPRESENTABLE (e.g. the
    producer never issues a sub-threshold message while inflight<D), which the
    "producer-side min-S" option approaches but the prose files as merely
    "complementary."
  - D-INDEPENDENCE NUANCE: the convoy needs inflight to STAY full (D outstanding)
    while each carries ~1 row; convoy4 shows it at D=8. The empirical N=4 repro is
    consistent: higher per-thread reply rate (more slots K) makes the 1:1 schedule
    easier to sustain. But my formal models did NOT isolate a clean arithmetic
    threshold in (N, D, max_batch) — I can state the MECHANISM (degree collapses
    when reply turnaround < inter-arrival spacing) but not a proven critical N(D).
    Do not over-read the SAT as "N=4 specifically"; convoy4 is parameterized on D
    and total work, and abstracts K away. The N-specificity is empirical, not
    formally derived.
