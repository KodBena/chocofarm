# Confirmation (NOT the source of trust) for the SYNTHESIS composed model.
#
# Goal: exhibit ONE admissible composed execution of the WHOLE boundary that simultaneously
# respects every load-bearing fact the synthesis adopts, so the composed model is shown
# non-vacuous (its representable set is non-empty and contains the canonical trace):
#
#   (A) per-thread in-flight DEPTH == 1  (depth-1 fact; D dead). Each of T threads holds at
#       most one outstanding message; a thread's leaf k+1 send strictly follows its leaf k recv.
#   (B) FIFO-per-pipe: within ONE thread recv order == send order (the audit correction).
#       Reorder is admitted ONLY ACROSS distinct DEALER threads.
#   (C) cross-thread COALESCING: the single-threaded server, busy in a forward over thread A's
#       message, accumulates threads B,C messages that arrive during that forward; the NEXT
#       drain coalesces them into one batch (server self-batching, depth-1 per thread => the
#       server batch = # threads whose message arrived during the prior forward).
#   (D) forward-causality: every reply is sent strictly after the forward that produced it
#       completes; service time S>0 (not an instant); single-thread serialization (no overlap).
#   (E) reply-causal pacing (G4/C2): a thread cannot emit a reply-dependent request before its
#       reply is received and resumed.
#
# If sat, a concrete schedule witnesses the canonical composed regime (cross-thread coalescing
# under depth-1, with intra-thread FIFO and inter-thread reorder). Confirmation only.

from z3 import Solver, Real, Int, And, Or, Implies, sat

s = Solver()

# Three producer threads A,B,C (T=3), each one DEALER, depth 1. Each sends leaf-0 then leaf-1.
# Times are reals on one global clock. The single server runs serial forwards.
TH = ["A", "B", "C"]

# Per-thread, per-leaf: send time and recv time.
send = {(p, k): Real(f"send_{p}_{k}") for p in TH for k in (0, 1)}
recv = {(p, k): Real(f"recv_{p}_{k}") for p in TH for k in (0, 1)}

# Server forward intervals: we model 2 forwards. fwd_start/fwd_end for each.
# Forward 0 serves thread A's leaf-0 alone (A sent first). Forward 1 coalesces B,C leaf-0
# (both arrived while forward 0 ran).
f0s, f0e = Real("f0_start"), Real("f0_end")
f1s, f1e = Real("f1_start"), Real("f1_end")

# all times positive
for t in list(send.values()) + list(recv.values()) + [f0s, f0e, f1s, f1e]:
    s.add(t >= 0)

# Positive service time (not an instant), single-thread serialization (no overlap, f1 after f0).
s.add(f0e > f0s)          # (D) S0>0
s.add(f1e > f1s)          # (D) S1>0
s.add(f1s >= f0e)         # (D) serial: forward 1 cannot start before forward 0 ends

# (C) cross-thread coalescing: A sends first and is drained alone (forward 0);
#     B and C send their leaf-0 DURING forward 0's service window, so they queue and are
#     coalesced into forward 1.
s.add(send[("A", 0)] >= 0)
s.add(f0s >= send[("A", 0)])                      # forward 0 starts after A's request is drained
s.add(And(send[("B", 0)] > f0s, send[("B", 0)] < f0e))   # B arrives mid-forward-0
s.add(And(send[("C", 0)] > f0s, send[("C", 0)] < f0e))   # C arrives mid-forward-0
# A is alone in forward 0; B,C deferred to forward 1 (they arrived after the drain that fed f0).

# (D) forward-causality: replies sent only after the producing forward ends.
s.add(recv[("A", 0)] > f0e)                       # A leaf-0 reply after forward 0
s.add(recv[("B", 0)] > f1e)                       # B leaf-0 reply after forward 1
s.add(recv[("C", 0)] > f1e)                       # C leaf-0 reply after forward 1

# (A)+(E) depth-1 reply-causal pacing: each thread's leaf-1 send strictly follows its leaf-0 recv.
for p in TH:
    s.add(send[(p, 1)] > recv[(p, 0)])            # cannot emit reply-dependent req before reply
    s.add(recv[(p, 1)] > send[(p, 1)])            # a reply follows its request

# (B) FIFO-per-pipe within a thread: send order == recv order (trivially leaf0<leaf1 here);
#     assert no intra-thread reorder: recv leaf-0 precedes recv leaf-1 for each thread.
for p in TH:
    s.add(recv[(p, 0)] < recv[(p, 1)])

# (B) inter-thread REORDER admitted: demonstrate B's leaf-0 reply can precede A's leaf-1 reply
#     even though A's leaf-0 was sent first overall (cross-DEALER reorder is real and allowed).
s.add(recv[("B", 0)] < recv[("A", 1)])

# Also assert A's leaf-0 reply (from forward 0) precedes B/C leaf-0 replies (from forward 1):
s.add(recv[("A", 0)] < recv[("B", 0)])
s.add(recv[("A", 0)] < recv[("C", 0)])

print("checking the canonical composed execution (depth-1 + FIFO-per-pipe + cross-thread coalescing")
print("+ forward-causality + reply-causal pacing) is admissible...")
r = s.check()
print("result:", r)
if r == sat:
    m = s.model()
    def g(x): return m.eval(x)
    print("  forward 0:", g(f0s), "->", g(f0e), " (serves A alone)")
    print("  forward 1:", g(f1s), "->", g(f1e), " (coalesces B,C arrived during forward 0)")
    print("  send A0 =", g(send[("A",0)]), " send B0 =", g(send[("B",0)]), " send C0 =", g(send[("C",0)]))
    print("  recv A0 =", g(recv[("A",0)]), " recv B0 =", g(recv[("B",0)]), " recv C0 =", g(recv[("C",0)]))
    print("  send A1 =", g(send[("A",1)]), " recv A1 =", g(recv[("A",1)]))
    print("  cross-thread reorder witnessed: recv B0 < recv A1 holds.")
    print("ADMISSIBLE: the composed canonical regime is representable (confirmation only).")
else:
    print("UNSAT: composed canonical trace not admissible under these constraints (investigate).")
