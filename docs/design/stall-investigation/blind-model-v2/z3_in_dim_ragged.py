"""Minimal bounded confirmation of the load-bearing claim for
Q-exceptional-termination: the L53 ragged-in_dim guard in run_microbatch
(inference_server.py:51-53) is UNSAT under the conforming peer (all messages
carry one in_dim = feat_dim) and SAT under a non-conforming peer.

This confirms only the in_dim-uniformity step of the derivation; it is not the
source of trust. Bounded model: a drain of up to M messages, each with a header
in_dim; the guard fires iff some message's in_dim differs from message-0's.
"""
from z3 import Int, Solver, Distinct, Or, And, sat, unsat

M = 6          # bound: messages in one drain (any small M suffices)
FEAT = 37      # the peer's constant feat_dim (arbitrary positive)

def guard_fires(in_dims):
    # run_microbatch L48: in_dim = mats[0].shape[1]; L51 raises if any differ.
    in0 = in_dims[0]
    return Or([d != in0 for d in in_dims[1:]])

# --- Conforming peer: every message's header in_dim == FEAT (R1). ---
s = Solver()
ds = [Int(f"d{i}") for i in range(M)]
s.add([d == FEAT for d in ds])         # R1: uniform in_dim across the drain
s.add(guard_fires(ds))                 # ask: can the ragged guard fire?
r_conf = s.check()
assert r_conf == unsat, f"expected UNSAT under conforming peer, got {r_conf}"

# --- Non-conforming peer: one message free to carry a different in_dim. ---
s2 = Solver()
es = [Int(f"e{i}") for i in range(M)]
s2.add([e >= 1 for e in es])           # decode_request only requires in_dim>=1
s2.add(es[0] == FEAT)                  # message 0 is a normal one
# at least one rogue message with a self-consistent but different width
s2.add(Or([e != FEAT for e in es[1:]]))
s2.add(guard_fires(es))                # the L53 guard
r_nonconf = s2.check()
assert r_nonconf == sat, f"expected SAT under non-conforming peer, got {r_nonconf}"
m = s2.model()

print("CONFIRMED")
print(f"  conforming peer (R1 uniform in_dim):     ragged-guard reachable? {r_conf}  -> EXCEPTIONAL_TERMINATION NOT reachable")
print(f"  non-conforming peer (one rogue in_dim):  ragged-guard reachable? {r_nonconf}  -> EXCEPTIONAL_TERMINATION reachable")
print(f"  witness in_dims (non-conforming): {[m[e] for e in es]}")
