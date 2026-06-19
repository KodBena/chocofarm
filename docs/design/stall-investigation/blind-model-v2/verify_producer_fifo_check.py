"""Confirmation for verify-producer-too-permissive.md (NOT the source of trust).

Shows that the producer models' headline 'single-thread out-of-order reply'
(producer_check.py:82,87 ; producer_transport_check.py drain order 1,3,2) is
INADMISSIBLE once the real transport constraint is encoded: a single ZMQ_DEALER
(wire_leaf_pool.hpp:35) talks to one ZMQ_ROUTER (inference_server.py:153); both
pipes are FIFO, and the single-threaded server drains and replies in arrival order
(inference_server.py:173,197-200). Hence a single thread's replies are FIFO.

Part A: the model's claim (free reorder) is SAT  -> the model admits it.
Part B: add the FIFO-per-pipe constraint the code enforces -> UNSAT.
        The gap (SAT in A, UNSAT in B) IS the over-permission.
Public Domain (The Unlicense).
"""
from z3 import Real, Solver, sat

def part(label, fifo):
    s = Solver()
    # one DEALER, two messages c0 then c1 (corr_seq monotone: send0 < send1).
    send0, send1 = Real("send0"), Real("send1")
    # server forward starts and producer recv times for each corr.
    fwd0, fwd1 = Real("fwd0"), Real("fwd1")
    recv0, recv1 = Real("recv0"), Real("recv1")
    s.add(send0 > 0, send1 > send0)                      # monotone send order, one socket
    s.add(fwd0 >= send0, fwd1 >= send1)                  # forward after its send
    s.add(recv0 > fwd0, recv1 > fwd1)                    # reply after its forward
    # the model's representative execution asserts reversed recv order for ONE thread:
    s.add(recv1 < recv0)                                 # producer_check.py:87 (single-thread reorder)
    if fifo:
        # FIFO per pipe: the ROUTER receives this DEALER's msgs in send order, drains
        # and replies in arrival order => recv order == send order for one DEALER.
        s.add(recv0 < recv1)
    return s.check()

ra = part("A model-as-written (free reorder)", fifo=False)
rb = part("B with FIFO-per-pipe (the code's real constraint)", fifo=True)
print("Part A (model admits single-thread reorder):", ra)
print("Part B (code's FIFO-per-pipe forbids it):    ", rb)
print()
if ra == sat and rb != sat:
    print("CONFIRMED over-permission: the producer models ADMIT a single-thread")
    print("out-of-order reply (Part A sat) that the FIFO-per-pipe transport FORBIDS")
    print("(Part B unsat). Reorder is realizable only ACROSS distinct DEALERs (T>=2).")
else:
    print("Inconclusive — re-examine.")
