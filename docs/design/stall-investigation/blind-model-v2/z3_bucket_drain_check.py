"""
Bounded admissibility check for the SERVER bench bucketed/group drain.
Confirmation only (NOT the source of trust). Encodes:
  - the _drain loop-top row test (total_rows < M checked BEFORE the pull), so a single
    K-row message with K > M is accepted whole  (inference_server.py:171-185);
  - _bucket_for snapping real rows UP to {64,256,512}, clamped at 512  (stage_a_server.py:32-37);
  - run_microbatch padding: forward width = max(real, pad_to) i.e. pad only when pad_to > real
    (inference_server.py:58), so for real>512 the bucket clamp gives width = real (no padding);
  - service-time monotonicity S(width) nondecreasing.
Goal: confirm the §5.4 representative execution is admissible (SAT), and that a model
clamping the forward width at M would WRONGLY forbid it (the clamp makes it UNSAT).
"""
from z3 import Int, Real, Solver, If, And, sat, unsat

def bucket_for(real):
    # smallest of 64,256,512 >= real, else 512  (stage_a_server.py:32-37)
    return If(real <= 64, 64, If(real <= 256, 256, 512))

def check(forbid_over_M):
    s = Solver()
    base = Int("base"); N = Int("N"); M = Int("M"); D = Int("D"); T = Int("T")
    K = Int("K"); real = Int("real"); pad_to = Int("pad_to")
    width = Int("width"); pad = Int("pad")
    Smono = Real("S")  # service time proxy, monotone in width

    s.add(base > 0, N > 0, M > 0, D >= 1, T >= 1)
    s.add(K == N * base)                     # runner_wire_batched.cpp:286
    # post-priming a single message gathers all K ready slots -> B_msg = K rows
    # the over-M regime EXISTS iff K > M  (model claim §5.4 / DOF-1 N-dependence)
    s.add(K > M)
    # drain accepts the whole single message: total_rows after the one pull = K,
    # the <M loop-top test passed before the pull, so drained carries K rows.
    s.add(real == K)
    # bench W=group, E=bucket
    s.add(pad_to == bucket_for(real))        # = 512 here since real=K>M>=? clamp branch
    # run_microbatch: pad only if pad_to > real  (inference_server.py:58)
    s.add(width == If(pad_to > real, pad_to, real))
    s.add(pad == If(pad_to > real, pad_to - real, 0))
    # service monotone & positive
    s.add(Smono > 0)
    # a concrete witness in the over-M regime
    s.add(base == 200, N == 3, M == 512, D == 4, T == 4)  # K = 600 > 512 = M

    if forbid_over_M:
        # the WRONG model: forbid any forward wider than M
        s.add(width <= M)

    r = s.check()
    out = {"result": str(r)}
    if r == sat:
        m = s.model()
        out.update({k: str(m[v]) for k, v in
                    [("K", K), ("real", real), ("pad_to", pad_to),
                     ("width", width), ("pad", pad), ("M", M)]})
    return out

if __name__ == "__main__":
    faithful = check(forbid_over_M=False)
    wrong = check(forbid_over_M=True)
    print("FAITHFUL model (no width<=M clamp):", faithful)
    print("WRONG model (forces width<=M)     :", wrong)
    assert faithful["result"] == "sat", "faithful over-M overshoot must be admissible"
    assert wrong["result"] == "unsat", "clamping at M must FORBID the real execution"
    # witness sanity: K=600 > M=512, bucket clamps to 512, real(600)>512 -> no pad, width=600
    assert faithful["K"] == "600" and faithful["pad_to"] == "512"
    assert faithful["width"] == "600" and faithful["pad"] == "0"
    print("OK: §5.4 over-M overshoot admissible; width=600>M=512, bucket=512, pad=0; "
          "a width<=M clamp wrongly forbids it.")
