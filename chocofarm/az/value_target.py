#!/usr/bin/env python3
"""
chocofarm AZ — the value-target rule (suffix MC return-to-go + the Part B lower-variance blend).

ONE place for the λ-penalized return-to-go computation. It used to be duplicated across
`exit_loop.generate_episode` and `dataset.py:_episode_transitions` (the az-exit-loop §(f) audit
flagged this as two writers that would silently diverge the moment a TD(λ) blend was added). Part B
adds exactly that blend, so the suffix rule is extracted here first (per the audit's own
prescription) and both callers route through it.

The episode is a sequence of DECISIONS j = 0..D-1, each executing a non-TERMINATE action with
realized (r_j, dt_j); the episode ends with a single exit toll `exit_c` paid at the final location.

Pure Monte-Carlo target (the prior behavior, `lam_blend == 1.0` / `n_step is None`)
-----------------------------------------------------------------------------------
    G_j^MC = Σ_{t≥j} r_t  −  λ·( Σ_{t≥j} dt_t  +  exit_c )

i.e. the actual banked value from decision j onward minus λ times the actual remaining travel plus
the one end-of-episode exit toll. Built by suffix accumulation; the exit toll is charged once, in
every suffix. This is the HONEST realized return-to-go — high variance (one rollout), which (per the
feature-response finding) let the value collapse to a progress counter rather than fitting the
geometry/belief structure. `suffix_returns_to_go` computes exactly this.

Lower-variance blended target (Part B)
--------------------------------------
The search already produces, at every decision, a ~n_sims-averaged estimate of the CURRENT belief's
λ-penalized value (`GumbelAZSearch._root_search_value` — the visit-weighted mean of the root
actions' simulated returns). Call it `boot[j]` (the value of the belief decided-from at step j).
Bootstrapping off it trades the single-rollout MC variance for a much lower-variance n-sample
average, at no extra rollout cost. Two parametrizations of the same idea:

  * n-STEP (`n_step = n`): accumulate the realized λ-penalized reward for `n` steps, then bootstrap:
        G_j^(n) = Σ_{t=j}^{j+n-1} (r_t − λ·dt_t)  +  boot[j+n]
    truncating to the MC tail (incl. the exit toll) when j+n reaches the episode end. A finite
    large n (≥ episode length) is MATHEMATICALLY the MC target but accumulates term-by-term, so it
    is FP-close (~1 ULP), not bit-identical, to the suffix rule. `n_step = None` is the BIT-IDENTICAL
    pure-MC path (it dispatches to `suffix_returns_to_go` verbatim).

  * TD(λ) (`lam_blend = ℓ`, the forward view): the geometric average of all n-step returns
        G_j^λ = (1−ℓ) Σ_{n≥1} ℓ^{n-1} G_j^(n)  +  ℓ^{(D−j)} G_j^MC
    ℓ = 1 ⇒ pure MC (current behavior); ℓ → 0 ⇒ pure 1-step bootstrap. Computed by the standard
    backward recurrence so it is O(D) per episode, not O(D²):
        G_j^λ = (r_j − λ·dt_j) + ℓ·G_{j+1}^λ + (1−ℓ)·boot[j+1]
    with the boundary `boot[D] = −λ·exit_c` (the only continuation past the last executed decision
    is to exit — the continuation value at the terminal state IS the exit toll). This recurrence is
    algebraically the forward-view TD(λ) above (verified in `tests/test_az_loop.py`).

HONEST RISK (Part B, watched + reported): the bootstrap `boot[j]` is the SEARCH's value estimate,
which carries some of the optimism the pure-MC target (design F4) was chosen to avoid — the search
can over-value under-sampled deep-sensing beliefs. The blend keeps it TUNABLE (ℓ=1 / n=∞ recovers
the un-optimistic pure-MC target exactly), and the loop reports E[T] (the over-collection signature)
so an optimism regression is observable. See docs/results/az-parallel-exp.md §Part B.

Both knobs are mutually exclusive at the loop CLI; this module accepts whichever is set. When both
defaults hold (`n_step is None`, `lam_blend == 1.0`) the output is bit-identical to the prior
suffix-only rule (asserted by tests) — Part B is opt-in.
"""
from __future__ import annotations


def suffix_returns_to_go(step_rt, exit_c, lam):
    """Pure-MC λ-penalized return-to-go per decision (the prior behavior; the `lam_blend == 1`
    limit). `step_rt`: list of (r_j, dt_j) for each EXECUTED decision, in order. `exit_c`: the
    single end-of-episode exit toll. Returns a list `g[0..D-1]` aligned to `step_rt`.

        g_j = Σ_{t≥j} r_t − λ·(Σ_{t≥j} dt_t + exit_c)

    Suffix accumulation; the exit toll is in every suffix (charged once at episode end)."""
    out = [0.0] * len(step_rt)
    suffix_r = suffix_t = 0.0
    for j in range(len(step_rt) - 1, -1, -1):
        r_j, dt_j = step_rt[j]
        suffix_r += r_j
        suffix_t += dt_j
        out[j] = suffix_r - lam * (suffix_t + exit_c)
    return out


def blended_returns_to_go(step_rt, boot, exit_c, lam, lam_blend=1.0, n_step=None):
    """The Part B lower-variance value target, with the pure-MC suffix rule as the `lam_blend==1`
    / `n_step is None` limit.

    `step_rt`: list of (r_j, dt_j) for each executed decision (length D). `boot`: list of the
    search root-value bootstraps `boot[j]` for j=0..D-1 (the ~n_sims-averaged λ-penalized value of
    the belief decided-from at step j); MUST be the same length D as `step_rt`. `exit_c`/`lam` as
    above.

    Exactly one of `lam_blend` / `n_step` parametrizes the blend:
      * `n_step = n` (int ≥ 1): the n-step bootstrap target (see module docstring). `n = None`
        means n = ∞ = pure MC.
      * `lam_blend = ℓ` (0 ≤ ℓ ≤ 1): forward-view TD(λ). ℓ = 1 = pure MC.
    If `n_step` is given it takes precedence; otherwise `lam_blend` is used. Returns `g[0..D-1]`.

    The continuation past the last decision is the exit toll: `boot[D] := −λ·exit_c`. (At the
    terminal state the only action is to exit, whose value is exactly the exit toll — the same
    quantity the MC tail charges, so the n=∞/ℓ=1 limit collapses to `suffix_returns_to_go`.)"""
    D = len(step_rt)
    if D == 0:
        return []
    assert len(boot) == D, (len(boot), D)
    # the continuation value at the terminal state IS the exit toll (one action: exit)
    boot_term = -lam * exit_c

    # --- pure MC fast paths (bit-identical to suffix_returns_to_go) ---
    if n_step is None and lam_blend >= 1.0:
        return suffix_returns_to_go(step_rt, exit_c, lam)

    # --- n-step ---
    if n_step is not None:
        n = int(n_step)
        if n < 1:
            raise ValueError(f"n_step must be ≥ 1 (got {n})")
        out = [0.0] * D
        for j in range(D):
            acc = 0.0
            end = j + n
            if end >= D:
                # n-step horizon reaches the episode end: realized reward to the end + exit toll
                # (this IS the MC tail from j, with no bootstrap) → matches suffix rule for this j
                for t in range(j, D):
                    r_t, dt_t = step_rt[t]
                    acc += r_t - lam * dt_t
                acc += -lam * exit_c
            else:
                for t in range(j, end):
                    r_t, dt_t = step_rt[t]
                    acc += r_t - lam * dt_t
                acc += boot[end]   # bootstrap off the search value at the state reached after n steps
            out[j] = acc
        return out

    # --- TD(λ) forward view via the O(D) backward recurrence ---
    ell = float(lam_blend)
    if not (0.0 <= ell <= 1.0):
        raise ValueError(f"lam_blend must be in [0, 1] (got {ell})")
    out = [0.0] * D
    g_next = boot_term            # G_{D}^λ boundary = continuation past the last decision = exit toll
    boot_next = boot_term         # boot[D] = exit toll
    for j in range(D - 1, -1, -1):
        r_j, dt_j = step_rt[j]
        step = r_j - lam * dt_j
        # G_j^λ = step + ℓ·G_{j+1}^λ + (1−ℓ)·boot[j+1]
        g_j = step + ell * g_next + (1.0 - ell) * boot_next
        out[j] = g_j
        g_next = g_j
        boot_next = boot[j]
    return out
