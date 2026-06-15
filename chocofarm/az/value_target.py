#!/usr/bin/env python3
"""
chocofarm AZ — the AZ training-target rules: the VALUE-target (suffix MC return-to-go + the Part B
lower-variance blend) AND the POLICY-target (the Danihelka improved policy + v_mix completion).

Public Domain (The Unlicense).

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

POLICY-target rule (the Danihelka improved policy, audit item C)
----------------------------------------------------------------
The other half of the AZ target — the improved-policy target π′ the apprentice regresses onto —
also lives here, as PURE functions of explicit inputs (`v_mix`, `improved_policy`). They were
previously welded into `gumbel_search.py` as `_v_mix`/`_improved_policy`, side-reading the live
tree's node statistics so they could not be called or unit-tested outside a search. The math is
Danihelka et al. 2022 §3, UNCHANGED — only the inputs are now explicit:

  * `v_mix(root_value, visited_q, visited_n, prior, legal_slots)`  — §3 value completion for
    unvisited actions: the net leaf value blended with the PRIOR-weighted (not visit-weighted) mean
    of the visited actions' Q. The prior-weighting matters precisely because Sequential Halving
    makes the visit counts deliberately unequal (`tests/test_az_loop.py::test_vmix_prior_weighted`).

  * `improved_policy(logits, visited_q, visited_n, root_value, prior, legal_slots, c_visit, c_scale)`
    — π′ = softmax(logit + σ(completedQ)) over the legal slots, where σ(q) = (c_visit + max_a N(a))
    · c_scale · q is the paper's monotone transform and unvisited actions have their Q completed by
    `v_mix`. Returns an (n_slots,) probability row (exactly zero on illegal slots).

`GumbelAZSearch` collects the per-root-action (Q, N) etc. from the live node and CALLS these; its
`_v_mix`/`_improved_policy`/`_sigma_scale` are now thin wrappers that gather the node stats and
delegate. The math, the call order, and the byte-identical outputs are unchanged.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt

# The policy-target rule's final step is the masked softmax over the legal slots. It has ONE home —
# `ValueMLP._masked_softmax`, a pure numpy staticmethod with no net state (the SAME function the
# welded rule called as `net._masked_softmax`). We bind it here rather than re-implement it, so the
# extraction does not spawn a fourth copy of the masked-softmax math. `mlp` is a light numpy module
# (no JAX) and does not import this module, so there is no import cycle and no heavy dependency.
from chocofarm.az.mlp import ValueMLP as _ValueMLP

_masked_softmax = _ValueMLP._masked_softmax


def suffix_returns_to_go(step_rt: Sequence[tuple[float, float]], exit_c: float,
                         lam: float) -> list[float]:
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


def blended_returns_to_go(step_rt: Sequence[tuple[float, float]], boot: Sequence[float],
                          exit_c: float, lam: float, lam_blend: float = 1.0,
                          n_step: int | None = None) -> list[float]:
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


# ========================================================================================
# POLICY-target rule (Danihelka et al. 2022 §3 improved policy + v_mix) — audit item C
#
# Pure functions of explicit per-root-action inputs. `GumbelAZSearch` gathers the live node's
# statistics (root.value / root.prior / root.q / root.N over the legal slots) and CALLS these;
# the math is byte-for-byte the rule that used to live in `gumbel_search._v_mix`/`_improved_policy`.
#
# Per-slot input convention: `prior` is the (n_slots,) net-prior array (= root.prior, float32 by
# default); `visited_q`/`visited_n` are slot-indexed SEQUENCES (length ≥ max legal slot) and
# `legal_slots` is the list of legal slot indices. For a legal slot `s`:
#   * prior[s]      = the net's masked-softmax prior P(s,·) (= root.prior[s]),
#   * visited_n[s]  = N(s), the root selection count of slot s (0 if unvisited),
#   * visited_q[s]  = Q(s), the running λ-penalized return mean of slot s (read only when N(s)>0).
# This mirrors exactly what the search side-read from the node (root.prior[s], root.N[a], root.q(a)
# with a = slot_to_action(s)), now passed in explicitly so the rule is unit-testable without a tree.
#
# BYTE-IDENTITY NOTE: the caller passes `visited_q`/`visited_n` as PLAIN PYTHON float/int sequences
# (not numpy arrays). The welded rule multiplied the float32 `prior[s]` by `root.q(a)` (a Python
# float, weak under numpy promotion) → float32, while `σ·Q` in the completion used Q's full float64
# magnitude. Python scalars reproduce both; a numpy array would upcast (float64) or truncate
# (float32) the prior-weighted product and break byte-identity. `prior`, `logits` stay numpy.
# ========================================================================================

def sigma_scale(visited_n: Sequence[int], legal_slots: Sequence[int],
                c_visit: float, c_scale: float) -> float:
    """The Danihelka §3 monotone Q-transform scale σ-prefactor: (c_visit + max_a N(a)) · c_scale.

    `max_a N(a)` is taken over the root's selection counts; matches `GumbelAZSearch._sigma_scale`
    (= (c_visit + max(root.N.values())) * c_scale, with max 0 when no action was visited)."""
    max_n = max((visited_n[s] for s in legal_slots), default=0)
    return (c_visit + max_n) * c_scale


def v_mix(root_value: float, visited_q: Sequence[float], visited_n: Sequence[int],
          prior: npt.NDArray[np.floating], legal_slots: Sequence[int]) -> float:
    """Danihelka §3 value-completion for unvisited actions:

        v_mix = (v_net + ΣN · v̄) / (1 + ΣN),
        v̄ = Σ_{b:N(b)>0} π(b)·Q(b) / Σ_{b:N(b)>0} π(b)   (PRIOR-weighted, not visit-weighted)

    `root_value` = v_net (the net leaf value at the root); ΣN = Σ_s visited_n[s] over legal slots.
    Returns `root_value` unchanged when no action was visited (ΣN == 0) or all visited priors are 0
    — the degenerate fallback the search uses. Byte-identical to the former `_v_mix`.

    `sum_n` is forced to a plain Python `int` (as `sum(root.N.values())` was): a numpy int here would
    upcast the float32 `v_bar` to float64 in `sum_n * v_bar`, breaking the float32 promotion of the
    welded rule. The weak Python int keeps `sum_n * v_bar` at `v_bar`'s dtype — the original."""
    sum_n = int(sum(visited_n[s] for s in legal_slots))
    pw_num = pw_den = 0.0
    for s in legal_slots:
        if visited_n[s] > 0:
            pw_num += prior[s] * visited_q[s]
            pw_den += prior[s]
    if sum_n > 0 and pw_den > 0:
        v_bar = pw_num / pw_den
        return (root_value + sum_n * v_bar) / (1 + sum_n)
    return root_value


def improved_policy(logits: npt.NDArray[np.floating], visited_q: Sequence[float],
                    visited_n: Sequence[int], root_value: float,
                    prior: npt.NDArray[np.floating], legal_slots: Sequence[int],
                    c_visit: float, c_scale: float) -> npt.NDArray[np.float64]:
    """π′ = softmax(logit + σ(completedQ)) over the legal slots (Danihelka et al. 2022 §3).

    Unvisited legal actions have their Q "completed" by `v_mix` (the prior-weighted value estimate);
    σ(q) = (c_visit + max_a N(a)) · c_scale · q is the paper's monotone transform. `logits`/`prior`
    are (n_slots,) slot-indexed arrays (the root logit log P(s,·) and the prior). Returns an
    (n_slots,) probability row, exactly zero on illegal slots. Byte-identical to the former
    `_improved_policy` (same operation order; the masked softmax IS `mlp.ValueMLP._masked_softmax`,
    the same staticmethod the welded rule called — bound at module load, not re-implemented)."""
    n_slots = len(logits)
    sigma = sigma_scale(visited_n, legal_slots, c_visit, c_scale)
    vm = v_mix(root_value, visited_q, visited_n, prior, legal_slots)

    completed = np.full(n_slots, -1e30)
    for s in legal_slots:
        q = visited_q[s] if visited_n[s] > 0 else vm
        completed[s] = logits[s] + sigma * q
    # softmax over the legal slots — the SAME masked-softmax the welded rule called
    # (`net._masked_softmax`). `_masked_softmax` is a pure numpy staticmethod with no net state, so
    # we call it directly: ONE home for the masked-softmax (no copy), byte-identical by construction.
    legal_arr = np.zeros(n_slots)
    legal_arr[legal_slots] = 1.0
    # _masked_softmax returns a (1, n_slots) float64 row-batch; [0] selects the single row. numpy's
    # __getitem__ is Any-typed, so the cast states the (n_slots,) float64 row contract (no runtime change).
    return cast("npt.NDArray[np.float64]",
                _masked_softmax(completed[None, :], legal_arr[None, :])[0])
