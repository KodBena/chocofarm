#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/backpressure.py — Lyapunov drift-plus-penalty / backpressure issue-gate
(online family).

The in-band, per-thread admission control that treats each thread's ready backlog as a virtual queue and
gates it by minimizing the one-step Lyapunov drift-plus-penalty of L = 0.5 * sum_t Q[t]^2, with Q[t] the
thread's ready (parked-at-leaf, unsubmitted) count. This is MaxWeight-style admission (Tassiulas-Ephremides /
Neely "Stochastic Network Optimization"): the gate is driven by the LIVE queue lengths alone, which are a
sufficient statistic, so it inherits throughput-optimality WITHOUT knowledge of the arrival rate.

The per-thread differential traded each forward is

    serve-gain  =  V * mu_hat              (the V-weighted throughput term — letting the batch fly now)
    hold-gain   =  q[t]                    (the queue-drift term — accumulating toward a fatter coalesced
                                            batch lowers 0.5*Q^2 drift more the longer the queue)

and the drift-plus-penalty rule ALLOWS thread t iff serve-gain >= hold-gain (the V-weighted throughput beats
the queue-growth cost of admitting another in-flight unit) and DENIES otherwise (the queue is long enough
that HOLDING lowers the weighted drift — let it drain into a fat batch). q[t] = ready[t] / max(1, norm) is
the (optionally K-normalized) queue length; mu_hat is a slow EWMA of the realized per-served-thread
throughput contribution. V is the stability<->throughput knob: larger V admits more aggressively (chases
throughput at the cost of bigger queues), smaller V stabilizes the queues tightly.

ONLINE LEARNING. The reward fed to observe() is the per-forward throughput contribution (the forward's real
row count, higher-is-better). observe(reward) updates mu_hat — the slow estimate of realized throughput —
which is the V channel of the penalty term; act(obs) reads the current mu_hat. mu_hat is the single
low-dimensional learned scalar (closed-form EWMA, no arms to sweep) so it converges fast over the short run.
There is no held-arm window: backpressure acts on the LIVE queue every forward, and the EWMA's effective
window 1/(1-beta) is the reward-hold W — beta is chosen so the smoothing clears the pipeline lag (a few
forwards) without freezing mu_hat over a run of only hundreds-to-thousands of forwards. The ready backlog is
non-trivial only when issues are held, so the gate's own deny action creates the queue it then manages.

WIRE SUBTLETY (honored by construction). This method uses ONLY the live ready queue (and inflight for the
liveness override). It deliberately sidesteps every sentinel-0 channel: no rtt_us, no server_rows, and NO
first-difference of a cumulative counter (msgs / leaves) — so the "absent thread reads sentinel 0" hazard has
no surface here. The reward is normalized to a per-thread rate by len(obs.served) (this forward's served-tid
cardinality, the un-sentineled count) carried on the info mapping, never by differencing an absent thread.

Liveness override (DENY-ONLY gate semantics: the runner's effective gate is `inflight < D && allow`, and the
forced flush at inflight==0 is UNGATED): inflight[t]==0 -> force allow (a deny is a no-op there, so make the
override explicit rather than implicit).

Online family; cheap, non-throwing act on the per-forward path. Per-run state (mu_hat, the served count, the
denied tally) lives in the instance and is cleared in reset(). O(T) with numpy.

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext


class BackpressureGate:
    """Lyapunov drift-plus-penalty / backpressure admission gate on the per-thread ready queue Q[t]=ready[t].

    Per forward, ALLOW thread t iff the V-weighted throughput term V*mu_hat is at least the queue-drift
    holding term q[t] (=ready[t]/max(1,norm)); DENY otherwise (hold to let the queue drain into a fat batch).
    mu_hat is a slow EWMA of the realized per-served-thread throughput, updated by observe(reward) — the only
    learned state. inflight[t]==0 forces allow (the ungated flush). Online family; cheap, non-throwing,
    O(T)."""

    family: Family = "online"

    def __init__(self, v: float = 4.0, beta: float = 0.2, normalize: bool = True) -> None:
        if v <= 0.0:
            # fail loud (ADR-0002): V is the stability<->throughput knob; a non-positive V would deny every
            # thread with any backlog (serve-gain <= 0 <= q), collapsing the gate — a construction error.
            raise ValueError(f"BackpressureGate: V must be > 0, got {v}")
        if not (0.0 < beta <= 1.0):
            # the EWMA step must be a real convex weight in (0, 1]: beta=0 would freeze mu_hat (never learn);
            # beta>1 is not a smoothing weight. A degenerate step is a construction error, not a run surprise.
            raise ValueError(f"BackpressureGate: beta (EWMA step) must be in (0, 1], got {beta}")
        self.name = f"backpressure_v{v:g}"
        self._v = float(v)
        self._beta = float(beta)
        self._normalize = bool(normalize)
        # per-run state (sized/cleared in reset): the queue normalizer, the learned throughput EWMA, and
        # tallies for the dashboard.
        self._t = 1
        self._norm = 1.0                     # drift normalization (max(1, K) when normalize, else 1)
        self._mu_hat = 0.0                   # slow EWMA of realized per-served-thread throughput (the V chan)
        self._seen = False                   # has mu_hat been seeded by a real reward yet
        self._denied = 0                     # cumulative deny count emitted (metrics)
        self._last_n_allowed = 0             # n_allowed in the most recent act (metrics)
        self._last_mean_q = 0.0             # mean queue length seen in the most recent act (metrics)

    def reset(self, ctx: TrialContext) -> None:
        self._t = int(ctx.n_threads)
        # K is the capacity normalizer the feature wire omits; guard the max(1, .) divisor. With normalize on,
        # q[t] = ready[t]/K is a unit-free backlog fraction so V is commensurate with a per-thread queue;
        # with it off, q[t] = ready[t] is the raw backlog and V carries the units.
        self._norm = float(max(1, int(ctx.k_per_thread))) if self._normalize else 1.0
        self._mu_hat = 0.0
        self._seen = False
        self._denied = 0
        self._last_n_allowed = 0
        self._last_mean_q = 0.0

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        # ONLINE: fold the realized per-forward throughput contribution into the slow EWMA mu_hat — the
        # V-weighted penalty term's estimate of achievable throughput. Normalize the forward's row count to a
        # PER-SERVED-THREAD rate so mu_hat lives in the same units as a single thread's queue length q[t]
        # (V then trades one thread's backlog against one thread's throughput). The served-tid count rides on
        # info (the un-sentineled cardinality); fall back to the raw reward if it is absent so observe stays
        # total (ADR-0002: cheap and non-throwing on the hot path).
        r = float(reward)
        n_served = info.get("n_served", info.get("served_count"))
        try:
            n = int(n_served) if n_served is not None else 0
        except (TypeError, ValueError):
            n = 0
        rate = r / float(n) if n > 0 else r
        if not self._seen:
            self._mu_hat = rate            # seed on the first real reward (no warm-up bias toward 0)
            self._seen = True
        else:
            self._mu_hat += self._beta * (rate - self._mu_hat)   # closed-form EWMA, window ~ 1/(1-beta)

    def act(self, obs: Observation) -> Sequence[int]:
        t = self._t
        feats = obs.features
        # length-T live features; tolerate a short/absent list defensively (cheap, non-throwing per contract).
        ready = _fit(np.asarray(feats.get("ready", ()), dtype=np.float64), t)
        inflight = _fit(np.asarray(feats.get("inflight", ()), dtype=np.float64), t)

        # The virtual queue and the MaxWeight drift-plus-penalty differential, vectorized over threads.
        q = np.maximum(ready, 0.0) / self._norm        # Q[t] >= 0; ready is a count, clamp defensively
        serve_gain = self._v * self._mu_hat            # the V-weighted throughput term (scalar, the learned)
        # ALLOW iff serve-gain dominates the queue-drift holding term; DENY (hold to accumulate) otherwise.
        decision = (serve_gain >= q).astype(np.int64)

        # Liveness override (DENY-ONLY semantics): inflight==0 is an UNGATED forced flush — a deny is a no-op
        # there, so force-allow to keep the override explicit.
        decision[inflight <= 0.0] = 1

        self._last_n_allowed = int(np.count_nonzero(decision == 1))
        self._last_mean_q = float(q.mean()) if t > 0 else 0.0
        self._denied += int(t - self._last_n_allowed)
        return decision.tolist()

    def metrics(self) -> Mapping[str, float]:
        # Expose the learned scalar and the knobs for the dashboard: mu_hat (the learned throughput estimate),
        # V, the EWMA effective window (the reward-hold W), the live mean queue, n_allowed last forward, and
        # the cumulative deny tally.
        window = 1.0 / self._beta if self._beta > 0.0 else float("inf")
        return {
            "mu_hat": self._mu_hat,
            "V": self._v,
            "ewma_window": window,
            "mean_q": self._last_mean_q,
            "n_allowed": float(self._last_n_allowed),
            "denied": float(self._denied),
        }


def _fit(x: np.ndarray, t: int) -> np.ndarray:
    """Coerce a feature array to length T: truncate if long, zero-pad if short. Defensive so act() never
    throws on a malformed/empty feature list (ADR-0002: the per-forward path stays cheap and total)."""
    if x.shape[0] == t:
        return x
    out = np.zeros(t, dtype=np.float64)
    n = min(x.shape[0], t)
    if n:
        out[:n] = x[:n]
    return out


# Register additively (P2 seam: one entry + one class, no edit elsewhere). setdefault so a re-import or a
# harness-side override never clobbers an existing registration.
REGISTRY.setdefault("backpressure", BackpressureGate)
