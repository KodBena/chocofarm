#!/usr/bin/env python3
"""
cpp/stage_a/control_lab/methods/contextual_bandit.py — a homogeneous contextual bandit issue-gate
(online family) candidate for the issue-gate control lab.

Mechanism. A LinUCB / linear-Thompson contextual bandit with weights SHARED across all producer threads
(homogeneous parameter-sharing — the gate has one model, not T). Two arms per thread: deny(0) / allow(1).
Each forward, per thread t, a small context vector phi[t] is read off the feature wire:

    phi[t] = [ submit_pressure     = ready / max(1, D - inflight),     # queued work vs. the headroom to release it
               ready_backlog_norm  = ready / max(1, K),                # K-normalized parked-at-leaf backlog
               inflight_saturation = inflight / max(1, D),             # how close this arm is to the DENY-ONLY no-op
               coalesce_degree_inst= (Δleaves) / max(1, Δmsgs),        # rows-per-message achieved (windowed, served-diff)
               1.0 ]                                                   # bias

with D = ctx.d_ceiling and K = ctx.k_per_thread (the capacity normalizers the feature wire omits). The
context subset is a knob (`context`); the bias is always present, so the realized dimension d is DERIVED
from the chosen subset, never hardcoded.

ONE linear model per arm prices both choices from the SHARED ridge sufficient statistics: arm a carries
A[a] = lambda*I + sum phi phi^T and b[a] = sum r*phi, with theta[a] = A[a]^{-1} b[a]. The per-thread score
is the LinUCB upper confidence bound

    score[t,a] = theta[a]·phi[t]  +  alpha * sqrt( phi[t]^T A[a]^{-1} phi[t] )

and the gate emits the per-thread argmax over {deny, allow}. The exploration bonus is the linear-bandit
posterior-width term (alpha scales it); a tie (cold start, theta==0) breaks toward ALLOW so the cold model
starts at the baseline rather than throttling blind.

Credit assignment under pipeline lag (the reward-hold window W). The reward fed to observe() is the pool's
PER-FORWARD THROUGHPUT CONTRIBUTION (forward_rows — higher is better) and a gate decision shows up in that
number only a few forwards later. So a choice is HELD for W forwards: at a window OPEN the model re-solves
phi + the per-thread argmax and snapshots them; for the next W forwards act() REPLAYS the held action vector
(re-applying the liveness override fresh each forward); observe() accumulates the pool reward across the
window; at the next window open the windowed-MEAN reward is credited to the held (phi[t], action[t]) — one
ridge sample per thread (T samples/window: shared weights means each thread's chosen-arm row is a training
example) — and the closed-form ridge update fires before the new choice is taken. Holding W forwards lets
the in-flight pipeline clear so the reward scored against a choice is actually that choice's reward, not the
prior one's tail. The ridge target is the pool reward CENTERED on a running EMA baseline, so theta captures
each arm's ADVANTAGE (lower-variance, scale-stable exploration) rather than the raw row magnitude.

Non-stationarity. A forgetting factor gamma in (0,1] discounts both arms' statistics at every update
(A <- gamma*A + (1-gamma)*lambda*I; b <- gamma*b) before the new samples are added — the standard
discounted-LinUCB decay — so a regime shift (warm-up -> steady state, depth>1 turning D live) re-weights
toward recent evidence and the (1-gamma)*lambda*I re-injection keeps A conditioned.

Wire subtlety (honored). coalesce_degree_inst first-differences the CUMULATIVE counters msgs and leaves.
lab_server builds each length-T feature list fresh as [0]*T and fills ONLY the served tids, so a thread
ABSENT from a forward reads a sentinel 0, not its true cumulative. The per-thread msgs/leaves baselines are
therefore advanced ONLY for obs.served (against a per-thread baseline, and only once a thread has been seen
before) — an absent thread is never first-differenced (its sentinel 0 would manufacture a spurious negative
delta). An un-baselined or quiet thread reads coalesce_degree_inst = 1.0 (one leaf per message — the neutral
"no coalescing measured yet" value).

Liveness override (DENY-ONLY gate semantics: the runner's effective gate is `inflight < D && allow`, and the
forced flush at inflight==0 is UNGATED): inflight[t]==0 -> force allow, applied FRESH every forward (a deny
is a no-op there, so a thread with nothing in flight is never starved by a held deny).

ONLINE family: reset() clears all learner state (the per-arm A/b, the baselines, the hold window, the reward
EMA); observe() updates the model from the realized reward; act() uses the current learned policy; metrics()
exposes the learned state (per-arm weight norms, exploration scale, reward baseline, update count). The
decision path is O(T*d) numpy + a d*d solve only at window opens (d is tiny: <= 5), non-throwing — it rides
the per-forward critical path. Knobs: ridge lambda, exploration alpha, forgetting gamma, hold window W, and
the context subset.

Run the unit gate pinned + bounded:
    PYTHONPATH=cpp/stage_a /home/bork/w/vdc/venvs/generic/bin/python -m pytest tests/test_method_contextual_bandit.py -q

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from control_lab.adapter import REGISTRY, Family, Observation, TrialContext

# The full context layout (bias last, always present). The `context` knob selects a subset of the
# DISCRETIONARY features by name; the realized dimension d is derived from the selection + the bias, never
# hardcoded (ADR-0002: one source of truth for the layout, the consumers derive their view).
_DISCRETIONARY: tuple[str, ...] = (
    "submit_pressure",
    "ready_backlog_norm",
    "inflight_saturation",
    "coalesce_degree_inst",
)
_N_ARMS = 2  # arm 0 = deny, arm 1 = allow


class ContextualBanditGate:
    """A homogeneous (shared-weights) contextual-bandit issue gate (online). Two arms per thread (deny/allow)
    priced by one LinUCB model per arm over a small per-thread context vector built from the feature wire;
    the gate emits the per-thread argmax UCB. A choice is held W forwards, scored by the windowed-mean pool
    reward (centered on a running baseline), and the chosen arm's closed-form ridge statistics are updated
    (one shared sample per thread). A forgetting factor discounts old evidence for non-stationarity.
    inflight[t]==0 force-allows (a deny is a no-op there). O(T*d) numpy on the per-forward path, non-throwing."""

    family: Family = "online"

    def __init__(
        self,
        ridge_lambda: float = 1.0,
        alpha: float = 1.0,
        gamma: float = 0.999,
        window: int = 8,
        context: Sequence[str] | None = None,
    ) -> None:
        # fail loud (ADR-0002): degenerate hyperparameters are a CONSTRUCTION error, surfaced at build time,
        # never a silent surprise on the per-forward hot path.
        if ridge_lambda <= 0.0:
            raise ValueError(f"ContextualBanditGate: ridge_lambda must be > 0 (A invertible), got {ridge_lambda}")
        if alpha < 0.0:
            raise ValueError(f"ContextualBanditGate: alpha (exploration scale) must be >= 0, got {alpha}")
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"ContextualBanditGate: gamma (forgetting) must be in (0, 1], got {gamma}")
        if window < 1:
            raise ValueError(f"ContextualBanditGate: window (reward-hold) must be >= 1, got {window}")

        # Resolve the context subset against the closed feature vocabulary (ADR-0008: refuse a fuzzy match —
        # an unknown feature name is a construction error, not a silently-dropped column).
        if context is None:
            chosen = list(_DISCRETIONARY)
        else:
            chosen = [str(name) for name in context]
            unknown = [name for name in chosen if name not in _DISCRETIONARY]
            if unknown:
                raise ValueError(
                    f"ContextualBanditGate: unknown context feature(s) {unknown}; valid: {list(_DISCRETIONARY)}"
                )
            if not chosen:
                raise ValueError("ContextualBanditGate: context subset must select at least one feature")
        # the index map (into the full discretionary vector) + the derived dimension (subset + bias).
        self._feat_idx = np.array([_DISCRETIONARY.index(name) for name in chosen], dtype=np.int64)
        self._feat_names = tuple(chosen) + ("bias",)
        self._d = len(chosen) + 1  # +1 bias; DERIVED, never hardcoded

        self._lambda = float(ridge_lambda)
        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self._window = int(window)
        self.name = f"contextual_bandit_a{self._alpha:g}_w{self._window}"

        # --- per-run learner state (sized/cleared in reset) ---
        self._t = 1
        self._d_ceil = 1
        self._k = 1
        # ridge sufficient statistics, one (A, b) per arm; A starts at lambda*I (the prior), b at 0.
        self._A = np.stack([np.eye(self._d) * self._lambda for _ in range(_N_ARMS)])
        self._b = np.zeros((_N_ARMS, self._d), dtype=np.float64)
        # cached per-arm A^{-1} and theta = A^{-1} b, recomputed at each window open.
        self._A_inv = np.stack([np.eye(self._d) / self._lambda for _ in range(_N_ARMS)])
        self._theta = np.zeros((_N_ARMS, self._d), dtype=np.float64)
        # the hold window: snapshot of the choice currently in force.
        self._phi_held: np.ndarray | None = None  # (T, d) context at the window open
        self._act_held: np.ndarray | None = None  # (T,) arms chosen at the window open (0/1)
        self._since_open = 0                       # acts since the current window opened
        self._reward_sum = 0.0                     # pool reward accumulated over the held window
        self._reward_cnt = 0                       # observe() calls accumulated this window
        self._reward_ema = 0.0                     # running baseline (centering target); EMA over windows
        self._ema_seen = False
        # cumulative-counter baselines for the served-thread first-difference (the wire subtlety).
        self._msgs_prev = np.zeros(1, dtype=np.int64)
        self._leaves_prev = np.zeros(1, dtype=np.int64)
        self._seen = np.zeros(1, dtype=bool)
        # dashboard counters.
        self._updates = 0
        self._last_allow_frac = 1.0

    def reset(self, ctx: TrialContext) -> None:
        """Begin a fresh trial: size the per-thread baselines to T, capture D/K, and clear ALL learner state
        (per-arm ridge statistics, the hold window, the reward EMA, the first-difference baselines). D / K
        guard their max(1, .) divisors so a degenerate trial never divides by < 1."""
        self._t = int(ctx.n_threads)
        self._d_ceil = max(1, int(ctx.d_ceiling))
        self._k = max(1, int(ctx.k_per_thread))
        d = self._d
        # reset the shared model to its prior (lambda*I, 0); learning starts from scratch each trial.
        self._A = np.stack([np.eye(d) * self._lambda for _ in range(_N_ARMS)])
        self._b = np.zeros((_N_ARMS, d), dtype=np.float64)
        self._A_inv = np.stack([np.eye(d) / self._lambda for _ in range(_N_ARMS)])
        self._theta = np.zeros((_N_ARMS, d), dtype=np.float64)
        self._phi_held = None
        self._act_held = None
        self._since_open = 0
        self._reward_sum = 0.0
        self._reward_cnt = 0
        self._reward_ema = 0.0
        self._ema_seen = False
        self._msgs_prev = np.zeros(self._t, dtype=np.int64)
        self._leaves_prev = np.zeros(self._t, dtype=np.int64)
        self._seen = np.zeros(self._t, dtype=bool)
        self._updates = 0
        self._last_allow_frac = 1.0

    def observe(self, reward: float, info: Mapping[str, Any]) -> None:
        """Accumulate the realized pool reward (the forward's throughput contribution) into the current hold
        window. The credit fires at the NEXT window open (act), where the windowed mean is attributed to the
        held choice and the ridge model is updated. Before the first choice exists (no held phi yet) a reward
        is ignored — there is nothing to credit it to (the harness may call observe ahead of the first act)."""
        if self._phi_held is None:
            return  # no choice in force yet -> nothing to attribute this reward to.
        r = float(reward)
        if not np.isfinite(r):
            return  # a non-finite reward is dropped rather than poisoning the model (ADR-0002: do not coerce).
        self._reward_sum += r
        self._reward_cnt += 1

    def act(self, obs: Observation) -> Sequence[int]:
        """Advance the first-difference baselines (served threads only), and either OPEN a new window —
        crediting the just-closed window's reward to the held choice, ridge-updating, then re-solving the
        per-thread argmax UCB on fresh context — or REPLAY the held action vector. The inflight==0 liveness
        override is applied FRESH every forward. Cheap (O(T*d) numpy, a d*d solve only at window opens) and
        non-throwing — defaulted reads keep a malformed/short feature frame safe (the watchdog owns loudness
        on the hot path, ADR-0002)."""
        T = self._t
        feats = obs.features
        inflight = _fit(np.asarray(feats.get("inflight", ()), dtype=np.float64), T)
        ready = _fit(np.asarray(feats.get("ready", ()), dtype=np.float64), T)
        msgs = _fit(np.asarray(feats.get("msgs", ()), dtype=np.float64), T).astype(np.int64)
        leaves = _fit(np.asarray(feats.get("leaves", ()), dtype=np.float64), T).astype(np.int64)
        served = [i for i in obs.served if 0 <= i < T]

        # --- windowed coalescence (served-thread first-difference of the CUMULATIVE counters) ---
        # Only served & previously-seen threads carry a real delta; everyone else gets the neutral 1.0
        # (one leaf per message). Absent threads are NEVER first-differenced (their sentinel 0 would fake a
        # negative delta), and their baselines are NOT touched (they offered nothing this forward).
        coalesce = np.ones(T, dtype=np.float64)
        for i in served:
            if self._seen[i]:
                d_msgs = int(msgs[i] - self._msgs_prev[i])
                d_leaves = int(leaves[i] - self._leaves_prev[i])
                if d_msgs > 0 and d_leaves >= 0:
                    coalesce[i] = d_leaves / float(d_msgs)
            self._msgs_prev[i] = msgs[i]
            self._leaves_prev[i] = leaves[i]
            self._seen[i] = True

        # --- build the per-thread context phi (T, d): the discretionary features (subset) + bias ---
        phi = self._build_phi(inflight, ready, coalesce)

        # --- decide: open a new window (credit + update + re-solve) or replay the held choice ---
        if self._phi_held is None or self._since_open >= self._window:
            self._close_window_and_update()   # credit the just-closed window to the held choice (if any)
            self._refresh_model()             # recompute per-arm A^{-1} and theta from the updated statistics
            action = self._solve_argmax(phi)  # the new held choice (per-thread UCB argmax)
            self._phi_held = phi
            self._act_held = action
            self._since_open = 0
        else:
            assert self._act_held is not None
            action = self._act_held           # replay the held choice (the per-forward critical path)
        self._since_open += 1

        # --- liveness override: inflight==0 is an UNGATED forced flush -> a deny is a no-op, force allow.
        # Applied FRESH every forward (inflight moves within a held window), without mutating the held choice.
        decision = action.copy()
        decision[inflight <= 0.0] = 1

        self._last_allow_frac = float(np.count_nonzero(decision == 1)) / float(T) if T else 1.0
        return decision.astype(np.int64).tolist()

    def metrics(self) -> Mapping[str, float]:
        """Dashboard scalars exposing the LEARNED state: per-arm weight (theta) L2 norms, the exploration
        scale alpha, the running reward baseline, the ridge-update count, and the last forward's allow
        fraction. Empty-safe."""
        return {
            "w_norm_deny": float(np.linalg.norm(self._theta[0])),
            "w_norm_allow": float(np.linalg.norm(self._theta[1])),
            "exploration_alpha": self._alpha,
            "reward_baseline": float(self._reward_ema),
            "updates": float(self._updates),
            "allow_frac": float(self._last_allow_frac),
        }

    # ---------------------------------------------------------------- internals

    def _build_phi(self, inflight: np.ndarray, ready: np.ndarray, coalesce: np.ndarray) -> np.ndarray:
        """Assemble the (T, d) context: the full discretionary vector subset-selected by the `context` knob,
        plus a bias column of ones. All divisors are max(1, .)-guarded so phi is always finite."""
        D = float(self._d_ceil)
        headroom = np.maximum(1.0, D - inflight)            # the room to release (>= 1)
        submit_pressure = ready / headroom                  # queued work vs. headroom to release it
        ready_backlog_norm = ready / float(self._k)         # K-normalized parked backlog
        inflight_saturation = inflight / D                  # proximity to the DENY-ONLY no-op (j -> D)
        full = np.stack(                                    # (4, T) in the canonical _DISCRETIONARY order
            [submit_pressure, ready_backlog_norm, inflight_saturation, coalesce], axis=0
        )
        selected = full[self._feat_idx]                     # (len(subset), T)
        bias = np.ones((1, selected.shape[1]), dtype=np.float64)
        return np.concatenate([selected, bias], axis=0).T   # (T, d)

    def _solve_argmax(self, phi: np.ndarray) -> np.ndarray:
        """Per-thread LinUCB argmax over {deny, allow}: score[t,a] = theta[a]·phi[t] + alpha*sqrt(phi[t]^T
        A_inv[a] phi[t]). Ties (cold start, theta==0 with equal bonuses) break toward ALLOW (arm 1) so the
        cold model reproduces the all-allow baseline rather than throttling blind. Vectorized over threads."""
        # mean reward per (thread, arm): (T, d) @ (d, n_arms) -> (T, n_arms)
        mean = phi @ self._theta.T
        # exploration width per (thread, arm): sqrt( phi A_inv phi ) for each arm, clipped at 0 for numerics.
        # einsum gives the quadratic form per thread per arm without materializing T*n_arms d-vectors.
        quad = np.einsum("td,akd,tk->ta", phi, self._A_inv, phi, optimize=True)
        width = np.sqrt(np.clip(quad, 0.0, None))
        score = mean + self._alpha * width                  # (T, n_arms) UCB
        # argmax with allow-favoring tie-break: prefer arm 1 when score[:,1] >= score[:,0].
        return (score[:, 1] >= score[:, 0]).astype(np.int64)

    def _close_window_and_update(self) -> None:
        """Credit the just-closed hold window's reward to the held (phi, action) and ridge-update the chosen
        arm's statistics (one shared sample per thread). The target is the windowed-mean pool reward CENTERED
        on a running EMA baseline (so theta is each arm's advantage). A discount (gamma) decays both arms
        before the new samples land (non-stationarity). No-op if there is no held choice or no reward yet."""
        if self._phi_held is None or self._act_held is None or self._reward_cnt == 0:
            # nothing held, or no reward observed for this window -> reset the accumulator and bail (no update).
            self._reward_sum = 0.0
            self._reward_cnt = 0
            return

        mean_reward = self._reward_sum / float(self._reward_cnt)
        # update the running baseline (EMA over windows) and center the target on the PRE-update baseline so
        # the very first window has a defined (zero-centered) advantage rather than self-cancelling.
        baseline = self._reward_ema if self._ema_seen else mean_reward
        target = mean_reward - baseline
        beta = 0.1  # EMA horizon over windows (10-window memory) — short run, so track quickly.
        self._reward_ema = mean_reward if not self._ema_seen else (1.0 - beta) * self._reward_ema + beta * mean_reward
        self._ema_seen = True

        # discount BOTH arms toward the prior (discounted-LinUCB decay), keeping A conditioned via the
        # (1-gamma)*lambda*I re-injection. gamma==1 -> vanilla LinUCB (no forgetting).
        if self._gamma < 1.0:
            d = self._d
            eye = np.eye(d)
            for a in range(_N_ARMS):
                self._A[a] = self._gamma * self._A[a] + (1.0 - self._gamma) * self._lambda * eye
                self._b[a] = self._gamma * self._b[a]

        # add the held samples: each thread contributes one (phi[t], chosen arm, target) to the SHARED model.
        phi = self._phi_held
        act = self._act_held
        for a in range(_N_ARMS):
            sel = act == a
            if not np.any(sel):
                continue
            phi_a = phi[sel]                          # (n_a, d) the threads that chose arm a
            self._A[a] += phi_a.T @ phi_a            # sum phi phi^T (the ridge Gram contribution)
            self._b[a] += target * phi_a.sum(axis=0)  # sum r*phi (the ridge moment contribution)

        self._updates += 1
        self._reward_sum = 0.0
        self._reward_cnt = 0

    def _refresh_model(self) -> None:
        """Recompute each arm's A^{-1} and theta = A^{-1} b from the current statistics. Called only at window
        opens (d <= 5, so a couple of tiny solves is negligible vs. the per-forward path). A is SPD by
        construction (lambda*I floor + PSD Gram terms), so np.linalg.solve is well-posed; on the off chance of
        a numerical breakdown fall back to the pseudo-inverse rather than throwing on the control path."""
        d = self._d
        eye = np.eye(d)
        for a in range(_N_ARMS):
            try:
                self._A_inv[a] = np.linalg.solve(self._A[a], eye)
            except np.linalg.LinAlgError:
                self._A_inv[a] = np.linalg.pinv(self._A[a])  # defensive: never throw on the control path.
            self._theta[a] = self._A_inv[a] @ self._b[a]


def _fit(x: np.ndarray, t: int) -> np.ndarray:
    """Coerce a feature array to length T: truncate if long, zero-pad if short. Defensive so act() never
    throws on a malformed/empty feature list (ADR-0002: the per-forward path stays cheap and total; a
    zero-padded slot lands in the un-baselined / inflight==0 liveness path, i.e. neutral + force-allow)."""
    if x.shape[0] == t:
        return x
    out = np.zeros(t, dtype=np.float64)
    n = min(x.shape[0], t)
    if n:
        out[:n] = x[:n]
    return out


# Register additively into the FROZEN adapter.REGISTRY (one entry + one class — P2 seam discipline; the
# harness + dashboard discover methods here). setdefault so a re-import or a name clash never silently
# clobbers an existing registration.
REGISTRY.setdefault("contextual_bandit", ContextualBanditGate)
