<!-- docs/design/issue-controller-policy-v2/03-design-supervised.md — Public Domain (The Unlicense) -->


# Phase 3 — Supervised learning design space

> SUPERVISED LEARNING — collect a labeled dataset by running the wire-ab-bench fixture under an exploratory (random / perturbed) per-thread issue-gate policy, then fit a model f(features)->[0|1]^T that the swappable Python IssueEngine.policy evaluates each control tick. Grounded in: cpp/stage_a/issue_engine.py (the swap seam + on_features data hook), cpp/include/chocofarm/issue_controller.hpp + issue_control_bridge.hpp (the wire/feature contract), cpp/src/runner_wire_batched.cpp run_episodes_wire_pipelined (gate site line 746, publish site 784, measure window 570-574, wire_summary 834-840), cpp/src/wire_ab_bench.cpp (--control-endpoint / --controller-cadence-ms / --sweep-configs / --measure-decisions / --settle-decisions; dps = measure_decisions/measure_wall, RESULT line 437-441), and the existing throwaway harness pattern in cpp/stage_a/stage_b_poolbatch_sweep.py.

## 1. C0 — Constant / threshold baselines (NOT ML, the mandatory control arm)

Two zero-training reference policies the whole family is measured against: all-allow (identity) and a hand-fixed threshold on a single derived feature.

**Mechanism (features → per-thread {0,1}).** all-allow: allow[t]=1 for all t (identity_policy in issue_engine.py:59 — reproduces the fixed-D runner byte-for-byte per controller.hpp:57 / runner:746, so its dps IS the unmodified baseline). Threshold: allow[t] = NOT(deny-condition), e.g. deny (=0) when ready_backlog_norm[t] >= theta AND inflight[t] < d_ceiling (hold a thread that already has fat backlog so its next issue coalesces more); else allow. Pure if/else over the step-2 derived vector, no parameters learned.

**Hyperparameters.** all-allow: none. Threshold: theta (backlog fraction, ~0.3-0.7), optional ready_velocity sign guard.

**Data / training / tuning.** None. all-allow needs zero data; the threshold's theta is set by a tiny manual grid (3-5 values), not fitted.

**Pros.** Zero risk, zero training, fully interpretable; all-allow is the exact A/B control (dps delta against it is the only honest figure of merit); the threshold often captures most of the achievable gain and exposes whether the problem is even learnable.

**Cons.** No adaptation; a single global theta ignores cross-thread heterogeneity; cannot exploit interactions among features. Not a member of the SL family proper — included because per ADR-0009 every learned-model dps claim must be substantiated against it.

**Ergonomics.** Trivial — a few lines in the policy callback; deterministic; no model artifact; instant iteration. This is what every learned model below MUST beat to justify itself.

## 2. C1 — Per-thread logistic / linear gate (PRIMARY recommendation)

One small logistic regression (shared weights across threads) maps each thread's per-thread feature row to P(allow); threshold at 0.5 -> {0,1}.

**Mechanism (features → per-thread {0,1}).** Build a per-thread design row x[t] from the step-2 derived features evaluated for thread t: [ready_backlog_norm[t], inflight_saturation[t], coalesce_degree_inst[t], leaf_rate[t] (or its EWMA), ready_velocity[t], submit_pressure[t]] plus a few pool-context scalars broadcast to every row (cross_thread_spread of leaf_rate, mean_t leaf_rate, realized_dps_window, n_threads, d_ceiling). allow[t] = 1{ sigmoid(w·x[t] + b) >= 0.5 }. ONE weight vector applied independently to all T rows (so it generalizes across T and is permutation-invariant by construction). All inputs are online-computable in issue_engine; the policy already must keep prev-tick state for the rates, so x[t] costs O(T) per tick — negligible at T<=4.

**Hyperparameters.** L2 strength; the 0.5 (or a tuned) decision threshold; the input feature subset; EWMA alpha/tau for any smoothed inputs; class-weight if labels are imbalanced.

**Data / training / tuning.** Collect (x[t], label) pairs by logging every IssueEngine on_features snapshot (issue_engine.py:71,95) under an EXPLORATORY gate (see notes.data_collection) across a config grid (cf:S:D via --sweep-configs, plus --pool-threads/--pool-batch/--trees-per-thread). Label = the per-thread credit-assigned target (see notes.labels; default = label_future_leaf_gain attributed per thread, binarized to 'was-allowing-good-here'). Tens of runs x thousands of ticks x T rows = ample for a linear model. Train with sklearn/JAX in minutes; ship the weight vector as a tiny JSON the policy loads.

**Pros.** Tiny data appetite; permutation-invariant and T-agnostic (one model serves any thread count); interpretable coefficients double as a feature-importance readout to feed back into step-2; near-zero inference latency (matters — the control path latency feeds back into the policy's own prediction quality, per the bridge header). Strong, honest first cut.

**Cons.** Linear decision surface — misses feature interactions (e.g. 'deny only when backlog high AND velocity positive AND server not starved'); needs explicit interaction terms or it underfits; label noise from credit assignment hits it directly; cannot model temporal dynamics (it is memoryless beyond whatever rates/EWMAs you hand it).

**Ergonomics.** Best-in-class: the model is ~10 floats, trains in seconds, is fully inspectable (weight signs tell you which feature drives deny), and drops straight into the policy callback with no inference dependency. Reproducible and cheap to re-fit as features evolve.

## 3. C2 — Gradient-boosted / random-forest tree ensemble (per-thread row classifier)

Same per-thread-row framing as C1 but the mapping is a GBDT/RF that predicts P(allow) (or directly the binary gate) — captures nonlinear feature interactions with no manual cross-terms.

**Mechanism (features → per-thread {0,1}).** Identical design matrix to C1 (one row per (tick,thread), pool-context scalars broadcast in). allow[t] = 1{ ensemble.predict_proba(x[t]) >= tau }. Trees natively handle the 'high backlog AND positive velocity AND headroom under D' conjunctions and the sentinel columns (rtt_us, server_rows_per_forward are constant 0 today — trees simply never split on a constant, so leaving them in is harmless and they light up automatically once those channels are wired).

**Hyperparameters.** n_estimators, max_depth (keep shallow, 3-6, for speed+generalization), learning_rate (GBDT), min_samples_leaf, subsample, and the decision threshold tau (tune tau on a validation split against realized dps, not against label accuracy).

**Data / training / tuning.** Same exploratory-gate collection as C1, but trees want somewhat more data to not overfit the interactions (still easily satisfied by a modest sweep — thousands-to-tens-of-thousands of rows). Train xgboost/lightgbm/sklearn offline; serialize the model. Inference in Python is sub-millisecond for a shallow forest at T<=4.

**Pros.** Captures interactions and nonlinear thresholds C1 cannot; minimal preprocessing; resilient to irrelevant/constant features and to monotone feature scaling; strong tabular accuracy per unit tuning effort; importances guide which derived features actually matter.

**Cons.** Heavier dependency and artifact; can overfit small noisy datasets (the credit-assignment label noise is the real ceiling, not model capacity); per-tick inference latency higher than C1 (still fine at this cadence); loses C1's at-a-glance coefficient interpretability. Memoryless like C1 unless you fold lagged features in.

**Ergonomics.** Very good: minimal feature engineering (no manual interactions, no scaling), built-in feature-importance/SHAP for the step-2 feedback loop, robust to the cumulative-vs-rate mix if you pre-difference. Slightly heavier artifact + a runtime dep (lightgbm) than C1's bare weights.

## 4. C3 — Temporal sequence model: GRU / 1-D temporal CNN over the tick stream

A small recurrent (GRU) or causal 1-D CNN consumes the recent window of feature snapshots and emits T gate logits, so the policy conditions on TRENDS (filling vs draining, RTT rising) rather than a single tick.

**Mechanism (features → per-thread {0,1}).** State the per-tick observation as a [T, F] matrix (F = per-thread derived features) plus a context vector. A SHARED-WEIGHT GRU/TCN runs over the last L ticks; per-thread it can be applied threadwise (one recurrent cell per thread, shared weights) with a small pooled-context broadcast, preserving permutation invariance; output head -> T logits -> sigmoid -> threshold. The model itself replaces the hand-built EWMAs/velocities (it learns the smoothing/derivative), though seeding it with those derived features still helps. Online it needs a rolling hidden state / ring buffer of L ticks kept in the policy object.

**Hyperparameters.** window length L (or GRU hidden size), number of layers, learning rate, dropout, sequence-batch length for BPTT, decision threshold; for the TCN: kernel size + dilation stack (receptive field).

**Data / training / tuning.** Needs CONTIGUOUS per-run tick trajectories (not shuffled rows), each tagged with the per-tick label sequence (label_future_leaf_gain is the natural per-tick supervised target; label_terminal_dps as an auxiliary per-trajectory head). Collect many full runs under varied exploratory gates and configs. Train in JAX/optax (already in the env) with truncated BPTT; this is the most data- and compute-hungry option in the family. Ship a small params blob the policy loads; inference is a single recurrent step per tick.

**Pros.** Only member that natively models temporal structure — the publish cadence is irregular and rates are jittery (publish fires once per reply, not per tick, per runner:784 / step-2 notes), so a learned temporal filter can denoise and anticipate better than fixed EWMAs; can capture lead/lag between ready_velocity and realized leaf_rate; one model subsumes the smoothing hyperparameters.

**Cons.** Data-hungry and the credit-assignment label is the binding constraint — a powerful sequence model on a noisy/observational label mostly fits noise; risk of overfitting trajectory idiosyncrasies; stateful inference adds policy-side complexity and a reset discipline; latency per tick higher (still acceptable at ms cadence). Hard to justify before C1/C2 establish that the signal exists. (An HMM is the lighter sequence alternative — fit a few latent congestion regimes by EM, emit a per-regime gate — but it adds the same trajectory-storage burden for less expressiveness, so it is a fallback, not a first pick.)

**Ergonomics.** Heaviest: data must be stored as ordered trajectories, training has the usual sequence-model knobs, and the policy must carry/advance hidden state every tick and reset it per run. Still pure-Python inference, no C++ recompile. Iteration loop is the slowest of the family.

## 5. C4 — Behavioral cloning of a hindsight/oracle gate (imitation, any backbone)

Construct an offline near-optimal gate label via counterfactual/greedy hindsight, then train ANY of C1-C3's backbones to imitate it — turning the messy credit-assignment problem into clean supervised classification.

**Mechanism (features → per-thread {0,1}).** Same feature->gate mapping as whichever backbone (C1/C2/C3) you clone with; the novelty is the LABEL, not the mechanism. The target gate at each tick is produced offline by either (a) label_counterfactual_gate_advantage — run the DETERMINISTIC fixture twice from the same seed/config differing only in one thread's gate at a tick, label allow=the arm with higher label_terminal_dps (the all-allow arm is a clean control per controller.hpp:57); or (b) a cheap greedy hindsight oracle: label allow=0 where, in the recorded trajectory, holding the issue was followed by a measurably fatter next batch (coalesce_degree_inst jump) without a leaf_rate loss, else allow=1.

**Hyperparameters.** inherited from the chosen backbone, plus the oracle's own knobs (the counterfactual perturbation set / horizon H for the hindsight comparison; the 'measurably fatter' margin).

**Data / training / tuning.** Most EXPENSIVE labeling: (a) needs PAIRED counterfactual runs (2x+ the compute, but the fixture is deterministic so the control arm is exact); (b) is single-trajectory but still full-hindsight. Once labeled, training is ordinary supervised classification on whichever backbone — cheap.

**Pros.** Replaces the weakest link of the whole family — the noisy observational label — with a causally cleaner target (the counterfactual is the gold-standard supervision the online signals only approximate, per step-2 label_counterfactual_gate_advantage); makes even a powerful backbone learn something real; the deterministic fixture makes the control arm exact rather than estimated.

**Cons.** Counterfactual labeling is combinatorially expensive (you cannot afford to perturb every thread at every tick — must sample tick/thread perturbations), so coverage is sparse; the greedy hindsight oracle (b) is cheaper but is itself a heuristic that can bake in a suboptimal policy; this is really a LABELING strategy bolted onto C1-C3, not a standalone model class. Defer until a backbone + the data pipeline exist.

**Ergonomics.** Two-stage: an offline label-generation harness (the costly part — a throwaway script in the stage_b_*_sweep.py mold) then a standard fit. More moving parts than C1/C2, but the second stage is identical to them.

## Recommended first

Try C1 (per-thread logistic / linear gate) FIRST, with C0's all-allow as the mandatory A/B control reported alongside it. Rationale grounded in the fixture: (1) The swap seam already exists with zero friction — IssueEngine takes any f(dict)->[0|1]*T and its on_features hook (issue_engine.py:71,95) is exactly the data-collection tap, and wire_ab_bench.cpp already exposes --control-endpoint/--controller-cadence-ms/--sweep-configs/--measure-decisions, so both collection and deployment need NO C++ recompile. (2) The action space is per-thread-independent binary and T is tiny (<=4 on the host), so a single shared-weight linear model applied per thread-row is permutation-invariant, T-agnostic, and the right capacity for the data we can cheaply gather. (3) Its coefficients are an interpretable feature-importance readout that directly tells the step-2 catalog which derived features carry signal — this de-risks the whole program before spending compute on C2/C3. (4) Inference is ~10 multiply-adds, so it adds negligible latency to a control path whose realtime behavior feeds back into prediction quality (bridge header). Promote to C2 (tree ensemble) the moment C1's residual shows it is leaving interaction-shaped signal on the table (it will, if the gain depends on backlog-AND-velocity-AND-headroom conjunctions); reach for C4's counterfactual labels only once C0/C1 prove the problem is learnable but the observational label is the ceiling; treat C3 (GRU/TCN) as the last resort, justified only if denoising the jittery irregular-cadence tick stream is empirically the bottleneck. Across all candidates the binding constraint is the LABEL, not model capacity — so the first real experiment is C1 trained on label_future_leaf_gain (per-thread credit-assigned, binarized), with the all-allow dps as the honest baseline the dps delta is measured against.

## Notes

SCOPE: this maps the supervised-learning design space only; it does not diagnose the transport, pick a bottleneck, or characterize any failure mode — it assumes the gate is a lever worth learning and lays out how to learn it.

THE OBJECTIVE vs THE LABEL (a fact that shapes every candidate). The printed objective is dps = measure_decisions / measure_wall (wire_ab_bench.cpp:288,430; numerator is RECORDED GUMBEL DECISIONS, captured over the settle-excluded window opened at runner:570 and closed at 572-574). dps is therefore NOT directly leaves/sec. The step-2 leaf_rate / realized_dps_window are the closest ONLINE proxies (a decision consumes many leaf evals, so they are tightly coupled but not identical); the only exact objective is the post-run scalar label_terminal_dps. Consequence: train on a leaf/decision-rate proxy for the per-tick target, but ALWAYS validate by the realized end-to-end dps from the RESULT line, never by label-prediction accuracy.

DATA COLLECTION SCHEME (shared by C1-C4). (1) Exploratory gate: replace identity_policy with a stochastic gate — per thread, allow ~ Bernoulli(p) with p swept across runs (e.g. p in {0.3,0.5,0.7,0.9}), plus epsilon-perturbed versions of C0's threshold and of any current-best policy, to cover the state-action space the deterministic baselines never visit. The forced-flush liveness carve-out (runner:748-749, UNGATED) guarantees even an aggressive random-deny policy still makes progress and never deadlocks, so exploration is safe. (2) Logging: register on_features (issue_engine.py:71) to append each decoded snapshot PLUS the policy's own emitted allow[] vector (the feature surface does NOT echo the gate back — it is policy-side state) PLUS a monotonic timestamp (the frame carries none; the policy must clock itself, per step-2 notes) to a per-run JSONL/npz under ~/w/vdc (never /tmp, per the storage pref). (3) Coverage grid: drive --sweep-configs cf:S:D plus --pool-threads/--pool-batch/--trees-per-thread/--inflight-msgs so the model sees varied K, D, S_min and both the drain-all (chunk_floor off, depth~1, D-dead) and chunked (chunk_floor on, depth>1, D live) regimes — otherwise it overfits one regime. Pin cores 0,1,2,3 and mirror the stage_b sweep's 1:3 server/producer affinity. (4) Determinism: the fixture regenerates bit-identically per config, so paired/counterfactual arms (C4) and clean baselines are exact, not estimated.

LABEL / TARGET CHOICES (the crux). Per-tick supervised targets, in increasing fidelity/cost: (a) label_future_leaf_gain — pool leaves over the next H ticks read forward in the recorded trajectory (step-2 offline label); to get a PER-THREAD, PER-ACTION target, credit-assign it (attribute the pool gain to threads weighted by their leaf-delta, or regress the gain on the action) and binarize to 'allow-was-good-here' for classification — the recommended default. (b) realized_dps_window as a denser reward-correlated regression target. (c) label_terminal_dps as a per-trajectory auxiliary head / global scaler. (d) label_counterfactual_gate_advantage — the causally-clean gold standard (C4), expensive (paired runs). The honest framing: SL here learns to imitate/predict a good gate from hindsight returns; it is one-step / myopic-return supervision, NOT full sequential-credit RL. If the per-tick return label proves too noisy to separate good from bad gates, that is the signal to either (i) switch to C4's counterfactual label, or (ii) hand the problem to the RL family — SL's ceiling IS the label quality.

PREPROCESSING INVARIANTS (apply before any fit). msgs and leaves are CUMULATIVE (controller.hpp:47-48) — first-difference to rates and DROP the raw counters (their absolute magnitude grows with run length and is a spurious feature). rtt_us and server_rows_per_forward are HARD-WIRED 0 today (sentinel: the rtt arg is omitted at the sole publish site runner:784, and no server->producer metrics channel exists, controller.hpp:50) — keep the columns (trees ignore constants; linear weights go to ~0) so the model lights up automatically when those channels are fed, but do NOT let a scaler choke on a zero-variance column. The capacity normalizer K = trees_per_thread*ceil(pool_batch/pool_threads) (runner:451) is NOT in the feature surface — pass it as out-of-band config to the policy (preferred) or use the online running-max lower-bound (step-2 ready_backlog_norm form b). Watch the unit trap: the decoder exposes rtt under key 'rtt_us' as raw MICROSECONDS, not ms (issue_engine.py:49).

VALIDATION / SUBSTANTIATION (ADR-0009 posture). Split by RUN (never by row — rows within a run are autocorrelated, so row-wise CV leaks). The figure of merit is the realized-dps delta vs C0 all-allow on HELD-OUT configs/seeds, read off the RESULT line — not validation-set label accuracy. Report it with the same A/B rigor the existing stage_b harnesses use. A learned model that does not beat all-allow on held-out dps is a negative result to record, not to ship.

WHY THIS FITS THE SEAM. Every candidate lives entirely inside the Python IssueEngine.policy callback (issue_engine.py:63,97) — features in, [0|1]*T out — so the entire family is iterable with NO C++ recompile (only the bits are actuated, controller.hpp:112 / runner:746). The collection harness is a throwaway in the established stage_b_*_sweep.py mold (subprocess-drives chocofarm-wire-ab-bench, parses the RESULT line, writes under ~/w/vdc), consistent with the repo's bench-fixture conventions. NONE of this touches the production default path: no --control-endpoint => controller is nullptr => the wire path is byte-unchanged (wire_ab_bench.cpp:264, controller.hpp:57)."
