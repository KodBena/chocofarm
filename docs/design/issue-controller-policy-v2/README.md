<!-- docs/design/issue-controller-policy-v2/README.md — Public Domain (The Unlicense) -->


# Issue-controller policy design — exploration (v2, task-only brief)

A multi-agent design exploration for the **online per-thread issue-gate controller** in the leaf-eval transport: given the control interface (per-thread binary gate, the feature surface, the throughput objective, the deny-only / ungated-forced-flush constraints), map the controller design space and propose an ergonomic, ADR-0012-compliant way to test candidates.

## Provenance

- Run: `wf_97dabc10-d29`; 6 agents; 4 phases (atomic features → derived features → design space [static/supervised/online] → testing shape).
- **Brief discipline:** task-only. Each agent got the control interface and its assigned design step — *no* fed conclusions, *no* requested diagnosis of bottleneck/failure-mode. The exact shared brief is reproduced verbatim in [00-brief-verbatim.md](00-brief-verbatim.md) for audit.
- **Supersedes** the sibling `../issue-controller-policy/` (v1), whose brief stated a conclusion ("the win is variance/collapse-tail, not mean") as fact and so biased its three converging agents. v1 is kept un-edited as a point-in-time record (ADR-0005); this v2 is the trustworthy run.

## Phases

1. [Atomic feature surface](01-atomic-features.md) — the 8 directly-measured signals.
2. [Derived features](02-derived-features.md) — 19 statically/offline-derived features.
3. Design space — [static](03-design-static.md) · [supervised](03-design-supervised.md) · [online](03-design-online.md).
4. [Testing shape](04-testing-shape.md) — the ADR-0012-compliant harness.

## Headline recommendations (verbatim `recommended_first` / shape)

- **Static / engineering-informed:** C1 (per-thread coalescing-floor hysteresis gate). Try it first because it is the minimal static controller that (a) directly actuates the one real lever — deny while a thread's un-submitted backlog is below a target so the next issue coalesces more rows — using ONLY live, non-sentinel features (ready, inflight, plus an optional differenced ready_velocity); (b) has exactly one primary knob with an obvious physical meaning (the per-thread ready-backlog target B*, expressed as a fraction of K or, where K is unavailable in-band, of a running-max estimate), tunable by a 1-D sweep on the bench's own…
- **Supervised learning:** Try C1 (per-thread logistic / linear gate) FIRST, with C0's all-allow as the mandatory A/B control reported alongside it. Rationale grounded in the fixture: (1) The swap seam already exists with zero friction — IssueEngine takes any f(dict)->[0|1]*T and its on_features hook (issue_engine.py:71,95) is exactly the data-collection tap, and wire_ab_bench.cpp already exposes --control-endpoint/--controller-cadence-ms/--sweep-configs/--measure-decisions, so both collection and deployment need NO C++ recompile. (2) The action space is per-thread-independent binary and T is tiny (<=4 on the host), so …
- **Online learning:** C1 — the homogeneous threshold bandit. It is the unique design that satisfies all of the family's binding constraints simultaneously: (a) it converges inside a single short run because it has only a handful of arms and one pool-total reward (the bandits/RL that split evidence per-thread — C3, C6 — likely cannot); (b) it dissolves the pool-reward-vs-per-thread-action credit confound by parameter-sharing (one threshold, one reward) rather than fighting it; (c) all-allow is literally one of its arms, so it is a strict superset of the byte-identical baseline and bounds its own downside to explorat…
- **Testing shape:** A single throwaway Python harness — call it cpp/stage_a/issue_control_ab.py, built in the established stage_b_ab.py / stage_b_poolbatch_sweep.py mold (subprocess-drives chocofarm-wire-ab-bench against an in-process StageAServer, parses its result line, writes under ~/w/vdc, hard subprocess timeout, no background watcher) — that scores ANY candidate issue-gate policy against the byte-identical all-allow baseline on the SAME faithful warm pool, with ZERO changes to the C++ bench, the bridge, the controller, or issue_engine.py. End to end:

(1) ONE in-process StageAServer is stood up over ONE pub…
