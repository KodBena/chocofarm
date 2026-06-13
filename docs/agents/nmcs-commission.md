# NMCS agent — commission (verbatim)

> The exact prompt sent to the implementation agent. Its report will be in `nmcs-report.md`
> once it completes.

---

You are implementing **Nested Monte-Carlo Search (NMCS, Cazenave 2009)** as a pluggable policy in a small research project ("chocofarm"). Work EXCLUSIVELY in your git worktree at **/home/bork/w/vdc/chocobo-nmcs** (branch `feat/nmcs`). Do NOT touch /home/bork/w/omega (a different, unrelated project) or the sibling worktrees (chocobo, chocobo-ismcts, chocobo-consult).

PYTHON: `/home/bork/w/vdc/venvs/generic/bin/python` (has numpy, shapely). Always run commands under a `timeout`.

READ FIRST, end to end, in your worktree:
- `env.py` — the Environment (simulation model + belief + dynamics + unbiased rate eval) and the `Policy` contract. This is the interface you implement against; DO NOT modify it.
- `policies.py` — the existing `Policy` subclasses (GreedyPolicy, CertaintyEquivalentPolicy, RolloutPolicy, SparseSamplingPolicy). Use them as templates; `_base_value` shows the leaf-rollout pattern. Import from here; do not edit.
- `run.py` — the harness; note module-level `realizable_static(env)` and `clairvoyant_rate(env)` (importable; run.py is guarded so importing won't execute it).
- `docs/STATUS.md`, `docs/results/*.md` — context.

THE PROBLEM (adaptive stochastic orienteering / belief-MDP): 20 treasures; exactly 5 present per run (uniform without replacement → 15,504 equiprobable latent "worlds"; a world is a bitmask, bit t = treasure t present; i.i.d. re-roll each run). 16 detection regions give binary DISJUNCTIVE observations ("≥1 present among a covered set"); 4 δ treasures (observe==collect). Objective: maximize long-run rate = treasures/time (renewal-reward) via Dinkelbach — for rate λ, maximize E[Σ value − λ·Σ time]; a policy's rate is its Dinkelbach fixed point (`env.dinkelbach_rate` does this). Belief = the numpy array of surviving worlds; it SHRINKS as you observe. A clairvoyant policy (free perfect info) scores +70% over the static route, so there is large value-of-information to capture; the existing shallow policies capture almost none. NMCS is being tried because it is single-agent search built for deep contingent planning.

INTERFACE: subclass `Policy` and implement `decide(self, env, loc, bw, collected, lam, rng) -> action`, returning `('t', i)` (go collect/observe treasure i), `('d', i)` (go to detector i), or `TERMINATE` (=`("term", None)`, end the run). Primitives on `env`: `legal_actions(loc, bw, collected)`, `apply(loc, bw, collected, action, world) -> (reward, loc', bw', collected', dt)`, `marginals(bw)`, `filter_treasure/filter_detector`, `sample_world(bw, rng)`, `d(a,b)`, `exit_cost(loc)`, `route_time(start, seq)`, `worlds` (the full latent set), `cover_mask`, `detectors`, `N`, `K`, `value`, `entry`, `tp`. To DETERMINIZE (sample a concrete latent world consistent with the current belief), sample from `bw` (e.g. `int(rng.choice(bw))`); the belief already encodes consistency, so sampling is unbiased.

IMPLEMENT NMCS faithfully (Cazenave, "Nested Monte-Carlo Search", IJCAI 2009). Use WebSearch/WebFetch to confirm the exact recursive algorithm if useful (level-n: at each step, for each legal action try a level-(n−1) search, keep the best continuation, play it; level-0 = a base/random playout). Adapt to: (a) FINITE-HORIZON episodic (a run ends at TERMINATE / when no informative action remains); (b) STOCHASTIC outcomes — actions have random results since the latent world is hidden, so a playout samples a world (resample per playout) and is scored by the λ-penalized return `Σ value − λ·(Σ travel + exit)`, averaged over a few sampled worlds to cut variance; (c) the λ passed to `decide` is the penalty. Expose a `level` parameter (test level 1 and 2). You may reuse Greedy/CertaintyEquivalent as the base playout. If network is unavailable, implement from your own knowledge and say so.

DELIVERABLES (commit on `feat/nmcs`; do NOT push — the coordinator pushes/merges):
- `nmcs.py` — `NMCSPolicy(level=...)` subclassing `Policy`.
- `eval_nmcs.py` — imports `env`, your policy, and `from run import realizable_static, clairvoyant_rate`; measures your policy's rate via `env.dinkelbach_rate(...)` and prints rate, % of the clairvoyant ceiling, and % of the VoI gap clawed back vs static, for level 1 and 2.
- optional `docs/results/nmcs-result.md` (fixed label, not a live date — `Date.now()` style timestamps are unavailable).
- Do NOT edit `env.py`, `policies.py`, or `run.py` (keeps the branch conflict-free for merging).
- Stage by EXPLICIT path (never `git add -A`) and commit with a clear message ending `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

BOUNDED-SAFETY — CRITICAL: keep memory and wall-time STRICTLY bounded. No parallel processes, no `&`, no unbounded loops, no enumerating the belief space (the belief is a cheap numpy world-set — sampling/filtering is microseconds; never build a dict keyed over reachable beliefs). Run every command under a `timeout` (e.g. `timeout 240`). Start with SMALL budgets (NMCS level ≤2, a handful of playouts per action, ≤200 MC evaluation runs); scale down if slow. A previous agent on this project exhausted the machine's RAM by running unbounded solves in parallel — do not repeat that under any circumstances. If something is slow, shrink the budget; never fan out.

RETURN (final message): the algorithm as you implemented it + any literature you consulted (with what you took from it); the measured rate and % of the +70% ceiling clawed back at level 1 and 2; runtime; and honest caveats. Your final message is the record — make it complete.
