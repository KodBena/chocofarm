# ISMCTS agent — commission (verbatim)

> The exact prompt sent to the implementation agent. Its report is in `ismcts-report.md`.

---

You are implementing **Information-Set Monte-Carlo Tree Search (ISMCTS, Cowling, Powley & Whitehouse 2012)** as a pluggable policy in a small research project ("chocofarm"). Work EXCLUSIVELY in your git worktree at **/home/bork/w/vdc/chocobo-ismcts** (branch `feat/ismcts`). Do NOT touch /home/bork/w/omega (a different, unrelated project) or the sibling worktrees (chocobo, chocobo-nmcs, chocobo-consult).

PYTHON: `/home/bork/w/vdc/venvs/generic/bin/python` (has numpy, shapely). Always run commands under a `timeout`.

READ FIRST, end to end, in your worktree:
- `env.py` — the Environment (simulation model + belief + dynamics + unbiased rate eval) and the `Policy` contract. This is the interface you implement against; DO NOT modify it.
- `policies.py` — existing `Policy` subclasses (Greedy, CertaintyEquivalent, Rollout, SparseSampling). Templates; `_base_value` shows the leaf-rollout pattern. Import from here; do not edit.
- `run.py` — the harness; note module-level `realizable_static(env)` and `clairvoyant_rate(env)` (importable; run.py is guarded so importing won't execute it).
- `docs/STATUS.md`, `docs/results/*.md` — context.

THE PROBLEM (adaptive stochastic orienteering / belief-MDP): 20 treasures; exactly 5 present per run (uniform without replacement → 15,504 equiprobable latent "worlds"; a world is a bitmask, bit t = treasure t present; i.i.d. re-roll each run). 16 detection regions give binary DISJUNCTIVE observations ("≥1 present among a covered set"); 4 δ treasures (observe==collect). Objective: maximize long-run rate = treasures/time (renewal-reward) via Dinkelbach — for rate λ, maximize E[Σ value − λ·Σ time]; a policy's rate is its Dinkelbach fixed point (`env.dinkelbach_rate` does this). Belief = the numpy array of surviving worlds; it SHRINKS as you observe. This is a single-observer partially-observable problem; the **information set is exactly the belief** `bw`. A clairvoyant policy (free perfect info) scores +70% over the static route, so there is large value-of-information to capture; existing shallow policies capture almost none. ISMCTS is being tried because its determinization handles hidden state cleanly and its information-set search fits this structure.

INTERFACE: subclass `Policy` and implement `decide(self, env, loc, bw, collected, lam, rng) -> action`, returning `('t', i)`, `('d', i)`, or `TERMINATE` (=`("term", None)`). Primitives on `env`: `legal_actions(loc,bw,collected)`, `apply(loc,bw,collected,action,world)->(reward,loc',bw',collected',dt)`, `marginals(bw)`, `filter_treasure/filter_detector`, `sample_world(bw,rng)`, `d(a,b)`, `exit_cost(loc)`, `route_time(start,seq)`, `worlds`, `cover_mask`, `detectors`, `N`, `K`, `value`, `entry`, `tp`. To DETERMINIZE (sample a concrete latent world consistent with the belief), sample from `bw` (e.g. `int(rng.choice(bw))`) — the belief already encodes consistency, so this is unbiased; no particle filter needed.

IMPLEMENT SO-ISMCTS faithfully (Cowling, Powley, Whitehouse, "Information Set Monte Carlo Tree Search", IEEE TCIAIG 2012). Use WebSearch/WebFetch to confirm the algorithm if useful. Per iteration: (1) determinize — sample a world `w` from the current belief; (2) selection — descend the tree using UCB1 over actions legal/consistent in this determinization; (3) expansion; (4) simulation — a base playout (reuse Greedy/CertaintyEquivalent), scored by the λ-penalized return `Σ value − λ·(Σ travel + exit)`; (5) backpropagation. After the iteration budget, return the root's best action (most-visited or highest-value). The tree branches on actions; the determinization `w` resolves each action's observation outcome (present/absent, detector pos/neg) and hence which child you descend into — track this so statistics aggregate over the information set. Expose `iterations` and an exploration constant `c`. If network is unavailable, implement from your own knowledge and say so.

DELIVERABLES (commit on `feat/ismcts`; do NOT push — the coordinator pushes/merges):
- `ismcts.py` — `ISMCTSPolicy(iterations=..., c=...)` subclassing `Policy`.
- `eval_ismcts.py` — imports `env`, your policy, and `from run import realizable_static, clairvoyant_rate`; measures your policy's rate via `env.dinkelbach_rate(...)` and prints rate, % of the clairvoyant ceiling, and % of the VoI gap clawed back vs static, across a couple of iteration budgets.
- optional `docs/results/ismcts-result.md` (fixed label, not a live date — `Date.now()` style timestamps are unavailable).
- Do NOT edit `env.py`, `policies.py`, `run.py` (keep the branch conflict-free).
- Stage by EXPLICIT path (never `git add -A`) and commit, message ending `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

BOUNDED-SAFETY — CRITICAL: memory and wall-time STRICTLY bounded. No parallel processes, no `&`, no unbounded loops, no enumerating the belief space (belief is a cheap numpy world-set; never build a dict keyed over reachable beliefs; the MCTS tree must be bounded by the iteration budget). Every command under `timeout` (e.g. `timeout 240`). Start SMALL (a few hundred iterations, ≤200 evaluation runs); shrink if slow. A previous agent on this project exhausted the machine's RAM with unbounded parallel solves — do NOT repeat that. If slow, shrink the budget; never fan out.

RETURN (final message, complete — it is the record): the algorithm as implemented + literature consulted (and what you took from it); measured rate and % of the +70% ceiling clawed back at your iteration budgets; runtime; honest caveats.
