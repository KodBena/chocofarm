# C++ actor exit_loop SWAP — adversarial review (report, verbatim)

> The complete output of the multi-agent adversarial review commissioned for the
> `CppActorExecutor` exit_loop SWAP (the change that injects the C++ Gumbel actor into
> `exit_loop`'s generation step). Reproduced **verbatim** per the verbatim-record
> discipline (ADR-0005) and the hack-rationalization-detector rule that the audit
> artifact reaches the maintainer **unmediated** — no summarizing, softening, or
> omitting a line. The commission (the exact lens prompts) is in
> `cpp-actor-swap-review-commission.md`. A maintainer-disposition appendix at the very
> end records what was acted on; it is **appended**, and alters nothing above it.
>
> Review shape: 3 lenses (contract-fidelity, correctness/failure-modes/honesty,
> scope/docs/hack-rationalization); each raw finding adversarially verified
> (refuted-or-confirmed) by an independent agent. Tally (verbatim from the run log):
> **"reviewed 3 lenses; 14 findings, 7 confirmed, 7 refuted"** — 17 agents total.
>
> This review found a real miss the implementer (Claude Opus 4.8) had not caught: the
> C++ actor silently dropped `explore_plies` (always-greedy generation), an
> undischarged hack by the project's own fail-loud standard. All seven confirmed
> findings were fixed before commit; see the disposition appendix.

---

## Lens summaries (verbatim)

### Lens: contract-fidelity

CppActorExecutor matches the ParallelExecutor surface that exit_loop.run drives almost exactly: generate/evaluate/close signatures are positionally/kw-identical to the call sites (generate at exit_loop.py:426, evaluate at :471, the recs_all unpack at :438, the finally close at :519); the four (X,PI,M,Y) blocks are read with the spec dtype (result_spec.RESULT_DTYPE) and the executor's in_dim/n_slots, which are the same in_dim/n_slots the net was built from (exit_loop.py:222-223 -> :350-352); records are the right element types ((feat,pi,mask) numpy rows + a python float g); every runner CLI flag it builds is real in cpp/src/main.cpp; and the Part-B fail-loud guard is honest and well-scoped. There is ONE real contract-fidelity divergence: the `explore_plies` argument that the loop threads into generate() is silently accepted-and-dropped — the C++ Gumbel actor always executes the temperature-0 SH survivor, so it generates entirely-greedy self-play while both Python generation paths (serial and the worker pool) sample the executed action from π′ for the first explore_plies plies. This is a real behavioral difference in GENERATION (the lens's subject) that is NOT in the module's "known-deferred" list. A second, minor item: evaluate() uses a fixed construction-time eval seed rather than the loop's HOT eval_seed. Everything else in the contract is clean.

### Lens: correctness-failure-honesty

The swap is structurally sound: the executor contract (generate/evaluate/close/.run/.cores) matches ParallelExecutor, the import cycle is cleanly broken (cpp_executor never imports exit_loop; exit_loop imports it lazily in the --cpp-runner branch; GumbelPolicy is imported lazily), the Part-B guard is correct on every branch including the lam_blend=None edge (short-circuit avoids None<1.0), the empty-episode idx-gap handling is correct (runner.cpp:168 skips empty episodes, _read_records tolerates missing keys), and the deferred Part-B blend is honestly fail-loud, not a hidden hack. close() is always called in exit_loop's finally and is double-close safe. However, there is one real ADR-0002 violation (a silent empty-buffer path when the runner exits 0 but its result keys are absent — no count reconciliation against the runner's reported "wrote N episode(s)"), one deadlock-fix-H2 regression (the executor's redis connection has no bounded socket timeout and no fail-loud ping, unlike transport.connect()), and one honest-semantics defect (evaluate() claims to "mirror the serial path" but keys its search RNG off base_seed+10000 instead of the HOT eval_seed exit_loop actually drew eval_worlds from, so the C++-swap eval is not the same quantity the serial baseline computes — breaking A/B comparability). The remaining items are minor.

### Lens: scope-docs-hack

The SWAP is the right general structure: exit_loop is made generic over an injected generation executor satisfying the same (generate/evaluate/close + .run/.cores) contract ParallelExecutor satisfies, and CppActorExecutor drives GENERATION while eval/train/replay/checkpoint/hp-registry are inherited unchanged. The Part-B deferral is an HONEST discharge — generate() fails loud per ADR-0002, the guard's remedy ("Python pool --workers>0") is real (the Python generate_episode genuinely emits per-decision boots and blends via the one blended_returns_to_go), the documented C++ v_mix path matches that exact structure, and the pure-MC Y the runner emits is verified to be the same suffix-rule quantity as the Python pure-MC limit. cpp_actor_loop.py's supersession is handled cleanly (kept as a documented minimal demo, not dead code). ADR-0006 header is present. BUT the patch silently swallows TWO contract arguments: explore_plies (default 4) is accepted to satisfy the signature and then dropped with no guard/warning/doc — the C++ runner runs Gumbel at temperature 0 for every ply, so on DEFAULT settings the swap generates zero-exploration self-play data, a systematically different generation distribution than both the serial and parallel Python paths. This is the SAME class of gap as Part-B but left SILENT where Part-B is loud — it violates the ADR-0002 precedent the Part-B guard itself sets, and the 1-iter smoke could not surface it. Eval being in-process Python is a defensible reading of "swap into generation" but leaves the ~200-episode eval fan-out fully serial (no C++, no pool), slower than the parallel baseline it supersedes. The self.cores comment ("the runner self-schedules") misdescribes a single-threaded serial runner. Verdict: narrower-but-justified overall, but the silent explore_plies drop is an undischarged hack by the project's own fail-loud standard.

---

## Confirmed findings (verbatim)

### [contract-fidelity · major] generate() silently drops explore_plies — the C++ actor is always-greedy self-play, diverging from both Python generation paths

**Location:** `/home/bork/w/vdc/1/chocofarm/chocofarm/az/cpp_executor.py:86-118 (generate accepts explore_plies, drops it); cpp/src/runner.cpp:44 + cpp/include/chocofarm/gumbel.hpp:135 (always temperature-0 survivor)`

**Reasoning:**

Every link in the finding's chain verifies against the working-tree code.

PYTHON SIDE (the contract being matched):
- exit_loop.py:544 defines `--explore-plies` (default 4); it flows into the config snapshot, read at :388 as `explore_plies = snap.cfg.loop.explore_plies`. Registry default is also 4 (hp/schema.py:201, `explore_plies: int = hp(4, Mut.HOT, ...)`).
- exit_loop.py:426-428 threads `explore_plies` positionally (5th arg) into `executor.generate(net, it, gen_worlds, lam, explore_plies, lam_blend, n_step, ...)` for ANY injected executor.
- The serial path (generate_episode, exit_loop.py:97-101) sets `temp = 1.0 if ply < n_explore_plies else 0.0` and samples the EXECUTED action from π′ for the first explore_plies plies, argmax thereafter (design §6, "temperature on executed action to diversify trajectories").
- The parallel path honors it too: parallel.py:132/151 packs explore_plies into each task tuple; worker.py:182-194 forwards it into the same generate_episode. So BOTH Python generation paths apply the temperature-1 prefix.

C++ SIDE (the swap):
- cpp_executor.py:86-118 — generate() accepts `explore_plies` positionally but it is ABSENT from `cmd` (lines 105-112, which carry only --instance/--faces/--run/--phase/--version/--res-token/--episodes/--lam/--max-steps/--seed/--policy/--gumbel-m/--gumbel-n-sims and the _RUNNER_HOT_KNOBS c_*/max_depth). The only guards in generate() are the Part-B n_step/lam_blend checks (lines 96-100). explore_plies is read and dropped.
- main.cpp has no --explore-plies flag (only --gumbel-* knobs, confirmed by grep of the help text and the arg-parsing block at 173-179).
- runner.cpp:44 calls `policy.decide_target(env, loc, bw, collected, lam, rng)` with no temperature, every ply.
- gumbel.hpp:135 documents GumbelAZPolicy returns "the EXECUTED action = the SH survivor at temperature 0"; :152/:174 confirm decide_target returns that survivor. No ply-dependent temperature-1 executed-action path exists.

NET EFFECT: the C++ actor produces entirely greedy self-play; the temperature-1 explore_plies prefix the operator configured (default 4) is silently ignored, changing the GENERATION trajectory/state distribution the net trains on relative to both Python actors. This is the exact behavioral divergence the contract-fidelity lens targets.

HONESTY/SCOPE: the module docstring (cpp_executor.py:22-29) names ONLY the Part-B value-target blend as known-deferred; explore_plies is unmentioned and unguarded. This is asymmetric with the Part-B handling, which IS fail-loud-guarded and documented per ADR-0002 — so the codebase's own honesty standard is not met for this second wire-crossing gap.

The new test even demonstrates it: test_cpp_actor_executor_partb_fails_loud passes explore_plies=4 to generate() and the swap-turns test runs at the default explore_plies=4, both proceeding with no banner/guard about the dropped exploration.

Refutation attempts failed: explore_plies default is genuinely nonzero (4); both Python paths genuinely honor it; the divergence is not benign (it narrows the training state distribution); and it is not already documented or guarded. The one mitigating nuance is that the PI training TARGET is still the correct improved-π in both paths and individual transition value targets are not corrupted — so this degrades exploration diversity rather than producing crashes or wrong values, which is why this is major and not a blocker.

**Fix:**

Mirror the Part-B guard's ADR-0002 honesty. Minimum (option a): in CppActorExecutor.generate, after the Part-B guard, add `if explore_plies and explore_plies > 0: raise RuntimeError(...)` explaining the C++ actor plays temperature-0 greedy self-play and cannot honor the temperature-1 explore_plies prefix across the wire (suggest running --explore-plies 0 with --cpp-runner, or --workers for the exploration prefix); and add explore_plies to the module docstring's known-deferred list alongside the Part-B blend. Preferred (option b): wire it through — add a `--explore-plies` flag to the runner (main.cpp) and a temperature-1 executed-action sample path in GumbelAZPolicy::decide_target for the first N plies (mirroring generate_episode's `temp = 1.0 if ply < n_explore_plies else 0.0`), then pass `--explore-plies` in cpp_executor.py's cmd. Either way the swap must not silently ignore a generation-shaping argument the contract passes it.

### [correctness-failure-honesty · major] generate() trains on an empty buffer SILENTLY when the runner exits 0 but its result keys are absent (no count reconciliation) — the exact ADR-0002 silent-empty-buffer it claims to prevent

**Location:** `chocofarm/az/cpp_executor.py:113-118,120-144 (and main.cpp:168,219)`

**Reasoning:**

Verified against the working-tree code. cpp_executor.py:113-118 guards ONLY rc!=0 (and subprocess.run's timeout). The success path _read_records (cpp_executor.py:120-144) derives per-episode existence purely from redis key presence: `if yb is None or xb is None: continue` (line 132-133). It therefore cannot distinguish a legitimately empty episode (runner's `if ep.n==0: continue`, runner.cpp:168 — wrote nothing) from a blob that was written and then evicted before the parent read it. The transport instance is allkeys-lru with a maxmemory cap (config.py:11, documented there as 'the safety net for the transport's short-lived weight/result churn'), so eviction-before-read is a real window under memory pressure: the runner writes E result blobs across a full run (runner.cpp:170) while prior-iteration churn and the published weight blob coexist on the capped instance, and the parent only reads after the subprocess returns.

The runner already emits the exact reconciliation datum: `std::cerr << prog << ": wrote " << *written << " episode(s)"` (main.cpp:219), where `written` counts only non-empty episodes actually written (runner.cpp:141-175). subprocess.run uses capture_output=True so this lands in proc.stderr — but proc.stderr is consulted ONLY on the rc!=0 branch (cpp_executor.py:115); on the success path the count is never parsed or reconciled.

The contrast with the Python pool is load-bearing and correct: ParallelExecutor.generate drives read_and_delete_results off `metas` (idx,n,fd,ns) returned through the multiprocessing pipe (parallel.py:155-156, transport.py:192-227). For a meta with n>0 whose blob is gone, conn.get returns None and `np.frombuffer(None, dtype=dt).reshape(n, fd)` raises a loud TypeError (I confirmed: 'a bytes-like object is required, not NoneType'). So a worker that ran but lost its blob fails LOUD on the Python path and is SILENTLY skipped on the C++ path. The C++ path has no meta channel and derives existence from key presence alone.

Downstream silence confirmed: a partial loss (a subset of episodes evicted) yields valid-shape X/PI/M/Y over the surviving subset, finite stats, and trains cleanly on the subset with zero indication anything vanished. A total loss collapses to recs_all=[] -> X=np.asarray([],f32) shape (0,) (exit_loop.py:440), buf.add stores it, train_epochs calls net.set_value_scale(Y.mean()) where Y.mean() on an empty array is nan with only a RuntimeWarning, not a raise (verified; set_value_scale clamps y_std to 1.0 but keeps y_mean=nan, mlp.py:324-328), then n=0 so steps=max(1,0)=1 and every batch is empty -> `if len(b)==0: continue`, training zero steps. This is precisely the silent-empty-buffer ADR-0002 is cited to forbid (the new module's own docstring invokes ADR-0002 for the Part-B guard).

Neither new test catches it: test_cpp_actor_exit_loop_swap_turns asserts only exit 0 / 'DONE 1 iters' / checkpoint-exists with no transition-count assertion; the Part-B test only covers the blend guard. The established 1-iter smoke ran a tiny batch under no memory pressure, so it could not surface eviction — exactly the gap the prompt flags.

One over-claim to discount: the finding's secondary 'env-var skew between parent CHOCO_TRANSPORT_REDIS_DB and the subprocess env' scenario is weak — subprocess.run inherits the parent's full env by default, and both sides resolve the same CHOCO_TRANSPORT_REDIS_* contract with identical defaults (config.py vs cpp/src/transport.cpp:81-83), so they land on the same db absent a deliberately inconsistent env. The defect does not depend on that path; the eviction path is sufficient and real.

Severity major rather than blocker: it is a latent silent-failure that requires real transport memory pressure to trigger (not the every-run/smoke path, which is why the green smoke and the 196-pass suite did not catch it), but it is a genuine ADR-0002 violation on a path whose entire purpose is to be loud, with a clean cheap fix the runner already enables.

**Fix:**

Reconcile the read count against the runner's reported count. Minimal, in cpp_executor.generate(): parse the runner's 'wrote N episode(s)' from proc.stderr (or, cleaner, have the runner emit `wrote=N` on stdout in a stable parseable form) and, after _read_records, assert that the number of distinct episode indices whose blobs were found equals N; raise RuntimeError loudly on mismatch. As a floor even without parsing: raise when _read_records returns fewer non-empty episodes than the runner reported written, and in particular raise when it returns [] but n_eps>0 unless the runner explicitly reported 0 written. Do not let a non-empty-requested generation collapse to a smaller-or-empty buffer without a loud failure — mirror the Python pool's structural meta-driven reconciliation (transport.read_and_delete_results raising on a missing blob for an n>0 meta).

### [correctness-failure-honesty · minor] Executor's redis connection has no bounded socket timeout and no fail-loud ping — re-introduces the deadlock-fix-H2 hang and defers a dead-redis failure

**Location:** `chocofarm/az/cpp_executor.py:83`

**Reasoning:**

The factual base is correct: cpp_executor.py:83 builds `self._conn = redis.Redis(**transport_redis_params())`, and transport_redis_params() (config.py:55-59) carries only host/port/db — no socket_timeout, no socket_connect_timeout, no .ping(). All redis ops in the executor ride this connection: publish_weights' pipe.execute (via line 102), and _read_records' conn.get/conn.delete (lines 131/143). The contract it claims to be a drop-in for, ParallelExecutor, instead builds its parent connection via the HARDENED path: `transport.RedisTransport(_connect())` where _connect is transport.connect (parallel.py:116, import at :87), and connect() sets config.redis_socket_timeout()/redis_connect_timeout() and pings (transport.py:141-148). So there IS a real divergence from the project's documented, env-overridable timeouts (60s/10s) and the fail-loud ping discipline (transport.py docstring line 8 cites this as deadlock-fix-H2/ADR-0002). The bare pattern was copied from the superseded cpp_actor_loop.py (HEAD line 93), which is the deliberately minimal demo, into the production swap path.

BUT the load-bearing claim — that this 're-introduces the H2 hang' and 'a stalled redis read can block the whole loop indefinitely' — is REFUTED on this working tree. The installed redis-py is 8.0.0 (verified: import redis; redis.__version__). Its redis.Redis(...) defaults socket_timeout=5 and socket_connect_timeout=5 — confirmed not just from the __init__ signature but from the actual connection_pool.connection_kwargs of a bare-constructed client. So a stalled socket read raises redis.TimeoutError in ~5s, NOT a forever futex-wait. The H2 fix's own premise (transport.py:130: 'The default socket_timeout=None ... block FOREVER') is specific to an older redis-py and does not hold here. There is no indefinite loop hang in this code; the 'major' severity rests on a failure mode that cannot occur on the installed stack. pyproject.toml carries no redis version pin, so the runtime is the shared venv's 8.0.0.

The .ping() deferral sub-claim is true but minor: connect() pings at construction so unreachable redis fails loud immediately; cpp_executor.py:83 defers that to the first publish_weights inside generate(), where it still fails loud within the 5s connect timeout — a few lines later, not at __init__, and never silent. Net: a genuine consistency/defense-in-depth gap (route through connect() like the contract peer does), but the asserted indefinite-hang / H2-reintroduction does not occur, so minor, not major.

**Fix:**

Construct the connection via transport.connect() so the executor inherits the project's explicit, env-overridable socket_timeout/socket_connect_timeout and the fail-loud ping at construction, matching ParallelExecutor (parallel.py:116). Concretely, in cpp_executor.py:83 replace `self._conn = redis.Redis(**transport_redis_params())` with `from chocofarm.az.transport import connect as _connect` and `self._conn = _connect()`; this also lets the unused `import redis` and `from chocofarm.config import transport_redis_params` be dropped. This is consistency/defense-in-depth (don't rely on a library default timeout that merely happens to be 5s); it does not fix a live indefinite hang, since redis-py 8.0.0 already bounds the socket at 5s.

### [correctness-failure-honesty · minor] evaluate() does not compute the same quantity as exit_loop's serial eval: it keys the search RNG off base_seed+10000, not the HOT eval_seed exit_loop drew eval_worlds from — the docstring's 'mirroring the serial path' is false

**Location:** `chocofarm/az/cpp_executor.py:82,156 vs chocofarm/az/exit_loop.py:476`

**Reasoning:**

The factual core is verified true. Serial eval (exit_loop.py:476) seeds its search RNG with ev_rng=np.random.default_rng(eval_seed), eval_seed=snap.cfg.eval.eval_seed (HOT, line 390, default 12345) — the SAME seed that draws eval_worlds (lines 418-419). CppActorExecutor.evaluate() (cpp_executor.py:156) seeds rng=np.random.default_rng(self._eval_seed), self._eval_seed=base_seed+10_000 (line 82), where base_seed is master_seed=cfg0.loop.seed (exit_loop.py:315,351) — a construction-frozen value structurally unrelated to eval_seed (a separate config field). With the test's --seed 7 the C++ eval seed is 10007 vs serial 12345: definitively different.

The RNG materially affects the eval result even at temperature 0: gumbel_search.py:250 draws g=rng.gumbel(...), line 251-253 picks the considered set by argsort(logits+g), line 257-258 runs Sequential Halving (drop by g+logits+σ(q̂)) whose survivor IS the executed action at temp 0 (line 277-278), and visits draw determinizations env.sample_world(bw,rng) (lines 334,350) that feed q̂. env.simulate (env.py:374) threads this exact rng into policy.decide. So a different search seed => different greedy action sequence => a different greedy-rate ESTIMATE of the same net. The worlds match (parent's eval_worlds passed in), so only the search noise diverges — exactly as the finding states.

So the eval is NOT bit-for-bit comparable to a serial eval of the same checkpoint. The literal docstring (cpp_executor.py:151) "The eval randomness is fixed across iterations (only the net changes), mirroring the serial path" is the contested claim. Under a strict parse, "mirroring the serial path" attaches to the structural property "fixed across iterations (only the net changes)," which IS genuinely shared (both reseed identically each iteration) — so the claim is technically defensible. But combined with lines 18-20 ("exit_loop's OWN ... eval ... measures the net's greedy rate") and the drop-in framing, a reader doing an A/B between serial and C++-swap eval would reasonably read it as implying comparability the code does not deliver. In a codebase whose ADR-0002/ADR-0009 explicitly forbid unsubstantiated equivalence claims, that overstatement is a real documentation-honesty defect.

But it is ONLY that — a docstring phrasing issue, not a code-behavior bug, resource leak, or contract mismatch. The eval is functionally correct: it estimates the net's greedy rate over the parent's worlds with a fixed reproducible seed, the same conceptual quantity the serial path measures. The reviewer's "major" and "silently breaks A/B comparability" are overstated: the accepted ParallelExecutor.evaluate path ALSO diverges from eval_seed via a worker seed-fold (parallel.py, verified) and is the established baseline — so this non-bit-equivalence is the norm, not a C++-unique correctness break. The finding itself concedes this. No test depends on the contested claim (the new tests only check the swap turns and the Part-B guard). Hence: real, but minor — a precision/honesty fix to the docstring, not the behavioral eval_seed-threading change (which would itself shift the eval number and carries its own A/B implications, out of scope for correcting a false claim).

**Fix:**

Correct the docstrings rather than change behavior. At cpp_executor.py:149-151, replace "mirroring the serial path" with a plain statement: the eval uses an independent, construction-fixed search seed (base_seed+10000), distinct from the serial path's HOT eval_seed, so its greedy-rate is a valid estimate of the same quantity but is NOT bit-for-bit comparable to a serial/eval_seed-driven eval of the same checkpoint (the same non-equivalence the Python pool's evaluate() carries via its worker seed-fold). Optionally soften lines 18-20 similarly. If exact serial comparability is later wanted, the behavioral alternative is to give evaluate() an eval_seed parameter threaded from snap.cfg.eval.eval_seed and seed np.random.default_rng(eval_seed) — but that is a deliberate scope decision, not required to remove the false claim.

### [scope-docs-hack · major] explore_plies is silently dropped — the swap generates zero-exploration self-play data on default settings (ADR-0002 violation, the Part-B guard's own standard)

**Location:** `chocofarm/az/cpp_executor.py:86-93 (generate signature accepts explore_plies, body ignores it); cpp/src/gumbel.cpp:587-588 (temp 0 only); chocofarm/az/exit_loop.py:97 (Python path uses it); chocofarm/hp/schema.py:201 (default=4)`

**Reasoning:**

Every claim verifies against the working tree.

1) Contract mismatch is real. `CppActorExecutor.generate` accepts `explore_plies` (cpp_executor.py:87) but the body never references it again — it is not added to `cmd` (lines 105-112 only forward m/n_sims and the hot knobs) and there is no runner flag to forward it to. Confirmed `grep` finds `explore_plies` exactly once in cpp_executor.py (the signature). Contrast: `ParallelExecutor.generate` threads `explore_plies` into each task tuple (parallel.py:151) and the serial/worker path honors it as temperature (exit_loop.py:97 `temp = 1.0 if ply < n_explore_plies else 0.0`).

2) The C++ side genuinely runs temperature 0 only. gumbel.cpp:580/587-588: the executed action is the SH survivor "at temperature 0", "the temperature>0 sampling path is a 1b/production concern". runner.cpp:44 calls `policy.decide_target(...)` with no temperature/n_explore_plies parameter at all, inside the per-ply loop (runner.cpp:36), so EVERY executed action across every ply is temperature 0. main.cpp's CLI (lines 54-75, 173-179) exposes no exploration/temperature flag.

3) It is the DEFAULT path. `--explore-plies` defaults to 4 (exit_loop.py:544); the registry default is 4 (schema.py:201); it flows `snap.cfg.loop.explore_plies` (exit_loop.py:388) positionally into `executor.generate(...)` (exit_loop.py:426-427) for ALL executors including the C++ one. There is no override that the C++ executor consults. So the literal default `exit_loop --cpp-runner ...` produces fully greedy, zero-exploration self-play data.

4) The swallow is genuinely silent. The module docstring (lines 1-35) and the generate docstring (lines 90-93) discuss the Part-B value-target deferral at length but say nothing about explore_plies, temperature, or greedy-only generation. No warning, no raise.

5) The Part-B comparison is apt and is the module's own standard. For an injected argument it cannot honor (the blend), the module RAISES (lines 96-100) and documents it (lines 22-29). For explore_plies — the same class of unhonored-contract-argument — it does neither. This is the exact internal inconsistency ADR-0002 (fail loudly) governs.

6) It is substantively load-bearing, not cosmetic. docs/design/alphazero-surrogate-design.md:506 names "executed-action temperature" as the documented mitigation that keeps deep-sensing lines in the training data and stops the approach from "quietly becoming NMCS-with-a-net". Dropping it changes the generation distribution in the exact dimension that matters for value-net calibration, and silently breaks the serial/parallel/cpp A/B comparison the three executors exist to support.

Refutation considered and rejected: "the C++ search's temperature-0-only scope is a known/documented 1a limitation, so the swap just surfaces it, not a new defect." The C++ scope boundary is the cause, not the defect. The defect is the Python contract boundary (cpp_executor.py:86-93) accepting and silently swallowing an argument it cannot honor, on the default path, contradicting the same module's Part-B precedent. That code is entirely in this working-tree change.

Severity: major rather than blocker. It does not crash, corrupt data, or block the honestly-deferred Part-B story; the smoke test turns. But on the DEFAULT invocation it silently produces a systematically different generation distribution than both Python paths, contradicts a documented design mitigation, and violates the module's own fail-loud standard — degrading any real run and misleading the A/B the executors exist for. The established 1-iter/4-episode smoke ("118 transitions, turned end-to-end") cannot surface this: it proves the pipe is connected, not that the generation distribution matches.

**Fix:**

Mirror the Part-B guard in `CppActorExecutor.generate`. Immediately after the Part-B check (cpp_executor.py:100), add:

    if explore_plies and explore_plies > 0:
        raise RuntimeError(
            "CppActorExecutor: the C++ Gumbel actor executes the Sequential-Halving survivor at "
            f"temperature 0 every ply; explore_plies={explore_plies} (sample the executed action from "
            "pi' for the first this-many plies) does not cross the C++ wire. Run the Python pool "
            "(--workers>0) for executed-action exploration, or pass --explore-plies 0 to acknowledge "
            "greedy (temperature-0) generation with the C++ actor. The documented path to honor it: "
            "expose temperature>0 sampling on the C++ Decision so the runner samples the first "
            "explore_plies executed actions from pi'.")

This makes the unhonored argument fail loud exactly as Part-B does, and forces the operator to either run the Python pool or explicitly opt into greedy generation via --explore-plies 0. Additionally, add an explore_plies note to the module docstring alongside the Part-B note (lines 22-29) so the divergence and its remedy are documented, not just guarded. (Note: the new test `test_cpp_actor_executor_partb_fails_loud` passes explore_plies=4 to generate(); once the guard lands, those calls must pass explore_plies=0 so they reach the Part-B assertion they target, and a dedicated test should assert the explore_plies>0 raise.)

### [scope-docs-hack · minor] self.cores comment claims 'the runner self-schedules' — it is a single-threaded serial episode loop (ADR-0009 unsubstantiated perf framing)

**Location:** `chocofarm/az/cpp_executor.py:81; cpp/src/runner.cpp:142`

**Reasoning:**

Verified the factual core against the working-tree code.

1. The comment exists verbatim: chocofarm/az/cpp_executor.py:81 sets `self.cores: list[int] = []  # the runner self-schedules; no per-worker core pin`.

2. The production runner is strictly serial. cpp/src/runner.cpp:142 is `for (int idx = 0; idx < cfg.episodes; ++idx)` driving `run_episode(...)` one episode at a time. A grep over runner.cpp for thread|affinity|pool|self.?schedul|parallel|work.?steal|jthread|async returned NONE; main.cpp (which calls `chocofarm::run` at line 212) also has no thread/pool/affinity/OMP/Eigen-parallelism references. So the C++ runner is one subprocess executing episodes serially — it does not schedule work across cores in any sense.

3. The finding's supporting claim is accurate: SerialRuntime/PoolRuntime/search_runtime and std::thread/sched_setaffinity appear ONLY in cpp/src/{wire_bench,wire_pool_bench,search_runtime_bench,fiber_proto,serial_runtime_check,local_mlp_bench,wire_parallel_bench}.cpp plus the search_runtime.{hpp,cpp} library — never in runner.cpp or main.cpp. docs/design/cpp-search-runtime.md frames those runtimes as a benchmark-harness concern. So the work-stealing pool is not wired into the production runner.

Therefore "self-schedules" is genuinely misleading: a single-threaded serial loop is the opposite of self-scheduling across cores. Under ADR-0009 (attach substantiation to perf/equivalence claims) and the project's "honest and mechanistic" posture, this is a real framing defect: the swap replaces a 4-core Python pool (Part A, ~1.9x ceiling) with a single-threaded C++ subprocess, and the comment's wording implies a parallelism the runner does not have.

Calibration to minor (not major): the inaccuracy is confined to one explanatory word in a code comment. The `self.cores = []` VALUE is correct — there genuinely are no per-worker cores to pin — and the attribute is never read by the loop in the --cpp-runner path (exit_loop.py only prints executor.cores in the --workers branch at line 365; the cpp branch's print at 351-356 does not reference .cores). So there is no runtime defect, contract mismatch, or leak — only an honesty-of-framing issue that the project's own ADR-0009 elevates to a first-class concern.

**Fix:**

Reword the comment at chocofarm/az/cpp_executor.py:81 to the honest mechanism, e.g.: `self.cores: list[int] = []  # the runner is one subprocess running episodes serially; no per-worker core pin`. Optionally, in the module docstring's "swap into GENERATION" narrative, note that generation is now serial-in-C++ vs 4-core-in-Python (Part A) so that the net throughput comparison (per-decision C++ speedup vs the lost ~1.9x pool parallelism) is flagged as unsubstantiated per ADR-0009 rather than implied to be a clear win.

### [scope-docs-hack · nit] Part-B guard condition carries a defensive `lam_blend is not None` that cannot be None — harmless but slightly misleading about the contract

**Location:** `chocofarm/az/cpp_executor.py:96; chocofarm/hp/schema.py:119`

**Reasoning:**

Verified against the working-tree code. The guard at chocofarm/az/cpp_executor.py:96 is `if (n_step is not None) or (lam_blend is not None and lam_blend < 1.0):`. The `lam_blend is not None` arm is dead, and the finding's factual chain holds:

1. exit_loop.py:386 sets `lam_blend = snap.cfg.value.td_lambda`; that is the only producer (the executor.generate call at exit_loop.py:426-427 passes this through).
2. schema.py:119 declares `td_lambda: float = hp(1.0, ...)` — a float defaulting to 1.0.
3. The codec actively forbids None: schema.py:280 `if not (0.0 <= v.td_lambda <= 1.0):` would itself raise a TypeError on a None comparison, so a None td_lambda can never survive decode.
4. Both generate() signatures type the param as non-Optional `lam_blend: float` (parallel.py:132-133 and cpp_executor.py:87), so the contract itself excludes None.

Decisively, this is inconsistent with the codebase's own established convention: every sibling site that tests the blend compares directly without a None guard — schema.py:273 `if v.n_step is not None and v.td_lambda < 1.0:`, exit_loop.py:326 `if cfg0.value.n_step is not None and cfg0.value.td_lambda < 1.0:`, and exit_loop.py:330. Only the new cpp_executor.py:96 adds the redundant `lam_blend is not None and`.

Not a bug: because of the `and lam_blend < 1.0` short-circuit the None arm can never be the deciding factor, and the tests (test_cpp_runner.py:450 region) exercise lam_blend=0.5 and lam_blend=1.0/n_step=3 with correct behavior either way. It is a genuine but purely cosmetic imprecision — the guard suggests a None case the schema rules out, and diverges from three sibling call sites. The reviewer correctly classified it as a nit and explicitly disclaimed it as not a bug. Confirmed as a real, low-severity (nit) observation about THIS code.

**Fix:**

Drop the dead arm to match the schema guarantee and the three sibling call sites: change cpp_executor.py:96 to `if (n_step is not None) or (lam_blend < 1.0):`. (Alternatively keep it but add a comment that the None branch guards a case the schema codec forbids — but matching the existing `td_lambda < 1.0` convention is cleaner and removes the misleading implication of a None case in the contract.)

---

## Refuted findings (verbatim)

### [contract-fidelity] evaluate() uses a fixed construction-time eval seed, not the loop's HOT eval_seed

**Why refuted:**

The finding's core consequence-claim is refuted by the code. It asserts "a registry eval_seed retune lands on the serial/worker eval RNG and NOT on the C++ executor's eval RNG," but the WORKER POOL eval rollout RNG does NOT track eval_seed either.

Tracing the three eval paths' rollout RNG (the RNG passed into env.simulate):
- Serial (exit_loop.py:476): `ev_rng = np.random.default_rng(eval_seed)` with `eval_seed = snap.cfg.eval.eval_seed` (HOT) — tracks HOT eval_seed.
- Worker pool (worker.py:140, 220, 226): `task_rng` -> `_fold_seed(self.base_seed, version, "eval", idx)`, and `self.base_seed` is `master_seed = cfg0.loop.seed` (RESTART, exit_loop.py:315,362). It folds base_seed, NOT eval_seed.
- C++ executor (cpp_executor.py:156): `np.random.default_rng(self._eval_seed)` with `_eval_seed = base_seed + 10_000`, base_seed = master_seed (exit_loop.py:351). Folds base_seed, NOT eval_seed.

So the C++ executor's rollout RNG is CONSISTENT with the worker pool (the production parallel path) — both fold the RESTART base_seed and ignore HOT eval_seed. The actual rollout-RNG divergence is serial-vs-parallel and is pre-existing in the substrate; the C++ swap does not introduce it.

The finding's operator-impact claim ("a no-op under --cpp-runner") is also false. `eval_seed` additionally seeds the held-out world draw `eval_worlds` (exit_loop.py:418-419), which happens in the PARENT regardless of executor and is passed into executor.evaluate(...) at line 471. So retuning eval_seed DOES change the held-out world set under --cpp-runner — it is not a no-op; only the rollout-RNG portion is pinned, matching the worker pool.

What remains is a minor docstring imprecision: cpp_executor.py:151 says eval randomness is "fixed across iterations, mirroring the serial path" — but it actually mirrors the WORKER POOL path (folds base_seed), not the serial path (which tracks eval_seed). That is a nit in prose accuracy, not a contract or behavior defect, and the finding's stated rationale for why it matters is itself incorrect. Default to not-a-defect.

### [correctness-failure-honesty] Partial-write / partial-presence relies on the runner's rc and on np.frombuffer(None) raising, with weaker diagnostics than the parent transport

**Why refuted:**

The finding's factual sub-claims check out, but its load-bearing framing — "weaker diagnostics than the parent transport" — is false, and that inversion is what makes it not a defect in this working-tree code.

Verified facts (all true):
- C++ write_results issues four sequential, non-pipelined SETs in order X→PI→M→Y (cpp/src/transport.cpp:231-234), each short-circuiting on failure via `if (!rx) return rx`.
- The protecting invariant is sound and complete: a failed SET → write_results returns std::unexpected → run() propagates `if (!wr) return std::unexpected(wr.error())` (cpp/src/runner.cpp:172) → main returns 1 (cpp/src/main.cpp:212-218) → generate() catches non-zero rc and raises BEFORE _read_records (cpp_executor.py:114-118). So a partial write from a failed SET can never reach the reader.
- _read_records guards only `yb is None or xb is None` (cpp_executor.py:132), not pib/mb. A None PI → bare TypeError from np.frombuffer(None); a wrong-length X (instance skew) → numpy reshape ValueError (cpp_executor.py:137-139).

Refutation of the core claim ("weaker than the parent"): I read the parent read_and_delete_results (transport.py:212-222). It does `xb, pib, mb, yb = blobs[...]` then UNCONDITIONALLY np.frombuffer(pib).reshape(n, ns) with NO None-guard on any blob and NO named reshape-mismatch message. So for the exact two scenarios the finding cites — PI absent, wrong-length X — the parent raises the IDENTICAL bare TypeError / ValueError. The new code is in fact strictly STRONGER on presence: it None-guards yb/xb and continues (the legitimate empty-episode path), which the parent does not even do. The new module therefore does not regress, diverge below, or fall short of the standard it is measured against; the finding's title/detail invert the comparison.

What's left is a pure friendliness suggestion applied to behavior that is already ADR-0002-compliant: every cited path is LOUD (an exception, never a silent mis-decode), which the finding itself concedes ("loud (acceptable under ADR-0002)"). The triggers are either cross-deferred to finding #1 (independent LRU eviction of one key on the 6380 allkeys-lru instance — a separate finding) or operator misconfiguration of --cpp-instance, which the new CLI help already flags ("must describe the SAME env ... so the (X,PI,M,Y) dims match the net"). No genuine defect in THIS code: correctness holds, the invariant holds, and diagnostics are at parity-or-better with the canonical sibling.

### [correctness-failure-honesty] The 'worlds is used only for COUNT; the runner draws its own worlds from base_seed+version' divergence from ParallelExecutor is documented but is a real reproducibility difference worth surfacing at the loop level

**Why refuted:**

The finding's technical claims are accurate but it does not describe a defect. Verified facts: (1) exit_loop.py:417 draws gen_worlds via numpy; ParallelExecutor.generate passes the exact list through (parallel.py:151-153, enumerate(worlds) -> int(w) per task). (2) CppActorExecutor.generate uses only n_eps=len(worlds) (cpp_executor.py:104) and subprocesses the runner with --seed base_seed+version (cpp_executor.py:108); the runner redraws each episode's world via fold_seed(cfg.seed, idx) uniformly from env.worlds() (runner.cpp:143-146). So the two paths DO play different episodes at identical seeds. That much is true.

But this is by-design behavior that is already documented in THREE places and matches the codebase's standing posture, not a defect introduced by this change: (a) the generate() docstring states it plainly (cpp_executor.py:90-93: 'worlds is used only for its COUNT — the actor's reproducibility rides its seed, not the parent's world list'); (b) the runner's own comment names it explicitly (runner.cpp:122-124: 'the RNGs differ across the language boundary by design, so parity is the ADR-0012 P6 behavioral bar, not byte-identity'); (c) ADR-0012's two-tier bar (adr-synopsis.md:279, 'behavioral float32-equivalence, not byte-identity, for ML') is exactly the framework under which a cross-language RNG/world-draw difference is the EXPECTED, accepted state.

The finding refutes itself: it concedes 'This IS documented in the generate() docstring and is a defensible choice... so it is honest — not a silent trap,' and its only suggested fix is adding a clause to a launch banner. There is no correctness consequence — both paths draw the same COUNT of worlds uniformly from the same env.worlds with the same value-target semantics; only the specific seeds differ, which is the cross-language RNG difference ADR-0012 already sanctions. The parallel.py:138 reproducibility guarantee ('reproducible regardless of worker count') is a WITHIN-Python serial≈parallel invariant; it was never a Python-vs-C++ invariant, so 'A/B not apples-to-apples' compares against a bar that never applied to the C++ path. The banner already surfaces the one genuinely surprising semantic (pure-MC value target / Part-B guard); the world-draw is documented at the executor and runner where a reader reasoning about C++-actor reproducibility will be. This is a stylistic documentation nice-to-have, explicitly conceded as honest, not a defect in this working-tree code.

### [correctness-failure-honesty] Part-B guard treats lam_blend>1.0 as pure-MC silently; runs inside generate() after the redis connection is already open

**Why refuted:**

Both observations are factually true but neither is a genuine defect in this code.

(1) "lam_blend>1.0 silently treated as pure-MC." The guard at cpp_executor.py:96 is `(n_step is not None) or (lam_blend is not None and lam_blend < 1.0)`. It is not the validation boundary for the td_lambda RANGE — `check_invariants` (schema.py:280-281: `if not (0.0 <= v.td_lambda <= 1.0): raise RegistryDecodeError`) is. That invariant runs on EVERY config path that can feed the loop: from_argparse (registry.py:471, launch-time CLI), set_fields (registry.py:377, live HOT override), and decode (registry.py:187). In exit_loop, cfg0 is built at line 239 via from_argparse BEFORE the executor is even constructed (line 350), and the lam_blend passed to generate() (line 426) is `snap.cfg.value.td_lambda` (line 386) — a value that already passed check_invariants. So a `--td-lambda 1.5` dies with RegistryDecodeError at line 239, before any redis connection opens, and can never reach generate() as 1.5. The finding itself concedes "not reachable as a valid config." For 1.5 to reach generate() one must bypass the entire schema layer with a direct in-code call (as the test does with 0.5) — at which point the guard is not the relevant defense; the schema is. The predicate `< 1.0` correctly and precisely names the only meaningful TD(λ) blend region, matching the module docstring and schema; `!= 1.0` would conflate "out-of-range" (schema's job) with "unsupported-but-valid blend" (the guard's job), a clarity regression.

(2) "Guard runs in generate() after __init__ opened redis." True (redis opens at line 83, guard at line 96) but correct by design. lam_blend/n_step are per-GENERATE contract arguments, not construction args — the signature is identical to ParallelExecutor.generate (parallel.py:132-133), and exit_loop reads them live per-iteration from the HOT registry snapshot (lines 385-386), so a blend could toggle mid-run. __init__ cannot validate a value it never receives, so the guard structurally belongs in generate(). Moreover the reference ParallelExecutor also opens its redis connection in __init__ (parallel.py:116) — CppActorExecutor mirrors it exactly (structural parity, not a leak), and the connection is released by close()/__exit__ (lines 166-177). The "must have redis up to reach the guard" property is shared with the executor it is a drop-in for.

The finding's own hedges ("not reachable as a valid config", "Low priority given schema validation upstream") concede the point. Nit at most, no behavioral defect.

### [scope-docs-hack] evaluate() leaves the ~200-episode eval fan-out fully serial in-process Python — slower than the parallel baseline it supersedes

**Why refuted:**

The finding's FACTUAL premise checks out, but its classification as a defect does not. Verified facts: CppActorExecutor.evaluate() (cpp_executor.py:159-163) is a plain serial `for w in worlds:` loop running the Python GumbelPolicy in-process — no pool, no C++. ParallelExecutor.evaluate() (parallel.py:158-180) does fan eval across the core-pinned pool via self.pool.map(...). Defaults are E=300 (exit_loop.py:533) and eval-n=200 (exit_loop.py:546). So switching to --cpp-runner does run a ~200-episode eval serially where the Part-A parallel path used 4 cores. All true.

But this is not a defect in THIS code, for three reasons:

1) The finding concedes its own case. It states the design is "a defensible reading of 'swap into GENERATION'" and "Acceptable as a deliberate scope choice." The only residual charge is documentation tone — that the docstring "presents [eval] only as a clean division of labour, not as a perf regression." There is no contract mismatch (evaluate's signature and (totR, totT, ets) return exactly match ParallelExecutor), no correctness bug, no resource leak, no dishonest scope. The greedy-rate quantity eval measures is unaffected.

2) The "dishonest docstring" charge fails on inspection. The docstring is explicit and accurate: cpp_executor.py:18-20 says eval "runs ... IN-PROCESS (Python) ... No subprocess, no redis: a pure-Python search," and lines 149-151 repeat "run IN-PROCESS ... over the held-out worlds." A grep for parallel/core/fast/slow/perf shows the module makes ZERO perf or parallelism claim about eval. Per ADR-0009, substantiation attaches to perf/equivalence claims that are MADE; an omitted comparative caveat against a path eval no longer uses is not a false claim. The author was honest about what eval does (serial, in-process, the for-loop is right there in the method); they merely declined to editorialize a comparison.

3) The reviewer's own "perf regression vs the parallel baseline" framing is itself slanted. The C++ path does not introduce a NEW serial eval — the in-process serial baseline (--workers 0) already runs eval serially (exit_loop.py:473-480). The C++ path declines to PARALLELIZE eval, vs only the --workers>0 path, on only the eval slice, while moving the dominant cost (E=300 full-search generation) to native C++. Calling the net change a perf regression overstates it.

Net: the underlying observation is accurate and worth a one-line docstring note at most, but it lands as nit-level additive polish the finding itself labels "acceptable," not a genuine defect. Defaulting to is_real=false per the self-limiting language and the absence of any false claim, broken contract, or leak.

### [scope-docs-hack] evaluate() docstring claims it mirrors the serial path's eval seed, but uses base_seed+10_000 instead of cfg.eval.eval_seed

**Why refuted:**

The finding's factual substrate checks out: cpp_executor.py:82 sets self._eval_seed = base_seed + 10_000, where base_seed = master_seed = cfg0.loop.seed (exit_loop.py:315,351), which defaults to 7 (schema.py:202, RESTART). The serial path seeds its eval rng from eval_seed = snap.cfg.eval.eval_seed (exit_loop.py:390,476), which defaults to 12345 (schema.py:210, HOT). These are genuinely distinct, independently-configurable seed sources, so the within-episode sensing draws differ between the C++-swap eval and the serial eval. The held-out WORLDS, however, ARE drawn from cfg.eval.eval_seed in exit_loop (line 419, eval_rng = default_rng(eval_seed)) and passed into evaluate() as `worlds`, so the comparable held-out quantity is shared.

But the alleged defect — that the docstring "mirroring the serial path" (line 151) is inaccurate — does not hold up against the actual text. The sentence is: "The eval randomness is fixed across iterations (only the net changes), mirroring the serial path." Grammatically, "mirroring the serial path" modifies the asserted property — "fixed across iterations (only the net changes)" — NOT the seed value. That structural property is literally TRUE of both paths: the serial path creates a fresh default_rng(eval_seed) each iteration from a constant seed (only the net's decisions change), and the C++ path creates a fresh default_rng(base_seed+10_000) each iteration from a constant seed (only the net changes). The docstring nowhere claims "same seed as serial," "uses cfg.eval.eval_seed," or "eval numbers are comparable across executors." The reviewer over-reads "mirroring" as a claim about the seed SOURCE; the text only asserts the fixed-across-iterations STRUCTURE, which is accurate.

The reviewer also concedes the two facts that defang the concern: (1) "the executor DOES mirror the serial path's structural property," and (2) "the parallel path already diverges from serial here, so cross-executor eval comparability was never exact." The shipped parallel executor folds `version` into its eval stream (worker.py:122-140 _fold_seed includes version), so it is NEITHER fixed-across-iterations NOR seeded from eval_seed. The C++ path is therefore no more divergent from serial than the already-production parallel path — and is arguably closer, since it at least preserves the fixed-across-iterations property the parallel path loses. There is no broken invariant, no behavioral bug (eval is deterministic, uses the shared held-out worlds, fails loud nowhere relevant), and no scope/honesty violation. At most this is a docstring-precision nit where the literal statement is defensible; it does not rise to a genuine defect. Default to not-a-defect per the speculative/already-defensible bar.

### [scope-docs-hack] Orientation surfaces are silent on the actor-swap; the live handoff names a DIFFERENT top priority (the ZmqNetClient consult)

**Why refuted:**

The finding's factual claims all verify correct, but it does not identify a defect in this working-tree code, and its one substantive recommendation is already discharged by the change itself.

Verified facts (all correct):
- docs/STATUS.md is dated 2026-06-13 ("# chocofarm — status & contention (2026-06-13)") and is about the OR problem/contention; the swap does not make it inaccurate. The working tree touches no orientation docs (git diff --stat over docs/ is empty).
- The handoff docs/handoff-2026-06-16-zmq-async-gumbel.md (read end to end) declares main=51b13b9, names the ZmqNetClient consult as THE top priority (TL;DR #1), lists the 3->2->1 sequence (lines 79-82), and mentions neither cpp_actor_loop.py nor any exit_loop swap. `git merge-base --is-ancestor 51b13b9 6d8ab98` returns YES and `git log --diff-filter=A` shows cpp_actor_loop.py first appears at 6d8ab98 ("goal 2 complete"), so the handoff genuinely predates the whole actor-loop arc.
- No ADR Revisit-when fires. I read every Revisit-when section. The only plausible candidate is ADR-0012 #5 ("The async actor-learner restructure lands — scaling-and-cpp-seam.md Shape C ... relaxes the aggregate bit-determinism P6"). I read Shape C (scaling-and-cpp-seam.md:53-62): it is the sync->async restructure — continuously-running decoupled learner+actors over a streaming buffer that "relaxes the parallel≈serial bit-determinism." The swap under review is the OPPOSITE: CppActorExecutor is a drop-in for ParallelExecutor INSIDE the unchanged synchronous `for it: generate->train->eval->checkpoint` loop (cpp_executor.py docstring: "ZERO change to the loop's orchestration"); it does not decouple learner from actor and does not relax aggregate determinism. So #5 does not fire, confirming the finding's own claim.
- The swap reuses the documented ADR-0012 P7 weight/result seam (publish_weights at the `az:w:<run>:<phase>:<version>` channel; result_keys/RESULT_DTYPE decode), so it does not contradict ADR-0012 P7.

Why not-a-defect:
1. The finding explicitly grades itself a nit and identifies no code defect — it is an orientation/process observation about a commit message and a handoff that do not yet exist (the change is uncommitted).
2. Its core recommendation — "a one-line note that CppActorExecutor is the production exit_loop home and cpp_actor_loop.py is now the minimal demo" — is ALREADY PRESENT in the change's own docstrings at the SSOT (the code). The working-tree diff to cpp_actor_loop.py rewrites its docstring to: "For the FULL ExIt run with the C++ actor, prefer the SWAP: chocofarm/az/cpp_executor.CppActorExecutor injects the C++ Gumbel actor into exit_loop's GENERATION ... This standalone loop remains the minimal, dependency-light demonstration." And cpp_executor.py:9-11 states it "SUPERSEDES the minimal standalone cpp_actor_loop.py ... for the full ExIt run." That is exactly the production-home-vs-demo note the suggested fix asks for, placed where the codebase's own discipline wants it (code, not immutable prose).
3. The residual asks (a commit-message TL;DR line; confirm with the maintainer whether to proceed ahead of the handoff's named priority) are reasonable courtesies for the eventual commit, but they are not defects in this working-tree code; they cannot even exist for an uncommitted tree, and the priority-ordering question is the user's call (the user explicitly requested this swap, which the finding itself concedes is in scope).

---

## Maintainer disposition (APPENDED — not part of the audit above)

> Per the hack-rationalization-detector's verbatim-return rule, everything above is
> the auditors' output unaltered. This section is the maintainer's response, appended.
> Recorded by Claude Opus 4.8, who commissioned and acted on the review.

All seven confirmed findings were fixed before commit:

- **explore_plies silently dropped** (contract-fidelity major + scope-docs major, the same finding from two lenses — the most important catch; the implementer had missed it) — FIXED with the reviewer's option (a): `generate()` now FAILS LOUD on `explore_plies>0`, mirroring the Part-B guard; the module docstring's known-deferred list gained an `explore_plies` entry; the `exit_loop --cpp-runner` help states it requires `--explore-plies 0`; the swap-turns test now passes `--explore-plies 0`, the Part-B test isolates with `explore_plies=0`, and a new `test_cpp_actor_executor_explore_plies_fails_loud` asserts the refusal. Option (b) — threading a temperature>0 executed-action sample onto the C++ `Decision`/runner — is the documented follow-up (parallel to Part-B's `v_mix` path).
- **silent empty-buffer** (correctness major) — FIXED: `generate()` parses the runner's `wrote N episode(s)` from stderr and reconciles it against the non-empty episodes `_read_records` actually read (now returned as a count), raising loudly on a mismatch (the LRU-eviction window); a floor check raises on `[] with n_eps>0` when the count can't be parsed.
- **redis connection unhardened** (correctness minor) — FIXED: the executor now builds its connection via `transport.connect()` (bounded socket timeouts + fail-loud ping at construction), matching `ParallelExecutor`; the bare `import redis` / `transport_redis_params` are dropped.
- **evaluate() "mirroring the serial path" overclaim** (correctness minor) — FIXED: the docstring now states the eval uses an independent construction-fixed seed (a valid estimate, NOT bit-comparable to a serial/eval_seed eval), and notes it actually matches the worker-pool path (folds base_seed).
- **self.cores "self-schedules" comment** (scope-docs minor) — FIXED: reworded to "the runner is one subprocess running episodes serially; no per-worker core pin."
- **dead `lam_blend is not None` arm** (scope-docs nit) — FIXED: dropped to `if (n_step is not None) or (lam_blend < 1.0):`, matching the three sibling call sites.

Verdict on the SWAP itself: `narrower-but-justified` (the reviewer's overall verdict), with the one undischarged-hack (explore_plies) now discharged honestly. The full default suite (196 passed incl. mypy-strict) and the three opt-in cpp tests pass after the fixes.
