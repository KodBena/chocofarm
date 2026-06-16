# Fiber-leaf SSOT dedup — adversarial review (report, verbatim)

> The complete output of the multi-agent adversarial review commissioned for the
> fiber-leaf SSOT dedup (the `refactor/fiber-leaf-ssot` change, merged to `main` as
> `74d5749`). Reproduced **verbatim** per the verbatim-record discipline (ADR-0005)
> and the hack-rationalization-detector rule that the audit artifact is the
> deliverable, surfaced to the maintainer **unmediated** — no summarizing, softening,
> or omitting a line. The commission (the exact lens prompts) is in
> `fiber-leaf-dedup-review-commission.md`. A maintainer-disposition appendix at the
> very end records what was acted on; it is **appended**, and alters nothing above it.
>
> Review shape: 4 lenses (behavior-equivalence, ODR/lifetime/fiber-safety,
> scope/hack-rationalization, docs/IWYU); each raw finding adversarially
> verified (refuted-or-confirmed) by an independent agent. Tally (verbatim from the
> run log): **"reviewed 4 lenses; 9 raw findings, 6 confirmed, 3 refuted"** — 13 agents total.

---

## Lens summaries (verbatim)

### Lens: behavior-equivalence

The refactor is faithful at the level the deterministic gate exercises: CyclicGumbelSource is a byte-for-byte rename of the three inlined ScriptedGumbelSource copies (same sample_world->bw[0], same table_[(idx_++) % size] cycling, same idx_=0 init), FiberLeafChannel maps one-to-one onto the old YieldCtx (caller/features(was leaf_features)/value(was leaf_value)/at_leaf, same defaults), the canonical YieldingNetEvaluator body is identical, the fixedsize_stack is still 512*1024 everywhere, member-init order is sound (ch before ynet before policy before src; the init-list only takes addresses), and the proof still seeds BOTH the direct CyclicGumbelSource(gtable) and the fibered TreeState's internal CyclicGumbelSource(gtable copy) from a fresh idx_=0 identical table — so direct and fibered draw byte-identical Gumbel streams, and the start()/resume_with() loop is leaf-count- and termination-equivalent to the old run_fibered loop (each parked leaf predicted once, fed back, counted; same at_leaf terminator). I found no semantic divergence in current behavior. The one substantive thing PASSING does not prove is a lifetime-contract shift the refactor introduced: by moving the drive loop out of run_fibered (one frame that owned everything) into TreeState whose start() returns while the fiber is still suspended, the fiber's [this,&loc,&bw,&coll,lam] lambda now holds references to start()'s reference-parameters that the search reads across leaf-yield boundaries (after start() returns) — a caller-must-keep-alive contract that all three current callers happen to satisfy but that is undocumented in fiber_tree.hpp. ODR/no-move discipline is otherwise clean.

### Lens: odr-lifetime-fiber

The refactor is ODR-clean and lifetime-correct for all three current callers. The two new headers (cyclic_gumbel.hpp, fiber_tree.hpp) and the pre-existing fiber_leaf.hpp contain only class/struct definitions whose every member body is in-class (hence implicitly inline) and zero free functions, so there is no non-inline ODR hazard; the single header text per type is included under #pragma once and is identical across all including TUs. The TreeState member init order (ch -> ynet -> policy -> src) matches the dependency chain exactly, so there is no use-before-init: ynet binds ch after ch is constructed, policy binds ynet+env after both exist. Destruction is safe (WorldSource carries a virtual dtor, so CyclicGumbelSource final destroys correctly). Lifetime holds in every caller as written: fiber_proto's stack-local TreeState ts is never moved/copied/returned; both benches hold TreeState behind unique_ptr so vector growth relocates the pointer, never the object, and no TreeState is moved after start(); and in all three the loc/bw/coll arguments to start() are named lvalues owned by main (the pool bench captures them by reference into the worker lambda, and all threads join before those locals die), so the fiber's captured references stay valid across every resume_with. Behavior is preserved vs the pre-refactor inlined copies (the shared types are line-for-line the same logic; only includes were rebalanced — boost/cmath now correctly live behind fiber_tree.hpp/DetNet respectively). The lens is therefore clean on the asked hazards; the two findings below are latent fragilities that the passing deterministic gate cannot exercise, not present-tense bugs.

### Lens: scope-hack-rationalization

Run out of frame, the code dedup holds up as the correct general invariant rather than a rationalized hack. GENERAL FIX: every cross-file-duplicated fiber-leaf contract (channel, yielding evaluator, scripted source, per-tree fixture) gets exactly ONE header home that every driver derives from, and only token-identical contracts are unified. PATCH SHIPPED: CyclicGumbelSource (3 inline copies -> cyclic_gumbel.hpp), TreeState (2 inline copies -> fiber_tree.hpp), and the inline YieldCtx/YieldingNetEvaluator deleted from fiber_proto.cpp + wire_parallel_bench.cpp in favor of the canonical FiberLeafChannel/YieldingNetEvaluator in fiber_leaf.hpp; fiber_proto's bespoke run_fibered() driver rewritten to drive the shared TreeState; field access repointed (leaf_features/leaf_value -> ch.features/ch.value). WRITER/COPY DELTA is CLEAN: after the change there is exactly one chocofarm::TreeState, one CyclicGumbelSource, one FiberLeafChannel, one YieldingNetEvaluator anywhere under cpp/; no inline copy survives that should have been unified. gumbel_dump.cpp::ScriptedGumbelSource survives and CORRECTLY so — its 2-arg ctor (gumbels, world_idxs) scripts world selection with non-negative modulo and asserts on empty tables, a genuinely different contract; unifying it would have been an ADR-0008 fuzzy-match / ADR-0003 premature-abstraction error, and cyclic_gumbel.hpp's docstring explicitly documents WHY it is left distinct. Behavior is preserved: both fiber_proto runs still feed the SAME gtable to byte-identical CyclicGumbelSource draws; the rename was applied completely (no stale leaf_features/leaf_value reads); include hygiene is correct (cmath kept where std::sin is used, dropped where it is not). The EXPANSION beyond the literal ask (also extracting CyclicGumbelSource + TreeState, also rewriting fiber_proto's driver) is justified by P1 SSOT and is the correct invariant, not over-scoping. The one real scope concern is OUTSIDE the code: the working tree also carries an unrelated 655-line forward-looking design doc (docs/design/cpp-search-runtime.md) that designs a future SearchRuntime seam (not the dedup) and contains a dangling cross-reference to a nonexistent, mis-filed consult. VERDICT: general (code dedup); the doc is a separate over-scope/discipline flag. Plus two minor header-vs-use nits.

### Lens: docs-iwyu

test

---

## Confirmed findings (verbatim)

### [behavior-equivalence · minor] TreeState::start captures &loc/&bw/&coll read after start() returns — undocumented caller lifetime contract the original run_fibered made structurally impossible to violate

**Location:** `cpp/include/chocofarm/fiber_tree.hpp:49-57`

**Reasoning:**

Verified against the working tree. The mechanism is real: TreeState::start (fiber_tree.hpp:49-57) builds a fiber whose entry lambda captures [this, &loc, &bw, &coll, lam]. loc/bw/coll bind to start()'s const-reference parameters, which alias the caller's objects; lam is captured by value (a plain double copy) and is genuinely safe. The captured references are dereferenced long after start() returns: start() only advances to the first parked leaf; the drive loop calling resume_with() (fiber_tree.hpp:63-67) lives in the caller. GumbelAZPolicy stores no belief (members are cfg_/net_/env_/fb_/n_slots_/term_slot_, gumbel.cpp:233), so run_search (gumbel.cpp:527, takes loc/bw/collected by const-ref) re-reads them on every leaf: run_search -> evaluate (fb_.build(loc.pt, bw, collected), gumbel.cpp:239) -> net_.predict(feat), where YieldingNetEvaluator::predict (fiber_leaf.hpp) is the fiber suspend point. So loc/bw/coll are read on every leaf across every resume_with(), i.e. the caller must keep them alive for the whole drive loop, not just across start().

The 'structurally impossible before' claim is correct about what it cites but narrower than its framing implies. git show HEAD:cpp/src/fiber_proto.cpp confirms the pre-refactor fiber_proto had run_fibered owning the entire while(ctx.at_leaf) drive loop in ONE frame, so its caller references could not dangle. But the diff shows the OLD wire_parallel_bench and wire_pool_bench already had the split start()/resume_with() TreeState with this exact lifetime contract — so the refactor newly imposes the split-frame contract only on the fiber_proto path, while consolidating an identical pre-existing contract for the two benches. The hazard is therefore not brand-new to the codebase, only newly centered in a reusable shared header.

Behavior is preserved today (the finding concedes this): fiber_proto loc/bw/collected are main locals with the loop in main (fiber_proto.cpp:117-139); wire_parallel the same (wire_parallel_bench.cpp:119-121); wire_pool loc/bw/coll are main locals (wire_pool_bench.cpp:155-157) captured [&] into the worker lambda whose threads are joined before main returns. The deterministic gate cannot reveal otherwise.

The documentation gap is real: fiber_tree.hpp already documents the this/no-move lifetime contract (lines 14-16) but the start() comment (line 48) and purpose block say nothing about loc/bw/coll having to outlive the fiber. Since the header's stated purpose (lines 9-12) is to be 'the single primitive THREE drivers multiplex/drive,' it explicitly invites future callers — and a future caller passing a temporary belief (e.g. start(loc, env.worlds(), {}, lam): env.worlds() returns a temporary vector bound to the const-ref param, captured by reference, then dereferenced after the temporary is destroyed) gets silent UB no current config exercises. This is consistent with ADR-0002 (fail loudly / make implicit preconditions explicit). Not a current defect; a genuine doc-discipline gap on a newly-shared primitive with a UB failure mode. Severity minor is well-calibrated.

**Fix:**

Discharge the implicit precondition with a one-line header note, matching the existing this/no-move note style. In fiber_tree.hpp, extend the purpose block (near lines 14-16) and the start() doc-comment (line 48) to state: the caller must keep loc/bw/coll valid until `running` becomes false — the fiber's entry lambda captures them by reference and the search re-reads them on every leaf across all resume_with() calls, not only during start() (lam is captured by value, so it is exempt). Caution future callers specifically against passing a temporary belief (e.g. start(loc, env.worlds(), {}, lam)). Optionally harden by snapshotting the belief into TreeState members (copy bw/coll into the struct, capture only `this`) so the fiber owns its inputs the way the single-frame run_fibered implicitly did; if kept by-reference, the header note is the minimum to discharge the precondition the original made unnecessary.

### [odr-lifetime-fiber · minor] TreeState is implicitly movable despite a "never move after start()" contract — and against the codebase's own delete-copy/move precedent

**Location:** `cpp/include/chocofarm/fiber_tree.hpp:36-46`

**Reasoning:**

Verified against the actual working-tree code and confirmed empirically with a compile probe.

SELF-REFERENCE CHAIN (all confirmed):
- fiber_leaf.hpp:53 — YieldingNetEvaluator holds `FiberLeafChannel& ch_`; TreeState::ynet binds it to TreeState::ch (fiber_tree.hpp:46 `ynet(ch)`).
- gumbel.hpp:233-234 — GumbelAZPolicy holds `const NetEvaluator& net_` and `const Environment& env_`; TreeState::policy binds net_ to TreeState::ynet (fiber_tree.hpp:46 `policy(cfg, ynet, env)`).
- fiber_tree.hpp:52 — start()'s fiber entry lambda captures `this`.
So a relocated TreeState leaves ynet.ch_, policy.net_, and the fiber's captured `this` all pointing at the moved-from object — exactly what the header docstring (fiber_tree.hpp:14-16) names: "never move it after start(): the fiber's entry lambda captures `this`, so a move would dangle those captured references."

THE TYPE DOES NOT ENFORCE IT (confirmed by mechanism + probe):
- boost::context::fiber (fiber_fcontext.hpp:359-373) is move-only: move ctor/assign defined, copy ctor/assign `= delete`.
- TreeState declares only a parameterized ctor (fiber_tree.hpp:45) — no destructor, no copy, no move special members. Per the C++ special-member rules, the deleted-copy `fib` member makes TreeState's implicit COPY deleted, but the implicit MOVE constructor remains declared and non-deleted. ynet/policy carry reference members (their move-ASSIGN is deleted, but their move-CONSTRUCTOR — reference re-init — is valid), so the synthesized TreeState move-CONSTRUCTOR is valid.
- Compile probe (g++ -std=c++23, fsyntax-only, EXIT=0): `is_move_constructible_v<TreeState>` true, `is_copy_constructible_v<TreeState>` false, `is_move_assignable_v<TreeState>` false. So `auto t = std::move(other_tree)` after start() compiles, passes -Wall -Wextra, and silently dangles.

WHY THE GREEN GATE PROVES NOTHING HERE: all three current drivers avoid moving a started TreeState — fiber_proto.cpp:132 is a never-moved stack local; wire_parallel_bench.cpp:141/147 and wire_pool_bench.cpp:186/203 hold `vector<unique_ptr<TreeState>>` + make_unique, so only the unique_ptr ever moves, never the TreeState. Behavior-preservation and the deterministic PASS are therefore silent on a future fourth driver or a refactor of these.

PRECEDENT (confirmed, with a nuance the finding states correctly): RedisClient (transport.hpp:76-77) and ZmqNetClient (zmq_net_client.hpp:52-53) `= delete` copy but DEFINE move — they are non-copyable-yet-movable because their hand-written move transfers ownership safely (std::exchange of a raw handle). TreeState is strictly more fragile: its correct relocation semantics are "no move at all," since self-refs + a this-capturing fiber cannot be re-pointed. The finding's claim that it "warrants the stronger form" is sound — the convention is "declare relocation semantics explicitly on resource-holding types," and TreeState's correct semantics are delete-move.

This is a genuine latent defect introduced by this (new, untracked) file: the invariant lives only in a comment, and ADR-0002 (fail loudly) favors turning a latent dangle into a hard compile error. Severity "minor" is well-calibrated: real, but latent — no current call site is incorrect, and the remedy is cost-free.

One minor over-statement in the suggested fix: copy is ALREADY auto-deleted (probe-confirmed), so spelling out the copy `= delete` is documentation-only (harmless, matches precedent). The load-bearing part is deleting MOVE.

**Fix:**

In fiber_tree.hpp, immediately after the TreeState constructor (line 46), make the "never move" contract compile-enforced:

    TreeState(const TreeState&) = delete;             // already auto-deleted (fib is move-only); explicit = intent
    TreeState& operator=(const TreeState&) = delete;
    TreeState(TreeState&&) = delete;                  // self-refs (ynet→ch, policy→ynet) + this-capturing fiber: a move dangles
    TreeState& operator=(TreeState&&) = delete;

All three existing call sites are unaffected (fiber_proto.cpp uses a never-moved stack local; both benches hold vector<unique_ptr<TreeState>>, so only the unique_ptr moves). This turns any future `std::move`d-after-start() call site into a hard compile error rather than a silent dangle, matching the RedisClient/ZmqNetClient precedent of declaring relocation semantics explicitly — and honoring the header docstring's own stated invariant per ADR-0002.

### [odr-lifetime-fiber · minor] start() captures its reference parameters into the fiber — a temporary argument would dangle silently across resume_with

**Location:** `cpp/include/chocofarm/fiber_tree.hpp:49-57`

**Reasoning:**

The lifetime claim is correct and provable from the code. In fiber_tree.hpp:52 the fiber entry lambda captures `&loc, &bw, &coll`, and its body (line 54) forwards them into policy.run_search(loc, bw, coll, lam, src). run_search (gumbel.cpp:527-528) takes all three by const& and reads them across the leaf-yield suspension points — evaluate() at line 543 and especially sequential_halving() at line 581, inside which the parking predict() calls live. Because run_search executes incrementally across many resume_with() calls (fiber_tree.hpp:63-67, each resuming the suspended search), the referents of loc/bw/coll must stay alive until the FINAL resume_with, not merely until start() returns. A temporary bound to these const& params (e.g. ts.start(Loc{env.entry_point()}, env.worlds(), {}, lam)) would be destroyed at the end of the start() full-expression, so every later resume_with would read freed memory — and it would still compile clean and likely pass a single-tree smoke run by allocator reuse, exactly the class of hazard a green gate cannot catch.

The "no present bug" half also checks out: all three callers pass long-lived lvalues. fiber_proto.cpp:117-119 are main locals driven by the loop at 132-139 in the same scope; wire_parallel_bench drives its trees within main; wire_pool_bench.cpp:155-157 define loc/bw/coll as main locals, the worker lambda captures them by-ref ([&]), and the threads are join()ed at line 242 before main returns, so they outlive every resume_with in every worker. So this is a latent API-surface hazard, not a live defect.

It is not a false positive or already-handled: the docstring (fiber_tree.hpp:14-16) explicitly documents the SIBLING hazard — the no-move rule ("never move it after start(): the fiber's entry lambda captures `this`, so a move would dangle those captured references") — but says nothing about the by-reference start() args, which are the identical unenforced-lifetime class. The gap is real and asymmetric. Severity is correctly minor: the API is currently used correctly everywhere, so this is a documentation-discipline observation, not a correctness break. The finding's secondary suggestion (deleted rvalue overloads) is rightly relegated to "ideally" — and is in fact over-engineering here: a `void start(..., std::set<int>&&, ...) = delete;` overload would also reject the idiomatic, safe-looking empty-set literal `start(loc, bw, {}, lam)`, so the one-line doc note is the proportionate fix and matches the file's existing posture of documenting (not mechanically enforcing) the capture-lifetime contract.

**Fix:**

Add a one-line caller-contract note to the TreeState docstring in cpp/include/chocofarm/fiber_tree.hpp, mirroring the existing no-move warning, e.g. after line 16: "start() captures loc/bw/coll BY REFERENCE into the fiber (run_search reads them across every leaf-yield), so they must outlive the LAST resume_with(), not just start() — pass named lvalues, never temporaries." Do not add the rvalue-ref =delete overloads as a blanket fix: a deleted std::set<int>&& overload would also reject the idiomatic safe empty-set literal start(loc, bw, {}, lam). The doc note is proportionate to the minor severity and consistent with how the file already documents the no-move capture-lifetime contract.

### [scope-hack-rationalization · minor] Unrelated 655-line design doc rides along in the same working tree as the dedup, and cites a consult that does not exist at a path that violates the consult-directory convention

**Location:** `docs/design/cpp-search-runtime.md:9 (and:13)`

**Reasoning:**

The dangling-reference core of the finding is genuine and verifiable. docs/design/cpp-search-runtime.md:9 cites `docs/notes/consult/opus-consult-2026-06-16-zmq-net-client-blocking-req.md` as a "load-bearing premise" (the "matched-pair finding"; leaned on again at lines 39, 48, 298, 299, 615). I verified that file exists NOWHERE in the tree (`find ... -name 'opus-consult-*'` and a search for `zmq-net-client-blocking-req` both empty), and that the directory `docs/notes/consult/` does not exist. The repo's consult convention, codified in ADR-0005 Rule 2 ("Consult records ... live under docs/consults/, as consult-NNN-*") and applied uniformly everywhere else (every other consult reference repo-wide uses docs/consults/consult-NNN-*), is violated on both the directory and the naming axis. This is exactly the dangling/misfiled cross-reference ADR-0005 Rule 2/3 and CLAUDE.md's "dangling consult-002 §4" caution name: a cited document a reader cannot resolve. The doc is untracked (not in HEAD), so it is live working-tree state, not a frozen point-in-time record exempt from repointing.

But the finding's SCOPE-HACK FRAMING is refuted, and two sub-claims are wrong. (1) The doc does not "ride along with the dedup." It is the design record for the already-LANDED SearchRuntime work: it is referenced by SIX files committed in HEAD's ancestry — cpp/include/chocofarm/search_runtime.hpp, cpp/src/serial_runtime_check.cpp, cpp/src/wire_bench.cpp, cpp/src/dealer_probe.cpp (landed in commits 6831601/c7a5c40/f00306e), plus two tracked docs/notes files. Its mtime (2026-06-16 05:43) precedes the dedup edits (10:43–10:48) by ~5 hours. It belongs to a different, already-committed line of work and merely happens to be a still-untracked file coexisting in the tree. So "split it from the dedup commit" is moot: per CLAUDE.md the committer stages by explicit path (never git add -A), so this doc would never enter the dedup commit regardless. (2) The finding claims the doc "name-drops FiberLeafChannel once" — a case-insensitive grep finds ZERO mentions of FiberLeafChannel/fiber_leaf. (3) Correctly, the doc mentions none of TreeState/fiber_tree/cyclic_gumbel — but that is expected, since it is not about the dedup.

Net: the observable code defect (unresolvable, convention-violating consult citation in live working-tree doc) is real at minor severity; the "scope creep on the dedup" rationale that generated it is the wrong lens for this artifact (the doc is not part of the dedup change). The correct remedy is the cross-ref repair, not a commit-split.

**Fix:**

In docs/design/cpp-search-runtime.md, repair the dangling consult citation per ADR-0005 Rule 3 (a cited document must exist and resolve). Either (a) land the referenced consult under the convention directory as docs/consults/consult-NNN-... before relying on it, then repoint line 9 (and the dependent mentions at 39/48/298/299/615) to that path; or (b) if the "matched-pair finding" actually lives in an existing record, repoint to its real path; or (c) if no such consult exists, remove the citation and inline the load-bearing premise as the doc's own assertion. Do not file it under docs/notes/consult/ — that directory is not the convention. Separately, drop the dedup-coupling concern: this untracked doc predates the dedup and belongs to the already-committed SearchRuntime work, so no commit-split is needed; just stage the dedup by explicit path (never git add -A) so the doc is not swept into the dedup commit.

### [scope-hack-rationalization · nit] wire_pool_bench.cpp uses chocofarm::NetPrediction directly but no longer includes net_evaluator.hpp — it now relies on a transitive include through fiber_tree.hpp

**Location:** `cpp/src/wire_pool_bench.cpp:221`

**Reasoning:**

The factual claims check out. wire_pool_bench.cpp:221 directly constructs `chocofarm::NetPrediction pred;` (verified, working tree). The include block (lines 57-64) includes `chocofarm/fiber_tree.hpp` but no `chocofarm/net_evaluator.hpp`. net_evaluator.hpp is the defining header (`struct NetPrediction` at net_evaluator.hpp:40). Inclusion is purely transitive: fiber_tree.hpp:32 includes fiber_leaf.hpp, which (fiber_leaf.hpp:22) includes net_evaluator.hpp. wire_parallel_bench.cpp does keep an explicit `#include "chocofarm/net_evaluator.hpp"` (line 49) while also naming NetPrediction (line 191), so the two benches are indeed inconsistent in IWYU posture. So far the finding is accurate.

However, two qualifications keep this firmly at nit and argue it is NOT a defect introduced by THIS refactor: (1) The gap is PRE-EXISTING, not introduced by the change under review. `git show HEAD:cpp/src/wire_pool_bench.cpp` shows the pre-refactor file also did NOT explicitly include net_evaluator.hpp — it included fiber_leaf.hpp (HEAD line 61) and relied on the same fiber_leaf.hpp -> net_evaluator.hpp transitivity to get NetPrediction (named at HEAD lines 149, 267). The refactor swapped fiber_leaf.hpp for fiber_tree.hpp, lengthening the transitive chain by exactly one hop but not introducing the reliance-on-transitivity; that reliance was already there and unchanged in kind. Likewise the cross-bench inconsistency pre-dates this diff: HEAD's wire_parallel_bench already had the explicit include and HEAD's wire_pool_bench already lacked it. (2) There is zero correctness, ODR, lifetime, or behavior hazard — the TU compiles and is correct; the only failure mode the finding posits ("if fiber_tree.hpp's includes are ever pruned") is hypothetical and remote, since fiber_tree.hpp structurally MUST include net_evaluator.hpp (its TreeState::resume_with takes `const NetPrediction&` at fiber_tree.hpp:63 and the YieldingNetEvaluator member needs it), so the type cannot realistically be pruned out of the closure. The finding correctly self-classifies as a nit. It is a legitimate, accurately-described IWYU/consistency observation worth a trivial one-line fix, but it is a stylistic non-issue rather than a defect this refactor introduced.

**Fix:**

Optional IWYU/consistency tidy (not required for correctness): add `#include "chocofarm/net_evaluator.hpp"` to wire_pool_bench.cpp's include block (lines 57-64), since the TU directly names chocofarm::NetPrediction at line 221. This matches wire_parallel_bench.cpp:49. Note this gap pre-existed the refactor (HEAD's wire_pool_bench already lacked the explicit include and got NetPrediction transitively via fiber_leaf.hpp), so it is a pre-existing-cleanup opportunity, not a regression from this change.

### [docs-iwyu · minor] fiber_tree memory include is gratuitous

**Location:** `cpp/include/chocofarm/fiber_tree.hpp:23`

**Reasoning:**

The header at cpp/include/chocofarm/fiber_tree.hpp:23 includes <memory>, but nothing in the header's own code uses a <memory> symbol. Grep for unique_ptr|shared_ptr|weak_ptr|make_unique|make_shared|addressof|std::allocator over the file returns only the docstring comment at line 14 ("e.g. behind a unique_ptr") and std::allocator_arg at line 51. std::allocator_arg is declared in <utility> (already included at line 25), NOT in <memory>. The header's other std symbols are all otherwise covered: std::move and std::allocator_arg from <utility> (25), std::set from <set> (24), std::vector from <vector> (26). So <memory> is genuinely unused by this header's code — the IWYU verdict holds.

I tried to refute it three ways and each refutation failed:
1. "Maybe a transitive consumer relies on it." The only includers of fiber_tree.hpp are the three driver TUs (grep confirmed; fiber_leaf.hpp:9 mentions it only in a comment, no #include). The two that actually hold TreeState behind a smart pointer — wire_parallel_bench.cpp (std::vector<std::unique_ptr<TreeState>> at :141, make_unique at :147) and wire_pool_bench.cpp (:186, :203) — already #include <memory> themselves (wire_parallel_bench.cpp:35, wire_pool_bench.cpp:48), so they are IWYU-correct independent of this header. fiber_proto.cpp constructs TreeState by value (:132) and uses no smart pointer. Removing the include breaks nothing.
2. "std::allocator_arg counts as memory usage." No — it is a <utility> facility, and <utility> is already present.
3. "The docstring documents unique_ptr ownership, so the include supports the contract." A module-docstring comment is not a use; strict IWYU (the reviewer's lens) places the include in the using TU, which is exactly where it already correctly lives.

This is a true but harmless dead include — no ODR/lifetime/behavior consequence, consistent with the established warning-clean build. Severity minor (arguably nit) is right.

**Fix:**

Delete line 23 (`#include <memory>`) from cpp/include/chocofarm/fiber_tree.hpp. The header uses no <memory> symbol (std::allocator_arg at line 51 comes from <utility>, already included). The two drivers that hold TreeState via std::unique_ptr/make_unique (wire_parallel_bench.cpp, wire_pool_bench.cpp) already include <memory> directly, so they are unaffected.

---

## Refuted findings (verbatim)

### [scope-hack-rationalization] fiber_tree.hpp's contract reads 'Heap-allocate it (e.g. behind a unique_ptr)' but the actual invariant is only 'never move after start()', and the proof's stack-local use diverges from the stated guidance

**Why refuted:**

All factual claims in the finding check out, but the conclusion (that the header is contradicted by the proof site) is refuted by the header's own wording.

The true invariant IS address stability after start(). Confirmed by reading the captures: fiber_tree.hpp:52 the fiber entry lambda captures `this`; fiber_leaf.hpp:43/53 YieldingNetEvaluator holds `FiberLeafChannel& ch_` bound to TreeState::ch (fiber_tree.hpp:46 `ynet(ch)`); GumbelAZPolicy holds the ynet (fiber_tree.hpp:46 `policy(cfg, ynet, env)`). A move after start() would dangle both the captured `this` and `ynet.ch_`. So heap allocation is sufficient, not necessary — exactly as the finding states.

But the header already encodes this correctly. fiber_tree.hpp:14-16 reads: "Heap-allocate it (e.g. behind a unique_ptr) and never move it after start(): the fiber's entry lambda captures `this`, so a move would dangle those captured references." Grammatically the binding conjunct is "never move it after start()"; "e.g. behind a unique_ptr" is parenthetically and explicitly marked as illustrative (the "e.g." does the exact disambiguation the finding asks for), and the trailing clause states the underlying reason. A reader is given the real invariant, the reason, and a signal that heap is one option — not a mandate.

The one divergent site is self-documenting: fiber_proto.cpp:130-131 carries an inline justification — "A stack local that never moves (the fiber captures `this`); one tree, so no heap/vector is needed" — tied to the identical reason the header gives. So the proof site does not read as contradicting the contract; it reads as a deliberately annotated second embodiment of the same invariant. The benches' vector<unique_ptr<TreeState>> form (wire_parallel_bench.cpp:141/147, wire_pool_bench.cpp:186/203) is the other.

There is no ODR/lifetime hazard (all uses satisfy address-stability), no behavior divergence, and no ADR-0006 header gap (fiber_tree.hpp carries the path+purpose+Public Domain header). What remains is a pure lead-with-the-invariant phrasing preference. The suggested reword is a reasonable polish, but the current text is technically correct and not misleading to a careful reader, and the finding itself concedes the code is safe. This is a stylistic non-issue, not a defect in this working-tree code.

### [docs-iwyu] stale P1 cleanup line in the session note

**Why refuted:**

The finding misapplies ADR-0005 Rule 8 and misjudges scope. Three facts refute it.

(1) The file is NOT part of the change under review. `git status` shows the working tree as: M cpp/include/chocofarm/fiber_leaf.hpp, M fiber_proto.cpp, M wire_parallel_bench.cpp, M wire_pool_bench.cpp, ?? cyclic_gumbel.hpp, ?? fiber_tree.hpp, ?? docs/design/cpp-search-runtime.md (+ .claude/). `docs/notes/cpp-search-runtime-benchmark-status-2026-06-16.md` is committed and clean (`git diff` on it is empty). The refactor does not touch it.

(2) It is a point-in-time session record, not a live cross-reference. The note's own header (lines 5-7) calls it an "Autonomous-session record" on branch `feat/cpp-search-runtime-serial` (off main = c85b97a) — a different branch from `main` where this working-tree refactor lives. Lines 38-39 are a forward-looking, CONDITIONAL TODO: "extract to a shared fiber-leaf header WHEN THE PRODUCTION POOL LANDS." That precondition has not fired — the production pool (FiberMuxRuntime/DEALER client) is, per the working-tree design doc docs/design/cpp-search-runtime.md (lines 5, 60, 304, 613), explicitly "not built," sequenced after the Gumbel port. The working tree extracted the primitives early as a pure SSOT cleanup, ahead of the condition the note names. So the note's statement is not even false against its own terms.

(3) ADR-0005 Rule 8 says the opposite of what the finding claims. Rule 8 ("Sibling revisions / dated corrections over silent edits of point-in-time records," ADR-0005 lines 147-160) triggers only "When an authoritative record is found wrong in a load-bearing way," and its load-bearing point is to PRESERVE point-in-time records and not silently rewrite them. A forward-looking TODO in a dated session log whose precondition has not fired is not "found wrong in a load-bearing way." ADR-0005's Neutral clause (lines 206-209) — "No retroactive rewrite required... incremental retrofit when files are touched for other reasons; no blanket rewrite pass" — and Rule 6 (lines 128-137, "status documents record slowly-aging decisions, never a live task queue"; cautionary instance: the 24-seconds-stale handoff) both say a stale forward-looking item in a point-in-time record is expected, not a defect to chase. CLAUDE.md echoes: "leave point-in-time records... un-retro-edited." Demanding a dated correction here is the retroactive-edit-of-an-unrelated-point-in-time-record posture ADR-0005 declines. No other doc carries the claim (grep finds no live referrer), so nothing resolves wrongly.

The verified code change itself (fiber_proto.cpp line 78, wire_parallel_bench.cpp lines 21/86-88 now cite fiber_leaf.hpp / cyclic_gumbel.hpp / fiber_tree.hpp instead of inlining) is correct and on-point; the note is simply not its concern.

### [docs-iwyu] wire_pool_bench gets NetPrediction transitively

**Why refuted:**

The finding's factual core checks out but it is not a defect in this working-tree change. Facts verified against the code:

- `cpp/src/wire_pool_bench.cpp:221` is `chocofarm::NetPrediction pred;` — yes, it names NetPrediction.
- `NetPrediction` is defined in `cpp/include/chocofarm/net_evaluator.hpp:40`.
- The working-tree include block (lines 58-64) does NOT include `net_evaluator.hpp` directly; it comes in transitively because `fiber_tree.hpp:32` includes `net_evaluator.hpp` (and `fiber_tree.hpp` also re-includes `fiber_leaf.hpp`, which itself includes `net_evaluator.hpp` at line 22).

Crucially, this is NOT introduced by the refactor — it is strictly pre-existing, exactly as the finding's own detail concedes ("pre existing and not a regression"). At HEAD (6d67c59), `wire_pool_bench.cpp` already used `NetPrediction` (its own inlined struct, lines 149/267) with NO direct `net_evaluator.hpp` include — it obtained the type transitively via the then-included `fiber_leaf.hpp` (which includes `net_evaluator.hpp`). `git show HEAD:cpp/src/wire_pool_bench.cpp | grep net_evaluator` returns nothing. The diff merely swapped the directly-included header `fiber_leaf.hpp` → `fiber_tree.hpp`; both transitively supply `net_evaluator.hpp`, so the transitive-dependence situation is byte-for-byte unchanged in kind.

There is no correctness, ODR, or lifetime hazard here — NetPrediction has exactly one definition in net_evaluator.hpp, reached identically by every translation unit. An IWYU/include-what-you-use observation is at most a stylistic preference, and the reviewer self-classified it as `nit` with no suggested fix while explicitly stating it is not a regression. Per the brief's instruction to default to not-a-defect when the finding is stylistic and already acknowledged as a non-regression, this is a false positive as a defect.

---

## Maintainer disposition (APPENDED — not part of the audit above)

> Per the hack-rationalization-detector's verbatim-return rule, everything above is
> the auditors' output unaltered. This section is the maintainer's response, appended.
> Recorded by Claude Opus 4.8, who commissioned and acted on the review.

All six confirmed findings were acted on in the same `refactor/fiber-leaf-ssot` change before merge (`74d5749`):

- **TreeState implicitly movable** (odr-lifetime-fiber, minor) — FIXED: `=delete`'d all four copy/move special members on `TreeState`, compile-enforcing the "never move after start()" invariant (the load-bearing one is the move ctor; copy was already auto-deleted).
- **start() captures loc/bw/coll by reference** (behavior-equivalence + odr-lifetime-fiber, two convergent minors) — FIXED by documentation: extended the `fiber_tree.hpp` purpose block and `start()` comment to state the caller must keep `loc/bw/coll` alive until `running` is false, and to warn against temporaries. The reviewers' convergent recommendation was a doc note (both warned the rvalue-`=delete` overload would reject the safe `{}` literal); the snapshot-the-belief alternative was considered and declined (the benches deliberately share one read-only root by reference — per-tree copies would be wasteful).
- **gratuitous `<memory>` include** (docs-iwyu, minor) — FIXED: removed.
- **`wire_pool_bench.cpp` transitive `NetPrediction`** (scope-hack, nit) — FIXED: added the explicit `net_evaluator.hpp` include (IWYU + parity with `wire_parallel_bench.cpp`).
- **dangling consult citation in `docs/design/cpp-search-runtime.md`** (scope-hack, minor) — NOT fixed here: this is an untracked doc on a separate (SearchRuntime) line of work, not part of the dedup change, and is surfaced to the maintainer as a separate item (the reviewer itself refuted the "rides along with the dedup" framing). The verdict on the dedup itself is `general`.

Flag on the audit's own process (appended, not altering the audit): the `docs-iwyu` lens returned a malformed one-word summary, literally `test`, reproduced verbatim above. Its individual finding (the `<memory>` include) was nonetheless concrete, independently verified, and acted on; the malformed summary is recorded rather than papered over.
