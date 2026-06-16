# Fiber-leaf SSOT dedup — adversarial review (commission, verbatim)

> The exact prompts sent to the multi-agent adversarial review for the fiber-leaf
> SSOT dedup. The review ran as a Workflow: a shared CONTEXT prepended to each of
> 4 independent lens prompts (the "Review" phase); each finding then adversarially
> verified by an independent agent (the "Verify" phase). The reports are in
> `fiber-leaf-dedup-review-report.md`. Recorded per the verbatim-record discipline.

---

## Shared context (prepended to every lens prompt)

Repo: /home/bork/w/vdc/1/chocofarm (a numpy/JAX OR scratch project; C++ search runtime under cpp/).
Under review: an UNCOMMITTED working-tree refactor that consolidates duplicated fiber-leaf primitives
into shared headers (ADR-0012 P1 single-source-of-truth). The ADRs are load-bearing; key ones:
  - ADR-0012 P1 (SSOT / one home), P7 (serialization vs transport), P9 (functional core / effect at edges).
  - ADR-0002 (fail loudly), ADR-0006 (every source file carries a module-docstring header:
    path + purpose + "Public Domain (The Unlicense)").

What changed (working tree vs git HEAD = commit 6d67c59):
  NEW  cpp/include/chocofarm/cyclic_gumbel.hpp   (chocofarm::CyclicGumbelSource — RNG-free scripted source)
  NEW  cpp/include/chocofarm/fiber_tree.hpp      (chocofarm::TreeState — one Gumbel tree in a fiber)
  MOD  cpp/include/chocofarm/fiber_leaf.hpp      (header note discharged)
  MOD  cpp/src/fiber_proto.cpp                   (Option-A proof: now drives the shared TreeState)
  MOD  cpp/src/wire_parallel_bench.cpp           (round-synchronous bench: uses shared TreeState)
  MOD  cpp/src/wire_pool_bench.cpp               (greedy-async corr-id pool: uses shared TreeState)

Background: before this change, fiber_proto.cpp and wire_parallel_bench.cpp each inlined their OWN
YieldCtx + YieldingNetEvaluator; the canonical FiberLeafChannel + YieldingNetEvaluator already lived in
fiber_leaf.hpp. A simple table-cycling scripted GumbelSource was inlined in THREE fiber files; a per-tree
TreeState fixture in TWO. (gumbel_dump.cpp has its OWN ScriptedGumbelSource with a DIFFERENT signature
— (gumbels, world_idxs), scripting world selection too — left untouched on purpose.)

How to see the diff (you have Bash + Read):
  - modified files:    cd /home/bork/w/vdc/1/chocofarm && git diff -- <path>
  - new header bodies: Read the two new .hpp files directly (untracked, so not in git diff)
  - the PRE-refactor inlined originals: git show HEAD:cpp/src/fiber_proto.cpp (and wire_parallel_bench.cpp,
    wire_pool_bench.cpp) — compare against the working-tree versions to judge behavior-preservation.

Already established by a deterministic gate (do NOT re-run; reason about the code, not the test result):
  - fiber-proto PASSes byte-identical (direct vs fibered: survivor_slot/argmax/n_spent match, 47 leaves);
  - all three wire axes PASS (sync, parallel, pool decided=32/32);
  - full cpp build is warning-clean under -Wall -Wextra.
Your job is to find what PASSING tests do NOT prove: latent ODR/lifetime hazards, subtle behavior
divergence on other configs, scope mis-calibration, doc/header-discipline gaps.

---

## Lens 1 — behavior-equivalence

Lens: BEHAVIOR-PRESERVATION. Compare the new shared types (CyclicGumbelSource, TreeState,
and the canonical YieldingNetEvaluator the proof/benches now use) and fiber_proto.cpp's rewritten drive
loop against the PRE-refactor inlined originals at HEAD. Hunt for ANY semantic divergence that PASSING on
one config would not reveal: the fixedsize_stack size (was 512*1024 — still?), field meaning/order, the
ctor member-init order vs the self-references, the fiber entry lambda's captures (the originals captured
[&] in run_fibered vs [this,&loc,&bw,&coll,lam] in TreeState — any observable difference?), the
start()/resume_with() semantics vs run_fibered's loop, the leaf-count semantics, and the proof feeding
net.predict(...).value() (was the original identical?). Also: do the direct and fibered runs still draw
from CyclicGumbelSource(gtable) with IDENTICAL cycling state? Confirm or refute exact equivalence.

## Lens 2 — odr-lifetime-fiber

Lens: ODR + LIFETIME + FIBER SAFETY. The new headers define chocofarm::{CyclicGumbelSource,
TreeState} now included by multiple TUs. (1) ODR: are they ODR-safe (class defs with in-class member
bodies are implicitly inline; identical across TUs)? Any free function in a header without inline? (2)
TreeState is self-referential: ynet holds FiberLeafChannel ref ch, policy holds NetEvaluator ref ynet, and
the fiber entry lambda captures 'this'. Is fiber_proto's STACK-LOCAL 'chocofarm::TreeState ts' safe (it
must never be moved/copied after start() — confirm TreeState is effectively non-movable and the local
never moves)? Are the benches' std::vector<unique_ptr<TreeState>> uses safe (the unique_ptr keeps the
address stable across vector growth; confirm no TreeState is moved after start())? (3) The fiber lambda
captures refs to loc/bw/coll — do those outlive the fiber in every caller? (4) Member init-order: ch
before ynet before policy — any use-before-init? Confirm or refute each hazard.

## Lens 3 — scope-hack-rationalization

Lens: SCOPE / HACK-RATIONALIZATION (run it OUT OF FRAME — you did not write this diff; treat
the implementer's expansion as the object of suspicion, not context to agree with). The LITERAL ask was
"retrofit fiber_proto + wire_parallel's inlined YieldCtx/YieldingNetEvaluator onto fiber_leaf.hpp." The
implementer EXPANDED it: also extracted CyclicGumbelSource (3 copies) and TreeState (2 copies) into new
headers, and rewrote fiber_proto to drive the shared TreeState. Produce the artifact:
  GENERAL FIX:   <the most general correct invariant for this dedup, one sentence>
  PATCH SHIPPED: <what the diff actually does>
  DOWNGRADE/EXPANSION: <was anything narrowed-with-a-discipline-word, OR expanded beyond need? quote words>
  WRITER/COPY DELTA: <independently enumerate every remaining copy of each consolidated contract —
     grep the tree yourself for 'struct YieldCtx', GumbelSource subclasses, 'struct TreeState',
     'YieldingNetEvaluator' under cpp/ — does any copy survive that should have been unified, or was any
     unified that should NOT have been (e.g. gumbel_dump's different-signature source)?>
  VERDICT: general | narrower-but-justified | over-scoped | UNDISCHARGED-HACK
Be concrete. Under-scoping (a surviving copy) and over-scoping (touching unrelated code) are both findings.

## Lens 4 — docs-iwyu

Lens: DOCUMENTATION DISCIPLINE + INCLUDE HYGIENE. (1) ADR-0006: do BOTH new headers
(cyclic_gumbel.hpp, fiber_tree.hpp) carry the module-docstring header — first line the repo-relative path,
a purpose, and "Public Domain (The Unlicense)"? (2) Is the fiber_leaf.hpp "retrofit on touch" note now
correctly discharged (it previously said the two .cpp files inline equivalents — does it now reflect
reality)? Any other cross-reference (in the 3 .cpp file headers, or elsewhere under cpp/ or docs/) that
now describes these primitives inaccurately or points at a path/section that no longer resolves? (3) IWYU:
for EACH of the 5 touched files, are the includes honest — every directly-used symbol's header included,
no gratuitous include left (e.g. a now-unused boost/context/fiber.hpp, net_evaluator.hpp, cmath, deque)?
(4) Does CMakeLists.txt need any change (the new files are headers — likely not, but confirm no target
lists sources that moved)? (5) Any ADR "Revisit when..." trigger fired by introducing two new headers?
Report concretely.

---

## Verify phase (applied to every raw finding)

A reviewer (lens: <lens>) raised this finding about the diff: <title / severity / location / detail /
suggested_fix>. Adversarially VERIFY it against the actual code (read the files / git show HEAD:<path> as
needed). Try to REFUTE it: is it a genuine defect in THIS working-tree code, or a false positive /
already-handled / stylistic non-issue? Default to is_real=false if the claim is speculative or the code
already addresses it.
