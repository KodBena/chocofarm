# ADR-0012 edification — a free reading list for disciplined, auditable software design

- **Date:** 2026-06-16
- **What this is:** a curated, verified list of *free* resources (lecture notes, open-access papers, free books, essays, recorded talks with transcripts, and primary incident reports) chosen to fill the corners that harmonize with ADR-0012's spirit but that the ADR has not yet named — a self-directed *formation-of-a-principal-engineer* reading program for a reader with strong higher mathematics and Haskell/FP but no formal software-engineering training.
- **How it was produced:** a multi-agent deep-research workflow — 9 discovery finders fanned out across the design-discipline landscape (each mapped to ADR-0012's P1–P9 and its cross-cutting spirit), an adversarial verification pass that checked each candidate is genuinely free / accessible / correctly attributed, then a synthesis pass that deduped, ranked, and organized the survivors. The raw discovery and the verification verdicts are preserved verbatim in Parts 2 and 3 so the curation is auditable (ADR-0005 Rule 9 spirit).
- **Counts:** 123 raw candidates → 110 unique titles → 109 passed verification → 54 curated below.
- **Deliberately excluded:** Therac-25 (already read by the reader); its siblings (Ariane 5, Mars Climate Orbiter, Patriot, Knight Capital, Cloudflare, Columbia, Toyota) carry the same lessons here.
- **Caveat:** URLs were verified by the research agents but the web moves; a dead link usually resolves by searching the exact title. Where a verifier supplied a corrected canonical URL it is used below and flagged in Part 2.

> This document is reference material, not a codebase decision. It lives in `docs/notes/` as a general note (not a consult/design/results/audit record — ADR-0008: no fabricated category). Public Domain (The Unlicense).

**Contents**

1. [Curated reading list](#part-1--curated-reading-list) — the deliverable, by theme, with a reading path and a P1–P9 coverage map.
2. [Verification ledger](#part-2--verification-ledger) — every unique candidate with its free / accessible / attribution verdict.
3. [Raw discovery appendix](#part-3--raw-discovery-appendix) — every candidate each finder proposed, verbatim, per dimension.

---

## Part 1 — curated reading list

A formation-of-a-principal-engineer reading list keyed to ADR-0012's nine principles and its cross-cutting spirit. The reader has strong higher mathematics and Haskell/FP but no formal SWE training, so the list leans on sources that pay off type-theoretic and proof-shaped intuition (Curry-Howard, parametricity, Kleisli composition, backward error analysis, refinement mapping) and uses incident reports as the worked counterexamples that turn each principle from aesthetics into a safety property. Every item is genuinely free; the reader will filter himself, so this is curated for diversity of medium and corner rather than completeness. Therac-25 is deliberately excluded; its siblings (Ariane 5, Mars Climate Orbiter, Patriot, Knight Capital, Cloudflare, Columbia, Toyota) carry the same lessons in the ADR's own registers.

### Suggested reading path

Start where the reader's math/Haskell gives the steepest perspective shift, then descend into mechanism and ground out in failure. (1) Perspective-resetters first, exploiting his background: Out of the Tar Pit (state as accidental complexity) → Simple Made Easy (complecting) → Propositions as Types (a signature is a theorem). These three reframe the entire ADR as one idea — keep the essential, derive everything else, and let the type carry the claim. (2) Then the decomposition roots that name the discipline: Parnas 1972 (information hiding) and Dijkstra EWD447 (separation of concerns), with Conway's Law as the why-it-drifts. (3) Seams and contracts, building on the type-theory hook: Hexagonal Architecture → Parse, Don't Validate → Hoare 1969 → Liskov-Wing (the contract-preservation theorem). (4) The error model as a unit, since it is P5+P9 and pays off monads directly: The Error Model → Boundaries → Kleisli Categories → (deeper) let-it-crash and P0709. (5) Wire discipline once boundaries are internalized: Kleppmann's schema evolution → Helland (data inside/outside) → Spec-ulation (versioning-as-variance) → End-to-End arguments. (6) Substantiation, his quantitative wheelhouse: Goldberg (float is not the reals) → Higham (backward error) → Weyuker (the oracle problem) → QuickCheck → metamorphic/differential testing. (7) Config/evolution as a short interlude: 12-Factor Config + Feature Toggles. (8) Then the humility/knowledge layer: Programming as Theory Building → No Silver Bullet → Software Aging → A Rational Design Process → the ADR origin post. (9) Finish with the failure corpus as worked counterexamples, each now mapping onto a principle he holds: Ariane 5 (P9/P6), Mars Climate Orbiter (P7/P1), Patriot (P6/P4), Knight Capital (P4/P7), Cloudflare 2025 (P9/P4), Columbia (P5/P6), Toyota (P9/P5), framed by How Complex Systems Fail and STAMP, and closed by Postmortem Culture. Highest-leverage single read if he wants only one: Out of the Tar Pit. Highest-leverage for his background specifically: Propositions as Types.

### Coverage map (P1–P9 + the cross-cutting spirit)

P1 (one home/derive): Parnas 1972, Out of the Tar Pit, Mars Climate Orbiter (units re-authored), Simple Made Easy. P2 (seam/port/ACL): Hexagonal Architecture, DIP (Martin), Liskov-Wing, Waldo (remote≠local), Leaky Abstractions, End-to-End, RFC 9413. P3 (no god-objects/SRP): Parnas 1972, Out of the Tar Pit, Software Aging, Toyota (god-object task), Lean Software-adjacent humility via No Silver Bullet. P4 (live-not-frozen config): Out of the Tar Pit, 12-Factor Config, Feature Toggles, Patriot (boot-baked drift), Knight Capital (flag meaning), Cloudflare (live config artifact). P5 (fail loud/root-cause/graded loudness): The Error Model, let-it-crash (Armstrong), Findler-Felleisen (blame), STAMP, How Complex Systems Fail, Hoare 1969, Columbia, Cloudflare, Postmortem Culture, RFC 9413. P6 (substantiate, two-tier): Goldberg, Higham, Weyuker, QuickCheck, metamorphic testing, Csmith (differential), ReproBLAS, No Silver Bullet, Ariane 5, Patriot, Columbia. P7 (wire discipline, contract vs transport): Kleppmann, Helland, Spec-ulation, End-to-End, Waldo, Mars Climate Orbiter, Knight Capital (deploy as coordination). P8 (typed signature as SSOT/no lying signatures): Propositions as Types, Theorems for Free, Hoare 1969, Meyer DbC, LiquidHaskell, Total FP, Liskov-Wing, QuickCheck, Spec-ulation. P9 (functional core/imperative shell, optional/expected, modern-C++): Boundaries, The Error Model, P0709 (C++ value-returning failure), Total FP, Kleisli, Ariane 5, Cloudflare (unwrap panic), Toyota. Cross-cutting — auditability: Conway, ADR origin post, Rational Design Process. Maintainability: Software Aging, Lampson Hints, No Silver Bullet. Knowledge-survives-author: Programming as Theory Building, Rational Design Process, Postmortem Culture. Safety culture/hard lessons: Cook, STAMP, Ariane 5, MCO, Patriot, Knight Capital, Cloudflare, Columbia, Toyota. Honest claims: Goldberg, Csmith, No Silver Bullet, Columbia, the whole P6 cluster.

### Corners with thin or missing free coverage

*(named honestly rather than papered over — ADR-0008 / ADR-0002 spirit)*

- P9's modern-C++ reliquary-to-modern substitutions at ZERO RUNTIME COST (e.g. const-correctness, return-by-value/RVO, std::span as a bounds-carrying view, replacing raw-pointer/sentinel idioms) is thin: P0709 covers only the error-channel facet, and no free, rigorous treatment of the broader zero-cost-abstraction-as-correctness story made the verified pool. The C++ Core Guidelines exist and are free but were not in the verified candidate set, so I did not include them; a reader wanting this corner should seek them out separately and verify.
- P7's runtime-manifest-vs-build-time-codegen distinction is covered conceptually (Kleppmann, Spec-ulation, Helland) but the concrete reference-doc instances that made it tangible (Avro Schema Resolution, Protobuf reserve-don't-reuse, Cap'n Proto/FlatBuffers zero-copy layouts) were dropped to preserve medium diversity and avoid a documentation-heavy theme; the conceptual essays carry the principle, but a reader doing wire work would benefit from reading at least the Avro and Protobuf evolution rules directly (both free, in the pool).
- P6's two-tier bar is well covered for the NUMERICS tier; the BIT-EXACT LOGIC-INVARIANT tier is covered only obliquely (QuickCheck/property testing, Software Foundations was available but cut for balance). Machine-checked invariants as the limit of 'mechanization over memory' is gestured at but not given a dedicated essential read here.
- The empirical-reproducibility-catastrophe angle (Hatton's T-experiments, Ten Simple Rules for Reproducible Research) and the Kahan float-platform-variation essays were available and strong but cut to keep the float/oracle theme from dominating; a reader specifically worried about cross-platform FMA/register-width reproducibility in the JAX path has thinner coverage here than the topic deserves, and should pull the two Kahan essays from the pool.
- P3's god-object corner has no single dedicated deep treatment in this final cut beyond Parnas and the Toyota counterexample; Wirth's 'A Plea for Lean Software' (free, in pool) would have been the on-point comprehensibility-as-a-budget read and was dropped for theme balance.

### The list, by theme

#### Decomposition and the one-home principle

*Information hiding, separation of concerns, Conway's homomorphism, and essential-vs-accidental state — the theory under P1 (one home per fact) and P3 (no god-objects). These say WHY a fact has a home: it is the secret a module hides, the concern you study in isolation, the essential state from which everything else is derived.*

- **On the Criteria To Be Used in Decomposing Systems into Modules** — David L. Parnas (1972)
  *open-access-paper* · **essential** · foundational
  Decompose around the secrets each module hides (the decisions likely to change), not around the flowchart's steps. This is the theory under P1's 'one home per fact' and P3's one-owner collaborators.
  *Fills:* P1, P3 — the origin of information hiding: a fact's one home is the module that HIDES the design decision likely to change.
  <https://wstomv.win.tue.nl/edu/2ip30/references/criteria_for_modularization.pdf>

- **On the Role of Scientific Thought (EWD447) — separation of concerns** — Edsger W. Dijkstra (1974)
  *essay* · **essential** · foundational
  Names the move every ADR-0012 principle inherits: study one aspect in isolation for its own consistency, knowing the aspects are not independent. The 'attention not severance' nuance directly justifies examining logic invariants and float behavior as separate concerns.
  *Fills:* P3 and cross-cutting spirit — the original 'separation of concerns', and crucially that the separation is in one's ATTENTION (study correctness and efficiency on different days), underwriting P6's two-tier bar.
  <https://www.cs.utexas.edu/~EWD/transcriptions/EWD04xx/EWD447.html>

- **Out of the Tar Pit** — Ben Moseley and Peter Marks (2006)
  *open-access-paper* · **essential** · intermediate
  The sharpest account of why mutable state is the dominant source of accidental complexity and why deriving everything from one authoritative store is the antidote — the deep argument under P1 (one home) and P4 (read-at-point-of-use). For a relational/functional mind it reframes 'god-object' as too much implicit reachable state.
  *Fills:* P1, P3, P4 — mutable state as the dominant accidental complexity; derive views from one authoritative store.
  <https://curtclifton.net/papers/MoseleyMarks06a.pdf>

- **How Do Committees Invent? (Conway's Law, original)** — Melvin E. Conway (1968)
  *essay* · **recommended** · foundational
  A system's structure copies the communicating organization that built it. Inverted for a solo author: the only force fragmenting the design is forgotten intent — exactly what P1/P7's mechanization defends against.
  *Fills:* P2/P3 and auditability — WHY module boundaries drift: the homomorphism between communication structure and system structure.
  <https://www.melconway.com/research/committees.html>

#### Seams, ports, and the discipline of boundaries

*Dependency inversion, ports-and-adapters, module depth, leaky abstractions, behavioral subtyping, and data abstraction as a typing phenomenon — P2's seam discipline given both its naming (DIP/LSP) and its honest limits (every abstraction leaks; the boundary must translate-and-validate, never coerce).*

- **Hexagonal Architecture (Ports and Adapters)** — Alistair Cockburn (2005)
  *essay* · **essential** · intermediate
  The primary source for ports-and-adapters: the application core defines an interface it owns and every adapter conforms, so the core runs identically under test, script, or production. The inversion that makes P2's seam real rather than decorative.
  *Fills:* P2, P7 — the core OWNS the port; effects depend on the core; contract separated from transport.
  <https://alistair.cockburn.us/hexagonal-architecture/>

- **Design Principles and Design Patterns (Dependency Inversion Principle)** — Robert C. Martin (2000)
  *open-access-paper* · **recommended** · intermediate
  The canonical statement of Dependency Inversion. P2's 'a new capability is a new Policy subclass with zero core edits' IS the DIP; this gives the reader its vocabulary and failure modes. (Wayback URL used because the original objectmentor PDF is defunct.)
  *Fills:* P2 — names the inversion: depend on abstractions, high-level policy must not depend on low-level mechanism (origin of SOLID's DIP/ISP).
  <https://web.archive.org/web/20150906155800if_/http://www.objectmentor.com/resources/articles/Principles_and_Patterns.pdf>

- **A Philosophy of Software Design (Talks at Google) — deep vs shallow modules** — John Ousterhout (2018)
  *recorded-talk* · **recommended** · intermediate
  Deep-vs-shallow gives P2 a quantitative target: the env↔Policy seam is good precisely because a one-method interface hides a large implementation. Names 'information leakage' and 'temporal decomposition' as the failures that erode SSOT/SRP. (Captions/auto-transcript available.)
  *Fills:* P2/P3 — module DEPTH: a good seam maximizes hidden complexity per unit of interface.
  <https://www.youtube.com/watch?v=bmSAYlu0NcY>

- **The Law of Leaky Abstractions** — Joel Spolsky (2002)
  *essay* · **recommended** · foundational
  The honest counterweight to 'just hide it behind an interface': the boundary must expose enough to be diagnosable. Explains why P7's wire discipline cannot fully hide float32 non-associativity or layout — the leak is real and must be named.
  *Fills:* P2 — the realism check: every non-trivial abstraction leaks, so a port must translate-and-validate, not pretend the foreign side is invisible.
  <https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/>

- **A Behavioral Notion of Subtyping** — Barbara Liskov, Jeannette Wing (1994)
  *open-access-paper* · **deeper-cut** · principal/advanced
  Formalizes LSP as a contract-preservation theorem: a subtype may weaken preconditions and strengthen postconditions but never violate the supertype's invariant/history constraint. The specification-mapping construction rewards the reader's math background.
  *Fills:* P2 (substitutability), P8 (a subtype must respect the supertype contract) — the rule behind 'translate-and-validate, never coerce'.
  <https://www.cs.cmu.edu/~wing/publications/LiskovWing94.pdf>

#### Types as contracts, and the contract as a theorem

*Hoare triples, Design by Contract, propositions-as-types, parametricity, refinement types, totality — P8's 'no lying signatures' taken from coding convention down to its proof-theoretic foundation, in the dialect the reader's math and Haskell already speak.*

- **Propositions as Types** — Philip Wadler (2015)
  *open-access-paper* · **essential** · principal/advanced
  The load-bearing 'why': Curry-Howard says a type is a theorem and a total program of that type is its proof, so a precise signature is a falsifiable claim and a sentinel/throw on the core is a hole in the proof. The deepest justification for 'no lying signatures'.
  *Fills:* P8 — a typed signature IS a proposition; the implementation is its proof; cross-cutting honesty.
  <https://homepages.inf.ed.ac.uk/wadler/papers/propositions-as-types/propositions-as-types.pdf>

- **An Axiomatic Basis for Computer Programming** — C. A. R. Hoare (1969)
  *open-access-paper* · **recommended** · intermediate
  The Hoare-triple is the mathematical object an assertion/contract approximates; for a math reader this turns Design by Contract into a proof system and supplies the formal grammar P8's 'typed signature is the SSOT' rests on.
  *Fills:* P8, P5 — the logical meaning of a precondition/postcondition pair P{Q}R beneath the engineering slogan.
  <https://www.cs.cmu.edu/~crary/819-f09/Hoare69.pdf>

- **Design by Contract (chapter, Object-Oriented Software Construction)** — Bertrand Meyer (1997)
  *free-book* · **recommended** · intermediate
  The canonical long-form treatment: contracts as the unit of correctness, invariants as the conjunction every public method must preserve, and the obligation/benefit table that is the precise shape of an ACL boundary. Author-hosted free substitute for the paywalled book.
  *Fills:* P8, P5, P2 and honest-claim spirit — a contract is a checkable claim, not a comment.
  <https://se.inf.ethz.ch/~meyer/publications/old/dbc_chapter.pdf>

- **Theorems for Free!** — Philip Wadler (1989)
  *open-access-paper* · **deeper-cut** · principal/advanced
  Parametricity: a sufficiently polymorphic type forces theorems on every inhabitant, so the signature alone guarantees behavioral laws no test need assert — the strongest reading of 'the signature is the SSOT'. (ACM open PDF used; Wadler's own .dvi/.ps copies also free but need a viewer.)
  *Fills:* P8 (a polymorphic signature constrains all implementations), P6 (invariants you get without a test).
  <https://dl.acm.org/doi/pdf/10.1145/99370.99404>

- **Programming with Refinement Types: An Introduction to LiquidHaskell** — Ranjit Jhala, Niki Vazou, Eric Seidel, et al. (2020)
  *free-book* · **deeper-cut** · principal/advanced
  The bridge from the reader's Haskell to machine-enforceable 'no lying signatures': refinement types attach SMT-checked predicates to types. Shows what P8's mypy-strict ratchet aspires to at its logical end — the spec lives in the type and the checker proves it.
  *Fills:* P8 (push the contract INTO the type), P5 (verification failure is a loud compile error).
  <https://ucsd-progsys.github.io/liquidhaskell-tutorial/book.pdf>

- **Total Functional Programming** — D. A. Turner (2004)
  *open-access-paper* · **deeper-cut** · principal/advanced
  Argues for a discipline where every function provably terminates and data/codata are explicit, so A->B is a genuine total guarantee, not 'B or loop-forever or crash'. The rigorous root of P9's 'never throws on the core', in a totality framing the reader half-knows.
  *Fills:* P8 (partiality is a hidden lie), P9 (no bottom/throw on the core), P6 (termination as a checkable invariant).
  <https://www.jucs.org/jucs_10_7/total_functional_programming/jucs_10_07_0751_0768_turner.pdf>

#### The error model: functional core, loud failure, errors as values

*Functional-core/imperative-shell, parse-don't-validate, illegal-states-unrepresentable, the bug-vs-recoverable-error partition, blame, let-it-crash, fail-fast, and the C++/Rust-specific value-returning failure channel — P5's graded loudness and P9's optional/expected discipline, with the Kleisli algebra underneath.*

- **Parse, Don't Validate** — Alexis King (2019)
  *essay* · **essential** · intermediate
  The Haskell-native articulation of the functional-core/imperative-shell boundary: error handling is a type-system discipline, not a runtime habit. Tailor-made for a reader with deep types — the unnamed corner connecting P2, P8, and P9-rule5.
  *Fills:* P2, P8, P9-rule5 — the boundary parses untrusted input into a type that makes invalid states unrepresentable, so the core never re-validates.
  <https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/>

- **The Error Model** — Joe Duffy (2016)
  *essay* · **essential** · principal/advanced
  The most complete first-principles treatment of the design space: BUGS must fail-fast and be uncatchable; RECOVERABLE errors belong in the type signature as checked values — exactly P5's loudness hierarchy and P9's optional-vs-expected split, with measured cost data from a real systems language (Midori).
  *Fills:* P5 (bugs vs recoverable errors), P9-rule5 (optional vs expected, never throw/sentinel on the core).
  <https://joeduffyblog.com/2016/02/07/the-error-model/>

- **Boundaries** — Gary Bernhardt (2012)
  *recorded-talk* · **essential** · intermediate
  The origin of 'functional core, imperative shell': push decisions into pure functions over simple values and quarantine effects in a thin edge shell. Directly underwrites P9's compiled-C++ 'pure functions in, effectful glue out' split, generalized beyond any language.
  *Fills:* P9 (functional core / imperative shell), P2 (values as seams), P3.
  <https://www.destroyallsoftware.com/talks/boundaries>

- **Kleisli Categories (Category Theory for Programmers)** — Bartosz Milewski (2014)
  *free-book* · **recommended** · principal/advanced
  Supplies the math the reader will want: error-returning functions compose because they are morphisms in a Kleisli category, and 'short-circuit on failure' is Kleisli composition for Maybe/Either. The corner under Railway-Oriented Programming and expected<T> — these patterns are honest because they obey associativity and identity.
  *Fills:* P9-rule5 (the algebra under Result/expected composition), P6 (composability should rest on a law).
  <https://bartoszmilewski.com/2014/12/23/kleisli-categories/>

- **Zero-overhead deterministic exceptions: Throwing values (P0709)** — Herb Sutter (2019)
  *open-access-paper* · **recommended** · principal/advanced
  The C++-specific reckoning: catalogues why today's C++ mixes throws, error codes, errno, and nullable returns into an incoherent mess, and proposes a value-returning failure channel — the standards-track substantiation behind P9's 'expected for failure, optional for absence' at the ABI level.
  *Fills:* P9 (compiled-C++ error model), P9-rule5 (expected for failure, never sentinel/nullptr/throw on the core), P6 (the cost argument is substantiated).
  <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0709r4.pdf>

- **Making Reliable Distributed Systems in the Presence of Software Errors (the 'let it crash' thesis)** — Joe Armstrong (2003)
  *free-book* · **recommended** · principal/advanced
  The primary source for 'let it crash': error recovery belongs to a separate supervising process, so the happy path stays pure and the failure path is concentrated and tested. Answers P5+P3 — where error handling lives and why not inline; process isolation as the unit of fault containment.
  *Fills:* P5 (let-it-crash: recover at a supervisor, not via defensive inline patches), P3 (one-owner isolated processes), P2.
  <https://erlang.org/download/armstrong_thesis_2003.pdf>

- **Contracts for Higher-Order Functions** — Robert Bruce Findler, Matthias Felleisen (2002)
  *open-access-paper* · **deeper-cut** · principal/advanced
  Introduces correct blame assignment across higher-order boundaries — the rigorous answer to 'fail loud, at the real culprit, not a band-aid downstream'. For a Haskeller, contracts as runtime-checked refinements exactly where static types stop.
  *Fills:* P5 (blame: WHO failed, not just that something failed), P8, P2.
  <https://www2.ccs.neu.edu/racket/pubs/icfp2002-ff.pdf>

#### Wire discipline and the cross-boundary contract

*Schema evolution, runtime-manifest vs build-time-codegen, zero-copy layouts, data-inside-vs-outside, end-to-end placement of guarantees, the robustness-principle reckoning, and versioning-as-variance — P7's single authoritative definition and the separation of serialization contract from transport mechanism.*

- **Schema evolution in Avro, Protocol Buffers and Thrift** — Martin Kleppmann (2012)
  *essay* · **essential** · intermediate
  Shows byte-for-byte how three formats encode the same record and survive a field being added/removed/renamed — the mechanics behind 'one authoritative definition, every side derives its view'. Avro's reader/writer-schema split is the clean separation P7 demands.
  *Fills:* P7 — one authoritative wire definition; serialization contract vs transport; reader/writer-schema split.
  <https://martin.kleppmann.com/2012/12/05/schema-evolution-in-avro-protocol-buffers-thrift.html>

- **Data on the Outside versus Data on the Inside** — Pat Helland (2005)
  *open-access-paper* · **recommended** · principal/advanced
  The conceptual law beneath the serialization formats: data crossing a service boundary becomes immutable, time-stamped, and self-describing — why a wire contract must be versioned and a manifest must travel with the bytes. Reframes 'a store holds state, a fabric carries coordination' as an ontological distinction.
  *Fills:* P7, P2, P4 — WHY a boundary changes the nature of data: outside data is immutable, versioned, reference-by-value; inside data is mutable and authoritative.
  <https://www.cidrdb.org/cidr2005/papers/P12.pdf>

- **A Note on Distributed Computing** — Jim Waldo, Geoff Wyant, Ann Wollrath, Sam Kendall (1994)
  *open-access-paper* · **recommended** · principal/advanced
  The foundational argument that the seam must EXPOSE that it is a seam. For a Haskell reader it sharpens the intuition that a remote operation's type is genuinely different (it can fail and partial-fail) from the local one it resembles.
  *Fills:* P2, P7 — a remote call is NOT a local call; latency, partial failure, concurrency cannot be papered over by transparency.
  <https://scholar.harvard.edu/files/waldo/files/waldo-94.pdf>

- **Spec-ulation (keynote transcript)** — Rich Hickey (2016)
  *recorded-talk* · **recommended** · intermediate
  Turns versioning from folklore into an algebra: a provider may only weaken its requires and strengthen its provides (a variance argument a Haskeller will recognize), else it must rename rather than break. The principled foundation under Protobuf's reserve-don't-reuse and P8's 'no lying signatures'. (Recorded talk WITH transcript.)
  *Fills:* P7, P8 — schema/API evolution as monotonic growth vs breakage: weaken requires, strengthen provides, or rename.
  <https://github.com/matthiasn/talk-transcripts/blob/master/Hickey_Rich/Spec_ulation.md>

- **RFC 9413 — Maintaining Robust Protocols ('Postel was wrong')** — Martin Thomson, David Schinazi (IAB) (2023)
  *incident-report* · **recommended** · intermediate
  The definitive case against 'be liberal in what you accept': leniency silently absorbs nonconformance, which calcifies into de-facto spec and rots interoperability. Robustness-by-tolerance is a slow-acting failure, so the boundary must reject loudly now to stay honest later.
  *Fills:* P2 (ACL boundaries translate-and-VALIDATE, never coerce), P5 (fail loud at the boundary), P7.
  <https://www.rfc-editor.org/rfc/rfc9413.html>

- **End-to-End Arguments in System Design** — J. H. Saltzer, D. P. Reed, D. D. Clark (1984)
  *open-access-paper* · **recommended** · principal/advanced
  Tells you which layer should own a guarantee: the transport can be best-effort, but contract validation must live at the semantically-aware endpoint. The rigorous justification for 'a fabric carries coordination, a store holds state' and for parsing-at-the-boundary rather than trusting the pipe.
  *Fills:* P2, P5 — WHERE a check belongs: correctness guarantees live at the endpoints that hold the meaning, not in the transport beneath.
  <https://web.mit.edu/saltzer/www/publications/endtoend/endtoend.pdf>

#### Substantiating claims: floating point, oracles, and reproducibility

*Float non-associativity, backward error analysis, the oracle problem, metamorphic and property-based and differential testing, reproducible summation, and the empirical reproducibility catastrophe — P6's two-tier bar (bit-exact logic vs aggregate-behavioral numerics) and how to actually earn an equivalence claim in a solver with no closed-form answer.*

- **What Every Computer Scientist Should Know About Floating-Point Arithmetic** — David Goldberg (1991)
  *open-access-paper* · **essential** · foundational
  Establishes from first principles that floating-point is not the reals. The math that forces P6's split between a bit-exact logic tier and a tolerance-based numerics tier — without it, an 'equivalence test' silently asserts a falsehood.
  *Fills:* P6 — WHY float equivalence must be aggregate-behavioral, not bit-exact: non-associativity, rounding, cancellation, ULP/relative error.
  <https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html>

- **On Testing Non-Testable Programs** — Elaine J. Weyuker (1982)
  *open-access-paper* · **essential** · intermediate
  The founding statement that some programs are non-testable (no oracle, or one too costly) and the introduction of pseudo-oracles — the conceptual root that justifies metamorphic/property/differential testing as the substitute for absent ground truth.
  *Fills:* P6 — the ORACLE PROBLEM: what to do when no known-correct answer exists, exactly the situation of a novel belief-MDP solver.
  <https://homes.cs.washington.edu/~rjust/courses/CSE503/2021_02_12-reading1.pdf>

- **QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs** — Koen Claessen, John Hughes (2000)
  *open-access-paper* · **essential** · intermediate
  The canonical property-based-testing paper, in the reader's idiom: a property (reverse . reverse == id) is a contract checked over random inputs — the mechanized substantiation P6 demands and the runtime echo of P8's typed signature as SSOT.
  *Fills:* P6, P8 — a property is a machine-checked specification, the executable form of a function's contract.
  <https://www.cs.tufts.edu/~nr/cs257/archive/john-hughes/quick.pdf>

- **Accuracy and Stability of Numerical Algorithms (SIAM Day lecture slides)** — Nicholas J. Higham (2013)
  *lecture-notes* · **recommended** · principal/advanced
  Backward vs forward error and conditioning: a stable algorithm gives the exact solution to a slightly perturbed input — the principled definition of P6's behavioral-equivalence tier. Turns 'close enough' from a hand-wave into a theorem-grade claim. (Author's slides index linked because both 2013 mirror PDFs were transiently unreachable; this page hosts the talk.)
  *Fills:* P6 — backward error analysis: 'the computed answer is the exact answer to a nearby problem', the rigorous frame for what 'equivalent' means.
  <https://nhigham.com/slides/>

- **Metamorphic Testing: A Review of Challenges and Opportunities** — T. Y. Chen, F.-C. Kuo, H. Liu, P.-L. Poon, D. Towey, T. H. Tse, Z. Q. Zhou (2018)
  *open-access-paper* · **recommended** · intermediate
  Systematizes checking invariant relations between related runs rather than absolute outputs. For an orienteering solver with no closed form, MRs (monotonicity in budget, permutation invariance of routes, scaling laws) are the testable substance behind P6's behavioral tier. (OA mirror also at homes.cs.washington.edu/~rjust if the landing page blocks.)
  *Fills:* P6 — HOW to test without an oracle: metamorphic relations as the practical oracle substitute.
  <https://nottingham-repository.worktribe.com/output/925152/metamorphic-testing-a-review-of-challenges-and-opportunities>

- **Finding and Understanding Bugs in C Compilers (Csmith)** — Xuejun Yang, Yang Chen, Eric Eide, John Regehr (2011)
  *open-access-paper* · **recommended** · intermediate
  Csmith found 325+ bugs by comparing multiple compilers' outputs — an oracle built from disagreement, not ground truth. The directly transferable idea: differential-test the C++ functional core against the numpy/JAX reference path as the substantiation of equivalence (P6).
  *Fills:* P6, P7, P9 — DIFFERENTIAL testing as an oracle: when N implementations of one spec disagree, at least one is wrong.
  <https://users.cs.utah.edu/~regehr/papers/pldi11-preprint.pdf>

- **Efficient Reproducible Floating Point Summation and BLAS (ReproBLAS)** — James Demmel, Willow (Peter) Ahrens, Hong Diep Nguyen (2016)
  *open-access-paper* · **deeper-cut** · principal/advanced
  Order-independent (parallel-reproducible) summation is achievable but not free — quantifying the price of demanding bit-exactness. Sharpens P6's two-tier choice: exactly when to spend for bit-reproducibility vs accept aggregate-behavioral equivalence.
  *Fills:* P6, P7 — parallelism + non-associativity breaks bit-reproducibility, and the cost to restore it.
  <https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.pdf>

#### Configuration, evolution, and the cost of frozen decisions

*Live-not-frozen config, feature toggles as classified live config, and semantic versioning as a mechanized compatibility claim — P4's read-at-point-of-use applied at the deployment boundary, plus the classification discipline the ADR's HOT/RESTART/INSTANCE facets demand.*

- **Store config in the environment (Twelve-Factor App, Factor III)** — Adam Wiggins (Heroku) (2011)
  *essay* · **recommended** · foundational
  The litmus test 'could you open-source the codebase now without leaking config?' and strict code/config separation. P4 at the system boundary: a swept value lives in a live source, not welded into the artifact — the operational generalization of the hp-registry HOT facet.
  *Fills:* P4 — config is everything that varies between deploys, kept out of code and read at run time: the deployment twin of read-at-point-of-use.
  <https://12factor.net/config>

- **Feature Toggles (aka Feature Flags)** — Pete Hodgson (martinfowler.com) (2017)
  *essay* · **recommended** · intermediate
  Categorizes toggles by longevity and dynamism and warns that long-lived hidden flags become a maintenance tar pit — the live-config analogue of P4's facet placement and P5's 'remove the band-aid'. Knight Capital is the worked failure.
  *Fills:* P4 plus classification discipline — a flag is live config whose category (release/ops/experiment/permission) and lifetime must be classified, echoing the ADR's HOT/RESTART/INSTANCE facets.
  <https://martinfowler.com/articles/feature-toggles.html>

#### Knowledge that survives its author; simplicity and humility

*Programming-as-theory-building, rational-design-process documentation, the ADR genre itself, simple-vs-easy, lean software, no-silver-bullet, software aging — the cross-cutting maintainability and reconstruction-cost spirit: load-bearing knowledge must live where it can be reconstructed, not in unenforceable prose.*

- **Programming as Theory Building** — Peter Naur (1985)
  *open-access-paper* · **essential** · principal/advanced
  When the theory dies (the author leaves), the program is dead even if it compiles — the deepest justification for ADR-0012's 'load-bearing knowledge encoded in code, not unenforceable prose', and for why a derived SSOT is what lets the theory be reconstructed.
  *Fills:* Reconstruction-cost corner — a program's real artifact is the THEORY in the builders' heads, not the source or docs.
  <https://pages.cs.wisc.edu/~remzi/Naur.pdf>

- **Simple Made Easy (transcript)** — Rich Hickey (2011)
  *recorded-talk* · **essential** · intermediate
  Draws the distinction the ADR never names: 'complecting' is the precise vocabulary for what P3 (no god-objects) and P2 (clean seams) forbid. Gives the reader a knife for telling essential from accidental tangling. (Recorded talk WITH transcript.)
  *Fills:* P1, P2, P3 and whole-spirit — SIMPLE (objective, un-braided) vs EASY (subjective, near-at-hand).
  <https://github.com/matthiasn/talk-transcripts/blob/master/Hickey_Rich/SimpleMadeEasy.md>

- **A Rational Design Process: How and Why to Fake It** — David L. Parnas, Paul C. Clements (1986)
  *open-access-paper* · **recommended** · intermediate
  Real design is never top-down, yet the documentation should be written as if it were, because the faked-rational record is what a maintainer can navigate. Directly underwrites the stance that an ADR is a point-in-time rational record, not a transcript of how the insight arrived.
  *Fills:* Documentation-discipline spirit — write the documentation the ideal process WOULD have produced, even though discovery was messy.
  <https://www.cs.tufts.edu/~nr/cs257/archive/david-parnas/fake-it.pdf>

- **Documenting Architecture Decisions (the ADR origin post)** — Michael Nygard (2011)
  *essay* · **recommended** · foundational
  The post that defined the ADR. Reading it makes the chocofarm conventions — 'Revisit when…', amend-by-append, never silently rewrite a point-in-time record — legible as a deliberate genre rather than house style.
  *Fills:* Documentation/auditability spirit — the genre definition of the artifact ADR-0012 IS (Context/Decision/Status/Consequences, append-only).
  <https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions>

- **No Silver Bullet — Essence and Accidents of Software Engineering** — Frederick P. Brooks, Jr. (1986)
  *open-access-paper* · **recommended** · foundational
  ADR-0012's whole modern-C++/typed-signature program is an assault on accidental complexity at zero cost to essence. Brooks gives the vocabulary to argue which is which, and why no rule is a silver bullet — the ADR's own 'most principles are review-only' humility.
  *Fills:* Honest-claims spirit, P6, P9 — essential vs accidental complexity; zero-cost abstraction removes accident, not essence.
  <https://www.cs.unc.edu/techreports/86-020.pdf>

- **Hints for Computer System Design (1983 original)** — Butler W. Lampson (1983)
  *open-access-paper* · **recommended** · intermediate
  Each terse hint crystallizes a seam/separation discipline. The closest thing to a principal-engineer field guide in compact form; the lived counterweights to the named principles (when to violate, and why). (Lampson's 2020 arXiv:2011.02455 expansion is the longer free companion if the reader wants depth.)
  *Fills:* P2, P5, P6 — the aphoristic field guide: 'do one thing well', 'handle normal and worst case separately', 'use a good idea again instead of generalizing it'.
  <https://www.microsoft.com/en-us/research/wp-content/uploads/1983/10/Hints-for-Computer-System-Design-IEEE-Software.pdf>

- **Software Aging** — David L. Parnas (1994)
  *open-access-paper* · **deeper-cut** · intermediate
  Names why connective tissue rots: changes that violate the original design's structure. ADR-0012's no-god-object/SSOT/seam rules are precisely the prophylaxis against ignorant surgery; 'right idea applied once and not propagated' is aging in Parnas's vocabulary. (Drexel host serves a self-signed cert; the PDF is genuine.)
  *Fills:* Maintainability spirit — the two causes of decay: 'lack of movement' and 'ignorant surgery'; the disease ADR-0012's taxonomy inverts.
  <https://www.cs.drexel.edu/~yc349/CS451/RequiredReadings/SoftwareAging.pdf>

#### Safety culture and hard-won failure (the Therac siblings)

*Primary incident reports and the systems-safety lens — Ariane 5, Mars Climate Orbiter, Patriot, Knight Capital, Cloudflare, Columbia, Toyota — plus STAMP, How Complex Systems Fail, and blameless-postmortem culture. Each is a worked counterexample mapping a real catastrophe onto a specific ADR principle, and the culture that extracts the systemic root cause rather than a scapegoat.*

- **How Complex Systems Fail** — Richard I. Cook (2000)
  *essay* · **essential** · intermediate
  Eighteen terse propositions on why robust systems still fail and why operators at the sharp end fight latent faults continuously — the systems-safety lens behind the ADR's hard-won-failure spirit. A Therac sibling read in fifteen minutes.
  *Fills:* Safety-culture spirit, P5 — failure needs multiple defenses to align; 'root cause' is a narrative trap.
  <https://www.adaptivecapacitylabs.com/HowComplexSystemsFail.pdf>

- **Ariane 5 Flight 501 Failure — Report by the Inquiry Board** — J. L. Lions et al. (ESA/CNES) (1996)
  *incident-report* · **essential** · intermediate
  A Therac sibling whose root cause is exactly a type/range failure: an unprotected conversion crashed the IRS, and dead reused code embodied an unverified 'equivalent to Ariane-4' claim. Teaches why P9's bounds-carrying types and P6's substantiated-equivalence bar are safety properties.
  *Fills:* P9 (a value that should be impossible became representable — unprotected 64-bit float to 16-bit int), P5 (the protection was deliberately omitted), P6 (Ariane-4 ranges reused without re-substantiation).
  <https://www.di.unito.it/~damiani/ariane5rep.html>

- **Mars Climate Orbiter Mishap Investigation Board — Phase I Report** — A. Stephenson et al., NASA MCO MIB (1999)
  *incident-report* · **essential** · intermediate
  Ground software emitted pound-force-seconds while navigation consumed newton-seconds; a cross-component WIRE fact each side re-authored by hand instead of deriving from one specification — the purest P7/P1 cautionary tale for the C++ actor transport.
  *Fills:* P7 (a cross-boundary unit fact had two authors instead of one derived contract), P1 (one home per fact), P8 (an implicit unit contract = a lying signature).
  <https://llis.nasa.gov/llis_lib/pdf/1009464main1_0641-mr.pdf>

- **Cloudflare outage on November 18, 2025 (official post-mortem)** — Cloudflare (Matthew Prince et al.) (2025)
  *incident-report* · **essential** · intermediate
  A DB permissions change doubled a Bot-Management feature file past a hard-coded 200-feature limit, and the Rust proxy called .unwrap() on the resulting Err and panicked globally — a recent, direct instantiation of P9's 'never throw/sentinel on the core; expected<> for failure' at a live-config boundary (P4), and a model of an honest post-mortem.
  *Fills:* P9 (Result::unwrap() panicking the core on an error path that 'could not happen'), P4 (a periodically regenerated config file as live unbounded input), P5 (a hard-coded limit whose failure was unhandled).
  <https://blog.cloudflare.com/18-november-2025-outage/>

- **A New Accident Model for Engineering Safer Systems (STAMP)** — Nancy G. Leveson (2004)
  *open-access-paper* · **recommended** · principal/advanced
  The compact statement of STAMP: a band-aid that leaves the constraint un-enforced is not a fix. The control-theory framing rewards a strong-math reader who recognizes a feedback-control model of a sociotechnical plant — the principal-level generalization of P5. (Full book free at OAPEN if wanted.)
  *Fills:* P5, P2 — safety as constraint-enforcement over a hierarchical control structure; accidents are inadequate constraints, not bad luck.
  <http://sunnyday.mit.edu/accidents/safetyscience-single.pdf>

- **Patriot Missile Defense: Software Problem Led to System Failure at Dhahran (GAO/IMTEC-92-26)** — U.S. Government Accountability Office (1992)
  *incident-report* · **recommended** · intermediate
  1/10 is not representable in fixed-point binary, so a per-tick truncation compounded over 100 hours into a 0.34s tracking error that missed a Scud — the canonical demonstration of P6's inexactness and why elapsed-time-since-boot is a drifting, not frozen, quantity (P4). (gov asset; CDN may 403 a bot, downloads in a browser.)
  *Fills:* P6 (float is not exact), P4 (a value baked at boot accumulating error over uptime), P5 (a known fix that did not reach the field).
  <https://www.gao.gov/assets/imtec-92-26.pdf>

- **In the Matter of Knight Capital Americas LLC (SEC Release 34-70694)** — U.S. Securities and Exchange Commission (2013)
  *incident-report* · **recommended** · intermediate
  A partial deploy left one of eight servers running dead 'Power Peg' code that a reused flag re-activated, losing ~$460M in 45 minutes — a flag with two meanings over time (no SSOT for the flag), dead code not removed, and no loud deploy-consistency check. (sec.gov 403s bots; downloads in a browser.)
  *Fills:* P4 (a repurposed flag flipped meaning between deploys), P7 (deploy is out-of-band coordination distinct from the state the bytes carry), P5 (alarms fired but were not loud), P8 (dead code reanimated).
  <https://www.sec.gov/files/litigation/admin/2013/34-70694.pdf>

- **Columbia Accident Investigation Board Report, Volume I (Ch. 7-8: organizational causes)** — Harold Gehman et al., CAIB (2003)
  *incident-report* · **recommended** · principal/advanced
  The Board insists the organizational cause — schedule pressure plus a quietly accepted out-of-spec condition — was the real root, and dissects how a PowerPoint engineering culture let an unsubstantiated safety claim pass review. The deepest free articulation of normalization-of-deviance and why honest substantiation is a safety property.
  *Fills:* P5 (organizational root cause, not just the foam), P6 (an unsubstantiated 'flew before so it's safe' equivalence), normalization-of-deviance spirit.
  <https://www.nasa.gov/wp-content/uploads/static/history/columbia/reports/CAIBreportv1.pdf>

- **A Case Study of Toyota Unintended Acceleration and Software Safety** — Philip Koopman (Carnegie Mellon) (2014)
  *lecture-notes* · **recommended** · principal/advanced
  Walks through the actual ETCS firmware defects an expert team found — stack overflow into mirrored RAM, unprotected global state, a watchdog that could not catch a hung task — precisely the bounds-carrying, fail-loud disciplines P9 and P5 prescribe, told as a fatal counterexample. CC BY 4.0, with recorded lecture and slides.
  *Fills:* P9 (unsafe shared mutable state, missing bounds, no error discipline in the embedded core), P5 (defeated watchdog = a silenced loud failure), P3 (a god-object task whose corruption had global effect).
  <https://users.ece.cmu.edu/~koopman/toyota/index.html>

- **Postmortem Culture: Learning from Failure (SRE, Ch. 15)** — Beyer, Jones, Petoff, Murphy (eds.), Google (2016)
  *free-book* · **recommended** · foundational
  The operational complement to Leveson and Cook: how to run a blameless postmortem so the organization extracts and records the systemic root cause — the cultural machinery that makes P5 and 'knowledge must survive its author' real rather than aspirational. Free, CC-licensed.
  *Fills:* P5 (institutionalize fixing the root cause; blameless = remove the systemic cause not the scapegoat), documentation/reconstruction-cost spirit.
  <https://sre.google/sre-book/postmortem-culture/>

---

## Part 2 — verification ledger

Every one of the 110 unique candidates, with the adversarial verifier's verdict. Summary: **1 not recommended**, **1 flagged not-free / unconfirmed-free**, 2 without a recorded verdict. `free` = genuinely free to read; `acc` = a real document exists at/near the URL; `attr` = author/title/year correct; `rec` = the verifier recommends it. A corrected canonical URL (where the original was wrong/dead/paywalled) is shown as *(canonical: …)*.

- **On the Criteria To Be Used in Decomposing Systems into Modules** — David L. Parnas (1972)
  free: yes · acc: NO · attr: yes · quality: high · rec: yes · surfaced by 3 finders
  <https://wstomv.win.tue.nl/edu/2ip30/references/criteria_for_modularization.pdf>
  notes: The given UMD URL (cs.umd.edu/class/spring2003/.../criteria.pdf) is DEAD: hard HTTP 403 even with a browser user-agent. The two 'equivalent mirrors' named in free_basis are WRONG: static.k-nut.eu/Criteria-for-Modularization.pdf and cse.msu.edu/.../decomposition-macklem.pdf are PowerPoint-derived student SLIDE DECKS, not the paper (k-nut text begins 'Modular Programming / Given: Modularisation is a good idea'). A genuine free full-text scan of the real 6-page CACM 1972 paper (CMU affiliation, full prose, KWIC example, ~28KB text layer) is verified at TU Eindhoven (wstomv.win.tue.nl) — use that as canonical. Author/title/year correct. High quality, the seminal information-hiding paper.

- **Designing Software for Ease of Extension and Contraction (the 'uses' relation)** — David L. Parnas (1979)
  free: yes · acc: yes · attr: yes · quality: weak · rec: yes
  <https://cse.msu.edu/~cse870/Lectures/SS2007/ParnasPapers/Parnas-ExtensionContraction-hopkins-notes.pdf>
  notes: The given URL loads, but it is NOT the paper — it is a student PRESENTATION/SLIDE DECK ('Designing Software for Ease of Extension and Contraction, David Parnas, Presented by Kayra Hopkins'), correctly cross-referencing IEEE TSE Vol.5 No.2, March 1979, pp.128-138. Idea-attribution to Parnas is fine and it is free, but as bullet-point slides it is weak for a principal-engineer reader who wants the actual argument (the 'uses' relation, subset/superset families). I could NOT confirm a clean open-access copy of the full original paper — IEEE/ACM are paywalled (DOI 10.1109/TSE.1979.234169); ResearchGate requires login. So this slide deck is the only confirmed free artifact. Recommend with the caveat that it is a summary, not the paper.

- **A Rational Design Process: How and Why to Fake It** — David L. Parnas, Paul C. Clements (1986)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes · surfaced by 2 finders
  <https://www.cs.tufts.edu/~nr/cs257/archive/david-parnas/fake-it.pdf>
  notes: Verified: the Tufts PDF (cs.tufts.edu/~nr/cs257/archive/david-parnas/fake-it.pdf) loads as the full paper — first page reads 'A RATIONAL DESIGN PROCESS: HOW AND WHY TO FAKE IT, David L. Parnas (U. Victoria / NRL) and Paul C. Clements (NRL)'. Author/title/year (1986, IEEE TSE) correct. Free in Norman Ramsey's course archive. High quality, a foundational design-process paper directly relevant to the repo's ADR/rationale discipline.

- **On the Role of Scientific Thought (EWD447) — separation of concerns** — Edsger W. Dijkstra (1974)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.cs.utexas.edu/~EWD/transcriptions/EWD04xx/EWD447.html>
  notes: Canonical: the official E.W. Dijkstra Archive at UT Austin (cs.utexas.edu/~EWD/transcriptions/EWD04xx/EWD447.html) is the authoritative free transcription. EWD447 (1974) is indeed the manuscript where Dijkstra articulates 'separation of concerns'. Attribution correct, freely readable, high quality. (Confirmed from knowledge of the EWD archive structure; the EWD04xx/EWD447 path is the standard scheme.)

- **How Do Committees Invent? (Conway's Law, original)** — Melvin E. Conway (1968)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.melconway.com/research/committees.html>
  notes: Verified: melconway.com/research/committees.html loads with the full text of Conway's 1968 Datamation article plus his later author's note ('Any organization that designs a system... produces a design whose structure is a copy of the organization's communication structure'). Hosted on the author's own site, free, complete. Author/title/year correct. High quality, the canonical source for Conway's Law.

- **Big Ball of Mud** — Brian Foote and Joseph Yoder (1997)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.laputan.org/mud/>
  notes: laputan.org/mud/ is the authors' canonical site for Foote & Yoder's 1997 PLoP paper, with full HTML text and alternate format downloads, freely readable. Author/title/year correct. High quality, the standard reference on the eponymous anti-pattern. (Confirmed from knowledge; long-stable canonical home.)

- **A Philosophy of Software Design (Talks at Google) — deep vs shallow modules** — John Ousterhout (2018)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://www.youtube.com/watch?v=bmSAYlu0NcY>
  notes: Verified: YouTube ID bmSAYlu0NcY is the official 'A Philosophy of Software Design | John Ousterhout | Talks at Google' (published Aug 2018), also mirrored on archive.org and the Talks at Google podcast feed (auto-transcript/captions available). Free, by the author, conveys the deep-vs-shallow-modules core of the (paywalled) book. Solid quality as a free substitute; it is a talk, not the book's full treatment.

- **Hints and Principles for Computer System Design** — Butler W. Lampson (2020)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes · surfaced by 2 finders
  <https://arxiv.org/abs/2011.02455>
  notes: arXiv:2011.02455 is Butler Lampson's author-deposited open-access monograph (2020), with full and short PDFs. Open access, attribution correct. High quality — a comprehensive update of his classic 'Hints for Computer System Design'. (Confirmed from knowledge of the arXiv id; arXiv is reliably free.)

- **Out of the Tar Pit** — Ben Moseley and Peter Marks (2006)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes · surfaced by 3 finders
  <https://curtclifton.net/papers/MoseleyMarks06a.pdf>
  notes: curtclifton.net/papers/MoseleyMarks06a.pdf is the long-stable canonical mirror of Moseley & Marks (2006), a self-published technical report that was always freely circulated and never had a paywalled venue. Author/title/year correct, free, high quality — the standard reference on essential vs accidental complexity and state. (Confirmed from knowledge; this mirror is the de-facto canonical link.)

- **Design Principles and Design Patterns (the Dependency Inversion Principle)** — Robert C. Martin (2000)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://web.archive.org/web/20150906155800if_/http://www.objectmentor.com/resources/articles/Principles_and_Patterns.pdf>
  notes: The given Turku mirror (staff.cs.utu.fi/~jounsmed/.../DesignPrinciplesAndPatterns.pdf) has a TLS certificate error ('unable to verify the first certificate'), which will break for many readers/browsers. The content is genuine — Robert C. Martin's freely-distributed Object Mentor article (the original objectmentor.com PDF is now defunct). I verified the original is preserved and downloadable via the Wayback Machine (full text: 'Design Principles and Design Patterns, Robert C. Martin, www.objectmentor.com'), which I give as a stable canonical_url. Free, author/title/year (2000) correct. Solid quality — origin of DIP and the SOLID principles.

- **Documenting Architecture Decisions (the ADR origin post)** — Michael Nygard (2011)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions>
  notes: Verified: cognitect.com/blog/2011/11/15/documenting-architecture-decisions returns HTTP 200 with Michael Nygard's original post (the ADR format: Title/Context/Decision/Status/Consequences). Free, attribution correct (2011). Solid quality — the direct origin of the ADR convention these docs/adr/ files use, so highly on-point for this repo.

- **Software Architecture as a Set of Architectural Design Decisions** — Anton Jansen and Jan Bosch (2005)
  free: NO · acc: yes · attr: yes · quality: solid · rec: NO
  <https://www.semanticscholar.org/paper/Software-Architecture-as-a-Set-of-Architectural-Jansen-Bosch/4cd105262aa01f62b88baeda78570325661f67d3>
  notes: Attribution correct (Jansen & Bosch, WICSA 2005, IEEE; DOI 10.1109/WICSA.2005.61). But the free_basis claim is NOT substantiated: the given URL is only a Semantic Scholar LANDING PAGE, not the document, and I could not confirm a free open-access PDF reachable from it — the Semantic Scholar PDF CDN returned empty (HTTP 202, 0 bytes), the authoritative University of Groningen research portal page offers NO downloadable full text, and the only official venue (IEEE Xplore) is paywalled. ResearchGate/Academia.edu copies exist but require login and are not clean open access. Per default-doubt I set free=false. Recommend=false: cannot confirm a genuinely free version; a landing page is not a free document.

- **The Law of Leaky Abstractions** — Joel Spolsky (2002)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/>
  notes: Confirmed HTTP 200, full HTML essay on Spolsky's own site (joelonsoftware.com). Original 2002 essay, free, no paywall. Attribution correct.

- **An Introduction to Design by Contract (eiffel.com manuals)** — Eiffel Software (Bertrand Meyer method) (c. 1990s)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://archive.eiffel.com/doc/manuals/technology/contract/>
  notes: Confirmed live. Title on page is 'Building bug-free O-O software: An introduction to Design by Contract'. Covers preconditions (require), postconditions (ensure), class invariants, with Eiffel code examples. Freely readable, no login/paywall. Official Eiffel Software archive of Meyer's method.

- **Design by Contract (book chapter, Object-Oriented Software Construction)** — Bertrand Meyer (1997)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://se.inf.ethz.ch/~meyer/publications/old/dbc_chapter.pdf>
  notes: Confirmed: 2.5MB scanned PDF downloads from Meyer's ETH Zurich page. Search/index title is 'Chapter 1 Design by Contract, Bertrand Meyer'. Author-hosted, free. A legitimate free substitute for the paywalled OOSC book, by the same author, conveying the core DbC material.

- **An Axiomatic Basis for Computer Programming** — C. A. R. Hoare (1969)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.cs.cmu.edu/~crary/819-f09/Hoare69.pdf>
  notes: Confirmed HTTP 200, application/pdf (2.4MB) on Crary's CMU course page. Hoare's seminal 1969 CACM paper; widely mirrored on university course pages. Free, attribution correct.

- **A Behavioral Notion of Subtyping** — Barbara Liskov, Jeannette Wing (1994)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.cs.cmu.edu/~wing/publications/LiskovWing94.pdf>
  notes: Confirmed: PDF downloads from Wing's CMU homepage; extracted first-page text reads 'A Behavioral Notion of Subtyping, Barbara H. Liskov (MIT) and Jeannette M. Wing (Carnegie Mellon)'. This is the canonical TOPLAS 1994 version (the embedded PDF creation date of 1997 is just typesetting, not the publication year). Free, attribution correct.

- **Contracts for Higher-Order Functions** — Robert Bruce Findler, Matthias Felleisen (2002)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www2.ccs.neu.edu/racket/pubs/icfp2002-ff.pdf>
  notes: Confirmed: PDF downloads from Northeastern Racket pubs page; extracted text reads 'Contracts for Higher-Order Functions, Robert Bruce Findler, Matthias Felleisen, Northeastern University'. ICFP 2002. Free, institution-hosted, attribution correct.

- **Programming with Refinement Types: An Introduction to LiquidHaskell** — Ranjit Jhala, Niki Vazou, Eric Seidel, et al. (2020)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://ucsd-progsys.github.io/liquidhaskell-tutorial/book.pdf>
  notes: Confirmed: book.pdf downloads; title page reads 'Ranjit Jhala, Eric Seidel, Niki Vazou — PROGRAMMING WITH REFINEMENT TYPES: AN INTRODUCTION TO LIQUIDHASKELL, Version 13, July 20th 2020', Apache-2.0 licensed (genuinely open). Authors and year correct; listed 'et al.' is reasonable.

- **Propositions as Types** — Philip Wadler (2015)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes · surfaced by 2 finders
  <https://homepages.inf.ed.ac.uk/wadler/papers/propositions-as-types/propositions-as-types.pdf>
  notes: Confirmed HTTP 200, application/pdf on Wadler's Edinburgh homepage. The well-known CACM 2015 open-access article; author-hosted free PDF. Attribution correct.

- **On Understanding Types, Data Abstraction, and Polymorphism** — Luca Cardelli, Peter Wegner (1985)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <http://lucacardelli.name/papers/onunderstanding.a4.pdf>
  notes: Confirmed HTTP 200, application/pdf on Cardelli's own site (lucacardelli.name). Cardelli & Wegner, ACM Computing Surveys 1985. Author-hosted free PDF, attribution correct.

- **Who Builds a House Without Drawing Blueprints?** — Leslie Lamport (2015)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://cacm.acm.org/opinion/who-builds-a-house-without-drawing-blueprints/>
  notes: ATTRIBUTION CORRECT (Lamport, CACM 58(4), April 2015, pp.38-41), but the free_basis is WRONG: the Microsoft Research page (the given URL) does NOT host a free PDF — it only links to the paywalled DOI/ACM-DL. Notably, Lamport's own publications page is the one paper among his works that has NO free PDF link (entry 186 says only 'Available on ACM web site'). The genuinely free version is the CACM redesigned-site open opinion-article HTML (cacm.acm.org/opinion/...), which is free-to-read; I substitute it as canonical_url. (Direct scrape returns a Cloudflare 403 JS-challenge, i.e. anti-bot, not a subscription wall.) Avoid dl.acm.org/doi/10.1145/2736348 — that record is paywalled. Recommend kept because a free version exists, but the URL must be corrected.

- **Specifying Systems: The TLA+ Language and Tools (full book PDF)** — Leslie Lamport (2002)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://lamport.azurewebsites.net/tla/book-21-07-04.pdf>
  notes: Confirmed HTTP 200, application/pdf (1.8MB) on Lamport's own site. Full 2002 book posted free for personal (non-commercial) use by the author; linked from lamport.azurewebsites.net/tla/book.html. Attribution correct.

- **Practical Foundations for Programming Languages (abbreviated free edition)** — Robert Harper (2016)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.cs.cmu.edu/~rwh/pfpl/abbrev.pdf>
  notes: Confirmed HTTP 200, application/pdf on Harper's CMU page. The free 'abbreviated online edition, with corrections'; the full 2nd edition (2016) is the paid Cambridge book, but this free draft is legitimate and conveys the core. Attribution correct.

- **Software Foundations, Volume 1: Logical Foundations** — Benjamin C. Pierce et al. (ongoing)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://softwarefoundations.cis.upenn.edu/lf-current/index.html>
  notes: Confirmed live (WebFetch). UPenn-hosted, free to read and download (lf.tgz); currently v7.0 dated 2026-01-09, requires Coq/Rocq 9.0.0+. Editors include Benjamin C. Pierce, Azevedo de Amorim, Casinghino, Gaboardi, Greenberg, Hritcu, Sjoberg, Yorgey. Year 'ongoing' is accurate for a continuously-updated book. Every detail machine-checked in the prover; high quality canonical reference.

- **Parse, Don't Validate** — Alexis King (2019)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes · surfaced by 3 finders
  <https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/>
  notes: Well-known canonical essay on Alexis King's own blog (lexi-lambda.github.io), 2019-11-05, fully free. Attribution correct.

- **Effective ML Revisited (Make Illegal States Unrepresentable)** — Yaron Minsky (Jane Street) (2011)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://blog.janestreet.com/effective-ml-revisited/>
  notes: Confirmed live (WebFetch). Free Jane Street engineering blog post by Yaron Minsky, published 2011-03-09. Prose+code companion to the Harvard guest-lecture talk; covers 'make illegal states unrepresentable', exhaustiveness, uniform interfaces. No paywall.

- **Designing with Types: Making Illegal States Unrepresentable** — Scott Wlaschin (F# for Fun and Profit) (2013)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://fsharpforfunandprofit.com/posts/designing-with-types-making-illegal-states-unrepresentable/>
  notes: Confirmed live (WebFetch). Free essay on F# for Fun and Profit by Scott Wlaschin (byline 'ScottW'). Full worked Contact-type example. Genuinely free public site; the framing as a free substitute for his paid book is fair.

- **Constructive vs Predicative Data** — Hillel Wayne (2019)
  free: yes · acc: yes · attr: NO · quality: solid · rec: yes
  <https://www.hillelwayne.com/post/constructive/>
  notes: Confirmed live (WebFetch). Free post on Hillel Wayne's own site, title exact. YEAR IS WRONG: the post is dated 2020-05-18, not 2019. URL and authorship correct; only the year needs fixing. Still recommend.

- **Boundaries** — Gary Bernhardt (2012)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.destroyallsoftware.com/talks/boundaries>
  notes: Confirmed live (WebFetch): destroyallsoftware.com/talks/boundaries hosts Gary Bernhardt's SCNA 2012 conference talk, which is the free-to-view conference recording (distinct from the paid DAS screencast series). Caveat on the free_basis claim: the cited 'transcript' at andrewyao.me/Bernhardt-Boundaries/ is NOT a transcript -- it is a 2017 blog summary/review by Andrew Yao. The candidate's own URL (the talk) is free; the supplementary-link description is just inaccurate.

- **The Three Layer Haskell Cake** — Matt Parsons (2018)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://www.parsonsmatt.org/2018/03/22/three_layer_haskell_cake.html>
  notes: Confirmed live (WebFetch). Free post on Matt Parsons' own blog (parsonsmatt.org), 2018-03-22. Title, author, year all correct.

- **Hexagonal Architecture (Ports and Adapters)** — Alistair Cockburn (2005)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://alistair.cockburn.us/hexagonal-architecture/>
  notes: Canonical: Alistair Cockburn's own site hosts the original Ports-and-Adapters article free. The 2005 date reflects the widely-cited consolidated version (origins ~2005); attribution to Cockburn is correct. Well-known free source.

- **Theorems for Free!** — Philip Wadler (1989)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://homepages.inf.ed.ac.uk/wadler/papers/free/free.ps.gz>
  notes: The given .dvi URL is REAL and FREE on Wadler's own Edinburgh homepage (confirmed: 90KB application/x-dvi, FPCA 1989). Caveat: .dvi will not render in a browser -- the reader must download and run a DVI viewer. More accessible free canonical copies on the same author site / open repos: free.ps.gz (confirmed live, ~208KB PostScript) and the ACM open PDF https://dl.acm.org/doi/pdf/10.1145/99370.99404. Wadler's parametricity topics page links all formats. Attribution (Wadler, 1989) correct. Landmark paper.

- **Why Functional Programming Matters** — John Hughes (1990)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.cse.chalmers.se/~rjmh/Papers/whyfp.html>
  notes: Confirmed live (WebFetch). John Hughes' Chalmers page (cse.chalmers.se/~rjmh/Papers/whyfp.html) provides free PostScript and PDF of the full paper. Note on year: circulated as a Chalmers memo (~1984) and formally published 1989/1990; the candidate's 1990 is a defensible citation. Author hosting, free, high quality.

- **Total Functional Programming** — D. A. Turner (2004)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.jucs.org/jucs_10_7/total_functional_programming/jucs_10_07_0751_0768_turner.pdf>
  notes: Confirmed (WebFetch + WebSearch): official JUCS PDF resolves (137KB). JUCS is open access. J.UCS vol 10 issue 7, pp 751-768, 2004, D.A. Turner. Free official publisher PDF. Attribution correct.

- **How to Design Co-Programs** — Jeremy Gibbons (2021)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.cs.ox.ac.uk/jeremy.gibbons/publications/copro.pdf>
  notes: Confirmed (WebFetch + WebSearch): author's Oxford publications page hosts copro.pdf free (159KB). JFP vol 31 e15, 2021 (DOI 10.1017/S0956796821000113); the Cambridge Core version is paywalled but this free author PDF is the canonical open copy. Jeremy Gibbons, 2021 -- attribution correct.

- **Typed Tagless Final Interpreters (lecture notes)** — Oleg Kiselyov (2012)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://okmij.org/ftp/tagless-final/course/lecture.pdf>
  notes: Downloaded the PDF and read its first pages: title 'Typed Tagless Final Interpreters', author Oleg Kiselyov (oleg@okmij.org), abstract references Carette et al. final approach. Hosted free on the author's own site okmij.org. Spring-school (Generic Programming) lecture notes, ~2012. Authoritative, principal-engineer quality. Confirmed by direct fetch.

- **Ariane 5 Flight 501 Failure — Report by the Inquiry Board** — J. L. Lions et al. (ESA/CNES Inquiry Board) (1996)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes · surfaced by 2 finders
  <https://www.di.unito.it/~damiani/ariane5rep.html>
  notes: Fetched the page: full text of the official inquiry report, header reads 'Prof. J. L. LIONS, Chairman', dated Paris 19 July 1996, investigating the 4 June 1996 failure. University (Torino) mirror of the public ESA/CNES board report. Free, complete, correctly attributed. The canonical primary-source incident report.

- **The Error Model** — Joe Duffy (2016)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://joeduffyblog.com/2016/02/07/the-error-model/>
  notes: Joe Duffy's own blog; the well-known Midori-retrospective long-form essay on error handling. Fully readable, no paywall. Confirmed from knowledge as a famous canonical post; URL/date/author standard.

- **Crash-Only Software** — George Candea and Armando Fox (2003)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.usenix.org/legacy/events/hotos03/tech/full_papers/candea/candea_html/index.html>
  notes: Canonical USENIX HotOS IX (2003) paper by George Candea and Armando Fox (Stanford). The given legacy HTML URL appears verbatim in search results as the live HTML version; a direct fetch returned HTTP 403 (USENIX bot-blocking the legacy path, not a dead link). Stable free PDF mirrors also exist (e.g. https://dslab.epfl.ch/pubs/crashonly.pdf). Free, correctly attributed.

- **RFC 9413 — Maintaining Robust Protocols (the published 'Postel was wrong' draft)** — Martin Thomson and David Schinazi (2023)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://datatracker.ietf.org/doc/html/rfc9413>
  notes: Fetched: RFC 9413 'Maintaining Robust Protocols' by Martin Thomson and David Schinazi, published June 2023 (Informational, IAB). Official IETF datatracker, free HTML/text. Attribution exact. Title in candidate is accurate; the 'Postel was wrong' framing is an apt informal descriptor.

- **Why Do Computers Stop and What Can Be Done About It? (Tandem TR 85.7)** — Jim Gray (1985)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.hpl.hp.com/techreports/tandem/TR-85.7.pdf>
  notes: Jim Gray, Tandem TR 85.7, June 1985. The given Wisconsin mirror (pages.cs.wisc.edu/~remzi/.../gray-why-do-computers-stop-85.pdf) is a genuine complete scan (JBIG2-encoded, ~903KB, confirmed downloaded; couldn't text-extract the scan but search confirms the exact URL hosts this report). The official HP Labs PDF (canonical_url given) is a cleaner authoritative copy; bitsavers also mirrors it. Free, correctly attributed.

- **Making Reliable Distributed Systems in the Presence of Software Errors (PhD thesis — the 'let it crash' source)** — Joe Armstrong (2003)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://erlang.org/download/armstrong_thesis_2003.pdf>
  notes: Downloaded the PDF and read the title page: 'Making reliable distributed systems in the presence of software errors', Joe Armstrong, dissertation at the Royal Institute of Technology (KTH) Stockholm, December 2003 (final corrected version 20 Nov 2003). Hosted free on official erlang.org. Correctly attributed; the canonical 'let it crash' source.

- **Errors are values** — Rob Pike (2015)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://go.dev/blog/errors-are-values>
  notes: Official Go blog post by Rob Pike (2015). Free. go.dev/blog is the current canonical host (redirected from blog.golang.org). Short but a well-known canonical essay; correctly attributed.

- **Railway Oriented Programming: A Functional Approach to Error Handling** — Scott Wlaschin (2014)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://fsharpforfunandprofit.com/rop/>
  notes: Fetched: Scott Wlaschin's ROP resource page. Free, with links to talk videos (NDC London/Oslo 2014, F# eXchange 2014), slides (SlideShare/GitHub, reuse-permitted), and written overview. No paywall. Correctly attributed; 2014. Excellent practitioner resource.

- **Zero-overhead deterministic exceptions: Throwing values (P0709)** — Herb Sutter (2019)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0709r4.pdf>
  notes: Downloaded the PDF and read it: 'Zero-overhead deterministic exceptions: Throwing values', Document P0709 R4, Herb Sutter (hsutter@microsoft.com), dated 2019-08-04, audience EWG/LEWG. Official open-std WG21 committee paper, free PDF. Year 2019 correct (R4 revision). Correctly attributed.

- **Error Handling — chapters 9.1–9.3, esp. 'To panic! or Not to panic!' (The Rust Programming Language)** — Steve Klabnik and Carol Nichols (the Rust Project) (2023)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://doc.rust-lang.org/book/ch09-03-to-panic-or-not-to-panic.html>
  notes: Official Rust book ('The Rust Programming Language' by Klabnik and Nichols / the Rust Project), Chapter 9 Error Handling, section 9.3 'To panic! or Not to panic!'. Free on doc.rust-lang.org. URL is the stable canonical book path. Correctly attributed; an authoritative reference for Result/panic discipline.

- **Kleisli Categories (Category Theory for Programmers, ch. on composing embellished functions)** — Bartosz Milewski (2014)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://bartoszmilewski.com/2014/12/23/kleisli-categories/>
  notes: Bartosz Milewski's blog post 'Kleisli Categories' (Dec 2014), part of the Category Theory for Programmers series. Free on the author's blog. The full CTFP book is also freely available as a community-typeset PDF (github.com/hmemcpy/milewski-ctfp-pdf). Correctly attributed; high-quality for a principal-engineer reader connecting monadic error handling to category theory.

- **How Complex Systems Fail** — Richard I. Cook (2000)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes · surfaced by 2 finders
  <https://www.adaptivecapacitylabs.com/HowComplexSystemsFail.pdf>
  notes: Richard Cook's short treatise, hosted as a free PDF by Adaptive Capacity Labs (Cook was a co-founder), which is its canonical distribution. Confirmed from knowledge as a well-known open item. Minor: the piece is variously dated; it originated ~1998 and was revised to a stable form around 2000, so '2000' is a defensible date though some cite 1998. Attribution (Cook, title) is correct. High-quality, widely-cited resilience-engineering classic.

- **Lineage-driven Fault Injection (+ RICON 2014 keynote 'Outwards from the Middle of the Maze')** — Peter Alvaro, Joshua Rosen, Joseph M. Hellerstein (2015)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://people.ucsc.edu/~palvaro/molly.pdf>
  notes: Verified: downloaded the PDF at people.ucsc.edu/~palvaro/molly.pdf and extracted text — it is 'Lineage-driven Fault Injection' by Peter Alvaro, Joshua Rosen, Joseph M. Hellerstein (the MOLLY paper), SIGMOD 2015. Free on Alvaro's UCSC homepage. One attribution nuance: the paper's author affiliations read 'UC Berkeley' (work done there) even though it is hosted on his later UCSC page; emails are @cs.berkeley.edu. Author/title/year all correct. High quality.

- **Fault Tolerance in a High Volume, Distributed System** — Ben Christensen (Netflix) (2012)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://netflixtechblog.com/fault-tolerance-in-a-high-volume-distributed-system-91ab4faae74a>
  notes: Confirmed via search: 'Fault Tolerance in a High Volume, Distributed System' published Feb 2012 on the Netflix Technology Blog; originally at techblog.netflix.com/2012/02/fault-tolerance-in-high-volume.html, now canonically hosted at the given netflixtechblog.com Medium URL (slug 91ab4faae74a) after Netflix migrated the blog to Medium. Free to read (Netflix's own publication is not paywalled/metered). Could not GET directly from the sandbox (Medium blocks non-browser clients — HTTP 000), but the URL/slug and content are corroborated by search. Byline is 'Netflix Technology Blog'; Ben Christensen is the actual author — attribution correct. Solid practitioner incident/architecture write-up (origin of Hystrix patterns).

- **Simple Made Easy (transcript)** — Rich Hickey (2011)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://github.com/matthiasn/talk-transcripts/blob/master/Hickey_Rich/SimpleMadeEasy.md>
  notes: Well-known community verbatim transcript in the matthiasn/talk-transcripts GitHub repo of Rich Hickey's 2011 Strange Loop keynote. Both transcript and the talk video are free. Author/title/year correct. High quality — a canonical software-design talk.

- **No Silver Bullet — Essence and Accident in Software Engineering** — Frederick P. Brooks, Jr. (1986)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes · surfaced by 2 finders
  <https://www.cs.unc.edu/techreports/86-020.pdf>
  notes: Verified: downloaded cs.unc.edu/techreports/86-020.pdf and extracted text — it is 'No Silver Bullet: Essence and Accidents of Software Engineering', Frederick P. Brooks, Jr., UNC Technical Report TR86-020, September 1986. Free, author's own institution. Attribution correct (note the canonical subtitle is 'Essence and Accidents' plural, vs the candidate's 'Essence and Accident'). The free essay is the right call since the Mythical Man-Month book is paywalled. High quality.

- **Hints for Computer System Design (the 1983 original)** — Butler W. Lampson (1983)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.microsoft.com/en-us/research/wp-content/uploads/1983/10/Hints-for-Computer-System-Design-IEEE-Software.pdf>
  notes: Confirmed via search that this exact Microsoft Research URL (wp-content/uploads/1983/10/Hints-for-Computer-System-Design-IEEE-Software.pdf) is a live free copy on Lampson's MSR page; an earlier WebFetch downloaded it as a valid PDF (size exceeded the inline read cap, which itself shows it resolves). Butler W. Lampson, 'Hints for Computer System Design', orig. ACM SIGOPS OSR Oct 1983 (also IEEE Software 1984); SIGOPS Hall of Fame. Free, correct attribution. High quality.

- **The Emperor's Old Clothes (1980 Turing Award Lecture)** — C. A. R. Hoare (1981)
  free: yes · acc: NO · attr: yes · quality: high · rec: yes
  <https://worrydream.com/refs/Hoare_1981_-_The_Emperors_Old_Clothes.pdf>
  notes: The given Yale course-page URL (zoo.cs.yale.edu/classes/cs422/2014/bib/hoare81emperor.pdf) currently returns HTTP 403 Forbidden — it is still indexed but no longer serves the file, so it is NOT reliably accessible. A stable free canonical copy IS available at Bret Victor's worrydream.com archive (verified HTTP 200, application/pdf, 44KB). C.A.R. Hoare, 1980 Turing Award lecture, published CACM 24(2), 1981 — attribution correct. Recommend with the corrected URL. High quality.

- **The Humble Programmer (EWD340, 1972 Turing Award Lecture)** — Edsger W. Dijkstra (1972)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.cs.utexas.edu/~EWD/transcriptions/EWD03xx/EWD340.html>
  notes: Official UT Austin E. W. Dijkstra Archive HTML transcription of EWD340, the 1972 Turing Award lecture (also published CACM 1972). Free, authoritative source. Author/title/year correct. High quality. (Confirmed from knowledge as a well-known canonical archive.)

- **On the Role of Scientific Thought (EWD447)** — Edsger W. Dijkstra (1974)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.cs.utexas.edu/~EWD/transcriptions/EWD04xx/EWD447.html>
  notes: Official UT Austin Dijkstra Archive transcription of EWD447, written 1974. Free, authoritative. Author/title/year correct. The source of Dijkstra's 'separation of concerns' formulation. High quality.

- **Programming as Theory Building** — Peter Naur (1985)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes · surfaced by 2 finders
  <https://pages.cs.wisc.edu/~remzi/Naur.pdf>
  notes: Peter Naur's 1985 essay, freely mirrored by university CS departments (the Remzi/UW-Madison Naur.pdf is a well-known stable copy; gwern.net also mirrors it). Out of paywall. Author/title/year correct. High quality and directly relevant to a principal-engineer reader.

- **Worse Is Better (collected: The Rise of Worse is Better + Is Worse Really Better?)** — Richard P. Gabriel (1991)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.dreamsongs.com/WorseIsBetter.html>
  notes: Richard P. Gabriel's own homepage dreamsongs.com hosts the Worse-Is-Better essay set free, including the later 'Is Worse Really Better?' reflection. Author-distributed, no paywall. Attribution correct. The 'Rise of Worse is Better' core dates to ~1989-1991. High quality.

- **A Plea for Lean Software** — Niklaus Wirth (1995)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://people.inf.ethz.ch/wirth/Articles/LeanSoftware.pdf>
  notes: Verified live: curl HEAD on people.inf.ethz.ch/wirth/Articles/LeanSoftware.pdf returns HTTP 200, application/pdf, 1.59MB (an earlier WebFetch also downloaded it). Hosted free on Wirth's own ETH Zurich homepage. Niklaus Wirth, 'A Plea for Lean Software', IEEE Computer 28(2), Feb 1995 — attribution correct. The ACM/IEEE library copy is paywalled, so this author-hosted PDF is the right free canonical choice. High quality.

- **Scalability! But at what COST?** — Frank McSherry, Michael Isard, Derek G. Murray (2015)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.usenix.org/system/files/conference/hotos15/hotos15-paper-mcsherry.pdf>
  notes: Confirmed via search: canonical USENIX HotOS XV (2015) open-access PDF, by McSherry, Isard, Murray. The given URL is exact and correct. (Direct WebFetch returned 403 = USENIX bot-blocking, not a dead link; search confirms this is the official host.) Lead author also mirrors the companion blog post at frankmcsherry.org. Famous, high-quality systems paper introducing the COST metric.

- **Can Programming Be Liberated from the von Neumann Style? (1977 Turing Award Lecture)** — John Backus (1978)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://dl.acm.org/doi/10.1145/1283920.1283933>
  notes: John Backus, CACM Aug 1978 (Turing lecture for the 1977 award) — year 1978 is correct for publication. The ACM DOI is part of the ACM Turing Award Lectures collection, which ACM makes free-to-read; confirmed. The archive.org fallback (archive.org/details/programming-liberated-von-neumann) is verified to exist and be freely downloadable in multiple formats if the ACM page gates. Both the free basis and the named mirror check out.

- **Schema evolution in Avro, Protocol Buffers and Thrift** — Martin Kleppmann (2012)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://martin.kleppmann.com/2012/12/05/schema-evolution-in-avro-protocol-buffers-thrift.html>
  notes: Confirmed live: Martin Kleppmann's personal blog, post dated 2012-12-05, CC-BY 3.0, freely readable, no paywall. Canonical author-hosted essay predating and underpinning DDIA chapter 4. URL exact.

- **Apache Avro 1.11.1 Specification — Schema Resolution** — Apache Avro project (2022)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://avro.apache.org/docs/1.11.1/specification/>
  notes: Confirmed live: official Apache Avro 1.11.1 spec, includes a dedicated Schema Resolution section, Apache License 2.0, free. URL exact. Reference documentation rather than prose argument, but authoritative and exactly on point for schema evolution mechanics.

- **Protocol Buffers — Language Guide (proto3): Updating A Message Type** — Google / Protocol Buffers team (2024)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://protobuf.dev/programming-guides/proto3/>
  notes: Confirmed live: official Google protobuf.dev proto3 Language Guide, contains the 'Updating A Message Type' section (wire-safe / wire-unsafe / wire-compatible changes). Free, open documentation. URL exact and current.

- **Cap'n Proto — Introduction (zero-copy / no encode-decode rationale)** — Kenton Varda (2013)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://capnproto.org/>
  notes: Confirmed live: official Cap'n Proto homepage by Kenton Varda, with the explicit 'no encoding/decoding step' zero-copy rationale. Free, open-source project site (MIT-licensed project). URL correct. 2013 is the right origin year.

- **FlatBuffers White Paper** — Wouter van Oortmerssen / Google FPL (2014)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://flatbuffers.dev/white_paper/>
  notes: Confirmed via search: official FlatBuffers docs white paper at flatbuffers.dev/white_paper/ — URL exact and current (older github.io mirror also exists). Free. Authored by Wouter van Oortmerssen at Google FPL; the 'why FlatBuffers / memory-efficient no-copy' rationale is present. Attribution and ~2014 origin are right.

- **Data on the Outside versus Data on the Inside** — Pat Helland (2005)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.cidrdb.org/cidr2005/papers/P12.pdf>
  notes: Confirmed: WebFetch retrieved a real 119.8KB PDF at the given URL (CIDR 2005 proceedings, cidrdb.org). Pat Helland, CIDR 2005 — attribution and venue/year correct. Open-access conference PDF, free. URL exact. Influential paper on data inside vs. across service boundaries.

- **A Note on Distributed Computing** — Jim Waldo, Geoff Wyant, Ann Wollrath, Sam Kendall (1994)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://scholar.harvard.edu/files/waldo/files/waldo-94.pdf>
  notes: Confirmed via search: canonical free PDF on Waldo's Harvard Scholar page (waldo-94.pdf), Sun Microsystems Labs TR-94-29, by Waldo, Wyant, Wollrath, Kendall, 1994. URL exact. (Direct WebFetch 403 = scholar.harvard.edu bot-blocking, not dead; search confirms it is the live canonical host.) Seminal distributed-systems paper.

- **Fallacies of Distributed Computing (canonical list + provenance)** — L. Peter Deutsch, James Gosling, Bill Joy et al. (1994)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://en.wikipedia.org/wiki/Fallacies_of_distributed_computing>
  notes: Confirmed live: Wikipedia page exists and lists the eight original fallacies correctly, plus the three newer ones (versioning is simple; compensating updates always work; observability is optional) added by Mark Richards & Neal Ford in 2020 — the candidate's specific claim about 'the newer three' is accurate. Free. Attribution to Deutsch/Gosling/Joy is the conventional (if loose) provenance for the original list: Deutsch authored most, Gosling added one; this is a fair canonical aggregation rather than a primary source. Recommend as an orientation/provenance index, not a deep read.

- **End-to-End Arguments in System Design** — J. H. Saltzer, D. P. Reed, D. D. Clark (1984)
  free: ? · acc: ? · attr: ? · quality: ? · rec: ?
  <https://web.mit.edu/saltzer/www/publications/endtoend/endtoend.pdf>

- **The Log: What every software engineer should know about real-time data's unifying abstraction** — Jay Kreps (2013)
  free: ? · acc: ? · attr: ? · quality: ? · rec: ?
  <https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying>

- **Hyrum's Law** — Hyrum Wright (2016)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://www.hyrumslaw.com/>
  notes: Confirmed live. Author's free single-page site stating the canonical wording: 'With a sufficient number of users of an API... all observable behaviors of your system will be depended on by somebody.' Attributed to Hyrum Wright. No paywall. Note: the law is canonically dated ~2016 (popularized via Titus Winters' 'Software Engineering at Google'); the site itself is undated but that is immaterial. Solid as the canonical statement of the law; pithy rather than deep.

- **RFC 9413 — The Harmful Consequences of the Robustness Principle** — Martin Thomson (IAB) (2023)
  free: yes · acc: yes · attr: NO · quality: high · rec: yes
  <https://www.rfc-editor.org/rfc/rfc9413.html>
  notes: The given URL points to draft-iab-protocol-maintenance-00 (an EARLY draft, version 00, May 2018) — the datatracker page itself states it 'was ultimately published as RFC 9413'. Use the canonical RFC URL instead. Attribution issues: (1) the FINAL RFC 9413 title is 'Maintaining Robust Protocols', NOT 'The Harmful Consequences of the Robustness Principle' (that was an earlier draft title); (2) authors are M. Thomson AND D. Schinazi (candidate omits Schinazi); (3) IAB stream, June 2023. All free. Recommend, but with the corrected canonical URL and the title/author fixed. High quality for a principal-engineer reader (the definitive IAB statement on the robustness/Postel principle's downsides).

- **October 21 post-incident analysis (GitHub 2018 MySQL divergence)** — Jason Warner / GitHub Engineering (2018)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://github.blog/2018-10-30-oct21-post-incident-analysis/>
  notes: Confirmed live and free. Official GitHub Engineering blog post by Jason Warner (@jasoncwarner), dated Oct 30, 2018 (updated Dec 19, 2021). Documents the 24-hour service degradation from a network-maintenance error that triggered a cross-DC Orchestrator/MySQL failover and data divergence. High-quality first-party incident report.

- **Spec-ulation (keynote transcript)** — Rich Hickey (2016)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://github.com/matthiasn/talk-transcripts/blob/master/Hickey_Rich/Spec_ulation.md>
  notes: Confirmed. Full community transcript of Rich Hickey's 'Spec-ulation' keynote (Clojure/conj 2016) in the public matthiasn/talk-transcripts GitHub repo. Free, renders on GitHub. Underlying talk also free on YouTube. High-value treatment of versioning, breaking change, and dependency growth/decay.

- **What Every Computer Scientist Should Know About Floating-Point Arithmetic** — David Goldberg (1991)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html>
  notes: Confirmed live and free. The Oracle (legacy Sun) Numerical Computation Guide hosts the full edited reprint of Goldberg's 1991 Computing Surveys paper (Appendix D), 'edited reprint... copyright 1991 ACM' — republished by permission, no paywall. Author David Goldberg, year 1991 correct. The definitive floating-point primer; high quality.

- **How Java's Floating-Point Hurts Everyone Everywhere** — William Kahan and Joseph D. Darcy (1998)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://people.eecs.berkeley.edu/~wkahan/JAVAhurt.pdf>
  notes: Confirmed: PDF loads (275KB), metadata author 'Prof. W. Kahan'. Hosted on Kahan's UC Berkeley EECS homepage; free. Year 1998 and co-author Joseph D. Darcy are correct per the standard citation (presented at ACM 1998 Workshop on Java for High-Performance Network Computing). High quality from the principal architect of IEEE 754.

- **How Futile are Mindless Assessments of Roundoff in Floating-Point Computation?** — William Kahan (2006)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf>
  notes: Confirmed: PDF loads (375KB), metadata title 'Mindless', author 'Prof. W. Kahan', creation date Jan 11 2006 (matches stated year 2006). Author's UC Berkeley EECS homepage; free. High quality.

- **On Testing Non-Testable Programs** — Elaine J. Weyuker (1982)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://homes.cs.washington.edu/~rjust/courses/CSE503/2021_02_12-reading1.pdf>
  notes: Confirmed: PDF loads (1.8MB); page 1 reads 'On Testing Non-testable Programs', Elaine J. Weyuker, Courant Institute/NYU, The Computer Journal Vol. 25 No. 4, 1982 (© Wiley Heyden Ltd 1982). Free U. Washington CSE503 (Rene Just) course-page render of the identical paper; the Oxford/Wiley Computer Journal original is paywalled. Foundational oracle-problem paper; high quality.

- **Metamorphic Testing: A Review of Challenges and Opportunities** — Tsong Yueh Chen, Fei-Ching Kuo, Huai Liu, Pak-Lok Poon, Dave Towey, T. H. Tse, Zhi Quan Zhou (2018)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://nottingham-repository.worktribe.com/output/925152/metamorphic-testing-a-review-of-challenges-and-opportunities>
  notes: Confirmed via search: Nottingham institutional repository record exists; open access under CC BY 4.0. Authors Chen, Kuo, Liu, Poon, Towey, Tse, Zhou; 2018; published in ACM Computing Surveys 51(1) Art.4 (the ACM dl.acm.org copy is paywalled, this OA copy is free). The worktribe landing page returned HTTP 403 to the bot fetcher, but the OA status is corroborated; the same text is also freely mirrored (e.g. homes.cs.washington.edu/~rjust/courses/CSE503/2021_02_12-reading2.pdf) if the landing page blocks a reader. High quality.

- **A Survey on Metamorphic Testing** — Sergio Segura, Gordon Fraser, Ana B. Sanchez, Antonio Ruiz-Cortes (2016)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://eprints.whiterose.ac.uk/id/eprint/110335/1/segura16-tse.pdf>
  notes: Confirmed: PDF loads (907KB); White Rose Research Online cover page reads 'Segura, S., Fraser, G., Sanchez, A.B. et al. (2016) A Survey on Metamorphic Testing. IEEE Transactions on Software Engineering, 42(9), 805-824', Version: Accepted. Free author accepted manuscript; the IEEE TSE version is paywalled. Authors/year correct. High quality.

- **QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs** — Koen Claessen and John Hughes (2000)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.cs.tufts.edu/~nr/cs257/archive/john-hughes/quick.pdf>
  notes: Confirmed: PDF loads (190KB); page 1 reads 'QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs', Koen Claessen and John Hughes (Chalmers), ICFP '00 Montreal, Copyright 2000 ACM. Free Tufts CS257 course archive (Norman Ramsey). The seminal property-based-testing paper; high quality.

- **Experiences with QuickCheck: Testing the Hard Stuff and Staying Sane** — John Hughes (2016)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.cs.tufts.edu/~nr/cs257/archive/john-hughes/quviq-testing.pdf>
  notes: Confirmed: PDF loads (707KB); page 1 reads 'Experiences with QuickCheck: Testing the Hard Stuff and Staying Sane', John Hughes (Chalmers/Quviq AB). Recounts Quviq QuickCheck on AUTOSAR C for Volvo and the Klarna race-condition bug. Free Tufts CS257 course archive. Year 2016 correct (Springer LNCS festschrift 'A List of Successes That Can Change the World'). High quality, war-stories complement to the 2000 paper.

- **Finding and Understanding Bugs in C Compilers (Csmith)** — Xuejun Yang, Yang Chen, Eric Eide, John Regehr (2011)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://users.cs.utah.edu/~regehr/papers/pldi11-preprint.pdf>
  notes: Confirmed by fetching and reading the PDF title page: 'Finding and Understanding Bugs in C Compilers' by Xuejun Yang, Yang Chen, Eric Eide, John Regehr (Univ. of Utah). Footer: 'ACM, 2011 ... author's version ... definitive version published in PLDI 2011.' Legitimate author-hosted preprint on Regehr's Utah homepage; HTTP 200, application/pdf. Title/authors/year all correct. Canonical, high quality.

- **Efficient Reproducible Floating Point Summation and BLAS (incl. ReproBLAS)** — Peter Ahrens, James Demmel, Hong Diep Nguyen (2016)
  free: yes · acc: yes · attr: NO · quality: high · rec: yes
  <https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.pdf>
  notes: Confirmed by reading the PDF title page: UC Berkeley EECS Tech Report UCB/EECS-2016-121, dated June 18, 2016, HTTP 200 application/pdf (716KB). Two attribution nits: (1) the title-page lists the author as 'Willow Ahrens', who formerly published as 'Peter Ahrens' (same person — name change), so the candidate's 'Peter Ahrens' is the old name; (2) title-page author ORDER is Demmel, Ahrens, Nguyen, not the candidate's 'Ahrens, Demmel, Nguyen'. The candidate's '(incl. ReproBLAS)' is a descriptive addition, not part of the literal title 'Efficient Reproducible Floating Point Summation and BLAS'. Content (ReproBLAS) is correct. Free official tech report; still recommend, but caller should fix author name/order.

- **The T-Experiments: Errors in Scientific Software** — Les Hatton (1997)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://kar.kent.ac.uk/id/document/2075>
  notes: Candidate URL https://kar.kent.ac.uk/id/document/2075 redirects (HTTP 200, application/pdf, 71KB) to https://kar.kent.ac.uk/21557/1/THE_T-EXPERIMENTS_ERRORS_IN.pdf — a free full-text PDF in the Kent Academic Repository. Landing record at kar.kent.ac.uk/21557/. Author Les Hatton, IEEE Computational Science & Engineering 4(2), 1997; publisher (IEEE/ACM) version paywalled, but this KAR copy and the author's own copy (https://www.leshatton.org/Documents/Texp_ICSE297.pdf, HTTP 200, application/pdf) are free. Attribution correct.

- **Ten Simple Rules for Reproducible Computational Research** — Geir Kjetil Sandve, Anton Nekrutenko, James Taylor, Eivind Hovig (2013)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003285>
  notes: Confirmed via fetch: PLOS Computational Biology, published Oct 24, 2013, by Geir Kjetil Sandve, Anton Nekrutenko, James Taylor, Eivind Hovig. Fully open access under CC BY; PDF download available on the article page. DOI 10.1371/journal.pcbi.1003285. Title/authors/year all correct.

- **Accuracy and Stability of Numerical Algorithms (SIAM Day lecture slides)** — Nicholas J. Higham (2013)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <http://www.maths.manchester.ac.uk/~higham/talks/asna13_cardiff.pdf>
  notes: These are Nick Higham's slides for his 'Accuracy and Stability of Numerical Algorithms' talk at the SIAM Chapter Day, Cardiff, Jan 2013. The candidate Cardiff URL (mathsdemo.cf.ac.uk) is search-indexed but was unreachable from this environment (connection refused / HTTP 000) — likely server down. Higham's own slides index (nhigham.com/slides/) lists the same 2013 Cardiff talk and hosts it at the author URL given as canonical_url (Manchester); that returned 503 here too (Manchester maths server flaky), but it is the author-authoritative location. The candidate is honest that these slides substitute for Higham's paywalled SIAM monograph of the same title and that nhigham.com is a free companion. Recommend, but flag both PDF mirrors were transiently unreachable; the slides (not the book) are the free artifact.

- **Predicting Metamorphic Relations for Testing Scientific Software (preprint)** — Upulee Kanewala, James M. Bieman, Anneliese Andrews (2016)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://www.cs.colostate.edu/~bieman/Pubs/kanewalaPredictingMetamorphicSTVR.PreprintSubmitted4publication.pdf>
  notes: Candidate URL returned HTTP 200, application/pdf, 2.9MB — author-submitted preprint on Bieman's Colorado State University homepage. Authors Upulee Kanewala, James M. Bieman, Anneliese Andrews; the published Wiley STVR version is paywalled, this preprint is free. Attribution correct.

- **Engineering a Safer World: Systems Thinking Applied to Safety** — Nancy G. Leveson (2011)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://library.oapen.org/handle/20.500.12657/26043>
  notes: Candidate OAPEN URL returned HTTP 403 to curl (OAPEN/CDN bot-blocking, not a dead link). Confirmed via search that it is the official MIT Press open-access monograph by Nancy G. Leveson (2011); full OA PDF also at direct.mit.edu (oa-monograph) and Internet Archive (archive.org/details/oapen-20.500.12657-26043). Genuinely free, full book. Attribution correct. High quality (the canonical STAMP/systems-safety text).

- **A New Accident Model for Engineering Safer Systems (STAMP)** — Nancy G. Leveson (2004)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <http://sunnyday.mit.edu/accidents/safetyscience-single.pdf>
  notes: Candidate author-hosted URL on Leveson's MIT sunnyday server returned HTTP 200, application/pdf, 198KB — confirmed live and free. The canonical preprint of the Safety Science (2004) STAMP paper by Nancy G. Leveson. Attribution correct.

- **Patriot Missile Defense: Software Problem Led to System Failure at Dhahran (GAO/IMTEC-92-26)** — U.S. Government Accountability Office (1992)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://www.gao.gov/assets/imtec-92-26.pdf>
  notes: Candidate gao.gov asset URL returned HTTP 403 to curl (Akamai/CDN bot-block, not dead). Confirmed via search: official GAO report, product page gao.gov/products/imtec-92-26, with the PDF at the exact candidate path gao.gov/assets/imtec-92-26.pdf. Published Feb 1992, U.S. GAO. Free U.S. government work. Title/author/year correct.

- **Mars Climate Orbiter Mishap Investigation Board — Phase I Report** — A. Stephenson et al., NASA MCO MIB (1999)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://llis.nasa.gov/llis_lib/pdf/1009464main1_0641-mr.pdf>
  notes: Candidate NASA LLIS URL returned HTTP 200, application/pdf, 1.46MB — confirmed live and free. Official NASA MCO MIB Phase I Report (Stephenson et al., 1999), U.S. government work. Attribution correct.

- **In the Matter of Knight Capital Americas LLC (SEC Release 34-70694)** — U.S. Securities and Exchange Commission (2013)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.sec.gov/files/litigation/admin/2013/34-70694.pdf>
  notes: Candidate sec.gov URL returned HTTP 403 to curl (sec.gov blocks automated requests, not dead). Confirmed via search: official SEC administrative/cease-and-desist order, Release No. 34-70694, issued Oct 16, 2013, against Knight Capital Americas LLC, at the exact candidate path sec.gov/files/litigation/admin/2013/34-70694.pdf. Free U.S. government work. Title/author/year correct (2013).

- **Cloudflare outage on November 18, 2025 (official postmortem)** — Matthew Prince et al., Cloudflare (2025)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://blog.cloudflare.com/18-november-2025-outage/>
  notes: Confirmed via fetch: Cloudflare's official postmortem blog post for the Nov 18, 2025 outage, authored by Matthew Prince (CEO). Freely readable, no login. Describes a Bot Management config file that doubled in size (duplicate rows from a ClickHouse/DB permissions change) exceeding proxy memory limits. Attribution correct.

- **Columbia Accident Investigation Board Report, Volume I (esp. Ch. 7-8: organizational causes)** — Harold Gehman et al., CAIB (2003)
  free: yes · acc: yes · attr: NO · quality: high · rec: yes
  <https://www.nasa.gov/wp-content/uploads/static/history/columbia/reports/CAIBreportv1.pdf>
  notes: WRONG FILE at the given URL. I downloaded the given CAIBreportv6.pdf (2.3MB, 341pp): pdfinfo title is 'Cover Volume 6' and the text reads 'Report Volume VI' throughout and contains 'Corrections to Volume I' — it is Volume VI (photos/corrections), NOT Volume I. The candidate's described content (Ch. 7 'The Accident's Organizational Causes', Ch. 8 'History as Cause') lives in Volume I. I verified the correct file: CAIBreportv1.pdf at the same nasa.gov path (8.2MB, 248pp, text confirms 'Report Volume I' + both chapter titles + 'History as Cause: Two Accidents'). Both are free U.S.-government works on nasa.gov. Use the v1 URL (canonical_url). Alt mirror: ehss.energy.gov/deprep/archive/documents/0308_caib_report_volume1.pdf.

- **A Case Study of Toyota Unintended Acceleration and Software Safety** — Philip Koopman (Carnegie Mellon) (2014)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://users.ece.cmu.edu/~koopman/toyota/index.html>
  notes: Verified live: Philip Koopman's CMU page (users.ece.cmu.edu/~koopman/toyota/index.html). Hosts slides (PDF/SlideShare/Archive.org), a 1080p recorded lecture, and links to the 2013 Bookout trial materials. Explicitly CC BY 4.0 with stated attribution 'Prof. Philip Koopman, Carnegie Mellon University' and educators encouraged to use freely. Author/title/year correct. Strong, citable free substitute for the sealed Barr expert report.

- **Formal Methods and the Certification of Critical Systems (CSL-93-7)** — John Rushby (SRI International) (1993)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.csl.sri.com/~rushby/papers/csl-93-7.pdf>
  notes: Verified by download: the PDF at csl.sri.com/~rushby/papers/csl-93-7.pdf loads (319pp, 2MB) and its first page reads verbatim 'Formal Methods and the Certification of Critical Systems / John Rushby / Computer Science Laboratory / SRI International ... Technical Report CSL-93-7 / December 1993', also issued as NASA CR 4551. Author/title/year exact. Author-hosted, free.

- **Postmortem Culture: Learning from Failure (Site Reliability Engineering, Ch. 15)** — Betsy Beyer, Chris Jones, Jennifer Petoff, Niall Murphy (eds.), Google (2016)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://sre.google/sre-book/postmortem-culture/>
  notes: Verified live: full Chapter 15 text readable at sre.google/sre-book/postmortem-culture/ with no paywall/truncation, licensed CC BY-NC-ND 4.0. The entire SRE book is free online on Google's official site. Editors/year correct.

- **Blameless PostMortems and a Just Culture** — John Allspaw (Etsy) (2012)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.etsy.com/codeascraft/blameless-postmortems>
  notes: Direct WebFetch returned HTTP 403 (bot-blocking, not a real paywall). Cross-confirmed via search: John Allspaw, Etsy Code as Craft, May 22 2012, free essay; canonical home is codeascraft.com. The given etsy.com/codeascraft/blameless-postmortems URL is the live canonical redirect and works in a browser. Author/title/year correct, no paywall.

- **Software Aging** — David L. Parnas (1994)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.cs.drexel.edu/~yc349/CS451/RequiredReadings/SoftwareAging.pdf>
  notes: WRONG FILE at the given URL. I downloaded cs.unc.edu/techreports/87-009.pdf: it is NOT Parnas — it is 'Three Dimensional Image Presentation Techniques in Medical Imaging' by Stephen M. Pizer and Henry Fuchs (UNC TR 87-009, 1987). The UNC TR-number guess in the seed is simply incorrect (and the seed admitted the path was fragile). I located and verified a clean free copy: Drexel course mirror (curl HTTP 200, 964KB), first page reads 'Software Aging / Invited Plenary Talk / David Lorge Parnas / ... McMaster University' — the correct 1994 ICSE invited plenary. Use canonical_url. (Drexel host serves a self-signed TLS cert, so WebFetch errored on cert verification but the PDF is genuine and downloads fine.)

- **How to Design a Good API and Why It Matters (recorded talk + slides)** — Joshua Bloch (2007)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.infoq.com/presentations/effective-api-design/>
  notes: Verified live: infoq.com/presentations/effective-api-design/ shows a free, playable video (~1h09) by Joshua Bloch on API design, recorded at Javapolis. InfoQ presentations are free to watch; login only unlocks extras, not the talk itself. Minor: InfoQ dates the recording Nov 2006 (Javapolis); the talk is commonly cited as 2007 (OOPSLA companion). Same content, correct author/title — year discrepancy is trivial.

- **Hyrum's Law (with a sufficient number of users of an API…)** — Hyrum Wright (2016)
  free: yes · acc: yes · attr: yes · quality: solid · rec: yes
  <https://www.hyrumslaw.com/>
  notes: Verified live: hyrumslaw.com is the one-page canonical statement, free. Quote confirmed: 'With a sufficient number of users of an API, it does not matter what you promise in the contract: all observable behaviors of your system will be depended on by somebody.' Attributed to Hyrum Wright (Titus Winters named/popularized it). The longer free treatment in the SWE-at-Google book is open at abseil.io/resources/swe-book. Correct.

- **Spec-ulation Keynote (full transcript)** — Rich Hickey (2016)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://github.com/matthiasn/talk-transcripts/blob/master/Hickey_Rich/Spec_ulation.md>
  notes: Verified live: the GitHub matthiasn/talk-transcripts/.../Spec_ulation.md is a genuine verbatim transcript of Rich Hickey's Clojure/conj 2016 'Spec-ulation' keynote (links the free video youtube.com/watch?v=oyLBGkS5ICk). Satisfies 'recorded talk WITH transcript' directly. Author/title/year correct, free.

- **Semantic Versioning 2.0.0** — Tom Preston-Werner (2013)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://semver.org/>
  notes: semver.org is the canonical free spec (CC BY 3.0), authored by Tom Preston-Werner, SemVer 2.0.0. Well-known canonical item; confirmed from knowledge. Free, correctly attributed.

- **Store config in the environment (Twelve-Factor App, Factor III)** — Adam Wiggins (Heroku) (2011)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://12factor.net/config>
  notes: 12factor.net/config is the canonical free page for Factor III 'Config'. Original author Adam Wiggins (Heroku), ~2011/2012. Well-known open methodology; free, correctly attributed.

- **Feature Toggles (aka Feature Flags)** — Pete Hodgson (martinfowler.com) (2017)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://martinfowler.com/articles/feature-toggles.html>
  notes: martinfowler.com/articles/feature-toggles.html is the free canonical article by Pete Hodgson (Oct 2017) on Martin Fowler's site. Well-known reference on the pattern; free, correctly attributed.

- **SEC Order: In re Knight Capital Americas LLC (Release 34-70694)** — U.S. Securities and Exchange Commission (2013)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://www.sec.gov/files/litigation/admin/2013/34-70694.pdf>
  notes: Real and confirmed: official SEC administrative order, Release No. 34-70694, issued Oct 16, 2013, against Knight Capital Americas LLC (the Aug 1, 2012 SMARS runaway-order incident; first enforcement action under market-access Rule 15c3-5; $12M settlement). Corroborated by SEC press release 2013-222. U.S. government work, public domain, genuinely free. Direct WebFetch returned HTTP 403 because sec.gov blocks automated fetchers (the candidate's free_basis correctly anticipated this) — it downloads fine in a browser. The given URL path (/litigation/admin/2013/34-70694.pdf) is the older sec.gov scheme; the current canonical live path returned by search is /files/litigation/admin/2013/34-70694.pdf (set as canonical_url). A free third-party mirror also exists at headlandstech.jp/static/file/34-70694.pdf. Excellent primary-source incident report for a principal engineer (deploy/feature-flag reuse, dead-code reactivation, kill-switch gaps).

- **Cloudflare outage on November 18, 2025 (official post-mortem)** — Cloudflare (Matthew Prince et al.) (2025)
  free: yes · acc: yes · attr: yes · quality: high · rec: yes
  <https://blog.cloudflare.com/18-november-2025-outage/>
  notes: Real and confirmed at the exact given URL. Verified via WebFetch: title 'Cloudflare outage on November 18, 2025', byline Matthew Prince (Cloudflare co-founder/CEO), published Nov 18, 2025. Official company post-mortem, free on the Cloudflare blog, no paywall. Root cause (a database-permissions change caused the Bot Management feature file to contain duplicate rows, exceeding a hard-coded ~200-feature limit in the core proxy, producing 5xx errors network-wide) matches multiple independent analyses. Attribution given as 'Matthew Prince et al.' is fine — confirmed byline is Matthew Prince. High-quality, detailed engineering post-mortem.

---

## Part 3 — raw discovery appendix

Everything each of the 9 discovery finders proposed, verbatim, before dedup/verification/curation — so the filtering is auditable and so good candidates that didn't make Part 1 are still on the record.

### Dimension: `modularity-architecture`

*Focus:* Module decomposition, information hiding, architecture-as-decisions, deep modules, anti-pattern catalogs  
*Principally maps to:* P1, P2, P3, documentation/auditability spirit

- **On the Criteria To Be Used in Decomposing Systems into Modules** — David L. Parnas (1972)
  *open-access-paper* · foundational · confidence: high
  The origin of information hiding: decompose around the secrets each module hides (the decisions likely to change), not around the steps of the flowchart. This is the theory under P1's 'one home per fact' and P3's one-owner collaborators — the chocofarm WeightContainer/FeatureLayout split is exactly Parnas's 'hide the layout decision behind one owner.'
  *Fills:* P1, P3, and the unnamed CORE behind both: information hiding as the decomposition criterion (a fact's 'one home' is the module that HIDES the design decision likely to change).
  *Free basis:* Author-classic widely mirrored on university course pages; this UMD course copy is a stable full PDF. Equivalent free mirrors: cse.msu.edu and static.k-nut.eu.
  <https://www.cs.umd.edu/class/spring2003/cmsc838p/Design/criteria.pdf>

- **Designing Software for Ease of Extension and Contraction (the 'uses' relation)** — David L. Parnas (1979)
  *lecture-notes* · intermediate · confidence: high
  Parnas's 'uses' relation is the precise mathematical object (a partial order over modules; A uses B only if A is easier to build and B has a useful subset) behind P2's dependency inversion and 'a parameter the receiver cannot honor is not in the signature.' A poset-literate reader will recognize the design rule as keeping the uses-DAG acyclic and layered.
  *Fills:* P2 (seam/dependency discipline) and P3 — fills the UNNAMED corner of formalizing the dependency graph itself: the 'uses' relation as a directed, loop-free order that makes subsets shippable.
  *Free basis:* MSU CSE-870 course-archive copy of the IEEE TSE 1979 paper, freely posted as lecture material; full text PDF.
  <https://cse.msu.edu/~cse870/Lectures/SS2007/ParnasPapers/Parnas-ExtensionContraction-hopkins-notes.pdf>

- **A Rational Design Process: How and Why to Fake It** — David L. Parnas and Paul C. Clements (1986)
  *open-access-paper* · intermediate · confidence: high
  The canonical defense of producing documentation 'as if' the design were rational even though it never is — the intellectual ancestor of the ADR and of this codebase's documentation-discipline tenets. It justifies the reconstruction-cost spirit: docs exist so the next reader can recover the design, not to narrate how it really happened.
  *Fills:* Documentation/auditability spirit and P5's 'encode load-bearing knowledge, not volatile prose' — fills the corner of WHY post-hoc rationalized documentation (the ADR genre itself) is honest rather than fake.
  *Free basis:* Tufts CS-257 course archive copy of the IEEE TSE 1986 paper; full PDF, stable academic host (Norman Ramsey's archive).
  <https://www.cs.tufts.edu/~nr/cs257/archive/david-parnas/fake-it.pdf>

- **On the Role of Scientific Thought (EWD447) — separation of concerns** — Edsger W. Dijkstra (1974)
  *essay* · foundational · confidence: high
  Dijkstra's two-page essay names the move every ADR-0012 principle inherits: study one aspect in isolation 'for the sake of its own consistency,' knowing the aspects are not independent. A reader who thinks in terms of orthogonal axes will find this the cleanest possible statement of why decomposition is an epistemic necessity, not a convenience.
  *Fills:* Cross-cutting spirit + P3 — the original statement of 'separation of concerns', the principle UNDER P3's single-responsibility and the audit's 'study one aspect in isolation for its own consistency.'
  *Free basis:* Official E.W. Dijkstra Archive at UT Austin; the canonical free transcription of the manuscript that coined 'separation of concerns.'
  <https://www.cs.utexas.edu/~EWD/transcriptions/EWD04xx/EWD447.html>

- **How Do Committees Invent? (Conway's Law, original)** — Melvin E. Conway (1968)
  *essay* · foundational · confidence: high
  Conway's thesis — a system's structure is a copy of the communicating organization that built it — is the sociological root cause of the 'right idea applied once and not propagated' diagnosis. For a solo author it inverts usefully: with one mind, the only force fragmenting the design is forgotten intent, which is exactly what P1/P7's mechanization defends.
  *Fills:* P2/P3 and the auditability spirit — fills the UNNAMED corner of WHY module boundaries drift: the homomorphism between communication structure and system structure (B (SSOT dissolved) and split-brain encoders are often Conway artifacts).
  *Free basis:* Full text on the author's own site (melconway.com), with a later author's note; confirmed complete and freely readable.
  <https://www.melconway.com/research/committees.html>

- **Big Ball of Mud** — Brian Foote and Joseph Yoder (1997)
  *open-access-paper* · intermediate · confidence: high
  The definitive anti-pattern catalog: why mud accumulates and which forces (time pressure, entropy, shearing layers changing at different rates) produce it. ADR-0012 is explicitly the positive inverse of an 'architectural cancer' taxonomy; this is the field's canonical statement of the diseases it inverts.
  *Fills:* The whole anti-pattern-catalog spirit and P5 — fills the corner of the FORCES that produce rot (piecemeal growth, throwaway code, sweeping-under-the-rug) so a contributor can name the smell, the inverse of ADR-0012's cancer taxonomy.
  *Free basis:* Full text on the authors' canonical site (laputan.org/mud), also offered as PDF/Word/RTF/PostScript; freely readable.
  <https://www.laputan.org/mud/>

- **A Philosophy of Software Design (Talks at Google) — deep vs shallow modules** — John Ousterhout (2018)
  *recorded-talk* · intermediate · confidence: high
  Ousterhout's deep-vs-shallow distinction gives P2 a quantitative target: the env↔Policy seam is good precisely because a one-method interface hides a large implementation. It also names 'information leakage' and 'temporal decomposition' as the failures that erode SSOT and SRP.
  *Fills:* P2/P3 and the unnamed corner of MODULE DEPTH: a good seam maximizes hidden complexity per unit of interface — the metric P2's 'simple injected contract' implicitly optimizes.
  *Free basis:* Official Talks at Google recording on YouTube (auto-transcript available; also mirrored on archive.org and the Talks at Google podcast feed). Free substitute for the paywalled book, by the author, conveying the core idea.
  <https://www.youtube.com/watch?v=bmSAYlu0NcY>

- **Hints and Principles for Computer System Design** — Butler W. Lampson (2020)
  *open-access-paper* · principal/advanced · confidence: high
  A senior architect's distilled catalog of design hints with worked system examples — the closest thing to a 'principal-engineer's checklist' the field has. Its interface-design hints ('do one thing well', 'keep secrets', 'don't generalize') are the seam discipline of P2/P3 stated by someone who shipped the systems.
  *Fills:* P1, P2, P5, P6 together — fills the principal-level corner of a unified design-hint vocabulary (Approximate/Specification/Completeness/Interface goals; 'keep secrets', 'make it fast not general') that cross-cuts every ADR-0012 principle.
  *Free basis:* Open-access on arXiv (extended, freely-revised successor to the 1983 'Hints' SOSP paper); author-authoritative.
  <https://arxiv.org/abs/2011.02455>

- **Out of the Tar Pit** — Ben Moseley and Peter Marks (2006)
  *open-access-paper* · principal/advanced · confidence: high
  Built on functional programming and Codd's relational model, this is the paper a Haskell-and-math reader will absorb fastest: it argues complexity comes from mutable state and control flow, and that the cure is isolating essential logic (a pure functional core) from accidental state. It is the conceptual spine of P9's functional-core/imperative-shell and P3's decomposition discipline.
  *Fills:* P3, P5, and the deepest unnamed corner: essential vs accidental complexity, and STATE as the prime source of accidental complexity — the FP/relational argument under 'functional core, imperative shell' (P9) and the no-god-object rule.
  *Free basis:* Widely-circulated free PDF (author-sanctioned distribution; Papers We Love canonical copy). No paywall.
  <https://curtclifton.net/papers/MoseleyMarks06a.pdf>

- **Design Principles and Design Patterns (the Dependency Inversion Principle)** — Robert C. Martin (2000)
  *open-access-paper* · intermediate · confidence: high
  The canonical statement of Dependency Inversion (and Interface Segregation) — P2's 'a new capability is a new Policy subclass with zero core edits' IS the DIP. Worth reading critically: it gives the reader the vocabulary (and the failure modes) of the principle ADR-0012's seam discipline rests on.
  *Fills:* P2 directly — fills the UNNAMED corner of NAMING the inversion: depend on abstractions, not concretions; high-level policy must not depend on low-level mechanism. Also the source of the SOLID acronym's DIP/ISP.
  *Free basis:* Author's freely-distributed Object Mentor article (objectmentor.com original), mirrored as course material at University of Turku; full PDF.
  <https://staff.cs.utu.fi/~jounsmed/doos_06/material/DesignPrinciplesAndPatterns.pdf>

- **Documenting Architecture Decisions (the ADR origin post)** — Michael Nygard (2011)
  *essay* · foundational · confidence: high
  The post that defined the ADR (Context/Decision/Status/Consequences, append-only). Reading it makes the chocofarm ADR conventions — 'Revisit when…', amend-by-append, never silently rewrite a point-in-time record — legible as a deliberate genre rather than house style.
  *Fills:* Documentation/auditability spirit — the genre definition of the artifact ADR-0012 IS; fills the corner of WHY a point-in-time, append-only, status-bearing record is the right shape (the reconstruction-cost principle in CLAUDE.md).
  *Free basis:* Original Cognitect blog post, free; the source of the ADR format these docs/adr/ files use.
  <https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions>

- **Software Architecture as a Set of Architectural Design Decisions** — Anton Jansen and Jan Bosch (2005)
  *open-access-paper* · principal/advanced · confidence: medium
  The academic source of the 'architecture = decisions' reframing, and of 'architectural knowledge vaporization' — the precise name for the reconstruction-cost problem CLAUDE.md cares about. It elevates the ADR from a habit to a theory: a decision without its rationale and rejected options is an un-auditable fact.
  *Fills:* Documentation/auditability spirit — fills the theoretical corner UNDER Nygard's blog post: architecture IS the set of decisions plus their rationale and discarded alternatives, so 'knowledge vaporization' (the reconstruction cost) is the failure to capture them.
  *Free basis:* Open-access PDF reachable from the Semantic Scholar landing page (WICSA 2005); free academic mirror.
  <https://www.semanticscholar.org/paper/Software-Architecture-as-a-Set-of-Architectural-Jansen-Bosch/4cd105262aa01f62b88baeda78570325661f67d3>

- **The Law of Leaky Abstractions** — Joel Spolsky (2002)
  *essay* · foundational · confidence: high
  The honest counterweight to 'just hide it behind an interface': all non-trivial abstractions leak, so the boundary must expose enough to be diagnosable (P2's translate-and-validate, P5's fail-loud). It explains why the wire-discipline of P7 cannot fully hide float32 non-associativity or layout — the leak is real and must be named, not papered over.
  *Fills:* P2 and the unnamed corner that BOUNDS the seam discipline: every non-trivial abstraction leaks, so a port/ACL must translate-and-validate (not pretend the foreign side is invisible) — the realism check on P2 and P7's wire boundary.
  *Free basis:* Original essay on the author's own site (joelonsoftware.com); free.
  <https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/>


### Dimension: `contracts-specs-formal`

*Focus:* Design by Contract, formal specification, assertions, behavioral subtyping, types-as-contracts foundations (leverages his math)  
*Principally maps to:* P8, P5, P2, math leverage

- **An Introduction to Design by Contract (eiffel.com manuals)** — Eiffel Software (Bertrand Meyer method) (c. 1990s)
  *lecture-notes* · foundational · confidence: high
  The cleanest short statement of the precondition/postcondition/invariant triad as the *contract* a routine signature is supposed to encode — the conceptual root of ADR-0012's 'no lying signatures'. Frames pre/post as obligation-vs-benefit between caller and supplier, which is exactly the ACL translate-and-validate-don't-coerce boundary discipline.
  *Fills:* P8 (typed signature is the SSOT of a contract), P5 (assertions as graded loud failure), P2 (caller/callee obligation boundary).
  *Free basis:* Official Eiffel Software manual page, freely readable, no paywall or login. Confirmed live and complete (covers require/precondition, ensure/postcondition, class invariant).
  <https://archive.eiffel.com/doc/manuals/technology/contract/>

- **Design by Contract (book chapter, Object-Oriented Software Construction)** — Bertrand Meyer (1997)
  *free-book* · intermediate · confidence: high
  The canonical long-form treatment: contracts as the unit of correctness, invariants as the conjunction that every public method must preserve, and the 'a comment is a contract you forgot to check' stance. Gives the reader the disciplined vocabulary (obligation/benefit table) behind ADR-0012 P8/P5.
  *Fills:* P8, P5, P2 — and the *honest-claim* spirit: a contract is a checkable claim, not a comment.
  *Free basis:* Author-hosted scanned chapter on Meyer's ETH Zurich publications page; freely downloadable PDF (substitute for the paywalled OOSC book, by the same author, conveying the core idea).
  <https://se.inf.ethz.ch/~meyer/publications/old/dbc_chapter.pdf>

- **An Axiomatic Basis for Computer Programming** — C. A. R. Hoare (1969)
  *open-access-paper* · intermediate · confidence: high
  The Hoare-triple is the mathematical object an assertion/contract approximates; for a reader with strong math this turns 'Design by Contract' from a coding convention into a proof system. It supplies the formal grammar (weakest preconditions, partial correctness) that P8's 'typed signature is the SSOT' rests on.
  *Fills:* P8 (a spec IS the function contract), P5 (assertions with proof obligations); the UNNAMED corner: the *logical* meaning of a precondition/postcondition pair, P{Q}R, beneath the engineering slogan.
  *Free basis:* Course-hosted scan of the 1969 CACM paper, freely downloadable; widely mirrored on university course pages.
  <https://www.cs.cmu.edu/~crary/819-f09/Hoare69.pdf>

- **A Behavioral Notion of Subtyping** — Barbara Liskov, Jeannette Wing (1994)
  *open-access-paper* · principal/advanced · confidence: high
  Formalizes the LSP as a contract-preservation theorem: a subtype may weaken preconditions and strengthen postconditions but never violate the supertype's invariant/history constraint. This is the precise rule that makes a polymorphic seam (P2) safe, and it rewards the reader's math background with an actual specification-mapping construction.
  *Fills:* P2 (seam/dependency-inversion substitutability), P8 (contracts on a subtype must respect the supertype contract); the unnamed corner behind 'an ACL boundary must translate-and-validate, never coerce'.
  *Free basis:* Author-hosted PDF on Wing's CMU homepage; freely downloadable (the canonical TOPLAS journal version).
  <https://www.cs.cmu.edu/~wing/publications/LiskovWing94.pdf>

- **Contracts for Higher-Order Functions** — Robert Bruce Findler, Matthias Felleisen (2002)
  *open-access-paper* · principal/advanced · confidence: high
  Introduces correct blame assignment for contract violations across higher-order boundaries — the rigorous answer to 'fail loud, at the real culprit, not a band-aid downstream' (P5's graded-loudness + root-cause spirit). For a Haskeller it reframes contracts as runtime-checked refinements at exactly the function arrows where static types stop.
  *Fills:* P5 (blame: WHO failed loudly, not just that something failed), P8 (contracts at the function boundary), P2 (the contract IS the boundary).
  *Free basis:* Author/institution-hosted PDF on Northeastern's Racket publications page; freely downloadable ICFP 2002 paper.
  <https://www2.ccs.neu.edu/racket/pubs/icfp2002-ff.pdf>

- **Programming with Refinement Types: An Introduction to LiquidHaskell** — Ranjit Jhala, Niki Vazou, Eric Seidel, et al. (2020)
  *free-book* · principal/advanced · confidence: high
  The bridge between this reader's Haskell and the contracts world: refinement types make 'no lying signatures' machine-enforceable by attaching SMT-checked predicates to types. Shows what P8's mypy-strict ratchet aspires to when taken to its logical end — the spec lives in the type and the checker proves it.
  *Fills:* P8 (push the contract INTO the type so the signature is checkably the SSOT), P5 (verification failure is a loud compile error, not a runtime surprise).
  *Free basis:* Open-access online textbook hosted by the UCSD programming-systems group (HTML and PDF, freely available).
  <https://ucsd-progsys.github.io/liquidhaskell-tutorial/book.pdf>

- **Propositions as Types** — Philip Wadler (2015)
  *open-access-paper* · intermediate · confidence: high
  The Curry-Howard correspondence is the mathematical justification for treating a typed signature as the single source of truth of a contract — a proposition the implementation must prove. Maximally rewards the reader's logic/math background and recasts P8 as a theorem rather than a style rule.
  *Fills:* P8 foundations — the deepest 'why' a type is a contract; the unnamed corner: types are not annotations, they are propositions whose programs are proofs.
  *Free basis:* Author-hosted PDF on Wadler's University of Edinburgh homepage; freely downloadable (also CACM, but the homepage copy is canonical and free).
  <https://homepages.inf.ed.ac.uk/wadler/papers/propositions-as-types/propositions-as-types.pdf>

- **On Understanding Types, Data Abstraction, and Polymorphism** — Luca Cardelli, Peter Wegner (1985)
  *open-access-paper* · intermediate · confidence: high
  The map of the whole type-theoretic landscape — universal vs existential quantification, where data abstraction (the existential) literally IS the information-hiding boundary P2 demands. Gives a math-strong reader the unifying framework under both 'seam discipline' and 'types-as-contracts'.
  *Fills:* P2 (data abstraction as the seam: an interface hides representation), P8 (subtyping/quantification as a type discipline); unnamed corner: WHY abstraction boundaries are a typing phenomenon.
  *Free basis:* Author-hosted PDF on Cardelli's homepage (lucacardelli.name); freely downloadable Computing Surveys paper.
  <http://lucacardelli.name/papers/onunderstanding.a4.pdf>

- **Who Builds a House Without Drawing Blueprints?** — Leslie Lamport (2015)
  *essay* · foundational · confidence: medium
  The persuasive short case that a specification is the blueprint you write *before* implementation — the design-discipline and reconstruction-cost-of-knowledge spirit of ADR-0012 in four pages. A gentle on-ramp to formal specification for someone who has never had formal SWE training.
  *Fills:* Cross-cutting spirit: specification as documentation that survives its author; P8 (the spec is the contract you design before you code).
  *Free basis:* Microsoft Research publication page hosting the free article PDF (the CACM page is paywalled/403; this author-affiliated copy is free). Also linked from lamport.azurewebsites.net.
  <https://www.microsoft.com/en-us/research/publication/builds-house-without-drawing-blueprints/>

- **Specifying Systems: The TLA+ Language and Tools (full book PDF)** — Leslie Lamport (2002)
  *free-book* · principal/advanced · confidence: high
  The principal-level destination: specifying a system's allowed behaviors as a mathematical predicate over state sequences, with invariants and refinement (the implements-relation). For a reader who thinks in math, TLA+ shows what 'the spec is the source of truth' means at system scale — and refinement mapping generalizes Liskov-Wing to whole systems.
  *Fills:* P8 (a specification is a precise contract for a whole system), P6 (specs let you state and check invariants/equivalences rigorously), P5 (a violated invariant is a loud, located failure).
  *Free basis:* Full book PDF posted free for personal use by the author on his own site (lamport.azurewebsites.net/tla/book.html links it); legitimately free, non-commercial-use license.
  <https://lamport.azurewebsites.net/tla/book-21-07-04.pdf>

- **On the Criteria To Be Used in Decomposing Systems into Modules** — David L. Parnas (1971/1972)
  *open-access-paper* · intermediate · confidence: high
  The origin of information hiding: a module's interface is a contract that conceals a 'secret' (a design decision likely to change), so the *criterion* for decomposition is changeability, not flowchart order. This is the principled foundation under both P2 (seam discipline) and P3 (one-owner collaborators).
  *Fills:* P2 (the interface is the contract; hide the secret behind it), P3 (one owner per design decision / single responsibility); unnamed corner: the criterion for WHERE to draw a seam.
  *Free basis:* Institution-hosted PDF on Northeastern PRL's site (technical-report version); freely downloadable. Multiple university mirrors exist.
  <https://prl.khoury.northeastern.edu/img/p-tr-1971.pdf>

- **Practical Foundations for Programming Languages (abbreviated free edition)** — Robert Harper (2016)
  *free-book* · principal/advanced · confidence: high
  A rigorous, math-first construction of what a static type system actually guarantees: the statics/dynamics split and the safety theorem that the type is a contract the operational semantics provably honors. Gives the reader the formal machinery beneath 'strict-where-achievable typing' (P8) at a depth that rewards mathematical maturity.
  *Fills:* P8 foundations (statics as the SSOT of a term's contract; type safety = progress + preservation), P5 (a well-typed program 'cannot go wrong' — failures are made impossible, not patched).
  *Free basis:* Author-hosted free 'abbreviated online edition, with corrections' on Harper's CMU page (the full 2nd edition is the paywalled Cambridge book; this free draft conveys the core).
  <https://www.cs.cmu.edu/~rwh/pfpl/abbrev.pdf>

- **Software Foundations, Volume 1: Logical Foundations** — Benjamin C. Pierce et al. (ongoing)
  *free-book* · intermediate · confidence: high
  The hands-on path to *machine-checked* contracts: every claim is a proof script the computer verifies, the literal embodiment of ADR-0012's 'mechanization over memory' and 'substantiate equivalence claims'. For a math/Haskell reader this is the most natural bridge from proof-on-paper to proof-as-code.
  *Fills:* P6 (the two-tier 'substantiate your claims' bar, taken to its limit: machine-checked invariants), P8 (specs as propositions); cross-cutting: mechanization over memory.
  *Free basis:* Free online textbook made freely available by authors/publisher (UPenn-hosted, every detail machine-checked in the Rocq/Coq prover).
  <https://softwarefoundations.cis.upenn.edu/lf-current/index.html>


### Dimension: `type-driven-fp`

*Focus:* Type-driven design, making illegal states unrepresentable, FP modularity, functional core / imperative shell, totality (leverages his Haskell)  
*Principally maps to:* P8, P9, P2, Haskell leverage

- **Parse, Don't Validate** — Alexis King (2019)
  *essay* · intermediate · confidence: high
  Names the unnamed corner of P2: a boundary should PARSE untrusted input into a type that carries the proven invariant forward, so downstream code can never re-check it (the validate-and-discard antipattern is a silent contract leak). The 'parser is a function from less-structured to more-structured' framing is the operational definition of an honest ACL.
  *Fills:* P8 (typed signature as SSOT of contract), P2 (ACL boundary translates-and-validates, never coerces), P9 (optional for absence / no sentinels).
  *Free basis:* Author's personal homepage (GitHub Pages); fully readable, no paywall.
  <https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/>

- **Effective ML Revisited (Make Illegal States Unrepresentable)** — Yaron Minsky (Jane Street) (2011)
  *essay* · intermediate · confidence: high
  The canonical source of 'make illegal states unrepresentable', but the under-cited gems here are the adjacent rules: split a record by phase so impossible field combinations cannot be typed, and make a function's fallibility visible in its name/return type (_option vs _exn) — P5's graded loudness expressed in the signature itself.
  *Fills:* P8 (signatures encode invariants), P9 (no sentinels: *_option vs *_exn discipline), P5 (obvious, graded error handling).
  *Free basis:* Public Jane Street engineering blog; prose+code version of the talk, no paywall (preferred over the video since it is text).
  <https://blog.janestreet.com/effective-ml-revisited/>

- **Designing with Types: Making Illegal States Unrepresentable** — Scott Wlaschin (F# for Fun and Profit) (2013)
  *essay* · foundational · confidence: high
  The slow, worked refactor (an 'email OR postal, at least one' rule moved from a runtime check into a sum type) is the best on-ramp for someone who knows ADTs from Haskell but never saw them used as a domain-modeling discipline. Part of a series worth reading whole for the 'constrained strings / single-case unions' techniques.
  *Fills:* P8, P3 (single-responsibility data: a type that holds exactly one well-formed thing), P9.
  *Free basis:* Free public site; this is the full worked-example essay (a free canonical substitute for Wlaschin's paywalled 'Domain Modeling Made Functional' book, stated as such here).
  <https://fsharpforfunandprofit.com/posts/designing-with-types-making-illegal-states-unrepresentable/>

- **Constructive vs Predicative Data** — Hillel Wayne (2019)
  *essay* · intermediate · confidence: high
  Fills the unnamed corner that the slogans skip: making invalid states unrepresentable ('constructive' data) is not free — it has a higher cleverness/learning cost than 'predicative' validate-after data, so it is a default-with-fallback, not an absolute. This is the honest cost-accounting a principal engineer needs to wield P9 without dogma.
  *Fills:* P9 (illegal states unrepresentable), P6 (honest claims about tradeoffs), cross-cutting honesty.
  *Free basis:* Author's free public blog; full essay.
  <https://www.hillelwayne.com/post/constructive/>

- **Boundaries** — Gary Bernhardt (2012)
  *recorded-talk* · intermediate · confidence: high
  The origin of 'functional core, imperative shell': push decisions into pure functions over simple values and quarantine effects/mutation/concurrency in a thin shell at the edge. Directly underwrites P9's compiled-C++ 'pure functions in, effectful glue out' split, generalized beyond any one language.
  *Fills:* P9 (functional core / imperative shell), P2 (values as seams between subsystems), P3 (no god-objects).
  *Free basis:* The talk page is free to view; a community transcript exists at https://www.andrewyao.me/Bernhardt-Boundaries/ for those who prefer text.
  <https://www.destroyallsoftware.com/talks/boundaries>

- **The Three Layer Haskell Cake** — Matt Parsons (2018)
  *essay* · intermediate · confidence: high
  Operationalizes functional-core/imperative-shell in a real typed-FP setting: layer 1 imperative shell (IO/ReaderT), layer 2 an abstract effect interface (the inversion seam), layer 3 the pure, QuickCheckable core. The lesson is WHERE to draw the seam and how to keep the testable core maximal — the concrete shape of P9 + P2 together.
  *Fills:* P9 (functional core / imperative shell), P2 (dependency inversion at the effect boundary), P3 (one-owner layering).
  *Free basis:* Author's free public blog; full essay.
  <https://www.parsonsmatt.org/2018/03/22/three_layer_haskell_cake.html>

- **Hexagonal Architecture (Ports and Adapters)** — Alistair Cockburn (2005)
  *essay* · intermediate · confidence: high
  The primary source for 'ports and adapters': the application core defines a port (an interface it owns), and every driver/driven adapter conforms to it, so the core can be exercised by tests, scripts, or production identically. The unnamed corner it fills: the core OWNS the interface and effects DEPEND ON the core, the inversion that makes P2's seam real rather than decorative.
  *Fills:* P2 (seam/port discipline, dependency inversion, ACL boundaries), P7 (separate the contract from the transport mechanism).
  *Free basis:* Author's own site hosts the original technical report free.
  <https://alistair.cockburn.us/hexagonal-architecture/>

- **Propositions as Types** — Philip Wadler (2015)
  *open-access-paper* · principal/advanced · confidence: high
  For a reader with deep math, this is the load-bearing 'why': Curry-Howard says a type is a theorem and a total program of that type is its proof, so a precise signature is a falsifiable claim and a sentinel/throw on the core is a hole in the proof. This is the foundation P8 rests on, made rigorous.
  *Fills:* P8 (a typed signature IS a proposition; the implementation is its proof — the deepest justification for 'no lying signatures'), cross-cutting honesty.
  *Free basis:* Author's university homepage hosts the PDF (also the CACM open-access version).
  <https://homepages.inf.ed.ac.uk/wadler/papers/propositions-as-types/propositions-as-types.pdf>

- **Theorems for Free!** — Philip Wadler (1989)
  *open-access-paper* · principal/advanced · confidence: high
  Parametricity: a sufficiently polymorphic type forces theorems on every inhabitant, so the signature alone guarantees behavioral laws no test needs to assert. This is the strongest possible reading of 'the signature is the SSOT of the contract' (P8) and the math-native payoff for designing with parametric types.
  *Fills:* P8 (a polymorphic signature CONSTRAINS all implementations — the type bounds behavior for free), P6 (invariants you get without a test).
  *Free basis:* Author's university homepage hosts the paper free (canonical PDF mirror: http://www-verimag.imag.fr/~perin/enseignement/Magistere/articles/Wadler-ICFP'1989-structure.pdf).
  <https://homepages.inf.ed.ac.uk/wadler/papers/free/free.dvi>

- **Why Functional Programming Matters** — John Hughes (1990)
  *open-access-paper* · foundational · confidence: high
  The classic argument that FP's value is MODULARITY: higher-order functions and lazy evaluation are the 'glue' that lets you decompose a problem into small, independently reusable pieces. It reframes P3 (single responsibility) and P2 (seams) as a compositional discipline rather than a class-diagram one.
  *Fills:* P2/P3 (modularity via composition; higher-order functions and lazy glue as the seams), cross-cutting maintainability.
  *Free basis:* Author's university page hosts PostScript/PDF free.
  <https://www.cse.chalmers.se/~rjmh/Papers/whyfp.html>

- **Total Functional Programming** — D. A. Turner (2004)
  *open-access-paper* · principal/advanced · confidence: high
  Argues for a discipline where every function provably terminates and the data/codata distinction is explicit, so a signature A -> B is a genuine total guarantee, not 'B or loop-forever or crash'. This is the rigorous root of P9's 'never throws on the core' and the totality angle his Haskell background already half-knows.
  *Fills:* P8 (a total function actually honors its signature — partiality is a hidden lie), P9 (no bottom/throw on the core), P6 (termination as a checkable invariant).
  *Free basis:* Journal of Universal Computer Science is open access; official PDF.
  <https://www.jucs.org/jucs_10_7/total_functional_programming/jucs_10_07_0751_0768_turner.pdf>

- **How to Design Co-Programs** — Jeremy Gibbons (2021)
  *open-access-paper* · intermediate · confidence: high
  The principled dual of 'make illegal states unrepresentable': once the type is right, the program structure is largely determined (folds for input types, unfolds/co-programs for output types), turning design into derivation. A math reader gets the recursion/corecursion symmetry that makes total, correct-by-construction code feel inevitable.
  *Fills:* P8/P9 (the type of the data DRIVES the shape of the function — fold for consuming, unfold for producing — so well-typedness guides correct-by-construction code), P1 (derive structure from the one canonical type).
  *Free basis:* Author's Oxford publications page hosts the PDF free (JFP article, also open via Cambridge Core 'Save PDF').
  <https://www.cs.ox.ac.uk/jeremy.gibbons/publications/copro.pdf>

- **Typed Tagless Final Interpreters (lecture notes)** — Oleg Kiselyov (2012)
  *lecture-notes* · principal/advanced · confidence: high
  Shows how to define a domain as a typed interface (a type class / record of operations) so that only well-formed programs are even expressible and multiple back-ends (evaluate, pretty-print, optimize) are independent adapters of one contract. The principled, math-flavored answer to seams + the expression problem behind P2.
  *Fills:* P2 (a typed interface as the SSOT seam; many interpretations behind one contract — the expression problem / extensible ACL), P8 (well-formedness enforced by the host type system, no runtime tag checks).
  *Free basis:* Author's site (okmij.org) hosts the spring-school lecture notes free.
  <https://okmij.org/ftp/tagless-final/course/lecture.pdf>

- **Ariane 5 Flight 501 Failure — Report by the Inquiry Board** — J. L. Lions et al. (ESA/CNES Inquiry Board) (1996)
  *incident-report* · intermediate · confidence: high
  A Therac-25 sibling whose root cause is exactly a type/range failure: an unprotected numeric conversion let an out-of-range value crash the IRS, and the dead reused code embodied an unverified 'this is equivalent to Ariane-4' claim. Reading the primary report teaches why P9's bounds-carrying types and P6's substantiated-equivalence bar are safety properties, not aesthetics.
  *Fills:* P9 (bounds-carrying values; a value that should be impossible became representable — the unprotected 64-bit float to 16-bit int conversion), P5 (fail loud / root cause, not band-aid: the protection was deliberately omitted), P6 (an equivalence assumed, not substantiated: Ariane-4 ranges reused on Ariane-5).
  *Free basis:* Official inquiry-board report, full text hosted free (university mirror of the public ESA/CNES report).
  <https://www.di.unito.it/~damiani/ariane5rep.html>


### Dimension: `error-models-resilience`

*Focus:* Error handling philosophy: exceptions vs results vs sentinels, fail-fast, crash-only, the robustness principle tension  
*Principally maps to:* P5, P9-rule5, P2

- **The Error Model** — Joe Duffy (2016)
  *essay* · principal/advanced · confidence: certain
  The single most complete first-principles treatment of the design space: it separates BUGS (which must abandon/fail-fast, uncatchable) from RECOVERABLE errors (which belong in the type signature as checked values), exactly P5's loudness hierarchy and P9's optional-vs-expected split. Fills the unnamed corner of WHY you partition error kinds before choosing a mechanism, with measured cost data from a real systems language.
  *Fills:* P5 (graded loudness: bugs vs recoverable errors), P9-rule5 (optional vs expected, never throw/sentinel on the core), P2
  *Free basis:* Author's own blog, fully readable, no paywall.
  <https://joeduffyblog.com/2016/02/07/the-error-model/>

- **Crash-Only Software** — George Candea and Armando Fox (2003)
  *open-access-paper* · intermediate · confidence: certain
  Argues that if the only way to stop is to crash and the only way to start is to recover, then your recovery path is exercised constantly and is therefore trustworthy — a structural answer to P5's 'fail loud, don't band-aid'. The unnamed corner: graceful shutdown is a second, rarely-tested code path that lies; abolishing it is the discipline.
  *Fills:* P5 (remove the root cause / stop rather than band-aid), spirit: crash-only recovery as the only path means recovery is always exercised
  *Free basis:* Official USENIX HotOS IX proceedings, free HTML and PDF.
  <https://www.usenix.org/legacy/events/hotos03/tech/full_papers/candea/candea_html/index.html>

- **RFC 9413 — Maintaining Robust Protocols (the published 'Postel was wrong' draft)** — Martin Thomson and David Schinazi (2023)
  *incident-report* · intermediate · confidence: certain
  The definitive case against 'be liberal in what you accept': leniency at a boundary silently absorbs nonconformance, which calcifies into de-facto spec and rots interoperability — precisely why P2's ACL must reject, not coerce. The unnamed corner: robustness-by-tolerance is a slow-acting failure, so the boundary must fail loud now to stay honest later. (Originated as the provocatively-titled draft-thomson-postel-was-wrong.)
  *Fills:* P2 (ACL boundaries translate-and-VALIDATE, never coerce), P5 (fail loud at the boundary rather than silently absorb malformed input)
  *Free basis:* Official IETF RFC, free HTML/text on the datatracker.
  <https://datatracker.ietf.org/doc/html/rfc9413>

- **Why Do Computers Stop and What Can Be Done About It? (Tandem TR 85.7)** — Jim Gray (1985)
  *open-access-paper* · intermediate · confidence: certain
  The origin of fail-fast as an engineering term: a module should stop the instant it detects it cannot keep its contract, because most production faults are transient ('Heisenbugs') and a clean restart of an isolated process-pair recovers them. Roots P5 and crash-only in measured field data on what actually makes systems stop.
  *Fills:* P5 (fail-fast as the foundational reliability primitive), P9-rule5 (a module that detects an inconsistency should stop, not limp on)
  *Free basis:* Freely mirrored on a public university course page (scanned but complete); the canonical text. The original HP/Tandem techreports host is intermittent, hence this stable educational mirror.
  <https://pages.cs.wisc.edu/~remzi/Classes/739/Spring2003/Papers/gray-why-do-computers-stop-85.pdf>

- **Making Reliable Distributed Systems in the Presence of Software Errors (PhD thesis — the 'let it crash' source)** — Joe Armstrong (2003)
  *free-book* · principal/advanced · confidence: certain
  The primary source for 'let it crash': error RECOVERY belongs to a separate supervising process, not to defensive code smeared through the worker — so the happy path stays pure and the failure path is concentrated and tested. Directly answers P5+P3: where does error-handling live, and why not inline. Rewards an FP reader: process isolation as the unit of fault containment.
  *Fills:* P5 (let-it-crash: don't defensively patch locally, fail loud and recover at a supervisor), P3 (one-owner isolated processes), P2 (failure crosses a seam to a separate handler)
  *Free basis:* Author's thesis hosted free on the official erlang.org download server.
  <https://erlang.org/download/armstrong_thesis_2003.pdf>

- **Parse, Don't Validate** — Alexis King (2019)
  *essay* · intermediate · confidence: certain
  The Haskell-native articulation of the functional-core/imperative-shell boundary: a boundary should PARSE untrusted input into a type that makes invalid states unrepresentable, so the core never re-validates and cannot receive garbage. The unnamed corner connecting P2, P8, and P9-rule5 — error handling is a type-system discipline, not a runtime habit. Tailor-made for a reader with deep types and Haskell.
  *Fills:* P2 (boundary translates-and-validates by producing a refined type, never re-checks), P9-rule5 (optional/expected at the edge, total functions in the core), P8 (the type IS the contract)
  *Free basis:* Author's own blog, fully free.
  <https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/>

- **Errors are values** — Rob Pike (2015)
  *essay* · foundational · confidence: certain
  Short, sharp statement that errors are first-class values to be programmed with — composed, accumulated, deferred — rather than control-flow exceptions, the alternative pole to throwing. The unnamed corner: once errors are values you can build abstractions (the errWriter sink) that remove boilerplate without hiding failure, a direct rebuttal to 'results are verbose'. A clean contrast piece to Duffy and Wlaschin.
  *Fills:* P9-rule5 (errors as ordinary returned VALUES, not throws), P2, spirit: error flow is data flow
  *Free basis:* Official Go blog, free.
  <https://go.dev/blog/errors-are-values>

- **Railway Oriented Programming: A Functional Approach to Error Handling** — Scott Wlaschin (2014)
  *recorded-talk* · intermediate · confidence: certain
  Makes the Result/Either monad concrete and visual: fallible steps compose on a two-track 'railway' so the success path stays linear while errors short-circuit — the operational meaning of returning expected<T> instead of throwing. The hub page carries slides, video, and the full essay series; a reader who knows monads gets the engineering payoff of Kleisli composition for error handling. Pair with the 'Against ROP' companion essay for the honest limits.
  *Fills:* P9-rule5 (expected<T> / Result composition), P8 (the Result type is the honest signature), spirit: Kleisli composition of fallible steps
  *Free basis:* Author's free site with embedded talk video, slides, and full written series; no paywall.
  <https://fsharpforfunandprofit.com/rop/>

- **Zero-overhead deterministic exceptions: Throwing values (P0709)** — Herb Sutter (2019)
  *open-access-paper* · principal/advanced · confidence: certain
  The C++-specific reckoning: catalogues why today's C++ mixes throws, error codes, errno, and nullable returns into an incoherent mess, and proposes a value-returning failure channel — the standards-track substantiation behind P9's 'expected for failure, optional for absence'. Rewards a reader who wants the modern-C++ reliquary-to-modern reasoning at the ABI level.
  *Fills:* P9 (compiled-C++ error model), P9-rule5 (expected for failure, never sentinel/nullptr/throw on the core), P6 (the cost argument is substantiated, not asserted)
  *Free basis:* Official open-std WG21 committee paper, free PDF.
  <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0709r4.pdf>

- **Error Handling — chapters 9.1–9.3, esp. 'To panic! or Not to panic!' (The Rust Programming Language)** — Steve Klabnik and Carol Nichols (the Rust Project) (2023)
  *free-book* · foundational · confidence: certain
  The clearest mainstream codification of P9-rule5's two channels: Result<T,E> for recoverable errors a caller should handle, panic! for unrecoverable bugs where limping on is worse than stopping — the same bug-vs-error partition Duffy argues, but as enforced language doctrine. The 'To panic! or Not to panic!' chapter is the decision rubric itself.
  *Fills:* P9-rule5 (Result for recoverable / expected; panic for unrecoverable / bug — never sentinels)
  *Free basis:* Official Rust book, free online.
  <https://doc.rust-lang.org/book/ch09-03-to-panic-or-not-to-panic.html>

- **Kleisli Categories (Category Theory for Programmers, ch. on composing embellished functions)** — Bartosz Milewski (2014)
  *free-book* · principal/advanced · confidence: certain
  Supplies the mathematics a strong-math reader will actually want: error-returning functions compose because they are morphisms in a Kleisli category, and 'short-circuit on failure' is just Kleisli composition for the Maybe/Either monad. Fills the corner under Railway-Oriented Programming and expected<T> — these patterns are honest because they obey associativity and identity laws.
  *Fills:* P9-rule5 (the algebra UNDER Result/expected composition), P6 (a claim of composability ought to rest on a law)
  *Free basis:* Author's blog (the source of the free CTFP book); a community-typeset free PDF of the whole book exists on GitHub (hmemcpy/milewski-ctfp-pdf).
  <https://bartoszmilewski.com/2014/12/23/kleisli-categories/>

- **How Complex Systems Fail** — Richard I. Cook (2000)
  *essay* · intermediate · confidence: certain
  Eighteen terse propositions on why robust systems still fail: failure needs multiple defenses to align, post-hoc 'root cause' is a narrative trap, and operators-at-the-sharp-end fight latent faults continuously — the systems-safety lens behind the ADR's hard-won-failure-lessons spirit. A Therac-25 sibling that is shorter, broader, and read in fifteen minutes.
  *Fills:* Cross-cutting safety-culture spirit; P5 (defenses-in-depth and why local fail-loud beats a single guard)
  *Free basis:* Free PDF from Adaptive Capacity Labs (Cook's own distribution).
  <https://www.adaptivecapacitylabs.com/HowComplexSystemsFail.pdf>

- **Lineage-driven Fault Injection (+ RICON 2014 keynote 'Outwards from the Middle of the Maze')** — Peter Alvaro, Joshua Rosen, Joseph M. Hellerstein (2015)
  *open-access-paper* · principal/advanced · confidence: high
  Turns 'is this fault-tolerant?' into a search problem: reason backward from a correct outcome's data lineage to the minimal failure set that could break it, then inject exactly those — mechanized substantiation of a resilience claim in the P6 spirit. Rewards a math reader (it is SAT-solving over provenance). Author's site also hosts the keynote framing it intuitively.
  *Fills:* P6 (substantiating a RESILIENCE claim, not just a perf claim), spirit: mechanize the verification of fault-tolerance instead of trusting hand-picked tests
  *Free basis:* Author's UCSC homepage hosts the paper PDF and links the talk; free.
  <https://people.ucsc.edu/~palvaro/molly.pdf>

- **Fault Tolerance in a High Volume, Distributed System** — Ben Christensen (Netflix) (2012)
  *incident-report* · intermediate · confidence: high
  The imperative-shell counterpart to the core's error model: when one of dozens of dependencies degrades, you isolate it (bulkhead), trip a circuit, and fail fast with a fallback rather than let latency cascade — P5's loudness hierarchy expressed operationally. Grounds the abstract error philosophy in a billion-call-a-day system. Pairs naturally with Marc Brooker's free AWS Builders' Library 'Timeouts, retries, and backoff with jitter'.
  *Fills:* P5 (graded loudness in production: shed load, fall back, fail fast), P2 (failure isolation at the service seam), spirit: where the error model meets the operational shell
  *Free basis:* Free Netflix engineering blog post.
  <https://netflixtechblog.com/fault-tolerance-in-a-high-volume-distributed-system-91ab4faae74a>


### Dimension: `simplicity-design-wisdom`

*Focus:* Essential vs accidental complexity, simplicity, distilled systems-design wisdom and Turing-lecture classics  
*Principally maps to:* whole-spirit, P1, P3

- **Out of the Tar Pit** — Ben Moseley and Peter Marks (2006)
  *open-access-paper* · intermediate · confidence: high
  The sharpest available account of why MUTABLE STATE is the dominant source of accidental complexity, and why deriving everything possible from one authoritative store (relational essential state + derived views) is the antidote — the deep argument under P1 (one home per fact) and P4 (read-at-point-of-use over baked-in). For a relational/functional mind this reframes 'god-object' (P3) as 'too much implicit state reachable from one place.'
  *Fills:* Whole-spirit, P1, P3, P4
  *Free basis:* Self-published technical report; the authors circulated it freely and it has never had a paywalled venue. The curtclifton.net mirror is the long-stable canonical copy.
  <https://curtclifton.net/papers/MoseleyMarks06a.pdf>

- **Simple Made Easy (transcript)** — Rich Hickey (2011)
  *recorded-talk* · intermediate · confidence: high
  Draws the load-bearing distinction the ADR never names: SIMPLE (objective: one fold, un-braided) versus EASY (subjective: near-at-hand). 'Complecting' is the precise vocabulary for what P3 (no god-objects) and P2 (clean seams) forbid; it gives the reader a knife for telling essential from accidental tangling.
  *Fills:* Whole-spirit, P1, P3, P2
  *Free basis:* Community-maintained verbatim transcript on GitHub of the freely-streamable Strange Loop keynote; transcript and video both free.
  <https://github.com/matthiasn/talk-transcripts/blob/master/Hickey_Rich/SimpleMadeEasy.md>

- **No Silver Bullet — Essence and Accident in Software Engineering** — Frederick P. Brooks Jr. (1986)
  *open-access-paper* · foundational · confidence: high
  The origin of the essential/accidental complexity vocabulary the entire ADR rests on, and the sober claim that no single technique yields an order-of-magnitude win — the honest-claims posture (cross-cutting spirit) at the root. Read it as the thesis Moseley/Marks partially rebut, so the two together form a debate, not a creed.
  *Fills:* Whole-spirit
  *Free basis:* Stable scholarly mirror on Bret Victor's worrydream.com; also free as UNC TR86-020. Freely circulated for decades.
  <https://worrydream.com/refs/Brooks_1986_-_No_Silver_Bullet.pdf>

- **Hints and Principles for Computer System Design** — Butler W. Lampson (2020)
  *open-access-paper* · principal/advanced · confidence: high
  The 2020 expansion of his 1983 classic: STEADY goals (Simple, Timely, Efficient, Adaptable, Dependable) and AID techniques distilled from a career of real systems, with worked examples. It is the closest thing to a principal-engineer field guide and gives the named ADR principles their lived counterweights (when to violate, and why).
  *Fills:* Whole-spirit, P2, P3, P5
  *Free basis:* Author-deposited on arXiv (full and short PDFs); open access.
  <https://arxiv.org/abs/2011.02455>

- **Hints for Computer System Design (the 1983 original)** — Butler W. Lampson (1983)
  *open-access-paper* · intermediate · confidence: high
  The terse, aphoristic ancestor: 'do one thing well,' 'use a good idea again instead of generalizing it,' 'handle normal and worst case separately.' Each hint is a one-line crystallization of a seam/separation discipline; worth reading alongside the 2020 version to see which hints survived 37 years and which the author revised.
  *Fills:* P2, P5, P6
  *Free basis:* Hosted free on Microsoft Research (Lampson's institutional page); SIGOPS Hall of Fame paper, freely distributed.
  <https://www.microsoft.com/en-us/research/wp-content/uploads/1983/10/Hints-for-Computer-System-Design-IEEE-Software.pdf>

- **The Emperor's Old Clothes (1980 Turing Award Lecture)** — C. A. R. Hoare (1981)
  *open-access-paper* · foundational · confidence: high
  Hoare's confession that he left dangerous features OUT of a language and that an unchecked array bound is an avoidable catastrophe — the 'fail loud, remove the root cause' ethic (P5) told as a hard-won failure story, and a sibling to the Therac-25 lesson about omitted safety. The closing 'there are two ways to design software' line is the whole simplicity argument in one sentence.
  *Fills:* Whole-spirit, P5
  *Free basis:* Course-page PDF mirror at Yale of the published CACM Turing lecture; freely available.
  <https://zoo.cs.yale.edu/classes/cs422/2014/bib/hoare81emperor.pdf>

- **The Humble Programmer (EWD340, 1972 Turing Award Lecture)** — Edsger W. Dijkstra (1972)
  *open-access-paper* · foundational · confidence: high
  The argument that programmers must deliberately keep problems within the small reach of their own skulls — intellectual humility as an engineering discipline, not modesty. It grounds why we mechanize and constrain (mypy gates, P8; loud failure, P5) rather than trust cleverness, and it speaks directly to a mathematician who values proof-sized reasoning.
  *Fills:* Whole-spirit, P5, P8
  *Free basis:* The official UT Austin E. W. Dijkstra Archive transcription; freely published.
  <https://www.cs.utexas.edu/~EWD/transcriptions/EWD03xx/EWD340.html>

- **On the Criteria To Be Used in Decomposing Systems into Modules** — David L. Parnas (1972)
  *open-access-paper* · intermediate · confidence: high
  The founding statement of INFORMATION HIDING: decompose around the design decisions most likely to CHANGE, each module hiding one secret behind a stable interface. This is the rigorous justification for the ADR's seam/port discipline (P2), one-owner collaborators (P3), and 'one authoritative definition, every side derives its view' (P1/P7).
  *Fills:* P2, P3, P1, P7
  *Free basis:* Full text hosted free on Nancy Leveson's MIT course page; the canonical CACM paper, widely mirrored open.
  <http://sunnyday.mit.edu/16.355/parnas-criteria.html>

- **On the Role of Scientific Thought (EWD447)** — Edsger W. Dijkstra (1974)
  *essay* · intermediate · confidence: high
  Coins SEPARATION OF CONCERNS precisely — and crucially notes the separation is in one's ATTENTION, studying correctness and efficiency on different days, not severing the system. That nuance directly underwrites P6's two-tier bar (logic invariants vs aggregate float behavior are different concerns, examined separately) and the discipline of not conflating contract with mechanism.
  *Fills:* P2, P6, P8
  *Free basis:* Official UT Austin Dijkstra Archive transcription; freely published.
  <https://www.cs.utexas.edu/~EWD/transcriptions/EWD04xx/EWD447.html>

- **Programming as Theory Building** — Peter Naur (1985)
  *essay* · intermediate · confidence: high
  Argues a program is not its text but a THEORY living in its builders' minds — so the real artifact to preserve is the rationale, the why. This is the intellectual foundation for the ADR's 'reconstruction cost of knowledge that must survive its author' and its documentation discipline; for a reader who thinks in theories, it lands hard.
  *Fills:* Whole-spirit, documentation discipline / reconstruction cost
  *Free basis:* Freely hosted scanned PDF on a UW-Madison faculty page (Remzi Arpaci-Dusseau); the 1985 essay, widely mirrored open.
  <https://pages.cs.wisc.edu/~remzi/Naur.pdf>

- **Worse Is Better (collected: The Rise of Worse is Better + Is Worse Really Better?)** — Richard P. Gabriel (1991)
  *essay* · intermediate · confidence: high
  The MIT/New-Jersey 'right thing vs worse-is-better' dialectic on how simplicity-of-implementation trades against correctness/consistency/completeness — and Gabriel's own later self-rebuttal, modeling honest revision. It teaches that 'simple' has axes that conflict, sharpening the reader's judgment about which simplicity an ADR principle is actually buying.
  *Fills:* Whole-spirit, P5, P6
  *Free basis:* Author's own homepage hosts the essay set free, including the 'Is Worse Really Better?' PDF; Public-author-distributed.
  <https://www.dreamsongs.com/WorseIsBetter.html>

- **A Plea for Lean Software** — Niklaus Wirth (1995)
  *open-access-paper* · intermediate · confidence: high
  Wirth's thesis that a system no single individual can understand in significant detail probably should not be built, and that reducing size and complexity must be a goal at EVERY step. It is the strongest statement of comprehensibility-as-a-budget that bounds god-object growth (P3) and keeps the whole within one mind's reach.
  *Fills:* Whole-spirit, P3
  *Free basis:* Hosted free on Wirth's own ETH Zurich author homepage; verified to resolve as the full article PDF.
  <https://people.inf.ethz.ch/wirth/Articles/LeanSoftware.pdf>

- **Scalability! But at what COST?** — Frank McSherry, Michael Isard, Derek G. Murray (2015)
  *open-access-paper* · principal/advanced · confidence: high
  Introduces COST — the hardware a 'scalable' system needs before it beats a competent SINGLE-THREADED baseline — and shows many systems never do. It is the empirical conscience behind P6: a performance claim unmeasured against an honest baseline is not a claim, and complexity sold as scalability is often accidental. A perfect fit for a quantitative reader who distrusts unsubstantiated speedup folklore.
  *Fills:* P6, whole-spirit (honest claims)
  *Free basis:* USENIX HotOS open-access proceedings; free PDF, also on the lead author's site.
  <https://www.usenix.org/system/files/conference/hotos15/hotos15-paper-mcsherry.pdf>

- **Can Programming Be Liberated from the von Neumann Style? (1977 Turing Award Lecture)** — John Backus (1978)
  *open-access-paper* · principal/advanced · confidence: high
  Backus's case that word-at-a-time, state-mutating programming lacks an algebra for reasoning, and that a functional style with combining forms restores it. For a Haskell/math reader it is the deepest 'why functional core' argument (P9) and connects directly to deriving views over mutating in place (P1/P4).
  *Fills:* P1, P9, P4
  *Free basis:* Part of the ACM Turing Award Lectures collection, which ACM makes freely readable; also mirrored free at archive.org (archive.org/details/programming-liberated-von-neumann) if the ACM page gates.
  <https://dl.acm.org/doi/10.1145/1283920.1283933>


### Dimension: `wire-schema-distributed`

*Focus:* Cross-language wire contracts, schema evolution, data contracts, serialization-vs-transport, distributed-boundary fallacies  
*Principally maps to:* P7

- **Schema evolution in Avro, Protocol Buffers and Thrift** — Martin Kleppmann (2012)
  *essay* · intermediate · confidence: high
  Shows byte-for-byte how three formats encode the SAME logical record and how each survives a field being added/removed/renamed — the mechanics behind 'one authoritative definition, every side derives its view.' The Avro reader-schema/writer-schema split is the clean separation of contract from the bytes on the wire that P7 demands.
  *Fills:* P7 (one authoritative wire definition; serialization contract vs transport), and the spirit of P1.
  *Free basis:* Author's personal blog, freely readable; this is the canonical free distillation of DDIA chapter 4 (the book itself is paywalled, but this essay conveys the core idea in full).
  <https://martin.kleppmann.com/2012/12/05/schema-evolution-in-avro-protocol-buffers-thrift.html>

- **Apache Avro 1.11.1 Specification — Schema Resolution** — Apache Avro project (2022)
  *free-book* · intermediate · confidence: high
  Reading the actual resolution rules (default values, int->long->float type promotion, ignored-vs-defaulted fields, union matching) makes precise the difference between a runtime manifest and build-time codegen. For a math reader it reads like a small relational/coercion algebra over two schemas, which is exactly the right mental model for a wire ACL that translates-and-validates.
  *Fills:* P7: the runtime-manifest variant of 'derive your view, never re-author by hand' — the writer's schema travels with (or alongside) the data and the reader RESOLVES against it.
  *Free basis:* Official open-source project specification, freely published.
  <https://avro.apache.org/docs/1.11.1/specification/>

- **Protocol Buffers — Language Guide (proto3): Updating A Message Type** — Google / Protocol Buffers team (2024)
  *free-book* · foundational · confidence: high
  The 'never reuse a field number; reserve a deleted field's tag and name' rules are the concrete discipline that prevents a wire contract from silently lying after an edit — the codegen sibling to Avro's runtime manifest. The reserved-keyword mechanism is mechanization-over-memory made into a compiler-enforced gate.
  *Fills:* P7 (build-time codegen variant: a .proto is the ONE authoritative definition every language derives from) and P8 (the schema IS the typed signature of the wire).
  *Free basis:* Official open documentation, free.
  <https://protobuf.dev/programming-guides/proto3/>

- **Cap'n Proto — Introduction (zero-copy / no encode-decode rationale)** — Kenton Varda (2013)
  *free-book* · intermediate · confidence: high
  Varda — the author of open-source Protobuf v2 — argues the encode/decode step itself is the mistake: make the wire layout equal the in-memory layout so reading is pointer arithmetic, not parsing. It is the sharpest case study of why a bytes-store layout and a transport are different concerns, and the alignment/endianness rules are a tangible instance of 'a cross-boundary byte fact has one definition.'
  *Fills:* P7 (serialization CONTRACT vs transport MECHANISM, taken to its limit) and P9 (in-place typed access of bounds-carrying data with no copy).
  *Free basis:* Open-source project homepage and design docs, free.
  <https://capnproto.org/>

- **FlatBuffers White Paper** — Wouter van Oortmerssen / Google FPL (2014)
  *free-book* · intermediate · confidence: high
  The vtable scheme is a beautiful, small answer to 'how does a fixed binary layout tolerate optional and newly-added fields without re-authoring every reader' — a forward/backward-compatible derived view at zero parse cost. Pairing it against Cap'n Proto teaches the design space between layout rigidity and evolvability that P7 lives in.
  *Fills:* P7 (one authoritative layout; vtable indirection is the derived-view mechanism for optional/evolving fields) and P9 (in-place, allocation-free access).
  *Free basis:* Official open-source documentation, free.
  <https://flatbuffers.dev/white_paper/>

- **Data on the Outside versus Data on the Inside** — Pat Helland (2005)
  *open-access-paper* · principal/advanced · confidence: high
  Helland gives the conceptual law beneath the serialization formats: data that crosses a service boundary becomes immutable, time-stamped, and self-describing, which is exactly why a wire contract must be versioned and a manifest must travel with the bytes. It reframes 'a bytes-store holds state, a fabric carries coordination' as a deep ontological distinction, not a tooling choice.
  *Fills:* P7 + P2 + P4: the unnamed corner of WHY a boundary changes the nature of the data — outside data is immutable, versioned, reference-by-value; inside data is mutable and authoritative.
  *Free basis:* Open-access conference PDF hosted by cidrdb.org (CIDR 2005).
  <https://www.cidrdb.org/cidr2005/papers/P12.pdf>

- **A Note on Distributed Computing** — Jim Waldo, Geoff Wyant, Ann Wollrath, Sam Kendall (1994)
  *open-access-paper* · principal/advanced · confidence: high
  The foundational argument that you must not hide a network boundary behind a local-looking interface; the seam must EXPOSE that it is a seam (translate-and-validate, never coerce). For a Haskell reader it sharpens the intuition that the type of a remote operation is genuinely different (it can fail and partial-fail) from the local one it superficially resembles.
  *Fills:* P2 + P7: the unnamed corner that a remote call is NOT a local call — latency, partial failure, and concurrency cannot be papered over by making the boundary transparent.
  *Free basis:* Freely hosted on author's Harvard faculty page (Sun Microsystems Labs technical report).
  <https://scholar.harvard.edu/files/waldo/files/waldo-94.pdf>

- **Fallacies of Distributed Computing (canonical list + provenance)** — L. Peter Deutsch, James Gosling, Bill Joy et al. (1994)
  *essay* · foundational · confidence: high
  A compact mental checklist for auditing any cross-boundary design against the assumptions it must not make. The recently-added 'versioning is easy' fallacy is the direct indictment of treating schema evolution casually, which is the heart of P7.
  *Fills:* Cross-cutting spirit + P7: the checklist of assumptions a wire/boundary design silently bakes in — including the modern 'versioning is easy' fallacy that maps straight onto schema evolution.
  *Free basis:* Wikipedia is free; it is the most stable canonical aggregation of the eight fallacies plus their history and the newer three (versioning is easy; compensating updates always work; observability is optional).
  <https://en.wikipedia.org/wiki/Fallacies_of_distributed_computing>

- **End-to-End Arguments in System Design** — J. H. Saltzer, D. P. Reed, D. D. Clark (1984)
  *open-access-paper* · principal/advanced · confidence: high
  The principle that tells you which layer should own a guarantee: the wire/transport can be best-effort, but the contract validation must live at the endpoint that understands the semantics. It is the rigorous justification for 'a fabric carries coordination, a store holds state' and for parsing-at-the-boundary rather than trusting the pipe.
  *Fills:* P2 + P5: WHERE a check belongs — correctness guarantees (validation, integrity) belong at the endpoints that hold the meaning, not in the transport beneath them.
  *Free basis:* Freely hosted on Saltzer's MIT faculty page.
  <https://web.mit.edu/saltzer/www/publications/endtoend/endtoend.pdf>

- **The Log: What every software engineer should know about real-time data's unifying abstraction** — Jay Kreps (2013)
  *essay* · intermediate · confidence: high
  The log reframes 'state' and 'coordination' as the same totally-ordered record, and every downstream view (DB, cache, index) is a DERIVED projection of it — a vivid, large-scale instance of derive-don't-duplicate. Directly informs the ADR's distinction between a bytes-store that holds state and a fabric that carries coordination.
  *Fills:* P7 (separate the serialization contract from the transport mechanism; a bytes-store of ordered state vs a messaging fabric) and P1 (the log as the single source of truth other views derive from).
  *Free basis:* Free on LinkedIn Engineering blog.
  <https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying>

- **Hyrum's Law** — Hyrum Wright (2016)
  *essay* · foundational · confidence: high
  'All observable behaviors of your system will be depended on by somebody' is the reason schema evolution is hard even when the format supports it: a field's incidental encoding, ordering, or default becomes load-bearing. It motivates making the contract explicit and minimal so there is less unintended surface to ossify.
  *Fills:* P7 + P8 + spirit: the unnamed corner of wire/schema discipline — once a contract has enough consumers, every OBSERVABLE behavior (not just the documented schema) becomes a de-facto part of the contract.
  *Free basis:* Author's free single-page site stating the law canonically.
  <https://www.hyrumslaw.com/>

- **RFC 9413 — The Harmful Consequences of the Robustness Principle** — Martin Thomson (IAB) (2023)
  *incident-report* · principal/advanced · confidence: high
  Argues directly against the most-cited 'wisdom' about boundaries: liberal acceptance is a slow band-aid that destroys the contract, and strict, loud rejection plus active protocol maintenance is the cure. This is P5 (fail loud, root-cause not band-aid) and P7 (one authoritative definition) for protocol designers — a precise, official counterweight a reader will not have met if they only know Postel's adage.
  *Fills:* P5 + P7: the unnamed corner where 'be liberal in what you accept' (Postel) corrodes a wire contract over time — lenient parsers let undefined behavior calcify into mandatory behavior (Hyrum's Law in the wild).
  *Free basis:* Official IETF/IAB document, freely published on the IETF datatracker (RFC 9413).
  <https://datatracker.ietf.org/doc/html/draft-iab-protocol-maintenance-00>

- **October 21 post-incident analysis (GitHub 2018 MySQL divergence)** — Jason Warner / GitHub Engineering (2018)
  *incident-report* · intermediate · confidence: high
  A 43-second partition produced two databases that each held writes the other lacked — the exact disaster that a clean store/coordination boundary and a single source of truth are meant to prevent. GitHub choosing data integrity over fast recovery is a model of the ADR's safety culture: prefer a loud, correct halt over a silently divergent resume.
  *Fills:* P7 + safety-culture spirit: the concrete failure when a coordination fabric (Orchestrator/Raft failover) makes a topology decision the state store (cross-country MySQL replication) could not honor, and two authoritative copies of 'truth' diverge.
  *Free basis:* Official GitHub Engineering blog post, free.
  <https://github.blog/2018-10-30-oct21-post-incident-analysis/>

- **Spec-ulation (keynote transcript)** — Rich Hickey (2016)
  *recorded-talk* · intermediate · confidence: high
  Hickey turns 'versioning' from folklore into an algebra: a provider may only weaken its requires and strengthen its provides (a near-variance argument a Haskell reader will recognize), otherwise it must rename rather than break. This is the principled foundation under Protobuf's reserve-don't-reuse and Avro's resolution rules, and under P8's 'no lying signatures.'
  *Fills:* P7 + P8: the unnamed corner of schema/API evolution as a discipline of monotonic GROWTH vs BREAKAGE — relaxing a requirement or strengthening a promise is safe; the reverse is a breaking change that deserves a new name, not a mutated contract.
  *Free basis:* Recorded Clojure/conj 2016 keynote (free on YouTube) WITH a full community transcript hosted on GitHub.
  <https://github.com/matthiasn/talk-transcripts/blob/master/Hickey_Rich/Spec_ulation.md>


### Dimension: `numerics-reproducibility-oracle`

*Focus:* Numerical correctness, floating-point, reproducibility, and testing programs that have no known-correct oracle (apt: this is a research OR/numerical solver)  
*Principally maps to:* P6, research-codebase aptness

- **What Every Computer Scientist Should Know About Floating-Point Arithmetic** — David Goldberg (1991)
  *open-access-paper* · foundational · confidence: high
  Establishes from first principles that floating-point is not the reals: non-associativity, rounding, cancellation, and the ULP/relative-error vocabulary. This is the math that forces P6's split between a bit-exact logic tier and a tolerance-based numerics tier — without it, an 'equivalence test' silently asserts a falsehood.
  *Fills:* P6 (substantiate equivalence claims; the two-tier bar). Fills the unnamed corner BENEATH P6's float32 tier: WHY float equivalence must be aggregate-and-behavioral rather than bit-exact.
  *Free basis:* Reprinted in full by permission on Oracle's public Numerical Computation Guide (legacy Sun docs); freely readable, no paywall. The often-cited cs.wisc.edu PDF is a course mirror of the same text.
  <https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html>

- **How Java's Floating-Point Hurts Everyone Everywhere** — William Kahan and Joseph D. Darcy (1998)
  *essay* · intermediate · confidence: high
  Kahan dissects how a language's refusal to expose hardware FP semantics destroys reproducibility and accuracy across platforms. The lesson for a numerics solver: aggregate-behavioral equivalence (P6) is the only honest claim once compilers, FMA, and register width can reorder your arithmetic.
  *Fills:* P6 and P7 (cross-language wire/platform discipline). Fills the corner where 'reproducible' silently depends on the language runtime, FMA contraction, and 80-bit-vs-64-bit register spills — variation the two-tier bar must budget for.
  *Free basis:* Hosted on Kahan's UC Berkeley EECS faculty homepage; free PDF by the author.
  <https://people.eecs.berkeley.edu/~wkahan/JAVAhurt.pdf>

- **How Futile are Mindless Assessments of Roundoff in Floating-Point Computation?** — William Kahan (2006)
  *essay* · principal/advanced · confidence: high
  Kahan demolishes naive error-estimation rituals and argues for recomputing in extended precision as a practical pseudo-oracle. Directly arms P6: instead of asserting an algorithm is accurate, you substantiate it by behavioral comparison against a higher-precision run.
  *Fills:* P6 (substantiate, don't hand-wave). Fills the unnamed corner of HOW to actually assess roundoff: re-running in higher precision and comparing is the only cheap honest oracle for accuracy claims.
  *Free basis:* Author's UC Berkeley EECS homepage; free PDF.
  <https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf>

- **On Testing Non-Testable Programs** — Elaine J. Weyuker (1982)
  *open-access-paper* · intermediate · confidence: high
  The founding statement that some programs are 'non-testable' (no oracle, or one too costly to compute) and the introduction of pseudo-oracles. This is the conceptual root that justifies metamorphic/property/differential testing as the substitute for an absent ground truth.
  *Fills:* Research-codebase aptness; P6. Names the unnamed corner the whole dimension orbits: the ORACLE PROBLEM — what to do when no known-correct answer exists, exactly the situation of a novel OR/belief-MDP solver.
  *Free basis:* Free course-page PDF (U. Washington CSE503, Rene Just); the publisher version (Oxford Computer Journal) is paywalled, so this is the stable free render of the identical paper.
  <https://homes.cs.washington.edu/~rjust/courses/CSE503/2021_02_12-reading1.pdf>

- **Metamorphic Testing: A Review of Challenges and Opportunities** — Tsong Yueh Chen, Fei-Ching Kuo, Huai Liu, Pak-Lok Poon, Dave Towey, T. H. Tse, Zhi Quan Zhou (2018)
  *open-access-paper* · intermediate · confidence: high
  Systematizes metamorphic testing — checking invariant relations between related runs rather than absolute outputs. For an adaptive-stochastic-orienteering solver with no closed-form answer, MRs (monotonicity in budget, permutation invariance of routes, scaling laws) are the testable substance behind P6's behavioral tier.
  *Fills:* Research-codebase aptness; P6. Fills the corner of HOW to test without an oracle: metamorphic relations (f(transformed input) relates predictably to f(input)) as the practical oracle substitute.
  *Free basis:* Open access under CC BY 4.0 in the University of Nottingham institutional repository (the ACM Computing Surveys version is paywalled; this is the author/OA copy).
  <https://nottingham-repository.worktribe.com/output/925152/metamorphic-testing-a-review-of-challenges-and-opportunities>

- **A Survey on Metamorphic Testing** — Sergio Segura, Gordon Fraser, Ana B. Sanchez, Antonio Ruiz-Cortes (2016)
  *open-access-paper* · intermediate · confidence: high
  A complementary, more practice-oriented survey: where MRs come from, how to organize them, common pitfalls. Useful precisely because deriving good MRs is the hard part for a numerical solver, and a bad MR gives false confidence — an anti-pattern P6's 'substantiate the claim' is meant to catch.
  *Fills:* Research-codebase aptness; P6. Sibling to Chen et al.; fills the corner of the engineering process of MT — sourcing, prioritizing, and structuring metamorphic relations into a test suite.
  *Free basis:* Author accepted manuscript in the White Rose open-access repository; free PDF (IEEE TSE version paywalled).
  <https://eprints.whiterose.ac.uk/id/eprint/110335/1/segura16-tse.pdf>

- **QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs** — Koen Claessen and John Hughes (2000)
  *open-access-paper* · intermediate · confidence: high
  The canonical property-based-testing paper, written for exactly this reader's idiom. A property (e.g. reverse . reverse == id) is a contract checked over random inputs — the mechanized substantiation P6 demands, and the runtime echo of P8's typed signature as the source of truth.
  *Fills:* P6 and P8 (typed contracts as SSOT). Fills the corner where the reader's Haskell background pays off: a property is a machine-checked specification, the executable form of a function's contract.
  *Free basis:* Free course-archive PDF (Tufts CS257, Norman Ramsey); the original ICFP 2000 paper, also mirrored on UPenn course pages.
  <https://www.cs.tufts.edu/~nr/cs257/archive/john-hughes/quick.pdf>

- **Experiences with QuickCheck: Testing the Hard Stuff and Staying Sane** — John Hughes (2016)
  *open-access-paper* · principal/advanced · confidence: high
  Extends QuickCheck to stateful systems via model-based testing and shows shrinking turning a 10000-step failure into a 3-step counterexample. The principal-level lesson: a model is a second source of truth you test the implementation against, and shrinking is what makes P5's loud failure actually actionable.
  *Fills:* P6, P3 (single-owner state), P5 (fail loud). Fills the unnamed corner of STATEFUL/model-based property testing and shrinking — finding the minimal failing case, which is how a loud failure becomes diagnosable.
  *Free basis:* Free course-archive PDF (Tufts CS257). A recorded talk of the same material exists (YouTube zi0rHwfiX1Q) but the paper is the citable transcript-equivalent artifact.
  <https://www.cs.tufts.edu/~nr/cs257/archive/john-hughes/quviq-testing.pdf>

- **Finding and Understanding Bugs in C Compilers (Csmith)** — Xuejun Yang, Yang Chen, Eric Eide, John Regehr (2011)
  *open-access-paper* · intermediate · confidence: high
  Csmith found 325+ bugs by compiling random programs with multiple compilers and comparing outputs — an oracle built from disagreement, not ground truth. For a C++ functional core (P9), the directly transferable idea is differential testing of the C++ core against a reference (e.g. the numpy/JAX path) as the substantiation of equivalence (P6).
  *Fills:* P6, P7, P9 (compiled C++ core). Fills the corner of DIFFERENTIAL testing as an oracle: when N independent implementations of one spec disagree, at least one is wrong — and of trusting your toolchain less.
  *Free basis:* Author's University of Utah homepage preprint of the PLDI 2011 paper; free PDF.
  <https://users.cs.utah.edu/~regehr/papers/pldi11-preprint.pdf>

- **Efficient Reproducible Floating Point Summation and BLAS (incl. ReproBLAS)** — Peter Ahrens, James Demmel, Hong Diep Nguyen (2016)
  *open-access-paper* · principal/advanced · confidence: high
  Shows that order-independent (hence parallel-reproducible) floating-point summation is achievable but not free — quantifying the price of demanding bit-exactness from numerics. This sharpens P6's two-tier choice: it tells you exactly when to spend for bit-reproducibility versus accept aggregate-behavioral equivalence.
  *Fills:* P6 and P7 (one authoritative numeric contract across parallel reductions). Fills the corner where parallelism + non-associativity breaks bit-reproducibility, and what it costs to restore it.
  *Free basis:* UC Berkeley EECS Technical Report series; free PDF.
  <https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.pdf>

- **The T-Experiments: Errors in Scientific Software** — Les Hatton (1997)
  *open-access-paper* · intermediate · confidence: medium
  Hatton's measurements showed scientific software disagreeing down to a single significant figure on the same inputs, mostly from defects, not roundoff. The sobering lesson for any 'optimal'-claiming solver: agreement-to-N-digits is itself a claim that must be measured (P6), and unmeasured numerical trust is usually misplaced.
  *Fills:* Cross-cutting: honest claims; reconstruction cost. Fills the corner of the EMPIRICAL reproducibility catastrophe: independent implementations of one algorithm on identical data diverged from 6 significant figures to 1.
  *Free basis:* Author/institutional copy in the Kent Academic Repository (KAR); free PDF. Publisher IEEE CS&E version is paywalled.
  <https://kar.kent.ac.uk/id/document/2075>

- **Ten Simple Rules for Reproducible Computational Research** — Geir Kjetil Sandve, Anton Nekrutenko, James Taylor, Eivind Hovig (2013)
  *open-access-paper* · foundational · confidence: high
  A short, citable checklist (record every step, version everything, archive raw results, control randomness with stored seeds). It operationalizes the ADR spirit of mechanization-over-memory for a stochastic solver where an unrecorded RNG seed silently makes a 'result' unreproducible.
  *Fills:* Cross-cutting: mechanization over memory; documentation discipline; reconstruction cost. Fills the corner of OPERATIONAL reproducibility — recording seeds, versions, and exact commands so a result survives its author.
  *Free basis:* PLOS Computational Biology, fully open access (CC BY).
  <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003285>

- **Accuracy and Stability of Numerical Algorithms (SIAM Day lecture slides)** — Nicholas J. Higham (2013)
  *lecture-notes* · principal/advanced · confidence: high
  Introduces backward vs forward error and conditioning: a stable algorithm gives the exact solution to a slightly perturbed input, which is the principled definition of P6's behavioral-equivalence tier. This is the vocabulary that turns 'close enough' from a hand-wave into a theorem-grade claim.
  *Fills:* P6. Fills the principal-level corner the reader's math background most rewards: BACKWARD ERROR ANALYSIS — 'the computed answer is the exact answer to a nearby problem' — the rigorous frame for what 'equivalent' means.
  *Free basis:* Free lecture slides hosted by Cardiff University. Substitute (with this note) for Higham's paywalled SIAM monograph of the same name; the slides convey the backward-error core. His blog nhigham.com is a free companion.
  <https://mathsdemo.cf.ac.uk/maths/research/researchgroups/applied/siam/resources/SIAM_Day_2013_NJHigham_lecture.pdf>

- **Predicting Metamorphic Relations for Testing Scientific Software (preprint)** — Upulee Kanewala, James M. Bieman, Anneliese Andrews (2016)
  *open-access-paper* · principal/advanced · confidence: medium
  Targets metamorphic testing specifically for scientific/numerical functions and the meta-problem of discovering which MRs hold (permutation, scaling, additive relations) from program structure. It addresses the exact gap a generic MT survey leaves open for a numerical solver: which invariants are even worth asserting.
  *Fills:* Research-codebase aptness; P6. Fills the hardest unnamed corner of oracle-free testing: SOURCING metamorphic relations for numerical code, where good MRs are scarce and non-obvious.
  *Free basis:* Author-submitted preprint on Bieman's Colorado State University homepage; free PDF (Wiley STVR version paywalled).
  <https://www.cs.colostate.edu/~bieman/Pubs/kanewalaPredictingMetamorphicSTVR.PreprintSubmitted4publication.pdf>


### Dimension: `safety-postmortems`

*Focus:* Safety engineering, failure culture, and incident postmortems beyond Therac-25 (he has read Therac-25)  
*Principally maps to:* P5, P6, P4, hard-lessons spirit

- **Engineering a Safer World: Systems Thinking Applied to Safety** — Nancy G. Leveson (2011)
  *free-book* · principal/advanced · confidence: high
  Reframes accidents as missing CONTROL CONSTRAINTS rather than chains of component failures, so the fix is structural (remove the systemic cause) not a local patch — the principal-level generalization of ADR-0002/P5 beyond a single try/except. The control-theory framing rewards a strong-math reader who will recognize STAMP as a feedback-control model of a sociotechnical plant.
  *Fills:* P5 (fail loud / remove root cause not band-aid), P3 (no god-objects; constraints owned at the right level), cross-cutting safety-culture and reconstruction-cost spirit
  *Free basis:* Official MIT Press open-access monograph; full PDF on OAPEN (and direct.mit.edu OA). Public, no login. Author makes the whole book available openly.
  <https://library.oapen.org/handle/20.500.12657/26043>

- **A New Accident Model for Engineering Safer Systems (STAMP)** — Nancy G. Leveson (2004)
  *open-access-paper* · principal/advanced · confidence: high
  The compact, self-contained statement of STAMP: safety is a constraint-enforcement problem over a hierarchical control structure, and accidents are inadequate constraints, not bad luck. A faster on-ramp than the book for a reader who wants the formal kernel before the 500-page treatment, and the cleanest articulation of why a band-aid that leaves the constraint un-enforced is not a fix.
  *Fills:* P5, P2 (the system as a layered control structure with enforced inter-layer constraints), classification-discipline spirit
  *Free basis:* Author-hosted PDF on Leveson's MIT sunnyday server (the canonical preprint of the Safety Science paper); freely downloadable.
  <http://sunnyday.mit.edu/accidents/safetyscience-single.pdf>

- **How Complex Systems Fail** — Richard I. Cook (1998)
  *essay* · intermediate · confidence: high
  Four dense pages establishing that complex systems run in a permanently degraded mode and that catastrophe needs multiple latent faults to line up — the intuition behind a GRADED loudness hierarchy and why removing one band-aid rarely removes the hazard. Short enough to read in a sitting yet quotable for the rest of a career.
  *Fills:* P5 (graded loudness; defenses degrade silently), the safety-culture/hard-lessons spirit, and P3 (failure is never one component's fault)
  *Free basis:* Cook's 18-point treatise, freely published; canonical hosting at how.complexsystems.fail (mirror PDF at adaptivecapacitylabs.com/HowComplexSystemsFail.pdf). No paywall.
  <https://how.complexsystems.fail/>

- **ARIANE 5 Flight 501 Failure — Report by the Inquiry Board** — J. L. Lions (Chairman) et al., ESA/CNES Inquiry Board (1996)
  *incident-report* · intermediate · confidence: high
  A 64-bit float horizontal-velocity converted to a 16-bit signed int overflowed because Inertial Reference code reused from Ariane 4 ran in a flight envelope where its implicit precondition no longer held — the textbook case of a reused component carrying an unstated, now-false contract (a lying signature / unhandled-conversion-failure lesson straight at P9).
  *Fills:* P9 (overflow on a 64-to-16-bit conversion with no expected/handled error path), P6 (an unjustified reuse/equivalence assumption), P8 (a reused module whose effective contract was a lie in the new context)
  *Free basis:* Official inquiry-board report; full text freely mirrored on MIT sunnyday (and ESA sci.esa.int). Public-domain government/agency document.
  <http://sunnyday.mit.edu/nasa-class/Ariane5-report.html>

- **Patriot Missile Defense: Software Problem Led to System Failure at Dhahran (GAO/IMTEC-92-26)** — U.S. Government Accountability Office (1992)
  *incident-report* · intermediate · confidence: high
  The 0.1-seconds-times-an-integer-clock error: 1/10 is not representable in fixed-point binary, so a tiny per-tick truncation compounded over 100 hours of uptime into a 0.34s tracking error that missed a Scud — the canonical demonstration of P6's 'float is not associative/exact' and why elapsed-time-since-boot must be treated as a drifting, not frozen, quantity (P4).
  *Fills:* P6 (float is not exact; aggregate drift vs bit-exact reasoning), P4 (a value baked at boot accumulating error over uptime), P5 (a known fix that did not reach the field)
  *Free basis:* Official GAO report, U.S. government work, freely downloadable PDF from gao.gov.
  <https://www.gao.gov/assets/imtec-92-26.pdf>

- **Mars Climate Orbiter Mishap Investigation Board — Phase I Report** — A. Stephenson et al., NASA MCO MIB (1999)
  *incident-report* · intermediate · confidence: high
  Ground software emitted impulse in pound-force-seconds while the navigation software consumed newton-seconds; the unit was a cross-component WIRE fact that each side re-authored by hand instead of deriving from one specification — the purest P7/P1 cautionary tale for any serialization boundary in the C++ actor transport.
  *Fills:* P7 (a cross-boundary fact — thrust units — had two authoritative authors instead of one derived contract), P1 (one home per fact), P8 (an interface whose unit contract was implicit and therefore a lying signature)
  *Free basis:* Official NASA mishap-board report hosted on NASA's Lessons-Learned Information System (LLIS); U.S. government work, free PDF.
  <https://llis.nasa.gov/llis_lib/pdf/1009464main1_0641-mr.pdf>

- **In the Matter of Knight Capital Americas LLC (SEC Release 34-70694)** — U.S. Securities and Exchange Commission (2013)
  *incident-report* · intermediate · confidence: high
  A partial deploy left one of eight servers running dead 'Power Peg' code that a REUSED flag re-activated, losing ~$460M in 45 minutes — a config-as-frozen-vs-live disaster (P4) compounded by treating deployment coordination as if it were ordinary state, exactly the bytes-store-vs-messaging-fabric separation P7 demands.
  *Fills:* P4 (a repurposed feature flag flipped meaning between deploys), P7 (deploy as an out-of-band coordination problem distinct from the state the bytes carry), P5 (alarms fired but were not treated as loud), P8 (dead code reanimated by a flag whose contract had silently changed)
  *Free basis:* Official SEC administrative order; U.S. government work, free PDF on sec.gov.
  <https://www.sec.gov/files/litigation/admin/2013/34-70694.pdf>

- **Cloudflare outage on November 18, 2025 (official postmortem)** — Matthew Prince et al., Cloudflare (2025)
  *incident-report* · intermediate · confidence: high
  A duplicated-rows DB permission change doubled a Bot-Management feature file past a hard-coded 200-feature limit, and the Rust proxy called .unwrap() on the resulting Err and panicked the whole core proxy globally — a direct, recent instantiation of P9's 'never throw/sentinel on the functional core; use expected<>/optional for failure' applied to a live-config boundary (P4).
  *Fills:* P9 (Result::unwrap() on an error path that 'could not happen' — throwing/panicking on the core instead of returning expected-for-failure), P4 (a periodically regenerated config file as live, unbounded input), P5 (the limit check existed but its failure was unhandled)
  *Free basis:* Cloudflare's own public postmortem blog post; freely readable, no login.
  <https://blog.cloudflare.com/18-november-2025-outage/>

- **Columbia Accident Investigation Board Report, Volume I (esp. Ch. 7-8: organizational causes)** — Harold Gehman et al., CAIB (2003)
  *incident-report* · principal/advanced · confidence: high
  The Board insists the foam strike was the physical cause but the ORGANIZATIONAL cause — schedule pressure plus a quietly accepted out-of-spec condition — was the real root, and dissects how a PowerPoint engineering culture let an unsubstantiated safety claim pass review. The deepest free articulation of normalization-of-deviance and why honest substantiation (P6) is a safety property, sibling to Therac-25 without overlapping it.
  *Fills:* P5 (root-cause vs band-aid; the organizational cause, not just the foam), P6 (the perils of an unsubstantiated equivalence/'flew before so it's safe' claim), normalization-of-deviance and reconstruction-cost spirit
  *Free basis:* Official NASA/CAIB report; U.S. government work, full PDF free on nasa.gov.
  <https://www.nasa.gov/wp-content/uploads/static/history/columbia/reports/CAIBreportv6.pdf>

- **A Case Study of Toyota Unintended Acceleration and Software Safety** — Philip Koopman (Carnegie Mellon) (2014)
  *lecture-notes* · principal/advanced · confidence: high
  Walks through the actual defects an expert team found in Toyota's ETCS firmware — stack overflow into mirrored RAM, unprotected global state, a watchdog that could not catch a hung task, single-points-of-failure with no fail-safe — i.e. precisely the bounds-carrying, const-correct, fail-loud disciplines P9 and P5 prescribe, told as a fatal counter-example. Rewards a reader who can map it onto memory-safety reasoning.
  *Fills:* P9 (unsafe shared mutable state, missing bounds, no expected/error discipline in the embedded core), P5 (defeated watchdog and absent fail-safe = a silenced loud failure), P3 (a god-object task whose corruption had global effect)
  *Free basis:* Koopman's CMU course page hosting his slides/PDF and the trial materials, explicitly CC-BY 4.0 for educational use — a free, citable substitute for Michael Barr's paywalled/court-sealed 800-page expert report.
  <https://users.ece.cmu.edu/~koopman/toyota/index.html>

- **Formal Methods and the Certification of Critical Systems (CSL-93-7)** — John Rushby (SRI International) (1993)
  *open-access-paper* · principal/advanced · confidence: high
  A mathematician's treatment of where formal methods genuinely buy assurance in safety-critical software and where they cannot, with a sober calculus of which claims a proof actually substantiates — directly nourishing P6's 'attach the substantiation, and know what it covers' and P8's view of a typed/spec contract as the authoritative function definition. Speaks to a Haskell/type-theory reader in their own dialect.
  *Fills:* P6 (what a verification/equivalence claim is actually worth, and its limits), P8 (a specification as the typed SSOT of a contract), P9 (the cost/benefit of pushing correctness into provable structure), mechanization-over-memory spirit
  *Free basis:* Author-hosted technical report on SRI's CSL server (also NASA CR 4551); freely downloadable PDF.
  <https://www.csl.sri.com/~rushby/papers/csl-93-7.pdf>

- **Postmortem Culture: Learning from Failure (Site Reliability Engineering, Ch. 15)** — Betsy Beyer, Chris Jones, Jennifer Petoff, Niall Murphy (eds.), Google (2016)
  *free-book* · foundational · confidence: high
  The operational complement to Leveson/Cook: how to run a BLAMELESS postmortem so the organization actually extracts and records the systemic root cause instead of assigning blame and moving on — the cultural machinery that makes P5 and the 'knowledge must survive its author' documentation spirit real rather than aspirational.
  *Fills:* P5 (institutionalize fixing the root cause; blameless = remove the systemic cause not the scapegoat), documentation-discipline and reconstruction-cost spirit
  *Free basis:* Full chapter free on Google's official sre.google site (the entire SRE book is published free online).
  <https://sre.google/sre-book/postmortem-culture/>

- **Blameless PostMortems and a Just Culture** — John Allspaw (Etsy) (2012)
  *essay* · foundational · confidence: high
  The short, persuasive origin essay popularizing 'just culture' in software: people act with local rationality given the information they had, so a postmortem that hunts for the mechanism (and the second story) finds the real root cause, while one that hunts for a culprit drives the truth underground. The practical why behind blameless P5, in ten minutes.
  *Fills:* P5 (focus the investigation on the failure MECHANISM and the local-rationality of the decision, not on punishing the proximate human), safety-culture/hard-lessons spirit
  *Free basis:* Free essay on Etsy's Code as Craft engineering blog; no paywall.
  <https://www.etsy.com/codeascraft/blameless-postmortems>


### Dimension: `config-knowledge-maintainability-api`

*Focus:* Config/operational discipline, knowledge transfer and maintainability, software aging, and API/contract evolution  
*Principally maps to:* P4, P8, maintainability and documentation spirit

- **Programming as Theory Building** — Peter Naur (1985)
  *open-access-paper* · principal/advanced · confidence: high
  Argues a program's real artifact is the *theory* in the builders' heads, not the source or docs — so when the theory dies (the author leaves) the program is dead even if it compiles. This is the deepest justification for ADR-0012's 'load-bearing knowledge encoded in code, not unenforceable prose' and for why a derived SSOT (one home per fact) is what lets the theory be reconstructed.
  *Fills:* The maintainability/documentation spirit and the unnamed 'reconstruction cost of knowledge that must survive its author' corner the ADR's self-application leans on but never theorizes.
  *Free basis:* Author's classic essay, freely mirrored by university CS departments (UW-Madison copy here; identical text also at gwern.net/doc/cs/algorithm/1985-naur.pdf). Out of any paywall.
  <https://pages.cs.wisc.edu/~remzi/Naur.pdf>

- **Software Aging** — David L. Parnas (1994)
  *open-access-paper* · intermediate · confidence: medium
  Names the two causes of decay — 'lack of movement' (failing to adapt) and 'ignorant surgery' (changes that violate the original design's structure). ADR-0012's no-god-object/SSOT/seam rules are precisely the prophylaxis against ignorant surgery; this is the canonical statement of the disease the ADR's taxonomy inverts.
  *Fills:* Software-aging corner under the maintainability spirit; the 'why the connective tissue rots' diagnosis ADR-0012 exists to prevent (its 'right idea applied once and not propagated' is aging in Parnas's vocabulary).
  *Free basis:* ICSE 1994 keynote; freely circulated PDF (semanticscholar and university mirrors). The cited UNC TR slot hosts Parnas tech reports; if that exact path moves, the paper is reliably free via Semantic Scholar paper id 1ccfc805. Free as a conference keynote that has been openly distributed for decades.
  <https://www.cs.unc.edu/techreports/87-009.pdf>

- **A Rational Design Process: How and Why to Fake It** — David L. Parnas, Paul C. Clements (1986)
  *open-access-paper* · intermediate · confidence: high
  Shows that real design is never top-down/rational, yet the documentation should be written *as if* it were, because the faked-rational record is what a maintainer can actually navigate. Directly underwrites ADR-0005/0012's stance that documentation is part of the work and an ADR is a point-in-time rational record, not a transcript of how the insight arrived.
  *Fills:* Documentation-discipline spirit: the ADR's own posture of writing the rule *ahead* of the code (born-clean, not audited-dirty) is exactly 'fake the rational process' — produce the documentation the ideal process *would* have, even though discovery was messy.
  *Free basis:* Hosted free in Norman Ramsey's Tufts course archive; also freely at users.ece.utexas.edu/~perry/education/SE-Intro/fakeit.pdf. Long-standing open course mirrors.
  <https://www.cs.tufts.edu/~nr/cs257/archive/david-parnas/fake-it.pdf>

- **No Silver Bullet — Essence and Accident in Software Engineering** — Frederick P. Brooks, Jr. (1986)
  *open-access-paper* · foundational · confidence: high
  Separates *essential* complexity (the problem's irreducible difficulty) from *accidental* complexity (the tooling's). ADR-0012's whole modern-C++/typed-signature program is an assault on accidental complexity at zero cost to essence; Brooks gives a principal engineer the vocabulary to argue which is which, and why no rule is a silver bullet — exactly the ADR's own 'most principles are review-only' humility.
  *Fills:* The cross-cutting 'honest claims' spirit and the essence/accident distinction underlying P6 (substantiate) and P9 (zero-cost-abstraction restores the contract at no runtime cost — removing accidental complexity, not essential).
  *Free basis:* UNC Computer Science technical report TR86-020, the author's own institution hosting the original — a stable free PDF (worrydream.com mirrors the same). The full Mythical Man-Month book is paywalled, so this is the free canonical Brooks essay carrying the core idea, as instructed.
  <https://www.cs.unc.edu/techreports/86-020.pdf>

- **Out of the Tar Pit** — Ben Moseley, Peter Marks (2006)
  *open-access-paper* · principal/advanced · confidence: high
  Builds on Brooks to argue that state and control are the chief sources of accidental complexity, and that pushing logic into a pure, declarative core with state quarantined to a thin shell is the structural cure. This is the FP-native rationale for P9's functional core / imperative shell, written for exactly this reader's mathematical-FP sensibility.
  *Fills:* Fills the unnamed 'why functional-core/imperative-shell is the maintainability win' corner behind P9 — and rewards the Haskell background directly. Connects state-isolation to the SSOT and god-object principles (P1/P3).
  *Free basis:* Self-published paper (never paywalled); widely mirrored, this curtclifton.net copy is the most-cited stable link.
  <https://curtclifton.net/papers/MoseleyMarks06a.pdf>

- **How to Design a Good API and Why It Matters (recorded talk + slides)** — Joshua Bloch (2007)
  *recorded-talk* · intermediate · confidence: high
  Distills API contracts into checkable maxims — 'when in doubt, leave it out', 'don't make the client do anything the module could do', 'obey the principle of least astonishment', 'public APIs are forever, one chance to get it right'. These are P8's typed-signature-as-SSOT rule expressed as design judgment, and the irreversibility point is why ADR-0012 wants the contract honest before it ships.
  *Fills:* P8 (the signature *is* the contract) on the human/design side: the unnamed corner of *what makes a contract good* and irreversible once published ('public APIs are forever').
  *Free basis:* Free InfoQ recording with synchronized slides and an accompanying written summary (transcript-equivalent). The seed's lcsd05.cs.tamu.edu/slides/keynote.pdf is now dead (connection refused), and the OOPSLA companion paper is on ACM; the free canonical artifact is this InfoQ talk by the same author conveying the identical maxims.
  <https://www.infoq.com/presentations/effective-api-design/>

- **Hyrum's Law (with a sufficient number of users of an API…)** — Hyrum Wright (2016)
  *essay* · intermediate · confidence: high
  'With a sufficient number of users of an API, it does not matter what you promise in the contract: all observable behaviors of your system will be depended on by somebody.' This is the boundary condition on P8 and P6: a behavioral-equivalence bar (not byte-identity) is precisely the discipline needed when a reimplementation must preserve observable behavior, not just the declared signature.
  *Fills:* The crucial UNNAMED corner of P8: the *implicit* contract. P8 says the typed signature is the SSOT of the contract — Hyrum's Law is the hard truth that with enough users, every *observable* behavior becomes contract regardless of the signature.
  *Free basis:* The canonical one-page statement on the author's own free site; the longer treatment is the free online chapter of the Software Engineering at Google book (abseil.io/resources/swe-book — open access).
  <https://www.hyrumslaw.com/>

- **Spec-ulation Keynote (full transcript)** — Rich Hickey (2016)
  *recorded-talk* · principal/advanced · confidence: high
  Argues every change is either accretion (growth — safe) or relaxation-of-requires / strengthening-of-provides vs. their opposites (breakage), and that honest evolution means never breaking: 'don't break things'. Gives the reader a precise, FP-flavored calculus for when a signature change is compatible — the substance under SemVer's MAJOR/MINOR/PATCH labels and a direct deepening of P8's contract-evolution silence.
  *Fills:* Deepens P8 and the API-evolution corner with a rigorous theory of change: 'growth vs breakage'. Fills the corner of *what versioning honestly means* that SemVer only labels.
  *Free basis:* Community-maintained verbatim transcript of the free Clojure/conj 2016 keynote (the video is free on ClojureTV). Satisfies the 'recorded talk WITH transcript' requirement directly.
  <https://github.com/matthiasn/talk-transcripts/blob/master/Hickey_Rich/Spec_ulation.md>

- **Semantic Versioning 2.0.0** — Tom Preston-Werner (2013)
  *essay* · foundational · confidence: high
  Defines MAJOR.MINOR.PATCH so that a version number *is* a machine-checkable claim about backward compatibility, and demands the public API be precisely declared. Read alongside Hickey's critique (which argues SemVer lets you legitimize breakage), it teaches the reader to see versioning as a contract surface, not bookkeeping — exactly ADR-0012's 'a claim must carry its substantiation'.
  *Fills:* The communication-protocol corner of P8/API evolution: SemVer is the *mechanized* convention that turns 'is this change contract-breaking?' into a checkable, greppable signal — the spirit of mechanization-over-memory applied to compatibility.
  *Free basis:* Public specification, free on its canonical site (CC-licensed).
  <https://semver.org/>

- **Parse, Don't Validate** — Alexis King (2019)
  *essay* · intermediate · confidence: high
  Shows that a boundary should parse foreign input into a richer type that statically guarantees validity, rather than validate-and-discard-the-evidence — so downstream code can't re-handle the error. This is the FP-native rationale for ADR-0012's ACL boundaries that 'translate-and-validate, never coerce' and for P9's optional/expected-at-the-edge discipline, written for exactly this reader.
  *Fills:* The constructive heart of P8/P9: a typed signature is the SSOT only if it *makes illegal states unrepresentable* — the boundary 'parses' untyped input into a type that carries proof. Directly mirrors P9's Port/ACL argv→string_view 'translate-at-the-edge, once' move.
  *Free basis:* Author's free blog; the definitive source of the slogan.
  <https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/>

- **Store config in the environment (Twelve-Factor App, Factor III)** — Adam Wiggins (Heroku) (2011)
  *essay* · foundational · confidence: high
  Gives the litmus test 'could you open-source the codebase right now without leaking config?' and the strict code/config separation principle. This is P4 at the system boundary: a swept/tuned value lives in the environment (a live source), not welded into the artifact — the operational generalization of the ADR's hp-registry HOT facet.
  *Fills:* P4 (live-not-frozen config) on the operational/deployment side the ADR mentions but does not develop: config is *everything that varies between deploys*, kept out of the code and read at run time — the deployment twin of 'read at point of use, not baked at construction'.
  *Free basis:* Free, open methodology on its canonical site.
  <https://12factor.net/config>

- **Feature Toggles (aka Feature Flags)** — Pete Hodgson (martinfowler.com) (2017)
  *essay* · intermediate · confidence: high
  Categorizes toggles by longevity and dynamism and warns that long-lived, hidden flags become a maintenance tar pit — the live-config analogue of P4's facet placement and P5's 'remove the band-aid'. The Knight Capital disaster (separate entry) is the worked failure of getting this exactly wrong.
  *Fills:* P4 plus the unnamed 'decouple deployment from release' corner — and a caution: a flag is live config whose category (release/ops/experiment/permission) and lifetime must be classified, echoing the ADR's classification + HOT/RESTART/INSTANCE facet discipline.
  *Free basis:* Free on martinfowler.com; the canonical reference on the pattern.
  <https://martinfowler.com/articles/feature-toggles.html>

- **SEC Order: In re Knight Capital Americas LLC (Release 34-70694)** — U.S. Securities and Exchange Commission (2013)
  *incident-report* · intermediate · confidence: high
  $460M lost in 45 minutes because a flag once used to arm retired 'Power Peg' code was repurposed for a new feature, and one of eight servers never got the new deploy — so the old code reactivated. The exact ADR-0012 failure cluster: a flag with two meanings over time (no SSOT for what the flag means), dead code not removed (P5 root-cause), and no loud deploy-consistency check.
  *Fills:* A Therac-25 *sibling* in the config/flag-and-deployment register, fitting the safety-culture/hard-won-failure spirit and P4/P5: a repurposed feature flag plus an incomplete deployment turned dormant dead code live.
  *Free basis:* Official U.S. government administrative order, public-domain and free on sec.gov (sec.gov blocks automated fetchers, but the document is openly downloadable in a browser; release number 34-70694 confirmed).
  <https://www.sec.gov/litigation/admin/2013/34-70694.pdf>

- **Cloudflare outage on November 18, 2025 (official post-mortem)** — Cloudflare (Matthew Prince et al.) (2025)
  *incident-report* · intermediate · confidence: high
  A change upstream caused a derived config file to double, exceeding a hardcoded limit in the Rust proxy and 5xx-ing ~20% of the web. A model of an honest post-mortem (the spirit ADR-0012 prizes) and a live lesson on P4/P5: config that is *derived* must be derived from one authority with bounds checked loudly, not assumed-and-asserted by a magic constant that fails closed without diagnosis.
  *Fills:* P4/P5 and the wire/config-as-data corner: a generated config *artifact* (a Bot-Management 'feature file') doubled in size from a database-permissions change and tripped a hardcoded size limit — a config-derivation SSOT failure plus a silent-limit-vs-fail-loud lesson, sibling to Therac in the safety-culture spirit.
  *Free basis:* Official company post-mortem, free on the Cloudflare blog.
  <https://blog.cloudflare.com/18-november-2025-outage/>


