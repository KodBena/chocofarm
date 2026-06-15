# A centralized, live, redis-backed hyperparameter registry (2026-06-15)

A design specification, not an implementation. It answers one operational question: **how
do we change a hyperparameter — the motivating case is a manual learning-rate drop — on a
running chocofarm experiment, without restarting it, without clobbering a second experiment
sharing the same redis, and without silently coercing a malformed value into a default?**

The shape of the answer is a **typed dataclass schema** (the contract), a **redis key
namespace** (the live store), a **read-at-point-of-use path** with honest staleness
semantics, a **write path** (an operator CLI) that logs loudly, and a **per-field
mutability facet** that draws the line between knobs that are safe to hot-swap (lr, loss
weights, λ) and knobs baked into a constructed object, a `jax.jit` closure, or a weight-
matrix shape (hidden width, feature dim, action-slot count, residual on/off). The honest
core of the design is that this line is *not* a matter of taste: the codebase already
*has* it, in the precise places where a value is read fresh each step versus closed over
once at construction, and §4 traces it from the actual code rather than imposing it.

The tone is the project's: name the tradeoffs, name the failure modes, and prefer "plain
dataclasses plus a thin redis layer beats Hydra, because X" to a framework mandate. The
recommendation in §8 is the former, and §8 says why.

One scope note up front, because it is the difference between a useful tool and a second
config system bolted alongside the first: today's source of truth for these knobs is
**argparse** (`exit_loop.py` has the largest surface; `train_value.py`, `dataset.py`,
`eval_az.py`, `tb_runner.py`, and the dual-bound driver each have their own). The maintainer's
word is **consolidate**, and §6 reconciles the two by making argparse the *bootstrap seed*
of the registry, not a parallel authority — the registry **layers over** argparse, it does
not replace the CLI nor run beside it.

---

## 0. What the code forces us to design around (the load-bearing facts)

These are read out of the actual code, not assumed. They constrain the design more than any
library choice does.

**C1 — The redis instance is a 1 GB `allkeys-lru` scratch store, and the eviction is real
on two vectors.** `chocofarm/az/parallel.py` is the only redis client today. It connects to
`127.0.0.1:6380` db 0 (env-overridable via `CHOCO_REDIS_HOST/PORT/DB`) and sets explicit 1h
TTLs on everything it writes: result blobs `az:res:<token>:<idx>:{X,PI,M,Y}` via `set(...,
ex=3600)` (`CHOCO_RESULT_TTL`, default `3600`), and weight blobs `az:w:<run>:<version>:{m,b}`
via `expire(..., 3600)`. Independently, the live server (verified read-only) carries
`maxmemory 1073741824` and `maxmemory-policy allkeys-lru` — so under memory pressure *even a
TTL-less key is eviction-eligible*. A registry key written naively would therefore be doubly
exposed: it would carry a TTL if it copied the transport's pattern, and it would be LRU-
evictable regardless. §2 fixes both vectors.

**C2 — The redis is systemd-managed, and its config file is root-owned.** Verified
read-only: `redis-memcache.service` (`Type=notify`, `User=bork`,
`ExecStart=/sbin/redis-server /etc/redis/redis-memcache.conf`, enabled, `WantedBy=multi-user.target`,
Main PID 1228, parent systemd). `save ""` and `appendonly no` — **no data persistence**, so
on restart redis re-reads the conf file and comes up empty. The conf file is `root:root 644`
inside `/etc/redis` which is `root:redis 750`; the redis process runs as `bork`, who is
neither root nor in the `redis` group. **Consequence (load-bearing, §2.3): a live `CONFIG SET
maxmemory-policy volatile-lru` works without root and takes effect immediately, but `CONFIG
REWRITE` will FAIL** — the redis process (as `bork`) cannot write the root-owned conf in a
directory it cannot traverse. Persisting the policy across a restart requires a one-line
root edit to the conf file (then `systemctl restart redis-memcache`), or a root-run
`CONFIG REWRITE`. The spec must not pretend `CONFIG REWRITE` is the operator's path here.

**C3 — The transport already namespaces by a per-run token, and the registry must mirror it.**
`ParallelExecutor.__init__` mints `self.run = uuid4().hex[:12]` and every key is
`az:w:<run>:...` / `az:res:<token>:...`. Two `exit_loop` processes on the same redis never
collide because their run tokens differ. The registry needs the same discipline under an
operator-meaningful name (§5), because the whole point of requirement 5 is concurrent
experiments that don't clobber each other.

**C4 — The hot/cold line is already in the code, at the `jax.jit` and constructor
boundaries.** This is the single most important fact for requirement 3. Surveyed in §4 from
`mlp_jax_train.py`, `mlp.py`, `gumbel_search.py`, `features.py`, `value_target.py`: some
values are read *fresh at point of use* (the loss weights `alpha`/`beta` are traced call-args
to the jit'd step; `y_mean`/`y_std` are read off the net every `train_step`; `c_puct`,
`c_visit`, `max_depth` are read off `self` each selection; the entire `value_target.py` is
pure functions reading args per call). Others are *baked once*: `optax.adam(learning_rate=lr,
b1, b2, eps)` and the L2-folded jit update closures are built once in `JaxTrainer.__init__`
and close over `lr/l2/betas/eps`; the net's weight matrices are sized by `in_dim/hidden/
n_actions/residual` at `ValueMLP` construction; the search's `m`/`n_sims` size the Sequential-
Halving bracket per construction; `use_jax_mlp` binds a forward function. The mutability facet
(§4) is a *reading* of this boundary, not an invention.

**C5 — A net is fit against a specific env, so env constants are instance-defining, not
knobs.** `env.py` constants — `teleport_overhead=12.0` (TELE_OH), `K=5` (present-count),
`N` (treasure count, derived from `data/instance.json`), `entry="CSNE"`, the per-treasure
`value` vector, the 44-face arrangement geometry — define the belief-MDP itself. Changing any
of them mid-run does not "re-tune" a running experiment; it silently invalidates the net,
the dual bound, and every cached value. These belong in the schema for *completeness and
provenance* (so a run records the instance it was fit to), but their mutability facet is the
strongest possible **restart-required, and a change is a new experiment** — §4 marks them
distinctly.

---

## 1. The schema — a hierarchy of typed dataclasses (requirement 1)

The schema is the typed contract. It is a top-level `ExperimentConfig` composed of nested
`@dataclass` groups, one per axis of the codebase's hyperparameter taxonomy. The grouping
follows the *actual* code cut (the survey in §4), not an a-priori taxonomy: each nested
dataclass corresponds to one file or one constructed object, so a reader can map a field back
to the line that consumes it.

Every field carries three things beyond its value: a Python **type**, a **default** (the
argparse default, so the registry seeds identically to launching the CLI today), and a
**mutability facet** (`HOT` / `RESTART` / `INSTANCE`) attached via `field(metadata=...)`. The
facet is what §4 surveys and §3.4 acts on. A small helper `hp(default, mut, doc, codec=...)`
wraps `field(default=..., metadata={"mut": ..., "doc": ..., "codec": ...})` so the
declaration stays readable.

```python
from dataclasses import dataclass, field
from enum import Enum

class Mut(Enum):
    HOT      = "hot"       # read fresh at point of use; safe to change on a running experiment
    RESTART  = "restart"   # baked into a constructed object / jit closure / array shape
    INSTANCE = "instance"  # defines the belief-MDP itself; a change is a NEW experiment, not a re-tune

def hp(default, mut, doc, codec="json"):
    return field(default=default, metadata={"mut": mut, "doc": doc, "codec": codec})
```

The nine groups (the taxonomy — see §4 for the full per-field enumeration with defaults and
the source line each is consumed at):

```python
@dataclass
class EnvConfig:           # chocofarm/model/env.py — INSTANCE-defining (C5)
    instance_path: str | None = hp(None,    Mut.INSTANCE, "geometry source (data/instance.json)")
    teleport_overhead: float  = hp(12.0,    Mut.INSTANCE, "TELE_OH; exit toll added to every run")
    present_k: int            = hp(5,       Mut.INSTANCE, "treasures present per world (env.K)")
    entry: str                = hp("CSNE",  Mut.INSTANCE, "entry teleport / start location")
    value_vector: list[float] | None = hp(None, Mut.INSTANCE, "per-treasure reward; None=unit")
    max_steps: int            = hp(40,      Mut.HOT,      "rollout horizon cap (per-call, not instance)")

@dataclass
class SearchConfig:        # chocofarm/az/gumbel_search.py
    m: int          = hp(12,    Mut.RESTART, "Gumbel root actions; sizes the SH bracket")
    n_sims: int     = hp(48,    Mut.RESTART, "sim budget; baked into the SH phase loop")
    c_puct: float   = hp(1.25,  Mut.HOT,     "PUCT exploration coeff (read per selection)")
    c_visit: float  = hp(50.0,  Mut.HOT,     "Danihelka sigma additive const")
    c_scale: float  = hp(1.0,   Mut.HOT,     "Danihelka sigma multiplicative scale")
    c_outcome: int  = hp(2,     Mut.HOT,     "leaf outcome-averaging count (loop bound, read per sim)")
    max_depth: int  = hp(24,    Mut.HOT,     "interior PUCT descent cutoff (soft, per recursion)")
    use_jax_mlp: bool = hp(False, Mut.RESTART, "jit forward vs numpy fast path; binds a fn")

@dataclass
class ValueTargetConfig:   # chocofarm/az/value_target.py — all pure functions, all HOT
    td_lambda: float    = hp(1.0,  Mut.HOT, "TD(lambda) blend; 1.0=pure MC (mutually excl. n_step)")
    n_step: int | None  = hp(None, Mut.HOT, "n-step bootstrap horizon; None=inf=pure MC")

@dataclass
class FeatureConfig:       # chocofarm/az/features.py — layout sizes the net input
    # the per-block multipliers (5 per treasure, 3 per detector, 6 global) are the input
    # dimension; they are RESTART because they size ValueMLP.W1. Exposed for provenance.
    per_treasure_channels: int = hp(5, Mut.RESTART, "marg,collected,available,dist,unc")
    per_detector_channels: int = hp(3, Mut.RESTART, "informative,p_pos,dist")
    global_channels: int       = hp(6, Mut.RESTART, "sharpness,n_coll,Sum_marg,exit,nonempty,Sum_unc (+n_tele)")

@dataclass
class ArchConfig:          # chocofarm/az/mlp.py + actions.py — all weight-matrix shapes
    hidden: int      = hp(256,  Mut.RESTART, "trunk width; sizes every weight matrix")
    residual: bool   = hp(False, Mut.RESTART, "gates the H×H residual block params")
    init_seed: int   = hp(0,    Mut.RESTART, "He-init RNG; consumed only at construction")
    # in_dim and n_actions are DERIVED from env (feature_dim / n_action_slots); recorded,
    # not set — the registry stores them for the drift check (§7), they are not free knobs.
    dtype: str       = hp("float32", Mut.RESTART, "CHOCO_AZ_DTYPE; read once at import")

@dataclass
class TrainConfig:         # chocofarm/az/mlp_jax_train.py — the jit boundary (C4)
    lr: float    = hp(1e-3, Mut.RESTART, "Adam lr — BAKED into optax.adam at construction (§4)")
    l2: float    = hp(1e-4, Mut.RESTART, "L2 — closed over by the jit update closure (§4)")
    beta1: float = hp(0.9,   Mut.RESTART, "Adam b1 — baked into optax.adam")
    beta2: float = hp(0.999, Mut.RESTART, "Adam b2 — baked into optax.adam")
    eps: float   = hp(1e-8,  Mut.RESTART, "Adam eps — baked into optax.adam")
    alpha: float = hp(1.0,   Mut.HOT,     "policy CE weight — traced call-arg, read each step")
    beta:  float = hp(1.0,   Mut.HOT,     "value MSE weight — traced call-arg, read each step")
    epochs: int  = hp(2,     Mut.HOT,     "train epochs over the buffer per iter (loop bound)")
    batch: int   = hp(256,   Mut.HOT,     "minibatch size (loop bound, read at iter start)")

@dataclass
class ExItLoopConfig:      # chocofarm/az/exit_loop.py — the outer loop
    iters: int          = hp(40,     Mut.HOT,     "outer ExIt iterations (loop bound)")
    episodes: int       = hp(300,    Mut.HOT,     "self-play episodes/iter (read at iter start)")
    window: int         = hp(5,      Mut.HOT,     "replay window in iterations")
    lam: float          = hp(0.0855, Mut.HOT,     "pinned lambda0 (static-floor rate)")
    explore_plies: int  = hp(4,      Mut.HOT,     "plies sampling executed action from pi'")
    seed: int           = hp(7,      Mut.RESTART, "master RNG seed; folded into worker seeds at launch")

@dataclass
class EvalConfig:          # exit_loop eval block + eval_az.py
    eval_n: int    = hp(200,   Mut.HOT,     "held-out eval episodes/iter")
    eval_seed: int = hp(12345, Mut.HOT,     "eval world draw seed")

@dataclass
class ParallelConfig:      # chocofarm/az/parallel.py
    workers: int = hp(4,        Mut.RESTART, "process-pool size; pool built once before the loop")
    cores: str   = hp("0,1,2,3", Mut.RESTART, "core-pin list; set in the pool initializer")
    redis_host: str = hp("127.0.0.1", Mut.RESTART, "CHOCO_REDIS_HOST")
    redis_port: int = hp(6380,        Mut.RESTART, "CHOCO_REDIS_PORT")
    redis_db: int   = hp(0,           Mut.RESTART, "CHOCO_REDIS_DB")

@dataclass
class BoundsConfig:        # chocofarm/bounds/{eval_bound,info_relaxation}.py
    vhat: str          = hp("none",     Mut.HOT,  "V-hat generator: none|zero|analytic|decomp|exact|az-ckpt")
    vhat_lam: float | None = hp(None,   Mut.HOT,  "reference lambda* fixing V-hat (Route A); None=Route B")
    max_inner_states: int = hp(2_000_000, Mut.HOT, "inner-DP cap; ABORTS LOUDLY, never truncates (§7)")
    lam_lo: float      = hp(0.0,        Mut.HOT,  "lambda-scan bracket low")
    lam_hi: float      = hp(0.40,       Mut.HOT,  "lambda-scan bracket high")
    lam_tol: float     = hp(1e-4,       Mut.HOT,  "bisection convergence tolerance")
    max_iter: int      = hp(40,         Mut.HOT,  "bisection iteration cap")

@dataclass
class ExperimentConfig:
    experiment_id: str
    env:    EnvConfig    = field(default_factory=EnvConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    value:  ValueTargetConfig = field(default_factory=ValueTargetConfig)
    feat:   FeatureConfig = field(default_factory=FeatureConfig)
    arch:   ArchConfig   = field(default_factory=ArchConfig)
    train:  TrainConfig  = field(default_factory=TrainConfig)
    loop:   ExItLoopConfig = field(default_factory=ExItLoopConfig)
    eval:   EvalConfig   = field(default_factory=EvalConfig)
    par:    ParallelConfig = field(default_factory=ParallelConfig)
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    schema_version: int = 1   # bumped when the schema shape changes; gates the drift check (§7)
```

The dataclass *is* the contract: it is the single place the taxonomy lives, the defaults
live, the per-field mutability lives, and (via `metadata["codec"]`) the per-field
(de)serialization lives. Nothing about a hyperparameter is recorded in two places.

---

## 2. The redis store and the eviction fix (requirement 2 — be exact)

### 2.1 Registry keys carry NO TTL

The transport's pattern is `set(..., ex=3600)` / `expire(..., 3600)` — correct for a
transient blob that is read-and-deleted the same iteration, wrong for a registry value that
must outlive the experiment that reads it. **Registry writes use a bare `SET` with no `ex`,
no `expire`** (or, equivalently, `SET ... KEEPTTL` is *not* used; a fresh `SET` with no
expiry leaves TTL = -1, "no expiry"). This is the first eviction vector closed: a registry
key is never on a clock.

### 2.2 The policy must move to `volatile-lru`

A no-TTL key is still LRU-evictable under `allkeys-lru` (C1, second vector). The fix is to
change the eviction policy to **`volatile-lru`** — "evict only among keys that have a TTL."
This is exactly right for the mixed workload on this instance:

- **Registry keys** (no TTL) become eviction-*ineligible* — protected even at the 1 GB
  ceiling. Correct: losing a live hyperparameter under memory pressure is the silent failure
  ADR-0002 forbids.
- **Transport blobs** (`az:res:*`, `az:w:*`, all written with a 1h TTL) stay eviction-
  *eligible* — they still self-clean under pressure, preserving the transport's design
  intent (the `parallel.py` docstring explicitly relies on "the eviction window is small").

**Honest interaction worth recording (C1 + the post-mortem in `parallel.py`):** there is a
known leak of *TTL-less* `az:res:*` keys — when a generation fan-out aborts via the loud
timeout (Fix A), the parent never reaches the delete, and the post-mortem found ~980 such
keys with TTL = -1; a live read-only sample found ~145 of 300 `az:res:*` keys already
TTL-less. Under `volatile-lru`, *those leaked transport keys also become eviction-ineligible*
(they look like registry keys to the policy — no TTL). This is a real, small downside of the
move: the policy can no longer LRU-reap the leak. It is acceptable because (a) the leak is
bounded and rare (only on aborts), (b) the registry-protection benefit dominates, and (c) the
clean fix for the leak is orthogonal — the aborted-iteration cleanup should `DEL` its own
`res_token:*` keys in the abort path (a `parallel.py` change, out of scope here, but worth a
work-status note). The spec does not paper over this: moving to `volatile-lru` trades "LRU can
reap leaked transport keys" for "LRU cannot evict registry keys," and the trade is correct,
but the leak should be fixed at its source rather than relied on the policy to mop up.

### 2.3 Persisting the change across a redis restart — the exact mechanism

This is where C2 is load-bearing and where a naive spec would be wrong. Two layers:

1. **Live, immediate (no root, takes effect now):**
   ```
   redis-cli -p 6380 CONFIG SET maxmemory-policy volatile-lru
   ```
   Any client can issue this; it changes the running server's behavior immediately. This is
   what the registry's *bootstrap* (§6) issues on first use, idempotently (it is a no-op if
   already `volatile-lru`).

2. **Persistent across restart (requires root):** the server has `save ""` / `appendonly no`
   — no data persistence — and re-reads `/etc/redis/redis-memcache.conf` on every restart.
   The conf currently says `maxmemory-policy allkeys-lru`. **`CONFIG REWRITE` will fail here**
   (C2: the conf is `root:root 644` in `/etc/redis` `root:redis 750`; redis runs as `bork`
   and cannot write it). So the operator path is a **one-line root edit** of the conf,
   followed by a service restart:
   ```
   sudo sed -i 's/^maxmemory-policy allkeys-lru/maxmemory-policy volatile-lru/' \
       /etc/redis/redis-memcache.conf
   sudo systemctl restart redis-memcache
   ```
   (Equivalently, a root-run `redis-cli ... CONFIG REWRITE` would work *because the rewrite is
   then performed by a root process* — but the running server is not root, so the operator,
   not the server, must hold root. The conf edit is the clearer instruction.)

   **Restart caveat that the spec must state plainly:** because `save ""`/`appendonly no`, a
   `systemctl restart` **wipes all keys** — every registry value and every transport blob.
   So the restart-to-persist-the-policy step is destructive to live state. The correct order
   is: (i) do the conf edit + restart *before* launching experiments (one-time host setup), or
   (ii) if a restart is unavoidable mid-campaign, the registry must be re-seeded from the
   dataclass defaults + any operator overrides afterward (§6 bootstrap is idempotent and makes
   this a single re-run). The registry's bootstrap is designed to be the recovery path here.

The `maxmemory 1073741824` (1 GB) ceiling is left unchanged — the registry footprint is
negligible (a few KB per experiment as a single JSON blob, §5.2), so the ceiling is not the
constraint; the *policy* is.

---

## 3. The read path and its semantics (requirement 3 — be honest)

### 3.1 Per-access GET vs cached snapshot — the recommendation

Two extremes, and the chosen middle:

- **Per-access `GET`** at every point of use: a hyperparameter change is visible at the very
  next read, with zero staleness. But a `GET` is a redis round-trip; the value head's forward
  is read inside the search hot path (~hundreds of net evals per episode), and a per-eval
  `GET` of `c_puct` would add a network round-trip to the per-node cost the
  `alphazero-surrogate-design.md` already identified as latency-sensitive (the feature build is
  the per-node bottleneck at ~1.2 ms; a sub-ms `GET` per node is not free against that). This
  is over-eager.

- **Process-lifetime snapshot** read once at launch: zero per-read cost, but defeats the
  entire purpose — a change never takes effect on a running experiment.

**Recommended: a per-process cached snapshot with bounded-staleness refresh, keyed to the
loop's natural boundaries.** The registry value is read into a typed `ExperimentConfig`
snapshot **once per outer iteration boundary** in `exit_loop.run` (the top of the
`for it in range(...)` loop, where `gen_worlds`/`eval_worlds` are already drawn — a natural
re-read point), and the search/trainer read their HOT fields off that snapshot. Within an
iteration the snapshot is fixed (so an episode's hyperparameters do not change mid-episode —
a desirable atomicity, §3.3). Between iterations, the change propagates. The staleness window
is therefore **at most one outer iteration** — minutes, which is the right grain for a manual
lr drop (you do not need it to take effect mid-iteration; you need it to take effect without
killing the run).

This is cheap: one `GET` (or one `MGET`/blob `GET`, §5.2) per iteration, not per node.

### 3.2 Polling vs pub/sub for propagation — recommendation and why

- **Polling** (re-read the snapshot each iteration boundary): trivial, no extra connection,
  no missed-message risk, and the iteration boundary is already a synchronization point. The
  cost is the staleness window of §3.1 (≤ one iteration), which is acceptable for this use
  case.

- **redis pub/sub** (`PUBLISH choco:hp:<id>:changed`; the loop `SUBSCRIBE`s and refreshes on
  message): tighter propagation (sub-second), but it adds a second redis connection, a
  background listener thread, and a missed-message failure mode (pub/sub is fire-and-forget;
  a subscriber that was momentarily disconnected misses the notification and never refreshes —
  a *silent* staleness, exactly the ADR-0002 failure). The parallel workers are spawned
  processes; wiring a subscriber into each is more machinery than the use case earns.

**Recommended: polling at the iteration boundary, with pub/sub as a deferred enrichment only
if a future use case needs sub-iteration propagation.** Polling's worst case is bounded and
loud (a stale value is at most one iteration old and is logged on refresh, §4.5); pub/sub's
worst case is unbounded and silent (a missed message). For a manual-lr-drop tool, bounded-and-
loud beats tight-and-silent. (If pub/sub is later added, it must be a *hint to refresh* layered
over the poll, never the sole path — the poll remains the correctness floor.)

The parallel workers are a wrinkle: they run frozen weights published per version, and they do
not need most HOT knobs (the trainer is central in the parent — `parallel.py` docstring:
"TRAIN stays central in the parent"). The HOT training knobs (`alpha`, `beta`, `lr` re-pin if
that path is built, `epochs`, `batch`) are read in the **parent**, which is the only process
that polls. The few search knobs a worker uses (`c_puct` etc.) are constructed into the
worker's `GumbelAZSearch` at `_ensure_net` time, i.e. they refresh when the weight version
bumps each iteration — the worker's natural refresh point coincides with the parent's poll. So
the worker path needs no separate subscription; the existing version-bump cadence carries the
search-knob refresh for free.

### 3.3 The staleness / atomicity window

Stated plainly so it cannot surprise an operator:

- A HOT change written at time *t* is visible to the running experiment at the **next
  iteration boundary** after *t* — staleness ≤ one outer iteration.
- A change is **atomic within an iteration**: the snapshot is read once per iteration, so an
  episode never sees a half-applied multi-field change (§3.3 + the write-path `MULTI`/blob in
  §4.2 guarantee a reader sees either all of a related-set change or none).
- A RESTART/INSTANCE change written mid-run does **not** silently take effect; §3.4 defines
  what the reader does.

### 3.4 Mutability facet semantics — what a reader does (the recommendation)

This is the heart of requirement 3's "which knobs are safe to hot-swap." Each field's
`metadata["mut"]` drives reader behavior, decided at the iteration-boundary refresh:

- **HOT** field changed: apply it. Log the change loudly (§4.5). For HOT fields that are
  themselves read deeper (e.g. `c_puct` read per selection off `self`), the iteration-boundary
  refresh rebuilds the small constructed objects that hold them (the per-iteration
  `GumbelAZSearch`/`GumbelPolicy` are already reconstructed each iteration in the serial path,
  and the parallel path rebuilds them on the version bump) — so a HOT search-knob change is
  picked up by reconstructing the search object at the boundary, which the loop already does.

- **RESTART** field changed mid-run: **refuse loudly.** The reader compares the registry's
  value against the value the running process was *constructed* with (recorded at launch in
  the snapshot's `launched_with` shadow, §6). If a RESTART field differs, raise a loud
  `RuntimeError` naming the field, the construction-time value, and the new value, and instruct
  the operator to restart with `--resume` to adopt it. **Rationale for refuse-over-ignore:** an
  `lr` baked into `optax.adam` (C4) that the registry now says is `1e-4` but the running
  optimizer still uses at `1e-3` is a *lie the registry tells* — the operator believes they
  dropped the lr and they did not. Silently ignoring is the ADR-0002 silent-failure; warn-and-
  continue still leaves the operator believing a change took that did not. Refusing loudly is
  the only option that keeps the operator's mental model true. The refusal is cheap to recover
  from: the loop checkpoints every iteration, so a restart-with-`--resume` adopts the new
  RESTART value and loses nothing. (See §3.5 for the one motivating-case nuance: `lr` is
  RESTART in *this* codebase, and what that means for the manual-lr-drop use case.)

- **INSTANCE** field changed mid-run: refuse loudly, with a stronger message — this is not a
  re-tune, it is a different experiment; the running net is invalid against the new env. Same
  loud `RuntimeError`, different remediation text (start a *new* experiment_id, do not
  `--resume`).

### 3.5 The motivating case, honestly: lr is RESTART in this code

The maintainer's motivating example is a manual lr drop. The honest finding from §4 is that
**`lr` is `RESTART` in the current implementation**, because `JaxTrainer.__init__` builds
`optax.adam(learning_rate=self.lr, ...)` once and the jit'd update closures capture it (C4).
So as the code stands, dropping `lr` via the registry triggers the §3.4 *refuse-loudly* path:
the operator is told "lr change recorded; restart with `--resume runs/.../latest_net.npz` to
adopt it." Because the loop already checkpoints every iteration and `--resume` reloads the full
net and re-inits a fresh optax optimizer (verified in `exit_loop.run`), this is a clean,
near-zero-loss operation — and it is, notably, *exactly the manual-lr-drop workflow the handoff
doc already prescribes* (the queued anneal experiment resumes from `matched_reson` at
`--lr 1e-4`). The registry does not make lr hot by itself; it makes the lr-drop *recorded,
namespaced, logged, and one-command to adopt*.

There is a clean optional upgrade that would make `lr` genuinely HOT: `optax`'s
`inject_hyperparams` (or a schedule that reads a live scalar) lets the learning rate be a
traced per-step value rather than a closed-over constant. If the maintainer wants lr to drop
*without even a `--resume`*, the registry design supports it — flip `lr`'s facet to HOT once
`JaxTrainer` is refactored to inject lr per step. The spec flags this as the one targeted code
change that would upgrade the motivating case from "one-command restart" to "fully live," and
recommends it as a *follow-on*, not a prerequisite — the registry is useful the day it ships
even with lr as RESTART, because the recorded+logged+namespaced drop is the actual operational
win, and the `--resume` cost is one iteration's checkpoint.

### 3.6 Deserialization + validation back into the typed dataclass — fail loudly

redis stores bytes; the dataclass is typed. The read path is:

1. `GET` the blob (§5.2: one JSON blob per experiment).
2. `json.loads` → a plain dict.
3. **Validate and reconstruct** into `ExperimentConfig` via a typed decoder. Each field's
   `metadata["codec"]` (default `"json"`) names how to decode; the decoder checks the value's
   type against the dataclass field annotation and **raises a loud `RegistryDecodeError`** on
   any mismatch (wrong type, out-of-domain value, unknown key, missing required key). A
   malformed or missing value is **never coerced to a default** — that is the ADR-0002
   silent-failure this whole section exists to prevent. (The dual bound's own `vhat=None` vs
   `vhat_zero` confusion, recorded in `dual-bound.md` §4.2, is the cautionary tale for why a
   "fall back to a sensible default" decode is dangerous: a silently-defaulted hyperparameter
   produced a *wrong number that looked right*.)
4. Cross-field invariants are checked here too (the same loud failure): e.g. `td_lambda < 1.0`
   and `n_step is not None` together is the mutually-exclusive error `exit_loop` already
   raises (`ValueError`); the decoder enforces it at read time so a bad *combination* fails as
   loudly as a bad *value*.

`dacite` (§8) is the natural library for step 3 if a dependency is acceptable — it does
exactly "dict → nested dataclass with type checking." The §8 verdict weighs it.

---

## 4. The full hyperparameter surface, enumerated (requirement 6 — NOT just the trainer)

This is the survey requirement 6 exists to force. The maintainer flagged it explicitly to
prevent the malicious-compliance reading "the JaxTrainer knobs are the hyperparameters." They
are not. The surface spans **env, search, value-target, features, architecture, training/
optimizer, the ExIt loop, eval, parallelism, and the dual-bound solver** — read out of the
code, with each field's mutability facet justified by *where the code consumes it*.

### 4.1 Search — `gumbel_search.py`

| field | type | default | mut | why that facet |
|---|---|---|---|---|
| `m` | int | 12 | RESTART | sizes the Sequential-Halving bracket / `n_phases=⌈log2 m⌉`; baked per-construction |
| `n_sims` | int | 48 | RESTART | sim budget baked into the SH phase loop |
| `c_puct` | float | 1.25 | HOT | read off `self.c_puct` per selection in `_puct_select` |
| `c_visit` | float | 50.0 | HOT | read off `self` in `_sigma_scale` each call (Danihelka σ additive) |
| `c_scale` | float | 1.0 | HOT | read off `self` in `_sigma_scale` (σ multiplicative) |
| `c_outcome` | int | 2 | HOT | leaf outcome-averaging loop bound, read per simulation |
| `max_depth` | int | 24 | HOT | interior descent cutoff, compared per recursion |
| `use_jax_mlp` | bool | False | RESTART | binds the forward fn (`net.predict_both` vs jit MLP) at construction |

Currently hardcoded (not reachable as args today; the spec flags them for a future surface,
not as initial registry fields): the SH **halving fraction** (`len(scored)//2` — no knob to
drop a non-half fraction), the logit/prior sentinels (`-1e30`, `-1e29`, `1e-12`), the
per-call `temperature` (default `0.0`, a method arg not a constructor knob — the most
hot-swappable thing in the file). The `n_slots`/`term_slot`/bijection tables are env-derived
array shapes (RESTART, but not free knobs — they follow from env).

### 4.2 Value target — `value_target.py`

Pure functions, no constructed state, no jit — **everything here is HOT** (read as a per-call
argument): `td_lambda` (∈[0,1], 1.0=pure MC), `n_step` (≥1 or None=∞=pure MC, takes
precedence over `td_lambda`), `lam`, `exit_c`. The mutual-exclusion of `td_lambda<1` and
`n_step` is the cross-field invariant §3.6 enforces.

### 4.3 Features — `features.py`

The feature *dimension* (=241 on the live env) is **derived**: `N*5 + nD*3 + (6 + n_tel)`.
The per-block multipliers (5 per treasure, 3 per detector, 6 global) are RESTART — they size
`ValueMLP.W1` and are coupled to the `actions.py` mask slice offsets (`2N..3N`, `5N..`). The
distance normalizer `map_diag` (bbox diagonal) is RESTART (baked into `FeatureBuilder` and its
`_loc_cache`; a net trained on one scale cannot read another). `_belief_cache_cap` (50000) is
a pure memory/perf knob (HOT in principle, correctness-neutral).

### 4.4 Architecture — `mlp.py` + `actions.py`

| field | type | default | mut | why |
|---|---|---|---|---|
| `in_dim` | int | =feature_dim (241) | RESTART | sizes `W1`; derived from env, recorded for drift check |
| `hidden` (H) | int | 256 | RESTART | sizes every weight matrix (`W1,W2,Wv,Wp,Wr*`) |
| `n_actions` | int | =n_action_slots (65) | RESTART | sizes the policy head `Wp`; derived from env |
| `seed` | int | 0 | RESTART | He-init RNG, consumed only at construction |
| `residual` | bool | False | RESTART | gates whether the `Wr1/br1/Wr2/br2` block params exist |
| `y_mean` | float | 0.0 | HOT | de-standardization scalar, read each `predict_value`; re-settable via `set_value_scale` (the loop re-pins it from buffer stats each iter) |
| `y_std` | float | 1.0 | HOT | same; the trainer reads `net.y_mean/y_std` *fresh every train_step* |
| `dtype` (`CHOCO_AZ_DTYPE`) | enum | float32 | RESTART | read once at `dtypes.py` import; flips the f32 cache + train precision |

`n_action_slots(env) = N + nD + 1` (the `+1` is TERMINATE). Sizes the policy head — RESTART,
env-derived.

### 4.5 Training / optimizer — `mlp_jax_train.py` (the jit boundary; C4)

This is where the prompt's worry ("anything baked into a compiled JAX function") is concrete.
The `optax.adam` transform and the two `@jax.jit` update closures are built **once** in
`JaxTrainer.__init__`, closing over `lr`, `l2`, `betas`, `eps`:

| field | type | default | mut | why |
|---|---|---|---|---|
| `lr` | float | 1e-3 | **RESTART** | passed to `optax.adam(learning_rate=lr)` at construction; **not** re-read per step (the signature *looks* like it could be per-step, but `self.opt` is built once) — this is the §3.5 finding |
| `l2` | float | 1e-4 (0.0 in trainer default; 1e-4 from exit_loop) | **RESTART** | closed over by `_make_az_update(opt, l2)` at trace time as a Python float |
| `beta1, beta2` | float | 0.9, 0.999 | **RESTART** | baked into `optax.adam(b1, b2)` |
| `eps` | float | 1e-8 | **RESTART** | baked into `optax.adam(eps)` |
| `alpha` | float | 1.0 | **HOT** | traced call-arg to the jit'd `_az_update`, read fresh each `train_step` |
| `beta` | float | 1.0 | **HOT** | traced call-arg, read fresh each `train_step` |

The Adam *moment state* persists across steps (and resets on `sync_from_net`). The
hot/restart split here is the single most important correctness fact in the spec: **`alpha`/
`beta` are genuinely hot; `lr`/`l2`/`betas`/`eps` are genuinely baked.** The schema marks them
accordingly, and §3.4's refuse-loudly fires if an operator tries to drop `lr` on a running
process without `--resume`.

### 4.6 ExIt loop — `exit_loop.py`

`iters`, `episodes`, `window`, `epochs`, `batch` are loop bounds read at iteration start
(HOT). `lam` (=λ₀=0.0855, the pinned static-floor rate) is HOT (threaded into the value target
per episode). `explore_plies` HOT. `seed` is RESTART (folded into per-worker/per-episode seeds
at launch; changing it mid-run breaks the parallel≈serial determinism contract). `td_lambda`/
`n_step` are the §4.2 value-target knobs surfaced as loop args. `alpha`/`beta` (§4.5). The
warm-start/resume paths (`--init-weights`, `--resume`) are launch-time net-loading choices, not
registry knobs (they select an initial net, recorded for provenance).

### 4.7 Eval — `exit_loop` eval block + `eval_az.py` + `tb_runner.py`

`eval_n` (200), `eval_seed` (12345) — HOT (MC sample size + draw seed). `eval_az.py`:
`LAM0=0.0855` (the net's training λ; eval at a different λ is not apples-to-apples — treat as
fixed per checkpoint), `--it` (ISMCTS budget, 200, RESTART-per-run), `--n` (eval episodes,
300, HOT), `--chunk` (20, HOT logging granularity). `tb_runner.py`: `--method`
(nmcs/ismcts/uct, RESTART-per-run), `--configs`, `--batch` (1), `--rss_cap_mb` (1200, a memory
self-guard), the in-loop Dinkelbach EMA weights (0.7/0.3, hardcoded), initial λ (0.08).

### 4.8 Parallelism — `parallel.py`

`workers` (4) and `cores` ("0,1,2,3") are RESTART — the process pool is built once before the
loop and workers are core-pinned in the initializer. `redis_host/port/db` (127.0.0.1/6380/0,
env-driven) RESTART. The socket/result timeouts (`CHOCO_REDIS_SOCKET_TIMEOUT=60`,
`CHOCO_RESULT_TIMEOUT=600`, `CHOCO_POOL_JOIN_TIMEOUT=30`) and `CHOCO_RESULT_TTL=3600` are
operational safety knobs (env-driven; could be registry fields with RESTART facet, since the
pool/connection is built once).

### 4.9 Dual-bound solver — `eval_bound.py` + `info_relaxation.py`

| field | type | default | mut | why |
|---|---|---|---|---|
| `vhat` | enum | none | HOT | the V̂ generator: `none` (z≡0 no-penalty), `zero` (V̂≡0 reward-deviation), `analytic` (marginal V̂₀), `decomp` (DecompVhat), `exact` (ExactBeliefVhat, MiniEnv only), `az-ckpt` (frozen AZ value net) — injected per `PenalizedClairvoyant` |
| `vhat_lam` | float? | None | HOT | reference λ* (Route A fixes V̂; None=Route B rebuild-per-λ) |
| `max_inner_states` | int | 2_000_000 | HOT | the inner-DP cap that **ABORTS LOUDLY, never truncates** — §7 names this the one correctness-load-bearing knob |
| `lam_lo, lam_hi` | float | 0.0, 0.40 | HOT | λ-scan bisection bracket (driver default hi=0.30, callers pass 0.40; auto-widens ×1.5 ≤8 tries) |
| `lam_tol` | float | 1e-4 | HOT | bisection convergence tolerance |
| `max_iter` | int | 40 | HOT | bisection iteration cap |
| `restrict_faces` | bool | True | HOT | currently-inert over-approximation hook (a lower-bounding prune would break the bound — see §7) |
| `horizon` (DecompVhat) | int | 1 | HOT | macro look-ahead in the decomp V̂ |

The dual-bound knobs are nearly all HOT because the solver is reconstructed per invocation —
but `max_inner_states` and `restrict_faces` carry a *correctness contract* (§7) that the
registry must not let a careless change violate.

This is the whole surface. The trainer is one of ten groups; requirement 6 is satisfied by the
schema covering all ten.

---

## 5. The write path and the namespace (requirements 4 and 5)

### 5.1 Namespace isolation — the key schema (requirement 5)

Mirror the transport's run-token discipline (C3) under an operator-meaningful name. The
registry namespace is keyed by an **`experiment_id`**:

```
choco:hp:<experiment_id>            -> the serialized ExperimentConfig blob (§5.2)
choco:hp:<experiment_id>:meta       -> {schema_version, created_at, last_write_at, writer}
```

`<experiment_id>` is an operator-chosen, human-meaningful string (e.g. `lr1e4_resume`,
matching the handoff's run directory naming `runs/lr1e4_resume`) — *not* the random
`uuid4().hex[:12]` transport token, because the operator must be able to address it from the
CLI without first reading it out of a log. Two concurrent experiments choose two distinct
ids; their key prefixes never overlap; a write to one cannot touch the other. The id is
**passed to `exit_loop` at launch** (a new `--experiment-id` arg, defaulting to the ckpt-dir
basename so the existing `--ckpt-dir runs/lr1e4_resume` implies `experiment_id=lr1e4_resume`
with zero new ceremony), and a reader **binds to its namespace** via that same id held on the
config snapshot. The parallel workers inherit the id through the existing task args (the
parent already threads `run`, `lam`, etc. into each task tuple — the id rides the same way).

### 5.2 Single blob per experiment vs key-per-field — the tradeoff and the choice

Two designs:

- **Key-per-field:** `choco:hp:<id>:train.lr`, `choco:hp:<id>:search.c_puct`, … One small
  string per leaf. Pros: a write touches exactly one field; `MSET` of a related set is natural;
  a reader can `MGET` only the fields it needs. Cons: a *consistent multi-field read* needs a
  pipeline and is not atomic against a concurrent multi-field write (a reader could see lr
  updated but l2 not yet); the key count multiplies (~50 keys/experiment); a schema-shape change
  scatters across many keys.

- **Single serialized blob per experiment** (`choco:hp:<id>` → one JSON of the whole
  `ExperimentConfig`): a read is one `GET` + one decode; a write is one `SET` of the whole
  (mutated) blob. Pros: **atomic by construction** — a reader always sees a complete, self-
  consistent config (the §3.3 atomicity falls out for free, no `MULTI` needed for the common
  case); the §3.6 typed decode validates the *whole* config in one pass (cross-field invariants
  included); the drift check (§7) compares one blob's `schema_version`. Cons: a write must read-
  modify-write the whole blob (a lost-update race if two writers race — handled in §5.4); a
  field change rewrites the whole (small — a few KB) blob.

**Recommended: the single serialized JSON blob per experiment.** The atomicity and whole-config
validation it gives for free are worth more than key-per-field's finer write granularity, and
the blob is small enough (the whole `ExperimentConfig` is well under 4 KB of JSON) that read-
modify-write cost is negligible. JSON over msgpack: human-readable for `redis-cli GET`
inspection and operator debugging, which matters more here than the byte savings (the blob is
tiny). The atomic-multi-field-write requirement (4's "related set atomically") is then a
read-modify-write of the one blob under an optimistic-concurrency guard (§5.4), not a `MULTI`
over many keys.

### 5.3 The operator write CLI (requirement 4)

A small `chocofarm.hp.registry` CLI / module, fail-loud throughout:

```
# read the whole config (decoded + validated, pretty-printed)
python -m chocofarm.hp.registry get --experiment-id lr1e4_resume

# set one field (path.to.field), validated against the schema BEFORE write
python -m chocofarm.hp.registry set --experiment-id lr1e4_resume train.lr 1e-4

# set a related SET atomically (the motivating multi-field case: drop lr AND raise l2)
python -m chocofarm.hp.registry set --experiment-id lr1e4_resume \
       train.lr 1e-4 train.l2 5e-4

# seed a fresh experiment from the dataclass defaults (the §6 bootstrap, idempotent)
python -m chocofarm.hp.registry init --experiment-id lr1e4_resume [--from-argparse ...]
```

`set` mechanics: (i) `GET` the current blob, decode to `ExperimentConfig`; (ii) apply the
field path(s) to a copy, **type-checking each value against the schema field annotation and
running the cross-field invariants** — a bad value fails *here*, before any write touches redis
(ADR-0002: refuse a malformed write at the source, never store-then-discover); (iii) on a
RESTART/INSTANCE field, the CLI *warns the operator* that running processes will refuse-loudly
and a `--resume` restart is needed to adopt it (it still writes — the recorded change is the
point — but it tells the truth about when it takes effect); (iv) write back under the §5.4
guard; (v) **log the applied change loudly** (§4's "log every applied change").

### 5.4 Atomic multi-field change + the two-writers race (requirement 4 + the failure mode)

The motivating atomic case (drop lr *and* raise l2 together) and the two-writers-racing
failure mode are the same problem: a lost update on the read-modify-write of the blob. The
guard is **optimistic concurrency via redis `WATCH`/`MULTI`/`EXEC`** (a redis transaction):

```
WATCH choco:hp:<id>
cfg = decode(GET choco:hp:<id>)
cfg2 = apply_fields(cfg, {train.lr: 1e-4, train.l2: 5e-4})   # validated in-memory
MULTI
  SET choco:hp:<id> encode(cfg2)
  SET choco:hp:<id>:meta {... last_write_at, writer ...}
EXEC      # aborts (returns nil) if choco:hp:<id> changed since WATCH
```

If `EXEC` aborts (another writer wrote between the `WATCH` and the `EXEC`), the CLI **retries
the read-modify-write a bounded number of times, then fails loudly** if it still cannot land
(ADR-0002: do not silently drop the operator's change; do not silently clobber the other
writer's). The multi-field set is atomic against readers (the blob is replaced in one `SET`,
and a reader's `GET` sees either the old or the new complete blob — never a half) and atomic
against writers (the `WATCH` detects the race). This is the "related-set atomically" guarantee
requirement 4 asks for, with the lost-update race named and handled rather than wished away.

### 5.5 Observability — every applied change logged loudly (requirement 4 + design concern)

Every `set` that lands logs at the project's loud, filterable level: a structured line naming
the `experiment_id`, the field path(s), the old value(s), the new value(s), the writer (a
`CHOCO_HP_WRITER` env / `getpass.getuser()` attribution, mirroring the transport's
actor-awareness instinct), and the timestamp. The *reader* logs symmetrically: when an
iteration-boundary refresh detects a changed HOT field, it logs "applied <field>: <old> →
<new> at iter N"; when it detects a changed RESTART/INSTANCE field, it logs the loud refusal
(§3.4) before raising. So both the write *and* its application are visible in the run log — an
operator can always answer "did my lr drop take effect, and when," which is the whole point of
making the change loud (and the ADR-0002 instinct: never leave the operator in the dark about
whether the change they made is the change the running process sees).

---

## 6. Bootstrap, defaults, and the argparse relationship (the consolidation)

The maintainer's word is **consolidate** — the registry must not be a second config system
bolted alongside argparse. The reconciliation:

**The registry layers over argparse; argparse seeds the registry; neither is duplicated.**

- The **dataclass defaults are the argparse defaults** (§1 — each `hp(default, ...)` is the
  exact `ap.add_argument(..., default=...)` value). One source of defaults.
- At **launch**, `exit_loop` (and the other entry points) build an `ExperimentConfig` from the
  parsed argparse namespace (a thin `from_argparse(args)` adapter — argparse remains the launch
  CLI, unchanged), then **seed the registry**: `init --experiment-id <id>` writes that config
  as the experiment's blob *if it does not already exist* (idempotent — a `--resume` of an
  existing experiment re-binds to the existing blob rather than overwriting operator overrides).
  So the existing CLI invocation is the seed; the operator writes nothing by hand to start.
- During the **run**, the loop reads HOT fields from the registry snapshot (§3), not from
  `args`. The argparse namespace is captured once into the snapshot's `launched_with` shadow —
  this is what §3.4's RESTART-refusal compares against (the construction-time truth). So
  argparse defines *launch*; the registry defines *live*; the `launched_with` shadow is the
  bridge that lets the reader tell "you changed a baked field" from "you changed a hot one."
- The relationship is therefore **wrap + layer, not replace**: argparse still parses the CLI
  and still constructs the process; the registry takes over as the authority for HOT fields
  once running, seeded by and reconciled against the argparse values. There is exactly one
  place each hyperparameter's default lives (the dataclass), one place its live value lives
  (the registry blob), and one place its construction-time value lives (the `launched_with`
  shadow) — no field is recorded in two authorities.

This also gives the §2.3 restart-recovery a home: after a redis restart wipes the store, the
next `init` (or the launch seed) repopulates the experiment's blob from the dataclass defaults
+ the launch argparse values — the bootstrap *is* the recovery path.

---

## 7. Failure modes, named, each with its fail-loud response

The design concern "name each and the fail-loud response":

- **redis down / unreachable.** The transport already fails loud on this (`_connect` calls
  `r.ping()` and raises; the loop "must not silently fall back to a slow path"). The registry
  reader does the same: a failed `GET` at the iteration-boundary refresh raises a loud
  `RegistryUnavailable` rather than silently reusing the last snapshot forever (silently
  reusing would mean an operator's lr drop *never* lands and they are never told). The bounded
  socket timeout (`CHOCO_REDIS_SOCKET_TIMEOUT=60`) turns a stall into a loud error, as it
  already does for the transport.

- **Key missing (experiment never seeded, or wiped by a restart).** The reader raises a loud
  "no registry blob for experiment_id `<id>` — was it seeded? (run `init`, or the launch seed
  failed)". Never coerce to defaults silently (a run reading defaults it did not ask for is the
  silent failure). The §6 bootstrap is the remediation.

- **Schema / key drift between code version and stored blob.** The blob carries
  `schema_version` and the derived `in_dim`/`n_actions`. On decode, the reader checks
  `schema_version` against the code's and the derived dims against the running env; a mismatch
  is a loud `RegistrySchemaDrift` naming both versions/dims. This catches the case where the
  code was upgraded (new field, renamed field, changed feature layout) but the stored blob is
  old — exactly the "code version vs stored blob" drift, failed loudly rather than decoded into
  a subtly-wrong config. (A forward-compatible decode that *ignored* unknown keys or *defaulted*
  missing ones is the tempting, wrong move — it is the `vhat=None`-vs-`vhat_zero` silent-wrong-
  number failure in another costume.)

- **Two writers racing.** The §5.4 `WATCH`/`MULTI`/`EXEC` optimistic guard detects it; a lost
  `EXEC` retries bounded, then fails loud. Neither writer's change is silently dropped.

- **A malformed value written out-of-band** (someone `redis-cli SET`s a bad blob directly).
  The §3.6 typed decode catches it on the next read (loud `RegistryDecodeError`), and the
  reader does not proceed on a config it could not validate. The write CLI's pre-write
  validation (§5.3) prevents the in-band version; the read-side validation is the backstop for
  the out-of-band version.

- **A RESTART/INSTANCE field changed on a running process.** §3.4: refuse loudly, name the
  field and both values, instruct `--resume` (RESTART) or new-experiment (INSTANCE). This is
  the failure mode the mutability facet exists to make loud rather than silently-ineffective.

- **The dual-bound correctness knobs.** `max_inner_states` and `restrict_faces` carry a
  contract from `dual-bound.md` §4.3/§6: the inner solve must abort loudly (never truncate) and
  must never *restrict* the action set (only over-approximate), or the upper bound silently
  becomes invalid. The registry stores these as ordinary HOT fields, but the *code that
  consumes them* keeps its existing loud abort (`RuntimeError` on cap-hit) and its
  over-approximation-only discipline; the registry does not weaken that contract — a value that
  would force truncation still hits the existing loud abort. Worth a comment in the schema so a
  future operator does not "tune" `max_inner_states` down thinking it is a free perf knob.

---

## 8. Library survey — what would make this ergonomic, and the honest verdict (requirement 7)

The four pillars to weigh each candidate against: (a) **hierarchical typed schema**; (b)
**redis as the source of truth**; (c) **live point-of-use reads** with the mutability facet;
(d) **namespacing**. The dependency posture matters: the working venv is a *shared scratch*
(`/home/bork/w/vdc/venvs/generic`) carrying jax/optax/numba; the project's instinct (the
AZ design doc, the residual ablation) is to avoid dependencies that buy little. The candidates:

- **Hydra / OmegaConf.** Buys: rich hierarchical config composition, CLI override syntax,
  structured-config dataclass backing (OmegaConf), multirun sweeps. Costs: Hydra owns the
  *application entry point* (it wraps `main` with `@hydra.main`, takes over CLI parsing and the
  working directory) — a heavy, invasive lock-in that fights the existing argparse entry points
  and the "wrap + layer, not replace" §6 reconciliation. Worse, **its source of truth is YAML
  files on disk, not redis** — there is no live point-of-use read; you would be fighting Hydra's
  whole model to put redis underneath it. The composition power is aimed at *launch-time* sweep
  configuration, which is not the problem (the problem is *mid-run* live change). **Decline** —
  it solves a different problem (launch-time config composition) and its disk-file source-of-
  truth is the wrong substrate for a live redis registry; the impedance is high and the lock-in
  invasive.

- **Pydantic / pydantic-settings.** Buys: best-in-class typed validation (the §3.6 decode is
  exactly Pydantic's wheelhouse — type coercion *with* loud errors, nested models, custom
  validators for the cross-field invariants), `BaseSettings` for env/launch layering. Costs: a
  non-trivial dependency (pydantic-core is a compiled Rust extension — a real install in the
  shared venv); its *coercion* defaults lean toward "make it fit" which must be configured to
  strict to honor ADR-0002 (`model_config = ConfigDict(strict=True)`); pydantic-settings'
  source layering is env/file-oriented, not redis (you'd write a thin redis source). It does not
  do redis or live reads for you, but it does the *validation* pillar very well. **Tepid
  consider, lean decline for the first cut**: it is the strongest fit for pillar (a)+(c)-
  validation, but it only solves the validation third of the problem (redis, live reads, and
  namespacing are still hand-built), and a stdlib dataclass + a ~50-line typed decoder covers
  the same validation need without the compiled dependency. Revisit if the validation surface
  grows (many custom validators, nested unions) past what a hand decoder stays readable for.

- **attrs + cattrs.** Buys: `attrs` for the schema (more featureful than `dataclasses`),
  `cattrs` for structured (un)structuring (dict↔nested-attrs with typed hooks — the §3.6 decode
  again). Costs: two dependencies for what `dataclasses` + a small decoder do; `attrs` over
  `dataclasses` buys little here (the schema is simple). **Decline** — `cattrs` is a clean
  (de)serialization layer, but `dataclasses` + `dacite` (below) or a hand decoder is lighter and
  the schema does not need `attrs`' extra power.

- **`dataclasses` + `dacite`.** Buys: stdlib dataclasses for the schema (zero dependency for the
  schema itself — the contract is pure stdlib), and `dacite` (a single small pure-Python
  dependency) for "dict → nested dataclass with type checking and strict mode" — exactly the
  §3.6 decode, including `strict=True` to reject unknown keys (the drift check) and type checks
  that raise rather than coerce. Costs: `dacite` is one small dep; it does *only* the decode (no
  redis, no live reads, no namespacing — but those are ours to design anyway). **Recommend as
  the (optional) decode helper** if a dependency is wanted: it is the minimal, honest fit —
  stdlib schema, one tiny pure-Python lib for the typed strict decode, everything else (redis,
  the live read path, the write CLI, the namespace) hand-built thin. The fallback if even
  `dacite` is unwanted is a ~50-line recursive typed decoder over the dataclass `fields()` and
  annotations — entirely viable at this schema size and zero new deps.

- **Google `ml_collections` (`ConfigDict`).** Buys: a JAX-ecosystem-native mutable config with
  attribute access and type-locking (`ConfigDict` locks types after first set — a nice ADR-0002-
  shaped guard). Costs: it is a *dict*, not a typed dataclass — the hierarchical *typed* schema
  (requirement 1's explicit ask) is weaker (types are enforced on mutation, not declared as a
  contract); no redis, no live reads. **Decline** — it is ergonomic in JAX code but does not give
  the declared typed-dataclass contract requirement 1 asks for, and adds a dependency for a
  weaker version of what dataclasses already provide.

- **gin-config.** Buys: dependency-injection-style config (decorate functions, bind params from
  a `.gin` file) — powerful for deep parameter injection without threading args. Costs: its model
  is *file-based launch-time binding*, not live mutation; it is invasive (decorators on every
  configurable), and its source of truth is `.gin` files, not redis. **Decline** — same category
  error as Hydra (launch-time file config, not live redis), with more code-intrusion.

- **Dynaconf.** Buys: layered settings from many sources (env, files, vault, **redis**) — and it
  *does* have a redis backend, which is unusually on-point. Costs: its redis backend is oriented
  to *reading settings at startup* with optional refresh, not to the tight mutability-facet
  semantics (HOT vs RESTART-refuse) this design needs; it is dynamically-typed (settings are
  loosely typed, validators are bolt-on) — the typed-dataclass contract is not its model; it is
  a medium-weight dependency. **Tepid consider, decline**: it is the one candidate whose redis
  backing is genuinely relevant, but it would own the read path with looser typing and without
  the mutability facet, so we'd fight it to get the §3.4 semantics — more friction than building
  the thin redis layer ourselves on top of a clean dataclass schema.

- **redis-py pub/sub** (for live propagation). Buys: the sub-second propagation §3.2 weighed.
  Costs: §3.2's missed-message silent-staleness failure, plus a listener thread per process.
  **Decline for the first cut, hold as a deferred enrichment** — polling at the iteration
  boundary is bounded-and-loud; pub/sub is tight-and-silent-on-miss. Add it later only as a
  refresh *hint* layered over the poll, never as the sole path.

**The verdict.** **Plain stdlib `dataclasses` for the schema + a thin hand-written redis layer
(read snapshot, write CLI, namespace) beats pulling in Hydra/gin/Dynaconf, because the problem
those frameworks solve is launch-time config composition from disk files, and this problem is
mid-run live change with redis as the source of truth — the substrates are different, and the
frameworks would each fight the redis-live model while owning the entry point.** The one place a
library earns its keep is the **typed strict decode** (§3.6), where `dacite` (one small pure-
Python dep) or, if even that is unwanted, a ~50-line recursive decoder, does the job cleanly;
Pydantic is the heavier alternative there and is worth revisiting only if the validation surface
outgrows a hand decoder. So: dataclasses + a thin redis layer, with `dacite` as the optional
decode helper — and an explicit "no Hydra, because it solves launch-time disk config, not live
redis mutation." This matches the project's own instinct (the residual-block and `unc`-feature
results that warn against adding machinery that buys little), applied to the config layer.

---

## 9. One-screen summary

| element | decision |
|---|---|
| **Schema** | stdlib nested `@dataclass`: `ExperimentConfig` over `EnvConfig`/`SearchConfig`/`ValueTargetConfig`/`FeatureConfig`/`ArchConfig`/`TrainConfig`/`ExItLoopConfig`/`EvalConfig`/`ParallelConfig`/`BoundsConfig`; each field carries `(default, Mut facet, doc, codec)` via `field(metadata=...)`. Defaults = the argparse defaults. The dataclass is the single contract. |
| **Store** | redis `127.0.0.1:6380` db 0 (the existing transport instance). One JSON blob per experiment at `choco:hp:<experiment_id>` (+ `:meta`). Single blob (not key-per-field) for atomic reads + whole-config validation. |
| **Eviction fix** | registry keys carry **no TTL** (bare `SET`); policy → **`volatile-lru`** (protects no-TTL registry keys, lets TTL'd transport blobs self-clean). Live: `CONFIG SET maxmemory-policy volatile-lru` (no root). Persist across restart: **root edit of `/etc/redis/redis-memcache.conf` + `systemctl restart redis-memcache`** — `CONFIG REWRITE` FAILS (conf is root-owned, redis runs as `bork`). Restart wipes keys (`save ""`/`appendonly no`) → re-seed from §6 bootstrap. |
| **Read path** | per-process typed snapshot refreshed **once per outer-iteration boundary** (≤1-iteration staleness; atomic within an iteration); **poll**, not pub/sub (bounded-and-loud > tight-and-silent-on-miss). Decode = JSON → strict typed dataclass, **fail loud** on any type/domain/key/invariant mismatch (never coerce to default). |
| **Mutability facet** | per field: **HOT** (read fresh: `alpha`,`beta`,`c_puct`,`c_visit`,`c_scale`,`max_depth`,`c_outcome`,`td_lambda`,`n_step`,`lam`,`epochs`,`batch`,`eval_n`,`y_mean`/`y_std`, dual-bound knobs) — apply + log; **RESTART** (baked: `lr`,`l2`,`betas`,`eps`,`m`,`n_sims`,`hidden`,`residual`,`use_jax_mlp`,`workers`,`seed`,feature layout) — **refuse loudly**, instruct `--resume`; **INSTANCE** (env constants) — refuse loudly, instruct new experiment. |
| **The lr case (honest)** | `lr` is **RESTART** in this code (`optax.adam` built once, jit closure captures it). A registry lr-drop is recorded+logged+namespaced and adopted by a one-command `--resume` (the handoff's own anneal workflow). Optional follow-on: `optax.inject_hyperparams` to make lr genuinely HOT — flagged, not required. |
| **Write path** | `chocofarm.hp.registry` CLI (`get`/`set`/`init`). `set` validates against the schema **before** writing; multi-field atomic via `WATCH`/`MULTI`/`EXEC` (the lr+l2 case); two-writers race → bounded retry then **fail loud**; every applied change logged loudly (field, old→new, writer, ts). |
| **Namespacing** | `experiment_id` (operator-meaningful, defaults to ckpt-dir basename), passed at launch (`--experiment-id`), held on the snapshot; distinct ids → disjoint key prefixes → concurrent experiments never clobber. Mirrors the transport's per-run token under a human name. |
| **argparse relationship** | **layer over, not replace**: dataclass defaults = argparse defaults; launch seeds the registry (`init`, idempotent); HOT fields read live from the registry during the run; the argparse namespace is captured once as the `launched_with` shadow the RESTART-refusal compares against. One default authority, one live authority, one construction-time shadow. |
| **Library** | **stdlib dataclasses + thin hand-built redis layer**; `dacite` (one small pure-Python dep) optional for the strict typed decode, else a ~50-line decoder. **Decline Hydra/gin/Dynaconf** (launch-time disk config, wrong substrate for live redis); Pydantic only if the validation surface outgrows a hand decoder. |

---

## 10. Honest caveats and open questions

- **The lr motivating case needs a `--resume`, not a live drop, as the code stands.** §3.5 is
  honest about this: the registry's day-one win is recorded+logged+namespaced+one-command, not
  "lr changes without touching the process." Making it fully live is a real but separable
  `JaxTrainer` refactor (`inject_hyperparams`). If the maintainer's bar is "drop lr with *zero*
  restart," that refactor is a prerequisite and should be scoped as such; if the bar is
  "consolidate the lr-drop workflow and stop hand-editing the launch command," the registry
  delivers without it.
- **The `volatile-lru` move stops LRU from reaping the leaked TTL-less `az:res:*` keys.** §2.2 —
  a real, small downside; the right fix is at the source (the abort path `DEL`s its own result
  keys), which is a `parallel.py` change out of this spec's scope but worth a work-status note.
- **A redis restart is destructive** (`save ""`/`appendonly no`). The persist-the-policy step
  and any host-level redis restart wipe all registry + transport state; the §6 bootstrap is the
  designed recovery, but an operator must know a restart is not transparent. Doing the one-time
  conf edit *before* a campaign avoids the mid-campaign wipe.
- **The mutability facet is a reading of the current code, and the code can move.** If
  `JaxTrainer` is refactored to inject lr, lr's facet flips HOT; if the feature layout is made
  dynamic, those facets change. The facet lives in the schema next to the field precisely so it
  moves *with* the code — but it must be kept honest (a facet that says HOT while the code bakes
  the value is the §3.4 lie the design exists to prevent, now inverted). A small test that
  asserts "every RESTART field, when changed mid-run, triggers the loud refusal" would keep the
  facets honest against code drift; recommended but out of scope here.
- **The dual-bound correctness knobs are stored but contract-bound.** §7 — the registry must not
  become the place someone "tunes" `max_inner_states` down into a silent truncation; the code's
  loud abort holds regardless, but the schema comment is the human guard.
- **Single instance.** Like every chocofarm artifact, this is shaped to the one env and the one
  redis on this host; the C2 conf-ownership facts are host-specific and should be re-verified if
  the instance moves.

---

## Appendix A — commission prompt (verbatim)

> [Reproduced verbatim per the project's consult-record discipline; `docs/consults/consult-001-prompt.md`
> is the format reference.]

You are working on **chocofarm** (`/home/bork/w/vdc/chocobo`, github KodBena/chocofarm) — an Operations Research exercise (FFXIII gil-farming modeled as an adaptive stochastic orienteering / belief-MDP problem), NOT a game tool. Success = solver/OR quality and honest, mechanistic analysis. The maintainer strongly prefers "this probably won't pay, because X" over optimistic listing, and provable/mechanistic claims over hand-waving. The codebase posture is **fail-loudly (ADR-0002)**: surface deviations through the strongest channel, never silently coerce/retry/swallow. Public Domain (Unlicense).

This is a **DESIGN SPEC** task — analysis + design, NOT implementation. You will **read code and write ONE design document, then commit it on a branch**. Do **not** implement the registry, do **not** modify any existing source file, do **not** run the training pipeline.

## The deliverable

Write `docs/design/hyperparameter-registry.md` — a design specification for a **centralized, live, redis-backed hyperparameter registry** for the AlphaZero / belief-MDP experiment stack. Match the house style of the existing design notes (`docs/design/alphazero-surrogate-design.md`, `docs/design/dual-bound.md`): precise, sectioned, honest about tradeoffs, mechanistic. The document must be **self-contained** and readable by the maintainer with no other context.

## What the maintainer asked for (the six requirements — treat each as binding)

1. **Schema is a Python dataclass — preferably a hierarchical construction of nested dataclasses** mirroring the hyperparameter taxonomy of the codebase (e.g. a top-level `ExperimentConfig` composed of `EnvConfig`, `SearchConfig`, `ValueTargetConfig`, `ArchConfig`, `TrainConfig`, `ExItLoopConfig`, `EvalConfig`, `ParallelConfig`, `BoundsConfig` — discover the actual cut from the code, don't impose this exact one). The dataclass schema is the typed contract.

2. **Actual values come from the redis in-memory database** (the scratch redis at `127.0.0.1:6380`, raw-bytes transport — see `chocofarm/az/parallel.py` for the existing connection + key conventions). **The redis configuration must be changed so registry values do NOT auto-evict after 1 hour.** I have already determined the eviction mechanism for you (verify it read-only, do not mutate redis): (a) `chocofarm/az/parallel.py` sets explicit 1h TTLs — result blobs `az:res:*` via `ex=3600` (`CHOCO_RESULT_TTL`) and weight blobs `az:w:{run}:{version}:m|b` via `expire(..., 3600)`; (b) the live server has `maxmemory 1073741824` (1 GB) with `maxmemory-policy allkeys-lru`, so even a TTL-less key is eviction-eligible under memory pressure. Your spec must address BOTH vectors: registry keys carry **no TTL**, AND the policy should move to **`volatile-lru`** (evict only keys that have a TTL) so transient transport blobs still self-clean while registry keys are protected — and specify how the change **persists across a redis restart** (`CONFIG SET` + `CONFIG REWRITE`, or an edit to the redis.conf / unit, whichever you find governs this instance; investigate read-only). Be explicit and correct here.

3. **Values are read at point of use** — so that a change takes effect on the fly without restarting a running experiment. Design the read path and its semantics honestly: per-access `GET` vs a cached snapshot with invalidation; polling vs redis pub/sub for change propagation; the staleness/atomicity window; and the **deserialization + validation** back into the typed dataclass (redis stores bytes/strings; a malformed or missing value must **fail loudly** per ADR-0002, not coerce to a default silently). Address which hyperparameters are **safe to hot-swap mid-run** (e.g. learning rate — the motivating case) versus which **cannot** change without a restart (e.g. network `hidden`, feature dim, action-slot count, anything baked into a compiled JAX function or the net shape) and how the schema should mark that (a per-field mutability facet).

4. **Purpose: change hyperparameters on the fly.** The motivating example is **manual LR drops**, which are standard practice in AZ-type training (KataGo and LeelaZero both use hand-scheduled or manual learning-rate step-downs). Design the **write path**: a small operator CLI / function to set a value (and ideally a related-set atomically via `MULTI`/`EXEC` or a versioned snapshot), and exactly how a running loop observes the new value at its next point-of-use. Log every applied change loudly.

5. **Namespace-isolate hyperparameters so multiple experiments run concurrently without clobbering each other.** The existing transport already namespaces by a `run` id (`az:w:{run}:...`); design the registry key schema analogously (e.g. `choco:hp:{experiment_id}:{path.to.field}` or a single serialized blob per experiment — argue the tradeoff). Define how an experiment id is chosen/passed and how a reader binds to the right namespace.

6. **This is explicitly NOT just the JaxTrainer hyperparameters.** (The maintainer flagged this to prevent malicious compliance.) You must **survey the WHOLE hyperparameter surface of the codebase** and enumerate it in the taxonomy. Read the code — do not guess. The surface is spread across env, search, value-target, features, architecture, training/optimizer, the ExIt loop, eval, parallelism, and the dual-bound solver.

Additionally: **(7)** survey and honestly evaluate **libraries that would make this more ergonomic** — candidates include Hydra/OmegaConf, Pydantic / pydantic-settings, attrs + cattrs, Google `ml_collections` (`ConfigDict`), gin-config, Dynaconf, `dataclasses` + `dacite`, and redis-py pub/sub for live propagation. For each plausible one: what it buys for *this* problem (hierarchical typed schema + redis backing + live point-of-use reads + namespacing), what it costs (dependency weight, lock-in, impedance with redis as the source of truth), and a clear recommend / decline with the reason. The maintainer wants suggestions, not a mandate — and an honest "plain dataclasses + a thin redis layer beats pulling in Hydra because X" is a perfectly good answer if that's what you conclude.

## Survey targets (read these before writing — read each fully, do not act on grep fragments)

Project orientation: `docs/handoff-2026-06-15.md`, `docs/STATUS.md`, anything under `docs/agents/`, and `docs/design/alphazero-surrogate-design.md` (the design-doc style + the value/policy/feature design that bears many hyperparameters).

Hyperparameter-bearing code (the taxonomy source — read to enumerate every tunable):
- `chocofarm/az/exit_loop.py` — the ExIt loop CLI (the largest argparse surface).
- `chocofarm/az/parallel.py` — redis transport, key/TTL conventions, workers/cores, run-id namespacing.
- `chocofarm/az/gumbel_search.py` — search hyperparameters (root actions m, sims, Gumbel/Sequential-Halving constants, cPUCT).
- `chocofarm/az/value_target.py` — TD(λ) blend / n-step target params.
- `chocofarm/az/features.py` — feature config / dims.
- `chocofarm/az/mlp.py` and `chocofarm/az/mlp_jax_train.py` — architecture (hidden, residual) and the JaxTrainer/optax hyperparameters (lr, l2, betas, epochs, batch).
- `chocofarm/az/train_value.py`, `chocofarm/az/dataset.py` — their argparse surfaces.
- `chocofarm/model/env.py` — environment constants (teleport overhead, costs, N, k, geometry, max_steps, λ-related).
- `chocofarm/bounds/eval_bound.py` and `chocofarm/bounds/info_relaxation.py` — dual-bound solver hyperparameters (λ scan, inner-solver state caps, V̂ generator choice).
- `chocofarm/eval/eval_az.py`, `chocofarm/eval/tb_runner.py` — eval/TB params.

Format reference for the appendix: `docs/consults/consult-001-prompt.md` (the consult-record discipline — see CONSTRAINTS).

You may run **read-only** `redis-cli -p 6380 ...` to verify the eviction facts (`CONFIG GET maxmemory*`, sample `TTL` on an `az:w:*` or `az:res:*` key, `KEYS 'az:*'` is acceptable on this small scratch db). **Do NOT** run `FLUSHALL`/`FLUSHDB`/`CONFIG SET`/`CONFIG REWRITE`/any write — investigation only.

## Design concerns the spec must explicitly address

- **Bootstrap / defaults.** A fresh experiment seeds the registry from the dataclass defaults; argparse (today's source of truth) should seed the registry at launch, then live overrides come from redis. Be explicit about the relationship between the existing argparse CLI and the registry (does the registry replace, wrap, or layer over argparse?) — the maintainer's word is "consolidate," so reconcile the two without just bolting a second config system alongside the first.
- **Type fidelity across redis.** redis values are bytes; the dataclass is typed. Specify the (de)serialization (JSON? a typed codec per field? msgpack?) and where validation lives, failing loudly on mismatch.
- **Atomic multi-field change** (e.g. dropping lr and raising l2 together) — transaction or snapshot-version semantics.
- **Mutability facet** per field (hot-swappable vs restart-required), and what a reader does if a restart-required field is changed mid-run (refuse loudly? warn? ignore-until-restart? — recommend one).
- **Observability** — every applied change logged at the same loud, filterable level the project values.
- **Failure modes** — redis down, key missing, schema/key drift between code version and stored blob, two writers racing. Name each and the fail-loud response.

## CONSTRAINTS (hard)

- **Do NOT touch the running training job** (a live `chocofarm.az.exit_loop` process, PID ~77159, is mid-run using this same redis). Do not kill processes, do not write to redis, do not change redis config, do not edit code.
- You are in an **isolated git worktree** already. Create a branch named **`docs/hyperparam-registry-spec`** and commit ONLY your new doc on it: `git checkout -b docs/hyperparam-registry-spec` then `git add docs/design/hyperparameter-registry.md` (**EXPLICIT PATH ONLY — never `git add -A`/`git add .`**) then commit.
- Commit message: a concise subject + body describing the spec; **end the message with exactly**: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **Do NOT push.** The orchestrator handles the push.
- **Append this entire commission prompt verbatim as "Appendix A — commission prompt" in the document**, per the project's consult-record discipline (`consult-001-prompt.md` is the format reference). Match the existing docs' header/preamble convention if they have one.
- Keep the document honest and mechanistic; scope every recommendation; where you're uncertain, say so.

## Your final message back to me

Your returned message **IS the record** — make it a complete, self-contained rendering of the spec's substance (the taxonomy with the full enumerated hyperparameter list, the schema shape, the redis key + config design including the exact eviction fix, the read/write/point-of-use semantics, the namespacing scheme, the mutability-facet treatment, and the library recommendation with reasons). Also report: the exact branch name and commit SHA you created, and the file path. Do not make it a pointer to the file — render the substance.
