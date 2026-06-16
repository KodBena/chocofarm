# Separating training from optimization — an ACL + structural refactor that makes hot hyperparameters fall out (2026-06-15)

> **Dated amendment (2026-06-16) — `search.m` / `search.n_sims` are now HOT.** §4's facet table (the
> `search.m/n_sims | RESTART | … a structural bracket, a shape not a coefficient` row) is **superseded**:
> the SH bracket is recomputed per `decide()` from `self.m` / `self.n_sims`, so they were reclassified
> `Mut.RESTART → Mut.HOT` in `chocofarm/hp/schema.py` and now drive the per-iteration `hot_search` flow.
> Per ADR-0005 Rule 8 the original row is left intact as the point-in-time record; read this note as its
> correction. See `docs/design/cpp-actor-daemon.md` (the build amendment).

A design-and-audit note, analysis only — no code was changed and no job was
run. The question, handed down with the just-landed hyperparameter registry
(`feat/hp-registry`): **training and optimization are conflated in this
codebase, and they should be separated.** Audit that, then design the refactor
so two things become *consequences of the architecture* rather than bolted-on
patches:

1. `optax.inject_hyperparams` should **fall out naturally** — injecting the
   optimizer's hyperparameters as runtime state, rather than closing over them
   once in `JaxTrainer.__init__`, should be the obvious shape, not a graft.
2. Every hyperparameter that **ought to be HOT** should be **genuinely HOT** —
   live-updatable at point of use on a running experiment, not
   "RESTART-adopted-via-`--resume`."

The tone is the project's: name what each change costs, name where a field is
*genuinely* RESTART/INSTANCE and must stay non-hot, and prefer "this dissolves
that specific representable illegal state" over "this is cleaner." The headline,
up front, because it governs everything below:

> The conflation is one object — `JaxTrainer` — wearing two hats. It is the
> **Trainer** (it owns the loss, the data marshalling, the y-standardization
> re-pin, the write-back to the net) *and* the **Optimizer** (it builds
> `optax.adam`, closes over `lr`/`l2`/`betas`/`eps`, owns the moment state).
> Because the two are fused, the optimizer's hyperparameters are captured at
> *Trainer* construction time — which is the only time the Trainer is built — so
> they cannot move without rebuilding the Trainer, which means rebuilding the
> net's relationship to the optimizer, which means a restart. Split the two and
> the optimizer becomes a small object whose hyperparameters are *its* runtime
> state; once that object reads its hyperparameters from a live snapshot each
> step, `inject_hyperparams` is not a feature you add — it is the only way to
> write the object at all, and `lr`/`l2`/`betas`/`eps` are HOT because there is
> no longer anywhere to bake them.

The companion note
`docs/design/simulation-parallelization-viability.md` (branch
`docs/sim-parallelization-viability`) already characterizes the simulation hot
path and the compiled-core seam; §6 here composes with it and does not
contradict it — the simulation↔rest ACL this refactor draws is exactly the
boundary that makes its "drop-in C++/numba simulation core" a free benefit.

---

## 0. What the code forces us to design around (load-bearing facts)

Read out of the actual code on `feat/hp-registry`, not from taste. Each fact is
cited to the line that makes it true.

**L1 — `JaxTrainer` is the conflation site, and it is a single class.**
`chocofarm/az/mlp_jax_train.py:187` (`class JaxTrainer`). Its `__init__`
(`:206`) does, in one breath:

- builds the optimizer: `self.opt = optax.adam(learning_rate=self.lr, b1=b1,
  b2=b2, eps=eps)` (`:215`), capturing `lr`/`betas`/`eps` as Python floats;
- captures `l2` (`:209`) and bakes it into the jit closures:
  `self._az_update = _make_az_update(self.opt, self.l2)` (`:219`),
  `self._value_update = _make_value_update(self.opt, self.l2)` (`:220`);
- reads the net's weights into a jax pytree and initializes Adam's moment state
  (`:216`–`:217`).

`_make_az_update(opt, l2)` (`:154`) returns a `@jax.jit` closure that closes
over **both** `opt` (a `GradientTransformation` tuple-of-functions, not a jax
type, so it *must* be a closure) **and** `l2` (a Python float folded into the
loss at trace time so the `0.5·l2·‖W‖²` term carries no traced branch, `:140`).
This is the C4 jit boundary the registry spec §4.5 named.

**L2 — the per-step signature is a lie the docstring admits.**
`train_step(...)` (`:247`) and `train_epochs(trainer, ..., lr, l2, ...)`
(`exit_loop.py:139`) both *carry* `lr`/`l2` in their signatures, but neither
uses them. The `JaxTrainer` docstring is explicit: "`lr` and `l2` are passed
per-step (the loop varies neither today, but the manual path took them per-step,
so the signature is preserved)" (`:198`–`:199`), and `train_epochs`'
docstring: "the trainer's configured lr/l2 are authoritative" (`exit_loop.py
:148`). So there is a **vestigial per-step channel** for exactly the two values
the registry marks RESTART — a signature that *looks* live and is not. This is
the shape ADR-0002 warns against: an interface that suggests a capability the
implementation silently drops.

**L3 — the registry already reads HOT fields live; the loop already
reconstructs the search every iteration.** `exit_loop.run` refreshes a
`ConfigSnapshot` at each iteration boundary (`exit_loop.py:317`), reads the HOT
fields off `snap.cfg` (`:318`–`:336`), and rebuilds `GumbelAZSearch`/the worker
search on the version bump (`gumbel_search.py` construction at `exit_loop.py
:353` serial / `parallel.py:217` worker). So the **machinery for live HOT reads
already exists and is exercised** — the search knobs `c_puct`/`c_visit`/
`c_scale`/`c_outcome`/`max_depth` are genuinely HOT today because the search
object is cheap and rebuilt at the boundary. The trainer is the one object on
the per-iteration path that is *not* rebuilt — it is built once (`exit_loop.py
:248`) so Adam's moments persist — and that is precisely why its hyperparameters
are stuck.

**L4 — y-standardization is already the model for a HOT optimizer scalar read
live at point of use.** `train_epochs` calls
`net.set_value_scale(Y.mean(), Y.std())` each iteration (`exit_loop.py:151`),
and the trainer reads `self.net.y_mean`/`self.net.y_std` **fresh inside every
`train_step`** (`mlp_jax_train.py:254`–`255`, `:269`–`270`). So the codebase
*already has* a training-time scalar that is live-updatable at point of use,
threaded through a single owner (the net) and read per step. `lr`/`l2` want
exactly this shape and do not have it only because they were captured at
construction instead.

**L5 — `optax.inject_hyperparams` is the standard optax mechanism for exactly
this.** It wraps a transform so named scalar hyperparameters live in the
optimizer *state* (a `hyperparams` dict) rather than being closed over, and are
read each `update` call from the state. The registry spec §3.5 names it as "the
one targeted code change that would upgrade the motivating case from
one-command restart to fully live," and recommends it as a *follow-on*. This
note is that follow-on, generalized.

**L6 — the genuinely-RESTART/INSTANCE set is real and must stay non-hot.** Not
everything marked RESTART is an artifact. The net's weight-matrix shapes
(`hidden`, `residual`, `in_dim`, `n_actions`, `dtype`) size arrays at
construction (`mlp.py:59`–`78`); changing them mid-run is incoherent — the
optimizer's moment pytree, the jit trace, and the weights would all mismatch.
The env constants (`teleport_overhead`, `present_k`, `entry`, the instance
geometry) *define the belief-MDP*; a net is fit against a specific env
(`env.py:24`–`70`), so a mid-run change silently invalidates it
(INSTANCE). `m`/`n_sims` size the Sequential-Halving bracket
(`gumbel_search.py:264`–`265`) — a structural shape, RESTART. The master `seed`
folds into per-worker/per-episode seeds and underpins the parallel≈serial
determinism contract (`parallel.py:221`–`229`); changing it mid-run breaks that
contract. **The refactor must distinguish these from the
artifact-RESTART set and leave them RESTART — and, per MISU, make them
unrepresentable as live rather than merely refused.**

---

## 1. Diagnosis — the conflation, the DRY violations, the representable illegal states

### 1.1 The conflation, concretely

`JaxTrainer` fuses two responsibilities that have different *lifetimes* and
different *mutation semantics*:

| responsibility | what it owns | natural lifetime | should hyperparams be live? |
|---|---|---|---|
| **Trainer** | the AZ loss; X/π/mask/y marshalling to jax; the y-standardization re-pin; epochs/batch iteration; write-back to the net | per-run (built once; the moment state is the only thing that *must* persist) | `alpha`/`beta` already HOT (traced call-args, `:163`) |
| **Optimizer** | the `optax` transform; `lr`/`l2`/`betas`/`eps`; Adam's moment state | conceptually per-step (the moments persist; the *coefficients* are read each step) | `lr`/`l2`/`betas`/`eps` *want* to be HOT but are baked |

The fusion is the bug. Because there is one object and one construction point,
the optimizer's coefficients are captured when the *Trainer* is built — and the
Trainer is deliberately built once (L3) so the moments persist. The constraint
that makes the moments persist (don't rebuild the Trainer) is, accidentally,
the constraint that bakes `lr`. The two have nothing to do with each other; they
are conflated only because the two responsibilities share a constructor.

The smell is concrete in three places:

- `_make_az_update(opt, l2)` closes over **the optimizer object** and **a
  hyperparameter value** in the same closure (`:154`). A trainer that built the
  loss and called a separately-owned optimizer would close over neither — the
  optimizer would supply its own `update`, and `l2` would be a loss input.
- the vestigial `lr`/`l2` per-step signature (L2) is the fossil of the old
  manual path where lr *was* per-step; the JAX migration kept the signature
  shape but moved the value to construction — the signature and the behavior
  now disagree.
- `sync_from_net()` (`:240`) resets the optimizer state by calling
  `self.opt.init(...)` — the Trainer reaching into the optimizer's state because
  it owns both. With a separate Optimizer this is `optimizer.reset(params)`, one
  responsibility calling another's method, not one object mutating its own two
  halves.

### 1.2 SSOT/DRY violations in the train/optimize path

The registry schema (`schema.py`) is the **single source of truth** for
hyperparameter values (registry spec §0, §6 — "Nothing about a hyperparameter
is recorded in two places"). The conflation breaks that for the optimizer
coefficients:

- **D1 — `lr` lives in the schema AND is captured in `optax.adam`.** The
  schema holds `train.lr` (`schema.py:151`); `JaxTrainer.__init__` copies it
  into `self.lr` (`mlp_jax_train.py:208`) and bakes it into `optax.adam`
  (`:215`). After construction there are **two authorities** for the live
  learning rate: the registry blob (which the operator can `set`) and the
  closed-over copy inside `self.opt`. They can disagree silently — the registry
  says `1e-4`, the optimizer still steps at `1e-3` — and the *only* thing that
  catches the disagreement is the runtime RESTART-refusal (`registry.py:578`),
  which fires *after* the divergence is already representable. The same holds
  for `l2` (`schema.py:152` AND the `_make_*_update(opt, l2)` closure),
  `beta1`/`beta2`/`eps` (`schema.py:153`–`155` AND `optax.adam(b1, b2, eps)`).
  This is the exact analogue of the `_redis_params()` duplication that
  `config.py` just consolidated (registry.py:92, parallel.py:59 both re-read
  `os.environ` until `config.redis_params()` became the one authority) — a value
  with a schema home that is *also* captured at a construction site, free to
  drift.

- **D2 — `l2` has a second, looser default.** `schema.py:152` defaults
  `l2=1e-4`; `JaxTrainer.__init__` defaults `l2=0.0` (`:206`). The exit_loop
  path always passes the registry value (`exit_loop.py:248`), so the two never
  collide in production — but the trainer's own default is a *second* default
  for the same field, the kind of "two sources of a default" the registry
  consolidation (§6) exists to eliminate. `train_value.py` (the Stage-1 gate)
  constructs `JaxTrainer` directly and could pick up the `0.0` default while the
  schema says `1e-4`.

- **D3 — the loss couples L2 by reproducing optax's job by hand.**
  `_l2_sumsq` + the `0.5·l2·‖W‖²` term in `_az_loss` (`:140`) reproduce a weight
  decay the optimizer *could* own (optax has `add_decayed_weights`). The
  codebase chose coupled-in-loss L2 deliberately (to match the numpy path's
  `g + l2·W` exactly, weights-only scope — `:113`), and that choice is sound.
  But it means `l2` is a *loss* hyperparameter that currently flows through the
  *optimizer*'s construction (`_make_az_update(opt, l2)`). The refactor should
  put `l2` where it is consumed (the loss/Trainer), not bundled with the
  optimizer build — separating it cleanly resolves which object owns it.

### 1.3 Representable illegal states in the train/optimize path

The registry guards a stale hyperparameter with a *runtime* refusal: the illegal
state is **representable** but loudly **refused** at the next iteration boundary
(`registry.py:569`–`593`). MISU asks for more — make the illegal state
unrepresentable *by construction*. The currently-representable illegal states:

- **I1 — stale captured `lr` (the headline).** After construction,
  `self.opt` holds an `lr` that can diverge from `train.lr` in the registry. The
  state "registry says `lr=X`, optimizer steps at `lr=Y≠X`" is representable; it
  is caught only by `assert_no_restart_drift` at the next boundary. Between the
  `set` and the boundary, and on any path that does not refresh, the divergence
  is live and silent.

- **I2 — same for `l2`, `betas`, `eps`.** Each is a closed-over copy that can
  diverge from its schema home (D1). All four are representable-stale, all four
  caught only by the runtime refusal.

- **I3 — a field marked HOT in the schema that the consuming code has actually
  baked.** This is the dual of I1: the schema can *say* HOT while the code
  captures the value, and nothing checks the schema's facet against where the
  code actually reads the value. `alpha`/`beta` happen to be honestly HOT
  (traced call-args, `:163`), but the facet is a hand-maintained *reading* of
  the code (schema.py:7 — "a READING of where the code consumes each value"),
  so a future edit that captures a HOT field at construction would make the
  schema lie with nothing to catch it. The refactor should make the facet
  *follow from* the type (HOT fields are the ones the optimizer-state /
  call-arg path carries), not a comment that can rot.

- **I4 — net-shape / precision mismatch against the optimizer moment pytree.**
  Adam's moment state is initialized against the params pytree at construction
  (`:217`). If the net's shape changed under the optimizer (a different
  `hidden`, a toggled `residual` adding `Wr*` keys), the moment pytree and the
  params pytree would mismatch — a structural illegal state. Today this is
  guarded by the `_assert_no_derived_drift` check on re-bind (`registry.py
  :504`) and the RESTART facet on `hidden`/`residual`. It is *representable*
  (you can hand the optimizer a mismatched net) and refused; MISU asks that the
  optimizer's moment state be *bound to* the params it was built from, so a
  mismatch is rejected loudly at step time by jax's own tree check, needing no
  separate guard (a runtime rejection, but a structural one — see §5.2 I4 for the
  honest framing).

---

## 2. The proposed boundaries (ACLs) and the type structure

Three ACLs, drawn where the responsibilities actually divide. Each is an
**anti-corruption layer** in the precise sense: it translates between two models
(the registry's typed snapshot ↔ the optimizer's runtime state; the simulation's
data-in/data-out ↔ the training machinery) so neither leaks into the other.

### 2.1 ACL-1 — the Optimizer abstraction (owns the optax transform; hyperparameters as injected runtime state)

A new small object, `chocofarm/az/optimizer.py`, that owns the optax transform
and **nothing else**. Its hyperparameters are not closed over; they live in the
optax state via `inject_hyperparams` and are supplied **per step** from the live
snapshot. This is the object that makes `inject_hyperparams` fall out: there is
nowhere in it to bake `lr`, because the only construction it does is the
`inject_hyperparams`-wrapped transform, and that wrapper's entire purpose is to
put the scalars in the state.

```python
# chocofarm/az/optimizer.py  (sketch — signatures are the contract)

from typing import NamedTuple
import optax

class AdamHParams(NamedTuple):
    """The optimizer's live scalar hyperparameters — the HOT subset of TrainConfig,
    in the optimizer's own vocabulary. Built from a ConfigSnapshot each step (ACL
    translation, §2.4). betas are a pair to match optax; eps/lr/weight-decay scalars."""
    lr: float
    b1: float
    b2: float
    eps: float
    # l2 is NOT here — it is a LOSS coefficient (D3), owned by the Trainer/loss, §2.2.

class Optimizer:
    """Owns an optax GradientTransformation whose lr/b1/b2/eps are INJECTED runtime
    state (optax.inject_hyperparams), not closed-over constants. The transform is
    built once; the moment state + the injected-hparam state live in `opt_state`.
    Each `step` REQUIRES the live AdamHParams as a call ARGUMENT — there is no
    captured lr/b1/b2/eps, and (the load-bearing detail, per the §5.2 audit) no
    way to CALL `step` without supplying them, so a forgotten hyperparameter is an
    arity error, not a silent stale step."""

    def __init__(self, params):
        # inject_hyperparams makes lr/b1/b2/eps part of opt_state.hyperparams. The
        # init values are immediately overwritten by the first `step`'s `hp`; they are
        # NEVER the authoritative value on any real step (see _with_hparams below).
        self._tx = optax.inject_hyperparams(optax.adam)(
            learning_rate=1.0, b1=0.9, b2=0.999, eps=1e-8)
        self.opt_state = self._tx.init(params)              # moment pytree typed to `params` (I4)

    @staticmethod
    def _with_hparams(opt_state, hp: AdamHParams):
        """Return a copy of `opt_state` whose injected hparams are exactly `hp`. The ONE
        place the live scalars enter the optax state — and it is reached only THROUGH a
        `step` call that requires `hp`, so the injected dict is never read without first
        being set from the call's argument. There is no code path that steps on the
        __init__ placeholders."""
        hps = dict(opt_state.hyperparams)
        hps.update(learning_rate=hp.lr, b1=hp.b1, b2=hp.b2, eps=hp.eps)
        return opt_state._replace(hyperparams=hps)

    def step(self, grads, params, hp: AdamHParams):
        """One optax update with the LIVE hyperparameters. `hp` is REQUIRED — the
        injected state is set from it inside this call (`_with_hparams`), so the only
        writer of the effective lr/b1/b2/eps is this call's argument list. Returns the
        new params. No lr is captured; no step reads a stale or placeholder hparam."""
        st = self._with_hparams(self.opt_state, hp)
        updates, self.opt_state = self._tx.update(grads, st, params)
        return optax.apply_updates(params, updates)

    def reset(self, params):
        """Re-init the moment state against a (possibly replaced) params pytree —
        the `sync_from_net` semantics, now a method ON the optimizer (one responsibility
        calling its own), typed to the params it is given (I4)."""
        self.opt_state = self._tx.init(params)
```

Three properties to note, because they are the whole point:

- **`inject_hyperparams` is not optional here — it is the only construction, and
  `hp` is a required `step` argument.** The object cannot be written to bake
  `lr`: `__init__` builds the injected transform and `step` *requires* the live
  `AdamHParams`, threading it through `_with_hparams` into the state inside the
  call. There is no `self.lr` (compare the current `:215`, the captured copy),
  **and** there is no `step` overload that reads the injected dict without first
  setting it from the call argument — so omitting the hyperparameters is an arity
  error at the call, not a silent step on the `__init__` placeholder. This is the
  §5.2 audit's correction made structural: `lr`/`betas`/`eps` are bound into the
  call signature exactly the way `l2` is (§2.2), so both reach the same
  "no-forgettable-write-site" bar. This is what "falls out naturally" means: the
  architecture has no slot for a stale `lr` *and* no callable shape that skips
  supplying one.
- **`l2` is deliberately *not* on `AdamHParams`.** It is a loss coefficient (D3),
  so it belongs to the Trainer/loss (§2.2). Putting it there resolves the
  ownership ambiguity the current `_make_az_update(opt, l2)` bundling creates.
- **the moment state is typed to the params it was built from.** `init(params)`
  / `reset(params)` mean the optimizer's moment pytree is always against a
  concrete params pytree — I4 becomes "you cannot call `step` with grads of a
  different shape than `init` saw" (jax will reject the tree mismatch loudly),
  rather than a separate runtime guard.

### 2.2 ACL-2 — the Trainer, slimmed to loss + data + write-back

`JaxTrainer` keeps its genuine responsibilities and *delegates* the step to the
Optimizer. It owns the loss (including `l2`, now correctly a loss input), the
jax marshalling, the y-standardization read (already live, L4), the epoch/batch
iteration, and the write-back. It does **not** build an optax transform.

```python
class Trainer:
    """Owns the AZ loss, the data marshalling, the y-standardization read, and the
    write-back to the net. Delegates the parameter update to an Optimizer (§2.1).
    Holds the params pytree + the Optimizer; reads l2/alpha/beta as LIVE per-step
    inputs (no captured optimizer coefficients)."""

    def __init__(self, net):
        self.net = net
        self.has_policy = net.n_actions is not None
        self.params = self._read_params()
        self.optimizer = Optimizer(self.params)        # moments typed to these params (I4)

    def train_step(self, batch, hp: AdamHParams, l2, alpha, beta):
        """One AZ step with LIVE optimizer hparams (hp), LIVE loss coefficients
        (l2/alpha/beta). y-standardization read off the net per step (L4). The jit'd
        value_and_grad takes l2/alpha/beta as traced ARGS (they were already so for
        alpha/beta; l2 joins them — it stops being a closed-over loss constant and
        becomes a traced input, so a live l2 change lands without a re-trace)."""
        grads, (ce, vmse) = self._grads(self.params, batch, l2, alpha, beta)
        self.params = self.optimizer.step(grads, self.params, hp)
        self._write_params()
        return ce, vmse

    def reset_optimizer(self):
        self.optimizer.reset(self.params)              # was sync_from_net's opt.init
```

The one numeric change inside the loss: `l2` moves from a **closed-over Python
float** (`_make_az_update(opt, l2)`, baked at trace time, `:154`) to a **traced
argument** of the jit'd grad function (alongside the already-traced
`alpha`/`beta`, `:163`). This is the same move `alpha`/`beta` already embody;
it makes `l2` HOT for free and removes the last reason the loss closure captured
a hyperparameter. The `l2==0` short-circuit the current code preserves by
closing over `l2` (`:109`) becomes a `jnp.where`/unconditional term — a
negligible cost (one extra `0.5·0·‖W‖²` = 0 add when `l2==0`), and the
equivalence test pins that it stays numerically identical at the production
`l2=1e-4`.

### 2.3 ACL-3 — the simulation↔rest seam (already clean; this refactor keeps it clean)

The simulation seam already exists and is well-placed: `env.py` "knows nothing
about HOW a decision is made — that is a `Policy`, passed in" (`env.py:9`–`11`),
and `Environment.simulate(policy, world, lam, rng, max_steps)` (`env.py:138`) is
**data-in/data-out**: world + λ + rng + cap → (R, T, exit). The search consumes
the env only through `apply`/`filter_*`/`marginals`/`exit_cost` — pure functions
of (state, action, world) returning (reward, state', dt). No training,
optimization, or registry type crosses this line. The parallel transport already
treats the simulation as a black box: it ships *weights* in (raw bytes,
`parallel.py:92`) and *transition records* out (raw bytes, `:267`), with the
HOT search knobs supplied through a stable `hot_search` dict on the call
(`parallel.py:201`, `:356`).

The refactor's obligation to ACL-3 is **negative**: do not let the
training/optimization split leak across it. The Optimizer and the slimmed
Trainer live entirely on the *learner* side of the existing
generate→train→eval boundary (`exit_loop.run`); the simulation side
(`generate_episode` → `decide_with_value` → search → `env.apply`) never sees an
`AdamHParams` or an `Optimizer`. §6 verifies this seam is language-agnostic.

### 2.4 The registry-snapshot read path (the ACL translation that feeds ACL-1/ACL-2)

The snapshot read path is the translation layer between the registry's
`ExperimentConfig` (typed, validated, HOT/RESTART-faceted) and the optimizer's
`AdamHParams` (the optimizer's own vocabulary). One small adapter, read at the
iteration boundary the loop already has (L3):

```python
def adam_hparams_from(cfg: ExperimentConfig) -> AdamHParams:
    """ACL: translate the registry's TrainConfig (the SSOT, §3) into the optimizer's
    runtime hparams. The ONE place TrainConfig.{lr,beta1,beta2,eps} crosses into the
    optimizer — read live each iteration, never captured. l2/alpha/beta go to the
    Trainer's loss separately (they are loss inputs, not optimizer state)."""
    return AdamHParams(lr=cfg.train.lr, b1=cfg.train.beta1,
                       b2=cfg.train.beta2, eps=cfg.train.eps)
```

`exit_loop.run`'s training step becomes (the change is local to the TRAIN block,
`:372`–`:380`):

```python
# inside the iteration, after snap.refresh(it):
hp = adam_hparams_from(snap.cfg)                       # LIVE optimizer hparams
l2 = snap.cfg.train.l2                                  # LIVE loss coefficient
ce, vmse, r2 = train_epochs(trainer, bX, bPI, bM, bY,
                            epochs, batch, hp, l2, alpha, beta, train_rng)
```

`train_epochs` drops its dead `lr`/`l2` positional channel (L2) and takes
`hp`/`l2` as the live values, passing `hp` to `trainer.train_step`. The
vestigial signature is gone — the interface now carries exactly what it uses.

### 2.5 The shape of the result

After the split, the ownership map is unambiguous and the facet follows from the
type:

| value | owner | how it's read | facet (now *structural*, not a comment) |
|---|---|---|---|
| `lr`, `betas`, `eps` | Optimizer state (injected) | required `AdamHParams` arg of `step`, set into `opt_state` inside the call | **HOT** — bound into the `step` call signature; no captured copy, no forgettable write |
| `l2` | Trainer loss | traced arg per step | **HOT** — a traced input, like `alpha`/`beta` |
| `alpha`, `beta` | Trainer loss | traced arg per step (unchanged) | **HOT** (already) |
| Adam moment state | Optimizer | persists in `opt_state` | n/a (state, not a hyperparameter) |
| `hidden`, `residual`, `in_dim`, `n_actions`, `dtype` | net construction | array shapes | **RESTART** — the params pytree the Optimizer's moments are bound to; a mismatch is rejected loudly by jax at step time (I4) |
| env constants, `seed`, `m`, `n_sims` | env / search / loop construction | array shapes / determinism contract | **RESTART/INSTANCE** — see §4 |

---

## 3. The pure structural refactors (notwithstanding any ACL)

These reduce conflation and make hot-params fall out **independently of the
registry** — they would be improvements even with no live registry at all. They
are the structural half the brief asks for "notwithstanding any ACL."

- **S1 — extract `Optimizer` from `JaxTrainer`.** The single split of §2.1/§2.2.
  Even without `inject_hyperparams`, separating "build the optax transform + own
  the moments" from "own the loss + marshal data + write back" is the structural
  decoupling; `inject_hyperparams` then has an obvious home (the Optimizer's
  `step`) instead of being grafted into a god-object.

- **S2 — make `l2` a traced loss argument, not a closure constant.** Move `l2`
  from `_make_az_update(opt, l2)` (`:154`) into the `value_and_grad` arg list
  next to `alpha`/`beta` (`:163`). Purely structural (the loss already takes
  `l2` as a parameter, `_az_loss(..., l2)` at `:123`); the only change is *who
  supplies it* (a traced arg, not a closure). This removes the last
  hyperparameter from the jit closure and resolves D3's ownership question.

- **S3 — delete the vestigial `lr`/`l2` per-step signature channel (L2).**
  `train_step` (`:247`) and `train_epochs` (`exit_loop.py:139`) stop accepting
  `lr`/`l2` they don't use; the signature carries exactly what is consumed. This
  is the "interface tells the truth" cleanup ADR-0002 wants — independent of the
  registry, the dead channel is a latent lie.

- **S4 — `reset_optimizer` replaces `sync_from_net`'s reach-in.** The current
  `sync_from_net` (`:240`) re-reads the net's weights AND resets the optimizer
  state by calling `self.opt.init(...)` — one object mutating its two halves.
  Post-split it is `self.params = self._read_params(); self.optimizer.reset(self.params)`
  — the Trainer re-reads, then asks the Optimizer to reset. Same behavior,
  unconfused ownership.

- **S5 — fold the facet *out of* a comment and *into* the structure.** Today the
  schema's HOT/RESTART tag is a hand-maintained reading (schema.py:7). After the
  split, the HOT optimizer fields are exactly the members of `AdamHParams` + the
  traced loss args; the RESTART fields are exactly the params-pytree shapes the
  Optimizer is typed to. The facet can be *checked* against the structure (a
  test that every `Mut.HOT` train field appears in the live read path and no
  `Mut.RESTART` one does — §7 step 6), closing I3. This is structural in that it
  makes the facet a consequence of where the value flows, not an annotation.

None of S1–S5 require the registry; they are the decoupling that makes the
registry's HOT marking *honest*.

---

## 4. The HOT-ness table — what ought to be HOT, what blocks it, what the refactor does

The full train/optimize surface. "Ought to be HOT" = "changing it on a running
experiment is a coherent operation a researcher would want (anneal lr, tune
regularization, retune Adam) and does not invalidate the net or break a
determinism/shape contract." The genuinely-RESTART/INSTANCE set is listed second
and stays non-hot **by construction**, per MISU.

### 4.1 The artifact-RESTART set — RESTART today only because of the conflation; HOT after the refactor

| field | ought HOT? | HOT today? | what blocks it today | how the refactor makes it HOT |
|---|---|---|---|---|
| `train.lr` | **yes** (anneal/finetune is the motivating case, handoff §Pending-3) | no | baked into `optax.adam(learning_rate=lr)` at `JaxTrainer.__init__` (`:215`); captured copy `self.lr` | injected via `inject_hyperparams`, set inside `step` from its REQUIRED `AdamHParams` arg (§2.1). No captured copy, no forgettable write → I1 reduced to one required call argument |
| `train.l2` | **yes** (regularization retune mid-run) | no | closed over by `_make_*_update(opt, l2)` at trace time (`:154`); folded into the loss as a Python float (`:140`) | a traced loss arg (S2), like `alpha`/`beta`; a live change lands without re-trace → l2 bound into the call signature, no captured copy |
| `train.beta1` | **yes** (rare, but coherent) | no | `optax.adam(b1=...)` baked (`:215`) | injected scalar in `AdamHParams.b1` (§2.1) |
| `train.beta2` | **yes** (rare, but coherent) | no | `optax.adam(b2=...)` baked (`:215`) | injected scalar in `AdamHParams.b2` |
| `train.eps` | **yes** (rare) | no | `optax.adam(eps=...)` baked (`:215`) | injected scalar in `AdamHParams.eps` |
| `train.alpha` | yes | **yes** | — (already a traced call-arg, `:163`) | unchanged; the model the others now follow |
| `train.beta` | yes | **yes** | — (already traced, `:163`) | unchanged |
| `train.epochs` | yes | **yes** | — (loop bound read at iter start, `exit_loop.py:326`) | unchanged |
| `train.batch` | yes | **yes** | — (loop bound, `:327`) | unchanged |

After the refactor **all of `lr`/`l2`/`beta1`/`beta2`/`eps` flip from RESTART to
HOT in `schema.py`** (the facet edits at `:151`–`:155`), because the consuming
code now reads them live. The motivating lr-anneal (handoff §Pending-3, "resume
at `--lr 1e-4`") becomes a `set train.lr 1e-4` on the running experiment, no
`--resume` — and the §3.5 "recorded+namespaced+logged drop" win is retained *and*
upgraded to "lands live."

### 4.2 The genuinely-RESTART/INSTANCE set — stays non-hot, made unrepresentable-as-live

| field | facet | why it MUST stay non-hot | how MISU makes "live" unrepresentable |
|---|---|---|---|
| `arch.hidden` | RESTART | sizes every weight matrix (`mlp.py:59`–`78`); the optimizer's moment pytree is built against these shapes (`:217`) | the Optimizer's `opt_state` is typed to the params pytree it `init`'d from (§2.1); a different `hidden` is a different pytree → jax rejects `step` loudly. The value is not on `AdamHParams`, so there is no live channel to put it through |
| `arch.residual` | RESTART | gates whether `Wr*` params exist (`mlp.py:76`); toggling mid-run adds/removes pytree keys | same — toggling changes the params pytree the moments are typed to; unrepresentable as a live optimizer input |
| `arch.in_dim`/`n_actions` | RESTART | derived from env; size `W1`/`Wp`; recorded for the drift check (`registry.py:510`–`513`) | derived, not a free knob; the `_assert_no_derived_drift` guard stays; never on a live hparam path |
| `arch.dtype` | RESTART | read once at import (`dtypes.py:32`); flips the f32 cache + train precision | an import-time constant; structurally cannot be a runtime field |
| `arch.init_seed` | RESTART | consumed only at He-init construction (`mlp.py:54`) | construction-only; no point-of-use exists after init |
| env constants (`teleport_overhead`, `present_k`, `entry`, instance geometry, `value_vector`) | INSTANCE | *define the belief-MDP*; a net is fit against a specific env (`env.py:24`–`70`); changing them invalidates the net | INSTANCE refusal stays; the simulation ACL (§2.3) means these live entirely behind `Environment`, never on a train/optimize hparam path |
| `search.m`/`n_sims` | RESTART | size the SH bracket / phase loop (`gumbel_search.py:264`–`265`) | search-side, not optimizer-side; the search is rebuilt at the boundary but `m`/`n_sims` size a structural bracket, kept RESTART (a shape, not a coefficient) |
| `loop.seed` | RESTART | folds into per-worker/per-episode seeds; underpins parallel≈serial determinism (`parallel.py:221`) | a launch-time determinism contract; changing it mid-run is incoherent, kept RESTART |

The clean reading: **the artifact-RESTART set (4.1) becomes HOT because the
refactor gives each a live point-of-use; the genuine set (4.2) stays non-hot
because each is a *shape* or an *instance-defining constant*, and MISU makes
"live" for them unrepresentable — there is simply no `AdamHParams` slot or
traced-arg channel to route them through.** `max_steps` (already HOT,
`schema.py:91`) and the search knobs (already HOT) are unaffected — they were
never part of the conflation.

---

## 5. SSOT/DRY and MISU, each addressed explicitly

### 5.1 SSOT/DRY (the registry schema is the one authority, read live)

The refactor leaves **exactly one authority per value, read live**:

- **D1 dissolved.** `lr`/`betas`/`eps` have no captured copy after §2.1 — they
  are set inside each `step` from its required `AdamHParams` argument, which the
  loop builds from the live snapshot via `adam_hparams_from(snap.cfg)`. The
  registry blob is the sole authority; the optimizer holds a per-step *read* of
  it (passed as a call argument), not a construction-time *copy*. The one nuance
  the §5.2 audit surfaced: `inject_hyperparams` seeds the state with placeholder
  scalars at `__init__` (`learning_rate=1.0`), which are a *representable* second
  value — but they are never *authoritative*, because every `step` overwrites the
  injected hparams from `hp` before `update` and there is no `step` path that
  reads them un-set (it is set inside the same call that requires `hp`). So the
  sole *authoritative* source remains the snapshot, the way `config.redis_params()`
  left exactly one authority for the redis facts. (The §7 step-6 test asserts this
  directly — see below.)
- **D2 dissolved.** `JaxTrainer`'s own `l2=0.0` default goes away — the slimmed
  Trainer takes `l2` per step from the snapshot (`schema.py:152` default), so
  there is one default. `train_value.py` (the Stage-1 gate) routes through the
  same snapshot/`AdamHParams` path, so it picks up the schema default, not a
  second trainer-local one.
- **D3 dissolved.** `l2` is now unambiguously a *loss* coefficient owned by the
  Trainer (a traced arg), not bundled into the optimizer build. The
  coupled-in-loss L2 choice (sound, matches the numpy path) stays; only its
  ownership is now clean.

The single-authority test (§7 step 6): grep the train/optimize path for any
assignment of an optimizer/loss coefficient that is *not* sourced from a
`ConfigSnapshot` read that iteration. Post-refactor there should be exactly one
read site per value (`adam_hparams_from` for `lr`/`betas`/`eps`; the snapshot
read for `l2`/`alpha`/`beta`), and zero captured copies.

### 5.2 MISU (illegal states removed by construction, or reduced to a single guarded write-site / loud structural rejection)

Each currently-representable illegal state from §1.3, and how the refactor
removes or sharply narrows its *representability*. The honest taxonomy, after the
out-of-frame audit of this section: I1/I2 are removed *as captured copies* and
reduced to a single required call argument (no forgettable write); I4 is reduced
to a loud jax-level rejection needing no hand-written guard; I3 is made
test-catchable, not unrepresentable. Where a verb is weaker than
"unrepresentable" the text says so plainly — overclaiming "unrepresentable" for a
single guarded write-site would be the exact silent-failure this section is about.

- **I1 (stale captured `lr`) → eliminated as a captured copy; reduced to a single
  required call argument.** The honest statement, after the §5.2 out-of-frame
  audit: the *captured-copy* staleness is genuinely gone — there is no `self.lr`,
  so the old failure ("a construction-time `self.lr` that silently drifts from
  `train.lr` between boundaries") cannot be constructed. What remains is not a
  drifting copy but a **single write-site**: the effective `lr` is set inside the
  `step` call from its *required* `hp` argument (`_with_hparams`, §2.1). Because
  `hp` is required, omitting it is an arity error, not a silent step on the
  `__init__` placeholder — so this is the same "bound into the call signature, no
  forgettable write" guarantee `l2` has as a traced arg (§2.2), *not* a mutable
  dict the caller is trusted to overwrite. That is strictly stronger than the
  runtime RESTART-refusal (which fired only at the *next* boundary): there is now
  no representable optimizer that steps at a stale `lr`, because the only value it
  can step at is the one passed to the call. The facet flips to HOT, so the
  RESTART-refusal no longer applies to `lr` at all.

  *Precise scope of the claim:* "no stale `lr`" holds **at every call site that
  goes through `Optimizer.step`**. The §7 plan's Step 4 migrates the one other
  real construction-and-step path — `train_value.py`'s Stage-1 gate, which today
  builds `JaxTrainer` directly and steps via `train_step_value` outside the
  snapshot — onto the same `Optimizer`/`AdamHParams` contract, so the guarantee
  quantifies over *both* call sites, not just `exit_loop`. Until Step 4 lands,
  the property holds for the loop path and the gate is the named residual (§8).
- **I2 (stale `l2`/`betas`/`eps`) → same.** `betas`/`eps` ride the same required
  `hp` argument as `lr`; `l2` is a traced loss arg supplied from the snapshot
  each step. None is a captured copy and none is a forgettable dict write — each
  is a required argument of the call that consumes it.
- **I3 (schema says HOT but code baked it) → caught structurally.** Post-refactor
  the HOT train fields are *exactly* the `AdamHParams` members + the traced loss
  args, by construction. A test (§7 step 6) asserts the schema's `Mut.HOT` train
  set equals the live-read set and the `Mut.RESTART` train set equals the
  params-pytree-shape set. A future edit that bakes a HOT field fails the test —
  the facet can no longer silently rot.
- **I4 (net-shape/precision mismatch vs moment pytree) → rejected loudly by jax,
  no separate guard.** Honest framing (per the §5.2 audit): this is *not* a
  compile-time/type-system impossibility in Python — it is a **runtime** rejection,
  in the same fail-loud family as the RESTART-refusal, but at a different and
  cheaper layer. The Optimizer's `opt_state` is `init`'d/`reset` against a concrete
  params pytree (§2.1), so calling `step` with grads of a different tree shape is a
  jax tree mismatch that jax itself rejects at step time — no hand-written guard,
  no chance of a silently-wrong update. The improvement over the status quo is real
  (the moment state is structurally *bound* to its params, so the mismatch *cannot
  proceed silently*), but the accurate verb is "rejected loudly," not
  "unrepresentable." (The `_assert_no_derived_drift` re-bind guard at `registry.py
  :504` stays as the *operator-facing* explanation for the RESTART/INSTANCE class;
  the jax tree-check is the structural layer beneath it that needs no separate
  code.)

The honest residue: MISU here removes representability for the *optimizer-state*
illegal states (I1/I2/I4) and makes I3 test-catchable. It does **not** make the
genuine RESTART/INSTANCE *changes* unrepresentable in the registry blob — an
operator can still `set arch.hidden 512` on a running experiment; that write is
recorded and the running process refuses it loudly (`registry.py:578`). That
refusal is correct and stays: the blob is a durable record across restarts (the
operator *does* want the new value adopted on the next `--resume`/new-experiment),
so the value must be *representable as a recorded intent* even while it is
*unrepresentable as a live change*. The MISU win is that the live-change illegal
states in the train/optimize *runtime* are gone; the recorded-intent path keeps
its loud refusal because that is the right semantics for a durable record.

---

## 6. C++-readiness verification — the simulation seam is language-agnostic

The brief asks this be *verified as a property the proposed boundaries have*, not
treated as added scope. The companion note
`docs/design/simulation-parallelization-viability.md` already characterizes the
simulation as the hot path behind a clean data-in/data-out interface; this
section verifies the training/optimization refactor *keeps* that seam clean and
that a C++/numba simulation core is a drop-in behind it.

**The seam is `Environment` + `Policy`, and nothing in this refactor crosses it.**
The simulation surface a compiled core would reimplement is exactly:
`apply(loc, bw, collected, action, world) → (r, loc', bw', collected', dt)`
(`env.py:125`), `filter_treasure`/`filter_detector`/`marginals` (the bitwise
world-set reductions, `:99`–`135`), `exit_cost`/`d` (the static distance memo,
`:73`–`85`), and `simulate(policy, world, lam, rng, max_steps)` (`:138`). Every
one is a pure function of (state, action, world) returning numbers — **no
training type, no optimizer type, no registry type appears in any of these
signatures, and none is added by this refactor.** The `Optimizer` and the slimmed
`Trainer` live entirely on the learner side; the simulation never imports them.

**The hyperparameters the simulation needs already arrive through a stable,
language-agnostic interface.** The search's HOT knobs cross the
process/transport boundary as a plain `hot_search` dict of scalars
(`parallel.py:201`, `:356`, `:368`) and `max_steps` as a scalar — a flat
key→number map, the canonical language-agnostic shape (it is already serialized
across the spawn boundary and would serialize identically across an FFI
boundary). `lam` is a scalar arg (`env.py:138`). A C++ simulation core would
receive exactly these scalars and the world-set bytes (already raw `int64`
`tobytes()`, `parallel.py:267`), and return transition records as raw float32
bytes (already the wire format, `:267`). **The transport is already
pickle-free raw bytes** precisely so it does not depend on Python object
semantics — which is the same property that makes it FFI-friendly.

**Why the training/optimization split *helps* the C++ readiness.** Before the
split, an implementer reading `JaxTrainer` sees the optimizer, the loss, the
data marshalling, AND (via `train_epochs`' dead `lr`/`l2` channel) a signature
that *looks* like it threads optimizer hyperparameters through the per-step path
that the simulation feeds. That muddies the boundary: it is not obvious from the
code that the simulation→records→train pipeline carries no optimizer state.
After the split, the learner side is visibly three small objects (Optimizer,
Trainer, snapshot-adapter) with optimizer hyperparameters confined to
`AdamHParams` and the `step` call — so the data crossing the simulation seam is
*manifestly* just (weights in, records out, scalar knobs in). The refactor makes
the seam's language-agnosticism legible, not just true. A C++ core that produces
the same transition-record bytes from the same weights+scalars is a drop-in;
nothing in the Optimizer/Trainer would change a line.

**Composition with the companion note (no contradiction).** That note's ranked
verdict is: #1 widen the exact cross-episode fan-out (Axis A, bit-exact, already
built), #2 a compiled/columnar tree core *conditionally for latency*, #3 GPU
leaf-batching is gold-plating. This refactor touches **none** of those axes — it
is entirely on the learner side, which the companion note explicitly holds
*outside* the simulation hot path ("TRAIN stays central in the parent,"
`parallel.py:5`–`6`). So the two compose freely: a future #2 compiled tree core
drops in behind the `Environment`/`Policy` seam this refactor keeps clean, and
the cross-episode fan-out (#1) ships the same raw-byte records to the same
slimmed Trainer. The refactor makes the infrastructure *more flexible* (the
boundary is legible and the learner side is decomposed), at no cost to the
companion note's avenues.

---

## 7. Sequenced implementation plan

Each step is independently reviewable and independently shippable; each names its
verification. The ordering is contracts/pure-logic first, effectful glue last
(the project's authoring posture). The equivalence test
(`tests/test_jax_equivalence.py`) and the AZ-loop tests (`tests/test_az_loop.py`)
are the standing fidelity immune system — every step that touches numerics
re-runs them.

**Step 0 — characterize the current numeric baseline (no code change).** Run the
existing suite (`test_jax_equivalence.py`, `test_az_loop.py`,
`test_parallel_deadlock.py`) and record a short fixed-λ₀ smoke (a few iterations
of `exit_loop` at the handoff's params) as the before-baseline. *Verify:* tests
green; capture the smoke's per-iter CE/vMSE/R² as the reference the refactor must
reproduce. (This is the `az-perf.md` `max|ΔG|` discipline applied to the loss
trajectory.)

**Step 1 — extract `Optimizer` (S1), still closing over hparams (no behavior
change).** Create `chocofarm/az/optimizer.py` with `Optimizer` owning the optax
transform + moments, but *initially* built with plain `optax.adam` and fixed
hparams passed to `__init__` — i.e. a pure code-move, no `inject_hyperparams`
yet. `JaxTrainer` delegates `step`/`reset` to it. *Verify:* `test_jax_equivalence`
+ `test_az_loop` green; the Step-0 smoke reproduces the reference CE/vMSE/R²
bit-for-bit (it is a pure refactor — same `optax.adam`, same order). This is the
decoupling with zero numeric risk.

**Step 2 — move `l2` to a traced loss arg (S2), delete the dead `lr`/`l2`
channel (S3).** `l2` joins `alpha`/`beta` as a `value_and_grad` traced arg; the
`l2==0` short-circuit becomes an unconditional term. `train_step`/`train_epochs`
drop the unused `lr`/`l2` positional channel. *Verify:* `test_jax_equivalence`
green; the smoke reproduces the reference at `l2=1e-4` (the term is
algebraically identical; the test pins float32 roundoff). Add a regression assert
that a *changed* `l2` between steps produces a *different* gradient (proving it is
now live, not baked).

**Step 3 — introduce `inject_hyperparams` in `Optimizer` (the headline).** Swap
the Step-1 plain `optax.adam` for `optax.inject_hyperparams(optax.adam)(...)`;
`step` writes the live `AdamHParams` into `opt_state.hyperparams` before
`update`. With the *same fixed* hparams each step, this is numerically identical
to Step 1 (injected-state Adam with constant hparams == closed-over Adam).
*Verify:* `test_jax_equivalence` + `test_az_loop` green; the smoke reproduces the
reference. Add a unit test: stepping with `lr=0` leaves params unchanged; stepping
with a doubled `lr` doubles the (small-step) update direction magnitude — proving
the injected `lr` is read live.

**Step 4 — wire the snapshot ACL (`adam_hparams_from`) into `exit_loop.run`.**
Add `adam_hparams_from(cfg)`; the TRAIN block reads `hp = adam_hparams_from(
snap.cfg)` and `l2 = snap.cfg.train.l2` from the per-iteration snapshot and
threads them through `train_epochs → trainer.train_step`. The loop now supplies
live optimizer hparams each iteration. *Verify:* `test_az_loop` green; an
integration test that `set train.lr 1e-4` on a running experiment's registry blob
changes the optimizer's effective step the *next iteration* (no `--resume`),
observed via the `_log_hot_changes` line + a measurable change in the update.

**Step 5 — flip the schema facets (HOT) and the argparse-map nuances.** In
`schema.py`, change `lr`/`l2`/`beta1`/`beta2`/`eps` from `Mut.RESTART` to
`Mut.HOT` with updated rationale strings (pointing at this note's §4.1). Update
`TrainConfig`'s docstring (`:144`–`:149`) — it currently says these are "BAKED
into optax.adam at construction." Update `registry.py`'s `_cli_set` RESTART-note
(`:646`) — `train.lr` no longer prints the "restart with --resume" note. *Verify:*
the `assert_no_restart_drift` no longer fires on a `train.lr` change (it is HOT);
the §7-step-6 facet-consistency test passes.

**Step 6 — add the structural facet-consistency test (S5, closes I3) AND the
write-site test (closes the §5.2-audit residual on the injected dict).** Two
asserts:
(a) *facet consistency* — every `Mut.HOT` field of `TrainConfig` is read from the
live snapshot on the train path (and supplied to `AdamHParams` or the loss), and
every `Mut.RESTART` arch field corresponds to a params-pytree shape the Optimizer
is bound to; the test *fails* if a future edit bakes a HOT field or marks a live
field RESTART.
(b) *no-step-reads-the-placeholder* — the out-of-frame audit flagged that
`inject_hyperparams` seeds `opt_state.hyperparams` with placeholders at
`__init__`, so a `step` implementation that forgot to set them from `hp` would
silently step at `lr=1.0`. Pin it directly: construct an `Optimizer`, take a
`step` with an `AdamHParams(lr=0.0, ...)`, assert the params are unchanged (lr=0 ⇒
no update) — proving the *step's* lr came from `hp`, not the `1.0` placeholder;
and a second `step` with a doubled `lr` produces a proportionally larger update.
This asserts the single write-site (`_with_hparams` inside `step`) actually
fires, which set-membership alone (test a) does not catch. *Verify:* both pass
now; (b) *fails* if a refactor reintroduces a `step` path that reads the injected
dict without setting it from the required `hp`. These are the MISU guards that
the facet follows the structure AND the one write-site is exercised.

**Step 7 — documentation graph + registry-spec follow-on closure.** Update the
registry spec's §3.5 "optional follow-on" to point at this delivered note;
update `TrainConfig` docstrings; record the HOT-flip in the handoff's
implementation-context if a still-open work item references the lr-anneal as
"resume-adopted." *Verify:* the doc cross-references resolve; the handoff's
§Pending-3 lr-anneal note is updated to "now a live `set`, no `--resume`."

**Verification posture across all steps.** Steps 1–3 are *bit-reproducible*
against the Step-0 reference (pure moves + constant-hparam injection); Step 2's
`l2`-as-traced-arg is algebraically identical at the production value and pinned
to float32 roundoff by the equivalence test. Steps 4–6 are the *behavioral*
changes (live hparams), each gated by an integration test that a registry `set`
lands live. No step requires a long training run to verify — the smoke + unit
tests settle each.

---

## 8. Honest caveats, costs, tradeoffs

- **`inject_hyperparams` has a small per-step overhead and a state-shape change.**
  It wraps the transform so the hparams ride in `opt_state` as traced arrays; the
  `step` writes scalars into that state each call. The overhead is negligible
  against the training step's matmuls (and training is off the per-leaf hot
  path entirely, `mlp_jax_train.py:36`–`39`), but it is not zero, and the
  `opt_state` pytree gains a `hyperparams` dict — checkpoints of the optimizer
  state (if ever serialized) change shape. Today the optimizer state is *not*
  checkpointed (only the net weights are, `exit_loop.py:404`; `--resume`
  re-inits a fresh optimizer, `:210`), so this is moot now — but if optimizer-state
  checkpointing is ever added, the injected shape is the one to serialize.

- **Live `lr` mid-run interacts with Adam's moments.** Making `lr` HOT means an
  operator *can* anneal it live — but Adam's moment estimates were accumulated
  under the old `lr`. This is standard (LR schedules do exactly this) and is the
  *intended* behavior (a live anneal, not a reset), but it is a different
  trajectory than `--resume` (which re-inits the moments, `:210`). The note's
  claim is that live-`lr` is *coherent and wanted*, not that it is identical to
  the resume path. An operator who wants the moment reset still resumes; one who
  wants a smooth live anneal now has it. This should be stated in the HOT
  rationale so the semantics are not surprising.

- **`l2` as a traced arg loses the compile-time `l2==0` short-circuit.** The
  current code closes over `l2` so a `l2==0` run carries no traced L2 branch
  (`:109`). As a traced arg, the `0.5·l2·‖W‖²` term is always computed (it is
  `0` when `l2==0`, so numerically harmless). The cost is one extra
  reduction+multiply per step in the `l2==0` case — trivial against the step's
  matmuls, and only the Stage-1 value gate ever ran with `l2==0` in practice.
  Worth naming because it is a real (tiny) cost traded for `l2` being live.

- **The split is a non-trivial refactor of a numerically load-bearing file.**
  `mlp_jax_train.py` is pinned by the equivalence test for a reason — the weights
  numpy inference reads must match the jit'd forward training optimized
  (`:20`–`:24`). Steps 1–3 are designed to be bit-reproducible precisely so the
  refactor does not silently perturb that contract; the discipline is the
  `az-perf.md` `max|ΔG|=0.0` standard. But it is real surgery on a file the search
  fidelity depends on, and the step-by-step bit-reproduction is load-bearing, not
  ceremony.

- **The genuine RESTART/INSTANCE refusal stays — this refactor does not make
  *those* live.** It would be over-claiming to say "all hyperparameters become
  HOT." Net shape, precision, env constants, `m`/`n_sims`, and `seed` stay
  non-hot *by design* (§4.2), and the registry's loud refusal on a mid-run change
  to them is *correct* and retained. The scope is exactly the artifact-RESTART
  set (the optimizer coefficients), which is the set the conflation created.

- **`train_value.py` (the Stage-1 gate) must be migrated too.** It constructs
  `JaxTrainer` directly (registry spec §4.4 names it). The split changes that
  construction; the gate must route through the new `Trainer`/`Optimizer` and the
  snapshot adapter (or pass an explicit `AdamHParams` for the gate's fixed
  config). Small, but it is a second call site that the plan's Step 4 must cover,
  not just `exit_loop`.

- **This is design only.** No code was run; the bit-reproducibility claims (Steps
  1–3) are *arguments from the operations being unchanged*, not certified diffs.
  The plan's per-step `max|ΔG|`/equivalence verification is what *certifies* them;
  this note identifies where exactness is preservable and structures the steps so
  it is, but does not pre-certify any implementation.

---

## Appendix A — commission prompt (verbatim)

> Recorded verbatim per the consult-record discipline
> (`docs/consults/consult-001-prompt.md` is the format reference).

---

You are a **refactor auditor** on **chocofarm** (`/home/bork/w/vdc/chocobo`, github KodBena/chocofarm) — an Operations Research exercise (a belief-MDP / adaptive stochastic orienteering problem). Codebase posture: **fail-loudly (ADR-0002)**. Public Domain (Unlicense). The maintainer prefers honest, mechanistic "this costs X, buys Y" over optimism.

This is a **design + audit** task. You produce ONE **design note** — analysis AND a concrete, sequenced, implementation-ready plan — then commit it. **Do NOT implement anything**, do not modify source, do not run code or any job.

## Where to work
A worktree has been prepared for you at **`/home/bork/w/vdc/chocobo-trainopt`**, already checked out on branch **`docs/training-optimization-refactor`** (based on `feat/hp-registry`, so the just-landed hyperparameter registry + `chocofarm/config.py` are present as files). Do ALL work there (`cd` in first). Do **not** create another worktree; do **not** touch the other worktrees; do **not** modify `runs/`/`tb/` or redis.

## The core question
It has been suggested that **training and optimization are conflated** in this codebase, and they should be separated. Audit this and design the refactor. Specifically, suggest **(a) the necessary ACLs / boundaries** and **(b) the pure structural refactors (notwithstanding any ACL)** that together would:
1. make **`optax.inject_hyperparams` fall out completely naturally** — i.e. so that injecting the optimizer's hyperparameters as runtime state (rather than closing over them once in `JaxTrainer.__init__`) is the *obvious consequence* of the architecture, not a bolted-on patch; and
2. make **all the reasonable hyperparameters that OUGHT to be HOT, actually HOT** — genuinely live-updatable at point of use on a running experiment, not "RESTART-adopted-via-resume."

The motivating context: the hyperparameter registry (just landed on `feat/hp-registry`) classifies each field HOT / RESTART / INSTANCE. Several fields (notably `lr`/`l2`/`betas`/`eps`, and the search knobs) are marked **RESTART only because the consuming code captures them once at construction** (optax + the jit update closures in `mlp_jax_train.py`; the search object). That construction-time capture is the conflation. Distinguish honestly between fields that are *genuinely* RESTART/INSTANCE (net shape, feature dim, action-slot count, env constants — changing them mid-run is incoherent and must stay non-hot) and fields that are RESTART *only as an artifact* of how training and optimization are built together (these are the ones the refactor should dissolve into HOT).

## The axes the maintainer named — treat each as a first-class evaluation lens

- **SSOT / DRY.** The registry schema is the single source of truth for hyperparameters. Audit where a hyperparameter is currently *also* captured/duplicated elsewhere (e.g. a value lives in the schema AND is closed over in an optax transform built once, so the closure's copy can silently drift from the SSOT). The refactor must leave exactly one authority per value, read live. (The duplicated `_redis_params()` that was just consolidated into `config.py` is the *kind* of DRY defect to hunt — find the analogues in the train/optimize path.)
- **MISU — make illegal states unrepresentable.** The registry today guards a stale hyperparameter with a *runtime* RESTART-refusal (the illegal state is representable but loudly refused). Push further: design the types/boundaries so the illegal states are **unrepresentable by construction** — e.g. an optimizer that reads its hyperparameters from the live snapshot each step has *no captured `lr`* that can go stale, so "registry says lr=X but the optimizer still uses the old lr" becomes impossible rather than merely caught. Identify each currently-representable illegal state in the train/optimize path (stale captured hyperparameter; a field marked HOT in the schema that the consuming code has actually baked; a net-shape/precision mismatch) and show how the refactored boundary removes its representability.

## Eventual C++ simulation — a robustness property to verify, NOT a burden
The maintainer plans to eventually reimplement the **simulation** (the rollout / search engine) in C++. This should be **easy to accommodate** and is, in fact, a *free* benefit of doing the boundaries right: a correctly-placed ACL between the simulation and the training/optimization/registry machinery means a C++ simulation is a drop-in behind that boundary (data in, data out; hyperparameters supplied through a stable, language-agnostic interface). So treat C++-readiness as a property your proposed boundaries should **demonstrably have** — verify the simulation↔rest seam in your design is clean and language-agnostic enough that swapping the Python sim for a C++ one disturbs nothing in the training/optimization refactor. Frame it as the refactor making the infrastructure *more flexible/robust*, not as added scope. (The companion doc `docs/sim-parallelization-viability.md` on branch `docs/sim-parallelization-viability` — read it via `git show docs/sim-parallelization-viability:docs/design/simulation-parallelization-viability.md` — already characterizes the simulation hot path and a compiled core; compose with it, don't contradict it.)

## Survey targets (read each fully before citing — no grep fragments)
In your worktree (these are present on `feat/hp-registry`):
- `chocofarm/az/mlp_jax_train.py` — **the conflation site**: `JaxTrainer` builds `optax.adam` + the jit update closures, capturing lr/l2/betas/eps once.
- `chocofarm/az/exit_loop.py` — the ExIt loop; the registry wiring (seed → `ConfigSnapshot` per-iteration reads → RESTART-refusal); where training is driven.
- `chocofarm/az/mlp.py`, `chocofarm/az/mlp_jax.py` — the net (numpy + compiled forward).
- `chocofarm/az/value_target.py`, `chocofarm/az/gumbel_search.py`, `chocofarm/model/env.py` — the value target, the search (its hyperparameters + the sim boundary), the env.
- `chocofarm/hp/schema.py` — the registry's typed schema and the per-field Mut facets (HOT/RESTART/INSTANCE) with rationale; `chocofarm/hp/registry.py` — the read path / RESTART-refusal / `ConfigSnapshot`; `chocofarm/config.py` — the infra surface.
Cross-branch docs (read via `git show`):
- the registry spec: `git show docs/hyperparam-registry-spec:docs/design/hyperparameter-registry.md` (the consolidation + the "optional follow-on" this refactor makes natural).
- the sim-parallelization note (above).
Orientation: `docs/handoff-2026-06-15.md`, `docs/STATUS.md`, `docs/design/alphazero-surrogate-design.md`.

## The deliverable — a design NOTE (implementation-ready)
Write `docs/design/training-optimization-refactor.md`, matching the house style of `docs/design/*.md`. It must be implementation-ready: after the maintainer signs off, they (or an implementer) go straight to building it. Include:
1. **Diagnosis** of the training/optimization conflation — concretely, with file/line-level references — and the SSOT/DRY violations and representable-illegal-states it currently admits.
2. **The proposed boundaries (ACLs) and the type structure** — the Optimizer abstraction (owns the optax transform; hyperparameters as injectable runtime state → `inject_hyperparams` natural) separated from the Trainer (owns loss / data / epochs / the gradient step); the simulation↔rest seam; the registry-snapshot read path. Give the interfaces/signatures concretely.
3. **The pure structural refactors** (notwithstanding ACLs) that reduce conflation and make hot-params fall out.
4. **The HOT-ness table**: every hyperparameter that *ought* to be HOT, whether it is today, what blocks it, and how the refactor makes it genuinely HOT — vs the genuinely-RESTART/INSTANCE set that stays non-hot (and is made *unrepresentable* as live, per MISU).
5. **SSOT/DRY** and **MISU** each addressed explicitly as their own sections, mapped to the concrete changes.
6. **C++-readiness verification** — show the simulation seam is language-agnostic.
7. **A sequenced implementation plan** — ordered steps, each independently reviewable, with the test/verification at each step.
8. Honest caveats / costs / tradeoffs.

## Constraints
- Design/analysis ONLY. No code changes, no running anything, no redis/process side effects.
- Commit on `docs/training-optimization-refactor` with **EXPLICIT PATH ONLY** (`git add docs/design/training-optimization-refactor.md`; never `git add -A`/`.`). Commit message ends with exactly: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Do **NOT** push (the orchestrator pushes after review).
- Append this entire commission prompt verbatim as "Appendix A — commission prompt" (consult-record discipline; `docs/consults/consult-001-prompt.md` is the format reference).
- Honest, mechanistic, scoped; where uncertain, say so.

## Final message
Your returned message **IS the record** — render the design note's substance self-containedly: the diagnosis, the proposed boundaries + type structure, the HOT-ness table, the SSOT/DRY and MISU treatments, the C++-seam verification, and the sequenced implementation plan. Report the branch name, commit SHA, and file path. Not a pointer to the file — the substance.

---

## Appendix B — out-of-frame MISU audit (hack-rationalization-detector, verbatim)

> An out-of-frame adversarial auditor ran the hack-rationalization-detector
> against this note's §5.2 MISU "unrepresentable by construction" claims before
> commit. Its finding — that the note overclaimed "unrepresentable" for the
> `lr`/`betas`/`eps` injected-dict path while only `l2` (a traced call arg) truly
> earned it — was *applied*, not deflected: §2.1's `Optimizer.step` now binds the
> hyperparameters into the required call signature via `_with_hparams` (the same
> "no forgettable write-site" shape `l2` has), §5.2 I1/I2/I4 were reworded to the
> honest verb ("reduced to a single required call argument" / "rejected loudly by
> jax," not "unrepresentable"), §5.1 D1 now names the `__init__` placeholder
> nuance, and §7 step 6 gained a write-site test (`step` with `lr=0` leaves params
> unchanged) that catches a future `step` reading the placeholder. Recorded
> verbatim per the consult-record discipline.

---

### Commission prompt (verbatim)

You are an OUT-OF-FRAME adversarial auditor. You did not write the document under review and you must treat its claims as the object of suspicion, not as context to agree with. Do NOT modify any file. Do NOT run any job/redis/training. Read-only analysis.

Run the hack-rationalization-detector skill's procedure (the skill is at /home/bork/.claude/skills/hack-rationalization-detector — read its SKILL/README and use its scripts and known-cases as your few-shot) against ONE specific claim in a design note.

## The artifact under review
`docs/design/training-optimization-refactor.md` in the worktree `/home/bork/w/vdc/chocobo-trainopt`. Read it in full (it is a design note proposing to split a `JaxTrainer` god-object into a separate `Optimizer` + slimmed `Trainer`, using `optax.inject_hyperparams` so optimizer hyperparameters become live/HOT instead of baked at construction).

## The specific suspicion to stress-test (this is your frame to DISTRUST, not adopt)
The note's §5.2 (MISU — Make Illegal States Unrepresentable) claims four illegal states (I1 stale captured lr; I2 stale l2/betas/eps; I3 schema-says-HOT-but-code-baked; I4 net-shape/moment-pytree mismatch) become "unrepresentable by construction" rather than "merely refused at runtime." Pressure-test specifically:

1. **The I1 overclaim risk.** The note's `Optimizer.step` sketch (§2.1) still does `st.hyperparams["learning_rate"] = hp.lr` by hand each step — a WRITER that an implementer could forget or skip. Is "no captured self.lr" actually enough to call stale-lr *unrepresentable by construction*, or has the note merely RELOCATED the staleness risk from "a captured copy that drifts" to "a per-step manual write that could be omitted"? Is the claim honest, or is it the documented failure shape (a named-better-fix — e.g. binding hp into the call signature so there is no mutable hyperparams dict to forget — that got downgraded)?

2. **Enumerate the writers** of the optimizer's effective lr in the PROPOSED design: who/what sets the lr the step actually uses? (the snapshot adapter `adam_hparams_from`, the `step` hand-write into `opt_state.hyperparams`, any reset path, the train_value.py Stage-1 gate which the note admits is a second call site). Does the note's MISU claim quantify over ALL of them, or does it implicitly assume the single `exit_loop` call site?

3. **I3 / I4 honesty.** I3 is admitted to be "test-catchable" not unrepresentable — is that conflated with the I1/I2 "unrepresentable" claim in a way that overstates? I4 leans on jax tree-mismatch rejection — is that genuinely "by construction" or still a runtime failure (loud, but runtime)?

You do NOT need to read the actual chocofarm source unless a claim hinges on it; the note quotes line numbers you may spot-check (`chocofarm/az/mlp_jax_train.py`, `chocofarm/az/optimizer.py` does NOT exist yet — it is proposed). The optax `inject_hyperparams` mechanism: it stores named hyperparameters in `opt_state.hyperparams` as a mutable dict; whether they are read live depends on the caller writing them each step.

## Deliverable
Produce the hack-rationalization-detector's exact output template (FRAME CHECK / GENERAL FIX / PATCH SHIPPED / DOWNGRADE / WRITER DELTA / RUNTIME / TELLS / VERDICT / WHY / FINDINGS BEYOND VERDICT). Be concrete and honest; if the note's MISU framing is sound, say so; if it overclaims "unrepresentable" where it should say "single write-site, structurally checkable," name exactly which sentences overclaim and what the honest wording is. Return the full artifact verbatim as your final message — it will be read in full.

---

### Audit report (verbatim)

This is decisive. The note treats `l2` and `lr` asymmetrically, and the asymmetry is exactly the gap:

- **`l2`** becomes a **traced arg** of `value_and_grad` — bound into the call signature. There is genuinely no slot to forget: `train_step(self, batch, hp, l2, ...)` cannot be called without supplying `l2`, and `value_and_grad(...)(params, batch, l2, ...)` cannot omit it. For `l2`, "no captured copy" is structurally true.
- **`lr`/`betas`/`eps`** become a **mutable dict the `step` hand-writes** (`st.hyperparams["learning_rate"] = hp.lr`). This is the very signature-vs-dict distinction. The note even passes `hp` as a call arg to `step` — but then the `step` body relies on the implementer writing four dict keys. If `step` is implemented forgetting the write (or a future `step`-variant like `train_step_value`'s path), the injected dict retains its `__init__` placeholder (`learning_rate=1.0`) and steps stale/wrong — silently.

So the better fix for `lr` was *demonstrated in the same document for `l2`* (bind into the traced call signature so there is no mutable dict) but **not applied to `lr`** — and the note never names why. That is the named-better-fix-shown-elsewhere-and-not-generalized shape, though it is softer than the classic discipline-word downgrade (the scanner found no co-occurrence tell).

I have all evidence. Producing the artifact.

#### Hack-rationalization review: `training-optimization-refactor.md` §5.2 MISU "unrepresentable by construction" (I1/I2/I3/I4)

FRAME CHECK: **Out-of-frame.** I did not write the note and treated its §5.2 "unrepresentable by construction" claims as the object of suspicion, not as context. I read the note end to end, ran both skill scripts, and spot-checked the two real call sites it cites (`exit_loop.py`, `train_value.py`) and the current `JaxTrainer` step methods. Proceeding.

GENERAL FIX (for the I1/I2 staleness class, stated as one invariant):
*An optimizer coefficient is supplied as a required argument of the update call (so omitting it is a type/arity error), never as a mutable dict the caller is trusted to overwrite each step.* The note already realizes exactly this invariant for `l2` (a traced `value_and_grad` arg — no dict, nothing to forget). Generalizing the same shape to `lr`/`b1`/`b2`/`eps` — pass them as args to a wrapped update whose body writes them, or bind them so `step` cannot run without them — would make I1/I2 genuinely unrepresentable. The honest one-liner: *the only writer of the effective lr is the call's argument list.*

PATCH SHIPPED (what §2.1/§5.2 actually proposes for `lr`/`betas`/`eps`):
`inject_hyperparams`-wrapped `optax.adam` whose scalars live in a **mutable** `opt_state.hyperparams` dict, with `Optimizer.step` **hand-writing four keys each call** (`st.hyperparams["learning_rate"] = hp.lr`, …) before `update`. The note then declares "There is no object field that can hold a stale `lr`… cannot be constructed" (§5.2 I1) and "No captured copy exists → I1 unrepresentable" (§4.1 table).

DOWNGRADE: There is **no stated cost** for why `lr` keeps the mutable-dict-plus-manual-write shape while `l2` gets the genuinely-unforgettable traced-arg shape. The asymmetry is presented as if both reach the same "unrepresentable" endpoint, but only `l2`'s does. The discipline-word the scanner would key on is absent (clean prose — see TELLS); the downgrade here is silent, achieved by *describing* the dict-write as if it were the same kind of structural guarantee as the call-arg binding. No concrete cost (no ADR-0004 partial-visibility, no contract-freeze) is named because there isn't one — optax's `inject_hyperparams` does support being driven by passing hyperparameter values to the update, so the more-general fix was available.

WRITER DELTA: **claimed 1 (the single `exit_loop` snapshot read via `adam_hparams_from`) vs enumerated ≥3 in the proposed design.** Independently enumerated writers of the optimizer's *effective* lr after the refactor:
1. `adam_hparams_from(snap.cfg)` — the snapshot→`AdamHParams` adapter (the one the note's MISU framing quantifies over).
2. `Optimizer.step`'s four hand-writes into `st.hyperparams[...]` — the per-step manual writer that an implementer must not omit; this is a real write-site the dict shape creates and the §2.1 sketch shows literally.
3. `Optimizer.__init__`'s placeholder seed (`learning_rate=1.0, b1=0.9, …`) — a write that becomes the *live* value on any step where (2) is skipped. A forgotten write doesn't fail loudly; it silently steps at `lr=1.0`.
4. **`train_value.py:53` `JaxTrainer(net, lr=lr, l2=l2)` → `train_step_value` (line 65)** — a *second, today-real* construction-and-step path the note's §2.1 `Optimizer.step` sketch never models. It builds from `argparse` CLI args (`--lr`, `--l2`), entirely outside the registry/snapshot/`AdamHParams` path, and steps via a different method (`train_step_value`, not `train_step`). The note admits this site exists in an §8 caveat ("`train_value.py` … a second call site Step 4 must cover") — but the §5.2 MISU claim ("the Optimizer has no `self.lr`") is stated **universally**, not carved to the `exit_loop` path. The MISU sentences implicitly quantify over the single `exit_loop` call site; writers (2)(3)(4) are out of frame for them.

RUNTIME: **unverified — derived on paper.** The note states this itself (§8: "This is design only. No code was run"). `optimizer.py` does not exist; the claim is an argument from proposed structure. No repro of either the stale state or its claimed impossibility exists.

TELLS (Step 1): **No co-occurrence tells** (`grep_tells.py`: 4 minimality-terms, 18 named-fix cues, 0 within-window co-occurrences). The note's prose is disciplined — it does *not* argue a better fix down with a discipline-word; this is not the Case-A/Case-B "scope creep next to producer/owner" signature. The overclaim here is the *quieter* shape: a stronger guarantee proven for one field (`l2`, traced arg) is asserted in the same breath for a sibling field (`lr`, mutable dict) that only has the weaker guarantee. The scanner is built for the loud shape and correctly reports clean; the finding is from Step 2 (writer enumeration) and the judge pass, not Step 1.

VERDICT: **narrower-but-justified — with a required wording correction.** The refactor's *core* claim is sound and is not a hack: removing the captured `self.lr` field genuinely dissolves the *current* I1 (a construction-time copy that drifts from the registry between boundaries). That is a real MISU win and the writer enumeration confirms the captured-copy writer is eliminated. But the specific sentence "I1 → unrepresentable… cannot be constructed" **overclaims**: the staleness is **relocated, not eliminated** — from a captured copy that drifts, to a mutable `opt_state.hyperparams` dict plus a per-step manual write that an implementer (or the un-modeled `train_value.py` path) can omit, leaving the `__init__` placeholder live and silent. This is not an UNDISCHARGED-HACK (no better fix was named-and-bypassed with a mood; the general fix is *available and even demonstrated for `l2`*, and the plan is real), but the "unrepresentable by construction" framing is stronger than the structure supports for `lr`/`betas`/`eps`.

WHY: For `l2` the note binds the value into the call signature, so omission is an arity error — genuinely unrepresentable. For `lr`/`betas`/`eps` it keeps a mutable dict the `step` overwrites by hand, so omission is a silent wrong-value step, not a construction error — that is "single-write-site, structurally checkable," not "unrepresentable." The MISU claim quantifies over the one `exit_loop` writer and does not cover the per-step dict-write, the placeholder seed, or the second real `train_value.py`/`train_step_value` path the note itself flags in a caveat.

FINDINGS BEYOND VERDICT (required):

- **Exact sentences that overclaim, and the honest rewording.**
  - §5.2 I1: "There is no object field that can hold a stale `lr`, so 'registry says X, optimizer steps at Y≠X' cannot be constructed." — Overclaims. `opt_state.hyperparams["learning_rate"]` **is** a mutable field that can hold a stale (or placeholder `1.0`) `lr` whenever `step` is called without the write. Honest version: *"There is no construction-time captured copy of `lr`; the live value lives in `opt_state.hyperparams` and is overwritten each `step` from `AdamHParams`. Staleness is no longer a copy that silently drifts — it is reduced to a single per-step write-site, which the facet-consistency test (§7 step 6) must cover, and which the call signature should enforce."*
  - §4.1 table, `train.lr` row: "No captured copy exists → I1 unrepresentable." — Same overclaim; "No captured copy exists → I1 reduced to one structurally-checkable write-site" is honest.
  - §2.1 sketch comment: "so there is NO captured lr/b1/b2/eps to go stale (MISU: I1/I2 unrepresentable)." — The "to go stale" is true of the *captured copy*; "unrepresentable" is not true of the dict. The `l2` path earns "unrepresentable"; the dict path earns "single write-site."

- **The asymmetry is the load-bearing tell, and it is fixable cheaply.** The note proves the strong property for `l2` (traced arg) and the weak one for `lr` in the *same design*, without noting the gap. The cheapest honesty-restoring move is also the general fix: drive the injected hparams by *passing* `AdamHParams` into the update call so there is no mutable dict for `step` to forget — then `lr`/`betas`/`eps` reach the same "unrepresentable" bar `l2` already has, and §5.2's universal claim becomes true rather than scoped-to-`exit_loop`. If the author prefers the dict, the wording must downgrade to "single structurally-checked write-site," and §7 step 6's facet test must additionally assert `step` writes all four keys (the test as described checks schema-facet vs read-path membership, not that the dict is actually populated each call — so it would **not** catch a `step` that forgets the write).

- **I3 is correctly NOT claimed "unrepresentable" — and the note is honest about it.** §5.2 says I3 is "caught structurally" / "test-catchable," and §8 reinforces it. This is *not* conflated with the I1/I2 "unrepresentable" claim in the prose itself — the note distinguishes them. Residual caveat: the §7 step 6 test as specified asserts set-membership (HOT fields ∈ live-read set), which is a static structural check, not a runtime guarantee; calling it "MISU" is a slight stretch (it is a test, i.e. runtime-refusal-at-CI-time), but the note already labels it "test-catchable," so this is honest, not an overclaim.

- **I4 is "loud runtime rejection," not strictly "by construction" — and the note's own words admit this while the heading overclaims.** §5.2 I4 says the mismatch is "rejected loudly by jax itself, not by a separate guard" and "You cannot construct an optimizer stepping a net of a shape it was not initialized against." The honest core: the moment pytree is *typed to* the params pytree, so a mismatched `step` **fails** — but it fails **at jax trace/execution time**, which is a runtime rejection (loud, good, but runtime), not a compile-time/type-system impossibility in Python. "Unrepresentable by construction" is the right phrase only in the weak sense "jax will not let the update proceed." The note's §1.3 I4 wording ("typed to… so a mismatch cannot be constructed") is slightly stronger than the §5.2 body ("rejected loudly"). Honest unified version: *"the moment pytree is structurally bound to the params pytree, so a shape mismatch is rejected loudly by jax at step time — a runtime failure, but one that needs no separate guard."* That is a real improvement over the current `_assert_no_derived_drift` (and the note keeps that guard as operator-facing), but it is fail-loud-at-runtime, in the same family as the RESTART-refusal it claims to transcend — just at a different layer (jax tree-check vs registry boundary-check).

- **The "single authority" §5.1 D1 claim has the same dict caveat.** "There is no second place for the value to drift" is true of the construction copy but glosses the placeholder seed in `__init__` (`learning_rate=1.0`), which is a second representable value for `lr` that becomes authoritative on any step missing the hand-write. Minor, but it is the same relocation surfacing in the SSOT section.

- **`train_value.py` is the missed producer (the Case-A rhyme).** This is the structural rhyme with known-case A's third writer: the note's MISU framing reasons from the `exit_loop` path it can see; the second construction site (`train_value.py` → `train_step_value`, fed by `argparse`, no snapshot) is exactly where a per-call-site reasoning leaves the property unproven. The note flags it as a Step-4 migration chore in §8 but does not extend the §5.2 "unrepresentable" claim to cover it. Until that path is migrated and modeled, "the Optimizer has no `self.lr`" is true of one of two real callers, and `train_step_value` has no `AdamHParams` plumbing in any sketch in the note. The fix is the same general invariant (lr is a required call arg everywhere), which would discharge both sites at once instead of one at a time.

- **What is genuinely sound (so the verdict isn't read as condemnation).** The split itself, the removal of the captured `self.lr`, making `l2` a traced arg, deleting the vestigial `lr`/`l2` signature channel, and keeping the genuine RESTART/INSTANCE set non-hot are all well-reasoned and the §8 caveats are unusually honest (it self-flags the second call site, the moment-interaction with live lr, the lost `l2==0` short-circuit, and "design only, nothing run"). The single correction needed is to stop saying "unrepresentable by construction" for the mutable-dict `lr`/`betas`/`eps` path and either (a) adopt the call-arg binding that makes it true, or (b) reword to "single, structurally-checkable per-step write-site" and harden §7 step 6 to assert the write actually happens.

> **Disposition (author, in-frame):** correction (a) adopted — §2.1 now binds the
> hyperparameters into the required `step` call signature via `_with_hparams`, so
> `lr`/`betas`/`eps` reach the same no-forgettable-write bar as `l2`; the §5.2,
> §4.1, §5.1, and §1.3 wordings were corrected to the honest verbs; §7 step 6
> gained the write-site test the audit asked for; and the §7-step-4 migration of
> `train_value.py` is named in the I1 scope note so the guarantee quantifies over
> both call sites.
