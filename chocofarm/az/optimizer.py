#!/usr/bin/env python3
"""
chocofarm AZ — the Optimizer abstraction (audit item M; training-optimization-refactor.md §2.1).

The Optimizer⊥Trainer split, the half that owns ONLY the optax transform and its hyperparameters.
`JaxTrainer` used to wear two hats — Trainer (loss / data marshalling / y-standardization / write-
back) AND Optimizer (build `optax.adam`, own the moment state, hold lr/l2/betas/eps). Because the
two were fused in one constructor, the optimizer's coefficients were captured when the Trainer was
built, which is the only time it is built (the loop builds it ONCE so Adam's moments persist) — so
they could not move without a restart. Splitting them dissolves that: this object's hyperparameters
are ITS runtime state, read each step from a required argument, with nowhere to bake them.

  * `AdamHParams` is the optimizer's live scalar hyperparameters in its OWN vocabulary
    (lr/b1/b2/eps — the HOT subset of TrainConfig). `l2` is deliberately NOT here: it is a LOSS
    coefficient (a traced `value_and_grad` arg owned by the Trainer's loss), not optimizer state.

  * `Optimizer` owns an optax `GradientTransformation` whose lr/b1/b2/eps are INJECTED runtime state
    (`optax.inject_hyperparams`), not closed-over constants. The transform is built ONCE; the moment
    pytree + the injected-hparam state live in `opt_state`, which is typed to the params pytree it
    was `init`'d from (a shape mismatch is then rejected loudly by jax at step time, no separate
    guard — design §5.2 I4). `inject_hyperparams` is not a feature added here; it is the ONLY
    construction the Optimizer does, and there is no `self.lr` to bake.

The SINGLE-WRITER property is construction-enforced (the audit's out-of-frame correction, design
§2.1 + Appendix B): the live hyperparameters are bound into the REQUIRED call signature of the
jit'd step the Optimizer builds (`make_update`). They are written into `opt_state.hyperparams`
INSIDE that call from its traced `AdamHParams` arguments — the same "no forgettable write-site"
shape `l2` already has as a traced loss arg. There is no step path that reads the injected dict
without first setting it from the required `hp`, so omitting a hyperparameter is an arity error at
the call, not a silent step on the `inject_hyperparams` `__init__` placeholders. This is what makes
lr/betas/eps genuinely HOT: there is no slot for a stale value AND no callable shape that skips
supplying one (R13's hack-review noted the slot's single-writer was a convention; here it is
structural).

Public Domain (The Unlicense).
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax


class AdamHParams(NamedTuple):
    """The optimizer's live scalar hyperparameters — the HOT subset of `TrainConfig`, in the
    optimizer's own vocabulary (design §2.1). Built from a `ConfigSnapshot` each step by the loop's
    ACL adapter (`adam_hparams_from`). `betas` are the pair `(b1, b2)` to match optax; `lr`/`eps`
    are scalars. `l2` is NOT here — it is a LOSS coefficient owned by the Trainer's loss (design D3),
    a traced `value_and_grad` arg, not optimizer state.

    These four fields ARE the live writers of the effective Adam coefficients: they are passed as
    REQUIRED arguments into the jit'd update the Optimizer builds, and written into the injected
    optax state inside that call. There is no captured copy and no default — a missing hyperparameter
    is an arity error (the construction-enforced single-writer, design §2.1 / Appendix B)."""

    # Each field holds EITHER a python float (the default-construct / `adam_hparams_from` path) OR a
    # traced jax scalar (the `_hp_arrays` `jnp.asarray(...)` path that casts to the optax state's
    # dtype before the jit'd step) — both forms genuinely flow through these slots.
    lr: float | jax.Array
    b1: float | jax.Array
    b2: float | jax.Array
    eps: float | jax.Array


class Optimizer:
    """Owns an optax `GradientTransformation` whose lr/b1/b2/eps are INJECTED runtime state
    (`optax.inject_hyperparams(optax.adam)`), not closed-over constants (design §2.1). The transform
    is built once; the moment pytree + injected-hparam state live in `self.opt_state`, typed to the
    `params` pytree `__init__`/`reset` saw.

    Adam configuration matches the manual optimizer the JAX path replaced: COUPLED L2 on weight
    matrices only (the `0.5·l2·‖W‖²` penalty is in the Trainer's loss, so its gradient flows through
    Adam's preconditioner — exactly the numpy path's `g + l2·W`, NOT optax's decoupled
    `add_decayed_weights` which would also decay biases). optax stays plain (injected) Adam; L2 is
    the loss's.

    The injected hyperparameters are seeded with PLACEHOLDERS at construction; they are NEVER the
    authoritative value on any real step, because every update is reached only through a jit'd step
    (built by `make_update`) that REQUIRES an `AdamHParams` argument and writes it into the state
    (`_with_hparams`) before `opt.update`. There is no code path that steps on the placeholders."""

    def __init__(self, params: dict[str, Any]) -> None:
        # inject_hyperparams puts lr/b1/b2/eps in opt_state.hyperparams as traced values. The init
        # values are placeholders — immediately overwritten by the first step's AdamHParams (see
        # _with_hparams) and never authoritative on any real step. The moment pytree is typed to
        # `params` (design §5.2 I4: a different-shaped grads tree is a jax tree mismatch at step).
        # optax is a stub-gap (no py.typed); _tx and opt_state are typed `Any` at the seam (P8 use-site Any).
        self._tx: Any = optax.inject_hyperparams(optax.adam)(
            learning_rate=1.0, b1=0.9, b2=0.999, eps=1e-8)
        self.opt_state: Any = self._tx.init(params)

    @staticmethod
    def _with_hparams(opt_state: Any, hp: AdamHParams) -> Any:
        """Return a copy of `opt_state` whose injected hparams are exactly `hp`. The ONE place the
        live scalars enter the optax state — and it is reached only THROUGH a jit'd step that
        REQUIRES `hp`, so the injected dict is never read without first being set from the call's
        argument (the construction-enforced single-writer, design §2.1). `inject_hyperparams` keeps
        each hyperparameter as a jax array; cast to the slot's dtype so the assignment stays
        traceable. `opt_state` is `Any` (optax stub-gap; P8 use-site Any at the optax/jax seam)."""
        hps = dict(opt_state.hyperparams)
        hps["learning_rate"] = jnp.asarray(hp.lr, dtype=hps["learning_rate"].dtype)
        hps["b1"] = jnp.asarray(hp.b1, dtype=hps["b1"].dtype)
        hps["b2"] = jnp.asarray(hp.b2, dtype=hps["b2"].dtype)
        hps["eps"] = jnp.asarray(hp.eps, dtype=hps["eps"].dtype)
        return opt_state._replace(hyperparams=hps)

    def make_update(self, grad_fn: Callable[..., Any]) -> Callable[..., Any]:
        """Build the jit'd update step fusing `grad_fn` (the Trainer's loss `value_and_grad`) with the
        injected-hparam write and the optax update — ONE compiled kernel (XLA fuses forward, backward,
        and the Adam step, the equivalence-test contract). `grad_fn(params, *loss_args)` must return
        `(value, grads)` with `has_aux` style `value = (loss, aux)` so the kernel can surface `aux`
        for logging.

        The returned closure's signature is `(params, opt_state, hp, *loss_args) -> (params,
        opt_state, aux)`: `hp` is a REQUIRED `AdamHParams`, written into `opt_state` via
        `_with_hparams` INSIDE the jit from its traced fields before `opt.update`. So the only writer
        of the effective lr/b1/b2/eps is this call's `hp` argument — there is no overload that reads
        the injected dict un-set, and omitting `hp` is an arity error (design §2.1 / Appendix B). The
        Optimizer's `opt_state` is threaded by the caller (the Trainer holds `optimizer.opt_state`),
        keeping this closure pure for jit."""
        tx = self._tx

        @jax.jit
        def _update(params: dict[str, Any], opt_state: Any, hp: AdamHParams,
                    *loss_args: Any) -> tuple[Any, Any, Any]:
            (loss, aux), grads = grad_fn(params, *loss_args)
            opt_state = Optimizer._with_hparams(opt_state, hp)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, aux

        return _update

    def reset(self, params: dict[str, Any]) -> None:
        """Re-init the moment + injected-hparam state against a (possibly replaced) params pytree —
        the `sync_from_net` semantics, now a method ON the optimizer (one responsibility resetting its
        own state), typed to the params it is given (design §5.2 I4 / S4)."""
        self.opt_state = self._tx.init(params)
