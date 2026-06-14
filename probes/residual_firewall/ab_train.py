#!/usr/bin/env python3
"""Firewall probe — the decisive fixed-dataset supervised A/B for the residual block.

Trains residual-OFF vs residual-ON (and optional variants) on ONE frozen dataset with identical
init-seed / LR / batch order / steps. On fixed data a strictly more expressive net MUST reach
<= the baseline's loss; if residual-ON is worse it is an OPTIMIZATION flaw, not a target or
convergence confound.

Variants (selected by name on the CLI):
  off            baseline (no block)
  on             residual block, He-init (the shipped config)
  on_zeroinit    residual block, Wr2/br2 = 0 at init  -> block starts as exact identity
  on_smallinit   residual block, Wr1/Wr2 scaled x0.1  -> block starts near-identity
  on_noouterrelu  residual block but head_in = a2 + zr2 (NO outer ReLU; pure pre-activation skip)

`on_*` variants beyond `off`/`on` are built by post-hoc surgery on a fresh residual net so the
trunk/head init stays bit-identical across every arm. Run pinned + bounded:

    CHOCO_AZ_DTYPE=float64 timeout 600 taskset -c 0,1,2,3 \
        python probes/residual_firewall/ab_train.py <dataset.npz> <out.json> <epochs> <lr> [variants...]
"""
import sys, json, time
import numpy as np
from chocofarm.az.mlp import ValueMLP


def build(variant, in_dim, H, na, seed):
    """All arms share bit-identical trunk+head init (block params drawn last)."""
    residual = (variant != "off")
    net = ValueMLP(in_dim, hidden=H, n_actions=na, seed=seed, residual=residual)
    if variant == "on_zeroinit":
        net.Wr2 = np.zeros_like(net.Wr2); net.br2 = np.zeros_like(net.br2)
        net._init_adam()
    elif variant == "on_smallinit":
        net.Wr1 = net.Wr1 * 0.1; net.Wr2 = net.Wr2 * 0.1
        net._init_adam()
    elif variant == "on_noouterrelu":
        net._fw_no_outer_relu = True  # patched forward/backward below
    return net


def patch_no_outer_relu(net):
    """Replace the block's outer ReLU with a plain pre-activation skip: head_in = a2 + zr2.
    Monkeypatched _forward + _residual_backward on this one instance."""
    import types

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1; a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2 + self.b2; a2 = np.maximum(z2, 0.0)
        zr1 = a2 @ self.Wr1 + self.br1; ar1 = np.maximum(zr1, 0.0)
        zr2 = ar1 @ self.Wr2 + self.br2
        head_in = a2 + zr2                         # NO outer ReLU
        res_cache = (zr1, ar1, zr2, None)
        v_std = (head_in @ self.Wv + self.bv).ravel()
        logits = (head_in @ self.Wp + self.bp) if self.n_actions is not None else None
        return (X, z1, a1, z2, a2, head_in, res_cache), v_std, logits

    def _residual_backward(self, dhead, a2, res_cache):
        zr1, ar1, zr2, _ = res_cache
        da2 = dhead.copy()                         # skip: head_in = a2 + zr2 (no ReLU gate)
        dzr2 = dhead
        dWr2 = ar1.T @ dzr2; dbr2 = dzr2.sum(0)
        dar1 = dzr2 @ self.Wr2.T
        dzr1 = dar1 * (zr1 > 0)
        dWr1 = a2.T @ dzr1; dbr1 = dzr1.sum(0)
        da2 += dzr1 @ self.Wr1.T
        return da2, {"Wr1": dWr1, "br1": dbr1, "Wr2": dWr2, "br2": dbr2}

    net._forward = types.MethodType(_forward, net)
    net._residual_backward = types.MethodType(_residual_backward, net)


def block_grad_norm(net):
    """L2 norm of the residual-block grads at the current Adam state (m/v not the grad, but the
    last applied step magnitude is informative). Returns trunk vs block weight norms."""
    if not net.residual:
        return None
    bn = float(np.sqrt(sum(float((net.__dict__[k] ** 2).sum())
                           for k in ("Wr1", "Wr2"))))
    tn = float(np.sqrt(sum(float((net.__dict__[k] ** 2).sum())
                           for k in ("W1", "W2"))))
    return {"block_w_norm": bn, "trunk_w_norm": tn}


def run(variant, X, PI, M, Y, in_dim, H, na, seed, epochs, lr, l2, batch, data_seed):
    net = build(variant, in_dim, H, na, seed)
    if variant == "on_noouterrelu":
        patch_no_outer_relu(net)
    net.set_value_scale(float(Y.mean()), float(Y.std()))
    n = X.shape[0]
    steps = max(1, n // batch)
    rng = np.random.default_rng(data_seed)        # SAME batch order across arms (same data_seed)
    curve = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        ce_tot = v_tot = cnt = 0.0
        for s in range(steps):
            b = idx[s * batch:(s + 1) * batch]
            if len(b) == 0:
                continue
            ce, vl = net.train_step(X[b], PI[b], M[b], Y[b], lr, l2, alpha=1.0, beta=1.0)
            ce_tot += ce; v_tot += vl; cnt += 1
        # full-dataset eval CE/MSE (deterministic, comparable across arms)
        _, v_std, logits = net._forward(X)
        p = net._masked_softmax(logits, M)
        logp = np.where(p > 0, np.log(np.clip(p, 1e-12, 1.0)), 0.0)
        full_ce = float(-np.mean(np.sum(PI * logp, axis=1)))
        yz = (Y - net.y_mean) / net.y_std
        full_mse = float(np.mean((v_std - yz) ** 2))
        rec = {"epoch": ep, "train_ce": ce_tot / cnt, "train_mse": v_tot / cnt,
               "full_ce": full_ce, "full_mse": full_mse}
        bg = block_grad_norm(net)
        if bg:
            rec.update(bg)
        curve.append(rec)
    return curve


def main():
    ds, out, epochs, lr = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    variants = sys.argv[5:] if len(sys.argv) > 5 else ["off", "on"]
    z = np.load(ds)
    X, PI, M, Y = z["X"], z["PI"], z["M"], z["Y"]
    in_dim, na = X.shape[1], PI.shape[1]
    H, seed, l2, batch, data_seed = 256, 0, 1e-4, 256, 7
    results = {}
    t0 = time.time()
    for v in variants:
        c = run(v, X, PI, M, Y, in_dim, H, na, seed, epochs, lr, l2, batch, data_seed)
        results[v] = c
        last = c[-1]
        print(f"[{v:16s}] lr={lr:g} ep{epochs}: full_CE={last['full_ce']:.4f} "
              f"full_MSE={last['full_mse']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"lr": lr, "epochs": epochs, "n": int(X.shape[0]), "results": results},
              open(out, "w"), indent=2)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
