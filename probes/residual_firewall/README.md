# residual_firewall — RETIRED (2026-06-15)

The two probes that lived here — `gen_frozen.py` (froze one self-play dataset)
and `ab_train.py` (the fixed-dataset residual ON/OFF A/B trainer) — have been
removed. They are not lost work; the question they asked is **settled and
recorded authoritatively**, and the probes themselves had gone dead.

## Why retired (not de-forked)

1. **`ab_train.py` was already broken.** It drove the numpy `ValueMLP`'s training
   internals — `net.train_step`, `net._init_adam`, and a monkeypatch of
   `net._residual_backward`/`net._forward`. The JAX migration made `mlp.py`
   inference-only and **deleted those methods** (`cleanup/remove-numpy-adam`; see
   `docs/handoff-2026-06-15.md` §"Removal of Hand-Rolled Adam", and the
   `mlp.py` header: "the hand-rolled Adam + manual backprop that used to live
   here are gone"). The probe raises `AttributeError` on the first training step.
   Its historical A/B verdict was therefore measured on the numpy-Adam optimizer,
   **not** the production `JaxTrainer` the loop ships — the defect the audit
   flagged (`docs/notes/audit/architectural-audit-2026-06-15-appendix.md` §3
   "ab_train.py forks the entire training loop").

2. **A JAX de-fork would test a degenerate question.** The probe's analytical
   content was a 5-way variant sweep (`off`, `on`, `on_zeroinit`, `on_smallinit`,
   `on_noouterrelu`). The winning arm — `on_noouterrelu`, the no-outer-ReLU
   pre-activation skip `head_in = a2 + zr2` — **is now the only residual form the
   codebase has** (`chocofarm/az/forward.py`: `head_in = a2 + zr2 # … firewall
   A/B: best CE`). The OLD outer-ReLU `on` form and the two init-surgery variants
   the probe compared against no longer exist in the stack, so a re-run against
   `JaxTrainer` collapses to a trivial `off` vs `on(=shipped)` A/B with nothing
   left to discriminate. The probe's degrees of freedom were spent the moment its
   answer shipped.

## Where the answer lives now (authoritative records)

- **The representability / optimization verdict** (fixed-dataset A/B): residual-ON
  reaches ≤ baseline CE at every LR and on two datasets; the init-scale hypothesis
  was falsified; `on_noouterrelu` was the best arm — and it was adopted into the
  production forward. Full result, including the variant table this probe produced:
  **`docs/consults/firewall-residual-loss.md`**.

- **The online rate verdict** (the question that actually matters for the project):
  a matched **JAX-trained** trial (Residual ON vs OFF, m=24) found both plateau at
  ~0.10 — "calibration, not capacity." Data preserved under `runs/matched_reson`
  and `runs/matched_resoff` (gitignored). See **`docs/handoff-2026-06-15.md`**
  (§"matched experimental trial").

- **The shipped architecture** that embodies the winning variant:
  **`chocofarm/az/forward.py`** (the one forward core) and its toggle in
  `chocofarm/az/mlp_jax_train.py` / `chocofarm/az/mlp.py`.

If a *fresh* residual ablation is ever wanted, drive the production stack directly:
`python -m chocofarm.az.exit_loop --residual …` for ON vs the same command without
`--residual` for the bit-identical OFF baseline (the clean A/B the production loop
already supports), comparing eval rate at matched iteration count and `n_sims` —
the online-question protocol the consult's §5 caveat names.
