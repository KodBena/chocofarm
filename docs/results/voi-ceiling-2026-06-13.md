# Clairvoyant value-of-information ceiling (2026-06-13, point-in-time)

After the consult-001 detector fix (disjunctive cover from the real 17 overlaps). UNIT
values. The clairvoyant ceiling is detector-independent — it is handed the true present-set
for free, so it bounds the maximum any sensing could ever buy.

| policy | rate | vs static |
|---|---|---|
| realizable static (fixed route) | 0.0855 | — |
| greedy | 0.0806 | −6% |
| rollout (disjunctive detectors, own λ) | 0.0856 | +0% |
| **clairvoyant (free perfect info)** | **0.1454** | **+70%** |

## Conclusion

- **Adaptivity can pay ~+70%** — detector-independent, even under unit values. This refutes
  the earlier "unit values mute the margin" hypothesis: the value of knowing which five are
  present is structural and large (go straight to them, skip the absent, exit fast —
  clairvoyant banks 4.55/5 in E[T]=31 vs static's 16-treasure sweep at E[T]=47).
- **The bottleneck is policy quality, not values or the detector model** (now fixed). The
  rollout captures ~0% of the available 70%.

Source: `attic/experiments.py` (the standalone run); superseded by `run.py`'s harness, which
recomputes the same ceiling. Recorded here as the decisive finding.
