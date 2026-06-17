// cpp/include/chocofarm/features.hpp
// Purpose: the C++ port of the AZ belief featurization + the action<->slot mask, mirroring
//   chocofarm/az/features.py (the FeatureLayout / FeatureBuilder) and chocofarm/az/actions.py
//   (n_action_slots / the legal mask). EVERY dimension is DERIVED from the env (ADR-0012 P1) —
//   feat_dim = 5N + 3nD + 6 + n_tel (= 241 on the live env); n_slots = N + nD + 1 (= 65). Nothing
//   is hardcoded.
//
//   The feature vector is float-sensitive (held to the ADR-0012 P6 behavioral bar). The LEGAL MASK
//   is a logic invariant (ADR-0012 P6/P7): it is bit-identical to Python's for the same
//   (loc, belief) — illegal-slot mass == 0.0 exactly — so the parity harness asserts it bit-exact.
//
// Public Domain (The Unlicense).
#pragma once

#include <cstdint>
#include <set>
#include <span>
#include <vector>

#include "chocofarm/env.hpp"
#include "chocofarm/feature_layout.hpp"

namespace chocofarm {

// Fixed action-space size for this env (mirrors actions.n_action_slots): N collects + nD senses +
// 1 TERMINATE. Slot 0..N-1 = ("t", i); N..N+nD-1 = ("d", j); N+nD = TERMINATE (always legal).
int n_action_slots(const Environment& env);
int term_slot(const Environment& env);          // index of the always-legal TERMINATE slot

// The fixed slot for an action (mirrors actions.action_to_slot).
int action_to_slot(const Environment& env, const Action& a);

// The legal-action mask over the fixed slots (mirrors actions.legal_mask): 1.0 on each legal action
// slot + the always-legal TERMINATE slot, 0.0 elsewhere. This is the LOGIC INVARIANT M the runner
// emits — bit-identical to Python's by construction (the same env.legal_actions set mapped onto the
// same slot bijection). Returned as float (the wire dtype) so the comparison is exact.
std::vector<float> legal_mask(const Environment& env, const std::vector<uint32_t>& bw,
                              const std::set<int>& collected);

// The §2.2 feature vector for (loc, bw, collected), mirroring features.FeatureBuilder.build. The
// layout is the canonical ordered block table (per-treasure N×5, per-detector nD×3, global
// 6+n_tel). Returned as float64 internally, cast to float32 at the wire by the runner. Float-
// sensitive (ADR-0012 P6 behavioral bar), not asserted bit-exact.
class FeatureBuilder {
  public:
    explicit FeatureBuilder(const Environment& env);
    int dim() const { return dim_; }
    // `loc` is the current standing point (resolved Point, as in env.coord). `bw`/`collected` are
    // the live belief + collected set. Returns a length-`dim()` float64 vector.
    std::vector<double> build(const Point& loc, const std::vector<uint32_t>& bw,
                              const std::set<int>& collected) const;

    // The legal-action mask sliced from an ALREADY-BUILT feature vector (mirrors
    // actions.legal_mask_from_features): the per-treasure `available` block IS the collect-legal mask,
    // the per-detector `informative` block IS the sense-legal mask, TERMINATE always legal. The hot-path
    // mask — it REUSES build()'s belief sweep instead of recomputing it via env.legal_actions →
    // marginals (ADR-0012 P1: marg has ONE home, build's; the mask consumes it). `feat` is the
    // length-dim() vector build() returned. Bit-identical to legal_mask(env, bw, collected) for the same
    // state: available == (marg>0 ∧ ¬collected) == the collect test; informative == (0<cnt<nb) ==
    // env.informative.
    [[nodiscard]] std::vector<float> legal_mask_from_features(std::span<const float> feat) const;

  private:
    const Environment& env_;
    int N_;
    int nD_;
    int n_tel_;
    int dim_;
    double diag_;          // bounding-box diagonal over all coords (map_diag)
    double log_nworlds_;   // log(|worlds|)
    FeatureLayoutSpec layout_;  // the §2.2 block table, runtime-read from the Python-emitted SSOT
};

}  // namespace chocofarm
