// cpp/src/nmcs.cpp
// Purpose: the C++ NMCS Policy implementation (see nmcs.hpp). A faithful reimplementation of
//   chocofarm/solvers/nmcs.py against the C++ env port — the level-k nested recursion, the
//   memorize-and-replay best-line rule, the determinized GreedyPolicy leaf playout, the per-move
//   averaged evaluation, and the λ-penalized scoring — behind the composable Policy seam (ADR-0012
//   P2/P7: behavioral parity, NOT byte-identity; the env/runner core is untouched).
//
// Public Domain (The Unlicense).
#include "chocofarm/nmcs.hpp"

#include <algorithm>
#include <limits>

namespace chocofarm {

namespace {
// −inf sentinel for the argmax / best-line score (mirrors nmcs.py's -np.inf).
constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
}  // namespace

// ---- GreedyBase: the λ-rational myopic leaf base (mirrors solvers.base.GreedyPolicy) -------------
Action GreedyBase::decide(const Environment& env, const Loc& loc, const std::vector<uint32_t>& bw,
                          const std::set<int>& collected, double lam, std::mt19937_64& rng) const {
    (void)rng;  // GreedyPolicy is deterministic (Python calls it with rng=None)
    std::vector<double> marg = env.marginals(bw);
    double best = 0.0;
    Action act = terminate_action();
    for (int i = 0; i < env.N(); ++i) {
        if (collected.count(i) != 0 || marg[i] <= 0.0) continue;
        double s = marg[i] * env.value(i) - lam * env.dist(loc.pt, env.treasure_pt(i));
        if (s > best) {  // strict >: first treasure wins a tie (matches Python's `if s > best`)
            best = s;
            act = Action{ActionKind::Treasure, i};
        }
    }
    return act;
}

// ---- base_value: play the base to the end in a fixed world (mirrors solvers.base._base_value) ----
double base_value(const Environment& env, const Policy& base, Loc loc, std::vector<uint32_t> bw,
                  std::set<int> collected, uint32_t world, double lam) {
    double R = 0.0, T = 0.0;
    // env.max_steps() is the single episode-horizon home (mirrors _base_value's range(env.max_steps)),
    // read from the env so a playout's horizon cannot silently desync from the Python env's.
    std::mt19937_64 unused(0);  // the base is deterministic; rng is part of the seam signature only
    for (int step = 0; step < env.max_steps(); ++step) {
        Action a = base.decide(env, loc, bw, collected, lam, unused);
        if (a.kind == ActionKind::Terminate) break;
        StepResult sr = env.apply(loc, bw, collected, a, world);
        R += sr.reward;
        T += sr.dt;
    }
    return R - lam * (T + env.exit_cost(loc.pt));
}

// ---- candidate generation (mirrors solvers.base.candidate_actions, include_terminate=True) -------
std::vector<Action> NMCSPolicy::candidates(const Environment& env, const Loc& loc,
                                           const std::vector<uint32_t>& bw,
                                           const std::set<int>& collected) const {
    std::vector<double> marg = env.marginals(bw);

    // nearest `cand_det` still-informative detectors by env.d(loc, ("d", i)); stable on face id
    // (Python's `sorted` is stable, so a distance tie keeps ascending-id order).
    std::vector<int> dets;
    for (int i = 0; i < env.n_detectors(); ++i)
        if (env.informative(i, bw)) dets.push_back(i);
    std::stable_sort(dets.begin(), dets.end(), [&](int a, int b) {
        return env.dist(loc.pt, env.face_pt(a)) < env.dist(loc.pt, env.face_pt(b));
    });
    if (static_cast<int>(dets.size()) > cfg_.cand_det) dets.resize(cfg_.cand_det);

    // nearest `cand_tre` uncollected, marg>0 treasures by env.d(loc, ("t", i)); stable on treasure id.
    std::vector<int> tres;
    for (int i = 0; i < env.N(); ++i)
        if (collected.count(i) == 0 && marg[i] > 0.0) tres.push_back(i);
    std::stable_sort(tres.begin(), tres.end(), [&](int a, int b) {
        return env.dist(loc.pt, env.treasure_pt(a)) < env.dist(loc.pt, env.treasure_pt(b));
    });
    if (static_cast<int>(tres.size()) > cfg_.cand_tre) tres.resize(cfg_.cand_tre);

    // order: detectors, then treasures, then TERMINATE (matches candidate_actions' list build).
    std::vector<Action> cands;
    cands.reserve(dets.size() + tres.size() + 1);
    for (int i : dets) cands.push_back(Action{ActionKind::Detector, i});
    for (int i : tres) cands.push_back(Action{ActionKind::Treasure, i});
    cands.push_back(terminate_action());
    return cands;
}

// ---- level-0: determinized base playout (mirrors nmcs.py's _playout) -----------------------------
double NMCSPolicy::playout(const Environment& env, const Loc& loc, const std::vector<uint32_t>& bw,
                           const std::set<int>& collected, double lam, WorldSource& src) const {
    (void)env;  // the level-0 leaf value is owned by the WorldSource (production: real GreedyPolicy
                // playout; scripted: the FIFO) — env stays in the signature to mirror _playout(env,…)
    return src.playout_value(loc, bw, collected, lam);
}

// ---- per-move evaluation (mirrors nmcs.py's _eval_move) -------------------------------------------
double NMCSPolicy::eval_move(const Environment& env, const Loc& loc, const std::vector<uint32_t>& bw,
                             const std::set<int>& collected, const Action& a, double lam, int level,
                             WorldSource& src) const {
    double tot = 0.0;
    for (int s = 0; s < cfg_.step_samples; ++s) {
        if (bw.empty()) {  // no world to sample: only the exit penalty remains (matches w is None)
            tot += -lam * env.exit_cost(loc.pt);
            continue;
        }
        uint32_t w = src.sample_world(bw);
        Loc nloc = loc;
        std::vector<uint32_t> nbw = bw;
        std::set<int> nc = collected;
        StepResult sr = env.apply(nloc, nbw, nc, a, w);
        double step = sr.reward - lam * sr.dt;
        double cont;
        if (level <= 1)
            cont = playout(env, nloc, nbw, nc, lam, src);
        else
            cont = search(env, nloc, nbw, nc, lam, level - 1, src).first;
        tot += step + cont;
    }
    return tot / static_cast<double>(cfg_.step_samples);
}

// ---- level-n search from a state (mirrors nmcs.py's _search) --------------------------------------
std::pair<double, Action> NMCSPolicy::search(const Environment& env, const Loc& loc,
                                             const std::vector<uint32_t>& bw,
                                             const std::set<int>& collected, double lam, int level,
                                             WorldSource& src) const {
    Loc cur_loc = loc;
    std::vector<uint32_t> cur_bw = bw;
    std::set<int> cur_coll = collected;
    double acc = 0.0;                  // λ-penalized reward accumulated along this line
    bool have_first = false;
    Action first_action = terminate_action();
    double best_seq_score = NEG_INF;   // memorized best complete line's score
    bool have_best_first = false;
    Action best_seq_first = terminate_action();

    auto close_line = [&](void) -> std::pair<double, Action> {
        double line_score = acc - lam * env.exit_cost(cur_loc.pt);
        if (line_score > best_seq_score) {
            best_seq_score = line_score;
            best_seq_first = have_first ? first_action : terminate_action();
            have_best_first = true;
        }
        return {best_seq_score, have_best_first ? best_seq_first : terminate_action()};
    };

    for (int step = 0; step < cfg_.max_steps; ++step) {
        std::vector<Action> cands = candidates(env, cur_loc, cur_bw, cur_coll);
        // If only TERMINATE is available, the line ends here (cands == [TERMINATE]).
        if (cands.size() == 1 && cands[0].kind == ActionKind::Terminate) {
            if (!have_first) {
                first_action = terminate_action();
                have_first = true;
            }
            return close_line();
        }

        // Evaluate each candidate by a nested (level-1) search / playout of its result; argmax.
        double best_q = NEG_INF;
        Action best_a = terminate_action();
        bool have_best_a = false;
        for (const Action& a : cands) {
            double q;
            if (a.kind == ActionKind::Terminate)
                q = acc - lam * env.exit_cost(cur_loc.pt);
            else
                q = acc + eval_move(env, cur_loc, cur_bw, cur_coll, a, lam, level, src);
            if (q > best_q) {  // strict >: first candidate wins a tie (matches `if q > best_q`)
                best_q = q;
                best_a = a;
                have_best_a = true;
            }
        }
        (void)have_best_a;  // cands always non-empty (TERMINATE present), so best_a is always set

        if (!have_first) {
            first_action = best_a;
            have_first = true;
        }

        // Memorize-and-replay: TERMINATE closes the line.
        if (best_a.kind == ActionKind::Terminate) return close_line();

        // Otherwise play best_a forward in a determinized world and continue the line.
        if (cur_bw.empty()) return close_line();  // w is None -> close (matches Python)
        uint32_t w = src.sample_world(cur_bw);
        StepResult sr = env.apply(cur_loc, cur_bw, cur_coll, best_a, w);
        acc += sr.reward - lam * sr.dt;

        // Track the best complete line: closing here (exit now) gives this score.
        double line_score = acc - lam * env.exit_cost(cur_loc.pt);
        if (line_score > best_seq_score) {
            best_seq_score = line_score;
            best_seq_first = have_first ? first_action : terminate_action();
            have_best_first = true;
        }
    }

    // Horizon hit: close out.
    return close_line();
}

// ---- construction + Policy interface --------------------------------------------------------------
NMCSPolicy::NMCSPolicy(const NMCSConfig& cfg, const Policy* base)
    : cfg_(cfg), base_(base ? base : &default_base_) {}

namespace {
// The production world source: a real determinized GreedyPolicy playout off a single std::mt19937_64.
// `sample_world` draws uniformly from the belief (mirrors env.sample_world -> rng.choice(bw));
// `playout_value` is the mean-over-playout_samples GreedyPolicy base_value (mirrors nmcs.py _playout).
class RngWorldSource final : public WorldSource {
  public:
    RngWorldSource(const Environment& env, const Policy& base, int playout_samples,
                   std::mt19937_64& rng)
        : env_(env), base_(base), ps_(playout_samples), rng_(rng) {}

    uint32_t sample_world(const std::vector<uint32_t>& bw) override {
        std::uniform_int_distribution<size_t> pick(0, bw.size() - 1);
        return bw[pick(rng_)];
    }

    double playout_value(const Loc& loc, const std::vector<uint32_t>& bw,
                         const std::set<int>& collected, double lam) override {
        if (bw.empty()) return -lam * env_.exit_cost(loc.pt);  // matches nmcs.py len(bw)==0 branch
        double tot = 0.0;
        for (int s = 0; s < ps_; ++s) {
            uint32_t w = sample_world(bw);
            tot += base_value(env_, base_, loc, bw, collected, w, lam);
        }
        return tot / static_cast<double>(ps_);
    }

  private:
    const Environment& env_;
    const Policy& base_;
    int ps_;
    std::mt19937_64& rng_;
};
}  // namespace

Action NMCSPolicy::decide(const Environment& env, const Loc& loc, const std::vector<uint32_t>& bw,
                          const std::set<int>& collected, double lam, std::mt19937_64& rng) const {
    std::vector<Action> cands = candidates(env, loc, bw, collected);
    if (cands.size() == 1 && cands[0].kind == ActionKind::Terminate) return terminate_action();
    RngWorldSource src(env, *base_, cfg_.playout_samples, rng);
    int level = std::max(1, cfg_.level);  // mirrors decide's max(1, self.level)
    auto [score, first_action] = search(env, loc, bw, collected, lam, level, src);
    (void)score;
    return first_action;
}

}  // namespace chocofarm
