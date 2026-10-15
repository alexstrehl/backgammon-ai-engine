// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Alex Strehl
//
// Links Open Sage (bgsage v2.0.20260907, MPL-2.0).
//
// sage_bench.cpp -- Open Sage (bgsage v2.0.20260907) fixed-pool loops: its
// decision functions called directly, each item timed with CLOCK_MONOTONIC.
// Plies are PureTD's numbering (0, 1); Open Sage's is plies + 1.
//
//   cubeful checker  the C API's operation 0 (mobile.cpp) without its JSON
//                    reply: possible_boards, every candidate scored (0-ply
//                    evaluate_probs + cl2cf, 1-ply cubeful_probs_and_
//                    equity_nply), first by (equity, cubeless equity, board);
//                    1-ply items start on cleared caches (untimed), as after
//                    the 1-ply cube decision that precedes a roll
//   cube             operation 1 without its JSON reply and without the
//                    second evaluate_probs it runs for the reply; after a
//                    1-ply decision the C API's cache clear runs, untimed
//   cubeless checker possible_boards + MultiPlyStrategy::best_move_index,
//                    filter 5 / 0.08 (bgbot_cpp.create_multipy), with the
//                    bearoff database the analyzer attaches

#include "bgbot/bearoff.h"
#include "bgbot/board.h"
#include "bgbot/cube.h"
#include "bgbot/moves.h"
#include "bgbot/multipy.h"
#include "bgbot/neural_net.h"
#include "bgbot/pubeval.h"

#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace bgbot;

namespace {
std::string last_error;

double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
double cl_equity(double w, double gw, double bw, double gl, double bl) {
    return 2 * w - 1 + gw - gl + bw - bl;
}
}  // namespace

struct SageBench {
    BearoffDB db;
    std::shared_ptr<Strategy> strategy;       // the C API's engine
    std::shared_ptr<MultiPlyStrategy> cl[2];  // cubeless 0-ply, 1-ply
    std::vector<Board> cands;
    const MoveFilter filter{5, 0.08f};

    int checker_cubeful(const Board& board, int d1, int d2, const CubeInfo& ci, int plies) {
        possible_boards(board, d1, d2, cands);
        int best = 0;
        double be = 0, bc = 0;
        for (int i = 0; i < (int)cands.size(); ++i) {
            const Board& b = cands[i];
            double eq, cle;
            if (plies == 0) {
                auto p = strategy->evaluate_probs(b, board);
                auto [pp, op] = pip_counts(b);
                eq = cl2cf(p, ci, cube_efficiency(p, is_race(b), pp, op));
                cle = cl_equity(p[0], p[1], p[2], p[3], p[4]);
            } else {
                auto v = cubeful_probs_and_equity_nply(flip(b), flip_cube_perspective(ci),
                                                       *strategy, 2, filter, 1, nullptr, &board);
                eq = -v.equity;  // v.probs are the opponent's: invert
                cle = cl_equity(1.0 - v.probs[0], v.probs[3], v.probs[4], v.probs[1], v.probs[2]);
            }
            if (i == 0 || eq > be || (eq == be && (cle > bc || (cle == bc && b < cands[best])))) {
                best = i; be = eq; bc = cle;
            }
        }
        return best;
    }

    CubeDecision cube(const Board& board, const CubeInfo& ci, int plies) {
        if (plies == 0) {
            auto p = invert_probs(strategy->evaluate_probs(flip(board), is_race(board)));
            auto [pp, op] = pip_counts(board);
            return cube_decision_1ply(p, ci, cube_efficiency(p, is_race(board), pp, op));
        }
        PubEval prefilter;
        return cube_decision_nply(board, ci, *strategy, 2, filter, 1, &prefilter);
    }
};

extern "C" {

const char* sb_last_error() { return last_error.c_str(); }

SageBench* sb_create(const char* type, const char* const* paths, const int* hidden,
                     int count, const char* bearoff) {
    try {
        auto e = std::make_unique<SageBench>();
        std::vector<std::string> p(paths, paths + count);
        std::vector<int> h(hidden, hidden + count);
        const std::string t(type);
        auto make = [&]() -> std::shared_ptr<Strategy> {
            if (t == "backgame_pair_phased") return std::make_shared<BackgameAwarePairStrategy>(p, h, true);
            if (t == "backgame_pair") return std::make_shared<BackgameAwarePairStrategy>(p, h);
            if (t == "pair") return std::make_shared<GamePlanPairStrategy>(p, h);
            if (t == "5nn") return std::make_shared<GamePlanStrategy>(p, h);
            throw std::invalid_argument("unsupported model strategy");
        };
        if (!e->db.load(bearoff)) throw std::runtime_error("bearoff asset unavailable");
        e->strategy = std::make_shared<BearoffStrategy>(make(), &e->db);   // bgsage_mobile_create
        static auto pubeval = std::make_shared<PubEval>();
        for (int k = 0; k < 2; ++k) {  // create_multipy(..., parallel_evaluate=False)
            e->cl[k] = std::make_shared<MultiPlyStrategy>(make(), k + 1, e->filter, false, false, 0);
            e->cl[k]->set_move_prefilter(pubeval);
            e->cl[k]->set_bearoff_db(&e->db);  // as BgBotAnalyzer does
        }
        return e.release();
    } catch (const std::exception& ex) { last_error = ex.what(); return nullptr; }
}

void sb_destroy(SageBench* e) { delete e; }

// One loop for the three item kinds. kind 0 cubeful checker, 1 cube, 2
// cubeless checker. boards n x 26 (mover's frame), dice n x 2 (checker),
// cube value / owner (0 centred, 1 mover, 2 opponent) n each (cubeful).
// out_seconds n; out n x 26 chosen boards, or n cube decisions
// (1 double | 2 take).
int sb_bench(SageBench* e, int kind, int n, const int* boards, const int* dice,
             const int* cube_value, const int* cube_owner, int plies,
             double* out_seconds, int* out) {
    try {
        for (int i = 0; i < n; ++i) {
            Board b;
            std::copy(boards + 26 * i, boards + 26 * i + 26, b.begin());
            CubeInfo ci{};
            if (kind != 2)  // bgsage_mobile_analyze's money CubeInfo, Jacoby on
                ci = CubeInfo{cube_value[i], static_cast<CubeOwner>(cube_owner[i]),
                              {0, 0, false}, -1.0f, true, false};
            if (kind == 0 && plies == 1) e->cl[1]->clear_cache();  // untimed
            const double t0 = now_s();
            int k = 0;
            CubeDecision cd{};
            if (kind == 0) k = e->checker_cubeful(b, dice[2 * i], dice[2 * i + 1], ci, plies);
            else if (kind == 1) cd = e->cube(b, ci, plies);
            else {
                possible_boards(b, dice[2 * i], dice[2 * i + 1], e->cands);
                k = e->cl[plies]->best_move_index(e->cands, b);
            }
            out_seconds[i] = now_s() - t0;
            if (kind == 1 && plies == 1) e->cl[1]->clear_cache();  // the C API's
                // multipy.clear_cache() (position and cubeful caches), untimed
            if (kind == 1) out[i] = (cd.should_double ? 1 : 0) | (cd.should_take ? 2 : 0);
            else std::copy(e->cands[k].begin(), e->cands[k].end(), out + 26 * i);
        }
        return 0;
    } catch (const std::exception& ex) { last_error = ex.what(); return -1; }
}

}  // extern "C"
