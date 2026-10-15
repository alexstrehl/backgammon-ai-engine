// hh_shim.cpp - flat C API over the HedgeHog engine for ctypes.
// HedgeHog (hedgehog-public) is MIT, Copyright (c) 2026 Eran Lambooij; it is
// not included here, build_shim.sh compiles against a separate checkout.
//
// Conventions (all documented in hedgehog-public sources):
//   * BoardState: 16 bytes. Pip 0 = White bar, 1..24 board, 25 = Black bar.
//     White moves 1->24 (home 19-24), Black 24->1 (home 1-6). Borne-off
//     checkers are simply absent from the board (position >= 26 in
//     bg_set_state input = off).
//   * BG_BLACK = 0, BG_WHITE = 1.
//   * eval_ensemble(state, color, ctx) evaluates with `color` ON ROLL,
//     probabilities from color's perspective (community-search convention:
//     leaves are evaluated with stm and flipped to root POV).
//   * CubeOwner: 0 = centered, 1 = owned by White, 2 = owned by Black.
//
// Move selection here follows the community-search convention (NOT the
// unflipped cube_search.h cubeful_select_best_move): each successor (opponent
// to roll) is evaluated with eval_ensemble(succ, opp) and flipped to the
// mover's POV, then ranked by Janowski cubeful equity for the mover's cube
// ownership. depth=0 gives the engine's 0-ply; depth>=1 runs the community
// expectiminimax below each successor. At depth 0 hh_best_move and
// hh_best_move_cubeless evaluate the successors in one batch
// (eval_ensemble_successors: incremental accumulator from the parent);
// hh_pick_best* (the head-to-head agent) evaluate each candidate alone.

#include <cstdint>
#include <cstring>
#include <ctime>

#include "src/backgammon.hpp"
#include "src/ogid.h"
#include "src/eval/eval_ensemble.h"
#include "src/eval/equity.h"
#include "src/search/search_community.hpp"
#include "src/explore/explore_delta.h"
#include "src/cube/cube_types.h"
#include "src/cube/cube_decision.h"
#include "src/cube/cubeful_equity.h"

static inline BoardState board_from_bytes(const uint8_t b[16]) {
    BoardState s;
    std::memcpy(&s, b, 16);
    return s;
}
static inline void board_to_bytes(BoardState s, uint8_t out[16]) {
    std::memcpy(out, &s, 16);
}

static inline CubeState make_cube(int value, int owner) {
    CubeState c;
    c.value = (uint8_t)value;
    c.owner = (CubeOwner)owner;
    return c;
}

// Jacoby applies only with a centered cube, so it changes move ranking and
// double decisions but never a take. Default 0 = the Jacoby-unaware behaviour
// of the pinned engine; the agent sets it per game from match_state.jacoby.
static int g_jacoby = 0;
extern "C" void hh_set_jacoby(int on) { g_jacoby = on ? 1 : 0; }

// Evaluate `state` with `color` about to roll, from color's POV.
// depth 0 = static NN eval; depth d>0 = community expectiminimax.
// `cube` may be NULL (cubeless interior); when given, the search carries
// the cube down so interior replies are ranked cubefully too.
static EvalResult eval_pov(NNEnsembleContext* ctx, BoardState state,
                           int color, int depth, const CubeState* cube) {
    if (depth <= 0) return eval_ensemble(state, color, ctx);
    if (cube)
        return ogx_community::community_search_ensemble_cubeful(
            ctx, state, color, color, depth, nullptr,
            cube, DEFAULT_CUBE_EFFICIENCY);
    return ogx_community::community_search_ensemble(
        ctx, state, color, color, depth, nullptr);
}

extern "C" {

void* hh_create(const char* model_path) {
    return (void*)ensemble_context_create(model_path, nullptr);
}

void hh_free(void* ctx) {
    if (ctx) ensemble_context_free((NNEnsembleContext*)ctx);
}

void hh_initial_board(uint8_t out[16]) {
    board_to_bytes(bg_initial_state(), out);
}

// positions: 15 entries per color; 0..25 = pip (0 White bar, 25 Black bar),
// >=26 = borne off.
void hh_board_from_pips(const uint8_t white_pos[15], const uint8_t black_pos[15],
                        uint8_t out[16]) {
    uint8_t w[15], b[15];
    std::memcpy(w, white_pos, 15);
    std::memcpy(b, black_pos, 15);
    board_to_bytes(bg_set_state(w, b), out);
}

void hh_decode(const uint8_t board[16], uint8_t pips[26], uint8_t colors[26],
               int* white_off, int* black_off) {
    BoardDecode d;
    bg_decode(board_from_bytes(board), &d);
    std::memcpy(pips, d.pips, 26);
    std::memcpy(colors, d.colors, 26);
    *white_off = d.white_off;
    *black_off = d.black_off;
}

int hh_game_over(const uint8_t board[16]) {
    return bg_is_game_over(board_from_bytes(board));
}
int hh_winner(const uint8_t board[16]) {
    return bg_get_winner(board_from_bytes(board));
}
int hh_points(const uint8_t board[16], int winner) {
    return bg_get_points_won(board_from_bytes(board), (uint8_t)winner);
}

// Raw eval: color on roll, probs from color's POV. out5 = win,gw,bgw,gl,bgl.
void hh_eval(void* ctx, const uint8_t board[16], int color, float out5[5]) {
    EvalResult e = eval_ensemble(board_from_bytes(board), color,
                                 (NNEnsembleContext*)ctx);
    out5[0] = e.win; out5[1] = e.gammon_win; out5[2] = e.bg_win;
    out5[3] = e.gammon_loss; out5[4] = e.bg_loss;
}

// Number of legal successor states for color with dice d1,d2 (0 = must pass).
int hh_num_moves(const uint8_t board[16], int color, int d1, int d2) {
    static thread_local BoardState succ[MAX_SUCCESSOR_STATES];
    return explore_delta(board_from_bytes(board), color,
                         (uint8_t)d1, (uint8_t)d2, succ);
}

// List all legal successor boards (16 bytes each) into out (capacity cap).
// Returns total number of successors (may exceed cap; only cap written).
int hh_list_moves(const uint8_t board[16], int color, int d1, int d2,
                  uint8_t* out, int cap) {
    static thread_local BoardState succ[MAX_SUCCESSOR_STATES];
    int n = explore_delta(board_from_bytes(board), color,
                          (uint8_t)d1, (uint8_t)d2, succ);
    int w = n < cap ? n : cap;
    for (int i = 0; i < w; i++) board_to_bytes(succ[i], out + 16 * i);
    return n;
}

// Choose best checker move for `color` rolling (d1,d2), cubeful ranking.
// Returns number of legal successors (0 = pass; out = input board).
// depth: 0 = 0-ply (evaluate successors statically), 1/2 = deeper.
int hh_best_move(void* vctx, const uint8_t board[16], int color,
                 int d1, int d2, int cube_value, int cube_owner,
                 int depth, uint8_t out[16]) {
    NNEnsembleContext* ctx = (NNEnsembleContext*)vctx;
    BoardState state = board_from_bytes(board);
    static thread_local BoardState succ[MAX_SUCCESSOR_STATES];
    int n = explore_delta(state, color, (uint8_t)d1, (uint8_t)d2, succ);
    if (n <= 0) {
        board_to_bytes(state, out);
        return 0;
    }
    CubeState cube = make_cube(cube_value, cube_owner);
    CubeOwnership own = cube_ownership_for_player(&cube, (uint8_t)color);
    int opp = 1 - color;
    static thread_local EvalResult ev[MAX_SUCCESSOR_STATES];
    if (depth <= 0) eval_ensemble_successors(ctx, state, succ, n, opp, ev);
    int best_i = 0;
    float best_eq = -1e30f;
    for (int i = 0; i < n; i++) {
        EvalResult child;
        if (depth <= 0) {
            // opponent on roll in successor; flip to mover POV
            child = eval_flip(&ev[i]);
        } else {
            // Cubeful interior (see hh_pick_best): carry the cube down.
            child = ogx_community::community_search_ensemble_cubeful(
                ctx, succ[i], color, opp, depth, nullptr,
                &cube, DEFAULT_CUBE_EFFICIENCY);
        }
        float eq = cubeful_equity_by_ownership_jacoby(&child, DEFAULT_CUBE_EFFICIENCY, own, g_jacoby);
        if (eq > best_eq) { best_eq = eq; best_i = i; }
    }
    board_to_bytes(succ[best_i], out);
    return n;
}

// Cubeless twin of hh_best_move: native movegen + successor evals, ranked by
// plain cubeless money equity (eval_equity, as hh_pick_best_cubeless). Same
// work per successor as hh_best_move; only the O(1) final ranking differs.
int hh_best_move_cubeless(void* vctx, const uint8_t board[16], int color,
                          int d1, int d2, int depth, uint8_t out[16]) {
    NNEnsembleContext* ctx = (NNEnsembleContext*)vctx;
    BoardState state = board_from_bytes(board);
    static thread_local BoardState succ[MAX_SUCCESSOR_STATES];
    int n = explore_delta(state, color, (uint8_t)d1, (uint8_t)d2, succ);
    if (n <= 0) { board_to_bytes(state, out); return 0; }
    int opp = 1 - color;
    static thread_local EvalResult ev[MAX_SUCCESSOR_STATES];
    if (depth <= 0) eval_ensemble_successors(ctx, state, succ, n, opp, ev);
    int best_i = 0;
    float best_eq = -1e30f;
    for (int i = 0; i < n; i++) {
        EvalResult child;
        if (depth <= 0) {
            child = eval_flip(&ev[i]);
        } else {
            child = ogx_community::community_search_ensemble(
                ctx, succ[i], color, opp, depth, nullptr);
        }
        float eq = eval_equity(&child);
        if (eq > best_eq) { best_eq = eq; best_i = i; }
    }
    board_to_bytes(succ[best_i], out);
    return n;
}

// Pick the best successor for `color` among caller-supplied candidate boards
// (n boards, 16 bytes each; successors AFTER color's move, opponent to roll).
// Ranking identical to hh_best_move; legality is the caller's business.
// Returns the index of the best board, or -1 if n <= 0.
int hh_pick_best(void* vctx, const uint8_t* boards, int n, int color,
                 int cube_value, int cube_owner, int depth) {
    if (n <= 0) return -1;
    NNEnsembleContext* ctx = (NNEnsembleContext*)vctx;
    CubeState cube = make_cube(cube_value, cube_owner);
    CubeOwnership own = cube_ownership_for_player(&cube, (uint8_t)color);
    int opp = 1 - color;
    int best_i = 0;
    float best_eq = -1e30f;
    for (int i = 0; i < n; i++) {
        BoardState s = board_from_bytes(boards + 16 * i);
        EvalResult child;
        if (depth <= 0) {
            EvalResult e = eval_ensemble(s, opp, ctx);
            child = eval_flip(&e);
        } else {
            // Cubeful interior: the search carries the cube down so the
            // opponent's replies are ranked cubefully, not only the root.
            child = ogx_community::community_search_ensemble_cubeful(
                ctx, s, color, opp, depth, nullptr,
                &cube, DEFAULT_CUBE_EFFICIENCY);
        }
        float eq = cubeful_equity_by_ownership_jacoby(&child, DEFAULT_CUBE_EFFICIENCY, own, g_jacoby);
        if (eq > best_eq) { best_eq = eq; best_i = i; }
    }
    return best_i;
}

// Cubeless variant of hh_pick_best: ranks caller-supplied successor boards by
// HedgeHog's cubeless money equity eval_equity() (2w-1 + (gw-gl) +
// (bgw-bgl), nested-mode aware). No Janowski layer and no cube state; this is
// the ranking HedgeHog's community search uses when `cube` is null.
// depth as in hh_pick_best.
int hh_pick_best_cubeless(void* vctx, const uint8_t* boards, int n, int color,
                          int depth) {
    if (n <= 0) return -1;
    NNEnsembleContext* ctx = (NNEnsembleContext*)vctx;
    int opp = 1 - color;
    int best_i = 0;
    float best_eq = -1e30f;
    for (int i = 0; i < n; i++) {
        BoardState s = board_from_bytes(boards + 16 * i);
        EvalResult child;
        if (depth <= 0) {
            EvalResult e = eval_ensemble(s, opp, ctx);
            child = eval_flip(&e);
        } else {
            child = ogx_community::community_search_ensemble(
                ctx, s, color, opp, depth, nullptr);
        }
        float eq = eval_equity(&child);
        if (eq > best_eq) { best_eq = eq; best_i = i; }
    }
    return best_i;
}

// Pre-roll double decision for `color` (money game).
// Returns CubeAction enum: 0 NO_DOUBLE, 1 DOUBLE_TAKE, 2 DOUBLE_PASS,
// 3 TOO_GOOD, 4 REDOUBLE_TAKE, 5 REDOUBLE_PASS. -1 if cannot double.
// equities_out (optional, may be NULL): nd, dt, dp.
int hh_double_action(void* vctx, const uint8_t board[16], int color,
                     int cube_value, int cube_owner, int depth,
                     float* equities_out) {
    NNEnsembleContext* ctx = (NNEnsembleContext*)vctx;
    CubeState cube = make_cube(cube_value, cube_owner);
    if (!cube_can_double(&cube, (uint8_t)color, CUBE_MAX_VALUE)) return -1;
    EvalResult e = eval_pov(ctx, board_from_bytes(board), color, depth, &cube);
    CubeDecision d = cube_evaluate_double_jacoby(&e, DEFAULT_CUBE_EFFICIENCY,
                                          &cube, (uint8_t)color, g_jacoby);
    if (equities_out) {
        equities_out[0] = d.no_double_equity;
        equities_out[1] = d.double_take_equity;
        equities_out[2] = d.double_pass_equity;
    }
    return (int)d.action;
}

// Take/pass when doubled. `doubler` is on roll (pre-roll). Decision is for
// the other player. Returns 1 = take, 0 = pass.
int hh_take(void* vctx, const uint8_t board[16], int doubler,
            int cube_value, int cube_owner, int depth) {
    NNEnsembleContext* ctx = (NNEnsembleContext*)vctx;
    CubeState cube = make_cube(cube_value, cube_owner);
    EvalResult e = eval_pov(ctx, board_from_bytes(board), doubler, depth,
                            &cube);
    EvalResult mine = eval_flip(&e);
    return cube_should_take(&mine, DEFAULT_CUBE_EFFICIENCY, &cube);
}

// OGID parse helper (for cross-checks): returns 0 on success.
int hh_parse_ogid(const char* ogid, uint8_t out[16], int* color_to_move) {
    BoardState s; int c;
    if (ogid_parse(ogid, &s, &c) != 0) return -1;
    board_to_bytes(s, out);
    *color_to_move = c;
    return 0;
}

// Native benchmark loops: per-item timing of the decision entry points
// above with no Python/ctypes cost per call. Each item is one call of
// hh_best_move_cubeless / hh_best_move / hh_double_action, timed with
// CLOCK_MONOTONIC around that call alone. Inputs are caller-prepared
// arrays (boards: n x 16 bytes; turns, dice (n x 2), cube_values, owners:
// int32). Single thread; the Jacoby flag is whatever hh_set_jacoby last
// set.
// out_secs[i] = seconds for item i. Checker loops write the chosen board
// (n x 16 bytes) to out_choice; the cube loop writes the CubeAction (or -1)
// to out_decision.

static inline double mono_secs() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

void hh_bench_checker_cubeless(void* ctx, const uint8_t* boards,
                               const int32_t* turns, const int32_t* dice,
                               int n, int depth, double* out_secs,
                               uint8_t* out_choice) {
    for (int i = 0; i < n; i++) {
        const uint8_t* b = boards + 16 * i;
        int c = turns[i], d1 = dice[2 * i], d2 = dice[2 * i + 1];
        uint8_t* o = out_choice + 16 * i;
        double t0 = mono_secs();
        hh_best_move_cubeless(ctx, b, c, d1, d2, depth, o);
        out_secs[i] = mono_secs() - t0;
    }
}

void hh_bench_checker_cubeful(void* ctx, const uint8_t* boards,
                              const int32_t* turns, const int32_t* dice,
                              const int32_t* cube_values,
                              const int32_t* owners, int n, int depth,
                              double* out_secs, uint8_t* out_choice) {
    for (int i = 0; i < n; i++) {
        const uint8_t* b = boards + 16 * i;
        int c = turns[i], d1 = dice[2 * i], d2 = dice[2 * i + 1];
        int cv = cube_values[i], ow = owners[i];
        uint8_t* o = out_choice + 16 * i;
        double t0 = mono_secs();
        hh_best_move(ctx, b, c, d1, d2, cv, ow, depth, o);
        out_secs[i] = mono_secs() - t0;
    }
}

void hh_bench_cube(void* ctx, const uint8_t* boards, const int32_t* turns,
                   const int32_t* cube_values, const int32_t* owners, int n,
                   int depth, double* out_secs, int32_t* out_decision) {
    float eqs[3];
    for (int i = 0; i < n; i++) {
        const uint8_t* b = boards + 16 * i;
        int c = turns[i], cv = cube_values[i], ow = owners[i];
        double t0 = mono_secs();
        int a = hh_double_action(ctx, b, c, cv, ow, depth, eqs);
        out_secs[i] = mono_secs() - t0;
        out_decision[i] = a;
    }
}

// Cost of the timing bracket itself (two clock reads, nothing between):
// n samples into out_secs, to subtract or report as the floor.
void hh_bench_clock_floor(int n, double* out_secs) {
    for (int i = 0; i < n; i++) {
        double t0 = mono_secs();
        out_secs[i] = mono_secs() - t0;
    }
}

} // extern "C"
