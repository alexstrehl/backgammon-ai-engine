/*
 * puretd_collect.c -- batch-training self-play collector (bgc_collect).
 *
 * Port of trainer.collect_episode (cubeless) and
 * trainer._collect_episode_cubeful_money with a TDAgent at
 * boltzmann_temp 0 (argmin play), oneply_acting off, and the
 * perspective196 / cubeful_perspective196 encoders, and of
 * train_prob5._collect_one_episode (prob5 nets, perspective196): per
 * sample the row the Python collector stacks and the target it appends,
 * in its order, episode after episode on one random.Random(seed) stream
 * (the _collect_worker loops). The Python functions are the reference.
 *
 * Every value comes from the core (puretd.c): bgc_best_move (0-ply play,
 * its out_value = the Python bootstrap), bgc_value, value1 (1-ply
 * targets), bgc_offer_double / bgc_respond_to_double (cube phase);
 * prob5: bgc_i_best_move_prob5 / bgc_value_prob5 (0-ply) and vec1_prob5
 * through bgc_i_value1_play (1-ply). No heap allocation here: rows go
 * straight into the caller's buffers, the engine's own scratch does the
 * rest (it grows to its high-water mark in the first episodes, then is
 * reused).
 *
 * The game mode is the kind's: PROB = DMPMode, EQUITY =
 * CubelessMoneyMode, CUBEFUL = CubefulMoneyMode, PROB5 = train_prob5
 * (cubeless money, 5-vector targets). DMPMode with an equity net
 * (terminal target 1, not 1/2/3) has no kind here. Rows are written
 * with the engine's encoder (bgc_i_encoder: its base encoder, then its
 * cube block). Only encoders whose base encoder's table entry says
 * collect (puretd.c BGC_BASES: perspective196, and so
 * cubeful_perspective196) are collected; another one returns
 * BGC_COLLECT_UNSUPPORTED, and the trainers route it to the Python
 * collector (trainer.C_COLLECT_ENCODERS, the same list).
 */

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "puretd.h"
#include "puretd_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>

/* Episode outcomes. */
#define EP_KEPT 0
#define EP_DISCARDED 1          /* max_plies exceeded */
#define EP_ERROR (-1)
#define EP_FULL (-2)            /* caller's buffer exhausted */

/* The caller's buffers and the next free row. A row's target is
   tgt[n * out_dim ..]: out_dim 1, or 5 for PROB5. */
typedef struct {
    float *enc, *tgt;
    long cap, n;
    int in_size, out_dim;
    const bgc_encoder *en;  /* the engine's encoder: the row layout */
} sink;

/* Appends the sample row of s (base encoding; cubeful: the cube columns
   persp / is_cube_action) and returns its index, -1 when full. */
static long put_row(sink *k, const BoardState *s, int persp, int ica) {
    if (k->n >= k->cap) return -1;
    PROF_T0(t0);
    float *row = k->enc + (size_t)k->n * k->in_size;
    k->en->base->encode(s, row);
    if (k->en->cube->ncube) cube_cols(row + k->en->base->n, persp, ica);
    for (int j = 0; j < k->out_dim; j++) k->tgt[(size_t)k->n * k->out_dim + j] = NAN;
    PROF_ADD(t0, BGC_PT_ENCODE, 1);
    return k->n++;
}

static void roll(bgc_rng *rng, int *d1, int *d2) {
    *d1 = bgc_rng_randint(rng, 1, 6);              /* dice = (rng.randint(1, 6), */
    *d2 = bgc_rng_randint(rng, 1, 6);              /*         rng.randint(1, 6)) */
}

/* collect_episode, cubeless branch. dmp (PROB) or cubeless money (EQUITY):
   the mode is the kind's. */
static int episode_cubeless(bgc_engine *e, const bgc_collect_opts *o, bgc_rng *rng, sink *k) {
    int prob = bgc_kind(e) == BGC_KIND_PROB;
    BoardState s, next;
    int d1, d2, n_dice = 0, plies = 0;
    bgc_i_opening_roll(rng, &s, &d1, &d2, &n_dice);
    while (!board_is_game_over(&s)) {
        if (o->max_plies > 0 && plies >= o->max_plies) return EP_DISCARDED;
        plies++;
        long r = put_row(k, &s, 0, 0);
        if (r < 0) return EP_FULL;
        if (o->oneply) {
            /* value_oneply_checker(state): the 21-roll average, terminal
               replies folded in. Play at 0-ply (bgc_best_move's choice and
               score, from the same expansion), the target unused. */
            double v1;
            int idx = bgc_i_value1_play(e, &s, NULL, d1, d2, &v1, NULL, &next, NULL);
            if (idx == -2) return EP_ERROR;
            k->tgt[r] = (float)v1;
            if (idx >= 0) s = next;
            else board_switch_turn(&s);
            /* No break on a game-ending play: Python draws the next dice
               first, then leaves the loop on is_episode_over. */
        } else {
            float val;
            int idx;
            {
                PROF_ENTER(BGC_PC_SEL);
                idx = bgc_best_move(e, &s, d1, d2, NULL, 0, &next, NULL, &val);
                PROF_LEAVE(1);
            }
            if (idx == -2) return EP_ERROR;
            if (idx >= 0) {
                if (board_is_game_over(&next)) {
                    /* terminal_target(make_terminal_outcome): dmp 1.0,
                       cubeless money game_result. Episode ends, no dice. */
                    k->tgt[r] = prob ? 1.0f : (float)bgc_i_game_result(&next);
                    return EP_KEPT;
                }
                k->tgt[r] = val;   /* bootstrap = _flip_value(V_opp(chosen)) */
                s = next;
            } else {
                /* Forced pass: bootstrap_target(switch_turn(state)). */
                board_switch_turn(&s);
                double v;
                {
                    PROF_ENTER(BGC_PC_VAL);
                    v = (double)bgc_value(e, &s, NULL, 0);
                    PROF_LEAVE(1);
                }
                k->tgt[r] = (float)(prob ? 1.0 - v : -v);
            }
        }
        roll(rng, &d1, &d2);
    }
    return EP_KEPT;
}

/* Opponent's 5-vector -> the mover's target (train_prob5._flip_vec). */
static void flip5(const float *v, float *t) {
    t[0] = 1.0f - v[0];
    t[1] = v[3];
    t[2] = v[4];
    t[3] = v[1];
    t[4] = v[2];
}

/* train_prob5._collect_one_episode: one sample per turn (forced passes
   included), target = the mover's 5-vector. train_prob5 has no episode
   bound; max_plies (<= 0 there) discards as in the other kinds. */
static int episode_prob5(bgc_engine *e, const bgc_collect_opts *o, bgc_rng *rng, sink *k) {
    BoardState s, next;
    int d1, d2, n_dice = 0, plies = 0;
    bgc_i_opening_roll(rng, &s, &d1, &d2, &n_dice);
    while (!board_is_game_over(&s)) {
        if (o->max_plies > 0 && plies >= o->max_plies) return EP_DISCARDED;
        plies++;
        long r = put_row(k, &s, 0, 0);
        if (r < 0) return EP_FULL;
        float *t = k->tgt + (size_t)r * 5;
        if (o->oneply) {
            /* _oneply_target_vec (expand_21 path): the 21-roll average of
               the per-roll best reply's flipped vector, terminals folded
               (vec1_prob5). Play at 0-ply by the opponent-view equity,
               from the same expansion. */
            double v1;
            int idx = bgc_i_value1_play(e, &s, NULL, d1, d2, &v1, t, &next, NULL);
            if (idx == -2) return EP_ERROR;
            if (idx >= 0) s = next;
            else board_switch_turn(&s);
            /* No break on a game-ending play: dice first, then the loop
               ends on is_game_over (as the cubeless 1-ply path). */
        } else {
            /* argmin of the opponent-view clamped equity, a game-ending
               successor masked to -game_result (ProbAgent._choose_0ply). */
            float p[5];
            int idx;
            {
                PROF_ENTER(BGC_PC_SEL);
                idx = bgc_i_best_move_prob5(e, &s, d1, d2, &next, p);
                PROF_LEAVE(1);
            }
            if (idx == -2) return EP_ERROR;
            if (idx >= 0) {
                if (board_is_game_over(&next)) {
                    /* _terminal_target: [1, gammon, backgammon, 0, 0].
                       Episode ends, no dice. */
                    const int gr = bgc_i_game_result(&next);
                    t[0] = 1.0f;
                    t[1] = gr >= 2 ? 1.0f : 0.0f;
                    t[2] = gr >= 3 ? 1.0f : 0.0f;
                    t[3] = t[4] = 0.0f;
                    return EP_KEPT;
                }
                flip5(p, t);           /* _flip_vec(opp_probs[best]) */
                s = next;
            } else {
                /* Forced pass: _flip_vec of the opponent's view. */
                board_switch_turn(&s);
                int rc;
                {
                    PROF_ENTER(BGC_PC_VAL);
                    rc = bgc_value_prob5(e, &s, p, NULL);
                    PROF_LEAVE(1);
                }
                if (rc < 0) return EP_ERROR;
                flip5(p, t);
            }
        }
        roll(rng, &d1, &d2);
    }
    return EP_KEPT;
}

/* _collect_episode_cubeful_money. Targets are per-unit equity. */
static int episode_cubeful(bgc_engine *e, const bgc_collect_opts *o, bgc_rng *rng, sink *k) {
    BoardState s, next;
    int d1, d2, n_dice = 0, plies = 0, opening = 1;
    bgc_cube cube = {0, 1, o->jacoby ? 1 : 0};     /* mode.initial_match_state() */
    bgc_i_opening_roll(rng, &s, &d1, &d2, &n_dice);
    while (!board_is_game_over(&s)) {
        if (o->max_plies > 0 && plies >= o->max_plies) return EP_DISCARDED;
        plies++;
        int player = s.turn;

        /* Cube phase (_resolve_cube_decision), never on the opening roll.
           0-ply at both target plies: the 1-ply collector's
           _cube_phase_oneply_gen evaluates the same two rows as
           offer_double at 0-ply (encode(state, pre_persp, ica 0) and
           encode(state, THEIRS, ica 0) in one forward, v_no_double =
           V[0], v_double_take = 2 * V[1], the same _money_cube_decision,
           take iff v_double_take <= 1), so one bgc_offer_double(plies 0)
           serves both. */
        if (!opening && can_offer(cube.owner, player)) {
            int pre = cube_persp(cube.owner, player);
            bgc_cube_eval ce;
            int dbl;
            {
                PROF_ENTER(BGC_PC_CUBE);
                dbl = bgc_offer_double(e, &s, &cube, 0, &ce);
                PROF_LEAVE(1);
            }
            if (dbl < 0) return EP_ERROR;
            int take = 0;
            if (dbl) {
                take = bgc_respond_to_double(e, &s, &cube, 0, &ce);   /* v_dt <= 1 */
                if (take < 0) return EP_ERROR;
            }
            {
                long r = put_row(k, &s, pre, 1);
                if (r < 0) return EP_FULL;
                /* no double: v_no_double; double/take: v_double_take;
                   double/pass: +1 */
                k->tgt[r] = !dbl ? ce.v_no_double : take ? ce.v_double_take : 1.0f;
            }
            if (dbl && !take) return EP_KEPT;                /* dropped: episode ends */
            if (dbl) {                                       /* after_take(player) */
                cube.owner = (1 - player) + 1;
                cube.value *= 2;
            }
        }

        /* Checker phase: encode(state, post_persp, ica 0). */
        long r = put_row(k, &s, cube_persp(cube.owner, player), 0);
        if (r < 0) return EP_FULL;
        if (o->oneply) {
            /* value_oneply_checker_cubeful(state, match_state); 0-ply play. */
            double v1;
            int idx = bgc_i_value1_play(e, &s, &cube, d1, d2, &v1, NULL, &next, NULL);
            if (idx == -2) return EP_ERROR;
            k->tgt[r] = (float)v1;
            if (idx >= 0) {
                if (board_is_game_over(&next)) return EP_KEPT;   /* break, no dice */
                s = next;
            } else {
                board_switch_turn(&s);
            }
        } else {
            /* choose_checker_action_cubeful: -V_opp(chosen) with the
               opponent's cube perspective and is_cube_action =
               can_offer(opponent), terminal-masked (select0). */
            float val;
            int idx;
            {
                PROF_ENTER(BGC_PC_SEL);
                idx = bgc_best_move(e, &s, d1, d2, &cube, 0, &next, NULL, &val);
                PROF_LEAVE(1);
            }
            if (idx == -2) return EP_ERROR;
            if (idx >= 0) {
                if (board_is_game_over(&next)) {
                    /* CubefulMoneyMode.make_terminal_outcome: game_result,
                       1 under Jacoby with a centered cube. */
                    k->tgt[r] = (float)(gammons_count(&cube) ? bgc_i_game_result(&next) : 1);
                    return EP_KEPT;
                }
                k->tgt[r] = val;
                s = next;
            } else {
                /* Forced pass: -V(opp view, opp_persp, can_offer(opp)). */
                board_switch_turn(&s);
                PROF_ENTER(BGC_PC_VAL);
                k->tgt[r] = -bgc_value(e, &s, &cube, can_offer(cube.owner, s.turn));
                PROF_LEAVE(1);
            }
        }
        roll(rng, &d1, &d2);
        opening = 0;
    }
    return EP_KEPT;
}

long bgc_collect(bgc_engine *e, const bgc_collect_opts *o, unsigned long long seed,
                 int n_episodes, float *enc, float *tgt, long cap, int *n_discarded) {
    if (n_discarded) *n_discarded = 0;
    if (!e || !o || n_episodes < 0 || cap < 0 || (cap > 0 && (!enc || !tgt))) return -1;
    int kind = bgc_kind(e);
    if (kind != BGC_KIND_PROB && kind != BGC_KIND_EQUITY && kind != BGC_KIND_CUBEFUL
        && kind != BGC_KIND_PROB5)
        return -1;
    const bgc_encoder *en = bgc_i_encoder(e);
    if (!en->collect) {
        fprintf(stderr, "bgc_collect: encoder %s: no C collector (encoder table: "
                "collect 0); collect it in Python\n", en->name);
        return BGC_COLLECT_UNSUPPORTED;
    }
    PROF_ENTER(BGC_PC_COLLECT);
    bgc_rng rng;
    bgc_rng_seed(&rng, seed);                         /* random.Random(seed) */
    sink k = {enc, tgt, cap, 0, bgc_input_size(e), kind == BGC_KIND_PROB5 ? 5 : 1, en};
    int nd = 0;
    for (int ep = 0; ep < n_episodes; ep++) {
        long r0 = k.n;
        int rc = kind == BGC_KIND_CUBEFUL ? episode_cubeful(e, o, &rng, &k)
               : kind == BGC_KIND_PROB5   ? episode_prob5(e, o, &rng, &k)
                                          : episode_cubeless(e, o, &rng, &k);
        if (rc < 0) { PROF_LEAVE(0); return rc; }
        if (rc == EP_DISCARDED) {                     /* the whole episode goes */
            k.n = r0;
            nd++;
        }
    }
    if (n_discarded) *n_discarded = nd;
    PROF_LEAVE(k.n);
    return k.n;
}
