/*
 * puretd_match_collect.c -- the match-episode collector (bgc_collect_match).
 *
 * Port of trainer._collect_episode_matchplay driven by
 * trainer._collect_worker (rng_mode 0: every episode on one
 * random.Random(seed)) or by vectorized_matchplay's worker at B = 1
 * (rng_mode 1: episode i on random.Random(base.randint(0, 2**31 - 1)),
 * base = random.Random(seed)), with a TDAgent at boltzmann_temp 0, 0-ply
 * acting, a MET grounding every _eval_new_game scope, no oracle, no
 * --td-move-eps, no move exploration: per sample the row the Python
 * collector stacks and the target it appends, in its order. The Python
 * function is the reference; the RNG draws are its draws, including the
 * coins it draws at probability 0 (the curriculum coin at P = 0, the cube
 * exploration coins at eps = 0).
 *
 * Every value comes from the match core (puretd_match.c): bgc_match_offer
 * (the cube phase: 0-ply v_nd, v_dt, the MET's v_dp), bgc_match_best_move
 * (0-ply play; its value is the 0-ply bootstrap), bgc_match_value_1ply_play
 * (1-ply targets: the 1-ply value and the 0-ply play from one expansion),
 * bgc_match_value (the 0-ply forced-pass target), bgc_match_new_game_mwc
 * (the boundary targets). Rows go
 * straight into the caller's buffers (bgc_match_encode); the engine's
 * scratch does the rest, the episode's row viewers (outcome_mix) included:
 * it grows to its high-water mark, then no heap allocation per decision.
 */

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "puretd.h"
#include "puretd_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EP_KEPT 0
#define EP_DISCARDED 1          /* max_plies exceeded */
#define EP_ERROR (-1)
#define EP_FULL (-2)            /* caller's buffer exhausted */

static inline double py_min(double a, double b) { return b < a ? b : a; }
static inline double py_max(double a, double b) { return b > a ? b : a; }

typedef struct {
    float *enc, *tgt;
    long cap, n;
    long r0;                /* the episode's first row */
    int in_size;
} msink;

/* e->mview holds the viewer of the episode's rows (index row - r0). */
static int view_reserve(bgc_engine *e, long n) {
    if (n <= e->cap_mview) return 0;
    long cap = e->cap_mview ? e->cap_mview : 4096;
    while (cap < n) cap *= 2;
    unsigned char *v = realloc(e->mview, (size_t)cap);
    if (!v) return -1;
    e->mview = v;
    e->cap_mview = cap;
    return 0;
}

/* Appends _encode_matchplay(s, m, ica), viewer s->turn. The row index, or
   EP_FULL / EP_ERROR. */
static long put_row(bgc_engine *e, msink *k, const BoardState *s, const bgc_match *m, int ica) {
    if (k->n >= k->cap) return EP_FULL;
    if (view_reserve(e, k->n - k->r0 + 1)) return EP_ERROR;
    PROF_T0(t0);
    if (bgc_match_encode(e, s, m, ica, k->enc + (size_t)k->n * k->in_size)) return EP_ERROR;
    PROF_ADD(t0, BGC_PT_ENCODE, 1);
    k->tgt[k->n] = NAN;
    e->mview[k->n - k->r0] = (unsigned char)s->turn;
    return k->n++;
}

/* enc_list.append(enc_list[-1]); targets.append(t); viewers.append(player):
   the boundary upweight's copies of row r. 0, EP_FULL or EP_ERROR. */
static int dup_rows(bgc_engine *e, msink *k, long r, int copies) {
    for (int i = 0; i < copies; i++) {
        if (k->n >= k->cap) return EP_FULL;
        if (view_reserve(e, k->n - k->r0 + 1)) return EP_ERROR;
        memcpy(k->enc + (size_t)k->n * k->in_size, k->enc + (size_t)r * k->in_size,
               (size_t)k->in_size * sizeof(float));
        k->tgt[k->n] = k->tgt[r];
        e->mview[k->n - k->r0] = e->mview[r - k->r0];
        k->n++;
    }
    return 0;
}

/* trainer._curriculum_start. */
static void curriculum(const bgc_match_collect_opts *o, bgc_rng *rng, bgc_match *ms) {
    const int n = o->match_length;
    memset(ms, 0, sizeof(*ms));             /* MatchplayMode.initial_match_state() */
    ms->length = n;
    ms->away[0] = ms->away[1] = n;
    ms->cube_value = 1;
    if (o->force_my > 0) {                  /* TRAINER_CURRICULUM_FORCE: no draw */
        ms->away[0] = o->force_my;
        ms->away[1] = o->force_opp;
        ms->crawford_done = o->force_my == 1 || o->force_opp == 1;
        return;
    }
    if (n <= 1 || bgc_rng_random(rng) >= o->curriculum_p) return;    /* drawn at P = 0 too */
    int my = bgc_rng_randint(rng, 1, n);
    int opp = bgc_rng_randint(rng, 1, n);
    while (my == n && opp == n) {
        my = bgc_rng_randint(rng, 1, n);
        opp = bgc_rng_randint(rng, 1, n);
    }
    int cd = 0;
    if (my == 1 || opp == 1) cd = (my == 1 && opp == 1) || bgc_rng_random(rng) < 0.5;
    ms->away[0] = my;
    ms->away[1] = opp;
    ms->crawford_done = cd;
}

/* One match: trainer._collect_episode_matchplay (the per-game ply bound of
   vectorized_matchplay.collect_episode_matchplay_gen when max_plies > 0). */
static int episode(bgc_engine *e, const bgc_match_collect_opts *o, bgc_rng *rng, msink *k) {
    const int dups = o->boundary_upweight - 1;
    bgc_match ms;
    curriculum(o, rng, &ms);
    for (;;) {
        bgc_match g = bgc_m_start_game(&ms);
        BoardState s, next;
        int d1, d2, n_dice = 0, plies = 0, opening = 1, game_done = 0;
        bgc_i_opening_roll(rng, &s, &d1, &d2, &n_dice);
        while (!game_done) {
            if (o->max_plies > 0 && plies >= o->max_plies) return EP_DISCARDED;
            plies++;
            const int p = s.turn;

            /* ── cube phase ── */
            if (!opening && bgc_m_can_offer(&g, p)) {
                const long r = put_row(e, k, &s, &g, 1);
                if (r < 0) return (int)r;
                bgc_match_cube c;
                int dbl;
                {
                    PROF_ENTER(BGC_PC_CUBE);
                    dbl = bgc_match_offer(e, &s, &g, &c);
                    PROF_LEAVE(1);
                }
                if (dbl < 0) return EP_ERROR;
                /* optimal-action backup, whatever is executed */
                const double opt = py_max(c.v_nd, py_min(c.v_dt, c.v_dp));
                int do_double = dbl, explored = 0;
                if (bgc_rng_random(rng) < o->cube_explore_eps) {     /* drawn at eps 0 too */
                    do_double = bgc_rng_random(rng) < 0.5;
                    explored = 1;
                }
                if (do_double) {
                    int takes = c.v_dp > c.v_dt;                       /* respond_to_double */
                    if (bgc_rng_random(rng) < o->cube_explore_eps) {
                        takes = bgc_rng_random(rng) < 0.5;
                        explored = 1;
                    }
                    if (takes) {
                        k->tgt[r] = (float)opt;
                        g = bgc_m_after_take(&g, p);
                    } else {
                        ms = bgc_m_after_game(&g, 1, p);               /* dropped: p scores */
                        double t;
                        if (explored) t = opt;
                        else if (bgc_m_over(&ms)) t = 1.0;
                        else t = bgc_match_new_game_mwc(e, &ms, p);     /* "boundary" */
                        if (isnan(t)) return EP_ERROR;
                        k->tgt[r] = (float)t;
                        if (!explored) {
                            const int rc = dup_rows(e, k, r, dups);
                            if (rc) return rc;
                        }
                        game_done = 1;
                        continue;
                    }
                } else {
                    k->tgt[r] = (float)opt;
                }
            }

            /* ── checker phase ── */
            const long r = put_row(e, k, &s, &g, 0);
            if (r < 0) return (int)r;
            int idx;
            if (o->oneply) {
                /* value_oneply_checker_matchplay(state, gms) in every case:
                   normal play, forced pass, game-ending play */
                double v1;
                idx = bgc_match_value_1ply_play(e, &s, d1, d2, &g, &v1, &next);
                if (idx == -2) return EP_ERROR;
                k->tgt[r] = (float)v1;
                if (idx < 0) {
                    board_switch_turn(&s);
                } else if (board_is_game_over(&next)) {
                    ms = bgc_m_after_game(&g, bgc_i_game_result(&next), p);
                    const int rc = dup_rows(e, k, r, dups);
                    if (rc) return rc;
                    game_done = 1;
                    continue;
                } else {
                    s = next;
                }
            } else {
                float ov;
                {
                    PROF_ENTER(BGC_PC_SEL);
                    idx = bgc_match_best_move(e, &s, d1, d2, &g, &next, &ov);
                    PROF_LEAVE(1);
                }
                if (idx == -2) return EP_ERROR;
                if (idx < 0) {
                    /* forced pass: 1 - V(opp view, ica = can_offer(opp)) */
                    board_switch_turn(&s);
                    float v;
                    {
                        PROF_ENTER(BGC_PC_VAL);
                        v = bgc_match_value(e, &s, &g, bgc_m_can_offer(&g, s.turn));
                        PROF_LEAVE(1);
                    }
                    if (isnan(v)) return EP_ERROR;
                    k->tgt[r] = (float)(1.0 - (double)v);
                } else if (board_is_game_over(&next)) {
                    ms = bgc_m_after_game(&g, bgc_i_game_result(&next), p);
                    const double t = bgc_m_over(&ms) ? 1.0 : bgc_match_new_game_mwc(e, &ms, p);
                    if (isnan(t)) return EP_ERROR;
                    k->tgt[r] = (float)t;
                    const int rc = dup_rows(e, k, r, dups);
                    if (rc) return rc;
                    game_done = 1;
                    continue;
                } else {
                    k->tgt[r] = (float)(1.0 - (double)ov);       /* bootstrap */
                    s = next;
                }
            }
            d1 = bgc_rng_randint(rng, 1, 6);       /* dice = (rng.randint(1, 6), */
            d2 = bgc_rng_randint(rng, 1, 6);       /*         rng.randint(1, 6)) */
            opening = 0;
        }
        if (bgc_m_over(&ms)) break;
    }
    if (o->outcome_mix > 0.0) {
        /* t = (1 - mix) * t + mix * z in float32 (numpy: the Python float
           scalars become float32), z = 1 iff the row's viewer won */
        const float a = (float)(1.0 - o->outcome_mix), b = (float)o->outcome_mix;
        const int white_won = ms.away[0] <= 0;
        for (long i = k->r0; i < k->n; i++) {
            const int v = e->mview[i - k->r0];
            const float z = (v == 0 ? white_won : !white_won) ? 1.0f : 0.0f;
            const float x = a * k->tgt[i];
            const float y = b * z;
            k->tgt[i] = x + y;
        }
    }
    return EP_KEPT;
}

long bgc_collect_match(bgc_engine *e, const bgc_match_collect_opts *o, unsigned long long seed,
                       int n_episodes, float *enc, float *tgt, long cap, int *n_discarded) {
    if (n_discarded) *n_discarded = 0;
    if (!e || !o || n_episodes < 0 || cap < 0 || (cap > 0 && (!enc || !tgt))) return -1;
    if (bgc_kind(e) != BGC_KIND_MATCH) return -1;
    const bgc_encoder *en = bgc_i_encoder(e);
    if (!en->collect) {
        fprintf(stderr, "bgc_collect_match: encoder %s: no C collector (encoder table: "
                "collect 0); collect it in Python\n", en->name);
        return BGC_COLLECT_UNSUPPORTED;
    }
    if (!e->met_pre || o->match_length < 1 || o->match_length > e->met_n
        || (o->oneply != 0 && o->oneply != 1) || (o->rng_mode != 0 && o->rng_mode != 1)
        || o->force_my < 0 || o->force_opp < 0 || (o->force_my > 0) != (o->force_opp > 0)
        || !(o->curriculum_p >= 0.0 && o->curriculum_p <= 1.0)
        || !(o->cube_explore_eps >= 0.0 && o->cube_explore_eps <= 1.0)
        || !(o->outcome_mix >= 0.0 && o->outcome_mix <= 1.0)
        || o->boundary_upweight < 1)
        return -1;
    PROF_ENTER(BGC_PC_COLLECT);
    bgc_rng base, ep_rng;
    bgc_rng_seed(&base, seed);                        /* random.Random(seed) */
    msink k = {enc, tgt, cap, 0, 0, bgc_input_size(e)};
    int nd = 0;
    for (int ep = 0; ep < n_episodes; ep++) {
        bgc_rng *rng = &base;
        if (o->rng_mode == 1) {                       /* Random(base.randint(0, 2**31 - 1)) */
            bgc_rng_seed(&ep_rng, (unsigned long long)bgc_rng_randint(&base, 0, 2147483647));
            rng = &ep_rng;
        }
        k.r0 = k.n;
        const int rc = episode(e, o, rng, &k);
        if (rc < 0) { PROF_LEAVE(0); return rc; }
        if (rc == EP_DISCARDED) {                     /* the whole match goes */
            k.n = k.r0;
            nd++;
        }
    }
    if (n_discarded) *n_discarded = nd;
    PROF_LEAVE(k.n);
    return k.n;
}
