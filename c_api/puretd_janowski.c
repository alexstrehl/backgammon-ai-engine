/*
 * puretd_janowski.c -- p5jan's conversion in C: janowski_match.py's
 * gnubg-style cubeless -> cubeful match-winning-chance conversion and
 * janowski.cube_x (puretd.h "p5jan: the Janowski conversion").
 *
 * The Python functions are the reference, ported operation for operation:
 *   do_cubeful         bgc_jan_do_cubeful
 *   _rates, dead_mwc   jan_rates, bgc_i_jan_eval / bgc_jan_dead_mwc
 *   get_points         jan_points (lookups in bgc_i_jan_prep)
 *   cubeful_mwc        bgc_i_jan_prep + bgc_i_jan_eval = bgc_jan_cubeful_mwc
 *   cube_decision      bgc_jan_cube_decision
 *   cube_x             bgc_jan_cube_x
 *
 * Numerics: double throughout, every expression in Python's evaluation
 * order (left to right; a * b / c is (a * b) / c), no FMA contraction, the
 * same guards (_EPS = 1e-7, `abs(denom) > _EPS`). The MET lookups
 * (met.mwc_after through bgc_met_mwc_after) depend on the context only and
 * are done once per context (bgc_i_jan_prep) instead of once per call:
 * the same doubles, so the results are Python's bit for bit given the same
 * probabilities. A lookup the MET cannot price (Python raises) is NAN.
 */

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "puretd.h"
#include "puretd_internal.h"

#include <math.h>
#include <string.h>

#define JAN_EPS 1e-7           /* janowski_match._EPS */

/* _me(met, ctx', pts, win) for side k of ctx (k = 1: ctx.flip()). */
static double side_me(const bgc_met *t, const bgc_jan_ctx *c, int k, long long pts, int win) {
    const int my = k ? c->opp_away : c->my_away, opp = k ? c->my_away : c->opp_away;
    return bgc_met_mwc_after(t, my, opp, c->crawford_done != 0, c->is_crawford != 0, pts, win);
}

int bgc_jan_do_cubeful(const bgc_jan_ctx *c) {
    if (!c) return -1;
    if (c->is_crawford) return 0;
    if (c->cube >= c->my_away && c->cube >= c->opp_away) return 0;
    if (c->my_away == 2 && c->opp_away == 2) return 0;
    return 1;
}

int bgc_i_jan_prep(bgc_jan_prep *P, const bgc_met *t, const bgc_jan_ctx *c) {
    const long long v = c->cube;
    P->cubeful = bgc_jan_do_cubeful(c);
    for (int i = 0; i < 3; i++) {
        P->w[i] = side_me(t, c, 0, v * (i + 1), 1);
        P->l[i] = side_me(t, c, 0, v * (i + 1), 0);
    }
    P->dw1 = P->w[1] - P->w[0];
    P->dw2 = P->w[2] - P->w[1];
    P->dl1 = P->l[1] - P->l[0];
    P->dl2 = P->l[2] - P->l[1];
    /* get_points' ladder: climb while the doubled cube would still be alive
       for both (gnubg: i >= 2*nDead && j >= 2*nDead with i = away - 1). */
    const long long away[2] = {c->my_away, c->opp_away};
    long long n_dead = v;
    int n_max = 0;
    while (away[0] - 1 >= 2 * n_dead && away[1] - 1 >= 2 * n_dead) {
        if (++n_max >= BGC_JAN_LEVELS) return -1;
        n_dead *= 2;
    }
    P->n_max = n_max;
    long long cv = n_dead;
    for (int n = n_max; n >= 0; n--) {
        for (int k = 0; k < 2; k++) {
            bgc_jan_level *L = &P->lv[k][n];
            L->top = away[0] - 1 < 2 * cv || away[1] - 1 < 2 * cv;
            L->dp = side_me(t, c, k, cv, 1);
            if (L->top) {
                /* Automatic redouble: when the doubled cube is dead for the
                   DOUBLER only, the taker redoubles for free -> stakes at
                   the prime value 2cv. */
                const long long pv = (away[k] - 1 < 2 * cv && away[1 - k] - 1 >= 2 * cv)
                                     ? 2 * cv : cv;
                L->t0 = side_me(t, c, k, 2 * pv, 0);
                L->t1 = side_me(t, c, k, 4 * pv, 0);
                L->t2 = side_me(t, c, k, 6 * pv, 0);
                L->u0 = side_me(t, c, k, 2 * pv, 1);
                L->u1 = side_me(t, c, k, 4 * pv, 1);
                L->u2 = side_me(t, c, k, 6 * pv, 1);
            } else {
                L->t0 = side_me(t, c, k, 2 * cv, 0);         /* RDP */
                L->t1 = L->t2 = 0.0;
                L->u0 = side_me(t, c, k, 2 * cv, 1);
                L->u1 = side_me(t, c, k, 4 * cv, 1);
                L->u2 = side_me(t, c, k, 6 * cv, 1);
            }
        }
        cv /= 2;
    }
    return 0;
}

/* _rates: (g_me, bg_me, g_opp, bg_opp), gnubg's guarded ratios. */
static void jan_rates(const double pr[5], double r[4]) {
    const double p = pr[0];
    if (p > 0.0) {
        r[0] = (pr[1] - pr[2]) / p;
        r[1] = pr[2] / p;
    } else {
        r[0] = r[1] = 0.0;
    }
    if (p < 1.0) {
        r[2] = (pr[3] - pr[4]) / (1.0 - p);
        r[3] = pr[4] / (1.0 - p);
    } else {
        r[2] = r[3] = 0.0;
    }
}

/* get_points: the live cash points (cp_me, cp_opp), each in that side's own
   game-win-probability space. cp_live[k][n] is computed from level n + 1's
   cp_live[1 - k], so two rolling pairs replace the table. */
static void jan_points(const bgc_jan_prep *P, const double rates[4], double cp[2]) {
    const double r_g[2] = {rates[0], rates[2]}, r_bg[2] = {rates[1], rates[3]};
    double nxt[2] = {0.0, 0.0}, cur[2];
    for (int n = P->n_max; n >= 0; n--) {
        for (int k = 0; k < 2; k++) {
            const bgc_jan_level *L = &P->lv[k][n];
            const double g_win = r_g[k], bg_win = r_bg[k];
            const double g_lose = r_g[1 - k], bg_lose = r_bg[1 - k];
            if (L->top) {
                const double dtl = (1.0 - g_lose - bg_lose) * L->t0 + g_lose * L->t1
                                   + bg_lose * L->t2;
                const double dtw = (1.0 - g_win - bg_win) * L->u0 + g_win * L->u1
                                   + bg_win * L->u2;
                const double denom = dtl - dtw;
                cur[k] = fabs(denom) > JAN_EPS ? (dtl - L->dp) / denom : 1.0;
            } else {
                const double dtw = (1.0 - g_win - bg_win) * L->u0 + g_win * L->u1
                                   + bg_win * L->u2;
                const double denom = L->t0 - dtw;
                const double frac = fabs(denom) > JAN_EPS ? (L->dp - dtw) / denom : 0.0;
                cur[k] = 1.0 - nxt[1 - k] * frac;
            }
        }
        nxt[0] = cur[0];
        nxt[1] = cur[1];
    }
    cp[0] = nxt[0];
    cp[1] = nxt[1];
}

static double jan_dead(const bgc_jan_prep *P, const double pr[5]) {
    const double p = pr[0];
    return p * P->w[0] + pr[1] * P->dw1 + pr[2] * P->dw2 + (1.0 - p) * P->l[0]
           + pr[3] * P->dl1 + pr[4] * P->dl2;
}

double bgc_i_jan_eval(const bgc_jan_prep *P, const double pr[5], int persp, double x) {
    const double p = pr[0];
    const double dead = jan_dead(P, pr);
    if (!P->cubeful || p <= JAN_EPS || p >= 1.0 - JAN_EPS) return dead;
    double r[4], cp[2];
    jan_rates(pr, r);
    jan_points(P, r, cp);
    const double g_me = r[0], bg_me = r[1], g_opp = r[2], bg_opp = r[3];
    const double cp_me = cp[0], cp_opp = cp[1];
    const double w1 = P->w[0], l1 = P->l[0];          /* cash: plain win / loss of the cube */
    const double mwc_win = (1.0 - g_me - bg_me) * w1 + g_me * P->w[1] + bg_me * P->w[2];
    const double mwc_lose = (1.0 - g_opp - bg_opp) * l1 + g_opp * P->l[1] + bg_opp * P->l[2];
    double live;
    if (persp == PERSP_CENTERED) {
        const double opp_tg = 1.0 - cp_opp, tg = cp_me;
        if (p <= opp_tg)
            live = opp_tg > 0.0 ? mwc_lose + (l1 - mwc_lose) * p / opp_tg : mwc_lose;
        else if (p < tg)
            live = l1 + (w1 - l1) * (p - opp_tg) / (tg - opp_tg);
        else
            live = tg < 1.0 ? w1 + (mwc_win - w1) * (p - tg) / (1.0 - tg) : mwc_win;
    } else if (persp == PERSP_MINE) {
        const double tg = cp_me;
        if (p <= tg)
            live = tg > 0.0 ? mwc_lose + (w1 - mwc_lose) * p / tg : mwc_lose;
        else
            live = tg < 1.0 ? w1 + (mwc_win - w1) * (p - tg) / (1.0 - tg) : mwc_win;
    } else {                                          /* THEIRS: cube unavailable to us */
        const double opp_tg = 1.0 - cp_opp;
        if (p <= opp_tg)
            live = opp_tg > 0.0 ? mwc_lose + (l1 - mwc_lose) * p / opp_tg : mwc_lose;
        else
            live = l1 + (mwc_win - l1) * (p - opp_tg) / (1.0 - opp_tg);
    }
    return (1.0 - x) * dead + x * live;
}

/* ── the isolated entry points (no net) ─────────────────────────── */

static int met_ok(const bgc_met *t) {
    return t && t->pre && t->post && t->n >= 1;
}

double bgc_jan_dead_mwc(const bgc_met *t, const double probs[5], const bgc_jan_ctx *c) {
    bgc_jan_prep P;
    if (!met_ok(t) || !probs || !c || bgc_i_jan_prep(&P, t, c)) return NAN;
    return jan_dead(&P, probs);
}

int bgc_jan_get_points(const bgc_met *t, const double rates[4], const bgc_jan_ctx *c,
                       double cp[2]) {
    bgc_jan_prep P;
    if (!met_ok(t) || !rates || !c || !cp || bgc_i_jan_prep(&P, t, c)) return -1;
    jan_points(&P, rates, cp);
    return isnan(cp[0]) || isnan(cp[1]) ? -1 : 0;
}

double bgc_jan_cubeful_mwc(const bgc_met *t, const double probs[5], int persp,
                           const bgc_jan_ctx *c, double x) {
    bgc_jan_prep P;
    if (!met_ok(t) || !probs || !c || persp < 0 || persp > 2 || bgc_i_jan_prep(&P, t, c))
        return NAN;
    return bgc_i_jan_eval(&P, probs, persp, x);
}

int bgc_jan_cube_decision(const bgc_met *t, const double probs[5], const bgc_match *m,
                          int player, double x, bgc_match_cube *out) {
    if (out) out->v_nd = out->v_dt = out->v_dp = NAN;
    if (!met_ok(t) || !probs || !bgc_m_ok(m) || (player != 0 && player != 1)) return -1;
    /* ctx_for(m, player); cube_perspective(owner, player) */
    const bgc_jan_ctx c = {m->away[player], m->away[1 - player], m->crawford_done != 0,
                           m->is_crawford != 0, m->cube_value};
    bgc_jan_ctx cdt = c;
    cdt.cube = 2 * c.cube;                            /* ctx.with_cube(2 * ctx.cube) */
    bgc_jan_prep P;
    if (bgc_i_jan_prep(&P, t, &c)) return -1;
    const double v_nd = bgc_i_jan_eval(&P, probs, cube_persp(m->cube_owner, player), x);
    if (bgc_i_jan_prep(&P, t, &cdt)) return -1;
    const double v_dt = bgc_i_jan_eval(&P, probs, PERSP_THEIRS, x);
    const double v_dp = side_me(t, &c, 0, c.cube, 1);
    if (isnan(v_nd) || isnan(v_dt) || isnan(v_dp)) return -1;
    if (out) { out->v_nd = v_nd; out->v_dt = v_dt; out->v_dp = v_dp; }
    return (v_dp < v_dt ? v_dp : v_dt) > v_nd;       /* min(v_dt, v_dp) > v_nd */
}

double bgc_jan_cube_x(const BoardState *s) {
    if (!s) return NAN;
    /* backgammon_engine.is_race */
    int race = 0;
    if (!s->bar[WHITE] && !s->bar[BLACK]) {
        int w_max = -1, b_min = NUM_POINTS;
        for (int i = 0; i < NUM_POINTS; i++) {
            const int v = s->points[i];
            if (v > 0) w_max = i;
            else if (v < 0 && b_min == NUM_POINTS) b_min = i;
        }
        race = w_max < b_min;
    }
    if (!race) return 0.76;                           /* CONTACT_X["shallow"] */
    /* pip_count(state)[state.turn] */
    int wp = 25 * s->bar[WHITE], bp = 25 * s->bar[BLACK];
    for (int i = 0; i < NUM_POINTS; i++) {
        const int v = s->points[i];
        if (v > 0) wp += v * (i + 1);
        else if (v < 0) bp += -v * (NUM_POINTS - i);
    }
    const double x = 0.60 + 0.0025 * (double)(s->turn == WHITE ? wp : bp);
    const double lo = x > 0.60 ? x : 0.60;            /* max(RACE_MIN, x) */
    return lo < 0.80 ? lo : 0.80;                     /* min(RACE_MAX, .) */
}

/* ── the MATCH_JANOWSKI engine's rows (puretd_match.c's seam) ────── */

int bgc_i_jan_set(bgc_engine *e, const bgc_jan_ctx *c, int persp) {
    if (c->my_away < 1 || c->opp_away < 1 || c->my_away > e->met.n || c->opp_away > e->met.n)
        return -1;
    e->jpersp = persp;
    return bgc_i_jan_prep(&e->jprep, &e->met, c);
}

int bgc_i_jan_rows(bgc_engine *e, int r0, int n) {
    for (int r = r0; r < r0 + n; r++) {
        float q[5];
        bgc_i_prob5_clamp(e->y + (size_t)r * 5, q);
        const double pd[5] = {q[0], q[1], q[2], q[3], q[4]};
        const double v = bgc_i_jan_eval(&e->jprep, pd, e->jpersp, e->rx[r]);
        if (isnan(v)) return -1;
        e->vd[r] = v;
    }
    return 0;
}

int bgc_set_jan_x(bgc_engine *e, int constant, double x) {
    if (!e || e->kind != BGC_KIND_MATCH_JANOWSKI || (constant != 0 && constant != 1)
        || (constant && !isfinite(x)))
        return -1;
    e->jx_const = constant;
    e->jx = constant ? x : 0.0;
    bgc_i_match_memo_reset(e);              /* the memos' values used the old x */
    return 0;
}
