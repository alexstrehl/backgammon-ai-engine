/*
 * puretd_match.c -- match-play core: the BGC_KIND_MATCH entry points
 * (puretd.h bgc_set_met, bgc_match_*).
 *
 * Ports of the TDAgent matchplay methods (td_agent.py) with a MET grounding
 * every _eval_new_game scope (cube, terminal, boundary), boltzmann_temp 0
 * and cube_fold off, of modes.MatchState and of met.MatchEquityTable's
 * lookups; and, on a BGC_KIND_MATCH_JANOWSKI engine (a prob5 net, p5jan),
 * of janowski_match.JanowskiMatchAgent through the same entry points (the
 * evaluator seam below). The Python functions are the reference:
 *   _encode_matchplay            bgc_match_encode (base + match_extras)
 *   _eval_new_game               bgc_match_new_game_mwc
 *   evaluate_matchplay           bgc_match_value
 *   offer_double_matchplay /
 *   respond_to_double_matchplay  bgc_match_offer (take = v_dp > v_dt)
 *   choose_checker_action_matchplay   bgc_match_best_move / _score_moves
 *   value_oneply_checker_matchplay    bgc_match_value_1ply
 *
 * The batch machinery is puretd.c's (puretd_internal.h): expand21 with the
 * cross-roll row dedup, forward_succ (delta forward from the parent),
 * value1_play's roll matching. What differs from the money kinds is
 * confined to (a) the rows' columns after the base: the match block of the
 * current context (e->tail, written by match_extras; encode_base and
 * forward_succ copy it) instead of the cube columns, (b) terminal
 * successors priced by a per-call table e->mtv[gr] = float32(1 - mover_val)
 * (mover_val = 1 if the game wins the match, else the MET's new-game MWC),
 * (c) the 1-ply reduction: 1 - min per roll, rolls in dice order (forced
 * passes in place, not grouped), clamped to [0, 1], and the all-pass rule
 * (every roll a forced pass: the pass value, unclamped).
 *
 * Numerics: leaf values fp32 net outputs; MWCs, cube comparisons and the
 * 21-roll sum in double in Python's order; float32 rounding exactly where
 * Python rounds (terminal overrides stored into fp32 arrays, the MET block
 * columns, the final target cast). No FMA contraction.
 *
 * The evaluator seam: what the two kinds share is the
 * decision structure (move generation, the candidate / expansion batches,
 * the delta forward, value_1ply_play's roll matching, the memos, the
 * errors); what differs is confined to
 *   set_ctx       TD: the match block of the rows (e->tail); Janowski: the
 *                 conversion's context and perspective (bgc_i_jan_set; the
 *                 rows are the bare base)
 *   ev_rows       after a forward, TD: nothing (e->v is the value);
 *                 Janowski: e->vd[r] = cubeful_mwc of the row (bgc_i_jan_rows)
 *   set_tv        terminal successors, TD: fp32 opp-view e->mtv; Janowski:
 *                 double mover-view e->mtvd (met.mwc_after)
 *   select / reduce   TD: first argmin of fp32 opp-view values, 1-ply 1 - min
 *                 per roll, clamped, the all-pass rule; Janowski: first
 *                 argmax of double mover-view values, 1-ply max per roll,
 *                 unclamped (JanowskiMatchAgent.choose_checker_action_cubeful,
 *                 value_oneply)
 *   the cube      TD: two forwards; Janowski: one forward, two conversions
 *                 (cube_decision)
 *
 * Speed knobs (env, read at bgc_open; each bitwise: the collector's rows
 * and targets are the same on and off):
 *   BGC_MATCH_MEMO=0        the 1-ply memo off (default on)
 *   BGC_FAST_MOVEGEN=0      get_legal_plays instead of puretd_movegen.c's
 *                           bgc_i_legal_plays (the same play list, a cheaper
 *                           leaf dedup); default on for the MATCH kind,
 *                           "all" = every kind (tests)
 *   BGC_MATCH_SHM=0         no shared memo (default on with the memo, on
 *                           row-exact backends): the memo's values shared by
 *                           the engines of a round (every process on the same
 *                           weights file contents, library, backend config and
 *                           MET) through a POSIX shm table ("the shared memo")
 *   BGC_MATCH_SHM_PIPS=n    the positions it holds: pip sum >= n, none borne
 *                           off [315; the start position is 334]
 */

#define _GNU_SOURCE          /* dladdr, shm_open (the shared memo) */

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "puretd.h"
#include "puretd_internal.h"

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Python's min(a, b) / max(a, b): the first argument unless the second
   compares strictly smaller / larger. */
static inline double py_min(double a, double b) { return b < a ? b : a; }
static inline double py_max(double a, double b) { return b > a ? b : a; }

static void ms_attach(bgc_engine *e);

/* ── MET (met.MatchEquityTable) ─────────────────────────────────── */

int bgc_set_met(bgc_engine *e, const double *pre, const double *post, int n) {
    if (!e || !bgc_i_is_match(e) || !pre || !post || n < 1 || n > 4096) return -1;
    for (long i = 0; i < (long)n * n; i++) if (!isfinite(pre[i])) return -1;
    for (int i = 0; i < n; i++) if (!isfinite(post[i])) return -1;
    double *a = malloc((size_t)n * n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    if (!a || !b) { free(a); free(b); return -1; }
    memcpy(a, pre, (size_t)n * n * sizeof(double));
    memcpy(b, post, (size_t)n * sizeof(double));
    free(e->met_pre);
    free(e->met_post);
    e->met_pre = a;
    e->met_post = b;
    e->met_n = n;
    e->met = (bgc_met){a, b, n};
    bgc_i_match_memo_reset(e);                   /* the memo's values used the old MET */
    return 0;
}

void bgc_i_match_memo_reset(bgc_engine *e) {
    e->mm_n = 0;
    if (e->mm) memset(e->mm, 0, (size_t)e->mm_cap * sizeof(*e->mm));
    ms_attach(e);                                /* the shared memo of (weights, MET, ...) */
}

/* MatchEquityTable.mwc / mwc_after on the engine's table
   (puretd_internal.h bgc_met_mwc / bgc_met_mwc_after). */
static double met_mwc(const bgc_engine *e, int my, int opp, int cd, int ic) {
    return bgc_met_mwc(&e->met, my, opp, cd, ic);
}

static double met_mwc_after(const bgc_engine *e, int my, int opp, int cd, int ic,
                            long long pts, int me) {
    return bgc_met_mwc_after(&e->met, my, opp, cd, ic, pts, me);
}

/* outcome_block (13) / outcome_block_x6 (17), float32 as numpy stores
   them. -1 where Python raises. */
static int met_block(const bgc_engine *e, int my, int opp, int cd, int ic, int cv, int nmet,
                     float *out) {
    const long long cubes[2] = {cv, 2LL * cv};
    for (int s = 0; s < 2; s++) {
        for (int i = 0; i < 3; i++) {
            out[6 * s + i] = (float)met_mwc_after(e, my, opp, cd, ic, cubes[s] * (i + 1), 1);
            out[6 * s + 3 + i] = (float)met_mwc_after(e, my, opp, cd, ic, cubes[s] * (i + 1), 0);
        }
    }
    out[12] = (float)met_mwc_after(e, my, opp, cd, ic, cv, 1);    /* drop: the pre-double cube */
    if (nmet == 17) {
        const long long c4 = 4LL * cv;
        out[13] = (float)met_mwc_after(e, my, opp, cd, ic, c4 * 2, 1);
        out[14] = (float)met_mwc_after(e, my, opp, cd, ic, c4 * 3, 1);
        out[15] = (float)met_mwc_after(e, my, opp, cd, ic, c4 * 2, 0);
        out[16] = (float)met_mwc_after(e, my, opp, cd, ic, c4 * 3, 0);
    }
    for (int i = 0; i < nmet; i++) if (isnan(out[i])) return -1;
    return 0;
}

/* ── new-game MWC (MatchState helpers: puretd_internal.h bgc_m_*) ─ */

static double new_game_mwc(const bgc_engine *e, const bgc_match *after, int player) {
    const bgc_match ms = bgc_m_start_game(after);
    const double pw = met_mwc(e, ms.away[0], ms.away[1], ms.crawford_done != 0,
                              ms.is_crawford != 0);
    return player == 0 ? pw : 1.0 - pw;
}

double bgc_match_new_game_mwc(const bgc_engine *e, const bgc_match *after, int player) {
    if (!e || !bgc_i_is_match(e) || !e->met_pre || !bgc_m_ok(after)
        || (player != 0 && player != 1))
        return NAN;
    return new_game_mwc(e, after, player);
}

/* ── the match block ─────────────────────────────────────────────── */

/* MatchplayCube1H64Encoder._cube_slot; -1 where Python raises (KeyError). */
static int cube_slot(int v) {
    if (v >= 64) return 6;
    switch (v) {
        case 1: return 0;
        case 2: return 1;
        case 4: return 2;
        case 8: return 3;
        case 16: return 4;
        case 32: return 5;
        default: return -1;
    }
}

/* Matchplay1H64MET[6]Encoder.match_extras(*_match_features(m, viewer), ica)
   into x[nextra]. -1 where Python raises. */
static int match_extras(const bgc_engine *e, const bgc_match *m, int viewer, int ica,
                        float *x) {
    const bgc_match_block *mb = e->enc.match;
    const int persp = cube_persp(m->cube_owner, viewer);
    const int my = m->away[viewer], opp = m->away[1 - viewer];
    const int cv = m->cube_value, ic = m->is_crawford != 0;
    const int slot = cube_slot(cv);
    if (my < 1 || opp < 1 || slot < 0) return -1;
    memset(x, 0, (size_t)mb->nextra * sizeof(float));
    x[persp] = 1.0f;
    if (ica) x[3] = 1.0f;
    x[4 + slot] = 1.0f;
    x[11 + (my < 64 ? my : 64) - 1] = 1.0f;
    x[75 + (opp < 64 ? opp : 64) - 1] = 1.0f;
    if (ic) x[139] = 1.0f;
    if ((my == 1 || opp == 1) && !ic) x[140] = 1.0f;
    /* The encoder derives crawford_done as "someone is 1-away". */
    float blk[17];
    if (met_block(e, my, opp, my == 1 || opp == 1, ic, cv, mb->nmet, blk)) return -1;
    for (int i = 0; i < mb->nmet; i++) x[141 + i] = blk[i] - 0.5f;
    return 0;
}

static int is_jan(const bgc_engine *e) { return e->kind == BGC_KIND_MATCH_JANOWSKI; }

/* janowski_match.ctx_for(m, viewer, cube). */
static bgc_jan_ctx jan_ctx_for(const bgc_match *m, int viewer, int cube) {
    const bgc_jan_ctx c = {m->away[viewer], m->away[1 - viewer], m->crawford_done != 0,
                           m->is_crawford != 0, cube};
    return c;
}

/* The context of the rows that follow, positions with `viewer` on roll. TD:
   e->tail = their match block (the pad after it stays zero). Janowski:
   ctx_for(m, viewer) and cube_perspective(owner, viewer) (ica: no column). */
static int set_ctx(bgc_engine *e, const bgc_match *m, int viewer, int ica) {
    if (is_jan(e)) {
        const bgc_jan_ctx c = jan_ctx_for(m, viewer, m->cube_value);
        return bgc_i_jan_set(e, &c, cube_persp(m->cube_owner, viewer));
    }
    return match_extras(e, m, viewer, ica, e->tail);
}

/* The row values of rows [r0, r0 + n) after their forward: TD e->v as is;
   Janowski e->vd (the conversion at the current context). 0, -1. */
static int ev_rows(bgc_engine *e, int r0, int n) {
    return is_jan(e) ? bgc_i_jan_rows(e, r0, n) : 0;
}

/* e->mtv[gr]: a terminal successor's opp-view value when `mover` wins the
   game with result gr: float32(1 - mover_val), mover_val = 1 if that wins
   the match, else the new-game MWC ("terminal" scope). NAN where the MET
   cannot price it (Python raises only when such a successor occurs). */
static void set_tv(bgc_engine *e, const bgc_match *m, int mover) {
    if (is_jan(e)) {
        /* Janowski: the mover's MWC _me(met, ctx_me, ctx_me.cube * gr, True),
           double (and its fp32 opp view for the float outputs) */
        const bgc_jan_ctx c = jan_ctx_for(m, mover, m->cube_value);
        for (int gr = 1; gr <= 3; gr++) {
            e->mtvd[gr] = met_mwc_after(e, c.my_away, c.opp_away, c.crawford_done,
                                        c.is_crawford, (long long)c.cube * gr, 1);
            e->mtv[gr] = (float)(1.0 - e->mtvd[gr]);
        }
        return;
    }
    for (int gr = 1; gr <= 3; gr++) {
        const bgc_match a = bgc_m_after_game(m, gr, mover);
        const double mv = bgc_m_over(&a) ? 1.0 : new_game_mwc(e, &a, mover);
        e->mtv[gr] = (float)(1.0 - mv);
    }
}

static int ready(const bgc_engine *e, const BoardState *s, const bgc_match *m) {
    return e && s && bgc_i_is_match(e) && e->met_pre && bgc_m_ok(m);
}

int bgc_match_encode(bgc_engine *e, const BoardState *s, const bgc_match *m, int ica,
                     float *row) {
    if (!ready(e, s, m) || !row) return -1;
    e->enc.base->encode(s, row);
    if (is_jan(e)) return 0;                     /* Janowski: the base row */
    return match_extras(e, m, s->turn, ica != 0, row + e->nb);
}

/* ── 0-ply ───────────────────────────────────────────────────────── */

float bgc_match_value(bgc_engine *e, const BoardState *s, const bgc_match *m, int ica) {
    if (!ready(e, s, m) || set_ctx(e, m, s->turn, ica != 0)) return NAN;
    bgc_i_encode_base(e, 0, s);
    bgc_i_forward(e, 0, 1);
    if (is_jan(e)) return ev_rows(e, 0, 1) ? NAN : (float)e->vd[0];
    return e->v[0];
}

/* JanowskiMatchAgent.offer_double past its can_offer gate = cube_decision(
   probs(s), m, p, x(s)): one forward, v_nd at (persp, ctx), v_dt at
   (THEIRS, ctx with the doubled cube), v_dp = _me(ctx, cube, win). */
static int jan_offer(bgc_engine *e, const BoardState *s, const bgc_match *m, int p,
                     double *v_nd, double *v_dt, double *v_dp) {
    if (set_ctx(e, m, p, 0)) return -1;
    bgc_i_encode_base(e, 0, s);
    bgc_i_forward(e, 0, 1);                      /* Python: a batch-1 plain forward */
    if (ev_rows(e, 0, 1)) return -1;
    *v_nd = e->vd[0];
    const bgc_jan_ctx dt = jan_ctx_for(m, p, 2 * m->cube_value);
    if (bgc_i_jan_set(e, &dt, PERSP_THEIRS) || ev_rows(e, 0, 1)) return -1;
    *v_dt = e->vd[0];
    *v_dp = met_mwc_after(e, dt.my_away, dt.opp_away, dt.crawford_done, dt.is_crawford,
                          m->cube_value, 1);
    return isnan(*v_dp) ? -1 : 0;
}

int bgc_match_offer(bgc_engine *e, const BoardState *s, const bgc_match *m,
                    bgc_match_cube *out) {
    if (out) out->v_nd = out->v_dt = out->v_dp = NAN;
    if (!ready(e, s, m)) return -1;
    const int p = s->turn;
    if (!bgc_m_can_offer(m, p)) return 0;
    if (is_jan(e)) {
        double v_nd, v_dt, v_dp;
        if (jan_offer(e, s, m, p, &v_nd, &v_dt, &v_dp)) return -1;
        if (out) { out->v_nd = v_nd; out->v_dt = v_dt; out->v_dp = v_dp; }
        return py_min(v_dt, v_dp) > v_nd;
    }
    const bgc_match tk = bgc_m_after_take(m, p);
    double v_nd, v_dt;
    /* Python: two batch-1 reads. On a row-exact backend one 2-row forward
       gives the same bits. */
    if (set_ctx(e, m, p, 0)) return -1;
    bgc_i_encode_base(e, 0, s);
    if (e->row_exact) {
        if (set_ctx(e, &tk, p, 0)) return -1;
        bgc_i_encode_base(e, 1, s);
        bgc_i_forward(e, 0, 2);
        v_nd = (double)e->v[0];
        v_dt = (double)e->v[1];
    } else {
        bgc_i_forward(e, 0, 1);
        v_nd = (double)e->v[0];
        if (set_ctx(e, &tk, p, 0)) return -1;
        bgc_i_encode_base(e, 0, s);
        bgc_i_forward(e, 0, 1);
        v_dt = (double)e->v[0];
    }
    const bgc_match ps = bgc_m_after_game(m, 1, p);
    const double v_dp = bgc_m_over(&ps) ? 1.0 : new_game_mwc(e, &ps, p);   /* "cube" scope */
    if (isnan(v_dp)) return -1;
    if (out) { out->v_nd = v_nd; out->v_dt = v_dt; out->v_dp = v_dp; }
    return py_min(v_dt, v_dp) > v_nd;
}

/* The candidates in e->plays[0..n) of position s: their opp-view values
   (terminal successors overridden by e->mtv) into e->mval, the mover-view
   1 - value into e->cscore; returns the first argmin, -2 on a terminal
   value the MET cannot price. Context (e->tail, e->mtv) set by the caller. */
static int jselect0(bgc_engine *e, int n);

static int mselect0(bgc_engine *e, const BoardState *s, int n) {
    for (int i = 0; i < n; i++) bgc_i_encode_base(e, i, &e->plays[i].resulting_state);
    bgc_i_forward_succ(e, 0, n, s, 0, 0);
    if (is_jan(e)) return ev_rows(e, 0, n) ? -2 : jselect0(e, n);
    int best = 0;
    float best_v = 0.0f;
    for (int i = 0; i < n; i++) {
        const BoardState *rs = &e->plays[i].resulting_state;
        float v = e->v[i];
        if (board_is_game_over(rs)) {
            v = e->mtv[bgc_i_game_result(rs)];
            if (isnan(v)) return -2;
        }
        e->mval[i] = v;
        e->cscore[i] = 1.0 - (double)v;
        if (i == 0 || v < best_v) { best = i; best_v = v; }
    }
    return best;
}

/* Janowski: the candidate's mover-view value into e->cscore (the opp view
   1 - it, fp32, into e->mval), terminal or row `row`; NAN on a NaN. */
static double jscore(bgc_engine *e, int i, int row, int gr) {
    const double v = row >= 0 ? 1.0 - e->vd[row] : e->mtvd[gr];
    e->cscore[i] = v;
    e->mval[i] = (float)(1.0 - v);
    return v;
}

/* JanowskiMatchAgent.choose_checker_action_cubeful's selection over the
   candidates in e->plays (row i = candidate i, values in e->vd): the FIRST
   maximum of the mover-view doubles (strict >). -2 on a NaN. */
static int jselect0(bgc_engine *e, int n) {
    int best = 0;
    double best_v = -INFINITY;
    for (int i = 0; i < n; i++) {
        const BoardState *rs = &e->plays[i].resulting_state;
        const int term = board_is_game_over(rs);
        const double v = jscore(e, i, term ? -1 : i, term ? bgc_i_game_result(rs) : 0);
        if (isnan(v)) return -2;
        if (i == 0 || v > best_v) { best = i; best_v = v; }
    }
    return best;
}

/* choose_checker_action_matchplay: candidates into e->plays, index (first
   argmin), -1 forced pass, -2 error. *n_out = candidate count. Janowski:
   choose_checker_action_cubeful (first argmax of the mover-view values). */
static int mchoose(bgc_engine *e, const BoardState *s, int d1, int d2, const bgc_match *m,
                   int *n_out) {
    *n_out = 0;
    if (!ready(e, s, m) || d1 < 1 || d1 > 6 || d2 < 1 || d2 > 6) return -2;
    const int n = bgc_i_plays(e, s, d1, d2, e->plays);
    *n_out = n;
    if (n == 0) return -1;
    if (bgc_i_ensure_rows(e, n)) return -2;
    const int q = 1 - s->turn;
    if (set_ctx(e, m, q, bgc_m_can_offer(m, q))) return -2;
    set_tv(e, m, s->turn);
    return mselect0(e, s, n);
}

int bgc_match_best_move(bgc_engine *e, const BoardState *s, int d1, int d2,
                        const bgc_match *m, BoardState *out_state, float *out_value) {
    int n;
    const int idx = mchoose(e, s, d1, d2, m, &n);
    if (idx < 0) return idx;
    if (out_state) *out_state = e->plays[idx].resulting_state;
    if (out_value) *out_value = e->mval[idx];
    return idx;
}

int bgc_match_score_moves(bgc_engine *e, const BoardState *s, int d1, int d2,
                          const bgc_match *m, float *opp_values, BoardState *states) {
    int n;
    const int idx = mchoose(e, s, d1, d2, m, &n);
    if (idx == -2) return -2;
    for (int i = 0; i < n; i++) {
        if (opp_values) opp_values[i] = e->mval[i];
        if (states) states[i] = e->plays[i].resulting_state;
    }
    return n;
}

/* ── 1-ply ───────────────────────────────────────────────────────── */

/* The 1-ply memo. value_oneply_checker_matchplay is a pure function of the
 * board (points, bar, off, turn) and the match state (aways, cube owner and
 * value, is_crawford, crawford_done) for the engine's weights and MET, and
 * mvalue1k computes it the same way every time (the same rows in the same
 * batch), so a repeated (board, match) reuses the earlier double bitwise.
 * Every match of a round starts at the same score, so the opening and
 * early positions recur: ~2.6% of an 11pt worker's checker turns (67
 * matches). Cleared by bgc_set_met; the engine's weights never change.
 * Bounded at MM_MAX entries (then no more inserts). */
#define MM_MAX (1L << 17)   /* entries (a table of 2^18 slots, 12.6 MB, at most); a worker's 11pt round looks up ~25k */

/* The key of (X, m); 0 (no memo) for a context that does not fit it. */
static int mm_key(const BoardState *X, const bgc_match *m, uint64_t k[5]) {
    if (m->away[0] < 0 || m->away[0] > 0xffff || m->away[1] < 0 || m->away[1] > 0xffff)
        return 0;
    unsigned char b[32];
    for (int i = 0; i < NUM_POINTS; i++) b[i] = (unsigned char)X->points[i];
    b[24] = (unsigned char)X->bar[0];
    b[25] = (unsigned char)X->bar[1];
    b[26] = (unsigned char)X->off[0];
    b[27] = (unsigned char)X->off[1];
    b[28] = (unsigned char)X->turn;
    b[29] = b[30] = b[31] = 0;
    memcpy(k, b, sizeof b);
    k[4] = (uint64_t)m->away[0] | (uint64_t)m->away[1] << 16 | (uint64_t)m->cube_owner << 32
           | (uint64_t)m->cube_value << 34 | (uint64_t)(m->is_crawford != 0) << 59
           | (uint64_t)(m->crawford_done != 0) << 60 | (uint64_t)1 << 63;     /* nonzero */
    return 1;
}

static uint64_t mm_hash(const uint64_t k[5]) {
    uint64_t h = k[4] * 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; i++) h = (h ^ k[i]) * 0xC2B2AE3D27D4EB4FULL, h ^= h >> 29;
    return h;
}

/* The slot of key k: its entry, or the empty slot where it would go. */
static struct bgc_mmemo *mm_slot(bgc_engine *e, const uint64_t k[5]) {
    for (uint64_t p = mm_hash(k) & (uint64_t)(e->mm_cap - 1);; p = (p + 1) & (uint64_t)(e->mm_cap - 1)) {
        struct bgc_mmemo *s = &e->mm[p];
        if (!s->k[4] || !memcmp(s->k, k, sizeof s->k)) return s;
    }
}

/* ── the shared memo (BGC_MATCH_SHM) ─────────────────────────────
 * The local memo's (key, value) pairs, shared by every engine (process)
 * that opened the same weights file contents with the same library file,
 * backend configuration (bgc_describe: ISA, tiles, delta0, bf16), encoder
 * and MET: one POSIX shm table per such digest, /dev/shm/bgc_mm_<uid>_<hex>.
 * For those the 1-ply value of a key is the same double in every process
 * (mvalue1k: the same rows in the same batch on a row-exact backend), so a
 * value another worker of the round computed is the value this one would
 * compute. Only row-exact backends, and only positions whose two pip counts
 * sum to >= BGC_MATCH_SHM_PIPS with nothing borne off (the first turns of
 * a game: what recurs across the round's matches; the rest stays local).
 *
 * Race-free by construction: a slot goes 0 (empty) -> 1 (claimed by one
 * writer: CAS) -> 2 (key and value written, then a release store) and is
 * never written again; a reader uses a slot only after an acquire load of
 * 2, and skips claimed slots (a value computed twice is the same value).
 * Entries never move or disappear while mapped. The creator ftruncates and
 * then publishes the header magic (release); others wait for it (acquire)
 * and check the digest. The first engine to close unlinks the name (the
 * mappings stay; an engine opened later starts a new table). A killed
 * process can leave its table (32 MB) in /dev/shm: an engine that creates a
 * table first unlinks this user's tables (bgc_mm_<uid>_*) last modified
 * over an hour ago (ms_sweep; a round's table lives for its round, and
 * unlinking a live one changes no value: its users keep the mapping). */
#define MS_MAGIC 0x316d68736d6d6762ULL           /* "bgmmshm1" */
#define MS_SLOTS (1L << 19)                      /* 32 MB; inserts stop at 1/2 */
#define MS_PROBE 64
#define MS_MINPIP 315                            /* BGC_MATCH_SHM_PIPS default (start: 334) */
struct ms_slot { uint64_t state, k[5]; double v; uint64_t pad; };
struct ms_head { uint64_t magic, dig[2], cap, count, pad[3]; };
_Static_assert(sizeof(struct ms_slot) == 64, "ms_slot: one cache line");
_Static_assert(sizeof(struct ms_head) == 64, "ms_head: one cache line");

typedef struct { uint64_t a, b; } ms_hash;
static void msh_add(ms_hash *h, const void *p, size_t n) {
    const unsigned char *c = p;
    for (size_t i = 0; i < n; i++) {
        h->a = (h->a ^ c[i]) * 0x100000001B3ULL;                     /* FNV-1a 64 */
        h->b = (h->b + c[i] + 1) * 0x9E3779B97F4A7C15ULL;
        h->b ^= h->b >> 29;
    }
}

void bgc_i_match_opened(bgc_engine *e, const char *path) {
    const char *v = getenv("BGC_MATCH_SHM");              /* "0": off */
    const char *pp = getenv("BGC_MATCH_SHM_PIPS");
    e->ms_minpip = pp && *pp ? atoi(pp) : MS_MINPIP;
    e->ms_want = e->mm_on && e->row_exact && path && !(v && !strcmp(v, "0"));
    if (!e->ms_want) return;
    ms_hash h = {0xcbf29ce484222325ULL, 0x243F6A8885A308D3ULL};
    FILE *f = fopen(path, "rb");
    if (!f) { e->ms_want = 0; return; }
    unsigned char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) msh_add(&h, buf, n);
    if (ferror(f)) e->ms_want = 0;
    fclose(f);
    e->ms_wdig[0] = h.a;
    e->ms_wdig[1] = h.b;
}

void bgc_i_match_shm_detach(bgc_engine *e) {
    if (!e->ms_map) return;
    munmap(e->ms_map, e->ms_len);
    shm_unlink(e->ms_name);           /* the first to close removes the name */
    e->ms_map = NULL;
}

/* Unlink this user's tables older than an hour (stale: a killed process's). */
static void ms_sweep(void) {
    char pre[40];
    snprintf(pre, sizeof pre, "bgc_mm_%u_", (unsigned)getuid());
    const size_t lp = strlen(pre);
    DIR *d = opendir("/dev/shm");
    if (!d) return;
    const time_t now = time(NULL);
    struct dirent *de;
    while ((de = readdir(d))) {
        if (strncmp(de->d_name, pre, lp) || strlen(de->d_name) != lp + 32) continue;
        char path[300];
        struct stat st;
        snprintf(path, sizeof path, "/dev/shm/%s", de->d_name);
        if (stat(path, &st) || st.st_uid != getuid() || now - st.st_mtime < 3600) continue;
        char nm[300];
        snprintf(nm, sizeof nm, "/%s", de->d_name);
        shm_unlink(nm);
    }
    closedir(d);
}

static void ms_sleep_ms(void) {
    struct timespec ts = {0, 1000000};
    nanosleep(&ts, NULL);
}

static void ms_attach(bgc_engine *e) {
    bgc_i_match_shm_detach(e);
    e->ms_lookups = e->ms_hits = e->ms_puts = 0;
    if (!e->ms_want || !e->met_pre) return;
    ms_hash h = {e->ms_wdig[0], e->ms_wdig[1]};
    msh_add(&h, "bgc-mm-v1", 9);
    msh_add(&h, &e->met_n, sizeof e->met_n);
    msh_add(&h, e->met_pre, (size_t)e->met_n * e->met_n * sizeof(double));
    msh_add(&h, e->met_post, (size_t)e->met_n * sizeof(double));
    const char *d = bgc_describe(e);
    msh_add(&h, d, strlen(d));
    msh_add(&h, e->enc_name, strlen(e->enc_name));
    const int fl[4] = {e->kind, e->row_exact, e->bf16, e->in_size};
    msh_add(&h, fl, sizeof fl);
    if (is_jan(e)) {                             /* the Janowski values depend on x */
        msh_add(&h, &e->jx_const, sizeof e->jx_const);
        msh_add(&h, &e->jx, sizeof e->jx);
    }
    Dl_info di;
    struct stat st;
    if (!dladdr((void *)ms_attach, &di) || !di.dli_fname || stat(di.dli_fname, &st)) return;
    msh_add(&h, di.dli_fname, strlen(di.dli_fname));
    const long long id[5] = {(long long)st.st_dev, (long long)st.st_ino, (long long)st.st_size,
                             (long long)st.st_mtim.tv_sec, (long long)st.st_mtim.tv_nsec};
    msh_add(&h, id, sizeof id);
    snprintf(e->ms_name, sizeof e->ms_name, "/bgc_mm_%u_%016llx%016llx", (unsigned)getuid(),
             (unsigned long long)h.a, (unsigned long long)h.b);
    const size_t len = sizeof(struct ms_head) + (size_t)MS_SLOTS * sizeof(struct ms_slot);
    for (int attempt = 0; attempt < 4; attempt++) {
        int creator = 1;
        int fd = shm_open(e->ms_name, O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd < 0 && errno == EEXIST) {
            creator = 0;
            fd = shm_open(e->ms_name, O_RDWR, 0600);
        }
        if (fd < 0) {
            if (errno == ENOENT) continue;           /* unlinked meanwhile: create it */
            return;
        }
        if (creator) {
            if (ftruncate(fd, (off_t)len)) { close(fd); shm_unlink(e->ms_name); return; }
            ms_sweep();
        } else {
            int ok = 0;
            for (int i = 0; i < 2000 && !ok; i++) {
                if (fstat(fd, &st)) break;
                if ((size_t)st.st_size == len) ok = 1;
                else ms_sleep_ms();
            }
            if (!ok) { close(fd); shm_unlink(e->ms_name); return; }   /* broken: drop it */
        }
        void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) { if (creator) shm_unlink(e->ms_name); return; }
        struct ms_head *hd = p;
        if (creator) {
            hd->dig[0] = h.a;
            hd->dig[1] = h.b;
            hd->cap = MS_SLOTS;
            __atomic_store_n(&hd->magic, MS_MAGIC, __ATOMIC_RELEASE);
        } else {
            int ok = 0;
            for (int i = 0; i < 2000 && !ok; i++) {
                if (__atomic_load_n(&hd->magic, __ATOMIC_ACQUIRE) == MS_MAGIC) ok = 1;
                else ms_sleep_ms();
            }
            if (!ok || hd->dig[0] != h.a || hd->dig[1] != h.b || hd->cap != MS_SLOTS) {
                munmap(p, len);
                return;
            }
        }
        e->ms_map = p;
        e->ms_len = len;
        return;
    }
}

/* Positions the shared table holds: nothing borne off, pip sum >= ms_minpip. */
static int ms_gate(const bgc_engine *e, const BoardState *X) {
    if (!e->ms_map || X->off[0] || X->off[1]) return 0;
    int pips = 25 * (X->bar[0] + X->bar[1]);
    for (int i = 0; i < NUM_POINTS; i++) {
        const int v = X->points[i];
        pips += v > 0 ? v * (i + 1) : -v * (24 - i);
    }
    return pips >= e->ms_minpip;
}

static int ms_get(bgc_engine *e, const uint64_t k[5], double *v) {
    struct ms_slot *t = (struct ms_slot *)((struct ms_head *)e->ms_map + 1);
    e->ms_lookups++;
    uint64_t p = mm_hash(k) & (uint64_t)(MS_SLOTS - 1);
    for (int i = 0; i < MS_PROBE; i++, p = (p + 1) & (uint64_t)(MS_SLOTS - 1)) {
        struct ms_slot *s = &t[p];
        const uint64_t st = __atomic_load_n(&s->state, __ATOMIC_ACQUIRE);
        if (st == 0) return 0;
        if (st == 2 && !memcmp(s->k, k, sizeof s->k)) {
            *v = s->v;
            e->ms_hits++;
            return 1;
        }
    }
    return 0;
}

static void ms_put(bgc_engine *e, const uint64_t k[5], double v) {
    struct ms_head *hd = e->ms_map;
    struct ms_slot *t = (struct ms_slot *)(hd + 1);
    if (__atomic_load_n(&hd->count, __ATOMIC_RELAXED) >= (uint64_t)MS_SLOTS / 2) return;
    uint64_t p = mm_hash(k) & (uint64_t)(MS_SLOTS - 1);
    for (int i = 0; i < MS_PROBE; i++, p = (p + 1) & (uint64_t)(MS_SLOTS - 1)) {
        struct ms_slot *s = &t[p];
        uint64_t st = __atomic_load_n(&s->state, __ATOMIC_ACQUIRE);
        if (st == 0) {
            if (__atomic_compare_exchange_n(&s->state, &st, 1, 0, __ATOMIC_ACQ_REL,
                                            __ATOMIC_ACQUIRE)) {
                memcpy(s->k, k, sizeof s->k);
                s->v = v;
                __atomic_store_n(&s->state, 2, __ATOMIC_RELEASE);
                __atomic_fetch_add(&hd->count, 1, __ATOMIC_RELAXED);
                e->ms_puts++;
                return;
            }                                    /* st: what the other writer set */
        }
        if (st == 2 && !memcmp(s->k, k, sizeof s->k)) return;     /* present */
    }
}

static void mm_put_local(bgc_engine *e, const uint64_t k[5], double v);

static int mm_get(bgc_engine *e, const BoardState *X, const bgc_match *m, double *v) {
    uint64_t k[5];
    if (!e->mm_on || (!e->mm && !e->ms_map) || !mm_key(X, m, k)) return 0;
    if (e->mm) {
        e->mm_lookups++;
        const struct bgc_mmemo *s = mm_slot(e, k);
        if (s->k[4]) {
            e->mm_hits++;
            *v = s->v;
            return 1;
        }
    }
    if (ms_gate(e, X) && ms_get(e, k, v)) {     /* another worker's value: keep it here too */
        mm_put_local(e, k, *v);
        return 1;
    }
    return 0;
}

static void mm_put(bgc_engine *e, const BoardState *X, const bgc_match *m, double v) {
    uint64_t k[5];
    if (!e->mm_on || !mm_key(X, m, k)) return;
    if (ms_gate(e, X)) ms_put(e, k, v);
    mm_put_local(e, k, v);
}

static void mm_put_local(bgc_engine *e, const uint64_t k[5], double v) {
    if (e->mm_n >= MM_MAX) return;
    if (2 * (e->mm_n + 1) > e->mm_cap) {         /* grow (amortized; rehash) */
        long cap = e->mm_cap ? 2 * e->mm_cap : 4096;
        struct bgc_mmemo *old = e->mm, *nw = calloc((size_t)cap, sizeof(*nw));
        if (!nw) return;
        long oc = e->mm_cap;
        e->mm = nw;
        e->mm_cap = cap;
        for (long i = 0; i < oc; i++)
            if (old[i].k[4]) *mm_slot(e, old[i].k) = old[i];
        free(old);
    }
    struct bgc_mmemo *s = mm_slot(e, k);
    if (s->k[4]) return;
    memcpy(s->k, k, sizeof s->k);
    s->v = v;
    e->mm_n++;
}

/* JanowskiMatchAgent.value_oneply's reduction of expand21's slot 0 (values
   in e->vd, terminals in e->mtvd): per roll the mover's best value, max over
   the replies of 1 - vd (the opponent on roll) or the terminal MWC; a forced
   pass is the pass row's 1 - vd; the 21 summed in dice order (forced passes
   in place), no clamp, no all-pass rule. *err = 0 unless a NaN. */
static double jreduce1(bgc_engine *e, int *err) {
    const int *doff = e->cdoff;
    const unsigned char *pass = e->cpass;
    double best[21];
    for (int k = 0; k < 21; k++) {
        if (pass[k]) {
            best[k] = 1.0 - e->vd[e->rep_row[doff[k]]];
            continue;
        }
        double mx = -INFINITY;
        for (int r = doff[k]; r < doff[k + 1]; r++) {
            const double v = e->rep_row[r] >= 0 ? 1.0 - e->vd[e->rep_row[r]]
                                                 : e->mtvd[e->rep_gr[r]];
            if (isnan(v)) return NAN;
            if (v > mx) mx = v;
        }
        best[k] = mx;
    }
    *err = 0;
    return bgc_i_dice_sum(best, pass, 0, BGC_FLIP_NONE);
}

/* value_oneply_checker_matchplay(X, m), cube_fold off (Janowski:
   value_oneply). keep: expand21's (slot 0); the batch stays valid until the
   next forward. *err on an error (NAN returned). */
static double mvalue1k(bgc_engine *e, const BoardState *X, const bgc_match *m, int keep,
                       int *err) {
    *err = 1;
    if (!ready(e, X, m) || board_is_game_over(X)) return NAN;
    const int q = 1 - X->turn;
    if (set_ctx(e, m, q, bgc_m_can_offer(m, q))) return NAN;
    set_tv(e, m, X->turn);
    int nrow = 0, nrep = 0;
    if (bgc_i_expand21(e, X, 0, keep, &nrow, &nrep)) return NAN;
    bgc_i_forward_v1(e, 0, nrow, X, 0, 0);
    if (is_jan(e)) {
        if (ev_rows(e, 0, nrow)) return NAN;
        return jreduce1(e, err);
    }
    PROF_T0(t0);
    const int *doff = e->cdoff;
    const unsigned char *pass = e->cpass;
    double mins[21];
    int all_pass = 1;
    for (int k = 0; k < 21; k++) {
        if (pass[k]) {               /* the pass row (X, turn switched): its value */
            mins[k] = (double)e->v[e->rep_row[doff[k]]];
            continue;
        }
        all_pass = 0;
        double mn = INFINITY;
        for (int r = doff[k]; r < doff[k + 1]; r++) {
            double val;
            if (e->rep_row[r] >= 0) {
                val = (double)e->v[e->rep_row[r]];
            } else {
                const float tv = e->mtv[e->rep_gr[r]];
                if (isnan(tv)) return NAN;
                val = (double)tv;
            }
            if (val < mn) mn = val;
        }
        mins[k] = mn;
    }
    *err = 0;
    double r;
    if (all_pass) {
        r = 1.0 - mins[0];                       /* the forced value, unclamped */
    } else {
        /* total += prob * (1 - min), forced passes in place: ungrouped */
        const double total = bgc_i_dice_sum(mins, pass, 0, BGC_FLIP_ONE_MINUS);
        r = py_min(1.0, py_max(0.0, total));
    }
    PROF_ADD(t0, BGC_PT_REDUCE, nrow);
    return r;
}

double bgc_match_value_1ply(bgc_engine *e, const BoardState *s, const bgc_match *m) {
    int err;
    double v;
    if (ready(e, s, m) && mm_get(e, s, m, &v)) return v;
    PROF_ENTER(BGC_PC_V1);
    v = mvalue1k(e, s, m, -1, &err);
    PROF_LEAVE(1);
    if (!err) mm_put(e, s, m, v);
    return v;
}

/* The 0-ply scores of the roll's candidates from mvalue1k's batch
   (candidate i = reply e->pidx[i] of slot 0): mselect0's values. */
static int mscore_from_replies(bgc_engine *e, int n) {
    if (is_jan(e)) {                             /* jselect0 on the replies */
        int best = 0;
        double best_v = -INFINITY;
        for (int i = 0; i < n; i++) {
            const int r = e->pidx[i];
            const double v = jscore(e, i, e->rep_row[r], e->rep_gr[r]);
            if (isnan(v)) return -2;
            if (i == 0 || v > best_v) { best = i; best_v = v; }
        }
        return best;
    }
    int best = 0;
    float best_v = 0.0f;
    for (int i = 0; i < n; i++) {
        const int r = e->pidx[i];
        const float v = e->rep_row[r] >= 0 ? e->v[e->rep_row[r]] : e->mtv[e->rep_gr[r]];
        e->mval[i] = v;
        e->cscore[i] = 1.0 - (double)v;
        if (i == 0 || v < best_v) { best = i; best_v = v; }
    }
    return best;
}

/* mvalue1k(X) and mchoose(X, d1, d2) from one expansion on a row-exact
   backend (the roll's candidates are replies with the same match block and
   terminal table), the two calls otherwise. *v1 = the 1-ply value. Returns
   mchoose's index (e->plays / e->mval / e->cscore), -1 pass, -2 error. */
static int mvalue1_play(bgc_engine *e, const BoardState *X, const bgc_match *m, int d1, int d2,
                        double *v1) {
    int err, n;
    if (ready(e, X, m) && mm_get(e, X, m, v1)) {   /* memo hit: the 0-ply play alone */
        PROF_ENTER(BGC_PC_SEL);
        const int idx = mchoose(e, X, d1, d2, m, &n);
        PROF_LEAVE(1);
        return idx;
    }
    const int fast = e && e->row_exact && !e->bf16 && d1 >= 1 && d1 <= 6 && d2 >= 1 && d2 <= 6;
    {
        PROF_ENTER(BGC_PC_V1);
        *v1 = mvalue1k(e, X, m, fast ? bgc_i_dice_index(d1, d2) : -1, &err);
        PROF_LEAVE(1);
    }
    if (err) return -2;
    mm_put(e, X, m, *v1);
    int idx = -3;
    if (fast) {
        PROF_ENTER(BGC_PC_SEL);
        n = bgc_i_roll_replies(e, X, d1, d2);
        idx = n >= 0 ? mscore_from_replies(e, n) : n;
        PROF_LEAVE(1);
    }
    if (idx == -3) {
        PROF_ENTER(BGC_PC_SEL);
        idx = mchoose(e, X, d1, d2, m, &n);
        PROF_LEAVE(1);
    }
    return idx;
}

int bgc_match_value_1ply_play(bgc_engine *e, const BoardState *s, int d1, int d2,
                              const bgc_match *m, double *v1, BoardState *out_state) {
    double v = NAN;
    const int idx = mvalue1_play(e, s, m, d1, d2, &v);
    if (v1) *v1 = idx == -2 ? NAN : v;
    if (idx >= 0 && out_state) *out_state = e->plays[idx].resulting_state;
    return idx;
}

int bgc_match_shm_stats(const bgc_engine *e, long *lookups, long *hits, long *puts) {
    if (!e || !bgc_i_is_match(e)) return -1;
    if (lookups) *lookups = e->ms_lookups;
    if (hits) *hits = e->ms_hits;
    if (puts) *puts = e->ms_puts;
    return e->ms_map != NULL;
}

int bgc_match_memo_stats(const bgc_engine *e, long *lookups, long *hits, long *entries) {
    if (!e || !bgc_i_is_match(e)) return -1;
    if (lookups) *lookups = e->mm_lookups;
    if (hits) *hits = e->mm_hits;
    if (entries) *entries = e->mm_n;
    return e->mm_on;
}
