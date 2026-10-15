/*
 * puretd_janowski.c -- Prob5Jan: a cubeless prob5 net under the Janowski
 * money cube layer (BGC_KIND_PROB5JAN), and the bearoff tables it reads.
 *
 * The Python modules are the reference, ported function for function:
 *   janowski.py         bgc_jan_equity, _folded, _cubeless_equity, _fold,
 *                       _money_cube_action
 *   bearoff.py          bgc_bearoff (the flat file of export_c_tables),
 *                       exact_probs5, _hazards, _race_dp, _ts_wnd,
 *                       _exact_tier, race_equity, race_cube_values,
 *                       ts_win_prob
 *   janowski_agent.py   JanowskiAgent at its defaults (cube_fold,
 *                       fold_0ply, bearoff_exact iff a table is attached,
 *                       board_sanity per bgc_set_jan): _succ_values
 *                       (row_leaf), _pick, _screen_0ply, _oneply,
 *                       _value_oneply, _static_cube, _cube_values,
 *                       offer_double, respond_to_double,
 *                       choose_checker_action_cubeful.
 *
 * Numerics: double throughout, every expression in Python's evaluation
 * order (left to right; a * b / c is (a * b) / c), no FMA contraction.
 * Sums are sequential; bearoff.exact_probs5's np.dot (BLAS order) is the
 * one place the order may differ. Net probabilities are fp32 (sanity
 * mask, then the prob5 clamps) widened to double.
 */

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#define _POSIX_C_SOURCE 200809L

#include "puretd.h"
#include "puretd_internal.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define JAN_EPS 1e-7            /* janowski._EPS */
#define JAN_DEFAULT_X 0.76      /* janowski.DEFAULT_X */
#define BO_R 24                 /* bearoff.R_MAX */
#define BO_N 15                 /* checkers a side */
#define HC_SLOTS 1024           /* hazard-DP cache per engine */

/* _DICE_P: d1 = 1..6, d2 = d1..6. */
static const double DP21[21] = {
    1.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0,
    1.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0,
    1.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0,
    1.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0,
    1.0 / 36.0, 2.0 / 36.0,
    1.0 / 36.0,
};

static const int FLIP[3] = {PERSP_CENTERED, PERSP_THEIRS, PERSP_MINE};

/* ── conversion (janowski.py) ──────────────────────────────────── */

double bgc_jan_cubeless_equity(const double p[5]) {
    return (2.0 * p[0] + p[1] + p[2] - p[3] - p[4]) - 1.0;
}

/* Segment of _piecewise_linear: y0 + (y1 - y0) * (p - x0) / (x1 - x0). */
static double seg(double p, double x0, double y0, double x1, double y1) {
    return y0 + (y1 - y0) * (p - x0) / (x1 - x0);
}

/* The first segment with p < its right knot, else the last. */
static double pwl(double p, const double *kx, const double *ky, int nk) {
    for (int i = 0; i < nk - 2; i++)
        if (p < kx[i + 1]) return seg(p, kx[i], ky[i], kx[i + 1], ky[i + 1]);
    return seg(p, kx[nk - 2], ky[nk - 2], kx[nk - 1], ky[nk - 1]);
}

double bgc_jan_equity(const double p[5], int persp, double x, int jacoby) {
    if (persp < PERSP_CENTERED || persp > PERSP_THEIRS) return NAN;
    const double p0 = p[0];
    const double win = p0 > JAN_EPS ? p0 : JAN_EPS;
    const double lose = 1.0 - p0 > JAN_EPS ? 1.0 - p0 : JAN_EPS;
    const double W = 1.0 + (p[1] + p[2]) / win;
    const double L = 1.0 + (p[3] + p[4]) / lose;
    const double dead = jacoby && persp == PERSP_CENTERED ? 2.0 * p0 - 1.0
                                                          : bgc_jan_cubeless_equity(p);
    if (p0 <= JAN_EPS || p0 >= 1.0 - JAN_EPS) return dead;
    const double denom = W + L + 0.5;
    const double tp = (L - 0.5) / denom;
    const double cp = (L + 1.0) / denom;
    double live;
    if (persp == PERSP_CENTERED) {
        const double kx[4] = {0.0, tp, cp, 1.0};
        const double ky[4] = {jacoby ? -1.0 : -L, -1.0, 1.0, jacoby ? 1.0 : W};
        live = pwl(p0, kx, ky, 4);
    } else if (persp == PERSP_MINE) {
        const double kx[3] = {0.0, cp, 1.0}, ky[3] = {-L, 1.0, W};
        live = pwl(p0, kx, ky, 3);
    } else {
        const double kx[3] = {0.0, tp, 1.0}, ky[3] = {-L, -1.0, W};
        live = pwl(p0, kx, ky, 3);
    }
    return (1.0 - x) * dead + x * live;
}

double bgc_jan_fold(double nd, double dt) {
    return (dt >= nd && 1.0 >= nd) ? (dt < 1.0 ? dt : 1.0) : nd;
}

double bgc_jan_equity_folded(const double p[5], int persp, double x, int jacoby) {
    const double nd = bgc_jan_equity(p, persp, x, jacoby);
    if (persp != PERSP_CENTERED && persp != PERSP_MINE) return nd;
    return bgc_jan_fold(nd, 2.0 * bgc_jan_equity(p, PERSP_THEIRS, x, jacoby));
}

/* TDAgent._money_cube_decision */
static int money_double(double nd, double dt, int persp, int jacoby) {
    if (dt > 1.0) {
        if (jacoby && persp == PERSP_CENTERED) return 1;
        return 1.0 >= nd;
    }
    return dt > nd;
}

int bgc_jan_money_cube_action(double nd, double dt, int persp, int jacoby, int *take) {
    if (take) *take = dt <= 1.0;
    return money_double(nd, dt, persp, jacoby);
}

/* ── bearoff tables (bearoff.py) ───────────────────────────────── */

struct bgc_bearoff {
    void *map;
    size_t len;
    int n_os, n_ts, ts_max;
    const unsigned char *os_counts, *f, *g, *ts_counts, *P, *V;
    int *os_idx, *ts_idx;           /* combinatorial rank -> file row */
    int nc[7][BO_N + 1];            /* nc[k][s]: k-tuples with sum <= s = C(s + k, k) */
};

static inline double ldf(const unsigned char *b, size_t i) {
    float v;
    memcpy(&v, b + 4 * i, sizeof v);
    return (double)v;
}

/* Lexicographic rank of c[0..5] among the 6-tuples with sum <= nmax. */
static int bo_rank(const bgc_bearoff *t, const int c[6], int nmax) {
    int rank = 0, rem = nmax;
    for (int i = 0; i < 6; i++) {
        rank += t->nc[6 - i][rem] - t->nc[6 - i][rem - c[i]];
        rem -= c[i];
    }
    return rank;
}

static int rank_table(const bgc_bearoff *t, const unsigned char *counts, int n, int nmax,
                      int *idx) {
    for (int i = 0; i < n; i++) idx[i] = -1;
    for (int i = 0; i < n; i++) {
        int c[6], s = 0;
        for (int j = 0; j < 6; j++) { c[j] = counts[6 * i + j]; s += c[j]; }
        if (s > nmax) return -1;
        int r = bo_rank(t, c, nmax);
        if (r < 0 || r >= n || idx[r] >= 0) return -1;
        idx[r] = i;
    }
    return 0;
}

bgc_bearoff *bgc_bearoff_open(const char *path, char *err, size_t errlen) {
    if (err && errlen) err[0] = 0;
#define BO_FAIL(...) do { if (err && errlen) snprintf(err, errlen, __VA_ARGS__); goto fail; } while (0)
    bgc_bearoff *t = calloc(1, sizeof(*t));
    int fd = -1;
    if (!t) { if (err && errlen) snprintf(err, errlen, "bgc_bearoff_open: out of memory"); return NULL; }
    const uint32_t one = 1;
    if (*(const unsigned char *)&one != 1) BO_FAIL("bgc_bearoff_open: big-endian host");
    fd = path ? open(path, O_RDONLY) : -1;
    if (fd < 0) BO_FAIL("bgc_bearoff_open: cannot open %s", path ? path : "(null)");
    struct stat st;
    if (fstat(fd, &st) || st.st_size < 24) BO_FAIL("bgc_bearoff_open: %s: too short", path);
    t->len = (size_t)st.st_size;
    t->map = mmap(NULL, t->len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    fd = -1;
    if (t->map == MAP_FAILED) { t->map = NULL; BO_FAIL("bgc_bearoff_open: %s: mmap failed", path); }
    const unsigned char *b = t->map;
    uint32_t h[6];
    memcpy(h, b, sizeof h);
    if (memcmp(b, "BGBO", 4) || h[1] != 1)
        BO_FAIL("bgc_bearoff_open: %s: not a BGBO version 1 file (bearoff.export_c_tables)", path);
    for (int k = 0; k <= 6; k++)
        for (int s = 0; s <= BO_N; s++)
            t->nc[k][s] = k == 0 ? 1 : t->nc[k - 1][s] + (s ? t->nc[k][s - 1] : 0);
    t->n_os = (int)h[2];
    t->ts_max = (int)h[5];
    t->n_ts = (int)h[4];
    if (t->n_os != t->nc[6][BO_N] || h[3] != BO_R || t->ts_max < 1 || t->ts_max > BO_N
        || t->n_ts != t->nc[6][t->ts_max])
        BO_FAIL("bgc_bearoff_open: %s: unexpected sizes (n_os %u, R %u, n_ts %u, ts_max %u)",
                path, h[2], h[3], h[4], h[5]);
    const size_t nos = (size_t)t->n_os, nts = (size_t)t->n_ts;
    const size_t need = 24 + nos * 6 + 2 * nos * BO_R * 4 + nts * 6 + nts * nts * 4
                        + nts * nts * 12;
    if (t->len != need) BO_FAIL("bgc_bearoff_open: %s: %zu bytes, expected %zu", path, t->len, need);
    size_t o = 24;
    t->os_counts = b + o; o += nos * 6;
    t->f = b + o;         o += nos * BO_R * 4;
    t->g = b + o;         o += nos * BO_R * 4;
    t->ts_counts = b + o; o += nts * 6;
    t->P = b + o;         o += nts * nts * 4;
    t->V = b + o;
    t->os_idx = malloc(nos * sizeof(int));
    t->ts_idx = malloc(nts * sizeof(int));
    if (!t->os_idx || !t->ts_idx) BO_FAIL("bgc_bearoff_open: out of memory");
    if (rank_table(t, t->os_counts, t->n_os, BO_N, t->os_idx)
        || rank_table(t, t->ts_counts, t->n_ts, t->ts_max, t->ts_idx))
        BO_FAIL("bgc_bearoff_open: %s: position lists are not the complete sets", path);
    return t;
fail:
    if (fd >= 0) close(fd);
    bgc_bearoff_close(t);
    return NULL;
#undef BO_FAIL
}

void bgc_bearoff_close(bgc_bearoff *t) {
    if (!t) return;
    if (t->map) munmap(t->map, t->len);
    free(t->os_idx);
    free(t->ts_idx);
    free(t);
}

/* A position's bearoff view: home counts by distance 1..6 of the player
   on roll (me) and the opponent, their off counts, bearoff class. */
typedef struct { int me[6], opp[6], off_me, off_opp, cls; } bo_view;

static int sum6(const int c[6]) { return c[0] + c[1] + c[2] + c[3] + c[4] + c[5]; }

/* bearoff.is_bearoff_class + home_counts of a BoardState. */
static void view_state(const BoardState *s, bo_view *v) {
    const int me = s->turn, op = 1 - s->turn;
    v->cls = !s->bar[WHITE] && !s->bar[BLACK];
    for (int i = 6; i < 24 && v->cls; i++) if (s->points[i] > 0) v->cls = 0;
    for (int i = 0; i < 18 && v->cls; i++) if (s->points[i] < 0) v->cls = 0;
    for (int i = 0; i < 6; i++) {
        const int w = s->points[i] > 0 ? s->points[i] : 0;
        const int bl = s->points[23 - i] < 0 ? -s->points[23 - i] : 0;
        v->me[i] = me == WHITE ? w : bl;
        v->opp[i] = op == WHITE ? w : bl;
    }
    v->off_me = s->off[me];
    v->off_opp = s->off[op];
    if (sum6(v->me) > BO_N || sum6(v->opp) > BO_N) v->cls = 0;
}

/* janowski_agent._feature_counts of one point's thermometer. */
static inline int therm_count(const float *f) {
    return (int)rint((double)f[0] + (double)f[1] + (double)f[2] + 2.0 * (double)f[3]);
}

/* The same from a perspective196 row (_bearoff_class_features): the me
   block [0, 98), the opponent's [98, 196) in the same frame. */
static void view_row(const float *f, bo_view *v) {
    v->cls = rint((double)f[96] * 2.0) == 0.0 && rint((double)f[194] * 2.0) == 0.0;
    for (int k = 6; k < 24 && v->cls; k++) if (therm_count(f + 4 * k)) v->cls = 0;
    for (int k = 0; k < 18 && v->cls; k++) if (therm_count(f + 98 + 4 * k)) v->cls = 0;
    if (!v->cls) return;
    for (int i = 0; i < 6; i++) {
        v->me[i] = therm_count(f + 4 * i);
        v->opp[i] = therm_count(f + 98 + 4 * (23 - i));
    }
    v->off_me = (int)rint((double)f[97] * 15.0);
    v->off_opp = (int)rint((double)f[195] * 15.0);
    if (sum6(v->me) > BO_N || sum6(v->opp) > BO_N) v->cls = 0;
}

static inline int os_row(const bgc_bearoff *t, const int c[6]) {
    return t->os_idx[bo_rank(t, c, BO_N)];
}

static void os_dist(const bgc_bearoff *t, const unsigned char *tab, int row, double d[BO_R]) {
    for (int r = 0; r < BO_R; r++) d[r] = ldf(tab, (size_t)row * BO_R + r);
}

static void cumsum(const double *a, double *c) {
    c[0] = a[0];
    for (int r = 1; r < BO_R; r++) c[r] = c[r - 1] + a[r];
}

/* bearoff.exact_probs5 (np.dot as a sequential sum). */
static void exact_probs5(const bgc_bearoff *t, const bo_view *v, double p[5]) {
    double fm[BO_R], gm[BO_R], fo[BO_R], go[BO_R], Fo[BO_R], Go[BO_R], Gm[BO_R];
    const int im = os_row(t, v->me), io = os_row(t, v->opp);
    os_dist(t, t->f, im, fm);
    os_dist(t, t->g, im, gm);
    os_dist(t, t->f, io, fo);
    os_dist(t, t->g, io, go);
    cumsum(fo, Fo);
    cumsum(go, Go);
    cumsum(gm, Gm);
    double s = 0.0;
    for (int r = 0; r < BO_R; r++) s += fm[r] * (r ? 1.0 - Fo[r - 1] : 1.0);
    p[0] = s;
    p[1] = 0.0;
    if (v->off_opp == 0) {
        s = 0.0;
        for (int r = 0; r < BO_R; r++) s += fm[r] * (r ? 1.0 - Go[r - 1] : 1.0);
        p[1] = s;
    }
    p[2] = 0.0;
    p[3] = 0.0;
    if (v->off_me == 0) {
        s = 0.0;
        for (int r = 0; r < BO_R; r++) s += fo[r] * (1.0 - Gm[r]);
        p[3] = s;
    }
    p[4] = 0.0;
}

/* bearoff._hazards, then np.round(h, 12) = rint(h * 1e12) / 1e12. */
static void hazards(const double f[BO_R], double h[BO_R]) {
    double F[BO_R];
    cumsum(f, F);
    for (int k = 0; k < BO_R; k++) {
        const double surv = 1.0 - (k > 0 ? F[k - 1] : 0.0);
        double x = 1.0;
        if (!(surv <= 1e-12)) {
            const double q = f[k] / surv;
            x = q < 1.0 ? q : 1.0;
        }
        h[k] = rint(x * 1e12) / 1e12;
    }
}

/* bearoff._race_dp's memo: side x = 0 (A, on roll at the root) or 1. */
typedef struct {
    double h[2][BO_R];
    double v[2][BO_R + 1][BO_R + 1][3];
    unsigned char set[2][BO_R + 1][BO_R + 1][3];
} race_memo;

static double race_V(race_memo *m, int x, int k, int j, int c);

static double race_W(race_memo *m, int x, int k, int j, int c) {
    const double h = k < BO_R ? m->h[x][k] : 1.0;
    if (h >= 1.0) return 1.0;
    return h - (1.0 - h) * race_V(m, 1 - x, j, k + 1, FLIP[c]);
}

static double race_V(race_memo *m, int x, int k, int j, int c) {
    if (m->set[x][k][j][c]) return m->v[x][k][j][c];
    const double nd = race_W(m, x, k, j, c);
    double v = nd;
    if (c != PERSP_THEIRS) {
        const double dt = 2.0 * race_W(m, x, k, j, PERSP_THEIRS);
        const double cap = 1.0 < dt ? 1.0 : dt;
        v = cap > nd ? cap : nd;
    }
    m->set[x][k][j][c] = 1;
    m->v[x][k][j][c] = v;
    return v;
}

/* _race_dp(hm, ho, c) for c = 0, 1, 2 (one memo: the same values). */
static void race_dp3(const bgc_bearoff *t, int im, int io, double out[3]) {
    race_memo *m = malloc(sizeof(*m));
    double f[BO_R];
    if (!m) { out[0] = out[1] = out[2] = NAN; return; }
    memset(m->set, 0, sizeof(m->set));
    os_dist(t, t->f, im, f);
    hazards(f, m->h[0]);
    os_dist(t, t->f, io, f);
    hazards(f, m->h[1]);
    for (int c = 0; c < 3; c++) out[c] = race_W(m, 0, 0, 0, c);
    free(m);
}

/* bearoff._ts_wnd(ia, ib): window-free values, one dice pass over V with
   the legal plays of the parked one-sided board. plays: MAX_PLAYS. */
static void ts_wnd(const bgc_bearoff *t, int ia, int ib, Play *plays, double wnd[3]) {
    BoardState st;
    memset(&st, 0, sizeof st);
    for (int j = 0; j < 6; j++) st.points[j] = t->ts_counts[6 * ia + j];
    st.points[23] = -15;
    st.turn = WHITE;
    const int zero = t->ts_idx[0];
    const size_t vb = (size_t)ib * t->n_ts * 3;
    wnd[0] = wnd[1] = wnd[2] = 0.0;
    int k = 0;
    for (int d1 = 1; d1 <= 6; d1++) {
        for (int d2 = d1; d2 <= 6; d2++, k++) {
            const double pd = DP21[k];
            const int n = get_legal_plays(&st, d1, d2, plays, MAX_PLAYS);
            double m[3] = {INFINITY, INFINITY, INFINITY};
            int clears = 0;
            for (int i = 0; i < n; i++) {
                int c[6];
                for (int j = 0; j < 6; j++) {
                    const int q = plays[i].resulting_state.points[j];
                    c[j] = q > 0 ? q : 0;
                }
                const int s = t->ts_idx[bo_rank(t, c, t->ts_max)];
                if (s == zero) { clears = 1; break; }
                for (int q = 0; q < 3; q++) {
                    const double val = ldf(t->V, vb + (size_t)s * 3 + q);
                    if (val < m[q]) m[q] = val;
                }
            }
            if (clears) {
                wnd[0] += pd; wnd[1] += pd; wnd[2] += pd;
                continue;
            }
            wnd[PERSP_CENTERED] += pd * (-m[PERSP_CENTERED]);
            wnd[PERSP_MINE] += pd * (-m[PERSP_THEIRS]);
            wnd[PERSP_THEIRS] += pd * (-m[PERSP_MINE]);
        }
    }
}

enum { TIER_NONE = 0, TIER_TS = 1, TIER_HAZ = 2 };

/* bearoff._exact_tier: TS with (a, b) = TS rows of (me, opp); HAZ with
   (a, b) = one-sided rows; NONE. */
static int exact_tier(const bgc_bearoff *t, const bo_view *v, int *a, int *b) {
    if (!v->cls || v->off_me >= BO_N || v->off_opp >= BO_N) return TIER_NONE;
    if (sum6(v->me) <= t->ts_max && sum6(v->opp) <= t->ts_max) {
        *a = t->ts_idx[bo_rank(t, v->me, t->ts_max)];
        *b = t->ts_idx[bo_rank(t, v->opp, t->ts_max)];
        return TIER_TS;
    }
    if (v->off_me == 0 || v->off_opp == 0) return TIER_NONE;    /* gammons possible */
    *a = os_row(t, v->me);
    *b = os_row(t, v->opp);
    double s = 0.0;
    for (int r = 0; r < BO_R; r++) s += ldf(t->f, (size_t)*a * BO_R + r);
    return s < 0.5 ? TIER_NONE : TIER_HAZ;
}

static inline double ts_V(const bgc_bearoff *t, int a, int b, int c) {
    return ldf(t->V, ((size_t)a * t->n_ts + b) * 3 + c);
}

/* bearoff.ts_win_prob: 1 and *p, or 0 (None). */
static int ts_P(const bgc_bearoff *t, const bo_view *v, double *p) {
    if (!v->cls || sum6(v->me) > t->ts_max || sum6(v->opp) > t->ts_max) return 0;
    *p = ldf(t->P, (size_t)t->ts_idx[bo_rank(t, v->me, t->ts_max)] * t->n_ts
                   + t->ts_idx[bo_rank(t, v->opp, t->ts_max)]);
    return 1;
}

/* race_equity of a HAZ tier from its _race_dp values. */
static double haz_equity(const double dp[3], int persp, int fold) {
    const double nd = dp[persp];
    if (!(fold && persp != PERSP_THEIRS)) return nd;
    const double dt = 2.0 * dp[PERSP_THEIRS];
    const double cap = 1.0 < dt ? 1.0 : dt;
    return cap > nd ? cap : nd;
}

/* ── the tables alone ──────────────────────────────────────────── */

static int bo_args_ok(const bgc_bearoff *t, const BoardState *s) {
    return t && s && (s->turn == WHITE || s->turn == BLACK);
}

int bgc_bo_exact_probs5(const bgc_bearoff *t, const BoardState *s, double p[5]) {
    if (!bo_args_ok(t, s) || !p) return -1;
    bo_view v;
    view_state(s, &v);
    if (!v.cls) return 0;
    exact_probs5(t, &v, p);
    return 1;
}

int bgc_bo_ts_win_prob(const bgc_bearoff *t, const BoardState *s, double *p) {
    if (!bo_args_ok(t, s) || !p) return -1;
    bo_view v;
    view_state(s, &v);
    return ts_P(t, &v, p);
}

/* (nd, dt) of a tier (_tier_cube_values). plays: ts_wnd scratch or NULL
   (allocated here). 0, -1 on allocation failure. */
static int tier_cube_values(const bgc_bearoff *t, int tier, int a, int b, int persp,
                            Play *plays, double *nd, double *dt) {
    double w[3];
    if (tier == TIER_TS) {
        Play *own = plays ? NULL : malloc(MAX_PLAYS * sizeof(Play));
        if (!plays && !own) return -1;
        ts_wnd(t, a, b, plays ? plays : own, w);
        free(own);
    } else {
        race_dp3(t, a, b, w);
        if (isnan(w[0])) return -1;
    }
    *nd = w[persp];
    *dt = 2.0 * w[PERSP_THEIRS];
    return 0;
}

int bgc_bo_race_cube_values(const bgc_bearoff *t, const BoardState *s, int persp,
                            double *nd, double *dt) {
    if (!bo_args_ok(t, s) || !nd || !dt || persp < 0 || persp > 2) return -1;
    bo_view v;
    view_state(s, &v);
    int a, b;
    const int tier = exact_tier(t, &v, &a, &b);
    if (tier == TIER_NONE) return 0;
    return tier_cube_values(t, tier, a, b, persp, NULL, nd, dt) ? -1 : 1;
}

int bgc_bo_race_equity(const bgc_bearoff *t, const BoardState *s, int persp, int fold,
                       double *out) {
    if (!bo_args_ok(t, s) || !out || persp < 0 || persp > 2) return -1;
    bo_view v;
    view_state(s, &v);
    int a, b;
    const int tier = exact_tier(t, &v, &a, &b);
    if (tier == TIER_NONE) return 0;
    const int f = fold && persp != PERSP_THEIRS;
    if (tier == TIER_TS && (f || persp == PERSP_THEIRS)) {
        *out = ts_V(t, a, b, persp);
        return 1;
    }
    double nd, dt;
    if (tier_cube_values(t, tier, a, b, persp, NULL, &nd, &dt)) return -1;
    if (f) {
        const double cap = 1.0 < dt ? 1.0 : dt;
        *out = cap > nd ? cap : nd;
    } else {
        *out = nd;
    }
    return 1;
}

/* ── the engine's state ────────────────────────────────────────── */

/* A leaf (successor row or terminal reply), the leaf player's view:
   value per branch, cubeless equity, the probabilities used. */
typedef struct { double v[2], sec, p[5]; } leaf;

typedef struct { int a, b; double dp[3]; } haz_slot;

struct bgc_jan {
    double x;
    int board_sanity;
    const bgc_bearoff *bo;          /* NULL: bearoff_exact off */
    leaf *lf;                       /* per batch row */
    int cap_lf;
    leaf *c0;                       /* 0-ply leaves of the candidates, MAX_PLAYS */
    unsigned char *ex0;             /* candidate is exact (terminal, table tier) */
    double *sc, *ss;                /* 1-ply (score, sec) per candidate */
    Play *tsplays;                  /* ts_wnd scratch, MAX_PLAYS, lazily */
    haz_slot hc[HC_SLOTS];          /* _race_dp cache by one-sided rows */
};

struct bgc_jan *bgc_i_jan_new(void) {
    struct bgc_jan *j = calloc(1, sizeof(*j));
    if (!j) return NULL;
    j->x = JAN_DEFAULT_X;
    j->board_sanity = 1;
    j->c0 = malloc(MAX_PLAYS * sizeof(leaf));
    j->ex0 = malloc(MAX_PLAYS);
    j->sc = malloc(MAX_PLAYS * sizeof(double));
    j->ss = malloc(MAX_PLAYS * sizeof(double));
    for (int i = 0; i < HC_SLOTS; i++) j->hc[i].a = -1;
    if (!j->c0 || !j->ex0 || !j->sc || !j->ss) { bgc_i_jan_free(j); return NULL; }
    return j;
}

void bgc_i_jan_free(struct bgc_jan *j) {
    if (!j) return;
    free(j->lf); free(j->c0); free(j->ex0); free(j->sc); free(j->ss); free(j->tsplays);
    free(j);
}

int bgc_set_jan(bgc_engine *e, const bgc_jan_opts *o) {
    if (!e || bgc_kind(e) != BGC_KIND_PROB5JAN) return -1;
    struct bgc_jan *j = bgc_i_jan(e);
    const double x = o ? o->x : JAN_DEFAULT_X;
    if (!isfinite(x)) return -1;
    j->x = x;
    j->board_sanity = o ? o->board_sanity != 0 : 1;
    return 0;
}

int bgc_set_bearoff(bgc_engine *e, const bgc_bearoff *t) {
    if (!e || bgc_kind(e) != BGC_KIND_PROB5JAN) return -1;
    bgc_i_jan(e)->bo = t;
    return 0;
}

/* _race_dp values of one-sided rows (a, b), cached. */
static void haz_dp(struct bgc_jan *j, int a, int b, double dp[3]) {
    haz_slot *h = &j->hc[((unsigned)a * 40503u + (unsigned)b) & (HC_SLOTS - 1)];
    if (h->a == a && h->b == b) {
        memcpy(dp, h->dp, sizeof h->dp);
        return;
    }
    race_dp3(j->bo, a, b, dp);
    if (isnan(dp[0])) return;
    h->a = a;
    h->b = b;
    memcpy(h->dp, dp, sizeof h->dp);
}

/* Branch cubes: the leaf player's cube perspective, jacoby and whether
   gammons count, per root cube state. */
typedef struct { int n, persp[2], jac[2], gc[2]; } branches;

static void branch_set(branches *br, int b, const bgc_cube *c, int leaf_player) {
    br->persp[b] = cube_persp(c->owner, leaf_player);
    br->jac[b] = c->jacoby != 0;
    br->gc[b] = gammons_count(c);
}

/* MatchState.after_take(doubler). */
static bgc_cube after_take(const bgc_cube *c, int doubler) {
    bgc_cube t = *c;
    t.owner = (1 - doubler) + 1;
    t.value = c->value * 2;
    return t;
}

/* The terminal leaf: the player on roll there lost result r. */
static void leaf_terminal(const branches *br, int r, leaf *L) {
    L->p[0] = L->p[1] = L->p[2] = 0.0;
    L->p[3] = r >= 2 ? 1.0 : 0.0;
    L->p[4] = r >= 3 ? 1.0 : 0.0;
    for (int b = 0; b < br->n; b++) L->v[b] = -(br->gc[b] ? (double)r : 1.0);
    L->sec = bgc_jan_cubeless_equity(L->p);
}

/* The net's probabilities of a row: prob5_impossible_mask zeroing (when
   board_sanity), the prob5 clamps, widened to double. */
static void row_probs(const struct bgc_jan *j, const float *x, const float *y, double p[5]) {
    float q[5], c[5];
    memcpy(q, y, sizeof q);
    if (j->board_sanity) {
        unsigned char m[4];
        if (prob5_impossible_mask(x, 1, m))
            for (int k = 0; k < 4; k++) if (m[k]) q[1 + k] = 0.0f;
    }
    bgc_i_prob5_clamp(q, c);
    for (int k = 0; k < 5; k++) p[k] = (double)c[k];
}

/* _succ_values for one non-terminal row (fold on, exact_fold on): the
   table tiers, else exact probabilities in a gammonish bearoff, else the
   net's, through the folded conversion. Returns 1 for a table-tier row
   (exact: unexpanded at 1-ply). */
static int row_leaf(struct bgc_jan *j, const branches *br, const float *x, const float *y,
                    leaf *L) {
    int exact = 0;
    if (j->bo) {
        bo_view v;
        view_row(x, &v);
        if (v.cls) {
            const bgc_bearoff *t = j->bo;
            int a, b;
            const int tier = exact_tier(t, &v, &a, &b);
            if (tier == TIER_TS) {
                for (int k = 0; k < br->n; k++) L->v[k] = ts_V(t, a, b, br->persp[k]);
                L->p[0] = ldf(t->P, (size_t)a * t->n_ts + b);
                exact = 1;
            } else if (tier == TIER_HAZ) {
                double dp[3], ep[5];
                haz_dp(j, a, b, dp);
                for (int k = 0; k < br->n; k++) L->v[k] = haz_equity(dp, br->persp[k], 1);
                exact_probs5(t, &v, ep);
                L->p[0] = ep[0];
                exact = 1;
            } else {
                exact_probs5(t, &v, L->p);
            }
            if (exact) L->p[1] = L->p[2] = L->p[3] = L->p[4] = 0.0;
            else for (int k = 0; k < br->n; k++)
                L->v[k] = bgc_jan_equity_folded(L->p, br->persp[k], j->x, br->jac[k]);
            L->sec = bgc_jan_cubeless_equity(L->p);
            return exact;
        }
    }
    row_probs(j, x, y, L->p);
    for (int k = 0; k < br->n; k++)
        L->v[k] = bgc_jan_equity_folded(L->p, br->persp[k], j->x, br->jac[k]);
    L->sec = bgc_jan_cubeless_equity(L->p);
    return 0;
}

static int ensure_lf(struct bgc_jan *j, int n) {
    if (n <= j->cap_lf) return 0;
    int cap = j->cap_lf ? j->cap_lf : 4096;
    while (cap < n) cap *= 2;
    leaf *l = realloc(j->lf, (size_t)cap * sizeof(leaf));
    if (!l) return -1;
    j->lf = l;
    j->cap_lf = cap;
    return 0;
}

/* _pick's order: (value, sec) compared in float32, lexicographic; the
   first of equals. Returns 1 when (v, s) beats the best so far. */
static inline int pick_better(double v, double s, int have, float *bv, float *bs) {
    const float fv = (float)v, fs = (float)s;
    if (have && !(fv < *bv || (fv == *bv && fs < *bs))) return 0;
    *bv = fv;
    *bs = fs;
    return 1;
}

/* _oneply(S, branch cubes) with S non-terminal: v[b] (roller's view, per
   unit cube) and the roller's cubeless equity of branch 0's averaged
   probabilities. 0, -1 on allocation failure. */
static int jan_oneply(bgc_engine *e, const BoardState *S, const bgc_cube *cube, int nb,
                      double v[2], double *sec) {
    struct bgc_jan *j = bgc_i_jan(e);
    const int roller = S->turn, leaf_player = 1 - roller;
    const bgc_cube taken = after_take(cube, roller);
    branches br;
    br.n = nb;
    branch_set(&br, 0, cube, leaf_player);
    if (nb > 1) branch_set(&br, 1, &taken, leaf_player);
    int nrow = 0, nrep = 0;
    if (bgc_i_expand21(e, S, 0, -1, &nrow, &nrep) || ensure_lf(j, nrow)) return -1;
    bgc_i_forward_succ(e, 0, nrow, S, 0, 0);
    bgc_batch bt;
    bgc_i_batch(e, &bt);
    for (int r = 0; r < nrow; r++)
        row_leaf(j, &br, bt.x + (size_t)r * bt.ldx, bt.y + (size_t)r * bt.out_dim, &j->lf[r]);
    double acc[2] = {0.0, 0.0}, pacc[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    int k = 0;
    for (int d1 = 1; d1 <= 6; d1++) {
        for (int d2 = d1; d2 <= 6; d2++, k++) {
            const double w = d1 == d2 ? 1.0 : 2.0;
            for (int b = 0; b < nb; b++) {
                leaf best = {{0.0, 0.0}, 0.0, {0.0}}, t;
                float bv = 0.0f, bs = 0.0f;
                int have = 0;
                for (int r = bt.cdoff[k]; r < bt.cdoff[k + 1]; r++) {
                    const leaf *L;
                    if (bt.rep_row[r] >= 0) {
                        L = &j->lf[bt.rep_row[r]];
                    } else {
                        leaf_terminal(&br, bt.rep_gr[r], &t);
                        L = &t;
                    }
                    if (pick_better(L->v[b], L->sec, have, &bv, &bs)) { best = *L; have = 1; }
                }
                acc[b] += w * best.v[b];
                if (b == 0)
                    for (int c = 0; c < 5; c++) pacc[c] += w * best.p[c];
            }
        }
    }
    for (int b = 0; b < nb; b++) v[b] = -acc[b] / 36.0;
    double p[5], rp[5];
    for (int c = 0; c < 5; c++) p[c] = pacc[c] / 36.0;
    rp[0] = 1.0 - p[0];
    rp[1] = p[3];
    rp[2] = p[4];
    rp[3] = p[1];
    rp[4] = p[2];
    *sec = bgc_jan_cubeless_equity(rp);
    return 0;
}

/* _value_oneply(S, cube, fold=True): (value, sec), S->turn's view. */
static int jan_value_oneply(bgc_engine *e, const BoardState *S, const bgc_cube *cube,
                            double *val, double *sec) {
    double v[2];
    if (can_offer(cube->owner, S->turn)) {
        if (jan_oneply(e, S, cube, 2, v, sec)) return -1;
        *val = bgc_jan_fold(v[0], 2.0 * v[1]);
    } else {
        if (jan_oneply(e, S, cube, 1, v, sec)) return -1;
        *val = v[0];
    }
    return 0;
}

double bgc_i_jan_value1(bgc_engine *e, const BoardState *s, const bgc_cube *cube, int *err) {
    *err = 0;
    if (!cube || board_is_game_over(s)) return NAN;
    double v, sec;
    if (jan_value_oneply(e, s, cube, &v, &sec)) { *err = 1; return NAN; }
    return v;
}

/* choose_checker_action_cubeful over the n candidates in the batch's
   plays: 0-ply screen (opponent's view, folded), then at 1-ply every
   candidate (n > 1), exact ones keeping their 0-ply (value, sec). cscore
   = mover view = -value. Returns the pick, -2 on error. */
int bgc_i_jan_choose(bgc_engine *e, const BoardState *s, int n, const bgc_cube *cube,
                     int plies) {
    if (!cube) return -2;
    struct bgc_jan *j = bgc_i_jan(e);
    bgc_batch bt;
    bgc_i_batch(e, &bt);
    branches br;
    br.n = 1;
    branch_set(&br, 0, cube, 1 - s->turn);
    for (int i = 0; i < n; i++) bgc_i_encode_base(e, i, &bt.plays[i].resulting_state);
    bgc_i_forward_succ(e, 0, n, s, 0, 0);
    bgc_i_batch(e, &bt);
    for (int i = 0; i < n; i++) {
        const BoardState *rs = &bt.plays[i].resulting_state;
        if (board_is_game_over(rs)) {
            leaf_terminal(&br, bgc_i_game_result(rs), &j->c0[i]);
            j->ex0[i] = 1;
        } else {
            j->ex0[i] = (unsigned char)row_leaf(j, &br, bt.x + (size_t)i * bt.ldx,
                                                bt.y + (size_t)i * bt.out_dim, &j->c0[i]);
        }
        j->sc[i] = j->c0[i].v[0];
        j->ss[i] = j->c0[i].sec;
    }
    if (plies == 1 && n > 1) {
        for (int i = 0; i < n; i++) {
            if (j->ex0[i]) continue;
            const BoardState S = bt.plays[i].resulting_state;
            if (jan_value_oneply(e, &S, cube, &j->sc[i], &j->ss[i])) return -2;
        }
    }
    int best = 0;
    float bv = 0.0f, bs = 0.0f;
    for (int i = 0; i < n; i++) {
        if (pick_better(j->sc[i], j->ss[i], i > 0, &bv, &bs)) best = i;
        bt.cscore[i] = -j->sc[i];
    }
    return best;
}

/* The root's net probabilities (plain forward of row 0). */
static void root_probs(bgc_engine *e, const BoardState *s, double p[5]) {
    bgc_i_encode_base(e, 0, s);
    bgc_i_forward(e, 0, 1);
    bgc_batch bt;
    bgc_i_batch(e, &bt);
    row_probs(bgc_i_jan(e), bt.x, bt.y, p);
}

/* _cube_values at plies (0: _static_cube): (nd, dt) of s->turn. 0, -1
   on allocation failure. */
static int jan_cube_values(bgc_engine *e, const BoardState *s, const bgc_cube *cube,
                           int plies, double *nd, double *dt) {
    struct bgc_jan *j = bgc_i_jan(e);
    const int persp = cube_persp(cube->owner, s->turn);
    double p[5];
    int have_p = 0;
    if (j->bo) {
        bo_view v;
        view_state(s, &v);
        if (v.cls) {
            int a, b;
            const int tier = exact_tier(j->bo, &v, &a, &b);
            if (tier == TIER_TS) {
                if (!j->tsplays && !(j->tsplays = malloc(MAX_PLAYS * sizeof(Play)))) return -1;
                return tier_cube_values(j->bo, tier, a, b, persp, j->tsplays, nd, dt);
            }
            if (tier == TIER_HAZ) {
                double dp[3];
                haz_dp(j, a, b, dp);
                if (isnan(dp[0])) return -1;
                *nd = dp[persp];
                *dt = 2.0 * dp[PERSP_THEIRS];
                return 0;
            }
            if (plies == 0) { exact_probs5(j->bo, &v, p); have_p = 1; }
        }
    }
    if (plies == 1) {
        double v[2], sec;
        if (jan_oneply(e, s, cube, 2, v, &sec)) return -1;
        *nd = v[0];
        *dt = 2.0 * v[1];
        return 0;
    }
    if (!have_p) root_probs(e, s, p);
    *nd = bgc_jan_equity(p, persp, j->x, cube->jacoby);
    *dt = 2.0 * bgc_jan_equity(p, PERSP_THEIRS, j->x, cube->jacoby);
    return 0;
}

float bgc_i_jan_value(bgc_engine *e, const BoardState *s, const bgc_cube *cube) {
    double nd, dt;
    if (!cube || jan_cube_values(e, s, cube, 0, &nd, &dt)) return NAN;
    return (float)nd;
}

int bgc_i_jan_value_prob5(bgc_engine *e, const BoardState *s, float probs[5], float *eq) {
    bgc_i_encode_base(e, 0, s);
    bgc_i_forward(e, 0, 1);
    bgc_batch bt;
    bgc_i_batch(e, &bt);
    float q[5], c[5];
    memcpy(q, bt.y, sizeof q);
    if (bgc_i_jan(e)->board_sanity) {
        unsigned char m[4];
        if (prob5_impossible_mask(bt.x, 1, m))
            for (int k = 0; k < 4; k++) if (m[k]) q[1 + k] = 0.0f;
    }
    const float v = bgc_i_prob5_clamp(q, c);
    if (probs) memcpy(probs, c, sizeof c);
    if (eq) *eq = v;
    return 0;
}

int bgc_i_jan_offer(bgc_engine *e, const BoardState *s, const bgc_cube *cube, int plies,
                    bgc_cube_eval *out) {
    if (!cube || (plies != 0 && plies != 1)) return -1;
    const int persp = cube_persp(cube->owner, s->turn);
    if (persp == PERSP_THEIRS) return 0;
    double nd, dt;
    if (jan_cube_values(e, s, cube, plies, &nd, &dt)) return -1;
    int take;
    const int dbl = bgc_jan_money_cube_action(nd, dt, persp, cube->jacoby, &take);
    if (out) { out->v_no_double = (float)nd; out->v_double_take = (float)dt; out->take = take; }
    return dbl;
}

int bgc_i_jan_respond(bgc_engine *e, const BoardState *s, const bgc_cube *cube, int plies,
                      const bgc_cube_eval *hint) {
    if (!cube || (plies != 0 && plies != 1)) return -1;
    if (hint && hint->plies == plies && !isnan(hint->v_double_take)) return hint->take != 0;
    double nd, dt;
    if (jan_cube_values(e, s, cube, plies, &nd, &dt)) return -1;
    return dt <= 1.0;
}
