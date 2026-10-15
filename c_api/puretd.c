/*
 * puretd.c -- PureTD C API core: everything except the net forward.
 *
 * Ports of the Python agents (td_agent.TDAgent, prob_agent.ProbAgent)
 * and of the self-play loops (gnubg_eval.play_and_record_cubeful,
 * play_models.play_game cubeless). The Python functions are the reference;
 * see puretd.h for the public contract.
 *
 * Numerics: leaf values are fp32 net outputs; 21-dice sums, cube
 * equities and comparisons are in double, in Python's order, with no
 * FMA contraction (Python rounds every product). Terminal values
 * replace the net output on game-ending successors.
 */

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#define _POSIX_C_SOURCE 200809L

#include "puretd.h"
#include "puretd_backend.h"
#include "puretd_internal.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* _ONEPLY_PROBS: dice d1=1..6, d2=d1..6; doubles 1/36, others 2/36. */
static const double DICE_P[21] = {
    1.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0,
    1.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0,
    1.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0,
    1.0 / 36.0, 2.0 / 36.0, 2.0 / 36.0,
    1.0 / 36.0, 2.0 / 36.0,
    1.0 / 36.0,
};

/* The same in fp32 (train_prob5._ONEPLY_PROBS32): the prob5 1-ply target
   vector is summed in float. */
#define P1_36 ((float)(1.0 / 36.0))
#define P2_36 ((float)(2.0 / 36.0))
static const float DICE_P32[21] = {
    P1_36, P2_36, P2_36, P2_36, P2_36, P2_36,
    P1_36, P2_36, P2_36, P2_36, P2_36,
    P1_36, P2_36, P2_36, P2_36,
    P1_36, P2_36, P2_36,
    P1_36, P2_36,
    P1_36,
};

enum { FLIP_NONE = 0, FLIP_NEG = 1, FLIP_ONE_MINUS = 2 };

struct bgc_engine {
    struct bgc_backend *be;
    int kind, in_size, out_dim, ldx;
    bgc_encoder enc;    /* the export's encoder (bgc_open); enc.name -> enc_name */
    int nb;             /* enc.base->n: base features, the cube block's offset */
    int format_version; /* the export's BGNN header format (1: legacy, by width) */
    char enc_name[NN_ENCODER_NAME_MAX];
    /* forward batch: x rows ldx apart (row 0 64-byte aligned, pad zero),
       y raw outputs, v per-row value (prob5: clamped equity) */
    float *x, *y, *v;
    int cap_rows;
    Play *plays;        /* root candidates */
    Play *iplays;       /* 21-dice expansion */
    /* expansion replies: row index (-1 terminal) and game_result */
    int *rep_row, *rep_gr;
    int cap_rep;
    /* per batch slot: 22 reply offsets (dice-major) and 21 pass flags */
    int *cdoff;
    unsigned char *cpass;
    int *slot_cand;     /* batch slot -> candidate index, MAX_PLAYS */
    double *cscore;     /* per candidate, mover's view, MAX_PLAYS */
    float root_thresh;  /* bgc_set_search; 0 = full width */
    unsigned char *ckeep; /* root filter survivors, MAX_PLAYS */
    int row_exact;      /* bgc_backend_row_exact(): per-row results batch-independent */
    int *pidx;          /* value1_play: candidate -> reply index, MAX_PLAYS */
    uint64_t *pkey;     /* value1_play: board keys, MAX_PLAYS */
    /* expand21's cross-roll row dedup (row-exact backends): packed board
       per row (4 words), open-addressing table row index + generation */
    int dedup;          /* row_exact and not BGC_EXPAND_DEDUP=0 (read at open) */
    float *xpar;        /* forward_succ's parent row (ldx floats, aligned) */
    int bf16;           /* -DBGC_BF16 build with the bf16 forward usable: value1's rows */
    int bf16_now;       /* forward_p: this forward is value1's (bf16) */
    uint64_t *rkey;
    int *dd_row;
    uint32_t *dd_gen;
    uint32_t dd_cur;
    int dd_count;
    struct bgc_jan *jan;  /* PROB5JAN: puretd_janowski.c's state */
};

#define DD_SLOTS 4096                /* dedup table; at most DD_SLOTS / 2 rows per expansion */

/* ── small helpers ─────────────────────────────────────────────── */

static int game_result(const BoardState *s) {
    int w = board_winner(s);
    if (w < 0) return 0;
    int loser = 1 - w;
    if (s->off[loser] > 0) return 1;
    int in_home = 0;
    if (loser == WHITE) {
        for (int i = 18; i < 24; i++) if (s->points[i] > 0) { in_home = 1; break; }
    } else {
        for (int i = 0; i < 6; i++) if (s->points[i] < 0) { in_home = 1; break; }
    }
    if (s->bar[loser] > 0 || in_home) return 3;
    return 2;
}

/* cube_persp, can_offer, gammons_count: puretd_internal.h */

/* TDAgent._money_cube_decision */
static int money_decision(double v_nd, double v_dt, int persp_nd, int jacoby) {
    if (v_dt > 1.0) {
        if (jacoby && persp_nd == PERSP_CENTERED) return 1;
        return 1.0 >= v_nd;
    }
    return v_dt > v_nd;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* model.prob5_to_equity of a 5-vector in torch's fp32 order
   (((2p0 - p4) + p1) + p2) - p3, then - 1 (= train_prob5._prob5_equity_np). */
static float equity5(const float p[5]) {
    float e = p[0] * 2.0f;
    e -= p[4];
    e += p[1];
    e += p[2];
    e -= p[3];
    e -= 1.0f;
    return e;
}

/* model.prob5_postprocess then equity5. */
static float prob5_post(const float *in, float *p) {
    float q[5];
    q[0] = in[0];
    q[1] = in[1] < q[0] ? in[1] : q[0];
    float lose = 1.0f - q[0];
    q[3] = in[3] < lose ? in[3] : lose;
    q[2] = in[2] < q[1] ? in[2] : q[1];
    q[4] = in[4] < q[3] ? in[4] : q[3];
    if (p) memcpy(p, q, sizeof q);
    return equity5(q);
}

float bgc_i_prob5_clamp(const float in[5], float out[5]) { return prob5_post(in, out); }

/* ── buffers ───────────────────────────────────────────────────── */

static int ensure_rows(bgc_engine *e, int n) {
    if (n <= e->cap_rows) return 0;
    int cap = e->cap_rows ? e->cap_rows : 4096;
    while (cap < n) cap *= 2;
    size_t xbytes = (size_t)cap * e->ldx * sizeof(float);
    xbytes = (xbytes + 63) & ~(size_t)63;
    float *x = aligned_alloc(64, xbytes);
    float *y = malloc((size_t)cap * e->out_dim * sizeof(float));
    float *v = malloc((size_t)cap * sizeof(float));
    uint64_t *k = realloc(e->rkey, (size_t)cap * 4 * sizeof(uint64_t));
    if (k) e->rkey = k;
    if (!x || !y || !v || !k) { free(x); free(y); free(v); return -1; }
    if (e->x) memcpy(x, e->x, (size_t)e->cap_rows * e->ldx * sizeof(float));
    free(e->x); free(e->y); free(e->v);
    e->x = x; e->y = y; e->v = v;
    e->cap_rows = cap;
    return 0;
}

static int ensure_rep(bgc_engine *e, int n) {
    if (n <= e->cap_rep) return 0;
    int cap = e->cap_rep ? e->cap_rep : 8192;
    while (cap < n) cap *= 2;
    int *r = realloc(e->rep_row, (size_t)cap * sizeof(int));
    if (!r) return -1;
    e->rep_row = r;
    int *g = realloc(e->rep_gr, (size_t)cap * sizeof(int));
    if (!g) return -1;
    e->rep_gr = g;
    e->cap_rep = cap;
    return 0;
}

/* Row r = base encoding of s, pad and cube columns zero. */
static void encode_base(bgc_engine *e, int r, const BoardState *s) {
    float *row = e->x + (size_t)r * e->ldx;
    e->enc.base->encode(s, row);
    memset(row + e->nb, 0, (size_t)(e->ldx - e->nb) * sizeof(float));
}

/* Packed board: points, bar, off, turn as bytes (a base encoder's inputs). */
static inline void board_pack(const BoardState *s, uint64_t k[4]) {
    unsigned char b[32];
    for (int i = 0; i < NUM_POINTS; i++) b[i] = (unsigned char)s->points[i];
    b[24] = (unsigned char)s->bar[0];
    b[25] = (unsigned char)s->bar[1];
    b[26] = (unsigned char)s->off[0];
    b[27] = (unsigned char)s->off[1];
    b[28] = (unsigned char)s->turn;
    b[29] = b[30] = b[31] = 0;
    memcpy(k, b, sizeof b);
}

static inline uint32_t pack_hash(const uint64_t k[4]) {
    uint64_t h = k[0] * 0x9E3779B97F4A7C15ULL;
    h ^= (k[1] * 0xC2B2AE3D27D4EB4FULL) >> 7;
    h ^= (k[2] * 0x165667B19E3779F9ULL) >> 13;
    h ^= (k[3] * 0x27D4EB2F165667C5ULL) >> 19;
    h ^= h >> 32;
    return (uint32_t)h;
}

/* Starts a dedup scope: rows encoded from here on share by board. */
static void dedup_begin(bgc_engine *e) {
    if (++e->dd_cur == 0) {                     /* generation wrapped */
        memset(e->dd_gen, 0, DD_SLOTS * sizeof(uint32_t));
        e->dd_cur = 1;
    }
    e->dd_count = 0;
}

/* The batch row of reply board rs (ensure_rows done): the row of an equal
 * board earlier in this scope (one forward value serves every reply that
 * reaches it; the cube columns are the same for the whole expansion), else
 * a new base-encoded row. Without dedup (not row-exact, or
 * BGC_EXPAND_DEDUP=0 at bgc_open) always a new row: those backends keep
 * one row per reply. On a row-exact backend the values are the same
 * either way, bitwise (tests/test_c_api.py). */
static int reply_row(bgc_engine *e, const BoardState *rs, int *nrow) {
    if (e->dedup) {
        uint64_t key[4];
        board_pack(rs, key);
        for (uint32_t p = pack_hash(key) & (DD_SLOTS - 1);; p = (p + 1) & (DD_SLOTS - 1)) {
            if (e->dd_gen[p] != e->dd_cur) {          /* free: insert (while room) */
                if (e->dd_count < DD_SLOTS / 2) {
                    e->dd_gen[p] = e->dd_cur;
                    e->dd_row[p] = *nrow;
                    e->dd_count++;
                    memcpy(e->rkey + (size_t)*nrow * 4, key, sizeof key);
                }
                break;
            }
            const uint64_t *q = e->rkey + (size_t)e->dd_row[p] * 4;
            if (q[0] == key[0] && q[1] == key[1] && q[2] == key[2] && q[3] == key[3])
                return e->dd_row[p];
        }
    }
    encode_base(e, *nrow, rs);
    return (*nrow)++;
}

/* Cube columns of rows [r0, r1): one-hot perspective + is_cube_action. */
static void set_cube_cols(bgc_engine *e, int r0, int r1, int persp, int ica) {
    if (e->kind != BGC_KIND_CUBEFUL) return;
    PROF_T0(t0);
    for (int r = r0; r < r1; r++) cube_cols(e->x + (size_t)r * e->ldx + e->nb, persp, ica);
    PROF_ADD(t0, BGC_PT_CUBECOLS, r1 - r0);
}

#ifdef BGC_BF16
#pragma weak bgc_backend_bf16
#pragma weak bgc_backend_forward_bf16
#endif

/* Forward rows [r0, r0+n) in chunks; fills v (prob5: clamped equity).
 * par: NULL, or the rows' common parent row (bgc_backend_forward_delta). */
static void forward_p(bgc_engine *e, int r0, int n, const float *par) {
    PROF_T0(t0);
    for (int off = 0; off < n; off += BGC_CHUNK) {
        int cnt = n - off < BGC_CHUNK ? n - off : BGC_CHUNK;
#ifdef BGC_BF16
        if (e->bf16_now) {
            bgc_backend_forward_bf16(e->be, par, e->x + (size_t)(r0 + off) * e->ldx, e->ldx,
                                     cnt, e->y + (size_t)(r0 + off) * e->out_dim);
            continue;
        }
#endif
        if (par)
            bgc_backend_forward_delta(e->be, par, e->x + (size_t)(r0 + off) * e->ldx, e->ldx,
                                      cnt, e->y + (size_t)(r0 + off) * e->out_dim);
        else
            bgc_backend_forward(e->be, e->x + (size_t)(r0 + off) * e->ldx, e->ldx,
                                cnt, e->y + (size_t)(r0 + off) * e->out_dim);
    }
    if (e->kind == BGC_KIND_PROB5 || e->kind == BGC_KIND_PROB5JAN) {
        for (int r = r0; r < r0 + n; r++) e->v[r] = prob5_post(e->y + (size_t)r * 5, NULL);
    } else {
        for (int r = r0; r < r0 + n; r++) e->v[r] = e->y[r];
    }
    PROF_ADD(t0, BGC_PT_FORWARD, n);
}

static void forward(bgc_engine *e, int r0, int n) { forward_p(e, r0, n, NULL); }

/* Forward of rows [r0, r0+n) that are successors of position X (the 0-ply
 * candidates of X, or X's 21-roll replies), their cube columns (persp,
 * ica): the parent row is X with the turn switched and those columns. A
 * row's value depends on the row and X only (bgc_backend_forward_delta),
 * so the expansion and the 0-ply candidates agree bitwise on a
 * row-exact backend. */
static void forward_succ(bgc_engine *e, int r0, int n, const BoardState *X, int persp, int ica) {
    BoardState back = *X;
    board_switch_turn(&back);
    e->enc.base->encode(&back, e->xpar);
    memset(e->xpar + e->nb, 0, (size_t)(e->ldx - e->nb) * sizeof(float));
    if (e->kind == BGC_KIND_CUBEFUL) cube_cols(e->xpar + e->nb, persp, ica);
    forward_p(e, r0, n, e->xpar);
}

/* The 1-ply target rows' forward: forward_succ, or in a bf16 build the bf16
 * forward (value1 only: the play, the cube and every 0-ply value stay fp32). */
static void forward_v1(bgc_engine *e, int r0, int n, const BoardState *X, int persp, int ica) {
#ifdef BGC_BF16
    e->bf16_now = e->bf16;
    forward_succ(e, r0, n, X, persp, ica);
    e->bf16_now = 0;
#else
    forward_succ(e, r0, n, X, persp, ica);
#endif
}

/* ── engine ────────────────────────────────────────────────────── */

#ifdef BGC_PROFILE
bgc_prof_table bgc_prof;
int bgc_prof_ctx = BGC_PC_COLLECT;

/* Profile build only: copies the table (BGC_PC_N x BGC_PT_N, row-major
   by context) and returns BGC_PC_N * BGC_PT_N. */
__attribute__((visibility("default")))
int bgc_prof_read(double *t, long *calls, long *rows) {
    const int n = BGC_PC_N * BGC_PT_N;
    if (t) memcpy(t, bgc_prof.t, n * sizeof(double));
    if (calls) memcpy(calls, bgc_prof.calls, n * sizeof(long));
    if (rows) memcpy(rows, bgc_prof.rows, n * sizeof(long));
    return n;
}

__attribute__((visibility("default")))
void bgc_prof_reset(void) {
    memset(&bgc_prof, 0, sizeof(bgc_prof));
    bgc_prof_ctx = BGC_PC_COLLECT;
}
#endif

/* ── encoder table ─────────────────────────────────────────────
 * The encoders the C API knows (types: puretd_internal.h). Every row
 * layout decision of the engine and the collector comes from here.
 * bgc_i_encoder_resolve reads a name as encoding.get_encoder does: the
 * first cube block whose prefix it starts with ("cubeful_", then ""),
 * then the rest as a base encoder's name.
 *
 * ADDING A BASE ENCODER (a new encoding.ENCODERS entry): one row in
 * BGC_BASES { name, its num_features, a C function writing exactly those
 * features of a BoardState, bit-exact with the Python Encoder.encode,
 * collect 0 }. The function must depend on the board only (points, bar,
 * off, turn: expand21's row dedup keys rows by those). Its cubeful_
 * name then resolves too, with the cube block after its features. Set
 * collect to 1 once bgc_collect's rows on it are parity-tested
 * (tests/test_c_collect.py), together with the trainers'
 * C_COLLECT_ENCODERS (test_c_collect checks they agree). A new cube
 * block (another get_encoder wrapper) is a BGC_CUBE_BLOCKS row and,
 * unless it is the 3-hot + is_cube_action layout, its own cube_cols. */
static const bgc_base_enc BGC_BASES[] = {
    /* name             n       encode        collect */
    {"perspective196",  BGC_NB, encode_state, 1},
};
static const bgc_cube_block BGC_CUBE_BLOCKS[] = {
    /* prefix      ncube */
    {"cubeful_",   4},  /* CubefulEncoder */
    {"",           0},  /* the base alone (cubeless kinds) */
};
#define N_ARR(a) ((int)(sizeof(a) / sizeof((a)[0])))

static void set_err(char *err, size_t errlen, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));
static void set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || !errlen) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

int bgc_i_encoder_resolve(const char *name, bgc_encoder *out, char *err, size_t errlen) {
    if (!name || !*name) {
        set_err(err, errlen, "no encoder name");
        return -1;
    }
    const bgc_cube_block *cb = NULL;
    for (int i = 0; i < N_ARR(BGC_CUBE_BLOCKS) && !cb; i++)
        if (!strncmp(name, BGC_CUBE_BLOCKS[i].prefix, strlen(BGC_CUBE_BLOCKS[i].prefix)))
            cb = &BGC_CUBE_BLOCKS[i];
    const char *base_name = name + strlen(cb->prefix);      /* "" always matches */
    const bgc_base_enc *b = NULL;
    for (int i = 0; i < N_ARR(BGC_BASES) && !b; i++)
        if (!strcmp(base_name, BGC_BASES[i].name)) b = &BGC_BASES[i];
    if (!b) {
        char known[256] = "";
        for (int i = 0; i < N_ARR(BGC_BASES); i++)
            snprintf(known + strlen(known), sizeof(known) - strlen(known), "%s%s",
                     i ? ", " : "", BGC_BASES[i].name);
        if (*cb->prefix)
            set_err(err, errlen, "encoder '%s': unknown base encoder '%s' (C API base "
                    "encoders: %s)", name, base_name, known);
        else
            set_err(err, errlen, "unknown encoder '%s' (C API base encoders: %s, "
                    "each also as cubeful_<base>)", name, known);
        return -1;
    }
    out->name = name;
    out->base = b;
    out->cube = cb;
    out->n_in = b->n + cb->ncube;
    out->collect = b->collect;
    return 0;
}

const bgc_encoder *bgc_i_encoder(const bgc_engine *e) { return &e->enc; }

int bgc_encoder_lookup(const char *name, bgc_encoder_info *out) {
    bgc_encoder enc;
    if (bgc_i_encoder_resolve(name, &enc, NULL, 0)) return -1;
    if (out) {
        out->n_base = enc.base->n;
        out->ncube = enc.cube->ncube;
        out->input_size = enc.n_in;
        out->collect = enc.collect;
    }
    return 0;
}

/* LEGACY PATH: a format-1 BGNN file names no encoder. Its input width
   stands in for the name (200 inputs cubeful, 196 cubeless). Do
   not add rows: every other encoder is exported with its name (format 2). */
static const struct { int n_in; const char *name; } BGC_LEGACY_WIDTHS[] = {
    {196, "perspective196"},
    {200, "cubeful_perspective196"},
};

/* The engine kind of encoder `enc` with that output: CUBEFUL for a
   cubeful encoder with an equity output; else by output mode. -1: none. */
static int encoder_kind(const bgc_encoder *enc, int output_mode, int out) {
    if (enc->cube->ncube > 0)
        return output_mode == NN_OUTPUT_EQUITY && out == 1 ? BGC_KIND_CUBEFUL : -1;
    if (output_mode == NN_OUTPUT_PROB5 && out == 5) return BGC_KIND_PROB5;
    if (output_mode == NN_OUTPUT_EQUITY && out == 1) return BGC_KIND_EQUITY;
    if (output_mode == NN_OUTPUT_PROBABILITY && out == 1) return BGC_KIND_PROB;
    return -1;
}

bgc_engine *bgc_open(const char *path) {
    char err[512];
    bgc_engine *e = bgc_open_ex(path, err, sizeof(err));
    if (!e) fprintf(stderr, "%s\n", err);
    return e;
}

static bgc_engine *open_engine(const char *path, char *err, size_t errlen, int janowski);

bgc_engine *bgc_open_ex(const char *path, char *err, size_t errlen) {
    return open_engine(path, err, errlen, 0);
}

bgc_engine *bgc_open_janowski(const char *path, char *err, size_t errlen) {
    return open_engine(path, err, errlen, 1);
}

/* bgc_open_ex; janowski: a perspective196 PROB5 export as BGC_KIND_PROB5JAN. */
static bgc_engine *open_engine(const char *path, char *err, size_t errlen, int janowski) {
    set_err(err, errlen, "%s", "");
    NNModel m;
    if (nn_load(&m, path) != 0) {                        /* nn_load's reason: stderr */
        set_err(err, errlen, "bgc_open: cannot load BGNN model %s", path ? path : "(null)");
        return NULL;
    }
    int out = m.layer_out[m.num_hidden];
    const char *name = m.encoder_name;
    if (m.format_version == NN_FORMAT_LEGACY) {          /* legacy: by width */
        name = NULL;
        for (int i = 0; i < N_ARR(BGC_LEGACY_WIDTHS); i++)
            if (m.input_size == BGC_LEGACY_WIDTHS[i].n_in) name = BGC_LEGACY_WIDTHS[i].name;
        if (!name) {
            set_err(err, errlen, "bgc_open: %s: legacy BGNN file (no encoder name) with %d "
                    "inputs, not a known layout (196, 200); re-export it with "
                    "export_weights.py", path, m.input_size);
            nn_free(&m);
            return NULL;
        }
    }
    bgc_encoder enc;
    char why[384];
    if (bgc_i_encoder_resolve(name, &enc, why, sizeof(why))) {
        set_err(err, errlen, "bgc_open: %s: %s", path, why);
        nn_free(&m);
        return NULL;
    }
    if (m.input_size != enc.n_in) {
        set_err(err, errlen, "bgc_open: %s: encoder '%s' rows are %d wide (%s %d + %d cube "
                "columns), the net has %d inputs", path, name, enc.n_in, enc.base->name,
                enc.base->n, enc.cube->ncube, m.input_size);
        nn_free(&m);
        return NULL;
    }
    int kind = encoder_kind(&enc, m.output_mode, out);
    if (kind < 0) {
        set_err(err, errlen, "bgc_open: %s: unsupported model: encoder '%s' (%s) with output "
                "mode %d, %d outputs", path, name, enc.cube->ncube ? "cubeful" : "cubeless",
                m.output_mode, out);
        nn_free(&m);
        return NULL;
    }
    if (janowski) {
        if (kind != BGC_KIND_PROB5 || strcmp(enc.base->name, "perspective196")) {
            set_err(err, errlen, "bgc_open_janowski: %s: not a perspective196 prob5 export "
                    "(encoder '%s', output mode %d, %d outputs)", path, name, m.output_mode, out);
            nn_free(&m);
            return NULL;
        }
        kind = BGC_KIND_PROB5JAN;
    }
    bgc_engine *e = calloc(1, sizeof(*e));
    if (!e) { set_err(err, errlen, "bgc_open: out of memory"); nn_free(&m); return NULL; }
    e->kind = kind;
    e->in_size = m.input_size;
    snprintf(e->enc_name, sizeof(e->enc_name), "%s", name);
    e->enc = enc;
    e->enc.name = e->enc_name;
    e->nb = enc.base->n;
    e->format_version = m.format_version;
    e->out_dim = out;
    e->ldx = BGC_LDX(m.input_size);
    e->be = bgc_backend_create(&m);
    nn_free(&m);
    if (!e->be) {
        set_err(err, errlen, "bgc_open: backend create failed");
        free(e);
        return NULL;
    }
    e->plays = malloc(MAX_PLAYS * sizeof(Play));
    e->iplays = malloc(MAX_PLAYS * sizeof(Play));
    e->slot_cand = malloc(MAX_PLAYS * sizeof(int));
    e->cscore = malloc(MAX_PLAYS * sizeof(double));
    e->cdoff = malloc((size_t)MAX_PLAYS * 22 * sizeof(int));
    e->cpass = malloc((size_t)MAX_PLAYS * 21);
    e->ckeep = malloc(MAX_PLAYS);
    e->pidx = malloc(MAX_PLAYS * sizeof(int));
    e->xpar = aligned_alloc(64, ((size_t)e->ldx * sizeof(float) + 63) & ~(size_t)63);
    e->pkey = malloc(MAX_PLAYS * sizeof(uint64_t));
    e->row_exact = bgc_backend_row_exact();
    const char *dd = getenv("BGC_EXPAND_DEDUP");     /* "0": off (tests, A/B) */
    e->dedup = e->row_exact && !(dd && !strcmp(dd, "0"));
#ifdef BGC_BF16
    e->bf16 = bgc_backend_bf16 && bgc_backend_forward_bf16 && bgc_backend_bf16(e->be);
#endif
    e->dd_row = malloc(DD_SLOTS * sizeof(int));
    e->dd_gen = calloc(DD_SLOTS, sizeof(uint32_t));
    if (kind == BGC_KIND_PROB5JAN && !(e->jan = bgc_i_jan_new())) {
        set_err(err, errlen, "bgc_open: out of memory");
        bgc_close(e);
        return NULL;
    }
    if (!e->xpar || !e->dd_row || !e->dd_gen || !e->ckeep || !e->pidx || !e->pkey || !e->plays || !e->iplays || !e->slot_cand || !e->cscore || !e->cdoff || !e->cpass
        || ensure_rows(e, 4096) || ensure_rep(e, 8192)) {
        set_err(err, errlen, "bgc_open: out of memory");
        bgc_close(e);
        return NULL;
    }
    return e;
}

void bgc_close(bgc_engine *e) {
    if (!e) return;
    if (e->be) bgc_backend_destroy(e->be);
    free(e->x); free(e->y); free(e->v);
    free(e->plays); free(e->iplays);
    free(e->rep_row); free(e->rep_gr);
    free(e->cdoff); free(e->cpass); free(e->slot_cand); free(e->cscore);
    free(e->ckeep);
    free(e->pidx); free(e->pkey);
    free(e->rkey); free(e->dd_row); free(e->dd_gen); free(e->xpar);
    bgc_i_jan_free(e->jan);
    free(e);
}

#define BGC_STR_(x) #x
#define BGC_STR(x) BGC_STR_(x)
__attribute__((used, visibility("default")))
const char bgc_abi_tag[] = "puretd-abi:" BGC_STR(BGC_ABI_VERSION);

int bgc_abi_version(void) { return BGC_ABI_VERSION; }

int bgc_struct_sizes(size_t *out, int n) {
    const size_t sz[] = {
        sizeof(BoardState), sizeof(Play), sizeof(bgc_cube), sizeof(bgc_cube_eval),
        sizeof(bgc_search), sizeof(bgc_encoder_info), sizeof(bgc_jan_opts),
        sizeof(bgc_rng), sizeof(bgc_game_opts), sizeof(bgc_game_result),
        sizeof(bgc_collect_opts)};
    const int count = (int)(sizeof sz / sizeof sz[0]);
    for (int i = 0; i < n && i < count; i++) out[i] = sz[i];
    return count;
}

int bgc_kind(const bgc_engine *e) { return e->kind; }
int bgc_input_size(const bgc_engine *e) { return e->in_size; }
const char *bgc_encoder_name(const bgc_engine *e) { return e->enc.name; }
int bgc_format_version(const bgc_engine *e) { return e->format_version; }
const char *bgc_describe(const bgc_engine *e) { return bgc_backend_describe(e->be); }

/* ── 0-ply value ───────────────────────────────────────────────── */

float bgc_value(bgc_engine *e, const BoardState *s, const bgc_cube *cube, int ica) {
    if (e->kind == BGC_KIND_PROB5JAN) return bgc_i_jan_value(e, s, cube);
    if (e->kind == BGC_KIND_CUBEFUL && !cube) return NAN;
    PROF_T0(t0);
    encode_base(e, 0, s);
    PROF_ADD(t0, BGC_PT_ENCODE, 1);
    if (e->kind == BGC_KIND_CUBEFUL) set_cube_cols(e, 0, 1, cube_persp(cube->owner, s->turn), ica);
    forward(e, 0, 1);
    return e->v[0];
}

int bgc_value_prob5(bgc_engine *e, const BoardState *s, float probs[5], float *equity) {
    if (e->kind == BGC_KIND_PROB5JAN) return bgc_i_jan_value_prob5(e, s, probs, equity);
    if (e->kind != BGC_KIND_PROB5) return -1;
    encode_base(e, 0, s);
    bgc_backend_forward(e->be, e->x, e->ldx, 1, e->y);
    float eq = prob5_post(e->y, probs);
    if (equity) *equity = eq;
    return 0;
}

/* ── 21-dice expansion ─────────────────────────────────────────── *
 * Appends the replies of every dice roll at X (dice-major, d1=1..6,
 * d2=d1..6; plays in get_legal_plays order) to the batch. A
 * non-terminal reply gets a base-encoded row; a terminal reply carries
 * its game_result and no row; a forced pass is one row of X with the
 * turn switched. Replies of different rolls reaching the same board
 * share one row (reply_row; ~9% of the rows in self-play). Cube columns
 * are set by the caller. The plays of dice index `keep` (dice_index; -1
 * none) are also copied to e->plays. */
static int expand21(bgc_engine *e, const BoardState *X, int slot, int keep,
                    int *nrow, int *nrep) {
    int *doff = e->cdoff + (size_t)slot * 22;
    unsigned char *pass = e->cpass + (size_t)slot * 21;
    int k = 0;
    dedup_begin(e);
    for (int d1 = 1; d1 <= 6; d1++) {
        for (int d2 = d1; d2 <= 6; d2++, k++) {
            doff[k] = *nrep;
            PROF_T0(t0);
            int n = get_legal_plays(X, d1, d2, e->iplays, MAX_PLAYS);
            PROF_ADD(t0, BGC_PT_MOVEGEN, n);
            PROF_T0(t1);
            if (n == 0) {
                if (ensure_rows(e, *nrow + 1) || ensure_rep(e, *nrep + 1)) return -1;
                BoardState back = *X;
                board_switch_turn(&back);
                e->rep_row[*nrep] = reply_row(e, &back, nrow);
                e->rep_gr[*nrep] = 0;
                (*nrep)++;
                pass[k] = 1;
                PROF_ADD(t1, BGC_PT_ENCODE, 1);
                continue;
            }
            if (ensure_rows(e, *nrow + n) || ensure_rep(e, *nrep + n)) return -1;
            if (k == keep) memcpy(e->plays, e->iplays, (size_t)n * sizeof(Play));
            pass[k] = 0;
            for (int i = 0; i < n; i++) {
                const BoardState *rs = &e->iplays[i].resulting_state;
                if (board_is_game_over(rs)) {
                    e->rep_row[*nrep] = -1;
                    e->rep_gr[*nrep] = game_result(rs);
                } else {
                    e->rep_row[*nrep] = reply_row(e, rs, nrow);
                    e->rep_gr[*nrep] = 0;
                }
                (*nrep)++;
            }
            PROF_ADD(t1, BGC_PT_ENCODE, n);
        }
    }
    doff[21] = *nrep;
    return 0;
}

/* expand21's index of the dice {a, b}: d1 = 1..6, d2 = d1..6. */
static int dice_index(int a, int b) {
    int lo = a < b ? a : b, hi = a < b ? b : a;
    return (lo - 1) * (14 - lo) / 2 + (hi - lo);
}

/* Per dice: min over the replies of the value from the view of the
 * player on roll after the reply (rows at row_off + row). Terminal
 * reply (the player at X just won): 0.0 if zero_term, else -gr when
 * gc, else -1. */
static void dice_mins(const bgc_engine *e, int slot, int row_off, int zero_term,
                      int gc, double mins[21]) {
    const int *doff = e->cdoff + (size_t)slot * 22;
    for (int k = 0; k < 21; k++) {
        double m = INFINITY;
        for (int r = doff[k]; r < doff[k + 1]; r++) {
            double val;
            if (e->rep_row[r] >= 0) val = (double)e->v[row_off + e->rep_row[r]];
            else if (zero_term) val = 0.0;
            else val = -(double)(gc ? e->rep_gr[r] : 1);
            if (val < m) m = val;
        }
        mins[k] = m;
    }
}

static double flip(double m, int how) {
    if (how == FLIP_NEG) return -m;
    if (how == FLIP_ONE_MINUS) return 1.0 - m;
    return m;
}

/* Σ_dice p·flip(min), in double. grouped (ProbAgent): the forced-pass
 * dice first as one term (their summed probability times the one pass
 * value), then the other dice in order; else every dice in order. */
static double dice_sum(const double mins[21], const unsigned char *pass, int grouped, int how) {
    double s = 0.0;
    if (!grouped) {
        for (int k = 0; k < 21; k++) s += DICE_P[k] * flip(mins[k], how);
        return s;
    }
    double pp = 0.0, mp = 0.0;
    int have = 0;
    for (int k = 0; k < 21; k++) {
        if (pass[k]) { pp += DICE_P[k]; if (!have) { mp = mins[k]; have = 1; } }
    }
    if (pp > 0.0) s += pp * flip(mp, how);
    for (int k = 0; k < 21; k++)
        if (!pass[k]) s += DICE_P[k] * flip(mins[k], how);
    return s;
}

/* ── 1-ply value ───────────────────────────────────────────────── */

/* TDAgent.value_oneply_checker[_cubeful] / ProbAgent.value_oneply_checker.
 * *err set on allocation failure. keep: as expand21 (slot 0); the batch
 * (rows, values, reply records of slot 0) stays valid until the next
 * forward. */
static double value1k(bgc_engine *e, const BoardState *X, const bgc_cube *cube, int keep,
                      int *err) {
    *err = 0;
    if (e->kind == BGC_KIND_PROB5JAN) return bgc_i_jan_value1(e, X, cube, err);
    if (board_is_game_over(X)) {
        if (e->kind == BGC_KIND_PROB5) return -(double)game_result(X);
        return NAN;
    }
    if (e->kind == BGC_KIND_CUBEFUL && !cube) return NAN;
    int nrow = 0, nrep = 0;
    if (expand21(e, X, 0, keep, &nrow, &nrep)) { *err = 1; return NAN; }
    int q = 1 - X->turn;
    int gc = 1, persp = 0, ica = 0;
    if (e->kind == BGC_KIND_CUBEFUL) {
        persp = cube_persp(cube->owner, q);
        ica = can_offer(cube->owner, q);
        set_cube_cols(e, 0, nrow, persp, ica);
        gc = gammons_count(cube);
    }
    forward_v1(e, 0, nrow, X, persp, ica);
    PROF_T0(t0);
    double mins[21];
    dice_mins(e, 0, 0, e->kind == BGC_KIND_PROB, gc, mins);
    double r = dice_sum(mins, e->cpass, e->kind == BGC_KIND_PROB5,
                        e->kind == BGC_KIND_PROB ? FLIP_ONE_MINUS : FLIP_NEG);
    PROF_ADD(t0, BGC_PT_REDUCE, nrow);
    return r;
}

static double value1(bgc_engine *e, const BoardState *X, const bgc_cube *cube, int *err) {
    return value1k(e, X, cube, -1, err);
}

/* value1 of X under cube a and under cube b from one expansion: the rows
 * are encoded once and forwarded once per cube (each forward sets its own
 * cube columns). Same values as two value1 calls. 0, or -1 on allocation
 * failure. */
static int value1_pair(bgc_engine *e, const BoardState *X, const bgc_cube *a,
                       const bgc_cube *b, double *va, double *vb) {
    if (board_is_game_over(X)) {
        int err;
        *va = value1(e, X, a, &err);
        *vb = value1(e, X, b, &err);
        return 0;
    }
    int nrow = 0, nrep = 0;
    if (expand21(e, X, 0, -1, &nrow, &nrep)) return -1;
    int q = 1 - X->turn;
    const bgc_cube *cubes[2] = {a, b};
    double *outs[2] = {va, vb};
    for (int c = 0; c < 2; c++) {
        int persp = cube_persp(cubes[c]->owner, q);
        int ica = can_offer(cubes[c]->owner, q);
        set_cube_cols(e, 0, nrow, persp, ica);
        forward_v1(e, 0, nrow, X, persp, ica);
        PROF_T0(t0);
        double mins[21];
        dice_mins(e, 0, 0, 0, gammons_count(cubes[c]), mins);
        *outs[c] = dice_sum(mins, e->cpass, 0, FLIP_NEG);
        PROF_ADD(t0, BGC_PT_REDUCE, nrow);
    }
    return 0;
}


float bgc_value_1ply(bgc_engine *e, const BoardState *s, const bgc_cube *cube) {
    int err;
    return (float)value1(e, s, cube, &err);
}

/* PROB5: train_prob5._oneply_target_vec_expand21 (the 1-ply TD target
 * vector) from value1k's batch, slot 0. Per dice: every reply's vector
 * from the view of the player at X (a non-terminal reply's clamped
 * outputs flipped, [1 - p0, p3, p4, p1, p2] as _flip_rows; a terminal one
 * [1, gr >= 2, gr >= 3, 0, 0]), and the FIRST maximum of equity5 of that
 * flipped vector, not the min of the opponent-view equity value1 takes
 * (the two can round apart); the whole chosen vector is kept. The 21
 * vectors are summed in float, dice in order (forced-pass dice in place,
 * not grouped as in value1), each times its fp32 probability, then
 * clipped to [0, 1]. Returns -1 on a NaN equity (Python raises), else 0. */
static int vec1_prob5(const bgc_engine *e, float out[5]) {
    const int *doff = e->cdoff;
    float t[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < 21; k++) {
        float best[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, best_eq = 0.0f;
        for (int r = doff[k]; r < doff[k + 1]; r++) {
            float m[5];
            if (e->rep_row[r] >= 0) {
                float p[5];
                prob5_post(e->y + (size_t)e->rep_row[r] * 5, p);
                m[0] = 1.0f - p[0];
                m[1] = p[3];
                m[2] = p[4];
                m[3] = p[1];
                m[4] = p[2];
            } else {
                const int gr = e->rep_gr[r];
                m[0] = 1.0f;
                m[1] = gr >= 2 ? 1.0f : 0.0f;
                m[2] = gr >= 3 ? 1.0f : 0.0f;
                m[3] = m[4] = 0.0f;
            }
            const float eq = equity5(m);
            if (isnan(eq)) return -1;
            if (r == doff[k] || eq > best_eq) {
                memcpy(best, m, sizeof m);
                best_eq = eq;
            }
        }
        for (int c = 0; c < 5; c++) t[c] += DICE_P32[k] * best[c];
    }
    for (int c = 0; c < 5; c++) out[c] = t[c] < 0.0f ? 0.0f : t[c] > 1.0f ? 1.0f : t[c];
    return 0;
}

/* ── checker selection ─────────────────────────────────────────── */

/* 0-ply over the n candidates in e->plays. Fills cscore (mover view),
 * returns the argmin of the opponent-view values. */
static int select0(bgc_engine *e, const BoardState *s, int n, const bgc_cube *cube) {
    int opp = 1 - s->turn;
    PROF_T0(t0);
    for (int i = 0; i < n; i++) encode_base(e, i, &e->plays[i].resulting_state);
    PROF_ADD(t0, BGC_PT_ENCODE, n);
    int persp = PERSP_CENTERED, ica = 0;
    if (e->kind == BGC_KIND_CUBEFUL) {
        if (cube) { persp = cube_persp(cube->owner, opp); ica = can_offer(cube->owner, opp); }
        set_cube_cols(e, 0, n, persp, ica);
    }
    forward_succ(e, 0, n, s, persp, ica);
    PROF_T0(t1);
    int best = 0;
    float best_v = 0.0f;
    for (int i = 0; i < n; i++) {
        const BoardState *rs = &e->plays[i].resulting_state;
        float v = e->v[i];
        if (board_is_game_over(rs)) {
            int gr = game_result(rs);
            if (e->kind == BGC_KIND_PROB) v = 0.0f;
            else if (e->kind == BGC_KIND_CUBEFUL && cube && !gammons_count(cube)) v = -1.0f;
            else v = -(float)gr;
        }
        e->cscore[i] = e->kind == BGC_KIND_PROB ? 1.0 - (double)v : -(double)v;
        if (i == 0 || v < best_v) { best = i; best_v = v; }
    }
    PROF_ADD(t1, BGC_PT_REDUCE, n);
    return best;
}

/* Scores the batch slots [0, nslot) after their rows [0, nrow) are
 * expanded: cubeless kinds (TDAgent._choose_checker_oneply_cubeless,
 * ProbAgent._choose_1ply) or cubeful (TDAgent._choose_checker_oneply_cubeful). */
static int flush1(bgc_engine *e, int nslot, int nrow, const bgc_cube *cube, int mover) {
    if (nslot == 0) return 0;
    double mins[21];
    if (e->kind != BGC_KIND_CUBEFUL) {
        forward(e, 0, nrow);
        for (int s = 0; s < nslot; s++) {
            dice_mins(e, s, 0, e->kind == BGC_KIND_PROB, 1, mins);
            e->cscore[e->slot_cand[s]] = dice_sum(mins, e->cpass + (size_t)s * 21,
                                                  e->kind == BGC_KIND_PROB5, FLIP_NONE);
        }
        return 0;
    }
    int opp = 1 - mover;
    int opp_can = can_offer(cube->owner, opp);
    int opp_persp = cube_persp(cube->owner, opp);
    int gc_a = gammons_count(cube);
    set_cube_cols(e, 0, nrow, cube_persp(cube->owner, mover), can_offer(cube->owner, mover));
    if (opp_can) {
        if (ensure_rows(e, 2 * nrow)) return -1;
        memcpy(e->x + (size_t)nrow * e->ldx, e->x, (size_t)nrow * e->ldx * sizeof(float));
        set_cube_cols(e, nrow, 2 * nrow, PERSP_MINE, 1);
    }
    forward(e, 0, opp_can ? 2 * nrow : nrow);
    for (int s = 0; s < nslot; s++) {
        dice_mins(e, s, 0, 0, gc_a, mins);
        double u_a = dice_sum(mins, NULL, 0, FLIP_NONE);
        double val = u_a;
        if (opp_can) {
            dice_mins(e, s, nrow, 0, 1, mins);
            double u_b = dice_sum(mins, NULL, 0, FLIP_NONE);
            double v_dt_opp = -2.0 * u_b;
            if (money_decision(-u_a, v_dt_opp, opp_persp, cube->jacoby))
                val = v_dt_opp <= 1.0 ? 2.0 * u_b : -1.0;
        }
        e->cscore[e->slot_cand[s]] = val;
    }
    return 0;
}

/* 1-ply over the n candidates in e->plays (only those with keep[i] when
 * keep is non-NULL; the others score NAN); argmax of cscore. */
static int select1(bgc_engine *e, int n, const unsigned char *keep,
                   const bgc_cube *cube, int mover) {
    int gc = (e->kind == BGC_KIND_CUBEFUL) ? gammons_count(cube) : 1;
    int nslot = 0, nrow = 0, nrep = 0;
    for (int i = 0; i < n; i++) {
        if (keep && !keep[i]) {
            e->cscore[i] = NAN;
            continue;
        }
        const BoardState *rs = &e->plays[i].resulting_state;
        if (board_is_game_over(rs)) {
            int gr = game_result(rs);
            e->cscore[i] = (e->kind == BGC_KIND_PROB || !gc) ? 1.0 : (double)gr;
            continue;
        }
        e->slot_cand[nslot] = i;
        if (expand21(e, rs, nslot, -1, &nrow, &nrep)) return -2;
        nslot++;
        if (nrow >= BGC_CHUNK) {
            if (flush1(e, nslot, nrow, cube, mover)) return -2;
            nslot = nrow = nrep = 0;
        }
    }
    if (flush1(e, nslot, nrow, cube, mover)) return -2;
    int best = -1;
    double best_v = -INFINITY;
    for (int i = 0; i < n; i++)
        if ((!keep || keep[i]) && (best < 0 || e->cscore[i] > best_v)) {
            best_v = e->cscore[i];
            best = i;
        }
    return best;
}

/* 1-ply behind the root filter: 0-ply scores of every candidate, then
 * select1 over those within root_thresh of the best (game-ending
 * candidates always kept). A single survivor keeps its 0-ply score. */
static int select1_root(bgc_engine *e, const BoardState *s, int n, const bgc_cube *cube,
                        int mover) {
    select0(e, s, n, cube);
    double best0 = -INFINITY;
    for (int i = 0; i < n; i++) if (e->cscore[i] > best0) best0 = e->cscore[i];
    int nkeep = 0, only = -1;
    for (int i = 0; i < n; i++) {
        e->ckeep[i] = board_is_game_over(&e->plays[i].resulting_state)
                      || best0 - e->cscore[i] <= (double)e->root_thresh;
        if (e->ckeep[i]) { nkeep++; only = i; }
    }
    if (nkeep == 1) {
        for (int i = 0; i < n; i++) if (i != only) e->cscore[i] = NAN;
        return only;
    }
    return select1(e, n, e->ckeep, cube, mover);
}

int bgc_set_search(bgc_engine *e, const bgc_search *o) {
    if (!e) return -1;
    float t = o ? o->root_thresh : 0.0f;
    if (!(t >= 0.0f) || isinf(t)) return -1;
    if (e->kind == BGC_KIND_PROB5JAN && t > 0.0f) return -1;      /* full width only */
    e->root_thresh = t;
    return 0;
}

/* Generates the candidates into e->plays and selects. *n_out = count.
 * Returns the index, -1 on a forced pass, -2 on error. */
static int choose(bgc_engine *e, const BoardState *s, int d1, int d2,
                  const bgc_cube *cube, int plies, int *n_out) {
    *n_out = 0;
    if (!e || !s || d1 < 1 || d1 > 6 || d2 < 1 || d2 > 6 || (plies != 0 && plies != 1))
        return -2;
    PROF_T0(t0);
    int n = get_legal_plays(s, d1, d2, e->plays, MAX_PLAYS);
    PROF_ADD(t0, BGC_PT_MOVEGEN, n);
    *n_out = n;
    if (n == 0) return -1;
    if (e->kind == BGC_KIND_PROB5JAN) return bgc_i_jan_choose(e, s, n, cube, plies);
    if (e->kind == BGC_KIND_CUBEFUL && !cube) return select0(e, s, n, NULL);
    if (plies == 0) return select0(e, s, n, cube);
    if (e->root_thresh > 0.0f && n > 1) return select1_root(e, s, n, cube, s->turn);
    return select1(e, n, NULL, cube, s->turn);
}

/* Board hash for value1_play's candidate matching (equal keys are then
 * compared in full). */
static uint64_t board_key(const BoardState *s) {
    uint64_t h = (uint64_t)(s->bar[0] + 31 * s->bar[1] + 961 * s->off[0] + 29791 * s->off[1]);
    for (int i = 0; i < NUM_POINTS; i++) h = h * 0x9E3779B97F4A7C15ULL + (uint64_t)(s->points[i] + 16);
    return h;
}

/* select0's scoring of the n candidates in e->plays, their values taken
 * from value1k's batch: candidate i is reply e->pidx[i] of slot 0. */
static int score_from_replies(bgc_engine *e, int n, const bgc_cube *cube) {
    int best = 0;
    float best_v = 0.0f;
    for (int i = 0; i < n; i++) {
        const int r = e->pidx[i];
        float v;
        if (e->rep_row[r] >= 0) {
            v = e->v[e->rep_row[r]];
        } else {                     /* terminal: select0's mask */
            if (e->kind == BGC_KIND_PROB) v = 0.0f;
            else if (e->kind == BGC_KIND_CUBEFUL && cube && !gammons_count(cube)) v = -1.0f;
            else v = -(float)e->rep_gr[r];
        }
        e->cscore[i] = e->kind == BGC_KIND_PROB ? 1.0 - (double)v : -(double)v;
        if (i == 0 || v < best_v) { best = i; best_v = v; }
    }
    return best;
}

/* The 0-ply play for (d1, d2) after value1k(X, cube, dice_index(d1, d2)):
 * the roll's candidates are expansion replies, encoded with select0's
 * cube columns (value1's q is the mover's opponent: cube_persp(owner,
 * opp), can_offer(owner, opp)), so their values are already in the batch.
 * The candidate order is get_legal_plays(X, d1, d2)'s: the expansion's
 * own order when d1 <= d2 (the same call); for d1 > d2 the roll is
 * regenerated in that order and matched to the replies by board (the
 * same set either way). Returns choose()'s index, -1 pass, -3 when the
 * match fails (the caller then falls back to choose). */
static int play_from_replies(bgc_engine *e, const BoardState *X, int d1, int d2,
                             const bgc_cube *cube) {
    const int k = dice_index(d1, d2);
    if (e->cpass[k]) return -1;
    const int r0 = e->cdoff[k], n = e->cdoff[k + 1] - r0;
    if (d1 <= d2) {
        for (int i = 0; i < n; i++) e->pidx[i] = r0 + i;
    } else {
        PROF_T0(t0);
        int n2 = get_legal_plays(X, d1, d2, e->iplays, MAX_PLAYS);
        PROF_ADD(t0, BGC_PT_MOVEGEN, n2);
        if (n2 != n) return -3;
        for (int j = 0; j < n; j++) e->pkey[j] = board_key(&e->plays[j].resulting_state);
        for (int i = 0; i < n; i++) {
            const BoardState *rs = &e->iplays[i].resulting_state;
            const uint64_t key = board_key(rs);
            int j = 0;
            while (j < n && !(e->pkey[j] == key &&
                              !memcmp(&e->plays[j].resulting_state, rs, sizeof(BoardState))))
                j++;
            if (j == n) return -3;
            e->pidx[i] = r0 + j;
        }
        memcpy(e->plays, e->iplays, (size_t)n * sizeof(Play));
    }
    PROF_T0(t1);
    int best = score_from_replies(e, n, cube);
    PROF_ADD(t1, BGC_PT_REDUCE, n);
    return best;
}

/* value1 of X and the 0-ply play for (d1, d2): bgc_value_1ply and
 * bgc_best_move(plies 0) from one expansion and one forward on a
 * row-exact backend, the two calls otherwise. *v1 = value1's double;
 * vec5 (nullable, PROB5 only): vec1_prob5's target vector from the same
 * forward. Returns choose()'s index (candidates and scores in e->plays /
 * e->cscore), -1 forced pass, -2 error. */
static int value1_play(bgc_engine *e, const BoardState *X, const bgc_cube *cube,
                       int d1, int d2, double *v1, float *vec5) {
    int err, n;
    if (vec5 && (e->kind != BGC_KIND_PROB5 || board_is_game_over(X))) return -2;
    const int fast = e->row_exact && !e->bf16 && d1 >= 1 && d1 <= 6 && d2 >= 1 && d2 <= 6
                     && !board_is_game_over(X) && (e->kind != BGC_KIND_CUBEFUL || cube)
                     && e->kind != BGC_KIND_PROB5JAN;
    {
        PROF_ENTER(BGC_PC_V1);
        *v1 = value1k(e, X, cube, fast ? dice_index(d1, d2) : -1, &err);
        if (!err && vec5) {
            PROF_T0(t0);
            err = vec1_prob5(e, vec5);
            PROF_ADD(t0, BGC_PT_REDUCE, 0);
        }
        PROF_LEAVE(1);
    }
    if (err) return -2;
    int idx = -3;
    if (fast) {
        PROF_ENTER(BGC_PC_SEL);
        idx = play_from_replies(e, X, d1, d2, cube);
        PROF_LEAVE(1);
    }
    if (idx == -3) {
        PROF_ENTER(BGC_PC_SEL);
        idx = choose(e, X, d1, d2, cube, 0, &n);
        PROF_LEAVE(1);
    }
    return idx;
}

int bgc_i_value1_play(bgc_engine *e, const BoardState *s, const bgc_cube *cube,
                      int d1, int d2, double *v1, float *vec5, BoardState *out_state,
                      float *out_value) {
    int idx = value1_play(e, s, cube, d1, d2, v1, vec5);
    if (idx >= 0) {
        if (out_state) *out_state = e->plays[idx].resulting_state;
        if (out_value) *out_value = (float)e->cscore[idx];
    }
    return idx;
}

int bgc_i_best_move_prob5(bgc_engine *e, const BoardState *s, int d1, int d2,
                          BoardState *out_state, float probs[5]) {
    int n;
    if (e->kind != BGC_KIND_PROB5) return -2;
    int idx = choose(e, s, d1, d2, NULL, 0, &n);
    if (idx < 0) return idx;
    /* select0 forwards candidate i in batch row i; the batch is intact. */
    *out_state = e->plays[idx].resulting_state;
    prob5_post(e->y + (size_t)idx * 5, probs);
    return idx;
}

/* ── hidden helpers for puretd_janowski.c (puretd_internal.h) ── */

struct bgc_jan *bgc_i_jan(bgc_engine *e) { return e->jan; }

void bgc_i_batch(bgc_engine *e, bgc_batch *b) {
    b->x = e->x; b->y = e->y; b->ldx = e->ldx; b->out_dim = e->out_dim;
    b->plays = e->plays; b->iplays = e->iplays;
    b->cdoff = e->cdoff; b->cpass = e->cpass;
    b->rep_row = e->rep_row; b->rep_gr = e->rep_gr;
    b->cscore = e->cscore;
}

int bgc_i_expand21(bgc_engine *e, const BoardState *X, int slot, int keep, int *nrow, int *nrep) {
    return expand21(e, X, slot, keep, nrow, nrep);
}

void bgc_i_encode_base(bgc_engine *e, int r, const BoardState *s) { encode_base(e, r, s); }
void bgc_i_forward(bgc_engine *e, int r0, int n) { forward(e, r0, n); }
void bgc_i_forward_succ(bgc_engine *e, int r0, int n, const BoardState *X, int persp, int ica) {
    forward_succ(e, r0, n, X, persp, ica);
}
int bgc_i_ensure_rows(bgc_engine *e, int n) { return ensure_rows(e, n); }

int bgc_value_1ply_play(bgc_engine *e, const BoardState *s, int d1, int d2,
                        const bgc_cube *cube, float *value_1ply,
                        BoardState *out_state, float *out_value) {
    if (value_1ply) *value_1ply = NAN;
    if (!e || !s) return -2;
    double v1;
    int idx = bgc_i_value1_play(e, s, cube, d1, d2, &v1, NULL, out_state, out_value);
    if (idx != -2 && value_1ply) *value_1ply = (float)v1;
    return idx;
}

int bgc_best_move(bgc_engine *e, const BoardState *s, int d1, int d2,
                  const bgc_cube *cube, int plies,
                  BoardState *out_state, Play *out_play, float *out_value) {
    int n;
    int idx = choose(e, s, d1, d2, cube, plies, &n);
    if (idx < 0) return idx;
    if (out_state) *out_state = e->plays[idx].resulting_state;
    if (out_play) *out_play = e->plays[idx];
    if (out_value) *out_value = (float)e->cscore[idx];
    return idx;
}

int bgc_score_moves(bgc_engine *e, const BoardState *s, int d1, int d2,
                    const bgc_cube *cube, int plies,
                    float *scores, BoardState *states) {
    int n;
    int idx = choose(e, s, d1, d2, cube, plies, &n);
    if (idx == -2) return -2;
    for (int i = 0; i < n; i++) {
        if (scores) scores[i] = (float)e->cscore[i];
        if (states) states[i] = e->plays[i].resulting_state;
    }
    return n;
}

/* ── cube ──────────────────────────────────────────────────────── */

/* V(state, THEIRS from state->turn's side, is_cube_action 0) at plies,
 * times 2: the double/take equity of the player on roll. */
static double double_take(bgc_engine *e, const BoardState *s, const bgc_cube *cube,
                          int plies, int *err) {
    *err = 0;
    if (plies == 0) {
        encode_base(e, 0, s);
        set_cube_cols(e, 0, 1, PERSP_THEIRS, 0);
        forward(e, 0, 1);
        return 2.0 * (double)e->v[0];
    }
    bgc_cube theirs = *cube;
    theirs.owner = (1 - s->turn) + 1;
    return 2.0 * value1(e, s, &theirs, err);
}

int bgc_offer_double(bgc_engine *e, const BoardState *s, const bgc_cube *cube, int plies,
                     bgc_cube_eval *out) {
    if (out) { out->plies = plies; out->v_no_double = NAN; out->v_double_take = NAN; out->take = 0; }
    if (e->kind == BGC_KIND_PROB5JAN) return bgc_i_jan_offer(e, s, cube, plies, out);
    if (e->kind != BGC_KIND_CUBEFUL) return 0;
    if (!cube || (plies != 0 && plies != 1)) return -1;
    int persp_nd = cube_persp(cube->owner, s->turn);
    if (persp_nd == PERSP_THEIRS) return 0;
    double v_nd, v_dt;
    if (plies == 0) {
        PROF_T0(t0);
        encode_base(e, 0, s);
        encode_base(e, 1, s);
        PROF_ADD(t0, BGC_PT_ENCODE, 2);
        set_cube_cols(e, 0, 1, persp_nd, 0);
        set_cube_cols(e, 1, 2, PERSP_THEIRS, 0);
        forward(e, 0, 2);
        v_nd = (double)e->v[0];
        v_dt = 2.0 * (double)e->v[1];
    } else {
        bgc_cube theirs = *cube;
        theirs.owner = (1 - s->turn) + 1;
        if (value1_pair(e, s, cube, &theirs, &v_nd, &v_dt)) return -1;
        v_dt *= 2.0;
    }
    if (out) { out->v_no_double = (float)v_nd; out->v_double_take = (float)v_dt; out->take = v_dt <= 1.0; }
    return money_decision(v_nd, v_dt, persp_nd, cube->jacoby);
}

int bgc_respond_to_double(bgc_engine *e, const BoardState *s, const bgc_cube *cube, int plies,
                          const bgc_cube_eval *hint) {
    if (e->kind == BGC_KIND_PROB5JAN) return bgc_i_jan_respond(e, s, cube, plies, hint);
    if (e->kind != BGC_KIND_CUBEFUL) return 1;
    if (!cube || (plies != 0 && plies != 1)) return -1;
    double v_dt;
    if (hint && hint->plies == plies) {
        v_dt = (double)hint->v_double_take;
    } else {
        int err;
        v_dt = double_take(e, s, cube, plies, &err);
        if (err) return -1;
    }
    return v_dt <= 1.0;
}

/* ── RNG (CPython random: MT19937, init_by_array seeding) ──────── */

#define MT_N 624
#define MT_M 397

static void mt_init_genrand(bgc_rng *r, uint32_t s) {
    r->mt[0] = s;
    for (int i = 1; i < MT_N; i++)
        r->mt[i] = 1812433253U * (r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) + (uint32_t)i;
    r->idx = MT_N;
}

void bgc_rng_seed(bgc_rng *r, unsigned long long seed) {
    uint32_t key[2];
    int klen;
    key[0] = (uint32_t)(seed & 0xffffffffULL);
    key[1] = (uint32_t)(seed >> 32);
    klen = key[1] ? 2 : 1;
    mt_init_genrand(r, 19650218U);
    int i = 1, j = 0;
    for (int k = (MT_N > klen ? MT_N : klen); k; k--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1664525U))
                   + key[j] + (uint32_t)j;
        i++; j++;
        if (i >= MT_N) { r->mt[0] = r->mt[MT_N - 1]; i = 1; }
        if (j >= klen) j = 0;
    }
    for (int k = MT_N - 1; k; k--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1566083941U))
                   - (uint32_t)i;
        i++;
        if (i >= MT_N) { r->mt[0] = r->mt[MT_N - 1]; i = 1; }
    }
    r->mt[0] = 0x80000000U;
    r->idx = MT_N;
}

unsigned int bgc_rng_u32(bgc_rng *r) {
    static const uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
    uint32_t y;
    if (r->idx >= MT_N) {
        int kk;
        for (kk = 0; kk < MT_N - MT_M; kk++) {
            y = (r->mt[kk] & 0x80000000U) | (r->mt[kk + 1] & 0x7fffffffU);
            r->mt[kk] = r->mt[kk + MT_M] ^ (y >> 1) ^ mag01[y & 1U];
        }
        for (; kk < MT_N - 1; kk++) {
            y = (r->mt[kk] & 0x80000000U) | (r->mt[kk + 1] & 0x7fffffffU);
            r->mt[kk] = r->mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 1U];
        }
        y = (r->mt[MT_N - 1] & 0x80000000U) | (r->mt[0] & 0x7fffffffU);
        r->mt[MT_N - 1] = r->mt[MT_M - 1] ^ (y >> 1) ^ mag01[y & 1U];
        r->idx = 0;
    }
    y = r->mt[r->idx++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

unsigned int bgc_rng_getrandbits(bgc_rng *r, int k) {
    if (k < 1 || k > 32) abort();                 /* header: 1 <= k <= 32 */
    return bgc_rng_u32(r) >> (32 - k);
}

int bgc_rng_randint(bgc_rng *r, int a, int b) {
    if (b < a) abort();                           /* Python raises ValueError */
    unsigned int n = (unsigned int)(b - a + 1);
    int k = 0;
    for (unsigned int t = n; t; t >>= 1) k++;      /* bit_length(n) */
    unsigned int v = bgc_rng_getrandbits(r, k);
    while (v >= n) v = bgc_rng_getrandbits(r, k);
    return a + (int)v;
}

/* ── self-play ─────────────────────────────────────────────────── */

/* backgammon_engine.opening_roll */
static void opening_roll(bgc_rng *rng, BoardState *s, int *d1, int *d2, int *n_dice) {
    int w, b;
    do {
        w = bgc_rng_randint(rng, 1, 6);
        b = bgc_rng_randint(rng, 1, 6);
        *n_dice += 2;
    } while (w == b);
    board_init(s);
    s->turn = w > b ? WHITE : BLACK;
    *d1 = w > b ? w : b;
    *d2 = w > b ? b : w;
}

/* Exposed to puretd_collect.c (puretd_internal.h). */
void bgc_i_opening_roll(bgc_rng *rng, BoardState *s, int *d1, int *d2, int *n_dice) {
    opening_roll(rng, s, d1, d2, n_dice);
}

int bgc_i_game_result(const BoardState *s) { return game_result(s); }

int bgc_play_game(bgc_engine *white, bgc_engine *black, const bgc_game_opts *o,
                  bgc_rng *rng, bgc_game_result *res, bgc_record_fn rec, void *ctx) {
    if (!white || !black || !o || !rng || !res) return -1;
    memset(res, 0, sizeof(*res));
    BoardState s, next;
    int od1, od2;
    opening_roll(rng, &s, &od1, &od2, &res->n_dice);
    bgc_cube cube = {0, 1, o->jacoby ? 1 : 0};
    const bgc_cube *cp = o->cubeful ? &cube : NULL;
    int first = 1;

    while (!board_is_game_over(&s)) {
        int player = s.turn;
        bgc_engine *eng = player == WHITE ? white : black;
        bgc_engine *opp = player == WHITE ? black : white;
        int plies = player == WHITE ? o->plies_white : o->plies_black;
        int opp_plies = player == WHITE ? o->plies_black : o->plies_white;

        if (o->cubeful && !first && can_offer(cube.owner, player)) {
            bgc_cube_eval ce = {plies, NAN, NAN};
            int dbl = 0;
            res->n_cube_offer++;
            if (o->cube_enabled) {
                dbl = bgc_offer_double(eng, &s, &cube, plies, &ce);
                if (dbl < 0) return -1;
            }
            if (rec) rec(ctx, 1, &s, 0, 0, &cube, dbl, NULL, ce.v_double_take);
            if (dbl) {
                int take = bgc_respond_to_double(opp, &s, &cube, opp_plies,
                                                 opp == eng ? &ce : NULL);
                if (take < 0) return -1;
                res->n_cube_resp++;
                if (rec) rec(ctx, 2, &s, 0, 0, &cube, take, NULL, ce.v_double_take);
                if (take) {
                    cube.owner = (1 - player) + 1;
                    cube.value *= 2;
                } else {
                    res->winner = player;
                    res->result = 1;
                    res->cube_value = cube.value;
                    res->ended_by_drop = 1;
                    return 0;
                }
            }
        }

        int d1, d2;
        if (first) { d1 = od1; d2 = od2; }
        else {
            d1 = bgc_rng_randint(rng, 1, 6);
            d2 = bgc_rng_randint(rng, 1, 6);
            res->n_dice += 2;
        }
        first = 0;
        res->n_turns++;
        float val = NAN;
        int idx = bgc_best_move(eng, &s, d1, d2, cp, plies, &next, NULL, &val);
        if (idx == -2) return -1;
        if (o->cubeful || idx >= 0) res->n_checker++;
        if (rec) rec(ctx, 0, &s, d1, d2, cp, idx, idx >= 0 ? &next : NULL, idx >= 0 ? val : NAN);
        if (idx >= 0) {
            s = next;
        } else {
            res->n_dance++;
            board_switch_turn(&s);
        }
    }
    res->winner = board_winner(&s);
    res->result = game_result(&s);
    if (o->cubeful && cube.jacoby && cube.owner == 0) res->result = 1;
    res->cube_value = cube.value;
    return 0;
}

/* ── benchmark entry points ────────────────────────────────────── */

int bgc_bench_checker(bgc_engine *e, const BoardState *states, const int *dice,
                      const bgc_cube *cubes, int n, int plies, double *out_seconds) {
    BoardState next;
    for (int i = 0; i < n; i++) {
        double t0 = now_s();
        int idx = bgc_best_move(e, &states[i], dice[2 * i], dice[2 * i + 1],
                                cubes ? &cubes[i] : NULL, plies, &next, NULL, NULL);
        out_seconds[i] = now_s() - t0;
        if (idx == -2) return -1;
    }
    return 0;
}

int bgc_bench_cube(bgc_engine *e, const BoardState *states, const bgc_cube *cubes,
                   int n, int plies, double *out_seconds) {
    bgc_cube_eval ce;
    for (int i = 0; i < n; i++) {
        double t0 = now_s();
        int r = bgc_offer_double(e, &states[i], &cubes[i], plies, &ce);
        out_seconds[i] = now_s() - t0;
        if (r < 0) return -1;
    }
    return 0;
}

int bgc_bench_selfplay(bgc_engine *e, const bgc_game_opts *o, unsigned long long seed,
                       int warmup_games, int reseed_after_warmup, int games,
                       double *out_game_seconds, int *out_game_decisions,
                       long long totals[6]) {
    if (!e || !o) return -1;
    bgc_rng rng;
    bgc_game_result r;
    bgc_game_opts wo = *o;
    if (wo.cubeful) wo.cube_enabled = 1;
    bgc_rng_seed(&rng, seed);
    for (int g = 0; g < warmup_games; g++)
        if (bgc_play_game(e, e, &wo, &rng, &r, NULL, NULL)) return -1;
    if (reseed_after_warmup) bgc_rng_seed(&rng, seed);
    if (totals) memset(totals, 0, 6 * sizeof(long long));
    for (int g = 0; g < games; g++) {
        double t0 = now_s();
        if (bgc_play_game(e, e, o, &rng, &r, NULL, NULL)) return -1;
        double dt = now_s() - t0;
        /* Forced passes are not decisions (gnubg's count): n_checker less
           the cubeful loop's passes. */
        int nck = r.n_checker - (o->cubeful ? r.n_dance : 0);
        int dec = nck;
        if (o->cubeful && o->cube_enabled) dec += r.n_cube_offer + r.n_cube_resp;
        if (out_game_seconds) out_game_seconds[g] = dt;
        if (out_game_decisions) out_game_decisions[g] = dec;
        if (totals) {
            totals[0] += nck;
            totals[1] += r.n_dance;
            totals[2] += r.n_cube_offer;
            totals[3] += r.n_cube_resp;
            totals[4] += r.n_turns;
            totals[5] += 1;
        }
    }
    return 0;
}
