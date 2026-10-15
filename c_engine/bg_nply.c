/*
 * bg_nply.c -- Fused move generation + neural-net N-ply search in C.
 *
 * Runs 0/1/2-ply checker evaluation entirely in C (bg_engine move
 * generator, nn_eval weights, BLAS sgemm), with no Python in the inner
 * loop. Built by build_nply.sh into libbg_nply.so; called via ctypes
 * from c_engine/bg_nply.py.
 *
 * Cubeful-money conventions match td_agent.py:
 *   - per-unit equity at cube_value=1 (equity output mode)
 *   - successors encoded from the opponent's cube perspective
 *   - terminal magnitude = game_result (1/2/3), Jacoby-aware
 */

#include "bg_engine.h"
#include "../c_inference/nn_eval.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* sgemm for the batched layer matmuls, declared here to avoid a cblas.h
 * dependency; resolved at link (see build_nply.sh). Forced single-thread. */
extern void cblas_sgemm(int Order, int TransA, int TransB,
                        int M, int N, int K, float alpha,
                        const float *A, int lda, const float *B, int ldb,
                        float beta, float *C, int ldc);
extern void openblas_set_num_threads(int n);
#define CBLAS_ROW_MAJOR 101
#define CBLAS_NO_TRANS  111
#define CBLAS_TRANS     112

/* ── Global model + scratch ────────────────────────────────────────── */

static NNModel g_model;
static int g_loaded = 0;
static int g_in_size = 0;     /* model input size (196 cubeless / 200 cubeful) */
static int g_n_base = 196;    /* base perspective196 feature count */
static int g_max_dim = 0;     /* widest layer (for scratch) */
static int g_is_prob5 = 0;    /* output_mode == prob5 (5 outputs -> equity) */
static int g_out_dim = 1;     /* output layer width (1, or 5 for prob5) */

/* Clamp the nested-event inequalities, then reduce 5 sigmoids to money
 * equity (mirrors prob5_reduce in nn_eval.c). */
static float prob5_reduce_c(const float *r) {
    float p0 = r[0], p1 = r[1], p2 = r[2], p3 = r[3], p4 = r[4];
    if (p1 > p0) p1 = p0;
    float lose = 1.0f - p0;
    if (p3 > lose) p3 = lose;
    if (p2 > p1) p2 = p1;
    if (p4 > p3) p4 = p3;
    return 2.0f * p0 + p1 + p2 - p3 - p4 - 1.0f;
}

/* Value-mode flags for the cubeless 1/2-ply reduction. */
#define FLIP_EQUITY 0    /* mover_eq = -V(opp)      (equity / prob5 money) */
#define FLIP_PROB   1    /* mover_eq = 1 - V(opp)   (probability / DMP)    */
#define TERM_GAMMON 0    /* terminal mover_eq = game_result (1/2/3)        */
#define TERM_WIN    1    /* terminal mover_eq = 1.0 (DMP win/lose)         */

/* Sizing for one 21-dice expansion; far above any real position. */
#define BGN_MAX_ROWS    70000
#define BGN_MAX_REPLIES 90000

static float *g_feats   = NULL;   /* expand_21 base feats: ROWS*196          */
static int    g_counts[21];
static int   *g_reply_gr = NULL;  /* REPLIES                                 */
static float *g_in       = NULL;  /* forward input batch: ROWS*in_size       */
static float *g_bufA     = NULL;  /* forward scratch: ROWS*max_dim           */
static float *g_bufB     = NULL;  /* forward scratch: ROWS*max_dim           */
static float *g_val      = NULL;  /* per-row output value: ROWS              */
static Play  *g_plays    = NULL;  /* move-gen buffer for 0-ply best-move     */

/* Level-1 candidate scratch for 2-ply (shared by cubeful + generic). */
#define BGN_L1_MAX 8192      /* max level-1 non-terminal candidates */
#define BGN_TOPK_MAX 8
static BoardState g_l1_states[BGN_L1_MAX];   /* level-1 successor states  */
static int        g_l1_dice[BGN_L1_MAX];     /* which dice each belongs to */
static float      g_l1_eq[BGN_L1_MAX];       /* level-1 0-ply mover equity */

/* 21-dice probabilities in (d1=1..6, d2=d1..6) order = _ONEPLY_PROBS. */
static float g_probs[21];

static void init_probs(void) {
    int k = 0;
    for (int d1 = 1; d1 <= 6; d1++)
        for (int d2 = d1; d2 <= 6; d2++)
            g_probs[k++] = (d1 == d2) ? (1.0f / 36.0f) : (2.0f / 36.0f);
}

/* ── Terminal magnitude ────────────────────────────────────────────── */

/* 1/2/3 from the winner's view (mirrors game_result_c in bg_engine.c). */
static int gr_c(const BoardState *s) {
    int w = (s->off[WHITE] == NUM_CHECKERS) ? WHITE
          : (s->off[BLACK] == NUM_CHECKERS) ? BLACK : -1;
    if (w < 0) return 0;
    int loser = 1 - w;
    if (s->off[loser] > 0) return 1;                 /* normal */
    /* loser has borne off none: gammon unless backgammon */
    int loser_in_opp_home = 0;
    if (w == WHITE) {           /* WHITE home = points 0..5 */
        for (int p = 0; p < 6; p++)
            if (s->points[p] < 0) { loser_in_opp_home = 1; break; }
    } else {                    /* BLACK home = points 18..23 */
        for (int p = 18; p < 24; p++)
            if (s->points[p] > 0) { loser_in_opp_home = 1; break; }
    }
    if (s->bar[loser] > 0 || loser_in_opp_home) return 3;  /* backgammon */
    return 2;                                              /* gammon */
}

/* ── Forward pass (batched, layer-at-a-time) ───────────────────────── */

static inline float relu_f(float x) { return x > 0.0f ? x : 0.0f; }
static inline float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

/* Run the net over the `n` inputs staged in g_in (row-major
 * n*g_in_size); out[n] gets one scalar per row. Per layer: one sgemm,
 * then fused bias + activation. Activations live in g_bufA/B. */
static void forward_batch(int n, float *out) {
    const int total = g_model.num_hidden + 1;
    float *A = g_bufA, *B = g_bufB;
    memcpy(A, g_in, (size_t)n * g_in_size * sizeof(float));
    int cols = g_in_size;

    for (int L = 0; L < total; L++) {
        const int rows = g_model.layer_out[L];
        const float *W = g_model.weight[L];      /* [rows x cols] row-major */
        const float *b = g_model.bias[L];
        const int is_hidden = (L < g_model.num_hidden);
        const int equity = (g_model.output_mode == NN_OUTPUT_EQUITY);

        /* B[n x rows] = A[n x cols] * W^T[cols x rows]  (blocked GEMM) */
        cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                    n, rows, cols, 1.0f,
                    A, cols, W, cols, 0.0f, B, rows);

        /* bias + activation, fused (vectorizes over the contiguous row) */
        for (int s = 0; s < n; s++) {
            float *y = B + (size_t)s * rows;
            for (int i = 0; i < rows; i++) {
                float v = y[i] + b[i];
                if (is_hidden)      y[i] = relu_f(v);
                else if (equity)    y[i] = v;
                else                y[i] = sigmoid_f(v);
            }
        }
        float *tmp = A; A = B; B = tmp;
        cols = rows;
    }
    /* A holds [n * g_out_dim]. Reduce to one scalar per row. */
    if (g_is_prob5)
        for (int s = 0; s < n; s++) out[s] = prob5_reduce_c(A + (size_t)s * 5);
    else
        for (int s = 0; s < n; s++) out[s] = A[s];
}

/* As forward_batch, but out[n*g_out_dim] gets the raw output-layer
 * activations with no scalar reduce (prob5: the 5 sigmoids, as
 * model.ProbNetwork(x) with raw_logits=False). Used by bgn_forward5. */
static void forward_batch_raw(int n, float *out) {
    const int total = g_model.num_hidden + 1;
    float *A = g_bufA, *B = g_bufB;
    memcpy(A, g_in, (size_t)n * g_in_size * sizeof(float));
    int cols = g_in_size;

    for (int L = 0; L < total; L++) {
        const int rows = g_model.layer_out[L];
        const float *W = g_model.weight[L];
        const float *b = g_model.bias[L];
        const int is_hidden = (L < g_model.num_hidden);
        const int equity = (g_model.output_mode == NN_OUTPUT_EQUITY);

        cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                    n, rows, cols, 1.0f,
                    A, cols, W, cols, 0.0f, B, rows);

        for (int s = 0; s < n; s++) {
            float *y = B + (size_t)s * rows;
            for (int i = 0; i < rows; i++) {
                float v = y[i] + b[i];
                if (is_hidden)      y[i] = relu_f(v);
                else if (equity)    y[i] = v;
                else                y[i] = sigmoid_f(v);
            }
        }
        float *tmp = A; A = B; B = tmp;
        cols = rows;
    }
    /* A holds [n * g_out_dim]. Copy out verbatim (no reduce). */
    memcpy(out, A, (size_t)n * g_out_dim * sizeof(float));
}

/* ── Load / unload ─────────────────────────────────────────────────── */

/* 0; -1 unreadable, -2 input size not 196 or 200, -3 out of memory. */
int bgn_load(const char *path) {
    if (g_loaded) { nn_free(&g_model); g_loaded = 0; }
    if (nn_load(&g_model, path) != 0) return -1;
    if (g_model.input_size != 196 && g_model.input_size != 200) {
        nn_free(&g_model);
        return -2;
    }
    g_is_prob5 = (g_model.output_mode == NN_OUTPUT_PROB5);
    g_out_dim = g_is_prob5 ? NN_PROB5_OUTPUTS : 1;

    g_in_size = g_model.input_size;
    g_n_base = 196;
    g_max_dim = g_in_size;
    for (int i = 0; i < g_model.num_hidden; i++)
        if (g_model.hidden_sizes[i] > g_max_dim) g_max_dim = g_model.hidden_sizes[i];

    if (!g_feats)    g_feats    = malloc((size_t)BGN_MAX_ROWS * 196 * sizeof(float));
    if (!g_reply_gr) g_reply_gr = malloc((size_t)BGN_MAX_REPLIES * sizeof(int));
    if (!g_in)       g_in       = malloc((size_t)BGN_MAX_ROWS * g_in_size * sizeof(float));
    if (!g_bufA)     g_bufA     = malloc((size_t)BGN_MAX_ROWS * g_max_dim * sizeof(float));
    if (!g_bufB)     g_bufB     = malloc((size_t)BGN_MAX_ROWS * g_max_dim * sizeof(float));
    if (!g_val)      g_val      = malloc((size_t)BGN_MAX_ROWS * sizeof(float));
    if (!g_plays)    g_plays    = malloc((size_t)MAX_PLAYS * sizeof(Play));
    if (!g_feats || !g_reply_gr || !g_in || !g_bufA || !g_bufB || !g_val || !g_plays)
        return -3;

    openblas_set_num_threads(1);   /* never oversubscribe */
    init_probs();
    g_loaded = 1;
    return 0;
}

void bgn_unload(void) {
    if (g_loaded) { nn_free(&g_model); g_loaded = 0; }
}

/* ── 0-ply value ───────────────────────────────────────────────────── */

/* Stage row `row` of g_in: the base 196 features plus, for cubeful
 * nets, the cube block. Width follows the loaded model: 196 (cubeless)
 * or 200 (persp one-hot + is_cube_action). */
static inline void build_input(int row, const float *base196,
                               int cube_persp, int is_cube_action) {
    float *dst = g_in + (size_t)row * g_in_size;
    memcpy(dst, base196, 196 * sizeof(float));
    if (g_in_size > 196) {           /* cubeful: append the cube block */
        dst[196 + 0] = 0.0f; dst[196 + 1] = 0.0f; dst[196 + 2] = 0.0f;
        dst[196 + cube_persp] = 1.0f;
        dst[199] = is_cube_action ? 1.0f : 0.0f;
    }
}

float bgn_eval_cubeful(const BoardState *s, int cube_persp, int is_cube_action) {
    float base[196];
    encode_state(s, base);
    build_input(0, base, cube_persp, is_cube_action);
    float out;
    forward_batch(1, &out);
    return out;
}

/* Batched raw 5-output forward for prob5 1-ply targets. `feats` is
 * row-major n*196; `out` receives n*5 sigmoids. Returns 0, -1 if no
 * model / not prob5, -2 on overflow. */
int bgn_forward5(const float *feats, int n, float *out) {
    if (!g_loaded || !g_is_prob5) return -1;
    if (n <= 0) return 0;
    if (n > BGN_MAX_ROWS) return -2;
    for (int row = 0; row < n; row++)
        build_input(row, feats + (size_t)row * 196, 0, 0);
    forward_batch_raw(n, out);
    return 0;
}

/* ── 1-ply value (cubeful money) ───────────────────────────────────── */

/* Mirrors TDAgent.value_oneply_checker_cubeful: over the on-roll
 * player's 21 dice, the best reply per dice of { -V(opp successor),
 * terminal game_result }, weighted by dice probability.
 *
 * opp_persp / opp_cube_action / gammons_count come from Python (the
 * opponent's cube perspective after the move). Returns the per-unit
 * equity, or NAN on expansion overflow. */
float bgn_value_1ply_cubeful(const BoardState *s, int opp_persp,
                             int opp_cube_action, int gammons_count) {
    int nrep = get_legal_plays_encoded_21(s, g_feats, g_counts, g_reply_gr,
                                          BGN_MAX_ROWS, BGN_MAX_REPLIES);
    if (nrep < 0) return NAN;

    /* Build the forward input batch for the non-terminal rows. */
    int nrows = 0;
    for (int r = 0; r < nrep; r++) if (g_reply_gr[r] == 0) nrows++;
    for (int row = 0; row < nrows; row++)
        build_input(row, g_feats + (size_t)row * 196, opp_persp, opp_cube_action);
    if (nrows > 0) forward_batch(nrows, g_val);

    /* Reduce: per dice, mover takes the max equity over its replies. */
    double total = 0.0;
    int rep = 0, row = 0;
    for (int d = 0; d < 21; d++) {
        int cnt = g_counts[d];
        float best = -1e30f;
        for (int c = 0; c < cnt; c++, rep++) {
            int gr = g_reply_gr[rep];
            float eq;
            if (gr == 0)            eq = -g_val[row++];               /* -V(opp) */
            else if (gammons_count) eq = (float)gr;                  /* +1/2/3 */
            else                    eq = 1.0f;                       /* gammons off */
            if (eq > best) best = eq;
        }
        if (cnt > 0) total += (double)g_probs[d] * (double)best;
    }
    return (float)total;
}

/* Cubeless 1-ply value: same expansion, no cube features, gammons
 * count. Mirrors TDAgent.value_oneply_checker. */
float bgn_value_1ply_cubeless(const BoardState *s) {
    int nrep = get_legal_plays_encoded_21(s, g_feats, g_counts, g_reply_gr,
                                          BGN_MAX_ROWS, BGN_MAX_REPLIES);
    if (nrep < 0) return NAN;
    int nrows = 0;
    for (int r = 0; r < nrep; r++) if (g_reply_gr[r] == 0) nrows++;
    for (int row = 0; row < nrows; row++)
        build_input(row, g_feats + (size_t)row * 196, 0, 0);
    if (nrows > 0) forward_batch(nrows, g_val);

    double total = 0.0;
    int rep = 0, row = 0;
    for (int d = 0; d < 21; d++) {
        int cnt = g_counts[d];
        float best = -1e30f;
        for (int c = 0; c < cnt; c++, rep++) {
            int gr = g_reply_gr[rep];
            float eq = (gr == 0) ? -g_val[row++] : (float)gr;
            if (eq > best) best = eq;
        }
        if (cnt > 0) total += (double)g_probs[d] * (double)best;
    }
    return (float)total;
}

/* ── Generic cubeless 1-ply / 2-ply (DMP + prob5/money) ────────────── */

static inline float mover_eq_nonterm(float V, int flip_mode) {
    return (flip_mode == FLIP_PROB) ? (1.0f - V) : (-V);
}
static inline float mover_eq_term(int gr, int term_mode) {
    return (term_mode == TERM_WIN) ? 1.0f : (float)gr;
}

/* Cubeless 1-ply value. flip_mode/term_mode select the semantics:
 *   DMP         : FLIP_PROB   + TERM_WIN    (win probability)
 *   prob5/money : FLIP_EQUITY + TERM_GAMMON (signed money equity)
 * For prob5 the forward already reduces the 5 sigmoids to equity. */
float bgn_value_1ply_generic(const BoardState *s, int flip_mode, int term_mode) {
    int nrep = get_legal_plays_encoded_21(s, g_feats, g_counts, g_reply_gr,
                                          BGN_MAX_ROWS, BGN_MAX_REPLIES);
    if (nrep < 0) return NAN;
    int nrows = 0;
    for (int r = 0; r < nrep; r++) if (g_reply_gr[r] == 0) nrows++;
    for (int row = 0; row < nrows; row++)
        build_input(row, g_feats + (size_t)row * 196, 0, 0);
    if (nrows > 0) forward_batch(nrows, g_val);

    double total = 0.0;
    int rep = 0, row = 0;
    for (int d = 0; d < 21; d++) {
        int cnt = g_counts[d];
        float best = -1e30f;
        for (int c = 0; c < cnt; c++, rep++) {
            int gr = g_reply_gr[rep];
            float eq = (gr == 0) ? mover_eq_nonterm(g_val[row++], flip_mode)
                                 : mover_eq_term(gr, term_mode);
            if (eq > best) best = eq;
        }
        if (cnt > 0) total += (double)g_probs[d] * (double)best;
    }
    return (float)total;
}

/* Cubeless 2-ply value with a top-K 1-ply move filter. Same structure
 * as bgn_value_2ply_cubeful; with no cube, level-2 mover equity is just
 * flip(opp 1-ply value). */
float bgn_value_2ply_generic(const BoardState *s, int k,
                             int flip_mode, int term_mode) {
    if (k < 1) k = 1;
    if (k > BGN_TOPK_MAX) k = BGN_TOPK_MAX;
    float dice_term_best[21];
    int   dice_nrow[21];
    int   nrows = 0;
    int d = 0;
    for (int d1 = 1; d1 <= 6; d1++) {
        for (int d2 = d1; d2 <= 6; d2++, d++) {
            int n = get_legal_plays(s, d1, d2, g_plays, MAX_PLAYS);
            float tbest = -1e30f;
            int nr = 0;
            if (n == 0) {
                if (nrows >= BGN_L1_MAX) return NAN;
                BoardState opp_view = *s; board_switch_turn(&opp_view);
                float base[196]; encode_state(&opp_view, base);
                build_input(nrows, base, 0, 0);
                g_l1_states[nrows] = opp_view; g_l1_dice[nrows] = d;
                nr = 1; nrows++;
                dice_term_best[d] = -1e30f; dice_nrow[d] = nr; continue;
            }
            for (int i = 0; i < n; i++) {
                const BoardState *rs = &g_plays[i].resulting_state;
                if (board_is_game_over(rs)) {
                    float e = mover_eq_term(gr_c(rs), term_mode);
                    if (e > tbest) tbest = e;
                } else {
                    if (nrows >= BGN_L1_MAX) return NAN;
                    float base[196]; encode_state(rs, base);
                    build_input(nrows, base, 0, 0);
                    g_l1_states[nrows] = *rs; g_l1_dice[nrows] = d;
                    nr++; nrows++;
                }
            }
            dice_term_best[d] = (tbest > -1e29f) ? tbest : -1e30f;
            dice_nrow[d] = nr;
        }
    }
    if (nrows > 0) forward_batch(nrows, g_val);
    for (int r = 0; r < nrows; r++)
        g_l1_eq[r] = mover_eq_nonterm(g_val[r], flip_mode);  /* 0-ply rank key */

    double total = 0.0;
    int base_row = 0;
    for (d = 0; d < 21; d++) {
        int nr = dice_nrow[d];
        float best = dice_term_best[d];
        int idx[BGN_TOPK_MAX];
        int kk = (nr < k) ? nr : k;
        for (int a = 0; a < kk; a++) {
            int bestj = -1; float bestv = -1e30f;
            for (int r = 0; r < nr; r++) {
                int already = 0;
                for (int b = 0; b < a; b++) if (idx[b] == r) { already = 1; break; }
                if (already) continue;
                if (g_l1_eq[base_row + r] > bestv) { bestv = g_l1_eq[base_row + r]; bestj = r; }
            }
            idx[a] = bestj;
        }
        for (int a = 0; a < kk; a++) {
            BoardState succ = g_l1_states[base_row + idx[a]];
            float v_opp = bgn_value_1ply_generic(&succ, flip_mode, term_mode);
            float eq = mover_eq_nonterm(v_opp, flip_mode);
            if (eq > best) best = eq;
        }
        base_row += nr;
        if (nr > 0 || dice_term_best[d] > -1e29f)
            total += (double)g_probs[d] * (double)best;
    }
    return (float)total;
}

/* ── 0-ply best move (cubeful) ─────────────────────────────────────── */

/* Best move for (s, d1, d2) by 0-ply equity: terminal game_result or
 * -V(opp successor). Writes the resulting state to `out` and its mover
 * equity to out_val. Returns the play index, or -1 on a forced pass. */
int bgn_best_move_0ply_cubeful(const BoardState *s, int d1, int d2,
                               int succ_persp, int succ_cube_action,
                               BoardState *out, float *out_val) {
    int n = get_legal_plays(s, d1, d2, g_plays, MAX_PLAYS);
    if (n <= 0) return -1;

    /* Non-terminal successors go into the forward batch. Terminals are
     * scored directly. */
    int nrows = 0;
    static int row_of[MAX_PLAYS];   /* play idx -> forward row, or -1 terminal */
    float term_eq[MAX_PLAYS];
    for (int i = 0; i < n; i++) {
        const BoardState *rs = &g_plays[i].resulting_state;
        if (board_is_game_over(rs)) {
            row_of[i] = -1;
            term_eq[i] = (float)gr_c(rs);    /* mover wins by bearing off */
        } else {
            float base[196];
            encode_state(rs, base);
            build_input(nrows, base, succ_persp, succ_cube_action);
            row_of[i] = nrows++;
        }
    }
    if (nrows > 0) forward_batch(nrows, g_val);

    int best_i = 0; float best_eq = -1e30f;
    for (int i = 0; i < n; i++) {
        float eq = (row_of[i] < 0) ? term_eq[i] : -g_val[row_of[i]];
        if (eq > best_eq) { best_eq = eq; best_i = i; }
    }
    *out = g_plays[best_i].resulting_state;
    if (out_val) *out_val = best_eq;
    return best_i;
}

/* ── 2-ply value (cubeful money, top-K filtered) ───────────────────── */

/* Mirrors TDAgent.value_twoply_checker_cubeful_filtered. Level 1: rank
 * all dice candidates by 0-ply in one forward, keep the top-K
 * non-terminal per dice. Level 2: 1-ply leaf on each kept successor with
 * the opponent's cube decision; mover_eq = -v_opp.
 *
 * Cube params come from Python (the same for every successor, since the
 * opponent is on roll in all of them):
 *   succ_persp/succ_cube_action/gammons : opp cube perspective (as 1-ply)
 *   nd_*  : leaf params for opp's no-double 1-ply value
 *   at_*  : leaf params for opp's after-take 1-ply value
 *   at_applicable : whether the opponent may double at the successor
 * Returns NAN on expansion overflow (the caller falls back to Python). */
float bgn_value_2ply_cubeful(const BoardState *s, int k,
                             int succ_persp, int succ_cube_action, int gammons_count,
                             int nd_persp, int nd_cube_action, int nd_gammons,
                             int at_applicable, int at_persp, int at_cube_action,
                             int at_gammons) {
    if (k < 1) k = 1;
    if (k > BGN_TOPK_MAX) k = BGN_TOPK_MAX;

    /* Per dice: best terminal mover equity, and non-terminal count. */
    float dice_term_best[21];
    int   dice_nrow[21];
    int   nrows = 0;

    /* ── Level-1: gather candidates across 21 dice into one forward ── */
    int d = 0;
    for (int d1 = 1; d1 <= 6; d1++) {
        for (int d2 = d1; d2 <= 6; d2++, d++) {
            int n = get_legal_plays(s, d1, d2, g_plays, MAX_PLAYS);
            float tbest = -1e30f;
            int nr = 0;
            if (n == 0) {
                /* Forced pass: one non-terminal candidate, the same
                 * board with the opponent on roll. */
                if (nrows >= BGN_L1_MAX) return NAN;
                BoardState opp_view = *s;
                board_switch_turn(&opp_view);
                float base[196];
                encode_state(&opp_view, base);
                build_input(nrows, base, succ_persp, succ_cube_action);
                g_l1_states[nrows] = opp_view;
                g_l1_dice[nrows] = d;
                nr = 1;
                nrows++;
                dice_term_best[d] = -1e30f;
                dice_nrow[d] = nr;
                continue;
            }
            for (int i = 0; i < n; i++) {
                const BoardState *rs = &g_plays[i].resulting_state;
                if (board_is_game_over(rs)) {
                    float e = gammons_count ? (float)gr_c(rs) : 1.0f;
                    if (e > tbest) tbest = e;
                } else {
                    if (nrows >= BGN_L1_MAX) return NAN;
                    float base[196];
                    encode_state(rs, base);
                    build_input(nrows, base, succ_persp, succ_cube_action);
                    g_l1_states[nrows] = *rs;
                    g_l1_dice[nrows] = d;
                    nr++;
                    nrows++;
                }
            }
            dice_term_best[d] = (tbest > -1e29f) ? tbest : -1e30f;
            dice_nrow[d] = nr;
        }
    }
    if (nrows > 0) forward_batch(nrows, g_val);   /* level-1 0-ply values */
    for (int r = 0; r < nrows; r++)
        g_l1_eq[r] = -g_val[r];                   /* mover equity = -V(opp) */

    /* ── Level-2: per dice, deepen the top-K non-terminal candidates ──
     * Everything needed is now in g_l1_*, so the recursive 1-ply leaf
     * may reuse g_in/g_buf*. */
    double total = 0.0;
    int base_row = 0;
    for (d = 0; d < 21; d++) {
        int nr = dice_nrow[d];
        float best = dice_term_best[d];

        /* selection sort the top-K by level-1 equity within this dice */
        int idx[BGN_TOPK_MAX];
        int kk = (nr < k) ? nr : k;
        for (int a = 0; a < kk; a++) {
            int bestj = -1; float bestv = -1e30f;
            for (int r = 0; r < nr; r++) {
                int already = 0;
                for (int b = 0; b < a; b++) if (idx[b] == r) { already = 1; break; }
                if (already) continue;
                if (g_l1_eq[base_row + r] > bestv) { bestv = g_l1_eq[base_row + r]; bestj = r; }
            }
            idx[a] = bestj;
        }

        for (int a = 0; a < kk; a++) {
            BoardState succ = g_l1_states[base_row + idx[a]];
            float v_nd = bgn_value_1ply_cubeful(&succ, nd_persp, nd_cube_action, nd_gammons);
            float v_opp;
            if (at_applicable) {
                float v_at = bgn_value_1ply_cubeful(&succ, at_persp, at_cube_action, at_gammons);
                float v_dbl = 2.0f * v_at; if (v_dbl > 1.0f) v_dbl = 1.0f;
                v_opp = (v_nd > v_dbl) ? v_nd : v_dbl;
            } else {
                v_opp = v_nd;
            }
            float eq = -v_opp;
            if (eq > best) best = eq;
        }

        base_row += nr;
        if (nr > 0 || dice_term_best[d] > -1e29f)
            total += (double)g_probs[d] * (double)best;
    }
    return (float)total;
}
