/*
 * puretd_internal.h -- helpers shared inside libpuretd_<b>.so by puretd.c
 * and puretd_collect.c. Not part of the API (that is puretd.h); the
 * functions are hidden in the shared library.
 */

#ifndef PURETD_INTERNAL_H
#define PURETD_INTERNAL_H

#include "puretd.h"

#include <stddef.h>

#if defined(__GNUC__) || defined(__clang__)
#define BGC_HIDDEN __attribute__((visibility("hidden")))
#else
#define BGC_HIDDEN
#endif

#define BGC_NB 196                   /* perspective196's base features */

/* ── encoders ─────────────────────────────────────────────────────
 * What a net's input row is comes from the encoder name its export
 * carries (export_weights.py format 2; encoding.get_encoder names),
 * resolved by bgc_i_encoder_resolve (puretd.c) as get_encoder resolves
 * it: the name of a base encoder (bgc_base_enc), or a cube block's
 * prefix (bgc_cube_block) + a base encoder's name. A row is then
 * [base->n base features | cube->ncube cube columns]; bgc_open asserts
 * the net's input_size equals that, and the engine kind follows from
 * the cube block (cubeful) and the output mode. Both tables are in
 * puretd.c (BGC_BASES, BGC_CUBE_BLOCKS; how to add a base encoder: the
 * comment above them). Nothing in the engine or the collector derives
 * the row layout from the input width. */

/* encoding.ENCODERS: one row per base encoder. */
typedef void (*bgc_encode_fn)(const BoardState *s, float *out);
typedef struct {
    const char *name;        /* encoding.ENCODERS key */
    int n;                   /* num_features */
    bgc_encode_fn encode;    /* writes the n features of s (Encoder.encode, bit-exact) */
    int collect;             /* bgc_collect is parity-tested on its rows */
} bgc_base_enc;

/* The cube block a name prefix appends (encoding.get_encoder's wrappers):
   cube_cols writes it, the 3-hot cube perspective then is_cube_action.
   ncube 0: no block (the cubeless kinds). */
typedef struct {
    const char *prefix;      /* "cubeful_", "" */
    int ncube;               /* columns after the base (Encoder.cube_features) */
} bgc_cube_block;

/* A resolved encoder name. */
typedef struct {
    const char *name;        /* as resolved (the engine's copy) */
    const bgc_base_enc *base;
    const bgc_cube_block *cube;
    int n_in;                /* base->n + cube->ncube: the net's input_size */
    int collect;             /* base->collect */
} bgc_encoder;

/* Resolve `name` into *out (out->name = name, not copied). 0, or -1 with
   a message naming the encoder in err[errlen] (err nullable). */
BGC_HIDDEN int bgc_i_encoder_resolve(const char *name, bgc_encoder *out,
                                     char *err, size_t errlen);

/* The engine's encoder (resolved by bgc_open). */
BGC_HIDDEN const bgc_encoder *bgc_i_encoder(const bgc_engine *e);

/* encoding.CubePerspective = the cube columns' one-hot index. */
#define PERSP_CENTERED 0
#define PERSP_MINE 1
#define PERSP_THEIRS 2

/* modes.cube_perspective (owner: 0 centered, 1 WHITE, 2 BLACK). */
static inline int cube_persp(int owner, int player) {
    if (owner == 0) return PERSP_CENTERED;
    return owner == player + 1 ? PERSP_MINE : PERSP_THEIRS;
}

/* MatchState.can_offer */
static inline int can_offer(int owner, int player) {
    return cube_persp(owner, player) != PERSP_THEIRS;
}

/* Jacoby: gammons count once the cube is turned. */
static inline int gammons_count(const bgc_cube *c) {
    return !c->jacoby || c->owner != 0;
}

/* The cube block of a cubeful row (c = row + the base's n):
   CubefulEncoder.encode_with_base (one-hot perspective, is_cube_action). */
static inline void cube_cols(float *c, int persp, int ica) {
    c[0] = c[1] = c[2] = 0.0f;
    c[persp] = 1.0f;
    c[3] = ica ? 1.0f : 0.0f;
}

/* BoardState.game_result: 1 single, 2 gammon, 3 backgammon; 0 if not over. */
BGC_HIDDEN int bgc_i_game_result(const BoardState *s);

/* backgammon_engine.opening_roll draw-for-draw: (white, black) dice pairs
   until they differ; s = initial board, the higher die's player on roll,
   d1 = high, d2 = low. *n_dice += the dice drawn. */
BGC_HIDDEN void bgc_i_opening_roll(bgc_rng *rng, BoardState *s, int *d1, int *d2,
                                   int *n_dice);

/* bgc_value_1ply_play with value1's double (bgc_value_1ply rounds it to
   float): the 1-ply value of non-terminal `s` for the player on roll and
   the 0-ply play for (d1, d2), as bgc_best_move(plies 0) returns it.
   vec5 (nullable; non-NULL needs a PROB5 engine, else -2): the 1-ply
   target vector of train_prob5._oneply_target_vec_expand21 from the same
   expansion (puretd.c vec1_prob5); -2 also on a NaN output. */
BGC_HIDDEN int bgc_i_value1_play(bgc_engine *e, const BoardState *s, const bgc_cube *cube,
                                 int d1, int d2, double *v1, float *vec5,
                                 BoardState *out_state, float *out_value);

/* PROB5: bgc_best_move(s, d1, d2, NULL, plies 0) and the chosen
   candidate's 5 clamped probabilities (opponent's view: the successor's
   player on roll) from select0's forward. Returns as bgc_best_move (-2
   for another kind); out_state / probs are set only for an index >= 0. */
BGC_HIDDEN int bgc_i_best_move_prob5(bgc_engine *e, const BoardState *s, int d1, int d2,
                                     BoardState *out_state, float probs[5]);

/* ── Prob5Jan (puretd_janowski.c) ─────────────────────────────────
 * Engine internals the Janowski code drives; the batch pointers move on
 * any ensure_rows / expand21 (realloc): fetch the view again after one. */
typedef struct {
    float *x, *y;            /* rows ldx apart; raw outputs out_dim apart */
    int ldx, out_dim;
    Play *plays, *iplays;    /* root candidates; expand21's scratch */
    int *cdoff;              /* per slot 22 reply offsets (dice-major) */
    unsigned char *cpass;    /* per slot 21 forced-pass flags */
    int *rep_row, *rep_gr;   /* reply -> row (-1 terminal), game_result */
    double *cscore;          /* per candidate, mover's view */
} bgc_batch;
BGC_HIDDEN void bgc_i_batch(bgc_engine *e, bgc_batch *b);
BGC_HIDDEN int bgc_i_expand21(bgc_engine *e, const BoardState *X, int slot, int keep,
                              int *nrow, int *nrep);
BGC_HIDDEN void bgc_i_encode_base(bgc_engine *e, int r, const BoardState *s);
BGC_HIDDEN void bgc_i_forward(bgc_engine *e, int r0, int n);
BGC_HIDDEN void bgc_i_forward_succ(bgc_engine *e, int r0, int n, const BoardState *X,
                                   int persp, int ica);
BGC_HIDDEN int bgc_i_ensure_rows(bgc_engine *e, int n);
/* model.prob5_postprocess of in into out (fp32); returns their equity. */
BGC_HIDDEN float bgc_i_prob5_clamp(const float in[5], float out[5]);

struct bgc_jan;
BGC_HIDDEN struct bgc_jan *bgc_i_jan(bgc_engine *e);
BGC_HIDDEN struct bgc_jan *bgc_i_jan_new(void);
BGC_HIDDEN void bgc_i_jan_free(struct bgc_jan *j);
/* The PROB5JAN entry points behind bgc_value, bgc_value_prob5, value1k
   (bgc_value_1ply), choose (n candidates in the batch's plays; fills
   cscore), bgc_offer_double and bgc_respond_to_double. */
BGC_HIDDEN float bgc_i_jan_value(bgc_engine *e, const BoardState *s, const bgc_cube *cube);
BGC_HIDDEN int bgc_i_jan_value_prob5(bgc_engine *e, const BoardState *s, float probs[5],
                                     float *eq);
BGC_HIDDEN double bgc_i_jan_value1(bgc_engine *e, const BoardState *s, const bgc_cube *cube,
                                   int *err);
BGC_HIDDEN int bgc_i_jan_choose(bgc_engine *e, const BoardState *s, int n,
                                const bgc_cube *cube, int plies);
BGC_HIDDEN int bgc_i_jan_offer(bgc_engine *e, const BoardState *s, const bgc_cube *cube,
                               int plies, bgc_cube_eval *out);
BGC_HIDDEN int bgc_i_jan_respond(bgc_engine *e, const BoardState *s, const bgc_cube *cube,
                                 int plies, const bgc_cube_eval *hint);

/* ── profile build ────────────────────────────────────────────────
 * CFLAGS_EXTRA=-DBGC_PROFILE: CLOCK_MONOTONIC counters per (context,
 * step), read with bgc_prof_read / cleared with bgc_prof_reset (exported
 * only in that build; experiments/c_collect/p5_profile.py). One process-
 * wide table: profile one engine on one thread. Off by default (the
 * macros compile to nothing). */
enum { BGC_PC_V1,          /* value1: the 1-ply target            */
       BGC_PC_SEL,         /* choose/select0: the 0-ply play      */
       BGC_PC_CUBE,        /* bgc_offer_double                    */
       BGC_PC_VAL,         /* bgc_value (forced-pass bootstrap)   */
       BGC_PC_COLLECT,     /* puretd_collect.c's own work         */
       BGC_PC_N };
enum { BGC_PT_TOTAL,       /* the whole context call              */
       BGC_PT_MOVEGEN,     /* get_legal_plays                     */
       BGC_PT_ENCODE,      /* base encode + pad + reply records   */
       BGC_PT_CUBECOLS,    /* set_cube_cols                       */
       BGC_PT_FORWARD,     /* backend forward + value copy        */
       BGC_PT_REDUCE,      /* dice_mins/dice_sum, argmin          */
       BGC_PT_N };
#ifdef BGC_PROFILE
#include <time.h>
typedef struct {
    double t[BGC_PC_N][BGC_PT_N];
    long calls[BGC_PC_N][BGC_PT_N];
    long rows[BGC_PC_N][BGC_PT_N];
} bgc_prof_table;
BGC_HIDDEN extern bgc_prof_table bgc_prof;
BGC_HIDDEN extern int bgc_prof_ctx;
static inline double bgc_prof_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
#define PROF_T0(v) double v = bgc_prof_now()
#define PROF_ADD(v, step, nrows) do {                                       \
        bgc_prof.t[bgc_prof_ctx][step] += bgc_prof_now() - (v);             \
        bgc_prof.calls[bgc_prof_ctx][step]++;                               \
        bgc_prof.rows[bgc_prof_ctx][step] += (nrows);                       \
    } while (0)
/* Enter context c for the rest of the scope's PROF_* (restored by PROF_LEAVE). */
#define PROF_ENTER(c) int prof_saved_ctx_ = bgc_prof_ctx; bgc_prof_ctx = (c); PROF_T0(prof_tc_)
#define PROF_LEAVE(nrows) do { PROF_ADD(prof_tc_, BGC_PT_TOTAL, nrows);      \
        bgc_prof_ctx = prof_saved_ctx_; } while (0)
#else
#define PROF_T0(v) do { } while (0)
#define PROF_ADD(v, step, nrows) do { } while (0)
#define PROF_ENTER(c) do { } while (0)
#define PROF_LEAVE(nrows) do { } while (0)
#endif

#endif /* PURETD_INTERNAL_H */
