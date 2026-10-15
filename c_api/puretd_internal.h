/*
 * puretd_internal.h -- helpers shared inside libpuretd_<b>.so by puretd.c
 * and puretd_collect.c. Not part of the API (that is puretd.h); the
 * functions are hidden in the shared library.
 */

#ifndef PURETD_INTERNAL_H
#define PURETD_INTERNAL_H

#include "puretd.h"

#include <math.h>
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

/* The match block a name prefix appends (encoding.Matchplay1H64METEncoder
   and its MET6 subclass): 141 columns (cube perspective 0-2,
   is_cube_action 3, cube value one-hot 4-10, my_away one-hot 11-74,
   opp_away 75-138, is_crawford 139, post_crawford 140) then the MET
   block (met.outcome_block / outcome_block_x6, each minus 0.5f), from the
   row player's view. Written by puretd_match.c's match_extras. */
typedef struct {
    const char *prefix;      /* "matchplay1h64met_", "matchplay1h64met6_" */
    int nextra;              /* columns after the base: 141 + nmet */
    int nmet;                /* 13 (outcome_block) or 17 (outcome_block_x6) */
    int collect;             /* bgc_collect_match is parity-tested on its rows */
} bgc_match_block;

/* A resolved encoder name. */
typedef struct {
    const char *name;        /* as resolved (the engine's copy) */
    const bgc_base_enc *base;
    const bgc_cube_block *cube;   /* a match encoder: the "" block (no cube block) */
    const bgc_match_block *match; /* NULL: a money encoder */
    int n_in;                /* base->n + cube->ncube (+ match->nextra): the net's input_size */
    int collect;             /* base->collect (&& match->collect) */
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

/* ── the Janowski conversion's per-context part (puretd_janowski.c) ─
 * janowski_match.cubeful_mwc's MET lookups depend on the context only
 * (scores, Crawford flags, cube), never on the probabilities: bgc_i_jan_prep
 * does them once per context, bgc_i_jan_eval the per-row arithmetic (the
 * same doubles in the same order as Python's per-call lookups). */
#define BGC_JAN_LEVELS 32            /* get_points cube levels (log2 of an away, + 1) */
typedef struct {
    double t0, t1, t2;   /* top level: DTL's lose MWCs at 2pv, 4pv, 6pv; alive: t0 = RDP */
    double dp;           /* win cv (the cash) */
    double u0, u1, u2;   /* DTW's win MWCs: top at 2pv, 4pv, 6pv; alive at 2cv, 4cv, 6cv */
    int top;             /* the doubled cube would be dead (the ladder's top) */
} bgc_jan_level;
typedef struct {
    int cubeful;         /* do_cubeful(ctx) */
    int n_max;           /* get_points' ladder: levels 0..n_max */
    double w[3], l[3];   /* _me(ctx, k * cube, win / lose), k = 1, 2, 3 */
    double dw1, dw2, dl1, dl2;          /* w2 - w1, w3 - w2, l2 - l1, l3 - l2 (dead_mwc) */
    bgc_jan_level lv[2][BGC_JAN_LEVELS];   /* [side k: 0 me, 1 opp (k's view)][level n] */
} bgc_jan_prep;
/* ── the engine ───────────────────────────────────────────────────
 * Defined here so that puretd_match.c (the MATCH kind) works on the same
 * batch and scratch as puretd.c; nothing outside libpuretd sees it. */
struct bgc_backend;
struct bgc_mmemo { uint64_t k[5]; double v; };   /* puretd_match.c's memo entry; k[4] 0 = empty */
struct bgc_engine {
    struct bgc_backend *be;
    int kind, in_size, out_dim, ldx;
    bgc_encoder enc;    /* the export's encoder (bgc_open); enc.name -> enc_name */
    int nb;             /* enc.base->n: base features, the cube block's offset */
    int format_version; /* the export's BGNN header format (1: legacy, by width) */
    char enc_name[64];  /* NN_ENCODER_NAME_MAX */
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
    /* match play (BGC_KIND_MATCH, puretd_match.c) */
    double *met_pre, *met_post;  /* bgc_set_met: n x n row-major, n; NULL until set */
    int met_n;
    bgc_met met;                 /* {met_pre, met_post, met_n} (bgc_met_mwc's view) */
    float *tail;        /* MATCH: ldx - nb floats that encode_base and forward_succ put
                           after a row's base columns: the current context's match block,
                           then the zero pad (other kinds: unused, zeros are written) */
    float mtv[4];       /* MATCH: opp-view value of a terminal successor by game_result
                           (1..3) for the current mover: float32(1 - mover_val); NAN
                           where the MET cannot price it */
    float *mval;        /* MATCH: candidate opp-view values of the last select, MAX_PLAYS */
    unsigned char *mview; /* bgc_collect_match: the episode's row viewers (grows) */
    long cap_mview;
    /* MATCH 1-ply memo (puretd_match.c): (board, match context) -> the 1-ply
       value, open addressing; on unless BGC_MATCH_MEMO=0 at open */
    struct bgc_mmemo *mm;
    long mm_cap, mm_n;
    int mm_on;
    long mm_lookups, mm_hits;
    /* bgc_i_legal_plays's leaf table (puretd_movegen.c); fastgen: the engine's
       move generation goes through it (BGC_FAST_MOVEGEN, read at open) */
    struct bgc_mgtab *mg;
    int fastgen;
    /* MATCH: the 1-ply memo shared by the processes of one round
       (puretd_match.c ms_*, BGC_MATCH_SHM): a POSIX shm table named by the
       digest of (weights file, library file, backend, MET); NULL detached */
    int ms_want;                 /* BGC_MATCH_SHM at open (and the local memo on) */
    int ms_minpip;               /* BGC_MATCH_SHM_PIPS: positions shared */
    uint64_t ms_wdig[2];         /* digest of the weights file (open) */
    void *ms_map;
    size_t ms_len;
    char ms_name[64];
    long ms_lookups, ms_hits, ms_puts;
    /* MATCH_JANOWSKI (p5jan; puretd_janowski.c, the seam in puretd_match.c) */
    double *vd;                  /* per row: cubeful MWC of the row's player in the
                                    current context (bgc_i_jan_rows), cap_rows */
    double *rx;                  /* per row: its x (encode_base, from the row's board) */
    double mtvd[4];              /* terminal successor by game_result (1..3): the MOVER's
                                    MWC, met.mwc_after(ctx_me, cube * gr, win) */
    bgc_jan_prep jprep;          /* the current context's lookups (bgc_i_jan_set) */
    int jpersp;                  /* the current context's perspective (the row player's) */
    int jx_const;                /* bgc_set_jan_x: 1 = jx for every row, 0 = cube_x */
    double jx;
};

#define BGC_DD_SLOTS 4096            /* dedup table; at most BGC_DD_SLOTS / 2 rows per expansion */

/* ── puretd_movegen.c ─────────────────────────────────────────────
 * bgc_i_legal_plays = get_legal_plays (c_engine/bg_engine.c), the same
 * walk and the same play list (boards, order, num_moves, moves), with the
 * leaf dedup in table t (bgc_i_mgtab_new; never cleared: a generation per
 * call) instead of a 520 KB stack hash set cleared per call. max_plays
 * other than MAX_PLAYS (or t NULL): get_legal_plays itself. */
#define BGC_MG_SLOTS 8192            /* >= 2 * MAX_PLAYS: load <= 1/2 */
struct bgc_mgtab;
BGC_HIDDEN struct bgc_mgtab *bgc_i_mgtab_new(void);
BGC_HIDDEN int bgc_i_legal_plays(struct bgc_mgtab *t, const BoardState *state, int d1, int d2,
                                 Play *plays, int max_plays);

/* puretd_match.c: the shared memo's open / close hooks (bgc_open_ex for a
   MATCH engine: the knobs and the weights file digest; bgc_close). */
BGC_HIDDEN void bgc_i_match_opened(bgc_engine *e, const char *path);
BGC_HIDDEN void bgc_i_match_shm_detach(bgc_engine *e);
/* puretd_match.c: the memos' values are stale (bgc_set_met, bgc_set_jan_x):
   the local memo cleared, the shared one re-attached (its digest names the
   MET and, for MATCH_JANOWSKI, the x policy). */
BGC_HIDDEN void bgc_i_match_memo_reset(bgc_engine *e);
/* A match kind: BGC_KIND_MATCH or BGC_KIND_MATCH_JANOWSKI (the bgc_match_* calls). */
static inline int bgc_i_is_match(const bgc_engine *e) {
    return e->kind == BGC_KIND_MATCH || e->kind == BGC_KIND_MATCH_JANOWSKI;
}

/* The engine's move generation: bgc_i_legal_plays when e->fastgen, else
   get_legal_plays; into out[MAX_PLAYS]. */
static inline int bgc_i_plays(bgc_engine *e, const BoardState *s, int d1, int d2, Play *out) {
    return e->fastgen ? bgc_i_legal_plays(e->mg, s, d1, d2, out, MAX_PLAYS)
                      : get_legal_plays(s, d1, d2, out, MAX_PLAYS);
}

/* _ONEPLY_PROBS: dice d1=1..6, d2=d1..6; doubles 1/36, others 2/36 (puretd.c). */
BGC_HIDDEN extern const double bgc_dice_p[21];

enum { BGC_FLIP_NONE = 0, BGC_FLIP_NEG = 1, BGC_FLIP_ONE_MINUS = 2 };

/* puretd.c's batch helpers, unchanged, for puretd_match.c (each is the
   static function of the same name there):
     ensure_rows    the batch holds n rows (0, -1 out of memory)
     encode_base    row r = the base encoding of s, then e->tail (MATCH) or zeros
     expand21       the 21-roll replies of X into the batch from *nrow / *nrep
                    (slot's cdoff / cpass; the plays of dice index keep into
                    e->plays); 0, -1 out of memory
     dice_index     expand21's index of the roll {a, b}
     dice_sum       sum over the 21 rolls of p * flip(mins[k]) in dice order
                    (grouped: ProbAgent's forced-pass grouping)
     forward        rows [r0, r0 + n), plain
     forward_succ   rows [r0, r0 + n) that are successors of X (delta forward
                    from X with the turn switched, its columns after the base
                    those of the rows: cube_cols(persp, ica) / e->tail)
     forward_v1     forward_succ for value1's rows (bf16 in a -DBGC_BF16 build)
     roll_replies   value1_play's matching: after expand21(.., keep =
                    dice_index(d1, d2)), the candidates of get_legal_plays(X,
                    d1, d2) as replies of that roll (e->pidx[i] = reply index,
                    e->plays in that order); n, -1 forced pass, -3 no match */
BGC_HIDDEN int  bgc_i_ensure_rows(bgc_engine *e, int n);
BGC_HIDDEN void bgc_i_encode_base(bgc_engine *e, int r, const BoardState *s);
BGC_HIDDEN int  bgc_i_expand21(bgc_engine *e, const BoardState *X, int slot, int keep,
                               int *nrow, int *nrep);
BGC_HIDDEN int  bgc_i_dice_index(int a, int b);
BGC_HIDDEN double bgc_i_dice_sum(const double mins[21], const unsigned char *pass,
                                 int grouped, int how);
BGC_HIDDEN void bgc_i_forward(bgc_engine *e, int r0, int n);
BGC_HIDDEN void bgc_i_forward_succ(bgc_engine *e, int r0, int n, const BoardState *X,
                                   int persp, int ica);
BGC_HIDDEN void bgc_i_forward_v1(bgc_engine *e, int r0, int n, const BoardState *X,
                                 int persp, int ica);
BGC_HIDDEN int  bgc_i_roll_replies(bgc_engine *e, const BoardState *X, int d1, int d2);

/* ── met.MatchEquityTable lookups on a bgc_met (puretd_match.c,
 * puretd_janowski.c) ─────────────────────────────────────────── */

/* MatchEquityTable.mwc: P(my wins the match) at the start of a game. NAN
   where Python raises (_check_range: an away above the table). */
static inline double bgc_met_mwc(const bgc_met *t, int my, int opp, int cd, int ic) {
    if (my <= 0) return 1.0;
    if (opp <= 0) return 0.0;
    if (my > t->n || opp > t->n) return NAN;
    if (cd && !ic && (my == 1 || opp == 1)) {       /* post-Crawford */
        if (opp == 1) return t->post[my - 1];
        return 1.0 - t->post[opp - 1];
    }
    return t->pre[(size_t)(my - 1) * t->n + (opp - 1)];
}

/* MatchEquityTable.mwc_after: MWC at the start of the next game after
   someone scores `pts` (me: the "my" player wins them). */
static inline double bgc_met_mwc_after(const bgc_met *t, int my, int opp, int cd, int ic,
                                       long long pts, int me) {
    long long nm = my, no = opp;
    if (me) nm -= pts;
    else no -= pts;
    if (nm <= 0) return 1.0;
    if (no <= 0) return 0.0;
    const int was = cd || ic;
    const int nc = !was && (nm == 1 || no == 1);
    return bgc_met_mwc(t, (int)nm, (int)no, was || nc, nc);
}

/* ── the Janowski conversion (puretd_janowski.c; its types above the
 * engine) ────────────────────────────────────────────────────────── */

/* The context's lookups into *P. 0, or -1 for a context the ladder cannot
   hold (an away above 2^31; Python has no such limit). A lookup the MET
   cannot price is NAN and propagates to the values that use it. */
BGC_HIDDEN int bgc_i_jan_prep(bgc_jan_prep *P, const bgc_met *t, const bgc_jan_ctx *c);
/* cubeful_mwc(probs, persp, ctx, x) for the context of P (probs: the nested
   5 from the evaluating player's view, as doubles). */
BGC_HIDDEN double bgc_i_jan_eval(const bgc_jan_prep *P, const double probs[5], int persp,
                                 double x);

/* ── modes.MatchState on a bgc_match (WHITE's view; puretd_match.c,
 * puretd_match_collect.c) ─────────────────────────────────────── */

/* A bgc_match the match calls accept: a match (length >= 1), owner 0..2,
   cube >= 1. Aways are checked where Python checks them (encoding, MET). */
static inline int bgc_m_ok(const bgc_match *m) {
    return m && m->length >= 1 && m->cube_owner >= 0 && m->cube_owner <= 2
           && m->cube_value >= 1 && m->cube_value <= (1 << 24);
}

/* MatchState.can_offer (matchplay): not in the Crawford game, not with a
   dead cube (cube_value >= the doubler's away), centered or owned. */
static inline int bgc_m_can_offer(const bgc_match *m, int player) {
    if (m->is_crawford) return 0;
    if (m->cube_value >= m->away[player]) return 0;
    return cube_persp(m->cube_owner, player) != PERSP_THEIRS;
}

/* MatchState.after_take(doubler): the opponent owns the doubled cube. */
static inline bgc_match bgc_m_after_take(const bgc_match *m, int doubler) {
    bgc_match r = *m;
    r.cube_owner = (1 - doubler) + 1;
    r.cube_value = m->cube_value * 2;
    return r;
}

/* MatchState.after_game(game_result, winner_is_me = (winner == WHITE)):
   cube_value * game_result points, away floored at 0, cube reset,
   is_crawford cleared, crawford_done kept. */
static inline bgc_match bgc_m_after_game(const bgc_match *m, int gr, int winner) {
    bgc_match r = *m;
    const long long a = (long long)m->away[winner] - (long long)m->cube_value * gr;
    r.away[winner] = a > 0 ? (int)a : 0;
    r.cube_owner = 0;
    r.cube_value = 1;
    r.is_crawford = 0;
    return r;
}

/* MatchState.start_game: cube centered at 1; the Crawford game is the
   first game with someone 1-away (crawford_done latches it). */
static inline bgc_match bgc_m_start_game(const bgc_match *m) {
    bgc_match r = *m;
    const int c = !m->crawford_done && (m->away[0] == 1 || m->away[1] == 1);
    r.cube_owner = 0;
    r.cube_value = 1;
    r.is_crawford = c;
    r.crawford_done = m->crawford_done || c;
    return r;
}

static inline int bgc_m_over(const bgc_match *m) { return m->away[0] <= 0 || m->away[1] <= 0; }

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

/* ── MATCH_JANOWSKI rows (p5jan; puretd_janowski.c) ───────────────
 * The row value of the Janowski evaluator: after a forward of rows [r0, r0
 * + n), bgc_i_jan_rows writes e->vd[r] = cubeful_mwc(model.prob5_postprocess
 * of the row's outputs, e->jpersp, the context set by bgc_i_jan_set,
 * e->rx[r]). encode_base sets e->rx[r] from the row's board. */
BGC_HIDDEN void bgc_i_prob5_clamp(const float in[5], float out[5]);   /* puretd.c prob5_post */
/* The context of the rows that follow (ctx: the row player's view; persp:
   theirs). 0, or -1 for a context Python cannot price (an away below 1 or
   above the MET). */
BGC_HIDDEN int bgc_i_jan_set(bgc_engine *e, const bgc_jan_ctx *c, int persp);
/* 0, or -1 on a NaN value (a NaN net output). */
BGC_HIDDEN int bgc_i_jan_rows(bgc_engine *e, int r0, int n);
/* A row's x: the constant (bgc_set_jan_x) or janowski.cube_x of its board. */
static inline double bgc_i_jan_row_x(const bgc_engine *e, const BoardState *s) {
    return e->jx_const ? e->jx : bgc_jan_cube_x(s);
}

/* ── profile build ────────────────────────────────────────────────
 * CFLAGS_EXTRA=-DBGC_PROFILE: CLOCK_MONOTONIC counters per (context,
 * step), read with bgc_prof_read / cleared with bgc_prof_reset (exported
 * only in that build). One process-
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
