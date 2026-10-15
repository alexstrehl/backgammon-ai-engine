/*
 * puretd.h -- PureTD C API: move selection, cube decisions, 1-ply search
 * and self-play for exported PureTD nets (BGNN .bin, export_weights.py).
 *
 * Conventions (those of the repo):
 *   - state->turn is the player to act. get_legal_plays returns
 *     successors with the turn switched, so a successor's value is the
 *     opponent's view.
 *   - Values are per unit of cube (cubeful) or per game (cubeless).
 *   - Ties go to the first candidate in get_legal_plays order.
 *   - An engine is confined to one thread at a time. Engines share no
 *     state, so one engine per thread runs in parallel. get_legal_plays
 *     keeps ~0.5 MB on the stack: threads need >= 1 MB stacks.
 */

#ifndef PURETD_H
#define PURETD_H

#include <stddef.h>
#include <stdint.h>
#include "bg_engine.h"            /* BoardState, Play, WHITE, BLACK */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bgc_engine bgc_engine;      /* model + backend + scratch; thread-confined */

/* ABI version: bump on any change to a public struct or function
   signature. puretd_py.ABI_VERSION must equal it; puretd_py rebuilds or
   refuses a library whose version or struct sizes differ. The library
   also carries the string "puretd-abi:<version>" (puretd_py checks it
   without loading the library). */
#define BGC_ABI_VERSION 1
int bgc_abi_version(void);
/* sizeof the public structs, in this order: BoardState, Play, bgc_cube,
   bgc_cube_eval, bgc_search, bgc_encoder_info, bgc_jan_opts, bgc_rng,
   bgc_game_opts, bgc_game_result, bgc_collect_opts. Writes the first
   min(n, count) into out and returns the count. */
int bgc_struct_sizes(size_t *out, int n);

/* Engine kinds. bgc_open derives the kind from the export's encoder
   (named in its header, resolved through the C API's encoder table,
   puretd.c BGC_BASES / BGC_CUBE_BLOCKS) and output mode: a cubeful
   encoder (a cube block after the base) with an equity output is
   CUBEFUL; a cubeless one is PROB, EQUITY or PROB5 by output mode. Any
   other pairing does not open. */
enum { BGC_KIND_PROB = 0,     /* cubeless, sigmoid P(win): DMP nets                */
       BGC_KIND_EQUITY = 1,   /* cubeless, linear cubeless money equity            */
       BGC_KIND_CUBEFUL = 2,  /* cubeful, linear per-unit cubeful money equity     */
       BGC_KIND_PROB5 = 3,    /* cubeless, 5 sigmoids -> clamp -> money equity     */
       BGC_KIND_PROB5JAN = 6  /* a PROB5 net under the Janowski money cube layer:
                                 bgc_open_janowski only. 4, 5 reserved          */
};
/* Encoders the table knows (encoding.get_encoder names):
     perspective196              196 inputs: the base (cubeless kinds)
     cubeful_perspective196      200: [196 base | centered | mine | theirs |
                                 is_cube_action] (CubefulEncoder)
   bgc_encoder_name names an engine's encoder; bgc_input_size is its
   width. */

/* Rows per backend forward call. */
#define BGC_CHUNK 4096
/* Row stride of the core's input batches: input size rounded up to 16. */
#define BGC_LDX(in) ((((in) + 15) / 16) * 16)

typedef struct { int owner; int value; int jacoby; } bgc_cube;
/* owner: 0 centered, 1 WHITE owns, 2 BLACK owns (= modes.CubeOwner);
   value: 1, 2, 4, ...; jacoby: 0/1. NULL cube = cubeless (kinds 0,1,3). */

/* Env read at open: BGC_EXPAND_DEDUP=0 turns off the 1-ply expansion's
   cross-roll row dedup (results unchanged; for tests and A/B). */
/* bgc_open: the encoder is the one the header names (export_weights.py
   format 2), resolved through the encoder table; the net's input_size
   must be that encoder's width. Fails on an unknown encoder or base
   encoder name, a width mismatch or an output mode the encoder has no
   kind for, with a message naming the encoder. LEGACY PATH: a format-1
   file (no encoder name) opens by its width: 196 -> perspective196,
   200 -> cubeful_perspective196; any other width fails. */
bgc_engine *bgc_open(const char *bgnn_path);      /* NULL + stderr on failure */
/* bgc_open, the failure's message into err[errlen] (NUL-terminated,
   truncated to fit; err nullable) instead of stderr. */
bgc_engine *bgc_open_ex(const char *bgnn_path, char *err, size_t errlen);
void        bgc_close(bgc_engine *);
int         bgc_kind(const bgc_engine *);
int         bgc_input_size(const bgc_engine *);
const char *bgc_encoder_name(const bgc_engine *);  /* the encoder, resolved */
int         bgc_format_version(const bgc_engine *); /* the export's header format: 2,
                                                       1 = legacy (encoder by width) */
const char *bgc_describe(const bgc_engine *);      /* bgc_backend_describe() */

/* The encoder table's entry for an encoder name. */
typedef struct {
    int n_base;         /* base encoder features (perspective196: 196) */
    int ncube;          /* cube columns after them (0: cubeless) */
    int input_size;     /* n_base + ncube */
    int collect;        /* bgc_collect writes rows of it (else BGC_COLLECT_UNSUPPORTED) */
} bgc_encoder_info;
/* 0 and *out (nullable) for a name bgc_open accepts; -1 otherwise. */
int bgc_encoder_lookup(const char *encoder_name, bgc_encoder_info *out);

/* 0-ply net value of `state` from state->turn's view. CUBEFUL: cube required
   (NULL -> NAN), is_cube_action sets the 4th cube feature. prob5: clamped
   equity. */
float bgc_value(bgc_engine *, const BoardState *, const bgc_cube *, int is_cube_action);
/* PROB5 only: the 5 clamped probabilities and the equity. -1 for other kinds. */
int   bgc_value_prob5(bgc_engine *, const BoardState *, float probs[5], float *equity);
/* 1-ply checker value of `state` for the player on roll. Terminal state:
   NAN, except PROB5: -game_result (ProbAgent.value_oneply_checker).
   CUBEFUL with a NULL cube: NAN. */
float bgc_value_1ply(bgc_engine *, const BoardState *, const bgc_cube *);

/* Checker play at `plies` (0 or 1). Returns the chosen index into the
   get_legal_plays order, -1 on a forced pass (nothing evaluated), -2 on
   invalid arguments or allocation failure. out_state gets the resulting
   state (turn already switched, engine convention); out_play (nullable)
   the Play; out_value (nullable) the selector's score of the chosen play
   from the MOVER's view.
   CUBEFUL with a NULL cube: TDAgent.choose_checker_action (centered cube
   columns, is_cube_action 0, 0-ply at any `plies`). */
int bgc_best_move(bgc_engine *, const BoardState *, int d1, int d2,
                  const bgc_cube *, int plies,
                  BoardState *out_state, Play *out_play, float *out_value);

/* bgc_value_1ply(s, cube) and bgc_best_move(s, d1, d2, cube, plies 0) in
   one call: the roll's candidates are among the 1-ply expansion's replies
   (same order, same cube columns), so their 0-ply values come from the
   expansion's forward. Results equal the two calls' (bitwise on backends
   whose per-row results do not depend on the batch: ref, simd; the
   others make the two calls). Returns as bgc_best_move; *value_1ply as
   bgc_value_1ply (NAN on -2). */
int bgc_value_1ply_play(bgc_engine *, const BoardState *, int d1, int d2,
                        const bgc_cube *, float *value_1ply,
                        BoardState *out_state, float *out_value);

/* Every candidate's selector score from the mover's view (the out_value
   bgc_best_move would report for it), in get_legal_plays order.
   `scores` and `states` (nullable) hold MAX_PLAYS entries. Returns the
   candidate count (0 on a forced pass), -2 on error. */
int bgc_score_moves(bgc_engine *, const BoardState *, int d1, int d2,
                    const bgc_cube *, int plies,
                    float *scores, BoardState *states);

/* 1-ply root move filter. root_thresh > 0: a 1-ply checker decision first
   scores every candidate at 0-ply and expands at 1-ply only those within
   root_thresh of the best 0-ply score (game-ending candidates always kept).
   A single survivor is chosen without expansion (its score is the 0-ply
   one); dropped candidates score NAN in bgc_score_moves. Cube decisions
   stay full 1-ply. NULL or root_thresh 0: full width (the default).
   Returns 0, -1 on a bad argument. */
typedef struct { float root_thresh; } bgc_search;
int bgc_set_search(bgc_engine *, const bgc_search * /* nullable */);

/* Money cube actions, CUBEFUL and PROB5JAN (others: offer 0, respond 1). -1 on
   invalid arguments (NULL cube, plies not 0/1). */
typedef struct { int plies; float v_no_double; float v_double_take;
                 int take; /* the responder's decision, v_double_take <= 1 in double;
                              0 when not computed (values NAN) */ } bgc_cube_eval;
int bgc_offer_double(bgc_engine *, const BoardState *, const bgc_cube *, int plies,
                     bgc_cube_eval *out /* nullable */);         /* 1 double, 0 not */
/* state->turn is the DOUBLER; returns the responder's decision. The hint
   is used when hint->plies == plies. */
int bgc_respond_to_double(bgc_engine *, const BoardState *, const bgc_cube *, int plies,
                          const bgc_cube_eval *hint /* nullable */); /* 1 take, 0 pass */

/* ── Prob5Jan: prob5 + Janowski money cube (puretd_janowski.c) ──────
 * janowski_agent.JanowskiAgent at its defaults (cube_fold, fold_0ply,
 * board_sanity; bearoff_exact iff a table is attached). Cube required on
 * every call (NULL -> the error value). On a PROB5JAN engine:
 *   bgc_value         0-ply no-double value (_static_cube); ica ignored
 *   bgc_value_prob5   the net's probabilities: prob5_impossible_mask zeroing
 *                     (board_sanity), the clamps; their equity
 *   bgc_value_1ply    _value_oneply(s, cube, fold=True); terminal: NAN
 *   bgc_best_move / bgc_score_moves: mover view = -(leaf player's value);
 *                     ties in float32 go to the lower leaf-view cubeless
 *                     equity (not in the scores), then the first. 1-ply
 *                     expands every candidate except game-ending and
 *                     bearoff-table ones (their 0-ply score)
 *   bgc_value_1ply_play  the two calls
 *   bgc_offer_double  (nd, dt) at plies, double per TDAgent's money rule,
 *                     take = dt <= 1; THEIRS: 0, nothing computed
 *   bgc_respond_to_double  hint->take when the hint is for plies and was
 *                     computed, else the take of the same values
 *   bgc_set_search    root_thresh 0 only (-1 otherwise)
 *   bgc_collect       -1
 * bgc_play_game and the bench calls work with cubeful opts. */
/* A perspective196 prob5 export as PROB5JAN (x 0.76, board_sanity, no
   bearoff table). NULL + err on failure (err nullable). */
bgc_engine *bgc_open_janowski(const char *bgnn_path, char *err, size_t errlen);
typedef struct { double x; int board_sanity; } bgc_jan_opts;
/* x (cube-life index, finite) and board_sanity; NULL = defaults (0.76, 1).
   0, -1 on another kind or a non-finite x. */
int bgc_set_jan(bgc_engine *, const bgc_jan_opts *);
/* bearoff.py's tables from the flat file bearoff.export_c_tables writes;
   mmapped read-only, shareable by any number of engines and threads. Must
   outlive the engines it is attached to. NULL + err on failure. */
typedef struct bgc_bearoff bgc_bearoff;
bgc_bearoff *bgc_bearoff_open(const char *path, char *err, size_t errlen);
void bgc_bearoff_close(bgc_bearoff *);
/* Attach (NULL: detach = bearoff_exact off). 0, -1 on another kind. */
int bgc_set_bearoff(bgc_engine *, const bgc_bearoff *);
/* The conversion alone (janowski.py), double, Python's operation order.
   p = P(w), P(wg), P(wbg), P(lg), P(lbg) of the player on roll; persp =
   that player's cube perspective (PERSP 0 centered, 1 mine, 2 theirs). */
double bgc_jan_equity(const double p[5], int persp, double x, int jacoby);
double bgc_jan_equity_folded(const double p[5], int persp, double x, int jacoby);
double bgc_jan_cubeless_equity(const double p[5]);          /* gammons count */
double bgc_jan_fold(double nd, double dt);                  /* fold_cube_decision, dp 1 */
/* money_cube_action: returns double (TDAgent._money_cube_decision); *take
   (nullable) = dt <= 1. */
int bgc_jan_money_cube_action(double nd, double dt, int persp, int jacoby, int *take);
/* The tables alone (bearoff.py; state->turn on roll): 1 found, 0 not
   applicable (not bearoff class / no exact tier / not in the two-sided
   table), -1 bad arguments or allocation failure. */
int bgc_bo_exact_probs5(const bgc_bearoff *, const BoardState *, double p[5]);
int bgc_bo_race_equity(const bgc_bearoff *, const BoardState *, int persp, int fold,
                       double *v);
int bgc_bo_race_cube_values(const bgc_bearoff *, const BoardState *, int persp,
                            double *nd, double *dt);
int bgc_bo_ts_win_prob(const bgc_bearoff *, const BoardState *, double *p);

/* Python-compatible RNG: random.Random(seed) for non-negative int seeds. */
typedef struct { uint32_t mt[624]; int idx; } bgc_rng;
void         bgc_rng_seed(bgc_rng *, unsigned long long seed);
unsigned int bgc_rng_u32(bgc_rng *);                  /* genrand_uint32 */
unsigned int bgc_rng_getrandbits(bgc_rng *, int k);   /* 1 <= k <= 32 */
int          bgc_rng_randint(bgc_rng *, int a, int b);  /* inclusive, Python semantics */

/* Self-play. */
typedef struct {
    int cubeful;        /* 1: money game with cube; 0: cubeless (gammons count) */
    int jacoby;
    int cube_enabled;   /* cubeful only. 0: never double, offers still counted */
    int plies_white, plies_black;
} bgc_game_opts;
typedef struct {
    int winner, result, cube_value, ended_by_drop;
    int n_checker;      /* cubeful: every checker call incl. forced passes;
                           cubeless: excludes forced passes */
    int n_dance;        /* forced passes */
    int n_cube_offer, n_cube_resp;
    int n_turns;        /* dice rolls */
    int n_dice;         /* single dice drawn from the RNG, opening rerolls included */
} bgc_game_result;
/* Optional per-decision callback. kind:
   0 checker (d1,d2 valid; next_state = chosen or NULL on a pass),
   1 offer (decision = 0/1), 2 respond (decision = 0/1). value = the
   selector's mover-view score (checker; NAN on a pass) or v_double_take
   (cube; NAN when not computed). decision = chosen index (checker). */
typedef void (*bgc_record_fn)(void *ctx, int kind, const BoardState *state,
                              int d1, int d2, const bgc_cube *cube,
                              int decision, const BoardState *next_state,
                              float value);
/* Plays one game (gnubg_eval.play_and_record_cubeful, or
   play_models.play_game cubeless). white == black passes the offer as the
   response hint, as Python self-play does. 0 on success, -1 on error. */
int bgc_play_game(bgc_engine *white, bgc_engine *black, const bgc_game_opts *,
                  bgc_rng *, bgc_game_result *, bgc_record_fn /* nullable */, void *ctx);

/* Batch-training collector (puretd_collect.c): the samples of
   trainer.collect_episode with a TDAgent at boltzmann_temp 0, play at
   0-ply (oneply_acting off). Kind PROB = dmp (terminal target 1),
   EQUITY = cubeless money (terminal 1/2/3), CUBEFUL = cubeful money
   (cube-phase samples, per-unit targets), PROB5 = the samples of
   train_prob5._collect_one_episode (cubeless money, 5-vector targets:
   P(win, win gammon, win bg, lose gammon, lose bg) of the player on roll
   at the row; 0-ply: the chosen successor's clamped outputs flipped, or
   [1, gammon, bg, 0, 0] on a game-ending play; 1-ply:
   _oneply_target_vec). dmp with an equity net is not supported (the mode
   is the kind's), nor an encoder whose table entry has collect 0
   (bgc_encoder_lookup: BGC_COLLECT_UNSUPPORTED). */
typedef struct {
    int oneply;         /* 1-ply targets (play stays 0-ply) */
    int jacoby;         /* cubeful only */
    int max_plies;      /* turns; a longer episode is discarded whole.
                           <= 0 = unbounded (trainer default 1500;
                           train_prob5 has no bound: pass 0) */
} bgc_collect_opts;
/* Plays `n_episodes` self-play episodes on one RNG seeded like
   random.Random(seed) and appends one (row, target) per sample. enc:
   caller buffer, row stride bgc_input_size(e), capacity `cap` rows; tgt:
   `cap * out_dim` floats, row i's target at tgt[i * out_dim], out_dim =
   5 for PROB5, else 1. Returns the rows written, -1 on error, -2 when cap
   is too small (outputs partial: grow and call again with the same seed,
   a round is replayable), BGC_COLLECT_UNSUPPORTED (-3, nothing written)
   for an encoder without a C collector (collect 0 in the encoder table).
   n_discarded (nullable): episodes dropped for max_plies. */
#define BGC_COLLECT_UNSUPPORTED (-3)
long bgc_collect(bgc_engine *e, const bgc_collect_opts *o,
                 unsigned long long seed, int n_episodes,
                 float *enc, float *tgt, long cap, int *n_discarded);

/* Benchmark entry points: each item timed inside C with CLOCK_MONOTONIC.
   0 on success, -1 on error. */
int bgc_bench_checker(bgc_engine *, const BoardState *states, const int *dice /* 2n */,
                      const bgc_cube *cubes /* nullable */, int n, int plies,
                      double *out_seconds /* n */);
int bgc_bench_cube(bgc_engine *, const BoardState *states, const bgc_cube *cubes,
                   int n, int plies, double *out_seconds);
/* `warmup_games` untimed games first (cubeful: with cube_enabled = 1); if
   reseed_after_warmup the RNG is reseeded with `seed` before the timed
   games, else the stream continues. Per timed game: wall seconds and the
   decision count: checker decisions (turns with a legal play; forced
   passes are not decisions in either mode) + (cubeful and cube_enabled ?
   offers + responses : 0). totals: {checker (that count), dance, offers,
   responses, turns, games} over the timed games. */
int bgc_bench_selfplay(bgc_engine *, const bgc_game_opts *, unsigned long long seed,
                       int warmup_games, int reseed_after_warmup, int games,
                       double *out_game_seconds, int *out_game_decisions,
                       long long totals[6]);

#ifdef __cplusplus
}
#endif

#endif /* PURETD_H */
