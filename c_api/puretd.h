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
#define BGC_ABI_VERSION 3
int bgc_abi_version(void);
/* sizeof the public structs, in this order: BoardState, Play, bgc_cube,
   bgc_cube_eval, bgc_search, bgc_encoder_info, bgc_rng, bgc_game_opts,
   bgc_game_result, bgc_collect_opts, bgc_match, bgc_match_cube,
   bgc_match_collect_opts, bgc_met, bgc_jan_ctx (puretd_py._ABI_STRUCTS).
   Writes the first min(n, count) into out and returns the count. */
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
       BGC_KIND_MATCH = 4,    /* match play: a match block after the base, sigmoid
                                 P(row's player wins the MATCH) (bgc_match_*)     */
       BGC_KIND_MATCH_JANOWSKI = 5 };  /* match play with a PROB5 net under the
                                 Janowski conversion (p5jan; bgc_open_janowski)   */
/* Encoders the table knows (encoding.get_encoder names):
     perspective196              196 inputs: the base (cubeless kinds)
     cubeful_perspective196      200: [196 base | centered | mine | theirs |
                                 is_cube_action] (CubefulEncoder)
     matchplay1h64met_perspective196   350: [196 base | 154 match columns]
                                 (Matchplay1H64METEncoder: cube perspective 3,
                                 is_cube_action, cube 7, my_away 64, opp_away
                                 64, is_crawford, post_crawford, MET block 13)
     matchplay1h64met6_perspective196  354: the same with the 17-column MET
                                 block (Matchplay1H64MET6Encoder)
   The match encoders open as BGC_KIND_MATCH (probability output only) and are
   served by the bgc_match_* entry points, which need bgc_set_met first;
   the money entry points return their error value on a MATCH engine. A
   prob5 export opened with bgc_open_janowski is BGC_KIND_MATCH_JANOWSKI
   (p5jan), served by the same bgc_match_* calls ("p5jan" below).
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
   cross-roll row dedup (results unchanged; for tests and A/B).
   BGC_COLLECT_OFF=1 treats the encoder's table entry as collect 0 (a test
   hook for the BGC_COLLECT_UNSUPPORTED path of bgc_collect /
   bgc_collect_match). */
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
/* A prob5 export (a cubeless base encoder, 5 sigmoid outputs: what bgc_open
   opens as BGC_KIND_PROB5) opened as a match engine instead,
   BGC_KIND_MATCH_JANOWSKI: janowski_match.JanowskiMatchAgent (p5jan) through
   the bgc_match_* entry points (below, "p5jan"). Fails like bgc_open_ex, and
   on any other export. */
bgc_engine *bgc_open_janowski(const char *bgnn_path, char *err, size_t errlen);
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
    int ncube;          /* cube columns after them (0: cubeless; match: the whole
                           match block, 154 / 158) */
    int input_size;     /* n_base + ncube */
    int collect;        /* bgc_collect (match: bgc_collect_match) writes rows of it
                           (else BGC_COLLECT_UNSUPPORTED) */
    int match;          /* a match encoder (BGC_KIND_MATCH) */
    int nmet;           /* match: the MET block's columns (13 / 17), the block's last */
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

/* Money cube actions, CUBEFUL only (others: offer 0, respond 1). -1 on
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

/* Python-compatible RNG: random.Random(seed) for non-negative int seeds. */
typedef struct { uint32_t mt[624]; int idx; } bgc_rng;
void         bgc_rng_seed(bgc_rng *, unsigned long long seed);
unsigned int bgc_rng_u32(bgc_rng *);                  /* genrand_uint32 */
unsigned int bgc_rng_getrandbits(bgc_rng *, int k);   /* 1 <= k <= 32 */
int          bgc_rng_randint(bgc_rng *, int a, int b);  /* inclusive, Python semantics */
double       bgc_rng_random(bgc_rng *);               /* random.Random.random() */

/* ── match play (BGC_KIND_MATCH engines; puretd_match.c) ─────────────
 * Ports of the TDAgent matchplay methods (td_agent.py: _encode_matchplay,
 * _eval_new_game with a MET grounding every scope, offer_double_matchplay
 * / respond_to_double_matchplay, choose_checker_action_matchplay at
 * boltzmann_temp 0, value_oneply_checker_matchplay with cube_fold off)
 * and of modes.MatchState. A bgc_match is a MatchState in WHITE's view
 * (perspective flips at encode time only); every value is P(the player
 * named wins the MATCH), fp32 net outputs, MWCs and sums in double.
 * Every call needs bgc_set_met first (-1 / -2 / NAN otherwise) and a
 * bgc_match with length >= 1; an away score the MET cannot price
 * (Python raises) fails the call. */
typedef struct {            /* modes.MatchState, WHITE's view */
    int length, away[2];    /* away[0] = my_away (WHITE), away[1] = opp_away */
    int cube_owner;         /* 0 centered, 1 WHITE, 2 BLACK */
    int cube_value, is_crawford, crawford_done;
} bgc_match;
typedef struct { double v_nd, v_dt, v_dp; } bgc_match_cube;
/* MET tables (MatchEquityTable._pre n x n row-major, ._post n), copied into
   the engine; required before any bgc_match_* call (-1 / NAN otherwise).
   0, or -1 (not a MATCH engine, n < 1, a non-finite entry). */
int    bgc_set_met(bgc_engine *, const double *pre, const double *post, int n);
/* _encode_matchplay(s, m, ica): base(s) + match block from s->turn's view.
   row: bgc_input_size floats. 0, or -1 (Python raises: an away < 1, a cube
   value its one-hot has no slot for, an away the MET cannot price). */
int    bgc_match_encode(bgc_engine *, const BoardState *, const bgc_match *, int ica,
                        float *row /* bgc_input_size floats */);
/* TDAgent._eval_new_game(after_game, player): the MET's MWC of `player`
   at the start of the next game (start_game, then mwc). NAN on error. */
double bgc_match_new_game_mwc(const bgc_engine *, const bgc_match *after_game, int player);
/* evaluate_matchplay: V(_encode_matchplay(s, m, ica)), s->turn's view. */
float  bgc_match_value(bgc_engine *, const BoardState *, const bgc_match *, int ica);
/* 1 double, 0 not (out: v_nd, v_dt, v_dp), 0 + NANs when !can_offer, -1 error.
   Take = out->v_dp > out->v_dt. */
int    bgc_match_offer(bgc_engine *, const BoardState *, const bgc_match *, bgc_match_cube *out);
/* 0-ply: index in get_legal_plays(s, d1, d2) order, -1 pass, -2 error;
   out_value = the chosen successor's opp-view value (terminal-overridden). */
int    bgc_match_best_move(bgc_engine *, const BoardState *, int d1, int d2,
                           const bgc_match *, BoardState *out_state, float *out_value);
/* Every candidate's opp-view value (terminal-overridden, the values
   bgc_match_best_move takes the first argmin of) and successor, in
   get_legal_plays order; MAX_PLAYS entries each (nullable). Returns the
   candidate count (0 on a forced pass), -2 on error. */
int    bgc_match_score_moves(bgc_engine *, const BoardState *, int d1, int d2,
                             const bgc_match *, float *opp_values, BoardState *states);
/* value_oneply_checker_matchplay(s, m) (double; the collector stores
   float32 of it). NAN on error or a terminal s. */
double bgc_match_value_1ply(bgc_engine *, const BoardState *, const bgc_match *);
/* bgc_match_value_1ply(s, m) and bgc_match_best_move(s, d1, d2, m) from one
   expansion (the roll's candidates are expansion replies with the same
   match columns; bitwise the two calls on ref / simd, the two calls on the
   others). Returns as bgc_match_best_move; *v1 (NAN on -2); out_state
   (nullable) set for an index >= 0. */
int    bgc_match_value_1ply_play(bgc_engine *, const BoardState *, int d1, int d2,
                                 const bgc_match *, double *v1, BoardState *out_state);

/* The 1-ply memo: bgc_match_value_1ply / _1ply_play reuse the value of a
   (board, match state) seen before by this engine (bitwise the value they
   would compute: a pure function of the two for fixed weights and MET);
   cleared by bgc_set_met; off with BGC_MATCH_MEMO=0 at open. Its counters
   (nullable); returns 1 on, 0 off, -1 not a MATCH engine. */
int    bgc_match_memo_stats(const bgc_engine *, long *lookups, long *hits, long *entries);

/* The memo shared by the processes of a round (BGC_MATCH_SHM, puretd_match.c
   "the shared memo"): a value another engine on the same weights file
   contents, library file, backend configuration and MET computed. Its
   counters since bgc_set_met (nullable): lookups / hits in the shared table,
   values this engine put there; returns 1 attached, 0 not, -1 not a MATCH
   engine. */
int    bgc_match_shm_stats(const bgc_engine *, long *lookups, long *hits, long *puts);

/* ── p5jan: the match calls on a BGC_KIND_MATCH_JANOWSKI engine ─────
 * (puretd_match.c's evaluator seam). The same entry points
 * with janowski_match.JanowskiMatchAgent's value function: a row's value is
 * cubeful_mwc(clamped prob5(row), cube_perspective(owner, row player),
 * ctx_for(m, row player), x(row board)) in double; the net reads the base
 * columns only. bgc_set_met first, as for BGC_KIND_MATCH. Then:
 *   bgc_match_value       cubeful_mwc at s for s->turn (fp32 of it); ica ignored
 *   bgc_match_offer       offer_double / cube_decision: v_nd, v_dt (doubled
 *                         cube, THEIRS), v_dp = met.mwc_after(.., cube, win);
 *                         one forward; take = v_dp > v_dt as for the TD kind
 *   bgc_match_best_move / _score_moves  choose_checker_action_cubeful: the
 *                         FIRST MAXIMUM of the mover-view doubles (1 -
 *                         cubeful_mwc of the opponent on roll; a terminal
 *                         successor met.mwc_after(.., cube * result, win));
 *                         out_value / opp_values: float32(1 - that value)
 *   bgc_match_value_1ply / _1ply_play   value_oneply (per roll the mover's
 *                         best by that value, the 21 summed in dice order, no
 *                         clamp, no all-pass rule); the memos as for MATCH
 *   bgc_match_encode      the base row (input_size floats)
 *   bgc_match_new_game_mwc, _memo_stats, _shm_stats   as for MATCH
 * bgc_collect_match and the money entry points refuse the kind. */
/* x, the cube-life index: constant 0 = janowski.cube_x of the evaluated
   board with its player on roll (gnubg's shallow position classes; the
   default, JanowskiMatchAgent(x_policy=None)); constant 1 = `x` for every
   row (x_policy=x; finite). Clears the 1-ply memo. 0, or -1 (not a
   MATCH_JANOWSKI engine, a non-finite constant). */
int bgc_set_jan_x(bgc_engine *, int constant, double x);

/* The match-episode collector (puretd_match_collect.c): the samples of
   trainer._collect_episode_matchplay with a TDAgent at
   boltzmann_temp 0, 0-ply acting, the engine's MET on every scope, no
   oracle / td-move-eps / move exploration; rows (bgc_input_size floats)
   and float32 targets (P(row's player wins the match)) in the Python
   order, episodes in start order. The RNG draws are Python's, including
   the coins drawn at probability 0 (the curriculum coin at P 0, the cube
   exploration coins at eps 0). */
typedef struct {
    int match_length, oneply;     /* MatchplayMode.match_length; 1-ply targets (play 0-ply) */
    double curriculum_p;          /* trainer._SCORE_CURRICULUM_P */
    int force_my, force_opp;      /* TRAINER_CURRICULUM_FORCE; 0 = unset */
    double cube_explore_eps;      /* trainer._CUBE_EXPLORATION_EPS */
    int boundary_upweight;        /* trainer._BOUNDARY_UPWEIGHT, >= 1 */
    double outcome_mix;           /* MatchplayMode.outcome_mix */
    int max_plies;                /* <= 0 unbounded (serial); > 0 vectorized semantics:
                                     a game reaching max_plies turns discards the match */
    int rng_mode;                 /* 0: one stream (_collect_worker); 1: per-episode
                                     Random(base.randint(0, 2**31-1)) (vectorized) */
} bgc_match_collect_opts;
/* Plays n_episodes matches from random.Random(seed) (rng_mode as above) and
   appends one row and target per sample (cap rows; enc row stride
   bgc_input_size). Returns the rows written, -1 on an error or a bad
   argument (no MET, not a MATCH engine, match_length < 1 or above the MET,
   probabilities outside [0, 1], upweight < 1, ...), -2 when cap is too
   small (outputs partial: replay the same seed into a bigger buffer),
   BGC_COLLECT_UNSUPPORTED (nothing written) for an encoder whose entry has
   collect 0. n_discarded (nullable): matches dropped for max_plies. */
long bgc_collect_match(bgc_engine *, const bgc_match_collect_opts *, unsigned long long seed,
                       int n_episodes, float *enc, float *tgt, long cap, int *n_discarded);

/* ── p5jan: the Janowski conversion (puretd_janowski.c) ───────────────
 * janowski_match.py's gnubg-style cubeless -> cubeful match-winning-chance
 * conversion (gnubg 1.08.003 Cl2CfMatch / GetPoints / fDoCubeful,
 * janowski.cube_x), in double in Python's operation order without FMA
 * contraction: given the same prob5 outputs the values are Python's
 * (tests/test_janowski_c.py). No net, no engine. probs: the nested
 * P(win), P(win gammon), P(win bg), P(lose gammon), P(lose bg) of the
 * evaluating player; persp: 0 centered, 1 mine, 2 theirs
 * (encoding.CubePerspective). NAN / -1 where Python raises (an away the
 * MET cannot price) or on a NULL argument. */
typedef struct { const double *pre, *post; int n; } bgc_met;   /* MatchEquityTable._pre
                                                        (n x n row-major) and ._post (n) */
typedef struct { int my_away, opp_away, crawford_done, is_crawford, cube; } bgc_jan_ctx;
                                       /* janowski_match.MatchCtx: the evaluating player's view */
int    bgc_jan_do_cubeful(const bgc_jan_ctx *);                    /* do_cubeful: 1 / 0 */
double bgc_jan_dead_mwc(const bgc_met *, const double probs[5], const bgc_jan_ctx *);
/* get_points(rates, ctx): rates = (g_me, bg_me, g_opp, bg_opp) conditional;
   cp = (cp_me, cp_opp). 0, -1. */
int    bgc_jan_get_points(const bgc_met *, const double rates[4], const bgc_jan_ctx *,
                          double cp[2]);
double bgc_jan_cubeful_mwc(const bgc_met *, const double probs[5], int persp,
                           const bgc_jan_ctx *, double x);
/* cube_decision(probs, m, player, x): out (nullable) = v_nd, v_dt, v_dp, the
   doubler's MWCs (no can_offer gate, as in Python); returns should_double
   (min(v_dt, v_dp) > v_nd) 1 / 0, -1 on an error. take = v_dp > v_dt. */
int    bgc_jan_cube_decision(const bgc_met *, const double probs[5], const bgc_match *m,
                             int player, double x, bgc_match_cube *out);
/* janowski.cube_x(s): contact 0.76, race clamp(0.60 + 0.0025 * pips of
   s->turn, 0.60, 0.80). */
double bgc_jan_cube_x(const BoardState *);

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

/* Test hook (puretd_movegen.c): the plays of (s, d1, d2)
   into plays[MAX_PLAYS] (bg_engine.h) by the engine's fast generator
   (fast 1: bgc_i_legal_plays, BGC_FAST_MOVEGEN) or by get_legal_plays (fast
   0); the count. The two lists must be identical. */
int bgc_test_legal_plays(const BoardState *s, int d1, int d2, int fast, Play *plays);

#ifdef __cplusplus
}
#endif

#endif /* PURETD_H */
