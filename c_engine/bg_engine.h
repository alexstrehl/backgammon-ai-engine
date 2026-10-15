/*
 * bg_engine.h -- Backgammon engine: board state, move generation, encoding.
 *
 * Called from Python via ctypes (c_engine/bg_fast.py): plain C types and
 * caller-allocated buffers.
 *
 * Turn convention: get_legal_plays switches the turn on every
 * resulting_state, so plays[i].resulting_state.turn is the OPPONENT of
 * state.turn. `.turn` always means "the player to act next".
 */

#ifndef BG_ENGINE_H
#define BG_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

/* ── Constants ─────────────────────────────────────────────────── */

#define WHITE 0
#define BLACK 1

#define NUM_POINTS     24
#define NUM_CHECKERS   15
#define NUM_FEATURES_196   196
#define NUM_FEATURES       196   /* backward-compat alias */

#define BAR_SENTINEL   (-1)
#define OFF_SENTINEL   (-2)

#define MAX_MOVES_PER_PLAY  4
#define MAX_PLAYS         4096  /* most distinct legal plays seen is ~1200. Keep in sync with bg_fast.py MAX_PLAYS. */

/* ── Board state ───────────────────────────────────────────────── */

typedef struct {
    int points[NUM_POINTS]; /* >0 = WHITE checkers, <0 = BLACK */
    int bar[2];             /* [WHITE, BLACK] */
    int off[2];             /* [WHITE, BLACK] */
    int turn;               /* WHITE or BLACK */
} BoardState;

/* ── Move / Play ───────────────────────────────────────────────── */

typedef struct {
    int src;  /* point index (0-23), or BAR_SENTINEL */
    int dst;  /* point index (0-23), or OFF_SENTINEL */
} Move;

typedef struct {
    Move moves[MAX_MOVES_PER_PLAY];
    int num_moves;
    BoardState resulting_state;
} Play;

/* ── API functions ─────────────────────────────────────────────── */

/* Standard starting position, WHITE to move. */
void board_init(BoardState *state);

/* 1 if someone has borne off all 15. */
int board_is_game_over(const BoardState *state);

/* WHITE or BLACK, or -1 if the game is not over. */
int board_winner(const BoardState *state);

/*
 * All legal plays for state->turn with dice (d1, d2); fills plays[] and
 * returns the count. Enforces: enter from the bar first; use as many
 * dice as possible; if only one die is playable (non-doubles), use the
 * larger; exact/over-bear bear-off; doubles = 4 sub-moves; dedup by
 * resulting board. resulting_state.turn is switched (see top of file).
 */
int get_legal_plays(const BoardState *state, int d1, int d2,
                    Play *plays, int max_plays);

/* 196-float perspective encoding, from the on-roll player's view. */
void encode_state(const BoardState *state, float *features);


/* Flip the turn in place. */
void board_switch_turn(BoardState *state);

/*
 * get_legal_plays plus encode_state of every resulting state, in one
 * call. encoded_features holds max_plays * NUM_FEATURES floats, play i
 * at row i. Returns the play count.
 */
int get_legal_plays_encoded(const BoardState *state, int d1, int d2,
                            Play *plays, int max_plays,
                            float *encoded_features);

/* Copy plays[index].resulting_state to out_state. */
void get_play_resulting_state(const Play *plays, int index,
                              BoardState *out_state);

/*
 * 1-ply inner expansion of a candidate resulting state (opponent on
 * roll): for all 21 opponent dice, encode every non-terminal reply's
 * 196-float vector into out_feats (row-major, reply order), fill
 * out_dice_counts[21], and per reply set out_reply_gr to 0
 * (non-terminal, has a row) or the winner's game_result 1/2/3
 * (terminal, no row). Returns the reply count, or -1 on overflow.
 * See bg_engine.c for the reply/row mapping.
 */
int get_legal_plays_encoded_21(const BoardState *next_state,
                               float *out_feats,
                               int *out_dice_counts,
                               int *out_reply_gr,
                               int max_rows,
                               int max_replies);

/*
 * gnubg 1.08 SanityCheck "impossible" rules on n perspective196 rows
 * (mirrors prob_agent.prob5_impossible): mask[4*r + j] = 1 zeroes
 * P(wg), P(wbg), P(lg), P(lbg) of row r. Returns the rows with any 1.
 */
int prob5_impossible_mask(const float *x, int n, unsigned char *mask);

#ifdef __cplusplus
}
#endif

#endif /* BG_ENGINE_H */
