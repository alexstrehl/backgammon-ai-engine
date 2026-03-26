/*
 * bg_engine.h -- Backgammon engine: board state, move generation, encoding.
 *
 * Designed to be called from Python via ctypes.
 * All functions use simple C types and pre-allocated buffers.
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
#define NUM_FEATURES_210   210
#define NUM_FEATURES_224   224
#define NUM_FEATURES_246   246
#define NUM_FEATURES       196   /* default for backward compat */

#define BAR_SENTINEL   (-1)
#define OFF_SENTINEL   (-2)

#define MAX_MOVES_PER_PLAY  4
#define MAX_PLAYS         512   /* generous upper bound */

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

/*
 * Initialize a board to the standard backgammon starting position.
 * WHITE to move.
 */
void board_init(BoardState *state);

/*
 * Return 1 if the game is over (someone has borne off all 15).
 */
int board_is_game_over(const BoardState *state);

/*
 * Return the winner (WHITE or BLACK), or -1 if game not over.
 */
int board_winner(const BoardState *state);

/*
 * Find all legal plays for state->turn given dice (d1, d2).
 * Fills the plays[] array and returns the count.
 *
 * Enforces:
 *   - Must enter from bar before other moves.
 *   - Must use as many dice as possible.
 *   - If only one die can be used (non-doubles), must use the larger.
 *   - Bearing off exact/over-bear rules.
 *   - Deduplication by resulting board state.
 *   - Doubles = 4 sub-moves.
 */
int get_legal_plays(const BoardState *state, int d1, int d2,
                    Play *plays, int max_plays);

/*
 * Encode a board state into a 196-element float vector
 * (perspective encoding, always from on-roll player's view).
 */
void encode_state(const BoardState *state, float *features);

/*
 * Encode a board state into a 210-element float vector:
 * 196 base features + 14 gnubg Group-1 expert features.
 */
void encode_state_210(const BoardState *state, float *features);

/*
 * Combined: get all legal plays AND encode with 210-feature encoding.
 */
int get_legal_plays_encoded_210(const BoardState *state, int d1, int d2,
                                Play *plays, int max_plays,
                                float *encoded_features);

/*
 * Encode a board state into a 224-element float vector:
 * 196 base + 14 Group-1 + 14 Group-2 features.
 */
void encode_state_224(const BoardState *state, float *features);

/*
 * Combined: get all legal plays AND encode with 224-feature encoding.
 */
int get_legal_plays_encoded_224(const BoardState *state, int d1, int d2,
                                Play *plays, int max_plays,
                                float *encoded_features);

void encode_state_246(const BoardState *state, float *features);

int get_legal_plays_encoded_246(const BoardState *state, int d1, int d2,
                                Play *plays, int max_plays,
                                float *encoded_features);

/*
 * Flip the turn (WHITE <-> BLACK) in place.
 */
void board_switch_turn(BoardState *state);

/*
 * Combined: get all legal plays AND encode all resulting states in one call.
 *
 * This avoids per-play Python<->C round trips during training.
 * The resulting board states are kept in plays[]; the encoded features
 * for each resulting state are written to encoded_features as a flat
 * array of (num_plays * NUM_FEATURES) floats.
 *
 * Parameters:
 *   state:            current board state
 *   d1, d2:           dice roll
 *   plays:            output buffer for plays (pre-allocated, size max_plays)
 *   max_plays:        capacity of plays buffer
 *   encoded_features: output buffer for features (pre-allocated, max_plays * 198)
 *
 * Returns: number of legal plays found.
 */
int get_legal_plays_encoded(const BoardState *state, int d1, int d2,
                            Play *plays, int max_plays,
                            float *encoded_features);

/*
 * Copy a single resulting board state from the plays buffer.
 * Used after choosing the best play index to extract just that state.
 */
void get_play_resulting_state(const Play *plays, int index,
                              BoardState *out_state);

#ifdef __cplusplus
}
#endif

#endif /* BG_ENGINE_H */
