/*
 * bg_engine.c -- Backgammon engine: board state, move generation, encoding.
 *
 * Port of backgammon_engine.py and encoding.py to C for performance.
 * Designed to be compiled as a shared library and called via ctypes.
 */

#include "bg_engine.h"
#include <assert.h>
#include <string.h>  /* memcpy, memset */

/* ── Board state ───────────────────────────────────────────────── */

void board_init(BoardState *s) {
    memset(s, 0, sizeof(BoardState));
    /* WHITE checkers (positive) */
    s->points[23] = 2;    /* point 24 */
    s->points[12] = 5;    /* point 13 */
    s->points[7]  = 3;    /* point 8  */
    s->points[5]  = 5;    /* point 6  */
    /* BLACK checkers (negative) */
    s->points[0]  = -2;   /* point 1  */
    s->points[11] = -5;   /* point 12 */
    s->points[16] = -3;   /* point 17 */
    s->points[18] = -5;   /* point 19 */
    s->turn = WHITE;
}

int board_is_game_over(const BoardState *s) {
    return s->off[WHITE] == NUM_CHECKERS || s->off[BLACK] == NUM_CHECKERS;
}

int board_winner(const BoardState *s) {
    if (s->off[WHITE] == NUM_CHECKERS) return WHITE;
    if (s->off[BLACK] == NUM_CHECKERS) return BLACK;
    return -1;
}

void board_switch_turn(BoardState *s) {
    s->turn = 1 - s->turn;
}

/* ── Encoding ──────────────────────────────────────────────────── */

void encode_state(const BoardState *s, float *x) {
    int idx, v, n;

    memset(x, 0, NUM_FEATURES * sizeof(float));

    /* Perspective-based encoding: always from on-roll player's view.
     *   [0..95]   MY checkers: 24 points * 4 units
     *   [96]      MY bar / 2
     *   [97]      MY off / 15
     *   [98..193] OPPONENT checkers: 24 points * 4 units
     *   [194]     OPPONENT bar / 2
     *   [195]     OPPONENT off / 15
     *
     * When BLACK is on roll, players are swapped and point indices
     * are mirrored (idx -> 23-idx) so "my home board" is always
     * in the same feature positions.
     */

    if (s->turn == WHITE) {
        /* WHITE on roll: normal encoding */
        for (idx = 0; idx < 24; idx++) {
            v = s->points[idx];
            if (v > 0) {
                int off = idx * 4;
                x[off] = 1.0f;
                if (v >= 2) x[off + 1] = 1.0f;
                if (v >= 3) x[off + 2] = 1.0f;
                if (v >= 4) x[off + 3] = (v - 3) * 0.5f;
            }
            else if (v < 0) {
                n = -v;
                int off = 98 + idx * 4;
                x[off] = 1.0f;
                if (n >= 2) x[off + 1] = 1.0f;
                if (n >= 3) x[off + 2] = 1.0f;
                if (n >= 4) x[off + 3] = (n - 3) * 0.5f;
            }
        }
        x[96]  = s->bar[WHITE] * 0.5f;
        x[97]  = s->off[WHITE] / 15.0f;
        x[194] = s->bar[BLACK] * 0.5f;
        x[195] = s->off[BLACK] / 15.0f;
    } else {
        /* BLACK on roll: swap players, mirror indices */
        for (idx = 0; idx < 24; idx++) {
            v = s->points[idx];
            if (v < 0) {
                /* BLACK's checkers -> MY block, mirrored */
                n = -v;
                int off = (23 - idx) * 4;
                x[off] = 1.0f;
                if (n >= 2) x[off + 1] = 1.0f;
                if (n >= 3) x[off + 2] = 1.0f;
                if (n >= 4) x[off + 3] = (n - 3) * 0.5f;
            }
            else if (v > 0) {
                /* WHITE's checkers -> OPPONENT block, mirrored */
                int off = 98 + (23 - idx) * 4;
                x[off] = 1.0f;
                if (v >= 2) x[off + 1] = 1.0f;
                if (v >= 3) x[off + 2] = 1.0f;
                if (v >= 4) x[off + 3] = (v - 3) * 0.5f;
            }
        }
        x[96]  = s->bar[BLACK] * 0.5f;
        x[97]  = s->off[BLACK] / 15.0f;
        x[194] = s->bar[WHITE] * 0.5f;
        x[195] = s->off[WHITE] / 15.0f;
    }
}


/* ── Internal: single-move generation ──────────────────────────── */

typedef struct {
    Move moves[32];  /* generous: max possible single moves */
    int count;
} MoveList;

static void single_moves(const BoardState *s, int die, MoveList *ml) {
    int player = s->turn;
    int is_white = (player == WHITE);
    int direction = is_white ? -1 : 1;
    int src, target, v, opp_count, d, farthest;
    int can_bear_off;

    ml->count = 0;

    /* Must enter from bar first */
    if (s->bar[player] > 0) {
        target = is_white ? (24 - die) : (die - 1);
        v = s->points[target];
        opp_count = is_white ? (-v) : v;
        if (opp_count < 2) {
            ml->moves[ml->count].src = BAR_SENTINEL;
            ml->moves[ml->count].dst = target;
            ml->count++;
        }
        return;  /* while on bar, no other moves allowed */
    }

    /* Check if all checkers are in home board */
    can_bear_off = 1;
    if (is_white) {
        for (int i = 6; i < 24; i++) {
            if (s->points[i] > 0) { can_bear_off = 0; break; }
        }
    } else {
        for (int i = 0; i < 18; i++) {
            if (s->points[i] < 0) { can_bear_off = 0; break; }
        }
    }

    /* Pre-compute farthest checker in home board (for over-bear) */
    farthest = -1;
    if (can_bear_off) {
        if (is_white) {
            for (int i = 5; i >= 0; i--) {
                if (s->points[i] > 0) { farthest = i; break; }
            }
        } else {
            for (int i = 18; i < 24; i++) {
                if (s->points[i] < 0) { farthest = i; break; }
            }
        }
    }

    /* Iterate all points */
    for (src = 0; src < 24; src++) {
        v = s->points[src];
        if (is_white) {
            if (v <= 0) continue;
        } else {
            if (v >= 0) continue;
        }

        target = src + direction * die;

        /* Normal move (target on the board) */
        if (target >= 0 && target <= 23) {
            int tv = s->points[target];
            opp_count = is_white ? (-tv) : tv;
            if (opp_count < 2) {
                ml->moves[ml->count].src = src;
                ml->moves[ml->count].dst = target;
                ml->count++;
            }
            continue;
        }

        /* Bearing off attempt */
        if (!can_bear_off) continue;

        d = is_white ? (src + 1) : (24 - src);
        if (die == d) {
            /* Exact bear-off */
            ml->moves[ml->count].src = src;
            ml->moves[ml->count].dst = OFF_SENTINEL;
            ml->count++;
        } else if (die > d && src == farthest) {
            /* Over-bear from farthest checker */
            ml->moves[ml->count].src = src;
            ml->moves[ml->count].dst = OFF_SENTINEL;
            ml->count++;
        }
    }
}

/* ── Internal: apply a single move ─────────────────────────────── */

static void apply_move(const BoardState *s, const Move *m, BoardState *out) {
    int player, sign, opp_count;

    memcpy(out, s, sizeof(BoardState));
    player = out->turn;
    sign = (player == WHITE) ? 1 : -1;

    /* Remove checker from source */
    if (m->src == BAR_SENTINEL) {
        out->bar[player]--;
    } else {
        out->points[m->src] -= sign;
    }

    /* Place checker at destination */
    if (m->dst == OFF_SENTINEL) {
        out->off[player]++;
    } else {
        /* Hit? Opponent has exactly 1 checker */
        int tv = out->points[m->dst];
        opp_count = (player == WHITE) ? (-tv) : tv;
        if (opp_count == 1) {
            out->points[m->dst] += sign;  /* remove opponent */
            out->bar[1 - player]++;
        }
        out->points[m->dst] += sign;  /* place ours */
    }
}

/* ── Internal: board state hashing for deduplication ───────────── */

/*
 * Simple hash of a BoardState for dedup.
 * We use a hash table with open addressing.
 */
#define HASH_TABLE_SIZE 4096
#define HASH_MASK       (HASH_TABLE_SIZE - 1)

static unsigned int board_hash(const BoardState *s) {
    /* FNV-1a style hash over the board data */
    const unsigned char *data = (const unsigned char *)s;
    unsigned int h = 2166136261u;
    int nbytes = sizeof(s->points) + sizeof(s->bar) + sizeof(s->off) + sizeof(s->turn);
    for (int i = 0; i < nbytes; i++) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

static int board_equal(const BoardState *a, const BoardState *b) {
    return memcmp(a->points, b->points, sizeof(a->points)) == 0
        && a->bar[0] == b->bar[0] && a->bar[1] == b->bar[1]
        && a->off[0] == b->off[0] && a->off[1] == b->off[1]
        && a->turn == b->turn;
}

/* Hash set for board states (used for deduplication during play generation).
 * Stores full board states to handle hash collisions correctly. */
typedef struct {
    unsigned int hashes[HASH_TABLE_SIZE];
    BoardState   states[HASH_TABLE_SIZE];
    int occupied[HASH_TABLE_SIZE];
    int play_idx[HASH_TABLE_SIZE];  /* index into GenContext.plays, or -1 */
    int count;
} BoardHashSet;

static void hashset_init(BoardHashSet *hs) {
    memset(hs->occupied, 0, sizeof(hs->occupied));
    hs->count = 0;
}

/* Insert (or find) a state. Sets *existed to 1 if it was already present.
 * Returns the slot index, or -1 if the table is full (treat as new). */
static int hashset_insert(BoardHashSet *hs, const BoardState *s, int *existed) {
    unsigned int h = board_hash(s);
    unsigned int idx = h & HASH_MASK;

    *existed = 0;
    /* Linear probing with full state comparison */
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        unsigned int probe = (idx + i) & HASH_MASK;
        if (!hs->occupied[probe]) {
            /* Empty slot: insert */
            hs->hashes[probe] = h;
            memcpy(&hs->states[probe], s, sizeof(BoardState));
            hs->occupied[probe] = 1;
            hs->play_idx[probe] = -1;
            hs->count++;
            return (int)probe;  /* newly inserted */
        }
        if (hs->hashes[probe] == h && board_equal(&hs->states[probe], s)) {
            *existed = 1;
            return (int)probe;  /* exact duplicate */
        }
    }
    return -1;  /* table full: no slot; caller records the play anyway */
}

/* ── Internal: recursive play generation ───────────────────────── */

typedef struct {
    Play *plays;           /* output buffer */
    int count;             /* number of plays found */
    int max_plays;         /* buffer capacity */
    int max_dice_used;     /* best dice count seen */
    BoardHashSet *seen;    /* deduplication */
} GenContext;

static void generate_plays(
    const BoardState *state,
    const int *remaining_dice,
    int num_remaining,
    Move *current_moves,
    int num_current_moves,
    int dice_used,
    GenContext *ctx
) {
    /* Invariants: callers always pass num_current_moves + num_remaining
     * ≤ initial dice count (≤4 for doubles, 2 otherwise), so neither
     * exceeds MAX_MOVES_PER_PLAY. Asserting loudly is much safer than
     * a silent early-return guard — see commit 72a964b for the bug
     * that pattern caused (dropped every full-depth doubles play). */
    assert(num_current_moves >= 0);
    assert(num_remaining >= 0);
    assert(num_current_moves <= MAX_MOVES_PER_PLAY);
    assert(num_current_moves + num_remaining <= MAX_MOVES_PER_PLAY);

    int found = 0;
    int seen_die[7] = {0};  /* track which die values tried at this level */
    MoveList ml;

    for (int i = 0; i < num_remaining; i++) {
        int die = remaining_dice[i];
        if (seen_die[die]) continue;
        seen_die[die] = 1;

        /* Build new remaining_dice without this die */
        int new_remaining[4];
        int new_num = 0;
        for (int j = 0; j < num_remaining; j++) {
            if (j != i) new_remaining[new_num++] = remaining_dice[j];
        }

        single_moves(state, die, &ml);

        for (int m = 0; m < ml.count; m++) {
            found = 1;
            BoardState new_state;
            apply_move(state, &ml.moves[m], &new_state);

            assert(num_current_moves < MAX_MOVES_PER_PLAY);
            current_moves[num_current_moves] = ml.moves[m];

            generate_plays(
                &new_state, new_remaining, new_num,
                current_moves, num_current_moves + 1,
                dice_used + 1, ctx
            );
        }
    }

    if (!found) {
        /* Terminal node */
        if (dice_used > ctx->max_dice_used) {
            ctx->max_dice_used = dice_used;
        }

        /* Deduplicate by board state during recursion (like Python).
         * On a collision keep the play using MORE dice — see
         * backgammon_engine._generate_plays for why (false forced pass). */
        int existed = 0;
        int slot = hashset_insert(ctx->seen, state, &existed);
        if (!existed) {
            /* New unique state: record this play */
            if (ctx->count < ctx->max_plays) {
                Play *p = &ctx->plays[ctx->count];
                p->num_moves = num_current_moves;
                for (int i = 0; i < num_current_moves; i++) {
                    p->moves[i] = current_moves[i];
                }
                memcpy(&p->resulting_state, state, sizeof(BoardState));
                if (slot >= 0) {
                    ctx->seen->play_idx[slot] = ctx->count;
                }
                ctx->count++;
            }
        } else if (slot >= 0) {
            int pi = ctx->seen->play_idx[slot];
            /* pi may be stale if max_plays was hit: state seen, play not stored */
            if (pi >= 0 && pi < ctx->count
                && num_current_moves > ctx->plays[pi].num_moves) {
                Play *p = &ctx->plays[pi];
                p->num_moves = num_current_moves;
                for (int i = 0; i < num_current_moves; i++) {
                    p->moves[i] = current_moves[i];
                }
                /* resulting_state is identical by definition */
            }
        }
    }
}

/* ── Internal: check if a play uses a specific die value ───────── */

static int play_uses_die(const Play *play, const BoardState *original, int die) {
    int src, dst, player, direction, d;
    if (play->num_moves != 1) return 0;

    src = play->moves[0].src;
    dst = play->moves[0].dst;
    player = original->turn;
    direction = (player == WHITE) ? -1 : 1;

    if (src == BAR_SENTINEL) {
        int expected = (player == WHITE) ? (24 - die) : (die - 1);
        return dst == expected;
    }
    if (dst == OFF_SENTINEL) {
        d = (player == WHITE) ? (src + 1) : (24 - src);
        if (die == d) return 1;  /* exact bear-off */
        if (die > d) {
            /* Over-bear: only valid from the farthest occupied point */
            int i, farthest = -1;
            if (player == WHITE) {
                for (i = 5; i >= 0; i--)
                    if (original->points[i] > 0) { farthest = i; break; }
            } else {
                for (i = 18; i < 24; i++)
                    if (original->points[i] < 0) { farthest = i; break; }
            }
            return src == farthest;
        }
        return 0;
    }
    return dst == src + direction * die;
}

/* ── Public: get_legal_plays ───────────────────────────────────── *
 *
 * NOTE (v2 wart-free convention): unlike the v1 c_engine, this
 * version switches the turn on every resulting_state before
 * returning. After this call, plays[i].resulting_state.turn is the
 * OPPONENT of state.turn — i.e. "the next player to act," which
 * preserves the BoardState invariant that .turn always means "the
 * player whose turn it is to act next".
 */

int get_legal_plays(const BoardState *state, int d1, int d2,
                    Play *plays, int max_plays) {
    int remaining_dice[4];
    int num_remaining;
    Move current_moves[MAX_MOVES_PER_PLAY];
    GenContext ctx;
    BoardHashSet seen;
    int i, j, out_count;

    if (d1 == d2) {
        remaining_dice[0] = remaining_dice[1] = remaining_dice[2] = remaining_dice[3] = d1;
        num_remaining = 4;
    } else {
        remaining_dice[0] = d1;
        remaining_dice[1] = d2;
        num_remaining = 2;
    }

    /* First pass: collect all terminal plays */
    ctx.plays = plays;
    ctx.count = 0;
    ctx.max_plays = max_plays;
    ctx.max_dice_used = 0;
    ctx.seen = &seen;
    hashset_init(&seen);

    generate_plays(state, remaining_dice, num_remaining,
                   current_moves, 0, 0, &ctx);

    /* No moves possible */
    if (ctx.max_dice_used == 0) {
        return 0;
    }

    /* Filter: keep only plays using max dice.
     * Dedup already happened during recursion via the hash set. */
    out_count = 0;
    for (i = 0; i < ctx.count; i++) {
        if (plays[i].num_moves == ctx.max_dice_used) {
            if (out_count != i) {
                plays[out_count] = plays[i];
            }
            out_count++;
        }
    }

    /* Tie-break: if non-doubles and only one die usable, must use larger */
    if (d1 != d2 && ctx.max_dice_used == 1) {
        int big = (d1 > d2) ? d1 : d2;
        int big_count = 0;

        /* Count how many use the bigger die */
        for (i = 0; i < out_count; i++) {
            if (play_uses_die(&plays[i], state, big)) {
                big_count++;
            }
        }

        if (big_count > 0) {
            /* Filter to only those using the bigger die */
            j = 0;
            for (i = 0; i < out_count; i++) {
                if (play_uses_die(&plays[i], state, big)) {
                    if (j != i) plays[j] = plays[i];
                    j++;
                }
            }
            out_count = j;
        }
    }

    /* v2 wart-free convention: switch the turn on every resulting
     * state before returning. After this loop the on-roll player at
     * resulting_state[i] is the OPPONENT of the original mover. */
    for (i = 0; i < out_count; i++) {
        plays[i].resulting_state.turn = 1 - plays[i].resulting_state.turn;
    }

    return out_count;
}

/* ── Combined get-plays + encode ───────────────────────────────── */

int get_legal_plays_encoded(const BoardState *state, int d1, int d2,
                            Play *plays, int max_plays,
                            float *encoded_features) {
    int count = get_legal_plays(state, d1, d2, plays, max_plays);

    /* v2: get_legal_plays already switched the turn on resulting_state,
     * so we can encode each one directly — no temporary switch needed. */
    for (int i = 0; i < count; i++) {
        encode_state(&plays[i].resulting_state,
                     encoded_features + i * NUM_FEATURES);
    }

    return count;
}

void get_play_resulting_state(const Play *plays, int index,
                              BoardState *out_state) {
    memcpy(out_state, &plays[index].resulting_state, sizeof(BoardState));
}

/* ── Game result (1/2/3) for a terminal state, winner's POV ─────────
 * Mirrors BoardState.game_result in backgammon_engine.py exactly. */
static int game_result_c(const BoardState *s) {
    int w = (s->off[WHITE] == NUM_CHECKERS) ? WHITE
          : (s->off[BLACK] == NUM_CHECKERS) ? BLACK : -1;
    if (w < 0) return 0;
    int loser = 1 - w;
    if (s->off[loser] > 0) return 1;               /* normal */
    int loser_in_opp_home = 0;
    if (loser == WHITE) {
        for (int i = 18; i < 24; i++)
            if (s->points[i] > 0) { loser_in_opp_home = 1; break; }
    } else {
        for (int i = 0; i < 6; i++)
            if (s->points[i] < 0) { loser_in_opp_home = 1; break; }
    }
    if (s->bar[loser] > 0 || loser_in_opp_home) return 3;  /* backgammon */
    return 2;                                              /* gammon */
}

/*
 * One-shot 1-ply inner expansion for a single candidate's resulting
 * state (opponent on roll). Enumerates all 21 distinct opponent dice
 * outcomes (d1=1..6, d2=d1..6 — same order as _DICE_OUTCOMES in
 * td_agent.py); for each, generates the opponent's legal replies and
 * writes:
 *   - out_feats:        196-float base encoding of every NON-TERMINAL
 *                       reply (and the back-to-us state on a forced
 *                       pass), packed row-major in reply order.
 *   - out_dice_counts:  [21] number of replies per dice bucket
 *                       (>=1 always: a forced pass counts as 1).
 *   - out_reply_gr:     per reply, 0 if non-terminal, else the winner's
 *                       game_result (1/2/3) — a terminal reply writes
 *                       NO row to out_feats.
 *
 * Reply ordering is dice-major; within a dice, opponent-play order.
 * Non-terminal replies consume rows of out_feats sequentially, so the
 * caller maps the k-th non-terminal reply to row k.
 *
 * Returns the total number of replies, or -1 on buffer overflow.
 */
int get_legal_plays_encoded_21(const BoardState *next_state,
                               float *out_feats,
                               int *out_dice_counts,
                               int *out_reply_gr,
                               int max_rows,
                               int max_replies) {
    static Play splays[MAX_PLAYS];
    int row = 0, rep = 0, di = 0;
    for (int d1 = 1; d1 <= 6; d1++) {
        for (int d2 = d1; d2 <= 6; d2++) {
            int cnt = get_legal_plays(next_state, d1, d2, splays, MAX_PLAYS);
            if (cnt == 0) {
                /* Forced pass — opponent can't move, play returns to us. */
                if (row >= max_rows || rep >= max_replies) return -1;
                BoardState back = *next_state;
                board_switch_turn(&back);
                encode_state(&back, out_feats + (size_t)row * NUM_FEATURES_196);
                out_reply_gr[rep] = 0;
                row++; rep++;
                out_dice_counts[di] = 1;
            } else {
                for (int i = 0; i < cnt; i++) {
                    if (rep >= max_replies) return -1;
                    const BoardState *rs = &splays[i].resulting_state;
                    if (board_is_game_over(rs)) {
                        out_reply_gr[rep] = game_result_c(rs);
                    } else {
                        if (row >= max_rows) return -1;
                        encode_state(rs,
                            out_feats + (size_t)row * NUM_FEATURES_196);
                        out_reply_gr[rep] = 0;
                        row++;
                    }
                    rep++;
                }
                out_dice_counts[di] = cnt;
            }
            di++;
        }
    }
    return rep;
}
