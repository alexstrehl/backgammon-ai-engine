/*
 * bg_engine.c -- Backgammon engine: board state, move generation, encoding.
 *
 * C port of backgammon_engine.py and encoding.py. Compiled as a shared
 * library and called via ctypes (bg_fast.py).
 */

#include "bg_engine.h"
#include <stdint.h>
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

    /* Perspective encoding, from the on-roll player's view:
     *   [0..95]   MY checkers: 24 points * 4 units
     *   [96]      MY bar / 2
     *   [97]      MY off / 15
     *   [98..193] OPPONENT checkers: 24 points * 4 units
     *   [194]     OPPONENT bar / 2
     *   [195]     OPPONENT off / 15
     * When BLACK is on roll, players are swapped and points mirrored
     * (idx -> 23-idx) so "my home board" is always in the same slots.
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
    Move moves[32];  /* upper bound on single moves for one die */
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
        return;  /* on the bar, nothing else may move */
    }

    /* Bear-off allowed only with every checker in the home board */
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

    /* Farthest checker from off (over-bear is allowed only from it) */
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

        /* Off the board: bear-off */
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

/* Open-addressing hash set of BoardStates. */
#define HASH_TABLE_SIZE 4096
#define HASH_MASK       (HASH_TABLE_SIZE - 1)

static unsigned int board_hash(const BoardState *s) {
    /* Every field fits in a signed byte: pack the board into 4 words. */
    unsigned char b[32];
    for (int i = 0; i < NUM_POINTS; i++) b[i] = (unsigned char)s->points[i];
    b[24] = (unsigned char)s->bar[0];
    b[25] = (unsigned char)s->bar[1];
    b[26] = (unsigned char)s->off[0];
    b[27] = (unsigned char)s->off[1];
    b[28] = (unsigned char)s->turn;
    b[29] = b[30] = b[31] = 0;
    uint64_t k[4];
    memcpy(k, b, sizeof(k));
    uint64_t h = k[0] * 0x9E3779B97F4A7C15ULL ^ k[1] * 0xC2B2AE3D27D4EB4FULL
               ^ k[2] * 0x165667B19E3779F9ULL ^ k[3] * 0x27D4EB2F165667C5ULL;
    h ^= h >> 32;   /* the low half now depends on every input byte */
    return (unsigned int)((h * 0x9E3779B97F4A7C15ULL) >> 32);
}

static int board_equal(const BoardState *a, const BoardState *b) {
    return memcmp(a->points, b->points, sizeof(a->points)) == 0
        && a->bar[0] == b->bar[0] && a->bar[1] == b->bar[1]
        && a->off[0] == b->off[0] && a->off[1] == b->off[1]
        && a->turn == b->turn;
}

/* Stores full states, so a hash collision still compares exactly.
 * hashset_init bumps gen instead of clearing occupied[]. */
typedef struct {
    unsigned int hashes[HASH_TABLE_SIZE];
    BoardState   states[HASH_TABLE_SIZE];
    unsigned int occupied[HASH_TABLE_SIZE];  /* slot in use iff == gen */
    unsigned int gen;
    int play_idx[HASH_TABLE_SIZE];  /* index into GenContext.plays, or -1 */
    int count;
} BoardHashSet;

/* get_legal_plays' table: thread-local, since a 512 KB stack frame
 * costs more than the walk. */
static _Thread_local BoardHashSet movegen_seen;

#ifdef BG_TEST_HOOKS
/* Test builds only: set the generation (to exercise its wraparound). */
void bg_test_set_dedup_gen(unsigned int gen) { movegen_seen.gen = gen; }
#endif

static void hashset_init(BoardHashSet *hs) {
    if (++hs->gen == 0) {
        memset(hs->occupied, 0, sizeof(hs->occupied));
        hs->gen = 1;
    }
    hs->count = 0;
}

/* Insert or find a state; *existed = 1 if already present. Returns the
 * slot, or -1 if the table is full (treat as new). */
static int hashset_insert(BoardHashSet *hs, const BoardState *s, int *existed) {
    unsigned int h = board_hash(s);
    unsigned int idx = h & HASH_MASK;

    *existed = 0;
    /* Linear probing */
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        unsigned int probe = (idx + i) & HASH_MASK;
        if (hs->occupied[probe] != hs->gen) {
            hs->hashes[probe] = h;
            memcpy(&hs->states[probe], s, sizeof(BoardState));
            hs->occupied[probe] = hs->gen;
            hs->play_idx[probe] = -1;
            hs->count++;
            return (int)probe;
        }
        if (hs->hashes[probe] == h && board_equal(&hs->states[probe], s)) {
            *existed = 1;
            return (int)probe;
        }
    }
    return -1;  /* table full */
}

/* ── Internal: recursive play generation ───────────────────────── *
 *
 * The walk is a DFS over move sequences: at each node the dice are
 * tried in remaining_dice order, and for each die the moves in
 * single_moves order (ascending src; a bar entry is the only move while
 * on the bar). Plays are emitted at the leaves, deduplicated by board,
 * in order of FIRST DISCOVERY. That order is part of the contract
 * (callers break exact ties by the first candidate), so the walk below
 * must not change it.
 *
 * Doubles: the orderings of the same four moves reach the same board,
 * so the full walk visits ~10x more leaves than unique plays (fixed
 * pool: 2477 leaves for 241 plays per position over the 6 doubles).
 * Pruning keeps one ordering of each move multiset: the one the full
 * walk visits FIRST (lexicographically smallest src sequence among the
 * legal orderings). A sequence is pruned at move m_j when, for some
 * earlier level L, src(m_j) < src(m_L) and m_j was already legal at
 * level L's state. The mask (below) forbids src(m_j) for the WHOLE
 * subtree under level L, so by construction the pruned m_j is the FIRST
 * departure from its src below L: none of m_L..m_{j-1} departs src(m_j)
 * (src(m_L) is larger; any later one would itself have been pruned).
 * Playing m_j at position L instead is then legal: m_j was legal at L's
 * state, and each of m_L..m_{j-1} stays legal with m_j played before it,
 * since nothing it needs is taken away (m_j departs a src none of them
 * departs; arrivals only add checkers, bringing a checker home only
 * enables bear-off, the farthest checker only gets nearer, opponent
 * points only lose blots; with one die, a src names one move). That
 * sequence reaches the same board and is lexicographically smaller. So
 * no pruned sequence is the first discovery of its board, and the
 * pruned walk visits a subset of the full walk's leaves, in the same
 * order, that contains every board's first discovery: the play list
 * (boards, order, num_moves, the stored moves) is identical to the full
 * walk's (tests/test_movegen_order_stable.py).
 *
 * Non-doubles (d1, d2 in the order passed; the full walk tries d1 at
 * the root first): the sequence d2:y, d1:x is pruned when d1:x was
 * already legal at the root. Then d1:x, d2:y is legal: if x and y
 * leave different points, by the stays-legal reasoning above; if both
 * leave the same point b, then since x is still legal after y and
 * nothing lands on b in between, b held >= 2 checkers at the root, so
 * one is left for y after x. It reaches the same board and comes
 * first. So the d2-first subtree only keeps the x that were not legal
 * at the root (y made them possible), and the same identity argument
 * holds.
 *
 * `forbid` carries the pruned set as a mask of srcs (bit src+1, bar =
 * bit 0; with one die, a src names one move): doubles, every src legal
 * at an ancestor node and smaller than the src the ancestor moved;
 * non-doubles, below the root's d2 moves, every src with a legal d1 move
 * at the root. A node whose legal moves are all forbidden is NOT a
 * leaf (it has legal moves; its plays are generated elsewhere), so
 * max_dice_used and the dedup see exactly the full walk's leaves.
 */

typedef struct {
    Play *plays;           /* output buffer */
    int count;             /* number of plays found */
    int max_plays;         /* buffer capacity */
    int max_dice_used;     /* best dice count seen */
    BoardHashSet *seen;    /* deduplication */
    int doubles;           /* d1 == d2: selects the pruning rule (above) */
} GenContext;

static void generate_plays(
    const BoardState *state,
    const int *remaining_dice,
    int num_remaining,
    Move *current_moves,
    int num_current_moves,
    int dice_used,
    GenContext *ctx,
    unsigned int forbid    /* srcs (bit src+1) not to move here; see above */
) {
    /* num_current_moves + num_remaining <= the initial dice count, so
     * neither exceeds MAX_MOVES_PER_PLAY. Assert, never return silently. */
    assert(num_current_moves >= 0);
    assert(num_remaining >= 0);
    assert(num_current_moves <= MAX_MOVES_PER_PLAY);
    assert(num_current_moves + num_remaining <= MAX_MOVES_PER_PLAY);

    int found = 0;
    int seen_die[7] = {0};  /* die values already tried at this level */
    unsigned int first_die_srcs = 0;  /* srcs with a legal remaining_dice[0] move */
    MoveList ml;

    for (int i = 0; i < num_remaining; i++) {
        int die = remaining_dice[i];
        if (seen_die[die]) continue;
        seen_die[die] = 1;

        /* remaining_dice without this die */
        int new_remaining[4];
        int new_num = 0;
        for (int j = 0; j < num_remaining; j++) {
            if (j != i) new_remaining[new_num++] = remaining_dice[j];
        }

        single_moves(state, die, &ml);
        if (ml.count > 0) found = 1;   /* legal moves: never a leaf */

        unsigned int listed = 0;   /* srcs listed before move m (ascending) */
        for (int m = 0; m < ml.count; m++) {
            unsigned int bit = 1u << (ml.moves[m].src + 1);
            unsigned int below = listed;
            listed |= bit;
            if (forbid & bit) continue;   /* a smaller sequence plays it earlier */
            /* doubles: srcs legal here below this one; non-doubles: the
             * second die's subtree forbids the first die's root moves */
            unsigned int child_forbid = ctx->doubles ? (forbid | below)
                                      : (i == 0 ? 0u : first_die_srcs);

            BoardState new_state;
            apply_move(state, &ml.moves[m], &new_state);

            assert(num_current_moves < MAX_MOVES_PER_PLAY);
            current_moves[num_current_moves] = ml.moves[m];

            generate_plays(
                &new_state, new_remaining, new_num,
                current_moves, num_current_moves + 1,
                dice_used + 1, ctx, child_forbid
            );
        }
        if (i == 0) first_die_srcs = listed;
    }

    if (!found) {
        /* Leaf: no die can be played from here */
        if (dice_used > ctx->max_dice_used) {
            ctx->max_dice_used = dice_used;
        }

        /* Dedup by resulting board. On a duplicate keep the play using
         * MORE dice. See backgammon_engine._generate_plays (false forced
         * pass). */
        int existed = 0;
        int slot = hashset_insert(ctx->seen, state, &existed);
        if (!existed) {
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
            /* pi is -1 if max_plays was hit: state seen, play not stored */
            if (pi >= 0 && pi < ctx->count
                && num_current_moves > ctx->plays[pi].num_moves) {
                Play *p = &ctx->plays[pi];
                p->num_moves = num_current_moves;
                for (int i = 0; i < num_current_moves; i++) {
                    p->moves[i] = current_moves[i];
                }
                /* resulting_state is the same board */
            }
        }
    }
}

/* ── Internal: does a single-move play use this die value? ─────── */

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
            /* Over-bear: only from the farthest occupied point */
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
 * Turn convention: every resulting_state has its turn switched, so
 * plays[i].resulting_state.turn is the OPPONENT of state.turn (the
 * next player to act).
 */

int get_legal_plays(const BoardState *state, int d1, int d2,
                    Play *plays, int max_plays) {
    int remaining_dice[4];
    int num_remaining;
    Move current_moves[MAX_MOVES_PER_PLAY];
    GenContext ctx;
    int i, j, out_count;

    if (d1 == d2) {
        remaining_dice[0] = remaining_dice[1] = remaining_dice[2] = remaining_dice[3] = d1;
        num_remaining = 4;
    } else {
        remaining_dice[0] = d1;
        remaining_dice[1] = d2;
        num_remaining = 2;
    }

    /* Collect every maximal play */
    ctx.plays = plays;
    ctx.count = 0;
    ctx.max_plays = max_plays;
    ctx.max_dice_used = 0;
    ctx.seen = &movegen_seen;
    ctx.doubles = (d1 == d2);
    hashset_init(&movegen_seen);

    generate_plays(state, remaining_dice, num_remaining,
                   current_moves, 0, 0, &ctx, 0u);
    assert(movegen_seen.count < HASH_TABLE_SIZE);  /* the table never fills: max seen 1248 leaves */

    if (ctx.max_dice_used == 0) {
        return 0;
    }

    /* Keep only plays using the max number of dice (already deduped) */
    out_count = 0;
    for (i = 0; i < ctx.count; i++) {
        if (plays[i].num_moves == ctx.max_dice_used) {
            if (out_count != i) {
                plays[out_count] = plays[i];
            }
            out_count++;
        }
    }

    /* Non-doubles with only one die playable: must use the larger */
    if (d1 != d2 && ctx.max_dice_used == 1) {
        int big = (d1 > d2) ? d1 : d2;
        int big_count = 0;

        for (i = 0; i < out_count; i++) {
            if (play_uses_die(&plays[i], state, big)) {
                big_count++;
            }
        }

        if (big_count > 0) {
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

    /* Turn convention: the opponent is on roll in every resulting state */
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

    /* resulting_state.turn is already switched: encode directly */
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

/* ── Game result (1/2/3) of a terminal state, winner's view ─────────
 * Mirrors BoardState.game_result in backgammon_engine.py. */
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
 * 1-ply inner expansion of a candidate resulting state (opponent on
 * roll). For each of the 21 opponent dice (d1=1..6, d2=d1..6, the order
 * of _DICE_OUTCOMES in td_agent.py) generate the opponent's replies and
 * write:
 *   - out_feats:        196-float encoding of every NON-TERMINAL reply
 *                       (a forced pass = the back-to-us state), packed
 *                       row-major in reply order.
 *   - out_dice_counts:  [21] replies per dice (>=1: a forced pass is 1).
 *   - out_reply_gr:     per reply, 0 if non-terminal, else the winner's
 *                       game_result 1/2/3; a terminal reply has NO row.
 *
 * Replies are dice-major, then play order. Non-terminal replies take
 * rows of out_feats in sequence: the k-th non-terminal reply is row k.
 *
 * Returns the total reply count, or -1 on buffer overflow.
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
                /* Forced pass: play returns to us */
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


/* Rules from GNU Backgammon 1.08 SanityCheck (eval.c); independent
   reimplementation. Same code as nn_eval.c prob5_sanity_zero; parity:
   tests/test_prob5_sanity.py TestCCopiesParity. */
int prob5_impossible_mask(const float *x, int n, unsigned char *mask) {
    int any = 0;
    for (int r = 0; r < n; r++, x += NUM_FEATURES, mask += 4) {
        /* s = 0: my side (gnubg side 1), 1: the opponent (gnubg side 0) */
        int ac[2], back[2], cross[2], gcross[2];
        for (int s = 0; s < 2; s++) {
            const float *b = x + 98 * s;
            ac[s] = cross[s] = back[s] = 0;
            gcross[s] = 1;
            for (int k = 0; k < 24; k++) {
                const float *f = b + 4 * k;
                if (f[0] == 0.0f) continue;
                int c = (int)(f[0] + f[1] + f[2] + 2.0f * f[3] + 0.5f);
                int i = s ? 23 - k : k;          /* gnubg point index */
                if (i > back[s]) back[s] = i;
                ac[s] += c;
                cross[s] += (1 + i / 6) * c;
                gcross[s] += (i / 6) * c;
            }
            int bar = (int)(2.0f * b[96] + 0.5f);
            if (bar) {
                back[s] = 24; ac[s] += bar; cross[s] += 5 * bar; gcross[s] += 4 * bar;
            }
        }
        int race = back[0] + back[1] < 24;
        mask[0] = ac[1] < 15 || (race && cross[0] > 8 * gcross[1]);
        mask[1] = ac[1] < 15 || (race && back[1] < 18);
        mask[2] = ac[0] < 15 || (race && cross[1] > 8 * gcross[0] - 4);
        mask[3] = ac[0] < 15 || (race && back[0] < 18);
        any += mask[0] | mask[1] | mask[2] | mask[3];
    }
    return any;
}
