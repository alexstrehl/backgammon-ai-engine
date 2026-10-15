/*
 * puretd_movegen.c -- bgc_i_legal_plays: c_engine/bg_engine.c's
 * get_legal_plays with the same walk and a cheaper leaf dedup (the fast move
 * generator, BGC_FAST_MOVEGEN).
 *
 * The walk (generate_plays: the DFS over dice and single_moves order, the
 * `forbid` pruning, leaves in first-discovery order, the MORE-dice update on
 * a duplicate) and the post-pass (max dice, the larger-die rule, the turn
 * flip) are bg_engine.c's, copied verbatim; single_moves and apply_move too.
 * Only the leaf dedup differs:
 *
 *   bg_engine.c   a BoardHashSet on the stack per call (4096 x (state, hash,
 *                 occupied, play_idx), 520 KB; 16 KB of `occupied` cleared
 *                 per call), FNV-1a over the state's 116 bytes per leaf
 *   here          an engine-owned table of BGC_MG_SLOTS (8192) slots
 *                 {generation, 32-bit hash, play index}: no clearing (a
 *                 generation per call), a multiply-add hash of the 29 ints,
 *                 equal hashes compared in full against the stored play's
 *                 state (the leaf state before the turn flip, as the
 *                 BoardHashSet stored it)
 *
 * The play list (boards, order, num_moves, stored moves) is therefore the
 * same: dedup is exact either way, and the table "full" condition is the
 * same (bg_engine.c: 4096 distinct boards = HASH_TABLE_SIZE = MAX_PLAYS, a
 * new board after that is dropped; here a new board is dropped once
 * MAX_PLAYS are stored, at load <= 1/2). Only max_plays == MAX_PLAYS is
 * served (other callers: get_legal_plays itself). Checked against
 * get_legal_plays on random and self-play boards, all 36 ordered rolls
 * (tests/test_c_match.py::test_fast_movegen_identical, bgc_test_legal_plays).
 */

#include "puretd.h"
#include "puretd_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── verbatim from bg_engine.c ───────────────────────────────────── */

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

/* ── the leaf dedup ──────────────────────────────────────────────── */

_Static_assert(sizeof(BoardState) == 29 * sizeof(int), "BoardState: 29 ints, no padding");

/* Odd 64-bit multipliers, one per int of the state (points, bar, off, turn). */
static const uint64_t MG_K[29] = {
    0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL, 0x27D4EB2F165667C5ULL,
    0xFF51AFD7ED558CCDULL, 0xC4CEB9FE1A85EC53ULL, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL,
    0xD6E8FEB86659FD93ULL, 0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL, 0x8EBC6AF09C88C6E3ULL,
    0x589965CC75374CC3ULL, 0x1D8E4E27C47D124FULL, 0xEB44ACCAB455D165ULL, 0x9FB21C651E98DF25ULL,
    0xC6A4A7935BD1E995ULL, 0x87C37B91114253D5ULL, 0x4CF5AD432745937FULL, 0xDB4F0B9175AE2165ULL,
    0x5851F42D4C957F2DULL, 0x14057B7EF767814FULL, 0x2545F4914F6CDD1DULL, 0x9E6C63D0676A9A99ULL,
    0xD1B54A32D192ED03ULL, 0xAEF17502108EF2D9ULL, 0x8CB92BA72F3D8DD7ULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL,
};

static inline uint32_t mg_hash(const BoardState *s) {
    const int *w = (const int *)s;
    uint64_t h = 0;
    for (int i = 0; i < 29; i++) h += (uint64_t)(uint32_t)w[i] * MG_K[i];
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 29;
    return (uint32_t)h;
}

struct bgc_mgtab {
    uint32_t cur;
    uint32_t gen[BGC_MG_SLOTS];
    uint32_t h[BGC_MG_SLOTS];
    int idx[BGC_MG_SLOTS];
};

struct bgc_mgtab *bgc_i_mgtab_new(void) { return calloc(1, sizeof(struct bgc_mgtab)); }

typedef struct {
    Play *plays;
    int count;
    int max_plays;
    int max_dice_used;
    struct bgc_mgtab *t;
    int doubles;
} GenContext;

/* bg_engine.c's leaf: dedup by resulting board, on a duplicate keep the play
   using MORE dice. */
static void leaf(const BoardState *state, const Move *current_moves, int num_current_moves,
                 GenContext *ctx) {
    struct bgc_mgtab *t = ctx->t;
    const uint32_t hh = mg_hash(state);
    uint32_t p = hh & (BGC_MG_SLOTS - 1);
    for (;; p = (p + 1) & (BGC_MG_SLOTS - 1)) {
        if (t->gen[p] != t->cur) break;                    /* free: a new board */
        if (t->h[p] == hh) {
            Play *q = &ctx->plays[t->idx[p]];
            if (!memcmp(&q->resulting_state, state, sizeof(BoardState))) {
                if (num_current_moves > q->num_moves) {
                    q->num_moves = num_current_moves;
                    for (int i = 0; i < num_current_moves; i++) q->moves[i] = current_moves[i];
                }
                return;
            }
        }
    }
    if (ctx->count >= ctx->max_plays) return;              /* bg_engine.c: table full */
    t->gen[p] = t->cur;
    t->h[p] = hh;
    t->idx[p] = ctx->count;
    Play *q = &ctx->plays[ctx->count++];
    q->num_moves = num_current_moves;
    for (int i = 0; i < num_current_moves; i++) q->moves[i] = current_moves[i];
    memcpy(&q->resulting_state, state, sizeof(BoardState));
}

/* bg_engine.c's generate_plays (the pruning argument is there), with the
   leaf above. */
static void generate_plays(const BoardState *state, const int *remaining_dice,
                           int num_remaining, Move *current_moves, int num_current_moves,
                           int dice_used, GenContext *ctx, unsigned int forbid) {
    int found = 0;
    int seen_die[7] = {0};
    unsigned int first_die_srcs = 0;
    MoveList ml;

    for (int i = 0; i < num_remaining; i++) {
        int die = remaining_dice[i];
        if (seen_die[die]) continue;
        seen_die[die] = 1;

        int new_remaining[4];
        int new_num = 0;
        for (int j = 0; j < num_remaining; j++) {
            if (j != i) new_remaining[new_num++] = remaining_dice[j];
        }

        single_moves(state, die, &ml);
        if (ml.count > 0) found = 1;

        unsigned int listed = 0;
        for (int m = 0; m < ml.count; m++) {
            unsigned int bit = 1u << (ml.moves[m].src + 1);
            unsigned int below = listed;
            listed |= bit;
            if (forbid & bit) continue;
            unsigned int child_forbid = ctx->doubles ? (forbid | below)
                                      : (i == 0 ? 0u : first_die_srcs);

            BoardState new_state;
            apply_move(state, &ml.moves[m], &new_state);
            current_moves[num_current_moves] = ml.moves[m];

            generate_plays(&new_state, new_remaining, new_num, current_moves,
                           num_current_moves + 1, dice_used + 1, ctx, child_forbid);
        }
        if (i == 0) first_die_srcs = listed;
    }

    if (!found) {
        if (dice_used > ctx->max_dice_used) ctx->max_dice_used = dice_used;
        leaf(state, current_moves, num_current_moves, ctx);
    }
}

int bgc_i_legal_plays(struct bgc_mgtab *t, const BoardState *state, int d1, int d2,
                      Play *plays, int max_plays) {
    if (!t || max_plays != MAX_PLAYS || d1 < 1 || d1 > 6 || d2 < 1 || d2 > 6)
        return get_legal_plays(state, d1, d2, plays, max_plays);
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
    if (++t->cur == 0) {                         /* generation wrapped */
        memset(t->gen, 0, sizeof t->gen);
        t->cur = 1;
    }
    ctx.plays = plays;
    ctx.count = 0;
    ctx.max_plays = max_plays;
    ctx.max_dice_used = 0;
    ctx.t = t;
    ctx.doubles = (d1 == d2);

    generate_plays(state, remaining_dice, num_remaining, current_moves, 0, 0, &ctx, 0u);

    if (ctx.max_dice_used == 0) return 0;

    out_count = 0;
    for (i = 0; i < ctx.count; i++) {
        if (plays[i].num_moves == ctx.max_dice_used) {
            if (out_count != i) plays[out_count] = plays[i];
            out_count++;
        }
    }

    if (d1 != d2 && ctx.max_dice_used == 1) {
        int big = (d1 > d2) ? d1 : d2;
        int big_count = 0;
        for (i = 0; i < out_count; i++)
            if (play_uses_die(&plays[i], state, big)) big_count++;
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

    for (i = 0; i < out_count; i++)
        plays[i].resulting_state.turn = 1 - plays[i].resulting_state.turn;
    return out_count;
}

/* Test hook (puretd.h): bgc_i_legal_plays on a private table, or
   get_legal_plays (fast 0), into plays[MAX_PLAYS]. */
int bgc_test_legal_plays(const BoardState *s, int d1, int d2, int fast, Play *plays) {
    static struct bgc_mgtab *t;
    if (!s || !plays) return -1;
    if (!fast) return get_legal_plays(s, d1, d2, plays, MAX_PLAYS);
    if (!t && !(t = bgc_i_mgtab_new())) return -1;
    return bgc_i_legal_plays(t, s, d1, d2, plays, MAX_PLAYS);
}
