/*
 * bg_engine.c -- Backgammon engine: board state, move generation, encoding.
 *
 * Port of backgammon_engine.py and encoding.py to C for performance.
 * Designed to be compiled as a shared library and called via ctypes.
 */

#include "bg_engine.h"
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

/* ── gnubg Group-1 expert features (7 per side) ──────────────── */

static void perspective_arrays(const BoardState *s,
                               int *my_pts, int *opp_pts,
                               int *my_bar, int *my_off,
                               int *opp_bar, int *opp_off) {
    int i, v;
    for (i = 0; i < 24; i++) { my_pts[i] = 0; opp_pts[i] = 0; }

    if (s->turn == WHITE) {
        for (i = 0; i < 24; i++) {
            v = s->points[i];
            if (v > 0) my_pts[i] = v;
            else if (v < 0) opp_pts[i] = -v;
        }
        *my_bar  = s->bar[WHITE]; *my_off  = s->off[WHITE];
        *opp_bar = s->bar[BLACK]; *opp_off = s->off[BLACK];
    } else {
        for (i = 0; i < 24; i++) {
            v = s->points[i];
            if (v < 0) my_pts[23 - i] = -v;
            else if (v > 0) opp_pts[23 - i] = v;
        }
        *my_bar  = s->bar[BLACK]; *my_off  = s->off[BLACK];
        *opp_bar = s->bar[WHITE]; *opp_off = s->off[WHITE];
    }
}

static void gnubg_group1_7(const int *my_pts, const int *opp_pts,
                            int my_bar, int my_off, float *f) {
    int off, back, back_anchor, fwd, opp_back, free_pips, i;

    /* I_OFF1/2/3: 3-node thermometer, capped at 8 */
    off = my_off < 8 ? my_off : 8;
    f[0] = (off < 3 ? off : 3) / 3.0f;
    f[1] = (off > 3 ? (off - 3 < 3 ? off - 3 : 3) : 0) / 3.0f;
    f[2] = (off > 6 ? off - 6 : 0) / 2.0f;

    /* I_BACK_CHEQUER */
    if (my_bar > 0) {
        back = 24;
    } else {
        back = 0;
        for (i = 23; i >= 0; i--) {
            if (my_pts[i] > 0) { back = i; break; }
        }
    }
    f[3] = back / 24.0f;

    /* I_BACK_ANCHOR */
    back_anchor = 0;
    for (i = 23; i >= 0; i--) {
        if (my_pts[i] >= 2) { back_anchor = i; break; }
    }
    f[4] = back_anchor / 24.0f;

    /* I_FORWARD_ANCHOR */
    fwd = -1;
    for (i = 23; i > 11; i--) {
        if (my_pts[i] >= 2) { fwd = i; break; }
    }
    f[5] = (fwd < 0) ? 2.0f : (fwd + 1) / 6.0f;

    /* I_FREEPIP */
    opp_back = 24;
    for (i = 0; i < 24; i++) {
        if (opp_pts[i] > 0) { opp_back = i; break; }
    }
    free_pips = 0;
    for (i = 0; i < opp_back; i++) {
        free_pips += my_pts[i] * (i + 1);
    }
    f[6] = free_pips / 100.0f;
}

void encode_state_210(const BoardState *s, float *x) {
    int my_pts[24], opp_pts[24];
    int my_bar, my_off, opp_bar, opp_off;
    int opp_flipped[24], my_flipped[24];
    int i;

    /* First 196 features = standard perspective encoding */
    encode_state(s, x);

    perspective_arrays(s, my_pts, opp_pts, &my_bar, &my_off, &opp_bar, &opp_off);

    /* My-side Group-1 features [196-202] */
    gnubg_group1_7(my_pts, opp_pts, my_bar, my_off, x + 196);

    /* Opp-side Group-1 features [203-209]: flip arrays so idx 0 = opp's ace */
    for (i = 0; i < 24; i++) {
        opp_flipped[i] = opp_pts[23 - i];
        my_flipped[i]  = my_pts[23 - i];
    }
    gnubg_group1_7(opp_flipped, my_flipped, opp_bar, opp_off, x + 203);
}

/* ── Group-2: Roll-enumeration features ───────────────────────── */

static int can_hit(int blot_idx, const int *opp_pts, const int *my_pts,
                   int d1, int d2) {
    int src, k, n, blocked, inter;
    if (d1 == d2) {
        int d = d1;
        for (n = 1; n <= 4; n++) {
            src = blot_idx + n * d;
            if (src >= 24) break;
            if (opp_pts[src] == 0) continue;
            blocked = 0;
            for (k = 1; k < n; k++) {
                inter = blot_idx + k * d;
                if (inter >= 0 && inter < 24 && my_pts[inter] >= 2) {
                    blocked = 1; break;
                }
            }
            if (!blocked) return 1;
        }
    } else {
        /* Direct d1 */
        src = blot_idx + d1;
        if (src >= 0 && src < 24 && opp_pts[src] > 0) return 1;
        /* Direct d2 */
        src = blot_idx + d2;
        if (src >= 0 && src < 24 && opp_pts[src] > 0) return 1;
        /* Combined d1+d2 */
        src = blot_idx + d1 + d2;
        if (src >= 0 && src < 24 && opp_pts[src] > 0) {
            int i1 = blot_idx + d1;
            int i2 = blot_idx + d2;
            int v1 = (i1 >= 0 && i1 < 24 && my_pts[i1] < 2);
            int v2 = (i2 >= 0 && i2 < 24 && my_pts[i2] < 2);
            if (v1 || v2) return 1;
        }
    }
    return 0;
}

static int escapes_fn(int pos, const int *opp_pts, int d1, int d2,
                      int target, int inclusive) {
    /* inclusive=0: strictly past (dest < target)
     * inclusive=1: landing on target counts (dest <= target) */
    int dest, n, k, blocked, inter;
    if (d1 == d2) {
        int d = d1;
        for (n = 1; n <= 4; n++) {
            dest = pos - n * d;
            if (dest < 0) return 1;
            blocked = 0;
            for (k = 1; k < n; k++) {
                inter = pos - k * d;
                if (inter >= 0 && inter < 24 && opp_pts[inter] >= 2) {
                    blocked = 1; break;
                }
            }
            if (blocked) continue;
            if (inclusive ? (dest <= target) : (dest < target)) {
                if (opp_pts[dest] < 2) return 1;
            }
        }
    } else {
        /* d1 only */
        dest = pos - d1;
        if (dest < 0) return 1;
        if ((inclusive ? dest <= target : dest < target) && opp_pts[dest] < 2)
            return 1;
        /* d2 only */
        dest = pos - d2;
        if (dest < 0) return 1;
        if ((inclusive ? dest <= target : dest < target) && opp_pts[dest] < 2)
            return 1;
        /* d1+d2 combined */
        dest = pos - d1 - d2;
        if (dest < 0) return 1;
        if ((inclusive ? dest <= target : dest < target) && opp_pts[dest] < 2) {
            int i1 = pos - d1;
            int i2 = pos - d2;
            int v1 = (i1 < 0 || opp_pts[i1] < 2);
            int v2 = (i2 < 0 || opp_pts[i2] < 2);
            if (v1 || v2) return 1;
        }
    }
    return 0;
}

static void gnubg_group2_7(const int *my_pts, const int *opp_pts,
                            int my_bar, int my_off,
                            int opp_bar, int opp_off, float *f) {
    int d1, d2, i, hits, roll_pip_loss;
    int pip_loss_total = 0, p1_count = 0, p2_count = 0;
    int esc_count = 0, esc1_count = 0;
    int enter_loss = 0, closed = 0;
    int back_pos = -1, opp_back = 24;

    /* Find blots */
    int blots[24], num_blots = 0;
    for (i = 0; i < 24; i++) {
        if (my_pts[i] == 1) blots[num_blots++] = i;
    }

    /* PIPLOSS, P1, P2 */
    for (d1 = 1; d1 <= 6; d1++) {
        for (d2 = 1; d2 <= 6; d2++) {
            hits = 0; roll_pip_loss = 0;
            for (i = 0; i < num_blots; i++) {
                if (can_hit(blots[i], opp_pts, my_pts, d1, d2)) {
                    hits++;
                    roll_pip_loss += blots[i] + 1;
                }
            }
            pip_loss_total += roll_pip_loss;
            if (hits >= 1) p1_count++;
            if (hits >= 2) p2_count++;
        }
    }
    f[0] = pip_loss_total / (12.0f * 36.0f);
    f[1] = p1_count / 36.0f;
    f[2] = p2_count / 36.0f;

    /* Back checker and opp back */
    if (my_bar > 0) {
        back_pos = -1;
    } else {
        for (i = 23; i >= 0; i--) {
            if (my_pts[i] > 0) { back_pos = i; break; }
        }
    }
    for (i = 0; i < 24; i++) {
        if (opp_pts[i] > 0) { opp_back = i; break; }
    }

    /* BACKESCAPES, BACKRESCAPES */
    if (back_pos > 0 && back_pos > opp_back) {
        for (d1 = 1; d1 <= 6; d1++) {
            for (d2 = 1; d2 <= 6; d2++) {
                if (escapes_fn(back_pos, opp_pts, d1, d2, opp_back, 0))
                    esc_count++;
                if (escapes_fn(back_pos, opp_pts, d1, d2, opp_back, 1))
                    esc1_count++;
            }
        }
    }
    f[3] = esc_count / 36.0f;
    f[4] = esc1_count / 36.0f;

    /* ENTER */
    for (d1 = 1; d1 <= 6; d1++) {
        for (d2 = 1; d2 <= 6; d2++) {
            if (d1 == d2) {
                if (opp_pts[24 - d1] >= 2)
                    enter_loss += 4 * d1;
            } else {
                if (opp_pts[24 - d1] >= 2) enter_loss += d1;
                if (opp_pts[24 - d2] >= 2) enter_loss += d2;
            }
        }
    }
    f[5] = enter_loss / (36.0f * 49.0f / 6.0f);

    /* ENTER2 */
    for (i = 18; i < 24; i++) {
        if (opp_pts[i] >= 2) closed++;
    }
    f[6] = (36 - (closed - 6) * (closed - 6)) / 36.0f;
}

void encode_state_224(const BoardState *s, float *x) {
    int my_pts[24], opp_pts[24];
    int my_bar, my_off, opp_bar, opp_off;
    int opp_flipped[24], my_flipped[24];
    int i;

    /* First 210 features = standard 210-encoding */
    encode_state_210(s, x);

    perspective_arrays(s, my_pts, opp_pts, &my_bar, &my_off, &opp_bar, &opp_off);

    /* My-side Group-2 features [210-216] */
    gnubg_group2_7(my_pts, opp_pts, my_bar, my_off, opp_bar, opp_off, x + 210);

    /* Opp-side Group-2 features [217-223] */
    for (i = 0; i < 24; i++) {
        opp_flipped[i] = opp_pts[23 - i];
        my_flipped[i]  = my_pts[23 - i];
    }
    gnubg_group2_7(opp_flipped, my_flipped, opp_bar, opp_off, my_bar, my_off, x + 217);
}

/* ── Group-3: Positional/strategic features ───────────────────── */

static int count_escapes(int pos, const int *blocking_pts, int target) {
    int count = 0, d1, d2;
    for (d1 = 1; d1 <= 6; d1++)
        for (d2 = 1; d2 <= 6; d2++)
            if (escapes_fn(pos, blocking_pts, d1, d2, target, 0))
                count++;
    return count;
}

static void gnubg_group3_11(const int *my_pts, const int *opp_pts,
                             int my_bar, int my_off,
                             int opp_bar, int opp_off, float *f) {
    int i, j, d1, d2, d, dest;
    int opp_back_hi, total_checkers, total_pips;
    float mean_pip, variance;

    for (i = 0; i < 11; i++) f[i] = 0.0f;

    /* Opp's back checker (highest perspective index) */
    opp_back_hi = -1;
    for (i = 23; i >= 0; i--) {
        if (opp_pts[i] > 0) { opp_back_hi = i; break; }
    }

    /* I_BREAK_CONTACT */
    if (opp_back_hi >= 0) {
        int break_pips = 0;
        for (i = opp_back_hi; i < 24; i++)
            if (my_pts[i] > 0)
                break_pips += my_pts[i] * (i - opp_back_hi + 1);
        if (my_bar > 0)
            break_pips += my_bar * (24 - opp_back_hi + 1);
        f[0] = break_pips / 167.0f;
    }

    /* I_ACONTAIN: anchor containment of opp's back checker */
    if (opp_back_hi > 0) {
        int min_esc = 36, upper = opp_back_hi < 8 ? opp_back_hi : 8;
        for (i = 0; i <= upper; i++) {
            if (my_pts[i] >= 2) {
                int esc = count_escapes(opp_back_hi, my_pts, i);
                if (esc < min_esc) min_esc = esc;
            }
        }
        if (min_esc < 36)
            f[1] = (36 - min_esc) / 36.0f;
    }
    f[2] = f[1] * f[1];

    /* I_CONTAIN: prime quality from point 9 to home */
    {
        int min_esc = 36;
        for (i = 0; i < 8; i++) {
            if (my_pts[i] >= 2) {
                int esc = count_escapes(8, my_pts, i);
                if (esc < min_esc) min_esc = esc;
            }
        }
        if (min_esc < 36)
            f[3] = (36 - min_esc) / 36.0f;
    }
    f[4] = f[3] * f[3];

    /* I_MOBILITY */
    {
        int mobility = 0;
        for (i = 6; i < 24; i++) {
            if (my_pts[i] == 0) continue;
            int dist = i - 5;
            int esc = 0;
            for (d1 = 1; d1 <= 6; d1++) {
                for (d2 = 1; d2 <= 6; d2++) {
                    int can_move = 0;
                    for (j = 0; j < 2 && !can_move; j++) {
                        d = (j == 0) ? d1 : d2;
                        dest = i - d;
                        if (dest < 0 || (dest < 24 && opp_pts[dest] < 2))
                            can_move = 1;
                    }
                    if (!can_move && d1 != d2) {
                        dest = i - d1 - d2;
                        if (dest < 0) can_move = 1;
                        else if (dest < 24 && opp_pts[dest] < 2) {
                            int i1 = i - d1, i2 = i - d2;
                            if ((i1 < 0 || opp_pts[i1] < 2) || (i2 < 0 || opp_pts[i2] < 2))
                                can_move = 1;
                        }
                    }
                    if (can_move) esc++;
                }
            }
            mobility += my_pts[i] * dist * esc;
        }
        if (my_bar > 0) {
            int bar_esc = 0;
            for (d1 = 1; d1 <= 6; d1++) {
                for (d2 = 1; d2 <= 6; d2++) {
                    int can_enter = 0;
                    for (j = 0; j < 2 && !can_enter; j++) {
                        d = (j == 0) ? d1 : d2;
                        int entry = 24 - d;
                        if (entry >= 0 && entry < 24 && opp_pts[entry] < 2)
                            can_enter = 1;
                    }
                    if (can_enter) bar_esc++;
                }
            }
            mobility += my_bar * 19 * bar_esc;
        }
        f[5] = mobility / 3600.0f;
    }

    /* I_MOMENT2 */
    total_checkers = my_bar;
    total_pips = my_bar * 25;
    for (i = 0; i < 24; i++) {
        total_checkers += my_pts[i];
        total_pips += my_pts[i] * (i + 1);
    }
    if (total_checkers > 0) {
        mean_pip = (float)total_pips / total_checkers;
        variance = 0.0f;
        for (i = 0; i < 24; i++) {
            if (my_pts[i] > 0 && (i + 1) > mean_pip) {
                float diff = (i + 1) - mean_pip;
                variance += my_pts[i] * diff * diff;
            }
        }
        if (my_bar > 0 && 25 > mean_pip) {
            float diff = 25 - mean_pip;
            variance += my_bar * diff * diff;
        }
        f[6] = variance / 400.0f;
    }

    /* I_TIMING */
    {
        int timing = 0;
        for (i = 6; i < 24; i++)
            if (my_pts[i] > 0)
                timing += my_pts[i] * (i - 5);
        if (my_bar > 0)
            timing += my_bar * 19;
        for (i = 0; i < 6; i++)
            if (my_pts[i] > 3)
                timing += my_pts[i] - 3;
        f[7] = timing / 100.0f;
    }

    /* I_BACKBONE */
    {
        int made[24], num_made = 0, consecutive = 0;
        for (i = 0; i < 24; i++)
            if (my_pts[i] >= 2) made[num_made++] = i;
        if (num_made >= 2) {
            for (j = 0; j < num_made - 1; j++)
                if (made[j + 1] - made[j] == 1) consecutive++;
            f[8] = (float)consecutive / (num_made - 1);
        }
    }

    /* I_BACKG / I_BACKG1 */
    {
        int anchors = 0, total = 0;
        for (i = 18; i < 24; i++) {
            if (my_pts[i] >= 2) anchors++;
            total += my_pts[i];
        }
        if (anchors >= 2)
            f[9] = (total - 3) > 0 ? (total - 3) / 4.0f : 0.0f;
        if (anchors == 1)
            f[10] = total / 8.0f;
    }
}

void encode_state_246(const BoardState *s, float *x) {
    int my_pts[24], opp_pts[24];
    int my_bar, my_off, opp_bar, opp_off;
    int opp_flipped[24], my_flipped[24];
    int i;

    encode_state_224(s, x);

    perspective_arrays(s, my_pts, opp_pts, &my_bar, &my_off, &opp_bar, &opp_off);

    gnubg_group3_11(my_pts, opp_pts, my_bar, my_off, opp_bar, opp_off, x + 224);

    for (i = 0; i < 24; i++) {
        opp_flipped[i] = opp_pts[23 - i];
        my_flipped[i]  = my_pts[23 - i];
    }
    gnubg_group3_11(opp_flipped, my_flipped, opp_bar, opp_off, my_bar, my_off, x + 235);
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
    int count;
} BoardHashSet;

static void hashset_init(BoardHashSet *hs) {
    memset(hs->occupied, 0, sizeof(hs->occupied));
    hs->count = 0;
}

/* Returns 1 if the state was already in the set, 0 if newly inserted. */
static int hashset_insert(BoardHashSet *hs, const BoardState *s) {
    unsigned int h = board_hash(s);
    unsigned int idx = h & HASH_MASK;

    /* Linear probing with full state comparison */
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        unsigned int probe = (idx + i) & HASH_MASK;
        if (!hs->occupied[probe]) {
            /* Empty slot: insert */
            hs->hashes[probe] = h;
            memcpy(&hs->states[probe], s, sizeof(BoardState));
            hs->occupied[probe] = 1;
            hs->count++;
            return 0;  /* newly inserted */
        }
        if (hs->hashes[probe] == h && board_equal(&hs->states[probe], s)) {
            return 1;  /* exact duplicate */
        }
    }
    return 0;  /* table full, insert anyway */
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
         * Only record if this resulting state hasn't been seen yet. */
        if (!hashset_insert(ctx->seen, state)) {
            /* New unique state: record this play */
            if (ctx->count < ctx->max_plays) {
                Play *p = &ctx->plays[ctx->count];
                p->num_moves = num_current_moves;
                for (int i = 0; i < num_current_moves; i++) {
                    p->moves[i] = current_moves[i];
                }
                memcpy(&p->resulting_state, state, sizeof(BoardState));
                ctx->count++;
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
        return die >= d;
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

/* ── Combined get-plays + 210-feature encode ──────────────────── */

int get_legal_plays_encoded_210(const BoardState *state, int d1, int d2,
                                Play *plays, int max_plays,
                                float *encoded_features) {
    int count = get_legal_plays(state, d1, d2, plays, max_plays);

    /* v2: resulting_state is already switched. */
    for (int i = 0; i < count; i++) {
        encode_state_210(&plays[i].resulting_state,
                         encoded_features + i * NUM_FEATURES_210);
    }

    return count;
}

/* ── Combined get-plays + 224-feature encode ──────────────────── */

int get_legal_plays_encoded_224(const BoardState *state, int d1, int d2,
                                Play *plays, int max_plays,
                                float *encoded_features) {
    int count = get_legal_plays(state, d1, d2, plays, max_plays);

    /* v2: resulting_state is already switched. */
    for (int i = 0; i < count; i++) {
        encode_state_224(&plays[i].resulting_state,
                         encoded_features + i * NUM_FEATURES_224);
    }

    return count;
}

int get_legal_plays_encoded_246(const BoardState *state, int d1, int d2,
                                Play *plays, int max_plays,
                                float *encoded_features) {
    int count = get_legal_plays(state, d1, d2, plays, max_plays);

    /* v2: resulting_state is already switched. */
    for (int i = 0; i < count; i++) {
        encode_state_246(&plays[i].resulting_state,
                         encoded_features + i * NUM_FEATURES_246);
    }

    return count;
}
