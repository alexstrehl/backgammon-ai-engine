/*
 * play_models.c -- Play two backgammon models against each other.
 *
 * Simplified C equivalent of play_models.py.
 * No multiprocessing, no gnubg -- just two models head-to-head.
 *
 * Usage:
 *   ./play_models model1.bin model2.bin [num_games] [seed]
 *
 * Build:
 *   gcc -O2 -o play_models play_models.c bg_engine.c nn_eval.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "bg_engine.h"
#include "nn_eval.h"

/* ── Choose best play ──────────────────────────────────────────── */

static int choose_play(const NNModel *model, const Play *plays, int num_plays) {
    float features[NUM_FEATURES_196];
    float best_val = 2.0f;
    int best_idx = 0;

    for (int i = 0; i < num_plays; i++) {
        BoardState s = plays[i].resulting_state;
        board_switch_turn(&s);
        encode_state(&s, features);
        float val = nn_forward(model, features);
        if (val < best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    return best_idx;
}

/* ── Play one game ─────────────────────────────────────────────── */

static int play_game(const NNModel *model_white, const NNModel *model_black) {
    BoardState state;
    Play plays[MAX_PLAYS];

    board_init(&state);

    while (!board_is_game_over(&state)) {
        int d1 = (rand() % 6) + 1;
        int d2 = (rand() % 6) + 1;
        int num_plays = get_legal_plays(&state, d1, d2, plays, MAX_PLAYS);

        if (num_plays > 0) {
            const NNModel *model = (state.turn == WHITE) ? model_white : model_black;
            int best = choose_play(model, plays, num_plays);
            state = plays[best].resulting_state;
        }
        board_switch_turn(&state);
    }

    return board_winner(&state);
}

/* ── Wilson score confidence interval ──────────────────────────── */

static void wilson_ci(int wins, int n, double z, double *lo, double *hi) {
    double p = (double)wins / n;
    double denom = 1.0 + z * z / n;
    double center = p + z * z / (2.0 * n);
    double margin = z * sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n);
    *lo = (center - margin) / denom;
    *hi = (center + margin) / denom;
}

/* ── Main ──────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model1.bin> <model2.bin> [num_games] [seed]\n", argv[0]);
        return 1;
    }

    const char *path1 = argv[1];
    const char *path2 = argv[2];
    int num_games = 100;
    unsigned int seed = (unsigned int)time(NULL);

    if (argc >= 4) num_games = atoi(argv[3]);
    if (argc >= 5) seed = (unsigned int)atoi(argv[4]);
    if (num_games < 1) num_games = 1;

    srand(seed);

    /* Load models */
    NNModel model1, model2;
    printf("Loading model1: %s\n", path1);
    if (nn_load(&model1, path1) != 0) return 1;
    printf("Loading model2: %s\n", path2);
    if (nn_load(&model2, path2) != 0) { nn_free(&model1); return 1; }

    printf("\nPlaying %d games (seed=%u)...\n", num_games, seed);

    int model1_wins = 0;
    int model2_wins = 0;
    clock_t t_start = clock();

    for (int g = 0; g < num_games; g++) {
        const NNModel *white_model, *black_model;
        int model1_is_white;

        /* Randomly assign colors */
        if (rand() % 2 == 0) {
            white_model = &model1;
            black_model = &model2;
            model1_is_white = 1;
        } else {
            white_model = &model2;
            black_model = &model1;
            model1_is_white = 0;
        }

        int winner = play_game(white_model, black_model);

        if ((winner == WHITE && model1_is_white) ||
            (winner == BLACK && !model1_is_white)) {
            model1_wins++;
        } else {
            model2_wins++;
        }
    }

    double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
    double rate = num_games / elapsed;

    /* Results */
    double pct1 = 100.0 * model1_wins / num_games;
    double pct2 = 100.0 * model2_wins / num_games;
    double lo, hi;
    wilson_ci(model1_wins, num_games, 1.96, &lo, &hi);

    printf("\n");
    printf("============================================================\n");
    printf("RESULTS (%d games, %.1fs, %.0f games/sec)\n", num_games, elapsed, rate);
    printf("============================================================\n");
    printf("Model1: %d wins (%.1f%%)\n", model1_wins, pct1);
    printf("Model2: %d wins (%.1f%%)\n", model2_wins, pct2);
    printf("\nModel1 win rate 95%% CI: [%.1f%%, %.1f%%]\n", lo * 100, hi * 100);
    printf("============================================================\n");

    nn_free(&model1);
    nn_free(&model2);
    return 0;
}
