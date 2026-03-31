/*
 * test_nn.c -- Verify C inference matches Python on known positions.
 *
 * Evaluates the initial position and a few derived positions,
 * printing float values to compare against Python output.
 *
 * Build:
 *   gcc -O2 -o test_nn test_nn.c bg_engine.c nn_eval.c -lm -I. -I/tmp/c_api
 *
 * Usage:
 *   ./test_nn model.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bg_engine.h"
#include "nn_eval.h"

static void print_features(const float *f, int n, const char *label) {
    printf("  %s first 10: [", label);
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%.4f%s", f[i], i < 9 ? ", " : "");
    }
    printf("]\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.bin>\n", argv[0]);
        return 1;
    }

    NNModel model;
    if (nn_load(&model, argv[1]) != 0) return 1;

    printf("Model: %d inputs, %d hidden layers [", model.input_size, model.num_hidden);
    for (int i = 0; i < model.num_hidden; i++)
        printf("%d%s", model.hidden_sizes[i], i < model.num_hidden - 1 ? "," : "");
    printf("], activation=%d\n\n", model.activation);

    float features[NUM_FEATURES_196];
    BoardState state;
    Play plays[MAX_PLAYS];

    /* ── Test 1: Initial position, WHITE to move ──────────────── */
    board_init(&state);
    encode_state(&state, features);
    float val = nn_forward(&model, features);
    printf("Test 1: Initial position (WHITE to move)\n");
    print_features(features, NUM_FEATURES_196, "features");
    printf("  nn_forward = %.8f\n\n", val);

    /* ── Test 2: Initial position, BLACK to move ──────────────── */
    board_init(&state);
    board_switch_turn(&state);
    encode_state(&state, features);
    val = nn_forward(&model, features);
    printf("Test 2: Initial position (BLACK to move)\n");
    print_features(features, NUM_FEATURES_196, "features");
    printf("  nn_forward = %.8f\n\n", val);

    /* ── Test 3: After WHITE plays opening 3-1 ────────────────── */
    board_init(&state);
    int num = get_legal_plays(&state, 3, 1, plays, MAX_PLAYS);
    if (num > 0) {
        /* Use first legal play */
        BoardState after = plays[0].resulting_state;
        board_switch_turn(&after);
        encode_state(&after, features);
        val = nn_forward(&model, features);
        printf("Test 3: After WHITE's first legal 3-1 play (BLACK to move)\n");
        printf("  Play: %d moves [", plays[0].num_moves);
        for (int i = 0; i < plays[0].num_moves; i++)
            printf("(%d->%d)%s", plays[0].moves[i].src, plays[0].moves[i].dst,
                   i < plays[0].num_moves - 1 ? " " : "");
        printf("]\n");
        print_features(features, NUM_FEATURES_196, "features");
        printf("  nn_forward = %.8f\n\n", val);
    }

    /* ── Test 4: After WHITE plays opening 6-4 ────────────────── */
    board_init(&state);
    num = get_legal_plays(&state, 6, 4, plays, MAX_PLAYS);
    if (num > 0) {
        BoardState after = plays[0].resulting_state;
        board_switch_turn(&after);
        encode_state(&after, features);
        val = nn_forward(&model, features);
        printf("Test 4: After WHITE's first legal 6-4 play (BLACK to move)\n");
        printf("  Play: %d moves [", plays[0].num_moves);
        for (int i = 0; i < plays[0].num_moves; i++)
            printf("(%d->%d)%s", plays[0].moves[i].src, plays[0].moves[i].dst,
                   i < plays[0].num_moves - 1 ? " " : "");
        printf("]\n");
        print_features(features, NUM_FEATURES_196, "features");
        printf("  nn_forward = %.8f\n\n", val);
    }

    nn_free(&model);
    return 0;
}
