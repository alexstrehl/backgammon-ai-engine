/*
 * example_selfplay.c -- PureTD C API sample: seeded games between one or
 * two exported nets, with the result tally and throughput.
 *
 *   puretd_selfplay_<b> [-n games] [-s seed] [-p plies] [-P black_plies]
 *                       [--cubeless] [--no-cube] [--no-jacoby]
 *                       white.bin [black.bin]
 *
 * Export a checkpoint with: python export_weights.py model.pt model.bin
 */

#define _POSIX_C_SOURCE 200809L

#include "puretd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [-n games] [-s seed] [-p plies] [-P black_plies]\n"
            "          [--cubeless] [--no-cube] [--no-jacoby] white.bin [black.bin]\n",
            argv0);
}

int main(int argc, char **argv) {
    int games = 100, plies = 0, black_plies = -1;
    unsigned long long seed = 1;
    bgc_game_opts o = {1, 1, 1, 0, 0};
    const char *paths[2] = {NULL, NULL};
    int npath = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) games = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) plies = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-P") && i + 1 < argc) black_plies = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cubeless")) o.cubeful = 0;
        else if (!strcmp(argv[i], "--no-cube")) o.cube_enabled = 0;
        else if (!strcmp(argv[i], "--no-jacoby")) o.jacoby = 0;
        else if (argv[i][0] == '-') { usage(argv[0]); return 2; }
        else if (npath < 2) paths[npath++] = argv[i];
        else { usage(argv[0]); return 2; }
    }
    if (npath == 0 || games <= 0) { usage(argv[0]); return 2; }
    o.plies_white = plies;
    o.plies_black = black_plies >= 0 ? black_plies : plies;

    bgc_engine *white = bgc_open(paths[0]);
    if (!white) return 1;
    bgc_engine *black = white;
    if (npath == 2) {
        black = bgc_open(paths[1]);
        if (!black) { bgc_close(white); return 1; }
    }
    printf("white %s (kind %d), black %s (kind %d), backend %s\n",
           paths[0], bgc_kind(white), npath == 2 ? paths[1] : paths[0],
           bgc_kind(black), bgc_describe(white));
    printf("%s, %s, jacoby %s, plies %d/%d, %d games, seed %llu\n",
           o.cubeful ? "cubeful money" : "cubeless money",
           o.cubeful ? (o.cube_enabled ? "cube on" : "cube off") : "no cube",
           o.jacoby ? "on" : "off", o.plies_white, o.plies_black, games, seed);

    bgc_rng rng;
    bgc_rng_seed(&rng, seed);
    long long points[2] = {0, 0}, wins[2] = {0, 0}, kinds[2][4] = {{0}}, drops = 0;
    long long decisions = 0;
    double t0 = now_s();
    for (int g = 0; g < games; g++) {
        bgc_game_result r;
        if (bgc_play_game(white, black, &o, &rng, &r, NULL, NULL)) {
            fprintf(stderr, "game %d failed\n", g);
            return 1;
        }
        wins[r.winner]++;
        points[r.winner] += (long long)r.result * r.cube_value;
        kinds[r.winner][r.result]++;
        drops += r.ended_by_drop;
        decisions += r.n_checker + (o.cubeful && o.cube_enabled ? r.n_cube_offer + r.n_cube_resp : 0);
    }
    double dt = now_s() - t0;

    printf("white: %lld wins (%lld single, %lld gammon, %lld backgammon), %lld points\n",
           wins[WHITE], kinds[WHITE][1], kinds[WHITE][2], kinds[WHITE][3], points[WHITE]);
    printf("black: %lld wins (%lld single, %lld gammon, %lld backgammon), %lld points\n",
           wins[BLACK], kinds[BLACK][1], kinds[BLACK][2], kinds[BLACK][3], points[BLACK]);
    if (o.cubeful) printf("games ended by a drop: %lld\n", drops);
    printf("white ppg %+.4f | %lld decisions, %.1f dec/s | %.2f games/s | %.2f s\n",
           (double)(points[WHITE] - points[BLACK]) / games, decisions,
           decisions / dt, games / dt, dt);

    if (black != white) bgc_close(black);
    bgc_close(white);
    return 0;
}
