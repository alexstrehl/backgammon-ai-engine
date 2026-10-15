#!/usr/bin/env python3
"""Compute XG++ PR with a bootstrap CI from an XG per-game stats export.

Input: the tab-separated .txt XG exports from its game-list statistics view
(one row per player per game; CRLF; columns like "Eq Move", "Doubles",
"Eq per Decision", "Opp Name").

PR convention (per the official XG2 manual, "Performance" section):
PR = 500 * sum(-Eq errors) / sum(decisions), where per row Eq errors =
Eq Move + Eq Double + Eq Take (the Wrong Double/Take columns are subsets
of those, fully penalized, never re-added) and decisions = Moves +
Doubles + Takes, XG's own non-obvious-decision counts (forced/trivial
moves and obvious cube actions are already excluded by XG upstream).

Two XG display quirks. (1) The profile Detail tab's "Resigns" row count
is the number of GAMES, not resign decisions, and it is NOT in the PR
denominator. (2) The profile Summary tab "Overall Level" is a recency-
weighted PR (exponential, decision half-life 10,000), not the flat PR
this tool computes.

The 95% CI is a game-level percentile bootstrap of that same weighted ratio
(games are the resampling unit, pooling both seats of a self-play game),
B=100,000 by default.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"no data rows in {path}")
    return rows


def row_err_dec(r):
    """(-PR error, -all error, decisions, checker/cube split) for one row.

    The two error values are intentionally IDENTICAL. XG's Wrong
    Double/Wrong Take columns are subsets of Eq Double/Eq Take, so
    "Eq per Decision" is exactly PR/500.
    """
    eq_move = float(r["Eq Move"])
    eq_cube = float(r["Eq Double"]) + float(r["Eq Take"])
    n_move = int(r["Moves"])
    n_cube = int(r["Doubles"]) + int(r["Takes"])
    err = -(eq_move + eq_cube)
    return (err, err, n_move + n_cube,
            (-eq_move, n_move, -eq_cube, n_cube))


def weighted_pr(err, dec):
    return 500.0 * err.sum() / dec.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="XG per-game stats export (.txt, tab-separated)")
    ap.add_argument("--player", default=None,
                    help="keep only this player's rows: rows whose 'Opp Name' "
                         "does NOT start with this prefix (the export names "
                         "only the opponent). Default: all rows")
    ap.add_argument("--rows-per-game", type=int, default=2,
                    help="rows forming one bootstrap unit (2 = both seats of "
                         "a self-play game; use 1 for single-player exports)")
    ap.add_argument("--n-boot", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_rows(args.results)
    # The export names only the OPPONENT. --player keeps rows where
    # Opp Name does NOT start with the given prefix.
    if args.player:
        rows = [r for r in rows if not r["Opp Name"].startswith(args.player)]
        if not rows:
            raise SystemExit(f"--player {args.player!r} matched no rows")

    err = np.empty(len(rows))
    err_all = np.empty(len(rows))
    dec = np.empty(len(rows))
    mv_err = np.empty(len(rows))  # per-row checker / cube split
    mv_dec = np.empty(len(rows))
    cu_err = np.empty(len(rows))
    cu_dec = np.empty(len(rows))
    seats = {}
    for i, r in enumerate(rows):
        e, ea, d, (me, mn, ce, cn) = row_err_dec(r)
        err[i], err_all[i], dec[i] = e, ea, d
        mv_err[i], mv_dec[i], cu_err[i], cu_dec[i] = me, mn, ce, cn
        seat = r["Opp Name"]
        s = seats.setdefault(seat, [0.0, 0])
        s[0] += e
        s[1] += d
    ck_err = np.array([mv_err.sum(), cu_err.sum()])
    ck_dec = np.array([mv_dec.sum(), cu_dec.sum()], dtype=np.int64)

    g = args.rows_per_game
    if len(rows) % g:
        raise SystemExit(f"{len(rows)} rows not divisible by --rows-per-game {g}")
    game_err = err.reshape(-1, g).sum(axis=1)
    game_dec = dec.reshape(-1, g).sum(axis=1)
    n_games = len(game_err)

    pr = weighted_pr(err, dec)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n_games, size=(args.n_boot, n_games))

    def boot_ci(row_e, row_d):
        """Percentile CI of the game-bootstrapped weighted PR (same
        resample indices for every metric, so CIs are comparable)."""
        ge = row_e.reshape(-1, g).sum(axis=1)
        gd = row_d.reshape(-1, g).sum(axis=1)
        # Guard the denominator. A cubeless export has 0 cube decisions
        # in every resample.
        boot = 500.0 * ge[idx].sum(axis=1) / np.maximum(gd[idx].sum(axis=1), 1)
        boot.sort()
        return boot[int(0.025 * args.n_boot)], boot[int(0.975 * args.n_boot)]

    lo, hi = boot_ci(err, dec)
    mv_lo, mv_hi = boot_ci(mv_err, mv_dec)
    cu_lo, cu_hi = boot_ci(cu_err, cu_dec)

    print(f"{len(rows)} rows, {n_games} games, {int(dec.sum())} decisions "
          f"({int(ck_dec[0])} checker, {int(ck_dec[1])} cube)")
    print(f"XG PR: {pr:.3f}  95% CI (game bootstrap, B={args.n_boot:,}): "
          f"[{lo:.3f}, {hi:.3f}]")
    print(f"  avg equity lost per decision: {err_all.sum() / dec.sum():.5f}")
    print(f"  checker-only PR: {500 * ck_err[0] / max(ck_dec[0], 1):.3f} "
          f"[{mv_lo:.3f}, {mv_hi:.3f}]")
    print(f"  cube-only PR:    {500 * ck_err[1] / max(ck_dec[1], 1):.3f} "
          f"[{cu_lo:.3f}, {cu_hi:.3f}]")
    for seat, (e, d) in sorted(seats.items()):
        print(f"  vs {seat:<16} PR {500 * e / d:.3f}  ({int(d)} decisions)")


if __name__ == "__main__":
    main()
