"""test_match_length.py — pit two GnubgNNCubefulAgents with different
match-length approximations against each other.

Used to find the best match-length approximation for money-game cube
decisions in our gnubg-cubeful opponent. We want the approximation
that best mimics money-game cube economics — the pair of agents that
ties most closely (smallest equity gap) tells us which ml values are
best converged toward true money play.

Usage:
    python tests/test_match_length.py --ml1 13 --ml2 21 --games 100000 --workers 192

Reports head-to-head equity, with bootstrap CI, plus tail diagnostics.
"""
import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import multiprocessing as mp
import random
import time

import numpy as np

from agents import GnubgNNCubefulAgent
from backgammon_engine import WHITE, BLACK
from gnubg_eval import play_and_record_cubeful


def _play_one_signed(agent1, agent2, jacoby=True):
    """Random side assignment; signed equity from agent1's perspective."""
    a1_is_white = random.random() < 0.5
    first = agent1 if a1_is_white else agent2
    second = agent2 if a1_is_white else agent1
    record = play_and_record_cubeful(first, second, jacoby=jacoby)
    winner = record.winner
    if record.ended_by_drop:
        stake = record.cube_value
    else:
        stake = record.cube_value * record.result
    white_equity = stake if winner == WHITE else -stake
    return white_equity if a1_is_white else -white_equity


def _worker(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    ml1, ml2, plies, num_games, jacoby = args
    agent1 = GnubgNNCubefulAgent(plies=plies, match_length=ml1)
    agent2 = GnubgNNCubefulAgent(plies=plies, match_length=ml2)
    a1_wins = a2_wins = 0
    a1_eq = 0
    per_game = []
    for _ in range(num_games):
        signed = _play_one_signed(agent1, agent2, jacoby=jacoby)
        if signed > 0:
            a1_wins += 1
        else:
            a2_wins += 1
        a1_eq += signed
        per_game.append(signed)
    return a1_wins, a2_wins, a1_eq, per_game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml1", type=int, default=13)
    parser.add_argument("--ml2", type=int, default=21)
    parser.add_argument("--games", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=192)
    parser.add_argument("--plies", type=int, default=0)
    parser.add_argument("--jacoby", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    args = parser.parse_args()

    print(f"GnubgNNCubeful ml={args.ml1} vs ml={args.ml2}, "
          f"{args.games} games, {args.workers} workers")
    t0 = time.perf_counter()

    base = args.games // args.workers
    rem = args.games % args.workers
    counts = [base + (1 if i < rem else 0) for i in range(args.workers)]
    worker_args = [
        (args.ml1, args.ml2, args.plies, n, args.jacoby) for n in counts
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = pool.map(_worker, worker_args)

    a1_wins = sum(r[0] for r in results)
    a2_wins = sum(r[1] for r in results)
    a1_eq = sum(r[2] for r in results)
    per_game = []
    for r in results:
        per_game.extend(r[3])
    per_game_arr = np.asarray(per_game, dtype=np.float64)
    elapsed = time.perf_counter() - t0
    n = a1_wins + a2_wins
    print(f"  {n} games completed in {elapsed:.1f}s "
          f"({n/elapsed:.0f} games/sec)")
    print()

    # Bootstrap CI on capped equity (cap ±128 — robust to runaway tails)
    cap = 128.0
    capped = np.clip(per_game_arr, -cap, cap)
    mean = float(capped.mean())
    rng = np.random.default_rng(0)
    boot_means = []
    for _ in range(args.n_bootstrap):
        sample = rng.choice(capped, size=len(capped), replace=True)
        boot_means.append(sample.mean())
    boot = np.array(boot_means)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    a1_pct = 100 * a1_wins / n
    print(f"  ml={args.ml1}: {a1_wins} wins ({a1_pct:.2f}%)")
    print(f"  ml={args.ml2}: {a2_wins} wins ({100 - a1_pct:.2f}%)")
    print()
    print(f"  Capped mean (±{int(cap)}) for ml={args.ml1}: "
          f"{mean*1000:+.1f} mEq/game  CI [{ci_lo*1000:+.1f}, {ci_hi*1000:+.1f}]")
    raw_mean = float(per_game_arr.mean())
    print(f"  Raw mean: {raw_mean*1000:+.1f} mEq/game")
    print()

    # Tail diagnostics
    print("  Tail counts (per-game equity, signed for ml1):")
    for thresh in (32, 128, 512, 2048, 8192):
        n_neg = int((per_game_arr <= -thresh).sum())
        n_pos = int((per_game_arr >= thresh).sum())
        print(f"    |eq| >= {thresh:>5}: neg={n_neg:>5}  pos={n_pos:>5}")


if __name__ == "__main__":
    main()
