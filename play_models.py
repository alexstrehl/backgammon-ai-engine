"""
play_models.py -- Play two saved models against each other for N games.

Randomly assigns WHITE/BLACK each game for fairness.
Reports win counts, percentages, and statistical significance.

Usage examples:
    python play_models.py --model1 model_a.pt --model2 model_b.pt --games 1000
    python play_models.py --model1 model.pt --model2 random --games 500
"""

import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import random
import time
import math
import multiprocessing as mp
from typing import Tuple

from backgammon_engine import (
    BoardState, WHITE, BLACK,
    get_legal_plays, switch_turn,
)
from agents import Agent, RandomAgent, GnubgNNAgent
from td_agent import TDAgent
from model import TDNetwork


def play_game(
    agent_white: Agent,
    agent_black: Agent,
) -> int:
    """Play one full game. Returns the winner (WHITE or BLACK)."""
    state = BoardState.initial()

    while not state.is_game_over():
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        plays = get_legal_plays(state, (d1, d2))

        if plays:
            agent = agent_white if state.turn == WHITE else agent_black
            play, next_state = agent.choose_play(state, (d1, d2), plays)
            state = switch_turn(next_state)
        else:
            state = switch_turn(state)

    return state.winner()


def _play_matches_worker(args):
    """Worker function for parallel game playing."""
    os.environ["OMP_NUM_THREADS"] = "1"
    model1_path, model2_path, num_games, model2_type, plies = args

    net1 = TDNetwork.load(model1_path)
    agent1 = TDAgent(net1)

    if model2_type == "random":
        agent2 = RandomAgent()
    elif model2_type == "gnubg":
        agent2 = GnubgNNAgent(plies=plies)
    else:
        net2 = TDNetwork.load(model2_path)
        agent2 = TDAgent(net2)

    a1_wins = 0
    a2_wins = 0
    for _ in range(num_games):
        if random.random() < 0.5:
            winner = play_game(agent1, agent2)
            if winner == WHITE:
                a1_wins += 1
            else:
                a2_wins += 1
        else:
            winner = play_game(agent2, agent1)
            if winner == WHITE:
                a2_wins += 1
            else:
                a1_wins += 1
    return a1_wins, a2_wins


def play_matches(
    agent1: Agent,
    agent2: Agent,
    num_games: int,
    print_progress: bool = True,
) -> Tuple[int, int]:
    """
    Play agent1 vs agent2 for num_games.
    Each game randomly assigns WHITE/BLACK (50/50 chance).
    Returns (agent1_wins, agent2_wins).
    """
    agent1_wins = 0
    agent2_wins = 0
    start_time = time.time()
    progress_interval = max(100, num_games // 10)

    for i in range(1, num_games + 1):
        # Randomly decide who goes first
        if random.random() < 0.5:
            # agent1 is WHITE, agent2 is BLACK
            winner = play_game(agent1, agent2)
            if winner == WHITE:
                agent1_wins += 1
            else:
                agent2_wins += 1
        else:
            # agent2 is WHITE, agent1 is BLACK
            winner = play_game(agent2, agent1)
            if winner == WHITE:
                agent2_wins += 1
            else:
                agent1_wins += 1

        if print_progress and i % progress_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"  Game {i:>5d}/{num_games}: "
                f"Model1 {agent1_wins:>4d}  Model2 {agent2_wins:>4d}  "
                f"({elapsed:.1f}s)"
            )

    return agent1_wins, agent2_wins


def play_matches_parallel(
    model1_path: str,
    model2_path: str,
    num_games: int,
    workers: int,
    model2_type: str = "model",
    plies: int = 0,
) -> Tuple[int, int]:
    """Play games in parallel across multiple workers.
    Returns (agent1_wins, agent2_wins).
    """
    # Split games across workers
    base = num_games // workers
    remainder = num_games % workers
    game_counts = [base + (1 if i < remainder else 0) for i in range(workers)]

    args_list = [
        (model1_path, model2_path, n, model2_type, plies)
        for n in game_counts
    ]

    ctx = mp.get_context('spawn')
    start_time = time.time()
    with ctx.Pool(workers) as pool:
        results = pool.map(_play_matches_worker, args_list)

    elapsed = time.time() - start_time
    total_a1 = sum(r[0] for r in results)
    total_a2 = sum(r[1] for r in results)
    print(f"  {num_games} games completed in {elapsed:.1f}s "
          f"({num_games/elapsed:.0f} games/sec, {workers} workers)")

    return total_a1, total_a2


def compute_binomial_pvalue(wins: int, trials: int) -> float:
    """
    Compute two-sided p-value for a binomial test.
    Null hypothesis: p = 0.5 (both models equally likely to win).

    Tries scipy.stats.binomtest first; falls back to normal approximation.
    """
    try:
        from scipy.stats import binomtest
        result = binomtest(wins, trials, 0.5, alternative='two-sided')
        return result.pvalue
    except (ImportError, AttributeError):
        # Fallback to normal approximation
        # z = (wins/n - 0.5) / sqrt(0.25/n)
        p_hat = wins / trials
        z = (p_hat - 0.5) / math.sqrt(0.25 / trials)

        # Two-sided p-value from standard normal
        from math import erfc
        pvalue = erfc(abs(z) / math.sqrt(2))
        return pvalue


def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.
    Returns (lower, upper) bounds for the win rate.

    Args:
        wins: Number of wins (successes)
        n: Total number of trials
        z: z-score for confidence level (1.96 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) as proportions [0, 1]
    """
    p_hat = wins / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    half_width = (z / denom) * math.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))
    return center - half_width, center + half_width


def get_verdict(wins: int, total: int) -> str:
    """
    Determine verdict based on Wilson 95% confidence interval.

    Rules:
    1. "Model1 wins" — CI entirely above 50%
    2. "Model2 wins" — CI entirely below 50%
    3. "Tied" — CI includes 50% AND CI width is tight (< 2%)
    4. "Inconclusive" — CI includes 50% but too wide to call
    """
    lower, upper = wilson_ci(wins, total, z=1.96)
    ci_width = upper - lower

    # Check if CI is entirely above 50%
    if lower > 0.50:
        return "Model1 is significantly stronger (p<0.01)"

    # Check if CI is entirely below 50%
    if upper < 0.50:
        return "Model2 is significantly stronger (p<0.01)"

    # CI includes 50%. Only call "tied" if we have enough data
    # (tight CI) AND the point estimate is close to 50%.
    if ci_width < 0.02 and lower >= 0.49 and upper <= 0.51:
        return "Models are effectively tied (difference < 1%)"

    # Otherwise, inconclusive — CI includes 50% but too wide or
    # point estimate leans enough that more games could resolve it
    return "Inconclusive — need more games to determine winner"


def main():
    parser = argparse.ArgumentParser(
        description="Play two backgammon models against each other"
    )
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Path to first model (.pt file)",
    )
    parser.add_argument(
        "--model2",
        type=str,
        default=None,
        help="Path to second model (.pt file) or 'random' (optional if --random is set)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to play (default 100)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Replace model2 with RandomAgent (shorthand for --model2 random)",
    )
    parser.add_argument(
        "--gnubg",
        action="store_true",
        help="Replace model2 with GnubgNNAgent (shorthand for --model2 gnubg)",
    )
    parser.add_argument(
        "--plies",
        type=int,
        default=0,
        help="Gnubg evaluation depth when using --gnubg (0=fastest, 2=strongest, default 0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )
    args = parser.parse_args()

    # Override model2 if shorthand flags are set
    if args.random:
        args.model2 = "random"
    elif args.gnubg:
        args.model2 = "gnubg"
    elif args.model2 is None:
        parser.error("--model2 is required unless --random or --gnubg is specified")

    # Load model1
    print(f"Loading model1: {args.model1}")
    net1 = TDNetwork.load(args.model1)
    agent1 = TDAgent(net1)

    # Load model2 (or use random / gnubg)
    if args.model2 == "random":
        agent2 = RandomAgent()
        model2_label = "RandomAgent"
    elif args.model2 == "gnubg":
        agent2 = GnubgNNAgent(plies=args.plies)
        model2_label = f"GnubgNN ({args.plies}-ply)"
    else:
        print(f"Loading model2: {args.model2}")
        net2 = TDNetwork.load(args.model2)
        agent2 = TDAgent(net2)
        model2_label = args.model2

    print(f"\nPlaying {args.games} games...")
    print(f"  Model1: {args.model1}")
    print(f"  Model2: {model2_label}")
    if args.workers > 1:
        print(f"  Workers: {args.workers}")
    print()

    # Play all games with random first-mover assignment
    if args.workers > 1:
        model2_type = "random" if args.model2 == "random" else (
            "gnubg" if args.model2 == "gnubg" else "model")
        agent1_wins, agent2_wins = play_matches_parallel(
            args.model1, args.model2, args.games,
            workers=args.workers, model2_type=model2_type, plies=args.plies,
        )
    else:
        agent1_wins, agent2_wins = play_matches(agent1, agent2, args.games)
    total_games = agent1_wins + agent2_wins
    print()

    # Compute statistics
    agent1_pct = 100 * agent1_wins / total_games
    agent2_pct = 100 * agent2_wins / total_games

    pvalue = compute_binomial_pvalue(agent1_wins, total_games)
    lower_ci, upper_ci = wilson_ci(agent1_wins, total_games, z=1.96)
    verdict = get_verdict(agent1_wins, total_games)

    # Print summary
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model1: {agent1_wins} wins ({agent1_pct:.1f}%)")
    print(f"Model2: {agent2_wins} wins ({agent2_pct:.1f}%)")
    print()
    print(f"Model1 win rate 95% CI: [{lower_ci*100:.1f}%, {upper_ci*100:.1f}%]")
    print(f"p-value: {pvalue:.4f}")
    print(f"Verdict: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
