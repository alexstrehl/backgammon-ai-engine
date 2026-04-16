"""
evaluate.py -- Play games between two agents and report results.

Usage examples:
    # TD agent vs random agent (100 games)
    python evaluate.py --model td_model_final.pt --opponent random --games 100

    # Two TD agents against each other
    python evaluate.py --model model_a.pt --opponent model_b.pt --games 100
"""

import random
import time
from typing import Tuple

from backgammon_engine import (
    BoardState, WHITE, BLACK,
    get_legal_plays, switch_turn,
)
from agents import Agent, RandomAgent
from td_agent import TDAgent
from model import TDNetwork


def play_game(
    agent_white: Agent,
    agent_black: Agent,
    verbose: bool = False,
) -> Tuple[int, int]:
    """Play one full game.  Returns (winner, num_moves)."""
    from backgammon_engine import play_label

    state = BoardState.initial()
    num_moves = 0

    while not state.is_game_over():
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        plays = get_legal_plays(state, (d1, d2))

        if plays:
            agent = agent_white if state.turn == WHITE else agent_black
            play, next_state = agent.choose_checker_action(state, (d1, d2), plays)

            if verbose:
                side = "W" if state.turn == WHITE else "B"
                print(f"  {side} rolls ({d1},{d2}): {play_label(play)}")

            # next_state already has turn switched (engine convention)
            state = next_state
        else:
            if verbose:
                side = "W" if state.turn == WHITE else "B"
                print(f"  {side} rolls ({d1},{d2}): no legal moves")
            state = switch_turn(state)

        num_moves += 1

    winner = state.winner()
    result = state.game_result()
    result_names = {1: "", 2: " (gammon)", 3: " (backgammon)"}

    if verbose:
        side = "WHITE" if winner == WHITE else "BLACK"
        print(f"  -> {side} wins{result_names[result]} in {num_moves} moves")

    return winner, num_moves


def evaluate(
    agent_white: Agent,
    agent_black: Agent,
    num_games: int = 100,
    verbose: bool = False,
) -> dict:
    """Play *num_games* and return summary statistics."""
    wins = {WHITE: 0, BLACK: 0}
    gammons = {WHITE: 0, BLACK: 0}
    backgammons = {WHITE: 0, BLACK: 0}
    total_moves = 0
    start_time = time.time()

    for i in range(1, num_games + 1):
        winner, moves = play_game(agent_white, agent_black, verbose=verbose)
        wins[winner] += 1
        total_moves += moves

        # Check for gammon/backgammon by looking at the final state
        # (play_game already returns, so we just count wins here)

        if i % max(1, num_games // 10) == 0:
            elapsed = time.time() - start_time
            print(
                f"  Game {i:>5d}/{num_games}: "
                f"W {wins[WHITE]:>4d}  B {wins[BLACK]:>4d}  "
                f"({elapsed:.1f}s)"
            )

    elapsed = time.time() - start_time

    return {
        "num_games": num_games,
        "white_wins": wins[WHITE],
        "black_wins": wins[BLACK],
        "white_pct": 100 * wins[WHITE] / num_games,
        "black_pct": 100 * wins[BLACK] / num_games,
        "avg_moves": total_moves / num_games,
        "elapsed": elapsed,
        "games_per_sec": num_games / elapsed if elapsed > 0 else 0,
    }


def print_results(results: dict, white_label: str, black_label: str):
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f"  {white_label} (WHITE) vs {black_label} (BLACK)")
    print(f"  {results['num_games']} games in {results['elapsed']:.1f}s "
          f"({results['games_per_sec']:.1f} games/sec)")
    print(f"{'='*50}")
    print(f"  WHITE wins: {results['white_wins']:>5d}  ({results['white_pct']:.1f}%)")
    print(f"  BLACK wins: {results['black_wins']:>5d}  ({results['black_pct']:.1f}%)")
    print(f"  Avg moves:  {results['avg_moves']:.1f}")
    print(f"{'='*50}\n")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate backgammon agents")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to TD model (.pt) for WHITE")
    parser.add_argument("--opponent", type=str, default="random",
                        help="'random' or path to a .pt model for BLACK")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--verbose", action="store_true",
                        help="Print each move")
    parser.add_argument("--swap", action="store_true",
                        help="Also run with sides swapped and average")
    args = parser.parse_args()

    # Build agents
    print(f"Loading WHITE model: {args.model}")
    white_net = TDNetwork.load(args.model)
    agent_white = TDAgent(white_net)
    white_label = args.model

    if args.opponent == "random":
        agent_black = RandomAgent()
        black_label = "Random"
    else:
        print(f"Loading BLACK model: {args.opponent}")
        black_net = TDNetwork.load(args.opponent)
        agent_black = TDAgent(black_net)
        black_label = args.opponent

    # Run evaluation
    print(f"\nPlaying {args.games} games...")
    results = evaluate(agent_white, agent_black, args.games, verbose=args.verbose)
    print_results(results, white_label, black_label)

    # Optionally run with sides swapped
    if args.swap:
        print(f"Playing {args.games} games with sides swapped...")
        results_swap = evaluate(
            agent_black, agent_white, args.games, verbose=args.verbose
        )
        print_results(results_swap, black_label, white_label)

        # Combined
        total = args.games * 2
        model_wins = results["white_wins"] + results_swap["black_wins"]
        opp_wins = results["black_wins"] + results_swap["white_wins"]
        print(f"Combined ({total} games): "
              f"Model {model_wins} ({100*model_wins/total:.1f}%)  "
              f"Opponent {opp_wins} ({100*opp_wins/total:.1f}%)")
