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
from agents import Agent, RandomAgent, GnubgNNAgent, GnubgNNCubefulAgent
from td_agent import TDAgent
from model import TDNetwork


def play_game(
    agent_white: Agent,
    agent_black: Agent,
    record: bool = False,
    cubeful: bool = False,
    jacoby: bool = True,
):
    """Play one full game.

    Cubeless (default): returns `(winner, stake)` where stake is
    1/2/3 (single / gammon / backgammon), or `(winner, stake,
    GameRecord)` when `record=True`.

    Cubeful: returns `(winner, stake)` where stake is the actual
    points won (cube_value × result, or cube_value on a drop).
    Always records internally (the cubeful loop needs match_state)
    and returns the GameRecord when `record=True`.
    """
    if cubeful:
        from gnubg_eval import play_and_record_cubeful, CubeRecord
        game_record = play_and_record_cubeful(
            agent_white, agent_black, jacoby=jacoby,
        )
        winner = game_record.winner
        if game_record.ended_by_drop:
            stake = game_record.cube_value
        else:
            stake = game_record.cube_value * game_record.result
        if record:
            return winner, stake, game_record
        return winner, stake

    state = BoardState.initial()
    game_record = None

    if record:
        from gnubg_eval import GameRecord, MoveRecord
        game_record = GameRecord()

    while not state.is_game_over():
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        plays = get_legal_plays(state, (d1, d2))

        if plays:
            agent = agent_white if state.turn == WHITE else agent_black
            play, next_state = agent.choose_checker_action(state, (d1, d2), plays)
            if game_record is not None:
                game_record.moves.append(MoveRecord(
                    player=state.turn, dice=(d1, d2), play=play))
            # next_state already has turn switched (engine convention)
            state = next_state
        else:
            if game_record is not None:
                game_record.moves.append(MoveRecord(
                    player=state.turn, dice=(d1, d2), play=()))
            state = switch_turn(state)

    winner = state.winner()
    stake = state.game_result()  # 1=single, 2=gammon, 3=backgammon

    if game_record is not None:
        game_record.winner = winner
        game_record.result = stake
        return winner, stake, game_record

    return winner, stake


def _play_one_signed(agent1, agent2, cubeful, jacoby, record=False):
    """Play one game with random side assignment. Returns
    (signed_equity, game_record_or_None) where signed_equity is
    positive when agent1 wins."""
    a1_is_white = random.random() < 0.5
    first = agent1 if a1_is_white else agent2
    second = agent2 if a1_is_white else agent1
    result = play_game(first, second, record=record,
                       cubeful=cubeful, jacoby=jacoby)
    if record:
        winner, stake, game_record = result
    else:
        winner, stake = result
        game_record = None
    white_equity = stake if winner == WHITE else -stake
    signed = white_equity if a1_is_white else -white_equity
    return signed, game_record


def _play_matches_worker(args):
    """Worker function for parallel game playing."""
    os.environ["OMP_NUM_THREADS"] = "1"
    (model1_path, model2_path, num_games, model2_type, plies,
     cubeless_money, cubeful, jacoby, oneply1, oneply2) = args

    net1 = TDNetwork.load(model1_path)
    agent1 = TDAgent(net1, oneply=oneply1)

    if model2_type == "random":
        agent2 = RandomAgent()
    elif model2_type == "gnubg":
        agent2 = GnubgNNAgent(plies=plies, cubeless_money=cubeless_money)
    elif model2_type == "gnubg-cubeful":
        agent2 = GnubgNNCubefulAgent(plies=plies)
    else:
        net2 = TDNetwork.load(model2_path)
        agent2 = TDAgent(net2, oneply=oneply2)

    a1_wins = 0
    a2_wins = 0
    a1_equity = 0
    a1_equity_sq = 0
    for _ in range(num_games):
        signed, _ = _play_one_signed(agent1, agent2, cubeful, jacoby)
        if signed > 0:
            a1_wins += 1
        else:
            a2_wins += 1
        a1_equity += signed
        a1_equity_sq += signed * signed
    return a1_wins, a2_wins, a1_equity, a1_equity_sq


def play_matches(
    agent1: Agent,
    agent2: Agent,
    num_games: int,
    print_progress: bool = True,
    record: bool = False,
    cubeful: bool = False,
    jacoby: bool = True,
):
    """
    Play agent1 vs agent2 for num_games.
    Each game randomly assigns WHITE/BLACK (50/50 chance).
    Returns (agent1_wins, agent2_wins, agent1_equity) or
    (agent1_wins, agent2_wins, agent1_equity, records) if record=True.

    `agent1_equity` is the signed sum of per-game stakes from
    agent1's perspective:
      - cubeless: stake = result (1/2/3).
      - cubeful:  stake = cube_value × result, or cube_value on a drop.
    """
    agent1_wins = 0
    agent2_wins = 0
    agent1_equity = 0
    agent1_equity_sq = 0
    records = [] if record else None
    start_time = time.time()
    progress_interval = max(100, num_games // 10)

    for i in range(1, num_games + 1):
        signed, game_record = _play_one_signed(
            agent1, agent2, cubeful, jacoby, record=record,
        )
        if record and game_record is not None:
            records.append(game_record)
        if signed > 0:
            agent1_wins += 1
        else:
            agent2_wins += 1
        agent1_equity += signed
        agent1_equity_sq += signed * signed

        if print_progress and i % progress_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"  Game {i:>5d}/{num_games}: "
                f"Model1 {agent1_wins:>4d}  Model2 {agent2_wins:>4d}  "
                f"({elapsed:.1f}s)"
            )

    if record:
        return agent1_wins, agent2_wins, agent1_equity, agent1_equity_sq, records
    return agent1_wins, agent2_wins, agent1_equity, agent1_equity_sq


def play_matches_parallel(
    model1_path: str,
    model2_path: str,
    num_games: int,
    workers: int,
    model2_type: str = "model",
    plies: int = 0,
    cubeless_money: bool = False,
    cubeful: bool = False,
    jacoby: bool = True,
    oneply1: bool = False,
    oneply2: bool = False,
) -> Tuple[int, int, int, int]:
    """Play games in parallel across multiple workers.
    Returns (agent1_wins, agent2_wins, agent1_equity, agent1_equity_sq).
    """
    # Split games across workers
    base = num_games // workers
    remainder = num_games % workers
    game_counts = [base + (1 if i < remainder else 0) for i in range(workers)]

    args_list = [
        (model1_path, model2_path, n, model2_type, plies, cubeless_money,
         cubeful, jacoby, oneply1, oneply2)
        for n in game_counts
    ]

    ctx = mp.get_context('spawn')
    start_time = time.time()
    with ctx.Pool(workers) as pool:
        results = pool.map(_play_matches_worker, args_list)

    elapsed = time.time() - start_time
    total_a1 = sum(r[0] for r in results)
    total_a2 = sum(r[1] for r in results)
    total_eq = sum(r[2] for r in results)
    total_eq_sq = sum(r[3] for r in results)
    print(f"  {num_games} games completed in {elapsed:.1f}s "
          f"({num_games/elapsed:.0f} games/sec, {workers} workers)")

    return total_a1, total_a2, total_eq, total_eq_sq


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
        "--gnubg-cubeful",
        action="store_true",
        help="Replace model2 with GnubgNNCubefulAgent (cubeful money play "
             "via gnubg-nn). Implies --game-mode cubeful-money.",
    )
    parser.add_argument(
        "--plies",
        type=int,
        default=0,
        help="Gnubg evaluation depth when using --gnubg/--gnubg-cubeful "
             "(0=fastest, 2=strongest, default 0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )
    parser.add_argument(
        "--save-games",
        type=str,
        default=None,
        help="Directory to save games as .mat files (Jellyfish format for gnubg import)",
    )
    parser.add_argument(
        "--game-mode",
        choices=["dmp", "cubeless-money", "cubeful-money"],
        default="dmp",
        help="dmp: pick by P(win), score by win/loss. "
             "cubeless-money: money equity scoring with gammon/backgammon "
             "weighting but no doubling cube. "
             "cubeful-money: money play with doubling cube; models must "
             "be cubeful (cubeful_perspective196 encoder). Stakes are "
             "cube_value × result (or cube_value on a drop).",
    )
    parser.add_argument(
        "--jacoby",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Jacoby rule for cubeful-money. Ignored otherwise.",
    )
    parser.add_argument(
        "--oneply1",
        action="store_true",
        help="1-ply lookahead for model1 checker and cube decisions.",
    )
    parser.add_argument(
        "--oneply2",
        action="store_true",
        help="1-ply lookahead for model2 checker and cube decisions "
             "(ignored when model2 is gnubg or random).",
    )
    parser.add_argument(
        "--oneply",
        action="store_true",
        help="1-ply lookahead for BOTH models (shorthand for --oneply1 --oneply2).",
    )
    args = parser.parse_args()
    if args.oneply:
        args.oneply1 = True
        args.oneply2 = True
    cubeless_money = (args.game_mode == "cubeless-money")
    cubeful = (args.game_mode == "cubeful-money")

    # Override model2 if shorthand flags are set
    if args.random:
        args.model2 = "random"
    elif args.gnubg:
        args.model2 = "gnubg"
    elif args.gnubg_cubeful:
        args.model2 = "gnubg-cubeful"
        cubeful = True  # force cubeful mode
    elif args.model2 is None:
        parser.error("--model2 is required unless --random, --gnubg, or --gnubg-cubeful is specified")

    # Load model1
    print(f"Loading model1: {args.model1}")
    net1 = TDNetwork.load(args.model1)
    agent1 = TDAgent(net1, oneply=args.oneply1)
    if args.oneply1:
        print("  Model1 using 1-ply lookahead")

    # Load model2 (or use random / gnubg)
    if args.model2 == "random":
        agent2 = RandomAgent()
        model2_label = "RandomAgent"
    elif args.model2 == "gnubg":
        agent2 = GnubgNNAgent(plies=args.plies, cubeless_money=cubeless_money)
        mode_tag = ", cubeless-money" if cubeless_money else ""
        model2_label = f"GnubgNN ({args.plies}-ply{mode_tag})"
    elif args.model2 == "gnubg-cubeful":
        agent2 = GnubgNNCubefulAgent(plies=args.plies)
        model2_label = f"GnubgNN-cubeful ({args.plies}-ply)"
    else:
        print(f"Loading model2: {args.model2}")
        net2 = TDNetwork.load(args.model2)
        agent2 = TDAgent(net2, oneply=args.oneply2)
        model2_label = args.model2
        if args.oneply2:
            print("  Model2 using 1-ply lookahead")

    print(f"\nPlaying {args.games} games...")
    print(f"  Model1: {args.model1}")
    print(f"  Model2: {model2_label}")
    if args.workers > 1:
        print(f"  Workers: {args.workers}")
    print()

    # Play all games with random first-mover assignment
    save_games = args.save_games is not None
    if args.workers > 1:
        if save_games:
            print("WARNING: --save-games not supported with --workers > 1, "
                  "falling back to sequential play")
            result = play_matches(
                agent1, agent2, args.games, record=True,
                cubeful=cubeful, jacoby=args.jacoby,
            )
            agent1_wins, agent2_wins, agent1_equity, agent1_equity_sq, records = result
        else:
            model2_type = (
                "random" if args.model2 == "random"
                else "gnubg" if args.model2 == "gnubg"
                else "gnubg-cubeful" if args.model2 == "gnubg-cubeful"
                else "model"
            )
            agent1_wins, agent2_wins, agent1_equity, agent1_equity_sq = play_matches_parallel(
                args.model1, args.model2, args.games,
                workers=args.workers, model2_type=model2_type, plies=args.plies,
                cubeless_money=cubeless_money,
                cubeful=cubeful, jacoby=args.jacoby,
                oneply1=args.oneply1, oneply2=args.oneply2,
            )
    else:
        if save_games:
            result = play_matches(
                agent1, agent2, args.games, record=True,
                cubeful=cubeful, jacoby=args.jacoby,
            )
            agent1_wins, agent2_wins, agent1_equity, agent1_equity_sq, records = result
        else:
            agent1_wins, agent2_wins, agent1_equity, agent1_equity_sq = play_matches(
                agent1, agent2, args.games,
                cubeful=cubeful, jacoby=args.jacoby,
            )

    # Save games as .mat files if requested
    if save_games:
        from gnubg_eval import export_mat
        os.makedirs(args.save_games, exist_ok=True)
        for i, rec in enumerate(records):
            mat_content = export_mat(
                rec, game_id=i + 1, money_game=cubeful,
            )
            mat_path = os.path.join(args.save_games, f"game_{i+1}.mat")
            with open(mat_path, "w") as f:
                f.write(mat_content)
        print(f"Saved {len(records)} games to {args.save_games}/")

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
    if cubeless_money or cubeful:
        import math as _math
        eq_per_game = agent1_equity / total_games
        # Sample variance from collected stakes (much tighter than worst-case bound)
        mean_sq = agent1_equity_sq / total_games
        var = mean_sq - eq_per_game * eq_per_game
        if var < 0:
            var = 0.0
        se = _math.sqrt(var / total_games)
        ci_half = 1.96 * se
        ci_lo = eq_per_game - ci_half
        ci_hi = eq_per_game + ci_half
        # Two-sided z-test p-value for "equity != 0"
        z = eq_per_game / se if se > 0 else 0.0
        try:
            from math import erf, sqrt
            pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        except Exception:
            pval = float("nan")
        tag = "Cubeful money" if cubeful else "Cubeless money"
        print()
        print(f"{tag} equity (Model1): "
              f"{eq_per_game:+.4f} pts/game ({eq_per_game*1000:+.1f} mEq/game)")
        print(f"  95% CI: [{ci_lo*1000:+.1f}, {ci_hi*1000:+.1f}] mEq/game "
              f"(±{ci_half*1000:.1f})")
        print(f"  Equity p-value: {pval:.4f}")
        if ci_lo > 0:
            print("  Equity verdict: Model1 significantly stronger (p<0.05)")
        elif ci_hi < 0:
            print("  Equity verdict: Model2 significantly stronger (p<0.05)")
        else:
            print("  Equity verdict: No significant equity difference")
    print("=" * 60)


if __name__ == "__main__":
    main()
