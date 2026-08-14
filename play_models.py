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
    WHITE,
    get_legal_plays, switch_turn, opening_roll,
)
from agents import Agent, RandomAgent, GnubgNNAgent
from td_agent import TDAgent
from model import ProbNetwork, load_model
from prob_agent import ProbAgent


def _load_agent(path: str, oneply: bool = False, twoply_k: int = 0,
                bf16: bool = False) -> Agent:
    """Build the right agent for a checkpoint, dispatching on model_type.

    prob5 checkpoints -> ProbAgent (cubeless money equity, like the
    scalar cubeless-money models); everything else -> TDAgent. twoply_k>0
    maps to ProbAgent 2-ply (it uses its own prune_threshold, not
    top-K). bf16 enables bf16 inference for ProbAgent (~1.5x faster);
    ignored for TDAgent.

    ProbAgent has no DMP switch: like scalar TDAgents, prob5 selects moves
    by its trained money equity even in DMP games — game_mode dmp only
    changes the SCORING to win/loss, never the move selection (see
    tests/test_prob5.py::test_prob5_uses_money_equity_not_dmp).
    """
    net = load_model(path)
    if isinstance(net, ProbNetwork):
        plies = 2 if twoply_k > 0 else (1 if oneply else 0)
        return ProbAgent(net, plies=plies, bf16_inference=bf16)
    return TDAgent(net, oneply=oneply, twoply_k=twoply_k)


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
        from gnubg_eval import play_and_record_cubeful
        # When caller doesn't want the GameRecord, skip per-move Play
        # bookkeeping — avoids a get_legal_plays call per turn (the
        # dominant remaining cost after the C cubeful fast path).
        game_record = play_and_record_cubeful(
            agent_white, agent_black, jacoby=jacoby,
            record_moves=record,
        )
        winner = game_record.winner
        if game_record.ended_by_drop:
            stake = game_record.cube_value
        else:
            stake = game_record.cube_value * game_record.result
        if record:
            return winner, stake, game_record
        return winner, stake

    if record:
        # Recording path: gnubg_eval.play_and_record is the one recording
        # game loop (same opening rule + dice logic; also recovers the Play
        # for successor-only agents so export_mat can render moves).
        from gnubg_eval import play_and_record
        game_record = play_and_record(agent_white, agent_black)
        return game_record.winner, game_record.result, game_record

    # Standard opening: random opener, no doublet on the first roll
    # (matches the cubeful path's opening_roll()); subsequent turns roll
    # normally (doublets allowed mid-game).
    state, opening_dice = opening_roll()
    is_opening = True

    while not state.is_game_over():
        if is_opening:
            d1, d2 = opening_dice
            is_opening = False
        else:
            d1, d2 = random.randint(1, 6), random.randint(1, 6)
        plays = get_legal_plays(state, (d1, d2))

        if plays:
            agent = agent_white if state.turn == WHITE else agent_black
            play, next_state = agent.choose_checker_action(state, (d1, d2), plays)
            # next_state already has turn switched (engine convention)
            state = next_state
        else:
            state = switch_turn(state)

    return state.winner(), state.game_result()


def _play_one_signed(agent1, agent2, cubeful, jacoby, record=False,
                     a1_is_white=None):
    """Play one game. Side assignment is random unless ``a1_is_white``
    is given (fixed colors, e.g. for XG imports that bind one agent per
    color — fair because the opening roll still randomizes who moves
    first). Returns (signed_equity, game_record_or_None) where
    signed_equity is positive when agent1 wins."""
    if a1_is_white is None:
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


def _sage_filter_kw():
    """SAGE_UNPRUNED=1 disables Open Sage's root move filter (both the
    cubeless agent and the cubeful analyzer) in every process that builds
    a side — env vars propagate to spawned H-H workers."""
    if os.environ.get("SAGE_UNPRUNED", "0") == "1":
        return {"filter_max_moves": 1000, "filter_threshold": 10.0}
    return {}


def _build_side(spec, spec_type, *, plies, sage_plies, cubeless_money,
                cubeful, oneply, twoply_k, bf16):
    """Build an agent for one side of a head-to-head from a spec.

    `spec_type` is one of "model" (then `spec` is a checkpoint path), "gnubg",
    "opensage", or "random". `plies` is the gnubg search depth;
    `sage_plies` is Open Sage's ply (1=0-ply/1P, 2=1-ply/2P). This lets EITHER
    side be an engine (e.g. gnubg vs Open Sage), not just model2.
    """
    if spec_type == "random":
        return RandomAgent()
    if spec_type == "gnubg":
        return GnubgNNAgent(plies=plies, cubeless_money=cubeless_money)
    if spec_type == "gnubg-ext":
        # TRUE gnubg 1.08 via the external-player interface: production
        # config (pruning nets + Normal filter), native money cube +
        # Jacoby (no 21pt approximation, no cube clamp). One gnubg
        # subprocess per worker. See gnubg_external.py.
        from gnubg_external import GnubgExternalAgent
        return GnubgExternalAgent(plies=plies, cubeless=cubeless_money)
    if spec_type == "gnubg-ext-full":
        # gnubg 1.08 totally unpruned: no pruning nets, keep-all move
        # filter (every legal move deep-evaluated — the full-width
        # search of the historical gnubg-nn wrapper, ~4x slower).
        from gnubg_external import GnubgExternalAgent
        return GnubgExternalAgent(plies=plies, cubeless=cubeless_money,
                                  full=True)
    if spec_type == "gnubg-ext-rootfull":
        # Every legal move deep-evaluated at the root, gnubg's default
        # pruning for the opponent's replies (pruning nets stay ON) --
        # the paper's stated protocol for matched 2-ply comparisons.
        from gnubg_external import GnubgExternalAgent
        return GnubgExternalAgent(plies=plies, cubeless=cubeless_money,
                                  root_full=True)
    if spec_type == "opensage":
        if cubeful:
            raise ValueError(
                "opensage is cubeless/DMP only; for cubeful-money use "
                "'opensage-analyzer' (Sage's self-play-identical, cube-aware "
                "config).")
        from experiments.opensage_agents import OpenSageAgent
        return OpenSageAgent(
            sage_plies=sage_plies,
            mode="cubeless-money" if cubeless_money else "dmp",
            **_sage_filter_kw(),
        )
    if spec_type == "opensage-analyzer":
        # Self-play-identical Open Sage (BgBotAnalyzer: cube-aware checker
        # play + native cube_action). Cubeful-money only.
        if not cubeful:
            raise ValueError("opensage-analyzer is cubeful-money only; "
                             "use 'opensage' for cubeless/DMP")
        from experiments.opensage_agents import OpenSageAnalyzerAgent
        return OpenSageAnalyzerAgent(sage_plies=sage_plies, **_sage_filter_kw())
    return _load_agent(spec, oneply=oneply, twoply_k=twoply_k, bf16=bf16)


def _seed_worker_rng():
    """Pool initializer: explicitly seed this worker's RNG from OS entropy.

    With the 'spawn' start method Python already auto-seeds each fresh
    interpreter, so this is belt-and-braces — it guarantees independent
    dice streams per worker/process/machine even if the start method is
    ever changed to 'fork' (where children inherit the parent's RNG
    state) or a deterministic default seed is introduced upstream.
    Independent streams are what makes pooling runs from several
    machines (combine_head_to_head) statistically valid.
    """
    random.seed(int.from_bytes(os.urandom(16), "little"))


def _play_matches_worker(args):
    """Worker function for parallel game playing."""
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch as _torch
    _torch.set_num_threads(1)
    (model1_path, model2_path, num_games, model1_type, model2_type, plies,
     sage_plies, cubeless_money, cubeful, jacoby, oneply1, oneply2,
     twoply_k1, twoply_k2, extreme_threshold, bf16) = args

    agent1 = _build_side(
        model1_path, model1_type, plies=plies, sage_plies=sage_plies,
        cubeless_money=cubeless_money, cubeful=cubeful,
        oneply=oneply1, twoply_k=twoply_k1, bf16=bf16,
    )
    agent2 = _build_side(
        model2_path, model2_type, plies=plies, sage_plies=sage_plies,
        cubeless_money=cubeless_money, cubeful=cubeful,
        oneply=oneply2, twoply_k=twoply_k2, bf16=bf16,
    )

    a1_wins = 0
    a2_wins = 0
    a1_equity = 0
    per_game = []
    extremes = []
    capture_extremes = extreme_threshold > 0 and cubeful
    for _ in range(num_games):
        signed, rec = _play_one_signed(
            agent1, agent2, cubeful, jacoby, record=capture_extremes,
        )
        if signed > 0:
            a1_wins += 1
        else:
            a2_wins += 1
        a1_equity += signed
        per_game.append(signed)
        if capture_extremes and abs(signed) >= extreme_threshold:
            extremes.append((signed, rec))
    return a1_wins, a2_wins, a1_equity, per_game, extremes


def play_matches(
    agent1: Agent,
    agent2: Agent,
    num_games: int,
    print_progress: bool = True,
    record: bool = False,
    cubeful: bool = False,
    jacoby: bool = True,
    a1_is_white: bool = None,
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
    per_game = []
    records = [] if record else None
    start_time = time.time()
    progress_interval = max(100, num_games // 10)

    for i in range(1, num_games + 1):
        signed, game_record = _play_one_signed(
            agent1, agent2, cubeful, jacoby, record=record,
            a1_is_white=a1_is_white,
        )
        if record and game_record is not None:
            records.append(game_record)
        if signed > 0:
            agent1_wins += 1
        else:
            agent2_wins += 1
        agent1_equity += signed
        per_game.append(signed)

        if print_progress and i % progress_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"  Game {i:>5d}/{num_games}: "
                f"Model1 {agent1_wins:>4d}  Model2 {agent2_wins:>4d}  "
                f"({elapsed:.1f}s)",
                flush=True,  # visible in nohup/piped logs during multi-day runs
            )

    if record:
        return agent1_wins, agent2_wins, agent1_equity, per_game, records
    return agent1_wins, agent2_wins, agent1_equity, per_game


def play_matches_parallel(
    model1_path: str,
    model2_path: str,
    num_games: int,
    workers: int,
    model2_type: str = "model",
    model1_type: str = "model",
    plies: int = 0,
    sage_plies: int = 1,
    cubeless_money: bool = False,
    cubeful: bool = False,
    jacoby: bool = True,
    oneply1: bool = False,
    oneply2: bool = False,
    twoply_k1: int = 0,
    twoply_k2: int = 0,
    extreme_threshold: float = 0.0,
    bf16: bool = False,
):
    """Play games in parallel across multiple workers.
    Returns (a1_wins, a2_wins, eq_sum, per_game, extremes).
    `extremes` is a list of (signed_equity, GameRecord) for games with
    |equity| >= extreme_threshold (empty if threshold == 0).

    Either side may be an engine: `model1_type`/`model2_type` in
    {"model","gnubg","opensage","random"}. `plies` is the
    gnubg depth; `sage_plies` is Open Sage's ply (1=0-ply, 2=1-ply).
    """
    # Split games across workers
    base = num_games // workers
    remainder = num_games % workers
    game_counts = [base + (1 if i < remainder else 0) for i in range(workers)]

    args_list = [
        (model1_path, model2_path, n, model1_type, model2_type, plies,
         sage_plies, cubeless_money, cubeful, jacoby, oneply1, oneply2,
         twoply_k1, twoply_k2, extreme_threshold, bf16)
        for n in game_counts
    ]

    ctx = mp.get_context('spawn')
    start_time = time.time()
    with ctx.Pool(workers, initializer=_seed_worker_rng) as pool:
        results = pool.map(_play_matches_worker, args_list)

    elapsed = time.time() - start_time
    total_a1 = sum(r[0] for r in results)
    total_a2 = sum(r[1] for r in results)
    total_eq = sum(r[2] for r in results)

    all_per_game = []
    for r in results:
        all_per_game.extend(r[3])
    extremes = []
    for r in results:
        extremes.extend(r[4])
    print(f"  {num_games} games completed in {elapsed:.1f}s "
          f"({num_games/elapsed:.0f} games/sec, {workers} workers)")

    return total_a1, total_a2, total_eq, all_per_game, extremes


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

    Returns one of:
    1. "Model1 is significantly stronger (p<0.05)" — CI entirely above 50%
    2. "Model2 is significantly stronger (p<0.05)" — CI entirely below 50%
    3. "Models are effectively tied (difference < 1%)" — CI includes 50%
       AND CI width is tight (< 2%)
    4. "Inconclusive — need more games to determine winner" — CI includes
       50% but too wide to call
    """
    lower, upper = wilson_ci(wins, total, z=1.96)
    ci_width = upper - lower

    # Check if CI is entirely above 50%
    if lower > 0.50:
        return "Model1 is significantly stronger (p<0.05)"

    # Check if CI is entirely below 50%
    if upper < 0.50:
        return "Model2 is significantly stronger (p<0.05)"

    # CI includes 50%. Only call "tied" if we have enough data
    # (tight CI) AND the point estimate is close to 50%.
    if ci_width < 0.02 and lower >= 0.49 and upper <= 0.51:
        return "Models are effectively tied (difference < 1%)"

    # Otherwise, inconclusive — CI includes 50% but too wide or
    # point estimate leans enough that more games could resolve it
    return "Inconclusive — need more games to determine winner"


def report_head_to_head(
    agent1_wins, agent2_wins, agent1_equity, per_game,
    *, cubeless_money=False, cubeful=False, n_bootstrap=10000, do_print=True,
):
    """Summarize a head-to-head result. Returns a dict of all reported
    statistics; prints the human-readable RESULTS block when ``do_print``.

    Single source of truth for H-H reporting — both ``main()`` (CLI) and
    ``run_head_to_head()`` (library) call this, so their output stays in sync.
    """
    total_games = agent1_wins + agent2_wins
    agent1_pct = 100 * agent1_wins / total_games
    agent2_pct = 100 * agent2_wins / total_games

    pvalue = compute_binomial_pvalue(agent1_wins, total_games)
    lower_ci, upper_ci = wilson_ci(agent1_wins, total_games, z=1.96)
    verdict = get_verdict(agent1_wins, total_games)

    result = {
        "agent1_wins": agent1_wins, "agent2_wins": agent2_wins,
        "total_games": total_games,
        "agent1_pct": agent1_pct, "agent2_pct": agent2_pct,
        "win_ci_pct": (lower_ci * 100, upper_ci * 100),
        "win_pvalue": pvalue, "verdict": verdict,
        "cubeless_money": cubeless_money, "cubeful": cubeful,
    }

    if do_print:
        print()
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
        import numpy as np
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        from stats import (
            bootstrap_ci, bootstrap_ci_statistic, trimmed_mean, capped_mean,
        )
        eq_per_game = agent1_equity / total_games
        tag = "Cubeful money" if cubeful else "Cubeless money"
        result["eq_per_game_pts"] = eq_per_game
        result["eq_mEq_per_game"] = eq_per_game * 1000
        if do_print:
            print()
            print(f"{tag} equity (Model1): "
                  f"{eq_per_game:+.4f} pts/game ({eq_per_game*1000:+.1f} mEq/game)")

        # Share one ProcessPool across all 2-4 bootstrap calls; saves
        # the ~0.2-0.3s pool-startup hit per call.
        n_jobs = min(os.cpu_count() or 1, 32)
        with ProcessPoolExecutor(max_workers=n_jobs) as boot_pool:
            ci_lo, ci_hi, pval = bootstrap_ci(
                per_game, n_boot=n_bootstrap, executor=boot_pool,
            )
            ci_half = (ci_hi - ci_lo) / 2
            if ci_lo > 0:
                eq_verdict = "Model1 significantly stronger (p<0.05)"
            elif ci_hi < 0:
                eq_verdict = "Model2 significantly stronger (p<0.05)"
            else:
                eq_verdict = "No significant equity difference"
            result["equity_ci_mEq"] = (ci_lo * 1000, ci_hi * 1000)
            result["equity_pvalue"] = pval
            result["equity_verdict"] = eq_verdict
            if do_print:
                print(f"  95% CI (bootstrap): [{ci_lo*1000:+.1f}, {ci_hi*1000:+.1f}] mEq/game "
                      f"(±{ci_half*1000:.1f})")
                print(f"  Equity p-value (bootstrap): {pval:.4f}")
                print(f"  Equity verdict: {eq_verdict}")

            # Robust estimators (relevant when cube escalation creates
            # heavy-tailed per-game point distributions).
            per_game_arr = np.asarray(per_game, dtype=np.float64)

            med = float(np.median(per_game_arr))
            m_lo, m_hi, m_p = bootstrap_ci_statistic(
                per_game_arr, np.median,
                n_boot=n_bootstrap, executor=boot_pool,
            )
            result["median_mEq"] = med * 1000
            result["median_ci_mEq"] = (m_lo * 1000, m_hi * 1000)
            result["median_pvalue"] = m_p
            if do_print:
                print(f"  Median equity:   {med*1000:+.1f} mEq/game  "
                      f"[{m_lo*1000:+.1f}, {m_hi*1000:+.1f}]  p={m_p:.4f}")

            tmean = trimmed_mean(per_game_arr, trim=0.01)
            t_lo, t_hi, t_p = bootstrap_ci_statistic(
                per_game_arr,
                partial(trimmed_mean, trim=0.01),
                n_boot=n_bootstrap, executor=boot_pool,
            )
            result["trimmed_mean_mEq"] = tmean * 1000
            result["trimmed_ci_mEq"] = (t_lo * 1000, t_hi * 1000)
            result["trimmed_pvalue"] = t_p
            if do_print:
                print(f"  Trimmed mean (1%): {tmean*1000:+.1f} mEq/game  "
                      f"[{t_lo*1000:+.1f}, {t_hi*1000:+.1f}]  p={t_p:.4f}")

            if cubeful:
                # Capped/Winsorized mean — equity under a per-game point cap.
                cap_pts = 128.0
                cmean = capped_mean(per_game_arr, cap=cap_pts)
                c_lo, c_hi, c_p = bootstrap_ci_statistic(
                    per_game_arr,
                    partial(capped_mean, cap=cap_pts),
                    n_boot=n_bootstrap, executor=boot_pool,
                )
                result["capped_mean_mEq"] = cmean * 1000
                result["capped_ci_mEq"] = (c_lo * 1000, c_hi * 1000)
                result["capped_pvalue"] = c_p
                if do_print:
                    print(f"  Capped mean (±{int(cap_pts)}): {cmean*1000:+.1f} mEq/game  "
                          f"[{c_lo*1000:+.1f}, {c_hi*1000:+.1f}]  p={c_p:.4f}")
    if do_print:
        print("=" * 60)
    return result


def _detect_side_type(spec):
    """Map a side spec to its type. Engine keywords resolve to engines;
    anything else is treated as a checkpoint path ("model"). An existing
    file always wins over a keyword (a checkpoint literally named "gnubg"
    is a model, not the engine); pass model*_type explicitly to override."""
    if os.path.exists(spec):
        return "model"
    return {
        "random": "random", "gnubg": "gnubg", "opensage": "opensage",
        "gnubg-ext": "gnubg-ext", "gnubg-ext-full": "gnubg-ext-full",
        "gnubg-ext-rootfull": "gnubg-ext-rootfull",
        "opensage-analyzer": "opensage-analyzer",
    }.get(spec, "model")


def save_head_to_head(out_dir, per_game, *, game_mode, side1, side2, n_games):
    """Persist one run's raw per-game equities (+ metadata) to ``out_dir`` as a
    unique ``.npz`` file. Keeping the raw per-game values lets multiple runs
    (even across machines — just copy the files together) be pooled later for a
    larger, exact bootstrap CI. ``per_game[i] > 0`` means side1 won game i;
    its magnitude is the cubeless/cubeful stake.
    """
    import platform
    import sys
    import time
    import uuid
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"hh_{int(time.time())}_{uuid.uuid4().hex[:8]}.npz")
    np.savez(
        path,
        per_game=np.asarray(per_game, dtype=np.float64),
        game_mode=str(game_mode), side1=str(side1), side2=str(side2),
        n_games=int(n_games),
        # Provenance: enough to reconstruct the exact configuration later
        # (added after a replication audit where a published run's Sage
        # filter config could not be confirmed from its raws).
        argv=" ".join(sys.argv),
        sage_unpruned=os.environ.get("SAGE_UNPRUNED", "0"),
        hostname=platform.node(),
    )
    return path


def combine_head_to_head(out_dir, *, n_bootstrap=10000, do_print=True):
    """Pool every saved run in ``out_dir`` (written by :func:`save_head_to_head`)
    and report the combined head-to-head over the union of games — i.e. extend a
    run, or merge runs from several machines, into one larger exact bootstrap CI.

    All runs in the directory must share the same game mode and matchup (side1,
    side2); mixing them is an error. Returns the same result dict as
    :func:`report_head_to_head`.
    """
    import glob
    import numpy as np
    files = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no saved H-H runs (*.npz) in {out_dir!r}")
    parts, modes, s1s, s2s = [], set(), set(), set()
    seen_hashes = {}
    for f in files:
        d = np.load(f, allow_pickle=False)
        parts.append(d["per_game"])
        modes.add(str(d["game_mode"]))
        s1s.add(str(d["side1"]))
        s2s.add(str(d["side2"]))
        # Duplicate-stream guard: two runs with identical per-game
        # sequences means a shared RNG seed (e.g. a replayed command with
        # deterministic seeding) — pooling them would double-count games
        # and silently shrink the CI. Refuse rather than warn.
        import hashlib
        h = hashlib.sha256(
            np.ascontiguousarray(d["per_game"]).tobytes()).hexdigest()
        if h in seen_hashes:
            raise ValueError(
                f"duplicate game stream: {f} is identical to "
                f"{seen_hashes[h]} — same RNG seed? Remove one before "
                f"pooling.")
        seen_hashes[h] = f
    if len(modes) > 1:
        raise ValueError(f"mixed game modes in {out_dir}: {sorted(modes)}")
    if len(s1s) > 1 or len(s2s) > 1:
        raise ValueError(
            f"mixed matchups in {out_dir}: side1={sorted(s1s)} side2={sorted(s2s)}")
    game_mode = modes.pop()
    pooled = np.concatenate(parts)
    cubeless_money = game_mode == "cubeless-money"
    cubeful = game_mode == "cubeful-money"
    agent1_wins = int((pooled > 0).sum())
    agent2_wins = int((pooled <= 0).sum())
    agent1_equity = float(pooled.sum())
    if do_print:
        print(f"Pooled {len(files)} run(s) -> {len(pooled)} games "
              f"({s1s.pop()} vs {s2s.pop()}, {game_mode})")
    return report_head_to_head(
        agent1_wins, agent2_wins, agent1_equity, pooled,
        cubeless_money=cubeless_money, cubeful=cubeful,
        n_bootstrap=n_bootstrap, do_print=do_print,
    )


def run_head_to_head(
    model1, model2, *, num_games, workers=1, game_mode="dmp", plies=0,
    sage_plies=1, jacoby=True, oneply1=False, oneply2=False,
    twoply_k1=0, twoply_k2=0, bf16=False, n_bootstrap=10000,
    model1_type=None, model2_type=None, save=None, do_print=True,
    runaway_threshold=0.0,
):
    """Run a head-to-head between two sides and return a result dict.

    Thin orchestration over :func:`play_matches_parallel` +
    :func:`report_head_to_head`; agents are built by the same code path as the
    CLI. EITHER side may be a checkpoint path or an engine keyword
    (``"gnubg"``/``"gnubg-ext"``/``"opensage"``/``"random"``), auto-detected
    unless ``model1_type``/``model2_type`` are given. ``plies`` is gnubg's
    depth; ``sage_plies`` is Open Sage's ply (1=0-ply, 2=1-ply).
    ``game_mode`` is one of ``"dmp"``, ``"cubeless-money"``, ``"cubeful-money"``.
    """
    cubeless_money = game_mode == "cubeless-money"
    cubeful = game_mode == "cubeful-money"
    if model1_type is None:
        model1_type = _detect_side_type(model1)
    if model2_type is None:
        model2_type = _detect_side_type(model2)
    (agent1_wins, agent2_wins, agent1_equity, per_game,
     extremes) = play_matches_parallel(
        model1, model2, num_games, workers=max(1, workers),
        model1_type=model1_type, model2_type=model2_type,
        plies=plies, sage_plies=sage_plies,
        cubeless_money=cubeless_money, cubeful=cubeful, jacoby=jacoby,
        oneply1=oneply1, oneply2=oneply2,
        twoply_k1=twoply_k1, twoply_k2=twoply_k2, bf16=bf16,
        extreme_threshold=runaway_threshold,
    )
    result = report_head_to_head(
        agent1_wins, agent2_wins, agent1_equity, per_game,
        cubeless_money=cubeless_money, cubeful=cubeful,
        n_bootstrap=n_bootstrap, do_print=do_print,
    )
    if save:
        path = save_head_to_head(
            save, per_game, game_mode=game_mode,
            side1=model1, side2=model2, n_games=num_games,
        )
        if do_print:
            print(f"Saved {len(per_game)} raw per-game results -> {path}")
    if extremes:
        # Log runaway / high-stake games in full (moves + cube actions)
        # as .mat, so redouble-loop games can be inspected in gnubg/XG.
        from gnubg_eval import export_mat
        out_dir = save if save else "."
        rec_dir = os.path.join(out_dir, "runaway_games")
        os.makedirs(rec_dir, exist_ok=True)
        extremes.sort(key=lambda sr: -abs(sr[0]))
        for k, (signed, rec) in enumerate(extremes, 1):
            fn = os.path.join(
                rec_dir,
                f"runaway_{k:03d}_cube{rec.cube_value}_stake{int(abs(signed))}.mat")
            with open(fn, "w") as f:
                f.write(export_mat(rec, game_id=k, money_game=True))
        if do_print:
            print(f"Logged {len(extremes)} runaway game(s) "
                  f"(|stake| >= {runaway_threshold:g}) -> {rec_dir}")
    return result


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
        help="Gnubg evaluation depth when using --gnubg or a gnubg-ext side "
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
        "--save-hh",
        type=str,
        default=None,
        help="Directory to save the raw per-game equity vector as .npz, so the "
             "exact bootstrap CI can be recreated or pooled across runs later "
             "(see combine_head_to_head).",
    )
    parser.add_argument(
        "--save-extreme-games",
        type=str,
        default=None,
        help="Directory to save only games whose |equity| >= --extreme-threshold "
             "(works with --workers > 1; cubeful only).",
    )
    parser.add_argument(
        "--extreme-threshold",
        type=float,
        default=128.0,
        help="Equity magnitude threshold for --save-extreme-games (default: 128 pts).",
    )
    parser.add_argument(
        "--game-mode",
        choices=["dmp", "cubeless-money", "cubeful-money"],
        default="dmp",
        help="dmp: score by win/loss. "
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
    parser.add_argument(
        "--twoply-k1", type=int, default=0,
        help="2-ply checker/cube decisions for model1 with top-K 1-ply "
             "filter at the depth-2 expansion (0 = off). Works for both "
             "cubeful-money and dmp/cubeless modes. Takes precedence over "
             "--oneply1 for move selection.",
    )
    parser.add_argument(
        "--twoply-k2", type=int, default=0,
        help="2-ply checker/cube decisions for model2 with top-K 1-ply "
             "filter (0 = off; ignored when model2 is gnubg or random).",
    )
    parser.add_argument(
        "--sage-plies", type=int, default=1,
        help="Open Sage ply when --model2 opensage (1 = 0-ply/1P, "
             "2 = 1-ply/2P; default 1).",
    )
    parser.add_argument(
        "--model1-white", action="store_true",
        help="Pin model1 to WHITE every game instead of randomizing "
             "colors (for XG imports that bind one agent per color). "
             "First mover is still decided by the opening roll, so this "
             "introduces no first-move bias. Sequential play only.",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap resamples for equity CI (default: 10000).",
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="Use bf16 inference for prob5 (ProbAgent) models (~1.5x faster, "
             "negligible strength loss; ignored for scalar models).",
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
    elif args.model2 is None:
        parser.error("--model2 is required unless --random or --gnubg is specified")
    if args.model2 == "opensage" and args.sage_plies < 1:
        parser.error("--sage-plies must be >= 1 (1 = 0-ply/1P convention)")

    # Load model1
    print(f"Loading model1: {args.model1}")
    agent1 = _load_agent(args.model1, oneply=args.oneply1, twoply_k=args.twoply_k1, bf16=args.bf16)
    if args.twoply_k1 > 0:
        print(f"  Model1 using 2-ply lookahead (K={args.twoply_k1})")
    elif args.oneply1:
        print("  Model1 using 1-ply lookahead")

    # Load model2 (engine keyword or checkpoint path) via the same
    # dispatch the parallel workers use (_detect_side_type + _build_side).
    model2_type = _detect_side_type(args.model2)
    if model2_type == "model":
        print(f"Loading model2: {args.model2}")
    try:
        agent2 = _build_side(
            args.model2, model2_type, plies=args.plies,
            sage_plies=args.sage_plies, cubeless_money=cubeless_money,
            cubeful=cubeful, oneply=args.oneply2, twoply_k=args.twoply_k2,
            bf16=args.bf16,
        )
    except ValueError as e:
        parser.error(str(e))
    mode_tag = ", cubeless-money" if cubeless_money else ""
    model2_label = {
        "random": "RandomAgent",
        "gnubg": f"GnubgNN ({args.plies}-ply{mode_tag})",
        "gnubg-ext": f"gnubg-ext ({args.plies}-ply, real gnubg 1.08)",
        "gnubg-ext-full": f"gnubg-ext-full ({args.plies}-ply, unpruned)",
        "gnubg-ext-rootfull": (f"gnubg-ext-rootfull ({args.plies}-ply, "
                               "keep-all root, default reply pruning)"),
        "opensage": f"OpenSage (sage_plies={args.sage_plies})",
        "opensage-analyzer": f"OpenSage-analyzer (sage_plies={args.sage_plies})",
    }.get(model2_type, args.model2)
    if model2_type == "model":
        if args.twoply_k2 > 0:
            print(f"  Model2 using 2-ply lookahead (K={args.twoply_k2})")
        elif args.oneply2:
            print("  Model2 using 1-ply lookahead")

    _has_prob5 = isinstance(agent1, ProbAgent) or isinstance(agent2, ProbAgent)
    if cubeful and _has_prob5:
        parser.error("prob5 (ProbAgent) is cubeless-only and has no cube policy; "
                     "drop cubeful mode for prob5 models.")
    # prob5 in DMP mode: like scalar TDAgents, ProbAgent keeps selecting by
    # its trained money equity; game_mode dmp only switches the scoring to
    # win/loss.

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
                a1_is_white=(True if args.model1_white else None),
            )
            agent1_wins, agent2_wins, agent1_equity, per_game, records = result
        else:
            extreme_threshold = args.extreme_threshold if args.save_extreme_games else 0.0
            (agent1_wins, agent2_wins, agent1_equity,
             per_game, extremes) = play_matches_parallel(
                args.model1, args.model2, args.games,
                workers=args.workers, model2_type=model2_type, plies=args.plies,
                sage_plies=args.sage_plies,
                cubeless_money=cubeless_money,
                cubeful=cubeful, jacoby=args.jacoby,
                oneply1=args.oneply1, oneply2=args.oneply2,
                twoply_k1=args.twoply_k1, twoply_k2=args.twoply_k2,
                extreme_threshold=extreme_threshold, bf16=args.bf16,
            )

            if args.save_extreme_games and extremes:
                from gnubg_eval import export_mat
                os.makedirs(args.save_extreme_games, exist_ok=True)
                extremes.sort(key=lambda x: -abs(x[0]))
                for i, (signed, rec) in enumerate(extremes):
                    mat = export_mat(rec, game_id=i + 1, money_game=cubeful)
                    name = f"extreme_{i+1:03d}_eq{int(signed):+d}.mat"
                    with open(os.path.join(args.save_extreme_games, name), "w") as f:
                        f.write(mat)
                print(f"Saved {len(extremes)} extreme games "
                      f"(|eq| >= {extreme_threshold}) to {args.save_extreme_games}/")
    else:
        if save_games:
            result = play_matches(
                agent1, agent2, args.games, record=True,
                cubeful=cubeful, jacoby=args.jacoby,
                a1_is_white=(True if args.model1_white else None),
            )
            agent1_wins, agent2_wins, agent1_equity, per_game, records = result
        else:
            agent1_wins, agent2_wins, agent1_equity, per_game = play_matches(
                agent1, agent2, args.games,
                cubeful=cubeful, jacoby=args.jacoby,
                a1_is_white=(True if args.model1_white else None),
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

    report_head_to_head(
        agent1_wins, agent2_wins, agent1_equity, per_game,
        cubeless_money=cubeless_money, cubeful=cubeful,
        n_bootstrap=args.n_bootstrap,
    )

    if args.save_hh:
        path = save_head_to_head(
            args.save_hh, per_game, game_mode=args.game_mode,
            side1=args.model1, side2=model2_label, n_games=len(per_game),
        )
        print(f"Saved {len(per_game)} raw per-game equities -> {path}")


if __name__ == "__main__":
    main()
