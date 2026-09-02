"""
gnubg_eval_matchplay.py -- mEMG error rates for matchplay self-play.

Plays full N-point matches with a matchplay TDAgent (self-play),
exports each match as a multi-game Jellyfish .mat file with real
match scores per game, and runs gnubg batch analysis at the actual
score — gnubg's "Error rate mEMG (MWC)" is EMG-normalized per
unforced decision, the standard play-quality metric.

Reuses the recording/export/analysis primitives from gnubg_eval.py;
that module stays cubeless/money-only.

Usage:
    python gnubg_eval_matchplay.py --model models/foo.pt \
        --matches 100 --match-length 3 --plies 2 --workers 16
"""

import argparse
import multiprocessing as mp
import os
import random
import tempfile

import numpy as np

from backgammon_engine import (
    WHITE, get_legal_plays, opening_roll, switch_turn,
)
from gnubg_eval import (
    CubeRecord, GameRecord, MoveRecord, export_mat, run_gnubg_analysis,
)
from modes import MatchState


def play_and_record_match(agent_white, agent_black, match_length, rng):
    """Play one full N-point match, recording every game.

    Returns a list of (GameRecord, white_score_before, black_score_before)
    in game order. MatchState away scores are WHITE's view, so
    score = match_length - away.
    """
    ms = MatchState(
        jacoby=False, match_length=match_length,
        my_away=match_length, opp_away=match_length,
    )
    games = []
    while True:
        gms = ms.start_game()
        w_score = match_length - ms.my_away
        b_score = match_length - ms.opp_away
        rec = GameRecord()
        state, dice = opening_roll(rng)
        is_opening = True
        while True:
            player = state.turn
            actor = agent_white if player == WHITE else agent_black
            opp_agent = agent_black if player == WHITE else agent_white

            # ── Cube phase (never on the opening roll / Crawford) ──
            if not is_opening and gms.can_offer(player):
                offer = actor.offer_double(state, gms)
                if offer.should_double:
                    new_cv = gms.cube_value * 2
                    rec.moves.append(CubeRecord(
                        player=player, action="double",
                        new_cube_value=new_cv,
                    ))
                    hint = offer if opp_agent is actor else None
                    takes = opp_agent.respond_to_double(state, gms, hint=hint)
                    opponent = 1 - player
                    if takes:
                        rec.moves.append(CubeRecord(
                            player=opponent, action="take",
                            new_cube_value=new_cv,
                        ))
                        gms = gms.after_take(player)
                    else:
                        rec.moves.append(CubeRecord(
                            player=opponent, action="drop",
                            new_cube_value=new_cv,
                        ))
                        rec.winner = player
                        rec.result = 1
                        rec.cube_value = gms.cube_value  # pre-double
                        rec.ended_by_drop = True
                        ms = gms.after_game(1, winner_is_me=(player == WHITE))
                        break

            # ── Checker phase ──────────────────────────────────────
            result = actor.choose_checker_action_cubeful(state, dice, gms)
            if result is None:
                rec.moves.append(MoveRecord(
                    player=player, dice=dice, play=(),
                ))
                state = switch_turn(state)
            else:
                next_state = result[0]
                play = ()
                for p, s in get_legal_plays(state, dice):
                    if s == next_state:
                        play = p
                        break
                rec.moves.append(MoveRecord(
                    player=player, dice=dice, play=play,
                ))
                if next_state.is_game_over():
                    gr = next_state.game_result()
                    rec.winner = next_state.winner()
                    rec.result = gr
                    rec.cube_value = gms.cube_value
                    ms = gms.after_game(gr, winner_is_me=(player == WHITE))
                    break
                state = next_state
            dice = (rng.randint(1, 6), rng.randint(1, 6))
            is_opening = False

        games.append((rec, w_score, b_score))
        if ms.match_over():
            return games


def export_match_mat(games, match_length):
    """Multi-game .mat for one match. Reuses export_mat's game-body
    builder per game (its lines[3:] are " Game N", the score line,
    and the move rows) and swaps in the real running score."""
    out = ["", f" {match_length} point match", ""]
    for i, (rec, w_score, b_score) in enumerate(games, 1):
        body = export_mat(rec, game_id=i).split("\n")[3:]
        body[1] = f" white : {w_score:<22}black : {b_score}"
        out.extend(body)
    return "\n".join(out)


# ── Parallel generation / analysis ──────────────────────────────────


def _load_met(met_path):
    if not met_path:
        return None
    from met import MatchEquityTable
    return MatchEquityTable.from_xml(met_path)


def _gen_worker(args):
    (model_path, n_matches, match_length, work_dir, start_idx,
     seed, met_path) = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from model import TDNetwork
    from td_agent import TDAgent
    net = TDNetwork.load(model_path)
    agent = TDAgent(net, met=_load_met(met_path))
    rng = random.Random(seed)
    mat_files = []
    for i in range(n_matches):
        games = play_and_record_match(agent, agent, match_length, rng)
        path = os.path.join(work_dir, f"match_{start_idx + i:04d}.mat")
        with open(path, "w") as f:
            f.write(export_match_mat(games, match_length))
        mat_files.append(path)
    return mat_files


def _analyze_worker(args):
    mat_files, work_dir, plies = args
    return run_gnubg_analysis(
        mat_files, work_dir, verbose=False, plies=plies,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--met", default=None, metavar="TABLE",
                   help="Match equity table to PLAY with ('auto', a gnubg "
                        "MET name, or a path). Must match how the model "
                        "was trained. Note the analysis side (gnubg CLI) "
                        "uses its own default table, Kazaross-XG2.")
    p.add_argument("--matches", type=int, default=50)
    p.add_argument("--match-length", type=int, default=3)
    p.add_argument("--plies", type=int, default=2,
                   help="gnubg analysis depth (2 is ~10x faster than 3)")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel match-generation workers")
    p.add_argument("--gnubg-workers", type=int, default=8,
                   help="parallel gnubg analysis processes")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--keep-mats", default=None, metavar="DIR",
                   help="write .mat files here instead of a temp dir")
    args = p.parse_args()

    work_dir = args.keep_mats or tempfile.mkdtemp(prefix="gnubg_mp_")
    os.makedirs(work_dir, exist_ok=True)
    print(f"Generating {args.matches} self-play {args.match_length}pt "
          f"matches -> {work_dir}")

    per = [args.matches // args.workers] * args.workers
    for i in range(args.matches % args.workers):
        per[i] += 1
    from met import resolve_met_path
    met_path = resolve_met_path(args.met)
    if met_path:
        print(f"Playing with MET: {met_path}")
    gen_args = []
    start = 0
    for i, n in enumerate(per):
        if n:
            gen_args.append((args.model, n, args.match_length, work_dir,
                             start, args.seed + 1000 * i, met_path))
            start += n
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(gen_args)) as pool:
        mat_files = [f for chunk in pool.map(_gen_worker, gen_args)
                     for f in chunk]
    print(f"  {len(mat_files)} match files written")

    print(f"Analyzing with gnubg at {args.plies}-ply "
          f"({args.gnubg_workers} processes)...")
    n_chunks = min(args.gnubg_workers, len(mat_files))
    chunks = [mat_files[i::n_chunks] for i in range(n_chunks)]
    an_args = [(chunk, work_dir, args.plies) for chunk in chunks]
    with ctx.Pool(n_chunks) as pool:
        results = pool.map(_analyze_worker, an_args)

    chequer, cube, overall = [], [], []
    for c, k, o in results:
        chequer.extend(c)
        cube.extend(k)
        overall.extend(o)

    if len(overall) != len(mat_files):
        print(f"  WARNING: parsed {len(overall)} stat blocks for "
              f"{len(mat_files)} matches")

    # gnubg's "Error rate mEMG (MWC)" divides the MWC error by an
    # equivalent-money-game normalizer. In MATCH play that normalizer
    # collapses toward zero at degenerate scores (dead cube, 1-away),
    # so single player-matches can come back at 1e20+ and a plain mean
    # is meaningless -- observed live: a 300-match run reported a cube
    # mEMG of 1.5e16 whose median was in the normal range. Report robust
    # statistics and say how many samples were discarded.
    OUTLIER_MEMG = 1000.0  # far above "beginner" (>12); anything here is a blowup

    def report(name, rates):
        # Self-play: pool white and black (same agent both sides).
        vals = np.array([v for pair in rates for v in pair], dtype=np.float64)
        if not len(vals):
            print(f"  {name}: no data")
            return
        finite = vals[np.isfinite(vals)]
        keep = finite[finite < OUTLIER_MEMG]
        n_bad = len(vals) - len(keep)
        if not len(keep):
            print(f"  {name:8s} mEMG: ALL {len(vals)} samples were blowups")
            return
        se = keep.std(ddof=1) / np.sqrt(len(keep)) if len(keep) > 1 else float("nan")
        flag = f"  [{n_bad} blowups dropped]" if n_bad else ""
        print(f"  {name:8s} mEMG: mean {keep.mean():6.2f} ± {se:.2f}   "
              f"median {np.median(keep):6.2f}   "
              f"(n={len(keep)}/{len(vals)}){flag}")

    print(f"\nSelf-play mEMG, {args.match_length}pt matches, "
          f"gnubg {args.plies}-ply reference (lower is better):")
    report("chequer", chequer)
    report("cube", cube)
    report("overall", overall)


if __name__ == "__main__":
    main()
