"""
mine_cube_errors.py -- Find positions where a matchplay model's cube
decisions disagree badly with gnubg's score-aware verdict.

Plays self-play N-point matches; at every cube-eligible pre-roll
decision, compares the model's joint cube action (double? take?)
against gnubg's evaluate_cube_decision at the REAL score and cube
(1-ply, the same reference eval_matchplay's gnubg opponent uses), and
scores the deviation as MWC lost per gnubg's own mwcND/mwcDT/mwcDP.

Output: JSON list of the worst decisions with full position + match
state + both sides' numbers, for the learnability experiment.

Usage:
    python mine_cube_errors.py --model models/foo.pt --matches 1000 \
        --match-length 3 --workers 16 --out /tmp/cube_errors.json
"""

import argparse
import json
import multiprocessing as mp
import os
import random

from backgammon_engine import WHITE, opening_roll, switch_turn
from modes import MatchState, cube_perspective
from encoding import CubePerspective


def gnubg_cube_verdict(state, match_state, cube_plies=1):
    """gnubg's (action_double, action_take, mwcND, mwcDT, mwcDP) at the
    real score/cube for the pre-roll doubler `state.turn`. Mirrors
    GnubgNNMatchplayAgent._cube_decision, verbose."""
    import gnubg_nn
    from agents import GnubgNNAgent
    doubler = state.turn
    if doubler == 0:
        d_away, o_away = match_state.my_away, match_state.opp_away
    else:
        d_away, o_away = match_state.opp_away, match_state.my_away
    board = GnubgNNAgent._board_to_gnubg(state)
    pid = gnubg_nn.position_id(board)
    gnubg_nn.set.score(d_away, o_away)
    persp = cube_perspective(match_state.cube_owner, doubler)
    if persp == CubePerspective.CENTERED:
        gnubg_nn.set.cube(1, b'X')
    else:
        gnubg_nn.set.cube(match_state.cube_value, b'O')
    r = gnubg_nn.evaluate_cube_decision(pid, n=cube_plies, i=1)
    return bool(r[0]), bool(r[1]), float(r[3]), float(r[4]), float(r[5])


def action_value(double, take, mwc_nd, mwc_dt, mwc_dp):
    if not double:
        return mwc_nd
    return mwc_dt if take else mwc_dp


def mine_worker(args):
    (model_path, n_matches, match_length, seed, met_path,
     min_mmwc) = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from model import TDNetwork
    from td_agent import TDAgent

    net = TDNetwork.load(model_path)
    met = None
    if met_path:
        from met import MatchEquityTable
        met = MatchEquityTable.from_xml(met_path)
    agent = TDAgent(net, met=met)
    rng = random.Random(seed)
    errors = []

    for _ in range(n_matches):
        ms = MatchState(
            jacoby=False, match_length=match_length,
            my_away=match_length, opp_away=match_length,
        )
        while True:  # games in match
            gms = ms.start_game()
            state, dice = opening_roll(rng)
            is_opening = True
            while True:  # turns in game
                player = state.turn
                if not is_opening and gms.can_offer(player):
                    offer = agent.offer_double_matchplay(state, gms)
                    c = offer._cache
                    d_model = offer.should_double
                    t_model = c["v_dp"] > c["v_dt"]  # responder rule
                    g_d, g_t, mwc_nd, mwc_dt, mwc_dp = gnubg_cube_verdict(
                        state, gms,
                    )
                    v_opt = action_value(g_d, g_t, mwc_nd, mwc_dt, mwc_dp)
                    # Model's joint action: gnubg's responder answers the
                    # model's double per gnubg (isolates the DOUBLE error);
                    # the take error is scored separately below.
                    v_dbl = action_value(d_model, g_t, mwc_nd, mwc_dt, mwc_dp)
                    dbl_loss = (v_opt - v_dbl) * 1000.0
                    # Take error: if a double is correct per gnubg, compare
                    # the model's response against gnubg's.
                    take_loss = 0.0
                    if g_d and t_model != g_t:
                        take_loss = abs(mwc_dt - mwc_dp) * 1000.0
                    if dbl_loss > min_mmwc or take_loss > min_mmwc:
                        errors.append({
                            "points": list(state.points),
                            "bar": list(state.bar),
                            "off": list(state.off),
                            "turn": state.turn,
                            "cube_owner": int(gms.cube_owner),
                            "cube_value": gms.cube_value,
                            "my_away": gms.my_away,
                            "opp_away": gms.opp_away,
                            "is_crawford": gms.is_crawford,
                            "crawford_done": gms.crawford_done,
                            "model_double": bool(d_model),
                            "model_take": bool(t_model),
                            "gnubg_double": g_d,
                            "gnubg_take": g_t,
                            "model_v_nd": c["v_nd"],
                            "model_v_dt": c["v_dt"],
                            "model_v_dp": c["v_dp"],
                            "gnubg_mwc_nd": mwc_nd,
                            "gnubg_mwc_dt": mwc_dt,
                            "gnubg_mwc_dp": mwc_dp,
                            "dbl_loss_mmwc": round(dbl_loss, 2),
                            "take_loss_mmwc": round(take_loss, 2),
                        })
                    # Execute the MODEL's action so the game follows the
                    # model's natural self-play distribution.
                    if d_model:
                        if t_model:
                            gms = gms.after_take(player)
                        else:
                            ms = gms.after_game(1, winner_is_me=(player == 0))
                            break

                result = agent.choose_checker_action_cubeful(state, dice, gms)
                if result is None:
                    state = switch_turn(state)
                else:
                    next_state = result[0]
                    if next_state.is_game_over():
                        gr = next_state.game_result()
                        ms = gms.after_game(gr, winner_is_me=(player == 0))
                        break
                    state = next_state
                dice = (rng.randint(1, 6), rng.randint(1, 6))
                is_opening = False
            if ms.match_over():
                break
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--met", default=None, metavar="TABLE",
                   help="Match equity table to PLAY with. NOTE: the oracle "
                        "here is gnubg-nn, whose MET is Snowie, while "
                        "'auto' is Kazaross-XG2 -- they differ by up to "
                        "7.6 mMWC at 3-point scores, so keep --min-mwc "
                        "above that or MET artifacts land in the mined set.")
    p.add_argument("--min-mwc", type=float, default=0.5, metavar="MMWC",
                   help="Minimum decision loss (in mMWC) for a disagreement "
                        "to count as an error. The default 0.5 is far below "
                        "the ~7.6 mMWC gap between gnubg-nn's Snowie oracle "
                        "and a Kazaross-XG2-grounded model, so raise it to "
                        "~10 when mining a MET-grounded net or the set fills "
                        "with table artifacts rather than real errors.")
    p.add_argument("--matches", type=int, default=1000)
    p.add_argument("--match-length", type=int, default=3)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--out", default="/tmp/cube_errors.json")
    p.add_argument("--top", type=int, default=200,
                   help="keep this many worst decisions")
    args = p.parse_args()

    per = [args.matches // args.workers] * args.workers
    for i in range(args.matches % args.workers):
        per[i] += 1
    from met import resolve_met_path
    _met = resolve_met_path(args.met)
    wargs = [(args.model, n, args.match_length, args.seed + 7777 * i, _met,
              args.min_mwc)
             for i, n in enumerate(per) if n]
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(wargs)) as pool:
        errors = [e for chunk in pool.map(mine_worker, wargs) for e in chunk]

    errors.sort(key=lambda e: max(e["dbl_loss_mmwc"], e["take_loss_mmwc"]),
                reverse=True)
    kept = errors[:args.top]
    with open(args.out, "w") as f:
        json.dump(kept, f, indent=1)

    # Pattern summary
    from collections import Counter
    n_dbl = sum(1 for e in errors if e["dbl_loss_mmwc"] > 0.5)
    n_take = sum(1 for e in errors if e["take_loss_mmwc"] > 0.5)
    miss = sum(1 for e in errors
               if e["gnubg_double"] and not e["model_double"])
    extra = sum(1 for e in errors
                if e["model_double"] and not e["gnubg_double"])
    by_score = Counter(
        (e["my_away"], e["opp_away"], e["cube_value"], e["is_crawford"])
        for e in errors
    )
    print(f"{len(errors)} flagged cube decisions over {args.matches} matches"
          f" (kept top {len(kept)} -> {args.out})")
    print(f"  double errors: {n_dbl} (missed dbl {miss}, wrong dbl {extra})"
          f"  take errors: {n_take}")
    print("  by (my_away, opp_away, cube, crawford):")
    for k, v in by_score.most_common(10):
        print(f"    {k}: {v}")
    if kept:
        w = kept[0]
        print(f"  worst: loss {max(w['dbl_loss_mmwc'], w['take_loss_mmwc'])}"
              f" mMWC at ({w['my_away']},{w['opp_away']}) cube"
              f" {w['cube_value']} model_dbl={w['model_double']}"
              f" gnubg_dbl={w['gnubg_double']}")


if __name__ == "__main__":
    main()
