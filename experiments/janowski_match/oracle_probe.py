"""oracle_probe.py -- p5jan as reference instrument for the trained
matchplay net.

Two probes, both over states sampled from CHAMPION self-play (0-ply
acting, the same distribution training sees):

1. CHECKER / 1-ply-target bias probe. Per pre-move checker state s
   (the sample the collector encodes with is_cube_action=0):
       V  = champion 0-ply value               (what training moves)
       T1 = champion 1-ply target              (where 1-ply pulls it)
       O  = oracle cubeful MWC (prob5+Janowski, mover view)
   Compares mean(T1 - V) against mean(O - V) per context class, and
   mean(T1 - O) as residual target bias.

2. CUBE decision audit. Per cube-eligible state: champion's
   (v_nd, v_dt, v_dp, action) vs the oracle's exact-table reference
   (janowski_match.cube_decision). Disagreements priced in oracle MWC.

Usage:
    python experiments/janowski_match/oracle_probe.py \
        --champion best_models/matchplay_1h64met_512_512_256_256.pt \
        --prob5 best_models/cubeless_prob5_512_512_256_128.pt \
        --matches 60 [--out probe.npz]
"""

import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from backgammon_engine import opening_roll, switch_turn
from janowski_match import JanowskiMatchAgent, cube_decision, cubeful_mwc, ctx_for
from met import default_met
from model import ProbNetwork, TDNetwork
from modes import MatchState, cube_perspective
from td_agent import TDAgent


def classify(state, gms):
    """Context tags for a pre-move checker state."""
    mover = state.turn
    opp = 1 - mover
    persp = cube_perspective(gms.cube_owner, mover)
    tags = []
    tags.append("crawford" if gms.is_crawford else
                ("post_crawford" if gms.crawford_done else "normal"))
    tags.append(f"cube{gms.cube_value}")
    tags.append({0: "centered", 1: "own_cube", 2: "their_cube"}[int(persp)])
    tags.append("opp_can_double" if gms.can_offer(opp) else "opp_cube_blocked")
    my_away = gms.my_away if mover == 0 else gms.opp_away
    opp_away = gms.opp_away if mover == 0 else gms.my_away
    tags.append(f"away{my_away}v{opp_away}")
    return tags


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--champion", required=True)
    p.add_argument("--prob5", required=True)
    p.add_argument("--matches", type=int, default=60)
    p.add_argument("--match-length", type=int, default=3)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--sample-rate", type=float, default=0.5)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import torch
    torch.set_num_threads(2)

    met = default_met()
    champ = TDAgent(TDNetwork.load(args.champion))
    oracle = JanowskiMatchAgent(ProbNetwork.load(args.prob5), met=met)
    rng = random.Random(args.seed)

    # ── sample states from champion self-play ────────────────────────
    checker_rows = []   # (V, T1, O, tags)
    cube_rows = []      # dict per cube-eligible state

    for _ in range(args.matches):
        ms = MatchState(jacoby=False, match_length=args.match_length,
                        my_away=args.match_length,
                        opp_away=args.match_length)
        while True:
            gms = ms.start_game()
            state, dice = opening_roll(rng)
            opening = True
            while True:
                player = state.turn
                if not opening and gms.can_offer(player):
                    offer = champ.offer_double_matchplay(state, gms)
                    c = offer._cache
                    if rng.random() < args.sample_rate:
                        probs = oracle._probs_state(state)
                        ref = cube_decision(probs, gms, player,
                                            oracle._x(state), met)
                        cube_rows.append({
                            "net_nd": c["v_nd"], "net_dt": c["v_dt"],
                            "net_dp": c["v_dp"],
                            "net_double": bool(offer.should_double),
                            "net_take": c["v_dp"] > c["v_dt"],
                            "orc_nd": ref["v_nd"], "orc_dt": ref["v_dt"],
                            "orc_dp": ref["v_dp"],
                            "orc_double": bool(ref["should_double"]),
                            "orc_take": bool(ref["take"]),
                            "tags": classify(state, gms),
                        })
                    if offer.should_double:
                        takes = champ.respond_to_double_matchplay(
                            state, gms, hint=offer)
                        if takes:
                            gms = gms.after_take(player)
                        else:
                            ms = gms.after_game(1, winner_is_me=(player == 0))
                            break
                res = champ.choose_checker_action_matchplay(state, dice, gms)
                if res is not None and not res[0].is_game_over() \
                        and not opening and rng.random() < args.sample_rate:
                    V = champ.evaluate_matchplay(state, gms,
                                                 is_cube_action=False)
                    T1 = champ.value_oneply_checker_matchplay(state, gms)
                    T1f = champ.value_oneply_checker_matchplay(
                        state, gms, cube_fold=True)
                    probs = oracle._probs_state(state)
                    persp = cube_perspective(gms.cube_owner, player)
                    ctx = ctx_for(gms, player)
                    O = cubeful_mwc(probs, persp, ctx,
                                    oracle._x(state), met)
                    checker_rows.append((V, T1, O, classify(state, gms),
                                         T1f))
                if res is None:
                    state = switch_turn(state)
                else:
                    if res[0].is_game_over():
                        ms = gms.after_game(res[0].game_result(),
                                            winner_is_me=(player == 0))
                        break
                    state = res[0]
                dice = (rng.randint(1, 6), rng.randint(1, 6))
                opening = False
            if ms.match_over():
                break

    # ── probe 1 report: 1-ply pull vs oracle direction ──────────────
    V = np.array([r[0] for r in checker_rows])
    T1 = np.array([r[1] for r in checker_rows])
    O = np.array([r[2] for r in checker_rows])
    T1f = np.array([r[4] for r in checker_rows])

    def stat(x):
        return f"{x.mean()*1000:+7.2f} ({x.std()/np.sqrt(len(x))*1000:.2f})"

    print(f"\n== PROBE 1: checker states n={len(checker_rows)} "
          f"(units: mMWC, mean (SE)) ==")
    print(f"{'class':>22} {'n':>5} {'T1-V (1ply pull)':>18} "
          f"{'O-V (true err)':>16} {'T1-O (tgt bias)':>16} "
          f"{'T1fold-O':>14}")
    print(f"{'ALL':>22} {len(V):>5} {stat(T1-V):>18} {stat(O-V):>16} "
          f"{stat(T1-O):>16} {stat(T1f-O):>14}")
    by = defaultdict(list)
    for i, (_, _, _, tags, _f) in enumerate(checker_rows):
        for t in tags:
            by[t].append(i)
    for tag in sorted(by, key=lambda t: -len(by[t])):
        idx = np.array(by[tag])
        if len(idx) < 30:
            continue
        print(f"{tag:>22} {len(idx):>5} {stat((T1-V)[idx]):>18} "
              f"{stat((O-V)[idx]):>16} {stat((T1-O)[idx]):>16} "
              f"{stat((T1f-O)[idx]):>14}")

    # ── probe 2 report: cube decisions vs oracle ────────────────────
    n = len(cube_rows)
    dbl_dis = [r for r in cube_rows if r["net_double"] != r["orc_double"]]
    take_dis = [r for r in cube_rows
                if r["net_double"] and r["orc_double"]
                and r["net_take"] != r["orc_take"]]
    print(f"\n== PROBE 2: cube-eligible states n={n} ==")
    print(f"double-decision disagreements: {len(dbl_dis)}/{n} "
          f"= {100*len(dbl_dis)/max(n,1):.1f}%")
    over = [r for r in dbl_dis if r["net_double"]]
    under = [r for r in dbl_dis if not r["net_double"]]
    # Price each disagreement in ORACLE MWC: value of net's action minus
    # value of oracle's action, both under oracle values.
    def orc_val(r, double):
        return min(r["orc_dt"], r["orc_dp"]) if double else r["orc_nd"]
    cost_over = [orc_val(r, True) - orc_val(r, False) for r in over]
    cost_under = [orc_val(r, False) - orc_val(r, True) for r in under]
    print(f"  net doubles / oracle says no: {len(over):>4}  "
          f"mean oracle cost {np.mean(cost_over)*1000 if over else 0:+.2f} mMWC")
    print(f"  net holds  / oracle doubles:  {len(under):>4}  "
          f"mean oracle cost {np.mean(cost_under)*1000 if under else 0:+.2f} mMWC")
    print(f"take/pass disagreements (both double): {len(take_dis)}")
    nd_err = np.array([r["net_nd"] - r["orc_nd"] for r in cube_rows])
    dt_err = np.array([r["net_dt"] - r["orc_dt"] for r in cube_rows])
    dp_err = np.array([r["net_dp"] - r["orc_dp"] for r in cube_rows])
    print(f"value errors vs oracle (mMWC, mean (SE)):")
    print(f"  v_nd {stat(nd_err)}   v_dt {stat(dt_err)}   "
          f"v_dp {stat(dp_err)}")
    by2 = defaultdict(list)
    for i, r in enumerate(cube_rows):
        for t in r["tags"]:
            by2[t].append(i)
    print(f"{'class':>22} {'n':>5} {'nd err':>14} {'dt err':>14} "
          f"{'dp err':>14} {'dis%':>6}")
    for tag in sorted(by2, key=lambda t: -len(by2[t])):
        idx = np.array(by2[tag])
        if len(idx) < 25:
            continue
        dis = sum(1 for i in idx
                  if cube_rows[i]["net_double"] != cube_rows[i]["orc_double"])
        print(f"{tag:>22} {len(idx):>5} {stat(nd_err[idx]):>14} "
              f"{stat(dt_err[idx]):>14} {stat(dp_err[idx]):>14} "
              f"{100*dis/len(idx):>5.1f}%")

    if args.out:
        np.savez_compressed(
            args.out,
            V=V, T1=T1, O=O, T1f=T1f,
            checker_tags=np.array(
                ["|".join(r[3]) for r in checker_rows]),
            allow_pickle=False,
        )
        print(f"\nraw rows -> {args.out}")


if __name__ == "__main__":
    main()
