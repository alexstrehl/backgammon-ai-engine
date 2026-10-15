"""
train_from_position.py -- Cube-decision learnability experiment.

Takes a position where the model's cube decision disagrees badly with
gnubg (mined by mine_cube_errors.py) and overtrains the model on
episodes that START at that exact position (cube phase live on the
first turn), tracking where v_nd / v_dt / v_dp converge.

Interpretation:
  - values converge to the Monte-Carlo truth (under the model's own
    play) and the decision flips    -> DISTRIBUTION: the position
    class is simply too rare in natural self-play.
  - values converge but to something far from MC truth -> TARGETS:
    the TD backup at this state is biased.
  - values barely move despite thousands of targeted episodes ->
    OPTIMIZATION/CAPACITY.
  - control values (opening MET) drift while the position trains ->
    INTERFERENCE: shared trunk can't hold both.

Usage:
    python train_from_position.py --model models/foo.pt \
        --errors /tmp/cube_errors.json --index 0 \
        --rounds 50 --episodes-per-round 100
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import random

import numpy as np
import torch

from backgammon_engine import BoardState, opening_roll, switch_turn
from modes import MatchState, CubeOwner


def load_position(err):
    state = BoardState(
        points=list(err["points"]), bar=list(err["bar"]),
        off=list(err["off"]), turn=err["turn"],
    )
    gms = MatchState(
        cube_owner=CubeOwner(err["cube_owner"]),
        cube_value=err["cube_value"],
        jacoby=False,
        match_length=3,
        my_away=err["my_away"], opp_away=err["opp_away"],
        is_crawford=err["is_crawford"],
        crawford_done=err["crawford_done"],
    )
    return state, gms


# ── Episode collection from a fixed mid-game position ────────────────


def collect_from_position(agent, state0, gms0, rng,
                          force_first_cube=None, outcome_mix=0.0):
    """One full-match episode starting at (state0, gms0) with the cube
    phase LIVE on the first turn. Faithful adaptation of trainer's
    _collect_episode_matchplay target scheme (cube exploration,
    optimal-action backups, boundary upweights). The outer match
    continues to match_over via fresh games.

    force_first_cube: None (normal), "nd" (suppress the first cube
    phase), or "dt" (apply double/take immediately) — used by the
    Monte-Carlo truth rollouts, with recording disabled.
    """
    from trainer import _CUBE_EXPLORATION_EPS, _BOUNDARY_UPWEIGHT

    enc_list, targets, viewers = [], [], []

    def _emit(enc, t, viewer):
        enc_list.append(enc)
        targets.append(t)
        viewers.append(viewer)

    match_state = gms0  # away scores already reflect the live game
    first_turn = True
    state, game_ms = state0, gms0
    mover0 = state0.turn

    while True:
        game_done = False
        if not first_turn:
            game_ms = match_state.start_game()
            state, dice = opening_roll(rng)
            is_opening = True
        else:
            dice = None  # rolled after the cube phase below
            is_opening = False

        while not game_done:
            player = state.turn

            if not is_opening and game_ms.can_offer(player):
                skip_phase = False
                if first_turn and force_first_cube == "nd":
                    skip_phase = True
                elif first_turn and force_first_cube == "dt":
                    game_ms = game_ms.after_take(player)
                    skip_phase = True
                if not skip_phase:
                    cube_enc = agent._encode_matchplay(
                        state, game_ms, is_cube_action=True,
                    )
                    offer = agent.offer_double_matchplay(state, game_ms)
                    c = offer._cache
                    v_cube_opt = max(c["v_nd"], min(c["v_dt"], c["v_dp"]))
                    do_double = offer.should_double
                    explored = False
                    if rng.random() < _CUBE_EXPLORATION_EPS:
                        do_double = rng.random() < 0.5
                        explored = True
                    if do_double:
                        takes = agent.respond_to_double_matchplay(
                            state, game_ms, hint=offer,
                        )
                        if rng.random() < _CUBE_EXPLORATION_EPS:
                            takes = rng.random() < 0.5
                            explored = True
                        if takes:
                            _emit(cube_enc, v_cube_opt, player)
                            game_ms = game_ms.after_take(player)
                        else:
                            match_state = game_ms.after_game(
                                1, winner_is_me=(player == 0),
                            )
                            if explored:
                                t = v_cube_opt
                            elif match_state.match_over():
                                t = 1.0
                            else:
                                t = agent._eval_new_game(match_state, player)
                            _emit(cube_enc, t, player)
                            if not explored:
                                for _ in range(_BOUNDARY_UPWEIGHT - 1):
                                    _emit(cube_enc, t, player)
                            game_done = True
                            continue
                    else:
                        _emit(cube_enc, v_cube_opt, player)

            if dice is None:
                dice = (rng.randint(1, 6), rng.randint(1, 6))

            chk_enc = agent._encode_matchplay(
                state, game_ms, is_cube_action=False,
            )
            result = agent.choose_checker_action_matchplay(
                state, dice, game_ms,
            )
            if result is None:
                opp_view = switch_turn(state)
                opp_ca = game_ms.can_offer(opp_view.turn)
                v_opp = agent.evaluate_matchplay(
                    opp_view, game_ms, is_cube_action=opp_ca,
                )
                _emit(chk_enc, 1.0 - v_opp, player)
                state = opp_view
            else:
                next_state, bootstrap = result
                if next_state.is_game_over():
                    gr = next_state.game_result()
                    match_state = game_ms.after_game(
                        gr, winner_is_me=(player == 0),
                    )
                    t = (1.0 if match_state.match_over()
                         else agent._eval_new_game(match_state, player))
                    _emit(chk_enc, t, player)
                    for _ in range(_BOUNDARY_UPWEIGHT - 1):
                        _emit(chk_enc, t, player)
                    game_done = True
                    continue
                _emit(chk_enc, bootstrap, player)
                state = next_state

            dice = (rng.randint(1, 6), rng.randint(1, 6))
            is_opening = False
            first_turn = False

        first_turn = False
        if match_state.match_over():
            break

    # Did WHITE win the match? (away scores are WHITE's view)
    white_won = match_state.my_away <= 0
    mover_won = white_won if mover0 == 0 else not white_won
    t_arr = np.asarray(targets, np.float32)
    if outcome_mix > 0.0:
        z = np.asarray(
            [1.0 if (white_won if v == 0 else not white_won) else 0.0
             for v in viewers], np.float32)
        t_arr = (1.0 - outcome_mix) * t_arr + outcome_mix * z
    return np.stack(enc_list), t_arr, mover_won


# ── Monte-Carlo ground truth under the model's own play ─────────────


def _mc_worker(args):
    model_path, err, branch, n, seed = args
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    from model import TDNetwork
    from td_agent import TDAgent
    net = TDNetwork.load(model_path)
    agent = TDAgent(net)
    state0, gms0 = load_position(err)
    rng = random.Random(seed)
    wins = 0
    for _ in range(n):
        _, _, mover_won = collect_from_position(
            agent, state0, gms0, rng, force_first_cube=branch,
        )
        wins += mover_won
    return wins, n


def mc_truth(model_path, err, branch, n_total, workers, seed=99):
    per = [n_total // workers] * workers
    for i in range(n_total % workers):
        per[i] += 1
    args = [(model_path, err, branch, n, seed + 31 * i)
            for i, n in enumerate(per) if n]
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(args)) as pool:
        res = pool.map(_mc_worker, args)
    wins = sum(w for w, _ in res)
    n = sum(c for _, c in res)
    return wins / n


# ── Experiment ───────────────────────────────────────────────────────


def snapshot(agent, state0, gms0, control_states):
    offer = agent.offer_double_matchplay(state0, gms0)
    c = offer._cache
    agent.clear_matchplay_memo()
    controls = {}
    for name, (s, ms) in control_states.items():
        controls[name] = agent.evaluate_matchplay(s, ms)
    return {
        "v_nd": c["v_nd"], "v_dt": c["v_dt"], "v_dp": c["v_dp"],
        "double": offer.should_double,
        "take": c["v_dp"] > c["v_dt"],
        **controls,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--errors", default="/tmp/cube_errors.json")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--episodes-per-round", type=int, default=100)
    p.add_argument("--lr", type=float, default=7e-5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--mc", type=int, default=2000,
                   help="Monte-Carlo rollouts per branch (0 to skip)")
    p.add_argument("--mc-workers", type=int, default=16)
    p.add_argument("--outcome-mix", type=float, default=0.0,
                   help="blend fraction of realized match outcome "
                        "into every target (0 = pure TD(0))")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--save", default=None,
                   help="save the overtrained net here (optional)")
    args = p.parse_args()

    errs = json.load(open(args.errors))
    err = errs[args.index]
    print(f"Position #{args.index}: score=({err['my_away']},{err['opp_away']})"
          f" cube={err['cube_value']} turn={err['turn']}"
          f" model_dbl={err['model_double']} gnubg_dbl={err['gnubg_double']}"
          f" gnubg_take={err['gnubg_take']}"
          f" loss={max(err['dbl_loss_mmwc'], err['take_loss_mmwc'])} mMWC")
    print(f"  gnubg: mwcND={err['gnubg_mwc_nd']:.3f}"
          f" mwcDT={err['gnubg_mwc_dt']:.3f} mwcDP={err['gnubg_mwc_dp']:.3f}")

    if args.mc:
        print(f"Monte-Carlo truth under model's own play"
              f" ({args.mc} rollouts/branch):")
        for branch in ("nd", "dt"):
            v = mc_truth(args.model, err, branch, args.mc, args.mc_workers,
                         seed=args.seed * 13)
            print(f"  MC_{branch}: {v:.3f}")

    torch.set_num_threads(8)
    from model import TDNetwork
    from td_agent import TDAgent
    from trainer import Trainer
    net = TDNetwork.load(args.model)
    agent = TDAgent(net)
    trainer = Trainer(agent, lr=args.lr)
    state0, gms0 = load_position(err)

    # Controls: drift detectors for global knowledge.
    init = BoardState.initial()
    controls = {
        "ctrl_open33": (init, MatchState(
            jacoby=False, match_length=3, my_away=3, opp_away=3)),
        "ctrl_open23": (init, MatchState(
            jacoby=False, match_length=3, my_away=2, opp_away=3)),
    }

    rng = random.Random(args.seed)
    s0 = snapshot(agent, state0, gms0, controls)
    print(f"\nround   v_nd   v_dt   v_dp  dbl take  ctrl33  ctrl23  loss")
    print(f"{0:5d}  {s0['v_nd']:.3f}  {s0['v_dt']:.3f}  {s0['v_dp']:.3f}"
          f"  {int(s0['double'])}   {int(s0['take'])}   {s0['ctrl_open33']:.3f}"
          f"   {s0['ctrl_open23']:.3f}     -")

    for rnd in range(1, args.rounds + 1):
        encs, tgts = [], []
        for _ in range(args.episodes_per_round):
            e, t, _ = collect_from_position(
                agent, state0, gms0, rng, outcome_mix=args.outcome_mix)
            encs.append(e)
            tgts.append(t)
        X = np.concatenate(encs)
        y = np.concatenate(tgts)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        losses = []
        for i in range(0, len(idx), args.batch_size):
            b = idx[i:i + args.batch_size]
            losses.append(trainer.train_step(X[b], y[b].tolist()))
        agent.clear_matchplay_memo()
        s = snapshot(agent, state0, gms0, controls)
        print(f"{rnd:5d}  {s['v_nd']:.3f}  {s['v_dt']:.3f}  {s['v_dp']:.3f}"
              f"  {int(s['double'])}   {int(s['take'])}   {s['ctrl_open33']:.3f}"
              f"   {s['ctrl_open23']:.3f}  {np.mean(losses):.5f}")

    if args.save:
        net.save(args.save)
        print(f"saved overtrained net to {args.save}")


if __name__ == "__main__":
    main()
