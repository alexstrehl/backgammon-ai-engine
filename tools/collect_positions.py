#!/usr/bin/env python3
"""Collect self-play positions for distillation.

Plays `--games` self-play games with the acting model through the
trainers' own episode collectors and writes every encoded row to an
npz, so the rows are exactly what TD training would see for that net:
  cubeless-prob5   train_prob5._collect_one_episode: one row per turn,
                   196 features
  cubeful-money    trainer.collect_episode: cube-phase rows
                   (is_cube_action=1) plus checker-phase rows, 200 features
  cubeless-money,  trainer.collect_episode: one row per turn, 196 features
  dmp
Over-length episodes (1500 plies) are discarded, as in training.

Targets are not written unless --targets: a bank is labelled later
(tools/distill.py) by whichever teacher is wanted.

npz keys: `enc` (N x features); with --targets `tgt` (N for scalar nets,
N x 5 for prob5); metadata model, model_md5, game_mode, jacoby, oneply,
seed, games, argv.

Usage:
  python tools/collect_positions.py --model best_models/cubeful_money_512_512_256_256.pt \\
      --games 40000 --seed 1 --out bank.npz
"""
import argparse
import hashlib
import multiprocessing as mp
import os
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

GAME_MODES = ("dmp", "cubeless-money", "cubeful-money", "cubeless-prob5")


def derive_game_mode(net) -> str:
    """Game mode a net was trained for. An equity net with a plain
    encoder can also play dmp. Pass --game-mode for that."""
    if getattr(net, "output_mode", None) == "prob5":
        return "cubeless-prob5"
    if net.encoder_name.startswith("cubeful_"):
        return "cubeful-money"
    if net.output_mode == "probability":
        return "dmp"
    return "cubeless-money"


def check_game_mode(net, game_mode: str) -> None:
    """Reject net/mode pairs the collectors cannot play."""
    is_p5 = getattr(net, "output_mode", None) == "prob5"
    if is_p5 != (game_mode == "cubeless-prob5"):
        raise SystemExit(f"{game_mode} needs a {'prob5' if not is_p5 else 'scalar'} net")
    cubeful_net = net.encoder_name.startswith("cubeful_")
    if cubeful_net != (game_mode == "cubeful-money"):
        raise SystemExit(f"{game_mode} needs a {'cubeful' if not cubeful_net else 'plain'} encoder")
    if not is_p5:
        from train_cli import build_mode
        try:
            build_mode(game_mode).validate_agent(net)
        except ValueError as e:
            raise SystemExit(str(e))


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _worker(args):
    """Play n_games. Return (enc, tgt, n_discarded, n_kept)."""
    (wid, n_games, model_path, game_mode, jacoby, oneply, base_seed) = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from model import load_model
    rng = random.Random(base_seed * 100_003 + wid)
    net = load_model(model_path)
    net.eval()
    if game_mode == "cubeless-prob5":
        from train_prob5 import NUM_OUTPUTS, _collect_one_episode, _resolve_engine_fns
        encode_fn, gpe_fn = _resolve_engine_fns(net.encoder_name)
        tgt_shape = (0, NUM_OUTPUTS)

        def play():
            return _collect_one_episode(net, rng, encode_fn, gpe_fn, oneply=False, device="cpu")
    else:
        from td_agent import TDAgent
        from train_cli import build_mode
        from trainer import collect_episode
        agent = TDAgent(net, oneply=oneply)
        mode = build_mode(game_mode, jacoby=jacoby)
        tgt_shape = (0,)

        def play():
            return collect_episode(agent, mode, rng, oneply=oneply)

    encs, tgts, n_disc = [], [], 0
    for _ in range(n_games):
        enc, tgt = play()
        if len(enc):
            encs.append(enc.astype(np.float32, copy=False))
            tgts.append(tgt.astype(np.float32, copy=False))
        else:
            n_disc += 1
    if not encs:
        return (np.empty((0, net.input_size), np.float32),
                np.empty(tgt_shape, np.float32), n_disc, 0)
    return np.concatenate(encs, 0), np.concatenate(tgts, 0), n_disc, len(encs)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Self-play position bank (npz) for distillation.")
    ap.add_argument("--model", required=True, help="acting model (.pt)")
    ap.add_argument("--game-mode", choices=GAME_MODES, default=None,
                    help="default: derived from the model")
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--workers", type=int, default=None, help="default: physical cores")
    ap.add_argument("--seed", type=int, required=True,
                    help="base seed; use a different one per bank so banks never overlap")
    ap.add_argument("--jacoby", action=argparse.BooleanOptionalAction, default=True,
                    help="cubeful-money only")
    ap.add_argument("--oneply", action="store_true",
                    help="1-ply acting (and 1-ply targets), as train_batch --oneply-acting; "
                         "scalar nets only")
    ap.add_argument("--targets", action="store_true",
                    help="also save the collector's TD targets as `tgt`")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import numpy as np
    from gpu_context import physical_cores
    from model import load_model

    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    net = load_model(a.model)
    game_mode = a.game_mode or derive_game_mode(net)
    check_game_mode(net, game_mode)
    if a.oneply and game_mode == "cubeless-prob5":
        raise SystemExit("--oneply: the prob5 collector acts at 0-ply only")
    workers = max(1, min(a.workers or physical_cores(), a.games))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    per = [a.games // workers + (1 if i < a.games % workers else 0) for i in range(workers)]
    jobs = [(i, n, a.model, game_mode, a.jacoby, a.oneply, a.seed)
            for i, n in enumerate(per) if n]
    t0 = time.time()
    if len(jobs) == 1:
        res = [_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            res = pool.map(_worker, jobs)
    enc = np.concatenate([r[0] for r in res], 0)
    tgt = np.concatenate([r[1] for r in res], 0)
    n_disc = sum(r[2] for r in res)
    n_games = sum(r[3] for r in res)
    dt = time.time() - t0
    extra = {"tgt": tgt} if a.targets else {}
    np.savez(a.out, enc=enc, **extra, model=os.path.basename(a.model),
             model_md5=_md5(a.model), game_mode=game_mode, jacoby=a.jacoby,
             oneply=a.oneply, seed=a.seed, games=n_games, argv=" ".join(sys.argv))
    print(f"{a.out}: {len(enc)} rows x {enc.shape[1]} ({game_mode}) from {n_games} games "
          f"({n_disc} discarded) in {dt:.0f}s = {n_games/max(dt,1e-9):.1f} g/s, "
          f"{len(enc)/max(n_games,1):.1f} rows/game")


if __name__ == "__main__":
    main()
