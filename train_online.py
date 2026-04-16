#!/usr/bin/env python3
"""
train_online.py -- Online TD(0) self-play training for backgammon.

Online TD(0) updates the network after every transition, with the
target computed against the network's CURRENT weights. Best paired
with plain SGD (the default here).

Examples:

    # Fresh DMP probability agent, 4000 episodes
    python3 train_online.py --game-mode dmp \\
        --output-mode probability --hidden 40 \\
        --num-episodes 4000 --lr 0.1 \\
        --save dmp_4000.pt

    # Resume from a saved network
    python3 train_online.py --game-mode dmp \\
        --resume dmp_4000.pt --num-episodes 1000 \\
        --save dmp_5000.pt --eval-vs-random 200
"""

# Pin BLAS / PyTorch to a single thread BEFORE importing torch. Online
# TD(0) is dominated by tiny single-sample forward passes; with 64
# BLAS threads contending on every step the wall clock balloons by
# ~10x for no benefit. Matches the legacy td_train.py convention.
import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import time

import torch
torch.set_num_threads(1)

from td_agent import TDAgent
from train_cli import add_common_args, build_mode, build_network, eval_vs_random
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Online TD(0) self-play training for backgammon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)

    g_opt = parser.add_argument_group("optimizer (SGD)")
    g_opt.add_argument("--lr", type=float, default=0.1,
                       help="SGD learning rate.")
    g_opt.add_argument("--momentum", type=float, default=0.0,
                       help="SGD momentum.")

    parser.add_argument("--log-every", type=int, default=100,
                        help="Print recent step loss every N episodes (0 to disable).")
    args = parser.parse_args()

    # Same cubeful-money auto-upgrade as train_batch.py.
    if args.game_mode == "cubeful-money" and not args.resume:
        if args.encoder == "perspective196":
            args.encoder = "cubeful_perspective196"
        elif not args.encoder.startswith("cubeful_"):
            raise ValueError(
                f"cubeful-money requires a cubeful_* encoder; "
                f"got --encoder {args.encoder!r}"
            )
        if args.output_mode == "probability":
            args.output_mode = "equity"
        elif args.output_mode != "equity":
            raise ValueError(
                f"cubeful-money requires --output-mode equity; "
                f"got {args.output_mode!r}"
            )

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)

    net = build_network(args)
    agent = TDAgent(net, device=args.device)
    mode = build_mode(args.game_mode, jacoby=args.jacoby)

    # Equity output is unbounded; high lr causes gradient explosion -> NaN.
    # Probability output is sigmoid-bounded so 0.1 is fine.
    if net.output_mode == "equity" and args.lr > 0.05:
        print(
            f"WARNING: lr={args.lr} is likely too high for equity output_mode "
            f"(unbounded targets); training may diverge to NaN. "
            f"Recommended: --lr 0.01 or lower for equity."
        )

    optimizer_kwargs = {}
    if args.momentum > 0:
        optimizer_kwargs["momentum"] = args.momentum
    trainer = Trainer(
        agent,
        lr=args.lr,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs=optimizer_kwargs,
    )

    print(
        f"Online TD(0) | game={args.game_mode} | output={net.output_mode} "
        f"| hidden={net.hidden_sizes} | lr={args.lr} momentum={args.momentum}"
    )
    print(f"Training for {args.num_episodes} episodes...")
    t0 = time.perf_counter()
    losses = trainer.train_online(
        mode,
        num_episodes=args.num_episodes,
        seed=args.seed,
        log_every=args.log_every,
    )
    elapsed = time.perf_counter() - t0
    eps_per_s = args.num_episodes / elapsed if elapsed > 0 else 0.0
    print(
        f"Done in {elapsed:.1f}s ({eps_per_s:.1f} ep/s, "
        f"{len(losses)} TD steps)"
    )

    if args.save:
        net.save(args.save)
        print(f"Saved network to {args.save}")

    if args.eval_vs_random > 0:
        wr = eval_vs_random(agent, n_games=args.eval_vs_random)
        print(f"vs Random ({args.eval_vs_random} games): {wr:.1%}")


if __name__ == "__main__":
    main()
