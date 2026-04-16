#!/usr/bin/env python3
"""
train_batch.py -- Batch TD(0) self-play training for backgammon.

Round-based: each "round" collects `episodes_per_round` self-play
episodes, shuffles them, and trains for `epochs_per_round` passes in
fixed-size batches via the agent's predict / loss primitives. Default
optimizer is Adam.

Examples:

    # Fresh cubeless-money equity agent, 50000 episodes
    python3 train_batch.py --game-mode cubeless-money --output-mode equity \\
        --hidden 80,40 --num-episodes 50000 \\
        --batch-size 256 --episodes-per-round 200 \\
        --lr 1e-3 --save money_50k.pt

    # Resume and continue
    python3 train_batch.py --game-mode cubeless-money \\
        --resume money_50k.pt --num-episodes 10000 \\
        --save money_60k.pt --eval-vs-random 200
"""

# Pin BLAS / PyTorch threads BEFORE importing torch. Even with batch
# training, the network is small enough that thread contention can
# dominate, and parallel-worker collection (when added) wants each
# worker subprocess to be single-threaded. Matches the legacy
# td_batch_train.py convention.
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
        description="Batch TD(0) self-play training for backgammon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)

    g_opt = parser.add_argument_group("optimizer")
    g_opt.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    g_opt.add_argument("--lr", type=float, default=1e-3,
                       help="Optimizer learning rate.")
    g_opt.add_argument("--end-lr", type=float, default=None,
                       help="If set, linearly anneal lr -> end_lr over the run.")
    g_opt.add_argument("--momentum", type=float, default=0.0,
                       help="SGD momentum (ignored for Adam).")
    g_opt.add_argument("--grad-clip", type=float, default=None,
                       help="Max gradient norm (None disables clipping).")

    g_batch = parser.add_argument_group("batch / round")
    g_batch.add_argument("--batch-size", type=int, default=256)
    g_batch.add_argument("--episodes-per-round", type=int, default=100,
                         help="Episodes collected before each shuffle+train pass.")
    g_batch.add_argument("--epochs-per-round", type=int, default=1,
                         help="Passes over the round's pool per round.")
    g_batch.add_argument("--workers", type=int, default=1,
                         help="Number of subprocesses for parallel episode "
                              "collection. >1 distributes the per-round "
                              "episode collection across N workers; the "
                              "training step still runs in the master.")
    g_batch.add_argument("--oneply", action="store_true",
                         help="Compute training targets via 1-ply lookahead "
                              "(enumerate all 21 dice outcomes per state). "
                              "Lower variance, ~5-10x slower per turn. "
                              "Move selection during collection stays at 0-ply.")
    g_batch.add_argument("--warmup-cycles", type=int, default=0,
                         help="Ramp LR from lr*0.1 up to scheduled lr over the "
                              "first N rounds. Useful when starting from a "
                              "depth-expanded model.")

    parser.add_argument("--log-every", type=int, default=1,
                        help="Print recent batch loss every N rounds (0 to disable).")
    args = parser.parse_args()

    # Cubeful-money auto-upgrades: the encoder must include the cube
    # one-hot and the output must be equity. Override silently if the
    # user left the defaults; error if they set conflicting flags.
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

    # Equity output is unbounded; high lr can cause gradient explosion -> NaN.
    # SGD is more sensitive than Adam (which normalizes by gradient moments).
    if net.output_mode == "equity":
        lr_limit = 0.05 if args.optimizer == "sgd" else 0.01
        if args.lr > lr_limit:
            print(
                f"WARNING: lr={args.lr} is likely too high for equity output_mode "
                f"with {args.optimizer} (unbounded targets may cause NaN). "
                f"Recommended: --lr {lr_limit} or lower for equity + {args.optimizer}."
            )

    optimizer_cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.SGD
    optimizer_kwargs = {}
    if args.momentum > 0 and args.optimizer == "sgd":
        optimizer_kwargs["momentum"] = args.momentum
    trainer = Trainer(
        agent,
        lr=args.lr,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        grad_clip=args.grad_clip,
    )

    print(
        f"Batch TD(0) | game={args.game_mode} | output={net.output_mode} "
        f"| hidden={net.hidden_sizes} | opt={args.optimizer} lr={args.lr}"
    )
    print(
        f"Training for {args.num_episodes} episodes "
        f"(batch={args.batch_size}, ep/round={args.episodes_per_round}, "
        f"epochs/round={args.epochs_per_round})..."
    )
    t0 = time.perf_counter()
    losses = trainer.train(
        mode,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        episodes_per_round=args.episodes_per_round,
        epochs_per_round=args.epochs_per_round,
        seed=args.seed,
        log_every=args.log_every,
        workers=args.workers,
        oneply=args.oneply,
        end_lr=args.end_lr,
        warmup_cycles=args.warmup_cycles,
    )
    elapsed = time.perf_counter() - t0
    eps_per_s = args.num_episodes / elapsed if elapsed > 0 else 0.0
    print(
        f"Done in {elapsed:.1f}s ({eps_per_s:.1f} ep/s, "
        f"{len(losses)} batches)"
    )

    if args.save:
        net.save(args.save)
        print(f"Saved network to {args.save}")

    if args.eval_vs_random > 0:
        wr = eval_vs_random(agent, n_games=args.eval_vs_random)
        print(f"vs Random ({args.eval_vs_random} games): {wr:.1%}")


if __name__ == "__main__":
    main()
