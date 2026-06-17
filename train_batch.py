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
from train_cli import (
    add_common_args, apply_cubeful_money_upgrades, build_mode, build_network,
    eval_vs_random, resolve_save_path,
)
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
                              "Move selection during collection stays at 0-ply "
                              "unless --oneply-acting is also set.")
    g_batch.add_argument("--oneply-acting", action="store_true",
                         help="Use 1-ply move selection during episode "
                              "collection (the agent picks moves via 1-ply "
                              "lookahead). Implies --oneply for targets.")
    g_batch.add_argument("--boltzmann-temp", type=float, default=0.0,
                         help="Boltzmann temperature for checker play "
                              "exploration. 0.0 (default) = greedy. "
                              "Higher = more exploration.")
    g_batch.add_argument("--warmup-cycles", type=int, default=0,
                         help="Ramp LR from lr*0.1 up to scheduled lr over the "
                              "first N rounds. Useful when starting from a "
                              "depth-expanded model.")
    g_batch.add_argument("--bf16-collect", action="store_true",
                         help="Run 1-ply target NN forward in bf16 inside "
                              "collection workers (~1.9x matmul speedup). "
                              "Master training step stays fp32. Introduces "
                              "~1e-3 numerical drift in 1-ply targets.")
    g_batch.add_argument("--cube-targets-1ply", action="store_true",
                         help="EXPERIMENTAL: in --oneply training mode, use "
                              "value_oneply_checker_cubeful() for cube-decision "
                              "targets (pure 1-ply, conceptually a deeper "
                              "search). Default uses the 0-ply formula "
                              "(evaluate_cubeful at is_cube_action=False) for "
                              "cube targets, which empirically gives much "
                              "better cube_mEMG. No effect without --oneply.")
    g_batch.add_argument("--pipeline-collect", action="store_true",
                         help="Overlap worker collection of round N+1 with the "
                              "master training step of round N. Workers in round "
                              "N+1 see weights from end of round N-1 (one-round "
                              "stale targets). Negligible TD impact at typical "
                              "1-ply LRs but should be validated when changing "
                              "recipes. Reproducibility preserved. Requires "
                              "--workers > 1; no-op otherwise.")
    g_batch.add_argument("--save-every", type=int, default=0, metavar="N",
                         help="Checkpoint at the first round boundary on "
                              "or after every N episodes (effective "
                              "granularity is max(N, --episodes-per-round)). "
                              "Saves to {--save}_ep{episodes}.pt; final "
                              "save still goes to --save. 0 disables; "
                              "requires --save.")

    parser.add_argument("--log-every", type=int, default=1,
                        help="Print recent batch loss every N rounds (0 to disable).")
    args = parser.parse_args()

    apply_cubeful_money_upgrades(args)

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)

    net = build_network(args)
    # --oneply-acting implies --oneply for targets
    if args.oneply_acting:
        args.oneply = True
    agent = TDAgent(net, device=args.device, oneply=args.oneply_acting,
                    boltzmann_temp=args.boltzmann_temp)
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
    # Resolve save path up-front so periodic checkpoints and the final
    # save share the same normalized target.
    resolved_save = (
        resolve_save_path(args.save, args.game_mode, net.hidden_sizes)
        if args.save else None
    )
    if args.save_every > 0 and not resolved_save:
        raise ValueError("--save-every requires --save")

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
        oneply_acting=args.oneply_acting,
        end_lr=args.end_lr,
        warmup_cycles=args.warmup_cycles,
        boltzmann_temp=args.boltzmann_temp,
        bf16_collect=args.bf16_collect,
        save_path=resolved_save,
        save_every=args.save_every,
        cube_targets_1ply=args.cube_targets_1ply,
        pipeline_collect=args.pipeline_collect,
    )
    elapsed = time.perf_counter() - t0
    eps_per_s = args.num_episodes / elapsed if elapsed > 0 else 0.0
    print(
        f"Done in {elapsed:.1f}s ({eps_per_s:.1f} ep/s, "
        f"{len(losses)} batches)"
    )

    if resolved_save:
        net.save(resolved_save)
        print(f"Saved network to {resolved_save}")

    if args.eval_vs_random > 0:
        wr = eval_vs_random(agent, n_games=args.eval_vs_random)
        print(f"vs Random ({args.eval_vs_random} games): {wr:.1%}")


if __name__ == "__main__":
    main()
