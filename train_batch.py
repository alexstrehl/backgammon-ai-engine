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
# dominate, and parallel-worker collection wants each worker subprocess
# to be single-threaded.
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
from trainer import DEFAULT_MAX_EPISODE_PLIES, Trainer


def resolve_cuda_oneply_collector(args):
    """Route `--oneply --device cuda` to the vectorized GPU collector.

    Returns the list of notes to print (pure-ish: mutates only
    args.vectorized_collect; no I/O), so the routing table is unit-
    testable without a GPU or a parser.

    Rules:
      * explicit --vectorized-collect and --no-vectorized-collect
        together is an error;
      * AUTO-enable only for --oneply --device cuda, and only when the
        user chose no other collector accelerator (--c-engine /
        --bf16-collect are explicit CPU-collector choices and win);
      * --oneply-acting keeps the CPU collector (the vectorized worker
        hard-refuses it; auto-routing must not manufacture that error).
        Callers must normalize "--oneply-acting implies --oneply" BEFORE
        calling this;
      * the default is NOT a no-op: an existing `--oneply --device
        cuda` command is auto-routed onto the vectorized collector
        (bit-exact, so a speed choice, not a results change); a run
        that must keep the CPU collector says --no-vectorized-collect.
    The CUDA/workers availability checks stay in Trainer.train (which
    also runs the VRAM preflight and can fall back loudly).
    """
    notes = []
    if args.vectorized_collect and args.no_vectorized_collect:
        raise SystemExit(
            "ERROR: --vectorized-collect and --no-vectorized-collect "
            "are mutually exclusive")
    if not (args.oneply and args.device == "cuda"):
        return notes
    if args.no_vectorized_collect:
        notes.append(
            "NOTE: --no-vectorized-collect: keeping the CPU collector; "
            "the master training step runs on cuda.")
        return notes
    if args.vectorized_collect:
        return notes
    if args.c_engine or args.bf16_collect:
        notes.append(
            "NOTE: --oneply --device cuda would default to the "
            "vectorized GPU collector, but an explicit CPU-collector "
            "accelerator (--c-engine/--bf16-collect) was chosen and "
            "wins. Drop it to use the GPU collector.")
        return notes
    if args.oneply_acting:
        # The vectorized collector refuses --oneply-acting (hard error in
        # Trainer.train); auto-routing would turn a valid command into
        # that error, blaming a flag the user never passed.
        notes.append(
            "NOTE: --oneply-acting is not supported by the vectorized "
            "GPU collector; keeping the CPU collector. The master "
            "training step runs on cuda.")
        return notes
    args.vectorized_collect = True
    notes.append(
        "AUTO: --oneply --device cuda -> vectorized GPU collector "
        "(pass --no-vectorized-collect for the CPU collector; the VRAM "
        "preflight may still cap workers or fall back, and will say so).")
    return notes


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
    g_batch.add_argument("--episodes-per-round", type=int, default=1000,
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
                              "first N rounds. For task transfers, where a "
                              "fresh head sits on a trained trunk.")
    g_batch.add_argument("--c-engine", action="store_true",
                         help="compute the 1-ply cubeful target's net-forward "
                              "in C via libbg_nply (OpenBLAS sgemm) instead of "
                              "PyTorch. Requires --oneply and "
                              "c_engine/libbg_nply.so (build_nply.sh). Targets "
                              "are numerically equivalent; only the forward "
                              "moves to C. ~+17%% on top of --pipeline-collect.")
    g_batch.add_argument("--bf16-collect", action="store_true",
                         help="Run 1-ply target NN forward in bf16 inside "
                              "collection workers (~1.9x matmul speedup). "
                              "Master training step stays fp32. Introduces "
                              "~1e-3 numerical drift in 1-ply targets.")
    g_batch.add_argument("--pipeline-collect", action="store_true",
                         help="Overlap worker collection of round N+1 with the "
                              "master training step of round N. Workers in round "
                              "N+1 see weights from end of round N-1 (one-round "
                              "stale targets). Negligible TD impact at typical "
                              "1-ply LRs but should be validated when changing "
                              "recipes. Reproducibility preserved. Requires "
                              "--workers > 1; no-op otherwise.")
    g_batch.add_argument("--max-episode-plies", type=int,
                         default=DEFAULT_MAX_EPISODE_PLIES, metavar="N",
                         help="Safety bound on self-play episode length, in "
                              "plies (turns). An episode exceeding it is "
                              "DISCARDED (its samples are dropped, not "
                              "truncated) and counted in the round log line. "
                              "Guards against a degenerate policy producing "
                              "non-terminating games, which otherwise hang the "
                              "round barrier forever while worker RSS grows "
                              "until the OOM killer fires. Never triggers on "
                              "healthy play (healthy max is ~341 samples "
                              "i.e. well under 1500 plies). 0 disables.")
    g_batch.add_argument("--vectorized-collect", action="store_true",
                         help="Collect on the GPU: each worker process runs "
                              "--vectorized-batch games in lockstep as "
                              "generators and batches every pending network "
                              "forward across those games into ONE local CUDA "
                              "call (no IPC, no inference server). Requires "
                              "--oneply, --workers > 1 and CUDA; warns and "
                              "falls back to the CPU collector otherwise. "
                              "DISABLES --pipeline-collect, --c-engine and "
                              "--bf16-collect (each logged explicitly). Use a "
                              "large --episodes-per-round (>= workers * "
                              "batch * 8) or the lockstep tail-drain eats the "
                              "throughput gain. --max-episode-plies is "
                              "enforced per game, as on the CPU paths.")
    g_batch.add_argument("--no-vectorized-collect", action="store_true",
                         help="Keep the CPU collector even when --oneply "
                              "--device cuda would default to the vectorized "
                              "GPU collector. The master training step still "
                              "runs on cuda. Mutually exclusive with an "
                              "explicit --vectorized-collect.")
    g_batch.add_argument("--vectorized-batch", type=int, default=32,
                         metavar="B",
                         help="Games in flight per worker process for "
                              "--vectorized-collect (default 32). Larger B "
                              "means bigger GPU batches but more GPU memory "
                              "per process and a longer lockstep tail.")
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

    # --oneply-acting implies --oneply for targets. Normalize BEFORE the
    # collector routing so the resolver sees the real ply config (it must
    # not depend on flag-spelling: --oneply-acting alone and
    # --oneply --oneply-acting have to route identically).
    if args.oneply_acting:
        args.oneply = True

    for note in resolve_cuda_oneply_collector(args):
        print(note, flush=True)

    apply_cubeful_money_upgrades(args)

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)

    net = build_network(args)
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
        c_engine=args.c_engine,
        save_path=resolved_save,
        save_every=args.save_every,
        pipeline_collect=args.pipeline_collect,
        max_episode_plies=args.max_episode_plies,
        vectorized_collect=args.vectorized_collect,
        vectorized_batch=args.vectorized_batch,
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
