#!/usr/bin/env python3
"""
train_batch.py -- Batch TD(0) self-play training for backgammon.

Round-based: each round collects `episodes_per_round` self-play
episodes, shuffles them, and trains `epochs_per_round` passes in
fixed-size batches. Default optimizer is Adam. Thin CLI over
trainer.Trainer.train; shared flags live in train_cli.py.

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

# Pin BLAS / PyTorch threads BEFORE importing torch: collection workers
# must be single-threaded.
import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import time

import torch
torch.set_num_threads(1)

from c_api.puretd_py import BACKENDS as CAPI_BACKENDS
from td_agent import TDAgent
from train_cli import (
    add_common_args, apply_cubeful_money_upgrades,
    build_mode, build_network, eval_vs_random, resolve_save_path,
)
from trainer import (DEFAULT_MAX_EPISODE_PLIES, CCollectUnavailable, Trainer,
                     c_collect_fallbacks, c_collect_library)


def c_collect_default_fallbacks(args, net, mode):
    """Why the DEFAULT C collector cannot collect this run: the
    preconditions (trainer.c_collect_fallbacks), else a libpuretd that
    cannot be built or loaded (built on demand here). [] when it can, or
    when the default does not apply (an explicit --c-collect,
    --no-c-collect or --vectorized-collect decides on its own). Needs the
    built network and mode; resolve_cuda_oneply_collector takes the
    result."""
    if (getattr(args, "c_collect", False) or getattr(args, "no_c_collect", False)
            or args.vectorized_collect):
        return []
    reasons = c_collect_fallbacks(
        net, mode, oneply_acting=args.oneply_acting,
        boltzmann_temp=args.boltzmann_temp, bf16_collect=args.bf16_collect,
        c_engine=args.c_engine)
    if not reasons:
        why = c_collect_library(args.c_collect_backend)
        if why is not None:
            reasons = [why]
    return reasons


def resolve_cuda_oneply_collector(args, c_fallbacks=None):
    """Pick the collector: the C collector (the default),
    the vectorized GPU collector or the Python CPU collector.

    `c_fallbacks`: c_collect_default_fallbacks(), the reasons the default
    C collector cannot collect this run (None: not known, the C collector
    is assumed to apply).

    Mutates only args.c_collect / args.c_collect_auto /
    args.vectorized_collect / args.vectorized_auto and returns the notes
    to print: one per fallback reason, then at most one routing note.

    Rules:
      * --c-collect with --no-c-collect, --vectorized-collect with
        --no-vectorized-collect, and --c-collect with
        --vectorized-collect are errors;
      * the C collector is the default (args.c_collect_auto); --c-collect
        is accepted (it only makes a missing library an error);
      * the default with `c_fallbacks`: a NOTE per reason, then the
        non-C routing (below), so `--oneply --device
        cuda` auto-routes to the vectorized GPU collector.
        An explicit --c-collect keeps Trainer.train's NOTE + Python
        collector instead;
      * an explicit --vectorized-collect wins over the default (NOTE);
      * the C collector is a CPU collector: --oneply --device cuda keeps
        it (NOTE), and --no-vectorized-collect does not change it (NOTE;
        with --no-c-collect it means the Python CPU collector);
      * --no-c-collect: the non-C routing.
        --oneply --device cuda auto-enables the vectorized GPU collector
        unless --no-vectorized-collect, a CPU-collector accelerator
        (--c-engine / --bf16-collect) or --oneply-acting keeps the CPU
        collector.
    Callers must apply "--oneply-acting implies --oneply" BEFORE calling
    this. CUDA/workers checks and the VRAM preflight stay in
    Trainer.train.
    """
    notes = []
    explicit_c = bool(getattr(args, "c_collect", False))
    no_c = bool(getattr(args, "no_c_collect", False))
    if explicit_c and no_c:
        raise SystemExit(
            "ERROR: --c-collect and --no-c-collect/--python-api are "
            "mutually exclusive")
    if args.vectorized_collect and args.no_vectorized_collect:
        raise SystemExit(
            "ERROR: --vectorized-collect and --no-vectorized-collect "
            "are mutually exclusive")
    if explicit_c and args.vectorized_collect:
        raise SystemExit(
            "ERROR: --c-collect and --vectorized-collect are mutually "
            "exclusive")
    default_c = not (explicit_c or no_c or args.vectorized_collect)
    fallback = default_c and bool(c_fallbacks)
    args.c_collect_auto = default_c and not fallback
    args.c_collect = explicit_c or args.c_collect_auto
    oneply_cuda = args.oneply and args.device == "cuda"
    if args.c_collect:
        who = "--c-collect" if explicit_c else "the C collector (default)"
        if args.no_vectorized_collect:
            notes.append(
                f"NOTE: --no-vectorized-collect: {who} collects (a CPU "
                f"collector); add --no-c-collect for the Python CPU "
                f"collector.")
        elif oneply_cuda and not (args.c_engine or args.bf16_collect
                                  or args.oneply_acting):
            # (Those three keep the Python collector; Trainer.train's
            # NOTE says so.)
            notes.append(
                f"NOTE: {who}: --oneply --device cuda keeps the C collector "
                f"(no auto-routing to the vectorized GPU collector: "
                f"--vectorized-collect or --no-c-collect for it); the master "
                f"training step runs on cuda.")
        return notes
    if args.vectorized_collect and not no_c:
        notes.append(
            "NOTE: --vectorized-collect: the vectorized GPU collector "
            "replaces the default C collector (--c-collect with it is an "
            "error).")
        return notes
    route = _non_c_routing(args, oneply_cuda)
    if fallback:
        where = ("the vectorized GPU collector" if args.vectorized_collect
                 else "the Python collector")
        notes += [f"NOTE: --c-collect (default): {why}; falling back to "
                  f"{where} (the --no-c-collect routing)."
                  for why in c_fallbacks]
    return notes + route


def _non_c_routing(args, oneply_cuda):
    """resolve_cuda_oneply_collector's routing without the C collector
    (--no-c-collect, or the default ruled out): the notes to print."""
    notes = []
    if not oneply_cuda:
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
        # Auto-routing must not turn a valid command into Trainer.train's
        # --oneply-acting error.
        notes.append(
            "NOTE: --oneply-acting is not supported by the vectorized "
            "GPU collector; keeping the CPU collector. The master "
            "training step runs on cuda.")
        return notes
    args.vectorized_collect = True
    args.vectorized_auto = True   # preflight fallbacks are stricter for auto
    notes.append(
        "AUTO: --oneply --device cuda -> vectorized GPU collector "
        "(shared-context topology by default: the eval-server process "
        "holds the only CUDA context, ~1-2 GB of VRAM regardless of "
        "worker count; --vectorized-per-worker-context for one context "
        "per worker, whose VRAM preflight may cap workers or fall back "
        "and will say so; --no-vectorized-collect for the CPU collector).")
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
                              "Lower variance, several times slower per turn. "
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
                              "moves to C.")
    g_batch.add_argument("--bf16-collect", action="store_true",
                         help="Run the 1-ply target forward in bf16 inside "
                              "collection workers; the master training step "
                              "stays fp32. ~1e-3 drift in 1-ply targets.")
    g_batch.add_argument("--pipeline-collect", action="store_true",
                         help="Overlap worker collection of round N+1 with the "
                              "master training step of round N: workers see "
                              "one-round-stale weights. Requires --workers > 1; "
                              "no-op otherwise.")
    g_batch.add_argument("--max-episode-plies", type=int,
                         default=DEFAULT_MAX_EPISODE_PLIES, metavar="N",
                         help="Safety bound on self-play episode length, in "
                              "plies (turns). An episode exceeding it is "
                              "DISCARDED (its samples are dropped, not "
                              "truncated) and counted in the round log line. "
                              "Guards against a degenerate policy producing "
                              "non-terminating games; healthy episodes stay "
                              "far below it (longest seen ~341 samples). "
                              "0 disables.")
    g_batch.add_argument("--vectorized-collect", action="store_true",
                         help="Collect on the GPU, replacing the default C "
                              "collector: each worker process runs "
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
                         help="With --no-c-collect: keep the Python CPU "
                              "collector even when --oneply --device cuda "
                              "would auto-route to the vectorized GPU "
                              "collector (without it the C collector "
                              "collects anyway). The master training step "
                              "still runs on cuda. Mutually exclusive with "
                              "an explicit --vectorized-collect.")
    g_batch.add_argument("--vectorized-workers", type=int, default=None,
                         metavar="N",
                         help="Exact number of GPU collection worker "
                              "processes. Default: min(--workers, physical "
                              "cores, 32) -- 1-ply throughput is flat well "
                              "before that and, on the per-worker-context "
                              "collector, every extra worker is another "
                              "CUDA context. The VRAM preflight may still "
                              "cap it lower.")
    g_batch.add_argument("--vectorized-shared-context", action="store_true",
                         help="GPU collector topology: ONE CUDA context in an "
                              "eval-server process; workers stay on the CPU "
                              "and ship their fused lockstep batches through "
                              "shared memory. VRAM is independent of worker "
                              "count.")
    g_batch.add_argument("--vectorized-per-worker-context",
                         action="store_true",
                         help="GPU collector topology: the net on the GPU in "
                              "every worker process (one CUDA context per "
                              "worker; VRAM grows linearly with workers).")
    g_batch.add_argument("--gpu-eval-servers", type=int, default=None,
                         metavar="N",
                         help="Shared-context collector: number of eval-server "
                              "processes to shard the workers across (each is "
                              "one small CUDA context). Default: one per 16 "
                              "workers.")
    g_batch.add_argument("--vectorized-batch", type=int, default=32,
                         metavar="B",
                         help="Games in flight per worker process for "
                              "--vectorized-collect (default 32). Larger B "
                              "means bigger GPU batches but more GPU memory "
                              "per process and a longer lockstep tail.")
    # default=SUPPRESS: the help must not print "(default: False)" for
    # the default collector; resolve_cuda_oneply_collector sets the attr.
    g_batch.add_argument("--c-collect", action="store_true",
                         default=argparse.SUPPRESS,
                         help="The default collector (this flag is "
                              "accepted for old command lines; it only "
                              "makes a library that cannot be built an "
                              "error): the self-play episodes are "
                              "collected in C (bgc_collect, c_api/; a "
                              "missing libpuretd_<backend>.so is built on "
                              "demand with `bash c_api/build.sh "
                              "<backend>`), the samples the Python "
                              "collector produces for the same seed, 0-ply "
                              "play, 0-/1-ply targets, dmp / "
                              "cubeless-money / cubeful-money with the "
                              "perspective196 / cubeful_perspective196 "
                              "encoders. Works with --pipeline-collect, "
                              "--save-every and --workers 1. Ruled out, "
                              "with a NOTE per reason, by --boltzmann-temp "
                              "!= 0, --oneply-acting, --bf16-collect, "
                              "--c-engine, another encoder, dmp with an "
                              "equity net, or a library that cannot be "
                              "built: the default then routes as "
                              "--no-c-collect does (with --oneply --device "
                              "cuda: the vectorized GPU collector), an "
                              "explicit --c-collect keeps the Python "
                              "collector. An explicit --vectorized-collect "
                              "replaces it (with --c-collect: an error); "
                              "--oneply --device cuda keeps it. Rows reach "
                              "the master through shared memory "
                              "(row_transport.py; BG_COLLECT_SHM=0 for the "
                              "pool pipe, same pool, bitwise).")
    g_batch.add_argument("--no-c-collect", "--python-api", action="store_true",
                         dest="no_c_collect",
                         help="Collect with the Python collector, or with "
                              "--oneply --device cuda the vectorized GPU "
                              "collector. --python-api "
                              "is the same flag (play_models.py's name).")
    g_batch.add_argument("--c-collect-backend", choices=CAPI_BACKENDS,
                         default="simd",
                         help="libpuretd_<backend>.so the C collection "
                              "workers load (c_api/build.sh <backend>).")
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

    # --oneply-acting implies --oneply. Normalize BEFORE the collector
    # routing so both spellings route identically.
    if args.oneply_acting:
        args.oneply = True

    if args.vectorized_shared_context and args.vectorized_per_worker_context:
        raise SystemExit(
            "ERROR: --vectorized-shared-context and "
            "--vectorized-per-worker-context are mutually exclusive")
    shared_ctx = (True if args.vectorized_shared_context
                  else False if args.vectorized_per_worker_context
                  else None)   # None -> gpu_context.VEC_SHARED_CONTEXT_DEFAULT

    apply_cubeful_money_upgrades(args)

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)

    net = build_network(args)
    agent = TDAgent(net, device=args.device, oneply=args.oneply_acting,
                    boltzmann_temp=args.boltzmann_temp)
    mode = build_mode(args.game_mode, jacoby=args.jacoby)

    # The collector, once the network is built: whether the default C
    # collector applies depends on its encoder and output (a run it rules
    # out gets the non-C routing).
    for note in resolve_cuda_oneply_collector(
            args, c_collect_default_fallbacks(args, net, mode)):
        print(note, flush=True)

    # Equity output is unbounded. A high lr can blow up to NaN. SGD is
    # more sensitive than Adam.
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
    # Resolved once so checkpoints and the final save agree.
    resolved_save = (
        resolve_save_path(args.save, args.game_mode, net.hidden_sizes)
        if args.save else None
    )
    if args.save_every > 0 and not resolved_save:
        raise ValueError("--save-every requires --save")

    # Stamped into every checkpoint. Trainer.train sets c_collect to
    # whether the C collector collected (a failed precondition or a
    # missing library falls back to Python).
    train_params = {
        "game_mode": args.game_mode,
        "encoder": net.encoder_name,
        "num_episodes": args.num_episodes,
        "lr": args.lr,
        "end_lr": args.end_lr,
        "seed": args.seed,
        "oneply": args.oneply,
        "jacoby": args.jacoby,
        "c_collect": bool(args.c_collect),
    }

    t0 = time.perf_counter()
    try:
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
            vectorized_workers=args.vectorized_workers,
            vectorized_shared_context=shared_ctx,
            gpu_eval_servers=args.gpu_eval_servers,
            vectorized_auto=getattr(args, "vectorized_auto", False),
            train_params=train_params,
            c_collect=args.c_collect,
            c_collect_backend=args.c_collect_backend,
            c_collect_auto=args.c_collect_auto,
        )
    except CCollectUnavailable as e:
        raise SystemExit(f"ERROR: {e}")
    elapsed = time.perf_counter() - t0
    eps_per_s = args.num_episodes / elapsed if elapsed > 0 else 0.0
    print(
        f"Done in {elapsed:.1f}s ({eps_per_s:.1f} ep/s, "
        f"{len(losses)} batches)"
    )

    if resolved_save:
        net.save(resolved_save, train_params=train_params)
        print(f"Saved network to {resolved_save}")

    if args.eval_vs_random > 0:
        wr = eval_vs_random(agent, n_games=args.eval_vs_random)
        print(f"vs Random ({args.eval_vs_random} games): {wr:.1%}")


if __name__ == "__main__":
    main()
