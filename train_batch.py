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

from td_agent import TDAgent
from train_cli import (
    add_common_args, apply_cubeful_money_upgrades,
    build_mode, build_network, eval_vs_random, resolve_save_path,
)
from trainer import (C_COLLECT_BACKENDS, C_COLLECT_RNG_MODES,
                     DEFAULT_MAX_EPISODE_PLIES, Trainer, c_collect_error,
                     c_collect_fallbacks, c_collect_library)


def c_collect_default_fallbacks(args, net, mode):
    """Why the DEFAULT C collector (money modes) cannot collect this run:
    the preconditions (trainer.c_collect_fallbacks), else a libpuretd that
    cannot be built or loaded (built on demand here). [] when it can, or
    when the default does not apply (match play, an explicit --c-collect
    or --no-c-collect decides on its own). Needs the built network and
    mode; resolve_collector takes the result."""
    if (args.game_mode == "matchplay" or getattr(args, "c_collect", False)
            or args.no_c_collect):
        return []
    reasons = c_collect_fallbacks(
        net, mode, oneply_acting=args.oneply_acting,
        boltzmann_temp=args.boltzmann_temp, bf16_collect=args.bf16_collect)
    if not reasons:
        why = c_collect_library(args.c_collect_backend or "simd")
        if why is not None:
            reasons = [why]
    return reasons


def resolve_collector(args, c_fallbacks=None):
    """Pick the collector: the C collector or the Python collector.

    `c_fallbacks`: c_collect_default_fallbacks(), the reasons the default
    C collector cannot collect this run (None: not known, the C collector
    is assumed to apply).

    Mutates only args.c_collect / args.c_collect_auto and returns the
    notes to print: one per fallback reason, then at most one routing
    note.

    Rules:
      * --c-collect with --no-c-collect is an error;
      * match play: the Python collector unless --c-collect (an explicit
        opt-in; Trainer.train errors on any unmet precondition);
      * money modes: the C collector is the default (args.c_collect_auto);
        --c-collect is accepted (it only makes a missing library an
        error); the default with `c_fallbacks`: a NOTE per reason and the
        Python collector, while an explicit --c-collect keeps
        Trainer.train's NOTE + Python collector; --no-c-collect: the
        Python collector; the C collector is a CPU collector, so --oneply
        --device cuda keeps it (NOTE).
    """
    notes = []
    explicit_c = bool(getattr(args, "c_collect", False))
    no_c = bool(args.no_c_collect)
    if explicit_c and no_c:
        raise SystemExit(
            "ERROR: --c-collect and --no-c-collect/--python-api are "
            "mutually exclusive")
    if args.game_mode == "matchplay":
        args.c_collect_auto = False
        args.c_collect = explicit_c
        return notes
    default_c = not (explicit_c or no_c)
    fallback = default_c and bool(c_fallbacks)
    args.c_collect_auto = default_c and not fallback
    args.c_collect = explicit_c or args.c_collect_auto
    if args.c_collect:
        if args.oneply and args.device == "cuda" and not (
                args.bf16_collect or args.oneply_acting):
            # (Those two keep the Python collector; Trainer.train's NOTE
            # says so.)
            who = "--c-collect" if explicit_c else "the C collector (default)"
            notes.append(
                f"NOTE: {who}: --oneply --device cuda keeps the C collector "
                f"(a CPU collector; --no-c-collect for the Python "
                f"collector); the master training step runs on cuda.")
        return notes
    if fallback:
        notes += [f"NOTE: --c-collect (default): {why}; falling back to "
                  f"the Python collector (the routing from before the "
                  f"default)." for why in c_fallbacks]
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
    g_opt.add_argument("--resume-episodes", type=int, default=0, metavar="N",
                       help="Crash recovery: the --resume checkpoint already "
                            "has N episodes of THIS schedule (e.g. from its "
                            "_ep{N} / _ep{N}_sigterm name). Pass the ORIGINAL "
                            "--lr/--end-lr/--num-episodes; the run then starts "
                            "at lr(N) = lr + (end_lr - lr) * N / num_episodes, "
                            "trains the remaining num_episodes - N, and names "
                            "checkpoints by the global episode count. Give a "
                            "new --seed, or the worker seeds replay from ep 0.")
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
    g_batch.add_argument("--bf16-collect", action="store_true",
                         help="Run the 1-ply target forward in bf16 inside "
                              "collection workers; the master training step "
                              "stays fp32. ~1e-3 drift in 1-ply targets.")
    g_batch.add_argument("--oracle-targets", default=None, metavar="SPEC",
                         help="Supervised oracle training (matchplay "
                              "only): prob5 checkpoint path, optional "
                              "':x=V'. Acting stays with the trained "
                              "net; every collected target is replaced "
                              "by the p5jan oracle's value of the "
                              "encoded state (janowski_match). "
                              "Incompatible with --oneply, --met and "
                              "--outcome-mix > 0.")
    g_batch.add_argument("--oracle-eval-server", action="store_true",
                         help="With --oracle-targets: host the teacher "
                              "net on ONE shared GPU process; workers "
                              "ship their 1-ply batches to it over "
                              "shared memory (one CUDA context total "
                              "instead of one per worker).")
    g_batch.add_argument("--oracle-plies", type=int, default=0,
                         choices=[0, 1], metavar="N",
                         help="With --oracle-targets: teacher lookahead "
                              "depth. 1 prices every row with the "
                              "oracle's own 1-ply value (21-dice "
                              "enumeration, exact MET terminals) — "
                              "stronger targets, ~20x collection cost. "
                              "--oracle-move-eps rows are priced at "
                              "the same depth (uniform-target "
                              "invariant).")
    g_batch.add_argument("--oracle-move-eps", type=float, default=None,
                         metavar="EPS",
                         help="With --oracle-targets: also supervise "
                              "every legal successor whose oracle value "
                              "is within EPS (MWC) of the oracle's best "
                              "move -- the near-optimal action set the "
                              "net must rank correctly.")
    g_batch.add_argument("--outcome-mix", type=float, default=0.0,
                         metavar="BETA",
                         help="Matchplay: blend BETA of the realized "
                              "match outcome into every collected "
                              "target (t = (1-BETA)*backup + BETA*z). "
                              "Grounds cube-decision samples, which "
                              "pure TD(0) never grounds; BETA=0.3 "
                              "validated on stuck cube positions.")
    g_batch.add_argument("--met", default=None, metavar="TABLE",
                         help="Matchplay: ground all cross-game equity in a "
                              "real match equity table instead of the "
                              "network's own new-game evaluation. Accepts "
                              "'auto' (the bundled "
                              "met_tables/Kazaross-XG2.xml, "
                              "falling back to gnubg's system table in "
                              "/usr/share/gnubg/met), a table name from "
                              "/usr/share/gnubg/met, or a path to a MET "
                              "XML. This pins v_dp, the take "
                              "point, terminal gammon pricing and the "
                              "collector's game-boundary target, all of "
                              "which are otherwise bootstraps off the net's "
                              "own values. Omit to keep the old "
                              "implicit-MET behaviour.")
    g_batch.add_argument("--met-scope", default="all", metavar="SCOPES",
                         help="Which consumers --met grounds, comma-separated: "
                              "'cube' (v_dp, the double/pass branch -- a PURE "
                              "table lookup with no board dependence), "
                              "'terminal' (gammon pricing of terminal "
                              "successors), 'boundary' (the collector's "
                              "game-boundary target, the only one that trains "
                              "the net's own scale toward the table), or "
                              "'all'. v_nd/v_dt are board-dependent and can "
                              "never be grounded from a table.")
    g_batch.add_argument("--pipeline-collect", action="store_true",
                         help="Overlap worker collection of round N+1 with the "
                              "master training step of round N: workers see "
                              "one-round-stale weights. Requires --workers > 1; "
                              "no-op otherwise.")
    g_batch.add_argument("--vectorized-collect", action="store_true",
                         help="MATCHPLAY + --oneply only. Collect with the "
                              "lockstep generator driver "
                              "(vectorized_matchplay.py): each worker runs "
                              "--vectorized-batch matches concurrently and "
                              "batches all their pending network forwards "
                              "into one GPU call per step, instead of one "
                              "CPU forward per game per step. Produces "
                              "equivalent training data (bit-identical when "
                              "driven single-game on CPU). Requires "
                              "--workers > 1, CUDA and --boltzmann-temp 0; "
                              "prints a WARN and falls back otherwise. "
                              "Set --episodes-per-round >= workers x "
                              "--vectorized-batch or the fused batches "
                              "never reach B (2-3x slower). "
                              "Disables --pipeline-collect.")
    g_batch.add_argument("--oracle-act-only", action="store_true",
                         help="Use the teacher for MOVE SELECTION only: no "
                              "oracle value targets are emitted, so training "
                              "targets come from the normal (--oneply) TD "
                              "path. Requires --oracle-targets, "
                              "--oracle-move-eps and --oracle-act-prob > 0. "
                              "Motivation: p5jan outplays the champion at "
                              "checker by 5.84 mMWC, yet distilling its VALUES "
                              "damages the net -- its ranking is better than "
                              "its calibration. This transfers the ranking "
                              "through the visited trajectory while keeping "
                              "the student's own value targets.")
    g_batch.add_argument("--td-eps-only", action="store_true",
                         help="MATCHPLAY + --td-move-eps. Drop the "
                              "on-trajectory CHECKER rows, keeping the eps "
                              "rows, the cube rows and the game-boundary "
                              "rows. Pure self-distillation of the 1-ply "
                              "value into the 0-ply read, with no "
                              "bootstrapped trajectory target. Isolates "
                              "whether those rows are what degrades a "
                              "converged net.")
    g_batch.add_argument("--td-oneply-act", type=int, default=0, metavar="K",
                         help="MATCHPLAY + --oneply. EXPERT ITERATION: the "
                              "collector ACTS with 1-ply move selection over "
                              "the top-K successors by the 0-ply read, "
                              "instead of acting 0-ply. 1-ply selection "
                              "measures +2.00pp over 0-ply on identical "
                              "weights, so trajectories come from the "
                              "stronger policy and the net learns to "
                              "reproduce choices it cannot yet make at 0 "
                              "ply. K=3 matched FULL 1-ply selection on the "
                              "gnubg-confirmed set at 5.2x less compute. "
                              "0 = act 0-ply (default).")
    g_batch.add_argument("--oracle-1ply-eps", type=float, default=0.0,
                         metavar="EPS",
                         help="MATCHPLAY + --oracle-plies>0. Use the CHEAP "
                              "0-ply oracle as a filter for the expensive "
                              "deep one: only successors within EPS of the "
                              "0-ply oracle's best move are re-priced at "
                              "--oracle-plies; the rest keep their 0-ply "
                              "value. The deep call is ~20x a shallow one "
                              "and dominates the leg, so this is where the "
                              "cost is. WARNING: mixes target depths within "
                              "a leg -- see _oracle_move_rows. 0 = off.")
    g_batch.add_argument("--td-move-eps", type=float, default=0.0,
                         metavar="EPS",
                         help="MATCHPLAY + --oneply. Also emit a training "
                              "row for every legal successor within EPS "
                              "(MWC) of the best 0-ply value, each targeted "
                              "at its OWN 1-ply value. Does NOT change which "
                              "move is played, so the state distribution is "
                              "untouched (unlike --move-explore-eps). Closes "
                              "the coverage hole: ~540 successor reads feed "
                              "each 1-ply target but only the played one "
                              "becomes a row, leaving the skipped siblings "
                              "undervalued by ~12 mMWC. Try 0.01.")
    g_batch.add_argument("--move-explore-eps", type=float, default=0.0,
                         metavar="EPS",
                         help="MATCHPLAY. Fraction of checker decisions that "
                              "execute a RANDOM legal move instead of the "
                              "greedy one (default 0.0 = fully greedy, the "
                              "historical behaviour). Targets are unaffected: "
                              "under --oneply the target is a max over "
                              "successors computed from the state, and in the "
                              "0-ply path the greedy successor's bootstrap is "
                              "kept -- so this is off-policy TD, not the "
                              "random policy's value. Fixes the closed loop "
                              "where greedy play only ever visits the states "
                              "the net already prefers.")
    g_batch.add_argument("--oracle-act-prob", type=float, default=0.0,
                         metavar="BETA",
                         help="Requires --oracle-targets and "
                              "--oracle-move-eps. Probability of walking "
                              "into the TEACHER's move instead of the "
                              "student's at each checker decision (DAgger "
                              "beta-mix; default 0.0 = student acts, the "
                              "historical behaviour). The teacher's move is "
                              "already computed as the eps-set anchor, so "
                              "this is free. BETA=1.0 is pure behaviour "
                              "cloning: the student then never visits the "
                              "states its own errors lead to, so prefer an "
                              "intermediate value. Cube decisions are "
                              "unaffected.")
    g_batch.add_argument("--vectorized-batch", type=int, default=32,
                         metavar="B",
                         help="Concurrent matches per vectorized worker "
                              "(default 32, must be >= 1). Larger B means "
                              "bigger fused GPU batches and more RSS per "
                              "worker (~730 MB/driver at B=32).")
    g_batch.add_argument("--vectorized-device", default="cuda",
                         help="Device for the vectorized driver net "
                              "(default cuda). 'cpu' is for testing only. "
                              "NOTE: every worker holds its OWN CUDA "
                              "context (~500 MB), so --workers is bounded "
                              "by VRAM (~20 on a 12 GB card), not by "
                              "cores; the trainer refuses and falls back "
                              "if they will not fit.")
    g_batch.add_argument("--gpu-eval-server", action="store_true",
                         help="MATCHPLAY + --oneply only. The OTHER GPU "
                              "collection design (see --vectorized-collect): "
                              "start ONE server process holding the only "
                              "CUDA context and have the CPU workers ship "
                              "feature matrices to it over per-worker shared "
                              "memory. VRAM grows with the width of the fused "
                              "batch rather than with the number of CUDA "
                              "contexts (measured 412 MB at 8 slots, 1262 MB "
                              "at a saturated 48, vs ~475 MB PER worker for "
                              "--vectorized-collect), at the cost of one "
                              "queue round-trip per forward. Routes the "
                              "1-ply target forward (96%% of collected rows). "
                              "NOT bitwise reproducible unless you add "
                              "--gpu-eval-deterministic. Requires --workers > "
                              "1 and CUDA; prints a WARN and falls back "
                              "otherwise. REJECTED (hard error) together with "
                              "--pipeline-collect, --vectorized-collect, or "
                              "TRAINER_MAXTASKSPERCHILD.")
    g_batch.add_argument("--gpu-eval-deterministic", action="store_true",
                         help="--gpu-eval-server only: one request per GPU "
                              "call, no cross-slot fusion. Fused batches are "
                              "composed by wall-clock arrival order and fp32 "
                              "reductions are batch-composition sensitive, so "
                              "by default a seeded routed run does NOT "
                              "reproduce bit for bit (a last-ulp difference "
                              "can flip a min() tie and change the "
                              "trajectory). With this flag routed collection "
                              "is bit-exact against the unrouted collector at "
                              "any --workers, at the cost of batching.")
    g_batch.add_argument("--gpu-eval-device", default="cuda",
                         help="Device for the eval server's net (default "
                              "cuda). 'cpu' is for testing only. Unlike "
                              "--vectorized-device this is ONE context "
                              "regardless of --workers.")
    g_batch.add_argument("--gpu-eval-route-moves", action="store_true",
                         help="Also route 0-ply move selection "
                              "(choose_checker_action_matchplay) through the "
                              "eval server. That call site is only 4%% of "
                              "rows but ~50%% of forwards (mean batch 18), so "
                              "it roughly doubles the server request rate to "
                              "capture a twentieth of the FLOPs -- separate "
                              "flag so the A/B can isolate it. Batch-1 "
                              "evaluate_matchplay is never routed.")
    g_batch.add_argument("--max-episode-plies", type=int, default=None,
                         metavar="N",
                         help="Safety bound on self-play episode length, in "
                              "plies (turns). Money modes (default "
                              f"{DEFAULT_MAX_EPISODE_PLIES}): an episode "
                              "exceeding it is DISCARDED (its samples are "
                              "dropped, not truncated) and counted in the "
                              "round log line; healthy episodes stay far "
                              "below it. Match play (default 0 = off): a "
                              "per-GAME bound of --vectorized-collect and "
                              "--c-collect only; a match exceeding it is "
                              "discarded whole (a runaway game in lockstep "
                              "pins a driver slot and stalls the round). The "
                              "serial match collector does not apply it, and "
                              "a --vectorized-collect fallback says so. 0 "
                              "disables.")
    # default=SUPPRESS: the help must not print "(default: False)" for
    # the default money collector; resolve_collector sets the attr.
    g_batch.add_argument("--c-collect", action="store_true",
                         default=argparse.SUPPRESS,
                         help="Money modes: the default collector (this flag "
                              "is accepted for old command lines; it only "
                              "makes a library that cannot be built an "
                              "error): the self-play episodes are collected "
                              "in C (bgc_collect, c_api/; a missing "
                              "libpuretd_<backend>.so is built on demand "
                              "with `bash c_api/build.sh <backend>`), the "
                              "samples the Python collector produces for the "
                              "same seed, 0-ply play, 0-/1-ply targets, dmp / "
                              "cubeless-money / cubeful-money with the "
                              "perspective196 / cubeful_perspective196 "
                              "encoders. Ruled out, with a NOTE per reason "
                              "and the Python collector, by --boltzmann-temp "
                              "!= 0, --oneply-acting, --bf16-collect, another "
                              "encoder, dmp with an equity net, or a library "
                              "that cannot be built. "
                              "MATCHPLAY (an explicit opt-in; without it the "
                              "Python collector collects): collect every "
                              "round in C "
                              "(bgc_collect_match, c_api/): "
                              "0-ply play, 0-/1-ply targets, the samples the "
                              "Python collector produces for the same seed, "
                              "on --c-collect-backend; the workers hold no "
                              "CUDA context, so --workers is bounded by "
                              "cores (W=60 on a 64-core box). Needs --met "
                              "with all scopes, the matchplay1h64met / "
                              "matchplay1h64met6 encoders, --boltzmann-temp "
                              "0. Any unmet precondition (--vectorized-"
                              "collect, --gpu-eval-server, oracle / "
                              "--td-move-eps / "
                              "--move-explore-eps / --td-oneply-act, a MET "
                              "mismatch, ...) is "
                              "an ERROR naming every reason; there is no "
                              "silent fallback. Works with --pipeline-"
                              "collect (one-round-stale targets), "
                              "--save-every, --resume and --workers 1.")
    g_batch.add_argument("--no-c-collect", "--python-api", action="store_true",
                         dest="no_c_collect",
                         help="Money modes: collect with the Python collector "
                              "(the routing from before the C collector became "
                              "the default). --python-api is the same flag "
                              "(play_models.py's name). Match play collects "
                              "in Python unless --c-collect.")
    g_batch.add_argument("--c-collect-backend", choices=C_COLLECT_BACKENDS,
                         default=None,
                         help="The libpuretd_<backend>.so the C collection "
                              "workers load (built on demand with "
                              "c_api/build.sh). Default simd. Match play: "
                              "with --c-collect only.")
    g_batch.add_argument("--c-collect-rng", choices=sorted(C_COLLECT_RNG_MODES),
                         default=None,
                         help="MATCHPLAY, with --c-collect: the RNG streams of a "
                              "worker's episodes. 'vectorized' (default, the "
                              "production leg): one stream per episode "
                              "derived from the worker seed, as "
                              "--vectorized-collect plays them, so a C round "
                              "plays the vectorized recipe's matches for the "
                              "same --seed and --workers. 'serial': all of a "
                              "worker's episodes on one stream, as the "
                              "standard collector (the parity oracle).")
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

    # A batch < 1 fills no lockstep slot, so the worker would collect
    # ZERO episodes and the run would report success having trained on
    # nothing. Reject at the CLI rather than let that be discovered from
    # a flat loss curve.
    if args.vectorized_batch < 1:
        parser.error(
            f"--vectorized-batch must be >= 1 (got {args.vectorized_batch}); "
            "a batch < 1 collects zero episodes per worker"
        )
    explicit_c = bool(getattr(args, "c_collect", False))
    if explicit_c and args.vectorized_collect:
        parser.error("--c-collect and --vectorized-collect are mutually "
                     "exclusive (two collectors for the same rounds): pick one")
    if args.game_mode == "matchplay":
        if not explicit_c:
            for flag in ("c_collect_backend", "c_collect_rng"):
                if getattr(args, flag) is not None:
                    parser.error(f"--{flag.replace('_', '-')} requires --c-collect")
    else:
        if args.c_collect_rng is not None:
            parser.error("--c-collect-rng is match play only (the money C "
                         "collector plays each worker's episodes on one "
                         "stream)")
        if args.vectorized_collect:
            parser.error("--vectorized-collect is match play only (the "
                         "money GPU collector is not on this branch)")

    apply_cubeful_money_upgrades(args)

    # Matchplay auto-upgrades: encoder must carry the match features
    # and the output must be probability (P(win match)).
    if args.game_mode == "matchplay":
        if not args.resume:
            if args.encoder == "perspective196":
                args.encoder = "matchplay_perspective196"
            elif not args.encoder.startswith(("matchplay_", "matchplay1h_", "matchplay1h64_",
            "matchplay1h64met_", "matchplaymet_")):
                raise ValueError(
                    f"matchplay requires a matchplay_* encoder; "
                    f"got --encoder {args.encoder!r}"
                )
            if args.output_mode != "probability":
                raise ValueError(
                    f"matchplay requires --output-mode probability; "
                    f"got {args.output_mode!r}"
                )
        if args.oneply_acting:
            raise ValueError(
                "matchplay supports 0-ply acting with 0-ply or 1-ply "
                "(--oneply) targets; drop --oneply-acting"
            )

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)

    net = build_network(args)
    from met import MatchEquityTable, resolve_met_path
    met_path = resolve_met_path(getattr(args, "met", None))
    met_table = None
    if met_path:
        if args.game_mode != "matchplay":
            raise ValueError("--met only applies to --game-mode matchplay")
        met_table = MatchEquityTable.from_xml(met_path)
        # Keep encoder-side default_met() (MET-feature encoders, all
        # spawned workers via env inheritance) on the SAME table as the
        # grounding MET -- review finding: independent knobs diverge.
        os.environ["BG_MET_PATH"] = met_path
        print(f"MET ground truth: {met_table.name} ({met_path})")
    if args.met_scope in (None, "", "all"):
        met_scopes = frozenset(("cube", "terminal", "boundary"))
    else:
        met_scopes = frozenset(
            s.strip() for s in args.met_scope.split(",") if s.strip()
        )
        bad = met_scopes - {"cube", "terminal", "boundary"}
        if bad:
            parser.error(f"--met-scope: unknown {sorted(bad)}; "
                         f"choose from cube,terminal,boundary or 'all'")
        if not met_path:
            parser.error("--met-scope requires --met")
    if met_path:
        print(f"MET scopes grounded: {sorted(met_scopes)}")
    agent = TDAgent(net, device=args.device, oneply=args.oneply_acting,
                    boltzmann_temp=args.boltzmann_temp, met=met_table,
                    met_scopes=met_scopes)
    if args.oracle_targets:
        if args.game_mode != "matchplay":
            raise ValueError("--oracle-targets is matchplay-only")
        if args.oneply and not args.oracle_act_only:
            raise ValueError(
                "--oracle-targets is 0-ply supervision; drop --oneply "
                "(or pass --oracle-act-only to use the teacher for moves "
                "only, with TD targets)")
        if args.outcome_mix > 0.0:
            raise ValueError(
                "--oracle-targets replaces every target; "
                "--outcome-mix would dilute the oracle")
        if args.met and not args.oracle_act_only:
            raise ValueError(
                "--oracle-targets carries its own exact MET; drop --met "
                "(--oracle-act-only needs it: its targets come from the TD "
                "path, which grounds on the MET)")
        if args.vectorized_collect:
            raise ValueError(
                "--oracle-targets is not implemented in the vectorized "
                "collector; drop --vectorized-collect")
        if args.oracle_act_only and not args.oracle_act_prob:
            raise ValueError("--oracle-act-only requires --oracle-act-prob > 0")
        if args.oracle_act_only and not args.oneply:
            raise ValueError("--oracle-act-only needs --oneply for targets")
        if args.oracle_act_prob and args.oracle_move_eps is None:
            raise ValueError(
                "--oracle-act-prob needs --oracle-move-eps (the teacher's "
                "move is the eps-set anchor)")
        if not (0.0 <= args.oracle_act_prob <= 1.0):
            raise ValueError(
                f"--oracle-act-prob must be in [0,1], got "
                f"{args.oracle_act_prob}")
    elif args.oracle_move_eps is not None:
        raise ValueError("--oracle-move-eps requires --oracle-targets")
    elif args.oracle_plies:
        raise ValueError("--oracle-plies requires --oracle-targets")
    elif args.oracle_act_prob:
        raise ValueError("--oracle-act-prob requires --oracle-targets")
    elif args.oracle_eval_server:
        raise ValueError("--oracle-eval-server requires --oracle-targets")
    mode = build_mode(
        args.game_mode, jacoby=args.jacoby,
        match_length=args.match_length,
        outcome_mix=args.outcome_mix,
        oracle_spec=args.oracle_targets,
        oracle_move_eps=args.oracle_move_eps,
        oracle_plies=args.oracle_plies,
        oracle_act_prob=args.oracle_act_prob,
        move_explore_eps=args.move_explore_eps,
        td_move_eps=args.td_move_eps,
        oracle_1ply_eps=args.oracle_1ply_eps,
        oneply_act_k=args.td_oneply_act,
        td_eps_only=args.td_eps_only,
        oracle_act_only=args.oracle_act_only,
    )

    # The collector, once the network is built: whether the default C
    # collector applies depends on its encoder and output (a run it rules
    # out gets the Python collector).
    for note in resolve_collector(
            args, c_collect_default_fallbacks(args, net, mode)):
        print(note, flush=True)

    if args.td_eps_only and not args.td_move_eps:
        raise ValueError(
            "--td-eps-only keeps only the eps rows; it needs --td-move-eps "
            "to produce any")
    if args.td_oneply_act and args.game_mode != "matchplay":
        raise ValueError("--td-oneply-act is matchplay-only")
    if args.td_oneply_act and not args.oneply:
        raise ValueError(
            "--td-oneply-act needs --oneply (it re-prices successors with "
            "the 1-ply value the target path already computes)")
    if args.td_oneply_act < 0:
        raise ValueError(
            f"--td-oneply-act must be >= 0, got {args.td_oneply_act}")
    if args.oracle_1ply_eps and not args.oracle_plies:
        raise ValueError(
            "--oracle-1ply-eps filters which rows get DEEP oracle pricing; "
            "it does nothing without --oracle-plies > 0")
    if args.oracle_1ply_eps and not args.oracle_targets:
        raise ValueError("--oracle-1ply-eps requires --oracle-targets")
    if not (0.0 <= args.oracle_1ply_eps <= 1.0):
        raise ValueError(
            f"--oracle-1ply-eps must be in [0,1], got {args.oracle_1ply_eps}")
    if args.td_move_eps and args.game_mode != "matchplay":
        raise ValueError("--td-move-eps is matchplay-only")
    if args.td_move_eps and args.oracle_targets:
        raise ValueError(
            "--td-move-eps and --oracle-targets both emit successor rows; "
            "run them in separate legs so the attribution stays clean")
    if args.td_move_eps and not args.oneply:
        raise ValueError(
            "--td-move-eps targets successors at their 1-ply value; "
            "it needs --oneply")
    if not (0.0 <= args.td_move_eps <= 1.0):
        raise ValueError(
            f"--td-move-eps must be in [0,1], got {args.td_move_eps}")
    if args.move_explore_eps and args.game_mode != "matchplay":
        raise ValueError("--move-explore-eps is matchplay-only")
    if not (0.0 <= args.move_explore_eps <= 1.0):
        raise ValueError(
            f"--move-explore-eps must be in [0,1], got {args.move_explore_eps}")
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

    if args.resume_episodes:
        n_done, n_total = args.resume_episodes, args.num_episodes
        if not args.resume:
            raise ValueError("--resume-episodes requires --resume")
        if not (0 < n_done < n_total):
            raise ValueError(
                f"--resume-episodes must be in (0, --num-episodes={n_total}),"
                f" got {n_done}")
        lr0 = args.lr
        if args.end_lr is not None:
            args.lr = lr0 + (args.end_lr - lr0) * n_done / n_total
        args.num_episodes = n_total - n_done
        print(
            f"Resume-episodes: {n_done}/{n_total} already done -> "
            f"lr {lr0:g} -> {args.lr:.6g} at ep {n_done}"
            + (f", annealing to {args.end_lr:g}" if args.end_lr is not None
               else " (constant)")
            + f"; {args.num_episodes} episodes remain"
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

    if args.c_collect and args.game_mode == "matchplay":
        # Every unmet precondition at once, before anything starts
        # (Trainer.train raises the same error for API callers).
        _c_reasons = c_collect_fallbacks(
            net, mode, oneply=args.oneply, oneply_acting=args.oneply_acting,
            boltzmann_temp=args.boltzmann_temp, met_path=met_path,
            met_scopes=met_scopes,
            vectorized_collect=args.vectorized_collect,
            gpu_eval_server=args.gpu_eval_server,
            oracle_eval_server=args.oracle_eval_server)
        if _c_reasons:
            parser.error(c_collect_error(_c_reasons))

    # Stamped into every checkpoint. Trainer.train sets c_collect to
    # whether the C collector collected (a failed precondition or a
    # missing library falls back to Python). Match play adds its own
    # fields (provenance only: no loader reads them back).
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
    if args.game_mode == "matchplay":
        del train_params["jacoby"]                  # a money-game rule
        train_params.update({
            "match_length": args.match_length,
            "met": args.met,
            "met_scope": args.met_scope,
            "boltzmann_temp": args.boltzmann_temp,
            "episodes_per_round": args.episodes_per_round,
            "c_collect_backend": args.c_collect_backend,
            "c_collect_rng": args.c_collect_rng,
            "resume": args.resume,
        })

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
        pipeline_collect=args.pipeline_collect,
        met_path=met_path,
        met_scopes=met_scopes,
        vectorized_collect=args.vectorized_collect,
        vectorized_batch=args.vectorized_batch,
        vectorized_device=args.vectorized_device,
        max_episode_plies=args.max_episode_plies,
        gpu_eval_server=args.gpu_eval_server,
        oracle_eval_server=args.oracle_eval_server,
        gpu_eval_device=args.gpu_eval_device,
        gpu_eval_route_moves=args.gpu_eval_route_moves,
        gpu_eval_deterministic=args.gpu_eval_deterministic,
        episode_offset=args.resume_episodes,
        train_params=train_params,
        c_collect=args.c_collect,
        c_collect_backend=args.c_collect_backend or "simd",
        c_collect_auto=args.c_collect_auto,
        c_collect_rng=args.c_collect_rng or "vectorized",
    )
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
        if agent.is_matchplay:
            print(
                "NOTE: --eval-vs-random is game-based and not meaningful "
                "for matchplay; use eval_matchplay.py instead."
            )
        else:
            wr = eval_vs_random(agent, n_games=args.eval_vs_random)
            print(f"vs Random ({args.eval_vs_random} games): {wr:.1%}")


if __name__ == "__main__":
    main()
