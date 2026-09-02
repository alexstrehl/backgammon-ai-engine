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
    if getattr(args, "game_mode", None) == "matchplay":
        # The active matchplay line chooses its collector explicitly;
        # auto-routing here would silently switch it (the matchplay
        # vectorized driver also has extra requirements, e.g.
        # --boltzmann-temp 0). Money-mode UX only.
        notes.append(
            "NOTE: matchplay does not auto-route to the vectorized GPU "
            "collector; pass --vectorized-collect explicitly if wanted.")
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
    g_batch.add_argument("--cube-targets-1ply", action="store_true",
                         help="EXPERIMENTAL: in --oneply training mode, use "
                              "value_oneply_checker_cubeful() for cube-decision "
                              "targets (pure 1-ply, conceptually a deeper "
                              "search). Default uses the 0-ply formula "
                              "(evaluate_cubeful at is_cube_action=False) for "
                              "cube targets, which empirically gives much "
                              "better cube_mEMG. No effect without --oneply.")
    g_batch.add_argument("--oracle-targets", default=None, metavar="SPEC",
                         help="Supervised oracle training (matchplay "
                              "only): prob5 checkpoint path, optional "
                              "':x=V'. Acting stays with the trained "
                              "net; every collected target is replaced "
                              "by the p5jan oracle's value of the "
                              "encoded state (janowski_match). "
                              "Incompatible with --oneply and --met.")
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
                              "every legal successor that EITHER the "
                              "teacher rates within EPS (MWC) of its own "
                              "best move, OR the student rates within EPS "
                              "of the teacher's best. The union matters: "
                              "the second set is where the student's "
                              "inflated favourites get corrected, and "
                              "greedy play never visits those states.")
    g_batch.add_argument("--score-curriculum", type=float, default=0.0,
                         metavar="P",
                         help="Matchplay: fraction of episodes that "
                              "start at a seeded away-score pair "
                              "(uniform over reachable pairs, Crawford "
                              "handled; see trainer._curriculum_start) "
                              "instead of the natural full-match "
                              "score. Default 0 = off. The published "
                              "champion was trained with 0.5.")
    g_batch.add_argument("--met", default=None, metavar="TABLE",
                         help="Matchplay: ground all cross-game equity in a "
                              "real match equity table instead of the "
                              "network's own new-game evaluation. Accepts "
                              "'auto' (gnubg's Kazaross-XG2 default), a "
                              "table name from /usr/share/gnubg/met, or a "
                              "path to a MET XML. This pins v_dp, the take "
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
                              "i.e. well under 1500 plies). On the matchplay "
                              "vectorized path this is a per-GAME turn bound "
                              "(a runaway game in lockstep pins a driver slot "
                              "and stalls the round). 0 disables.")
    g_batch.add_argument("--vectorized-collect", action="store_true",
                         help="Collect on the GPU with the lockstep generator "
                              "driver: each worker process runs "
                              "--vectorized-batch games (matchplay: matches, "
                              "via vectorized_matchplay.py) concurrently and "
                              "batches every pending network forward across "
                              "them into ONE local CUDA call (no IPC, no "
                              "inference server). Requires --oneply, "
                              "--workers > 1 and CUDA (matchplay additionally "
                              "--boltzmann-temp 0); warns and falls back to "
                              "the CPU collector otherwise. DISABLES "
                              "--pipeline-collect, --c-engine and "
                              "--bf16-collect (each logged explicitly). Use a "
                              "large --episodes-per-round (>= workers * "
                              "batch * 8) or the lockstep tail-drain eats the "
                              "throughput gain. --max-episode-plies is "
                              "enforced per game, as on the CPU paths.")
    g_batch.add_argument("--no-vectorized-collect", action="store_true",
                         help="Keep the CPU collector even when --oneply "
                              "--device cuda would default to the vectorized "
                              "GPU collector (money modes only; matchplay "
                              "never auto-routes). The master training step "
                              "still runs on cuda. Mutually exclusive with an "
                              "explicit --vectorized-collect.")
    g_batch.add_argument("--vectorized-batch", type=int, default=32,
                         metavar="B",
                         help="Games in flight per worker process for "
                              "--vectorized-collect (default 32, must be "
                              ">= 1). Larger B means bigger GPU batches but "
                              "more GPU memory per process and a longer "
                              "lockstep tail (~730 MB/driver at B=32 on the "
                              "matchplay path).")
    g_batch.add_argument("--oracle-act-only", action="store_true",
                         help="Use the teacher for MOVE SELECTION only: no "
                              "oracle value targets are emitted, so training "
                              "targets come from the normal (--oneply) TD "
                              "path. Requires --oracle-targets, "
                              "--oracle-move-eps and --oracle-act-prob > 0. "
                              "Transfers the teacher's ranking through the "
                              "visited trajectory while keeping the "
                              "student's own value targets (no teacher "
                              "calibration error imported).")
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
                              "untouched. Closes "
                              "the coverage hole: ~540 successor reads feed "
                              "each 1-ply target but only the played one "
                              "becomes a row, leaving the skipped siblings "
                              "undervalued by ~12 mMWC. Try 0.01.")
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
                              "contexts, at the cost of one queue "
                              "round-trip per forward. Routes the "
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

    # A batch < 1 fills no lockstep slot, so the worker would collect
    # ZERO episodes and the run would report success having trained on
    # nothing. Reject at the CLI rather than let that be discovered from
    # a flat loss curve.
    if args.vectorized_batch < 1:
        parser.error(
            f"--vectorized-batch must be >= 1 (got {args.vectorized_batch}); "
            "a batch < 1 collects zero episodes per worker"
        )

    # --oneply-acting implies --oneply for targets. Normalize BEFORE the
    # collector routing so the resolver sees the real ply config (it must
    # not depend on flag-spelling: --oneply-acting alone and
    # --oneply --oneply-acting have to route identically).
    if args.oneply_acting:
        args.oneply = True

    for note in resolve_cuda_oneply_collector(args):
        print(note, flush=True)

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
    # If the loaded/built network is a cubeful_take_* model, the
    # take-decision-state collector must be used (the new encoder bit
    # is only valid in the new state graph). Conversely, --take-decision-state
    # on a non-take encoder is a user error.
    take_decision_state = bool(getattr(args, "take_decision_state", False))
    if agent.is_cubeful_take and not take_decision_state:
        print(
            "  NOTE: detected cubeful_take_* encoder; enabling "
            "--take-decision-state mode automatically."
        )
        take_decision_state = True
    elif take_decision_state and not agent.is_cubeful_take:
        raise ValueError(
            "--take-decision-state requires a cubeful_take_* encoder; "
            f"got encoder={net.encoder_name!r}"
        )
    if args.oracle_targets:
        if args.game_mode != "matchplay":
            raise ValueError("--oracle-targets is matchplay-only")
        if args.oneply and not args.oracle_act_only:
            raise ValueError(
                "--oracle-targets is 0-ply supervision; drop --oneply "
                "(or pass --oracle-act-only to use the teacher for moves "
                "only, with TD targets)")
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
        take_decision_state=take_decision_state,
        match_length=args.match_length,
        score_curriculum=args.score_curriculum,
        oracle_spec=args.oracle_targets,
        oracle_move_eps=args.oracle_move_eps,
        oracle_plies=args.oracle_plies,
        oracle_act_prob=args.oracle_act_prob,
        td_move_eps=args.td_move_eps,
        oracle_1ply_eps=args.oracle_1ply_eps,
        oracle_act_only=args.oracle_act_only,
    )

    # Equity output is unbounded; high lr can cause gradient explosion -> NaN.
    # SGD is more sensitive than Adam (which normalizes by gradient moments).
    if args.cube_targets_1ply and args.game_mode == "matchplay" \
            and not args.oneply:
        raise ValueError(
            "--cube-targets-1ply prices cube targets with the 1-ply value; "
            "it needs --oneply, otherwise the checker rows it must stay "
            "consistent with are 0-ply themselves.")
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
        cube_targets_1ply=args.cube_targets_1ply,
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
