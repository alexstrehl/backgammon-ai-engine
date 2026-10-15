#!/usr/bin/env python3
"""Head-to-head runner.

Plays two sides head-to-head and reports win rate and (for money modes)
per-game equity, each with bootstrap 95% confidence intervals.

This file contains NO game/statistics logic. It lives in play_models
(run_head_to_head = play_matches_parallel + report_head_to_head) and in the
engine agents (gnubg_external.GnubgExternalAgent, OpenSageAgent).

Each side (--side1 / --side2) is a checkpoint path OR an engine keyword:
  gnubg-ext | gnubg-ext-full | opensage | opensage-native | opensage-analyzer | random |
  hedgehog:<model.ogxf> | janowski:<prob5.pt> | bo:<cubeful.pt>
--plies sets gnubg's search depth; --sage-plies sets Open Sage's ply
(1 = 0-ply, 2 = 1-ply); --oneply1/2 puts a hedgehog side at
depth 1 (HEDGEHOG.md). janowski: and bo: are the two sides of the
cube-layer comparison, cubeful money only (experiments/janowski/README.md).

Examples (cubeless money, 1M games each):
  # PureTD 0-ply vs Open Sage 0-ply
  python experiments/head_to_head/run.py \
      --side1 best_models/cubeless_prob5_512_512_256_256.pt \
      --side2 opensage --sage-plies 1 \
      --game-mode cubeless-money --games 1000000 --workers 32 --save DIR

  # gnubg 0-ply vs Open Sage 0-ply (real gnubg 1.08 over the external socket)
  python experiments/head_to_head/run.py --side1 gnubg-ext --side2 opensage \
      --plies 0 --sage-plies 1 --game-mode cubeless-money --games 1000000 --workers 32 --save DIR

  # PureTD 0-ply vs gnubg 1-ply (unpruned 1-ply, move-filter comparison)
  python experiments/head_to_head/run.py \
      --side1 best_models/cubeless_prob5_512_512_256_256.pt \
      --side2 gnubg-ext-full --plies 1 --game-mode cubeless-money --games 1000000 --workers 32 --save DIR

  # PureTD 0-ply vs Open Sage 1-ply
  python experiments/head_to_head/run.py \
      --side1 best_models/cubeless_prob5_512_512_256_256.pt \
      --side2 opensage --sage-plies 2 --game-mode cubeless-money --games 1000000 --workers 32 --save DIR

Open Sage sides:
  ``opensage`` is cubeless-money / DMP only. For ``--game-mode
  cubeful-money`` use ``opensage-native`` (0/1-ply, Sage's native C API) or
  ``opensage-analyzer`` (any ply, Sage's ``BgBotAnalyzer``); same decisions.
  See OPENSAGE.md.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from play_models import run_head_to_head, combine_head_to_head


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side1",
                    help="model path | gnubg-ext | gnubg-ext-full | opensage | opensage-native | opensage-analyzer | random | hedgehog:<model.ogxf> | janowski:<prob5.pt> | bo:<cubeful.pt>")
    ap.add_argument("--side2",
                    help="model path | gnubg-ext | gnubg-ext-full | opensage | opensage-native | opensage-analyzer | random | hedgehog:<model.ogxf> | janowski:<prob5.pt> | bo:<cubeful.pt>")
    ap.add_argument("--combine", default=None,
                    help="Directory of saved runs (--save dir): pool all games "
                         "and report the combined result; no new games are played.")
    ap.add_argument("--game-mode", default="cubeless-money",
                    choices=["dmp", "cubeless-money", "cubeful-money"])
    ap.add_argument("--games", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--save", default=None,
                    help="Directory to save raw per-game results (one .npz per "
                         "run) so runs can be pooled/extended later.")
    ap.add_argument("--plies", type=int, default=0,
                    help="gnubg search depth (for a gnubg-ext side)")
    ap.add_argument("--sage-plies", type=int, default=1,
                    help="Open Sage ply for an opensage side: 1=0-ply, 2=1-ply")
    ap.add_argument("--jacoby", action=argparse.BooleanOptionalAction, default=True,
                    help="Jacoby rule (cubeful-money only).")
    ap.add_argument("--oneply1", action="store_true",
                    help="1-ply lookahead for side1 (model checkpoints; "
                         "depth 1 for a hedgehog side)")
    ap.add_argument("--oneply2", action="store_true",
                    help="1-ply lookahead for side2 (model checkpoints; "
                         "depth 1 for a hedgehog side)")
    ap.add_argument("--bf16", action="store_true",
                    help="bf16 inference for prob5 model sides")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--paired-dice", action="store_true",
                    help="variance reduction: play games in pairs that share "
                         "one dice stream with the seats swapped (roll k goes "
                         "to whoever is on move at turn k, opening roll "
                         "included). Each game is still fair; CIs bootstrap "
                         "over pairs; the npz records the pairing and "
                         "--combine pools such runs by pair (never together "
                         "with independent runs). --games must be even.")
    ap.add_argument("--cube-cap", type=int, default=None, metavar="N",
                    help="cubeful only: neither side is offered a double "
                         "once the cube reaches N (64 = HedgeHog's "
                         "CUBE_MAX_VALUE ceiling, applied to both sides). "
                         "Default: uncapped. Recorded in the npz; capped "
                         "and uncapped runs refuse to pool.")
    ap.add_argument("--log-runaway", type=float, default=0.0, metavar="STAKE",
                    help="cubeful only: save the full game record (moves + "
                         "cube actions, as .mat under <save>/runaway_games/) "
                         "for any game with |stake| >= STAKE. Catches "
                         "redouble/take-loop games. E.g. 256 = cube 128+.")
    ap.add_argument("--c-api", action="store_true",
                    help="require the C API play path (the default): a "
                         "configuration it cannot serve is an error instead "
                         "of a Python fallback.")
    ap.add_argument("--python-api", action="store_true",
                    help="checkpoint and janowski: sides play on the Python "
                         "path. Default: the C API (c_api/, libpuretd, built "
                         "on demand; 0/1-ply, fp32), with a NOTE and the "
                         "Python path for anything it cannot serve (2-ply, "
                         "--bf16 prob5). The path is recorded in the npz; "
                         "runs on different paths refuse to pool.")
    ap.add_argument("--c-api-backend", choices=["simd", "cblas", "torch", "ref"],
                    default=None,
                    help="forward backend of the C API path (default: "
                         "PURETD_BACKEND, else simd). Recorded in the npz "
                         "with the host's ISA; backends refuse to pool.")
    a = ap.parse_args()
    if a.c_api and a.python_api:
        ap.error("--c-api and --python-api are mutually exclusive")
    if a.c_api_backend and a.python_api:
        ap.error("--c-api-backend needs the C API (drop --python-api)")

    if a.combine:
        combine_head_to_head(a.combine, n_bootstrap=a.n_bootstrap)
        return
    if not (a.side1 and a.side2):
        ap.error("--side1 and --side2 are required (unless --combine is given)")

    # Before any game: bearoff tables (janowski:/bo:), supported depths,
    # the pinned gnubg (gnubg-ext*) and bgsage (opensage*).
    from play_models import _detect_side_type, preflight_side
    try:
        for side in (a.side1, a.side2):
            preflight_side(side, _detect_side_type(side))
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        ap.error(str(e))

    run_head_to_head(
        a.side1, a.side2,
        num_games=a.games, workers=a.workers, game_mode=a.game_mode,
        plies=a.plies, sage_plies=a.sage_plies, jacoby=a.jacoby,
        oneply1=a.oneply1, oneply2=a.oneply2,
        bf16=a.bf16, n_bootstrap=a.n_bootstrap, save=a.save,
        runaway_threshold=a.log_runaway, paired_dice=a.paired_dice,
        cube_cap=a.cube_cap,
        c_api=True if a.c_api else (False if a.python_api else None),
        c_api_backend=a.c_api_backend,
    )


if __name__ == "__main__":
    main()
