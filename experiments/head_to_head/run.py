#!/usr/bin/env python3
"""Head-to-head runner (public).

Plays two sides head-to-head and reports win rate and (for money modes)
per-game equity, each with bootstrap 95% confidence intervals.

This file contains NO game/statistics logic — everything lives in the library:
  * agent construction + parallel play : play_models.play_matches_parallel
  * win-rate / equity stats + report   : play_models.report_head_to_head
  * composed by                        : play_models.run_head_to_head
  * engine agents                      : gnubg_external.GnubgExternalAgent / OpenSageAgent

Each side (--side1 / --side2) is a checkpoint path OR an engine keyword:
  gnubg-ext | gnubg-ext-full | opensage | opensage-analyzer | random
--plies sets gnubg's search depth; --sage-plies sets Open Sage's ply
(1 = 0-ply / "1P", 2 = 1-ply / "2P").

Examples (cubeless money, 1M games each):
  # PureTD 0-ply vs Open Sage 0-ply
  python experiments/head_to_head/run.py \
      --side1 best_models/cubeless_prob5_512_512_256_256.pt \
      --side2 opensage --sage-plies 1 \
      --game-mode cubeless-money --games 1000000 --workers 32 --save ~/hh_out

  # gnubg 0-ply vs Open Sage 0-ply (real gnubg 1.08 over the external socket)
  python experiments/head_to_head/run.py --side1 gnubg-ext --side2 opensage \
      --plies 0 --sage-plies 1 --game-mode cubeless-money --games 1000000 --workers 32 --save ~/hh_out

  # PureTD 0-ply vs gnubg 1-ply (unpruned paper config)
  python experiments/head_to_head/run.py \
      --side1 best_models/cubeless_prob5_512_512_256_256.pt \
      --side2 gnubg-ext-full --plies 1 --game-mode cubeless-money --games 1000000 --workers 32 --save ~/hh_out

  # PureTD 0-ply vs Open Sage 1-ply
  python experiments/head_to_head/run.py \
      --side1 best_models/cubeless_prob5_512_512_256_256.pt \
      --side2 opensage --sage-plies 2 --game-mode cubeless-money --games 1000000 --workers 32 --save ~/hh_out

Open Sage sides:
  ``opensage`` is cubeless-money / DMP only. For ``--game-mode
  cubeful-money`` use ``opensage-analyzer``: Sage driven through its own
  ``BgBotAnalyzer`` exactly as its self-play generator does (cube-aware
  ``checker_play`` + native ``cube_action``), verified
  decision-for-decision identical to Sage self-play. See OPENSAGE.md.
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
                    help="model path | gnubg-ext | gnubg-ext-full | opensage | opensage-analyzer | random")
    ap.add_argument("--side2",
                    help="model path | gnubg-ext | gnubg-ext-full | opensage | opensage-analyzer | random")
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
                    help="1-ply lookahead for side1 (model checkpoints only)")
    ap.add_argument("--oneply2", action="store_true",
                    help="1-ply lookahead for side2 (model checkpoints only)")
    ap.add_argument("--bf16", action="store_true",
                    help="bf16 inference for prob5 model sides (~1.5x faster)")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--log-runaway", type=float, default=0.0, metavar="STAKE",
                    help="cubeful only: save the full game record (moves + "
                         "cube actions, as .mat under <save>/runaway_games/) "
                         "for any game with |stake| >= STAKE. Catches "
                         "redouble/take-loop games. E.g. 256 = cube 128+.")
    a = ap.parse_args()

    if a.combine:
        combine_head_to_head(a.combine, n_bootstrap=a.n_bootstrap)
        return
    if not (a.side1 and a.side2):
        ap.error("--side1 and --side2 are required (unless --combine is given)")

    # Preflight the two-sided bearoff DB once (~0.2s), before any game;
    # _ExtServer's own assert only fires per worker, after pool spawn.
    if any(str(s).startswith("gnubg-ext") for s in (a.side1, a.side2)):
        from gnubg_eval import assert_gnubg_bearoff
        assert_gnubg_bearoff()

    run_head_to_head(
        a.side1, a.side2,
        num_games=a.games, workers=a.workers, game_mode=a.game_mode,
        plies=a.plies, sage_plies=a.sage_plies, jacoby=a.jacoby,
        oneply1=a.oneply1, oneply2=a.oneply2,
        bf16=a.bf16, n_bootstrap=a.n_bootstrap, save=a.save,
        runaway_threshold=a.log_runaway,
    )


if __name__ == "__main__":
    main()
