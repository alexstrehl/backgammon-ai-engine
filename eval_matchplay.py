"""
eval_matchplay.py -- Head-to-head N-point match evaluation.

Plays full matches (games chained via MatchState.start_game/after_game)
between a matchplay TDAgent and an opponent, alternating colors for
fairness, and reports the match win rate with a binomial CI.

Opponents (--model2):
  random        RandomAgent (never doubles, always takes)
  gnubg-ext[-full][:N]  real gnubg 1.08 over its external socket
                interface -- score-aware chequer play, native match
                cube handling, Kazaross-XG2 MET. Preferred referee.
  gnubg[:N]     gnubg-nn wrapper: score-BLIND chequer play and a
                Snowie MET. Kept for continuity with older numbers.
  dmp:PATH      cubeless TDAgent on a 196-input net (never doubles,
                always takes, 0-ply checker play) — isolates the value
                of cube + score awareness
  dmpcube:PATH  same DMP checker play, but always doubles when allowed
                and always takes — near-optimal doctrine at 2pt
                matches, so a strong dummy baseline
  model:PATH    another matchplay TDAgent (full cube + score play)

Also prints the model's implicit match-equity table (MWC at each
reachable away-score combination) — compare against published MET
values to sanity-check score awareness.
"""

import argparse
import math
import os
import multiprocessing as mp
import random

from backgammon_engine import opening_roll, switch_turn
from modes import MatchState


def play_match(agent_white, agent_black, match_length, rng):
    """Play one full match. Returns 0 if WHITE wins the match, 1 if BLACK."""
    ms = MatchState(
        jacoby=False, match_length=match_length,
        my_away=match_length, opp_away=match_length,
    )
    while True:
        gms = ms.start_game()
        state, dice = opening_roll(rng)
        is_opening = True
        while True:
            player = state.turn
            actor = agent_white if player == 0 else agent_black

            # Cube phase
            if not is_opening and gms.can_offer(player):
                offer = actor.offer_double(state, gms)
                if offer.should_double:
                    responder = agent_black if player == 0 else agent_white
                    same_agent = responder is actor
                    takes = responder.respond_to_double(
                        state, gms, hint=offer if same_agent else None,
                    )
                    if takes:
                        gms = gms.after_take(player)
                    else:
                        ms = gms.after_game(1, winner_is_me=(player == 0))
                        break  # game over (drop)

            # Checker phase
            result = actor.choose_checker_action_cubeful(state, dice, gms)
            if result is None:
                state = switch_turn(state)
            else:
                next_state = result[0]
                if next_state.is_game_over():
                    gr = next_state.game_result()
                    ms = gms.after_game(gr, winner_is_me=(player == 0))
                    break  # game over
                state = next_state
            dice = (rng.randint(1, 6), rng.randint(1, 6))
            is_opening = False

        if ms.match_over():
            # ms away scores are WHITE's view.
            return 0 if ms.my_away <= 0 else 1


def _build_p5jan(spec):
    """`p5jan:PATH[:x=VAL]` -- prob5 net + gnubg-style match-play
    Janowski cube layer (janowski_match.py). Always MET-exact (that is
    the point of the construction); `x=` overrides the gnubg
    position-class cube-life policy with a constant."""
    from janowski_match import JanowskiMatchAgent
    from model import ProbNetwork

    rest = spec[len("p5jan:"):]
    x_policy = None
    if ":x=" in rest:
        rest, _, xs = rest.rpartition(":x=")
        x_policy = float(xs)
    return JanowskiMatchAgent(ProbNetwork.load(rest), x_policy=x_policy)


def _build_agents(model1_path, model2_spec, met_path=None, met2_path=None,
                  referee_met_path=None):
    from model import TDNetwork
    from td_agent import TDAgent
    from agents import RandomAgent

    def _load_met(path):
        if not path:
            return None
        from met import MatchEquityTable
        return MatchEquityTable.from_xml(path)

    # Each side plays as it was TRAINED. In a cross-arm head-to-head one
    # model may be MET-grounded and the other not; forcing a single
    # setting on both would measure two agents neither of which was
    # trained that way.
    met = _load_met(met_path)
    met2 = _load_met(met2_path if met2_path is not None else met_path)

    if model1_path.startswith("p5jan:"):
        a1 = _build_p5jan(model1_path)
    elif model1_path.startswith("gnubg-ext"):
        # gnubg vs gnubg (e.g. --model1 gnubg-ext:1 --model2 gnubg-ext):
        # sizes the ply gap in match wins with the same referee stack.
        from gnubg_external import GnubgExternalAgent
        head, _, ply_s = model1_path.partition(":")
        a1 = GnubgExternalAgent(
            plies=int(ply_s) if ply_s else 0,
            full=head.endswith("-full"),
            met_path=referee_met_path,
        )
    else:
        net1 = TDNetwork.load(model1_path)
        a1 = TDAgent(net1, met=met)
        assert a1.is_matchplay, \
            f"--model1 must be a matchplay net; got encoder={net1.encoder_name}"

    if model2_spec == "random":
        a2 = RandomAgent()
    elif model2_spec.startswith("gnubg-ext"):
        # "gnubg-ext[-full][:PLIES]" -- real gnubg 1.08 over its external
        # socket interface. Preferred over the gnubg-nn wrapper for match
        # play: score-aware chequer play, native match cube handling, and
        # the Kazaross-XG2 MET rather than gnubg-nn's Snowie.
        #
        # The referee is pinned to ONE table regardless of how the model
        # under test was trained, so every arm of an ablation faces an
        # identical opponent.
        from gnubg_external import GnubgExternalAgent
        head, _, ply_s = model2_spec.partition(":")
        a2 = GnubgExternalAgent(
            plies=int(ply_s) if ply_s else 0,
            full=head.endswith("-full"),
            met_path=referee_met_path,
        )
    elif model2_spec.startswith("gnubg"):
        # "gnubg" or "gnubg:N" (checker plies; cube plies = max(N, 1)).
        from agents import GnubgNNMatchplayAgent
        plies = int(model2_spec.split(":", 1)[1]) if ":" in model2_spec else 0
        a2 = GnubgNNMatchplayAgent(plies=plies)
    elif model2_spec.startswith("dmp:"):
        net2 = TDNetwork.load(model2_spec[len("dmp:"):])
        a2 = TDAgent(net2)
        assert not a2.is_matchplay and not a2.is_cubeful
    elif model2_spec.startswith("dmpcube:"):
        from agents import AlwaysCubeAgent
        net2 = TDNetwork.load(model2_spec[len("dmpcube:"):])
        base = TDAgent(net2)
        assert not base.is_matchplay and not base.is_cubeful
        a2 = AlwaysCubeAgent(base)
    elif model2_spec.startswith("p5jan:"):
        a2 = _build_p5jan(model2_spec)
    elif model2_spec.startswith("model:"):
        m2 = model2_spec[len("model:"):]
        net2 = TDNetwork.load(m2)
        a2 = TDAgent(net2, met=met2)
        assert a2.is_matchplay
    else:
        raise ValueError(f"Unknown --model2 spec: {model2_spec!r}")
    return a1, a2


_WORKER = None


def _init_worker(model1_path, model2_spec, match_length, met_path=None,
                 met2_path=None, referee_met_path=None):
    global _WORKER
    import torch
    torch.set_num_threads(1)
    a1, a2 = _build_agents(model1_path, model2_spec, met_path, met2_path,
                           referee_met_path)
    _WORKER = (a1, a2, match_length)


def _run_chunk(args):
    """Returns per-match outcomes, not just the total: pooled CIs have
    to stay recomputable from raws after the fact."""
    import numpy as np
    n_matches, seed = args
    a1, a2, match_length = _WORKER
    rng = random.Random(seed)
    won = np.zeros(n_matches, dtype=np.int8)
    white = np.zeros(n_matches, dtype=np.int8)
    for i in range(n_matches):
        # Alternate colors: even i -> model1 is WHITE.
        if i % 2 == 0:
            won[i] = play_match(a1, a2, match_length, rng) == 0
            white[i] = 1
        else:
            won[i] = play_match(a2, a1, match_length, rng) == 1
    return won, white, seed


def mwc_report(agent, match_length):
    """Print the model's implicit match-equity table: P(win match) at
    the start of a game for every reachable away-score combination."""
    print(f"\nImplicit MET (new-game MWC, {match_length}pt match):")
    print("  (my_away, opp_away, crawford_done) -> P(I win match)")
    for my in range(match_length, 0, -1):
        for opp in range(match_length, 0, -1):
            done = (my == 1 or opp == 1)
            ms = MatchState(
                jacoby=False, match_length=match_length,
                my_away=my, opp_away=opp,
                crawford_done=(done and not (my == match_length
                                             or opp == match_length)),
            )
            # Always the NETWORK's own value, never the pinned table —
            # the point of this diagnostic is what the net believes. With
            # a MET pinned, `err` is how far it has drifted from truth.
            v = agent._eval_new_game_net(ms, 0)
            tag = ""
            if not ms.crawford_done and (my == 1 or opp == 1):
                tag = "  (Crawford game)"
            if agent.met is not None:
                truth = agent.met.mwc(
                    ms.my_away, ms.opp_away, ms.crawford_done, ms.is_crawford,
                )
                tag = f"  [MET {truth:.3f}, err {v - truth:+.3f}]{tag}"
            print(f"  ({my}-away, {opp}-away) -> {v:.3f}{tag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model1", required=True)
    p.add_argument("--model2", default="random",
                   help="random | dmp:PATH | dmpcube:PATH | model:PATH | "
                        "p5jan:PATH[:x=V]")
    p.add_argument("--matches", type=int, default=500)
    p.add_argument("--match-length", type=int, default=2)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--met", default=None, metavar="TABLE",
                   help="Play with a real match equity table ('auto', a "
                        "gnubg MET name, or a path). MUST match how the "
                        "model was trained: a --met-trained model "
                        "evaluated without it reverts v_dp, the take "
                        "point and terminal gammon pricing to the "
                        "network's own drifting estimates.")
    p.add_argument("--gnubg-met", default="auto", metavar="TABLE",
                   help="MET the gnubg-ext REFEREE plays with. Pinned to "
                        "one table for every arm of an ablation so the "
                        "opponent is identical regardless of how the model "
                        "under test was trained. 'none' leaves gnubg on its "
                        "own default (already Kazaross-XG2).")
    p.add_argument("--met2", default=None, metavar="TABLE",
                   help="MET for --model2 when it is a matchplay model "
                        "trained differently from --model1. Use the "
                        "literal 'none' for an ungrounded opponent. "
                        "Defaults to --met.")
    p.add_argument("--raw-out", default=None, metavar="PATH",
                   help="Save per-match outcomes + provenance to an .npz "
                        "so the CI can be recomputed and runs pooled.")
    p.add_argument("--met-only", action="store_true",
                   help="Only print the implicit MET, skip head-to-head")
    args = p.parse_args()

    from met import resolve_met_path
    met_path = resolve_met_path(args.met)
    if args.met2 is None:
        met2_path = met_path
    elif args.met2 == "none":
        met2_path = None
    else:
        met2_path = resolve_met_path(args.met2)
    referee_met_path = (
        None if args.gnubg_met == "none" else resolve_met_path(args.gnubg_met)
    )
    a1, _ = _build_agents(args.model1, args.model2, met_path, met2_path,
                          referee_met_path)
    if hasattr(a1, "_eval_new_game_net"):
        mwc_report(a1, args.match_length)
    else:
        print("\n(model1 has no learned implicit MET -- p5jan is "
              "MET-exact by construction; skipping the report)")
    if args.met_only:
        return

    if args.workers <= 1:
        _init_worker(args.model1, args.model2, args.match_length,
                     met_path, met2_path, referee_met_path)
        chunk_results = [_run_chunk((args.matches, args.seed))]
    else:
        per = [args.matches // args.workers] * args.workers
        for i in range(args.matches % args.workers):
            per[i] += 1
        chunks = [(n, args.seed + 1000 * i) for i, n in enumerate(per) if n]
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            len(chunks), initializer=_init_worker,
            initargs=(args.model1, args.model2, args.match_length,
                      met_path, met2_path, referee_met_path),
        ) as pool:
            chunk_results = pool.map(_run_chunk, chunks)

    import numpy as np
    won = np.concatenate([c[0] for c in chunk_results])
    white = np.concatenate([c[1] for c in chunk_results])
    wins = int(won.sum())

    if args.raw_out:
        import subprocess
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True,
                text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
            ).stdout.strip()
        except Exception:
            commit = "unknown"
        np.savez(
            args.raw_out, won=won, model1_is_white=white,
            model1=args.model1, model2=args.model2,
            match_length=args.match_length, seed=args.seed,
            met=met_path or "", met2=met2_path or "",
            referee_met=referee_met_path or "", commit=commit,
            seeds=np.array([c[2] for c in chunk_results]),
        )
        print(f"Raw per-match results -> {args.raw_out}")

    n = args.matches
    wr = wins / n
    se = math.sqrt(max(wr * (1 - wr), 1e-9) / n)
    lo, hi = wr - 1.96 * se, wr + 1.96 * se
    print(f"\n{args.match_length}pt matches vs {args.model2}: "
          f"{wins}/{n} = {wr:.1%}  (95% CI [{lo:.1%}, {hi:.1%}])")


if __name__ == "__main__":
    main()
