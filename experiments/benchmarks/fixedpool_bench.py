#!/usr/bin/env python3
"""Fixed-pool per-decision benchmark — native engines (PureTD/prob5 + Open Sage).

ONE entry point for every NATIVE fixed-pool row (gnubg is separate,
fixedpool_bench_gnubg.py, run inside the gnubg binary). Replays a FROZEN pool
of decision points through each engine's production methods, so per-decision
compute is isolated from game length / dice. Block-bootstrap 95% CIs
(bench_common.rate). Dispatch with --mode:

  cubeless : prob5 + Open Sage checker throughput on the committed 8000-pos
             pool (fixedpool_positions.json). Configs via --configs:
             prob5-0ply, prob5-1ply, sage-0ply (XG 1P), sage-1ply (XG 2P).
  cubeful  : PureTD + Open Sage (0-ply + 1-ply analyzer) cubeful checker AND
             cube cost on the COMMITTED
             cubeful pool (fixedpool_cubeful_positions.json): the first 10,000
             checker points (board + dice + cube state) and first 6,000 cube
             points (board + cube state) of one seeded prod-cubeful self-play
             pass. ALL engines (incl. gnubg via fixedpool_bench_gnubg.py,
             which reads the same file) replay identical positions with
             identical cube states — ~half the cube points are recube
             situations (cube >= 2, owned). --regen-pool regenerates the
             artifact (deterministic: seed 1).

Heavy engine deps (bgbot_cpp/bgsage for cubeless, gnubg_eval for cubeful)
are imported lazily per mode, so a cubeful-PureTD run needs no bgsage.

Single pinned core (BENCH_CORE / BENCH_NO_PIN). On AMD preload
c_engine/libfakeintel.so.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "experiments"))
sys.path.insert(0, os.path.join(_REPO, "c_engine"))

# bench_common import side effects: OMP/MKL=1 before torch, pin to one
# core (BENCH_CORE / BENCH_NO_PIN=1), torch single-thread.
import bench_common  # noqa: F401

import torch
torch.set_grad_enabled(False)
torch.set_num_threads(1)

from bench_common import load_fixedpool, rate, sage_conv
from play_models import _load_agent

PROB5 = os.path.join(_REPO, "best_models", "cubeless_prob5_512_512_256_128.pt")
CUBEFUL = "best_models/cubeful_money_512_512_256_256.pt"
CUBEFUL_POOL = os.path.join(_HERE, "fixedpool_cubeful_positions.json")


# ------------------------------------------------------------------ cubeless
def run_cubeless(cap, configs, out):
    import numpy as np
    from backgammon_engine import get_legal_plays

    wanted = [c.strip() for c in configs.split(",") if c.strip()]

    chk = load_fixedpool()
    print(f"[fixedpool] loaded {len(chk)} checker decisions from the committed pool",
          flush=True)

    p0 = _load_agent(PROB5)               # ProbAgent, 0-ply
    p1 = _load_agent(PROB5, oneply=True)  # ProbAgent, 1-ply full-width

    def pchk(agent, sd):
        s, d = sd
        plays = get_legal_plays(s, d)
        if plays:
            agent.choose_checker_action(s, d, plays)

    def pchk_fused(agent, sd):
        s, d = sd
        feats, next_states = agent._get_legal_plays_encoded(s, d)
        if len(next_states):
            int(np.argmin(agent._score(agent._probs(
                torch.from_numpy(feats))).numpy()))

    CHK = {
        "prob5-0ply": (lambda sd: pchk_fused(p0, sd), chk),
        "prob5-1ply": (lambda sd: pchk(p1, sd), chk),
    }

    if any(c.startswith("sage") for c in wanted):
        # bgsage only when a Sage config is requested — prob5-only runs
        # need no bgsage install.
        import bgbot_cpp
        from bgsage.weights import WeightConfig
        from bgsage.board import possible_moves

        w = WeightConfig.default()
        s1 = bgbot_cpp.create_multipy_stage9(w.weight_paths_list, w.hidden_sizes_list,
                                             n_plies=1, parallel_evaluate=False)
        s2 = bgbot_cpp.create_multipy_stage9(w.weight_paths_list, w.hidden_sizes_list,
                                             n_plies=2, parallel_evaluate=False)
        sb = [(sage_conv(s.points, s.bar, s.turn), d[0], d[1]) for s, d in chk]

        def schk(strat, bd):
            b, d1, d2 = bd
            cand = possible_moves(b, d1, d2)
            if cand:
                strat.best_move_index(cand, b)

        CHK["sage-0ply"] = (lambda bd: schk(s1, bd), sb)
        CHK["sage-1ply"] = (lambda bd: schk(s2, bd), sb)

    header = (f"=== Fixed-pool cubeless per-decision (native, single pinned core, "
              f"committed pool of {len(chk)}, cap={cap}) ===")
    lines = [header]
    for name in wanted:
        fn, items = CHK[name]
        m, lo, hi = rate(fn, items, cap)
        lines.append(f"{name:<16} checker {m:9.1f} moves/s  95%CI [{lo:.1f},{hi:.1f}]")
    msg = "\n".join(lines)
    print("\n" + msg, flush=True)
    if out:
        with open(out, "w") as f:
            f.write(msg + "\n")


# ------------------------------------------------------------------- cubeful
N_CHECKER = 10000   # committed pool size: first 10k checker decisions...
N_CUBE = 6000       # ...and first 6k cube decisions of the seeded self-play


def _capture_cubeful_pool(n_checker=N_CHECKER, n_cube=N_CUBE, seed=1):
    """Seeded cubeful self-play with the prod model until the pool targets
    are reached, capturing every checker point (state, dice, match_state)
    and cube point (state, match_state) in order; truncate to exactly the
    first n_checker / n_cube decisions (same convention as the cubeless
    pool's 'first 8000 decisions'). States deep-copied (the C fast path
    reuses buffers). Deterministic: seed + whole games."""
    import random
    from gnubg_eval import play_and_record_cubeful
    random.seed(seed)
    agent = _load_agent(CUBEFUL)
    checker_pts, cube_pts = [], []
    orig_choose = agent.choose_checker_action_cubeful
    orig_offer = agent.offer_double

    def choose(state, dice, ms):
        checker_pts.append((state.copy(), tuple(dice), ms))
        return orig_choose(state, dice, ms)

    def offer(state, ms):
        cube_pts.append((state.copy(), ms))
        return orig_offer(state, ms)

    agent.choose_checker_action_cubeful = choose
    agent.offer_double = offer
    games = 0
    while len(checker_pts) < n_checker or len(cube_pts) < n_cube:
        play_and_record_cubeful(agent, agent, jacoby=True, record_moves=False)
        games += 1
    return checker_pts[:n_checker], cube_pts[:n_cube], games


def _gnubg_board(state, mover):
    """BoardState -> gnubg 2x25 (opp_row, mover_row); row[j] = that side's
    checkers on its own point j+1, row[24] = its bar. Round-trip-verified
    against all 8000 records of fixedpool_positions.json (0 mismatches)."""
    from gnubg_external import _mover_arrays
    mine, my_bar, theirs, opp_bar = _mover_arrays(state, mover)
    return ([theirs[p] for p in range(1, 25)] + [opp_bar],
            [mine[p] for p in range(1, 25)] + [my_bar])


def export_cubeful_pool(seed=1, path=CUBEFUL_POOL):
    """Capture the cubeful decision pool and freeze it as a committed JSON
    artifact every engine replays — native fields (points/bar/off/turn +
    cube_value/cube_owner) for PureTD/Sage, precomputed gnubg 2x25 boards
    for the embedded-python gnubg script (which can import nothing)."""
    import json
    checker_pts, cube_pts, games = _capture_cubeful_pool(seed=seed)

    def rec(state, ms, dice=None):
        b0, b1 = _gnubg_board(state, state.turn)
        r = {"points": list(state.points), "bar": list(state.bar),
             "off": list(state.off), "turn": state.turn,
             "cube_value": ms.cube_value, "cube_owner": int(ms.cube_owner),
             "board": [b0, b1]}
        if dice is not None:
            r["dice"] = list(dice)
        return r

    art = {
        "meta": {
            "recipe": ("prod 4L cubeful (cubeful_money_512_512_256_256) money "
                       "self-play, jacoby on, random.seed(%d), %d games: first "
                       "%d choose_checker_action_cubeful points (with dice) and "
                       "first %d offer_double points, recorded in order"
                       % (seed, games, len(checker_pts), len(cube_pts))),
            "n_checker": len(checker_pts), "n_cube": len(cube_pts),
            "games": games,
            "board_convention": "gnubg 2x25, board[1] = player on roll, [24]=bar",
            "cube_owner_convention": ("absolute: 0=centered, 1=WHITE, 2=BLACK; "
                                      "gnubg-relative owner = -1 if 0 else "
                                      "(0 if cube_owner == turn+1 else 1)"),
        },
        "checker": [rec(s, ms, d) for s, d, ms in checker_pts],
        "cube": [rec(s, ms) for s, ms in cube_pts],
    }
    with open(path, "w") as f:
        json.dump(art, f, separators=(",", ":"))
    print(f"[cubeful pool] wrote {len(checker_pts)} checker + {len(cube_pts)} "
          f"cube points ({games} games) -> {path}", flush=True)


def load_cubeful_pool(path=CUBEFUL_POOL):
    """Committed artifact -> (checker_pts, cube_pts) with reconstructed
    BoardState/MatchState, exactly as _capture_cubeful_pool returns them
    (verified equal field-for-field against a fresh capture)."""
    import json
    from backgammon_engine import BoardState
    from modes import MatchState, CubeOwner
    art = json.load(open(path))

    def state_ms(r):
        s = BoardState(points=r["points"], bar=r["bar"], off=r["off"],
                       turn=r["turn"])
        ms = MatchState(cube_owner=CubeOwner(r["cube_owner"]),
                        cube_value=r["cube_value"], jacoby=True)
        return s, ms

    checker_pts, cube_pts = [], []
    for r in art["checker"]:
        s, ms = state_ms(r)
        checker_pts.append((s, tuple(r["dice"]), ms))
    for r in art["cube"]:
        s, ms = state_ms(r)
        cube_pts.append((s, ms))
    return checker_pts, cube_pts


def _time_cubeful_engine(name, agent, checker_pts, cube_pts, cap):
    rck, lck, hck = rate(
        lambda it: agent.choose_checker_action_cubeful(*it), checker_pts, cap)
    rcu, lcu, hcu = rate(
        lambda it: agent.offer_double(*it), cube_pts, cap)
    print(f"{name:<26} checker {rck:9.1f} dec/s 95%CI [{lck:.1f},{hck:.1f}] ({1000/rck:.3f} ms)"
          f"  |  cube {rcu:9.1f} dec/s 95%CI [{lcu:.1f},{hcu:.1f}] ({1000/rcu:.3f} ms)", flush=True)


def run_cubeful(cap, engines):
    checker_pts, cube_pts = load_cubeful_pool()
    print(f"[cubeful pool] loaded {len(checker_pts)} checker pts, "
          f"{len(cube_pts)} cube pts from the committed artifact (cap {cap})",
          flush=True)

    want = [e.strip() for e in engines.split(",")]
    if "puretd" in want:
        _time_cubeful_engine("PureTD 0-ply (4L cubeful)", _load_agent(CUBEFUL),
                             checker_pts, cube_pts, cap)
    if "sage0" in want:
        from experiments.opensage_agents import OpenSageAnalyzerAgent
        # Default (production) move filter — matches the self-play
        # "SAGE-AN production" rows' configuration.
        _time_cubeful_engine("Open Sage 0-ply (analyzer)",
                             OpenSageAnalyzerAgent(sage_plies=1),
                             checker_pts, cube_pts, cap)
    if "sage" in want:
        from experiments.opensage_agents import OpenSageAnalyzerAgent
        # Default (production) move filter — matches the self-play
        # "SAGE-AN production" rows' configuration.
        _time_cubeful_engine("Open Sage 1-ply (analyzer)",
                             OpenSageAnalyzerAgent(sage_plies=2),
                             checker_pts, cube_pts, cap)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["cubeless", "cubeful"], required=True)
    ap.add_argument("--cap", type=int, default=0,
                    help="decisions timed per config (default 8000 cubeless / 5000 cubeful)")
    # cubeless only
    ap.add_argument("--configs", default="prob5-0ply,prob5-1ply,sage-0ply,sage-1ply")
    ap.add_argument("--out", default="", help="cubeless: optional result file")
    # cubeful only
    ap.add_argument("--engines", default="puretd,sage0,sage")
    ap.add_argument("--regen-pool", action="store_true",
                    help="cubeful: regenerate the committed decision-pool "
                         "artifact (deterministic: seed 1, first 10k checker "
                         "/ 6k cube decisions)")
    a = ap.parse_args()

    if a.mode == "cubeless":
        run_cubeless(a.cap or 8000, a.configs, a.out)
    elif a.regen_pool:
        export_cubeful_pool()
    else:
        run_cubeful(a.cap or 5000, a.engines)


if __name__ == "__main__":
    main()
