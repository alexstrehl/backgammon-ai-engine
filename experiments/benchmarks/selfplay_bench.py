#!/usr/bin/env python3
"""Self-play computational benchmark — native engines (PureTD/prob5 + Open Sage).

ONE entry point for every NATIVE self-play row (gnubg is a separate file,
selfplay_bench_gnubg.py, because it must run inside the gnubg binary's own
embedded Python). Dispatch with --mode {cubeless,cubeful} and
--engine {puretd,sage}:

  cubeful puretd : PureTD 4L cubeful money self-play, cube OFF then ON
                   (per-type checker/cube attribution solved from the pair)
  cubeful sage   : Open Sage BgBotAnalyzer cubeful money self-play, cube on
  cubeless puretd: prod cubeless prob5 self-play (fused choose_next_state)
  cubeless sage  : OpenSageAgent native cubeless-money selection

--plies is OUR ply numbering (0-ply, 1-ply, 2-ply); for Sage it maps to the
XG-numbered sage_plies = plies + 1 internally. Decision counting is
per-decision-type so game length cannot influence conclusions:
  checker = choose_checker_action[_cubeful] calls with dice
  cube    = offer_double + respond_to_double calls (cubeful only)
Each printed rate carries a game-level bootstrap 95% CI (bench_common.ratio_ci).

Single-threaded by design (torch.set_num_threads(1)); pins to one core
(BENCH_CORE / BENCH_NO_PIN). On AMD run with LD_PRELOAD=c_engine/libfakeintel.so.
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # bench_common

# bench_common import side effects (shared by all native benches): OMP/MKL=1
# set BEFORE torch import, pin to one core (BENCH_CORE / BENCH_NO_PIN=1 —
# with multi-threaded internals, e.g. Sage's analyzer's ~2-thread floor,
# pinning makes wall-time equal true single-core cost), torch single-thread.
import bench_common  # noqa: F401
import torch

from play_models import _load_agent
from gnubg_eval import play_and_record, play_and_record_cubeful
from bench_common import ratio_ci

PURETD_CUBEFUL = "best_models/cubeful_money_512_512_256_256.pt"
PROB5 = "best_models/cubeless_prob5_512_512_256_128.pt"



# --------------------------------------------------------------------------
# shared counting/timing helpers
# --------------------------------------------------------------------------
def _count_cubeful(agent, counts, cube_on=True):
    """Monkeypatch the agent's cubeful decision methods to count calls.
    cube_on=False suppresses actual doubles (pure checker cost) while still
    counting the offer opportunity. Returns a restore() callable."""
    from agents import CubeOffer
    orig_offer = agent.offer_double
    orig_resp = agent.respond_to_double
    orig_choose = agent.choose_checker_action_cubeful

    def offer(s, ms):
        counts["cube_offer"] += 1
        if not cube_on:
            return CubeOffer(should_double=False, _cache=None)
        return orig_offer(s, ms)

    def resp(s, ms, hint=None):
        counts["cube_resp"] += 1
        return orig_resp(s, ms, hint=hint)

    def choose(s, d, ms):
        counts["checker"] += 1
        return orig_choose(s, d, ms)

    agent.offer_double = offer
    agent.respond_to_double = resp
    agent.choose_checker_action_cubeful = choose

    def restore():
        agent.offer_double = orig_offer
        agent.respond_to_double = orig_resp
        agent.choose_checker_action_cubeful = orig_choose
    return restore


def _timed_games(play_one, games, running_count):
    """Play `games` games, timing each; running_count() reads the running
    decision total. Returns (gdts, gdecs) — per-game wall times and decision
    counts, so sum(gdts)/sum(gdecs) reproduce the whole-loop point estimate
    and bench_common.ratio_ci gives the game-level bootstrap CI."""
    gdts, gdecs = [], []
    for _ in range(games):
        d0 = running_count()
        t0 = time.perf_counter()
        play_one()
        gdts.append(time.perf_counter() - t0)
        gdecs.append(running_count() - d0)
    return gdts, gdecs


# --------------------------------------------------------------------------
# cubeful PureTD
# --------------------------------------------------------------------------
def _bench_puretd(agent, games, cube_on, counts):
    restore = _count_cubeful(agent, counts, cube_on=cube_on)
    try:
        gdts, gdecs = _timed_games(
            lambda: play_and_record_cubeful(agent, agent, jacoby=True,
                                            record_moves=False),
            games,
            lambda: counts["checker"] + ((counts["cube_offer"]
                                          + counts["cube_resp"])
                                         if cube_on else 0),
        )
        return sum(gdts), gdts, gdecs
    finally:
        restore()


def run_cubeful_puretd(model, plies, games, seed):
    random.seed(seed)
    agent = _load_agent(model, oneply=(plies == 1), twoply_k=0)

    # warmup
    for _ in range(5):
        play_and_record_cubeful(agent, agent, jacoby=True, record_moves=False)

    results = {}
    for cube_on in (False, True):
        random.seed(seed)
        counts = {"cube_offer": 0, "cube_resp": 0, "checker": 0}
        dt, gdts, gdecs = _bench_puretd(agent, games, cube_on, counts)
        results[cube_on] = (dt, dict(counts))
        tag = "cube-on " if cube_on else "cube-off"
        tot = counts["checker"] + (counts["cube_offer"] + counts["cube_resp"] if cube_on else 0)
        lo, hi = ratio_ci(gdts, gdecs)
        print("PURETD %d-ply %s: %d games %.2fs (%.1f g/s) | checker %d (%.1f dec/s)"
              " | cube-offers %d responses %d | combined %.1f dec/s 95%%CI [%.1f,%.1f]"
              % (plies, tag, games, dt, games / dt,
                 counts["checker"], counts["checker"] / dt,
                 counts["cube_offer"], counts["cube_resp"], tot / dt, lo, hi),
              flush=True)

    # per-type attribution: cube-off run is pure checker cost
    dt0, c0 = results[False]
    dt1, c1 = results[True]
    c_ch = dt0 / c0["checker"]
    ncube1 = c1["cube_offer"] + c1["cube_resp"]
    c_cu = (dt1 - c1["checker"] * c_ch) / max(ncube1, 1)
    print("PURETD %d-ply ATTRIBUTION: checker %.3f ms/dec (%.1f dec/s) | cube %.3f ms/dec (%.1f dec/s)"
          % (plies, 1000 * c_ch, 1 / c_ch, 1000 * c_cu, 1 / c_cu if c_cu > 0 else float("inf")),
          flush=True)


# --------------------------------------------------------------------------
# cubeful Open Sage
# --------------------------------------------------------------------------
def run_cubeful_sage(sage_plies, unpruned, games, seed):
    from experiments.opensage_agents import OpenSageAnalyzerAgent
    random.seed(seed)
    kw = ({"filter_max_moves": 1000, "filter_threshold": 10.0} if unpruned else {})
    agent = OpenSageAnalyzerAgent(sage_plies=sage_plies, **kw)

    counts = {"cube_offer": 0, "cube_resp": 0, "checker": 0}
    _count_cubeful(agent, counts)

    for _ in range(2):  # warmup
        play_and_record_cubeful(agent, agent, jacoby=True, record_moves=False)
    counts.update({"cube_offer": 0, "cube_resp": 0, "checker": 0})

    gdts, gdecs = _timed_games(
        lambda: play_and_record_cubeful(agent, agent, jacoby=True,
                                        record_moves=False),
        games,
        lambda: counts["checker"] + counts["cube_offer"] + counts["cube_resp"],
    )
    dt = sum(gdts)

    tot = counts["checker"] + counts["cube_offer"] + counts["cube_resp"]
    lo, hi = ratio_ci(gdts, gdecs)
    tag = "unpruned" if unpruned else "production"
    print("SAGE-AN %dP %s cube-on: %d games %.2fs (%.2f g/s) | checker %d (%.1f dec/s)"
          " | cube-offers %d responses %d | combined %.1f dec/s 95%%CI [%.1f,%.1f]"
          % (sage_plies, tag, games, dt, games / dt,
             counts["checker"], counts["checker"] / dt,
             counts["cube_offer"], counts["cube_resp"], tot / dt, lo, hi),
          flush=True)


# --------------------------------------------------------------------------
# cubeless prob5 / Open Sage
# --------------------------------------------------------------------------
def _build_cubeless_agent(player, sage_unpruned):
    if player.startswith("prob5"):
        return _load_agent(PROB5, oneply=player.endswith("1ply"))
    from experiments.opensage_agents import OpenSageAgent
    kw = ({"filter_max_moves": 1000, "filter_threshold": 10.0}
          if sage_unpruned else {})
    return OpenSageAgent(sage_plies=1 if player.endswith("0ply") else 2,
                         mode="cubeless-money", **kw)


def run_cubeless(player, sage_unpruned, games, seed):
    random.seed(seed)
    agent = _build_cubeless_agent(player, sage_unpruned)

    counts = {"checker": 0}

    if player.startswith("prob5"):
        # Record-free loop via ProbAgent.choose_next_state: at 0-ply this is
        # the FUSED path (move-gen + encode in one C call + one batched
        # forward) -- the engine-native cost, bit-identical selection to the
        # actions-list path (parity-verified over 1920 decisions 2026-07-27).
        from backgammon_engine import switch_turn, opening_roll

        def play_one():
            state, opening = opening_roll()
            first = True
            while not state.is_game_over():
                dice = opening if first else (random.randint(1, 6),
                                              random.randint(1, 6))
                first = False
                ns = agent.choose_next_state(state, dice)
                if ns is None:
                    state = switch_turn(state)   # dance (not counted)
                else:
                    counts["checker"] += 1
                    state = ns
    else:
        orig = agent.choose_checker_action

        def choose(state, dice, actions):
            counts["checker"] += 1
            return orig(state, dice, actions)

        agent.choose_checker_action = choose

        def play_one():
            play_and_record(agent, agent)

    for _ in range(2):  # warmup
        play_one()
    counts["checker"] = 0

    gdts, gdecs = _timed_games(play_one, games, lambda: counts["checker"])
    dt = sum(gdts)

    lo, hi = ratio_ci(gdts, gdecs)
    tag = ("unpruned" if sage_unpruned else "production") \
        if player.startswith("sage") else "prod"
    print("CUBELESS %s %s: %d games %.2fs (%.2f g/s) | checker %d (%.1f dec/s, %.3f ms/dec) 95%%CI [%.1f,%.1f]"
          % (player, tag, games, dt, games / dt,
             counts["checker"], counts["checker"] / dt,
             1000 * dt / max(counts["checker"], 1), lo, hi), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["cubeless", "cubeful"], required=True)
    ap.add_argument("--engine", choices=["puretd", "sage"], required=True)
    ap.add_argument("--plies", type=int, default=0,
                    help="our numbering (0=0-ply, 1=1-ply, 2=2-ply); "
                         "Sage maps to sage_plies = plies + 1")
    ap.add_argument("--sage-unpruned", action="store_true",
                    help="Sage only: disable the root move filter (1000/10.0)")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=None,
                    help="override the PureTD cubeful model path "
                         "(--mode cubeful --engine puretd only)")
    a = ap.parse_args()

    # PureTD supports 0/1-ply here; Sage supports 0/1/2 (XG 1P/2P/3P).
    # Without this check --plies 2 --engine puretd would silently run 0-ply.
    if a.engine == "puretd" and a.plies not in (0, 1):
        ap.error("--engine puretd supports --plies 0 or 1 only")
    if a.engine == "sage" and a.plies not in (0, 1, 2):
        ap.error("--engine sage supports --plies 0, 1 or 2")
    if a.model and not (a.mode == "cubeful" and a.engine == "puretd"):
        ap.error("--model only applies to --mode cubeful --engine puretd")

    torch.set_num_threads(1)

    if a.mode == "cubeful" and a.engine == "puretd":
        run_cubeful_puretd(a.model or PURETD_CUBEFUL, a.plies, a.games, a.seed)
    elif a.mode == "cubeful" and a.engine == "sage":
        run_cubeful_sage(a.plies + 1, a.sage_unpruned, a.games, a.seed)
    elif a.mode == "cubeless" and a.engine == "puretd":
        run_cubeless("prob5-%dply" % a.plies, a.sage_unpruned, a.games, a.seed)
    else:  # cubeless sage
        run_cubeless("sage-%dply" % a.plies, a.sage_unpruned, a.games, a.seed)


if __name__ == "__main__":
    main()
