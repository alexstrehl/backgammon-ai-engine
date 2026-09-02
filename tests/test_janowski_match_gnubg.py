"""Cross-validate janowski_match against the real gnubg 1.08.003.

The strongest available ground truth for the conversion: drive the
gnubg CLI over a pool of self-play positions at real match
scores/cube states, read back gnubg's OWN printed 0-ply
probabilities, its cubeless MWC, its cubeful MWC, and (where a double
is legal) its ND/DT/DP cubeful triple -- then feed gnubg's
probabilities through OUR converter and require agreement.

Because the probabilities are gnubg's on both sides of the
comparison, this isolates the CONVERSION (Cl2CfMatch + GetPoints port
+ MET plumbing) from any difference between the nets.

Positions reach gnubg via `set matchid` / `set board` using the
encoders in gnubg_ids.py (anchored against gnubg's own output).

Needs the pinned referee build: /usr/local/bin/gnubg (1.08.003, built
with both bearoff databases). Skips cleanly without it.
"""

import os
import random
import re
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backgammon_engine import BoardState, opening_roll, pip_count, switch_turn
from encoding import CubePerspective
from gnubg_ids import ids_for
from janowski_match import cubeful_mwc, ctx_for, dead_mwc, _me
from met import default_met
from modes import CubeOwner, MatchState, cube_perspective

GNUBG = os.environ.get("GNUBG_CMD_XCHECK", "/usr/local/bin/gnubg")
MET_XML = "/usr/local/share/gnubg/met/Kazaross-XG2.xml"
MET = default_met()

pytestmark = pytest.mark.skipif(
    not (os.path.exists(GNUBG) and os.path.exists(MET_XML)),
    reason="pinned gnubg 1.08.003 build not present",
)

PROD = "best_models/cubeless_prob5_512_512_256_128.pt"

# gnubg EvalEfficiency, shallow-ply (0-1 ply) constants -- eval.c.
_CONTACT_X = 0.76


def _gnubg_x(evaluator: str, state: BoardState) -> float:
    """x gnubg used for this eval, from its printed position class."""
    if evaluator in ("Contact", "Crashed"):
        return _CONTACT_X
    if evaluator == "Race":
        pips = pip_count(state)[state.turn]
        return min(0.80, max(0.60, 0.60 + 0.0025 * pips))
    return None            # bearoff / exact classes: not x-converted


def _sample_pool(n_target=140, seed=13):
    """(state, match_state, match_length) rows from p5jan self-play,
    covering centered/MINE/THEIRS cube states across 3pt/5pt/7pt
    scores. Pre-roll points only (that is what eval prices)."""
    from janowski_match import JanowskiMatchAgent
    from model import ProbNetwork

    net = ProbNetwork.load(PROD)
    agent = JanowskiMatchAgent(net, met=MET, x_policy=0.68)
    rng = random.Random(seed)
    pool = []

    for length in (3, 5, 7, 3, 3, 5, 3, 7, 3, 5):
        ms = MatchState(jacoby=False, match_length=length,
                        my_away=length, opp_away=length)
        while True:
            gms = ms.start_game()
            state, dice = opening_roll(rng)
            opening = True
            while True:
                player = state.turn
                if not opening and len(pool) < n_target \
                        and rng.random() < 0.30 \
                        and not state.is_game_over():
                    pool.append((state, gms, length))
                if not opening and gms.can_offer(player):
                    offer = agent.offer_double(state, gms)
                    if offer.should_double:
                        if agent.respond_to_double(state, gms, hint=offer):
                            gms = gms.after_take(player)
                        else:
                            ms = gms.after_game(1, winner_is_me=(player == 0))
                            break
                res = agent.choose_checker_action_cubeful(state, dice, gms)
                if res is None:
                    state = switch_turn(state)
                else:
                    if res[0].is_game_over():
                        ms = gms.after_game(
                            res[0].game_result(), winner_is_me=(player == 0))
                        break
                    state = res[0]
                dice = (rng.randint(1, 6), rng.randint(1, 6))
                opening = False
            if ms.match_over() or len(pool) >= n_target:
                break
        if len(pool) >= n_target:
            break
    return pool


_STATIC = re.compile(
    r"static:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)%\s+([\d.]+)%")
_CUBE_LINE = re.compile(
    r"\d\. (No double|Double, take|Double, pass|No redouble|"
    r"Redouble, take|Redouble, pass)\s+([\d.]+)%")


def _drive_gnubg(rows):
    """One gnubg process, one eval per row; returns parsed records."""
    cmds = [
        "set player 0 human", "set player 1 human",
        "set display off",
        "set output mwc on", "set output digits 6",
        f"set matchequity {MET_XML}",
        "set evaluation chequer eval plies 0",
        "set evaluation cubedecision eval plies 0",
    ]
    for state, gms, length in rows:
        pid, mid = ids_for(state, gms, length)
        if not cmds or not cmds[-1].startswith("new match"):
            pass
        cmds += [f"new match {length}",
                 f"set matchid {mid}", f"set board {pid}", "eval"]
    out = subprocess.run(
        [GNUBG, "-t", "-q"], input="\n".join(cmds) + "\n",
        capture_output=True, text=True, timeout=600,
    ).stdout
    # eval echoes "Position ID:\t..." (tab); board echoes use spaces.
    recs = out.split("Position ID:\t")[1:]
    assert len(recs) == len(rows), (len(recs), len(rows))
    parsed = []
    for rec in recs:
        m = _STATIC.search(rec)
        ev = re.search(r"Evaluator: \t(\w[\w ()-]*)", rec)
        cubes = _CUBE_LINE.findall(rec)
        parsed.append({
            "probs": np.array([float(g) for g in m.groups()[:5]]),
            "mwc_dead": float(m.group(6)) / 100.0,
            "mwc_cubeful": float(m.group(7)) / 100.0,
            "evaluator": ev.group(1).strip() if ev else "?",
            "cube_triple": {k: float(v) / 100.0 for k, v in cubes},
        })
    return parsed


@pytest.mark.skipif(not os.path.exists(PROD), reason="no prob5 checkpoint")
def test_conversion_matches_gnubg():
    """Our Cl2CfMatch port must reproduce gnubg's dead AND cubeful
    MWCs on gnubg's own probabilities, across cube states, scores and
    position classes -- and the ND/DT/DP triple where a double is
    legal."""
    pool = _sample_pool()
    parsed = _drive_gnubg(pool)

    checked = dead_max = cf_max = 0
    triple_checked = triple_max = 0
    skipped = {}
    for (state, gms, length), rec in zip(pool, parsed):
        x = _gnubg_x(rec["evaluator"], state)
        if x is None:
            skipped[rec["evaluator"]] = skipped.get(rec["evaluator"], 0) + 1
            continue
        player = state.turn
        persp = cube_perspective(gms.cube_owner, player)
        ctx = ctx_for(gms, player)
        probs = rec["probs"]

        d = dead_mwc(probs, ctx, MET)
        dead_max = max(dead_max, abs(d - rec["mwc_dead"]))
        v = cubeful_mwc(probs, persp, ctx, x, MET)
        cf_max = max(cf_max, abs(v - rec["mwc_cubeful"]))
        checked += 1

        trip = rec["cube_triple"]
        if trip:
            nd_key = ("No double" if "No double" in trip else "No redouble")
            v_nd = cubeful_mwc(probs, persp, ctx, x, MET)
            v_dt = cubeful_mwc(
                probs, CubePerspective.THEIRS,
                ctx.with_cube(2 * ctx.cube), x, MET)
            v_dp = _me(MET, ctx, ctx.cube, True)
            for key, ours in (
                (nd_key, v_nd),
                ("Double, take", v_dt), ("Redouble, take", v_dt),
                ("Double, pass", v_dp), ("Redouble, pass", v_dp),
            ):
                if key in trip:
                    triple_max = max(triple_max, abs(ours - trip[key]))
                    triple_checked += 1

    print(f"\nchecked {checked} rows (skipped classes: {skipped})")
    print(f"max |dead diff|    = {dead_max:.2e}")
    print(f"max |cubeful diff| = {cf_max:.2e}")
    print(f"ND/DT/DP entries checked {triple_checked}, "
          f"max diff = {triple_max:.2e}")
    assert checked >= 60, "pool too thin after class filtering"
    # gnubg prints 5 sig-decimals of percent -> quantization 5e-6 in
    # MWC, plus float32-vs-float64 arithmetic; 5e-5 catches any real
    # structural divergence while tolerating print rounding.
    assert dead_max < 5e-5
    assert cf_max < 5e-5
    assert triple_checked >= 20
    assert triple_max < 5e-5
