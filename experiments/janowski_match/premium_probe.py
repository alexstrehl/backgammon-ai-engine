"""Cube-ownership premium: V(s | I own cube) - V(s | opp owns cube),
same position, same score, cube=2. Must be >= 0 by weak dominance
(owning the cube is an option, options have non-negative value).
Compares the champion net's premium against the oracle's."""
import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # repo root
import numpy as np
from backgammon_engine import opening_roll, switch_turn
from janowski_match import JanowskiMatchAgent, cubeful_mwc, ctx_for
from encoding import CubePerspective
from met import default_met
from model import ProbNetwork, TDNetwork
from modes import CubeOwner, MatchState
from td_agent import TDAgent

met = default_met()
champ = TDAgent(TDNetwork.load("best_models/matchplay_1h64met_512_512_256_256.pt"))
oracle = JanowskiMatchAgent(ProbNetwork.load("best_models/cubeless_prob5_512_512_256_128.pt"), met=met)
rng = random.Random(31)

def ms(owner):
    return MatchState(cube_owner=owner, cube_value=2, jacoby=False,
                      match_length=5, my_away=4, opp_away=4)

rows = []
state, dice = opening_roll(rng)
n = 0
while n < 400:
    # advance a random self-play game (champion 0-ply moves at centered ms)
    g = MatchState(jacoby=False, match_length=5, my_away=4, opp_away=4).start_game()
    res = champ.choose_checker_action_matchplay(state, dice, g)
    if res is None:
        state = switch_turn(state)
    elif res[0].is_game_over():
        state, dice = opening_roll(rng); continue
    else:
        state = res[0]
    dice = (rng.randint(1,6), rng.randint(1,6))
    if rng.random() < 0.4 and not state.is_game_over():
        p = state.turn
        own  = ms(CubeOwner(p + 1))       # mover owns cube 2
        their = ms(CubeOwner(2 - p))      # opponent owns cube 2
        v_own  = champ.evaluate_matchplay(state, own,  is_cube_action=False)
        v_their = champ.evaluate_matchplay(state, their, is_cube_action=False)
        probs = oracle._probs_state(state)
        x = oracle._x(state)
        o_own  = cubeful_mwc(probs, CubePerspective.MINE,   ctx_for(own,  p), x, met)
        o_their = cubeful_mwc(probs, CubePerspective.THEIRS, ctx_for(their, p), x, met)
        rows.append((v_own - v_their, o_own - o_their))
        n += 1

net_p = np.array([r[0] for r in rows]) * 1000
orc_p = np.array([r[1] for r in rows]) * 1000
print(f"n={n}  cube-ownership premium (mMWC), same position, cube 2, 4a4a:")
print(f"  net premium   : mean {net_p.mean():+7.2f} (SE {net_p.std()/np.sqrt(n):.2f})   "
      f"negative in {100*(net_p<0).mean():.1f}% of positions")
print(f"  oracle premium: mean {orc_p.mean():+7.2f} (SE {orc_p.std()/np.sqrt(n):.2f})   "
      f"negative in {100*(orc_p<0).mean():.1f}% of positions")
