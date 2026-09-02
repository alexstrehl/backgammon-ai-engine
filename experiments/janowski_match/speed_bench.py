"""Decision-throughput benchmark: trained match-play champion vs p5jan.
Both single-threaded on identical positions, 0-ply acting (the objective)."""
import os, sys, time, random
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)  # model paths below are repo-relative
from backgammon_engine import get_legal_plays, opening_roll, switch_turn
from model import TDNetwork, ProbNetwork
from td_agent import TDAgent
from janowski_match import JanowskiMatchAgent
from met import default_met
from modes import MatchState

CH = "best_models/matchplay_1h64met_512_512_256_256.pt"
P5 = "best_models/cubeless_prob5_512_512_256_128.pt"
MET = default_met()
champ = TDAgent(TDNetwork.load(CH), met=MET)
p5jan = JanowskiMatchAgent(ProbNetwork.load(P5), met=MET)

# collect a shared position pool by self-play with the champion
rng = random.Random(11)
ms = MatchState(jacoby=False, match_length=11, my_away=11, opp_away=11).start_game()
pool, cube_pool = [], []
state, dice = opening_roll(rng)
while len(pool) < 400:
    plays = get_legal_plays(state, dice)
    if len(plays) >= 2:
        pool.append((state, dice))
        if len(cube_pool) < 200 and ms.can_offer(state.turn):
            cube_pool.append(state)
    r = champ.choose_checker_action_matchplay(state, dice, ms)
    if r is None: state = switch_turn(state)
    elif r[0].is_game_over(): state, dice = opening_roll(rng); continue
    else: state = r[0]
    dice = (rng.randint(1, 6), rng.randint(1, 6))

def timeit(fn, items, warm=20):
    for it in items[:warm]: fn(it)
    t = time.perf_counter()
    for it in items: fn(it)
    el = time.perf_counter() - t
    return len(items)/el, el/len(items)*1e6      # per-sec, microseconds

print(f"pool: {len(pool)} checker positions, {len(cube_pool)} cube points\n")
print(f"{'decision':<22}{'champion':>22}{'p5jan':>22}")
r1 = timeit(lambda p: champ.choose_checker_action_matchplay(p[0], p[1], ms), pool)
r2 = timeit(lambda p: p5jan.choose_checker_action_cubeful(p[0], p[1], ms), pool)
print(f"{'checker (0-ply)':<22}{r1[0]:>10.0f}/s {r1[1]:>8.0f}us{r2[0]:>10.0f}/s {r2[1]:>8.0f}us")
c1 = timeit(lambda s: champ.offer_double_matchplay(s, ms), cube_pool)
c2 = timeit(lambda s: p5jan.offer_double(s, ms), cube_pool)
print(f"{'cube offer':<22}{c1[0]:>10.0f}/s {c1[1]:>8.0f}us{c2[0]:>10.0f}/s {c2[1]:>8.0f}us")
# take/pass with NO hint -- what a head-to-head between DIFFERENT agents
# pays (eval_matchplay passes hint only when both sides are the same agent)
t1 = timeit(lambda s: champ.respond_to_double_matchplay(s, ms, hint=None), cube_pool)
t2 = timeit(lambda s: p5jan.respond_to_double(s, ms, hint=None), cube_pool)
print(f"{'take/pass (no hint)':<22}{t1[0]:>10.0f}/s {t1[1]:>8.0f}us{t2[0]:>10.0f}/s {t2[1]:>8.0f}us")
# and with the offer's cache reused (self-play path)
h1 = [champ.offer_double_matchplay(s, ms) for s in cube_pool]
h2 = [p5jan.offer_double(s, ms) for s in cube_pool]
w1 = timeit(lambda i: champ.respond_to_double_matchplay(cube_pool[i], ms, hint=h1[i]), list(range(len(cube_pool))))
w2 = timeit(lambda i: p5jan.respond_to_double(cube_pool[i], ms, hint=h2[i]), list(range(len(cube_pool))))
print(f"{'take/pass (hinted)':<22}{w1[0]:>10.0f}/s {w1[1]:>8.0f}us{w2[0]:>10.0f}/s {w2[1]:>8.0f}us")
print(f"\nchecker: p5jan runs at {r2[0]/r1[0]:.2f}x the champion's rate")
print(f"cube   : p5jan runs at {c2[0]/c1[0]:.2f}x the champion's rate")

