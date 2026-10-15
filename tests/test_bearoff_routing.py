"""Exact bearoff routing of the Janowski comparison's agents.

Every bearoff-class value goes through bearoff._exact_tier: the
two-sided table when both sides have <= TS_MAX checkers, the hazard DP
for larger gammon-free bearoffs, the exact-probability x-conversion
(JanowskiAgent only) while gammons are possible. Checker play (0-ply
successors with the opponent's cube fold, 1-ply leaves) and cube
decisions share it, for JanowskiAgent and BearoffTDAgent (`bo:`).

The agents run on tiny, barely trained 1-hidden-layer nets: their
bearoff-class decisions must still be exactly the tables'.

Fixture tests/bearoff_gnubg_ts.json: 76 two-sided bearoffs (<= 6
checkers per side) with gnubg 1.08.003's 0-ply cubeful hint per
candidate and cfevaluate ND/DT; 46 where the hazard DP picks another
move than the table, case 0 = gnubg ID ygAAYBAAAAAAAA:AQEFAAAAAAAA.
Measured: candidates max |err| 9.1e-5, ND/DT 1.7e-4 (gnubg's db is
quantized).
"""

import json
import os

import numpy as np
import pytest
import torch

import bearoff
from backgammon_engine import (BoardState, WHITE, BLACK, get_legal_plays,
                               switch_turn)
from encoding import CubePerspective
from modes import CubeOwner, MatchState, cube_perspective

pytestmark = pytest.mark.skipif(
    not (os.path.exists(bearoff.DEFAULT_TABLE)
         and os.path.exists(bearoff.DEFAULT_TS_TABLE)),
    reason="bearoff tables not built")

CENT = CubePerspective.CENTERED
MINE = CubePerspective.MINE
THEIRS = CubePerspective.THEIRS
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bearoff_gnubg_ts.json")
GNUBG_TOL = 3e-4     # gnubg's ts db quantization (max seen 1.7e-4)
TIE = 2e-7          # float32 resolution: JanowskiAgent compares in float32


# ── helpers ──────────────────────────────────────────────────────────


def _state(cme, copp, turn=WHITE, outside=None, bar=None):
    """On-roll side's home counts `cme` (points 1..6), opponent's `copp`.
    `outside`: {(player, point 7..24): n} extra checkers."""
    pts = [0] * 24
    W, B = (cme, copp) if turn == WHITE else (copp, cme)
    for i, v in enumerate(W):
        pts[i] += v
    for i, v in enumerate(B):
        pts[23 - i] -= v
    on = {WHITE: sum(W), BLACK: sum(B)}
    for (player, point), n in (outside or {}).items():
        if player == WHITE:
            pts[point - 1] += n
        else:
            pts[24 - point] -= n
        on[player] += n
    b = list(bar or [0, 0])
    on[WHITE] += b[WHITE]
    on[BLACK] += b[BLACK]
    return BoardState(points=pts, bar=b,
                      off=[15 - on[WHITE], 15 - on[BLACK]], turn=turn)


def _key(s):
    return tuple(s.points), tuple(s.bar), tuple(s.off), s.turn


def _ms(rel, turn, cube=1, jacoby=True):
    """MatchState with the cube centered / the on-roll player's / theirs."""
    owner = {"centered": CubeOwner.CENTERED, "mine": CubeOwner(turn + 1),
             "theirs": CubeOwner(2 - turn)}[rel]
    return MatchState(cube_owner=owner, cube_value=cube, jacoby=jacoby)


DICE = [(d1, d2) for d1 in range(1, 7) for d2 in range(d1, 7)]
DICE_P = [(1.0 if a == b else 2.0) / 36.0 for a, b in DICE]


def _succ_value_opp(s, opp_persp):
    """Gammon-free successor value, opponent's view, from the tables."""
    if s.is_game_over():
        return -1.0
    v = bearoff.race_equity(s, opp_persp, with_own_fold=True)
    assert v is not None
    return v


def _table_best(state, dice, ms):
    """(best opp-view value, {keys of successors achieving it})."""
    op = cube_perspective(ms.cube_owner, 1 - state.turn)
    vals = [(_succ_value_opp(s, op), _key(s))
            for _, s in get_legal_plays(state, dice)]
    best = min(v for v, _ in vals)
    return best, {k for v, k in vals if v <= best + TIE}


def _table_cube(state, ms, name="bo"):
    """(should_double, takes) from the tables with the agents' shared
    money cube rule (TDAgent._money_cube_decision, take iff DT <= 1)."""
    from td_agent import TDAgent
    p = cube_perspective(ms.cube_owner, state.turn)
    nd, dt = bearoff.race_cube_values(state, p)
    should = (p != THEIRS
              and TDAgent._money_cube_decision(nd, dt, p, ms.jacoby))
    return should, dt <= 1.0


def _rand_home(rng, lo, hi):
    c = [0] * 6
    for _ in range(int(rng.integers(lo, hi + 1))):
        c[int(rng.integers(0, 6))] += 1
    return c


# ── tiny nets and agents ─────────────────────────────────────────────


def _barely_trained(net, n_in, n_out, seed):
    torch.manual_seed(seed)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.5)
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    x = torch.rand(64, n_in)
    y = torch.rand(64, n_out).squeeze(-1)
    for _ in range(5):
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    net.eval()
    return net


def _prob5_net(seed):
    from model import ProbNetwork
    net = ProbNetwork(hidden_sizes=[8], encoder_name="perspective196")
    return _barely_trained(net, net.input_size, 5, seed)


def _cubeful_net(seed):
    from model import TDNetwork
    net = TDNetwork(hidden_sizes=[8], encoder_name="cubeful_perspective196",
                    output_mode="equity")
    return _barely_trained(net, net.input_size, 1, seed)


@pytest.fixture(scope="module")
def agents():
    """name -> (agent on net seed 1, agent on net seed 2)."""
    from janowski_agent import JanowskiAgent
    from bearoff_td import BearoffTDAgent
    out = {}
    for seed in (1, 2):
        pn, cn = _prob5_net(seed), _cubeful_net(seed)
        for name, a in (("jan0", JanowskiAgent(pn)),
                        ("jan1", JanowskiAgent(pn, oneply=True)),
                        ("bo0", BearoffTDAgent(cn)),
                        ("bo1", BearoffTDAgent(cn, oneply=True))):
            out.setdefault(name, []).append(a)
    return out


ALL = ("jan0", "jan1", "bo0", "bo1")


def _choose(agent, state, dice, ms):
    out = agent.choose_checker_action_cubeful(state, dice, ms)
    return _key(out[0]), out[1]


def _cube(agent, state, ms):
    offer = agent.offer_double(state, ms)
    return offer.should_double, agent.respond_to_double(state, ms)


# ── routing tiers ────────────────────────────────────────────────────


def test_tier_boundaries():
    eight = [2, 2, 1, 1, 1, 1]
    nine = [2, 2, 2, 1, 1, 1]
    assert bearoff._exact_tier(_state(eight, eight))[0] == "ts"
    assert bearoff._exact_tier(_state(nine, [1, 0, 0, 0, 0, 0]))[0] == "hazard"
    assert bearoff._exact_tier(_state([1, 0, 0, 0, 0, 0], nine))[0] == "hazard"
    assert bearoff._exact_tier(_state(nine, nine))[0] == "hazard"
    for turn in (WHITE, BLACK):
        # gammons possible: no exact cubeful tier
        g = _state([3, 3, 3, 2, 2, 2], [1, 0, 0, 0, 0, 0], turn)
        assert g.off[turn] == 0 and bearoff.is_bearoff_class(g)
        assert bearoff._exact_tier(g) is None
        assert bearoff.race_equity(g, CENT, with_own_fold=True) is None
        assert bearoff.race_cube_values(g, CENT) is None
    # not bearoff class: outside checker, bar
    out = _state([1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
                 outside={(WHITE, 7): 1})
    bar = _state([1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], bar=[0, 1])
    for s in (out, bar):
        assert not bearoff.is_bearoff_class(s)
        assert bearoff.race_cube_values(s, CENT) is None
        assert bearoff.race_equity(s, CENT, with_own_fold=True) is None
    # game over: the mover cleared, opponent on roll
    over = BoardState(points=[0] * 23 + [-1], off=[15, 14], turn=BLACK)
    assert over.is_game_over()
    assert bearoff._exact_tier(over) is None
    assert bearoff.race_equity(over, CENT, with_own_fold=True) is None


def test_eight_checkers_table_nine_hazard():
    """8 per side: values are the table's even where the hazard DP is
    off; 9: the hazard DP's."""
    rng = np.random.default_rng(8)
    gaps = 0
    for _ in range(60):
        s8 = _state(_rand_home(rng, 8, 8), _rand_home(rng, 1, 8))
        s9 = _state(_rand_home(rng, 9, 9), _rand_home(rng, 1, 8))
        for p in (CENT, MINE, THEIRS):
            nd, dt = bearoff.race_cube_values(s8, p)
            assert (nd, dt) == bearoff.ts_cube_values(s8, p)
            assert bearoff.race_equity(s8, p, with_own_fold=True) == \
                pytest.approx(bearoff.ts_equity(s8, p), abs=1e-12)
            hm, ho = bearoff.race_hazards(s8)
            gaps += abs(bearoff._race_dp(hm, ho, int(p)) - nd) > 1e-3
            hm, ho = bearoff.race_hazards(s9)
            nd9, dt9 = bearoff.race_cube_values(s9, p)
            assert nd9 == bearoff._race_dp(hm, ho, int(p))
            assert dt9 == 2.0 * bearoff._race_dp(hm, ho, int(THEIRS))
            assert bearoff.race_equity(s9, p) == nd9
    assert gaps > 0          # the boundary is visible in the values


def test_fold_is_one_function_of_the_cube_values():
    """race_equity == max(ND, min(DT, 1)) of race_cube_values (centered,
    owned), == ND (theirs, or no fold), in both tiers and both colours:
    checker play and cube decisions read the same numbers."""
    rng = np.random.default_rng(5)
    for k in range(120):
        hi = 8 if k % 2 else 12
        s = _state(_rand_home(rng, 1, hi), _rand_home(rng, 1, hi),
                   turn=k % 3 % 2)
        if bearoff._exact_tier(s) is None:
            continue
        for p in (CENT, MINE, THEIRS):
            nd, dt = bearoff.race_cube_values(s, p)
            folded = nd if p == THEIRS else max(nd, min(dt, 1.0))
            assert bearoff.race_equity(s, p, with_own_fold=True) == \
                pytest.approx(folded, abs=2e-6)       # V is float32
            assert bearoff.race_equity(s, p) == pytest.approx(nd, abs=2e-6)


# ── folded value vs an independent brute force ───────────────────────


def _brute_points(cme, copp, owner, cube, memo):
    """Pre-roll money equity in points, on-roll view, with the on-roll
    player's own double window; gammon-free bearoff, unlimited cube.
    owner: "centered" / "roller" / "other". Real BoardStates and dice,
    nothing from bearoff.py."""
    key = (cme, copp, owner, cube)
    if key in memo:
        return memo[key]

    def roll(own, cv):
        s = _state(list(cme), list(copp))
        tot = 0.0
        for dice, pd in zip(DICE, DICE_P):
            best = None
            for _, ns in get_legal_plays(s, dice):
                if ns.is_game_over():
                    v = float(cv)
                else:
                    nme = tuple(max(0, ns.points[i]) for i in range(6))
                    flip = {"centered": "centered", "roller": "other",
                            "other": "roller"}[own]
                    v = -_brute_points(copp, nme, flip, cv, memo)
                best = v if best is None else max(best, v)
            tot += pd * best
        return tot

    nd = roll(owner, cube)
    v = nd
    if owner in ("centered", "roller"):
        dt = roll("other", 2 * cube)
        dp = float(cube)
        if dt >= nd and dp >= nd:           # gnubg GetECF3
            v = min(dt, dp)
    memo[key] = v
    return v


SMALL = [
    ((0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1)),   # man-on-6 each: doubles
    ((1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)),
    ((0, 1, 1, 0, 0, 0), (1, 0, 0, 1, 0, 0)),
    ((2, 1, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1)),
    ((0, 0, 0, 1, 1, 0), (2, 0, 0, 0, 0, 0)),
    ((0, 0, 0, 0, 2, 0), (1, 1, 0, 0, 0, 0)),   # fold binds as a pass
]


@pytest.mark.parametrize("cme,copp", SMALL)
@pytest.mark.parametrize("cube", [1, 2, 4])
@pytest.mark.parametrize("rel", ["centered", "mine", "theirs"])
@pytest.mark.parametrize("turn", [WHITE, BLACK])
def test_folded_value_vs_brute_force(cme, copp, cube, rel, turn):
    """Successor form (opponent on roll) per cube state, from real
    MatchStates: race_equity * cube == brute force in points."""
    memo = _BRUTE_MEMO
    s = _state(list(cme), list(copp), turn=turn)
    ms = _ms(rel, turn, cube)
    p = cube_perspective(ms.cube_owner, turn)
    owner = {"centered": "centered", "mine": "roller",
             "theirs": "other"}[rel]
    bf = _brute_points(cme, copp, owner, cube, memo)
    assert cube * bearoff.race_equity(s, p, with_own_fold=True) == \
        pytest.approx(bf, abs=2e-5)
    # no-double value and doubled-taken value of the cube decision
    nd, dt = bearoff.race_cube_values(s, p)
    if owner != "other":
        bf_nd = _roll_only(cme, copp, owner, cube, memo)
        bf_dt = _roll_only(cme, copp, "other", 2 * cube, memo)
        assert cube * nd == pytest.approx(bf_nd, abs=2e-5)
        assert cube * dt == pytest.approx(bf_dt, abs=4e-5)
    else:
        assert cube * nd == pytest.approx(bf, abs=2e-5)


_BRUTE_MEMO: dict = {}


def _roll_only(cme, copp, owner, cube, memo):
    """Brute-force value without the on-roll player's own window."""
    s = _state(list(cme), list(copp))
    tot = 0.0
    for dice, pd in zip(DICE, DICE_P):
        best = None
        for _, ns in get_legal_plays(s, dice):
            if ns.is_game_over():
                v = float(cube)
            else:
                nme = tuple(max(0, ns.points[i]) for i in range(6))
                flip = {"centered": "centered", "roller": "other",
                        "other": "roller"}[owner]
                v = -_brute_points(copp, nme, flip, cube, memo)
            best = v if best is None else max(best, v)
        tot += pd * best
    return tot


def test_brute_force_exercises_every_fold_branch():
    """The SMALL set covers no double, double/take and double/pass."""
    seen = set()
    for cme, copp in SMALL:
        s = _state(list(cme), list(copp))
        nd, dt = bearoff.race_cube_values(s, CENT)
        seen.add("pass" if dt > 1 and nd <= 1 else
                 "take" if dt >= nd else "nodouble")
    assert seen == {"pass", "take", "nodouble"}


# ── table vs hazard: the agents take the table's move ────────────────


REPRO = dict(cme=[2, 0, 0, 0, 0, 1], copp=[0, 1, 1, 0, 2, 0])


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("turn", [WHITE, BLACK])
def test_repro_takes_table_move(agents, name, turn):
    """gnubg ygAAYBAAAAAAAA:AQEFAAAAAAAA, 21, roller owns a 2-cube:
    6/4 1/off (0.889) beats 6/3 (0.601); the hazard DP rates both 0.889."""
    s = _state(REPRO["cme"], REPRO["copp"], turn=turn)
    ms = _ms("mine", turn, cube=2)
    plays = {tuple(ns.points): ns for _, ns in get_legal_plays(s, (2, 1))}
    assert len(plays) == 2
    op = cube_perspective(ms.cube_owner, 1 - turn)
    vals = {k: -bearoff.race_equity(ns, op, with_own_fold=True)
            for k, ns in plays.items()}
    haz = {}
    for k, ns in plays.items():
        hm, ho = bearoff.race_hazards(ns)
        haz[k] = -bearoff._race_dp(hm, ho, int(op))
    assert sorted(vals.values()) == pytest.approx([0.6013, 0.8889], abs=1e-4)
    assert max(haz.values()) - min(haz.values()) < 1e-6
    best = max(vals, key=vals.get)
    for a in agents[name]:
        key, v = _choose(a, s, (2, 1), ms)
        assert key[0] == best
        if v is not None:
            assert v == pytest.approx(vals[best], abs=1e-9)


# ── barebones nets: bearoff decisions are the tables' ────────────────


def _bearoff_positions(seed, n, lo, hi, cube_states=True):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        turn = int(rng.integers(0, 2))
        s = _state(_rand_home(rng, lo, hi), _rand_home(rng, lo, hi), turn)
        if bearoff._exact_tier(s) is None:
            continue
        dice = DICE[int(rng.integers(0, 21))]
        rel = ["centered", "mine", "theirs"][int(rng.integers(0, 3))]
        cube = 1 if rel == "centered" else [1, 2, 4][int(rng.integers(0, 3))]
        jac = bool(rng.integers(0, 2))
        out.append((s, dice, _ms(rel, turn, cube, jac)))
    return out


@pytest.mark.parametrize("tier,lo,hi", [("ts", 1, 8), ("hazard", 9, 14)])
@pytest.mark.parametrize("name", ALL)
def test_garbage_net_moves_are_table_moves(agents, name, tier, lo, hi):
    seed = 10 * ALL.index(name) + (tier == "ts")
    for s, dice, ms in _bearoff_positions(seed, 25, lo, hi):
        best, keys = _table_best(s, dice, ms)
        for a in agents[name]:
            key, v = _choose(a, s, dice, ms)
            assert key in keys, (tier, name, s, dice, ms)
            if v is not None:                   # bo: returns its value
                assert v == pytest.approx(-best, abs=1e-9)


@pytest.mark.parametrize("tier,lo,hi", [("ts", 1, 8), ("hazard", 9, 14)])
@pytest.mark.parametrize("name", ALL)
def test_garbage_net_cube_actions_are_table_actions(agents, name, tier, lo,
                                                    hi):
    for s, _, ms in _bearoff_positions(7 + len(name), 25, lo, hi):
        want = _table_cube(s, ms, name)
        for a in agents[name]:
            assert _cube(a, s, ms) == want, (tier, name, s, ms)


@pytest.mark.parametrize("name", ALL)
def test_cube_value_and_jacoby_do_not_change_bearoff_decisions(agents, name):
    """Gammon-free: values are per unit cube, so neither the cube size
    nor Jacoby changes checker play, and cube actions are the tables'
    under the agent's cube rule."""
    a = agents[name][0]
    for s, dice, _ in _bearoff_positions(3, 10, 1, 8):
        for rel in ("centered", "mine", "theirs"):
            cubes = (1,) if rel == "centered" else (1, 2, 4)
            moves = {_choose(a, s, dice, _ms(rel, s.turn, c, j))[0]
                     for c in cubes for j in (False, True)}
            assert len(moves) == 1
            for j in (False, True):
                acts = {_cube(a, s, _ms(rel, s.turn, c, j)) for c in cubes}
                assert acts == {_table_cube(s, _ms(rel, s.turn, 1, j), name)}


@pytest.mark.parametrize("name", ["jan0", "jan1"])
def test_janowski_gammonish_bearoff_is_net_free(agents, name):
    """Gammons possible: exact probs through the x-conversion at 0-ply
    and (via the routed leaves) at 1-ply; both nets agree."""
    from janowski import DEFAULT_X, janowski_equity_folded
    rng = np.random.default_rng(11)
    n = 0
    while n < 6:
        turn = int(rng.integers(0, 2))
        heavy = _rand_home(rng, 15, 15)
        light = _rand_home(rng, 1, 6)
        cme, copp = (heavy, light) if n % 2 else (light, heavy)
        s = _state(cme, copp, turn)
        dice = DICE[int(rng.integers(0, 21))]
        ms = _ms(["centered", "mine", "theirs"][n % 3], turn)
        k1, k2 = (_choose(a, s, dice, ms)[0] for a in agents[name])
        assert k1 == k2
        assert _cube(agents[name][0], s, ms) == _cube(agents[name][1], s, ms)
        if name == "jan0":
            op = cube_perspective(ms.cube_owner, 1 - turn)
            vals = {}
            for _, ns in get_legal_plays(s, dice):
                if ns.is_game_over():
                    r = 1.0 if ms.jacoby and ms.cube_owner == 0 else \
                        float(ns.game_result())
                    vals[_key(ns)] = -r
                    continue
                v = bearoff.race_equity(ns, op, with_own_fold=True)
                if v is None:
                    v = float(janowski_equity_folded(
                        bearoff.exact_probs5(ns), op, DEFAULT_X,
                        ms.jacoby))
                vals[_key(ns)] = v
            best = min(vals.values())
            assert vals[k1] <= best + TIE
        n += 1


@pytest.mark.parametrize("name", ALL)
def test_garbage_net_plays_contact(agents, name):
    """Non-bearoff decisions are the net's (anything legal)."""
    s = BoardState.initial()
    ms = MatchState()
    for a in agents[name]:
        key, _ = _choose(a, s, (3, 1), ms)
        assert key[3] == BLACK
        _cube(a, s, ms)


# ── 1-ply leaves ─────────────────────────────────────────────────────


def _leaf_root():
    """WHITE on roll with its last checker on its 7-point (every roll
    brings it home), BLACK 3 checkers home: every successor is bearoff
    class or terminal, the root is not."""
    return _state([0] * 6, [1, 1, 1, 0, 0, 0], WHITE,
                  outside={(WHITE, 7): 1})


def _jan_oneply_branches(s, branches):
    """1-ply over exact leaves, roller (s.turn) view per root cube state:
    leaves fold the leaf player's own cube decision; per roll each branch
    takes its own best leaf."""
    roller, leaf = s.turn, 1 - s.turn
    lps = [cube_perspective(b.cube_owner, leaf) for b in branches]
    tot = [0.0] * len(branches)
    for dice, pd in zip(DICE, DICE_P):
        rows = []
        for _, ns in get_legal_plays(s, dice):
            if ns.is_game_over():
                rows.append([1.0 if not _gammons(b) else
                             float(ns.game_result()) for b in branches])
            else:
                assert bearoff._exact_tier(ns) is not None
                rows.append([-bearoff.race_equity(ns, lp, with_own_fold=True)
                             for lp in lps])
        for b in range(len(branches)):
            tot[b] += pd * max(r[b] for r in rows)
    return tot


def _gammons(ms):
    return not (ms.jacoby and ms.cube_owner == CubeOwner.CENTERED)


@pytest.mark.parametrize("rel", ["centered", "mine", "theirs"])
def test_janowski_oneply_leaves_routed(agents, rel):
    """_value_oneply_cubeful at a non-bearoff root = 1-ply over the exact
    folded leaf values, for both nets."""
    s = _leaf_root()
    assert not bearoff.is_bearoff_class(s)
    for fold in (False, True):
        ms = _ms(rel, WHITE)
        branches = [ms] + ([ms.after_take(WHITE)]
                           if fold and ms.can_offer(WHITE) else [])
        vals = _jan_oneply_branches(s, branches)
        want = vals[0] if len(vals) == 1 else \
            max(vals[0], min(2 * vals[1], 1.0))
        for a in agents["jan1"]:
            got = a._value_oneply_cubeful(s, ms, fold=fold)
            assert got == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("rel", ["centered", "mine", "theirs"])
def test_bo_oneply_value_leaves_routed(agents, rel):
    """value_oneply_checker_cubeful = expectimax over the exact folded
    leaf values, for both nets."""
    s = _leaf_root()
    ms = _ms(rel, WHITE)
    lp = cube_perspective(ms.cube_owner, BLACK)
    want = 0.0
    for dice, pd in zip(DICE, DICE_P):
        best = max(1.0 if ns.is_game_over()
                   else -bearoff.race_equity(ns, lp, with_own_fold=True)
                   for _, ns in get_legal_plays(s, dice))
        want += pd * best
    for a in agents["bo1"]:
        assert a.value_oneply_checker_cubeful(s, ms) == \
            pytest.approx(want, abs=1e-9)


def _oneply_cand_root():
    """BLACK on roll, all home; WHITE's last checker on its 7-point.
    BLACK's candidates are not bearoff class; every WHITE reply is (or
    wins)."""
    return switch_turn(_leaf_root())


@pytest.mark.parametrize("rel", ["centered", "mine", "theirs"])
@pytest.mark.parametrize("name", ["jan1", "bo1"])
def test_oneply_choice_over_routed_leaves(agents, name, rel):
    """1-ply choice at a root whose candidates expand into bearoff-class
    leaves: the candidate values come from the tables only, so both
    nets pick a move of the independent expectimax's best value."""
    s = _oneply_cand_root()
    mover, opp = BLACK, WHITE
    ms = _ms(rel, mover)
    for dice in [(2, 1), (5, 3), (6, 6)]:
        cands = {}
        for _, c in get_legal_plays(s, dice):
            assert not bearoff.is_bearoff_class(c)
            if c.is_game_over():
                cands[_key(c)] = 1.0
                continue

            def u(branch, fold):
                lp = cube_perspective(branch.cube_owner, mover)
                tot = 0.0
                for d, pd in zip(DICE, DICE_P):
                    worst = None
                    for _, leaf in get_legal_plays(c, d):
                        v = (-1.0 if leaf.is_game_over() else
                             bearoff.race_equity(leaf, lp,
                                                 with_own_fold=fold))
                        worst = v if worst is None else min(worst, v)
                    tot += pd * worst
                return tot              # mover's view

            if name == "bo1":
                ua = u(ms, True)
                if ms.can_offer(opp):
                    from td_agent import TDAgent
                    ub = u(ms.after_take(opp), True)
                    if TDAgent._money_cube_decision(
                            -ua, -2 * ub,
                            cube_perspective(ms.cube_owner, opp), ms.jacoby):
                        ua = 2 * ub if -2 * ub <= 1.0 else -1.0
                cands[_key(c)] = ua
            else:
                branches = [ms] + ([ms.after_take(opp)]
                                   if ms.can_offer(opp) else [])
                v = _jan_oneply_branches(c, branches)     # opp's view
                nd = v[0] if len(v) == 1 else max(v[0], min(2 * v[1], 1.0))
                cands[_key(c)] = -nd
        best = max(cands.values())
        for a in agents[name]:
            key, _ = _choose(a, s, dice, ms)
            assert cands[key] >= best - TIE, (dice, rel)


# ── gnubg two-sided bearoff fixture ──────────────────────────────────


def _fixture():
    for c in json.load(open(FIXTURE))["cases"]:
        s = BoardState(points=c["points"], off=c["off"], turn=c["turn"])
        ms = MatchState(cube_owner=CubeOwner(c["cube_owner"]),
                        cube_value=c["cube_value"], jacoby=True)
        yield c, s, ms


def test_fixture_shape():
    cases = list(_fixture())
    assert len(cases) == 76
    assert sum(c["hazard_differs"] for c, _, _ in cases) == 46
    assert {int(ms.cube_owner) for _, _, ms in cases} == {0, 1, 2}
    assert {s.turn for _, s, _ in cases} == {WHITE, BLACK}
    for c, s, _ in cases:
        assert bearoff._exact_tier(s)[0] == "ts"


def test_candidates_match_gnubg():
    worst = 0.0
    for c, s, ms in _fixture():
        op = cube_perspective(ms.cube_owner, 1 - s.turn)
        legal = {tuple(ns.points) for _, ns in get_legal_plays(
            s, tuple(c["dice"]))}
        for m in c["cands"]:
            ns = BoardState(points=m["points"], off=m["off"],
                            turn=1 - s.turn)
            assert tuple(ns.points) in legal
            worst = max(worst, abs(-_succ_value_opp(ns, op) - m["eq"]))
    assert worst < GNUBG_TOL, worst


def test_cube_values_match_gnubg():
    worst = 0.0
    for c, s, ms in _fixture():
        p = cube_perspective(ms.cube_owner, s.turn)
        nd, dt = bearoff.race_cube_values(s, p)
        worst = max(worst, abs(nd - c["nd"]))
        if p != THEIRS:
            worst = max(worst, abs(dt - c["dt"]))
    assert worst < GNUBG_TOL, worst


@pytest.mark.parametrize("name", ALL)
def test_agents_play_gnubg_moves(agents, name):
    """Every agent (tiny nets) picks a gnubg-best move (gnubg equity
    within its db quantization of the best)."""
    for c, s, ms in _fixture():
        geq = {tuple(m["points"]): m["eq"] for m in c["cands"]}
        top = max(geq.values())
        for a in agents[name]:
            key, _ = _choose(a, s, tuple(c["dice"]), ms)
            assert geq[key[0]] >= top - GNUBG_TOL, (c["dice"], c["points"])


@pytest.mark.parametrize("name", ALL)
def test_agents_cube_actions_match_gnubg(agents, name):
    """Double/no-double and take/pass as gnubg, outside exact cash ties
    (ND or DT == 1 within the db quantization: gnubg's tie rule)."""
    n = 0
    for c, s, ms in _fixture():
        p = cube_perspective(ms.cube_owner, s.turn)
        if p == THEIRS:
            continue
        if abs(c["nd"] - 1.0) < GNUBG_TOL or abs(c["dt"] - 1.0) < GNUBG_TOL \
                or abs(c["dt"] - c["nd"]) < GNUBG_TOL:
            continue
        act = c["action"].lower()
        g_double = act.startswith(("double", "redouble"))
        g_take = c["dt"] <= 1.0
        for a in agents[name]:
            should, takes = _cube(a, s, ms)
            assert (should, takes) == (g_double, g_take), (act, c["nd"],
                                                           c["dt"])
        n += 1
    assert n >= 20
