"""Janowski cube layer (janowski.py, janowski_agent.py, bearoff_td.py).

Numeric fixtures: Janowski (1993), its Senkiewicz-Ballard example and
its Tables 1/2, plus hand-derived values and self-consistency checks at
fixed x. The agent tests need the production checkpoints and the
bearoff tables (bearoff.py).
"""

import os

import numpy as np
import pytest

from backgammon_engine import (BoardState, WHITE, BLACK, get_legal_plays,
                               pip_count, is_race, switch_turn)
from encoding import CubePerspective
from janowski import (
    DEFAULT_X, cash_point, cubeless_equity, fold_cube_decision,
    janowski_equity, janowski_equity_folded, money_cube_action, money_live,
    take_point, w_l,
)


# ── Engine helpers ───────────────────────────────────────────────────


def test_initial_pips_and_contact():
    s = BoardState.initial()
    assert pip_count(s) == (167, 167)
    assert not is_race(s)


def _race_state(turn=WHITE):
    pts = [0] * 24
    pts[0] = pts[1] = pts[2] = 5      # white 15 checkers, 30 pips
    pts[21] = pts[22] = pts[23] = -5  # black 15 checkers, 30 pips
    return BoardState(points=pts, turn=turn)


def test_race_detection_and_bar():
    r = _race_state()
    assert is_race(r)
    assert pip_count(r) == (30, 30)
    onbar = BoardState(points=r.points[:], bar=[1, 0])
    assert not is_race(onbar)


# ── (W, L) and Janowski's worked example ─────────────────────────────
#
# Senkiewicz-Ballard (Janowski 1993, p.6): White's cubeless chances 35% (31 single,
# 4 gammon), Black 65% (47/17/1). From White's view:
#   W = 1.114, L = 1.292; TP_dead = .3292, TP_live = .2725, TP_2/3 = .2892

_SENK = np.array([0.35, 0.04, 0.0, 0.18, 0.01])


def test_janowski_1993_senkiewicz_wl():
    W, L, p = w_l(_SENK)
    assert W == pytest.approx(1.1143, abs=1e-4)
    assert L == pytest.approx(1.2923, abs=1e-4)
    assert p == pytest.approx(0.35)


def test_janowski_1993_senkiewicz_take_points():
    W, L, _ = w_l(_SENK)
    assert take_point(W, L, 0.0) == pytest.approx(0.3292, abs=1e-4)
    assert take_point(W, L, 1.0) == pytest.approx(0.2725, abs=1e-4)
    assert take_point(W, L, 2.0 / 3.0) == pytest.approx(0.2892, abs=1e-4)


def test_janowski_1993_take_point_tables():
    # Janowski (1993) Table 1a (dead): W=2, L=2 -> 37.5%; W=1, L=1 -> 25%
    assert take_point(2, 2, 0.0) == pytest.approx(0.375)
    assert take_point(1, 1, 0.0) == pytest.approx(0.25)
    # Janowski (1993) Table 1b (live): W=1, L=1 -> 20%; W=2, L=2 -> 33.3%
    assert take_point(1, 1, 1.0) == pytest.approx(0.20)
    assert take_point(2, 2, 1.0) == pytest.approx(1.5 / 4.5, abs=1e-4)
    # Janowski (1993) Table 1c (x=2/3): W=1.5, L=1.5 -> 30.0%
    assert take_point(1.5, 1.5, 2.0 / 3.0) == pytest.approx(0.30, abs=1e-3)
    # Janowski (1993) take-equity Table 2c: W=L=1, x=2/3 -> -0.571
    tp = take_point(1, 1, 2.0 / 3.0)
    e_take = tp * 2.0 - 1.0
    assert e_take == pytest.approx(-0.5714, abs=1e-4)


# ── Piecewise live curve ─────────────────────────────────────────────


def test_money_live_kinks_centered():
    W, L = 1.3, 1.2
    tp = (L - 0.5) / (W + L + 0.5)
    cp = (L + 1.0) / (W + L + 0.5)
    f = lambda p: float(money_live(np.float64(p), W, L,
                                   CubePerspective.CENTERED, jacoby=False))
    assert f(0.0) == pytest.approx(-L)
    assert f(tp) == pytest.approx(-1.0)
    assert f(cp) == pytest.approx(1.0)
    assert f(1.0) == pytest.approx(W)


def test_money_live_jacoby_clamps():
    W, L = 2.0, 2.0
    f = lambda p, j: float(money_live(np.float64(p), W, L,
                                      CubePerspective.CENTERED, jacoby=j))
    assert f(0.0, True) == pytest.approx(-1.0)   # no gammon loss while centered
    assert f(1.0, True) == pytest.approx(1.0)
    assert f(0.0, False) == pytest.approx(-2.0)
    assert f(1.0, False) == pytest.approx(2.0)


def test_money_live_owned_and_unavailable():
    W, L = 1.4, 1.1
    tp = (L - 0.5) / (W + L + 0.5)
    cp = (L + 1.0) / (W + L + 0.5)
    owned = lambda p: float(money_live(np.float64(p), W, L,
                                       CubePerspective.MINE, jacoby=True))
    unav = lambda p: float(money_live(np.float64(p), W, L,
                                      CubePerspective.THEIRS, jacoby=True))
    assert owned(0.0) == pytest.approx(-L)
    assert owned(cp) == pytest.approx(1.0)
    assert owned(1.0) == pytest.approx(W)
    assert unav(0.0) == pytest.approx(-L)
    assert unav(tp) == pytest.approx(-1.0)
    assert unav(1.0) == pytest.approx(W)


def test_money_live_is_interp_through_knots():
    """money_live = np.interp through (0,-L), (TP,-1), (CP,+1), (1,W)
    (CENTERED; Jacoby: end values -1/+1), (0,-L), (CP,+1), (1,W) (MINE),
    (0,-L), (TP,-1), (1,W) (THEIRS), on a dense p grid, vectorized over
    (W, L)."""
    p = np.linspace(0.0, 1.0, 2001)
    wl = np.array([1.0, 1.05, 1.2, 1.5, 2.0, 2.5, 3.0])
    Wg, Lg, P = np.meshgrid(wl, wl, p, indexing="ij")
    worst = 0.0
    for persp in CubePerspective:
        for jac in (False, True):
            got = money_live(P, Wg, Lg, persp, jac)
            for a, W in enumerate(wl):
                for b, L in enumerate(wl):
                    tp = (L - 0.5) / (W + L + 0.5)
                    cp = (L + 1.0) / (W + L + 0.5)
                    if persp == CubePerspective.CENTERED:
                        lo, hi = (-1.0, 1.0) if jac else (-L, W)
                        knots = [(0, lo), (tp, -1), (cp, 1), (1, hi)]
                    elif persp == CubePerspective.MINE:
                        knots = [(0, -L), (cp, 1), (1, W)]
                    else:
                        knots = [(0, -L), (tp, -1), (1, W)]
                    want = np.interp(p, *zip(*knots))
                    worst = max(worst, float(np.abs(got[a, b] - want).max()))
    assert worst <= 1e-12, worst


# ── Full conversion ──────────────────────────────────────────────────


def test_dead_limit_equals_cubeless():
    probs = np.array([0.6, 0.2, 0.05, 0.1, 0.02])
    for persp in CubePerspective:
        eq = janowski_equity(probs, persp, 0.0, jacoby=False)
        assert float(eq) == pytest.approx(float(cubeless_equity(probs)))
    # Jacoby + centered at x=0: gammons collapse
    eq = janowski_equity(probs, CubePerspective.CENTERED, 0.0, jacoby=True)
    assert float(eq) == pytest.approx(0.2)   # 2p - 1


def test_hand_derived_fixture():
    # probs (.6,.2,.05,.1,.02): W=1.41667, L=1.3, p=.6, cubeless E=.33
    # THEIRS, x=0.68: TP_live=(1.3-.5)/3.21667=.24870;
    # live=-1+(W+1)(p-TP)/(1-TP)=-1+2.41667*.35130/.75130=.13005
    # eq=.32*.33+.68*.13005=.19404
    probs = np.array([0.6, 0.2, 0.05, 0.1, 0.02])
    eq = janowski_equity(probs, CubePerspective.THEIRS, 0.68, jacoby=True)
    assert float(eq) == pytest.approx(0.19404, abs=2e-4)


def test_degenerate_positions_fall_back_to_dead():
    won = np.array([1.0, 0.4, 0.1, 0.0, 0.0])
    eq = janowski_equity(won, CubePerspective.MINE, 0.7, jacoby=False)
    assert float(eq) == pytest.approx(float(cubeless_equity(won)))


def test_batch_and_per_row_x():
    probs = np.tile(np.array([0.55, 0.15, 0.02, 0.12, 0.01]), (3, 1))
    xs = np.array([0.60, 0.68, 0.76])
    eq = janowski_equity(probs, CubePerspective.CENTERED, xs, jacoby=True)
    assert eq.shape == (3,)
    # Winning side (p=.55): livelier cube -> more equity from ownership
    assert eq[0] < eq[1] < eq[2]


def test_cash_point_and_fold():
    # W=L=1, x=2/3: CP=(1+.5+1/3)/(2+1/3)=.7857
    assert cash_point(1, 1, 2.0 / 3.0) == pytest.approx(11.0 / 14.0, abs=1e-4)
    # GetECF3 fold: double only when both DT and DP beat ND
    assert fold_cube_decision(nd=0.4, dt=0.7) == pytest.approx(0.7)   # take
    assert fold_cube_decision(nd=0.4, dt=1.5) == pytest.approx(1.0)   # pass
    assert fold_cube_decision(nd=0.8, dt=0.7) == pytest.approx(0.8)   # no double
    assert fold_cube_decision(nd=1.2, dt=1.5) == pytest.approx(1.2)   # too good


_PROBS = np.array([
    [0.55, 0.15, 0.02, 0.12, 0.01],
    [0.30, 0.05, 0.00, 0.20, 0.03],
    [0.80, 0.30, 0.05, 0.02, 0.00],
    [0.50, 0.00, 0.00, 0.00, 0.00],
])


def _with_p(probs, p):
    """probs with P(win) replaced by p, the gammon rates per win/loss
    kept (W and L fixed)."""
    W, L, p0 = w_l(probs)
    out = np.empty((len(p), 5))
    out[:, 0] = p
    out[:, 1:3] = np.outer(p, probs[1:3] / p0)
    out[:, 3:5] = np.outer(1.0 - p, probs[3:5] / (1.0 - p0))
    return out


def test_default_x():
    assert DEFAULT_X == 0.76


@pytest.mark.parametrize("jacoby", [False, True])
def test_equity_monotone_in_p(jacoby):
    p = np.linspace(0.001, 0.999, 999)
    for base in _PROBS[:3]:
        probs = _with_p(base, p)
        for persp in CubePerspective:
            for f in (janowski_equity, janowski_equity_folded):
                eq = f(probs, persp, DEFAULT_X, jacoby)
                assert np.all(np.diff(eq) >= -1e-12), (persp, f.__name__)


@pytest.mark.parametrize("jacoby", [False, True])
def test_equity_between_dead_and_live(jacoby):
    p = np.linspace(0.001, 0.999, 999)
    for base in _PROBS[:3]:
        probs = _with_p(base, p)
        W, L, pp = w_l(probs)
        for persp in CubePerspective:
            jc = jacoby and persp == CubePerspective.CENTERED
            dead = cubeless_equity(probs, jacoby_centered=jc)
            live = money_live(pp, W, L, persp, jacoby)
            assert np.allclose(janowski_equity(probs, persp, 0.0, jacoby), dead,
                               atol=1e-12)
            assert np.allclose(janowski_equity(probs, persp, 1.0, jacoby), live,
                               atol=1e-12)
            eq = janowski_equity(probs, persp, DEFAULT_X, jacoby)
            lo, hi = np.minimum(dead, live), np.maximum(dead, live)
            assert np.all((eq >= lo - 1e-12) & (eq <= hi + 1e-12))


def test_take_and_cash_points_match_the_fold():
    """Without Jacoby, 2 x the THEIRS equity (DT) crosses DP = 1 at the
    cash point and the take point is where the taker's DT equals -1
    (gammon rates fixed), at any x."""
    for base in _PROBS[:3]:
        W, L, _ = w_l(base)
        for x in (0.0, 0.5, DEFAULT_X, 1.0):
            cp = float(cash_point(W, L, x))
            dt = 2.0 * janowski_equity(_with_p(base, np.array([cp])),
                                       CubePerspective.THEIRS, x, False)
            assert float(dt[0]) == pytest.approx(1.0, abs=1e-9)
            # the taker owns the cube after the take: MINE at 1 - p
            tp = float(take_point(L, W, x))
            flip = np.array([base[0], base[3], base[4], base[1], base[2]])
            flip[0] = 1.0 - base[0]
            taker = 2.0 * janowski_equity(_with_p(flip, np.array([tp])),
                                          CubePerspective.MINE, x, False)
            assert float(taker[0]) == pytest.approx(-1.0, abs=1e-9)
            # the fold: pass beyond the cash point (DT > DP), take below
            for p, takes in ((cp + 1e-3, False), (cp - 1e-3, True)):
                pr = _with_p(base, np.array([p]))
                nd = float(janowski_equity(pr, CubePerspective.CENTERED, x, False)[0])
                dtv = 2.0 * float(janowski_equity(pr, CubePerspective.THEIRS, x, False)[0])
                folded = float(janowski_equity_folded(
                    pr, CubePerspective.CENTERED, x, False)[0])
                assert folded == pytest.approx(fold_cube_decision(nd, dtv))
                assert (dtv <= 1.0) is takes


def test_money_cube_action_is_tdagent_rule():
    from td_agent import TDAgent
    act = money_cube_action
    C, M = CubePerspective.CENTERED, CubePerspective.MINE
    assert act(0.5, 0.8, C, False) == (True, True)       # double/take
    assert act(0.5, 1.4, C, False) == (True, False)      # double/pass
    assert act(0.5, 0.4, C, False) == (False, True)      # no double/take
    assert act(1.2, 1.4, M, False) == (False, False)     # too good/pass
    assert act(1.2, 0.9, M, False) == (False, True)      # too good/take
    assert act(0.5, 1.0, C, False) == (True, True)       # DT == DP takes
    assert act(0.9, 1.4, C, True) == (True, False)       # Jacoby cash
    rng = np.random.default_rng(3)
    for nd, dt in rng.uniform(-1.5, 2.5, size=(500, 2)):
        for persp in (C, M):
            for jac in (False, True):
                assert act(nd, dt, persp, jac) == (
                    TDAgent._money_cube_decision(nd, dt, persp, jac), dt <= 1.0)


# ── Agent-level tests (production checkpoints + bearoff tables) ──────

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROB5 = os.path.join(_REPO, "best_models", "cubeless_prob5_512_512_256_256.pt")
CUBEFUL = os.path.join(_REPO, "best_models", "cubeful_money_512_512_256_256.pt")
FIXEDPOOL_CUBEFUL = os.path.join(_REPO, "experiments", "benchmarks",
                                 "fixedpool_cubeful_positions.json")


@pytest.fixture(scope="module")
def agent0():
    import bearoff
    if not os.path.exists(PROB5):
        pytest.skip("prob5 checkpoint not available")
    if not (os.path.exists(bearoff.DEFAULT_TABLE)
            and os.path.exists(bearoff.DEFAULT_TS_TABLE)):
        pytest.skip("bearoff tables not built")
    from model import load_model
    from janowski_agent import JanowskiAgent
    return JanowskiAgent(load_model(PROB5))


@pytest.fixture(scope="module")
def agent1(agent0):
    from model import load_model
    from janowski_agent import JanowskiAgent
    return JanowskiAgent(load_model(PROB5), oneply=True)


def test_agent_flags(agent0):
    assert agent0.is_cubeful
    assert agent0.output_mode == "equity"
    assert agent0.x == DEFAULT_X


def test_janowski_x_env_reaches_agent(agent0, monkeypatch):
    from play_models import _build_side
    kw = dict(plies=0, sage_plies=1, cubeless_money=False, cubeful=True,
              oneply=False, twoply_k=0, bf16=False)
    monkeypatch.delenv("JANOWSKI_X", raising=False)
    assert _build_side(f"janowski:{PROB5}", "janowski", **kw).x == DEFAULT_X
    monkeypatch.setenv("JANOWSKI_X", "0.6")
    assert _build_side(f"janowski:{PROB5}", "janowski", **kw).x == 0.6


def test_0ply_decisions_on_pool(agent0):
    """0-ply smoke on committed-pool positions: legal moves, consistent
    cube actions, values = the conversion at DEFAULT_X off the bearoff
    class."""
    import json
    import bearoff
    from modes import MatchState, CubeOwner, cube_perspective
    pool = json.load(open(FIXEDPOOL_CUBEFUL))
    for r in pool["checker"][:20]:
        s = BoardState(points=r["points"], bar=r["bar"], off=r["off"], turn=r["turn"])
        ms = MatchState(cube_owner=CubeOwner(r["cube_owner"]),
                        cube_value=r["cube_value"])
        out = agent0.choose_checker_action_cubeful(s, tuple(r["dice"]), ms)
        legal = [t for _, t in get_legal_plays(s, tuple(r["dice"]))]
        assert (out is None) == (not legal)
        if out is not None:
            assert out[0] in legal
    n = 0
    for r in pool["cube"][:40]:
        s = BoardState(points=r["points"], bar=r["bar"], off=r["off"], turn=r["turn"])
        ms = MatchState(cube_owner=CubeOwner(r["cube_owner"]),
                        cube_value=r["cube_value"])
        persp = cube_perspective(ms.cube_owner, s.turn)
        o = agent0.offer_double(s, ms)
        if persp == CubePerspective.THEIRS:
            assert not o.should_double
            continue
        nd, dt = o._cache["v_no_double_0ply"], o._cache["v_double_take_0ply"]
        assert (o.should_double, agent0.respond_to_double(s, ms)) == \
            money_cube_action(nd, dt, persp, ms.jacoby)
        if not bearoff.is_bearoff_class(s):
            probs = agent0._probs_state(s)
            assert nd == pytest.approx(float(janowski_equity(
                probs, persp, DEFAULT_X, ms.jacoby)), abs=1e-12)
            assert dt == pytest.approx(2 * float(janowski_equity(
                probs, CubePerspective.THEIRS, DEFAULT_X, ms.jacoby)), abs=1e-12)
            n += 1
    assert n >= 20


def test_agent_never_doubles_from_theirs(agent0):
    from modes import MatchState, CubeOwner
    s = BoardState.initial()
    ms = MatchState(cube_owner=CubeOwner(BLACK + 1), cube_value=2)
    offer = agent0.offer_double(s, ms)          # WHITE on roll, BLACK owns
    assert offer.should_double is False


def test_agent_takes_good_position_passes_hopeless(agent0):
    from modes import MatchState
    ms = MatchState()
    # Opening position doubled by BLACK: near-50/50, easy take for WHITE.
    s = BoardState.initial()
    s.turn = BLACK
    assert agent0.respond_to_double(s, ms) is True
    # Hopeless for the responder: doubler (WHITE on roll) nearly done, big
    # race lead -> responder must pass. White 3 checkers on its 1-point
    # vs black 15 on its 6-, 7- and 8-points (indices 18, 17, 16), none off.
    pts = [0] * 24
    pts[0] = 3
    pts[16] = pts[17] = pts[18] = -5
    hopeless = BoardState(points=pts, off=[12, 0], turn=WHITE)
    assert agent0.respond_to_double(hopeless, ms) is False


def test_agent_doubles_strong_position(agent0):
    from modes import MatchState
    # Strong but not decided race lead for WHITE, centered cube.
    pts = [0] * 24
    pts[0] = pts[1] = pts[2] = pts[3] = 3
    pts[4] = 3
    pts[20] = pts[21] = pts[22] = -5
    s = BoardState(points=pts, turn=WHITE)
    ms = MatchState()
    offer = agent0.offer_double(s, ms)
    v_nd = offer._cache["v_no_double_0ply"]
    v_dt = offer._cache["v_double_take_0ply"]
    # Consistency: decision matches the shared TDAgent logic.
    from td_agent import TDAgent
    from encoding import CubePerspective as CP
    assert offer.should_double == TDAgent._money_cube_decision(
        v_nd, v_dt, CP.CENTERED, True)


def test_cubeful_checker_action_runs(agent0):
    from modes import MatchState
    s = BoardState.initial()
    out = agent0.choose_checker_action_cubeful(s, (3, 1), MatchState())
    assert out is not None
    ns, _ = out
    assert isinstance(ns, BoardState)
    assert ns.turn == BLACK


def test_oneply_value_and_fold_direction(agent1):
    from modes import MatchState
    s = BoardState.initial()
    ms = MatchState()
    v_nofold = agent1._value_oneply_cubeful(s, ms, fold=False)
    v_fold = agent1._value_oneply_cubeful(s, ms, fold=True)
    assert -3.0 <= v_nofold <= 3.0
    # GetECF3 fold can only raise the roller's value (max(ND, ...)).
    assert v_fold >= v_nofold - 1e-12


def _screen(agent, s, dice, ms):
    """0-ply opponent-view candidate scores and the successors."""
    from modes import cube_perspective
    feats, nxt = agent._get_legal_plays_encoded(s, dice)
    succ = [nxt[i] for i in range(len(nxt))]
    opp_persp = cube_perspective(ms.cube_owner, 1 - s.turn)
    eq, _ = agent._screen_scores_0ply(feats, succ, opp_persp, ms)
    return eq, succ


def test_0ply_checker_folds_opponent_cube(agent0):
    """0-ply candidates carry the opponent's cube decision; fold_0ply=False
    and cube_fold=False give the plain conversion; an opponent without
    cube access is unaffected."""
    import json
    import torch
    from model import load_model
    from janowski import janowski_equity, janowski_equity_folded
    from janowski_agent import JanowskiAgent
    from modes import MatchState, CubeOwner
    plain_agent = JanowskiAgent(load_model(PROB5), fold_0ply=False)
    nofold = JanowskiAgent(load_model(PROB5), cube_fold=False)
    assert agent0.fold_0ply and not plain_agent.fold_0ply and not nofold.fold_0ply
    pool = json.load(open(FIXEDPOOL_CUBEFUL))
    bound = 0
    for r in pool["checker"][:300]:
        if r["off"] != [0, 0]:
            continue                      # keep the bearoff routing out
        s = BoardState(points=r["points"], bar=r["bar"], off=r["off"], turn=r["turn"])
        dice = tuple(r["dice"])
        for owner in (CubeOwner.CENTERED, CubeOwner(2 - s.turn), CubeOwner(1 + s.turn)):
            ms = MatchState(cube_owner=owner, cube_value=1 if owner == CubeOwner.CENTERED else 2)
            folded, succ = _screen(agent0, s, dice, ms)
            if not succ:
                break
            plain, _ = _screen(plain_agent, s, dice, ms)
            assert np.array_equal(plain, _screen(nofold, s, dice, ms)[0])
            if owner == CubeOwner(1 + s.turn):        # mover owns: no fold
                assert np.array_equal(folded, plain)
                continue
            # the fold is max(ND, min(DT, DP)): never below ND
            assert np.all(folded >= plain - 1e-12)
            from modes import cube_perspective
            persp = cube_perspective(owner, 1 - s.turn)
            feats, _ = agent0._get_legal_plays_encoded(s, dice)
            probs = agent0._probs_np(torch.from_numpy(feats))
            ref = janowski_equity_folded(probs, persp, DEFAULT_X, True)
            live = np.array([not t.is_game_over() for t in succ])
            assert np.allclose(folded[live], ref[live], atol=1e-12)
            assert np.allclose(plain[live], janowski_equity(probs, persp, DEFAULT_X, True)[live],
                               atol=1e-12)
            bound += int(np.any(folded > plain + 1e-6))
    assert bound >= 10, bound


def test_oneply_checker_action_runs(agent1):
    from modes import MatchState
    s = BoardState.initial()
    out = agent1.choose_checker_action_cubeful(s, (6, 5), MatchState())
    assert out is not None
    assert out[0].turn == BLACK


def test_full_cubeful_games_selfplay(agent0):
    """End-to-end: the cubeful game loop completes with cube actions."""
    import random
    from play_models import play_game
    random.seed(1234)
    stakes = []
    doubled = 0
    for _ in range(12):
        winner, stake, rec = play_game(
            agent0, agent0, record=True, cubeful=True, jacoby=True)
        assert winner in (WHITE, BLACK)
        assert stake >= 1
        stakes.append(stake)
        if rec.cube_value > 1 or rec.ended_by_drop:
            doubled += 1
    # A Janowski player must actually use the cube in self-play.
    assert doubled > 0, f"no cube action in 12 games (stakes={stakes})"


def test_cubeless_checker_ablation(agent0):
    from model import load_model
    from janowski_agent import JanowskiAgent
    from modes import MatchState
    abl = JanowskiAgent(load_model(PROB5), cubeless_checker=True)
    s = BoardState.initial()
    out = abl.choose_checker_action_cubeful(s, (3, 1), MatchState())
    assert out is not None and out[0].turn == BLACK
    # Cube decisions unchanged by the flag (same static ND/DT):
    assert (abl.offer_double(s, MatchState()).should_double
            == agent0.offer_double(s, MatchState()).should_double)
    with pytest.raises(ValueError):
        JanowskiAgent(load_model(PROB5), cubeless_checker=True, oneply=True)


def test_bearoff_routing(agent0):
    """Bearoff-class cube decisions route to exact values, not net+x."""
    import bearoff
    from modes import MatchState
    # gammon-free two-sided bearoff: 4 checkers each, several off
    pts = [0] * 24
    pts[0] = pts[1] = 2
    pts[22] = pts[23] = -2
    s = BoardState(points=pts, off=[11, 11], turn=WHITE)
    assert bearoff.is_bearoff_class(s)
    ms = MatchState()
    nd_dt = bearoff.race_cube_values(s, CubePerspective.CENTERED)
    assert nd_dt is not None
    offer = agent0.offer_double(s, ms)
    assert offer._cache["v_no_double_0ply"] == pytest.approx(nd_dt[0], abs=1e-9)
    assert offer._cache["v_double_take_0ply"] == pytest.approx(nd_dt[1], abs=1e-9)
    # ablation flag really disables the routing
    from model import load_model
    from janowski_agent import JanowskiAgent
    off_agent = JanowskiAgent(load_model(PROB5), bearoff_exact=False)
    offer2 = off_agent.offer_double(s, ms)
    assert offer2._cache["v_no_double_0ply"] != pytest.approx(nd_dt[0], abs=1e-6)


def test_bearoff_checker_routing_runs(agent0):
    from modes import MatchState
    pts = [0] * 24
    pts[0] = pts[2] = pts[4] = 2
    pts[19] = pts[21] = pts[23] = -2
    s = BoardState(points=pts, off=[9, 9], turn=WHITE)
    out = agent0.choose_checker_action_cubeful(s, (5, 2), MatchState())
    assert out is not None
    assert out[0].turn == BLACK


def test_bearoff_td_agent(agent0):
    """4L + bearoff routing: exact in class, stock TDAgent elsewhere."""
    import bearoff
    from encoding import CubePerspective as CP
    from modes import MatchState
    from model import load_model
    from td_agent import TDAgent
    from bearoff_td import BearoffTDAgent
    if not os.path.exists(CUBEFUL):
        pytest.skip("cubeful 4L unavailable")
    bo = BearoffTDAgent(load_model(CUBEFUL))
    plain = TDAgent(load_model(CUBEFUL))
    ms = MatchState()
    # bearoff-class: exact values in the cache
    pts = [0] * 24
    pts[0] = pts[1] = 2
    pts[22] = pts[23] = -2
    s = BoardState(points=pts, off=[11, 11], turn=WHITE)
    nd_dt = bearoff.race_cube_values(s, CP.CENTERED)
    offer = bo.offer_double(s, ms)
    assert offer._cache["v_no_double_0ply"] == pytest.approx(nd_dt[0])
    assert offer._cache["v_double_take_0ply"] == pytest.approx(nd_dt[1])
    # contact: identical decision to the stock agent
    s0 = BoardState.initial()
    assert (bo.offer_double(s0, ms).should_double
            == plain.offer_double(s0, ms).should_double)
    out = bo.choose_checker_action_cubeful(s0, (3, 1), ms)
    out_plain = plain.choose_checker_action_cubeful(s0, (3, 1), ms)
    assert out[0] == out_plain[0]
    # a full game JanowskiAgent vs BearoffTDAgent completes
    import random
    from play_models import play_game
    random.seed(5)
    winner, stake = play_game(agent0, bo, cubeful=True, jacoby=True)
    assert winner in (WHITE, BLACK) and stake >= 1


def test_bearoff_td_agent_oneply_race_root(agent0):
    """1-ply BearoffTDAgent on a race root with exact and net candidates."""
    from modes import MatchState
    from model import load_model
    from bearoff_td import BearoffTDAgent
    if not os.path.exists(CUBEFUL):
        pytest.skip("cubeful 4L unavailable")
    bo1 = BearoffTDAgent(load_model(CUBEFUL), oneply=True)
    pts = [0] * 24
    pts[0] = pts[2] = pts[4] = pts[7] = 2
    pts[19] = pts[21] = pts[23] = -2
    s = BoardState(points=pts, off=[7, 9], turn=WHITE)
    assert is_race(s)
    out = bo1.choose_checker_action_cubeful(s, (5, 2), MatchState())
    assert out is not None and out[0].turn == BLACK


def test_side_specs_build_both_agents(agent0):
    from play_models import _build_side, _detect_side_type
    from janowski_agent import JanowskiAgent
    from bearoff_td import BearoffTDAgent
    if not os.path.exists(CUBEFUL):
        pytest.skip("cubeful 4L unavailable")
    kw = dict(plies=0, sage_plies=1, cubeless_money=False, cubeful=True,
              oneply=False, twoply_k=0, bf16=False)
    j = "janowski:" + PROB5
    b = "bo:" + CUBEFUL
    assert _detect_side_type(j) == "janowski"
    assert _detect_side_type(b) == "bo"
    assert isinstance(_build_side(j, "janowski", **kw), JanowskiAgent)
    assert isinstance(_build_side(b, "bo", **kw), BearoffTDAgent)
    with pytest.raises(ValueError):
        _build_side(j, "janowski", **{**kw, "cubeful": False,
                                      "cubeless_money": True})
    with pytest.raises(ValueError):
        _build_side(b, "bo", **{**kw, "cubeful": False,
                                "cubeless_money": True})


def test_full_game_vs_cubeful_4l(agent0):
    import random
    if not os.path.exists(CUBEFUL):
        pytest.skip("cubeful 4L checkpoint not available")
    from model import load_model
    from td_agent import TDAgent
    from play_models import play_game
    opp = TDAgent(load_model(CUBEFUL))
    random.seed(7)
    for _ in range(3):
        winner, stake = play_game(agent0, opp, cubeful=True, jacoby=True)
        assert winner in (WHITE, BLACK) and stake >= 1


def test_gammonish_bearoff_uses_exact_probs(agent0):
    """Gammons possible in a bearoff-class position: exact probabilities
    through the conversion at the agent's x, for the cube and for
    checker-play successors."""
    import bearoff
    from modes import MatchState, CubeOwner, cube_perspective
    pts = [0] * 24
    pts[5] = 12                      # WHITE: 12 on its 6-point, 3 off
    pts[23 - 5] = -15                # BLACK: 15 on its 6-point, none off
    s = BoardState(points=pts, off=[3, 0], turn=WHITE)
    assert bearoff.is_bearoff_class(s) and bearoff.race_cube_values(
        s, CubePerspective.CENTERED) is None
    probs = bearoff.exact_probs5(s)
    for owner in (CubeOwner.CENTERED, CubeOwner(WHITE + 1)):
        ms = MatchState(cube_owner=owner, cube_value=1 if owner == 0 else 2,
                        jacoby=False)
        p = cube_perspective(owner, WHITE)
        nd, dt = agent0._static_nd_dt(s, ms)
        assert nd == pytest.approx(float(janowski_equity(
            probs, p, DEFAULT_X, False)), abs=1e-12)
        assert dt == pytest.approx(2 * float(janowski_equity(
            probs, CubePerspective.THEIRS, DEFAULT_X, False)), abs=1e-12)
    # BLACK to move 2-1: every successor is gammonish bearoff class
    t = BoardState(points=pts, off=[3, 0], turn=BLACK)
    ms = MatchState(jacoby=False)
    feats, nxt = agent0._get_legal_plays_encoded(t, (2, 1))
    succ = [nxt[i] for i in range(len(nxt))]
    eq, _ = agent0._screen_scores_0ply(feats, succ, CubePerspective.CENTERED, ms)
    for e, ns in zip(eq, succ):
        assert e == pytest.approx(float(janowski_equity_folded(
            bearoff.exact_probs5(ns), CubePerspective.CENTERED,
            DEFAULT_X, False)), abs=1e-12)


def test_0ply_exact_tie_goes_to_better_cubeless(agent0):
    """Cubeful tie (Jacoby, centered cube, certain win: every candidate is
    worth +1 to the mover): the candidate with the better cubeless money
    equity wins, here the gammon."""
    from model import load_model
    from janowski_agent import JanowskiAgent
    from modes import MatchState

    class Fixed(JanowskiAgent):
        def _probs_np(self, t):
            n = t.shape[0]
            p = np.zeros((n, 5))           # opponent loses for sure ...
            p[-1, 3] = 0.5                 # ... the last one half gammons
            return p

    a = Fixed(load_model(PROB5))
    s = BoardState.initial()
    feats, nxt = a._get_legal_plays_encoded(s, (3, 1))
    last = nxt[len(nxt) - 1]
    assert len(nxt) > 2
    out, _ = a.choose_checker_action_cubeful(s, (3, 1), MatchState())
    assert out == last


def test_certain_win_passes(agent0):
    """Roller owns the cube, wins for sure next roll, no gammons: ND = 1
    and DT = 2 (to rounding), the opponent passes; redouble or not, the
    roller wins the cube."""
    from modes import MatchState, CubeOwner
    pts = [0] * 24
    pts[0] = 1                         # WHITE: last checker on its ace
    pts[18] = -3                       # BLACK: 3 on its 6-point
    s = BoardState(points=pts, off=[14, 12], turn=WHITE)
    ms = MatchState(cube_owner=CubeOwner(WHITE + 1), cube_value=2)
    offer = agent0.offer_double(s, ms)
    assert offer._cache["v_no_double_0ply"] == pytest.approx(1.0, abs=1e-12)
    assert offer._cache["v_double_take_0ply"] == pytest.approx(2.0, abs=1e-12)
    assert not agent0.respond_to_double(s, ms)


def _oneply_reference(agent, S, ms, raw=False, shared=False):
    """1-ply written out per leaf over BoardStates: S.turn on roll,
    S.turn's view; per roll and root cube branch the reply is that
    branch's best leaf (ties to the better cubeless equity; `shared`:
    branch 0's reply for every branch), leaves fold the leaf player's own
    cube decision at agent.x, and the roller's decision is folded on top
    (`raw`: the per-branch values unfolded)."""
    from backgammon_engine import get_legal_plays, switch_turn
    from modes import cube_perspective
    from prob_agent import _DICE_OUTCOMES
    roller = S.turn
    leaf = 1 - roller
    branches = [ms] + ([ms.after_take(roller)] if ms.can_offer(roller) else [])
    acc = np.zeros(len(branches))
    for d1, d2 in _DICE_OUTCOMES:
        w = 1.0 if d1 == d2 else 2.0
        ls = [s for _, s in get_legal_plays(S, (d1, d2))] or [switch_turn(S)]
        pr = agent._probs_np(agent._encode_batch(ls))
        sec = cubeless_equity(pr)
        vals = [np.asarray(janowski_equity_folded(
            pr, cube_perspective(br.cube_owner, leaf), agent.x,
            br.jacoby)).reshape(-1) for br in branches]
        for b, v in enumerate(vals):
            j = int(np.lexsort((sec, vals[0] if shared else v))[0])
            acc[b] += w * v[j]
    v = -acc / 36.0
    if raw:
        return v
    if len(v) == 1:
        return v[0]
    return fold_cube_decision(v[0], 2.0 * v[1])


DICE = [(d1, d2) for d1 in range(1, 7) for d2 in range(d1, 7)]


def test_oneply_structure(agent1):
    """1-ply value: leaves fold the leaf player's cube decision at the
    agent's x, each root cube branch takes its own best reply per roll
    (as TDAgent's cubeful 1-ply), the roller's decision folded on top.
    At least one position has branch replies that differ."""
    import json
    import bearoff
    from modes import MatchState, CubeOwner
    pool = json.load(open(FIXEDPOOL_CUBEFUL))
    picked = {"contact": 0, "race": 0}
    differ = 0
    for r in pool["checker"]:
        s = BoardState(points=r["points"], bar=r["bar"], off=r["off"], turn=r["turn"])
        kind = "race" if is_race(s) else "contact"
        if picked[kind] == 3 or bearoff.may_reach_bearoff(s, 1) \
                or any(t.is_game_over() for d in DICE for _, t in get_legal_plays(s, d)):
            continue
        # the leaf player (the roller's opponent) is near its doubling window
        nd, dt = agent1._static_nd_dt(switch_turn(s), MatchState())
        if not (0.85 * nd <= dt <= 1.0):
            continue
        for owner in (CubeOwner.CENTERED, CubeOwner(1 + s.turn), CubeOwner(2 - s.turn)):
            ms = MatchState(cube_owner=owner, cube_value=1 if owner == 0 else 2)
            got = agent1._value_oneply_cubeful(s, ms, fold=True)
            assert got == pytest.approx(_oneply_reference(agent1, s, ms), abs=1e-6)
            if ms.can_offer(s.turn):
                br = [ms, ms.after_take(s.turn)]
                own = _oneply_reference(agent1, s, ms, raw=True)
                assert agent1._oneply(s, br)[0] == pytest.approx(own, abs=1e-6)
                shared = _oneply_reference(agent1, s, ms, raw=True, shared=True)
                assert own[0] == shared[0]
                assert own[1] >= shared[1] - 1e-12
                differ += int(own[1] > shared[1] + 1e-5)
        picked[kind] += 1
        if picked == {"contact": 3, "race": 3}:
            break
    assert picked == {"contact": 3, "race": 3}
    assert differ >= 1


def test_oneply_expands_every_candidate(agent1):
    """1-ply checker play scores every non-exact candidate at 1-ply (no
    move filter)."""
    from modes import MatchState
    s = BoardState.initial()
    ms = MatchState()
    seen = []
    orig = agent1._value_oneply

    def spy(state, match_state, fold):
        seen.append(state)
        return orig(state, match_state, fold)

    agent1._value_oneply = spy
    wide = 0
    try:
        for dice in [(3, 1), (6, 5), (2, 1)]:
            seen.clear()
            feats, nxt = agent1._get_legal_plays_encoded(s, dice)
            succ = [nxt[i] for i in range(len(nxt))]
            agent1.choose_checker_action_cubeful(s, dice, ms)
            assert seen == succ
            wide += len(succ) > 8
    finally:
        del agent1._value_oneply
    assert wide >= 2
