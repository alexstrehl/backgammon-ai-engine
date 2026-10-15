"""Janowski cube layer (janowski.py, janowski_agent.py, bearoff_td.py).

Numeric fixtures: Janowski (1993), the Senkiewicz-Ballard example and
Tables 1/2, plus hand-derived values and self-consistency checks at
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


# ── (W, L) and the paper's worked example ────────────────────────────
#
# Senkiewicz-Ballard (paper p.6): White's cubeless chances 35% (31 single,
# 4 gammon), Black 65% (47/17/1). From White's view:
#   W = 1.114, L = 1.292; TP_dead = .3292, TP_live = .2725, TP_2/3 = .2892

_SENK = np.array([0.35, 0.04, 0.0, 0.18, 0.01])


def test_paper_senkiewicz_wl():
    W, L, p = w_l(_SENK)
    assert W == pytest.approx(1.1143, abs=1e-4)
    assert L == pytest.approx(1.2923, abs=1e-4)
    assert p == pytest.approx(0.35)


def test_paper_senkiewicz_take_points():
    W, L, _ = w_l(_SENK)
    assert take_point(W, L, 0.0) == pytest.approx(0.3292, abs=1e-4)
    assert take_point(W, L, 1.0) == pytest.approx(0.2725, abs=1e-4)
    assert take_point(W, L, 2.0 / 3.0) == pytest.approx(0.2892, abs=1e-4)


def test_paper_take_point_tables():
    # Table 1a (dead): W=2, L=2 -> 37.5%; W=1, L=1 -> 25%
    assert take_point(2, 2, 0.0) == pytest.approx(0.375)
    assert take_point(1, 1, 0.0) == pytest.approx(0.25)
    # Table 1b (live): W=1, L=1 -> 20%; W=2, L=2 -> 33.3%
    assert take_point(1, 1, 1.0) == pytest.approx(0.20)
    assert take_point(2, 2, 1.0) == pytest.approx(1.5 / 4.5, abs=1e-4)
    # Table 1c (x=2/3): W=1.5, L=1.5 -> 30.0%
    assert take_point(1.5, 1.5, 2.0 / 3.0) == pytest.approx(0.30, abs=1e-3)
    # Take-equity table 2c: W=L=1, x=2/3 -> -0.571
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
FIXEDPOOL_CUBEFUL = os.path.join(_REPO, "tools", "fixedpool_cubeful_positions.json")



# The JanowskiAgent / bearoff cases of the money-game engine's
# test_janowski.py need janowski_agent.py, bearoff.py and bearoff_td.py
# (money-only modules, not in this tree); only the formula tests above are
# kept here.
