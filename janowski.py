"""janowski.py -- Janowski cube formulas over nested 5-outcome probabilities.

Formulas: R. Janowski, "Take-Points in Money Games" (1993). Reference
implementation: GNU Backgammon 1.08 (eval.c). Independent implementation;
no gnubg source is included.

Converts prob5's nested P(win), P(wg), P(wbg), P(lg), P(lbg) (on-roll
view) into cubeful money equity at a cube perspective:

    E_cubeful = (1 - x) * E_dead + x * E_live

E_dead is the cubeless money equity (Jacoby-aware on a centered cube),
E_live Janowski's live-cube equity, piecewise-linear in p with knots at
the live take and cash points, x the cube-life index (DEFAULT_X).

Every public function is vectorized: `probs` is (..., 5); `x` is a
scalar or an array broadcastable to the leading shape.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from backgammon_engine import BoardState, pip_count, is_race
from encoding import CubePerspective

DEFAULT_X = 0.76

_EPS = 1e-7            # beyond this the position is decided
ArrayLike = Union[float, np.ndarray]


# ── Cube-life index policy (match play: janowski_match.py's p5jan) ───
#
# gnubg EvalEfficiency constants (eval.c:365). gnubg indexes these by
# search depth (i = ply>=2 ? 0 : 1); as a 0/1-ply player we default to
# the shallow set and keep the deep set available for reference.
CONTACT_X = {"shallow": 0.76, "deep": 0.68}
RACE_COEF = {"shallow": 0.60, "deep": 0.55}
RACE_FACTOR = {"shallow": 0.0025, "deep": 0.00125}
RACE_MIN = {"shallow": 0.60, "deep": 0.60}
RACE_MAX = {"shallow": 0.80, "deep": 0.70}


def cube_x(state: BoardState, depth: str = "shallow") -> float:
    """gnubg-style position-class cube-life index for `state`.

    Contact/crashed -> constant; race -> linear in the on-roll player's
    pip count, clamped. (gnubg uses the roller's pips; Sage's avg-pips
    variant only matters at odd-ply lookahead depths we don't use.)
    """
    if not is_race(state):
        return CONTACT_X[depth]
    pips = pip_count(state)[state.turn]
    x = RACE_COEF[depth] + RACE_FACTOR[depth] * pips
    return float(min(RACE_MAX[depth], max(RACE_MIN[depth], x)))


def cube_x_batch(states: Sequence[BoardState], depth: str = "shallow") -> np.ndarray:
    return np.asarray([cube_x(s, depth) for s in states], dtype=np.float64)


def w_l(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(W, L, p): average win/loss values and win probability.

    W = 1 + (P(wg)+P(wbg))/P(w), L likewise on the loss side. Where p is
    within _EPS of 0 or 1 the ratios are meaningless: W and L are set to
    1 and callers fall back to the dead equity through `degenerate`.
    """
    probs = np.asarray(probs, dtype=np.float64)
    p = probs[..., 0]
    win = np.maximum(p, _EPS)
    lose = np.maximum(1.0 - p, _EPS)
    W = 1.0 + (probs[..., 1] + probs[..., 2]) / win
    L = 1.0 + (probs[..., 3] + probs[..., 4]) / lose
    degenerate = (p <= _EPS) | (p >= 1.0 - _EPS)
    W = np.where(degenerate, 1.0, W)
    L = np.where(degenerate, 1.0, L)
    return W, L, p


def cubeless_equity(probs: np.ndarray, jacoby_centered: bool = False) -> np.ndarray:
    """Cubeless money equity. `jacoby_centered`: gammons do not count."""
    probs = np.asarray(probs, dtype=np.float64)
    if jacoby_centered:
        return 2.0 * probs[..., 0] - 1.0
    return (2.0 * probs[..., 0] + probs[..., 1] + probs[..., 2]
            - probs[..., 3] - probs[..., 4]) - 1.0


def take_point(W: ArrayLike, L: ArrayLike, x: ArrayLike) -> ArrayLike:
    """Janowski eq. (3): minimum cubeless p for a correct take."""
    return (np.asarray(L) - 0.5) / (np.asarray(W) + np.asarray(L) + 0.5 * np.asarray(x))


def cash_point(W: ArrayLike, L: ArrayLike, x: ArrayLike) -> ArrayLike:
    """General-model cash point: (L + 0.5 + 0.5x) / (W + L + 0.5x)."""
    return (np.asarray(L) + 0.5 + 0.5 * np.asarray(x)) / (
        np.asarray(W) + np.asarray(L) + 0.5 * np.asarray(x))


def _piecewise_linear(p: np.ndarray, knots) -> np.ndarray:
    """np.interp through `knots` [(x, y), ...] (increasing x, first 0,
    last 1), every coordinate an array broadcastable with p."""
    segs = list(zip(knots[:-1], knots[1:]))
    (x0, y0), (x1, y1) = segs[-1]
    out = y0 + (y1 - y0) * (p - x0) / (x1 - x0)
    for (x0, y0), (x1, y1) in reversed(segs[:-1]):
        out = np.where(p < x1, y0 + (y1 - y0) * (p - x0) / (x1 - x0), out)
    return out


def money_live(
    p: np.ndarray, W: np.ndarray, L: np.ndarray,
    persp: CubePerspective, jacoby: bool,
) -> np.ndarray:
    """Live-cube (x = 1) money equity, piecewise-linear in p through the
    knots (0, -L), (TP, -1), (CP, +1), (1, W) with TP = (L-0.5)/(W+L+0.5)
    and CP = (L+1)/(W+L+0.5). CENTERED uses all four (Jacoby: end values
    -1 and +1), MINE drops TP, THEIRS drops CP."""
    p = np.asarray(p, dtype=np.float64)
    denom = W + L + 0.5
    tp = (L - 0.5) / denom
    cp = (L + 1.0) / denom
    if persp == CubePerspective.CENTERED:
        lo, hi = (-1.0, 1.0) if jacoby else (-L, W)
        return _piecewise_linear(p, [(0.0, lo), (tp, -1.0), (cp, 1.0), (1.0, hi)])
    if persp == CubePerspective.MINE:
        return _piecewise_linear(p, [(0.0, -L), (cp, 1.0), (1.0, W)])
    return _piecewise_linear(p, [(0.0, -L), (tp, -1.0), (1.0, W)])


def janowski_equity(
    probs: np.ndarray,
    persp: CubePerspective,
    x: ArrayLike,
    jacoby: bool,
) -> np.ndarray:
    """Cubeful money equity per unit cube, on-roll player's view:
    (1-x)*dead + x*live.

    `persp` is the on-roll player's cube perspective. `jacoby` only acts
    while the cube is centered. Decided positions (p within _EPS of 0/1)
    return the dead equity.
    """
    probs = np.asarray(probs, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    W, L, p = w_l(probs)
    jacoby_centered = jacoby and persp == CubePerspective.CENTERED
    dead = cubeless_equity(probs, jacoby_centered=jacoby_centered)
    live = money_live(p, W, L, persp, jacoby)
    eq = (1.0 - x) * dead + x * live
    degenerate = (p <= _EPS) | (p >= 1.0 - _EPS)
    return np.where(degenerate, dead, eq)


def money_cube_action(nd: float, dt: float, persp: CubePerspective,
                      jacoby: bool) -> Tuple[bool, bool]:
    """(double, take): the repo's money cube rule, roller's view per unit
    cube (persp = the roller's cube perspective, DP = 1). Double per
    TDAgent._money_cube_decision; take iff DT <= DP."""
    from td_agent import TDAgent
    return (bool(TDAgent._money_cube_decision(nd, dt, persp, jacoby)),
            bool(dt <= 1.0))


def fold_cube_decision(nd: ArrayLike, dt: ArrayLike, dp: ArrayLike = 1.0) -> ArrayLike:
    """Pre-roll value with the roller's cube decision folded in: the
    roller doubles iff both DT and DP beat ND, and the opponent then
    picks min(DT, DP). Roller's view. Scalars in, float out; arrays
    elementwise."""
    nd, dt, dp = np.asarray(nd, np.float64), np.asarray(dt, np.float64), np.asarray(dp, np.float64)
    out = np.where((dt >= nd) & (dp >= nd), np.minimum(dt, dp), nd)
    return float(out) if out.ndim == 0 else out


def janowski_equity_folded(
    probs: np.ndarray,
    persp: CubePerspective,
    x: ArrayLike,
    jacoby: bool,
) -> np.ndarray:
    """Pre-roll cubeful equity with the on-roll player's own cube decision
    folded in (fold_cube_decision) when the cube is centered or owned by
    the on-roll player. Doubled branch: same probs and x, cube owned by the opponent, times
    2; DP = 1. Beavers do not enter."""
    nd = janowski_equity(probs, persp, x, jacoby)
    if persp == CubePerspective.THEIRS:
        return nd
    dt = 2.0 * janowski_equity(probs, CubePerspective.THEIRS, x, jacoby)
    return np.asarray(fold_cube_decision(nd, dt))
