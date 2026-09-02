"""janowski.py -- Janowski cube formulas over nested 5-outcome probabilities.

Converts a cubeless outcome distribution (our prob5 convention: nested
P(win), P(wg), P(wbg), P(lg), P(lbg), all from the on-roll player's view)
into cubeful money equity at a given cube perspective, following the
de-facto engine standard (gnubg `Cl2CfMoney`/`MoneyLive`, eval.c):

    E_cubeful = (1 - x) * E_dead  +  x * E_live

where E_dead is the cubeless money equity (Jacoby-aware when the cube is
centered) and E_live is Janowski's live-cube equity, piecewise-linear in p
with kinks at the live take/cash points. `x` is the cube-life index.

Reference: R. Janowski, "Take-Points in Money Games" (1993),
bkgm.com/articles/Janowski/cubeformulae.pdf.

All public functions are vectorized: `probs` is (..., 5); `x` may be a
scalar or an array broadcastable to the leading shape.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from backgammon_engine import BoardState, pip_count, is_race
from encoding import CubePerspective

_EPS = 1e-7            # gnubg's epsilon: beyond this, treat as decided
ArrayLike = Union[float, np.ndarray]

# ── Cube-life index policy ───────────────────────────────────────────
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


# ── (W, L, p) from nested probs ──────────────────────────────────────


def w_l(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average win/loss values (W, L) and win prob p from nested probs.

    W = 1 + (P(wg)+P(wbg))/P(w), L likewise on the loss side — exact for
    nested events since each nesting level adds one point. Guarded like
    gnubg: where p is within _EPS of 0 or 1 the ratios are meaningless
    (the position is decided); callers fall back to dead equity there via
    the `degenerate` mask, so W/L are clamped to 1 only to stay finite.
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
    """Cubeless money equity; with `jacoby_centered`, gammons don't count
    (gnubg Utility's Jacoby handling for a centered cube)."""
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


# ── Live-cube equity (piecewise, gnubg MoneyLive) ────────────────────


def money_live(
    p: np.ndarray, W: np.ndarray, L: np.ndarray,
    persp: CubePerspective, jacoby: bool,
) -> np.ndarray:
    """Live-cube (x=1) money equity, piecewise-linear in p.

    Kinks at the LIVE take/cash points TP=(L-0.5)/(W+L+0.5) and
    CP=(L+1)/(W+L+0.5). Jacoby applies only to the centered cube: the
    outer segments clamp to -1/+1 (no gammon value before the cube
    turns). Exact port of gnubg MoneyLive (eval.c:3608).
    """
    p = np.asarray(p, dtype=np.float64)
    denom = W + L + 0.5
    tp = (L - 0.5) / denom
    cp = (L + 1.0) / denom

    if persp == CubePerspective.CENTERED:
        low = -1.0 if jacoby else (-L + (-1.0 + L) * p / tp)
        mid = -1.0 + 2.0 * (p - tp) / (cp - tp)
        high = 1.0 if jacoby else (1.0 + (W - 1.0) * (p - cp) / (1.0 - cp))
        return np.where(p < tp, low, np.where(p < cp, mid, high))
    if persp == CubePerspective.MINE:
        low = -L + (1.0 + L) * p / cp
        high = 1.0 + (W - 1.0) * (p - cp) / (1.0 - cp)
        return np.where(p < cp, low, high)
    # THEIRS (cube unavailable)
    low = -L + (-1.0 + L) * p / tp
    high = -1.0 + (W + 1.0) * (p - tp) / (1.0 - tp)
    return np.where(p < tp, low, high)


# ── Cubeful equity ───────────────────────────────────────────────────


def janowski_equity(
    probs: np.ndarray,
    persp: CubePerspective,
    x: ArrayLike,
    jacoby: bool,
    variant: str = "gnubg",
) -> np.ndarray:
    """Cubeful money equity per unit cube, on-roll player's view.

    `persp` is the on-roll player's cube perspective; `jacoby` is the
    rule flag (it only has effect while the cube is centered). `x` is
    scalar or per-row. Where p is within _EPS of 0/1 the position is
    decided and the dead (cubeless) equity is returned, like gnubg.

    variant="gnubg" (default): (1-x)*dead + x*piecewise-live.
    variant="paper": Janowski's linear Appendix-1 forms
        E_O = E + 0.5xp,  E_U = E - 0.5x(1-p),  E_C = 4/(4-x)(E + 0.5x(p-0.5))
      -- ablation only; for CENTERED under Jacoby the paper's E_C does
      not apply, so we substitute the Jacoby-collapsed E.
    """
    probs = np.asarray(probs, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    W, L, p = w_l(probs)
    jacoby_centered = jacoby and persp == CubePerspective.CENTERED
    dead = cubeless_equity(probs, jacoby_centered=jacoby_centered)

    if variant == "paper":
        if persp == CubePerspective.MINE:
            eq = dead + 0.5 * x * p
        elif persp == CubePerspective.THEIRS:
            eq = dead - 0.5 * x * (1.0 - p)
        else:
            eq = 4.0 / (4.0 - x) * (dead + 0.5 * x * (p - 0.5))
    elif variant == "gnubg":
        live = money_live(p, W, L, persp, jacoby)
        eq = (1.0 - x) * dead + x * live
    else:
        raise ValueError(f"unknown variant {variant!r}")

    degenerate = (p <= _EPS) | (p >= 1.0 - _EPS)
    return np.where(degenerate, dead, eq)


def fold_cube_decision(nd: float, dt: float, dp: float = 1.0) -> float:
    """gnubg GetECF3 fold: the roller doubles iff both DT and DP beat ND;
    the opponent then picks min(DT, DP). Returns the folded equity from
    the roller's view."""
    if dt >= nd and dp >= nd:
        return min(dt, dp)
    return nd
