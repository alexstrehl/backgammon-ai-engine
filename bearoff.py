"""bearoff.py -- bearoff tables and race cube equities.

In bearoff-class positions the cube-life interpolation is replaced by
table values, as in gnubg:

  * one-sided table over the C(21,6) = 54,264 positions of <= 15
    checkers on points 1-6: distribution of rolls to bear all off (f,
    play minimizing expected rolls to clear) and to the first checker
    off (g, play minimizing expected rolls to the first checker off),
    gnubg_os0.bd's two policies;
  * outcome probabilities of a two-sided position by cross-side
    convolution of f and g (gammons only while the opponent has 0 off),
    exact for that policy;
  * two-sided exact table (P, V) within TS_MAX checkers per side: full
    expectimax with a double window before every roll;
  * beyond TS_MAX, gammon-free: a money cube DP on the conditional clear
    hazards. Approximate: the policy ignores the opponent and the cube,
    and the DP sees only the roll count, not each side's position.

Tables are built once per machine and cached outside the repo
(~/.cache/bg_bearoff_os6g.npz and bg_bearoff_ts8.npz; BG_BEAROFF_TABLE
and BG_BEAROFF_TS_TABLE override):

    python -c 'import bearoff; bearoff.build_table(); bearoff.build_ts_table()'

tests/test_bearoff.py checks both tiers against gnubg's databases.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

from backgammon_engine import (
    BoardState, WHITE, BLACK, get_legal_plays,
)
from encoding import CubePerspective

R_MAX = 24          # distribution length (max rolls to clear; tail asserted ~0)
N_POINTS = 6
N_CHECKERS = 15

_DICE = [(d1, d2) for d1 in range(1, 7) for d2 in range(d1, 7)]
_DICE_P = np.array([(1.0 if d1 == d2 else 2.0) / 36.0 for d1, d2 in _DICE])

DEFAULT_TABLE = os.environ.get(
    "BG_BEAROFF_TABLE",
    os.path.expanduser("~/.cache/bg_bearoff_os6g.npz"))
# load_table rejects one-sided tables without it (g under the clearing policy).
_G_POLICY = "first_off"


# ── position enumeration / indexing ──────────────────────────────────


def _all_positions():
    """Every (c1..c6) with sum <= 15, in increasing pip order."""
    out = []
    for c1 in range(16):
        for c2 in range(16 - c1):
            for c3 in range(16 - c1 - c2):
                for c4 in range(16 - c1 - c2 - c3):
                    for c5 in range(16 - c1 - c2 - c3 - c4):
                        for c6 in range(16 - c1 - c2 - c3 - c4 - c5):
                            out.append((c1, c2, c3, c4, c5, c6))
    out.sort(key=lambda c: sum((i + 1) * v for i, v in enumerate(c)))
    return out


def _counts_from_board(succ: BoardState) -> Tuple[int, ...]:
    return tuple(max(0, succ.points[i]) for i in range(N_POINTS))


# ── table build ──────────────────────────────────────────────────────


def build_table(path: str = DEFAULT_TABLE, progress: bool = False) -> dict:
    """Build and save the one-sided table. f: play minimizing expected
    rolls to clear; g: play minimizing expected rolls to the first
    checker off. Deterministic tie-break on the successor counts."""
    positions = _all_positions()
    index: Dict[Tuple[int, ...], int] = {c: i for i, c in enumerate(positions)}
    n = len(positions)
    f = np.zeros((n, R_MAX), dtype=np.float64)   # P(clear in exactly r) at [r-1]
    g = np.zeros((n, R_MAX), dtype=np.float64)   # P(first off at roll r)
    mu = np.zeros(n, dtype=np.float64)           # E[rolls to clear]
    mu_g = np.zeros(n, dtype=np.float64)         # E[rolls to first off]

    # Black parked in its own home never meets white's bearoff.
    parked = [0] * 24
    parked[23] = -15

    for i, counts in enumerate(positions):
        if sum(counts) == 0:
            continue                             # cleared: f==0 by convention
        pts = parked[:]
        for j, v in enumerate(counts):
            pts[j] = v
        state = BoardState(points=pts, turn=WHITE)
        fi = np.zeros(R_MAX)
        gi = np.zeros(R_MAX)
        mi = 1.0
        mgi = 1.0
        for (d1, d2), pd in zip(_DICE, _DICE_P):
            plays = get_legal_plays(state, (d1, d2))
            best = best_g = None
            for _, s in plays:
                sc = _counts_from_board(s)
                key = (mu[index[sc]], sc)
                if best is None or key < best[0]:
                    best = (key, sc)
                key_g = (0.0 if s.off[WHITE] else mu_g[index[sc]], sc)
                if best_g is None or key_g < best_g[0]:
                    best_g = (key_g, sc, s.off[WHITE])
            _, sc = best
            k = index[sc]
            if sum(sc) == 0:
                fi[0] += pd                       # cleared this roll
            else:
                fi[1:] += pd * f[k, :-1]
                mi += pd * mu[k]
            _, sc_g, n_off = best_g
            if n_off > 0:
                gi[0] += pd                       # first checker off now
            else:
                gi[1:] += pd * g[index[sc_g], :-1]
                mgi += pd * mu_g[index[sc_g]]
        f[i] = fi
        g[i] = gi
        mu[i] = mi
        mu_g[i] = mgi
        if progress and i % 5000 == 0:
            print(f"  {i}/{n}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        counts=np.array(positions, dtype=np.uint8),
        f=f.astype(np.float32), g=g.astype(np.float32),
        mu=mu.astype(np.float32), g_policy=_G_POLICY,
    )
    return {"path": path, "n": n}


_TABLE = None


def load_table(path: str = DEFAULT_TABLE):
    """Load and cache the one-sided table. Build it before launching
    parallel runs: every worker would otherwise build its own."""
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"bearoff table not found at {path}; build it first:\n"
            f"  python -c 'import bearoff; bearoff.build_table()'")
    d = np.load(path)
    if (str(d["g_policy"]) if "g_policy" in d else "") != _G_POLICY:
        raise ValueError(
            f"{path}: one-sided table from before the first-off policy "
            f"for g; rebuild it:\n"
            f"  python -c 'import bearoff; bearoff.build_table()'")
    counts = [tuple(int(v) for v in row) for row in d["counts"]]
    _TABLE = {
        "index": {c: i for i, c in enumerate(counts)},
        "f": d["f"].astype(np.float64),
        "g": d["g"].astype(np.float64),
        "mu": d["mu"].astype(np.float64),
    }
    return _TABLE


def dists(counts: Tuple[int, ...]):
    t = load_table()
    i = t["index"][tuple(counts)]
    return t["f"][i], t["g"][i]


# ── state classification ─────────────────────────────────────────────


def is_bearoff_class(state: BoardState) -> bool:
    """Both sides fully in their home boards, nobody on the bar."""
    if state.bar[WHITE] or state.bar[BLACK]:
        return False
    for i in range(N_POINTS, 24):
        if state.points[i] > 0:                  # white outside 0-5
            return False
    for i in range(0, 24 - N_POINTS):
        if state.points[i] < 0:                  # black outside 18-23
            return False
    return True


def home_counts(state: BoardState, player: int) -> Tuple[int, ...]:
    """Counts by distance-to-off 1..6 for `player` (bearoff-class only)."""
    if player == WHITE:
        return tuple(max(0, state.points[i]) for i in range(N_POINTS))
    return tuple(max(0, -state.points[23 - i]) for i in range(N_POINTS))


# ── exact outcome probabilities ──────────────────────────────────────


def exact_probs5(state: BoardState) -> np.ndarray:
    """Nested 5-outcome probabilities, on-roll player's view, exact
    given the table policy. No backgammons in the bearoff class."""
    me, opp = state.turn, 1 - state.turn
    f_me, g_me = dists(home_counts(state, me))
    f_opp, g_opp = dists(home_counts(state, opp))
    off_me = state.off[me]
    off_opp = state.off[opp]

    F_opp = np.cumsum(f_opp)
    G_opp = np.cumsum(g_opp)
    G_me = np.cumsum(g_me)
    # On roll: my r-th roll precedes the opponent's r-th.
    # P(win) = sum_r f_me[r] * (1 - F_opp[r-1]); index [r-1] = "exactly r".
    surv_opp = np.concatenate(([1.0], 1.0 - F_opp[:-1]))
    p = float(np.dot(f_me, surv_opp))
    if off_opp == 0:
        surv_g = np.concatenate(([1.0], 1.0 - G_opp[:-1]))
        pwg = float(np.dot(f_me, surv_g))
    else:
        pwg = 0.0
    if off_me == 0:
        plg = float(np.dot(f_opp, 1.0 - G_me))   # by the opponent's r-th I had r rolls
    else:
        plg = 0.0
    return np.array([p, pwg, 0.0, plg, 0.0])


# ── money cube DP on clear hazards (gammon-free, approximate) ───────


def _hazards(fdist: np.ndarray) -> np.ndarray:
    """h[k] = P(clear on (k+1)-th roll | not cleared in first k)."""
    F = np.cumsum(fdist)
    h = np.ones(R_MAX)
    for k in range(R_MAX):
        surv = 1.0 - (F[k - 1] if k > 0 else 0.0)
        if surv <= 1e-12:
            h[k] = 1.0
        else:
            h[k] = min(1.0, fdist[k] / surv)
    return h


_CENT = int(CubePerspective.CENTERED)
_MINE = int(CubePerspective.MINE)
_THEIRS = int(CubePerspective.THEIRS)
_FLIP = {_CENT: _CENT, _MINE: _THEIRS, _THEIRS: _MINE}


@lru_cache(maxsize=200_000)
def _race_dp(hm: tuple, ho: tuple, persp: int) -> float:
    """Cubeful money equity per unit cube, on-roll player's view.

    hm/ho: clear hazards of the on-roll player / opponent. V_x(k, j, c):
    side x on roll having survived k of its own rolls, the other side j,
    c side x's cube perspective. A double/take/pass window precedes every
    roll (gnubg GetECF3 fold). W_x plays the roll out.
    """
    memo_a: Dict[Tuple[int, int, int], float] = {}
    memo_b: Dict[Tuple[int, int, int], float] = {}

    def V_a(k: int, j: int, c: int) -> float:      # side A on roll
        key = (k, j, c)
        got = memo_a.get(key)
        if got is not None:
            return got
        nd = W_a(k, j, c)
        if c in (_CENT, _MINE):
            dt = 2.0 * W_a(k, j, _THEIRS)
            v = max(nd, min(dt, 1.0))
        else:
            v = nd
        memo_a[key] = v
        return v

    def W_a(k: int, j: int, c: int) -> float:
        h = hm[k] if k < R_MAX else 1.0
        if h >= 1.0:
            return 1.0
        return h - (1.0 - h) * V_b(j, k + 1, _FLIP[c])

    def V_b(k: int, j: int, c: int) -> float:      # side B on roll
        key = (k, j, c)
        got = memo_b.get(key)
        if got is not None:
            return got
        nd = W_b(k, j, c)
        if c in (_CENT, _MINE):
            dt = 2.0 * W_b(k, j, _THEIRS)
            v = max(nd, min(dt, 1.0))
        else:
            v = nd
        memo_b[key] = v
        return v

    def W_b(k: int, j: int, c: int) -> float:
        h = ho[k] if k < R_MAX else 1.0
        if h >= 1.0:
            return 1.0
        return h - (1.0 - h) * V_a(j, k + 1, _FLIP[c])

    # The on-roll player's own immediate double is the caller's decision.
    return W_a(0, 0, persp)


def race_hazards(state: BoardState) -> Optional[Tuple[tuple, tuple]]:
    """Clear hazards (hm, ho) of a gammon-free bearoff-class position,
    or None (not bearoff class, gammons still possible, degenerate)."""
    if not is_bearoff_class(state):
        return None
    if state.off[WHITE] == 0 or state.off[BLACK] == 0:
        return None                                # gammons possible
    me, opp = state.turn, 1 - state.turn
    f_me, _ = dists(home_counts(state, me))
    f_opp, _ = dists(home_counts(state, opp))
    if f_me.sum() < 0.5:                           # cleared/degenerate
        return None
    hm = tuple(np.round(_hazards(f_me), 12))
    ho = tuple(np.round(_hazards(f_opp), 12))
    return hm, ho


def _exact_tier(state: BoardState):
    """Routing of every exact bearoff value: ("ts", (ia, ib)) when both
    sides are within the two-sided table, ("hazard", (hm, ho)) for a
    larger gammon-free bearoff, None otherwise (not bearoff class,
    gammons possible, game over)."""
    if not is_bearoff_class(state):
        return None
    if state.off[WHITE] >= N_CHECKERS or state.off[BLACK] >= N_CHECKERS:
        return None
    ij = _ts_indices(state)
    if ij is not None:
        return "ts", ij
    r = race_hazards(state)
    return None if r is None else ("hazard", r)


def _tier_cube_values(tier, persp: CubePerspective) -> Tuple[float, float]:
    kind, arg = tier
    if kind == "ts":
        wnd = _ts_wnd(*arg)
        return float(wnd[int(persp)]), 2.0 * float(wnd[_THEIRS])
    hm, ho = arg
    return (_race_dp(hm, ho, int(persp)),
            2.0 * _race_dp(hm, ho, _THEIRS))


def race_cube_values(state: BoardState, persp: CubePerspective
                     ) -> Optional[Tuple[float, float]]:
    """(v_no_double, v_double_take) per unit cube, on-roll view: the
    exact two-sided table where both sides are within it, the hazard DP
    beyond it, None when neither applies (_exact_tier)."""
    tier = _exact_tier(state)
    return None if tier is None else _tier_cube_values(tier, persp)


def race_equity(state: BoardState, persp: CubePerspective,
                with_own_fold: bool = False) -> Optional[float]:
    """Cubeful money equity per unit cube, on-roll view, routed as
    race_cube_values. `with_own_fold` folds the on-roll player's own
    double window, max(ND, min(DT, 1)) when it may double (cube
    centered or its own): the form for a successor position (GetECF3)."""
    tier = _exact_tier(state)
    if tier is None:
        return None
    fold = with_own_fold and persp != CubePerspective.THEIRS
    if tier[0] == "ts" and (fold or persp == CubePerspective.THEIRS):
        # V is the fold of _ts_wnd by construction (build_ts_table).
        ia, ib = tier[1]
        return float(load_ts_table()["V"][ia, ib, int(persp)])
    nd, dt = _tier_cube_values(tier, persp)
    return max(nd, min(dt, 1.0)) if fold else nd


def may_reach_bearoff(state: BoardState, plies: int) -> bool:
    """Can a bearoff-class position arise within `plies` (1 or 2) moves,
    the on-roll side moving first? A move covers at most 24 pips."""
    def pips_out(player):
        n = 19 * state.bar[player]
        for i in range(24):
            v = state.points[i]
            if player == WHITE and v > 0 and i >= N_POINTS:
                n += v * (i - N_POINTS + 1)
            elif player == BLACK and v < 0 and i < 24 - N_POINTS:
                n += -v * (24 - N_POINTS - i)
        return n
    me, opp = state.turn, 1 - state.turn
    return pips_out(me) <= 24 and pips_out(opp) <= (24 if plies >= 2 else 0)


# ── two-sided exact table (gnubg's TS class) ─────────────────────────
#
# Full expectimax over position pairs: exact win probability and exact
# cubeful money values with a double window before every roll, checker
# play maximizing cubeful equity per cube branch. Beyond TS_MAX the
# hazard DP above serves.

TS_MAX = 8          # gnubg's ts db stops at 6; 8 keeps the hazard DP off
                    # every race but the long ones
DEFAULT_TS_TABLE = os.environ.get(
    "BG_BEAROFF_TS_TABLE",
    os.path.expanduser("~/.cache/bg_bearoff_ts8.npz"))


def _ts_positions(ts_max: int = TS_MAX):
    out = []
    rng = range
    for c1 in rng(ts_max + 1):
        for c2 in rng(ts_max + 1 - c1):
            for c3 in rng(ts_max + 1 - c1 - c2):
                for c4 in rng(ts_max + 1 - c1 - c2 - c3):
                    for c5 in rng(ts_max + 1 - c1 - c2 - c3 - c4):
                        for c6 in rng(ts_max + 1 - c1 - c2 - c3 - c4 - c5):
                            out.append((c1, c2, c3, c4, c5, c6))
    out.sort(key=lambda c: sum((i + 1) * v for i, v in enumerate(c)))
    return out


def build_ts_table(path: str = DEFAULT_TS_TABLE, progress: bool = False,
                   ts_max: int = TS_MAX):
    """Build the exact two-sided table. V[ia, ib, c] = cubeful money
    equity of position ia on roll against ib at cube perspective c,
    including the on-roll player's own double window. P[ia, ib] = exact
    cubeless win probability. Window-free values are derived at runtime
    (_ts_wnd)."""
    positions = _ts_positions(ts_max)
    idx = {c: i for i, c in enumerate(positions)}
    n = len(positions)
    pips = np.array([sum((i + 1) * v for i, v in enumerate(c))
                     for c in positions])

    # successor lists per (position, dice): indices into `positions`
    parked = [0] * 24
    parked[23] = -15
    succs = [[None] * len(_DICE) for _ in range(n)]
    for i, counts in enumerate(positions):
        if sum(counts) == 0:
            continue
        pts = parked[:]
        for j, v in enumerate(counts):
            pts[j] = v
        state = BoardState(points=pts, turn=WHITE)
        for di, (d1, d2) in enumerate(_DICE):
            ss = {idx[_counts_from_board(s)]
                  for _, s in get_legal_plays(state, (d1, d2))}
            succs[i][di] = np.fromiter(ss, dtype=np.int32)

    P = np.zeros((n, n))                    # exact cubeless P(on-roll wins)
    V = np.zeros((n, n, 3))                 # cubeful, with own-window fold
    zero = idx[(0, 0, 0, 0, 0, 0)]
    # per (ia, dice): does any play clear? plus the non-clearing succ set
    clears = [[False] * len(_DICE) for _ in range(n)]
    for i in range(n):
        for di in range(len(_DICE)):
            sl = succs[i][di]
            if sl is None:
                continue
            clears[i][di] = bool((sl == zero).any())
            succs[i][di] = sl[sl != zero]

    dice_p = _DICE_P
    # Pairs by total pips ascending: every dependency P/V[ib, s] is final.
    pairs = sorted(((ia, ib) for ia in range(n) for ib in range(n)
                    if ia != zero and ib != zero),
                   key=lambda ab: pips[ab[0]] + pips[ab[1]])
    for np_done, (ia, ib) in enumerate(pairs):
        acc = 0.0
        wnd = np.zeros(3)
        Vb = V[ib]
        Pb = P[ib]
        for di in range(len(_DICE)):
            sl = succs[ia][di]
            pd = dice_p[di]
            if clears[ia][di]:
                # A clearing play wins outright and dominates.
                acc += pd
                wnd += pd
                continue
            # The best successor minimizes the opponent's value.
            acc += pd * (1.0 - Pb[sl].min())
            m = Vb[sl].min(axis=0)          # per opponent-perspective minima
            wnd[_CENT] += pd * (-m[_CENT])
            wnd[_MINE] += pd * (-m[_THEIRS])    # my MINE -> opp THEIRS
            wnd[_THEIRS] += pd * (-m[_MINE])
        P[ia, ib] = acc
        # GetECF3 fold of the on-roll player's own window; DT is the same
        # for centered and owned (the cube lands with the opponent).
        dt_capped = min(2.0 * wnd[_THEIRS], 1.0)
        V[ia, ib, _CENT] = max(wnd[_CENT], dt_capped)
        V[ia, ib, _MINE] = max(wnd[_MINE], dt_capped)
        V[ia, ib, _THEIRS] = wnd[_THEIRS]
        if progress and np_done % 100000 == 0:
            print(f"  {np_done}/{len(pairs)}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        counts=np.array(positions, dtype=np.uint8),
        P=P.astype(np.float32), V=V.astype(np.float32),
        ts_max=np.int64(ts_max),
    )
    return {"path": path, "n": n, "ts_max": ts_max}


_TS = None


def load_ts_table(path: str = DEFAULT_TS_TABLE):
    global _TS
    if _TS is not None:
        return _TS
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"two-sided bearoff table not found at {path}; build it:\n"
            f"  python -c 'import bearoff; bearoff.build_ts_table()'")
    d = np.load(path)
    ts_max = int(d["ts_max"]) if "ts_max" in d else 6
    if ts_max != TS_MAX:
        raise ValueError(
            f"{path}: two-sided table for {ts_max} checkers, expected "
            f"{TS_MAX}; rebuild it:\n"
            f"  python -c 'import bearoff; bearoff.build_ts_table()'")
    counts = [tuple(int(v) for v in row) for row in d["counts"]]
    _TS = {"index": {c: i for i, c in enumerate(counts)},
           "counts": counts,
           "P": d["P"].astype(np.float64),
           "V": d["V"].astype(np.float64),
           "ts_max": ts_max}
    return _TS


def preflight():
    """Load both tables before any game (FileNotFoundError / ValueError
    with the build command if missing or stale)."""
    load_table(DEFAULT_TABLE)
    load_ts_table(DEFAULT_TS_TABLE)


# ── flat file for the C API (c_api/puretd_janowski.c) ─────────────

DEFAULT_C_TABLE = os.environ.get(
    "BG_BEAROFF_C_TABLE",
    os.path.expanduser("~/.cache/bg_bearoff_c.bin"))


def export_c_tables(path: str = DEFAULT_C_TABLE,
                    os_path: str = DEFAULT_TABLE,
                    ts_path: str = DEFAULT_TS_TABLE) -> str:
    """Write both tables as bgc_bearoff_open reads them: little-endian
    "BGBO", uint32 version 1, n_os, R_MAX, n_ts, ts_max; then os counts
    uint8[n_os][6], f, g float32[n_os][R_MAX], ts counts uint8[n_ts][6],
    P float32[n_ts][n_ts], V float32[n_ts][n_ts][3]: the npz arrays as
    stored. Written to a temp file and renamed into place."""
    load_table(os_path)                          # the staleness checks
    load_ts_table(ts_path)
    d, t = np.load(os_path), np.load(ts_path)
    arrs = [np.ascontiguousarray(d["counts"], dtype=np.uint8),
            np.ascontiguousarray(d["f"], dtype="<f4"),
            np.ascontiguousarray(d["g"], dtype="<f4"),
            np.ascontiguousarray(t["counts"], dtype=np.uint8),
            np.ascontiguousarray(t["P"], dtype="<f4"),
            np.ascontiguousarray(t["V"], dtype="<f4")]
    for a, src in zip(arrs, (d["counts"], d["f"], d["g"], t["counts"], t["P"], t["V"])):
        if not np.array_equal(a, src):
            raise ValueError(f"{os_path} / {ts_path}: unexpected table dtypes")
    n_os, n_ts = len(arrs[0]), len(arrs[3])
    header = np.array([1, n_os, R_MAX, n_ts, int(t["ts_max"])], dtype="<u4")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(b"BGBO")
        fh.write(header.tobytes())
        for a in arrs:
            fh.write(a.tobytes())
    os.replace(tmp, path)
    return path


def c_table_path(path: str = DEFAULT_C_TABLE) -> str:
    """The C table file, (re)exported when missing or older than either
    npz; one exporter at a time (a lock file beside it)."""
    import fcntl

    def stale():
        if not os.path.exists(path):
            return True
        t = os.path.getmtime(path)
        return any(os.path.getmtime(p) > t
                   for p in (DEFAULT_TABLE, DEFAULT_TS_TABLE) if os.path.exists(p))

    if stale():
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if stale():
                export_c_tables(path)
    return path


def _ts_indices(state: BoardState) -> Optional[Tuple[int, int]]:
    """(idx_on_roll, idx_opp) into the TS table, or None if out of class."""
    if not is_bearoff_class(state):
        return None
    ts = load_ts_table()
    cme = home_counts(state, state.turn)
    copp = home_counts(state, 1 - state.turn)
    if sum(cme) > ts["ts_max"] or sum(copp) > ts["ts_max"]:
        return None
    return ts["index"][cme], ts["index"][copp]


def ts_win_prob(state: BoardState) -> Optional[float]:
    ij = _ts_indices(state)
    if ij is None:
        return None
    return float(load_ts_table()["P"][ij[0], ij[1]])


def ts_equity(state: BoardState, persp: CubePerspective) -> Optional[float]:
    """Exact cubeful equity with the on-roll player's double window
    folded (successor form)."""
    ij = _ts_indices(state)
    if ij is None:
        return None
    return float(load_ts_table()["V"][ij[0], ij[1], int(persp)])


@lru_cache(maxsize=100_000)
def _ts_wnd(ia: int, ib: int) -> Tuple[float, float, float]:
    """Window-free values of position ia on roll against ib, per cube
    perspective, one dice pass over the V table."""
    ts = load_ts_table()
    parked = [0] * 24
    parked[23] = -15
    pts = parked[:]
    for j, v in enumerate(ts["counts"][ia]):
        pts[j] = v
    st = BoardState(points=pts, turn=WHITE)
    zero = ts["index"][(0, 0, 0, 0, 0, 0)]
    Vt = ts["V"]
    wnd = np.zeros(3)
    for (d1, d2), pd in zip(_DICE, _DICE_P):
        sl = {ts["index"][_counts_from_board(s)]
              for _, s in get_legal_plays(st, (d1, d2))}
        if zero in sl:
            wnd += pd
            continue
        arr = Vt[ib][sorted(sl)]
        m = arr.min(axis=0)
        wnd[_CENT] += pd * (-m[_CENT])
        wnd[_MINE] += pd * (-m[_THEIRS])
        wnd[_THEIRS] += pd * (-m[_MINE])
    return float(wnd[0]), float(wnd[1]), float(wnd[2])


def ts_cube_values(state: BoardState, persp: CubePerspective
                   ) -> Optional[Tuple[float, float]]:
    """(v_no_double, v_double_take) exact from the TS table, on-roll
    view. None when out of class."""
    ij = _ts_indices(state)
    if ij is None:
        return None
    wnd = _ts_wnd(ij[0], ij[1])
    return float(wnd[int(persp)]), 2.0 * float(wnd[_THEIRS])
