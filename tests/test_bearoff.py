"""Bearoff tables (bearoff.py) against gnubg and brute force.

  * one-sided table vs gnubg's os database (gnubg_nn.bearoff_probabilities)
  * two-sided exact table (P, V) vs gnubg's exact BEAROFF2 evaluations
    (gnubg_nn probs, CLI `hint` cubeful money equities)
  * both vs an independent cube expectimax over real BoardStates
  * the hazard-DP fallback vs the exact tier

Skipped as a whole when the tables are not built (bearoff.py).
"""

import os
import re
import shutil
import subprocess

import numpy as np
import pytest

import bearoff
from backgammon_engine import BoardState, WHITE, BLACK, get_legal_plays
from encoding import CubePerspective

pytestmark = pytest.mark.skipif(
    not (os.path.exists(bearoff.DEFAULT_TABLE)
         and os.path.exists(bearoff.DEFAULT_TS_TABLE)),
    reason="bearoff tables not built")

try:
    import gnubg_nn
    from third_party_opponents.gnubg.gnubg_nn_agent import GnubgNNAgent
    HAVE_GNUBG_NN = True
except Exception:
    HAVE_GNUBG_NN = False

GNUBG_CLI = shutil.which("gnubg")

CENT = CubePerspective.CENTERED
MINE = CubePerspective.MINE
THEIRS = CubePerspective.THEIRS


def _state(cw, cb, turn=WHITE):
    pts = [0] * 24
    for i, v in enumerate(cw):
        pts[i] = v
    for i, v in enumerate(cb):
        pts[23 - i] = -v
    return BoardState(points=pts, off=[15 - sum(cw), 15 - sum(cb)], turn=turn)


def _rand_home(rng, nmax, npts=6):
    c = [0] * 6
    for _ in range(rng.integers(1, nmax + 1)):
        c[rng.integers(0, npts)] += 1
    return tuple(c)


# ── one-sided table ──────────────────────────────────────────────────


def test_known_distributions():
    f, g = bearoff.dists((1, 0, 0, 0, 0, 0))
    assert f[0] == pytest.approx(1.0)          # 1 checker on the ace: 1 roll
    assert g[0] == pytest.approx(1.0)
    f, _ = bearoff.dists((0, 0, 0, 0, 0, 1))   # 1 checker on the 6-point
    assert f[0] == pytest.approx(27.0 / 36.0)  # off unless 21,31,41,32,42,11
    assert f[1] == pytest.approx(9.0 / 36.0)


def test_distributions_sum_to_one():
    t = bearoff.load_table()
    rng = np.random.default_rng(0)
    for i in rng.integers(0, len(t["f"]), 500):
        s = t["f"][i].sum()
        assert (s == pytest.approx(1.0, abs=2e-5)     # float32 storage
                or s == pytest.approx(0.0))    # cleared position rows are 0


@pytest.mark.skipif(not HAVE_GNUBG_NN, reason="gnubg_nn unavailable")
def test_one_sided_vs_gnubg_db():
    """Extensive: 1500 random positions vs gnubg's os database."""
    t = bearoff.load_table()
    rng = np.random.default_rng(42)
    keys = list(t["index"].keys())
    worst = 0.0
    for j in rng.integers(0, len(keys), 1500):
        c = keys[int(j)]
        if sum(c) == 0:
            continue
        ours = t["f"][t["index"][c]]
        gn = np.array(gnubg_nn.bearoff_probabilities(tuple(c)))
        L = min(len(gn), len(ours))
        d = float(np.max(np.abs(ours[:L] - gn[:L])))
        worst = max(worst, d)
    # One mu-tie position differs by ~1.4e-3 (a different equal-E[rolls]
    # move than gnubg's db); the rest is db quantization.
    assert worst < 2e-3, f"one-sided dist deviates from gnubg db by {worst}"


# ── two-sided exact table vs gnubg exact evals ───────────────────────


@pytest.mark.skipif(not HAVE_GNUBG_NN, reason="gnubg_nn unavailable")
def test_two_sided_p_vs_gnubg_exact():
    """500 random TS positions: the table's exact P vs gnubg's exact eval."""
    rng = np.random.default_rng(7)
    diffs = []
    for _ in range(500):
        s = _state(_rand_home(rng, 6), _rand_home(rng, 6))
        p_g = gnubg_nn.probabilities(GnubgNNAgent._board_to_gnubg(s), 0)[0]
        p_o = bearoff.ts_win_prob(s)
        assert p_o is not None
        diffs.append(abs(p_o - p_g))
    diffs = np.array(diffs)
    assert diffs.max() < 2e-3, f"max {diffs.max()}"
    assert diffs.mean() < 2e-4, f"mean {diffs.mean()}"


# ── independent brute force (cube expectimax over real states) ───────


def _brute(cw, cb, persp, memo, with_fold=True):
    """Full expectimax with double windows, independent of bearoff.py."""
    key = (cw, cb, int(persp), with_fold)
    if key in memo:
        return memo[key]
    s = _state(cw, cb)
    total = 0.0
    for d1 in range(1, 7):
        for d2 in range(d1, 7):
            pd = (1.0 if d1 == d2 else 2.0) / 36.0
            best = None
            for _, ns in get_legal_plays(s, (d1, d2)):
                if ns.is_game_over():
                    v = 1.0
                else:
                    ncw = tuple(max(0, ns.points[i]) for i in range(6))
                    flip = {CENT: CENT, MINE: THEIRS, THEIRS: MINE}[persp]
                    v = -_brute(cb, ncw, flip, memo, with_fold=True)
                if best is None or v > best:
                    best = v
            total += pd * best
    if with_fold and persp in (CENT, MINE):
        dt = 2.0 * _brute(cw, cb, THEIRS, memo, with_fold=False)
        total = max(total, min(dt, 1.0))
    memo[key] = total
    return total


def test_ts_table_vs_brute_force():
    """Exact table == independent expectimax on small positions."""
    memo = {}
    cases = [
        ((1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0)),
        ((0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1)),   # man-on-6 vs man-on-6
        ((1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)),
        ((0, 1, 1, 0, 0, 0), (1, 0, 0, 1, 0, 0)),
        ((2, 1, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1)),
        ((1, 1, 1, 0, 0, 0), (1, 1, 0, 0, 0, 1)),
    ]
    for cw, cb in cases:
        s = _state(cw, cb)
        for persp in (CENT, MINE, THEIRS):
            bf = _brute(cw, cb, persp, memo, with_fold=True)
            ours = bearoff.ts_equity(s, persp)
            assert ours == pytest.approx(bf, abs=5e-6), (cw, cb, persp)
            nd_bf = _brute(cw, cb, persp, memo, with_fold=False)
            dt_bf = 2.0 * _brute(cw, cb, THEIRS, memo, with_fold=False)
            nd, dt = bearoff.ts_cube_values(s, persp)
            assert nd == pytest.approx(nd_bf, abs=5e-6)
            assert dt == pytest.approx(dt_bf, abs=1e-5)


def test_man_on_six_classic():
    """Man-on-6 vs man-on-6 (Janowski p.2): the on-roll player doubles."""
    s = _state((0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1))
    p = bearoff.ts_win_prob(s)
    nd, dt = bearoff.ts_cube_values(s, CENT)
    assert p > 0.75                             # on-roll heavy favourite
    assert dt > nd                              # double is right


# ── gnubg CLI hint: exact cubeful money equities ─────────────────────


def _hint_cube_gnubg(s: BoardState, owner_rel: str, plies=0):
    """ND/DT/DP cubeful equities from real gnubg for the on-roll player.
    In the BEAROFF2 class these are gnubg's exact CFMONEY values."""
    posid = gnubg_nn.position_id(GnubgNNAgent._board_to_gnubg(s))
    owner_cmd = {"centered": "set cube centre",
                 "mine": "set cube owner 0",
                 "theirs": "set cube owner 1"}[owner_rel]
    lines = [
        "set automatic game off", "set automatic roll off",
        "set player 0 human", "set player 1 human",
        "new session", "set jacoby off", "set cube use on",
        "set evaluation cubedecision evaluation plies %d" % plies,
        "set turn 0", "set board %s" % posid,
        "set cube value 1", owner_cmd, "hint",
    ]
    out = subprocess.run(
        [GNUBG_CLI, "-t", "-q"], input="\n".join(lines) + "\n",
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "LC_ALL": "C"}).stdout
    pat = re.compile(
        r"(No (?:re)?double|(?:Re)?[Dd]ouble, take|(?:Re)?[Dd]ouble, pass)"
        r"\s*:?\s*([+-]?\d+[.,]\d+)")
    eqs = {}
    for m in pat.finditer(out):
        label = m.group(1).lower().replace("redouble", "double")
        val = float(m.group(2).replace(",", "."))
        if "no double" in label:
            eqs["nd"] = val
        elif "take" in label:
            eqs["dt"] = val
        elif "pass" in label:
            eqs["dp"] = val
    return eqs if {"nd", "dt"} <= set(eqs) else None


@pytest.mark.skipif(GNUBG_CLI is None or not HAVE_GNUBG_NN,
                    reason="gnubg CLI unavailable")
def test_cubeful_vs_gnubg_hint():
    """15 random TS positions x cube states: the table's exact ND/DT vs
    gnubg's exact CFMONEY hint equities."""
    rng = np.random.default_rng(11)
    checked = 0
    worst = 0.0
    fails = []
    while checked < 15:
        s = _state(_rand_home(rng, 6), _rand_home(rng, 6))
        p = bearoff.ts_win_prob(s)
        if p is None or p < 0.02 or p > 0.98:
            continue                        # hint prints trivial lines there
        owner_rel, persp = [("centered", CENT), ("mine", MINE)][checked % 2]
        eqs = _hint_cube_gnubg(s, owner_rel)
        if eqs is None:
            continue
        nd, dt = bearoff.ts_cube_values(s, persp)
        d = max(abs(nd - eqs["nd"]), abs(dt - eqs["dt"]))
        worst = max(worst, d)
        if d > 0.02:
            fails.append((bearoff.home_counts(s, WHITE),
                          bearoff.home_counts(s, BLACK), owner_rel,
                          (nd, dt), eqs))
        checked += 1
    assert not fails, f"worst={worst}: {fails[:3]}"


# ── hazard-DP fallback: bound the approximation vs exact ─────────────


def test_hazard_dp_vs_exact_on_ts_class():
    rng = np.random.default_rng(3)
    gaps = []
    for _ in range(200):
        s = _state(_rand_home(rng, 6), _rand_home(rng, 6))
        if s.off[WHITE] == 0 or s.off[BLACK] == 0:
            continue
        r = bearoff.race_hazards(s)
        if r is None:
            continue
        hm, ho = r
        for persp in (CENT, MINE, THEIRS):
            approx = bearoff._race_dp(hm, ho, int(persp))
            nd_exact, _ = bearoff.ts_cube_values(s, persp)
            gaps.append(abs(approx - nd_exact))
    gaps = np.array(gaps)
    # The exact TS tier covers this zone at run time; the tail comes from
    # 1-3-roll positions where min-E[rolls] play differs from cube-aware play.
    assert np.median(gaps) < 0.005, f"median {np.median(gaps)}"
    assert gaps.mean() < 0.06, f"mean {gaps.mean()}"


# ── gammon accounting ────────────────────────────────────────────────


def test_exact_probs5_gammon_path():
    # WHITE about to clear 2 checkers; BLACK 15 on their ace, none off
    s = _state((2, 0, 0, 0, 0, 0), (15, 0, 0, 0, 0, 0))
    p = bearoff.exact_probs5(s)
    assert p[0] > 0.99                       # white nearly always wins
    assert 0.0 < p[1] <= p[0]                # gammon chances live (black 0 off)
    assert p[2] == 0.0 and p[4] == 0.0       # no backgammons in this class
    # once black has one off, no gammons
    s2 = _state((2, 0, 0, 0, 0, 0), (14, 0, 0, 0, 0, 0))
    assert bearoff.exact_probs5(s2)[1] == 0.0


def test_first_off_policy_saves_the_gammon():
    """g follows the play that bears a checker off soonest (gnubg's os
    db), not the clearing play. WHITE on roll with 4 on its 3-point, 3
    on its 5-point and 8 on its 6-point; BLACK's last checker, on its
    3-point, comes off next roll. Only 4-4 bears nothing off."""
    s = _state((0, 0, 4, 0, 3, 8), (0, 0, 1, 0, 0, 0))
    assert bearoff.exact_probs5(s)[3] == pytest.approx(1 / 36, abs=1e-6)


def test_stale_one_sided_table_refused(tmp_path, monkeypatch):
    t = bearoff.load_table()
    old = tmp_path / "os6.npz"
    np.savez(old, counts=np.zeros((1, 6), np.uint8), f=t["f"][:1],
             g=t["g"][:1], mu=t["mu"][:1])
    monkeypatch.setattr(bearoff, "_TABLE", None)
    with pytest.raises(ValueError, match="rebuild"):
        bearoff.load_table(str(old))
    monkeypatch.setattr(bearoff, "_TABLE", t)


@pytest.mark.skipif(not HAVE_GNUBG_NN, reason="gnubg_nn unavailable")
def test_gammon_probs_vs_gnubg_os_class():
    """One-sided class with gammons live (one side all 15 home, none
    off): p and the gammon probabilities vs gnubg's os-class eval."""
    rng = np.random.default_rng(19)
    full = sorted(c for c in bearoff.load_table()["index"] if sum(c) == 15)
    dp, dg = [], []
    for k in range(600):
        heavy = full[int(rng.integers(0, len(full)))]
        light = _rand_home(rng, 8)
        cw, cb = (heavy, light) if k % 2 else (light, heavy)
        s = _state(tuple(cw), tuple(cb))
        probs_g = gnubg_nn.probabilities(GnubgNNAgent._board_to_gnubg(s), 0)
        ours = bearoff.exact_probs5(s)
        dp.append(abs(ours[0] - probs_g[0]))
        dg.append(max(abs(ours[1] - probs_g[1]), abs(ours[3] - probs_g[3])))
    dp, dg = np.array(dp), np.array(dg)
    assert dp.mean() < 2e-3, f"p mean {dp.mean()}"
    assert dg.mean() < 5e-5, f"gammon mean {dg.mean()}"
    assert (dg > 1e-3).sum() <= 2, np.sort(dg)[-5:]
