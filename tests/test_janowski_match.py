"""Structural + doctrine tests for janowski_match.py (no gnubg needed).

Every number asserted here is DERIVED from the MET at test time, never
hard-coded, so the tests hold under any table. The gnubg cross-check
(same conversion vs the real binary's cubeful MWCs) is a separate,
binary-dependent validation.

The conversion tests run on both implementations (the `jm` fixture,
tests/janowski_impl.py): janowski_match itself and its C port
(c_api/puretd_janowski.c).
"""

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoding import CubePerspective
from janowski_impl import jm  # noqa: F401  (fixture: py / c implementation)
from janowski_match import MatchCtx, ctx_for, _me
from met import default_met
from modes import CubeOwner, MatchState

MET = default_met()

CENTERED = CubePerspective.CENTERED
MINE = CubePerspective.MINE
THEIRS = CubePerspective.THEIRS


def nested(p, wg=0.0, wbg=0.0, lg=0.0, lbg=0.0):
    """Build a valid nested prob5 vector."""
    assert wbg <= wg <= p and lbg <= lg <= 1.0 - p
    return np.array([p, wg, wbg, lg, lbg], dtype=np.float64)


def rand_nested(rng):
    p = rng.uniform(0.02, 0.98)
    wg = rng.uniform(0.0, p)
    wbg = rng.uniform(0.0, wg)
    lg = rng.uniform(0.0, 1.0 - p)
    lbg = rng.uniform(0.0, lg)
    return np.array([p, wg, wbg, lg, lbg])


def mirror(probs):
    """The same position from the other player's view."""
    return np.array([1.0 - probs[0], probs[3], probs[4],
                     probs[1], probs[2]])


def rand_ctx(rng, allow_crawford=False):
    n = rng.choice([3, 5, 7])
    my = rng.randint(1, n)
    opp = rng.randint(1, n)
    cube = rng.choice([1, 2, 4])
    one_away = (my == 1 or opp == 1)
    if one_away:
        if allow_crawford and rng.random() < 0.5:
            return MatchCtx(my, opp, True, True, 1)     # Crawford game
        return MatchCtx(my, opp, True, False, cube)     # post-Crawford
    return MatchCtx(my, opp, False, False, cube)


# ── structural invariants ────────────────────────────────────────────


def test_mirror_symmetry(jm):
    """V_me + V_opp(mirrored everything) == 1, all perspectives."""
    rng = random.Random(1)
    pairs = {CENTERED: CENTERED, MINE: THEIRS, THEIRS: MINE}
    for _ in range(400):
        ctx = rand_ctx(rng)
        probs = rand_nested(rng)
        x = rng.uniform(0.0, 1.0)
        for persp, mpersp in pairs.items():
            a = jm.cubeful_mwc(probs, persp, ctx, x, MET)
            b = jm.cubeful_mwc(mirror(probs), mpersp, ctx.flip(), x, MET)
            assert a + b == pytest.approx(1.0, abs=1e-9), (ctx, persp)


def test_dead_gates_return_dead(jm):
    """Crawford, cube >= both aways, and 2a2a are pure dead cube."""
    rng = random.Random(2)
    dead_ctxs = [
        MatchCtx(1, 3, True, True, 1),     # Crawford
        MatchCtx(2, 2, False, False, 1),   # gnubg's -2,-2 special
        MatchCtx(2, 3, False, False, 4),   # cube >= both aways
        MatchCtx(1, 1, True, False, 1),    # 1a1a
    ]
    for ctx in dead_ctxs:
        assert not jm.do_cubeful(ctx)
        for _ in range(20):
            probs = rand_nested(rng)
            d = jm.dead_mwc(probs, ctx, MET)
            for persp in (CENTERED, MINE, THEIRS):
                assert jm.cubeful_mwc(probs, persp, ctx, 0.68, MET) == \
                    pytest.approx(d, abs=1e-12)


def test_dmp_reduces_to_pwin(jm):
    """At 1-away/1-away the cubeful MWC is exactly P(win)."""
    ctx = MatchCtx(1, 1, True, False, 1)
    for p in (0.1, 0.4, 0.5, 0.9):
        probs = nested(p, wg=p * 0.3, lg=(1 - p) * 0.3)
        for persp in (CENTERED, MINE, THEIRS):
            assert jm.cubeful_mwc(probs, persp, ctx, 0.68, MET) == \
                pytest.approx(p, abs=1e-12)


def test_monotone_and_bounded_in_p(jm):
    """Fixed rates, sweeping p: cubeful MWC is non-decreasing and
    stays inside [min, max] of the endpoint MWCs."""
    rng = random.Random(3)
    for _ in range(60):
        ctx = rand_ctx(rng)
        g_w, g_l = rng.uniform(0, 0.4), rng.uniform(0, 0.4)
        x = rng.uniform(0.0, 1.0)
        for persp in (CENTERED, MINE, THEIRS):
            last = -1.0
            for p in np.linspace(0.001, 0.999, 101):
                probs = nested(p, wg=g_w * p, lg=g_l * (1 - p))
                v = jm.cubeful_mwc(probs, persp, ctx, x, MET)
                assert v >= last - 1e-9
                assert -1e-9 <= v <= 1.0 + 1e-9
                last = v


def test_continuity_at_kinks(jm):
    """The piecewise-live curve is continuous across its kinks."""
    rng = random.Random(4)
    for _ in range(60):
        ctx = rand_ctx(rng)
        if not jm.do_cubeful(ctx):
            continue
        rates = (0.2, 0.02, 0.15, 0.01)
        cp_me, cp_opp = jm.get_points(rates, ctx, MET)
        for kink in (cp_me, 1.0 - cp_opp):
            if not (0.01 < kink < 0.99):
                continue
            for persp in (CENTERED, MINE, THEIRS):
                def probs_at(p):
                    return nested(
                        p, wg=(rates[0] + rates[1]) * p,
                        wbg=rates[1] * p,
                        lg=(rates[2] + rates[3]) * (1 - p),
                        lbg=rates[3] * (1 - p),
                    )
                lo = jm.cubeful_mwc(probs_at(kink - 1e-6), persp, ctx, 1.0, MET)
                hi = jm.cubeful_mwc(probs_at(kink + 1e-6), persp, ctx, 1.0, MET)
                assert abs(hi - lo) < 1e-4


# ── doctrine tests (numbers derived from the MET at test time) ──────


def _ms(my, opp, length=None, cube=1, owner=CubeOwner.CENTERED,
        crawford_done=False, is_crawford=False):
    return MatchState(
        cube_owner=owner, cube_value=cube, jacoby=False,
        match_length=length or max(my, opp),
        my_away=my, opp_away=opp,
        is_crawford=is_crawford, crawford_done=crawford_done,
    )


def test_2a2a_doctrine(jm):
    """2a2a: double on any clear advantage; take up to the 1a2a MWC.

    Derivation from the table (gammonless): the cube-2 branches end
    the match, so v_dt = p; v_nd is the dead mix at cube 1; passing
    gives the doubler mwc(1a, 2a). Thresholds fall out of the MET.
    """
    ms = _ms(2, 2, length=3)
    w1 = _me(MET, ctx_for(ms, 0), 1, True)     # -> Crawford 1a2a
    # Double point: p where min(p, w1) crosses the dead line.
    for p, expect in ((0.52, True), (0.48, False)):
        d = jm.cube_decision(nested(p), ms, 0, 0.68, MET)
        assert d["should_double"] == expect, (p, d)
    # Take while v_dp > v_dt = p, i.e. p < w1.
    d = jm.cube_decision(nested(w1 - 0.02), ms, 0, 0.68, MET)
    assert d["take"]
    d = jm.cube_decision(nested(w1 + 0.02), ms, 0, 0.68, MET)
    assert not d["take"]


def test_post_crawford_trailer_always_doubles(jm):
    """Post-Crawford trailer: doubling dominates at any p."""
    ms = _ms(2, 1, length=3, crawford_done=True)
    assert ms.can_offer(0)
    for p in (0.3, 0.5, 0.7):
        d = jm.cube_decision(nested(p, wg=0.15 * p, lg=0.15 * (1 - p)),
                          ms, 0, 0.68, MET)
        assert d["should_double"], (p, d)


def test_post_crawford_free_drop(jm):
    """Leader 1-away facing the mandatory double at 2-away
    post-Crawford: pass iff behind in the game (the free drop --
    passing burns one of the trailer's even points)."""
    # Doubler (trailer, 2-away) view: v_dp = trailer MWC after
    # scoring 1 point -> 1a1a post-Crawford = 0.5. v_dt: cube 2 ends
    # the match either way -> v_dt = p_trailer (gammonless).
    ms = _ms(2, 1, length=3, crawford_done=True)
    d = jm.cube_decision(nested(0.55), ms, 0, 0.68, MET)
    assert not d["take"]        # trailer 55% -> leader passes
    d = jm.cube_decision(nested(0.45), ms, 0, 0.68, MET)
    assert d["take"]            # trailer 45% -> leader takes


def test_3a3a_thresholds_sane(jm):
    """3a3a centered cube-1: the double and cash points land in the
    conventional bands (soft check; exact values vary by table)."""
    ms = _ms(3, 3)
    grid = np.linspace(0.30, 0.95, 400)
    dbl = [p for p in grid
           if jm.cube_decision(nested(p), ms, 0, 0.68, MET)["should_double"]]
    take = [p for p in grid
            if jm.cube_decision(nested(p), ms, 0, 0.68, MET)["take"]]
    assert dbl, "no doubling point found at 3a3a"
    assert 0.5 < dbl[0] < 0.8, dbl[0]          # double point
    assert take, "no take region at 3a3a"
    assert 0.6 < take[-1] < 0.85, take[-1]     # cash point


def test_money_limit_cash_point(jm):
    """25a25a (deep match ~ money): the live cash point approaches
    Janowski's money live CP = (L + 1) / (W + L + 0.5)."""
    ctx = MatchCtx(25, 25, False, False, 1)
    g = (0.2, 0.02, 0.2, 0.02)
    W = 1.0 + g[0] + g[1]
    L = 1.0 + g[2] + g[3]
    cp_money = (L + 1.0) / (W + L + 0.5)
    cp_me, cp_opp = jm.get_points(g, ctx, MET)
    assert cp_me == pytest.approx(cp_money, abs=0.03)
    assert cp_opp == pytest.approx(cp_money, abs=0.03)


def test_cube_decision_matches_manual_composition(jm):
    """cube_decision == the three cubeful_mwc calls it is built from."""
    rng = random.Random(5)
    for _ in range(40):
        my, opp = rng.randint(2, 5), rng.randint(2, 5)
        ms = _ms(my, opp, length=max(my, opp) + 1)
        probs = rand_nested(rng)
        d = jm.cube_decision(probs, ms, 0, 0.68, MET)
        ctx = ctx_for(ms, 0)
        assert d["v_nd"] == pytest.approx(
            jm.cubeful_mwc(probs, CENTERED, ctx, 0.68, MET))
        assert d["v_dt"] == pytest.approx(
            jm.cubeful_mwc(probs, THEIRS, ctx.with_cube(2), 0.68, MET))
        assert d["v_dp"] == pytest.approx(_me(MET, ctx, 1, True))


def test_ctx_for_flips_for_black():
    ms = _ms(3, 2, length=5, cube=2)
    c0 = ctx_for(ms, 0)
    c1 = ctx_for(ms, 1)
    assert (c0.my_away, c0.opp_away) == (3, 2)
    assert (c1.my_away, c1.opp_away) == (2, 3)
    assert c0.cube == c1.cube == 2


# ── agent integration (needs the prob5 checkpoint) ──────────────────

PROD = "best_models/cubeless_prob5_512_512_256_128.pt"


@pytest.mark.skipif(not os.path.exists(PROD), reason="no prob5 checkpoint")
def test_agent_plays_full_matches():
    """Self-play smoke: full 3pt matches through eval_matchplay's own
    loop -- exercises cube phase, checker phase, terminal handling,
    Crawford transitions."""
    from eval_matchplay import play_match
    from janowski_match import JanowskiMatchAgent
    from model import ProbNetwork

    net = ProbNetwork.load(PROD)
    agent = JanowskiMatchAgent(net, met=MET, x_policy=0.68)
    rng = random.Random(7)
    wins = [play_match(agent, agent, 3, rng) for _ in range(6)]
    assert set(wins) <= {0, 1}


@pytest.mark.skipif(not os.path.exists(PROD), reason="no prob5 checkpoint")
def test_agent_dmp_checker_matches_pwin_ranking():
    """At 1a1a the cubeful chooser must pick the max-P(win) move."""
    from backgammon_engine import BoardState, opening_roll
    from janowski_match import JanowskiMatchAgent
    from model import ProbNetwork

    net = ProbNetwork.load(PROD)
    agent = JanowskiMatchAgent(net, met=MET, x_policy=0.68)
    ms = _ms(1, 1, length=3, crawford_done=True)
    rng = random.Random(11)
    state, dice = opening_roll(rng)
    chosen, _ = agent.choose_checker_action_cubeful(state, dice, ms)
    # Recompute by brute force: argmin of opponent P(win).
    from backgammon_engine import get_legal_plays
    plays = get_legal_plays(state, dice)
    best, best_p = None, 2.0
    for _, s in plays:
        p_opp = agent._probs_state(s)[0]
        if p_opp < best_p:
            best_p, best = p_opp, s
    assert chosen == best
