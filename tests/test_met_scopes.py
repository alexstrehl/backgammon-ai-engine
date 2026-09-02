"""Per-consumer MET grounding (`--met-scope`).

`--met` originally grounded four consumers at once because they all
route through `_eval_new_game`. But only some of them CAN be grounded on
principle:

  cube     — v_dp, the double/pass branch. A PURE table lookup: being
             dropped scores cube_value points and what that is worth at
             this score is a MET entry, with no board dependence at all.
  terminal — gammon/backgammon pricing of terminal successors.
  boundary — the collector's game-boundary target; the only scope that
             trains the net's OWN scale toward the table.

v_nd and v_dt are deliberately ungroundable: they need this position's
win/gammon/backgammon rates, which no table contains.
"""

import random

import numpy as np
import pytest
import torch

from backgammon_engine import BoardState
from met import MatchEquityTable, DEFAULT_MET
from model import TDNetwork
from modes import MatchState, MatchplayMode
from td_agent import TDAgent
from trainer import _collect_episode_matchplay

ALL = ("cube", "terminal", "boundary")


@pytest.fixture(scope="module")
def table():
    return MatchEquityTable.from_xml(DEFAULT_MET)


def agent(met=None, scopes=None, seed=0):
    torch.manual_seed(seed)
    net = TDNetwork(hidden_sizes=[32, 16], input_size=207,
                    encoder_name="matchplay1h_perspective196",
                    output_mode="probability")
    return TDAgent(net, met=met, met_scopes=scopes)


def ms(my_away=2, opp_away=2, **kw):
    return MatchState(jacoby=False, match_length=3,
                      my_away=my_away, opp_away=opp_away, **kw)


# ── the flag itself ─────────────────────────────────────────────────


def test_default_grounds_everything(table):
    assert agent(met=table).met_scopes == frozenset(ALL)


def test_unknown_scope_rejected(table):
    with pytest.raises(ValueError, match="unknown met scope"):
        agent(met=table, scopes=("cube", "bogus"))


def test_scope_is_inert_without_a_table():
    """No table means the net answers regardless of scope."""
    a = agent(met=None, scopes=("cube",))
    b = agent(met=None, scopes=ALL)
    m = ms(1, 3)
    assert a._eval_new_game(m, 0, scope="cube") == b._eval_new_game(m, 0, scope="cube")


# ── cube scope: the double/pass branch ──────────────────────────────


def test_cube_scope_grounds_v_dp_exactly(table):
    """v_dp at 2-away/2-away must be the table's 1-away/2-away entry --
    the canary that drifted 0.680 -> 0.639 under pure TD(0)."""
    a = agent(met=table, scopes=("cube",))
    offer = a.offer_double_matchplay(BoardState.initial(), ms(2, 2))
    assert offer._cache["v_dp"] == pytest.approx(table.mwc(1, 2), abs=1e-12)
    assert offer._cache["v_dp"] == pytest.approx(0.67736, abs=1e-5)


def test_cube_scope_leaves_v_nd_and_v_dt_learned(table):
    """The board-dependent branches must still come from the network --
    two different nets must disagree on them while agreeing on v_dp."""
    a1 = agent(met=table, scopes=("cube",), seed=1)
    a2 = agent(met=table, scopes=("cube",), seed=2)
    s, m = BoardState.initial(), ms(2, 3)
    o1 = a1.offer_double_matchplay(s, m)._cache
    o2 = a2.offer_double_matchplay(s, m)._cache
    assert o1["v_dp"] == pytest.approx(o2["v_dp"], abs=1e-12)
    assert o1["v_nd"] != o2["v_nd"]
    assert o1["v_dt"] != o2["v_dt"]


def test_cube_scope_alone_does_not_ground_terminal_or_boundary(table):
    a_cube = agent(met=table, scopes=("cube",), seed=3)
    a_none = agent(met=table, scopes=(), seed=3)
    a_all = agent(met=table, scopes=ALL, seed=3)
    m = ms(2, 3)
    # terminal / boundary fall back to the net under cube-only...
    for scope in ("terminal", "boundary"):
        assert a_cube._eval_new_game(m, 0, scope=scope) == pytest.approx(
            a_none._eval_new_game(m, 0, scope=scope), abs=1e-12)
        # ...and are grounded under "all", which must differ from the net.
        assert a_all._eval_new_game(m, 0, scope=scope) != pytest.approx(
            a_none._eval_new_game(m, 0, scope=scope), abs=1e-4)


def test_take_point_uses_the_cube_scope(table):
    """respond_to_double recomputes v_dp when given no hint; it must be
    grounded by the cube scope too, or doubler and responder would
    disagree about the same quantity."""
    a = agent(met=table, scopes=("cube",))
    s, m = BoardState.initial(), ms(2, 2)
    offer = a.offer_double_matchplay(s, m)
    from_hint = a.respond_to_double_matchplay(s, m, hint=offer)
    recomputed = a.respond_to_double_matchplay(s, m, hint=None)
    assert from_hint == recomputed


# ── boundary scope: the training target ─────────────────────────────


def _scope_calls(a, match_length, oneply, episodes=10):
    """How often each scope is actually consulted during collection."""
    hits = {"cube": 0, "terminal": 0, "boundary": 0}
    orig = TDAgent._eval_new_game

    def spy(self, m, p, scope="cube"):
        hits[scope] += 1
        return orig(self, m, p, scope=scope)

    mode = MatchplayMode(match_length=match_length)
    TDAgent._eval_new_game = spy
    try:
        for seed in range(episodes):
            _collect_episode_matchplay(
                a, mode, random.Random(seed), mode.initial_match_state(),
                oneply=oneply)
    finally:
        TDAgent._eval_new_game = orig
    return hits


def test_cube_scope_dominates_the_others(table):
    """MEASURED, and it reframes what "--met" ever tested.

    `_eval_new_game` is consulted almost exclusively for v_dp. Terminal
    pricing needs a game-ENDING move to be legal, and the boundary
    target only fires when a game ends without ending the MATCH -- at
    short match lengths a finished game usually finishes the match, so
    the target is a bare 1.0 and the table is never consulted.

    Consequence: grounding "everything" is, in practice, ~99% grounding
    the double/pass branch alone. The Phase 2 factorial therefore
    measured essentially v_dp grounding, and found it null.
    """
    a = agent(met=table, scopes=ALL, seed=5)
    hits = _scope_calls(a, match_length=3, oneply=False)
    assert hits["cube"] > 20 * (hits["terminal"] + hits["boundary"] + 1), hits


def test_oneply_does_not_amplify_terminal_pricing(table):
    """Terminal pricing stays O(game endings), not O(rows), at 1-ply.

    Since the conditioned-boundary fix, the 1-ply collector prices the
    game-ending turn through the unconditional 21-dice enumeration, so
    terminal-scope reads scale with endings and near-terminal states
    (measured ~56 per episode set vs 2 at 0-ply) -- still a tiny
    fraction of total evals, refuting the once-suspected amplification
    of terminal-pricing bias into most targets. 1-ply must not use the
    realized boundary read at endings at all, and the cube path is a
    chance-free decision node, so its scope traffic must be
    ply-invariant.
    """
    a = agent(met=table, scopes=ALL, seed=5)
    zero = _scope_calls(a, match_length=3, oneply=False)
    one = _scope_calls(a, match_length=3, oneply=True)
    assert one["terminal"] < 200, one
    assert one["boundary"] <= zero["boundary"], (
        "1-ply reintroduced realized boundary reads at game endings")
    assert one["cube"] == zero["cube"], "cube path must be ply-independent"


# ── the ungroundable pair, stated as a test ─────────────────────────


def test_v_nd_and_v_dt_have_no_scope(table):
    """There is deliberately no scope for them. They are board-dependent
    -- they need this position's win/gammon/backgammon rates -- so a
    table cannot supply them at any granularity. Grounding those needs a
    different OUTPUT representation (a per-outcome head combined with the
    table), not a finer flag.
    """
    a = agent(met=table, scopes=ALL)
    assert a.met_scopes == frozenset(ALL)
    with pytest.raises(ValueError, match="unknown met scope"):
        agent(met=table, scopes=("v_nd",))
