"""Phase 2: MET as ground truth.

`_eval_new_game` is the single point through which all cross-game
equity enters matchplay — v_dp, the take point, terminal gammon
pricing, and the collector's boundary target. These tests pin the
swap: with a MET attached every one of those becomes an exact table
lookup, and the learned version stays reachable as a diagnostic.
"""

import numpy as np
import pytest
import torch

from backgammon_engine import BoardState
from encoding import MatchplayCube1HEncoder
from met import MatchEquityTable, DEFAULT_MET, resolve_met_path
from model import TDNetwork
from modes import MatchState
from td_agent import TDAgent


@pytest.fixture(scope="module")
def table():
    return MatchEquityTable.from_xml(DEFAULT_MET)


def _agent(met=None, seed=0):
    torch.manual_seed(seed)
    net = TDNetwork(
        hidden_sizes=[24, 16], input_size=207,
        encoder_name="matchplay1h_perspective196", output_mode="probability",
    )
    return TDAgent(net, met=met)


def _ms(my_away, opp_away, **kw):
    return MatchState(match_length=3, my_away=my_away, opp_away=opp_away, **kw)


# ── the swap ────────────────────────────────────────────────────────


def test_defaults_to_network(table):
    agent = _agent()
    assert agent.met is None
    ms = _ms(3, 3)
    assert agent._eval_new_game(ms, 0) == agent._eval_new_game_net(ms, 0)


def test_met_agent_returns_table_values(table):
    agent = _agent(met=table)
    for my, opp in [(1, 2), (2, 1), (2, 2), (3, 3), (1, 3), (3, 2)]:
        got = agent._eval_new_game(_ms(my, opp), 0)
        assert got == pytest.approx(table.mwc(my, opp), abs=1e-12)


def test_met_agent_ignores_network_weights(table):
    """The whole point: the value no longer depends on the net."""
    a1, a2 = _agent(met=table, seed=1), _agent(met=table, seed=2)
    # An ASYMMETRIC score: at my_away == opp_away the net's averaged
    # estimate is exactly 0.5 for any weights (the initial position is
    # symmetric, so the two perspectives cancel), which would make this
    # test vacuous.
    ms = _ms(1, 3)
    assert a1._eval_new_game(ms, 0) == a2._eval_new_game(ms, 0)
    # ...whereas the learned version does depend on the weights.
    assert a1._eval_new_game_net(ms, 0) != a2._eval_new_game_net(ms, 0)


def test_net_met_is_exactly_half_at_symmetric_scores():
    """Structural, not learned: at equal away scores the averaged
    new-game estimate cancels to 0.5 for any weights. A "2a2a = 0.50"
    convergence claim is therefore not evidence of anything having been
    learned."""
    for seed in range(3):
        agent = _agent(seed=seed)
        for away in (2, 3, 5):
            assert agent._eval_new_game_net(_ms(away, away), 0) == pytest.approx(
                0.5, abs=1e-6
            )


def test_player_symmetry(table):
    agent = _agent(met=table)
    for my, opp in [(1, 2), (3, 2), (2, 2)]:
        ms = _ms(my, opp)
        assert (agent._eval_new_game(ms, 0)
                + agent._eval_new_game(ms, 1)) == pytest.approx(1.0, abs=1e-12)


def test_crawford_flag_is_applied(table):
    """start_game() schedules Crawford; the lookup must see it."""
    agent = _agent(met=table)
    fresh = _ms(1, 3, crawford_done=False)
    spent = _ms(1, 3, crawford_done=True)
    assert agent._eval_new_game(fresh, 0) == pytest.approx(table.mwc(1, 3))
    assert agent._eval_new_game(spent, 0) == pytest.approx(
        table.mwc(1, 3, crawford_done=True)
    )
    assert agent._eval_new_game(fresh, 0) != agent._eval_new_game(spent, 0)


# ── what it fixes ───────────────────────────────────────────────────


def test_v_dp_is_pinned_to_the_canary(table):
    """The canary: v_dp at 2-away/2-away must be the MET's
    1-away/2-away value. Under pure TD(0) it drifted 0.680 -> 0.639."""
    agent = _agent(met=table)
    state = BoardState.initial()
    ms = _ms(2, 2)
    offer = agent.offer_double_matchplay(state, ms)
    assert offer._cache["v_dp"] == pytest.approx(table.mwc(1, 2), abs=1e-12)
    assert offer._cache["v_dp"] == pytest.approx(0.67736, abs=1e-5)


def test_v_dp_survives_arbitrary_weights(table):
    """The drift mode was v_dp tracking the net. It must not, now."""
    state = BoardState.initial()
    ms = _ms(2, 2)
    seen = set()
    for seed in range(4):
        agent = _agent(met=table, seed=seed)
        seen.add(round(
            agent.offer_double_matchplay(state, ms)._cache["v_dp"], 12
        ))
    assert len(seen) == 1


def test_v_dp_wins_the_match_when_it_should(table):
    """At 1-away, being dropped ends the match — 1.0, not a lookup."""
    agent = _agent(met=table)
    ms = _ms(1, 3)
    offer = agent.offer_double_matchplay(BoardState.initial(), ms)
    if offer._cache is not None:  # can_offer may block at 1-away/cube 1
        assert offer._cache["v_dp"] == 1.0


def test_terminal_gammon_pricing_uses_the_table(table):
    """Move selection prices terminal successors through the same
    function, so gammon value at the current score is now exact."""
    agent = _agent(met=table)
    # 2-away: a gammon (2 points) wins the match, a single does not.
    ms = _ms(2, 3)
    single = table.mwc_after(2, 3, False, False, 1, True)
    gammon = table.mwc_after(2, 3, False, False, 2, True)
    assert gammon == 1.0
    assert single < 1.0
    assert agent._eval_new_game(
        ms.after_game(1, winner_is_me=True), 0
    ) == pytest.approx(single, abs=1e-12)


# ── plumbing ────────────────────────────────────────────────────────


def test_resolve_met_path_forms():
    assert resolve_met_path(None) is None
    assert resolve_met_path("") is None
    assert resolve_met_path("auto") == DEFAULT_MET
    assert resolve_met_path("snowie").endswith("snowie.xml")
    assert resolve_met_path(DEFAULT_MET) == DEFAULT_MET
    with pytest.raises(ValueError, match="neither a file nor a table"):
        resolve_met_path("no-such-met")


def test_worker_met_is_cached_per_path():
    from trainer import _worker_met
    assert _worker_met(None) is None
    a = _worker_met(DEFAULT_MET)
    b = _worker_met(DEFAULT_MET)
    assert a is b
    assert a.name == MatchEquityTable.from_xml(DEFAULT_MET).name


def test_different_tables_give_different_grounding():
    """Guards the Kazaross/Snowie referee split: picking a table must
    actually change the grounding, or --met is silently a no-op."""
    kaz = MatchEquityTable.from_xml(resolve_met_path("Kazaross-XG2"))
    snowie = MatchEquityTable.from_xml(resolve_met_path("snowie"))
    ms = _ms(2, 2)
    a_kaz = _agent(met=kaz)._eval_new_game(ms.after_game(1, True), 0)
    a_sno = _agent(met=snowie)._eval_new_game(ms.after_game(1, True), 0)
    assert a_kaz == pytest.approx(0.67736, abs=1e-5)
    assert a_sno == pytest.approx(0.68500, abs=1e-5)
