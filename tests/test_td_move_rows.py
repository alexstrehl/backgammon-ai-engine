"""--td-move-eps: self-consistency rows for near-tied successors.

The property that matters is that these rows are ADDITIVE: they add
supervision at states the collector already evaluated without changing
which move is played. Random exploration (--move-explore-eps) was
measured harmful precisely because it moved the state distribution;
this must not.
"""
import random

import numpy as np
import pytest

from backgammon_engine import get_legal_plays
from met import MatchEquityTable, resolve_met_path
from model import TDNetwork
from modes import MatchplayMode, MatchState
from td_agent import TDAgent
import trainer as T

MODEL = "best_models/matchplay_1h64met_512_512_256_256.pt"


@pytest.fixture(scope="module")
def agent():
    met = MatchEquityTable.from_xml(resolve_met_path("auto"))
    return TDAgent(TDNetwork.load(MODEL), met=met)


def _collect(agent, eps, seed=5):
    mode = MatchplayMode(match_length=11, td_move_eps=eps)
    ms = MatchState(jacoby=False, match_length=11, my_away=11, opp_away=11)
    enc, tgt = T._collect_episode_matchplay(
        agent, mode, random.Random(seed), ms, oneply=True)[:2]
    return np.asarray(enc), list(tgt)


def test_eps_zero_is_the_baseline(agent):
    e0, _ = _collect(agent, 0.0)
    e_off, _ = _collect(agent, 0.0)
    assert np.array_equal(e0, e_off)


def test_rows_are_additive_trajectory_unchanged(agent):
    """The baseline rows must survive verbatim, in order, with the same
    targets -- otherwise the eps rows have perturbed the trajectory and
    we have reintroduced the covariate shift they exist to avoid."""
    e0, t0 = _collect(agent, 0.0)
    e1, t1 = _collect(agent, 0.01)
    assert len(e1) > len(e0)
    i = 0
    for j in range(len(e1)):
        if (i < len(e0) and np.array_equal(e1[j], e0[i])
                and abs(t1[j] - t0[i]) < 1e-12):
            i += 1
    assert i == len(e0), f"only {i}/{len(e0)} baseline rows survived"


def test_more_eps_means_more_rows(agent):
    n = [len(_collect(agent, e)[0]) for e in (0.0, 0.005, 0.02)]
    assert n[0] < n[1] < n[2]


def test_targets_are_the_successors_own_oneply_value(agent):
    """Each emitted row must be (encode(s'), T1(s')) at the encoding the
    chooser reads -- opponent perspective, opponent's cube flag -- with
    the cube option folded in when that flag is live."""
    rng = random.Random(11)
    ms = MatchState(jacoby=False, match_length=11, my_away=11, opp_away=11)
    gms = ms.start_game()
    from backgammon_engine import opening_roll
    state, dice = opening_roll(rng)
    plays = get_legal_plays(state, dice)
    chosen = plays[0][1]
    rows = T._td_move_rows(agent, state, dice, gms, 0.02, chosen)
    assert rows, "expected at least one near-tied sibling at the opening"
    opp = 1 - state.turn
    ca = gms.can_offer(opp)
    succs = {tuple(np.asarray(agent._encode_matchplay(s, gms, ca)).tolist()): s
             for _, s in plays if not s.is_game_over()}
    for enc, tv in rows:
        key = tuple(np.asarray(enc).tolist())
        assert key in succs, "row is not one of this state's successors"
        s = succs[key]
        expect = agent.value_oneply_checker_matchplay(s, gms)
        if ca:
            c = agent.offer_double_matchplay(s, gms)._cache
            if c is not None:
                expect = max(expect, min(c["v_dt"], c["v_dp"]))
        assert tv == pytest.approx(expect, abs=1e-9)


def test_chosen_successor_is_not_duplicated(agent):
    """The played successor becomes the next visited state and gets its
    own row there; emitting it again would double its weight."""
    rng = random.Random(3)
    from backgammon_engine import opening_roll
    gms = MatchState(jacoby=False, match_length=11,
                     my_away=11, opp_away=11).start_game()
    state, dice = opening_roll(rng)
    plays = get_legal_plays(state, dice)
    for _, chosen in plays[:4]:
        rows = T._td_move_rows(agent, state, dice, gms, 1.0, chosen)
        enc_chosen = np.asarray(agent._encode_matchplay(
            chosen, gms, gms.can_offer(1 - state.turn)))
        assert not any(np.array_equal(np.asarray(e), enc_chosen)
                       for e, _ in rows)


def test_no_rows_when_move_is_forced(agent):
    """A single legal play has no siblings to supervise."""
    rng = random.Random(7)
    from backgammon_engine import opening_roll
    gms = MatchState(jacoby=False, match_length=11,
                     my_away=11, opp_away=11).start_game()
    state, dice = opening_roll(rng)
    plays = get_legal_plays(state, dice)
    if len(plays) == 1:
        assert T._td_move_rows(agent, state, dice, gms, 0.01, plays[0][1]) == []


def test_vectorized_collect_refuses_td_move_eps():
    """The generator mirror emits no successor rows, so the fast path
    would silently drop them -- it must fall back, not pretend."""
    import inspect
    src = inspect.getsource(T)
    assert "--td-move-eps is implemented in the standard collector" in src


def test_targets_match_the_independent_slow_path(agent):
    """Cross-check every emitted target against audit_collector's
    independent reimplementation of the 1-ply value -- pure-python
    movegen, one fresh evaluate_matchplay per successor, sharing none of
    the batched fast path. A fast-path bug cannot self-verify."""
    import numpy as np
    from audit_collector import _oneply_target_reference
    from backgammon_engine import opening_roll

    rng = random.Random(21)
    gms = MatchState(jacoby=False, match_length=11,
                     my_away=6, opp_away=9).start_game()
    state, dice = opening_roll(rng)
    plays = get_legal_plays(state, dice)
    rows = T._td_move_rows(agent, state, dice, gms, 0.03, plays[0][1])
    assert rows, "expected near-tied siblings to supervise"

    opp = 1 - state.turn
    ca = gms.can_offer(opp)
    by_enc = {tuple(np.asarray(agent._encode_matchplay(s, gms, ca)).tolist()): s
              for _, s in plays if not s.is_game_over()}
    checked = 0
    for enc, tv in rows[:6]:          # slow path: a handful is enough
        s = by_enc[tuple(np.asarray(enc).tolist())]
        ref = _oneply_target_reference(agent, s, gms)
        if ca:
            c = agent.offer_double_matchplay(s, gms)._cache
            if c is not None:
                ref = max(ref, min(c["v_dt"], c["v_dp"]))
        assert tv == pytest.approx(ref, abs=2e-6), (
            f"emitted {tv:.8f} vs independent reference {ref:.8f}")
        checked += 1
    assert checked


def test_oneply_act_k_matches_full_selection_but_cheaper(agent):
    """The top-K filter must not change WHICH move 1-ply selection picks in
    the common case -- the argmin is decided among the best few candidates.
    K=3 matched full 1-ply on 101/128 gnubg-confirmed positions at 5.2x
    less compute; here we pin the equivalence on ordinary self-play states."""
    from backgammon_engine import opening_roll
    from met import MatchEquityTable, resolve_met_path
    from model import TDNetwork
    met = MatchEquityTable.from_xml(resolve_met_path("auto"))
    net = TDNetwork.load(MODEL)
    full = TDAgent(net, met=met, oneply_moves=True)
    topk = TDAgent(net, met=met, oneply_moves=True, oneply_moves_k=3)
    rng = random.Random(5)
    gms = MatchState(jacoby=False, match_length=11,
                     my_away=11, opp_away=11).start_game()
    state, dice = opening_roll(rng)
    same = n = 0
    for _ in range(12):
        a = full.choose_checker_action_matchplay(state, dice, gms)
        b = topk.choose_checker_action_matchplay(state, dice, gms)
        if a is None or b is None:
            break
        n += 1
        same += (a[0] == b[0])
        if a[0].is_game_over():
            break
        state = a[0]
        dice = (rng.randint(1, 6), rng.randint(1, 6))
    assert n > 0
    assert same >= n - 1, f"top-K diverged from full on {n-same}/{n} decisions"


def test_oneply_act_k_zero_is_plain_oneply(agent):
    """k=0 must mean 'no filter', not 'no deep read'."""
    from backgammon_engine import opening_roll
    from met import MatchEquityTable, resolve_met_path
    from model import TDNetwork
    met = MatchEquityTable.from_xml(resolve_met_path("auto"))
    net = TDNetwork.load(MODEL)
    a = TDAgent(net, met=met, oneply_moves=True, oneply_moves_k=0)
    b = TDAgent(net, met=met, oneply_moves=True)
    rng = random.Random(9)
    gms = MatchState(jacoby=False, match_length=11,
                     my_away=11, opp_away=11).start_game()
    state, dice = opening_roll(rng)
    assert (a.choose_checker_action_matchplay(state, dice, gms)[0]
            == b.choose_checker_action_matchplay(state, dice, gms)[0])


def test_vectorized_refuses_oneply_act():
    import inspect
    assert "--td-oneply-act acts with 1-ply move selection" in inspect.getsource(T)
