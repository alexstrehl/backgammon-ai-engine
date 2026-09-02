"""--td-move-eps: self-consistency rows for near-tied successors.

The property that matters is that these rows are ADDITIVE: they add
supervision at states the collector already evaluated without changing
which move is played. Random exploration was measured harmful precisely
because it moved the state distribution; this must not.
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


def test_cube_targets_1ply_changes_only_cube_rows(agent):
    """--cube-targets-1ply must re-price the cube rows and leave every
    checker row byte-identical: it exists to make the two channels share a
    depth, not to alter the checker target."""
    import numpy as np
    mode = MatchplayMode(match_length=11)
    ms = MatchState(jacoby=False, match_length=11, my_away=11, opp_away=11)
    e0, t0 = T._collect_episode_matchplay(
        agent, mode, random.Random(4), ms, oneply=True,
        cube_targets_1ply=False)[:2]
    e1, t1 = T._collect_episode_matchplay(
        agent, mode, random.Random(4), ms, oneply=True,
        cube_targets_1ply=True)[:2]
    assert len(t0) == len(t1)
    assert np.array_equal(np.asarray(e0), np.asarray(e1)), \
        "encodings must be unchanged -- only target VALUES may move"
    changed = [i for i in range(len(t0)) if t0[i] != t1[i]]
    assert changed, "flag had no effect"
    # every changed row must be a cube row (is_cube_action set in extras)
    n_base = agent.encoder._base.num_features
    for i in changed:
        assert np.asarray(e0[i])[n_base + 3] == 1.0, \
            f"row {i} changed but is not a cube-action row"


def test_cube_targets_1ply_needs_oneply():
    """0-ply checker rows with 1-ply cube rows would be the same mixed-depth
    defect in reverse."""
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "train_batch.py", "--game-mode", "matchplay",
         "--match-length", "11", "--cube-targets-1ply",
         "--resume", MODEL, "--num-episodes", "1", "--save", "/tmp/x.pt"],
        capture_output=True, text=True, cwd=".")
    assert "needs --oneply" in (r.stderr + r.stdout)
