"""Phase 3: the 13 MET input features (220-input encoder)."""

import argparse

import numpy as np
import pytest
import torch

from backgammon_engine import BoardState
from encoding import (
    CubePerspective, MatchplayCube1HEncoder, MatchplayEncoder,
    MatchplayMETEncoder, get_encoder,
)
from met import MatchEquityTable, DEFAULT_MET, N_MET_FEATURES
from model import TDNetwork
from modes import MatchState
from train_cli import _warm_start_matchplay_met, build_network


@pytest.fixture(scope="module")
def table():
    return MatchEquityTable.from_xml(DEFAULT_MET)


@pytest.fixture(scope="module")
def enc():
    return MatchplayMETEncoder()


# ── the crawford_done derivation the block depends on ───────────────


def test_crawford_done_equals_someone_is_one_away():
    """MatchplayMETEncoder derives crawford_done as "someone is 1-away"
    because the encoder is not handed the latch. Away scores never
    increase and MatchState latches the flag the first time start_game
    sees a 1-away score, so the two must coincide on every state
    reachable by playing games out.
    """
    checked = 0
    for match_len in (3, 5, 7):
        frontier = [MatchState(
            match_length=match_len, my_away=match_len, opp_away=match_len,
        ).start_game()]
        seen = set()
        while frontier:
            ms = frontier.pop()
            key = (ms.my_away, ms.opp_away, ms.crawford_done, ms.is_crawford)
            if key in seen:
                continue
            seen.add(key)
            assert ms.crawford_done == (ms.my_away == 1 or ms.opp_away == 1), key
            checked += 1
            for cube in (1, 2, 4):
                for result in (1, 2, 3):
                    for me in (True, False):
                        nxt = MatchState(
                            match_length=match_len, my_away=ms.my_away,
                            opp_away=ms.opp_away, cube_value=cube,
                            is_crawford=ms.is_crawford,
                            crawford_done=ms.crawford_done,
                        ).after_game(result, me)
                        if not nxt.match_over():
                            frontier.append(nxt.start_game())
    assert checked > 30


# ── shape and content ───────────────────────────────────────────────


def test_shape_and_name(enc):
    assert enc.num_features == 220
    assert enc.name == "matchplaymet_perspective196"
    assert get_encoder("matchplaymet_perspective196").num_features == 220


def test_is_recognised_as_matchplay(enc):
    assert isinstance(enc, MatchplayEncoder)
    assert isinstance(enc, MatchplayCube1HEncoder)


def test_first_207_columns_match_the_1h_encoder(enc):
    """The MET block is strictly appended — the 1h layout must be a
    prefix, or the 207 -> 220 warm start silently misaligns."""
    mp1h = MatchplayCube1HEncoder()
    state = BoardState.initial()
    for cube_state in CubePerspective:
        for cube in (1, 2, 4):
            for my, opp, cf in [(3, 3, False), (1, 2, True), (2, 1, False)]:
                a = mp1h.encode(state, cube_state, cube, my, opp, cf)
                b = enc.encode(state, cube_state, cube, my, opp, cf)
                np.testing.assert_array_equal(a, b[:207])


def test_block_values_are_centred_met_lookups(enc, table):
    state = BoardState.initial()
    for my, opp, cube, cf in [(3, 3, 1, False), (2, 3, 2, False),
                              (1, 2, 1, True), (2, 1, 1, False)]:
        x = enc.encode(state, CubePerspective.CENTERED, cube, my, opp, cf)
        want = table.outcome_block(
            my, opp, (my == 1 or opp == 1), cf, cube,
        )
        np.testing.assert_allclose(x[207:], want - 0.5, atol=1e-6)
        assert x[207:].shape == (N_MET_FEATURES,)


def test_is_crawford_does_not_affect_the_block(enc):
    """All 13 values are FUTURE-game MWCs, and mwc_after only consults
    is_crawford through `crawford_done or is_crawford`. Since
    crawford_done is derived as "someone is 1-away", and is_crawford
    can only be set when someone is 1-away, that disjunction is already
    True whenever is_crawford could be. So the block is a function of
    (my_away, opp_away, cube_value) alone — by the time any of these
    games starts, Crawford is necessarily spent.
    """
    state = BoardState.initial()
    for my, opp in [(2, 1), (1, 2), (3, 1), (1, 3)]:
        a = enc.encode(state, CubePerspective.CENTERED, 1, my, opp, True)
        b = enc.encode(state, CubePerspective.CENTERED, 1, my, opp, False)
        np.testing.assert_array_equal(a[207:], b[207:])


def test_crawford_is_scheduled_on_the_transition_into_one_away(enc, table):
    """Where the Crawford logic does real work: from 3-away/3-away a
    gammon lands on 1-away/3-away with Crawford still to come, so the
    value is the PRE-Crawford 0.75076, not the post-Crawford one."""
    x = enc.encode(BoardState.initial(), CubePerspective.CENTERED,
                   1, 3, 3, False)
    win_gammon = x[207 + 1] + 0.5
    assert win_gammon == pytest.approx(table.mwc(1, 3), abs=1e-6)
    assert win_gammon == pytest.approx(0.75076, abs=1e-5)
    assert win_gammon != pytest.approx(
        table.mwc(1, 3, crawford_done=True), abs=1e-4)


def test_drop_column_is_the_canary(enc, table):
    """Column 219 at 2-away/2-away cube 1 is the 0.680 -> 0.639 value."""
    x = enc.encode(BoardState.initial(), CubePerspective.CENTERED, 1, 2, 2, False)
    assert x[219] == pytest.approx(table.mwc(1, 2) - 0.5, abs=1e-6)


def test_block_varies_with_cube_and_score(enc):
    state = BoardState.initial()
    base = enc.encode(state, CubePerspective.CENTERED, 1, 3, 3, False)[207:]
    for cube in (2, 4):
        other = enc.encode(state, CubePerspective.CENTERED, cube, 3, 3, False)[207:]
        assert not np.allclose(base, other)
    for my, opp in [(2, 3), (1, 3), (3, 2)]:
        other = enc.encode(state, CubePerspective.CENTERED, 1, my, opp, False)[207:]
        assert not np.allclose(base, other)


def test_block_is_independent_of_board(enc):
    """MET features are a pure score/cube function — they must not
    respond to the board at all."""
    from backgammon_engine import get_legal_plays
    s = BoardState.initial()
    ref = enc.encode(s, CubePerspective.CENTERED, 1, 3, 3, False)[207:]
    s2 = get_legal_plays(s, (3, 1))[0][1]
    got = enc.encode(s2, CubePerspective.MINE, 1, 3, 3, False)[207:]
    np.testing.assert_array_equal(ref, got)


def test_saturation_at_three_points(enc):
    """Documented expectation: at 3 points with cube 2 the win branches
    already end the match, so those columns pin to +0.5."""
    x = enc.encode(BoardState.initial(), CubePerspective.CENTERED, 2, 2, 3, False)
    # double-take block win columns (indices 213..215 absolute)
    np.testing.assert_allclose(x[213:216], 0.5, atol=1e-6)


# ── warm start 207 -> 220 ───────────────────────────────────────────


@pytest.fixture
def source_207(tmp_path):
    torch.manual_seed(3)
    net = TDNetwork(
        hidden_sizes=[16, 12], input_size=207,
        encoder_name="matchplay1h_perspective196", output_mode="probability",
    )
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    path = tmp_path / "mp1h.pt"
    net.save(str(path))
    return net, str(path)


def test_warm_start_shapes(source_207):
    src, path = source_207
    net = _warm_start_matchplay_met(path)
    assert net.input_size == 220
    assert net.encoder_name == "matchplaymet_perspective196"
    assert net.hidden_sizes == src.hidden_sizes


def test_warm_start_is_function_preserving(source_207, enc):
    """220-input net must produce bit-identical output to the 207-input
    source at every score, since the new columns are zero-weighted."""
    src, path = source_207
    net = _warm_start_matchplay_met(path)
    mp1h = MatchplayCube1HEncoder()
    state = BoardState.initial()
    for cube_state in CubePerspective:
        for cube in (1, 2, 4):
            for my, opp, cf in [(3, 3, False), (1, 2, True), (2, 3, False)]:
                x_old = torch.from_numpy(
                    mp1h.encode(state, cube_state, cube, my, opp, cf))
                x_new = torch.from_numpy(
                    enc.encode(state, cube_state, cube, my, opp, cf))
                torch.testing.assert_close(
                    net(x_new), src(x_old), atol=1e-7, rtol=0)


def test_warm_start_zeroes_only_the_new_columns(source_207):
    src, path = source_207
    net = _warm_start_matchplay_met(path)
    w = net.hidden_layers[0].weight.detach()
    torch.testing.assert_close(w[:, :207], src.hidden_layers[0].weight.detach())
    assert torch.all(w[:, 207:] == 0)


def test_met_columns_take_gradient(source_207, enc):
    _, path = source_207
    net = _warm_start_matchplay_met(path)
    x = torch.from_numpy(enc.encode(
        BoardState.initial(), CubePerspective.CENTERED, 1, 2, 3, False))
    net(x).backward()
    g = net.hidden_layers[0].weight.grad[:, 207:]
    assert torch.any(g != 0)


def test_rejects_wrong_source(tmp_path):
    net = TDNetwork(hidden_sizes=[8], input_size=200,
                    encoder_name="cubeful_perspective196", output_mode="equity")
    p = tmp_path / "cf.pt"
    net.save(str(p))
    with pytest.raises(ValueError, match="matchplay1h_perspective196"):
        _warm_start_matchplay_met(str(p))


# ── CLI ─────────────────────────────────────────────────────────────


def _args(**kw):
    base = dict(
        resume=None, expand=None, expand_depth=None, hidden=None,
        output_mode=None, encoder=None, met_features=False,
        warm_start_equity=None, warm_start_cubeful=None,
        warm_start_matchplay=None,
        warm_start_matchplay1h=None, warm_start_matchplay_cubeful=None,
        warm_start_matchplay_met=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_warm_start_met(source_207):
    _, path = source_207
    net = build_network(_args(warm_start_matchplay_met=path))
    assert net.input_size == 220


def test_cli_met_features_flag_from_cubeful():
    import os
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "best_models", "cubeful_money_512_512_256_256.pt")
    if not os.path.exists(src):
        pytest.skip("4L cubeful money model not present")
    net = build_network(
        _args(warm_start_matchplay_cubeful=src, met_features=True))
    assert net.input_size == 220
    assert net.encoder_name == "matchplaymet_perspective196"
    assert net.hidden_sizes == [512, 512, 256, 256]
