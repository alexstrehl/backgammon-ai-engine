"""Tests for --warm-start-matchplay-cubeful (cubeful money -> matchplay).

The load-bearing claim is that the cubeful money encoder's 200 columns
are a prefix of the matchplay encoders' layout. MatchplayEncoder's
docstring asserts this; `test_cubeful_columns_are_a_prefix` checks it,
and `test_warm_started_net_reproduces_source` checks the consequence
end to end.
"""

import argparse

import numpy as np
import pytest
import torch

from backgammon_engine import BoardState
from encoding import (
    CubefulEncoder, CubePerspective, MatchplayCube1HEncoder, MatchplayEncoder,
)
from model import TDNetwork
from train_cli import _warm_start_matchplay_cubeful, build_network


CUBE_STATES = (
    CubePerspective.CENTERED, CubePerspective.MINE, CubePerspective.THEIRS,
)


def _positions():
    s = BoardState.initial()
    out = [s]
    rng = np.random.default_rng(0)
    from backgammon_engine import get_legal_plays
    for _ in range(6):
        dice = (int(rng.integers(1, 7)), int(rng.integers(1, 7)))
        plays = get_legal_plays(s, dice)
        if not plays:
            break
        s = plays[int(rng.integers(0, len(plays)))][1]
        if s.is_game_over():
            break
        out.append(s)
    return out


# ── the prefix claim ────────────────────────────────────────────────


@pytest.mark.parametrize("cube_state", CUBE_STATES)
@pytest.mark.parametrize("is_cube_action", [False, True])
def test_cubeful_columns_are_a_prefix(cube_state, is_cube_action):
    """The 200 cubeful-money columns must equal the first 200 columns of
    the matchplay encoders, for every board / cube perspective."""
    cf = CubefulEncoder()
    mp1h = MatchplayCube1HEncoder()
    mp = MatchplayEncoder()
    for state in _positions():
        a = cf.encode(state, cube_state, is_cube_action=is_cube_action)
        for enc in (mp1h, mp):
            b = enc.encode(
                state, cube_state, cube_value=1, my_away=3, opp_away=3,
                is_crawford=False, is_cube_action=is_cube_action,
            )
            assert a.shape[0] == 200
            np.testing.assert_array_equal(a, b[:200])


def test_prefix_holds_regardless_of_score():
    """Columns 0..199 must not depend on score — otherwise the copy
    would be transferring score-contaminated weights."""
    mp1h = MatchplayCube1HEncoder()
    state = BoardState.initial()
    ref = mp1h.encode(state, CubePerspective.CENTERED, 1, 3, 3, False)[:200]
    for my_away, opp_away, cube, crawford in [
        (1, 2, 1, True), (2, 3, 2, False), (5, 1, 4, False),
    ]:
        got = mp1h.encode(
            state, CubePerspective.CENTERED, cube, my_away, opp_away, crawford
        )[:200]
        np.testing.assert_array_equal(ref, got)


# ── the warm start ──────────────────────────────────────────────────


@pytest.fixture
def source(tmp_path):
    net = TDNetwork(
        hidden_sizes=[16, 12], input_size=200, encoder_name="cubeful_perspective196",
        output_mode="equity",
    )
    with torch.no_grad():  # make every column distinctive
        for p in net.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    path = tmp_path / "cubeful_money.pt"
    net.save(str(path))
    return net, str(path)


def test_shapes_and_metadata(source):
    src, path = source
    net = _warm_start_matchplay_cubeful(path)
    assert net.input_size == 207
    assert net.encoder_name == "matchplay1h_perspective196"
    assert net.output_mode == "probability"
    assert net.hidden_sizes == src.hidden_sizes


def test_copies_shared_columns_and_zeroes_the_rest(source):
    src, path = source
    net = _warm_start_matchplay_cubeful(path)
    w_new = net.hidden_layers[0].weight.detach()
    w_old = src.hidden_layers[0].weight.detach()
    # 196 board + 4 cube-context columns copied verbatim...
    torch.testing.assert_close(w_new[:, :200], w_old[:, :200])
    # ...cube level (4-hot), away scores, crawford zeroed.
    assert torch.all(w_new[:, 200:] == 0)
    assert w_new.shape[1] == 207


def test_deeper_layers_and_head_carried_over(source):
    src, path = source
    net = _warm_start_matchplay_cubeful(path)
    for i in (1,):
        torch.testing.assert_close(
            net.hidden_layers[i].weight, src.hidden_layers[i].weight
        )
    torch.testing.assert_close(net.fc_output.weight, src.fc_output.weight)
    torch.testing.assert_close(net.fc_output.bias, src.fc_output.bias)
    torch.testing.assert_close(
        net.hidden_layers[0].bias, src.hidden_layers[0].bias
    )


def test_warm_started_net_reproduces_source(source):
    """End to end: the warm-started matchplay net must output exactly
    sigmoid(source equity) at every score, since the score columns are
    zero-weighted and the shared columns are identical."""
    src, path = source
    net = _warm_start_matchplay_cubeful(path)
    cf, mp1h = CubefulEncoder(), MatchplayCube1HEncoder()
    for state in _positions():
        for cube_state in CUBE_STATES:
            x_cf = torch.from_numpy(cf.encode(state, cube_state))
            want = torch.sigmoid(src(x_cf))
            for my_away, opp_away, cube in [(3, 3, 1), (1, 2, 2), (5, 4, 4)]:
                x_mp = torch.from_numpy(mp1h.encode(
                    state, cube_state, cube, my_away, opp_away, False
                ))
                torch.testing.assert_close(net(x_mp), want, atol=1e-6, rtol=0)


def test_zero_init_columns_are_not_dead(source):
    """Zero-initialised INPUT columns still receive gradient — unlike the
    width_expand dead-unit failure, where zeroed OUTGOING weights starved
    the incoming ones."""
    _, path = source
    net = _warm_start_matchplay_cubeful(path)
    mp1h = MatchplayCube1HEncoder()
    x = torch.from_numpy(mp1h.encode(
        BoardState.initial(), CubePerspective.CENTERED, 2, 3, 2, False
    ))
    net(x).backward()
    grad = net.hidden_layers[0].weight.grad
    # away-score columns (204, 205) are non-zero features -> live gradient
    assert torch.any(grad[:, 204] != 0)
    assert torch.any(grad[:, 205] != 0)
    # the active cube-level one-hot column (cube 2 -> index 201) too
    assert torch.any(grad[:, 201] != 0)


# ── CLI wiring ──────────────────────────────────────────────────────


def test_build_network_dispatches(source):
    _, path = source
    args = argparse.Namespace(
        resume=None, expand=None, expand_depth=None, hidden=None,
        output_mode=None, encoder=None,
        warm_start_equity=None, warm_start_cubeful=None,
        warm_start_matchplay=None,
        warm_start_matchplay1h=None, warm_start_matchplay_cubeful=path,
    )
    net = build_network(args)
    assert net.input_size == 207
    assert net.encoder_name == "matchplay1h_perspective196"


def test_mutual_exclusivity(source):
    _, path = source
    args = argparse.Namespace(
        resume=None, expand=None, expand_depth=None, hidden=None,
        output_mode=None, encoder=None,
        warm_start_equity=None, warm_start_cubeful=None,
        warm_start_matchplay="x.pt",
        warm_start_matchplay1h=None, warm_start_matchplay_cubeful=path,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_network(args)


def test_rejects_wrong_encoder(tmp_path):
    net = TDNetwork(hidden_sizes=[8], input_size=196,
                    encoder_name="perspective196", output_mode="probability")
    p = tmp_path / "dmp.pt"
    net.save(str(p))
    with pytest.raises(ValueError, match="cubeful_perspective196"):
        _warm_start_matchplay_cubeful(str(p))


def test_rejects_wrong_input_size(tmp_path):
    net = TDNetwork(hidden_sizes=[8], input_size=201,
                    encoder_name="cubeful_perspective196", output_mode="equity")
    p = tmp_path / "odd.pt"
    net.save(str(p))
    with pytest.raises(ValueError, match="200-input source"):
        _warm_start_matchplay_cubeful(str(p))


# ── the real production checkpoint ──────────────────────────────────


def test_real_4L_cubeful_money_model():
    """The actual intended source: best_models/cubeful_money_512_512_256_256.pt.
    Warm-starting it must give a 4-layer 207-input matchplay net."""
    import os
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "best_models", "cubeful_money_512_512_256_256.pt",
    )
    if not os.path.exists(path):
        pytest.skip("4L cubeful money model not present")
    net = _warm_start_matchplay_cubeful(path)
    assert net.hidden_sizes == [512, 512, 256, 256]
    assert net.input_size == 207
    assert net.output_mode == "probability"
    # Sane, non-degenerate outputs at the initial position.
    mp1h = MatchplayCube1HEncoder()
    x = torch.from_numpy(mp1h.encode(
        BoardState.initial(), CubePerspective.CENTERED, 1, 3, 3, False
    ))
    p = float(net(x))
    assert 0.3 < p < 0.7, p
