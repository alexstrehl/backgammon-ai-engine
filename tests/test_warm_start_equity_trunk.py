"""Tests for prob5's --warm-start-equity-trunk.

The copy must actually happen, and every way of getting it wrong must raise
rather than train from a random init.
"""
import argparse

import pytest
import torch

from model import ProbNetwork, TDNetwork
from train_prob5 import _build_network, _warm_start_equity_trunk

EQUITY_HIDDEN = [512, 512, 256, 256]
PROB5_HIDDEN = [512, 512, 256, 128]


@pytest.fixture
def equity_ckpt(tmp_path):
    """A cubeless-money equity checkpoint, saved the way TDNetwork saves."""
    net = TDNetwork(hidden_sizes=EQUITY_HIDDEN, input_size=196)
    path = tmp_path / "cm4L.pt"
    torch.save(
        {
            "hidden_sizes": EQUITY_HIDDEN,
            "input_size": 196,
            "activation": "relu",
            "encoder_name": "perspective196",
            "output_mode": "equity",
            "state_dict": net.state_dict(),
        },
        path,
    )
    return path, net


def _args(**kw):
    base = dict(resume=None, expand=None, expand_depth=None,
                expand_depth_size=None, warm_start_equity_trunk=None,
                hidden=None, activation="relu", raw_logits=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_copies_matching_layers_and_leaves_head_fresh(equity_ckpt):
    path, src = equity_ckpt
    fresh = ProbNetwork(hidden_sizes=PROB5_HIDDEN, input_size=196)
    net = _warm_start_equity_trunk(str(path), PROB5_HIDDEN, "relu", False)

    ssd, nsd = src.state_dict(), net.state_dict()
    # Layers 0-2 match in shape and must be copied bit-for-bit.
    for i in (0, 1, 2):
        for suf in ("weight", "bias"):
            assert torch.equal(
                nsd[f"trunk.{2 * i}.{suf}"], ssd[f"hidden_layers.{i}.{suf}"]
            ), f"layer {i}.{suf} was not copied"
    # Layer 3 mismatches (256 vs 128 out) and must stay fresh.
    assert nsd["trunk.6.weight"].shape == (128, 256)
    # The 5-output head must never take anything from a 1-output equity head.
    assert nsd["head.weight"].shape == (5, 128)
    assert "fc_output.weight" not in nsd


def test_head_is_random_not_copied(equity_ckpt):
    """Warm-starting twice must give identical trunks (deterministic copy)
    but different heads (freshly initialised, not copied)."""
    path, _ = equity_ckpt
    a = _warm_start_equity_trunk(str(path), PROB5_HIDDEN, "relu", False)
    b = _warm_start_equity_trunk(str(path), PROB5_HIDDEN, "relu", False)
    asd, bsd = a.state_dict(), b.state_dict()
    for i in (0, 1, 2):
        assert torch.equal(asd[f"trunk.{2 * i}.weight"],
                           bsd[f"trunk.{2 * i}.weight"])
    assert not torch.equal(asd["head.weight"], bsd["head.weight"])
    # The un-copied trunk layer is fresh too, so it must also differ.
    assert not torch.equal(asd["trunk.6.weight"], bsd["trunk.6.weight"])


def test_reports_expected_fraction(equity_ckpt, capsys):
    path, _ = equity_ckpt
    _warm_start_equity_trunk(str(path), PROB5_HIDDEN, "relu", False)
    out = capsys.readouterr().out
    # The summary must be loud enough that a silent no-op is impossible.
    assert "Warm-started trunk from EQUITY checkpoint" in out
    assert "COPIED" in out and "FRESH" in out
    assert "93.7%" in out


def test_rejects_a_prob5_checkpoint(tmp_path):
    p = tmp_path / "prob5.pt"
    ProbNetwork(hidden_sizes=PROB5_HIDDEN, input_size=196).save(str(p))
    with pytest.raises(ValueError, match="model_type|hidden_layers"):
        _warm_start_equity_trunk(str(p), PROB5_HIDDEN, "relu", False)


def test_raises_when_nothing_would_be_copied(equity_ckpt):
    """A fully incompatible --hidden must fail loudly, not train from scratch."""
    path, _ = equity_ckpt
    with pytest.raises(ValueError, match="ZERO layers"):
        _warm_start_equity_trunk(str(path), [64, 64], "relu", False)


def test_rejects_activation_mismatch(equity_ckpt):
    """Trunk weights are meaningless under a different activation. The
    mismatch must refuse, not silently reinterpret tanh weights as relu."""
    path, _ = equity_ckpt  # saved with activation="relu"
    with pytest.raises(ValueError, match="activation"):
        _warm_start_equity_trunk(str(path), PROB5_HIDDEN, "tanh", False)


def test_missing_activation_warns_not_raises(tmp_path, capsys):
    """Old checkpoints without an activation field stay loadable, loudly."""
    net = TDNetwork(hidden_sizes=EQUITY_HIDDEN, input_size=196)
    ckpt = {
        "hidden_sizes": EQUITY_HIDDEN,
        "input_size": 196,
        "encoder_name": "perspective196",
        "output_mode": "equity",
        "state_dict": net.state_dict(),
    }
    path = tmp_path / "old.pt"
    torch.save(ckpt, path)
    _warm_start_equity_trunk(str(path), PROB5_HIDDEN, "relu", False)
    assert "records no activation" in capsys.readouterr().out


def test_no_island_copy_after_a_mismatch(tmp_path):
    """A later layer that re-matches after a mismatch must NOT be copied.
    Layer dims chain, so a re-match implies the mismatch was at layer 0.
    That is the zero-copied case, which must raise."""
    src_hidden = [512, 256, 512, 512]
    net = TDNetwork(hidden_sizes=src_hidden, input_size=196)
    path = tmp_path / "island_src.pt"
    torch.save(
        {
            "hidden_sizes": src_hidden,
            "input_size": 196,
            "activation": "relu",
            "encoder_name": "perspective196",
            "output_mode": "equity",
            "state_dict": net.state_dict(),
        },
        path,
    )
    # Layer 0 mismatches (128 vs 512 out). Layers 2-3 coincidentally
    # re-match (512x256, 512x512).
    with pytest.raises(ValueError, match="ZERO layers"):
        _warm_start_equity_trunk(str(path), [128, 256, 512, 512], "relu",
                                 False)


def test_missing_file_raises(tmp_path):
    """A missing checkpoint file must raise, not train from random init."""
    with pytest.raises(Exception):
        _warm_start_equity_trunk(
            str(tmp_path / "does_not_exist.pt"), PROB5_HIDDEN, "relu", False,
        )


def test_build_network_requires_hidden(equity_ckpt):
    path, _ = equity_ckpt
    with pytest.raises(ValueError, match="requires --hidden"):
        _build_network(_args(warm_start_equity_trunk=str(path)))


def test_build_network_dispatches(equity_ckpt):
    path, src = equity_ckpt
    net = _build_network(_args(warm_start_equity_trunk=str(path),
                               hidden="512,512,256,128"))
    assert net.hidden_sizes == PROB5_HIDDEN
    assert torch.equal(net.state_dict()["trunk.0.weight"],
                       src.state_dict()["hidden_layers.0.weight"])


def test_mutually_exclusive_with_resume(equity_ckpt):
    path, _ = equity_ckpt
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_network(_args(warm_start_equity_trunk=str(path),
                             resume=str(path), hidden="512,512,256,128"))
