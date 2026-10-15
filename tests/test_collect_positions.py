"""tools/collect_positions.py: mode derivation, npz keys, and rows
identical to the trainers' own collectors."""
import hashlib
import importlib.util
import os
import random

import numpy as np
import pytest

from model import ProbNetwork, TDNetwork
from modes import CubelessMoneyMode
from td_agent import TDAgent
from trainer import collect_episode
from train_prob5 import _collect_one_episode, _resolve_engine_fns

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tools", "collect_positions.py")
_spec = importlib.util.spec_from_file_location("collect_positions", _PATH)
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)

SEED = 3


def _nets():
    return {
        "dmp": TDNetwork(hidden_sizes=[8], output_mode="probability"),
        "cubeless-money": TDNetwork(hidden_sizes=[8], output_mode="equity"),
        "cubeful-money": TDNetwork(hidden_sizes=[8], output_mode="equity",
                                   encoder_name="cubeful_perspective196"),
        "cubeless-prob5": ProbNetwork(hidden_sizes=[8]),
    }


def _run(tmp_path, net, name, *extra):
    model = str(tmp_path / f"{name}.pt")
    net.save(model)
    out = str(tmp_path / f"{name}.npz")
    cp.main(["--model", model, "--games", "2", "--workers", "1", "--seed", str(SEED),
             "--out", out, *extra])
    return model, np.load(out)


def test_derive_game_mode():
    for mode, net in _nets().items():
        assert cp.derive_game_mode(net) == mode


def test_check_game_mode_rejects_mismatches():
    nets = _nets()
    with pytest.raises(SystemExit):
        cp.check_game_mode(nets["cubeless-prob5"], "cubeful-money")
    with pytest.raises(SystemExit):
        cp.check_game_mode(nets["cubeless-money"], "cubeful-money")
    with pytest.raises(SystemExit):
        cp.check_game_mode(nets["dmp"], "cubeless-money")
    cp.check_game_mode(nets["cubeless-money"], "dmp")  # equity net may play dmp


def test_cubeful_bank_keys_and_width(tmp_path):
    model, d = _run(tmp_path, _nets()["cubeful-money"], "cf")
    assert set(d.files) == {"enc", "model", "model_md5", "game_mode", "jacoby",
                            "oneply", "seed", "games", "argv"}
    assert d["enc"].dtype == np.float32 and d["enc"].shape[1] == 200
    assert str(d["game_mode"]) == "cubeful-money" and bool(d["jacoby"])
    assert str(d["model"]) == "cf.pt"
    assert str(d["model_md5"]) == hashlib.md5(open(model, "rb").read()).hexdigest()
    with pytest.raises(SystemExit):  # never overwrites
        cp.main(["--model", model, "--games", "1", "--seed", "1", "--out", str(tmp_path / "cf.npz")])


def test_cubeless_rows_match_trainer_collector(tmp_path):
    net = _nets()["cubeless-money"]
    _, d = _run(tmp_path, net, "cbl", "--targets")
    rng = random.Random(SEED * 100_003)
    agent = TDAgent(net)
    encs, tgts = zip(*(collect_episode(agent, CubelessMoneyMode(), rng) for _ in range(2)))
    assert np.array_equal(d["enc"], np.concatenate(encs))
    assert np.array_equal(d["tgt"], np.concatenate(tgts))
    assert d["enc"].shape[1] == 196 and d["tgt"].ndim == 1


def test_prob5_rows_match_trainer_collector(tmp_path):
    net = _nets()["cubeless-prob5"]
    _, d = _run(tmp_path, net, "p5", "--targets")
    rng = random.Random(SEED * 100_003)
    fns = _resolve_engine_fns(net.encoder_name)
    encs, tgts = zip(*(_collect_one_episode(net, rng, *fns, oneply=False, device="cpu")
                       for _ in range(2)))
    assert np.array_equal(d["enc"], np.concatenate(encs))
    assert np.array_equal(d["tgt"], np.concatenate(tgts))
    assert d["tgt"].shape == (len(d["enc"]), 5)
