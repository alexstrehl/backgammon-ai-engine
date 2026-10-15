"""Parity tests for the CUBELESS branch of the vectorized GPU collector.

With the same seed and the same CPU net, the generator
(_collect_episode_cubeless_money_gen) driven by hand must reproduce
trainer.collect_episode's (encodings, targets) bit-for-bit, including
the discard decision. The cubeful generator is covered by
tests/test_vectorized_collect.py.

DMPMode routes through the same cubeless generator branch
(mode.initial_match_state() is None), so a DMP case is included.
"""
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from model import TDNetwork
from modes import CubelessMoneyMode, DMPMode
from td_agent import TDAgent
import trainer as T
import vectorized_collect as V


def _net(output_mode):
    torch.manual_seed(0)
    net = TDNetwork(
        hidden_sizes=[32, 16], input_size=196, output_mode=output_mode,
        encoder_name="perspective196",
    )
    net.eval()
    return net


def _drive_one(net, mode, seed, cap=1500):
    """Run a single cubeless generator game with a hand-driven CPU forward."""
    agent = TDAgent(net, oneply=False, boltzmann_temp=0.0, device="cpu")
    g = V._collect_episode_cubeless_money_gen(
        agent, mode, random.Random(seed), max_episode_plies=cap,
    )
    try:
        feat = next(g)
        with torch.no_grad():
            while True:
                out = net(torch.from_numpy(feat)).reshape(-1)
                feat = g.send(out.numpy())
    except StopIteration as e:
        return e.value


class TestCubelessGeneratorParity:
    def _run(self, mode, output_mode, n_seeds=6, min_compared=3):
        net = _net(output_mode)
        n_compared = 0
        for seed in range(n_seeds):
            enc_v, tgt_v, disc = _drive_one(net, mode, seed)
            agent = TDAgent(net, oneply=False, boltzmann_temp=0.0,
                            device="cpu")
            enc_r, tgt_r = T.collect_episode(
                agent, mode, random.Random(seed), oneply=True,
                max_episode_plies=1500,
            )
            assert disc == (enc_r.shape[0] == 0), (
                f"seed {seed}: discard decision differs "
                f"(generator={disc}, reference={enc_r.shape[0] == 0})")
            assert enc_v.shape == enc_r.shape, f"seed {seed}: shape"
            assert np.array_equal(enc_v, enc_r), f"seed {seed}: encodings"
            assert np.array_equal(tgt_v, tgt_r), f"seed {seed}: targets"
            if not disc:
                n_compared += 1
        assert n_compared >= min_compared, (
            f"only {n_compared} non-discarded episodes compared -- "
            f"near-vacuous")

    def test_cubeless_money_equity_bitwise(self):
        """The release target: cubeless-money equity."""
        self._run(CubelessMoneyMode(), "equity")

    def test_dmp_probability_bitwise(self):
        """DMP rides the same cubeless generator branch."""
        self._run(DMPMode(), "probability")
