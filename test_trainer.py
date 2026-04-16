"""Tests for trainer.py: episode collection, Trainer class."""

import math
import random

import numpy as np
import pytest
import torch

from backgammon_engine import BoardState
from model import TDNetwork
from modes import DMPMode, CubelessMoneyMode
from td_agent import TDAgent
from trainer import Trainer, collect_episode


# ── collect_episode ───────────────────────────────────────────────────


class TestCollectEpisode:
    def test_money_equity_returns_arrays(self):
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        rng = random.Random(0)
        encs, targets = collect_episode(agent, CubelessMoneyMode(), rng)
        assert isinstance(encs, np.ndarray)
        assert isinstance(targets, np.ndarray)
        assert encs.dtype == np.float32
        assert targets.dtype == np.float32
        assert encs.ndim == 2 and encs.shape[1] == 196
        assert encs.shape[0] == targets.shape[0]
        assert encs.shape[0] > 5

    def test_dmp_probability_terminal_in_range(self):
        net = TDNetwork(hidden_sizes=[16], output_mode="probability")
        agent = TDAgent(net)
        rng = random.Random(1)
        encs, targets = collect_episode(agent, DMPMode(), rng)
        assert (targets >= 0.0).all() and (targets <= 1.0).all()
        assert targets[-1] == 1.0

    def test_money_equity_terminal_target(self):
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        rng = random.Random(2)
        encs, targets = collect_episode(agent, CubelessMoneyMode(), rng)
        assert targets[-1] in (1.0, 2.0, 3.0)


# ── Trainer primitives ────────────────────────────────────────────────


class TestTrainerPrimitives:
    def test_train_step_decreases_loss_on_repeat(self):
        """A single batch trained on repeatedly should reduce its own loss."""
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        trainer = Trainer(agent, lr=1e-2)
        rng = random.Random(0)
        encs, targets = collect_episode(agent, CubelessMoneyMode(), rng)
        first = trainer.train_step(encs, targets)
        for _ in range(20):
            trainer.train_step(encs, targets)
        last = trainer.train_step(encs, targets)
        assert last < first, f"loss did not decrease: first={first} last={last}"

    def test_eval_loss_does_not_step(self):
        """eval_loss must not modify network weights."""
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        trainer = Trainer(agent, lr=1e-2)
        rng = random.Random(0)
        encs, targets = collect_episode(agent, CubelessMoneyMode(), rng)
        before = [p.clone() for p in net.parameters()]
        for _ in range(5):
            trainer.eval_loss(encs, targets)
        after = list(net.parameters())
        for b, a in zip(before, after):
            assert torch.equal(b, a), "eval_loss must not modify weights"

    def test_grad_clip_runs(self):
        """grad_clip=1.0 path should run without crashing."""
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        trainer = Trainer(agent, lr=1e-2, grad_clip=1.0)
        rng = random.Random(0)
        encs, targets = collect_episode(agent, CubelessMoneyMode(), rng)
        loss = trainer.train_step(encs, targets)
        assert math.isfinite(loss)


# ── value_oneply_checker terminal handling (equity mode) ─────────────


class TestValueOneplyCheckerEquityTerminals:
    """Equity-mode 1-ply lookahead must distinguish single / gammon /
    backgammon wins among the terminal successors it enumerates.
    Probability mode collapses them all to 1.0 so it doesn't need
    this differentiation."""

    @staticmethod
    def _near_bear_off(black_off: int, black_in_white_home: bool) -> BoardState:
        """Build a synthetic position where WHITE has 14 of 15 borne
        off and one checker on point 1 — every dice roll bears that
        last checker off and ends the game with WHITE winning. The
        kind of win depends on BLACK's borne-off count and whether
        BLACK still has any checkers in WHITE's home board:
          - black_off=1: single (1)
          - black_off=0, no checker in white's home: gammon (2)
          - black_off=0, checker in white's home: backgammon (3)
        """
        state = BoardState(
            points=[0] * 24, bar=[0, 0], off=[14, black_off], turn=0,
        )
        state.points[0] = 1  # white's last checker
        if black_in_white_home:
            state.points[5] = -1
            state.points[23] = -(15 - black_off - 1)
        else:
            state.points[23] = -(15 - black_off)
        return state

    def _eval_equity_oneply(self, state):
        # torch_seed pinned so the network init is deterministic; the
        # test only depends on terminal-detection logic, not the
        # network's actual numbers, but we want reproducibility.
        torch.manual_seed(0)
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        return agent.value_oneply_checker(state)

    def test_single_win_returns_one(self):
        state = self._near_bear_off(black_off=1, black_in_white_home=False)
        assert self._eval_equity_oneply(state) == pytest.approx(1.0)

    def test_gammon_win_returns_two(self):
        state = self._near_bear_off(black_off=0, black_in_white_home=False)
        assert self._eval_equity_oneply(state) == pytest.approx(2.0)

    def test_backgammon_win_returns_three(self):
        state = self._near_bear_off(black_off=0, black_in_white_home=True)
        assert self._eval_equity_oneply(state) == pytest.approx(3.0)

    def test_gammon_strictly_greater_than_single(self):
        single = self._eval_equity_oneply(
            self._near_bear_off(black_off=1, black_in_white_home=False)
        )
        gammon = self._eval_equity_oneply(
            self._near_bear_off(black_off=0, black_in_white_home=False)
        )
        backgammon = self._eval_equity_oneply(
            self._near_bear_off(black_off=0, black_in_white_home=True)
        )
        assert single < gammon < backgammon


# ── End-to-end training ───────────────────────────────────────────────


def _all_finite(xs):
    return all(math.isfinite(x) for x in xs)


@pytest.mark.slow
class TestEndToEndTraining:
    """Smoke: training runs end-to-end without crashing and produces
    finite losses. Deeper convergence checks are manual."""

    def test_train_dmp_probability(self):
        net = TDNetwork(hidden_sizes=[40], output_mode="probability")
        agent = TDAgent(net)
        trainer = Trainer(agent, lr=1e-3)
        losses = trainer.train(
            DMPMode(), num_episodes=200, batch_size=128, seed=20, log_every=0,
        )
        assert len(losses) > 0
        assert _all_finite(losses)

    def test_train_money_equity(self):
        net = TDNetwork(hidden_sizes=[40], output_mode="equity")
        agent = TDAgent(net)
        trainer = Trainer(agent, lr=1e-3)
        losses = trainer.train(
            CubelessMoneyMode(), num_episodes=200, batch_size=128, seed=21, log_every=0,
        )
        assert len(losses) > 0
        assert _all_finite(losses)

    def test_train_online_dmp_probability_with_sgd(self):
        net = TDNetwork(hidden_sizes=[40], output_mode="probability")
        agent = TDAgent(net)
        trainer = Trainer(agent, lr=0.1, optimizer_cls=torch.optim.SGD)
        losses = trainer.train_online(
            DMPMode(), num_episodes=50, seed=30, log_every=0,
        )
        assert len(losses) > 0
        assert _all_finite(losses)

    def test_train_online_money_equity_with_sgd(self):
        net = TDNetwork(hidden_sizes=[40], output_mode="equity")
        agent = TDAgent(net)
        # Equity targets (±1/2/3) need a smaller LR than probability
        # mode (targets in [0, 1]) to avoid divergence in online SGD.
        trainer = Trainer(agent, lr=0.01, optimizer_cls=torch.optim.SGD)
        losses = trainer.train_online(
            CubelessMoneyMode(), num_episodes=50, seed=31, log_every=0,
        )
        assert len(losses) > 0
        assert _all_finite(losses)
