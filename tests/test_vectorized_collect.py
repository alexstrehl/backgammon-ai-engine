"""Tests for vectorized_collect.py -- the lockstep GPU collector.

These run on CPU (the driver's `device` is a parameter), so they are
device-independent: the properties under test are algorithmic, not
numerical-backend properties.

The important ones:

  * `test_generator_matches_reference_bitwise` -- the generator mirror
    must compute exactly what trainer._collect_episode_cubeful_money
    computes. Driven one game at a time on the same CPU net with the
    same rng seed, every encoding and target must match BIT FOR BIT.
    This is what makes the vectorized path safe to train on.

  * `test_lockstep_batch_size_does_not_change_results` -- move-gen in
    bg_fast writes into shared C buffers, and a generator suspended at
    a `yield` has other games' move-gen running underneath it. If any
    buffer the generator reads after resuming were a view rather than a
    copy, B=8 would disagree with B=1. It must not.

  * the ply-guard tests -- the guard is the reason this port exists in
    a branch that has one; a lockstep runaway pins a driver slot and
    stalls the whole round, so it must fire per GAME.
"""

import random

import numpy as np
import pytest
import torch

from model import TDNetwork
from modes import CubefulMoneyMode
from td_agent import TDAgent
import trainer as T
import vectorized_collect as V


def _net():
    torch.manual_seed(0)
    net = TDNetwork(
        hidden_sizes=[32, 16], input_size=200, output_mode="equity",
        encoder_name="cubeful_perspective196",
    )
    net.eval()
    return net


def _worker_args(net, mode, num_episodes, seed, B, cap):
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    return (sd, [32, 16], 200, "equity", "cubeful_perspective196",
            "relu", mode, num_episodes, seed, True, False, 0.0, False,
            B, "cpu", cap)


def _drive_one(net, mode, seed, cap=1500):
    """Run a single generator game to completion with a CPU forward."""
    agent = TDAgent(net, oneply=False, boltzmann_temp=0.0, device="cpu")
    g = V._collect_episode_cubeful_money_gen(
        agent, mode, random.Random(seed), mode.initial_match_state(),
        max_episode_plies=cap,
    )
    try:
        feat = next(g)
        with torch.no_grad():
            while True:
                out = net(torch.from_numpy(feat)).reshape(-1)
                feat = g.send(out.numpy())
    except StopIteration as e:
        return e.value


class TestGeneratorParity:
    def test_generator_matches_reference_bitwise(self):
        """Generator vs reference collector, same seed, same CPU net.

        The small random fixture net plays degenerately, so a fair
        number of seeds hit the ply bound. That is itself worth
        asserting -- the two implementations must agree on the DISCARD
        DECISION as well as on the arrays -- but the test would be
        vacuous if every seed discarded, so it also requires a minimum
        number of real (non-discarded) episodes to have been compared.
        """
        net, mode = _net(), CubefulMoneyMode()
        n_compared = 0
        for seed in range(12):
            enc_v, tgt_v, disc = _drive_one(net, mode, seed)
            agent = TDAgent(net, oneply=False, boltzmann_temp=0.0,
                            device="cpu")
            enc_r, tgt_r = T._collect_episode_cubeful_money(
                agent, mode, random.Random(seed),
                mode.initial_match_state(), oneply=True,
                max_episode_plies=1500,
            )
            # The reference signals a discard by returning empty.
            assert disc == (enc_r.shape[0] == 0), (
                f"seed {seed}: discard decision differs "
                f"(generator={disc}, reference={enc_r.shape[0] == 0})")
            assert enc_v.shape == enc_r.shape, f"seed {seed}: shape"
            assert np.array_equal(enc_v, enc_r), f"seed {seed}: encodings"
            assert np.array_equal(tgt_v, tgt_r), f"seed {seed}: targets"
            if not disc:
                n_compared += 1
        assert n_compared >= 3, (
            f"only {n_compared} non-discarded episodes compared -- the "
            f"fixture net degenerated and the test is near-vacuous")

    def test_worker_returns_three_tuple(self):
        """Arity must match trainer._collect_worker so the trainer's
        round accounting is shared."""
        net, mode = _net(), CubefulMoneyMode()
        res = V.vectorized_collect_worker(
            _worker_args(net, mode, 4, 11, 4, 1500))
        assert len(res) == 3
        enc, tgt, disc = res
        assert enc.shape[0] == tgt.shape[0]
        assert isinstance(disc, int)

    def test_lockstep_batch_size_does_not_change_results(self):
        """B=1 (no interleaving) vs B=8 (max interleaving), same seeds.

        Direct test for shared-C-buffer aliasing across a suspended
        generator's `yield`.
        """
        net, mode = _net(), CubefulMoneyMode()
        e1, t1, d1 = V.vectorized_collect_worker(
            _worker_args(net, mode, 16, 5150, 1, 1500))
        e8, t8, d8 = V.vectorized_collect_worker(
            _worker_args(net, mode, 16, 5150, 8, 1500))
        assert d1 == d8 == 0
        assert e1.shape == e8.shape
        # Games finish in a different ORDER at B=8, so compare the rows
        # as a set (lexicographic sort on the encoding).
        k1 = np.lexsort(e1.T[::-1])
        k8 = np.lexsort(e8.T[::-1])
        assert np.array_equal(e1[k1], e8[k8])
        np.testing.assert_allclose(t1[k1], t8[k8], rtol=0, atol=1e-5)


class TestPlyGuard:
    @pytest.mark.parametrize("B", [1, 8])
    def test_tiny_cap_discards_every_episode(self, B):
        """cap=1 kills every game; the worker must still RETURN (with a
        full discard count) rather than stall or hand back a short pool.
        """
        net, mode = _net(), CubefulMoneyMode()
        enc, tgt, disc = V.vectorized_collect_worker(
            _worker_args(net, mode, 12, 7, B, 1))
        assert disc == 12
        assert enc.shape[0] == 0 and tgt.shape[0] == 0

    def test_all_episodes_accounted_when_quota_exceeds_batch(self):
        """num_episodes > B with a cap that kills every game before its
        first yield. Such a game fills no slot, so an initial fill that
        only ATTEMPTED B times would leave `slots` empty, skip the
        driver loop entirely, and silently return B episodes instead of
        num_episodes.
        """
        net, mode = _net(), CubefulMoneyMode()
        enc, tgt, disc = V.vectorized_collect_worker(
            _worker_args(net, mode, 100, 7, 32, 1))
        assert disc == 100, f"only {disc}/100 episodes accounted for"
        assert enc.shape[0] == 0

    def test_discard_count_independent_of_batch_size(self):
        """Lockstep must not change WHICH episodes the guard kills.

        Swept over caps rather than pinned to one, so the test does not
        depend on the fixture net's (degenerate) episode-length
        distribution: the guard must be monotone in the cap, the
        extremes must bracket, and B must never matter.
        """
        net, mode = _net(), CubefulMoneyMode()
        n_ep = 12
        # Small caps only: with the degenerate fixture net a large cap
        # lets episodes run for thousands of plies, which would make
        # this a minute-scale test for no extra coverage. cap=0 /
        # very-large is covered by test_cap_zero_disables_guard.
        prev_disc = None
        for cap in (1, 20, 100, 400):
            e1, t1, d1 = V.vectorized_collect_worker(
                _worker_args(net, mode, n_ep, 31, 1, cap))
            e8, t8, d8 = V.vectorized_collect_worker(
                _worker_args(net, mode, n_ep, 31, 8, cap))
            assert d1 == d8, (
                f"cap={cap}: discard count depends on B "
                f"({d1} at B=1 vs {d8} at B=8)")
            assert e1.shape == e8.shape, f"cap={cap}: kept rows differ"
            assert e1.shape[0] == t1.shape[0]
            if prev_disc is not None:
                assert d1 <= prev_disc, (
                    f"cap={cap}: discards rose with a larger cap "
                    f"({prev_disc} -> {d1})")
            prev_disc = d1
        assert prev_disc < n_ep, (
            "every episode discarded even at cap=400 -- test is vacuous")

    def test_cap_zero_disables_guard(self):
        net, mode = _net(), CubefulMoneyMode()
        e0, t0, d0 = V.vectorized_collect_worker(
            _worker_args(net, mode, 6, 99, 8, 0))
        eb, tb, db = V.vectorized_collect_worker(
            _worker_args(net, mode, 6, 99, 8, 100000))
        assert d0 == db == 0
        assert e0.shape == eb.shape
        assert np.array_equal(e0, eb)

    def test_enc_and_targets_stay_in_lockstep_under_discards(self):
        net, mode = _net(), CubefulMoneyMode()
        for cap in (5, 15, 40, 1500):
            enc, tgt, disc = V.vectorized_collect_worker(
                _worker_args(net, mode, 16, 4, 8, cap))
            assert enc.shape[0] == tgt.shape[0], f"cap={cap}"


class TestTrainerWiring:
    def test_trainer_vec_args_match_worker_arity(self, monkeypatch):
        """Trainer._build_vec_args (producer) vs
        vectorized_collect_worker (consumer): the tuple arity and field
        ORDER must agree.

        This is not paranoia. The two are joined only by positional
        unpacking across a process boundary, and both carry a comment
        saying so. A mismatch would surface as a ValueError (or, worse,
        a silent field shift) only once a real GPU run reached its
        first round.

        Driven without a GPU by intercepting the real pool.map in the
        parent process and swapping the trainer's hardcoded "cuda"
        driver device for "cpu". Everything else -- the args the trainer
        actually builds, the real worker, the trainer's 3-tuple
        accounting -- is exercised for real.
        """
        import multiprocessing.pool as mpp
        from trainer import Trainer

        captured = []

        def fake_map(self, fn, iterable, chunksize=None):
            out = []
            for args in iterable:
                captured.append(args)
                a = list(args)
                # Driver device is the second-to-last field.
                assert a[-2] == "cuda", "device field moved"
                a[-2] = "cpu"
                out.append(fn(tuple(a)))
            return out

        monkeypatch.setattr(mpp.Pool, "map", fake_map)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        # The VRAM preflight (trainer.py) calls mem_get_info; the real one
        # raises on GPU-less boxes and reports whatever is free on busy
        # ones (falling back to the CPU collector under ~1.35GB). Both
        # would break this no-GPU-required test, so report plenty free.
        monkeypatch.setattr(torch.cuda, "mem_get_info",
                            lambda *a, **kw: (64 << 30, 64 << 30))

        net, mode = _net(), CubefulMoneyMode()
        tr = Trainer(TDAgent(net, oneply=False, device="cpu"), lr=1e-6)
        tr.train(
            mode, num_episodes=4, episodes_per_round=4, batch_size=8,
            workers=2, oneply=True, seed=3, log_every=0,
            vectorized_collect=True, vectorized_batch=2,
            max_episode_plies=60,
        )

        assert len(captured) == 2, "one args tuple per worker"
        for args in captured:
            # Same arity the worker unpacks, and the ply bound is last.
            # 16 fields since 2026-08-05: `activation` joined the spec
            # block (index 5) -- the silent-relu-rebuild fix.
            assert len(args) == 16, f"arity drifted: {len(args)}"
            assert args[-1] == 60, "max_episode_plies is not the last field"
            assert args[-3] == 2, "vectorized_batch field moved"
            assert args[5] == "relu", "activation field missing/moved"
            assert args[9] is True, "oneply field moved"

    def test_vec_and_cpu_worker_arities_are_distinct_but_pinned(self):
        """Both arg tuples end with max_episode_plies; the CPU tuple
        carries c_engine where the vectorized one carries (B, device).
        Pinned so a future field addition has to touch both call sites.
        """
        import inspect
        import trainer as T

        cpu_src = inspect.getsource(T._collect_worker)
        assert "max_episode_plies) = args" in cpu_src
        vec_src = inspect.getsource(V.vectorized_collect_worker)
        assert "max_episode_plies) = args" in vec_src
        assert "c_engine" not in vec_src


class TestGuards:
    def test_oneply_acting_is_refused(self):
        """--oneply-acting would silently get 0-ply move selection."""
        net, mode = _net(), CubefulMoneyMode()
        args = list(_worker_args(net, mode, 2, 1, 2, 1500))
        args[10] = True  # oneply_acting (index 10 since activation joined)
        with pytest.raises(AssertionError, match="oneply-acting"):
            V.vectorized_collect_worker(tuple(args))

    def test_trainer_refuses_oneply_acting(self):
        from trainer import Trainer
        net = _net()
        agent = TDAgent(net, oneply=True, device="cpu")
        tr = Trainer(agent, lr=1e-6)
        with pytest.raises(ValueError, match="oneply-acting"):
            tr.train(
                CubefulMoneyMode(), num_episodes=2, episodes_per_round=2,
                workers=2, oneply=True, oneply_acting=True,
                vectorized_collect=True,
            )
