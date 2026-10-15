"""Parity tests for the prob5 vectorized collector (CPU, no GPU).

In the spirit of tests/test_vectorized_cubeless.py: the lockstep
coroutine rewrite is a pure restructuring of the stock collection path,
so driven with identical fp32 CPU forwards it must reproduce
_collect_one_episode's (encodings, targets) bitwise.

  * B=1 runs games strictly sequentially -> ordered bitwise equality.
  * B>1 fuses blocks from several games into one forward, and fp32
    forwards are not batch-composition invariant (see
    experiments/gpu_1ply_bench/README.md). The B>1 contract is
    therefore: encodings bit-equal as a multiset (same trajectories),
    targets within 1 ulp when paired by position.
  * The per-dice fallback (_oneply_target_vec_perdice_gen) and the
    0-ply generator branch are exercised explicitly.
  * The 10-field _build_vec_args tuple is pinned against
    _vectorized_collect_worker (positional unpack across a process
    boundary), mirroring tests/test_vectorized_collect.py's trainer pin.
  * Activation threading: tanh weights rebuilt in the worker must not
    silently become relu.
"""
from _pool_fakes import patch_pool_map  # noqa: E402
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

import train_prob5 as P5
from model import ProbNetwork



@pytest.fixture(autouse=True, scope="module")
def _one_torch_thread():
    """This module's references run torch on one thread; restored after."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _make_net(hidden, activation="relu", seed=0):
    torch.manual_seed(seed)
    net = ProbNetwork(hidden_sizes=hidden, input_size=196,
                      activation=activation, encoder_name="perspective196",
                      raw_logits=False)
    net.eval()
    return net


def _ref_games(net, seed, n):
    """Reference episodes with the SAME per-game rng spawn as the worker."""
    encode_fn, gpe_fn = P5._resolve_engine_fns("perspective196")
    base = random.Random(seed)
    out = []
    for _ in range(n):
        g_rng = random.Random(base.randint(0, 2**31 - 1))
        out.append(P5._collect_one_episode(
            net, g_rng, encode_fn, gpe_fn, oneply=True, device="cpu"))
    return out


def _vec_run(net, seed, n, B):
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    args = (sd, net.hidden_sizes, net.input_size, net.activation,
            net.encoder_name, net.raw_logits, n, seed, B, "cpu", P5.DEFAULT_MAX_EPISODE_PLIES)
    return P5._vectorized_collect_worker(args)[:2]


def _sort_by_enc(enc, tgt):
    """Order rows by their encoding so the two runs pair position-wise
    (completion order differs under lockstep)."""
    order = np.lexsort(enc.T[::-1])
    return enc[order], tgt[order]


SEED = 12345
N_GAMES = 3


class TestOneplyLockstepParity:
    _refs = None

    @classmethod
    def refs(cls):
        if cls._refs is None:
            net = _make_net([64, 32])
            games = _ref_games(net, SEED, N_GAMES)
            cls._refs = (
                net,
                np.concatenate([e for e, t in games], axis=0),
                np.concatenate([t for e, t in games], axis=0),
            )
        return cls._refs

    def test_b1_is_bitwise_ordered(self):
        net, r_enc, r_tgt = self.refs()
        v_enc, v_tgt = _vec_run(net, SEED, N_GAMES, B=1)
        assert v_enc.shape == r_enc.shape
        assert np.array_equal(v_enc, r_enc), "encodings diverged"
        assert np.array_equal(v_tgt, r_tgt), "targets diverged"

    @pytest.mark.parametrize("B", [2, 4])
    def test_lockstep_same_games_within_one_ulp(self, B):
        net, r_enc, r_tgt = self.refs()
        v_enc, v_tgt = _vec_run(net, SEED, N_GAMES, B=B)
        assert v_enc.shape == r_enc.shape
        re, rt = _sort_by_enc(r_enc, r_tgt)
        ve, vt = _sort_by_enc(v_enc, v_tgt)
        # Trajectories must be identical: bit-equal encodings.
        assert np.array_equal(re, ve), f"B={B}: encodings diverged"
        # Targets: fused-batch composition may move rows by 1 fp32 ulp
        # (2^-24 at magnitude ~1). Anything larger is a driver bug.
        d = np.abs(rt.astype(np.float64) - vt).max()
        assert d <= 1.2e-7, f"B={B}: max target delta {d} exceeds 1 ulp"

    @pytest.mark.skipif(not P5._BG_FAST_AVAILABLE,
                        reason="per-dice fallback forcing needs bg_fast")
    def test_perdice_fallback_is_bitwise_ordered(self, monkeypatch):
        """Force use_expand21=False so both sides take
        _oneply_target_vec_perdice(_gen)."""
        monkeypatch.setattr(P5._bg_fast, "_HAS_EXPAND21", False)
        net = _make_net([32, 16])
        games = _ref_games(net, seed=777, n=3)
        r_enc = np.concatenate([e for e, t in games], axis=0)
        r_tgt = np.concatenate([t for e, t in games], axis=0)
        v_enc, v_tgt = _vec_run(net, seed=777, n=3, B=1)
        assert np.array_equal(v_enc, r_enc)
        assert np.array_equal(v_tgt, r_tgt)


class TestZeroPlyGeneratorParity:
    def _drive(self, net, rng):
        encode_fn, gpe_fn = P5._resolve_engine_fns("perspective196")
        g = P5._collect_one_episode_gen(
            rng, encode_fn, gpe_fn, oneply=False, input_size=196)
        try:
            feat = next(g)
            while True:
                with torch.no_grad():
                    out = net(torch.from_numpy(
                        np.ascontiguousarray(feat, dtype=np.float32)))
                    out = P5.prob5_postprocess(out)
                feat = g.send(out.numpy())
        except StopIteration as e:
            return e.value

    def test_bitwise_vs_reference(self):
        net = _make_net([32, 16])
        encode_fn, gpe_fn = P5._resolve_engine_fns("perspective196")
        for seed in range(4):
            v_enc, v_tgt = self._drive(net, random.Random(seed))
            r_enc, r_tgt = P5._collect_one_episode(
                net, random.Random(seed), encode_fn, gpe_fn,
                oneply=False, device="cpu")
            assert np.array_equal(v_enc, r_enc), f"seed {seed}: encodings"
            assert np.array_equal(v_tgt, r_tgt), f"seed {seed}: targets"


class TestActivationReachesVecWorker:
    """Same tanh weights, same seed, rebuilt as tanh vs relu: the two
    runs MUST differ, else the worker silently rebuilds relu."""

    def _run(self, activation):
        src = _make_net([16, 8], activation="tanh", seed=7)
        sd = {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}
        args = (sd, [16, 8], 196, activation, "perspective196", False,
                2, 3, 2, "cpu", P5.DEFAULT_MAX_EPISODE_PLIES)
        return P5._vectorized_collect_worker(args)[:2]

    def test_vec_worker_honors_activation(self):
        enc_t, tgt_t = self._run("tanh")
        enc_r, tgt_r = self._run("relu")
        assert not (np.array_equal(enc_t, enc_r)
                    and np.array_equal(tgt_t, tgt_r)), (
            "tanh vs relu rebuild produced identical episodes -- "
            "activation is not reaching the vectorized worker")

    def test_same_seed_deterministic(self):
        a = self._run("tanh")
        b = self._run("tanh")
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


class TestSharedContextTopologyParity:
    """The shared-context topology (device="remote": the fused batch goes
    through an eval-server slot instead of a local net) is a transport
    change, not an arithmetic one. Driven with a CPU eval server it must
    reproduce the local-net ("cpu") driver BITWISE at every B.
    No GPU needed; the CUDA twin is tests/test_shared_context_parity.py.
    """

    @staticmethod
    def _remote_run(net, seed, n, B):
        import multiprocessing as mp
        from gpu_eval_server import EvalServer, claim_slot
        from vectorized_collect import remote_max_rows
        spec = {"net_type": "prob5", "hidden_sizes": net.hidden_sizes,
                "input_size": net.input_size, "activation": net.activation,
                "encoder_name": net.encoder_name, "raw_logits": net.raw_logits,
                "postprocess": "prob5"}
        srv = EvalServer(mp.get_context("spawn"), num_slots=1,
                         n_feat=net.input_size, out_dim=P5.NUM_OUTPUTS,
                         max_rows=remote_max_rows(B), build_spec=spec,
                         init_state_dict=net.state_dict(), device="cpu",
                         coalesce=False)
        srv.start()
        try:
            P5._WORKER_REMOTE = claim_slot(srv.client_handles())
            sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            args = (sd, net.hidden_sizes, net.input_size, net.activation,
                    net.encoder_name, net.raw_logits, n, seed, B, "remote", P5.DEFAULT_MAX_EPISODE_PLIES)
            return P5._vectorized_collect_worker(args)[:2]
        finally:
            P5._WORKER_REMOTE = None
            srv.stop()

    @pytest.mark.parametrize("B", [1, pytest.param(2, marks=pytest.mark.slow), 4])
    def test_remote_equals_local_bitwise(self, B):
        net = _make_net([64, 32])
        l_enc, l_tgt = _vec_run(net, SEED, N_GAMES, B=B)
        r_enc, r_tgt = self._remote_run(net, SEED, N_GAMES, B=B)
        assert r_enc.shape == l_enc.shape
        assert np.array_equal(r_enc, l_enc), f"B={B}: encodings diverged"
        assert np.array_equal(r_tgt, l_tgt), f"B={B}: targets diverged"

    def test_remote_b1_is_bitwise_vs_reference(self):
        net = _make_net([64, 32])
        games = _ref_games(net, SEED, N_GAMES)
        r_enc = np.concatenate([e for e, t in games], axis=0)
        r_tgt = np.concatenate([t for e, t in games], axis=0)
        v_enc, v_tgt = self._remote_run(net, SEED, N_GAMES, B=1)
        assert np.array_equal(v_enc, r_enc)
        assert np.array_equal(v_tgt, r_tgt)

    def test_remote_without_slot_fails_loudly(self):
        net = _make_net([16, 8])
        P5._WORKER_REMOTE = None
        sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        args = (sd, net.hidden_sizes, net.input_size, net.activation,
                net.encoder_name, net.raw_logits, 2, 3, 2, "remote", P5.DEFAULT_MAX_EPISODE_PLIES)
        with pytest.raises(RuntimeError, match="no eval-server slot"):
            P5._vectorized_collect_worker(args)


class TestVecArgsWiring:
    def test_build_vec_args_matches_worker_arity(self, monkeypatch):
        """train()._build_vec_args (producer) vs _vectorized_collect_worker
        (consumer): tuple arity and field ORDER must agree; they are
        joined only by positional unpacking across a process boundary.
        Intercepts pool.map in the parent and runs the real worker on the
        args train() builds."""
        import multiprocessing.pool as mpp

        captured = []

        def fake_map(self, fn, iterable, chunksize=None):
            out = []
            for args in iterable:
                captured.append(args)
                out.append(fn(args))
            return out

        patch_pool_map(monkeypatch, fake_map)

        net = _make_net([16, 8])
        P5.train(
            net, num_episodes=4, lr=1e-6, end_lr=None,
            episodes_per_round=4, batch_size=8, workers=2, device="cpu",
            oneply=True, pw_weight=1.0, use_mse=False, seed=3,
            log_every=0, save_path=None, save_every=0, warmup_cycles=0,
            vectorized_collect=2,
        )

        assert len(captured) == 2, "one args tuple per worker"
        for args in captured:
            assert len(args) == 11, f"arity drifted: {len(args)}"
            assert args[3] == "relu", "activation field missing/moved"
            assert args[6] == 2, "episodes-per-worker field moved"
            assert args[-3] == 2, "batch (B) field moved"
            assert args[-2] == "cpu", "collect device field moved"
            assert args[-1] == P5.DEFAULT_MAX_EPISODE_PLIES, "max_plies is not the last field"

    def test_worker_unpack_sites_are_pinned(self):
        """Both unpack sites carry the stay-in-sync note's field order:
        a future field addition has to touch producer and consumer."""
        import inspect

        vec_src = inspect.getsource(P5._vectorized_collect_worker)
        assert "batch_games, device, max_plies) = args" in vec_src
        cpu_src = inspect.getsource(P5._collect_worker)
        assert "oneply, oneply_device, max_plies) = args" in cpu_src
