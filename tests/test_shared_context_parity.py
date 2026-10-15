"""Shared-context (one CUDA context) collector == per-worker-context
collector, bitwise.

The shared-context topology moves each worker's fused lockstep forward
from a local cuda net into the EvalServer process. The arithmetic does
not change (same fp32 GEMM, same row count per forward: the server runs
each request on its own, `coalesce=False`), so the collected encodings
and targets must be IDENTICAL, not merely close.

Needs CUDA (skips cleanly without it). Drives the real worker functions
in-process with a live EvalServer, exactly as the pool initializer would
wire them.
"""
import multiprocessing as mp
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

import vectorized_collect as V
import train_prob5 as P5
from gpu_eval_server import EvalServer, RemoteEvaluator
from model import TDNetwork, ProbNetwork
from modes import CubefulMoneyMode, CubelessMoneyMode

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUBEFUL_MODEL = os.path.join(_REPO, "best_models", "cubeful_money_512_512_256_256.pt")
PROB5_MODEL = os.path.join(_REPO, "best_models", "cubeless_prob5_512_512_256_256.pt")


class LiveServer:
    """EvalServer + one RemoteEvaluator installed as the module-global the
    worker reads, exactly like the pool initializer does."""

    def __init__(self, module, spec, sd, n_feat, out_dim, max_rows):
        self.module, self.spec, self.sd = module, spec, sd
        self.n_feat, self.out_dim, self.max_rows = n_feat, out_dim, max_rows

    def __enter__(self):
        self.srv = EvalServer(
            mp.get_context("spawn"), num_slots=1, n_feat=self.n_feat,
            out_dim=self.out_dim, max_rows=self.max_rows,
            build_spec=self.spec, init_state_dict=self.sd,
            device="cuda", coalesce=False,
        )
        self.srv.start()
        h = self.srv.client_handles()
        self.module._WORKER_REMOTE = RemoteEvaluator(h.free_slots.get(), h)
        return self

    def __exit__(self, *a):
        self.module._WORKER_REMOTE = None
        self.srv.stop()
        return False


# ── cubeful (train_batch.py / trainer.py) ───────────────────────────

def _cubeful_net():
    return TDNetwork.load(CUBEFUL_MODEL)


def _cubeful_args(net, device, n=12, B=4, seed=11):
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return (sd, list(net.hidden_sizes), net.input_size, net.output_mode,
            net.encoder_name, net.activation, CubefulMoneyMode(), n, seed,
            True, False, 0.0, False, B, device, 1500)


def _td_spec(net):
    return {"net_type": "td", "hidden_sizes": list(net.hidden_sizes),
            "input_size": net.input_size, "output_mode": net.output_mode,
            "encoder_name": net.encoder_name, "activation": net.activation}


class TestCubefulParity:
    def test_bitwise_identical_collection(self):
        net = _cubeful_net()
        enc_a, tgt_a, disc_a = V.vectorized_collect_worker(
            _cubeful_args(net, "cuda"))
        with LiveServer(V, _td_spec(net), net.state_dict(), net.input_size,
                        1, V.remote_max_rows(4)):
            enc_b, tgt_b, disc_b = V.vectorized_collect_worker(
                _cubeful_args(net, "remote"))
        assert enc_a.shape == enc_b.shape and enc_a.shape[0] > 200
        assert np.array_equal(enc_a, enc_b)
        assert np.array_equal(tgt_a, tgt_b)
        assert disc_a == disc_b

    def test_chunked_overflow_is_close_not_identical_guard(self):
        """A step larger than the slot is split into several forwards.
        That changes the GEMM shape, so bitwise identity is NOT promised
        there, but the values must agree to fp32 GEMM tolerance."""
        net = _cubeful_net()
        enc_a, tgt_a, _ = V.vectorized_collect_worker(
            _cubeful_args(net, "cuda", n=6, B=6))
        # 256 rows is far below one step (B=6 games x ~450 rows each).
        with LiveServer(V, _td_spec(net), net.state_dict(), net.input_size,
                        1, 256):
            enc_b, tgt_b, _ = V.vectorized_collect_worker(
                _cubeful_args(net, "remote", n=6, B=6))
        assert np.array_equal(enc_a, enc_b)
        assert np.allclose(tgt_a, tgt_b, atol=2e-6, rtol=0), \
            float(np.abs(tgt_a - tgt_b).max())

    def test_remote_without_slot_fails_loudly(self):
        net = _cubeful_net()
        V._WORKER_REMOTE = None
        with pytest.raises(RuntimeError, match="no EvalServer slot"):
            V.vectorized_collect_worker(_cubeful_args(net, "remote", n=2, B=2))

    def test_server_honors_activation(self):
        """The server rebuilds the net from the spec; `activation` is not
        a parameter tensor, so a dropped field would silently serve relu
        for a tanh net. Same weights, tanh vs relu spec: must differ."""
        torch.manual_seed(3)
        base = _cubeful_net()   # for the cubeful encoder name / input size
        net = TDNetwork(hidden_sizes=[32, 16], input_size=base.input_size,
                        output_mode="equity", encoder_name=base.encoder_name,
                        activation="tanh")
        outs = {}
        for act in ("tanh", "relu"):
            spec = {**_td_spec(net), "activation": act}
            with LiveServer(V, spec, net.state_dict(), net.input_size, 1, 8192):
                _, tgt, _ = V.vectorized_collect_worker(
                    _cubeful_args(net, "remote", n=3, B=3, seed=5))
            outs[act] = tgt
        # tanh spec == local tanh driver, bitwise
        _, tgt_local, _ = V.vectorized_collect_worker(
            _cubeful_args(net, "cuda", n=3, B=3, seed=5))
        assert np.array_equal(outs["tanh"], tgt_local)
        # relu spec: different values, hence (almost surely) a different
        # trajectory and a different number of samples.
        assert outs["tanh"].shape != outs["relu"].shape or \
            not np.allclose(outs["tanh"], outs["relu"])


# ── prob5 (train_prob5.py) ──────────────────────────────────────────

def _prob5_net():
    return ProbNetwork.load(PROB5_MODEL)


def _prob5_args(net, device, n=12, B=4, seed=11):
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return (sd, net.hidden_sizes, net.input_size, net.activation,
            net.encoder_name, net.raw_logits, n, seed, B, device,
            P5.DEFAULT_MAX_EPISODE_PLIES)


def _prob5_spec(net):
    return {"net_type": "prob5", "hidden_sizes": net.hidden_sizes,
            "input_size": net.input_size, "activation": net.activation,
            "encoder_name": net.encoder_name, "raw_logits": net.raw_logits,
            "postprocess": "prob5"}


class TestProb5Parity:
    def test_bitwise_identical_collection(self):
        net = _prob5_net()
        enc_a, tgt_a, _ = P5._vectorized_collect_worker(_prob5_args(net, "cuda"))
        with LiveServer(P5, _prob5_spec(net), net.state_dict(),
                        net.input_size, P5.NUM_OUTPUTS, V.remote_max_rows(4)):
            enc_b, tgt_b, _ = P5._vectorized_collect_worker(
                _prob5_args(net, "remote"))
        assert enc_a.shape == enc_b.shape and enc_a.shape[0] > 200
        assert tgt_a.shape[1] == P5.NUM_OUTPUTS
        assert np.array_equal(enc_a, enc_b)
        assert np.array_equal(tgt_a, tgt_b)

    def test_postprocess_is_applied_server_side(self):
        """Without `postprocess: prob5` the server would return raw
        sigmoids without the nested-event clamp and the targets would
        differ. Guards the spec field."""
        net = _prob5_net()
        _, tgt_a, _ = P5._vectorized_collect_worker(_prob5_args(net, "cuda", n=4, B=4))
        spec = dict(_prob5_spec(net))
        del spec["postprocess"]
        with LiveServer(P5, spec, net.state_dict(), net.input_size,
                        P5.NUM_OUTPUTS, V.remote_max_rows(4)):
            _, tgt_raw, _ = P5._vectorized_collect_worker(
                _prob5_args(net, "remote", n=4, B=4))
        # Same encodings/trajectory is not guaranteed once values differ,
        # so only assert that dropping the field is observable.
        assert not (tgt_raw.shape == tgt_a.shape and np.array_equal(tgt_raw, tgt_a))

    def test_chunked_overflow_is_close_not_identical_guard(self):
        """Same contract as the cubeful test: a split step changes the
        GEMM shape, so bitwise identity is not promised, but the values
        must agree to fp32 GEMM tolerance, and the split path must have
        run (otherwise the test probes nothing)."""
        net = _prob5_net()
        enc_a, tgt_a, _ = P5._vectorized_collect_worker(
            _prob5_args(net, "cuda", n=6, B=6))
        # 256 rows is far below one 1-ply step (B=6 games x ~450 rows).
        with LiveServer(P5, _prob5_spec(net), net.state_dict(), net.input_size,
                        P5.NUM_OUTPUTS, 256):
            remote = P5._WORKER_REMOTE
            # The split path goes through forward(); the in-slot path
            # through forward_from_slot(). Count the former.
            n_split = [0]
            orig_forward = remote.forward

            def counting_forward(feats):
                n_split[0] += 1
                return orig_forward(feats)

            remote.forward = counting_forward
            enc_b, tgt_b, _ = P5._vectorized_collect_worker(
                _prob5_args(net, "remote", n=6, B=6))
        assert n_split[0] > 0, "no step overflowed the slot: null probe"
        assert np.array_equal(enc_a, enc_b)
        assert np.allclose(tgt_a, tgt_b, atol=2e-6, rtol=0), \
            float(np.abs(tgt_a - tgt_b).max())

    def test_remote_without_slot_fails_loudly(self):
        net = _prob5_net()
        P5._WORKER_REMOTE = None
        with pytest.raises(RuntimeError, match="no eval-server slot"):
            P5._vectorized_collect_worker(_prob5_args(net, "remote", n=2, B=2))

    def test_server_honors_activation(self):
        """Same guard as the cubeful one: `activation` is not a parameter
        tensor, so a spec that dropped it would silently serve relu for a
        tanh net. Same weights, tanh vs relu spec: must differ."""
        torch.manual_seed(3)
        base = _prob5_net()   # for the prob5 encoder name / input size
        net = ProbNetwork(hidden_sizes=[32, 16], input_size=base.input_size,
                          activation="tanh", encoder_name=base.encoder_name,
                          raw_logits=base.raw_logits)
        outs = {}
        for act in ("tanh", "relu"):
            spec = {**_prob5_spec(net), "activation": act}
            with LiveServer(P5, spec, net.state_dict(), net.input_size,
                            P5.NUM_OUTPUTS, 8192):
                _, tgt, _ = P5._vectorized_collect_worker(
                    _prob5_args(net, "remote", n=3, B=3, seed=5))
            outs[act] = tgt
        # tanh spec == local tanh driver, bitwise
        _, tgt_local, _ = P5._vectorized_collect_worker(
            _prob5_args(net, "cuda", n=3, B=3, seed=5))
        assert np.array_equal(outs["tanh"], tgt_local)
        # relu spec: different values, hence (almost surely) a different
        # trajectory and a different number of samples.
        assert outs["tanh"].shape != outs["relu"].shape or \
            not np.allclose(outs["tanh"], outs["relu"])
