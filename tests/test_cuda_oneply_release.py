"""Release-behavior tests for --oneply --device cuda.

Covers, without needing a GPU:
  * the train_batch routing table (AUTO vectorized under cuda+oneply,
    the --no-vectorized-collect escape, deference to explicit CPU
    collector accelerators, the mutual-exclusion error);
  * the VRAM preflight arithmetic (pure function);
  * the prob5 cuda+oneply rejection;
  * activation actually reaching the rebuilt worker nets on BOTH
    collector paths -- the silent-relu-rebuild regression test. A
    tanh checkpoint driven through each worker entry point must produce
    different collected data than the same weights rebuilt as relu
    (pre-fix, activation was dropped and the two were identical).
"""
import argparse
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

import trainer as T
import train_batch as TB
import train_prob5 as P5
import vectorized_collect as V
from model import TDNetwork
from modes import CubelessMoneyMode


# ── routing table ────────────────────────────────────────────────────

def _args(**kw):
    d = dict(oneply=True, device="cuda", vectorized_collect=False,
             no_vectorized_collect=False, c_engine=False,
             bf16_collect=False, oneply_acting=False)
    d.update(kw)
    return argparse.Namespace(**d)


class TestCudaOneplyRouting:
    def test_auto_enables_vectorized(self):
        a = _args()
        notes = TB.resolve_cuda_oneply_collector(a)
        assert a.vectorized_collect is True
        assert any("AUTO" in n for n in notes)

    def test_no_flag_keeps_cpu_collector(self):
        a = _args(no_vectorized_collect=True)
        notes = TB.resolve_cuda_oneply_collector(a)
        assert a.vectorized_collect is False
        assert any("keeping the CPU collector" in n for n in notes)

    def test_explicit_flag_untouched_no_note(self):
        a = _args(vectorized_collect=True)
        assert TB.resolve_cuda_oneply_collector(a) == []
        assert a.vectorized_collect is True

    def test_cpu_accelerators_win(self):
        for kw in ({"c_engine": True}, {"bf16_collect": True}):
            a = _args(**kw)
            notes = TB.resolve_cuda_oneply_collector(a)
            assert a.vectorized_collect is False, kw
            assert any("wins" in n for n in notes)

    def test_not_cuda_or_not_oneply_is_noop(self):
        for kw in ({"device": "cpu"}, {"oneply": False}):
            a = _args(**kw)
            assert TB.resolve_cuda_oneply_collector(a) == []
            assert a.vectorized_collect is False

    def test_both_flags_error(self):
        with pytest.raises(SystemExit):
            TB.resolve_cuda_oneply_collector(
                _args(vectorized_collect=True, no_vectorized_collect=True))

    def test_oneply_acting_keeps_cpu_collector(self):
        """Auto-routing must not manufacture the vectorized worker's
        --oneply-acting hard error out of a previously valid command
        (--oneply --oneply-acting --device cuda, no vectorized flags)."""
        a = _args(oneply_acting=True)
        notes = TB.resolve_cuda_oneply_collector(a)
        assert a.vectorized_collect is False
        assert any("oneply-acting" in n for n in notes)

    def test_oneply_acting_explicit_vectorized_untouched(self):
        # Explicitly asking for both is left for Trainer.train's loud
        # ValueError -- the resolver only guards the AUTO path.
        a = _args(oneply_acting=True, vectorized_collect=True)
        assert TB.resolve_cuda_oneply_collector(a) == []
        assert a.vectorized_collect is True


# ── VRAM preflight arithmetic ────────────────────────────────────────

GB = 2**30
MB = 2**20


class TestVramCap:
    def test_plenty_returns_requested(self):
        assert T.vectorized_vram_cap(24, 32 * GB) == 24

    def test_8gb_card_caps_between_16_and_18(self):
        # ~7.9 GB free (idle 1070 Ti): the measured envelope says W=24
        # barely fit and W=32 OOM'd; the conservative estimate must land
        # in the teens, not at either extreme.
        cap = T.vectorized_vram_cap(64, int(7.9 * GB))
        assert 14 <= cap <= 19, cap

    def test_tiny_vram_returns_zero(self):
        assert T.vectorized_vram_cap(24, 600 * MB) == 0

    def test_free_below_reserve_returns_zero(self):
        assert T.vectorized_vram_cap(24, T.VEC_RESERVE_BYTES - 1) == 0

    def test_zero_requested(self):
        assert T.vectorized_vram_cap(0, 32 * GB) == 0

    def test_monotone_in_free_bytes(self):
        caps = [T.vectorized_vram_cap(64, f * GB) for f in range(0, 33, 2)]
        assert caps == sorted(caps)


# ── prob5 rejection ──────────────────────────────────────────────────

class TestProb5CudaRejection:
    def test_oneply_cuda_rejected(self):
        with pytest.raises(SystemExit) as ei:
            P5.reject_cuda_oneply(
                argparse.Namespace(oneply=True, device="cuda"))
        assert "prob5 1-ply" in str(ei.value)
        assert "prob5_collect_levers_ab" in str(ei.value)

    def test_oneply_cpu_ok(self):
        P5.reject_cuda_oneply(argparse.Namespace(oneply=True, device="cpu"))

    def test_zeroply_cuda_ok(self):
        P5.reject_cuda_oneply(argparse.Namespace(oneply=False, device="cuda"))


# ── activation threading (silent-relu-rebuild regression) ────────────

def _tanh_sd():
    torch.manual_seed(7)
    net = TDNetwork(hidden_sizes=[16, 8], input_size=196,
                    output_mode="equity", encoder_name="perspective196",
                    activation="tanh")
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _run_cpu_worker(sd, activation, seed=3, n=2):
    args = (sd, [16, 8], 196, "equity", "perspective196", activation,
            CubelessMoneyMode(), n, seed, True, False, 0.0, False,
            False, 1500)
    enc, tgt, disc = T._collect_worker(args)
    return enc, tgt


def _run_vec_worker(sd, activation, seed=3, n=2, B=2):
    args = (sd, [16, 8], 196, "equity", "perspective196", activation,
            CubelessMoneyMode(), n, seed, True, False, 0.0, False,
            B, "cpu", 1500)
    enc, tgt, disc = V.vectorized_collect_worker(args)
    return enc, tgt


class TestActivationReachesWorkers:
    """Same tanh weights, same seed, rebuilt as tanh vs relu: the two
    runs MUST differ. Pre-fix, activation was dropped from the args
    tuple and both rebuilt as relu -- this test fails on that code."""

    def test_cpu_worker_honors_activation(self):
        sd = _tanh_sd()
        enc_t, tgt_t = _run_cpu_worker(sd, "tanh")
        enc_r, tgt_r = _run_cpu_worker(sd, "relu")
        assert not (np.array_equal(enc_t, enc_r)
                    and np.array_equal(tgt_t, tgt_r)), (
            "tanh vs relu rebuild produced identical episodes -- "
            "activation is not reaching the CPU worker")

    def test_vec_worker_honors_activation(self):
        sd = _tanh_sd()
        enc_t, tgt_t = _run_vec_worker(sd, "tanh")
        enc_r, tgt_r = _run_vec_worker(sd, "relu")
        assert not (np.array_equal(enc_t, enc_r)
                    and np.array_equal(tgt_t, tgt_r)), (
            "tanh vs relu rebuild produced identical episodes -- "
            "activation is not reaching the vectorized worker")

    def test_vec_worker_same_seed_deterministic(self):
        sd = _tanh_sd()
        a = _run_vec_worker(sd, "tanh")
        b = _run_vec_worker(sd, "tanh")
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
