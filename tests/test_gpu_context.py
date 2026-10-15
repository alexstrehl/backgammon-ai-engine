"""gpu_context: the startup decisions shared by both 1-ply GPU collectors.

Pure-function coverage runs anywhere. The spawn-inheritance check needs
only a CPU. Nothing here needs CUDA.
"""
import multiprocessing as mp
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gpu_context as GC
from model import TDNetwork

MB = 1 << 20
GB = 1 << 30


class TestDefaults:
    def test_shared_context_is_the_default_topology(self):
        # Pinned deliberately: flipping it changes the VRAM footprint of
        # every --oneply --device cuda run. See
        # experiments/gpu_1ply_bench/README.md.
        assert GC.VEC_SHARED_CONTEXT_DEFAULT is True


class TestPhysicalCores:
    def test_sane_range(self):
        n = GC.physical_cores()
        assert 1 <= n <= (os.cpu_count() or 1)


class TestAllocatorEnv:
    def test_sets_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        assert GC.set_allocator_env() is True
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == GC.VEC_ALLOC_CONF

    def test_respects_operator_value(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
        assert GC.set_allocator_env() is False
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:64"

    def test_spawn_children_inherit(self, monkeypatch):
        """The variable must reach the pool's spawn children: it has to
        be set in the parent BEFORE Pool()."""
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        GC.set_allocator_env()
        with mp.get_context("spawn").Pool(2) as pool:
            seen = pool.map(_read_alloc_env, range(2))
        assert seen == [GC.VEC_ALLOC_CONF] * 2


def _read_alloc_env(_):
    return os.environ.get("PYTORCH_CUDA_ALLOC_CONF")


class TestDefaultWorkers:
    def test_explicit_wins(self):
        assert GC.default_vectorized_workers(96, explicit=64, phys_cores=16) \
            == (64, "")

    def test_caps_at_physical_cores(self):
        n, why = GC.default_vectorized_workers(32, None, phys_cores=16)
        assert n == 16 and "16 physical cores" in why

    def test_caps_at_hard_cap(self):
        n, why = GC.default_vectorized_workers(96, None, phys_cores=96)
        assert n == GC.VEC_MAX_DEFAULT_WORKERS and "hard cap" in why

    def test_hard_cap_zero_disables_it(self):
        # Shared-context mode: no per-worker VRAM term, so only the core
        # count bounds the pool.
        assert GC.default_vectorized_workers(96, None, 96, hard_cap=0) \
            == (96, "")

    def test_no_clamp_no_reason(self):
        assert GC.default_vectorized_workers(12, None, phys_cores=16) == (12, "")

    def test_zero(self):
        assert GC.default_vectorized_workers(0, None, 16) == (0, "")


class TestVramCap:
    def test_uses_measurement_when_given(self):
        free = 16 * GB
        fallback = GC.vectorized_vram_cap(64, free)
        measured = GC.vectorized_vram_cap(64, free, 450 * MB)
        assert measured > fallback
        assert measured == min(64, (free - GC.VEC_RESERVE_BYTES) // (450 * MB))

    def test_zero_measurement_means_fallback(self):
        assert GC.vectorized_vram_cap(64, 16 * GB, 0) == \
            GC.vectorized_vram_cap(64, 16 * GB)

    def test_never_exceeds_request_or_goes_negative(self):
        assert GC.vectorized_vram_cap(4, 90 * GB, 300 * MB) == 4
        assert GC.vectorized_vram_cap(4, 100 * MB, 300 * MB) == 0
        assert GC.vectorized_vram_cap(0, 90 * GB) == 0

    def test_margin_is_additive_and_multiplicative(self):
        m = GC.with_probe_margin(400 * MB)
        assert m > 400 * MB * (1 + GC.VEC_PROBE_MARGIN_FRAC)
        assert m == int(400 * MB * (1 + GC.VEC_PROBE_MARGIN_FRAC)) \
            + GC.VEC_PROBE_MARGIN_BYTES

    def test_auto_floor_is_above_explicit_floor(self):
        # Auto-routed runs need VEC_MIN_WORKERS. Explicit ones are honored
        # down to 2. An 8 GB card on the fallback constant sits between.
        assert GC.VEC_MIN_WORKERS > 2
        cap8 = GC.vectorized_vram_cap(96, 8 * GB)
        assert 2 <= cap8 < GC.VEC_MIN_WORKERS

    def test_shared_min_free_scales_with_batch(self):
        b32 = GC.shared_context_min_free_bytes(32)
        b125 = GC.shared_context_min_free_bytes(125)
        assert 800 * MB <= b32 <= 1300 * MB, b32 / MB      # measured ~1 GiB
        assert 2 * GB <= b125 <= 3 * GB, b125 / MB          # measured ~2.5 GB
        assert GC.shared_context_min_free_bytes(1) >= GC.SHARED_SERVER_BASE_BYTES

    def test_probe_rows_scale_with_batch(self):
        assert GC.probe_rows(32) == 32 * GC.PROBE_ROWS_PER_GAME
        assert GC.probe_rows(1) >= 1024
        assert GC.probe_rows(10**6) == GC.PROBE_ROWS_MAX


class TestBuildNetFromSpec:
    def test_activation_is_carried(self):
        """activation is not a parameter tensor, so load_state_dict cannot
        catch a mismatch: identical weights under tanh vs relu must give
        different outputs, and the spec must be what selects them."""
        torch.manual_seed(0)
        src = TDNetwork(hidden_sizes=[16, 8], input_size=196,
                        output_mode="equity", activation="tanh")
        sd = src.state_dict()
        spec = {"net_type": "td", "hidden_sizes": [16, 8], "input_size": 196,
                "output_mode": "equity", "encoder_name": "perspective196",
                "activation": "tanh"}
        a = GC.build_net_from_spec(spec)
        a.load_state_dict(sd)
        b = GC.build_net_from_spec({**spec, "activation": "relu"})
        b.load_state_dict(sd)
        x = torch.rand(5, 196)
        with torch.no_grad():
            assert torch.equal(a(x), src(x))
            assert not torch.allclose(a(x), b(x))

    def test_prob5_spec(self):
        spec = {"net_type": "prob5", "hidden_sizes": [16, 8], "input_size": 196,
                "encoder_name": "perspective196", "activation": "relu",
                "raw_logits": True}
        net = GC.build_net_from_spec(spec)
        assert net.raw_logits is True
        assert net(torch.rand(3, 196)).shape == (3, 5)


class TestProbe:
    def test_returns_none_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert GC.probe_worker_vram({}, {}, 1024) is None

    @pytest.mark.cuda
    def test_measures_a_context_sized_footprint(self):
        """One spawned worker with a tiny net must still cost a CUDA
        context's worth of VRAM. Bounds are generous: the value is
        driver/torch/GPU dependent."""
        torch.manual_seed(0)
        net = TDNetwork(hidden_sizes=[16, 8], input_size=196,
                        output_mode="equity")
        spec = {"net_type": "td", "hidden_sizes": [16, 8], "input_size": 196,
                "output_mode": "equity", "encoder_name": "perspective196",
                "activation": "relu"}
        got = GC.probe_worker_vram(spec, net.state_dict(), 4096)
        assert got is not None
        assert 64 * MB <= got <= 4 * GB, got
