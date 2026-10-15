"""Release-behavior tests for --oneply --device cuda.

Covers, without needing a GPU:
  * the train_batch and train_prob5 routing tables: the C collector by
    default; under --no-c-collect, AUTO vectorized under cuda+oneply, the
    --no-vectorized-collect escape, deference to explicit collector
    choices, the mutual-exclusion error;
  * the VRAM preflight arithmetic (pure functions);
  * the prob5 PER-WORKER cuda+oneply rejection, and the collect-device
    fallbacks: once the vectorized collector is disabled (VRAM
    preflight, workers<=1), no dispatch path may put 1-ply collection
    on a non-CPU device;
  * activation reaching the rebuilt worker nets on BOTH train_batch
    collector paths: a tanh checkpoint driven through each worker entry
    point must produce different collected data than the same weights
    rebuilt as relu. The prob5 vectorized worker's twin lives in
    tests/test_prob5_vectorized_parity.py.
"""
from _pool_fakes import patch_pool_map  # noqa: E402
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
from model import ProbNetwork, TDNetwork
from modes import CubelessMoneyMode


# ── routing table ────────────────────────────────────────────────────

def _args(**kw):
    """train_batch's CLI namespace, --no-c-collect unless overridden."""
    d = dict(oneply=True, device="cuda", vectorized_collect=False,
             no_vectorized_collect=False, c_engine=False,
             bf16_collect=False, oneply_acting=False, c_collect=False,
             no_c_collect=True)
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
        """Auto-routing must keep the CPU collector under --oneply-acting
        (the vectorized worker hard-errors on it)."""
        a = _args(oneply_acting=True)
        notes = TB.resolve_cuda_oneply_collector(a)
        assert a.vectorized_collect is False
        assert any("oneply-acting" in n for n in notes)

    def test_oneply_acting_explicit_vectorized_untouched(self):
        # Asking for both explicitly raises ValueError in Trainer.train;
        # the resolver guards only the AUTO path.
        a = _args(oneply_acting=True, vectorized_collect=True)
        assert TB.resolve_cuda_oneply_collector(a) == []
        assert a.vectorized_collect is True

    def test_c_collect_is_not_auto_routed(self):
        """--c-collect --oneply --device cuda stays on the C collector."""
        a = _args(c_collect=True, no_c_collect=False)
        notes = TB.resolve_cuda_oneply_collector(a)
        assert a.vectorized_collect is False
        assert not getattr(a, "vectorized_auto", False)
        assert any("--c-collect" in n and "keeps the C collector" in n
                   for n in notes)

    def test_c_collect_with_vectorized_is_an_error(self):
        for kw in ({}, {"device": "cpu"}, {"oneply": False}):
            with pytest.raises(SystemExit, match="mutually exclusive"):
                TB.resolve_cuda_oneply_collector(
                    _args(c_collect=True, no_c_collect=False,
                          vectorized_collect=True, **kw))

    def test_c_collect_off_cuda_or_zeroply_is_noop(self):
        for kw in ({"device": "cpu"}, {"oneply": False}):
            a = _args(c_collect=True, no_c_collect=False, **kw)
            assert TB.resolve_cuda_oneply_collector(a) == []
            assert a.vectorized_collect is False

    def test_no_c_collect_resolves_to_the_python_collector(self):
        for kw in ({}, {"device": "cpu"}, {"oneply": False}):
            a = _args(**kw)
            TB.resolve_cuda_oneply_collector(a)
            assert a.c_collect is False and a.c_collect_auto is False

    def test_c_collect_with_no_c_collect_is_an_error(self):
        with pytest.raises(SystemExit, match="mutually exclusive"):
            TB.resolve_cuda_oneply_collector(_args(c_collect=True))


class TestDefaultRouting:
    """No --c-collect / --no-c-collect: the C collector."""

    @staticmethod
    def _a(**kw):
        return _args(no_c_collect=False, **kw)

    def test_default_is_the_c_collector(self):
        for kw in ({}, {"device": "cpu"}, {"oneply": False}):
            a = self._a(**kw)
            notes = TB.resolve_cuda_oneply_collector(a)
            assert a.c_collect is True and a.c_collect_auto is True, kw
            assert a.vectorized_collect is False
            assert not getattr(a, "vectorized_auto", False)
            # Only --oneply --device cuda (the --no-c-collect auto-route)
            # gets a NOTE.
            assert len(notes) == (0 if kw else 1), notes

    def test_oneply_cuda_note(self):
        a = self._a()
        notes = TB.resolve_cuda_oneply_collector(a)
        assert notes and notes[0].startswith(
            "NOTE: the C collector (default): --oneply --device cuda keeps "
            "the C collector")

    def test_explicit_c_collect_is_the_same_collector(self):
        a = self._a(c_collect=True)
        TB.resolve_cuda_oneply_collector(a)
        assert a.c_collect is True and a.c_collect_auto is False

    def test_explicit_vectorized_wins_with_a_note(self):
        for kw in ({}, {"device": "cpu"}, {"oneply": False}):
            a = self._a(vectorized_collect=True, **kw)
            notes = TB.resolve_cuda_oneply_collector(a)
            assert a.c_collect is False and a.c_collect_auto is False
            assert a.vectorized_collect is True
            assert not getattr(a, "vectorized_auto", False)
            assert notes == ["NOTE: --vectorized-collect: the vectorized GPU "
                             "collector replaces the default C collector "
                             "(--c-collect with it is an error)."]

    def test_no_vectorized_collect_keeps_c(self):
        a = self._a(no_vectorized_collect=True)
        notes = TB.resolve_cuda_oneply_collector(a)
        assert a.c_collect is True and a.vectorized_collect is False
        assert len(notes) == 1 and "add --no-c-collect for the Python CPU" in notes[0]

    def test_python_collector_options_get_no_routing_note(self):
        """Without the fallback reasons (c_fallbacks=None) the resolver
        assumes C applies: --c-engine / --bf16-collect / --oneply-acting
        then fall back in Trainer.train (its NOTE); the resolver adds
        none."""
        for kw in ({"c_engine": True}, {"bf16_collect": True},
                   {"oneply_acting": True}):
            a = self._a(**kw)
            assert TB.resolve_cuda_oneply_collector(a) == []
            assert a.c_collect is True and a.vectorized_collect is False

    WHY = "--boltzmann-temp 0.5 (the C collector plays argmin)"

    def test_default_ruled_out_falls_back_to_the_no_c_collect_routing(self):
        """A precondition miss under the default: a NOTE per reason, then
        the --no-c-collect routing: --oneply --device cuda auto-routes to
        the vectorized GPU collector."""
        a = self._a()
        notes = TB.resolve_cuda_oneply_collector(a, [self.WHY, "second reason"])
        assert a.c_collect is False and a.c_collect_auto is False
        assert a.vectorized_collect is True and a.vectorized_auto is True
        assert len(notes) == 3
        assert notes[0] == (f"NOTE: --c-collect (default): {self.WHY}; falling "
                            f"back to the vectorized GPU collector (the "
                            f"--no-c-collect routing).")
        assert notes[1].startswith("NOTE: --c-collect (default): second reason;")
        assert notes[2].startswith("AUTO: --oneply --device cuda -> vectorized")

    def test_default_ruled_out_elsewhere_is_python(self):
        for kw in ({"device": "cpu"}, {"oneply": False}):
            a = self._a(**kw)
            notes = TB.resolve_cuda_oneply_collector(a, [self.WHY])
            assert a.c_collect is False and a.vectorized_collect is False
            assert notes == [f"NOTE: --c-collect (default): {self.WHY}; falling "
                             f"back to the Python collector (the --no-c-collect "
                             f"routing)."]

    def test_default_ruled_out_keeps_the_cpu_collector_cases(self):
        """--oneply-acting / --c-engine / --no-vectorized-collect keep the
        CPU collector, with their NOTE, as under --no-c-collect."""
        for kw, needle in (({"oneply_acting": True}, "--oneply-acting is not supported"),
                           ({"c_engine": True}, "accelerator"),
                           ({"no_vectorized_collect": True}, "keeping the CPU collector")):
            a = self._a(**kw)
            notes = TB.resolve_cuda_oneply_collector(a, ["why"])
            assert a.c_collect is False and a.vectorized_collect is False, kw
            assert len(notes) == 2 and "falling back to the Python collector" in notes[0]
            assert needle in notes[1], (kw, notes)

    def test_explicit_c_collect_ignores_the_reasons(self):
        """--c-collect: Trainer.train's NOTE + Python collector;
        c_collect_default_fallbacks returns [] for it anyway."""
        a = _args(c_collect=True, no_c_collect=False)
        notes = TB.resolve_cuda_oneply_collector(a, [self.WHY])
        assert a.c_collect is True and a.vectorized_collect is False
        assert len(notes) == 1 and "keeps the C collector" in notes[0]

    def test_default_fallbacks(self, monkeypatch):
        """c_collect_default_fallbacks: the preconditions, then the library;
        nothing when an explicit flag decides."""
        from modes import CubefulMoneyMode
        net = TDNetwork(hidden_sizes=[8], input_size=200, output_mode="equity",
                        encoder_name="cubeful_perspective196")
        lib = []
        monkeypatch.setattr(TB, "c_collect_library", lambda b: lib[0] if lib else None)

        def ns(**kw):
            d = dict(c_collect=False, no_c_collect=False, vectorized_collect=False,
                     oneply_acting=False, boltzmann_temp=0.0, bf16_collect=False,
                     c_engine=False, c_collect_backend="simd")
            d.update(kw)
            return argparse.Namespace(**d)

        mode = CubefulMoneyMode()
        assert TB.c_collect_default_fallbacks(ns(), net, mode) == []
        r = TB.c_collect_default_fallbacks(ns(boltzmann_temp=0.5), net, mode)
        assert len(r) == 1 and r[0].startswith("--boltzmann-temp 0.5")
        lib.append("libpuretd_simd.so not found and ... did not build it")
        assert TB.c_collect_default_fallbacks(ns(), net, mode) == lib
        for kw in ({"c_collect": True}, {"no_c_collect": True},
                   {"vectorized_collect": True}):
            assert TB.c_collect_default_fallbacks(ns(boltzmann_temp=0.5, **kw),
                                                  net, mode) == []


# ── VRAM preflight arithmetic ────────────────────────────────────────

GB = 2**30
MB = 2**20


class TestVramCap:
    def test_plenty_returns_requested(self):
        assert T.vectorized_vram_cap(24, 32 * GB) == 24

    def test_fallback_is_shared_with_prob5(self):
        # The fallback (used only when the startup probe cannot run) is
        # ONE constant shared by both trainers.
        import train_prob5 as P5
        assert T.VEC_PER_WORKER_BYTES == P5.VEC_PER_WORKER_BYTES
        assert T.VEC_RESERVE_BYTES == P5.VEC_RESERVE_BYTES
        assert T.vectorized_vram_cap(64, int(7.9 * GB)) == \
            P5.vectorized_vram_cap(64, int(7.9 * GB))

    def test_fallback_is_conservative_on_8gb(self):
        # ~7.9 GB free, no measurement: the cap must stay conservative.
        cap = T.vectorized_vram_cap(64, int(7.9 * GB))
        assert 2 <= cap < 16, cap

    def test_measured_per_worker_is_honored(self):
        # The probe's budget replaces the fallback: 400 MiB/worker on the
        # same 7.9 GB card admits ~17, not the fallback's 7.
        cap = T.vectorized_vram_cap(64, int(7.9 * GB), 400 * MB)
        assert 16 <= cap <= 18, cap
        # ...and a big measured context caps hard.
        assert T.vectorized_vram_cap(64, int(7.9 * GB), 1500 * MB) <= 5

    def test_tiny_vram_returns_zero(self):
        assert T.vectorized_vram_cap(24, 600 * MB) == 0

    def test_free_below_reserve_returns_zero(self):
        assert T.vectorized_vram_cap(24, T.VEC_RESERVE_BYTES - 1) == 0

    def test_zero_requested(self):
        assert T.vectorized_vram_cap(0, 32 * GB) == 0

    def test_monotone_in_free_bytes(self):
        caps = [T.vectorized_vram_cap(64, f * GB) for f in range(0, 33, 2)]
        assert caps == sorted(caps)


# ── prob5: PER-WORKER cuda rejection (the vectorized path is separate) ─
#
# --oneply --device cuda auto-selects the vectorized GPU collector. The
# rejection applies to the PER-WORKER path only, i.e. when the vectorized
# collector has been declined.

class TestProb5PerWorkerCudaRejection:
    def test_oneply_cuda_rejected_when_vectorized_declined(self):
        with pytest.raises(SystemExit) as ei:
            P5.reject_cuda_oneply(
                argparse.Namespace(oneply=True, device="cuda"))
        assert "prob5 1-ply" in str(ei.value)
        assert "per-worker" in str(ei.value)

    def test_oneply_cpu_ok(self):
        P5.reject_cuda_oneply(argparse.Namespace(oneply=True, device="cpu"))

    def test_zeroply_cuda_ok(self):
        P5.reject_cuda_oneply(argparse.Namespace(oneply=False, device="cuda"))


def _p5_args(**kw):
    # --no-c-collect unless overridden.
    base = dict(oneply=True, device="cuda", workers=32,
                episodes_per_round=1000, vectorized_collect=0,
                no_vectorized_collect=False, vectorized_auto=False,
                gpu_eval_server=False, pipeline_collect=False,
                c_collect=False, no_c_collect=True)
    base.update(kw)
    return argparse.Namespace(**base)


class TestProb5CudaOneplyRouting:
    def test_auto_enables_vectorized_with_single_batch_default(self):
        a = _p5_args(workers=32, episodes_per_round=1000)
        notes = P5.resolve_cuda_oneply_collector(a)
        assert a.vectorized_collect == 32   # ceil(1000/32)
        assert a.vectorized_auto is True
        assert any("AUTO" in n for n in notes)

    def test_cuda_index_device_auto_routes(self):
        # "cuda:0" etc. must hit the same routing as bare "cuda".
        a = _p5_args(device="cuda:0")
        notes = P5.resolve_cuda_oneply_collector(a)
        assert a.vectorized_collect == 32
        assert any("AUTO" in n for n in notes)

    def test_cpu_device_is_untouched(self):
        a = _p5_args(device="cpu")
        assert P5.resolve_cuda_oneply_collector(a) == []
        assert a.vectorized_collect == 0

    def test_zeroply_is_untouched(self):
        a = _p5_args(oneply=False)
        assert P5.resolve_cuda_oneply_collector(a) == []
        assert a.vectorized_collect == 0

    def test_explicit_batch_is_respected(self):
        a = _p5_args(vectorized_collect=8)
        P5.resolve_cuda_oneply_collector(a)
        assert a.vectorized_collect == 8
        assert a.vectorized_auto is False

    def test_explicit_collector_choice_wins(self):
        # Explicit beats implicit: --gpu-eval-server / --pipeline-collect
        # are collector choices. AUTO must not discard them.
        for kw in ({"gpu_eval_server": True}, {"pipeline_collect": True}):
            a = _p5_args(**kw)
            notes = P5.resolve_cuda_oneply_collector(a)
            assert a.vectorized_collect == 0, kw
            assert any("wins" in n for n in notes), kw

    def test_opt_out_leaves_it_off_so_rejection_fires(self):
        a = _p5_args(no_vectorized_collect=True)
        notes = P5.resolve_cuda_oneply_collector(a)
        assert a.vectorized_collect == 0
        assert any("--device cpu" in n for n in notes)
        with pytest.raises(SystemExit):
            P5.reject_cuda_oneply(a)

    def test_mutually_exclusive(self):
        a = _p5_args(vectorized_collect=8, no_vectorized_collect=True)
        with pytest.raises(SystemExit) as ei:
            P5.resolve_cuda_oneply_collector(a)
        assert "mutually exclusive" in str(ei.value)


class TestProb5VectorizedBatchDefault:
    def test_single_batch_rule(self):
        # B must cover a worker's whole round in one fused forward.
        assert P5.default_vectorized_batch(1000, 32) == 32   # 31.25 -> 32
        assert P5.default_vectorized_batch(1000, 72) == 14   # 13.9  -> 14
        assert P5.default_vectorized_batch(1000, 24) == 42
        assert P5.default_vectorized_batch(1000, 1000) == 1

    def test_degenerate_workers(self):
        assert P5.default_vectorized_batch(1000, 0) == 0


class TestProb5VramPreflight:
    GiB = 1024 ** 3

    def test_8gb_card_caps_hard(self):
        # 8 GB fits only a handful of contexts.
        assert P5.vectorized_vram_cap(96, 8 * self.GiB) < 16

    def test_16gb_card_still_usable(self):
        # Fewer workers, but above the 2-worker fallback threshold.
        assert P5.vectorized_vram_cap(96, 16 * self.GiB) >= 2

    def test_96gb_card_does_not_cap_a_modest_request(self):
        assert P5.vectorized_vram_cap(32, 90 * self.GiB) == 32

    def test_never_exceeds_request(self):
        assert P5.vectorized_vram_cap(8, 90 * self.GiB) == 8

    def test_no_vram_yields_zero(self):
        assert P5.vectorized_vram_cap(32, 0) == 0
        assert P5.vectorized_vram_cap(0, 90 * self.GiB) == 0

    def test_8gb_card_is_below_the_auto_floor(self):
        # An 8 GB card must trip the auto-routing fallback.
        assert P5.vectorized_vram_cap(96, 8 * self.GiB) < P5.VEC_MIN_WORKERS


# ── prob5 collect-device fallbacks (no per-worker GPU, ever) ─────────

def _p5_net():
    torch.manual_seed(11)
    net = ProbNetwork(hidden_sizes=[16, 8], input_size=196,
                      activation="relu", encoder_name="perspective196",
                      raw_logits=False)
    net.eval()
    return net


def _p5_train_kwargs(**kw):
    base = dict(num_episodes=4, lr=1e-6, end_lr=None,
                episodes_per_round=4, batch_size=8, workers=2,
                device="cuda", oneply=True, pw_weight=1.0, use_mse=False,
                seed=3, log_every=0, save_path=None, save_every=0,
                warmup_cycles=0)
    base.update(kw)
    return base


_EMPTY_EP = (np.empty((0, 196), np.float32), np.empty((0, 5), np.float32))


class TestProb5CollectDeviceFallbacks:
    """Once the vectorized collector is disabled, no dispatch path may
    put 1-ply collection on a non-CPU device: the 'falling back to the
    CPU collector' message must be what happens."""

    def test_vram_fallback_dispatches_cpu_workers(self, monkeypatch):
        """VRAM preflight fallback: the pooled _collect_worker args must
        carry a CPU collect device, not the master's cuda device."""
        import multiprocessing.pool as mpp

        captured = []
        probed = []

        def fake_map(self, fn, iterable, chunksize=None):
            batch = list(iterable)
            captured.extend(batch)
            return [_EMPTY_EP + (0,) for _ in batch]

        patch_pool_map(monkeypatch, fake_map)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "mem_get_info",
            lambda *a: probed.append(a) or (2 << 30, 8 << 30))
        # The per-worker preflight PROBES one worker's VRAM in a spawned
        # child. Stub it (500 MiB -> ~640 MiB budget, so 2 GB free fits
        # one worker: below the auto floor).
        import gpu_context as GC
        monkeypatch.setattr(GC, "probe_worker_vram", lambda *a, **kw: 500 << 20)

        # This is the PER-WORKER-context preflight (the only one with a
        # per-worker VRAM cap). Pin the topology so the default cannot
        # repurpose the test.
        P5.train(_p5_net(), **_p5_train_kwargs(
            device="cuda:1", vectorized_collect=2, vectorized_auto=True,
            vectorized_shared_context=False))

        # Preflight probed the device the user named, not device 0.
        assert probed and probed[0][0] == torch.device("cuda:1")
        assert len(captured) == 2, "one args tuple per worker"
        for args in captured:
            assert len(args) == 11, f"arity drifted: {len(args)}"
            assert args[-3] is True, "oneply field moved"
            assert args[-2] == "cpu", (
                "VRAM fallback dispatched a non-CPU collect device")

    def test_explicit_request_survives_below_the_auto_floor(self, monkeypatch):
        """Same card, but --vectorized-collect was explicit: honored down
        to 2 workers, so with room for exactly 2 the GPU collector runs
        (the vectorized args carry the cuda device, not a CPU fallback)."""
        import multiprocessing.pool as mpp

        captured = []

        def fake_map(self, fn, iterable, chunksize=None):
            batch = list(iterable)
            captured.extend(batch)
            return [_EMPTY_EP + (0,) for _ in batch]

        patch_pool_map(monkeypatch, fake_map)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        # 1 GiB reserve + 2 x 640 MiB budgets -> fits 2, not 8.
        monkeypatch.setattr(torch.cuda, "mem_get_info",
                            lambda *a: ((1 << 30) + (1300 << 20), 8 << 30))
        import gpu_context as GC
        monkeypatch.setattr(GC, "probe_worker_vram", lambda *a, **kw: 500 << 20)

        P5.train(_p5_net(), **_p5_train_kwargs(
            device="cuda:1", vectorized_collect=2, vectorized_auto=False,
            vectorized_shared_context=False))
        assert captured and all(len(a) == 11 for a in captured)
        assert all(a[-2] == "cuda:1" for a in captured), (
            "explicit request below the auto floor was not honored")

    def test_auto_small_pool_with_plenty_of_vram_stays_on_gpu(self, monkeypatch):
        """An auto-routed --workers 4 on a card that fits far more must
        NOT fall back: the auto floor is about VRAM starving the pool."""
        import multiprocessing.pool as mpp

        captured = []

        def fake_map(self, fn, iterable, chunksize=None):
            batch = list(iterable)
            captured.extend(batch)
            return [_EMPTY_EP + (0,) for _ in batch]

        patch_pool_map(monkeypatch, fake_map)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info",
                            lambda *a: (64 << 30, 96 << 30))
        import gpu_context as GC
        monkeypatch.setattr(GC, "probe_worker_vram", lambda *a, **kw: 500 << 20)

        P5.train(_p5_net(), **_p5_train_kwargs(
            workers=4, device="cuda:0", vectorized_collect=2,
            vectorized_auto=True, vectorized_shared_context=False))
        assert len(captured) == 4
        assert all(len(a) == 11 and a[-2] == "cuda:0" for a in captured), (
            "auto-routed small pool fell back to the CPU collector")

    def test_auto_vram_bound_below_floor_falls_back(self, monkeypatch):
        """...but when VRAM caps a bigger request below the floor, an
        auto-routed run does fall back (16 asked, 6 fit, floor 8)."""
        import multiprocessing.pool as mpp

        captured = []

        def fake_map(self, fn, iterable, chunksize=None):
            batch = list(iterable)
            captured.extend(batch)
            return [_EMPTY_EP + (0,) for _ in batch]

        patch_pool_map(monkeypatch, fake_map)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        # 1 GiB reserve + 6 x 640 MiB budgets.
        monkeypatch.setattr(torch.cuda, "mem_get_info",
                            lambda *a: ((1 << 30) + (6 * 640 << 20) + (10 << 20), 8 << 30))
        import gpu_context as GC
        monkeypatch.setattr(GC, "probe_worker_vram", lambda *a, **kw: 500 << 20)
        monkeypatch.setattr(GC, "physical_cores", lambda: 64)

        P5.train(_p5_net(), **_p5_train_kwargs(
            workers=16, device="cuda:0", vectorized_collect=2,
            vectorized_auto=True, vectorized_shared_context=False))
        # 4 episodes over a 16-wide CPU pool -> 4 non-empty tasks, all CPU.
        assert captured, "nothing dispatched"
        assert all(len(a) == 11 and a[-2] == "cpu" for a in captured), (
            "VRAM-bound auto run below the floor did not fall back")

    def test_shared_topology_auto_falls_back_when_vram_is_tiny(self, monkeypatch):
        """Shared topology has no per-worker contexts, so the floor rule
        does not apply. Its analogue is the eval server's minimum free
        VRAM. An auto-routed run below it must dispatch CPU workers (no
        server is started)."""
        import multiprocessing.pool as mpp

        captured = []

        def fake_map(self, fn, iterable, chunksize=None):
            batch = list(iterable)
            captured.extend(batch)
            return [_EMPTY_EP + (0,) for _ in batch]

        patch_pool_map(monkeypatch, fake_map)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info",
                            lambda *a: (256 << 20, 8 << 30))

        P5.train(_p5_net(), **_p5_train_kwargs(
            device="cuda:1", vectorized_collect=2, vectorized_auto=True,
            vectorized_shared_context=True))
        assert len(captured) == 2
        for args in captured:
            assert len(args) == 11 and args[-2] == "cpu", (
                "shared-topology VRAM fallback dispatched a non-CPU device")

    def test_workers1_cuda_vectorized_raises(self):
        """The vectorized collector needs a pool. With --workers 1 a
        cuda run must fail loudly, not crawl on per-worker cuda (or
        silently on the single-process CPU collector)."""
        with pytest.raises(SystemExit) as ei:
            P5.train(_p5_net(), **_p5_train_kwargs(
                workers=1, vectorized_collect=8, vectorized_auto=True))
        assert "--workers" in str(ei.value)

    def test_single_process_collects_on_cpu(self, monkeypatch):
        """Library route (no CLI rejection): single-process 1-ply with a
        cuda master device must still collect on the CPU."""
        seen = []

        def fake_collect(net, rng, encode_fn, gpe_fn, oneply, device,
                         remote=None, max_plies=0):
            seen.append(device)
            return _EMPTY_EP

        monkeypatch.setattr(P5, "_collect_one_episode", fake_collect)
        P5.train(_p5_net(), **_p5_train_kwargs(workers=1))
        assert seen and set(seen) == {"cpu"}

    def test_negative_batch_rejected(self):
        with pytest.raises(SystemExit) as ei:
            P5.train(_p5_net(), **_p5_train_kwargs(
                device="cpu", vectorized_collect=-1))
        assert "--vectorized-collect" in str(ei.value)


# ── activation threading: tanh must not be rebuilt as relu ──────────

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
    runs MUST differ."""

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
