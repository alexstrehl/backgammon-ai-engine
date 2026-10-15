"""Per-request eval server (`coalesce=False`): failure reporting and the
weight-push / forward serialisation. CPU-only: the service loop's control
flow is device-independent.

Both tests drive the server OUTSIDE the trainers' call ordering: the
guarantees here are the server's own, not the callers'.
"""
import multiprocessing as mp
import os
import sys
import threading
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu_eval_server import EvalServer, RemoteEvaluator, claim_slot
from model import TDNetwork

N_FEAT = 196


def _spec(hidden, input_size=N_FEAT):
    return {"net_type": "td", "hidden_sizes": list(hidden),
            "input_size": input_size, "output_mode": "equity",
            "encoder_name": "perspective196", "activation": "relu"}


def _net(spec, seed):
    torch.manual_seed(seed)
    return TDNetwork(**{k: v for k, v in spec.items() if k != "net_type"})


def _server(net, spec, slots=1, max_rows=4096):
    srv = EvalServer(mp.get_context("spawn"), num_slots=slots, n_feat=N_FEAT,
                     out_dim=1, max_rows=max_rows, build_spec=spec,
                     init_state_dict=net.state_dict(), device="cpu",
                     coalesce=False, n_threads=4)
    srv.start()
    return srv


class TestFailureReporting:
    def test_forward_error_raises_promptly_with_cause(self, monkeypatch):
        """A forward that raises inside a service thread must surface in
        the client at once and name the real cause."""
        # Failure injection: a spec whose input width disagrees with the
        # slot's n_feat, so every forward raises a shape error in net().
        spec = _spec([16, 8], input_size=100)
        net = _net(spec, 0)
        # Short timeout: an unpropagated error fails fast instead of hanging.
        monkeypatch.setattr(RemoteEvaluator, "WAIT_TIMEOUT_S", 30.0)
        srv = _server(net, spec)
        try:
            r = claim_slot(srv.client_handles())
            x = np.random.rand(8, N_FEAT).astype(np.float32)
            t0 = time.monotonic()
            with pytest.raises(RuntimeError, match="failed on slot 0") as ei:
                r.forward(x)
            assert time.monotonic() - t0 < 10.0, "took the timeout path"
            msg = str(ei.value)
            assert "did not answer" not in msg
            # torch's own text for the injected fault, not a guess.
            assert "RuntimeError" in msg and ("mat" in msg or "shape" in msg), msg
            # The process is up and its threads still serve: a weight
            # push acks, a second request is answered (with the same
            # failure, at once), and nothing hangs.
            assert srv.proc.is_alive()
            srv.update_weights(net.state_dict())
            t0 = time.monotonic()
            with pytest.raises(RuntimeError, match="failed on slot 0"):
                r.forward(x)
            assert time.monotonic() - t0 < 10.0
            assert srv.proc.is_alive()
        finally:
            srv.stop()

    def test_handles_without_channel_still_work(self):
        """A RemoteEvaluator built on handles that carry no failure channel
        (errors=None) still works: the flag is simply absent."""
        spec = _spec([16, 8])
        net = _net(spec, 1)
        srv = _server(net, spec)
        try:
            h = srv.client_handles()
            h.errors = None
            h.error_msg = None
            r = claim_slot(h)
            assert r.err is None
            x = np.random.rand(3, N_FEAT).astype(np.float32)
            with torch.no_grad():
                ref = net(torch.from_numpy(x)).numpy()
            assert np.array_equal(r.forward(x), ref)
        finally:
            srv.stop()


class TestWeightForwardSerialisation:
    def test_no_torn_weights_under_concurrent_push(self):
        """A weight push overlapping in-flight requests must never yield a
        forward computed on a MIX of old and new parameters: every answer
        must equal one of the two references exactly.

        The references are taken THROUGH the server while it is quiescent,
        so the comparison never depends on cross-process CPU numerics."""
        spec = _spec([256, 256])          # a forward long enough to be hit
        net_a, net_b = _net(spec, 1), _net(spec, 2)
        x = np.random.rand(1024, N_FEAT).astype(np.float32)
        slots = 4
        srv = _server(net_a, spec, slots=slots)
        try:
            h = srv.client_handles()
            remotes = [claim_slot(h) for _ in range(slots)]
            ref_a = remotes[0].forward(x)
            srv.update_weights(net_b.state_dict())
            ref_b = remotes[0].forward(x)
            assert not np.allclose(ref_a, ref_b)

            bad = []
            stop = threading.Event()

            def hammer(r):
                while not stop.is_set():
                    out = r.forward(x)
                    if not (np.array_equal(out, ref_a) or np.array_equal(out, ref_b)):
                        bad.append(float(min(np.abs(out - ref_a).max(),
                                             np.abs(out - ref_b).max())))

            ts = [threading.Thread(target=hammer, args=(r,)) for r in remotes]
            for t in ts:
                t.start()
            sds = [net_a.state_dict(), net_b.state_dict()]
            for i in range(60):
                srv.update_weights(sds[i % 2])
            stop.set()
            for t in ts:
                t.join(timeout=120)
            assert not any(t.is_alive() for t in ts)
        finally:
            srv.stop()
        assert not bad, f"{len(bad)} torn answers, e.g. min|d| to either ref = {bad[0]:.3g}"
