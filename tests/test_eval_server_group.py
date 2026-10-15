"""EvalServerGroup: sharding shared-context workers across several
eval-server processes. CPU-only: the transport, slot dealing and weight
push under test are device-independent.
"""
import multiprocessing as mp
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gpu_context as GC
from gpu_eval_server import EvalServerGroup, GroupHandles, claim_slot
from model import TDNetwork


def _spec():
    return {"net_type": "td", "hidden_sizes": [16, 8], "input_size": 196,
            "output_mode": "equity", "encoder_name": "perspective196",
            "activation": "relu"}


class TestDefaultServers:
    def test_one_per_sixteen(self):
        assert GC.default_eval_servers(1) == 1
        assert GC.default_eval_servers(16) == 1
        assert GC.default_eval_servers(17) == 2
        assert GC.default_eval_servers(96) == 6

    def test_explicit_wins(self):
        assert GC.default_eval_servers(96, explicit=2) == 2
        assert GC.default_eval_servers(96, explicit=0) == 1


class TestGroup:
    def test_slots_dealt_round_robin_and_forwards_agree(self):
        torch.manual_seed(1)
        net = TDNetwork(**{k: v for k, v in _spec().items() if k != "net_type"})
        ctx = mp.get_context("spawn")
        grp = EvalServerGroup(ctx, num_slots=5, n_servers=2, n_feat=196, out_dim=1,
                              max_rows=64, build_spec=_spec(),
                              init_state_dict=net.state_dict(), device="cpu",
                              coalesce=False)
        assert grp.n_servers == 2
        assert [s.num_slots for s in grp.servers] == [3, 2]
        grp.start()
        try:
            h = grp.client_handles()
            assert isinstance(h, GroupHandles)
            remotes = [claim_slot(h) for _ in range(5)]
            # Worker i -> server i % 2; the slot ids within a server are
            # whatever its free-slot queue hands out.
            servers = [r.req_q for r in remotes]
            assert servers[0] is not servers[1] or servers[0] is servers[2]
            x = np.random.rand(7, 196).astype(np.float32)
            with torch.no_grad():
                ref = net(torch.from_numpy(x)).numpy()
            for r in remotes:
                out = r.forward(x)
                assert out.shape == (7,)
                assert np.array_equal(out, ref)
            # Weight push reaches every server.
            for p in net.parameters():
                p.data.add_(0.5)
            grp.update_weights(net.state_dict())
            with torch.no_grad():
                ref2 = net(torch.from_numpy(x)).numpy()
            assert not np.allclose(ref, ref2)
            for r in remotes:
                assert np.array_equal(r.forward(x), ref2)
        finally:
            grp.stop()

    def test_single_handles_still_claimable(self):
        torch.manual_seed(2)
        net = TDNetwork(**{k: v for k, v in _spec().items() if k != "net_type"})
        ctx = mp.get_context("spawn")
        grp = EvalServerGroup(ctx, num_slots=2, n_servers=1, n_feat=196, out_dim=1,
                              max_rows=64, build_spec=_spec(),
                              init_state_dict=net.state_dict(), device="cpu",
                              coalesce=False)
        grp.start()
        try:
            # The legacy single-server ClientHandles path.
            single = grp.servers[0].client_handles()
            r = claim_slot(single)
            x = np.random.rand(3, 196).astype(np.float32)
            with torch.no_grad():
                ref = net(torch.from_numpy(x)).numpy()
            assert np.array_equal(r.forward(x), ref)
        finally:
            grp.stop()
