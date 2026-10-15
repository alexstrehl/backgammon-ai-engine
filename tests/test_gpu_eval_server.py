"""Correctness tests for `gpu_eval_server.py`, the central GPU inference
server for matchplay 1-ply collection (`--gpu-eval-server`).

WHAT IS PROVEN, AND WITH WHAT TOLERANCE
=======================================
The load-bearing property is `TestBitExactCollection`:

    With the server running on device='cpu' with num_slots=1, an agent
    whose forwards are routed through it produces episodes that are
    BIT-IDENTICAL -- `np.array_equal`, zero tolerance, encodings AND
    targets AND shape -- to `trainer._collect_episode_matchplay` driven
    with the same `random.Random` seed against the same net.

READ THAT SCOPE CAREFULLY: num_slots=1 is a configuration `trainer.train`
never runs in (it sets num_slots=workers and falls back to the plain
collector below 2 workers). The original version of this file stopped
there, which meant the proof covered no shipping configuration. The
multi-slot statement now lives in
`TestDeterministicRoutingReproducesTheUnroutedRun`:

    With `--gpu-eval-deterministic` (one request per GPU call) a real
    multi-worker `Trainer.train` produces the SAME final weights, md5
    for md5, as the same seeded run with no server at all -- and the
    same weights on a rerun.

WITHOUT that flag it does not, and `TestCoalescingAndDeterminism` pins
why: the server fuses concurrently-pending requests, fusion is composed
by wall-clock arrival order, and fp32 is batch-composition sensitive at
the 1e-8 level, which is enough to flip a `min()` tie over candidate
moves. That is a reproducibility loss, not a correctness one, and the
trainer now prints which of the two modes a run is in.

Zero tolerance is achievable there, so anything looser would be hiding
something. Three conditions make it so, and all three are properties of
the port rather than luck:

  1. num_slots=1 means `_server_loop._run_batch` takes its `len(items)
     == 1` branch and forwards `in_arrs[slot][:n]` directly. No
     `np.concatenate`, so the tensor handed to the net is byte-identical
     to the one the serial path would have built, at the same offsets.
  2. device='cpu' means no host/device transfer and no cuDNN kernel
     selection: the same ATen kernel runs on the same bytes.
  3. `_server_loop` calls `torch.set_num_threads(1)`. The reference
     never did. Multi-threaded MKL sgemm changes the reduction order
     relative to the single-threaded collection worker, which would
     break exactness for a reason having nothing to do with this port.
     (The `single_threaded` fixture pins the test process to match.)

The sweep runs match lengths 2 and 3 (2pt exercises Crawford and
dead-cube suppression much harder), `met=None` vs the real
Kazaross-XG2 table (`_eval_new_game` takes a completely different
branch, changing the number of forwards), and both routing modes
(1-ply only, and +move-selection).

Exactness here is not decoration. It pins the RNG stream: the
`random.Random` draws are value-independent but their BRANCH structure
is conditioned on network values (a double gates the take/pass coins; a
game-ending move skips the reroll). A routing bug that perturbed one
value by 1e-7 near a `min()` tie would change the chosen move, hence the
trajectory, hence the sample count -- which surfaces immediately as a
SHAPE mismatch under exact comparison and would be entirely invisible
under a value tolerance applied to sorted rows.

WHERE A TOLERANCE IS USED, AND WHY IT CANNOT HIDE A BUG
=======================================================
Two configurations are NOT bit-exact against the serial path, by
construction rather than by defect:

  * num_slots > 1, where the server coalesces several workers' requests
    into one fused GEMM. A row then sits at an arbitrary offset in a
    larger matrix and fp32 reductions are batch-composition sensitive.
  * any CUDA server, which additionally differs in kernel selection.
    (Empirically this one often agrees exactly anyway -- a 3620-row
    request on the production 512/512/256/256 net matched the CPU
    forward with max |diff| 0.0 -- but that is a property of one net
    and one batch shape, not a guarantee, so the test does not assert
    it.)

`TestCrossSlotScatter` handles the first WITHOUT a tolerance, by
removing the confound instead of tolerating it: it drives the server
with a deliberately batch-composition-INVARIANT net (one hidden unit
selecting a single input column, so the forward is exactly
`sigmoid(relu(x[:, 0]))` elementwise and provably offset-independent in
fp32). Under that net a coalesced 4-slot batch must scatter back BIT
FOR BIT, so any surviving difference is necessarily a misaligned offset
or cross-slot contamination, not arithmetic. That is the test with
teeth for the scatter logic.

`TestCudaServer` (skipped without a GPU) is the only place a tolerance
appears, at atol=1e-5 on VALUES only, while shape, row count and row
order stay at zero tolerance. That cannot hide a real bug because every
failure mode that actually worries us here -- a permuted scatter, a
dropped terminal override, a `1.0 -` applied on the wrong side of the
`min()` -- moves values by O(0.1) to O(1.0), four to five orders of
magnitude above the tolerance, and a permutation is caught structurally
by the row-order test rather than numerically.

WHAT IS NOT CLAIMED
===================
Nothing here measures throughput. And `TestWorkerDeath`
proves the SERVER survives an abandoned request and keeps serving other
slots; it does not claim `multiprocessing.pool.Pool` recovers from a
dead worker. It cannot: the replacement re-runs the initializer and
finds no free slot. What IS claimed, in `TestPoolWorkerDeathIsLoud`, is
that the master finds out -- the bounded `free_slots.get` alone did not
achieve that, because it raises inside a throwaway replacement process
whose exception the pool swallows while it spawns the next victim.
"""

import multiprocessing as mp
import os
import random
import signal
import threading
import time

import numpy as np
import pytest
import torch

from met import DEFAULT_MET, MatchEquityTable
from model import TDNetwork
from modes import MatchplayMode
from td_agent import TDAgent
import trainer as T
from trainer import Trainer

import gpu_eval_server as GES
from gpu_eval_server import (
    ClientHandles,
    EvalServer,
    EvalServerError,
    EvalServerGone,
    EvalServerTimeout,
    RemoteEvaluator,
    build_spec_from,
)


# ── fixtures / helpers ──────────────────────────────────────────────


@pytest.fixture
def single_threaded():
    """Pin the TEST process to one torch thread, matching what
    `_server_loop` and `trainer._collect_worker` do. Without this the
    serial reference forward could use a different reduction order than
    the server and break exactness for an unrelated reason."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _net(encoder_name="matchplay_perspective196", hidden=(16, 8), seed=0,
         activation="relu"):
    torch.manual_seed(seed)
    n = TDNetwork(
        hidden_sizes=list(hidden), encoder_name=encoder_name,
        output_mode="probability", activation=activation,
    )
    n.eval()
    return n


def _agent(net, met=None):
    return TDAgent(net, met=met)


_HAVE_MET = os.path.exists(DEFAULT_MET)
_met_table = None


def _met():
    global _met_table
    if _met_table is None:
        _met_table = MatchEquityTable.from_xml(DEFAULT_MET)
    return _met_table


def _column_picker_net(n_feat):
    """A net whose forward is provably batch-composition invariant in
    fp32: one hidden unit reading input column 0 with weight 1, output
    weight 1, all biases 0, so `y = sigmoid(relu(x[:, 0]))` elementwise.

    Every matmul term but one is an exact 0.0, and adding zeros is exact
    in IEEE754 regardless of summation order, so a row's output cannot
    depend on which other rows share its batch. That is what lets
    `TestCrossSlotScatter` assert BITWISE equality on a coalesced batch.
    """
    n = TDNetwork(hidden_sizes=[1], input_size=n_feat,
                  output_mode="probability", activation="relu")
    with torch.no_grad():
        n.hidden_layers[0].weight.zero_()
        n.hidden_layers[0].weight[0, 0] = 1.0
        n.hidden_layers[0].bias.zero_()
        n.fc_output.weight.fill_(1.0)
        n.fc_output.bias.zero_()
    n.eval()
    return n


def _local_values(net, feats):
    """The exact expression the routed forward replaces."""
    with torch.no_grad():
        return net(torch.from_numpy(feats)).cpu().numpy().ravel()


class _Server:
    """Context manager around EvalServer: always stops, always unlinks,
    so a failing assertion cannot leak /dev/shm into the next test."""

    def __init__(self, net, num_slots=1, max_rows=4096, device="cpu",
                 route_moves=False, wait_timeout=60.0, start=True,
                 coalesce=True, max_batch_rows=GES.MAX_BATCH_ROWS_DEFAULT):
        self.ctx = mp.get_context("spawn")
        n_feat = net.input_size
        self.srv = EvalServer(
            self.ctx, num_slots=num_slots, n_feat=n_feat, out_dim=1,
            max_rows=max_rows, build_spec=build_spec_from(net),
            init_state_dict=net.state_dict(), device=device,
            route_moves=route_moves, wait_timeout=wait_timeout,
            coalesce=coalesce, max_batch_rows=max_batch_rows,
        )
        self._start = start

    def __enter__(self):
        if self._start:
            self.srv.start()
        return self.srv

    def __exit__(self, *exc):
        self.srv.stop()
        return False


def _shm_names(srv):
    return set(srv.handles.in_names) | set(srv.handles.out_names)


def _serial_episode(agent, mode, seed, initial_ms=None):
    return T._collect_episode_matchplay(
        agent, mode, random.Random(seed),
        initial_ms if initial_ms is not None else mode.initial_match_state(),
        oneply=True,
    )


# ── module-level entry points for spawned children ──────────────────
# These must be importable by name in a spawn child, so they live at
# module scope rather than inside the tests that use them.


def _child_report_slot(handles, out_q):
    """Claim a slot exactly as `trainer._worker_init` does, prove the
    slot's buffers are private by writing a fingerprint, and report."""
    ev = RemoteEvaluator(handles.free_slots.get(timeout=30), handles)
    ev.in_arr[0, 0] = float(ev.slot) + 100.0
    out_q.put((ev.slot, os.getpid()))
    time.sleep(2.0)  # hold the slot so siblings cannot reuse it


def _child_forward_loop(handles, slot, ready_evt):
    """Attach a slot and hammer the server until killed. Used to prove
    the server survives a client that dies mid-request."""
    ev = RemoteEvaluator(slot, handles)
    feats = np.zeros((512, handles.n_feat), dtype=np.float32)
    ready_evt.set()
    while True:
        ev.forward(feats)


# ── 1. bit-exact collection vs the serial collector ─────────────────


class TestBitExactCollection:
    """Routed collection vs `trainer._collect_episode_matchplay`, zero
    tolerance, on a CPU server with one slot."""

    @pytest.mark.parametrize("match_len", [2, 3])
    @pytest.mark.parametrize("use_met", [False, True])
    @pytest.mark.parametrize("route_moves", [False, True])
    def test_episode_is_bit_identical(self, single_threaded, match_len,
                                      use_met, route_moves):
        if use_met and not _HAVE_MET:
            pytest.skip("no MET xml on this host")
        met = _met() if use_met else None
        net = _net(seed=3)
        mode = MatchplayMode(match_len)
        seed = 7 * match_len + int(use_met)

        ref_enc, ref_tgt = _serial_episode(_agent(net, met), mode, seed)

        with _Server(net, num_slots=1, route_moves=route_moves) as srv:
            agent = _agent(net, met)
            agent._remote_eval = RemoteEvaluator(0, srv.client_handles())
            agent._remote_eval_moves = route_moves
            got_enc, got_tgt = _serial_episode(agent, mode, seed)

        assert got_enc.shape == ref_enc.shape, (
            "row count differs -- the routed run took a different "
            "trajectory, not merely different values"
        )
        assert np.array_equal(got_enc, ref_enc), "encodings differ"
        assert np.array_equal(got_tgt, ref_tgt), "targets differ"
        assert got_tgt.dtype == ref_tgt.dtype

    def test_many_episodes_stay_exact(self, single_threaded):
        """A single episode can get lucky; a chain of them cannot. Also
        exercises weight-independent reuse of the same slot."""
        net = _net(seed=11)
        mode = MatchplayMode(2)
        with _Server(net, num_slots=1) as srv:
            handles = srv.client_handles()
            for seed in range(6):
                ref = _serial_episode(_agent(net), mode, seed)
                agent = _agent(net)
                agent._remote_eval = RemoteEvaluator(0, handles)
                got = _serial_episode(agent, mode, seed)
                assert np.array_equal(got[0], ref[0]), f"enc seed={seed}"
                assert np.array_equal(got[1], ref[1]), f"tgt seed={seed}"

    def test_non_relu_activation_is_carried(self, single_threaded):
        """The reference's build_spec dropped `activation`, silently
        rebuilding every net as relu. That is invisible whenever the
        checkpoint happens to be relu -- which is every checkpoint in
        this repo today, so only an explicit test catches it."""
        net = _net(seed=5, activation="tanh")
        assert build_spec_from(net)["activation"] == "tanh"
        mode = MatchplayMode(2)
        ref = _serial_episode(_agent(net), mode, 4)
        with _Server(net, num_slots=1) as srv:
            agent = _agent(net)
            agent._remote_eval = RemoteEvaluator(0, srv.client_handles())
            got = _serial_episode(agent, mode, 4)
        assert np.array_equal(got[0], ref[0])
        assert np.array_equal(got[1], ref[1])


# ── 2. the routed forward's contract ────────────────────────────────


class TestForwardContract:
    """Row order, dtype, shape, perspective and the boundary cases."""

    def test_values_match_local_forward_exactly(self, single_threaded):
        net = _net(seed=1)
        rng = np.random.default_rng(0)
        feats = rng.standard_normal(
            (257, net.input_size), dtype=np.float32)
        with _Server(net) as srv:
            ev = RemoteEvaluator(0, srv.client_handles())
            got = ev.forward(feats)
        assert np.array_equal(got, _local_values(net, feats))

    def test_dtype_and_shape(self, single_threaded):
        net = _net(seed=1)
        feats = np.zeros((5, net.input_size), dtype=np.float32)
        with _Server(net) as srv:
            got = RemoteEvaluator(0, srv.client_handles()).forward(feats)
        assert got.dtype == np.float32
        assert got.shape == (5,), "out_dim==1 must return 1-D, like .ravel()"
        assert got.flags.writeable, (
            "value_oneply writes terminal overrides into this array"
        )

    def test_row_order_is_preserved(self, single_threaded):
        """A transposed or permuted scatter is the failure mode a value
        check alone would miss, so permute the INPUT and require the
        output to follow it exactly."""
        net = _net(seed=1)
        rng = np.random.default_rng(1)
        feats = rng.standard_normal((64, net.input_size), dtype=np.float32)
        perm = rng.permutation(64)
        with _Server(net) as srv:
            ev = RemoteEvaluator(0, srv.client_handles())
            straight = ev.forward(feats)
            shuffled = ev.forward(np.ascontiguousarray(feats[perm]))
        assert np.array_equal(shuffled, straight[perm])

    def test_perspective_is_not_flipped_by_routing(self, single_threaded):
        """The routing layer must return the raw opp-view network value.
        `value_oneply_checker_matchplay` applies `1.0 - x` exactly once,
        AFTER the per-dice `min()`; a flip inside the transport would
        silently turn that min into a max and still produce plausible
        in-range targets."""
        net = _net(seed=2)
        rng = np.random.default_rng(2)
        feats = rng.standard_normal((33, net.input_size), dtype=np.float32)
        local = _local_values(net, feats)
        with _Server(net) as srv:
            agent = _agent(net)
            agent._remote_eval = RemoteEvaluator(0, srv.client_handles())
            routed = agent._matchplay_values_np(feats)
        assert np.array_equal(routed, local)
        assert not np.allclose(routed, 1.0 - local), (
            "fixture net is degenerate: cannot distinguish v from 1-v"
        )

    def test_empty_request_short_circuits(self, single_threaded):
        net = _net(seed=1)
        with _Server(net) as srv:
            got = RemoteEvaluator(0, srv.client_handles()).forward(
                np.zeros((0, net.input_size), dtype=np.float32))
        assert got.shape == (0,)
        assert got.dtype == np.float32

    def test_oversized_request_returns_none(self, single_threaded, capsys):
        net = _net(seed=1)
        with _Server(net, max_rows=8) as srv:
            ev = RemoteEvaluator(0, srv.client_handles())
            assert ev.forward(
                np.zeros((9, net.input_size), dtype=np.float32)) is None
            assert ev.n_overflow == 1
            assert ev.forward(
                np.zeros((8, net.input_size), dtype=np.float32)) is not None
        assert "exceeds max_rows" in capsys.readouterr().err, (
            "the overflow cliff is silent by construction and must warn"
        )

    def test_overflow_falls_back_to_a_correct_local_forward(
            self, single_threaded):
        """max_rows=1 forces `value_oneply_checker_matchplay` down the
        local path on essentially every call. The episode must still be
        bit-exact -- an overflow is a throughput cliff, never a wrong
        answer."""
        net = _net(seed=9)
        mode = MatchplayMode(2)
        ref = _serial_episode(_agent(net), mode, 21)
        with _Server(net, max_rows=1) as srv:
            agent = _agent(net)
            ev = RemoteEvaluator(0, srv.client_handles())
            agent._remote_eval = ev
            got = _serial_episode(agent, mode, 21)
        assert ev.n_overflow > 0, "test did not actually exercise fallback"
        assert np.array_equal(got[0], ref[0])
        assert np.array_equal(got[1], ref[1])

    def test_move_routing_is_off_by_default(self, single_threaded):
        """`_remote_eval_moves` gates the second call site. With it off,
        move selection must issue zero server requests."""
        net = _net(seed=1)
        mode = MatchplayMode(2)
        with _Server(net) as srv:
            agent = _agent(net)
            ev = RemoteEvaluator(0, srv.client_handles())
            agent._remote_eval = ev
            _serial_episode(agent, mode, 3)
            only_1ply = ev.n_requests

            agent2 = _agent(net)
            ev2 = RemoteEvaluator(0, srv.client_handles())
            agent2._remote_eval = ev2
            agent2._remote_eval_moves = True
            _serial_episode(agent2, mode, 3)

        assert ev2.n_requests > only_1ply, (
            "routing move selection must add requests"
        )


class TestDefaultPathUnchanged:
    """The flag is opt-in: with no evaluator attached nothing changes."""

    def test_agent_defaults_to_local(self):
        agent = _agent(_net())
        assert agent._remote_eval is None
        assert agent._remote_eval_moves is False

    def test_values_np_matches_the_expression_it_replaced(self):
        net = _net(seed=6)
        agent = _agent(net)
        feats = np.random.default_rng(4).standard_normal(
            (17, net.input_size), dtype=np.float32)
        with torch.no_grad():
            expect = net(torch.from_numpy(feats)).cpu().numpy().ravel()
        assert np.array_equal(agent._matchplay_values_np(feats), expect)


# ── 3. cross-slot batching / scatter alignment ──────────────────────


class TestCrossSlotScatter:
    """Coalescing several slots into one fused forward must scatter the
    rows back to the right slot at the right offset."""

    def test_coalesced_batch_is_bitwise_correct(self):
        """Deterministic: queue four requests BEFORE starting the server,
        so its first `get()` plus `get_nowait()` drain is guaranteed to
        fuse all four into a single `_run_batch` call.

        Driven by `_column_picker_net`, whose output is provably
        offset-independent, so this asserts BITWISE equality: any
        difference is a misaligned scatter or cross-slot contamination,
        never arithmetic.
        """
        n_feat = 32
        net = _column_picker_net(n_feat)
        sizes = [11, 5, 23, 2]
        holder = _Server(net, num_slots=4, max_rows=64, start=False)
        srv = holder.srv
        try:
            handles = srv.client_handles()
            evs = [RemoteEvaluator(s, handles) for s in range(4)]
            expected = []
            for s, ev in enumerate(evs):
                feats = np.zeros((sizes[s], n_feat), dtype=np.float32)
                # Fingerprint column 0 per (slot, row) -- the only column
                # the net reads, so a swapped row is a wrong value.
                feats[:, 0] = np.arange(sizes[s], dtype=np.float32) + 10 * s
                ev.in_arr[:sizes[s]] = feats
                ev.event.clear()
                expected.append(_local_values(net, feats))
            for s in range(4):
                handles.req_q.put((s, sizes[s]))

            srv.start()
            for s, ev in enumerate(evs):
                assert ev.event.wait(timeout=30), f"slot {s} never answered"
                got = ev.out_arr[:sizes[s], 0].copy()
                assert np.array_equal(got, expected[s]), (
                    f"slot {s} got another slot's rows or a wrong offset"
                )
        finally:
            holder.srv.stop()

    def test_concurrent_clients_do_not_cross_talk(self):
        """The same property under real concurrency, where coalescing is
        opportunistic rather than forced."""
        n_feat = 24
        net = _column_picker_net(n_feat)
        errors = []
        with _Server(net, num_slots=4, max_rows=128) as srv:
            handles = srv.client_handles()
            barrier = threading.Barrier(4)

            def client(slot):
                try:
                    ev = RemoteEvaluator(slot, handles)
                    feats = np.zeros((16 + slot, n_feat), dtype=np.float32)
                    feats[:, 0] = (np.arange(16 + slot, dtype=np.float32)
                                   + 100 * slot)
                    want = _local_values(net, feats)
                    for _ in range(40):
                        barrier.wait(timeout=30)
                        got = ev.forward(feats)
                        if not np.array_equal(got, want):
                            errors.append((slot, got, want))
                            return
                except Exception as e:            # pragma: no cover
                    errors.append((slot, repr(e)))

            threads = [threading.Thread(target=client, args=(s,))
                       for s in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=120)
        assert not errors, errors[:2]


# ── 4. weight updates ───────────────────────────────────────────────


class TestWeightUpdate:
    """A stale-weights bug trains against a frozen net: the run
    completes, the loss curve looks normal, and nothing surfaces it.
    So assert the new weights are in force EXACTLY, not just that
    something changed."""

    def test_update_takes_effect_exactly(self, single_threaded):
        net_a = _net(seed=1)
        net_b = _net(seed=2)
        rng = np.random.default_rng(7)
        feats = rng.standard_normal((48, net_a.input_size), dtype=np.float32)
        va, vb = _local_values(net_a, feats), _local_values(net_b, feats)
        assert not np.allclose(va, vb), "fixture nets are indistinguishable"

        with _Server(net_a) as srv:
            ev = RemoteEvaluator(0, srv.client_handles())
            assert np.array_equal(ev.forward(feats), va)
            srv.update_weights(net_b.state_dict())
            got = ev.forward(feats)
        assert np.array_equal(got, vb), "server kept serving stale weights"

    def test_update_blocks_until_applied(self, single_threaded):
        """`update_weights` must not return before the server has loaded
        them -- otherwise the very first request of the next round races
        the load and silently uses the previous generation."""
        net_a, net_b = _net(seed=1), _net(seed=2)
        feats = np.random.default_rng(8).standard_normal(
            (4, net_a.input_size), dtype=np.float32)
        with _Server(net_a) as srv:
            ev = RemoteEvaluator(0, srv.client_handles())
            for _ in range(20):
                srv.update_weights(net_b.state_dict())
                assert np.array_equal(ev.forward(feats),
                                      _local_values(net_b, feats))
                srv.update_weights(net_a.state_dict())
                assert np.array_equal(ev.forward(feats),
                                      _local_values(net_a, feats))

    def test_trainer_pushes_weights_every_round(self, monkeypatch):
        """End to end: the master must call update_weights once per
        round, before dispatching that round's collect."""
        calls = []
        real = EvalServer.update_weights

        def spy(self, sd):
            calls.append({k: v.detach().clone() for k, v in sd.items()})
            return real(self, sd)

        monkeypatch.setattr(EvalServer, "update_weights", spy)
        losses = _tiny_train(workers=2, episodes=4, per_round=2,
                             gpu_eval_server=True, gpu_eval_device="cpu")
        assert len(calls) == 2, f"expected one push per round, got {len(calls)}"
        # Round 2's weights must differ from round 1's: identical pushes
        # would mean the training step never reached the server.
        k = next(iter(calls[0]))
        assert not torch.equal(calls[0][k], calls[1][k]), (
            "the second round pushed the same weights -- the server would "
            "have been collecting against a frozen net"
        )
        assert losses


# ── 5. slot claiming ────────────────────────────────────────────────


class TestSlotClaiming:
    def test_n_workers_claim_n_distinct_slots(self):
        """N spawned children, N slots, one each, no duplicates, and each
        child's input buffer is private to it."""
        net = _net(seed=1)
        n = 4
        with _Server(net, num_slots=n, max_rows=16) as srv:
            handles = srv.client_handles()
            ctx = mp.get_context("spawn")
            out_q = ctx.Queue()
            procs = [ctx.Process(target=_child_report_slot,
                                 args=(handles, out_q)) for _ in range(n)]
            for p in procs:
                p.start()
            got = [out_q.get(timeout=60) for _ in range(n)]
            for p in procs:
                p.join(timeout=30)

            slots = sorted(s for s, _ in got)
            pids = {pid for _, pid in got}
            assert slots == list(range(n)), f"slot collision: {slots}"
            assert len(pids) == n
            # Each child stamped slot+100 into its OWN buffer only.
            for s in range(n):
                arr = np.ndarray((16, net.input_size), dtype=np.float32,
                                 buffer=srv._in_shms[s].buf)
                assert arr[0, 0] == float(s) + 100.0, (
                    f"slot {s} buffer holds another slot's data"
                )

    def test_free_slots_is_exhausted_after_claiming(self):
        """There is exactly one slot per worker and slots are never
        returned -- the invariant that makes pool recycling fatal."""
        import queue as _q
        net = _net(seed=1)
        with _Server(net, num_slots=2, max_rows=8) as srv:
            h = srv.client_handles()
            assert {h.free_slots.get(timeout=10),
                    h.free_slots.get(timeout=10)} == {0, 1}
            with pytest.raises(_q.Empty):
                h.free_slots.get(timeout=1.0)

    def test_worker_init_raises_instead_of_blocking(self, monkeypatch):
        """A recycled or respawned worker finds no slot. It must raise a
        named error, not block forever with no traceback."""
        net = _net(seed=1)
        monkeypatch.setattr(T, "_WORKER_REMOTE_EVAL", None, raising=False)
        with _Server(net, num_slots=1, max_rows=8) as srv:
            h = srv.client_handles()
            T._worker_init(h)                      # claims slot 0
            assert T._WORKER_REMOTE_EVAL.slot == 0
            with pytest.raises(RuntimeError, match="no free shared-memory"):
                _worker_init_fast(h)
        T._WORKER_REMOTE_EVAL = None


def _worker_init_fast(handles):
    """`trainer._worker_init` with its 60 s slot wait shortened, so the
    exhaustion test does not cost a minute."""
    import queue as _q
    real_get = handles.free_slots.get

    class _Fast:
        def get(self, timeout=None):
            return real_get(timeout=1.0)

    class _H:
        def __getattr__(self, k):
            return getattr(handles, k)
        free_slots = _Fast()

    return T._worker_init(_H())


# ── 6. shutdown / leaks ─────────────────────────────────────────────


def _shm_exists(name):
    return os.path.exists(f"/dev/shm/{name}")


class TestShutdown:
    def test_stop_unlinks_every_segment_and_reaps_the_process(self):
        net = _net(seed=1)
        holder = _Server(net, num_slots=3, max_rows=64)
        srv = holder.__enter__()
        names = _shm_names(srv)
        assert all(_shm_exists(n) for n in names), "segments were not created"
        pid = srv.proc.pid
        holder.__exit__()
        assert not srv.proc.is_alive()
        leaked = [n for n in names if _shm_exists(n)]
        assert not leaked, f"leaked shared memory: {leaked}"
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    def test_stop_is_idempotent(self):
        net = _net(seed=1)
        with _Server(net) as srv:
            pass
        srv.stop()
        srv.stop()

    def test_stop_works_on_a_never_started_server(self):
        net = _net(seed=1)
        holder = _Server(net, start=False)
        names = _shm_names(holder.srv)
        holder.srv.stop()
        assert not [n for n in names if _shm_exists(n)]

    def test_trainer_leaves_nothing_behind(self):
        """The full opt-in path: no orphan server, no orphan segments."""
        seen = {}

        real_start = EvalServer.start

        def spy(self):
            real_start(self)
            seen["names"] = _shm_names(self)
            seen["pid"] = self.proc.pid

        EvalServer.start = spy
        try:
            losses = _tiny_train(workers=2, episodes=2, per_round=2,
                                 gpu_eval_server=True, gpu_eval_device="cpu")
        finally:
            EvalServer.start = real_start
        assert losses
        assert "names" in seen, "the server never started"
        assert not [n for n in seen["names"] if _shm_exists(n)]
        with pytest.raises(ProcessLookupError):
            os.kill(seen["pid"], 0)


# ── 7. failure modes: bounded, not hung ─────────────────────────────


class TestWorkerDeath:
    """A dead client must not wedge the server or its siblings.

    NOTE what this does and does not claim. It claims the SERVER keeps
    serving. It does not claim `multiprocessing.pool.Pool` recovers: it
    cannot, and `TestSlotClaiming::test_worker_init_raises_instead_of_
    blocking` covers the consolation prize (a loud error rather than a
    silent hang).
    """

    def test_abandoned_request_does_not_wedge_the_server(self, single_threaded):
        """Deterministic stand-in for 'killed after put, before wait':
        push a request nobody will ever wait on, then require another
        slot to be served correctly."""
        net = _net(seed=1)
        with _Server(net, num_slots=2, max_rows=64) as srv:
            h = srv.client_handles()
            orphan = RemoteEvaluator(1, h)
            orphan.in_arr[:8] = 0.0
            h.req_q.put((1, 8))                      # never waited on

            ev = RemoteEvaluator(0, h)
            feats = np.random.default_rng(5).standard_normal(
                (12, net.input_size), dtype=np.float32)
            assert np.array_equal(ev.forward(feats),
                                  _local_values(net, feats))

    def test_sigkilled_client_fails_bounded_never_hangs(self,
                                                        single_threaded):
        """A SIGKILLed client CAN permanently wedge the transport, and
        this test pins down the consequence.

        MEASURED, not assumed: run repeatedly, the surviving sibling
        sometimes gets its correct answer and sometimes never does. The
        cause is inherent to the reference's transport rather than to
        this port -- `multiprocessing.Queue` guards its pipe with a
        cross-process lock held by a feeder thread, and a process killed
        while holding it never releases it, so every other producer
        blocks forever. The per-slot `Event` has the same shape of
        hazard on the server's `set()`.

        The reference's response to that was an unbounded
        `event.wait()`: an infinite hang, no traceback, no checkpoint,
        no log line. The property this port actually guarantees, and the
        one asserted here, is BOUNDEDNESS -- within the wait budget the
        survivor either returns the right answer or raises
        EvalServerGone/EvalServerTimeout, which `pool.map` propagates so
        the run dies loudly and restartably. It never blocks forever.

        Asserting only the happy path here would be asserting a coin
        flip, which is why this test does not.
        """
        net = _net(seed=1)
        budget = 10.0
        with _Server(net, num_slots=2, max_rows=1024,
                     wait_timeout=budget) as srv:
            h = srv.client_handles()
            ctx = mp.get_context("spawn")
            ready = ctx.Event()
            child = ctx.Process(target=_child_forward_loop,
                                args=(h, 1, ready))
            child.start()
            try:
                assert ready.wait(timeout=60)
                time.sleep(0.5)              # let it get mid-request
                os.kill(child.pid, signal.SIGKILL)
                child.join(timeout=30)

                ev = RemoteEvaluator(0, h)
                feats = np.random.default_rng(6).standard_normal(
                    (20, net.input_size), dtype=np.float32)
                want = _local_values(net, feats)
                t0 = time.monotonic()
                try:
                    got = ev.forward(feats)
                except EvalServerError as e:
                    # Loud, bounded failure: acceptable.
                    assert time.monotonic() - t0 < 4 * budget, (
                        f"raised {e!r}, but only after "
                        f"{time.monotonic() - t0:.1f}s"
                    )
                else:
                    # Survived: then the answer must still be correct.
                    assert np.array_equal(got, want)
            finally:
                if child.is_alive():                  # pragma: no cover
                    child.kill()
                    child.join(timeout=10)


class TestServerFaultsRaise:
    """The reference waited on `event.wait()` with no timeout, so a dead
    server hung every worker and the master's `pool.map` forever -- no
    traceback, no checkpoint, no log line. These must raise instead."""

    def test_dead_server_raises_gone(self, monkeypatch):
        monkeypatch.setattr(GES, "_POLL_SECONDS", 0.05)
        net = _net(seed=1)
        holder = _Server(net, wait_timeout=30.0)
        srv = holder.__enter__()
        try:
            ev = RemoteEvaluator(0, srv.client_handles())
            srv.proc.kill()
            srv.proc.join(timeout=10)       # reap, so the pid is really gone
            t0 = time.monotonic()
            with pytest.raises(EvalServerGone):
                ev.forward(np.zeros((4, net.input_size), dtype=np.float32))
            assert time.monotonic() - t0 < 10.0, "raised, but far too late"
        finally:
            holder.__exit__()

    def test_unanswered_request_times_out(self, monkeypatch):
        monkeypatch.setattr(GES, "_POLL_SECONDS", 0.05)
        net = _net(seed=1)
        holder = _Server(net, wait_timeout=0.4, start=False)  # never started
        srv = holder.srv
        try:
            ev = RemoteEvaluator(0, srv.client_handles())
            with pytest.raises(EvalServerTimeout):
                ev.forward(np.zeros((4, net.input_size), dtype=np.float32))
        finally:
            srv.stop()


# ── 8. flag enforcement ─────────────────────────────────────────────


def _tiny_train(workers=2, episodes=2, per_round=2, match_len=1, **kw):
    net = _net(seed=1)
    tr = Trainer(_agent(net), lr=1e-3)
    return tr.train(
        MatchplayMode(match_len), num_episodes=episodes,
        episodes_per_round=per_round, batch_size=8, workers=workers,
        oneply=True, seed=1, log_every=1000, **kw,
    )


class TestMutualExclusionIsEnforced:
    """Enforced, not documented. The reference silently downgraded
    `--pipeline-collect`, which is the wrong failure: a mixed-weight
    round produces a normal-looking loss curve."""

    def test_pipeline_collect_is_rejected(self):
        with pytest.raises(ValueError, match="incompatible with "
                                             "--pipeline-collect"):
            _tiny_train(gpu_eval_server=True, pipeline_collect=True,
                        gpu_eval_device="cpu")

    def test_vectorized_collect_is_rejected(self):
        with pytest.raises(ValueError, match="incompatible with "
                                             "--vectorized-collect"):
            _tiny_train(gpu_eval_server=True, vectorized_collect=True,
                        gpu_eval_device="cpu")

    def test_maxtasksperchild_is_rejected(self, monkeypatch):
        monkeypatch.setenv("TRAINER_MAXTASKSPERCHILD", "5")
        with pytest.raises(ValueError,
                           match="TRAINER_MAXTASKSPERCHILD"):
            _tiny_train(gpu_eval_server=True, gpu_eval_device="cpu")

    def test_rejection_happens_before_any_process_is_spawned(self,
                                                            monkeypatch):
        """The raise must precede pool/server construction, or a rejected
        run leaks a pool and a CUDA context."""
        started = []
        monkeypatch.setattr(EvalServer, "start",
                            lambda self: started.append(1))
        with pytest.raises(ValueError):
            _tiny_train(gpu_eval_server=True, pipeline_collect=True,
                        gpu_eval_device="cpu")
        assert not started

    def test_the_three_flags_exist_on_the_cli(self):
        """train_batch builds its parser inside main(), so ask the CLI
        itself rather than reaching for a factory that does not exist."""
        import subprocess
        import sys
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(
            [sys.executable, os.path.join(repo, "train_batch.py"), "--help"],
            capture_output=True, text=True, timeout=300, cwd=repo,
        ).stdout
        for flag in ("--gpu-eval-server", "--gpu-eval-device",
                     "--gpu-eval-route-moves"):
            assert flag in out, f"{flag} missing from --help"


class TestSoftPreconditionsFallBack:
    """Unmet soft preconditions must NAME themselves and fall back, not
    crash and not silently pretend to work."""

    def test_cuda_unavailable_falls_back(self, monkeypatch, capsys):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        losses = _tiny_train(workers=2, gpu_eval_server=True,
                             gpu_eval_device="cuda")
        out = capsys.readouterr().out
        assert "INACTIVE" in out and "torch.cuda.is_available" in out
        assert losses, "the run must complete on the standard collector"

    def test_bad_device_string_is_named(self, monkeypatch, capsys):
        losses = _tiny_train(workers=2, gpu_eval_server=True,
                             gpu_eval_device="tpu")
        out = capsys.readouterr().out
        assert "INACTIVE" in out and "gpu-eval-device" in out
        assert losses

    def test_single_worker_falls_back(self, capsys):
        losses = _tiny_train(workers=1, gpu_eval_server=True,
                             gpu_eval_device="cpu")
        assert "--workers must be > 1" in capsys.readouterr().out
        assert losses

    def test_not_oneply_falls_back(self, capsys):
        net = _net(seed=1)
        tr = Trainer(_agent(net), lr=1e-3)
        tr.train(MatchplayMode(1), num_episodes=2, episodes_per_round=2,
                 batch_size=8, workers=2, oneply=False, seed=1,
                 log_every=1000, gpu_eval_server=True,
                 gpu_eval_device="cpu")
        assert "--oneply is required" in capsys.readouterr().out

    def test_device_reason_accepts_cpu_and_rejects_junk(self):
        assert GES.device_reason("cpu") is None
        assert "not a valid device" in GES.device_reason("nonsense:")
        assert "only 'cpu' and 'cuda" in GES.device_reason("mps")

    def test_bad_ordinal_is_caught(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        assert "invalid device ordinal" in GES.device_reason("cuda:7")
        assert GES.device_reason("cuda:0") is None


# ── 9. end-to-end through the trainer ───────────────────────────────


class TestTrainerIntegration:
    def test_run_completes_and_produces_data(self, capsys):
        losses = _tiny_train(workers=2, episodes=4, per_round=2,
                             match_len=2, gpu_eval_server=True,
                             gpu_eval_device="cpu")
        out = capsys.readouterr().out
        assert "GPU inference server ON" in out
        assert "2 worker slots" in out
        assert losses and all(np.isfinite(l) for l in losses)

    def test_route_moves_runs_too(self):
        losses = _tiny_train(workers=2, episodes=4, per_round=2,
                             match_len=2, gpu_eval_server=True,
                             gpu_eval_device="cpu",
                             gpu_eval_route_moves=True)
        assert losses

    def test_pool_does_not_recycle_workers(self, monkeypatch):
        """maxtasksperchild must be None when the server is active --
        the invariant that keeps each worker's slot claim valid.
        (Trainer.train builds its pool as pool_guard.GuardedPool.)"""
        seen = {}
        real_pool = T.GuardedPool

        def _spy(*a, **kw):
            seen.update(kw)
            return real_pool(*a, **kw)

        monkeypatch.setattr(T, "GuardedPool", _spy)
        _tiny_train(workers=2, episodes=2, per_round=2,
                    gpu_eval_server=True, gpu_eval_device="cpu")
        assert seen.get("maxtasksperchild", "missing") is None
        assert seen.get("initializer") is T._worker_init


# ── 10. CUDA (skipped without a GPU) ────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
class TestCudaServer:
    """A CUDA server differs from the CPU serial path in fp32 reduction
    order and kernel selection, so this is the ONE place a tolerance is
    used -- on values only, with structure still at zero tolerance."""

    def test_values_track_the_cpu_forward(self, single_threaded):
        net = _net(seed=1)
        feats = np.random.default_rng(9).standard_normal(
            (300, net.input_size), dtype=np.float32)
        want = _local_values(net, feats)
        with _Server(net, device="cuda") as srv:
            got = RemoteEvaluator(0, srv.client_handles()).forward(feats)
        assert got.shape == want.shape and got.dtype == want.dtype
        assert np.allclose(got, want, atol=1e-5), np.abs(got - want).max()

    def test_row_order_is_exact_even_though_values_are_not(self,
                                                           single_threaded):
        """A permuted scatter is caught structurally, not numerically:
        the permutation identity holds bitwise on the GPU too, because
        both calls run the same kernel on the same batch size."""
        net = _net(seed=1)
        rng = np.random.default_rng(10)
        feats = rng.standard_normal((128, net.input_size), dtype=np.float32)
        perm = rng.permutation(128)
        with _Server(net, device="cuda") as srv:
            ev = RemoteEvaluator(0, srv.client_handles())
            straight = ev.forward(feats)
            shuffled = ev.forward(np.ascontiguousarray(feats[perm]))
        assert np.array_equal(shuffled, straight[perm])

    def test_collected_episode_matches_within_tolerance(self,
                                                        single_threaded):
        net = _net(seed=3)
        mode = MatchplayMode(2)
        ref = _serial_episode(_agent(net), mode, 5)
        with _Server(net, device="cuda") as srv:
            agent = _agent(net)
            agent._remote_eval = RemoteEvaluator(0, srv.client_handles())
            got = _serial_episode(agent, mode, 5)
        # Shape at ZERO tolerance: a value perturbation that flipped a
        # move choice would change the trajectory and show up here.
        assert got[0].shape == ref[0].shape
        assert np.allclose(got[0], ref[0], atol=1e-5)
        assert np.allclose(got[1], ref[1], atol=1e-5)


# ── 11. determinism / coalescing ────────────────────────────────────
#
# Everything below this line was written in response to an adversarial
# review of the original port. Each class names the defect it pins.


def _saturating_client(handles, slot, n_rows, n_reqs, out_q):
    """Hammer one slot as fast as it will go, so several slots really do
    have requests pending at the same instant. Reports its own values so
    the caller can check them against a local forward."""
    ev = RemoteEvaluator(slot, handles)
    rng = np.random.default_rng(1000 + slot)
    feats = rng.standard_normal((n_rows, handles.n_feat), dtype=np.float32)
    vals = None
    for _ in range(n_reqs):
        vals = ev.forward(feats)
    out_q.put((slot, feats[:2].tolist(), vals.tolist()))


def _drive_saturated(srv, n_slots, n_rows, n_reqs):
    """Run `n_slots` saturating clients as threads (one shared process,
    so this costs one core on a busy box, not `n_slots`)."""
    import queue as _q
    out_q = _q.Queue()
    handles = srv.client_handles()
    threads = [
        threading.Thread(target=_saturating_client,
                         args=(handles, s, n_rows, n_reqs, out_q))
        for s in range(n_slots)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
        assert not t.is_alive(), "a saturating client never finished"
    return out_q


class TestCoalescingAndDeterminism:
    """The original port's headline bit-exactness proof covered
    num_slots=1, which `trainer.train` never runs in: it sets
    num_slots=workers and falls back to the plain collector at
    workers<=1, so the flag is only ever ACTIVE at num_slots>=2.

    At num_slots>=2 the server fuses whatever is concurrently pending
    into one GEMM. Fusion is decided by wall-clock arrival order, fp32
    reductions are batch-composition sensitive, and a last-ulp change
    can flip a min() tie over candidate moves -- so identically-seeded
    routed runs did not reproduce each other, let alone the unrouted
    collector, and nothing said so.

    The fix is a knob plus honesty: `coalesce=False`
    (`--gpu-eval-deterministic`) makes multi-slot routing bit-exact
    again, and the trainer states which mode a run is in.
    """

    def test_coalesce_false_never_fuses(self):
        """One request per GPU call, exactly -- under a driver that
        saturates every slot, which is when fusion is most likely."""
        net = _net(seed=1)
        with _Server(net, num_slots=4, max_rows=1024, coalesce=False) as srv:
            _drive_saturated(srv, n_slots=4, n_rows=64, n_reqs=25)
            st = srv.stats()
        assert st["requests"] == 100
        assert st["gpu_calls"] == 100, st
        assert st["requests_per_call"] == 1.0

    def test_saturated_coalescing_does_fuse(self):
        """The mirror image, so `coalesce=False` is shown to be doing
        something rather than describing the only behaviour there is.

        Asserted as a strict inequality on TOTALS, not on any single
        call: with 4 slots hammering, at least one pair must land in one
        GPU call somewhere in 100 requests."""
        net = _net(seed=1)
        with _Server(net, num_slots=4, max_rows=1024) as srv:
            _drive_saturated(srv, n_slots=4, n_rows=64, n_reqs=25)
            st = srv.stats()
        assert st["requests"] == 100
        assert st["gpu_calls"] < 100, (
            "no fusion at all under a saturating 4-slot driver; the "
            "coalescing path may be dead", st)

    def test_trainer_states_which_reproducibility_mode_it_is_in(self,
                                                                capsys):
        _tiny_train(workers=2, episodes=2, per_round=2, match_len=2,
                    gpu_eval_server=True, gpu_eval_device="cpu")
        out = capsys.readouterr().out
        assert "NOT bitwise reproducible" in out, (
            "a coalescing run must say so; discovering it from a diff "
            "of two seeded runs is not acceptable")

        _tiny_train(workers=2, episodes=2, per_round=2, match_len=2,
                    gpu_eval_server=True, gpu_eval_device="cpu",
                    gpu_eval_deterministic=True)
        out = capsys.readouterr().out
        assert "DETERMINISTIC mode" in out


def _train_fingerprint(seed=99, workers=2, episodes=8, per_round=4,
                       match_len=2, **kw):
    """md5 of the final weights after a real (tiny) `Trainer.train`."""
    import hashlib
    net = _net(seed=1)
    tr = Trainer(_agent(net), lr=1e-3)
    tr.train(
        MatchplayMode(match_len), num_episodes=episodes,
        episodes_per_round=per_round, batch_size=8, workers=workers,
        oneply=True, seed=seed, log_every=1000, **kw,
    )
    h = hashlib.md5()
    for k, v in net.state_dict().items():
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


class TestDeterministicRoutingReproducesTheUnroutedRun:
    """End-to-end version of the property above, at the multi-slot
    worker count the flag actually runs in."""

    def test_multi_slot_deterministic_run_is_bit_equal_to_no_server(self):
        plain = _train_fingerprint()
        det_a = _train_fingerprint(gpu_eval_server=True,
                                   gpu_eval_device="cpu",
                                   gpu_eval_deterministic=True)
        det_b = _train_fingerprint(gpu_eval_server=True,
                                   gpu_eval_device="cpu",
                                   gpu_eval_deterministic=True)
        assert det_a == det_b, "deterministic mode did not reproduce itself"
        assert det_a == plain, (
            "deterministic routing changed the trained weights; routing "
            "must not perturb the collector at all")


# ── 12. bounded weight push ─────────────────────────────────────────


def _call_bounded(fn, timeout):
    """Run `fn` on a daemon thread and report (finished?, exception).

    A test for "this used to hang forever" must not itself be able to
    hang the suite, so the call is never joined without a bound.
    """
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:      # noqa: BLE001 - reporting, not handling
            box["exc"] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    return (not th.is_alive()), box


class TestWeightPushIsBounded:
    """`update_weights` waited on a bare `_ack_q.get()`. It runs on the
    MASTER at the top of every round, so a server that died (a CUDA OOM
    in `net.to(device)` is the realistic cause) hung the whole run with
    no traceback -- the exact failure the module docstring claims to
    have eliminated, while `stop()` two methods below already bounded
    the identical queue."""

    def test_dead_server_raises_instead_of_hanging(self):
        net = _net(seed=1)
        holder = _Server(net, num_slots=1, max_rows=64)
        srv = holder.__enter__()
        srv.weight_timeout = 20.0
        try:
            os.kill(srv.proc.pid, signal.SIGKILL)
            srv.proc.join(timeout=10)
            finished, box = _call_bounded(
                lambda: srv.update_weights(net.state_dict()), 30.0)
            assert finished, (
                "update_weights is still blocked after 30s on a dead "
                "server -- unbounded _ack_q.get()")
            assert isinstance(box.get("exc"), EvalServerError), box
        finally:
            holder.__exit__()

    def test_timeout_is_bounded_even_when_the_server_is_alive(self):
        """A wedged (not dead) server must also raise. Simulated with a
        never-started server, whose queue nobody reads."""
        net = _net(seed=1)
        holder = _Server(net, num_slots=1, max_rows=64, start=False)
        srv = holder.__enter__()
        try:
            finished, box = _call_bounded(
                lambda: srv.update_weights(net.state_dict(), timeout=3.0),
                25.0)
            assert finished, "update_weights hung on an unresponsive server"
            assert isinstance(box.get("exc"), EvalServerTimeout), box
        finally:
            holder.__exit__()


# ── 13. zombie liveness ─────────────────────────────────────────────


def _exit_now():
    os._exit(0)


class TestZombieServerReadsAsDead:
    """`os.kill(pid, 0)` succeeds for an unreaped child, and in the
    production topology the master is blocked in `pool.map` and never
    reaps the server -- so the fast `EvalServerGone` path was
    unreachable and every worker burned the FULL GPU_EVAL_TIMEOUT
    (default 300 s) before raising a timeout instead."""

    def test_pid_alive_reports_a_zombie_as_dead(self):
        ctx = mp.get_context("spawn")
        p = ctx.Process(target=_exit_now)
        p.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with open(f"/proc/{p.pid}/stat", "rb") as f:
                    data = f.read()
                if data[data.rindex(b")") + 2:data.rindex(b")") + 3] == b"Z":
                    break
            except OSError:
                break
            time.sleep(0.05)
        try:
            os.kill(p.pid, 0)          # the naive probe still succeeds
            assert GES._pid_alive(p.pid) is False, (
                "an unreaped dead child reads as alive; EvalServerGone "
                "is then unreachable while the master sits in pool.map")
        finally:
            p.join(timeout=10)

    def test_forward_raises_gone_fast_on_an_unreaped_dead_server(self):
        net = _net(seed=1)
        holder = _Server(net, num_slots=1, max_rows=64, wait_timeout=120.0)
        srv = holder.__enter__()
        try:
            ev = RemoteEvaluator(0, srv.client_handles())
            feats = np.zeros((4, net.input_size), dtype=np.float32)
            os.kill(srv.proc.pid, signal.SIGKILL)
            # Deliberately do NOT join: that is what the trainer does,
            # and joining is what made the old test pass.
            t0 = time.monotonic()
            with pytest.raises(EvalServerGone):
                ev.forward(feats)
            assert time.monotonic() - t0 < 30, (
                "took most of the wait budget: the liveness probe did "
                "not notice the zombie")
        finally:
            holder.__exit__()


# ── 14. a dead pool worker is loud, not a silent hang ───────────────


class _NeverReadyResult:
    def get(self, timeout=None):
        raise mp.TimeoutError()


class _FakePool:
    def __init__(self):
        self.terminated = False

    def map_async(self, fn, args):
        return _NeverReadyResult()

    def terminate(self):
        self.terminated = True


class TestPoolWorkerDeathIsLoud:
    """`multiprocessing.Pool` swallows a dead worker: it respawns
    replacements forever, each of which dies in the initializer with no
    free slot, in a process whose exception nobody reads. The master
    stayed in `pool.map` indefinitely -- and since `pool.map` never
    returned, the `finally` never ran and /dev/shm was never unlinked.

    Measured before the fix: 200 s with no result and 6 distinct worker
    pids for a pool of 2."""

    def test_worker_init_publishes_the_failure(self, monkeypatch):
        net = _net(seed=1)
        monkeypatch.setattr(T, "_WORKER_REMOTE_EVAL", None, raising=False)
        with _Server(net, num_slots=1, max_rows=8) as srv:
            h = srv.client_handles()
            T._worker_init(h)                       # claims the only slot
            assert not srv.worker_failed()
            with pytest.raises(RuntimeError, match="no free shared-memory"):
                _worker_init_fast(h)
            assert srv.worker_failed(), (
                "the initializer failed in a throwaway replacement "
                "process and told nobody")
        T._WORKER_REMOTE_EVAL = None

    def test_map_watched_fails_the_round_on_a_worker_death(self):
        net = _net(seed=1)
        holder = _Server(net, num_slots=1, max_rows=8, start=False)
        srv = holder.__enter__()
        try:
            srv.fail_event.set()
            pool = _FakePool()
            with pytest.raises(GES.EvalWorkerGone):
                srv.map_watched(pool, len, [[1]], poll=0.2)
            assert pool.terminated
        finally:
            holder.__exit__()

    def test_map_watched_fails_the_round_on_a_server_death(self):
        net = _net(seed=1)
        holder = _Server(net, num_slots=1, max_rows=8)
        srv = holder.__enter__()
        try:
            os.kill(srv.proc.pid, signal.SIGKILL)
            pool = _FakePool()
            with pytest.raises(EvalServerGone):
                srv.map_watched(pool, len, [[1]], poll=0.2)
            assert pool.terminated
        finally:
            holder.__exit__()

    def test_the_round_map_goes_through_the_watchdog(self, monkeypatch):
        """Wiring check: the trainer must USE map_watched when the
        server is on, or none of the above helps a real run."""
        calls = []
        real = GES.EvalServer.map_watched

        def spy(self, pool, fn, args, **kw):
            calls.append(len(args))
            return real(self, pool, fn, args, **kw)

        monkeypatch.setattr(GES.EvalServer, "map_watched", spy)
        _tiny_train(workers=2, episodes=2, per_round=2, match_len=2,
                    gpu_eval_server=True, gpu_eval_device="cpu")
        assert calls, "the round still ran through a bare pool.map"


# ── 15. request ids / stale answers ─────────────────────────────────


class TestStaleAnswersAreRefused:
    """The wire carried no request id, so a slot's Event could not
    distinguish "your answer" from "the previous, abandoned answer".
    After an EvalServerTimeout the next forward() on that slot returned
    the PREVIOUS request's values: right dtype, plausible range, wrong
    rows, no error."""

    def _handles_without_a_server(self, net, wait_timeout=2.0):
        holder = _Server(net, num_slots=1, max_rows=64,
                         wait_timeout=wait_timeout, start=False)
        return holder, holder.__enter__()

    def test_next_forward_after_a_timeout_does_not_return_the_old_answer(self):
        net = _net(seed=1)
        holder, srv = self._handles_without_a_server(net)
        try:
            h = srv.client_handles()
            ev = RemoteEvaluator(0, h)
            a = np.full((8, net.input_size), 0.25, dtype=np.float32)
            with pytest.raises(EvalServerTimeout):
                ev.forward(a)
            # The abandoned answer arrives late: buffer written, Event set.
            sentinel = 0.123456
            ev.out_arr[:8] = sentinel
            h.events[0].set()
            b = np.full((8, net.input_size), 0.75, dtype=np.float32)
            with pytest.raises(EvalServerError):
                got = ev.forward(b)
                assert not np.allclose(got, sentinel), (
                    "returned the abandoned request's values as if they "
                    "were this request's")
        finally:
            holder.__exit__()

    def test_a_mismatched_echo_is_detected(self):
        """Same hazard on a fresh evaluator (so the poison flag cannot
        be what saves us): an abandoned request's answer lands WHILE
        this one is waiting, which is the real interleaving -- the late
        writer sets the Event after `forward` has cleared it."""
        net = _net(seed=1)
        holder, srv = self._handles_without_a_server(net, wait_timeout=20.0)
        try:
            h = srv.client_handles()
            ev = RemoteEvaluator(0, h)

            def late_answer():
                time.sleep(0.5)
                ev.out_arr[:8] = 0.5       # some other request's rows
                ev.ids_arr[0] = 999        # ... and its id
                h.events[0].set()

            th = threading.Thread(target=late_answer, daemon=True)
            th.start()
            with pytest.raises(GES.EvalServerDesync):
                ev.forward(np.zeros((8, net.input_size), dtype=np.float32))
            th.join(timeout=10)
        finally:
            holder.__exit__()

    def test_ids_do_not_disturb_the_happy_path(self):
        net = _net(seed=1)
        feats = np.random.default_rng(4).standard_normal(
            (64, net.input_size), dtype=np.float32)
        want = _local_values(net, feats)
        with _Server(net, num_slots=1, max_rows=256) as srv:
            ev = RemoteEvaluator(0, srv.client_handles())
            for _ in range(3):
                assert np.array_equal(ev.forward(feats), want)
            assert int(ev.ids_arr[0]) == 3


# ── 16. fused-batch row cap (VRAM) ──────────────────────────────────


class TestFusedBatchIsCapped:
    """Peak VRAM is O(num_slots x rows/request) and was bounded
    nowhere: 48 slots x max_rows=8192 fused into a 393216-row forward
    and CUDA-OOMed the server outright."""

    def test_cap_splits_the_batch(self):
        net = _net(seed=1)
        # 64-row requests, cap 100: two can never share a call.
        with _Server(net, num_slots=4, max_rows=1024,
                     max_batch_rows=100) as srv:
            _drive_saturated(srv, n_slots=4, n_rows=64, n_reqs=20)
            st = srv.stats()
        assert st["requests"] == 80
        assert st["gpu_calls"] == 80, ("the row cap did not bound the "
                                       "fused batch", st)

    def test_capped_requests_still_get_their_own_correct_values(self):
        net = _column_picker_net(16)
        with _Server(net, num_slots=3, max_rows=512,
                     max_batch_rows=70) as srv:
            out_q = _drive_saturated(srv, n_slots=3, n_rows=64, n_reqs=8)
        got_slots = set()
        while not out_q.empty():
            slot, _head, vals = out_q.get()
            got_slots.add(slot)
            want = _local_values(
                net,
                np.random.default_rng(1000 + slot).standard_normal(
                    (64, 16), dtype=np.float32),
            )
            assert np.array_equal(np.array(vals, dtype=np.float32), want)
        assert got_slots == {0, 1, 2}

    def test_default_cap_is_documented_and_finite(self):
        assert 0 < GES.MAX_BATCH_ROWS_DEFAULT < 21 * 1024 * 48
        assert "GPU_EVAL_MAX_BATCH_ROWS" in GES.__doc__


# ── 17. the worker's own net is the master's net ────────────────────


class TestWorkerNetMatchesTheMaster:
    """`build_spec_from` deliberately carries `activation` to the
    server, but the worker rebuilt its LOCAL net from an args tuple
    that had no activation field, so on a tanh checkpoint one routed
    episode mixed targets from two different functions (routed: tanh;
    evaluate_matchplay / _eval_new_game / cube values: relu).

    Measured divergence between the two nets on a tanh checkpoint:
    mean 0.0202, max 0.0565 in MWC units."""

    def _captured_worker_args(self, net, monkeypatch):
        """The args the master actually builds for `_collect_worker`.
        Trainer.train builds its pool as pool_guard.GuardedPool and waits
        on it through pool_guard.PoolWatch (worker list, ready / wait /
        get), so the fake stands in for both."""
        captured = []

        class _FakePoolCapture:
            _pool = []          # PoolWatch's worker list: none to watch

            def map_async(self, fn, args):
                captured.extend(args)

                class _R:
                    def ready(self_inner):
                        return True

                    def wait(self_inner, timeout=None):
                        return None

                    def get(self_inner, timeout=None):
                        return [(np.zeros((0, 1), dtype=np.float32),
                                 np.zeros((0, 1), dtype=np.float32))]
                return _R()

            def map(self, fn, args):
                captured.extend(args)
                return [(np.zeros((0, 1), dtype=np.float32),
                         np.zeros((0, 1), dtype=np.float32))]

            def close(self):
                pass

            def join(self):
                pass

        monkeypatch.setattr(T, "GuardedPool",
                            lambda *a, **kw: _FakePoolCapture())
        tr = Trainer(_agent(net), lr=1e-3)
        tr.train(MatchplayMode(2), num_episodes=2, episodes_per_round=2,
                 batch_size=8, workers=2, oneply=True, seed=1,
                 log_every=1000)
        assert captured
        return captured[0]

    def test_master_ships_the_activation(self, monkeypatch):
        net = _net(seed=2, activation="tanh")
        args = self._captured_worker_args(net, monkeypatch)
        assert "tanh" in args, (
            "the worker args carry no activation, so the worker's local "
            "net silently defaults to relu")

    def test_worker_rebuilds_the_same_function(self, monkeypatch):
        """End to end: feed the master's own args tuple to
        `_collect_worker` and watch which net it builds."""
        net = _net(seed=2, activation="tanh")
        args = self._captured_worker_args(net, monkeypatch)
        import model
        real = model.TDNetwork
        seen = []

        def spy(*a, **kw):
            seen.append(kw.get("activation", "relu"))
            return real(*a, **kw)

        monkeypatch.setattr(model, "TDNetwork", spy)
        T._collect_worker(args)
        assert seen and all(s == "tanh" for s in seen), seen

    def test_server_and_worker_agree(self):
        """The two nets a routed worker uses must be one function."""
        net = _net(seed=2, activation="tanh")
        assert build_spec_from(net)["activation"] == "tanh"


# ── 18. bf16 really is a no-op on matchplay ─────────────────────────


class TestBf16DoesNotTouchMatchplay:
    """The NOTE printed for `--bf16-collect --gpu-eval-server` claimed
    "only the local overflow fallback would use it". No matchplay path
    consults the bf16 net, fallback included."""

    def test_the_note_does_not_claim_a_bf16_fallback(self):
        note = T.BF16_SERVER_NOTE
        assert "NO effect" in note
        assert "would use it" not in note

    def test_a_poisoned_bf16_net_cannot_change_a_matchplay_value(self):
        """Behavioural, not a grep: zero out the bf16 copy and check
        every matchplay value path is unmoved."""
        net = _net(seed=5)
        agent = TDAgent(net)
        # Force the bf16 copy into existence even on a CPU whose ISA
        # would make TDAgent decline it: the question here is which net
        # the matchplay code READS, not whether bf16 is fast.
        agent.bf16_inference = True
        agent._build_bf16_net()
        assert agent._bf16_net is not None
        feats = np.random.default_rng(6).standard_normal(
            (32, net.input_size), dtype=np.float32)
        before = agent._matchplay_values_np(feats)
        with torch.no_grad():
            for p in agent._bf16_net.parameters():
                p.zero_()
        after = agent._matchplay_values_np(feats)
        assert np.array_equal(before, after)
        assert np.array_equal(before, _local_values(net, feats))


# ── 19. the docs must match the measurements ────────────────────────


class TestDocsMatchMeasurements:
    """Quantitative claims in the module docstring were reproduced as
    FALSE by an independent re-measurement. Docs are the interface to a
    benchmark nobody will rerun, so they get tests too."""

    def test_vram_is_not_claimed_flat(self):
        """Measured on the real server, saturated: 328 MB at 1 slot ->
        1262 MB at 48. The growing term is the fused batch's
        activations, not the CUDA context."""
        low = GES.__doc__.lower()
        assert "flat in w" not in low
        assert "~350 mb flat" not in low
        assert "1262" in GES.__doc__, (
            "the measured slot-count sweep is not recorded anywhere")

    def test_shm_per_slot_arithmetic(self):
        """8192 rows x 207 features x 4 bytes = 6.5 MB per slot, not 7.2
        (7.2 is the 220-feature number; production here is 207-feature)."""
        assert "7.2 MB per worker" not in GES.__doc__
        assert "6.5 MB for the 207-feature" in GES.__doc__

    def test_noise_floor_is_disclosed(self):
        """Four measurements of an IDENTICAL CPU-serial workload inside
        15 minutes spanned 1.71x (3.70 - 6.33 ep/s), so two-rep min-max
        ranges must not be read as confidence intervals."""
        low = GES.__doc__.lower()
        assert "noise floor" in low
        assert "1.71" in low

    def test_move_routing_gain_is_not_quoted_as_a_point_estimate(self):
        """Direction reproduced (5/5 positive); magnitude spanned +14%
        to +121%."""
        assert "+26%)" not in GES.__doc__

    def test_determinism_is_documented_where_someone_will_read_it(self):
        assert "reproducib" in GES.__doc__.lower()
        assert "--gpu-eval-deterministic" in GES.__doc__
