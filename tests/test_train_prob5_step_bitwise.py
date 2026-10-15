"""`BG_TRAIN_CUDA_GRAPH` in train_prob5.py must leave training bitwise
unchanged: same weights (torch.equal,
tensor by tensor) in the final network and in every periodic checkpoint,
the same per-batch losses and per-output loss averages, and the same log
lines as the stock eager step, for the same seed and inputs.

Stock reference: the eager step (`BG_TRAIN_CUDA_GRAPH=0`) with a
`.item()` per batch (the "stock" arm patches it into
train_prob5.train). Arms: the eager step reading its losses back once
per round (`BG_TRAIN_CUDA_GRAPH=0`), and `BG_TRAIN_CUDA_GRAPH` modes adam
and fwdbwd, 1 (auto) and unset, the default
(cuda_graph_step.py; the net leaves the device after every train phase,
GraphedTrainStep.bind brings it back into the graph's storage).

Whole runs go through `train_prob5.train` with the prod prob5 net, 2
collection workers, 3 rounds of 400 episodes, cuda: the C collector and
the Python collector, pipelined and synchronous, an LR anneal with a
warm-up, and the loss variants (MSE with a P(win) weight, BCEWithLogits
on a raw-logits net). train_prob5 has no epochs-per-round option (one
pass over the pool per round), so the 2-epoch case of
tests/test_train_step_bitwise.py has no twin here.
CUDA only; the simd backend is built as in tests/test_c_api.py.
"""

import contextlib
import io
import re
import shutil

import numpy as np
import pytest
import torch

import cuda_graph_step
import train_prob5 as TP
from cuda_graph_step import GraphedTrainStep, StepSpec
from model import ProbNetwork
from test_c_api import _build, PROD_PROB5

pytestmark = pytest.mark.cuda

ENV_KEYS = ("BG_TRAIN_CUDA_GRAPH", "BG_TRAIN_GRAPH_ADAM_CHUNK")
ARMS = {
    "stock": {"BG_TRAIN_CUDA_GRAPH": "0"},
    "eager": {"BG_TRAIN_CUDA_GRAPH": "0"},
    "graph_adam": {"BG_TRAIN_CUDA_GRAPH": "adam"},
    "graph_fwdbwd": {"BG_TRAIN_CUDA_GRAPH": "fwdbwd"},
    "graph_auto": {"BG_TRAIN_CUDA_GRAPH": "1"},
    "graph_default": {},            # unset: on (auto)
}
CASES = {
    "c_pipelined": dict(c_collect=True, pipeline_collect=True),
    "python_sync": dict(c_collect=False),
    "c_sync_anneal_warmup": dict(c_collect=True, end_lr=1e-6, warmup_cycles=2),
    "python_pipelined_mse_pw": dict(c_collect=False, pipeline_collect=True,
                                    use_mse=True, pw_weight=2.0),
    "c_pipelined_raw_logits": dict(c_collect=True, pipeline_collect=True,
                                   raw_logits=True),
}
GRAPH_ARMS = {
    "c_pipelined": ["eager", "graph_adam", "graph_fwdbwd", "graph_default"],
    "python_sync": ["eager", "graph_adam", "graph_fwdbwd"],
    "c_sync_anneal_warmup": ["eager", "graph_auto", "graph_fwdbwd"],
    "python_pipelined_mse_pw": ["graph_adam"],
    "c_pipelined_raw_logits": ["graph_adam"],
}
EPR, ROUNDS = 400, 3


@pytest.fixture(scope="module")
def simd():
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    ok, log = _build("simd")
    if not ok:
        pytest.skip(f"simd backend not built:\n{log[-500:]}")
    return "simd"


def _net(raw_logits=False):
    net = ProbNetwork.load(PROD_PROB5)
    if raw_logits:
        # The prod weights under BCEWithLogits (no sigmoid in forward).
        raw = ProbNetwork(hidden_sizes=net.hidden_sizes, raw_logits=True)
        raw.load_state_dict(net.state_dict())
        net = raw
    return net


_step = TP._train_step


def _item_per_batch(*a):
    # The stock loop: a .item() host sync after every batch.
    loss, per_out = _step(*a)
    return torch.tensor(loss.item(), dtype=torch.float32), per_out


def _log_lines(out):
    """The round and per-output lines without the timing fields."""
    keep = []
    for ln in out.splitlines():
        if ln.startswith("Round ") or "per-output loss:" in ln:
            keep.append(re.sub(r"\| collect .*$", "", ln))
    return keep


def _train(monkeypatch, tmp_path, arm, case, backend, seed=3):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    for k, v in ARMS[arm].items():
        monkeypatch.setenv(k, v)
    if arm == "stock":
        monkeypatch.setattr(TP, "_train_step", _item_per_batch)
    replays = []
    real_replay = GraphedTrainStep._replay

    def counting_replay(self, *a):
        replays.append(1)
        return real_replay(self, *a)

    monkeypatch.setattr(GraphedTrainStep, "_replay", counting_replay)
    kw = dict(case)
    raw = kw.pop("raw_logits", False)
    torch.manual_seed(seed)
    net = _net(raw)
    init = {k: v.detach().clone() for k, v in net.state_dict().items()}
    save = tmp_path / arm / "m.pt"
    save.parent.mkdir(parents=True)
    metrics = {}
    args = dict(num_episodes=EPR * ROUNDS, lr=1e-4, end_lr=None,
                episodes_per_round=EPR, batch_size=256, workers=2,
                device="cuda", oneply=False, pw_weight=1.0, use_mse=False,
                seed=seed, log_every=1, save_path=str(save), save_every=EPR,
                warmup_cycles=0, c_collect_backend=backend,
                metrics_out=metrics)
    args.update(kw)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        TP.train(net, **args)
    out = buf.getvalue()
    ckpts = {}
    for n in range(1, ROUNDS + 1):
        ckpts[n] = torch.load(save.parent / f"m_ep{n * EPR}.pt", weights_only=True)
    final = torch.load(save, weights_only=True)
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return dict(sd=sd, init=init, ckpts=ckpts, final=final, out=out,
                metrics=metrics, replays=len(replays))


def _assert_same_ckpt(a, b, what):
    assert a.keys() == b.keys(), what
    for k in a:
        if k == "state_dict":
            assert a[k].keys() == b[k].keys(), what
            for t in a[k]:
                assert a[k][t].device.type == "cpu", (what, t)
                assert torch.equal(a[k][t], b[k][t]), (what, t)
        else:
            assert a[k] == b[k], (what, k)


@pytest.fixture(scope="module")
def stock_runs(simd, tmp_path_factory):
    """Stock run per case, computed once."""
    mp = pytest.MonkeyPatch()
    try:
        return {name: _train(mp, tmp_path_factory.mktemp(name), "stock",
                             case, simd)
                for name, case in CASES.items()}
    finally:
        mp.undo()


@pytest.mark.parametrize("case_name,arm", [
    (c, a) for c in CASES for a in GRAPH_ARMS[c]])
@pytest.mark.slow
def test_run_bitwise_equal_to_stock(monkeypatch, tmp_path, simd, stock_runs,
                                    case_name, arm):
    ref = stock_runs[case_name]
    got = _train(monkeypatch, tmp_path, arm, CASES[case_name], simd)
    out = got["out"]
    n_batches = len(ref["metrics"]["batch_losses"])
    if arm.startswith("graph"):
        # The graph must actually have run (no silent eager fallback).
        assert "first replay bitwise equal to the eager step" in out, out
        assert "WARN: BG_TRAIN_CUDA_GRAPH" not in out, out
        mode = "fwdbwd" if arm == "graph_fwdbwd" else "adam"
        assert f"(mode {mode})" in out, out
        if arm == "graph_default":
            assert "BG_TRAIN_CUDA_GRAPH (default on;" in out, out
        # Every full batch but the run's first replays the graph.
        assert got["replays"] >= n_batches - 1 - ROUNDS, (got["replays"], n_batches)
    else:
        assert got["replays"] == 0 and "BG_TRAIN_CUDA_GRAPH" not in out
    # The weights moved: the equalities below are not trivially true.
    assert any(not torch.equal(ref["sd"][k], ref["init"][k]) for k in ref["sd"])
    assert n_batches > 3 * 80
    assert got["metrics"]["batch_losses"] == ref["metrics"]["batch_losses"]
    assert torch.equal(torch.tensor(got["metrics"]["batch_losses"]),
                       torch.tensor(ref["metrics"]["batch_losses"]))
    assert len(got["metrics"]["per_output"]) == ROUNDS
    for a, b in zip(got["metrics"]["per_output"], ref["metrics"]["per_output"]):
        assert np.array_equal(a, b)
    assert _log_lines(out) == _log_lines(ref["out"]) and len(_log_lines(out)) == 2 * ROUNDS
    for n in ref["ckpts"]:
        _assert_same_ckpt(got["ckpts"][n], ref["ckpts"][n], f"checkpoint ep{n * EPR}")
    _assert_same_ckpt(got["final"], ref["final"], "final")
    assert got["sd"].keys() == ref["sd"].keys()
    for k in ref["sd"]:
        assert torch.equal(got["sd"][k], ref["sd"][k]), k


def test_graph_ignored_on_cpu_with_note(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("BG_TRAIN_CUDA_GRAPH", "1")
    TP.train(ProbNetwork(hidden_sizes=[8]), num_episodes=2, lr=1e-3,
             end_lr=None, episodes_per_round=2, batch_size=16, workers=1,
             device="cpu", oneply=False, pw_weight=1.0, use_mse=False,
             seed=1, log_every=1, save_path=None, save_every=0,
             warmup_cycles=0)
    out = capsys.readouterr().out
    assert "NOTE: BG_TRAIN_CUDA_GRAPH ignored (device is cpu); eager train step." in out
    # The default (unset) does not apply off cuda and says nothing.
    monkeypatch.delenv("BG_TRAIN_CUDA_GRAPH")
    TP.train(ProbNetwork(hidden_sizes=[8]), num_episodes=2, lr=1e-3,
             end_lr=None, episodes_per_round=2, batch_size=16, workers=1,
             device="cpu", oneply=False, pw_weight=1.0, use_mse=False,
             seed=1, log_every=1, save_path=None, save_every=0,
             warmup_cycles=0)
    assert "BG_TRAIN_CUDA_GRAPH" not in capsys.readouterr().out


# ── GraphedTrainStep on a StepSpec directly (no pool) ────────────────


def _toy(seed=0, n=1500, raw_logits=False, use_mse=False):
    torch.manual_seed(seed)
    net = ProbNetwork(hidden_sizes=[64, 32], raw_logits=raw_logits)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    w = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0]).cuda()
    loss_fn = TP._make_loss_fn(raw_logits, use_mse, w)
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((n, 196), generator=g).cuda()
    y = torch.rand((n, 5), generator=g).cuda()
    return net, opt, loss_fn, x, y


def _rounds(net, opt, loss_fn, x, y, batch, graph=None, rounds=3):
    """train_prob5's round shape: the net comes to the device, trains a
    randperm pass, goes back to the CPU."""
    losses, per_outs = [], []
    torch.manual_seed(11)
    for _ in range(rounds):
        if graph is not None:
            graph.bind()
        else:
            net.to("cuda")
        perm = torch.randperm(len(x), device="cuda")
        for s in range(0, len(x), batch):
            idx = perm[s:s + batch]
            if graph is not None:
                loss, per_out = graph.step(x, y, idx)
            else:
                loss, per_out = TP._train_step(net, opt, loss_fn, x[idx], y[idx])
            losses.append(loss)
            per_outs.append(per_out)
        if graph is not None:
            graph.unbind()
        net.to("cpu")
    return torch.stack(losses), torch.stack(per_outs)


# n=1440: every round ends on a full batch, i.e. on a replay, so in mode
# fwdbwd the graph's own grad buffers are .grad when the net leaves the
# device (unbind() is what keeps Module.to off them).
@pytest.mark.parametrize("n", [1500, 1440], ids=["ragged", "full_last"])
@pytest.mark.parametrize("mode,loss", [
    ("adam", "bce"), ("fwdbwd", "bce"), ("adam", "mse"), ("adam", "logits"),
    ("fwdbwd", "logits")])
def test_prob5_spec_steps_match_eager(mode, loss, n):
    kw = dict(raw_logits=loss == "logits", use_mse=loss == "mse", n=n)
    net_e, opt_e, lf_e, x, y = _toy(**kw)
    le, pe = _rounds(net_e, opt_e, lf_e, x, y, 96)
    net_g, opt_g, lf_g, x, y = _toy(**kw)
    spec = StepSpec(net_g, opt_g, lf_g,
                    lambda xb, yb: TP._train_step(net_g, opt_g, lf_g, xb, yb),
                    device="cuda")
    gs = GraphedTrainStep(spec, 96, 196, mode=mode, log=lambda *a, **k: None,
                          target_shape=(5,))
    lg, pg = _rounds(net_g, opt_g, lf_g, x, y, 96, gs)
    assert gs.replays >= 3 * 15 - 1 and not gs.disabled
    assert torch.equal(lg, le) and torch.equal(pg, pe)
    for a, b in zip(net_e.parameters(), net_g.parameters()):
        assert a.device.type == b.device.type == "cpu"
        assert torch.equal(a, b)
    for pa, pb in zip(opt_e.param_groups[0]["params"], opt_g.param_groups[0]["params"]):
        for k in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(opt_e.state[pa][k], opt_g.state[pb][k])


def test_bind_keeps_the_storage():
    net, opt, loss_fn, x, y = _toy()
    spec = StepSpec(net, opt, loss_fn,
                    lambda xb, yb: TP._train_step(net, opt, loss_fn, xb, yb),
                    device="cuda")
    gs = GraphedTrainStep(spec, 96, 196, mode="adam", log=lambda *a, **k: None,
                          target_shape=(5,))
    gs.bind()
    ptrs = [p.data_ptr() for p in net.parameters()]
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    vals = [p.detach().clone() for p in net.parameters()]
    gs.unbind()
    net.to("cpu")
    with torch.no_grad():
        next(net.parameters()).add_(1.0)       # a CPU-side change comes along
    vals[0] += 1.0
    gs.bind()
    assert [p.data_ptr() for p in net.parameters()] == ptrs
    assert all(p.grad is None for p in net.parameters())
    for p, v in zip(net.parameters(), vals):
        assert p.device.type == "cuda" and torch.equal(p, v)


def test_graph_leaves_the_cuda_rng_alone():
    """train_prob5 shuffles with torch.randperm on the CUDA generator:
    a capture and replays must not move it."""
    net, opt, loss_fn, x, y = _toy()
    spec = StepSpec(net, opt, loss_fn,
                    lambda xb, yb: TP._train_step(net, opt, loss_fn, xb, yb),
                    device="cuda")
    gs = GraphedTrainStep(spec, 96, 196, mode="adam", log=lambda *a, **k: None,
                          target_shape=(5,))
    gs.bind()
    idx = torch.arange(96, device="cuda")
    gs.step(x, y, idx)
    state = torch.cuda.get_rng_state()
    ref = torch.randperm(10000, device="cuda")
    torch.cuda.set_rng_state(state)
    for _ in range(4):
        gs.step(x, y, idx)          # eager, capture + checked, replays
    assert gs.replays >= 3 and not gs.disabled
    assert torch.equal(torch.cuda.get_rng_state(), state)
    assert torch.equal(torch.randperm(10000, device="cuda"), ref)


def test_spec_unsupported_reasons():
    net, opt, loss_fn, x, y = _toy()
    spec = StepSpec(net, opt, loss_fn, None, device="cuda")
    assert cuda_graph_step.unsupported_reason(spec, "adam") is None   # net on cpu: bind() moves it
    assert cuda_graph_step.resolve_mode(spec, "auto") == ("adam", None)
    assert "device is cpu" in cuda_graph_step.unsupported_reason(
        spec._replace(device="cpu"), "adam")
    assert "not on a CUDA device" in cuda_graph_step.unsupported_reason(
        spec._replace(device=None), "fwdbwd")
    assert "grad_clip" in cuda_graph_step.unsupported_reason(
        spec._replace(grad_clip=1.0), "adam")
    assert cuda_graph_step.resolve_mode(spec._replace(grad_clip=1.0), "auto") == ("fwdbwd", None)
    net.register_buffer("b", torch.zeros(1))
    assert "buffers" in cuda_graph_step.unsupported_reason(spec, "fwdbwd")
