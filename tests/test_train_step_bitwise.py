"""The train-step options (`BG_POOL_ON_DEVICE`, `BG_TRAIN_CUDA_GRAPH`) must
leave training bitwise unchanged: same weights (torch.equal, tensor by
tensor) and the same per-batch losses as the stock eager step, for the
same seed and inputs.

Stock reference: the host pool (`BG_POOL_ON_DEVICE` unset) with a
`.item()` per batch and the eager step (`BG_TRAIN_CUDA_GRAPH=0`); the
"stock" arm patches the `.item()` into `Trainer.train`. Arms: the host
pool reading its losses back once per pass,
`BG_POOL_ON_DEVICE=1` (both eager, `BG_TRAIN_CUDA_GRAPH=0`), and
`BG_TRAIN_CUDA_GRAPH` (cuda_graph_step.py): modes fwdbwd and adam, and
unset, the default (auto = adam here).

Whole runs go through `Trainer.train` with the real prod nets, 2 Python
collection workers (c_collect=False), 3 rounds of 100 episodes (pipelined
cubeful-money, and dmp with an LR anneal and 2 epochs per round). CUDA
only.
"""

import os

import pytest
import torch

import cuda_graph_step
from cuda_graph_step import GraphedTrainStep
from model import TDNetwork
from modes import CubefulMoneyMode, DMPMode
from td_agent import TDAgent
from trainer import Trainer

pytestmark = pytest.mark.cuda

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CF4 = os.path.join(REPO, "best_models", "cubeful_money_512_512_256_256.pt")
DMP4 = os.path.join(REPO, "best_models", "dmp_512_512_256_256.pt")

ENV_KEYS = ("BG_POOL_ON_DEVICE", "BG_TRAIN_CUDA_GRAPH")
ARMS = {
    "stock": {"BG_TRAIN_CUDA_GRAPH": "0"},
    "host_pool": {"BG_TRAIN_CUDA_GRAPH": "0"},
    "pool_on_device": {"BG_POOL_ON_DEVICE": "1", "BG_TRAIN_CUDA_GRAPH": "0"},
    "graph_fwdbwd": {"BG_TRAIN_CUDA_GRAPH": "fwdbwd"},
    "graph_adam": {"BG_TRAIN_CUDA_GRAPH": "adam"},
    "graph_default": {},            # unset: on (auto)
}
CASES = {
    "cubeful_pipelined": dict(model=CF4, mode=CubefulMoneyMode,
                              kw=dict(pipeline_collect=True)),
    "dmp_anneal_2epochs": dict(model=DMP4, mode=DMPMode,
                               kw=dict(end_lr=1e-6, epochs_per_round=2)),
}


_nosync = Trainer.train_step_nosync


def _item_per_batch(self, batch_e, batch_t):
    # The stock host-pool loop: a .item() host sync after every batch.
    loss = _nosync(self, batch_e, batch_t).item()
    return torch.tensor(loss, dtype=torch.float32)


def _train(monkeypatch, arm, case, seed=3):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    for k, v in ARMS[arm].items():
        monkeypatch.setenv(k, v)
    if arm == "stock":
        monkeypatch.setattr(Trainer, "train_step_nosync", _item_per_batch)
    torch.manual_seed(seed)
    net = TDNetwork.load(case["model"])
    agent = TDAgent(net, device="cuda")
    trainer = Trainer(agent, lr=1e-5)
    losses = trainer.train(
        case["mode"](), num_episodes=300, batch_size=256,
        episodes_per_round=100, seed=seed, workers=2, log_every=0,
        c_collect=False, **case["kw"])
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return sd, losses


@pytest.fixture(scope="module")
def stock_runs():
    """Stock run per case, computed once."""
    mp = pytest.MonkeyPatch()
    try:
        return {name: _train(mp, "stock", case)
                for name, case in CASES.items()}
    finally:
        mp.undo()


@pytest.mark.parametrize("case_name", sorted(CASES))
@pytest.mark.parametrize("arm", ["host_pool", "pool_on_device",
                                 "graph_fwdbwd", "graph_adam",
                                 "graph_default"])
@pytest.mark.slow
def test_run_bitwise_equal_to_stock(monkeypatch, capsys, stock_runs,
                                    case_name, arm):
    ref_sd, ref_losses = stock_runs[case_name]
    capsys.readouterr()
    sd, losses = _train(monkeypatch, arm, CASES[case_name])
    out = capsys.readouterr().out
    if arm.startswith("graph"):
        # The graph must actually have run (no silent eager fallback).
        assert "first replay bitwise equal to the eager step" in out, out
        assert "WARN: BG_TRAIN_CUDA_GRAPH" not in out, out
        if arm == "graph_default":
            assert "BG_TRAIN_CUDA_GRAPH (default on;" in out and "(mode adam)" in out, out
    else:
        # BG_TRAIN_CUDA_GRAPH=0: the eager step, and nothing said about it.
        assert "BG_TRAIN_CUDA_GRAPH" not in out, out
    assert len(losses) == len(ref_losses) > 60
    assert losses == ref_losses
    assert sd.keys() == ref_sd.keys()
    for k in ref_sd:
        assert torch.equal(sd[k], ref_sd[k]), k


# ── GraphedTrainStep directly (no pool) ─────────────────────────────


def _toy(seed=0, n=2000, nfeat=200, opt="adam"):
    torch.manual_seed(seed)
    net = TDNetwork(hidden_sizes=[64, 32], input_size=nfeat,
                    output_mode="equity", encoder_name="cubeful_perspective196")
    agent = TDAgent(net, device="cuda")
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((n, nfeat), generator=g).cuda()
    y = torch.randn((n,), generator=g).cuda()
    perm = torch.randperm(n, generator=g).cuda()
    if opt == "sgd_clip":
        tr = Trainer(agent, lr=1e-3, optimizer_cls=torch.optim.SGD,
                     optimizer_kwargs={"momentum": 0.9}, grad_clip=0.05)
    else:
        tr = Trainer(agent, lr=1e-3)
    return tr, x, y, perm


def _steps(trainer, x, y, perm, batch, step):
    out = []
    for s in range(0, len(perm), batch):
        out.append(step(x, y, perm[s:s + batch]))
    return torch.stack(out).tolist()


@pytest.mark.parametrize("mode,opt,chunk", [
    ("adam", "adam", None), ("adam", "adam", "1000"), ("adam", "adam", "0"),
    ("fwdbwd", "adam", None), ("fwdbwd", "sgd_clip", None)])
def test_graphed_steps_match_eager(monkeypatch, mode, opt, chunk):
    if chunk is not None:
        # 1000: uneven pieces of every tensor; 0: one entry per tensor.
        monkeypatch.setenv(cuda_graph_step.CHUNK_ENV, chunk)
    tr_e, x, y, perm = _toy(opt=opt)
    eager = _steps(tr_e, x, y, perm, 96,
                   lambda px, py, i: tr_e.train_step_dev(px[i], py[i]))
    tr_g, x, y, perm = _toy(opt=opt)
    gs = GraphedTrainStep(tr_g, 96, x.shape[1], mode=mode, log=lambda *a, **k: None)
    graphed = _steps(tr_g, x, y, perm, 96, gs.step)
    assert gs.replays >= 19 and not gs.disabled
    assert graphed == eager
    for a, b in zip(tr_e.agent.network.parameters(),
                    tr_g.agent.network.parameters()):
        assert torch.equal(a, b)


def test_self_check_falls_back_to_the_eager_result(monkeypatch):
    """A graph that disagrees with the eager step must be dropped, and
    the run continues from the eager result as if it never existed."""
    tr_e, x, y, perm = _toy(1)
    eager = _steps(tr_e, x, y, perm, 96,
                   lambda px, py, i: tr_e.train_step_dev(px[i], py[i]))
    tr_g, x, y, perm = _toy(1)
    msgs = []
    gs = GraphedTrainStep(tr_g, 96, x.shape[1], mode="adam",
                          log=lambda m, **k: msgs.append(m))
    real_replay = GraphedTrainStep._replay

    def bad_replay(self, *a):
        loss = real_replay(self, *a)
        with torch.no_grad():
            self.params[0].add_(1e-3)
        return loss

    monkeypatch.setattr(GraphedTrainStep, "_replay", bad_replay)
    graphed = _steps(tr_g, x, y, perm, 96, gs.step)
    assert gs.disabled and gs.replays == 1
    assert any("NOT bitwise equal" in m for m in msgs)
    assert graphed == eager
    for a, b in zip(tr_e.agent.network.parameters(),
                    tr_g.agent.network.parameters()):
        assert torch.equal(a, b)


def test_mode_from_env(monkeypatch):
    # Unset (or empty) is on. 0 is the only way off.
    monkeypatch.delenv("BG_TRAIN_CUDA_GRAPH", raising=False)
    assert cuda_graph_step.mode_from_env() == "auto"
    assert not cuda_graph_step.env_explicit()
    assert cuda_graph_step.env_label().startswith("BG_TRAIN_CUDA_GRAPH (default on;")
    for v, want in (("", "auto"), ("0", None), ("1", "auto"),
                    ("adam", "adam"), ("fwdbwd", "fwdbwd")):
        monkeypatch.setenv("BG_TRAIN_CUDA_GRAPH", v)
        assert cuda_graph_step.mode_from_env() == want
        assert cuda_graph_step.env_explicit() is (v != "")
    assert cuda_graph_step.env_label() == "BG_TRAIN_CUDA_GRAPH=fwdbwd"
    monkeypatch.setenv("BG_TRAIN_CUDA_GRAPH", "yes")
    with pytest.raises(ValueError):
        cuda_graph_step.mode_from_env()


def test_unsupported_reasons():
    tr, *_ = _toy()
    assert cuda_graph_step.unsupported_reason(tr, "adam") is None
    tr.grad_clip = 1.0
    assert "grad_clip" in cuda_graph_step.unsupported_reason(tr, "adam")
    assert cuda_graph_step.unsupported_reason(tr, "fwdbwd") is None
    assert cuda_graph_step.resolve_mode(tr, "auto") == ("fwdbwd", None)
    tr.grad_clip = None
    assert cuda_graph_step.resolve_mode(tr, "auto") == ("adam", None)
    sgd = Trainer(tr.agent, lr=1e-3, optimizer_cls=torch.optim.SGD)
    assert "Adam" in cuda_graph_step.unsupported_reason(sgd, "adam")
    assert cuda_graph_step.unsupported_reason(sgd, "fwdbwd") is None
    assert cuda_graph_step.resolve_mode(sgd, "adam")[0] is None
