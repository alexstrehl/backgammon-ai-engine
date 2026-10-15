"""The train-step levers must leave match-play training bitwise
unchanged: same weights (state_dict bytes and torch.equal, tensor
by tensor), same Adam state, the same per-batch losses and the same round
log lines as the stock eager step, for the same seed and pool.

Stock reference: `BG_TRAIN_CUDA_GRAPH=0` and `BG_POOL_ON_DEVICE` unset, i.e.
the host pool with a `.item()` host sync per batch, as `Trainer.train` did
before `train_step_nosync`: the "stock" arm patches that per-batch
`.item()` back in.
Levers: the host pool reading its losses back once per pass
(`train_step_nosync`: arm `host_pool`), `BG_TRAIN_CUDA_GRAPH=0` (the same
host loop), `BG_POOL_ON_DEVICE=1` (eager step on the device pool), and
`BG_TRAIN_CUDA_GRAPH` = 1 (auto, = adam here), adam, fwdbwd, and unset
(the default: auto; cuda_graph_step.py, which implies the device
pool).

Whole runs go through `Trainer.train` with the champion net
(`best_models/matchplay_1h64met_512_512_256_256.pt`, md5 dbe2579a) and
the pool recorded from an earlier champion (md5 db492172), 3-point
match play, >= 3 rounds, on a REAL pool collected once by the serial
collector with the leg's settings (1-ply targets, MET auto, all scopes;
`tests/m4_make_pool.py`): the file named by `BG_TRAIN_STEP_POOL`, else a
233-match pool collected into a temp dir, ~25 s on 8 workers). A pool
must give every train() call more than 200 batches and end on a match
longer than its row count mod the batch size (the full_last_batch case);
233 matches of NET's play do, with margin (33,324 rows, last match 223).
The round's `collect_episode` calls hand out its matches in order, so every arm trains
on the same rows; the round loop (seed draws, shuffle, LR schedule,
batching, checkpoints) is the real one. Cases: the leg's LR (3.1e-6),
`--lr 1e-9`, an anneal with warm-up and 2 epochs per round, a resumed
optimizer state (Adam state_dict saved and loaded into a fresh Trainer),
`save_every` checkpoints (file bytes) and a resume from one (the last two
at both LRs), and rounds that end on a full batch. CUDA only.
"""

import contextlib
import hashlib
import io
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import cuda_graph_step
import trainer as trainer_mod
from cuda_graph_step import GraphedTrainStep
from met import MatchEquityTable, resolve_met_path
from model import TDNetwork
from modes import MatchplayMode
from td_agent import TDAgent
from trainer import Trainer

pytestmark = pytest.mark.cuda

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(REPO, "best_models", "matchplay_1h64met_512_512_256_256.pt")
POOL = os.environ.get("BG_TRAIN_STEP_POOL", "")
MET_PATH = resolve_met_path("auto")
SCOPES = frozenset(("boundary", "cube", "terminal"))
LEG_LR = 3.1e-6
BATCH = 256

ENV_KEYS = ("BG_POOL_ON_DEVICE", "BG_TRAIN_CUDA_GRAPH")
ARMS = {
    "stock": {"BG_TRAIN_CUDA_GRAPH": "0"},      # eager host loop + .item() patch
    "host_pool": {"BG_TRAIN_CUDA_GRAPH": "0"},  # the same, unpatched: losses once per pass
    "pool_on_device": {"BG_POOL_ON_DEVICE": "1", "BG_TRAIN_CUDA_GRAPH": "0"},
    "graph_auto": {"BG_TRAIN_CUDA_GRAPH": "1"},
    "graph_adam": {"BG_TRAIN_CUDA_GRAPH": "adam"},
    "graph_fwdbwd": {"BG_TRAIN_CUDA_GRAPH": "fwdbwd"},
    "graph_default": {},            # unset: on (auto)
}


# ── the pool ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def episodes(tmp_path_factory):
    """The pool's matches as a list of (enc, tgt), in collection order."""
    path = POOL
    if not path:
        path = str(tmp_path_factory.mktemp("train_step_pool") / "pool_3pt_233.npz")
        env = dict(os.environ, TRAINER_SCORE_CURRICULUM_P="0.0",
                   OMP_NUM_THREADS="1")
        subprocess.run([sys.executable, os.path.join(
            REPO, "tests", "m4_make_pool.py"),
            "--out", path, "--episodes", "233", "--workers", "8"],
            check=True, env=env, cwd=REPO)
    z = np.load(path)
    # The prebuilt pool was collected with an earlier champion net
    # (db492172); a pool collected here comes from the current champion
    # file (NET).
    with open(NET, "rb") as f:
        net_md5 = hashlib.md5(f.read()).hexdigest()
    assert str(z["net_md5"]).startswith(("db492172", net_md5)), z["net_md5"]
    assert z["enc"].shape[1] == 350 and z["enc"].dtype == np.float32
    bounds = np.concatenate([[0], np.cumsum(z["ep_rows"])])
    return [(z["enc"][a:b], z["tgt"][a:b])
            for a, b in zip(bounds[:-1], bounds[1:])]


class _Replay:
    """Stands in for `trainer.collect_episode`: match k of a round is
    match k of the pool. `full_batches`: the round's last match is cut
    so that the round ends on a full batch."""

    def __init__(self, episodes, full_batches=False):
        self.eps = episodes
        self.full = full_batches
        self.k = 0
        if full_batches:
            n = sum(len(t) for _, t in episodes)
            self.cut = n - n % BATCH
            assert len(episodes[-1][1]) > n % BATCH

    def __call__(self, agent, mode, rng, **kw):
        i = self.k % len(self.eps)
        self.k += 1
        enc, tgt = self.eps[i]
        if self.full and i == len(self.eps) - 1:
            keep = len(tgt) - (sum(len(t) for _, t in self.eps) - self.cut)
            enc, tgt = enc[:keep], tgt[:keep]
        return enc.copy(), tgt.copy()


# ── runs ─────────────────────────────────────────────────────────────


def _trainer(lr, net=None):
    net = net if net is not None else TDNetwork.load(NET)
    agent = TDAgent(net, device="cuda", met=MatchEquityTable.from_xml(MET_PATH),
                    met_scopes=SCOPES)
    return Trainer(agent, lr=lr)


def _fit(tr, episodes, rounds, seed=7, full_batches=False, **kw):
    mp = pytest.MonkeyPatch()
    mp.setattr(trainer_mod, "collect_episode", _Replay(episodes, full_batches))
    try:
        return tr.train(MatchplayMode(match_length=3),
                        num_episodes=rounds * len(episodes),
                        batch_size=BATCH, episodes_per_round=len(episodes),
                        seed=seed, workers=1, log_every=1, oneply=True,
                        met_path=MET_PATH, met_scopes=SCOPES, **kw)
    finally:
        mp.undo()


def _sd_bytes(sd):
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in sd.items()}, buf)
    return buf.getvalue()


def _snapshot(tr):
    sd = {k: v.detach().cpu().clone()
          for k, v in tr.agent.network.state_dict().items()}
    opt = tr.optimizer.state_dict()["state"]
    adam = {(i, k): v.detach().cpu().clone()
            for i, st in opt.items() for k, v in st.items()}
    return sd, adam


def _case_plain(episodes, tmp, lr, rounds=3, **kw):
    tr = _trainer(lr)
    losses = _fit(tr, episodes, rounds, **kw)
    return tr, [losses], {}


def _case_resume_optimizer(episodes, tmp, lr=LEG_LR):
    # Adam state with real step counts, saved and loaded into a fresh
    # Trainer: the graph's first steps then start from a loaded state.
    tr = _trainer(lr)
    l1 = _fit(tr, episodes, 2, seed=7)
    torch.save(tr.agent.network.state_dict(), tmp / "net.pt")
    torch.save(tr.optimizer.state_dict(), tmp / "opt.pt")
    net = TDNetwork.load(NET)
    net.load_state_dict(torch.load(tmp / "net.pt", map_location="cpu"))
    tr2 = _trainer(lr, net)
    tr2.optimizer.load_state_dict(torch.load(tmp / "opt.pt"))
    l2 = _fit(tr2, episodes, 3, seed=8)
    return tr2, [l1, l2], {}


def _case_save_every(episodes, tmp, lr=LEG_LR):
    # Checkpoints every round; then `--resume` from the round-2 one (the
    # checkpoint is weights-only: a fresh Adam, as train_batch.py does).
    tr = _trainer(lr)
    save = tmp / "ck.pt"
    l1 = _fit(tr, episodes, 3, save_path=str(save), save_every=len(episodes))
    files = {}
    for r in (1, 2, 3):
        p = tmp / f"ck_ep{r * len(episodes)}.pt"
        with open(p, "rb") as f:
            files[p.name] = hashlib.md5(f.read()).hexdigest()
    net = TDNetwork.load(str(tmp / f"ck_ep{2 * len(episodes)}.pt"))
    tr2 = _trainer(lr, net)
    l2 = _fit(tr2, episodes, 3, seed=9)
    return tr2, [l1, l2], files


CASES = {
    "leg_lr": lambda e, t: _case_plain(e, t, LEG_LR),
    "lr_1e-9": lambda e, t: _case_plain(e, t, 1e-9),
    "anneal_warmup_2epochs": lambda e, t: _case_plain(
        e, t, LEG_LR, end_lr=1.1e-6, warmup_cycles=2, epochs_per_round=2),
    "full_last_batch": lambda e, t: _case_plain(e, t, LEG_LR,
                                                full_batches=True),
    "resume_optimizer": _case_resume_optimizer,
    "resume_optimizer_lr1e-9": lambda e, t: _case_resume_optimizer(e, t, 1e-9),
    "save_every_resume": _case_save_every,
    "save_every_resume_lr1e-9": lambda e, t: _case_save_every(e, t, 1e-9),
}
# Every case runs these; the rest of ARMS on the leg's case only.
CORE_ARMS = ("host_pool", "pool_on_device", "graph_auto", "graph_fwdbwd")
# Arms on the host-pool loop (train_step_nosync per batch); the others
# train on the device pool and never call it.
HOST_ARMS = ("stock", "host_pool")

_nosync = Trainer.train_step_nosync


def _item_per_batch(self, batch_e, batch_t):
    # The pre-port host-pool loop: a .item() host sync after every batch
    # (`losses.append(self.train_step(...))`). The float goes back into
    # a float32 tensor exactly, so the loop's stack().tolist() yields it.
    loss = _nosync(self, batch_e, batch_t).item()
    return torch.tensor(loss, dtype=torch.float32)


def _run(monkeypatch, tmp_path, arm, case, episodes):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    for k, v in ARMS[arm].items():
        monkeypatch.setenv(k, v)
    host_steps = []
    inner = _item_per_batch if arm == "stock" else _nosync

    def counting_nosync(self, *a):
        host_steps.append(1)
        return inner(self, *a)

    monkeypatch.setattr(Trainer, "train_step_nosync", counting_nosync)
    replays = []
    real = GraphedTrainStep._replay

    def counting(self, *a):
        replays.append(1)
        return real(self, *a)

    monkeypatch.setattr(GraphedTrainStep, "_replay", counting)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tr, loss_lists, files = CASES[case](episodes, tmp_path)
    out = buf.getvalue()
    sd, adam = _snapshot(tr)
    rounds = [ln for ln in out.splitlines() if ln.startswith("Round ")]
    return dict(sd=sd, adam=adam, losses=loss_lists, files=files,
                rounds=rounds, out=out, replays=len(replays),
                host_steps=len(host_steps))


@pytest.fixture(scope="module")
def stock_runs(episodes, tmp_path_factory):
    """The stock run of every case, computed once."""
    mp = pytest.MonkeyPatch()
    try:
        return {name: _run(mp, tmp_path_factory.mktemp(f"stock_{name}"),
                           "stock", name, episodes)
                for name in CASES}
    finally:
        mp.undo()


def _full_batches(episodes, case):
    n = sum(len(t) for _, t in episodes)
    per_round = n // BATCH
    # (rounds x epochs) per train() call, one GraphedTrainStep per call.
    calls = {"leg_lr": [3], "lr_1e-9": [3], "anneal_warmup_2epochs": [6],
             "full_last_batch": [3], "resume_optimizer": [2, 3],
             "save_every_resume": [3, 3]}[case.replace("_lr1e-9", "")]
    return [c * per_round for c in calls]


PAIRS = ([(c, a) for c in sorted(CASES) for a in CORE_ARMS]
         + [("leg_lr", a) for a in ARMS if a not in CORE_ARMS + ("stock",)])


@pytest.mark.parametrize("case,arm", PAIRS)
def test_run_bitwise_equal_to_stock(monkeypatch, tmp_path, episodes,
                                    stock_runs, case, arm):
    ref = stock_runs[case]
    got = _run(monkeypatch, tmp_path, arm, case, episodes)
    out = got["out"]
    if arm.startswith("graph_"):
        # The graph must actually have run (no silent eager fallback),
        # every full batch after the first of each call replayed.
        assert "first replay bitwise equal to the eager step" in out, out
        assert "WARN: BG_TRAIN_CUDA_GRAPH" not in out, out
        mode = "fwdbwd" if arm == "graph_fwdbwd" else "adam"
        assert f"(mode {mode})" in out, out
        assert got["replays"] == sum(f - 1 for f in
                                     _full_batches(episodes, case))
    else:
        assert "BG_TRAIN_CUDA_GRAPH" not in out, out
        assert got["replays"] == 0
        if arm == "pool_on_device":
            assert "BG_POOL_ON_DEVICE=1: the pool is kept on cuda" in out
        else:
            assert "BG_POOL_ON_DEVICE" not in out, out
    assert "BG_TRAIN_CUDA_GRAPH" not in ref["out"]
    assert "BG_POOL_ON_DEVICE" not in ref["out"]
    # The host arms train through train_step_nosync, one call per batch
    # (stock through the per-batch .item() patch); the device arms never.
    assert ref["host_steps"] == sum(len(x) for x in ref["losses"])
    assert got["host_steps"] == (sum(len(x) for x in got["losses"])
                                 if arm in HOST_ARMS else 0)
    assert [len(x) for x in got["losses"]] == [len(x) for x in ref["losses"]]
    assert all(len(x) > 200 for x in ref["losses"])
    assert got["losses"] == ref["losses"]
    assert got["rounds"] == ref["rounds"] and len(ref["rounds"]) >= 3
    assert got["files"] == ref["files"]
    assert _sd_bytes(got["sd"]) == _sd_bytes(ref["sd"])
    assert got["sd"].keys() == ref["sd"].keys()
    for k in ref["sd"]:
        assert torch.equal(got["sd"][k], ref["sd"][k]), k
    assert got["adam"].keys() == ref["adam"].keys()
    for k in ref["adam"]:
        assert torch.equal(got["adam"][k], ref["adam"][k]), k


def test_weights_moved_at_the_leg_lr(stock_runs):
    """The bitwise test above is not vacuous: the leg's LR moves weights
    in every case, and 1e-9 moves them too (Adam's step is ~lr)."""
    base = TDNetwork.load(NET).state_dict()
    for name, r in stock_runs.items():
        diff = max((r["sd"][k] - base[k]).abs().max().item() for k in base)
        assert diff > 0, name
        if "1e-9" not in name:
            assert diff > 1e-6, (name, diff)


# ── GraphedTrainStep directly (no pool) ─────────────────────────────


def _toy(seed=0, n=2000, nfeat=350, opt="adam"):
    torch.manual_seed(seed)
    net = TDNetwork(hidden_sizes=[64, 32], input_size=nfeat,
                    output_mode="probability",
                    encoder_name="matchplay1h64met_perspective196")
    agent = TDAgent(net, device="cuda")
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((n, nfeat), generator=g).cuda()
    y = torch.rand((n,), generator=g).cuda()
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


def test_train_step_dev_equals_train_step():
    """The device step, the host step and the host step without the
    per-batch sync on the same batch: same loss, same weights (the device
    pool's eager step is the stock one)."""
    tr_h, x, y, perm = _toy(2)
    tr_d, *_ = _toy(2)
    tr_n, *_ = _toy(2)
    for s in range(0, 960, 96):
        i = perm[s:s + 96]
        lh = tr_h.train_step(x[i].cpu().numpy(), y[i].cpu().numpy())
        ld = tr_d.train_step_dev(x[i], y[i]).item()
        ln = tr_n.train_step_nosync(x[i].cpu().numpy(), y[i].cpu().numpy())
        assert ln.is_cuda and ln.dim() == 0 and not ln.requires_grad
        assert lh == ld == ln.item()
    for a, b, c in zip(tr_h.agent.network.parameters(),
                       tr_d.agent.network.parameters(),
                       tr_n.agent.network.parameters()):
        assert torch.equal(a, b) and torch.equal(a, c)


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


def test_bad_env_fails_before_training(monkeypatch, episodes):
    monkeypatch.setenv("BG_TRAIN_CUDA_GRAPH", "yes")
    tr = _trainer(LEG_LR)
    before = _snapshot(tr)[0]
    with pytest.raises(ValueError, match="BG_TRAIN_CUDA_GRAPH"):
        _fit(tr, episodes, 1)
    after = _snapshot(tr)[0]
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_cpu_device_ignores_the_levers(monkeypatch, capsys, episodes):
    """Off cuda the explicit levers say so and the run is the stock one."""
    def run(env):
        for k in ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        net = TDNetwork.load(NET)
        agent = TDAgent(net, device="cpu",
                        met=MatchEquityTable.from_xml(MET_PATH),
                        met_scopes=SCOPES)
        tr = Trainer(agent, lr=LEG_LR)
        losses = _fit(tr, episodes[:8], 1)
        return losses, capsys.readouterr().out
    ref, out0 = run({})
    got, out1 = run({"BG_TRAIN_CUDA_GRAPH": "1", "BG_POOL_ON_DEVICE": "1"})
    assert got == ref
    assert "BG_TRAIN_CUDA_GRAPH" not in out0
    assert "BG_TRAIN_CUDA_GRAPH ignored (device is cpu)" in out1
    assert "BG_POOL_ON_DEVICE=1: ignored (device is cpu)" in out1


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
