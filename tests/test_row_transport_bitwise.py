"""The shared-memory row transport (`BG_COLLECT_SHM`, on by default,
row_transport.py) must leave training bitwise unchanged. The pool a round
trains on holds the same rows in the same order as with the pool pipe, so
the weights (torch.equal, tensor by tensor), the periodic checkpoints and
the per-batch losses must equal the stock transport's for the same seed.

Whole `Trainer.train` runs with the prod 4L nets, `--c-collect` (simd), 4
workers, 3 rounds of 400 episodes on cuda, each shared-memory run (the
default, `BG_COLLECT_SHM` unset) against the stock transport
(`BG_COLLECT_SHM=0`) under the same env: the graphed step
(`BG_TRAIN_CUDA_GRAPH=1`, device pool, slabs copied straight to the device
from the pinned arena), the eager step on the device pool, and the eager
step on the host pool (concatenate from slab views; both
`BG_TRAIN_CUDA_GRAPH=0`). Cases: cubeful money pipelined (two arenas
in flight) and unpipelined (one), and dmp pipelined with an LR anneal and
2 epochs per round. Also: forced slab overflows (the pipe fallback),
`Engine.collect(out=...)`, the NOTEs when the transport does not apply,
a pool restart, no `/dev/shm` block left behind, and nothing from the
multiprocessing resource tracker at exit. ~1 min.
"""

import os
import re
import shutil
import sys

import numpy as np
import pytest
import torch

from test_c_api import _build, PROD_CUBEFUL, REPO  # noqa: E402

import row_transport  # noqa: E402
from model import TDNetwork  # noqa: E402
from modes import CubefulMoneyMode, DMPMode  # noqa: E402
from td_agent import TDAgent  # noqa: E402
from trainer import Trainer  # noqa: E402

pytestmark = pytest.mark.cuda

BACKEND = "simd"
DMP4 = os.path.join(REPO, "best_models", "dmp_512_512_256_256.pt")
ENV_KEYS = ("BG_POOL_ON_DEVICE", "BG_TRAIN_CUDA_GRAPH", row_transport.ENV,
            "BG_COLLECT_DUMP")
ARMS = {
    "graph": {"BG_TRAIN_CUDA_GRAPH": "1"},
    "eager_device_pool": {"BG_POOL_ON_DEVICE": "1", "BG_TRAIN_CUDA_GRAPH": "0"},
    "eager_host_pool": {"BG_TRAIN_CUDA_GRAPH": "0"},
}
CASES = {
    "cubeful_pipelined": dict(model=PROD_CUBEFUL, mode=CubefulMoneyMode,
                              kw=dict(pipeline_collect=True)),
    "cubeful_sync": dict(model=PROD_CUBEFUL, mode=CubefulMoneyMode, kw={}),
    "dmp_pipelined_anneal_2epochs": dict(
        model=DMP4, mode=DMPMode,
        kw=dict(pipeline_collect=True, end_lr=1e-6, epochs_per_round=2)),
}


@pytest.fixture(scope="module")
def simd():
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    ok, log = _build(BACKEND)
    if not ok:
        pytest.skip(f"{BACKEND} backend not built:\n{log[-500:]}")
    return BACKEND


def _left_behind(names):
    """The arenas among `names` still present in /dev/shm."""
    return [n for n in names if os.path.exists(os.path.join("/dev/shm", n))]


def _train(mp, tmp, case, env, shm, seed=5):
    """One run; returns (state_dict, losses, checkpoints, stdout lines of
    the run's rounds and notes, metrics)."""
    for k in ENV_KEYS:
        mp.delenv(k, raising=False)
    for k, v in env.items():
        mp.setenv(k, v)
    if not shm:
        mp.setenv(row_transport.ENV, "0")    # unset (shm) is the default
    torch.manual_seed(seed)
    net = TDNetwork.load(case["model"])
    trainer = Trainer(TDAgent(net, device="cuda"), lr=1e-5)
    save = os.path.join(tmp, "shm" if shm else "pipe", "m.pt")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    made = []
    real_init = row_transport.Arena.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        made.append(self.name)

    mp.setattr(row_transport.Arena, "__init__", init)
    m = {}
    losses = trainer.train(
        case["mode"](), num_episodes=1200, batch_size=256,
        episodes_per_round=400, seed=seed, workers=4, log_every=1,
        c_collect=True, c_collect_backend=BACKEND, save_path=save,
        save_every=400, metrics_out=m, **case["kw"])
    mp.setattr(row_transport.Arena, "__init__", real_init)
    assert bool(made) == shm
    assert not _left_behind(made), "a shared-memory arena was left behind"
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    ckpts = {}
    for ep in (400, 800, 1200):
        p = save.replace(".pt", f"_ep{ep}.pt")
        ckpts[ep] = torch.load(p, map_location="cpu", weights_only=True)["state_dict"]
    return sd, losses, ckpts, m


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


@pytest.fixture(scope="module")
def stock_runs(simd, tmp_path_factory):
    """stock_runs(case, arm): the stock transport's run, computed once."""
    cache = {}

    def get(cname, arm):
        if (cname, arm) not in cache:
            mp = pytest.MonkeyPatch()
            try:
                tmp = str(tmp_path_factory.mktemp(f"{cname}_{arm}"))
                cache[cname, arm] = _train(mp, tmp, CASES[cname], ARMS[arm],
                                           shm=False)
            finally:
                mp.undo()
        return cache[cname, arm]
    return get


@pytest.mark.parametrize("case_name", sorted(CASES))
@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.slow
def test_shm_bitwise_equal_to_pipe(monkeypatch, capsys, tmp_path, stock_runs,
                                   case_name, arm):
    ref_sd, ref_losses, ref_ckpts, ref_m = stock_runs(case_name, arm)
    capsys.readouterr()
    sd, losses, ckpts, m = _train(monkeypatch, str(tmp_path), CASES[case_name],
                                  ARMS[arm], shm=True)
    out = capsys.readouterr().out
    assert ref_m["c_collect_transport"] == "pipe"
    assert m["c_collect_transport"] == "shm"
    assert "C collection rows: shared-memory arenas (the default" in out, out
    assert m.get("shm_overflows", 0) == 0
    if arm == "graph":
        assert "first replay bitwise equal to the eager step" in out, out
    assert len(losses) == len(ref_losses) > 200
    assert losses == ref_losses
    _same(sd, ref_sd)
    for ep in ref_ckpts:
        _same(ckpts[ep], ref_ckpts[ep])


@pytest.mark.slow
def test_slab_overflow_falls_back_bitwise(monkeypatch, capsys, tmp_path,
                                          stock_runs):
    """Slabs far too small: every worker of round 1 and most later ones
    replay into their own buffer and return through the pipe; the pool is
    still the same."""
    ref_sd, ref_losses, ref_ckpts, _ = stock_runs("cubeful_pipelined", "graph")
    monkeypatch.setattr(row_transport, "FIRST_ROWS_PER_EPISODE", 1.0)
    monkeypatch.setattr(row_transport, "MARGIN", 1.0)
    monkeypatch.setattr(row_transport, "PAD", 0)
    capsys.readouterr()
    sd, losses, ckpts, m = _train(monkeypatch, str(tmp_path),
                                  CASES["cubeful_pipelined"], ARMS["graph"],
                                  shm=True)
    out = capsys.readouterr().out
    assert "outgrew their slab in round 1 " not in out, out   # first-guess sizing: quiet
    assert m["shm_overflows"] >= 4
    assert losses == ref_losses
    _same(sd, ref_sd)
    for ep in ref_ckpts:
        _same(ckpts[ep], ref_ckpts[ep])


# ── Engine.collect(out=...) ──────────────────────────────────────────

def test_engine_collect_into_caller_buffers(simd, tmp_path):
    from c_api.puretd_py import Engine
    from export_weights import export_live
    from model import load_model
    path = str(tmp_path / "w.bin")
    export_live(load_model(PROD_CUBEFUL), path)
    eng = Engine(path, simd)
    try:
        enc0, tgt0, nd0 = eng.collect(None, 123, 6)
        n = len(tgt0)
        buf = (np.full((n + 7, 200), -1, np.float32), np.full(n + 7, -1, np.float32))
        enc, tgt, nd = eng.collect(None, 123, 6, out=buf)
        assert enc.ctypes.data == buf[0].ctypes.data and len(enc) == n
        assert np.array_equal(enc, enc0) and np.array_equal(tgt, tgt0)
        assert nd == nd0 and (buf[1][n:] == -1).all()
        small = (np.empty((n - 1, 200), np.float32), np.empty(n - 1, np.float32))
        enc, tgt, nd = eng.collect(None, 123, 6, out=small)   # -2: replay
        assert enc.ctypes.data != small[0].ctypes.data
        assert np.array_equal(enc, enc0) and np.array_equal(tgt, tgt0)
        for bad in ((np.empty((4, 196), np.float32), np.empty(4, np.float32)),
                    (np.empty((4, 200), np.float64), np.empty(4, np.float32)),
                    (np.empty((4, 200), np.float32), np.empty(5, np.float32)),
                    (np.empty((200, 4), np.float32).T, np.empty(4, np.float32))):
            with pytest.raises(ValueError):
                eng.collect(None, 123, 6, out=bad)
    finally:
        eng.close()


# ── where the transport does not apply ───────────────────────────────

def _tiny():
    return TDNetwork(hidden_sizes=[8], input_size=200, output_mode="equity",
                     encoder_name="cubeful_perspective196")


@pytest.mark.parametrize("c_collect,workers,needle", [
    (False, 2, "BG_COLLECT_SHM=1 ignored (the Python collector)"),
    (True, 1, "BG_COLLECT_SHM=1 ignored (--workers 1 collects in the master)"),
])
def test_shm_ignored_with_a_note(simd, monkeypatch, capsys, c_collect,
                                 workers, needle):
    monkeypatch.setenv(row_transport.ENV, "1")
    m = {}
    Trainer(TDAgent(_tiny(), device="cpu"), lr=1e-9).train(
        CubefulMoneyMode(), num_episodes=4, episodes_per_round=4,
        workers=workers, seed=1, max_episode_plies=30, log_every=0,
        c_collect=c_collect, c_collect_backend=simd, metrics_out=m)
    out = capsys.readouterr().out
    assert needle in out, out
    assert m.get("c_collect_transport") == ("in-process" if c_collect else None)


@pytest.mark.parametrize("c_collect,workers", [(False, 2), (True, 1)])
def test_shm_default_is_quiet_where_it_does_not_apply(simd, monkeypatch, capsys,
                                                      c_collect, workers):
    monkeypatch.delenv(row_transport.ENV, raising=False)
    Trainer(TDAgent(_tiny(), device="cpu"), lr=1e-9).train(
        CubefulMoneyMode(), num_episodes=4, episodes_per_round=4,
        workers=workers, seed=1, max_episode_plies=30, log_every=0,
        c_collect=c_collect, c_collect_backend=simd)
    assert row_transport.ENV not in capsys.readouterr().out


def test_shm_env_values(monkeypatch):
    monkeypatch.delenv(row_transport.ENV, raising=False)
    assert row_transport.enabled() and not row_transport.explicit()
    for v, on in (("", True), ("1", True), ("0", False)):
        monkeypatch.setenv(row_transport.ENV, v)
        assert row_transport.enabled() is on
        assert row_transport.explicit() is (v != "")
    monkeypatch.setenv(row_transport.ENV, "yes")
    with pytest.raises(ValueError):
        row_transport.enabled()


def test_shm_off_uses_the_pipe(simd, monkeypatch, capsys):
    monkeypatch.setenv(row_transport.ENV, "0")
    m = {}
    Trainer(TDAgent(_tiny(), device="cpu"), lr=1e-9).train(
        CubefulMoneyMode(), num_episodes=4, episodes_per_round=4,
        workers=2, seed=1, max_episode_plies=30, log_every=0,
        c_collect=True, c_collect_backend=simd, metrics_out=m)
    out = capsys.readouterr().out
    assert m["c_collect_transport"] == "pipe"
    assert "C collection rows: pickled through the pool pipe (BG_COLLECT_SHM=0)" in out, out


# ── process exit: the resource tracker ───────────────────────────────

TRACKER_NOISE = ("resource_tracker", "leaked shared_memory", "Traceback")


def _cli(simd, tmp_path, episodes=800):
    return [sys.executable, "-u", "train_batch.py", "--game-mode",
            "cubeful-money", "--resume", PROD_CUBEFUL, "--num-episodes",
            str(episodes), "--episodes-per-round", "80", "--workers", "4",
            "--pipeline-collect", "--device", "cuda", "--lr", "1e-9",
            "--c-collect", "--c-collect-backend", simd,
            "--save", str(tmp_path / "c.pt")]


@pytest.mark.slow
def test_no_resource_tracker_warning_at_exit(simd, tmp_path):
    """Workers attach each arena by name, which on Python 3.12 registers
    it with the multiprocessing resource tracker they share with the
    master; the master's unlink unregisters it once. Nothing may be
    reported at exit (stderr in full, the tracker's own included: it
    writes to the inherited stderr before the pipe closes)."""
    import subprocess
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
               BG_TRAIN_CUDA_GRAPH="1", **{row_transport.ENV: "1"})
    p = subprocess.run(_cli(simd, tmp_path), cwd=REPO, env=env, text=True,
                       capture_output=True, timeout=240)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "C collection rows: shared-memory arenas" in p.stdout
    noise = [ln for ln in p.stderr.splitlines()
             if any(k in ln for k in TRACKER_NOISE)]
    assert not noise, p.stderr
    made = re.findall(rf"{row_transport.ENV}: arena \d/\d (\S+):", p.stdout)
    assert made and not _left_behind(made)


# ── pool restart ──────────────────────────────────────────────────────

@pytest.mark.slow
def test_shm_recovers_from_a_dead_worker(simd, monkeypatch, tmp_path):
    """A worker SIGKILLed mid-run (maybe mid-slab): the pool and the arenas
    are rebuilt, the run finishes, nothing is left in /dev/shm."""
    from test_pool_guard import _run_and_kill_a_worker
    monkeypatch.setenv(row_transport.ENV, "1")
    monkeypatch.setenv("BG_TRAIN_CUDA_GRAPH", "1")
    rc, out = _run_and_kill_a_worker(_cli(simd, tmp_path, episodes=960))
    assert rc == 0, out[-3000:]
    assert "C collection rows: shared-memory arenas" in out
    assert "rebuilding the collection pool" in out
    made = re.findall(rf"{row_transport.ENV}: arena \d/\d (\S+):", out)
    assert len(made) >= 4, out          # the restart recreated them
    assert "Round   12/12" in out
    assert (tmp_path / "c.pt").exists()
    assert not _left_behind(made)
    # stdout and stderr merged, the tracker's exit report included
    assert not [ln for ln in out.splitlines()
                if any(k in ln for k in TRACKER_NOISE)], out
