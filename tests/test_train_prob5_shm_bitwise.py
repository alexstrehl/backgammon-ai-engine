"""The shared-memory row transport in train_prob5.py (`BG_COLLECT_SHM`,
on by default with the C collector and a worker pool,
row_transport.py with (n, 5) targets) must leave training bitwise
unchanged. The pool a round trains on holds the same rows in the same
order as with the pool pipe, so the final weights (torch.equal, tensor by
tensor), every periodic checkpoint, the per-batch losses, the per-output
loss averages and the round log lines must equal the pipe's
(`BG_COLLECT_SHM=0`) for the same seed.

Whole `train_prob5.train` runs with the prod prob5 net, the C collector
(simd), 4 workers, 3 rounds of 400 episodes, each shared-memory run (the
default, `BG_COLLECT_SHM` unset) against the pipe under the same env:
the graphed step (`BG_TRAIN_CUDA_GRAPH=1`, slabs copied straight from the
pinned arena into the device pool), the eager step on cuda (the same
direct copy), the graphed step with `BG_ROUND_PHASES=1` (against the
graph pipe run without it), and a cpu device (a concatenate from the slab
views). Cases: 0-ply pipelined (two arenas in flight) and synchronous
(one), and 1-ply pipelined with an LR anneal, two warm-up rounds and a
log every 2 rounds (the rounds that are not logged skip the host copy of
the targets). Also: forced slab overflows (the pipe fallback, bitwise),
`BG_COLLECT_DUMP` under shm, the NOTEs where the transport does not
apply, a SIGKILLed worker (pool and arenas rebuilt), no `/dev/shm` block
left behind, and nothing from the multiprocessing resource tracker at
exit. CUDA only; ~3 min.
"""

import contextlib
import glob
import io
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_c_api import _build, PROD_PROB5, REPO  # noqa: E402

import row_transport  # noqa: E402
import train_prob5 as TP  # noqa: E402
from model import ProbNetwork  # noqa: E402

pytestmark = pytest.mark.cuda

BACKEND = "simd"
ENV_KEYS = ("BG_TRAIN_CUDA_GRAPH", row_transport.ENV, "BG_ROUND_PHASES",
            "BG_COLLECT_DUMP", "BG_POOL_ON_DEVICE")
# arm -> (env, device)
ARMS = {
    "graph": ({"BG_TRAIN_CUDA_GRAPH": "1"}, "cuda"),
    "eager": ({"BG_TRAIN_CUDA_GRAPH": "0"}, "cuda"),
    "graph_phases": ({"BG_TRAIN_CUDA_GRAPH": "1", "BG_ROUND_PHASES": "1"}, "cuda"),
    "cpu": ({}, "cpu"),
}
# The pipe run an arm's shm run is compared with (phases: the plain graph).
REF_ARM = {"graph_phases": "graph"}
CASES = {
    "0ply_pipelined": dict(pipeline_collect=True),
    "0ply_sync": dict(),
    "1ply_pipelined_anneal_warmup_log2": dict(
        oneply=True, pipeline_collect=True, end_lr=1e-6, warmup_cycles=2,
        log_every=2),
}
CASE_ARMS = {
    "0ply_pipelined": ["graph", "eager", "graph_phases", "cpu"],
    "0ply_sync": ["graph", "eager", "cpu"],
    "1ply_pipelined_anneal_warmup_log2": ["graph"],
}
EPR, ROUNDS, WORKERS = 400, 3, 4
SHM_ON = ("C collection rows: shared-memory arenas (the default, "
          "BG_COLLECT_SHM=0 for the pool pipe;")
PIPE = "C collection rows: pickled through the pool pipe (BG_COLLECT_SHM=0)"


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


def _log_lines(out):
    """The round and per-output lines without the timing fields."""
    keep = []
    for ln in out.splitlines():
        if ln.startswith("Round ") or "per-output loss:" in ln:
            keep.append(re.sub(r"\| collect .*$", "", ln))
    return keep


def _train(mp, tmp, case, arm, shm, seed=3, env_extra=None):
    """One train_prob5.train run; returns a dict of its state_dict, init
    weights, checkpoints (periodic by round, final), stdout, metrics and
    the arenas it created."""
    env, device = ARMS[arm]
    for k in ENV_KEYS:
        mp.delenv(k, raising=False)
    for k, v in dict(env, **(env_extra or {})).items():
        mp.setenv(k, v)
    if not shm:
        mp.setenv(row_transport.ENV, "0")      # unset (shm) is the default
    made = []
    real_init = row_transport.Arena.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        made.append(self.name)

    mp.setattr(row_transport.Arena, "__init__", init)
    torch.manual_seed(seed)
    net = ProbNetwork.load(PROD_PROB5)
    init_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
    save = os.path.join(tmp, "shm" if shm else "pipe", "m.pt")
    os.makedirs(os.path.dirname(save))
    metrics = {}
    args = dict(num_episodes=EPR * ROUNDS, lr=1e-4, end_lr=None,
                episodes_per_round=EPR, batch_size=256, workers=WORKERS,
                device=device, oneply=False, pw_weight=1.0, use_mse=False,
                seed=seed, log_every=1, save_path=save, save_every=EPR,
                warmup_cycles=0, c_collect=True, c_collect_backend=BACKEND,
                metrics_out=metrics)
    args.update(case)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        TP.train(net, **args)
    mp.setattr(row_transport.Arena, "__init__", real_init)
    ckpts = {n: torch.load(save.replace(".pt", f"_ep{n * EPR}.pt"),
                           map_location="cpu", weights_only=True)
             for n in range(1, ROUNDS + 1)}
    ckpts["final"] = torch.load(save, map_location="cpu", weights_only=True)
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return dict(sd=sd, init=init_sd, ckpts=ckpts, out=buf.getvalue(),
                metrics=metrics, made=made)


def _same_ckpt(a, b, what):
    assert a.keys() == b.keys(), what
    for k in a:
        if k == "state_dict":
            assert a[k].keys() == b[k].keys(), what
            for t in a[k]:
                assert torch.equal(a[k][t], b[k][t]), (what, t)
        else:
            assert a[k] == b[k], (what, k)


def _assert_bitwise(got, ref):
    m, rm = got["metrics"], ref["metrics"]
    # The weights moved: the equalities below are not trivially true.
    assert any(not torch.equal(ref["sd"][k], ref["init"][k]) for k in ref["sd"])
    assert len(m["batch_losses"]) == len(rm["batch_losses"]) > ROUNDS * 80
    assert m["batch_losses"] == rm["batch_losses"]
    assert torch.equal(torch.tensor(m["batch_losses"]),
                       torch.tensor(rm["batch_losses"]))
    assert len(m["per_output"]) == len(rm["per_output"]) == ROUNDS
    for a, b in zip(m["per_output"], rm["per_output"]):
        assert np.array_equal(a, b)
    assert _log_lines(got["out"]) == _log_lines(ref["out"])
    assert _log_lines(got["out"])
    for n in ref["ckpts"]:
        _same_ckpt(got["ckpts"][n], ref["ckpts"][n], f"checkpoint {n}")
    assert got["sd"].keys() == ref["sd"].keys()
    for k in ref["sd"]:
        assert torch.equal(got["sd"][k], ref["sd"][k]), k


@pytest.fixture(scope="module")
def pipe_runs(simd, tmp_path_factory):
    """pipe_runs(case, arm): the pool pipe's run, computed once."""
    cache = {}

    def get(cname, arm):
        if (cname, arm) not in cache:
            mp = pytest.MonkeyPatch()
            try:
                tmp = str(tmp_path_factory.mktemp(f"{cname}_{arm}"))
                cache[cname, arm] = _train(mp, tmp, CASES[cname], arm, shm=False)
            finally:
                mp.undo()
        return cache[cname, arm]
    return get


@pytest.mark.parametrize("case_name,arm", [
    (c, a) for c in CASES for a in CASE_ARMS[c]])
@pytest.mark.slow
def test_shm_bitwise_equal_to_pipe(monkeypatch, tmp_path, pipe_runs,
                                   case_name, arm):
    ref = pipe_runs(case_name, REF_ARM.get(arm, arm))
    got = _train(monkeypatch, str(tmp_path), CASES[case_name], arm, shm=True)
    out = got["out"]
    assert ref["metrics"]["c_collect_transport"] == "pipe" and PIPE in ref["out"]
    assert not ref["made"]
    assert got["metrics"]["c_collect_transport"] == "shm"
    assert SHM_ON in out, out
    slots = 2 if CASES[case_name].get("pipeline_collect") else 1
    assert f"; {slots} in flight, a slab per worker;" in out, out
    assert len(got["made"]) >= slots, got["made"]
    assert not _left_behind(got["made"]), "a shared-memory arena was left behind"
    assert got["metrics"].get("shm_overflows", 0) == 0
    if ARMS[arm][0].get("BG_TRAIN_CUDA_GRAPH") == "1":
        assert "first replay bitwise equal to the eager step" in out, out
    if arm == "graph_phases":
        ph = got["metrics"]["round_phases"]
        assert len(ph) == ROUNDS
        assert all(p["assemble"] < 1.0 and p["rows"] > 0 for p in ph)
        lines = re.findall(r"^  phases round (\d+): collect .* h2d .* steps .* "
                           r"\| rows (\d+) ", out, re.M)
        assert [int(r) for r, _ in lines] == list(range(1, ROUNDS + 1)), out
    _assert_bitwise(got, ref)


@pytest.mark.slow
def test_slab_overflow_falls_back_bitwise(monkeypatch, tmp_path, pipe_runs):
    """Slabs far too small: every worker of round 1 and most later ones
    replay into their own buffer and return through the pipe; the pool is
    still the same."""
    ref = pipe_runs("0ply_pipelined", "graph")
    monkeypatch.setattr(row_transport, "FIRST_ROWS_PER_EPISODE", 1.0)
    monkeypatch.setattr(row_transport, "MARGIN", 1.0)
    monkeypatch.setattr(row_transport, "PAD", 0)
    got = _train(monkeypatch, str(tmp_path), CASES["0ply_pipelined"], "graph",
                 shm=True)
    out = got["out"]
    assert "outgrew their slab in round 1 " not in out, out   # first-guess sizing: quiet
    assert got["metrics"]["shm_overflows"] >= WORKERS
    assert not _left_behind(got["made"])
    _assert_bitwise(got, ref)


@pytest.mark.slow
def test_dump_under_shm_equals_pipe(monkeypatch, tmp_path, simd):
    """BG_COLLECT_DUMP with the direct device copy: the dumped pools
    (concatenated from the slab views) and the weights equal the pipe's."""
    runs = {}
    for shm in (True, False):
        d = tmp_path / ("dump_shm" if shm else "dump_pipe")
        runs[shm] = _train(monkeypatch, str(tmp_path), CASES["0ply_sync"],
                           "graph", shm=shm, env_extra={"BG_COLLECT_DUMP": str(d)})
        runs[shm]["dump"] = sorted(glob.glob(str(d / "round_*.npz")))
    assert len(runs[True]["dump"]) == len(runs[False]["dump"]) == ROUNDS
    for fa, fb in zip(runs[True]["dump"], runs[False]["dump"]):
        a, b = np.load(fa), np.load(fb)
        assert a.files == b.files
        for k in a.files:
            assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), (fa, k)
        assert a["tgt"].shape == (len(a["enc"]), 5)
        assert list(a["worker_rows"]) and min(a["worker_rows"]) > 0
    _assert_bitwise(runs[True], runs[False])


# ── where the transport does not apply ───────────────────────────────

def _tiny_train(workers, c_collect, metrics=None):
    TP.train(ProbNetwork(hidden_sizes=[8]), num_episodes=4, lr=1e-9,
             end_lr=None, episodes_per_round=4, batch_size=16,
             workers=workers, device="cpu", oneply=False, pw_weight=1.0,
             use_mse=False, seed=1, log_every=0, save_path=None,
             save_every=0, warmup_cycles=0, c_collect=c_collect,
             c_collect_backend=BACKEND, metrics_out=metrics)


@pytest.mark.parametrize("c_collect,workers,needle", [
    (False, 2, "BG_COLLECT_SHM=1 ignored (the Python collector); rows go "
               "through the pool pipe."),
    (True, 1, "BG_COLLECT_SHM=1 ignored (--workers 1 collects in the "
              "master); rows stay in-process."),
])
def test_shm_ignored_with_a_note(simd, monkeypatch, capsys, c_collect,
                                 workers, needle):
    monkeypatch.setenv(row_transport.ENV, "1")
    m = {}
    _tiny_train(workers, c_collect, m)
    out = capsys.readouterr().out
    assert needle in out, out
    assert m.get("c_collect_transport") == ("in-process" if c_collect else None)
    if c_collect:
        assert "C collection rows: in-process (one collector, no transport)" in out


@pytest.mark.parametrize("c_collect,workers", [(False, 2), (True, 1)])
def test_shm_default_is_quiet_where_it_does_not_apply(simd, monkeypatch, capsys,
                                                      c_collect, workers):
    monkeypatch.delenv(row_transport.ENV, raising=False)
    _tiny_train(workers, c_collect)
    assert row_transport.ENV not in capsys.readouterr().out


def test_shm_off_uses_the_pipe_and_a_bad_value_fails_early(simd, monkeypatch,
                                                           capsys):
    monkeypatch.setenv(row_transport.ENV, "0")
    m = {}
    _tiny_train(2, True, m)
    out = capsys.readouterr().out
    assert m["c_collect_transport"] == "pipe" and PIPE in out, out
    monkeypatch.setenv(row_transport.ENV, "yes")
    with pytest.raises(ValueError, match="BG_COLLECT_SHM='yes'"):
        _tiny_train(2, True)
    # Before the C collector's probe export and the pool.
    assert "C collection: ON" not in capsys.readouterr().out


def test_explicit_shm_on_cpu_device(simd, monkeypatch, capsys):
    """BG_COLLECT_SHM=1 with the C collector and a pool on a cpu device:
    the explicit label, a concatenate from the slab views."""
    monkeypatch.setenv(row_transport.ENV, "1")
    m = {}
    _tiny_train(2, True, m)
    out = capsys.readouterr().out
    assert m["c_collect_transport"] == "shm" and m.get("shm_overflows", 0) == 0
    assert ("C collection rows: shared-memory arenas (BG_COLLECT_SHM=1; 1 in "
            "flight, a slab per worker;") in out, out


# ── the CLI: process exit and a dead worker ──────────────────────────

TRACKER_NOISE = ("resource_tracker", "leaked shared_memory", "Traceback")


def _cli(simd, tmp_path, episodes):
    return [sys.executable, "-u", "train_prob5.py", "--resume", PROD_PROB5,
            "--num-episodes", str(episodes), "--episodes-per-round", "80",
            "--workers", str(WORKERS), "--pipeline-collect", "--device", "cuda",
            "--lr", "1e-9", "--c-collect-backend", simd,
            "--save", str(tmp_path / "p.pt")]


def _arenas(out):
    return re.findall(rf"{row_transport.ENV}: arena \d/\d (\S+):", out)


@pytest.mark.slow
def test_no_resource_tracker_warning_at_exit(simd, tmp_path):
    """The CLI defaults (no collector flags or env): the C collector with the
    shared-memory transport. Workers attach each arena by name, which on
    Python 3.12 registers it with the resource tracker they share with
    the master; the master's unlink unregisters it once. Nothing may be
    reported at exit (stderr in full, the tracker's own included)."""
    env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    p = subprocess.run(_cli(simd, tmp_path, 800), cwd=REPO, env=env, text=True,
                       capture_output=True, timeout=300)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert SHM_ON in p.stdout, p.stdout
    assert "C collection: ON (the default" in p.stdout, p.stdout
    noise = [ln for ln in p.stderr.splitlines()
             if any(k in ln for k in TRACKER_NOISE)]
    assert not noise, p.stderr
    made = _arenas(p.stdout)
    assert len(made) == 2 and not _left_behind(made)
    assert "Round   10/10" in p.stdout and (tmp_path / "p.pt").exists()


@pytest.mark.slow
def test_shm_recovers_from_a_dead_worker(simd, monkeypatch, tmp_path):
    """A worker SIGKILLed mid-run (maybe mid-slab): the pool and the arenas
    are rebuilt, the run finishes, nothing is left in /dev/shm and the
    tracker reports nothing."""
    from test_pool_guard import _run_and_kill_a_worker
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    rc, out = _run_and_kill_a_worker(_cli(simd, tmp_path, 960))
    assert rc == 0, out[-3000:]
    assert SHM_ON in out
    assert "rebuilding the collection pool" in out
    made = _arenas(out)
    assert len(made) >= 4, out          # the restart recreated them
    assert "Round   12/12" in out
    assert (tmp_path / "p.pt").exists()
    assert not _left_behind(made)
    # stdout and stderr merged, the tracker's exit report included
    assert not [ln for ln in out.splitlines()
                if any(k in ln for k in TRACKER_NOISE)], out
