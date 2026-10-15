"""--c-collect in train_prob5.train.

* The pool a round trains on: 2 rounds of 200 episodes on 2 workers, prob5
  0-ply (pipelined, with --save-every) and 1-ply (synchronous), prod 4L
  prob5 net, the same seed through the Python collector and through the C
  one, compared from the trainer's BG_COLLECT_DUMP: per-worker row counts
  and seeds equal, encodings bit-exact, each of the 5 target components
  within 1e-5 (0-ply) / 1e-4 (1-ply); a 1-ply row out of bound must be a
  near-tie target (tests/test_c_collect.prob5_tie_target: another
  near-tied reply's vector at some roll).
* --workers 1: the master collects the round in C on a seed drawn from
  the run RNG; its pool equals _collect_worker's on that seed.
* The precondition (a non-perspective196 encoder) prints its NOTE and
  keeps the Python collector; --vectorized-collect or --gpu-eval-server
  with --c-collect is an error; --oneply --device cuda does not auto-route
  away from the C collector; the private weights dir is removed.
* The CLI default: the C collector unless --no-c-collect or an
  explicit --vectorized-collect / --gpu-eval-server, each with its NOTE.
* A worker killed mid-run: the pool restart re-exports the round and the
  run finishes (tests/test_pool_guard.py's harness).

The simd backend is built as in tests/test_c_api.py (CAPI_BUILD=0 skips
the staleness rebuild). ~1 min, most of it the Python 1-ply arm.
"""
import argparse
import glob
import os
import random
import re
import shutil
import sys

import numpy as np
import pytest
import torch

from test_c_api import _build, PROD_PROB5  # noqa: E402
from test_c_collect import prob5_tie_target  # noqa: E402

import train_prob5 as TP  # noqa: E402
from model import ProbNetwork  # noqa: E402

BACKEND = "simd"
TOL0, TOL1 = 1e-5, 1e-4      # tests/test_c_api.py (prob5 targets are in [0, 1])


@pytest.fixture(scope="module")
def simd():
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    ok, log = _build(BACKEND)
    if not ok:
        pytest.skip(f"{BACKEND} backend not built:\n{log[-500:]}")
    return BACKEND


def _train(net, dump=None, monkeypatch=None, **kw):
    """train_prob5.train at a frozen lr, cpu, logging every round."""
    if dump is not None:
        monkeypatch.setenv("BG_COLLECT_DUMP", str(dump))
    args = dict(num_episodes=2, lr=1e-9, end_lr=None, episodes_per_round=2,
                batch_size=256, workers=1, device="cpu", oneply=False,
                pw_weight=1.0, use_mse=False, seed=1, log_every=1,
                save_path=None, save_every=0, warmup_cycles=0)
    args.update(kw)
    TP.train(net, **args)


def _load_dump(d):
    """[(enc, tgt, worker_rows, worker_seeds)] per round, in round order."""
    out = []
    for f in sorted(glob.glob(os.path.join(str(d), "round_*.npz"))):
        z = np.load(f)
        out.append((z["enc"], z["tgt"], z["worker_rows"], z["worker_seeds"]))
    return out


def _prod():
    return ProbNetwork.load(PROD_PROB5)


# ── the pool: C vs Python, same seed ──────────────────────────────────

@pytest.mark.parametrize("oneply", [False, True], ids=["0ply", "1ply"])
@pytest.mark.slow
def test_c_pool_matches_python(simd, tmp_path, monkeypatch, capsys, oneply):
    pools = {}
    for arm in ("python", "c"):
        kw = dict(num_episodes=400, episodes_per_round=200, workers=2,
                  seed=20261010, oneply=oneply, c_collect=arm == "c",
                  c_collect_backend=simd)
        if not oneply:
            kw.update(pipeline_collect=True,
                      save_path=str(tmp_path / f"{arm}.pt"), save_every=200)
        _train(_prod(), tmp_path / arm, monkeypatch, **kw)
        out = capsys.readouterr().out
        assert ("C collection: ON" in out) is (arm == "c"), out
        pools[arm] = _load_dump(tmp_path / arm)
        if not oneply:
            assert (tmp_path / f"{arm}_ep200.pt").exists()
            assert (tmp_path / f"{arm}_ep400.pt").exists()
    tol = TOL1 if oneply else TOL0
    assert len(pools["python"]) == len(pools["c"]) == 2
    for r, ((ep, tp, wp, sp), (ec, tc, wc, sc)) in enumerate(
            zip(pools["python"], pools["c"]), 1):
        assert list(wp) == list(wc) and len(wp) == 2 and min(wp) > 0
        assert list(sp) == list(sc) and len(sp) == 2
        assert ec.dtype == np.float32 and ec.shape == ep.shape
        assert tc.dtype == np.float32 and tc.shape == tp.shape == (len(ep), 5)
        assert np.array_equal(ec, ep), f"round {r}: encodings differ"
        d = np.abs(tc.astype(np.float64) - tp)
        out = np.flatnonzero(d.max(axis=1) > tol)
        ties = [i for i in out if oneply and prob5_tie_target(PROD_PROB5, ep[i], tc[i], tol)
                is not None]
        ok = np.ones(len(d), dtype=bool)
        ok[ties] = False
        print(f"\n[p6 trainer] {'1' if oneply else '0'}-ply round {r}: {len(tp)} rows, "
              f"workers {list(wp)}, target max |d| {d[ok].max():.3g}, mean {d[ok].mean():.3g}; "
              f"{len(ties)} near-tie 1-ply targets"
              + (f" (|d| up to {d[ties].max():.3g})" if ties else ""))
        assert len(ties) == len(out), f"round {r}: rows {sorted(set(out) - set(ties))} out of bound"


def test_single_process_c_matches_worker_seed(simd, tmp_path, monkeypatch, capsys):
    """workers=1: one round in the master, on rng.randint(0, 2**31 - 1) of
    random.Random(seed), equals the Python worker loop on that seed."""
    seed, n = 77, 12
    _train(_prod(), tmp_path, monkeypatch, num_episodes=n, episodes_per_round=n,
           workers=1, seed=seed, c_collect=True, c_collect_backend=simd)
    assert "C collection: ON" in capsys.readouterr().out
    (enc_c, tgt_c, wr, ws), = _load_dump(tmp_path)
    wseed = random.Random(seed).randint(0, 2**31 - 1)
    assert list(ws) == [wseed]
    net = _prod()
    sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
    enc_p, tgt_p, _ = TP._collect_worker((sd, net.hidden_sizes, net.input_size,
                                          net.activation, net.encoder_name,
                                          net.raw_logits, n, wseed, False, "cpu",
                                          TP.DEFAULT_MAX_EPISODE_PLIES))
    assert list(wr) == [len(tgt_p)]
    assert np.array_equal(enc_c, enc_p)
    assert np.abs(tgt_c.astype(np.float64) - tgt_p).max() <= TOL0


def test_c_worker_result_shapes(simd, tmp_path):
    """The C worker returns what _collect_worker returns: (n, 196) and
    (n, 5) float32 and the discards; zero episodes give (0, 196) / (0, 5)."""
    from export_weights import export_live
    path = str(tmp_path / "w.bin")
    export_live(_prod(), path)
    w = TP._CWeights(path, simd)
    enc, tgt, nd = TP._collect_worker((w, None, 196, None, None, None, 3, 5, True,
                                       "cpu", TP.DEFAULT_MAX_EPISODE_PLIES))
    assert enc.dtype == tgt.dtype == np.float32
    assert enc.shape == (len(enc), 196) and tgt.shape == (len(enc), 5) and len(enc) > 0
    assert nd == 0
    enc, tgt, nd = TP._c_collect(w, 0, 5, False, TP.DEFAULT_MAX_EPISODE_PLIES)
    assert enc.shape == (0, 196) and tgt.shape == (0, 5)


# ── preconditions, errors, routing ────────────────────────────────────

def test_encoder_precondition_falls_back_with_note(simd, capsys, monkeypatch):
    """A net whose encoder the C collector does not write (here a
    perspective196 copy under another name) keeps the Python collector,
    with one NOTE."""
    import encoding
    monkeypatch.setitem(encoding.ENCODERS, "p196_copy", encoding.Perspective196Encoder)
    net = ProbNetwork(hidden_sizes=[8], encoder_name="p196_copy")
    assert TP.c_collect_fallbacks(net) and not TP.c_collect_fallbacks(_prod())
    _train(net, num_episodes=2, episodes_per_round=2, workers=1, c_collect=True,
           c_collect_backend=simd)
    out = capsys.readouterr().out
    notes = [ln for ln in out.splitlines() if ln.strip().startswith("NOTE: --c-collect:")]
    assert len(notes) == 1 and "encoder 'p196_copy'" in notes[0], out
    assert "falling back to the Python collector" in notes[0]
    assert "C collection: ON" not in out
    assert "Round    1/1" in out


@pytest.mark.parametrize("kw", [dict(vectorized_collect=4, workers=2),
                                dict(gpu_eval_server=True, workers=2)],
                         ids=["vectorized", "gpu_eval_server"])
def test_c_collect_with_another_collector_is_an_error(kw):
    with pytest.raises(SystemExit, match="mutually exclusive"):
        _train(ProbNetwork(hidden_sizes=[8]), oneply=True, c_collect=True, **kw)


def _args(**kw):
    """The CLI namespace; no --c-collect / --no-c-collect = the default."""
    d = dict(oneply=True, device="cuda", vectorized_collect=0,
             no_vectorized_collect=False, gpu_eval_server=False,
             pipeline_collect=False, c_collect=False, no_c_collect=False,
             episodes_per_round=4000, workers=48)
    d.update(kw)
    return argparse.Namespace(**d)


def test_c_collect_is_not_auto_routed():
    """--c-collect (or the default) --oneply --device cuda stays on the C
    collector; with --no-c-collect the same args auto-route to the
    vectorized collector."""
    for kw, who in (({"c_collect": True}, "--c-collect"),
                    ({}, "the C collector (default)")):
        a = _args(**kw)
        notes = TP.resolve_cuda_oneply_collector(a)
        assert a.c_collect is True and a.c_collect_auto is (not kw)
        assert a.vectorized_collect == 0 and not getattr(a, "vectorized_auto", False)
        assert len(notes) == 1 and notes[0].startswith(f"NOTE: {who}:")
        assert "keeps the C collector" in notes[0]
    b = _args(no_c_collect=True)
    TP.resolve_cuda_oneply_collector(b)
    assert b.c_collect is False and b.vectorized_collect > 0 and b.vectorized_auto
    for kw in ({"device": "cpu"}, {"oneply": False}):
        for c in ({"c_collect": True}, {}):
            a = _args(**c, **kw)
            assert TP.resolve_cuda_oneply_collector(a) == [] and a.vectorized_collect == 0
            assert a.c_collect is True


@pytest.mark.parametrize("kw,flag", [(dict(vectorized_collect=8), "--vectorized-collect"),
                                     (dict(gpu_eval_server=True), "--gpu-eval-server")],
                         ids=["vectorized", "gpu_eval_server"])
@pytest.mark.parametrize("dev", ["cuda", "cpu"])
def test_explicit_collector_wins_over_the_default(kw, flag, dev):
    a = _args(device=dev, **kw)
    notes = TP.resolve_cuda_oneply_collector(a)
    assert a.c_collect is False and a.c_collect_auto is False
    assert notes == [f"NOTE: {flag}: replaces the default C collector "
                     f"(--c-collect with it is an error)."]
    assert a.vectorized_collect == kw.get("vectorized_collect", 0)


def test_no_vectorized_collect_keeps_the_default_c_collector():
    a = _args(no_vectorized_collect=True)
    notes = TP.resolve_cuda_oneply_collector(a)
    assert a.c_collect is True and a.vectorized_collect == 0
    assert len(notes) == 1 and "add --no-c-collect for the Python CPU" in notes[0]
    TP.reject_cuda_oneply(_args(oneply=False))      # 0-ply: never refused
    # --no-c-collect: the per-worker refusal.
    b = _args(no_vectorized_collect=True, no_c_collect=True)
    TP.resolve_cuda_oneply_collector(b)
    assert b.c_collect is False and b.vectorized_collect == 0
    with pytest.raises(SystemExit, match="drop --no-c-collect"):
        TP.reject_cuda_oneply(b)


WHY = "encoder 'p196_copy' with 196 inputs (...)"


def test_default_ruled_out_falls_back_to_the_no_c_collect_routing():
    """A precondition miss under the default: a NOTE per reason, then the
    --no-c-collect routing (--oneply --device cuda -> vectorized, B auto)."""
    a = _args()
    notes = TP.resolve_cuda_oneply_collector(a, [WHY])
    assert a.c_collect is False and a.c_collect_auto is False
    assert a.vectorized_collect == TP.default_vectorized_batch(4000, 48)
    assert a.vectorized_auto is True
    assert len(notes) == 2
    assert notes[0] == (f"NOTE: --c-collect (default): {WHY}; falling back to "
                        f"the vectorized GPU collector (the --no-c-collect "
                        f"routing).")
    assert notes[1].startswith("AUTO: --oneply --device cuda -> vectorized")
    # --pipeline-collect wins over the auto-route: Python.
    b = _args(pipeline_collect=True)
    notes = TP.resolve_cuda_oneply_collector(b, [WHY])
    assert b.c_collect is False and b.vectorized_collect == 0
    assert "falling back to the Python collector" in notes[0] and "wins" in notes[1]
    # --no-vectorized-collect: the per-worker refusal.
    c = _args(no_vectorized_collect=True)
    TP.resolve_cuda_oneply_collector(c, [WHY])
    assert c.c_collect is False and c.vectorized_collect == 0
    with pytest.raises(SystemExit, match="per-worker"):
        TP.reject_cuda_oneply(c)
    # 0-ply: Python, one NOTE.
    d = _args(oneply=False)
    assert len(TP.resolve_cuda_oneply_collector(d, [WHY])) == 1
    assert d.c_collect is False and d.vectorized_collect == 0
    # An explicit --c-collect is not rerouted (train(): NOTE + Python).
    e = _args(c_collect=True)
    TP.resolve_cuda_oneply_collector(e, [WHY])
    assert e.c_collect is True and e.vectorized_collect == 0


@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit"])
def test_main_routes_a_ruled_out_default(monkeypatch, capsys, explicit):
    """train_prob5.main builds the net, then resolves: an encoder the C
    collector does not write sends the default to the vectorized
    collector (both NOTEs); --c-collect keeps the C request for train()
    (its NOTE + Python). train() itself is stubbed."""
    got = {}
    monkeypatch.setattr(TP, "c_collect_fallbacks", lambda net: [WHY])
    monkeypatch.setattr(TP, "train", lambda net, **kw: got.update(kw))
    monkeypatch.setattr(sys, "argv", [
        "train_prob5.py", "--hidden", "8", "--num-episodes", "2",
        "--episodes-per-round", "2", "--workers", "2", "--oneply",
        "--device", "cuda"] + (["--c-collect"] if explicit else []))
    TP.main()
    out = capsys.readouterr().out
    if explicit:
        assert got["c_collect"] is True and got["vectorized_collect"] == 0
        assert "NOTE: --c-collect: --oneply --device cuda keeps the C collector" in out
        assert "AUTO:" not in out
    else:
        assert got["c_collect"] is False and got["c_collect_auto"] is False
        assert got["vectorized_collect"] == 1 and got["vectorized_auto"] is True
        assert f"NOTE: --c-collect (default): {WHY}; falling back to the vectorized" in out
        assert "AUTO: --oneply --device cuda -> vectorized GPU collector" in out


def test_c_collect_and_no_c_collect_is_an_error():
    with pytest.raises(SystemExit, match="mutually exclusive"):
        TP.resolve_cuda_oneply_collector(_args(c_collect=True, no_c_collect=True))


@pytest.mark.parametrize("kw", [dict(vectorized_collect=8), dict(gpu_eval_server=True)],
                         ids=["vectorized", "gpu_eval_server"])
def test_cli_conflicts_are_errors(kw):
    for dev in ("cuda", "cpu"):
        with pytest.raises(SystemExit, match="mutually exclusive"):
            TP.resolve_cuda_oneply_collector(_args(c_collect=True, device=dev, **kw))


@pytest.mark.cuda
@pytest.mark.slow
def test_cli_cuda_oneply_c_collect_runs(simd, tmp_path):
    """The CLI path: --oneply --device cuda --c-collect is neither
    auto-routed nor refused by reject_cuda_oneply, collects in C, and
    removes its private weights dir."""
    import subprocess
    r = subprocess.run(
        [sys.executable, "-u", "train_prob5.py", "--resume", PROD_PROB5,
         "--num-episodes", "8", "--episodes-per-round", "4", "--workers", "2",
         "--oneply", "--device", "cuda", "--lr", "1e-9", "--c-collect",
         "--c-collect-backend", simd, "--save", str(tmp_path / "m.pt")],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, OMP_NUM_THREADS="1"))
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "keeps the C collector" in out and "C collection: ON" in out
    assert "AUTO:" not in out and "vectorized collection: ON" not in out
    m = re.search(r"weights exported once per round to (\S+)\)", out)
    assert m and not os.path.exists(m.group(1))
    assert (tmp_path / "m.pt").exists() and "Round    2/2" in out


# ── pool restart ──────────────────────────────────────────────────────

@pytest.mark.slow
def test_c_collect_recovers_from_a_dead_worker(simd, tmp_path):
    from test_pool_guard import _run_and_kill_a_worker
    rc, out = _run_and_kill_a_worker([
        sys.executable, "-u", "train_prob5.py", "--resume", PROD_PROB5,
        "--num-episodes", "960", "--episodes-per-round", "80", "--workers", "4",
        "--pipeline-collect", "--device", "cpu", "--lr", "1e-9", "--c-collect",
        "--c-collect-backend", simd, "--save", str(tmp_path / "c.pt")])
    assert rc == 0, out[-3000:]
    assert "C collection: ON" in out
    assert "rebuilding the collection pool" in out
    assert (tmp_path / "c.pt").exists()
    assert "Round   12/12" in out
