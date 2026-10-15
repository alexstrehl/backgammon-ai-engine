"""--c-collect in Trainer.train.

* The pool a round trains on: 2 rounds of 200 episodes on 2 workers,
  cubeful money 0-ply (pipelined, with --save-every) and 1-ply
  (synchronous), prod 4L net, the same seed through the Python collector
  and through the C one, compared from the trainer's BG_COLLECT_DUMP
  (experiments/c_collect/round_parity.py): per-worker row counts equal,
  encodings bit-exact, targets within 1e-5 (0-ply) / 1e-4 (1-ply)
  (test_c_api.close).
* --workers 1: the master collects one worker's share in C on a seed
  drawn from the run RNG; its pool equals _collect_worker's on that seed.
* Every precondition prints its NOTE and keeps the Python collector;
  --vectorized-collect with c_collect is an error.
* The library: a missing libpuretd_<backend>.so is built on demand;
  one that cannot be built is a NOTE and the Python collector under the
  default (c_collect_auto) and an error under an explicit --c-collect;
  train_params["c_collect"] says what collected.
* A worker killed mid-run: the pool restart re-exports the round and the
  run finishes (tests/test_pool_guard.py's harness), on the pool pipe
  (BG_COLLECT_SHM=0; the shared-memory restart is
  tests/test_row_transport_bitwise.py's).

The simd backend is built as in tests/test_c_api.py (CAPI_BUILD=0 skips
the staleness rebuild). ~1.5 min, most of it the Python 1-ply arm.
"""
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest

from test_c_api import _build, PROD_CUBEFUL, PROD_EQUITY, REPO  # noqa: E402

import trainer as T  # noqa: E402
from model import TDNetwork, load_model  # noqa: E402
from modes import CubefulMoneyMode, CubelessMoneyMode, DMPMode  # noqa: E402
from td_agent import TDAgent  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "experiments", "c_collect"))
import round_parity as RP  # noqa: E402

BACKEND = "simd"


@pytest.fixture(scope="module")
def simd():
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    ok, log = _build(BACKEND)
    if not ok:
        pytest.skip(f"{BACKEND} backend not built:\n{log[-500:]}")
    return BACKEND


def _train(net, mode, dump=None, monkeypatch=None, **kw):
    """Trainer.train at a frozen lr; returns (metrics_out, losses)."""
    if dump is not None:
        monkeypatch.setenv("BG_COLLECT_DUMP", str(dump))
    m = {}
    tr = T.Trainer(TDAgent(net, device="cpu"), lr=1e-9)
    losses = tr.train(mode, metrics_out=m, log_every=0, **kw)
    return m, losses


# ── the pool: C vs Python, same seed ──────────────────────────────────

@pytest.mark.parametrize("oneply", [False, True], ids=["0ply", "1ply"])
@pytest.mark.slow
def test_c_pool_matches_python(simd, tmp_path, monkeypatch, oneply):
    pools = {}
    for arm in ("python", "c"):
        d = tmp_path / arm
        kw = dict(num_episodes=400, episodes_per_round=200, workers=2,
                  seed=20261010, oneply=oneply, c_collect=arm == "c",
                  c_collect_backend=simd)
        if not oneply:
            kw.update(pipeline_collect=True,
                      save_path=str(tmp_path / f"{arm}.pt"), save_every=200)
        m, _ = _train(load_model(PROD_CUBEFUL), CubefulMoneyMode(), d,
                      monkeypatch, **kw)
        assert m["c_collect"] is (arm == "c")
        pools[arm] = RP.load_dump(str(d))
        if not oneply:
            assert (tmp_path / f"{arm}_ep200.pt").exists()
            assert (tmp_path / f"{arm}_ep400.pt").exists()
    lines, ok, parted = RP.compare(pools["python"], pools["c"],
                                   RP.TOL1 if oneply else RP.TOL0)
    print("\n" + "\n".join(lines))
    assert ok and not parted, "\n".join(lines)
    assert len(pools["c"]) == 2
    for (ep, tp, wp, sp), (ec, tc, wc, sc) in zip(pools["python"], pools["c"]):
        assert list(wp) == list(wc) and len(wp) == 2 and min(wp) > 0
        assert list(sp) == list(sc) and len(sp) == 2
        assert ec.dtype == np.float32 and ec.shape == ep.shape
        assert np.array_equal(ec, ep)


def test_single_process_c_matches_worker_seed(simd, tmp_path, monkeypatch):
    """workers=1: one round in the master, on rng.randint(0, 2**31 - 1) of
    random.Random(seed), equals the Python worker loop on that seed."""
    seed, n = 77, 12
    m, _ = _train(load_model(PROD_EQUITY), CubelessMoneyMode(), tmp_path,
                  monkeypatch, num_episodes=n, episodes_per_round=n, workers=1,
                  seed=seed, c_collect=True, c_collect_backend=simd)
    assert m["c_collect"] is True
    (enc_c, tgt_c, wr, ws), = RP.load_dump(str(tmp_path))
    assert list(ws) == [random.Random(seed).randint(0, 2**31 - 1)]
    net = load_model(PROD_EQUITY)
    sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
    args = (sd, list(net.hidden_sizes), net.input_size, net.output_mode,
            net.encoder_name, net.activation, CubelessMoneyMode(), n,
            random.Random(seed).randint(0, 2**31 - 1), False, False, 0.0,
            False, False, T.DEFAULT_MAX_EPISODE_PLIES)
    enc_p, tgt_p, nd = T._collect_worker(args)
    assert list(wr) == [len(tgt_p)] and nd == 0
    assert np.array_equal(enc_c, enc_p)
    assert not RP.out_of_bound(tgt_c, tgt_p, RP.TOL0).any()


def test_empty_c_result_shape(simd, tmp_path):
    """A worker whose every episode is discarded returns (0, input_size)
    (the Python worker returns (0, 0)); the master skips both."""
    from export_weights import export_live
    net = load_model(PROD_CUBEFUL)
    path = str(tmp_path / "w.bin")
    export_live(net, path)
    enc, tgt, nd = T._c_collect(T._CWeights(path, simd), CubefulMoneyMode(),
                                3, 5, False, 2)
    assert enc.shape == (0, 200) and tgt.shape == (0,) and nd == 3


# ── preconditions: NOTE + Python collector ────────────────────────────

def _tiny(output_mode="equity", encoder="cubeful_perspective196"):
    n_in = 200 if encoder == "cubeful_perspective196" else 196
    return TDNetwork(hidden_sizes=[8], input_size=n_in, output_mode=output_mode,
                     encoder_name=encoder)


FALLBACKS = {
    "boltzmann": (dict(boltzmann_temp=0.5), "--boltzmann-temp 0.5"),
    "oneply_acting": (dict(oneply=True, oneply_acting=True), "--oneply-acting"),
    "bf16": (dict(bf16_collect=True), "--bf16-collect"),
    "c_engine": (dict(c_engine=True), "--c-engine"),
    "dmp_equity": (dict(net=("equity", "perspective196"), mode=DMPMode()),
                   "dmp with an equity-output net"),
}


@pytest.mark.parametrize("case", sorted(FALLBACKS))
def test_precondition_falls_back_with_note(simd, capsys, case):
    kw, needle = FALLBACKS[case]
    kw = dict(kw)
    net = _tiny(*kw.pop("net", ("equity", "cubeful_perspective196")))
    mode = kw.pop("mode", CubefulMoneyMode())
    m, _ = _train(net, mode, num_episodes=2, episodes_per_round=2, workers=1,
                  seed=1, max_episode_plies=20, c_collect=True,
                  c_collect_backend=simd, **kw)
    out = capsys.readouterr().out
    notes = [ln for ln in out.splitlines() if ln.strip().startswith("NOTE: --c-collect:")]
    assert len(notes) == 1 and needle in notes[0], out
    assert "falling back to the Python collector" in notes[0]
    assert "C collection: ON" not in out
    assert m["c_collect"] is False


def test_supported_configs_need_no_fallback():
    """The three modes with their prod-kind tiny nets pass every check."""
    for net, mode in ((_tiny("probability", "perspective196"), DMPMode()),
                      (_tiny("equity", "perspective196"), CubelessMoneyMode()),
                      (_tiny(), CubefulMoneyMode(jacoby=False))):
        assert T.c_collect_fallbacks(net, mode) == []
    # A cubeless net in the cubeful mode (or the reverse) has no kind.
    assert T.c_collect_fallbacks(_tiny("equity", "perspective196"), CubefulMoneyMode())
    assert T.c_collect_fallbacks(_tiny(), CubelessMoneyMode())


def test_vectorized_and_c_collect_is_an_error():
    tr = T.Trainer(TDAgent(_tiny(), device="cpu"), lr=1e-9)
    with pytest.raises(ValueError, match="mutually exclusive"):
        tr.train(CubefulMoneyMode(), num_episodes=1, episodes_per_round=1,
                 oneply=True, workers=2, vectorized_collect=True, c_collect=True)


# ── the library: built on demand, or NOTE / error ─────────────────────

@pytest.mark.slow
def test_missing_library_is_built_on_demand(simd, tmp_path, monkeypatch, capsys):
    """An empty PURETD_LIBDIR: the trainer builds libpuretd_simd.so there
    (bash c_api/build.sh simd, quietly, once) and collects in C."""
    from c_api import puretd_py as P
    monkeypatch.setenv("PURETD_LIBDIR", str(tmp_path))
    params = {"c_collect": True}
    m, _ = _train(_tiny(), CubefulMoneyMode(), num_episodes=2,
                  episodes_per_round=2, workers=1, seed=1,
                  max_episode_plies=20, c_collect=True, c_collect_auto=True,
                  c_collect_backend=simd, train_params=params)
    out = capsys.readouterr().out
    so = tmp_path / f"libpuretd_{simd}.so"
    assert so.is_file() and P.ensure_lib(simd) == str(so)
    assert f"NOTE: built {so} on demand" in out, out
    assert "C collection: ON (the default, --no-c-collect" in out, out
    assert "[simd] compiling" not in out          # the build log is captured
    assert m["c_collect"] is True and params["c_collect"] is True


@pytest.fixture
def unbuildable(tmp_path, monkeypatch):
    """An empty PURETD_LIBDIR and a build.sh that fails."""
    from c_api import puretd_py as P
    libdir = tmp_path / "lib"
    libdir.mkdir()
    bad = tmp_path / "build.sh"
    bad.write_text("echo 'gcc: not found' >&2; exit 7\n")
    monkeypatch.setenv("PURETD_LIBDIR", str(libdir))
    monkeypatch.setattr(P, "BUILD_SH", str(bad))
    return libdir


def test_unbuildable_library_falls_back_under_the_default(simd, unbuildable,
                                                          capsys):
    params = {"c_collect": True}
    m, _ = _train(_tiny(), CubefulMoneyMode(), num_episodes=2,
                  episodes_per_round=2, workers=1, seed=1,
                  max_episode_plies=20, c_collect=True, c_collect_auto=True,
                  c_collect_backend=simd, train_params=params)
    out = capsys.readouterr().out
    notes = [ln for ln in out.splitlines() if "NOTE: --c-collect" in ln]
    assert len(notes) == 1, out
    assert notes[0].strip().startswith("NOTE: --c-collect (default):")
    assert "did not build it (exit 7: gcc: not found)" in notes[0]
    assert "falling back to the Python collector" in notes[0]
    assert "C collection: ON" not in out
    assert m["c_collect"] is False and params["c_collect"] is False


def test_unbuildable_library_is_an_error_when_explicit(simd, unbuildable):
    with pytest.raises(RuntimeError, match="--c-collect: .*did not build it"):
        _train(_tiny(), CubefulMoneyMode(), num_episodes=2,
               episodes_per_round=2, workers=1, seed=1, max_episode_plies=20,
               c_collect=True, c_collect_backend=simd)


def test_default_precondition_note_is_labelled(simd, capsys):
    """Under the default the precondition NOTE says it was the default
    that fell back; the provenance stamp follows."""
    params = {"c_collect": True}
    m, _ = _train(_tiny(), CubefulMoneyMode(), num_episodes=2,
                  episodes_per_round=2, workers=1, seed=1,
                  max_episode_plies=20, c_collect=True, c_collect_auto=True,
                  c_collect_backend=simd, boltzmann_temp=0.5,
                  train_params=params)
    out = capsys.readouterr().out
    assert "NOTE: --c-collect (default): --boltzmann-temp 0.5" in out, out
    assert m["c_collect"] is False and params["c_collect"] is False


# ── pool restart ──────────────────────────────────────────────────────

@pytest.mark.slow
def test_c_collect_recovers_from_a_dead_worker(simd, tmp_path, monkeypatch):
    from test_pool_guard import _run_and_kill_a_worker
    monkeypatch.setenv("BG_COLLECT_SHM", "0")
    rc, out = _run_and_kill_a_worker([
        sys.executable, "-u", "train_batch.py", "--game-mode", "cubeful-money",
        "--resume", PROD_CUBEFUL, "--num-episodes", "960",
        "--episodes-per-round", "80", "--workers", "4", "--pipeline-collect",
        "--device", "cpu", "--lr", "1e-9", "--c-collect",
        "--c-collect-backend", simd, "--save", str(tmp_path / "c.pt")])
    assert rc == 0, out[-3000:]
    assert "C collection: ON" in out
    assert "C collection rows: pickled through the pool pipe" in out
    assert "rebuilding the collection pool" in out
    assert (tmp_path / "c.pt").exists()
    assert "Round   12/12" in out
