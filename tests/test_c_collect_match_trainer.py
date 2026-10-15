"""--c-collect in Trainer.train / train_batch.py for match play.

Two engines run the same trainer-path tests:

* "double" (always): tests/c_match_double.py stands in for the C binding
  (Engine.set_met / collect_match) in the master and in every spawn
  worker. It plays what bgc_collect_match must reproduce -- the serial
  stream (rng_mode 0) or the vectorized per-episode streams (rng_mode 1)
  -- through the generator mirror on the round's BGNN export, torch
  forward, so a C-path round must equal the Python round BIT FOR BIT:
  worker split, the master's seed draws, row order, empties, accounting,
  the exported weights, the opts plumbing.
* "c" (needs the real collector: Engine.collect_match and the matchplay
  encoder table). The Python arm runs under BG_FAST_FORWARD=simd, the C
  arm on simd with BGC_SIMD_DELTA0=0 (one backend kernel, row-exact), so
  again bitwise.

Cells (3-point, the champion net (md5 dbe2579a), --met auto all scopes,
TRAINER_SCORE_CURRICULUM_P=0.0 as the leg, W=4, 16 episodes per round, 2
rounds, --lr 1e-9 unless stated):
* the round pool == the serial Python pool (rng serial) and == the
  vectorized worker at B=1 on the CPU driver (rng vectorized): per-worker
  row counts, seeds, encodings and targets equal;
* 2 trained rounds at a real LR (3.4e-6): final weights and losses bitwise
  equal to the Python round's (same device);
* --pipeline-collect + --save-every: pools, checkpoints and weights equal
  to the Python pipelined run; --workers 1: one seed drawn as a one-worker
  pool would, the pool equal to _collect_worker on it;
* --max-episode-plies: the vectorized discard semantics (counts and
  pools equal to the vectorized B=1 arm);
* BG_TRAIN_CUDA_GRAPH=1 after a C round (cuda): weights equal to the
  Python round's eager step;
* BG_COLLECT_DUMP contents; the per-round export files (0700 dir, one
  unique file per round, removed after its collection, dir removed);
* every precondition reason is an error naming it, before any process
  starts; the library missing / not buildable / without collect_match is
  an error before the pool; C_COLLECT_MATCH_ENCODERS is the C table's
  (needs the real collector); train_batch.py's CLI errors.
"""
import glob
import os
import random
import shutil
import stat
import subprocess
import sys

import numpy as np
import pytest
import torch

from c_api import puretd_py as P
import c_match_double as D
import trainer as T
from met import DEFAULT_MET, resolve_met_path
from model import TDNetwork, load_model
from modes import MatchplayMode
from td_agent import TDAgent

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(REPO, "best_models", "matchplay_1h64met_512_512_256_256.pt")
MET = resolve_met_path("auto")
BACKEND = "simd"
LEG_LR = 3.4e-6
ALL_SCOPES = frozenset(("cube", "terminal", "boundary"))
PLY_BOUND = 60      # per game: some 3pt matches discarded, some kept

HAVE_C = hasattr(P.Engine, "collect_match") and hasattr(P.Engine, "set_met")
needs_c = pytest.mark.skipif(not HAVE_C, reason="needs Engine.collect_match "
                             "(the C matchplay collector)")
pytestmark = pytest.mark.skipif(not os.path.exists(MET) or not os.path.exists(NET),
                                reason="needs the gnubg MET and the champion net")
ENGINES = ["double", pytest.param("c", marks=needs_c)]


@pytest.fixture(scope="module")
def simd():
    """libpuretd_simd.so / libbgt_simd.so built (test_c_api's staleness
    rebuild; CAPI_BUILD=0 skips it)."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    from test_c_api import _build
    ok, log = _build(BACKEND)
    if not ok:
        pytest.skip(f"{BACKEND} backend not built:\n{log[-500:]}")
    return BACKEND


@pytest.fixture(autouse=True)
def _leg_env(monkeypatch):
    """What train_batch.py --met auto sets (the encoder MET), the leg's
    curriculum (P = 0.0, in this process and in spawn workers), and no
    stray collector env."""
    monkeypatch.setenv("BG_MET_PATH", MET)
    monkeypatch.setenv("TRAINER_SCORE_CURRICULUM_P", "0.0")
    monkeypatch.setattr(T, "_SCORE_CURRICULUM_P", 0.0)
    for k in ("BG_FAST_FORWARD", "BG_COLLECT_DUMP", "BG_TRAIN_CUDA_GRAPH",
              "BG_POOL_ON_DEVICE", "TRAINER_CURRICULUM_FORCE",
              "TRAINER_MOVE_EXPLORATION_EPS", "BGC_SIMD_DELTA0"):
        monkeypatch.delenv(k, raising=False)


def _need_enabled_c():
    """The real collector's encoder table must serve the match encoders
    (collect set in the C encoder table)."""
    info = P.encoder_info(T.C_COLLECT_MATCH_ENCODERS[0], BACKEND)
    if not (info and info.get("collect")):
        pytest.skip("needs the enabled match collector (the C "
                    f"encoder table says collect 0: {info})")


def _engine_env(engine, monkeypatch):
    """The double: installed here and in every worker; both arms torch.
    The real collector: both arms on the simd kernel (plain forward:
    row-exact)."""
    if engine == "double":
        D.patch_trainer(monkeypatch)
    else:
        _need_enabled_c()
        monkeypatch.setenv("BG_FAST_FORWARD", "simd")
        monkeypatch.setenv("BGC_SIMD_DELTA0", "0")


def _train(*, c, monkeypatch, dump=None, device="cpu", lr=1e-9, workers=4,
           epr=16, rounds=2, seed=20261011, rng="serial", net=None, **kw):
    """One Trainer.train run from the champion net; (net, losses, metrics)."""
    if dump is not None:
        monkeypatch.setenv("BG_COLLECT_DUMP", str(dump))
    else:
        monkeypatch.delenv("BG_COLLECT_DUMP", raising=False)
    net = net if net is not None else load_model(NET)
    agent = TDAgent(net, device=device, met=T._worker_met(MET),
                    met_scopes=ALL_SCOPES)
    tr = T.Trainer(agent, lr=lr)
    m = {}
    kw.setdefault("oneply", True)
    losses = tr.train(
        MatchplayMode(kw.pop("match_length", 3)), num_episodes=epr * rounds,
        episodes_per_round=epr, workers=workers, seed=seed, met_path=MET,
        met_scopes=ALL_SCOPES, log_every=0, metrics_out=m, c_collect=c,
        c_collect_backend=BACKEND, c_collect_rng=rng, **kw)
    return net, losses, m


def _dump(d):
    return [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(str(d), "round_*.npz")))]


def _assert_pools_equal(py, c, rounds=2, workers=4, py_collector="serial"):
    assert len(py) == len(c) == rounds
    for i, (a, b) in enumerate(zip(py, c)):
        assert str(a["collector"]) == py_collector and str(b["collector"]) == "c"
        assert list(a["worker_seeds"]) == list(b["worker_seeds"]), i
        assert len(b["worker_seeds"]) == (workers if workers > 1 else 1)
        assert list(a["worker_rows"]) == list(b["worker_rows"]), i
        assert int(b["worker_rows"].sum()) == len(b["enc"]) > 0
        assert a["enc"].dtype == b["enc"].dtype == np.float32
        assert a["tgt"].dtype == b["tgt"].dtype == np.float32
        assert a["enc"].shape == b["enc"].shape, (i, a["enc"].shape, b["enc"].shape)
        assert np.array_equal(a["enc"], b["enc"]), f"round {i + 1}: encodings differ"
        assert np.array_equal(a["tgt"], b["tgt"]), (
            f"round {i + 1}: targets differ, max |d| "
            f"{np.abs(a['tgt'].astype(np.float64) - b['tgt']).max():.3g}")
        assert int(a["n_discarded"]) == int(b["n_discarded"])


def _state_bytes(net):
    return {k: v.detach().cpu().numpy().tobytes() for k, v in net.state_dict().items()}


def _master_seeds(seed, workers, rounds, epochs=1):
    """The run RNG's draws: per round one randint per worker (one when the
    master collects in C), then the shuffle seed per epoch."""
    r = random.Random(seed)
    out = []
    for _ in range(rounds):
        out.append([r.randint(0, 2**31 - 1) for _ in range(max(workers, 1))])
        for _ in range(epochs):
            r.randint(0, 2**31 - 1)
    return out


# ── the round pool ──────────────────────────────────────────────────

@pytest.mark.parametrize("engine", ENGINES)
def test_round_pool_equals_serial(simd, tmp_path, monkeypatch, engine):
    _engine_env(engine, monkeypatch)
    _, _, mp_ = _train(c=False, monkeypatch=monkeypatch, dump=tmp_path / "py")
    _, _, mc = _train(c=True, monkeypatch=monkeypatch, dump=tmp_path / "c")
    assert mp_["c_collect"] is False and mc["c_collect"] is True
    py, c = _dump(tmp_path / "py"), _dump(tmp_path / "c")
    _assert_pools_equal(py, c)
    assert [list(r["worker_seeds"]) for r in c] == _master_seeds(20261011, 4, 2)


@pytest.mark.parametrize("engine", ENGINES)
def test_round_pool_equals_vectorized_b1(simd, tmp_path, monkeypatch, engine):
    """rng vectorized: the per-episode streams of vectorized_matchplay_worker,
    driven at B=1 on the CPU driver (BG_FAST_FORWARD=simd with the real
    collector)."""
    _engine_env(engine, monkeypatch)
    _train(c=False, monkeypatch=monkeypatch, dump=tmp_path / "vec",
           vectorized_collect=True, vectorized_batch=1, vectorized_device="cpu")
    _train(c=True, monkeypatch=monkeypatch, dump=tmp_path / "c", rng="vectorized")
    _assert_pools_equal(_dump(tmp_path / "vec"), _dump(tmp_path / "c"),
                        py_collector="vectorized")


@pytest.mark.parametrize("engine", ENGINES)
def test_trained_weights_bitwise(simd, tmp_path, monkeypatch, engine):
    """2 rounds from the same parent at a real LR: the C round and the
    Python round train the same rows, so the weights (and every batch
    loss) are bitwise equal."""
    _engine_env(engine, monkeypatch)
    net_p, loss_p, _ = _train(c=False, monkeypatch=monkeypatch, lr=LEG_LR,
                              dump=tmp_path / "py")
    net_c, loss_c, _ = _train(c=True, monkeypatch=monkeypatch, lr=LEG_LR,
                              dump=tmp_path / "c")
    _assert_pools_equal(_dump(tmp_path / "py"), _dump(tmp_path / "c"))
    assert loss_p == loss_c and len(loss_c) > 2
    assert _state_bytes(net_p) == _state_bytes(net_c)
    # ... and training moved them (the comparison is not of two parents).
    assert _state_bytes(net_c) != _state_bytes(load_model(NET))


@pytest.mark.parametrize("engine", ENGINES)
def test_pipeline_and_save_every(simd, tmp_path, monkeypatch, engine):
    """--pipeline-collect with --save-every: the same pools (round N+1 on
    the weights from before round N's step, in both arms), checkpoints at
    every boundary, the same final weights.

    The master trains on cuda when there is one. On a CPU master the PYTHON
    arm's pipelined args were once views of the live parameters (`.cpu()`
    of a CPU tensor), which the pool's task thread pickled while the
    master's optimizer stepped (a race of the Python pipelined path, since
    fixed with .clone(); the C arm exports synchronously): there only
    round 1 is compared."""
    _engine_env(engine, monkeypatch)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = {}
    for arm in ("py", "c"):
        nets[arm], _, _ = _train(
            c=arm == "c", monkeypatch=monkeypatch, dump=tmp_path / arm,
            lr=LEG_LR, rounds=3, pipeline_collect=True, device=dev,
            save_path=str(tmp_path / f"{arm}.pt"), save_every=16)
        for ep in (16, 32, 48):
            assert (tmp_path / f"{arm}_ep{ep}.pt").exists(), (arm, ep)
    py, c = _dump(tmp_path / "py"), _dump(tmp_path / "c")
    if dev == "cpu":
        _assert_pools_equal(py[:1], c[:1], rounds=1)
        return
    _assert_pools_equal(py, c, rounds=3)
    assert _state_bytes(nets["py"]) == _state_bytes(nets["c"])
    for ep in (16, 32, 48):
        a = load_model(str(tmp_path / f"py_ep{ep}.pt"))
        b = load_model(str(tmp_path / f"c_ep{ep}.pt"))
        assert _state_bytes(a) == _state_bytes(b), ep


@pytest.mark.parametrize("engine", ENGINES)
def test_workers_1(simd, tmp_path, monkeypatch, engine):
    """--workers 1: the master collects the round in C on ONE seed drawn as
    a one-worker pool would; the pool equals _collect_worker on that seed."""
    _engine_env(engine, monkeypatch)
    _, _, m = _train(c=True, monkeypatch=monkeypatch, dump=tmp_path / "c",
                     workers=1, epr=6)
    c = _dump(tmp_path / "c")
    seeds = _master_seeds(20261011, 1, 2)
    assert [list(r["worker_seeds"]) for r in c] == seeds
    net = load_model(NET)
    sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
    args = (sd, list(net.hidden_sizes), net.input_size, net.output_mode,
            net.encoder_name, MatchplayMode(3), 6, seeds[0][0], True, False,
            0.0, False, MET, ALL_SCOPES, net.activation)
    enc_p, tgt_p = (D._REAL_WORKER or T._collect_worker)(args)
    assert list(c[0]["worker_rows"]) == [len(tgt_p)]
    assert np.array_equal(c[0]["enc"], enc_p)
    assert np.array_equal(c[0]["tgt"], tgt_p)
    assert str(c[0]["collector"]) == "c"


@pytest.mark.parametrize("engine", ENGINES)
def test_max_plies_discards_as_vectorized(simd, tmp_path, monkeypatch,
                                          capsys, engine):
    """--max-episode-plies: the C collector's bound has the vectorized
    semantics (per game, whole match discarded, logged): counts and pools
    equal the vectorized B=1 arm's at the same bound."""
    _engine_env(engine, monkeypatch)
    kw = dict(max_episode_plies=PLY_BOUND, rng="vectorized")
    _train(c=False, monkeypatch=monkeypatch, dump=tmp_path / "vec",
           vectorized_collect=True, vectorized_batch=1,
           vectorized_device="cpu", **kw)
    _train(c=True, monkeypatch=monkeypatch, dump=tmp_path / "c", **kw)
    vec, c = _dump(tmp_path / "vec"), _dump(tmp_path / "c")
    assert sum(int(r["n_discarded"]) for r in c) > 0
    _assert_pools_equal(vec, c, py_collector="vectorized")
    out = capsys.readouterr().out
    assert f"episodes DISCARDED (exceeded {PLY_BOUND} plies/game)" in out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("engine", ENGINES)
def test_cuda_graph_after_a_c_round(simd, tmp_path, monkeypatch, engine):
    """A C round followed by the graphed train step (cuda master) ==
    the Python round with the stock eager step, weights bitwise."""
    _engine_env(engine, monkeypatch)
    net_p, loss_p, _ = _train(c=False, monkeypatch=monkeypatch, device="cuda",
                              lr=LEG_LR, dump=tmp_path / "py")
    monkeypatch.setenv("BG_TRAIN_CUDA_GRAPH", "1")
    net_c, loss_c, _ = _train(c=True, monkeypatch=monkeypatch, device="cuda",
                              lr=LEG_LR, dump=tmp_path / "c")
    _assert_pools_equal(_dump(tmp_path / "py"), _dump(tmp_path / "c"))
    assert loss_p == loss_c
    assert _state_bytes(net_p) == _state_bytes(net_c)


# ── the dump, the export files ────────────────────────────────────────

def test_dump_contents(simd, tmp_path, monkeypatch):
    D.patch_trainer(monkeypatch)
    _train(c=True, monkeypatch=monkeypatch, dump=tmp_path, workers=3, epr=7,
           rounds=2)
    files = sorted(os.listdir(tmp_path))
    assert files == ["round_0001.npz", "round_0002.npz"]
    seeds = _master_seeds(20261011, 3, 2)
    for i, f in enumerate(files):
        z = np.load(tmp_path / f)
        assert set(z.files) == {"enc", "tgt", "worker_rows", "worker_seeds",
                                "n_discarded", "collector"}
        assert z["enc"].dtype == np.float32 and z["enc"].shape[1] == 350
        assert z["tgt"].shape == (len(z["enc"]),)
        assert z["worker_rows"].dtype == np.int64 and len(z["worker_rows"]) == 3
        assert int(z["worker_rows"].sum()) == len(z["enc"])
        assert list(z["worker_seeds"]) == seeds[i]
        assert int(z["n_discarded"]) == 0 and str(z["collector"]) == "c"
        assert np.all((z["tgt"] >= 0) & (z["tgt"] <= 1))
    # The C opts the workers saw are the leg's (the double asserts the
    # trainer constants; here: match length, 1-ply, rng serial, no bound).


def test_export_files(simd, tmp_path, monkeypatch):
    """Each round's weights: one unique file in a private 0700 dir,
    exported before the round's collection and removed once it returned;
    pipelined at most two exist; the dir is gone after the run."""
    D.patch_trainer(monkeypatch)
    import export_weights
    real = export_weights.export_live
    seen = []

    def spy(model, path):
        d = os.path.dirname(path)
        seen.append((path, sorted(os.listdir(d)),
                     stat.S_IMODE(os.stat(d).st_mode)))
        return real(model, path)

    monkeypatch.setattr(export_weights, "export_live", spy)
    _, _, m = _train(c=True, monkeypatch=monkeypatch, workers=2, epr=4,
                     rounds=3, pipeline_collect=True)
    d = m["c_collect_dir"]
    assert not os.path.exists(d)
    paths = [p for p, _, _ in seen]
    assert paths[0].endswith("w_probe.bin")
    assert [os.path.basename(p) for p in paths[1:]] == [
        "w_00001.bin.tmp", "w_00002.bin.tmp", "w_00003.bin.tmp"]
    assert all(mode == 0o700 for _, _, mode in seen)
    assert all(os.path.dirname(p) == d for p in paths)
    # Pipelined: when round N+1 is exported, round N's file is still
    # pending (submitted, not yet collected) -- never more than that.
    for p, listing, _ in seen[1:]:
        assert len([f for f in listing if f.endswith(".bin")]) <= 1, listing


def test_export_files_sync_removed_after_each_round(simd, tmp_path,
                                                    monkeypatch):
    D.patch_trainer(monkeypatch)
    seen = []
    real = T._c_collect_match

    def spy(weights, *a, **kw):
        seen.append((weights.path, os.path.exists(weights.path),
                     sorted(os.listdir(os.path.dirname(weights.path)))))
        return real(weights, *a, **kw)

    monkeypatch.setattr(T, "_c_collect_match", spy)
    _, _, m = _train(c=True, monkeypatch=monkeypatch, workers=1, epr=3,
                     rounds=3)
    assert [os.path.basename(p) for p, _, _ in seen] == [
        "w_00001.bin", "w_00002.bin", "w_00003.bin"]
    for p, existed, listing in seen:
        assert existed and listing == [os.path.basename(p)]
        assert not os.path.exists(p)
    assert not os.path.exists(m["c_collect_dir"])


# ── preconditions: one error naming every reason, before the pool ─────

def _tiny(encoder="matchplay1h64met_perspective196", **kw):
    from encoding import get_encoder
    torch.manual_seed(0)
    return TDNetwork(hidden_sizes=[8], input_size=get_encoder(encoder).num_features,
                     output_mode="probability", encoder_name=encoder, **kw)


class _Fwd(TDNetwork):
    def forward(self, x):
        return super().forward(x)


@pytest.fixture
def no_pool(monkeypatch):
    """Any process start fails the test: errors must come first."""
    class _NoMP:
        def get_context(self, *a, **k):
            raise AssertionError("a pool was started before the error")
    monkeypatch.setattr(T, "mp", _NoMP())


# name -> (train kwargs / net / mode / env, the reason's text)
REASONS = {
    "vectorized": (dict(vectorized_collect=True), "--vectorized-collect (two collectors"),
    "gpu_eval_server": (dict(gpu_eval_server=True), "--gpu-eval-server"),
    "oracle_eval_server": (dict(oracle_eval_server=True), "--oracle-eval-server"),
    "boltzmann": (dict(boltzmann_temp=0.5), "--boltzmann-temp 0.5"),
    "oneply_acting": (dict(oneply_acting=True), "--oneply-acting"),
    "td_oneply_act": (dict(mode=dict(oneply_act_k=3)), "--td-oneply-act 3"),
    "oracle_targets": (dict(mode=dict(oracle_spec="p5.pt")), "--oracle-targets"),
    "oracle_act": (dict(mode=dict(oracle_act_prob=0.5)), "--oracle-act-only / --oracle-act-prob"),
    "td_move_eps": (dict(mode=dict(td_move_eps=0.01)), "--td-move-eps 0.01"),
    "td_eps_only": (dict(mode=dict(td_eps_only=True)), "--td-eps-only"),
    "move_explore": (dict(mode=dict(move_explore_eps=0.1)), "--move-explore-eps 0.1"),
    "move_explore_env": (dict(move_eps_env=0.2), "TRAINER_MOVE_EXPLORATION_EPS 0.2"),
    "no_met": (dict(met_path=None), "no --met"),
    "met_scope": (dict(met_scopes=frozenset(("cube", "terminal"))), "--met-scope cube,terminal"),
    "met_mismatch": (dict(env={"BG_MET_PATH": "/usr/share/gnubg/met/g11.xml"}),
                     "the encoder MET (BG_MET_PATH=/usr/share/gnubg/met/g11.xml) is not "
                     "the grounding MET"),
    "match_length": (dict(match_length=26), "--match-length 26 is above the 25-point MET"),
    "encoder": (dict(net=dict(encoder="matchplay1h64_perspective196")),
                "encoder 'matchplay1h64_perspective196'"),
    "curriculum_force": (dict(env={"TRAINER_CURRICULUM_FORCE": "0,3"}),
                         "TRAINER_CURRICULUM_FORCE='0,3'"),
}


def _reason_case(case, monkeypatch):
    kw = dict(REASONS[case][0])
    net = _tiny(**kw.pop("net", {}))
    mode = MatchplayMode(kw.pop("match_length", 3), **kw.pop("mode", {}))
    for k, v in kw.pop("env", {}).items():
        monkeypatch.setenv(k, v)
    if "move_eps_env" in kw:
        monkeypatch.setattr(T, "_MOVE_EXPLORATION_EPS", kw.pop("move_eps_env"))
    kw.setdefault("met_path", MET)
    kw.setdefault("met_scopes", ALL_SCOPES)
    return net, mode, kw


@pytest.mark.parametrize("case", sorted(REASONS))
def test_each_reason_is_an_error(case, monkeypatch, no_pool):
    net, mode, kw = _reason_case(case, monkeypatch)
    needle = REASONS[case][1]
    reasons = T.c_collect_fallbacks(net, mode, oneply=True, **{
        k: v for k, v in kw.items() if k in (
            "oneply_acting", "boltzmann_temp", "met_path", "met_scopes",
            "vectorized_collect", "gpu_eval_server", "oracle_eval_server")})
    assert any(needle in r for r in reasons), reasons
    assert len(reasons) == 1, reasons
    tr = T.Trainer(TDAgent(net, met=T._worker_met(MET)), lr=1e-9)
    with pytest.raises(ValueError) as ei:
        tr.train(mode, num_episodes=2, episodes_per_round=2, workers=2,
                 seed=1, oneply=True, log_every=0, c_collect=True,
                 c_collect_backend=BACKEND, **kw)
    msg = str(ei.value)
    assert msg.startswith("--c-collect cannot collect this configuration in C")
    assert needle in msg, msg
    assert "Drop --c-collect" in msg


def test_net_reasons():
    """Structural net checks (TDAgent itself refuses some of these nets,
    so they are read off c_collect_fallbacks directly)."""
    mode = MatchplayMode(3)
    kw = dict(met_path=MET, met_scopes=ALL_SCOPES)
    wide = _tiny()
    wide.input_size = 351
    assert any("a 351-input net with the 350-column encoder" in r
               for r in T.c_collect_fallbacks(wide, mode, **kw))
    eq = _tiny()
    eq.output_mode = "equity"
    assert any("output mode 'equity'" in r for r in T.c_collect_fallbacks(eq, mode, **kw))
    sub = _Fwd(hidden_sizes=[8], input_size=350, output_mode="probability",
               encoder_name="matchplay1h64met_perspective196")
    assert any("a _Fwd (the C engine runs the plain TDNetwork MLP" in r
               for r in T.c_collect_fallbacks(sub, mode, **kw))
    from modes import DMPMode
    r = T.c_collect_fallbacks(_tiny(), DMPMode(), **kw)
    assert any("game mode DMPMode is not match play" in x for x in r), r


def test_supported_configs_have_no_reason(monkeypatch):
    kw = dict(met_path=MET, met_scopes=ALL_SCOPES)
    for enc in T.C_COLLECT_MATCH_ENCODERS:
        for oneply in (True, False):
            assert T.c_collect_fallbacks(_tiny(enc), MatchplayMode(11),
                                         oneply=oneply, **kw) == []
    # served in C: met_scopes None (= all), curriculum, cube eps, upweight,
    # outcome mix, a well-formed force, 25-point matches
    monkeypatch.setenv("TRAINER_CURRICULUM_FORCE", "1,5")
    assert T.c_collect_fallbacks(_tiny(), MatchplayMode(25, outcome_mix=0.3),
                                 met_path=MET, met_scopes=None) == []


def test_every_reason_is_named_together(monkeypatch, no_pool):
    net = _tiny("matchplay1h64_perspective196")
    tr = T.Trainer(TDAgent(net, met=T._worker_met(MET)), lr=1e-9)
    with pytest.raises(ValueError) as ei:
        tr.train(MatchplayMode(3, td_move_eps=0.01), num_episodes=2,
                 episodes_per_round=2, workers=2, seed=1, oneply=True,
                 log_every=0, boltzmann_temp=0.25, met_path=MET,
                 met_scopes=frozenset(("cube",)), c_collect=True,
                 vectorized_collect=True)
    msg = str(ei.value)
    assert "(5 reasons)" in msg, msg
    for needle in ("--vectorized-collect", "--boltzmann-temp 0.25",
                   "--td-move-eps 0.01", "--met-scope cube",
                   "encoder 'matchplay1h64_perspective196'"):
        assert needle in msg, (needle, msg)


def test_backends_are_the_bindings():
    assert T.C_COLLECT_BACKENDS == P.BACKENDS


def test_bad_rng_mode_is_an_error(no_pool):
    tr = T.Trainer(TDAgent(_tiny(), met=T._worker_met(MET)), lr=1e-9)
    with pytest.raises(ValueError, match="c_collect_rng must be one of"):
        tr.train(MatchplayMode(3), num_episodes=2, episodes_per_round=2,
                 workers=2, oneply=True, met_path=MET, c_collect=True,
                 c_collect_rng="stream")
    assert T.c_collect_rng_mode("serial") == 0
    assert T.c_collect_rng_mode("vectorized") == 1


# ── the library ───────────────────────────────────────────────────────

@pytest.fixture
def unbuildable(tmp_path, monkeypatch):
    """An empty PURETD_LIBDIR and a build.sh that fails."""
    libdir = tmp_path / "lib"
    libdir.mkdir()
    bad = tmp_path / "build.sh"
    bad.write_text("echo 'gcc: not found' >&2; exit 7\n")
    monkeypatch.setenv("PURETD_LIBDIR", str(libdir))
    monkeypatch.setattr(P, "BUILD_SH", str(bad))
    monkeypatch.setattr(P, "_build_failed", {})
    monkeypatch.setattr(P, "_libs", {})
    return libdir


def test_library_missing_is_an_error_before_the_pool(simd, unbuildable,
                                                     monkeypatch, no_pool):
    import tempfile
    D.install(monkeypatch)           # the binding is not what fails here
    dirs = lambda: set(glob.glob(os.path.join(tempfile.gettempdir(),  # noqa: E731
                                              "bg_ccollect_*")))
    before = dirs()
    tr = T.Trainer(TDAgent(load_model(NET), met=T._worker_met(MET)), lr=1e-9)
    with pytest.raises(RuntimeError,
                       match=r"--c-collect: .*libpuretd_simd.so not found and "
                             r"`bash c_api/build.sh simd` did not build it "
                             r"\(exit 7: gcc: not found\)"):
        tr.train(MatchplayMode(3), num_episodes=8, episodes_per_round=4,
                 workers=4, seed=1, oneply=True, met_path=MET,
                 met_scopes=ALL_SCOPES, c_collect=True)
    assert dirs() == before          # no export dir either


def test_missing_library_is_built_on_demand(simd, tmp_path, monkeypatch,
                                            capsys):
    monkeypatch.setenv("PURETD_LIBDIR", str(tmp_path))
    monkeypatch.setattr(P, "_libs", {})
    D.install(monkeypatch)
    assert T.c_collect_library(BACKEND) is None
    so = tmp_path / f"libpuretd_{BACKEND}.so"
    assert so.is_file()
    assert f"NOTE: built {so} on demand" in capsys.readouterr().out


def test_binding_without_collect_match_is_an_error(simd, monkeypatch, no_pool):
    monkeypatch.delattr(P.Engine, "collect_match", raising=False)
    why = T.c_collect_library(BACKEND)
    assert why is not None and "Engine.collect_match" in why
    assert "C matchplay collector" in why
    tr = T.Trainer(TDAgent(load_model(NET), met=T._worker_met(MET)), lr=1e-9)
    with pytest.raises(RuntimeError, match="--c-collect: c_api/puretd_py.py has no"):
        tr.train(MatchplayMode(3), num_episodes=8, episodes_per_round=4,
                 workers=4, seed=1, oneply=True, met_path=MET,
                 met_scopes=ALL_SCOPES, c_collect=True)


def test_probe_refuses_a_non_match_engine(simd, monkeypatch, no_pool):
    """The start-up probe opens the export before the pool: an engine of
    another kind is an error there, not inside pool.map."""
    D.install(monkeypatch)

    class _Money(D.FakeMatchEngine):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.kind = P.KIND_CUBEFUL

    monkeypatch.setattr(P, "Engine", _Money)
    tr = T.Trainer(TDAgent(load_model(NET), met=T._worker_met(MET)), lr=1e-9)
    with pytest.raises(RuntimeError, match="not the matchplay kind 4"):
        tr.train(MatchplayMode(3), num_episodes=8, episodes_per_round=4,
                 workers=4, seed=1, oneply=True, met_path=MET,
                 met_scopes=ALL_SCOPES, c_collect=True)


@needs_c
def test_match_encoders_are_the_c_table(simd):
    """trainer.C_COLLECT_MATCH_ENCODERS (the preconditions, no library) are
    the C API encoder table's entries with collect set, over every
    matchplay encoder name encoding.get_encoder builds."""
    _need_enabled_c()
    from encoding import ENCODERS, get_encoder
    prefixes = ("matchplay_", "matchplay1h_", "matchplay1h64_",
                "matchplay1h64met_", "matchplay1h64met4_", "matchplay1h64met5_",
                "matchplay1h64met6_", "matchplaymet_")
    names = [p + b for p in prefixes for b in ENCODERS]
    infos = {n: P.encoder_info(n, BACKEND) for n in names}
    served = {n for n, i in infos.items() if i and i["collect"]}
    assert served == set(T.C_COLLECT_MATCH_ENCODERS), infos
    for n in served:
        assert infos[n]["input_size"] == get_encoder(n).num_features


# ── train_batch.py ────────────────────────────────────────────────────

def _cli(*flags, timeout=300):
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    for k in ("BG_COLLECT_DUMP", "BG_FAST_FORWARD", "BG_MET_PATH"):
        env.pop(k, None)
    r = subprocess.run([sys.executable, "train_batch.py", *flags], cwd=REPO,
                       env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


BASE = ["--game-mode", "matchplay", "--match-length", "3", "--oneply",
        "--met", "auto", "--resume", NET, "--device", "cpu"]


def test_cli_c_collect_with_vectorized_is_an_error():
    rc, out = _cli(*BASE, "--c-collect", "--vectorized-collect",
                   "--num-episodes", "2")
    assert rc == 2 and "--c-collect and --vectorized-collect are mutually " \
                       "exclusive" in out, out[-2000:]


def test_cli_options_need_c_collect():
    rc, out = _cli(*BASE, "--c-collect-rng", "serial", "--num-episodes", "2")
    assert rc == 2 and "--c-collect-rng requires --c-collect" in out, out[-2000:]


def test_cli_reasons_are_an_error():
    rc, out = _cli(*BASE, "--c-collect", "--boltzmann-temp", "0.5",
                   "--met-scope", "cube", "--num-episodes", "2")
    assert rc == 2, out[-2000:]
    assert "--c-collect cannot collect this configuration in C (2 reasons)" in out
    assert "--boltzmann-temp 0.5" in out and "--met-scope cube" in out


@needs_c
def test_cli_resume_save_every_pipeline(simd, tmp_path):
    """The leg's command shape, small: --resume, --save-every, --pipeline-
    collect, 2 workers; one ON line, checkpoints, a final save."""
    _need_enabled_c()
    rc, out = _cli(*BASE, "--c-collect", "--workers", "2",
                   "--episodes-per-round", "4", "--num-episodes", "12",
                   "--save-every", "4", "--pipeline-collect", "--lr", "1e-9",
                   "--seed", "3", "--save", str(tmp_path / "m.pt"))
    assert rc == 0, out[-3000:]
    assert out.count("C collection: ON (bgc_collect_match on ") == 1, out
    assert "W=2, rng=vectorized)" in out
    for ep in (4, 8, 12):
        assert (tmp_path / f"m_ep{ep}.pt").exists()
    assert (tmp_path / "m.pt").exists()
    assert "Round    3/3" in out
