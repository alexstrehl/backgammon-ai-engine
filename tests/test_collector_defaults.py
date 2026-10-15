"""The trainer defaults, through the real train_batch.py CLI on cuda:

* no flags or env: the C collector (`C collection: ON (the default`), the
  graphed train step (`BG_TRAIN_CUDA_GRAPH (default on`) and the
  shared-memory row transport (`C collection rows: shared-memory arenas
  (the default`);
* each opt-out turns off its own option only: `--no-c-collect` (the C
  collector, and with it the row transport, which needs it; the graph
  stays), `BG_TRAIN_CUDA_GRAPH=0`, `BG_COLLECT_SHM=0`;
* a run the default C collector cannot collect (a precondition) gets the
  `--no-c-collect` routing: `--oneply --device cuda` with
  `--boltzmann-temp` goes to the vectorized GPU collector with the
  fallback NOTE and the AUTO NOTE; an explicit `--c-collect` keeps
  Trainer.train's NOTE + Python collector;
* `--vectorized-collect` alone routes to the GPU collector with a NOTE;
  `--c-collect --vectorized-collect` and `--c-collect --no-c-collect` are
  errors;
* the bitwise statement: a recipe-shaped command (cubeful money, prod 4L
  net, 4 workers, 3 rounds of 400 episodes, --pipeline-collect, LR anneal,
  --save-every) gives checkpoints bitwise equal (torch.equal over every
  state_dict, the periodic ones included) under the defaults and under
  `BG_TRAIN_CUDA_GRAPH=0 BG_COLLECT_SHM=0`, at 0-ply and at 1-ply.

CUDA only; the simd backend is built as in tests/test_c_api.py. ~1.5 min.
"""
import os
import shutil
import subprocess
import sys

import pytest
import torch

from test_c_api import _build, PROD_CUBEFUL, REPO  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

COLLECTOR_ENV = ("BG_TRAIN_CUDA_GRAPH", "BG_COLLECT_SHM", "BG_POOL_ON_DEVICE",
                 "BG_COLLECT_DUMP")
C_ON = "C collection: ON (the default, --no-c-collect for the Python collector;"
GRAPH_ON = "NOTE: BG_TRAIN_CUDA_GRAPH (default on; BG_TRAIN_CUDA_GRAPH=0 for the eager step): full batches replay"
GRAPH_OK = "first replay bitwise equal to the eager step"
SHM_ON = "C collection rows: shared-memory arenas (the default, BG_COLLECT_SHM=0 for the pool pipe;"
PIPE = "C collection rows: pickled through the pool pipe (BG_COLLECT_SHM=0)"


@pytest.fixture(scope="module")
def simd():
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    ok, log = _build("simd")
    if not ok:
        pytest.skip(f"simd backend not built:\n{log[-500:]}")
    return "simd"


def _run(args, env=None, timeout=300):
    """train_batch.py with COLLECTOR_ENV cleared (then `env`);
    returns (returncode, stdout + stderr)."""
    e = {k: v for k, v in os.environ.items() if k not in COLLECTOR_ENV}
    e.update(OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1", **(env or {}))
    r = subprocess.run([sys.executable, "-u", "train_batch.py", *args],
                       cwd=REPO, env=e, capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def _small(tmp_path, *extra):
    return ["--game-mode", "cubeful-money", "--resume", PROD_CUBEFUL,
            "--device", "cuda", "--workers", "2", "--num-episodes", "16",
            "--episodes-per-round", "8", "--lr", "1e-9", "--pipeline-collect",
            "--save", str(tmp_path / "m.pt"), *extra]


def _stamp(path):
    return torch.load(path, map_location="cpu", weights_only=True)["train_params"]


# ── which options are on ──────────────────────────────────────────────

def test_no_flags_turns_all_three_on(simd, tmp_path):
    rc, out = _run(_small(tmp_path))
    assert rc == 0, out[-3000:]
    assert C_ON in out and SHM_ON in out, out
    assert GRAPH_ON in out and GRAPH_OK in out, out
    assert "WARN: BG_TRAIN_CUDA_GRAPH" not in out and "falling back" not in out, out
    assert _stamp(tmp_path / "m.pt")["c_collect"] is True


def test_no_c_collect_turns_off_the_collector_and_its_transport(simd, tmp_path):
    rc, out = _run(_small(tmp_path, "--no-c-collect"))
    assert rc == 0, out[-3000:]
    assert "C collection" not in out and "BG_COLLECT_SHM" not in out, out
    # The graph belongs to the train step, not the collector.
    assert GRAPH_ON in out and GRAPH_OK in out, out
    assert _stamp(tmp_path / "m.pt")["c_collect"] is False


def test_graph_off_leaves_the_other_two(simd, tmp_path):
    rc, out = _run(_small(tmp_path), env={"BG_TRAIN_CUDA_GRAPH": "0"})
    assert rc == 0, out[-3000:]
    assert C_ON in out and SHM_ON in out, out
    assert "BG_TRAIN_CUDA_GRAPH" not in out, out


def test_shm_off_leaves_the_other_two(simd, tmp_path):
    rc, out = _run(_small(tmp_path), env={"BG_COLLECT_SHM": "0"})
    assert rc == 0, out[-3000:]
    assert C_ON in out and PIPE in out and "shared-memory" not in out, out
    assert GRAPH_ON in out and GRAPH_OK in out, out


# ── routing ───────────────────────────────────────────────────────────

def test_vectorized_collect_alone_routes_to_the_gpu_collector(tmp_path):
    rc, out = _run(["--game-mode", "cubeful-money", "--hidden", "32,32",
                    "--oneply", "--device", "cuda", "--vectorized-collect",
                    "--vectorized-batch", "2", "--workers", "2",
                    "--num-episodes", "4", "--episodes-per-round", "4",
                    "--max-episode-plies", "200", "--lr", "1e-9"])
    assert rc == 0, out[-3000:]
    assert ("NOTE: --vectorized-collect: the vectorized GPU collector "
            "replaces the default C collector") in out, out
    assert "Vectorized GPU collection: ON" in out, out
    assert "C collection" not in out and "AUTO:" not in out, out


RULED_OUT = {
    "boltzmann": (["--boltzmann-temp", "0.5"], "--boltzmann-temp 0.5"),
}


@pytest.mark.parametrize("case", sorted(RULED_OUT))
@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit"])
def test_ruled_out_default_falls_back_to_the_no_c_collect_routing(simd, tmp_path, case, explicit):
    """--oneply --device cuda with a precondition the C collector cannot
    meet: under the default the vectorized GPU collector (the --no-c-collect
    routing), with the fallback NOTE and the AUTO NOTE; under an explicit
    --c-collect Trainer.train's NOTE + Python collector."""
    extra, why = RULED_OUT[case]
    rc, out = _run(["--game-mode", "cubeful-money", "--hidden", "32,32",
                    *extra, "--oneply", "--device", "cuda", "--workers", "2",
                    "--vectorized-batch", "2", "--num-episodes", "4",
                    "--episodes-per-round", "4", "--max-episode-plies", "200",
                    "--lr", "1e-9", "--save", str(tmp_path / "m.pt")]
                   + (["--c-collect"] if explicit else []))
    assert rc == 0, out[-3000:]
    assert "C collection: ON" not in out, out
    if explicit:
        assert f"NOTE: --c-collect: {why}" in out, out
        assert "falling back to the Python collector." in out, out
        assert "Vectorized GPU collection: ON" not in out and "AUTO:" not in out, out
    else:
        assert (f"NOTE: --c-collect (default): {why}") in out, out
        assert ("falling back to the vectorized GPU collector (the "
                "--no-c-collect routing)") in out, out
        assert "AUTO: --oneply --device cuda -> vectorized GPU collector" in out, out
        assert "Vectorized GPU collection: ON" in out, out
    assert _stamp(tmp_path / "m.pt")["c_collect"] is False


@pytest.mark.parametrize("flags", [["--c-collect", "--vectorized-collect"],
                                   ["--c-collect", "--no-c-collect"]],
                         ids=["vectorized", "no_c_collect"])
def test_explicit_c_collect_conflicts_are_errors(tmp_path, flags):
    rc, out = _run(_small(tmp_path, "--oneply", *flags))
    assert rc != 0 and "mutually exclusive" in out, out[-2000:]


# ── the bitwise statement ─────────────────────────────────────────────

EPR, ROUNDS = 400, 3


def _recipe(tmp_path, arm, oneply):
    """The shape of the cf_seed (0-ply) / cf_const (1-ply) recipes in
    experiments/retraining/README.md, shortened."""
    save = tmp_path / arm / "m.pt"
    save.parent.mkdir()
    args = ["--game-mode", "cubeful-money", "--resume", PROD_CUBEFUL,
            "--num-episodes", str(EPR * ROUNDS), "--episodes-per-round",
            str(EPR), "--optimizer", "adam", "--lr", "6e-5", "--end-lr",
            "5e-7", "--workers", "4", "--device", "cuda", "--pipeline-collect",
            "--seed", "11", "--torch-seed", "11", "--save-every", str(EPR),
            "--save", str(save)]
    if oneply:
        args += ["--oneply", "--max-episode-plies", "1500"]
    return args, save


def _ckpts(save):
    out = {"final": save}
    for n in range(1, ROUNDS + 1):
        out[n] = save.parent / f"m_ep{n * EPR}.pt"
    return {k: torch.load(p, map_location="cpu", weights_only=True)
            for k, p in out.items()}


@pytest.mark.parametrize("oneply", [False, True], ids=["0ply", "1ply"])
def test_defaults_bitwise_equal_to_eager_and_pipe(simd, tmp_path, oneply):
    runs = {}
    for arm, env in (("defaults", {}),
                     ("eager_pipe", {"BG_TRAIN_CUDA_GRAPH": "0",
                                     "BG_COLLECT_SHM": "0"})):
        args, save = _recipe(tmp_path, arm, oneply)
        rc, out = _run(args, env=env, timeout=600)
        assert rc == 0, out[-3000:]
        assert C_ON in out, out
        if arm == "defaults":
            assert SHM_ON in out and GRAPH_ON in out and GRAPH_OK in out, out
        else:
            assert PIPE in out and "BG_TRAIN_CUDA_GRAPH" not in out, out
        runs[arm] = _ckpts(save)
    parent = torch.load(PROD_CUBEFUL, map_location="cpu", weights_only=True)
    a, b = runs["defaults"], runs["eager_pipe"]
    assert any(not torch.equal(a["final"]["state_dict"][k], parent["state_dict"][k])
               for k in parent["state_dict"]), "the weights did not move"
    for key in a:
        sa, sb = a[key]["state_dict"], b[key]["state_dict"]
        assert sa.keys() == sb.keys()
        for k in sa:
            assert torch.equal(sa[k], sb[k]), (key, k)
