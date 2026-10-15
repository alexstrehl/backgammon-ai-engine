"""Without a C compiler: play_models.py, train_batch.py and train_prob5.py
run the Python path after one NOTE (default), run it silently with
--python-api, and refuse --c-api / --c-collect with a clean error.

Each test runs a copy of the sources without any built library, under a
PATH whose compilers are removed; the shared build is never touched."""
import os
import shutil
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIRS = ("c_engine", "c_api", "c_inference", "third_party_opponents",
         "experiments", "tools")
_IGNORE = shutil.ignore_patterns("*.so", "*.o", "__pycache__", "results",
                                 "*.npz", "puretd_selfplay_*", ".build.lock")
_COMPILERS = ("gcc", "cc", "c++", "g++", "clang", "tcc", "c89", "c99", "cpp")


def _is_compiler(name):
    base = name.rsplit("-", 1)[0] if name[-1:].isdigit() else name
    return (base in _COMPILERS or name.startswith(("gcc", "g++", "clang"))
            or "-gcc" in name or "-g++" in name)


@pytest.fixture(scope="module")
def nogcc(tmp_path_factory):
    root = tmp_path_factory.mktemp("nogcc")
    repo = root / "repo"
    repo.mkdir()
    for f in os.listdir(_REPO):
        if f.endswith(".py"):
            shutil.copy2(os.path.join(_REPO, f), repo / f)
    for d in _DIRS:
        shutil.copytree(os.path.join(_REPO, d), repo / d, ignore=_IGNORE)
    bin_dir = root / "bin"
    bin_dir.mkdir()
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not os.path.isdir(d):
            continue
        for n in os.listdir(d):
            if not _is_compiler(n) and not (bin_dir / n).exists():
                os.symlink(os.path.join(d, n), bin_dir / n)
    env = {k: v for k, v in os.environ.items()
           if k not in ("CC", "PURETD_LIBDIR", "PURETD_BACKEND",
                        "PURETD_ROOT_FILTER", "BG_COLLECT_SHM")}
    env["PATH"] = str(bin_dir)
    assert shutil.which("gcc", path=env["PATH"]) is None
    assert shutil.which("bash", path=env["PATH"])

    def run(*args):
        r = subprocess.run([sys.executable, "-u", *args], cwd=repo, env=env,
                           capture_output=True, text=True, timeout=600)
        return r.returncode, r.stdout + r.stderr

    run.dir = root
    return run


def _train(nogcc, script, flag):
    save = nogcc.dir / f"{script[:-3]}{flag}.pt"
    args = [script, "--hidden", "8", "--num-episodes", "2",
            "--episodes-per-round", "2", "--workers", "1", "--device", "cpu",
            "--save", str(save)]
    if script == "train_batch.py":
        args += ["--game-mode", "cubeful-money"]
    rc, out = nogcc(*args, *([flag] if flag else []))
    return rc, out, save


@pytest.mark.slow
@pytest.mark.parametrize("script", ["train_batch.py", "train_prob5.py"])
def test_trainers_without_gcc(nogcc, script):
    rc, out, save = _train(nogcc, script, "")
    assert rc == 0, out[-3000:]
    assert out.count("NOTE: --c-collect (default):") == 1, out
    assert "falling back to the Python collector" in out
    assert "Traceback" not in out and save.exists()

    rc, out, save = _train(nogcc, script, "--python-api")
    assert rc == 0, out[-3000:]
    assert "NOTE" not in out and "Traceback" not in out and save.exists()

    rc, out, save = _train(nogcc, script, "--c-collect")
    assert rc != 0
    assert "ERROR: --c-collect:" in out and "did not build it" in out, out
    assert "Traceback" not in out and not save.exists()


@pytest.mark.slow
def test_play_models_without_gcc(nogcc):
    rc, out, model = _train(nogcc, "train_batch.py", "--python-api")
    assert rc == 0, out[-3000:]
    base = ["play_models.py", "--model1", str(model), "--random",
            "--games", "2", "--workers", "1"]

    rc, out = nogcc(*base)
    assert rc == 0, out[-3000:]
    assert out.count("NOTE: the C API (default play path) cannot serve") == 1
    assert "Traceback" not in out and "RESULTS" in out

    rc, out = nogcc(*base, "--python-api")
    assert rc == 0, out[-3000:]
    assert "Play path: Python (--python-api)" in out
    assert "NOTE" not in out and "Traceback" not in out and "RESULTS" in out

    rc, out = nogcc(*base, "--c-api")
    assert rc == 2
    assert "error:" in out and "did not build it" in out, out
    assert "Traceback" not in out and "RESULTS" not in out
