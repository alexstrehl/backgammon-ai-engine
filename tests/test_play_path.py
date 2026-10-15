"""Play-path routing (play_models.resolve_c_api) and the C API ABI guard.

* the default plays checkpoint sides through the C API (simd), building
  the library on demand; 2-ply sides, --bf16 prob5 sides and an
  unbuildable library fall back to the Python path with one NOTE;
  --c-api turns the fallback into an error; --python-api forces Python;
  --c-api with --python-api is an error (play_models, head_to_head/run.py,
  the trainers' --python-api alias of --no-c-collect);
* puretd_py.ensure_lib rebuilds a library whose ABI tag is stale (or,
  rebuild_stale=False, refuses it naming the rebuild); load_lib refuses a
  library whose bgc_abi_version or struct sizes differ.

simd is built as in tests/test_c_api.py.
"""
import os
import shutil
import subprocess
import sys

import pytest

from test_c_api import _build, LIBDIR, PROD_CUBEFUL, PROD_PROB5, REPO  # noqa: E402

PY = sys.executable


@pytest.fixture(scope="module")
def simd():
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    ok, log = _build("simd")
    if not ok:
        pytest.skip(f"simd backend not built:\n{log[-500:]}")
    return os.path.join(LIBDIR, "libpuretd_simd.so")


@pytest.fixture
def P(monkeypatch):
    sys.path.insert(0, REPO)
    from c_api import puretd_py
    monkeypatch.delenv("PURETD_BACKEND", raising=False)
    monkeypatch.delenv("PURETD_ROOT_FILTER", raising=False)
    monkeypatch.setattr(puretd_py, "_build_failed", {})
    monkeypatch.setattr(puretd_py, "_built", {})
    return puretd_py


def _resolve(c_api, *sides, cubeful=False, bf16=False, backend=None):
    import play_models as PM
    notes = []
    use_c, b = PM.resolve_c_api(c_api, list(sides), cubeful=cubeful, bf16=bf16,
                                backend=backend, log=notes.append)
    return use_c, b, notes


RANDOM = ("random", "random", False, 0)


def test_default_routes_to_c_simd(simd, P):
    import play_models as PM
    use_c, b, notes = _resolve(None, (PROD_PROB5, "model", False, 0), RANDOM)
    assert (use_c, b) == (True, "simd")
    assert notes == ["Play path: C API (simd)"]
    agent = PM._build_side(PROD_PROB5, "model", plies=0, sage_plies=1,
                           cubeless_money=True, cubeful=False, oneply=False,
                           twoply_k=0, bf16=False, c_api=use_c, c_api_backend=b)
    assert isinstance(agent, P.CAgent) and agent.engine.backend == "simd"


def test_no_checkpoint_side_needs_no_library(P, monkeypatch):
    monkeypatch.setattr(P, "ready_lib", lambda *a, **k: pytest.fail("built"))
    assert _resolve(None, RANDOM, ("gnubg-nn", "gnubg-nn", False, 0)) == (False, "simd", [])


@pytest.mark.parametrize("side,bf16,why", [
    ((PROD_CUBEFUL, "model", False, 3), False, "0- and 1-ply sides only"),
    ((PROD_PROB5, "model", False, 0), True, "fp32 only (--bf16)"),
], ids=["2ply", "bf16"])
def test_default_falls_back_with_one_note(simd, P, side, bf16, why):
    from prob_agent import ProbAgent
    from td_agent import TDAgent
    import play_models as PM
    use_c, _, notes = _resolve(None, side, RANDOM, cubeful=side[3] > 0, bf16=bf16)
    assert use_c is False
    assert len(notes) == 1 and notes[0].startswith("NOTE:") and why in notes[0]
    agent = PM._build_side(side[0], "model", plies=0, sage_plies=1,
                           cubeless_money=False, cubeful=side[3] > 0,
                           oneply=False, twoply_k=side[3], bf16=bf16, c_api=use_c)
    assert isinstance(agent, (TDAgent, ProbAgent))
    with pytest.raises(ValueError, match=f"--c-api: .*{why[:10]}"):
        _resolve(True, side, RANDOM, cubeful=side[3] > 0, bf16=bf16)


def test_default_falls_back_when_the_library_cannot_build(P, tmp_path, monkeypatch):
    monkeypatch.setenv("PURETD_LIBDIR", str(tmp_path))
    fake = tmp_path / "build.sh"
    fake.write_text("echo no compiler >&2; exit 1\n")
    monkeypatch.setattr(P, "BUILD_SH", str(fake))
    use_c, _, notes = _resolve(None, (PROD_PROB5, "model", False, 0), RANDOM)
    assert use_c is False and len(notes) == 1 and "no compiler" in notes[0]
    with pytest.raises(RuntimeError, match="did not build it"):
        _resolve(True, (PROD_PROB5, "model", False, 0), RANDOM)


def test_on_demand_messages_name_out(simd, P, tmp_path, monkeypatch, capsys):
    """With PURETD_LIBDIR set, the on-demand build NOTE and the build
    failure name OUT=<dir> in the build command."""
    monkeypatch.setenv("PURETD_LIBDIR", str(tmp_path))
    fake = tmp_path / "build.sh"
    fake.write_text(f'cp "{simd}" "$OUT/libpuretd_simd.so"\n')
    monkeypatch.setattr(P, "BUILD_SH", str(fake))
    assert P.ready_lib("simd") is None
    assert f"on demand (OUT={tmp_path} bash c_api/build.sh simd)" in capsys.readouterr().out
    fake.write_text("exit 1\n")
    why = P.ready_lib("ref")
    assert f"`OUT={tmp_path} bash c_api/build.sh ref` did not build it" in why


def _build_sh(tmp_path, *backends, **env):
    return subprocess.run(["bash", os.path.join(REPO, "c_api", "build.sh"), *backends],
                          capture_output=True, text=True,
                          env=dict(os.environ, OUT=str(tmp_path), **env))


def test_build_sh_named_backend_skipped_exits_nonzero(tmp_path):
    r = _build_sh(tmp_path, "nosuch")
    assert r.returncode == 1 and "skipped: nosuch" in r.stderr
    r = _build_sh(tmp_path, "torch", VENV=str(tmp_path), PURETD_TORCH_PYTHON="")
    assert r.returncode == 1
    assert f"no usable torch in {tmp_path} (from VENV)" in r.stderr
    assert "must be an interpreter path" in r.stderr


def test_python_api_forces_python(P, monkeypatch):
    monkeypatch.setattr(P, "ready_lib", lambda *a, **k: pytest.fail("built"))
    use_c, _, notes = _resolve(False, (PROD_PROB5, "model", False, 0), RANDOM)
    assert use_c is False and notes == ["Play path: Python (--python-api)"]
    with pytest.raises(ValueError, match="janowski-c"):
        _resolve(False, ("janowski-c:" + PROD_PROB5, "janowski-c", False, 0), RANDOM)


def test_root_filter_requires_c(P, tmp_path, monkeypatch):
    monkeypatch.setenv("PURETD_ROOT_FILTER", "0.08")
    monkeypatch.setenv("PURETD_LIBDIR", str(tmp_path))
    fake = tmp_path / "build.sh"
    fake.write_text("exit 1\n")
    monkeypatch.setattr(P, "BUILD_SH", str(fake))
    with pytest.raises(RuntimeError):
        _resolve(None, (PROD_PROB5, "model", True, 0), RANDOM)


@pytest.mark.parametrize("cmd,msg", [
    (["play_models.py", "--model1", PROD_PROB5, "--random", "--c-api", "--python-api"],
     "--c-api and --python-api are mutually exclusive"),
    (["experiments/head_to_head/run.py", "--side1", PROD_PROB5, "--side2", "random",
      "--c-api", "--python-api"],
     "--c-api and --python-api are mutually exclusive"),
    (["train_batch.py", "--game-mode", "cubeful-money", "--num-episodes", "1",
      "--c-collect", "--python-api"],
     "--c-collect and --no-c-collect/--python-api are mutually exclusive"),
    (["train_prob5.py", "--num-episodes", "1", "--c-collect", "--python-api"],
     "--c-collect and --no-c-collect/--python-api are mutually exclusive"),
], ids=["play_models", "run.py", "train_batch", "train_prob5"])
def test_c_api_with_python_api_is_an_error(tmp_path, cmd, msg):
    r = subprocess.run([PY, os.path.join(REPO, cmd[0]), *cmd[1:],
                        *(["--save", str(tmp_path / "x.pt")] if cmd[0].startswith("train") else [])],
                       capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert r.returncode != 0 and msg in r.stdout + r.stderr, r.stdout + r.stderr


# ── ABI guard ───────────────────────────────────────────────────────

def test_real_library_matches_the_ctypes_declarations(simd, P):
    lib = P.load_lib("simd")
    assert lib.bgc_abi_version() == P.ABI_VERSION
    assert P._abi_mismatch(lib) is None
    assert P.lib_abi_ok(simd)


def test_ensure_lib_rebuilds_a_stale_library(simd, P, tmp_path, monkeypatch, capsys):
    """A library without this ABI's tag is rebuilt (the default path), or
    refused naming the rebuild (rebuild_stale=False)."""
    monkeypatch.setenv("PURETD_LIBDIR", str(tmp_path))
    so = tmp_path / "libpuretd_abitest.so"
    so.write_bytes(b"\x7fELF stale, no tag")
    fake = tmp_path / "build.sh"
    fake.write_text(f'cp "{simd}" "$OUT/libpuretd_$1.so"\n')
    monkeypatch.setattr(P, "BUILD_SH", str(fake))
    with pytest.raises(RuntimeError, match="another C API ABI.*bash c_api/build.sh abitest"):
        P.ensure_lib("abitest", rebuild_stale=False)
    assert P.ready_lib("abitest") is None
    assert P.lib_abi_ok(str(so))
    assert "NOTE: rebuilt (stale ABI)" in capsys.readouterr().out
    assert P.ready_lib("abitest") is None                  # no second build, no NOTE
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("body,match", [
    ("void *bgc_open(const char *p) { (void)p; return 0; }\n",
     "built before the C API ABI check"),
    ("int bgc_abi_version(void) { return 999; }\n", "C API ABI 999"),
    ("#include <stddef.h>\nint bgc_abi_version(void) { return %d; }\n"
     "int bgc_struct_sizes(size_t *o, int n) { for (int i = 0; i < n; i++) o[i] = 1; return n; }\n",
     "struct sizes"),
], ids=["pre_abi", "version", "sizes"])
def test_load_lib_refuses_a_mismatched_library(P, tmp_path, monkeypatch, body, match):
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    src = tmp_path / "old.c"
    src.write_text(body.replace("%d", str(P.ABI_VERSION)))
    name = "abi_" + match.split()[0]
    so = tmp_path / f"libpuretd_{name}.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
    monkeypatch.setenv("PURETD_LIBDIR", str(tmp_path))
    with pytest.raises(RuntimeError, match=f"{match}.*bash c_api/build.sh {name}"):
        P.load_lib(name)
    assert name not in P._libs
