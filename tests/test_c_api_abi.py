"""The C API ABI guard (the ABI tests of the money-game engine's
tests/test_play_path.py, adapted: this tree has no play-path routing).

* the built library matches the ctypes declarations: bgc_abi_version() is
  ABI_VERSION, _abi_mismatch is None, the file carries the puretd-abi tag;
  every ctypes Structure the binding declares is in _ABI_STRUCTS, in the
  order of puretd.h's comment and of puretd.c's bgc_struct_sizes;
* puretd_py.ensure_lib builds a missing library and rebuilds one whose
  ABI tag is stale (or, rebuild_stale=False, refuses it naming the
  rebuild); ready_lib prints one NOTE when it built and none on the next
  call; a failed rebuild is remembered for the process;
* load_lib refuses a library without bgc_abi_version, with another
  version, without bgc_struct_sizes, or whose struct sizes or struct
  count (e.g. a money-only 11-struct table) differ from the ctypes
  declarations, naming the rebuild;
* the real build.sh rebuilds a library of another ABI in a PURETD_LIBDIR
  through ready_lib, and leaves no temp output or object directory
  behind, also when a compile or a link fails.

simd is built as in tests/test_c_api.py; the real rebuilds build ref into
tmp_path (one process, a few seconds each).
"""
import ctypes
import inspect
import os
import re
import shutil
import subprocess
import sys

import pytest

from test_c_api import _build, CAPI, LIBDIR, REPO  # noqa: E402

# _ABI_STRUCTS (ctypes) -> the C struct bgc_struct_sizes reports.
C_NAMES = {
    "_CBoardState": "BoardState", "_CPlay": "Play", "CCube": "bgc_cube",
    "CCubeEval": "bgc_cube_eval", "CSearch": "bgc_search",
    "CEncoderInfo": "bgc_encoder_info", "CRng": "bgc_rng",
    "CGameOpts": "bgc_game_opts", "CGameResult": "bgc_game_result",
    "CCollectOpts": "bgc_collect_opts", "CMatch": "bgc_match",
    "CMatchCube": "bgc_match_cube", "CMatchCollectOpts": "bgc_match_collect_opts",
    "CMet": "bgc_met", "CJanCtx": "bgc_jan_ctx",
}


def _need_gcc():
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")


@pytest.fixture(scope="module")
def simd():
    _need_gcc()
    ok, log = _build("simd")
    if not ok:
        pytest.skip(f"simd backend not built:\n{log[-500:]}")
    return os.path.join(LIBDIR, "libpuretd_simd.so")


@pytest.fixture
def P(monkeypatch):
    sys.path.insert(0, REPO)
    from c_api import puretd_py
    monkeypatch.delenv("PURETD_BACKEND", raising=False)
    monkeypatch.setattr(puretd_py, "_build_failed", {})
    monkeypatch.setattr(puretd_py, "_built", {})
    return puretd_py


def _so(tmp_path, name, body):
    """A tiny libpuretd_<name>.so in tmp_path from C source `body`."""
    _need_gcc()
    src = tmp_path / f"{name}.c"
    src.write_text(body)
    so = tmp_path / f"libpuretd_{name}.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
    return so


def _leftovers(d):
    return sorted(f for f in os.listdir(d) if f.startswith((".tmp.", "obj_")))


# ── the real library ───────────────────────────────────────────────

def test_real_library_matches_the_ctypes_declarations(simd, P):
    lib = P.load_lib("simd")
    assert lib.bgc_abi_version() == P.ABI_VERSION
    assert P._abi_mismatch(lib) is None
    assert P.lib_abi_ok(simd)
    assert lib.bgc_struct_sizes(None, 0) == len(P._ABI_STRUCTS)


def test_abi_structs_cover_the_binding(P):
    """Every ctypes Structure puretd_py declares (and bg_fast's board and
    play, which it passes) is size-checked, once."""
    from bg_fast import _CBoardState, _CPlay
    declared = {c for c in vars(P).values()
                if inspect.isclass(c) and issubclass(c, ctypes.Structure)
                and c.__module__ == P.__name__}
    assert len(set(P._ABI_STRUCTS)) == len(P._ABI_STRUCTS)
    assert set(P._ABI_STRUCTS) == declared | {_CBoardState, _CPlay}


def test_struct_order_matches_puretd_h_and_c(P):
    want = [C_NAMES[t.__name__] for t in P._ABI_STRUCTS]
    with open(os.path.join(CAPI, "puretd.h")) as f:
        h = f.read()
    m = re.search(r"sizeof the public structs, in this order:(.*?)\(puretd_py\._ABI_STRUCTS\)",
                  h, re.S)
    assert m, "puretd.h: bgc_struct_sizes order comment not found"
    assert [w.strip() for w in m.group(1).split(",")] == want
    with open(os.path.join(CAPI, "puretd.c")) as f:
        c = f.read()
    m = re.search(r"int bgc_struct_sizes\(.*?\{(.*?)\};", c, re.S)
    assert m, "puretd.c: bgc_struct_sizes not found"
    assert re.findall(r"sizeof\((\w+)\)", m.group(1)) == want


# ── ensure_lib / ready_lib ─────────────────────────────────────────

def _fake_build(P, tmp_path, monkeypatch, script):
    """BUILD_SH replaced by `script` (ensure_lib runs it with OUT set)."""
    fake = tmp_path / "build.sh"
    fake.write_text(script)
    monkeypatch.setattr(P, "BUILD_SH", str(fake))


def test_ensure_lib_rebuilds_a_stale_library(simd, P, tmp_path, monkeypatch, capsys):
    """A library whose tag is not this ABI's (here the real library with
    its tag patched to the next ABI) is rebuilt (the default), or refused
    naming the rebuild (rebuild_stale=False); ready_lib's NOTE says so
    once."""
    monkeypatch.setenv("PURETD_LIBDIR", str(tmp_path))
    so = tmp_path / "libpuretd_abitest.so"
    data = open(simd, "rb").read()
    tag = f"puretd-abi:{P.ABI_VERSION}\0".encode()
    assert data.count(tag) == 1
    so.write_bytes(data.replace(tag, f"puretd-abi:{P.ABI_VERSION + 1}\0".encode()))
    assert not P.lib_abi_ok(str(so))
    _fake_build(P, tmp_path, monkeypatch, f'cp "{simd}" "$OUT/libpuretd_$1.so"\n')
    with pytest.raises(RuntimeError, match=f"another C API ABI \\(not {P.ABI_VERSION}\\); "
                                           f"rebuild it: OUT={re.escape(str(tmp_path))} "
                                           f"bash c_api/build.sh abitest"):
        P.ensure_lib("abitest", rebuild_stale=False)
    assert P.ready_lib("abitest") is None
    assert P.lib_abi_ok(str(so))
    assert (f"NOTE: rebuilt (stale ABI) {so} on demand (bash c_api/build.sh abitest)."
            in capsys.readouterr().out)
    assert P.ready_lib("abitest") is None                  # no second build, no NOTE
    assert capsys.readouterr().out == ""


def test_missing_library_is_built_with_one_note(simd, P, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PURETD_LIBDIR", str(tmp_path))
    so = tmp_path / "libpuretd_abinew.so"
    _fake_build(P, tmp_path, monkeypatch, f'cp "{simd}" "$OUT/libpuretd_$1.so"\n')
    assert P.ready_lib("abinew") is None
    assert so.is_file()
    assert (f"NOTE: built {so} on demand (bash c_api/build.sh abinew)."
            in capsys.readouterr().out)
    assert P.ready_lib("abinew") is None
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("script,why", [
    ('echo "gcc: boom" >&2; exit 3\n', "exit 3: gcc: boom"),
    ('echo stale-again > "$OUT/libpuretd_$1.so"\n', "exit 0: "),    # built, still no tag
], ids=["fails", "no_tag"])
def test_failed_rebuild_is_remembered(P, tmp_path, monkeypatch, script, why):
    monkeypatch.setenv("PURETD_LIBDIR", str(tmp_path))
    so = tmp_path / "libpuretd_abifail.so"
    so.write_bytes(b"\x7fELF stale, no tag")
    runs = tmp_path / "runs"
    _fake_build(P, tmp_path, monkeypatch, f'echo run >> "{runs}"\n' + script)
    msg = P.ready_lib("abifail")
    assert msg is not None and msg.startswith(
        f"{so} is stale (ABI) and `bash c_api/build.sh abifail` did not build it ("), msg
    assert why in msg, msg
    assert P.ready_lib("abifail") == msg                   # remembered, not rebuilt
    assert runs.read_text() == "run\n"
    assert "abifail" not in P._libs


# ── load_lib ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name,body,match", [
    ("abi_pre", "void *bgc_open(const char *p) { (void)p; return 0; }\n",
     "built before the C API ABI check"),
    ("abi_version", "int bgc_abi_version(void) { return 999; }\n",
     "C API ABI 999, this module expects %d"),
    ("abi_nosizes", "int bgc_abi_version(void) { return %d; }\n",
     "without bgc_struct_sizes"),
    ("abi_sizes", "#include <stddef.h>\nint bgc_abi_version(void) { return %d; }\n"
     "int bgc_struct_sizes(size_t *o, int n) { for (int i = 0; i < n; i++) o[i] = 1; "
     "return n; }\n",
     "struct sizes"),
    # a money-only build: the same version, an 11-struct money table
    ("abi_count", "#include <stddef.h>\nint bgc_abi_version(void) { return %d; }\n"
     "int bgc_struct_sizes(size_t *o, int n) { for (int i = 0; i < n && i < 11; i++) "
     "o[i] = 1; return 11; }\n",
     "struct sizes"),
], ids=["pre_abi", "version", "no_sizes", "sizes", "count"])
def test_load_lib_refuses_a_mismatched_library(P, tmp_path, monkeypatch, name, body, match):
    v = str(P.ABI_VERSION)
    _so(tmp_path, name, body.replace("%d", v))
    monkeypatch.setenv("PURETD_LIBDIR", str(tmp_path))
    out = re.escape(str(tmp_path))
    with pytest.raises(RuntimeError, match=f"{match.replace('%d', v)}.*rebuild it: "
                                           f"OUT={out} bash c_api/build.sh {name}"):
        P.load_lib(name)
    assert name not in P._libs


# ── the real build.sh ──────────────────────────────────────────────

def test_real_build_rebuilds_a_stale_abi_library(P, tmp_path, monkeypatch, capsys):
    """A libpuretd_ref.so of the next ABI (tag and bgc_abi_version
    ABI_VERSION + 1) in a
    PURETD_LIBDIR: refused by ensure_lib(rebuild_stale=False) and by
    load_lib, naming the rebuild; ready_lib runs the real build.sh, which
    renames the new library over it (no temp file or object directory
    left), loads it and says so once. The refusal and the rebuild use two
    directories: a process that loaded a path keeps that library."""
    other = P.ABI_VERSION + 1
    body = ('#include <stddef.h>\n'
            f'__attribute__((used)) const char bgc_abi_tag[] = "puretd-abi:{other}";\n'
            f'int bgc_abi_version(void) {{ return {other}; }}\n'
            'int bgc_struct_sizes(size_t *o, int n) { (void)o; (void)n; return 0; }\n')
    monkeypatch.setattr(P, "_libs", {})
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    so_a = _so(a, "ref", body)
    shutil.copy(so_a, b / "libpuretd_ref.so")
    so_b = b / "libpuretd_ref.so"
    assert not P.lib_abi_ok(str(so_a))

    monkeypatch.setenv("PURETD_LIBDIR", str(a))
    out = re.escape(str(a))
    with pytest.raises(RuntimeError, match=f"another C API ABI.*rebuild it: OUT={out} "
                                           f"bash c_api/build.sh ref"):
        P.ensure_lib("ref", rebuild_stale=False)
    with pytest.raises(RuntimeError, match=f"C API ABI {other}, this module expects "
                                           f"{P.ABI_VERSION}; rebuild it: OUT={out} "
                                           f"bash c_api/build.sh ref"):
        P.load_lib("ref")
    assert "ref" not in P._libs

    monkeypatch.setenv("PURETD_LIBDIR", str(b))
    ino = os.stat(so_b).st_ino
    assert P.ready_lib("ref") is None
    assert os.stat(so_b).st_ino != ino                     # renamed into place
    assert P.lib_abi_ok(str(so_b))
    assert P._libs["ref"].bgc_abi_version() == P.ABI_VERSION
    assert (f"NOTE: rebuilt (stale ABI) {so_b} on demand (bash c_api/build.sh ref)."
            in capsys.readouterr().out)
    assert _leftovers(b) == []
    assert {"libpuretd_ref.so", "libbgt_ref.so", "puretd_selfplay_ref"} <= set(os.listdir(b))


@pytest.mark.parametrize("how", ["compile", "link"])
def test_failed_build_leaves_no_temp_files(tmp_path, how):
    """build.sh fails (a bad compile flag; a linker that writes a partial
    selfplay binary and fails): the EXIT trap removes the per-process
    object directory and every .tmp.<pid>.* output."""
    _need_gcc()
    out = tmp_path / "out"
    env = dict(os.environ, OUT=str(out))
    if how == "compile":
        env["CFLAGS_EXTRA"] = "-Werror=nonexistent-option"
    else:
        wrap = tmp_path / "bin"
        wrap.mkdir()
        gcc = wrap / "gcc"
        gcc.write_text(
            '#!/bin/bash\n'
            'for ((i=1;i<=$#;i++)); do [ "${!i}" = "-o" ] && { j=$((i+1)); o="${!j}"; }; done\n'
            'case "$o" in *puretd_selfplay_*) echo partial > "$o"; '
            'echo "fake ld: failed" >&2; exit 1;; esac\n'
            f'exec {shutil.which("gcc")} "$@"\n')
        gcc.chmod(0o755)
        env["PATH"] = f"{wrap}{os.pathsep}{env['PATH']}"
    r = subprocess.run(["bash", os.path.join(CAPI, "build.sh"), "ref"],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode != 0, r.stdout + r.stderr
    assert ("fake ld: failed" if how == "link" else "nonexistent-option") in r.stderr
    assert _leftovers(out) == [], os.listdir(out)
    if how == "link":                       # the libraries before it were installed
        assert {"libpuretd_ref.so", "libbgt_ref.so"} <= set(os.listdir(out))
        assert "puretd_selfplay_ref" not in os.listdir(out)
