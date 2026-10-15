"""Stale-binary hints for the format-2 export header ("BGNV").

A library built before format 2 rejects a format-2 export at its magic;
its loader names the rebuild instead. nn_load's bad-magic message names
the magics it accepts. No repo library is built here: bg_nply's library
is faked, nn_eval.c is compiled into a throwaway harness.
"""
import os
import shutil
import subprocess

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HINT = ("predates the format-2 export header; "
         "rebuild: bash c_engine/build_nply.sh")


class _FailingLib:
    def bgn_load(self, path):
        return -1


@pytest.fixture
def cn(monkeypatch):
    cn = pytest.importorskip("c_engine.bg_nply")
    monkeypatch.setattr(cn, "_lib", _FailingLib())
    return cn


def _so(tmp_path, monkeypatch, cn, body):
    so = tmp_path / "libbg_nply.so"
    so.write_bytes(b"\x7fELF" + body)
    monkeypatch.setattr(cn, "_SO", str(so))
    return so


def test_bg_nply_stale_lib_names_rebuild(cn, tmp_path, monkeypatch):
    so = _so(tmp_path, monkeypatch, cn, b"nn_load: bad magic in '%s'\n")
    binp = tmp_path / "m.bin"
    binp.write_bytes(b"BGNV" + b"\0" * 100)
    with pytest.raises(RuntimeError) as e:
        cn.load_model(str(binp))
    assert f"{so} {_HINT}" in str(e.value)


@pytest.mark.parametrize("so_body,magic", [
    (b"is BGNV format version", b"BGNV"),   # format-2-aware library
    (b"bad magic", b"BGNN"),                # legacy export: stale lib reads it
])
def test_bg_nply_other_failures_no_hint(cn, tmp_path, monkeypatch, so_body,
                                        magic):
    _so(tmp_path, monkeypatch, cn, so_body)
    binp = tmp_path / "m.bin"
    binp.write_bytes(magic + b"\0" * 100)
    with pytest.raises(RuntimeError, match=r"bgn_load\(.*\) failed rc=-1") as e:
        cn.load_model(str(binp))
    assert "predates" not in str(e.value)


_HARNESS = r'''#include "nn_eval.h"
int main(int c, char **v) { NNModel m; (void)c; return nn_load(&m, v[1]) ? 1 : 0; }
'''


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_nn_load_bad_magic_names_formats(tmp_path):
    cinf = os.path.join(_REPO, "c_inference")
    src = tmp_path / "h.c"
    src.write_text(_HARNESS)
    exe = str(tmp_path / "h")
    r = subprocess.run(["gcc", "-O0", "-o", exe, str(src),
                        os.path.join(cinf, "nn_eval.c"), "-I", cinf, "-lm"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    cases = {
        b"BGNX" + b"\0" * 80: "is not a BGNN/BGNV export (magic BGNX)",
        b"\x00\x01AB" + b"\0" * 80: "is not a BGNN/BGNV export (magic ??AB)",
        b"BG": "is not a BGNN/BGNV export (2 bytes, no 4-byte magic)",
    }
    for i, (raw, want) in enumerate(cases.items()):
        p = tmp_path / f"{i}.bin"
        p.write_bytes(raw)
        r = subprocess.run([exe, str(p)], capture_output=True, text=True)
        assert r.returncode == 1
        assert r.stderr == f"nn_load: '{p}' {want}\n", r.stderr
