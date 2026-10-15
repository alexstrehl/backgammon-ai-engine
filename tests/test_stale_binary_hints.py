"""nn_load's bad-magic message names the magics it accepts and shows the
bytes it read (the nn_eval.c part; this tree has no c_engine/bg_nply.py,
so its loader-hint cases do not apply).

nn_eval.c is compiled into a throwaway harness; no repo library is built.
"""
import os
import shutil
import subprocess

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
