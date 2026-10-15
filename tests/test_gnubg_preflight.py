"""The gnubg preflight (pinned version, gnubg.wd md5): play_models.py and
experiments/head_to_head/run.py refuse a gnubg-ext side on a missing or
unpinned gnubg with a parser error before any game, not a traceback."""
import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_gnubg(root, version, wd=None):
    (root / "bin").mkdir(parents=True)
    exe = root / "bin" / "gnubg"
    exe.write_text(f'#!/bin/sh\necho "GNU Backgammon {version}"\n')
    exe.chmod(0o755)
    if wd is not None:
        (root / "share" / "gnubg").mkdir(parents=True)
        (root / "share" / "gnubg" / "gnubg.wd").write_bytes(wd)
    return str(exe)


@pytest.fixture(params=["missing"] + [pytest.param(k, marks=pytest.mark.slow)
                                     for k in ("version", "no-wd", "wd-md5")])
def bad_gnubg(request, tmp_path):
    kind = request.param
    if kind == "missing":
        return str(tmp_path / "nope"), "cannot run gnubg"
    if kind == "version":
        return _fake_gnubg(tmp_path, "1.06.002"), "is version '1.06.002'"
    if kind == "no-wd":
        return _fake_gnubg(tmp_path, "1.08.003"), "cannot locate gnubg.wd"
    return _fake_gnubg(tmp_path, "1.08.003", b"x"), "has md5"


@pytest.mark.parametrize("cmd", [
    ["play_models.py", "--model1", "random", "--model2", "gnubg-ext"],
    ["experiments/head_to_head/run.py", "--side1", "random", "--side2", "gnubg-ext-full"],
], ids=["play_models", "run.py"])
def test_cli_refuses_cleanly(bad_gnubg, cmd):
    exe, why = bad_gnubg
    env = {k: v for k, v in os.environ.items()
           if k not in ("GNUBG_ALLOW_VERSION_MISMATCH", "GNUBG_ALLOW_ONESIDED_BEAROFF")}
    env["GNUBG_CMD"] = exe
    r = subprocess.run([sys.executable, *cmd, "--games", "2", "--workers", "1"],
                       cwd=_REPO, env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "error:" in r.stderr and why in r.stderr, r.stderr
    assert "Traceback" not in r.stderr and "Playing" not in r.stdout


def test_gnubg_nn_warning_cubeful_only_once(monkeypatch, capsys):
    """The gnubg-nn cube-approximation warning: cubeful sides only, once
    per process tree (the env mark reaches spawned workers)."""
    pytest.importorskip("gnubg_nn")
    from third_party_opponents.gnubg import gnubg_nn_agent as G
    monkeypatch.setenv(G._WARNED_ENV, "0")   # restored after the test
    G.GnubgNNAgent(plies=0, cubeless_money=True)
    G.GnubgNNAgent(plies=0)
    assert "WARNING" not in capsys.readouterr().err
    G.GnubgNNAgent(plies=0, cubeless_money=True, cubeful=True)
    G.GnubgNNAgent(plies=0, cubeless_money=True, cubeful=True)
    assert capsys.readouterr().err.count("[gnubg-nn] WARNING") == 1
    assert os.environ[G._WARNED_ENV] == "1"
