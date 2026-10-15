"""gnubg-ext replies the agent cannot apply raise GnubgReplyError (no
fallback move or default cube decision); an H-H run hitting one fails
without saving, and saved runs with a gnubg-ext side record
gnubg_fallbacks = "0"."""

import glob
import os
import subprocess
import sys

import numpy as np
import pytest

from backgammon_engine import BoardState, get_legal_plays
from third_party_opponents.gnubg.gnubg_external import GnubgExternalAgent, GnubgReplyError
from modes import MatchState
from play_models import save_head_to_head

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Server:
    def __init__(self, reply):
        self.reply = reply
        self.lines = []

    def request(self, line):
        self.lines.append(line)
        return self.reply


def _agent(reply):
    ag = GnubgExternalAgent()
    ag._server = _Server(reply)
    return ag


def _dance_state():
    """White on the bar against Black's closed board."""
    pts = [0] * 24
    for i in range(18, 24):
        pts[i] = -2
    pts[0] = 2
    return BoardState(points=pts, bar=[1, 0], off=[12, 3], turn=0)


@pytest.mark.parametrize("reply,what", [
    ("24/1", "matches no legal successor"),
    ("", "no move with legal plays"),
    ("Error: badly formed board", "matches no legal successor"),
])
def test_checker_reply_raises_with_context(reply, what):
    s, dice = BoardState.initial(), (3, 1)
    for call in ("cubeless", "cubeful"):
        ag = _agent(reply)
        with pytest.raises(GnubgReplyError) as e:
            if call == "cubeless":
                ag.choose_checker_action(s, dice, get_legal_plays(s, dice))
            else:
                ag.choose_checker_action_cubeful(s, dice, MatchState())
        msg = str(e.value)
        assert what in msg and repr(reply) in msg and "(3, 1)" in msg
        assert repr(ag._server.lines[0]) in msg and "points=" in msg


def test_dance():
    s = _dance_state()
    assert not get_legal_plays(s, (6, 6))
    assert _agent("").choose_checker_action_cubeful(s, (6, 6), MatchState()) is None
    with pytest.raises(GnubgReplyError, match="move with no legal play"):
        _agent("bar/19").choose_checker_action_cubeful(s, (6, 6), MatchState())


def test_cube_replies():
    s, ms = BoardState.initial(), MatchState()
    assert _agent("double").offer_double(s, ms).should_double
    assert not _agent("roll").offer_double(s, ms).should_double
    assert _agent("take").respond_to_double(s, ms) is True
    assert _agent("drop").respond_to_double(s, ms) is False
    for reply in ("", "Error: badly formed board", "take"):
        with pytest.raises(GnubgReplyError, match="not double/roll"):
            _agent(reply).offer_double(s, ms)
    for reply in ("", "beaver", "accept", "double"):
        with pytest.raises(GnubgReplyError, match="not take/drop"):
            _agent(reply).respond_to_double(s, ms)


def test_saved_runs_record_zero_fallbacks(tmp_path):
    f = save_head_to_head(str(tmp_path), [1.0, -1.0], game_mode="cubeless-money",
                          side1="gnubg-ext", side2="random", n_games=2)
    assert str(np.load(f)["gnubg_fallbacks"]) == "0"
    f = save_head_to_head(str(tmp_path), [1.0, -1.0], game_mode="cubeless-money",
                          side1="random", side2="random", n_games=2)
    assert "gnubg_fallbacks" not in np.load(f)


_FAKE_GNUBG = r'''
import os, re, socket, sys
if "-c" in sys.argv:
    sys.exit(0)
port = int(re.search(r"external localhost:(\d+)", sys.stdin.read()).group(1))
srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port))
srv.listen(1)
conn, _ = srv.accept()
fh = conn.makefile("rw")
for line in fh:
    if line.strip() == "exit":
        break
    if os.environ["FAKE_GNUBG_REPLY"] == "__hang__":
        continue
    fh.write(os.environ["FAKE_GNUBG_REPLY"] + "\n")
    fh.flush()
'''


def _fake(tmp_path):
    fake = tmp_path / "fake_gnubg.py"
    fake.write_text(f"#!{sys.executable}\n" + _FAKE_GNUBG)
    fake.chmod(0o755)
    return fake


def test_request_timeout(tmp_path, monkeypatch):
    from third_party_opponents.gnubg.gnubg_external import _ExtServer
    for k, v in (("FAKE_GNUBG_REPLY", "__hang__"),
                 ("GNUBG_ALLOW_VERSION_MISMATCH", "1"),
                 ("GNUBG_ALLOW_ONESIDED_BEAROFF", "1")):
        monkeypatch.setenv(k, v)
    srv = _ExtServer(0, gnubg_cmd=str(_fake(tmp_path)), timeout=1.0)
    try:
        with pytest.raises(GnubgReplyError, match="no reply within 1 s.*'board:x'"):
            srv.request("board:x")
    finally:
        srv.close()


def test_cube_clamp_counted(capsys):
    ag = _agent("roll")
    ag.offer_double(BoardState.initial(), MatchState(cube_value=8192))
    assert ag.cube_clamps == 1
    assert ":4096:" in ag._server.lines[0]
    assert "cube 8192 sent as 4096" in capsys.readouterr().err
    ag.offer_double(BoardState.initial(), MatchState(cube_value=4096))
    assert ag.cube_clamps == 1


@pytest.mark.parametrize("reply", ["24/1", pytest.param("", marks=pytest.mark.slow)])
def test_hh_run_fails_on_bad_reply(tmp_path, reply):
    fake = _fake(tmp_path)
    out = tmp_path / "runs"
    env = dict(os.environ, GNUBG_CMD=str(fake), FAKE_GNUBG_REPLY=reply,
               GNUBG_ALLOW_VERSION_MISMATCH="1",
               GNUBG_ALLOW_ONESIDED_BEAROFF="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [sys.executable, os.path.join(_REPO, "experiments", "head_to_head", "run.py"),
         "--side1", "gnubg-ext", "--side2", "random", "--games", "2",
         "--workers", "1", "--n-bootstrap", "50", "--save", str(out)],
        cwd=_REPO, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert "GnubgReplyError" in r.stderr and repr(reply) in r.stderr
    assert "request 'board:gnubg:opp:9999" in r.stderr
    assert not glob.glob(str(out / "*.npz"))


def _run_live(tmp_path):
    out = tmp_path / "runs"
    r = subprocess.run(
        [sys.executable, os.path.join(_REPO, "experiments", "head_to_head", "run.py"),
         "--side1", "gnubg-ext", "--side2", "random", "--games", "2",
         "--workers", "1", "--n-bootstrap", "50", "--save", str(out)],
        cwd=_REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    (f,) = glob.glob(str(out / "*.npz"))
    assert str(np.load(f)["gnubg_fallbacks"]) == "0"


try:
    from tests.test_gnubg_external import needs_gnubg
except ImportError:                     # run from tests/ without the package
    from test_gnubg_external import needs_gnubg
test_hh_run_with_real_gnubg_records_zero = needs_gnubg(_run_live)
