"""Tests for the HedgeHog integration (third_party_opponents/hedgehog).

Two tiers, mirroring test_gnubg_external.py:
* lib-free: spec detection, md5 provenance, import safety;
* behavioral: need libhhagent.so (skipped when not built) and, for the
  agent tests, a HedgeHog .ogxf model (skipped when absent).
"""
import hashlib
import os

import pytest

from third_party_opponents.hedgehog import hh_lib

_LIB = hh_lib.lib_path()
needs_hhlib = pytest.mark.skipif(
    _LIB is None, reason="libhhagent.so not built "
    "(third_party_opponents/hedgehog/shim/build_shim.sh)")

_MODEL = os.environ.get("HH_MODEL", os.path.join(
    os.path.dirname(hh_lib.__file__), "default-small_v1.0_2026-05-13.ogxf"))
needs_model = pytest.mark.skipif(
    not os.path.exists(_MODEL), reason="no HedgeHog .ogxf model available")


# ── lib-free ─────────────────────────────────────────────────────────

def test_spec_detection_and_md5(tmp_path):
    from play_models import _detect_side_type, side_model_md5
    assert _detect_side_type("hedgehog:/some/model.ogxf") == "hedgehog"
    m = tmp_path / "m.ogxf"
    m.write_bytes(b"hh-weights")
    assert side_model_md5(f"hedgehog:{m}") == \
        hashlib.md5(b"hh-weights").hexdigest()


def test_import_without_lib():
    """hh_lib and the agent module must import cleanly and fail with a
    helpful pointer at USE time when the shim is absent. Run in a fresh
    interpreter so an already-loaded lib in this process can't mask it."""
    import subprocess
    import sys
    code = (
        "import os; os.environ['HH_LIB']='/nonexistent/libhhagent.so'\n"
        "import sys; sys.path.insert(0, %r)\n"
        "from third_party_opponents.hedgehog import hh_lib, hedgehog_agent\n"
        "try:\n"
        "    hh_lib._load()\n"
        "except RuntimeError as e:\n"
        "    assert 'build_shim' in str(e), e\n"
        "    print('OK')\n"
    ) % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),)
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "OK" in out.stdout, (out.stdout,
                                                       out.stderr)


def test_missing_model_raises_filenotfound():
    from third_party_opponents.hedgehog.hedgehog_agent import HedgehogAgent
    with pytest.raises(FileNotFoundError, match="README"):
        HedgehogAgent("/nonexistent/model.ogxf")


# ── behavioral: board conversion (lib, no model needed) ──────────────

@needs_hhlib
def test_board_roundtrip_initial_and_asymmetric():
    from backgammon_engine import BoardState
    from third_party_opponents.hedgehog.hedgehog_agent import board_to_hh, hh_to_ours
    for state in (
        BoardState.initial(),
        BoardState(points=[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0] + [0] * 12,
                   bar=[1, 2], off=[11, 11], turn=0),
    ):
        back = hh_to_ours(board_to_hh(state), state.turn)
        assert back.points == state.points
        assert back.bar == state.bar and back.off == state.off


@needs_hhlib
def test_successor_sets_agree_on_opening_roll():
    from backgammon_engine import BoardState, get_legal_plays
    from third_party_opponents.hedgehog.hedgehog_agent import board_to_hh
    from third_party_opponents.hedgehog.hh_lib import HH
    s = BoardState.initial()
    ours = {board_to_hh(ns) for _, ns in get_legal_plays(s, (3, 1))}
    n, theirs = HH.list_moves(board_to_hh(s), s.turn, 3, 1)
    assert ours == set(theirs)


# ── behavioral: agent (lib + model) ──────────────────────────────────

@needs_hhlib
@needs_model
def test_agent_plays_a_full_cubeful_game():
    """Self-play with cube action: seed 7 produces doubles and a cube
    response, so the cube path is exercised, not just checker play."""
    import random
    from play_models import play_game
    from third_party_opponents.hedgehog.hedgehog_agent import HedgehogAgent
    random.seed(7)
    a = HedgehogAgent(_MODEL, depth=0)
    winner, stake = play_game(a, a, cubeful=True, jacoby=True)
    assert winner in (0, 1) and stake >= 1
    assert a.n_moves > 0
    assert a.n_doubles_offered > 0
    assert a.n_takes + a.n_passes > 0


@needs_hhlib
@needs_model
def test_take_is_evaluated_from_the_doubler_side():
    """Pin the doubler-side convention: with the DOUBLER far ahead the
    take answer must be pass; with the roles reversed, take."""
    from backgammon_engine import BoardState
    from modes import MatchState
    from third_party_opponents.hedgehog.hedgehog_agent import HedgehogAgent
    a = HedgehogAgent(_MODEL, depth=0)
    # WHITE massively ahead (14 off, one checker on the ace) vs BLACK
    # with all 15 in its home board, none off.
    pts = [1] + [0] * 18 + [-3, -3, -3, -3, -3]
    ahead = BoardState(points=pts, off=[14, 0], turn=0)
    ms = MatchState()
    assert a.respond_to_double(ahead, ms) is False   # doubler wins: pass
    behind = BoardState(points=pts, off=[14, 0], turn=1)
    assert a.respond_to_double(behind, ms) is True   # doubler losing: take


@needs_hhlib
@needs_model
def test_agent_cubeless_pick_is_legal():
    from backgammon_engine import BoardState, get_legal_plays
    from third_party_opponents.hedgehog.hedgehog_agent import HedgehogAgent
    a = HedgehogAgent(_MODEL, depth=0)
    s = BoardState.initial()
    plays = get_legal_plays(s, (6, 5))
    _, nxt = a.choose_checker_action(s, (6, 5), plays)
    assert any(nxt == p[1] for p in plays)
