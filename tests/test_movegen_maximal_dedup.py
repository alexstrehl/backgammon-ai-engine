"""Regression: a shorter game-ending play must not shadow the maximal play.

A 1-die bear-off (e.g. 2/off with the 2 of a 21) can reach the same final
board as the full both-dice play (2/1, 1/off). The board-key dedup used to
keep whichever play was generated first; when that was the shorter one, the
maximal-dice filter then discarded the position entirely — a false forced
pass on a WINNING play (found by differential testing against gnubg's
move generator; ~1 per 7k turns of real self-play, always the mover's
last checker).

Both engines (Python get_legal_plays and the C fast path) must return the
play in BOTH dice orders.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "c_engine"))

from backgammon_engine import BLACK, WHITE, BoardState, get_legal_plays


def _lone_checker_state(point_1based, turn=WHITE):
    """Mover's last checker on their `point_1based`; opponent's checkers
    kept OUT of the mover's home board so the walk play is never blocked."""
    pts = [0] * 24
    if turn == WHITE:
        pts[point_1based - 1] = 1
        pts[18] = -5          # black's own home region (indices 18-23)
        pts[19] = -5
        pts[20] = -5
        off = [14, 0]
    else:
        pts[24 - point_1based] = -1
        pts[5] = 5            # white's own home region (indices 0-5)
        pts[4] = 5
        pts[3] = 5
        off = [0, 14]
    return BoardState(points=pts, bar=[0, 0], off=off, turn=turn)


# (point, dice) pairs where the direct big-die bear-off collides with the
# walk-then-off play on the identical (game-over) board.
CASES = [
    (2, (2, 1)),
    (3, (3, 1)),
    (3, (3, 2)),
    (3, (6, 2)),
    (4, (5, 3)),
]


@pytest.mark.parametrize("point,dice", CASES)
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("turn", [WHITE, BLACK])
def test_python_no_false_pass(point, dice, swap, turn):
    s = _lone_checker_state(point, turn)
    d = (dice[1], dice[0]) if swap else dice
    plays = get_legal_plays(s, d)
    assert len(plays) == 1, f"pt{point} dice {d}: expected 1 play, got {plays}"
    play, succ = plays[0]
    # the surviving play must use BOTH dice (must-play-both rule)
    assert len(play) == 2
    assert succ.off[turn] == 15  # and it wins the game


@pytest.mark.parametrize("point,dice", CASES)
@pytest.mark.parametrize("swap", [False, True])
def test_c_fast_path_no_false_pass(point, dice, swap):
    bg_fast = pytest.importorskip("bg_fast")
    s = _lone_checker_state(point, WHITE)
    d = (dice[1], dice[0]) if swap else dice
    feats, lazy = bg_fast.get_legal_plays_encoded(s, d)
    assert len(lazy) == 1, f"pt{point} dice {d}: C path returned {len(lazy)}"


def test_normal_positions_unaffected():
    """Opening position: dedup change must not alter ordinary movegen."""
    s = BoardState.initial()
    for dice in [(3, 1), (6, 5), (4, 4), (2, 1)]:
        a = {ns._key() for _, ns in get_legal_plays(s, dice)}
        b = {ns._key() for _, ns in get_legal_plays(s, (dice[1], dice[0]))}
        assert a == b and len(a) > 0
