"""A shorter game-ending play must not shadow the maximal play.

A 1-die bear-off (e.g. 2/off with the 2 of a 21) can reach the same final
board as the full both-dice play (2/1, 1/off). The board-key dedup must keep
the maximal one, or the maximal-dice filter drops the position and reports a
false forced pass. Both engines (Python get_legal_plays and the C fast path)
must return the play in BOTH dice orders.
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
    assert succ.off[turn] == 15


@pytest.mark.parametrize("point,dice", CASES)
@pytest.mark.parametrize("swap", [False, True])
def test_c_fast_path_no_false_pass(point, dice, swap):
    bg_fast = pytest.importorskip("bg_fast")
    s = _lone_checker_state(point, WHITE)
    d = (dice[1], dice[0]) if swap else dice
    feats, lazy = bg_fast.get_legal_plays_encoded(s, d)
    assert len(lazy) == 1, f"pt{point} dice {d}: C path returned {len(lazy)}"


def test_normal_positions_unaffected():
    """Opening position: the board-key dedup leaves ordinary movegen unchanged."""
    s = BoardState.initial()
    for dice in [(3, 1), (6, 5), (4, 4), (2, 1)]:
        a = {ns._key() for _, ns in get_legal_plays(s, dice)}
        b = {ns._key() for _, ns in get_legal_plays(s, (dice[1], dice[0]))}
        assert a == b and len(a) > 0


# ── Doubles walk: pruned nodes are not leaves ────────────────────────
#
# The C generator keeps one ordering per move multiset for doubles (the
# one the unpruned walk finds first); a node whose legal moves are ALL
# pruned has legal moves, so it is not a leaf and must not add a shorter
# play. Example traced through the walk: WHITE on points 1 and 3, 1-1.
# Moving 3/2 first forbids 1/off below it (1/off was already legal);
# after 3/2 2/1 the only legal moves are 1/off x2, both forbidden: a node
# with 2 dice used and every move pruned. The single legal play is still
# the 4-move bear-off of both checkers (3/2/1/off 1/off). Exhaustively:
# every 2-3 checker bear-off (mover's home board only), with and without
# opponent points / blots in that home board, both colors, all doubles:
# C == the Python reference (same plays, same order, same moves) and
# every play uses the maximum number of dice.

def _home_configs(n):
    from itertools import combinations_with_replacement
    return list(combinations_with_replacement(range(6), n))


# opponent pieces inside the mover's home (WHITE-relative indices 0-5):
# none / a made point / a made point + a blot / two made points
_OPP_IN_HOME = [(), ((3, 2),), ((4, 2), (1, 1)), ((2, 2), (5, 2))]


def _bearoff_state(cfg, opp, turn):
    pts = [0] * 24
    for i in cfg:                       # mover, WHITE-relative index
        j = i if turn == WHITE else 23 - i
        pts[j] += 1 if turn == WHITE else -1
    n_opp = 0
    for i, k in opp:
        j = i if turn == WHITE else 23 - i
        if pts[j] != 0:
            return None
        pts[j] = -k if turn == WHITE else k
        n_opp += k
    # rest of the opponent far from the mover's home
    far = 18 if turn == WHITE else 5
    pts[far] = -(15 - n_opp) if turn == WHITE else (15 - n_opp)
    off = [0, 0]
    off[turn] = 15 - len(cfg)
    return BoardState(points=pts, bar=[0, 0], off=off, turn=turn)


def test_traced_all_pruned_node_case():
    bg_fast = pytest.importorskip("bg_fast")
    for turn in (WHITE, BLACK):
        s = _bearoff_state((0, 2), (), turn)
        py = get_legal_plays(s, (1, 1))
        assert len(py) == 1 and len(py[0][0]) == 4 and py[0][1].off[turn] == 15
        feats, lazy = bg_fast.get_legal_plays_encoded(s, (1, 1))
        assert len(lazy) == 1 and lazy[0].off[turn] == 15


@pytest.mark.parametrize("n_checkers", [2, 3])
def test_doubles_bearoff_only_maximal_plays(n_checkers):
    import ctypes
    bg_fast = pytest.importorskip("bg_fast")
    lib = ctypes.CDLL(os.environ.get("BG_MOVEGEN_LIB") or bg_fast._lib._name)
    lib.get_legal_plays.argtypes = [
        ctypes.POINTER(bg_fast._CBoardState), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(bg_fast._CPlay), ctypes.c_int]
    lib.get_legal_plays.restype = ctypes.c_int
    buf = (bg_fast._CPlay * bg_fast.MAX_PLAYS)()
    checked = short = 0
    for cfg in _home_configs(n_checkers):
        for opp in _OPP_IN_HOME:
            for turn in (WHITE, BLACK):
                s = _bearoff_state(cfg, opp, turn)
                if s is None:
                    continue
                c = bg_fast._python_to_c(s)
                for d in range(1, 7):
                    py = get_legal_plays(s, (d, d))
                    n = lib.get_legal_plays(ctypes.byref(c), d, d, buf, bg_fast.MAX_PLAYS)
                    assert n == len(py), f"{cfg} opp={opp} turn={turn} {d}{d}: C {n} vs Python {len(py)}"
                    if not py:
                        continue
                    # the Python walk's most_dice_used (it keeps only maximal plays)
                    most = max(len(p) for p, _ in py)
                    for k, (play, ns) in enumerate(py):
                        cp = buf[k]
                        assert cp.num_moves == len(play) == most
                        rs = cp.resulting_state
                        assert (list(rs.points), list(rs.bar), list(rs.off), rs.turn) == (
                            ns.points, ns.bar, ns.off, ns.turn), \
                            f"{cfg} opp={opp} turn={turn} {d}{d}: play {k} differs"
                        assert [(cp.moves[i].src, cp.moves[i].dst)
                                for i in range(cp.num_moves)] == list(play)
                    short += most < 4
                    checked += 1
    assert checked > 200 and short > 0   # some rolls cannot use all 4 dice
