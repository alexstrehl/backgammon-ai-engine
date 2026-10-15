"""bg_cstructs.py -- ctypes layouts of bg_engine.h's board_state / play
and the Python <-> C state conversion. Loads no library, so the C API
binding (c_api/puretd_py.py) imports it without libbg_engine.so."""

import ctypes

from backgammon_engine import BoardState

MAX_PLAYS = 4096  # keep in sync with bg_engine.h MAX_PLAYS
MAX_MOVES_PER_PLAY = 4
NUM_POINTS = 24


class _CBoardState(ctypes.Structure):
    _fields_ = [
        ("points", ctypes.c_int * NUM_POINTS),
        ("bar", ctypes.c_int * 2),
        ("off", ctypes.c_int * 2),
        ("turn", ctypes.c_int),
    ]


class _CMove(ctypes.Structure):
    _fields_ = [
        ("src", ctypes.c_int),
        ("dst", ctypes.c_int),
    ]


class _CPlay(ctypes.Structure):
    _fields_ = [
        ("moves", _CMove * MAX_MOVES_PER_PLAY),
        ("num_moves", ctypes.c_int),
        ("resulting_state", _CBoardState),
    ]


def _python_to_c(state: BoardState) -> _CBoardState:
    c = _CBoardState()
    pts = state.points
    for i in range(NUM_POINTS):
        c.points[i] = pts[i]
    c.bar[0] = state.bar[0]
    c.bar[1] = state.bar[1]
    c.off[0] = state.off[0]
    c.off[1] = state.off[1]
    c.turn = state.turn
    return c


def _c_to_python(c_state: _CBoardState) -> BoardState:
    v = memoryview(c_state).cast("B").cast("i").tolist()
    return BoardState(v[:24], v[24:26], v[26:28], v[28])
