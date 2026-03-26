"""
bg_fast.py -- Python wrapper for the C backgammon engine via ctypes.

Drop-in replacement for backgammon_engine.py + encoding.py.
Provides the same API (BoardState, get_legal_plays, encode_state, etc.)
but backed by compiled C code for ~20-50x faster move generation.

Usage:
    # Instead of:
    #   from backgammon_engine import BoardState, get_legal_plays, ...
    #   from encoding import encode_state
    # Use:
    from bg_fast import BoardState, get_legal_plays, encode_state, ...

Build the shared library first:
    Windows:  build_win.bat
    Linux:    ./build_linux.sh
"""

import ctypes
import os
import platform
import subprocess
import sys
import numpy as np
from typing import List, Tuple, Optional

# ── Constants (matching backgammon_engine.py) ────────────────────

WHITE = 0
BLACK = 1
NUM_POINTS = 24
NUM_CHECKERS = 15
BAR = -1
OFF = -2

Move = Tuple[int, int]
Play = Tuple[Move, ...]

# ── Load shared library ─────────────────────────────────────────

def _auto_build(here):
    """Try to compile the C engine automatically. Returns True on success."""
    source = os.path.join(here, "bg_engine.c")
    if not os.path.isfile(source):
        return False
    if platform.system() == "Windows":
        lib_name = "bg_engine.dll"
        cmd = ["gcc", "-O2", "-shared", "-o", lib_name, "bg_engine.c", "-Wall"]
    else:
        lib_name = "libbg_engine.so"
        cmd = ["gcc", "-O2", "-shared", "-fPIC", "-o", lib_name, "bg_engine.c", "-Wall"]
    try:
        subprocess.run(cmd, cwd=here, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _find_library():
    """Find the compiled shared library, auto-building if missing or stale."""
    here = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(here, "bg_engine.c")
    if platform.system() == "Windows":
        names = ["bg_engine.dll", "libbg_engine.dll"]
    else:
        names = ["libbg_engine.so"]

    # Check if library exists and is up-to-date
    for name in names:
        path = os.path.join(here, name)
        if os.path.isfile(path):
            # Rebuild if source is newer than binary
            if os.path.isfile(source) and os.path.getmtime(source) > os.path.getmtime(path):
                print("C engine source is newer than binary, rebuilding...", flush=True)
                if _auto_build(here):
                    print("C engine rebuilt successfully.", flush=True)
                else:
                    print("Warning: auto-rebuild failed, using stale binary.", flush=True)
            return path

    # Also check parent directory
    parent = os.path.dirname(here)
    for name in names:
        path = os.path.join(parent, name)
        if os.path.isfile(path):
            return path

    # Not found — try to build
    print("C engine binary not found, building...", flush=True)
    if _auto_build(here):
        print("C engine built successfully.", flush=True)
        for name in names:
            path = os.path.join(here, name)
            if os.path.isfile(path):
                return path

    raise FileNotFoundError(
        f"Cannot find or build C engine. Install gcc and retry, or build manually:\n"
        f"  Windows: cd c_engine && build_win.bat\n"
        f"  Linux:   cd c_engine && ./build_linux.sh"
    )


_lib = ctypes.CDLL(_find_library())

# ── C struct definitions ─────────────────────────────────────────

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
        ("moves", _CMove * 4),
        ("num_moves", ctypes.c_int),
        ("resulting_state", _CBoardState),
    ]


# ── Set up function signatures ───────────────────────────────────

_lib.board_init.argtypes = [ctypes.POINTER(_CBoardState)]
_lib.board_init.restype = None

_lib.board_is_game_over.argtypes = [ctypes.POINTER(_CBoardState)]
_lib.board_is_game_over.restype = ctypes.c_int

_lib.board_winner.argtypes = [ctypes.POINTER(_CBoardState)]
_lib.board_winner.restype = ctypes.c_int

_lib.board_switch_turn.argtypes = [ctypes.POINTER(_CBoardState)]
_lib.board_switch_turn.restype = None

_lib.get_legal_plays.argtypes = [
    ctypes.POINTER(_CBoardState),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_CPlay),
    ctypes.c_int,
]
_lib.get_legal_plays.restype = ctypes.c_int

_lib.encode_state.argtypes = [
    ctypes.POINTER(_CBoardState),
    ctypes.POINTER(ctypes.c_float),
]
_lib.encode_state.restype = None

_lib.get_legal_plays_encoded.argtypes = [
    ctypes.POINTER(_CBoardState),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_CPlay),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]
_lib.get_legal_plays_encoded.restype = ctypes.c_int

_lib.get_play_resulting_state.argtypes = [
    ctypes.POINTER(_CPlay),
    ctypes.c_int,
    ctypes.POINTER(_CBoardState),
]
_lib.get_play_resulting_state.restype = None

_lib.encode_state_210.argtypes = [
    ctypes.POINTER(_CBoardState),
    ctypes.POINTER(ctypes.c_float),
]
_lib.encode_state_210.restype = None

_lib.get_legal_plays_encoded_210.argtypes = [
    ctypes.POINTER(_CBoardState),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_CPlay),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]
_lib.get_legal_plays_encoded_210.restype = ctypes.c_int

_lib.encode_state_224.argtypes = [
    ctypes.POINTER(_CBoardState),
    ctypes.POINTER(ctypes.c_float),
]
_lib.encode_state_224.restype = None

_lib.get_legal_plays_encoded_224.argtypes = [
    ctypes.POINTER(_CBoardState),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_CPlay),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]
_lib.get_legal_plays_encoded_224.restype = ctypes.c_int

_lib.encode_state_246.argtypes = [
    ctypes.POINTER(_CBoardState),
    ctypes.POINTER(ctypes.c_float),
]
_lib.encode_state_246.restype = None

_lib.get_legal_plays_encoded_246.argtypes = [
    ctypes.POINTER(_CBoardState),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_CPlay),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]
_lib.get_legal_plays_encoded_246.restype = ctypes.c_int

# ── Pre-allocate buffers ─────────────────────────────────────────

_MAX_PLAYS = 512
_play_buf = (_CPlay * _MAX_PLAYS)()
# Pre-allocated feature buffers
_feature_buf = np.zeros(_MAX_PLAYS * 196, dtype=np.float32)
_feature_buf_210 = np.zeros(_MAX_PLAYS * 210, dtype=np.float32)
_feature_buf_224 = np.zeros(_MAX_PLAYS * 224, dtype=np.float32)
_feature_buf_246 = np.zeros(_MAX_PLAYS * 246, dtype=np.float32)

# ── BoardState wrapper ───────────────────────────────────────────

class BoardState:
    """Python wrapper around the C BoardState struct.

    API-compatible with backgammon_engine.BoardState.
    """

    __slots__ = ("_c",)

    def __init__(self, points=None, bar=None, off=None, turn=WHITE):
        self._c = _CBoardState()
        if points is not None:
            for i in range(NUM_POINTS):
                self._c.points[i] = points[i]
        if bar is not None:
            self._c.bar[0] = bar[0]
            self._c.bar[1] = bar[1]
        if off is not None:
            self._c.off[0] = off[0]
            self._c.off[1] = off[1]
        self._c.turn = turn

    @classmethod
    def initial(cls) -> "BoardState":
        s = cls.__new__(cls)
        s._c = _CBoardState()
        _lib.board_init(ctypes.byref(s._c))
        return s

    @classmethod
    def _from_c(cls, c_state: _CBoardState) -> "BoardState":
        """Wrap an existing C struct (copies it)."""
        s = cls.__new__(cls)
        s._c = _CBoardState()
        ctypes.memmove(ctypes.byref(s._c), ctypes.byref(c_state),
                       ctypes.sizeof(_CBoardState))
        return s

    def copy(self) -> "BoardState":
        return BoardState._from_c(self._c)

    # ── Properties for compatibility ─────────────────────────────

    @property
    def points(self):
        return list(self._c.points)

    @property
    def bar(self):
        return [self._c.bar[0], self._c.bar[1]]

    @property
    def off(self):
        return [self._c.off[0], self._c.off[1]]

    @property
    def turn(self):
        return self._c.turn

    @turn.setter
    def turn(self, value):
        self._c.turn = value

    # ── Queries ──────────────────────────────────────────────────

    def is_game_over(self) -> bool:
        return bool(_lib.board_is_game_over(ctypes.byref(self._c)))

    def winner(self) -> Optional[int]:
        w = _lib.board_winner(ctypes.byref(self._c))
        return w if w >= 0 else None

    def game_result(self) -> Optional[int]:
        w = self.winner()
        if w is None:
            return None
        loser = 1 - w
        if self._c.off[loser] > 0:
            return 1  # normal
        # Check for backgammon
        if loser == WHITE:
            loser_in_opp_home = any(self._c.points[i] > 0 for i in range(18, 24))
        else:
            loser_in_opp_home = any(self._c.points[i] < 0 for i in range(0, 6))
        if self._c.bar[loser] > 0 or loser_in_opp_home:
            return 3
        return 2

    def checker_count(self, player: int, idx: int) -> int:
        v = self._c.points[idx]
        if player == WHITE:
            return max(v, 0)
        else:
            return max(-v, 0)

    # ── Display ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"BoardState(turn={'W' if self.turn == WHITE else 'B'}, "
            f"bar={self.bar}, off={self.off})"
        )


# ── Free functions ───────────────────────────────────────────────

def switch_turn(state: BoardState) -> BoardState:
    """Return a copy with the turn flipped."""
    s = state.copy()
    _lib.board_switch_turn(ctypes.byref(s._c))
    return s


def get_legal_plays(
    state: BoardState, dice: Tuple[int, int]
) -> List[Tuple[Play, BoardState]]:
    """Return all legal plays as [(play, resulting_state), ...].

    API-compatible with backgammon_engine.get_legal_plays.
    """
    d1, d2 = dice
    count = _lib.get_legal_plays(
        ctypes.byref(state._c), d1, d2, _play_buf, _MAX_PLAYS
    )

    if count == 0:
        return []

    result = []
    for i in range(count):
        cp = _play_buf[i]

        # Convert C moves to Python tuples
        moves = tuple(
            (cp.moves[j].src, cp.moves[j].dst)
            for j in range(cp.num_moves)
        )

        # Wrap resulting state
        rs = BoardState._from_c(cp.resulting_state)

        result.append((moves, rs))

    return result


def encode_state(state: BoardState) -> np.ndarray:
    """Encode a board state as a 196-element float32 vector."""
    features = np.zeros(196, dtype=np.float32)
    _lib.encode_state(
        ctypes.byref(state._c),
        features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return features


# ── Move labels (for display/debugging) ──────────────────────────

def move_label(move: Move) -> str:
    src, dst = move
    src_str = "bar" if src == BAR else str(src + 1)
    dst_str = "off" if dst == OFF else str(dst + 1)
    return f"{src_str}/{dst_str}"


def play_label(play: Play) -> str:
    return "  ".join(move_label(m) for m in play)


# ── Optimized training API ───────────────────────────────────────
#
# These functions minimize Python<->C round trips by doing all the
# heavy work (move gen + encoding) in a single C call, returning
# a numpy array ready for PyTorch.

def get_plays_and_features(
    state: BoardState, dice: Tuple[int, int]
) -> Tuple[int, np.ndarray]:
    """Get legal plays and encode all resulting states in one C call.

    Returns (num_plays, features) where features is a numpy array
    of shape (num_plays, 196), ready to be wrapped as a torch tensor.

    The plays are kept in an internal C buffer. After choosing the
    best index, call get_chosen_state(index) to retrieve that state.

    This is ~10x faster than calling get_legal_plays() + encode_state()
    separately because it avoids per-play Python<->C round trips.
    """
    d1, d2 = dice
    count = _lib.get_legal_plays_encoded(
        ctypes.byref(state._c), d1, d2,
        _play_buf, _MAX_PLAYS,
        _feature_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    if count == 0:
        return 0, np.empty((0, 196), dtype=np.float32)

    # Return a VIEW into the pre-allocated buffer (no copy).
    # The caller must use this before the next get_plays_and_features call.
    features = _feature_buf[:count * 196].reshape(count, 196)
    return count, features


def get_chosen_state(index: int) -> BoardState:
    """Retrieve the resulting BoardState for the play at *index*.

    Call this after get_plays_and_features() to get the state
    corresponding to the chosen play.
    """
    out = BoardState.__new__(BoardState)
    out._c = _CBoardState()
    _lib.get_play_resulting_state(_play_buf, index, ctypes.byref(out._c))
    return out


def encode_single(state: BoardState) -> np.ndarray:
    """Encode a single board state. Returns a 196-element float32 array.

    Same as encode_state() but uses the pre-allocated buffer to avoid
    a numpy allocation.
    """
    _lib.encode_state(
        ctypes.byref(state._c),
        _feature_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    # Return a copy since the buffer will be reused
    return _feature_buf[:196].copy()


def switch_turn_inplace(state: BoardState):
    """Flip the turn in place (no copy)."""
    _lib.board_switch_turn(ctypes.byref(state._c))


# ── 210-feature (gnubg_group1) encoding API ──────────────────────

def encode_state_210(state: BoardState) -> np.ndarray:
    """Encode a board state as a 210-element float32 vector (196 base + 14 gnubg Group-1)."""
    features = np.zeros(210, dtype=np.float32)
    _lib.encode_state_210(
        ctypes.byref(state._c),
        features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return features


def get_plays_and_features_210(
    state: BoardState, dice: Tuple[int, int]
) -> Tuple[int, np.ndarray]:
    """Like get_plays_and_features() but with 210-feature gnubg_group1 encoding."""
    d1, d2 = dice
    count = _lib.get_legal_plays_encoded_210(
        ctypes.byref(state._c), d1, d2,
        _play_buf, _MAX_PLAYS,
        _feature_buf_210.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    if count == 0:
        return 0, np.empty((0, 210), dtype=np.float32)

    features = _feature_buf_210[:count * 210].reshape(count, 210)
    return count, features


# ── 224-feature (gnubg_group2) encoding API ──────────────────────

def encode_state_224(state: BoardState) -> np.ndarray:
    """Encode a board state as a 224-element float32 vector (196 base + 14 Group-1 + 14 Group-2)."""
    features = np.zeros(224, dtype=np.float32)
    _lib.encode_state_224(
        ctypes.byref(state._c),
        features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return features


def get_plays_and_features_224(
    state: BoardState, dice: Tuple[int, int]
) -> Tuple[int, np.ndarray]:
    """Like get_plays_and_features() but with 224-feature gnubg_group2 encoding."""
    d1, d2 = dice
    count = _lib.get_legal_plays_encoded_224(
        ctypes.byref(state._c), d1, d2,
        _play_buf, _MAX_PLAYS,
        _feature_buf_224.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    if count == 0:
        return 0, np.empty((0, 224), dtype=np.float32)

    features = _feature_buf_224[:count * 224].reshape(count, 224)
    return count, features


# ── 246-feature (gnubg_group3) encoding API ──────────────────────

def encode_state_246(state: BoardState) -> np.ndarray:
    """Encode a board state as a 246-element float32 vector (all gnubg contact features)."""
    features = np.zeros(246, dtype=np.float32)
    _lib.encode_state_246(
        ctypes.byref(state._c),
        features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return features


def get_plays_and_features_246(
    state: BoardState, dice: Tuple[int, int]
) -> Tuple[int, np.ndarray]:
    """Like get_plays_and_features() but with 246-feature gnubg_group3 encoding."""
    d1, d2 = dice
    count = _lib.get_legal_plays_encoded_246(
        ctypes.byref(state._c), d1, d2,
        _play_buf, _MAX_PLAYS,
        _feature_buf_246.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    if count == 0:
        return 0, np.empty((0, 246), dtype=np.float32)

    features = _feature_buf_246[:count * 246].reshape(count, 246)
    return count, features
