"""
bg_fast.py -- minimal Python wrapper for libbg_engine.so.

Exposes one function for TDAgent's fast path:

  get_legal_plays_encoded(state, dice, encoder=None)
    -> (features, next_states)

where `features` is (N, 196) float32 (perspective-encoded successors,
ready for the network) and `next_states` is a lazy view that
materialises a Python BoardState only when accessed by index.
Resulting states are already switch_turn'd to match the
backgammon_engine convention (`.turn` = next player to act).

The shared library is built/loaded lazily on first import; an
ImportError on failure lets callers fall back to backgammon_engine.

TODO (task #6 / #7 — cube + matchplay): the C encoder is hard-coded
to 196 perspective features. The cubeful experiments build their
199-feature input by appending 3 cube one-hot features in Python
after a 196 C encode call. When we add cube / matchplay support
here, follow that pattern: a thin `get_legal_plays_encoded_cubeful`
(and matchplay variant) that calls the existing C function and
appends the small features in Python. Verify the cube perspective
flip (MINE ↔ THEIRS) against cubeful_exp1 for the exact rule.
"""

import ctypes
import os
import subprocess

import numpy as np

from backgammon_engine import BoardState

NUM_FEATURES = 196
MAX_PLAYS = 512
MAX_MOVES_PER_PLAY = 4
NUM_POINTS = 24


# ── Library loading ─────────────────────────────────────────────────


def _find_or_build_library() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(here, "libbg_engine.so")
    source = os.path.join(here, "bg_engine.c")

    if (
        os.path.isfile(lib_path)
        and os.path.isfile(source)
        and os.path.getmtime(lib_path) >= os.path.getmtime(source)
    ):
        return lib_path

    cmd = [
        "gcc", "-O2", "-shared", "-fPIC",
        "-o", "libbg_engine.so",
        "bg_engine.c", "-Wall",
    ]
    try:
        subprocess.run(
            cmd, cwd=here, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise ImportError(
            f"Cannot build libbg_engine.so in {here}: {e}. "
            f"Build manually with: cd c_engine && bash build_unix.sh"
        )
    if not os.path.isfile(lib_path):
        raise ImportError(
            f"Built libbg_engine.so but cannot find it at {lib_path}"
        )
    return lib_path


_lib = ctypes.CDLL(_find_or_build_library())


# ── C struct definitions ────────────────────────────────────────────


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


# ── Function signatures ─────────────────────────────────────────────


_lib.get_legal_plays_encoded.argtypes = [
    ctypes.POINTER(_CBoardState),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_CPlay),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]
_lib.get_legal_plays_encoded.restype = ctypes.c_int

_lib.encode_state.argtypes = [
    ctypes.POINTER(_CBoardState),
    ctypes.POINTER(ctypes.c_float),
]
_lib.encode_state.restype = None


# ── Pre-allocated buffers ───────────────────────────────────────────
#
# These are reused across calls. Callers must consume the result
# (or copy what they need) before the next call.

_play_buf = (_CPlay * MAX_PLAYS)()
_feature_buf = np.zeros(MAX_PLAYS * NUM_FEATURES, dtype=np.float32)
_feature_buf_ptr = _feature_buf.ctypes.data_as(
    ctypes.POINTER(ctypes.c_float)
)

# Single-state encoding buffer (reused across calls).
_single_feature_buf = np.zeros(NUM_FEATURES, dtype=np.float32)
_single_feature_buf_ptr = _single_feature_buf.ctypes.data_as(
    ctypes.POINTER(ctypes.c_float)
)


# ── Python ↔ C state conversion (used internally) ──────────────────


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
    return BoardState(
        points=list(c_state.points),
        bar=[c_state.bar[0], c_state.bar[1]],
        off=[c_state.off[0], c_state.off[1]],
        turn=c_state.turn,
    )


# ── Public API ──────────────────────────────────────────────────────


_SUPPORTED_ENCODERS = frozenset({"perspective196"})


class _LazyNextStates:
    """List-like view over `_play_buf`. `view[i]` lazily materialises
    the i'th resulting state as a Python BoardState — one conversion
    per turn instead of N. Invalidated by the next
    `get_legal_plays_encoded` call (shared buffer); callers must
    consume synchronously.
    """

    __slots__ = ("_count",)

    def __init__(self, count: int):
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    def __getitem__(self, idx: int) -> BoardState:
        if idx < 0 or idx >= self._count:
            raise IndexError(idx)
        return _c_to_python(_play_buf[idx].resulting_state)


def encode_state(state: BoardState, encoder=None) -> np.ndarray:
    """Single-state perspective encoding via the C engine. Returns
    a fresh (196,) float32 numpy array (a copy of the internal
    buffer; safe to retain across calls).

    `encoder` is validated against `_SUPPORTED_ENCODERS` (same as
    `get_legal_plays_encoded`).
    """
    if encoder is not None:
        encoder_name = getattr(encoder, "name", None)
        if callable(encoder_name):
            encoder_name = encoder_name()
        if encoder_name not in _SUPPORTED_ENCODERS:
            raise ValueError(
                f"bg_fast only supports encoders "
                f"{sorted(_SUPPORTED_ENCODERS)}; got {encoder_name!r}."
            )
    c_state = _python_to_c(state)
    _lib.encode_state(ctypes.byref(c_state), _single_feature_buf_ptr)
    return _single_feature_buf.copy()


def get_legal_plays_encoded(state: BoardState, dice, encoder=None):
    """Move generation + perspective encoding in one C call.

    Args:
        state, dice: as for backgammon_engine.get_legal_plays_encoded.
        encoder: optional. If supplied its `name` must be in
                 `_SUPPORTED_ENCODERS` (currently {"perspective196"}).
                 The C encoder is hard-coded so unsupported encoders
                 raise ValueError rather than silently producing
                 wrong features.

    Returns `(features, next_states)`:
        features:    (N, 196) float32 (copy of internal buffer)
        next_states: lazy list-like view (see `_LazyNextStates`)

    Play info is not returned — the trainer discards it. Eval
    scripts that need Plays go through TDAgent's explicit-actions
    path.
    """
    if encoder is not None:
        encoder_name = getattr(encoder, "name", None)
        if callable(encoder_name):
            encoder_name = encoder_name()
        if encoder_name not in _SUPPORTED_ENCODERS:
            raise ValueError(
                f"bg_fast only supports encoders {sorted(_SUPPORTED_ENCODERS)}; "
                f"got {encoder_name!r}. Use the backgammon_engine Python path "
                f"for other encoders, or extend bg_fast with a new "
                f"C function and dispatch."
            )

    c_state = _python_to_c(state)
    d1, d2 = dice
    count = _lib.get_legal_plays_encoded(
        ctypes.byref(c_state),
        d1,
        d2,
        _play_buf,
        MAX_PLAYS,
        _feature_buf_ptr,
    )
    if count == 0:
        return np.empty((0, NUM_FEATURES), dtype=np.float32), _LazyNextStates(0)

    # Slice the feature buffer to (count, 196) and copy so the caller
    # can hold onto it across subsequent calls.
    features = _feature_buf[: count * NUM_FEATURES].reshape(count, NUM_FEATURES).copy()
    return features, _LazyNextStates(count)
