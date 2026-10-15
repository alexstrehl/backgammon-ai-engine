"""
bg_fast.py -- ctypes wrapper for libbg_engine.so (C move generation +
196-feature perspective encoding).

  get_legal_plays_encoded(state, dice, encoder=None)
    -> (features, next_states)

`features` is (N, 196) float32; `next_states` is a lazy view that
builds a Python BoardState on index access. Resulting states follow
the backgammon_engine convention (`.turn` = next player to act).

The library is built if missing or stale at import. A failure raises
ImportError so callers can fall back to backgammon_engine. Cubeful
models append the cube features in Python after the C encode.
"""

import ctypes
import os
import subprocess

import numpy as np

from backgammon_engine import BoardState
from bg_cstructs import (  # noqa: F401 -- re-exported
    MAX_MOVES_PER_PLAY, MAX_PLAYS, NUM_POINTS, _CBoardState, _CMove, _CPlay,
    _c_to_python, _python_to_c,
)

NUM_FEATURES = 196


# ── Library loading ─────────────────────────────────────────────────


def _find_or_build_library() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(here, "libbg_engine.so")
    source = os.path.join(here, "bg_engine.c")
    # Staleness must include the header: a .so built against an old
    # bg_engine.h (e.g. MAX_PLAYS) is silently wrong, not a build error.
    deps = [source, os.path.join(here, "bg_engine.h")]

    if (
        os.path.isfile(lib_path)
        and os.path.isfile(source)
        and os.path.getmtime(lib_path) >= max(
            os.path.getmtime(d) for d in deps if os.path.isfile(d)
        )
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

_lib.prob5_impossible_mask.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
_lib.prob5_impossible_mask.restype = ctypes.c_int

_HAS_EXPAND21 = hasattr(_lib, "get_legal_plays_encoded_21")
if _HAS_EXPAND21:
    _lib.get_legal_plays_encoded_21.argtypes = [
        ctypes.POINTER(_CBoardState),
        ctypes.POINTER(ctypes.c_float),  # out_feats
        ctypes.POINTER(ctypes.c_int),    # out_dice_counts[21]
        ctypes.POINTER(ctypes.c_int),    # out_reply_gr[max_replies]
        ctypes.c_int,                    # max_rows
        ctypes.c_int,                    # max_replies
    ]
    _lib.get_legal_plays_encoded_21.restype = ctypes.c_int


# ── Pre-allocated buffers ───────────────────────────────────────────
#
# Reused across calls: consume or copy the result before the next call.

_play_buf = (_CPlay * MAX_PLAYS)()
_feature_buf = np.zeros(MAX_PLAYS * NUM_FEATURES, dtype=np.float32)
_feature_buf_ptr = _feature_buf.ctypes.data_as(
    ctypes.POINTER(ctypes.c_float)
)

_single_feature_buf = np.zeros(NUM_FEATURES, dtype=np.float32)
_single_feature_buf_ptr = _single_feature_buf.ctypes.data_as(
    ctypes.POINTER(ctypes.c_float)
)


# `_play_buf` as flat int32: play i's resulting_state (points[24],
# bar[2], off[2], turn) is `_play_ints[i * _PLAY_INTS + _STATE_COL:][:29]`.
_PLAY_INTS = ctypes.sizeof(_CPlay) // 4
_STATE_COL = _CPlay.resulting_state.offset // 4
if (_PLAY_INTS * 4 != ctypes.sizeof(_CPlay)
        or ctypes.sizeof(_CBoardState) != 29 * 4):
    raise ImportError("bg_fast: c_int is not 4 bytes; use backgammon_engine")
_play_ints = memoryview(_play_buf).cast("B").cast("i")


# ── Public API ──────────────────────────────────────────────────────


_SUPPORTED_ENCODERS = frozenset({"perspective196"})


class _LazyNextStates:
    """List-like view over `_play_buf`; `view[i]` converts the i'th
    resulting state on demand. Invalidated by the next
    `get_legal_plays_encoded` call (shared buffer)."""

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
        o = idx * _PLAY_INTS + _STATE_COL
        v = _play_ints[o:o + 29].tolist()
        return BoardState(v[:24], v[24:26], v[26:28], v[28])


def encode_state(state: BoardState, encoder=None) -> np.ndarray:
    """Encode one state via the C engine. Returns a fresh (196,)
    float32 array. `encoder`, if given, must be in `_SUPPORTED_ENCODERS`."""
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


def prob5_impossible(x: np.ndarray):
    """C prob_agent.prob5_impossible on a C-contiguous (N, 196) float32
    array: (N, 4) bool mask, or None when nothing is impossible."""
    mask = np.empty((x.shape[0], 4), dtype=np.bool_)
    if _lib.prob5_impossible_mask(x.ctypes.data, x.shape[0], mask.ctypes.data):
        return mask
    return None


def get_legal_plays_encoded(state: BoardState, dice, encoder=None):
    """Move generation + perspective encoding in one C call.

    `encoder`, if given, must be in `_SUPPORTED_ENCODERS` (the C
    encoder is fixed; anything else raises ValueError).

    Returns `(features, next_states)`:
        features:    (N, 196) float32 (a copy)
        next_states: lazy view (see `_LazyNextStates`)

    Plays are not returned. Callers that need them use TDAgent's
    explicit-actions path.
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

    features = _feature_buf[: count * NUM_FEATURES].reshape(count, NUM_FEATURES).copy()
    return features, _LazyNextStates(count)


# ── 1-ply inner expansion (all 21 opponent dice in one C call) ──────

# Buffers for expand_21 (~47 MB, allocated once). On overflow the C
# call returns -1 and the caller falls back to the per-dice path.
_EXP21_MAX_ROWS = 60000
_EXP21_MAX_REPLIES = 80000
if _HAS_EXPAND21:
    _exp21_feats = np.zeros(_EXP21_MAX_ROWS * NUM_FEATURES, dtype=np.float32)
    _exp21_feats_ptr = _exp21_feats.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    _exp21_counts = np.zeros(21, dtype=np.int32)
    _exp21_counts_ptr = _exp21_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    _exp21_gr = np.zeros(_EXP21_MAX_REPLIES, dtype=np.int32)
    _exp21_gr_ptr = _exp21_gr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def expand_21(next_state: BoardState):
    """1-ply inner expansion over all 21 opponent rolls in one C call.

    Returns `(base_feats, dice_counts, reply_gr, n_rows)`:
        base_feats:  (n_rows, 196) float32, non-terminal replies in
                     reply order.
        dice_counts: (21,) int32, replies per roll (>=1).
        reply_gr:    (n_replies,) int32; 0 = non-terminal (takes the
                     next base_feats row), 1/2/3 = terminal (no row).
        n_rows:      number of base_feats rows.

    Returns None if the C function is missing or a buffer overflowed
    (fall back to the per-dice path).
    """
    if not _HAS_EXPAND21:
        return None
    c_state = _python_to_c(next_state)
    n_replies = _lib.get_legal_plays_encoded_21(
        ctypes.byref(c_state),
        _exp21_feats_ptr,
        _exp21_counts_ptr,
        _exp21_gr_ptr,
        _EXP21_MAX_ROWS,
        _EXP21_MAX_REPLIES,
    )
    if n_replies < 0:
        return None
    gr = _exp21_gr[:n_replies].copy()
    n_rows = int(np.count_nonzero(gr == 0))
    base_feats = (
        _exp21_feats[: n_rows * NUM_FEATURES]
        .reshape(n_rows, NUM_FEATURES)
        .copy()
    )
    return base_feats, _exp21_counts.copy(), gr, n_rows
