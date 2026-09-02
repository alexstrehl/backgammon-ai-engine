"""bg_nply -- Python binding for libbg_nply.so (fused move-gen + nn-eval
N-ply search in C). Zero PyTorch in the inner loop.

Load a model once with `load_model(path_to_bin)` (a .bin exported by
export_weights.py), then call the value/move functions with Python
BoardState objects. Cube perspective / cube-action / gammons-count flags
follow the same convention as td_agent.py's cubeful methods.
"""
import atexit
import ctypes
import os
import shutil
import tempfile

from backgammon_engine import BoardState, switch_turn  # noqa
from c_engine.bg_fast import _CBoardState, _python_to_c  # reuse layout

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "libbg_nply.so")

_lib = None
_loaded_path = None
_export_dir = None


def _private_export_path() -> str:
    """Per-process weight-export path inside a freshly created 0700 temp
    dir (unpredictable name, unreadable by other users, removed at exit).
    Model weights must never land at a predictable world-readable path."""
    global _export_dir
    if _export_dir is None:
        _export_dir = tempfile.mkdtemp(prefix="bgn_w_")
        atexit.register(shutil.rmtree, _export_dir, ignore_errors=True)
    return os.path.join(_export_dir, "weights.bin")


def _ensure_lib():
    global _lib
    if _lib is None:
        if not os.path.exists(_SO):
            raise RuntimeError(
                f"{_SO} not found; build it with c_engine/build_nply.sh")
        _lib = ctypes.CDLL(_SO)
        P = ctypes.POINTER
        F = ctypes.c_float
        BS = P(_CBoardState)
        _lib.bgn_load.argtypes = [ctypes.c_char_p]; _lib.bgn_load.restype = ctypes.c_int
        _lib.bgn_unload.argtypes = []; _lib.bgn_unload.restype = None
        _lib.bgn_eval_cubeful.argtypes = [BS, ctypes.c_int, ctypes.c_int]
        _lib.bgn_eval_cubeful.restype = F
        _lib.bgn_value_1ply_cubeful.argtypes = [BS, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _lib.bgn_value_1ply_cubeful.restype = F
        _lib.bgn_value_1ply_cubeless.argtypes = [BS]
        _lib.bgn_value_1ply_cubeless.restype = F
        _lib.bgn_best_move_0ply_cubeful.argtypes = [
            BS, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            BS, P(F)]
        _lib.bgn_best_move_0ply_cubeful.restype = ctypes.c_int
        _lib.bgn_value_2ply_cubeful.argtypes = [BS] + [ctypes.c_int] * 11
        _lib.bgn_value_2ply_cubeful.restype = F
        _lib.bgn_value_1ply_generic.argtypes = [BS, ctypes.c_int, ctypes.c_int]
        _lib.bgn_value_1ply_generic.restype = F
        _lib.bgn_value_2ply_generic.argtypes = [BS, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _lib.bgn_value_2ply_generic.restype = F
        _lib.bgn_forward5.argtypes = [P(F), ctypes.c_int, P(F)]
        _lib.bgn_forward5.restype = ctypes.c_int
    return _lib


# flip/terminal modes (must match bg_nply.c)
FLIP_EQUITY, FLIP_PROB = 0, 1
TERM_GAMMON, TERM_WIN = 0, 1
# convenience presets
DMP = (FLIP_PROB, TERM_WIN)       # probability win/lose
MONEY = (FLIP_EQUITY, TERM_GAMMON)  # signed money equity (prob5 / cubeless-money)


def value_1ply_generic(state: BoardState, flip_mode: int, term_mode: int) -> float:
    c = _python_to_c(state)
    return float(_lib.bgn_value_1ply_generic(ctypes.byref(c), int(flip_mode), int(term_mode)))


def value_2ply_generic(state: BoardState, k: int, flip_mode: int, term_mode: int) -> float:
    c = _python_to_c(state)
    return float(_lib.bgn_value_2ply_generic(ctypes.byref(c), int(k), int(flip_mode), int(term_mode)))


def forward5(feats):
    """Batched raw 5-output forward for a loaded prob5 model.

    `feats` is an (n, 196) float32 array of perspective196 encodings; the
    returned (n, 5) array holds the raw sigmoid outputs (no nested-event
    postprocess, no scalar reduce) -- matching ProbNetwork(x) for a
    sigmoid-output net. Use to drive prob5 1-ply target generation through
    the C engine while keeping the Python flip/postprocess/equity logic.
    """
    import numpy as np
    lib = _ensure_lib()
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[1] != 196:
        raise ValueError(f"forward5 expects (n,196), got {feats.shape}")
    n = feats.shape[0]
    out = np.empty((n, 5), dtype=np.float32)
    if n == 0:
        return out
    rc = lib.bgn_forward5(
        feats.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(n),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if rc != 0:
        raise RuntimeError(f"bgn_forward5 failed rc={rc}")
    return out


def load_model(bin_path: str) -> None:
    """Load a .bin model into the C engine (replaces any previous)."""
    global _loaded_path
    lib = _ensure_lib()
    rc = lib.bgn_load(bin_path.encode())
    if rc != 0:
        raise RuntimeError(f"bgn_load({bin_path}) failed rc={rc}")
    _loaded_path = bin_path


def enable_c_engine(agent, twoply_k: int = 0):
    """Route an agent's cubeful 1-ply (and optional 2-ply) checker TARGET
    through libbg_nply. Exports the agent's current weights to a
    per-process .bin, loads it, and monkeypatches the target methods. Call
    once per round after the agent's weights are refreshed. Returns True on
    success, False if the C engine is unavailable (caller keeps PyTorch).
    """
    from export_weights import export_live
    from modes import cube_perspective, CubeOwner
    try:
        _ensure_lib()
    except Exception:
        return False
    binp = _private_export_path()
    export_live(agent.network, binp)
    load_model(binp)

    def _v1(state, match_state):
        opp = 1 - state.turn
        return value_1ply_cubeful(
            state, int(cube_perspective(match_state.cube_owner, opp)),
            match_state.can_offer(opp),
            (not match_state.jacoby
             or match_state.cube_owner != CubeOwner.CENTERED))
    agent.value_oneply_checker_cubeful = _v1

    if twoply_k > 0:
        def _v2(state, match_state, k=twoply_k):
            return value_2ply_cubeful(state, match_state, k=k)
        agent.value_twoply_checker_cubeful_filtered = _v2
    return True


def eval_cubeful(state: BoardState, cube_persp: int, is_cube_action: bool) -> float:
    c = _python_to_c(state)
    return float(_lib.bgn_eval_cubeful(ctypes.byref(c), int(cube_persp),
                                       1 if is_cube_action else 0))


def value_1ply_cubeful(state: BoardState, opp_persp: int,
                       opp_cube_action: bool, gammons_count: bool) -> float:
    c = _python_to_c(state)
    return float(_lib.bgn_value_1ply_cubeful(
        ctypes.byref(c), int(opp_persp),
        1 if opp_cube_action else 0, 1 if gammons_count else 0))


def value_1ply_cubeless(state: BoardState) -> float:
    c = _python_to_c(state)
    return float(_lib.bgn_value_1ply_cubeless(ctypes.byref(c)))


def value_2ply_cubeful(state: BoardState, match_state, k: int = 3) -> float:
    """C 2-ply cubeful value, parity with
    TDAgent.value_twoply_checker_cubeful_filtered. Cube params are derived
    here (constant across successors, which all share the opponent on roll)."""
    from modes import cube_perspective, CubeOwner
    player = state.turn
    opp = 1 - player
    gammons = (not match_state.jacoby
               or match_state.cube_owner != CubeOwner.CENTERED)
    succ_persp = int(cube_perspective(match_state.cube_owner, opp))
    succ_cube_action = match_state.can_offer(opp)
    nd_persp = int(cube_perspective(match_state.cube_owner, player))
    nd_cube_action = match_state.can_offer(player)
    at_applicable = match_state.can_offer(opp)
    if at_applicable:
        at_ms = match_state.after_take(opp)
        at_persp = int(cube_perspective(at_ms.cube_owner, player))
        at_cube_action = at_ms.can_offer(player)
        at_gammons = (not at_ms.jacoby or at_ms.cube_owner != CubeOwner.CENTERED)
    else:
        at_persp = at_cube_action = at_gammons = 0
    c = _python_to_c(state)
    return float(_lib.bgn_value_2ply_cubeful(
        ctypes.byref(c), int(k),
        succ_persp, 1 if succ_cube_action else 0, 1 if gammons else 0,
        nd_persp, 1 if nd_cube_action else 0, 1 if gammons else 0,
        1 if at_applicable else 0, at_persp,
        1 if at_cube_action else 0, 1 if at_gammons else 0))


def best_move_0ply_cubeful(state: BoardState, dice, succ_persp: int,
                           succ_cube_action: bool):
    """Return (idx, resulting_state, mover_equity) or (-1, None, None)."""
    c = _python_to_c(state)
    out = _CBoardState()
    val = ctypes.c_float(0.0)
    idx = _lib.bgn_best_move_0ply_cubeful(
        ctypes.byref(c), int(dice[0]), int(dice[1]),
        int(succ_persp), 1 if succ_cube_action else 0,
        ctypes.byref(out), ctypes.byref(val))
    if idx < 0:
        return -1, None, None
    rs = BoardState(
        points=[out.points[i] for i in range(24)],
        bar=[out.bar[0], out.bar[1]],
        off=[out.off[0], out.off[1]],
        turn=out.turn,
    )
    return idx, rs, float(val.value)
