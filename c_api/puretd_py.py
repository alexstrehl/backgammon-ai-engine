"""puretd_py -- ctypes binding for the PureTD C API (libpuretd_<backend>.so).

    from c_api.puretd_py import Engine, CAgent
    eng = Engine("best_models/cubeful_money_512_512_256_256.pt")   # or a .bin
    idx, next_state, value = eng.best_move(state, (3, 1), match_state, plies=1)
    agent = CAgent("best_models/cubeless_prob5_512_512_256_256.pt", plies=1)

Backend: `backend=` argument, else the PURETD_BACKEND env, else the first
of torch, simd, cblas, ref whose libpuretd_<b>.so is next to this module
(or in PURETD_LIBDIR). play_backend() is the play paths' choice (simd
unless named). `ensure_lib(backend)` builds a missing library, or one
whose ABI tag differs from ABI_VERSION, on demand; load_lib refuses a
library whose bgc_abi_version or struct sizes differ, naming the rebuild.
1-ply root move filter: `Engine(..., root_filter=T)` / `set_root_filter(T)`
(0 = full width, the default); CAgent reads PURETD_ROOT_FILTER when not
given one.
A .pt is exported to BGNN in a private 0700 temp dir (removed at exit);
the export names the net's encoder (export_weights.py format 2) and
bgc_open resolves it through the C API's encoder table (Engine.encoder;
`encoder_info(name)` reads the table). A .bin path is opened as is: a
legacy format-1 file (no encoder name) opens by its input width.
An Engine is confined to one thread; open one per thread / process.

Match play (KIND_MATCH: the matchplay1h64met_ / matchplay1h64met6_ nets):
`eng.set_met(met)` with a met.MatchEquityTable first, then `match_value /
match_offer / match_best_move / match_score_moves / match_value_1ply /
match_value_1ply_play / match_encode / match_new_game_mwc` with a
modes.MatchState (converted by
`match_from_ms`); `eng.collect_match(match_collect_opts(...), seed, n)` is
the match-episode collector (trainer._collect_episode_matchplay's
samples).
"""
import atexit
import ctypes
import itertools
import os
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "c_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents import Agent, CubeOffer  # noqa: E402
from backgammon_engine import BoardState  # noqa: E402
from bg_fast import _CBoardState, _CPlay, _python_to_c, _c_to_python, MAX_PLAYS  # noqa: E402

BACKENDS = ("torch", "simd", "cblas", "ref")
PLAY_BACKEND = "simd"       # fastest at play batch sizes

# puretd.h BGC_ABI_VERSION; bump both on any public struct or signature change.
ABI_VERSION = 3
_ABI_TAG = f"puretd-abi:{ABI_VERSION}\0".encode()

KIND_PROB, KIND_EQUITY, KIND_CUBEFUL, KIND_PROB5, KIND_MATCH = 0, 1, 2, 3, 4
KIND_MATCH_JANOWSKI = 5     # a prob5 net under the Janowski conversion (p5jan)
MATCH_KINDS = (KIND_MATCH, KIND_MATCH_JANOWSKI)
# The kind follows from the export's encoder (Engine.encoder) and output
# mode: KIND_CUBEFUL covers the 200-input cubeful_perspective196 nets.
# bgc_collect returns COLLECT_UNSUPPORTED for an encoder whose table entry
# has no C collector (encoder_info(name)["collect"]).
COLLECT_UNSUPPORTED = -3
_OUTPUT_MODE = {KIND_PROB: "probability", KIND_EQUITY: "equity",
                KIND_CUBEFUL: "equity", KIND_PROB5: "prob5",
                KIND_MATCH: "probability", KIND_MATCH_JANOWSKI: "prob5"}


# ── C structs ──────────────────────────────────────────────────────

class CCube(ctypes.Structure):
    _fields_ = [("owner", ctypes.c_int), ("value", ctypes.c_int),
                ("jacoby", ctypes.c_int)]


class CCubeEval(ctypes.Structure):
    _fields_ = [("plies", ctypes.c_int), ("v_no_double", ctypes.c_float),
                ("v_double_take", ctypes.c_float), ("take", ctypes.c_int)]


class CSearch(ctypes.Structure):
    """bgc_search (puretd.h): 1-ply root filter threshold; 0 = full width."""
    _fields_ = [("root_thresh", ctypes.c_float)]


def root_filter_from_env():
    """PURETD_ROOT_FILTER as a float threshold (unset/empty: 0, full width)."""
    v = os.environ.get("PURETD_ROOT_FILTER", "").strip()
    t = float(v) if v else 0.0
    if not t >= 0.0:
        raise ValueError(f"PURETD_ROOT_FILTER must be >= 0 (got {v!r})")
    return t


def search_desc(root_filter):
    """Label of a 1-ply search setting: "" (full width) or
    "root-filter 0.08"."""
    return f"root-filter {root_filter:g}" if root_filter else ""


class CEncoderInfo(ctypes.Structure):
    """bgc_encoder_info (puretd.h): an encoder table entry."""
    _fields_ = [(n, ctypes.c_int) for n in (
        "n_base", "ncube", "input_size", "collect", "match", "nmet")]


class CMatch(ctypes.Structure):
    """bgc_match (puretd.h): a modes.MatchState in WHITE's view."""
    _fields_ = [("length", ctypes.c_int), ("away", ctypes.c_int * 2),
                ("cube_owner", ctypes.c_int), ("cube_value", ctypes.c_int),
                ("is_crawford", ctypes.c_int), ("crawford_done", ctypes.c_int)]


class CMatchCube(ctypes.Structure):
    """bgc_match_cube (puretd.h): offer_double_matchplay's v_nd, v_dt, v_dp."""
    _fields_ = [("v_nd", ctypes.c_double), ("v_dt", ctypes.c_double),
                ("v_dp", ctypes.c_double)]


class CMet(ctypes.Structure):
    """bgc_met (puretd.h): a MatchEquityTable's _pre (n x n) and _post (n)."""
    _fields_ = [("pre", ctypes.POINTER(ctypes.c_double)),
                ("post", ctypes.POINTER(ctypes.c_double)), ("n", ctypes.c_int)]


class CJanCtx(ctypes.Structure):
    """bgc_jan_ctx (puretd.h): a janowski_match.MatchCtx."""
    _fields_ = [(n, ctypes.c_int) for n in (
        "my_away", "opp_away", "crawford_done", "is_crawford", "cube")]


class CRng(ctypes.Structure):
    _fields_ = [("mt", ctypes.c_uint32 * 624), ("idx", ctypes.c_int)]


class CGameOpts(ctypes.Structure):
    _fields_ = [("cubeful", ctypes.c_int), ("jacoby", ctypes.c_int),
                ("cube_enabled", ctypes.c_int), ("plies_white", ctypes.c_int),
                ("plies_black", ctypes.c_int)]


class CCollectOpts(ctypes.Structure):
    """bgc_collect_opts (puretd.h)."""
    _fields_ = [("oneply", ctypes.c_int), ("jacoby", ctypes.c_int),
                ("max_plies", ctypes.c_int)]


class CMatchCollectOpts(ctypes.Structure):
    """bgc_match_collect_opts (puretd.h), the field names of the C struct."""
    _fields_ = [("match_length", ctypes.c_int), ("oneply", ctypes.c_int),
                ("curriculum_p", ctypes.c_double),
                ("force_my", ctypes.c_int), ("force_opp", ctypes.c_int),
                ("cube_explore_eps", ctypes.c_double),
                ("boundary_upweight", ctypes.c_int),
                ("outcome_mix", ctypes.c_double),
                ("max_plies", ctypes.c_int),
                ("rng_mode", ctypes.c_int)]

    def as_dict(self):
        return {n: getattr(self, n) for n, _ in self._fields_}


def match_collect_opts(match_length=11, oneply=True, curriculum_p=0.0, force_my=0,
                       force_opp=0, cube_explore_eps=0.0, boundary_upweight=1,
                       outcome_mix=0.0, max_plies=0, rng_mode=0):
    """bgc_match_collect_opts. The trainer's knobs: curriculum_p =
    trainer._SCORE_CURRICULUM_P (TRAINER_SCORE_CURRICULUM_P; the trainer's
    default is 0.5, the production legs run 0.0), force_my / force_opp =
    TRAINER_CURRICULUM_FORCE ("my,opp"; 0 = unset), cube_explore_eps =
    trainer._CUBE_EXPLORATION_EPS, boundary_upweight =
    trainer._BOUNDARY_UPWEIGHT, outcome_mix = MatchplayMode.outcome_mix;
    max_plies <= 0 unbounded (the serial collector), > 0 the vectorized
    per-game bound (the match discarded); rng_mode 0 = _collect_worker's one
    stream, 1 = the vectorized worker's per-episode streams."""
    return CMatchCollectOpts(int(match_length), int(bool(oneply)), float(curriculum_p),
                             int(force_my), int(force_opp), float(cube_explore_eps),
                             int(boundary_upweight), float(outcome_mix),
                             int(max_plies), int(rng_mode))


# = trainer.DEFAULT_MAX_EPISODE_PLIES (checked by tests/test_c_collect.py;
# not imported, since trainer imports torch).
DEFAULT_MAX_PLIES = 1500


class CGameResult(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in (
        "winner", "result", "cube_value", "ended_by_drop", "n_checker",
        "n_dance", "n_cube_offer", "n_cube_resp", "n_turns", "n_dice")]

    def as_dict(self):
        return {n: getattr(self, n) for n, _ in self._fields_}


RECORD_FN = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_CBoardState),
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(CCube), ctypes.c_int,
    ctypes.POINTER(_CBoardState), ctypes.c_float)


# ── library loading ────────────────────────────────────────────────

def lib_dir():
    """PURETD_LIBDIR env (build.sh OUT=...), else this module's directory."""
    return os.environ.get("PURETD_LIBDIR") or _HERE


def lib_path(backend):
    return os.path.join(lib_dir(), f"libpuretd_{backend}.so")


def available_backends():
    """Backends whose libpuretd_<b>.so exists next to this module."""
    return [b for b in BACKENDS if os.path.isfile(lib_path(b))]


def play_backend(backend=None):
    """The play paths' backend: `backend`, else PURETD_BACKEND, else simd."""
    return backend or os.environ.get("PURETD_BACKEND", "").strip() or PLAY_BACKEND


def default_backend():
    env = os.environ.get("PURETD_BACKEND", "").strip()
    if env:
        return env
    avail = available_backends()
    if not avail:
        raise RuntimeError(f"no libpuretd_<backend>.so in {lib_dir()}; run c_api/build.sh")
    return avail[0]


BUILD_SH = os.path.join(_HERE, "build.sh")
_build_failed = {}          # lib path -> why the on-demand build failed
_built = {}                 # lib path -> "built" / "rebuilt (stale ABI)"


def lib_abi_ok(path):
    """The file at `path` carries this module's ABI tag (read, not loaded)."""
    try:
        with open(path, "rb") as f:
            return _ABI_TAG in f.read()
    except OSError:
        return False


def _rebuild_cmd(backend):
    out = f"OUT={lib_dir()} " if os.environ.get("PURETD_LIBDIR") else ""
    return f"{out}bash c_api/build.sh {backend}"


def ensure_lib(backend, rebuild_stale=True):
    """The path of libpuretd_<backend>.so, built on demand when it is
    missing or (rebuild_stale) its ABI tag is not ABI_VERSION's: `bash
    c_api/build.sh <backend>` into lib_dir(), output captured, under a
    lock so concurrent starts build it once; build.sh renames into place,
    so processes holding the old library are unaffected. A failure is
    remembered for the process and raised as RuntimeError with the log's
    tail; a stale library with rebuild_stale=False raises, naming the
    rebuild."""
    path = lib_path(backend)
    if os.path.isfile(path) and lib_abi_ok(path):
        return path
    if os.path.isfile(path) and not rebuild_stale:
        raise RuntimeError(f"{path} was built for another C API ABI (not "
                           f"{ABI_VERSION}); rebuild it: {_rebuild_cmd(backend)}")
    if path in _build_failed:
        raise RuntimeError(_build_failed[path])
    import fcntl
    import subprocess
    out = lib_dir()
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, ".build.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.isfile(path) and lib_abi_ok(path):   # built meanwhile
            return path
        stale = os.path.isfile(path)
        try:
            r = subprocess.run(["bash", BUILD_SH, backend], capture_output=True,
                               text=True, env=dict(os.environ, OUT=out))
            log, rc = r.stdout + r.stderr, r.returncode
        except OSError as e:
            log, rc = str(e), -1
    if rc != 0 or not lib_abi_ok(path):
        tail = " | ".join(log.strip().splitlines()[-3:])
        what = "is stale (ABI)" if stale else "not found"
        _build_failed[path] = (f"{path} {what} and `bash c_api/build.sh "
                               f"{backend}` did not build it (exit {rc}: {tail})")
        raise RuntimeError(_build_failed[path])
    _built[path] = "rebuilt (stale ABI)" if stale else "built"
    return path


def ready_lib(backend, rebuild_stale=True):
    """None when libpuretd_<backend>.so loads, after ensure_lib (one NOTE
    when it built the library); else why not (RuntimeError/OSError text)."""
    try:
        path = ensure_lib(backend, rebuild_stale=rebuild_stale)
        load_lib(backend)
    except (RuntimeError, OSError) as e:
        return str(e)
    how = _built.pop(path, None)
    if how:
        print(f"  NOTE: {how} {path} on demand (bash c_api/build.sh "
              f"{backend}).", flush=True)
    return None


_libs = {}


def load_lib(backend):
    """The ctypes handle of libpuretd_<backend>.so, signatures set."""
    if backend in _libs:
        return _libs[backend]
    path = lib_path(backend)
    if not os.path.isfile(path):
        raise RuntimeError(f"{path} not found; run c_api/build.sh {backend}")
    L = ctypes.CDLL(path)
    # Not rebuilt here: this process now holds the library (ensure_lib
    # rebuilds before a load).
    why = _abi_mismatch(L)
    if why:
        raise RuntimeError(f"{path}: {why}; rebuild it: {_rebuild_cmd(backend)}")
    P, I, F, D = ctypes.POINTER, ctypes.c_int, ctypes.c_float, ctypes.c_double
    E, BS = ctypes.c_void_p, P(_CBoardState)
    sig = {
        "bgc_open": ([ctypes.c_char_p], E),
        "bgc_open_ex": ([ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t], E),
        "bgc_close": ([E], None),
        "bgc_kind": ([E], I),
        "bgc_input_size": ([E], I),
        "bgc_encoder_name": ([E], ctypes.c_char_p),
        "bgc_format_version": ([E], I),
        "bgc_encoder_lookup": ([ctypes.c_char_p, P(CEncoderInfo)], I),
        "bgc_describe": ([E], ctypes.c_char_p),
        "bgc_value": ([E, BS, P(CCube), I], F),
        "bgc_value_prob5": ([E, BS, P(F), P(F)], I),
        "bgc_value_1ply": ([E, BS, P(CCube)], F),
        "bgc_best_move": ([E, BS, I, I, P(CCube), I, BS, P(_CPlay), P(F)], I),
        "bgc_score_moves": ([E, BS, I, I, P(CCube), I, P(F), BS], I),
        "bgc_value_1ply_play": ([E, BS, I, I, P(CCube), P(F), BS, P(F)], I),
        "bgc_set_search": ([E, P(CSearch)], I),
        "bgc_offer_double": ([E, BS, P(CCube), I, P(CCubeEval)], I),
        "bgc_respond_to_double": ([E, BS, P(CCube), I, P(CCubeEval)], I),
        "bgc_rng_seed": ([P(CRng), ctypes.c_ulonglong], None),
        "bgc_rng_u32": ([P(CRng)], ctypes.c_uint),
        "bgc_rng_getrandbits": ([P(CRng), I], ctypes.c_uint),
        "bgc_rng_randint": ([P(CRng), I, I], I),
        "bgc_rng_random": ([P(CRng)], D),
        "bgc_set_met": ([E, P(D), P(D), I], I),
        "bgc_match_encode": ([E, BS, P(CMatch), I, P(F)], I),
        "bgc_match_new_game_mwc": ([E, P(CMatch), I], D),
        "bgc_match_value": ([E, BS, P(CMatch), I], F),
        "bgc_match_offer": ([E, BS, P(CMatch), P(CMatchCube)], I),
        "bgc_match_best_move": ([E, BS, I, I, P(CMatch), BS, P(F)], I),
        "bgc_match_score_moves": ([E, BS, I, I, P(CMatch), P(F), BS], I),
        "bgc_match_value_1ply": ([E, BS, P(CMatch)], D),
        "bgc_match_value_1ply_play": ([E, BS, I, I, P(CMatch), P(D), BS], I),
        "bgc_collect_match": ([E, P(CMatchCollectOpts), ctypes.c_ulonglong, I, P(F), P(F),
                               ctypes.c_long, P(I)], ctypes.c_long),
        "bgc_match_memo_stats": ([E, P(ctypes.c_long), P(ctypes.c_long), P(ctypes.c_long)], I),
        "bgc_play_game": ([E, E, P(CGameOpts), P(CRng), P(CGameResult),
                           RECORD_FN, ctypes.c_void_p], I),
        "bgc_collect": ([E, P(CCollectOpts), ctypes.c_ulonglong, I, P(F), P(F),
                         ctypes.c_long, P(I)], ctypes.c_long),
        "bgc_bench_checker": ([E, BS, P(I), P(CCube), I, I, P(D)], I),
        "bgc_bench_cube": ([E, BS, P(CCube), I, I, P(D)], I),
        "bgc_bench_selfplay": ([E, P(CGameOpts), ctypes.c_ulonglong, I, I, I,
                                P(D), P(I), P(ctypes.c_longlong)], I),
        # shared memo, fast movegen, p5jan (every library passing the ABI check)
        "bgc_match_shm_stats": ([E, P(ctypes.c_long), P(ctypes.c_long), P(ctypes.c_long)], I),
        "bgc_test_legal_plays": ([BS, I, I, I, P(_CPlay)], I),
        "bgc_jan_do_cubeful": ([P(CJanCtx)], I),
        "bgc_jan_dead_mwc": ([P(CMet), P(D), P(CJanCtx)], D),
        "bgc_jan_get_points": ([P(CMet), P(D), P(CJanCtx), P(D)], I),
        "bgc_jan_cubeful_mwc": ([P(CMet), P(D), I, P(CJanCtx), D], D),
        "bgc_jan_cube_decision": ([P(CMet), P(D), P(CMatch), I, D, P(CMatchCube)], I),
        "bgc_jan_cube_x": ([BS], D),
        "bgc_open_janowski": ([ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t], E),
        "bgc_set_jan_x": ([E, I, D], I),
    }
    for name, (args, res) in sig.items():
        fn = getattr(L, name)
        fn.argtypes = args
        fn.restype = res
    _libs[backend] = L
    return L


# bgc_struct_sizes order (puretd.h).
_ABI_STRUCTS = (_CBoardState, _CPlay, CCube, CCubeEval, CSearch, CEncoderInfo,
                CRng, CGameOpts, CGameResult, CCollectOpts, CMatch, CMatchCube,
                CMatchCollectOpts, CMet, CJanCtx)


def _abi_mismatch(L):
    """Why the loaded library `L` does not match this module's ctypes
    declarations, or None."""
    if not hasattr(L, "bgc_abi_version"):
        return f"built before the C API ABI check (this module: ABI {ABI_VERSION})"
    L.bgc_abi_version.argtypes = []
    L.bgc_abi_version.restype = ctypes.c_int
    v = L.bgc_abi_version()
    if v != ABI_VERSION:
        return f"C API ABI {v}, this module expects {ABI_VERSION}"
    if not hasattr(L, "bgc_struct_sizes"):
        return f"C API ABI {v} without bgc_struct_sizes"
    n = len(_ABI_STRUCTS)
    sizes = (ctypes.c_size_t * n)()
    L.bgc_struct_sizes.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    L.bgc_struct_sizes.restype = ctypes.c_int
    count = L.bgc_struct_sizes(sizes, n)
    want = [ctypes.sizeof(t) for t in _ABI_STRUCTS]
    if count != n or list(sizes) != want:
        return (f"struct sizes {list(sizes)[:count]} differ from the ctypes "
                f"declarations {want}")
    return None


def encoder_info(name, backend=None):
    """The C API encoder table's entry for encoder `name` (an
    encoding.get_encoder name) as a dict: n_base, ncube, input_size,
    collect (bool: bgc_collect writes its rows), match (bool: a match
    encoder), nmet; None when bgc_open does not accept the name."""
    info = CEncoderInfo()
    lib = load_lib(backend or default_backend())
    if lib.bgc_encoder_lookup(str(name).encode(), ctypes.byref(info)) != 0:
        return None
    d = {n: getattr(info, n) for n, _ in CEncoderInfo._fields_}
    for k in ("collect", "match"):
        d[k] = bool(d[k])
    return d


# ── weight export ──────────────────────────────────────────────────

_export_dir = None
_export_seq = itertools.count()


def _private_export_path(stem):
    """A fresh path in this process's private (0700) temp dir, removed at exit."""
    global _export_dir
    if _export_dir is None:
        _export_dir = tempfile.mkdtemp(prefix="puretd_w_")
        atexit.register(shutil.rmtree, _export_dir, ignore_errors=True)
    return os.path.join(_export_dir, f"{stem}_{next(_export_seq)}.bin")


def export_bin(model):
    """BGNN export of a .pt checkpoint (or a loaded net) to a private path."""
    import export_weights
    from model import load_model
    net = load_model(model) if isinstance(model, str) else model
    stem = os.path.basename(model)[:-3] if isinstance(model, str) else "net"
    path = _private_export_path(stem)
    export_weights.export_live(net, path)
    return path


# ── conversions ────────────────────────────────────────────────────

def cube_from_ms(ms):
    """MatchState -> CCube (None -> None)."""
    if ms is None:
        return None
    return CCube(int(ms.cube_owner), int(ms.cube_value), int(bool(ms.jacoby)))


def _cube_arg(ms):
    """byref(CCube) of a MatchState; None (NULL cube) for None."""
    return None if ms is None else ctypes.byref(cube_from_ms(ms))


def match_from_ms(ms):
    """modes.MatchState -> CMatch (WHITE's view, as MatchState stores it)."""
    a = (ctypes.c_int * 2)(int(ms.my_away), int(ms.opp_away))
    return CMatch(int(ms.match_length), a, int(ms.cube_owner), int(ms.cube_value),
                  int(bool(ms.is_crawford)), int(bool(ms.crawford_done)))


# ── p5jan: the Janowski conversion (puretd_janowski.c) ────────────

def c_met(met):
    """The CMet of a met.MatchEquityTable (float64 copies of _pre / _post,
    cached on the table with the arrays it points into)."""
    hit = getattr(met, "_c_met", None)
    if hit is None:
        pre = np.array(met._pre, dtype=np.float64, order="C")
        post = np.array(met._post, dtype=np.float64, order="C")
        n = pre.shape[0]
        if pre.shape != (n, n) or post.shape != (n,):
            raise ValueError(f"MET tables must be (n, n) and (n,), got {pre.shape} "
                             f"and {post.shape}")
        DP = ctypes.POINTER(ctypes.c_double)
        hit = (CMet(pre.ctypes.data_as(DP), post.ctypes.data_as(DP), n), pre, post)
        met._c_met = hit
    return hit[0]


def jan_ctx(ctx):
    """janowski_match.MatchCtx -> CJanCtx."""
    return CJanCtx(int(ctx.my_away), int(ctx.opp_away), int(bool(ctx.crawford_done)),
                   int(bool(ctx.is_crawford)), int(ctx.cube))


def _d5(probs):
    return (ctypes.c_double * 5)(*(float(probs[i]) for i in range(5)))


class JanowskiC:
    """janowski_match's conversion functions on the C port, with the Python
    signatures (`met` None = met.default_met(), as there):

        J = JanowskiC()            # any built backend: the conversion has no net
        J.cubeful_mwc(probs, persp, ctx, x, met)
        J.cube_decision(probs, match_state, player, x, met)   # the dict
        J.dead_mwc(probs, ctx, met); J.get_points(rates, ctx, met)
        J.do_cubeful(ctx); J.cube_x(state)

    A value the MET cannot price (Python raises) is NaN here, a failed
    get_points / cube_decision raises ValueError."""

    def __init__(self, backend=None):
        self.backend = backend or default_backend()
        self.lib = load_lib(self.backend)    # its ABI check guarantees bgc_jan_*

    @staticmethod
    def _met(met):
        if met is None:
            from met import default_met
            met = default_met()
        return ctypes.byref(c_met(met))

    def do_cubeful(self, ctx):
        return bool(self.lib.bgc_jan_do_cubeful(ctypes.byref(jan_ctx(ctx))))

    def dead_mwc(self, probs, ctx, met=None):
        return self.lib.bgc_jan_dead_mwc(self._met(met), _d5(probs), ctypes.byref(jan_ctx(ctx)))

    def get_points(self, rates, ctx, met=None):
        cp = (ctypes.c_double * 2)()
        r = (ctypes.c_double * 4)(*(float(v) for v in rates))
        if self.lib.bgc_jan_get_points(self._met(met), r, ctypes.byref(jan_ctx(ctx)), cp):
            raise ValueError(f"bgc_jan_get_points failed for {ctx}")
        return cp[0], cp[1]

    def cubeful_mwc(self, probs, persp, ctx, x, met=None):
        return self.lib.bgc_jan_cubeful_mwc(self._met(met), _d5(probs), int(persp),
                                            ctypes.byref(jan_ctx(ctx)), float(x))

    def cube_decision(self, probs, match_state, player, x, met=None):
        c = CMatchCube()
        r = self.lib.bgc_jan_cube_decision(self._met(met), _d5(probs),
                                           ctypes.byref(match_from_ms(match_state)),
                                           int(player), float(x), ctypes.byref(c))
        if r < 0:
            raise ValueError(f"bgc_jan_cube_decision failed for {match_state}")
        return {"v_nd": c.v_nd, "v_dt": c.v_dt, "v_dp": c.v_dp,
                "should_double": bool(r), "take": c.v_dp > c.v_dt}

    def cube_x(self, state):
        return self.lib.bgc_jan_cube_x(ctypes.byref(_python_to_c(state)))


class Rng:
    """random.Random(seed)-compatible dice source backed by bgc_rng."""

    def __init__(self, seed, backend=None):
        if not 0 <= int(seed) < 1 << 64:
            raise ValueError("Rng seed must be an int in [0, 2**64); use "
                             "Rng.from_state for other random.Random states")
        self._lib = load_lib(backend or default_backend())
        self.c = CRng()
        self._lib.bgc_rng_seed(ctypes.byref(self.c), int(seed))

    @classmethod
    def from_state(cls, state, backend=None):
        """From random.getstate() (version 3): same stream as that Random."""
        version, internal, _ = state
        if version != 3 or len(internal) != 625:
            raise ValueError("unsupported random state")
        r = cls.__new__(cls)
        r._lib = load_lib(backend or default_backend())
        r.c = CRng()
        r.c.mt[:] = internal[:624]
        r.c.idx = internal[624]
        return r

    def getstate(self, gauss_next=None):
        """random.setstate()-compatible state of this stream."""
        return (3, tuple(self.c.mt) + (self.c.idx,), gauss_next)

    def u32(self):
        return self._lib.bgc_rng_u32(ctypes.byref(self.c))

    def getrandbits(self, k):
        return self._lib.bgc_rng_getrandbits(ctypes.byref(self.c), k)

    def randint(self, a, b):
        return self._lib.bgc_rng_randint(ctypes.byref(self.c), a, b)

    def random(self):
        """random.Random.random()."""
        return self._lib.bgc_rng_random(ctypes.byref(self.c))


class CheckerPool:
    """Checker points [(state, dice) or (state, dice, match_state)] as ctypes
    arrays, converted once."""

    def __init__(self, points):
        n = len(points)
        self.n = n
        self.states = (_CBoardState * n)()
        self.dice = (ctypes.c_int * (2 * n))()
        has_ms = n > 0 and len(points[0]) > 2 and points[0][2] is not None
        self.cubes = (CCube * n)() if has_ms else None
        for i, p in enumerate(points):
            self.states[i] = _python_to_c(p[0])
            self.dice[2 * i], self.dice[2 * i + 1] = p[1]
            if has_ms:
                self.cubes[i] = cube_from_ms(p[2])


class CubePool:
    """Cube points [(state, match_state)] as ctypes arrays, converted once."""

    def __init__(self, points):
        n = len(points)
        self.n = n
        self.states = (_CBoardState * n)()
        self.cubes = (CCube * n)()
        for i, (s, ms) in enumerate(points):
            self.states[i] = _python_to_c(s)
            self.cubes[i] = cube_from_ms(ms)


def collect_opts(oneply=False, jacoby=True, max_plies=DEFAULT_MAX_PLIES):
    """bgc_collect_opts; max_plies <= 0 = unbounded (trainer semantics)."""
    return CCollectOpts(int(bool(oneply)), int(bool(jacoby)), int(max_plies))


def game_opts(cubeful=True, jacoby=True, cube_enabled=True, plies_white=0,
              plies_black=None):
    return CGameOpts(int(cubeful), int(jacoby), int(cube_enabled), plies_white,
                     plies_white if plies_black is None else plies_black)


# ── Engine ─────────────────────────────────────────────────────────

class Engine:
    """One bgc_engine: a model on one backend. Thread-confined.

    janowski=True opens a prob5 net as a match engine
    (KIND_MATCH_JANOWSKI, bgc_open_janowski): JanowskiMatchAgent (p5jan)
    through the match_* methods, after set_met (and set_jan_x for a
    constant x)."""

    def __init__(self, model, backend=None, root_filter=0.0, janowski=False):
        self.backend = backend or default_backend()
        self.lib = load_lib(self.backend)    # its ABI check guarantees bgc_open_janowski
        if isinstance(model, str) and model.endswith(".bin"):
            path, tmp = model, False
        else:
            path, tmp = export_bin(model), True
        self.model = model if isinstance(model, str) else "<net>"
        err = ctypes.create_string_buffer(1024)
        opener = self.lib.bgc_open_janowski if janowski else self.lib.bgc_open_ex
        self.h = opener(path.encode(), err, len(err))
        if tmp:
            os.remove(path)
        if not self.h:
            raise RuntimeError(f"bgc_open failed for {self.model}: "
                               f"{err.value.decode(errors='replace')}")
        self.kind = self.lib.bgc_kind(self.h)
        self.input_size = self.lib.bgc_input_size(self.h)
        self.encoder = self.lib.bgc_encoder_name(self.h).decode()
        self.format_version = self.lib.bgc_format_version(self.h)   # 1: legacy .bin
        self.describe = self.lib.bgc_describe(self.h).decode()
        self._out_state = _CBoardState()
        self._val = ctypes.c_float()
        self._scores = (ctypes.c_float * MAX_PLAYS)()
        self._states = (_CBoardState * MAX_PLAYS)()
        self.root_filter = 0.0
        self.set_root_filter(root_filter)

    def set_root_filter(self, thresh):
        """1-ply root move filter threshold (bgc_set_search); 0 = full width."""
        t = float(thresh or 0.0)
        if self.lib.bgc_set_search(self.h, ctypes.byref(CSearch(t))) != 0:
            raise ValueError(f"bad root filter threshold {thresh!r}")
        self.root_filter = t

    def close(self):
        if getattr(self, "h", None):
            self.lib.bgc_close(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # values
    def value(self, state, ms=None, is_cube_action=False):
        return self.lib.bgc_value(self.h, ctypes.byref(_python_to_c(state)),
                                  _cube_arg(ms), int(is_cube_action))

    def value_prob5(self, state):
        probs = (ctypes.c_float * 5)()
        eq = ctypes.c_float()
        if self.lib.bgc_value_prob5(self.h, ctypes.byref(_python_to_c(state)),
                                    probs, ctypes.byref(eq)) != 0:
            raise ValueError("value_prob5 needs a prob5 model")
        return np.array(probs, dtype=np.float32), eq.value

    def value_1ply(self, state, ms=None):
        return self.lib.bgc_value_1ply(self.h, ctypes.byref(_python_to_c(state)),
                                       _cube_arg(ms))

    # checker
    def best_move(self, state, dice, ms=None, plies=0):
        """(index, next_state, mover-view value); (-1, None, nan) on a pass."""
        idx = self.lib.bgc_best_move(
            self.h, ctypes.byref(_python_to_c(state)), dice[0], dice[1],
            _cube_arg(ms), plies, ctypes.byref(self._out_state), None,
            ctypes.byref(self._val))
        if idx == -2:
            raise RuntimeError("bgc_best_move failed")
        if idx < 0:
            return -1, None, float("nan")
        return idx, _c_to_python(self._out_state), self._val.value

    def value_1ply_play(self, state, dice, ms=None):
        """bgc_value_1ply_play: (index, next_state, mover-view score,
        1-ply value) = best_move(state, dice, ms, 0) and value_1ply(state,
        ms) from one expansion; (-1, None, nan, v) on a forced pass."""
        v = ctypes.c_float()
        idx = self.lib.bgc_value_1ply_play(
            self.h, ctypes.byref(_python_to_c(state)), dice[0], dice[1],
            _cube_arg(ms), ctypes.byref(v), ctypes.byref(self._out_state),
            ctypes.byref(self._val))
        if idx == -2:
            raise RuntimeError("bgc_value_1ply_play failed")
        if idx < 0:
            return -1, None, float("nan"), v.value
        return idx, _c_to_python(self._out_state), self._val.value, v.value

    def score_moves(self, state, dice, ms=None, plies=0):
        """(scores float32 array, successor states), get_legal_plays order."""
        n = self.lib.bgc_score_moves(
            self.h, ctypes.byref(_python_to_c(state)), dice[0], dice[1],
            _cube_arg(ms), plies, self._scores, self._states)
        if n < 0:
            raise RuntimeError("bgc_score_moves failed")
        return (np.array(self._scores[:n], dtype=np.float32),
                [_c_to_python(self._states[i]) for i in range(n)])

    # cube
    def offer_double(self, state, ms, plies=0):
        """(should_double, CCubeEval with the take)."""
        ce = CCubeEval()
        r = self.lib.bgc_offer_double(self.h, ctypes.byref(_python_to_c(state)),
                                      ctypes.byref(cube_from_ms(ms)), plies,
                                      ctypes.byref(ce))
        if r < 0:
            raise RuntimeError("bgc_offer_double failed")
        return bool(r), ce

    def respond_to_double(self, state, ms, plies=0, hint=None):
        r = self.lib.bgc_respond_to_double(
            self.h, ctypes.byref(_python_to_c(state)),
            ctypes.byref(cube_from_ms(ms)), plies,
            ctypes.byref(hint) if hint is not None else None)
        if r < 0:
            raise RuntimeError("bgc_respond_to_double failed")
        return bool(r)

    # match play (KIND_MATCH, KIND_MATCH_JANOWSKI)
    def _need_match(self):
        if self.kind not in MATCH_KINDS:
            raise ValueError(f"{self.encoder} is not a match encoder (kind {self.kind})")

    def set_jan_x(self, x_policy=None):
        """bgc_set_jan_x: JanowskiMatchAgent's x_policy (None = the gnubg
        position-class cube_x of each evaluated board, the default; a
        number = that constant)."""
        if self.kind != KIND_MATCH_JANOWSKI:
            raise ValueError(f"set_jan_x needs a Janowski engine (kind {self.kind})")
        const = x_policy is not None
        if self.lib.bgc_set_jan_x(self.h, int(const),
                                  float(x_policy) if const else 0.0) != 0:
            raise ValueError(f"bad x policy {x_policy!r}")
        self.x_policy = None if x_policy is None else float(x_policy)

    def set_met(self, met):
        """bgc_set_met: copy a met.MatchEquityTable's _pre (n x n) and _post
        (n) into the engine (float64, C-contiguous); required before any
        match_* call."""
        self._need_match()
        pre = np.ascontiguousarray(met._pre, dtype=np.float64)
        post = np.ascontiguousarray(met._post, dtype=np.float64)
        n = pre.shape[0]
        if pre.shape != (n, n) or post.shape != (n,):
            raise ValueError(f"MET tables must be (n, n) and (n,), got {pre.shape} "
                             f"and {post.shape}")
        DP = ctypes.POINTER(ctypes.c_double)
        if self.lib.bgc_set_met(self.h, pre.ctypes.data_as(DP), post.ctypes.data_as(DP),
                                n) != 0:
            raise ValueError("bgc_set_met failed (non-finite entry?)")
        self.met = met

    def match_encode(self, state, ms, is_cube_action=False):
        """TDAgent._encode_matchplay(state, ms, is_cube_action): float32 row."""
        row = np.empty(self.input_size, dtype=np.float32)
        r = self.lib.bgc_match_encode(
            self.h, ctypes.byref(_python_to_c(state)), ctypes.byref(match_from_ms(ms)),
            int(bool(is_cube_action)), row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if r != 0:
            raise ValueError("bgc_match_encode failed (finished match, cube value, "
                             "away above the MET, or no MET set)")
        return row

    def match_new_game_mwc(self, after_game_ms, player):
        """TDAgent._eval_new_game(after_game_ms, player) with the MET."""
        return self.lib.bgc_match_new_game_mwc(self.h, ctypes.byref(match_from_ms(after_game_ms)),
                                               int(player))

    def match_value(self, state, ms, is_cube_action=False):
        """evaluate_matchplay: P(state.turn wins the match), fp32."""
        return self.lib.bgc_match_value(self.h, ctypes.byref(_python_to_c(state)),
                                        ctypes.byref(match_from_ms(ms)),
                                        int(bool(is_cube_action)))

    def match_offer(self, state, ms):
        """(should_double, CMatchCube v_nd / v_dt / v_dp); (False, NaNs) when
        state.turn cannot offer. Take = v_dp > v_dt."""
        c = CMatchCube()
        r = self.lib.bgc_match_offer(self.h, ctypes.byref(_python_to_c(state)),
                                     ctypes.byref(match_from_ms(ms)), ctypes.byref(c))
        if r < 0:
            raise RuntimeError("bgc_match_offer failed")
        return bool(r), c

    def match_best_move(self, state, dice, ms):
        """(index, next_state, opp-view value of the chosen successor);
        (-1, None, nan) on a forced pass. Bootstrap = 1 - value."""
        idx = self.lib.bgc_match_best_move(
            self.h, ctypes.byref(_python_to_c(state)), dice[0], dice[1],
            ctypes.byref(match_from_ms(ms)), ctypes.byref(self._out_state),
            ctypes.byref(self._val))
        if idx == -2:
            raise RuntimeError("bgc_match_best_move failed")
        if idx < 0:
            return -1, None, float("nan")
        return idx, _c_to_python(self._out_state), self._val.value

    def match_score_moves(self, state, dice, ms):
        """(opp-view values float32 array, successor states), get_legal_plays
        order (choose_checker_action_matchplay's `values`)."""
        n = self.lib.bgc_match_score_moves(
            self.h, ctypes.byref(_python_to_c(state)), dice[0], dice[1],
            ctypes.byref(match_from_ms(ms)), self._scores, self._states)
        if n < 0:
            raise RuntimeError("bgc_match_score_moves failed")
        return (np.array(self._scores[:n], dtype=np.float32),
                [_c_to_python(self._states[i]) for i in range(n)])

    def match_value_1ply(self, state, ms):
        """value_oneply_checker_matchplay(state, ms) (double)."""
        return self.lib.bgc_match_value_1ply(self.h, ctypes.byref(_python_to_c(state)),
                                             ctypes.byref(match_from_ms(ms)))

    def match_value_1ply_play(self, state, dice, ms):
        """(index, next_state, 1-ply value) = match_best_move's choice and
        match_value_1ply from one expansion; (-1, None, v) on a pass."""
        v = ctypes.c_double()
        idx = self.lib.bgc_match_value_1ply_play(
            self.h, ctypes.byref(_python_to_c(state)), dice[0], dice[1],
            ctypes.byref(match_from_ms(ms)), ctypes.byref(v),
            ctypes.byref(self._out_state))
        if idx == -2:
            raise RuntimeError("bgc_match_value_1ply_play failed")
        if idx < 0:
            return -1, None, v.value
        return idx, _c_to_python(self._out_state), v.value

    def match_memo_stats(self):
        """The 1-ply memo's {on, lookups, hits, entries} (bgc_match_memo_stats)."""
        L = [ctypes.c_long() for _ in range(3)]
        on = self.lib.bgc_match_memo_stats(self.h, *(ctypes.byref(x) for x in L))
        return {"on": on == 1, "lookups": L[0].value, "hits": L[1].value,
                "entries": L[2].value}

    def match_shm_stats(self):
        """The shared memo's {on, lookups, hits, puts} (bgc_match_shm_stats;
        BGC_MATCH_SHM)."""
        L = [ctypes.c_long() for _ in range(3)]
        on = self.lib.bgc_match_shm_stats(self.h, *(ctypes.byref(x) for x in L))
        return {"on": on == 1, "lookups": L[0].value, "hits": L[1].value, "puts": L[2].value}

    def collect_match(self, opts, seed, n_episodes, max_rows=None, out=None):
        """bgc_collect_match: `n_episodes` matches from random.Random(seed)
        (opts.rng_mode 0: one stream, the _collect_worker loop; 1: the
        vectorized worker's per-episode streams), the samples
        trainer._collect_episode_matchplay would return. `opts`: a
        CMatchCollectOpts or a dict of match_collect_opts keywords (None:
        the defaults). Returns (enc float32 (n, input_size), tgt float32
        (n,), n_discarded); enc/tgt are views of the buffers. `max_rows`:
        initial row capacity (default 1024 per match; an 11pt match is ~518
        rows); on -2 the buffers double and the seed is replayed. `out`:
        (enc, tgt) caller buffers (float32, C-contiguous, writeable, enc
        (cap, input_size), tgt (cap,)) written first; when the rows do not
        fit, the seed is replayed into the call's own buffers from twice
        that capacity (the result then does not share memory with `out`)."""
        self._need_match()
        if isinstance(opts, CMatchCollectOpts):
            o = CMatchCollectOpts(**opts.as_dict())
        else:
            o = match_collect_opts(**dict(opts or {}))
        if not 0 <= int(seed) < 1 << 64:
            raise ValueError("collect seed must be an int in [0, 2**64)")
        n_episodes = int(n_episodes)
        cap = int(max_rows) if max_rows is not None else 1024 * max(n_episodes, 1)
        cap = max(cap, 1)
        nd = ctypes.c_int()
        FP = ctypes.POINTER(ctypes.c_float)
        if out is not None:
            enc, tgt = out
            ok = (isinstance(enc, np.ndarray) and isinstance(tgt, np.ndarray)
                  and enc.dtype == np.float32 and tgt.dtype == np.float32
                  and enc.ndim == 2 and enc.shape[1] == self.input_size
                  and tgt.shape == (enc.shape[0],)
                  and enc.flags.c_contiguous and tgt.flags.c_contiguous
                  and enc.flags.writeable and tgt.flags.writeable)
            if not ok:
                raise ValueError(f"collect_match out= needs writeable C-contiguous float32 "
                                 f"(cap, {self.input_size}) and (cap,) arrays")
            n = self.lib.bgc_collect_match(self.h, ctypes.byref(o), int(seed), n_episodes,
                                           enc.ctypes.data_as(FP), tgt.ctypes.data_as(FP),
                                           len(enc), ctypes.byref(nd))
            if n >= 0:
                return enc[:n], tgt[:n], nd.value
            if n != -2:
                raise RuntimeError(self._collect_match_error(n))
            cap = max(2 * len(enc), 1)
        while True:
            enc = np.empty((cap, self.input_size), dtype=np.float32)
            tgt = np.empty((cap,), dtype=np.float32)
            n = self.lib.bgc_collect_match(self.h, ctypes.byref(o), int(seed), n_episodes,
                                           enc.ctypes.data_as(FP), tgt.ctypes.data_as(FP),
                                           cap, ctypes.byref(nd))
            if n != -2:
                break
            cap *= 2
        if n < 0:
            raise RuntimeError(self._collect_match_error(n))
        return enc[:n], tgt[:n], nd.value

    def _collect_match_error(self, n):
        if n == COLLECT_UNSUPPORTED:
            return (f"bgc_collect_match: no C collector for encoder {self.encoder} "
                    f"({self.input_size} inputs); collect it in Python")
        return (f"bgc_collect_match failed ({n}): bad options, no MET set, or an "
                f"error during the episodes")

    # batch-training collector
    def collect(self, opts=None, seed=0, n_episodes=1, max_rows=None, out=None,
                **kwargs):
        """bgc_collect: `n_episodes` self-play episodes on one
        random.Random(seed) stream, the samples trainer._collect_worker
        would return. `opts`: a CCollectOpts, a dict of collect_opts
        keywords, or None; keyword arguments override it (oneply, jacoby,
        max_plies). `max_rows`: initial row capacity
        (default 128 per episode); on -2 the buffers double and the same
        seed is replayed. Returns (enc float32 (n, input_size), tgt
        float32 (n,), n_discarded); a prob5 engine's tgt is (n, 5)
        (train_prob5._collect_worker's targets; pass max_plies=0, its
        episodes are unbounded). enc/tgt are views of the buffers.

        `out`: (enc, tgt) caller buffers to write into first (float32,
        C-contiguous, writeable; enc (cap, input_size), tgt (cap,), or
        (cap, 5) for prob5), e.g. a slab of shared memory. When the rows
        fit, the result is `out[0][:n], out[1][:n]`. When they do not
        (-2), the seed is replayed into the call's own buffers from twice
        that capacity, and the result does not share memory with `out`
        (compare `enc.ctypes.data`)."""
        if isinstance(opts, CCollectOpts):
            o = CCollectOpts(opts.oneply, opts.jacoby, opts.max_plies)
            for k, v in kwargs.items():
                setattr(o, k, int(v))
        else:
            o = collect_opts(**dict(opts or {}, **kwargs))
        if not 0 <= int(seed) < 1 << 64:
            raise ValueError("collect seed must be an int in [0, 2**64)")
        n_episodes = int(n_episodes)
        cap = int(max_rows) if max_rows is not None else 128 * max(n_episodes, 1)
        cap = max(cap, 1)
        nd = ctypes.c_int()
        FP = ctypes.POINTER(ctypes.c_float)
        tshape = (5,) if self.kind == KIND_PROB5 else ()     # bgc_collect's out_dim
        if out is not None:
            enc, tgt = out
            ok = (isinstance(enc, np.ndarray) and isinstance(tgt, np.ndarray)
                  and enc.dtype == np.float32 and tgt.dtype == np.float32
                  and enc.ndim == 2 and enc.shape[1] == self.input_size
                  and tgt.shape == (enc.shape[0],) + tshape
                  and enc.flags.c_contiguous and tgt.flags.c_contiguous
                  and enc.flags.writeable and tgt.flags.writeable)
            if not ok:
                raise ValueError(
                    f"collect out= needs writeable C-contiguous float32 "
                    f"(cap, {self.input_size}) and {(('cap',) + tshape)!r} arrays")
            n = self.lib.bgc_collect(self.h, ctypes.byref(o), int(seed), n_episodes,
                                     enc.ctypes.data_as(FP), tgt.ctypes.data_as(FP),
                                     len(enc), ctypes.byref(nd))
            if n >= 0:
                return enc[:n], tgt[:n], nd.value
            if n != -2:
                raise RuntimeError(self._collect_error(n))
            cap = max(2 * len(enc), 1)
        while True:
            enc = np.empty((cap, self.input_size), dtype=np.float32)
            tgt = np.empty((cap,) + tshape, dtype=np.float32)
            n = self.lib.bgc_collect(self.h, ctypes.byref(o), int(seed), n_episodes,
                                     enc.ctypes.data_as(FP), tgt.ctypes.data_as(FP),
                                     cap, ctypes.byref(nd))
            if n != -2:
                break
            cap *= 2
        if n < 0:
            raise RuntimeError(self._collect_error(n))
        return enc[:n], tgt[:n], nd.value

    def _collect_error(self, n):
        if n == COLLECT_UNSUPPORTED:
            return (f"bgc_collect: no C collector for encoder {self.encoder} "
                    f"({self.input_size} inputs); collect it in Python")
        return f"bgc_collect failed ({n})"

    # benchmarks
    def bench_checker(self, points, plies=0):
        """Per-item seconds (timed in C) of bgc_best_move over the points."""
        pool = points if isinstance(points, CheckerPool) else CheckerPool(points)
        out = np.zeros(pool.n, dtype=np.float64)
        r = self.lib.bgc_bench_checker(
            self.h, pool.states, pool.dice, pool.cubes, pool.n, plies,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if r:
            raise RuntimeError("bgc_bench_checker failed")
        return out

    def bench_cube(self, points, plies=0):
        """Per-item seconds (timed in C) of bgc_offer_double over the points."""
        pool = points if isinstance(points, CubePool) else CubePool(points)
        out = np.zeros(pool.n, dtype=np.float64)
        r = self.lib.bgc_bench_cube(
            self.h, pool.states, pool.cubes, pool.n, plies,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if r:
            raise RuntimeError("bgc_bench_cube failed")
        return out

    def bench_selfplay(self, opts, seed, warmup, reseed, games):
        """(game_seconds float64[games], game_decisions int32[games],
        totals dict) from bgc_bench_selfplay."""
        secs = np.zeros(games, dtype=np.float64)
        decs = np.zeros(games, dtype=np.int32)
        tot = (ctypes.c_longlong * 6)()
        r = self.lib.bgc_bench_selfplay(
            self.h, ctypes.byref(opts), seed, warmup, int(reseed), games,
            secs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            decs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), tot)
        if r:
            raise RuntimeError("bgc_bench_selfplay failed")
        keys = ("checker", "dance", "offers", "responses", "turns", "games")
        return secs, decs, dict(zip(keys, list(tot)))


def play_game(white, black, opts, rng, record=False):
    """One game through bgc_play_game. `rng` is an Rng (its stream
    advances) or an int seed. Returns (result dict, records); records is a
    list of dicts (kind, state, dice, cube, decision, next_state, value)
    when `record`, else None."""
    if white.lib is not black.lib:
        raise ValueError("both engines must use the same backend library")
    if not isinstance(rng, Rng):
        rng = Rng(rng, white.backend)
    res = CGameResult()
    recs = [] if record else None

    def _cb(ctx, kind, st, d1, d2, cube, decision, nxt, value):
        cb = cube.contents if cube else None
        recs.append({
            "kind": kind, "state": _c_to_python(st.contents),
            "dice": (d1, d2) if kind == 0 else None,
            "cube": (cb.owner, cb.value, cb.jacoby) if cb is not None else None,
            "decision": decision,
            "next_state": _c_to_python(nxt.contents) if nxt else None,
            "value": value,
        })

    fn = RECORD_FN(_cb) if record else RECORD_FN()
    r = white.lib.bgc_play_game(white.h, black.h, ctypes.byref(opts),
                                ctypes.byref(rng.c), ctypes.byref(res), fn, None)
    if r:
        raise RuntimeError("bgc_play_game failed")
    return res.as_dict(), recs


# ── Agent ──────────────────────────────────────────────────────────

class CAgent(Agent):
    """Agent over the C API: the decisions bgc_play_game makes, callable
    from the Python game loops. `plies` 0 or 1."""

    def __init__(self, model, plies=0, backend=None, engine=None, root_filter=None):
        """root_filter: 1-ply root filter threshold (None: PURETD_ROOT_FILTER)."""
        if plies not in (0, 1):
            raise ValueError(f"CAgent supports plies 0 or 1 (got {plies})")
        self.engine = engine if engine is not None else Engine(model, backend)
        if root_filter is None and engine is None:
            root_filter = root_filter_from_env()
        if root_filter is not None:
            self.engine.set_root_filter(root_filter)
        self.plies = plies
        self.oneply = plies == 1
        self.kind = self.engine.kind
        self.is_cubeful = self.kind == KIND_CUBEFUL
        self.output_mode = _OUTPUT_MODE[self.kind]

    # checker
    def choose_checker_action(self, state, dice, actions=None):
        """(play, next_state), or None on a forced pass. With `actions`,
        the entry whose resulting state equals the C choice. Cubeful
        models choose as TDAgent.choose_checker_action (centered cube)."""
        idx, ns, _ = self.engine.best_move(state, dice, None, self.plies)
        if idx < 0:
            return None
        if actions is None:
            return None, ns
        for p, s in actions:
            if s == ns:
                return p, s
        raise RuntimeError("C choice is not among the given actions")

    def choose_next_state(self, state, dice):
        idx, ns, _ = self.engine.best_move(state, dice, None, self.plies)
        return ns if idx >= 0 else None

    def choose_checker_action_cubeful(self, state, dice, match_state):
        """(next_state, mover-view value), or None on a forced pass."""
        idx, ns, v = self.engine.best_move(
            state, dice, match_state if self.is_cubeful else None, self.plies)
        if idx < 0:
            return None
        return ns, v

    # cube
    def offer_double(self, state, match_state):
        if not self.is_cubeful:
            return CubeOffer(should_double=False, _cache=None)
        d, ce = self.engine.offer_double(state, match_state, self.plies)
        return CubeOffer(should_double=d, _cache=ce)

    def respond_to_double(self, state, match_state, hint=None):
        if not self.is_cubeful:
            return True
        ce = getattr(hint, "_cache", None)
        if not isinstance(ce, CCubeEval):
            ce = None
        return self.engine.respond_to_double(state, match_state, self.plies, ce)

    # values
    def evaluate(self, state, match_state=None, is_cube_action=False):
        """0-ply value from state.turn's view. Cubeful: the cube of
        match_state (default: centered, is_cube_action 0)."""
        if self.is_cubeful and match_state is None:
            from modes import MatchState
            match_state = MatchState()
        return self.engine.value(state, match_state, is_cube_action)

    def evaluate_batch(self, states):
        return np.asarray([self.evaluate(s) for s in states], dtype=np.float32)

    def value_oneply_checker(self, state):
        return self.engine.value_1ply(state)

    def value_oneply_checker_cubeful(self, state, match_state):
        return self.engine.value_1ply(state, match_state)
