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
Prob5Jan (prob5 + Janowski money cube, janowski_agent.JanowskiAgent at
its defaults): `Engine(prob5_model, janowski=True, x=None,
board_sanity=True, bearoff=True)` / `CAgent(..., janowski=True)`; the
bearoff tables come from `BearoffTable` (bearoff.c_table_path()'s flat
file, one mmap per process and backend); `JanowskiC` mirrors janowski.py's
conversion and bearoff.py's exact values for parity tests.
A .pt is exported to BGNN in a private 0700 temp dir (removed at exit);
the export names the net's encoder (export_weights.py format 2) and
bgc_open resolves it through the C API's encoder table (Engine.encoder;
`encoder_info(name)` reads the table). A .bin path is opened as is: a
legacy format-1 file (no encoder name) opens by its input width.
An Engine is confined to one thread; open one per thread / process.
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
from bg_cstructs import _CBoardState, _CPlay, _python_to_c, _c_to_python, MAX_PLAYS  # noqa: E402

BACKENDS = ("torch", "simd", "cblas", "ref")
PLAY_BACKEND = "simd"       # fastest at play batch sizes

# puretd.h BGC_ABI_VERSION; bump both on any public struct or signature change.
ABI_VERSION = 1
_ABI_TAG = f"puretd-abi:{ABI_VERSION}\0".encode()

KIND_PROB, KIND_EQUITY, KIND_CUBEFUL, KIND_PROB5 = 0, 1, 2, 3
KIND_PROB5JAN = 6           # bgc_open_janowski: prob5 + Janowski money cube
# The kind follows from the export's encoder (Engine.encoder) and output
# mode: KIND_CUBEFUL covers the 200-input cubeful_perspective196 nets.
# bgc_collect returns COLLECT_UNSUPPORTED for an encoder whose table entry
# has no C collector (encoder_info(name)["collect"]).
COLLECT_UNSUPPORTED = -3
_OUTPUT_MODE = {KIND_PROB: "probability", KIND_EQUITY: "equity",
                KIND_CUBEFUL: "equity", KIND_PROB5: "prob5",
                KIND_PROB5JAN: "equity"}


# ── C structs ──────────────────────────────────────────────────────

class CCube(ctypes.Structure):
    _fields_ = [("owner", ctypes.c_int), ("value", ctypes.c_int),
                ("jacoby", ctypes.c_int)]


class CCubeEval(ctypes.Structure):
    _fields_ = [("plies", ctypes.c_int), ("v_no_double", ctypes.c_float),
                ("v_double_take", ctypes.c_float), ("take", ctypes.c_int)]


class CJanOpts(ctypes.Structure):
    """bgc_jan_opts (puretd.h)."""
    _fields_ = [("x", ctypes.c_double), ("board_sanity", ctypes.c_int)]


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
        "n_base", "ncube", "input_size", "collect")]


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
    """PURETD_BACKEND, else simd if built, else the first built backend."""
    env = os.environ.get("PURETD_BACKEND", "").strip()
    if env:
        return env
    avail = available_backends()
    if not avail:
        raise RuntimeError(f"no libpuretd_<backend>.so in {lib_dir()}; run c_api/build.sh")
    return PLAY_BACKEND if PLAY_BACKEND in avail else avail[0]


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
        _build_failed[path] = (f"{path} {what} and `{_rebuild_cmd(backend)}` "
                               f"did not build it (exit {rc}: {tail})")
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
        print(f"  NOTE: {how} {path} on demand ({_rebuild_cmd(backend)}).",
              flush=True)
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
        "bgc_play_game": ([E, E, P(CGameOpts), P(CRng), P(CGameResult),
                           RECORD_FN, ctypes.c_void_p], I),
        "bgc_collect": ([E, P(CCollectOpts), ctypes.c_ulonglong, I, P(F), P(F),
                         ctypes.c_long, P(I)], ctypes.c_long),
        "bgc_bench_checker": ([E, BS, P(I), P(CCube), I, I, P(D)], I),
        "bgc_bench_cube": ([E, BS, P(CCube), I, I, P(D)], I),
        "bgc_bench_selfplay": ([E, P(CGameOpts), ctypes.c_ulonglong, I, I, I,
                                P(D), P(I), P(ctypes.c_longlong)], I),
    }
    D5, BO = P(D), ctypes.c_void_p
    sig.update({
        "bgc_open_janowski": ([ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t], E),
        "bgc_set_jan": ([E, P(CJanOpts)], I),
        "bgc_bearoff_open": ([ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t], BO),
        "bgc_bearoff_close": ([BO], None),
        "bgc_set_bearoff": ([E, BO], I),
        "bgc_jan_equity": ([D5, I, D, I], D),
        "bgc_jan_equity_folded": ([D5, I, D, I], D),
        "bgc_jan_cubeless_equity": ([D5], D),
        "bgc_jan_fold": ([D, D], D),
        "bgc_jan_money_cube_action": ([D, D, I, I, P(I)], I),
        "bgc_bo_exact_probs5": ([BO, BS, D5], I),
        "bgc_bo_race_equity": ([BO, BS, I, I, P(D)], I),
        "bgc_bo_race_cube_values": ([BO, BS, I, P(D), P(D)], I),
        "bgc_bo_ts_win_prob": ([BO, BS, P(D)], I),
    })
    for name, (args, res) in sig.items():
        fn = getattr(L, name)
        fn.argtypes = args
        fn.restype = res
    _libs[backend] = L
    return L


# bgc_struct_sizes order (puretd.h).
_ABI_STRUCTS = (_CBoardState, _CPlay, CCube, CCubeEval, CSearch, CEncoderInfo,
                CJanOpts, CRng, CGameOpts, CGameResult, CCollectOpts)


def _abi_mismatch(L):
    """Why the loaded library `L` does not match this module's ctypes
    declarations, or None."""
    if not hasattr(L, "bgc_abi_version"):
        return f"built before the C API ABI check (this module: ABI {ABI_VERSION})"
    L.bgc_abi_version.restype = ctypes.c_int
    v = L.bgc_abi_version()
    if v != ABI_VERSION:
        return f"C API ABI {v}, this module expects {ABI_VERSION}"
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
    collect (bool: bgc_collect writes its rows); None when bgc_open does
    not accept the name."""
    info = CEncoderInfo()
    lib = load_lib(backend or default_backend())
    if lib.bgc_encoder_lookup(str(name).encode(), ctypes.byref(info)) != 0:
        return None
    d = {n: getattr(info, n) for n, _ in CEncoderInfo._fields_}
    d["collect"] = bool(d["collect"])
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


# ── Prob5Jan ───────────────────────────────────────────────────────

def _jan_lib(backend):
    lib = load_lib(backend)
    if not hasattr(lib, "bgc_open_janowski"):
        raise RuntimeError(f"{lib_path(backend)} predates Prob5Jan; rebuild it: "
                           f"bash c_api/build.sh {backend}")
    return lib


class BearoffTable:
    """bgc_bearoff: bearoff.py's tables mmapped from the flat file
    (path None: bearoff.c_table_path(), exported on demand). Read-only;
    engines holding it keep a reference. `BearoffTable.default()` is one
    shared instance per (backend, path)."""

    _cache = {}

    def __init__(self, path=None, backend=None):
        import bearoff
        self.backend = backend or default_backend()
        self.lib = _jan_lib(self.backend)
        self.path = path or bearoff.c_table_path()
        err = ctypes.create_string_buffer(1024)
        self.h = self.lib.bgc_bearoff_open(self.path.encode(), err, len(err))
        if not self.h:
            raise RuntimeError(err.value.decode(errors="replace"))

    @classmethod
    def default(cls, backend=None, path=None):
        key = (backend or default_backend(), path)
        if key not in cls._cache:
            cls._cache[key] = cls(path, key[0])
        return cls._cache[key]

    def close(self):
        if getattr(self, "h", None):
            self.lib.bgc_bearoff_close(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _bearoff_table(bearoff, backend):
    """Engine's bearoff= argument -> a BearoffTable or None."""
    if bearoff is True:
        return BearoffTable.default(backend)
    if bearoff is None or bearoff is False:
        return None
    if isinstance(bearoff, str):
        return BearoffTable.default(backend, bearoff)
    if isinstance(bearoff, BearoffTable):
        if bearoff.lib is not load_lib(backend):
            raise ValueError("BearoffTable opened on another backend library")
        return bearoff
    raise TypeError(f"bearoff must be True/False, a path or a BearoffTable "
                    f"(got {type(bearoff).__name__})")


def _probs5(p):
    a = np.ascontiguousarray(np.asarray(p, dtype=np.float64).reshape(5))
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class JanowskiC:
    """janowski.py's conversion and bearoff.py's exact values through the
    C API (scalar forms), for parity tests. `bearoff`: a BearoffTable, or
    None for the default table (opened on first use)."""

    def __init__(self, backend=None, bearoff=None):
        self.backend = backend or default_backend()
        self.lib = _jan_lib(self.backend)
        self._bo = bearoff

    @property
    def table(self):
        if self._bo is None:
            self._bo = BearoffTable.default(self.backend)
        return self._bo

    def janowski_equity(self, probs, persp, x, jacoby):
        a, ptr = _probs5(probs)
        return self.lib.bgc_jan_equity(ptr, int(persp), float(x), int(bool(jacoby)))

    def janowski_equity_folded(self, probs, persp, x, jacoby):
        a, ptr = _probs5(probs)
        return self.lib.bgc_jan_equity_folded(ptr, int(persp), float(x), int(bool(jacoby)))

    def cubeless_equity(self, probs):
        a, ptr = _probs5(probs)
        return self.lib.bgc_jan_cubeless_equity(ptr)

    def fold_cube_decision(self, nd, dt):
        return self.lib.bgc_jan_fold(float(nd), float(dt))

    def money_cube_action(self, nd, dt, persp, jacoby):
        """(double, take)."""
        take = ctypes.c_int()
        d = self.lib.bgc_jan_money_cube_action(float(nd), float(dt), int(persp),
                                               int(bool(jacoby)), ctypes.byref(take))
        return bool(d), bool(take.value)

    def _check(self, r, what):
        if r < 0:
            raise RuntimeError(f"{what} failed")
        return r == 1

    def exact_probs5(self, state):
        """bearoff.exact_probs5; None when not bearoff class."""
        p = np.zeros(5)
        ok = self._check(self.lib.bgc_bo_exact_probs5(
            self.table.h, ctypes.byref(_python_to_c(state)),
            p.ctypes.data_as(ctypes.POINTER(ctypes.c_double))), "bgc_bo_exact_probs5")
        return p if ok else None

    def race_equity(self, state, persp, with_own_fold=False):
        v = ctypes.c_double()
        ok = self._check(self.lib.bgc_bo_race_equity(
            self.table.h, ctypes.byref(_python_to_c(state)), int(persp),
            int(bool(with_own_fold)), ctypes.byref(v)), "bgc_bo_race_equity")
        return v.value if ok else None

    def race_cube_values(self, state, persp):
        nd, dt = ctypes.c_double(), ctypes.c_double()
        ok = self._check(self.lib.bgc_bo_race_cube_values(
            self.table.h, ctypes.byref(_python_to_c(state)), int(persp),
            ctypes.byref(nd), ctypes.byref(dt)), "bgc_bo_race_cube_values")
        return (nd.value, dt.value) if ok else None

    def ts_win_prob(self, state):
        p = ctypes.c_double()
        ok = self._check(self.lib.bgc_bo_ts_win_prob(
            self.table.h, ctypes.byref(_python_to_c(state)), ctypes.byref(p)),
            "bgc_bo_ts_win_prob")
        return p.value if ok else None


# ── Engine ─────────────────────────────────────────────────────────

class Engine:
    """One bgc_engine: a model on one backend. Thread-confined.
    janowski=True: a prob5 model as Prob5Jan (x None = janowski.DEFAULT_X;
    bearoff: True = the default BearoffTable, a path, a BearoffTable, or
    False = no exact bearoff)."""

    def __init__(self, model, backend=None, root_filter=0.0, janowski=False, x=None,
                 board_sanity=True, bearoff=True):
        self.backend = backend or default_backend()
        self.lib = _jan_lib(self.backend) if janowski else load_lib(self.backend)
        self._bearoff = None
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
        # -DBGC_BF16 build with the bf16 forward in use for 1-ply rows.
        self.bf16 = "bf16" in self.describe.split()
        self._out_state = _CBoardState()
        self._val = ctypes.c_float()
        self._scores = (ctypes.c_float * MAX_PLAYS)()
        self._states = (_CBoardState * MAX_PLAYS)()
        self.root_filter = 0.0
        self.set_root_filter(root_filter)
        if janowski:
            self.set_jan(x, board_sanity)
            self.set_bearoff(bearoff)

    def set_jan(self, x=None, board_sanity=True):
        """Prob5Jan's x (None: janowski.DEFAULT_X) and board_sanity."""
        from janowski import DEFAULT_X
        o = CJanOpts(DEFAULT_X if x is None else float(x), int(bool(board_sanity)))
        if self.lib.bgc_set_jan(self.h, ctypes.byref(o)) != 0:
            raise ValueError(f"bgc_set_jan failed (kind {self.kind}, x {x!r})")
        self.x, self.board_sanity = o.x, bool(board_sanity)

    def set_bearoff(self, table=True):
        """Prob5Jan's exact bearoff tables: True (default table), a path,
        a BearoffTable, or None/False (off)."""
        t = _bearoff_table(table, self.backend)
        if self.lib.bgc_set_bearoff(self.h, t.h if t is not None else None) != 0:
            raise ValueError(f"bgc_set_bearoff failed (kind {self.kind})")
        self._bearoff = t                         # keeps the mmap alive

    def set_root_filter(self, thresh):
        """1-ply root move filter threshold (bgc_set_search); 0 = full width."""
        t = float(thresh or 0.0)
        if t and self.kind == KIND_PROB5JAN:
            raise ValueError("the C Prob5Jan has no root filter (full width only)")
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
        (train_prob5._collect_worker's targets). enc/tgt are views of the
        buffers.

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

    def __init__(self, model, plies=0, backend=None, engine=None, root_filter=None,
                 janowski=False, **jan_kw):
        """root_filter: 1-ply root filter threshold (None: PURETD_ROOT_FILTER,
        ignored for janowski; a nonzero explicit one raises for janowski).
        janowski: Prob5Jan (jan_kw: Engine's x, board_sanity, bearoff)."""
        if plies not in (0, 1):
            raise ValueError(f"CAgent supports plies 0 or 1 (got {plies})")
        self.engine = engine if engine is not None else Engine(
            model, backend, janowski=janowski, **jan_kw)
        if root_filter is None and engine is None and self.engine.kind != KIND_PROB5JAN:
            root_filter = root_filter_from_env()
        if root_filter is not None:
            self.engine.set_root_filter(root_filter)
        self.plies = plies
        self.oneply = plies == 1
        self.twoply_k = 0
        self.kind = self.engine.kind
        self.is_cubeful = self.kind in (KIND_CUBEFUL, KIND_PROB5JAN)
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
