"""Open Sage through its native C API (``bgsage_mobile_*``, cpp/include/bgbot/mobile.h).

One engine handle loads the production nets and the bearoff database once;
each call is single-threaded and returns JSON. Plies are XG-numbered and the
API supports 1 (0-ply) and 2 (1-ply) only. At 1-ply every candidate is
re-scored through the cube-aware tree (as ``BgBotAnalyzer``), which scores every
opponent reply; cube decisions pre-rank replies with PubEval and keep the top 15
when more than 16 are legal.

Build (once per host): ``bash third_party_opponents/opensage/opensage_mobile_build.sh``, or
point ``BGSAGE_MOBILE_LIB`` at an existing ``libbgsage_mobile.so``.
"""

import ctypes as C
import json
import os
import sys

from agents import Agent, CubeOffer

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIB = os.path.join(_HERE, "opensage_mobile_build", "libbgsage_mobile.so")
BUDGET_MS = 1000  # the API maximum; checked between candidates


class Request(C.Structure):
    _fields_ = [("board", C.c_int * 26)] + [(n, C.c_int) for n in (
        "die1", "die2", "ply", "cube_value", "cube_owner", "away1", "away2",
        "is_crawford", "jacoby", "beaver", "budget_ms")]


_OWNER = {"centered": 0, "player": 1, "opponent": 2}


class SageMobile:
    """One native engine handle (not thread-safe; one per process)."""

    def __init__(self, lib_path=None):
        from third_party_opponents.opensage.opensage_agents import assert_sage_version
        assert_sage_version()
        from bgsage.weights import default_weights, bearoff_db_path
        path = lib_path or os.environ.get("BGSAGE_MOBILE_LIB", DEFAULT_LIB)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"libbgsage_mobile.so not found at {path}: build it with "
                "bash third_party_opponents/opensage/opensage_mobile_build.sh "
                "or set BGSAGE_MOBILE_LIB")
        lib = C.CDLL(path)
        lib.bgsage_mobile_create.argtypes = [
            C.c_char_p, C.POINTER(C.c_char_p), C.POINTER(C.c_int), C.c_int,
            C.c_char_p]
        lib.bgsage_mobile_create.restype = C.c_void_p
        lib.bgsage_mobile_analyze.argtypes = [C.c_void_p, C.c_int,
                                              C.POINTER(Request)]
        lib.bgsage_mobile_analyze.restype = C.c_void_p
        lib.bgsage_mobile_free.argtypes = [C.c_void_p]
        lib.bgsage_mobile_destroy.argtypes = [C.c_void_p]
        lib.bgsage_mobile_last_error.restype = C.c_char_p
        w = default_weights()
        paths = (C.c_char_p * len(w.weight_paths_list))(
            *[str(p).encode() for p in w.weight_paths_list])
        hidden = (C.c_int * len(w.hidden_sizes_list))(*w.hidden_sizes_list)
        h = lib.bgsage_mobile_create(w.strategy_type.encode(), paths, hidden,
                                     len(paths), str(bearoff_db_path()).encode())
        if not h:
            raise RuntimeError(f"bgsage_mobile_create: {lib.bgsage_mobile_last_error()}")
        self._lib, self._h = lib, h
        self._req = Request()

    def close(self):
        if self._h:
            self._lib.bgsage_mobile_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def request(self, board, ply, die1=1, die2=1, cube_value=1,
                cube_owner="centered", jacoby=True, beaver=False, out=None):
        """Fill ``out`` (default: the reusable request struct), money play."""
        r = self._req if out is None else out
        r.board[:] = board
        r.die1, r.die2, r.ply = die1, die2, ply
        r.cube_value, r.cube_owner = cube_value, _OWNER[cube_owner]
        r.away1 = r.away2 = r.is_crawford = 0
        r.jacoby, r.beaver, r.budget_ms = int(jacoby), int(beaver), BUDGET_MS
        return r

    def raw(self, operation, req):
        """One native call -> JSON bytes, or None (invalid / over budget)."""
        p = self._lib.bgsage_mobile_analyze(self._h, operation, C.byref(req))
        if not p:
            return None
        try:
            return C.string_at(p)
        finally:
            self._lib.bgsage_mobile_free(p)

    def analyze(self, operation, **kw):
        """Parsed result, or None when the call ran past BUDGET_MS."""
        out = self.raw(operation, self.request(**kw))
        if out is None:
            err = self._lib.bgsage_mobile_last_error().decode()
            if err == "native budget exceeded":
                return None
            raise RuntimeError(f"bgsage_mobile_analyze: {err}")
        return json.loads(out)


class OpenSageMobileAgent(Agent):
    """Cubeful money Open Sage on the native C API: decisions equal
    ``OpenSageAnalyzerAgent`` at production settings (sage_plies 1 or 2)
    without its per-cube-call network reload. A call that runs past the
    API's 1 s budget is answered by ``OpenSageAnalyzerAgent`` instead
    (counted in ``fallbacks``)."""

    def __init__(self, sage_plies: int = 1, beaver: bool = False):
        if sage_plies not in (1, 2):
            raise ValueError("the native API supports sage_plies 1 (0-ply) and 2 (1-ply)")
        self._eng = SageMobile()
        self.sage_plies = sage_plies
        self.beaver = beaver
        self.is_cubeful = True
        self.fallbacks = 0
        self._an = None

    def _analyzer(self):
        self.fallbacks += 1
        print(f"opensage-native: over the {BUDGET_MS} ms budget, analyzer "
              f"fallback #{self.fallbacks} (pid {os.getpid()})",
              file=sys.stderr, flush=True)
        if self._an is None:
            from third_party_opponents.opensage.opensage_agents import OpenSageAnalyzerAgent
            self._an = OpenSageAnalyzerAgent(self.sage_plies, beaver=self.beaver)
        return self._an

    def _kw(self, state, match_state):
        from third_party_opponents.opensage.opensage_agents import OpenSageAgent, OpenSageAnalyzerAgent
        return dict(board=OpenSageAgent._to_sage_board(state, state.turn),
                    ply=self.sage_plies,
                    cube_value=min(int(match_state.cube_value), 4096),
                    cube_owner=OpenSageAnalyzerAgent._owner_str(state, match_state),
                    jacoby=match_state.jacoby, beaver=self.beaver)

    def choose_checker_action_cubeful(self, state, dice, match_state):
        from backgammon_engine import get_legal_plays
        from third_party_opponents.opensage.opensage_agents import OpenSageAgent
        plays = get_legal_plays(state, dice)
        if not plays:
            return None
        mover = state.turn
        r = self._eng.analyze(0, die1=dice[0], die2=dice[1],
                              **self._kw(state, match_state))
        if r is None:
            return self._analyzer().choose_checker_action_cubeful(
                state, dice, match_state)
        legal = {tuple(OpenSageAgent._to_sage_board(ns, mover)): ns
                 for _play, ns in plays}
        for m in r["moves"]:
            ns = legal.get(tuple(m["board"]))
            if ns is not None:
                return ns, 0.0
        raise RuntimeError("OpenSage move-gen mismatch (native API)")

    def _cube(self, state, match_state):
        r = self._eng.analyze(1, **self._kw(state, match_state))
        if r is None:
            a = self._analyzer()._cube_action(state, match_state)
            r = {"should_double": a.should_double, "should_take": a.should_take}
        return r

    def offer_double(self, state, match_state) -> CubeOffer:
        r = self._cube(state, match_state)
        return CubeOffer(should_double=bool(r["should_double"]),
                         _cache={"should_take": bool(r["should_take"])})

    def respond_to_double(self, state, match_state, hint=None) -> bool:
        if hint is not None and hint._cache and "should_take" in hint._cache:
            return bool(hint._cache["should_take"])
        return bool(self._cube(state, match_state)["should_take"])
