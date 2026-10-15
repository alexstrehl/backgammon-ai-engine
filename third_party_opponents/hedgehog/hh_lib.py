"""ctypes wrapper for libhhagent.so (HedgeHog engine shim).

Conventions (HedgeHog): BG_BLACK=0, BG_WHITE=1.
Pips: 0 = White bar, 1..24 absolute board points, 25 = Black bar.
White moves 1->24 (home 19-24, bears off past 24); Black 24->1 (home 1-6).
Cube owner: 0 centered, 1 White, 2 Black.
eval(state, color): `color` on roll, probs from color's POV.

The shared library is loaded lazily, so importing this module needs no
shim. $HH_LIB is used exclusively when set; otherwise ``libhhagent.so``
next to this file. Build it with ``shim/build_shim.sh`` (README.md).
"""
import ctypes as C
import os

import numpy as np

HH_BLACK, HH_WHITE = 0, 1
ACT_NO_DOUBLE, ACT_DOUBLE_TAKE, ACT_DOUBLE_PASS, ACT_TOO_GOOD, \
    ACT_REDOUBLE_TAKE, ACT_REDOUBLE_PASS = range(6)

_DEFAULT_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "libhhagent.so")

_lib = None


def lib_path():
    """Path the shim will be loaded from, or None if not found."""
    env = os.environ.get("HH_LIB")
    if env:
        return env if os.path.exists(env) else None
    return _DEFAULT_LIB if os.path.exists(_DEFAULT_LIB) else None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    path = lib_path()
    if path is None:
        env = os.environ.get("HH_LIB")
        where = (f"$HH_LIB={env} (used exclusively when set)" if env
                 else _DEFAULT_LIB)
        raise RuntimeError(
            f"libhhagent.so not found (checked {where}). Build it with "
            "third_party_opponents/hedgehog/shim/build_shim.sh — see the README.")
    lib = C.CDLL(path)
    # The Jacoby flag is a shim global, default off; HH.set_jacoby sets it
    # per game from match_state.jacoby.
    if hasattr(lib, 'hh_set_jacoby'):
        lib.hh_set_jacoby.argtypes = [C.c_int]
        lib.hh_set_jacoby(0)
    lib.hh_create.restype = C.c_void_p
    lib.hh_create.argtypes = [C.c_char_p]
    lib.hh_free.argtypes = [C.c_void_p]
    lib.hh_initial_board.argtypes = [C.c_char_p]
    lib.hh_board_from_pips.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p]
    lib.hh_decode.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p,
                              C.POINTER(C.c_int), C.POINTER(C.c_int)]
    lib.hh_game_over.argtypes = [C.c_char_p]
    lib.hh_winner.argtypes = [C.c_char_p]
    lib.hh_points.argtypes = [C.c_char_p, C.c_int]
    lib.hh_eval.argtypes = [C.c_void_p, C.c_char_p, C.c_int,
                            C.POINTER(C.c_float)]
    lib.hh_num_moves.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_int]
    lib.hh_best_move.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_int,
                                 C.c_int, C.c_int, C.c_int, C.c_int,
                                 C.c_char_p]
    lib.hh_best_move_cubeless.argtypes = [C.c_void_p, C.c_char_p, C.c_int,
                                          C.c_int, C.c_int, C.c_int,
                                          C.c_char_p]
    lib.hh_list_moves.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_int,
                                  C.c_char_p, C.c_int]
    lib.hh_pick_best.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_int,
                                 C.c_int, C.c_int, C.c_int]
    lib.hh_pick_best_cubeless.argtypes = [C.c_void_p, C.c_char_p, C.c_int,
                                          C.c_int, C.c_int]
    lib.hh_double_action.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_int,
                                     C.c_int, C.c_int, C.POINTER(C.c_float)]
    lib.hh_take.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_int,
                            C.c_int, C.c_int]
    lib.hh_parse_ogid.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_int)]
    # Native benchmark loops, bound when the shim exports them; array args
    # are raw pointers into contiguous numpy buffers.
    P = C.c_void_p
    if hasattr(lib, "hh_bench_checker_cubeless"):
        lib.hh_bench_checker_cubeless.argtypes = [P, P, P, P, C.c_int, C.c_int,
                                                  P, P]
        lib.hh_bench_checker_cubeful.argtypes = [P, P, P, P, P, P, C.c_int,
                                                 C.c_int, P, P]
        lib.hh_bench_cube.argtypes = [P, P, P, P, P, C.c_int, C.c_int, P, P]
        lib.hh_bench_clock_floor.argtypes = [C.c_int, P]
    _lib = lib
    return _lib


class HH:
    """One loaded HedgeHog model (NN ensemble context)."""

    def __init__(self, model_path):
        self.ctx = _load().hh_create(os.fsencode(model_path))
        if not self.ctx:
            raise RuntimeError(f"could not load model {model_path}")

    @staticmethod
    def set_jacoby(on: bool):
        """Couple the shim's Jacoby flag to the arena's rule. A shim without
        hh_set_jacoby can only play Jacoby off."""
        lib = _load()
        if hasattr(lib, "hh_set_jacoby"):
            lib.hh_set_jacoby(1 if on else 0)
        elif on:
            raise RuntimeError(
                "match_state.jacoby is on but this shim has no hh_set_jacoby"
                "; rebuild it with shim/build_shim.sh (engine e8847f4).")

    def close(self):
        if self.ctx:
            _load().hh_free(self.ctx)
            self.ctx = None

    @staticmethod
    def initial_board():
        b = C.create_string_buffer(16)
        _load().hh_initial_board(b)
        return b.raw

    @staticmethod
    def board_from_pips(white_pos, black_pos):
        w = bytes(white_pos)
        b = bytes(black_pos)
        assert len(w) == 15 and len(b) == 15
        out = C.create_string_buffer(16)
        _load().hh_board_from_pips(w, b, out)
        return out.raw

    @staticmethod
    def decode(board):
        pips = C.create_string_buffer(26)
        colors = C.create_string_buffer(26)
        woff = C.c_int()
        boff = C.c_int()
        _load().hh_decode(board, pips, colors, C.byref(woff), C.byref(boff))
        return (list(pips.raw), list(colors.raw), woff.value, boff.value)

    @staticmethod
    def game_over(board):
        return bool(_load().hh_game_over(board))

    @staticmethod
    def winner(board):
        return _load().hh_winner(board)

    @staticmethod
    def points(board, winner):
        return _load().hh_points(board, winner)

    @staticmethod
    def num_moves(board, color, d1, d2):
        return _load().hh_num_moves(board, color, d1, d2)

    @staticmethod
    def list_moves(board, color, d1, d2, cap=4096):
        buf = C.create_string_buffer(16 * cap)
        n = _load().hh_list_moves(board, color, d1, d2, buf, cap)
        k = min(n, cap)
        return n, [buf.raw[16 * i:16 * (i + 1)] for i in range(k)]

    def eval(self, board, color):
        out = (C.c_float * 5)()
        _load().hh_eval(self.ctx, board, color, out)
        return list(out)

    def best_move(self, board, color, d1, d2, cube_value, cube_owner,
                  depth=0):
        out = C.create_string_buffer(16)
        n = _load().hh_best_move(self.ctx, board, color, d1, d2,
                                 cube_value, cube_owner, depth, out)
        return n, out.raw

    def best_move_cubeless(self, board, color, d1, d2, depth=0):
        """In-engine cubeless move (native movegen + eval_equity ranking)."""
        out = C.create_string_buffer(16)
        n = _load().hh_best_move_cubeless(self.ctx, board, color, d1, d2,
                                          depth, out)
        return n, out.raw

    def pick_best(self, boards, color, cube_value, cube_owner, depth=0):
        """boards: list of 16-byte successor boards. Returns best index."""
        blob = b"".join(boards)
        return _load().hh_pick_best(self.ctx, blob, len(boards), color,
                                    cube_value, cube_owner, depth)

    def pick_best_cubeless(self, boards, color, depth=0):
        """Cubeless money ranking (eval_equity, no Janowski/cube layer)."""
        blob = b"".join(boards)
        return _load().hh_pick_best_cubeless(self.ctx, blob, len(boards),
                                             color, depth)

    def double_action(self, board, color, cube_value, cube_owner, depth=0):
        eqs = (C.c_float * 3)()
        act = _load().hh_double_action(self.ctx, board, color, cube_value,
                                       cube_owner, depth, eqs)
        return act, list(eqs)

    def take(self, board, doubler, cube_value, cube_owner, depth=0):
        return bool(_load().hh_take(self.ctx, board, doubler, cube_value,
                                    cube_owner, depth))

    # Native benchmark loops. Items are the tuples fixedpool_bench builds
    # from board_to_hh boards. Each item is one call of the per-call entry
    # point, timed in C. Return (secs float64[n], choices): chosen boards
    # uint8[n, 16] for checker play, CubeAction int32[n] (-1 = cannot
    # double) for the cube.

    @staticmethod
    def _bench_lib():
        lib = _load()
        if not hasattr(lib, "hh_bench_checker_cubeless"):
            raise RuntimeError("this libhhagent.so has no hh_bench_* loops; "
                               "rebuild it with shim/build_shim.sh")
        return lib

    @staticmethod
    def _cols(items, ncol):
        boards = np.frombuffer(b"".join(it[0] for it in items),
                               dtype=np.uint8).copy()
        rest = np.ascontiguousarray(
            np.array([it[1:ncol] for it in items], dtype=np.int32)
            .reshape(len(items), ncol - 1))
        return boards, rest

    @staticmethod
    def _p(a):
        return a.ctypes.data

    def bench_checker_cubeless(self, items, depth=0):
        """items: (board, turn, d1, d2)."""
        lib = self._bench_lib()
        n = len(items)
        boards, r = self._cols(items, 4)
        turns = np.ascontiguousarray(r[:, 0])
        dice = np.ascontiguousarray(r[:, 1:3])
        secs = np.zeros(n, np.float64)
        out = np.zeros((n, 16), np.uint8)
        lib.hh_bench_checker_cubeless(self.ctx, self._p(boards),
                                      self._p(turns), self._p(dice), n,
                                      depth, self._p(secs), self._p(out))
        return secs, out

    def bench_checker_cubeful(self, items, depth=0):
        """items: (board, turn, d1, d2, cube_value, hh_owner)."""
        lib = self._bench_lib()
        n = len(items)
        boards, r = self._cols(items, 6)
        turns = np.ascontiguousarray(r[:, 0])
        dice = np.ascontiguousarray(r[:, 1:3])
        cv = np.ascontiguousarray(r[:, 3])
        own = np.ascontiguousarray(r[:, 4])
        secs = np.zeros(n, np.float64)
        out = np.zeros((n, 16), np.uint8)
        lib.hh_bench_checker_cubeful(self.ctx, self._p(boards),
                                     self._p(turns), self._p(dice),
                                     self._p(cv), self._p(own), n, depth,
                                     self._p(secs), self._p(out))
        return secs, out

    def bench_cube(self, items, depth=0):
        """items: (board, turn, cube_value, hh_owner)."""
        lib = self._bench_lib()
        n = len(items)
        boards, r = self._cols(items, 4)
        turns = np.ascontiguousarray(r[:, 0])
        cv = np.ascontiguousarray(r[:, 1])
        own = np.ascontiguousarray(r[:, 2])
        secs = np.zeros(n, np.float64)
        dec = np.zeros(n, np.int32)
        lib.hh_bench_cube(self.ctx, self._p(boards), self._p(turns),
                          self._p(cv), self._p(own), n, depth,
                          self._p(secs), self._p(dec))
        return secs, dec

    @staticmethod
    def bench_clock_floor(n=10000):
        """Seconds per empty timing bracket (n samples)."""
        lib = HH._bench_lib()
        secs = np.zeros(n, np.float64)
        lib.hh_bench_clock_floor(n, HH._p(secs))
        return secs

    @staticmethod
    def parse_ogid(ogid):
        out = C.create_string_buffer(16)
        c = C.c_int()
        r = _load().hh_parse_ogid(ogid.encode(), out, C.byref(c))
        if r != 0:
            raise ValueError(f"bad ogid {ogid}")
        return out.raw, c.value
