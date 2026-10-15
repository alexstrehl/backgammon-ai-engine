"""The C move generator's play ORDER is frozen.

Callers break exact ties by the first candidate in get_legal_plays order
(c_api/puretd.h, every argmax over successors), so a change to the walk
that keeps the play SET but reorders it silently changes decisions on
ties. This test pins the full ordered output of c_engine get_legal_plays:

  * a frozen SHA-256 of the ordered (resulting board, num_moves) lists,
    and one of the stored moves, over a fixed position set (the committed
    fixed pool + generator-independent random boards from fixed seeds),
    every ordered roll (36: non-doubles in both orders, as callers pass
    them unsorted);
  * C vs the Python reference generator (backgammon_engine), same order
    element by element, on a random subset (an independent oracle: the
    Python walk is the unpruned DFS the C walk must reproduce).

The frozen hashes are the unpruned walk's (every 4! ordering), which
the pruned walk reproduces.
If this fails, the generator's output order changed: that is a decision
change, not a refactor. Update the hashes only deliberately.

BG_MOVEGEN_LIB=/path/to/candidate.so tests a candidate build instead of
c_engine/libbg_engine.so (check a change before rebuilding in place).
"""
import ctypes
import hashlib
import json
import os
import random
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "c_engine"))

from backgammon_engine import BLACK, WHITE, BoardState, get_legal_plays  # noqa: E402

bf = pytest.importorskip("bg_fast")

POOL = os.path.join(REPO, "experiments", "benchmarks", "fixedpool_positions.json")
ROLLS = [(a, b) for a in range(1, 7) for b in range(1, 7)]
PLAY_INTS = ctypes.sizeof(bf._CPlay) // 4   # 8 move ints, num_moves, 29 board ints

# Pool (8000) + 4 x 600 random boards, 36 ordered rolls each.
FROZEN_ORDER_SHA = "d6562944097d29523446e4b7b99d28aee16067e1b39e964efdc7aec24a8595c8"
FROZEN_MOVES_SHA = "9a165db2888501442e97cdafa224d77050e9622861d460a87efa7d1e99f040b3"


def _lib():
    path = os.environ.get("BG_MOVEGEN_LIB") or bf._lib._name
    lib = ctypes.CDLL(path)
    lib.get_legal_plays.argtypes = [
        ctypes.POINTER(bf._CBoardState), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(bf._CPlay), ctypes.c_int]
    lib.get_legal_plays.restype = ctypes.c_int
    return lib


def _place(rng, pts, n, lo, hi, sign):
    """Put n checkers of `sign` on random points in [lo, hi] not held by
    the other color. False if every such point is held."""
    free = [i for i in range(lo, hi + 1) if pts[i] * sign >= 0]
    if not free:
        return False
    for _ in range(n):
        pts[rng.choice(free)] += sign
    return True


def _random_board(rng, kind):
    """A legal random board from rng alone (independent of any move
    generator, so the position set never depends on the code under test).
    kind: contact / bar / bearoff / endgame (1-4 mover checkers left)."""
    while True:
        b = _try_board(rng, kind)
        if b is not None:
            return b


def _try_board(rng, kind):
    turn = rng.randint(0, 1)
    ok = True
    pts = [0] * 24
    bar = [0, 0]
    off = [0, 0]
    # mover = turn; sign +1 = WHITE. Home: WHITE 0-5, BLACK 18-23.
    for side in (WHITE, BLACK):
        sign = 1 if side == WHITE else -1
        home = (0, 5) if side == WHITE else (18, 23)
        mover = side == turn
        if kind == "contact" or not mover:
            if kind == "bar" and not mover:
                # opponent holds a partly closed home board against the mover
                n_on = 15
                ohome = (18, 23) if side == BLACK else (0, 5)
                k = rng.randint(2, 6)
                made = rng.sample(range(ohome[0], ohome[1] + 1), k)
                for i in made:
                    if pts[i] * sign >= 0 and n_on >= 2:
                        pts[i] += 2 * sign
                        n_on -= 2
                ok &= _place(rng, pts, n_on, 0, 23, sign)
            else:
                b = rng.choice([0, 0, 0, 0, 1])
                o = rng.choice([0, 0, 0, rng.randint(0, 10)])
                bar[side] = b
                off[side] = o
                ok &= _place(rng, pts, 15 - b - o, 0, 23, sign)
        elif kind == "bar":
            b = rng.randint(1, 3)
            bar[side] = b
            ok &= _place(rng, pts, 15 - b, 0, 23, sign)
        elif kind == "bearoff":
            o = rng.randint(0, 12)
            off[side] = o
            ok &= _place(rng, pts, 15 - o, home[0], home[1], sign)
        else:  # endgame
            left = rng.randint(1, 4)
            off[side] = 15 - left
            ok &= _place(rng, pts, left, home[0], home[1], sign)
    if not ok:
        return None
    return BoardState(points=pts, bar=bar, off=off, turn=turn)


def _positions():
    pool = json.load(open(POOL))["positions"]
    out = [BoardState(points=list(p["points"]), bar=list(p["bar"]),
                      off=list(p["off"]), turn=p["turn"]) for p in pool]
    for seed, kind in enumerate(("contact", "bar", "bearoff", "endgame")):
        rng = random.Random(1000 + seed)
        out += [_random_board(rng, kind) for _ in range(600)]
    return out


def _ordered(lib, c, d1, d2, buf, arr):
    """(n, 38) int32 copy of the ordered plays; move slots past num_moves
    zeroed (they hold stale buffer contents)."""
    n = lib.get_legal_plays(ctypes.byref(c), d1, d2, buf, bf.MAX_PLAYS)
    a = arr[:n].copy()
    nm = a[:, 8]
    for k in range(4):
        a[nm <= k, 2 * k:2 * k + 2] = 0
    return a


@pytest.fixture(scope="module")
def positions():
    return _positions()


def test_frozen_order_hash(positions):
    lib = _lib()
    buf = (bf._CPlay * bf.MAX_PLAYS)()
    arr = np.frombuffer(buf, dtype=np.int32).reshape(bf.MAX_PLAYS, PLAY_INTS)
    h_order = hashlib.sha256()
    h_moves = hashlib.sha256()
    for s in positions:
        c = bf._python_to_c(s)
        for d1, d2 in ROLLS:
            a = _ordered(lib, c, d1, d2, buf, arr)
            h_order.update(np.int32(len(a)).tobytes())
            h_order.update(a[:, 8:].tobytes())
            h_moves.update(a[:, :8].tobytes())
    order, moves = h_order.hexdigest(), h_moves.hexdigest()
    assert order == FROZEN_ORDER_SHA, (
        f"get_legal_plays ORDER (or set) changed: {order}. Ties go to the "
        f"first candidate, so this changes decisions.")
    assert moves == FROZEN_MOVES_SHA, (
        f"get_legal_plays stored moves changed (boards and order did not): {moves}")


def test_c_order_equals_python_reference(positions):
    """Element-by-element: same boards, num_moves and moves, same order."""
    lib = _lib()
    buf = (bf._CPlay * bf.MAX_PLAYS)()
    arr = np.frombuffer(buf, dtype=np.int32).reshape(bf.MAX_PLAYS, PLAY_INTS)
    rng = random.Random(7)
    subset = rng.sample(positions[:8000], 50) + positions[8000::24]
    checked = 0
    for s in subset:
        c = bf._python_to_c(s)
        for d1, d2 in ROLLS:
            a = _ordered(lib, c, d1, d2, buf, arr)
            py = get_legal_plays(s, (d1, d2))
            assert len(a) == len(py), f"{s._key()} {d1}{d2}: C {len(a)} vs Python {len(py)}"
            for k, (play, ns) in enumerate(py):
                row = a[k]
                board = list(ns.points) + list(ns.bar) + list(ns.off) + [ns.turn]
                assert row[9:].tolist() == board and row[8] == len(play), (
                    f"{s._key()} roll {d1}{d2}: play {k} differs from the Python "
                    f"reference (order or content)")
                mv = [x for m in play for x in m]
                assert row[:2 * len(play)].tolist() == mv, (
                    f"{s._key()} roll {d1}{d2}: play {k} stored moves differ")
            checked += 1
    assert checked == len(subset) * 36
