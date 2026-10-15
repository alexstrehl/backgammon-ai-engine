"""HedgeHog shim native benchmark loops (hh_bench_*) make the same decisions
as the per-call entry points they time. Needs libhhagent.so with the loops
and a HedgeHog .ogxf model ($HH_MODEL); skipped otherwise."""
import os
import random

import numpy as np
import pytest

from third_party_opponents.hedgehog import hh_lib

_LIB = hh_lib.lib_path()
_MODEL = os.environ.get("HH_MODEL", os.path.join(
    os.path.dirname(hh_lib.__file__), "default-small_v1.0_2026-05-13.ogxf"))


def _has_loops():
    if _LIB is None:
        return False
    return hasattr(hh_lib._load(), "hh_bench_checker_cubeless")


pytestmark = [
    pytest.mark.skipif(not _has_loops(),
                       reason="libhhagent.so with hh_bench_* not built"),
    pytest.mark.skipif(not os.path.exists(_MODEL),
                       reason="no HedgeHog .ogxf model available"),
]


def _positions(n, seed=7):
    """(board, color) pairs from random playouts, game-over boards skipped."""
    HH = hh_lib.HH
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board, color = HH.initial_board(), rng.randrange(2)
        while not HH.game_over(board) and len(out) < n:
            out.append((board, color))
            k, succ = HH.list_moves(board, color, rng.randint(1, 6),
                                    rng.randint(1, 6))
            if k:
                board = rng.choice(succ)
            color = 1 - color
    return out


@pytest.fixture(scope="module")
def hh():
    h = hh_lib.HH(_MODEL)
    yield h
    h.close()


def test_bench_loops_match_per_call(hh):
    rng = random.Random(11)
    pos = _positions(300)
    cubes = [(1, 0), (2, 1), (2, 2), (4, 1), (8, 2), (64, 1)]
    chk = [(b, c, rng.randint(1, 6), rng.randint(1, 6)) for b, c in pos]
    cbf = [it + rng.choice(cubes) for it in chk]
    cub = [(b, c) + rng.choice(cubes[:-1]) for b, c in pos]

    secs, got = hh.bench_checker_cubeless(chk, 0)
    want = [hh.best_move_cubeless(*it, 0)[1] for it in chk]
    assert [bytes(r) for r in got] == want
    assert secs.shape == (len(chk),) and (secs > 0).all()

    secs, got = hh.bench_checker_cubeful(cbf, 0)
    want = [hh.best_move(*it, 0)[1] for it in cbf]
    assert [bytes(r) for r in got] == want
    assert (secs > 0).all()

    secs, got = hh.bench_cube(cub, 0)
    want = [hh.double_action(*it, 0)[0] for it in cub]
    assert got.tolist() == want
    assert (secs > 0).all()
    assert set(want) - {-1}  # some positions can double


def test_bench_loops_depth1_match(hh):
    rng = random.Random(3)
    pos = _positions(8, seed=5)
    chk = [(b, c, rng.randint(1, 6), rng.randint(1, 6)) for b, c in pos]
    _, got = hh.bench_checker_cubeless(chk, 1)
    assert [bytes(r) for r in got] == \
        [hh.best_move_cubeless(*it, 1)[1] for it in chk]
    _, got = hh.bench_checker_cubeful([it + (2, 0) for it in chk], 1)
    assert [bytes(r) for r in got] == \
        [hh.best_move(*it, 2, 0, 1)[1] for it in chk]
    _, got = hh.bench_cube([(b, c, 1, 0) for b, c in pos], 1)
    assert got.tolist() == [hh.double_action(b, c, 1, 0, 1)[0]
                            for b, c in pos]


def test_clock_floor():
    fl = hh_lib.HH.bench_clock_floor(1000)
    assert fl.shape == (1000,) and (fl >= 0).all() and fl.mean() < 1e-5
