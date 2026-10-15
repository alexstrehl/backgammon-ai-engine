"""get_legal_plays output is frozen across the dedup table's reuse.

The dedup table is thread-local and reused across calls (a generation
stamp marks live slots), so this pins the ordered output, byte for byte,
over call sequences that stress that reuse:

  * seeded self-play trajectories (contact, bar entry, bear-off, forced
    passes as they arise in play), every ordered roll per position;
  * truncated output buffers (max_plays 1, 2, 3, 5, 17);
  * the same workload in 4 concurrent threads (ctypes drops the GIL);
  * the generation counter wrapping to 0 (test-hook build, -DBG_TEST_HOOKS).

The digests are the generator's frozen output; update only deliberately.
BG_MOVEGEN_LIB=/path/to/candidate.so tests a candidate build.
"""
import ctypes
import hashlib
import os
import random
import shutil
import subprocess
import sys
import threading

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "c_engine"))

bf = pytest.importorskip("bg_fast")

ROLLS = [(a, b) for a in range(1, 7) for b in range(1, 7)]
PLAY_INTS = ctypes.sizeof(bf._CPlay) // 4   # 8 move ints, num_moves, 29 board ints
CAPS = (1, 2, 3, 5, 17)

# 60 games from seed 2026, 36 ordered rolls per position, then every
# 7th position again with each truncated cap.
FROZEN_SHA = "4dd6254e49a535bd177fc474fcfa2e2c9722e8fa1434c73303cae4b8d9f295e9"
FROZEN_CALLS = 366228
FROZEN_PASSES = 10137


def _lib(path=None):
    path = path or os.environ.get("BG_MOVEGEN_LIB") or bf._lib._name
    lib = ctypes.CDLL(path)
    lib.get_legal_plays.argtypes = [
        ctypes.POINTER(bf._CBoardState), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(bf._CPlay), ctypes.c_int]
    lib.get_legal_plays.restype = ctypes.c_int
    return lib


class _Gen:
    def __init__(self, lib):
        self.lib = lib
        self.buf = (bf._CPlay * bf.MAX_PLAYS)()
        self.arr = np.frombuffer(self.buf, dtype=np.int32).reshape(bf.MAX_PLAYS, PLAY_INTS)

    def plays(self, c, d1, d2, cap=bf.MAX_PLAYS):
        """(n, 38) copy of the ordered plays, stale move slots zeroed."""
        n = self.lib.get_legal_plays(ctypes.byref(c), d1, d2, self.buf, cap)
        a = self.arr[:n].copy()
        nm = a[:, 8]
        for k in range(4):
            a[nm <= k, 2 * k:2 * k + 2] = 0
        return a


def _trajectories(gen, games=60, seed=2026):
    """Positions from seeded self-play (uniformly random legal plays)."""
    rng = random.Random(seed)
    out = []
    for _ in range(games):
        c = bf._CBoardState()
        c.points[0], c.points[5], c.points[7], c.points[11] = -2, 5, 3, -5
        c.points[12], c.points[16], c.points[18], c.points[23] = 5, -3, -5, 2
        c.turn = rng.randint(0, 1)
        while c.off[0] < 15 and c.off[1] < 15:
            out.append(bf._CBoardState.from_buffer_copy(c))
            a = gen.plays(c, rng.randint(1, 6), rng.randint(1, 6))
            if len(a):
                row = a[rng.randrange(len(a))]
                c = bf._CBoardState.from_buffer_copy(row[9:].astype(np.int32).tobytes())
            else:
                c.turn = 1 - c.turn
    return out


def _digest(gen, positions):
    h = hashlib.sha256()
    calls = passes = 0
    for c in positions:
        for d1, d2 in ROLLS:
            a = gen.plays(c, d1, d2)
            h.update(np.int32(len(a)).tobytes())
            h.update(a.tobytes())
            calls += 1
            passes += len(a) == 0
    for c in positions[::7]:
        for cap in CAPS:
            for d1, d2 in ROLLS:
                a = gen.plays(c, d1, d2, cap)
                h.update(np.int32(len(a)).tobytes())
                h.update(a.tobytes())
                calls += 1
    return h.hexdigest(), calls, passes


@pytest.fixture(scope="module")
def positions():
    return _trajectories(_Gen(_lib()))


def test_frozen_sequence_digest(positions):
    sha, calls, passes = _digest(_Gen(_lib()), positions)
    assert (calls, passes) == (FROZEN_CALLS, FROZEN_PASSES)
    assert sha == FROZEN_SHA, f"get_legal_plays output changed: {sha}"


def test_concurrent_threads_identical(positions):
    lib = _lib()
    sub = positions[::3]
    want = _digest(_Gen(lib), sub)
    got = [None] * 4

    def run(i):
        got[i] = _digest(_Gen(lib), sub)

    ts = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert got == [want] * 4


# Wrapping the generation to 0 must clear every slot: output stays the
# frozen digest across the wrap (gen starts 16 calls below 2^32).
def test_generation_wrap_digest(tmp_path):
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc unavailable")
    so = tmp_path / "libbg_hooks.so"
    subprocess.run([gcc, "-O2", "-shared", "-fPIC", "-DBG_TEST_HOOKS",
                    "-o", str(so), os.path.join(REPO, "c_engine", "bg_engine.c")],
                   check=True)
    lib = _lib(str(so))
    lib.bg_test_set_dedup_gen.argtypes = [ctypes.c_uint]
    lib.bg_test_set_dedup_gen(0xFFFFFFF0)
    positions = _trajectories(_Gen(lib))
    lib.bg_test_set_dedup_gen(0xFFFFFFF0)
    assert _digest(_Gen(lib), positions) == (FROZEN_SHA, FROZEN_CALLS, FROZEN_PASSES)
