"""Shared helpers for the single-thread compute benchmarks and the tests
that read the committed position pools.

Fixed-pool loading, the seeded pool sampler, and the clock ramp used
before timed runs. Import side effects: sets OMP/MKL/torch to 1 thread,
turns autograd off, and pins the process to one core (BENCH_NO_PIN=1 /
BENCH_CORE=N to override).
"""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Pin to a single core (equivalent to `taskset -c 8`) so timing is not perturbed
# by other load. Override: BENCH_CORE=N / BENCH_NO_PIN=1.
if not os.environ.get("BENCH_NO_PIN"):
    _env = os.environ.get("BENCH_CORE", "").strip()
    if _env:
        # An explicitly requested core must be honored.
        _core = int(_env)
        if _core == 0:
            raise SystemExit("BENCH_CORE=0 is the IRQ core; pick another or set BENCH_NO_PIN=1")
        try:
            os.sched_setaffinity(0, {_core})
        except AttributeError:
            raise SystemExit("sched_setaffinity unavailable on this platform; "
                             "set BENCH_NO_PIN=1 to run unpinned (results are NOT comparable)")
        except OSError as e:
            raise SystemExit("cannot pin to core %d (%s). Pick a core this host/cgroup "
                             "actually has, or set BENCH_NO_PIN=1 (results are NOT "
                             "comparable)." % (_core, e))
    else:
        # No core requested: prefer 8 (core 0 handles IRQs), else the highest
        # core this process may use (small hosts, cgroup-limited containers).
        try:
            _avail = os.sched_getaffinity(0)
            _core = 8 if 8 in _avail else max(_avail)
            os.sched_setaffinity(0, {_core})
            if _core != 8:
                print("[bench] BENCH_CORE unset; pinned to core %d" % _core,
                      file=sys.stderr)
        except (AttributeError, OSError):
            print("[bench] cannot pin on this platform; running unpinned "
                  "(results are NOT comparable across engines/hosts)",
                  file=sys.stderr)
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "c_engine"))

import torch
torch.set_grad_enabled(False)
torch.set_num_threads(1)
from backgammon_engine import BoardState

# Committed uncompressed .json (gnubg's embedded Python can't rely on gzip).
FIXEDPOOL = os.path.join(_HERE, "fixedpool_positions.json")


def load_fixedpool(cap=None):
    """Load the committed benchmark pool -> list of (BoardState, dice).

    The pool is a frozen artifact. Provenance is in its meta block.
    """
    import json
    with open(FIXEDPOOL) as f:
        art = json.load(f)
    recs = art["positions"][:cap] if cap else art["positions"]
    return [(BoardState(points=r["points"], bar=r["bar"], off=r["off"],
                        turn=r["turn"]), tuple(r["dice"])) for r in recs]


# Warm the core for a fixed wall-time before every timed config, because a
# cold core reads low. Override: BENCH_WARMUP_S.
WARMUP_S = float(os.environ.get("BENCH_WARMUP_S", "3.0"))


def _spin_ramp(seconds):
    """Ramp the pinned core to boost clock with a cache-neutral scalar busy
    loop. It must not touch the engine or the timed positions, or tiny-eval
    configs read all-cache-hot."""
    if seconds <= 0:
        return
    t0 = time.perf_counter()
    x = 1.0
    while time.perf_counter() - t0 < seconds:
        for _ in range(20000):
            x = x * 1.0000001 + 1.0
    return x


_M64 = 0xFFFFFFFFFFFFFFFF


def sample_indices(n, k, seed=1):
    """k of range(n) drawn uniformly without replacement, sorted: a partial
    Fisher-Yates shuffle over a splitmix64 stream from `seed`. A gnubg
    benchmark script (gnubg's Python cannot import this module) keeps an
    inline copy so that every engine draws the same items."""
    idx = list(range(n))
    s = seed & _M64
    for i in range(k):
        s = (s + 0x9E3779B97F4A7C15) & _M64
        z = ((s ^ (s >> 30)) * 0xBF58476D1CE4E5B9) & _M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
        z ^= z >> 31
        j = i + z % (n - i)
        idx[i], idx[j] = idx[j], idx[i]
    return sorted(idx[:k])


def sample(items, cap, seed=1):
    """The pool when cap is 0 or covers it, else `cap` items drawn uniformly
    at random (sample_indices), in pool order: the same items for every
    engine."""
    n = len(items)
    if not cap or cap >= n:
        return items
    return [items[i] for i in sample_indices(n, cap, seed)]
