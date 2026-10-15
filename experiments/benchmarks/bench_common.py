"""Shared helpers for the single-thread compute benchmarks.

Fixed-pool loading, Open Sage board conversion, and the bootstrap moves/s
timer used by the fixed-pool benchmarks. Import side effects: sets
OMP/MKL/torch to 1 thread and pins the process to one core (BENCH_NO_PIN=1 /
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
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "c_engine"))

import torch
torch.set_grad_enabled(False)
torch.set_num_threads(1)
from backgammon_engine import BoardState, WHITE, BLACK

# Committed uncompressed .json (the gnubg embedded-python script can't
# rely on gzip).
FIXEDPOOL = os.path.join(_HERE, "fixedpool_positions.json")


def load_fixedpool(cap=None):
    """Load the committed benchmark pool -> list of (BoardState, dice).

    The pool is a frozen artifact. Provenance is in its meta block and the
    README.
    """
    import json
    with open(FIXEDPOOL) as f:
        art = json.load(f)
    recs = art["positions"][:cap] if cap else art["positions"]
    return [(BoardState(points=r["points"], bar=r["bar"], off=r["off"],
                        turn=r["turn"]), tuple(r["dice"])) for r in recs]


def sage_conv(points, bar, mover):
    """BoardState -> Open Sage 26-int board, from the mover's perspective."""
    if mover == WHITE:
        body = list(points)
        mbar, obar = bar[WHITE], bar[BLACK]
    else:
        body = [-points[23 - i] for i in range(24)]
        mbar, obar = bar[BLACK], bar[WHITE]
    return [obar] + body + [mbar]


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


def _warmup(fn, items, seconds=WARMUP_S):
    if not items:
        return
    _spin_ramp(seconds)          # clock ramp only (cache-neutral)
    for it in items[:20]:        # warm weights/code paths, not the whole pool
        fn(it)


_M64 = 0xFFFFFFFFFFFFFFFF


def sample_indices(n, k, seed=1):
    """k of range(n) drawn uniformly without replacement, sorted: a partial
    Fisher-Yates shuffle over a splitmix64 stream from `seed`.
    fixedpool_bench_gnubg.py has an inline copy (gnubg's Python cannot import
    this module)."""
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


def rate(fn, items, cap):
    """calls/s and percentile-bootstrap 95% CI, resampling ITEMS.

    Point estimate AND bootstrap statistic are the RATIO OF SUMS
    (total calls / total wall-time), not the mean of per-item rates,
    which is biased upward when per-call cost varies.

    Items are resampled individually. Per-call perf_counter overhead is
    included."""
    items = items[:cap]
    _warmup(fn, items)  # ramp core clock to steady state before timing
    dts = []
    for it in items:
        t = time.perf_counter()
        fn(it)
        dts.append(time.perf_counter() - t)
    return rate_from_dts(dts)


def rate_from_dts(dts):
    """`rate`'s estimator on given per-item seconds: (point, lo, hi) of
    calls/s, the ratio of sums with an item-resampling percentile
    bootstrap 95% CI."""
    n = len(dts)
    point = n / sum(dts)
    rng = random.Random(1)
    bm = []
    for _ in range(1000):
        sd = 0.0
        for _ in range(n):
            sd += dts[rng.randrange(n)]
        bm.append(n / sd)
    bm.sort()
    return point, bm[25], bm[975]


def c_engine(model, backend):
    """A C API Engine (c_api/puretd_py) on `backend`, after printing the
    backend header line of the C rows."""
    from c_api import puretd_py as P
    eng = P.Engine(model, backend)
    print(f"# C API backend {eng.backend}: {eng.describe}", flush=True)
    return eng


def model_stamp(path) -> str:
    """'<basename> md5 <8hex>' for a model path, for bench output headers.
    The rate depends on which model was the workload."""
    import hashlib
    import os
    if not path or not os.path.isfile(path):
        return f"{path} (MISSING)"
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{os.path.basename(path)} md5 {h.hexdigest()[:8]}"


# HedgeHog .ogxf weights are not tracked: $HH_FOX / $HH_AUREUS name the
# files, default third_party_opponents/hedgehog/<published filename>.
_HH_FILES = {"fox": "fox_v0.34_2026-09-26.ogxf",
             "aureus": "aureus_v0.12_2026-09-26.ogxf"}


def hh_model_path(name):
    import os
    hh_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "..", "third_party_opponents", "hedgehog")
    return (os.environ.get("HH_" + name.upper())
            or os.path.join(hh_dir, _HH_FILES[name]))
