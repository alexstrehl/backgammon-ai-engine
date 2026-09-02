"""Shared helpers for the single-thread compute benchmarks.

Fixed-pool loading, Open Sage board conversion, and the bootstrap
moves/s timer used by the fixed-pool benchmarks. Import side effects: sets OMP/MKL/torch to 1 thread and
pins the process to one core (BENCH_NO_PIN=1 / BENCH_CORE=N to override).
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
        # An explicitly requested core must be honored — silently running on a
        # different core makes cross-engine/cross-sweep numbers incomparable.
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

    The pool is a frozen artifact (provenance in its meta block and the
    README): 8000 checker-decision points from seeded gnubg 0-ply
    cubeless-money self-play, recorded once so running the benchmarks
    needs no position-generating engine.
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


# Timing is dominated, for the tiny-eval configs (e.g. prob5-0ply), by
# per-call overhead that scales with the core's clock. A cold core sits at
# base clock and ramps to boost over ~1-3 s of sustained load, so a few-call
# warmup reads ~20-25% low if it runs first. Warm the core for a fixed
# WALL-TIME (BENCH_WARMUP_S, default 3s) before every timed config.
WARMUP_S = float(os.environ.get("BENCH_WARMUP_S", "3.0"))


def _spin_ramp(seconds):
    """Ramp the pinned core from idle base clock to boost, WITHOUT touching
    the engine or the timed board pool. A cache-neutral scalar busy loop:
    it must not pre-load the exact positions being timed, or tiny-eval
    configs (gnubg-0ply, Sage's position cache) read all-cache-hot and
    inflate 2-7x above their realistic (self-play) cost."""
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


def rate(fn, items, cap):
    """calls/s and percentile-bootstrap 95% CI, resampling ITEMS.

    Point estimate AND bootstrap statistic are the RATIO OF SUMS
    (total calls / total wall-time), not the mean of per-item rates:
    with variable per-call cost (e.g. Sage's analyzer position cache)
    the arithmetic mean is Jensen-biased upward by ~CV^2 of the times —
    measured +40% on the cubeful Sage checker row. Ratio-of-sums is the
    faithful throughput and matches a plain total/dt timing exactly.

    Items are resampled individually. Per-call perf_counter costs ~113 ns,
    under 0.1% except on sub-10us rows."""
    items = items[:cap]
    _warmup(fn, items)  # ramp core clock to steady state before timing
    dts = []
    for it in items:
        t = time.perf_counter()
        fn(it)
        dts.append(time.perf_counter() - t)
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


def ratio_ci(dts, counts, B=10000, seed=1):
    """Game-level percentile bootstrap 95% CI of sum(counts)/sum(dts).

    The self-play combined rate is a ratio of two sums over games (total
    decisions / total wall-time), so the natural resampling unit is the
    game: resample games with replacement B times and take the 2.5/97.5
    percentiles of the resampled ratio. Identical estimator to
    selfplay_bench_gnubg.ratio_ci (that script inlines its own copy because
    gnubg's embedded Python can't import this module)."""
    n = len(dts)
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        sd = sc = 0.0
        for _ in range(n):
            j = rng.randrange(n)
            sd += dts[j]
            sc += counts[j]
        out.append(sc / sd)
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]


def model_stamp(path) -> str:
    """'<basename> md5 <8hex>' for a model path, for bench output headers.

    Bench rows record a MACHINE, but the number depends on which model was
    the workload: prob5 tail-256 is 2-6% slower than tail-128 on identical
    hardware, large enough to be misread as a hardware or code regression
    across eras. Stamping the md5 makes that impossible.
    """
    import hashlib
    import os
    if not path or not os.path.isfile(path):
        return f"{path} (MISSING)"
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{os.path.basename(path)} md5 {h.hexdigest()[:8]}"
