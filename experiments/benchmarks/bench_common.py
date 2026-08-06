"""Shared helpers for the single-thread compute benchmarks.

Fixed-pool loading, Open Sage board conversion, and the block-bootstrap
moves/s timer used by the fixed-pool benchmarks. Import side effects: sets OMP/MKL/torch to 1 thread and
pins the process to one core (BENCH_NO_PIN=1 / BENCH_CORE=N to override).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Pin to a single core (equivalent to `taskset -c 0`) so timing is not perturbed
# by other load. Override: BENCH_NO_PIN=1.
if not os.environ.get("BENCH_NO_PIN"):
    try:
        os.sched_setaffinity(0, {int(os.environ.get("BENCH_CORE", "0"))})
    except (AttributeError, OSError):
        pass
import sys
import time
import random
import statistics as st

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


def rate(fn, items, cap, blocks):
    """moves/s and block-bootstrap 95% CI over `blocks` equal blocks.

    Point estimate AND bootstrap statistic are the RATIO OF SUMS
    (total calls / total wall-time), not the mean of per-block rates:
    with variable per-call cost (e.g. Sage's analyzer position cache)
    the arithmetic mean of block rates is Jensen-biased upward by
    ~CV^2 of the block times — measured +40% on the cubeful Sage
    checker row. Ratio-of-sums is the faithful throughput and matches
    a plain total/dt timing exactly."""
    items = items[:cap]
    _warmup(fn, items)  # ramp core clock to steady state before timing
    n = len(items)
    bs = max(1, n // blocks)
    cnts, dts = [], []
    for i in range(0, n - bs + 1, bs):
        blk = items[i:i + bs]
        t = time.perf_counter()
        for it in blk:
            fn(it)
        dts.append(time.perf_counter() - t)
        cnts.append(len(blk))
    done = (n // bs) * bs
    ntail = n - done          # leftover tail: timed + counted, not a block
    if ntail:
        blk = items[done:]
        t = time.perf_counter()
        for it in blk:
            fn(it)
        tail_dt = time.perf_counter() - t
    point = (sum(cnts) + ntail) / (sum(dts) + (tail_dt if ntail else 0.0))
    nb = len(dts)
    rng = random.Random(1)
    bm = []
    for _ in range(1000):
        sc = sd = 0.0
        for _ in range(nb):
            j = rng.randrange(nb)
            sc += cnts[j]
            sd += dts[j]
        bm.append(sc / sd)
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
