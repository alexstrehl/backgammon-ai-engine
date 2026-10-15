"""Fixed-pool gnubg benchmark: OFFICIAL binary, embedded Python.

Run INSIDE a threads-disabled gnubg 1.08.003 build (built with gnubg_bench/build.sh):

    GNUBG_FP_PLIES=0,1 \
      $HOME/gnubg-bench/bin/gnubg -t -q -p third_party_opponents/gnubg/fixedpool_bench_gnubg.py

Times gnubg's own C code at DEFAULT settings (move generation + filters +
prune nets + eval) over the committed position pools, the same ones
fixedpool_bench.py replays. gnubg's engine is only reachable from the
binary's own embedded Python, which can't import torch/numpy/bgsage, or even
`random` (hence _LCG). The official binary is required: gnubg's Python
binding understates gnubg (leaf pruning disabled).

A gnubg built with gnubg_bench/build.sh has gnubg.benchpool, which times
each item in C around the engine calls behind findbestmove/cfevaluate.
Other builds time each findbestmove/cfevaluate call from Python, call
overhead included. The output's "# gnubg timing:" line says which path ran.

Three per-decision costs, selected by GNUBG_FP_MODE (cubeless|cubeful|both,
default both):
  cubeless checker : findbestmove at a CUBELESS evalcontext over
                     fixedpool_positions.json.
  cubeful  checker : findbestmove at a CUBEFUL evalcontext over the checker
                     points of fixedpool_cubeful_positions.json, each with
                     its own cube state.
  cube             : cfevaluate -> FindCubeDecision, the function gnubg's
                     native money player calls, over the cubeful pool's cube
                     points with each point's own cubeinfo (cube value +
                     owner; jacoby inherited from the session default = on).

Env knobs: GNUBG_FP_MODE (default "both"), GNUBG_POOL / GNUBG_CUBEFUL_POOL
(override pool paths; default = the committed experiments/benchmarks/
pools, relative to the cwd: run from the repo root),
GNUBG_FP_PLIES (default "0,1"), BENCH_CORE / BENCH_NO_PIN (pin; default: core 8 if present,
else the highest available), BENCH_WARMUP_S (clock-ramp secs, default 3).
GNUBG_FP_CAP times a seeded uniform random sample, the same items as
fixedpool_bench.py --cap.
"""
import json
import os
import time

import gnubg


class _LCG:
    """Math-free deterministic PRNG for bootstrap resampling. gnubg's
    embedded Python can't import `random` (it pulls in `math`, which this
    interpreter lacks). Default seed 1, as random.Random(1); the stream
    differs."""

    def __init__(self, seed=1):
        self.s = (seed * 2862933555777941757 + 3037000493) & 0xFFFFFFFFFFFFFFFF

    def _n(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.s >> 11

    def randrange(self, n):
        return self._n() % n

if not os.environ.get("BENCH_NO_PIN"):
    _env = os.environ.get("BENCH_CORE", "").strip()
    if _env:
        # An explicitly requested core must be honored; otherwise exit.
        _core = int(_env)
        if _core == 0:
            raise SystemExit("BENCH_CORE=0 is the IRQ core; pick another core")
        try:
            os.sched_setaffinity(0, {_core})
        except AttributeError:
            raise SystemExit("sched_setaffinity unavailable on this platform; "
                             "set BENCH_NO_PIN=1 to run unpinned")
        except OSError as e:
            raise SystemExit("cannot pin to core %d (%s); pick a core this host has"
                             % (_core, e))
    else:
        # No core requested: prefer 8 (core 0 handles IRQs), else the highest
        # core this process may use.
        try:
            _avail = os.sched_getaffinity(0)
            _core = 8 if 8 in _avail else max(_avail)
            os.sched_setaffinity(0, {_core})
            if _core != 8:
                print("[bench] BENCH_CORE unset; pinned to core %d" % _core)
        except (AttributeError, OSError):
            print("[bench] cannot pin on this platform; running unpinned")

# Runs inside gnubg's embedded Python (`gnubg -t -q -p`), where `__file__`
# is undefined and gzip/zlib may fail to import. Locate the pool by env or
# cwd-relative path and read plain JSON.
POOL = os.environ.get("GNUBG_POOL") or os.path.join(
    "experiments", "benchmarks", "fixedpool_positions.json")
CUBEFUL_POOL = os.environ.get("GNUBG_CUBEFUL_POOL") or os.path.join(
    "experiments", "benchmarks", "fixedpool_cubeful_positions.json")
MODE = os.environ.get("GNUBG_FP_MODE", "both")   # cubeless | cubeful | both
PLIES = [int(x) for x in os.environ.get("GNUBG_FP_PLIES", "0,1").split(",")]
# GNUBG_FP_CAP caps every ply (0 = whole pool); smoke tests only.
CAP = int(os.environ.get("GNUBG_FP_CAP", "0"))
# Warm the core to boost clock before timing (see bench_common); default 3s.
WARMUP_S = float(os.environ.get("BENCH_WARMUP_S", "3.0"))


NATIVE = hasattr(gnubg, "benchpool")
print("# gnubg timing: native benchpool" if NATIVE
      else "# gnubg timing: per-call Python (findbestmove/cfevaluate)", flush=True)


def rate_stats(dts):
    """(total_dt, rate, lo, hi) over per-item seconds, with a
    percentile-bootstrap 95% CI resampling ITEMS, the same estimator as
    bench_common.rate(). Point estimate AND bootstrap statistic are the
    RATIO OF SUMS (total items / total wall-time), not the mean of per-item
    rates."""
    dt_total = sum(dts)
    n = len(dts)
    point = n / dt_total
    rng = _LCG(1)
    bm = []
    for _ in range(1000):
        sd = 0.0
        for _ in range(n):
            sd += dts[rng.randrange(n)]
        bm.append(n / sd)
    bm.sort()
    return dt_total, point, bm[25], bm[975]


_M64 = 0xFFFFFFFFFFFFFFFF


def sample_indices(n, k, seed=1):
    """bench_common.sample_indices (inline copy): k of range(n) drawn
    uniformly without replacement, sorted."""
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


def timed_row(kind, items, ec, m):
    """Ramp the clock, run 20 untimed items, then time m items: the pool, or
    a uniform random sample of m items in pool order. kind is "move" (items =
    (board, cubeinfo, dice)) or "cube" (items = (board, cubeinfo)). The
    per-call path includes the Python call overhead."""
    if m < len(items):
        items = [items[i] for i in sample_indices(len(items), m)]
    _spin_ramp(WARMUP_S)
    if NATIVE:
        gnubg.benchpool(kind, items[:min(m, 20)], ec)
        dts, _ = gnubg.benchpool(kind, items[:m], ec)
        return rate_stats(dts)
    if kind == "move":
        call = lambda it: gnubg.findbestmove(it[0], it[1], ec, it[2])
    else:
        call = lambda it: gnubg.cfevaluate(it[0], it[1], ec)
    for it in items[:min(m, 20)]:
        call(it)
    dts = []
    for it in items[:m]:
        t0 = time.perf_counter()
        call(it)
        dts.append(time.perf_counter() - t0)
    return rate_stats(dts)


def _spin_ramp(seconds):
    """Ramp the core clock cache-neutrally: a scalar busy loop that does NOT
    pre-load the timed board pool (that would leave gnubg's 0-ply eval
    all-cache-hot)."""
    if seconds <= 0:
        return
    t0 = time.perf_counter()
    x = 1.0
    while time.perf_counter() - t0 < seconds:
        for _ in range(20000):
            x = x * 1.0000001 + 1.0

data = json.load(open(POOL))
recs = data["positions"]
boards = [(tuple(r["board"][0]), tuple(r["board"][1])) for r in recs]
dice = [tuple(r["dice"]) for r in recs]
n = len(recs)
print("pool: %d positions from %s (mode=%s)" % (n, POOL, MODE), flush=True)

ci = gnubg.cubeinfo(1, -1, 0, 0, (0, 0), 0, 0)   # money, centered cube
cl_items = [(boards[i], ci, dice[i]) for i in range(n)]

# CUBELESS checker: findbestmove at a cubeless evalcontext (arg0 = 0).
if MODE in ("cubeless", "both"):
    for plies in PLIES:
        ec = gnubg.evalcontext(0, plies, 0, 1, 0.0)   # cubeless, default prune ON
        m = n
        if CAP: m = min(m, CAP)
        dt, r, lo, hi = timed_row("move", cl_items, ec, m)
        print("FIXEDPOOL findbestmove %d-ply (default filters): %d pos %.2fs"
              " | %.1f moves/s 95%%CI [%.1f,%.1f] | %.3f ms/move"
              % (plies, m, dt, r, lo, hi, 1000 / r), flush=True)

# CUBEFUL: the shared committed cubeful pool, with the positions, dice and cube
# states that PureTD/Sage replay in fixedpool_bench.py --mode cubeful.
if MODE in ("cubeful", "both"):
    cf = json.load(open(CUBEFUL_POOL))

    def _cinfos(recs):
        """Per-point cubeinfo from the stored absolute cube state.
        cubeinfo(cube, owner, move=0, matchto=0, score, crawford=0, bgv=0);
        owner: -1 centered, 0 = mover, 1 = opponent (cube_owner is absolute
        0/1/2 = centered/WHITE/BLACK, turn is the mover's color). jacoby is
        inherited from the session default (on), matching the pool's games."""
        out = []
        for r in recs:
            co, t = r["cube_owner"], r["turn"]
            rel = -1 if co == 0 else (0 if co == t + 1 else 1)
            out.append(gnubg.cubeinfo(r["cube_value"], rel, 0, 0, (0, 0), 0, 0))
        return out

    ck = [((tuple(r["board"][0]), tuple(r["board"][1])), c, tuple(r["dice"]))
          for r, c in zip(cf["checker"], _cinfos(cf["checker"]))]
    cu = [((tuple(r["board"][0]), tuple(r["board"][1])), c)
          for r, c in zip(cf["cube"], _cinfos(cf["cube"]))]
    print("cubeful pool: %d checker + %d cube points from %s"
          % (len(ck), len(cu), CUBEFUL_POOL), flush=True)

    # checker: findbestmove at a cubeful evalcontext with the point's cube state
    for plies in PLIES:
        ec = gnubg.evalcontext(1, plies, 0, 1, 0.0)   # cubeful, default prune ON
        m = len(ck)
        if CAP: m = min(m, CAP)
        dt, r, lo, hi = timed_row("move", ck, ec, m)
        print("FIXEDPOOL cubeful findbestmove %d-ply (default filters): %d pos %.2fs"
              " | %.1f moves/s 95%%CI [%.1f,%.1f] | %.3f ms/move"
              % (plies, m, dt, r, lo, hi, 1000 / r), flush=True)

    # cube: cfevaluate -> FindCubeDecision with the point's cube state
    for plies in PLIES:
        ec = gnubg.evalcontext(1, plies, 0, 1, 0.0)   # cubeful eval
        m = len(cu)
        if CAP: m = min(m, CAP)
        dt, r, lo, hi = timed_row("cube", cu, ec, m)
        print("FIXEDPOOL cube cfevaluate %d-ply: %d pos %.2fs | %.1f evals/s"
              " 95%%CI [%.1f,%.1f] | %.4f ms/eval"
              % (plies, m, dt, r, lo, hi, 1000 / r), flush=True)

print("FIXEDPOOL GNUBG ALLDONE", flush=True)
