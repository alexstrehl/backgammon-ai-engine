"""Fixed-pool gnubg benchmark — OFFICIAL binary, embedded Python.

Run INSIDE a threads-disabled gnubg 1.08.003 build (see README recipe):

    GNUBG_FP_PLIES=0,1 \
      $HOME/gnubg-st/bin/gnubg -t -q -p experiments/benchmarks/fixedpool_bench_gnubg.py

Times gnubg's own C code at DEFAULT settings (move generation + filters +
prune nets + eval) over the shared committed position pool
(fixedpool_positions.json, same pool as fixedpool_bench.py). This is the ONE
gnubg fixed-pool file — it must be separate from the native fixedpool_bench.py
because gnubg's engine is only reachable from the binary's own embedded Python
(which can't import torch/numpy/bgsage, or even `random` — hence the inline
math-free _LCG). The official binary is required: gnubg's Python binding
understates gnubg (leaf pruning disabled).

Three per-decision costs, selected by GNUBG_FP_MODE (cubeless|cubeful|both,
default both):
  cubeless checker : findbestmove at a CUBELESS evalcontext over the shared
                     8000-board pool (fixedpool_positions.json) — feeds the
                     cubeless fixed-pool table.
  cubeful  checker : findbestmove at a CUBEFUL evalcontext over the SHARED
                     CUBEFUL pool (fixedpool_cubeful_positions.json) — the
                     same checker points, dice and per-position cube states
                     (~49% recubes) that PureTD/Sage replay in
                     fixedpool_bench.py --mode cubeful.
  cube             : cfevaluate -> FindCubeDecision, the SAME function gnubg's
                     native money player calls (the faithful cube-action cost,
                     not a proxy), over the shared cubeful pool's cube points
                     with each point's own cubeinfo (cube value + owner;
                     jacoby inherited from the session default = on).

The cubeful sections replay the committed cubeful pool so ALL engines time
identical positions AND identical cube states (per-position cube states
matter: centered cfevaluate measures ~6-21% costlier than owned-cube at 1-ply,
so a constant-centered convention would overstate gnubg's cube cost).

Env knobs: GNUBG_FP_MODE (default "both"), GNUBG_POOL / GNUBG_CUBEFUL_POOL
(override pool paths; default = committed artifacts next to this script),
GNUBG_FP_PLIES (default "0,1" — 2-ply is not part of the suite, not used
in the paper), GNUBG_FP_CAP2 (position cap for plies >= 2,
default 4000), BENCH_CORE /
BENCH_NO_PIN (pin; default: core 8 if present, else the highest available),
BENCH_WARMUP_S (clock-ramp secs, default 3).
"""
import json
import os
import time

import gnubg


class _LCG:
    """Math-free deterministic PRNG for bootstrap resampling. gnubg's
    embedded Python 3.14 can't import `random` (it pulls in `math`, a C
    extension absent from this interpreter), so use a tiny inline LCG.
    Seeded like random.Random(1): reproducible across runs."""

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
        # An explicitly requested core must be honored — fail loudly.
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

# NOTE: run inside gnubg's embedded Python (`gnubg -t -q -p`), where
# `__file__` is undefined and compiled-extension imports (gzip/zlib) can
# mismatch gnubg's libpython. So: locate the pool by env or cwd-relative
# path (no `__file__`), and read plain JSON (json falls back to pure
# Python; the committed pool is uncompressed for this reason).
POOL = os.environ.get("GNUBG_POOL") or os.path.join(
    "experiments", "benchmarks", "fixedpool_positions.json")
CUBEFUL_POOL = os.environ.get("GNUBG_CUBEFUL_POOL") or os.path.join(
    "experiments", "benchmarks", "fixedpool_cubeful_positions.json")
MODE = os.environ.get("GNUBG_FP_MODE", "both")   # cubeless | cubeful | both
PLIES = [int(x) for x in os.environ.get("GNUBG_FP_PLIES", "0,1").split(",")]
CAP2 = int(os.environ.get("GNUBG_FP_CAP2", "4000"))
# GNUBG_FP_CAP caps EVERY ply (0 = whole pool). Exists so the suite can be
# smoke-tested cheaply before a multi-hour run; leave unset for real runs.
CAP = int(os.environ.get("GNUBG_FP_CAP", "0"))
# Warm the core to boost clock before timing (see bench_common); default 3s.
WARMUP_S = float(os.environ.get("BENCH_WARMUP_S", "3.0"))


def timed_rate(call, m):
    """Time each call(i) individually; return (total_dt, rate, lo, hi) with a
    percentile-bootstrap 95% CI resampling CALLS — the same estimator as
    bench_common.rate() on the engines side. Point estimate AND bootstrap
    statistic are the RATIO OF SUMS (total calls / total wall-time), not the
    mean of per-call rates, which is Jensen-biased upward by ~CV^2 of the
    times on variable-cost workloads.

    Per-call timing costs ~113 ns. That is <0.1% on most rows but ~4% on the
    0-ply cube primitive (~2.8 us/call), whose cost is a slight over-estimate."""
    dts = []
    for i in range(m):
        t0 = time.perf_counter()
        call(i)
        dts.append(time.perf_counter() - t0)
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


def _spin_ramp(seconds):
    """Ramp the core clock cache-neutrally: a scalar busy loop that does NOT
    pre-load the timed board pool (looping the pool would leave gnubg's tiny
    0-ply eval all-cache-hot, inflating it 2x+ above its realistic cost)."""
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

# CUBELESS checker: findbestmove at a cubeless evalcontext (arg0 = 0).
if MODE in ("cubeless", "both"):
    for plies in PLIES:
        ec = gnubg.evalcontext(0, plies, 0, 1, 0.0)   # cubeless, default prune ON
        m = min(n, CAP2) if plies >= 2 else n
        if CAP: m = min(m, CAP)
        _spin_ramp(WARMUP_S)
        for i in range(min(m, 20)):
            gnubg.findbestmove(boards[i], ci, ec, dice[i])
        dt, r, lo, hi = timed_rate(
            lambda i: gnubg.findbestmove(boards[i], ci, ec, dice[i]), m)
        print("FIXEDPOOL findbestmove %d-ply (default filters): %d pos %.2fs"
              " | %.1f moves/s 95%%CI [%.1f,%.1f] | %.3f ms/move"
              % (plies, m, dt, r, lo, hi, 1000 / r), flush=True)

# CUBEFUL: the shared committed cubeful pool — same positions, dice and cube
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

    ck = [((tuple(r["board"][0]), tuple(r["board"][1])), tuple(r["dice"]))
          for r in cf["checker"]]
    ck_ci = _cinfos(cf["checker"])
    cu = [(tuple(r["board"][0]), tuple(r["board"][1])) for r in cf["cube"]]
    cu_ci = _cinfos(cf["cube"])
    print("cubeful pool: %d checker + %d cube points from %s"
          % (len(ck), len(cu), CUBEFUL_POOL), flush=True)

    # checker: findbestmove at a cubeful evalcontext with the point's cube state
    for plies in PLIES:
        ec = gnubg.evalcontext(1, plies, 0, 1, 0.0)   # cubeful, default prune ON
        m = min(len(ck), CAP2) if plies >= 2 else len(ck)
        if CAP: m = min(m, CAP)
        _spin_ramp(WARMUP_S)
        for i in range(min(m, 20)):
            gnubg.findbestmove(ck[i][0], ck_ci[i], ec, ck[i][1])
        dt, r, lo, hi = timed_rate(
            lambda i: gnubg.findbestmove(ck[i][0], ck_ci[i], ec, ck[i][1]), m)
        print("FIXEDPOOL cubeful findbestmove %d-ply (default filters): %d pos %.2fs"
              " | %.1f moves/s 95%%CI [%.1f,%.1f] | %.3f ms/move"
              % (plies, m, dt, r, lo, hi, 1000 / r), flush=True)

    # cube: cfevaluate -> FindCubeDecision with the point's cube state
    for plies in PLIES:
        ec = gnubg.evalcontext(1, plies, 0, 1, 0.0)   # cubeful eval
        m = min(len(cu), CAP2) if plies >= 2 else len(cu)
        if CAP: m = min(m, CAP)
        _spin_ramp(WARMUP_S)
        for i in range(min(m, 20)):
            gnubg.cfevaluate(cu[i], cu_ci[i], ec)
        dt, r, lo, hi = timed_rate(lambda i: gnubg.cfevaluate(cu[i], cu_ci[i], ec), m)
        print("FIXEDPOOL cube cfevaluate %d-ply: %d pos %.2fs | %.1f evals/s"
              " 95%%CI [%.1f,%.1f] | %.4f ms/eval"
              % (plies, m, dt, r, lo, hi, 1000 / r), flush=True)

print("FIXEDPOOL GNUBG ALLDONE", flush=True)
