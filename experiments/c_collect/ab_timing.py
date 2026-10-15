"""Paired single-core A/B timing of Engine.collect for two C API builds.

    taskset -c $CORE python experiments/c_collect/ab_timing.py \
        --a DIR_A --b DIR_B [--cells cubeful0,cubeful1] [--rounds 4] [--backend simd]

Each arm loads libpuretd_<backend>.so from its directory (PURETD_LIBDIR; a
puretd_py.py there replaces c_api's); --env-a / --env-b set environment
variables for one arm only, so --a and --b may be the same build. Each round
runs one subprocess per arm, alternating the order (A B, then B A, ...).
Per cell a subprocess makes one warm-up call, then `seeds` timed calls on
seeds 1000.. (0-ply cells: --episodes0 episodes per call, 1-ply cells:
--episodes1). A seed does the same work in both arms unless the change moves
a near-tie decision, so the per-seed time ratio is paired. Prints per cell:
ep/s of each arm (its calls pooled), the median per-seed speedup B/A with
min..max over seeds x rounds, rows per episode, and the 1-minute load before
each subprocess. Runs on the parent's CPU affinity.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = {"dmp": "dmp_512_512_256_256.pt",
          "cubeless": "cubeless_money_512_512_256_256.pt",
          "cubeful": "cubeful_money_512_512_256_256.pt"}
# name -> (model, oneply, jacoby)
CELLS = {"cubeful0": ("cubeful", 0, 1), "cubeful1": ("cubeful", 1, 1),
         "cubeful0j0": ("cubeful", 0, 0), "cubeful1j0": ("cubeful", 1, 0),
         "cubeless0": ("cubeless", 0, 1), "cubeless1": ("cubeless", 1, 1),
         "dmp0": ("dmp", 0, 1), "dmp1": ("dmp", 1, 1)}


def worker(a):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path[:0] = [REPO, os.path.join(REPO, "c_engine")]
    own = os.path.join(os.environ.get("PURETD_LIBDIR", ""), "puretd_py.py")
    if os.path.isfile(own):         # that build's own binding (its symbol set)
        import importlib.util
        spec = importlib.util.spec_from_file_location("puretd_py_arm", own)
        P = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(P)
    else:
        from c_api import puretd_py as P
    out = {"lib": P.lib_path(a.backend), "cells": {}}
    engines = {}
    for cell in a.cells.split(","):
        m, o, j = CELLS[cell]
        e = engines.get(m) or engines.setdefault(
            m, P.Engine(os.path.join(REPO, "best_models", MODELS[m]), a.backend))
        n = a.episodes1 if o else a.episodes0
        opts = P.collect_opts(oneply=o, jacoby=j)
        e.collect(opts, 999, max(1, n // 4))
        times, rows = [], []
        for r in range(a.seeds):
            t0 = time.perf_counter()
            enc, tgt, nd = e.collect(opts, 1000 + r, n, max_rows=200 * n)
            times.append(time.perf_counter() - t0)
            rows.append(len(tgt))
        out["cells"][cell] = {"n": n, "times": times, "rows": rows}
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="library dir of arm A")
    ap.add_argument("--b", required=True, help="library dir of arm B")
    ap.add_argument("--backend", default="simd")
    ap.add_argument("--cells", default="cubeful0,cubeful1",
                    help="comma list of: " + ",".join(CELLS))
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes0", type=int, default=400)
    ap.add_argument("--episodes1", type=int, default=20)
    ap.add_argument("--env-a", default="", help="KEY=VAL,... set for arm A only")
    ap.add_argument("--env-b", default="", help="KEY=VAL,... set for arm B only")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    dirs = {"A": os.path.abspath(a.a), "B": os.path.abspath(a.b)}
    res = {"A": [], "B": []}
    loads = []
    for rd in range(a.rounds):
        for arm in ("AB" if rd % 2 == 0 else "BA"):
            loads.append(os.getloadavg()[0])
            env = dict(os.environ, PURETD_LIBDIR=dirs[arm])
            extra = a.env_a if arm == "A" else a.env_b
            env.update(kv.split("=", 1) for kv in extra.split(",") if kv)
            cmd = [sys.executable, os.path.abspath(__file__), "--worker",
                   "--a", a.a, "--b", a.b, "--backend", a.backend, "--cells", a.cells,
                   "--seeds", str(a.seeds), "--episodes0", str(a.episodes0),
                   "--episodes1", str(a.episodes1)]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
            res[arm].append(json.loads(r.stdout.strip().splitlines()[-1]))
    loads.append(os.getloadavg()[0])
    print(f"# A = {res['A'][0]['lib']} {a.env_a}\n# B = {res['B'][0]['lib']} {a.env_b}")
    print(f"# affinity {sorted(os.sched_getaffinity(0))}; {a.rounds} rounds x {a.seeds} seeds; "
          f"1-min load per run: {' '.join(f'{x:.1f}' for x in loads)}")
    print("| cell | A ep/s | B ep/s | B/A pooled | B/A per seed, median (min..max) | rows/ep A, B |")
    print("|---|---|---|---|---|---|")
    for cell in a.cells.split(","):
        n = res["A"][0]["cells"][cell]["n"]
        ta = [t for run in res["A"] for t in run["cells"][cell]["times"]]
        tb = [t for run in res["B"] for t in run["cells"][cell]["times"]]
        ra = sum(r for run in res["A"] for r in run["cells"][cell]["rows"])
        rb = sum(r for run in res["B"] for r in run["cells"][cell]["rows"])
        ratios = [x / y for x, y in zip(ta, tb)]
        epa, epb = n * len(ta) / sum(ta), n * len(tb) / sum(tb)
        print(f"| {cell} | {epa:.1f} | {epb:.1f} | {epb / epa:.3f} | "
              f"{statistics.median(ratios):.3f} ({min(ratios):.3f}..{max(ratios):.3f}) | "
              f"{ra / (n * len(ta)):.1f}, {rb / (n * len(tb)):.1f} |")


if __name__ == "__main__":
    main()
