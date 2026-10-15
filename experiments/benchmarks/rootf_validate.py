#!/usr/bin/env python3
"""Decision agreement of the C API 1-ply root filter with full width, on
the first N items of the committed benchmark pools (prob5 4L on the
cubeless pool, cubeful 4L on the cubeful checker pool; default: all).

    python experiments/benchmarks/rootf_validate.py [--n 10000] [--workers 16]

Per threshold: the share of decisions (items with at least two candidates)
whose filtered choice equals the full-width choice, the mean full-width
equity lost by the filtered choice, and the mean number of candidates
expanded. Needs c_api/libpuretd_simd.so (bash c_api/build.sh simd).
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["BENCH_NO_PIN"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path[:0] = [REPO, os.path.join(REPO, "experiments", "benchmarks"),
                os.path.join(REPO, "c_engine")]
MODELS = {"prob5": "best_models/cubeless_prob5_512_512_256_256.pt",
          "cubeful": "best_models/cubeful_money_512_512_256_256.pt"}


def points(kind, n):
    """[(state, dice, match_state or None)] of the first n pool items."""
    if kind == "prob5":
        from bench_common import load_fixedpool
        return [(s, d, None) for s, d in load_fixedpool(n)]
    from fixedpool_bench import load_cubeful_pool
    return load_cubeful_pool()[0][:n]


def work(args):
    kind, n, lo, hi, thresholds = args
    from c_api import puretd_py as P
    e = P.Engine(os.path.join(REPO, MODELS[kind]), "simd")
    out = []
    for s, d, ms in points(kind, n)[lo:hi]:
        e.set_root_filter(0.0)
        full = e.score_moves(s, d, ms, 1)[0]
        if len(full) < 2:
            continue
        ib = e.best_move(s, d, ms, 1)[0]
        row = [len(full)]
        for t in thresholds:
            e.set_root_filter(t)
            kept = e.score_moves(s, d, ms, 1)[0]
            i = e.best_move(s, d, ms, 1)[0]
            row += [i == ib, float(full[ib] - full[i]), int(np.sum(~np.isnan(kept)))]
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10000, help="pool items per kind (default 10000, capped at the pool size)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--thresholds", default="0.08,0.16")
    a = ap.parse_args()
    ts = tuple(float(x) for x in a.thresholds.split(","))
    for kind in ("prob5", "cubeful"):
        n = len(points(kind, a.n))
        step = (n + a.workers - 1) // a.workers
        jobs = [(kind, a.n, lo, min(lo + step, n), ts) for lo in range(0, n, step)]
        with mp.get_context("spawn").Pool(a.workers) as p:
            rows = sum(p.map(work, jobs), [])
        r = np.array(rows, dtype=np.float64)
        print(f"{kind}: {n} pool items, {len(r)} with >= 2 candidates "
              f"(mean {r[:, 0].mean():.1f} candidates)")
        for j, t in enumerate(ts):
            agree, loss, kept = r[:, 1 + 3 * j], r[:, 2 + 3 * j], r[:, 3 + 3 * j]
            print(f"  root {t}: agreement {100 * agree.mean():.2f}% "
                  f"({int((1 - agree).sum())} differ), mean loss "
                  f"{1000 * loss.mean():.4f} mEq/decision (max {1000 * loss.max():.1f}), "
                  f"mean expanded {kept.mean():.2f} of {r[:, 0].mean():.1f}")


if __name__ == "__main__":
    main()
