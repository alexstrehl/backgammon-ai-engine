"""Where a bgc_collect position's time goes: the BGC_PROFILE counters.

    OUT=$SCRATCH/prof CFLAGS_EXTRA=-DBGC_PROFILE bash c_api/build.sh simd
    PURETD_LIBDIR=$SCRATCH/prof taskset -c 61 .venv/bin/python \
        experiments/c_collect/p5_profile.py [--cells cubeful1,cubeful0] [--episodes1 30]

Per cell: one warm-up call, then one profiled Engine.collect call (prod 4L
net, simd unless --backend). The counters (c_api/puretd_internal.h) are
per context (V1: value1, the 1-ply target; SEL: the 0-ply play; CUBE:
bgc_offer_double; VAL: forced-pass bootstrap; COLLECT: the collector's
own rows and loop) and per step (movegen, encode, cube columns, forward,
reduce). Printed: us per call of each context and the share of the
collect wall time; rows per call; per-forward-call rows. The timer pairs
cost ~40 ns each; the unprofiled ep/s is printed for the overhead.
"""
import argparse
import ctypes
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from c_api import puretd_py as P  # noqa: E402

CTX = ["V1", "SEL", "CUBE", "VAL", "COLLECT"]
STEP = ["total", "movegen", "encode", "cubecols", "forward", "reduce"]
MODELS = {"cubeful": "cubeful_money_512_512_256_256.pt",
          "cubeless": "cubeless_money_512_512_256_256.pt",
          "dmp": "dmp_512_512_256_256.pt"}
# name -> (model, oneply, jacoby)
CELLS = {"cubeful1": ("cubeful", 1, 1), "cubeful0": ("cubeful", 0, 1),
         "cubeless1": ("cubeless", 1, 1), "cubeless0": ("cubeless", 0, 1),
         "dmp0": ("dmp", 0, 1), "dmp1": ("dmp", 1, 1)}


def read(lib):
    n = len(CTX) * len(STEP)
    t = (ctypes.c_double * n)()
    c = (ctypes.c_long * n)()
    r = (ctypes.c_long * n)()
    lib.bgc_prof_read(t, c, r)
    k = len(STEP)
    return ({(CTX[i // k], STEP[i % k]): t[i] for i in range(n)},
            {(CTX[i // k], STEP[i % k]): c[i] for i in range(n)},
            {(CTX[i // k], STEP[i % k]): r[i] for i in range(n)})


LAYERS = [(200, 512), (512, 512), (512, 256), (256, 256), (256, 1)]   # cubeful 4L


def read_simd(lib):
    if not hasattr(lib, "bgc_simd_prof_read"):
        return None
    n = 9
    t, r, k = (ctypes.c_double * n)(), (ctypes.c_long * n)(), (ctypes.c_long * n)()
    lib.bgc_simd_prof_read(t, r, k)
    return list(t), list(r), list(k)


def report_simd(sp, in_size):
    t, r, k = sp
    layers = [(in_size, LAYERS[0][1])] + LAYERS[1:]
    tot = sum(t)
    print("\nsimd forward by layer (k entries = k-list length x tile rows, per row):")
    print("| layer | K x M | share of forward | k entries/row | density | MAC/row | GMAC/s |")
    print("|---|---|---|---|---|---|---|")
    for l, (K, M) in enumerate(layers):
        if not r[l]:
            continue
        kr = k[l] / r[l]
        mp = (M + 15) // 16 * 16
        print(f"| {l} | {K}x{M} | {100 * t[l] / tot:.1f}% | {kr:.1f} | {kr / K:.3f} | "
              f"{kr * mp:.0f} | {k[l] * mp / t[l] / 1e9:.1f} |")
    rows = max(r)                    # layer 0 is not counted on the bf16 + delta path
    macs = sum(k[l] * ((M + 15) // 16 * 16) for l, (K, M) in enumerate(layers))
    dense = sum(K * M for K, M in layers) * rows
    print(f"total {macs / rows:.0f} MAC/row walked of {dense / rows:.0f} dense "
          f"({macs / dense:.3f}); {macs / tot / 1e9:.1f} GMAC/s; "
          f"{1e6 * tot / rows:.3f} us/row in the layer loops")


def report(name, t, c, r, wall, n_ep, rows):
    tot = t[("COLLECT", "total")]
    print(f"\n## {name}: {n_ep} episodes, {rows} rows, wall {wall:.3f} s, "
          f"profiled collect {tot:.3f} s ({n_ep / tot:.1f} ep/s profiled)")
    print("\n| context | calls | us/call | share | movegen | encode | cubecols | forward "
          "| reduce | other | rows fwd/call |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    inner = 0.0
    for cx in CTX:
        calls = c[(cx, "total")]
        if cx == "COLLECT":
            body = tot - sum(t[(x, "total")] for x in CTX if x != "COLLECT")
            calls = n_ep
        else:
            body = t[(cx, "total")]
            inner += body
        if calls == 0:
            continue
        parts = [t[(cx, s)] for s in STEP[1:]]
        other = body - sum(parts)
        fc = c[(cx, "forward")]
        fr = r[(cx, "forward")] / fc if fc else 0.0
        cells = " | ".join(f"{1e6 * p / calls:.2f}" for p in parts + [other])
        print(f"| {cx} | {calls} | {1e6 * body / calls:.2f} | {100 * body / tot:.1f}% "
              f"| {cells} | {fr:.1f} |")
    print("\nshares of the collect wall time, by step over all contexts:")
    for s in STEP[1:]:
        v = sum(t[(cx, s)] for cx in CTX)
        print(f"  {s:9s} {100 * v / tot:5.1f}%  ({1e6 * v / n_ep:.0f} us/episode)")
    rest = tot - sum(t[(cx, s)] for cx in CTX for s in STEP[1:])
    print(f"  {'other':9s} {100 * rest / tot:5.1f}%  ({1e6 * rest / n_ep:.0f} us/episode)")
    mg_rows = r[("V1", "movegen")]
    if c[("V1", "total")]:
        n1 = c[("V1", "total")]
        print(f"V1 per position: {mg_rows / n1:.1f} replies, "
              f"{r[('V1', 'forward')] / n1:.1f} forwarded rows, "
              f"{c[('V1', 'movegen')] / n1:.1f} move-gen calls")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="simd")
    ap.add_argument("--cells", default="cubeful1,cubeful0")
    ap.add_argument("--episodes0", type=int, default=2000)
    ap.add_argument("--episodes1", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    lib = P.load_lib(a.backend)
    if not hasattr(lib, "bgc_prof_read"):
        sys.exit(f"{P.lib_path(a.backend)} is not a -DBGC_PROFILE build")
    lib.bgc_prof_read.restype = ctypes.c_int
    lib.bgc_prof_reset.restype = None
    print(f"# {P.lib_path(a.backend)}; affinity {sorted(os.sched_getaffinity(0))}; "
          f"load {os.getloadavg()[0]:.1f}")
    engines = {}
    for cell in a.cells.split(","):
        m, o, j = CELLS[cell]
        e = engines.get(m) or engines.setdefault(
            m, P.Engine(os.path.join(REPO, "best_models", MODELS[m]), a.backend))
        n = a.episodes1 if o else a.episodes0
        opts = P.collect_opts(oneply=o, jacoby=j)
        e.collect(opts, a.seed - 1, max(1, n // 4))            # warm-up
        lib.bgc_prof_reset()
        if hasattr(lib, "bgc_simd_prof_reset"):
            lib.bgc_simd_prof_reset()
        t0 = time.perf_counter()
        enc, tgt, nd = e.collect(opts, a.seed, n, max_rows=200 * n)
        wall = time.perf_counter() - t0
        report(f"{cell} ({e.describe})", *read(lib), wall, n, len(tgt))
        sp = read_simd(lib)
        if sp and m == "cubeful":
            report_simd(sp, e.input_size)
    print(f"\n# load at end {os.getloadavg()[0]:.1f}")


if __name__ == "__main__":
    main()
