"""Per-decision cost of the network-size ladder on the fixed pools.

Protocol of fixedpool_bench.py (one pinned core, clock ramp, 20 untimed
items, ratio of sums, per-sweep bootstrap CI); quote the t-interval over
eight fresh-process sweeps (suite.sh job cap, or capacity_sweeps.sh).
Every net runs through the C API, one row per --backends entry, labelled
`<label>[<backend>]`: the 0-ply checker decision, and for cubeful nets
also the cube decision, each item timed inside C. --backends python times
the Python agents instead (unlabelled rows).

Checkpoints resolve by label through MANIFEST and are pinned by md5 (a
basename is not an identity: best_models/ and small_models/ both hold a
cubeful_money_512_512_256.pt). Search order: best_models/,
best_models/experimental/, best_models/experimental/small_models/.
"""
import argparse
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from bench_common import load_fixedpool, rate, sample  # noqa: E402 (pins the core)
from fixedpool_bench import load_cubeful_pool  # noqa: E402

# label -> (family, filename, accepted md5 prefixes). The committed sweeps
# timed the first md5; a second is a checkpoint of the same shape.
MANIFEST = {
    # cubeful money
    "cfs-1L":    ("cubeful", "cubeful_money_512.pt", ("39d75783",)),
    "cfs-2L":    ("cubeful", "cubeful_money_512_512.pt", ("1ddbb775",)),
    "cf3L-mod":  ("cubeful", "cubeful_money_512_512_256.pt", ("27a728cf",)),
    "cf4L":      ("cubeful", "cubeful_money_512_512_256_256.pt", ("6afda0f8",)),
    "cf5L-f9":   ("cubeful", "cubeful_money_512_512_256_256_256.pt",
                  ("37e41475", "a56adba2")),
    "cf768w5L":  ("cubeful", "cubeful_money_768_512_256_256_256.pt", ("f65c49fe",)),
    "cfW1":      ("cubeful", "cubeful_money_768_512_256_256_256_256.pt",
                  ("4f86345a", "9097ea9c")),
    "cf768x768": ("cubeful", "cubeful_money_768_768_256_256_256.pt", ("de9e6e84",)),
    # cubeless prob5
    "p5s-1L":    ("prob5", "cubeless_prob5_512.pt", ("fa2c6935",)),
    "p5s-2L":    ("prob5", "cubeless_prob5_512_512.pt", ("e8c0c760",)),
    "p5s-3L":    ("prob5", "cubeless_prob5_512_512_256.pt", ("3da8e498",)),
    "p5-4L256":  ("prob5", "cubeless_prob5_512_512_256_256.pt", ("ac95a983",)),
    "p5-5L":     ("prob5", "cubeless_prob5_512_512_256_256_256.pt",
                  ("bade7d51", "953d27c8")),
    "p5-768w5L": ("prob5", "cubeless_prob5_768_512_256_256_256.pt", ("bd1a7931",)),
    "p5-768w6L": ("prob5", "cubeless_prob5_768_512_256_256_256_256.pt",
                  ("a9df2ffc", "90f2bc1d")),
    "p5-768x768": ("prob5", "cubeless_prob5_768_768_256_256_256.pt", ("b3fad4eb",)),
}


def _md5(path, prefix=8):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:prefix]


def resolve(label):
    family, fname, md5s = MANIFEST[label]
    cands = [os.path.join(_REPO, "best_models", *sub, fname)
             for sub in ((), ("experimental",), ("experimental", "small_models"))]
    tried = []
    for p in cands:
        if not os.path.exists(p):
            tried.append(f"{p} (absent)")
            continue
        m = _md5(p)
        if any(m.startswith(x) for x in md5s):
            return family, p, m
        tried.append(f"{p} (md5 {m} != {'/'.join(md5s)})")
    raise SystemExit(f"[{label}] no candidate matches md5 {'/'.join(md5s)}:\n  "
                     + "\n  ".join(tried))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="all",
                    help="comma-separated labels, or 'all'")
    ap.add_argument("--cap", type=int, default=0,
                    help="decisions timed per row, a uniform random sample "
                         "(bench_common.sample; default: full pools; smoke tests)")
    ap.add_argument("--out", default="", help="optional result file (appended)")
    ap.add_argument("--backends", default="torch,simd",
                    help="C API forward backends, one row each; 'python' = the "
                         "Python/PyTorch agents")
    a = ap.parse_args()

    labels = list(MANIFEST) if a.models == "all" else [
        m.strip() for m in a.models.split(",") if m.strip()]

    import numpy as np
    import torch
    from play_models import _load_agent
    from backgammon_engine import get_legal_plays  # noqa: F401 (engine init)

    chk = sample(load_fixedpool(), a.cap)
    checker_pts, cube_pts = load_cubeful_pool()
    checker_pts = sample(checker_pts, a.cap)
    cube_pts = sample(cube_pts, a.cap)
    cap_cl = len(chk)
    cap_cf = len(checker_pts)
    lines = [f"=== capacity fixed-pool per-decision (pools: {len(chk)} cubeless chk, "
             f"{len(checker_pts)}/{len(cube_pts)} cubeful chk/cube; cap={a.cap or 'full'}) ==="]

    backends = [b.strip() for b in a.backends.split(",") if b.strip()]
    if any(b != "python" for b in backends):
        from c_api.puretd_py import CheckerPool, CubePool
        from fixedpool_bench import _c_rate
        from bench_common import c_engine
        pools = {"chk": (CheckerPool(chk[:20]), CheckerPool(chk[:cap_cl])),
                 "cfc": (CheckerPool(checker_pts[:20]), CheckerPool(checker_pts[:cap_cf])),
                 "cfu": (CubePool(cube_pts[:20]),
                         CubePool(cube_pts[:min(cap_cf, len(cube_pts))]))}

    for label, b in [(l, b) for l in labels for b in backends]:
        family, path, m = resolve(label)
        if b != "python":
            eng = c_engine(path, b)
            tag = f"{label}[{eng.backend}]"
            if family == "prob5":
                r, lo, hi = _c_rate(lambda p: eng.bench_checker(p, 0), *pools["chk"])
                row = (f"{tag:<16} {m}  checker {r:9.1f} dec/s "
                       f"[{lo:.1f},{hi:.1f}] ({1000/r:.4f} ms)")
            else:
                rck, lck, hck = _c_rate(lambda p: eng.bench_checker(p, 0), *pools["cfc"])
                rcu, lcu, hcu = _c_rate(lambda p: eng.bench_cube(p, 0), *pools["cfu"])
                row = (f"{tag:<16} {m}  checker {rck:9.1f} dec/s "
                       f"[{lck:.1f},{hck:.1f}] ({1000/rck:.4f} ms)  |  cube "
                       f"{rcu:9.1f} dec/s [{lcu:.1f},{hcu:.1f}] ({1000/rcu:.4f} ms)")
            eng.close()
            print(row, flush=True)
            lines.append(row)
            continue
        if family == "prob5":
            ag = _load_agent(path)
            def fused(sd, ag=ag):
                s, d = sd
                feats, next_states = ag._get_legal_plays_encoded(s, d)
                if len(next_states):
                    int(np.argmin(ag._score(ag._probs(
                        torch.from_numpy(feats))).numpy()))
            r, lo, hi = rate(fused, chk, cap_cl)
            row = (f"{label:<10} {m}  checker {r:9.1f} dec/s "
                   f"[{lo:.1f},{hi:.1f}] ({1000/r:.4f} ms)")
        else:
            ag = _load_agent(path)
            rck, lck, hck = rate(
                lambda it, ag=ag: ag.choose_checker_action_cubeful(*it),
                checker_pts, cap_cf)
            rcu, lcu, hcu = rate(
                lambda it, ag=ag: ag.offer_double(*it), cube_pts,
                min(cap_cf, len(cube_pts)))
            row = (f"{label:<10} {m}  checker {rck:9.1f} dec/s "
                   f"[{lck:.1f},{hck:.1f}] ({1000/rck:.4f} ms)  |  cube "
                   f"{rcu:9.1f} dec/s [{lcu:.1f},{hcu:.1f}] ({1000/rcu:.4f} ms)")
        print(row, flush=True)
        lines.append(row)

    if a.out:
        with open(a.out, "a") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
