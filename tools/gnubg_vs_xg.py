#!/usr/bin/env python3
"""Compare gnubg 'show statistics match' outputs (one file per match) with XG's
per-player per-match exports on the same recorded matches.

Totals are in EMG (both tools: equity lost in equivalent-money-game units), so
per-player sums and per-match pairs compare directly, independent of each
tool's decision-count definition. PR-like rate = 500 * total EMG lost /
decisions (XG: its non-trivial counts; gnubg: unforced moves + close-or-actual
cube decisions).
"""
import argparse, csv, glob, os, re, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--gnubg-dir", required=True)
ap.add_argument("--xg", nargs=2, default=None, metavar=("P1_TXT", "P2_TXT"), help="XG per-player exports (optional)")
ap.add_argument("--names", nargs=2, default=["white", "black"],
                help="the two player names as gnubg prints them (the .mat exporters here write white / black)")
ap.add_argument("--n-boot", type=int, default=20000)
a = ap.parse_args()

NUM = r"([-+]?\d+\.\d+)\s*\(\s*([-+]?\d+\.\d+)%\)"
INT = r"(\d+)\s+(\d+)"

def parse_gnubg(path):
    t = open(path, errors="replace").read()
    m = re.search(r"^Player\s+(\S+)\s+(\S+)", t, re.M)
    if not m: return None
    order = (m.group(1), m.group(2))
    sec = {}
    # sections in order: Chequerplay, Luck, Cube, Overall; take the lines in sequence
    blocks = re.split(r"\n(Checkerplay statistics|Chequerplay statistics|Luck statistics|Cube statistics|Overall statistics)[ \t]*\n", t)
    cur = None; parts = {}
    for i in range(1, len(blocks), 2):
        parts[blocks[i]] = blocks[i + 1]
    def two(block, label, pat=NUM):
        mm = re.search(re.escape(label) + r"\s+" + pat + r"\s+" + pat, block)
        return mm
    out = {}
    cb = parts.get("Checkerplay statistics", parts.get("Chequerplay statistics", "")); kb = parts.get("Cube statistics", ""); ob = parts.get("Overall statistics", ""); lb = parts.get("Luck statistics", "")
    mu = re.search(r"Unforced moves\s+" + INT, cb); me = two(cb, "Error total EMG (MWC)")
    mc = re.search(r"Close or actual cube decisions\s+" + INT, kb); ke = two(kb, "Error total EMG (MWC)")
    oe = two(ob, "Error total EMG (MWC)"); lu = two(lb, "Luck total EMG (MWC)")
    if not (mu and me and mc and ke and oe): return None
    for k, name in enumerate(order):
        out[name] = dict(moves=int(mu.group(k + 1)), ck_emg=-float(me.group(2 * k + 1)), ck_mwc=-float(me.group(2 * k + 2)),
                         cube_n=int(mc.group(k + 1)), cube_emg=-float(ke.group(2 * k + 1)), cube_mwc=-float(ke.group(2 * k + 2)),
                         all_emg=-float(oe.group(2 * k + 1)), all_mwc=-float(oe.group(2 * k + 2)),
                         luck_mwc=float(lu.group(2 * k + 2)) if lu else float("nan"))
    return out

files = sorted(glob.glob(os.path.join(a.gnubg_dir, "match_*.txt")))
G = {}
for f in files:
    r = parse_gnubg(f)
    if r: G[os.path.basename(f)[:-4]] = r
print(f"gnubg: {len(G)} of {len(files)} match outputs parsed")

def load_xg(p):
    return list(csv.DictReader(open(p, newline=""), delimiter="\t"))
keys = sorted(G)
if a.xg is None:
    # gnubg-only summary: per-player totals and the per-match player difference
    print(f"\nPer-player totals over {len(keys)} matches (gnubg, EMG lost; lower is better):")
    print(f"{'player':10s} {'checker EMG':>12s} {'dec':>6s} {'PR-like':>8s} {'cube EMG':>10s} {'dec':>5s} {'PR-like':>8s} {'all EMG':>9s} {'PR-like':>8s} {'MWC pp/match':>13s} {'luck pp/match':>14s}")
    V = {}
    for name in a.names:
        g = [G[k][name] for k in keys]
        gck = np.array([v["ck_emg"] for v in g]); gckn = np.array([v["moves"] for v in g]); gcu = np.array([v["cube_emg"] for v in g]); gcun = np.array([v["cube_n"] for v in g]); gall = np.array([v["all_emg"] for v in g]); gmwc = np.array([v["all_mwc"] for v in g]); gl = np.array([v["luck_mwc"] for v in g])
        print(f"{name:10s} {gck.sum():12.2f} {gckn.sum():6d} {500*gck.sum()/gckn.sum():8.3f} {gcu.sum():10.2f} {gcun.sum():5d} {500*gcu.sum()/gcun.sum():8.3f} {gall.sum():9.2f} {500*gall.sum()/(gckn.sum()+gcun.sum()):8.3f} {gmwc.mean():13.2f} {np.nanmean(gl):14.2f}")
        V[name] = (gck, gcu, gmwc)
    rng = np.random.default_rng(0); m = len(keys)
    print("\nPlayer difference (%s minus %s), per match, match bootstrap CI:" % tuple(a.names))
    for lab, i in (("checker EMG", 0), ("cube EMG", 1), ("total MWC pp", 2)):
        d = V[a.names[0]][i] - V[a.names[1]][i]
        bs = np.array([d[rng.integers(0, m, m)].mean() for _ in range(a.n_boot)])
        print(f"  {lab:14s} {d.mean():+8.3f} per match  95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
    sys.exit(0)
X = {a.names[0]: load_xg(a.xg[0]), a.names[1]: load_xg(a.xg[1])}
n = len(X[a.names[0]])
keys = keys[:n]
print(f"XG rows: {n} per player; pairing XG row i with {keys[0]}..{keys[-1]} in file order")

def xg_vals(rows):
    ck = np.array([-float(r["Eq Move"]) for r in rows]); ckn = np.array([int(r["Moves"]) for r in rows])
    cu = np.array([-(float(r["Eq Double"]) + float(r["Eq Take"])) for r in rows]); cun = np.array([int(r["Doubles"]) + int(r["Takes"]) for r in rows])
    cost = np.array([-(float(r["Cost Move"]) + float(r["Cost Double"]) + float(r["Cost Take"])) for r in rows]) * 100
    return ck, ckn, cu, cun, cost

print("\nPer-player totals over the matches (EMG lost; lower is better):")
print(f"{'player':10s} {'tool':6s} {'checker EMG':>12s} {'dec':>6s} {'PR-like':>8s} {'cube EMG':>10s} {'dec':>5s} {'PR-like':>8s} {'all EMG':>9s} {'PR-like':>8s} {'MWC pp/match':>13s}")
summary = {}
for name in a.names:
    ck, ckn, cu, cun, cost = xg_vals(X[name])
    print(f"{name:10s} {'XG':6s} {ck.sum():12.2f} {ckn.sum():6d} {500*ck.sum()/ckn.sum():8.3f} {cu.sum():10.2f} {cun.sum():5d} {500*cu.sum()/cun.sum():8.3f} {ck.sum()+cu.sum():9.2f} {500*(ck.sum()+cu.sum())/(ckn.sum()+cun.sum()):8.3f} {cost.mean():13.2f}")
    g = [G[k][name] for k in keys]
    gck = np.array([v["ck_emg"] for v in g]); gckn = np.array([v["moves"] for v in g]); gcu = np.array([v["cube_emg"] for v in g]); gcun = np.array([v["cube_n"] for v in g]); gall = np.array([v["all_emg"] for v in g]); gmwc = np.array([v["all_mwc"] for v in g])
    print(f"{name:10s} {'gnubg':6s} {gck.sum():12.2f} {gckn.sum():6d} {500*gck.sum()/gckn.sum():8.3f} {gcu.sum():10.2f} {gcun.sum():5d} {500*gcu.sum()/gcun.sum():8.3f} {gall.sum():9.2f} {500*gall.sum()/(gckn.sum()+gcun.sum()):8.3f} {gmwc.mean():13.2f}")
    summary[name] = dict(xg=(ck, cu, cost), g=(gck, gcu, gmwc))
    print(f"{'':10s} per-match correlation XG vs gnubg: checker EMG r={np.corrcoef(ck, gck)[0,1]:.3f}, cube EMG r={np.corrcoef(cu, gcu)[0,1]:.3f}, total MWC r={np.corrcoef(cost, gmwc)[0,1]:.3f}")

# the difference between the players, each tool, paired by match, bootstrap
rng = np.random.default_rng(0); m = len(keys)
print("\nPlayer difference (%s minus %s), per match, with a match bootstrap CI:" % tuple(a.names))
for tool, idx in (("XG", "xg"), ("gnubg", "g")):
    A = summary[a.names[0]][idx]; B = summary[a.names[1]][idx]
    for lab, i in (("checker EMG", 0), ("cube EMG", 1), ("total MWC pp", 2)):
        d = A[i] - B[i]
        bs = np.array([d[rng.integers(0, m, m)].mean() for _ in range(a.n_boot)])
        print(f"  {tool:6s} {lab:14s} {d.mean():+8.3f} per match  95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
