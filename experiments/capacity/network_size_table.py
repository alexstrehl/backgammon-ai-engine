#!/usr/bin/env python3
"""Network-size table and figure data from the committed results.

    python experiments/capacity/network_size_table.py           # strength and cost table, Markdown
    python experiments/capacity/network_size_table.py --bench   # decisions/s table with t-intervals
    python experiments/capacity/network_size_table.py --dat     # results/capacity_{cubeful,prob5}.dat

Head-to-head cells: experiments/head_to_head/results/h2h_experimental_vs_prod.csv.
Decisions/s: experiments/benchmarks/results/<host>_20261014/fast/capacity_sweep*.txt, mean
over the sweeps of each host. Cost factor = production net's checker
dec/s over the row's, same backend and host. A row without sweeps prints
"pending".
"""
import argparse
import csv
import glob
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from statistics import mean, stdev

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "experiments", "benchmarks"))
from aggregate_sweeps import parse, t_crit  # noqa: E402

H2H = os.path.join(_REPO, "experiments", "head_to_head", "results", "h2h_experimental_vs_prod.csv")
SWEEPS = os.path.join(_REPO, "experiments", "benchmarks", "results")
HOSTS = {"zen5": "zen5_9700x", "zen3": "zen3_5950x"}
RUN = "20261014"

# (family, row name, figure label, side1 in the CSV or None for production,
#  model file, hidden layers, capacity_costbench.py label)
ROWS = [
    ("cubeful", "1-layer", "1L", "cubeful_money_512",
     "best_models/experimental/small_models/cubeful_money_512.pt", "[512]", "cfs-1L"),
    ("cubeful", "2-layer", "2L", "cubeful_money_512_512",
     "best_models/experimental/small_models/cubeful_money_512_512.pt", "[512,512]", "cfs-2L"),
    ("cubeful", "3-layer", "3L", "cubeful_money_512_512_256",
     "best_models/experimental/small_models/cubeful_money_512_512_256.pt", "[512,512,256]", "cf3L-mod"),
    ("cubeful", "4-layer (production)", "4L", None,
     "best_models/cubeful_money_512_512_256_256.pt", "[512,512,256,256]", "cf4L"),
    ("cubeful", "5-layer", "5L", "cubeful_money_512_512_256_256_256",
     "best_models/experimental/cubeful_money_512_512_256_256_256.pt", "[512,512,256,256,256]", "cf5L-f9"),
    ("cubeful", "5-layer wide", "5Lw", "cubeful_money_768_512_256_256_256",
     "best_models/experimental/cubeful_money_768_512_256_256_256.pt", "[768,512,256,256,256]", "cf768w5L"),
    ("cubeful", "6-layer wide", "6Lw", "cubeful_money_768_512_256_256_256_256",
     "best_models/experimental/cubeful_money_768_512_256_256_256_256.pt", "[768,512,256,256,256,256]", "cfW1"),
    ("cubeful", "5-layer 768x768", "5Lww", "cubeful_money_768_768_256_256_256",
     "best_models/experimental/cubeful_money_768_768_256_256_256.pt", "[768,768,256,256,256]", "cf768x768"),
    ("prob5", "1-layer", "1L", "cubeless_prob5_512",
     "best_models/experimental/small_models/cubeless_prob5_512.pt", "[512]", "p5s-1L"),
    ("prob5", "2-layer", "2L", "cubeless_prob5_512_512",
     "best_models/experimental/small_models/cubeless_prob5_512_512.pt", "[512,512]", "p5s-2L"),
    ("prob5", "3-layer", "3L", "cubeless_prob5_512_512_256",
     "best_models/experimental/small_models/cubeless_prob5_512_512_256.pt", "[512,512,256]", "p5s-3L"),
    ("prob5", "4-layer (production)", "4L", None,
     "best_models/cubeless_prob5_512_512_256_256.pt", "[512,512,256,256]", "p5-4L256"),
    ("prob5", "5-layer", "5L", "cubeless_prob5_512_512_256_256_256",
     "best_models/experimental/cubeless_prob5_512_512_256_256_256.pt", "[512,512,256,256,256]", "p5-5L"),
    ("prob5", "5-layer wide", "5Lw", "cubeless_prob5_768_512_256_256_256",
     "best_models/cubeless_prob5_768_512_256_256_256.pt", "[768,512,256,256,256]", "p5-768w5L"),
    ("prob5", "6-layer wide", "6Lw", "cubeless_prob5_768_512_256_256_256_256",
     "best_models/experimental/cubeless_prob5_768_512_256_256_256_256.pt", "[768,512,256,256,256,256]", "p5-768w6L"),
    ("prob5", "5-layer 768x768", "5Lww", "cubeless_prob5_768_768_256_256_256",
     "best_models/experimental/cubeless_prob5_768_768_256_256_256.pt", "[768,768,256,256,256]", "p5-768x768"),
]
PRODUCTION = {"cubeful": ("6afda0f8", 562945), "prob5": ("ac95a983", 561925)}
PROD_LABEL = {r[0]: r[6] for r in ROWS if r[3] is None}


def read_h2h():
    with open(H2H) as f:
        return {r["side1"]: r for r in csv.DictReader(ln for ln in f if not ln.startswith("#"))}


def read_rates():
    """{host: {(bench label, backend): (mean, lo, hi, md5)}}: checker dec/s,
    mean and 95% t-interval over the host's sweeps."""
    out = {}
    for host, d in HOSTS.items():
        files = sorted(glob.glob(os.path.join(SWEEPS, f"{d}_{RUN}", "fast", "capacity_sweep*.txt")))
        vals = {}
        for p in files:
            for key, v in parse(p).items():
                if not key.startswith("FP ") or not key.endswith(" checker"):
                    continue
                label, md5 = key[3:-len(" checker")].split()
                if "[" not in label:
                    continue
                lab, b = label[:-1].split("[")
                vals.setdefault((lab, b), []).append((v, md5))
        rates = {}
        for k, xs in vals.items():
            v = [x for x, _ in xs]
            m = mean(v)
            h = t_crit(len(v) - 1) * stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
            rates[k] = (m, m - h, m + h, xs[-1][1])
        out[host] = rates
    return out


def cells():
    h2h, rates = read_h2h(), read_rates()
    for fam, name, lab, side1, path, hidden, blabel in ROWS:
        if side1 is None:
            md5, params = PRODUCTION[fam]
            edge = lo = hi = games = None
        else:
            r = h2h[side1]
            md5, params = r["side1_md5"], int(r["params"])
            edge, lo, hi = (float(r[k]) for k in ("edge_mEq", "ci_lo", "ci_hi"))
            games = int(r["games"])
        base = {(h, b): rates[h].get((PROD_LABEL[fam], b)) for h in HOSTS for b in ("torch", "simd")}
        z = {(h, b): rates[h].get((blabel, b)) for h in HOSTS for b in ("torch", "simd")}
        yield dict(family=fam, name=name, label=lab, path=path, hidden=hidden, md5=md5,
                   params=params, edge=edge, lo=lo, hi=hi, games=games, rates=z, base=base)


def hu(x, q):
    return Decimal(repr(x)).quantize(Decimal(q), rounding=ROUND_HALF_UP)


def num(x):
    if x >= 999.5:
        return f"{hu(x, '1'):,}"
    return f"{hu(x, '1')}" if x >= 99.95 else f"{hu(x, '0.1')}"


def fac(f):
    if abs(f - 1) < 1e-9:
        return "1.00×"
    return f"{hu(f, '0.01')}×"


def rate_cell(c, host, b, ci):
    r, base = c["rates"][(host, b)], c["base"][(host, b)]
    if r is None or base is None:
        return "pending"
    s = f"{num(r[0])} [{num(r[1])}, {num(r[2])}]" if ci else num(r[0])
    return f"{s} ({fac(base[0] / r[0])})"


COLS = [("zen5", "torch"), ("zen3", "torch"), ("zen5", "simd"), ("zen3", "simd")]
HOSTNAME = {"zen5": "Zen 5", "zen3": "Zen 3"}


def print_table():
    print("| model | file | hidden | params | md5 | vs production, mEq [95% CI] | games |"
          + "".join(f" {HOSTNAME[h]} {b} |" for h, b in COLS))
    print("|---|---|---|---|---|---|---|" + "---|" * len(COLS))
    for c in cells():
        prod = c["edge"] is None
        link = f"[`{os.path.basename(c['path'])}`](../../{c['path']})"
        eq = "---" if prod else f"{c['edge']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}]"
        games = "" if prod else f"{c['games'] // 1_000_000}M"
        print(f"| {c['family']} {c['name']} | {link} | {c['hidden']} | {c['params']:,} | `{c['md5']}` "
              f"| {eq} | {games} |" + "".join(f" {rate_cell(c, h, b, False)} |" for h, b in COLS))


def print_bench():
    print("| model | hidden | params |" + "".join(f" {HOSTNAME[h]} {b} |" for h, b in COLS))
    print("|---|---|---|" + "---|" * len(COLS))
    for c in cells():
        print(f"| {c['family']} {c['name']} | {c['hidden']} | {c['params']:,} |"
              + "".join(f" {rate_cell(c, h, b, True)} |" for h, b in COLS))


def write_dat():
    cols = "params\tedge\tehi\telo\tkind\tlabel\tanc\tzen5\tanc2"
    rows = list(cells())
    pending = [f"{c['family']} {c['name']}" for c in rows if c["rates"][("zen5", "torch")] is None]
    if pending:
        sys.exit("Zen 5 torch rates pending (capacity sweeps): " + ", ".join(pending))
    for fam in ("cubeful", "prob5"):
        lines = [cols]
        for c in rows:
            if c["family"] != fam:
                continue
            if c["edge"] is None:
                edge, ehi, elo = "0", "0", "0"
            else:
                edge = f"{c['edge']:.1f}"
                ehi, elo = f"{c['hi'] - c['edge']:.1f}", f"{c['edge'] - c['lo']:.1f}"
            lines.append("\t".join([
                str(c["params"] // 1000), edge, ehi, elo,
                "direct", c["label"], "east" if c["label"] == "5Lw" else "west",
                str(int(round(c["rates"][("zen5", "torch")][0]))),
                "east" if c["label"] == "6Lw" else "west"]))
        out = os.path.join(_HERE, "results", f"capacity_{fam}.dat")
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("wrote", os.path.relpath(out, _REPO))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--bench", action="store_true", help="decisions/s table with 95%% t-intervals")
    g.add_argument("--dat", action="store_true", help="write the figure data files")
    a = ap.parse_args()
    (write_dat if a.dat else print_bench if a.bench else print_table)()
