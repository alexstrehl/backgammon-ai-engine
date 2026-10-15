#!/usr/bin/env python3
"""Tables for p8_results.md from a sweep output dir (sweep.tsv and run dirs
in p4_sweep.py's layout): per cell, the pipe and shm runs (ep/s over rounds
2..N, the 3..N rate), their means, shm / pipe, and the mean BG_ROUND_PHASES breakdown (s/round) per arm.

    python experiments/c_collect/p8_table.py $OUT
"""
import csv
import os
import re
import statistics as st
import sys

PH = ("collect", "submit", "assemble", "h2d", "perm", "steps")


def phases(log):
    s = open(os.path.join(os.path.dirname(log), "summary.txt")).read()
    m = re.search(r"phases s/round \(BG_ROUND_PHASES, rounds 2..N\): (.*)", s)
    kv = m.group(1).split() if m else []
    iv = re.search(r"round interval s\s+mean ([\d.]+)", s)
    out = {k: float(v) for k, v in zip(kv[::2], kv[1::2])}
    out["interval"] = float(iv.group(1)) if iv else float("nan")
    return out


def main():
    d = sys.argv[1]
    rows = [r for r in csv.DictReader(open(os.path.join(d, "sweep.tsv")),
                                      delimiter="\t") if r["clean"] == "1"]
    cells = {}
    for r in rows:
        cell, arm = r["label"].rsplit("_", 1)
        cells.setdefault(cell, {}).setdefault(arm, []).append(r)
    print("| cell | arm | ep/s per run (3..N) | mean ep/s | shm / pipe | "
          "round s | collect | train | load1 at start | foreign cores |")
    print("|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for cell, arms in cells.items():
        means = {a: st.mean(float(r["eps"]) for r in rs) for a, rs in arms.items()}
        for a in ("pipe", "shm"):
            rs = arms.get(a, [])
            if not rs:
                continue
            ph = [phases(r["log"]) for r in rs]
            ratio = (f"**{means['shm'] / means['pipe']:.3f}**"
                     if a == "shm" and "pipe" in means else "")
            print(f"| {cell} | {a} | "
                  + ", ".join(f"{float(r['eps']):.0f} ({float(r['eps3']):.0f})" for r in rs)
                  + f" | **{means[a]:.0f}** | {ratio} | "
                  f"{st.mean(p['interval'] for p in ph):.3f} | "
                  f"{st.mean(float(r['collect']) for r in rs):.3f} | "
                  f"{st.mean(float(r['train']) for r in rs):.3f} | "
                  + ", ".join(r["load1"] for r in rs) + " | "
                  + ", ".join(r["foreign_mean"] for r in rs) + " |")
    print()
    print("| cell | arm | " + " | ".join(PH) + " |")
    print("|---|---|" + "---:|" * len(PH))
    for cell, arms in cells.items():
        for a in ("pipe", "shm"):
            ph = [phases(r["log"]) for r in arms.get(a, [])]
            if not ph:
                continue
            print(f"| {cell} | {a} | " + " | ".join(
                f"{st.mean(p[k] for p in ph):.4f}" if all(k in p for p in ph) else "-"
                for k in PH) + " |")


if __name__ == "__main__":
    main()
