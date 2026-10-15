#!/usr/bin/env python3
"""Aggregate K sweeps of bench.sh into per-row means and 95% t-intervals.

    python experiments/benchmarks/aggregate_sweeps.py \
        $OUTDIR/bench_sweep*.txt

The sweep is the unit of analysis, not the item: a sweep's bootstrap CI
over items misses the run-to-run variation. Each sweep contributes ONE
number per row and the interval is a t-interval over those K numbers.

Also prints each row's spread across sweeps (the between-run sd).
"""
import re
import sys
from statistics import mean, median, stdev

# two-sided 95% t critical values by degrees of freedom (K-1)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086}


def t_crit(df):
    if df in T95:
        return T95[df]
    # round df DOWN to the nearest tabulated value: fewer df -> larger t, so
    # a gap df never yields an interval that is too narrow
    return T95[max(k for k in T95 if k <= df)]


class _Labels(dict):
    def __setitem__(self, key, value):
        super().__setitem__(" ".join(key.split()), value)


def parse(path):
    """-> {row_label: rate}. One entry per timed row in one sweep; runs of
    whitespace in a label collapse to one space."""
    out = _Labels()
    for ln in open(path):
        ln = ln.rstrip("\n")
        if not ln or ln.startswith("#"):
            continue

        # fixed-pool gnubg: "FIXEDPOOL <what>: N pos Xs | R moves/s" (or evals/s)
        m = re.match(r"(FIXEDPOOL [^:]*?)\s*:\s*\d+ pos .*?\|\s*([\d.]+) (?:moves|evals)/s", ln)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
            continue

        # fixed-pool native cubeless: "<engine>  checker  R moves/s  95%CI [...]"
        # (the label may hold spaces: "prob5c[simd]-1ply root-filter 0.08")
        m = re.match(r"(\S.*?)\s+checker\s+([\d.]+) moves/s", ln)
        if m:
            out["FP " + m.group(1) + " checker"] = float(m.group(2))
            continue

        # fixed-pool native cubeful: "<engine...>  checker R dec/s ... | cube R dec/s ..."
        m = re.match(r"(.+?)\s+checker\s+([\d.]+) dec/s.*?\|\s*cube\s+([\d.]+) dec/s", ln)
        if m:
            eng = m.group(1).strip()
            out["FP " + eng + " checker"] = float(m.group(2))
            out["FP " + eng + " cube"] = float(m.group(3))
            continue

        # capacity_costbench.py prob5 rows: "<label> <md5>  checker R dec/s ..."
        m = re.match(r"(\S+\s+[0-9a-f]{8})\s+checker\s+([\d.]+) dec/s", ln)
        if m:
            out["FP " + m.group(1) + " checker"] = float(m.group(2))
            continue
    return dict(out)


def main(paths):
    sweeps = [parse(p) for p in paths]
    k = len(sweeps)
    if k < 2:
        sys.exit("need >=2 sweep files; got %d" % k)

    labels = [l for l in sweeps[0] if all(l in s for s in sweeps)]
    dropped = sorted(set().union(*(s.keys() for s in sweeps)) - set(labels))

    w = max([len(l) for l in labels] + [3]) + 2
    print(f"{k} sweeps, t({k-1} df) = {t_crit(k-1)}\n")
    print(f"{'row':<{w}}{'mean':>12}{'95% t-interval':>26}{'+-%':>8}{'sd%':>8}")
    sds = []
    for lab in labels:
        vals = [s[lab] for s in sweeps]
        m = mean(vals)
        sd = stdev(vals)
        half = t_crit(k - 1) * sd / (k ** 0.5)
        sds.append(100 * sd / m)
        print(f"{lab:<{w}}{m:>12.1f}{'['+f'{m-half:.1f}, {m+half:.1f}'+']':>26}"
              f"{100*half/m:>7.2f}%{100*sd/m:>7.2f}%")

    if sds:
        print(f"\n  between-sweep sd across rows: median {median(sds):.2f}%, "
              f"range {min(sds):.2f}-{max(sds):.2f}%")
        print("  (the design assumes ~1.5%; if these are much larger, add sweeps)")
    if dropped:
        print("\n  rows absent from at least one sweep (skipped): " + ", ".join(dropped))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
