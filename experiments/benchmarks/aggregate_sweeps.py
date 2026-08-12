#!/usr/bin/env python3
"""Aggregate K sweeps of bench.sh into per-row means and 95% t-intervals.

    python experiments/benchmarks/aggregate_sweeps.py \
        $OUTDIR/bench_<host>_sweep*.txt

The sweep is the unit of analysis, NOT the game. Games inside one sweep share
that sweep's thermal/boost state, so pooling raw per-game points across sweeps
treats clustered data as independent and yields an interval several times too
narrow. Here each sweep contributes ONE number per row and the interval is a
t-interval over those K numbers.

Also prints each row's spread across sweeps (the between-run sd), which is the
quantity the design assumes is ~1.5%; check it rather than trusting it.
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


def parse(path):
    """-> {row_label: rate}. One entry per timed row in one sweep."""
    out = {}
    for ln in open(path):
        ln = ln.rstrip("\n")
        if not ln or ln.startswith("#"):
            continue

        # self-play: "<LABEL>: N games Xs ... combined R dec/s"
        m = re.match(r"([A-Z][^:]*?)\s*:\s*\d+ games .*?combined ([\d.]+) dec/s", ln)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
            continue

        # self-play cubeless: "<LABEL>: N games Xs ... (R dec/s, Y ms/dec)"
        m = re.match(r"(CUBELESS [^:]*?)\s*:\s*\d+ games .*?\(([\d.]+) dec/s", ln)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
            continue

        # fixed-pool gnubg: "FIXEDPOOL <what>: N pos Xs | R moves/s" (or evals/s)
        m = re.match(r"(FIXEDPOOL [^:]*?)\s*:\s*\d+ pos .*?\|\s*([\d.]+) (?:moves|evals)/s", ln)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
            continue

        # fixed-pool native cubeless: "<engine>  checker  R moves/s  95%CI [...]"
        m = re.match(r"(\S+)\s+checker\s+([\d.]+) moves/s", ln)
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
    return out


def main(paths):
    sweeps = [parse(p) for p in paths]
    k = len(sweeps)
    if k < 2:
        sys.exit("need >=2 sweep files; got %d" % k)

    labels = [l for l in sweeps[0] if all(l in s for s in sweeps)]
    dropped = sorted(set().union(*(s.keys() for s in sweeps)) - set(labels))

    print(f"{k} sweeps, t({k-1} df) = {t_crit(k-1)}\n")
    print(f"{'row':<40}{'mean':>12}{'95% t-interval':>26}{'+-%':>8}{'sd%':>8}")
    sds = []
    for lab in labels:
        vals = [s[lab] for s in sweeps]
        m = mean(vals)
        sd = stdev(vals)
        half = t_crit(k - 1) * sd / (k ** 0.5)
        sds.append(100 * sd / m)
        print(f"{lab[:39]:<40}{m:>12.1f}{'['+f'{m-half:.1f}, {m+half:.1f}'+']':>26}"
              f"{100*half/m:>7.2f}%{100*sd/m:>7.2f}%")

    if sds:
        print(f"\n  between-sweep sd across rows: median {median(sds):.2f}%, "
              f"range {min(sds):.2f}-{max(sds):.2f}%")
        print("  (the design assumes ~1.5%; if these are much larger, add sweeps)")
    if dropped:
        print("\n  rows absent from at least one sweep (skipped): " + ", ".join(dropped))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
