#!/bin/bash
# Reproduce the single-thread moves/s benchmark.
#
# For each engine config we run several seeds and report mean +/- sd moves/s.
# Strictly single-thread (OMP/MKL/torch = 1). Defaults reproduce the paper
# table; override with: ./run.sh <n> "<seeds>" <path|agent|lean|both>
#
# No hard-coded paths -- resolved relative to this script.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

N="${1:-40000}"                 # positions per seed
SEEDS="${2:-1 2 3 4 5}"         # seeds to average over
WHICH="${3:-both}"              # agent | lean | both

run_path () {                   # $1 = script, $2 = label
    local script="$1" label="$2"
    echo "===================================================================="
    echo " $label   (n=$N per seed, seeds: $SEEDS, single-thread)"
    echo "===================================================================="
    for cfg in ours0 gnubg0 gnubg1; do
        for s in $SEEDS; do
            python3 "$HERE/$script" --config "$cfg" --n "$N" --seed "$s"
        done
    done
    echo
}

[ "$WHICH" = "agent" ] || [ "$WHICH" = "both" ] && run_path bench_agent.py "AGENT path"
[ "$WHICH" = "lean"  ] || [ "$WHICH" = "both" ] && run_path bench_lean.py  "LEAN path"
