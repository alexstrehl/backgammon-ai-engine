#!/usr/bin/env bash
# Eight-sweep driver for capacity_costbench.py, the protocol of bench.sh: a
# fresh process per sweep on one pinned core, SWEEP_GAP idle seconds between
# sweeps. Each sweep file starts with the "MKL shim" stamp. Aggregate with
# aggregate_sweeps.py.
#
# Env: VENV [.venv/bin/python]  BENCH_CORE [auto]  SWEEPS [8]  SWEEP_GAP [90]
#      BENCH_WARMUP_S [5]  MODELS [all]  CAP [full pools]
#      BACKENDS [torch,simd; `bash c_api/build.sh <b>`]
#      FAKEINTEL  opt-in: LD_PRELOAD this libfakeintel.so; 1 = c_engine/libfakeintel.so [unset: no preload]
#      OUTDIR [$HOME/bench_results/capacity]
#
# Run from the repo root:  bash experiments/benchmarks/capacity_sweeps.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
VENV=${VENV:-.venv/bin/python}
SWEEPS=${SWEEPS:-8}
SWEEP_GAP=${SWEEP_GAP:-90}
MODELS=${MODELS:-all}
BACKENDS=${BACKENDS:-torch,simd}
OUTDIR=${OUTDIR:-$HOME/bench_results/capacity}
FAKEINTEL="${FAKEINTEL:-}"; [ "$FAKEINTEL" = 0 ] && FAKEINTEL=""
[ "$FAKEINTEL" = 1 ] && FAKEINTEL="$PWD/c_engine/libfakeintel.so"
SHIM_STATE="off"
if [ -n "$FAKEINTEL" ]; then
  [ -f "$FAKEINTEL" ] || { echo "FATAL: FAKEINTEL=$FAKEINTEL not found (bash c_engine/build_fakeintel.sh)" >&2; exit 1; }
  SHIM_STATE="on ($FAKEINTEL md5 $(md5sum "$FAKEINTEL" | cut -d' ' -f1))"
fi
mkdir -p "$OUTDIR"
export BENCH_CORE=${BENCH_CORE:-} BENCH_WARMUP_S=${BENCH_WARMUP_S:-5}
for i in $(seq 1 "$SWEEPS"); do
  OUT="$OUTDIR/capacity_sweep${i}.txt"
  echo "== sweep $i/$SWEEPS -> $OUT"
  printf '# MKL shim: %s\n# core: %s  date: %s\n' "$SHIM_STATE" "$BENCH_CORE" "$(date -u +%FT%TZ)" > "$OUT"
  LD_PRELOAD=$FAKEINTEL OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$VENV" experiments/benchmarks/capacity_costbench.py \
      --models "$MODELS" --backends "$BACKENDS" ${CAP:+--cap "$CAP"} --out "$OUT"
  [ "$i" -lt "$SWEEPS" ] && sleep "$SWEEP_GAP"
done
echo "done: python experiments/benchmarks/aggregate_sweeps.py $OUTDIR/capacity_sweep*.txt"
