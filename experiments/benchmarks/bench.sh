#!/usr/bin/env bash
# bench.sh: ONE entry point to regenerate every computational-benchmark
# row (the paper's compute tables): fixed-pool decisions, timed per item
# on a single idle, pinned core; labeled result blocks go to $OUT.
#
#   PureTD (C API), Open Sage, HedgeHog:         fixedpool_bench.py
#   gnubg (own embedded interpreter, separate):
#     third_party_opponents/gnubg/fixedpool_bench_gnubg.py
#
# Quote the t-interval over SWEEPS runs, not the per-sweep bootstrap CI.
# See README.
#
# The rows form two tiers (TIER): `fast`, every 0-ply row plus gnubg 1-ply
# and Open Sage 1-ply cubeless, and `slow`, every other 1-ply row and gnubg
# 2-ply. `all` runs both in one sweep. suite.sh drives the two tiers as resumable groups.
#
# Host config (override via env):
#   VENV        python for the native engines; MUST have torch (the binding
#               exports .pt nets). The Open Sage rows also need bgsage and
#               libsage_bench.so
#               (third_party_opponents/opensage/opensage_bench/build.sh),
#               else they are skipped
#                                               [.venv/bin/python]
#   GNUBG       threads-disabled gnubg 1.08 with gnubg.benchpool
#               (third_party_opponents/gnubg/gnubg_bench/build.sh)
#                                               [$HOME/gnubg-bench/bin/gnubg]
#   BENCH_CORE  pinned core; NEVER 0 (IRQ core)  [auto: 8 if present, else the
#                                                 highest core available]
#   FAKEINTEL   opt-in: LD_PRELOAD this libfakeintel.so into the native
#               (MKL) runs; 1 = c_engine/libfakeintel.so   [unset: no preload]
#   TIER        fast | slow | all                            [all]
#   SWEEPS      repeat the whole suite N times            [1]
#   SWEEP_GAP   idle seconds between sweeps               [90]
#   FP_CAP      cap EVERY fixed-pool row at a seeded uniform random sample
#               (smoke tests)                             [full pools]
#   P1_ROOT_FILTER  root move filter of every PureTD/prob5 1-ply row: 0-ply
#               screen, then 1-ply expansion of the moves within this of
#               the best 0-ply score; cube decisions stay full 1-ply.
#               0 = full width                            [0.08]
#   ROWS        rows to run, any of gnubg puretd sage      [all]
#   CAPI_BACKENDS forward backends of the PureTD rows, one row each (C API,
#               c_api/libpuretd_<b>.so; bash c_api/build.sh <b>)  ["torch simd"]
#   OUTDIR      result directory, outside the repo  [$HOME/bench_results]
#   OUT         full output file path     [$OUTDIR/bench[_<tier>].txt]
#
# Run from the repo root:  bash experiments/benchmarks/bench.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

VENV="${VENV:-$REPO/.venv/bin/python}"
GNUBG="${GNUBG:-$HOME/gnubg-bench/bin/gnubg}"
BENCH_CORE="${BENCH_CORE:-}"   # empty = each script picks 8 if present, else the highest core
FAKEINTEL="${FAKEINTEL:-}"; [ "$FAKEINTEL" = 0 ] && FAKEINTEL=""
[ "$FAKEINTEL" = 1 ] && FAKEINTEL="$REPO/c_engine/libfakeintel.so"
SWEEPS="${SWEEPS:-1}"          # repeat the whole suite N times; aggregate with aggregate_sweeps.py
SWEEP="${SWEEP:-}"             # set internally when SWEEPS>1
TIER="${TIER:-all}"
case "$TIER" in fast|slow|all) ;; *) echo "TIER must be fast, slow or all (got '$TIER')" >&2; exit 1;; esac
fast() { [ "$TIER" != slow ]; }
slow() { [ "$TIER" != fast ]; }
OUTDIR="${OUTDIR:-$HOME/bench_results}"
mkdir -p "$OUTDIR"
TIERTAG=""; [ "$TIER" != all ] && TIERTAG="_$TIER"
OUT="${OUT:-$OUTDIR/bench$TIERTAG${SWEEP:+_sweep$SWEEP}.txt}"

# LD_PRELOAD the native (torch/MKL) runs only when FAKEINTEL is set.
LDP=""
SHIM_STATE="off"
if [ -n "$FAKEINTEL" ]; then
  [ -f "$FAKEINTEL" ] || { echo "FATAL: FAKEINTEL=$FAKEINTEL not found (bash c_engine/build_fakeintel.sh)" >&2; exit 1; }
  LDP="$FAKEINTEL"; SHIM_STATE="on ($FAKEINTEL md5 $(md5sum "$FAKEINTEL" | cut -d' ' -f1))"
fi

B="experiments/benchmarks"
GB="third_party_opponents/gnubg"
SB="third_party_opponents/opensage/opensage_bench"
export BENCH_CORE OUTDIR TIER   # must reach the per-sweep child processes

banner() { printf '\n########## %s ##########\n' "$*" | tee -a "$OUT"; }

# Stamp the protocol so a result file says which one produced it.
printf '# MKL shim: %s\n# core: %s  tier: %s  date: %s\n' \
  "$SHIM_STATE" "$BENCH_CORE" "$TIER" "$(date -u +%FT%TZ)" | tee "$OUT"

# Repeat the whole suite SWEEPS times, each a fresh process, then stop. The
# per-sweep spread is the uncertainty that matters (see README).
if [ "$SWEEPS" -gt 1 ] && [ -z "$SWEEP" ]; then
    for i in $(seq 1 "$SWEEPS"); do
        echo "===== sweep $i/$SWEEPS ====="
        SWEEP="$i" SWEEPS=1 OUT="" bash "$0" "$@"
        [ "$i" -lt "$SWEEPS" ] && sleep "${SWEEP_GAP:-90}" || true  # let the core cool
    done
    echo "all $SWEEPS sweeps done -> $OUTDIR/bench${TIERTAG}_sweep*.txt"
    echo "aggregate: python $B/aggregate_sweeps.py $OUTDIR/bench${TIERTAG}_sweep*.txt"
    exit 0
fi
native() { LD_PRELOAD="$LDP" "$VENV" "$@" 2>&1 | tee -a "$OUT"; }
FP_CAP="${FP_CAP:-}"          # cap EVERY fixed-pool row (smoke tests only; empty = full pool)
ROWS=" ${ROWS:-all} "; ROWS="${ROWS//,/ }"
want() { [[ "$ROWS" == *" all "* || "$ROWS" == *" $1 "* ]]; }
# comma list of the non-empty arguments
csv() { local IFS=,; local a=(); for x in "$@"; do [ -n "$x" ] && a+=("$x"); done; echo "${a[*]}"; }
P1_ROOT_FILTER="${P1_ROOT_FILTER:-0.08}"      # PureTD/prob5 1-ply rows: root filter (0 = full width)
P1_TAG="1-ply root-filter $P1_ROOT_FILTER"
awk -v t="$P1_ROOT_FILTER" 'BEGIN{exit !(t+0 == 0)}' && P1_TAG="1-ply full width"
CAPI_BACKENDS="${CAPI_BACKENDS:-torch simd}"
for b in $CAPI_BACKENDS; do
    [ -f "${PURETD_LIBDIR:-$REPO/c_api}/libpuretd_$b.so" ] \
        || { echo "FATAL: libpuretd_$b.so not built: bash c_api/build.sh $b"; exit 1; }
done
CAPI_LIST="${CAPI_BACKENDS// /,}"
gnubg_run() { "$GNUBG" -t -q -p "$@" 2>&1 | tee -a "$OUT"; }

echo "bench.sh  core=${BENCH_CORE:-auto}  tier=$TIER  fpcap=${FP_CAP:-full}  p1root=$P1_ROOT_FILTER  rows=${ROWS// /,}${SWEEP:+  sweep=$SWEEP}" | tee -a "$OUT"
[ "$BENCH_CORE" = "0" ] && { echo "REFUSING BENCH_CORE=0 (IRQ core)"; exit 1; }
"$VENV" -c "import torch" 2>/dev/null || { echo "VENV has no torch: $VENV"; exit 1; }
S_FP=""
if want sage; then
  if ! "$VENV" -c "import bgsage" 2>/dev/null; then
    echo "WARNING: $VENV has no bgsage — Open Sage rows SKIPPED."
  elif [ ! -f "${SAGE_BENCH_LIB:-$SB/build/libsage_bench.so}" ]; then
    echo "WARNING: no libsage_bench.so — Open Sage rows SKIPPED (bash $SB/build.sh)."
  else S_FP=1; fi
fi
P_FP=""; want puretd && P_FP=1
G_OK=""; [ -x "$GNUBG" ] || echo "WARNING: gnubg binary not found ($GNUBG) — gnubg rows skipped."
[ -x "$GNUBG" ] && want gnubg && G_OK=1

if fast; then
if [ -n "$G_OK" ]; then
banner "FIXED-POOL gnubg 0/1-ply (cubeless checker + cubeful checker + cube)"
GNUBG_FP_MODE=both GNUBG_FP_PLIES=0,1 GNUBG_FP_CAP="${FP_CAP:-0}" \
    gnubg_run "$GB/fixedpool_bench_gnubg.py"
fi

CFG=$(csv "${P_FP:+prob5c-0ply}" "${S_FP:+sage-0ply,sage-1ply}")
[ -n "$CFG" ] && { banner "FIXED-POOL native cubeless ($CFG; checker)"
native "$B/fixedpool_bench.py" --mode cubeless --configs "$CFG" \
    --backends "$CAPI_LIST" ${FP_CAP:+--cap "$FP_CAP"}; }

ENG=$(csv "${P_FP:+puretdc}" "${S_FP:+sage0}")
[ -n "$ENG" ] && { banner "FIXED-POOL native cubeful 0-ply ($ENG; checker + cube)"
native "$B/fixedpool_bench.py" --mode cubeful --engines "$ENG" \
    --backends "$CAPI_LIST" ${FP_CAP:+--cap "$FP_CAP"}; }
fi

if slow; then
if [ -n "$G_OK" ]; then
banner "FIXED-POOL gnubg 2-ply (cubeless checker + cubeful checker + cube)"
GNUBG_FP_MODE=both GNUBG_FP_PLIES=2 GNUBG_FP_CAP="${FP_CAP:-0}" \
    gnubg_run "$GB/fixedpool_bench_gnubg.py"
fi

[ -n "$P_FP" ] && { banner "FIXED-POOL native cubeless 1-ply (PureTD prob5 C API [$CAPI_LIST] $P1_TAG)"
native "$B/fixedpool_bench.py" --mode cubeless --configs prob5c-1ply \
    --backends "$CAPI_LIST" ${FP_CAP:+--cap "$FP_CAP"} --root-filter "$P1_ROOT_FILTER"; }

ENG=$(csv "${P_FP:+puretdc}" "${S_FP:+sage}")
[ -n "$ENG" ] && { banner "FIXED-POOL native cubeful 1-ply ($ENG; PureTD $P1_TAG)"
native "$B/fixedpool_bench.py" --mode cubeful --engines "$ENG" --plies 1 \
    --backends "$CAPI_LIST" ${FP_CAP:+--cap "$FP_CAP"} \
    --root-filter "$P1_ROOT_FILTER"; }
fi

banner "DONE -> $OUT"
