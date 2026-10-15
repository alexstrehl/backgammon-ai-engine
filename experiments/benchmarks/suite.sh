#!/usr/bin/env bash
# suite.sh OUT [fast|slow|all]: the whole benchmark suite as two resumable
# groups of NSWEEPS (default 8) fresh-process sweeps, one benchmark process
# at a time on one pinned core (the protocol of bench.sh / README "Protocol").
#
#   fast  bench.sh TIER=fast (every 0-ply row, gnubg 1-ply, Open Sage 1-ply
#         cubeless), the HedgeHog 0-ply rows and the network-size ladder
#         (capacity_costbench.py)
#   slow  bench.sh TIER=slow (every other 1-ply row, gnubg 2-ply) and the
#         HedgeHog 1-ply rows
# Every row is a fixed-pool row.
#
# One sweep = every job of the group once, back to back; SWEEP_GAP idle
# seconds between sweeps. Each job writes OUT/<group>/<job>_sweep<i>.txt;
# a job whose file exists is skipped, so re-running the same command resumes.
# A failed job is retried once (its output goes to OUT/contaminated/).
#
# ROWS selects rows (default all): bench.sh's tokens (gnubg puretd sage)
# and this script's jobs (hh, cap); a group runs only the selected rows
# of its tier.
#
# Env: VENV GNUBG BENCH_CORE FAKEINTEL CAPI_BACKENDS P1_ROOT_FILTER pass
# through; NSWEEPS [8]  SWEEP_GAP [90]  BENCH_WARMUP_S [5]  SMOKE=1 (tiny counts,
# one sweep)  HH_FOX / HH_AUREUS / HH_LIB (HedgeHog models and shim).
#
# Run from the repo root:
#   BENCH_CORE=<core> VENV=... GNUBG=... bash experiments/benchmarks/suite.sh ~/bench_results/<run> fast
#   python experiments/benchmarks/aggregate_sweeps.py ~/bench_results/<run>/fast/bench_sweep*.txt
set -uo pipefail
OUT=${1:?usage: suite.sh OUT [fast|slow|all]}; GROUPS_=${2:-all}
case "$GROUPS_" in fast|slow) ;; all) GROUPS_="fast slow";; *) echo "group must be fast, slow or all" >&2; exit 1;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO" || exit 1
B=experiments/benchmarks
VENV="${VENV:-$REPO/.venv/bin/python}"
GNUBG="${GNUBG:-$HOME/gnubg-bench/bin/gnubg}"
FAKEINTEL="${FAKEINTEL:-}"; [ "$FAKEINTEL" = 0 ] && FAKEINTEL=""
[ "$FAKEINTEL" = 1 ] && FAKEINTEL="$REPO/c_engine/libfakeintel.so"
CAPI_BACKENDS="${CAPI_BACKENDS:-torch simd}"; CAPI_LIST="${CAPI_BACKENDS// /,}"
export BENCH_CORE="${BENCH_CORE:-}" BENCH_WARMUP_S="${BENCH_WARMUP_S:-5}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VENV GNUBG FAKEINTEL CAPI_BACKENDS
NS=${NSWEEPS:-8}; GAP=${SWEEP_GAP:-90}
# --cap of the hh and cap jobs (set by SMOKE; empty = full pools).
FPC=""
export ROWS="${ROWS:-all}"
want(){ [[ " ${ROWS//,/ } " == *" all "* || " ${ROWS//,/ } " == *" $1 "* ]]; }
PRECHECK=${PRECHECK:-1}   # 0 = skip the idle-box check (smoke tests)
if [ "${SMOKE:-0}" = 1 ]; then
  export FP_CAP=30 BENCH_WARMUP_S=0.5
  FPC="--cap 30"; NS=${NSWEEPS:-1}; PRECHECK=0
fi
SHIM_STATE="off"
if [ -n "$FAKEINTEL" ]; then
  [ -f "$FAKEINTEL" ] || { echo "FATAL: FAKEINTEL=$FAKEINTEL not found (bash c_engine/build_fakeintel.sh)" >&2; exit 1; }
  SHIM_STATE="on ($FAKEINTEL md5 $(md5sum "$FAKEINTEL" | cut -d' ' -f1))"
fi
mkdir -p "$OUT/contaminated"
P=$OUT/progress.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$P"; }

# The box must be idle apart from us: host-wide busy cores (/proc/stat over
# 10 s) and load average, re-checked every minute until clean.
precheck(){
  read -r _ a1 b1 c1 d1 e1 f1 g1 h1 _ < /proc/stat; sleep 10; read -r _ a2 b2 c2 d2 e2 f2 g2 h2 _ < /proc/stat
  local busy=$(( (a2+b2+c2+f2+g2+h2) - (a1+b1+c1+f1+g1+h1) )) tot=$(( (a2+b2+c2+d2+e2+f2+g2+h2) - (a1+b1+c1+d1+e1+f1+g1+h1) ))
  local bc; bc=$(awk -v b=$busy -v t=$tot -v n="$(nproc --all)" 'BEGIN{printf "%.2f", n*b/t}')
  local la; la=$(cut -d' ' -f1 /proc/loadavg)
  PRE="busy_cores=$bc loadavg1=$la"
  awk -v b="$bc" -v l="$la" 'BEGIN{exit !(b < 0.5 && l < 1.5)}'
}
wait_clean(){
  local n=0
  until precheck; do n=$((n+1)); [ $((n % 5)) -eq 1 ] && log "precheck DIRTY before $1: $PRE"; sleep 60; done
  log "precheck CLEAN before $1: $PRE"
}
# run_job TAG FILE LOG cmd...: fresh process; stdout to LOG, or to FILE.tmp
# when LOG is empty; FILE.tmp -> FILE on success, one retry.
run_job(){
  local tag=$1 of=$2 logf=$3; shift 3; local fails=0
  [ -f "$of" ] && { log "SKIP $tag (done)"; return 0; }
  RAN=1
  while true; do
    log "START $tag"
    "$@" > "${logf:-$of.tmp}" 2>&1; local rc=$?
    if [ $rc -ne 0 ]; then
      fails=$((fails+1)); log "FAIL $tag rc=$rc: $(tail -3 "$of.tmp" | tr '\n' ' ' | cut -c1-300)"
      mv "$of.tmp" "$OUT/contaminated/$(basename "$of").fail$fails"
      [ $fails -ge 2 ] && { log "FATAL $tag failed twice"; touch "$OUT/FATAL"; exit 1; }
      sleep "$GAP"; continue
    fi
    mv "$of.tmp" "$of"; log "DONE $tag"; return 0
  done
}
nat(){ LD_PRELOAD="$FAKEINTEL" "$VENV" "$@"; }
stamp(){ printf '# commit %s  BENCH_CORE=%s  BENCH_WARMUP_S=%s  MKL shim %s  date %s\n' \
  "$(git rev-parse --short HEAD 2>/dev/null || cat CODE_SHA 2>/dev/null || echo unknown)" "$BENCH_CORE" "$BENCH_WARMUP_S" "$SHIM_STATE" "$(date -u +%FT%TZ)"; }
multi(){ stamp; local c; for c in "$@"; do printf '\n########## %s ##########\n' "$c"; eval "$c" || return 1; done; }

# jobs, one sweep each
job_bench(){  # bench.sh, one tier: it writes the result file itself (OUT)
  local tier=$1 i=$2
  want gnubg || want puretd || want sage || return 0
  local f=$OUT/$tier/bench_sweep$i.txt
  run_job "${tier}_bench_s$i" "$f" "$OUT/$tier/bench_sweep$i.log" \
    env TIER="$tier" SWEEP="$i" SWEEPS=1 OUT="$f.tmp" OUTDIR="$OUT/$tier" bash $B/bench.sh
}
job_hh0(){ want hh || return 0; run_job "fast_hh0_s$1" "$OUT/fast/hh0_sweep$1.txt" "" multi \
  "nat $B/fixedpool_bench.py --mode cubeless --configs prob5c-0ply,hh-fox-0ply,hh-aureus-0ply --backends $CAPI_LIST $FPC" \
  "nat $B/fixedpool_bench.py --mode cubeful --engines puretdc,hh-fox0,hh-aureus0 --backends $CAPI_LIST $FPC"; }
job_cap(){ want cap || return 0; run_job "fast_cap_s$1" "$OUT/fast/capacity_sweep$1.txt" "" multi \
  "nat $B/capacity_costbench.py --models all --backends $CAPI_LIST $FPC"; }
job_hh1(){ want hh || return 0; run_job "slow_hh1_s$1" "$OUT/slow/hh1_sweep$1.txt" "" multi \
  "nat $B/fixedpool_bench.py --mode cubeless --configs hh-fox-1ply,hh-aureus-1ply $FPC" \
  "nat $B/fixedpool_bench.py --mode cubeful --engines hh-fox1,hh-aureus1 $FPC"; }

# The gap follows a sweep that ran; a resume skips finished sweeps without it.
group_fast(){ for i in $(seq 1 "$NS"); do mkdir -p "$OUT/fast"; RAN=0
  job_bench fast "$i"; job_hh0 "$i"; job_cap "$i"
  [ "$i" -lt "$NS" ] && [ "$RAN" = 1 ] && sleep "$GAP"; done; }
group_slow(){ for i in $(seq 1 "$NS"); do mkdir -p "$OUT/slow"; RAN=0
  job_bench slow "$i"; job_hh1 "$i"
  [ "$i" -lt "$NS" ] && [ "$RAN" = 1 ] && sleep "$GAP"; done; }

log "SUITE start core=$BENCH_CORE warmup=$BENCH_WARMUP_S sweeps=$NS gap=$GAP groups='$GROUPS_' rows='$ROWS' smoke=${SMOKE:-0} shim=$SHIM_STATE venv=$VENV"
for g in $GROUPS_; do
  [ -f "$OUT/FATAL" ] && rm -f "$OUT/FATAL"
  [ "$PRECHECK" = 1 ] && wait_clean "$g"
  log "GROUP $g begin"; group_$g; log "GROUP $g end"
done
log "ALLDONE"
echo "aggregate each job over its sweeps, e.g.:"
for g in $GROUPS_; do for j in "$OUT/$g"/*_sweep1.txt; do [ -f "$j" ] && echo "  python $B/aggregate_sweeps.py ${j%_sweep1.txt}_sweep*.txt"; done; done
