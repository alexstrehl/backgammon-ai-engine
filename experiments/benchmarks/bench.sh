#!/usr/bin/env bash
# ==========================================================================
# bench.sh — ONE entry point to regenerate every computational-benchmark
# row (paper Tables 8 cubeless + 9 cubeful). Runs the four bench scripts on a
# single idle, pinned core and writes labeled result blocks to $OUT.
#
#   Native engines (PureTD/prob5 + Open Sage):  selfplay_bench.py / fixedpool_bench.py
#   gnubg (own embedded interpreter, separate):  selfplay_bench_gnubg.py /
#                                                 fixedpool_bench_gnubg.py
#
# Two methodologies: fixed-pool (per-decision, isolated) and self-play
# (realistic decision mix, cube amortized). Quote the t-interval over SWEEPS
# runs, not the per-sweep bootstrap CI; see README. The deep-search rows
# (RUN_DEEP, off by default) cost hours each and are single-run only.
#
# Host config (override via env):
#   VENV        python for the native engines — MUST have torch; bgsage
#               optional (without it the Open Sage rows are skipped)
#                                               [.venv/bin/python]
#   GNUBG       threads-disabled gnubg 1.08 binary   [$HOME/gnubg-st/bin/gnubg]
#   BENCH_CORE  pinned core; NEVER 0 (IRQ core)  [auto: 8 if present, else the
#                                                 highest core available]
#   FAKEINTEL   libfakeintel.so LD_PRELOADed for the native (MKL) runs on AMD
#                                               [c_engine/libfakeintel.so if present]
#   SWEEPS      repeat the whole suite N times            [1]
#   SWEEP_GAP   idle seconds between sweeps               [90]
#   G_FAST      games/row for rows under ~100 ms/game     [600]
#   G_SLOW      games/row for the Open Sage analyzer rows [200]
#   FP_CAP      cap fixed-pool positions (smoke tests)    [full pool]
#   GAMES       games for the RUN_DEEP rows only          [300]
#   RUN_DEEP    1 = also run the multi-hour deep-search self-play rows [0]
#   OUTDIR      where result files are written; kept OUTSIDE the repo so a
#               benchmark run never dirties the tracked tree
#                                        [$HOME/bench_results]
#   OUT         full output file path     [$OUTDIR/bench_$(hostname -s).txt]
#
# Run from the repo root:  bash experiments/benchmarks/bench.sh
# ==========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

VENV="${VENV:-$REPO/.venv/bin/python}"
GNUBG="${GNUBG:-$HOME/gnubg-st/bin/gnubg}"
BENCH_CORE="${BENCH_CORE:-}"   # empty = each script picks 8 if present, else the highest core
GAMES="${GAMES:-300}"
RUN_DEEP="${RUN_DEEP:-0}"
FAKEINTEL="${FAKEINTEL:-$REPO/c_engine/libfakeintel.so}"
SWEEPS="${SWEEPS:-1}"          # repeat the whole suite N times; aggregate with aggregate_sweeps.py
SWEEP="${SWEEP:-}"             # set internally when SWEEPS>1
# Default OUTSIDE the repo: the committed results/ dir holds curated, published
# result files, and a benchmark run must not dirty the tracked tree.
OUTDIR="${OUTDIR:-$HOME/bench_results}"
mkdir -p "$OUTDIR"
OUT="${OUT:-$OUTDIR/bench_$(hostname -s)${SWEEP:+_sweep$SWEEP}.txt}"

# LD_PRELOAD only the native (torch/MKL) runs, and only if the shim exists.
LDP=""; [ -f "$FAKEINTEL" ] && LDP="$FAKEINTEL"

B="experiments/benchmarks"
export BENCH_CORE OUTDIR   # OUTDIR must reach the per-sweep child processes

banner() { printf '\n########## %s ##########\n' "$*" | tee -a "$OUT"; }

# Repeat the whole suite SWEEPS times, each a fresh process, then stop. The
# per-sweep spread is the uncertainty that matters (see README); one long run
# cannot substitute for it, as it samples a single thermal/scheduling state.
if [ "$SWEEPS" -gt 1 ] && [ -z "$SWEEP" ]; then
    for i in $(seq 1 "$SWEEPS"); do
        echo "===== sweep $i/$SWEEPS ====="
        SWEEP="$i" SWEEPS=1 OUT="" bash "$0" "$@"
        [ "$i" -lt "$SWEEPS" ] && sleep "${SWEEP_GAP:-90}" || true  # let the core cool
    done
    echo "all $SWEEPS sweeps done -> $OUTDIR/bench_$(hostname -s)_sweep*.txt"
    echo "aggregate: python $B/aggregate_sweeps.py $OUTDIR/bench_$(hostname -s)_sweep*.txt"
    exit 0
fi
native() { LD_PRELOAD="$LDP" "$VENV" "$@" 2>&1 | tee -a "$OUT"; }
# Two game-count tiers by MEASURED per-game cost, not by search depth -- ply is
# a poor proxy here: Sage's analyzer at 0-ply costs 696 ms/game, 10x more than
# gnubg at 1-ply (55 ms/game). Only the two Sage analyzer rows are slow enough
# to need a reduced count. Within-run sampling error is not the binding
# constraint (between-run variation is, ~1.5% sd), so these put every row's
# within-run se under that rather than equalising it.
G_FAST="${G_FAST:-600}"   # everything under ~100 ms/game
G_SLOW="${G_SLOW:-200}"   # the two Sage analyzer rows (0.7-2.4 s/game)
FP_CAP="${FP_CAP:-}"      # cap fixed-pool positions (smoke tests only; empty = full pool)
gnubg_run() { "$GNUBG" -t -q -p "$@" 2>&1 | tee -a "$OUT"; }

# --- preflight ------------------------------------------------------------
echo "bench.sh  core=${BENCH_CORE:-auto}  fast=$G_FAST  slow=$G_SLOW  deep=$RUN_DEEP${SWEEP:+  sweep=$SWEEP}" | tee "$OUT"
[ "$BENCH_CORE" = "0" ] && { echo "REFUSING BENCH_CORE=0 (IRQ core)"; exit 1; }
"$VENV" -c "import torch" 2>/dev/null || { echo "VENV has no torch: $VENV"; exit 1; }
HAS_SAGE=1
"$VENV" -c "import bgsage" 2>/dev/null || { HAS_SAGE=0; echo "WARNING: $VENV has no bgsage — Open Sage rows SKIPPED. Set VENV to an env that has it for the full suite."; }
[ -x "$GNUBG" ] || echo "WARNING: gnubg binary not found ($GNUBG) — gnubg rows skipped."

# ==========================================================================
# FIXED-POOL (per-decision, isolated)
# ==========================================================================
banner "FIXED-POOL gnubg (cubeless checker + cubeful checker + cube)"
[ -x "$GNUBG" ] && GNUBG_FP_MODE=both GNUBG_FP_PLIES=0,1 GNUBG_FP_CAP="${FP_CAP:-0}" \
    gnubg_run "$B/fixedpool_bench_gnubg.py"

if [ "$HAS_SAGE" = "1" ]; then
    banner "FIXED-POOL native cubeless (prob5 + Open Sage checker)"
    native "$B/fixedpool_bench.py" --mode cubeless ${FP_CAP:+--cap "$FP_CAP"}

    banner "FIXED-POOL native cubeful (PureTD + Open Sage checker + cube)"
    native "$B/fixedpool_bench.py" --mode cubeful --cap "${FP_CAP:-10000}"  # full committed 10k/6k pool
else
    banner "FIXED-POOL native cubeless (prob5 only — no bgsage)"
    native "$B/fixedpool_bench.py" --mode cubeless --configs prob5-0ply,prob5-1ply ${FP_CAP:+--cap "$FP_CAP"}

    banner "FIXED-POOL native cubeful (PureTD only — no bgsage)"
    native "$B/fixedpool_bench.py" --mode cubeful --engines puretd --cap "${FP_CAP:-10000}"  # full committed 10k/6k pool
fi

# ==========================================================================
# SELF-PLAY (realistic decision mix)
# ==========================================================================
banner "SELF-PLAY gnubg (cube off/on, 0/1-ply)"
[ -x "$GNUBG" ] && GNUBG_BENCH_PLIES=0,1 \
    GNUBG_BENCH_G0="$G_FAST" GNUBG_BENCH_G1="$G_FAST" \
    gnubg_run "$B/selfplay_bench_gnubg.py"

banner "SELF-PLAY native cubeless (prob5 + Open Sage)"
native "$B/selfplay_bench.py" --mode cubeless --engine puretd --plies 0 --games "$G_FAST"
[ "$HAS_SAGE" = "1" ] && native "$B/selfplay_bench.py" --mode cubeless --engine sage   --plies 0 --games "$G_FAST"
[ "$HAS_SAGE" = "1" ] && native "$B/selfplay_bench.py" --mode cubeless --engine sage   --plies 1 --games "$G_FAST"

banner "SELF-PLAY native cubeful (PureTD + Open Sage)"
native "$B/selfplay_bench.py" --mode cubeful --engine puretd --plies 0 --games "$G_FAST"
[ "$HAS_SAGE" = "1" ] && native "$B/selfplay_bench.py" --mode cubeful --engine sage   --plies 0 --games "$G_SLOW"  # Sage 0-ply analyzer (0.70 s/game)
[ "$HAS_SAGE" = "1" ] && native "$B/selfplay_bench.py" --mode cubeful --engine sage   --plies 1 --games "$G_SLOW"  # Sage 1-ply analyzer (2.39 s/game)

# --- deep-search self-play rows: hours each -------------------------------
if [ "$RUN_DEEP" = "1" ]; then
    banner "SELF-PLAY deep-search rows (hours each)"
    native "$B/selfplay_bench.py" --mode cubeless --engine puretd --plies 1 --games "$GAMES"                 # prob5-1ply
    native "$B/selfplay_bench.py" --mode cubeful  --engine puretd --plies 1 --games "$GAMES"                 # PureTD 1-ply
    [ "$HAS_SAGE" = "1" ] && native "$B/selfplay_bench.py" --mode cubeful  --engine sage   --plies 1 --sage-unpruned --games "$GAMES" # Sage 1-ply unpruned
    [ "$HAS_SAGE" = "1" ] && native "$B/selfplay_bench.py" --mode cubeful  --engine sage   --plies 2 --sage-unpruned --games "$GAMES" # Sage 2-ply unpruned
else
    echo "" | tee -a "$OUT"
    echo "# deep-search self-play rows SKIPPED (RUN_DEEP=1 to include): cubeless prob5-1ply," | tee -a "$OUT"
    echo "# cubeful PureTD-1ply, Sage 1-ply unpruned, Sage 2-ply unpruned (hours each)." | tee -a "$OUT"
fi

banner "DONE -> $OUT"
