#!/usr/bin/env bash
# ==========================================================================
# reproduce.sh — ONE entry point to regenerate every computational-benchmark
# row (paper Tables 8 cubeless + 9 cubeful). Runs the four bench scripts on a
# single idle, pinned core and writes labeled result blocks to $OUT.
#
#   Native engines (PureTD/prob5 + Open Sage):  selfplay_bench.py / fixedpool_bench.py
#   gnubg (own embedded interpreter, separate):  selfplay_bench_gnubg.py /
#                                                 fixedpool_bench_gnubg.py
#
# Methodology: fixed-pool (per-decision, isolated) for the cubeless table and
# self-play (realistic decision mix, cube amortized) for the cubeful table;
# both are the conservative choice. Every row carries a bootstrap 95% CI
# except the deep-search self-play rows (RUN_DEEP, off by default) which cost
# hours each and are reported as single-run point estimates.
#
# Host config (override via env):
#   VENV        python for the native engines — MUST have torch + bgsage
#                                               [.venv/bin/python]
#   GNUBG       threads-disabled gnubg 1.08 binary   [$HOME/gnubg-st/bin/gnubg]
#   BENCH_CORE  pinned core; NEVER 0 (IRQ core)       [8]
#   FAKEINTEL   libfakeintel.so LD_PRELOADed for the native (MKL) runs on AMD
#                                               [c_engine/libfakeintel.so if present]
#   GAMES       self-play games per config             [300]
#   RUN_DEEP    1 = also run the multi-hour deep-search self-play rows [0]
#   OUT         output file            [results/reproduce_$(hostname -s).txt]
#
# Run from the repo root:  bash experiments/benchmarks/reproduce.sh
# ==========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

VENV="${VENV:-$REPO/.venv/bin/python}"
GNUBG="${GNUBG:-$HOME/gnubg-st/bin/gnubg}"
BENCH_CORE="${BENCH_CORE:-8}"
GAMES="${GAMES:-300}"
RUN_DEEP="${RUN_DEEP:-0}"
FAKEINTEL="${FAKEINTEL:-$REPO/c_engine/libfakeintel.so}"
OUT="${OUT:-$REPO/experiments/benchmarks/results/reproduce_$(hostname -s).txt}"

# LD_PRELOAD only the native (torch/MKL) runs, and only if the shim exists.
LDP=""; [ -f "$FAKEINTEL" ] && LDP="$FAKEINTEL"

B="experiments/benchmarks"
export BENCH_CORE

banner() { printf '\n########## %s ##########\n' "$*" | tee -a "$OUT"; }
native() { LD_PRELOAD="$LDP" "$VENV" "$@" 2>&1 | tee -a "$OUT"; }
gnubg_run() { "$GNUBG" -t -q -p "$@" 2>&1 | tee -a "$OUT"; }

# --- preflight ------------------------------------------------------------
echo "reproduce.sh  core=$BENCH_CORE  games=$GAMES  deep=$RUN_DEEP" | tee "$OUT"
[ "$BENCH_CORE" = "0" ] && { echo "REFUSING BENCH_CORE=0 (IRQ core)"; exit 1; }
"$VENV" -c "import torch" 2>/dev/null || { echo "VENV has no torch: $VENV"; exit 1; }
"$VENV" -c "import bgsage" 2>/dev/null || echo "WARNING: $VENV has no bgsage — Sage rows will fail. Set VENV to an env that has it."
[ -x "$GNUBG" ] || echo "WARNING: gnubg binary not found ($GNUBG) — gnubg rows skipped."

# ==========================================================================
# FIXED-POOL (per-decision, isolated)
# ==========================================================================
banner "FIXED-POOL gnubg (cubeless checker + cubeful checker + cube)"
[ -x "$GNUBG" ] && GNUBG_FP_MODE=both GNUBG_FP_PLIES=0,1 gnubg_run "$B/fixedpool_bench_gnubg.py"

banner "FIXED-POOL native cubeless (prob5 + Open Sage checker)"
native "$B/fixedpool_bench.py" --mode cubeless

banner "FIXED-POOL native cubeful (PureTD + Open Sage checker + cube)"
native "$B/fixedpool_bench.py" --mode cubeful --cap 10000  # full committed 10k/6k pool

# ==========================================================================
# SELF-PLAY (realistic decision mix)
# ==========================================================================
banner "SELF-PLAY gnubg (cube off/on, 0/1-ply)"
[ -x "$GNUBG" ] && GNUBG_BENCH_PLIES=0,1 \
    GNUBG_BENCH_G0="$GAMES" GNUBG_BENCH_G1="$GAMES" \
    gnubg_run "$B/selfplay_bench_gnubg.py"

banner "SELF-PLAY native cubeless (prob5 + Open Sage)"
native "$B/selfplay_bench.py" --mode cubeless --engine puretd --plies 0 --games "$GAMES"
native "$B/selfplay_bench.py" --mode cubeless --engine sage   --plies 0 --games "$GAMES"
native "$B/selfplay_bench.py" --mode cubeless --engine sage   --plies 1 --games "$GAMES"
native "$B/selfplay_bench.py" --mode cubeless --engine sage   --plies 1 --sage-unpruned --games "$GAMES"

banner "SELF-PLAY native cubeful (PureTD + Open Sage)"
native "$B/selfplay_bench.py" --mode cubeful --engine puretd --plies 0 --games "$GAMES"
native "$B/selfplay_bench.py" --mode cubeful --engine sage   --plies 0 --games "$GAMES"  # SAGE-AN 1P
native "$B/selfplay_bench.py" --mode cubeful --engine sage   --plies 1 --games "$GAMES"  # SAGE-AN 2P

# --- deep-search self-play rows: hours each -------------------------------
if [ "$RUN_DEEP" = "1" ]; then
    banner "SELF-PLAY deep-search rows (hours each)"
    native "$B/selfplay_bench.py" --mode cubeless --engine puretd --plies 1 --games "$GAMES"                 # prob5-1ply
    native "$B/selfplay_bench.py" --mode cubeful  --engine puretd --plies 1 --games "$GAMES"                 # PureTD 1-ply
    native "$B/selfplay_bench.py" --mode cubeful  --engine sage   --plies 1 --sage-unpruned --games "$GAMES" # SAGE-AN 2P unpruned
    native "$B/selfplay_bench.py" --mode cubeful  --engine sage   --plies 2 --sage-unpruned --games "$GAMES" # SAGE-AN 3P unpruned
else
    echo "" | tee -a "$OUT"
    echo "# deep-search self-play rows SKIPPED (RUN_DEEP=1 to include): cubeless prob5-1ply," | tee -a "$OUT"
    echo "# cubeful PureTD-1ply, Sage-2P-unpruned, Sage-3P-unpruned (hours each)." | tee -a "$OUT"
fi

banner "DONE -> $OUT"
