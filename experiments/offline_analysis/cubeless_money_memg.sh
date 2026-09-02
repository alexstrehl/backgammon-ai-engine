#!/usr/bin/env bash
# ============================================================================
# Cubeless-money mEMG — reproduction script
# ----------------------------------------------------------------------------
# Computes cubeless-money mEMG (mean error per move, milli-equity, judged by
# gnubg at N-ply) for three engines, each at 0-ply and 1-ply:
#       PureTD-prob5 (ours)      gnubg 1.08 (gnubg-ext)       Open Sage
# Move selection AND the gnubg judge use CUBELESS money equity — gammons and
# backgammons count, no cube/recube leverage (the objective our cubeless models
# are trained on). Lower mEMG = stronger.
#
# All real logic lives in the codebase; this file is just the invocations + combine:
#   experiments/offline_analysis/run_cubeless_memg.py  self-play -> gnubg N-ply scoring -> bootstrap CIs
#                                   (players: prob5/sage-{0,1}ply, gnubg-ext-{0,1}ply; cubeless default)
#   gnubg_eval.py                   export_mat(money_game=True) writes a "0 point match"
#                                   (money) session; _write_gnubg_script emits
#                                   'set analysis chequerplay evaluation cubeful off'
#                                   + 'set analysis cube off' when GNUBG_CUBELESS_EVAL=1
#                                   (the runner sets it by default; --cubeful-eval disables).
#
# Prerequisites:
#   gnubg 1.08.003 CLI on GNUBG_CMD (judge AND gnubg-ext players). Recent
#   Debian/Ubuntu apt ships 1.08.003; older releases ship 1.06 -- the runtime
#   preflight verifies version + weights md5 and refuses anything else.
#   pip install bgsage                    (sage-0ply / sage-1ply players; needs AVX2)
#   torch + this repo; model best_models/cubeless_prob5_512_512_256_256.pt (prod; override with PROB5_MODEL)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."                    # repo root

# OUT must be a fresh directory per campaign: the per-game writer APPENDS,
# so reusing a directory silently pools new games onto old ones.
OUT="${OUT:-/tmp/cubeless_memg_repro}"
GAMES="${GAMES:-4000}"
PLIES="${PLIES:-4}"
PLAY_WORKERS="${PLAY_WORKERS:-32}"            # ~physical cores (self-play peaks ~56 on a 96c box)
GNUBG_WORKERS="${GNUBG_WORKERS:-32}"          # gnubg N-ply analysis scales ~1 per physical core

run() {  # run "<comma-separated players>"
  python3 experiments/offline_analysis/run_cubeless_memg.py --players "$1" \
      --games "$GAMES" --plies "$PLIES" \
      --play-workers "$PLAY_WORKERS" --gnubg-workers "$GNUBG_WORKERS" \
      --save-pergame "$OUT"
  # cubeless-money is the default; add --cubeful-eval for the cube-leveraged variant.
}

run prob5-0ply,prob5-1ply        # PureTD-prob5 (ours)
run gnubg-ext-0ply,gnubg-ext-1ply  # real gnubg 1.08 (external socket; 1-ply unpruned)
run sage-0ply                    # Open Sage 0-ply (production filter = published row)
# The published sage-1ply row (1.670) is UNPRUNED, the paper convention; drop
# the env var and you are measuring Sage's production filter instead.
SAGE_UNPRUNED=1 run sage-1ply    # Open Sage 1-ply unpruned

# --- process output: pooled equal-n table (chequer + overall mEMG, bootstrap 95% CIs) ---
python3 experiments/offline_analysis/run_cubeless_memg.py --combine "$OUT"
