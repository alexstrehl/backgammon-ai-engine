#!/bin/bash
# Build libhhagent.so: HedgeHog engine (hedgehog-public) + hh_shim.cpp.
#
# Usage:  build_shim.sh [path-to-hedgehog-public] [output-dir]
#   engine:  arg 1, else $HH_ENGINE, else ./hedgehog-public
#   output:  arg 2, else third_party_opponents/hedgehog/ (where hh_lib.py looks)
#   objects: $BUILD_DIR, else shim/build/ (gitignored)
#
# The engine (MIT) is not vendored (https://gitlab.com/eranlambooij/hedgehog-public).
# Pin e8847f498f7be8798c0b71314e83c8b4075f2043. Minimum: upstream 3db8598
# (the shim needs the Jacoby cube API and the 3-argument cube_can_double;
# the check below tests both).
#   git clone https://gitlab.com/eranlambooij/hedgehog-public.git
#   (cd hedgehog-public && git checkout e8847f4 && make example -j && make run-tests)
#
# The .so is built -march=native: build it on the machine that runs it. A
# copied .so can SIGILL on another CPU.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
HH=${1:-${HH_ENGINE:-./hedgehog-public}}
[ -d "$HH/src" ] || { echo "hedgehog-public not found at $HH" >&2; exit 1; }
{ grep -rq 'cubeful_equity_by_ownership_jacoby' "$HH/src/cube/" &&
  grep -q 'cube_can_double(const CubeState\* *cube, uint8_t player,' "$HH/src/cube/cube_types.h"
} 2>/dev/null || {
  echo "ERROR: engine at $HH lacks the Jacoby cube API or the 3-argument" \
       "cube_can_double (need >= 3db8598, pin e8847f4)." >&2; exit 1; }
DEST=${2:-$HERE/..}
OUT=${BUILD_DIR:-$HERE/build}
mkdir -p "$OUT"
CXX=${CXX:-g++}
FLAGS="-O3 -Wall -std=c++17 -march=native -fPIC -I$HH -I$HH/src -I$HH/lib -I$HH/hwy"

SRCS="
$HH/hwy/hwy/targets.cc
$HH/hwy/hwy/abort.cc
$HH/src/backgammon.cpp
$HH/src/ogid.cpp
$HH/src/nn/nn_eval.cpp
$HH/src/nn/nn_features.cpp
$HH/src/nn/nn_extract_1hot_simd.cpp
$HH/src/nn/nn_io.cpp
$HH/src/nn/dense_chain.cpp
$HH/src/nn/nn_quant.cpp
$HH/src/nn/nn_expert.cpp
$HH/src/nn/nn_expert_registry.cpp
$HH/src/cube/cube_decision.c
$HH/src/cube/cubeful_equity.c
$HH/src/endgame/endgame_stub.cpp
$HERE/hh_shim.cpp
"
for f in $HH/src/nn/experts/*.c $HH/src/nn/experts/*.cpp; do
  [ -e "$f" ] && SRCS="$SRCS $f"
done

OBJS=""
for s in $SRCS; do
  base=$(basename "$s"); base=${base%.*}
  o="$OUT/$base.o"
  if [ ! -f "$o" ] || [ "$s" -nt "$o" ]; then
    echo "CC $base"
    $CXX $FLAGS -c "$s" -o "$o"
  fi
  OBJS="$OBJS $o"
done
$CXX -shared -o "$DEST/libhhagent.so" $OBJS
echo "built $DEST/libhhagent.so"
