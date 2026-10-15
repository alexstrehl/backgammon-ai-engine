#!/usr/bin/env bash
# Build libsage_bench.so (sage_bench.cpp, the Open Sage fixed-pool loops)
# against Open Sage's engine library from the pinned release source
# (MPL-2.0, cloned at build time, not included here).
#   BGSAGE_SRC  clone of github.com/markbgsage/bgsage at the tag (default:
#               shallow clone into OUT)
#   OUT         build dir (default third_party_opponents/opensage/opensage_bench/build)
set -euo pipefail
TAG=v2.0.20260907
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/build}"
mkdir -p "$OUT"
if [ -z "${BGSAGE_SRC:-}" ]; then
  BGSAGE_SRC="$OUT/bgsage-src"
  [ -d "$BGSAGE_SRC" ] || git clone -q --depth 1 --branch "$TAG" \
      https://github.com/markbgsage/bgsage "$BGSAGE_SRC"
fi
BGSAGE_SRC="$(cd "$BGSAGE_SRC" && pwd)"
got=$(git -C "$BGSAGE_SRC" describe --tags --exact-match 2>/dev/null || echo none)
[ "$got" = "$TAG" ] || { echo "BGSAGE_SRC is at '$got', need $TAG" >&2; exit 1; }
cmake -S "$HERE" -B "$OUT/cmake" -DCMAKE_BUILD_TYPE=Release -DBGSAGE_SRC="$BGSAGE_SRC" > "$OUT/cmake.log"
cmake --build "$OUT/cmake" -j "${JOBS:-8}" --target sage_bench >> "$OUT/cmake.log"
cp "$OUT/cmake/libsage_bench.so" "$OUT/"
echo "built $OUT/libsage_bench.so"
