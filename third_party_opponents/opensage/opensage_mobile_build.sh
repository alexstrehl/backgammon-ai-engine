#!/usr/bin/env bash
# Build Open Sage's native C API (libbgsage_mobile.so) from the pinned release
# source, with the wheel's engine flags (-O3 -ffast-math -funroll-loops -mavx2
# -mfma, set by the release's CMakeLists_cpu.txt). Open Sage is MPL-2.0;
# its source is cloned at build time, not included here.
#   BGSAGE_SRC  existing clone of github.com/markbgsage/bgsage (default: fresh
#               shallow clone of the tag into the build dir)
#   OUT         build dir (default third_party_opponents/opensage/opensage_mobile_build)
set -euo pipefail
TAG=v2.0.20260907
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/opensage_mobile_build}"
mkdir -p "$OUT"
if [ -z "${BGSAGE_SRC:-}" ]; then
  BGSAGE_SRC="$OUT/bgsage-src"
  [ -d "$BGSAGE_SRC" ] || git clone -q --depth 1 --branch "$TAG" \
      https://github.com/markbgsage/bgsage "$BGSAGE_SRC"
fi
got=$(git -C "$BGSAGE_SRC" describe --tags --exact-match 2>/dev/null || echo none)
[ "$got" = "$TAG" ] || { echo "BGSAGE_SRC is at '$got', need $TAG" >&2; exit 1; }
cmake -S "$BGSAGE_SRC/mobile" -B "$OUT/cmake" -DCMAKE_BUILD_TYPE=Release \
      -DBGSAGE_MOBILE_REFERENCE_PYTHON=OFF > "$OUT/cmake.log"
cmake --build "$OUT/cmake" -j "${JOBS:-8}" --target bgsage_mobile >> "$OUT/cmake.log"
cp "$OUT/cmake/libbgsage_mobile.so" "$OUT/"
echo "built $OUT/libbgsage_mobile.so"
