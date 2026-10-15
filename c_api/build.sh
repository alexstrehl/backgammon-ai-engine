#!/usr/bin/env bash
# Build the PureTD C API, one shared library per forward backend:
#
#   libpuretd_<backend>.so    core (puretd.c, puretd_collect.c, puretd_janowski.c)
#                             + backend
#   libbgt_<backend>.so       backend + test shim (backend_shim.c)
#   puretd_selfplay_<backend> example binary (example_selfplay.c)
#
#   bash build.sh                 # every backend whose backend_<b>.build.sh finds its deps
#   bash build.sh ref simd        # named backends only; exit 1 if one is skipped
#
# Env:
#   OUT=dir         output directory (default: this directory)
#   PORTABLE=1      drop -march=native (runtime-dispatched backends still use SIMD)
#   CFLAGS_EXTRA    appended to every compile
#
# A backend's build file (backend_<b>.build.sh) is sourced and sets
# BACKEND_SRC, BACKEND_CC, BACKEND_CFLAGS, BACKEND_LDFLAGS, BACKEND_LINKER;
# it returns non-zero when its dependencies are missing (the backend is
# skipped; the exit status is 1 only for a backend named on the command
# line).
#
# Each output is linked to a temp name and renamed into place, so a
# concurrent process never loads a half-written library. Objects go to a
# per-process directory (concurrent builds do not share them).
#
# Never add -ffast-math (see puretd_backend.h).
set -euo pipefail
cd "$(dirname "$0")"

ENGINE=../c_engine
INFER=../c_inference
OUT="${OUT:-.}"
mkdir -p "$OUT"

BASE_CFLAGS="-O3 -Wall -fPIC -I. -I$ENGINE -I$INFER"
if [ "${PORTABLE:-0}" != 1 ]; then BASE_CFLAGS="$BASE_CFLAGS -march=native"; fi
BASE_CFLAGS="$BASE_CFLAGS ${CFLAGS_EXTRA:-}"

backends=("$@")
named=${#backends[@]}
if [ "$named" -eq 0 ]; then backends=(ref simd cblas torch); fi

built=()
skipped=()
objdirs=()
trap 'rm -rf "${objdirs[@]}" "$OUT"/.tmp.$$.*' EXIT
for b in "${backends[@]}"; do
    bf="backend_$b.build.sh"
    if [ ! -f "$bf" ]; then echo "[$b] skip: no $bf"; skipped+=("$b"); continue; fi
    BACKEND_SRC=""; BACKEND_CC=gcc; BACKEND_CFLAGS=""; BACKEND_LDFLAGS=""; BACKEND_LINKER=gcc
    if ! source "./$bf"; then echo "[$b] skip: dependencies missing"; skipped+=("$b"); continue; fi
    if [ ! -f "$BACKEND_SRC" ]; then echo "[$b] skip: no source ($BACKEND_SRC)"; skipped+=("$b"); continue; fi

    obj="$OUT/obj_$b.$$"
    objdirs+=("$obj")
    mkdir -p "$obj"
    echo "[$b] compiling ($BACKEND_CC $BACKEND_CFLAGS)"
    gcc $BASE_CFLAGS -c "$ENGINE/bg_engine.c" -o "$obj/bg_engine.o"
    gcc $BASE_CFLAGS -c "$INFER/nn_eval.c"    -o "$obj/nn_eval.o"
    gcc $BASE_CFLAGS -c backend_shim.c         -o "$obj/backend_shim.o"
    $BACKEND_CC $BASE_CFLAGS $BACKEND_CFLAGS -c "$BACKEND_SRC" -o "$obj/backend.o"

    $BACKEND_LINKER -shared -o "$OUT/.tmp.$$.libbgt_$b.so" \
        "$obj/backend_shim.o" "$obj/backend.o" "$obj/nn_eval.o" "$obj/bg_engine.o" \
        $BACKEND_LDFLAGS -lm
    mv -f "$OUT/.tmp.$$.libbgt_$b.so" "$OUT/libbgt_$b.so"
    echo "[$b] -> $OUT/libbgt_$b.so"

    gcc $BASE_CFLAGS -c puretd.c -o "$obj/puretd.o"
    gcc $BASE_CFLAGS -c puretd_collect.c -o "$obj/puretd_collect.o"
    gcc $BASE_CFLAGS -c puretd_janowski.c -o "$obj/puretd_janowski.o"
    $BACKEND_LINKER -shared -o "$OUT/.tmp.$$.libpuretd_$b.so" \
        "$obj/puretd.o" "$obj/puretd_collect.o" "$obj/puretd_janowski.o" "$obj/backend.o" "$obj/nn_eval.o" "$obj/bg_engine.o" \
        $BACKEND_LDFLAGS -lm
    mv -f "$OUT/.tmp.$$.libpuretd_$b.so" "$OUT/libpuretd_$b.so"
    echo "[$b] -> $OUT/libpuretd_$b.so"
    gcc $BASE_CFLAGS -o "$OUT/.tmp.$$.puretd_selfplay_$b" example_selfplay.c \
        -L"$OUT" -lpuretd_$b -Wl,-rpath,"$(cd "$OUT" && pwd)" -lm
    mv -f "$OUT/.tmp.$$.puretd_selfplay_$b" "$OUT/puretd_selfplay_$b"
    echo "[$b] -> $OUT/puretd_selfplay_$b"
    built+=("$b")
done
echo "built: ${built[*]:-none}"
if [ "$named" -gt 0 ] && [ ${#skipped[@]} -gt 0 ]; then
    echo "skipped: ${skipped[*]} (named on the command line)" >&2
    exit 1
fi
