#!/usr/bin/env bash
# Build gnubg 1.08.003 with gnubg.benchpool (benchpool.patch): the native
# per-item timing loop fixedpool_bench_gnubg.py uses when present.
#
#   build.sh PREFIX [SRC] [SIMD]
#
#   PREFIX  install prefix (binary lands in PREFIX/bin/gnubg)
#   SRC     gnubg-1.08.003 source dir or source tarball; default: download
#           gnubg-release-1.08.003-sources.tar.gz from ftp.gnu.org/gnu/gnubg.
#           A source dir is copied, never built in place.
#   SIMD    configure --enable-simd value (fma|avx|sse2|no); default: the
#           best the host CPU supports. configure's own auto-detect picks avx
#           on FMA hosts (it tests ax_cv_have_fma_ext, which is never set),
#           so the default here passes fma explicitly.
#
# Env: JOBS (make -j, default nproc), BUILD_DIR (default PREFIX/src).
# CPPFLAGS / LDFLAGS / PKG_CONFIG_PATH pass through to configure (glib must
# be findable). Needs flex, bison and python3 headers.
set -euo pipefail

PREFIX=${1:?usage: build.sh PREFIX [SRC] [SIMD]}
SRC=${2:-}
SIMD=${3:-}
JOBS=${JOBS:-$(nproc)}
PATCH=$(cd "$(dirname "$0")" && pwd)/benchpool.patch
VER=1.08.003
TARBALL=gnubg-release-$VER-sources.tar.gz

mkdir -p "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)
BUILD_DIR=${BUILD_DIR:-$PREFIX/src}

if [ -z "$SIMD" ]; then
    flags=$(grep -m1 '^flags' /proc/cpuinfo || true)
    if [[ " $flags " == *" fma "* && " $flags " == *" avx "* ]]; then SIMD=fma
    elif [[ " $flags " == *" avx "* ]]; then SIMD=avx
    elif [[ " $flags " == *" sse2 "* ]]; then SIMD=sse2
    else SIMD=no
    fi
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
if [ -z "$SRC" ]; then
    curl -fsSL -o "$BUILD_DIR/$TARBALL" "https://ftp.gnu.org/gnu/gnubg/$TARBALL"
    SRC=$BUILD_DIR/$TARBALL
fi
if [ -f "$SRC" ]; then
    tar xzf "$SRC" -C "$BUILD_DIR"
else
    cp -a "$SRC" "$BUILD_DIR/gnubg-$VER"
fi
cd "$BUILD_DIR/gnubg-$VER"
if [ -f Makefile ]; then make distclean >/dev/null; fi
patch -p1 --forward < "$PATCH"

./configure --disable-threads --without-gtk --with-python=python3 \
    --enable-simd="$SIMD" --prefix="$PREFIX" > configure.out
grep -m1 'checking for SIMD CPU instructions' configure.out || true
make -j"$JOBS" > make.out
make install > install.out
echo "built $PREFIX/bin/gnubg (simd=$SIMD)"
