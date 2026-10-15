#!/bin/bash
# Build libbg_nply.so: fused move-gen + nn-eval + N-ply search.
#
# Needs OpenBLAS, not a generic BLAS: bg_nply.c calls
# openblas_set_num_threads and links the LP64 cblas_sgemm.
# On Debian/Ubuntu: `sudo apt install libopenblas-dev`.
# numpy's bundled OpenBLAS (ILP64, mangled symbols) does NOT export
# cblas_sgemm and cannot satisfy this link.
set -e
cd "$(dirname "$0")"

# Resolve OpenBLAS: explicit override, then system (apt), then a conda tree.
if [ -n "${BLAS_LIB:-}" ]; then
    BLAS_LINK="$BLAS_LIB"
elif ldconfig -p 2>/dev/null | grep -q "libopenblas\.so"; then
    BLAS_LINK="-lopenblas"
elif [ -f /opt/conda/lib/libopenblas.so.0 ]; then
    BLAS_LINK="-Wl,-rpath,/opt/conda/lib /opt/conda/lib/libopenblas.so.0"
else
    echo "FATAL: OpenBLAS not found. bg_nply.c needs openblas_set_num_threads, so a generic BLAS will not link." >&2
    echo "  install it with:  sudo apt install libopenblas-dev" >&2
    echo "  or set BLAS_LIB to the link flags of an OpenBLAS exporting cblas_sgemm," >&2
    echo "  e.g. BLAS_LIB='-Wl,-rpath,/path/to/lib /path/to/lib/libopenblas.so.0'" >&2
    exit 1
fi
echo "Building libbg_nply.so (BLAS: $BLAS_LINK) ..."

# NO -ffast-math. It implies -ffinite-math-only, but the sigmoid's expf()
# legitimately overflows to inf, and the flag turns that into NaNs.
gcc -O3 -march=native -funroll-loops -shared -fPIC \
    -o libbg_nply.so \
    bg_nply.c bg_engine.c ../c_inference/nn_eval.c \
    -I. -I../c_inference \
    $BLAS_LINK -lm -Wall
echo "Success: libbg_nply.so created."
