#!/bin/bash
# Build libbg_nply.so: fused move-gen + nn-eval + N-ply search.
#
# Needs a CBLAS exporting the standard LP64 symbol names (cblas_sgemm).
# On Debian/Ubuntu: `sudo apt install libopenblas-dev`.
#
# NOTE: numpy ships its own OpenBLAS in numpy.libs/, but that build is ILP64
# with mangled symbols (scipy_cblas_sgemm64_) and does NOT export cblas_sgemm,
# so it cannot satisfy this link without editing the sgemm calls in bg_nply.c.
set -e
cd "$(dirname "$0")"

# Resolve a BLAS: explicit override, then system (apt), then a conda tree.
if [ -n "${BLAS_LIB:-}" ]; then
    BLAS_LINK="$BLAS_LIB"
elif ldconfig -p 2>/dev/null | grep -q "libopenblas\.so"; then
    BLAS_LINK="-lopenblas"
elif [ -f /opt/conda/lib/libopenblas.so.0 ]; then
    BLAS_LINK="-Wl,-rpath,/opt/conda/lib /opt/conda/lib/libopenblas.so.0"
elif ldconfig -p 2>/dev/null | grep -q "libblas\.so"; then
    BLAS_LINK="-lblas"
else
    echo "FATAL: no usable CBLAS found." >&2
    echo "  install one with:  sudo apt install libopenblas-dev" >&2
    echo "  or set BLAS_LIB to a library exporting cblas_sgemm." >&2
    exit 1
fi
echo "Building libbg_nply.so (BLAS: $BLAS_LINK) ..."

# NO -ffast-math. It implies -ffinite-math-only, which lets the compiler
# assume expf() never overflows to inf -- but the sigmoid is
# 1/(1+expf(-x)), and a large negative pre-activation legitimately
# overflows (mathematically -> 0). With -ffast-math that path produced
# **NaN on ~0.13% of real positions** (11 of 8218 sampled encodings), which
# would silently poison BCE targets in training. Measured cost of removing
# it: none -- the build is actually 1.6% FASTER without it, because the hot
# path is OpenBLAS sgemm (hand-tuned asm), which compiler math flags do not
# touch. See PROB5_C_ENGINE_FINDINGS.md.
gcc -O3 -march=native -funroll-loops -shared -fPIC \
    -o libbg_nply.so \
    bg_nply.c bg_engine.c ../c_inference/nn_eval.c \
    -I. -I../c_inference \
    $BLAS_LINK -lm -Wall
echo "Success: libbg_nply.so created."
