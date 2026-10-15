# Sourced by build.sh. Hand-written SIMD backend: runtime ISA dispatch
# (AVX-512 / AVX2+FMA / scalar), no dependencies.
BACKEND_SRC=backend_simd.c
BACKEND_CC=gcc
BACKEND_CFLAGS="-std=gnu11"
BACKEND_LDFLAGS=""
BACKEND_LINKER=gcc
