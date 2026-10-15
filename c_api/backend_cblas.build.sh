# Sourced by build.sh. OpenBLAS backend (LP64 CBLAS: cblas_sgemm/sgemv).
# Resolution: $OPENBLAS_DIR, else pkg-config openblas, else ldconfig.
# numpy's bundled OpenBLAS (ILP64, scipy_openblas64_ symbols) is never used.
BACKEND_SRC=backend_cblas.c
BACKEND_CC=gcc
BACKEND_LINKER=gcc
BACKEND_CFLAGS=""
if [ -n "${OPENBLAS_DIR:-}" ]; then
    if [ ! -e "$OPENBLAS_DIR/lib/libopenblas.so" ]; then
        echo "[cblas] OPENBLAS_DIR=$OPENBLAS_DIR has no lib/libopenblas.so" >&2
        return 1
    fi
    BACKEND_LDFLAGS="-L$OPENBLAS_DIR/lib -lopenblas -Wl,-rpath,$OPENBLAS_DIR/lib"
elif pkg-config --exists openblas 2>/dev/null; then
    _ob_libdir=$(pkg-config --variable=libdir openblas)
    BACKEND_LDFLAGS="$(pkg-config --libs openblas) -Wl,-rpath,$_ob_libdir"
elif _ob_so=$(ldconfig -p 2>/dev/null | grep -o '/[^ ]*/libopenblas[^ /]*\.so[.0-9]*$' | grep -v 64_ | head -1) \
        && [ -n "$_ob_so" ]; then
    BACKEND_LDFLAGS="$_ob_so -Wl,-rpath,$(dirname "$_ob_so")"
else
    return 1
fi
echo "[cblas] OpenBLAS: $BACKEND_LDFLAGS"
