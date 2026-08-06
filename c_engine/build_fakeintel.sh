#!/usr/bin/env bash
# Build the MKL "fakeintel" shim. On AMD CPUs Intel MKL drops to a slow SSE
# fallback; this overrides mkl_serv_intel_cpu_true() to force the AVX2/AVX-512
# path. LD_PRELOAD libfakeintel.so BEFORE importing torch.
cd "$(dirname "$0")"
gcc -shared -fPIC -o libfakeintel.so fakeintel.c && echo "built libfakeintel.so"
