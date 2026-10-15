#!/usr/bin/env bash
# Build the MKL "fakeintel" shim. On AMD CPUs Intel MKL drops to a slow SSE
# fallback. This overrides mkl_serv_intel_cpu_true() to force the AVX2/AVX-512
# path. LD_PRELOAD libfakeintel.so BEFORE importing torch.
# Opt-in (FAKEINTEL=1 / SHIM=1). Overrides MKL's CPU-vendor check at runtime,
# please check MKL's license (Intel Simplified Software License) before use.
# No effect from MKL 2025.0.
cd "$(dirname "$0")"
gcc -shared -fPIC -o libfakeintel.so fakeintel.c && echo "built libfakeintel.so"
