# Sourced by build.sh. libtorch/ATen from the Python torch wheel (MKL).
# The interpreter whose torch is linked: PURETD_TORCH_PYTHON, else VENV,
# else the repo's .venv, else python3.
# VENV / PURETD_TORCH_PYTHON name an interpreter, not a venv directory.
if [ -n "${PURETD_TORCH_PYTHON:-}" ]; then PY=$PURETD_TORCH_PYTHON; _py_from=PURETD_TORCH_PYTHON
elif [ -n "${VENV:-}" ]; then PY=$VENV; _py_from=VENV
elif [ -x ../.venv/bin/python ]; then PY=../.venv/bin/python; _py_from=.venv
else PY=python3; _py_from=python3
fi
_torch_skip() {
    echo "[torch] no usable torch in $PY (from $_py_from): $1. VENV and" \
         "PURETD_TORCH_PYTHON must be an interpreter path (e.g." \
         "VENV=/path/to/venv/bin/python)" >&2
    return 1
}
TORCH=$("$PY" -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null) \
    || { _torch_skip "cannot run it or import torch"; return 1; }
ABI=$("$PY" -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' 2>/dev/null) \
    || { _torch_skip "no torch._C._GLIBCXX_USE_CXX11_ABI"; return 1; }
[ -f "$TORCH/include/ATen/ATen.h" ] && [ -f "$TORCH/lib/libtorch_cpu.so" ] \
    || { _torch_skip "$TORCH has no include/ATen/ATen.h or lib/libtorch_cpu.so"; return 1; }
BACKEND_SRC=backend_torch.cpp
BACKEND_CC=g++
BACKEND_LINKER=g++
BACKEND_CFLAGS="-std=c++17 -I$TORCH/include -I$TORCH/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=$ABI"
BACKEND_LDFLAGS="-L$TORCH/lib -ltorch_cpu -lc10 -Wl,-rpath,$TORCH/lib"
