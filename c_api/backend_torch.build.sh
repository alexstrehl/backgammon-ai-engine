# Sourced by build.sh. libtorch/ATen from the Python torch wheel (MKL).
# The interpreter whose torch is linked: PURETD_TORCH_PYTHON, else VENV,
# else the repo's .venv, else python3.
PY=${PURETD_TORCH_PYTHON:-${VENV:-}}
[ -n "$PY" ] || { [ -x ../.venv/bin/python ] && PY=../.venv/bin/python; } || PY=python3
TORCH=$("$PY" -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null) || return 1
ABI=$("$PY" -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' 2>/dev/null) || return 1
[ -f "$TORCH/include/ATen/ATen.h" ] && [ -f "$TORCH/lib/libtorch_cpu.so" ] || return 1
BACKEND_SRC=backend_torch.cpp
BACKEND_CC=g++
BACKEND_LINKER=g++
BACKEND_CFLAGS="-std=c++17 -I$TORCH/include -I$TORCH/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=$ABI"
BACKEND_LDFLAGS="-L$TORCH/lib -ltorch_cpu -lc10 -Wl,-rpath,$TORCH/lib"
