"""
fast_forward.py -- the c_api forward backends (libbgt_<backend>.so) as a
drop-in replacement for a TDNetwork's torch forward on CPU.

Only the net forward moves to C; encoding, move generation and every
decision rule stay in Python. That is what lets it serve any input size
and hidden depth, including the matchplay nets (350-356 inputs, 4-5
hidden layers, one sigmoid output), which the PureTD core (puretd.c,
money play only) does not cover.

    from c_api.fast_forward import FastForward
    ff = FastForward(net, backend="simd")   # TDNetwork, ProbNetwork or .pt
    v = ff(features)    # (N, F) fp32 -> (N,) fp32, or (N, 5) for prob5

prob5 nets return the 5 sigmoids (no clamp: the caller post-processes,
as with the torch net). The weights are copied, and re-copied
automatically when any parameter of the source net changes in place
(optimizer step, load_state_dict: torch's tensor version counters), so a
training master's agent never reads stale weights. Opt-in through BG_FAST_FORWARD=<backend>
(torch, simd, cblas, ref; unset/"off" = the torch Python path) for the
matchplay TDAgent (td_agent.TDAgent.enable_fast_forward) and the p5jan
agent (janowski_match.JanowskiMatchAgent).

Backends (c_api/README.md): `torch` runs the same ATen addmm kernels as
nn.Linear; `simd` is the hand-written AVX-512/AVX2 kernel (agrees to
~1e-7, not bitwise); BGC_SIMD_ISA=avx2|scalar caps its ISA.
"""

import ctypes
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

BACKENDS = ("torch", "simd", "cblas", "ref")
ENV = "BG_FAST_FORWARD"

_libs = {}


def lib_dir():
    """PURETD_LIBDIR (build.sh OUT=...), else this module's directory."""
    return os.environ.get("PURETD_LIBDIR") or _HERE


def backend_from_env():
    """BG_FAST_FORWARD as a backend name, or None (off)."""
    v = os.environ.get(ENV, "").strip().lower()
    if v in ("", "0", "off", "none", "python"):
        return None
    if v not in BACKENDS:
        raise ValueError(f"{ENV}={v!r}: expected one of {BACKENDS} or off")
    return v


def _loaded_libtorch():
    """Distinct real paths of libtorch_cpu.so mapped into this process."""
    try:
        with open("/proc/self/maps") as f:
            return {os.path.realpath(ln.split(None, 5)[5].strip())
                    for ln in f if ln.rstrip().endswith("/libtorch_cpu.so")}
    except OSError:
        return set()


def load_lib(backend):
    """ctypes handle of libbgt_<backend>.so with signatures set."""
    if backend in _libs:
        return _libs[backend]
    path = os.path.join(lib_dir(), f"libbgt_{backend}.so")
    if not os.path.isfile(path):
        raise RuntimeError(f"{path} not found; run: bash c_api/build.sh {backend}")
    if backend == "torch":
        import torch  # noqa: F401  (this process's libtorch first)
    L = ctypes.CDLL(path)
    if backend == "torch" and len(_loaded_libtorch()) > 1:
        # libbgt_torch.so was linked against another interpreter's torch:
        # two libtorch_cpu copies in one process is undefined behaviour.
        raise RuntimeError(
            f"{path} links a different libtorch than this interpreter's "
            f"({sorted(_loaded_libtorch())}); rebuild it for this venv: "
            f"VENV={sys.executable} OUT=<dir> bash c_api/build.sh torch, "
            f"then PURETD_LIBDIR=<dir>")
    V, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
    for name, args, res in (
        ("bgt_load", [ctypes.c_char_p], V),
        ("bgt_free", [V], None),
        ("bgt_in_size", [V], I),
        ("bgt_out_dim", [V], I),
        ("bgt_describe", [V], ctypes.c_char_p),
        ("bgt_forward", [V, F, I, I, F], None),
    ):
        fn = getattr(L, name)
        fn.argtypes = args
        fn.restype = res
    _libs[backend] = L
    return L


def supports(net):
    """True when `net` computes exactly the MLP the BGNN export describes:
    a TDNetwork or ProbNetwork whose forward is not overridden."""
    from model import ProbNetwork, TDNetwork
    return any(isinstance(net, c) and type(net).forward is c.forward
               for c in (TDNetwork, ProbNetwork))


def _export(net):
    """BGNN export of a loaded net to a private temp file (path)."""
    import tempfile
    import export_weights
    fd, path = tempfile.mkstemp(prefix="bgt_", suffix=".bin")
    os.close(fd)
    try:
        export_weights.export_live(net, path)
    except BaseException:
        os.unlink(path)
        raise
    return path


class FastForward:
    """A TDNetwork's forward on a c_api backend. Not thread-safe: one
    instance per thread (each holds its own scratch buffers)."""

    def __init__(self, net, backend="torch"):
        from model import ProbNetwork, load_model
        if isinstance(net, str):
            net = load_model(net)
        if not supports(net):
            raise ValueError(f"FastForward: {type(net).__name__} is not a "
                             "plain TDNetwork/ProbNetwork forward")
        self.out_dim = 5 if isinstance(net, ProbNetwork) else 1
        if self.out_dim == 1 and getattr(net, "output_mode", None) not in (
                "probability", "equity"):
            raise ValueError("FastForward: unsupported net (output_mode="
                             f"{getattr(net, 'output_mode', None)!r})")
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self._lib = load_lib(backend)
        self._h = None
        self.refresh(net)

    @staticmethod
    def _version(params):
        return sum(p._version for p in params)

    def refresh(self, net):
        """Re-copy `net`'s current weights."""
        self._net = net
        self._params = list(net.parameters())
        self._ver = self._version(self._params)
        path = _export(net)
        try:
            h = self._lib.bgt_load(path.encode())
        finally:
            os.unlink(path)
        if not h:
            raise RuntimeError(f"bgt_load failed ({self.backend})")
        self.close()
        self._h = h
        self.input_size = self._lib.bgt_in_size(h)
        if self.backend == "torch":
            import torch
            built = self.describe().split()[1]          # "torch <ver> ..."
            if built.split("+")[0] != torch.__version__.split("+")[0]:
                self.close()
                raise RuntimeError(
                    f"libbgt_torch.so was built against torch {built}, this "
                    f"interpreter runs {torch.__version__}; rebuild: "
                    f"VENV={sys.executable} OUT=<dir> bash c_api/build.sh "
                    f"torch, then PURETD_LIBDIR=<dir>")
        out_dim = self._lib.bgt_out_dim(h)
        if out_dim != self.out_dim:
            raise ValueError(f"FastForward expects {self.out_dim} outputs, "
                             f"the export has {out_dim}")

    def describe(self):
        return self._lib.bgt_describe(self._h).decode()

    def close(self):
        if self._h:
            self._lib.bgt_free(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __call__(self, x):
        """(N, F) or (F,) float32 -> (N,) float32, or (N, 5) for prob5
        (fresh array)."""
        if self._version(self._params) != self._ver:
            self.refresh(self._net)        # weights changed in place
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        n, f = x.shape
        if f != self.input_size:
            raise ValueError(f"input has {f} features, net expects {self.input_size}")
        if x.strides[1] != 4 or x.strides[0] % 4:
            x = np.ascontiguousarray(x)
        out = np.empty((n, self.out_dim) if self.out_dim > 1 else n,
                       dtype=np.float32)
        if n:
            self._lib.bgt_forward(
                self._h, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                x.strides[0] // 4, n,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out
