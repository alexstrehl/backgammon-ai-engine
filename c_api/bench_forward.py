#!/usr/bin/env python3
"""Forward-pass parity and throughput for one or more backends
(libbgt_<backend>.so, built by build.sh), against torch on the same rows.

    python c_api/bench_forward.py --backends ref,simd \
        --model best_models/cubeful_money_512_512_256_256.pt

Rows are real perspective196 encodings from random playouts (plus random
cube columns for a cubeful encoder: the 3-hot perspective and
is_cube_action; for a matchplay encoder its match block of a random match
context: length up to 25, Crawford / post-Crawford, owned cubes up to 16),
so the input sparsity matches play. --bin times a given BGNN file instead
of the model's export (e.g. the format-1 twin for a library built before
format 2).
Exports are cached under CAPI_BINS (default ~/.cache/puretd/bins), keyed
as `<stem>_<md5[:12]>_bgnv<format version>.bin`.
Timing runs inside C (bgt_time_forward); the torch line is the
single-thread torch forward on the same rows. Pins to BENCH_CORE like the
benchmarks (tools/bench_common.py).

Output: one line per (backend, n): us/call and GFLOPS, and the parity
block (max |C - torch| over the parity rows). --json writes everything.
"""
import argparse
import ctypes
import hashlib
import json
import os
import random
import struct
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "c_engine"))
sys.path.insert(0, os.path.join(_REPO, "tools"))

import bench_common  # the import pins the core and single-threads torch
import numpy as np
import torch

from backgammon_engine import get_legal_plays, opening_roll, switch_turn
from model import load_model
import bg_fast

DATA_BINS = os.environ.get(
    "CAPI_BINS", os.path.expanduser("~/.cache/puretd/bins"))
N_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
_FP = ctypes.POINTER(ctypes.c_float)


def cache_path(pt_path):
    """Where export_bin caches `pt_path`'s export: `<stem>_<md5[:12]>_bgnv<format
    version>.bin` (the format in the key: a cached file is always this
    exporter's format)."""
    import export_weights
    h = hashlib.md5(open(pt_path, "rb").read()).hexdigest()[:12]
    stem = os.path.basename(pt_path)[:-3]
    return os.path.join(DATA_BINS, f"{stem}_{h}_bgnv{export_weights.FORMAT_VERSION}.bin")


def export_bin(pt_path):
    """BGNN export of a checkpoint, cached under DATA_BINS (cache_path).
    A cached file is reused only when its header is this format and names
    the checkpoint's encoder and width; else (or when missing) the
    checkpoint is exported again, through a temp file renamed into place."""
    import export_weights
    os.makedirs(DATA_BINS, exist_ok=True)
    out = cache_path(pt_path)
    net = None
    if os.path.isfile(out):
        net = load_model(pt_path)
        try:
            hd = export_weights.read_header(out)
            ok = (hd["format_version"] == export_weights.FORMAT_VERSION
                  and hd["encoder_name"] == net.encoder_name
                  and hd["input_size"] == net.input_size)
        except (ValueError, OSError, struct.error):
            ok = False
        if ok:
            return out
    tmp = f"{out}.{os.getpid()}.tmp"
    export_weights.export_live(net if net is not None else load_model(pt_path), tmp)
    os.chmod(tmp, 0o600)
    os.replace(tmp, out)
    return out


def _random_match_extras(enc, rng):
    """`enc`'s match block for a random valid match context."""
    from encoding import CubePerspective
    L = rng.choice((1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25))
    my, opp = rng.randint(1, L), rng.randint(1, L)
    craw = (my == 1) != (opp == 1) and rng.random() < 0.35
    value, persp = 1, CubePerspective.CENTERED
    if not craw and rng.random() < 0.6:
        value = rng.choice((2, 2, 4, 4, 8, 16))
        persp = rng.choice((CubePerspective.MINE, CubePerspective.THEIRS))
    return enc.match_extras(persp, value, my, opp, craw, bool(rng.randrange(2)))


def playout_rows(n, in_size, seed=0, encoder_name=None):
    """(n, in_size) float32 encodings of random-playout positions; the
    cube or match block (if any) is `encoder_name`'s (encoding.get_encoder;
    by default the 196-input base alone)."""
    from encoding import CubefulEncoder, get_encoder
    enc = get_encoder(encoder_name or "perspective196")
    assert enc.num_features == in_size, (encoder_name, enc.num_features, in_size)
    cubeful = isinstance(enc, CubefulEncoder)
    match = hasattr(enc, "match_extras")
    assert (enc.base_name if cubeful or match else enc.name) == "perspective196"   # bg_fast's
    nb = (in_size - enc.cube_features if cubeful
          else enc._base.num_features if match else in_size)
    rng = random.Random(seed)
    rows = np.zeros((n, in_size), dtype=np.float32)
    k = 0
    while k < n:
        state, dice = opening_roll(rng)
        while not state.is_game_over() and k < n:
            rows[k, :nb] = bg_fast.encode_state(state)
            if cubeful:
                rows[k, nb + rng.randrange(3)] = 1.0
                rows[k, nb + 3] = float(rng.randrange(2))
            elif match:
                rows[k, nb:] = _random_match_extras(enc, rng)
            k += 1
            plays = get_legal_plays(state, dice)
            state = rng.choice(plays)[1] if plays else switch_turn(state)
            dice = (rng.randint(1, 6), rng.randint(1, 6))
    return rows


class Backend:
    def __init__(self, name, out_dir, bin_path):
        so = os.path.join(out_dir, f"libbgt_{name}.so")
        self.name = name
        self.lib = ctypes.CDLL(so)
        L = self.lib
        L.bgt_load.argtypes = [ctypes.c_char_p]; L.bgt_load.restype = ctypes.c_void_p
        L.bgt_free.argtypes = [ctypes.c_void_p]
        L.bgt_in_size.argtypes = [ctypes.c_void_p]; L.bgt_in_size.restype = ctypes.c_int
        L.bgt_out_dim.argtypes = [ctypes.c_void_p]; L.bgt_out_dim.restype = ctypes.c_int
        L.bgt_params.argtypes = [ctypes.c_void_p]; L.bgt_params.restype = ctypes.c_longlong
        L.bgt_describe.argtypes = [ctypes.c_void_p]; L.bgt_describe.restype = ctypes.c_char_p
        L.bgt_forward.argtypes = [ctypes.c_void_p, _FP, ctypes.c_int, ctypes.c_int, _FP]
        L.bgt_time_forward.argtypes = [ctypes.c_void_p, _FP, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, _FP]
        L.bgt_time_forward.restype = ctypes.c_double
        self.h = L.bgt_load(bin_path.encode())
        if not self.h:
            raise RuntimeError(f"{so}: bgt_load failed for {bin_path}")
        self.in_size = L.bgt_in_size(self.h)
        self.out_dim = L.bgt_out_dim(self.h)
        self.params = L.bgt_params(self.h)
        self.describe = L.bgt_describe(self.h).decode()

    def forward(self, x):
        """x: (n, ldx) float32, C-contiguous; ldx >= in_size."""
        out = np.empty((x.shape[0], self.out_dim), dtype=np.float32)
        self.lib.bgt_forward(self.h, x.ctypes.data_as(_FP), x.shape[1], x.shape[0],
                             out.ctypes.data_as(_FP))
        return out

    def time(self, x, reps):
        out = np.empty((x.shape[0], self.out_dim), dtype=np.float32)
        return self.lib.bgt_time_forward(self.h, x.ctypes.data_as(_FP), x.shape[1],
                                         x.shape[0], reps, out.ctypes.data_as(_FP))


def padded_layout(rows):
    """The core's layout: ldx = in_size rounded up to 16, zero pad
    columns, row 0 64-byte aligned. A view; its base keeps the buffer alive."""
    n, in_size = rows.shape
    ldx = (in_size + 15) // 16 * 16
    raw = np.zeros(n * ldx + 16, dtype=np.float32)
    off = (-raw.ctypes.data % 64) // 4
    x = raw[off: off + n * ldx].reshape(n, ldx)
    x[:, :in_size] = rows
    assert x.ctypes.data % 64 == 0 and x.flags.c_contiguous
    return x


def thread_count():
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("Threads:"):
                    return int(ln.split()[1])
    except OSError:
        pass
    return -1


def torch_forward(net, x):
    with torch.no_grad():
        y = net(torch.from_numpy(x))
        if getattr(net, "raw_logits", False):
            y = torch.sigmoid(y)
    return y.reshape(x.shape[0], -1).numpy()


def timed(fn, target_s=0.4):
    """Seconds per call: fn(reps) returns the seconds `reps` calls took;
    reps grows until one run takes about target_s."""
    reps = 1
    while True:
        t = fn(reps)
        if t >= target_s or reps >= 1 << 20:
            return t / reps
        reps = max(reps * 2, int(reps * target_s / max(t, 1e-9)) + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default="ref")
    ap.add_argument("--model", default="best_models/cubeful_money_512_512_256_256.pt")
    ap.add_argument("--out-dir", default=_HERE, help="where the libbgt_*.so live")
    ap.add_argument("--n-list", default=",".join(map(str, N_LIST)))
    ap.add_argument("--parity-rows", type=int, default=4096)
    ap.add_argument("--no-torch", action="store_true", help="skip the torch timing line")
    ap.add_argument("--json", default="")
    ap.add_argument("--bin", default="", help="time this BGNN file (of --model's "
                    "weights) instead of the model's export")
    a = ap.parse_args()

    pt = a.model if os.path.isabs(a.model) else os.path.join(_REPO, a.model)
    binp = a.bin or export_bin(pt)
    net = load_model(pt).eval()
    in_size = net.input_size
    n_list = [int(v) for v in a.n_list.split(",") if v]
    rows = playout_rows(max(max(n_list), a.parity_rows), in_size,
                        encoder_name=net.encoder_name)
    ref = torch_forward(net, rows[: a.parity_rows])
    params = sum(p.numel() for p in net.parameters())

    print(f"model {os.path.basename(pt)}  in={in_size} params={params:,}  "
          f"core={os.environ.get('BENCH_CORE', 'auto')}")
    results = {"model": os.path.basename(pt), "params": params, "rows": {}}

    tight = np.ascontiguousarray(rows[: a.parity_rows])   # ldx == in_size
    padded = padded_layout(rows)                           # the core's layout

    for name in [s.strip() for s in a.backends.split(",") if s.strip()]:
        be = Backend(name, a.out_dir, binp)
        thr0 = thread_count()
        # Parity at both layouts: a backend may not assume pad columns.
        d_tight = np.abs(be.forward(tight) - ref).max()
        d_pad = np.abs(be.forward(padded[: a.parity_rows]) - ref).max()
        print(f"[{name}] {be.describe}: parity vs torch over {a.parity_rows} rows: "
              f"max|d| tight={d_tight:.3e} padded={d_pad:.3e}")
        results["rows"][name] = {"describe": be.describe,
                                 "max_abs_diff": float(max(d_tight, d_pad)),
                                 "per_n": {}}
        bench_common._spin_ramp(bench_common.WARMUP_S)
        for n in n_list:
            x = padded[:n]
            be.time(x, 3)
            s = timed(lambda reps: be.time(x, reps))
            gf = 2.0 * params * n / s / 1e9
            print(f"[{name}] n={n:5d}  {1e6 * s:9.2f} us/call  {gf:7.1f} GFLOPS", flush=True)
            results["rows"][name]["per_n"][n] = {"us": 1e6 * s, "gflops": gf}
        thr1 = thread_count()
        print(f"[{name}] process threads before/after: {thr0}/{thr1}"
              + ("  <-- backend spawned threads" if thr1 > thr0 else ""))
        results["rows"][name]["threads"] = [thr0, thr1]

    if not a.no_torch:
        results["rows"]["torch-py"] = {"describe": f"python torch {torch.__version__} "
                                   f"threads={torch.get_num_threads()}", "per_n": {}}
        bench_common._spin_ramp(bench_common.WARMUP_S)
        for n in n_list:
            x = torch.from_numpy(rows[:n])

            def run(reps):
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(reps):
                        net(x)
                return time.perf_counter() - t0
            run(3)
            s = timed(run)
            gf = 2.0 * params * n / s / 1e9
            print(f"[torch-py] n={n:5d}  {1e6 * s:9.2f} us/call  {gf:7.1f} GFLOPS", flush=True)
            results["rows"]["torch-py"]["per_n"][n] = {"us": 1e6 * s, "gflops": gf}

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
