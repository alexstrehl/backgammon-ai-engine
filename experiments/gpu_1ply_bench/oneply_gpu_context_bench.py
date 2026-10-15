#!/usr/bin/env python3
"""1-ply GPU collector benchmark: per-worker-context vs shared-context.

Runs train_batch.py (cubeful) and train_prob5.py (prob5) over a worker
ladder under both GPU collector topologies and reports, per cell:

  * steady-state ep/s from the "Round N/M" line timestamps, rounds 2..n
    (startup excluded);
  * peak GPU memory (nvidia-smi at 4 Hz) minus the idle baseline;
  * a least-squares fit VRAM = a + b*W per topology.

Per-worker-context runs go through the trainers' VRAM preflight, so a
ladder that overshoots the card is capped, not OOM'd. The table reports
the worker count actually used.

--validate checks that both topologies collect IDENTICAL data through
the real CLIs: BG_COLLECT_DUMP saves every round's (enc, tgt) pool, the
saved networks are compared bitwise, and a per-worker run is repeated
to establish the run-to-run determinism baseline.

Examples:

    python experiments/gpu_1ply_bench/oneply_gpu_context_bench.py
    python experiments/gpu_1ply_bench/oneply_gpu_context_bench.py --validate
    python experiments/gpu_1ply_bench/oneply_gpu_context_bench.py \\
        --workers 8,16,32,64 --rounds 8 --repeats 3 --workloads prob5
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from gpu_context import physical_cores  # noqa: E402

PY = sys.executable
ROUND_RE = re.compile(r"^Round\s+(\d+)/(\d+)")

WORKLOADS = {
    "cubeful": {
        "script": "train_batch.py",
        "model": "best_models/cubeful_money_512_512_256_256.pt",
        "fixed": ["--game-mode", "cubeful-money", "--oneply", "--device", "cuda",
                  "--log-every", "1"],
        # trainer.py: "Vectorized GPU collection: ON (N procs x B=..."
        "actual_w": re.compile(r"Vectorized GPU collection: ON \((\d+) procs"),
    },
    "prob5": {
        "script": "train_prob5.py",
        "model": "best_models/cubeless_prob5_512_512_256_256.pt",
        "fixed": ["--oneply", "--device", "cuda", "--log-every", "1"],
        # train_prob5.py: "vectorized collection: ON (B=.., workers=N, ..."
        "actual_w": re.compile(r"vectorized collection: ON \(.*?workers=(\d+)"),
    },
}
MODES = {
    "per-worker": "--vectorized-per-worker-context",
    "shared": "--vectorized-shared-context",
}


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def gpu_info(index: int):
    """(name, total MiB, driver) via nvidia-smi, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    r = _run(["nvidia-smi", "-i", str(index), "--query-gpu=name,memory.total,driver_version",
              "--format=csv,noheader,nounits"])
    if r.returncode != 0 or not r.stdout.strip():
        return None
    name, total, drv = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    return name, int(total), drv


def gpu_used_mib(index: int) -> int:
    r = _run(["nvidia-smi", "-i", str(index), "--query-gpu=memory.used",
              "--format=csv,noheader,nounits"])
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def torch_probe():
    """("<version> <cuda>", cuda_available) for the interpreter that runs the trainers."""
    r = _run([PY, "-c", "import torch;print(torch.__version__, torch.version.cuda, "
              "int(torch.cuda.is_available()))"], cwd=_REPO)
    parts = r.stdout.split()
    if r.returncode != 0 or len(parts) != 3:
        raise SystemExit(f"cannot import torch with {PY}:\n{r.stderr.strip()}")
    return f"{parts[0]} {parts[1]}", parts[2] == "1"


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


class VramSampler:
    """Polls nvidia-smi in a thread; .peak is the max used MiB seen."""

    def __init__(self, index: int, period: float = 0.25):
        self.index, self.period = index, period
        self.peak = -1
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            v = gpu_used_mib(self.index)
            if v > self.peak:
                self.peak = v
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=2)
        return False


class RunResult:
    def __init__(self):
        self.ok = False
        self.actual_w = None
        self.round_times = []      # (round_idx, monotonic time)
        self.n_rounds = 0
        self.steady_eps = None
        self.peak_vram = None
        self.wall = None
        self.notes = []
        self.tail = ""
        self.exit = None


def run_once(workload: str, mode: str, W: int, args, gpu_index: int,
             extra_env=None, extra_args=None, log_path=None) -> RunResult:
    wl = WORKLOADS[workload]
    epr = args.episodes_per_round
    cmd = [PY, "-u", wl["script"], "--resume", args.model[workload],
           *wl["fixed"], "--workers", str(W), "--vectorized-workers", str(W),
           "--num-episodes", str(epr * args.rounds),
           "--episodes-per-round", str(epr),
           "--lr", str(args.lr), "--seed", str(args.seed),
           "--torch-seed", str(args.seed), MODES[mode]]
    if args.extra_args:
        cmd += args.extra_args.split()
    if mode == "shared" and args.extra_shared_args:
        cmd += args.extra_shared_args.split()
    if extra_args:
        cmd += list(extra_args)
    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if extra_env:
        env.update(extra_env)
    res = RunResult()
    lines = []
    baseline = gpu_used_mib(gpu_index)
    t0 = time.monotonic()
    with VramSampler(gpu_index) as vs:
        p = subprocess.Popen(cmd, cwd=_REPO, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in p.stdout:
                now = time.monotonic()
                lines.append(line)
                m = ROUND_RE.match(line)
                if m:
                    res.round_times.append((int(m.group(1)), now))
                    res.n_rounds = int(m.group(2))
                m2 = wl["actual_w"].search(line)
                if m2:
                    res.actual_w = int(m2.group(1))
                s = line.strip()
                if s.startswith(("NOTE:", "WARN:", "ERROR")) and (
                        "capped" in s or "IGNORED" in s or "->" in s or "ERROR" in s):
                    res.notes.append(s)
            p.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            res.notes.append(f"TIMEOUT after {args.timeout}s")
    res.wall = time.monotonic() - t0
    res.exit = p.returncode
    res.tail = "".join(lines[-15:])
    if log_path:
        with open(log_path, "w") as f:
            f.writelines(lines)
    peak = vs.peak
    res.peak_vram = (peak - max(baseline, 0)) if peak >= 0 else None
    rt = res.round_times
    if p.returncode == 0 and len(rt) >= 3:
        # Episodes completed between the round-1 and round-n log lines,
        # over that interval: startup is excluded entirely.
        n = rt[-1][0] - rt[0][0]
        dt = rt[-1][1] - rt[0][1]
        res.steady_eps = (n * epr) / dt if dt > 0 else None
        res.ok = res.steady_eps is not None
    return res


def fit_line(xs, ys):
    """Least squares y = a + b x; returns (a, b) or None."""
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 2 or len({x for x, _ in pts}) < 2:
        return None
    X = np.array([p[0] for p in pts], dtype=float)
    Y = np.array([p[1] for p in pts], dtype=float)
    b, a = np.polyfit(X, Y, 1)
    return float(a), float(b)


def fmt(v, nd=1, unit=""):
    return "-" if v is None else f"{v:.{nd}f}{unit}"


def report(results, args, host_line, P):
    P(host_line)
    P(f"protocol: rounds={args.rounds} episodes/round={args.episodes_per_round} "
      f"repeats={args.repeats} lr={args.lr} seed={args.seed}; ep/s = steady state over "
      f"rounds 2..n (Round-line timestamps); VRAM = peak nvidia-smi used minus idle"
      + (f"; extra args: {args.extra_args!r}" if args.extra_args else "")
      + (f"; extra shared args: {args.extra_shared_args!r}" if args.extra_shared_args else ""))
    P()
    fits = {}
    for wl in args.workloads:
        P(f"== {wl} ({WORKLOADS[wl]['script']})")
        P(f"{'W req':>6} {'mode':>10} {'W used':>6} {'ep/s per repeat':>26} {'mean':>7} "
          f"{'spread':>7} {'peak VRAM MiB per repeat':>26} {'max':>7}  notes")
        for W in args.workers:
            for mode in args.modes:
                rs = [r for (w_, m_, W_, _), r in results.items()
                      if w_ == wl and m_ == mode and W_ == W]
                if not rs:
                    continue
                eps = [r.steady_eps for r in rs]
                vr = [r.peak_vram for r in rs]
                good = [e for e in eps if e is not None]
                mean = sum(good) / len(good) if good else None
                spread = ((max(good) - min(good)) / mean * 100) if len(good) > 1 and mean else None
                vmax = max([v for v in vr if v is not None], default=None)
                used = rs[0].actual_w
                notes = "; ".join(sorted({n for r in rs for n in r.notes}))[:90]
                P(f"{W:>6} {mode:>10} {fmt(used, 0):>6} "
                  f"{'/'.join(fmt(e) for e in eps):>26} {fmt(mean):>7} "
                  f"{fmt(spread, 0, '%'):>7} {'/'.join(fmt(v, 0) for v in vr):>26} "
                  f"{fmt(vmax, 0):>7}  {notes}")
        for mode in args.modes:
            xs, ys = [], []
            for (w_, m_, W_, _), r in results.items():
                if w_ == wl and m_ == mode and r.peak_vram is not None and r.actual_w:
                    xs.append(r.actual_w)
                    ys.append(r.peak_vram)
            f = fit_line(xs, ys)
            if f and f[1] <= 0:
                # No growth with W: report flat at the largest peak seen
                # instead of extrapolating a negative slope.
                f = (float(max(ys)), 0.0)
            fits[(wl, mode)] = f
            if f:
                a, b = f
                if b > 0:
                    P(f"   fit {mode:>10}: peak VRAM ~= {a:.0f} MiB + {b:.0f} MiB x W   "
                      f"(over W used = {sorted(set(xs))})")
                else:
                    P(f"   fit {mode:>10}: peak VRAM flat, <= {a:.0f} MiB at every W   "
                      f"(over W used = {sorted(set(xs))}; slope not > 0)")
        preds = [w for w in (16, 32, 48, 64, 96, 128) if w > max(args.workers)]
        if preds:
            for mode in args.modes:
                f = fits.get((wl, mode))
                if f:
                    a, b = f
                    P(f"   extrapolated {mode:>10}: " + "  ".join(
                        f"W={w}: {(a + b * w) / 1024:.1f} GB" for w in preds)
                      + ("" if b > 0 else
                         "  (flat; add ~0.5-1 GB per extra eval-server process, "
                         "one per 16 workers)"))
        P()
    return fits


def _load_rounds(d):
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".npz"):
            z = np.load(os.path.join(d, fn))
            out[fn] = (z["enc"], z["tgt"])
    return out


def _load_sd(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck.get("state_dict") or ck.get("model_state_dict") or ck
    return {k: v.numpy() for k, v in sd.items()}


def _compare(label, a, b, P):
    """Elementwise comparison of two dicts of arrays; prints one line."""
    if a.keys() != b.keys():
        P(f"   {label}: KEY MISMATCH {sorted(set(a) ^ set(b))[:5]}")
        return False
    worst = 0.0
    shape_mismatch = []
    for k in a:
        if a[k].shape != b[k].shape:
            shape_mismatch.append((k, a[k].shape, b[k].shape))
            continue
        if a[k].size:
            worst = max(worst, float(np.abs(a[k].astype(np.float64)
                                            - b[k].astype(np.float64)).max()))
    if shape_mismatch:
        P(f"   {label}: SHAPE MISMATCH (trajectories diverged) e.g. {shape_mismatch[0]}")
        return False
    P(f"   {label}: max|diff| = {worst:.3g} over {len(a)} arrays "
      f"-> {'BITWISE IDENTICAL' if worst == 0.0 else 'DIFFERS'}")
    return worst == 0.0


def validate(args, gpu_index, P):
    P("== validation: collected data and trained weights, per-worker vs shared")
    P(f"   protocol: W={args.validate_workers} rounds={args.validate_rounds} "
      f"episodes/round={args.validate_episodes_per_round} lr={args.validate_lr} "
      f"seed={args.seed}")
    all_ok = True
    tmp = tempfile.mkdtemp(prefix="oneply_ctx_validate_")
    try:
        for wl in args.workloads:
            P(f"-- {wl}")
            runs = {}
            for tag, mode in (("A", "per-worker"), ("B", "shared"), ("A2", "per-worker")):
                d = os.path.join(tmp, f"{wl}_{tag}")
                save = os.path.join(tmp, f"{wl}_{tag}.pt")
                va = argparse.Namespace(**vars(args))
                va.rounds = args.validate_rounds
                va.episodes_per_round = args.validate_episodes_per_round
                va.lr = args.validate_lr
                r = run_once(wl, mode, args.validate_workers, va, gpu_index,
                             extra_env={"BG_COLLECT_DUMP": d},
                             extra_args=["--save", save],
                             log_path=os.path.join(tmp, f"{wl}_{tag}.log"))
                if r.exit != 0:
                    P(f"   run {tag} ({mode}) FAILED exit={r.exit}\n{r.tail}")
                    all_ok = False
                    continue
                runs[tag] = (d, save)
            if len(runs) < 3:
                continue
            ra, rb, ra2 = _load_rounds(runs["A"][0]), _load_rounds(runs["B"][0]), \
                _load_rounds(runs["A2"][0])

            def flat(rr):
                return {f"{k}/{n}": v for k, (e, t) in rr.items() for n, v in (("enc", e), ("tgt", t))}
            base_ok = _compare("determinism baseline (per-worker run 1 vs run 2), collected pools",
                               flat(ra), flat(ra2), P)
            ok1 = _compare("per-worker vs shared, collected pools (every round)",
                           flat(ra), flat(rb), P)
            sa, sb, sa2 = _load_sd(runs["A"][1]), _load_sd(runs["B"][1]), _load_sd(runs["A2"][1])
            # The weight comparison is vacuous unless training moved the
            # weights away from the resumed checkpoint.
            s0 = _load_sd(os.path.join(_REPO, args.model[wl]))
            moved = max(float(np.abs(sa[k].astype(np.float64) - s0[k].astype(np.float64)).max())
                        for k in sa if k in s0 and sa[k].shape == s0[k].shape)
            P(f"   weights moved from the resumed checkpoint by max|d| = {moved:.3g} "
              f"({'ok' if moved > 0 else 'NOT MOVED -- raise --validate-lr'})")
            base_w = _compare("determinism baseline, final weights", sa, sa2, P)
            ok2 = _compare("per-worker vs shared, final weights", sa, sb, P)
            all_ok = all_ok and moved > 0
            if not base_ok or not base_w:
                P("   NOTE: the per-worker path is not run-to-run deterministic on this "
                  "host; judge the topology comparison against that baseline.")
            all_ok = all_ok and ok1 and (ok2 or not base_w)
        P(f"== validation {'PASSED' if all_ok else 'FAILED'}")
    finally:
        if args.keep_validate_dir:
            P(f"   validation artefacts kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return all_ok


def default_ladder(phys: int):
    cand = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192]
    ladder = [c for c in cand if c <= phys]
    if phys not in ladder:
        ladder.append(phys)
    return sorted(set(ladder)) or [max(2, phys)]


class _Formatter(argparse.RawDescriptionHelpFormatter,
                 argparse.ArgumentDefaultsHelpFormatter):
    pass


def main():
    phys = physical_cores()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=_Formatter)
    ap.add_argument("--workers", default=",".join(map(str, default_ladder(phys))),
                    help="comma-separated worker ladder (default from physical cores)")
    ap.add_argument("--workloads", default="cubeful,prob5",
                    help="comma-separated subset of cubeful,prob5")
    ap.add_argument("--modes", default="per-worker,shared",
                    help="comma-separated subset of per-worker,shared")
    ap.add_argument("--rounds", type=int, default=6,
                    help="rounds per run; ep/s uses rounds 2..n")
    ap.add_argument("--episodes-per-round", type=int, default=500,
                    help="episodes per round of every run")
    ap.add_argument("--repeats", type=int, default=2,
                    help="repeats per cell, interleaved across cells")
    ap.add_argument("--lr", type=float, default=1e-8,
                    help="net effectively frozen so arms cannot diverge through training")
    ap.add_argument("--seed", type=int, default=12345, help="--seed and --torch-seed of every run")
    ap.add_argument("--gpu-index", type=int, default=0, help="nvidia-smi GPU index to sample")
    ap.add_argument("--timeout", type=int, default=1800, help="per run, seconds")
    ap.add_argument("--extra-args", default="",
                    help="extra CLI args appended to EVERY run, e.g. "
                         "'--episodes-per-round 1000' (quoted, space-separated)")
    ap.add_argument("--extra-shared-args", default="",
                    help="extra CLI args appended to shared-context runs only, e.g. "
                         "'--gpu-eval-servers 2'")
    ap.add_argument("--model-cubeful", default=WORKLOADS["cubeful"]["model"],
                    help="checkpoint for the cubeful workload, relative to the repo root")
    ap.add_argument("--model-prob5", default=WORKLOADS["prob5"]["model"],
                    help="checkpoint for the prob5 workload, relative to the repo root")
    ap.add_argument("--out", default=None,
                    help="results file; unset = results/oneply_gpu_context_<host>_<date>.txt "
                         "next to this script")
    ap.add_argument("--no-out", action="store_true", help="do not write a results file")
    ap.add_argument("--log-dir", default=None, help="keep every run's stdout here")
    ap.add_argument("--validate", action="store_true",
                    help="also check per-worker == shared on collected data and weights")
    ap.add_argument("--validate-only", action="store_true",
                    help="run the validation and skip the ladder")
    ap.add_argument("--validate-workers", type=int, default=4, help="W for validation runs")
    ap.add_argument("--validate-rounds", type=int, default=3, help="rounds per validation run")
    ap.add_argument("--validate-episodes-per-round", type=int, default=100,
                    help="episodes per round of each validation run")
    ap.add_argument("--validate-lr", type=float, default=1e-4,
                    help="large enough that the weights actually move")
    ap.add_argument("--keep-validate-dir", action="store_true",
                    help="keep the validation pools, logs and checkpoints")
    args = ap.parse_args()

    args.workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    try:
        args.workers = [int(x) for x in args.workers.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"--workers must be comma-separated integers, got {args.workers!r}")
    if not args.workers or min(args.workers) < 2:
        raise SystemExit("--workers needs at least one entry, each >= 2")
    args.model = {"cubeful": args.model_cubeful, "prob5": args.model_prob5}
    for wl in args.workloads:
        if wl not in WORKLOADS:
            raise SystemExit(f"unknown workload {wl!r} (choose from {','.join(WORKLOADS)})")
        if not os.path.exists(os.path.join(_REPO, args.model[wl])):
            raise SystemExit(f"model not found: {args.model[wl]} (relative to {_REPO})")
    for m in args.modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m!r} (choose from {','.join(MODES)})")
    if not args.workloads or not args.modes:
        raise SystemExit("--workloads and --modes must not be empty")

    if not shutil.which("nvidia-smi"):
        raise SystemExit("nvidia-smi not found on PATH; VRAM sampling needs it")
    gi = gpu_info(args.gpu_index)
    if not gi:
        raise SystemExit(f"nvidia-smi reports no GPU at index {args.gpu_index}")
    torch_ver, cuda_ok = torch_probe()
    if not cuda_ok:
        raise SystemExit(f"torch {torch_ver} in {PY} has no CUDA device "
                         "(check CUDA_VISIBLE_DEVICES and the torch build)")
    host_line = (f"host: {socket.gethostname()} | {cpu_model()} "
                 f"| {phys} physical cores ({os.cpu_count()} logical) | "
                 f"GPU {gi[0]} {gi[1]} MiB driver {gi[2]}"
                 f" | {torch_ver} | {_dt.date.today()}")
    print(host_line)
    print(f"worker ladder: {args.workers}  workloads: {args.workloads}  modes: {args.modes}")
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    out_lines = []
    P = lambda s="": (print(s), out_lines.append(s))  # noqa: E731

    if args.validate or args.validate_only:
        validate(args, args.gpu_index, P)
        if args.validate_only:
            _write_out(args, out_lines)
            return

    results = {}
    total = len(args.workloads) * len(args.workers) * len(args.modes) * args.repeats
    k = 0
    # Repeat-outer, config-inner, so drift affects all cells alike.
    for rep in range(args.repeats):
        for wl in args.workloads:
            for W in args.workers:
                for mode in args.modes:
                    k += 1
                    lp = (os.path.join(args.log_dir, f"{wl}_{mode}_W{W}_r{rep}.log")
                          if args.log_dir else None)
                    print(f"[{k}/{total}] {wl} {mode} W={W} rep={rep} ...", end="", flush=True)
                    r = run_once(wl, mode, W, args, args.gpu_index, log_path=lp)
                    results[(wl, mode, W, rep)] = r
                    print(f" ep/s={fmt(r.steady_eps)} peakVRAM={fmt(r.peak_vram, 0)} MiB "
                          f"Wused={r.actual_w} wall={r.wall:.0f}s"
                          + (f" exit={r.exit}" if r.exit else "")
                          + (f" [{'; '.join(r.notes)[:80]}]" if r.notes else ""))
                    if r.exit:
                        print(r.tail)
    print()
    report(results, args, host_line, P)
    _write_out(args, out_lines)


def _write_out(args, out_lines):
    if args.no_out:
        return
    path = args.out or os.path.join(
        _HERE, "results",
        f"oneply_gpu_context_{socket.gethostname()}_{_dt.date.today():%Y%m%d}.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"results written to {path}")


if __name__ == "__main__":
    main()
