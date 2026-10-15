#!/usr/bin/env python3
"""Parity level 3 (SPEC.md): one training round from the Python collector
and from --c-collect, same --seed and worker count, compared from the
trainer's own dump (BG_COLLECT_DUMP: round_NNNN.npz with the pool's enc,
tgt and per-worker row counts).

    python experiments/c_collect/round_parity.py                   # cubeful 0-ply
    python experiments/c_collect/round_parity.py --oneply --episodes 1000
    python experiments/c_collect/round_parity.py --game-mode dmp --rounds 2

Both arms run train_batch.py on the mode's 4L prod net with --lr 1e-9
(frozen: round 2+ collect on the same weights in both arms), the given
--workers / --episodes-per-round / --seed, --device cpu by default.
Every worker gets the same seed in both arms (the run RNG draws them in
the same order), so its rows agree until the first near-tie decision C
takes differently (tests/test_c_collect.py: C and torch scores within
1e-5); after that its stream is a different, equally valid episode
sequence.

Report per round: workers whose rows agree to the end; per worker, the
aligned prefix (rows before the first encoding that differs); targets on
the aligned rows against the level 1 bound (1e-5 at 0-ply, 1e-4 at 1-ply,
relative x10 above 1: test_c_api.close); target mean / std and the
per-feature means of each arm's whole pool. --check-ties replays every
worker whose streams part through tests/test_c_collect.py and prints
the near-tie decision and its margin. Exit 1 if an aligned row's target
is out of bound, a parting is not a near-tie, or a dump is missing,
else 0.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS = {
    "cubeful-money": "best_models/cubeful_money_512_512_256_256.pt",
    "cubeless-money": "best_models/cubeless_money_512_512_256_256.pt",
    "dmp": "best_models/dmp_512_512_256_256.pt",
}
TOL0, TOL1 = 1e-5, 1e-4      # tests/test_c_api.py


def out_of_bound(a, b, tol):
    """Elementwise not test_c_api.close(a, b, tol): |a-b| <= tol, relative
    tol*10 where |b| > 1."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    lim = np.where(np.abs(b) > 1.0, np.maximum(tol, tol * 10 * np.abs(b)), tol)
    return d > lim


def load_dump(d):
    """[(enc, tgt, worker_rows, worker_seeds)] per round, in round order
    (worker_seeds empty when the dump has none)."""
    out = []
    for f in sorted(glob.glob(os.path.join(d, "round_*.npz"))):
        z = np.load(f)
        if "worker_rows" not in z:
            raise SystemExit(f"{f}: no worker_rows (dump from a trainer before P4)")
        seeds = z["worker_seeds"] if "worker_seeds" in z else np.zeros(0, np.int64)
        out.append((z["enc"], z["tgt"], z["worker_rows"], seeds))
    return out


def split(enc, tgt, rows):
    edges = np.concatenate([[0], np.cumsum(rows)])
    return [(enc[a:b], tgt[a:b]) for a, b in zip(edges[:-1], edges[1:])]


def compare(py_rounds, c_rounds, tol):
    """(lines, ok, parted): the report, whether every aligned target is in
    bound (and both arms drew the same worker seeds), and the
    (round index, worker) pairs whose streams part."""
    lines, ok, parted = [], True, []
    if len(py_rounds) != len(c_rounds) or not py_rounds:
        return [f"ERROR: {len(py_rounds)} Python rounds, {len(c_rounds)} C rounds"], False, []
    tot_rows = tot_aligned = 0
    for r, ((ep, tp, wp, sp), (ec, tc, wc, sc)) in enumerate(zip(py_rounds, c_rounds), 1):
        if len(wp) != len(wc):
            lines.append(f"round {r}: ERROR {len(wp)} Python workers, {len(wc)} C workers")
            ok = False
            continue
        if not np.array_equal(sp, sc):
            lines.append(f"round {r}: ERROR the arms drew different worker seeds")
            ok = False
        agree, aligned, n_oob, max_d, divs = 0, 0, 0, 0.0, []
        for w, ((e1, t1), (e2, t2)) in enumerate(zip(split(ep, tp, wp), split(ec, tc, wc))):
            n = min(len(t1), len(t2))
            same = np.all(e1[:n] == e2[:n], axis=1) if n else np.zeros(0, bool)
            k = n if same.all() else int(np.argmin(same))
            aligned += k
            if k:
                bad = out_of_bound(t2[:k], t1[:k], tol)
                n_oob += int(bad.sum())
                max_d = max(max_d, float(np.max(np.abs(t2[:k].astype(np.float64) - t1[:k]))))
            if k == n and len(t1) == len(t2):
                agree += 1
            else:
                divs.append((w, k, len(t1), len(t2)))
                parted.append((r - 1, w))
        tot_rows += len(tp)
        tot_aligned += aligned
        ok &= n_oob == 0
        lines.append(
            f"round {r}: {agree}/{len(wp)} workers agree to the end; rows Python "
            f"{len(tp)} C {len(tc)}; aligned {aligned} ({100 * aligned / max(len(tp), 1):.2f}% "
            f"of Python's); aligned targets max |d| {max_d:.3g}, out of bound {n_oob}")
        for w, k, n1, n2 in divs:
            lines.append(f"    worker {w}: streams part at row {k} (rows Python {n1}, C {n2})")
        lines.append(
            f"    target mean/std  Python {tp.mean():+.6f} / {tp.std():.6f}   "
            f"C {tc.mean():+.6f} / {tc.std():.6f}")
        fm = np.abs(ep.mean(axis=0, dtype=np.float64) - ec.mean(axis=0, dtype=np.float64))
        lines.append(
            f"    per-feature means: max |d| {fm.max():.3g} (feature {int(fm.argmax())}), "
            f"mean |d| {fm.mean():.3g} over {ep.shape[1]} features")
    lines.append(f"total: {tot_aligned}/{tot_rows} Python rows aligned "
                 f"({100 * tot_aligned / max(tot_rows, 1):.2f}%), bound {tol:g}: "
                 + ("OK" if ok else "OUT OF BOUND"))
    return lines, ok, parted


def check_ties(args, py_rounds, parted):
    """(lines, ok): each parted worker's stream replayed through
    tests/test_c_collect.compare_stream (Python collector vs
    Engine.collect on the worker's seed; it asserts that the first
    decision taken differently is a near-tie, C scores within 1e-5).
    Only rounds collected on the initial weights replay exactly: round 1,
    and round 2 when pipelined."""
    for p in (REPO, os.path.join(REPO, "tests")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import test_c_collect as TC
    import trainer as T
    kind = {"cubeful-money": "cubeful", "cubeless-money": "cubeless", "dmp": "dmp"}[args.game_mode]
    model = os.path.join(REPO, args.model or MODELS[args.game_mode])
    cell = TC.Cell(kind, args.oneply, args.jacoby, False, net=model)
    eng = TC.engine(args.backend, model)
    split = T._split_episodes(args.episodes, args.workers)
    lines, ok = [], True
    for r, w in parted:
        seeds = py_rounds[r][3]
        if r > (1 if args.pipeline else 0) or len(seeds) <= w:
            lines.append(f"round {r + 1} worker {w}: not replayed (trained weights or no seed)")
            continue
        t = TC.Tally()
        try:
            TC.compare_stream(eng, cell, int(seeds[w]), split[w], TC.DEFAULT_PLIES, t)
        except AssertionError as err:
            lines.append(f"round {r + 1} worker {w}: NOT A NEAR-TIE: {err}")
            ok = False
            continue
        lines.append(f"round {r + 1} worker {w}: " + t.line(f"seed {int(seeds[w])}")
                     .replace("[ccollect] ", ""))
    return lines, ok


def run_arm(args, dump, c_collect):
    cmd = [args.python, "-u", os.path.join(REPO, "train_batch.py"),
           "--game-mode", args.game_mode, "--resume", args.model or MODELS[args.game_mode],
           "--device", args.device, "--workers", str(args.workers),
           "--episodes-per-round", str(args.episodes),
           "--num-episodes", str(args.episodes * args.rounds),
           "--lr", "1e-9", "--seed", str(args.seed), "--torch-seed", str(args.seed),
           "--log-every", "1"]
    if args.oneply:
        cmd += ["--oneply"]
        if args.device.startswith("cuda") and not c_collect:
            cmd += ["--no-vectorized-collect"]
    if args.pipeline:
        cmd += ["--pipeline-collect"]
    if not args.jacoby:
        cmd += ["--no-jacoby"]
    if c_collect:
        cmd += ["--c-collect", "--c-collect-backend", args.backend]
    else:
        cmd += ["--no-c-collect"]       # the C collector is the default
    env = dict(os.environ, BG_COLLECT_DUMP=dump, OMP_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", BG_ROUND_TIMING="1")
    print(f"[{'C' if c_collect else 'Python'}] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    log = r.stdout + r.stderr
    with open(os.path.join(dump, "train.log"), "w") as f:
        f.write(log)
    for ln in log.splitlines():
        if ln.startswith(("Round", "C collection", "  NOTE", "Done in")):
            print("    " + ln, flush=True)
    if r.returncode:
        raise SystemExit(f"train_batch.py exited {r.returncode}:\n{log[-3000:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--game-mode", choices=sorted(MODELS), default="cubeful-money")
    ap.add_argument("--model", default=None, help="override the mode's prod net")
    ap.add_argument("--oneply", action="store_true")
    ap.add_argument("--episodes", type=int, default=2000, help="episodes per round")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--pipeline", action="store_true", help="--pipeline-collect in both arms")
    ap.add_argument("--jacoby", action=argparse.BooleanOptionalAction, default=True,
                    help="cubeful-money Jacoby rule (train_batch.py --[no-]jacoby)")
    ap.add_argument("--device", default="cpu", help="the master's training device")
    ap.add_argument("--backend", default="simd")
    ap.add_argument("--out", default=None,
                    help="dump dir (default: a fresh dir in the system temp dir)")
    ap.add_argument("--compare-only", nargs=2, metavar=("PY_DUMP", "C_DUMP"),
                    help="compare two existing dumps, run nothing")
    ap.add_argument("--check-ties", action="store_true",
                    help="replay each parted worker through tests/test_c_collect's "
                         "near-tie check (Python speed: ~250 episodes per worker)")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    tol = TOL1 if args.oneply else TOL0

    if args.compare_only:
        d_py, d_c = args.compare_only
    else:
        base = args.out
        if base is None:
            base = tempfile.mkdtemp(prefix="round_parity_")
        d_py, d_c = os.path.join(base, "python"), os.path.join(base, "c")
        os.makedirs(d_py, exist_ok=True)
        os.makedirs(d_c, exist_ok=True)
        print(f"round parity: {args.game_mode}{'' if args.jacoby else ' no-jacoby'} "
              f"{'1-ply' if args.oneply else '0-ply'}, "
              f"{args.rounds} x {args.episodes} episodes, {args.workers} workers, "
              f"seed {args.seed}{', pipelined' if args.pipeline else ''}; dumps in {base}")
        run_arm(args, d_py, c_collect=False)
        run_arm(args, d_c, c_collect=True)
    py_rounds = load_dump(d_py)
    lines, ok, parted = compare(py_rounds, load_dump(d_c), tol)
    print("\n".join(lines), flush=True)
    if args.check_ties and parted:
        tl, tok = check_ties(args, py_rounds, parted)
        print("near-tie check:\n  " + "\n  ".join(tl))
        ok &= tok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
