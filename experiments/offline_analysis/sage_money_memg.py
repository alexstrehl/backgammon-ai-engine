"""Open Sage cubeful-money mEMG via gnubg N-ply analysis.

Scores a directory of Open Sage money self-play transcripts (XG .txt, both sides
Open Sage, Jacoby + Beaver on) with gnubg at --plies and reports chequer, cube,
and overall mEMG with bootstrap 95% CIs over the per-decision errors of both
sides. The transcripts come from Open Sage's own generator (see
Sage_mEMG_calculation.md for the command and pinned bgsage commit); this script
needs only a gnubg binary, not bgsage.

gnubg's `show statistics match` prints three "Error rate ... mEMG" lines per
game, in order chequer, cube, overall, each a (player1, player2) pair. We reduce
each game to its per-game mean (p1+p2)/2 and bootstrap over those (n = games
scored), matching run_cubeful_memg.py / run_cubeless_memg.py for comparable CIs.

Usage:
  python sage_money_memg.py --src <dir-of-.txt> --plies 4 --workers 30
"""
import os, sys, glob, re, statistics as st, argparse, json
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from stats import bootstrap_ci   # shared per-game-mean bootstrap (seed 42, n=10000)
from gnubg_eval import run_gnubg_analysis

GNUBG = os.environ.get("GNUBG_CMD", "/usr/games/gnubg")


def clean(src_dir, work_clean):
    """Strip `;`-comment lines (gnubg's `import mat` rejects them) and copy each
    transcript to <work>/clean/<name>.mat."""
    os.makedirs(work_clean, exist_ok=True)
    out = []
    for f in sorted(glob.glob(os.path.join(src_dir, "*.txt"))):
        b = os.path.basename(f)[:-4]
        p = os.path.join(work_clean, f"{b}.mat")
        with open(f) as fh, open(p, "w") as o:
            for line in fh:
                if not line.lstrip().startswith(";"):
                    o.write(line)
        out.append(p)
    return out


def _opens_with_doublet(mat_path):
    """True if the game's opening roll is a doublet. bgsage's self-play generator
    does NOT enforce the no-opening-doublet rule (a real game can't start with a
    doublet), so ~1/6 of its games open illegally. The opening roll is exogenous
    (random, independent of play), so dropping these is an unbiased down-sample."""
    with open(mat_path) as fh:
        for line in fh:
            m = re.match(r"\s*1\)\s+([1-6])([1-6]):", line)
            if m:
                return m.group(1) == m.group(2)
    return False


def analyze_worker(args):
    """Score one chunk of .mat files with the shared gnubg judge
    (gnubg_eval.run_gnubg_analysis: script writer, section-keyed parse,
    n/a + >CAP filtering). Returns per-game (white+black)/2 means."""
    idx, chunk, plies, work = args
    wd = os.path.join(work, f"ana_{plies}_{idx}")
    os.makedirs(wd, exist_ok=True)
    cheq, cube, over = run_gnubg_analysis(
        chunk, wd, gnubg_cmd=GNUBG, verbose=False, plies=plies,
        extra_commands=["set threads 1"])
    return {"chequer": [(w + b) / 2 for w, b in cheq],
            "cube":    [(w + b) / 2 for w, b in cube],
            "overall": [(w + b) / 2 for w, b in over],
            "ngames":  len(cheq), "dropped": len(chunk) - len(cheq)}


def chunks(lst, k):
    """Split lst into min(k, len) contiguous chunks of near-equal size.
    (ceil-based slicing under-fills workers: 31 files / k=30 gave 16 chunks
    of 2, doubling wall time; this yields 30 chunks of 1-2.)"""
    n = min(k, len(lst))
    if n == 0:
        return []
    base, extra = divmod(len(lst), n)
    out, start = [], 0
    for i in range(n):
        sz = base + (1 if i < extra else 0)
        out.append(lst[start:start + sz])
        start += sz
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of Sage money .txt transcripts")
    ap.add_argument("--plies", type=int, default=4)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="Open Sage 0-ply (1P)")
    ap.add_argument("--work", default="/tmp/sage_money_ana")
    ap.add_argument("--out", default="", help="result .txt path (default: <work>/result_<plies>ply.txt)")
    a = ap.parse_args()

    files = clean(a.src, os.path.join(a.work, "clean"))
    # Drop games whose opening roll is a doublet (bgsage's generator does not enforce
    # the no-opening-doublet rule). The opening roll is exogenous, so this is an
    # unbiased down-sample.
    before = len(files)
    files = [f for f in files if not _opens_with_doublet(f)]
    print(f"legal-opening filter: dropped {before - len(files)}/{before} "
          f"doublet-opening games (unbiased down-sample)", flush=True)
    if a.limit:
        files = files[:a.limit]
    print(f"src={a.src} games={len(files)} plies={a.plies} workers={a.workers}", flush=True)

    jobs = [(i, ch, a.plies, a.work) for i, ch in enumerate(chunks(files, a.workers))]
    with mp.Pool(a.workers) as pool:
        out = pool.map(analyze_worker, jobs)

    ng = sum(o.get("ngames", 0) for o in out)
    nd = sum(o.get("dropped", 0) for o in out)
    print(f"games scored={ng}  dropped(n/a or >CAP)={nd}", flush=True)

    raw = {kind: [r for o in out for r in o[kind]]
           for kind in ("chequer", "cube", "overall")}
    rawpath = os.path.join(a.work, f"raw_{a.plies}ply.json")
    with open(rawpath, "w") as f:
        json.dump(raw, f)

    lines = [f"=== {a.label} MONEY self-play, gnubg {a.plies}-ply mEMG (both sides) ===",
             f"games={len(files)}  src={a.src}"]
    for kind in ("chequer", "cube", "overall"):
        v = raw[kind]
        m = st.mean(v) if v else 0.0
        ci = bootstrap_ci(v, n_boot=100_000) if v else (0.0, 0.0)
        lo, hi = ci[0], ci[1]
        lines.append(f"{a.label} [{kind:<7}] {m:.3f}  95%CI [{lo:.3f},{hi:.3f}]  (n={len(v)})")
    lines.append(f"raw samples saved -> {rawpath}")
    msg = "\n".join(lines)
    print(msg, flush=True)
    outpath = a.out or os.path.join(a.work, f"result_{a.plies}ply.txt")
    with open(outpath, "w") as f:
        f.write(msg + "\n")


if __name__ == "__main__":
    main()
