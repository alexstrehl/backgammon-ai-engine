"""
4-ply cubeful-money mEMG, uniform treatment:
  ours-0ply / ours-1ply          (best 4L cubeful model, 0/1-ply)
  gnubg-ext-0ply / gnubg-ext-1ply (real gnubg 1.08 binary, external-player
                                   socket; 1-ply = full/unpruned PAPER config)
  sage-analyzer-1ply             (Open Sage cube-aware analyzer, XG 2P;
                                   SAGE_UNPRUNED=1 = paper config; not in
                                   --players all - request explicitly)
Requires the gnubg binary (GNUBG_CMD, default /usr/games/gnubg).

Each player self-plays cubeful-money games (Jacoby), exported as a money game
("0 point match"), then gnubg scores every chequer AND cube decision at --plies
(default 4). Reports chequer, cube, and overall mEMG with bootstrap 95% CIs.

Reuses gnubg_eval's cubeful play/export/analysis helpers. There is NO gnubg
timeout: slow chunks are only logged and every completed game is scored; a
chunk that hard-errors is dropped and logged, never aborting the run.
"""
import os, sys, argparse, tempfile, time
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "c_engine"))
import multiprocessing as mp

from gnubg_eval import (
    play_and_record_cubeful, export_mat, _avg_rates,
    _load_eval_agent, GNUBG_CMD,
)
from stats import bootstrap_ci

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # memg_common
from memg_common import _safe_gnubg_worker, append_pergame, load_pergame  # noqa: E402

CUBEFUL_MODEL = os.environ.get(  # env override must survive spawn-pool re-import
    "CUBEFUL_MODEL", "best_models/cubeful_money_512_512_256_256.pt")  # 4L cubeful
JACOBY = True
PLAYERS = ["ours-0ply", "ours-1ply", "gnubg-ext-0ply", "gnubg-ext-1ply"]


def _build_agent(player):
    if player == "ours-0ply":
        return _load_eval_agent(CUBEFUL_MODEL, oneply=False, cubeful=True)
    if player == "ours-1ply":
        return _load_eval_agent(CUBEFUL_MODEL, oneply=True, cubeful=True)
    if player == "sage-analyzer-1ply":
        # Open Sage cube-aware analyzer at 1-ply (XG 2P). SAGE_UNPRUNED=1
        # disables its move filter (nothing in opensage_agents reads the
        # env var — the kwargs must be passed here, as in play_models).
        from experiments.opensage_agents import OpenSageAnalyzerAgent
        kw = ({"filter_max_moves": 1000, "filter_threshold": 10.0}
              if os.environ.get("SAGE_UNPRUNED") == "1" else {})
        return OpenSageAnalyzerAgent(sage_plies=2, **kw)
    if player == "gnubg-ext-0ply":
        from gnubg_external import GnubgExternalAgent
        return GnubgExternalAgent(plies=0)
    if player == "gnubg-ext-1ply":
        from gnubg_external import GnubgExternalAgent
        return GnubgExternalAgent(plies=1, full=True)
    raise ValueError(player)


def _play_worker(args):
    player, n, work_dir, start_idx = args
    os.environ["OMP_NUM_THREADS"] = "1"
    agent = _build_agent(player)
    paths = []
    for i in range(n):
        gid = start_idx + i
        rec = play_and_record_cubeful(agent, agent, jacoby=JACOBY)   # self-play
        mat = export_mat(rec, game_id=gid, money_game=True)          # money game ("0 point match")
        p = os.path.join(work_dir, f"game_{gid}.mat")
        with open(p, "w") as f:
            f.write(mat)
        paths.append(p)
    return paths


def _stat(rates):
    _, _, avg = _avg_rates(rates)
    per = [(w + b) / 2 for w, b in rates] if rates else []
    ci = bootstrap_ci(per, n_boot=100_000) if per else (0, 0, 1)
    return avg, (ci[0], ci[1]), per


def run_player(player, games, play_workers, gnubg_workers, plies, chunk, verbose=True):
    work_dir = tempfile.mkdtemp(prefix=f"cube_memg_{player}_")
    t0 = time.perf_counter()
    jobs, start = [], 1
    for i in range(play_workers):
        n = games // play_workers + (1 if i < games % play_workers else 0)
        if n > 0:
            jobs.append((player, n, work_dir, start)); start += n
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(jobs)) as pool:
        res = pool.map(_play_worker, jobs)
    mat_files = [f for c in res for f in c]
    if verbose:
        print(f"  [{player}] generated {len(mat_files)} games in {time.perf_counter()-t0:.0f}s",
              flush=True)

    t1 = time.perf_counter()
    chunks = []
    for i in range(0, len(mat_files), chunk):
        cd = os.path.join(work_dir, f"chunk_{len(chunks)}"); os.makedirs(cd, exist_ok=True)
        chunks.append((mat_files[i:i + chunk], cd, GNUBG_CMD, len(chunks), plies))
    ctxf = mp.get_context("fork")
    with ctxf.Pool(min(gnubg_workers, len(chunks))) as pool:
        cr = pool.map(_safe_gnubg_worker, chunks)
    chequer = [er for chk, _, _ in cr for er in chk]
    cube = [er for _, cub, _ in cr for er in cub]
    overall = [er for _, _, ovr in cr for er in ovr]
    if verbose:
        print(f"  [{player}] analyzed {len(overall)}/{len(mat_files)} at {plies}-ply "
              f"in {time.perf_counter()-t1:.0f}s", flush=True)

    chk_avg, chk_ci, chk_per = _stat(chequer)
    cub_avg, cub_ci, cub_per = _stat(cube)
    ovr_avg, ovr_ci, ovr_per = _stat(overall)
    return {
        "player": player, "n": len(overall),
        "chequer": chk_avg, "chequer_ci": chk_ci,
        "cube": cub_avg, "cube_ci": cub_ci,
        "overall": ovr_avg, "overall_ci": ovr_ci,
        "chequer_pergame": chk_per, "cube_pergame": cub_per, "overall_pergame": ovr_per,
    }


def combine_dir(out_dir, players):
    print(f"\n{'='*84}\n### CUBEFUL mEMG COMBINED (pooled per-game, bootstrapped) ###")
    print(f"{'player':<12} {'n':>6}  {'chequer [95% CI]':<22} {'cube [95% CI]':<20} {'overall [95% CI]'}")
    for p in players:
        path = os.path.join(out_dir, f"{p}.jsonl")
        if not os.path.exists(path):
            print(f"{p:<12} {'--':>6}  (no data at {path})"); continue
        v = load_pergame(path, ("chequer", "cube", "overall"))
        chk, cub, ovr = v["chequer"], v["cube"], v["overall"]
        cc = bootstrap_ci(chk, n_boot=100_000); kc = bootstrap_ci(cub, n_boot=100_000); oc = bootstrap_ci(ovr, n_boot=100_000)
        print(f"{p:<12} {len(chk):>6}  "
              f"{sum(chk)/len(chk):.2f} [{cc[0]:.2f},{cc[1]:.2f}]   "
              f"{sum(cub)/len(cub):.2f} [{kc[0]:.2f},{kc[1]:.2f}]   "
              f"{sum(ovr)/len(ovr):.2f} [{oc[0]:.2f},{oc[1]:.2f}]")
    print(f"{'='*84}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--plies", type=int, default=4)
    ap.add_argument("--play-workers", type=int, default=48)
    ap.add_argument("--gnubg-workers", type=int, default=90)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--players", default="all")
    ap.add_argument("--save-pergame", default=None)
    ap.add_argument("--combine", default=None)
    args = ap.parse_args()
    players = PLAYERS if args.players == "all" else args.players.split(",")

    if args.combine:
        combine_dir(args.combine, players)
        return

    print(f"### CUBEFUL {args.plies}-ply mEMG | {args.games} self-play games/player "
          f"| model={CUBEFUL_MODEL} | jacoby={JACOBY} ###", flush=True)
    rows = []
    for p in players:
        print(f"\n--- {p} ---", flush=True)
        r = run_player(p, args.games, args.play_workers, args.gnubg_workers, args.plies, args.chunk)
        rows.append(r)
        if args.save_pergame:
            path = append_pergame(
                args.save_pergame, p, args.plies,
                {"chequer": r["chequer_pergame"], "cube": r["cube_pergame"],
                 "overall": r["overall_pergame"]})
            print(f"  [{p}] per-game appended -> {path}", flush=True)

    print(f"\n{'='*84}\n### CUBEFUL {args.plies}-ply mEMG SUMMARY ({args.games} games/player) ###")
    print(f"{'player':<12} {'n':>6}  {'chequer [95% CI]':<22} {'cube [95% CI]':<20} {'overall [95% CI]'}")
    for r in rows:
        print(f"{r['player']:<12} {r['n']:>6}  "
              f"{r['chequer']:.2f} [{r['chequer_ci'][0]:.2f},{r['chequer_ci'][1]:.2f}]   "
              f"{r['cube']:.2f} [{r['cube_ci'][0]:.2f},{r['cube_ci'][1]:.2f}]   "
              f"{r['overall']:.2f} [{r['overall_ci'][0]:.2f},{r['overall_ci'][1]:.2f}]")
    print(f"{'='*84}")


if __name__ == "__main__":
    main()
