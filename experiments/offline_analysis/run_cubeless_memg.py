"""
Cubeless-money mEMG: equity counts gammons but no cube.
Each player self-plays N games (exported as money games, i.e. 0-point matches),
and gnubg scores every move at --plies. Chequer mEMG is the reliable metric; the
cube column is near-empty since play is cubeless. The gnubg judge and pipeline
are shared via memg_common; only the players and evaluation mode differ.

Players: prob5-{0,1}ply (prob5 evaluator), ours-cl3L-0ply (cubeless
reference), sage-{0,1}ply (Open Sage money), gnubg-ext-{0,1}ply (real
gnubg 1.08 binary over the external-player socket; 1-ply = full/unpruned
PAPER config). Requires the gnubg binary (GNUBG_CMD, default
/usr/games/gnubg).
"""
import os, sys, argparse, tempfile, time
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "c_engine"))
import multiprocessing as mp

from gnubg_eval import play_and_record, export_mat, _avg_rates, GNUBG_CMD
from stats import bootstrap_ci
from memg_common import _safe_gnubg_worker, append_pergame, combine_dir

# PROB5_MODEL must be an env var, not a monkey-patch: the play pool uses the
# "spawn" context, so workers re-import this module and would see the default.
PROB5 = os.environ.get(
    "PROB5_MODEL",
    # Prod prob5 (4L256, ac95a983); PROB5_MODEL re-measures any other.
    os.path.join(_REPO, "best_models", "cubeless_prob5_512_512_256_256.pt"))
CL3L  = os.path.join(_REPO, "best_models", "cubeless_money_512_512_256.pt")
PLAYERS = ["prob5-0ply", "prob5-1ply", "gnubg-ext-0ply", "gnubg-ext-1ply",
           "sage-0ply", "sage-1ply"]
# ours-cl3L-0ply (3L cubeless reference) is also available via --players but not in 'all'.


def _build_agent(player):
    if player == "prob5-0ply":
        from play_models import _load_agent
        return _load_agent(PROB5)
    if player == "prob5-1ply":
        from play_models import _load_agent
        return _load_agent(PROB5, oneply=True)
    if player == "ours-cl3L-0ply":
        from play_models import _load_agent
        return _load_agent(CL3L)
    # Native Open Sage (best_move_index money selection) — the SAME adapter
    # the H-H runs use (--side2 opensage), so mEMG and H-H measure identical
    # play. SAGE_UNPRUNED honored via play_models._sage_filter_kw, as there.
    # sage_plies is 1-based: 1 = Open Sage 0-ply, 2 = Open Sage 1-ply.
    if player == "sage-0ply":
        from play_models import _sage_filter_kw
        from experiments.opensage_agents import OpenSageAgent
        return OpenSageAgent(sage_plies=1, mode="cubeless-money",
                             **_sage_filter_kw())
    if player == "sage-1ply":
        from play_models import _sage_filter_kw
        from experiments.opensage_agents import OpenSageAgent
        return OpenSageAgent(sage_plies=2, mode="cubeless-money",
                             **_sage_filter_kw())
    if player == "gnubg-ext-0ply":
        from gnubg_external import GnubgExternalAgent
        return GnubgExternalAgent(plies=0, cubeless=True)
    if player == "gnubg-ext-1ply":
        from gnubg_external import GnubgExternalAgent
        return GnubgExternalAgent(plies=1, cubeless=True, full=True)
    raise ValueError(player)


def _play_worker(args):
    player, n, work_dir, start_idx = args
    os.environ["OMP_NUM_THREADS"] = "1"
    agent = _build_agent(player)
    paths = []
    for i in range(n):
        gid = start_idx + i
        rec = play_and_record(agent, agent)                  # self-play
        mat = export_mat(rec, game_id=gid, money_game=True)  # 0-point match = cubeless money
        p = os.path.join(work_dir, f"game_{gid}.mat")
        with open(p, "w") as f:
            f.write(mat)
        paths.append(p)
    return paths


def run_player(player, games, play_workers, gnubg_workers, plies, chunk, verbose=True):
    work_dir = tempfile.mkdtemp(prefix=f"run_cubeless_memg_{player}_")
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
    t_play = time.perf_counter() - t0
    if verbose:
        print(f"  [{player}] generated {len(mat_files)} games in {t_play:.0f}s "
              f"({len(mat_files)/max(t_play,1e-9):.1f}/s)", flush=True)

    t1 = time.perf_counter()
    chunks = []
    for i in range(0, len(mat_files), chunk):
        cd = os.path.join(work_dir, f"chunk_{len(chunks)}"); os.makedirs(cd, exist_ok=True)
        chunks.append((mat_files[i:i + chunk], cd, GNUBG_CMD, len(chunks), plies))
    ctxf = mp.get_context("fork")
    with ctxf.Pool(min(gnubg_workers, len(chunks))) as pool:
        cr = pool.map(_safe_gnubg_worker, chunks)
    overall = [er for _, _, ovr in cr for er in ovr]
    chequer = [er for chk, _, _ in cr for er in chk]
    t_an = time.perf_counter() - t1

    _, _, ovr_avg = _avg_rates(overall)
    _, _, chk_avg = _avg_rates(chequer)
    ovr_per = [(w + b) / 2 for w, b in overall] if overall else []
    chk_per = [(w + b) / 2 for w, b in chequer] if chequer else []
    ovr_ci = bootstrap_ci(ovr_per, n_boot=100_000) if ovr_per else (0, 0, 1)
    chk_ci = bootstrap_ci(chk_per, n_boot=100_000) if chk_per else (0, 0, 1)
    if verbose:
        print(f"  [{player}] analyzed {len(chequer)}/{len(mat_files)} at {plies}-ply "
              f"in {t_an:.0f}s", flush=True)
    return {
        "player": player, "n": len(chequer),
        "overall": ovr_avg, "overall_ci": (ovr_ci[0], ovr_ci[1]),
        "chequer": chk_avg, "chequer_ci": (chk_ci[0], chk_ci[1]),
        "overall_pergame": ovr_per, "chequer_pergame": chk_per,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--plies", type=int, default=4)   # 4-ply judge, matching the other scorers + README
    ap.add_argument("--play-workers", type=int, default=32)
    ap.add_argument("--gnubg-workers", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--players", default="all",
                    help="comma list of " + ",".join(PLAYERS) + " or 'all'")
    ap.add_argument("--save-pergame", default=None)
    ap.add_argument("--combine", default=None)
    ap.add_argument("--cubeful-eval", action="store_true",
                    help="Score chequer play with CUBEFUL money equity (gnubg default). "
                         "Default is cubeless-money (gammons count, no cube) — the objective "
                         "our cubeless models train on. See gnubg_eval._write_gnubg_script.")
    args = ap.parse_args()
    players = PLAYERS if args.players == "all" else args.players.split(",")

    # Cubeless-money chequer scoring unless --cubeful-eval. Set BEFORE the gnubg
    # worker pool spawns so workers inherit it; read in gnubg_eval._write_gnubg_script.
    os.environ["GNUBG_CUBELESS_EVAL"] = "0" if args.cubeful_eval else "1"

    # See run_cubeful_memg: a one-sided-only gnubg is an unrecorded config
    # change. Cheap to check, impossible to notice afterwards.
    if not args.combine:
        from gnubg_eval import assert_gnubg_bearoff
        assert_gnubg_bearoff(GNUBG_CMD)

    if args.combine:
        combine_dir(args.combine, players)
        return

    mode = "cubeful" if args.cubeful_eval else "cubeless"
    print(f"### {mode}-money {args.plies}-ply mEMG | {args.games} games/player "
          f"| players: {','.join(players)} ###", flush=True)
    # Always name the model being measured: a typo'd env var name (or a
    # non-exported var) silently falls back to the default, and the run
    # would otherwise complete with results attributed to the candidate.
    override = "PROB5_MODEL override" if os.environ.get("PROB5_MODEL") \
        else "default; set PROB5_MODEL to override"
    from memg_common import file_md5
    print(f"### prob5 model: {PROB5} (md5 {file_md5(PROB5)}; {override}) ###",
          flush=True)
    rows = []
    for p in players:
        print(f"\n--- {p} ---", flush=True)
        r = run_player(p, args.games, args.play_workers,
                       args.gnubg_workers, args.plies, args.chunk)
        rows.append(r)
        if args.save_pergame:
            path = append_pergame(
                args.save_pergame, p, args.plies,
                {"overall": r["overall_pergame"], "chequer": r["chequer_pergame"]})
            print(f"  [{p}] per-game appended -> {path}", flush=True)

    print(f"\n{'='*72}\n### Cubeless-money {args.plies}-ply mEMG SUMMARY ({args.games} games/player) ###")
    print(f"{'player':<16} {'n':>5}  {'chequer mEMG [95% CI]'}")
    for r in rows:
        print(f"{r['player']:<16} {r['n']:>5}  "
              f"{r['chequer']:.2f} [{r['chequer_ci'][0]:.2f}, {r['chequer_ci'][1]:.2f}]")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
