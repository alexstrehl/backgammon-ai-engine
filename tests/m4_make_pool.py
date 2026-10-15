"""Collect a small real match-play pool once, for the train-step bitwise
test (tests/test_train_step_bitwise.py) to replay.

The training collector settings on the CPU serial collector
(`trainer.collect_episode`, what `_collect_worker` runs): 1-ply targets,
0-ply greedy acting, MET `auto` grounding all three scopes (and the
encoders' `BG_MET_PATH`), `TRAINER_SCORE_CURRICULUM_P=0.0`. Episodes are
split over `--workers` spawn processes, worker k on seed `--seed + k`;
the file keeps every episode's rows in order, with its row count, so the
test can hand them out one `collect_episode` call at a time.

    TRAINER_SCORE_CURRICULUM_P=0.0 OMP_NUM_THREADS=1 \\
        python tests/m4_make_pool.py --out pool_3pt_200.npz

The test builds a 233-match pool this way (--episodes 233) in its temp
dir unless BG_TRAIN_STEP_POOL names a prebuilt one.
"""

import argparse
import hashlib
import multiprocessing as mp
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

NET = os.path.join(REPO, "best_models", "matchplay_1h64met_512_512_256_256.pt")
SCOPES = ("boundary", "cube", "terminal")


def _work(args):
    net_path, met_path, match_length, n_eps, seed = args
    os.environ["BG_MET_PATH"] = met_path
    import torch
    torch.set_num_threads(1)
    from met import MatchEquityTable
    from model import TDNetwork
    from modes import MatchplayMode
    from td_agent import TDAgent
    import trainer
    net = TDNetwork.load(net_path)
    agent = TDAgent(net, met=MatchEquityTable.from_xml(met_path),
                    met_scopes=frozenset(SCOPES))
    mode = MatchplayMode(match_length=match_length)
    rng = random.Random(seed)
    out = []
    for _ in range(n_eps):
        enc, tgt = trainer.collect_episode(agent, mode, rng, oneply=True)
        out.append((np.asarray(enc, dtype=np.float32),
                    np.asarray(tgt, dtype=np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--net", default=NET)
    ap.add_argument("--match-length", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4100)
    a = ap.parse_args()
    if os.environ.get("TRAINER_SCORE_CURRICULUM_P", "") != "0.0":
        sys.exit("set TRAINER_SCORE_CURRICULUM_P=0.0 (the training setting)")
    from met import resolve_met_path
    met_path = resolve_met_path("auto")
    per = [a.episodes // a.workers + (k < a.episodes % a.workers)
           for k in range(a.workers)]
    jobs = [(a.net, met_path, a.match_length, n, a.seed + k)
            for k, n in enumerate(per) if n]
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        eps = [e for chunk in pool.map(_work, jobs) for e in chunk]
    dt = time.time() - t0
    rows = np.array([len(t) for _, t in eps], dtype=np.int64)
    enc = np.concatenate([e for e, _ in eps if len(e)], axis=0)
    tgt = np.concatenate([t for _, t in eps if len(t)], axis=0)
    with open(a.net, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, enc=enc, tgt=tgt, ep_rows=rows,
             seeds=np.array([j[4] for j in jobs]), per_worker=np.array(per),
             net_md5=md5, met_path=met_path, match_length=a.match_length)
    print(f"{a.out}: {len(eps)} episodes, {len(tgt)} rows x {enc.shape[1]} "
          f"(rows/episode {rows.mean():.1f}, min {rows.min()}, max {rows.max()}), "
          f"net {md5[:8]}, {dt:.1f} s on {len(jobs)} workers")


if __name__ == "__main__":
    main()
