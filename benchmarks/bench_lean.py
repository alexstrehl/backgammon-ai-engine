"""Single-thread moves/s benchmark -- LEAN path (Python orchestration stripped).

Same evaluators as bench_agent.py, but with all per-move Python overhead
removed so the timing reflects (C move-gen + native eval) as closely as
possible:
  ours0  : inline bg_fast.get_legal_plays_encoded + one batched net() + argmin
  gnubg0 : gnubg_nn.best_move at 0-ply, boards pre-converted out of the loop
  gnubg1 : gnubg_nn.best_move at 1-ply, boards pre-converted out of the loop

Strictly single-thread. Paths resolved relative to this file.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
import argparse
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "c_engine"))

import torch
torch.set_grad_enabled(False)
torch.set_num_threads(1)

from backgammon_engine import BoardState, get_legal_plays
from model import TDNetwork
from agents import GnubgNNAgent
import gnubg_nn
import bg_fast

DEFAULT_MODEL = os.path.join(_HERE, "models", "dmp_512_512_256_256.pt")  # 4L DMP, cubeless
_b2g = GnubgNNAgent._board_to_gnubg


def gen_positions(n, seed=1234):
    rng = random.Random(seed)
    out = []
    s = BoardState.initial(); s.turn = 0
    d = (rng.randint(1, 6), rng.randint(1, 6))
    while len(out) < n:
        if s.is_game_over():
            s = BoardState.initial(); s.turn = 0
            d = (rng.randint(1, 6), rng.randint(1, 6)); continue
        pl = get_legal_plays(s, d)
        if pl:
            out.append((s, d)); _, s = pl[rng.randrange(len(pl))]
        else:
            s.turn ^= 1
        d = (rng.randint(1, 6), rng.randint(1, 6))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40000)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--config", required=True, choices=["ours0", "gnubg0", "gnubg1"])
ap.add_argument("--model", default=DEFAULT_MODEL)
args = ap.parse_args()
bf16 = os.environ.get("BF16", "0") == "1"
pos = gen_positions(args.n, args.seed)

if args.config == "ours0":
    net = TDNetwork.load(args.model).eval()
    if bf16:
        net = net.to(torch.bfloat16)
    dtype = torch.bfloat16 if bf16 else torch.float32
    def loop(items):
        for s, d in items:
            feats, _ = bg_fast.get_legal_plays_encoded(s, (d[0], d[1]))   # C move-gen + encode
            if len(feats):
                v = net(torch.from_numpy(feats).to(dtype))               # one batched forward
                int(torch.argmin(v))                                     # pick
else:
    plies = 0 if args.config == "gnubg0" else 1
    boards = [_b2g(s) for s, d in pos]            # convert once, out of the timed loop
    def loop(items):
        for i, (s, d) in enumerate(items):
            gnubg_nn.best_move(boards[i], d[0], d[1], plies)             # native C move-gen + eval

loop(pos[:20])                                    # warmup
t0 = time.perf_counter()
loop(pos)
dt = time.perf_counter() - t0
print(f"{args.config:<8} {args.n:>6} moves  {dt*1000/args.n:8.4f} ms/move   "
      f"{args.n/dt:10.1f} moves/s | LEAN path | bf16={bf16}", flush=True)
