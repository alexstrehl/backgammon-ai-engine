"""Single-thread moves/s benchmark -- AGENT path (realistic harness call).

Times the production decision path each engine actually uses:
  ours0  : TDAgent.choose_checker_action (C move-gen via bg_fast + batched PyTorch eval)
  gnubg0 : gnubg_nn.best_move at 0-ply  (native C move-gen + eval; 1 board convert/move)
  gnubg1 : gnubg_nn.best_move at 1-ply

All paths are strictly single-thread (OMP/MKL/torch = 1). Move generation
runs in C for every config; each evaluator then uses its fastest native path.

Paths are resolved relative to this file -- no hard-coded directories.
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
from td_agent import TDAgent
from model import TDNetwork
from agents import GnubgNNAgent
import gnubg_nn

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
    net = TDNetwork.load(args.model); net.eval()
    agent = TDAgent(net, oneply=False, bf16_inference=bf16)
    def decide(s, d):
        agent.choose_checker_action(s, d)                 # C move-gen (bg_fast) + batch eval
else:
    plies = 0 if args.config == "gnubg0" else 1
    def decide(s, d):
        board = _b2g(s)                                   # 1 conversion per move
        gnubg_nn.best_move(board, d[0], d[1], plies)      # native C move-gen + eval

for s, d in pos[:20]:
    decide(s, d)                                          # warmup
t0 = time.perf_counter()
for s, d in pos:
    decide(s, d)
dt = time.perf_counter() - t0
print(f"{args.config:<8} {args.n:>6} moves  {dt*1000/args.n:8.4f} ms/move   "
      f"{args.n/dt:10.1f} moves/s | AGENT path | bf16={bf16}", flush=True)
