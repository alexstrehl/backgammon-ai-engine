"""Parity gate for the prob5 GPU eval server (`--gpu-eval-server`).

Routing a collection forward to the server must change WHERE the matmul
runs and nothing else. This drives real self-play states through
`_oneply_target_vec` twice -- once with `remote=None` (the CPU reference)
and once through a live `EvalServer` on cuda -- and compares the 1-ply
targets.

Bar: fp32 GPU and fp32 CPU GEMMs use different reduction orders, so
BIT-exactness is not achievable here (unlike the CPU-to-CPU generator
parity in tests/test_vectorized_cubeless.py, where it is the correct
bar). What must hold is agreement to fp32 rounding -- max|d| < 1e-5 on
probabilities in [0,1] -- and, critically, IDENTICAL greedy move
selection, since a flipped argmin would fork the trajectory and make
any A/B comparison meaningless.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing as mp
import random

import numpy as np
import torch

import train_prob5 as P5
from model import ProbNetwork, prob5_postprocess, prob5_to_equity
from backgammon_engine import opening_roll, switch_turn

MODEL = "best_models/cubeless_prob5_512_512_256_256.pt"
N_STATES = 120
TOL = 1e-5


def _load_net():
    ck = torch.load(MODEL, map_location="cpu", weights_only=True)
    net = ProbNetwork(
        hidden_sizes=ck["hidden_sizes"],
        input_size=ck.get("input_size", 196),
        activation=ck.get("activation", "relu"),
        encoder_name=ck.get("encoder_name", "perspective196"),
        raw_logits=ck.get("raw_logits", False),
    )
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck["state_dict"]
    net.load_state_dict(sd)
    net.eval()
    return net


def _walk_states(net, encode_fn, gpe_fn, n):
    """Collect `n` real mid-game states by greedy 0-ply self-play."""
    rng = random.Random(4242)
    states = []
    state, dice = opening_roll(rng)
    while len(states) < n:
        if state.is_game_over():
            state, dice = opening_roll(rng)
            continue
        feats, nxt = gpe_fn(state, dice)
        if len(nxt) == 0:
            state = switch_turn(state)
        else:
            with torch.no_grad():
                p = net(torch.from_numpy(feats))
                if net.raw_logits:
                    p = torch.sigmoid(p)
                p = prob5_postprocess(p)
                eq = prob5_to_equity(p).numpy()
            for j, ns in enumerate(nxt):
                if ns.is_game_over():
                    eq[j] = -float(ns.game_result())
            ch = nxt[int(np.argmin(eq))]
            states.append(state)
            from backgammon_engine import BoardState
            state = BoardState(points=list(ch.points), bar=list(ch.bar),
                               off=list(ch.off), turn=ch.turn)
        dice = (rng.randint(1, 6), rng.randint(1, 6))
    return states


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0

    net = _load_net()
    encode_fn, gpe_fn = P5._resolve_engine_fns(net.encoder_name)
    states = _walk_states(net, encode_fn, gpe_fn, N_STATES)
    print(f"collected {len(states)} real self-play states")

    # CPU reference
    ref = np.stack([
        P5._oneply_target_vec(s, net, encode_fn, gpe_fn, "cpu", None)
        for s in states
    ])

    # Live server, one slot, driven from this process
    from gpu_eval_server import EvalServer, RemoteEvaluator
    ctx = mp.get_context("spawn")
    server = EvalServer(
        ctx, num_slots=1, n_feat=net.input_size, out_dim=P5.NUM_OUTPUTS,
        max_rows=8192,
        build_spec={
            "net_type": "prob5",
            "hidden_sizes": net.hidden_sizes,
            "input_size": net.input_size,
            "activation": net.activation,
            "encoder_name": net.encoder_name,
            "raw_logits": net.raw_logits,
        },
        init_state_dict=net.state_dict(),
        device="cuda",
    )
    server.start()
    try:
        handles = server.client_handles()
        handles.free_slots.get()          # claim slot 0
        remote = RemoteEvaluator(0, handles)
        got = np.stack([
            P5._oneply_target_vec(s, net, encode_fn, gpe_fn, "cpu", remote)
            for s in states
        ])
    finally:
        server.stop()

    d = np.abs(ref - got)
    print(f"targets compared : {ref.shape}")
    print(f"max |delta|      : {d.max():.3e}")
    print(f"mean |delta|     : {d.mean():.3e}")

    # Equity is what move selection consumes, so check it directly too.
    eq_ref = P5._equity_from_probs_np(ref) if hasattr(P5, "_equity_from_probs_np") else None
    ok = d.max() < TOL
    print(f"\n{'PASS' if ok else 'FAIL'}: max|delta| {d.max():.3e} "
          f"{'<' if ok else '>='} {TOL:g}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
