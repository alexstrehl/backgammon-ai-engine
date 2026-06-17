#!/usr/bin/env python3
"""train_prob5.py -- Batch TD(0) self-play training for a 5-output
probability network (cubeless money).

Outputs (from the on-roll player's perspective; all sigmoid):
  0: P(win)
  1: P(win gammon)
  2: P(win backgammon)
  3: P(lose gammon)
  4: P(lose backgammon)

Money equity derived by :func:`model.prob5_to_equity` as
    eq = 2*P(win) + P(wg) + P(wbg) - P(lg) - P(lbg) - 1

TD targets
~~~~~~~~~~
At a non-terminal next state ``s'`` evaluated from the opponent's view,
``v = net(s')``, the mover's target for the pre-move state is
    [1 - v[0], v[3], v[4], v[1], v[2]]
(i.e. swap win↔loss on output 0, swap the win/lose-gammon and
win/lose-backgammon pairs). Terminal states where the mover just won
give ``[1, is_g, is_bg, 0, 0]``.

0-ply: target bootstraps from the chosen next state. 1-ply: target is
the dice-expected value of the best move over all 21 dice outcomes —
see :func:`_oneply_target_vec`.

Usage is intentionally similar to ``train_batch.py``::

    python3 train_prob5.py --hidden 512,512,256 \
        --num-episodes 200000 --lr 1e-3 --save working_models/prob5
"""

# Pin BLAS / PyTorch threads BEFORE importing torch — matches the
# convention in train_batch.py.
import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import math
import multiprocessing as mp
import random
import sys
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

from backgammon_engine import (
    BoardState,
    get_legal_plays_encoded as _py_get_legal_plays_encoded,
    opening_roll,
    switch_turn,
)
from encoding import (
    encode_state as _py_encode_state,
    get_encoder, OPP_OFF_INDEX, TERMINAL_OFF_THRESHOLD,
)
from model import ProbNetwork, prob5_to_equity, prob5_postprocess
from train_cli import parse_hidden_sizes, resolve_save_path

try:
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _c_path = os.path.join(_here, "c_engine")
    if _c_path not in _sys.path:
        _sys.path.insert(0, _c_path)
    import bg_fast as _bg_fast  # noqa: E402
    _BG_FAST_AVAILABLE = True
except Exception:
    _bg_fast = None
    _BG_FAST_AVAILABLE = False


NUM_OUTPUTS = ProbNetwork.NUM_OUTPUTS  # 5

# All 21 distinct dice outcomes with per-roll probabilities.
_DICE_OUTCOMES: List[Tuple[Tuple[int, int], float]] = [
    ((d1, d2), (1.0 / 36.0) if d1 == d2 else (2.0 / 36.0))
    for d1 in range(1, 7)
    for d2 in range(d1, 7)
]


# ── Target helpers ────────────────────────────────────────────────────


def _terminal_target(next_state: BoardState) -> np.ndarray:
    """5-vector target when the mover just ended the game with a win.

    The mover's on-roll view after the winning move still has *them* as
    the winner; ``game_result()`` returns 1/2/3 for single/gammon/bg.
    """
    r = next_state.game_result()
    return np.array(
        [1.0, float(r >= 2), float(r >= 3), 0.0, 0.0],
        dtype=np.float32,
    )


def _flip_vec(v: np.ndarray) -> np.ndarray:
    """Opponent's 5-vector -> mover's TD target: swap win/lose roles."""
    return np.array(
        [1.0 - v[0], v[3], v[4], v[1], v[2]],
        dtype=np.float32,
    )


# ── 1-ply exact Bellman backup ────────────────────────────────────────


def _oneply_target_vec(
    state: BoardState,
    network: ProbNetwork,
    encode_fn: Callable[[BoardState], np.ndarray],
    gpe_fn: Callable,
    device: str,
) -> np.ndarray:
    target = np.zeros(NUM_OUTPUTS, dtype=np.float32)

    # Two separate buffers so row lookup is unambiguous.
    pass_buf: List[np.ndarray] = []
    pass_dice: List[int] = []  # dice_idx for each row in pass_buf
    play_buf: List[np.ndarray] = []
    # play_info[d_idx] = list of (terminal_vec_or_None, play_buf_row_or_None)
    play_info: dict = {}

    for d_idx, (dice, prob) in enumerate(_DICE_OUTCOMES):
        features, next_states = gpe_fn(state, dice)
        if len(next_states) == 0:
            # Forced pass (mover has no legal move): the board is unchanged,
            # so the opponent's view can never be terminal here. Record it for
            # the batched 1-ply backup below.
            pass_dice.append(d_idx)
            pass_buf.append(encode_fn(switch_turn(state)))
            continue

        # Each successor encoding (row of `features`) is already the
        # opponent's on-roll view. Terminal successors (mover bore off
        # all 15) are detected from the OPP_OFF feature.
        per_move: List[Tuple[Optional[np.ndarray], Optional[int]]] = []
        term_mask = features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
        for i in range(len(next_states)):
            if term_mask[i]:
                per_move.append((_terminal_target(next_states[i]), None))
            else:
                play_buf.append(features[i])
                per_move.append((None, len(play_buf) - 1))
        play_info[d_idx] = per_move

    # One forward pass over the combined buffer.
    total_rows = len(pass_buf) + len(play_buf)
    if total_rows:
        stack = pass_buf + play_buf
        x = torch.from_numpy(np.stack(stack)).to(device)
        with torch.no_grad():
            out = network(x)
            if network.raw_logits:
                out = torch.sigmoid(out)
            out = prob5_postprocess(out)
        out_np = out.detach().cpu().numpy()
        pass_out = out_np[: len(pass_buf)]
        play_out = out_np[len(pass_buf):]
    else:
        pass_out = np.empty((0, NUM_OUTPUTS), dtype=np.float32)
        play_out = np.empty((0, NUM_OUTPUTS), dtype=np.float32)

    # Apply per-dice best-move selection.
    pass_dice_to_row = {d: i for i, d in enumerate(pass_dice)}
    for d_idx, (dice, prob) in enumerate(_DICE_OUTCOMES):
        if d_idx in play_info:
            move_targets: List[np.ndarray] = []
            for term_vec, play_row in play_info[d_idx]:
                if term_vec is not None:
                    move_targets.append(term_vec)
                else:
                    move_targets.append(_flip_vec(play_out[play_row]))
            move_arr = np.stack(move_targets)
            eqs = prob5_to_equity(torch.from_numpy(move_arr)).numpy()
            target += prob * move_arr[int(np.argmax(eqs))]
        elif d_idx in pass_dice_to_row:
            target += prob * _flip_vec(pass_out[pass_dice_to_row[d_idx]])
        # else: unreachable — every dice is either a play or a forced pass.

    # Float-drift clamp. Each per-dice contribution is in [0, 1] and
    # dice probs sum to 1, so mathematically target ∈ [0, 1], but
    # fp32 summation can overshoot by ~1e-7 which BCE rejects.
    return np.clip(target, 0.0, 1.0)


# ── Episode collection ───────────────────────────────────────────────


def _collect_one_episode(
    network: ProbNetwork,
    rng: random.Random,
    encode_fn: Callable[[BoardState], np.ndarray],
    gpe_fn: Callable,
    oneply: bool,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Play one self-play episode. Returns (encodings Nx196, targets Nx5)."""
    state, dice = opening_roll(rng)
    encs: List[np.ndarray] = []
    tgts: List[np.ndarray] = []

    while not state.is_game_over():
        encs.append(encode_fn(state))

        # Move generation + successor encoding in one call (C fast path
        # when available). Each row of `features` is the encoding of the
        # corresponding `next_states` entry (opponent on roll).
        features, next_states = gpe_fn(state, dice)

        if oneply:
            tgts.append(_oneply_target_vec(
                state, network, encode_fn, gpe_fn, device,
            ))
            # Move selection (0-ply): pick the move with best derived
            # equity from mover's view.
            if len(next_states):
                with torch.no_grad():
                    x = torch.from_numpy(features).to(device)
                    opp_probs = network(x)
                    if network.raw_logits:
                        opp_probs = torch.sigmoid(opp_probs)
                    opp_probs = prob5_postprocess(opp_probs)
                    # Equity from *opponent's* view -> pick min for mover.
                    opp_eq = prob5_to_equity(opp_probs).cpu().numpy()
                # Terminal correction: a move that ends the game (mover won)
                # gets its exact result, not the net's arbitrary output on the
                # terminal encoding -- matches play-time ProbAgent._choose_0ply.
                for j, ns in enumerate(next_states):
                    if ns.is_game_over():
                        opp_eq[j] = -float(ns.game_result())
                state = next_states[int(np.argmin(opp_eq))]
            else:
                state = switch_turn(state)
        else:
            if len(next_states):
                with torch.no_grad():
                    x = torch.from_numpy(features).to(device)
                    opp_probs = network(x)
                    if network.raw_logits:
                        opp_probs = torch.sigmoid(opp_probs)
                    opp_probs = prob5_postprocess(opp_probs)
                    opp_eq = prob5_to_equity(opp_probs).cpu().numpy()
                    opp_probs_np = opp_probs.detach().cpu().numpy()
                # Terminal correction: prefer a game-ending (winning) move with
                # its exact result -- matches play-time ProbAgent._choose_0ply.
                for j, ns in enumerate(next_states):
                    if ns.is_game_over():
                        opp_eq[j] = -float(ns.game_result())
                best = int(np.argmin(opp_eq))
                next_state = next_states[best]
                if next_state.is_game_over():
                    tgts.append(_terminal_target(next_state))
                    break
                # Otherwise: target = flipped opponent's output at chosen.
                tgts.append(_flip_vec(opp_probs_np[best]))
                state = next_state
            else:
                # Forced pass: board unchanged, so opp_view can't be terminal;
                # bootstrap the target from the opponent's flipped value.
                opp_view = switch_turn(state)
                with torch.no_grad():
                    x = torch.from_numpy(
                        encode_fn(opp_view)[None, :]
                    ).to(device)
                    p = network(x)
                    if network.raw_logits:
                        p = torch.sigmoid(p)
                    p = prob5_postprocess(p)
                    p = p.detach().cpu().numpy()[0]
                tgts.append(_flip_vec(p))
                state = opp_view

        dice = (rng.randint(1, 6), rng.randint(1, 6))

    if not encs:
        return (
            np.empty((0, network.input_size), dtype=np.float32),
            np.empty((0, NUM_OUTPUTS), dtype=np.float32),
        )
    return np.stack(encs), np.stack(tgts)


def _resolve_engine_fns(encoder_name: str = "perspective196"):
    """Return (encode_fn, gpe_fn).

    ``gpe_fn(state, dice) -> (features Nx196, next_states)`` does move
    generation and perspective encoding of every successor in one call
    (the engine has already switched turn, so each successor encoding is
    the opponent's on-roll view — exactly what the targets/selection
    use). Uses the C engine (``bg_fast``) for the perspective196 encoder,
    else the pure-Python engine. Both impls share the
    ``(state, dice, encoder)`` signature and produce identical features.
    """
    enc = get_encoder(encoder_name)
    fast = _BG_FAST_AVAILABLE and encoder_name == "perspective196"
    encode_fn = _bg_fast.encode_state if fast else _py_encode_state
    impl = _bg_fast.get_legal_plays_encoded if fast else _py_get_legal_plays_encoded

    def gpe_fn(state, dice, _impl=impl, _enc=enc):
        return _impl(state, dice, _enc)

    return encode_fn, gpe_fn


# ── Parallel collection workers ──────────────────────────────────────


def _split_episodes(total: int, n_workers: int) -> List[int]:
    base = total // n_workers
    rem = total % n_workers
    return [base + (1 if i < rem else 0) for i in range(n_workers)]


def _collect_worker(args):
    """Run by a worker subprocess. Collects N episodes, returns
    concatenated (encodings, targets) numpy arrays.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    (state_dict, hidden_sizes, input_size, activation, encoder_name,
     raw_logits, num_episodes, seed, oneply, oneply_device) = args

    net = ProbNetwork(
        hidden_sizes=hidden_sizes,
        input_size=input_size,
        activation=activation,
        encoder_name=encoder_name,
        raw_logits=raw_logits,
    )
    net.load_state_dict(state_dict)
    net.eval()
    if oneply and oneply_device != "cpu":
        net.to(oneply_device)

    encode_fn, gpe_fn = _resolve_engine_fns(encoder_name)
    rng = random.Random(seed)

    enc_chunks: List[np.ndarray] = []
    tgt_chunks: List[np.ndarray] = []
    for _ in range(num_episodes):
        enc, tgt = _collect_one_episode(
            net, rng, encode_fn, gpe_fn,
            oneply=oneply, device=oneply_device,
        )
        if len(enc) > 0:
            enc_chunks.append(enc)
            tgt_chunks.append(tgt)

    if not enc_chunks:
        return (
            np.empty((0, input_size), dtype=np.float32),
            np.empty((0, NUM_OUTPUTS), dtype=np.float32),
        )
    return np.concatenate(enc_chunks, axis=0), np.concatenate(tgt_chunks, axis=0)


# ── Training loop ─────────────────────────────────────────────────────


def train(
    net: ProbNetwork,
    num_episodes: int,
    lr: float,
    end_lr: Optional[float],
    episodes_per_round: int,
    batch_size: int,
    workers: int,
    device: str,
    oneply: bool,
    pw_weight: float,
    use_mse: bool,
    seed: Optional[int],
    log_every: int,
    save_path: Optional[str],
    save_every: int,
    warmup_cycles: int,
) -> None:
    """Round-based batch TD(0) on the 5-output prob network.

    One round = collect ``episodes_per_round`` self-play episodes,
    shuffle, train one epoch in ``batch_size`` slices. Loss is per-
    output BCE (or BCEWithLogits if ``net.raw_logits``), with ``pw_weight``
    upweighting the P(win) output, averaged across outputs.
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    if net.raw_logits:
        loss_name = "BCEWithLogits"
    elif use_mse:
        loss_name = "MSE"
    else:
        loss_name = "BCE"
    per_output_weights = torch.tensor(
        [pw_weight, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device,
    )

    print(
        f"Prob5 TD(0) | cubeless-money | hidden={net.hidden_sizes} "
        f"| input={net.input_size} | loss={loss_name} | "
        f"lr={lr} -> {end_lr} | pw_weight={pw_weight} | oneply={oneply} "
        f"| workers={workers} | device={device}"
    )

    pool = (
        mp.get_context("spawn").Pool(processes=workers)
        if workers > 1
        else None
    )

    rng = random.Random(seed)
    num_rounds = math.ceil(num_episodes / episodes_per_round)
    episodes_played = 0
    t0_all = time.time()

    try:
        for round_idx in range(num_rounds):
            round_size = min(
                episodes_per_round, num_episodes - episodes_played,
            )

            # LR schedule: linear anneal + optional warmup ramp.
            if end_lr is not None:
                frac = episodes_played / max(num_episodes, 1)
                cur_lr = lr + (end_lr - lr) * frac
            else:
                cur_lr = lr
            if warmup_cycles > 0 and round_idx < warmup_cycles:
                warmup_frac = (round_idx + 1) / warmup_cycles
                cur_lr = cur_lr * (0.1 + 0.9 * warmup_frac)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            # ── Collect ───────────────────────────────────────────
            t_c0 = time.perf_counter()
            net_cpu_sd = {
                k: v.detach().cpu() for k, v in net.state_dict().items()
            }
            if pool is None:
                encode_fn, get_legal_plays_fn = _resolve_engine_fns()
                oneply_dev = device if oneply else "cpu"
                net.eval()
                if oneply and oneply_dev != "cpu":
                    net.to(oneply_dev)
                enc_chunks: List[np.ndarray] = []
                tgt_chunks: List[np.ndarray] = []
                for _ in range(round_size):
                    enc, tgt = _collect_one_episode(
                        net, rng, encode_fn, get_legal_plays_fn,
                        oneply=oneply, device=oneply_dev,
                    )
                    if len(enc) > 0:
                        enc_chunks.append(enc)
                        tgt_chunks.append(tgt)
                if oneply and oneply_dev != "cpu":
                    net.to("cpu")
            else:
                splits = _split_episodes(round_size, workers)
                oneply_dev = device if oneply else "cpu"
                worker_args = [
                    (
                        net_cpu_sd, net.hidden_sizes, net.input_size,
                        net.activation, net.encoder_name, net.raw_logits,
                        n, rng.randint(0, 2**31 - 1), oneply, oneply_dev,
                    )
                    for n in splits if n > 0
                ]
                results = pool.map(_collect_worker, worker_args)
                enc_chunks = [r[0] for r in results if len(r[0]) > 0]
                tgt_chunks = [r[1] for r in results if len(r[1]) > 0]
            t_collect = time.perf_counter() - t_c0

            episodes_played += round_size
            if not enc_chunks:
                continue
            pool_enc = np.concatenate(enc_chunks, axis=0)
            pool_tgt = np.concatenate(tgt_chunks, axis=0)
            n = len(pool_enc)

            # ── Train: shuffle + per-output loss ──────────────────
            t_t0 = time.perf_counter()
            net.to(device)
            net.train()
            x_all = torch.from_numpy(pool_enc).to(device)
            y_all = torch.from_numpy(pool_tgt).to(device)
            perm = torch.randperm(n, device=device)

            total_loss = 0.0
            per_output_sum = torch.zeros(NUM_OUTPUTS, device=device)
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb = x_all[idx]
                yb = y_all[idx]
                pred = net(xb)
                if net.raw_logits:
                    per_out = torch.stack([
                        F.binary_cross_entropy_with_logits(pred[:, j], yb[:, j])
                        for j in range(NUM_OUTPUTS)
                    ])
                elif use_mse:
                    per_out = torch.stack([
                        F.mse_loss(pred[:, j], yb[:, j])
                        for j in range(NUM_OUTPUTS)
                    ])
                else:
                    per_out = torch.stack([
                        F.binary_cross_entropy(pred[:, j], yb[:, j])
                        for j in range(NUM_OUTPUTS)
                    ])
                loss = (per_out * per_output_weights).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                per_output_sum += per_out.detach()
                n_batches += 1

            net.to("cpu")
            t_train = time.perf_counter() - t_t0

            avg_loss = total_loss / max(n_batches, 1)
            per_out_avg = (per_output_sum / max(n_batches, 1)).cpu().numpy()

            # ── Log ───────────────────────────────────────────────
            if log_every and (round_idx + 1) % log_every == 0:
                mean_tgt = pool_tgt.mean(axis=0)
                mean_eq = (
                    2.0 * mean_tgt[0]
                    + mean_tgt[1]
                    + mean_tgt[2]
                    - mean_tgt[3]
                    - mean_tgt[4]
                    - 1.0
                )
                elapsed = time.time() - t0_all
                gps = episodes_played / elapsed if elapsed > 0 else 0.0
                print(
                    f"Round {round_idx + 1:4d}/{num_rounds} "
                    f"| eps {episodes_played}/{num_episodes} "
                    f"| pool {n:6d} "
                    f"| loss {avg_loss:.4f} "
                    f"| mean_eq {mean_eq:+.3f} "
                    f"| Pw {mean_tgt[0]:.3f} "
                    f"| lr {cur_lr:.2e} "
                    f"| collect {t_collect:5.1f}s train {t_train:4.1f}s "
                    f"| {gps:.0f} g/s"
                )
                print(
                    f"    per-output loss: Pw={per_out_avg[0]:.4f} "
                    f"Pgw={per_out_avg[1]:.4f} Pbw={per_out_avg[2]:.4f} "
                    f"Pgl={per_out_avg[3]:.4f} Pbl={per_out_avg[4]:.4f}"
                )

            if save_path and save_every > 0 and episodes_played % save_every < round_size:
                ck_path = f"{save_path}_ep{episodes_played}.pt"
                net.save(ck_path)
                print(f"    -> checkpoint saved to {ck_path}")

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if save_path:
        net.save(save_path)
        print(f"Saved final network to {save_path}")

    elapsed = time.time() - t0_all
    gps = num_episodes / elapsed if elapsed > 0 else 0.0
    print(f"Done in {elapsed:.1f}s ({gps:.1f} ep/s)")


# ── CLI ───────────────────────────────────────────────────────────────


def _build_network(args) -> ProbNetwork:
    """Resolve --resume / --expand / --expand-depth / fresh."""
    sources = [args.resume, args.expand, args.expand_depth]
    if sum(1 for s in sources if s) > 1:
        raise ValueError(
            "--resume, --expand, --expand-depth are mutually exclusive"
        )

    target_hidden = (
        parse_hidden_sizes(args.hidden) if args.hidden else None
    )

    if args.resume:
        net = ProbNetwork.load(args.resume)
        print(f"Resumed from {args.resume}: hidden={net.hidden_sizes}")
        if args.raw_logits and not net.raw_logits:
            net.raw_logits = True
            print("  --raw-logits override: switching to logits mode.")
        if target_hidden and target_hidden != net.hidden_sizes:
            print(
                f"  WARN: --hidden {args.hidden} ignored "
                f"(saved network has {net.hidden_sizes})"
            )
        return net

    if args.expand:
        src = ProbNetwork.load(args.expand)
        if target_hidden is None:
            raise ValueError("--expand requires --hidden <target sizes>")
        net = ProbNetwork.width_expand(src, target_hidden)
        print(
            f"Width-expanded from {args.expand}: "
            f"{src.hidden_sizes} -> {target_hidden}"
        )
        return net

    if args.expand_depth:
        src = ProbNetwork.load(args.expand_depth)
        net = ProbNetwork.depth_expand(
            src, new_layer_size=args.expand_depth_size,
        )
        print(
            f"Depth-expanded from {args.expand_depth}: "
            f"{src.hidden_sizes} -> {net.hidden_sizes}"
        )
        return net

    if target_hidden is None:
        target_hidden = [80]
    return ProbNetwork(
        hidden_sizes=target_hidden,
        activation=args.activation,
        raw_logits=args.raw_logits,
    )


def main():
    p = argparse.ArgumentParser(
        description=(
            "Batch TD(0) self-play for a 5-output probability net "
            "(cubeless money)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g_net = p.add_argument_group("network")
    g_net.add_argument("--hidden", type=str, default=None,
                       help="Comma-separated hidden sizes, e.g. 512,512,256.")
    g_net.add_argument("--activation", default="relu",
                       choices=["relu", "leaky_relu", "tanh", "sigmoid"])
    g_net.add_argument("--raw-logits", action="store_true",
                       help="Train on raw logits (pair with BCEWithLogits).")

    g_io = p.add_argument_group("i/o")
    g_io.add_argument("--resume", default=None,
                      help="Resume training from a saved prob5 checkpoint.")
    g_io.add_argument("--expand", default=None,
                      help="Width-expand from a smaller prob5 checkpoint "
                           "to --hidden.")
    g_io.add_argument("--expand-depth", default=None,
                      help="Depth-expand a prob5 checkpoint by appending "
                           "one hidden layer.")
    g_io.add_argument("--expand-depth-size", type=int, default=None,
                      help="Size of the new layer under --expand-depth.")
    g_io.add_argument("--save", default=None,
                      help="Save final network to this path.")
    g_io.add_argument("--save-every", type=int, default=0,
                      help="Also save an intermediate checkpoint every N "
                           "episodes (0 disables). Written to "
                           "{--save}_ep{N}.pt.")

    g_train = p.add_argument_group("training")
    g_train.add_argument("--num-episodes", type=int, required=True)
    g_train.add_argument("--lr", type=float, default=1e-3)
    g_train.add_argument("--end-lr", type=float, default=None,
                         help="Linear LR anneal from --lr down to --end-lr "
                              "over the whole run.")
    g_train.add_argument("--warmup-cycles", type=int, default=0,
                         help="Ramp LR from lr*0.1 to lr over first N rounds.")
    g_train.add_argument("--batch-size", type=int, default=256)
    g_train.add_argument("--episodes-per-round", type=int, default=1000)
    g_train.add_argument("--workers", type=int, default=1,
                         help="Parallel collection workers (spawn-based).")
    g_train.add_argument("--device", default="cpu",
                         help="torch device for the master training step "
                              "(and for 1-ply worker inference).")
    g_train.add_argument("--oneply", action="store_true",
                         help="Use exact 1-ply Bellman backup targets.")
    g_train.add_argument("--pw-weight", type=float, default=1.0,
                         help="Relative loss weight on P(win) output.")
    g_train.add_argument("--mse", action="store_true",
                         help="Use MSE loss per output instead of BCE.")
    g_train.add_argument("--seed", type=int, default=None)
    g_train.add_argument("--log-every", type=int, default=1)
    g_train.add_argument("--torch-seed", type=int, default=None)

    args = p.parse_args()

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)

    net = _build_network(args)

    save_path = None
    if args.save:
        save_path = resolve_save_path(
            args.save, "cubeless-money", net.hidden_sizes,
        )

    train(
        net,
        num_episodes=args.num_episodes,
        lr=args.lr,
        end_lr=args.end_lr,
        episodes_per_round=args.episodes_per_round,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        oneply=args.oneply,
        pw_weight=args.pw_weight,
        use_mse=args.mse,
        seed=args.seed,
        log_every=args.log_every,
        save_path=save_path,
        save_every=args.save_every,
        warmup_cycles=args.warmup_cycles,
    )


if __name__ == "__main__":
    main()
