"""
td_batch_train.py -- Batch TD(0) training for backgammon.

Instead of updating weights after every move (online TD), this script:

  1. COLLECT: Play N games with frozen weights, recording
     (state_encoding, target_value) pairs at every step.
  2. TRAIN:   Feed the batch through standard PyTorch MSE loss + optimizer.
  3. REPEAT.

The TD(0) update is the gradient of MSE loss: L = 0.5 * (target - V(s))^2,
so batch TD(0) = supervised regression on bootstrap targets.

With --oneply, targets use 1-ply lookahead (averaging over all 21 dice
outcomes) instead of a single sampled roll, giving lower-variance targets.

Benefits:
  - Collection phase is parallelized across workers (--workers flag).
  - Training phase can use GPU and large batch sizes.
  - Clean separation of game logic and ML.
"""

import csv
import os
# Single-threaded PyTorch is 2x faster for small single-vector forward passes
# (thread coordination overhead > computation savings for 196->80->1 networks).
# Must be set before importing torch.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from model import TDNetwork

# ── Engine imports ───────────────────────────────────────────────────────────

from backgammon_engine import (
    BoardState, WHITE, BLACK,
    get_legal_plays, switch_turn,
)
from encoding import encode_state


# ── 1-ply lookahead ───────────────────────────────────────────────────────────

# All 21 distinct dice outcomes with probabilities.
_DICE_OUTCOMES = [(d1, d2) for d1 in range(1, 7) for d2 in range(d1, 7)]


def _oneply_value(state, network, eng, _switch_turn, encode_fn, get_plays, device="cpu"):
    """Compute 1-ply lookahead value for a single state.

    Returns P(on-roll player wins) by enumerating all 21 dice outcomes,
    finding the best move for each, and probability-weighting the results.

    The on-roll player at `state` picks the move that MAXIMIZES their win
    probability for each dice roll. After their move + switch_turn, the
    resulting position is evaluated from the next player's perspective via
    the network, giving P(next player wins). We use 1 - that = P(on-roll wins).

    With the C engine's get_plays_and_features (which switches turn before
    encoding), V(features) = P(player after switch wins). The on-roll player
    wants to MAXIMIZE that from their own perspective:
      - features are from next-player's view after switch_turn
      - V = P(next player wins)
      - on-roll player wants argmin(V) = minimize next player's win prob
      - best_value = min(V) = P(next player wins after on-roll's best move)
      - P(on-roll wins | this dice) = 1 - best_value

    Args:
        state: BoardState where on-roll player is about to move.
        network: Value network (eval mode).
        eng: C engine module (or None for Python path).
        _switch_turn: switch_turn function.
        encode_fn: encoding function.
        get_plays: get_legal_plays function (Python path only).
        device: torch device.

    Returns:
        float: P(on-roll player wins) with 1-ply lookahead.
    """
    if state.is_game_over():
        # Shouldn't normally be called on terminal states, but handle it.
        # On-roll player at a terminal state: the previous player won.
        return 0.0

    oneply_sum = 0.0

    for d1, d2 in _DICE_OUTCOMES:
        prob = 1.0 / 36.0 if d1 == d2 else 2.0 / 36.0

        if eng is not None:
            count, features = eng.get_plays_and_features(state, (d1, d2))
            if count > 0:
                # C engine switches turn before encoding.
                # V(features) = P(next player wins after move + switch_turn).
                # On-roll player wants argmin = minimize next player's win prob.
                x = torch.tensor(features, dtype=torch.float32, device=device)
                with torch.no_grad():
                    values = network(x)
                # Terminal detection: features are encoded after switch_turn,
                # so feature[195] = "opponent" off/15 = the MOVER's off/15.
                # If the mover bore off all 15 (game over), set V=0 since
                # the player-to-move in the encoded state has already lost.
                terminal = x[:, 195] >= (1.0 - 1e-6)
                if terminal.any():
                    values = values.clone()
                    values[terminal] = 0.0
                best_for_next = torch.min(values).item()
                oneply_sum += prob * (1.0 - best_for_next)
            else:
                # No legal moves: pass (switch turn)
                s_pass = _switch_turn(state)
                x = torch.tensor(encode_fn(s_pass), dtype=torch.float32, device=device)
                with torch.no_grad():
                    v = network(x).item()
                oneply_sum += prob * (1.0 - v)
        else:
            plays = get_plays(state, (d1, d2))
            if plays:
                # Check for terminal states explicitly
                encoded_list = []
                terminal_indices = []
                for i, (_, s) in enumerate(plays):
                    if s.is_game_over():
                        terminal_indices.append(i)
                        encoded_list.append(encode_fn(_switch_turn(s)))
                    else:
                        encoded_list.append(encode_fn(_switch_turn(s)))
                encoded = np.stack(encoded_list)
                x = torch.tensor(encoded, dtype=torch.float32, device=device)
                with torch.no_grad():
                    values = network(x)
                # Override terminal states with exact value
                if terminal_indices:
                    values = values.clone()
                    for i in terminal_indices:
                        values[i] = 0.0
                best_for_next = torch.min(values).item()
                oneply_sum += prob * (1.0 - best_for_next)
            else:
                s_pass = _switch_turn(state)
                x = torch.tensor(encode_fn(s_pass), dtype=torch.float32, device=device)
                with torch.no_grad():
                    v = network(x).item()
                oneply_sum += prob * (1.0 - v)

    return oneply_sum


# ── Data collection ──────────────────────────────────────────────────────────

def _split_games(total: int, n_workers: int):
    """Split *total* games as evenly as possible across *n_workers*."""
    base = total // n_workers
    remainder = total % n_workers
    return [base + (1 if i < remainder else 0) for i in range(n_workers)]


def _collect_worker(
    state_dict, hidden_sizes, input_size, activation, encoder_name,
    num_games, fast, oneply=False,
):
    """Worker function for parallel data collection (runs in subprocess).

    When oneply=True, uses 1-ply lookahead for target computation only
    (move selection stays at 0-ply for speed and stability).
    """
    # Ensure single-threaded OpenMP in each subprocess
    os.environ["OMP_NUM_THREADS"] = "1"
    network = TDNetwork(
        hidden_sizes=hidden_sizes,
        input_size=input_size,
        activation=activation,
        encoder_name=encoder_name,
    )
    network.load_state_dict(state_dict)
    network.eval()
    return collect_training_data(
        network, num_games, fast=fast, encoder_name=encoder_name,
        oneply=oneply,
    )


def collect_training_data(
    network: TDNetwork,
    num_games: int,
    fast: bool = False,
    encoder_name: str = None,
    oneply: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Play *num_games* of self-play with frozen weights, collecting
    (state_encoding, target_value) pairs at every step.

    When oneply=True, computes the target as the 1-ply Bellman backup at the
    CURRENT state: target = E_dice[max_move(1 - V(next))].  This enumerates
    all 21 dice outcomes for the on-roll player and picks their best move for
    each, giving a low-variance unbiased estimate of V(s).

    This is still TD bootstrap, but the target averages over all 21 dice
    outcomes instead of using a single sampled roll, giving lower variance.
    Move selection stays at 0-ply (fast) just to advance the game and generate
    a diverse set of training positions.

    Returns:
        encodings: np.ndarray of shape (N, num_features), float32
        targets:   np.ndarray of shape (N,), float32

    The network weights are NOT modified during collection.
    """
    if encoder_name is None:
        encoder_name = getattr(network, 'encoder_name', 'perspective196')
    from encoding import get_encoder
    encode_fn = get_encoder(encoder_name).encode

    if fast:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__) or ".", "c_engine"))
        import bg_fast as eng
        _BoardState = eng.BoardState
        _switch_turn = eng.switch_turn
        encode_fn = eng.encode_state
        _get_plays = eng.get_legal_plays
    else:
        _BoardState = BoardState
        _switch_turn = switch_turn
        _get_plays = get_legal_plays
        eng = None

    all_encodings = []
    all_targets = []

    network.eval()

    for _ in range(num_games):
        state = _BoardState.initial()
        if random.random() < 0.5:
            state = _switch_turn(state)

        while not state.is_game_over():
            # Encode current state (this is what the network will learn from)
            current_enc = encode_fn(state)

            if oneply:
                # ── 1-ply target: Bellman backup at CURRENT state ─────
                # target = E_dice[max_move(1 - V(next))] computed at the
                # current state.  This averages over the MOVER's dice,
                # giving a low-variance, unbiased estimate of V(s).
                target = _oneply_value(
                    state, network, eng, _switch_turn,
                    encode_fn, _get_plays,
                )

            # Roll dice
            d1, d2 = random.randint(1, 6), random.randint(1, 6)

            # ── 0-ply move selection (always) ─────────────────────
            if fast:
                count, features = eng.get_plays_and_features(state, (d1, d2))
                if count > 0:
                    x = torch.tensor(features, dtype=torch.float32)
                    with torch.no_grad():
                        values = network(x)
                    idx = torch.argmin(values).item()
                    state = eng.get_chosen_state(idx)
                    eng.switch_turn_inplace(state)
                else:
                    state = _switch_turn(state)
            else:
                plays = _get_plays(state, (d1, d2))
                if plays:
                    encoded = np.stack([
                        encode_fn(_switch_turn(s)) for _, s in plays
                    ])
                    x = torch.tensor(encoded, dtype=torch.float32)
                    with torch.no_grad():
                        values = network(x)
                    idx = torch.argmin(values).item()
                    _, next_state = plays[idx]
                    state = _switch_turn(next_state)
                else:
                    state = _switch_turn(state)

            # ── Compute target (0-ply or terminal) ────────────────────
            if not oneply:
                if state.is_game_over():
                    mover = 1 - state.turn
                    assert state.winner() == mover, (
                        f"Terminal state: expected winner={mover}, got {state.winner()}"
                    )
                    target = 1.0
                else:
                    # 0-ply target: V(state) = P(opponent wins), target = 1 - V
                    with torch.no_grad():
                        x_next = torch.tensor(
                            encode_fn(state), dtype=torch.float32
                        )
                        target = 1.0 - network(x_next).item()

            all_encodings.append(current_enc)
            all_targets.append(target)

    encodings = np.array(all_encodings, dtype=np.float32)
    targets = np.array(all_targets, dtype=np.float32)
    return encodings, targets


# ── Batch training step ─────────────────────────────────────────────────────

def train_on_batch(
    network: TDNetwork,
    encodings: np.ndarray,
    targets: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 256,
    epochs: int = 1,
    device: str = "cpu",
    train_split: float = 1.0,
) -> dict:
    """Train the network on a batch of (encoding, target) pairs using MSE loss.

    This is equivalent to the TD(0) update but batched:
      online:  w += lr * (target - V(s)) * grad V(s)
      batch:   minimize  0.5 * mean((target - V(s))^2)  with optimizer

    The optimizer is created externally and passed in so that stateful
    optimizers like Adam retain their momentum buffers across batches.

    Args:
        train_split: Fraction of data used for training (rest is validation).
                     1.0 means no validation split (backward-compatible).

    Returns dict with keys: train_loss, eeval_loss (held-out validation data, None if no split), grad_norm.
    """
    network.train()
    network.to(device)

    x_all = torch.tensor(encodings, dtype=torch.float32, device=device)
    y_all = torch.tensor(targets, dtype=torch.float32, device=device)

    loss_fn = nn.MSELoss()
    n = len(x_all)
    total_loss = 0.0
    num_batches = 0
    total_grad_norm = 0.0

    # Split off validation set (if requested) using tail of first shuffle
    if train_split < 1.0:
        n_train = max(1, int(n * train_split))
    else:
        n_train = n
    x_val, y_val = None, None

    for epoch in range(epochs):
        # Shuffle all data — first n_train used for training, rest for val
        perm = torch.randperm(n, device=device)
        x_shuffled = x_all[perm]
        y_shuffled = y_all[perm]

        if n_train < n:
            x_val = x_shuffled[n_train:]
            y_val = y_shuffled[n_train:]

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            pred = network(x_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()

            # Measure gradient norm (max_norm=inf means no clipping)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                network.parameters(), max_norm=float('inf')
            )
            total_grad_norm += grad_norm.item()

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_train_loss = total_loss / max(num_batches, 1)
    avg_grad_norm = total_grad_norm / max(num_batches, 1)

    # Compute validation loss
    eval_loss = None
    if x_val is not None and len(x_val) > 0:
        network.eval()
        with torch.no_grad():
            val_pred = network(x_val)
            eval_loss = loss_fn(val_pred, y_val).item()

    return {"train_loss": avg_train_loss, "eval_loss": eval_loss, "grad_norm": avg_grad_norm}


# ── Main training loop ──────────────────────────────────────────────────────

def train(
    num_episodes: int = 100_000,
    hidden_sizes=None,
    activation: str = "relu",
    lr: float = 0.1,
    end_lr: float = None,
    games_per_cycle: int = 1000,
    batch_size: int = 256,
    epochs_per_batch: int = 1,
    device: str = "cpu",
    save_path: Optional[str] = None,
    save_every: int = 10_000,
    print_every: int = 1_000,
    eval_every: int = 0,
    eval_games: int = 5,
    gnubg_cmd: Optional[str] = None,
    network: Optional[TDNetwork] = None,
    fast: bool = False,
    use_adam: bool = False,
    encoder_name: str = "perspective196",
    workers: int = 1,
    train_split: float = 1.0,
    csv_log: Optional[str] = None,
    warmup_cycles: int = 0,
    oneply: bool = False,
) -> TDNetwork:
    """Batch TD(0) training with optional 1-ply lookahead targets.

    Alternates between:
      1. Collect *games_per_cycle* self-play games with frozen weights.
      2. (Optional) Recompute targets using 1-ply minimax over all 21 dice outcomes.
      3. Train on the collected data with MSE loss.

    1-ply targets enumerate all 21 possible dice rolls (6 doubles + 15 non-doubles)
    and compute the weighted average of opponent's best response for each roll.
    This gives much lower-variance targets than 0-ply bootstrap.

    Args:
        num_episodes:      Total games to play across all batches.
        hidden_sizes:      List of hidden layer sizes (e.g. [80] or [80, 40]).
        activation:        Hidden layer activation: sigmoid, relu, tanh, leaky_relu.
        lr:                Learning rate for optimizer.
        end_lr:            If set, linearly anneal lr from *lr* to *end_lr*.
        games_per_cycle:   Games to collect before each training step.
        batch_size:        Mini-batch size for training.
        epochs_per_batch:  Optimizer epochs per collected batch (1 = single pass).
        device:            'cpu' or 'cuda'.
        save_path:         Path prefix for saved models.
        save_every:        Save every N episodes.
        print_every:       Print progress every N episodes.
        eval_every:        gnubg eval every N episodes (0 = off).
        eval_games:        Games per gnubg evaluation.
        gnubg_cmd:         Path to gnubg executable.
        network:           Pre-existing network to continue training.
        fast:              Use the C engine.
        use_adam:          Use Adam optimizer instead of SGD.
        train_split:       Fraction of collected data for training (rest = validation).
        csv_log:           Path to CSV log file (appended each print cycle).
        warmup_cycles:     Ramp LR from lr/10 to lr over first N cycles (default: 0).
        oneply:            If True, use 1-ply lookahead targets instead of 0-ply bootstrap.
    """
    if network is None:
        network = TDNetwork(hidden_sizes=hidden_sizes, activation=activation, encoder_name=encoder_name)

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # ── Create optimizer (once, so Adam buffers persist across batches) ──
    if use_adam:
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        print(f"Using Adam optimizer (lr={lr})", flush=True)
    else:
        optimizer = torch.optim.SGD(network.parameters(), lr=lr)
        print(f"Using SGD optimizer (lr={lr})", flush=True)

    if end_lr is not None:
        print(f"LR annealing: {lr} -> {end_lr} over {num_episodes} episodes", flush=True)

    if warmup_cycles > 0:
        print(f"LR warmup: lr/10 -> lr over first {warmup_cycles} cycles", flush=True)

    if fast:
        print("Using C engine (--fast)", flush=True)

    print(f"  workers: {workers}", flush=True)

    # ── CSV logger ───────────────────────────────────────────────────
    csv_file = None
    csv_writer = None
    if csv_log:
        csv_file = open(csv_log, "a", newline="")
        csv_writer = csv.writer(csv_file)
        # Write header if file is empty
        if csv_file.tell() == 0:
            csv_writer.writerow([
                "episode", "train_loss", "eval_loss", "grad_norm",  # eval_loss = held-out validation data
                "mEMG", "lr", "games_per_sec", "samples",
            ])
        print(f"CSV logging to: {csv_log}", flush=True)

    total_played = 0
    total_samples = 0
    total_collect_time = 0.0
    total_train_time = 0.0
    next_print = print_every
    next_save = save_every
    next_eval = eval_every if eval_every > 0 else None
    start_time = time.time()
    last_mEMG = None

    # Create worker pool once (avoids re-spawning + re-importing each cycle)
    pool = None
    if workers > 1:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(workers)

    try:
        while total_played < num_episodes:
            # ── LR annealing ──────────────────────────────────────────
            cycle = total_played // games_per_cycle
            if end_lr is not None:
                frac = total_played / max(num_episodes, 1)
                current_lr = lr + (end_lr - lr) * frac
            else:
                current_lr = lr

            # Warmup: ramp from lr/10 to scheduled LR over first N cycles
            if warmup_cycles > 0 and cycle < warmup_cycles:
                warmup_frac = (cycle + 1) / warmup_cycles
                current_lr = current_lr * (0.1 + 0.9 * warmup_frac)

            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # How many games this batch?
            batch_games = min(games_per_cycle, num_episodes - total_played)

            # ── Phase 1: Collect ─────────────────────────────────────
            # When oneply=True, workers use 0-ply move selection (fast)
            # with 1-ply target computation (lower variance).
            t_collect = time.perf_counter()
            if pool is not None:
                state_dict = network.state_dict()
                game_splits = _split_games(batch_games, workers)
                args_list = [
                    (state_dict, network.hidden_sizes, network.input_size,
                     network.activation, encoder_name, n, fast, oneply)
                    for n in game_splits
                ]
                results = pool.starmap(_collect_worker, args_list)
                encodings = np.concatenate([r[0] for r in results])
                targets = np.concatenate([r[1] for r in results])
            else:
                encodings, targets = collect_training_data(
                    network, batch_games, fast=fast, encoder_name=encoder_name,
                    oneply=oneply,
                )
            collect_time = time.perf_counter() - t_collect

            total_played += batch_games
            total_samples += len(encodings)

            # ── Phase 2: Train ───────────────────────────────────────
            t_train = time.perf_counter()
            train_result = train_on_batch(
                network, encodings, targets,
                optimizer=optimizer, batch_size=batch_size,
                epochs=epochs_per_batch, device=device,
                train_split=train_split,
            )
            avg_loss = train_result["train_loss"]
            eval_loss = train_result["eval_loss"]
            grad_norm = train_result["grad_norm"]
            train_time = time.perf_counter() - t_train
            total_collect_time += collect_time
            total_train_time += train_time

            # ── Reporting ────────────────────────────────────────────
            if total_played >= next_print or total_played >= num_episodes:
                elapsed = time.time() - start_time
                eps_per_sec = total_played / elapsed if elapsed > 0 else 0
                samples_per_game = len(encodings) / max(batch_games, 1)
                collect_pct = 100 * total_collect_time / elapsed if elapsed > 0 else 0
                train_pct = 100 * total_train_time / elapsed if elapsed > 0 else 0
                eval_str = f" eval {eval_loss:.4f}" if eval_loss is not None else ""  # held-out validation data
                msg = (
                    f"Episode {total_played:>7d} | "
                    f"{eps_per_sec:.1f} games/sec | "
                    f"lr {current_lr:.5f} | "
                    f"loss {avg_loss:.4f}{eval_str} | "
                    f"grad {grad_norm:.4f} | "
                    f"samples {len(encodings)} ({samples_per_game:.0f}/game) | "
                    f"collect {collect_time:.1f}s ({collect_pct:.0f}%) "
                    f"train {train_time:.1f}s ({train_pct:.0f}%)"
                )
                if oneply:
                    msg += " | 1ply"
                print(msg, flush=True)
                # CSV logging
                if csv_writer:
                    csv_writer.writerow([
                        total_played,
                        f"{avg_loss:.6f}",
                        f"{eval_loss:.6f}" if eval_loss is not None else "",
                        f"{grad_norm:.6f}",
                        f"{last_mEMG:.1f}" if last_mEMG is not None else "",
                        f"{current_lr:.6f}",
                        f"{eps_per_sec:.1f}",
                        len(encodings),
                    ])
                    csv_file.flush()
                while next_print <= total_played:
                    next_print += print_every

            # ── Saving ───────────────────────────────────────────────
            if save_path and total_played >= next_save:
                path = f"{save_path}_ep{total_played}.pt"
                network.save(path)
                print(f"  -> saved {path}", flush=True)
                while next_save <= total_played:
                    next_save += save_every

            # ── Evaluation ───────────────────────────────────────────
            if next_eval is not None and total_played >= next_eval:
                # Save temp model for parallel eval
                import tempfile
                network.to("cpu")
                tmp_path = os.path.join(
                    tempfile.gettempdir(), f"_eval_tmp_{os.getpid()}.pt"
                )
                network.save(tmp_path)

                try:
                    from play_models import play_matches_parallel, compute_binomial_pvalue
                    eval_workers = min(workers, 30) if workers > 1 else 14
                    wins, losses = play_matches_parallel(
                        tmp_path, "gnubg", eval_games,
                        workers=eval_workers,
                        model2_type="gnubg", plies=0,
                    )
                    total_eval = wins + losses
                    win_pct = 100 * wins / total_eval if total_eval > 0 else 0
                    p_val = compute_binomial_pvalue(wins, total_eval)
                    print(
                        f"  -> vs gnubg 0-ply (ep {total_played}): "
                        f"{win_pct:.1f}% ({wins}/{total_eval}, p={p_val:.4f})",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  -> eval failed: {e}", flush=True)
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

                network.to(device)
                while next_eval <= total_played:
                    next_eval += eval_every

    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        if csv_file is not None:
            csv_file.close()

    # ── Final save ───────────────────────────────────────────────────
    if save_path:
        path = f"{save_path}_final.pt"
        network.save(path)
        print(f"  -> saved {path}", flush=True)

    elapsed = time.time() - start_time
    print(
        f"\nBatch training complete: {total_played} episodes, "
        f"{total_samples} total samples, {elapsed:.1f}s "
        f"({total_played / elapsed:.1f} games/sec)",
        flush=True,
    )

    return network


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch TD(0) training with optional 1-ply lookahead targets"
    )
    parser.add_argument("--episodes", type=int, default=100_000,
                        help="Total self-play games")
    parser.add_argument("--hidden", type=int, nargs='+', default=[40],
                        help="Hidden layer sizes (e.g. --hidden 80 or --hidden 80 40)")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["sigmoid", "relu", "tanh", "leaky_relu", "hardsigmoid"],
                        help="Hidden layer activation function (default: relu)")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--end-lr", type=float, default=None,
                        help="If set, linearly anneal lr to this value")
    parser.add_argument("--games-per-cycle", type=int, default=1000,
                        help="Games to collect before each training step")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size for SGD")
    parser.add_argument("--epochs-per-batch", type=int, default=1,
                        help="SGD epochs per collected batch")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for training (cpu or cuda)")
    parser.add_argument("--save-path", type=str, default="models/td_batch",
                        help="Path prefix for saved models")
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--print-every", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=5)
    parser.add_argument("--gnubg", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to saved model to resume from")
    parser.add_argument("--expand", type=str, default=None,
                        help="Path to smaller model to width-expand from (use with --hidden for target size)")
    parser.add_argument("--expand-depth", type=str, default=None,
                        help="Path to trained model to depth-expand from: appends one near-identity hidden layer")
    parser.add_argument("--fast", action="store_true",
                        help="Use C engine")
    parser.add_argument("--adam", action="store_true",
                        help="Use Adam optimizer instead of SGD")
    parser.add_argument("--encoder", type=str, default="perspective196",
                        help="Encoder name (default: perspective196)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel collection workers (default: 1)")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Fraction of data for training, rest for validation (default: 0.8)")
    parser.add_argument("--csv-log", type=str, default=None,
                        help="Path to CSV log file (appended each print cycle)")
    parser.add_argument("--warmup-cycles", type=int, default=0,
                        help="Ramp LR from lr/10 to lr over first N cycles (default: 0)")
    parser.add_argument("--oneply", action="store_true",
                        help="Use 1-ply lookahead targets instead of 0-ply bootstrap")
    args = parser.parse_args()

    print(f"  hidden_sizes: {args.hidden}")
    print(f"  activation: {args.activation}")
    print(f"  lr: {args.lr}")
    if args.end_lr:
        print(f"  end_lr: {args.end_lr}")
    print(f"  optimizer: {'adam' if args.adam else 'sgd'}")
    print(f"  games_per_cycle: {args.games_per_cycle}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs_per_batch: {args.epochs_per_batch}")
    print(f"  workers: {args.workers}")
    if args.oneply:
        print(f"  target: 1-ply lookahead (enumerates all 21 dice outcomes)")

    # Mutual exclusivity check
    resume_opts = [args.expand, args.expand_depth, args.resume]
    if sum(1 for x in resume_opts if x) > 1:
        parser.error("--expand, --expand-depth, and --resume are mutually exclusive")

    if args.expand:
        print(f"Width-expanding from {args.expand} to {args.hidden}")
        source = TDNetwork.load(args.expand)
        network = TDNetwork.width_expand(source, args.hidden)
        print(f"  Expanded: {source.hidden_sizes} -> {network.hidden_sizes} "
              f"({sum(p.numel() for p in source.parameters())} -> "
              f"{sum(p.numel() for p in network.parameters())} params)")
    elif args.expand_depth:
        source = TDNetwork.load(args.expand_depth)
        # Use last element of --hidden as new layer size if provided and different from source
        new_layer_size = None
        if args.hidden and len(args.hidden) == len(source.hidden_sizes) + 1:
            new_layer_size = args.hidden[-1]
        print(f"Depth-expanding from {args.expand_depth}" +
              (f" (new layer size: {new_layer_size})" if new_layer_size else ""))
        network = TDNetwork.depth_expand(source, new_layer_size=new_layer_size)
        print(f"  Expanded: {source.hidden_sizes} -> {network.hidden_sizes} "
              f"({sum(p.numel() for p in source.parameters())} -> "
              f"{sum(p.numel() for p in network.parameters())} params)")
    elif args.resume:
        print(f"Resuming from {args.resume}")
        network = TDNetwork.load(args.resume)
    else:
        network = None

    train(
        num_episodes=args.episodes,
        hidden_sizes=args.hidden,
        activation=args.activation,
        lr=args.lr,
        end_lr=args.end_lr,
        games_per_cycle=args.games_per_cycle,
        batch_size=args.batch_size,
        epochs_per_batch=args.epochs_per_batch,
        device=args.device,
        save_path=args.save_path,
        save_every=args.save_every,
        print_every=args.print_every,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        gnubg_cmd=args.gnubg,
        network=network,
        fast=args.fast,
        use_adam=args.adam,
        encoder_name=args.encoder,
        workers=args.workers,
        train_split=args.train_split,
        csv_log=args.csv_log,
        warmup_cycles=args.warmup_cycles,
        oneply=args.oneply,
    )
