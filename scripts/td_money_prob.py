"""
td_money_prob196.py -- Batch TD(0) cubeless money training with 5 probability outputs.

Uses the standard 196-feature perspective encoding (no cube features).
Designed to work with the base backgammon-ai-engine codebase.

Outputs (from on-roll player's perspective):
  0: P(win)
  1: P(win gammon)
  2: P(win backgammon)
  3: P(lose gammon)
  4: P(lose backgammon)

Equity derived as:
  eq = P(win) - P(lose) + P(wg) - P(lg) + P(wbg) - P(lbg)
     = 2*P(win) + P(wg) + P(wbg) - P(lg) - P(lbg) - 1

Move selection: pick the move that maximizes derived equity (from mover's
perspective, which means minimizing opponent's equity after switch_turn).
"""

import os
import argparse
import multiprocessing as mp
import random
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn

from backgammon_engine import BoardState, WHITE, BLACK, get_legal_plays, switch_turn
from encoding import encode_state

NUM_OUTPUTS = 5
NUM_FEATURES = 196


# ── Network ───────────────────────────────────────────────────────────────────

class ProbNet(nn.Module):
    """5-output probability network for cubeless money games."""

    def __init__(self, hidden_sizes=None, input_size=NUM_FEATURES, activation="relu",
                 raw_logits=False):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [80]
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.activation = activation
        self.raw_logits = raw_logits
        act_fn = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU,
                  "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[activation]

        layers = []
        prev = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(act_fn())
            prev = size
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, NUM_OUTPUTS)

    def forward(self, x):
        raw = self.head(self.trunk(x))
        if self.raw_logits:
            return raw
        return torch.sigmoid(raw)

    def save(self, path):
        torch.save({
            "model_type": "prob5",
            "hidden_sizes": self.hidden_sizes,
            "input_size": self.input_size,
            "raw_logits": self.raw_logits,
            "activation": self.activation,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def width_expand(cls, source, new_hidden_sizes):
        """Expand each hidden layer to a wider size, tiling weights."""
        if len(new_hidden_sizes) != len(source.hidden_sizes):
            raise ValueError(f"Layer count mismatch: {len(source.hidden_sizes)} vs {len(new_hidden_sizes)}")
        expanded = cls(new_hidden_sizes, source.input_size, source.activation)
        src_sd = source.state_dict()
        dst_sd = expanded.state_dict()
        with torch.no_grad():
            for i, (s_out, d_out) in enumerate(zip(source.hidden_sizes, new_hidden_sizes)):
                for suffix in ['weight', 'bias']:
                    sk = f'trunk.{i*2}.{suffix}'
                    sw = src_sd[sk]
                    dw = dst_sd[sk]
                    if suffix == 'weight':
                        s_in = sw.shape[1]
                        d_in = dw.shape[1]
                        dw.zero_()
                        dw[:s_out, :s_in] = sw
                        # Tile extra rows
                        for r in range(s_out, d_out):
                            dw[r, :s_in] = sw[r % s_out]
                        # Tile extra cols (from expanded previous layer)
                        if d_in > s_in:
                            dw[:, s_in:d_in] = dw[:, :d_in - s_in]
                        dw += torch.randn_like(dw) * 0.01
                    else:
                        dw[:s_out] = sw
                        for r in range(s_out, d_out):
                            dw[r] = sw[r % s_out]
                        dw += torch.randn(d_out) * 0.01
            # Head
            s_prev = source.hidden_sizes[-1]
            d_prev = new_hidden_sizes[-1]
            hw = dst_sd['head.weight']
            hw.zero_()
            hw[:, :s_prev] = src_sd['head.weight']
            if d_prev > s_prev:
                hw[:, s_prev:d_prev] = hw[:, :d_prev - s_prev]
            hw += torch.randn_like(hw) * 0.01
            dst_sd['head.bias'] = src_sd['head.bias'].clone()
        expanded.load_state_dict(dst_sd)
        return expanded

    @classmethod
    def depth_expand(cls, source, new_layer_size=None):
        """Add one near-identity hidden layer at the end."""
        last = source.hidden_sizes[-1]
        if new_layer_size is None:
            new_layer_size = last
        if new_layer_size > last:
            raise ValueError(f"new_layer_size ({new_layer_size}) must be <= last hidden ({last})")
        new_hidden = source.hidden_sizes + [new_layer_size]
        expanded = cls(new_hidden, source.input_size, source.activation)
        src_sd = source.state_dict()
        dst_sd = expanded.state_dict()
        with torch.no_grad():
            # Copy existing trunk layers
            for i in range(len(source.hidden_sizes)):
                for suffix in ['weight', 'bias']:
                    sk = f'trunk.{i*2}.{suffix}'
                    dst_sd[sk] = src_sd[sk].clone()
            # New layer: near-identity
            new_idx = len(source.hidden_sizes) * 2
            w = dst_sd[f'trunk.{new_idx}.weight']  # (new_layer_size, last)
            w.zero_()
            eye_size = min(new_layer_size, last)
            w[:eye_size, :eye_size] = torch.eye(eye_size)
            w += torch.randn_like(w) * 0.01
            dst_sd[f'trunk.{new_idx}.bias'] = torch.randn(new_layer_size) * 0.01
            # Head: truncate to new_layer_size
            dst_sd['head.weight'] = src_sd['head.weight'][:, :new_layer_size].clone() + torch.randn(NUM_OUTPUTS, new_layer_size) * 0.01
            dst_sd['head.bias'] = src_sd['head.bias'].clone()
        expanded.load_state_dict(dst_sd)
        return expanded

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = cls(ckpt["hidden_sizes"], ckpt["input_size"], ckpt["activation"],
                  raw_logits=ckpt.get("raw_logits", False))
        net.load_state_dict(ckpt["state_dict"])
        return net


# ── Equity from probabilities ─────────────────────────────────────────────────

def prob_to_equity(p):
    """Convert 5 probability outputs to money equity.

    Outputs: P(win), P(wg), P(wbg), P(lg), P(lbg)
    Equity = 2*P(win) + P(wg) + P(wbg) - P(lg) - P(lbg) - 1
    """
    money_weight = torch.tensor([2, 1, 1, -1, -1], dtype=p.dtype, device=p.device)
    if p.ndim == 1:
        return torch.sum(p * money_weight) - 1
    return torch.sum(p * money_weight, dim=1) - 1


def postprocess(prediction):
    """Enforce logical consistency on probability outputs.

    Ensures: P(wg) <= P(win), P(lg) <= P(lose),
             P(wbg) <= P(wg), P(lbg) <= P(lg)

    Credit: Øystein Schønning-Johansen
    """
    if prediction.ndim == 1:
        prediction[1] = min(prediction[0], prediction[1])
        lose = 1 - prediction[0]
        prediction[3] = min(lose, prediction[3])
        prediction[2] = min(prediction[1], prediction[2])
        prediction[4] = min(prediction[3], prediction[4])
        return prediction

    prediction[:, 1] = torch.min(prediction[:, 0], prediction[:, 1])
    lose = 1 - prediction[:, 0]
    prediction[:, 3] = torch.min(lose, prediction[:, 3])
    prediction[:, 2] = torch.min(prediction[:, 1], prediction[:, 2])
    prediction[:, 4] = torch.min(prediction[:, 3], prediction[:, 4])
    return prediction


def value_state(network, x):
    """Evaluate positions, returning equity."""
    with torch.no_grad():
        pred = network(x)
        if network.raw_logits:
            pred = torch.sigmoid(pred)
        pred = postprocess(pred)
        return prob_to_equity(pred)


# ── TD targets ────────────────────────────────────────────────────────────────

def terminal_target(state):
    """5-element target when mover wins."""
    result = state.game_result()  # 1=normal, 2=gammon, 3=backgammon
    is_gam = float(result >= 2)
    is_bg = float(result >= 3)
    return np.array([1.0, is_gam, is_bg, 0.0, 0.0], dtype=np.float32)


def flip_target(v):
    """Convert opponent's outputs to our TD target."""
    return np.array([
        1 - v[0], v[3], v[4], v[1], v[2],
    ], dtype=np.float32)


# ── 1-ply exact Bellman backup ─────────────────────────────────────────────────

# All 21 distinct dice outcomes with probabilities
_DICE = []
for d1 in range(1, 7):
    for d2 in range(d1, 7):
        _DICE.append(((d1, d2), 1/36 if d1 == d2 else 2/36))


def oneply_target(state, network, encode_fn, _get_legal_plays, _switch_turn,
                  device="cpu"):
    """Compute 1-ply target: exact expectation over all 21 dice outcomes.

    Batches all positions across all 21 dice into a single forward pass
    for GPU efficiency.
    Returns 5-element numpy array.
    """
    # Per-dice data: best move (post-switch) state info, plus probability
    dice_data = []  # list of (prob, mode, payload) where mode is 'pass'/'plays'
    pass_encs = []
    play_encs_per_dice = []  # list of (start_idx, count, terminal_results, plays_states)

    target = np.zeros(NUM_OUTPUTS, dtype=np.float32)

    # Collect all encodings to batch
    pass_indices = []  # (dice_idx, "pass")
    play_indices = []  # (dice_idx, move_idx_within_dice)
    all_pass_encs = []
    all_play_encs = []
    dice_plays_info = {}  # dice_idx -> (terminal_eqs_or_None_per_move, plays_states)

    for dice_idx, (dice, prob) in enumerate(_DICE):
        plays = _get_legal_plays(state, dice)
        if not plays:
            switched = _switch_turn(state)
            if switched.is_game_over():
                t = terminal_target(switched)
                target += prob * flip_target(t)
            else:
                pass_indices.append(dice_idx)
                all_pass_encs.append(encode_fn(switched))
        else:
            switched_states = [_switch_turn(s) for _, s in plays]
            move_terminals = []
            for m_idx, sw in enumerate(switched_states):
                if sw.is_game_over():
                    # Mover wins (we picked the move)
                    move_terminals.append(np.array([
                        1.0,
                        float(sw.game_result() >= 2),
                        float(sw.game_result() >= 3),
                        0.0, 0.0], dtype=np.float32))
                else:
                    move_terminals.append(None)
                    play_indices.append((dice_idx, m_idx))
                    all_play_encs.append(encode_fn(sw))
            dice_plays_info[dice_idx] = (move_terminals, switched_states)

    # Batch forward pass
    play_outputs = None
    pass_outputs = None
    if all_play_encs or all_pass_encs:
        all_encs = all_play_encs + all_pass_encs
        x = torch.tensor(np.stack(all_encs), dtype=torch.float32, device=device)
        with torch.no_grad():
            out = network(x)
            if network.raw_logits:
                out = torch.sigmoid(out)
            # For move selection we use derived equity (single value per move)
            # For target we need the 5 probability outputs (flipped)
            out_np = out.cpu().numpy()
        n_play = len(all_play_encs)
        play_outputs = out_np[:n_play]  # (n_play, 5)
        pass_outputs = out_np[n_play:]  # (n_pass, 5)

    # Compute equity from outputs for move selection
    def _eq_from_probs(probs_arr):
        # Use absolute formula: eq = 2*P(win) + P(wg) + P(wbg) - P(lg) - P(lbg) - 1
        return 2 * probs_arr[..., 0] + probs_arr[..., 1] + probs_arr[..., 2] \
               - probs_arr[..., 3] - probs_arr[..., 4] - 1

    # Process passes
    pass_iter = iter(range(len(pass_indices)))
    pass_lookup = {}
    for i, dice_idx in enumerate(pass_indices):
        pass_lookup[dice_idx] = pass_outputs[i]

    # Process plays: build per-dice arrays
    play_lookup = {}  # (dice_idx, move_idx) -> 5-vector
    for i, (dice_idx, move_idx) in enumerate(play_indices):
        play_lookup[(dice_idx, move_idx)] = play_outputs[i]

    for dice_idx, (dice, prob) in enumerate(_DICE):
        if dice_idx in dice_plays_info:
            move_terminals, switched_states = dice_plays_info[dice_idx]
            # For each move: get either terminal target or flipped network output
            move_targets = []  # 5-vectors from our perspective
            for m_idx, term in enumerate(move_terminals):
                if term is not None:
                    move_targets.append(term)
                else:
                    v_next = play_lookup[(dice_idx, m_idx)]
                    move_targets.append(flip_target(v_next))
            move_targets_arr = np.stack(move_targets)
            # Pick move with best equity (highest, since these are from our view)
            equities = _eq_from_probs(move_targets_arr)
            best_idx = int(np.argmax(equities))
            target += prob * move_targets_arr[best_idx]
        elif dice_idx in pass_lookup:
            v_next = pass_lookup[dice_idx]
            target += prob * flip_target(v_next)
        # else: terminal pass already added above

    return target


# ── Data collection ───────────────────────────────────────────────────────────

def _try_import_c_engine():
    try:
        import sys
        c_path = os.path.join(os.path.dirname(__file__) or ".", "c_engine")
        if c_path not in sys.path:
            sys.path.insert(0, c_path)
        import bg_fast
        return bg_fast
    except ImportError:
        return None


def collect_data(network, num_games, encode_fn, eng=None, oneply=False,
                 teacher=None, oneply_device="cpu"):
    """Play self-play games, return (encodings, targets) arrays.

    With oneply=True, targets are exact 1-ply Bellman backups (averaged over
    all 21 dice) computed at the current state before the move.
    With oneply=False, targets bootstrap from the next state after the move.
    If teacher is provided, it is used for move selection instead of network.
    """
    all_enc, all_tgt = [], []
    _switch_turn = eng.switch_turn if eng else switch_turn
    _BoardState = eng.BoardState if eng else BoardState
    _get_legal_plays = eng.get_legal_plays if eng else get_legal_plays

    for _ in range(num_games):
        state = _BoardState.initial()
        if random.random() < 0.5:
            state = _switch_turn(state)

        while not state.is_game_over():
            # Encode current state
            all_enc.append(encode_fn(state))

            if oneply:
                # 1-ply: exact Bellman backup at current state
                all_tgt.append(oneply_target(
                    state, network, encode_fn, _get_legal_plays, _switch_turn,
                    device=oneply_device))

            # Play a move — use teacher for move selection if provided
            d1, d2 = random.randint(1, 6), random.randint(1, 6)
            plays = _get_legal_plays(state, (d1, d2))
            if plays:
                encoded = np.stack([encode_fn(_switch_turn(s)) for _, s in plays])
                if teacher is not None:
                    with torch.no_grad():
                        # Teacher may use 199 features (cubeful encoder)
                        t_input = torch.tensor(encoded, dtype=torch.float32)
                        if teacher.input_size > encoded.shape[-1]:
                            # Append cubeless cube features [1,0,0]
                            pad = torch.zeros(len(encoded), teacher.input_size - encoded.shape[-1])
                            pad[:, 0] = 1.0
                            t_input = torch.cat([t_input, pad], dim=1)
                        eq = teacher(t_input)
                    idx = torch.argmin(eq).item()
                else:
                    eq = value_state(network, torch.tensor(encoded, dtype=torch.float32))
                    idx = torch.argmin(eq).item()
                _, next_state = plays[idx]
                state = _switch_turn(next_state)
            else:
                state = _switch_turn(state)

            if not oneply:
                # 0-ply: bootstrap from next state
                if state.is_game_over():
                    all_tgt.append(terminal_target(state))
                else:
                    with torch.no_grad():
                        x_next = torch.tensor(encode_fn(state), dtype=torch.float32)
                        v_next = network(x_next).numpy()
                    all_tgt.append(flip_target(v_next))

    return np.array(all_enc), np.array(all_tgt)


def _collect_worker(state_dict, hidden_sizes, input_size, activation, num_games,
                    oneply=False, teacher_path=None):
    os.environ["OMP_NUM_THREADS"] = "1"
    network = ProbNet(hidden_sizes, input_size, activation)
    network.load_state_dict(state_dict)
    network.eval()
    teacher = None
    if teacher_path:
        from model import TDNetwork
        teacher = TDNetwork.load(teacher_path)
        teacher.eval()
    eng = _try_import_c_engine()
    enc_fn = eng.encode_state if eng else encode_state
    return collect_data(network, num_games, enc_fn, eng=eng, oneply=oneply,
                        teacher=teacher)


def _split(total, n):
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n) if base + (1 if i < rem else 0) > 0]


# ── Batch training ────────────────────────────────────────────────────────────

def train_batch(
    num_episodes=100_000, hidden_sizes=None, activation="relu",
    lr=1e-3, end_lr=None, games_per_cycle=1000, batch_size=256,
    save_path=None, save_every=10_000, print_every=1_000,
    network=None, workers=1, device="cpu", pw_weight=1.0, use_mse=False,
    oneply=False, teacher_path=None,
):
    if network is None:
        if hidden_sizes is None:
            hidden_sizes = [80]
        network = ProbNet(hidden_sizes, NUM_FEATURES, activation)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    _raw_logits = network.raw_logits if network else False
    if _raw_logits:
        base_loss_fn = nn.BCEWithLogitsLoss()
        loss_fn_name = "BCEWithLogits"
    elif use_mse:
        base_loss_fn = nn.MSELoss()
        loss_fn_name = "MSE"
    else:
        base_loss_fn = nn.BCELoss()
        loss_fn_name = "BCE"
    loss_weights = torch.tensor([pw_weight, 1, 1, 1, 1], dtype=torch.float32)

    print(f"Prob5 batch TD(0), cubeless money (196 features)", flush=True)
    print(f"  arch: {network.hidden_sizes}, lr: {lr} -> {end_lr}, "
          f"workers: {workers}, device: {device}, pw_weight: {pw_weight}, "
          f"loss: {loss_fn_name}, oneply: {oneply}", flush=True)

    pool = None
    if workers > 1:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(workers)

    total_played = 0
    start_time = time.time()
    next_print = print_every
    next_save = save_every

    try:
        while total_played < num_episodes:
            if end_lr is not None:
                frac = total_played / max(num_episodes, 1)
                current_lr = lr + (end_lr - lr) * frac
            else:
                current_lr = lr
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            batch_games = min(games_per_cycle, num_episodes - total_played)

            # Collect
            t0 = time.perf_counter()
            if pool is not None:
                state_dict = network.state_dict()
                splits = _split(batch_games, workers)
                args_list = [(state_dict, network.hidden_sizes, network.input_size,
                              network.activation, n, oneply, teacher_path) for n in splits]
                results = pool.starmap(_collect_worker, args_list)
                encodings = np.concatenate([r[0] for r in results])
                targets = np.concatenate([r[1] for r in results])
            else:
                eng = _try_import_c_engine()
                enc_fn = eng.encode_state if eng else encode_state
                teacher = None
                if teacher_path:
                    from model import TDNetwork
                    teacher = TDNetwork.load(teacher_path)
                    teacher.eval()
                encodings, targets = collect_data(network, batch_games, enc_fn, eng=eng,
                                                  oneply=oneply, teacher=teacher)
            t_collect = time.perf_counter() - t0

            total_played += batch_games

            # Train
            t1 = time.perf_counter()
            network.to(device)
            x_all = torch.tensor(encodings, dtype=torch.float32, device=device)
            y_all = torch.tensor(targets, dtype=torch.float32, device=device)

            network.train()
            perm = torch.randperm(len(x_all), device=device)
            total_loss = 0.0
            per_output_loss = torch.zeros(NUM_OUTPUTS)
            n_batches = 0
            for i in range(0, len(x_all), batch_size):
                idx = perm[i:i+batch_size]
                pred = network(x_all[idx])
                tgt = y_all[idx]
                if _raw_logits:
                    per_loss = torch.stack([nn.functional.binary_cross_entropy_with_logits(
                        pred[:, j], tgt[:, j]) for j in range(NUM_OUTPUTS)])
                elif use_mse:
                    per_loss = torch.stack([nn.functional.mse_loss(
                        pred[:, j], tgt[:, j]) for j in range(NUM_OUTPUTS)])
                else:
                    per_loss = torch.stack([nn.functional.binary_cross_entropy(
                        pred[:, j], tgt[:, j]) for j in range(NUM_OUTPUTS)])
                loss = (per_loss * loss_weights.to(device)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                with torch.no_grad():
                    for j in range(NUM_OUTPUTS):
                        if _raw_logits:
                            per_output_loss[j] += nn.functional.binary_cross_entropy_with_logits(
                                pred[:, j], y_all[idx, j]).item()
                        elif use_mse:
                            per_output_loss[j] += nn.functional.mse_loss(
                                pred[:, j], y_all[idx, j]).item()
                        else:
                            per_output_loss[j] += nn.functional.binary_cross_entropy(
                                pred[:, j], y_all[idx, j]).item()
                n_batches += 1
            per_output_loss /= max(n_batches, 1)

            if device != "cpu":
                network.to("cpu")
            t_train = time.perf_counter() - t1
            avg_loss = total_loss / max(n_batches, 1)

            if total_played >= next_print or total_played >= num_episodes:
                elapsed = time.time() - start_time
                gps = total_played / elapsed if elapsed > 0 else 0
                spg = len(encodings) / max(batch_games, 1)
                mean_tgt = np.mean(targets, axis=0)
                mean_eq = mean_tgt[0] * (1 + mean_tgt[1] + mean_tgt[2]) - \
                          (1 - mean_tgt[0]) * (1 + mean_tgt[3] + mean_tgt[4])
                pol = per_output_loss
                print(
                    f"Ep {total_played:>7d} | {gps:.0f} g/s | "
                    f"lr {current_lr:.6f} | loss {avg_loss:.4f} | "
                    f"samples {len(encodings)} ({spg:.0f}/g) | "
                    f"collect {t_collect:.1f}s train {t_train:.1f}s | "
                    f"eq {mean_eq:.3f} Pw {mean_tgt[0]:.3f} "
                    f"Pgw {mean_tgt[1]:.3f} Pgl {mean_tgt[3]:.3f}\n"
                    f"  per-output loss: Pw={pol[0]:.4f} Pgw={pol[1]:.4f} "
                    f"Pbw={pol[2]:.4f} Pgl={pol[3]:.4f} Pbl={pol[4]:.4f}",
                    flush=True,
                )
                next_print += print_every

            if save_path and total_played >= next_save:
                path = f"{save_path}_ep{total_played}.pt"
                network.save(path)
                print(f"  -> saved {path}", flush=True)
                next_save += save_every

    finally:
        if pool:
            pool.terminate()
            pool.join()

    if save_path:
        path = f"{save_path}_final.pt"
        network.save(path)
        print(f"  -> saved {path}", flush=True)

    elapsed = time.time() - start_time
    print(f"\nDone: {total_played} episodes, {elapsed:.1f}s "
          f"({total_played/elapsed:.0f} g/s)", flush=True)
    return network


# ── Evaluation vs gnubg ───────────────────────────────────────────────────────

def _board_to_gnubg(state):
    """Convert BoardState to gnubg's 2x25 board format."""
    white_board = [0] * 25
    black_board = [0] * 25
    for i in range(24):
        v = state.points[i]
        if v > 0:
            white_board[i] = v
        elif v < 0:
            black_board[23 - i] = -v
    white_board[24] = state.bar[WHITE]
    black_board[24] = state.bar[BLACK]
    if state.turn == WHITE:
        return [black_board, white_board]
    else:
        return [white_board, black_board]


def _eval_worker(args):
    """Worker for parallel eval vs gnubg."""
    os.environ["OMP_NUM_THREADS"] = "1"
    state_dict, hidden_sizes, input_size, activation, num_games, start_as_white = args

    import gnubg_nn
    network = ProbNet(hidden_sizes, input_size, activation)
    network.load_state_dict(state_dict)
    network.eval()

    eq_total = 0.0
    for i in range(num_games):
        state = BoardState.initial()  # WHITE always goes first
        my_color = WHITE if (start_as_white + i) % 2 == 0 else BLACK

        while not state.is_game_over():
            d1, d2 = random.randint(1, 6), random.randint(1, 6)
            plays = get_legal_plays(state, (d1, d2))
            if plays:
                if state.turn == my_color:
                    encoded = np.stack([encode_state(switch_turn(s)) for _, s in plays])
                    eq = value_state(network, torch.tensor(encoded, dtype=torch.float32))
                    idx = torch.argmin(eq).item()
                    _, next_state = plays[idx]
                else:
                    # gnubg money-optimal
                    best_eq = 999
                    best_next = plays[0][1]
                    for play, ns in plays:
                        switched = switch_turn(ns)
                        board = _board_to_gnubg(switched)
                        probs = gnubg_nn.probabilities(board, 0)
                        w, wg, wbg, lg, lbg = probs
                        eq_val = w - (1 - w) + wg - lg + wbg - lbg
                        if eq_val < best_eq:
                            best_eq = eq_val
                            best_next = ns
                    next_state = best_next
                state = switch_turn(next_state)
            else:
                state = switch_turn(state)

        result = state.game_result()
        if state.winner() == my_color:
            eq_total += result
        else:
            eq_total -= result

    return eq_total, num_games


def eval_vs_gnubg(network, num_games=10000, workers=1):
    """Play money games vs gnubg-nn 0-ply, return equity/game."""
    try:
        import gnubg_nn
    except ImportError:
        print("gnubg-nn not installed (pip install gnubg-nn)", flush=True)
        return None

    print(f"Evaluating vs gnubg-nn 0-ply ({num_games} games, {workers} workers)...",
          flush=True)

    if workers > 1:
        state_dict = network.state_dict()
        splits = _split(num_games, workers)
        args_list = [
            (state_dict, network.hidden_sizes, network.input_size,
             network.activation, n, i)
            for i, n in enumerate(splits)
        ]
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers) as pool:
            results = pool.map(_eval_worker, args_list)
        eq_total = sum(r[0] for r in results)
        total_games = sum(r[1] for r in results)
    else:
        eq_total, total_games = _eval_worker(
            (network.state_dict(), network.hidden_sizes, network.input_size,
             network.activation, num_games, 0))

    eq_per_game = eq_total / total_games
    print(f"\nResult: {total_games} games, eq/game = {eq_per_game:.4f}", flush=True)
    return eq_per_game


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prob5 cubeless money TD(0) (196 features)")
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--hidden", type=int, nargs="+", default=[80])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--end-lr", type=float, default=None)
    parser.add_argument("--games-per-cycle", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--print-every", type=int, default=1_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--expand", type=str, default=None,
                        help="Width-expand from a smaller model (use with --hidden for target size)")
    parser.add_argument("--expand-depth", type=str, default=None,
                        help="Depth-expand: append a near-identity hidden layer")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pw-weight", type=float, default=1.0,
                        help="Weight for P(win) loss relative to other outputs")
    parser.add_argument("--mse", action="store_true",
                        help="Use MSE loss instead of BCE")
    parser.add_argument("--oneply", action="store_true",
                        help="Use 1-ply exact Bellman backups for targets")
    parser.add_argument("--raw-logits", action="store_true",
                        help="Remove sigmoid, output raw logits (train with MSE)")
    parser.add_argument("--teacher", type=str, default=None,
                        help="Path to equity model for move selection during training")
    parser.add_argument("--eval-gnubg", type=int, default=0, metavar="N",
                        help="Evaluate vs gnubg-nn 0-ply for N money games (requires gnubg-nn)")
    args = parser.parse_args()

    # Load / expand / create network
    load_opts = [args.resume, args.expand, args.expand_depth]
    if sum(1 for x in load_opts if x) > 1:
        parser.error("--resume, --expand, and --expand-depth are mutually exclusive")

    network = None
    if args.resume:
        network = ProbNet.load(args.resume)
        print(f"Resumed: {network.hidden_sizes}", flush=True)
    elif args.expand:
        source = ProbNet.load(args.expand)
        network = ProbNet.width_expand(source, args.hidden)
        print(f"Width-expanded: {source.hidden_sizes} -> {network.hidden_sizes} "
              f"({sum(p.numel() for p in source.parameters())} -> "
              f"{sum(p.numel() for p in network.parameters())} params)", flush=True)
    elif args.expand_depth:
        source = ProbNet.load(args.expand_depth)
        new_size = args.hidden[-1] if args.hidden and len(args.hidden) == len(source.hidden_sizes) + 1 else None
        network = ProbNet.depth_expand(source, new_layer_size=new_size)
        print(f"Depth-expanded: {source.hidden_sizes} -> {network.hidden_sizes} "
              f"({sum(p.numel() for p in source.parameters())} -> "
              f"{sum(p.numel() for p in network.parameters())} params)", flush=True)

    # Switch to raw logits mode if requested
    if args.raw_logits and network is not None:
        network.raw_logits = True
        print(f"  raw_logits mode enabled (no sigmoid)", flush=True)

    if args.eval_gnubg > 0:
        if network is None:
            parser.error("--eval-gnubg requires --resume")
        eval_vs_gnubg(network, num_games=args.eval_gnubg, workers=args.workers)
    else:
        train_batch(
            num_episodes=args.episodes, hidden_sizes=args.hidden,
            activation=args.activation, lr=args.lr, end_lr=args.end_lr,
            games_per_cycle=args.games_per_cycle, batch_size=args.batch_size,
            save_path=args.save_path, save_every=args.save_every,
            print_every=args.print_every, network=network,
            workers=args.workers, device=args.device,
            pw_weight=args.pw_weight, use_mse=args.mse,
            oneply=args.oneply, teacher_path=args.teacher,
        )
