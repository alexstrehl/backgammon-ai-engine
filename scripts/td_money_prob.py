"""
td_money_prob196.py -- Batch TD(0) cubeless money training with 5 probability outputs.

Uses the standard 196-feature perspective encoding (no cube features).
Designed to work with the base backgammon-ai-engine codebase.

Outputs (all sigmoid, from on-roll player's perspective):
  0: P(win)
  1: P(gammon | win)
  2: P(backgammon | win)
  3: P(gammon | loss)
  4: P(backgammon | loss)

Equity derived as:
  eq = P(win) * (1 + P(g|w) + P(bg|w)) - (1 - P(win)) * (1 + P(g|l) + P(bg|l))

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

    def __init__(self, hidden_sizes=None, input_size=NUM_FEATURES, activation="relu"):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [80]
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.activation = activation
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
        return torch.sigmoid(self.head(self.trunk(x)))

    def save(self, path):
        torch.save({
            "model_type": "prob5",
            "hidden_sizes": self.hidden_sizes,
            "input_size": self.input_size,
            "activation": self.activation,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = cls(ckpt["hidden_sizes"], ckpt["input_size"], ckpt["activation"])
        net.load_state_dict(ckpt["state_dict"])
        return net


# ── Equity from probabilities ─────────────────────────────────────────────────

def prob_to_equity(p):
    """Convert 5 probability outputs to equity. Works on batches or single."""
    win = p[..., 0]
    win_mult = 1 + p[..., 1] + p[..., 2]
    loss_mult = 1 + p[..., 3] + p[..., 4]
    return win * win_mult - (1 - win) * loss_mult


def value_state(network, x):
    """Evaluate positions, returning equity."""
    with torch.no_grad():
        return prob_to_equity(network(x))


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


def collect_data(network, num_games, encode_fn, eng=None):
    """Play self-play games, return (encodings, targets) arrays."""
    all_enc, all_tgt = [], []
    _switch_turn = eng.switch_turn if eng else switch_turn
    _BoardState = eng.BoardState if eng else BoardState

    for _ in range(num_games):
        state = _BoardState.initial()
        if random.random() < 0.5:
            state = _switch_turn(state)

        while not state.is_game_over():
            all_enc.append(encode_fn(state))

            d1, d2 = random.randint(1, 6), random.randint(1, 6)

            if eng:
                count, features = eng.get_plays_and_features(state, (d1, d2))
                if count > 0:
                    eq = value_state(network, torch.tensor(features, dtype=torch.float32))
                    idx = torch.argmin(eq).item()
                    state = eng.get_chosen_state(idx)
                    eng.switch_turn_inplace(state)
                else:
                    state = _switch_turn(state)
            else:
                plays = get_legal_plays(state, (d1, d2))
                if plays:
                    encoded = np.stack([encode_fn(_switch_turn(s)) for _, s in plays])
                    eq = value_state(network, torch.tensor(encoded, dtype=torch.float32))
                    idx = torch.argmin(eq).item()
                    _, next_state = plays[idx]
                    state = _switch_turn(next_state)
                else:
                    state = _switch_turn(state)

            if state.is_game_over():
                all_tgt.append(terminal_target(state))
            else:
                with torch.no_grad():
                    x_next = torch.tensor(encode_fn(state), dtype=torch.float32)
                    v_next = network(x_next).numpy()
                all_tgt.append(flip_target(v_next))

    return np.array(all_enc), np.array(all_tgt)


def _collect_worker(state_dict, hidden_sizes, input_size, activation, num_games):
    os.environ["OMP_NUM_THREADS"] = "1"
    network = ProbNet(hidden_sizes, input_size, activation)
    network.load_state_dict(state_dict)
    network.eval()
    eng = _try_import_c_engine()
    enc_fn = eng.encode_state if eng else encode_state
    return collect_data(network, num_games, enc_fn, eng=eng)


def _split(total, n):
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n) if base + (1 if i < rem else 0) > 0]


# ── Batch training ────────────────────────────────────────────────────────────

def train_batch(
    num_episodes=100_000, hidden_sizes=None, activation="relu",
    lr=1e-3, end_lr=None, games_per_cycle=1000, batch_size=256,
    save_path=None, save_every=10_000, print_every=1_000,
    network=None, workers=1, device="cpu",
):
    if network is None:
        if hidden_sizes is None:
            hidden_sizes = [80]
        network = ProbNet(hidden_sizes, NUM_FEATURES, activation)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    bce = nn.BCELoss()

    print(f"Prob5 batch TD(0), cubeless money (196 features)", flush=True)
    print(f"  arch: {network.hidden_sizes}, lr: {lr} -> {end_lr}, "
          f"workers: {workers}, device: {device}", flush=True)

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
                              network.activation, n) for n in splits]
                results = pool.starmap(_collect_worker, args_list)
                encodings = np.concatenate([r[0] for r in results])
                targets = np.concatenate([r[1] for r in results])
            else:
                eng = _try_import_c_engine()
                enc_fn = eng.encode_state if eng else encode_state
                encodings, targets = collect_data(network, batch_games, enc_fn, eng=eng)
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
            n_batches = 0
            for i in range(0, len(x_all), batch_size):
                idx = perm[i:i+batch_size]
                pred = network(x_all[idx])
                loss = bce(pred, y_all[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

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
                print(
                    f"Ep {total_played:>7d} | {gps:.0f} g/s | "
                    f"lr {current_lr:.6f} | loss {avg_loss:.4f} | "
                    f"samples {len(encodings)} ({spg:.0f}/g) | "
                    f"collect {t_collect:.1f}s train {t_train:.1f}s | "
                    f"eq {mean_eq:.3f} Pw {mean_tgt[0]:.3f} "
                    f"Pgw {mean_tgt[1]:.3f} Pgl {mean_tgt[3]:.3f}",
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
        state = BoardState.initial()
        my_color = WHITE if (start_as_white + i) % 2 == 0 else BLACK
        if my_color == BLACK:
            state = switch_turn(state)

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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-gnubg", type=int, default=0, metavar="N",
                        help="Evaluate vs gnubg-nn 0-ply for N money games (requires gnubg-nn)")
    args = parser.parse_args()

    network = ProbNet.load(args.resume) if args.resume else None

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
        )
