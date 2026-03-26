"""
td_train.py -- TD(0) self-play training for backgammon.

Implements the training algorithm from Tesauro's TD-Gammon:

    1. Start a game from the initial position.
    2. At each state s_t, compute V(s_t) = network(encode(s_t)).
    3. Backpropagate to get grad_w V(s_t).
    4. The current player picks the move maximizing/minimizing V
       (WHITE maximizes, BLACK minimizes -- greedy self-play).
    5. Observe the next state s_{t+1}.
    6. TD error:  delta_t = V(s_{t+1}) - V(s_t)
       (at terminal state, V(s_T) = 1.0 if WHITE won, 0.0 if BLACK won)
    7. Update weights:  w += alpha * delta_t * grad_w V(s_t)
    8. Repeat until game ends.

The network always outputs P(WHITE wins) from a fixed perspective.
The encoding always represents the board from WHITE's viewpoint.

Note on correctness: each iteration does a fresh forward pass with the
current weights, then immediately backpropagates.  This avoids the
subtle issue of calling backward() on a stale computation graph after
in-place weight modification.
"""

import os
# Single-threaded PyTorch is 2x faster for small single-vector forward passes
# (thread coordination overhead > computation savings for 196->80->1 networks).
# Must be set before importing torch.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import random
import time
from typing import Optional

import numpy as np
import torch

from model import TDNetwork

# ── Engine imports (Python or C, selected by use_py_engine) ──────────────────


def _import_python_engine():
    """Import the pure-Python engine."""
    from backgammon_engine import (
        BoardState, WHITE, BLACK, NUM_CHECKERS,
        get_legal_plays, switch_turn,
    )
    from encoding import encode_state
    return BoardState, WHITE, BLACK, NUM_CHECKERS, get_legal_plays, switch_turn, encode_state


def _import_c_engine():
    """Import the C engine (requires compiled shared library)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "c_engine"))
    from bg_fast import (
        BoardState, WHITE, BLACK, NUM_CHECKERS,
        get_legal_plays, switch_turn, encode_state,
        get_plays_and_features, get_chosen_state,
        switch_turn_inplace, encode_single,
    )
    return BoardState, WHITE, BLACK, NUM_CHECKERS, get_legal_plays, switch_turn, encode_state


# Default to Python engine at module level
from backgammon_engine import (
    BoardState, WHITE, BLACK, NUM_CHECKERS,
    get_legal_plays, switch_turn,
)
from encoding import encode_state


def _pick_best_play(network, state, plays, encode_fn=None):
    """Greedy move selection with perspective encoding.

    Resulting states from get_legal_plays have turn == mover (unchanged).
    But after this move, it's the OPPONENT's turn.  We must switch_turn
    before encoding so the network evaluates from the correct perspective.

    After switch_turn, encode_state gives the opponent's view, so
    V = P(opponent wins).  The mover wants to MINIMIZE this = argmin.
    """
    if encode_fn is None:
        encode_fn = encode_state
    encoded = np.stack([
        encode_fn(switch_turn(s)) for _, s in plays
    ])
    x = torch.tensor(encoded, dtype=torch.float32)

    with torch.no_grad():
        values = network(x)

    # V = P(opponent wins), mover wants to minimize
    idx = torch.argmin(values).item()
    return plays[idx]


_pick_best_play_fast_fn = None  # cached import, set by train() when fast=True


def _pick_best_play_fast(network, state, dice):
    """Greedy move selection using the C engine's optimized API.

    Does move generation + encoding in a single C call.
    Returns (best_index, num_plays) or (-1, 0) if no legal plays.
    The caller uses get_chosen_state(best_index) to get the result.

    The C engine's get_legal_plays_encoded switches turn before encoding,
    so V = P(opponent wins).  Mover wants argmin.
    """
    count, features = _pick_best_play_fast_fn(state, dice)
    if count == 0:
        return -1, 0

    x = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        values = network(x)

    # V = P(opponent wins) after switch_turn encoding. Mover wants argmin.
    idx = torch.argmin(values).item()
    return idx, count


def train(
    num_episodes: int = 100_000,
    hidden_sizes: Optional[list] = None,
    lr: float = 0.1,
    end_lr: Optional[float] = None,
    save_path: Optional[str] = None,
    save_every: int = 10_000,
    print_every: int = 1_000,
    eval_every: int = 0,       # 0 = disabled
    eval_games: int = 100,
    gnubg_cmd: Optional[str] = None,
    network: Optional[TDNetwork] = None,
    use_py_engine: bool = False,
    activation: str = "sigmoid",
    encoder_name: str = "perspective196",
) -> TDNetwork:
    """Run TD(0) self-play training.

    Args:
        num_episodes:   Number of self-play games.
        hidden_sizes:   List of hidden layer sizes (e.g. [80] or [80, 40]).
                        Ignored if *network* is provided.
        lr:             Starting learning rate (alpha).
        end_lr:         Ending learning rate.  If set, lr is linearly
                        interpolated from lr -> end_lr over the full run.
                        If None, lr is held constant.
        save_path:      Path prefix for periodic model saves (e.g. "models/td").
                        Saves as "{save_path}_ep{N}.pt".
        save_every:     Save model every this many episodes.
        print_every:    Print progress every this many episodes.
        eval_every:     Run gnubg evaluation every this many episodes (0 = off).
        eval_games:     Number of self-play games per gnubg evaluation.
        gnubg_cmd:      Path to gnubg executable (None = auto-detect).
        network:        Optional pre-existing network to continue training.
        use_py_engine:  Use the pure-Python engine instead of the C engine.
        activation:     Activation function: sigmoid, relu, tanh, leaky_relu.
                        Ignored if *network* is provided.

    Returns:
        The trained TDNetwork.
    """
    from encoding import get_encoder
    encoder = get_encoder(encoder_name)

    if network is None:
        network = TDNetwork(
            hidden_sizes=hidden_sizes, activation=activation,
            encoder_name=encoder_name,
        )

    # Auto-create directory for save path if needed
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # Import the appropriate engine
    global _pick_best_play_fast_fn
    if not use_py_engine:
        import sys
        c_engine_path = os.path.join(os.path.dirname(__file__) or ".", "c_engine")
        if c_engine_path not in sys.path:
            sys.path.insert(0, c_engine_path)
        import bg_fast as _ceng
        _BoardState = _ceng.BoardState
        _get_legal_plays = _ceng.get_legal_plays
        _switch_turn = _ceng.switch_turn
        _encode_state = _ceng.encode_state
        _pick_best_play_fast_fn = _ceng.get_plays_and_features
        print("Using C engine (--fast)", flush=True)
    else:
        _BoardState = BoardState
        _get_legal_plays = get_legal_plays
        _switch_turn = switch_turn
        _encode_state = encoder.encode
        print("Using Python engine (--use-py-engine)", flush=True)

    # ── Print training parameters (so log files have them) ──────────
    train_params = {
        "episodes": num_episodes,
        "hidden_sizes": network.hidden_sizes,
        "input_size": network.input_size,
        "activation": network.activation,
        "encoder_name": network.encoder_name,
        "lr": lr,
        "end_lr": end_lr,
        "optimizer": "sgd",
        "save_path": save_path,
        "resumed_from": getattr(network, '_resumed_from', None),
    }
    print("Training parameters:", flush=True)
    for k, v in train_params.items():
        print(f"  {k}: {v}", flush=True)
    print(flush=True)

    wins = {WHITE: 0, BLACK: 0}
    total_moves = 0
    start_time = time.time()

    # Per-phase timing accumulators (reset each print interval).
    # Tracks where wall time goes within the training loop.
    _t = {"movegen": 0.0, "encode_eval": 0.0, "fwd_bwd": 0.0,
          "td_update": 0.0}
    _t_last_reset = 0  # episode count at last reset

    # Loss accumulator: mean squared TD error since last print
    _loss_sum = 0.0
    _loss_count = 0

    for episode in range(1, num_episodes + 1):

        # ── Compute current learning rate ─────────────────────────────
        if end_lr is not None and num_episodes > 1:
            t = (episode - 1) / (num_episodes - 1)
            current_lr = lr + (end_lr - lr) * t
        else:
            current_lr = lr

        # ── Set up a new game ────────────────────────────────────────
        state = _BoardState.initial()

        # Randomly decide who goes first (cosmetic -- both sides use
        # the same network, so this just adds variety).
        if random.random() < 0.5:
            state = _switch_turn(state)

        game_moves = 0

        # ── Play one game ────────────────────────────────────────────
        while not state.is_game_over():

            # -- 1. Fresh forward pass for current state ---------------
            _t0 = time.perf_counter()
            x = torch.tensor(_encode_state(state), dtype=torch.float32)
            value = network(x)  # scalar tensor with grad

            # -- 2. Backward to get grad_w V(s_t) ---------------------
            network.zero_grad()
            value.backward()
            _t1 = time.perf_counter()
            _t["fwd_bwd"] += _t1 - _t0

            # -- 3. Pick a move (greedy self-play) ---------------------
            d1, d2 = random.randint(1, 6), random.randint(1, 6)

            if not use_py_engine:
                best_idx, count = _pick_best_play_fast(network, state, (d1, d2))
                if count > 0:
                    state = _ceng.get_chosen_state(best_idx)
                    _ceng.switch_turn_inplace(state)
                else:
                    state = _switch_turn(state)
            else:
                plays = _get_legal_plays(state, (d1, d2))
                if plays:
                    _, next_state = _pick_best_play(network, state, plays, _encode_state)
                    state = _switch_turn(next_state)
                else:
                    state = _switch_turn(state)
            _t2 = time.perf_counter()
            _t["movegen"] += _t2 - _t1

            game_moves += 1

            # -- 4. Compute TD error -----------------------------------
            #
            # With perspective encoding, V(s) = P(on-roll player wins).
            # After switch_turn, the next state is from the OPPONENT's
            # perspective.  V(next) = P(opponent wins).  From the current
            # player's view, that's (1 - V(next)).
            #
            # Terminal: the on-roll player at step 1 either won or lost.
            # If the mover won, target = 1.  If the mover lost, target = 0.
            if state.is_game_over():
                # Who was the mover? The turn has been switched, so the
                # mover is the OPPOSITE of state.turn.
                mover = 1 - state.turn
                # In backgammon the mover always wins at a terminal state
                # (you can only end the game by bearing off your own checkers).
                assert state.winner() == mover, (
                    f"Terminal state has unexpected winner: "
                    f"winner={state.winner()}, mover={mover}, turn={state.turn}"
                )
                target = 1.0
            else:
                # Non-terminal: V(next) is from opponent's perspective.
                # Target for current player = 1 - V(next).
                x_next_np = _encode_state(state)
                with torch.no_grad():
                    x_next = torch.tensor(x_next_np, dtype=torch.float32)
                    target = 1.0 - network(x_next).item()
            _t3 = time.perf_counter()
            _t["encode_eval"] += _t3 - _t2

            td_error = target - value.item()
            _loss_sum += td_error ** 2
            _loss_count += 1

            # -- 5. Update weights (TD(0): direct gradient update) --------
            with torch.no_grad():
                for p in network.parameters():
                    p += current_lr * td_error * p.grad
            _t4 = time.perf_counter()
            _t["td_update"] += _t4 - _t3

        # ── End of game bookkeeping ──────────────────────────────────
        winner = state.winner()
        wins[winner] += 1
        total_moves += game_moves

        # ── Periodic reporting ───────────────────────────────────────
        if episode % print_every == 0:
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed if elapsed > 0 else 0
            avg_moves = total_moves / episode

            # Compute timing breakdown for this interval
            _t_total = sum(_t.values())
            if _t_total > 0:
                pcts = {k: 100 * v / _t_total for k, v in _t.items()}
                timing_str = (
                    f" | movegen {pcts['movegen']:.0f}%"
                    f" fwd/bwd {pcts['fwd_bwd']:.0f}%"
                    f" eval {pcts['encode_eval']:.0f}%"
                    f" update {pcts['td_update']:.0f}%"
                )
            else:
                timing_str = ""

            interval_loss = _loss_sum / _loss_count if _loss_count > 0 else 0.0
            print(
                f"Episode {episode:>7d} | "
                f"W {100 * wins[WHITE] / episode:5.1f}%  "
                f"B {100 * wins[BLACK] / episode:5.1f}% | "
                f"avg moves {avg_moves:.0f} | "
                f"{eps_per_sec:.1f} games/sec | "
                f"lr {current_lr:.5f} | "
                f"loss {interval_loss:.6f}"
                f"{timing_str}",
                flush=True,
            )

            # Reset timing and loss accumulators for next interval
            for k in _t:
                _t[k] = 0.0
            _t_last_reset = episode
            _loss_sum = 0.0
            _loss_count = 0

        # ── Periodic saving ──────────────────────────────────────────
        if save_path and episode % save_every == 0:
            path = f"{save_path}_ep{episode}.pt"
            save_params = {**train_params, "episodes_completed": episode}
            network.save(path, train_params=save_params)
            print(f"  -> saved {path}", flush=True)

        # ── Periodic gnubg evaluation ───────────────────────────────
        if eval_every > 0 and episode % eval_every == 0:
            from td_agent import TDAgent as _TDAgent
            from gnubg_eval import evaluate_with_gnubg
            eval_agent = _TDAgent(network)
            results = evaluate_with_gnubg(
                eval_agent, eval_agent,
                num_games=eval_games,
                gnubg_cmd=gnubg_cmd,
                verbose=False,
            )
            if results["avg_mEMG"] is not None:
                print(
                    f"  -> gnubg eval (ep {episode}): "
                    f"avg mEMG = {results['avg_mEMG']:.1f}  "
                    f"(W={sum(results['white_mEMG'])/len(results['white_mEMG']):.1f} "
                    f"B={sum(results['black_mEMG'])/len(results['black_mEMG']):.1f})",
                    flush=True,
                )
            else:
                print(f"  -> gnubg eval (ep {episode}): no results (gnubg error?)",
                      flush=True)

    # ── Final save ───────────────────────────────────────────────────
    if save_path:
        path = f"{save_path}_final.pt"
        save_params = {**train_params, "episodes_completed": num_episodes}
        network.save(path, train_params=save_params)
        print(f"  -> saved {path}", flush=True)

    elapsed = time.time() - start_time
    print(
        f"\nTraining complete: {num_episodes} episodes in {elapsed:.1f}s "
        f"({num_episodes / elapsed:.1f} games/sec)",
        flush=True,
    )
    print(
        f"Final win rates: WHITE {100 * wins[WHITE] / num_episodes:.1f}%  "
        f"BLACK {100 * wins[BLACK] / num_episodes:.1f}%",
        flush=True,
    )

    return network


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TD(0) self-play training")
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--hidden", type=int, nargs='+', default=[40],
                        help="Hidden layer sizes (e.g. --hidden 80 or --hidden 80 40)")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["sigmoid", "relu", "tanh", "leaky_relu", "hardsigmoid"],
                        help="Activation function for hidden layers (default: relu)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Starting learning rate (default: 0.1 for sigmoid, 0.05 for relu/leaky_relu/tanh)")
    parser.add_argument("--end-lr", type=float, default=None,
                        help="Ending learning rate. If set, lr is linearly "
                             "interpolated from --lr to --end-lr over the run.")
    parser.add_argument("--save-path", type=str, default="models/td_model",
                        help="Path prefix for saved models (e.g. models/td_model)")
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--print-every", type=int, default=1_000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved model to resume training from")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Run gnubg evaluation every N episodes (0 = off)")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="Number of self-play games per gnubg evaluation")
    parser.add_argument("--gnubg", type=str, default=None,
                        help="Path to gnubg executable")
    parser.add_argument("--use-py-engine", action="store_true",
                        help="Use pure-Python engine instead of C engine")
    parser.add_argument("--encoder", type=str, default="perspective196",
                        help="Encoder name (default: perspective196)")
    args = parser.parse_args()

    # Auto-select learning rate based on activation if not specified
    if args.lr is None:
        if args.activation == "sigmoid":
            args.lr = 0.1
        else:
            args.lr = 0.05  # ReLU/tanh/leaky_relu need lower lr
        print(f"Auto-selected lr={args.lr} for activation={args.activation}")

    if args.resume and args.resume_extend:
        parser.error("--resume and --resume-extend are mutually exclusive")

    if args.resume:
        print(f"Resuming from {args.resume}")
        network = TDNetwork.load(args.resume)
        if hasattr(network, '_train_params') and network._train_params:
            print(f"Previous training params: {network._train_params}")
        # Auto-detect encoder from checkpoint if not explicitly specified
        if args.encoder == "perspective196" and hasattr(network, 'encoder_name'):
            if network.encoder_name != "perspective196":
                print(f"Auto-detected encoder '{network.encoder_name}' from checkpoint")
                args.encoder = network.encoder_name
    else:
        network = None

    train(
        num_episodes=args.episodes,
        hidden_sizes=args.hidden,
        lr=args.lr,
        end_lr=args.end_lr,
        save_path=args.save_path,
        save_every=args.save_every,
        print_every=args.print_every,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        gnubg_cmd=args.gnubg,
        network=network,
        use_py_engine=args.use_py_engine,
        activation=args.activation,
        encoder_name=args.encoder,
    )
