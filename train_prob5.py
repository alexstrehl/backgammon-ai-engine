#!/usr/bin/env python3
"""train_prob5.py -- Batch TD(0) self-play training for a 5-output
probability network (cubeless money).

Outputs (on-roll player's perspective; all sigmoid):
  0: P(win)
  1: P(win gammon)
  2: P(win backgammon)
  3: P(lose gammon)
  4: P(lose backgammon)

Money equity (:func:`model.prob5_to_equity`):
    eq = 2*P(win) + P(wg) + P(wbg) - P(lg) - P(lbg) - 1

TD targets: for a non-terminal next state ``s'`` seen from the
opponent's view, ``v = net(s')``, the mover's target is
    [1 - v[0], v[3], v[4], v[1], v[2]]
(swap the win/lose roles). A terminal state where the mover just won
gives ``[1, is_g, is_bg, 0, 0]``.

0-ply: the target bootstraps from the chosen next state. 1-ply: the
target is the dice-expected value of the best move over all 21 dice
outcomes (:func:`_oneply_target_vec`).

Usage mirrors ``train_batch.py``::

    python3 train_prob5.py --hidden 512,512,256 \
        --num-episodes 200000 --lr 1e-3 --save working_models/prob5
"""

# Pin BLAS / PyTorch threads before importing torch.
import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import atexit
import math
import multiprocessing as mp
import random
import shutil
import sys
import tempfile
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
from model import (ProbNetwork, prob5_to_equity, prob5_postprocess,
                   suggested_depth_expand_eps)
import row_transport
from c_api.puretd_py import BACKENDS as CAPI_BACKENDS
from cuda_graph_step import (ENV as GRAPH_ENV, GraphedTrainStep, StepSpec,
                             env_explicit as graph_env_explicit,
                             env_label as graph_env_label,
                             mode_from_env, resolve_mode as resolve_graph_mode)
from trainer import (DEFAULT_MAX_EPISODE_PLIES, CCollectUnavailable,
                     _CWeights, c_collect_library, c_collect_tmpdir)
from train_cli import parse_hidden_sizes, resolve_save_path
from pool_guard import (PoolWatch, WorkerDied, announce_restart,
                        max_pool_restarts, shutdown_pool, GuardedPool)

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

# The same 21 probabilities in _DICE_OUTCOMES order. float32 so the
# product is bit-identical to the per-dice path's scalar `prob * vec`.
_ONEPLY_PROBS32 = np.array(
    [p for _, p in _DICE_OUTCOMES], dtype=np.float32,
)


# ── Target helpers ────────────────────────────────────────────────────


def _terminal_target(next_state: BoardState) -> np.ndarray:
    """5-vector target when the mover just won. ``game_result()`` is
    1/2/3 for single/gammon/backgammon."""
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


def _flip_rows(v: np.ndarray) -> np.ndarray:
    """Vectorised `_flip_vec` over an (N, 5) float32 block; bit-identical
    to applying `_flip_vec` row by row."""
    out = np.empty((v.shape[0], NUM_OUTPUTS), dtype=np.float32)
    out[:, 0] = 1.0 - v[:, 0]
    out[:, 1] = v[:, 3]
    out[:, 2] = v[:, 4]
    out[:, 3] = v[:, 1]
    out[:, 4] = v[:, 2]
    return out


def _prob5_equity_np(m: np.ndarray) -> np.ndarray:
    """Bit-exact numpy mirror of ``model.prob5_to_equity`` for an
    (N, 5) float32 array.

    The statement order is load-bearing: it reproduces torch's fp32
    summation order,
        (((p0 + p4) + p1) + p2) + p3        then ``- 1.0``
    The ``- 1.0`` stays (rounding can move a first-occurrence tie).
    ``tests/test_prob5_oneply_fast.py`` asserts the equality.
    """
    e = m[:, 0] * np.float32(2.0)
    e -= m[:, 4]
    e += m[:, 1]
    e += m[:, 2]
    e -= m[:, 3]
    e -= np.float32(1.0)
    return e


# ── GPU eval server routing (gpu_eval_server.py) ─────────────────────
# One GPU process runs all workers' forwards. Each worker's own net stays
# on the CPU (row-cap fallback only), so no worker builds a CUDA context.
_WORKER_REMOTE = None   # set per worker by _worker_init_gpu in server mode


def _worker_init_gpu(handles):
    """Pool initializer (server mode): claim a shared-memory slot and
    build this worker's RemoteEvaluator. The slot is never returned, so
    the pool must not recycle workers (no maxtasksperchild).
    """
    global _WORKER_REMOTE
    from gpu_eval_server import claim_slot
    _WORKER_REMOTE = claim_slot(handles)   # one server, or a sharded group


def _net_raw(network, xnp, device, remote):
    """Raw network output (pre-sigmoid, pre-postprocess) as a torch
    tensor: through the GPU eval server when `remote` is attached, else a
    local forward. `remote.forward` returns None when the rows exceed the
    shared-memory cap. That falls through to the local path. The caller
    applies sigmoid/postprocess either way.
    """
    if remote is not None:
        out = remote.forward(np.ascontiguousarray(xnp, dtype=np.float32))
        if out is not None:
            return torch.from_numpy(np.ascontiguousarray(out))
    return network(torch.from_numpy(xnp).to(device))


# ── 1-ply exact Bellman backup ────────────────────────────────────────


def _oneply_target_vec(
    state: BoardState,
    network: ProbNetwork,
    encode_fn: Callable[[BoardState], np.ndarray],
    gpe_fn: Callable,
    device: str,
    remote=None,
) -> np.ndarray:
    """Exact 1-ply Bellman target for `state`: the batched `expand_21`
    path when the C engine is in use (`gpe_fn.use_expand21`), else the
    per-dice reference path (also used on an `expand_21` buffer
    overflow). The two are bit-exact.
    """
    if getattr(gpe_fn, "use_expand21", False):
        fast = _oneply_target_vec_expand21(
            state, network, encode_fn, device, remote,
        )
        if fast is not None:
            return fast
    return _oneply_target_vec_perdice(
        state, network, encode_fn, gpe_fn, device, remote,
    )


def _oneply_target_vec_expand21(
    state: BoardState,
    network: ProbNetwork,
    encode_fn: Callable[[BoardState], np.ndarray],
    device: str,
    remote=None,
) -> Optional[np.ndarray]:
    """`_oneply_target_vec_perdice` with the 21 move-gen calls replaced
    by one `bg_fast.expand_21` call plus a numpy segment reduction.
    Bit-exact with the reference path.

    Returns None if `expand_21` is unavailable or overflows its buffers.

    `expand_21` returns, for the 21 dice in `_DICE_OUTCOMES` order:
      * `base_feats` (n_rows, 196) -- encoding of every NON-terminal
        reply, in reply order; a forced-pass dice contributes one row,
        the encoding of `switch_turn(state)`.
      * `counts` (21,) -- replies per dice, always >= 1.
      * `gr` (n_replies,) -- 0 for a non-terminal reply (consumes the
        next `base_feats` row), else the winner's game_result 1/2/3.
    All are fresh copies (no aliasing of a C buffer).

    ROW ORDER IS LOAD-BEARING. The reference path feeds the network all
    forced-pass rows first, and a CPU MLP forward is not bit-invariant
    under a row permutation, so forced-pass rows are gathered to the
    front before the forward and the outputs scattered back.
    """
    exp = _bg_fast.expand_21(state)
    if exp is None:
        return None
    base_feats, counts, gr, n_rows = exp
    n_replies = gr.shape[0]

    nonterm = gr == 0
    # Reply-space segment starts; strictly increasing (counts >= 1), as
    # the reduceat calls below require.
    seg = np.empty(len(_DICE_OUTCOMES), dtype=np.int64)
    seg[0] = 0
    np.cumsum(counts[:-1], out=seg[1:])

    # ── Forced-pass detection (see ROW ORDER above) ──────────────────
    # A forced pass is the only way a dice bucket holds exactly one
    # non-terminal reply encoded as `switch_turn(state)`.
    order = None
    if n_rows and bool((counts == 1).any()):
        single = np.flatnonzero(counts == 1)
        rep = seg[single]
        single_nonterm = nonterm[rep]
        if single_nonterm.any():
            # reply index -> base_feats row index, for non-terminal replies
            row_of = np.cumsum(nonterm) - 1
            rows = row_of[rep[single_nonterm]]
            pass_enc = encode_fn(switch_turn(state))
            is_pass = (base_feats[rows] == pass_enc).all(axis=1)
            if is_pass.any():
                pass_rows = rows[is_pass]
                keep = np.ones(n_rows, dtype=bool)
                keep[pass_rows] = False
                order = np.concatenate([pass_rows, np.flatnonzero(keep)])

    # ── One forward over every non-terminal reply ────────────────────
    if n_rows:
        x_np = base_feats if order is None else base_feats[order]
        with torch.no_grad():
            out = _net_raw(network, x_np, device, remote)
            if network.raw_logits:
                out = torch.sigmoid(out)
            out = prob5_postprocess(out)
        out_np = out.detach().cpu().numpy()
        if order is not None:
            inv = np.empty(n_rows, dtype=np.int64)
            inv[order] = np.arange(n_rows)
            out_np = out_np[inv]
    else:
        out_np = np.empty((0, NUM_OUTPUTS), dtype=np.float32)

    # ── Per-reply mover target vectors ───────────────────────────────
    if n_rows == n_replies:
        # No terminal reply (the common case): skip the scatter.
        moves = _flip_rows(out_np)
    else:
        moves = np.empty((n_replies, NUM_OUTPUTS), dtype=np.float32)
        moves[nonterm] = _flip_rows(out_np)
        term_gr = gr[~nonterm]
        # `_terminal_target` rebuilt from the C game_result (1/2/3).
        term = np.empty((term_gr.shape[0], NUM_OUTPUTS), dtype=np.float32)
        term[:, 0] = 1.0
        term[:, 1] = term_gr >= 2
        term[:, 2] = term_gr >= 3
        term[:, 3] = 0.0
        term[:, 4] = 0.0
        moves[~nonterm] = term

    # ── Per-dice argmax over derived equity, keeping the WHOLE vector ─
    # Must pick the FIRST maximum like `np.argmax` (ties are frequent):
    # reduce to the max, then to the lowest index hitting it.
    eqs = _prob5_equity_np(moves)
    seg_max = np.maximum.reduceat(eqs, seg)
    hit = eqs == np.repeat(seg_max, counts)
    cand = np.where(hit, np.arange(n_replies), n_replies)
    best = np.minimum.reduceat(cand, seg)
    if best.max() >= n_replies:
        # Only reachable when a bucket's equities contain NaN (no hit):
        # fail now (the one intentional difference from the reference).
        raise ValueError(
            "prob5 1-ply target: network produced non-finite outputs "
            "(NaN equity in at least one dice bucket)"
        )

    # float32 throughout, dice in _DICE_OUTCOMES order: bit-identical to
    # the reference's `target += prob * vec` loop (parity tests).
    target = (moves[best] * _ONEPLY_PROBS32[:, None]).sum(axis=0)
    return np.clip(target, 0.0, 1.0)


def _oneply_target_vec_perdice(
    state: BoardState,
    network: ProbNetwork,
    encode_fn: Callable[[BoardState], np.ndarray],
    gpe_fn: Callable,
    device: str,
    remote=None,
) -> np.ndarray:
    """Reference 1-ply target: 21 separate move-gen calls. The definition
    of correctness for `_oneply_target_vec_expand21` and its fallback;
    don't change the arithmetic without re-running
    `tests/test_prob5_oneply_fast.py`.
    """
    target = np.zeros(NUM_OUTPUTS, dtype=np.float32)

    # Separate buffers so row lookup is unambiguous.
    pass_buf: List[np.ndarray] = []
    pass_dice: List[int] = []  # dice_idx for each row in pass_buf
    play_buf: List[np.ndarray] = []
    # play_info[d_idx] = list of (terminal_vec_or_None, play_buf_row_or_None)
    play_info: dict = {}

    for d_idx, (dice, prob) in enumerate(_DICE_OUTCOMES):
        features, next_states = gpe_fn(state, dice)
        if len(next_states) == 0:
            # Forced pass: board unchanged, so the opponent's view can't
            # be terminal.
            pass_dice.append(d_idx)
            pass_buf.append(encode_fn(switch_turn(state)))
            continue

        # Each row of `features` is already the opponent's on-roll view.
        # Terminal successors are detected from the OPP_OFF feature.
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
        with torch.no_grad():
            out = _net_raw(network, np.stack(stack), device, remote)
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
        # else: unreachable; every dice is either a play or a forced pass.

    # fp32 summation can overshoot [0, 1] by ~1e-7, which BCE rejects.
    return np.clip(target, 0.0, 1.0)


# ── Episode collection ───────────────────────────────────────────────


def _collect_one_episode(
    network: ProbNetwork,
    rng: random.Random,
    encode_fn: Callable[[BoardState], np.ndarray],
    gpe_fn: Callable,
    oneply: bool,
    device: str,
    remote=None,
    max_plies: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Play one self-play episode. Returns (encodings Nx196, targets Nx5).
    `max_plies` > 0: an episode reaching turn max_plies + 1 is DISCARDED
    (empty arrays), as trainer.collect_episode's max_episode_plies."""
    state, dice = opening_roll(rng)
    encs: List[np.ndarray] = []
    tgts: List[np.ndarray] = []

    while not state.is_game_over():
        if max_plies > 0 and len(encs) >= max_plies:
            encs = []
            break
        encs.append(encode_fn(state))

        # Move generation + successor encoding in one call. Row i of
        # `features` encodes `next_states[i]` (opponent on roll).
        features, next_states = gpe_fn(state, dice)

        if oneply:
            # 0-ply move selection. MUST run before `_oneply_target_vec`:
            # `features` and `next_states` are only valid until the next
            # `gpe_fn` call.
            if len(next_states):
                with torch.no_grad():
                    opp_probs = _net_raw(network, features, device, remote)
                    if network.raw_logits:
                        opp_probs = torch.sigmoid(opp_probs)
                    opp_probs = prob5_postprocess(opp_probs)
                    # Opponent's-view equity -> the mover picks the min.
                    opp_eq = prob5_to_equity(opp_probs).cpu().numpy()
                # Terminal correction: a winning move gets its exact
                # result (matches ProbAgent._choose_0ply).
                for j in np.flatnonzero(
                        features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD):
                    opp_eq[j] = -float(next_states[int(j)].game_result())
                chosen = next_states[int(np.argmin(opp_eq))]
                # Copy: `_oneply_target_vec` overwrites the pooled
                # successor objects on its next `gpe_fn` call.
                next_state = BoardState(
                    points=list(chosen.points), bar=list(chosen.bar),
                    off=list(chosen.off), turn=chosen.turn,
                )
            else:
                next_state = switch_turn(state)
            tgts.append(_oneply_target_vec(
                state, network, encode_fn, gpe_fn, device, remote,
            ))
            state = next_state
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
                # Terminal correction, as above.
                for j in np.flatnonzero(
                        features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD):
                    opp_eq[j] = -float(next_states[int(j)].game_result())
                best = int(np.argmin(opp_eq))
                next_state = next_states[best]
                if next_state.is_game_over():
                    tgts.append(_terminal_target(next_state))
                    break
                # Target = the opponent's output at `best`, flipped.
                tgts.append(_flip_vec(opp_probs_np[best]))
                state = next_state
            else:
                # Forced pass: board unchanged, so opp_view can't be
                # terminal. Bootstrap from its flipped value.
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

    ``gpe_fn(state, dice) -> (features Nx196, next_states)`` generates
    the moves and encodes every successor (opponent's on-roll view) in
    one call: the C engine (``bg_fast``) for the perspective196 encoder,
    else pure Python. Both produce identical features.

    ``gpe_fn.use_expand21`` tells ``_oneply_target_vec`` whether the
    batched all-21-dice C expansion is available.
    """
    enc = get_encoder(encoder_name)
    fast = _BG_FAST_AVAILABLE and encoder_name == "perspective196"
    encode_fn = _bg_fast.encode_state if fast else _py_encode_state
    impl = _bg_fast.get_legal_plays_encoded if fast else _py_get_legal_plays_encoded

    def gpe_fn(state, dice, _impl=impl, _enc=enc):
        return _impl(state, dice, _enc)

    gpe_fn.use_expand21 = bool(
        fast and getattr(_bg_fast, "_HAS_EXPAND21", False)
    )
    return encode_fn, gpe_fn


# ── Parallel collection workers ──────────────────────────────────────


def _split_episodes(total: int, n_workers: int) -> List[int]:
    base = total // n_workers
    rem = total % n_workers
    return [base + (1 if i < rem else 0) for i in range(n_workers)]


# ── C collector (--c-collect) ─────────────────────────────────────────
# trainer.py's design (experiments/c_collect/SPEC.md): the master
# exports each round's weights once, the workers play the round's
# episodes in bgc_collect (c_api/puretd_collect.c, the prob5 kind). The
# worker args carry trainer._CWeights. The preconditions are prob5's own.
# Rows come back through shared memory (row_transport.py, (n, 5)
# targets; BG_COLLECT_SHM) or the pool pipe.


# The encoders bgc_collect writes prob5 rows for: the cubeless entries of
# the C API encoder table with collect set (puretd_py.encoder_info;
# tests/test_c_collect.py checks they agree).
C_COLLECT_ENCODERS = ("perspective196",)


def c_collect_fallbacks(net: ProbNetwork) -> List[str]:
    """Why the C collector cannot collect for `net`: one reason per unmet
    precondition, [] when it can. train() prints each as a NOTE and keeps
    the Python collector. (Every other train_prob5 collector setting has a
    C twin: play is argmin, raw_logits nets get the sigmoid Python
    applies, and max_episode_plies bounds both alike.)"""
    reasons = []
    from encoding import get_encoder
    if (net.encoder_name not in C_COLLECT_ENCODERS
            or net.input_size != get_encoder(net.encoder_name).num_features):
        reasons.append(
            f"encoder {net.encoder_name!r} with {net.input_size} inputs (the C "
            f"collector writes {' / '.join(C_COLLECT_ENCODERS)} 196-input "
            f"rows only)")
    return reasons


def _c_collect(weights: _CWeights, num_episodes: int, seed: int, oneply: bool,
               max_plies: int):
    """The _collect_worker episode loop in C: `num_episodes` episodes on
    one random.Random(seed) stream through bgc_collect, `max_plies` as in
    _collect_one_episode. Same (encodings (n, 196), targets (n, 5),
    n_discarded) result. With a slab (`weights.slab`, BG_COLLECT_SHM) the rows go into
    shared memory and the result is a row_transport.SlabRows, unless
    they outgrew it (then the (enc, tgt) pair, through the pipe). The
    engine is opened per call (no weight reload in bgc_open; the open is
    < 0.5 ms)."""
    from c_api.puretd_py import Engine, KIND_PROB5
    eng = Engine(weights.path, weights.backend)
    try:
        if eng.kind != KIND_PROB5:
            raise RuntimeError(
                f"C collector: {weights.path} opens as engine kind {eng.kind}, "
                f"not prob5")
        if weights.slab is not None:
            return row_transport.collect_into_slab(
                eng, weights.slab, seed, num_episodes, oneply=oneply,
                max_plies=max_plies)
        return eng.collect(None, seed, num_episodes, oneply=oneply,
                           max_plies=max_plies)
    finally:
        eng.close()


def _collect_worker(args):
    """Worker subprocess: collect N episodes, return concatenated
    (encodings, targets, n_discarded). A _CWeights in the state_dict slot plays the
    episodes in C instead (--c-collect). With a slab in it the result is
    a row_transport.SlabRows (BG_COLLECT_SHM)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    (state_dict, hidden_sizes, input_size, activation, encoder_name,
     raw_logits, num_episodes, seed, oneply, oneply_device, max_plies) = args

    if isinstance(state_dict, _CWeights):
        return _c_collect(state_dict, num_episodes, seed, oneply, max_plies)

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
    n_discarded = 0
    for _ in range(num_episodes):
        enc, tgt = _collect_one_episode(
            net, rng, encode_fn, gpe_fn,
            oneply=oneply, device=oneply_device,
            remote=_WORKER_REMOTE, max_plies=max_plies,
        )
        if len(enc) > 0:
            enc_chunks.append(enc)
            tgt_chunks.append(tgt)
        else:
            n_discarded += 1

    if not enc_chunks:
        return (
            np.empty((0, input_size), dtype=np.float32),
            np.empty((0, NUM_OUTPUTS), dtype=np.float32),
            n_discarded,
        )
    return (np.concatenate(enc_chunks, axis=0),
            np.concatenate(tgt_chunks, axis=0), n_discarded)



# ── Vectorized collection ─────────────────────────────────────────────
# B games per worker run as coroutines in lockstep. Every pending forward
# is fused into ONE batched call per step. The default GPU 1-ply path
# (see resolve_cuda_oneply_collector).


def _oneply_target_vec_expand21_gen(state, encode_fn):
    """Coroutine twin of `_oneply_target_vec_expand21`: identical body
    with the forward replaced by a yield. Keep the two in sync.
    """
    exp = _bg_fast.expand_21(state)
    if exp is None:
        return None
    base_feats, counts, gr, n_rows = exp
    n_replies = gr.shape[0]

    nonterm = gr == 0
    # Reply-space segment starts; strictly increasing (counts >= 1), as
    # the reduceat calls below require.
    seg = np.empty(len(_DICE_OUTCOMES), dtype=np.int64)
    seg[0] = 0
    np.cumsum(counts[:-1], out=seg[1:])

    # ── Forced-pass detection (see ROW ORDER above) ──────────────────
    # A forced pass is the only way a dice bucket holds exactly one
    # non-terminal reply encoded as `switch_turn(state)`.
    order = None
    if n_rows and bool((counts == 1).any()):
        single = np.flatnonzero(counts == 1)
        rep = seg[single]
        single_nonterm = nonterm[rep]
        if single_nonterm.any():
            # reply index -> base_feats row index, for non-terminal replies
            row_of = np.cumsum(nonterm) - 1
            rows = row_of[rep[single_nonterm]]
            pass_enc = encode_fn(switch_turn(state))
            is_pass = (base_feats[rows] == pass_enc).all(axis=1)
            if is_pass.any():
                pass_rows = rows[is_pass]
                keep = np.ones(n_rows, dtype=bool)
                keep[pass_rows] = False
                order = np.concatenate([pass_rows, np.flatnonzero(keep)])

    # ── One forward over every non-terminal reply ────────────────────
    if n_rows:
        x_np = base_feats if order is None else base_feats[order]
        out_np = yield x_np   # forward done by the driver
        if order is not None:
            inv = np.empty(n_rows, dtype=np.int64)
            inv[order] = np.arange(n_rows)
            out_np = out_np[inv]
    else:
        out_np = np.empty((0, NUM_OUTPUTS), dtype=np.float32)

    # ── Per-reply mover target vectors ───────────────────────────────
    if n_rows == n_replies:
        # No terminal reply (the common case): skip the scatter.
        moves = _flip_rows(out_np)
    else:
        moves = np.empty((n_replies, NUM_OUTPUTS), dtype=np.float32)
        moves[nonterm] = _flip_rows(out_np)
        term_gr = gr[~nonterm]
        # `_terminal_target` rebuilt from the C game_result (1/2/3).
        term = np.empty((term_gr.shape[0], NUM_OUTPUTS), dtype=np.float32)
        term[:, 0] = 1.0
        term[:, 1] = term_gr >= 2
        term[:, 2] = term_gr >= 3
        term[:, 3] = 0.0
        term[:, 4] = 0.0
        moves[~nonterm] = term

    # ── Per-dice argmax over derived equity, keeping the WHOLE vector ─
    # Must pick the FIRST maximum like `np.argmax` (ties are frequent):
    # reduce to the max, then to the lowest index hitting it.
    eqs = _prob5_equity_np(moves)
    seg_max = np.maximum.reduceat(eqs, seg)
    hit = eqs == np.repeat(seg_max, counts)
    cand = np.where(hit, np.arange(n_replies), n_replies)
    best = np.minimum.reduceat(cand, seg)
    if best.max() >= n_replies:
        # Only reachable when a bucket's equities contain NaN (no hit):
        # fail now (the one intentional difference from the reference).
        raise ValueError(
            "prob5 1-ply target: network produced non-finite outputs "
            "(NaN equity in at least one dice bucket)"
        )

    # float32 throughout, dice in _DICE_OUTCOMES order: bit-identical to
    # the reference's `target += prob * vec` loop (parity tests).
    target = (moves[best] * _ONEPLY_PROBS32[:, None]).sum(axis=0)
    return np.clip(target, 0.0, 1.0)



def _oneply_target_vec_perdice_gen(state, encode_fn, gpe_fn):
    """Coroutine form of `_oneply_target_vec_perdice`. Yields one (N,196)
    block, resumed with the post-processed (N,5) probabilities. Returns
    the 5-vector target."""
    target = np.zeros(NUM_OUTPUTS, dtype=np.float32)
    pass_buf: List[np.ndarray] = []
    pass_dice: List[int] = []
    play_buf: List[np.ndarray] = []
    play_info: dict = {}

    for d_idx, (dice, prob) in enumerate(_DICE_OUTCOMES):
        features, next_states = gpe_fn(state, dice)
        if len(next_states) == 0:
            pass_dice.append(d_idx)
            pass_buf.append(encode_fn(switch_turn(state)))
            continue
        per_move: List[Tuple[Optional[np.ndarray], Optional[int]]] = []
        term_mask = features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
        for i in range(len(next_states)):
            if term_mask[i]:
                # Read before the next gpe_fn call, so the lazy successor
                # view is still valid.
                per_move.append((_terminal_target(next_states[i]), None))
            else:
                # `features` is a copy, so these rows outlive later
                # gpe_fn calls.
                play_buf.append(features[i])
                per_move.append((None, len(play_buf) - 1))
        play_info[d_idx] = per_move

    total_rows = len(pass_buf) + len(play_buf)
    if total_rows:
        # np.stack copies: sibling games' gpe_fn calls can't alias the
        # yielded block while this coroutine is suspended.
        out_np = yield np.stack(pass_buf + play_buf)
        pass_out = out_np[: len(pass_buf)]
        play_out = out_np[len(pass_buf):]
    else:
        pass_out = np.empty((0, NUM_OUTPUTS), dtype=np.float32)
        play_out = np.empty((0, NUM_OUTPUTS), dtype=np.float32)

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

    return np.clip(target, 0.0, 1.0)



def _oneply_target_vec_gen(state, encode_fn, gpe_fn):
    """Coroutine form of `_oneply_target_vec`: same dispatch (expand_21
    fast path, per-dice fallback), forwards yielded to the driver."""
    if getattr(gpe_fn, "use_expand21", False):
        fast = yield from _oneply_target_vec_expand21_gen(state, encode_fn)
        if fast is not None:
            return fast
    return (yield from _oneply_target_vec_perdice_gen(state, encode_fn, gpe_fn))


def _collect_one_episode_gen(rng, encode_fn, gpe_fn, oneply, input_size,
                             max_plies=0):
    """Coroutine form of `_collect_one_episode`. Yields (N,196) blocks,
    resumed with post-processed (N,5) probabilities. Returns
    (encodings Nx196, targets Nx5); empty when discarded (`max_plies`)."""
    state, dice = opening_roll(rng)
    encs: List[np.ndarray] = []
    tgts: List[np.ndarray] = []

    while not state.is_game_over():
        if max_plies > 0 and len(encs) >= max_plies:
            encs = []
            break
        encs.append(encode_fn(state))

        features, next_states = gpe_fn(state, dice)
        # `next_states` is a lazy view over a shared C buffer that sibling
        # games' gpe_fn calls invalidate: materialise every successor now.
        mat_states = list(next_states)

        if mat_states:
            opp_probs = yield features
            opp_eq = prob5_to_equity(torch.from_numpy(opp_probs)).numpy()
            for j, ns in enumerate(mat_states):
                if ns.is_game_over():
                    opp_eq[j] = -float(ns.game_result())
            best = int(np.argmin(opp_eq))
            next_state = mat_states[best]
        else:
            next_state = switch_turn(state)

        if oneply:
            tgts.append((yield from _oneply_target_vec_gen(
                state, encode_fn, gpe_fn,
            )))
            state = next_state
        else:
            if mat_states:
                if next_state.is_game_over():
                    tgts.append(_terminal_target(next_state))
                    break
                tgts.append(_flip_vec(opp_probs[best]))
                state = next_state
            else:
                opp_view = switch_turn(state)
                p = yield encode_fn(opp_view)[None, :]
                tgts.append(_flip_vec(p[0]))
                state = opp_view

        dice = (rng.randint(1, 6), rng.randint(1, 6))

    if not encs:
        return (
            np.empty((0, input_size), dtype=np.float32),
            np.empty((0, NUM_OUTPUTS), dtype=np.float32),
        )
    return np.stack(encs), np.stack(tgts)



def _vectorized_collect_worker(args):
    """Worker subprocess: drive `batch_games` episodes in lockstep, one
    fused forward per step."""
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # Arity must stay in sync with `_build_vec_args` in train().
    (state_dict, hidden_sizes, input_size, activation, encoder_name,
     raw_logits, num_episodes, seed, batch_games, device, max_plies) = args

    # device == "remote": shared-context mode. The net lives in the eval
    # server, which returns POSTPROCESSED probabilities.
    remote = None
    net = None
    if device == "remote":
        remote = _WORKER_REMOTE
        if remote is None:
            raise RuntimeError(
                "_vectorized_collect_worker got device='remote' but this "
                "process has no eval-server slot (pool initializer missing)")
    else:
        net = ProbNetwork(
            hidden_sizes=hidden_sizes, input_size=input_size,
            activation=activation, encoder_name=encoder_name,
            raw_logits=raw_logits,
        )
        net.load_state_dict(state_dict)
        net.eval()
        net.to(device)

    encode_fn, gpe_fn = _resolve_engine_fns(encoder_name)
    base_rng = random.Random(seed)

    enc_chunks: List[np.ndarray] = []
    tgt_chunks: List[np.ndarray] = []
    started = 0
    n_discarded = 0

    def _store(res):
        nonlocal n_discarded
        enc, tgt = res
        if len(enc) > 0:
            enc_chunks.append(enc)
            tgt_chunks.append(tgt)
        else:
            n_discarded += 1

    def _start_game():
        """Start games until one yields a forward request (or none left)."""
        nonlocal started
        while started < num_episodes:
            started += 1
            g = _collect_one_episode_gen(
                random.Random(base_rng.randint(0, 2**31 - 1)),
                encode_fn, gpe_fn, oneply=True, input_size=input_size,
                max_plies=max_plies,
            )
            try:
                return [g, next(g)]
            except StopIteration as e:
                _store(e.value)      # game ended without any forward
        return None

    slots = []
    while len(slots) < min(batch_games, num_episodes):
        s = _start_game()
        if s is None:
            break
        slots.append(s)

    while slots:
        blocks = [s[1] for s in slots]
        if remote is not None:
            n_total = sum(b.shape[0] for b in blocks)
            if n_total <= remote.max_rows:
                # Assemble the step directly in the shared-memory slot.
                dst = remote.in_arr[:n_total]
                if len(blocks) == 1:
                    dst[...] = blocks[0]
                else:
                    np.concatenate(blocks, axis=0, out=dst)
                out_np = remote.forward_from_slot(n_total)
            else:
                big = np.ascontiguousarray(
                    np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0],
                    dtype=np.float32)
                # Step larger than the slot: split it (rare; may move
                # the last ulp relative to the local path).
                out_np = np.concatenate([
                    remote.forward(big[s:s + remote.max_rows])
                    for s in range(0, big.shape[0], remote.max_rows)
                ], axis=0)
        else:
            big = np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]
            with torch.no_grad():
                out = net(torch.from_numpy(
                    np.ascontiguousarray(big, dtype=np.float32)
                ).to(device))
                if raw_logits:
                    out = torch.sigmoid(out)
                out = prob5_postprocess(out)
            out_np = out.cpu().numpy()

        next_slots = []
        off = 0
        for (g, req) in slots:
            n = len(req)
            chunk = out_np[off:off + n]
            off += n
            try:
                next_slots.append([g, g.send(chunk)])
            except StopIteration as e:
                _store(e.value)
                s = _start_game()
                if s is not None:
                    next_slots.append(s)
        slots = next_slots

    if not enc_chunks:
        return (
            np.empty((0, input_size), dtype=np.float32),
            np.empty((0, NUM_OUTPUTS), dtype=np.float32),
            n_discarded,
        )
    return (np.concatenate(enc_chunks, axis=0),
            np.concatenate(tgt_chunks, axis=0), n_discarded)



def _unique_ckpt_path(path: str) -> str:
    """Return `path`, or the first free `..._rN` variant if it exists
    (a resumed run must not overwrite the earlier run's checkpoints).
    """
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{root}_r{n}{ext}"):
        n += 1
    return f"{root}_r{n}{ext}"


# ── Training loop ─────────────────────────────────────────────────────


def _make_loss_fn(raw_logits: bool, use_mse: bool, weights: torch.Tensor):
    """The prob5 loss: one term per output (BCEWithLogits on a raw-logits
    net, else MSE with `use_mse`, else BCE), each the mean over the batch,
    weighted by `weights` (5,) and averaged. Returns (loss, per_out)."""
    if raw_logits:
        per_output = F.binary_cross_entropy_with_logits
    elif use_mse:
        per_output = F.mse_loss
    else:
        per_output = F.binary_cross_entropy

    def loss_fn(pred, yb):
        per_out = torch.stack([per_output(pred[:, j], yb[:, j])
                               for j in range(NUM_OUTPUTS)])
        return (per_out * weights).mean(), per_out
    return loss_fn


def _train_step(net, optimizer, loss_fn, xb, yb):
    """One optimizer step on a batch on the training device. Returns the
    detached (loss, per_out), without a host sync (the round reads its
    losses back once)."""
    loss, per_out = loss_fn(net(xb), yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach(), per_out.detach()


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
    pipeline_collect: bool = False,
    gpu_eval_server: bool = False,
    vectorized_collect: int = 0,
    vectorized_auto: bool = False,
    vectorized_workers: Optional[int] = None,
    vectorized_shared_context: Optional[bool] = None,
    gpu_eval_servers: Optional[int] = None,
    c_collect: bool = False,
    c_collect_backend: str = "simd",
    metrics_out: Optional[dict] = None,
    c_collect_auto: bool = False,
    max_episode_plies: int = DEFAULT_MAX_EPISODE_PLIES,
) -> None:
    """Round-based batch TD(0) on the 5-output prob network.

    One round = collect ``episodes_per_round`` self-play episodes,
    shuffle, train one epoch in ``batch_size`` slices. Loss is per-output
    BCE (BCEWithLogits if ``net.raw_logits``, MSE if ``use_mse``),
    averaged across outputs with ``pw_weight`` on P(win).

    ``c_collect``: collect in C (bgc_collect, c_api/puretd_collect.c) on
    the ``c_collect_backend`` libpuretd build. The master exports each
    round's weights once to a private dir. The workers play the episodes
    _collect_worker plays for their seed
    (tests/test_c_collect_prob5_trainer.py). Works with
    ``pipeline_collect``, ``save_every`` and the pool restart. With
    workers <= 1 the master collects the round in C on a seed drawn from
    the run RNG (the Python single-process path plays on the run RNG
    itself, so the episodes differ). Falls back to the Python collector,
    with one NOTE per reason, for c_collect_fallbacks(); an error with
    ``vectorized_collect`` or ``gpu_eval_server``. A missing
    libpuretd_<backend>.so is built on demand (trainer.c_collect_library).
    ``c_collect_auto``: True when ``c_collect`` is the CLI default rather
    than an explicit --c-collect; a library that cannot be built or
    loaded then falls back to the Python collector with a NOTE instead of
    raising.

    ``max_episode_plies``: an episode longer than this many turns is
    DISCARDED (its samples dropped, not truncated) on every collector and
    counted in the round line; 0 disables (as trainer.Trainer.train).

    ``BG_TRAIN_CUDA_GRAPH`` (env; unset or 1 = on, the default on a cuda
    device; 0 = the eager step): full batches replay a CUDA graph of the
    train step (cuda_graph_step.py), bitwise equal to the eager step
    (tests/test_train_prob5_step_bitwise.py).

    ``BG_COLLECT_SHM`` (env; unset or 1 = on, the default with the C
    collector and a worker pool; 0 = the pool pipe): the C workers write
    their rows into shared-memory slabs (row_transport.py, as in
    trainer.py) and return a few bytes. On a cuda device the slabs are
    copied straight into the device pool from the pinned arena. Same
    pool, same order: bitwise equal to the pipe
    (tests/test_train_prob5_shm_bitwise.py). ``BG_ROUND_PHASES=1``
    prints the master's seconds per round by phase.

    ``metrics_out``: a dict that gets, per round, the per-batch losses
    ("batch_losses", extended), the per-output loss averages
    ("per_output", one (5,) array per round) and the train-phase seconds
    ("train_times"); with the C collector the row transport
    ("c_collect_transport": "shm", "pipe" or "in-process") and, with
    shm, the slab overflows of the run ("shm_overflows"); under
    BG_ROUND_PHASES=1 one dict of phase seconds per round
    ("round_phases").
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    if net.raw_logits:
        loss_name = "BCEWithLogits"
    elif use_mse:
        loss_name = "MSE"
    else:
        loss_name = "BCE"
    # Moved to `device` in the first train phase (_make_train_step): the
    # device must stay untouched until the collector gates below have run.
    per_output_weights = torch.tensor(
        [pw_weight, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32,
    )

    print(
        f"Prob5 TD(0) | cubeless-money | hidden={net.hidden_sizes} "
        f"| input={net.input_size} | loss={loss_name} | "
        f"lr={lr} -> {end_lr} | pw_weight={pw_weight} | oneply={oneply} "
        f"| workers={workers} | device={device}"
    )

    if c_collect and vectorized_collect:
        raise SystemExit(
            "ERROR: --c-collect and --vectorized-collect are mutually "
            "exclusive (two collectors for the same rounds). Drop one of "
            "the two flags.")
    if c_collect and gpu_eval_server:
        raise SystemExit(
            "ERROR: --c-collect and --gpu-eval-server are mutually exclusive "
            "(the C collector runs its own forwards; the server only serves "
            "the Python collector). Drop one of the two flags.")

    # BG_TRAIN_CUDA_GRAPH / BG_COLLECT_SHM, read (and validated) before any
    # process starts.
    graph_env = mode_from_env()
    shm_env = row_transport.enabled()

    # C collector: preconditions, the library (built on demand), then one
    # probe open (a kind mismatch fails here, before the pool starts).
    c_active = False
    c_dir = None
    c_files = {}            # round index -> exported weights file
    c_label = "--c-collect (default)" if c_collect_auto else "--c-collect"
    if c_collect:
        reasons = c_collect_fallbacks(net)
        for why in reasons:
            print(f"  NOTE: {c_label}: {why}; falling back to the Python "
                  f"collector.", flush=True)
        c_active = not reasons
    if c_active:
        why = c_collect_library(c_collect_backend)
        if why is not None:
            if not c_collect_auto:
                raise CCollectUnavailable(f"--c-collect: {why}")
            print(f"  NOTE: {c_label}: {why}; falling back to the Python "
                  f"collector.", flush=True)
            c_active = False
    if c_active:
        import export_weights
        from c_api.puretd_py import Engine, KIND_PROB5
        c_dir = c_collect_tmpdir()
        try:
            probe = os.path.join(c_dir, "w_probe.bin")
            export_weights.export_live(net, probe)
            eng = Engine(probe, c_collect_backend)
            kind, n_in, desc = eng.kind, eng.input_size, eng.describe
            c_enc = eng.encoder
            eng.close()
            os.remove(probe)
            if kind != KIND_PROB5 or n_in != net.input_size or c_enc != net.encoder_name:
                raise RuntimeError(
                    f"C collector: the export opens as kind {kind} with "
                    f"{n_in} inputs, encoder {c_enc}; this {net.input_size}-input "
                    f"{net.encoder_name} prob5 net needs kind {KIND_PROB5}")
        except BaseException:
            shutil.rmtree(c_dir, ignore_errors=True)
            raise
        print(f"C collection: ON ("
              + ("the default, --no-c-collect for the Python collector; "
                 if c_collect_auto else "")
              + f"bgc_collect on {desc}, "
              f"{max(workers, 1)} process(es), "
              f"{'1-ply' if oneply else '0-ply'} targets, 0-ply play, "
              f"max_episode_plies={max_episode_plies}; weights exported "
              f"once per round to "
              f"{c_dir})", flush=True)

    if vectorized_collect < 0:
        raise SystemExit(
            "ERROR: --vectorized-collect must be >= 1 (B = games per worker)")
    if vectorized_collect and not oneply:
        print("  NOTE: --vectorized-collect ignored (1-ply only)"); vectorized_collect = 0
    if vectorized_collect and workers <= 1:
        if device != "cpu":
            # No silent fallback to single-process CPU collection.
            raise SystemExit(
                "ERROR: the vectorized GPU collector needs --workers > 1 "
                "(it runs B games in lockstep inside each pool worker). "
                "Pass --workers N (benchmarked optimum: 32), or use "
                "--device cpu --pipeline-collect for the CPU collector."
            )
        print("  NOTE: --vectorized-collect ignored (needs --workers > 1)")
        vectorized_collect = 0
    if vectorized_collect and gpu_eval_server:
        print("  NOTE: --gpu-eval-server ignored (--vectorized-collect active)"); gpu_eval_server = False

    # GPU collector setup (see gpu_context.py): allocator env, worker-count
    # default, VRAM preflight. Runs BEFORE pool creation: the POOL ITSELF
    # must be sized to what fits.
    #   * per-worker topology: an AUTO-routed run falls back to the CPU
    #     collector below VEC_MIN_WORKERS usable workers, an explicit
    #     --vectorized-collect is honored down to 2.
    #   * shared topology: below the eval server's minimum free VRAM an
    #     AUTO-routed run falls back, an explicit one proceeds.
    # Every fallback zeroes `vectorized_collect`, which forces
    # `collect_device` to the CPU below.
    vec_shared = False
    if vectorized_collect and device.startswith("cuda") and torch.cuda.is_available():
        import gpu_context as _gc
        if vectorized_shared_context is None:
            vectorized_shared_context = _gc.VEC_SHARED_CONTEXT_DEFAULT
        vec_shared = bool(vectorized_shared_context)
        if _gc.set_allocator_env():
            print(f"  NOTE: PYTORCH_CUDA_ALLOC_CONF={_gc.VEC_ALLOC_CONF} set "
                  f"for the GPU collection workers (inherited by the spawn "
                  f"pool).")
        phys = _gc.physical_cores()
        vec_workers, why = _gc.default_vectorized_workers(
            workers, vectorized_workers, phys,
            hard_cap=(0 if vec_shared else _gc.VEC_MAX_DEFAULT_WORKERS),
        )
        if vec_workers != workers:
            print(
                f"  NOTE: GPU collection workers {workers} -> {vec_workers} "
                f"({why}). 1-ply collection throughput is flat past this "
                f"point"
                + ("" if vec_shared else
                   " and every extra worker is another CUDA context")
                + "; pass --vectorized-workers N to override."
            )
        free_b, total_b = torch.cuda.mem_get_info(torch.device(device))
        if vec_shared:
            # The server's VRAM need is sized from B, and a capped pool
            # gets a larger auto B: re-derive B before the preflight.
            if vectorized_auto and vec_workers != workers:
                rebatched = default_vectorized_batch(episodes_per_round, vec_workers)
                if rebatched != vectorized_collect:
                    print(
                        f"  NOTE: auto batch re-derived for the capped worker "
                        f"count: B={vectorized_collect} -> {rebatched}"
                    )
                    vectorized_collect = rebatched
            need_b = _gc.shared_context_min_free_bytes(vectorized_collect)
            if free_b < need_b:
                if vectorized_auto:
                    print(
                        f"  WARN: vectorized GPU collector disabled (VRAM "
                        f"preflight: {free_b // (1024*1024)} MiB free on "
                        f"{device}, the shared-context eval server needs "
                        f"~{need_b // (1024*1024)} MiB for B="
                        f"{vectorized_collect}) -- falling back to the CPU "
                        f"collector; --device {device} still drives the "
                        f"master training step. Consider --device cpu "
                        f"--pipeline-collect on this card, or a smaller "
                        f"--vectorized-collect B."
                    )
                    vectorized_collect = 0
                    vec_workers = workers
                    vec_shared = False
                else:
                    print(
                        f"  NOTE: {free_b // (1024*1024)} MiB free on {device} "
                        f"is below the ~{need_b // (1024*1024)} MiB the "
                        f"shared-context eval server wants for B="
                        f"{vectorized_collect}; proceeding because "
                        f"--vectorized-collect was explicit (the server "
                        f"start fails loudly if it does not fit)."
                    )
        else:
            spec0 = {
                "net_type": "prob5",
                "hidden_sizes": net.hidden_sizes,
                "input_size": net.input_size,
                "activation": net.activation,
                "encoder_name": net.encoder_name,
                "raw_logits": net.raw_logits,
            }
            rows = _gc.probe_rows(vectorized_collect)
            measured = _gc.probe_worker_vram(spec0, net.state_dict(), rows,
                                             device=device)
            print("  " + _gc.describe_probe(measured, rows))
            per_worker = (_gc.with_probe_margin(measured)
                          if measured is not None else None)
            capped = vectorized_vram_cap(vec_workers, free_b, per_worker)
            floor = VEC_MIN_WORKERS if vectorized_auto else 2
            # The floor applies only when VRAM is the binding constraint,
            # not on a small --workers the operator asked for.
            vram_bound = capped < vec_workers
            if capped < 2 or (vram_bound and capped < floor):
                print(
                    f"  WARN: vectorized GPU collector disabled (VRAM preflight: "
                    f"{free_b // (1024*1024)} MiB free fits {capped} worker "
                    f"context(s), need >= {floor}"
                    + ("" if vectorized_auto else " for an explicit request")
                    + f") -- falling back to the CPU collector; --device "
                    f"{device} still drives the master training step. "
                    f"On this card use the shared-context collector "
                    f"(--vectorized-shared-context, ~1-2 GB) or --device "
                    f"cpu --pipeline-collect."
                )
                vectorized_collect = 0
                vec_workers = workers
            elif capped < vec_workers:
                print(
                    f"  NOTE: VRAM preflight capped GPU collection workers "
                    f"{vec_workers} -> {capped} ({free_b // (1024*1024)} MiB "
                    f"free, {(per_worker or _gc.VEC_FALLBACK_PER_WORKER_BYTES) // (1024*1024)} "
                    f"MiB/worker budget + {_gc.VEC_RESERVE_BYTES // (1024*1024)} "
                    f"MiB reserve). Pool is sized to the cap."
                )
                vec_workers = capped
        if vectorized_collect and vec_workers != workers:
            workers = vec_workers
            if vectorized_auto:
                rebatched = default_vectorized_batch(episodes_per_round, workers)
                if rebatched != vectorized_collect:
                    print(
                        f"  NOTE: auto batch re-derived for the capped worker "
                        f"count: B={vectorized_collect} -> {rebatched}"
                    )
                    vectorized_collect = rebatched
    if vectorized_collect and workers <= 1:
        # Reachable via --vectorized-workers 1: `pool` is None below, so
        # the single-process _collect_one_episode path runs on `device`.
        print(f"  NOTE: vectorized collection needs a worker pool; "
              f"workers={workers} runs the single-process 1-ply collector "
              f"on {device} instead.")
    elif vectorized_collect:
        print(f"  vectorized collection: ON (B={vectorized_collect} games/worker, "
              f"workers={workers}, device={device}, "
              + ("shared context: the eval-server process(es) hold the "
                 "only CUDA context(s), workers stay on the CPU"
                 if vec_shared else
                 f"per-worker context: {workers} CUDA contexts")
              + ")")
    if vectorized_collect and vec_shared and workers > 1:
        # Shared-context collector: each request is one worker's whole
        # step, run as its own forward (bitwise parity with the
        # per-worker path), so no coalescing.
        gpu_eval_server = True
        _shared_server_spec = {
            "net_type": "prob5",
            "hidden_sizes": net.hidden_sizes,
            "input_size": net.input_size,
            "activation": net.activation,
            "encoder_name": net.encoder_name,
            "raw_logits": net.raw_logits,
            "postprocess": "prob5",
        }
    else:
        _shared_server_spec = None

    # ── GPU eval server: one CUDA context total, in the server process ─
    eval_server = None
    _new_server = None
    if gpu_eval_server:
        if workers <= 1:
            print("  NOTE: --gpu-eval-server ignored (needs --workers > 1)")
        elif not oneply:
            print("  NOTE: --gpu-eval-server ignored (only used with --oneply)")
        elif not torch.cuda.is_available():
            print("  NOTE: --gpu-eval-server ignored (no CUDA device)")
        else:
            from gpu_eval_server import EvalServerGroup
            from vectorized_collect import remote_max_rows
            import gpu_context as _gc2
            if _shared_server_spec is not None:
                _spec = _shared_server_spec
                _max_rows = remote_max_rows(vectorized_collect)
                _coalesce = False
                _n_srv = _gc2.default_eval_servers(workers, gpu_eval_servers)
            else:
                _n_srv = 1
                _spec = {
                    "net_type": "prob5",
                    "hidden_sizes": net.hidden_sizes,
                    "input_size": net.input_size,
                    # Not a tensor, so load_state_dict can't catch a
                    # mismatch.
                    "activation": net.activation,
                    "encoder_name": net.encoder_name,
                    "raw_logits": net.raw_logits,
                }
                _max_rows = int(os.environ.get("GPU_EVAL_MAX_ROWS", "8192"))
                _coalesce = True
            def _new_server():
                # Factory for pool_guard recovery: the rebuilt server
                # starts from the CURRENT weights.
                srv = EvalServerGroup(
                    mp.get_context("spawn"),
                    num_slots=workers,
                    n_servers=_n_srv,
                    n_feat=net.input_size,
                    out_dim=NUM_OUTPUTS,
                    max_rows=_max_rows,
                    build_spec=_spec,
                    init_state_dict=net.state_dict(),
                    # Follow the master's cuda:N. A CPU master puts the
                    # server on the default GPU.
                    device=device if device.startswith("cuda") else "cuda",
                    coalesce=_coalesce,
                )
                srv.start()
                return srv
            eval_server = _new_server()
            print(f"  GPU eval server: ON ({eval_server.n_servers} process(es), "
                  f"{workers} slots x {_max_rows} rows, fp32 cuda, "
                  f"{'per-request forwards' if not _coalesce else 'coalesced forwards'}, "
                  f"net spec: hidden={net.hidden_sizes} "
                  f"act={net.activation} raw_logits={net.raw_logits})")

    # Worker-side 1-ply inference device: non-CPU only through the
    # vectorized collector or the eval server; otherwise collection runs
    # on the CPU and `device` drives the master training step only.
    collect_device = device
    if oneply and not vectorized_collect and collect_device != "cpu":
        collect_device = "cpu"
        if eval_server is None and not c_active:
            print(f"  NOTE: 1-ply collection runs on the CPU collector; "
                  f"--device {device} drives the master training step only.")

    def _new_pool(srv):
        if srv is not None:
            # maxtasksperchild MUST stay unset: each worker claims a slot
            # once in _worker_init_gpu and never returns it. A worker that
            # DIES is still replaced. PoolWatch turns that into WorkerDied.
            return GuardedPool(
                context=mp.get_context("spawn"),
                processes=workers,
                initializer=_worker_init_gpu,
                initargs=(srv.client_handles(),),
            )
        return (GuardedPool(processes=workers,
                            context=mp.get_context("spawn"))
                if workers > 1 else None)

    pool = _new_pool(eval_server)
    if eval_server is not None:
        # SIGTERM runs neither `finally` nor atexit, which would leave
        # the server alive holding VRAM. Order matters: pool first (else
        # workers block on a dead server's events), then server, then
        # os._exit so `finally` can't re-join a dead pool.
        import signal as _signal

        def _shutdown(signum, frame):
            try:
                shutdown_pool(pool)
            except Exception:
                pass
            try:
                eval_server.stop()
            except Exception:
                pass
            os._exit(128 + signum)

        for _sig in (_signal.SIGTERM, _signal.SIGINT):
            try:
                _signal.signal(_sig, _shutdown)
            except (ValueError, OSError):
                pass    # not on the main thread

    # Dead-worker detection and recovery (pool_guard). `pool` and
    # `eval_server` are rebound on a restart. `_shutdown` above reads them
    # at call time, so it always tears down the live ones.
    watch = PoolWatch(pool, eval_server) if pool is not None else None
    pool_restarts = 0
    restart_limit = max_pool_restarts()

    rng = random.Random(seed)
    num_rounds = math.ceil(num_episodes / episodes_per_round)
    episodes_played = 0
    t0_all = time.time()

    def _c_export(round_idx: int) -> str:
        """Round `round_idx`'s weights as BGNN, exported once for all its
        workers. A retried round (pool restart) replaces its file
        atomically."""
        path = os.path.join(c_dir, f"w_{round_idx + 1:04d}.bin")
        export_weights.export_live(net, path + ".tmp")
        os.replace(path + ".tmp", path)
        c_files[round_idx] = path
        return path

    def _c_release(round_idx: int) -> None:
        """Remove round `round_idx`'s file once its collection returned.
        Pipelined, the next round's file (submitted before this round
        finished) stays: at most two exist at a time."""
        path = c_files.pop(round_idx, None)
        if path is not None:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    # Worker seeds by round index, for BG_COLLECT_DUMP (popped when the
    # round is collected).
    round_seeds = {}

    def _build_worker_args(size: int, round_idx: int):
        """Worker args carrying the CURRENT weights. Called eagerly when
        pipelining, which makes those targets one round stale."""
        if c_active:
            sd = _CWeights(_c_export(round_idx), c_collect_backend)
        else:
            # .clone() is REQUIRED: on a CPU net `.cpu()` returns a VIEW
            # of the live parameter, which map_async pickles WHILE
            # optimizer.step() mutates it (torn weights).
            sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        # In server mode the worker's local net is the row-cap fallback
        # only and must stay on the CPU. Otherwise `collect_device`.
        dev = "cpu" if eval_server is not None else (
            collect_device if oneply else "cpu")
        sizes = [n for n in _split_episodes(size, workers) if n > 0]
        seeds = [rng.randint(0, 2**31 - 1) for _ in sizes]
        round_seeds[round_idx] = seeds
        # BG_COLLECT_SHM: each C worker gets its own slab.
        per_worker = ([sd._replace(slab=sl)
                       for sl in shm.slabs(round_idx, sizes)]
                      if shm is not None else [sd] * len(sizes))
        return [
            (
                w, net.hidden_sizes, net.input_size, net.activation,
                net.encoder_name, net.raw_logits, n, seed, oneply, dev,
                max_episode_plies,
            )
            for w, n, seed in zip(per_worker, sizes, seeds)
        ]

    # Arity must stay in sync with `_vectorized_collect_worker`
    # (pinned by tests/test_prob5_vectorized_parity.py).
    def _build_vec_args(size: int):
        # No .clone() needed: synchronous pool.map finishes pickling
        # before optimizer.step() can mutate the parameters.
        sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        # "remote": shared-context mode, the worker forwards through its
        # eval-server slot. Otherwise `collect_device`.
        vdev = "remote" if _shared_server_spec is not None else collect_device
        return [
            (
                sd, net.hidden_sizes, net.input_size, net.activation,
                net.encoder_name, net.raw_logits, n,
                rng.randint(0, 2**31 - 1), vectorized_collect, vdev,
                max_episode_plies,
            )
            for n in _split_episodes(size, workers) if n > 0
        ]

    if pipeline_collect and vectorized_collect:
        print("  NOTE: --pipeline-collect disabled (vectorized collection active)")
        pipeline_collect = False

    # ── Pipeline pre-roll ─────────────────────────────────────────────
    # Overlap round N+1's collection with round N's training step.
    # Mutually exclusive with the server (the SystemExit says why).
    if pipeline_collect and eval_server is not None:
        raise SystemExit(
            "ERROR: --pipeline-collect and --gpu-eval-server are mutually "
            "exclusive. The server's between-round weight push requires that "
            "no worker is mid-request, but pipelining keeps the next round's "
            "workers running across that push. Pick one."
        )
    use_pipeline = pipeline_collect and pool is not None
    if pipeline_collect and pool is None:
        print("  NOTE: --pipeline-collect ignored (needs --workers > 1)")
    elif use_pipeline:
        print("  pipelined collection: ON (round N+1 collects during "
              "round N's training step)")

    # Row transport, C collector (row_transport.py, as in trainer.py):
    # shared-memory arenas (one per round in flight; BG_COLLECT_SHM unset
    # or 1, the default), or the pool pipe (BG_COLLECT_SHM=0).
    shm = None
    if c_active and pool is not None and shm_env:
        shm = row_transport.ShmTransport(
            net.input_size, slots=2 if use_pipeline else 1,
            out_dim=NUM_OUTPUTS)
    if c_active:
        if shm is not None:
            transport = "shm"
            how = (f"shared-memory arenas ("
                   + (f"{row_transport.ENV}=1" if row_transport.explicit()
                      else f"the default, {row_transport.ENV}=0 for the "
                           f"pool pipe")
                   + f"; {shm.slots} in flight, a slab per worker; pinned "
                   f"for a device pool)")
        elif pool is not None:
            transport = "pipe"
            how = f"pickled through the pool pipe ({row_transport.ENV}=0)"
        else:
            transport = "in-process"
            how = "in-process (one collector, no transport)"
        print(f"  C collection rows: {how}", flush=True)
        if metrics_out is not None:
            metrics_out["c_collect_transport"] = transport
    if shm_env and row_transport.explicit() and shm is None:
        # The default stays quiet where it does not apply.
        why = ("the Python collector" if not c_active
               else "--workers 1 collects in the master")
        print(f"  NOTE: {row_transport.ENV}=1 ignored ({why}); rows "
              f"{'stay in-process' if c_active else 'go through the pool pipe'}.",
              flush=True)

    # BG_ROUND_PHASES=1: one line per round with the master's time by
    # phase, in seconds (as trainer.py's; experiments/c_collect/results/
    # p11_results.md); also kept in metrics_out["round_phases"].
    phases_on = os.environ.get("BG_ROUND_PHASES") == "1"
    ph = {}

    pending = None
    if use_pipeline:
        pending = pool.map_async(
            _collect_worker,
            _build_worker_args(min(episodes_per_round, num_episodes), 0),
        )

    # ── The train step ────────────────────────────────────────────────
    # _train_step, or under BG_TRAIN_CUDA_GRAPH (unset or 1 = on, the
    # default; 0 = off) its full batches replayed from a CUDA graph
    # (cuda_graph_step.py, as in trainer.py), bitwise equal to it. Built
    # in the first train phase (_make_train_step): the device stays
    # untouched until then. The net still leaves the device after every
    # train phase. graph_step.bind() brings it back into the graph's
    # storage.
    loss_fn = graph_step = None
    _graph_mode = graph_env
    if _graph_mode is not None:
        on_cuda = str(device).startswith("cuda")
        if not on_cuda:
            _graph_mode, why = None, f"device is {device}"
        else:
            # Only the net, the optimizer and the device decide.
            _graph_mode, why = resolve_graph_mode(
                StepSpec(net, optimizer, None, None, device=device),
                _graph_mode)
        if why is None:
            print(f"  NOTE: {graph_env_label()}: full "
                  f"batches replay a CUDA graph of the train step (mode "
                  f"{_graph_mode}) on {device}.", flush=True)
        elif on_cuda or graph_env_explicit():
            # The default stays quiet off cuda.
            print(f"  NOTE: {GRAPH_ENV} ignored ({why}); eager train "
                  f"step.", flush=True)

    def _make_train_step():
        """(loss_fn, graph_step or None) on `device`, with one device copy
        of the loss weights for the run."""
        lf = _make_loss_fn(net.raw_logits, use_mse,
                           per_output_weights.to(device))
        if _graph_mode is None:
            return lf, None
        spec = StepSpec(net, optimizer, lf,
                        lambda xb, yb: _train_step(net, optimizer, lf, xb, yb),
                        device=device)
        return lf, GraphedTrainStep(spec, batch_size, net.input_size,
                                    mode=_graph_mode,
                                    target_shape=(NUM_OUTPUTS,))

    def _collect_round(round_size, round_idx, pending):
        """One round's collection -> (enc_chunks, tgt_chunks, pending,
        worker_rows, n_discarded). Raises WorkerDied (pool_guard) instead of hanging on
        a dead worker. The caller rebuilds the pool and calls it again.
        worker_rows: the rows of each worker's result, in worker order
        (BG_COLLECT_DUMP). BG_COLLECT_SHM: the chunks are views of the
        round's slabs."""
        if pool is None and c_active:
            # Single-process C: the round on a seed drawn from the run RNG.
            weights = _CWeights(_c_export(round_idx), c_collect_backend)
            round_seeds[round_idx] = [rng.randint(0, 2**31 - 1)]
            results = [_c_collect(weights, round_size, round_seeds[round_idx][0],
                                  oneply, max_episode_plies)]
        elif pool is None:
            encode_fn, get_legal_plays_fn = _resolve_engine_fns()
            oneply_dev = collect_device if oneply else "cpu"
            net.eval()
            if oneply and oneply_dev != "cpu":
                net.to(oneply_dev)
            enc_chunks: List[np.ndarray] = []
            tgt_chunks: List[np.ndarray] = []
            for _ in range(round_size):
                enc, tgt = _collect_one_episode(
                    net, rng, encode_fn, get_legal_plays_fn,
                    oneply=oneply, device=oneply_dev,
                    max_plies=max_episode_plies,
                )
                if len(enc) > 0:
                    enc_chunks.append(enc)
                    tgt_chunks.append(tgt)
            if oneply and oneply_dev != "cpu":
                net.to("cpu")
            return (enc_chunks, tgt_chunks, pending,
                    [sum(len(c) for c in enc_chunks)],
                    round_size - len(enc_chunks))
        elif vectorized_collect:
            if eval_server is not None:
                # Push weights between rounds (no request in flight).
                watch.push_weights(net.state_dict())
            results = watch.map(
                _vectorized_collect_worker, _build_vec_args(round_size),
            )
        elif use_pipeline:
            if pending is None:
                # First round after a pool restart: resubmit this round.
                pending = watch.map_async(
                    _collect_worker, _build_worker_args(round_size, round_idx))
            # Submit the NEXT round's collect before training so it
            # overlaps the train step (one-round-stale weights).
            next_pending = None
            _t_submit = time.perf_counter()
            if round_idx + 1 < num_rounds:
                next_size = min(
                    episodes_per_round,
                    num_episodes - (episodes_played + round_size),
                )
                if next_size > 0:
                    next_pending = watch.map_async(
                        _collect_worker,
                        _build_worker_args(next_size, round_idx + 1),
                    )
            ph["submit"] = time.perf_counter() - _t_submit
            results = watch.get(pending)
            pending = next_pending
        else:
            # Push weights before dispatch; safe between rounds only.
            if eval_server is not None:
                watch.push_weights(net.state_dict())
            results = watch.map(_collect_worker,
                               _build_worker_args(round_size, round_idx))
        if shm is not None:
            # Only the pool's C workers get slabs: each result is a
            # SlabRows (-> views of its slab) or an overflow's pipe pair.
            slabs = shm.begin_take(round_idx)
            results = [shm.take(slabs, i, r) for i, r in enumerate(results)]
        enc_chunks = [r[0] for r in results if len(r[0]) > 0]
        tgt_chunks = [r[1] for r in results if len(r[1]) > 0]
        return (enc_chunks, tgt_chunks, pending, [len(r[0]) for r in results],
                sum(int(r[2]) for r in results))

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
            ph.clear()
            t_c0 = time.perf_counter()
            while True:
                try:
                    (enc_chunks, tgt_chunks, pending, worker_rows,
                     n_discarded) = _collect_round(round_size, round_idx, pending)
                    break
                except WorkerDied as err:
                    # Rebuild the pool (and servers, from the current
                    # weights) and redo the round.
                    pool_restarts += 1
                    if pool_restarts > restart_limit:
                        raise SystemExit(
                            f"ERROR: {err} in round {round_idx + 1}; giving "
                            f"up after {restart_limit} pool restart(s) "
                            f"(BG_MAX_POOL_RESTARTS). Resume from the last "
                            f"checkpoint.") from err
                    announce_restart(err, pool_restarts, restart_limit,
                                     round_idx)
                    shutdown_pool(pool)
                    if shm is not None:
                        # New arenas: a killed worker may have been
                        # writing into the old ones.
                        shm.reset()
                    if eval_server is not None:
                        eval_server.stop()
                        eval_server = _new_server()
                    pool = _new_pool(eval_server)
                    watch = PoolWatch(pool, eval_server)
                    pending = None   # its tasks died with the old pool
            t_collect = time.perf_counter() - t_c0
            if c_active:
                _c_release(round_idx)
            seeds_used = round_seeds.pop(round_idx, [])
            if shm is not None:
                shm.observe(sum(worker_rows), round_size)
                # Slabs sized by the first-round guess overflow routinely: quiet.
                if shm.round_overflows and not shm.round_sized_by_guess:
                    print(f"  NOTE: {row_transport.ENV}: "
                          f"{shm.round_overflows}/{len(worker_rows)} "
                          f"worker(s) outgrew their slab in round "
                          f"{round_idx + 1} (rows replayed and sent "
                          f"through the pipe; {shm.overflows} in the run)",
                          flush=True)
                if metrics_out is not None:
                    metrics_out["shm_overflows"] = shm.overflows

            episodes_played += round_size
            discard_note = (f" | discarded {n_discarded} overlong episodes"
                            f" (>{max_episode_plies} plies)"
                            if n_discarded else "")
            if not enc_chunks:
                if n_discarded:
                    print(f"Round {round_idx + 1:4d}/{num_rounds} "
                          f"| eps {episodes_played}/{num_episodes} "
                          f"| pool      0 "
                          f"| DISCARDED {n_discarded}/{round_size} overlong "
                          f"episodes (>{max_episode_plies} plies) -- no "
                          f"usable samples this round", flush=True)
                continue
            # BG_COLLECT_SHM with the pool on a cuda device: the chunks are
            # slab views, copied to the device one by one below. No host
            # concatenate of the encodings unless the round is dumped; the
            # targets (5 floats a row) only for the round log's mean.
            shm_direct = shm is not None and str(device).startswith("cuda")
            _dump_dir = os.environ.get("BG_COLLECT_DUMP") or None
            log_round = bool(log_every) and (round_idx + 1) % log_every == 0
            n = sum(len(t) for t in tgt_chunks)
            _t = time.perf_counter()
            pool_enc = pool_tgt = None
            if not shm_direct or _dump_dir:
                pool_enc = np.concatenate(enc_chunks, axis=0)
            if not shm_direct or _dump_dir or log_round:
                pool_tgt = np.concatenate(tgt_chunks, axis=0)
            ph["assemble"] = time.perf_counter() - _t
            _t = time.perf_counter()
            if _dump_dir:
                # Validation hook (as in trainer.py): dump the round's pool
                # to compare collector topologies elementwise, with the
                # rows and the seed of each worker (seeds: the CPU
                # collectors' _collect_worker seeds, else empty).
                os.makedirs(_dump_dir, exist_ok=True)
                np.savez(os.path.join(_dump_dir, f"round_{round_idx + 1:04d}.npz"),
                         enc=pool_enc, tgt=pool_tgt,
                         worker_rows=np.asarray(worker_rows, dtype=np.int64),
                         worker_seeds=np.asarray(seeds_used, dtype=np.int64))
            ph["dump"] = time.perf_counter() - _t
            # Release the chunks: pool_enc/pool_tgt own a copy. (shm_direct:
            # slab views, consumed by the device copy below; none may
            # outlive the round.)
            slab_chunks = (enc_chunks, tgt_chunks) if shm_direct else None
            enc_chunks = tgt_chunks = None

            # ── Train: shuffle + per-output loss ──────────────────
            t_t0 = time.perf_counter()
            if loss_fn is None:
                loss_fn, graph_step = _make_train_step()
            if graph_step is not None:
                graph_step.bind()       # net.to(device), into the graph's storage
            else:
                net.to(device)
            net.train()
            _t = time.perf_counter()
            ph["bind"] = _t - t_t0
            if shm_direct:
                # Pinned slabs -> the device pool (the tensors the
                # concatenate + .to(device) below gives).
                x_all, y_all = shm.to_device(*slab_chunks, device)
                slab_chunks = None
            else:
                x_all = torch.from_numpy(pool_enc).to(device)
                y_all = torch.from_numpy(pool_tgt).to(device)
            if phases_on and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            _t_perm = time.perf_counter()
            ph["h2d"] = _t_perm - _t
            perm = torch.randperm(n, device=device)
            _t_steps = time.perf_counter()
            ph["perm"] = _t_steps - _t_perm

            per_output_sum = torch.zeros(NUM_OUTPUTS, device=device)
            round_losses = []
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                if graph_step is not None:
                    loss, per_out = graph_step.step(x_all, y_all, idx)
                else:
                    loss, per_out = _train_step(net, optimizer, loss_fn,
                                                x_all[idx], y_all[idx])
                round_losses.append(loss)
                per_output_sum += per_out
            n_batches = len(round_losses)
            # One host sync per round, not per batch: the same
            # float32 values, summed in the same order.
            batch_losses = torch.stack(round_losses).tolist()
            total_loss = 0.0
            for v in batch_losses:
                total_loss += v
            _t = time.perf_counter()
            ph["steps"] = _t - _t_steps

            if graph_step is not None:
                graph_step.unbind()
            net.to("cpu")
            t_train = time.perf_counter() - t_t0
            ph["unbind"] = time.perf_counter() - _t

            avg_loss = total_loss / max(n_batches, 1)
            per_out_avg = (per_output_sum / max(n_batches, 1)).cpu().numpy()
            if metrics_out is not None:
                metrics_out.setdefault("batch_losses", []).extend(batch_losses)
                metrics_out.setdefault("per_output", []).append(per_out_avg)
                metrics_out.setdefault("train_times", []).append(t_train)

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
                    + (f"| collect {t_collect:.3f}s train {t_train:.3f}s "
                       if os.environ.get("BG_ROUND_TIMING") else
                       f"| collect {t_collect:5.1f}s train {t_train:4.1f}s ")
                    + f"| {gps:.0f} g/s" + discard_note
                )
                print(
                    f"    per-output loss: Pw={per_out_avg[0]:.4f} "
                    f"Pgw={per_out_avg[1]:.4f} Pbw={per_out_avg[2]:.4f} "
                    f"Pgl={per_out_avg[3]:.4f} Pbl={per_out_avg[4]:.4f}"
                )
            if phases_on:
                # bind / unbind: the net to the device (graph storage) and
                # back to the CPU, inside the round line's train seconds.
                ph.update(collect=t_collect, rows=n, t_collect0=t_c0,
                          t_steps0=_t_steps,
                          mb=n * (net.input_size + NUM_OUTPUTS) * 4 / 1e6)
                print(f"  phases round {round_idx + 1}: "
                      + " ".join(f"{k} {ph[k]:.4f}" for k in (
                          "collect", "submit", "assemble", "dump", "bind",
                          "h2d", "perm", "steps", "unbind") if k in ph)
                      + f" | rows {n} ({ph['mb']:.1f} MB)", flush=True)
                if metrics_out is not None:
                    metrics_out.setdefault("round_phases", []).append(dict(ph))

            if save_path and save_every > 0 and episodes_played % save_every < round_size:
                # Never overwrite (see _unique_ckpt_path). `--save m.pt`
                # gives m_ep10000.pt.
                _root, _ext = os.path.splitext(save_path)
                ck_path = _unique_ckpt_path(
                    f"{_root}_ep{episodes_played}{_ext or '.pt'}")
                net.save(ck_path)
                print(f"    -> checkpoint saved to {ck_path}")

    finally:
        if pool is not None:
            if sys.exc_info()[0] is None:
                pool.close()
                pool.join()
            else:
                # A failed run may hold a dead or stuck worker: a plain
                # terminate()/join() can block forever (pool_guard).
                shutdown_pool(pool)
        if eval_server is not None:
            # After the pool, so no worker waits on a dead server's event.
            eval_server.stop()
        if c_dir is not None:
            # After the pool: no worker reads a weights file any more.
            shutil.rmtree(c_dir, ignore_errors=True)
        if shm is not None:
            # After the pool: no worker writes a slab any more.
            shm.close()

    if save_path:
        # Same never-overwrite rule as the periodic checkpoints.
        final_path = _unique_ckpt_path(save_path)
        net.save(final_path)
        if final_path != save_path:
            print(f"NOTE: {save_path} already exists -- final network "
                  f"saved to {final_path} instead (never overwrite).")
        else:
            print(f"Saved final network to {final_path}")

    elapsed = time.time() - t0_all
    gps = num_episodes / elapsed if elapsed > 0 else 0.0
    print(f"Done in {elapsed:.1f}s ({gps:.1f} ep/s)")


# ── CLI ───────────────────────────────────────────────────────────────


def _warm_start_equity_trunk(
    path: str,
    target_hidden: List[int],
    activation: str,
    raw_logits: bool,
) -> ProbNetwork:
    """Initialise a fresh prob5 net's leading trunk layers from a cubeless
    EQUITY checkpoint (``TDNetwork``).

    ``hidden_layers.{i}`` is copied to ``trunk.{2*i}`` for the LEADING
    PREFIX of layers whose shapes match. Copying stops at the first
    mismatch. ``head`` is never written. The checkpoint type is
    validated here (``TDNetwork.save`` records no ``model_type``).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    model_type = ckpt.get("model_type")
    if model_type is not None and model_type != "td":
        raise ValueError(
            f"--warm-start-equity-trunk: {path} has model_type="
            f"{model_type!r}; expected a cubeless-money EQUITY (TDNetwork) "
            "checkpoint."
        )
    output_mode = ckpt.get("output_mode")
    if output_mode is not None and output_mode != "equity":
        raise ValueError(
            f"--warm-start-equity-trunk: {path} has output_mode="
            f"{output_mode!r}, not 'equity'."
        )
    src_act = ckpt.get("activation")
    if src_act is not None and src_act != activation:
        raise ValueError(
            f"--warm-start-equity-trunk: {path} was trained with "
            f"activation={src_act!r} but the target net uses "
            f"activation={activation!r}. Trunk weights are not "
            f"transferable across activations; pass --activation "
            f"{src_act} or pick a matching source checkpoint."
        )
    if src_act is None:
        print(f"  WARNING: {path} records no activation; assuming it "
              f"matches the target's ({activation}).")
    src_sd = ckpt.get("state_dict")
    if not src_sd or not any(k.startswith("hidden_layers.") for k in src_sd):
        raise ValueError(
            f"--warm-start-equity-trunk: {path} has no 'hidden_layers.*' "
            "parameters — it is not a TDNetwork checkpoint."
        )

    src_hidden = ckpt.get("hidden_sizes")
    net = ProbNetwork(
        hidden_sizes=target_hidden,
        input_size=ckpt.get("input_size", 196),
        activation=activation,
        encoder_name=ckpt.get("encoder_name", "perspective196"),
        raw_logits=raw_logits,
    )
    dst_sd = net.state_dict()
    new_sd = {k: v.clone() for k, v in dst_sd.items()}

    copied: List[int] = []
    n_copied = 0
    for i in range(len(target_hidden)):
        pairs = [
            (f"hidden_layers.{i}.{suf}", f"trunk.{2 * i}.{suf}")
            for suf in ("weight", "bias")
        ]
        if not all(
            sk in src_sd and dk in dst_sd
            and src_sd[sk].shape == dst_sd[dk].shape
            for sk, dk in pairs
        ):
            # Leading prefix only: a layer re-matching after a mismatch
            # would sit on random-init layers and get garbage activations.
            break
        for sk, dk in pairs:
            new_sd[dk] = src_sd[sk].clone()
            n_copied += new_sd[dk].numel()
        copied.append(i)
    net.load_state_dict(new_sd)

    # Summary, so a misconfigured warm start is visible.
    total = sum(v.numel() for v in new_sd.values())
    print(f"Warm-started trunk from EQUITY checkpoint {path}")
    print(f"  source hidden={src_hidden} -> target hidden={target_hidden}")
    for i in range(len(target_hidden)):
        dst_shape = tuple(dst_sd[f"trunk.{2 * i}.weight"].shape)
        sk = f"hidden_layers.{i}.weight"
        src_shape = tuple(src_sd[sk].shape) if sk in src_sd else None
        if i in copied:
            print(f"  layer {i} -> trunk.{2 * i}: COPIED {dst_shape}")
        elif src_shape == dst_shape:
            print(f"  layer {i} -> trunk.{2 * i}: FRESH (shapes match, "
                  "but an earlier layer mismatched -- only a leading "
                  "prefix is copied)")
        else:
            print(f"  layer {i} -> trunk.{2 * i}: FRESH "
                  f"(source {src_shape} vs target {dst_shape})")
    print("  head (5-output): FRESH — never copied from a 1-output "
          "equity head")
    print(f"  => {n_copied}/{total} params warm-started "
          f"({100.0 * n_copied / total:.1f}%)")

    if not copied:
        raise ValueError(
            "--warm-start-equity-trunk copied ZERO layers — refusing to "
            "train from what would be a random init. Check that --hidden "
            f"is compatible with the source architecture {src_hidden}."
        )
    return net


def _build_network(args) -> ProbNetwork:
    """Build the net from --resume / --expand / --expand-depth /
    --warm-start-equity-trunk, or fresh."""
    sources = [args.resume, args.expand, args.expand_depth,
               args.warm_start_equity_trunk]
    if sum(1 for s in sources if s) > 1:
        raise ValueError(
            "--resume, --expand, --expand-depth, --warm-start-equity-trunk "
            "are mutually exclusive"
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
        if args.expand_depth_eps is None:
            w = args.expand_depth_size or src.hidden_sizes[-1]
            raise SystemExit(
                "ERROR: --expand-depth requires --expand-depth-eps. There is "
                "no default on purpose -- the safe value is net-specific "
                "(DEPTH_EXPAND_EPS_SCALE in model.py is a starting point).\n"
                f"  --expand-depth-eps 0.0     exact copy, but symmetric (the new "
                f"layer differentiates slowly)\n"
                f"  --expand-depth-eps {suggested_depth_expand_eps(w):.6f}  suggested "
                f"starting point for width {w}\n"
                "MEASURE the equity deviation on real positions before "
                "spending a training budget on the result."
            )
        net = ProbNetwork.depth_expand(
            src, new_layer_size=args.expand_depth_size, eps=args.expand_depth_eps,
        )
        print(
            f"Depth-expanded from {args.expand_depth}: "
            f"{src.hidden_sizes} -> {net.hidden_sizes} "
            f"(eps={args.expand_depth_eps:g})"
        )
        return net

    if args.warm_start_equity_trunk:
        if target_hidden is None:
            raise ValueError(
                "--warm-start-equity-trunk requires --hidden <target sizes> "
                "(otherwise the default [80] would silently be used)"
            )
        return _warm_start_equity_trunk(
            args.warm_start_equity_trunk, target_hidden,
            args.activation, args.raw_logits,
        )

    if target_hidden is None:
        target_hidden = [80]
    return ProbNetwork(
        hidden_sizes=target_hidden,
        activation=args.activation,
        raw_logits=args.raw_logits,
    )



# ── VRAM preflight for the vectorized GPU collector ───────────────────
# Constants shared with trainer.py; documented in gpu_context.py.
from gpu_context import (  # noqa: E402
    VEC_FALLBACK_PER_WORKER_BYTES as VEC_PER_WORKER_BYTES,
    VEC_MIN_WORKERS,
    VEC_RESERVE_BYTES,
    vectorized_vram_cap,
)


def default_vectorized_batch(episodes_per_round: int, workers: int) -> int:
    """Smallest B that lets a worker finish its round in one fused
    batch: ceil(episodes_per_round / workers).
    """
    if workers <= 0:
        return 0
    return max(1, -(-episodes_per_round // workers))


def c_collect_default_fallbacks(args, net: ProbNetwork) -> List[str]:
    """Why the DEFAULT C collector cannot collect for `net`:
    c_collect_fallbacks(), else a libpuretd that cannot be built or
    loaded (built on demand here). [] when it can, or when the default
    does not apply (an explicit --c-collect, --no-c-collect,
    --vectorized-collect or --gpu-eval-server decides on its own)."""
    if (getattr(args, "c_collect", False) or getattr(args, "no_c_collect", False)
            or args.vectorized_collect or args.gpu_eval_server):
        return []
    reasons = c_collect_fallbacks(net)
    if not reasons:
        why = c_collect_library(args.c_collect_backend)
        if why is not None:
            reasons = [why]
    return reasons


def resolve_cuda_oneply_collector(args, c_fallbacks=None):
    """Pick the collector: the C collector (the default),
    the vectorized GPU collector, the GPU eval server or the Python CPU
    collector.

    ``c_fallbacks``: c_collect_default_fallbacks(), the reasons the
    default C collector cannot collect this run (None: not known, the C
    collector is assumed to apply).

    Returns the notes to print: one per fallback reason, then at most one
    routing note. Mutates only ``args.c_collect``, ``args.c_collect_auto``,
    ``args.vectorized_collect`` and ``args.vectorized_auto``.

    * ``--c-collect`` with ``--no-c-collect``, ``--vectorized-collect``
      with ``--no-vectorized-collect``, and an explicit ``--c-collect``
      with ``--vectorized-collect`` or ``--gpu-eval-server`` are errors.
    * The C collector is the default (``args.c_collect_auto``);
      ``--c-collect`` is accepted (it only makes a missing library an
      error).
    * The default with ``c_fallbacks``: a NOTE per reason, then the
      non-C routing (below), so ``--oneply --device
      cuda`` auto-selects the vectorized GPU collector. An
      explicit ``--c-collect`` keeps train()'s NOTE + Python collector.
    * An explicit ``--vectorized-collect`` or ``--gpu-eval-server`` wins
      over the default (NOTE).
    * The C collector is a CPU collector: ``--oneply --device cuda`` keeps
      it (NOTE) and ``--no-vectorized-collect`` does not change it (NOTE).
    * ``--no-c-collect``: the non-C routing.
      ``--oneply --device cuda`` auto-selects the vectorized GPU collector
      unless an explicit collector choice (``--gpu-eval-server`` /
      ``--pipeline-collect``) wins. ``--no-vectorized-collect`` opts out
      but still refuses per-worker CUDA.
    """
    notes = []
    explicit_c = bool(getattr(args, "c_collect", False))
    no_c = bool(getattr(args, "no_c_collect", False))
    if explicit_c and no_c:
        raise SystemExit(
            "ERROR: --c-collect and --no-c-collect/--python-api are "
            "mutually exclusive")
    if args.vectorized_collect and args.no_vectorized_collect:
        raise SystemExit(
            "ERROR: --vectorized-collect and --no-vectorized-collect "
            "are mutually exclusive")
    if explicit_c and args.vectorized_collect:
        raise SystemExit(
            "ERROR: --c-collect and --vectorized-collect are mutually "
            "exclusive")
    if explicit_c and args.gpu_eval_server:
        raise SystemExit(
            "ERROR: --c-collect and --gpu-eval-server are mutually "
            "exclusive")
    other = ("--vectorized-collect" if args.vectorized_collect
             else "--gpu-eval-server" if args.gpu_eval_server else None)
    default_c = not (explicit_c or no_c or other)
    fallback = default_c and bool(c_fallbacks)
    args.c_collect_auto = default_c and not fallback
    args.c_collect = explicit_c or args.c_collect_auto
    if args.c_collect:
        who = "--c-collect" if explicit_c else "the C collector (default)"
        if args.no_vectorized_collect:
            notes.append(
                f"NOTE: --no-vectorized-collect: {who} collects (a CPU "
                f"collector); add --no-c-collect for the Python CPU "
                f"collector (at 1-ply: --device cpu --pipeline-collect).")
        elif args.oneply and args.device.startswith("cuda"):
            notes.append(
                f"NOTE: {who}: --oneply --device cuda keeps the C collector "
                f"(no auto-routing to the vectorized GPU collector: "
                f"--vectorized-collect or --no-c-collect for it); the master "
                f"training step runs on cuda.")
        return notes
    if other and not no_c:
        notes.append(
            f"NOTE: {other}: replaces the default C collector "
            f"(--c-collect with it is an error).")
        return notes
    route = _non_c_routing(args)
    if fallback:
        where = ("the vectorized GPU collector" if args.vectorized_collect
                 else "the Python collector")
        notes += [f"NOTE: --c-collect (default): {why}; falling back to "
                  f"{where} (the --no-c-collect routing)."
                  for why in c_fallbacks]
    return notes + route


def _non_c_routing(args):
    """resolve_cuda_oneply_collector's routing without the C collector
    (--no-c-collect, or the default ruled out): the notes to print."""
    notes = []
    if not (args.oneply and args.device.startswith("cuda")):
        return notes
    if args.no_vectorized_collect:
        notes.append(
            "NOTE: --no-vectorized-collect with --oneply --device cuda: "
            "per-worker CUDA collection is refused below; use "
            "--device cpu --pipeline-collect for the CPU collector "
            "(or drop --no-vectorized-collect: the shared-context GPU "
            "collector needs only ~1-2 GB of VRAM).")
        return notes
    if args.vectorized_collect:
        return notes
    if args.gpu_eval_server or args.pipeline_collect:
        notes.append(
            "NOTE: --oneply --device cuda would default to the vectorized "
            "GPU collector, but an explicit collector choice "
            "(--gpu-eval-server/--pipeline-collect) was made and wins; "
            "collection forwards stay off the per-worker GPU path. Drop "
            "the flag to use the vectorized collector.")
        return notes
    args.vectorized_collect = default_vectorized_batch(
        args.episodes_per_round, args.workers)
    args.vectorized_auto = True
    notes.append(
        f"AUTO: --oneply --device cuda -> vectorized GPU collector "
        f"(B={args.vectorized_collect} = ceil(episodes-per-round/workers); "
        f"shared-context topology by default: the eval-server process "
        f"holds the only CUDA context, ~1-2 GB of VRAM regardless of "
        f"worker count). Pass --vectorized-collect N to override B; "
        f"--vectorized-per-worker-context for one context per worker "
        f"(its VRAM preflight may cap workers or fall back, and will say "
        f"so); --device cpu --pipeline-collect for the CPU collector.")
    return notes


def reject_cuda_oneply(args):
    """Refuse PER-WORKER ``--oneply`` on a non-CPU device (one CUDA
    context per worker). Fires only when every supported collection path
    for it (the C collector, the vectorized GPU collector, the eval
    server, the pipelined CPU collector) has been declined. 0-ply keeps
    full ``--device cuda`` support.
    """
    if args.oneply and args.device != "cpu":
        raise SystemExit(
            "ERROR: prob5 1-ply training does not support per-worker "
            f"--device {args.device!r}: one CUDA context per worker OOMs "
            "consumer GPUs and measures ~0.13x the CPU collector. Use "
            "the C collector (the default: drop --no-c-collect), the "
            "vectorized GPU collector (drop --no-vectorized-collect; its "
            "shared-context topology needs ~1-2 GB of VRAM on any card), "
            "or run --device cpu --pipeline-collect for the Python CPU "
            "collector. --device cuda remains supported (and "
            "recommended, +82%) for 0-ply training."
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
    g_io.add_argument("--warm-start-equity-trunk", default=None,
                      help="Init leading trunk layers from a cubeless-money "
                           "EQUITY checkpoint (TDNetwork); mismatched deeper "
                           "layers and the 5-head output start fresh")
    g_io.add_argument("--expand", default=None,
                      help="Width-expand from a smaller prob5 checkpoint "
                           "to --hidden.")
    g_io.add_argument("--expand-depth", default=None,
                      help="Depth-expand a prob5 checkpoint by appending "
                           "one hidden layer.")
    g_io.add_argument("--expand-depth-size", type=int, default=None,
                      help="Size of the new layer under --expand-depth.")
    g_io.add_argument("--expand-depth-eps", type=float, default=None,
                      help="REQUIRED with --expand-depth. Symmetry-breaking "
                           "noise on the near-identity init. No default -- "
                           "0.0 is exact but symmetric, injected deviation "
                           "grows as eps*sqrt(width), and the equity cost is "
                           "net-specific. Measure it (see model.py, "
                           "DEPTH_EXPAND_EPS_SCALE).")
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
    g_train.add_argument("--max-episode-plies", type=int,
                         default=DEFAULT_MAX_EPISODE_PLIES, metavar="N",
                         help="Safety bound on self-play episode length, in "
                              "plies (turns). An episode exceeding it is "
                              "DISCARDED (its samples are dropped, not "
                              "truncated) and counted in the round log line. "
                              "Guards against a degenerate policy producing "
                              "non-terminating games; healthy episodes stay "
                              "far below it. 0 disables.")
    g_train.add_argument("--pipeline-collect", action="store_true",
                         help="overlap round N+1's collection with round N's "
                              "training step (needs --workers>1). Targets "
                              "become one round stale.")
    g_train.add_argument("--gpu-eval-server", action="store_true",
                         help="route 1-ply collection forwards to a single "
                              "GPU process that batches across all workers "
                              "(needs --workers>1 and --oneply). Workers stay "
                              "on the CPU, so only ONE CUDA context exists. "
                              "Mutually exclusive with --pipeline-collect.")

    g_train.add_argument("--vectorized-collect", type=int, default=0,
                         metavar="B",
                         help="B games per worker driven in lockstep with "
                              "their forwards fused into one batched call "
                              "(1-ply only), replacing the default C "
                              "collector. Selected automatically by "
                              "--no-c-collect --oneply --device cuda; pass "
                              "a value to override the ceil(epr/workers) "
                              "default.")
    g_train.add_argument("--no-vectorized-collect", action="store_true",
                         help="With --no-c-collect: do not auto-select the "
                              "vectorized GPU collector for --oneply --device "
                              "cuda (the Python CPU collector, --device cpu "
                              "--pipeline-collect, is then the supported "
                              "1-ply path). Without it the C collector "
                              "collects anyway. Mutually exclusive with an "
                              "explicit --vectorized-collect.")
    g_train.add_argument("--vectorized-workers", type=int, default=None,
                         metavar="N",
                         help="Exact number of GPU collection worker "
                              "processes. Default: min(--workers, physical "
                              "cores, 32) -- 1-ply throughput is flat well "
                              "before that and, on the per-worker-context "
                              "collector, every extra worker is another "
                              "CUDA context. The VRAM preflight may still "
                              "cap it lower.")
    g_train.add_argument("--vectorized-shared-context", action="store_true",
                         help="GPU collector topology: ONE CUDA context in an "
                              "eval-server process; workers stay on the CPU "
                              "and ship their fused lockstep batches through "
                              "shared memory. VRAM is independent of worker "
                              "count.")
    g_train.add_argument("--vectorized-per-worker-context",
                         action="store_true",
                         help="GPU collector topology: the net on the GPU in "
                              "every worker process (one CUDA context per "
                              "worker; VRAM grows linearly with workers).")
    # default=SUPPRESS: the help must not print "(default: False)" for
    # the default collector; resolve_cuda_oneply_collector sets the attr.
    g_train.add_argument("--c-collect", action="store_true",
                         default=argparse.SUPPRESS,
                         help="The default collector (this flag is "
                              "accepted for old command lines; it only "
                              "makes a library that cannot be built an "
                              "error): the self-play episodes are "
                              "collected in C (bgc_collect, c_api/; a "
                              "missing libpuretd_<backend>.so is built on "
                              "demand with `bash c_api/build.sh "
                              "<backend>`), the samples the Python "
                              "collector produces for the same seed, 0-ply "
                              "play, 0-/1-ply targets. Works with "
                              "--pipeline-collect, --save-every and "
                              "--workers 1. Ruled out, with a NOTE, for a "
                              "net whose encoder is not perspective196 or "
                              "a library that cannot be built: the default "
                              "then routes as --no-c-collect does (with "
                              "--oneply --device cuda: the vectorized GPU "
                              "collector), an explicit --c-collect keeps "
                              "the Python collector. An explicit "
                              "--vectorized-collect or --gpu-eval-server "
                              "replaces it (with --c-collect: an error); "
                              "--oneply --device cuda keeps it. Rows reach "
                              "the master through shared memory "
                              "(row_transport.py; BG_COLLECT_SHM=0 for the "
                              "pool pipe, same pool, bitwise).")
    g_train.add_argument("--no-c-collect", "--python-api", action="store_true",
                         dest="no_c_collect",
                         help="Collect with the Python collector, or with "
                              "--oneply --device cuda the vectorized GPU "
                              "collector. --python-api "
                              "is the same flag (play_models.py's name).")
    g_train.add_argument("--c-collect-backend", choices=CAPI_BACKENDS,
                         default="simd",
                         help="libpuretd_<backend>.so the C collection "
                              "workers load (c_api/build.sh <backend>).")
    g_train.add_argument("--gpu-eval-servers", type=int, default=None,
                         metavar="N",
                         help="Shared-context collector: number of eval-server "
                              "processes to shard the workers across (each is "
                              "one small CUDA context). Default: one per 16 "
                              "workers.")

    args = p.parse_args()

    args.vectorized_auto = False
    if args.vectorized_shared_context and args.vectorized_per_worker_context:
        raise SystemExit(
            "ERROR: --vectorized-shared-context and "
            "--vectorized-per-worker-context are mutually exclusive")
    shared_ctx = (True if args.vectorized_shared_context
                  else False if args.vectorized_per_worker_context
                  else None)   # None -> gpu_context.VEC_SHARED_CONTEXT_DEFAULT

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)

    net = _build_network(args)

    # The collector, once the network is built: whether the default C
    # collector applies depends on its encoder (a run it rules out gets
    # the non-C routing).
    for note in resolve_cuda_oneply_collector(
            args, c_collect_default_fallbacks(args, net)):
        print(note, flush=True)
    # The C collector (the default), --gpu-eval-server and
    # --pipeline-collect keep worker nets off the GPU, so only a bare
    # 1-ply run without the C collector can reach the per-worker device
    # path.
    if not (args.vectorized_collect or args.gpu_eval_server
            or args.pipeline_collect or args.c_collect):
        reject_cuda_oneply(args)

    save_path = None
    if args.save:
        save_path = resolve_save_path(
            args.save, "cubeless-money", net.hidden_sizes,
        )

    try:
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
            pipeline_collect=args.pipeline_collect,
            gpu_eval_server=args.gpu_eval_server,
            vectorized_collect=args.vectorized_collect,
            vectorized_auto=getattr(args, "vectorized_auto", False),
            vectorized_workers=args.vectorized_workers,
            vectorized_shared_context=shared_ctx,
            gpu_eval_servers=args.gpu_eval_servers,
            c_collect=args.c_collect,
            c_collect_backend=args.c_collect_backend,
            c_collect_auto=args.c_collect_auto,
            max_episode_plies=args.max_episode_plies,
        )
    except CCollectUnavailable as e:
        raise SystemExit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
