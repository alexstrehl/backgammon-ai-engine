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

# Same 21 probabilities as a float32 vector, in _DICE_OUTCOMES order.
# float32 (not float64) so that `chosen * _ONEPLY_PROBS32[:, None]` is
# bit-identical to the scalar `prob * vec` of the per-dice reference
# path: numpy casts a Python float scalar down to the array's float32
# before multiplying, so the two agree exactly.
_ONEPLY_PROBS32 = np.array(
    [p for _, p in _DICE_OUTCOMES], dtype=np.float32,
)


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


def _flip_rows(v: np.ndarray) -> np.ndarray:
    """Vectorised `_flip_vec` over an (N, 5) float32 block.

    Bit-identical to stacking `_flip_vec` row by row: every element is
    produced by the same float32 op (`1.0 - x` or a plain copy), and
    numpy keeps a weak Python-float scalar at float32.
    """
    out = np.empty((v.shape[0], NUM_OUTPUTS), dtype=np.float32)
    out[:, 0] = 1.0 - v[:, 0]
    out[:, 1] = v[:, 3]
    out[:, 2] = v[:, 4]
    out[:, 3] = v[:, 1]
    out[:, 4] = v[:, 2]
    return out


def _prob5_equity_np(m: np.ndarray) -> np.ndarray:
    """Bit-exact numpy mirror of ``model.prob5_to_equity`` for an
    (N, 5) **float32** array.

    Why not just call ``prob5_to_equity(torch.from_numpy(m))``: on the
    1-ply target path that costs ~5 torch dispatches per dice outcome
    (21 per position) for what is a 5-element dot product.

    Why this exact expression: ``prob5_to_equity`` is
    ``(probs * w).sum(dim=-1) - 1.0`` with ``w = [2, 1, 1, -1, -1]``.
    The elementwise product is unambiguous, but the fp32 reduction over
    the length-5 last dim is NOT associative, and ATen's CPU sum kernel
    uses a 4-accumulator cascade: lane j seeds accumulator j, the tail
    element 4 folds back into accumulator 0, and the accumulators are
    then combined left to right. That is exactly

        (((p0 + p4) + p1) + p2) + p3        then  ``- 1.0``

    which is what the statements below evaluate (``x + (-y)`` is
    bit-identical to ``x - y`` in IEEE-754, including the signed-zero
    cases). Every other plausible order — plain left-to-right, numpy's
    ``sum``, ``@``, ``einsum``, pairwise trees — disagrees with torch on
    ~25-60% of random rows by 1 ulp, and a 1-ulp change is enough to
    flip an ``argmax`` and so change a whole target vector. The
    ``- 1.0`` is kept (rather than dropped as an argmax-invariant
    constant) because rounding can map two distinct pre-subtraction
    values onto one, which would move a first-occurrence tie.

    ``tests/test_prob5_oneply_fast.py`` asserts this equality directly,
    so a torch upgrade that changes the reduction order fails loudly
    instead of silently drifting.
    """
    e = m[:, 0] * np.float32(2.0)
    e -= m[:, 4]
    e += m[:, 1]
    e += m[:, 2]
    e -= m[:, 3]
    e -= np.float32(1.0)
    return e


# ── 1-ply exact Bellman backup ────────────────────────────────────────


def _oneply_target_vec(
    state: BoardState,
    network: ProbNetwork,
    encode_fn: Callable[[BoardState], np.ndarray],
    gpe_fn: Callable,
    device: str,
) -> np.ndarray:
    """Exact 1-ply Bellman target for `state`.

    Dispatches to the batched `expand_21` fast path when the C engine is
    in use (`gpe_fn.use_expand21`, set by `_resolve_engine_fns`), and
    falls back to the per-dice reference implementation otherwise — or
    when `expand_21` reports a buffer overflow. The two are bit-exact
    equivalents; see `_oneply_target_vec_expand21`.
    """
    if getattr(gpe_fn, "use_expand21", False):
        fast = _oneply_target_vec_expand21(state, network, encode_fn, device)
        if fast is not None:
            return fast
    return _oneply_target_vec_perdice(
        state, network, encode_fn, gpe_fn, device,
    )


def _oneply_target_vec_expand21(
    state: BoardState,
    network: ProbNetwork,
    encode_fn: Callable[[BoardState], np.ndarray],
    device: str,
) -> Optional[np.ndarray]:
    """`_oneply_target_vec_perdice` with the 21 move-gen calls and the
    per-successor Python loops replaced by one `bg_fast.expand_21` C
    call plus a numpy segment reduction.

    Returns None if `expand_21` is unavailable or overflows its buffers,
    in which case the caller must use the per-dice path.

    `expand_21` hands back, for the 21 dice in `_DICE_OUTCOMES` order
    (verified identical to the C loop `d1=1..6, d2=d1..6` in
    `c_engine/bg_engine.c:1074`):
      * `base_feats` (n_rows, 196) — the perspective196 encoding of every
        NON-terminal reply, packed in reply order; a forced-pass dice
        contributes exactly one row, the encoding of
        `switch_turn(state)`, which is precisely what the reference path
        puts in `pass_buf`.
      * `counts` (21,) — replies per dice, always >= 1.
      * `gr` (n_replies,) — 0 for a non-terminal reply (consumes the next
        `base_feats` row), else the winner's game_result 1/2/3, from
        which `_terminal_target`'s `[1, r>=2, r>=3, 0, 0]` is rebuilt.
        `game_result_c` mirrors `BoardState.game_result` exactly.
    All four are fresh copies (`bg_fast.expand_21`), so nothing here
    aliases a C buffer that a later call would invalidate — and
    `get_legal_plays_encoded_21` uses its own static play array, so it
    does not clobber `_play_buf` behind an outstanding `_LazyNextStates`.

    ROW ORDER IS LOAD-BEARING. The reference path feeds the network
    `pass_buf + play_buf` (all forced-pass rows first); `expand_21`
    returns replies interleaved in dice order. A CPU MLP forward is
    *not* bit-invariant under a row permutation — measured 26 differing
    outputs in 87k on a [80] net and 15 in 82k on [512,512,256], each
    1 ulp — and 1 ulp is enough to move an argmax. So when any forced
    pass is present its rows are gathered to the front before the
    forward and the outputs are scattered back afterwards. Positions
    with no forced pass (the overwhelming majority) skip the gather
    entirely, because the two orders already coincide.
    """
    exp = _bg_fast.expand_21(state)
    if exp is None:
        return None
    base_feats, counts, gr, n_rows = exp
    n_replies = gr.shape[0]

    nonterm = gr == 0
    # Reply-space segment starts (counts >= 1, so strictly increasing —
    # required for the reduceat calls below).
    seg = np.empty(len(_DICE_OUTCOMES), dtype=np.int64)
    seg[0] = 0
    np.cumsum(counts[:-1], out=seg[1:])

    # ── Forced-pass detection (see ROW ORDER above) ──────────────────
    # A forced pass is the only way a dice bucket holds exactly one
    # non-terminal reply whose encoding equals `switch_turn(state)`: any
    # legal play moves at least one checker, so a real reply's board —
    # and hence its 196-vector — always differs.
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
        x = torch.from_numpy(x_np).to(device)
        with torch.no_grad():
            out = network(x)
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
        # No terminal reply anywhere — the common case; skip the scatter.
        moves = _flip_rows(out_np)
    else:
        moves = np.empty((n_replies, NUM_OUTPUTS), dtype=np.float32)
        moves[nonterm] = _flip_rows(out_np)
        term_gr = gr[~nonterm]
        # `_terminal_target`, rebuilt from the C game_result: the mover
        # just bore off all 15, so it wins, and gr is 1/2/3.
        term = np.empty((term_gr.shape[0], NUM_OUTPUTS), dtype=np.float32)
        term[:, 0] = 1.0
        term[:, 1] = term_gr >= 2
        term[:, 2] = term_gr >= 3
        term[:, 3] = 0.0
        term[:, 4] = 0.0
        moves[~nonterm] = term

    # ── Per-dice argmax over derived equity, keeping the WHOLE vector ─
    # `np.argmax` returns the FIRST maximum; reproduce that exactly with
    # a max-then-lowest-hitting-index reduction. A plain
    # `argmax.reduceat` does not exist, and reducing on the values alone
    # would leave the tie-break to whatever order the reduction happened
    # to visit — which on transpositions (two play sequences reaching
    # the same board) is a real, frequent tie.
    eqs = _prob5_equity_np(moves)
    seg_max = np.maximum.reduceat(eqs, seg)
    hit = eqs == np.repeat(seg_max, counts)
    cand = np.where(hit, np.arange(n_replies), n_replies)
    best = np.minimum.reduceat(cand, seg)
    if best.max() >= n_replies:
        # Only reachable if some bucket's equities contain NaN: NaN
        # poisons `seg_max` and then compares unequal to everything, so
        # the bucket has no hit. The reference path would instead return
        # a NaN target here (np.argmax picks the first NaN), which BCE
        # turns into a NaN loss and a dead net several rounds later.
        # This is the one intentional behavioural difference: fail now,
        # loudly, instead of poisoning the run silently.
        raise ValueError(
            "prob5 1-ply target: network produced non-finite outputs "
            "(NaN equity in at least one dice bucket)"
        )

    # Accumulate exactly as the reference loop does: float32 throughout,
    # dice in _DICE_OUTCOMES order. `sum(axis=0)` over a (21, 5) float32
    # block visits rows in order and is bit-identical to the reference's
    # `target += prob * vec` loop (asserted in the parity tests).
    target = (moves[best] * _ONEPLY_PROBS32[:, None]).sum(axis=0)
    return np.clip(target, 0.0, 1.0)


def _oneply_target_vec_perdice(
    state: BoardState,
    network: ProbNetwork,
    encode_fn: Callable[[BoardState], np.ndarray],
    gpe_fn: Callable,
    device: str,
) -> np.ndarray:
    """Reference 1-ply target: 21 separate move-gen calls.

    Kept as the definition of correctness for
    `_oneply_target_vec_expand21` (and as its fallback). Do not "clean
    up" the arithmetic here without re-running
    `tests/test_prob5_oneply_fast.py`.
    """
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
            # Move selection (0-ply): pick the move with best derived equity
            # from the mover's view. This MUST run *before* `_oneply_target_vec`
            # below: on the per-dice fallback that call re-invokes `gpe_fn` 21
            # times, and the C fast path (`bg_fast.get_legal_plays_encoded`)
            # reuses an internal successor buffer -- so `features` and the
            # `next_states` objects are only valid until the next `gpe_fn`
            # call. (The batched `expand_21` path happens to use its own
            # buffers, but the ordering constraint stands: the fallback is
            # reachable and both must be safe.)
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
                chosen = next_states[int(np.argmin(opp_eq))]
                # Independent copy: `_oneply_target_vec` overwrites the pooled
                # successor objects on its next `gpe_fn` call.
                next_state = BoardState(
                    points=list(chosen.points), bar=list(chosen.bar),
                    off=list(chosen.off), turn=chosen.turn,
                )
            else:
                next_state = switch_turn(state)
            tgts.append(_oneply_target_vec(
                state, network, encode_fn, gpe_fn, device,
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

    ``gpe_fn.use_expand21`` tells ``_oneply_target_vec`` whether the
    batched all-21-dice C expansion is available for this configuration.
    It rides on the function rather than being recomputed there because
    the 1-ply target only ever sees ``gpe_fn``, not ``encoder_name``.
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


def _unique_ckpt_path(path: str) -> str:
    """Return `path`, or the first free `..._rN` variant if it already exists.

    Periodic checkpoints are named from an episode counter that is local to a
    single invocation. Resuming a partially-finished leg restarts that counter,
    so without this a resumed run silently overwrites the earlier run's
    checkpoints -- and the file that survives is labelled with FEWER episodes
    than it actually contains, which is a correctness trap as much as a data
    loss (it will be banked and compared under the wrong label).
    """
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{root}_r{n}{ext}"):
        n += 1
    return f"{root}_r{n}{ext}"


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
    pipeline_collect: bool = False,
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

    def _build_worker_args(size: int):
        """Worker args carrying the CURRENT weights. Called eagerly when
        pipelining, which is what makes those targets one round stale."""
        # .clone() is REQUIRED, not defensive — the same race dd3cdc9 fixed on
        # the trainer.py side. train_prob5 does not import trainer.py, so that
        # fix did not reach here. On a CPU net `.cpu()` is a no-op returning a
        # VIEW sharing storage with the live parameter, so the tensors handed
        # to map_async are pickled by the pool's task thread WHILE
        # optimizer.step() mutates them in place — workers can receive torn
        # weights. Reachable in exactly the prob5 1-ply recipe (--device cpu +
        # --pipeline-collect + map_async); NOT on --device cuda, where .cpu()
        # genuinely copies in the master thread. Harm is bounded (a worker sees
        # an element-wise blend of net(t) and net(t+1), inside the one-round
        # staleness pipelining already accepts) and no bad value reaches the
        # master — workers return only (encodings, targets). But it is a real
        # nondeterminism source: identical CPU runs diverge. Cost: ~2MB/round.
        sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        dev = device if oneply else "cpu"
        return [
            (
                sd, net.hidden_sizes, net.input_size, net.activation,
                net.encoder_name, net.raw_logits, n,
                rng.randint(0, 2**31 - 1), oneply, dev,
            )
            for n in _split_episodes(size, workers) if n > 0
        ]

    # ── Pipeline pre-roll ─────────────────────────────────────────────
    # Overlap round N+1's collection with round N's training step, so the
    # train step hides behind collection instead of serialising after it.
    use_pipeline = pipeline_collect and pool is not None
    if pipeline_collect and pool is None:
        print("  NOTE: --pipeline-collect ignored (needs --workers > 1)")
    elif use_pipeline:
        print("  pipelined collection: ON (round N+1 collects during "
              "round N's training step)")

    pending = None
    if use_pipeline:
        pending = pool.map_async(
            _collect_worker,
            _build_worker_args(min(episodes_per_round, num_episodes)),
        )

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
            elif use_pipeline:
                # This round's collect was submitted last iteration (or
                # pre-loop). Submit the NEXT round's collect BEFORE
                # training, so those workers run during the master's train
                # step. They see the CURRENT weights, which makes their
                # targets one round stale.
                next_pending = None
                if round_idx + 1 < num_rounds:
                    next_size = min(
                        episodes_per_round,
                        num_episodes - (episodes_played + round_size),
                    )
                    if next_size > 0:
                        next_pending = pool.map_async(
                            _collect_worker, _build_worker_args(next_size),
                        )
                results = pending.get()
                pending = next_pending
                enc_chunks = [r[0] for r in results if len(r[0]) > 0]
                tgt_chunks = [r[1] for r in results if len(r[1]) > 0]
            else:
                results = pool.map(_collect_worker,
                                   _build_worker_args(round_size))
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
                # NEVER overwrite. `episodes_played` counts within THIS run, so
                # a leg resumed from a partial restarts it at 0 and would
                # silently clobber the earlier run's checkpoints -- and the
                # surviving file then carries a name that understates how much
                # training it contains, which is worse than losing it.
                ck_path = _unique_ckpt_path(f"{save_path}_ep{episodes_played}.pt")
                net.save(ck_path)
                print(f"    -> checkpoint saved to {ck_path}")

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if save_path:
        # Same never-overwrite rule as the periodic checkpoints: a resumed
        # leg re-run with the same --save must not clobber the previous
        # leg's final model.
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

    The two architectures share **no** parameter keys: ``TDNetwork`` stores
    ``hidden_layers.{i}`` + ``fc_output``, while ``ProbNetwork`` uses an
    ``nn.Sequential`` trunk (``trunk.{2*i}``, with the activation at the odd
    indices) + ``head``. Hidden layer ``i`` therefore maps
    ``hidden_layers.{i}`` -> ``trunk.{2*i}``, copied for the LEADING
    PREFIX of layers whose shapes match exactly (copying stops at the
    first mismatch). ``head`` is never written: it is a different task
    head (5 probabilities vs 1 equity).

    Note ``TDNetwork.save`` does not record ``model_type``, so
    ``ProbNetwork.load``'s guard does not catch an equity checkpoint — which
    is why ``--resume`` on one dies with a raw torch key error instead of a
    useful message. Hence the explicit validation here.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

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
            # Copy only a LEADING prefix. A later layer that happens to
            # re-match after a mismatch would sit on top of random-init
            # earlier layers and receive garbage activations -- worse
            # than a fresh init, and invisible in the summary.
            break
        for sk, dk in pairs:
            new_sd[dk] = src_sd[sk].clone()
            n_copied += new_sd[dk].numel()
        copied.append(i)
    net.load_state_dict(new_sd)

    # Loud summary: this whole flag previously did nothing at all, silently,
    # so it must be impossible to not notice what it did.
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
    """Resolve --resume / --expand / --expand-depth / --warm-start-equity-trunk
    / fresh."""
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
        net = ProbNetwork.depth_expand(
            src, new_layer_size=args.expand_depth_size,
        )
        print(
            f"Depth-expanded from {args.expand_depth}: "
            f"{src.hidden_sizes} -> {net.hidden_sizes}"
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



def reject_cuda_oneply(args):
    """Refuse ``--oneply`` with a non-CPU device, loudly.

    prob5 1-ply GPU collection is UNSUPPORTED in this release, by
    measurement, not omission: a bit-exact vectorized GPU collector was
    built and A/B'd on this codebase and lost to the CPU pipeline
    (130.6 ep/s best GPU sweep vs 251 ep/s CPU at the production
    W=64 --pipeline-collect config; see
    experiments/benchmarks/prob5_collect_levers_ab.md). Worse, the previous
    behavior of this flag combination put the net on the GPU in EVERY
    worker (~64 CUDA contexts ~= 16 GB) -- an OOM bomb on consumer
    cards. Refusing is strictly better than either: nothing is lost,
    because under --pipeline-collect the master train step hides behind
    collection, so cuda buys ~0%% at 1-ply even where it fits.

    0-ply training (no --oneply) keeps full --device cuda support
    (+82%% on the train step); every published 1-ply recipe of record
    already passes --device cpu.
    """
    if args.oneply and args.device != "cpu":
        raise SystemExit(
            "ERROR: prob5 1-ply training does not support --device "
            f"{args.device!r} in this release. The GPU collector was "
            "built, validated bit-exact, and measured SLOWER than the "
            "CPU pipeline on this codebase "
            "(experiments/benchmarks/prob5_collect_levers_ab.md); the old behavior "
            "also opened one CUDA context per worker and OOM'd consumer "
            "GPUs. Use --device cpu for 1-ply legs (the recipes of "
            "record all do); --device cuda remains supported (and "
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
    g_train.add_argument("--pipeline-collect", action="store_true",
                         help="overlap round N+1's collection with round N's "
                              "training step (needs --workers>1). Targets "
                              "become one round stale.")

    args = p.parse_args()

    reject_cuda_oneply(args)

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
        pipeline_collect=args.pipeline_collect,
    )


if __name__ == "__main__":
    main()
