"""Vectorized (lockstep) self-play for 1-ply collection.

Each game is a Python generator that `yield`s a feature matrix
(N, n_feat) whenever it needs a network forward and receives the (N,)
values back via `.send()`. A per-process driver runs B games in
lockstep, concatenates the B pending requests into one forward, and
refills finished slots until the episode quota is met. Move-gen runs
single-threaded per process, so bg_fast's shared C buffers are used
serially.

Where the one forward per step runs (`device` in the worker args):
  * "cuda"   -- per-worker context: the net lives in this process.
  * "remote" -- shared context (default): the fused batch goes through
                a shared-memory slot of an EvalServer process
                (`remote_worker_init` claims the slot), which runs it
                as its own forward: bitwise the same values as "cuda".

Scope: cubeful-money and cubeless-money, 1-ply targets, 0-ply move
selection (oneply_acting=False). Mirrors the trainer.py collectors.

Episode-length guard: every generator takes `max_episode_plies` with
the same semantics as trainer.py (DISCARD the whole episode past the
bound). The bound must be per-game, inside the generator: a runaway
game would otherwise pin a driver slot forever. Generators return
`(enc, tgt, discarded)` and `vectorized_collect_worker` returns
`(enc, tgt, n_discarded)`.
"""

from __future__ import annotations

import random
from typing import List

import os
import sys
import time

import numpy as np
import torch

from backgammon_engine import opening_roll, switch_turn
from encoding import OPP_OFF_INDEX, TERMINAL_OFF_THRESHOLD
from modes import CubeOwner, cube_perspective
from model import TDNetwork
import td_agent as _ta
from td_agent import TDAgent
from trainer import DEFAULT_MAX_EPISODE_PLIES, _cube_phase_oneply_gen

_bg_fast = _ta._bg_fast
_ONEPLY_PROBS = _ta._ONEPLY_PROBS


_EMPTY_RESULT = (
    np.empty((0, 0), dtype=np.float32),
    np.empty((0,), dtype=np.float32),
)


def _discarded():
    """Return value for an episode killed by the ply bound."""
    return _EMPTY_RESULT[0], _EMPTY_RESULT[1], True


# ── generator mirrors of the forward-using methods ──────────────────
def _value_oneply_checker_cubeful_gen(agent, state, match_state):
    """Generator mirror of TDAgent.value_oneply_checker_cubeful (fast
    path). Yields the (n_rows, n_feat) feature matrix. Receives the
    (n_rows,) fp32 values. Returns the 1-ply per-unit equity (float)."""
    player = state.turn
    opponent = 1 - player
    opp_persp = cube_perspective(match_state.cube_owner, opponent)
    opp_cube_action = match_state.can_offer(opponent)
    base = agent.encoder._base
    n_feat = agent.encoder.num_features
    n_base = base.num_features
    gammons_count = (
        not match_state.jacoby
        or match_state.cube_owner != CubeOwner.CENTERED
    )

    if agent._c_base_available:
        exp = _bg_fast.expand_21(state)
        if exp is not None:
            base_feats, counts, gr, n_rows = exp
            opp_col = int(opp_persp)
            if n_rows:
                feat = np.zeros((n_rows, n_feat), dtype=np.float32)
                feat[:, :n_base] = base_feats
                feat[:, n_base + opp_col] = 1.0
                if opp_cube_action:
                    feat[:, n_base + 3] = 1.0
                V = (yield feat)
                V = np.asarray(V, dtype=np.float32).ravel()
            else:
                V = np.empty(0, dtype=np.float32)
            nonterm = gr == 0
            vmover = gr.astype(np.float64)
            if not gammons_count:
                vmover[~nonterm] = 1.0
            vmover[nonterm] = -V
            seg = np.empty(counts.shape[0], dtype=np.int64)
            seg[0] = 0
            np.cumsum(counts[:-1], out=seg[1:])
            dice_max = np.maximum.reduceat(vmover, seg)
            return float(_ONEPLY_PROBS @ dice_max)

    # C path unavailable or expand_21 overflow: the real method does its
    # own local forward. No yield.
    return agent.value_oneply_checker_cubeful(state, match_state)


def _choose_checker_action_cubeful_0ply_gen(agent, state, dice, match_state):
    """Generator mirror of TDAgent.choose_checker_action_cubeful 0-ply
    path. Yields the (n_plays, n_feat) feature matrix. Receives the
    (n_plays,) fp32 values. Returns (chosen_state, bootstrap) or None."""
    if agent._c_base_available:
        base_feats, _lazy = _bg_fast.get_legal_plays_encoded(state, dice)
        n_plays = len(_lazy)
        if n_plays == 0:
            return None
        next_states = [_lazy[i] for i in range(n_plays)]
    else:
        from backgammon_engine import get_legal_plays
        plays = get_legal_plays(state, dice)
        if not plays:
            return None
        next_states = [s for _, s in plays]
        base = agent.encoder._base
        base_feats = np.stack([base.encode(s) for s in next_states])

    opponent = 1 - state.turn
    opp_persp = cube_perspective(match_state.cube_owner, opponent)
    opp_cube_action = match_state.can_offer(opponent)
    extra = np.zeros(
        (len(next_states), agent.encoder.cube_features), dtype=np.float32,
    )
    extra[:, int(opp_persp)] = 1.0
    if opp_cube_action:
        extra[:, 3] = 1.0
    features = np.concatenate([base_feats, extra], axis=1)

    values_np = (yield features)
    values = torch.from_numpy(np.asarray(values_np, dtype=np.float32).ravel())

    terminal_mask = features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
    if terminal_mask.any():
        gammons_count = (
            not match_state.jacoby
            or match_state.cube_owner != CubeOwner.CENTERED
        )
        values = values.clone()
        for i in np.nonzero(terminal_mask)[0].tolist():
            gr = float(next_states[i].game_result())
            values[i] = -gr if gammons_count else -1.0
    idx = agent._boltzmann_select(values)
    chosen = next_states[idx]
    bootstrap = -float(values[idx].item())
    return chosen, bootstrap


def _collect_episode_cubeful_money_gen(
    agent, mode, rng, match_state,
    max_episode_plies=DEFAULT_MAX_EPISODE_PLIES,
):
    """Generator mirror of trainer._collect_episode_cubeful_money for
    oneply=True. Yields feature matrices. Returns
    (enc_array, tgt_array, discarded) via StopIteration.

    Cube targets are 0-ply. Only the checker phase is 1-ply.
    `max_episode_plies`: checked at the top of each ply; the episode is
    discarded on reaching the bound. 0 disables.
    """
    state, dice = opening_roll(rng)
    enc_list: List[np.ndarray] = []
    targets: List[float] = []
    is_opening = True
    plies = 0

    while not mode.is_episode_over(state):
        if max_episode_plies > 0 and plies >= max_episode_plies:
            # Runaway episode: discard, which also frees the driver slot.
            return _discarded()
        plies += 1
        player = state.turn

        # ── Cube decision phase (if eligible) ───────────────────────
        if not is_opening and match_state.can_offer(player):
            # Shared with the CPU collector (trainer.py); the lockstep
            # driver serves the yield.
            dec = yield from _cube_phase_oneply_gen(agent, state, match_state)
            enc_list.append(dec.enc)
            targets.append(dec.target)
            match_state = dec.match_state
            if dec.dropped:
                break

        # ── Checker decision phase ──────────────────────────────────
        post_persp = cube_perspective(match_state.cube_owner, player)
        enc_list.append(agent._encode_cubeful(
            state, post_persp, is_cube_action=False,
        ))

        checker_target = yield from _value_oneply_checker_cubeful_gen(
            agent, state, match_state,
        )
        targets.append(checker_target)

        result = yield from _choose_checker_action_cubeful_0ply_gen(
            agent, state, dice, match_state,
        )
        if result is not None:
            next_state, _bootstrap = result
            if next_state.is_game_over():
                break
            state = next_state
        else:
            state = switch_turn(state)

        dice = (rng.randint(1, 6), rng.randint(1, 6))
        is_opening = False

    if not enc_list:
        return _EMPTY_RESULT[0], _EMPTY_RESULT[1], False
    return (
        np.stack(enc_list),
        np.asarray(targets, dtype=np.float32),
        False,
    )


# ── cubeless-money generator mirrors ────────────────────────────────
_DICE_OUTCOMES = _ta._DICE_OUTCOMES


def _value_oneply_checker_cubeless_gen(agent, state):
    """Generator mirror of TDAgent.value_oneply_checker: one yield per
    dice outcome, like the reference.

    Per-dice because fp32 forwards are not batch-composition invariant
    and tests/test_vectorized_cubeless.py requires bit-exact targets.

    Details that bitwise equality depends on:
      * the forward input INCLUDES terminal successor rows (overridden
        afterwards), as in the reference;
      * terminal overrides are read from `next_states` BEFORE yielding:
        the C fast path's `next_states` is a lazy view over a shared
        buffer that other games clobber while this generator is
        suspended (`features` is a copy);
      * accumulation is Python float in `_DICE_OUTCOMES` order;
      * forced pass = a (1, F) forward of `switch_turn(state)`, cached.
    """
    oneply_sum = 0.0
    pass_v = None            # cached evaluate(switch_turn(state))
    for idx, (d1, d2) in enumerate(_DICE_OUTCOMES):
        prob = float(_ONEPLY_PROBS[idx])
        features, next_states = agent._get_legal_plays_encoded(
            state, (d1, d2))

        if len(next_states) == 0:
            # Forced pass: evaluate(switch_turn(state)).
            if pass_v is None:
                row = agent._encode(switch_turn(state))
                V = (yield np.ascontiguousarray(row[None, :]))
                pass_v = float(np.asarray(V, dtype=np.float32).ravel()[0])
            oneply_sum += prob * agent._flip_value(pass_v)
            continue

        # Terminal overrides BEFORE yielding (lazy-buffer hazard).
        term = {
            int(i): agent._terminal_opp_value(next_states[int(i)])
            for i in np.flatnonzero(
                features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
            )
        }
        V = (yield features)
        values = np.asarray(V, dtype=np.float32).ravel().copy()
        for i, tv in term.items():
            values[i] = tv
        best_for_next = float(values.min())
        oneply_sum += prob * agent._flip_value(best_for_next)
    return oneply_sum


def _value_oneply_checker_cubeless_gen_fast(agent, state):
    """Fast mirror of TDAgent.value_oneply_checker: one batched
    `expand_21` forward per position (one yield) instead of 21.

    NOT bit-exact with the CPU reference (the reduction differs in the
    last ulp), so the per-dice generator is the default. Select this
    one with BG_VEC_CUBELESS_FAST=1.

    Yields one (n_rows, n_feat) matrix of all non-terminal 21-dice
    replies. Receives (n_rows,) fp32 values. Returns the 1-ply per-unit
    value for the on-roll player. Mover-view flip and terminal handling
    follow agent.output_mode as value_oneply_checker does:
        equity:      non-term -> -V,  term -> +game_result
        probability: non-term -> 1-V, term -> +1
    Forced-pass dice are folded into expand_21's buckets.
    """
    is_equity = agent.output_mode == "equity"

    # expand_21 returns replies in _DICE_OUTCOMES order with a per-dice
    # count. reduceat over the segment offsets gives the per-dice max.
    if agent._c_base_available:
        exp = _bg_fast.expand_21(state)
        if exp is not None:
            base_feats, counts, gr, n_rows = exp
            if n_rows:
                # base_feats is the full cubeless encoding.
                V = (yield np.ascontiguousarray(base_feats))
                V = np.asarray(V, dtype=np.float32).ravel()
            else:
                V = np.empty(0, dtype=np.float32)
            nonterm = gr == 0
            # Per-unit mover value per reply.
            vmover = np.empty(gr.shape[0], dtype=np.float64)
            if is_equity:
                vmover[~nonterm] = gr[~nonterm]
                vmover[nonterm] = -V
            else:
                vmover[~nonterm] = 1.0
                vmover[nonterm] = 1.0 - V
            seg = np.empty(counts.shape[0], dtype=np.int64)
            seg[0] = 0
            np.cumsum(counts[:-1], out=seg[1:])
            dice_max = np.maximum.reduceat(vmover, seg)
            return float(_ONEPLY_PROBS @ dice_max)

    # Python path: per-dice expansion accumulated into ONE yielded batch.
    # Forced-pass dice evaluate switch_turn(state) once.
    feat_chunks = []
    chunk_meta = []          # (prob, start, end, term_overrides)
    pass_prob = 0.0
    pass_feat = None
    offset = 0
    for di, (d1, d2) in enumerate(_DICE_OUTCOMES):
        prob = float(_ONEPLY_PROBS[di])
        features, next_states = agent._get_legal_plays_encoded(state, (d1, d2))
        if len(next_states) == 0:
            pass_prob += prob
            if pass_feat is None:
                pass_feat = agent.encode_state(switch_turn(state))
            continue
        n = len(next_states)
        feat_chunks.append(features)
        term = {
            int(i): agent._terminal_opp_value(next_states[int(i)])
            for i in np.flatnonzero(
                features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
            )
        }
        chunk_meta.append((prob, offset, offset + n, term))
        offset += n

    # One batch: legal-reply chunks plus (optionally) one pass row.
    parts = list(feat_chunks)
    pass_off = None
    if pass_prob > 0.0:
        pass_off = offset
        parts.append(pass_feat[None, :])
        offset += 1
    if not parts:
        return 0.0
    big = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    allv = (yield big)
    allv = np.asarray(allv, dtype=np.float32).ravel()

    oneply_sum = 0.0
    for prob, s, e, term in chunk_meta:
        v = allv[s:e].copy()
        for i, tv in term.items():
            v[i] = tv
        oneply_sum += prob * agent._flip_value(float(v.min()))
    if pass_off is not None:
        oneply_sum += pass_prob * agent._flip_value(float(allv[pass_off]))
    return oneply_sum


# Cubeless 1-ply generator: the per-dice mirror (bit-exact with the CPU
# reference) unless BG_VEC_CUBELESS_FAST=1 selects the batched expand_21
# path (1 ulp off). See experiments/retraining/README.md.
_CUBELESS_1PLY_GEN = (
    _value_oneply_checker_cubeless_gen_fast
    if os.environ.get("BG_VEC_CUBELESS_FAST") == "1"
    else _value_oneply_checker_cubeless_gen
)

def _choose_checker_action_cubeless_0ply_gen(agent, state, dice):
    """Generator mirror of TDAgent.choose_checker_action, 0-ply cubeless
    path. Yields the (n_plays, n_feat) feature matrix. Receives the
    (n_plays,) fp32 values. Returns (chosen_state, bootstrap) or None.
    Terminal masking and Boltzmann selection match the reference.
    """
    features, lazy_next = agent._get_legal_plays_encoded(state, dice)
    n_plays = len(lazy_next)
    if n_plays == 0:
        return None
    # Materialize successors BEFORE yielding: next_states is a view over
    # a shared C buffer that other games clobber. `features` is a copy.
    next_states = [lazy_next[i] for i in range(n_plays)]

    values_np = (yield features)
    values = torch.from_numpy(np.asarray(values_np, dtype=np.float32).ravel())

    terminal_mask = features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
    if terminal_mask.any():
        values = values.clone()
        for i in np.nonzero(terminal_mask)[0].tolist():
            values[i] = agent._terminal_opp_value(next_states[i])
    idx = agent._boltzmann_select(values)
    chosen = next_states[idx]
    bootstrap = agent._flip_value(float(values[idx].item()))
    return chosen, bootstrap


def _collect_episode_cubeless_money_gen(
    agent, mode, rng, max_episode_plies=DEFAULT_MAX_EPISODE_PLIES,
):
    """Generator mirror of trainer.collect_episode (cubeless branch) for
    oneply=True. Yields feature matrices. Returns
    (enc_array, tgt_array, discarded) via StopIteration.

    Per ply: encode the state (no cube features), 1-ply checker target,
    0-ply move selection. No cube phase. `max_episode_plies` as in
    trainer.collect_episode; 0 disables.
    """
    state, dice = opening_roll(rng)
    enc_list: List[np.ndarray] = []
    targets: List[float] = []
    plies = 0

    while not mode.is_episode_over(state):
        if max_episode_plies > 0 and plies >= max_episode_plies:
            return _discarded()
        plies += 1
        enc_list.append(agent.encode_state(state))

        # 1-ply target for the current (checker) state.
        target = yield from _CUBELESS_1PLY_GEN(agent, state)
        targets.append(target)

        # 0-ply move selection.
        result = yield from _choose_checker_action_cubeless_0ply_gen(
            agent, state, dice,
        )
        if result is not None:
            next_state, _bootstrap = result
            state = next_state
        else:
            state = switch_turn(state)

        dice = (rng.randint(1, 6), rng.randint(1, 6))

    if not enc_list:
        return _EMPTY_RESULT[0], _EMPTY_RESULT[1], False
    return (
        np.stack(enc_list),
        np.asarray(targets, dtype=np.float32),
        False,
    )


# ── shared-context transport (one CUDA context total) ───────────────
# The pool initializer claims one EvalServer slot per worker process. It
# is never returned, so the pool must not recycle workers.
_WORKER_REMOTE = None

# Shared-memory slot rows per in-flight game. A step larger than the
# slot is split into chunked forwards (`_remote_forward`). Defined in
# gpu_context so the VRAM preflight sizes the server from the same number.
from gpu_context import SHARED_ROWS_PER_GAME as REMOTE_ROWS_PER_GAME  # noqa: E402


def remote_max_rows(batch_games: int) -> int:
    return int(max(8192, batch_games * REMOTE_ROWS_PER_GAME))


def remote_worker_init(handles):
    """Pool initializer for the shared-context collector. `handles` is one
    server's ClientHandles or an EvalServerGroup's GroupHandles."""
    global _WORKER_REMOTE
    from gpu_eval_server import claim_slot
    _WORKER_REMOTE = claim_slot(handles)


def _remote_forward(remote, parts, n: int) -> np.ndarray:
    """One fused lockstep batch through the server. `parts` are the B
    pending feature matrices (n rows total), concatenated straight into
    the shared-memory slot (one copy). Bitwise identical to the
    per-worker-context driver, except that a step larger than the slot
    is split, which can move that step's last ulp."""
    if n <= remote.max_rows:
        dst = remote.in_arr[:n]
        if len(parts) == 1:
            dst[...] = parts[0]
        else:
            np.concatenate(parts, axis=0, out=dst)
        out = remote.forward_from_slot(n)
        return out.reshape(-1) if out.ndim > 1 else out
    big = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    outs = []
    for s in range(0, n, remote.max_rows):
        out = remote.forward(big[s:s + remote.max_rows])
        outs.append(out.reshape(-1) if out.ndim > 1 else out)
    return np.concatenate(outs)


# ── per-process driver ──────────────────────────────────────────────
def vectorized_collect_worker(args):
    """Worker subprocess: run B games in lockstep, one forward per step.
    Returns (encodings, targets, n_discarded) for `num_episodes` games,
    the same shape as trainer._collect_worker. `n_discarded` counts
    episodes killed by `max_episode_plies` (an empty result is not a
    discard).

    The args tuple arity must stay in sync with `_build_vec_args` in
    Trainer.train.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    (state_dict, hidden_sizes, input_size, output_mode, encoder_name,
     activation, mode, num_episodes, seed, oneply, oneply_acting,
     boltzmann_temp, bf16_collect, B, device, max_episode_plies) = args
    assert oneply, "vectorized_collect is oneply-only"
    # The generators do 0-ply move SELECTION only (1-ply is for targets).
    # Accepting oneply_acting would silently play a different policy.
    assert not oneply_acting, (
        "vectorized_collect does not implement 1-ply move selection "
        "(--oneply-acting); it does 1-ply targets with 0-ply acting"
    )

    # Driver net on `device` runs the one forward per step. With
    # device == "remote" the net lives in the EvalServer process.
    remote = None
    net = None
    if device == "remote":
        remote = _WORKER_REMOTE
        if remote is None:
            raise RuntimeError(
                "vectorized_collect_worker got device='remote' but this "
                "process has no EvalServer slot: the pool must be created "
                "with initializer=remote_worker_init")
    else:
        net = TDNetwork(
            hidden_sizes=hidden_sizes, input_size=input_size,
            output_mode=output_mode, encoder_name=encoder_name,
            # activation is not a parameter tensor, so load_state_dict
            # cannot catch a mismatch. It must be passed.
            activation=activation,
        )
        net.load_state_dict(state_dict)
        net.to(device).eval()

    # The agent is built on CPU so move-gen + encode use the C `bg_fast`
    # path (CPU-only). Its own CPU net copy serves only the rare
    # in-agent fallback forward (e.g. expand_21 overflow).
    agent_net = TDNetwork(
        hidden_sizes=hidden_sizes, input_size=input_size,
        output_mode=output_mode, encoder_name=encoder_name,
        activation=activation,
    )
    agent_net.load_state_dict(state_dict)
    agent_net.eval()
    agent = TDAgent(
        agent_net, oneply=oneply_acting, boltzmann_temp=boltzmann_temp,
        bf16_inference=False, device="cpu",
    )

    base_rng = random.Random(seed)
    enc_chunks: List[np.ndarray] = []
    tgt_chunks: List[np.ndarray] = []
    episodes_started = 0
    n_discarded = 0

    # Cubeless modes (cubeless-money, dmp) have no MatchState.
    initial_ms = mode.initial_match_state()
    is_cubeless = initial_ms is None

    def _store(res):
        nonlocal n_discarded
        enc, tgt, discarded = res
        if discarded:
            n_discarded += 1
            return
        if len(enc) > 0:
            enc_chunks.append(enc)
            tgt_chunks.append(tgt)

    def _start_game():
        nonlocal episodes_started
        episodes_started += 1
        r = random.Random(base_rng.randint(0, 2**31 - 1))
        if is_cubeless:
            g = _collect_episode_cubeless_money_gen(
                agent, mode, r, max_episode_plies=max_episode_plies,
            )
        else:
            g = _collect_episode_cubeful_money_gen(
                agent, mode, r, mode.initial_match_state(),
                max_episode_plies=max_episode_plies,
            )
        try:
            feat = next(g)
        except StopIteration as e:
            _store(e.value)
            return None
        return [g, feat]

    # Initial fill: keep starting games until B slots are live or the
    # quota is exhausted, NOT just B attempts: a game that finishes
    # without yielding fills no slot.
    slots = []
    while len(slots) < B and episodes_started < num_episodes:
        s = _start_game()
        if s is not None:
            slots.append(s)

    # BG_VEC_PROFILE=1: per-task step timing on stderr. Diagnostic only.
    _prof = bool(os.environ.get("BG_VEC_PROFILE"))
    _t_task0 = time.perf_counter()
    _t_fwd = 0.0
    _n_steps = 0
    _n_rows = 0

    with torch.no_grad():
        while slots:
            sizes = [s[1].shape[0] for s in slots]
            n_total = sum(sizes)
            if _prof:
                _t0 = time.perf_counter()
            if remote is not None:
                out = _remote_forward(remote, [s[1] for s in slots], n_total)
            else:
                if len(slots) == 1:
                    big = slots[0][1]
                else:
                    big = np.concatenate([s[1] for s in slots], axis=0)
                t = torch.from_numpy(big).to(device)
                out = net(t).reshape(-1).to("cpu", torch.float32).numpy()
            if _prof:
                _t_fwd += time.perf_counter() - _t0
                _n_steps += 1
                _n_rows += n_total

            off = 0
            new_slots = []
            for s, n in zip(slots, sizes):
                vals = out[off:off + n]
                off += n
                try:
                    feat = s[0].send(vals)
                    s[1] = feat
                    new_slots.append(s)
                except StopIteration as e:
                    _store(e.value)
                    while episodes_started < num_episodes:
                        ns = _start_game()
                        if ns is not None:
                            new_slots.append(ns)
                            break
                        # else: game ended without a forward; keep refilling
            slots = new_slots

    if _prof and _n_steps:
        _tot = time.perf_counter() - _t_task0
        print(
            f"[vec-profile pid={os.getpid()} dev={device}] steps={_n_steps} "
            f"rows/step={_n_rows / _n_steps:.0f} step={_tot / _n_steps * 1e3:.2f}ms "
            f"fwd={_t_fwd / _n_steps * 1e3:.2f}ms "
            f"({_t_fwd / _tot * 100:.0f}% of task)",
            file=sys.stderr, flush=True,
        )

    if not enc_chunks:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            n_discarded,
        )
    return (
        np.concatenate(enc_chunks, axis=0),
        np.concatenate(tgt_chunks, axis=0),
        n_discarded,
    )
