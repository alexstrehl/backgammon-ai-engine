"""Option 2b: generator-based vectorized self-play for cubeful-money
1-ply collection.

Each game is a Python generator that runs the normal episode logic but
`yield`s a feature matrix (N, n_feat) whenever it needs a network
forward, receiving the (N,) values back via `.send()`. A per-process
driver runs B games in lockstep, concatenates all B pending feature
requests into ONE local GPU forward (net on cuda, in-process, no IPC),
and scatters the results back. Refills finished slots until the
episode quota is met.

This removes the per-forward IPC / single-server serialization of the
2a inference server: forwards from many games (and from different
logical phases — cube decision, 1-ply target, 0-ply move selection)
all batch together since they go through the same network. Move-gen
(`expand_21`, C) parallelizes across the K worker processes; within a
process everything is single-threaded so the shared C buffers in
bg_fast are used serially (safe).

Scope: cubeful-money, oneply targets, 0-ply move selection
(oneply_acting=False). Mirrors trainer._collect_episode_cubeful_money
exactly; rare expand_21 overflow falls back to the agent's own method.

EPISODE-LENGTH GUARD
Every generator here takes `max_episode_plies` and enforces it with
exactly the semantics of the non-vectorized collectors in trainer.py
(see DEFAULT_MAX_EPISODE_PLIES there): count plies (turns) at the top
of the `while not mode.is_episode_over(state)` loop and DISCARD the
whole episode once the bound is exceeded. This is load-bearing, not
defensive: the pathology the guard exists for (mutual-hitting
recirculation in prob5-seeded cubeful nets) is *worse* in lockstep
than in the serial collector. A single runaway game here does not just
grow one worker's RSS -- it pins a driver slot forever, so the round
never finishes even though the other B-1 games completed. The bound
therefore has to be per-GAME, inside the generator, and it is.

Discards are reported explicitly rather than inferred from an empty
return: each generator returns `(enc, tgt, discarded)` and
`vectorized_collect_worker` returns `(enc, tgt, n_discarded)` so the
count flows into the same round-log line as the CPU paths.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from backgammon_engine import opening_roll, switch_turn
from encoding import (
    CUBE_FEATURES, CubePerspective, OPP_OFF_INDEX, TERMINAL_OFF_THRESHOLD,
)
from modes import CubeOwner, cube_perspective
from model import TDNetwork
import td_agent as _ta
from td_agent import TDAgent
from trainer import DEFAULT_MAX_EPISODE_PLIES

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
    path). Yields the (n_rows, n_feat) feature matrix; receives the
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

    # Rare: C path unavailable / expand_21 overflow → fall back to the
    # real method (does its own local forward). No yield.
    return agent.value_oneply_checker_cubeful(state, match_state)


def _choose_checker_action_cubeful_0ply_gen(agent, state, dice, match_state):
    """Generator mirror of TDAgent.choose_checker_action_cubeful 0-ply
    path. Yields the (n_plays, n_feat) feature matrix; receives the
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
    extra = np.zeros((len(next_states), CUBE_FEATURES), dtype=np.float32)
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
    oneply=True. Yields feature matrices for batched forwards; returns
    (enc_array, tgt_array, discarded) via StopIteration.

    Cube targets are 0-ply (one batched forward(batch=2) over the
    no-double and double-take encodings), exactly as
    trainer._collect_episode_cubeful_money does on this branch. The
    1-ply machinery applies to the CHECKER phase only.

    `max_episode_plies` matches trainer._collect_episode_cubeful_money
    exactly: the counter increments once per iteration of the
    `is_episode_over` loop and the episode is discarded (empty arrays,
    discarded=True) the moment the count reaches the bound -- checked
    BEFORE any work for that ply, so a capped episode yields nothing
    further and its slot is freed for a refill. 0 disables.
    """
    state, dice = opening_roll(rng)
    enc_list: List[np.ndarray] = []
    targets: List[float] = []
    is_opening = True
    plies = 0

    while not mode.is_episode_over(state):
        if max_episode_plies > 0 and plies >= max_episode_plies:
            # Runaway episode -- discard. In the lockstep driver this
            # also releases the slot, which a truncate-and-continue
            # policy would not do safely (see module docstring).
            return _discarded()
        plies += 1
        player = state.turn

        # ── Cube decision phase (if eligible) ───────────────────────
        if not is_opening and match_state.can_offer(player):
            pre_persp = cube_perspective(match_state.cube_owner, player)
            enc_list.append(agent._encode_cubeful(
                state, pre_persp, is_cube_action=True,
            ))

            # 0-ply cube targets, batched into one forward(batch=2).
            x_nd = agent._encode_cubeful(
                state, pre_persp, is_cube_action=False,
            )
            x_dt = agent._encode_cubeful(
                state, CubePerspective.THEIRS, is_cube_action=False,
            )
            values = (yield np.stack([x_nd, x_dt]))
            values = np.asarray(values, dtype=np.float32).ravel()
            v_no_double = float(values[0])
            v_double_take = 2.0 * float(values[1])
            do_double = agent._money_cube_decision(
                v_no_double, v_double_take, pre_persp, match_state.jacoby,
            )

            if do_double:
                takes = v_double_take <= 1.0
                if takes:
                    targets.append(v_double_take)
                    match_state = match_state.after_take(player)
                else:
                    targets.append(1.0)
                    break
            else:
                targets.append(v_no_double)

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
    """Generator mirror of TDAgent.value_oneply_checker — PER-DICE, one
    yield per dice outcome, exactly like the reference.

    WHY PER-DICE (2026-08-05). The reference runs a SEPARATE small
    forward for each of the 21 dice outcomes, with terminal successor
    rows INCLUDED in the batch and overridden afterwards. fp32 forwards
    are not batch-composition invariant (kernel blocking varies with M;
    measured 1-ulp row differences), so the previous single-batched-
    forward version of this mirror produced targets 1 ulp off the
    reference on essentially every episode — caught by
    tests/test_vectorized_cubeless.py, which requires bit-exact targets.
    Per-dice yields restore bitwise parity with the CPU path. The GPU
    win survives: the lockstep driver still batches each yield step
    ACROSS the B in-flight games.

    (Post-release option, gated on a strength twin: batch the CPU
    reference itself over all 21 dice — as the cubeful reference already
    does — then this mirror can go back to one yield per position.)

    Mirrored details, all load-bearing for bitwise equality:
      * forward input INCLUDES terminal successor rows (net output then
        overridden), matching the reference batch content;
      * terminal overrides are materialised from `next_states` BEFORE
        yielding — the C fast path's `next_states` is a lazy view over a
        shared buffer that other lockstep games invalidate while this
        generator is suspended (`features` is a copy and safe to yield);
      * accumulation is Python-float (f64) in `_DICE_OUTCOMES` order;
      * forced pass mirrors `evaluate(switch_turn(state))`: a (1, F)
        forward of the plain encoding. The value is cached — repeat
        forwards of an identical (1, F) input are deterministic, so the
        cached value equals the reference's per-dice recomputation.
    """
    oneply_sum = 0.0
    pass_v = None            # cached mirror of evaluate(switch_turn(state))
    for idx, (d1, d2) in enumerate(_DICE_OUTCOMES):
        prob = float(_ONEPLY_PROBS[idx])
        features, next_states = agent._get_legal_plays_encoded(
            state, (d1, d2))

        if len(next_states) == 0:
            # Forced pass: reference does evaluate(switch_turn(state))
            # = net(encode(state')[None]).item().
            if pass_v is None:
                row = agent._encode(switch_turn(state))
                V = (yield np.ascontiguousarray(row[None, :]))
                pass_v = float(np.asarray(V, dtype=np.float32).ravel()[0])
            oneply_sum += prob * agent._flip_value(pass_v)
            continue

        # Terminal overrides BEFORE yielding (lazy-buffer hazard above).
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


def _choose_checker_action_cubeless_0ply_gen(agent, state, dice):
    """Generator mirror of TDAgent.choose_checker_action 0-ply cubeless
    path. Yields the (n_plays, n_feat) feature matrix; receives the
    (n_plays,) fp32 values. Returns (chosen_state, bootstrap) or None.

    No cube columns; terminal masking and Boltzmann selection match
    choose_checker_action / _apply_terminal_mask exactly.
    """
    features, lazy_next = agent._get_legal_plays_encoded(state, dice)
    n_plays = len(lazy_next)
    if n_plays == 0:
        return None
    # Materialize successors BEFORE yielding: the C fast path backs
    # next_states with a shared buffer that other games' move-gen calls
    # clobber while this generator is suspended at the yield. `features`
    # is already a private copy (bg_fast copies it), so it's safe.
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
    oneply=True. Yields feature matrices for batched forwards; returns
    (enc_array, tgt_array, discarded) via StopIteration.

    Mirrors the non-vectorized cubeless path exactly:
      - encode current state (no cube features),
      - 1-ply checker target via value_oneply_checker,
      - 0-ply checker move selection.
    No cube-decision phase.

    `max_episode_plies`: as trainer.collect_episode -- count plies at
    the top of the loop, discard the episode on exceeding the bound.
    0 disables.
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
        target = yield from _value_oneply_checker_cubeless_gen(agent, state)
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


# ── per-process driver ──────────────────────────────────────────────
def vectorized_collect_worker(args):
    """Worker subprocess: run B games in lockstep, batching all pending
    forwards into one local GPU call per step. Returns
    (encodings, targets, n_discarded) for `num_episodes` games --
    the same 3-tuple shape as trainer._collect_worker, so the trainer's
    round accounting and log line are identical on both paths.

    `n_discarded` counts episodes killed by `max_episode_plies`. It is
    taken from each generator's explicit `discarded` flag, NOT inferred
    from an empty result, so a legitimately-empty episode is never
    miscounted as a runaway.

    NOTE: the args tuple arity must stay in sync with `_build_vec_args`
    in Trainer.train.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    (state_dict, hidden_sizes, input_size, output_mode, encoder_name,
     activation, mode, num_episodes, seed, oneply, oneply_acting,
     boltzmann_temp, bf16_collect, B, device,
     max_episode_plies) = args
    assert oneply, "vectorized_collect is oneply-only"
    # The generators implement 0-ply move SELECTION only (1-ply is used
    # for TARGETS). Accepting oneply_acting here would silently play a
    # different policy than requested, changing the data distribution
    # with no error -- refuse instead.
    assert not oneply_acting, (
        "vectorized_collect does not implement 1-ply move selection "
        "(--oneply-acting); it does 1-ply targets with 0-ply acting"
    )

    # Driver net lives on `device` (cuda): it runs the single batched
    # forward per step over all B games' pending feature requests.
    net = TDNetwork(
        hidden_sizes=hidden_sizes, input_size=input_size,
        output_mode=output_mode, encoder_name=encoder_name,
        # activation MUST be applied: it is not a parameter tensor, so
        # load_state_dict cannot catch a mismatch — see _collect_worker.
        activation=activation,
    )
    net.load_state_dict(state_dict)
    net.to(device).eval()

    # The AGENT is built on CPU so its move-gen + encode use the fast C
    # `bg_fast` path (which requires a CPU agent) and the 1-ply target
    # uses the batched `expand_21` fast path — all of which only build
    # feature matrices that the generators YIELD; the actual forwards
    # are done by the cuda driver net above. The agent wraps its own CPU
    # net copy purely so any rare in-agent fallback forward (e.g.
    # expand_21 overflow in the cubeful gen) runs on a matching device.
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

    # Cubeless modes (cubeless-money, dmp) have no MatchState/cube; route
    # them to the cubeless generator. Cubeful modes return a MatchState.
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

    # Initial fill. Keep starting games until either B slots are live or
    # the episode quota is exhausted -- NOT just B attempts. A game that
    # finishes without ever yielding (e.g. every episode discarded by a
    # very small max_episode_plies) returns None and fills no slot; if
    # we only attempted B times, `slots` could end up empty, the driver
    # loop below would never run, and the worker would silently return
    # having played only B of num_episodes episodes.
    slots = []
    while len(slots) < B and episodes_started < num_episodes:
        s = _start_game()
        if s is not None:
            slots.append(s)

    with torch.no_grad():
        while slots:
            sizes = [s[1].shape[0] for s in slots]
            if len(slots) == 1:
                big = slots[0][1]
            else:
                big = np.concatenate([s[1] for s in slots], axis=0)
            t = torch.from_numpy(big).to(device)
            out = net(t).reshape(-1).to("cpu", torch.float32).numpy()

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
                        # else game ended with no forward; keep refilling
            slots = new_slots

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
