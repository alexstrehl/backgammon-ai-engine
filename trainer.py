"""
trainer.py -- Trainer class for backgammon self-play TD learning.

Trainer pairs an Agent with a GameMode and owns the optimizer and
training loop. The Agent owns its representation (forward pass,
target shape, loss).

Two methods:
  - `train`: round-based batch TD(0). Adam by default. Supports
    parallel collection via `workers=N`.
  - `train_online`: classical online TD(0), one update per
    transition. Pair with SGD via `optimizer_cls=torch.optim.SGD`.

`collect_episode` is also exposed as a free function for tests and
external callers.
"""

import math
import multiprocessing as mp
import os
import random
import time
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from agents import Agent
from backgammon_engine import BoardState, opening_roll, switch_turn
from encoding import CubePerspective
from modes import GameMode, cube_perspective


# Safety bound on self-play episode length, in PLIES (turns), not
# samples. Neither episode loop below has any natural termination
# guarantee: they run `while not mode.is_episode_over(state)`, and a
# degenerate policy can produce games that never reach a terminal
# state (mutual hitting / recirculation, or a bear-in the net refuses
# to complete). Such an episode spins at full turn throughput forever
# while its sample lists grow without bound -- the master blocks on the
# round barrier (pool.map get()) and worker RSS climbs until the OOM
# killer fires.
#
# Healthy play is ~65-340 plies, so 1500 never fires on a healthy run
# while still bounding a runaway episode to a few thousand samples.
DEFAULT_MAX_EPISODE_PLIES = 1500

# VRAM preflight for the vectorized GPU collector. Each vectorized
# worker process owns a full CUDA context (~420 MB); the estimate is
# deliberately conservative -- a capped-but-running collector beats an
# OOM forty minutes into a leg.
VEC_PER_WORKER_BYTES = 420 * 2**20
VEC_RESERVE_BYTES = 512 * 2**20   # master context / display / allocator slack


def vectorized_vram_cap(requested_workers: int, free_bytes: int) -> int:
    """How many vectorized GPU worker processes fit in `free_bytes`.

    Pure function so the arithmetic is unit-testable without a GPU.
    Returns min(requested, floor((free - reserve) / per_worker)),
    clamped at >= 0. The CALLER decides what to do with 0 or 1 (fall
    back to the CPU collector -- one worker forfeits lockstep batching).
    """
    if requested_workers <= 0:
        return 0
    budget = free_bytes - VEC_RESERVE_BYTES
    if budget <= 0:
        return 0
    return min(requested_workers, int(budget // VEC_PER_WORKER_BYTES))


# ── Episode collection (mode-agnostic, agent-driven) ──────────────────


def collect_episode(
    agent: Agent,
    mode: GameMode,
    rng: random.Random,
    oneply: bool = False,
    cube_targets_1ply: bool = False,
    max_episode_plies: int = DEFAULT_MAX_EPISODE_PLIES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Play one self-play episode and return (encodings, targets).

    Encodings are computed once per turn via `agent.encode_state` so
    the training pass can skip re-encoding. Cubeless move selection
    is always 0-ply; cubeful respects agent.oneply.

    `max_episode_plies`: safety bound (see DEFAULT_MAX_EPISODE_PLIES).
    An episode that exceeds it is DISCARDED -- the function returns
    empty arrays. Discarding rather than truncating is deliberate: a
    runaway episode's samples come from a degenerate region of state
    space and would poison the pool, and truncate-and-bootstrap would
    change TD semantics. Empty arrays are the discard signal; a
    legitimate episode always emits at least one sample, so callers can
    treat "empty" as "discarded". Pass <= 0 to disable the bound.
    Applies to the money collectors and the cubeless body below; the
    matchplay serial collector deliberately does NOT bound episode
    length (matches are long and healthy; its vectorized path bounds
    per GAME instead).

    Targets:
      oneply=False: bootstrap from the value already computed during
                    selection (with_target=True), or terminal target
                    at episode end.
      oneply=True:  `agent.value_oneply_checker(state)` at every turn —
                    enumerates all 21 dice for a low-variance estimate.
                    Terminal handling is folded into the average
                    (winning moves contribute their value), so no
                    special terminal-target override is needed.

    Cubeful modes: if `mode.initial_match_state()` returns a
    MatchState, dispatches to the cubeful collector which also
    generates cube-decision samples.
    """
    initial_match_state = mode.initial_match_state()
    if initial_match_state is not None:
        if initial_match_state.is_matchplay:
            return _collect_episode_matchplay(
                agent, mode, rng, initial_match_state, oneply=oneply,
                cube_targets_1ply=cube_targets_1ply,
            )
        if getattr(mode, "take_decision_state", False):
            return _collect_episode_cubeful_money_take(
                agent, mode, rng, initial_match_state,
                oneply=oneply, cube_targets_1ply=cube_targets_1ply,
            )
        return _collect_episode_cubeful_money(
            agent, mode, rng, initial_match_state,
            oneply=oneply, cube_targets_1ply=cube_targets_1ply,
            max_episode_plies=max_episode_plies,
        )

    state, dice = opening_roll(rng)
    enc_list: List[np.ndarray] = []
    targets: List[float] = []
    plies = 0

    while not mode.is_episode_over(state):
        if max_episode_plies > 0 and plies >= max_episode_plies:
            # Runaway episode -- discard (see docstring).
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        plies += 1
        enc_list.append(agent.encode_state(state))

        if oneply:
            # 1-ply target for the current state (checker decision).
            targets.append(agent.value_oneply_checker(state))
            # Move selection at 0-ply (fast).
            result = agent.choose_checker_action(state, dice, with_target=False)
            if result is not None:
                _, next_state = result
                state = next_state
            else:
                state = switch_turn(state)
        else:
            # 0-ply target reused from selection.
            result = agent.choose_checker_action(state, dice, with_target=True)
            if result is not None:
                _, next_state, bootstrap = result
                if next_state.is_game_over():
                    outcome = mode.make_terminal_outcome(next_state)
                    targets.append(agent.terminal_target(outcome))
                    break
                targets.append(bootstrap)
                state = next_state
            else:
                opp_view = switch_turn(state)
                targets.append(agent.bootstrap_target(opp_view))
                state = opp_view

        dice = (rng.randint(1, 6), rng.randint(1, 6))

    if not enc_list:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.stack(enc_list),
        np.asarray(targets, dtype=np.float32),
    )


# ── Cubeful episode collection (money only) ──────────────────────────


def _collect_episode_cubeful_money(
    agent: Agent,
    mode: GameMode,
    rng: random.Random,
    match_state,
    oneply: bool = False,
    cube_targets_1ply: bool = False,
    max_episode_plies: int = DEFAULT_MAX_EPISODE_PLIES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cubeful MONEY-game episode collector.

    The cube decision is a proper TD transition: each turn that is
    eligible for doubling emits a cube-phase sample (is_cube_action=1)
    whose target is the checker-phase value after the cube action
    resolves:
      - No double:   target = V(state, is_cube_action=0, same perspective)
      - Double/pass: target = +1.0 (doubler wins 1 per unit)
      - Double/take: target = 2 * V(state, is_cube_action=0, theirs)

    The checker-phase sample (is_cube_action=0) follows as before, with
    bootstrap = -V(opp view) or terminal game_result at episode end.

    All targets are per-unit equity (cube_value = 1).

    `max_episode_plies`: see collect_episode. Exceeding it discards the
    episode (returns empty arrays). Pass <= 0 to disable.
    """
    from dataclasses import replace
    from encoding import CubePerspective
    from modes import CubeOwner, cube_perspective

    state, dice = opening_roll(rng)
    enc_list: List[np.ndarray] = []
    targets: List[float] = []
    is_opening = True  # no doubling on the opening roll
    plies = 0

    while not mode.is_episode_over(state):
        if max_episode_plies > 0 and plies >= max_episode_plies:
            # Runaway episode -- discard (see collect_episode docstring).
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        plies += 1
        player = state.turn

        # ── Cube decision phase (if eligible) ───────────────────────
        oneply_no_double = None  # cached for checker-phase reuse
        if not is_opening and match_state.can_offer(player):
            pre_persp = cube_perspective(match_state.cube_owner, player)

            # Emit cube-phase sample: is_cube_action=True
            enc_list.append(agent._encode_cubeful(
                state, pre_persp, is_cube_action=True,
            ))

            if oneply:
                opponent = 1 - player
                theirs_ms = replace(
                    match_state, cube_owner=CubeOwner(opponent + 1),
                )
                if cube_targets_1ply:
                    # EXPERIMENTAL: pure 1-ply cube targets. Conceptually
                    # the deeper search (1-ply lookahead at cube decisions),
                    # but empirically inflates slack at boundary positions
                    # via a subtle bug in value_oneply_checker_cubeful, which
                    # degrades cube performance vs the 0-ply-target default.
                    v_no_double = agent.value_oneply_checker_cubeful(
                        state, match_state,
                    )
                    oneply_no_double = v_no_double
                    v_double_take = 2.0 * agent.value_oneply_checker_cubeful(
                        state, theirs_ms,
                    )
                else:
                    # Default: 0-ply cube targets while keeping 1-ply for
                    # the checker phase. Avoids the 1-ply cube-target
                    # inflation that produced r8_tmp's cube_mEMG regression.
                    # Batched into one forward(batch=2) — saves one network
                    # dispatch per cube-eligible turn.
                    x_nd = agent._encode_cubeful(
                        state, pre_persp, is_cube_action=False,
                    )
                    x_dt = agent._encode_cubeful(
                        state, CubePerspective.THEIRS, is_cube_action=False,
                    )
                    t = torch.from_numpy(np.stack([x_nd, x_dt]))
                    t = agent._to_device(t)
                    with torch.no_grad():
                        values = agent.network(t).flatten()
                    v_no_double = float(values[0].item())
                    v_double_take = 2.0 * float(values[1].item())
                    # leave oneply_no_double = None so checker_target
                    # below recomputes via 1-ply (don't reuse 0-ply v_nd)
                do_double = agent._money_cube_decision(
                    v_no_double, v_double_take, pre_persp, match_state.jacoby,
                )
                offer = None
            else:
                offer = agent.offer_double(state, match_state)
                do_double = offer.should_double
                v_double_take = offer._cache["v_double_take_0ply"]
                v_no_double = offer._cache["v_no_double_0ply"]

            if do_double:
                if oneply:
                    takes = v_double_take <= 1.0
                else:
                    takes = agent.respond_to_double(
                        state, match_state, hint=offer,
                    )
                if takes:
                    # Cube target: stake doubles, cube transfers.
                    targets.append(v_double_take)
                    match_state = match_state.after_take(player)
                    oneply_no_double = None
                else:
                    # Cube target: opponent drops, we win 1 per unit.
                    targets.append(1.0)
                    break
            else:
                # No double: cube target = checker-phase value.
                targets.append(v_no_double)

        # ── Checker decision phase ──────────────────────────────────
        post_persp = cube_perspective(match_state.cube_owner, player)
        enc_list.append(agent._encode_cubeful(
            state, post_persp, is_cube_action=False,
        ))

        if oneply:
            if oneply_no_double is not None:
                checker_target = oneply_no_double
            else:
                checker_target = agent.value_oneply_checker_cubeful(
                    state, match_state,
                )
            targets.append(checker_target)
            result = agent.choose_checker_action_cubeful(
                state, dice, match_state,
            )
            if result is not None:
                next_state, _bootstrap = result
                if next_state.is_game_over():
                    break
                state = next_state
            else:
                state = switch_turn(state)
        else:
            result = agent.choose_checker_action_cubeful(
                state, dice, match_state,
            )
            if result is not None:
                next_state, bootstrap = result
                if next_state.is_game_over():
                    outcome = mode.make_terminal_outcome(
                        next_state, match_state,
                    )
                    targets.append(agent.terminal_target(outcome))
                    break
                targets.append(bootstrap)
                state = next_state
            else:
                opp_view = switch_turn(state)
                opp_persp = cube_perspective(
                    match_state.cube_owner, opp_view.turn,
                )
                opp_cube_action = match_state.can_offer(opp_view.turn)
                v_opp = agent.evaluate_cubeful(opp_view, opp_persp, is_cube_action=opp_cube_action)
                targets.append(-v_opp)
                state = opp_view

        dice = (rng.randint(1, 6), rng.randint(1, 6))
        is_opening = False

    if not enc_list:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.stack(enc_list),
        np.asarray(targets, dtype=np.float32),
    )


# ── Matchplay episode collection ───────────────────────────────────────

# Boundary-sample weight. Keep the default 1: boundary rows fire only
# on the winner's game-ending move (a conditioned branch of the
# pre-move state), so any weight > 1 biases the weighted fixed point
# upward near game ends -- upweighting is NOT fixed-point-neutral.
#
_BOUNDARY_UPWEIGHT = 1

# Cube-action exploration: eps of cube-eligible turns execute a random
# double and a random take. The cube-phase target is the OPTIMAL-action
# backup max(v_nd, min(v_dt, v_dp)) regardless of the executed action,
# so targets stay unbiased -- but exploration reshapes the state
# distribution. Default 0 -- and note --score-curriculum also defaults
# to 0, so a default run gets score coverage only from naturally
# reached scores: neither seeding nor cube exploration widens it. If a
# net's cube policy ever collapses to never-double, cube-state coverage
# starves; the env override is the escape hatch.
_CUBE_EXPLORATION_EPS = float(os.environ.get(
    "TRAINER_CUBE_EXPLORATION_EPS", "0.0"))

def _curriculum_start(initial_match_state, rng, p):
    """Sample a curriculum start score for one matchplay episode:
    with probability 1 - p the natural full-match start, otherwise a
    uniformly random reachable away-score pair, so every score context
    trains in parallel with the natural full-match distribution (at
    longer match lengths the intermediate scores otherwise rely on
    chained games to appear at all). 1-away scores split between the
    Crawford game itself and the post-Crawford continuation; (1,1) is
    always post-Crawford in a real match. `p` is
    MatchplayMode.score_curriculum (--score-curriculum); p == 0 means
    every episode starts at the natural full-match score."""
    from dataclasses import replace as _replace
    n = initial_match_state.match_length
    # Diagnostic override: pin EVERY episode's start score, e.g.
    # TRAINER_CURRICULUM_FORCE="1,1" (crawford_done set when either
    # side is 1-away, mirroring how such scores are reached in play).
    _force = os.environ.get("TRAINER_CURRICULUM_FORCE")
    if _force:
        fmy, fopp = (int(x) for x in _force.split(","))
        return _replace(
            initial_match_state, my_away=fmy, opp_away=fopp,
            crawford_done=(fmy == 1 or fopp == 1),
        )
    if n <= 1 or p <= 0.0 or rng.random() >= p:
        return initial_match_state
    my, opp = rng.randint(1, n), rng.randint(1, n)
    while (my, opp) == (n, n):
        my, opp = rng.randint(1, n), rng.randint(1, n)
    crawford_done = False
    if my == 1 or opp == 1:
        crawford_done = (my == 1 and opp == 1) or rng.random() < 0.5
    return _replace(
        initial_match_state, my_away=my, opp_away=opp,
        crawford_done=crawford_done,
    )


_ORACLE_CACHE = {}


def _get_oracle(spec):
    """Per-process p5jan oracle for supervised matchplay training.
    `spec` = prob5 checkpoint path, optional ':x=V' suffix. Cached:
    spawn workers construct it once and reuse across episodes."""
    ora = _ORACLE_CACHE.get(spec)
    if ora is None:
        from janowski_match import JanowskiMatchAgent
        from model import ProbNetwork
        path, x = spec, None
        if ":x=" in spec:
            path, _, xs = spec.rpartition(":x=")
            x = float(xs)
        # BG_ORACLE_DEVICE=cuda puts each worker's teacher net on the
        # GPU (~500MB ctx/worker) — the oracle legs otherwise leave it
        # idle, and value_oneply's ~350-row forwards are GPU-shaped.
        _dev = os.environ.get("BG_ORACLE_DEVICE", "cpu")
        ora = JanowskiMatchAgent(ProbNetwork.load(path), x_policy=x,
                                 device=_dev)
        # Shared GPU teacher: when this worker holds a slot on an
        # oracle-serving eval server, big forwards route there.
        if (_WORKER_REMOTE_EVAL is not None
                and _WORKER_REMOTE_KIND == "oracle"):
            ora._remote = _WORKER_REMOTE_EVAL
        _ORACLE_CACHE[spec] = ora
    return ora


def _oracle_value_from_probs(oracle, probs, state, gms, is_cube_action):
    """Oracle value of an encoded (state, flags) row given precomputed
    prob5 outputs. ca=1 with a live window: the optimal-action
    composition (what a pending cube decision is worth); otherwise the
    plain cubeful MWC from the on-roll player's view."""
    import janowski_match as _jm
    from modes import cube_perspective as _cp
    x = oracle._x(state)
    if is_cube_action and gms.can_offer(state.turn):
        d = _jm.cube_decision(probs, gms, state.turn, x, oracle.met)
        return max(d["v_nd"], min(d["v_dt"], d["v_dp"]))
    persp = _cp(gms.cube_owner, state.turn)
    ctx = _jm.ctx_for(gms, state.turn)
    return _jm.cubeful_mwc(probs, persp, ctx, x, oracle.met)


def _oracle_target(oracle, state, gms, is_cube_action, plies=0):
    """Oracle value of the state a collector row encodes. The uniform
    oracle-row invariant: EVERY emitted row's target equals this for
    its encoded (state, match_state, is_cube_action).

    plies=1 (--oracle-plies 1): the checker read is the oracle's
    1-ply value (JanowskiMatchAgent.value_oneply); a live ca=1 read
    composes max(v_nd, min(v_dt, v_dp)) from 1-ply branch values
    (v_dp stays exact MET) — the 0-ply cube_decision composition one
    ply deeper. Uniformity note: EVERY row of a plies=1 run must be
    priced at 1 ply (mixed-ply targets are the same class of bug as
    mixed conditioned/unconditioned targets); the ε-set move rows are
    therefore priced through this same funnel at the same depth."""
    if plies == 0:
        return _oracle_value_from_probs(
            oracle, oracle._probs_state(state), state, gms, is_cube_action)
    import janowski_match as _jm
    if is_cube_action and gms.can_offer(state.turn):
        v_nd = _oracle_oneply_memo(oracle, state, gms)
        v_dt = _oracle_oneply_memo(oracle, state,
                                   gms.after_take(state.turn))
        ctx = _jm.ctx_for(gms, state.turn)
        v_dp = _jm._me(oracle.met, ctx, ctx.cube, True)
        return max(v_nd, min(v_dt, v_dp))
    return _oracle_oneply_memo(oracle, state, gms)


# Within-step dedup for plies=1: at every cube-eligible step the
# collector prices 4 reads of which only 2 are distinct (the ca row's
# v_nd duplicates the state row; its v_dt duplicates the take-branch
# row). ~26 ms per 1-ply call makes that worth a tiny bounded memo;
# 64 entries comfortably covers a step, and eviction keeps long-lived
# workers flat.
_ORACLE_1PLY_MEMO = OrderedDict()
_ORACLE_1PLY_MEMO_MAX = 64


def _oracle_oneply_memo(oracle, state, gms):
    k = (id(oracle), state._key(), gms.cube_owner, gms.cube_value,
         gms.my_away, gms.opp_away, gms.is_crawford, gms.crawford_done,
         gms.match_length)
    memo = _ORACLE_1PLY_MEMO
    v = memo.get(k)
    if v is None:
        v = oracle.value_oneply(state, gms)
        memo[k] = v
        if len(memo) > _ORACLE_1PLY_MEMO_MAX:
            memo.popitem(last=False)
    return v


def _td_move_rows(agent, state, dice, gms, eps, chosen_succ):
    """Self-consistency rows for the near-tied successors the chooser
    evaluated and did NOT play.

    Only the successor actually played ever receives a TD row, and the
    passed-over siblings are not wrong at random: argmin selects
    whichever sibling is favourably erred, so V systematically
    UNDERVALUES the skipped ones while a skipped sibling's own 1-ply
    value is unbiased -- the correction exists, it just never becomes a
    row. Emitting a row for a successor does NOT play it: the
    trajectory is untouched -- unlike random exploration, which walks
    off the near-optimal manifold and measured harmful.

    Membership uses the cheap 0-ply values the chooser ranks on, so the
    set matches what actually decided the move; that is a selection rule,
    not a target. Rows are encoded exactly as both the chooser and the
    1-ply target read them (opponent perspective, opponent's
    cube-eligibility flag), and when that flag is live the target is
    composed as max(v_nd, min(v_dt, v_dp)) -- the same composition the
    cube rows use -- so the ca=1 channel is never trained toward a plain
    no-double value. Terminals are excluded: the chooser never consults
    the net on them, they are priced exactly by the MET.
    """
    from backgammon_engine import get_legal_plays

    mover = state.turn
    opp = 1 - mover
    plays = get_legal_plays(state, dice)
    if len(plays) < 2:
        return []
    succs = [s for _, s in plays]
    ca = gms.can_offer(opp)

    vals = [None] * len(succs)
    live = []
    for i, s in enumerate(succs):
        if s.is_game_over():
            new_ms = gms.after_game(s.game_result(), winner_is_me=(mover == 0))
            vals[i] = (1.0 if new_ms.match_over()
                       else agent._eval_new_game(new_ms, mover, scope="terminal"))
        else:
            vals[i] = 1.0 - agent.evaluate_matchplay(s, gms, is_cube_action=ca)
            live.append(i)
    best = max(vals)

    rows = []
    for i in live:
        if vals[i] < best - eps:
            continue
        s = succs[i]
        if chosen_succ is not None and s == chosen_succ:
            continue          # it becomes the next visited state anyway
        t = agent.value_oneply_checker_matchplay(s, gms)
        if ca:
            off = agent.offer_double_matchplay(s, gms)
            c = off._cache
            if c is not None:
                t = max(t, min(c["v_dt"], c["v_dp"]))
        rows.append((agent._encode_matchplay(s, gms, is_cube_action=ca), t))
    return rows


# [rows considered for deep pricing, rows actually deep-priced] under
# --oracle-1ply-eps. Per-process (each collection worker has its own), so
# read it in-process from a benchmark, not across the pool.
_DEEP_STATS = [0, 0]


def _oracle_move_rows(oracle, agent, state, dice, gms, eps, plies=0,
                      deep_eps=0.0, deep_stats=None):
    """Near-optimal action supervision: one row per legal successor in
    the UNION of
      (a) the oracle's eps-set: V_oracle(a) >= V_oracle(best) - eps
          (dense coverage of good moves), and
      (b) the student-competitive set: V_student(a) >=
          V_student(a_oracle_best) - eps (every move the student rates
          competitive with the oracle's choice -- including inflated
          favorites the oracle rates far below best, the reads that
          actually drive wrong argmaxes).
    Membership uses cheap 0-ply values on both sides; that is a
    selection rule, not a target.

    `deep_eps` (> 0) turns the cheap oracle into a FILTER for the
    expensive one, the way gnubg's 2-ply uses a top-K 1-ply filter:
    only successors with V_0ply(a) >= V_0ply(a_max) - deep_eps are
    re-priced at `plies`; the rest keep the 0-ply oracle value they
    already have. Cost falls with the width of the filter, because
    the 1-ply oracle call is ~20x a 0-ply one and dominates the leg.

    CAVEAT, deliberately loud: this MIXES target depths within a leg,
    which this file elsewhere calls the same class of defect as mixing
    conditioned and unconditional targets (see _oracle_target). It is
    safe only to the extent that 0-ply and 1-ply oracle values are the
    same quantity at different precision rather than systematically
    offset -- unmeasured. The invariant-preserving alternative is to
    DROP the filtered-out rows instead of 0-ply pricing them, which
    keeps targets uniform but loses the student-competitive rows that
    are the whole transport channel for deflating inflated favorites. Rows are encoded exactly as move
    selection reads them (opponent perspective, opponent's
    cube-eligibility flag) and every row's target comes from the
    _oracle_target funnel at `plies`, so the uniform-target invariant
    holds at either depth. Terminal successors count toward both
    anchors (exact MET value) but are not emitted -- the chooser never
    consults the net on them."""
    from backgammon_engine import get_legal_plays
    import janowski_match as _jm
    import td_agent as _tda

    mover = state.turn
    opp = 1 - mover
    plays = get_legal_plays(state, dice)
    if not plays:
        return [], None
    succs = [s for _, s in plays]
    ca = gms.can_offer(opp)
    if agent._c_base_available:
        feats = np.stack([_tda._bg_fast.encode_state(s) for s in succs])
    else:
        base = agent.encoder._base
        feats = np.stack([base.encode(s) for s in succs])
    probs_all = oracle._forward(feats)
    vals = []      # (mover-view value, opp-row target or None)
    ctx_me = _jm.ctx_for(gms, mover)
    for i, s in enumerate(succs):
        if s.is_game_over():
            v = _jm._me(oracle.met, ctx_me,
                        gms.cube_value * s.game_result(), True)
            vals.append((v, None))
        else:
            t = _oracle_value_from_probs(oracle, probs_all[i], s, gms, ca)
            vals.append((1.0 - t, t))
    v_best = max(v for v, _ in vals)
    i_best = max(range(len(vals)), key=lambda i: vals[i][0])

    # Student's mover-view value of every successor: the exact read its
    # chooser compares (terminal -> exact new-game/match value).
    enc_rows = [agent._encode_matchplay(s, gms, is_cube_action=ca)
                for s in succs]
    import torch as _torch
    with _torch.no_grad():
        opp_v = agent.network(
            _torch.from_numpy(np.stack(enc_rows))).reshape(-1).numpy()
    v_stu = np.empty(len(succs))
    for i, s in enumerate(succs):
        if s.is_game_over():
            new_ms = gms.after_game(s.game_result(),
                                    winner_is_me=(mover == 0))
            v_stu[i] = (1.0 if new_ms.match_over()
                        else agent._eval_new_game(new_ms, mover,
                                                  scope="terminal"))
        else:
            v_stu[i] = 1.0 - float(opp_v[i])
    stu_anchor = v_stu[i_best]

    rows = []
    for i, (v, t) in enumerate(vals):
        if t is None:
            continue  # terminal: never emitted
        if v < v_best - eps and v_stu[i] < stu_anchor - eps:
            continue
        if plies:
            deep = (deep_eps <= 0.0) or (v >= v_best - deep_eps)
            if deep:
                t = _oracle_target(oracle, succs[i], gms, ca, plies=plies)
            if deep_stats is not None:
                deep_stats[0] += 1
                deep_stats[1] += int(deep)
        rows.append((enc_rows[i], t))
    return rows, succs[i_best]


def _collect_episode_matchplay(
    agent: Agent,
    mode: GameMode,
    rng: random.Random,
    initial_match_state,
    oneply: bool = False,
    start: Optional[tuple] = None,
    cube_targets_1ply: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Matchplay episode collector. One episode = one full MATCH:
    games chain via MatchState.start_game/after_game until
    match_over(). All targets are P(on-roll player wins the MATCH)
    ∈ [0, 1]; cube_value never multiplies a target — it only drives
    score transitions.

    Per eligible turn, a cube-phase sample (is_cube_action=1) is
    emitted whose target is the MWC of the action actually taken:
      - No double:   v_nd  (playing on at the current cube)
      - Double/take: v_dt  (cube transferred, value doubled)
      - Double/pass: 1.0 if scoring cube_value points wins the match,
                     else the new-game MWC at the updated score
    The checker-phase sample (is_cube_action=0) follows with
    bootstrap = 1 - V(opp view of chosen successor); at game end the
    target bridges games: 1.0 if the match is won, else the new-game
    MWC at the updated score (the network is its own match-equity
    table).

    Perspective invariants (see td_agent matchplay section): the
    outer `match_state` and per-game `game_ms` always store away
    scores in WHITE's view; `after_game(..., winner_is_me=)` means
    "winner is WHITE"; all encode-time flipping happens inside
    agent._encode_matchplay keyed on state.turn.
    """
    from modes import cube_perspective  # noqa: F401  (parity with money collector)

    enc_list: List[np.ndarray] = []
    targets: List[float] = []
    match_state = initial_match_state

    # Score curriculum (see _curriculum_start).
    match_state = _curriculum_start(
        initial_match_state, rng, getattr(mode, "score_curriculum", 0.0))

    # Targeted-training entry: start=(state0, gms0) begins the FIRST
    # game mid-board at that exact position (cube phase live on the
    # first turn, dice rolled fresh); later games chain normally.
    # Everything else — targets, exploration, boundary handling — is
    # the production scheme, which is the point: probes measure what
    # real training would do at this position.
    if start is not None:
        match_state = start[1]
    _first_game = True

    # Supervised oracle mode: acting stays with `agent`; every target
    # below is REPLACED by the oracle value of the encoded state.
    oracle = None
    # Act-only: the teacher supplies MOVES but no targets -- transfers
    # the teacher's ranking via trajectories while the student keeps its
    # own 1-ply TD targets, so no teacher calibration error is imported.
    _act_only = bool(getattr(mode, "oracle_act_only", False))
    _oplies = int(getattr(mode, "oracle_plies", 0) or 0)
    _spec = getattr(mode, "oracle_spec", None)
    if _spec:
        oracle = _get_oracle(_spec)

    while True:
        if start is not None and _first_game:
            game_ms = start[1]
            state = start[0]
            dice = (rng.randint(1, 6), rng.randint(1, 6))
            is_opening = False  # mid-game: cube live if eligible
        else:
            game_ms = match_state.start_game()
            state, dice = opening_roll(rng)
            is_opening = True  # no doubling on the opening roll
        _first_game = False
        game_done = False

        while not game_done:
            player = state.turn

            # ── Cube decision phase (if eligible) ────────────────────
            if not is_opening and game_ms.can_offer(player):
                if oracle is not None and not _act_only:
                    # Double/take-branch supervision: the v_dt read
                    # (state at the doubled cube, owner flipped,
                    # ca=0) is what offer/respond_to_double consult
                    # and is never on-trajectory otherwise.
                    ms_take_row = game_ms.after_take(player)
                    enc_list.append(agent._encode_matchplay(
                        state, ms_take_row, is_cube_action=False))
                    targets.append(_oracle_target(
                        oracle, state, ms_take_row, False,
                        plies=_oplies))
                enc_list.append(agent._encode_matchplay(
                    state, game_ms, is_cube_action=True,
                ))
                offer = agent.offer_double_matchplay(state, game_ms)
                c = offer._cache
                # Cube-phase target: optimal-action backup, independent
                # of any exploration of the executed action below.
                #
                # DEPTH CONSISTENCY. offer_double_matchplay prices v_nd and
                # v_dt with 0-ply reads; under --oneply the checker rows
                # train toward a 21-dice enumeration, so composing cube
                # targets from 0-ply reads trains the two channels toward
                # different depths of the same quantity, and the
                # inconsistency grows as checker training moves V. v_dp is
                # an exact MET lookup and needs no re-pricing.
                if cube_targets_1ply and oneply:
                    v_nd = agent.value_oneply_checker_matchplay(state, game_ms)
                    v_dt = agent.value_oneply_checker_matchplay(
                        state, game_ms.after_take(player))
                    v_cube_opt = max(v_nd, min(v_dt, c["v_dp"]))
                else:
                    v_cube_opt = max(c["v_nd"], min(c["v_dt"], c["v_dp"]))
                if oracle is not None and not _act_only:
                    v_cube_opt = _oracle_target(
                        oracle, state, game_ms, True, plies=_oplies)

                do_double = offer.should_double
                explored = False
                if rng.random() < _CUBE_EXPLORATION_EPS:
                    do_double = rng.random() < 0.5
                    explored = True
                if do_double:
                    takes = agent.respond_to_double_matchplay(
                        state, game_ms, hint=offer,
                    )
                    if rng.random() < _CUBE_EXPLORATION_EPS:
                        takes = rng.random() < 0.5
                        explored = True
                    if takes:
                        targets.append(v_cube_opt)
                        game_ms = game_ms.after_take(player)
                    else:
                        # Opponent passes: doubler scores cube_value
                        # points (pre-double cube) and the game ends.
                        match_state = game_ms.after_game(
                            1, winner_is_me=(player == 0),
                        )
                        if explored or oracle is not None:
                            # Off-policy drop: keep the optimal backup;
                            # the grounded outcome reflects a random
                            # action, not the state's value. Under
                            # oracle supervision the state's value IS
                            # the target regardless of the action.
                            t = v_cube_opt
                        elif match_state.match_over():
                            t = 1.0
                        else:
                            t = agent._eval_new_game(
                                match_state, player, scope="boundary")
                        targets.append(t)
                        # Game-boundary samples carry the cross-game
                        # (match-equity) grounding — upweight greedy
                        # (on-policy) ones.
                        if not explored:
                            for _ in range(_BOUNDARY_UPWEIGHT - 1):
                                enc_list.append(enc_list[-1])
                                targets.append(t)
                        game_done = True
                        continue
                else:
                    targets.append(v_cube_opt)

            # ── Checker decision phase ───────────────────────────────
            _eps = getattr(mode, "oracle_move_eps", None)
            teacher_succ = None
            if oracle is not None and _eps is not None:
                _rows, teacher_succ = _oracle_move_rows(
                    oracle, agent, state, dice, game_ms, _eps,
                    plies=_oplies,
                    deep_eps=float(getattr(mode, "oracle_1ply_eps", 0.0) or 0.0),
                    deep_stats=_DEEP_STATS)
                if not _act_only:
                    for enc_row, tv in _rows:
                        enc_list.append(enc_row)
                        targets.append(tv)
            # Teacher-forced acting (DAgger beta-mix). The teacher's move
            # is already computed as the eps-set anchor, so acting on it
            # costs nothing. beta=0 is pure student acting (the historical
            # behaviour); beta=1 is pure behaviour cloning, which trades
            # our covariate shift for its mirror image -- the student then
            # never sees the states its OWN errors lead to. The value of
            # this switch is that the teacher's preferred successor is
            # exactly the state class the student undervalues and never
            # visits on-policy, so trajectories reach the subtrees the
            # eps-rows can only patch one row at a time.
            _act_p = float(getattr(mode, "oracle_act_prob", 0.0) or 0.0)
            force_teacher = (teacher_succ is not None and _act_p > 0.0
                             and rng.random() < _act_p)
            enc_list.append(agent._encode_matchplay(
                state, game_ms, is_cube_action=False,
            ))
            t_oracle = (None if (oracle is None or _act_only)
                        else _oracle_target(oracle, state, game_ms, False,
                                            plies=_oplies))
            result = agent.choose_checker_action_matchplay(
                state, dice, game_ms, rng=rng,
            )
            if force_teacher and result is not None:
                # Same successor set; only which one we walk into changes.
                result = (teacher_succ, result[1])
            if result is None:
                # Forced pass: dice go to the opponent.
                opp_view = switch_turn(state)
                if t_oracle is not None:
                    t = t_oracle
                elif oneply:
                    # Same convention: the 0-ply read below is
                    # conditioned on having danced; 1-ply rows keep
                    # the unconditional dice-average target.
                    t = agent.value_oneply_checker_matchplay(
                        state, game_ms)
                else:
                    opp_cube_action = game_ms.can_offer(opp_view.turn)
                    v_opp = agent.evaluate_matchplay(
                        opp_view, game_ms, is_cube_action=opp_cube_action,
                    )
                    t = 1.0 - v_opp
                targets.append(t)
                state = opp_view
            else:
                next_state, bootstrap = result
                if next_state.is_game_over():
                    game_result = next_state.game_result()
                    match_state = game_ms.after_game(
                        game_result, winner_is_me=(player == 0),
                    )
                    if t_oracle is not None:
                        # Oracle supervision: the row encodes the
                        # PRE-move state; its target is that state's
                        # oracle value, not the realized boundary.
                        t = t_oracle
                    elif oneply:
                        # 1-ply rows stay unconditional at the
                        # game-ending turn too: the enumeration prices
                        # terminal branches exactly, whereas the
                        # realized boundary is emitted only when the
                        # mover's roll won -- conditioning this row
                        # class on winning is an upward bias the
                        # bootstrap transports match-wide. The money
                        # collector never overrides 1-ply targets at
                        # terminals.
                        t = agent.value_oneply_checker_matchplay(
                            state, game_ms)
                    elif match_state.match_over():
                        # The mover just won the match; the sample is
                        # encoded from the mover's perspective, so the
                        # target is unconditionally 1.0.
                        t = 1.0
                    else:
                        t = agent._eval_new_game(
                            match_state, player, scope="boundary")
                    targets.append(t)
                    # Upweight the game-boundary grounding sample (see
                    # the drop path above).
                    for _ in range(_BOUNDARY_UPWEIGHT - 1):
                        enc_list.append(enc_list[-1])
                        targets.append(t)
                    game_done = True
                    continue
                if t_oracle is not None:
                    targets.append(t_oracle)
                elif oneply:
                    # 1-ply target: 21-dice enumeration with exact
                    # terminal (gammon-aware) pricing. Acting stays
                    # 0-ply via the chosen move above.
                    targets.append(
                        agent.value_oneply_checker_matchplay(
                            state, game_ms,
                        )
                    )
                else:
                    targets.append(bootstrap)
                # Self-consistency rows for the near-tied successors we
                # evaluated and did not play. Appended HERE, after this
                # state's own target: enc_list runs one ahead of targets
                # until that append, so inserting earlier would pair the
                # state row with the wrong target.
                _te = float(getattr(mode, "td_move_eps", 0.0) or 0.0)
                if _te > 0.0 and oracle is None:
                    for _enc, _tv in _td_move_rows(
                        agent, state, dice, game_ms, _te, next_state,
                    ):
                        enc_list.append(_enc)
                        targets.append(_tv)
                state = next_state

            dice = (rng.randint(1, 6), rng.randint(1, 6))
            is_opening = False

        if match_state.match_over():
            break

    if not enc_list:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    t_arr = np.asarray(targets, dtype=np.float32)
    return (
        np.stack(enc_list),
        t_arr,
    )


# ── Take-decision-state collector (CubefulTakeEncoder) ────────────────


def _collect_episode_cubeful_money_take(
    agent: Agent,
    mode: GameMode,
    rng: random.Random,
    match_state,
    oneply: bool = False,
    cube_targets_1ply: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cubeful MONEY-game collector with explicit take-decision state.

    For agents that use a CubefulTakeEncoder: the take/pass decision
    is a first-class node in the state graph (not an analytical
    threshold). Each cube offer can produce up to two transitions:

      (doubler, cube_offer, is_cube_action=1) → V(next state in doubler POV)
      (responder, take_decision, is_cube_take_decision=1) → V(next state in responder POV)

    Targets are the value of the actually-chosen next state (standard
    one-step TD; no explicit max anywhere). The +1 ceiling on the
    doubler's value-of-offering and the "too good" / Jacoby cases all
    emerge implicitly because V(take_decision) ≥ −1 at convergence
    (the responder always has the −1 pass alternative).

    All targets are per-unit equity at the current cube_value (the −1
    pass terminal is invariant under cube_value: absolute pass loss =
    1 × cube, per-unit = 1).

    Modes:
      oneply=False (default): all values are 0-ply model evaluations;
        decisions made by comparing model V at the relevant states.
      oneply=True, cube_targets_1ply=False (default 1-ply training):
        chequer targets come from `value_oneply_checker_cubeful`;
        cube_offer and take_decision targets and decisions use 0-ply
        model V at the relevant states. Mirrors the existing 1-ply
        training default in `_collect_episode_cubeful_money` —
        empirically 1-ply cube targets degrade cube_mEMG, so the
        default keeps cube targets tied to the model's current V.
      oneply=True, cube_targets_1ply=True: experimental, 1-ply
        everywhere (chequer + cube + take_decision).
    """
    from modes import cube_perspective

    state, dice = opening_roll(rng)
    enc_list: List[np.ndarray] = []
    targets: List[float] = []
    is_opening = True  # no doubling on the opening roll

    # When oneply=True, cache v_no_double_1ply from the cube phase so
    # the checker phase can reuse it (avoid a redundant 21-dice
    # enumeration on the same state).
    oneply_no_double = None
    pass_value = -1.0

    while not mode.is_episode_over(state):
        player = state.turn
        oneply_no_double = None  # reset per-turn

        # ── Cube decision phase (if eligible) ───────────────────────
        if not is_opening and match_state.can_offer(player):
            doubler_persp = cube_perspective(match_state.cube_owner, player)

            # Doubler's offer transition: (state, doubler POV, is_cube_action=1)
            enc_list.append(agent._encode_cubeful(
                state, doubler_persp,
                is_cube_action=True, is_cube_take_decision=False,
            ))

            responder = 1 - player
            responder_view = switch_turn(state)
            responder_persp = cube_perspective(
                match_state.cube_owner, responder,
            )

            # Cube/take_decision values for the doubler's offer and the
            # responder's take/pass: 0-ply by default, 1-ply only when
            # the user explicitly opts in via cube_targets_1ply=True.
            use_1ply_cube = oneply and cube_targets_1ply

            if use_1ply_cube:
                v_no_double = agent.value_oneply_checker_cubeful(
                    state, match_state,
                )
                oneply_no_double = v_no_double  # reusable for chequer phase
                post_take_ms = match_state.after_take(player)
                v_doubler_post = agent.value_oneply_checker_cubeful(
                    state, post_take_ms,
                )
                take_value = -2.0 * v_doubler_post
                if take_value > pass_value:
                    v_take_decision = take_value
                    takes_at_init = True
                else:
                    v_take_decision = pass_value
                    takes_at_init = False
                v_offered = -v_take_decision
            else:
                # 0-ply cube/take_decision values from the model directly.
                v_no_double = agent.evaluate_cubeful(
                    state, doubler_persp, is_cube_action=False,
                )
                v_take_decision = agent.evaluate_cubeful(
                    responder_view, responder_persp,
                    is_cube_action=False, is_cube_take_decision=True,
                )
                v_offered = -v_take_decision
                post_take_ms = None
                takes_at_init = None
                take_value = None

            do_double = v_offered > v_no_double

            if not do_double:
                # No-double: target = v_no_double. Under oneply=True,
                # cube_targets_1ply=False this is the 0-ply chequer V
                # — the cube_offer state's target stays 0-ply while
                # the next-emitted chequer transition target is 1-ply,
                # matching the existing collector's default.
                targets.append(v_no_double)
            else:
                # Double: target for the cube_offer state.
                targets.append(v_offered)

                # Take_decision transition: (responder POV, is_cube_take=1)
                enc_list.append(agent._encode_cubeful(
                    responder_view, responder_persp,
                    is_cube_action=False, is_cube_take_decision=True,
                ))

                if use_1ply_cube:
                    takes = takes_at_init
                else:
                    post_take_ms = match_state.after_take(player)
                    doubler_persp_post = cube_perspective(
                        post_take_ms.cube_owner, player,
                    )
                    v_doubler_post = agent.evaluate_cubeful(
                        state, doubler_persp_post, is_cube_action=False,
                    )
                    take_value = -2.0 * v_doubler_post
                    takes = take_value > pass_value

                if takes:
                    targets.append(take_value)
                    match_state = post_take_ms
                    oneply_no_double = None  # cube changed, cache stale
                else:
                    targets.append(pass_value)
                    break

        # ── Checker decision phase ──────────────────────────────────
        post_persp = cube_perspective(match_state.cube_owner, player)
        enc_list.append(agent._encode_cubeful(
            state, post_persp,
            is_cube_action=False, is_cube_take_decision=False,
        ))

        if oneply:
            if oneply_no_double is not None:
                checker_target = oneply_no_double
            else:
                checker_target = agent.value_oneply_checker_cubeful(
                    state, match_state,
                )
            targets.append(checker_target)
            result = agent.choose_checker_action_cubeful(
                state, dice, match_state,
            )
            if result is not None:
                next_state, _bootstrap = result
                if next_state.is_game_over():
                    break
                state = next_state
            else:
                state = switch_turn(state)
        else:
            result = agent.choose_checker_action_cubeful(
                state, dice, match_state,
            )
            if result is not None:
                next_state, bootstrap = result
                if next_state.is_game_over():
                    outcome = mode.make_terminal_outcome(
                        next_state, match_state,
                    )
                    targets.append(agent.terminal_target(outcome))
                    break
                targets.append(bootstrap)
                state = next_state
            else:
                opp_view = switch_turn(state)
                opp_persp = cube_perspective(
                    match_state.cube_owner, opp_view.turn,
                )
                opp_cube_action = match_state.can_offer(opp_view.turn)
                v_opp = agent.evaluate_cubeful(
                    opp_view, opp_persp,
                    is_cube_action=opp_cube_action,
                )
                targets.append(-v_opp)
                state = opp_view

        dice = (rng.randint(1, 6), rng.randint(1, 6))
        is_opening = False

    if not enc_list:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.stack(enc_list),
        np.asarray(targets, dtype=np.float32),
    )


# ── Parallel collection worker ────────────────────────────────────────


def _split_episodes(total: int, n_workers: int) -> List[int]:
    """Split *total* episodes as evenly as possible across n_workers."""
    base = total // n_workers
    remainder = total % n_workers
    return [base + (1 if i < remainder else 0) for i in range(n_workers)]


_WORKER_MET_CACHE = {}


def _worker_met(met_path):
    """Build (once per worker process) the MET for a collection worker.

    The table is passed as a path rather than an object so that workers
    re-parse it locally instead of shipping numpy arrays through the
    pool on every round.
    """
    if not met_path:
        return None
    hit = _WORKER_MET_CACHE.get(met_path)
    if hit is None:
        from met import MatchEquityTable
        hit = MatchEquityTable.from_xml(met_path)
        _WORKER_MET_CACHE[met_path] = hit
    return hit


# Set once per worker process by `_worker_init` when the GPU inference
# server is active: this worker's RemoteEvaluator (forwards -> server).
_WORKER_REMOTE_EVAL = None
_WORKER_REMOTE_KIND = "student"  # "student" | "oracle" (which net is served)
_WORKER_ROUTE_MOVES = False


# Printed when --bf16-collect is combined with --gpu-eval-server. An
# earlier revision said "only the local overflow fallback would use it",
# which is false: NO matchplay path consults the bf16 net. The bf16 copy
# is reached only through `TDAgent._forward_inference`, whose callers are
# all money-game paths; every matchplay value path (including the
# overflow fallback in `_matchplay_values_np`) calls `self.network`
# directly. `tests/test_gpu_eval_server.py::TestBf16DoesNotTouchMatchplay`
# pins that behaviourally, not by grep.
BF16_SERVER_NOTE = (
    "--bf16-collect has NO effect on matchplay collection: routed "
    "forwards are fp32 on the server, and every local matchplay forward "
    "(including the overflow fallback) uses the fp32 net directly. The "
    "bf16 inference copy is a money-game path only."
)

GPU_EVAL_COALESCE_NOTE = (
    "NOTE: this run is NOT bitwise reproducible. The server fuses "
    "concurrently-pending requests into one GEMM, fusion depends on "
    "wall-clock arrival order, and a last-ulp difference can flip a "
    "min() tie over candidate moves. Values are correct; a seeded rerun "
    "may not reproduce. Use --gpu-eval-deterministic if it must."
)

GPU_EVAL_DETERMINISTIC_NOTE = (
    "DETERMINISTIC mode: one request per GPU call, no cross-slot "
    "fusion. Routed collection is bit-exact against the unrouted "
    "collector and a seeded rerun reproduces, at the cost of batching."
)


def _worker_init(handles, kind="student"):
    """Pool initializer (GPU-server mode only): claim one shared-memory
    slot for this worker process and build its RemoteEvaluator.

    Runs ONCE per process, and the slot is never returned. That is why
    `train()` refuses to start the server when TRAINER_MAXTASKSPERCHILD
    is set: a recycled worker would claim a second slot, drain
    `free_slots`, and the next initializer would block forever.

    The bounded `get` below stops THIS process from blocking forever,
    but on its own it fixes nothing the operator can see: the exception
    is raised inside a replacement process that `multiprocessing.Pool`
    created and whose failure it swallows, so the master stays in
    `pool.map` while the pool respawns a fresh victim every ~60s. That
    is why the failure is also published on the shared `fail_event`,
    which `EvalServer.map_watched` polls and turns into `EvalWorkerGone`
    on the master.
    """
    global _WORKER_REMOTE_EVAL, _WORKER_ROUTE_MOVES, _WORKER_REMOTE_KIND
    _WORKER_REMOTE_KIND = kind
    import queue as _q
    from gpu_eval_server import RemoteEvaluator
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        slot = handles.free_slots.get(timeout=60)
    except _q.Empty:
        fail_event = getattr(handles, "fail_event", None)
        if fail_event is not None:
            fail_event.set()
        raise RuntimeError(
            "gpu-eval-server: no free shared-memory slot for this worker. "
            "The pool allocated exactly one slot per worker and slots are "
            "never returned, so this means a worker process was replaced "
            "(crash, or maxtasksperchild recycling)."
        )
    _WORKER_REMOTE_EVAL = RemoteEvaluator(slot, handles)
    _WORKER_ROUTE_MOVES = bool(getattr(handles, "route_moves", False))


def _collect_worker(args):
    """Worker subprocess: rebuild the agent from a state_dict, play
    `num_episodes` episodes, return
    (encodings, targets, n_discarded) — the concatenated numpy arrays
    plus the count of episodes discarded for exceeding
    `max_episode_plies` (money paths only; the matchplay collector is
    unbounded). The GameMode instance is pickled directly into
    `args` — simpler and more general than name-based dispatch,
    because modes can carry config (e.g. CubefulMoneyMode.jacoby).

    Set env LEAK_DIAG=1 to log RSS at entry/exit (for leak hunts).
    NOTE: the args tuple arity must stay in sync with
    `_build_worker_args` in Trainer.train.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # The 15 leading fields are the money-line layout (pinned by
    # tests/test_vectorized_collect.py and test_cuda_oneply_release.py);
    # the matchplay fields are appended and defaulted so a money-layout
    # tuple still unpacks. `activation` must be carried: the worker's
    # local net has to be the SAME FUNCTION as the master's, and with
    # --gpu-eval-server also the same function as the server's (which
    # does carry it). Defaulting to relu here while the server honoured
    # a tanh checkpoint mixed two different networks' values into one
    # episode's targets.
    (state_dict, hidden_sizes, input_size, output_mode, encoder_name,
     activation, mode, num_episodes, seed, oneply, oneply_acting,
     boltzmann_temp, bf16_inference, c_engine,
     max_episode_plies) = args[:15]
    # Matchplay extras (defaulted for money-layout tuples):
    cube_targets_1ply = args[15] if len(args) > 15 else False
    met_path = args[16] if len(args) > 16 else None
    met_scopes = args[17] if len(args) > 17 else None

    diag = os.environ.get("LEAK_DIAG") == "1"

    def _rss_kb():
        try:
            with open("/proc/self/statm") as f:
                # fields in pages: size resident shared text lib data dt
                resident = int(f.read().split()[1])
            return resident * (os.sysconf("SC_PAGE_SIZE") // 1024)
        except Exception:
            return -1

    rss_in = _rss_kb() if diag else None

    from model import TDNetwork
    from td_agent import TDAgent

    net = TDNetwork(
        hidden_sizes=hidden_sizes,
        input_size=input_size,
        output_mode=output_mode,
        encoder_name=encoder_name,
        activation=activation,
    )
    net.load_state_dict(state_dict)
    agent = TDAgent(
        net, oneply=oneply_acting, boltzmann_temp=boltzmann_temp,
        bf16_inference=bf16_inference, met=_worker_met(met_path),
        met_scopes=met_scopes,
    )
    # GPU eval server: route this agent's matchplay forwards through the
    # slot claimed in `_worker_init`. No-op unless the server is active.
    # The agent's own device stays 'cpu' on purpose -- only the forward
    # moves. Setting agent.device='cuda' would disable the C base-encode
    # fast path (`_c_base_available` requires a CPU device).
    if _WORKER_REMOTE_EVAL is not None and _WORKER_REMOTE_KIND == "student":
        agent._remote_eval = _WORKER_REMOTE_EVAL
        agent._remote_eval_moves = _WORKER_ROUTE_MOVES

    # C engine: replaces the 1-ply cubeful TARGET forward with libbg_nply
    # (OpenBLAS sgemm). Move selection stays PyTorch. No-op unless oneply.
    # Weights are this round's, so the per-process .bin export is current.
    #
    # enable_c_engine returns False when libbg_nply.so is missing, and
    # upstream IGNORES that -- so a run with --c-engine set silently degrades
    # to PyTorch and an A/B can be PyTorch on both arms. Fail loudly.
    if c_engine and oneply:
        import c_engine.bg_nply as _cn
        if not _cn.enable_c_engine(agent):
            raise RuntimeError(
                "--c-engine requested but libbg_nply is unavailable; "
                "build it with `bash c_engine/build_nply.sh` "
                "(needs a CBLAS exporting cblas_sgemm)"
            )

    rng = random.Random(seed)

    enc_chunks: List[np.ndarray] = []
    tgt_chunks: List[np.ndarray] = []
    n_discarded = 0
    for _ in range(num_episodes):
        encs, tgts = collect_episode(
            agent, mode, rng, oneply=oneply,
            cube_targets_1ply=cube_targets_1ply,
            max_episode_plies=max_episode_plies,
        )
        if len(encs) > 0:
            enc_chunks.append(encs)
            tgt_chunks.append(tgts)
        else:
            # Empty == discarded overlong episode (a legitimate episode
            # always emits at least one sample).
            n_discarded += 1
    if not enc_chunks:
        result = (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            n_discarded,
        )
    else:
        result = (
            np.concatenate(enc_chunks, axis=0),
            np.concatenate(tgt_chunks, axis=0),
            n_discarded,
        )

    if diag:
        rss_out = _rss_kb()
        print(
            f"[worker {os.getpid()}] RSS {rss_in/1024:.1f} -> {rss_out/1024:.1f} MB "
            f"(Δ {(rss_out-rss_in)/1024:+.1f} MB), {num_episodes} eps",
            flush=True,
        )
    return result



# ── Trainer class ─────────────────────────────────────────────────────


def _unique_ckpt_path(path: str) -> str:
    """Return `path`, or the first free `..._rN` variant if it already exists.

    Periodic checkpoints are named from an episode counter local to a single
    invocation. Resuming a partially-finished leg restarts that counter, so
    without this a resumed run silently overwrites the earlier run's
    checkpoints -- and the surviving file is then labelled with FEWER episodes
    than it actually contains, which is worse than losing it: it will be banked
    and compared under the wrong label.
    """
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{root}_r{n}{ext}"):
        n += 1
    return f"{root}_r{n}{ext}"


class Trainer:
    """Owns the optimizer + training loop. Bare agents (no optimizer)
    can be used for play / collection without constructing a Trainer.
    """

    def __init__(
        self,
        agent: Agent,
        lr: float = 1e-3,
        optimizer_cls: type = torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
        grad_clip: Optional[float] = None,
    ):
        """
        Args:
            agent: must expose `agent.network` so the optimizer can
                be constructed over its parameters.
            lr: learning rate.
            optimizer_cls: defaults to Adam. For online TD(0) prefer
                `torch.optim.SGD`.
            optimizer_kwargs: forwarded to the optimizer (e.g.
                `{"momentum": 0.9}`).
            grad_clip: max gradient norm. None disables clipping.
        """
        self.agent = agent
        self.optimizer = optimizer_cls(
            agent.network.parameters(), lr=lr, **(optimizer_kwargs or {}),
        )
        self.grad_clip = grad_clip

    # ── Per-step primitives ──────────────────────────────────────────

    def train_step(
        self,
        states_or_encoded: Union[Sequence[BoardState], np.ndarray],
        targets: Sequence,
    ) -> float:
        """Forward + loss + backward + optimizer step on a batch.
        Accepts either a list of BoardStates (slow path: agent
        encodes them) or a pre-encoded numpy array (fast path:
        skips encoding). Returns the scalar loss.
        """
        if isinstance(states_or_encoded, np.ndarray):
            pred = self.agent.forward_encoded(states_or_encoded)
        else:
            pred = self.agent.predict(states_or_encoded)
        target_tensor = self.agent.stack_targets(targets)
        loss = self.agent.loss(pred, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.agent.network.parameters(), self.grad_clip,
            )
        self.optimizer.step()
        # Weights changed: invalidate the matchplay new-game MWC memo
        # (only matters when the same agent keeps collecting in-process).
        memo = getattr(self.agent, "_new_game_memo", None)
        if memo:
            memo.clear()
        return loss.item()

    def eval_loss(
        self,
        states_or_encoded: Union[Sequence[BoardState], np.ndarray],
        targets: Sequence,
    ) -> float:
        """Loss on a batch without stepping. Used for held-out
        validation. Same input dispatch as train_step.
        """
        with torch.no_grad():
            if isinstance(states_or_encoded, np.ndarray):
                pred = self.agent.forward_encoded(states_or_encoded)
            else:
                pred = self.agent.predict(states_or_encoded)
            target_tensor = self.agent.stack_targets(targets)
            return self.agent.loss(pred, target_tensor).item()

    # ── Batch (round-based) TD training ──────────────────────────────

    def train(
        self,
        mode: GameMode,
        num_episodes: int,
        batch_size: int = 256,
        episodes_per_round: int = 100,
        epochs_per_round: int = 1,
        seed: Optional[int] = None,
        log_every: int = 1,
        workers: int = 1,
        oneply: bool = False,
        oneply_acting: bool = False,
        end_lr: Optional[float] = None,
        warmup_cycles: int = 0,
        boltzmann_temp: float = 0.0,
        bf16_collect: bool = False,
        save_path: Optional[str] = None,
        save_every: int = 0,
        metrics_out: Optional[dict] = None,
        cube_targets_1ply: bool = False,
        pipeline_collect: bool = False,
        c_engine: bool = False,
        met_path: Optional[str] = None,
        vectorized_collect: bool = False,
        vectorized_batch: int = 32,
        vectorized_device: str = "cuda",
        max_episode_plies: int = 0,
        met_scopes=None,
        gpu_eval_server: bool = False,
        oracle_eval_server: bool = False,
        gpu_eval_device: str = "cuda",
        gpu_eval_route_moves: bool = False,
        gpu_eval_deterministic: bool = False,
    ) -> List[float]:
        """Round-based batch TD(0). A round collects
        `episodes_per_round` episodes, then trains `epochs_per_round`
        shuffled passes over the round's pool in `batch_size` batches.

        `log_every` is in rounds. `workers > 1` distributes per-round
        collection across N subprocesses; the training step always
        runs in the master. `oneply=True` uses 1-ply lookahead targets
        (lower variance, slower per turn).

        `end_lr`: if set, linearly anneals the optimizer LR from its
        starting value down to `end_lr` over `num_episodes`. The
        starting LR is read from the optimizer at entry.

        `warmup_cycles`: if > 0, ramps the (annealed) LR from
        `lr * 0.1` up to the full scheduled LR over the first
        `warmup_cycles` rounds. For task transfers (DMP -> cubeless,
        cubeless -> cubeful), where a freshly-initialised head sits on
        a trained trunk; not for expansions.
        Composes with `end_lr` (multiplicative).

        `pipeline_collect`: when True (and workers > 1), overlap the
        worker collection of round N+1 with the master training step
        of round N. Workers for round N+1 see weights from end of
        round N-1 (one-round stale targets); the staleness is
        negligible at typical 1-ply LRs (1e-6 to 1e-5) but should be
        validated when changing recipes. Reproducibility is preserved
        (the rng.randint() sequence for worker seeds is unchanged).
        No effect with workers <= 1.

        `vectorized_collect`: OPT-IN. Matchplay + `--oneply` only.
        Lockstep generator driver (`vectorized_matchplay.py`): each
        worker runs `vectorized_batch` matches concurrently, batching
        their pending forwards into one GPU call per step. Inactive
        (printed WARN) unless every precondition holds: matchplay,
        oneply, workers > 1, vectorized_batch >= 1, a usable
        `vectorized_device`, enough FREE VRAM for `workers` driver
        contexts (`vectorized_matchplay.vram_reason`), boltzmann_temp
        == 0. Keep `episodes_per_round >= workers * vectorized_batch`
        for throughput (NOTE printed). Disables `pipeline_collect`
        (both want the GPU). Data is equivalent to the serial path
        (`tests/test_vectorized_matchplay.py`).

        `max_episode_plies`: safety bound on self-play episode length
        (see DEFAULT_MAX_EPISODE_PLIES). Money paths (single-process,
        parallel, pipelined, and the money GPU collector) DISCARD and
        count episodes exceeding it -- a degenerate policy otherwise
        hangs the round barrier while worker RSS grows until the OOM
        killer fires. On the MATCHPLAY vectorized path it is a per-GAME
        turn bound (a runaway game in lockstep pins a driver slot); the
        matchplay serial collector deliberately does not bound episode
        length, and the matchplay-vec INACTIVE fallback says so loudly.
        0 disables everywhere.

        `c_engine` (money + oneply): replace the 1-ply cubeful TARGET
        forward with the libbg_nply C engine in each worker (move
        selection stays PyTorch). Hard-errors if the .so is missing.

        `gpu_eval_server`: OPT-IN, the OTHER way to put matchplay
        1-ply collection on the GPU: ONE server process holds the only
        CUDA context and N pure-CPU workers ship feature matrices to it
        over per-worker shared memory. VRAM grows with the fused
        batch's width, not the worker count (far below vectorized, but
        NOT flat); the cost is a queue round-trip per forward. Routes
        `TDAgent.value_oneply_checker_matchplay` (96% of collected
        rows); add `gpu_eval_route_moves` to also route
        `choose_checker_action_matchplay` (4% of rows but 50% of
        forwards, so it roughly doubles the request rate). Never routes
        `evaluate_matchplay`, which is always batch-1 and would lose to
        IPC. HARD errors (not warnings) on `pipeline_collect`,
        `vectorized_collect`, and TRAINER_MAXTASKSPERCHILD; warns and
        falls back to the serial collector on any soft precondition
        (matchplay, oneply, workers > 1, a usable `gpu_eval_device`).

        `gpu_eval_deterministic`: run one request per GPU call instead
        of fusing whatever is concurrently pending. Fusion makes a
        routed run NOT bitwise reproducible -- which requests share a
        GEMM depends on wall-clock arrival order, fp32 reductions are
        batch-composition sensitive, and a last-ulp difference can flip
        a `min()` tie over candidate moves and change the trajectory.
        With this flag routed collection is bit-exact against the
        unrouted collector at any worker count, and a seeded rerun
        reproduces. Costs cross-slot fusion (worth ~1.67 requests per
        call at W=8, more at high W).

        Returns a flat list of per-batch losses.
        """
        mode.validate_agent(self.agent)

        # ── GPU eval server: HARD incompatibilities ───────────────────
        # These raise rather than warn-and-downgrade. Each is a
        # silent-corruption or silent-hang failure, not a performance
        # regression, so "the run completed and the loss curve looked
        # normal" is exactly the outcome that must not be reachable.
        _mt_env = os.environ.get("TRAINER_MAXTASKSPERCHILD")
        if gpu_eval_server:
            if pipeline_collect:
                raise ValueError(
                    "--gpu-eval-server is incompatible with "
                    "--pipeline-collect. Weights are pushed to the server "
                    "between rounds over the SAME queue the workers use, "
                    "so the push is only safe when no worker is "
                    "mid-request. Pipelined, round N+1's forwards overlap "
                    "round N's weight push and an arbitrary subset of "
                    "targets is computed against the wrong weight "
                    "generation -- the run completes, the loss curve "
                    "looks normal, and nothing surfaces the corruption. "
                    "Drop one of the two flags."
                )
            if vectorized_collect:
                raise ValueError(
                    "--gpu-eval-server is incompatible with "
                    "--vectorized-collect: they are two different GPU "
                    "collection designs (one central CUDA context vs one "
                    "per worker), and the vectorized worker replaces "
                    "_collect_worker entirely, so the server's slot "
                    "machinery would be attached to a pool that never "
                    "issues a request. Pick one."
                )
            if _mt_env:
                raise ValueError(
                    "--gpu-eval-server is incompatible with "
                    f"TRAINER_MAXTASKSPERCHILD={_mt_env!r}. Each worker "
                    "claims one shared-memory slot once, in the pool "
                    "initializer, and never returns it; a recycled worker "
                    "would drain the slot queue and the next worker would "
                    "block forever with no traceback. Unset the env var."
                )

        if isinstance(self.optimizer, torch.optim.SGD):
            import warnings
            warnings.warn(
                "SGD with batch TD may be unstable, Adam recommended.",
                stacklevel=2,
            )
        # ── Vectorized (lockstep GPU) collection preconditions ────────
        # Opt-in only; every failed precondition is named rather than
        # silently ignored, because falling back to the CPU path with a
        # flag set looks identical in the logs otherwise.
        vec_active = False
        money_vec_active = False
        vec_workers = workers
        _ims_vc = mode.initial_match_state()
        _mode_is_matchplay = _ims_vc is not None and _ims_vc.is_matchplay
        if vectorized_collect and not _mode_is_matchplay:
            # ── Money-line GPU collector (vectorized_collect.py) ──────
            # Lockstep generators; one local CUDA call per step per
            # worker. Eligibility + VRAM preflight run BEFORE pool
            # creation: pool tasks rotate across processes, so a
            # per-round task cap would NOT cap CUDA contexts. The pool
            # itself is sized to the capped worker count.
            if oneply_acting:
                # The vectorized generators implement 0-ply move
                # SELECTION only; silently ignoring --oneply-acting
                # would play a different policy than asked for and
                # change the data distribution, so refuse outright.
                raise ValueError(
                    "--vectorized-collect does not support --oneply-acting "
                    "(the vectorized collector does 1-ply TARGETS with "
                    "0-ply move selection). Drop one of the two flags."
                )
            money_vec_active = bool(
                workers > 1 and oneply and torch.cuda.is_available()
            )
            if not money_vec_active:
                reasons = []
                if not oneply:
                    reasons.append("--oneply is required (1-ply targets)")
                if workers <= 1:
                    reasons.append("--workers must be > 1")
                if not torch.cuda.is_available():
                    reasons.append("CUDA is unavailable")
                print(
                    "WARNING: --vectorized-collect requested but INACTIVE: "
                    + "; ".join(reasons)
                    + ". Falling back to the standard collector."
                )
            else:
                free_b, total_b = torch.cuda.mem_get_info()
                vec_workers = vectorized_vram_cap(workers, free_b)
                if vec_workers < 2:
                    print(
                        f"  WARN: --vectorized-collect IGNORED (VRAM "
                        f"preflight: {free_b / 2**20:.0f} MB free supports "
                        f"{vec_workers} worker context(s), need >= 2) -- "
                        f"falling back to the CPU collector", flush=True,
                    )
                    money_vec_active = False
                    vec_workers = workers
                elif vec_workers < workers:
                    print(
                        f"  NOTE: VRAM preflight capped vectorized workers "
                        f"{workers} -> {vec_workers} "
                        f"({free_b / 2**20:.0f} MB free, "
                        f"~{VEC_PER_WORKER_BYTES / 2**20:.0f} MB/context + "
                        f"{VEC_RESERVE_BYTES / 2**20:.0f} MB reserve). "
                        f"Pool is sized to the cap.", flush=True,
                    )
            if money_vec_active and pipeline_collect:
                print(
                    "  NOTE: --pipeline-collect disabled (worker drivers "
                    "and the master training step would contend for the "
                    "same GPU)."
                )
        elif vectorized_collect:
            import vectorized_matchplay as _vm
            _ims = mode.initial_match_state()
            _is_matchplay = _ims is not None and _ims.is_matchplay
            reasons = []
            if not _is_matchplay:
                reasons.append("mode is not matchplay")
            if not oneply:
                reasons.append("--oneply is required (1-ply targets)")
            if oneply_acting:
                reasons.append("--oneply-acting is not a matchplay option")
            if workers <= 1:
                reasons.append("--workers must be > 1")
            if vectorized_batch < 1:
                # A batch < 1 fills no driver slot: every worker would
                # return empty and the run would "succeed" having
                # trained on nothing (no rows, and no discards either,
                # so not even the discard log would show it).
                reasons.append(
                    f"--vectorized-batch must be >= 1 "
                    f"(got {vectorized_batch}; a batch < 1 collects "
                    "zero episodes)"
                )
            # Device string AND ordinal, not just torch.cuda.is_available():
            # a bad ordinal ('cuda:7') otherwise surfaces as a RuntimeError
            # inside pool.map, killing the run with no checkpoint.
            _dev_reason = _vm.device_reason(vectorized_device)
            if _dev_reason:
                reasons.append(_dev_reason)
            # Every driver process holds its OWN CUDA context (~500 MB),
            # so `workers` multiplies VRAM. Refuse here rather than let
            # round 1 die of OOM inside pool.map.
            _vram = _vm.vram_reason(vectorized_device, workers)
            if _vram:
                reasons.append(_vram)
            if float(getattr(mode, "td_move_eps", 0.0) or 0.0) > 0.0:
                reasons.append(
                    "--td-move-eps is implemented in the standard collector "
                    "only; the generator mirror emits no successor rows, so "
                    "the vectorized path would silently drop them"
                )
            if reasons:
                print(
                    "WARNING: --vectorized-collect requested but INACTIVE: "
                    + "; ".join(reasons)
                    + ". Falling back to the standard collector."
                )
                if max_episode_plies > 0:
                    # The ply bound lives in the vectorized collector
                    # only, so falling back silently drops it -- and the
                    # difference is total (every episode kept vs. every
                    # episode discarded), so it must not be inferred
                    # from the absence of a discard line.
                    print(
                        f"  NOTE: --max-episode-plies {max_episode_plies} "
                        "is a vectorized-path-only bound and is NOT "
                        "applied by the fallback collector: no episode "
                        "will be discarded for length."
                    )
            else:
                vec_active = True
                print(
                    f"Vectorized lockstep collection ON: "
                    f"batch={vectorized_batch} games/worker, "
                    f"device={vectorized_device}, workers={workers}"
                )
                if pipeline_collect:
                    print(
                        "  NOTE: --pipeline-collect disabled (worker "
                        "drivers and the master training step would "
                        "contend for the same GPU)."
                    )
                if bf16_collect:
                    print(
                        "  NOTE: --bf16-collect has no effect here (no "
                        "matchplay forward routes through the bf16 "
                        "inference net); the driver net is fp32."
                    )
                if max_episode_plies > 0:
                    print(
                        f"  NOTE: per-game ply bound {max_episode_plies}; "
                        "episodes exceeding it are DISCARDED whole."
                    )
                if episodes_per_round < workers * vectorized_batch:
                    # The whole speedup is the size of the fused batch.
                    # A round only reaches batch B if each worker is
                    # handed >= B episodes; below that the drivers run
                    # part-full and the round's tail is one long match.
                    # Measured: same config at 96 eps/round does 20-40
                    # ep/s vs 54-66 ep/s at 600.
                    print(
                        f"  NOTE: --episodes-per-round "
                        f"{episodes_per_round} < workers x batch "
                        f"({workers} x {vectorized_batch} = "
                        f"{workers * vectorized_batch}), so each worker "
                        f"gets only ~{episodes_per_round // workers} "
                        "episodes and the fused batches stay below B. "
                        "Raise --episodes-per-round to >= workers x "
                        "batch for the benchmarked throughput."
                    )

        rng = random.Random(seed)
        losses: List[float] = []
        start_lr = self.optimizer.param_groups[0]["lr"]
        if end_lr is not None:
            print(f"LR annealing: {start_lr:g} -> {end_lr:g} over {num_episodes} episodes")
        if warmup_cycles > 0:
            print(f"LR warmup: lr*0.1 -> lr over first {warmup_cycles} rounds")

        next_save = save_every if (save_path and save_every > 0) else None
        if next_save is not None:
            print(f"Checkpoint every {save_every} episodes -> {save_path} (with _ep{{N}} suffix)")

        # Worker pool created once for the whole run (avoids per-round
        # subprocess startup cost). None means single-process.
        #
        # We use the "spawn" start method, not the Linux default "fork".
        # With fork, replacement workers (and the initial workers
        # themselves) inherit the master's RSS at the moment of fork via
        # copy-on-write — and because the master's peak RSS scales with
        # pool size (pool_enc + sh_enc + pool.map deserialization
        # buffers), a replacement worker spawned mid-training starts at
        # multiple GB, and at 128 workers a single anomalous-pool round
        # can push combined RSS over a terabyte. Spawn gives every
        # worker a pristine interpreter that only re-imports torch and
        # the model modules, so per-worker RSS is bounded by the
        # worker's own workload, not the master's history.
        #
        # TRAINER_MAXTASKSPERCHILD env var recycles workers after N
        # calls — useful if you observe residual per-worker growth over
        # a long run (glibc arena fragmentation, torch allocator cache).
        # With spawn, recycled workers start pristine, so this is now a
        # real knob rather than a diagnostic stub.
        _mt = os.environ.get("TRAINER_MAXTASKSPERCHILD")
        maxtasks = int(_mt) if _mt else None
        _ctx = mp.get_context("spawn")

        # ── GPU eval server: SOFT preconditions ───────────────────────
        # Same house style as the vectorized block: every failed
        # precondition is named, because a flag that silently does
        # nothing looks identical in the logs to one that worked.
        eval_srv = None
        if gpu_eval_server:
            import gpu_eval_server as _ges
            _ims = mode.initial_match_state()
            reasons = []
            if not (_ims is not None and _ims.is_matchplay):
                reasons.append("mode is not matchplay")
            if not oneply:
                reasons.append("--oneply is required (1-ply targets)")
            if workers <= 1:
                reasons.append("--workers must be > 1")
            _dev_reason = _ges.device_reason(gpu_eval_device)
            if _dev_reason:
                reasons.append(_dev_reason)
            if reasons:
                print(
                    "WARNING: --gpu-eval-server requested but INACTIVE: "
                    + "; ".join(reasons)
                    + ". Falling back to the standard collector."
                )
            else:
                _net_m = self.agent.network
                n_feat = self.agent.encoder.num_features
                with torch.no_grad():
                    _probe = _net_m(
                        torch.zeros(1, n_feat, device=self.agent.device)
                    )
                out_dim = 1 if _probe.dim() == 1 else _probe.shape[-1]
                max_rows = int(os.environ.get(
                    "GPU_EVAL_MAX_ROWS", _ges.MAX_ROWS_DEFAULT))
                max_batch_rows = int(os.environ.get(
                    "GPU_EVAL_MAX_BATCH_ROWS", _ges.MAX_BATCH_ROWS_DEFAULT))
                eval_srv = _ges.EvalServer(
                    _ctx, num_slots=workers, n_feat=n_feat, out_dim=out_dim,
                    max_rows=max_rows,
                    build_spec=_ges.build_spec_from(_net_m),
                    init_state_dict=_net_m.state_dict(),
                    device=gpu_eval_device,
                    route_moves=gpu_eval_route_moves,
                    wait_timeout=float(os.environ.get(
                        "GPU_EVAL_TIMEOUT", _ges._WAIT_TIMEOUT_DEFAULT)),
                    coalesce=not gpu_eval_deterministic,
                    max_batch_rows=max_batch_rows,
                )
                eval_srv.start()
                print(
                    f"GPU inference server ON: {workers} worker slots, fp32 "
                    f"on {gpu_eval_device}, max_rows={max_rows} "
                    f"({eval_srv.shm_bytes() / 2**20:.0f} MB of /dev/shm), "
                    f"routing 1-ply"
                    + (" + move selection" if gpu_eval_route_moves else "")
                )
                # Reproducibility is a property of the run, not a
                # footnote: say which one this run has, every time.
                print("  " + (
                    GPU_EVAL_DETERMINISTIC_NOTE if gpu_eval_deterministic
                    else GPU_EVAL_COALESCE_NOTE
                ))
                if bf16_collect:
                    print("  NOTE: " + BF16_SERVER_NOTE)

        _srv_kind = "student"
        if oracle_eval_server and eval_srv is None:
            import gpu_eval_server as _ges
            _spec_o = getattr(mode, "oracle_spec", None)
            reasons = []
            if not _spec_o:
                reasons.append("--oracle-targets is required")
            if workers <= 1:
                reasons.append("--workers must be > 1")
            if _mt_env:
                reasons.append(
                    f"TRAINER_MAXTASKSPERCHILD={_mt_env!r} is set "
                    "(slots are claimed once per process)")
            _dev_reason = _ges.device_reason(gpu_eval_device)
            if _dev_reason:
                reasons.append(_dev_reason)
            if reasons:
                print(
                    "WARNING: --oracle-eval-server requested but INACTIVE: "
                    + "; ".join(reasons)
                    + ". Teacher forwards stay on worker CPUs."
                )
            else:
                from model import ProbNetwork as _PN
                _p_path = _spec_o
                if ":x=" in _p_path:
                    _p_path = _p_path.rpartition(":x=")[0]
                _pnet = _PN.load(_p_path)
                with torch.no_grad():
                    _probe = _pnet(torch.zeros(1, _pnet.input_size))
                _out_dim = 1 if _probe.dim() == 1 else _probe.shape[-1]
                eval_srv = _ges.EvalServer(
                    _ctx, num_slots=workers, n_feat=_pnet.input_size,
                    out_dim=_out_dim,
                    max_rows=int(os.environ.get(
                        "GPU_EVAL_MAX_ROWS", _ges.MAX_ROWS_DEFAULT)),
                    build_spec=_ges.build_spec_from(_pnet),
                    init_state_dict=_pnet.state_dict(),
                    device=gpu_eval_device,
                    route_moves=False,
                    wait_timeout=float(os.environ.get(
                        "GPU_EVAL_TIMEOUT", _ges._WAIT_TIMEOUT_DEFAULT)),
                    max_batch_rows=int(os.environ.get(
                        "GPU_EVAL_MAX_BATCH_ROWS",
                        _ges.MAX_BATCH_ROWS_DEFAULT)),
                )
                eval_srv.start()
                _srv_kind = "oracle"
                print(
                    f"ORACLE GPU server ON: teacher {_p_path} "
                    f"({_pnet.input_size}-in/{_out_dim}-out) on "
                    f"{gpu_eval_device}, {workers} worker slots "
                    f"({eval_srv.shm_bytes() / 2**20:.0f} MB /dev/shm); "
                    "teacher is frozen — no weight pushes."
                )

        if workers <= 1:
            pool = None
        elif eval_srv is not None:
            # maxtasksperchild is FORCED None here: slots are claimed
            # once per process and never returned. The env var was
            # already rejected above, so this is belt-and-braces.
            pool = _ctx.Pool(
                processes=workers, maxtasksperchild=None,
                initializer=_worker_init,
                initargs=(eval_srv.client_handles(), _srv_kind),
            )
        else:
            pool = _ctx.Pool(
                processes=(vec_workers if money_vec_active else workers),
                maxtasksperchild=maxtasks,
            )

        # Master-side leak diagnostic: RSS at pool creation + per-round
        # delta. Uses the same /proc/self/statm trick as the worker.
        _diag_master = os.environ.get("LEAK_DIAG") == "1"

        def _master_rss_mb():
            try:
                with open("/proc/self/statm") as f:
                    resident = int(f.read().split()[1])
                return resident * (os.sysconf("SC_PAGE_SIZE") // 1024) / 1024.0
            except Exception:
                return -1.0

        if _diag_master:
            print(
                f"[master {os.getpid()}] RSS at pool start: "
                f"{_master_rss_mb():.1f} MB",
                flush=True,
            )
        net = self.agent.network
        hidden_sizes = list(net.hidden_sizes)
        input_size = getattr(net, "input_size", 196)
        output_mode = getattr(net, "output_mode", "probability")
        encoder_name = getattr(net, "encoder_name", "perspective196")
        activation = getattr(net, "activation", "relu")

        # Helper: build worker_args for a given round_size using current
        # net weights. Called either eagerly (pipeline pre-submit) or
        # lazily (sync path).
        def _build_worker_args(round_size: int):
            splits = _split_episodes(round_size, workers)
            # .clone() is REQUIRED, not defensive. On a CPU net `.cpu()` is a
            # no-op returning a VIEW of the live parameter, so the tensors
            # handed to map_async are pickled by the pool's task thread WHILE
            # optimizer.step() mutates them — workers can receive torn
            # weights. Not reachable on --device cuda (there .cpu() genuinely
            # copies). Cost is one ~2MB copy per round.
            state_dict = {
                k: v.detach().cpu().clone()
                for k, v in net.state_dict().items()
            }
            return [
                (state_dict, hidden_sizes, input_size, output_mode,
                 encoder_name, activation,
                 mode, n_eps, rng.randint(0, 2**31 - 1),
                 oneply, oneply_acting, boltzmann_temp,
                 bf16_collect, c_engine, max_episode_plies,
                 cube_targets_1ply, met_path, met_scopes)
                for n_eps in splits
            ]

        # Twin of _build_worker_args for the vectorized path. Keeps the
        # same leading 14 fields (so the two stay reviewable side by
        # side) and consumes rng.randint identically, so worker seeding
        # is unchanged; appends (B, device, max_episode_plies).
        def _build_vec_args(round_size: int):
            splits = _split_episodes(round_size, workers)
            # .clone() REQUIRED — same torn-weights hazard as
            # _build_worker_args above.
            state_dict = {
                k: v.detach().cpu().clone()
                for k, v in net.state_dict().items()
            }
            return [
                (state_dict, hidden_sizes, input_size, output_mode,
                 encoder_name,
                 mode, n_eps, rng.randint(0, 2**31 - 1),
                 oneply, oneply_acting, boltzmann_temp,
                 bf16_collect, cube_targets_1ply, met_path,
                 vectorized_batch, vectorized_device, max_episode_plies,
                 met_scopes, activation)
                for n_eps in splits
            ]

        # Money-line twin for vectorized_collect.py. NOTE: the tuple
        # arity/order must stay in sync with `vectorized_collect_worker`
        # (it hardcodes "cuda": the money eligibility above requires
        # torch.cuda.is_available()).
        def _build_vec_args_money(round_size: int):
            splits = _split_episodes(round_size, vec_workers)
            # .clone() REQUIRED — same torn-weights hazard as
            # _build_worker_args above.
            state_dict = {
                k: v.detach().cpu().clone()
                for k, v in net.state_dict().items()
            }
            return [
                (state_dict, hidden_sizes, input_size, output_mode,
                 encoder_name, activation,
                 mode, n_eps, rng.randint(0, 2**31 - 1),
                 oneply, oneply_acting, boltzmann_temp,
                 bf16_collect,
                 vectorized_batch, "cuda",
                 max_episode_plies)
                for n_eps in splits
            ]

        # Pipeline pre-roll: if pipelining and we have a worker pool,
        # start round 0's collect before entering the loop. The loop
        # will then wait for "pending" results at the top of each
        # iteration and submit the NEXT round's collect before training.
        # No-op when pool is None or pipeline_collect is False.
        use_pipeline = (pipeline_collect and pool is not None
                        and not vec_active and not money_vec_active)
        pending = None
        pending_round_size = 0
        if use_pipeline:
            pending_round_size = min(episodes_per_round, num_episodes)
            pending = pool.map_async(
                _collect_worker, _build_worker_args(pending_round_size),
            )

        try:
            num_rounds = math.ceil(num_episodes / episodes_per_round)
            episodes_played = 0
            for round_idx in range(num_rounds):
                round_size = min(
                    episodes_per_round, num_episodes - episodes_played,
                )
                rss_round_start = _master_rss_mb() if _diag_master else 0.0

                # ── LR annealing (linear, by episodes_played) ──────
                if end_lr is not None:
                    frac = episodes_played / max(num_episodes, 1)
                    current_lr = start_lr + (end_lr - start_lr) * frac
                else:
                    current_lr = start_lr

                # ── LR warmup (first `warmup_cycles` rounds) ───────
                if warmup_cycles > 0 and round_idx < warmup_cycles:
                    warmup_frac = (round_idx + 1) / warmup_cycles
                    current_lr = current_lr * (0.1 + 0.9 * warmup_frac)

                if end_lr is not None or warmup_cycles > 0:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = current_lr

                # ── Collect this round into (encodings, targets) ───
                _t_collect_start = time.perf_counter()
                enc_chunks: List[np.ndarray] = []
                tgt_chunks: List[np.ndarray] = []
                n_discarded = 0

                if vec_active:
                    # Lockstep GPU drivers. Own unpack branch: the
                    # vectorized worker returns a 3-tuple (the extra
                    # field is the ply-guard discard count).
                    from vectorized_matchplay import (
                        vectorized_matchplay_worker,
                    )
                    for w_enc, w_tgt, w_disc in pool.map(
                        vectorized_matchplay_worker,
                        _build_vec_args(round_size),
                    ):
                        n_discarded += w_disc
                        if len(w_enc) > 0:
                            enc_chunks.append(w_enc)
                            tgt_chunks.append(w_tgt)
                    if n_discarded:
                        print(
                            f"  round {round_idx}: {n_discarded}/"
                            f"{round_size} episodes DISCARDED "
                            f"(exceeded {max_episode_plies} plies/game)"
                        )
                elif money_vec_active:
                    # Money GPU collection: each worker drives
                    # `vectorized_batch` games in lockstep and batches
                    # their forwards into one local GPU call. Returns
                    # the same (enc, tgt, n_discarded) 3-tuple as
                    # _collect_worker, so accounting below is shared.
                    from vectorized_collect import vectorized_collect_worker
                    for w_enc, w_tgt, w_disc in pool.map(
                        vectorized_collect_worker,
                        _build_vec_args_money(round_size),
                    ):
                        if len(w_enc) > 0:
                            enc_chunks.append(w_enc)
                            tgt_chunks.append(w_tgt)
                        n_discarded += w_disc
                elif pool is None:
                    # Single-process: master agent does the collection,
                    # so its bf16 inference copy (if enabled) must be
                    # re-synced to the current fp32 weights each round —
                    # and its matchplay new-game memo (net-answered
                    # entries) invalidated, per the documented invariant.
                    if hasattr(self.agent, "clear_matchplay_memo"):
                        self.agent.clear_matchplay_memo()
                    if bf16_collect:
                        if not self.agent.bf16_inference:
                            self.agent.bf16_inference = True
                        self.agent.refresh_bf16_inference()
                    # Same for the C engine: it holds a weight COPY, so it
                    # must be re-exported each round or targets would come
                    # from the previous round's net.
                    if c_engine and oneply:
                        import c_engine.bg_nply as _cn
                        if not _cn.enable_c_engine(self.agent):
                            raise RuntimeError(
                                "--c-engine requested but libbg_nply is "
                                "unavailable; build with "
                                "`bash c_engine/build_nply.sh`"
                            )
                    for _ in range(round_size):
                        encs, tgts = collect_episode(
                            self.agent, mode, rng, oneply=oneply,
                            cube_targets_1ply=cube_targets_1ply,
                            max_episode_plies=max_episode_plies,
                        )
                        if len(encs) > 0:
                            enc_chunks.append(encs)
                            tgt_chunks.append(tgts)
                        else:
                            # Empty == discarded overlong episode.
                            n_discarded += 1
                elif use_pipeline:
                    # Pipelined: this round's collect was started in the
                    # previous iteration (or pre-loop for round 0). Submit
                    # the NEXT round's collect BEFORE training, so workers
                    # run in parallel with the master's GPU train step.
                    # The next round's workers see CURRENT weights (V_N
                    # before this iter trains), making their targets one
                    # round stale relative to a strict sync schedule.
                    next_pending = None
                    next_round_size = 0
                    if round_idx + 1 < num_rounds:
                        next_round_size = min(
                            episodes_per_round,
                            num_episodes - (episodes_played + round_size),
                        )
                        next_pending = pool.map_async(
                            _collect_worker, _build_worker_args(next_round_size),
                        )
                    for _row in pending.get():
                        w_enc, w_tgt = _row[0], _row[1]
                        if len(w_enc) > 0:
                            enc_chunks.append(w_enc)
                            tgt_chunks.append(w_tgt)
                        # 3rd field (discard count) is absent on legacy
                        # 2-tuples (e.g. test fakes) -- treat as 0.
                        n_discarded += _row[2] if len(_row) > 2 else 0
                    pending = next_pending
                    pending_round_size = next_round_size
                else:
                    # Non-pipelined parallel: submit and wait synchronously.
                    # The GPU server must hold THIS round's weights before
                    # any worker issues a forward: workers also build a
                    # local net from the same state_dict (for the paths
                    # that stay local), and a version skew between the two
                    # is a correctness bug, not just a staleness one.
                    if eval_srv is not None and _srv_kind == "student":
                        eval_srv.update_weights(net.state_dict())
                    # With the server active the round's map is POLLED
                    # (`map_watched`) rather than blocking: a SIGKILLed
                    # pool worker or a dead server otherwise stalls the
                    # master indefinitely, and because `pool.map` never
                    # returns, the `finally` never runs and /dev/shm is
                    # never unlinked either.
                    _results = (
                        eval_srv.map_watched(
                            pool, _collect_worker,
                            _build_worker_args(round_size),
                        )
                        if eval_srv is not None
                        else pool.map(
                            _collect_worker, _build_worker_args(round_size),
                        )
                    )
                    for _row in _results:
                        w_enc, w_tgt = _row[0], _row[1]
                        if len(w_enc) > 0:
                            enc_chunks.append(w_enc)
                            tgt_chunks.append(w_tgt)
                        # 3rd field (discard count) is absent on legacy
                        # 2-tuples (e.g. test fakes) -- treat as 0.
                        n_discarded += _row[2] if len(_row) > 2 else 0

                episodes_played += round_size
                if metrics_out is not None:
                    metrics_out.setdefault("collection_times", []).append(
                        time.perf_counter() - _t_collect_start
                    )
                    metrics_out.setdefault("round_sizes", []).append(round_size)
                    metrics_out.setdefault("discarded", []).append(n_discarded)
                if not enc_chunks:
                    # Nothing usable this round. If that is because every
                    # episode ran away, say so loudly -- otherwise the run
                    # would spin here producing no training signal at all.
                    if n_discarded:
                        print(
                            f"Round {round_idx + 1:4d}/{num_rounds} "
                            f"| episodes {episodes_played}/{num_episodes} "
                            f"| pool     0 "
                            f"| DISCARDED {n_discarded}/{round_size} overlong "
                            f"episodes (>{max_episode_plies} plies) -- "
                            f"no usable samples this round",
                            flush=True,
                        )
                    continue

                pool_enc = np.concatenate(enc_chunks, axis=0)
                pool_tgt = np.concatenate(tgt_chunks, axis=0)
                _t_train_start = time.perf_counter()
                # Release the per-worker chunks — pool_enc/pool_tgt own
                # a copy. Without this, chunks + pool arrays are both
                # live during the training step (≈2× pool bytes, and
                # pool bytes scale with pool size, so on an anomalous-
                # length round this was a GB-class redundant footprint).
                enc_chunks = None
                tgt_chunks = None
                n = len(pool_enc)

                # Fancy-index per batch: a materialized `pool_enc[perm]`
                # would double peak RSS for the round, and the per-batch
                # op cost is negligible next to forward/backward.
                for _epoch in range(epochs_per_round):
                    perm = np.arange(n)
                    rng_np = np.random.default_rng(rng.randint(0, 2**31 - 1))
                    rng_np.shuffle(perm)
                    for start in range(0, n, batch_size):
                        idx = perm[start:start + batch_size]
                        batch_e = pool_enc[idx]
                        batch_t = pool_tgt[idx]
                        losses.append(self.train_step(batch_e, batch_t))

                if metrics_out is not None:
                    metrics_out.setdefault("train_times", []).append(
                        time.perf_counter() - _t_train_start
                    )

                if log_every and (round_idx + 1) % log_every == 0:
                    recent = losses[-10:] if losses else [0.0]
                    avg = sum(recent) / len(recent)
                    discard_note = (
                        f" | discarded {n_discarded} overlong episodes"
                        f" (>{max_episode_plies} plies)"
                        if n_discarded else ""
                    )
                    print(
                        f"Round {round_idx + 1:4d}/{num_rounds} "
                        f"| episodes {episodes_played}/{num_episodes} "
                        f"| pool {n:5d} "
                        f"| recent batch loss {avg:.3e}"
                        f"{discard_note}"
                    )

                if next_save is not None and episodes_played >= next_save:
                    root, ext = os.path.splitext(save_path)
                    ckpt_path = _unique_ckpt_path(
                        f"{root}_ep{episodes_played}{ext or '.pt'}")
                    self.agent.network.save(ckpt_path)
                    print(f"  -> checkpoint saved: {ckpt_path}", flush=True)
                    while next_save <= episodes_played:
                        next_save += save_every

                if _diag_master:
                    rss_round_end = _master_rss_mb()
                    print(
                        f"[master round={round_idx + 1}] RSS "
                        f"{rss_round_start:.1f} -> {rss_round_end:.1f} MB "
                        f"(Δ {rss_round_end - rss_round_start:+.1f}) "
                        f"| losses={len(losses)}",
                        flush=True,
                    )
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            # AFTER the pool is joined, never before: stop() unlinks the
            # shared-memory segments the workers have mapped, and pulling
            # them out from under a live worker is a segfault.
            if eval_srv is not None:
                eval_srv.stop()

        return losses

    # ── Online TD(0) training ────────────────────────────────────────

    def train_online(
        self,
        mode: GameMode,
        num_episodes: int,
        seed: Optional[int] = None,
        log_every: int = 100,
    ) -> List[float]:
        """Online TD(0): one update per transition via the agent's
        `td_update` fast path. Reads `lr` from the Trainer's
        optimizer but does not use it for the update — batch-of-1
        updates don't benefit from optimizer state. Pair with SGD.

        Dispatches to the cubeful variant when the mode carries a
        MatchState.

        Returns a flat list of per-step squared TD errors.
        """
        mode.validate_agent(self.agent)
        if mode.initial_match_state() is not None:
            return self._train_online_cubeful_money(
                mode, num_episodes, seed=seed, log_every=log_every,
            )

        rng = random.Random(seed)
        lr = self.optimizer.param_groups[0]["lr"]
        losses: List[float] = []
        agent = self.agent  # local alias

        for episode in range(1, num_episodes + 1):
            state, dice = opening_roll(rng)
            while not mode.is_episode_over(state):
                result = agent.choose_checker_action(
                    state, dice, with_target=True,
                )
                if result is not None:
                    _, next_state, bootstrap = result
                    if next_state.is_game_over():
                        outcome = mode.make_terminal_outcome(next_state)
                        target = agent.terminal_target(outcome)
                        losses.append(agent.td_update(state, target, lr))
                        break
                    losses.append(agent.td_update(state, bootstrap, lr))
                    state = next_state
                else:
                    opp_view = switch_turn(state)
                    target = agent.bootstrap_target(opp_view)
                    losses.append(agent.td_update(state, target, lr))
                    state = opp_view
                dice = (rng.randint(1, 6), rng.randint(1, 6))

            if log_every and episode % log_every == 0:
                recent = losses[-100:] if losses else [0.0]
                avg = sum(recent) / len(recent)
                print(f"Episode {episode:6d} | recent step loss {avg:.4f}")

        return losses

    def _train_online_cubeful_money(
        self,
        mode: GameMode,
        num_episodes: int,
        seed: Optional[int] = None,
        log_every: int = 100,
    ) -> List[float]:
        """Online TD(0) for cubeful money games. Mirrors the cubeful
        collector's D1/D3 per-turn sample generation, but does a
        `td_update_encoded` on each sample as it's produced — so the
        very next forward pass (checker selection after a take, or
        the next turn's cube decision) uses the updated weights.
        """
        rng = random.Random(seed)
        lr = self.optimizer.param_groups[0]["lr"]
        losses: List[float] = []
        agent = self.agent

        for episode in range(1, num_episodes + 1):
            state, dice = opening_roll(rng)
            match_state = mode.initial_match_state()
            is_opening = True  # no doubling on the opening roll

            while not mode.is_episode_over(state):
                player = state.turn

                # ── Cube decision phase (if eligible) ──────────
                if not is_opening and match_state.can_offer(player):
                    pre_persp = cube_perspective(
                        match_state.cube_owner, player,
                    )
                    cube_enc = agent._encode_cubeful(
                        state, pre_persp, is_cube_action=True,
                    )
                    offer = agent.offer_double(state, match_state)
                    ply_tag = "1ply" if agent.oneply else "0ply"
                    if offer.should_double:
                        v_double_take = offer._cache[f"v_double_take_{ply_tag}"]
                        takes = agent.respond_to_double(
                            state, match_state, hint=offer,
                        )
                        if takes:
                            losses.append(
                                agent.td_update_encoded(
                                    cube_enc, v_double_take, lr)
                            )
                            match_state = match_state.after_take(player)
                        else:
                            losses.append(
                                agent.td_update_encoded(cube_enc, 1.0, lr)
                            )
                            break
                    else:
                        # No double: cube target = checker-phase value.
                        v_no_double = offer._cache[f"v_no_double_{ply_tag}"]
                        losses.append(
                            agent.td_update_encoded(
                                cube_enc, v_no_double, lr)
                        )

                # ── Checker decision phase ─────────────────────
                post_persp = cube_perspective(match_state.cube_owner, player)
                enc = agent._encode_cubeful(
                    state, post_persp, is_cube_action=False,
                )
                result = agent.choose_checker_action_cubeful(
                    state, dice, match_state,
                )
                if result is not None:
                    next_state, bootstrap = result
                    if next_state.is_game_over():
                        outcome = mode.make_terminal_outcome(
                            next_state, match_state,
                        )
                        target = agent.terminal_target(outcome)
                        losses.append(
                            agent.td_update_encoded(enc, target, lr)
                        )
                        break
                    losses.append(
                        agent.td_update_encoded(enc, bootstrap, lr)
                    )
                    state = next_state
                else:
                    opp_view = switch_turn(state)
                    opp_persp = cube_perspective(
                        match_state.cube_owner, opp_view.turn,
                    )
                    opp_cube_action = match_state.can_offer(opp_view.turn)
                    v_opp = agent.evaluate_cubeful(
                        opp_view, opp_persp,
                        is_cube_action=opp_cube_action,
                    )
                    losses.append(
                        agent.td_update_encoded(enc, -v_opp, lr)
                    )
                    state = opp_view

                dice = (rng.randint(1, 6), rng.randint(1, 6))
                is_opening = False

            if log_every and episode % log_every == 0:
                recent = losses[-100:] if losses else [0.0]
                avg = sum(recent) / len(recent)
                print(f"Episode {episode:6d} | recent step loss {avg:.4f}")

        return losses
