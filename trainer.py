"""
trainer.py -- Trainer class for backgammon self-play TD learning.

Trainer pairs an Agent with a GameMode and owns the optimizer and
training loop. The Agent owns its representation (forward pass,
target shape, loss).

Two methods:
  - `train`: round-based batch TD(0). Adam by default. Parallel
    collection via `workers=N`; the C collectors (money: bgc_collect,
    match play: bgc_collect_match), and for match play the lockstep
    GPU collector (vectorized_matchplay.py) and the GPU eval server
    (gpu_eval_server.py).
  - `train_online`: online TD(0), one update per transition. Pair
    with SGD via `optimizer_cls=torch.optim.SGD`.

`collect_episode` is also a free function for tests and external
callers.
"""

import atexit
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from agents import Agent
from backgammon_engine import BoardState, opening_roll, switch_turn
from cuda_graph_step import (ENV as GRAPH_ENV, GraphedTrainStep, mode_from_env,
                             env_explicit as graph_env_explicit,
                             env_label as graph_env_label,
                             resolve_mode as resolve_graph_mode)
from encoding import CubePerspective
from modes import GameMode, cube_perspective
from pool_guard import (PoolWatch, WorkerDied, announce_restart,
                        max_pool_restarts, shutdown_pool, GuardedPool)


# Safety bound on self-play episode length, in plies (turns), for the
# money collectors. Match play has its own per-game bound (0 = none,
# the default; see Trainer.train's `max_episode_plies`).
DEFAULT_MAX_EPISODE_PLIES = 1500


# ── Episode collection (mode-agnostic, agent-driven) ──────────────────


def collect_episode(
    agent: Agent,
    mode: GameMode,
    rng: random.Random,
    oneply: bool = False,
    max_episode_plies: int = DEFAULT_MAX_EPISODE_PLIES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Play one self-play episode and return (encodings, targets).

    Cubeless move selection is always 0-ply. Cubeful respects
    agent.oneply.

    Targets:
      oneply=False: the bootstrap value computed during selection, or
                    the terminal target at episode end.
      oneply=True:  `agent.value_oneply_checker(state)` every turn
                    (average over all 21 dice; terminal moves are
                    folded into the average).

    Cubeful modes (`mode.initial_match_state()` is a MatchState)
    dispatch to the cubeful collector, which also emits cube-decision
    samples; match play (a MatchState with match_length > 0) to the
    match collector.

    `max_episode_plies` (money modes): an episode exceeding it is
    DISCARDED (empty arrays returned), not truncated. A legitimate
    episode always emits at least one sample, so empty means discarded.
    <= 0 disables the bound. The match collector ignores it.
    """
    initial_match_state = mode.initial_match_state()
    if initial_match_state is not None and initial_match_state.is_matchplay:
        return _collect_episode_matchplay(
            agent, mode, rng, initial_match_state, oneply=oneply,
        )
    if initial_match_state is not None:
        return _collect_episode_cubeful_money(
            agent, mode, rng, initial_match_state, oneply=oneply,
            max_episode_plies=max_episode_plies,
        )

    state, dice = opening_roll(rng)
    enc_list: List[np.ndarray] = []
    targets: List[float] = []
    plies = 0

    while not mode.is_episode_over(state):
        if max_episode_plies > 0 and plies >= max_episode_plies:
            # Runaway episode: discard.
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        plies += 1
        enc_list.append(agent.encode_state(state))

        if oneply:
            # 1-ply target; 0-ply move selection.
            targets.append(agent.value_oneply_checker(state))
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


class _CubeDecision(NamedTuple):
    """Resolved cube-decision phase. enc/target: the cube-phase sample.
    match_state: after the cube action (transfers on a take). dropped:
    double/pass, the episode ends here."""
    enc: np.ndarray
    target: float
    match_state: object
    dropped: bool


def _resolve_cube_decision(
    agent,
    state: BoardState,
    match_state,
    pre_persp,
    do_double: bool,
    takes: bool,
    v_no_double: float,
    v_double_take: float,
) -> _CubeDecision:
    """What the cube-decision phase emits. Every cubeful collector
    resolves its cube phase here (the 1-ply generator below, driven by
    both the CPU and the vectorized collector, and the CPU 0-ply
    branch), so the emission semantics cannot drift between them.

      double/take:  target = v_double_take, cube transfers. Always
                    emitted (cube ownership is a live input).
      double/pass:  target = +1.0, episode ends. Always emitted.
      no double:    target = v_no_double.
    """
    player = state.turn
    if do_double:
        enc = agent._encode_cubeful(state, pre_persp, is_cube_action=True)
        if takes:
            return _CubeDecision(
                enc, v_double_take, match_state.after_take(player), False,
            )
        return _CubeDecision(enc, 1.0, match_state, True)
    enc = agent._encode_cubeful(state, pre_persp, is_cube_action=True)
    return _CubeDecision(enc, v_no_double, match_state, False)


def _cube_phase_oneply_gen(agent, state: BoardState, match_state):
    """The cube-decision phase of both 1-ply cubeful-money collectors.

    Yields one (2, n_feat) stack [no-double, double-take], receives the
    two fp32 values, returns a _CubeDecision. The CPU collector drives
    it against the agent's own net (_drive_cube_phase_local). The
    vectorized collector delegates via `yield from` so its lockstep
    driver batches the forward.

    Cube targets are direct net evaluations (no dice to average over),
    both in one forward(batch=2).
    """
    player = state.turn
    pre_persp = cube_perspective(match_state.cube_owner, player)
    x_nd = agent._encode_cubeful(state, pre_persp, is_cube_action=False)
    x_dt = agent._encode_cubeful(
        state, CubePerspective.THEIRS, is_cube_action=False,
    )
    values = yield np.stack([x_nd, x_dt])
    values = np.asarray(values, dtype=np.float32).ravel()
    v_no_double = float(values[0])
    v_double_take = 2.0 * float(values[1])
    do_double = agent._money_cube_decision(
        v_no_double, v_double_take, pre_persp, match_state.jacoby,
    )
    takes = (v_double_take <= 1.0) if do_double else False
    return _resolve_cube_decision(
        agent, state, match_state, pre_persp,
        do_double, takes, v_no_double, v_double_take,
    )


def _drive_cube_phase_local(agent, gen) -> _CubeDecision:
    """Run _cube_phase_oneply_gen to completion with the agent's own
    net doing the single forward."""
    feats = next(gen)
    t = agent._to_device(torch.from_numpy(feats))
    with torch.no_grad():
        values = agent.network(t).flatten()
    try:
        gen.send(values.cpu().numpy())
    except StopIteration as e:
        return e.value
    raise RuntimeError("cube phase generator yielded more than once")


def _collect_episode_cubeful_money(
    agent: Agent,
    mode: GameMode,
    rng: random.Random,
    match_state,
    oneply: bool = False,
    max_episode_plies: int = DEFAULT_MAX_EPISODE_PLIES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cubeful money-game episode collector.

    Each turn eligible for doubling emits a cube-phase sample
    (is_cube_action=1) whose target is the checker-phase value after
    the cube action resolves:
      - No double:   V(state, is_cube_action=0, same perspective)
      - Double/pass: +1.0
      - Double/take: 2 * V(state, is_cube_action=0, theirs)

    Then the checker-phase sample (is_cube_action=0) with bootstrap
    -V(opp view), or the terminal target at episode end.

    All targets are per-unit equity (cube_value = 1).

    `max_episode_plies`: see collect_episode.
    """
    state, dice = opening_roll(rng)
    enc_list: List[np.ndarray] = []
    targets: List[float] = []
    is_opening = True  # no doubling on the opening roll
    plies = 0

    while not mode.is_episode_over(state):
        if max_episode_plies > 0 and plies >= max_episode_plies:
            # Runaway episode: discard.
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        plies += 1
        player = state.turn

        # ── Cube decision phase (if eligible) ───────────────────────
        if not is_opening and match_state.can_offer(player):
            if oneply:
                # Shared with the vectorized collector, driven locally.
                dec = _drive_cube_phase_local(agent, _cube_phase_oneply_gen(
                    agent, state, match_state,
                ))
            else:
                pre_persp = cube_perspective(match_state.cube_owner, player)
                offer = agent.offer_double(state, match_state)
                do_double = offer.should_double
                takes = agent.respond_to_double(
                    state, match_state, hint=offer,
                ) if do_double else False
                dec = _resolve_cube_decision(
                    agent, state, match_state, pre_persp,
                    do_double, takes,
                    offer._cache["v_no_double_0ply"],
                    offer._cache["v_double_take_0ply"],
                )
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

        if oneply:
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

# Duplication factor for game-boundary samples in the matchplay
# collector. The match-equity grounding (score-transition value) only
# enters at game boundaries; upweighting accelerates its propagation
# into the score-feature columns without changing the fixed point.
# Boundary-sample weight. HISTORY: introduced as 3 with the
# claim it accelerates match-equity propagation "without changing the
# fixed point" -- that claim is FALSE and was the root cause of a TD
# degradation seen in training. Boundary rows fire only on the
# winner's game-ending move (a conditioned branch of the pre-move
# state); the fixed-weights drift probe measured them at +13 mMWC on a
# calibrated net, and tripling them biases the weighted fixed point
# upward near game ends (0-ply TD from the oracle net: -2.1pp vs
# parent at weight 3, exactly 0.0 at weight 1, n=20k). Default is now
# the unbiased 1; the env override reproduces historical runs.
_BOUNDARY_UPWEIGHT = int(os.environ.get("TRAINER_BOUNDARY_UPWEIGHT", "1"))

# Cube-action exploration rate during matchplay collection. Without
# it, a noise-level early cube policy can double nearly every game,
# making every 2pt match a single doubled game — Crawford and
# post-Crawford scores then never appear in the data and their values
# (the implicit MET) never train. The cube-phase sample target is the
# OPTIMAL-action backup max(v_nd, min(v_dt, v_dp)) regardless of the
# executed action, so exploration diversifies trajectories without
# corrupting the value definition.
# Cube exploration (added with the boundary
# upweight): eps of cube-eligible turns execute a random double and a
# random take. Targets stay unbiased (optimal-action backup), but it
# feeds --outcome-mix's z with a mis-cubing policy's outcomes and
# reshapes the state distribution. Default 0
# (author's call): the score curriculum handles score coverage, and a
# calibrated net's greedy cube play generates doubled-cube states
# naturally. Env override reproduces historical runs (0.25). If a
# future net's cube policy collapses to never-double, cube-state
# coverage starves again -- the override is the escape hatch.
_CUBE_EXPLORATION_EPS = float(os.environ.get(
    "TRAINER_CUBE_EXPLORATION_EPS", "0.0"))

# Checker-move exploration rate. The collector was fully greedy on checker
# play, so the TD leg only ever visited states its own argmax led to: a
# closed loop whose 1-ply residual sits at ~3e-6 from the first round, with
# no mechanism to discover the errors it is self-consistent about.
# Exploration is target-safe for the same reason the cube version is: under
# --oneply the checker target is value_oneply_checker_matchplay(state), a max
# over successors computed from the STATE, so it is unchanged by which move
# is executed. In the 0-ply path the bootstrap comes from the chosen
# successor, so the collector backs up the GREEDY successor's value while
# executing the explored one -- ordinary off-policy TD, not the random
# policy's value.
_MOVE_EXPLORATION_EPS = float(os.environ.get(
    "TRAINER_MOVE_EXPLORATION_EPS", "0.0"))

# Fraction of matchplay episodes that start directly at a later score
# (uniform over reachable away-score pairs) so every score context
# trains in parallel with the natural full-match distribution. The
# original 2pt fix seeded only 1-away scores; at longer match lengths
# the intermediate scores (e.g. (2,3) and (2,2) at 3pt) otherwise rely
# on chained games and exploration passes to appear at all.
_SCORE_CURRICULUM_P = float(os.environ.get(
    "TRAINER_SCORE_CURRICULUM_P", "0.5"))


def _curriculum_start(initial_match_state, rng):
    """Sample a curriculum start score for one matchplay episode:
    with probability 1 - _SCORE_CURRICULUM_P the natural full-match
    start, otherwise a uniformly random reachable away-score pair.
    1-away scores split between the Crawford game itself and the
    post-Crawford continuation; (1,1) is always post-Crawford in a
    real match."""
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
    if n <= 1 or rng.random() >= _SCORE_CURRICULUM_P:
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

    The 1-ply target at `state` is an average over 21 dice of a max over
    that dice's successors, so ONE target consumes ~540 successor reads
    (measured on real self-play) while only the successor actually played
    ever receives a row. Those read-only states are not wrong at random:
    argmin selects whichever sibling is favourably erred, so the ones
    passed over stay stale. Measured over 128 positions that
    gnubg confirms at both 2 and 4 ply: V undervalues the skipped sibling
    by +12.30 +/- 1.64 mMWC, while that sibling's OWN 1-ply value is
    statistically unbiased against rollout truth (+0.36 +/- 2.27). So the
    correction we need already exists -- it just never becomes a row.

    Emitting a row for a successor does NOT play it. The trajectory is
    untouched, which is what separates this from --move-explore-eps:
    random exploration walks off the near-optimal manifold (measured
    harmful, monotone in temperature, two bases), whereas the untrained
    mass IS the near-optimal siblings.

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
          actually drive wrong argmaxes; targeted-probe diagnosis).
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

    from dataclasses import replace as _replace

    enc_list: List[np.ndarray] = []
    targets: List[float] = []
    viewers: List[int] = []  # sample viewer (player), for outcome_mix
    match_state = initial_match_state

    # Score curriculum (see _curriculum_start).
    match_state = _curriculum_start(initial_match_state, rng)

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
    # Act-only: the teacher supplies MOVES but no targets. p5jan outplays the
    # champion at checker by 5.84 mMWC yet distilling its VALUES damages the
    # net (-1.85/-2.05), i.e. its ranking is better than its calibration. This
    # mode transfers the ranking (via trajectories) and keeps the student's own
    # 1-ply TD targets, so no calibration error is imported.
    # Indices of the on-trajectory CHECKER rows, so --td-eps-only can drop
    # them at the end. Recorded rather than skipped inline because targets
    # are appended from several branches; filtering once at the end keeps
    # enc_list/targets/viewers aligned by construction.
    _traj_rows = []
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
                    viewers.append(player)
                    targets.append(_oracle_target(
                        oracle, state, ms_take_row, False,
                        plies=_oplies))
                enc_list.append(agent._encode_matchplay(
                    state, game_ms, is_cube_action=True,
                ))
                viewers.append(player)
                offer = agent.offer_double_matchplay(state, game_ms)
                c = offer._cache
                # Cube-phase target: optimal-action backup, independent
                # of any exploration of the executed action below. v_nd
                # and v_dt are offer_double_matchplay's 0-ply reads (also
                # under --oneply); v_dp is an exact MET lookup.
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
                                viewers.append(player)
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
                        viewers.append(1 - player)
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
            _traj_rows.append(len(enc_list))
            enc_list.append(agent._encode_matchplay(
                state, game_ms, is_cube_action=False,
            ))
            viewers.append(player)
            t_oracle = (None if (oracle is None or _act_only)
                        else _oracle_target(oracle, state, game_ms, False,
                                            plies=_oplies))
            result = agent.choose_checker_action_matchplay(
                state, dice, game_ms, rng=rng,
            )
            if force_teacher and result is not None:
                # Same successor set; only which one we walk into changes.
                result = (teacher_succ, result[1])
            _mx = float(getattr(mode, "move_explore_eps", 0.0) or 0.0)
            if _mx <= 0.0:
                _mx = _MOVE_EXPLORATION_EPS
            if (result is not None and _mx > 0.0 and not force_teacher
                    and rng.random() < _mx):
                # Execute a random legal move; keep the greedy successor's
                # bootstrap so the target is still the optimal-action backup.
                from backgammon_engine import get_legal_plays as _glp
                _plays = _glp(state, dice)
                if len(_plays) > 1:
                    _, _rs = _plays[rng.randrange(len(_plays))]
                    result = (_rs, result[1])
            if result is None:
                # Forced pass: dice go to the opponent.
                opp_view = switch_turn(state)
                if t_oracle is not None:
                    t = t_oracle
                elif oneply:
                    # Same convention: the 0-ply read below is
                    # conditioned on having danced; 1-ply rows keep
                    # the unconditional dice-average target.
                    t = agent.value_oneply_checker_matchplay(state, game_ms)
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
                        # class on winning (+20 mMWC vs T1 at
                        # oracle-v2 weights), an upward bias the
                        # bootstrap transports match-wide. The money
                        # collector never overrides 1-ply targets at
                        # terminals.
                        t = agent.value_oneply_checker_matchplay(state, game_ms)
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
                        viewers.append(player)
                    game_done = True
                    continue
                _te = float(getattr(mode, "td_move_eps", 0.0) or 0.0)
                if t_oracle is not None:
                    targets.append(t_oracle)
                elif oneply:
                    # 1-ply target: 21-dice enumeration with exact
                    # terminal (gammon-aware) pricing. Acting stays
                    # 0-ply via the chosen move above.
                    targets.append(agent.value_oneply_checker_matchplay(state, game_ms))
                else:
                    targets.append(bootstrap)
                # Self-consistency rows for the near-tied successors we
                # evaluated and did not play. Appended HERE, after this
                # state's own target: enc_list runs one ahead of targets
                # until that append, so inserting earlier would pair the
                # state row with the wrong target.
                if _te > 0.0 and oracle is None:
                    for _enc, _tv in _td_move_rows(
                        agent, state, dice, game_ms, _te, next_state,
                    ):
                        enc_list.append(_enc)
                        targets.append(_tv)
                        viewers.append(1 - player)
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
    mix = getattr(mode, "outcome_mix", 0.0)
    if mix > 0.0:
        # Ground every target with the realized match outcome from the
        # sample viewer's perspective. Away scores are WHITE's view.
        assert len(viewers) == len(targets)
        white_won = match_state.my_away <= 0
        z = np.asarray(
            [1.0 if (white_won if v == 0 else not white_won) else 0.0
             for v in viewers],
            dtype=np.float32,
        )
        t_arr = (1.0 - mix) * t_arr + mix * z
    if bool(getattr(mode, "td_eps_only", False)) and _traj_rows:
        # Keep the eps rows (successor -> its own 1-ply value), the cube
        # rows and the game-boundary rows; drop the on-trajectory checker
        # targets.
        #
        # NOT "unbootstrapped": T1 is computed from the network's own V, so
        # the eps rows bootstrap too. This changes WHICH self-referential
        # operator is iterated, nothing more. Worth distinguishing because
        # every TD variant tried degrades a converged net with dose, which
        # is what iterating toward a fixed point WORSE than the oracle-set
        # weights looks like -- and batch loss already sits at ~1.6e-6
        # (RMSE ~1.3 mMWC), i.e. V is near its own 1-ply fixed point with
        # little left to learn.
        _drop = set(_traj_rows)
        _keep = [i for i in range(len(enc_list)) if i not in _drop]
        enc_list = [enc_list[i] for i in _keep]
        t_arr = t_arr[_keep]
    return (
        np.stack(enc_list),
        t_arr,
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


# ── C collector (--c-collect) ─────────────────────────────────────────
#
# Two C collectors share the --c-collect plumbing: bgc_collect for the
# money modes (the default collector of train_batch.py's money runs) and
# bgc_collect_match for match play (an explicit opt-in).


def c_collect_tmpdir() -> str:
    """A private temp dir for the C collector's weight exports, removed at
    exit and on SIGTERM (which runs no atexit). The SIGTERM handler is
    installed only when none is set; forked children re-raise without
    touching the dir."""
    import signal
    d = tempfile.mkdtemp(prefix="bg_ccollect_")    # mode 0700
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    owner = os.getpid()
    try:
        if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
            def _term(signum, frame):
                if os.getpid() == owner:
                    shutil.rmtree(d, ignore_errors=True)
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
            signal.signal(signal.SIGTERM, _term)
    except ValueError:
        pass    # not on the main thread
    return d


class _CWeights(NamedTuple):
    """Stands in for the state_dict in the worker args on the C collector
    path: the round's weights, exported once by the master
    (export_weights.export_live) into its private dir, and the libpuretd
    backend to open them with. Match play only: bgc_collect_match's RNG
    mode (C_COLLECT_RNG_MODES) and its per-game ply bound (<= 0 unbounded,
    the serial collector; > 0 the vectorized collector's
    --max-episode-plies semantics); the money collector takes its bound
    from the worker args."""
    path: str
    backend: str
    rng_mode: int = 1
    max_plies: int = 0


# The encoders bgc_collect writes rows for: the C API encoder table's
# entries with collect set (c_api/puretd.c BGC_BASES / BGC_CUBE_BLOCKS,
# read by puretd_py.encoder_info; tests/test_c_collect.py checks the two
# agree). Listed here so the preconditions need no library.
C_COLLECT_ENCODERS = ("perspective196", "cubeful_perspective196")

# The encoders bgc_collect_match writes rows for:
# the C API encoder table's match-block entries with collect set (read by
# puretd_py.encoder_info; tests/test_c_collect_match_trainer.py checks the
# two agree).
C_COLLECT_MATCH_ENCODERS = ("matchplay1h64met_perspective196",
                            "matchplay1h64met6_perspective196")

# --c-collect-rng -> bgc_match_collect_opts.rng_mode. "serial" (0): all of a
# worker's episodes on one random.Random(seed), as _collect_worker plays them
# (the parity oracle). "vectorized" (1): one Random(base.randint(0, 2**31-1))
# per episode from base = Random(seed), as vectorized_matchplay_worker plays
# them -- the production leg's mode, so a C round plays the vectorized
# recipe's matches for the same --seed and W.
C_COLLECT_RNG_MODES = {"serial": 0, "vectorized": 1}

# c_api.puretd_py.BACKENDS, the libpuretd_<backend>.so builds (the test
# checks they agree): train_batch.py's --c-collect-backend choices without
# importing the C API binding (and bg_fast) into every training run.
C_COLLECT_BACKENDS = ("torch", "simd", "cblas", "ref")

# puretd.h BGC_KIND_MATCH: the kind bgc_open gives a
# matchplay export. The start-up probe and every worker refuse another.
_C_KIND_MATCH = 4

_ALL_MET_SCOPES = frozenset(("cube", "terminal", "boundary"))


def _is_match_mode(mode) -> bool:
    """True when `mode` plays matches (a MatchState with match_length > 0)."""
    ims = mode.initial_match_state()
    return ims is not None and ims.is_matchplay


def c_collect_rng_mode(rng) -> int:
    """bgc_match_collect_opts.rng_mode of a --c-collect-rng value
    ('serial', 'vectorized', or the int 0 / 1); ValueError otherwise."""
    if isinstance(rng, str):
        if rng in C_COLLECT_RNG_MODES:
            return C_COLLECT_RNG_MODES[rng]
    elif not isinstance(rng, bool) and rng in (0, 1):
        return int(rng)
    raise ValueError(f"c_collect_rng must be one of "
                     f"{sorted(C_COLLECT_RNG_MODES)} (got {rng!r})")


def _c_curriculum_force() -> Tuple[int, int]:
    """TRAINER_CURRICULUM_FORCE as (my, opp) aways; (0, 0) when unset --
    bgc_match_collect_opts.force_my/force_opp. Read per call, as
    _curriculum_start reads it. ValueError when it does not parse."""
    f = os.environ.get("TRAINER_CURRICULUM_FORCE")
    if not f:
        return 0, 0
    fmy, fopp = (int(x) for x in f.split(","))
    return fmy, fopp


def _c_mode_kind(mode) -> Optional[int]:
    """The puretd engine kind that plays `mode` in bgc_collect (the kind
    decides the mode: PROB = dmp, EQUITY = cubeless money, CUBEFUL =
    cubeful money), or None when no kind does."""
    from modes import CubefulMoneyMode, CubelessMoneyMode, DMPMode
    from c_api import puretd_py as P
    return {DMPMode: P.KIND_PROB, CubelessMoneyMode: P.KIND_EQUITY,
            CubefulMoneyMode: P.KIND_CUBEFUL}.get(type(mode))


def _c_net_kind(net) -> Optional[int]:
    """The kind bgc_open gives an export of `net` (c_api/puretd.c), for
    the encoders bgc_collect writes; None otherwise. bgc_open's rule on
    the encoder's facts (encoding.get_encoder): the net's width must be
    the encoder's, a cubeful encoder with an equity output is CUBEFUL, a
    cubeless one is PROB / EQUITY by output mode."""
    from c_api import puretd_py as P
    from encoding import CubefulEncoder, get_encoder
    from model import ProbNetwork
    if isinstance(net, ProbNetwork):
        return P.KIND_PROB5
    enc = getattr(net, "encoder_name", "perspective196")
    if enc not in C_COLLECT_ENCODERS:
        return None
    encoder = get_encoder(enc)
    out = getattr(net, "output_mode", "probability")
    if getattr(net, "input_size", encoder.num_features) != encoder.num_features:
        return None                     # bgc_open refuses the width mismatch
    if isinstance(encoder, CubefulEncoder):
        return P.KIND_CUBEFUL if out == "equity" else None
    return {"probability": P.KIND_PROB, "equity": P.KIND_EQUITY}.get(out)


def c_collect_fallbacks(net, mode, oneply=True, oneply_acting=False,
                        boltzmann_temp=0.0, met_path=None, met_scopes=None,
                        vectorized_collect=False,
                        gpu_eval_server=False, oracle_eval_server=False,
                        bf16_collect=False) -> List[str]:
    """Why the C collector cannot collect this configuration: one reason
    per unmet precondition, [] when it can. Needs no library.

    Match play (a match mode, or a match-play net in any mode):
    bgc_collect_match's preconditions (_c_collect_match_fallbacks); an
    explicit --c-collect with any reason is an ERROR naming all of them.
    Money modes: bgc_collect's (_c_collect_money_fallbacks);
    Trainer.train prints each as a NOTE and keeps the Python collector.
    `met_scopes` None = all three (TDAgent's default)."""
    enc = getattr(net, "encoder_name", None) or ""
    if _is_match_mode(mode) or enc.startswith("matchplay"):
        return _c_collect_match_fallbacks(
            net, mode, oneply=oneply, oneply_acting=oneply_acting,
            boltzmann_temp=boltzmann_temp, met_path=met_path,
            met_scopes=met_scopes,
            vectorized_collect=vectorized_collect,
            gpu_eval_server=gpu_eval_server,
            oracle_eval_server=oracle_eval_server)
    return _c_collect_money_fallbacks(
        net, mode, oneply_acting=oneply_acting, boltzmann_temp=boltzmann_temp,
        bf16_collect=bf16_collect)


def _c_collect_money_fallbacks(net, mode, oneply_acting=False,
                               boltzmann_temp=0.0,
                               bf16_collect=False) -> List[str]:
    """bgc_collect's unmet preconditions."""
    reasons = []
    if boltzmann_temp != 0:
        reasons.append(
            f"--boltzmann-temp {boltzmann_temp:g} (the C collector plays "
            f"argmin; torch.multinomial sampling has no C twin)")
    if oneply_acting:
        reasons.append("--oneply-acting (the C collector plays at 0-ply)")
    if bf16_collect:
        reasons.append("--bf16-collect (a Python-collector forward option)")
    enc = getattr(net, "encoder_name", "perspective196")
    out = getattr(net, "output_mode", "probability")
    mode_kind = _c_mode_kind(mode)
    if mode_kind is None:
        reasons.append(f"game mode {type(mode).__name__} has no C collector")
    elif enc not in C_COLLECT_ENCODERS:
        reasons.append(
            f"encoder {enc!r} (the C collector writes "
            f"{' / '.join(C_COLLECT_ENCODERS)} rows only)")
    elif _c_net_kind(net) != mode_kind:
        from modes import DMPMode
        if type(mode) is DMPMode and out == "equity":
            reasons.append(
                "dmp with an equity-output net (the engine kind decides the "
                "mode: an equity net collects cubeless money in C, terminal "
                "target 1/2/3 instead of dmp's 1)")
        else:
            reasons.append(
                f"{type(mode).__name__} with a {getattr(net, 'input_size', '?')}"
                f"-input {out} {enc} net (no matching C engine kind)")
    return reasons


def _c_collect_match_fallbacks(net, mode, oneply=True, oneply_acting=False,
                               boltzmann_temp=0.0, met_path=None,
                               met_scopes=None,
                               vectorized_collect=False,
                               gpu_eval_server=False,
                               oracle_eval_server=False) -> List[str]:
    """bgc_collect_match's unmet preconditions (what stays on Python). An
    explicit --c-collect with any reason is an ERROR naming all of them,
    raised before any process starts (Trainer.train, train_batch.py):
    falling back to the serial Python collector would be a 4-10x cliff
    that a long leg shows only as slow rounds.

    Served in C, so no reason: 0-ply targets (oneply False), the score
    curriculum (TRAINER_SCORE_CURRICULUM_P, TRAINER_CURRICULUM_FORCE), cube
    exploration, boundary upweight, --outcome-mix, --max-episode-plies,
    --pipeline-collect, --workers 1, --bf16-collect (no matchplay forward reads the bf16 net)."""
    from met import DEFAULT_MET
    from model import TDNetwork
    from modes import MatchplayMode
    reasons = []
    ims = mode.initial_match_state()
    is_match = (isinstance(mode, MatchplayMode) and ims is not None
                and ims.is_matchplay)
    if not is_match:
        reasons.append(f"game mode {type(mode).__name__} is not match play "
                       f"(bgc_collect_match plays matches only)")
    if vectorized_collect:
        reasons.append("--vectorized-collect (two collectors for the same "
                       "rounds: pick one)")
    if gpu_eval_server:
        reasons.append("--gpu-eval-server (routes the Python collector's "
                       "forwards; the C collector forwards in-process)")
    if oracle_eval_server:
        reasons.append("--oracle-eval-server (an --oracle-targets option)")
    if boltzmann_temp != 0:
        reasons.append(f"--boltzmann-temp {boltzmann_temp:g} (the C "
                       f"collector plays the argmin; Boltzmann sampling has "
                       f"no C twin)")
    if oneply_acting:
        reasons.append("--oneply-acting (the C collector acts at 0 ply)")
    k = int(getattr(mode, "oneply_act_k", 0) or 0)
    if k > 0:
        reasons.append(f"--td-oneply-act {k} (1-ply acting stays on Python)")
    if getattr(mode, "oracle_spec", None):
        reasons.append("--oracle-targets (oracle targets stay on Python)")
    if (getattr(mode, "oracle_act_only", False)
            or float(getattr(mode, "oracle_act_prob", 0.0) or 0.0) > 0.0):
        reasons.append("--oracle-act-only / --oracle-act-prob (teacher "
                       "acting stays on Python)")
    te = float(getattr(mode, "td_move_eps", 0.0) or 0.0)
    if te > 0.0:
        reasons.append(f"--td-move-eps {te:g} (successor rows stay on "
                       f"Python)")
    if getattr(mode, "td_eps_only", False):
        reasons.append("--td-eps-only (a --td-move-eps option)")
    mx = float(getattr(mode, "move_explore_eps", 0.0) or 0.0)
    if mx > 0.0 or _MOVE_EXPLORATION_EPS > 0.0:
        reasons.append(f"--move-explore-eps {mx:g} / "
                       f"TRAINER_MOVE_EXPLORATION_EPS "
                       f"{_MOVE_EXPLORATION_EPS:g} (checker exploration "
                       f"stays on Python)")
    met = None
    if not met_path:
        reasons.append("no --met (the net's implicit MET, memoised forwards "
                       "on the initial board, is not served in C; pass "
                       "--met auto)")
    else:
        scopes = (frozenset(met_scopes) if met_scopes is not None
                  else _ALL_MET_SCOPES)
        if scopes != _ALL_MET_SCOPES:
            reasons.append(f"--met-scope {','.join(sorted(scopes))} (the C "
                           f"collector grounds all three scopes; the others "
                           f"would be the net's implicit MET)")
        env_met = os.environ.get("BG_MET_PATH")
        enc_met = env_met or DEFAULT_MET
        if os.path.realpath(enc_met) != os.path.realpath(met_path):
            src = (f"BG_MET_PATH={env_met}" if env_met
                   else f"the default {DEFAULT_MET}")
            reasons.append(f"the encoder MET ({src}) is not the grounding "
                           f"MET (--met {met_path}); the C engine takes one "
                           f"table for both")
        try:
            met = _worker_met(met_path)
        except Exception as e:  # noqa: BLE001 - reported, not raised
            reasons.append(f"--met {met_path} cannot be read ({e})")
    if is_match and met is not None and ims.match_length > met.length:
        reasons.append(f"--match-length {ims.match_length} is above the "
                       f"{met.length}-point MET {met_path}")
    if os.environ.get("TRAINER_CURRICULUM_FORCE"):
        try:
            fmy, fopp = _c_curriculum_force()
            ok = fmy >= 1 and fopp >= 1
        except ValueError:
            ok = False
        if not ok:
            reasons.append(
                f"TRAINER_CURRICULUM_FORCE="
                f"{os.environ['TRAINER_CURRICULUM_FORCE']!r} (the C "
                f"collector takes two away scores >= 1, 'my,opp')")
    enc = getattr(net, "encoder_name", None)
    if enc not in C_COLLECT_MATCH_ENCODERS:
        reasons.append(f"encoder {enc!r} (the C collector writes "
                       f"{' / '.join(C_COLLECT_MATCH_ENCODERS)} rows only)")
    else:
        from encoding import get_encoder
        n_feat = get_encoder(enc).num_features
        if getattr(net, "input_size", None) != n_feat:
            reasons.append(f"a {getattr(net, 'input_size', '?')}-input net "
                           f"with the {n_feat}-column encoder {enc}")
    if getattr(net, "output_mode", "probability") != "probability":
        reasons.append(f"output mode {net.output_mode!r} (match play nets "
                       f"output P(win match), 'probability')")
    if not (isinstance(net, TDNetwork)
            and type(net).forward is TDNetwork.forward):
        reasons.append(f"a {type(net).__name__} (the C engine runs the plain "
                       f"TDNetwork MLP of the BGNN export)")
    return reasons


def c_collect_error(reasons: Sequence[str]) -> str:
    """The error an explicit match-play --c-collect raises for
    c_collect_fallbacks()."""
    n = len(reasons)
    return (f"--c-collect cannot collect this configuration in C ({n} "
            f"reason{'' if n == 1 else 's'}): " + "; ".join(reasons)
            + ". Drop --c-collect to collect with the Python collector.")


def c_collect_library(backend: str) -> Optional[str]:
    """None when libpuretd_<backend>.so loads -- built on demand when it is
    missing or its ABI is stale (puretd_py.ready_lib; one NOTE says so) --
    and the binding has the matchplay collector (Engine.set_met /
    Engine.collect_match; this tree's binding always
    does, the money collector needs only the library); else why not, for
    the caller's NOTE (the default money collector falls back) or error
    (an explicit --c-collect)."""
    from c_api import puretd_py as P
    if backend not in P.BACKENDS:
        return (f"unknown backend {backend!r} (one of "
                f"{', '.join(P.BACKENDS)})")
    why = P.ready_lib(backend)
    if why:
        return why
    missing = [m for m in ("set_met", "collect_match")
               if not hasattr(P.Engine, m)]
    if missing:
        return ("c_api/puretd_py.py has no "
                + " / ".join(f"Engine.{m}" for m in missing)
                + ": the C matchplay collector is not in this tree")
    return None


def _c_collect(weights: _CWeights, mode, num_episodes, seed, oneply,
               max_episode_plies):
    """The money _collect_worker episode loop in C: `num_episodes`
    episodes on one random.Random(seed) stream through bgc_collect. Same
    (enc, tgt, n_discarded) result; an empty one has shape
    (0, input_size). The engine is opened per call: bgc_open has no
    weight reload and the open costs < 0.5 ms."""
    from c_api.puretd_py import Engine
    eng = Engine(weights.path, weights.backend)
    try:
        if eng.kind != _c_mode_kind(mode):
            raise RuntimeError(
                f"C collector: engine kind {eng.kind} of {weights.path} does "
                f"not play {type(mode).__name__}")
        return eng.collect(None, seed, num_episodes, oneply=oneply,
                           jacoby=getattr(mode, "jacoby", True),
                           max_plies=max_episode_plies)
    finally:
        eng.close()


def _c_match_opts(mode, oneply, rng_mode, max_plies) -> dict:
    """bgc_match_collect_opts for one collection call,
    from THIS process's trainer constants -- a spawn worker reads the same
    env as the master, and they are what the serial collector would read
    here -- and the pickled mode. A boundary upweight below 1 duplicates
    nothing, as 1 (range(w - 1) is empty)."""
    fmy, fopp = _c_curriculum_force()
    return {
        "match_length": int(mode.initial_match_state().match_length),
        "oneply": int(bool(oneply)),
        "curriculum_p": float(_SCORE_CURRICULUM_P),
        "force_my": int(fmy),
        "force_opp": int(fopp),
        "cube_explore_eps": float(_CUBE_EXPLORATION_EPS),
        "boundary_upweight": max(1, int(_BOUNDARY_UPWEIGHT)),
        "outcome_mix": float(getattr(mode, "outcome_mix", 0.0) or 0.0),
        "max_plies": int(max_plies),
        "rng_mode": int(rng_mode),
    }


def _c_open_match_engine(path, backend, met_path):
    """Engine(path, backend), refused unless it is a matchplay engine, with
    the grounding MET set (_worker_met(met_path): the table the Python
    worker would ground on; the encoder MET is the same file, checked by
    c_collect_fallbacks)."""
    from c_api import puretd_py as P
    eng = P.Engine(path, backend)
    try:
        kind = getattr(P, "KIND_MATCH", _C_KIND_MATCH)
        if eng.kind != kind:
            raise RuntimeError(
                f"C collector: {path} opens as engine kind {eng.kind} "
                f"({eng.encoder}, {eng.input_size} inputs), not the "
                f"matchplay kind {kind}")
        eng.set_met(_worker_met(met_path))
    except BaseException:
        eng.close()
        raise
    return eng


def _c_collect_match(weights: _CWeights, mode, num_episodes, seed, oneply,
                     met_path):
    """_collect_worker's episode loop in C: `num_episodes` matches through
    bgc_collect_match on `seed`'s streams (weights.rng_mode). Returns
    (enc, tgt, n_discarded), rows in episode order; an empty result has
    shape (0, input_size) / (0,), which the master skips like the Python
    worker's (0, 0). The engine is opened per call: bgc_open has no weight
    reload, the weights change every round, and an open costs < 1 ms.

    BG_C_COLLECT_TIMING=1 (benchmarks): one line per call with this
    process's wall time for it (open, collect, close) and its rates."""
    from c_api import puretd_py as P
    t0 = time.perf_counter()
    eng = _c_open_match_engine(weights.path, weights.backend, met_path)
    try:
        opts = _c_match_opts(mode, oneply, weights.rng_mode,
                             weights.max_plies)
        # The binding's ctypes struct when it has the helper
        # (puretd_py.match_collect_opts, same field names), else the dict.
        _mk = getattr(P, "match_collect_opts", None)
        if _mk is not None:
            opts = _mk(**opts)
        enc, tgt, n_disc = eng.collect_match(opts, int(seed),
                                             int(num_episodes))
    finally:
        eng.close()
    if os.environ.get("BG_C_COLLECT_TIMING") == "1":
        dt = max(time.perf_counter() - t0, 1e-9)
        print(f"C collect timing: pid {os.getpid()} episodes {num_episodes} "
              f"rows {len(enc)} s {dt:.3f} ep/s {num_episodes / dt:.3f} "
              f"rows/s {len(enc) / dt:.0f}", flush=True)
    return enc, tgt, int(n_disc)


def _dump_round(dump_dir, round_idx, enc, tgt, worker_rows, worker_seeds,
                n_discarded, collector):
    """BG_COLLECT_DUMP: one round's pool as `round_NNNN.npz` (1-based) with
    the rows of each worker result, the master's seed draws, the discard
    count and which collector ran (c / vectorized / serial / single)."""
    np.savez(os.path.join(dump_dir, f"round_{round_idx + 1:04d}.npz"),
             enc=enc, tgt=tgt,
             worker_rows=np.asarray(worker_rows, dtype=np.int64),
             worker_seeds=np.asarray(worker_seeds, dtype=np.int64),
             n_discarded=np.int64(n_discarded),
             collector=np.str_(collector))


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
    `num_episodes` episodes, return the concatenated (encodings, targets)
    numpy arrays -- for the money modes also n_discarded (episodes over
    `max_episode_plies`), (enc, tgt, n_discarded); the match collector
    never discards. The GameMode instance is pickled into `args` (modes
    carry config, e.g. CubefulMoneyMode.jacoby).

    A `_CWeights` in the state_dict slot (--c-collect) plays the episodes
    in C instead (`_c_collect_match` for match play, `_c_collect` for the
    money modes) and returns (encodings, targets, n_discarded).

    The args tuple must stay in sync with `_build_worker_args` in
    Trainer.train: 14 fields, then `activation` and `max_episode_plies`,
    both defaulted so a shorter tuple still unpacks. `activation` must be
    carried: it is not a parameter tensor, so load_state_dict cannot catch
    a mismatch, and with --gpu-eval-server the worker's local net must be
    the same function as the server's.

    Set env LEAK_DIAG=1 to log RSS at entry/exit (for leak hunts).
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    (state_dict, hidden_sizes, input_size, output_mode, encoder_name,
     mode, num_episodes, seed, oneply, oneply_acting, boltzmann_temp,
     bf16_inference, met_path, met_scopes) = args[:14]
    activation = args[14] if len(args) > 14 else "relu"
    max_episode_plies = (args[15] if len(args) > 15
                         else DEFAULT_MAX_EPISODE_PLIES)
    is_match = _is_match_mode(mode)

    if isinstance(state_dict, _CWeights):
        if is_match:
            return _c_collect_match(state_dict, mode, num_episodes, seed,
                                    oneply, met_path)
        return _c_collect(state_dict, mode, num_episodes, seed, oneply,
                          max_episode_plies)

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
    # Expert iteration: when the mode asks for it, the COLLECTOR acts with
    # 1-ply move selection (top-K filtered) instead of the raw 0-ply read.
    # 1-ply selection measures +2.00pp over 0-ply on identical weights
    # (12k matches, z=+4.38), so this generates trajectories from the
    # stronger policy while the targets stay what they were -- the network
    # then learns to reproduce choices it currently cannot make at 0 ply.
    # K=3 was measured to match FULL 1-ply selection (101/128 on the
    # gnubg-confirmed set) at 5.2x less compute.
    _oa = int(getattr(mode, "oneply_act_k", 0) or 0)
    agent = TDAgent(
        net, oneply=oneply_acting, boltzmann_temp=boltzmann_temp,
        bf16_inference=bf16_inference, met=_worker_met(met_path),
        met_scopes=met_scopes,
        oneply_moves=(_oa > 0), oneply_moves_k=_oa,
    )
    # GPU eval server: route this agent's matchplay forwards through the
    # slot claimed in `_worker_init`. No-op unless the server is active.
    # The agent's own device stays 'cpu' on purpose -- only the forward
    # moves. Setting agent.device='cuda' would disable the C base-encode
    # fast path (`_c_base_available` requires a CPU device).
    if _WORKER_REMOTE_EVAL is not None and _WORKER_REMOTE_KIND == "student":
        agent._remote_eval = _WORKER_REMOTE_EVAL
        agent._remote_eval_moves = _WORKER_ROUTE_MOVES

    rng = random.Random(seed)

    enc_chunks: List[np.ndarray] = []
    tgt_chunks: List[np.ndarray] = []
    n_discarded = 0
    for _ in range(num_episodes):
        encs, tgts = collect_episode(
            agent, mode, rng, oneply=oneply,
            max_episode_plies=max_episode_plies,
        )
        if len(encs) > 0:
            enc_chunks.append(encs)
            tgt_chunks.append(tgts)
        else:
            # Empty means discarded (a real episode has >= 1 sample).
            n_discarded += 1
    if not enc_chunks:
        result = (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    else:
        result = (
            np.concatenate(enc_chunks, axis=0),
            np.concatenate(tgt_chunks, axis=0),
        )
    if not is_match:
        result = result + (n_discarded,)

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


class Trainer:
    """Owns the optimizer and training loop. Agents play and collect
    without one.
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
            agent: must expose `agent.network`.
            lr: learning rate.
            optimizer_cls: default Adam; SGD for online TD(0).
            optimizer_kwargs: forwarded to the optimizer.
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
        """One optimizer step on a batch of BoardStates (agent encodes
        them) or a pre-encoded numpy array. Returns the scalar loss.
        """
        return self.train_step_nosync(states_or_encoded, targets).item()

    def train_step_nosync(
        self,
        states_or_encoded: Union[Sequence[BoardState], np.ndarray],
        targets: Sequence,
    ) -> torch.Tensor:
        """train_step returning the detached loss tensor: no `.item()`
        host sync per batch. The batch loop reads its losses back once
        per round (same values)."""
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
        return loss.detach()

    def train_step_dev(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """train_step on tensors already on the agent's device; returns
        the detached loss tensor (no host sync). See BG_POOL_ON_DEVICE."""
        pred = self.agent.network(x)
        loss = self.agent.loss(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.agent.network.parameters(), self.grad_clip,
            )
        self.optimizer.step()
        memo = getattr(self.agent, "_new_game_memo", None)
        if memo:
            memo.clear()
        return loss.detach()

    def eval_loss(
        self,
        states_or_encoded: Union[Sequence[BoardState], np.ndarray],
        targets: Sequence,
    ) -> float:
        """Loss on a batch without stepping (held-out validation).
        Same inputs as train_step.
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
        pipeline_collect: bool = False,
        max_episode_plies: Optional[int] = None,
        vectorized_collect: bool = False,
        vectorized_batch: int = 32,
        train_params: Optional[dict] = None,
        c_collect: bool = False,
        c_collect_backend: str = "simd",
        c_collect_auto: bool = False,
        met_path: Optional[str] = None,
        met_scopes=None,
        vectorized_device: str = "cuda",
        gpu_eval_server: bool = False,
        oracle_eval_server: bool = False,
        gpu_eval_device: str = "cuda",
        gpu_eval_route_moves: bool = False,
        gpu_eval_deterministic: bool = False,
        episode_offset: int = 0,
        c_collect_rng: str = "vectorized",
    ) -> List[float]:
        """Round-based batch TD(0). A round collects
        `episodes_per_round` episodes, then trains `epochs_per_round`
        shuffled passes over the round's pool in `batch_size` batches.

        `log_every` is in rounds. `workers > 1` collects in N
        subprocesses. The training step runs in the master.
        `oneply=True` uses 1-ply lookahead targets.

        `end_lr`: linearly anneal the LR from its starting value to
        `end_lr` over `num_episodes`. With `end_lr` None and
        `warmup_cycles` 0 the LR is never touched.

        `warmup_cycles`: ramp the LR from `lr * 0.1` to the scheduled
        LR over the first N rounds. Multiplies with `end_lr`.

        `pipeline_collect`: overlap collection of round N+1 with the
        training step of round N. Workers then see one-round-stale
        weights. Worker seeds are unchanged. No effect with
        workers <= 1.

        `max_episode_plies`: None = the mode's default. Money modes
        (DEFAULT_MAX_EPISODE_PLIES): an episode exceeding it is discarded
        and counted in the round log line. Match play (0 = unbounded):
        a per-GAME bound of the vectorized and C collectors only, a match
        exceeding it is discarded whole; the serial Python match
        collector ignores it (and says so when --vectorized-collect falls
        back to it).

        `train_params`: provenance dict stamped into periodic
        checkpoints (model.save). The final save is the caller's
        responsibility. `train_params["c_collect"]`, when present, is set
        to whether the C collector actually collects.

        `c_collect`: collect in C on the `c_collect_backend` libpuretd
        build. The master exports each round's weights once to a private
        dir (c_collect_tmpdir), the worker args carry a `_CWeights` in the
        state_dict slot, and the master's seed draws are the Python
        path's, so a C round consumes the run RNG as a Python round does.
        With workers <= 1 the master collects one worker's share on a
        seed drawn from the run RNG (the Python single-process path plays
        on the run RNG itself, so the episodes differ). Works with
        `pipeline_collect`, `save_every` and the pool restart.
          Money modes: bgc_collect, the episodes the Python collector
        plays for the seed. Each
        c_collect_fallbacks() reason prints a NOTE and keeps the Python
        collector; a missing library is built on demand
        (c_collect_library). `c_collect_auto`: True when `c_collect` is
        train_batch.py's default rather than an explicit --c-collect; a
        library that cannot be built or loaded then falls back with a
        NOTE instead of raising. `c_collect` with `vectorized_collect` is
        an error.
          Match play: bgc_collect_match, an explicit opt-in.
        ERRORS, before any process starts, naming every reason of
        c_collect_fallbacks(), and when the library cannot be loaded or
        built or the start-up probe (export, open, kind and encoder check,
        set_met) fails. `c_collect_rng`: 'vectorized' (default; the
        production leg, per-episode streams as
        vectorized_matchplay_worker) or 'serial' (one stream per worker as
        _collect_worker: the parity oracle).

        `vectorized_collect`: match play + `oneply` only, opt-in. Replaces
        the per-worker serial collector with the lockstep generator
        driver in `vectorized_matchplay.py`: each worker runs
        `vectorized_batch` matches concurrently and batches their pending
        forwards into one GPU call per step on `vectorized_device`.
        Inactive (with a printed WARN naming every reason) unless every
        precondition holds (see the block below). Disables
        `pipeline_collect`. The collected data is equivalent to the
        serial path (tests/test_vectorized_matchplay.py).

        `gpu_eval_server`: match play + `oneply`, opt-in; the OTHER way to
        put 1-ply collection on the GPU: ONE server process
        (gpu_eval_server.py) holds the only CUDA context and N CPU workers
        ship feature matrices to it over per-worker shared memory. Routes
        `TDAgent.value_oneply_checker_matchplay`; `gpu_eval_route_moves`
        also routes `choose_checker_action_matchplay`. HARD errors with
        `pipeline_collect`, `vectorized_collect` and
        TRAINER_MAXTASKSPERCHILD; WARN + the serial collector on a soft
        precondition. `gpu_eval_deterministic`: one request per GPU call
        (bit-exact against the unrouted collector) instead of fusing the
        concurrently pending ones (not bitwise reproducible).
        `oracle_eval_server`: the same server hosting the --oracle-targets
        teacher (frozen) instead of the student.

        `met_path`, `met_scopes`: match play only (collect_episode,
        TDAgent).

        `episode_offset`: episodes already done before this run (a
        `--resume-episodes` crash recovery). Affects checkpoint NAMES
        and the SIGTERM save's episode count only, so `_ep{N}` stays
        the global count; the LR schedule is the caller's business.

        SIGTERM/SIGINT save `{root}_ep{N}_sigterm.pt` / `_sigint.pt`
        (weights after exactly N episodes, a round boundary) and reap
        the pool before exiting -- see the handler below.

        A dead collection worker (pool_guard.PoolWatch) rebuilds the pool
        and redoes the round, at most BG_MAX_POOL_RESTARTS times; the GPU
        eval server path fails the round instead (EvalServer.map_watched).

        `BG_TRAIN_CUDA_GRAPH` (cuda_graph_step.py; unset = on on a cuda
        device, 0 = the eager step) replays full batches of the train step
        from a CUDA graph; `BG_POOL_ON_DEVICE=1` keeps the round's pool on
        the cuda device with the eager step (implied by the graph). Both
        give checkpoints and losses bitwise equal to the host-pool step
        (tests/test_train_step_bitwise.py).

        `BG_COLLECT_DUMP=<dir>` (a test hook, any collector): save each
        round's pool as `round_NNNN.npz` -- enc, tgt, worker_rows (rows per
        worker result, in worker order), worker_seeds (the master's seed
        draws; empty for the single-process Python path, which plays on
        the run RNG), n_discarded and collector. `BG_ROUND_PHASES=1`: one
        line per round with the master's time by phase.

        Returns a flat list of per-batch losses.
        """
        mode.validate_agent(self.agent)
        is_match = _is_match_mode(mode)
        if max_episode_plies is None:
            max_episode_plies = 0 if is_match else DEFAULT_MAX_EPISODE_PLIES
        if c_collect and vectorized_collect and not is_match:
            # Match play names it among the C collector's reasons below.
            raise ValueError(
                "--c-collect and --vectorized-collect are mutually exclusive "
                "(two collectors for the same rounds). Drop one of the two "
                "flags."
            )

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

        # ── C collector (--c-collect): preconditions, library, probe ──
        # All before any process starts: the preconditions, the library
        # (built on demand), then one probe open of this net's export (a
        # kind / encoder / width / MET failure surfaces here, not inside
        # pool.map).
        c_active = False
        c_dir = None            # private 0700 dir of the per-round exports
        c_files = {}            # round index -> exported weights file
        c_rng_mode = None
        c_label = "--c-collect (default)" if c_collect_auto else "--c-collect"
        if c_collect and is_match:
            # An explicit match-play --c-collect never falls back: every
            # unmet precondition is named in ONE error (the fallback would
            # be the serial collector, a 4-10x cliff).
            c_rng_mode = c_collect_rng_mode(c_collect_rng)
            _c_reasons = c_collect_fallbacks(
                self.agent.network, mode, oneply=oneply,
                oneply_acting=oneply_acting, boltzmann_temp=boltzmann_temp,
                met_path=met_path, met_scopes=met_scopes,
                vectorized_collect=vectorized_collect,
                gpu_eval_server=gpu_eval_server,
                oracle_eval_server=oracle_eval_server)
            if _c_reasons:
                raise ValueError(c_collect_error(_c_reasons))
            _why = c_collect_library(c_collect_backend)
            if _why is not None:
                raise RuntimeError(f"--c-collect: {_why}")
            import export_weights
            from c_api import puretd_py as _P
            c_dir = c_collect_tmpdir()
            try:
                _probe = os.path.join(c_dir, "w_probe.bin")
                export_weights.export_live(self.agent.network, _probe)
                _eng = _c_open_match_engine(_probe, c_collect_backend, met_path)
                try:
                    _c_desc = _eng.describe
                    _c_enc, _c_in = _eng.encoder, _eng.input_size
                finally:
                    _eng.close()
                os.remove(_probe)
                _n_net = getattr(self.agent.network, "input_size", None)
                _net_enc = getattr(self.agent.network, "encoder_name", None)
                if _c_enc != _net_enc or _c_in != _n_net:
                    raise RuntimeError(
                        f"--c-collect: the export opens as encoder {_c_enc} "
                        f"with {_c_in} inputs; the net is a {_n_net}-input "
                        f"{_net_enc}")
                _info = _P.encoder_info(_c_enc, c_collect_backend)
                if not (_info and _info.get("collect")):
                    raise RuntimeError(
                        f"--c-collect: the C API encoder table has no "
                        f"collector for {_c_enc} (encoder_info: {_info}); "
                        f"bgc_collect_match would return "
                        f"BGC_COLLECT_UNSUPPORTED")
            except BaseException:
                shutil.rmtree(c_dir, ignore_errors=True)
                raise
            c_active = True
            print(f"C collection: ON (bgc_collect_match on {_c_desc}, "
                  f"W={max(workers, 1)}, rng={c_collect_rng})", flush=True)
            print(f"  C collection: {_c_enc}, "
                  f"{'1-ply' if oneply else '0-ply'} targets, 0-ply play, "
                  f"max_episode_plies={max_episode_plies}, MET {met_path}; "
                  f"weights exported once per round to {c_dir}", flush=True)
            if workers <= 1:
                print("  NOTE: --c-collect with --workers 1: the master "
                      "collects each round in C on one seed drawn from the "
                      "run RNG, as a one-worker pool would (the Python "
                      "single-process path plays on the run RNG itself, so "
                      "the episodes differ).", flush=True)
            elif pipeline_collect:
                print("  NOTE: --pipeline-collect with --c-collect: round "
                      "N+1 is collected on the weights from before round N's "
                      "train step (one-round-stale targets) -- a recipe "
                      "change, not the gate/leg setting).", flush=True)
            if bf16_collect:
                print("  NOTE: --bf16-collect has no effect here (no "
                      "matchplay forward reads the bf16 net; the C forward "
                      "is fp32).", flush=True)
        elif c_collect:
            # Money modes: a reason is a NOTE and the Python collector.
            reasons = c_collect_fallbacks(
                self.agent.network, mode, oneply_acting=oneply_acting,
                boltzmann_temp=boltzmann_temp, bf16_collect=bf16_collect)
            for why in reasons:
                print(f"  NOTE: {c_label}: {why}; falling back to the "
                      f"Python collector.", flush=True)
            c_active = not reasons
            if c_active:
                why = c_collect_library(c_collect_backend)
                if why is not None:
                    if not c_collect_auto:
                        raise RuntimeError(f"--c-collect: {why}")
                    print(f"  NOTE: {c_label}: {why}; falling back to the "
                          f"Python collector.", flush=True)
                    c_active = False
            if c_active:
                import export_weights
                from c_api.puretd_py import Engine
                c_dir = c_collect_tmpdir()
                try:
                    probe = os.path.join(c_dir, "w_probe.bin")
                    export_weights.export_live(self.agent.network, probe)
                    eng = Engine(probe, c_collect_backend)
                    kind, n_in, desc = eng.kind, eng.input_size, eng.describe
                    c_enc = eng.encoder
                    eng.close()
                    os.remove(probe)
                    n_net = getattr(self.agent.network, "input_size", 196)
                    net_enc = getattr(self.agent.network, "encoder_name",
                                      "perspective196")
                    if (kind != _c_mode_kind(mode) or n_in != n_net
                            or c_enc != net_enc):
                        raise RuntimeError(
                            f"C collector: the export opens as kind {kind} "
                            f"with {n_in} inputs, encoder {c_enc}; "
                            f"{type(mode).__name__} with this {n_net}-input "
                            f"{net_enc} net needs kind {_c_mode_kind(mode)}")
                except BaseException:
                    shutil.rmtree(c_dir, ignore_errors=True)
                    raise
                print(f"C collection: ON ("
                      + ("the default, --no-c-collect for the Python "
                         "collector; " if c_collect_auto else "")
                      + f"bgc_collect on {desc}, "
                      f"{max(workers, 1)} process(es), "
                      f"{'1-ply' if oneply else '0-ply'} targets, 0-ply play, "
                      f"max_episode_plies={max_episode_plies}; weights "
                      f"exported once per round to {c_dir})", flush=True)
        if metrics_out is not None:
            metrics_out["c_collect"] = c_active
            if c_dir is not None:
                metrics_out["c_collect_dir"] = c_dir
        if train_params is not None and "c_collect" in train_params:
            # Provenance: what collected, not what was asked for.
            train_params["c_collect"] = c_active

        # BG_COLLECT_DUMP=<dir> (test hook): each round's pool, per-worker
        # row counts and seeds (round_NNNN.npz; see the docstring).
        _dump_dir = os.environ.get("BG_COLLECT_DUMP") or None
        if _dump_dir:
            os.makedirs(_dump_dir, exist_ok=True)

        # ── Vectorized (lockstep GPU) collection preconditions ────────
        # Opt-in only; every failed precondition is named rather than
        # silently ignored, because falling back to the CPU path with a
        # flag set looks identical in the logs otherwise.
        vec_active = False
        if vectorized_collect:
            import vectorized_matchplay as _vm
            reasons = []
            if not is_match:
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
            if (float(getattr(mode, "move_explore_eps", 0.0) or 0.0) > 0.0
                    or _MOVE_EXPLORATION_EPS > 0.0):
                # Also catches the TRAINER_MOVE_EXPLORATION_EPS env form:
                # the collector falls back to it when the mode field is 0
                # (see the acting block), so checking only the field let
                # the vectorized path drop exploration silently AND desync
                # the RNG stream from the serial mirror.
                reasons.append(
                    "--move-explore-eps / TRAINER_MOVE_EXPLORATION_EPS is "
                    "implemented in the standard collector only (the "
                    "vectorized path would silently ignore it)"
                )
            if int(getattr(mode, "oneply_act_k", 0) or 0) > 0:
                reasons.append(
                    "--td-oneply-act acts with 1-ply move selection, which "
                    "the generator mirror does not implement; the vectorized "
                    "path would silently act 0-ply instead"
                )
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
                if is_match and max_episode_plies > 0:
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
                # MPS state is run provenance: it moves throughput by
                # up to 1.56x without moving the numbers, so a log that
                # does not record it cannot explain its own ep/s.
                if _vm.mps_active():
                    print(
                        f"  CUDA MPS: ON (pipe dir {_vm.mps_pipe_dir()}) "
                        f"-- the {workers} driver contexts share one GPU "
                        f"scheduler."
                        + ("  Worth ~0% at 1-ply (each trip already "
                           "carries ~350-400 rows); harmless."
                           if oneply else "")
                    )
                elif not oneply and workers >= _vm.MPS_MIN_WORKERS:
                    print(
                        f"  NOTE: CUDA MPS is OFF and this is the regime "
                        f"where it pays: 0-ply trips are small (~8 rows "
                        f"per slot-yield), so {workers} driver contexts "
                        f"time-slice instead of overlapping. Measured "
                        f"230 -> 356 ep/s (+56%) at W=32 on a 5090; "
                        f"+8% at W=16, so the gain scales with workers. "
                        f"Throughput only -- targets are unchanged. "
                        f"Start it with ./mps.sh start (HOST-WIDE "
                        f"daemon: not on a shared box)."
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

        # The train-step env vars, read (and validated) before any process
        # starts (a bad BG_TRAIN_CUDA_GRAPH value fails here).
        graph_env = mode_from_env()
        pool_dev_env = os.environ.get("BG_POOL_ON_DEVICE") == "1"

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

        # One worker pool for the whole run; None means single-process.
        # "spawn", not "fork": forked workers inherit the master's RSS
        # (copy-on-write at fork time, so a replacement worker spawned
        # mid-training starts at the master's multi-GB peak).
        # TRAINER_MAXTASKSPERCHILD=N recycles workers after N calls.
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
            reasons = []
            if not is_match:
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

        def _new_pool():
            """The run's collection pool (pool_guard.GuardedPool: worker
            exit codes for PoolWatch's message, faulthandler in every
            worker), or None for single-process collection."""
            if workers <= 1:
                return None
            if eval_srv is not None:
                # maxtasksperchild is FORCED None here: slots are claimed
                # once per process and never returned. The env var was
                # already rejected above, so this is belt-and-braces.
                return GuardedPool(
                    processes=workers, initializer=_worker_init,
                    initargs=(eval_srv.client_handles(), _srv_kind),
                    maxtasksperchild=None, context=_ctx,
                )
            return GuardedPool(processes=workers, maxtasksperchild=maxtasks,
                               context=_ctx)

        pool = _new_pool()
        # Dead-worker detection and recovery (pool_guard). `pool` and
        # `watch` are rebound on a restart; the closures below read them
        # at call time, so they always see the live ones.
        watch = PoolWatch(pool) if pool is not None else None
        pool_restarts = 0
        restart_limit = max_pool_restarts()

        # SIGTERM/SIGINT: the `finally` below cleans up correctly, but a
        # signal never reaches it -- the default disposition kills this
        # process outright, so `finally` and atexit are both skipped. The
        # eval server is a separate spawn()ed process, so it survives,
        # reparents to init and keeps its VRAM, pinned host memory and
        # /dev/shm slots. Without a server the same default disposition
        # orphans the spawn-pool workers and loses everything since the
        # last --save-every checkpoint.
        #
        # So on SIGTERM/SIGINT (installed on every run, server or not):
        #   1. If the master is between rounds or collecting, the weights
        #      are exactly "trained on `_sig['done']` episodes" -- save
        #      them NOW as `{root}_ep{N}_sig{term,int}.pt` (never
        #      clobbering; a `.1`, `.2` ... suffix is added if taken).
        #      If it is mid-train-step, the signal is DEFERRED to the end
        #      of that round (seconds), so N is always a round boundary
        #      and never a half-trained round. A second signal while
        #      deferred exits at once WITHOUT saving (escape hatch).
        #   2. Tear down: pool first, or the workers block forever on a
        #      dead server's semaphores; then the server; then the C
        #      export dir; then os._exit so the `finally` cannot re-join
        #      an already-terminated pool.
        # Only this (main) process ever saves; workers never see this
        # handler (spawn children start with default dispositions).
        _prev_sig_handlers = {}
        _sig = {"busy": False, "pending": None, "done": 0, "lr": None}
        _sched_start = self.optimizer.param_groups[0]["lr"]

        def _sched_lr(n_eps):
            # The un-warmed-up schedule value at `n_eps` episodes into
            # THIS run: what a resume from that point should start at.
            if end_lr is None:
                return _sched_start
            return _sched_start + (end_lr - _sched_start) * (
                n_eps / max(num_episodes, 1))

        def _teardown_and_exit(signum, save):
            if save:
                n = _sig["done"]
                n_glob = episode_offset + n
                lr_next = _sched_lr(n)
                if save_path:
                    root, ext = os.path.splitext(save_path)
                    tag = "sigint" if signum == _signal.SIGINT else "sigterm"
                    base = f"{root}_ep{n_glob}_{tag}"
                    path, k = f"{base}{ext or '.pt'}", 0
                    while os.path.exists(path):
                        k += 1
                        path = f"{base}.{k}{ext or '.pt'}"
                    try:
                        self.agent.network.save(path, train_params={
                            "signal": tag,
                            "episodes_done": n_glob,
                            "episodes_done_this_run": n,
                            "optimizer_lr": _sig["lr"],
                            "lr_at_episodes_done": lr_next,
                        })
                        print(
                            f"  -> signal {signum}: checkpoint saved: {path} "
                            f"| episodes done {n_glob} "
                            f"(this run {n}/{num_episodes}) "
                            f"| optimizer lr {_sig['lr']} "
                            f"| resume lr (schedule at this-run ep {n}) "
                            f"{lr_next:.6g}",
                            flush=True)
                    except Exception as e:  # never skip the teardown
                        print(f"  -> signal {signum}: SAVE FAILED: {e!r}",
                              flush=True)
                else:
                    print(f"  -> signal {signum}: no --save path, nothing "
                          f"saved (episodes done {n_glob})", flush=True)
            try:
                shutdown_pool(pool)
            except Exception:
                pass
            try:
                if eval_srv is not None:
                    eval_srv.stop()
            except Exception:
                pass
            # os._exit skips `finally` and atexit: remove the C exports here.
            if c_dir is not None:
                shutil.rmtree(c_dir, ignore_errors=True)
            sys.stdout.flush()
            os._exit(128 + signum)

        def _shutdown_on_signal(signum, _frame):
            if _sig["pending"] is not None:
                print(f"  -> signal {signum} again: exiting WITHOUT save",
                      flush=True)
                _teardown_and_exit(signum, save=False)
            if _sig["busy"]:
                _sig["pending"] = signum
                print(f"  -> signal {signum}: deferring save to end of "
                      f"this round's train step", flush=True)
                return
            _teardown_and_exit(signum, save=True)

        import signal as _signal
        for _s in (_signal.SIGTERM, _signal.SIGINT):
            try:
                _prev_sig_handlers[_s] = _signal.signal(
                    _s, _shutdown_on_signal)
            except (ValueError, OSError):
                # Not the main thread: the `finally` is the only path.
                pass

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

        def _c_export(round_idx: int) -> str:
            """Round `round_idx`'s weights as BGNN, exported once for all
            its workers: written as .tmp and renamed, so no worker ever
            opens a partial file, and a retried round (pool restart)
            replaces its file atomically."""
            path = os.path.join(c_dir, f"w_{round_idx + 1:05d}.bin")
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

        def _c_weights(round_idx: int) -> _CWeights:
            return _CWeights(
                _c_export(round_idx), c_collect_backend,
                c_rng_mode if c_rng_mode is not None else 1,
                max_episode_plies if is_match else 0)

        # Worker seeds by round index, for BG_COLLECT_DUMP (popped when
        # the round is collected).
        round_seeds = {}

        # Worker args for one round, with the current weights. Arity must
        # stay in sync with `_collect_worker`.
        def _build_worker_args(round_size: int, round_idx: int):
            splits = _split_episodes(round_size, workers)
            if c_active:
                state_dict = _c_weights(round_idx)
            else:
                # .clone() is REQUIRED: on a CPU net `.cpu()` returns a
                # view of the live parameter, which map_async pickles
                # WHILE optimizer.step() mutates it (torn weights).
                state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in net.state_dict().items()
                }
            seeds = [rng.randint(0, 2**31 - 1) for _ in splits]
            round_seeds[round_idx] = seeds
            return [
                (state_dict, hidden_sizes, input_size, output_mode,
                 encoder_name,
                 mode, n_eps, seed,
                 oneply, oneply_acting, boltzmann_temp,
                 bf16_collect, met_path,
                 met_scopes, activation, max_episode_plies)
                for n_eps, seed in zip(splits, seeds)
            ]

        # Twin of _build_worker_args for the vectorized match path (arity:
        # `VecArgs` in vectorized_matchplay.py). Keeps the same leading 13
        # fields and consumes rng.randint identically, so worker seeding is
        # unchanged; appends (B, device, max_episode_plies, met_scopes,
        # activation).
        def _build_vec_args(round_size: int, round_idx: int):
            splits = _split_episodes(round_size, workers)
            state_dict = {
                k: v.detach().cpu().clone()
                for k, v in net.state_dict().items()
            }
            seeds = [rng.randint(0, 2**31 - 1) for _ in splits]
            round_seeds[round_idx] = seeds
            return [
                (state_dict, hidden_sizes, input_size, output_mode,
                 encoder_name,
                 mode, n_eps, seed,
                 oneply, oneply_acting, boltzmann_temp,
                 bf16_collect, met_path,
                 vectorized_batch, vectorized_device, max_episode_plies,
                 met_scopes, activation)
                for n_eps, seed in zip(splits, seeds)
            ]

        # BG_ROUND_PHASES=1: one line per round with the master's time by
        # phase, in seconds; also kept in metrics_out["round_phases"].
        phases_on = os.environ.get("BG_ROUND_PHASES") == "1"
        ph = {}

        # Pipeline pre-roll: start round 0's collect before the loop.
        use_pipeline = pipeline_collect and pool is not None and not vec_active
        if use_pipeline:
            print("  pipelined collection: ON (round N+1 collects during "
                  "round N's training step)", flush=True)
        if c_active and not is_match:
            if pool is not None:
                transport, how = "pipe", "pickled through the pool pipe"
            else:
                transport = "in-process"
                how = "in-process (one collector, no transport)"
            print(f"  C collection rows: {how}", flush=True)
            if metrics_out is not None:
                metrics_out["c_collect_transport"] = transport
        pending = None
        if use_pipeline:
            pending = pool.map_async(
                _collect_worker,
                _build_worker_args(min(episodes_per_round, num_episodes), 0),
            )

        def _collect_round(round_size, round_idx, pending):
            """One round's collection -> (enc_chunks, tgt_chunks,
            n_discarded, pending, worker_rows). Raises WorkerDied
            (pool_guard) instead of hanging on a dead worker; the caller
            rebuilds the pool and calls it again. worker_rows: the rows
            of each worker's result, in worker order (BG_COLLECT_DUMP)."""
            enc_chunks: List[np.ndarray] = []
            tgt_chunks: List[np.ndarray] = []
            worker_rows: List[int] = []

            def _take(results) -> int:
                # Keep each worker's (enc, tgt) chunk; count discards. A
                # result is (enc, tgt) (the Python match worker) or
                # (enc, tgt, n_discarded) (C, vectorized, money).
                discarded = 0
                for res in results:
                    w_enc, w_tgt = res[0], res[1]
                    worker_rows.append(len(w_enc))
                    if len(w_enc) > 0:
                        enc_chunks.append(w_enc)
                        tgt_chunks.append(w_tgt)
                    if len(res) > 2:
                        discarded += res[2]
                return discarded

            n_discarded = 0
            if c_active and pool is None:
                # Single-process C: one worker's share on a seed drawn
                # from the run RNG, as a one-worker pool would.
                weights = _c_weights(round_idx)
                round_seeds[round_idx] = [rng.randint(0, 2**31 - 1)]
                if is_match:
                    res = _c_collect_match(
                        weights, mode, round_size, round_seeds[round_idx][0],
                        oneply, met_path)
                else:
                    res = _c_collect(
                        weights, mode, round_size, round_seeds[round_idx][0],
                        oneply, max_episode_plies)
                n_discarded = _take([res])
            elif vec_active:
                # Lockstep GPU drivers: (enc, tgt, n_discarded) per worker.
                from vectorized_matchplay import vectorized_matchplay_worker
                n_discarded = _take(watch.map(
                    vectorized_matchplay_worker,
                    _build_vec_args(round_size, round_idx),
                ))
            elif pool is None:
                # Single-process: the master agent collects, so its bf16
                # copy is re-synced to the current weights each round and
                # its matchplay new-game memo (net-answered entries)
                # invalidated, per the documented invariant.
                if hasattr(self.agent, "clear_matchplay_memo"):
                    self.agent.clear_matchplay_memo()
                if bf16_collect:
                    if not self.agent.bf16_inference:
                        self.agent.bf16_inference = True
                    self.agent.refresh_bf16_inference()
                n_rows = 0
                for _ in range(round_size):
                    encs, tgts = collect_episode(
                        self.agent, mode, rng, oneply=oneply,
                        max_episode_plies=max_episode_plies,
                    )
                    if len(encs) > 0:
                        enc_chunks.append(encs)
                        tgt_chunks.append(tgts)
                        n_rows += len(encs)
                    else:
                        n_discarded += 1
                worker_rows.append(n_rows)
            elif use_pipeline:
                if pending is None:
                    # First round after a pool restart: resubmit this round.
                    pending = watch.map_async(
                        _collect_worker,
                        _build_worker_args(round_size, round_idx))
                # Pipelined: submit the NEXT round's collect BEFORE
                # training (its workers see one-round-stale weights).
                next_pending = None
                _t_submit = time.perf_counter()
                if round_idx + 1 < num_rounds:
                    next_round_size = min(
                        episodes_per_round,
                        num_episodes - (episodes_played + round_size),
                    )
                    next_pending = watch.map_async(
                        _collect_worker,
                        _build_worker_args(next_round_size, round_idx + 1),
                    )
                ph["submit"] = time.perf_counter() - _t_submit
                n_discarded = _take(watch.get(pending))
                pending = next_pending
            elif eval_srv is not None:
                # The GPU server must hold THIS round's weights before any
                # worker issues a forward: workers also build a local net
                # from the same state_dict (for the paths that stay local),
                # and a version skew between the two is a correctness bug.
                if _srv_kind == "student":
                    eval_srv.update_weights(net.state_dict())
                # The round's map is POLLED (`map_watched`): a SIGKILLed
                # worker or a dead server otherwise stalls the master
                # indefinitely (the replacement worker finds no free slot).
                n_discarded = _take(eval_srv.map_watched(
                    pool, _collect_worker,
                    _build_worker_args(round_size, round_idx)))
            else:
                # Parallel, synchronous.
                n_discarded = _take(watch.map(
                    _collect_worker, _build_worker_args(round_size, round_idx),
                ))

            return enc_chunks, tgt_chunks, n_discarded, pending, worker_rows

        # BG_TRAIN_CUDA_GRAPH (unset or 1 = on on a cuda device, the
        # default; 0 = off): replay full batches of the train step from a
        # CUDA graph (cuda_graph_step.py). Bitwise equal to train_step_dev;
        # implies the on-device pool. BG_POOL_ON_DEVICE=1: the on-device
        # pool with the eager step (train_step_dev).
        on_cuda = str(self.agent.device).startswith("cuda")
        graph_step = None
        if graph_env is not None:
            if not on_cuda:
                _graph_mode, why = None, f"device is {self.agent.device}"
            else:
                _graph_mode, why = resolve_graph_mode(self, graph_env)
            if why is None:
                graph_step = GraphedTrainStep(
                    self, batch_size, input_size, mode=_graph_mode)
                print(f"  NOTE: {graph_env_label()}: full "
                      f"batches replay a CUDA graph of the train step (mode "
                      f"{_graph_mode}); the pool is kept on {self.agent.device}.",
                      flush=True)
            elif on_cuda or graph_env_explicit():
                # The default stays quiet off cuda.
                print(f"  NOTE: {GRAPH_ENV} ignored ({why}); eager train "
                      f"step.", flush=True)
        pool_on_dev = (pool_dev_env or graph_step is not None) and on_cuda
        if pool_dev_env and graph_step is None:
            print(f"  NOTE: BG_POOL_ON_DEVICE=1: "
                  + (f"the pool is kept on {self.agent.device} (eager "
                     f"train step)." if on_cuda else
                     f"ignored (device is {self.agent.device})."),
                  flush=True)

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
                _sig["lr"] = self.optimizer.param_groups[0]["lr"]

                # ── Collect this round into (encodings, targets) ───
                ph.clear()
                _t_collect_start = time.perf_counter()
                collector = ("c" if c_active else "vectorized" if vec_active
                             else "single" if pool is None else "serial")
                while True:
                    try:
                        (enc_chunks, tgt_chunks, n_discarded,
                         pending, worker_rows) = _collect_round(
                            round_size, round_idx, pending)
                        break
                    except WorkerDied as err:
                        # Rebuild the pool and redo the round (new worker
                        # seeds; a C round re-exports its weights).
                        pool_restarts += 1
                        if pool_restarts > restart_limit:
                            raise SystemExit(
                                f"ERROR: {err} in round {round_idx + 1}; "
                                f"giving up after {restart_limit} pool "
                                f"restart(s) (BG_MAX_POOL_RESTARTS). Resume "
                                f"from the last checkpoint.") from err
                        announce_restart(err, pool_restarts, restart_limit,
                                         round_idx)
                        shutdown_pool(pool)
                        pool = _new_pool()
                        watch = PoolWatch(pool)
                        pending = None   # its tasks died with the old pool
                if c_active:
                    _c_release(round_idx)
                seeds_used = round_seeds.pop(round_idx, [])
                if n_discarded and is_match:
                    print(
                        f"  round {round_idx}: {n_discarded}/"
                        f"{round_size} episodes DISCARDED "
                        f"(exceeded {max_episode_plies} plies/game)"
                    )

                episodes_played += round_size
                # From here to the end of the round the weights are
                # mid-update: a signal is deferred, not acted on.
                _sig["busy"] = True
                _collect_dt = time.perf_counter() - _t_collect_start
                if metrics_out is not None:
                    metrics_out.setdefault("collection_times", []).append(
                        _collect_dt)
                    metrics_out.setdefault("round_sizes", []).append(round_size)
                    metrics_out.setdefault("discarded", []).append(n_discarded)
                if not enc_chunks:
                    # Nothing usable this round; say so if every episode
                    # ran away.
                    if n_discarded and not is_match:
                        print(
                            f"Round {round_idx + 1:4d}/{num_rounds} "
                            f"| episodes {episodes_played}/{num_episodes} "
                            f"| pool     0 "
                            f"| DISCARDED {n_discarded}/{round_size} overlong "
                            f"episodes (>{max_episode_plies} plies) -- "
                            f"no usable samples this round",
                            flush=True,
                        )
                    if _dump_dir:
                        _dump_round(_dump_dir, round_idx,
                                    np.empty((0, input_size), np.float32),
                                    np.empty((0,), np.float32), worker_rows,
                                    seeds_used, n_discarded, collector)
                    _sig["done"] = episodes_played
                    _sig["busy"] = False
                    if _sig["pending"] is not None:
                        _teardown_and_exit(_sig["pending"], save=True)
                    continue

                _t = time.perf_counter()
                pool_enc = np.concatenate(enc_chunks, axis=0)
                pool_tgt = np.concatenate(tgt_chunks, axis=0)
                ph["assemble"] = time.perf_counter() - _t
                _t = time.perf_counter()
                if _dump_dir:
                    _dump_round(_dump_dir, round_idx, pool_enc, pool_tgt,
                                worker_rows, seeds_used, n_discarded,
                                collector)
                ph["dump"] = time.perf_counter() - _t
                _t_train_start = time.perf_counter()
                # Release the per-worker chunks: pool_enc/pool_tgt own a
                # copy, and keeping both doubles the round's footprint.
                enc_chunks = None
                tgt_chunks = None
                n = len(pool_enc)

                # Fancy-index per batch (a materialized `pool_enc[perm]`
                # would double peak RSS). Both paths read the losses back
                # once per pass, not per batch.
                if pool_on_dev:
                    # On-device pool (BG_POOL_ON_DEVICE=1 or the graph):
                    # one H2D copy per round, the same shuffle (numpy, same
                    # seed draw), the batch gathered on the device. Same
                    # values into the same kernels: bitwise equal to the
                    # host-pool loop (tests/test_train_step_bitwise.py).
                    _dev = self.agent.device
                    pool_enc_t = torch.from_numpy(pool_enc).to(
                        _dev, dtype=torch.float32)
                    pool_tgt_t = torch.as_tensor(
                        pool_tgt, dtype=torch.float32).to(_dev)
                    # The device owns the round now; the host copy (~2.9
                    # GB at 11pt epr 4000) is not read again.
                    pool_enc = pool_tgt = None
                    if phases_on:
                        torch.cuda.synchronize(_dev)
                ph["h2d"] = time.perf_counter() - _t_train_start
                ph["perm"] = ph["steps"] = 0.0
                for _epoch in range(epochs_per_round):
                    _t = time.perf_counter()
                    perm = np.arange(n)
                    rng_np = np.random.default_rng(rng.randint(0, 2**31 - 1))
                    rng_np.shuffle(perm)
                    if pool_on_dev:
                        perm_t = torch.from_numpy(perm).to(_dev)
                    _t_steps = time.perf_counter()
                    ph["perm"] += _t_steps - _t
                    ph.setdefault("t_steps0", _t_steps)
                    round_losses = []
                    if pool_on_dev:
                        for start in range(0, n, batch_size):
                            idx_t = perm_t[start:start + batch_size]
                            if graph_step is not None:
                                round_losses.append(graph_step.step(
                                    pool_enc_t, pool_tgt_t, idx_t))
                            else:
                                round_losses.append(self.train_step_dev(
                                    pool_enc_t[idx_t], pool_tgt_t[idx_t]))
                    else:
                        for start in range(0, n, batch_size):
                            idx = perm[start:start + batch_size]
                            batch_e = pool_enc[idx]
                            batch_t = pool_tgt[idx]
                            round_losses.append(
                                self.train_step_nosync(batch_e, batch_t))
                    losses.extend(torch.stack(round_losses).tolist())
                    ph["steps"] += time.perf_counter() - _t_steps
                round_losses = None
                if pool_on_dev:
                    perm_t = idx_t = None
                    pool_enc_t = pool_tgt_t = None
                    # Give the pool's VRAM back between rounds. Kept in the
                    # allocator's cache, the next round's pool (another row
                    # count) is a second allocation beside it at the first
                    # larger round; the collection workers share the card.
                    torch.cuda.empty_cache()
                    # As train_step after every step (replays skip it).
                    memo = getattr(self.agent, "_new_game_memo", None)
                    if memo:
                        memo.clear()

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
                        if n_discarded and not is_match else ""
                    )
                    # BG_ROUND_TIMING=1 appends the collect/train split.
                    _rt = ""
                    if os.environ.get("BG_ROUND_TIMING") == "1":
                        _rt = (f" | collect {_t_train_start - _t_collect_start:.2f}s"
                               f" train {time.perf_counter() - _t_train_start:.2f}s")
                    print(
                        f"Round {round_idx + 1:4d}/{num_rounds} "
                        f"| episodes {episodes_played}/{num_episodes} "
                        f"| pool {n:5d} "
                        f"| lr {_sig['lr']:.6g} "
                        f"| recent batch loss {avg:.3e}"
                        + discard_note + _rt
                    )
                if phases_on:
                    ph.update(collect=_collect_dt, rows=n,
                              t_collect0=_t_collect_start,
                              mb=n * (input_size + 1) * 4 / 1e6,
                              t_steps_end=time.perf_counter())
                    print(f"  phases round {round_idx + 1}: "
                          + " ".join(f"{k} {ph[k]:.4f}" for k in (
                              "collect", "submit", "assemble", "dump",
                              "h2d", "perm", "steps") if k in ph)
                          + f" | rows {n} ({ph['mb']:.1f} MB)", flush=True)
                    if metrics_out is not None:
                        metrics_out.setdefault("round_phases", []).append(
                            dict(ph))

                if next_save is not None and episodes_played >= next_save:
                    root, ext = os.path.splitext(save_path)
                    ckpt_path = _unique_ckpt_path(
                        f"{root}_ep{episode_offset + episodes_played}"
                        f"{ext or '.pt'}")
                    self.agent.network.save(
                        ckpt_path, train_params=train_params)
                    print(f"  -> checkpoint saved: {ckpt_path}", flush=True)
                    while next_save <= episodes_played:
                        next_save += save_every

                # Round fully trained (and checkpointed): weights are
                # now exactly "after `episodes_played` episodes".
                _sig["done"] = episodes_played
                _sig["busy"] = False
                if _sig["pending"] is not None:
                    _teardown_and_exit(_sig["pending"], save=True)

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
            # Put the previous signal dispositions back before anything
            # else, so a second signal during teardown behaves normally
            # instead of re-entering a handler whose pool is already gone.
            if _prev_sig_handlers:
                for _s, _prev in _prev_sig_handlers.items():
                    try:
                        _signal.signal(_s, _prev)
                    except (ValueError, OSError, TypeError):
                        pass
            if pool is not None:
                if sys.exc_info()[0] is None:
                    pool.close()
                    pool.join()
                else:
                    # A failed run may hold a dead or stuck worker: a plain
                    # terminate()/join() can block forever (pool_guard).
                    shutdown_pool(pool)
            # AFTER the pool is joined, never before: stop() unlinks the
            # shared-memory segments the workers have mapped, and pulling
            # them out from under a live worker is a segfault.
            if eval_srv is not None:
                eval_srv.stop()
            if c_dir is not None:
                # After the pool: no worker reads a weights file any more.
                shutil.rmtree(c_dir, ignore_errors=True)

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
        `td_update`. Uses the optimizer's `lr` but not the optimizer
        itself. Pair with SGD.

        Cubeful modes dispatch to the cubeful variant.

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
        """Online TD(0) for cubeful money games. Same per-turn samples
        as _collect_episode_cubeful_money, but each is applied with
        `td_update_encoded` as soon as it is produced.
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
                        # No double: target = checker-phase value.
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
