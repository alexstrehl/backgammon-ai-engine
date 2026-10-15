"""Generator-based vectorized (lockstep) self-play collection for
MATCHPLAY with 1-ply checker targets.

Design (ported from the money-game engine's cubeful-money
`vectorized_collect.py`, but every mirror rewritten for matchplay):

Each EPISODE (= one full match) is a Python generator that runs the
exact logic of `trainer._collect_episode_matchplay` but, wherever the
serial code would call `network(...)`, `yield`s the (N, n_feat) float32
feature matrix it was about to push through the net and receives the
(N,) float32 values back via `.send()`. A per-process driver holds B
live generators, concatenates their B pending matrices into ONE local
forward on `device` (in-process, no IPC), scatters the slices back, and
refills finished slots until the episode quota is met.

The correctness contract is deliberately narrow and mechanical:

    the matrix a generator yields is byte-identical to the input of the
    serial forward it replaces, and the value it receives is used in
    exactly the same expression.

Everything else in the episode -- the score curriculum, the cube phase
and its exploration coins, boundary upweighting, the outcome-mix pass,
the `random.Random` consumption order -- is a line-for-line mirror. That
is what makes single-game driving (B=1, CPU driver net) BIT-IDENTICAL to
`trainer._collect_episode_matchplay` on a shared seed, which is what
`tests/test_vectorized_matchplay.py` proves.

WHAT IS *NOT* YIELDED
`TDAgent._eval_new_game` stays on the worker's own CPU agent net. It is
either an exact MET lookup (zero forwards, with `--met`) or two batch-1
forwards memoised on a tiny score lattice, and it is reached from four
places -- `v_dp`, the responder's take point, every terminal successor
in both move selection and the 1-ply target, and the game-boundary
target -- several of them inside loops holding a live `bg_fast` lazy
states view. Yielding from there would force a restructuring of exactly
the most delicate code for a handful of microseconds per worker call.
Running it on the agent's CPU fp32 net keeps it bit-identical to the
serial collector. The memo is per-agent and order-independent (weights
are frozen for the round), so lockstep interleaving cannot change it.

Likewise `respond_to_double_matchplay(hint=offer)` is pure arithmetic on
the offer cache and consumes no forward at all.

WHAT IS NOT FUSED
`offer_double_matchplay` does two separate batch-1 forwards (v_nd and
v_dt). They are yielded SEPARATELY rather than stacked into one 2-row
request: torch's reductions are batch-composition sensitive at the
~1e-7 level, and a cube decision is a `min`/`max` comparison that gates
the RNG stream, so a fused pair would cost exact parity for one saved
lockstep step (which is already amortised across B games anyway).

For the same reason the 1-ply target keeps its per-dice
`get_legal_plays_encoded` loop rather than switching to the batched
`expand_21` (which omits terminal replies and would change the batch
composition of every remaining row), keeps its float64 Python
accumulation over `_DICE_OUTCOMES` rather than a dot product. Move
selection keeps the PYTHON `get_legal_plays`. (An earlier version of
this paragraph said the C fused generator emits plays in a different
order; it does not: the same successors in the same order on all 36
ordered rolls -- `c_api/movegen_fuzz.py`, 8,640,000 board x roll cases,
0 mismatches, and `tests/test_movegen_order_stable.py`. The serial collector
under BG_FAST_FORWARD uses the C generator and matches this one.)

EPISODE-LENGTH GUARD (NEW BEHAVIOUR, DEFAULT OFF)
Unlike the money-game collectors, the serial match collector has NO
ply bound of any kind. A guard is nevertheless operationally necessary here:
in lockstep a single runaway game does not merely grow one worker's RSS,
it pins a driver slot forever, so the round never finishes even though
the other B-1 games completed. `max_episode_plies` is therefore checked
at the top of the PER-GAME loop, before any work for that ply, and
DISCARDS the whole episode (the whole match). It defaults to 0
(disabled) precisely so that the default vectorized path is
behaviour-identical to the serial collector.

Note that a matchplay episode cannot loop forever at the MATCH level:
every `after_game` strictly decreases one away score, so a match is at
most 2*match_length-1 games. Only a single game can run away.

Discards are reported explicitly rather than inferred from an empty
return: each generator returns `(enc, tgt, discarded)` and the worker
returns `(enc, tgt, n_discarded)`, so a legitimately-empty episode is
never miscounted as a runaway.

SCOPE
Matchplay only, 1-ply targets (`oneply=True`), 0-ply move selection.
`--oneply-acting` does not exist for matchplay and is refused.
`boltzmann_temp > 0` is refused: `_boltzmann_select` draws from torch's
GLOBAL generator, whose call order under lockstep depends on B, which
would make results non-reproducible across batch sizes.
`B < 1` is refused: a zero/negative batch fills no driver slot, so the
worker would return an empty result for every episode it never started.

HOW EXACT IS "EQUIVALENT" AT B > 1
Exactly bit-exact at B=1 (that is a construction guarantee, see above),
and at B > 1 the row COUNT, row ORDER, ENCODINGS and trajectories are
still exactly those of the serial collector -- only the target VALUES
move, because a row sits at a different offset inside the fused GEMM.
Measured magnitude of that shift (production 512/512/256/256 net, 3pt,
24-60 episodes): <= 3.0e-7 with a CPU driver, <= 4.2e-7 with a CUDA
driver, mean target identical to 6 decimal places. It is pure fp32
batch composition, not aliasing: driven by a batch-composition-
INVARIANT network the B=1 and B=32 row multisets are EXACTLY equal
(`tests/test_vectorized_matchplay.py::TestBatchInvariance`).

VRAM
One CUDA context per DRIVER PROCESS: the design amortises contexts
across B games, it does not avoid multiplying them across workers.
Measured 434-466 MiB per driver on a 3080 Ti (B=24..32), i.e. ~500 MB
budgeted by `vram_mb_per_driver()`. `--workers 48` would therefore want
~24 GB and cannot run on a 12 GB card; `Trainer.train` refuses that up
front (see `vram_reason`) and this module additionally falls back to a
CPU driver if a CUDA failure escapes the check at run time, so a worker
never takes the whole run down with it.

BENCHMARK (RTX 3080 Ti, 3pt, 512/512/256/256, box shared
with a 40-core job -- ALL of these numbers move +/-20% with box load)
    CPU serial   K=8             8.0-8.9 ep/s
    CPU serial   K=12           11.2-11.5 ep/s
    vectorized   K=8   B=24     41.5-41.9 ep/s
    vectorized   K=12  B=24     54-66 ep/s
    vectorized   K=16  B=24     58-77 ep/s   (~7.0 GB VRAM)
    vectorized   K=20  B=24     78.5 ep/s    (~8.8 GB VRAM)
~5x per core over the CPU baseline at equal core count. The original
commit message for this file claimed "K=16 is past the knee" off a
single K=16 sample of 58.19 ep/s; that does not reproduce -- five
interleaved K=12/K=16 pairs put K=16 ahead every time, and GPU
utilisation at K=12 averages only ~46%. The knee is nearer K=16-20,
bounded by VRAM (~25 drivers at B=32 on a 12 GB card) rather than by
the GPU. Whole-box framing: the CPU path scales to ~45 ep/s at 48
workers, so the honest headline is ~1.7x peak throughput while freeing
~30 cores, not 5.7x.

These rates need a round big enough to keep the drivers full. The
benchmark gave each worker 50-200 episodes per `pool.map`; at the
trainer's default `--episodes-per-round 100` with K=12 each worker gets
~8, fewer than B, so batches shrink and the round's tail is one long
match -- measured 20-40 ep/s for the same config. Keep
`episodes_per_round >= workers * vectorized_batch`; `Trainer.train`
prints a NOTE when it is not.
"""

from __future__ import annotations

import os
import random
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
import torch

from agents import CubeOffer
from backgammon_engine import get_legal_plays, opening_roll, switch_turn
from encoding import OPP_OFF_INDEX, TERMINAL_OFF_THRESHOLD
from model import TDNetwork
import td_agent as _ta
from td_agent import TDAgent
from trainer import _BOUNDARY_UPWEIGHT, _CUBE_EXPLORATION_EPS, _curriculum_start

_bg_fast = _ta._bg_fast
_DICE_OUTCOMES = _ta._DICE_OUTCOMES


_EMPTY_ENC = np.empty((0, 0), dtype=np.float32)
_EMPTY_TGT = np.empty((0,), dtype=np.float32)


def _discarded():
    """Return value for an episode killed by the ply bound."""
    return _EMPTY_ENC, _EMPTY_TGT, True


# ── device / VRAM admission control ──────────────────────────────────
#
# A worker-side CUDA failure would otherwise kill the whole run from
# inside `pool.map` with no checkpoint, which is exactly the failure
# mode every OTHER unmet precondition avoids by falling back. These two
# predicates let `Trainer.train` refuse the config up front, in the same
# "name the reason, use the serial collector" style; `_run_lockstep`'s
# CPU retry (below) is the backstop for what they cannot foresee.

_VRAM_MB_PER_DRIVER = 500.0  # measured 434-466 MiB/driver, B=24..32


def vram_mb_per_driver() -> float:
    """Budgeted VRAM per driver process. Dominated by the fixed CUDA
    context, so it is nearly independent of B. `VEC_VRAM_MB_PER_DRIVER`
    overrides it for a different card / a measured B."""
    env = os.environ.get("VEC_VRAM_MB_PER_DRIVER")
    if env:
        try:
            v = float(env)
            if v > 0:
                return v
        except ValueError:
            pass
    return _VRAM_MB_PER_DRIVER


# ── CUDA MPS ────────────────────────────────────────────────────────
#
# Every process that opens a CUDA context gets its own, and contexts
# time-slice rather than run concurrently. 0-ply collection yields only
# ~8 rows per slot-yield (62.6% of yields are batch-1 cube rows), so the
# round trip is latency-bound and the W driver contexts spend their time
# being switched between instead of computing. MPS gives them one shared
# GPU scheduler and the time-slicing goes away.
#
# Measured on a 9980X + RTX 5090, matchplay 0-ply, B=64 epr=4096, same
# checkpoint both arms: collect 12.1s -> 5.8s, 230 -> 356 ep/s (+56%) at
# W=32. The gain is contention between contexts, so it scales with the
# WORKER COUNT, not with the ply depth: the same A/B at W=16 was only
# +8%, and 1-ply (~350-400 rows per trip, already GPU-bound) is ~0%.
#
# Detected and reported, never started: `nvidia-cuda-mps-control -d` is
# a HOST-WIDE daemon that changes GPU behaviour for every other CUDA
# client and weakens error isolation (one client's fault can poison the
# shared context). That is a decision for the operator of the box, not
# for a training script -- on a shared box it would silently affect
# somebody else's job. See ./mps.sh.
#
# Throughput only: MPS changes how kernels are scheduled, not what they
# compute. The A/B above held batch loss (2.1-3.3e-4) and pool size
# (2.13-2.15M) indistinguishable across arms.

_MPS_DEFAULT_PIPE_DIR = "/tmp/nvidia-mps"

# Below this many drivers there is too little contention to be worth the
# daemon: +8% measured at W=16 vs +56% at W=32.
MPS_MIN_WORKERS = 16


def mps_pipe_dir() -> str:
    """The control-pipe directory this process's CUDA client would use."""
    return os.environ.get("CUDA_MPS_PIPE_DIRECTORY") or _MPS_DEFAULT_PIPE_DIR


def mps_active() -> bool:
    """True if a live MPS control daemon is reachable on that pipe dir.

    Read from the filesystem rather than by making a CUDA call: the
    client library decides whether to route through MPS at
    context-creation time, from CUDA_MPS_PIPE_DIRECTORY, so this is the
    same question the driver asks -- and it costs nothing when MPS is
    absent.

    The pid is verified because a pipe directory outlives the daemon
    that made it: reporting "MPS ON" into a run log that is then used to
    explain a throughput number would be worse than reporting nothing.
    """
    d = mps_pipe_dir()
    if not os.path.exists(os.path.join(d, "control")):
        return False
    try:
        with open(os.path.join(d, "nvidia-cuda-mps-control.pid")) as fh:
            pid = int(fh.read().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False          # stale pipe dir, daemon is gone
    except PermissionError:
        pass                  # alive, owned by another user
    except OSError:
        return False
    return True


def device_reason(device: str) -> Optional[str]:
    """None if `device` can host a driver net on this host, else a
    human-readable reason. Catches the two run-killers the old
    `torch.cuda.is_available()` check let through: a malformed device
    string and a valid-looking but nonexistent ordinal ('cuda:7')."""
    if device == "cpu":
        return None
    try:
        dev = torch.device(device)
    except (RuntimeError, ValueError, TypeError) as e:
        return f"--vectorized-device {device!r} is not a valid device ({e})"
    if dev.type == "cpu":
        return None
    if dev.type != "cuda":
        return (
            f"--vectorized-device {device!r}: only 'cpu' and 'cuda[:N]' "
            "are supported"
        )
    if not torch.cuda.is_available():
        return (
            f"device {device!r} unavailable "
            "(torch.cuda.is_available() is False)"
        )
    n = torch.cuda.device_count()
    idx = dev.index if dev.index is not None else 0
    if idx >= n:
        return (
            f"--vectorized-device {device!r}: invalid device ordinal "
            f"(this host has {n} CUDA device{'s' if n != 1 else ''})"
        )
    return None


def _free_vram_mb(device: str) -> Optional[float]:
    """Free VRAM on `device` in MB, or None if that is not a measurable
    question here (cpu device, no CUDA, bad ordinal, driver refuses).
    Split out so the admission check is testable without a GPU."""
    try:
        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        idx = dev.index if dev.index is not None else 0
        if idx >= torch.cuda.device_count():
            return None
        free_b, _total_b = torch.cuda.mem_get_info(idx)
    except Exception:
        return None
    return free_b / (1024.0 * 1024.0)


def vram_reason(device: str, workers: int) -> Optional[str]:
    """None if `workers` drivers plausibly fit in the FREE VRAM of
    `device`, else a reason naming the arithmetic. Set
    `VEC_SKIP_VRAM_CHECK=1` to bypass (e.g. if you know the resident
    process is about to exit)."""
    if os.environ.get("VEC_SKIP_VRAM_CHECK") == "1":
        return None
    free_mb = _free_vram_mb(device)
    if free_mb is None:
        return None  # not measurable / not a CUDA device: let it try
    need_mb = workers * vram_mb_per_driver()
    # 0.9: mem_get_info is a snapshot, and the master's own training
    # step wants headroom on the same card.
    if need_mb > 0.9 * free_mb:
        return (
            f"--workers {workers} needs ~{need_mb:.0f} MB of VRAM "
            f"(~{vram_mb_per_driver():.0f} MB per driver process) but "
            f"only {free_mb:.0f} MB is free on {device} -- lower "
            f"--workers to <= {int(0.9 * free_mb / vram_mb_per_driver())}"
            " (or set VEC_SKIP_VRAM_CHECK=1)"
        )
    return None


def _is_cuda_failure(exc: BaseException) -> bool:
    """True for the RuntimeErrors a CUDA driver raises when the card is
    full / the ordinal is wrong / the context dies."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "CUDA" in str(exc)


# ── generator mirrors of the forward-using TDAgent methods ───────────


def _evaluate_matchplay_gen(agent, state, match_state, is_cube_action=False):
    """Generator mirror of TDAgent.evaluate_matchplay.

    Serial: `network(from_numpy(x).unsqueeze(0)).item()`. The yielded
    matrix is that same (1, n_feat) input.
    """
    x = agent._encode_matchplay(state, match_state, is_cube_action)
    vals = yield x.reshape(1, -1)
    return float(np.asarray(vals, dtype=np.float32).reshape(-1)[0])


def _offer_double_matchplay_gen(agent, state, match_state):
    """Generator mirror of TDAgent.offer_double_matchplay.

    Two separate batch-1 yields (v_nd, v_dt), deliberately NOT fused --
    see the module docstring. `v_dp` goes through `_eval_new_game`,
    which is not yielded.
    """
    player = state.turn
    if not match_state.can_offer(player):
        return CubeOffer(should_double=False, _cache=None)

    v_nd = yield from _evaluate_matchplay_gen(
        agent, state, match_state, is_cube_action=False,
    )
    ms_take = match_state.after_take(player)
    v_dt = yield from _evaluate_matchplay_gen(
        agent, state, ms_take, is_cube_action=False,
    )
    ms_pass = match_state.after_game(1, winner_is_me=(player == 0))
    if ms_pass.match_over():
        v_dp = 1.0
    else:
        v_dp = agent._eval_new_game(ms_pass, player)

    should_double = min(v_dt, v_dp) > v_nd
    return CubeOffer(
        should_double=should_double,
        _cache={"v_nd": v_nd, "v_dt": v_dt, "v_dp": v_dp},
    )


def _choose_checker_action_matchplay_gen(agent, state, dice, match_state,
                                         rng=None):
    """Generator mirror of TDAgent.choose_checker_action_matchplay
    (0-ply selection). Yields the (n_plays, n_feat) matrix; receives
    (n_plays,) values. Returns (chosen_state, bootstrap) or None.

    Uses the PYTHON `get_legal_plays`, as the serial method does without
    BG_FAST_FORWARD. The C generator (the serial method's path under
    BG_FAST_FORWARD) emits the same plays in the same order on all 36
    ordered rolls (c_api/movegen_fuzz.py, 8,640,000 cases, 0 mismatches;
    tests/test_movegen_order_stable.py), so argmin breaks ties alike
    either way; an earlier version of this note said otherwise.
    """
    plays = get_legal_plays(state, dice)
    if not plays:
        return None
    mover = state.turn
    opponent = 1 - mover
    next_states = [s for _, s in plays]

    if agent._c_base_available:
        base_feats = np.stack([
            _bg_fast.encode_state(s) for s in next_states
        ])
    else:
        base = agent.encoder._base
        base_feats = np.stack([base.encode(s) for s in next_states])

    persp, cube_value, my_away, opp_away, crawford = (
        agent._match_features(match_state, opponent)
    )
    opp_cube_action = match_state.can_offer(opponent)
    extra_row = agent.encoder.match_extras(
        persp, cube_value, my_away, opp_away, crawford, opp_cube_action,
    )
    extra = np.tile(extra_row, (len(next_states), 1))
    features = np.concatenate([base_feats, extra], axis=1)

    # `next_states` are real Python BoardStates (python move-gen) and
    # `features` is a freshly concatenated private array, so both are
    # safe to hold across the yield while other games' move-gen runs.
    vals = yield features

    # Shape must be the serial one, (n,): `TDNetwork.forward` ends in
    # `.squeeze(-1)`, so `values = self.network(t)` in the serial method
    # is 1-D. At temp 0 `_boltzmann_select` is an argmin and cannot tell
    # (n,) from (n, 1), but at temp > 0 it does `values.squeeze()` and a
    # softmax over dim 0 -- a (n, 1) tensor would sample from a
    # different distribution. Mirror the serial shape so that stays
    # true if the boltzmann refusal is ever relaxed.
    values = torch.from_numpy(
        np.asarray(vals, dtype=np.float32).reshape(-1).copy()
    )

    terminal_mask = features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
    if terminal_mask.any():
        values = values.clone()
        for i in np.nonzero(terminal_mask)[0].tolist():
            gr = next_states[i].game_result()
            new_ms = match_state.after_game(gr, winner_is_me=(mover == 0))
            if new_ms.match_over():
                mover_val = 1.0
            else:
                mover_val = agent._eval_new_game(new_ms, mover,
                                                 scope="terminal")
            values[i] = 1.0 - mover_val
    idx = agent._boltzmann_select(values, rng=rng)
    chosen = next_states[idx]
    bootstrap = 1.0 - float(values[idx].item())
    return chosen, bootstrap


def _value_oneply_checker_matchplay_gen(agent, state, match_state):
    """Generator mirror of TDAgent.value_oneply_checker_matchplay.

    Yields at most twice: once for the lazily-computed forced-pass row
    (only if some dice has no legal reply), then once for the big
    concatenated 21-dice reply matrix. Returns the 1-ply MWC float.

    The per-dice loop, the float32 terminal overrides, the float64
    Python accumulation over `_DICE_OUTCOMES` and the final [0, 1] clamp
    are all reproduced verbatim -- each of them is observable in the
    last bits of the target.
    """
    mover = state.turn
    opponent = 1 - mover
    persp, cube_value, my_away, opp_away, crawford = (
        agent._match_features(match_state, opponent)
    )
    opp_cube_action = match_state.can_offer(opponent)

    feat_chunks: List[np.ndarray] = []
    per_dice: List[Tuple[Optional[int], Optional[int], float, Optional[float]]] = []
    term_overrides: List[Tuple[int, float]] = []
    forced_value: Optional[float] = None
    pos = 0
    for d1, d2 in _DICE_OUTCOMES:
        prob = (1.0 / 36.0) if d1 == d2 else (2.0 / 36.0)
        if agent._c_base_available:
            base_feats, states_view = _bg_fast.get_legal_plays_encoded(
                state, (d1, d2),
            )
            n_moves = base_feats.shape[0] if base_feats is not None else 0
        else:
            plays = get_legal_plays(state, (d1, d2))
            n_moves = len(plays)
            if n_moves:
                states_view = [s for _, s in plays]
                base = agent.encoder._base
                base_feats = np.stack([base.encode(s) for s in states_view])
        if not n_moves:
            if forced_value is None:
                # The only yield inside this loop. No lazy bg_fast view
                # is live on this branch, and `feat_chunks` holds owned
                # copies (bg_fast copies its feature buffer), so
                # suspending here is safe.
                opp_view = switch_turn(state)
                v_opp = yield from _evaluate_matchplay_gen(
                    agent, opp_view, match_state,
                    is_cube_action=match_state.can_offer(opp_view.turn),
                )
                forced_value = 1.0 - v_opp
            per_dice.append((None, None, prob, forced_value))
            continue
        terminal_mask = base_feats[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
        if terminal_mask.any():
            # Materialize terminal successors NOW: `states_view` is a
            # lazy view invalidated by the next bg_fast call.
            for i in np.nonzero(terminal_mask)[0]:
                gr = states_view[int(i)].game_result()
                new_ms = match_state.after_game(gr, winner_is_me=(mover == 0))
                if new_ms.match_over():
                    mover_val = 1.0
                else:
                    mover_val = agent._eval_new_game(new_ms, mover,
                                                 scope="terminal")
                term_overrides.append((pos + int(i), 1.0 - mover_val))
        feat_chunks.append(base_feats)
        per_dice.append((pos, pos + n_moves, prob, None))
        pos += n_moves

    if pos == 0:
        return forced_value  # every dice is a forced pass (unclamped)

    big_base = (
        feat_chunks[0] if len(feat_chunks) == 1
        else np.concatenate(feat_chunks, axis=0)
    )
    extra_row = agent.encoder.match_extras(
        persp, cube_value, my_away, opp_away, crawford, opp_cube_action,
    )
    extra = np.tile(extra_row, (big_base.shape[0], 1))
    features = np.concatenate([big_base, extra], axis=1)

    vals = yield features
    # Serial: `network(t).cpu().numpy().ravel()` -- a float32 array that
    # the terminal overrides are then written into. Copy so we never
    # write into the driver's shared output buffer.
    values = np.array(
        np.asarray(vals, dtype=np.float32).reshape(-1), dtype=np.float32,
    )
    for idx, v in term_overrides:
        values[idx] = v

    total = 0.0
    for start, end, prob, forced in per_dice:
        if start is None:
            total += prob * forced
        else:
            total += prob * (1.0 - float(values[start:end].min()))
    return min(1.0, max(0.0, total))


# ── episode (= one full match) generator ─────────────────────────────


def collect_episode_matchplay_gen(
    agent,
    mode,
    rng: random.Random,
    initial_match_state,
    oneply: bool = True,
    max_episode_plies: int = 0,
):
    """Generator mirror of `trainer._collect_episode_matchplay`.

    Yields feature matrices for batched forwards; returns
    `(enc_array, tgt_array, discarded)` via StopIteration.

    With `max_episode_plies == 0` (the default) this is a line-for-line
    mirror: same RNG consumption order, same sample order, same
    boundary upweighting, same outcome-mix pass. Driven single-game
    against a CPU net it is bit-identical to the serial collector.

    `max_episode_plies > 0` bounds the PER-GAME turn count (reset at
    every `start_game()`) and discards the WHOLE episode -- the whole
    match -- on exceeding it, checked before any work for that ply so
    the driver slot is released immediately. See the module docstring
    for why truncate-and-continue is not an option here.
    """
    enc_list: List[np.ndarray] = []
    targets: List[float] = []
    viewers: List[int] = []

    # Score curriculum (consumes rng exactly as the serial collector).
    match_state = _curriculum_start(initial_match_state, rng)

    while True:
        game_ms = match_state.start_game()
        state, dice = opening_roll(rng)
        is_opening = True
        game_done = False
        plies = 0

        while not game_done:
            if max_episode_plies > 0 and plies >= max_episode_plies:
                return _discarded()
            plies += 1
            player = state.turn

            # ── Cube decision phase (if eligible) ────────────────────
            if not is_opening and game_ms.can_offer(player):
                enc_list.append(agent._encode_matchplay(
                    state, game_ms, is_cube_action=True,
                ))
                viewers.append(player)
                offer = yield from _offer_double_matchplay_gen(
                    agent, state, game_ms,
                )
                c = offer._cache
                v_cube_opt = max(c["v_nd"], min(c["v_dt"], c["v_dp"]))

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
                        match_state = game_ms.after_game(
                            1, winner_is_me=(player == 0),
                        )
                        if explored:
                            t = v_cube_opt
                        elif match_state.match_over():
                            t = 1.0
                        else:
                            t = agent._eval_new_game(match_state, player,
                                                 scope="boundary")
                        targets.append(t)
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
            enc_list.append(agent._encode_matchplay(
                state, game_ms, is_cube_action=False,
            ))
            viewers.append(player)
            result = yield from _choose_checker_action_matchplay_gen(
                agent, state, dice, game_ms, rng=rng,
            )
            if result is None:
                opp_view = switch_turn(state)
                if oneply:
                    # Parity with the serial collector: unconditional
                    # dice-average target (the 0-ply read below is
                    # conditioned on having danced).
                    t = yield from (
                        _value_oneply_checker_matchplay_gen(
                            agent, state, game_ms,
                        )
                    )
                else:
                    opp_cube_action = game_ms.can_offer(opp_view.turn)
                    v_opp = yield from _evaluate_matchplay_gen(
                        agent, opp_view, game_ms,
                        is_cube_action=opp_cube_action,
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
                    if oneply:
                        # Serial-collector parity: 1-ply rows keep the
                        # unconditional enumeration target at the
                        # game-ending turn; the realized boundary is
                        # conditioned on the mover's roll having won
                        # (see _collect_episode_matchplay).
                        t = yield from (
                            _value_oneply_checker_matchplay_gen(
                                agent, state, game_ms,
                            )
                        )
                    elif match_state.match_over():
                        t = 1.0
                    else:
                        t = agent._eval_new_game(match_state, player,
                                                 scope="boundary")
                    targets.append(t)
                    for _ in range(_BOUNDARY_UPWEIGHT - 1):
                        enc_list.append(enc_list[-1])
                        targets.append(t)
                        viewers.append(player)
                    game_done = True
                    continue
                if oneply:
                    v_oneply = yield from (
                        _value_oneply_checker_matchplay_gen(
                            agent, state, game_ms,
                        )
                    )
                    targets.append(v_oneply)
                else:
                    targets.append(bootstrap)
                state = next_state

            dice = (rng.randint(1, 6), rng.randint(1, 6))
            is_opening = False

        if match_state.match_over():
            break

    if not enc_list:
        return _EMPTY_ENC, _EMPTY_TGT, False
    t_arr = np.asarray(targets, dtype=np.float32)
    mix = getattr(mode, "outcome_mix", 0.0)
    if mix > 0.0:
        assert len(viewers) == len(targets)
        white_won = match_state.my_away <= 0
        z = np.asarray(
            [1.0 if (white_won if v == 0 else not white_won) else 0.0
             for v in viewers],
            dtype=np.float32,
        )
        t_arr = (1.0 - mix) * t_arr + mix * z
    return np.stack(enc_list), t_arr, False


# ── single-generator driver (used by tests and by the worker) ────────


def drive_episode(gen, net, device: str = "cpu"):
    """Run ONE generator to completion, serving each yielded matrix with
    its own forward on `net`. The forward's input is then exactly the
    matrix the generator yielded -- i.e. exactly the serial forward's
    input -- which is what makes B=1/CPU driving bit-exact.

    Returns the generator's `(enc, tgt, discarded)`.
    """
    with torch.no_grad():
        try:
            feat = next(gen)
            while True:
                t = torch.from_numpy(np.ascontiguousarray(feat))
                if device != "cpu":
                    t = t.to(device)
                out = net(t).reshape(-1).to("cpu", torch.float32).numpy()
                feat = gen.send(out)
        except StopIteration as e:
            return e.value


# ── per-process lockstep driver ──────────────────────────────────────


_VEC_FIELDS = (
    "state_dict hidden_sizes input_size output_mode encoder_name "
    "mode num_episodes seed oneply oneply_acting boltzmann_temp "
    "bf16_inference met_path B device "
    "max_episode_plies met_scopes activation"
)
# met_scopes defaults to None (= ground every scope) and activation to
# relu, so callers built before those fields existed keep working
# unchanged. `activation` is carried for the same reason the GPU eval
# server carries it: a rebuilt net that silently defaults to relu is a
# DIFFERENT FUNCTION from the master's on any non-relu checkpoint.
VecArgs = namedtuple("VecArgs", _VEC_FIELDS, defaults=(None, "relu"))


def vectorized_matchplay_worker(args):
    """Worker subprocess: run B matchplay episodes in lockstep, batching
    all pending forwards into one local forward per step. Returns
    `(encodings, targets, n_discarded)`.

    Validates the config, then runs the lockstep driver on the requested
    device. A CUDA failure (the card filled up between the master's
    admission check and this forward, a dead context) is caught and the
    whole worker call is RETRIED on the CPU driver: the retry restarts
    from the same seed with fresh local state, so it yields a valid
    round of data instead of taking the entire training run down from
    inside `pool.map` with no checkpoint. It is loud, and slow -- the
    master's `vram_reason` check is what should normally prevent it.

    NOTE: the args tuple arity must stay in sync with `_build_vec_args`
    in `Trainer.train`. It is the leading 13 fields of the `_collect_worker`
    tuple plus `(B, device, max_episode_plies, met_scopes, activation)`.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    a = VecArgs(*args)

    assert a.oneply, "vectorized_matchplay is oneply-target only"
    # 1-ply ACTING does not exist for matchplay (choose_checker_action_
    # cubeful dispatches to the matchplay method before any oneply
    # check), so accepting the flag would silently ignore it.
    assert not a.oneply_acting, (
        "vectorized_matchplay does 1-ply targets with 0-ply acting; "
        "--oneply-acting is not a matchplay option"
    )
    # temp > 0 is supported: _boltzmann_select draws from the EPISODE's
    # random.Random, not torch's global rng, so the sample order does not
    # depend on the batch size B and serial/generator parity holds.
    # B < 1 fills no driver slot, so the driver loop never runs and the
    # worker returns EMPTY for every episode it never started -- with a
    # discard count of 0, so not even the discard log would show it.
    assert a.B >= 1, (
        f"vectorized_matchplay requires --vectorized-batch >= 1 "
        f"(got {a.B}); a batch < 1 collects zero episodes"
    )

    initial_ms = a.mode.initial_match_state()
    assert initial_ms is not None and initial_ms.is_matchplay, (
        "vectorized_matchplay requires a matchplay mode"
    )

    if a.device == "cpu":
        return _run_lockstep(a, a.device)
    try:
        return _run_lockstep(a, a.device)
    except Exception as e:  # noqa: BLE001 - re-raised unless CUDA
        if not _is_cuda_failure(e):
            raise
        print(
            f"WARNING: vectorized worker (pid {os.getpid()}) hit a CUDA "
            f"failure on device {a.device!r}: {e}. Retrying this "
            "worker's episodes on the CPU driver -- the round will be "
            "slow but the run survives. Lower --workers / "
            "--vectorized-batch, or free VRAM.",
            flush=True,
        )
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return _run_lockstep(a, "cpu")


def _run_lockstep(a: "VecArgs", device: str):
    """The lockstep driver itself, on an explicit `device`. Separated
    from `vectorized_matchplay_worker` only so the CUDA failure path can
    re-run it from scratch on the CPU."""
    (state_dict, hidden_sizes, input_size, output_mode, encoder_name,
     mode, num_episodes, seed, oneply, boltzmann_temp, met_path,
     B, max_episode_plies, met_scopes) = (
        a.state_dict, a.hidden_sizes, a.input_size, a.output_mode,
        a.encoder_name, a.mode, a.num_episodes, a.seed, a.oneply,
        a.boltzmann_temp, a.met_path, a.B, a.max_episode_plies,
        a.met_scopes,
    )

    # Driver net on `device`: runs the single batched forward per step.
    net = TDNetwork(
        hidden_sizes=hidden_sizes, input_size=input_size,
        output_mode=output_mode, encoder_name=encoder_name,
        activation=a.activation,
    )
    net.load_state_dict(state_dict)
    net.to(device).eval()
    # BG_FAST_FORWARD (c_api/fast_forward.py): a CPU driver runs the
    # batched forward on a c_api backend (torch: bitwise the same).
    from c_api.fast_forward import FastForward, backend_from_env, supports
    _ff_backend = backend_from_env() if device == "cpu" else None
    fast = (FastForward(net, _ff_backend)
            if _ff_backend and supports(net) else None)

    # The AGENT is built on CPU so its move-gen / encode keep the bg_fast
    # C fast path (which requires a CPU agent). Its own net copy is only
    # used by `_eval_new_game` (see the module docstring); every other
    # forward is yielded to the driver net above. bf16 is ignored: no
    # matchplay forward routes through `_forward_inference`, so
    # --bf16-collect is a no-op on this mode either way.
    agent_net = TDNetwork(
        hidden_sizes=hidden_sizes, input_size=input_size,
        output_mode=output_mode, encoder_name=encoder_name,
        activation=a.activation,
    )
    agent_net.load_state_dict(state_dict)
    agent_net.eval()
    from trainer import _worker_met
    agent = TDAgent(
        agent_net, oneply=False, boltzmann_temp=boltzmann_temp,
        bf16_inference=False, device="cpu", met=_worker_met(met_path),
        met_scopes=met_scopes,
    )

    base_rng = random.Random(seed)
    enc_chunks: List[np.ndarray] = []
    tgt_chunks: List[np.ndarray] = []
    episodes_started = 0
    n_discarded = 0

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
        # Per-episode derived stream: lockstep interleaving must not
        # perturb an individual match's rng consumption.
        r = random.Random(base_rng.randint(0, 2**31 - 1))
        g = collect_episode_matchplay_gen(
            agent, mode, r, mode.initial_match_state(),
            oneply=oneply, max_episode_plies=max_episode_plies,
        )
        try:
            feat = next(g)
        except StopIteration as e:
            _store(e.value)
            return None
        return [g, feat]

    # Initial fill: keep starting games until B slots are LIVE or the
    # quota is exhausted -- not just B attempts. A game that finishes
    # without ever yielding (e.g. every episode discarded by a tiny
    # max_episode_plies) fills no slot; attempting only B times could
    # leave `slots` empty, skip the driver loop entirely and silently
    # return fewer episodes than requested.
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
            if fast is not None:
                out = fast(big)
            else:
                t = torch.from_numpy(np.ascontiguousarray(big))
                if device != "cpu":
                    t = t.to(device)
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
                        # else: episode ended with no forward; refill on
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
