"""
trainer.py -- Trainer class for backgammon self-play TD learning.

Trainer pairs an Agent with a GameMode and owns the optimizer and
training loop. The Agent owns its representation (forward pass,
target shape, loss).

Two methods:
  - `train`: round-based batch TD(0). Adam by default. Parallel
    collection via `workers=N`; optional GPU collector (see
    gpu_context.py).
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
import row_transport


# Safety bound on self-play episode length, in plies (turns).
DEFAULT_MAX_EPISODE_PLIES = 1500

# VRAM preflight for the per-context GPU collector (gpu_context.py),
# re-exported for callers and tests.
from gpu_context import (  # noqa: E402
    VEC_FALLBACK_PER_WORKER_BYTES as VEC_PER_WORKER_BYTES,
    VEC_RESERVE_BYTES,
    vectorized_vram_cap,
)


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
    samples.

    `max_episode_plies`: an episode exceeding it is DISCARDED (empty
    arrays returned), not truncated. A legitimate episode always emits
    at least one sample, so empty means discarded. <= 0 disables the
    bound.
    """
    initial_match_state = mode.initial_match_state()
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




# ── Parallel collection worker ────────────────────────────────────────


def _split_episodes(total: int, n_workers: int) -> List[int]:
    """Split *total* episodes as evenly as possible across n_workers."""
    base = total // n_workers
    remainder = total % n_workers
    return [base + (1 if i < remainder else 0) for i in range(n_workers)]


# ── C collector (--c-collect) ─────────────────────────────────────────



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
    (export_weights.export_live), and the libpuretd backend to open
    them with. `slab` (BG_COLLECT_SHM): where this worker writes its rows
    (row_transport.Slab); None returns them through the pool pipe."""
    path: str
    backend: str
    slab: Optional[object] = None


# The encoders bgc_collect writes rows for: the C API encoder table's
# entries with collect set (c_api/puretd.c BGC_BASES / BGC_CUBE_BLOCKS,
# read by puretd_py.encoder_info; tests/test_c_collect.py checks the two
# agree). Listed here so the preconditions need no library.
C_COLLECT_ENCODERS = ("perspective196", "cubeful_perspective196")


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


def c_collect_fallbacks(net, mode, oneply_acting=False, boltzmann_temp=0.0,
                        bf16_collect=False, c_engine=False) -> List[str]:
    """Why the C collector cannot collect this configuration: one reason
    per unmet precondition (experiments/c_collect/SPEC.md), [] when it
    can. Trainer.train prints each as a NOTE and keeps the Python
    collector."""
    reasons = []
    if boltzmann_temp != 0:
        reasons.append(
            f"--boltzmann-temp {boltzmann_temp:g} (the C collector plays "
            f"argmin; torch.multinomial sampling has no C twin)")
    if oneply_acting:
        reasons.append("--oneply-acting (the C collector plays at 0-ply)")
    if bf16_collect:
        reasons.append("--bf16-collect (a Python-collector forward option)")
    if c_engine:
        reasons.append("--c-engine (the Python collector's libbg_nply "
                       "1-ply target forward)")
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


class CCollectUnavailable(RuntimeError):
    """An explicit --c-collect whose library cannot be built or loaded."""


def c_collect_library(backend: str) -> Optional[str]:
    """None when libpuretd_<backend>.so loads, after building it on demand
    if it is missing or stale (puretd_py.ready_lib; one NOTE says it was
    built); else why not, for the caller's NOTE (the default collector
    falls back) or error (an explicit --c-collect)."""
    from c_api import puretd_py as P
    return P.ready_lib(backend)


def _c_collect(weights: _CWeights, mode, num_episodes, seed, oneply,
               max_episode_plies):
    """The _collect_worker episode loop in C: `num_episodes` episodes on
    one random.Random(seed) stream through bgc_collect. Same
    (enc, tgt, n_discarded) result; an empty one has shape
    (0, input_size). With a slab (BG_COLLECT_SHM) the rows go into
    shared memory and the result is a row_transport.SlabRows, unless
    they outgrew it. The engine is opened per call: bgc_open has no
    weight reload and the open costs < 0.5 ms
    (experiments/c_collect/results/p4_throughput.md)."""
    from c_api.puretd_py import Engine
    eng = Engine(weights.path, weights.backend)
    try:
        if eng.kind != _c_mode_kind(mode):
            raise RuntimeError(
                f"C collector: engine kind {eng.kind} of {weights.path} does "
                f"not play {type(mode).__name__}")
        kw = dict(oneply=oneply, jacoby=getattr(mode, "jacoby", True),
                  max_plies=max_episode_plies)
        if weights.slab is not None:
            return row_transport.collect_into_slab(
                eng, weights.slab, seed, num_episodes, **kw)
        return eng.collect(None, seed, num_episodes, **kw)
    finally:
        eng.close()


def _collect_worker(args):
    """Worker subprocess: rebuild the agent from a state_dict, play
    `num_episodes` episodes, return (encodings, targets, n_discarded).
    The GameMode instance is pickled into `args` (modes carry config,
    e.g. CubefulMoneyMode.jacoby). A _CWeights in the state_dict slot
    plays the episodes in C instead (--c-collect).

    The args tuple arity must stay in sync with `_build_worker_args`
    in Trainer.train.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    (state_dict, hidden_sizes, input_size, output_mode, encoder_name,
     activation, mode, num_episodes, seed, oneply, oneply_acting,
     boltzmann_temp, bf16_inference, c_engine, max_episode_plies) = args

    if isinstance(state_dict, _CWeights):
        return _c_collect(state_dict, mode, num_episodes, seed, oneply,
                          max_episode_plies)

    from model import TDNetwork
    from td_agent import TDAgent

    # `activation` must be passed: it is not a parameter tensor, so
    # load_state_dict cannot catch a mismatch.
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
        bf16_inference=bf16_inference,
    )

    # C engine: the 1-ply cubeful TARGET forward runs in libbg_nply.
    # Move selection stays PyTorch. A missing library must fail loudly.
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
            n_discarded,
        )
    else:
        result = (
            np.concatenate(enc_chunks, axis=0),
            np.concatenate(tgt_chunks, axis=0),
            n_discarded,
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
        c_engine: bool = False,
        max_episode_plies: int = DEFAULT_MAX_EPISODE_PLIES,
        vectorized_collect: bool = False,
        vectorized_batch: int = 32,
        vectorized_workers: Optional[int] = None,
        vectorized_shared_context: Optional[bool] = None,
        gpu_eval_servers: Optional[int] = None,
        vectorized_auto: bool = False,
        train_params: Optional[dict] = None,
        c_collect: bool = False,
        c_collect_backend: str = "simd",
        c_collect_auto: bool = False,
    ) -> List[float]:
        """Round-based batch TD(0). A round collects
        `episodes_per_round` episodes, then trains `epochs_per_round`
        shuffled passes over the round's pool in `batch_size` batches.

        `log_every` is in rounds. `workers > 1` collects in N
        subprocesses. The training step runs in the master.
        `oneply=True` uses 1-ply lookahead targets.

        `end_lr`: linearly anneal the LR from its starting value to
        `end_lr` over `num_episodes`.

        `warmup_cycles`: ramp the LR from `lr * 0.1` to the scheduled
        LR over the first N rounds. Multiplies with `end_lr`.

        `pipeline_collect`: overlap collection of round N+1 with the
        training step of round N. Workers then see one-round-stale
        weights. Worker seeds are unchanged. No effect with
        workers <= 1.

        `max_episode_plies`: episodes exceeding it are discarded and
        counted in the round log line.

        `vectorized_collect`: GPU collector. Each worker runs
        `vectorized_batch` games in lockstep and batches their net
        forwards into one GPU call. Requires oneply, workers > 1 and
        CUDA; ignored with a warning otherwise. Disables pipelining,
        the C engine and bf16 collection.
        `vectorized_workers`: exact GPU pool size (default: min(workers,
        physical cores[, gpu_context.VEC_MAX_DEFAULT_WORKERS for the
        per-worker topology]); the VRAM preflight can cap it lower).
        `vectorized_shared_context`: True = eval-server process(es)
        hold the only CUDA context(s), workers stay on the CPU; False =
        one context per worker; None = gpu_context default. Both
        collect identical data (tests/test_shared_context_parity.py).
        `gpu_eval_servers`: shared topology only; eval-server processes
        to shard the workers across (default one per
        gpu_context.VEC_WORKERS_PER_SERVER workers).
        `vectorized_auto`: True when auto-routed from
        `--oneply --device cuda`. Auto-routed runs fall back to the CPU
        collector when the preflight leaves fewer than
        gpu_context.VEC_MIN_WORKERS workers (or too little VRAM for the
        eval server). Explicit requests proceed and fail loudly.

        `train_params`: provenance dict stamped into periodic
        checkpoints (model.save). The final save is the caller's
        responsibility (train_batch.py stamps the same dict there).

        `c_collect`: collect in C (bgc_collect, c_api/puretd_collect.c)
        on the `c_collect_backend` libpuretd build. The master exports
        each round's weights once to a private dir. The workers play
        the same episodes the Python collector plays for their seed
        (experiments/c_collect/SPEC.md). Works with
        `pipeline_collect`, `save_every` and the pool restart. With
        workers <= 1 the master collects one worker's share on a seed
        drawn from the run RNG (the Python single-process path plays on
        the run RNG itself, so the episodes differ). Falls back to the
        Python collector, with one NOTE per reason, for
        c_collect_fallbacks(); an error with `vectorized_collect`. A
        missing libpuretd_<backend>.so is built on demand
        (c_collect_library). `train_params["c_collect"]`, when present,
        is set to whether the C collector actually collects.
        `c_collect_auto`: True when `c_collect` is train_batch.py's
        default rather than an explicit --c-collect; a library that
        cannot be built or loaded then falls back to the Python
        collector with a NOTE instead of raising.

        `BG_TRAIN_CUDA_GRAPH` (cuda_graph_step.py; unset = on on a cuda
        device, 0 = the eager step) and `BG_COLLECT_SHM`
        (row_transport.py; unset = on with the C collector and a worker
        pool, 0 = the pool pipe): see the blocks that read them.

        Returns a flat list of per-batch losses.
        """
        mode.validate_agent(self.agent)
        if vectorized_collect and oneply_acting:
            # The vectorized collector only does 0-ply move selection.
            # Ignoring the flag would change the data distribution.
            raise ValueError(
                "--vectorized-collect does not support --oneply-acting "
                "(the vectorized collector does 1-ply TARGETS with 0-ply "
                "move selection). Drop one of the two flags."
            )
        if c_collect and vectorized_collect:
            raise ValueError(
                "--c-collect and --vectorized-collect are mutually exclusive "
                "(two collectors for the same rounds). Drop one of the two "
                "flags."
            )
        if isinstance(self.optimizer, torch.optim.SGD):
            import warnings
            warnings.warn(
                "SGD with batch TD may be unstable, Adam recommended.",
                stacklevel=2,
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

        # One worker pool for the whole run; None means single-process.
        # "spawn", not "fork": forked workers inherit the master's RSS.
        # TRAINER_MAXTASKSPERCHILD=N recycles workers after N calls.
        _mt = os.environ.get("TRAINER_MAXTASKSPERCHILD")
        maxtasks = int(_mt) if _mt else None
        # BG_COLLECT_DUMP=<dir>: save each round's (enc, tgt) pool for
        # collector-equivalence checks, with the rows and the seed of each
        # worker (seeds: CPU collectors only, else empty).
        _dump_dir = os.environ.get("BG_COLLECT_DUMP") or None
        if _dump_dir:
            os.makedirs(_dump_dir, exist_ok=True)

        # BG_COLLECT_SHM / BG_TRAIN_CUDA_GRAPH, read (and validated)
        # before any process starts.
        shm_env = row_transport.enabled()
        graph_env = mode_from_env()

        # C collector: preconditions, the library (built on demand), then
        # one probe open (a kind mismatch fails here, before the pool
        # starts).
        c_active = False
        c_dir = None
        c_files = {}            # round index -> exported weights file
        c_label = "--c-collect (default)" if c_collect_auto else "--c-collect"
        if c_collect:
            reasons = c_collect_fallbacks(
                self.agent.network, mode, oneply_acting=oneply_acting,
                boltzmann_temp=boltzmann_temp, bf16_collect=bf16_collect,
                c_engine=c_engine)
            for why in reasons:
                print(f"  NOTE: {c_label}: {why}; falling back to the "
                      f"Python collector.", flush=True)
            c_active = not reasons
        if c_active:
            why = c_collect_library(c_collect_backend)
            if why is not None:
                if not c_collect_auto:
                    raise CCollectUnavailable(f"--c-collect: {why}")
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
                net_enc = getattr(self.agent.network, "encoder_name", "perspective196")
                if kind != _c_mode_kind(mode) or n_in != n_net or c_enc != net_enc:
                    raise RuntimeError(
                        f"C collector: the export opens as kind {kind} with "
                        f"{n_in} inputs, encoder {c_enc}; {type(mode).__name__} "
                        f"with this {n_net}-input {net_enc} net needs kind "
                        f"{_c_mode_kind(mode)}")
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
                  f"once per round to {c_dir})", flush=True)
        if metrics_out is not None:
            metrics_out["c_collect"] = c_active
        if train_params is not None and "c_collect" in train_params:
            # Provenance: what collected, not what was asked for.
            train_params["c_collect"] = c_active

        # Eligibility and the VRAM preflight run BEFORE pool creation:
        # the pool is sized to the capped worker count.
        vec_active = bool(
            vectorized_collect and workers > 1 and oneply
            and torch.cuda.is_available()
        )
        vec_workers = workers
        vec_shared = False
        eval_server = None
        _new_server = None
        if vec_active:
            import gpu_context as _gc
            if vectorized_shared_context is None:
                vectorized_shared_context = _gc.VEC_SHARED_CONTEXT_DEFAULT
            # Allocator config for the spawn children. Before the probe,
            # so it measures what the workers will use.
            if _gc.set_allocator_env():
                print(f"  NOTE: PYTORCH_CUDA_ALLOC_CONF={_gc.VEC_ALLOC_CONF} "
                      f"set for the GPU collection workers (inherited by "
                      f"the spawn pool).", flush=True)
            # Worker count: min(workers, physical cores, hard cap)
            # unless --vectorized-workers was given.
            phys = _gc.physical_cores()
            vec_workers, why = _gc.default_vectorized_workers(
                workers, vectorized_workers, phys,
                hard_cap=(0 if vectorized_shared_context
                          else _gc.VEC_MAX_DEFAULT_WORKERS),
            )
            if vec_workers != workers:
                print(
                    f"  NOTE: GPU collection workers {workers} -> "
                    f"{vec_workers} ({why}). 1-ply collection throughput "
                    f"is flat past this point"
                    + ("" if vectorized_shared_context else
                       " and every extra worker is another CUDA context")
                    + "; pass --vectorized-workers N to override.",
                    flush=True,
                )
            vec_shared = bool(vectorized_shared_context)
            free_b, total_b = torch.cuda.mem_get_info()
            if vec_shared:
                # Eval-server VRAM floor: auto-routed runs below it fall
                # back to the CPU collector, explicit ones proceed.
                need_b = _gc.shared_context_min_free_bytes(vectorized_batch)
                if free_b < need_b:
                    if vectorized_auto:
                        print(
                            f"  WARN: vectorized GPU collector disabled (VRAM "
                            f"preflight: {free_b / 2**20:.0f} MB free, the "
                            f"shared-context eval server needs "
                            f"~{need_b / 2**20:.0f} MB for B={vectorized_batch}) "
                            f"-- falling back to the CPU collector; the master "
                            f"training step stays on its device. Consider "
                            f"--device cpu --pipeline-collect on this card, or a "
                            f"smaller --vectorized-batch.", flush=True,
                        )
                        vec_active = False
                        vec_shared = False
                        vec_workers = 0
                    else:
                        print(
                            f"  NOTE: {free_b / 2**20:.0f} MB free is below the "
                            f"~{need_b / 2**20:.0f} MB the shared-context eval "
                            f"server wants for B={vectorized_batch}; proceeding "
                            f"because --vectorized-collect was explicit (the "
                            f"server start fails loudly if it does not fit).",
                            flush=True,
                        )
            else:
                # Per-context collector: probe one worker's VRAM and size
                # the pool to what fits.
                sd0 = {k: v.detach().cpu()
                       for k, v in self.agent.network.state_dict().items()}
                _n0 = self.agent.network
                spec0 = {
                    "net_type": "td",
                    "hidden_sizes": list(_n0.hidden_sizes),
                    "input_size": getattr(_n0, "input_size", 196),
                    "output_mode": getattr(_n0, "output_mode", "probability"),
                    "encoder_name": getattr(_n0, "encoder_name", "perspective196"),
                    "activation": getattr(_n0, "activation", "relu"),
                }
                rows = _gc.probe_rows(vectorized_batch)
                measured = _gc.probe_worker_vram(spec0, sd0, rows)
                print("  " + _gc.describe_probe(measured, rows), flush=True)
                per_worker = (_gc.with_probe_margin(measured)
                              if measured is not None else None)
                capped = vectorized_vram_cap(vec_workers, free_b, per_worker)
                # Floor: VEC_MIN_WORKERS (auto-routed) or 2 (explicit);
                # applies only when VRAM did the capping.
                floor = _gc.VEC_MIN_WORKERS if vectorized_auto else 2
                vram_bound = capped < vec_workers
                if capped < 2 or (vram_bound and capped < floor):
                    print(
                        f"  WARN: vectorized GPU collector disabled (VRAM "
                        f"preflight: {free_b / 2**20:.0f} MB free supports "
                        f"{capped} worker context(s), need >= {floor}"
                        + ("" if vectorized_auto else " for an explicit request")
                        + f") -- falling back to the CPU collector; the master "
                        f"training step stays on its device. On this card use "
                        f"the shared-context collector (the default, ~1-2 GB) "
                        f"or --device cpu --pipeline-collect.", flush=True,
                    )
                    vec_active = False
                    vec_workers = capped
                    # vec_workers != workers keeps the generic IGNORED
                    # print below quiet.
                elif capped < vec_workers:
                    print(
                        f"  NOTE: VRAM preflight capped GPU collection "
                        f"workers {vec_workers} -> {capped} "
                        f"({free_b / 2**20:.0f} MB free, "
                        f"{(per_worker or VEC_PER_WORKER_BYTES) / 2**20:.0f} "
                        f"MB/worker budget + "
                        f"{VEC_RESERVE_BYTES / 2**20:.0f} MB reserve). "
                        f"Pool is sized to the cap.", flush=True,
                    )
                    vec_workers = capped

        # Signal handlers installed by this call (shared-context path),
        # restored in the finally.
        _prev_handlers = {}
        if vec_active and vec_shared:
            # Eval-server process(es) own the net on the GPU. Each worker
            # claims its shared-memory slot in the pool initializer, so
            # the pool must not recycle processes.
            from gpu_eval_server import EvalServerGroup
            from vectorized_collect import (
                remote_worker_init, remote_max_rows,
            )
            if maxtasks is not None:
                print("  NOTE: TRAINER_MAXTASKSPERCHILD ignored: the "
                      "shared-context collector pins one server slot per "
                      "worker process for the life of the pool.",
                      flush=True)
                maxtasks = None
            _sn = self.agent.network
            spec = {
                "net_type": "td",
                "hidden_sizes": list(_sn.hidden_sizes),
                "input_size": getattr(_sn, "input_size", 196),
                "output_mode": getattr(_sn, "output_mode", "probability"),
                "encoder_name": getattr(_sn, "encoder_name", "perspective196"),
                "activation": getattr(_sn, "activation", "relu"),
            }
            n_srv = _gc.default_eval_servers(vec_workers, gpu_eval_servers)

            def _new_server():
                # Factory for pool_guard recovery: a rebuilt server starts
                # from the CURRENT weights.
                srv = EvalServerGroup(
                    mp.get_context("spawn"),
                    num_slots=vec_workers,
                    n_servers=n_srv,
                    n_feat=spec["input_size"],
                    out_dim=1,
                    max_rows=remote_max_rows(vectorized_batch),
                    build_spec=spec,
                    init_state_dict=_sn.state_dict(),
                    device="cuda",
                    coalesce=False,
                )
                srv.start()
                return srv

            eval_server = _new_server()
            print(f"  GPU eval server{'s' if eval_server.n_servers > 1 else ''}: "
                  f"{eval_server.n_servers} process(es) x "
                  f"{remote_max_rows(vectorized_batch)} rows/slot, fp32 cuda, "
                  f"one forward per request", flush=True)
            def _new_pool(srv):
                # A dead worker's replacement blocks in remote_worker_init
                # on its slot. PoolWatch turns that into WorkerDied.
                return GuardedPool(
                    context=mp.get_context("spawn"),
                    processes=vec_workers,
                    initializer=remote_worker_init,
                    initargs=(srv.client_handles(),),
                )

            pool = _new_pool(eval_server)
            # SIGTERM runs neither `finally` nor atexit, which would leave
            # the server alive holding VRAM. Order matters: pool, then
            # server (else workers block on a dead server's semaphores),
            # then os._exit so the finally cannot re-join a dead pool.
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
                    _prev_handlers[_sig] = _signal.signal(_sig, _shutdown)
                except (ValueError, OSError):
                    pass    # not on the main thread: only the finally below
        else:
            def _new_pool(srv):
                return (
                    GuardedPool(
                        context=mp.get_context("spawn"),
                        processes=(vec_workers if vec_active else workers),
                        maxtasksperchild=maxtasks,
                    )
                    if workers > 1 else None
                )

            pool = _new_pool(None)

        # Dead-worker detection and recovery (pool_guard). `pool` and
        # `eval_server` are rebound on a restart. `_shutdown` reads them
        # at call time, so it always tears down the live ones.
        watch = PoolWatch(pool, eval_server) if pool is not None else None
        pool_restarts = 0
        restart_limit = max_pool_restarts()

        # ── Vectorized GPU collection ──────────────────────────────────
        # Everything this path turns off is announced: dropping
        # pipelining changes the recipe (stale -> fresh targets).
        if vectorized_collect and not vec_active and vec_workers == workers:
            # VRAM-cap fallbacks printed their own WARN above.
            reasons = []
            if not oneply:
                reasons.append("needs --oneply")
            if workers <= 1:
                reasons.append("needs --workers > 1")
            if not torch.cuda.is_available():
                reasons.append("CUDA not available")
            print(
                "  WARN: --vectorized-collect IGNORED ("
                + "; ".join(reasons) + ") -- falling back to the CPU "
                "collector",
                flush=True,
            )
        if vec_active:
            print(
                f"Vectorized GPU collection: ON ({vec_workers} procs x "
                f"B={vectorized_batch} games/proc, fp32 on cuda, "
                f"max_episode_plies={max_episode_plies}, "
                + (f"shared context: {eval_server.n_servers} eval-server "
                   f"process(es) hold the only CUDA context(s), workers "
                   f"stay on the CPU"
                   if vec_shared else
                   f"per-worker context: {vec_workers} CUDA contexts")
                + ")",
                flush=True,
            )
            # Workers rebuild the net from these fields. Print exactly
            # what they will construct.
            _sn = self.agent.network
            print(
                f"  worker net spec: hidden={list(_sn.hidden_sizes)} "
                f"input={getattr(_sn, 'input_size', '?')} "
                f"activation={getattr(_sn, 'activation', 'relu')} "
                f"output_mode={getattr(_sn, 'output_mode', '?')} "
                f"encoder={getattr(_sn, 'encoder_name', '?')}",
                flush=True,
            )
            if pipeline_collect:
                print(
                    "  NOTE: --pipeline-collect DISABLED by "
                    "--vectorized-collect. The vectorized worker owns a "
                    "CUDA context and its forwards would contend with "
                    "the master's train step on the same GPU. RECIPE "
                    "EFFECT: collectors now see the CURRENT round's "
                    "weights instead of one-round-stale weights.",
                    flush=True,
                )
            if c_engine:
                print(
                    "  NOTE: --c-engine DISABLED by "
                    "--vectorized-collect. The vectorized collector "
                    "computes 1-ply targets by yielding feature "
                    "matrices to the GPU driver net; the C 1-ply engine "
                    "is a different (CPU) implementation and the two "
                    "cannot both own the 1-ply computation.",
                    flush=True,
                )
            if bf16_collect:
                print(
                    "  NOTE: --bf16-collect DISABLED by "
                    "--vectorized-collect (driver net runs fp32 on "
                    "cuda). Targets are fp32, i.e. MORE precise than "
                    "the bf16 CPU collector, not less.",
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
            its workers. A retried round (pool restart) replaces its file
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

        # Worker seeds by round index, for BG_COLLECT_DUMP (popped when
        # the round is collected).
        round_seeds = {}

        # Worker args for one round, with the current weights.
        def _build_worker_args(round_size: int, round_idx: int):
            splits = _split_episodes(round_size, workers)
            if c_active:
                state_dict = _CWeights(_c_export(round_idx), c_collect_backend)
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
            # BG_COLLECT_SHM: each C worker gets its own slab.
            per_worker = ([state_dict._replace(slab=sl)
                           for sl in shm.slabs(round_idx, splits)]
                          if shm is not None else [state_dict] * len(splits))
            return [
                (w, hidden_sizes, input_size, output_mode,
                 encoder_name, activation,
                 mode, n_eps, seed,
                 oneply, oneply_acting, boltzmann_temp,
                 bf16_collect, c_engine, max_episode_plies)
                for w, n_eps, seed in zip(per_worker, splits, seeds)
            ]

        # Args for the vectorized (GPU) workers. Arity must stay in sync
        # with `vectorized_collect_worker` in vectorized_collect.py.
        def _build_vec_args(round_size: int):
            splits = _split_episodes(round_size, vec_workers)
            # .clone() REQUIRED: see _build_worker_args.
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
                 vectorized_batch, ("remote" if vec_shared else "cuda"),
                 max_episode_plies)
                for n_eps in splits
            ]

        # BG_ROUND_PHASES=1: one line per round with the master's time by
        # phase, in seconds (experiments/c_collect/results/
        # p8_profile.md); also kept in metrics_out["round_phases"].
        phases_on = os.environ.get("BG_ROUND_PHASES") == "1"
        ph = {}

        # Pipeline pre-roll: start round 0's collect before the loop.
        use_pipeline = pipeline_collect and pool is not None and not vec_active
        if use_pipeline:
            print("  pipelined collection: ON (round N+1 collects during "
                  "round N's training step)", flush=True)

        # Row transport, C collector (row_transport.py): shared-memory
        # arenas (one per round in flight; BG_COLLECT_SHM unset or 1, the
        # default), or the pool pipe (BG_COLLECT_SHM=0).
        shm = None
        if c_active and pool is not None and shm_env:
            shm = row_transport.ShmTransport(
                input_size, slots=2 if use_pipeline else 1)
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
        pending = None
        pending_round_size = 0
        if use_pipeline:
            pending_round_size = min(episodes_per_round, num_episodes)
            pending = pool.map_async(
                _collect_worker, _build_worker_args(pending_round_size, 0),
            )

        def _collect_round(round_size, round_idx, pending):
            """One round's collection -> (enc_chunks, tgt_chunks,
            n_discarded, pending, worker_rows). Raises WorkerDied
            (pool_guard) instead of hanging on a dead worker. The caller
            rebuilds the pool and calls it again. worker_rows: the rows
            of each worker's result, in worker order (BG_COLLECT_DUMP)."""
            enc_chunks: List[np.ndarray] = []
            tgt_chunks: List[np.ndarray] = []
            worker_rows: List[int] = []

            def _take(results, round_idx=None) -> int:
                # Keep each worker's (enc, tgt) chunk; count discards.
                # BG_COLLECT_SHM: a chunk is a view of its slab.
                discarded = 0
                slabs = (shm.begin_take(round_idx) if shm is not None
                         and round_idx is not None else None)
                for i, res in enumerate(results):
                    w_enc, w_tgt, w_disc = (res if slabs is None
                                            else shm.take(slabs, i, res))
                    worker_rows.append(len(w_enc))
                    if len(w_enc) > 0:
                        enc_chunks.append(w_enc)
                        tgt_chunks.append(w_tgt)
                    discarded += w_disc
                return discarded

            n_discarded = 0
            if vec_active:
                # Same (enc, tgt, n_discarded) result as
                # _collect_worker.
                from vectorized_collect import vectorized_collect_worker
                if eval_server is not None:
                    # Safe between rounds: no request is in flight.
                    watch.push_weights(net.state_dict())
                n_discarded = _take(watch.map(
                    vectorized_collect_worker,
                    _build_vec_args(round_size),
                ))
            elif pool is None and c_active:
                # Single-process C: one worker's share on a seed drawn
                # from the run RNG.
                weights = _CWeights(_c_export(round_idx), c_collect_backend)
                round_seeds[round_idx] = [rng.randint(0, 2**31 - 1)]
                n_discarded = _take([_c_collect(
                    weights, mode, round_size, round_seeds[round_idx][0],
                    oneply, max_episode_plies,
                )])
            elif pool is None:
                # Single-process: the master agent collects, so re-sync
                # its bf16 copy and the C engine's weights each round.
                if bf16_collect:
                    if not self.agent.bf16_inference:
                        self.agent.bf16_inference = True
                    self.agent.refresh_bf16_inference()
                if c_engine and oneply:
                    import c_engine.bg_nply as _cn
                    if not _cn.enable_c_engine(self.agent):
                        raise RuntimeError(
                            "--c-engine requested but libbg_nply is "
                            "unavailable; build with "
                            "`bash c_engine/build_nply.sh`"
                        )
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
                next_round_size = 0
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
                n_discarded = _take(watch.get(pending), round_idx)
                pending = next_pending
            else:
                # Parallel, synchronous.
                n_discarded = _take(watch.map(
                    _collect_worker, _build_worker_args(round_size, round_idx),
                ), round_idx)

            return enc_chunks, tgt_chunks, n_discarded, pending, worker_rows

        # BG_TRAIN_CUDA_GRAPH (unset or 1 = on, the default; 0 = off):
        # replay full batches of the train step from a CUDA graph
        # (cuda_graph_step.py). Bitwise equal to train_step_dev; implies
        # the on-device pool.
        graph_step = None
        _graph_mode = graph_env
        if _graph_mode is not None:
            on_cuda = str(self.agent.device).startswith("cuda")
            if not on_cuda:
                _graph_mode, why = None, f"device is {self.agent.device}"
            else:
                _graph_mode, why = resolve_graph_mode(self, _graph_mode)
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

        try:
            num_rounds = math.ceil(num_episodes / episodes_per_round)
            episodes_played = 0
            for round_idx in range(num_rounds):
                round_size = min(
                    episodes_per_round, num_episodes - episodes_played,
                )
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
                ph.clear()
                _t_collect_start = time.perf_counter()
                while True:
                    try:
                        (enc_chunks, tgt_chunks, n_discarded,
                         pending, worker_rows) = _collect_round(
                            round_size, round_idx, pending)
                        break
                    except WorkerDied as err:
                        # Rebuild the pool (and servers, from the current
                        # weights) and redo the round.
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
                _collect_dt = time.perf_counter() - _t_collect_start
                if metrics_out is not None:
                    metrics_out.setdefault("collection_times", []).append(
                        time.perf_counter() - _t_collect_start
                    )
                    metrics_out.setdefault("round_sizes", []).append(round_size)
                    metrics_out.setdefault("discarded", []).append(n_discarded)
                if not enc_chunks:
                    # Nothing usable this round; say so if every episode
                    # ran away.
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

                # BG_POOL_ON_DEVICE=1 (experimental): keep the pool on the
                # training device and index it there. Same ops in the
                # same order: bitwise identical checkpoints
                # (tests/test_train_step_bitwise.py).
                pool_on_dev = ((os.environ.get("BG_POOL_ON_DEVICE") == "1"
                                or graph_step is not None)
                               and str(self.agent.device).startswith("cuda"))
                # BG_COLLECT_SHM with the pool on the device: the chunks
                # are slab views, copied to the device one by one below
                # (no host concatenate unless the round is dumped).
                shm_direct = shm is not None and pool_on_dev
                n = sum(len(t) for t in tgt_chunks)
                _t = time.perf_counter()
                if shm_direct and not _dump_dir:
                    pool_enc = pool_tgt = None
                else:
                    pool_enc = np.concatenate(enc_chunks, axis=0)
                    pool_tgt = np.concatenate(tgt_chunks, axis=0)
                ph["assemble"] = time.perf_counter() - _t
                _t = time.perf_counter()
                if _dump_dir:
                    # For elementwise collector comparison
                    # (experiments/gpu_1ply_bench/oneply_gpu_context_bench.py
                    # --validate).
                    np.savez(os.path.join(_dump_dir, f"round_{round_idx + 1:04d}.npz"),
                             enc=pool_enc, tgt=pool_tgt,
                             worker_rows=np.asarray(worker_rows, dtype=np.int64),
                             worker_seeds=np.asarray(seeds_used, dtype=np.int64))
                ph["dump"] = time.perf_counter() - _t
                _t_train_start = time.perf_counter()
                # Release the chunks: pool_enc/pool_tgt own a copy, and
                # keeping both doubles the round's footprint. (shm_direct:
                # slab views, consumed by the device copy just below; none
                # may outlive the round.)
                slab_chunks = (enc_chunks, tgt_chunks) if shm_direct else None
                enc_chunks = None
                tgt_chunks = None

                # Fancy-index per batch (a materialized `pool_enc[perm]`
                # would double peak RSS). Both paths read the losses back
                # once per pass, not per batch.
                if pool_on_dev:
                    _dev = self.agent.device
                    if shm_direct:
                        pool_enc_t, pool_tgt_t = shm.to_device(
                            *slab_chunks, _dev)
                        slab_chunks = None
                    else:
                        pool_enc_t = torch.from_numpy(pool_enc).to(_dev, dtype=torch.float32)
                        pool_tgt_t = torch.as_tensor(pool_tgt, dtype=torch.float32).to(_dev)
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
                    if pool_on_dev:
                        round_losses = []
                        for start in range(0, n, batch_size):
                            idx_t = perm_t[start:start + batch_size]
                            if graph_step is not None:
                                round_losses.append(graph_step.step(
                                    pool_enc_t, pool_tgt_t, idx_t))
                                continue
                            round_losses.append(self.train_step_dev(
                                pool_enc_t[idx_t], pool_tgt_t[idx_t]))
                        losses.extend(torch.stack(round_losses).tolist())
                        ph["steps"] += time.perf_counter() - _t_steps
                        continue
                    round_losses = []
                    for start in range(0, n, batch_size):
                        idx = perm[start:start + batch_size]
                        batch_e = pool_enc[idx]
                        batch_t = pool_tgt[idx]
                        round_losses.append(
                            self.train_step_nosync(batch_e, batch_t))
                    losses.extend(torch.stack(round_losses).tolist())
                    ph["steps"] += time.perf_counter() - _t_steps

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
                        f"| recent batch loss {avg:.4f}"
                        f"{discard_note}"
                        + (f" | collect {_collect_dt:.2f}s train "
                           f"{time.perf_counter() - _t_train_start:.2f}s"
                           if os.environ.get("BG_ROUND_TIMING") else "")
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
                        f"{root}_ep{episodes_played}{ext or '.pt'}")
                    self.agent.network.save(
                        ckpt_path, train_params=train_params)
                    print(f"  -> checkpoint saved: {ckpt_path}", flush=True)
                    while next_save <= episodes_played:
                        next_save += save_every

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
                # After the pool, so no worker waits on a dead server.
                eval_server.stop()
            if c_dir is not None:
                # After the pool: no worker reads a weights file any more.
                shutil.rmtree(c_dir, ignore_errors=True)
            if shm is not None:
                # After the pool: no worker writes a slab any more.
                shm.close()
            if _prev_handlers:
                import signal as _signal
                for _sig, _h in _prev_handlers.items():
                    try:
                        _signal.signal(_sig, _h)
                    except (ValueError, OSError):
                        pass

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
