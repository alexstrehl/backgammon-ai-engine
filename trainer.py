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
    max_episode_plies: int = DEFAULT_MAX_EPISODE_PLIES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Play one self-play episode and return (encodings, targets).

    Encodings are computed once per turn via `agent.encode_state` so
    the training pass can skip re-encoding. Cubeless move selection
    is always 0-ply; cubeful respects agent.oneply.

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

    `max_episode_plies`: safety bound (see DEFAULT_MAX_EPISODE_PLIES).
    An episode that exceeds it is DISCARDED -- the function returns
    empty arrays. Discarding rather than truncating is deliberate: a
    runaway episode's samples come from a degenerate region of state
    space and would poison the pool, and truncate-and-bootstrap would
    change TD semantics. Empty arrays are the discard signal; a
    legitimate episode always emits at least one sample, so callers can
    treat "empty" as "discarded". Pass <= 0 to disable the bound.
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
        if not is_opening and match_state.can_offer(player):
            pre_persp = cube_perspective(match_state.cube_owner, player)

            # Emit cube-phase sample: is_cube_action=True
            enc_list.append(agent._encode_cubeful(
                state, pre_persp, is_cube_action=True,
            ))

            if oneply:
                # Cube targets are direct network evaluations. A cube
                # action has no chance node (the transition is
                # deterministic — no dice roll to average over), so the
                # network's own value IS the exact one-step backup; the
                # 1-ply machinery only applies to the checker phase.
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


def _collect_worker(args):
    """Worker subprocess: rebuild the agent from a state_dict, play
    `num_episodes` episodes, return
    (encodings, targets, n_discarded) — the concatenated numpy arrays
    plus the count of episodes discarded for exceeding
    `max_episode_plies`. The GameMode instance is pickled directly into
    `args` — simpler and more general than name-based dispatch,
    because modes can carry config (e.g. CubefulMoneyMode.jacoby).

    NOTE: the args tuple arity must stay in sync with
    `_build_worker_args` in Trainer.train.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    (state_dict, hidden_sizes, input_size, output_mode, encoder_name,
     activation, mode, num_episodes, seed, oneply, oneply_acting,
     boltzmann_temp, bf16_inference, c_engine, max_episode_plies) = args

    from model import TDNetwork
    from td_agent import TDAgent

    # `activation` MUST be threaded through: it is not a parameter
    # tensor, so load_state_dict succeeds even when the rebuilt net's
    # activation is wrong — a tanh/sigmoid checkpoint would silently
    # train against relu-computed targets.
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
        pipeline_collect: bool = False,
        c_engine: bool = False,
        max_episode_plies: int = DEFAULT_MAX_EPISODE_PLIES,
        vectorized_collect: bool = False,
        vectorized_batch: int = 32,
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

        `max_episode_plies`: safety bound on self-play episode length
        (see DEFAULT_MAX_EPISODE_PLIES). Episodes exceeding it are
        discarded and counted; the count appears in the round log line
        so a diverging run becomes visible rather than silently
        stalling. Applies to every collection path (single-process,
        parallel, pipelined, and the vectorized GPU collector).

        `vectorized_collect`: opt-in GPU collector. Each of `workers`
        processes runs `vectorized_batch` games in lockstep as Python
        generators, batching every pending network forward across those
        games into ONE local GPU call (no IPC). Requires oneply,
        workers > 1 and CUDA; ignored (with a warning) otherwise. It
        replaces the worker function only -- round structure, weight
        shipping, LR schedule, discard accounting and the training step
        are unchanged. It is MUTUALLY EXCLUSIVE with pipelining and
        with the C 1-ply engine; both are disabled with an explicit log
        line rather than silently (see below).

        Returns a flat list of per-batch losses.
        """
        mode.validate_agent(self.agent)
        if vectorized_collect and oneply_acting:
            # Checked before any pool exists. The vectorized generators
            # implement 0-ply move SELECTION only; silently ignoring
            # --oneply-acting would play a different policy than asked
            # for and change the data distribution, so refuse outright.
            raise ValueError(
                "--vectorized-collect does not support --oneply-acting "
                "(the vectorized collector does 1-ply TARGETS with 0-ply "
                "move selection). Drop one of the two flags."
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
        # With spawn, recycled workers start pristine, so the knob
        # meaningfully bounds per-worker growth.
        _mt = os.environ.get("TRAINER_MAXTASKSPERCHILD")
        maxtasks = int(_mt) if _mt else None

        # Vectorized-eligibility and the VRAM preflight run BEFORE pool
        # creation: pool tasks rotate across processes, so a per-round
        # task cap would NOT cap CUDA contexts (each process that ever
        # ran a vectorized task keeps its context for life). The pool
        # itself is sized to the capped worker count.
        vec_active = bool(
            vectorized_collect and workers > 1 and oneply
            and torch.cuda.is_available()
        )
        vec_workers = workers
        if vec_active:
            free_b, total_b = torch.cuda.mem_get_info()
            vec_workers = vectorized_vram_cap(workers, free_b)
            if vec_workers < 2:
                print(
                    f"  WARN: --vectorized-collect IGNORED (VRAM preflight: "
                    f"{free_b / 2**20:.0f} MB free supports "
                    f"{vec_workers} worker context(s), need >= 2) -- "
                    f"falling back to the CPU collector", flush=True,
                )
                vec_active = False
                # vec_workers stays < 2: the generic IGNORED print below
                # is guarded on vec_workers == workers so the VRAM WARN
                # above is the only message for this case.
            elif vec_workers < workers:
                print(
                    f"  NOTE: VRAM preflight capped vectorized workers "
                    f"{workers} -> {vec_workers} "
                    f"({free_b / 2**20:.0f} MB free, "
                    f"~{VEC_PER_WORKER_BYTES / 2**20:.0f} MB/context + "
                    f"{VEC_RESERVE_BYTES / 2**20:.0f} MB reserve). "
                    f"Pool is sized to the cap.", flush=True,
                )

        pool = (
            mp.get_context("spawn").Pool(
                processes=(vec_workers if vec_active else workers),
                maxtasksperchild=maxtasks,
            )
            if workers > 1 else None
        )

        # ── Vectorized GPU collection (opt-in) ─────────────────────────
        # Each worker runs `vectorized_batch` games in lockstep and does
        # ONE batched GPU forward per step across all of them. Needs
        # oneply (the collector is 1-ply-only by construction), a real
        # worker pool, and CUDA.
        #
        # Everything this path turns off is announced — silently
        # dropping pipelining changes the recipe (one-round-stale
        # targets -> fresh targets), which must be visible in logs.
        if vectorized_collect and not vec_active and vec_workers == workers:
            # (VRAM-cap fallback already printed its own WARN above.)
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
                f"max_episode_plies={max_episode_plies})",
                flush=True,
            )
            # Spec echo: workers REBUILD the net from these fields, so
            # print exactly what they will construct. A field missing
            # here defaults silently in the worker.
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
                 bf16_collect, c_engine, max_episode_plies)
                for n_eps in splits
            ]

        # Helper: build args for the vectorized (GPU) workers. Same
        # leading fields as _build_worker_args, then the per-process
        # game batch B and the driver device in place of the c_engine
        # flag (which the vectorized path cannot use, see above), and
        # the ply bound last.
        #
        # NOTE: arity must stay in sync with
        # `vectorized_collect_worker` in vectorized_collect.py.
        def _build_vec_args(round_size: int):
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
        use_pipeline = pipeline_collect and pool is not None and not vec_active
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
                    # Vectorized GPU collection: each worker drives
                    # `vectorized_batch` games in lockstep and batches
                    # their forwards into one local GPU call. Returns
                    # the same (enc, tgt, n_discarded) 3-tuple as
                    # _collect_worker, so accounting below is shared.
                    from vectorized_collect import vectorized_collect_worker
                    for w_enc, w_tgt, w_disc in pool.map(
                        vectorized_collect_worker,
                        _build_vec_args(round_size),
                    ):
                        if len(w_enc) > 0:
                            enc_chunks.append(w_enc)
                            tgt_chunks.append(w_tgt)
                        n_discarded += w_disc
                elif pool is None:
                    # Single-process: master agent does the collection,
                    # so its bf16 inference copy (if enabled) must be
                    # re-synced to the current fp32 weights each round.
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
                            max_episode_plies=max_episode_plies,
                        )
                        if len(encs) > 0:
                            enc_chunks.append(encs)
                            tgt_chunks.append(tgts)
                        else:
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
                    for w_enc, w_tgt, w_disc in pending.get():
                        if len(w_enc) > 0:
                            enc_chunks.append(w_enc)
                            tgt_chunks.append(w_tgt)
                        n_discarded += w_disc
                    pending = next_pending
                    pending_round_size = next_round_size
                else:
                    # Non-pipelined parallel: submit and wait synchronously.
                    for w_enc, w_tgt, w_disc in pool.map(
                        _collect_worker, _build_worker_args(round_size),
                    ):
                        if len(w_enc) > 0:
                            enc_chunks.append(w_enc)
                            tgt_chunks.append(w_tgt)
                        n_discarded += w_disc

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
                        f"| recent batch loss {avg:.4f}"
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

        finally:
            if pool is not None:
                pool.close()
                pool.join()

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
