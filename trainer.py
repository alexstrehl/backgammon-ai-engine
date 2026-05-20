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
from modes import GameMode


# ── Episode collection (mode-agnostic, agent-driven) ──────────────────


def collect_episode(
    agent: Agent,
    mode: GameMode,
    rng: random.Random,
    oneply: bool = False,
    cube_targets_1ply: bool = False,
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
    """
    initial_match_state = mode.initial_match_state()
    if initial_match_state is not None:
        return _collect_episode_cubeful_money(
            agent, mode, rng, initial_match_state,
            oneply=oneply, cube_targets_1ply=cube_targets_1ply,
        )

    state, dice = opening_roll(rng)
    enc_list: List[np.ndarray] = []
    targets: List[float] = []

    while not mode.is_episode_over(state):
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
    """
    from dataclasses import replace
    from encoding import CubePerspective
    from modes import CubeOwner, cube_perspective

    state, dice = opening_roll(rng)
    enc_list: List[np.ndarray] = []
    targets: List[float] = []
    is_opening = True  # no doubling on the opening roll

    while not mode.is_episode_over(state):
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
                    v_no_double = agent.evaluate_cubeful(
                        state, pre_persp, is_cube_action=False,
                    )
                    v_double_take = 2.0 * agent.evaluate_cubeful(
                        state, CubePerspective.THEIRS, is_cube_action=False,
                    )
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




# ── Parallel collection worker ────────────────────────────────────────


def _split_episodes(total: int, n_workers: int) -> List[int]:
    """Split *total* episodes as evenly as possible across n_workers."""
    base = total // n_workers
    remainder = total % n_workers
    return [base + (1 if i < remainder else 0) for i in range(n_workers)]


def _collect_worker(args):
    """Worker subprocess: rebuild the agent from a state_dict, play
    `num_episodes` episodes, return concatenated (encodings, targets)
    numpy arrays. The GameMode instance is pickled directly into
    `args` — simpler and more general than name-based dispatch,
    because modes can carry config (e.g. CubefulMoneyMode.jacoby).

    Set env LEAK_DIAG=1 to log RSS at entry/exit (for leak hunts).
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    (state_dict, hidden_sizes, input_size, output_mode, encoder_name,
     mode, num_episodes, seed, oneply, oneply_acting, boltzmann_temp,
     bf16_inference, cube_targets_1ply) = args

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
    )
    net.load_state_dict(state_dict)
    agent = TDAgent(
        net, oneply=oneply_acting, boltzmann_temp=boltzmann_temp,
        bf16_inference=bf16_inference,
    )

    rng = random.Random(seed)

    enc_chunks: List[np.ndarray] = []
    tgt_chunks: List[np.ndarray] = []
    for _ in range(num_episodes):
        encs, tgts = collect_episode(
            agent, mode, rng, oneply=oneply,
            cube_targets_1ply=cube_targets_1ply,
        )
        if len(encs) > 0:
            enc_chunks.append(encs)
            tgt_chunks.append(tgts)
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

    if diag:
        rss_out = _rss_kb()
        print(
            f"[worker {os.getpid()}] RSS {rss_in/1024:.1f} -> {rss_out/1024:.1f} MB "
            f"(Δ {(rss_out-rss_in)/1024:+.1f} MB), {num_episodes} eps",
            flush=True,
        )
    return result


# ── Trainer class ─────────────────────────────────────────────────────


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
        cube_targets_1ply: bool = False,
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
        `warmup_cycles` rounds. Useful when starting from a depth-
        expanded model whose new layer needs gentle initial steps.
        Composes with `end_lr` (multiplicative).

        Returns a flat list of per-batch losses.
        """
        mode.validate_agent(self.agent)
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
        # With spawn, recycled workers start pristine, so this is now a
        # real knob rather than a diagnostic stub.
        _mt = os.environ.get("TRAINER_MAXTASKSPERCHILD")
        maxtasks = int(_mt) if _mt else None
        pool = (
            mp.get_context("spawn").Pool(
                processes=workers, maxtasksperchild=maxtasks,
            )
            if workers > 1 else None
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

                if pool is None:
                    # Single-process: master agent does the collection,
                    # so its bf16 inference copy (if enabled) must be
                    # re-synced to the current fp32 weights each round.
                    if bf16_collect:
                        if not self.agent.bf16_inference:
                            self.agent.bf16_inference = True
                        self.agent.refresh_bf16_inference()
                    for _ in range(round_size):
                        encs, tgts = collect_episode(
                            self.agent, mode, rng, oneply=oneply,
                            cube_targets_1ply=cube_targets_1ply,
                        )
                        if len(encs) > 0:
                            enc_chunks.append(encs)
                            tgt_chunks.append(tgts)
                else:
                    splits = _split_episodes(round_size, workers)
                    state_dict = {
                        k: v.detach().cpu()
                        for k, v in net.state_dict().items()
                    }
                    worker_args = [
                        (state_dict, hidden_sizes, input_size, output_mode,
                         encoder_name,
                         mode, n_eps, rng.randint(0, 2**31 - 1),
                         oneply, oneply_acting, boltzmann_temp,
                         bf16_collect, cube_targets_1ply)
                        for n_eps in splits
                    ]
                    for w_enc, w_tgt in pool.map(
                        _collect_worker, worker_args,
                    ):
                        if len(w_enc) > 0:
                            enc_chunks.append(w_enc)
                            tgt_chunks.append(w_tgt)

                episodes_played += round_size
                if metrics_out is not None:
                    metrics_out.setdefault("collection_times", []).append(
                        time.perf_counter() - _t_collect_start
                    )
                    metrics_out.setdefault("round_sizes", []).append(round_size)
                if not enc_chunks:
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

                # Shuffle + batch via index slicing. We used to
                # materialize `sh_enc = pool_enc[perm]` — a full shuffled
                # copy of the entire pool — which doubled peak RSS for
                # the whole training phase of the round. Fancy-indexing
                # per batch holds only a batch-sized copy at a time,
                # and the per-batch op cost is negligible next to the
                # forward/backward pass.
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
                    print(
                        f"Round {round_idx + 1:4d}/{num_rounds} "
                        f"| episodes {episodes_played}/{num_episodes} "
                        f"| pool {n:5d} "
                        f"| recent batch loss {avg:.4f}"
                    )

                if next_save is not None and episodes_played >= next_save:
                    root, ext = os.path.splitext(save_path)
                    ckpt_path = f"{root}_ep{episodes_played}{ext or '.pt'}"
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
        from modes import cube_perspective
        from encoding import CubePerspective

        rng = random.Random(seed)
        lr = self.optimizer.param_groups[0]["lr"]
        losses: List[float] = []
        agent = self.agent

        for episode in range(1, num_episodes + 1):
            state, dice = opening_roll(rng)
            match_state = mode.initial_match_state()

            while not mode.is_episode_over(state):
                player = state.turn

                # ── Cube decision phase (if eligible) ──────────
                if match_state.can_offer(player):
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

            if log_every and episode % log_every == 0:
                recent = losses[-100:] if losses else [0.0]
                avg = sum(recent) / len(recent)
                print(f"Episode {episode:6d} | recent step loss {avg:.4f}")

        return losses
