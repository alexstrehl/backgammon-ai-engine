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
from typing import Callable, List, Optional, Sequence, Tuple, Union

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
            agent, mode, rng, initial_match_state, oneply=oneply,
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Cubeful MONEY-game episode collector.

    Money-specific assumptions baked in:
      - A pass ends the episode with per-unit target 1.0.
      - A terminal ends the episode with per-unit game_result.
      - A take bootstrap is literally 2·v_cube_theirs (stake doubles).
      - There is no "next game" — one game is one episode.

    Matchplay will need a separate collector (or Option B from the
    matchplay plan: push game-end handling into the Mode so a single
    cubeful collector can drive both money and matchplay).

    Per turn, up to two samples may be emitted:
      D1 (cube decision): encoded at the doubler's cube perspective
        BEFORE the double. Target = 2·v_cube_theirs (per-unit) if the
        opponent takes, else 1.0 (doubler wins 1 point per-unit on a
        drop).
      D3 (checker decision): encoded at the post-double cube
        perspective. Target is the usual bootstrap (−V(opp view)),
        or terminal game_result (per-unit) at episode end.

    All targets are per-unit equity (cube_value = 1). The agent's
    network learns V(pos, cube_perspective) normalised to stake=1;
    cube_value appears on TerminalOutcome as metadata only.
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

        # ── D1: cube decision (if eligible) ─────────────────────────
        # Doubling is not allowed on the opening roll.
        # When oneply=True, compute v_pre_double and v_cube_theirs via
        # 1-ply lookahead (lower variance) and cache v_pre_double so
        # the D3 checker target can reuse it when no double happens.
        oneply_no_double = None  # cached for D3 reuse
        if not is_opening and match_state.can_offer(player):
            if oneply:
                v_pre_double = agent.value_oneply_checker_cubeful(
                    state, match_state,
                )
                oneply_no_double = v_pre_double
                # v_cube_theirs: same state, but cube is THEIRS (= we
                # doubled and they took). Temporarily flip cube_owner
                # to the opponent for the 1-ply eval.
                opponent = 1 - player
                theirs_ms = replace(
                    match_state, cube_owner=CubeOwner(opponent + 1),
                )
                v_cube_theirs = agent.value_oneply_checker_cubeful(
                    state, theirs_ms,
                )
                pre_persp = cube_perspective(match_state.cube_owner, player)
                do_double = agent._money_cube_decision(
                    v_pre_double, v_cube_theirs, pre_persp, match_state.jacoby,
                )
                offer = None  # placeholder; we use the 1-ply values directly
            else:
                offer = agent.offer_double(state, match_state)
                do_double = offer.should_double
                pre_persp = cube_perspective(match_state.cube_owner, player)
                v_cube_theirs = (
                    offer._cache["v_cube_theirs_0ply"] if do_double else None
                )

            if do_double:
                cube_sample_enc = agent._encode_cubeful(state, pre_persp)
                if oneply:
                    # Same take-point as TDAgent.respond_to_double;
                    # inlined here to avoid a redundant forward pass.
                    takes = (2.0 * v_cube_theirs) <= 1.0
                else:
                    takes = agent.respond_to_double(
                        state, match_state, hint=offer,
                    )
                if takes:
                    # Target = 2 * v_cube_theirs (doubled stake, per-unit).
                    enc_list.append(cube_sample_enc)
                    targets.append(2.0 * v_cube_theirs)
                    match_state = match_state.after_take(player)
                    # After take, cube_owner changed → oneply_no_double
                    # is no longer valid for D3 (computed at old persp).
                    oneply_no_double = None
                else:
                    # Pass → game ends. Doubler wins 1 per unit.
                    enc_list.append(cube_sample_enc)
                    targets.append(1.0)
                    break

        # ── D3: checker decision ────────────────────────────────────
        post_persp = cube_perspective(match_state.cube_owner, player)
        enc_list.append(agent._encode_cubeful(state, post_persp))

        if oneply:
            # Target: 1-ply at current state under current cube persp.
            # Reuse the no-double value if we have it (no cube turn
            # since it was computed).
            if oneply_no_double is not None:
                checker_target = oneply_no_double
            else:
                checker_target = agent.value_oneply_checker_cubeful(
                    state, match_state,
                )
            targets.append(checker_target)
            # Move selection at 0-ply (fast) — just advance the game.
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
                # Forced pass: opponent gets the roll on the same position.
                opp_view = switch_turn(state)
                opp_persp = cube_perspective(
                    match_state.cube_owner, opp_view.turn,
                )
                v_opp = agent.evaluate_cubeful(opp_view, opp_persp)
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
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    (state_dict, hidden_sizes, output_mode, encoder_name,
     mode, num_episodes, seed, oneply) = args

    from model import TDNetwork
    from td_agent import TDAgent

    net = TDNetwork(
        hidden_sizes=hidden_sizes,
        output_mode=output_mode,
        encoder_name=encoder_name,
    )
    net.load_state_dict(state_dict)
    agent = TDAgent(net)

    rng = random.Random(seed)

    enc_chunks: List[np.ndarray] = []
    tgt_chunks: List[np.ndarray] = []
    for _ in range(num_episodes):
        encs, tgts = collect_episode(agent, mode, rng, oneply=oneply)
        if len(encs) > 0:
            enc_chunks.append(encs)
            tgt_chunks.append(tgts)
    if not enc_chunks:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return np.concatenate(enc_chunks, axis=0), np.concatenate(tgt_chunks, axis=0)


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
        end_lr: Optional[float] = None,
        warmup_cycles: int = 0,
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

        # Worker pool created once for the whole run (avoids per-round
        # subprocess startup cost). None means single-process.
        pool = mp.Pool(processes=workers) if workers > 1 else None
        net = self.agent.network
        hidden_sizes = list(net.hidden_sizes)
        output_mode = getattr(net, "output_mode", "probability")
        encoder_name = getattr(net, "encoder_name", "perspective196")

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
                enc_chunks: List[np.ndarray] = []
                tgt_chunks: List[np.ndarray] = []

                if pool is None:
                    for _ in range(round_size):
                        encs, tgts = collect_episode(
                            self.agent, mode, rng, oneply=oneply,
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
                        (state_dict, hidden_sizes, output_mode, encoder_name,
                         mode, n_eps, rng.randint(0, 2**31 - 1), oneply)
                        for n_eps in splits
                    ]
                    for w_enc, w_tgt in pool.map(
                        _collect_worker, worker_args,
                    ):
                        if len(w_enc) > 0:
                            enc_chunks.append(w_enc)
                            tgt_chunks.append(w_tgt)

                episodes_played += round_size
                if not enc_chunks:
                    continue

                pool_enc = np.concatenate(enc_chunks, axis=0)
                pool_tgt = np.concatenate(tgt_chunks, axis=0)
                n = len(pool_enc)

                # ── Train: shuffle + slice + step (master only) ────
                for _epoch in range(epochs_per_round):
                    perm = np.arange(n)
                    rng_np = np.random.default_rng(rng.randint(0, 2**31 - 1))
                    rng_np.shuffle(perm)
                    sh_enc = pool_enc[perm]
                    sh_tgt = pool_tgt[perm]
                    for start in range(0, n, batch_size):
                        batch_e = sh_enc[start:start + batch_size]
                        batch_t = sh_tgt[start:start + batch_size]
                        losses.append(self.train_step(batch_e, batch_t))

                if log_every and (round_idx + 1) % log_every == 0:
                    recent = losses[-10:] if losses else [0.0]
                    avg = sum(recent) / len(recent)
                    print(
                        f"Round {round_idx + 1:4d}/{num_rounds} "
                        f"| episodes {episodes_played}/{num_episodes} "
                        f"| pool {n:5d} "
                        f"| recent batch loss {avg:.4f}"
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

                # ── D1: cube decision (if eligible) ─────────────
                if match_state.can_offer(player):
                    offer = agent.offer_double(state, match_state)
                    if offer.should_double:
                        pre_persp = cube_perspective(
                            match_state.cube_owner, player,
                        )
                        cube_enc = agent._encode_cubeful(state, pre_persp)
                        takes = agent.respond_to_double(
                            state, match_state, hint=offer,
                        )
                        if takes:
                            v_theirs = offer._cache["v_cube_theirs_0ply"]
                            target = 2.0 * v_theirs
                            losses.append(
                                agent.td_update_encoded(cube_enc, target, lr)
                            )
                            match_state = match_state.after_take(player)
                        else:
                            losses.append(
                                agent.td_update_encoded(cube_enc, 1.0, lr)
                            )
                            break

                # ── D3: checker play ────────────────────────────
                post_persp = cube_perspective(match_state.cube_owner, player)
                enc = agent._encode_cubeful(state, post_persp)
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
                    v_opp = agent.evaluate_cubeful(opp_view, opp_persp)
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
