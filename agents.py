"""
agents.py -- Agent interface and lightweight implementations for backgammon.

No torch dependency.

An Agent has three roles:
  1. Action selection         (choose_checker_action, cube methods)
  2. Value queries            (evaluate / evaluate_batch)
  3. Learning primitives      (bootstrap_target / terminal_target /
                               predict / stack_targets / loss)
RandomAgent implements role 1 only (GnubgNNAgent, roles 1 and 2, is in
third_party_opponents/gnubg/gnubg_nn_agent.py).
TDAgent implements all three.

The training loop lives in trainer.Trainer, which owns the optimizer
and backprop. The agent supplies `predict` (forward pass in its own
prediction space) and `loss` (predictions vs targets in that space).
Targets are opaque to the trainer, and `stack_targets` turns a list of
them into a tensor.
"""

import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from backgammon_engine import BoardState, WHITE, BLACK, Play


@dataclass
class CubeOffer:
    """Result of offer_double(). `_cache` is opaque to the caller. Self-play
    passes it back to respond_to_double() to skip recomputation."""
    should_double: bool
    _cache: Any = None

class Agent:
    """Base class for backgammon agents."""

    # ── action selection ────────────────────────────────────────────────

    def choose_checker_action(
        self,
        state: BoardState,
        dice: Tuple[int, int],
        actions: List[Tuple[Play, BoardState]],
    ) -> Tuple[Play, BoardState]:
        """Pick one (play, resulting_state) from the legal checker actions.

        `resulting_state.turn` is already the opponent, so
        `evaluate(resulting_state)` is the OPPONENT's value. The default
        batch-evaluates every successor and takes the minimum.
        """
        next_states = [s for _, s in actions]
        values = self.evaluate_batch(next_states)
        idx = int(np.argmin(values))
        return actions[idx]

    # ── value primitive (no grad, scalar, on-roll's view) ──────────────

    def evaluate(self, state: BoardState) -> float:
        """Scalar value of `state` from `state.turn`'s view (raw output for
        scalar agents, derived equity for multi-output ones)."""
        raise NotImplementedError

    def evaluate_batch(self, states: Sequence[BoardState]) -> np.ndarray:
        """Batched evaluate(). Default is a Python loop. Learning agents
        override with one forward pass."""
        return np.asarray([self.evaluate(s) for s in states], dtype=np.float32)

    # ── training-side primitives (used by trainer) ─────────────────────

    def bootstrap_target(self, next_state: BoardState):
        """TD target for the state PRECEDING `next_state`, from the
        agent's own value. `next_state.turn` is the opponent, so the
        value is flipped into the mover's view (`1 - V` for win-prob,
        `-V` for equity). No grad."""
        raise NotImplementedError

    def terminal_target(self, terminal_info):
        """TD target for a finished episode. `terminal_info` is the
        mode-defined record from the game engine (winner, gammon flag,
        cube value, ...)."""
        raise NotImplementedError

    def predict(self, states: Sequence[BoardState]):
        """Forward pass on a batch, in the agent's prediction space (1-D
        for scalar agents, [B, K] for multi-output). Grad enabled; the
        trainer wraps it in `torch.no_grad()` for eval."""
        raise NotImplementedError

    def stack_targets(self, targets):
        """List of targets (from `bootstrap_target` / `terminal_target`)
        -> tensor on the agent's device, ready for `loss`."""
        raise NotImplementedError

    def loss(self, predictions, targets):
        """Scalar loss tensor for `predict` output vs `stack_targets`
        output. The trainer does `backward()` and the optimizer step."""
        raise NotImplementedError

    # ── cube actions (cubeful modes only) ──────────────────────────────
    #
    # Defaults are a valid "never offer, always take" policy for
    # cubeless agents in a cubeful game. Cubeful agents override.

    def offer_double(self, state: BoardState, match_state) -> CubeOffer:
        """Pre-roll cube decision: should the on-roll player double?
        Default: never."""
        return CubeOffer(should_double=False, _cache=None)

    def respond_to_double(
        self,
        state: BoardState,
        match_state,
        hint: Optional[CubeOffer] = None,
    ) -> bool:
        """Response of the player just doubled: True = take, False = pass.
        Default: always take."""
        return True

    def choose_checker_action_cubeful(
        self, state: BoardState, dice, match_state,
    ):
        """Cubeful move selection: `(next_state, bootstrap)`, or None on a
        forced pass. Only the trainer reads `bootstrap`.

        Default: ignore the cube and use `choose_checker_action`, with
        bootstrap=0.0. Cubeful agents override.
        """
        from backgammon_engine import get_legal_plays
        plays = get_legal_plays(state, dice)
        if not plays:
            return None
        _, next_state = self.choose_checker_action(state, dice, plays)
        return next_state, 0.0


class RandomAgent(Agent):
    """Picks a uniformly random legal action."""

    def choose_checker_action(self, state, dice, actions):
        return random.choice(actions)
