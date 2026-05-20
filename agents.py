"""
agents.py -- Agent interface and lightweight implementations for backgammon.

No torch dependency — safe to import without a PyTorch installation.

Agent interface
---------------

A single Agent class supports three roles:
  1. Action selection         (choose_checker_action)
  2. Value queries            (evaluate / evaluate_batch)
  3. Learning primitives      (bootstrap_target / terminal_target /
                               predict / stack_targets / loss)
Lightweight non-learning agents (RandomAgent,
GnubgNNAgent) only implement role 1 (and optionally 2). Learning agents
(TDAgent, multi-output variants) implement all three.

The optimizer / training loop is outside the agent — it lives in
trainer.Trainer. The agent provides representation-specific primitives:
`predict` runs a forward pass and returns raw outputs in the agent's
own prediction space, and `loss` scores those predictions against targets in
that same space. The Trainer owns backprop, the optimizer, gradient
clipping, schedules, etc. This makes agents safe to use in worker
processes.

Targets are opaque to the trainer. A scalar-equity agent's target is a
float; a multi-output agent's target is a length-K vector. The agent
knows how to stack a list of them into a tensor (`stack_targets`) and
how to compute loss against its own predictions.
"""

import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from backgammon_engine import BoardState, WHITE, BLACK, Play


@dataclass
class CubeOffer:
    """Result of an Agent's offer_double() call.

    `_cache` is opaque to the caller — it lets respond_to_double()
    skip recomputation when self-play threads it back through.
    """
    should_double: bool
    _cache: Any = None

try:
    import gnubg_nn
    _GNUBG_NN_AVAILABLE = True
except ImportError:
    _GNUBG_NN_AVAILABLE = False


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

        Each action's `resulting_state.turn` is the opponent (already
        switched by `backgammon_engine.get_legal_plays`), so
        `evaluate(resulting_state)` returns the value to the OPPONENT.
        The on-roll player picks the action that minimises that.

        Default implementation evaluates every successor in one batch.
        Subclasses may override (e.g. RandomAgent ignores values).
        """
        next_states = [s for _, s in actions]
        values = self.evaluate_batch(next_states)
        idx = int(np.argmin(values))
        return actions[idx]

    # ── value primitive (no grad, scalar, on-roll's view) ──────────────

    def evaluate(self, state: BoardState) -> float:
        """Scalar value of `state` from `state.turn`'s perspective.

        For scalar-equity agents this is the raw network output;
        for multi-output agents this is equity derived from the
        predicted outcome distribution.
        """
        raise NotImplementedError

    def evaluate_batch(self, states: Sequence[BoardState]) -> np.ndarray:
        """Batched evaluate(). Default falls back to a Python loop;
        learning agents should override with a single forward pass.
        """
        return np.asarray([self.evaluate(s) for s in states], dtype=np.float32)

    # ── training-side primitives (used by trainer) ─────────────────────
    #
    # The Trainer composes targets from these primitives and owns the
    # optimizer / backprop step.

    def bootstrap_target(self, next_state: BoardState):
        """Target for the state PRECEDING `next_state`, derived by
        bootstrapping the agent's own value function.

        For checker plays, `next_state.turn` is the opponent of the mover,
        so a raw evaluation of `next_state` is from the opponent's view.
        The agent flips this into the mover's target representation
        (e.g. `1 - V` for win-prob, `-V` for equity).

        No grad. Used by the trainer to construct TD targets.
        """
        raise NotImplementedError

    def terminal_target(self, terminal_info):
        """Target representing a terminal outcome. Used as the TD target
        when an episode ends. `terminal_info` is a mode-defined record
        produced by the game engine (winner, gammon flag, cube value, etc.).
        """
        raise NotImplementedError

    def predict(self, states: Sequence[BoardState]):
        """Forward pass on a batch of states, returning predictions in
        the agent's prediction space (a tensor whose shape depends on
        the agent type — 1-D for scalar TDAgent, [B, K] for
        multi-output). Trainer uses this to compute loss with grad
        enabled (and wraps in `torch.no_grad()` for eval).
        """
        raise NotImplementedError

    def stack_targets(self, targets):
        """Convert a python list of targets (as produced by
        `bootstrap_target` / `terminal_target`) into a tensor of the
        appropriate shape on the agent's device. Trainer calls this
        after collecting targets and before passing them to `loss`.
        """
        raise NotImplementedError

    def loss(self, predictions, targets):
        """Compute the training loss between `predictions` (from
        `predict`) and `targets` (from `stack_targets`). Returns a
        scalar tensor. The trainer is responsible for `backward()` and
        the optimizer step.
        """
        raise NotImplementedError

    # ── cube actions (cubeful modes only) ──────────────────────────────
    #
    # Defaults are a valid "never offer, always take" cube policy —
    # sensible for RandomAgent and cubeless-trained TDAgents when
    # dropped into a cubeful game. Cubeful learning agents override.

    def offer_double(self, state: BoardState, match_state) -> CubeOffer:
        """Pre-roll cube decision: should the on-roll player double?

        Returns a CubeOffer carrying the decision plus an opaque cache
        that `respond_to_double` can consume to skip recomputation
        during self-play. Inter-agent play passes `hint=None`.

        Default: never doubles. Override for cubeful agents.
        """
        return CubeOffer(should_double=False, _cache=None)

    def respond_to_double(
        self,
        state: BoardState,
        match_state,
        hint: Optional[CubeOffer] = None,
    ) -> bool:
        """Cube response from the player who was just doubled.
        Returns True for take, False for pass.

        Default: always takes. Override for cubeful agents.
        """
        return True

    def choose_checker_action_cubeful(
        self, state: BoardState, dice, match_state,
    ):
        """Cubeful move selection. Returns `(next_state, bootstrap)`
        or None on forced pass. `bootstrap` is only consumed by the
        trainer; non-training callers (eval, play_models) just read
        `next_state`.

        Default: ignore the cube context and pick via the cubeless
        `choose_checker_action`, returning bootstrap=0.0. Cubeful
        learning agents override to bake the opponent's cube
        perspective into successor encodings.
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


class GnubgNNAgent(Agent):
    """Uses gnubg-nn (0-ply or higher) as the value function.

    With cubeless_money=False (default): minimises opponent's P(win) — DMP play.
    With cubeless_money=True: minimises opponent's cubeless money equity,
        computed from gnubg's 5-tuple
        (P_win, P_win_gammon, P_win_bg, P_lose_gammon, P_lose_bg) as
        (P_win - P_lose) + (P_wg - P_lg) + (P_wbg - P_lbg).
    """

    def __init__(self, plies: int = 0, cubeless_money: bool = False):
        if not _GNUBG_NN_AVAILABLE:
            raise ImportError("gnubg-nn is not installed. Run: pip install gnubg-nn")
        self.plies = plies
        self.cubeless_money = cubeless_money

    @staticmethod
    def _board_to_gnubg(state: BoardState) -> list:
        """Convert BoardState to gnubg's 2x25 format. board[1] = on-roll player.
        gnubg_nn.probabilities returns probs[0] = P(on-roll wins).
        """
        white_board = [0] * 25
        black_board = [0] * 25
        for i in range(24):
            v = state.points[i]
            if v > 0:
                white_board[i] = v
            elif v < 0:
                black_board[23 - i] = -v
        white_board[24] = state.bar[WHITE]
        black_board[24] = state.bar[BLACK]

        if state.turn == WHITE:
            return [black_board, white_board]
        else:
            return [white_board, black_board]

    @staticmethod
    def _probs_to_equity(probs) -> float:
        """5-tuple → cubeless money equity for the on-roll player."""
        p_win, p_wg, p_wbg, p_lg, p_lbg = probs
        p_lose = 1.0 - p_win
        return (p_win - p_lose) + (p_wg - p_lg) + (p_wbg - p_lbg)

    def evaluate(self, state: BoardState) -> float:
        """Value at this state from the on-roll player's view.
        DMP mode: P(on-roll wins). Cubeless money mode: cubeless money equity.
        """
        board = self._board_to_gnubg(state)
        probs = gnubg_nn.probabilities(board, self.plies)
        if self.cubeless_money:
            return self._probs_to_equity(probs)
        return float(probs[0])


class GnubgNNCubefulAgent(Agent):
    """Cubeful money agent backed by gnubg-nn.

    Checker play: for each legal move we call ``gnubg_nn.probabilities``
    on the resulting position and convert the 5-tuple to cubeless money
    equity, then pick the move minimising the opponent's equity.

    Note on cube-aware checker play: gnubg's desktop "Cubeful checker
    evaluations" GUI toggle does take cube ownership into account, but
    we could not find a way to enable that path through gnubg-nn —
    every ``set.*`` combination we tried left both ``probabilities``
    and ``best_move`` invariant to cube ownership at every ply. There
    is a ``best_move`` API that could be used in place of the equity
    argmin above, but in a 400k-game H-H at 0-ply, calling it both at
    money score (0,0) and at the 21-point 0-0 match-equity approximation
    was statistically inferior to the equity approach, so we leave that
    path unused.

    Cube decisions: gnubg_nn.evaluate_cube_decision at a simulated
    match (default 21pt, 0-0) approximates money play. Match-length
    sweep against ml=21 baseline at 800k games showed ml ∈ {13,17,21,25}
    all within ±2 mEq of each other (effective tie); ml ∈ {31, 41} are
    broken in gnubg-nn (runaway tails / weak play). 21pt picked as a
    safe middle. Jacoby override: when gnubg says "no double, too good"
    on a centered cube, we force a double — under Jacoby, gammons don't
    count on a centered cube so doubling is monotonic.

    Cube plies clamp: gnubg-nn's evaluate_cube_decision at n=0 is
    broken (returns stale state — see __init__ comment). We default
    cube_plies to max(plies, 1) so a "0-ply" GnubgNNCubefulAgent still
    makes sane cube decisions, at ~25-40% throughput cost. Pass
    cube_plies=0 to reproduce the historical (broken) behavior.
    """

    def __init__(
        self,
        plies: int = 0,
        match_length: int = 21,
        cube_plies: Optional[int] = None,
    ):
        # cube_plies controls plies for evaluate_cube_decision calls.
        # Defaults to max(plies, 1) — i.e. clamped to at least 1.
        #
        # Why the clamp: gnubg-nn's evaluate_cube_decision is BROKEN at
        # n=0. In gnubg-nn/analyze/danalyze.cc the else branch of
        # Analyze::R1::cubefulEquities (the nPlies==0 case) calls
        # cubefulEquity once and discards the result, never writing
        # matchProbNoDouble / matchProbDoubleTake / matchProbDoubleDrop.
        # setDecision then reads those three fields from stale state
        # left over from a previous cube call on the same Player object,
        # so the decision is essentially random.
        #
        # Verified empirically: gnubg's CLI `hint` at 0-ply gives the
        # right answer; gnubg-nn's evaluate_cube_decision(n=0) does not.
        # At n>=1 the bug doesn't trigger (the n>0 branch writes the
        # three fields correctly).
        #
        # Speed cost of the clamp: cube actions are ~5-10/game vs
        # ~50-100 chequer moves, so going from n=0 to n=1 on cube alone
        # only adds ~25-40% wall time per game. Worth it for sane cube
        # decisions.
        #
        # Set cube_plies=0 explicitly if you specifically want the
        # broken-at-0-ply behavior (e.g. to reproduce historical numbers).
        if not _GNUBG_NN_AVAILABLE:
            raise ImportError("gnubg-nn is not installed. Run: pip install gnubg-nn")
        self.plies = plies
        self.match_length = match_length
        self.cube_plies = cube_plies if cube_plies is not None else max(plies, 1)

    def _board_and_probs(self, state: BoardState):
        board = GnubgNNAgent._board_to_gnubg(state)
        probs = gnubg_nn.probabilities(board, self.plies)
        return board, probs

    def _equity(self, probs) -> float:
        return GnubgNNAgent._probs_to_equity(probs)

    def evaluate(self, state: BoardState) -> float:
        _, probs = self._board_and_probs(state)
        return self._equity(probs)

    def choose_checker_action(self, state, dice, actions):
        """Pick by cubeless money equity (argmin over opponent's equity)."""
        next_states = [s for _, s in actions]
        values = np.asarray(
            [self.evaluate(s) for s in next_states], dtype=np.float32,
        )
        idx = int(np.argmin(values))
        return actions[idx]

    @staticmethod
    def _set_cube_for(match_state, on_roll_player: int) -> None:
        """Configure gnubg's cube state to match the actual game's cube
        ownership (perspective-corrected).

        gnubg-nn convention (verified empirically against gnubg CLI):
        the on-roll player at board[1] maps to gnubg's "O" internally,
        even though gnubg's CLI labels it X. So we keep the default
        s parameter in evaluate_cube_decision (xOnPlay=false = "O on
        play"), and pair owned cubes with owner=b'O' (xOwns=false,
        matches xOnPlay).
        """
        from modes import cube_perspective, CubePerspective
        persp = cube_perspective(match_state.cube_owner, on_roll_player)
        if persp == CubePerspective.CENTERED:
            gnubg_nn.set.cube(1, b'X')   # owner ignored at cube=1
        elif persp == CubePerspective.MINE:
            gnubg_nn.set.cube(2, b'O')   # on-roll owns; matches default s
        else:
            gnubg_nn.set.cube(2, b'X')   # off-roll owns (defensive — usually unreachable)

    def offer_double(self, state: BoardState, match_state) -> CubeOffer:
        from modes import CubeOwner
        board, probs = self._board_and_probs(state)
        equity = self._equity(probs)

        # gnubg cube decision at simulated match play.
        # Verbose (i=1) returns:
        #   (actionDouble, actionTake, tooGood,
        #    mwcND, mwcDT, mwcDP)
        pid = gnubg_nn.position_id(board)
        gnubg_nn.set.score(self.match_length, self.match_length)
        self._set_cube_for(match_state, state.turn)
        r = gnubg_nn.evaluate_cube_decision(pid, n=self.cube_plies, i=1)
        action_double = r[0]
        action_take = r[1]
        too_good = r[2]
        should_double = (action_double == 1)

        # Jacoby override for centered cube: under Jacoby gammons
        # don't count on a centered cube, so doubling is monotonic
        # regardless of opponent's take/drop. Drop the action_take
        # gate the older code had.
        if (
            not should_double
            and too_good
            and match_state.jacoby
            and match_state.cube_owner == CubeOwner.CENTERED
        ):
            should_double = True

        return CubeOffer(
            should_double=should_double,
            _cache={"equity": equity, "action_take": action_take},
        )

    def respond_to_double(
        self,
        state: BoardState,
        match_state,
        hint: Optional[CubeOffer] = None,
    ) -> bool:
        """Take/pass using gnubg's match evaluation. If the offer
        hint carries gnubg's actionTake, reuse it (same position,
        same evaluation). Otherwise query gnubg fresh.
        """
        if hint is not None and hint._cache and "action_take" in hint._cache:
            return hint._cache["action_take"] == 1
        # Fresh evaluation
        board = GnubgNNAgent._board_to_gnubg(state)
        pid = gnubg_nn.position_id(board)
        gnubg_nn.set.score(self.match_length, self.match_length)
        self._set_cube_for(match_state, state.turn)
        r = gnubg_nn.evaluate_cube_decision(pid, n=self.cube_plies, i=1)
        return r[1] == 1  # actionTake
