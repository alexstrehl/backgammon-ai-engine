"""
modes.py -- Game-mode abstractions for backgammon training.

A GameMode describes one game type: initial state, episode (full game
or match) boundary, terminal outcome construction.

Mode × output_mode pairing
--------------------------
GameMode and TDAgent.output_mode are orthogonal axes paired at
training time. The coupling is the `TerminalOutcome` dataclass. The
Mode populates fields and the Agent reads the fields it cares about.

  | Game          | Output mode | Terminal value         | Status     |
  |---------------|-------------|------------------------|------------|
  | DMP           | probability | 1.0                    | ✓ exact    |
  | DMP           | equity      | 1.0                    | ✓ equiv    |
  | CubelessMoney | equity      | 1 / 2 / 3              | ✓ exact    |
  | CubefulMoney  | equity      | game_result × cube_val | ✓ exact    |

Terminal call convention: at terminal `state.turn` is the LOSER
because the engine already switched after the winning move. The
mode asserts `state.winner() == 1 - state.turn`.

Cubeful modes also carry a MatchState (cube ownership and cube value)
alongside the BoardState. Cubeless modes never produce one. Their
`initial_match_state()` returns None.
"""

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Optional

from backgammon_engine import BoardState
from encoding import CubePerspective
from td_agent import TerminalOutcome


# ── Cube state ───────────────────────────────────────────────────────


class CubeOwner(IntEnum):
    """Absolute cube ownership (game-level)."""
    CENTERED = 0
    WHITE = 1
    BLACK = 2


def cube_perspective(cube_owner: CubeOwner, on_roll_player: int) -> CubePerspective:
    """Return CubePerspective from on-roll's view given absolute cube owner."""
    if cube_owner == CubeOwner.CENTERED:
        return CubePerspective.CENTERED
    # CubeOwner.WHITE = WHITE(0)+1, CubeOwner.BLACK = BLACK(1)+1
    if cube_owner == CubeOwner(on_roll_player + 1):
        return CubePerspective.MINE
    return CubePerspective.THEIRS


@dataclass(frozen=True)
class MatchState:
    """Cube state outside the BoardState, plus the match score.

    Money modes use match_length=0 (score fields ignored).
    Matchplay modes set match_length > 0 and track away-scores.
    """
    cube_owner: CubeOwner = CubeOwner.CENTERED
    cube_value: int = 1
    jacoby: bool = True
    # Matchplay fields (ignored when match_length=0).
    # Away scores are ALWAYS stored in WHITE's view ("my" = WHITE);
    # perspective flipping happens at encode time, never here.
    match_length: int = 0
    my_away: int = 0
    opp_away: int = 0
    is_crawford: bool = False
    # Latch: True once the Crawford game has been scheduled. Without
    # it, after_game's is_crawford reset would let start_game re-flag
    # Crawford on every game after a player reaches 1-away.
    crawford_done: bool = False

    @property
    def is_matchplay(self) -> bool:
        return self.match_length > 0

    def can_offer(self, on_roll_player: int) -> bool:
        """Can the on-roll player offer a double?
        Crawford game: no doubling allowed.
        Matchplay dead cube: no doubling when winning at the current
        cube value already wins the match for the doubler."""
        if self.is_crawford:
            return False
        if self.is_matchplay:
            doubler_away = (
                self.my_away if on_roll_player == 0 else self.opp_away
            )
            if self.cube_value >= doubler_away:
                return False
        persp = cube_perspective(self.cube_owner, on_roll_player)
        return persp in (CubePerspective.CENTERED, CubePerspective.MINE)

    def after_take(self, doubler: int) -> "MatchState":
        """New MatchState after a double has been taken: cube transfers
        to the opponent of the doubler, value doubles."""
        opponent = 1 - doubler
        return replace(
            self,
            cube_owner=CubeOwner(opponent + 1),
            cube_value=self.cube_value * 2,
        )

    def start_game(self) -> "MatchState":
        """Reset cube for a new game within a match. Sets the Crawford
        flag for the FIRST game after a player reaches 1-away, and
        latches `crawford_done` so it fires exactly once per match."""
        crawford = (
            not self.crawford_done
            and (self.my_away == 1 or self.opp_away == 1)
        )
        return replace(
            self,
            cube_owner=CubeOwner.CENTERED,
            cube_value=1,
            is_crawford=crawford,
            crawford_done=self.crawford_done or crawford,
        )

    def after_game(self, game_result: int, winner_is_me: bool) -> "MatchState":
        """Update match score after a game. game_result is 1/2/3;
        points = cube_value × game_result. Returns new MatchState
        with updated away-scores."""
        points = self.cube_value * game_result
        if winner_is_me:
            new_my = max(0, self.my_away - points)
            new_opp = self.opp_away
        else:
            new_my = self.my_away
            new_opp = max(0, self.opp_away - points)
        return replace(
            self,
            my_away=new_my,
            opp_away=new_opp,
            cube_owner=CubeOwner.CENTERED,
            cube_value=1,
            is_crawford=False,
        )

    def match_over(self) -> bool:
        return self.is_matchplay and (self.my_away <= 0 or self.opp_away <= 0)

    def i_won_match(self) -> bool:
        return self.my_away <= 0


# ── GameMode ─────────────────────────────────────────────────────────


class GameMode:
    """Base class for game-type-specific behaviour."""

    def initial_state(self) -> BoardState:
        return BoardState.initial()

    def initial_match_state(self) -> Optional[MatchState]:
        """Cubeless modes return None. Cubeful modes return a fresh
        MatchState (centered cube, value 1)."""
        return None

    def is_episode_over(self, state: BoardState) -> bool:
        raise NotImplementedError

    def make_terminal_outcome(
        self,
        state: BoardState,
        match_state: Optional[MatchState] = None,
    ) -> TerminalOutcome:
        """Construct a TerminalOutcome at episode end. Must assert
        that `state` is terminal and that `state.turn == loser`.
        """
        raise NotImplementedError

    def validate_agent(self, agent) -> None:
        """Reject incompatible agents at training-loop entry. Default
        accepts everything. Modes with a hard requirement override.
        """
        return


def _assert_terminal(state: BoardState) -> None:
    assert state.is_game_over(), \
        "make_terminal_outcome called on non-terminal state"
    assert state.winner() == 1 - state.turn, (
        f"state.turn={state.turn} but winner={state.winner()}; "
        f"expected state.winner() == 1 - state.turn"
    )


def _require_equity_output(mode_name: str, agent) -> None:
    output_mode = getattr(agent, "output_mode", None)
    if output_mode != "equity":
        raise ValueError(
            f"{mode_name} requires an equity-output agent; got "
            f"output_mode={output_mode!r}. Probability output cannot "
            f"represent gammon (2) or backgammon (3) terminal values."
        )


class CubelessMoneyMode(GameMode):
    """Cubeless money: episode = single game. Gammons (2x) and
    backgammons (3x) carried in TerminalOutcome. Requires an
    equity-output agent.
    """

    def is_episode_over(self, state: BoardState) -> bool:
        return state.is_game_over()

    def make_terminal_outcome(
        self,
        state: BoardState,
        match_state: Optional[MatchState] = None,
    ) -> TerminalOutcome:
        _assert_terminal(state)
        result = state.game_result()  # 1 single, 2 gammon, 3 backgammon
        return TerminalOutcome(
            won_gammon=(result >= 2),
            won_backgammon=(result == 3),
        )

    def validate_agent(self, agent) -> None:
        _require_equity_output("CubelessMoneyMode", agent)


class DMPMode(GameMode):
    """Double Match Point: episode = single game. Gammons don't matter."""

    def is_episode_over(self, state: BoardState) -> bool:
        return state.is_game_over()

    def make_terminal_outcome(
        self,
        state: BoardState,
        match_state: Optional[MatchState] = None,
    ) -> TerminalOutcome:
        _assert_terminal(state)
        return TerminalOutcome(won_gammon=False, won_backgammon=False, cube_value=1)


class MatchplayMode(GameMode):
    """N-point match play. Episode = one full MATCH (multiple games
    chained via MatchState.start_game/after_game/match_over); the
    matchplay collector owns the game loop. The network output is
    P(on-roll player wins the MATCH) ∈ [0, 1], so a probability-output
    agent with a MatchplayEncoder is required. No Jacoby in matchplay;
    gammons/backgammons always count toward the score.
    """

    def __init__(self, match_length: int = 2, outcome_mix: float = 0.0,
                 oracle_spec: str = None, oracle_move_eps: float = None,
                 oracle_plies: int = 0,
                 oracle_act_prob: float = 0.0,
                 move_explore_eps: float = 0.0,
                 td_move_eps: float = 0.0,
                 oracle_1ply_eps: float = 0.0,
                 oneply_act_k: int = 0,
                 td_eps_only: bool = False,
                 oracle_act_only: bool = False):
        if match_length < 1:
            raise ValueError(f"match_length must be >= 1, got {match_length}")
        if not (0.0 <= outcome_mix <= 1.0):
            raise ValueError(f"outcome_mix must be in [0,1], got {outcome_mix}")
        self.match_length = match_length
        # Supervised oracle training: when set (a prob5 checkpoint
        # path, optional ":x=V" suffix), the collector keeps acting
        # with the trained net but REPLACES every target with the
        # p5jan oracle's value of the encoded state (janowski_match):
        # ca=1 rows get max(v_nd, min(v_dt, v_dp)), ca=0 rows the
        # cubeful MWC. Plain supervised regression on a fixed external
        # oracle -- no bootstrap self-reference anywhere.
        self.oracle_spec = oracle_spec
        # Near-optimal action supervision (needs oracle_spec): also
        # emit one row per legal successor whose oracle value is
        # within this epsilon (MWC) of the oracle's best move --
        # training the exact reads move selection compares.
        self.oracle_move_eps = oracle_move_eps
        # Oracle lookahead depth: 0 = static p5jan (the default),
        # 1 = the oracle's own 1-ply value (value_oneply) for every
        # row — a strictly stronger teacher at ~20x collection cost.
        # oracle_move_eps rows are priced at the same depth, keeping
        # the uniform-target invariant.
        self.oracle_plies = int(oracle_plies)
        if not (0.0 <= float(oracle_act_prob) <= 1.0):
            raise ValueError(
                f"oracle_act_prob must be in [0,1], got {oracle_act_prob}")
        self.oracle_act_prob = float(oracle_act_prob)
        if not (0.0 <= float(move_explore_eps) <= 1.0):
            raise ValueError(
                f"move_explore_eps must be in [0,1], got {move_explore_eps}")
        self.move_explore_eps = float(move_explore_eps)
        # TD self-consistency rows: also emit one row per legal
        # successor within this epsilon (MWC) of the best 0-ply value,
        # each targeted at its OWN 1-ply value. No teacher involved.
        # Closes the coverage hole where ~540 successor reads feed each
        # target but only the played one ever becomes a row; the
        # skipped siblings are undervalued by +12.3 mMWC while their
        # own 1-ply value is unbiased.
        # Emitting a row does not play the move, so the state
        # distribution is unchanged.
        if not (0.0 <= float(td_move_eps) <= 1.0):
            raise ValueError(
                f"td_move_eps must be in [0,1], got {td_move_eps}")
        self.td_move_eps = float(td_move_eps)
        # Cheap-oracle FILTER for the expensive one: with oracle_plies>0,
        # only successors within this epsilon of the 0-ply oracle's best
        # get re-priced at depth; the rest keep their 0-ply value. Cuts
        # the dominant cost of a 1-ply oracle leg. Mixes target depths --
        # see _oracle_move_rows for why that is a real risk.
        if not (0.0 <= float(oracle_1ply_eps) <= 1.0):
            raise ValueError(
                f"oracle_1ply_eps must be in [0,1], got {oracle_1ply_eps}")
        self.oracle_1ply_eps = float(oracle_1ply_eps)
        # Expert iteration: collector acts with 1-ply move selection over
        # the top-K successors by the 0-ply read. 0 = act 0-ply (default).
        if int(oneply_act_k) < 0:
            raise ValueError(
                f"oneply_act_k must be >= 0, got {oneply_act_k}")
        self.oneply_act_k = int(oneply_act_k)
        # Drop the on-trajectory checker rows, keeping eps rows, cube rows
        # and boundary rows. Isolates whether the bootstrapped trajectory
        # targets are what degrades a converged net.
        self.td_eps_only = bool(td_eps_only)
        self.oracle_act_only = bool(oracle_act_only)
        # Fraction of the realized match outcome blended into every
        # collected target: t = (1-mix)*backup + mix*z. Pure TD(0)
        # (mix=0) lets cube-decision values sit at self-consistent
        # wrong fixed points: cube samples are never outcome-grounded
        # and their max/min backup re-injects sibling cube-context
        # values into the bootstrap chain at every cube-eligible turn
        # (diagnosed + fix validated via train_from_position.py).
        self.outcome_mix = outcome_mix

    def initial_match_state(self) -> MatchState:
        return MatchState(
            jacoby=False,
            match_length=self.match_length,
            my_away=self.match_length,
            opp_away=self.match_length,
        )

    def validate_agent(self, agent) -> None:
        if not getattr(agent, "is_matchplay", False):
            raise ValueError(
                "MatchplayMode requires a MatchplayEncoder agent "
                f"(encoder='matchplay_<base>'); got encoder="
                f"{getattr(getattr(agent, 'encoder', None), 'name', None)!r}."
            )
        output_mode = getattr(agent, "output_mode", None)
        if output_mode != "probability":
            raise ValueError(
                "MatchplayMode requires output_mode='probability' "
                f"(P(win match)); got {output_mode!r}."
            )

    def is_episode_over(self, state: BoardState) -> bool:
        # Safe fallback only: the matchplay collector tracks the real
        # episode boundary via match_state.match_over().
        return state.is_game_over()

    def make_terminal_outcome(
        self,
        state: BoardState,
        match_state: Optional[MatchState] = None,
    ) -> TerminalOutcome:
        _assert_terminal(state)
        result = state.game_result()
        return TerminalOutcome(
            won_gammon=(result >= 2),
            won_backgammon=(result == 3),
            cube_value=(match_state.cube_value if match_state else 1),
        )


class CubefulMoneyMode(GameMode):
    """Cubeful money. Carries a MatchState. Terminal target multiplies
    game_result by cube_value. Under Jacoby (default), gammons and
    backgammons only count once the cube has been turned.
    """

    def __init__(self, jacoby: bool = True):
        self.jacoby = jacoby

    def initial_match_state(self) -> MatchState:
        return MatchState(jacoby=self.jacoby)

    def validate_agent(self, agent) -> None:
        _require_equity_output("CubefulMoneyMode", agent)

    def is_episode_over(self, state: BoardState) -> bool:
        return state.is_game_over()

    def make_terminal_outcome(
        self,
        state: BoardState,
        match_state: Optional[MatchState] = None,
    ) -> TerminalOutcome:
        _assert_terminal(state)
        assert match_state is not None, \
            "CubefulMoneyMode requires a MatchState at terminal"
        result = state.game_result()  # 1 / 2 / 3
        # Jacoby: gammons & backgammons collapse to single while cube
        # is centered.
        if match_state.jacoby and match_state.cube_owner == CubeOwner.CENTERED:
            result = 1
        return TerminalOutcome(
            won_gammon=(result >= 2),
            won_backgammon=(result == 3),
            cube_value=match_state.cube_value,
        )

