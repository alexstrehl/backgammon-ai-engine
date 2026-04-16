"""
modes.py -- Game-mode abstractions for backgammon training.

A GameMode describes one game type: initial state, episode (full game or match) 
boundary, terminal outcome construction.

Mode × output_mode pairing
--------------------------
GameMode and TDAgent.output_mode are orthogonal axes paired at
training time. The coupling is the `TerminalOutcome` dataclass: the
Mode populates fields and the Agent reads the fields it cares about.
New modes can extend TerminalOutcome.

  | Game          | Output mode | Terminal value         | Status     |
  |---------------|-------------|------------------------|------------|
  | DMP           | probability | 1.0                    | ✓ exact    |
  | DMP           | equity      | 1.0                    | ✓ equiv    |
  | CubelessMoney | equity      | 1 / 2 / 3              | ✓ exact    |
  | CubefulMoney  | equity      | game_result × cube_val | ✓ exact    |
  | Matchplay     | -           | -                      | future     |

Terminal call convention: at terminal `state.turn` is the LOSER
because the engine already switched after the winning move; the
mode asserts `state.winner() == 1 - state.turn`.

Cubeful modes additionally carry a MatchState (cube ownership and
cube value) alongside the BoardState. Cubeless modes never produce
one — `initial_match_state()` returns None.
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
    """State outside the BoardState: cube + optional match score.

    Money modes use match_length=0 (score fields ignored).
    Matchplay modes set match_length > 0 and track away-scores.
    """
    cube_owner: CubeOwner = CubeOwner.CENTERED
    cube_value: int = 1
    jacoby: bool = True
    # Matchplay fields (ignored when match_length=0):
    match_length: int = 0
    my_away: int = 0
    opp_away: int = 0
    is_crawford: bool = False

    @property
    def is_matchplay(self) -> bool:
        return self.match_length > 0

    def can_offer(self, on_roll_player: int) -> bool:
        """Can the on-roll player offer a double?
        Crawford game: no doubling allowed."""
        if self.is_crawford:
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
        """Reset cube for a new game within a match. Sets Crawford
        flag when one side is exactly 1-away."""
        crawford = (
            not self.is_crawford  # Crawford only happens once
            and (self.my_away == 1 or self.opp_away == 1)
        )
        return replace(
            self,
            cube_owner=CubeOwner.CENTERED,
            cube_value=1,
            is_crawford=crawford,
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
        accepts everything; modes with a hard requirement override.
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
    """Cubeless money: episode = single game; gammons (2x) and
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
    """Double Match Point: episode = single game; gammons don't matter."""

    def is_episode_over(self, state: BoardState) -> bool:
        return state.is_game_over()

    def make_terminal_outcome(
        self,
        state: BoardState,
        match_state: Optional[MatchState] = None,
    ) -> TerminalOutcome:
        _assert_terminal(state)
        return TerminalOutcome(won_gammon=False, won_backgammon=False, cube_value=1)


class CubefulMoneyMode(GameMode):
    """Cubeful money. Carries a MatchState; terminal target multiplies
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

