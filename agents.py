"""
agents.py -- Agent interface and lightweight implementations for backgammon.

No torch dependency — safe to import without a PyTorch installation.
For the torch-based TDAgent see td_agent.py.

An agent chooses a play (sequence of moves) given the current board
state, dice roll, and list of legal plays.
"""

import random
from typing import List, Tuple

from backgammon_engine import BoardState, WHITE, BLACK, Play, switch_turn

try:
    import gnubg_nn
    _GNUBG_NN_AVAILABLE = True
except ImportError:
    _GNUBG_NN_AVAILABLE = False


class Agent:
    """Base class for backgammon agents."""

    def choose_play(
        self,
        state: BoardState,
        dice: Tuple[int, int],
        plays: List[Tuple[Play, BoardState]],
    ) -> Tuple[Play, BoardState]:
        """Pick one (play, resulting_state) from the legal plays."""
        raise NotImplementedError


class RandomAgent(Agent):
    """Picks a random legal play."""

    def choose_play(self, state, dice, plays):
        return random.choice(plays)


class GnubgNNAgent(Agent):
    """Uses gnubg-nn (0-ply) to evaluate positions and pick the best move.

    Evaluates all resulting states after switch_turn and picks the move
    that minimizes the opponent's win probability (argmin of P(on-roll wins)
    from opponent's perspective), consistent with perspective encoding.

    Requires: pip install gnubg-nn
    """

    def __init__(self, plies: int = 0):
        if not _GNUBG_NN_AVAILABLE:
            raise ImportError("gnubg-nn is not installed. Run: pip install gnubg-nn")
        self.plies = plies

    @staticmethod
    def _board_to_gnubg(state: BoardState) -> list:
        """Convert BoardState to gnubg's 2x25 board format.

        gnubg convention: board[1] = on-roll player, board[0] = opponent.
        probs[0] from gnubg_nn.probabilities = P(on-roll player wins).
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
            return [black_board, white_board]   # board[1] = WHITE (on roll)
        else:
            return [white_board, black_board]   # board[1] = BLACK (on roll)

    def choose_play(self, state, dice, plays):
        # Evaluate each resulting state from the opponent's perspective
        # (after switch_turn). Pick the move that minimises opponent's win prob.
        best_idx = 0
        best_val = float("inf")
        for i, (play, next_state) in enumerate(plays):
            opponent_state = switch_turn(next_state)
            board = self._board_to_gnubg(opponent_state)
            probs = gnubg_nn.probabilities(board, self.plies)
            val = probs[0]   # P(on-roll opponent wins) — minimise this
            if val < best_val:
                best_val = val
                best_idx = i
        return plays[best_idx]
