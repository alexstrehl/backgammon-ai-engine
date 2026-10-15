"""GnubgNNAgent: gnubg-nn (pip install gnubg-nn) as a value function.

Cube-blind approximation of gnubg (no cube, no Jacoby): H-H sanity checks
only. Real gnubg is gnubg_external.GnubgExternalAgent (gnubg-ext sides).
gnubg_nn is imported on first construction.
"""

import os
import sys

from agents import Agent
from backgammon_engine import BoardState, WHITE, BLACK

# Set once warned; spawned workers inherit it, so a run warns once.
_WARNED_ENV = "_GNUBG_NN_CUBE_WARNED"


def warn_gnubg_nn_cubeful():
    """Cubeful runs only, once per process tree: gnubg-nn's cube play is
    an approximation of gnubg. Real gnubg is the gnubg-ext specs."""
    if os.environ.get(_WARNED_ENV) != "1":
        print("[gnubg-nn] WARNING: fast cube-blind APPROXIMATION of gnubg "
              "(21pt cube approx, no Jacoby). H-H sanity checks only; never "
              "label results 'gnubg'. Real gnubg = gnubg-ext.", file=sys.stderr)
        os.environ[_WARNED_ENV] = "1"


def _gnubg_nn():
    try:
        import gnubg_nn
    except ImportError:
        raise ImportError("gnubg-nn is not installed. Run: pip install gnubg-nn")
    return gnubg_nn


class GnubgNNAgent(Agent):
    """gnubg-nn (any ply) as the value function.

    cubeless_money=False: value = P(win) (DMP play).
    cubeless_money=True: value = cubeless money equity from gnubg's
        5-tuple, (P_win - P_lose) + (P_wg - P_lg) + (P_wbg - P_lbg).
    """

    def __init__(self, plies: int = 0, cubeless_money: bool = False,
                 cubeful: bool = False):
        if cubeful:
            warn_gnubg_nn_cubeful()
        _gnubg_nn()
        self.plies = plies
        self.cubeless_money = cubeless_money

    @staticmethod
    def _board_to_gnubg(state: BoardState) -> list:
        """BoardState -> gnubg's 2x25 format, board[1] = on-roll player
        (so probs[0] from gnubg_nn.probabilities = P(on-roll wins))."""
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
        """5-tuple -> cubeless money equity for the on-roll player."""
        p_win, p_wg, p_wbg, p_lg, p_lbg = probs
        p_lose = 1.0 - p_win
        return (p_win - p_lose) + (p_wg - p_lg) + (p_wbg - p_lbg)

    def evaluate(self, state: BoardState) -> float:
        """On-roll player's P(win), or cubeless money equity in
        cubeless_money mode."""
        board = self._board_to_gnubg(state)
        probs = _gnubg_nn().probabilities(board, self.plies)
        if self.cubeless_money:
            return self._probs_to_equity(probs)
        return float(probs[0])

