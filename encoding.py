"""
encoding.py -- Board state encoding for neural network input.

Always encodes from the ON-ROLL PLAYER's perspective.  The network
learns a single function: P(on-roll player wins | board).

When WHITE is on roll, WHITE's checkers go in the "my" block [0-97]
and BLACK's in the "opponent" block [98-195], with normal point order.

When BLACK is on roll, BLACK's checkers go in the "my" block [0-97]
and WHITE's in the "opponent" block [98-195], with MIRRORED point
indices (so "my home board" is always in the same feature positions).

Layout (196 features):
    MY block (98 features):
        For each of 24 points: 4 units for on-roll player's checkers = 96
        My bar / 2.0                                                  =  1
        My off / 15.0                                                 =  1

    OPPONENT block (98 features):
        For each of 24 points: 4 units for opponent's checkers        = 96
        Opponent bar / 2.0                                            =  1
        Opponent off / 15.0                                           =  1

    Total: 98 + 98 = 196

Network output interpretation:
    output = P(on-roll player wins)
    P(WHITE wins) = output          when WHITE is on roll
    P(WHITE wins) = 1 - output      when BLACK is on roll
"""

import numpy as np
from backgammon_engine import BoardState, WHITE, BLACK


def _encode_checkers(x: np.ndarray, offset: int, count: int):
    """Write the 4-unit thermometer encoding for *count* checkers."""
    if count >= 1:
        x[offset] = 1.0
    if count >= 2:
        x[offset + 1] = 1.0
    if count >= 3:
        x[offset + 2] = 1.0
        if count >= 4:
            x[offset + 3] = (count - 3) * 0.5


# ── Encoder abstraction ──────────────────────────────────────────────────────

class Encoder:
    """Base class for board state encoders."""

    @property
    def num_features(self) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    def encode(self, state: BoardState) -> np.ndarray:
        raise NotImplementedError


class Perspective196Encoder(Encoder):
    """196-feature perspective encoding (4-unit thermometer per point).

    Encodes from the on-roll player's perspective.  BLACK's board is
    mirrored so that "my home board" is always in the same positions.
    """

    @property
    def num_features(self) -> int:
        return 196

    @property
    def name(self) -> str:
        return "perspective196"

    def encode(self, state: BoardState) -> np.ndarray:
        x = np.zeros(196, dtype=np.float32)
        pts = state.points

        if state.turn == WHITE:
            # MY block [0-97]: WHITE's checkers at normal indices
            for idx in range(24):
                v = pts[idx]
                if v > 0:
                    _encode_checkers(x, idx * 4, v)
            x[96] = state.bar[WHITE] * 0.5
            x[97] = state.off[WHITE] / 15.0

            # OPPONENT block [98-195]: BLACK's checkers at normal indices
            for idx in range(24):
                v = pts[idx]
                if v < 0:
                    _encode_checkers(x, 98 + idx * 4, -v)
            x[194] = state.bar[BLACK] * 0.5
            x[195] = state.off[BLACK] / 15.0

        else:  # BLACK on roll
            # MY block [0-97]: BLACK's checkers at mirrored indices
            for idx in range(24):
                v = pts[idx]
                if v < 0:
                    _encode_checkers(x, (23 - idx) * 4, -v)
            x[96] = state.bar[BLACK] * 0.5
            x[97] = state.off[BLACK] / 15.0

            # OPPONENT block [98-195]: WHITE's checkers at mirrored indices
            for idx in range(24):
                v = pts[idx]
                if v > 0:
                    _encode_checkers(x, 98 + (23 - idx) * 4, v)
            x[194] = state.bar[WHITE] * 0.5
            x[195] = state.off[WHITE] / 15.0

        return x


ENCODERS = {
    "perspective196": Perspective196Encoder,
}


def get_encoder(name: str) -> Encoder:
    """Look up an encoder by name and return a new instance."""
    if name not in ENCODERS:
        raise ValueError(
            f"Unknown encoder: {name!r}. Available: {sorted(ENCODERS)}"
        )
    return ENCODERS[name]()


# ── Module-level backward-compatible API ─────────────────────────────────────

_default_encoder = Perspective196Encoder()

NUM_FEATURES = _default_encoder.num_features  # 196


def encode_state(state: BoardState) -> np.ndarray:
    """Encode a board state from the on-roll player's perspective.

    Delegates to the default Perspective196Encoder.
    """
    return _default_encoder.encode(state)
