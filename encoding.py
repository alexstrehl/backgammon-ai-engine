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

from enum import IntEnum

import numpy as np
from backgammon_engine import BoardState, WHITE, BLACK


# ── Feature layout constants ─────────────────────────────────────────
#
# Index arithmetic for the 196-feature perspective encoding.
# "MY" block = on-roll player's checkers; "OPP" block = opponent's.

FEATURES_PER_POINT = 4
NUM_POINTS = 24

MY_BLOCK_OFFSET = 0
MY_POINT_FEATURES = NUM_POINTS * FEATURES_PER_POINT  # 96
MY_BAR_INDEX = MY_POINT_FEATURES                      # 96
MY_OFF_INDEX = MY_POINT_FEATURES + 1                  # 97
MY_BLOCK_SIZE = MY_POINT_FEATURES + 2                 # 98

OPP_BLOCK_OFFSET = MY_BLOCK_SIZE                      # 98
OPP_POINT_FEATURES = NUM_POINTS * FEATURES_PER_POINT  # 96
OPP_BAR_INDEX = OPP_BLOCK_OFFSET + OPP_POINT_FEATURES  # 194
OPP_OFF_INDEX = OPP_BLOCK_OFFSET + OPP_POINT_FEATURES + 1  # 195
OPP_BLOCK_SIZE = OPP_POINT_FEATURES + 2              # 98

BAR_SCALE = 0.5       # bar count divided by 2
OFF_SCALE = 1.0 / 15  # off count divided by 15

# Terminal detection: when OPP_OFF_INDEX feature ≥ 1.0 (off/15 = 15/15),
# the opponent has borne off all checkers → the mover of the ENCODED
# successor has won.
TERMINAL_OFF_THRESHOLD = 1.0 - 1e-6


class CubePerspective(IntEnum):
    """Cube state from the on-roll player's perspective (network input)."""
    CENTERED = 0   # either player can double
    MINE = 1       # on-roll player owns the cube
    THEIRS = 2     # opponent owns the cube


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
        x = np.zeros(MY_BLOCK_SIZE + OPP_BLOCK_SIZE, dtype=np.float32)
        pts = state.points

        if state.turn == WHITE:
            # MY block: WHITE's checkers at normal indices
            for idx in range(NUM_POINTS):
                v = pts[idx]
                if v > 0:
                    _encode_checkers(x, idx * FEATURES_PER_POINT, v)
            x[MY_BAR_INDEX] = state.bar[WHITE] * BAR_SCALE
            x[MY_OFF_INDEX] = state.off[WHITE] * OFF_SCALE

            # OPPONENT block: BLACK's checkers at normal indices
            for idx in range(NUM_POINTS):
                v = pts[idx]
                if v < 0:
                    _encode_checkers(x, OPP_BLOCK_OFFSET + idx * FEATURES_PER_POINT, -v)
            x[OPP_BAR_INDEX] = state.bar[BLACK] * BAR_SCALE
            x[OPP_OFF_INDEX] = state.off[BLACK] * OFF_SCALE

        else:  # BLACK on roll
            # MY block: BLACK's checkers at mirrored indices
            for idx in range(NUM_POINTS):
                v = pts[idx]
                if v < 0:
                    _encode_checkers(x, (23 - idx) * FEATURES_PER_POINT, -v)
            x[MY_BAR_INDEX] = state.bar[BLACK] * BAR_SCALE
            x[MY_OFF_INDEX] = state.off[BLACK] * OFF_SCALE

            # OPPONENT block: WHITE's checkers at mirrored indices
            for idx in range(NUM_POINTS):
                v = pts[idx]
                if v > 0:
                    _encode_checkers(x, OPP_BLOCK_OFFSET + (23 - idx) * FEATURES_PER_POINT, v)
            x[OPP_BAR_INDEX] = state.bar[WHITE] * BAR_SCALE
            x[OPP_OFF_INDEX] = state.off[WHITE] * OFF_SCALE

        return x


class CubefulEncoder(Encoder):
    """Wraps a base encoder, appending a 3-feature one-hot for cube state.

    Features appended: [cube_centered, cube_mine, cube_theirs] —
    exactly one is 1.0. Cube perspective is the on-roll player's view
    (CENTERED / MINE / THEIRS), produced by `cube_perspective()` in
    modes.py from the absolute CubeOwner.
    """

    def __init__(self, base_encoder_name: str = "perspective196"):
        self._base = ENCODERS[base_encoder_name]()
        self.base_name = base_encoder_name

    @property
    def name(self) -> str:
        return f"cubeful_{self.base_name}"

    @property
    def num_features(self) -> int:
        return self._base.num_features + 3

    def encode(self, state: BoardState, cube_state: CubePerspective) -> np.ndarray:
        return self.encode_with_base(self._base.encode(state), cube_state)

    def encode_with_base(
        self, base_features: np.ndarray, cube_state: CubePerspective,
    ) -> np.ndarray:
        """Append the cube one-hot to a precomputed base-encoder array.
        Lets callers reuse a fast (e.g. C) base encoding without
        re-running the Python base encoder.
        """
        x = np.empty(self.num_features, dtype=np.float32)
        x[: base_features.shape[0]] = base_features
        x[base_features.shape[0]:] = 0.0
        x[base_features.shape[0] + int(cube_state)] = 1.0
        return x


ENCODERS = {
    "perspective196": Perspective196Encoder,
}


def get_encoder(name: str) -> Encoder:
    """Look up an encoder by name and return a new instance.
    Names matching `cubeful_<base>` build a `CubefulEncoder` over the
    named base encoder.
    """
    if name.startswith("cubeful_"):
        base_name = name[len("cubeful_"):]
        if base_name not in ENCODERS:
            raise ValueError(
                f"Unknown base encoder: {base_name!r}. "
                f"Available: {sorted(ENCODERS)}"
            )
        return CubefulEncoder(base_name)
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
