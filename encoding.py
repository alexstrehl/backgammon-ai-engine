"""
encoding.py -- Board state encoding for neural network input.

Always encoded from the ON-ROLL player's perspective: the on-roll
player's checkers fill the "my" block [0-97], the opponent's the
"opponent" block [98-195]. When BLACK is on roll the point indices
are mirrored.

Layout (196 features), per block (98):
    24 points x 4-unit thermometer   = 96
    bar / 2.0                        =  1
    off / 15.0                       =  1

Network output is also from the on-roll player's view (win
probability, equity, or prob5; see model.py).
"""

from enum import IntEnum

import numpy as np
from backgammon_engine import BoardState, WHITE, BLACK


# ── Feature layout constants ─────────────────────────────────────────

FEATURES_PER_POINT = 4
NUM_POINTS = 24

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

# Terminal test on encoded successors: OPP off feature >= 1.0 means
# the opponent (of the encoded on-roll player) has borne off all 15.
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
    """196-feature perspective encoding (see module docstring)."""

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
            # Normal indices.
            for idx in range(NUM_POINTS):
                v = pts[idx]
                if v > 0:
                    _encode_checkers(x, idx * FEATURES_PER_POINT, v)
            x[MY_BAR_INDEX] = state.bar[WHITE] * BAR_SCALE
            x[MY_OFF_INDEX] = state.off[WHITE] * OFF_SCALE

            for idx in range(NUM_POINTS):
                v = pts[idx]
                if v < 0:
                    _encode_checkers(x, OPP_BLOCK_OFFSET + idx * FEATURES_PER_POINT, -v)
            x[OPP_BAR_INDEX] = state.bar[BLACK] * BAR_SCALE
            x[OPP_OFF_INDEX] = state.off[BLACK] * OFF_SCALE

        else:
            # BLACK on roll: mirrored indices.
            for idx in range(NUM_POINTS):
                v = pts[idx]
                if v < 0:
                    _encode_checkers(x, (23 - idx) * FEATURES_PER_POINT, -v)
            x[MY_BAR_INDEX] = state.bar[BLACK] * BAR_SCALE
            x[MY_OFF_INDEX] = state.off[BLACK] * OFF_SCALE

            for idx in range(NUM_POINTS):
                v = pts[idx]
                if v > 0:
                    _encode_checkers(x, OPP_BLOCK_OFFSET + (23 - idx) * FEATURES_PER_POINT, v)
            x[OPP_BAR_INDEX] = state.bar[WHITE] * BAR_SCALE
            x[OPP_OFF_INDEX] = state.off[WHITE] * OFF_SCALE

        return x


CUBE_FEATURES = 4  # 3-hot cube perspective + 1 is_cube_action bit


class CubefulEncoder(Encoder):
    """Base encoder plus 4 cube features: one-hot
    [centered, mine, theirs] (on-roll player's view, from
    `modes.cube_perspective()`) and is_cube_action (1 at the cube
    decision, 0 at the checker decision).
    """

    def __init__(self, base_encoder_name: str = "perspective196"):
        self._base = ENCODERS[base_encoder_name]()
        self.base_name = base_encoder_name

    @property
    def name(self) -> str:
        return f"cubeful_{self.base_name}"

    @property
    def cube_features(self) -> int:
        """Cube columns appended after the base features."""
        return CUBE_FEATURES

    @property
    def num_features(self) -> int:
        return self._base.num_features + self.cube_features

    def encode(
        self, state: BoardState, cube_state: CubePerspective,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        return self.encode_with_base(
            self._base.encode(state), cube_state, is_cube_action,
        )

    def encode_with_base(
        self, base_features: np.ndarray, cube_state: CubePerspective,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        """Append cube features to a precomputed base encoding (e.g.
        from the C encoder)."""
        x = np.empty(self.num_features, dtype=np.float32)
        n = base_features.shape[0]
        x[:n] = base_features
        x[n:] = 0.0
        x[n + int(cube_state)] = 1.0
        if is_cube_action:
            x[n + 3] = 1.0
        return x


ENCODERS = {
    "perspective196": Perspective196Encoder,
}


def get_encoder(name: str) -> Encoder:
    """New encoder instance by name; `cubeful_<base>` wraps the base
    in a `CubefulEncoder`."""
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


# ── Module-level API (default encoder) ───────────────────────────────────────

_default_encoder = Perspective196Encoder()

NUM_FEATURES = _default_encoder.num_features  # 196


def encode_state(state: BoardState) -> np.ndarray:
    """Default (Perspective196) encoding of a board state."""
    return _default_encoder.encode(state)
