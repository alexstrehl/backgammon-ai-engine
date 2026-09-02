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


CUBE_FEATURES = 4  # 3-hot cube perspective + 1 is_cube_action bit
CUBE_TAKE_FEATURES = 5  # CUBE_FEATURES + 1 is_cube_take_decision bit


class CubefulEncoder(Encoder):
    """Wraps a base encoder, appending cube state features.

    Features appended (4 total):
      [cube_centered, cube_mine, cube_theirs]  3  (one-hot)
      is_cube_action                         1  (binary)

    Cube perspective is the on-roll player's view (CENTERED / MINE /
    THEIRS), produced by `cube_perspective()` in modes.py from the
    absolute CubeOwner. is_cube_action=1 at the cube decision
    point (before doubling/not-doubling), 0 at the checker decision
    point.
    """

    def __init__(self, base_encoder_name: str = "perspective196"):
        self._base = ENCODERS[base_encoder_name]()
        self.base_name = base_encoder_name

    @property
    def name(self) -> str:
        return f"cubeful_{self.base_name}"

    @property
    def num_features(self) -> int:
        return self._base.num_features + CUBE_FEATURES

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
        """Append cube features to a precomputed base-encoder array.
        Lets callers reuse a fast (e.g. C) base encoding without
        re-running the Python base encoder.
        """
        x = np.empty(self.num_features, dtype=np.float32)
        n = base_features.shape[0]
        x[:n] = base_features
        x[n:] = 0.0
        x[n + int(cube_state)] = 1.0
        if is_cube_action:
            x[n + 3] = 1.0
        return x


class CubefulTakeEncoder(CubefulEncoder):
    """Cubeful encoder with an explicit take-decision flag.

    Inherits from CubefulEncoder so `isinstance(_, CubefulEncoder)` checks
    treat instances of this class as cubeful. Adds one binary feature
    `is_cube_take_decision` to distinguish the responder's take/pass
    decision state from the doubler's offer state and from the
    chequer-play state.

    Features appended (5 total):
      [cube_centered, cube_mine, cube_theirs]  3  (one-hot)
      is_cube_action                         1  (binary)
      is_cube_take_decision                  1  (binary)

    First 4 cube features (indices [n, n+3]) match CubefulEncoder
    bit-for-bit when `is_cube_take_decision=False`, so a CubefulEncoder
    checkpoint can be warm-started into this one by extending the
    input projection by a single zero-initialized column for the new
    bit (handled by the warm-start utility).

    States:
      (is_cube_action=0, is_cube_take_decision=0): checker decision
      (is_cube_action=1, is_cube_take_decision=0): doubler's offer
      (is_cube_action=0, is_cube_take_decision=1): responder's take/pass

    The (1, 1) combination is unused.
    """

    @property
    def name(self) -> str:
        return f"cubeful_take_{self.base_name}"

    @property
    def num_features(self) -> int:
        return self._base.num_features + CUBE_TAKE_FEATURES

    def encode(
        self, state: BoardState, cube_state: CubePerspective,
        is_cube_action: bool = False,
        is_cube_take_decision: bool = False,
    ) -> np.ndarray:
        return self.encode_with_base(
            self._base.encode(state), cube_state,
            is_cube_action, is_cube_take_decision,
        )

    def encode_with_base(
        self, base_features: np.ndarray, cube_state: CubePerspective,
        is_cube_action: bool = False,
        is_cube_take_decision: bool = False,
    ) -> np.ndarray:
        assert not (is_cube_action and is_cube_take_decision), \
            "is_cube_action and is_cube_take_decision are mutually exclusive"
        x = np.empty(self.num_features, dtype=np.float32)
        n = base_features.shape[0]
        x[:n] = base_features
        x[n:] = 0.0
        x[n + int(cube_state)] = 1.0
        if is_cube_action:
            x[n + 3] = 1.0
        if is_cube_take_decision:
            x[n + 4] = 1.0
        return x


MATCHPLAY_FEATURES = 8  # 3-hot cube + is_cube_action + cube_value + 2 away + crawford
MATCHPLAY_1H_FEATURES = 11  # 3-hot cube persp + is_cube_action + 4-hot cube value + 2 away + crawford
MATCHPLAY_1H64_FEATURES = 141  # 3-hot persp + ca + 7-hot cube + 64-hot away x2 + crawford + post_crawford


class MatchplayEncoder(Encoder):
    """Match-aware encoder: base board features + cube + match score.

    Standalone (NOT a CubefulEncoder subclass) so money-game cube
    paths in TDAgent — which assume equity output and per-unit money
    semantics — never trigger for matchplay agents.

    Features appended (8 total):
      [cube_centered, cube_mine, cube_theirs]  3  (one-hot, on-roll view)
      is_cube_action                           1  (binary)
      cube_value                               1  (absolute: 1, 2, 4, ...)
      my_away                                  1  (absolute, on-roll view)
      opp_away                                 1  (absolute, on-roll view)
      is_crawford                              1  (binary)

    The first 4 appended features match CubefulEncoder bit-for-bit, so
    a cubeful money checkpoint can also be warm-started into this
    layout later by zero-extending 4 columns.

    Away scores are ABSOLUTE (not normalized by match length) and
    fully determine the strategic situation for any match length —
    (my_away, opp_away, cube, crawford) is the sufficient statistic,
    so no match_length feature is needed and the same network can in
    principle play any length match. Normalizing scores by match
    length was tried previously and collapsed different lengths into
    indistinguishable inputs.

    Perspective: callers must pass away scores already flipped to the
    on-roll player's view (TDAgent._encode_matchplay does this).
    """

    def __init__(self, base_encoder_name: str = "perspective196"):
        self._base = ENCODERS[base_encoder_name]()
        self.base_name = base_encoder_name

    @property
    def name(self) -> str:
        return f"matchplay_{self.base_name}"

    @property
    def num_features(self) -> int:
        return self._base.num_features + MATCHPLAY_FEATURES

    def encode(
        self, state: BoardState, cube_state: CubePerspective,
        cube_value: int, my_away: int, opp_away: int, is_crawford: bool,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        return self.encode_with_base(
            self._base.encode(state), cube_state, cube_value,
            my_away, opp_away, is_crawford, is_cube_action,
        )

    def match_extras(
        self, cube_state: CubePerspective, cube_value: int,
        my_away: int, opp_away: int, is_crawford: bool,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        """The appended match-feature block as a standalone vector —
        batched callers tile this next to precomputed base features."""
        x = np.zeros(MATCHPLAY_FEATURES, dtype=np.float32)
        x[int(cube_state)] = 1.0
        if is_cube_action:
            x[3] = 1.0
        x[4] = float(cube_value)
        x[5] = float(my_away)
        x[6] = float(opp_away)
        x[7] = 1.0 if is_crawford else 0.0
        return x

    def encode_with_base(
        self, base_features: np.ndarray, cube_state: CubePerspective,
        cube_value: int, my_away: int, opp_away: int, is_crawford: bool,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        n = base_features.shape[0]
        x = np.empty(self.num_features, dtype=np.float32)
        x[:n] = base_features
        x[n:] = self.match_extras(
            cube_state, cube_value, my_away, opp_away, is_crawford,
            is_cube_action,
        )
        return x


class MatchplayCube1HEncoder(MatchplayEncoder):
    """MatchplayEncoder variant with the cube VALUE one-hot encoded.

    A raw-scalar cube_value lets learned values alias across cube
    states (v_nd == v_dt at positions whose true values differ by
    0.1+ MWC); one-hot slots give each cube level its own column.
    Cube values are always powers of two; slot 3 absorbs >= 8 (cube 8
    is already dead at any score it can occur at in short matches).

    Features appended (11 total):
      [cube_centered, cube_mine, cube_theirs]   3  (one-hot)
      is_cube_action                            1  (binary)
      [cube==1, cube==2, cube==4, cube>=8]      4  (one-hot)
      my_away                                   1  (absolute)
      opp_away                                  1  (absolute)
      is_crawford                               1  (binary)

    Function-preserving warm start from MatchplayEncoder nets: set
    each one-hot column's input weights to (old cube_value column
    weights) * 2^k — identical outputs at every reachable cube.
    """

    @property
    def name(self) -> str:
        return f"matchplay1h_{self.base_name}"

    @property
    def num_features(self) -> int:
        return self._base.num_features + MATCHPLAY_1H_FEATURES

    @staticmethod
    def _cube_slot(cube_value: int) -> int:
        if cube_value >= 8:
            return 3
        return {1: 0, 2: 1, 4: 2}[cube_value]

    def match_extras(
        self, cube_state: CubePerspective, cube_value: int,
        my_away: int, opp_away: int, is_crawford: bool,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        x = np.zeros(MATCHPLAY_1H_FEATURES, dtype=np.float32)
        x[int(cube_state)] = 1.0
        if is_cube_action:
            x[3] = 1.0
        x[4 + self._cube_slot(cube_value)] = 1.0
        x[8] = float(my_away)
        x[9] = float(opp_away)
        x[10] = 1.0 if is_crawford else 0.0
        return x

    def encode_with_base(
        self, base_features: np.ndarray, cube_state: CubePerspective,
        cube_value: int, my_away: int, opp_away: int, is_crawford: bool,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        n = base_features.shape[0]
        x = np.empty(self.num_features, dtype=np.float32)
        x[:n] = base_features
        x[n:] = self.match_extras(
            cube_state, cube_value, my_away, opp_away, is_crawford,
            is_cube_action,
        )
        return x


class MatchplayCube1H64Encoder(MatchplayCube1HEncoder):
    """Long-match variant of the 1-hot encoder, sized for matches up
    to 64 points (nobody plays longer than 50).

    Away scores are ONE-HOT (64 slots per side): a new away level is
    a fresh input column, so bumping match length never forces
    extrapolation or disturbs lower aways. Cube one-hot widened to 7
    slots (1/2/4/8/16/32/>=64; the old >=8 bucket aliased cube 8 vs
    16). post_crawford is explicit — derivable, but a discrete
    strategy regime (mandatory trailer double, free drop).

    Extras layout (141): persp 0-2 | is_cube_action 3 | cube 4-10 |
    my_away 11-74 | opp_away 75-138 | crawford 139 | post_crawford 140.
    """

    @property
    def name(self) -> str:
        return f"matchplay1h64_{self.base_name}"

    @property
    def num_features(self) -> int:
        return self._base.num_features + MATCHPLAY_1H64_FEATURES

    @staticmethod
    def _cube_slot(cube_value: int) -> int:
        if cube_value >= 64:
            return 6
        return {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[cube_value]

    def match_extras(
        self, cube_state: CubePerspective, cube_value: int,
        my_away: int, opp_away: int, is_crawford: bool,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        assert my_away >= 1 and opp_away >= 1, (
            "encoding a finished match: callers must check match_over() "
            f"first (my_away={my_away}, opp_away={opp_away})")
        x = np.zeros(MATCHPLAY_1H64_FEATURES, dtype=np.float32)
        x[int(cube_state)] = 1.0
        if is_cube_action:
            x[3] = 1.0
        x[4 + self._cube_slot(cube_value)] = 1.0
        x[11 + min(int(my_away), 64) - 1] = 1.0
        x[75 + min(int(opp_away), 64) - 1] = 1.0
        if is_crawford:
            x[139] = 1.0
        if (my_away == 1 or opp_away == 1) and not is_crawford:
            x[140] = 1.0
        return x


class Matchplay1H64METEncoder(MatchplayCube1H64Encoder):
    """MatchplayCube1H64Encoder plus the 13 MET outcome-price columns
    of MatchplayMETEncoder (see its docstring for the factorisation
    rationale): MWC-after for win/lose x normal/gammon/backgammon at
    the current cube (6) and the doubled cube (6), plus the
    double-drop value (1), all centred at 0 (mwc - 0.5). 141 + 13 =
    154 extras; 350 inputs over perspective196."""

    def __init__(self, base_encoder_name: str = "perspective196", met=None):
        super().__init__(base_encoder_name)
        if met is None:
            from met import default_met
            met = default_met()
        self.met = met

    @property
    def name(self) -> str:
        return f"matchplay1h64met_{self.base_name}"

    @property
    def num_features(self) -> int:
        from met import N_MET_FEATURES  # lazy: met imports at first use
        if not hasattr(self, "_nf"):
            self._nf = (self._base.num_features + MATCHPLAY_1H64_FEATURES
                        + N_MET_FEATURES)
        return self._nf

    def match_extras(
        self, cube_state: CubePerspective, cube_value: int,
        my_away: int, opp_away: int, is_crawford: bool,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        base = super().match_extras(
            cube_state, cube_value, my_away, opp_away, is_crawford,
            is_cube_action,
        )
        crawford_done = (my_away == 1 or opp_away == 1)
        block = self.met.outcome_block(
            my_away, opp_away, crawford_done, is_crawford, cube_value,
        )
        return np.concatenate([base, block - np.float32(0.5)])


class MatchplayMETEncoder(MatchplayCube1HEncoder):
    """MatchplayCube1HEncoder plus the match-equity consequences of
    every cube branch, read from a real MET.

    Features appended (24 total = the 11 above, then 13 more):
      [no-double]   MWC after win/lose x normal/gammon/backgammon
                    at the CURRENT cube value              6
      [double-take] the same at 2 x cube value             6
      [double-drop] MWC after collecting cube_value points 1

    Motivation: MWC factorises as sum_outcome P(outcome) x
    MWC_after(outcome); the second factor is a pure score/cube
    function that these columns supply directly instead of making the
    MLP derive it.

    Values are centred (mwc - 0.5) so a neutral score contributes
    zero. Expect heavy saturation in short matches: at 3 points with
    cube 2 most win branches already end the match.

    `crawford_done` is not a caller-supplied feature; it is exactly
    "someone is 1-away" (away scores never increase, and MatchState
    latches the flag when start_game first sees a 1-away score), so
    it is derived here. tests/test_met_encoder.py checks that
    derivation against MatchState over all reachable transitions.

    `is_crawford` is accepted for signature parity but provably does
    not affect the block: every value is a FUTURE-game MWC, and by the
    time any of those games starts Crawford is necessarily spent. The
    Crawford game itself is still priced correctly, via the transition
    INTO a 1-away score (3-away/3-away + gammon -> 1-away/3-away picks
    up the pre-Crawford value).
    """

    def __init__(self, base_encoder_name: str = "perspective196", met=None):
        super().__init__(base_encoder_name)
        if met is None:
            from met import default_met
            met = default_met()
        self.met = met

    @property
    def name(self) -> str:
        return f"matchplaymet_{self.base_name}"

    @property
    def num_features(self) -> int:
        from met import N_MET_FEATURES
        return (self._base.num_features + MATCHPLAY_1H_FEATURES
                + N_MET_FEATURES)

    def match_extras(
        self, cube_state: CubePerspective, cube_value: int,
        my_away: int, opp_away: int, is_crawford: bool,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        base = super().match_extras(
            cube_state, cube_value, my_away, opp_away, is_crawford,
            is_cube_action,
        )
        crawford_done = (my_away == 1 or opp_away == 1)
        block = self.met.outcome_block(
            my_away, opp_away, crawford_done, is_crawford, cube_value,
        )
        return np.concatenate([base, block - np.float32(0.5)])


ENCODERS = {
    "perspective196": Perspective196Encoder,
}


def get_encoder(name: str) -> Encoder:
    """Look up an encoder by name and return a new instance.

    Special prefixes:
      `cubeful_<base>`      → CubefulEncoder over the named base.
      `cubeful_take_<base>` → CubefulTakeEncoder over the named base
                              (adds is_cube_take_decision flag).
      `matchplay_<base>`    → MatchplayEncoder over the named base
                              (cube + away scores + crawford).
    """
    if name.startswith("matchplaymet_"):
        base_name = name[len("matchplaymet_"):]
        if base_name not in ENCODERS:
            raise ValueError(
                f"Unknown base encoder: {base_name!r}. "
                f"Available: {sorted(ENCODERS)}"
            )
        return MatchplayMETEncoder(base_name)
    if name.startswith("matchplay1h64met_"):
        base_name = name[len("matchplay1h64met_"):]
        if base_name not in ENCODERS:
            raise ValueError(
                f"Unknown base encoder: {base_name!r}. "
                f"Available: {sorted(ENCODERS)}"
            )
        return Matchplay1H64METEncoder(base_name)
    if name.startswith("matchplay1h64_"):
        base_name = name[len("matchplay1h64_"):]
        if base_name not in ENCODERS:
            raise ValueError(
                f"Unknown base encoder: {base_name!r}. "
                f"Available: {sorted(ENCODERS)}"
            )
        return MatchplayCube1H64Encoder(base_name)
    if name.startswith("matchplay1h_"):
        base_name = name[len("matchplay1h_"):]
        if base_name not in ENCODERS:
            raise ValueError(
                f"Unknown base encoder: {base_name!r}. "
                f"Available: {sorted(ENCODERS)}"
            )
        return MatchplayCube1HEncoder(base_name)
    if name.startswith("matchplay_"):
        base_name = name[len("matchplay_"):]
        if base_name not in ENCODERS:
            raise ValueError(
                f"Unknown base encoder: {base_name!r}. "
                f"Available: {sorted(ENCODERS)}"
            )
        return MatchplayEncoder(base_name)
    if name.startswith("cubeful_take_"):
        base_name = name[len("cubeful_take_"):]
        if base_name not in ENCODERS:
            raise ValueError(
                f"Unknown base encoder: {base_name!r}. "
                f"Available: {sorted(ENCODERS)}"
            )
        return CubefulTakeEncoder(base_name)
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
