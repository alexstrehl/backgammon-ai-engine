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

    The raw-scalar cube_value feature lets the trunk underweight cube
    context: the same board under cube 1 / 2 / 4 maps to nearly
    identical activations, and learned values alias across cube
    states (diagnosed via train_from_position.py: v_nd == v_dt at
    positions where their true values differ by 0.1+ MWC). One-hot
    slots give each cube level its own input column so gradients can
    decouple them. Cube values are always powers of two; slot 3
    absorbs >= 8 (cube 8 is already dead at any score it can occur
    at in short matches).

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

    Away scores are ONE-HOT (64 slots per side) instead of raw
    scalars: a new away level is a fresh input column rather than an
    out-of-distribution scalar value, so bumping match length never
    forces linear extrapolation and never disturbs the learned
    meaning of lower aways. The cube one-hot is widened to 7 slots
    (1/2/4/8/16/32/>=64) -- the old >=8 bucket aliased cube 8 vs 16,
    which are genuinely different states once someone is more than
    8-away. post_crawford is exposed as an explicit input; it is
    derivable ((someone 1-away) and not crawford) but sits on a
    discrete strategy regime (mandatory trailer double, free drop)
    that should not have to be inferred through a scalar threshold.

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


class Matchplay1H64MET6Encoder(Matchplay1H64METEncoder):
    """Matchplay1H64METEncoder with the MET block widened 13 -> 17.

    met4 without its two exact duplicates. met4 appended all six 4x-cube
    prices, but x4_win_n and x4_lose_n price 4C, which dt_win_g and
    dt_lose_g already price -- the block is a function of POINTS STAKED,
    not cube level, so a gammon at 2C and a single at 4C are the same
    lookup. Identical over all 1,694 reachable (away, away, crawford,
    cube) states at 11pt; the 19-wide met4 block is rank 13 and this
    17-wide one is rank 13, so nothing is lost.

    Kept: the 8C and 12C prices, win and lose. The 12C pair is constant
    at 11pt (a backgammon at 4x cube is 12+ points, which always decides
    an 11-point match) but becomes informative at longer match lengths.

    The first 350 columns are bit-identical to matchplay1h64met, so a
    350-input net expands by zero-padding with exact function
    preservation (surgery_1h64met6.py). 141 + 17 = 158 extras; 354
    inputs over perspective196."""

    @property
    def name(self) -> str:
        return f"matchplay1h64met6_{self.base_name}"

    @property
    def num_features(self) -> int:
        from met import N_MET6_FEATURES
        if not hasattr(self, "_nf6"):
            self._nf6 = (self._base.num_features + MATCHPLAY_1H64_FEATURES
                         + N_MET6_FEATURES)
        return self._nf6

    def match_extras(
        self, cube_state: CubePerspective, cube_value: int,
        my_away: int, opp_away: int, is_crawford: bool,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        base = MatchplayCube1H64Encoder.match_extras(
            self, cube_state, cube_value, my_away, opp_away, is_crawford,
            is_cube_action,
        )
        crawford_done = (my_away == 1 or opp_away == 1)
        block = self.met.outcome_block_x6(
            my_away, opp_away, crawford_done, is_crawford, cube_value,
        )
        return np.concatenate([base, block - np.float32(0.5)])


class Matchplay1H64MET4Encoder(Matchplay1H64METEncoder):
    """Matchplay1H64METEncoder with the MET block widened 13 -> 19.

    Adds the six MWC-after prices at 4 x the current cube (win/lose x
    normal/gammon/backgammon), the level a double-and-recube reaches.
    The first 350 columns are bit-identical to matchplay1h64met, so a
    350-input net expands by zero-padding with exact function
    preservation (surgery_1h64met4.py). 141 + 19 = 160 extras; 356
    inputs over perspective196."""

    @property
    def name(self) -> str:
        return f"matchplay1h64met4_{self.base_name}"

    @property
    def num_features(self) -> int:
        from met import N_MET4_FEATURES
        if not hasattr(self, "_nf8"):
            self._nf8 = (self._base.num_features + MATCHPLAY_1H64_FEATURES
                         + N_MET4_FEATURES)
        return self._nf8

    def match_extras(
        self, cube_state: CubePerspective, cube_value: int,
        my_away: int, opp_away: int, is_crawford: bool,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        # Grandparent, not super(): we replace the 13-wide MET block
        # rather than appending to it.
        base = MatchplayCube1H64Encoder.match_extras(
            self, cube_state, cube_value, my_away, opp_away, is_crawford,
            is_cube_action,
        )
        crawford_done = (my_away == 1 or opp_away == 1)
        block = self.met.outcome_block_x4(
            my_away, opp_away, crawford_done, is_crawford, cube_value,
        )
        return np.concatenate([base, block - np.float32(0.5)])


class Matchplay1H64MET5Encoder(Matchplay1H64MET4Encoder):
    """Matchplay1H64MET4Encoder minus the dead lose_bg column: 355 inputs.

    MEASURED, over all 568 reachable (my_away, opp_away, crawford, cube)
    states at 11pt: the two backgammon prices at 4x cube are CONSTANT --
    win_bg is +0.5 and lose_bg is -0.5 in every single state, std exactly
    0.000000. At 4x cube a backgammon is 12+ points, which always decides
    an 11-point match, so the MWC pins at 1.0 / 0.0 and never moves. The
    pair is rank 1 and carries no information at this match length (the
    19-wide block has rank 17, not 19).

    They are not harmless. A constant input yields a constant gradient,
    so its weight ramps LINEARLY and never plateaus -- measured 0.099 per
    1M episodes across 3.4M episodes of 0-ply, with the two dead columns
    carrying the LARGEST weight of all six new ones, and still growing
    while the informative columns shrank. The net was spending them as
    extra bias.

    One copy is kept rather than both: at longer match lengths 12 points
    stops being decisive, the two columns separate, and the slot becomes
    real information. Dropping the LAST column keeps indices 0..354
    aligned with met4, so met4 -> met5 is a plain truncation and
    met -> met5 stays a zero-padded append (surgery_1h64met5.py).
    """

    @property
    def name(self) -> str:
        return f"matchplay1h64met5_{self.base_name}"

    @property
    def num_features(self) -> int:
        if not hasattr(self, "_nf5"):
            self._nf5 = super().num_features - 1
        return self._nf5

    def match_extras(
        self, cube_state: CubePerspective, cube_value: int,
        my_away: int, opp_away: int, is_crawford: bool,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        # Drop the trailing lose_bg price; everything before it is
        # bit-identical to met4.
        return super().match_extras(
            cube_state, cube_value, my_away, opp_away, is_crawford,
            is_cube_action,
        )[:-1]


class MatchplayMETEncoder(MatchplayCube1HEncoder):
    """MatchplayCube1HEncoder plus the match-equity consequences of
    every cube branch, read from a real MET.

    Features appended (24 total = the 11 above, then 13 more):
      [no-double]   MWC after win/lose x normal/gammon/backgammon
                    at the CURRENT cube value              6
      [double-take] the same at 2 x cube value             6
      [double-drop] MWC after collecting cube_value points 1

    Motivation: MWC factorises as sum_outcome P(outcome) x
    MWC_after(outcome). The first factor is a board function, the
    second a pure score/cube function — and an MLP approximates
    products badly. The net already receives everything needed to
    *derive* the second factor (away scores, cube one-hot, crawford),
    so these columns add no information; they make the price of each
    outcome directly available instead of recoverable, which is what
    a gammon-pricing decision actually multiplies against.

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
    """New encoder instance by name; `cubeful_<base>` wraps the base
    in a `CubefulEncoder`, `matchplay_<base>` (and the matchplay1h_,
    matchplay1h64_, matchplay1h64met*_, matchplaymet_ variants) in the
    matching match-play encoder."""
    if name.startswith("matchplaymet_"):
        base_name = name[len("matchplaymet_"):]
        if base_name not in ENCODERS:
            raise ValueError(
                f"Unknown base encoder: {base_name!r}. "
                f"Available: {sorted(ENCODERS)}"
            )
        return MatchplayMETEncoder(base_name)
    if name.startswith("matchplay1h64met5_"):
        base_name = name[len("matchplay1h64met5_"):]
        if base_name not in ENCODERS:
            raise ValueError(
                f"Unknown base encoder: {base_name!r}. "
                f"Available: {sorted(ENCODERS)}"
            )
        return Matchplay1H64MET5Encoder(base_name)
    if name.startswith("matchplay1h64met6_"):
        base_name = name[len("matchplay1h64met6_"):]
        if base_name not in ENCODERS:
            raise ValueError(
                f"Unknown base encoder: {base_name!r}. "
                f"Available: {sorted(ENCODERS)}"
            )
        return Matchplay1H64MET6Encoder(base_name)
    if name.startswith("matchplay1h64met4_"):
        base_name = name[len("matchplay1h64met4_"):]
        if base_name not in ENCODERS:
            raise ValueError(
                f"Unknown base encoder: {base_name!r}. "
                f"Available: {sorted(ENCODERS)}"
            )
        return Matchplay1H64MET4Encoder(base_name)
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
