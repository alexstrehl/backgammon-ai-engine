"""gnubg_ids.py -- gnubg Position ID / Match ID encoders.

Faithful ports of gnubg 1.08.003 `positionid.c` (oldPositionKey /
oldPositionID) and `matchid.c` (MatchID). Both formats are
little-endian bit streams packed into bytes LSB-first, then base64'd
with the standard alphabet (no padding).

Used to drive the real gnubg CLI (`set matchid` / `set board`) for
position-level cross-validation -- e.g. feeding gnubg's own printed
0-ply probabilities through our Cl2CfMatch port (janowski_match.py)
and comparing cubeful MWCs -- and as a debugging aid: any (state,
match_state) pair in our framework can be printed as the two IDs and
pasted straight into gnubg or XG for inspection.
"""

from __future__ import annotations

from typing import Optional, Tuple

from backgammon_engine import BoardState, NUM_POINTS

_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


class _BitWriter:
    """Little-endian bit stream: bit i of a value lands at stream
    position (pos + i); stream bit j lives in byte j//8, bit j%8."""

    def __init__(self, nbytes: int):
        self.buf = bytearray(nbytes)

    def set_bits(self, pos: int, nbits: int, value: int) -> None:
        for i in range(nbits):
            if (value >> i) & 1:
                j = pos + i
                self.buf[j // 8] |= 1 << (j % 8)

    def base64(self) -> str:
        out = []
        b = self.buf
        full, rem = divmod(len(b), 3)
        for g in range(full):
            p = b[3 * g: 3 * g + 3]
            out.append(_B64[p[0] >> 2])
            out.append(_B64[((p[0] & 0x03) << 4) | (p[1] >> 4)])
            out.append(_B64[((p[1] & 0x0F) << 2) | (p[2] >> 6)])
            out.append(_B64[p[2] & 0x3F])
        if rem == 1:
            p = b[3 * full]
            out.append(_B64[p >> 2])
            out.append(_B64[(p & 0x03) << 4])
        elif rem == 2:
            p, q = b[3 * full], b[3 * full + 1]
            out.append(_B64[p >> 2])
            out.append(_B64[((p & 0x03) << 4) | (q >> 4)])
            out.append(_B64[(q & 0x0F) << 2])
        return "".join(out)


def _tan_board(state: BoardState) -> Tuple[list, list]:
    """(opponent_row, roller_row): 25 counts each, points 1..24 from
    that player's own numbering plus the bar at index 24 -- gnubg's
    TanBoard with anBoard[1] = the player on roll."""
    rows = []
    for player in (1 - state.turn, state.turn):
        row = [0] * 25
        for p in range(1, NUM_POINTS + 1):
            idx = (p - 1) if player == 0 else (NUM_POINTS - p)
            v = state.points[idx]
            n = v if player == 0 else -v
            if n > 0:
                row[p - 1] = n
        row[24] = state.bar[player]
        rows.append(row)
    return rows[0], rows[1]


def position_id(state: BoardState) -> str:
    """gnubg Position ID (14 chars) for `state`, relative to the
    player on roll (`state.turn`) -- gnubg's convention."""
    opp, me = _tan_board(state)
    w = _BitWriter(10)
    pos = 0
    for row in (opp, me):
        for nc in row:
            if nc:
                w.set_bits(pos, nc, (1 << nc) - 1)
                pos += nc + 1
            else:
                pos += 1
    return w.base64()


def match_id(
    cube_value: int,
    cube_owner: int,          # 0 / 1 player index, -1 for centered
    on_roll: int,             # fMove = fTurn
    crawford: bool,
    match_to: int,
    score0: int,
    score1: int,
    dice: Optional[Tuple[int, int]] = None,
    doubled: bool = False,
    game_state: int = 1,      # 1 = GAME_PLAYING
) -> str:
    """gnubg Match ID (12 chars). `dice=None` encodes the pre-roll
    cube-decision point (both dice 0). Scores are points SCORED (not
    away). Jacoby is encoded as off (match play)."""
    w = _BitWriter(9)
    w.set_bits(0, 4, cube_value.bit_length() - 1)      # LogCube
    w.set_bits(4, 2, cube_owner & 0x3)
    w.set_bits(6, 1, on_roll)
    w.set_bits(7, 1, int(crawford))
    w.set_bits(8, 3, game_state)
    w.set_bits(11, 1, on_roll)                         # fTurn
    w.set_bits(12, 1, int(doubled))
    w.set_bits(13, 2, 0)                               # fResigned
    if dice is not None:
        hi, lo = max(dice), min(dice)
        w.set_bits(15, 3, hi & 0x7)
        w.set_bits(18, 3, lo & 0x7)
    w.set_bits(21, 15, match_to & 0x7FFF)
    w.set_bits(36, 15, score0 & 0x7FFF)
    w.set_bits(51, 15, score1 & 0x7FFF)
    w.set_bits(66, 1, 1)                               # !fJacobyInUse
    return w.base64()


def ids_for(state: BoardState, match_state, match_length: int) -> Tuple[str, str]:
    """(position_id, match_id) for a pre-roll cube-decision point at
    `state` under `match_state` (modes.MatchState, WHITE-view away
    scores). gnubg player 0 <-> our WHITE."""
    from modes import CubeOwner
    owner = {CubeOwner.CENTERED: -1, CubeOwner.WHITE: 0,
             CubeOwner.BLACK: 1}[match_state.cube_owner]
    return (
        position_id(state),
        match_id(
            cube_value=match_state.cube_value,
            cube_owner=owner,
            on_roll=state.turn,
            crawford=match_state.is_crawford,
            match_to=match_length,
            score0=match_length - match_state.my_away,
            score1=match_length - match_state.opp_away,
        ),
    )
