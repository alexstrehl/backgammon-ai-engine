"""
backgammon_engine.py — Board representation and move generation for backgammon.

Board layout (index 0 = point 1, index 23 = point 24):

    Point:  13 14 15 16 17 18    19 20 21 22 23 24
            ----- BAR -----
    Point:  12 11 10  9  8  7     6  5  4  3  2  1

    WHITE (positive) moves high→low  (24 → 1), bears off past point 1.
    BLACK (negative) moves low→high  (1 → 24), bears off past point 24.

Internal storage:
    points : list[int]  — 24 entries; >0 = WHITE checkers, <0 = BLACK
    bar    : list[int]  — [white_on_bar, black_on_bar]
    off    : list[int]  — [white_borne_off, black_borne_off]
    turn   : int        — WHITE (0) or BLACK (1)

`BoardState.initial()` returns the start position with WHITE to move.
`opening_roll(rng)` draws a real (non-doublet) opening roll and sets
who moves first.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

WHITE = 0
BLACK = 1

NUM_POINTS = 24
NUM_CHECKERS = 15

BAR = -1   # sentinel for "from bar"
OFF = -2   # sentinel for "to off (bear-off)"

Move = Tuple[int, int]          # (source, destination)  — BAR / OFF as sentinels
Play = Tuple[Move, ...]         # a full turn (sequence of single moves)


# ── Board state ──────────────────────────────────────────────────────────────

class BoardState:
    """Immutable-style backgammon board (call copy() before mutating).

    `.turn` = the player to act next. `get_legal_plays` returns
    successors already turn-switched, so at a terminal state `.turn`
    is the LOSER.
    """

    __slots__ = ("points", "bar", "off", "turn")

    def __init__(
        self,
        points: Optional[List[int]] = None,
        bar: Optional[List[int]] = None,
        off: Optional[List[int]] = None,
        turn: int = WHITE,
    ):
        self.points: List[int] = points if points is not None else [0] * 24
        self.bar: List[int] = bar if bar is not None else [0, 0]
        self.off: List[int] = off if off is not None else [0, 0]
        self.turn: int = turn

    # ── Factories ────────────────────────────────────────────────────────

    @classmethod
    def initial(cls) -> BoardState:
        """Standard backgammon starting position, WHITE to move."""
        pts = [0] * 24
        # WHITE checkers (positive)
        pts[23] = 2    # point 24
        pts[12] = 5    # point 13
        pts[7]  = 3    # point 8
        pts[5]  = 5    # point 6
        # BLACK checkers (negative)
        pts[0]  = -2   # point 1
        pts[11] = -5   # point 12
        pts[16] = -3   # point 17
        pts[18] = -5   # point 19
        return cls(points=pts, bar=[0, 0], off=[0, 0], turn=WHITE)

    # ── Copying & hashing ────────────────────────────────────────────────

    def copy(self) -> BoardState:
        return BoardState(
            points=self.points[:],
            bar=self.bar[:],
            off=self.off[:],
            turn=self.turn,
        )

    def _key(self) -> tuple:
        return (tuple(self.points), tuple(self.bar), tuple(self.off), self.turn)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoardState):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    # ── Queries ──────────────────────────────────────────────────────────

    def is_game_over(self) -> bool:
        return self.off[WHITE] == NUM_CHECKERS or self.off[BLACK] == NUM_CHECKERS

    def winner(self) -> Optional[int]:
        if self.off[WHITE] == NUM_CHECKERS:
            return WHITE
        if self.off[BLACK] == NUM_CHECKERS:
            return BLACK
        return None

    def game_result(self) -> Optional[int]:
        """Return 1 (normal), 2 (gammon), or 3 (backgammon) for the winner,
        or None if the game is not over. Value is from the *winner*'s POV."""
        w = self.winner()
        if w is None:
            return None
        loser = 1 - w
        if self.off[loser] > 0:
            return 1  # normal
        # Gammon: loser has borne off nothing
        # Check for backgammon: loser has checkers on bar or in winner's home
        if loser == WHITE:
            loser_in_opp_home = any(self.points[i] > 0 for i in range(18, 24))
        else:
            loser_in_opp_home = any(self.points[i] < 0 for i in range(0, 6))
        if self.bar[loser] > 0 or loser_in_opp_home:
            return 3  # backgammon
        return 2  # gammon

    # ── Display ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"BoardState(turn={'W' if self.turn == WHITE else 'B'}, "
            f"bar={self.bar}, off={self.off})"
        )



# ── Move generation ──────────────────────────────────────────────────────────

def _single_moves(state: BoardState, die: int) -> List[Move]:
    """All legal single-checker moves for *state.turn* using one die.
    Hot path: reads state.points directly, no helper calls."""
    player = state.turn
    pts = state.points
    moves: List[Move] = []

    # ── Must enter from bar first ────────────────────────────────────
    if state.bar[player] > 0:
        target = (24 - die) if player == WHITE else (die - 1)

        # Blocked if the opponent holds 2+ checkers there.
        v = pts[target]
        opp_count = (-v) if player == WHITE else v
        if opp_count < 2:
            moves.append((BAR, target))
        return moves  # while on bar, no other moves allowed

    # ── Regular moves and bearing off ────────────────────────────────
    #
    # WHITE: checkers positive, moves toward index 0.
    # BLACK: checkers negative, moves toward index 23.
    is_white = (player == WHITE)
    direction = -1 if is_white else 1

    # May bear off only with every checker in the home board.
    can_bear_off = (state.bar[player] == 0)
    if can_bear_off:
        if is_white:
            for i in range(6, 24):
                if pts[i] > 0:
                    can_bear_off = False
                    break
        else:
            for i in range(0, 18):
                if pts[i] < 0:
                    can_bear_off = False
                    break

    # Farthest checker from the edge (for the over-bear rule).
    farthest = -1
    if can_bear_off:
        if is_white:
            for i in range(5, -1, -1):  # 5 down to 0
                if pts[i] > 0:
                    farthest = i
                    break
        else:
            for i in range(18, 24):     # 18 up to 23
                if pts[i] < 0:
                    farthest = i
                    break

    for src in range(24):
        v = pts[src]
        # Skip points without our checkers.
        if is_white:
            if v <= 0:
                continue
        else:
            if v >= 0:
                continue

        target = src + direction * die

        # Normal move (target on the board)
        if 0 <= target <= 23:
            tv = pts[target]
            opp_count = (-tv) if is_white else tv
            if opp_count < 2:
                moves.append((src, target))
            continue

        # Bear-off (target off the board): exact die, or a larger die
        # from the farthest checker.
        if not can_bear_off:
            continue

        d = (src + 1) if is_white else (24 - src)
        if die == d:
            moves.append((src, OFF))
        elif die > d and src == farthest:
            moves.append((src, OFF))

    return moves


def _apply_move(state: BoardState, move: Move) -> BoardState:
    """Apply a single move, returning a NEW BoardState. Handles hits.
    Hot path: everything inlined."""
    pts = state.points[:]
    bar = state.bar[:]
    off = state.off[:]
    player = state.turn
    src, dst = move

    sign = 1 if player == WHITE else -1

    if src == BAR:
        bar[player] -= 1
    else:
        pts[src] -= sign

    if dst == OFF:
        off[player] += 1
    else:
        # Hit: a lone opposing checker goes to the bar.
        tv = pts[dst]
        opp_count = (-tv) if player == WHITE else tv
        if opp_count == 1:
            pts[dst] += sign
            bar[1 - player] += 1
        pts[dst] += sign

    return BoardState(pts, bar, off, state.turn)


def _generate_plays(
    state: BoardState,
    remaining_dice: Tuple[int, ...],
    current_play: Play,
    results: dict,              # MUTATED: board key -> (play, state)
    dice_used: int,
    max_used: List[int],        # MUTATED: [most dice any play used]
):
    """Recursively build all legal plays into `results`, keyed by board
    state so move orderings that reach the same position collapse to
    one entry."""
    found = False

    # Try each die value once per level (doubles would otherwise
    # explore the same subtree 4 times).
    seen_this_level: set = set()

    for i, die in enumerate(remaining_dice):
        if die in seen_this_level:
            continue
        seen_this_level.add(die)

        new_remaining_dice = remaining_dice[:i] + remaining_dice[i + 1:]

        for move in _single_moves(state, die):
            found = True
            new_state = _apply_move(state, move)
            new_play = current_play + (move,)
            _generate_plays(
                new_state, new_remaining_dice, new_play, results,
                dice_used + 1, max_used,
            )

    if not found:
        if dice_used > max_used[0]:
            max_used[0] = dice_used
        key = state._key()
        # On a collision keep the play using MORE dice, or the max-dice
        # filter drops the position (a false forced pass).
        prev = results.get(key)
        if prev is None or len(current_play) > len(prev[0]):
            results[key] = (current_play, state)


def get_legal_plays(
    state: BoardState, dice: Tuple[int, int]
) -> List[Tuple[Play, BoardState]]:
    """Return every legal play for *state.turn* given *dice*, as
    ``(play, resulting_state)`` pairs. *play* is a tuple of ``(src, dst)``
    moves. ``resulting_state.turn`` is already the opponent, so callers
    must NOT switch_turn again. Deduplicated by resulting board.

    Rules enforced:
    * All four sub-moves for doubles.
    * Must use both dice if possible; if only one, must use the larger.
    * Checkers on the bar must enter before any other move.
    * Bearing off respects exact / over-bear rules.
    """
    d1, d2 = dice
    if d1 == d2:
        remaining_dice = (d1, d1, d1, d1)
    else:
        remaining_dice = (d1, d2)

    results: dict = {}
    max_used: List[int] = [0]

    _generate_plays(state, remaining_dice, (), results, 0, max_used)

    # Must use as many dice as possible. 0 = forced pass.
    most_dice_used = max_used[0]
    if most_dice_used == 0:
        return []
    filtered = [
        (play, s)
        for play, s in results.values()
        if len(play) == most_dice_used
    ]

    # Non-doubles with only one die playable: must use the larger.
    if d1 != d2 and most_dice_used == 1:
        big = max(d1, d2)
        uses_big = [(p, s) for p, s in filtered if _play_uses_die(p, state, big)]
        if uses_big:
            filtered = uses_big

    return [(p, switch_turn(s)) for p, s in filtered]


def get_legal_plays_encoded(state: BoardState, dice, encoder=None):
    """Move generation + perspective encoding, same signature as
    `bg_fast.get_legal_plays_encoded`.

    Returns `(features, next_states)`:
      - features: (N, num_features) float32
      - next_states: list of N BoardState (turn already switched)
    """
    from encoding import get_encoder  # circular import otherwise
    if encoder is None:
        encoder = get_encoder("perspective196")
    plays = get_legal_plays(state, dice)
    if not plays:
        return np.empty((0, encoder.num_features), dtype=np.float32), []
    next_states = [s for _, s in plays]
    features = np.stack([encoder.encode(s) for s in next_states])
    return features, next_states


def _play_uses_die(play: Play, original: BoardState, die: int) -> bool:
    """True if the single-move *play* uses *die*."""
    if len(play) != 1:
        return False
    src, dst = play[0]
    player = original.turn
    direction = -1 if player == WHITE else 1

    if src == BAR:
        if player == WHITE:
            expected_dst = 24 - die
        else:
            expected_dst = die - 1
        return dst == expected_dst

    if dst == OFF:
        if player == WHITE:
            d = src + 1
        else:
            d = 24 - src
        if die == d:
            return True
        # Over-bear: only from the farthest checker.
        if die > d:
            pts = original.points
            if player == WHITE:
                farthest = max((i for i in range(6) if pts[i] > 0), default=-1)
            else:
                farthest = min((i for i in range(18, 24) if pts[i] < 0), default=24)
            return src == farthest
        return False
    else:
        return dst == src + direction * die


# ── Convenience helpers ──────────────────────────────────────────────────────

def switch_turn(state: BoardState) -> BoardState:
    """Return a copy with the turn flipped."""
    s = state.copy()
    s.turn = 1 - s.turn
    return s


def pip_count(state: BoardState) -> Tuple[int, int]:
    """(white_pips, black_pips). WHITE bears off past index 0 (pip = i+1),
    BLACK past index 23 (pip = 24-i). A checker on the bar counts 25."""
    wp = 25 * state.bar[WHITE]
    bp = 25 * state.bar[BLACK]
    for i, v in enumerate(state.points):
        if v > 0:
            wp += v * (i + 1)
        elif v < 0:
            bp += (-v) * (NUM_POINTS - i)
    return wp, bp


def is_race(state: BoardState) -> bool:
    """True when no contact is possible: nobody on the bar and every
    WHITE checker (moving toward index 0) below every BLACK checker."""
    if state.bar[WHITE] or state.bar[BLACK]:
        return False
    w_max = -1
    b_min = NUM_POINTS
    for i, v in enumerate(state.points):
        if v > 0:
            w_max = i
        elif v < 0 and b_min == NUM_POINTS:
            b_min = i
    return w_max < b_min


def move_label(move: Move) -> str:
    """Human-readable label for a single move, e.g. '13/7' or 'bar/20'."""
    src, dst = move
    src_str = "bar" if src == BAR else str(src + 1)
    dst_str = "off" if dst == OFF else str(dst + 1)
    return f"{src_str}/{dst_str}"


def play_label(play: Play) -> str:
    """Human-readable label for a full play."""
    return "  ".join(move_label(m) for m in play)


def opening_roll(rng=None) -> Tuple[BoardState, Tuple[int, int]]:
    """Opening roll: each player rolls one die, the higher wins and
    plays both. Doublets are rerolled.

    `rng` needs `randint(a, b)` (default: the `random` module).
    Returns `(state, (high, low))` with `state.turn` = the winner.
    """
    if rng is None:
        import random as _random
        rng = _random
    while True:
        white_die = rng.randint(1, 6)
        black_die = rng.randint(1, 6)
        if white_die != black_die:
            break
    state = BoardState.initial()
    state.turn = WHITE if white_die > black_die else BLACK
    high, low = max(white_die, black_die), min(white_die, black_die)
    return state, (high, low)

