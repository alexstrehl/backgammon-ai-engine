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

`BoardState.initial()` returns a position with WHITE to move. For
self-play training and matchplay where the opening must follow real
backgammon rules, use `opening_roll(rng)` to draw a non-doublet
opening and determine which player goes first.
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

    def checker_count(self, player: int, idx: int) -> int:
        """Number of *player*'s checkers on point index *idx* (0-based)."""
        v = self.points[idx]
        if player == WHITE:
            return max(v, 0)
        else:
            return max(-v, 0)

    def has_checker(self, player: int, idx: int) -> bool:
        return self.checker_count(player, idx) > 0

    def is_blocked(self, player: int, idx: int) -> bool:
        """True if the opponent has 2+ checkers on *idx* (blocked for *player*)."""
        opp = 1 - player
        return self.checker_count(opp, idx) >= 2

    def all_in_home(self, player: int) -> bool:
        """True if all of *player*'s remaining_dice checkers are in the home board."""
        if self.bar[player] > 0:
            return False
        if player == WHITE:
            # Home = indices 0–5.  No white checkers on 6–23.
            return all(self.points[i] <= 0 for i in range(6, 24))
        else:
            # Home = indices 18–23.  No black checkers on 0–17.
            return all(self.points[i] >= 0 for i in range(0, 18))

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

    # ── Mutation helpers (operate on copies) ─────────────────────────────

    def _add(self, player: int, idx: int, count: int = 1):
        """Place *count* of *player*'s checkers on index *idx*."""
        self.points[idx] += count if player == WHITE else -count

    def _remove(self, player: int, idx: int, count: int = 1):
        self._add(player, idx, -count)

    # ── Display ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"BoardState(turn={'W' if self.turn == WHITE else 'B'}, "
            f"bar={self.bar}, off={self.off})"
        )

    def show(self) -> str:
        """ASCII art board for debugging."""
        lines = []
        lines.append("+13-14-15-16-17-18--+---+--19-20-21-22-23-24--+")

        # Top half (points 13-24 going left to right)
        for row in range(6):
            parts_left = []
            for i in range(12, 18):   # points 13-18
                parts_left.append(self._cell(i, row))
            parts_right = []
            for i in range(18, 24):   # points 19-24
                parts_right.append(self._cell(i, row))
            bar_ch = self._bar_cell(row, top=True)
            lines.append(
                f"|{'  '.join(parts_left)}  |{bar_ch}|  {'  '.join(parts_right)}  |"
            )

        lines.append("|                   |   |                      |")

        # Bottom half (points 12-1 going left to right)
        for row in range(5, -1, -1):
            parts_left = []
            for i in range(11, 5, -1):   # points 12-7
                parts_left.append(self._cell(i, row))
            parts_right = []
            for i in range(5, -1, -1):   # points 6-1
                parts_right.append(self._cell(i, row))
            bar_ch = self._bar_cell(row, top=False)
            lines.append(
                f"|{'  '.join(parts_left)}  |{bar_ch}|  {'  '.join(parts_right)}  |"
            )

        lines.append("+12-11-10--9--8--7--+---+---6--5--4--3--2--1--+")

        lines.append(
            f"  White bar: {self.bar[WHITE]}   Black bar: {self.bar[BLACK]}   "
            f"White off: {self.off[WHITE]}   Black off: {self.off[BLACK]}   "
            f"Turn: {'WHITE' if self.turn == WHITE else 'BLACK'}"
        )
        return "\n".join(lines)

    def _cell(self, idx: int, row: int) -> str:
        v = self.points[idx]
        count = abs(v)
        if row < count:
            return "W" if v > 0 else "B"
        return "."

    def _bar_cell(self, row: int, top: bool) -> str:
        if top:
            return "B" if row < self.bar[BLACK] else " "
        else:
            return "W" if row < self.bar[WHITE] else " "


# ── Move generation ──────────────────────────────────────────────────────────

def _single_moves(state: BoardState, die: int) -> List[Move]:
    """All legal single-checker moves for *state.turn* using one die.

    Performance-critical: accesses state.points directly to avoid
    method-call overhead (checker_count was 30M calls in profiling).
    """
    player = state.turn
    pts = state.points
    moves: List[Move] = []

    # ── Must enter from bar first ────────────────────────────────────
    if state.bar[player] > 0:
        target = (24 - die) if player == WHITE else (die - 1)

        # Inline is_blocked: opponent has 2+ checkers?
        v = pts[target]
        opp_count = (-v) if player == WHITE else v
        if opp_count < 2:
            moves.append((BAR, target))
        return moves  # while on bar, no other moves allowed

    # ── Regular moves and bearing off ────────────────────────────────
    #
    # WHITE: checkers are positive, direction = -1
    # BLACK: checkers are negative, direction = +1
    is_white = (player == WHITE)
    direction = -1 if is_white else 1

    # Inline all_in_home check
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

    # Pre-compute farthest checker in home board (only if bearing off).
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
        # Inline has_checker: for WHITE, v > 0; for BLACK, v < 0
        if is_white:
            if v <= 0:
                continue
        else:
            if v >= 0:
                continue

        target = src + direction * die

        # Normal move (target on the board)
        if 0 <= target <= 23:
            # Inline is_blocked
            tv = pts[target]
            opp_count = (-tv) if is_white else tv
            if opp_count < 2:
                moves.append((src, target))
            continue

        # Bearing off attempt (target off the board)
        if not can_bear_off:
            continue

        # Distance from bearing off
        d = (src + 1) if is_white else (24 - src)
        if die == d:
            moves.append((src, OFF))
        elif die > d and src == farthest:
            moves.append((src, OFF))

    return moves


def _apply_move(state: BoardState, move: Move) -> BoardState:
    """Apply a single move, returning a NEW BoardState.  Handles hits.

    Performance-critical: inlines _add/_remove/checker_count to avoid
    method-call overhead (~2M calls during move generation).
    """
    # Inline copy: direct list slicing is faster than calling state.copy()
    pts = state.points[:]
    bar = state.bar[:]
    off = state.off[:]
    player = state.turn
    src, dst = move

    # sign: +1 for WHITE checkers, -1 for BLACK
    sign = 1 if player == WHITE else -1

    # Remove checker from source
    if src == BAR:
        bar[player] -= 1
    else:
        pts[src] -= sign

    # Place checker at destination
    if dst == OFF:
        off[player] += 1
    else:
        # Hit? Opponent has exactly 1 checker on dst
        tv = pts[dst]
        opp_count = (-tv) if player == WHITE else tv
        if opp_count == 1:
            pts[dst] += sign   # remove opponent's checker
            bar[1 - player] += 1
        pts[dst] += sign  # place our checker

    return BoardState(pts, bar, off, state.turn)


def _generate_plays(
    state: BoardState,          # read-only (copies made via _apply_move)
    remaining_dice: Tuple[int, ...], # read-only (new tuples created each level)
    current_play: Play,         # read-only (immutable tuple)
    results: dict,              # MUTATED: accumulates all terminal plays
    dice_used: int,             # read-only (immutable int, new value per level)
    max_used: List[int],        # MUTATED: tracks global max (num of) dice used
):
    """Recursively build all legal plays, collecting results.

    For each remaining_dice die value, we find every legal single-checker move
    with that die, apply it, and recurse with the remaining_dice dice.

    When no die can be used (either none left or all blocked), this is a
    terminal node and we record the resulting board state.

    Different orderings of moves can reach the same board position
    (e.g. move A then B vs B then A).  We deduplicate by keying results
    on the board state, so the caller only sees distinct outcomes.

    max_used tracks the highest dice_used across all terminal nodes.
    max_used is a 1-element list (rather than an int) so that recursive
    calls can mutate it.
    """
    found = False

    # Avoid exploring the same die value twice at this recursion level.
    # For non-doubles this is a no-op (two distinct values), but for
    # doubles [6,6,6,6] it prevents generating the same subtree 4 times.
    seen_this_level: set = set()

    for i, die in enumerate(remaining_dice):
        if die in seen_this_level:
            continue
        seen_this_level.add(die)

        # Remove this die from the remaining_dice tuple (order-preserving).
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
        # No legal move with any remaining_dice die -- this path is complete.
        # Record the resulting position (deduplicated by board state).
        if dice_used > max_used[0]:
            max_used[0] = dice_used
        key = state._key()
        if key not in results:
            results[key] = (current_play, state)


def get_legal_plays(
    state: BoardState, dice: Tuple[int, int]
) -> List[Tuple[Play, BoardState]]:
    """Return every legal play for *state.turn* given *dice*.

    Each entry is ``(play, resulting_state)`` where *play* is a tuple
    of ``(src, dst)`` single moves and ``resulting_state.turn`` is
    already switched to the opponent (the next player to act).
    Callers should NOT switch_turn again. Results are deduplicated by
    resulting board state.

    Rules enforced:
    * All four sub-moves for doubles.
    * Must use both dice if possible; if only one, must use the larger.
    * Checkers on the bar must enter before any other move.
    * Bearing off respects exact / over-bear rules.
    """
    # Doubles get 4 sub-moves; non-doubles get 2.
    d1, d2 = dice
    if d1 == d2:
        remaining_dice = (d1, d1, d1, d1)
    else:
        remaining_dice = (d1, d2)

    # results: maps board-state key -> (play, state), giving us
    #          automatic deduplication of plays that reach the same position.
    # max_used: tracks the most dice any complete play managed to use.
    results: dict = {}
    max_used: List[int] = [0]

    _generate_plays(state, remaining_dice, (), results, 0, max_used)

    # Rule: must use as many dice as possible.  Discard plays that used fewer.
    # When no dice can be used at all, most_dice_used == 0 and we return [].
    most_dice_used = max_used[0]
    if most_dice_used == 0:
        return []
    filtered = [
        (play, s)
        for play, s in results.values()
        if len(play) == most_dice_used
    ]

    # Rule: if non-doubles and only one die can be used, must use the larger.
    # (If using the smaller die would *also* allow using the larger, that
    # 2-die play already survived the max filter above, so this only
    # matters when exactly one die is playable.)
    if d1 != d2 and most_dice_used == 1:
        big = max(d1, d2)
        uses_big = [(p, s) for p, s in filtered if _play_uses_die(p, state, big)]
        if uses_big:
            filtered = uses_big

    # Switch turn on each result so .turn is the opponent (next to act).
    return [(p, switch_turn(s)) for p, s in filtered]


def get_legal_plays_encoded(state: BoardState, dice, encoder=None):
    """Combined move generation + perspective encoding. Mirrors
    `bg_fast.get_legal_plays_encoded` so callers use the same
    signature regardless of which engine is active.

    Returns `(features, next_states)`:
      - features: (N, num_features) float32, perspective-encoded
      - next_states: list of N BoardState objects (already switched)
    """
    # Lazy import to avoid a circular dep with encoding.py.
    from encoding import get_encoder
    if encoder is None:
        encoder = get_encoder("perspective196")
    plays = get_legal_plays(state, dice)
    if not plays:
        return np.empty((0, encoder.num_features), dtype=np.float32), []
    next_states = [s for _, s in plays]
    features = np.stack([encoder.encode(s) for s in next_states])
    return features, next_states


def _play_uses_die(play: Play, original: BoardState, die: int) -> bool:
    """Check whether a single-move play corresponds to using *die*."""
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
        # Could be exact or over-bear — compute distance
        if player == WHITE:
            d = src + 1
        else:
            d = 24 - src
        return die >= d  # exact or over-bear with this die
    else:
        return dst == src + direction * die


# ── Convenience helpers ──────────────────────────────────────────────────────

def switch_turn(state: BoardState) -> BoardState:
    """Return a copy with the turn flipped."""
    s = state.copy()
    s.turn = 1 - s.turn
    return s


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
    """Standard backgammon opening: each player rolls one die. The
    higher die wins and uses both dice as the first move; doublets are
    rerolled (you cannot start a game with a doublet).

    Args:
        rng: an object with a `randint(a, b)` method (a `random.Random`
            instance, or the `random` module itself). Defaults to the
            global `random` module.

    Returns:
        (state, dice) where state is the standard initial position with
        `state.turn` set to the winner of the opening roll, and dice is
        `(higher_die, lower_die)`.
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

