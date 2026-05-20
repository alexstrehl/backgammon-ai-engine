"""Tests for backgammon_engine.py — run with: pytest tests/test_backgammon_engine.py -v"""

import random
from backgammon_engine import (
    BoardState, WHITE, BLACK, BAR, OFF, NUM_CHECKERS,
    get_legal_plays, opening_roll, switch_turn,
)


def test_starting_position_checker_count():
    b = BoardState.initial()
    assert sum(max(v, 0) for v in b.points) == NUM_CHECKERS
    assert sum(max(-v, 0) for v in b.points) == NUM_CHECKERS


def test_opening_3_1():
    b = BoardState.initial()
    plays = get_legal_plays(b, (3, 1))
    assert len(plays) > 0
    assert all(len(p) == 2 for p, _ in plays)
    assert len(plays) == 16


def test_opening_6_6():
    b = BoardState.initial()
    plays = get_legal_plays(b, (6, 6))
    assert len(plays) > 0


def test_bar_entry_enforced():
    b = BoardState.initial()
    b.bar[WHITE] = 1
    b.points[5] = 4  # was 5, now 4
    plays = get_legal_plays(b, (3, 1))
    for play, _ in plays:
        assert play[0][0] == BAR, f"First move should be bar entry, got {play[0]}"


def test_bearing_off():
    b = BoardState(
        points=[0] * 24,
        bar=[0, 0],
        off=[10, 10],
        turn=WHITE,
    )
    for i in range(5):
        b.points[i] = 1
    for i in range(19, 24):
        b.points[i] = -1
    plays = get_legal_plays(b, (6, 5))
    assert len(plays) > 0
    for play, st in plays:
        for m in play:
            assert m[1] == OFF or (0 <= m[1] <= 5), f"Bad bearing off move {m}"


def test_over_bear_farthest_checker():
    b = BoardState(
        points=[0] * 24,
        bar=[0, 0],
        off=[13, 15],
        turn=WHITE,
    )
    b.points[0] = 1  # point 1 (distance 1)
    b.points[1] = 1  # point 2 (distance 2)
    plays = get_legal_plays(b, (5, 6))
    for play, st in plays:
        assert st.off[WHITE] == 15, "Both should be borne off"


def test_fully_blocked_bar():
    b = BoardState(
        points=[0] * 24,
        bar=[1, 0],
        off=[0, 0],
        turn=WHITE,
    )
    for i in range(18, 24):
        b.points[i] = -2
    b.points[5] = 14
    plays = get_legal_plays(b, (3, 1))
    assert plays == []


def test_random_rollout():
    """200-step random game -- just verify no crashes."""
    random.seed(42)
    state = BoardState.initial()
    for _ in range(200):
        if state.is_game_over():
            break
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        plays = get_legal_plays(state, (d1, d2))
        if plays:
            _, state = random.choice(plays)
        state = switch_turn(state)


# ── opening_roll ─────────────────────────────────────────────────────


def test_opening_roll_no_doublets():
    """opening_roll never returns doublets — they should be rerolled."""
    rng = random.Random(0)
    for _ in range(500):
        _, (d1, d2) = opening_roll(rng)
        assert d1 != d2, f"opening_roll returned doublet ({d1},{d2})"
        assert 1 <= d1 <= 6 and 1 <= d2 <= 6
        assert d1 >= d2, "opening_roll should return (high, low)"


def test_opening_roll_starts_at_initial_position():
    """The state returned should be the standard starting position with
    only the turn possibly differing from BoardState.initial()."""
    rng = random.Random(7)
    state, _ = opening_roll(rng)
    canonical = BoardState.initial()
    assert state.points == canonical.points
    assert state.bar == canonical.bar
    assert state.off == canonical.off
    assert state.turn in (WHITE, BLACK)


def test_opening_roll_picks_both_players_over_time():
    """Over many calls each player should sometimes go first."""
    rng = random.Random(123)
    counts = {WHITE: 0, BLACK: 0}
    for _ in range(500):
        state, _ = opening_roll(rng)
        counts[state.turn] += 1
    assert counts[WHITE] > 100 and counts[BLACK] > 100, (
        f"unbalanced opening rolls: {counts}"
    )


# ── Bear-off regression tests ────────────────────────────────────────────────
# These test a bug where _play_uses_die incorrectly claimed an exact bear-off
# could "use" a larger die (e.g. bearing off from point 4 with die=4 was
# reported as also using die=5), causing the "must use larger die" filter
# to discard valid moves.

def _bearoff_position():
    """X (WHITE) bearing off, O (BLACK) blocks point 1."""
    board = [0] * 24
    board[1] = 3    # point 2: 3 X
    board[2] = 3    # point 3: 3 X
    board[3] = 2    # point 4: 2 X
    board[5] = 2    # point 6: 2 X
    board[0] = -4   # point 1: 4 O (blocks)
    board[11] = -2  # point 12: 2 O
    board[17] = -2  # point 18: 2 O
    board[18] = -2  # point 19: 2 O
    board[19] = -2  # point 20: 2 O
    board[22] = -2  # point 23: 2 O
    return BoardState(points=board, turn=WHITE, bar=[0, 1], off=[5, 0])


def _midgame_position():
    """X (WHITE) with checkers spread across the board."""
    board = [0] * 24
    board[2] = 2; board[3] = 2; board[4] = 2; board[5] = 2
    board[11] = 1; board[13] = 2; board[16] = 1; board[18] = 1; board[19] = 2
    board[23] = -3
    return BoardState(points=board, turn=WHITE, bar=[0, 0], off=[0, 12])


def test_bearoff_45_python():
    """Bear-off with 4-5: die 5 is blocked (6->1 occupied by 4 opp),
    so only die 4 is playable, giving 2 moves: 4->off and 6->2."""
    state = _bearoff_position()
    plays = get_legal_plays(state, (4, 5))
    assert len(plays) == 2, f"Expected 2, got {len(plays)}"


def test_bearoff_45_c_engine():
    """Same bear-off test using the C engine."""
    import sys
    sys.path.insert(0, "c_engine")
    try:
        import bg_fast
        import importlib
        importlib.reload(bg_fast)
    except (ImportError, OSError):
        import pytest
        pytest.skip("C engine not built")

    state = _bearoff_position()
    feats, states = bg_fast.get_legal_plays_encoded(state, (4, 5))
    assert len(states) == 2, f"Expected 2, got {len(states)}"


def test_midgame_double1s_python():
    """Mid-game double 1s: 536 distinct legal plays (Python engine)."""
    state = _midgame_position()
    plays = get_legal_plays(state, (1, 1))
    assert len(plays) == 536, f"Expected 536, got {len(plays)}"


def test_midgame_double1s_c_engine():
    """Mid-game double 1s: 536 distinct legal plays (C engine)."""
    import sys
    sys.path.insert(0, "c_engine")
    try:
        import bg_fast
        import importlib
        importlib.reload(bg_fast)
    except (ImportError, OSError):
        import pytest
        pytest.skip("C engine not built")

    state = _midgame_position()
    feats, states = bg_fast.get_legal_plays_encoded(state, (1, 1))
    assert len(states) == 536, f"Expected 536, got {len(states)}"


def _bearoff_position_2():
    """X (WHITE) bearing off, O (BLACK) has 2 on bar and 3 at point 24."""
    board = [0] * 24
    board[0] = 2    # point 1: 2 X
    board[1] = 2    # point 2: 2 X
    board[2] = 2    # point 3: 2 X
    board[5] = 1    # point 6: 1 X
    board[23] = -3  # point 24: 3 O
    return BoardState(points=board, turn=WHITE, bar=[0, 2], off=[8, 10])


def test_bearoff_52_python():
    """Bear-off with 5-2: 3 legal moves (verified vs gnubg)."""
    state = _bearoff_position_2()
    plays = get_legal_plays(state, (5, 2))
    assert len(plays) == 3, f"Expected 3, got {len(plays)}"


def test_bearoff_52_c_engine():
    """Same bear-off 5-2 test using the C engine."""
    import sys
    sys.path.insert(0, "c_engine")
    try:
        import bg_fast
        import importlib
        importlib.reload(bg_fast)
    except (ImportError, OSError):
        import pytest
        pytest.skip("C engine not built")
    state = _bearoff_position_2()
    feats, states = bg_fast.get_legal_plays_encoded(state, (5, 2))
    assert len(states) == 3, f"Expected 3, got {len(states)}"


def test_bearoff_and_midgame_engines_agree():
    """Python and C engines produce the same move count for all test positions."""
    import sys
    sys.path.insert(0, "c_engine")
    try:
        import bg_fast
        import importlib
        importlib.reload(bg_fast)
    except (ImportError, OSError):
        import pytest
        pytest.skip("C engine not built")

    for state, dice, label in [
        (_bearoff_position(), (4, 5), "bearoff 4-5"),
        (_midgame_position(), (1, 1), "midgame 1-1"),
        (_bearoff_position_2(), (5, 2), "bearoff 5-2"),
    ]:
        py_plays = get_legal_plays(state, dice)
        c_feats, c_states = bg_fast.get_legal_plays_encoded(state, dice)
        assert len(py_plays) == len(c_states), (
            f"{label}: Python={len(py_plays)} vs C={len(c_states)}"
        )


def _bg_fast_or_skip():
    import sys
    sys.path.insert(0, "c_engine")
    try:
        import bg_fast
        import importlib
        importlib.reload(bg_fast)
        return bg_fast
    except (ImportError, OSError):
        import pytest
        pytest.skip("C engine not built")


def test_initial_doubles_c_engine_nonzero():
    """Regression: every doubles roll from the initial position has
    legal 4-move plays. A previous bounds-guard bug in the C engine's
    generate_plays recursion made the leaf at depth-4 return early,
    silently dropping every full-doubles play and returning 0."""
    bg_fast = _bg_fast_or_skip()
    b = BoardState.initial()
    for d in range(1, 7):
        py_plays = get_legal_plays(b, (d, d))
        c_feats, c_states = bg_fast.get_legal_plays_encoded(b, (d, d))
        assert len(py_plays) > 0, f"Python returned 0 plays for ({d},{d}) — fixture bug"
        assert len(c_states) == len(py_plays), (
            f"({d},{d}) from initial position: "
            f"Python={len(py_plays)} vs C={len(c_states)}"
        )


def test_full_4move_doubles_c_engine_parity():
    """Targeted parity check on positions where the legal plays use
    all 4 dice. This is the exact regime the depth-guard bug broke
    (recursion never reaches the leaf-recording branch when num_moves
    hits MAX_MOVES_PER_PLAY=4)."""
    bg_fast = _bg_fast_or_skip()
    cases = [
        (BoardState.initial(), (1, 1)),
        (BoardState.initial(), (2, 2)),
        (BoardState.initial(), (6, 6)),
        (_midgame_position(), (1, 1)),  # 536 plays, all 4 moves
    ]
    for state, dice in cases:
        py_plays = get_legal_plays(state, dice)
        max_py = max((len(p) for p, _ in py_plays), default=0)
        assert max_py == 4, (
            f"fixture {dice}: expected a 4-move play to exist, "
            f"got max moves = {max_py}"
        )
        c_feats, c_states = bg_fast.get_legal_plays_encoded(state, dice)
        assert len(c_states) == len(py_plays), (
            f"{dice}: Python={len(py_plays)} vs C={len(c_states)}"
        )
