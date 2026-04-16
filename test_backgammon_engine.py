"""Tests for backgammon_engine.py — run with: pytest test_backgammon_engine.py -v"""

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
