"""MET parsing and lookup tests.

The load-bearing test here is `test_mwc_after_matches_matchstate`: the
MET is only useful as ground truth if its notion of a score transition
is identical to the engine's. Everything else is table sanity.
"""

import subprocess

import numpy as np
import pytest

import met as met_mod
from met import MatchEquityTable, DEFAULT_MET
from modes import MatchState


@pytest.fixture(scope="module")
def table() -> MatchEquityTable:
    return MatchEquityTable.from_xml(DEFAULT_MET)


# ── table values ────────────────────────────────────────────────────


def test_name_and_length(table):
    assert "Kazaross" in table.name
    assert table.length == 25


@pytest.mark.parametrize(
    "my_away,opp_away,expected",
    [
        (1, 1, 0.50000),   # DMP
        (1, 2, 0.67736),   # the Crawford value MATCHPLAY.md converged to
        (2, 1, 0.32264),
        (2, 2, 0.50000),
        (3, 3, 0.50000),
        (1, 3, 0.75076),
        (2, 3, 0.59947),
        (3, 2, 0.40053),
    ],
)
def test_pre_crawford_values(table, my_away, opp_away, expected):
    assert table.mwc(my_away, opp_away) == pytest.approx(expected, abs=1e-5)


def test_crawford_game_uses_pre_table(table):
    # The Crawford game is scored from the pre-Crawford table; the cube
    # being off is MatchState.can_offer's job, not the table's.
    assert table.mwc(1, 2, crawford_done=True, is_crawford=True) == pytest.approx(
        table.mwc(1, 2), abs=1e-12
    )


@pytest.mark.parametrize(
    "my_away,expected",
    [(1, 0.50000), (2, 0.48803), (3, 0.32264), (4, 0.31002)],
)
def test_post_crawford_values(table, my_away, expected):
    got = table.mwc(my_away, 1, crawford_done=True, is_crawford=False)
    assert got == pytest.approx(expected, abs=1e-5)


def test_post_crawford_is_perspective_symmetric(table):
    for away in range(1, 10):
        a = table.mwc(away, 1, crawford_done=True)
        b = table.mwc(1, away, crawford_done=True)
        assert a + b == pytest.approx(1.0, abs=1e-12)


def test_pre_crawford_is_complementary(table):
    for i in range(1, 16):
        for j in range(1, 16):
            assert table.mwc(i, j) + table.mwc(j, i) == pytest.approx(
                1.0, abs=1e-4
            )


def test_match_over_short_circuits(table):
    assert table.mwc(0, 3) == 1.0
    assert table.mwc(-2, 3) == 1.0
    assert table.mwc(3, 0) == 0.0


def test_leading_is_better(table):
    # More points to go is never better, holding the opponent fixed.
    for opp in range(1, 12):
        vals = [table.mwc(mine, opp) for mine in range(1, 12)]
        assert vals == sorted(vals, reverse=True)


def test_out_of_range_raises(table):
    with pytest.raises(ValueError, match="exceeds"):
        table.mwc(26, 3)


# ── transition consistency (the important one) ──────────────────────


def test_mwc_after_matches_matchstate(table):
    """mwc_after must agree with MatchState.after_game + start_game on
    every reachable 3..7 point transition, including Crawford latching.
    """
    checked = 0
    for match_len in (3, 5, 7):
        for my_away in range(1, match_len + 1):
            for opp_away in range(1, match_len + 1):
                for crawford_done in (False, True):
                    for is_crawford in (False, True):
                        if is_crawford and not (my_away == 1 or opp_away == 1):
                            continue
                        for cube_value in (1, 2, 4):
                            for result in (1, 2, 3):
                                for winner_is_me in (True, False):
                                    ms = MatchState(
                                        match_length=match_len,
                                        my_away=my_away,
                                        opp_away=opp_away,
                                        cube_value=cube_value,
                                        is_crawford=is_crawford,
                                        crawford_done=crawford_done or is_crawford,
                                    )
                                    nxt = ms.after_game(result, winner_is_me)
                                    got = table.mwc_after(
                                        my_away, opp_away,
                                        crawford_done, is_crawford,
                                        cube_value * result, winner_is_me,
                                    )
                                    if nxt.match_over():
                                        want = 1.0 if nxt.i_won_match() else 0.0
                                    else:
                                        started = nxt.start_game()
                                        want = table.mwc(
                                            started.my_away, started.opp_away,
                                            started.crawford_done,
                                            started.is_crawford,
                                        )
                                    assert got == pytest.approx(want, abs=1e-12), (
                                        f"{my_away}a{opp_away} cube={cube_value} "
                                        f"result={result} me={winner_is_me} "
                                        f"cd={crawford_done} cf={is_crawford}"
                                    )
                                    checked += 1
    assert checked > 2000


def test_crawford_fires_once(table):
    # 3-away/3-away, I win 2 -> 1-away/3-away schedules Crawford, so the
    # next game is scored from the pre-Crawford table.
    got = table.mwc_after(3, 3, False, False, points=2, winner_is_me=True)
    assert got == pytest.approx(table.mwc(1, 3), abs=1e-12)
    # Same transition once Crawford is already spent -> post-Crawford.
    got = table.mwc_after(3, 3, True, False, points=2, winner_is_me=True)
    assert got == pytest.approx(
        table.mwc(1, 3, crawford_done=True), abs=1e-12
    )
    assert table.mwc(1, 3) != pytest.approx(table.mwc(1, 3, crawford_done=True))


# ── the 13-feature block ────────────────────────────────────────────


def test_block_shape_and_immutability(table):
    b = table.outcome_block(3, 3, False, False, 1)
    assert b.shape == (13,)
    assert b.dtype == np.float32
    with pytest.raises(ValueError):
        b[0] = 0.5


def test_block_entries_match_direct_lookups(table):
    my_away, opp_away, cube = 3, 3, 1
    b = table.outcome_block(my_away, opp_away, False, False, cube)
    for slot, cv in ((0, cube), (6, 2 * cube)):
        for i, mult in enumerate((1, 2, 3)):
            assert b[slot + i] == pytest.approx(
                table.mwc_after(my_away, opp_away, False, False, cv * mult, True),
                abs=1e-6,
            )
            assert b[slot + 3 + i] == pytest.approx(
                table.mwc_after(my_away, opp_away, False, False, cv * mult, False),
                abs=1e-6,
            )
    assert b[12] == pytest.approx(
        table.mwc_after(my_away, opp_away, False, False, cube, True), abs=1e-6
    )


def test_block_monotonic_in_outcome_size(table):
    """Winning more points is never worse; losing more is never better."""
    for my_away in range(1, 6):
        for opp_away in range(1, 6):
            for cube in (1, 2, 4):
                b = table.outcome_block(my_away, opp_away, False, False, cube)
                for slot in (0, 6):
                    wins = b[slot:slot + 3]
                    loses = b[slot + 3:slot + 6]
                    assert np.all(np.diff(wins) >= -1e-6), (my_away, opp_away, cube)
                    assert np.all(np.diff(loses) <= 1e-6), (my_away, opp_away, cube)


def test_drop_equals_winning_cube_value(table):
    """The drop feature is exactly 'doubler collects the pre-double cube'
    — this is the value that drifted 0.680 -> 0.639 under pure TD(0).
    """
    # 2-away/2-away, centred cube: doubling and being dropped wins 1
    # point -> 1-away/2-away, which is the 0.6774 canary.
    b = table.outcome_block(2, 2, False, False, 1)
    assert b[12] == pytest.approx(table.mwc(1, 2), abs=1e-6)
    assert b[12] == pytest.approx(0.67736, abs=1e-5)


def test_dead_cube_outcomes_saturate(table):
    # 2-away, cube already 2: any win at the take level ends the match.
    b = table.outcome_block(2, 3, False, False, 2)
    assert np.all(b[6:9] == pytest.approx(1.0))


def test_block_is_cached(table):
    a = table.outcome_block(3, 3, False, False, 1)
    b = table.outcome_block(3, 3, False, False, 1)
    assert a is b


# ── cross-check against gnubg itself ────────────────────────────────


def _gnubg_met_dump() -> str:
    cmd = (
        f"set matchequity {DEFAULT_MET}\n"
        "show matchequitytable\nquit\n"
    )
    return subprocess.run(
        ["gnubg", "-t", "-q"], input=cmd, capture_output=True,
        text=True, timeout=120,
    ).stdout


@pytest.mark.slow
def test_matches_gnubg_own_table():
    """Parse gnubg's printed MET and compare cell-for-cell. If this
    fails, we are training against a different table than we are
    scored against.
    """
    out = _gnubg_met_dump()
    assert "Kazaross" in out, out[:400]
    table = MatchEquityTable.from_xml(DEFAULT_MET)

    lines = out.splitlines()
    pre_start = next(i for i, l in enumerate(lines) if "Pre-Crawford table" in l)
    post_start = next(i for i, l in enumerate(lines) if "Post-Crawford table" in l)

    def parse_rows(chunk):
        rows = {}
        for line in chunk:
            parts = line.split()
            # Skip the column header, whose cells are "N-away" labels.
            if len(parts) < 2 or not parts[0].endswith("-away"):
                continue
            try:
                cells = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            rows[int(parts[0].split("-")[0])] = cells
        return rows

    pre = parse_rows(lines[pre_start:post_start])
    assert len(pre) >= 25
    for my_away, cells in pre.items():
        if my_away > 25:
            continue
        for j, cell in enumerate(cells[:25]):
            assert table.mwc(my_away, j + 1) * 100 == pytest.approx(
                cell, abs=1e-3
            ), f"pre {my_away}a vs {j + 1}a"

    post = parse_rows(lines[post_start:])
    assert post, "no post-Crawford rows parsed"
    cells = next(iter(post.values()))
    # gnubg extends its internal tables to 64-away and in doing so
    # recomputes the final cell of the 25-entry post-Crawford row. Every
    # other entry is bit-identical to the XML; that one differs by 5e-5
    # MWC. Assert the agreement AND the known exception, so a future
    # divergence anywhere else still fails loudly.
    for j, cell in enumerate(cells[:24]):
        got = table.mwc(j + 1, 1, crawford_done=True) * 100
        assert got == pytest.approx(cell, abs=1e-3), f"post {j + 1}a vs 1a"
    boundary = table.mwc(25, 1, crawford_done=True) * 100
    assert boundary == pytest.approx(cells[24], abs=1e-2)
    assert boundary != pytest.approx(cells[24], abs=1e-3)


def test_default_met_is_gnubgs_default(table):
    assert met_mod.default_met().name == table.name


# ── the referee mismatch ────────────────────────────────────────────


@pytest.mark.slow
def test_gnubg_nn_referee_uses_snowie_not_kazaross():
    """The two instruments in MATCHPLAY.md score against DIFFERENT METs.

    `eval_matchplay.py`'s `gnubg:` side goes through gnubg-nn,
    whose MET is Snowie (verified exact below). `gnubg_eval_matchplay.py`
    drives the gnubg 1.08 CLI, whose default is Kazaross-XG2. The two
    disagree by up to 7.6 mMWC at 3-point scores.

    Consequence for the cube-error gate: any mining threshold below
    ~8 mMWC will surface pure MET artifacts as "errors". Keep it above.
    """
    gnubg_nn = pytest.importorskip("gnubg_nn")
    from backgammon_engine import BoardState
    from agents import GnubgNNAgent

    snowie = MatchEquityTable.from_xml(f"{met_mod.MET_DIR}/snowie.xml")
    board = GnubgNNAgent._board_to_gnubg(BoardState.initial())
    pid = gnubg_nn.position_id(board)

    # mwcDoublePass at cube 1 is the doubler's MWC after winning 1
    # point, i.e. mwc(a-1, o). Opponent-1-away cells are excluded:
    # gnubg-nn has no Crawford flag and always applies its own
    # post-Crawford logic there (MatchState.can_offer keeps the agent
    # out of Crawford games, so this never bites in practice).
    for a in range(2, 10):
        for o in range(2, 10):
            gnubg_nn.set.score(a, o)
            gnubg_nn.set.cube(1, b"X")
            implied = gnubg_nn.evaluate_cube_decision(pid, n=1, i=1)[5]
            assert implied == pytest.approx(
                snowie.mwc(a - 1, o), abs=1e-5
            ), f"gnubg-nn is no longer Snowie at {a - 1}a/{o}a"


def test_kazaross_snowie_gap_bounds_the_mining_threshold(table):
    snowie = MatchEquityTable.from_xml(f"{met_mod.MET_DIR}/snowie.xml")
    gaps = [
        abs(table.mwc(m, o) - snowie.mwc(m, o))
        for m in range(1, 4) for o in range(1, 4)
    ]
    assert max(gaps) == pytest.approx(0.00764, abs=1e-4)
