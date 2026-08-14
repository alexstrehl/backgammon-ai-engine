"""A dropped double must credit the DOUBLER, not the dropper.

XG (and gnubg) read the winner off the column the "Wins N points" token sits
in. export_mat used to append that token to whichever column held " Drops",
which credited the game to the player who dropped. That inverted every
white-doubles/black-drops game -- ~25% of a cubeful set -- in every .mat
export, while leaving the per-decision PR columns untouched, so it showed up
as a losing win-rate next to a winning PR.
"""
import gnubg_eval as ge

COL2 = 20  # offset at/after which text is in BLACK's column


def _mat(doubler, dropper, cube_before=1):
    rec = ge.GameRecord()
    rec.moves = [
        ge.MoveRecord(player=ge.WHITE, dice=(3, 1), play=((7, 4), (5, 4))),
        ge.MoveRecord(player=ge.BLACK, dice=(6, 2), play=((12, 6), (6, 4))),
        ge.CubeRecord(player=doubler, action="double",
                      new_cube_value=cube_before * 2),
        ge.CubeRecord(player=dropper, action="drop"),
    ]
    rec.winner = doubler
    rec.result = 1
    rec.cube_value = cube_before
    rec.ended_by_drop = True
    return ge.export_mat(rec, money_game=True)


def _wins_col(text):
    for line in text.splitlines():
        if "Wins" in line:
            return 0 if line.index("Wins") < COL2 else 1
    raise AssertionError("no 'Wins' line in:\n" + text)


def test_white_doubles_black_drops_credits_white():
    # the regressing case: both columns of the last row are occupied
    assert _wins_col(_mat(ge.WHITE, ge.BLACK)) == 0


def test_black_doubles_white_drops_credits_black():
    assert _wins_col(_mat(ge.BLACK, ge.WHITE)) == 1


def test_redouble_drop_also_credits_the_doubler():
    assert _wins_col(_mat(ge.WHITE, ge.BLACK, cube_before=2)) == 0
    assert _wins_col(_mat(ge.BLACK, ge.WHITE, cube_before=2)) == 1


def test_drop_points_are_the_pre_double_cube():
    txt = _mat(ge.WHITE, ge.BLACK, cube_before=2)
    assert "Wins 2 points" in txt, txt
