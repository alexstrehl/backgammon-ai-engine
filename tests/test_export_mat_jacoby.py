"""The .mat Jacoby header must match the rule the game was played under."""
from third_party_opponents.gnubg import gnubg_eval as ge


def _header(jacoby):
    rec = ge.GameRecord(jacoby=jacoby, winner=ge.WHITE, result=1)
    return ge.export_mat(rec, money_game=True)


def test_jacoby_on():
    assert '; [Jacoby "On"]' in _header(True)


def test_jacoby_off():
    txt = _header(False)
    assert '; [Jacoby "Off"]' in txt
    assert '"On"]' not in txt.split("[Beaver")[0]


def test_cubeless_has_no_rules_block():
    assert "Jacoby" not in ge.export_mat(ge.GameRecord(winner=ge.WHITE, result=1))
