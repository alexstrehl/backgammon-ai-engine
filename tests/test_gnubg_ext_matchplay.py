"""N-point match play through the gnubg external interface.

The dangerous field is the score pair: gnubg's two score slots are
REVERSED relative to their FIBS names (slot 5 carries the ON-ROLL
player's points), and getting it backwards produces plausible-but-wrong
play with no error. The string-level tests here pin it for both movers
and both line shapes; the live tests then check the result against
backgammon doctrine.
"""

import os
import shutil

import pytest

from backgammon_engine import BoardState, WHITE, BLACK
from modes import CubeOwner, MatchState

gnubg_external = pytest.importorskip("gnubg_external")
from gnubg_external import GnubgExternalAgent, _ExtServer  # noqa: E402

needs_gnubg = pytest.mark.skipif(
    not os.path.exists(gnubg_external.GNUBG_CMD),
    reason="gnubg binary not installed",
)

MET_DIR = "/usr/share/gnubg/met"
KAZAROSS = os.path.join(MET_DIR, "Kazaross-XG2.xml")
SNOWIE = os.path.join(MET_DIR, "snowie.xml")


def ms(match_length=3, my_away=3, opp_away=3, cube_value=1,
       cube_owner=CubeOwner.CENTERED, is_crawford=False, crawford_done=False):
    return MatchState(
        jacoby=False, match_length=match_length, my_away=my_away,
        opp_away=opp_away, cube_value=cube_value, cube_owner=cube_owner,
        is_crawford=is_crawford, crawford_done=crawford_done,
    )


def money():
    return MatchState(jacoby=True)


def line(agent, state, mover, match_state, dice=None, doubled=False):
    return agent._board_line(state, mover, match_state, dice, doubled)


@pytest.fixture(scope="module")
def agent():
    return GnubgExternalAgent(plies=0)


# ── string-level: no gnubg needed ───────────────────────────────────


def test_money_line_is_unchanged(agent):
    """Money rows must stay byte-identical: 9999, 0:0, crawfordrule off."""
    f = line(agent, BoardState.initial(), WHITE, money()).split(" ")[0].split(":")
    assert f[3] == "9999"
    assert (f[4], f[5]) == ("0", "0")
    assert (f[51], f[52]) == ("0", "0")
    assert "crawfordrule off" in line(agent, BoardState.initial(), WHITE, money())


def test_match_line_uses_points_scored_on_roll_last(agent):
    """3pt, WHITE 3-away (0 scored), BLACK 1-away (2 scored).
    Slot 5 = on-roll player's points; slot 4 = the other player's."""
    s, m = BoardState.initial(), ms(3, my_away=3, opp_away=1)
    f = line(agent, s, WHITE, m).split(" ")[0].split(":")
    assert f[3] == "3"
    assert (f[4], f[5]) == ("2", "0"), "WHITE on roll: own points last"
    f = line(agent, s, BLACK, m).split(" ")[0].split(":")
    assert (f[4], f[5]) == ("0", "2"), "BLACK on roll: own points last"


def test_doubled_line_scores_are_from_the_doubler(agent):
    """On a doubled line the board is encoded from the RESPONDER, so the
    on-roll player (slot 5) is the DOUBLER."""
    s, m = BoardState.initial(), ms(3, my_away=3, opp_away=1)
    # respond_to_double encodes with mover = responder = BLACK, so the
    # doubler is WHITE and slot 5 must carry WHITE's 0 points.
    f = line(agent, s, BLACK, m, doubled=True).split(" ")[0].split(":")
    assert (f[4], f[5]) == ("2", "0")


def test_scores_never_reach_match_length(agent):
    """score >= match length is 'badly formed board' in gnubg."""
    for L in (1, 3, 5, 7):
        for my in range(1, L + 1):
            for opp in range(1, L + 1):
                for mover in (WHITE, BLACK):
                    f = line(agent, BoardState.initial(), mover,
                             ms(L, my, opp)).split(" ")[0].split(":")
                    assert int(f[4]) < L and int(f[5]) < L


def test_post_crawford_flag(agent):
    s = BoardState.initial()
    def slot51(m):
        return line(agent, s, WHITE, m).split(" ")[0].split(":")[51]
    # pre-Crawford, and the Crawford game itself, are both 0 -- gnubg
    # infers Crawford from "1-away and slot 51 clear".
    assert slot51(ms(3, 3, 3)) == "0"
    assert slot51(ms(3, 3, 1, is_crawford=True, crawford_done=True)) == "0"
    assert slot51(ms(3, 3, 1, is_crawford=False, crawford_done=True)) == "1"


def test_crawfordrule_on_for_match_play(agent):
    assert "crawfordrule on" in line(agent, BoardState.initial(), WHITE, ms())


def test_may_double_never_both_zero(agent):
    """Both flags clear makes gnubg reinterpret an undoubled line as a
    TAKE decision -- a move request comes back 'take'."""
    s = BoardState.initial()
    for owner in (CubeOwner.CENTERED, CubeOwner.WHITE, CubeOwner.BLACK):
        for mover in (WHITE, BLACK):
            f = line(agent, s, mover,
                     ms(cube_owner=owner, cube_value=2 if owner else 1)
                     ).split(" ")[0].split(":")
            assert (f[38], f[39]) != ("0", "0")


def test_away_out_of_range_asserts(agent):
    with pytest.raises(AssertionError, match="out of range"):
        line(agent, BoardState.initial(), WHITE, ms(3, my_away=4, opp_away=1))


# ── live gnubg ──────────────────────────────────────────────────────


@needs_gnubg
def test_no_error_replies_across_scores(agent):
    """A malformed board now raises, so this sweep is self-checking."""
    s = BoardState.initial()
    for L in (3, 5):
        for my in range(1, L + 1):
            for opp in range(1, L + 1):
                for mover in (WHITE, BLACK):
                    agent._srv().request(line(agent, s, mover, ms(L, my, opp)))


@needs_gnubg
def test_two_away_two_away_doubles_where_money_rolls(agent):
    """At 2-away/2-away the cube is nearly worthless to the taker, so
    doctrine is to double aggressively. Checked from both sides, which
    is what catches a score-flip bug."""
    s = BoardState.initial()
    assert agent._srv().request(line(agent, s, WHITE, money())).lower() != "double"
    for mover in (WHITE, BLACK):
        reply = agent._srv().request(line(agent, s, mover, ms(3, 2, 2)))
        assert reply.lower() == "double", f"mover={mover} got {reply!r}"


@needs_gnubg
def test_one_away_never_doubles(agent):
    """At 1-away the cube is dead -- winning 1 point already wins."""
    s = BoardState.initial()
    for mover, m in ((WHITE, ms(3, 1, 3, crawford_done=True)),
                     (BLACK, ms(3, 3, 1, crawford_done=True))):
        reply = agent._srv().request(line(agent, s, mover, m))
        assert reply.lower() != "double", f"mover={mover} got {reply!r}"


@needs_gnubg
def test_crawford_kills_the_cube_post_crawford_frees_it(agent):
    """Same board and score; only the Crawford flag differs."""
    s = BoardState.initial()
    crawford = ms(3, my_away=3, opp_away=1, is_crawford=True, crawford_done=True)
    post = ms(3, my_away=3, opp_away=1, is_crawford=False, crawford_done=True)
    assert agent._srv().request(line(agent, s, WHITE, crawford)).lower() != "double"
    assert agent._srv().request(line(agent, s, WHITE, post)).lower() == "double"


@needs_gnubg
def test_checker_play_is_score_aware(agent):
    """The whole point of preferring gnubg-ext to gnubg-nn, whose own
    docstring concedes it has no score-aware chequer evaluation."""
    s, dice = BoardState.initial(), (6, 2)
    money_move = agent._srv().request(line(agent, s, WHITE, money(), dice))
    dmp_move = agent._srv().request(
        line(agent, s, WHITE, ms(3, 1, 1, crawford_done=True), dice))
    assert money_move != dmp_move, (money_move, dmp_move)


@needs_gnubg
def test_agent_api_round_trip_at_a_match_score(agent):
    """offer_double / respond_to_double / choose_checker_action_cubeful
    all drive the match line without raising."""
    s, m = BoardState.initial(), ms(3, 2, 2)
    assert agent.offer_double(s, m).should_double is True
    assert isinstance(agent.respond_to_double(s, m), bool)
    got = agent.choose_checker_action_cubeful(s, (3, 1), m)
    assert got is not None and got[0] is not None


# ── MET pinning ─────────────────────────────────────────────────────


JAC100 = os.path.join(MET_DIR, "jac100.xml")


@needs_gnubg
@pytest.mark.skipif(not os.path.exists(JAC100), reason="jac100 MET missing")
def test_met_pin_changes_the_referee():
    """Guards against the silent fallback: a MET that fails to load
    leaves gnubg on its compiled-in Zadeh table with no visible error,
    so we must prove the pinned table actually steers play.

    Kazaross-XG2 vs jac100 on the INITIAL position, 7-point match at
    3-away/5-away, dice 6-2. Kazaross prices the gammon-go line higher
    and splits to 16; jac100 makes the standard money play. Snowie is
    too close to Kazaross to flip anything here.
    """
    a_kaz = GnubgExternalAgent(plies=0, met_path=KAZAROSS)
    a_j100 = GnubgExternalAgent(plies=0, met_path=JAC100)
    try:
        s, m, dice = BoardState.initial(), ms(7, my_away=3, opp_away=5), (6, 2)
        kaz = a_kaz._srv().request(line(a_kaz, s, WHITE, m, dice))
        j100 = a_j100._srv().request(line(a_j100, s, WHITE, m, dice))
        assert kaz == "24/18 18/16", kaz
        assert j100 == "24/18 13/11", j100
    finally:
        for a in (a_kaz, a_j100):
            if a._server:
                a._server.close()


def test_validate_accepts_tables_from_xml_refuses():
    """The pre-flight is deliberately weaker than met.from_xml: gnubg
    can use per-player post-Crawford tables that met.py will not read,
    and refusing them here would needlessly narrow the referee choice."""
    from met import MatchEquityTable, validate_met_file
    if not os.path.exists(JAC100):
        pytest.skip("jac100 MET missing")
    validate_met_file(JAC100)                       # accepted
    with pytest.raises(NotImplementedError):        # but not readable by us
        MatchEquityTable.from_xml(JAC100)


def test_bad_met_path_raises_before_launch():
    with pytest.raises(FileNotFoundError, match="MET not found"):
        _ExtServer(plies=0, met_path="/nonexistent/table.xml")


def test_malformed_met_raises_before_launch(tmp_path):
    """A file that exists but is not a MET SIGSEGVs gnubg; parsing it
    ourselves fails fast instead of burning ~100s in connect retries."""
    bad = tmp_path / "bad.xml"
    bad.write_text("<met><info><name>x</name></info></met>")
    with pytest.raises(Exception) as e:
        _ExtServer(plies=0, met_path=str(bad))
    assert "could not start gnubg" not in str(e.value)


@needs_gnubg
def test_met_default_is_kazaross():
    """gnubg's default already matches our training grounding, so
    pinning is defensive rather than corrective."""
    import met as met_mod
    assert os.path.basename(met_mod.DEFAULT_MET) == "Kazaross-XG2.xml"
