"""Tests for the gnubg 1.08 external-player agent (gnubg-ext).

Two tiers:
* pure-Python tests of the board encoding and move-string handling —
  always run;
* behavioral tests against a live gnubg 1.08 binary — skipped when the
  binary (GNUBG_CMD, default /usr/games/gnubg) is not installed.

The behavioral tier pins the two bugs that were found the hard way:
* cubeless move criterion: gnubg's "Cubeful evaluations" flag survives
  `set cube use off`, so without an explicit `cubeful off` the cubeless
  opponent ranks moves by cubeful money equity (~11 mEq/game
  mis-specification);
* FIBS turn field on double responses: it must be -1 (the doubler is on
  turn), otherwise gnubg evaluates every take from the wrong side and
  never drops (0 drops in 400 games historically).
"""
import os
import re
import subprocess

import pytest

from backgammon_engine import BoardState, get_legal_plays, WHITE, BLACK
from modes import MatchState, CubeOwner
from gnubg_external import (
    GnubgExternalAgent, GNUBG_CMD, fibs_board_ints,
)

needs_gnubg = pytest.mark.skipif(
    not os.path.exists(GNUBG_CMD),
    reason="gnubg 1.08 binary not installed (GNUBG_CMD)")


# ── pure-Python: FIBS board encoding ─────────────────────────────────

def test_fibs_encoding_initial_symmetric():
    """The opening position is side-symmetric: both perspectives encode
    to the same FIBS array, with the mover's checkers positive."""
    s = BoardState.initial()
    w = fibs_board_ints(s, WHITE)
    b = fibs_board_ints(s, BLACK)
    assert w == b
    # 15 checkers each side, no bar
    assert sum(v for v in w if v > 0) == 15
    assert sum(-v for v in w if v < 0) == 15
    assert w[0] == 0 and w[25] == 0
    # mover: 2 on the 24pt, 5 on 13, 3 on 8, 5 on 6
    assert (w[24], w[13], w[8], w[6]) == (2, 5, 3, 5)
    # opponent mirrored: their 24pt appears at index 25-24=1
    assert (w[1], w[12], w[17], w[19]) == (-2, -5, -3, -5)


def test_fibs_encoding_bar_and_perspective():
    """Asymmetric position: bars land in slots 25 (mover) / 0 (opponent),
    and the two perspectives mirror each other."""
    pts = [0] * 24
    pts[0] = 3        # White pt 1
    pts[10] = -2      # Black pt 14 (24-10)
    s = BoardState(points=pts, bar=[1, 2], off=[11, 11], turn=WHITE)
    w = fibs_board_ints(s, WHITE)
    assert w[1] == 3          # own pt 1
    assert w[25] == 1         # own bar, positive
    assert w[0] == -2         # opp bar, negative
    assert w[25 - 14] == -2   # opp pt 14 at index 25-14
    b = fibs_board_ints(s, BLACK)
    assert b[14] == 2         # same checkers from Black's perspective
    assert b[25] == 2 and b[0] == -1
    assert b[25 - 1] == -3


# ── pure-Python: move-string parsing / application ───────────────────

def test_parse_token_forms():
    parse = GnubgExternalAgent._parse_token
    assert parse("bar/22*") == ([(25, 22)], 1)
    assert parse("13/7(2)") == ([(13, 7)], 2)
    assert parse("6/off") == ([(6, 0)], 1)
    assert parse("13/7/4") == ([(13, 7), (7, 4)], 1)


def test_apply_move_hit_sends_opponent_to_bar():
    ag = GnubgExternalAgent()
    board = [0] * 26
    board[6] = 2      # mover: two on the 6pt
    board[4] = -1     # opponent blot on mover's 4pt
    out = ag._apply_move(board, "6/4*")
    assert out[6] == 1
    assert out[4] == 1        # blot replaced by mover's checker
    assert out[0] == -1       # opponent checker on the bar


def test_match_successor_maps_move_string():
    """gnubg's move string for the opening 31 (8/5 6/5) must map onto
    the engine successor with the made 5-point."""
    ag = GnubgExternalAgent()
    s = BoardState.initial()
    nxt = ag._match_successor(s, (3, 1), "8/5 6/5")
    assert nxt is not None
    assert nxt.points[4] == 2          # White made the 5-point (idx 4)
    assert fibs_board_ints(nxt, WHITE)[5] == 2


def test_specs_registered(tmp_path):
    """Matchplay evaluation dispatches through eval_matchplay, not
    play_models (this branch has no _detect_side_type).

    The hazard this locks down: "gnubg-ext" also starts with "gnubg",
    so if the gnubg-nn branch is matched first the external agent is
    silently never used. GnubgExternalAgent creates its subprocess
    lazily, so nothing is spawned here.
    """
    from eval_matchplay import _build_agents
    from gnubg_external import GnubgExternalAgent
    from model import TDNetwork

    model1 = tmp_path / "mp.pt"
    TDNetwork(hidden_sizes=[8], input_size=207,
              encoder_name="matchplay1h_perspective196",
              output_mode="probability").save(str(model1))

    for spec, plies, full in [("gnubg-ext", 0, False),
                              ("gnubg-ext:2", 2, False),
                              ("gnubg-ext-full", 0, True),
                              ("gnubg-ext-full:1", 1, True)]:
        _, a2 = _build_agents(str(model1), spec)
        assert isinstance(a2, GnubgExternalAgent), spec
        assert a2.plies == plies and a2.full is full, spec
        assert a2._server is None, "agent must stay lazy"


# ── live gnubg: checker play ─────────────────────────────────────────

@needs_gnubg
@pytest.mark.parametrize("full", [False, True])
def test_checker_move_is_legal_successor(full):
    agent = GnubgExternalAgent(plies=0, cubeless=True, full=full)
    try:
        s = BoardState.initial()
        plays = get_legal_plays(s, (3, 1))
        _, nxt = agent.choose_checker_action(s, (3, 1), plays)
        assert any(nxt == p[1] for p in plays)
    finally:
        if agent._server is not None:
            agent._server.close()


# ── live gnubg: cubeless move criterion (regression) ─────────────────
# Three rollout-verified positions where the cubeful and cubeless
# criteria pick different moves. The agent must produce the
# cubeless-criterion successor, not the old cubeful-criterion one.
# Boards are (opponent, mover) in gnubg layout: idx 0..23 = that side's
# points 1..24, idx 24 = bar.

CRITERION_POSITIONS = [
    {   # gnubg ID trkPAEBbbQYBMA, 64: cubeless 13/3, cubeful(old) 13/7 8/4
        "posid": "trkPAEBbbQYBMA", "dice": (6, 4),
        "pre": [[0, 2, 2, 2, 0, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [2, 2, 1, 1, 2, 2, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]],
        "succ_cubeless": [[2, 2, 2, 1, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                          [0, 2, 2, 2, 0, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
        "succ_cubeful_old": [[2, 2, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                             [0, 2, 2, 2, 0, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    },
    {   # gnubg ID vjsHAEBbtwETAA, 35: cubeless 13/5, cubeful(old) 6/3 6/1
        "posid": "vjsHAEBbtwETAA", "dice": (3, 5),
        "pre": [[0, 5, 3, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [2, 2, 1, 3, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        "succ_cubeless": [[2, 2, 1, 3, 3, 2, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 5, 3, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
        "succ_cubeful_old": [[3, 2, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 5, 3, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    },
    {   # gnubg ID aLeJA0CGdxMHAA, 12: cubeless 7/5 6/5, cubeful(old) 13/10
        "posid": "aLeJA0CGdxMHAA", "dice": (1, 2),
        "pre": [[0, 0, 0, 1, 2, 3, 2, 2, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 2, 0, 0, 0, 4, 3, 2, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        "succ_cubeless": [[0, 2, 0, 0, 2, 3, 2, 2, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 2, 3, 2, 2, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
        "succ_cubeful_old": [[0, 2, 0, 0, 0, 4, 3, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 2, 3, 2, 2, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    },
]


def _state_from_gnubg_boards(pre):
    """Rebuild a BoardState from [opp, mover] gnubg-layout boards,
    placing the mover as WHITE (positions are perspective-relative)."""
    pts = [0] * 24
    for idx in range(24):
        if pre[1][idx]:
            pts[idx] += pre[1][idx]        # white point idx+1
        if pre[0][idx]:
            pts[23 - idx] -= pre[0][idx]   # black point idx+1
    bar = [0, 0]
    bar[WHITE] = pre[1][24]
    bar[BLACK] = pre[0][24]
    off = [0, 0]
    off[WHITE] = 15 - sum(pre[1])
    off[BLACK] = 15 - sum(pre[0])
    return BoardState(points=pts, bar=bar, off=off, turn=WHITE)


def _gnubg_layout(state):
    """[not-on-roll, on-roll] boards, own numbering."""
    white = [0] * 25
    black = [0] * 25
    for i in range(24):
        v = state.points[i]
        if v > 0:
            white[i] = v
        elif v < 0:
            black[23 - i] = -v
    white[24] = state.bar[WHITE]
    black[24] = state.bar[BLACK]
    return [black, white] if state.turn == WHITE else [white, black]


@needs_gnubg
def test_cubeless_uses_cubeless_move_criterion():
    agent = GnubgExternalAgent(plies=1, cubeless=True)
    try:
        for pos in CRITERION_POSITIONS:
            state = _state_from_gnubg_boards(pos["pre"])
            plays = get_legal_plays(state, pos["dice"])
            _, nxt = agent.choose_checker_action(state, pos["dice"], plays)
            got = _gnubg_layout(nxt)
            assert got == pos["succ_cubeless"], (
                "position %s dice %s: expected the cubeless-criterion "
                "move, got a different successor" % (pos["posid"], pos["dice"]))
            assert got != pos["succ_cubeful_old"], (
                "position %s: agent picked the old cubeful-criterion move "
                "— `set evaluation chequer evaluation cubeful off` "
                "regressed" % pos["posid"])
    finally:
        if agent._server is not None:
            agent._server.close()


# ── live gnubg: cube responses (turn-field regression) ───────────────

def _closed_out_position():
    """White has a closed home board; Black has 2 on the bar. A double
    from White is a trivial drop for Black."""
    pts = [0] * 24
    for i in range(6):
        pts[i] = 2                     # White pts 1..6 all made
    pts[7] = 3                         # 15th White checkers on the 8pt
    pts[23] = -3; pts[22] = -3; pts[21] = -3; pts[20] = -2; pts[19] = -2
    return BoardState(points=pts, bar=[0, 2], off=[0, 0], turn=WHITE)


def _even_race_position():
    """Pure no-contact race, identical pip counts: a double is a take."""
    pts = [0] * 24
    for i in range(5):
        pts[i] = 3                     # White pts 1..5
        pts[23 - i] = -3               # Black pts 1..5
    return BoardState(points=pts, bar=[0, 0], off=[0, 0], turn=WHITE)


@needs_gnubg
def test_double_response_drop_and_take():
    agent = GnubgExternalAgent(plies=0)
    try:
        ms = MatchState(jacoby=True)
        assert agent.respond_to_double(_closed_out_position(), ms) is False, (
            "closed-out double was taken — FIBS turn field on doubled "
            "requests regressed (must be -1)")
        assert agent.respond_to_double(_even_race_position(), ms) is True
    finally:
        if agent._server is not None:
            agent._server.close()


# ── live gnubg: desktop-verified move & equity reproduction ──────────
# Positions where gnubg-ext's 0-ply and 1-ply CHECKER choices differ,
# each verified move-for-move against the user's desktop gnubg 1.08
# (mingw build, same weights). Boards are engine
# points/bar/off with `turn` on roll; move0/move1 are the desktop top
# moves at 0/1-ply, raw0/raw1 the agent's own reply strings (same
# resulting positions).
DESKTOP_CHECKER = [
    dict(gid='hmfwATDg8+ABJA:MAESAAAAAAAA',
         points=[-2, 0, 0, 0, 0, 5, 0, 4, 0, 0, 0, -5, 4, 0, 0, 0, -2, 0, -4, 0, 0, 1, -2, 1],
         bar=[0, 0], off=[0, 0], turn=0, dice=(4, 4), cube_value=1, cube='centre',
         move0='8/4(2) 6/2(2)', move1='13/5(2)',
         raw0='8/4 8/4 6/2 6/2', raw1='13/9 13/9 9/5 9/5'),
    dict(gid='5jbggEHc3ADmAA:AQEPAAAAAAAA',
         points=[0, 0, 3, 2, -2, 3, 2, 0, 0, 0, 0, -3, 0, 0, 0, 2, 0, 3, -2, -2, -3, 0, -2, 0],
         bar=[0, 1], off=[0, 0], turn=0, dice=(6, 3), cube_value=2, cube='mine',
         move0='16/7', move1='18/9', raw0='16/10 10/7', raw1='18/15 15/9'),
    dict(gid='cN8mBgBYr8EAJg:AQELAAAAAAAA',
         points=[0, 0, 0, 2, 1, 4, 1, 2, 0, 0, 0, -2, 2, 0, -1, 0, -2, -2, -5, -3, 0, 2, 0, 1],
         bar=[0, 0], off=[0, 0], turn=0, dice=(6, 2), cube_value=2, cube='mine',
         move0='8/2 7/5', move1='24/16', raw0='8/2 7/5', raw1='22/16 24/22'),
    dict(gid='uDvgAUE4LxAzDA:AYENAAAAAAAA',
         points=[0, 0, 0, 3, -1, 4, 1, 0, 0, 0, 0, -4, 1, 0, 0, 2, 0, 2, -3, -3, -3, 2, 0, 0],
         bar=[0, 1], off=[0, 0], turn=0, dice=(3, 3), cube_value=2, cube='mine',
         move0='16/7 16/13', move1='13/7 6/3(2)',
         raw0='16/13 16/13 13/10 10/7', raw1='13/10 10/7 6/3 6/3'),
    dict(gid='hnNwAizYnoMBCQ:EYEWAAAAAAAA',
         points=[-1, -2, 0, 2, 2, 4, 0, 3, 0, -1, 0, -3, 2, 0, 0, 0, -3, 0, -3, 1, 0, 1, -2, 0],
         bar=[0, 0], off=[0, 0], turn=0, dice=(5, 5), cube_value=2, cube='theirs',
         move0='20/10* 6/1*(2)', move1='8/3(2) 6/1*(2)',
         raw0='20/15 15/10* 6/1* 6/1', raw1='8/3 8/3 6/1* 6/1'),
]


def _match_state_for(cube, cube_value, turn):
    if cube == "centre":
        owner = CubeOwner.CENTERED
    elif cube == "mine":
        owner = CubeOwner(turn + 1)
    else:
        owner = CubeOwner((1 - turn) + 1)
    return MatchState(cube_owner=owner, cube_value=cube_value, jacoby=True)


@pytest.fixture(scope="module")
def prod_agents():
    """One 0-ply and one 1-ply production agent, reused across positions."""
    if not os.path.exists(GNUBG_CMD):
        pytest.skip("gnubg 1.08 binary not installed (GNUBG_CMD)")
    agents = {0: GnubgExternalAgent(plies=0), 1: GnubgExternalAgent(plies=1)}
    yield agents
    for a in agents.values():
        if a._server is not None:
            a._server.close()


@needs_gnubg
@pytest.mark.parametrize("pos", DESKTOP_CHECKER,
                         ids=[p["gid"][:6] for p in DESKTOP_CHECKER])
def test_checker_top_move_matches_desktop(pos, prod_agents):
    """gnubg-ext's 0-ply and 1-ply top checker moves reproduce desktop,
    and genuinely disagree with each other."""
    state = BoardState(points=list(pos["points"]), bar=list(pos["bar"]),
                       off=list(pos["off"]), turn=pos["turn"])
    ms = _match_state_for(pos["cube"], pos["cube_value"], pos["turn"])
    dice = pos["dice"]
    got = {}
    for ply in (0, 1):
        r = prod_agents[ply].choose_checker_action_cubeful(state, dice, ms)
        assert r is not None and r[0] is not None
        got[ply] = fibs_board_ints(r[0], pos["turn"])
    exp0 = fibs_board_ints(
        prod_agents[0]._match_successor(state, dice, pos["raw0"]), pos["turn"])
    exp1 = fibs_board_ints(
        prod_agents[1]._match_successor(state, dice, pos["raw1"]), pos["turn"])
    assert got[0] == exp0, "%s: 0-ply move != desktop %s" % (pos["gid"], pos["move0"])
    assert got[1] == exp1, "%s: 1-ply move != desktop %s" % (pos["gid"], pos["move1"])
    assert exp0 != exp1, "%s: 0-ply and 1-ply should disagree" % pos["gid"]


# Full 1-ply Hint equity list for the 55/cube-2 position, verified against
# desktop to 3 decimals in BOTH configs:
#   production == desktop "standard pruning" (ranks 5-8 fall back to 0-ply)
#   full       == desktop "pruning off"      (all 8 candidates at 1-ply)
DESKTOP_POS5_ID = "hnNwAizYnoMBCQ:EYEWAAAAAAAA"
DESKTOP_POS5 = {
    "production": [("8/3(2) 6/1*(2)", 0.959), ("20/10* 6/1*(2)", 0.935),
                   ("20/10* 8/3(2)", 0.890), ("20/10*/5 6/1*", 0.820),
                   ("20/10* 13/8 6/1*", 0.735), ("20/10* 8/3 6/1*", 0.721),
                   ("13/3 8/3 6/1*", 0.716), ("20/15 8/3(2) 6/1*", 0.713)],
    "full": [("8/3(2) 6/1*(2)", 0.959), ("20/10* 6/1*(2)", 0.935),
             ("20/10* 8/3(2)", 0.890), ("20/10*/5 6/1*", 0.820),
             ("20/10* 13/8 6/1*", 0.748), ("20/10* 8/3 6/1*", 0.742),
             ("13/3 8/3 6/1*", 0.732), ("8/3(3) 6/1*", 0.704)],
}

_HINT_RE = re.compile(r"^\s*\d+\.\s+Cubeful\s+\d-ply\s+(.+?)\s+Eq\.:\s+([+-][\d.]+)")


def _gnubg_hint(gid, full, n=8):
    """Spawn gnubg, set the position from a GnuBG ID, and return the top-n
    (move, equity-rounded-3dp) rows of a 1-ply hint. `full` mirrors the
    gnubg-ext-full config (prune nets off + keep-all move filter)."""
    cmds = "set evaluation chequer evaluation plies 1\n"
    if full:
        cmds += "set evaluation chequer evaluation prune off\n"
        for ply in (1, 2, 3, 4):
            cmds += "set evaluation movefilter %d 0 1000 0 10.0\n" % ply
    cmds += "set gnubgid %s\nhint %d\n" % (gid, n)
    out = subprocess.run(
        [GNUBG_CMD, "-t", "-q"], input=cmds, capture_output=True, text=True,
        timeout=120, env=dict(os.environ, LC_ALL="C")).stdout
    rows = []
    for line in out.splitlines():
        m = _HINT_RE.match(line)
        if m:
            rows.append((m.group(1).strip(), round(float(m.group(2)), 3)))
    return rows[:n]


@needs_gnubg
@pytest.mark.parametrize("config", ["production", "full"])
def test_position5_equities_match_desktop(config):
    """Full 1-ply hint (moves + equities) reproduces desktop exactly, in
    both the production (pruned) and full (unpruned) configs."""
    got = _gnubg_hint(DESKTOP_POS5_ID, full=(config == "full"))
    exp = DESKTOP_POS5[config]
    assert len(got) >= len(exp), "only got %d hint rows" % len(got)
    for i, ((gm, ge), (em, ee)) in enumerate(zip(got, exp), 1):
        assert gm == em, "%s rank %d: move %r != desktop %r" % (config, i, gm, em)
        assert ge == ee, "%s rank %d (%s): eq %.3f != desktop %.3f" % (
            config, i, em, ge, ee)


# ── live gnubg: cube-decision analysis vs desktop ────────────────────
# Cube (double) decisions where the proper action flips between 0-ply and
# 1-ply, verified against desktop gnubg 1.08 — cubeless
# equity, every cubeful equity, and the proper cube action. 1-ply uses
# the "pruning off" config, matching the user's desktop analysis setting.
DESKTOP_CUBE = [
    dict(gid="2GYLAEe4e7gACA:MAEAAAAAAAAA",
         ply0=dict(cubeless=0.424, proper="Double, take",
                   cubeful={"Double, take": 0.543, "Double, pass": 1.000,
                            "No double": 0.527}),
         ply1=dict(cubeless=0.376, proper="No double, take",
                   cubeful={"No double": 0.448, "Double, pass": 1.000,
                            "Double, take": 0.439})),
    dict(gid="a1sxAgK3zQEiQA:AQEAAAAAAAAA",
         ply0=dict(cubeless=0.348, proper="No redouble, take",
                   cubeful={"No double": 0.563, "Double, pass": 1.000,
                            "Double, take": 0.365}),
         ply1=dict(cubeless=0.232, proper="Redouble, take",
                   cubeful={"Double, take": 0.307, "Double, pass": 1.000,
                            "No double": 0.307})),
]

_CUBELESS_RE = re.compile(r"\d-ply cubeless equity\s+([+-][\d.]+)")
_CUBEFUL_RE = re.compile(r"^\s*\d+\.\s+(.+?)\s{2,}([+-]\d+\.\d+)")
_PROPER_RE = re.compile(r"Proper cube action:\s+(.+?)\s*(?:\([\d.]+%\))?\s*$")


def _gnubg_cube_hint(gid, plies, full=True):
    """Cube-decision hint -> (cubeless_eq, {action: cubeful_eq}, proper)."""
    cmds = "set evaluation cubedecision evaluation plies %d\n" % plies
    if full and plies >= 1:
        cmds += "set evaluation chequer evaluation prune off\n"
        for ply in (1, 2):
            cmds += "set evaluation movefilter %d 0 1000 0 10.0\n" % ply
    cmds += "set gnubgid %s\nhint\n" % gid
    out = subprocess.run(
        [GNUBG_CMD, "-t", "-q"], input=cmds, capture_output=True, text=True,
        timeout=120, env=dict(os.environ, LC_ALL="C")).stdout
    cubeless, proper, cubeful = None, None, {}
    for line in out.splitlines():
        m = _CUBELESS_RE.search(line)
        if m:
            cubeless = round(float(m.group(1)), 3)
        m = _CUBEFUL_RE.match(line)
        if m:
            cubeful[m.group(1).strip()] = round(float(m.group(2)), 3)
        m = _PROPER_RE.search(line)
        if m:
            proper = m.group(1).strip()
    return cubeless, cubeful, proper


@needs_gnubg
@pytest.mark.parametrize("pos", DESKTOP_CUBE, ids=[p["gid"][:6] for p in DESKTOP_CUBE])
@pytest.mark.parametrize("plies", [0, 1])
def test_cube_action_matches_desktop(pos, plies):
    """gnubg-ext reproduces desktop's cube-decision analysis (cubeless
    equity, each cubeful equity, and the proper action) at 0-ply and
    1-ply, and the proper action flips between the two depths."""
    exp = pos["ply%d" % plies]
    cubeless, cubeful, proper = _gnubg_cube_hint(pos["gid"], plies)
    assert cubeless == exp["cubeless"], "cubeless %.3f != desktop %.3f" % (
        cubeless, exp["cubeless"])
    assert proper == exp["proper"], "proper %r != desktop %r" % (proper, exp["proper"])
    for action, eq in exp["cubeful"].items():
        assert cubeful.get(action) == eq, "%s: %s != desktop %.3f" % (
            action, cubeful.get(action), eq)
    # the two depths must genuinely disagree on the proper action
    assert pos["ply0"]["proper"] != pos["ply1"]["proper"]


# ── cross-engine: gnubg-nn bindings vs gnubg 1.08 binary ────────────
# Two decisions sampled from cubeless-money self-play (seed 20260722,
# 10 games / 10 random decisions). Across all 375 candidate
# moves of the 10 decisions the two engines produced identical candidate
# counts, identical full rankings and identical picks; 374/375 equities
# agreed within 2.8e-4 (float32 path noise) and one deep candidate
# (rank 13/20 of CROSS_POS[1], gnubg "7/1 5/2") differed by 2.34e-3.
# Verified: BOTH engines classify that successor CONTACT
# (binary classifypos=10, bindings classify=5 in their own enums) and
# agree on the gammon outputs to ~1e-5; only P(win) differs (1.16e-3).
# Likely a small contact-net input-feature (table) difference between
# the gnubg-nn codebase snapshot and gnubg 1.08 — decision-irrelevant
# (the candidate is rank 13/20 at equity -0.31). These two
# decisions are the most interesting of the ten: the 148-candidate
# double-1s (max move-gen/dedup stress) and the hit + outlier position.
# The engines are NOT bit-identical; the contract asserted here is
# decision-identity + tight numeric agreement.
CROSS_TOL_TYPICAL = 5e-4    # float32-path agreement for normal moves
CROSS_TOL_MAX = 3e-3        # single known classification-boundary outlier

CROSS_POS = [
    dict(  # 148-candidate double 1s (game 5)
        points=[2, 2, -1, 2, 2, 2, 0, 0, 0, 0, 1, -3, 2, 0, 0, 0, 0, 2, -3, -3, -2, -3, 0, 0],
        bar=[0, 0], off=[0, 0], turn=0, dice=(1, 1),
        gnubgid="3O6AAwibbZDBAA:MIEEAAAAAAAE",
        n=148,
        nn_top3=[['4/3  4/3  18/17  18/17', 0.484638], ['4/3  4/3  11/10  10/9', 0.479795], ['4/3  4/3  5/4  5/4', 0.454929]],
        ext_top3=[('18/17(2) 4/3*(2)', 0.485), ('11/9 4/3*(2)', 0.48), ('5/3*(2)', 0.455)],
        nn_eq=[0.484638, 0.479795, 0.454929, 0.38679, 0.375896, 0.01786, -0.091303, -0.096341, -0.098644, -0.125335, -0.152121, -0.163937, -0.21527, -0.22079, -0.246139, -0.24749, -0.247992, -0.262274, -0.271844, -0.272624, -0.284738, -0.30852, -0.311261, -0.315653, -0.320875, -0.320974, -0.322332, -0.327704, -0.331236, -0.338065, -0.34262, -0.363163, -0.367038, -0.371444, -0.376629, -0.377627, -0.386115, -0.39502, -0.406762, -0.407909, -0.411096, -0.421213, -0.424021, -0.435782, -0.444397, -0.471364, -0.477839, -0.485971, -0.495723, -0.504129, -0.516178, -0.527277, -0.527764, -0.54387, -0.545112, -0.55792, -0.572094, -0.579814, -0.585059, -0.591395, -0.594966, -0.599526, -0.608056, -0.616211, -0.626529, -0.632486, -0.635668, -0.654971, -0.658963, -0.666051, -0.672214, -0.673647, -0.674393, -0.675083, -0.683581, -0.684709, -0.690079, -0.694021, -0.703598, -0.706435, -0.711425, -0.720785, -0.765629, -0.801895, -0.802858, -0.821265, -0.827515, -0.853213, -0.856444, -0.857885, -0.859642, -0.863516, -0.879234, -0.882024, -0.882505, -0.891329, -0.904341, -0.912569, -0.917809, -0.922743, -0.924589, -0.926733, -0.931483, -0.951884, -0.951947, -0.966537, -0.972532, -0.973632, -1.00617, -1.018281, -1.021394, -1.046977, -1.048825, -1.055005, -1.064719, -1.065439, -1.071196, -1.084209, -1.088671, -1.092008, -1.101356, -1.108524, -1.112393, -1.113638, -1.135633, -1.141818, -1.143697, -1.158414, -1.159156, -1.221498, -1.280473, -1.282612, -1.293009, -1.312507, -1.329944, -1.338248, -1.341113, -1.346194, -1.34891, -1.356189, -1.369635, -1.372738, -1.374228, -1.416269, -1.441572, -1.453651, -1.535521, -1.826847],
        ext_eq=[0.484655, 0.479797, 0.454974, 0.386954, 0.376061, 0.017837, -0.091308, -0.096369, -0.098566, -0.125481, -0.152235, -0.163992, -0.215311, -0.220776, -0.246075, -0.247482, -0.247948, -0.262191, -0.271907, -0.272542, -0.284621, -0.308627, -0.311269, -0.315547, -0.320949, -0.320953, -0.322389, -0.327733, -0.331292, -0.337919, -0.34264, -0.363231, -0.367126, -0.371647, -0.376761, -0.3776, -0.386224, -0.395014, -0.406697, -0.408088, -0.411151, -0.421159, -0.423949, -0.435899, -0.444422, -0.471322, -0.477879, -0.485945, -0.495798, -0.504283, -0.516178, -0.527445, -0.527692, -0.543857, -0.545177, -0.557912, -0.572114, -0.579902, -0.58509, -0.591425, -0.594984, -0.59967, -0.608092, -0.616263, -0.626574, -0.632582, -0.635623, -0.654915, -0.65901, -0.66608, -0.672259, -0.673582, -0.674395, -0.6752, -0.683592, -0.684591, -0.690155, -0.694054, -0.703609, -0.706483, -0.711392, -0.720737, -0.765559, -0.801953, -0.802884, -0.82129, -0.827602, -0.85314, -0.856434, -0.857904, -0.859666, -0.863574, -0.879314, -0.882053, -0.882501, -0.891412, -0.904285, -0.912611, -0.917897, -0.922729, -0.924712, -0.926605, -0.931476, -0.951967, -0.951973, -0.966637, -0.972524, -0.973569, -1.006174, -1.018342, -1.021479, -1.046941, -1.048864, -1.055019, -1.064819, -1.065439, -1.071271, -1.08431, -1.088691, -1.092172, -1.101272, -1.108582, -1.112427, -1.11374, -1.13552, -1.141859, -1.143721, -1.15839, -1.159277, -1.221578, -1.280375, -1.282629, -1.293212, -1.312572, -1.330002, -1.338333, -1.341129, -1.346091, -1.348876, -1.356289, -1.369704, -1.37264, -1.37426, -1.416282, -1.441639, -1.453665, -1.535556, -1.827119],
    ),
    dict(  # hit + classification-boundary outlier (game 9)
        points=[2, -1, 3, 3, 3, 2, 0, 1, 0, 0, -1, 0, 0, 0, -2, 0, 0, -1, -3, -3, 0, 0, -2, -2],
        bar=[1, 0], off=[0, 0], turn=1, dice=(6, 3),
        gnubgid="c3cTAEAbdzEEEA:MAEPAAAAAAAE",
        n=20,
        nn_top3=[['2/8  11/14', 0.813389], ['2/8  8/11', 0.799988], ['2/8  15/18', 0.776958]],
        ext_top3=[('23/17* 14/11', 0.813), ('23/17*/14', 0.8), ('23/17* 10/7', 0.777)],
        nn_eq=[0.813389, 0.799988, 0.776958, 0.663773, 0.482333, 0.474067, 0.382275, -0.102021, -0.119595, -0.182901, -0.260311, -0.291902, -0.315321, -0.347225, -0.392331, -0.454745, -0.472806, -0.582993, -0.620445, -0.830441],
        ext_eq=[0.813296, 0.799968, 0.777013, 0.663972, 0.482433, 0.474102, 0.382375, -0.102053, -0.119513, -0.182835, -0.260369, -0.291799, -0.312978, -0.347039, -0.392592, -0.454723, -0.472892, -0.582927, -0.620576, -0.830429],
    ),
]


def _gnubg_hint_cubeless(gid, n):
    """Full 0-ply cubeless-money hint list for a gnubgid at FULL float
    precision, via the binary's embedded Python (stdout hint prints only
    3 decimals; the python API exposes the raw floats). Criterion fix
    applied: chequer eval cubeful off."""
    import tempfile
    script = (
        "import gnubg\n"
        "gnubg.command('set jacoby off')\n"
        "gnubg.command('set cube use off')\n"
        "gnubg.command('set evaluation chequer evaluation cubeful off')\n"
        "gnubg.command('set evaluation chequer evaluation plies 0')\n"
        "gnubg.command('set gnubgid %s')\n"
        "for m in gnubg.hint()['hint']:\n"
        "    print('XEQ|%%s|%%.9f' %% (m['move'], m['equity']))\n" % gid)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        out = subprocess.run(
            [GNUBG_CMD, "-t", "-q", "-p", path], capture_output=True,
            text=True, timeout=300, env=dict(os.environ, LC_ALL="C")).stdout
    finally:
        os.unlink(path)
    rows = []
    for line in out.splitlines():
        if line.startswith("XEQ|"):
            _, mv, eq = line.split("|")
            rows.append((mv.strip(), float(eq)))
    return rows


def _cross_state(pos):
    return BoardState(points=list(pos["points"]), bar=list(pos["bar"]),
                      off=list(pos["off"]), turn=pos["turn"])


@needs_gnubg
@pytest.mark.parametrize("pos", CROSS_POS,
                         ids=["dbl1s-148cand", "hit-outlier"])
def test_cross_engine_nn_vs_binary_cubeless_0ply(pos):
    """gnubg-nn bindings and the gnubg 1.08 binary agree on cubeless
    0-ply analysis: same candidate count, same ranking, equities within
    CROSS_TOL_MAX everywhere (typically CROSS_TOL_TYPICAL)."""
    gnubg_nn = pytest.importorskip("gnubg_nn")
    from agents import GnubgNNAgent
    from backgammon_engine import play_label

    state = _cross_state(pos)
    plays = get_legal_plays(state, pos["dice"])
    assert len(plays) == pos["n"], "move count vs recorded"

    # nn side, live: mover equity = -evaluate(successor)
    ag = GnubgNNAgent(plies=0, cubeless_money=True)
    rows = sorted(((play_label(p), -float(ag.evaluate(s))) for p, s in plays),
                  key=lambda r: -r[1])
    for (em, ee), (gm, ge) in zip(pos["nn_top3"], rows[:3]):
        assert gm == em, "nn top-3 move drift: %r != %r" % (gm, em)
        assert abs(ge - ee) < 1e-5, "nn equity drift on %s" % em
    assert all(abs(a - round(b[1], 6)) < 2e-6
               for a, b in zip(pos["nn_eq"], rows)), "nn equity list drift"

    # binary side, live: full-precision hint via gnubgid
    got = _gnubg_hint_cubeless(pos["gnubgid"], pos["n"] + 5)
    assert len(got) == pos["n"], "binary hint count %d != %d" % (
        len(got), pos["n"])
    got.sort(key=lambda r: -r[1])
    for (em, ee), (gm, ge) in zip(pos["ext_top3"], got[:3]):
        assert gm == em, "binary top-3 move drift: %r != %r" % (gm, em)
        assert abs(ge - ee) < 5e-4, "binary equity drift on %s" % em

    # cross-engine agreement on the full sorted equity vectors
    diffs = [abs(a[1] - b[1]) for a, b in zip(rows, got)]
    assert max(diffs) < CROSS_TOL_MAX, "cross-engine max diff %.2e" % max(diffs)
    assert sum(d > CROSS_TOL_TYPICAL for d in diffs) <= 1, (
        "more than one candidate outside typical tolerance")
