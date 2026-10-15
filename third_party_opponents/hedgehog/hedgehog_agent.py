"""HedgehogAgent: drives the HedgeHog engine (hedgehog-public) as a
money-game opponent implementing our Agent interface. H-H side spec:
``hedgehog:<model.ogxf>`` (cubeful or cubeless arenas; ``--oneply*``
selects HedgeHog depth 1).

Color mapping (verify_conversion.py round-trips it):
  our WHITE = 0  moves 24->1, home pts 1-6   <->  HH color 0 ("BG_BLACK"),
                                                  bar = HH pip 25
  our BLACK = 1  moves 1->24, home pts 19-24 <->  HH color 1 ("BG_WHITE"),
                                                  bar = HH pip 0
  our point k (1-based)                      <->  HH pip k  (identical)
  numeric player index is IDENTICAL between the two engines.

Cube owner mapping: our CubeOwner {CENTERED=0, WHITE=1, BLACK=2} refers to
our player ids; HH CubeOwner {0 centered, 1 owned-by-HH-white(=our BLACK),
2 owned-by-HH-black(=our WHITE)} -> ours 0->0, 1->2, 2->1.

Decisions delegated to HedgeHog:
  * checker play (cubeful): hh_pick_best; successors evaluated in the
    community-search convention (depth 0: static eval of each successor
    with the opponent on roll, flipped; depth>=1: community expectiminimax
    with the cube carried into the interior), ranked by Janowski cubeful
    equity for the mover's cube ownership;
  * checker play (cubeless): hh_pick_best_cubeless (plain cubeless money
    equity, no Janowski layer);
  * offer_double: hh_double_action (money game, Janowski, x=0.68);
  * respond_to_double: hh_take.

Legality ALWAYS comes from our ``get_legal_plays``; HedgeHog only ranks
the successors we hand it (its own movegen does not enforce the
must-play-larger-die rule in rare positions).

Engine limits:
  * Jacoby is a shim flag, set per game from match_state.jacoby
    (HH.set_jacoby); a shim without the setter can only play Jacoby off.
  * CUBE_MAX_VALUE is 64: it never doubles past that, and cube values are
    clamped before being handed to the engine.
  * No beavers.
"""
import os

from agents import Agent, CubeOffer
from backgammon_engine import BoardState, get_legal_plays, WHITE, BLACK
from third_party_opponents.hedgehog import hh_lib
from third_party_opponents.hedgehog.hh_lib import HH

_OWNER_MAP = {0: 0, 1: 2, 2: 1}

# HH double_action codes that mean "double"; NO_DOUBLE and TOO_GOOD mean
# roll on (and -1 = cannot double, absent by construction: the game loop
# only asks cube-eligible sides).
_DOUBLE_ACTIONS = {hh_lib.ACT_DOUBLE_TAKE, hh_lib.ACT_DOUBLE_PASS,
                   hh_lib.ACT_REDOUBLE_TAKE, hh_lib.ACT_REDOUBLE_PASS}


def board_to_hh(state) -> bytes:
    """Convert our BoardState to a HedgeHog 16-byte board."""
    hh_c0 = []  # HH color 0 = our WHITE
    hh_c1 = []  # HH color 1 = our BLACK
    for i, cnt in enumerate(state.points):
        if cnt > 0:
            hh_c0.extend([i + 1] * cnt)
        elif cnt < 0:
            hh_c1.extend([i + 1] * (-cnt))
    hh_c0.extend([25] * state.bar[WHITE])
    hh_c1.extend([0] * state.bar[BLACK])
    hh_c0.extend([255] * state.off[WHITE])
    hh_c1.extend([255] * state.off[BLACK])
    # Hard check (not assert): the C side memcpys exactly 15 bytes per
    # color, so a malformed board must fail loudly even under python -O.
    if len(hh_c0) != 15 or len(hh_c1) != 15:
        raise ValueError(f"checker count != 15: {hh_c0!r} {hh_c1!r}")
    # hh_board_from_pips(white_pos, black_pos): "white" = HH color 1
    return HH.board_from_pips(hh_c1, hh_c0)


def hh_to_ours(board: bytes, turn: int) -> BoardState:
    """Convert a HedgeHog board back to our BoardState (given whose turn)."""
    pips, colors, hh_white_off, hh_black_off = HH.decode(board)
    points = [0] * 24
    for p in range(1, 25):
        n = pips[p]
        if n:
            points[p - 1] = n if colors[p] == 0 else -n
    bar = [pips[25], pips[0]]           # [our WHITE, our BLACK]
    off = [hh_black_off, hh_white_off]  # HH color0 off = our WHITE off
    return BoardState(points=points, bar=bar, off=off, turn=turn)


class HedgehogAgent(Agent):
    """Money-game agent backed by the HedgeHog engine.

    ``depth=0`` is HedgeHog's static NN eval of each successor;
    ``depth=1`` runs its community expectiminimax one level below each
    successor (opponent's 21 rolls, best reply by 0-ply).
    """

    def __init__(self, model_path: str, depth: int = 0):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"HedgeHog model not found: {model_path} (.ogxf files are "
                "not tracked in the repo — see third_party_opponents/hedgehog/"
                "README.md for download URLs and md5s)")
        # One ensemble context per agent, freed with the process (each H-H
        # worker builds one agent per side).
        self.hh = HH(model_path)
        self.depth = depth
        self.model_path = model_path
        # decision counters (tests/test_hedgehog.py)
        self.n_moves = 0
        self.n_doubles_offered = 0
        self.n_takes = 0
        self.n_passes = 0

    # Cubeful interface.

    def choose_checker_action_cubeful(self, state, dice, match_state):
        """Legality from OUR engine; HedgeHog evaluates and ranks the
        legal successors (community-search convention, cube carried into
        the interior at depth>=1; see module docstring)."""
        HH.set_jacoby(match_state.jacoby)   # couple rule to the arena
        plays = get_legal_plays(state, dice)
        if not plays:
            return None
        player = state.turn
        cube_value = min(match_state.cube_value, 64)
        hh_owner = _OWNER_MAP[int(match_state.cube_owner)]
        # get_legal_plays already de-dups by resulting board, so plays
        # map 1:1 onto distinct successor boards.
        boards = [board_to_hh(s) for _, s in plays]
        idx = self.hh.pick_best(boards, player, cube_value, hh_owner,
                                self.depth)
        if idx < 0:
            raise RuntimeError(f"hh_pick_best failed: {state!r} dice={dice}")
        self.n_moves += 1
        return plays[idx][1], 0.0

    def offer_double(self, state, match_state) -> CubeOffer:
        HH.set_jacoby(match_state.jacoby)
        player = state.turn
        if match_state.cube_value >= 64:
            # HH's CUBE_MAX_VALUE is 64: it never doubles past that.
            return CubeOffer(should_double=False, _cache=None)
        hh_owner = _OWNER_MAP[int(match_state.cube_owner)]
        act, _eqs = self.hh.double_action(
            board_to_hh(state), player, min(match_state.cube_value, 32),
            hh_owner, self.depth)
        should = act in _DOUBLE_ACTIONS
        if should:
            self.n_doubles_offered += 1
        return CubeOffer(should_double=should, _cache=None)

    def respond_to_double(self, state, match_state, hint=None) -> bool:
        HH.set_jacoby(match_state.jacoby)
        doubler = state.turn  # loop calls us with the doubler still on roll
        hh_owner = _OWNER_MAP[int(match_state.cube_owner)]
        takes = self.hh.take(
            board_to_hh(state), doubler, min(match_state.cube_value, 64),
            hh_owner, self.depth)
        if takes:
            self.n_takes += 1
        else:
            self.n_passes += 1
        return takes

    # Cubeless interface. The cubeless loop calls choose_checker_action
    # with an explicit actions list. Ranking is HedgeHog's plain cubeless
    # money equity; cube decisions never occur in a cubeless game.

    def choose_checker_action(self, state, dice, actions):
        player = state.turn
        boards = [board_to_hh(s) for _, s in actions]
        idx = self.hh.pick_best_cubeless(boards, player, self.depth)
        if idx < 0:
            raise RuntimeError(
                f"hh_pick_best_cubeless failed: {state!r} dice={dice}")
        self.n_moves += 1
        return actions[idx]

    def evaluate(self, state):
        """Cubeless money equity from state.turn's (on-roll) perspective.

        Assumes a distribution-output model (all current .ogxf models
        are); for a scalar-equity model this formula is wrong.
        """
        w, gw, bgw, gl, bgl = self.hh.eval(board_to_hh(state), state.turn)
        return (2 * w - 1) + (gw - gl) + (bgw - bgl)
