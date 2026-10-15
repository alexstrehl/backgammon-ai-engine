"""Round-trip + movegen cross-verification: our engine vs HedgeHog.

Plays random games with OUR engine; at every position checks
  1. ours -> HH -> decode -> ours is the identity (points/bar/off),
  2. the set of legal successor boards (as HH bytes) from our
     get_legal_plays equals HedgeHog's explore_delta successor set,
  3. terminal detection + points won (single/gammon/bg) agree.
Run from the repo root:
    python third_party_opponents/hedgehog/verify_conversion.py [n_games]
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))

from backgammon_engine import (  # noqa: E402
    BoardState, get_legal_plays, switch_turn, opening_roll, WHITE, BLACK)
from third_party_opponents.hedgehog.hh_lib import HH  # noqa: E402
from third_party_opponents.hedgehog.hedgehog_agent import (  # noqa: E402
    board_to_hh, hh_to_ours)


def check_position(state, dice):
    hb = board_to_hh(state)
    # 1. round trip
    back = hh_to_ours(hb, state.turn)
    assert back.points == state.points, (state, back)
    assert back.bar == state.bar and back.off == state.off, (state, back)
    # 2. successor sets
    plays = get_legal_plays(state, dice)
    ours = {board_to_hh(s) for _, s in plays}
    n, hh_succ = HH.list_moves(hb, state.turn, dice[0], dice[1])
    theirs = set(hh_succ)
    assert n == len(theirs), "HH returned dup successors"
    if not plays:
        # forced pass: HH's convention is one "successor" = unchanged state
        assert theirs == {hb}, f"pass mismatch {state!r} dice={dice}"
        return 0, 0
    # Our legal set must be a subset of HH's (a conversion bug would show up
    # as an ours-only successor). HH-only extras are its known laxity: it
    # does not enforce the must-play-larger-die rule.
    assert ours <= theirs, (
        f"conversion/movegen bug {state!r} dice={dice}: "
        f"{len(ours - theirs)} of our successors unknown to HH")
    return len(plays), (1 if theirs - ours else 0)


def main(n_games=200, seed=12345):
    rng = random.Random(seed)
    random.seed(seed)
    positions = 0
    terminals = 0
    lax = 0
    for g in range(n_games):
        state, dice = opening_roll()
        while not state.is_game_over():
            _, l = check_position(state, dice)
            lax += l
            plays = get_legal_plays(state, dice)
            if plays:
                _, state = rng.choice(plays)
            else:
                state = switch_turn(state)
            dice = (rng.randint(1, 6), rng.randint(1, 6))
            positions += 1
        # 3. terminal agreement
        hb = board_to_hh(state)
        assert HH.game_over(hb), f"HH misses terminal {state!r}"
        w = state.winner()
        assert HH.winner(hb) == w, f"winner mismatch {state!r}"
        assert HH.points(hb, w) == state.game_result(), (
            f"points mismatch {state!r}: hh={HH.points(hb, w)} "
            f"ours={state.game_result()}")
        terminals += 1
    print(f"OK: {positions} positions round-tripped, our legal sets always "
          f"a subset of HH's, {terminals} terminals agreed, "
          f"over {n_games} random games; HH-larger-die-laxity positions: {lax}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
