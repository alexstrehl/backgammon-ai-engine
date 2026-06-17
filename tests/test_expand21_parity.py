"""Parity tests for the expand_21 C fast path in the 1-ply choose /
value routines.

Each test runs the same agent logic twice — once with the C engine
(use_fast_engine=True, the expand_21 path) and once with the pure-
Python fallback (use_fast_engine=False) — on identical states, dice,
and weights, asserting values agree to float32-forward tolerance and
chosen moves are identical.

Covers: terminal replies, forced-pass (dance) buckets, centered vs
owned cube (Jacoby gammons-don't-count vs count), scenario B (opponent
cube simulation), and both output modes.
"""

import random
from typing import List

import pytest
import torch

from backgammon_engine import (
    BoardState, get_legal_plays, opening_roll, switch_turn,
)
from modes import CubefulMoneyMode
from model import TDNetwork
from td_agent import TDAgent

VAL_TOL = 1e-4


def _small_net(encoder_name: str, output_mode: str, seed: int) -> TDNetwork:
    torch.manual_seed(seed)
    net = TDNetwork(
        hidden_sizes=[64, 32],
        encoder_name=encoder_name,
        output_mode=output_mode,
    )
    # Shrink weights so the untrained net produces small, non-saturated
    # outputs — better signal for detecting drift between paths.
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(0.05)
    return net


def _agent_pair(encoder_name: str, output_mode: str, seed: int = 777, **kw):
    """Fast (C expand_21) and slow (pure Python) agents sharing weights."""
    net = _small_net(encoder_name, output_mode, seed)
    fast = TDAgent(net, use_fast_engine=True, **kw)
    slow = TDAgent(net, use_fast_engine=False, **kw)
    assert fast._c_base_available, \
        "C engine must be available for expand_21 parity tests"
    assert not slow._c_base_available
    return fast, slow


def _random_states(n: int, seed: int) -> List[BoardState]:
    """Random-playout states: opening, midgame, and bearoff variety."""
    rng = random.Random(seed)
    out: List[BoardState] = []
    while len(out) < n:
        state, dice = opening_roll(rng)
        steps = 0
        while not state.is_game_over() and steps < 120:
            steps += 1
            plays = get_legal_plays(state, dice)
            if plays:
                state = rng.choice(plays)[1]
            else:
                state = switch_turn(state)
            out.append(state)
            if len(out) >= n:
                return out
            dice = (rng.randint(1, 6), rng.randint(1, 6))
    return out


def _dance_state() -> BoardState:
    """WHITE to move; BLACK has a checker on the bar against WHITE's
    closed home board. After any WHITE move that keeps the board
    closed, every BLACK reply is a forced pass (dance) — exercising
    the expand_21 forced-pass row for all 21 dice buckets.
    """
    pts = [0] * 24
    for i in range(6):  # WHITE home board, all six points made
        pts[i] = 2
    pts[12] = 2
    pts[23] = 1  # 15 WHITE total
    pts[18] = -5
    pts[16] = -5
    pts[11] = -4  # 14 BLACK on the board + 1 on the bar
    return BoardState(points=pts, bar=[0, 1], off=[0, 0], turn=0)


class TestValueBatchedExpand21Parity:
    """_value_oneply_checker_batched: fast vs Python fallback."""

    @pytest.mark.parametrize("output_mode", ["probability", "equity"])
    def test_random_states(self, output_mode):
        fast, slow = _agent_pair("perspective196", output_mode)
        for s in _random_states(30, seed=11):
            if s.is_game_over():
                continue
            v_fast = fast._value_oneply_checker_batched(s)
            v_slow = slow._value_oneply_checker_batched(s)
            assert abs(v_fast - v_slow) < VAL_TOL, \
                f"{output_mode}: {v_fast} != {v_slow} for {s._key()}"

    def test_dance_state_all_forced_pass(self):
        """On-roll player dances on every roll → all 21 buckets are
        forced-pass rows in expand_21.
        """
        fast, slow = _agent_pair("perspective196", "equity")
        s = _dance_state()
        s = switch_turn(s)  # BLACK on roll, on the bar, board closed
        v_fast = fast._value_oneply_checker_batched(s)
        v_slow = slow._value_oneply_checker_batched(s)
        assert abs(v_fast - v_slow) < VAL_TOL


class TestChooseCubelessExpand21Parity:
    """_choose_checker_oneply_cubeless: fast vs Python fallback."""

    @pytest.mark.parametrize("output_mode", ["probability", "equity"])
    def test_random_states(self, output_mode):
        fast, slow = _agent_pair(
            "perspective196", output_mode, oneply=True,
        )
        rng = random.Random(7)
        checked = 0
        for s in _random_states(25, seed=13):
            dice = (rng.randint(1, 6), rng.randint(1, 6))
            plays = get_legal_plays(s, dice)
            if not plays:
                continue
            r_fast = fast._choose_checker_oneply_cubeless(
                s, dice, actions=plays, with_target=True,
            )
            r_slow = slow._choose_checker_oneply_cubeless(
                s, dice, actions=plays, with_target=True,
            )
            assert abs(r_fast[2] - r_slow[2]) < VAL_TOL, \
                f"{output_mode}: value {r_fast[2]} != {r_slow[2]}"
            assert r_fast[1]._key() == r_slow[1]._key(), \
                f"{output_mode}: different move chosen"
            checked += 1
        assert checked >= 15

    def test_forced_pass_replies(self):
        """Every candidate move leaves the opponent dancing — all
        reply buckets are forced passes.
        """
        fast, slow = _agent_pair("perspective196", "equity", oneply=True)
        s = _dance_state()  # WHITE to move, BLACK dances after any move
        dice = (3, 1)
        plays = get_legal_plays(s, dice)
        assert plays
        r_fast = fast._choose_checker_oneply_cubeless(
            s, dice, actions=plays, with_target=True,
        )
        r_slow = slow._choose_checker_oneply_cubeless(
            s, dice, actions=plays, with_target=True,
        )
        assert abs(r_fast[2] - r_slow[2]) < VAL_TOL
        assert r_fast[1]._key() == r_slow[1]._key()


class TestChooseCubefulExpand21Parity:
    """_choose_checker_oneply_cubeful: fast vs Python fallback,
    covering centered cube (scenario B active, Jacoby gammons off),
    mover-owned cube (no scenario B), and opponent-owned cube
    (scenario B active, gammons count).
    """

    @pytest.fixture(scope="class")
    def pair(self):
        return _agent_pair("cubeful_perspective196", "equity")

    def test_random_states_cube_mix(self, pair):
        fast, slow = pair
        mode = CubefulMoneyMode(jacoby=True)
        ms_center = mode.initial_match_state()
        rng = random.Random(17)
        checked = 0
        for i, s in enumerate(_random_states(24, seed=19)):
            dice = (rng.randint(1, 6), rng.randint(1, 6))
            plays = get_legal_plays(s, dice)
            if not plays:
                continue
            if i % 3 == 0:
                ms = ms_center  # centered: opp can offer, gammons off
            elif i % 3 == 1:
                ms = ms_center.after_take(1 - s.turn)  # mover owns
            else:
                ms = ms_center.after_take(s.turn)  # opp owns
            c_fast = fast._choose_checker_oneply_cubeful(s, plays, ms)
            c_slow = slow._choose_checker_oneply_cubeful(s, plays, ms)
            assert abs(c_fast[1] - c_slow[1]) < VAL_TOL, \
                f"i={i}: value {c_fast[1]} != {c_slow[1]}"
            assert c_fast[0]._key() == c_slow[0]._key(), \
                f"i={i}: different move chosen"
            checked += 1
        assert checked >= 15

    def test_forced_pass_replies(self, pair):
        fast, slow = pair
        mode = CubefulMoneyMode(jacoby=True)
        ms = mode.initial_match_state()
        s = _dance_state()
        plays = get_legal_plays(s, (3, 1))
        assert plays
        c_fast = fast._choose_checker_oneply_cubeful(s, plays, ms)
        c_slow = slow._choose_checker_oneply_cubeful(s, plays, ms)
        assert abs(c_fast[1] - c_slow[1]) < VAL_TOL
        assert c_fast[0]._key() == c_slow[0]._key()
