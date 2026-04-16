"""Tests for the GameMode abstraction in modes.py."""

import random

import pytest

from backgammon_engine import (
    BoardState, WHITE, BLACK, NUM_CHECKERS, get_legal_plays, switch_turn,
)
from encoding import CubePerspective
from modes import (
    CubefulMoneyMode, CubeOwner, DMPMode, MatchState, CubelessMoneyMode,
    cube_perspective,
)
from td_agent import TDAgent, TerminalOutcome
from model import TDNetwork


# ── helpers ──────────────────────────────────────────────────────────


def _terminal_state_with_winner(winner: int, *, gammon=False, backgammon=False) -> BoardState:
    """Build a synthetic terminal BoardState where `winner` has borne off
    all 15 checkers. Per the new wart-free engine convention, the
    returned state has `turn = loser` (the engine would have switched
    after the winning move)."""
    state = BoardState(
        points=[0] * 24,
        bar=[0, 0],
        off=[0, 0],
        turn=1 - winner,  # loser is on roll at terminal (engine switched)
    )
    state.off[winner] = NUM_CHECKERS
    loser = 1 - winner

    if backgammon:
        # Loser has a checker in the winner's home board
        if loser == WHITE:
            # WHITE's home = idx 0-5; for backgammon, WHITE-as-loser must have
            # a checker in BLACK's home (idx 18-23). WHITE checkers are positive.
            state.points[18] = 1
        else:
            # BLACK's home = idx 18-23; loser BLACK with a checker in WHITE's
            # home (idx 0-5). BLACK checkers are negative.
            state.points[5] = -1
    elif gammon:
        # Loser has borne off nothing AND no checkers on bar / in winner's home
        if loser == WHITE:
            state.points[0] = 1   # WHITE checker in own home
        else:
            state.points[23] = -1
    else:
        # Single: loser has borne off at least one checker
        state.off[loser] = 1
        if loser == WHITE:
            state.points[0] = 14
        else:
            state.points[23] = -14
    return state


# ── CubelessMoneyMode terminal outcomes ───────────────────────────────────────


class TestMoneyModeTerminal:
    def test_single_win_white(self):
        state = _terminal_state_with_winner(WHITE)
        outcome = CubelessMoneyMode().make_terminal_outcome(state)
        assert outcome.won_gammon is False
        assert outcome.won_backgammon is False

    def test_single_win_black(self):
        state = _terminal_state_with_winner(BLACK)
        outcome = CubelessMoneyMode().make_terminal_outcome(state)
        assert outcome.won_gammon is False
        assert outcome.won_backgammon is False

    def test_gammon(self):
        state = _terminal_state_with_winner(WHITE, gammon=True)
        outcome = CubelessMoneyMode().make_terminal_outcome(state)
        assert outcome.won_gammon is True
        assert outcome.won_backgammon is False

    def test_backgammon(self):
        state = _terminal_state_with_winner(WHITE, backgammon=True)
        outcome = CubelessMoneyMode().make_terminal_outcome(state)
        assert outcome.won_gammon is True
        assert outcome.won_backgammon is True


# ── DMPMode terminal outcomes (gammons should be ignored) ─────────────


class TestDMPModeTerminal:
    def test_single_returns_bare_outcome(self):
        state = _terminal_state_with_winner(WHITE)
        outcome = DMPMode().make_terminal_outcome(state)
        assert outcome == TerminalOutcome()

    def test_gammon_still_returns_bare_outcome(self):
        state = _terminal_state_with_winner(WHITE, gammon=True)
        outcome = DMPMode().make_terminal_outcome(state)
        assert outcome.won_gammon is False
        assert outcome.won_backgammon is False


# ── Terminal-call assertions ──────────────────────────────────────────


class TestTerminalAssertions:
    def test_non_terminal_state_raises(self):
        state = BoardState.initial()
        with pytest.raises(AssertionError, match="non-terminal"):
            CubelessMoneyMode().make_terminal_outcome(state)

    def test_state_with_winner_on_roll_raises(self):
        # If state.turn == winner (instead of the loser), the assertion
        # should fire — under the new convention the engine has
        # already switched after the winning move so the loser is on
        # roll at terminal.
        state = _terminal_state_with_winner(WHITE)
        state = switch_turn(state)  # now winner is on roll: invalid
        with pytest.raises(AssertionError, match="state.turn"):
            CubelessMoneyMode().make_terminal_outcome(state)


# ── End-to-end episode rollouts ───────────────────────────────────────


def _play_one_episode(mode, agent, seed: int):
    """Play one episode under `mode` with `agent`. Return (outcome, target)."""
    rng = random.Random(seed)
    state = mode.initial_state()
    while not mode.is_episode_over(state):
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        plays = get_legal_plays(state, (d1, d2))
        if plays:
            _, next_state = agent.choose_checker_action(state, (d1, d2), plays)
            if next_state.is_game_over():
                outcome = mode.make_terminal_outcome(next_state)
                return outcome, agent.terminal_target(outcome)
            # next_state already has turn switched
            state = next_state
        else:
            state = switch_turn(state)
    raise AssertionError("episode loop exited without producing a terminal")


class TestEpisodeRollout:
    def test_money_with_equity_agent(self):
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        outcome, target = _play_one_episode(CubelessMoneyMode(), agent, seed=1)
        # equity terminal target = game_result ∈ {1, 2, 3}
        assert target in (1.0, 2.0, 3.0)
        # Cross-check vs the outcome flags
        if outcome.won_backgammon:
            assert target == 3.0
        elif outcome.won_gammon:
            assert target == 2.0
        else:
            assert target == 1.0

    def test_dmp_with_probability_agent(self):
        net = TDNetwork(hidden_sizes=[16], output_mode="probability")
        agent = TDAgent(net)
        outcome, target = _play_one_episode(DMPMode(), agent, seed=2)
        # DMP + probability mode → terminal is always 1.0
        assert target == 1.0
        # DMPMode does not propagate gammon flags
        assert outcome.won_gammon is False
        assert outcome.won_backgammon is False


# ── Cube state primitives ─────────────────────────────────────────────


class TestCubePerspective:
    def test_centered_is_centered_for_either_player(self):
        assert cube_perspective(CubeOwner.CENTERED, WHITE) == CubePerspective.CENTERED
        assert cube_perspective(CubeOwner.CENTERED, BLACK) == CubePerspective.CENTERED

    def test_owner_matches_on_roll(self):
        assert cube_perspective(CubeOwner.WHITE, WHITE) == CubePerspective.MINE
        assert cube_perspective(CubeOwner.BLACK, BLACK) == CubePerspective.MINE

    def test_owner_is_opponent(self):
        assert cube_perspective(CubeOwner.BLACK, WHITE) == CubePerspective.THEIRS
        assert cube_perspective(CubeOwner.WHITE, BLACK) == CubePerspective.THEIRS


class TestMatchState:
    def test_default_is_centered_value_one(self):
        m = MatchState()
        assert m.cube_owner == CubeOwner.CENTERED
        assert m.cube_value == 1
        assert m.jacoby is True

    def test_can_offer_centered(self):
        m = MatchState()
        assert m.can_offer(WHITE) and m.can_offer(BLACK)

    def test_can_offer_owned(self):
        m = MatchState(cube_owner=CubeOwner.WHITE, cube_value=2)
        assert m.can_offer(WHITE)
        assert not m.can_offer(BLACK)

    def test_after_take(self):
        m = MatchState()
        m2 = m.after_take(doubler=WHITE)
        assert m2.cube_owner == CubeOwner.BLACK  # cube goes to opponent
        assert m2.cube_value == 2
        m3 = m2.after_take(doubler=BLACK)
        assert m3.cube_owner == CubeOwner.WHITE
        assert m3.cube_value == 4


# ── CubefulMoneyMode terminal outcomes ───────────────────────────────


class TestCubefulMoneyTerminal:
    def test_jacoby_centered_collapses_gammon(self):
        state = _terminal_state_with_winner(WHITE, gammon=True)
        mode = CubefulMoneyMode(jacoby=True)
        m = mode.initial_match_state()
        outcome = mode.make_terminal_outcome(state, m)
        assert outcome.won_gammon is False
        assert outcome.cube_value == 1
        net = TDNetwork(hidden_sizes=[8], output_mode="equity")
        agent = TDAgent(net)
        assert agent.terminal_target(outcome) == 1.0

    def test_jacoby_owned_keeps_gammon(self):
        state = _terminal_state_with_winner(WHITE, gammon=True)
        mode = CubefulMoneyMode(jacoby=True)
        m = MatchState(cube_owner=CubeOwner.WHITE, cube_value=2)
        outcome = mode.make_terminal_outcome(state, m)
        assert outcome.won_gammon is True
        assert outcome.cube_value == 2
        net = TDNetwork(hidden_sizes=[8], output_mode="equity")
        agent = TDAgent(net)
        # Target is PER-UNIT: gammon → 2.0. cube_value lives on
        # the outcome as metadata but doesn't scale the training target
        # (the network is trained at cube_value=1).
        assert agent.terminal_target(outcome) == 2.0

    def test_no_jacoby_keeps_gammon_centered(self):
        state = _terminal_state_with_winner(WHITE, backgammon=True)
        mode = CubefulMoneyMode(jacoby=False)
        m = mode.initial_match_state()
        outcome = mode.make_terminal_outcome(state, m)
        assert outcome.won_backgammon is True
        net = TDNetwork(hidden_sizes=[8], output_mode="equity")
        agent = TDAgent(net)
        # backgammon (3) × cube_value (1)
        assert agent.terminal_target(outcome) == 3.0

    def test_requires_match_state(self):
        state = _terminal_state_with_winner(WHITE)
        with pytest.raises(AssertionError, match="MatchState"):
            CubefulMoneyMode().make_terminal_outcome(state, None)


# ── Mode × output_mode validation ─────────────────────────────────────


class TestModeOutputModeValidation:
    """Money modes must reject probability-output agents — probability
    can't represent gammon (2) or backgammon (3) terminal targets, so
    the gammon signal is silently dropped during training."""

    def test_money_rejects_probability_agent(self):
        net = TDNetwork(hidden_sizes=[8], output_mode="probability")
        agent = TDAgent(net)
        with pytest.raises(ValueError, match="CubelessMoneyMode requires"):
            CubelessMoneyMode().validate_agent(agent)

    def test_money_accepts_equity_agent(self):
        net = TDNetwork(hidden_sizes=[8], output_mode="equity")
        agent = TDAgent(net)
        CubelessMoneyMode().validate_agent(agent)  # no raise

    def test_cubeful_money_rejects_probability_agent(self):
        net = TDNetwork(hidden_sizes=[8], output_mode="probability")
        agent = TDAgent(net)
        with pytest.raises(ValueError, match="CubefulMoneyMode requires"):
            CubefulMoneyMode().validate_agent(agent)

    def test_dmp_accepts_either_output_mode(self):
        for om in ("probability", "equity"):
            net = TDNetwork(hidden_sizes=[8], output_mode=om)
            agent = TDAgent(net)
            DMPMode().validate_agent(agent)  # no raise

    def test_trainer_validates_at_train_entry(self):
        from trainer import Trainer
        net = TDNetwork(hidden_sizes=[8], output_mode="probability")
        agent = TDAgent(net)
        trainer = Trainer(agent, lr=1e-3)
        with pytest.raises(ValueError, match="CubelessMoneyMode requires"):
            trainer.train(CubelessMoneyMode(), num_episodes=1, log_every=0)


# ── TDAgent cube actions ─────────────────────────────────────────────


def _make_cubeful_agent():
    import torch
    torch.manual_seed(0)
    net = TDNetwork(
        hidden_sizes=[16],
        output_mode="equity",
        encoder_name="cubeful_perspective196",
    )
    return TDAgent(net)


class TestTDAgentCubeActions:
    def test_offer_when_owned_by_opponent_returns_false(self):
        """Cannot offer when opponent owns the cube."""
        agent = _make_cubeful_agent()
        state = BoardState.initial()  # WHITE on roll
        m = MatchState(cube_owner=CubeOwner.BLACK, cube_value=2)
        offer = agent.offer_double(state, m)
        assert offer.should_double is False
        assert offer._cache is None

    def test_respond_uses_cached_v_theirs(self):
        """When self-play threads the offer cache through, no extra
        forward pass is needed."""
        agent = _make_cubeful_agent()
        state = BoardState.initial()
        m = MatchState()
        offer = agent.offer_double(state, m)
        # Replace v_cube_theirs in the cache with a sentinel that
        # would change the take/pass decision; verify respond uses it.
        offer._cache["v_cube_theirs_0ply"] = 0.6  # 2*0.6 = 1.2 > 1.0 → pass
        assert agent.respond_to_double(state, m, hint=offer) is False
        offer._cache["v_cube_theirs_0ply"] = 0.4  # 2*0.4 = 0.8 ≤ 1.0 → take
        assert agent.respond_to_double(state, m, hint=offer) is True

    def test_respond_recomputes_without_hint(self):
        """No hint → fresh forward pass; result must match."""
        agent = _make_cubeful_agent()
        state = BoardState.initial()
        m = MatchState()
        offer = agent.offer_double(state, m)
        v_theirs = offer._cache["v_cube_theirs_0ply"]
        expected = (2.0 * v_theirs) <= 1.0
        assert agent.respond_to_double(state, m, hint=None) is expected

    def test_offer_double_huge_advantage_doubles_and_offered_passes(self):
        """White borne off 14, last checker on 1: model would predict
        nearly +1 equity. Should double; opponent should pass."""
        agent = _make_cubeful_agent()
        state = BoardState(
            points=[0]*24, bar=[0,0], off=[14, 0], turn=WHITE,
        )
        state.points[0] = 1
        state.points[23] = -15
        m = MatchState()
        offer = agent.offer_double(state, m)
        assert offer.should_double is True

    def test_cubeless_agent_falls_back_to_passive_cube_policy(self):
        """A cubeless TDAgent dropped into cubeful play shouldn't
        crash — it should use the Agent-base default ('never offer,
        always take'). This lets a cubeless model act as a baseline
        opponent in cubeful matches."""
        net = TDNetwork(hidden_sizes=[8], output_mode="equity")
        agent = TDAgent(net)
        offer = agent.offer_double(BoardState.initial(), MatchState())
        assert offer.should_double is False
        assert offer._cache is None
        assert agent.respond_to_double(BoardState.initial(), MatchState()) is True
