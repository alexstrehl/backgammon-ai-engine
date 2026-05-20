"""Tests for trainer.py: episode collection, Trainer class."""

import math
import random

import numpy as np
import pytest
import torch

from backgammon_engine import BoardState, get_legal_plays, switch_turn
from encoding import CubePerspective
from model import TDNetwork
from modes import DMPMode, CubelessMoneyMode, CubeOwner, MatchState, cube_perspective
from td_agent import TDAgent, _DICE_OUTCOMES
from trainer import Trainer, collect_episode


# ── collect_episode ───────────────────────────────────────────────────


class TestCollectEpisode:
    def test_money_equity_returns_arrays(self):
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        rng = random.Random(0)
        encs, targets = collect_episode(agent, CubelessMoneyMode(), rng)
        assert isinstance(encs, np.ndarray)
        assert isinstance(targets, np.ndarray)
        assert encs.dtype == np.float32
        assert targets.dtype == np.float32
        assert encs.ndim == 2 and encs.shape[1] == 196
        assert encs.shape[0] == targets.shape[0]
        assert encs.shape[0] > 5

    def test_dmp_probability_terminal_in_range(self):
        net = TDNetwork(hidden_sizes=[16], output_mode="probability")
        agent = TDAgent(net)
        rng = random.Random(1)
        encs, targets = collect_episode(agent, DMPMode(), rng)
        assert (targets >= 0.0).all() and (targets <= 1.0).all()
        assert targets[-1] == 1.0

    def test_money_equity_terminal_target(self):
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        rng = random.Random(2)
        encs, targets = collect_episode(agent, CubelessMoneyMode(), rng)
        assert targets[-1] in (1.0, 2.0, 3.0)


# ── Trainer primitives ────────────────────────────────────────────────


class TestTrainerPrimitives:
    def test_train_step_decreases_loss_on_repeat(self):
        """A single batch trained on repeatedly should reduce its own loss."""
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        trainer = Trainer(agent, lr=1e-2)
        rng = random.Random(0)
        encs, targets = collect_episode(agent, CubelessMoneyMode(), rng)
        first = trainer.train_step(encs, targets)
        for _ in range(20):
            trainer.train_step(encs, targets)
        last = trainer.train_step(encs, targets)
        assert last < first, f"loss did not decrease: first={first} last={last}"

    def test_eval_loss_does_not_step(self):
        """eval_loss must not modify network weights."""
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        trainer = Trainer(agent, lr=1e-2)
        rng = random.Random(0)
        encs, targets = collect_episode(agent, CubelessMoneyMode(), rng)
        before = [p.clone() for p in net.parameters()]
        for _ in range(5):
            trainer.eval_loss(encs, targets)
        after = list(net.parameters())
        for b, a in zip(before, after):
            assert torch.equal(b, a), "eval_loss must not modify weights"

    def test_grad_clip_runs(self):
        """grad_clip=1.0 path should run without crashing."""
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        trainer = Trainer(agent, lr=1e-2, grad_clip=1.0)
        rng = random.Random(0)
        encs, targets = collect_episode(agent, CubelessMoneyMode(), rng)
        loss = trainer.train_step(encs, targets)
        assert math.isfinite(loss)


# ── value_oneply_checker terminal handling (equity mode) ─────────────


class TestValueOneplyCheckerEquityTerminals:
    """Equity-mode 1-ply lookahead must distinguish single / gammon /
    backgammon wins among the terminal successors it enumerates.
    Probability mode collapses them all to 1.0 so it doesn't need
    this differentiation."""

    @staticmethod
    def _near_bear_off(black_off: int, black_in_white_home: bool) -> BoardState:
        """Build a synthetic position where WHITE has 14 of 15 borne
        off and one checker on point 1 — every dice roll bears that
        last checker off and ends the game with WHITE winning. The
        kind of win depends on BLACK's borne-off count and whether
        BLACK still has any checkers in WHITE's home board:
          - black_off=1: single (1)
          - black_off=0, no checker in white's home: gammon (2)
          - black_off=0, checker in white's home: backgammon (3)
        """
        state = BoardState(
            points=[0] * 24, bar=[0, 0], off=[14, black_off], turn=0,
        )
        state.points[0] = 1  # white's last checker
        if black_in_white_home:
            state.points[5] = -1
            state.points[23] = -(15 - black_off - 1)
        else:
            state.points[23] = -(15 - black_off)
        return state

    def _eval_equity_oneply(self, state):
        # torch_seed pinned so the network init is deterministic; the
        # test only depends on terminal-detection logic, not the
        # network's actual numbers, but we want reproducibility.
        torch.manual_seed(0)
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        return agent.value_oneply_checker(state)

    def test_single_win_returns_one(self):
        state = self._near_bear_off(black_off=1, black_in_white_home=False)
        assert self._eval_equity_oneply(state) == pytest.approx(1.0)

    def test_gammon_win_returns_two(self):
        state = self._near_bear_off(black_off=0, black_in_white_home=False)
        assert self._eval_equity_oneply(state) == pytest.approx(2.0)

    def test_backgammon_win_returns_three(self):
        state = self._near_bear_off(black_off=0, black_in_white_home=True)
        assert self._eval_equity_oneply(state) == pytest.approx(3.0)

    def test_gammon_strictly_greater_than_single(self):
        single = self._eval_equity_oneply(
            self._near_bear_off(black_off=1, black_in_white_home=False)
        )
        gammon = self._eval_equity_oneply(
            self._near_bear_off(black_off=0, black_in_white_home=False)
        )
        backgammon = self._eval_equity_oneply(
            self._near_bear_off(black_off=0, black_in_white_home=True)
        )
        assert single < gammon < backgammon


# ── End-to-end training ───────────────────────────────────────────────


def _all_finite(xs):
    return all(math.isfinite(x) for x in xs)


@pytest.mark.slow
class TestEndToEndTraining:
    """Smoke: training runs end-to-end without crashing and produces
    finite losses. Deeper convergence checks are manual."""

    def test_train_dmp_probability(self):
        net = TDNetwork(hidden_sizes=[40], output_mode="probability")
        agent = TDAgent(net)
        trainer = Trainer(agent, lr=1e-3)
        losses = trainer.train(
            DMPMode(), num_episodes=200, batch_size=128, seed=20, log_every=0,
        )
        assert len(losses) > 0
        assert _all_finite(losses)

    def test_train_money_equity(self):
        net = TDNetwork(hidden_sizes=[40], output_mode="equity")
        agent = TDAgent(net)
        trainer = Trainer(agent, lr=1e-3)
        losses = trainer.train(
            CubelessMoneyMode(), num_episodes=200, batch_size=128, seed=21, log_every=0,
        )
        assert len(losses) > 0
        assert _all_finite(losses)

    def test_train_online_dmp_probability_with_sgd(self):
        net = TDNetwork(hidden_sizes=[40], output_mode="probability")
        agent = TDAgent(net)
        trainer = Trainer(agent, lr=0.1, optimizer_cls=torch.optim.SGD)
        losses = trainer.train_online(
            DMPMode(), num_episodes=50, seed=30, log_every=0,
        )
        assert len(losses) > 0
        assert _all_finite(losses)

    def test_train_online_money_equity_with_sgd(self):
        net = TDNetwork(hidden_sizes=[40], output_mode="equity")
        agent = TDAgent(net)
        # Equity targets (±1/2/3) need a smaller LR than probability
        # mode (targets in [0, 1]) to avoid divergence in online SGD.
        trainer = Trainer(agent, lr=0.01, optimizer_cls=torch.optim.SGD)
        losses = trainer.train_online(
            CubelessMoneyMode(), num_episodes=50, seed=31, log_every=0,
        )
        assert len(losses) > 0
        assert _all_finite(losses)


# ── _choose_checker_oneply_cubeful correctness + performance ─────────


def _make_cubeful_agent(seed: int = 0) -> TDAgent:
    """Small deterministic cubeful-equity agent for 1-ply tests."""
    torch.manual_seed(seed)
    net = TDNetwork(
        hidden_sizes=[16],
        output_mode="equity",
        encoder_name="cubeful_perspective196",
    )
    return TDAgent(net, oneply=True)


def _near_bear_off_white(black_off: int, black_in_white_home: bool):
    """WHITE has 14 of 15 off with its last checker on point 1. Any
    dice rolls this last checker off → every candidate move is terminal.
    black_off / black_in_white_home controls single vs gammon vs bg."""
    state = BoardState(
        points=[0] * 24, bar=[0, 0], off=[14, black_off], turn=0,
    )
    state.points[0] = 1
    if black_in_white_home:
        state.points[5] = -1
        state.points[23] = -(15 - black_off - 1)
    else:
        state.points[23] = -(15 - black_off)
    return state


def _closed_home_with_black_barred():
    """WHITE on roll with a fully closed home board; BLACK has 2 on
    the bar. After any WHITE move that keeps home closed, every BLACK
    dice roll is a forced pass (can't enter)."""
    state = BoardState(
        points=[0] * 24, bar=[0, 2], off=[0, 0], turn=0,
    )
    for p in range(6):
        state.points[p] = 2       # WHITE: 12 in home
    state.points[12] = 3          # WHITE: 3 more → 15 total
    state.points[23] = -5         # BLACK on board
    state.points[18] = -4
    state.points[17] = -4         # 2 + 5 + 4 + 4 = 15
    return state


def _naive_oneply_cubeful(agent, state, plays, match_state):
    """Non-batched reference for _choose_checker_oneply_cubeful. Makes
    one evaluate_cubeful call per non-terminal encoding instead of
    folding everything into a single batched forward pass."""
    mover = state.turn
    opponent = 1 - mover
    opp_can_offer = match_state.can_offer(opponent)
    opp_persp_at_next = cube_perspective(match_state.cube_owner, opponent)
    our_persp_A = cube_perspective(match_state.cube_owner, mover)
    after_opp_is_cube_action_A = match_state.can_offer(mover)
    gammons_count_A = (
        not match_state.jacoby
        or match_state.cube_owner != CubeOwner.CENTERED
    )

    best_val = float("-inf")
    best_idx = 0

    for m_idx, (_, next_state) in enumerate(plays):
        if next_state.is_game_over():
            val = (float(next_state.game_result())
                   if gammons_count_A else 1.0)
        else:
            U_A = 0.0
            U_B = 0.0
            for d1, d2 in _DICE_OUTCOMES:
                prob = (1.0 / 36.0) if d1 == d2 else (2.0 / 36.0)
                opp_plays = get_legal_plays(next_state, (d1, d2))
                if not opp_plays:
                    back = switch_turn(next_state)
                    eqs_A = [agent.evaluate_cubeful(
                        back, our_persp_A, after_opp_is_cube_action_A,
                    )]
                    eqs_B = None
                    if opp_can_offer:
                        eqs_B = [agent.evaluate_cubeful(
                            back, CubePerspective.MINE, True,
                        )]
                else:
                    eqs_A = []
                    eqs_B = [] if opp_can_offer else None
                    for _, after_opp in opp_plays:
                        if after_opp.is_game_over():
                            gr = float(after_opp.game_result())
                            eqs_A.append(-gr if gammons_count_A else -1.0)
                            if opp_can_offer:
                                eqs_B.append(-gr)
                        else:
                            eqs_A.append(agent.evaluate_cubeful(
                                after_opp, our_persp_A,
                                after_opp_is_cube_action_A,
                            ))
                            if opp_can_offer:
                                eqs_B.append(agent.evaluate_cubeful(
                                    after_opp, CubePerspective.MINE, True,
                                ))
                U_A += prob * min(eqs_A)
                if opp_can_offer:
                    U_B += prob * min(eqs_B)

            if opp_can_offer:
                V_no_double_opp = -U_A
                V_double_take_opp = -2.0 * U_B
                opp_doubles = agent._money_cube_decision(
                    V_no_double_opp, V_double_take_opp,
                    opp_persp_at_next, match_state.jacoby,
                )
                if opp_doubles:
                    takes = V_double_take_opp <= 1.0
                    val = 2.0 * U_B if takes else -1.0
                else:
                    val = U_A
            else:
                val = U_A

        if val > best_val:
            best_val = val
            best_idx = m_idx

    return plays[best_idx][1], best_val


# Representative cube configurations for money play.
_MATCH_STATES = [
    ("centered_jacoby",   MatchState(cube_owner=CubeOwner.CENTERED, jacoby=True)),
    ("centered_nojacoby", MatchState(cube_owner=CubeOwner.CENTERED, jacoby=False)),
    ("we_own",            MatchState(cube_owner=CubeOwner.WHITE, jacoby=True)),
    ("opp_owns",          MatchState(cube_owner=CubeOwner.BLACK, jacoby=True)),
]


class TestChooseCheckerOneplyCubeful:
    """Correctness + performance contract for the batched 1-ply cubeful
    move selection path in TDAgent._choose_checker_oneply_cubeful."""

    @pytest.mark.parametrize("pos_name, dice", [
        ("starting", (3, 1)),
        ("closed",   (3, 4)),
    ])
    @pytest.mark.parametrize("ms_name", [n for n, _ in _MATCH_STATES])
    def test_parity_with_naive_reference(self, pos_name, dice, ms_name):
        """Batched path and a dead-simple per-state reference must agree
        on best_val across positions × cube configs. Use pytest.approx
        for numeric tolerance on float accumulation differences."""
        if pos_name == "starting":
            state = BoardState.initial()
        else:
            state = _closed_home_with_black_barred()
        match_state = dict(_MATCH_STATES)[ms_name]
        agent = _make_cubeful_agent(seed=0)

        plays = get_legal_plays(state, dice)
        assert plays, "expected legal plays for test setup"

        batched_chosen, batched_val = agent._choose_checker_oneply_cubeful(
            state, plays, match_state,
        )
        ref_chosen, ref_val = _naive_oneply_cubeful(
            agent, state, plays, match_state,
        )
        assert batched_val == pytest.approx(ref_val, abs=1e-5, rel=1e-5)

    def test_forced_pass_branch_is_actually_exercised(self):
        """Sanity: the closed-home position must produce at least one
        forced-pass opp dice outcome, otherwise the forced-pass parity
        case above is vacuous."""
        state = _closed_home_with_black_barred()
        plays = get_legal_plays(state, (3, 4))
        assert plays
        _, next_state = plays[0]
        forced = 0
        for d1, d2 in _DICE_OUTCOMES:
            if not get_legal_plays(next_state, (d1, d2)):
                forced += 1
        assert forced >= 1, \
            "expected at least one forced-pass opp dice in closed-home test pos"

    def test_single_forward_pass_per_call(self):
        """Performance contract: the whole candidate × 21 dice ×
        {scenario A, scenario B} tensor is folded into ONE network call.
        Count invocations of agent.network.forward."""
        agent = _make_cubeful_agent(seed=1)
        match_state = MatchState(cube_owner=CubeOwner.CENTERED, jacoby=True)
        state = BoardState.initial()
        plays = get_legal_plays(state, (3, 1))
        assert plays

        calls = []
        orig_forward = agent.network.forward

        def counting_forward(x):
            calls.append(x.shape[0] if x.dim() == 2 else 1)
            return orig_forward(x)

        agent.network.forward = counting_forward
        try:
            agent._choose_checker_oneply_cubeful(state, plays, match_state)
        finally:
            agent.network.forward = orig_forward

        assert len(calls) == 1, \
            f"expected exactly 1 network forward pass, got {len(calls)}: {calls}"

    def test_terminal_candidate_wins_and_gets_exact_equity(self):
        """If a candidate move ends the game with WHITE winning, that
        candidate's val must equal game_result (or 1.0 under Jacoby on
        centered cube). It must also be chosen as best."""
        state = _near_bear_off_white(black_off=0, black_in_white_home=False)
        agent = _make_cubeful_agent(seed=2)
        # Jacoby off → gammons count → val == 2.0 for a gammon finish.
        match_state = MatchState(cube_owner=CubeOwner.CENTERED, jacoby=False)
        # Any dice bears off WHITE's last checker (point 1 needs 1+).
        plays = get_legal_plays(state, (3, 5))
        assert plays
        # Every legal play must be terminal here (white's only checker
        # is on the ace and any roll bears it off).
        for _, ns in plays:
            assert ns.is_game_over()

        chosen, best_val = agent._choose_checker_oneply_cubeful(
            state, plays, match_state,
        )
        assert chosen.is_game_over()
        assert best_val == pytest.approx(2.0)

    def test_terminal_candidate_single_under_jacoby_on_centered_cube(self):
        """Under Jacoby on a centered cube, gammons don't count and the
        terminal val collapses to 1.0 (per code: gammons_count_A=False).
        """
        state = _near_bear_off_white(black_off=0, black_in_white_home=False)
        agent = _make_cubeful_agent(seed=3)
        match_state = MatchState(cube_owner=CubeOwner.CENTERED, jacoby=True)
        plays = get_legal_plays(state, (3, 5))
        assert plays

        _, best_val = agent._choose_checker_oneply_cubeful(
            state, plays, match_state,
        )
        assert best_val == pytest.approx(1.0)


# ── 0-ply selection: terminal-mask regression ────────────────────────


def _mixed_terminal_position():
    """WHITE has 13 off, 1 on point 5, 1 on point 0. BLACK has 1 off so
    WHITE's terminal finish is a single win (game_result=1), not a
    gammon. With dice (6, 1), legal plays include a terminal bear-off
    and a non-terminal reorganization — exactly the setup where 0-ply
    selection must handle terminals correctly."""
    state = BoardState(
        points=[0] * 24, bar=[0, 0], off=[13, 1], turn=0,
    )
    state.points[0] = 1
    state.points[5] = 1
    state.points[23] = -14
    return state


def _stub_network_terminal_looks_good(orig_forward):
    """Wrap a forward fn so terminal encodings (OPP_OFF ≥ threshold)
    return a HIGH value (0.9) and non-terminals return LOW (0.1).
    Pre-fix argmin would pick the non-terminal — the very bug we're
    guarding against."""
    from encoding import OPP_OFF_INDEX, TERMINAL_OFF_THRESHOLD

    def stub(x):
        is_term = (x[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD).float()
        return 0.9 * is_term + 0.1 * (1.0 - is_term)

    return stub


class TestZeroPlyTerminalMask:
    """Regression: 0-ply action selection must not let the network's
    arbitrary output on terminal encodings cause argmin to walk past
    a winning move."""

    def test_setup_has_both_terminal_and_nonterminal_plays(self):
        state = _mixed_terminal_position()
        plays = get_legal_plays(state, (6, 1))
        terminals = [s for _, s in plays if s.is_game_over()]
        non_terms = [s for _, s in plays if not s.is_game_over()]
        assert terminals and non_terms, (
            f"test setup expected ≥1 terminal and ≥1 non-terminal play; "
            f"got terminals={len(terminals)}, non_terminals={len(non_terms)}"
        )

    def test_cubeful_0ply_picks_terminal_over_fooling_network(self):
        """With a stubbed network that makes terminals look WORSE (higher
        value) than non-terminals, the cubeful 0-ply path must still
        pick the terminal via the mask."""
        state = _mixed_terminal_position()
        torch.manual_seed(0)
        net = TDNetwork(
            hidden_sizes=[16], output_mode="equity",
            encoder_name="cubeful_perspective196",
        )
        agent = TDAgent(net, oneply=False)
        match_state = MatchState(cube_owner=CubeOwner.CENTERED, jacoby=True)

        orig = agent.network.forward
        agent.network.forward = _stub_network_terminal_looks_good(orig)
        try:
            result = agent.choose_checker_action_cubeful(
                state, (6, 1), match_state,
            )
        finally:
            agent.network.forward = orig

        assert result is not None
        chosen, bootstrap = result
        assert chosen.is_game_over(), \
            "cubeful 0-ply picked non-terminal over an available terminal win"
        # Bootstrap for a Jacoby-centered single-win terminal should be
        # +1.0 (gammons_count=False → -1.0 from opp view → flip +1.0).
        assert bootstrap == pytest.approx(1.0)

    def test_cubeless_0ply_picks_terminal_over_fooling_network_fast_path(self):
        """Same regression for the cubeless 0-ply path (actions=None
        branch, uses _get_legal_plays_encoded)."""
        state = _mixed_terminal_position()
        torch.manual_seed(0)
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)

        orig = agent.network.forward
        agent.network.forward = _stub_network_terminal_looks_good(orig)
        try:
            result = agent.choose_checker_action(
                state, (6, 1), with_target=True,
            )
        finally:
            agent.network.forward = orig

        assert result is not None
        _, chosen, target = result
        assert chosen.is_game_over()
        # Equity mode: terminal_opp_value = -game_result. flip → +gr.
        # Single win → target == 1.0.
        assert target == pytest.approx(1.0)

    def test_cubeless_0ply_picks_terminal_over_fooling_network_actions_path(self):
        """Same regression for the `actions=[...]` branch (used by eval
        scripts that pass explicit (Play, state) lists)."""
        state = _mixed_terminal_position()
        torch.manual_seed(0)
        net = TDNetwork(hidden_sizes=[16], output_mode="equity")
        agent = TDAgent(net)
        plays = get_legal_plays(state, (6, 1))

        orig = agent.network.forward
        agent.network.forward = _stub_network_terminal_looks_good(orig)
        try:
            result = agent.choose_checker_action(
                state, (6, 1), actions=plays, with_target=True,
            )
        finally:
            agent.network.forward = orig

        assert result is not None
        _, chosen, target = result
        assert chosen.is_game_over()
        assert target == pytest.approx(1.0)
