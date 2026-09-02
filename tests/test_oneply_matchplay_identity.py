"""Two identities the matchplay 1-ply backup must satisfy.

1. A 1-ply target IS the dice-expectation of the 0-ply target.
   `value_oneply_checker_matchplay` claims to be
   sum_dice P(dice) * V_0ply(best successor). So enumerating the 21
   dice with their true probabilities and averaging the agent's OWN
   0-ply choice must reproduce it exactly -- and sampling dice at
   random must converge to it.

   These deliberately drive the agent's own 0-ply machinery
   (`choose_checker_action_matchplay`) rather than re-deriving terminal
   pricing here, so the test cannot pass by reproducing the same bug
   twice.

2. A cube decision is a decision node with NO chance element: the cube
   action does not consume a dice roll, so a 1-ply backup of a cube
   node degenerates to max over cube actions of V(child), which is
   exactly the 0-ply formula. Depth must therefore be a no-op for cube
   targets. `--oneply` also leaves move SELECTION at 0-ply, so the two
   trajectories coincide step for step and every cube target must be
   bit-identical.
"""

import random

import numpy as np
import pytest
import torch

from backgammon_engine import BoardState, get_legal_plays, switch_turn
from met import MatchEquityTable, DEFAULT_MET
from model import TDNetwork
from modes import MatchState, MatchplayMode
from td_agent import TDAgent
from trainer import _collect_episode_matchplay

# 21 unordered dice outcomes with their true probabilities.
DICE = [(a, b) for a in range(1, 7) for b in range(a, 7)]
assert len(DICE) == 21


def make_agent(seed=0, met=None):
    torch.manual_seed(seed)
    net = TDNetwork(
        hidden_sizes=[64, 32], input_size=207,
        encoder_name="matchplay1h_perspective196",
        output_mode="probability",
    )
    return TDAgent(net, met=met)


def ms(match_length=3, my_away=3, opp_away=3, **kw):
    return MatchState(jacoby=False, match_length=match_length,
                      my_away=my_away, opp_away=opp_away, **kw)


def zero_ply_value(agent, state, match_state, dice):
    """The mover's 0-ply value after playing `dice` greedily -- exactly
    what the collector bootstraps from at a checker sample."""
    got = agent.choose_checker_action_matchplay(state, dice, match_state)
    if got is None:                       # forced pass: dice go to the opponent
        opp_view = switch_turn(state)
        v_opp = agent.evaluate_matchplay(
            opp_view, match_state,
            is_cube_action=match_state.can_offer(opp_view.turn),
        )
        return 1.0 - v_opp
    return got[1]


def positions(n=6, seed=3):
    """A few reachable, non-terminal positions."""
    rng = random.Random(seed)
    s = BoardState.initial()
    out = [s]
    while len(out) < n:
        plays = get_legal_plays(s, (rng.randint(1, 6), rng.randint(1, 6)))
        if not plays:
            s = switch_turn(s)
            continue
        s = plays[rng.randrange(len(plays))][1]
        if s.is_game_over():
            break
        out.append(s)
    return out


# ── identity 1: 1-ply == dice-expectation of 0-ply ──────────────────


@pytest.mark.parametrize("my_away,opp_away", [(3, 3), (2, 3), (1, 2), (2, 1)])
def test_oneply_equals_exact_dice_expectation(my_away, opp_away):
    """Exact form: enumerate all 21 dice with true probabilities."""
    agent = make_agent()
    m = ms(3, my_away, opp_away, crawford_done=(my_away == 1 or opp_away == 1))
    for state in positions():
        expected = 0.0
        for d1, d2 in DICE:
            prob = (1.0 / 36.0) if d1 == d2 else (2.0 / 36.0)
            expected += prob * zero_ply_value(agent, state, m, (d1, d2))
        got = agent.value_oneply_checker_matchplay(state, m)
        assert got == pytest.approx(expected, abs=1e-6), (
            f"{my_away}a{opp_away}a: 1-ply {got:.6f} != dice-expectation "
            f"{expected:.6f}"
        )


def test_oneply_equals_exact_dice_expectation_with_met():
    """Same identity with MET grounding on: terminal pricing goes
    through an exact table instead of the net, on BOTH sides."""
    table = MatchEquityTable.from_xml(DEFAULT_MET)
    agent = make_agent(met=table)
    m = ms(3, 2, 2)
    for state in positions(4):
        expected = sum(
            ((1.0 / 36.0) if d1 == d2 else (2.0 / 36.0))
            * zero_ply_value(agent, state, m, (d1, d2))
            for d1, d2 in DICE
        )
        assert agent.value_oneply_checker_matchplay(state, m) == pytest.approx(
            expected, abs=1e-6)


def test_dice_probabilities_sum_to_one():
    """Guards the weighting itself: doubles 1/36, non-doubles 2/36."""
    total = sum((1.0 / 36.0) if a == b else (2.0 / 36.0) for a, b in DICE)
    assert total == pytest.approx(1.0, abs=1e-12)


def test_monte_carlo_0ply_converges_to_1ply():
    """Sampling form, as an end-to-end check that the weighting matches
    how dice are actually rolled: re-roll from the same position many
    times, play 0-ply, average. Must converge on the 1-ply value.

    Tolerance is set from the observed sample standard error rather than
    a magic constant, so this fails on bias without being flaky.
    """
    agent = make_agent()
    m = ms(3, 2, 3)
    state = BoardState.initial()
    target = agent.value_oneply_checker_matchplay(state, m)

    rng = random.Random(11)
    samples = np.array([
        zero_ply_value(agent, state, m,
                       (rng.randint(1, 6), rng.randint(1, 6)))
        for _ in range(4000)
    ])
    se = samples.std(ddof=1) / np.sqrt(len(samples))
    drift = abs(samples.mean() - target)
    assert drift < 4 * se, (
        f"MC mean {samples.mean():.5f} vs 1-ply {target:.5f}; "
        f"drift {drift:.5f} > 4 SE ({4 * se:.5f})"
    )


# ── identity 2: cube targets are depth-invariant ────────────────────


IS_CUBE_ACTION_COL = 199   # matchplay1h: 196 base + [persp 3-hot, is_cube_action]


def collect_twice(agent, mode, seed, met=None):
    """Same seed, oneply False vs True. Move selection is 0-ply in both,
    so the trajectories must coincide step for step."""
    a = _collect_episode_matchplay(
        agent, mode, random.Random(seed), mode.initial_match_state(),
        oneply=False)
    b = _collect_episode_matchplay(
        agent, mode, random.Random(seed), mode.initial_match_state(),
        oneply=True)
    return a, b


def test_cube_targets_identical_under_oneply():
    """A cube node has no chance element, so its 1-ply backup collapses
    to the 0-ply formula. Every cube target must be bit-identical."""
    agent = make_agent(seed=1)
    mode = MatchplayMode(match_length=3)
    for seed in (0, 1, 2, 3, 4):
        (enc0, tgt0), (enc1, tgt1) = collect_twice(agent, mode, seed)
        assert enc0.shape == enc1.shape, f"seed {seed}: trajectory diverged"
        np.testing.assert_array_equal(enc0, enc1)

        cube = enc0[:, IS_CUBE_ACTION_COL] == 1.0
        assert cube.any(), f"seed {seed}: no cube samples to compare"
        np.testing.assert_array_equal(
            tgt0[cube], tgt1[cube],
            err_msg=f"seed {seed}: cube targets changed with ply depth",
        )


def test_checker_targets_do_change_under_oneply():
    """Guards against the previous test passing vacuously (e.g. if
    --oneply silently did nothing): the CHECKER targets must differ."""
    agent = make_agent(seed=1)
    mode = MatchplayMode(match_length=3)
    differed = False
    for seed in (0, 1, 2, 3, 4):
        (enc0, tgt0), (_enc1, tgt1) = collect_twice(agent, mode, seed)
        checker = enc0[:, IS_CUBE_ACTION_COL] == 0.0
        if checker.any() and not np.array_equal(tgt0[checker], tgt1[checker]):
            differed = True
            break
    assert differed, "--oneply changed no checker target; the flag is inert"


def test_cube_targets_identical_under_oneply_with_met():
    table = MatchEquityTable.from_xml(DEFAULT_MET)
    agent = make_agent(seed=2, met=table)
    mode = MatchplayMode(match_length=3)
    for seed in (7, 8):
        (enc0, tgt0), (enc1, tgt1) = collect_twice(agent, mode, seed)
        np.testing.assert_array_equal(enc0, enc1)
        cube = enc0[:, IS_CUBE_ACTION_COL] == 1.0
        np.testing.assert_array_equal(tgt0[cube], tgt1[cube])


def test_cube_fold_variant():
    """cube_fold (explicit GetECF3 composition at the successors) is a
    no-op when the opponent has no cube window, and at the champion's
    weights measures within a few mMWC of the plain target when the
    window is live (the oracle probe measured the two as statistically
    identical -- the ca=1 channel is trained to the same composition).
    Guards the wiring, not a semantic difference."""
    import random
    from backgammon_engine import opening_roll
    from model import TDNetwork
    from modes import CubeOwner, MatchState
    from td_agent import TDAgent

    net = TDNetwork.load("best_models/matchplay_1h64met_512_512_256_256.pt")
    ag = TDAgent(net)
    rng = random.Random(9)
    state, _ = opening_roll(rng)

    blocked = MatchState(cube_owner=CubeOwner(state.turn + 1),
                         cube_value=2, jacoby=False,
                         match_length=5, my_away=4, opp_away=4)
    a = ag.value_oneply_checker_matchplay(state, blocked)
    b = ag.value_oneply_checker_matchplay(state, blocked, cube_fold=True)
    assert a == b

    live = MatchState(jacoby=False, match_length=3,
                      my_away=3, opp_away=3)
    a = ag.value_oneply_checker_matchplay(state, live)
    b = ag.value_oneply_checker_matchplay(state, live, cube_fold=True)
    assert 0.0 <= b <= 1.0
    assert abs(a - b) < 0.02


# ── identity 3: 1-ply targets are UNIFORMLY unconditional ───────────


class _T1Spy(TDAgent):
    """Records every 1-ply target the collector requests."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.t1_log = []

    def value_oneply_checker_matchplay(self, state, match_state,
                                       cube_fold=False):
        v = super().value_oneply_checker_matchplay(
            state, match_state, cube_fold)
        self.t1_log.append(v)
        return v


def test_oneply_targets_all_come_from_the_enumeration():
    """Regression: the collector used to fall into the realized
    boundary branch at the game-ending turn (and the 0-ply read at a
    forced pass), so ~2% of 1-ply rows carried a target conditioned on
    the mover's roll having won (or danced) -- +20 mMWC vs T1 at
    champion weights on the winner's side, never balanced -- while
    every other row was the unconditional dice average. In 1-ply mode
    EVERY checker row's target must be the enumeration value, in
    order, including the last row of each game. ML=1 keeps the cube
    silent so every emitted row is a checker row."""
    torch.manual_seed(0)
    net = TDNetwork(
        hidden_sizes=[64, 32], input_size=207,
        encoder_name="matchplay1h_perspective196",
        output_mode="probability",
    )
    agent = _T1Spy(net)
    mode = MatchplayMode(match_length=1)
    for seed in (0, 1, 2):
        agent.t1_log = []
        enc, tgt = _collect_episode_matchplay(
            agent, mode, random.Random(seed),
            mode.initial_match_state(), oneply=True)
        assert (enc[:, IS_CUBE_ACTION_COL] == 0.0).all()
        assert len(agent.t1_log) == len(tgt), (
            f"seed {seed}: {len(tgt)} rows but {len(agent.t1_log)} "
            "1-ply enumerations -- some row got a conditioned target")
        np.testing.assert_array_equal(
            tgt, np.asarray(agent.t1_log, dtype=np.float32),
            err_msg=(f"seed {seed}: some 1-ply row target is not the "
                     "unconditional enumeration value"))
