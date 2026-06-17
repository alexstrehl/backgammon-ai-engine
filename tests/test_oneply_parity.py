"""Parity harness for value_oneply_checker_cubeful.

Generates a deterministic fixture of cubeful-money states via seeded
self-play with a fixed-weight network, then runs current-vs-current
checks on value_oneply_checker_cubeful: determinism (repeated calls
agree exactly), bounded outputs, and bf16 inference parity with the
fp32 reference.

The fixture is regenerated on each test run (seeded), so there is no
on-disk golden file to keep in sync. A future C/fast path for 1-ply
target computation can plug into the same fixture and assert against
the current Python reference at ~1e-5 tolerance.
"""

import random
from dataclasses import replace
from typing import List, Tuple

import numpy as np
import pytest
import torch

from backgammon_engine import BoardState, opening_roll, switch_turn
from modes import CubefulMoneyMode, MatchState, CubeOwner
from model import TDNetwork
from td_agent import TDAgent, cpu_supports_fast_bf16

# bf16 inference is auto-disabled on CPUs without bf16 ISA support
# (torch emulation is ~20x slower than fp32), so its behavior can only
# be tested on hosts that have it.
requires_bf16 = pytest.mark.skipif(
    not cpu_supports_fast_bf16(),
    reason="CPU lacks bf16 ISA (avx512_bf16/amx_bf16); "
           "TDAgent auto-disables bf16 inference on this host",
)


def _make_fixed_net(seed: int = 12345) -> TDNetwork:
    """Build a [512,512,256] cubeful_perspective196 equity net with
    deterministic small weights. Architecture matches the production
    3-layer cubeful model.
    """
    torch.manual_seed(seed)
    net = TDNetwork(
        hidden_sizes=[512, 512, 256],
        encoder_name="cubeful_perspective196",
        output_mode="equity",
    )
    # Shrink initial weights so the untrained net produces small,
    # non-saturated outputs — better signal for detecting numerical
    # drift in a fast path.
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(0.05)
    return net


def _generate_states(
    agent: TDAgent, n: int, seed: int = 42,
) -> List[Tuple[BoardState, MatchState]]:
    """Seeded self-play to collect `n` (state, match_state) pairs at
    cubeful-money decision points. Includes a mix of opening, midgame,
    and bearoff positions. The cube may or may not have been turned.
    """
    mode = CubefulMoneyMode(jacoby=True)
    rng = random.Random(seed)
    out: List[Tuple[BoardState, MatchState]] = []
    games_tried = 0
    max_games = 400

    while len(out) < n and games_tried < max_games:
        games_tried += 1
        state, dice = opening_roll(rng)
        match_state = mode.initial_match_state()
        is_opening = True
        steps = 0
        while not mode.is_episode_over(state) and steps < 200:
            steps += 1
            player = state.turn

            # Sample cube-phase state when eligible.
            if not is_opening and match_state.can_offer(player):
                out.append((state, match_state))
                if len(out) >= n:
                    return out
                # 10% chance to actually double (to diversify cube_owner).
                if rng.random() < 0.1:
                    match_state = match_state.after_take(player)

            # Always sample the checker-phase state.
            out.append((state, match_state))
            if len(out) >= n:
                return out

            # Step checker phase via the agent.
            result = agent.choose_checker_action_cubeful(
                state, dice, match_state,
            )
            if result is not None:
                next_state, _ = result
                if next_state.is_game_over():
                    break
                state = next_state
            else:
                state = switch_turn(state)
            is_opening = False
            dice = (rng.randint(1, 6), rng.randint(1, 6))

    return out


# Number of fixture states generated per run (seed=42).
N_STATES = 200


class TestOneplyCubefulParity:
    """Current-vs-current determinism checks, plus a hook for the
    future fast path to plug into.
    """

    @pytest.fixture(scope="class")
    def fixture(self):
        net = _make_fixed_net()
        agent = TDAgent(net, oneply=False)
        states = _generate_states(agent, n=N_STATES, seed=42)
        assert len(states) == N_STATES
        return net, agent, states

    def test_fixture_shape(self, fixture):
        net, agent, states = fixture
        cube_phases = sum(
            1 for _, m in states if m.can_offer(0) or m.can_offer(1)
        )
        bearoffs = sum(
            1 for s, _ in states
            if s.off[0] > 0 or s.off[1] > 0
        )
        # We want variety — not all openings, not all cube-eligible.
        assert cube_phases > 20
        assert bearoffs > 20

    def test_ref_values_deterministic(self, fixture):
        """Same inputs, same weights → same outputs across two calls.
        This catches non-determinism (e.g. accidental dropout/train-mode).
        """
        net, agent, states = fixture
        v1 = [agent.value_oneply_checker_cubeful(s, m) for s, m in states]
        v2 = [agent.value_oneply_checker_cubeful(s, m) for s, m in states]
        for a, b in zip(v1, v2):
            assert a == b, f"non-deterministic: {a} != {b}"

    def test_ref_values_bounded(self, fixture):
        """All 1-ply equity values must lie in [-3, +3] (max single-
        game payoff with cube=1 is ±3 backgammon). Sanity check.
        """
        net, agent, states = fixture
        for s, m in states:
            v = agent.value_oneply_checker_cubeful(s, m)
            assert -3.0 - 1e-6 <= v <= 3.0 + 1e-6, \
                f"value {v} out of [-3, 3] bounds for state {s}"

    @requires_bf16
    def test_bf16_inference_parity(self, fixture):
        """Bf16 inference path must match fp32 reference within a bf16-
        precision tolerance. Bf16 has ~3 decimal digits of mantissa
        (7-bit fraction), so per-value error is dominated by
        accumulated rounding across the 4-layer MLP — empirically
        well under 1e-2 on our 1-ply sums (which mix in fp32 terminal
        results and probability weights, damping the error).
        """
        net, agent, states = fixture
        refs = [agent.value_oneply_checker_cubeful(s, m) for s, m in states]

        # Build a second agent with bf16_inference enabled, sharing
        # the same fp32 weights. This matches what a worker does.
        bf16_agent = TDAgent(net, bf16_inference=True)
        fasts = [
            bf16_agent.value_oneply_checker_cubeful(s, m) for s, m in states
        ]

        diffs = [abs(r - f) for r, f in zip(refs, fasts)]
        max_abs = max(diffs)
        mean_abs = sum(diffs) / len(diffs)
        # Absolute tolerance picked to allow for the full bf16 rounding
        # budget on a mid-depth MLP. Tighten if this drifts up.
        assert max_abs < 1e-2, (
            f"bf16 1-ply diverges from fp32: max abs diff = {max_abs:.2e}, "
            f"mean = {mean_abs:.2e}"
        )
        # Also require that the mean error is well below the tolerance
        # — a uniformly large error would indicate a bug, not just
        # rounding.
        assert mean_abs < 2e-3, (
            f"bf16 1-ply mean error too high: {mean_abs:.2e}"
        )

    @requires_bf16
    def test_bf16_refresh_after_weight_change(self, fixture):
        """After the fp32 weights change and refresh_bf16_inference()
        is called, bf16 outputs must track the new weights (not stay
        frozen at the pre-change snapshot).
        """
        net, agent, states = fixture
        bf16_agent = TDAgent(net, bf16_inference=True)
        s, m = states[0]
        v0 = bf16_agent.value_oneply_checker_cubeful(s, m)

        # Perturb fp32 weights: scale the output layer. This produces
        # a clean, large change in every output value, well above bf16
        # rounding noise.
        with torch.no_grad():
            net.fc_output.weight.mul_(1.5)
            net.fc_output.bias.add_(0.1)

        # Without refresh, bf16 copy is stale — should give v0.
        v_stale = bf16_agent.value_oneply_checker_cubeful(s, m)
        assert v_stale == v0, \
            "bf16 copy should be stale until refresh_bf16_inference()"

        # After refresh, it must track the new weights.
        bf16_agent.refresh_bf16_inference()
        v_new = bf16_agent.value_oneply_checker_cubeful(s, m)
        assert v_new != v0, "bf16 copy did not refresh to new weights"
