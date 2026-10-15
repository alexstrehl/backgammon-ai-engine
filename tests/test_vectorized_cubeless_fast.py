"""BG_VEC_CUBELESS_FAST parity. The batched cubeless 1-ply generator must
agree with the default per-dice generator to float32 round-off. Bitwise
equality is NOT expected, because the two paths batch forwards differently.
"""
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backgammon_engine import BoardState, get_legal_plays, switch_turn
from model import TDNetwork
from td_agent import TDAgent
import vectorized_collect as V

ATOL = 1e-6  # float32 round-off headroom


def _net(output_mode):
    torch.manual_seed(0)
    net = TDNetwork(
        hidden_sizes=[32, 16], input_size=196, output_mode=output_mode,
        encoder_name="perspective196",
    )
    net.eval()
    return net


def _sample_states(n=24, seed=7):
    """Non-terminal states from a random playout of the opening."""
    rng = random.Random(seed)
    states, s = [], BoardState.initial()
    while len(states) < n:
        dice = (rng.randint(1, 6), rng.randint(1, 6))
        plays = get_legal_plays(s, dice)
        s = rng.choice(plays)[1] if plays else s
        s = switch_turn(s)
        if s.is_game_over():
            s = BoardState.initial()
            continue
        states.append(s)
    return states


def _drive(gen_fn, agent, net, state):
    g = gen_fn(agent, state)
    try:
        feat = next(g)
        with torch.no_grad():
            while True:
                out = net(torch.from_numpy(np.asarray(feat, dtype=np.float32)))
                feat = g.send(out.reshape(-1).numpy())
    except StopIteration as e:
        return e.value


class TestCubelessFastPathParity:
    @pytest.mark.parametrize("output_mode", ["equity", "probability"])
    def test_fast_matches_default_to_roundoff(self, output_mode):
        """Both modes: DMP (probability) routes through the same generator."""
        net = _net(output_mode)
        agent = TDAgent(net, oneply=False, boltzmann_temp=0.0, device="cpu")
        states = _sample_states()
        refs, worst = [], 0.0
        for state in states:
            ref = float(_drive(V._value_oneply_checker_cubeless_gen, agent, net, state))
            fast = float(_drive(V._value_oneply_checker_cubeless_gen_fast, agent, net, state))
            refs.append(ref)
            worst = max(worst, abs(ref - fast))
            assert abs(ref - fast) <= ATOL, (
                f"{output_mode}: fast path diverged |d|={abs(ref-fast):.3e} > {ATOL}")
        # Guards against a vacuous comparison: enough states, all finite,
        # and genuinely different positions.
        assert len(refs) >= 20, f"only {len(refs)} states compared"
        assert all(np.isfinite(r) for r in refs), "non-finite target"
        assert max(refs) - min(refs) > 1e-6, (
            f"{output_mode}: all {len(refs)} targets identical ({refs[0]!r}) "
            f"-- the comparison would pass even if both paths were broken")

    def test_env_var_selects_fast_path(self):
        code = (
            "import os, vectorized_collect as V; "
            "print(V._CUBELESS_1PLY_GEN.__name__)"
        )
        for env_val, expected in ((None, "_value_oneply_checker_cubeless_gen"),
                                  ("1", "_value_oneply_checker_cubeless_gen_fast")):
            env = dict(os.environ)
            env.pop("BG_VEC_CUBELESS_FAST", None)
            if env_val is not None:
                env["BG_VEC_CUBELESS_FAST"] = env_val
            out = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True,
                env=env, cwd=os.path.join(os.path.dirname(__file__), ".."),
                timeout=120,
            )
            assert out.returncode == 0, out.stderr[-500:]
            assert out.stdout.strip() == expected
