"""
test_prob5.py -- Tests for the 5-output probability model (prob5).

Covers ProbNetwork save/load, load_model dispatch, the equity formula,
the nested-event postprocess clamp, ProbAgent move selection, and the
play_models prob5/scalar dispatch.

Run with: pytest tests/test_prob5.py -v
"""

import os
import shutil
import struct
import subprocess
import tempfile

import numpy as np
import torch
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from model import (
    TDNetwork, ProbNetwork, load_model,
    prob5_to_equity, prob5_postprocess,
)
from prob_agent import ProbAgent
from td_agent import TDAgent
from play_models import _load_agent
from backgammon_engine import opening_roll, get_legal_plays, switch_turn


class TestProbNetwork:
    def test_output_shape(self):
        net = ProbNetwork(input_size=196, hidden_sizes=[16, 16])
        out = net(torch.zeros(4, 196))
        assert out.shape == (4, 5)

    def test_save_load_roundtrip(self):
        net = ProbNetwork(input_size=196, hidden_sizes=[16, 8])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pt")
            net.save(p)
            back = ProbNetwork.load(p)
        assert back.hidden_sizes == [16, 8]
        assert back.encoder_name == net.encoder_name
        x = torch.randn(3, 196)
        assert torch.allclose(net(x), back(x))

    def test_raw_logits_roundtrip_and_forward(self):
        net = ProbNetwork(input_size=196, hidden_sizes=[12, 8], raw_logits=True)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pt")
            net.save(p)
            back = load_model(p)
        assert isinstance(back, ProbNetwork) and back.raw_logits is True
        x = torch.randn(4, 196)
        assert torch.allclose(net(x), back(x))
        # raw logits are unbounded; sigmoid maps them into (0, 1)
        probs = torch.sigmoid(net(x))
        assert (probs > 0).all() and (probs < 1).all()

    def test_load_rejects_scalar_checkpoint(self):
        scalar = TDNetwork(input_size=196, hidden_sizes=[8])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.pt")
            scalar.save(p)
            # Legacy scalar checkpoints have no model_type field, so the
            # rejection surfaces as a state_dict RuntimeError rather than a
            # clean ValueError. Either way it must not load as prob5.
            with pytest.raises((ValueError, RuntimeError)):
                ProbNetwork.load(p)


class TestLoadModelDispatch:
    def test_prob5_dispatch(self):
        net = ProbNetwork(input_size=196, hidden_sizes=[8])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pt")
            net.save(p)
            assert isinstance(load_model(p), ProbNetwork)

    def test_scalar_dispatch(self):
        net = TDNetwork(input_size=196, hidden_sizes=[8])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.pt")
            net.save(p)
            assert isinstance(load_model(p), TDNetwork)


class TestEquityAndPostprocess:
    def test_equity_extremes(self):
        assert float(prob5_to_equity(torch.tensor([[1.0, 0, 0, 0, 0]]))) == pytest.approx(1.0)
        assert float(prob5_to_equity(torch.tensor([[0.0, 0, 0, 0, 0]]))) == pytest.approx(-1.0)

    def test_equity_gammon_terms(self):
        # certain win + certain win-gammon => 2*1 + 1 - 1 = 2
        eq = prob5_to_equity(torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]))
        assert float(eq) == pytest.approx(2.0)

    def test_postprocess_enforces_inequalities(self):
        raw = torch.tensor([[0.6, 0.9, 0.8, 0.7, 0.6]])  # all violated
        p = prob5_postprocess(raw)[0]
        assert p[1] <= p[0] + 1e-6           # wg <= win
        assert p[2] <= p[1] + 1e-6           # wbg <= wg
        assert p[3] <= (1 - p[0]) + 1e-6     # lg <= lose
        assert p[4] <= p[3] + 1e-6           # lbg <= lg

    def test_postprocess_does_not_mutate_input(self):
        raw = torch.tensor([[0.6, 0.9, 0.8, 0.7, 0.6]])
        before = raw.clone()
        prob5_postprocess(raw)
        assert torch.equal(raw, before)


class TestProbAgentPlay:
    @pytest.mark.parametrize("plies", [0, 1, 2])
    def test_choose_returns_legal_play(self, plies):
        net = ProbNetwork(input_size=196, hidden_sizes=[16])
        agent = ProbAgent(net, plies=plies)
        state, (d1, d2) = opening_roll()
        plays = get_legal_plays(state, (d1, d2))
        play, nxt = agent.choose_checker_action(state, (d1, d2), plays)
        assert play in [pl for pl, _ in plays]

    def test_cube_decisions_rejected(self):
        agent = ProbAgent(ProbNetwork(input_size=196, hidden_sizes=[8]))
        with pytest.raises(NotImplementedError):
            agent.offer_double(None, None)
        with pytest.raises(NotImplementedError):
            agent.respond_to_double(None, None)


class TestPlayModelsDispatch:
    def _save(self, net, d, name):
        p = os.path.join(d, name)
        net.save(p)
        return p

    def test_prob5_routes_to_probagent(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._save(ProbNetwork(input_size=196, hidden_sizes=[8]), d, "p.pt")
            assert isinstance(_load_agent(p), ProbAgent)
            assert _load_agent(p, oneply=True).plies == 1
            assert _load_agent(p, twoply_k=5).plies == 2

    def test_scalar_routes_to_tdagent(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._save(TDNetwork(input_size=196, hidden_sizes=[8]), d, "s.pt")
            assert isinstance(_load_agent(p, oneply=True), TDAgent)

    def test_bf16_flag_threads_to_probagent(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._save(ProbNetwork(input_size=196, hidden_sizes=[8]), d, "p.pt")
            assert _load_agent(p, bf16=True).bf16_inference is True
            assert _load_agent(p, bf16=False).bf16_inference is False

    def test_prob5_uses_money_equity_not_dmp(self):
        # prob5 is a cubeless-money model; the play dispatch must never
        # switch it to DMP P(win) scoring.
        with tempfile.TemporaryDirectory() as d:
            p = self._save(ProbNetwork(input_size=196, hidden_sizes=[8]), d, "p.pt")
            assert _load_agent(p).dmp is False


class TestExpand:
    def test_width_expand_preserves_function(self):
        src = ProbNetwork(input_size=196, hidden_sizes=[16, 8])
        wide = ProbNetwork.width_expand(src, [32, 16])
        x = torch.randn(5, 196)
        assert torch.allclose(src(x), wide(x), atol=1e-5)

    def test_depth_expand_identity_preserves_function(self):
        # eps=0 -> exact identity; post-ReLU non-negativity makes the
        # appended ReLU layer an exact pass-through.
        src = ProbNetwork(input_size=196, hidden_sizes=[16, 8])
        deep = ProbNetwork.depth_expand(src, eps=0.0)
        assert deep.hidden_sizes == [16, 8, 8]
        x = torch.randn(5, 196)
        assert torch.allclose(src(x), deep(x), atol=1e-5)


class TestTrainTargets:
    def test_flip_vec_swaps_win_loss(self):
        from train_prob5 import _flip_vec
        v = np.array([0.7, 0.2, 0.1, 0.3, 0.05], dtype=np.float32)
        # mover target = [1-P(win), P(lg), P(lbg), P(wg), P(wbg)]
        assert np.allclose(_flip_vec(v), [0.3, 0.3, 0.05, 0.2, 0.1])

    def test_flip_vec_is_involution(self):
        from train_prob5 import _flip_vec
        v = np.array([0.6, 0.25, 0.1, 0.2, 0.05], dtype=np.float32)
        assert np.allclose(_flip_vec(_flip_vec(v)), v)

    def test_oneply_target_hand_derived(self):
        # Stub net returns a constant prob vector c for any input. At the
        # opening position every die has non-terminal moves and no forced
        # pass, so each per-die best-move target is flip(c); the
        # dice-probability-weighted sum is therefore exactly flip(c).
        import train_prob5 as T
        from train_prob5 import _flip_vec

        class _ConstNet:
            raw_logits = False

            def __init__(self, c):
                self.c = torch.tensor(c, dtype=torch.float32)

            def __call__(self, x):
                return self.c.expand(x.shape[0], 5).clone()

        enc, gpe = T._resolve_engine_fns("perspective196")
        state, _ = opening_roll(np.random.RandomState(7))
        c = [0.6, 0.2, 0.1, 0.15, 0.05]
        tgt = T._oneply_target_vec(state, _ConstNet(c), enc, gpe, "cpu")
        assert np.allclose(tgt, _flip_vec(np.array(c, dtype=np.float32)), atol=1e-6)


class TestZeroPlyTerminal:
    """0-ply move selection must take an immediately winning move even
    when the raw network output mis-ranks the terminal encoding."""

    def test_0ply_takes_winning_move(self):
        from backgammon_engine import BoardState, BLACK
        # WHITE has borne off all 15 (won a single: BLACK has 3 off, 12
        # still on the board); engine has switched turn to BLACK.
        points = [0] * 24
        points[12] = -12
        terminal = BoardState(points=points, bar=[0, 0], off=[15, 3], turn=BLACK)
        assert terminal.is_game_over() and terminal.game_result() == 1
        nonterminal, _ = opening_roll(np.random.RandomState(3))

        agent = ProbAgent(ProbNetwork(input_size=196, hidden_sizes=[8]), plies=0)
        # Adversarial ranking: network rates the winning move WORSE.
        agent.evaluate_batch = lambda states: np.array(
            [0.9 if s is terminal else -0.9 for s in states], dtype=np.float64)
        actions = [("win", terminal), ("other", nonterminal)]
        _, chosen = agent.choose_checker_action(terminal, (1, 2), actions)
        assert chosen is terminal


class TestFastEngineParity:
    """The C fast path must match the pure-Python engine exactly."""

    def _states(self, n=12, seed=4):
        rng = np.random.RandomState(seed)
        out = []
        while len(out) < n:
            state, (d1, d2) = opening_roll(rng)
            for _ in range(rng.randint(0, 10)):
                if state.is_game_over():
                    break
                plays = get_legal_plays(state, (d1, d2))
                if plays:
                    _, state = plays[rng.randint(len(plays))]
                else:
                    state = switch_turn(state)
                d1, d2 = rng.randint(1, 7), rng.randint(1, 7)
            if not state.is_game_over():
                out.append((state, (int(d1), int(d2))))
        return out

    def test_probagent_oneply_matches_python(self):
        net = ProbNetwork(input_size=196, hidden_sizes=[24, 16])
        fast = ProbAgent(net, plies=1, use_fast_engine=True)
        slow = ProbAgent(net, plies=1, use_fast_engine=False)
        if not fast._fast_engine:
            pytest.skip("C engine unavailable")
        for state, dice in self._states():
            assert abs(fast.value_oneply_checker(state)
                       - slow.value_oneply_checker(state)) < 1e-9
            plays = get_legal_plays(state, dice)
            pf, _ = fast.choose_checker_action(state, dice, plays)
            ps, _ = slow.choose_checker_action(state, dice, plays)
            assert pf == ps

    def test_train_oneply_target_matches_python(self):
        import train_prob5 as T
        net = ProbNetwork(input_size=196, hidden_sizes=[24, 16])
        enc_c, gpe_c = T._resolve_engine_fns("perspective196")
        if not T._BG_FAST_AVAILABLE:
            pytest.skip("C engine unavailable")
        orig = T._BG_FAST_AVAILABLE
        T._BG_FAST_AVAILABLE = False
        try:
            enc_p, gpe_p = T._resolve_engine_fns("perspective196")
        finally:
            T._BG_FAST_AVAILABLE = orig
        for state, _ in self._states():
            tc = T._oneply_target_vec(state, net, enc_c, gpe_c, "cpu")
            tp = T._oneply_target_vec(state, net, enc_p, gpe_p, "cpu")
            assert np.allclose(tc, tp, atol=1e-9)


class TestShippedCheckpoint:
    PATH = "best_models/cubeless_prob5_512_512_256_128.pt"

    @pytest.mark.skipif(not os.path.exists(PATH), reason="artifact not present")
    def test_loads_with_expected_metadata(self):
        net = load_model(self.PATH)
        assert isinstance(net, ProbNetwork)
        assert net.hidden_sizes == [512, 512, 256, 128]
        assert net.encoder_name == "perspective196"


class TestExportWeights:
    """export_weights.py emits a valid prob5 BGNN binary."""

    def test_prob5_binary_header_and_size(self):
        import export_weights as EW
        net = ProbNetwork(input_size=196, hidden_sizes=[16, 8])
        with tempfile.TemporaryDirectory() as d:
            pt = os.path.join(d, "m.pt"); net.save(pt)
            binp = os.path.join(d, "m.bin")
            EW.export_model(pt, binp)
            with open(binp, "rb") as f:
                raw = f.read()
        assert raw[:4] == b"BGNN"
        num_hidden, input_size, act, out_mode = struct.unpack("<iiii", raw[4:20])
        assert (num_hidden, input_size, out_mode) == (2, 196, EW.OUTPUT_MODE_PROB5)
        hidden = struct.unpack("<ii", raw[20:28])
        assert list(hidden) == [16, 8]
        # weights: layers (196->16, 16->8) + 5-wide output (8->5)
        params = (196 * 16 + 16) + (16 * 8 + 8) + (8 * 5 + 5)
        assert len(raw) == 28 + params * 4

    def test_scalar_export_still_one_output(self):
        import export_weights as EW
        net = TDNetwork(input_size=196, hidden_sizes=[8])
        with tempfile.TemporaryDirectory() as d:
            pt = os.path.join(d, "m.pt"); net.save(pt)
            binp = os.path.join(d, "m.bin")
            EW.export_model(pt, binp)
            with open(binp, "rb") as f:
                raw = f.read()
        _, _, _, out_mode = struct.unpack("<iiii", raw[4:20])
        assert out_mode != EW.OUTPUT_MODE_PROB5  # scalar: probability/equity


class TestGnubgEvalDispatch:
    def test_prob5_routes_to_probagent(self):
        import gnubg_eval as G
        with tempfile.TemporaryDirectory() as d:
            pt = os.path.join(d, "p.pt"); ProbNetwork(input_size=196, hidden_sizes=[8]).save(pt)
            assert isinstance(G._load_eval_agent(pt), ProbAgent)
            assert isinstance(G._load_eval_agent(pt, oneply=True), ProbAgent)

    def test_scalar_routes_to_tdagent(self):
        import gnubg_eval as G
        with tempfile.TemporaryDirectory() as d:
            pt = os.path.join(d, "s.pt"); TDNetwork(input_size=196, hidden_sizes=[8]).save(pt)
            assert isinstance(G._load_eval_agent(pt), TDAgent)

    def test_cubeful_prob5_rejected(self):
        import gnubg_eval as G
        with tempfile.TemporaryDirectory() as d:
            pt = os.path.join(d, "p.pt"); ProbNetwork(input_size=196, hidden_sizes=[8]).save(pt)
            # Parent-side validation (called before workers spawn) and the
            # loader backstop both reject prob5 + cubeful with a clean error.
            with pytest.raises(ValueError):
                G.assert_cubeful_supported(pt, cubeful=True)
            with pytest.raises(ValueError):
                G._load_eval_agent(pt, cubeful=True)


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
class TestCInferenceParity:
    """The C nn_eval prob5 forward must match Python ProbNetwork."""

    _HARNESS = r'''
#include "nn_eval.h"
#include <stdio.h>
#include <stdlib.h>
int main(int c, char **v) {
    NNModel m; if (nn_load(&m, v[1])) return 1;
    FILE *f = fopen(v[2], "rb");
    float *in = malloc((size_t)m.input_size * sizeof(float));
    if (fread(in, sizeof(float), m.input_size, f) != (size_t)m.input_size) return 3;
    fclose(f);
    float p[5]; float eq = nn_forward_prob5(&m, in, p);
    printf("%.8f %.8f %.8f %.8f %.8f %.8f\n", eq, p[0], p[1], p[2], p[3], p[4]);
    return 0;
}'''

    def test_c_matches_python(self):
        import export_weights as EW
        cinf = os.path.join(_REPO, "c_inference")
        net = ProbNetwork(input_size=196, hidden_sizes=[16, 8])
        with tempfile.TemporaryDirectory() as d:
            harness = os.path.join(d, "h.c")
            with open(harness, "w") as f:
                f.write(self._HARNESS)
            exe = os.path.join(d, "h")
            r = subprocess.run(
                ["gcc", "-O2", "-o", exe, harness,
                 os.path.join(cinf, "nn_eval.c"), "-I", cinf, "-lm"],
                capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            pt = os.path.join(d, "m.pt"); net.save(pt)
            binp = os.path.join(d, "m.bin"); EW.export_model(pt, binp)
            rng = np.random.RandomState(0)
            for _ in range(8):
                x = rng.randn(196).astype(np.float32)
                inp = os.path.join(d, "in.bin"); x.tofile(inp)
                out = subprocess.run([exe, binp, inp], capture_output=True, text=True).stdout.split()
                c_eq = float(out[0])
                with torch.no_grad():
                    p = prob5_postprocess(net(torch.from_numpy(x)[None, :]))
                    py_eq = float(prob5_to_equity(p)[0])
                assert abs(c_eq - py_eq) < 1e-4


class TestOneplyBatchedParity:
    """The batched 1-ply forward (one GEMM over all 21 dice) must match the
    pre-batching per-dice implementation within fp tolerance, and must not
    change move decisions. Not bit-exact: batched vs per-chunk GEMM differs
    by ~1 fp32 ULP (accumulation order), so the bar is <=1e-6, decisions equal.
    """

    @staticmethod
    def _ref_value_oneply(agent, state):
        """Reference: the per-dice implementation prob5 used before batching."""
        from prob_agent import _DICE_OUTCOMES
        from encoding import OPP_OFF_INDEX, TERMINAL_OFF_THRESHOLD
        if state.is_game_over():
            return -agent._terminal_mag(state)
        s = 0.0
        for d1, d2 in _DICE_OUTCOMES:
            prob = (1.0 / 36.0) if d1 == d2 else (2.0 / 36.0)
            features, next_states = agent._get_legal_plays_encoded(state, (d1, d2))
            if len(next_states) == 0:
                s += prob * (-agent.evaluate(switch_turn(state)))
                continue
            opp = agent._score(agent._probs(torch.from_numpy(features))).detach().cpu().numpy()
            for j in np.flatnonzero(features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD):
                opp[j] = -agent._terminal_mag(next_states[int(j)])
            s += prob * (-float(np.min(opp)))
        return s

    @staticmethod
    def _positions(n, seed):
        import random
        rng = random.Random(seed)
        out = []
        state, dice = opening_roll(rng)
        while len(out) < n:
            if state.is_game_over():
                state, dice = opening_roll(rng); continue
            out.append(state)
            plays = get_legal_plays(state, dice)
            state = rng.choice(plays)[1] if plays else switch_turn(state)
            dice = (rng.randint(1, 6), rng.randint(1, 6))
        return out

    def _agent(self):
        net = ProbNetwork(input_size=196, hidden_sizes=[64, 64],
                          encoder_name="perspective196")
        net.eval()
        return ProbAgent(net, plies=1)

    def test_value_parity(self):
        agent = self._agent()
        maxd = max(abs(agent.value_oneply_checker(s) - self._ref_value_oneply(agent, s))
                   for s in self._positions(40, seed=0))
        assert maxd < 1e-6, f"batched 1-ply diverged from per-dice: max |delta|={maxd:.2e}"

    def test_decision_parity(self):
        agent = self._agent()
        checked = 0
        for s in self._positions(40, seed=1):
            for d in [(3, 1), (6, 4), (5, 5), (2, 1)]:
                plays = get_legal_plays(s, d)
                if len(plays) < 2:
                    continue
                checked += 1
                _, ns_batched = agent.choose_checker_action(s, d, plays)
                ref = [(-agent._terminal_mag(ns) if ns.is_game_over()
                        else self._ref_value_oneply(agent, ns)) for _, ns in plays]
                chosen_idx = [p[1] for p in plays].index(ns_batched)
                # The batched choice must be reference-optimal (ties within 1e-6 ok).
                assert ref[chosen_idx] - min(ref) < 1e-6
        assert checked > 0
