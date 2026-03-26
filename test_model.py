"""
test_model.py -- Tests for model.py.

Includes:
    - Architecture unit tests (no training)
    - Save/load tests
    - Smoke training test (full pipeline)

Run with: pytest test_model.py -v
"""

import tempfile
import os
import random
import numpy as np
import torch
import pytest

from model import TDNetwork
from encoding import NUM_FEATURES, ENCODERS, get_encoder
from backgammon_engine import BoardState, WHITE, BLACK, get_legal_plays, switch_turn
from agents import RandomAgent
from td_agent import TDAgent


# ── Architecture Unit Tests ──────────────────────────────────────────────────

class TestArchitecture:
    """Fast architecture tests (no training)."""

    def test_single_layer_default(self):
        """Build default network (1 layer, 40 units, relu)."""
        net = TDNetwork()
        assert net.hidden_sizes == [40]
        assert net.activation == "relu"
        assert net.input_size == NUM_FEATURES

    def test_forward_pass_single_vector(self):
        """Forward pass on a single 196-vector."""
        net = TDNetwork()
        x = torch.randn(NUM_FEATURES)
        output = net(x)
        assert output.shape == ()  # scalar
        assert 0.0 <= output.item() <= 1.0

    def test_forward_pass_batch(self):
        """Forward pass on a batch of vectors."""
        net = TDNetwork()
        x = torch.randn(10, NUM_FEATURES)
        output = net(x)
        assert output.shape == (10,)
        assert torch.all((output >= 0.0) & (output <= 1.0))

    def test_two_layer_relu(self):
        """Build a 2-layer ReLU network [80, 40]."""
        net = TDNetwork(hidden_sizes=[80, 40], activation="relu")
        assert net.hidden_sizes == [80, 40]
        assert net.activation == "relu"

        x = torch.randn(NUM_FEATURES)
        output = net(x)
        assert output.shape == ()
        assert 0.0 <= output.item() <= 1.0

    def test_three_layer_tanh(self):
        """Build a 3-layer tanh network [100, 60, 30]."""
        net = TDNetwork(hidden_sizes=[100, 60, 30], activation="tanh")
        assert net.hidden_sizes == [100, 60, 30]
        assert net.activation == "tanh"

        x = torch.randn(10, NUM_FEATURES)
        output = net(x)
        assert output.shape == (10,)
        assert torch.all((output >= 0.0) & (output <= 1.0))

    def test_leaky_relu_activation(self):
        """Build network with leaky_relu activation."""
        net = TDNetwork(hidden_sizes=[64, 32], activation="leaky_relu")
        assert net.activation == "leaky_relu"

        x = torch.randn(NUM_FEATURES)
        output = net(x)
        assert 0.0 <= output.item() <= 1.0

    def test_invalid_activation_raises(self):
        """Invalid activation name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            TDNetwork(activation="invalid_activation")



# ── Save/Load Tests ──────────────────────────────────────────────────────────

class TestSaveLoad:
    """Test model persistence."""

    def test_save_and_load(self):
        """Save and load a model with hidden_sizes and activation."""
        net_orig = TDNetwork(hidden_sizes=[80, 40], activation="relu")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            net_orig.save(path)

            net_loaded = TDNetwork.load(path)
            assert net_loaded.hidden_sizes == [80, 40]
            assert net_loaded.activation == "relu"

    def test_save_includes_train_params(self):
        """Saved model includes train_params dict."""
        net = TDNetwork()
        train_params = {"episodes": 1000, "lr": 0.1}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            net.save(path, train_params=train_params)

            net_loaded = TDNetwork.load(path)
            assert hasattr(net_loaded, '_train_params')
            assert net_loaded._train_params == train_params

    def test_load_preserves_weights(self):
        """Load should restore exact weights."""
        net_orig = TDNetwork(hidden_sizes=[60, 30], activation="tanh")
        x = torch.randn(5, NUM_FEATURES)
        out_orig = net_orig(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            net_orig.save(path)

            net_loaded = TDNetwork.load(path)
            out_loaded = net_loaded(x)

            assert torch.allclose(out_orig, out_loaded, atol=1e-6)

    def test_resumed_from_attribute(self):
        """Load sets _resumed_from attribute."""
        net = TDNetwork()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            net.save(path)

            net_loaded = TDNetwork.load(path)
            assert hasattr(net_loaded, '_resumed_from')
            assert net_loaded._resumed_from == path


# ── Smoke Training Test ──────────────────────────────────────────────────────

class TestSmokeTraining:
    """End-to-end training test (slower, ~20 seconds)."""

    @pytest.mark.slow
    def test_smoke_training_default(self):
        """Train default network (40 HU, sigmoid) for 5000 episodes."""
        from td_train import train

        net = train(
            num_episodes=5000,
            hidden_sizes=[40],
            lr=0.1,
            print_every=1000,
            use_py_engine=True,
        )

        # Verify network was trained and has non-zero gradients
        assert net is not None
        assert net.hidden_sizes == [40]

        # Play games against random to check it learned something
        random_agent = RandomAgent()
        td_agent = TDAgent(net)

        wins = 0
        for _ in range(100):
            state = BoardState.initial()
            agents = {WHITE: td_agent, BLACK: random_agent}

            while not state.is_game_over():
                d1, d2 = random.randint(1, 6), random.randint(1, 6)
                plays = get_legal_plays(state, (d1, d2))
                if plays:
                    _, state = agents[state.turn].choose_play(state, (d1, d2), plays)
                state = switch_turn(state)

            winner = state.winner()
            if winner == WHITE:
                wins += 1

        win_rate = wins / 100
        print(f"\nTD vs Random: {win_rate:.1%} win rate")
        assert win_rate > 0.85, f"TD agent should beat random >85%, got {win_rate:.1%}"

    @pytest.mark.slow
    def test_smoke_training_multi_layer_relu(self):
        """Train a 2-layer ReLU network for 5000 episodes."""
        from td_train import train

        net = train(
            num_episodes=5000,
            hidden_sizes=[80, 40],
            activation="relu",
            lr=0.1,
            print_every=1000,
            use_py_engine=True,
        )

        assert net is not None
        assert net.hidden_sizes == [80, 40]
        assert net.activation == "relu"

        # Quick sanity check: can play a game
        random_agent = RandomAgent()
        td_agent = TDAgent(net)

        state = BoardState.initial()
        agents = {WHITE: td_agent, BLACK: random_agent}

        while not state.is_game_over():
            d1, d2 = random.randint(1, 6), random.randint(1, 6)
            plays = get_legal_plays(state, (d1, d2))
            if plays:
                _, state = agents[state.turn].choose_play(state, (d1, d2), plays)
            state = switch_turn(state)

        # Game completed without error
        assert state.is_game_over()


# ── Additional Edge Case Tests ───────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_hidden_unit(self):
        """Network with just 1 hidden unit."""
        net = TDNetwork(hidden_sizes=[1])
        x = torch.randn(NUM_FEATURES)
        output = net(x)
        assert 0.0 <= output.item() <= 1.0

    def test_very_large_hidden_layer(self):
        """Network with a very large hidden layer."""
        net = TDNetwork(hidden_sizes=[512])
        x = torch.randn(NUM_FEATURES)
        output = net(x)
        assert 0.0 <= output.item() <= 1.0

    def test_many_layers(self):
        """Network with many (5) hidden layers."""
        net = TDNetwork(hidden_sizes=[100, 80, 60, 40, 20])
        x = torch.randn(NUM_FEATURES)
        output = net(x)
        assert 0.0 <= output.item() <= 1.0

    def test_activation_case_insensitive(self):
        """Activation names should be case insensitive."""
        net1 = TDNetwork(activation="RELU")
        net2 = TDNetwork(activation="relu")
        assert net1.activation == net2.activation

    def test_gradient_flow(self):
        """Test that gradients flow through all layers."""
        net = TDNetwork(hidden_sizes=[80, 40])
        x = torch.randn(NUM_FEATURES, requires_grad=True)
        y = net(x)
        loss = y.sum()
        loss.backward()

        # All parameters should have gradients
        for param in net.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0), "Some gradients are zero"


# ── Encoder Tests ─────────────────────────────────────────────────────────────

class TestEncoderAbstraction:
    """Test the encoder registry and TDNetwork encoder_name integration."""

    def test_default_encoder_is_perspective196(self):
        """ENCODERS dict must contain perspective196 with 196 features."""
        assert "perspective196" in ENCODERS
        encoder = get_encoder("perspective196")
        assert encoder.name == "perspective196"
        assert encoder.num_features == 196

    def test_encoder_name_saved_and_loaded(self):
        """encoder_name must round-trip through save/load."""
        net = TDNetwork(hidden_sizes=[40], encoder_name="perspective196")
        assert net.encoder_name == "perspective196"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            net.save(path)

            checkpoint = torch.load(path, weights_only=False)
            assert "encoder_name" in checkpoint
            assert checkpoint["encoder_name"] == "perspective196"

            net_loaded = TDNetwork.load(path)
            assert net_loaded.encoder_name == "perspective196"

    def test_model_input_size_matches_encoder(self):
        """input_size must equal encoder num_features."""
        net = TDNetwork(encoder_name="perspective196")
        encoder = get_encoder("perspective196")
        assert net.input_size == encoder.num_features
        assert net.input_size == NUM_FEATURES

