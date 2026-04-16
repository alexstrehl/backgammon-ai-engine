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
from encoding import NUM_FEATURES, ENCODERS, get_encoder, CubePerspective, CubefulEncoder
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


class TestCubefulEncoder:
    """CubefulEncoder = base encoder + 3-feature cube one-hot."""

    def test_factory_recognises_cubeful_prefix(self):
        enc = get_encoder("cubeful_perspective196")
        assert isinstance(enc, CubefulEncoder)
        assert enc.name == "cubeful_perspective196"
        assert enc.num_features == 199

    def test_factory_unknown_base_raises(self):
        with pytest.raises(ValueError, match="Unknown base encoder"):
            get_encoder("cubeful_nonsense")

    def test_one_hot_appended(self):
        enc = get_encoder("cubeful_perspective196")
        state = BoardState.initial()
        for cube in (CubePerspective.CENTERED,
                     CubePerspective.MINE,
                     CubePerspective.THEIRS):
            x = enc.encode(state, cube)
            assert x.shape == (199,)
            assert x[196 + int(cube)] == 1.0
            assert x[196:].sum() == 1.0

    def test_base_features_match_perspective196(self):
        """First 196 features must equal Perspective196Encoder output."""
        cubeful = get_encoder("cubeful_perspective196")
        base = get_encoder("perspective196")
        state = BoardState.initial()
        x = cubeful.encode(state, CubePerspective.CENTERED)
        np.testing.assert_array_equal(x[:196], base.encode(state))

