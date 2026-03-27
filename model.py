"""
model.py -- Neural network for backgammon position evaluation.

Architecture:
    Input:  configurable (default 196: perspective encoding, 98 my + 98 opponent)
    Hidden: configurable (default [40], Tesauro used 40-80)
    Output: 1 unit     (sigmoid -> P(on-roll player wins))

Supports:
    - Multiple hidden layers of arbitrary sizes
    - Configurable activations: sigmoid, relu, tanh, leaky_relu
"""

import torch
import torch.nn as nn
from typing import List, Literal
from encoding import NUM_FEATURES


def _get_activation_fn(name: str):
    """Get activation function by name."""
    name = name.lower()
    if name == "sigmoid":
        return torch.sigmoid
    elif name == "relu":
        return torch.relu
    elif name == "tanh":
        return torch.tanh
    elif name == "leaky_relu":
        return torch.nn.functional.leaky_relu
    elif name == "hardsigmoid":
        return torch.nn.functional.hardsigmoid
    else:
        raise ValueError(f"Unknown activation: {name}. Choose from: sigmoid, relu, tanh, leaky_relu, hardsigmoid")


class TDNetwork(nn.Module):
    """Feedforward network for backgammon value estimation.

    Output is a single scalar in [0, 1] representing a win probability.
    Supports multiple hidden layers and configurable activation functions.
    """

    def __init__(
        self,
        hidden_sizes: List[int] = None,
        input_size: int = None,
        activation: str = "relu",
        encoder_name: str = "perspective196",
    ):
        super().__init__()
        self.encoder_name = encoder_name
        if input_size is None:
            from encoding import get_encoder
            input_size = get_encoder(encoder_name).num_features
        self.input_size = input_size

        if hidden_sizes is None:
            hidden_sizes = [40]

        self.hidden_sizes = hidden_sizes
        self.activation = activation.lower()
        self._activation_fn = _get_activation_fn(self.activation)

        # Build hidden layers
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, size))
            prev_size = size

        self._n_hidden = len(hidden_sizes)

        # Output layer: always sigmoid for win probability
        self.fc_output = nn.Linear(prev_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.  *x* can be a single 196-vector or a batch."""
        if self._n_hidden == 1:
            h = self._activation_fn(self.hidden_layers[0](x))
        else:
            h = x
            for layer in self.hidden_layers:
                h = self._activation_fn(layer(h))
        return torch.sigmoid(self.fc_output(h)).squeeze(-1)

    # ── Save / Load ──────────────────────────────────────────────────

    def save(self, path: str, train_params: dict = None):
        """Save model weights, architecture info, and optional training params."""
        data = {
            "hidden_sizes": self.hidden_sizes,
            "input_size": self.input_size,
            "activation": self.activation,
            "encoder_name": self.encoder_name,
            "state_dict": self.state_dict(),
        }
        if train_params:
            data["train_params"] = train_params
        torch.save(data, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "TDNetwork":
        """Load a saved model. Defaults to CPU to handle models saved on GPU."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        model = cls(
            hidden_sizes=checkpoint["hidden_sizes"],
            input_size=checkpoint.get("input_size", NUM_FEATURES),
            activation=checkpoint.get("activation", "relu"),
            encoder_name=checkpoint.get("encoder_name", "perspective196"),
        )

        model.load_state_dict(checkpoint["state_dict"])
        model._resumed_from = path
        model._train_params = checkpoint.get("train_params", {})
        return model

    @classmethod
    def width_expand(cls, source: "TDNetwork", new_hidden_sizes: List[int]) -> "TDNetwork":
        """Create a wider network initialized from a narrower one.

        Copies weights from *source* into the first N units of each layer
        in the new network.  New units get Kaiming-initialized input weights
        and zero output weights, so the expanded network is functionally
        identical to *source* at initialization.

        Args:
            source: The smaller trained network to expand from.
            new_hidden_sizes: Hidden layer sizes for the new (wider) network.
                Must have the same number of layers as source.

        Returns:
            A new TDNetwork with expanded width, preserving learned features.
        """
        old_sizes = source.hidden_sizes
        if len(new_hidden_sizes) != len(old_sizes):
            raise ValueError(
                f"Layer count mismatch: source has {len(old_sizes)} hidden layers, "
                f"new_hidden_sizes has {len(new_hidden_sizes)}"
            )
        for old_s, new_s in zip(old_sizes, new_hidden_sizes):
            if new_s < old_s:
                raise ValueError(
                    f"Cannot shrink: layer has {old_s} units, requested {new_s}"
                )

        expanded = cls(
            hidden_sizes=new_hidden_sizes,
            input_size=source.input_size,
            activation=source.activation,
            encoder_name=source.encoder_name,
        )

        with torch.no_grad():
            for i, (old_layer, new_layer) in enumerate(
                zip(source.hidden_layers, expanded.hidden_layers)
            ):
                old_out, old_in = old_layer.weight.shape
                # Copy existing weights and biases
                new_layer.weight[:old_out, :old_in] = old_layer.weight
                new_layer.bias[:old_out] = old_layer.bias
                # New input connections (rows beyond old_out) already Kaiming-init'd
                # by nn.Linear default. Zero out isn't needed for input weights.
                # But we need to handle the case where previous layer was also expanded:
                if i > 0:
                    prev_old = old_sizes[i - 1]
                    # New columns (connections from new units in previous layer)
                    # should be zero for existing units to preserve function
                    new_layer.weight[:old_out, prev_old:] = 0.0

            # Output layer: copy old weights, zero new connections
            old_out_w = source.fc_output.weight
            expanded.fc_output.weight[:, :old_sizes[-1]] = old_out_w
            expanded.fc_output.weight[:, old_sizes[-1]:] = 0.0
            expanded.fc_output.bias[:] = source.fc_output.bias

        expanded._expanded_from = f"{old_sizes} -> {new_hidden_sizes}"
        return expanded

    @classmethod
    def depth_expand(cls, source: "TDNetwork", new_layer_size: int = None, eps: float = 0.01) -> "TDNetwork":
        """Create a deeper network by appending one hidden layer, initialized as near-identity.

        The new layer is inserted before the output layer and initialized as
        ``I + ε·randn`` (identity plus small noise) so the network computes
        almost exactly the same function as *source* at initialization.
        Training then gradually activates the new layer's capacity.

        Because ReLU(h) = h for h ≥ 0 (all hidden activations after a ReLU
        layer are non-negative), the identity init gives exact function
        preservation at init; the ε noise breaks weight symmetry so neurons
        diversify during training.

        Args:
            source:          Trained network to deepen.
            new_layer_size:  Size of the appended layer.  Must equal the last
                             hidden size (required for square identity init).
                             Defaults to ``source.hidden_sizes[-1]``.
            eps:             Scale of random perturbation around identity.
                             0.01 is a good default: large enough for Adam to
                             differentiate neurons quickly, small enough not to
                             disturb the preserved function.

        Returns:
            A new TDNetwork with one additional hidden layer.
        """
        if new_layer_size is None:
            new_layer_size = source.hidden_sizes[-1]

        last_size = source.hidden_sizes[-1]
        if new_layer_size > last_size:
            raise ValueError(
                f"new_layer_size ({new_layer_size}) must be <= the source's last "
                f"hidden size ({last_size}) for near-identity initialization."
            )

        new_hidden_sizes = source.hidden_sizes + [new_layer_size]

        expanded = cls(
            hidden_sizes=new_hidden_sizes,
            input_size=source.input_size,
            activation=source.activation,
            encoder_name=source.encoder_name,
        )

        with torch.no_grad():
            # Copy all existing hidden layers unchanged
            for src_layer, dst_layer in zip(source.hidden_layers, expanded.hidden_layers[:-1]):
                dst_layer.weight.copy_(src_layer.weight)
                dst_layer.bias.copy_(src_layer.bias)

            # New last hidden layer: near-identity init, bias = 0
            new_layer = expanded.hidden_layers[-1]
            if new_layer_size == last_size:
                # Square case: I + ε·randn
                nn.init.eye_(new_layer.weight)
            else:
                # Non-square case: truncated identity (first new_layer_size rows
                # of the last_size×last_size identity) + ε·randn
                nn.init.zeros_(new_layer.weight)
                for i in range(new_layer_size):
                    new_layer.weight[i, i] = 1.0
            new_layer.weight.add_(torch.randn_like(new_layer.weight) * eps)
            nn.init.zeros_(new_layer.bias)

            # Output layer: copy first new_layer_size columns from source output weights
            # (truncated identity maps unit i → unit i, so output weights are the same
            # for the first new_layer_size units)
            expanded.fc_output.weight[:, :new_layer_size] = source.fc_output.weight[:, :new_layer_size]
            expanded.fc_output.bias.copy_(source.fc_output.bias)

        expanded._expanded_from = f"{source.hidden_sizes} -> {new_hidden_sizes} (depth)"
        return expanded

