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

    Supports two output modes:
      - "probability": sigmoid → P(on-roll player wins) ∈ [0, 1]
      - "equity": linear → expected payoff ∈ [-3, +3]

    Supports multiple hidden layers and configurable activation functions.
    """

    def __init__(
        self,
        hidden_sizes: List[int] = None,
        input_size: int = None,
        activation: str = "relu",
        encoder_name: str = "perspective196",
        output_mode: str = "probability",
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.output_mode = output_mode
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

        self.fc_output = nn.Linear(prev_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.  *x* can be a single 196-vector or a batch."""
        if self._n_hidden == 1:
            h = self._activation_fn(self.hidden_layers[0](x))
        else:
            h = x
            for layer in self.hidden_layers:
                h = self._activation_fn(layer(h))
        raw = self.fc_output(h).squeeze(-1)
        if self.output_mode == "equity":
            return raw
        return torch.sigmoid(raw)

    # ── Save / Load ──────────────────────────────────────────────────

    def save(self, path: str, train_params: dict = None):
        """Save model weights, architecture info, and optional training params."""
        data = {
            "hidden_sizes": self.hidden_sizes,
            "input_size": self.input_size,
            "activation": self.activation,
            "encoder_name": self.encoder_name,
            "output_mode": self.output_mode,
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
            output_mode=checkpoint.get("output_mode", "probability"),
        )

        model.load_state_dict(checkpoint["state_dict"])
        model._resumed_from = path
        model._train_params = checkpoint.get("train_params", {})
        return model

    @classmethod
    def width_expand(cls, source: "TDNetwork", new_hidden_sizes: List[int]) -> "TDNetwork":
        """Create a wider network initialized from a narrower one.

        Inspired by Net2Net (Chen, Goodfellow & Shlens, 2015;
        https://arxiv.org/abs/1511.05641). Same number of hidden layers;
        each layer can have more (not fewer) units, so the old model is
        embedded as a subgraph. Weights corresponding to the embedded
        old model are copied exactly. Weights that are inputs to new
        units get random Kaiming initialization (PyTorch default). All
        other new weights (new units to old units, or final output
        weights of new units) are zeroed. With no training the new
        model behaves identically to the old.

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
            output_mode=source.output_mode,
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
            output_mode=source.output_mode,
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


# ── Prob5 network (5-output cubeless money) ────────────────────────────


# Money-equity weights for the 5 outputs [P(win), P(wg), P(wbg), P(lg), P(lbg)]:
#   eq = 2*P(win) + P(wg) + P(wbg) - P(lg) - P(lbg) - 1
_PROB5_EQUITY_WEIGHTS = torch.tensor([2.0, 1.0, 1.0, -1.0, -1.0])


def prob5_to_equity(probs: torch.Tensor) -> torch.Tensor:
    """Convert 5-output probability tensor to per-position money equity.

    Inputs are in canonical order: P(win), P(wg), P(wbg), P(lg), P(lbg).
    Accepts any leading shape; the last dim must be 5.
    """
    w = _PROB5_EQUITY_WEIGHTS.to(device=probs.device, dtype=probs.dtype)
    return (probs * w).sum(dim=-1) - 1.0


def prob5_postprocess(probs: torch.Tensor) -> torch.Tensor:
    """Enforce nested-event inequalities on a ``(..., 5)`` prob tensor.

    Post-hoc clamp of 5 independent sigmoids onto the valid joint:
      P(wg)  <= P(win),       P(wbg) <= P(wg)
      P(lg)  <= 1 - P(win),   P(lbg) <= P(lg)

    Targets always satisfy these; the model's outputs are nudged to as
    well at inference time. Returns a fresh tensor — caller's input is
    untouched.
    """
    p = probs.clone()
    p[..., 1] = torch.minimum(p[..., 0], p[..., 1])
    lose = 1.0 - p[..., 0]
    p[..., 3] = torch.minimum(lose, p[..., 3])
    p[..., 2] = torch.minimum(p[..., 1], p[..., 2])
    p[..., 4] = torch.minimum(p[..., 3], p[..., 4])
    return p


class ProbNetwork(nn.Module):
    """5-output probability network for cubeless money games.

    Outputs (after sigmoid, from on-roll player's perspective):
      0: P(win)
      1: P(win gammon)
      2: P(win backgammon)
      3: P(lose gammon)
      4: P(lose backgammon)

    For scalar equity / move selection, reduce via :func:`prob5_to_equity`
    (apply :func:`prob5_postprocess` first at inference time).

    ``raw_logits=True`` drops the final sigmoid; pair with BCEWithLogits
    or MSE for stability when outputs saturate near 0/1.
    """

    NUM_OUTPUTS = 5

    def __init__(
        self,
        hidden_sizes: List[int] = None,
        input_size: int = None,
        activation: str = "relu",
        encoder_name: str = "perspective196",
        raw_logits: bool = False,
    ):
        super().__init__()
        if input_size is None:
            from encoding import get_encoder
            input_size = get_encoder(encoder_name).num_features
        if hidden_sizes is None:
            hidden_sizes = [80]

        self.hidden_sizes = list(hidden_sizes)
        self.input_size = input_size
        self.activation = activation.lower()
        self.encoder_name = encoder_name
        self.raw_logits = bool(raw_logits)
        # Set so TDAgent / play_models can dispatch without isinstance.
        self.output_mode = "prob5"

        act_cls = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }[self.activation]

        layers: List[nn.Module] = []
        prev = input_size
        for sz in self.hidden_sizes:
            layers.append(nn.Linear(prev, sz))
            layers.append(act_cls())
            prev = sz
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, self.NUM_OUTPUTS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.trunk(x))
        if self.raw_logits:
            return raw
        return torch.sigmoid(raw)

    # ── save / load ──────────────────────────────────────────────────

    def save(self, path: str, train_params: dict = None):
        data = {
            "model_type": "prob5",
            "hidden_sizes": self.hidden_sizes,
            "input_size": self.input_size,
            "activation": self.activation,
            "encoder_name": self.encoder_name,
            "raw_logits": self.raw_logits,
            "state_dict": self.state_dict(),
        }
        if train_params:
            data["train_params"] = train_params
        torch.save(data, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "ProbNetwork":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        mt = ckpt.get("model_type")
        if mt is not None and mt != "prob5":
            raise ValueError(
                f"ProbNetwork.load: {path} has model_type={mt!r}, not 'prob5'."
            )
        net = cls(
            hidden_sizes=ckpt["hidden_sizes"],
            input_size=ckpt.get("input_size", 196),
            activation=ckpt.get("activation", "relu"),
            encoder_name=ckpt.get("encoder_name", "perspective196"),
            raw_logits=ckpt.get("raw_logits", False),
        )
        net.load_state_dict(ckpt["state_dict"])
        net._resumed_from = path
        net._train_params = ckpt.get("train_params", {})
        return net

    # ── width / depth expansion (mirror TDNetwork semantics) ────────

    @classmethod
    def width_expand(
        cls, source: "ProbNetwork", new_hidden_sizes: List[int],
    ) -> "ProbNetwork":
        """Function-preserving Net2Net-style width expansion. Same depth."""
        if len(new_hidden_sizes) != len(source.hidden_sizes):
            raise ValueError(
                f"Layer count mismatch: source has "
                f"{len(source.hidden_sizes)}, target has "
                f"{len(new_hidden_sizes)}"
            )
        for old_s, new_s in zip(source.hidden_sizes, new_hidden_sizes):
            if new_s < old_s:
                raise ValueError(f"Cannot shrink: {old_s} -> {new_s}")

        expanded = cls(
            hidden_sizes=new_hidden_sizes,
            input_size=source.input_size,
            activation=source.activation,
            encoder_name=source.encoder_name,
            raw_logits=source.raw_logits,
        )
        src_sd = source.state_dict()
        with torch.no_grad():
            old_sizes = source.hidden_sizes
            for i in range(len(old_sizes)):
                wkey = f"trunk.{2 * i}.weight"
                bkey = f"trunk.{2 * i}.bias"
                dst_w = expanded.get_parameter(wkey)
                dst_b = expanded.get_parameter(bkey)
                src_w = src_sd[wkey]
                src_b = src_sd[bkey]
                old_out, old_in = src_w.shape
                # New-column contributions from expanded prior layer: 0
                # (preserves function exactly at init).
                dst_w.zero_()
                dst_w[:old_out, :old_in] = src_w
                dst_b.zero_()
                dst_b[:old_out] = src_b
            # Output head: copy existing rows, zero new columns.
            dst_hw = expanded.get_parameter("head.weight")
            dst_hb = expanded.get_parameter("head.bias")
            dst_hw.zero_()
            dst_hw[:, : old_sizes[-1]] = src_sd["head.weight"]
            dst_hb.copy_(src_sd["head.bias"])
        expanded._expanded_from = (
            f"{source.hidden_sizes} -> {new_hidden_sizes} (width)"
        )
        return expanded

    @classmethod
    def depth_expand(
        cls,
        source: "ProbNetwork",
        new_layer_size: int = None,
        eps: float = 0.01,
    ) -> "ProbNetwork":
        """Append one near-identity hidden layer before the output head."""
        last_size = source.hidden_sizes[-1]
        if new_layer_size is None:
            new_layer_size = last_size
        if new_layer_size > last_size:
            raise ValueError(
                f"new_layer_size ({new_layer_size}) must be <= last "
                f"hidden size ({last_size}) for near-identity init."
            )

        new_hidden = source.hidden_sizes + [new_layer_size]
        expanded = cls(
            hidden_sizes=new_hidden,
            input_size=source.input_size,
            activation=source.activation,
            encoder_name=source.encoder_name,
            raw_logits=source.raw_logits,
        )
        src_sd = source.state_dict()
        with torch.no_grad():
            # Copy existing trunk layers verbatim.
            for i in range(len(source.hidden_sizes)):
                wkey = f"trunk.{2 * i}.weight"
                bkey = f"trunk.{2 * i}.bias"
                expanded.get_parameter(wkey).copy_(src_sd[wkey])
                expanded.get_parameter(bkey).copy_(src_sd[bkey])
            # New layer: truncated identity + small noise.
            new_idx = len(source.hidden_sizes) * 2
            new_w = expanded.get_parameter(f"trunk.{new_idx}.weight")
            new_b = expanded.get_parameter(f"trunk.{new_idx}.bias")
            new_w.zero_()
            for i in range(min(new_layer_size, last_size)):
                new_w[i, i] = 1.0
            new_w.add_(torch.randn_like(new_w) * eps)
            new_b.zero_()
            # Head: truncate source's columns to new_layer_size.
            expanded.get_parameter("head.weight").copy_(
                src_sd["head.weight"][:, :new_layer_size]
            )
            expanded.get_parameter("head.bias").copy_(src_sd["head.bias"])
        expanded._expanded_from = (
            f"{source.hidden_sizes} -> {new_hidden} (depth)"
        )
        return expanded


def load_model(path: str, map_location: str = "cpu"):
    """Load a checkpoint and return either :class:`TDNetwork` or
    :class:`ProbNetwork` based on the saved ``model_type`` (legacy
    files without that field load as ``TDNetwork``).
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt.get("model_type") == "prob5":
        return ProbNetwork.load(path, map_location=map_location)
    # Guard: a prob5-shaped checkpoint missing model_type must NOT silently
    # load as a scalar TDNetwork (its 5-output head would be mis-read).
    sd = ckpt.get("state_dict", {})
    if "head.weight" in sd or ckpt.get("output_mode") == "prob5":
        raise ValueError(
            f"{path}: checkpoint has a prob5 head ('head.weight' in state_dict) "
            "but model_type != 'prob5'; refusing to load as scalar TDNetwork. "
            "Re-save with model_type='prob5'."
        )
    return TDNetwork.load(path, map_location=map_location)

