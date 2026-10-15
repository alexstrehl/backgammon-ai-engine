"""
model.py -- Neural networks for backgammon position evaluation.

Two feedforward nets over an encoder's feature vector (see encoding.py),
with configurable hidden sizes and activation:

  * TDNetwork: one output, either P(on-roll player wins) (sigmoid) or
    money equity (linear), chosen by ``output_mode``.
  * ProbNetwork: five outputs, P(win/wg/wbg/lg/lbg), for cubeless money
    ("prob5"); ``prob5_to_equity`` reduces them to equity.

Both save/load their own checkpoints (``load_model`` dispatches on the
saved ``model_type``) and support function-preserving width and depth
expansion.
"""

import torch
import torch.nn as nn
from typing import List
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


# Depth expansion appends a hidden layer initialised as ``I + eps*randn``.
# `eps` is REQUIRED (no default): choose it per net.
# suggested_depth_expand_eps() is a starting point: 0.001 * sqrt(256).
DEPTH_EXPAND_EPS_SCALE = 0.016


def suggested_depth_expand_eps(new_layer_size: int) -> float:
    """Starting point only: holds eps*sqrt(n) at the (n=256, eps=0.001)
    anchor. Measure before relying on it."""
    return DEPTH_EXPAND_EPS_SCALE / (float(new_layer_size) ** 0.5)


class TDNetwork(nn.Module):
    """Single-output value network.

    output_mode:
      - "probability": sigmoid -> P(on-roll player wins) in [0, 1]
      - "equity": linear -> expected payoff in [-3, +3]
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
        """Forward pass. *x* is one feature vector or a batch."""
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
        """Load a saved model (to CPU by default, so GPU-saved files work).
        `weights_only=True`: checkpoints hold only tensors and builtins.
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)

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
        """Wider network initialised from a narrower one (Net2Net-style,
        https://arxiv.org/abs/1511.05641): same depth, no layer shrinks.
        The function is unchanged at init: weights from new units into
        old units and into the output are zeroed.

        Args:
            new_hidden_sizes: same layer count as source, each >= old.
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
                # Copy the old block. New units' rows keep their random init.
                new_layer.weight[:old_out, :old_in] = old_layer.weight
                new_layer.bias[:old_out] = old_layer.bias
                if i > 0:
                    prev_old = old_sizes[i - 1]
                    # Old units must not see the previous layer's new units.
                    new_layer.weight[:old_out, prev_old:] = 0.0

            # Output layer: copy old weights, zero the new units' columns.
            old_out_w = source.fc_output.weight
            expanded.fc_output.weight[:, :old_sizes[-1]] = old_out_w
            expanded.fc_output.weight[:, old_sizes[-1]:] = 0.0
            expanded.fc_output.bias[:] = source.fc_output.bias

        expanded._expanded_from = f"{old_sizes} -> {new_hidden_sizes}"
        return expanded

    @classmethod
    def depth_expand(cls, source: "TDNetwork", new_layer_size: int = None, *, eps: float) -> "TDNetwork":
        """Deeper network: one hidden layer appended before the output,
        initialised as ``I + eps*randn``. With ReLU the function is
        preserved at init up to the eps term, which breaks weight
        symmetry.

        Args:
            new_layer_size: must equal the last hidden size (square
                identity). Defaults to ``source.hidden_sizes[-1]``.
            eps: REQUIRED, keyword-only; see DEPTH_EXPAND_EPS_SCALE.
                0.0 = exact copy but symmetric.
        """
        if new_layer_size is None:
            new_layer_size = source.hidden_sizes[-1]

        last_size = source.hidden_sizes[-1]
        if new_layer_size != last_size:
            raise ValueError(
                f"new_layer_size ({new_layer_size}) must EQUAL the source's last "
                f"hidden size ({last_size}): the appended layer is a near-identity "
                f"map, which only preserves the function when square. A smaller "
                f"layer drops hidden components; a larger one is unsupported."
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
            # Copy the existing hidden layers unchanged.
            for src_layer, dst_layer in zip(source.hidden_layers, expanded.hidden_layers[:-1]):
                dst_layer.weight.copy_(src_layer.weight)
                dst_layer.bias.copy_(src_layer.bias)

            # New last hidden layer: identity + eps noise, bias 0.
            new_layer = expanded.hidden_layers[-1]
            if new_layer_size == last_size:
                nn.init.eye_(new_layer.weight)
            else:
                # Unreachable (sizes are checked equal above): truncated identity.
                nn.init.zeros_(new_layer.weight)
                for i in range(new_layer_size):
                    new_layer.weight[i, i] = 1.0
            new_layer.weight.add_(torch.randn_like(new_layer.weight) * eps)
            nn.init.zeros_(new_layer.bias)

            # Output layer: unit i still maps to unit i, so the weights carry over.
            expanded.fc_output.weight[:, :new_layer_size] = source.fc_output.weight[:, :new_layer_size]
            expanded.fc_output.bias.copy_(source.fc_output.bias)

        expanded._expanded_from = f"{source.hidden_sizes} -> {new_hidden_sizes} (depth)"
        return expanded


# ── Prob5 network (5-output cubeless money) ────────────────────────────


# Money-equity weights for the 5 outputs [P(win), P(wg), P(wbg), P(lg), P(lbg)]:
#   eq = 2*P(win) + P(wg) + P(wbg) - P(lg) - P(lbg) - 1
_PROB5_EQUITY_WEIGHTS = torch.tensor([2.0, 1.0, 1.0, -1.0, -1.0])


def prob5_to_equity(probs: torch.Tensor) -> torch.Tensor:
    """Money equity from a ``(..., 5)`` tensor of P(win, wg, wbg, lg, lbg)."""
    w = _PROB5_EQUITY_WEIGHTS.to(device=probs.device, dtype=probs.dtype)
    return (probs * w).sum(dim=-1) - 1.0


def prob5_postprocess(probs: torch.Tensor) -> torch.Tensor:
    """Clamp a ``(..., 5)`` prob tensor onto the nested-event constraints
    (targets always satisfy them; the five sigmoids need not):
      P(wg)  <= P(win),       P(wbg) <= P(wg)
      P(lg)  <= 1 - P(win),   P(lbg) <= P(lg)
    Returns a new tensor.
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

    Outputs (after sigmoid, on-roll player's view): P(win), P(win
    gammon), P(win backgammon), P(lose gammon), P(lose backgammon).
    For equity / move selection apply :func:`prob5_postprocess` then
    :func:`prob5_to_equity`.

    ``raw_logits=True`` drops the sigmoid (for BCEWithLogits training).
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
        # Lets TDAgent / play_models dispatch without isinstance.
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
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
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

    # ── width / depth expansion (same semantics as TDNetwork) ───────

    @classmethod
    def width_expand(
        cls, source: "ProbNetwork", new_hidden_sizes: List[int],
    ) -> "ProbNetwork":
        """Function-preserving width expansion; same depth."""
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
                # Copy the old block. New units' rows keep their random
                # init. Old units must not see the previous layer's new
                # units.
                dst_w[:old_out, :old_in] = src_w
                dst_w[:old_out, old_in:] = 0.0
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
        *,
        eps: float,
    ) -> "ProbNetwork":
        """Append one near-identity hidden layer before the head. See
        TDNetwork.depth_expand."""
        last_size = source.hidden_sizes[-1]
        if new_layer_size is None:
            new_layer_size = last_size
        if new_layer_size != last_size:
            raise ValueError(
                f"new_layer_size ({new_layer_size}) must EQUAL last hidden size "
                f"({last_size}): the appended layer is a near-identity map, which "
                f"only preserves the function when square. A smaller layer drops "
                f"hidden components; a larger one is unsupported."
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
            # New layer: identity + eps noise, bias 0.
            new_idx = len(source.hidden_sizes) * 2
            new_w = expanded.get_parameter(f"trunk.{new_idx}.weight")
            new_b = expanded.get_parameter(f"trunk.{new_idx}.bias")
            new_w.zero_()
            for i in range(min(new_layer_size, last_size)):
                new_w[i, i] = 1.0
            new_w.add_(torch.randn_like(new_w) * eps)
            new_b.zero_()
            # Head: unit i still maps to unit i, so the weights carry over.
            expanded.get_parameter("head.weight").copy_(
                src_sd["head.weight"][:, :new_layer_size]
            )
            expanded.get_parameter("head.bias").copy_(src_sd["head.bias"])
        expanded._expanded_from = (
            f"{source.hidden_sizes} -> {new_hidden} (depth)"
        )
        return expanded


def load_model(path: str, map_location: str = "cpu"):
    """Load a checkpoint as :class:`TDNetwork` or :class:`ProbNetwork`
    by its saved ``model_type`` (absent = TDNetwork)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if ckpt.get("model_type") == "prob5":
        return ProbNetwork.load(path, map_location=map_location)
    # A prob5-shaped checkpoint without model_type must not load as a
    # scalar TDNetwork.
    sd = ckpt.get("state_dict", {})
    if "head.weight" in sd or ckpt.get("output_mode") == "prob5":
        raise ValueError(
            f"{path}: checkpoint has a prob5 head ('head.weight' in state_dict) "
            "but model_type != 'prob5'; refusing to load as scalar TDNetwork. "
            "Re-save with model_type='prob5'."
        )
    return TDNetwork.load(path, map_location=map_location)

