#!/usr/bin/env python3
"""
export_weights.py -- Export a TDNetwork .pt model to a flat binary format
readable by the C inference engine (c_engine/nn_eval.c).

Binary format (.bin):
    4 bytes:  magic "BGNN"
    int32:    num_hidden_layers
    int32:    input_size
    int32:    activation (0=relu, 1=sigmoid, 2=tanh, 3=leaky_relu, 4=hardsigmoid)
    int32:    output_mode (0=probability [sigmoid], 1=equity [linear],
                           2=prob5 [5 sigmoid outputs -> money equity])
    int32[]:  hidden_sizes (num_hidden_layers entries)

    Then for each layer (hidden_0 ... hidden_N-1, output):
        float32[out * in]:  weight matrix (row-major)
        float32[out]:       bias vector

    The output layer has 1 neuron except for prob5 (output_mode 2), which
    has 5. The C reader (nn_eval.c) applies sigmoid + the nested-event
    clamp + equity reduction for prob5.

Usage:
    python export_weights.py model.pt model.bin
    python export_weights.py model.pt               # writes model.bin
"""

import struct
import sys
import numpy as np
import os

import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ProbNetwork, load_model


ACTIVATION_MAP = {
    "relu": 0,
    "sigmoid": 1,
    "tanh": 2,
    "leaky_relu": 3,
    "hardsigmoid": 4,
}

OUTPUT_MODE_MAP = {
    "probability": 0,
    "equity": 1,
}

OUTPUT_MODE_PROB5 = 2


def export_model(model_path: str, output_path: str):
    """Load a .pt model (TDNetwork or prob5 ProbNetwork) and write binary weights."""
    model = load_model(model_path)
    model.eval()

    act_id = ACTIVATION_MAP.get(model.activation)
    if act_id is None:
        raise ValueError(f"Unsupported activation '{model.activation}' for C export. "
                         f"Supported: {list(ACTIVATION_MAP.keys())}")

    if isinstance(model, ProbNetwork):
        # prob5: hidden linears live in `trunk` (Linear/activation pairs),
        # the 5-output layer is `head`. C applies sigmoid to all 5.
        hidden_layers = [m for m in model.trunk if isinstance(m, nn.Linear)]
        output_layer = model.head
        output_mode_id = OUTPUT_MODE_PROB5
        n_out = output_layer.out_features
        mode_label = "prob5"
    else:
        hidden_layers = list(model.hidden_layers)
        output_layer = model.fc_output
        output_mode_id = OUTPUT_MODE_MAP.get(model.output_mode, 0)
        n_out = 1
        mode_label = model.output_mode

    with open(output_path, "wb") as f:
        # Header
        f.write(b"BGNN")
        f.write(struct.pack("<i", len(model.hidden_sizes)))
        f.write(struct.pack("<i", model.input_size))
        f.write(struct.pack("<i", act_id))
        f.write(struct.pack("<i", output_mode_id))
        for hs in model.hidden_sizes:
            f.write(struct.pack("<i", hs))

        # Hidden layers, then output layer
        total_params = 0
        for layer in hidden_layers + [output_layer]:
            w = layer.weight.detach().cpu().numpy()  # shape [out, in]
            b = layer.bias.detach().cpu().numpy()     # shape [out]
            f.write(w.astype(np.float32).tobytes())
            f.write(b.astype(np.float32).tobytes())
            total_params += w.size + b.size

    file_size = os.path.getsize(output_path)
    print(f"Exported: {output_path}")
    print(f"  Architecture: [{model.input_size}] -> {model.hidden_sizes} -> [{n_out}]")
    print(f"  Activation: {model.activation}")
    print(f"  Output mode: {mode_label}")
    print(f"  Parameters: {total_params:,}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")


def verify_export(model_path: str, bin_path: str):
    """Quick sanity check: print Python output on a test input."""
    import torch
    model = load_model(model_path)
    model.eval()

    np.random.seed(42)
    test_input = np.random.randn(model.input_size).astype(np.float32)
    with torch.no_grad():
        out = model(torch.tensor(test_input)[None, :])
        if isinstance(model, ProbNetwork):
            from model import prob5_postprocess, prob5_to_equity
            if model.raw_logits:
                out = torch.sigmoid(out)
            py_out = float(prob5_to_equity(prob5_postprocess(out))[0])
            print(f"  Verification: Python prob5 equity = {py_out:.8f}")
        else:
            print(f"  Verification: Python output = {float(out.reshape(-1)[0]):.8f}")
    print("  (Run C inference on same input to compare)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.pt> [output.bin]")
        sys.exit(1)

    model_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = os.path.splitext(model_path)[0] + ".bin"

    export_model(model_path, output_path)
    verify_export(model_path, output_path)
