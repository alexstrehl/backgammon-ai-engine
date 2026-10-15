#!/usr/bin/env python3
"""
export_weights.py -- Export a .pt model (TDNetwork or prob5 ProbNetwork)
to the flat BGNN binary read by c_inference/nn_eval.c.

Binary format (.bin), little-endian, format 2 (what this module writes):
    4 bytes:  magic "BGNV"
    int32:    format version (2)
    64 bytes: encoder name: the checkpoint's `encoder_name`
              (encoding.get_encoder name, e.g. "cubeful_perspective196"),
              ASCII, 1-63 characters, NUL padded
    int32:    num_hidden_layers
    int32:    input_size
    int32:    activation (0=relu, 1=sigmoid, 2=tanh, 3=leaky_relu, 4=hardsigmoid)
    int32:    output_mode (0=probability [sigmoid], 1=equity [linear],
                           2=prob5 [5 sigmoid outputs -> money equity])
    int32[]:  hidden_sizes (num_hidden_layers entries)

    Then for each layer (hidden_0 ... hidden_N-1, output):
        float32[out * in]:  weight matrix (row-major)
        float32[out]:       bias vector

    The output layer has 1 neuron, or 5 for prob5 (the C side applies
    sigmoid, the nested-event clamp and the equity reduction).

Format 1 (LEGACY): magic "BGNN", then the same fields from
num_hidden_layers on (no version, no name). nn_eval.c reads it; the C
API maps its input width to the encoder for the known layouts
(c_api/puretd.c). A format-1-only reader rejects a format-2 file at the
magic. A later format bumps the version; nn_eval.c rejects a version it
does not know. The encoder name is written as the net carries it,
unchecked against the encoding.py registry; bgc_open fails naming an
encoder the C API's table does not know.

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

MAGIC = b"BGNV"                 # format 2 (FORMAT_VERSION)
LEGACY_MAGIC = b"BGNN"          # format 1: no encoder name
FORMAT_VERSION = 2
ENCODER_NAME_MAX = 64           # bytes of the name field, NUL padding included


def _layers(model):
    """(hidden linears, output linear, output_mode id, outputs, mode label)."""
    if isinstance(model, ProbNetwork):
        # prob5: hidden linears are in `trunk`, the 5-output layer is `head`.
        hidden_layers = [m for m in model.trunk if isinstance(m, nn.Linear)]
        output_layer = model.head
        return (hidden_layers, output_layer, OUTPUT_MODE_PROB5,
                output_layer.out_features, "prob5")
    return (list(model.hidden_layers), model.fc_output,
            OUTPUT_MODE_MAP.get(model.output_mode, 0), 1, model.output_mode)


def encoder_name_field(name) -> bytes:
    """The 64-byte header field of encoder `name` (ValueError if it does
    not fit: 1-63 printable non-space ASCII characters)."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"BGNN export needs the net's encoder_name (got {name!r})")
    raw = name.encode("utf-8")
    if len(raw) >= ENCODER_NAME_MAX or any(c <= 0x20 or c > 0x7E for c in raw):
        raise ValueError(f"encoder name {name!r} does not fit the BGNN header "
                         f"(1-{ENCODER_NAME_MAX - 1} printable ASCII characters)")
    return raw.ljust(ENCODER_NAME_MAX, b"\0")


def _write(model, output_path: str) -> int:
    """Write `model` as a format-2 BGNN file; returns the parameter count."""
    act_id = ACTIVATION_MAP.get(model.activation)
    if act_id is None:
        raise ValueError(f"Unsupported activation '{model.activation}' for C export. "
                         f"Supported: {list(ACTIVATION_MAP.keys())}")
    hidden_layers, output_layer, output_mode_id, _, _ = _layers(model)
    name = encoder_name_field(getattr(model, "encoder_name", None))
    total_params = 0
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", FORMAT_VERSION))
        f.write(name)
        f.write(struct.pack("<i", len(model.hidden_sizes)))
        f.write(struct.pack("<i", model.input_size))
        f.write(struct.pack("<i", act_id))
        f.write(struct.pack("<i", output_mode_id))
        for hs in model.hidden_sizes:
            f.write(struct.pack("<i", hs))
        for layer in hidden_layers + [output_layer]:
            w = layer.weight.detach().cpu().numpy().astype(np.float32)  # [out, in]
            b = layer.bias.detach().cpu().numpy().astype(np.float32)    # [out]
            f.write(w.tobytes())
            f.write(b.tobytes())
            total_params += w.size + b.size
    return total_params


def read_header(path: str) -> dict:
    """The header of a BGNN file, either format: format_version (1 or 2),
    encoder_name (None in a format-1 file), num_hidden, input_size,
    activation, output_mode, hidden_sizes, data_offset (where the weights
    start). ValueError on a file that is not BGNN or a version this
    module does not know."""
    with open(path, "rb") as f:
        head = f.read(4 + 4 + ENCODER_NAME_MAX + 16)
        magic = head[:4]
        if magic == LEGACY_MAGIC:
            version, name, off = 1, None, 4
        elif magic == MAGIC:
            (version,) = struct.unpack("<i", head[4:8])
            if version != FORMAT_VERSION:
                raise ValueError(f"{path}: BGNV format version {version}, "
                                 f"this module knows {FORMAT_VERSION}")
            field = head[8:8 + ENCODER_NAME_MAX]
            name = field.split(b"\0", 1)[0].decode("ascii")
            if encoder_name_field(name) != field:
                raise ValueError(f"{path}: bad encoder name field {field!r}")
            off = 8 + ENCODER_NAME_MAX
        else:
            raise ValueError(f"{path}: not a BGNN file (magic {magic!r})")
        nh, n_in, act, mode = struct.unpack("<iiii", head[off:off + 16])
        f.seek(off + 16)
        hidden = list(struct.unpack(f"<{nh}i", f.read(4 * nh)))
    return {"format_version": version, "encoder_name": name, "num_hidden": nh,
            "input_size": n_in, "activation": act, "output_mode": mode,
            "hidden_sizes": hidden, "data_offset": off + 16 + 4 * nh}


def export_model(model_path: str, output_path: str):
    """Load a .pt model and write its BGNN binary."""
    model = load_model(model_path)
    model.eval()
    total_params = _write(model, output_path)
    _, _, _, n_out, mode_label = _layers(model)

    file_size = os.path.getsize(output_path)
    print(f"Exported: {output_path}")
    print(f"  Architecture: [{model.input_size}] -> {model.hidden_sizes} -> [{n_out}]")
    print(f"  Encoder: {model.encoder_name}")
    print(f"  Activation: {model.activation}")
    print(f"  Output mode: {mode_label}")
    print(f"  Parameters: {total_params:,}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")


def verify_export(model_path: str, bin_path: str):
    """Print the Python output on a fixed random input, for comparison
    with the C side."""
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


def export_live(model, output_path: str):
    """Export an in-memory model to BGNN binary (same format as
    `export_model`, no .pt round-trip). Used to refresh the C engine's
    weights between training rounds."""
    _write(model, output_path)


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
