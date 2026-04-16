#!/usr/bin/env python3
"""
describe_model.py -- Print architecture and training info for a saved model.

Usage:
    python3 describe_model.py model.pt
    python3 describe_model.py model1.pt model2.pt ...
"""

import argparse
import os
import sys

import torch


def describe(path: str) -> None:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)

    hidden = checkpoint.get("hidden_sizes", [])
    input_size = checkpoint.get("input_size", 196)
    activation = checkpoint.get("activation", "relu")
    encoder = checkpoint.get("encoder_name", "perspective196")
    output_mode = checkpoint.get("output_mode", "probability")
    train_params = checkpoint.get("train_params", {})

    # Count parameters from state_dict
    state_dict = checkpoint.get("state_dict", {})
    total_params = sum(v.numel() for v in state_dict.values())
    trainable_params = total_params  # all params are trainable in TDNetwork

    # Layer-by-layer breakdown
    layers = []
    prev_size = input_size
    for i, h in enumerate(hidden):
        w_key = f"hidden_layers.{i}.weight"
        b_key = f"hidden_layers.{i}.bias"
        w_params = state_dict[w_key].numel() if w_key in state_dict else 0
        b_params = state_dict[b_key].numel() if b_key in state_dict else 0
        layers.append({
            "name": f"hidden_{i}",
            "type": f"Linear({prev_size} → {h})",
            "activation": activation,
            "params": w_params + b_params,
        })
        prev_size = h

    # Output layer
    out_w = state_dict.get("fc_output.weight")
    out_b = state_dict.get("fc_output.bias")
    out_params = (out_w.numel() if out_w is not None else 0) + \
                 (out_b.numel() if out_b is not None else 0)
    out_act = "sigmoid" if output_mode == "probability" else "linear"
    layers.append({
        "name": "output",
        "type": f"Linear({prev_size} → 1)",
        "activation": out_act,
        "params": out_params,
    })

    # Print
    print(f"{'=' * 60}")
    print(f"  Model: {path}")
    print(f"  File size: {size_mb:.2f} MB")
    print(f"{'=' * 60}")
    print()

    print("Architecture:")
    print(f"  Input:       {input_size} features ({encoder})")
    print(f"  Hidden:      {hidden}")
    print(f"  Activation:  {activation}")
    print(f"  Output mode: {output_mode} ({'sigmoid [0,1]' if output_mode == 'probability' else 'linear (unbounded)'})")
    print(f"  Parameters:  {total_params:,}")
    print()

    print("Layer breakdown:")
    print(f"  {'Layer':<12s} {'Shape':<25s} {'Activation':<12s} {'Params':>10s}")
    print(f"  {'-'*12} {'-'*25} {'-'*12} {'-'*10}")
    for layer in layers:
        print(f"  {layer['name']:<12s} {layer['type']:<25s} "
              f"{layer['activation']:<12s} {layer['params']:>10,}")
    print(f"  {'':12s} {'':25s} {'TOTAL':>12s} {total_params:>10,}")
    print()

    if train_params:
        print("Training history:")
        for k, v in sorted(train_params.items()):
            print(f"  {k}: {v}")
        print()
    else:
        print("Training history: (none saved)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Describe a saved TDNetwork model (.pt file)"
    )
    parser.add_argument("models", nargs="+", help="Path(s) to .pt model files")
    args = parser.parse_args()

    for i, path in enumerate(args.models):
        if i > 0:
            print()
        describe(path)


if __name__ == "__main__":
    main()
