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

    is_prob5 = checkpoint.get("model_type") == "prob5"
    hidden = checkpoint.get("hidden_sizes", [])
    input_size = checkpoint.get("input_size", 196)
    activation = checkpoint.get("activation", "relu")
    encoder = checkpoint.get("encoder_name", "perspective196")
    train_params = checkpoint.get("train_params", {})
    if is_prob5:
        n_out = 5
        output_mode = "raw logits" if checkpoint.get("raw_logits") else "probability"
    else:
        n_out = 1
        output_mode = checkpoint.get("output_mode", "probability")

    # Count parameters from state_dict
    state_dict = checkpoint.get("state_dict", {})
    total_params = sum(v.numel() for v in state_dict.values())

    # Layer-by-layer breakdown. prob5 nets store linears as trunk.{2i}/head;
    # scalar nets as hidden_layers.{i}/fc_output.
    layers = []
    prev_size = input_size
    for i, h in enumerate(hidden):
        prefix = f"trunk.{2 * i}" if is_prob5 else f"hidden_layers.{i}"
        w = state_dict.get(f"{prefix}.weight")
        b = state_dict.get(f"{prefix}.bias")
        layers.append({
            "name": f"hidden_{i}",
            "type": f"Linear({prev_size} → {h})",
            "activation": activation,
            "params": (w.numel() if w is not None else 0) + (b.numel() if b is not None else 0),
        })
        prev_size = h

    # Output layer
    out_prefix = "head" if is_prob5 else "fc_output"
    out_w = state_dict.get(f"{out_prefix}.weight")
    out_b = state_dict.get(f"{out_prefix}.bias")
    out_params = (out_w.numel() if out_w is not None else 0) + \
                 (out_b.numel() if out_b is not None else 0)
    if is_prob5:
        # prob5 always applies sigmoid + nested-event clamp at inference,
        # regardless of raw_logits (which only moves the sigmoid out of forward()).
        out_act = "5x sigmoid + nested-event clamp (inference)"
    elif output_mode == "raw logits":
        out_act = "linear"
    elif output_mode == "probability":
        out_act = "sigmoid"
    else:
        out_act = "linear"
    layers.append({
        "name": "output",
        "type": f"Linear({prev_size} → {n_out})",
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
    if is_prob5:
        print(f"  Output mode: prob5 ({n_out} outputs: P(win),P(wg),P(wbg),"
              f"P(lg),P(lbg); sigmoid + nested-event clamp at inference"
              f"{'; stored as raw logits' if checkpoint.get('raw_logits') else ''})")
    else:
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
