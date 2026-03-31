#!/usr/bin/env python3
"""
test_nn.py -- Python counterpart to test_nn.c.

Evaluates the same positions and prints values for comparison.

Usage:
    PYTHONPATH=/workspace/bg-engine-dev python3 test_nn.py best_models/td_batch_relu_512_512_256_128_1ply_vi_final.pt
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, "/workspace/bg-engine-dev")

from model import TDNetwork
from backgammon_engine import BoardState, WHITE, BLACK, get_legal_plays, switch_turn
from encoding import get_encoder


def print_features(f, label):
    print(f"  {label} first 10: [{', '.join(f'{x:.4f}' for x in f[:10])}]")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.pt>")
        sys.exit(1)

    model = TDNetwork.load(sys.argv[1])
    model.eval()
    encoder = get_encoder(model.encoder_name)

    print(f"Model: {model.input_size} inputs, {len(model.hidden_sizes)} hidden layers "
          f"{model.hidden_sizes}, activation={model.activation}\n")

    # Test 1: Initial position, WHITE to move
    state = BoardState.initial()
    features = encoder.encode(state)
    with torch.no_grad():
        val = model(torch.tensor(features, dtype=torch.float32)).item()
    print("Test 1: Initial position (WHITE to move)")
    print_features(features, "features")
    print(f"  forward = {val:.8f}\n")

    # Test 2: Initial position, BLACK to move
    state = BoardState.initial()
    state = switch_turn(state)
    features = encoder.encode(state)
    with torch.no_grad():
        val = model(torch.tensor(features, dtype=torch.float32)).item()
    print("Test 2: Initial position (BLACK to move)")
    print_features(features, "features")
    print(f"  forward = {val:.8f}\n")

    # Test 3: After WHITE plays opening 3-1 (first legal play)
    state = BoardState.initial()
    plays = get_legal_plays(state, (3, 1))
    if plays:
        play, next_state = plays[0]
        switched = switch_turn(next_state)
        features = encoder.encode(switched)
        with torch.no_grad():
            val = model(torch.tensor(features, dtype=torch.float32)).item()
        print("Test 3: After WHITE's first legal 3-1 play (BLACK to move)")
        print(f"  Play: {play}")
        print_features(features, "features")
        print(f"  forward = {val:.8f}\n")

    # Test 4: After WHITE plays opening 6-4 (first legal play)
    state = BoardState.initial()
    plays = get_legal_plays(state, (6, 4))
    if plays:
        play, next_state = plays[0]
        switched = switch_turn(next_state)
        features = encoder.encode(switched)
        with torch.no_grad():
            val = model(torch.tensor(features, dtype=torch.float32)).item()
        print("Test 4: After WHITE's first legal 6-4 play (BLACK to move)")
        print(f"  Play: {play}")
        print_features(features, "features")
        print(f"  forward = {val:.8f}\n")


if __name__ == "__main__":
    main()
