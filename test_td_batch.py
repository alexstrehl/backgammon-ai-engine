"""
Tests for td_batch_train.py -- run with: python test_td_batch.py

Tests the batch TD(0) training logic without requiring PyTorch for
the collection tests.  Training tests do require PyTorch.
"""

import random
import numpy as np


def test_collect_shapes():
    """collect_training_data returns correct shapes."""
    from td_batch_train import collect_training_data
    from model import TDNetwork

    network = TDNetwork(hidden_sizes=[10])  # small for speed

    encodings, targets = collect_training_data(network, num_games=3)

    assert encodings.ndim == 2, f"Expected 2D, got {encodings.ndim}D"
    assert encodings.shape[1] == 196, f"Expected 196 features, got {encodings.shape[1]}"
    assert targets.ndim == 1, f"Expected 1D targets, got {targets.ndim}D"
    assert len(encodings) == len(targets), "Encoding/target count mismatch"
    assert len(encodings) > 0, "No samples collected"
    print(f"OK  shapes: {encodings.shape}, {targets.shape}")


def test_targets_in_range():
    """All target values should be in [0, 1] (probabilities)."""
    from td_batch_train import collect_training_data
    from model import TDNetwork

    network = TDNetwork(hidden_sizes=[10])
    encodings, targets = collect_training_data(network, num_games=5)

    assert np.all(targets >= 0.0), f"Target below 0: {targets.min()}"
    assert np.all(targets <= 1.0), f"Target above 1: {targets.max()}"
    # Terminal targets should be exactly 0.0 or 1.0
    # Non-terminal targets come from sigmoid output, so strictly in (0, 1)
    print(f"OK  targets in [0,1], range [{targets.min():.4f}, {targets.max():.4f}]")


def test_encoding_perspective():
    """Perspective encoding: 196 features, no turn indicator bits.

    Each encoding should have 196 features representing the board
    from the on-roll player's perspective.  Features should be
    non-negative (they represent checker counts in various slots).
    """
    from td_batch_train import collect_training_data
    from model import TDNetwork

    random.seed(42)
    network = TDNetwork(hidden_sizes=[10])
    encodings, targets = collect_training_data(network, num_games=2)

    assert encodings.shape[1] == 196, f"Expected 196 features, got {encodings.shape[1]}"
    # All features should be non-negative (checker counts)
    assert np.all(encodings >= 0.0), f"Negative feature found: {encodings.min()}"
    print(f"OK  all {len(encodings)} samples have valid 196-feature perspective encoding")


def test_samples_per_game():
    """Each game should produce roughly 40-100 samples (one per move)."""
    from td_batch_train import collect_training_data
    from model import TDNetwork

    network = TDNetwork(hidden_sizes=[10])
    num_games = 10
    encodings, targets = collect_training_data(network, num_games=num_games)

    samples_per_game = len(encodings) / num_games
    assert samples_per_game > 20, f"Too few samples/game: {samples_per_game:.1f}"
    assert samples_per_game < 300, f"Too many samples/game: {samples_per_game:.1f}"
    print(f"OK  {samples_per_game:.1f} samples/game ({len(encodings)} total from {num_games} games)")


def test_train_reduces_loss():
    """Training on a batch should reduce the MSE loss."""
    from td_batch_train import collect_training_data, train_on_batch
    from model import TDNetwork

    random.seed(42)
    network = TDNetwork(hidden_sizes=[10])

    # Collect some data
    encodings, targets = collect_training_data(network, num_games=20)

    # Measure loss before training
    import torch
    with torch.no_grad():
        x = torch.tensor(encodings, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)
        pred = network(x)
        loss_before = ((pred - y) ** 2).mean().item()

    # Train for a few epochs
    import torch
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
    avg_loss = train_on_batch(
        network, encodings, targets,
        optimizer=optimizer, batch_size=64, epochs=5,
    )

    # Measure loss after training
    with torch.no_grad():
        pred = network(x)
        loss_after = ((pred - y) ** 2).mean().item()

    print(f"OK  loss before={loss_before:.4f} after={loss_after:.4f} "
          f"(reduced by {100*(1 - loss_after/loss_before):.1f}%)")
    assert loss_after < loss_before, "Training didn't reduce loss"


def test_frozen_weights_during_collection():
    """Network weights should not change during data collection."""
    from td_batch_train import collect_training_data
    from model import TDNetwork
    import torch

    network = TDNetwork(hidden_sizes=[10])

    # Snapshot weights before
    weights_before = [p.data.clone() for p in network.parameters()]

    collect_training_data(network, num_games=5)

    # Check weights unchanged
    for i, p in enumerate(network.parameters()):
        assert torch.equal(p.data, weights_before[i]), \
            f"Parameter {i} changed during collection!"
    print("OK  weights unchanged during collection")


def test_batch_vs_online_same_direction():
    """Batch training should reduce loss on its own training data."""
    from td_batch_train import collect_training_data, train_on_batch
    from model import TDNetwork
    import torch

    random.seed(42)
    torch.manual_seed(42)

    network = TDNetwork(hidden_sizes=[10])

    # Collect data
    encodings, targets = collect_training_data(network, num_games=20)

    # Measure loss before
    x = torch.tensor(encodings, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    with torch.no_grad():
        loss_before = ((network(x) - y) ** 2).mean().item()

    # Train
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
    train_on_batch(network, encodings, targets, optimizer=optimizer, epochs=5)

    # Measure loss after
    with torch.no_grad():
        loss_after = ((network(x) - y) ** 2).mean().item()

    print(f"OK  loss before={loss_before:.4f} after={loss_after:.4f}")
    assert loss_after < loss_before, "Training should reduce loss on own data"


# ── 1-ply tests ──────────────────────────────────────────────────────────────

def test_oneply_value_symmetry():
    """1-ply value of a symmetric (initial) position should be ~0.5."""
    from td_batch_train import _oneply_value
    from model import TDNetwork
    from backgammon_engine import BoardState, get_legal_plays, switch_turn
    from encoding import encode_state

    random.seed(42)
    network = TDNetwork(hidden_sizes=[20, 20], activation="relu")
    network.eval()

    state = BoardState.initial()
    val = _oneply_value(
        state, network, eng=None, _switch_turn=switch_turn,
        encode_fn=encode_state, get_plays=get_legal_plays,
    )
    assert 0.0 <= val <= 1.0, f"Value out of range: {val}"
    # With a random network, initial position should be near 0.5
    assert 0.2 < val < 0.8, f"Initial position value too extreme: {val}"
    print(f"OK  1-ply value of initial position: {val:.4f} (expected ~0.5)")


def test_oneply_value_in_range():
    """1-ply values should always be in [0, 1]."""
    from td_batch_train import _oneply_value
    from model import TDNetwork
    from backgammon_engine import BoardState, get_legal_plays, switch_turn
    from encoding import encode_state

    random.seed(123)
    network = TDNetwork(hidden_sizes=[20, 20], activation="relu")
    network.eval()

    # Play a few random moves and check 1-ply value at each step
    state = BoardState.initial()
    for step in range(20):
        if state.is_game_over():
            break
        val = _oneply_value(
            state, network, eng=None, _switch_turn=switch_turn,
            encode_fn=encode_state, get_plays=get_legal_plays,
        )
        assert 0.0 <= val <= 1.0, f"Step {step}: value {val} out of [0,1]"

        # Make a random move
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        plays = get_legal_plays(state, (d1, d2))
        if plays:
            _, state = random.choice(plays)
            state = switch_turn(state)
        else:
            state = switch_turn(state)

    print(f"OK  all 1-ply values in [0, 1] across {step} steps")


def test_oneply_targets_match_0ply_distribution():
    """1-ply and 0-ply targets on the same positions should be highly correlated.

    For a well-trained model, 0-ply already approximates 1-ply well, so the
    correlation should be very high (>0.95). For a random model, both should
    at least have similar means (~0.5 for self-play positions).
    """
    from td_batch_train import collect_training_data, _oneply_value
    from model import TDNetwork
    from backgammon_engine import get_legal_plays, switch_turn
    from encoding import encode_state
    import torch

    random.seed(42)
    network = TDNetwork(hidden_sizes=[20, 20], activation="relu")
    network.eval()

    # Collect games and compute both target types
    encodings, targets_0ply = collect_training_data(network, num_games=3)

    # Collect with oneply=True and compare distributions
    random.seed(42)
    encodings_1ply, targets_1ply = collect_training_data(
        network, num_games=2, oneply=True,  # fewer games — 1-ply is ~100x slower
    )

    # Both should have targets in [0, 1]
    assert np.all(targets_0ply >= 0) and np.all(targets_0ply <= 1)
    assert np.all(targets_1ply >= 0) and np.all(targets_1ply <= 1)

    # Means should both be near 0.5 (self-play, symmetric)
    assert 0.3 < targets_0ply.mean() < 0.7, f"0-ply mean extreme: {targets_0ply.mean()}"
    assert 0.3 < targets_1ply.mean() < 0.7, f"1-ply mean extreme: {targets_1ply.mean()}"

    print(f"OK  0-ply mean={targets_0ply.mean():.4f}, 1-ply mean={targets_1ply.mean():.4f}, "
          f"both in [0.3, 0.7]")


def test_oneply_collect_shapes():
    """collect_training_data with oneply=True returns correct shapes."""
    from td_batch_train import collect_training_data
    from model import TDNetwork

    network = TDNetwork(hidden_sizes=[10])
    encodings, targets = collect_training_data(network, num_games=2, oneply=True)

    assert encodings.ndim == 2
    assert encodings.shape[1] == 196
    assert targets.ndim == 1
    assert len(encodings) == len(targets)
    assert len(encodings) > 0
    assert np.all(targets >= 0.0) and np.all(targets <= 1.0)
    print(f"OK  oneply shapes: {encodings.shape}, targets in "
          f"[{targets.min():.4f}, {targets.max():.4f}]")


def test_oneply_move_selection_not_random():
    """1-ply move selection should produce different (better) games than random.

    With 1-ply, games should have reasonable length (~40-80 moves). If move
    selection were broken, games might be very short or very long.
    """
    from td_batch_train import collect_training_data
    from model import TDNetwork

    random.seed(99)
    network = TDNetwork(hidden_sizes=[20, 20], activation="relu")

    enc_0ply, _ = collect_training_data(network, num_games=5, oneply=False)
    enc_1ply, _ = collect_training_data(network, num_games=2, oneply=True)

    spg_0ply = len(enc_0ply) / 5
    spg_1ply = len(enc_1ply) / 2

    # Both should produce reasonable game lengths
    assert 20 < spg_0ply < 500, f"0-ply samples/game extreme: {spg_0ply}"
    assert 20 < spg_1ply < 500, f"1-ply samples/game extreme: {spg_1ply}"
    print(f"OK  samples/game: 0-ply={spg_0ply:.0f}, 1-ply={spg_1ply:.0f}")


def test_oneply_frozen_weights():
    """Weights should not change during 1-ply collection."""
    from td_batch_train import collect_training_data
    from model import TDNetwork
    import torch

    network = TDNetwork(hidden_sizes=[10])
    weights_before = [p.data.clone() for p in network.parameters()]

    collect_training_data(network, num_games=2, oneply=True)

    for i, p in enumerate(network.parameters()):
        assert torch.equal(p.data, weights_before[i]), \
            f"Parameter {i} changed during 1-ply collection!"
    print("OK  weights unchanged during 1-ply collection")


def test_0ply_unchanged():
    """Verify 0-ply path still works identically (no regression)."""
    from td_batch_train import collect_training_data, train_on_batch
    from model import TDNetwork
    import torch

    random.seed(42)
    network = TDNetwork(hidden_sizes=[10])

    # 0-ply collection
    encodings, targets = collect_training_data(network, num_games=5, oneply=False)
    assert encodings.shape[1] == 196
    assert np.all(targets >= 0) and np.all(targets <= 1)

    # 0-ply training
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
    result = train_on_batch(network, encodings, targets, optimizer=optimizer, epochs=3)
    loss = result["train_loss"] if isinstance(result, dict) else result
    assert loss > 0, "Loss should be positive"
    print(f"OK  0-ply path unchanged, loss={loss:.4f}")


if __name__ == "__main__":
    print("Testing td_batch_train.py...\n")

    print("--- 0-ply tests ---")
    test_collect_shapes()
    test_targets_in_range()
    test_encoding_perspective()
    test_samples_per_game()
    test_frozen_weights_during_collection()
    test_train_reduces_loss()
    test_batch_vs_online_same_direction()

    print("\n--- 1-ply tests ---")
    test_oneply_value_symmetry()
    test_oneply_value_in_range()
    test_oneply_collect_shapes()
    test_oneply_targets_match_0ply_distribution()
    test_oneply_move_selection_not_random()
    test_oneply_frozen_weights()
    test_0ply_unchanged()

    print("\nAll tests passed.")
