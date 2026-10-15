"""RNG-independence guards for pooled head-to-head evaluation.

Pooling H-H runs (combine_head_to_head) is only valid if every run's
dice stream is independent:

1. Identical invocations must produce different game streams. Workers
   are seeded from OS entropy, never a fixed default.
2. combine_head_to_head must refuse to pool two files with identical
   per-game sequences (the symptom of a shared seed).
"""
import numpy as np
import pytest

from play_models import (
    play_matches_parallel, save_head_to_head, combine_head_to_head,
)


def _run_random_h2h(games=60, workers=2):
    _, _, _, per_game, _ = play_matches_parallel(
        "random", "random", games, workers=workers,
        model1_type="random", model2_type="random",
        cubeless_money=True,
    )
    return np.asarray(per_game, dtype=np.float32)


def test_identical_invocations_differ():
    """Two identical runs (same args, fresh pools) must not replay the
    same games; multi-machine pooling relies on this."""
    a = _run_random_h2h()
    b = _run_random_h2h()
    assert not np.array_equal(a, b), (
        "two identical H-H invocations produced byte-identical game "
        "streams — worker RNG is deterministically seeded")


def test_combine_rejects_duplicate_streams(tmp_path):
    """Pooling two files with the same per-game sequence must fail."""
    pg = np.random.default_rng(0).choice(
        [-2.0, -1.0, 1.0, 2.0], size=500).astype(np.float32)
    save_head_to_head(str(tmp_path), pg, game_mode="cubeless-money",
                      side1="a", side2="b", n_games=len(pg))
    save_head_to_head(str(tmp_path), pg, game_mode="cubeless-money",
                      side1="a", side2="b", n_games=len(pg))
    with pytest.raises(ValueError, match="duplicate game stream"):
        combine_head_to_head(str(tmp_path), n_bootstrap=10, do_print=False)


def test_combine_accepts_independent_streams(tmp_path):
    """Sanity: two different sequences pool fine."""
    rng = np.random.default_rng(1)
    for _ in range(2):
        pg = rng.choice([-2.0, -1.0, 1.0, 2.0], size=500).astype(np.float32)
        save_head_to_head(str(tmp_path), pg, game_mode="cubeless-money",
                          side1="a", side2="b", n_games=len(pg))
    res = combine_head_to_head(str(tmp_path), n_bootstrap=10, do_print=False)
    assert res is not None
