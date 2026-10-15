"""Paired (common-random-number) dice in eval_matchplay: seed derivation,
exact mirror self-play, pair statistics and the --combine guards."""

import os

import numpy as np
import pytest

import eval_matchplay as em

CHAMP = "best_models/matchplay_1h64met_512_512_256_256.pt"


def test_seed_derivation_deterministic_and_distinct():
    b = em.pair_seed_base(7)
    assert b == em.pair_seed_base(7) and len(b) == 32
    assert b != em.pair_seed_base(8)
    seeds = {em.game_seed(b, p, g) for p in range(50) for g in range(20)}
    assert len(seeds) == 50 * 20


def test_paired_stats_counts_and_factor():
    # 6 split pairs, 3 both, 1 neither
    won = [1, 0] * 6 + [1, 1] * 3 + [0, 0]
    pid = np.repeat(np.arange(10), 2)
    ps = em.paired_stats(won, pid)
    assert (ps["n_both"], ps["n_split"], ps["n_none"]) == (3, 6, 1)
    assert ps["wr"] == pytest.approx(12 / 20)
    pm = np.array([0.5] * 6 + [1.0] * 3 + [0.0])
    assert ps["se_paired"] == pytest.approx(np.sqrt(pm.var(ddof=1) / 10))
    assert ps["factor"] == pytest.approx(
        ps["se_unpaired"] ** 2 / ps["se_paired"] ** 2)


def test_paired_stats_rejects_unaligned_pairs():
    with pytest.raises(ValueError):
        em.paired_stats([1, 0, 1, 0], [0, 1, 0, 1])


def test_mirror_selfplay_every_pair_splits():
    """Same deterministic net on both sides: the second match of a pair is
    the first with the colours (and names) swapped, dice for dice, so
    model1 wins exactly one of the two -- 50% by construction."""
    if not os.path.exists(CHAMP):
        pytest.skip("no matchplay checkpoint")
    from met import resolve_met_path
    met = resolve_met_path("auto")
    a1, a2 = em._build_agents(CHAMP, "model:" + CHAMP, met, met)
    base = em.pair_seed_base(3)
    for pid in range(4):
        st = []
        wa, wb = em.play_pair(a1, a2, 3, base, pid, st)
        assert wa + wb == 1
        assert st[0]["n_games"] == st[1]["n_games"]


def _fake_npz(path, paired, base="00" * 16, seeds=(1,)):
    kw = dict(won=np.array([1, 0, 1, 1], dtype=np.int8),
              model1_is_white=np.array([1, 0, 1, 0], dtype=np.int8),
              model1="m1", model2="model:m2", match_length=11, seed=1,
              met="x", met2="x", referee_met="", commit="c",
              seeds=np.array(seeds))
    if paired:
        kw.update(paired=True, pair_seed_base=base,
                  pair_id=np.array([0, 0, 1, 1]),
                  n_games=np.array([5, 5, 6, 7]))
    np.savez(path, **kw)
    return str(path)


def test_combine_guards(tmp_path, capsys):
    p1 = _fake_npz(tmp_path / "a.npz", True, "11" * 16)
    p2 = _fake_npz(tmp_path / "b.npz", True, "22" * 16)
    u = _fake_npz(tmp_path / "u.npz", False)
    dup = _fake_npz(tmp_path / "d.npz", True, "11" * 16)
    em.combine([p1, p2])
    out = capsys.readouterr().out
    assert "4 pairs / 8 matches" in out
    with pytest.raises(SystemExit):
        em.combine([p1, u])
    with pytest.raises(SystemExit):
        em.combine([p1, dup])
