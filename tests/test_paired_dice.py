"""Paired dice (--paired-dice): the two games of a pair share one dice
stream with the seats swapped, so dice luck cancels within the pair.

What must hold for the variance reduction to be real and the mean to
stay unbiased:

1. The dice of a game are a pure function of the pair seed and of how
   many dice the game has consumed. Nothing else (colour choice,
   random tie-breaks, the worker RNG) may draw from that stream.
2. The mirror game therefore sees the identical roll sequence (opening
   roll included) with the nets swapped; a deterministic, colour-
   symmetric player against itself must make every pair sum to EXACTLY
   zero.
3. Different pairs get different dice; different runs get different
   pair seeds.
4. The default (unpaired) path is untouched: same fields in the npz,
   same draws from the global ``random`` module.
5. Pooling treats the pair as the unit and never mixes paired with
   independent runs.
"""
import os
import random
import shutil

import numpy as np
import pytest

from agents import Agent, CubeOffer, RandomAgent
from backgammon_engine import BLACK, WHITE
from play_models import (
    DiceStream, _OnPairs, _play_matches_parallel_ex, _play_one_signed,
    combine_head_to_head, pair_seed, paired_variance_ratio,
    report_head_to_head, save_head_to_head,
)


# ── deterministic, colour-symmetric test players ─────────────────────────

def _perspective(state, player):
    """The board as seen by ``player`` (own checkers positive, own
    direction of travel), so a WHITE position and its BLACK mirror map
    to the same tuple."""
    if player == WHITE:
        pts = tuple(state.points)
        bar = (state.bar[WHITE], state.bar[BLACK])
        off = (state.off[WHITE], state.off[BLACK])
    else:
        pts = tuple(-x for x in reversed(state.points))
        bar = (state.bar[BLACK], state.bar[WHITE])
        off = (state.off[BLACK], state.off[WHITE])
    return pts, bar, off


class MirrorAgent(Agent):
    """Picks the successor with the lexicographically smallest board from
    the mover's perspective: deterministic and exactly mirror-symmetric,
    so self-play under mirrored dice is a mirrored game."""

    def choose_checker_action(self, state, dice, actions):
        mover = state.turn
        return min(actions, key=lambda ps: _perspective(ps[1], mover))


class MirrorCubeAgent(MirrorAgent):
    """MirrorAgent that also uses the cube by a symmetric rule (double
    when ahead by >= 12 pips, always take), to exercise the cubeful
    loop: cube actions must not consume dice."""

    @staticmethod
    def _pips(state, player):
        pts, bar, _ = _perspective(state, player)
        # own checkers positive; a checker on point i (0-based, own
        # direction) needs 24 - i pips; bar = 25.
        return sum(x * (24 - i) for i, x in enumerate(pts) if x > 0) + 25 * bar[0]

    def offer_double(self, state, match_state):
        me = state.turn
        ahead = self._pips(state, 1 - me) - self._pips(state, me)
        return CubeOffer(should_double=ahead >= 12, _cache=None)

    def respond_to_double(self, state, match_state, hint=None):
        return True


def _record_dice(rec):
    return [m.dice for m in rec.moves if hasattr(m, "dice")]


# ── 1/3: the stream ──────────────────────────────────────────────────────

def test_dice_stream_is_a_pure_function_of_its_seed():
    a, b = DiceStream(12345), DiceStream(12345)
    xs = [a.randint(1, 6) for _ in range(200)]
    ys = [b.randint(1, 6) for _ in range(200)]
    assert xs == ys and a.n_dice == b.n_dice == 200
    c = DiceStream(12346)
    assert [c.randint(1, 6) for _ in range(200)] != xs


def test_pair_seeds_are_distinct_across_pairs_and_runs():
    base1, base2 = os.urandom(16).hex(), os.urandom(16).hex()
    s1 = [pair_seed(base1, i) for i in range(1000)]
    assert len(set(s1)) == 1000
    assert pair_seed(base1, 7) == pair_seed(base1, 7)
    assert pair_seed(base1, 7) != pair_seed(base2, 7)
    assert pair_seed(base1, 7) != pair_seed(base1, 8)


# ── 1: isolation from every other source of randomness ───────────────────

def test_pair_games_share_dice_even_with_random_players():
    """RandomAgent draws its moves from the GLOBAL random module. The
    two games of a pair must still see identical dice (roll k to whoever
    is on move at turn k), so the dice stream is dedicated, and
    different pairs must roll differently."""
    r1, r2 = RandomAgent(), RandomAgent()
    base = os.urandom(16).hex()
    first_rolls = set()
    for pid in range(30):
        seed = pair_seed(base, pid)
        _, ra = _play_one_signed(r1, r2, False, True, record=True,
                                 a1_is_white=True, dice_rng=DiceStream(seed))
        _, rb = _play_one_signed(r1, r2, False, True, record=True,
                                 a1_is_white=False, dice_rng=DiceStream(seed))
        da, db = _record_dice(ra), _record_dice(rb)
        n = min(len(da), len(db))
        assert n > 5
        assert da[:n] == db[:n], f"pair {pid}: dice diverged"
        # Opening roll (the one that decides who moves first) is shared,
        # and in the mirror game it goes to the other player.
        assert ra.moves[0].dice == rb.moves[0].dice
        assert ra.moves[0].player == rb.moves[0].player  # same COLOUR ...
        first_rolls.add(tuple(da[:6]))
    assert len(first_rolls) == 30                         # ... other net


# ── 2: mirrored games are exact mirrors ──────────────────────────────────

@pytest.mark.parametrize("cubeful,agent_cls", [
    (False, MirrorAgent), (True, MirrorCubeAgent),
])
def test_selfplay_pairs_sum_to_exactly_zero(cubeful, agent_cls):
    a1, a2 = agent_cls(), agent_cls()
    base = os.urandom(16).hex()
    n_pairs, nonzero, wins_a = 100, 0, 0
    for pid in range(n_pairs):
        seed = pair_seed(base, pid)
        sa_, _ = _play_one_signed(a1, a2, cubeful, True, a1_is_white=True,
                                  dice_rng=DiceStream(seed))
        sb_, _ = _play_one_signed(a1, a2, cubeful, True, a1_is_white=False,
                                  dice_rng=DiceStream(seed))
        nonzero += (sa_ + sb_) != 0
        wins_a += sa_ > 0
    assert nonzero == 0, f"{nonzero}/{n_pairs} pairs did not cancel"
    assert 0 < wins_a < n_pairs                     # both sides win games


# ── 4: default path untouched ────────────────────────────────────────────

def test_default_path_rolls_from_global_random_exactly_as_a_stream_would():
    """With dice_rng=None the game draws from the global ``random``
    module. Seeding that module as a DiceStream is seeded must reproduce
    the identical game: the default path makes the same draws in the
    same order."""
    a1, a2 = MirrorCubeAgent(), MirrorCubeAgent()
    for seed in (1, 2, 3):
        for cubeful in (False, True):
            random.seed(seed)
            s_default, _ = _play_one_signed(a1, a2, cubeful, True,
                                            a1_is_white=True)
            s_stream, _ = _play_one_signed(a1, a2, cubeful, True,
                                           a1_is_white=True,
                                           dice_rng=DiceStream(seed))
            assert s_default == s_stream


def test_unpaired_parallel_path_returns_no_pairing_and_writes_same_fields(tmp_path):
    res = _play_matches_parallel_ex(
        "random", "random", 40, 2, model1_type="random",
        model2_type="random", cubeless_money=True, paired_dice=False)
    assert len(res) == 6 and res[5] is None
    p = save_head_to_head(str(tmp_path), res[3], game_mode="cubeless-money",
                          side1="random", side2="random", n_games=40,
                          paired=res[5])
    d = np.load(p)
    assert sorted(d.files) == sorted([
        "per_game", "game_mode", "side1", "side2", "n_games", "argv",
        "sage_unpruned", "sage_version", "janowski_x", "janowski_no_fold",
        "janowski_no_fold_0ply", "janowski_cubeless_checker",
        "janowski_no_bearoff", "side1_md5", "side2_md5", "side1_depth",
        "side2_depth", "cube_cap", "jacoby", "bf16", "janowski_rev",
        "prob5_board_sanity", "hostname"])


# ── 3/5: parallel layout, npz, pooling ───────────────────────────────────

PAIRED_GAMES = 40


def _paired_run(out_dir, games=PAIRED_GAMES, workers=2):
    res = _play_matches_parallel_ex(
        "random", "random", games, workers, model1_type="random",
        model2_type="random", cubeless_money=True, paired_dice=True)
    diag = res[5]
    path = save_head_to_head(str(out_dir), res[3], game_mode="cubeless-money",
                             side1="random", side2="random", n_games=games,
                             paired=diag)
    return res, path


@pytest.fixture(scope="module")
def paired_runs(tmp_path_factory):
    """Two independent paired runs, played once per module."""
    return [_paired_run(tmp_path_factory.mktemp(f"paired{k}")) for k in range(2)]


def _copy_npz(path, tmp_path):
    """The run's npz copied into tmp_path (each test pools its own dir)."""
    dst = os.path.join(str(tmp_path), os.path.basename(path))
    shutil.copyfile(path, dst)
    return np.load(dst)


def test_paired_parallel_layout_and_npz_fields(tmp_path, paired_runs):
    (a1, a2, eq, pg, _, diag), path = paired_runs[0]
    d = _copy_npz(path, tmp_path)
    n, n_pairs = PAIRED_GAMES, PAIRED_GAMES // 2
    assert a1 + a2 == len(pg) == n
    pid = np.asarray(diag["pair_id"])
    assert np.array_equal(pid, np.repeat(np.arange(n_pairs), 2))   # global ids
    assert diag["a1_is_white"] == [True, False] * n_pairs
    assert bool(d["paired_dice"]) and str(d["pair_seed_base"]) == diag["seed_base"]
    assert len(str(d["pair_seed_base"])) == 32
    for key in ("pair_id", "a1_is_white", "n_rolls", "cube_value", "ended_by_drop"):
        assert len(d[key]) == n
    assert (d["cube_value"] == 1).all() and not d["ended_by_drop"].any()
    assert (d["n_rolls"] > 0).all()


def test_two_paired_runs_differ_and_pool_by_pair(tmp_path, paired_runs):
    d1, d2 = (_copy_npz(path, tmp_path) for _, path in paired_runs)
    assert str(d1["pair_seed_base"]) != str(d2["pair_seed_base"])
    assert not np.array_equal(d1["per_game"], d2["per_game"])
    res = combine_head_to_head(str(tmp_path), n_bootstrap=20, do_print=False)
    assert res["paired"] and res["n_pairs"] == PAIRED_GAMES
    assert res["total_games"] == 2 * PAIRED_GAMES
    assert "equity" in res["paired_gain"] and "win_rate" in res["paired_gain"]


def test_combine_refuses_paired_with_independent(tmp_path, paired_runs):
    _copy_npz(paired_runs[0][1], tmp_path)
    pg = np.random.default_rng(0).choice([-1.0, 1.0], size=PAIRED_GAMES)
    save_head_to_head(str(tmp_path), pg, game_mode="cubeless-money",
                      side1="random", side2="random", n_games=PAIRED_GAMES)
    with pytest.raises(ValueError, match="mixed paired-dice and independent"):
        combine_head_to_head(str(tmp_path), n_bootstrap=10, do_print=False)


def test_combine_refuses_duplicate_pair_seed_base(tmp_path, paired_runs):
    (_, _, _, pg, _, diag), path = paired_runs[0]
    _copy_npz(path, tmp_path)
    pg2 = list(pg)
    pg2[0], pg2[1] = pg2[1], pg2[0]      # same base, per-game not identical
    save_head_to_head(str(tmp_path), pg2, game_mode="cubeless-money",
                      side1="random", side2="random", n_games=PAIRED_GAMES,
                      paired=diag)
    with pytest.raises(ValueError, match="duplicate pair seed base"):
        combine_head_to_head(str(tmp_path), n_bootstrap=10, do_print=False)


def test_save_rejects_broken_pair_layout(tmp_path):
    diag = {"seed_base": "00" * 16, "pair_id": [0, 1, 1, 2],
            "a1_is_white": [True, False] * 2, "n_rolls": [1] * 4,
            "cube_value": [1] * 4, "ended_by_drop": [False] * 4}
    with pytest.raises(ValueError, match="pair layout"):
        save_head_to_head(str(tmp_path), [1.0, -1.0, 1.0, -1.0],
                          game_mode="cubeless-money", side1="a", side2="b",
                          n_games=4, paired=diag)


# ── the pair bootstrap and the gain diagnostic ───────────────────────────

def test_on_pairs_flattens_rows():
    rows = np.array([[1.0, -1.0], [3.0, 1.0]])
    assert _OnPairs(np.mean)(rows) == 1.0
    assert _OnPairs(np.max)(rows[:1]) == 1.0


def test_paired_variance_ratio_known_cases():
    # perfect cancellation -> infinite gain; independent-like -> ~2 games
    pairs = np.array([[1.0, -1.0], [2.0, -2.0], [-1.0, 1.0], [3.0, -3.0]])
    gpp, factor = paired_variance_ratio(pairs)
    assert gpp == float("inf") and factor == float("inf")
    rng = np.random.default_rng(0)
    pairs = rng.normal(size=(200000, 2))
    gpp, factor = paired_variance_ratio(pairs)
    assert abs(gpp - 2.0) < 0.05 and abs(factor - 1.0) < 0.025
    # perfectly correlated (no cancellation) -> one pair is worth ONE game
    x = rng.normal(size=100000)
    gpp, factor = paired_variance_ratio(np.stack([x, x], 1))
    assert abs(gpp - 1.0) < 1e-4          # ddof: (2n-1) vs (n-1)


def test_report_paired_ci_is_tighter_when_pairs_cancel():
    rng = np.random.default_rng(1)
    luck = rng.normal(size=5000)
    skill = 0.02
    pairs = np.stack([skill + luck + 0.1 * rng.normal(size=5000),
                      skill - luck + 0.1 * rng.normal(size=5000)], 1)
    pg = pairs.ravel()
    w1 = int((pg > 0).sum())
    indep = report_head_to_head(w1, len(pg) - w1, float(pg.sum()), pg,
                                cubeless_money=True, n_bootstrap=400,
                                do_print=False)
    paired = report_head_to_head(w1, len(pg) - w1, float(pg.sum()), pg,
                                 cubeless_money=True, n_bootstrap=400,
                                 do_print=False, pairs=pairs)
    assert paired["paired"] and not indep["paired"]
    assert paired["eq_mEq_per_game"] == indep["eq_mEq_per_game"]  # same mean
    w_i = indep["equity_ci_mEq"][1] - indep["equity_ci_mEq"][0]
    w_p = paired["equity_ci_mEq"][1] - paired["equity_ci_mEq"][0]
    assert w_p < 0.3 * w_i
    assert paired["paired_gain"]["equity"][1] > 10


# ── real checkpoint (slow; skipped when the model is absent) ─────────────

_PROD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "best_models", "cubeful_money_512_512_256_256.pt")


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isfile(_PROD), reason="prod checkpoint absent")
def test_prod_selfplay_paired_zero_sum_cubeful():
    from play_models import _load_agent
    a1, a2 = _load_agent(_PROD), _load_agent(_PROD)
    base = os.urandom(16).hex()
    for pid in range(150):
        seed = pair_seed(base, pid)
        sa, _ = _play_one_signed(a1, a2, True, True, a1_is_white=True,
                                 dice_rng=DiceStream(seed))
        sb, _ = _play_one_signed(a1, a2, True, True, a1_is_white=False,
                                 dice_rng=DiceStream(seed))
        assert sa + sb == 0, (pid, sa, sb)


# ── play_models.py --paired-dice ─────────────────────────────────────────

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROB5 = os.path.join(_REPO, "best_models", "cubeless_prob5_512_512_256_256.pt")


def _play_models(*args):
    import subprocess
    import sys
    return subprocess.run([sys.executable, os.path.join(_REPO, "play_models.py"),
                           *args], cwd=_REPO, capture_output=True, text=True,
                          timeout=900)


@pytest.mark.parametrize("extra", [["--games", "3"], ["--games", "4", "--model1-white"],
                                   ["--games", "4", "--save-games", "/nonexistent"]],
                         ids=["odd", "model1-white", "save-games"])
def test_play_models_paired_refusals(extra):
    r = _play_models("--model1", "random", "--random", "--paired-dice", *extra)
    assert r.returncode == 2 and "--paired-dice" in r.stderr, r.stderr
    assert "Playing" not in r.stdout


@pytest.mark.slow
@pytest.mark.parametrize("path", ["--c-api", "--python-api"])
@pytest.mark.parametrize("mode,model", [("cubeful-money", _PROD),
                                        ("cubeless-money", _PROB5)],
                         ids=["cubeful", "cubeless"])
def test_play_models_paired_selfplay(tmp_path, path, mode, model):
    """A deterministic net against itself: every pair sums to exactly 0;
    the report bootstraps over pairs and the npz records the pairing."""
    if not os.path.isfile(model):
        pytest.skip(f"{model} absent")
    games = 8
    r = _play_models("--model1", model, "--model2", model, "--game-mode", mode,
                     "--games", str(games), "--workers", "2", "--paired-dice",
                     path, "--save-hh", str(tmp_path))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (f"RESULTS  (paired dice: {games // 2} pairs, CIs bootstrap over pairs)"
            in r.stdout)
    assert "Paired-dice gain" in r.stdout
    assert ("Play path: C API" if path == "--c-api" else
            "Play path: Python (--python-api)") in r.stdout
    (f,) = os.listdir(tmp_path)
    d = np.load(os.path.join(tmp_path, f))
    assert bool(d["paired_dice"]) and len(str(d["pair_seed_base"])) == 32
    assert np.array_equal(d["pair_id"], np.repeat(np.arange(games // 2), 2))
    assert (d["per_game"].reshape(-1, 2).sum(axis=1) == 0).all()
