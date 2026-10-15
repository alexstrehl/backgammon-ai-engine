"""Correctness tests for `vectorized_matchplay.py`, the lockstep
generator collector for matchplay 1-ply training.

WHAT IS BEING PROVEN, AND WHY THIS FORMULATION
==============================================
The load-bearing property is `TestBitwiseEquivalence`:

    Driven single-game against the SAME CPU network with the SAME
    `random.Random` seed, `collect_episode_matchplay_gen` returns arrays
    that are BIT-IDENTICAL -- `np.array_equal`, zero tolerance -- to
    `trainer._collect_episode_matchplay`, in shape, row order and value.

That is the right formulation, and a weaker one would not be, for three
reasons.

1. Zero tolerance is achievable, so anything looser would hide bugs.
   The generator's contract is that the matrix it yields is byte-
   identical to the input of the serial forward it replaces; the single-
   game driver (`drive_episode`) therefore issues forwards with exactly
   the serial batch composition on exactly the serial weights, and fp32
   is deterministic under those conditions. Every arithmetic detail that
   would break exactness is deliberately mirrored rather than optimised:
   the two cube forwards are not fused into one 2-row batch, the 1-ply
   target keeps its per-dice loop instead of `expand_21`, its float64
   Python accumulation instead of a dot product, its float32 terminal
   overrides and its final [0,1] clamp, and move selection keeps the
   PYTHON move generator. (This header once said the fused C generator
   emits plays in a different order; it does not -- same plays, same
   order, all 36 ordered rolls, 8,640,000 cases, 0 mismatches:
   c_api/movegen_fuzz.py, tests/test_movegen_order_stable.py.)

2. Value equality alone is not enough, so we compare ENCODINGS too, row
   for row. The matchplay collector's failure modes are structural, not
   numeric: a cube sample emitted with the wrong `is_cube_action` flag,
   a boundary sample duplicated from the checker row instead of the cube
   row, an upweight applied on the wrong branch. All of those keep every
   target in [0,1] and every shape assertion satisfied. Only a row-for-
   row diff catches them, which is also why `TestBoundaryUpweight`,
   `TestCubeSample` and `TestCrawfordAndDeadCube` re-derive the same
   structure independently rather than trusting the diff alone.

3. Exactness pins the RNG stream, which is the real fragility. The
   `random.Random` draws are independent of network VALUES but their
   branch structure is CONDITIONED on them (`do_double` gates the
   take/pass coins; a game-ending move skips the dice reroll). A
   restructuring that hoists one draw across one yield produces a
   perfectly plausible but different trajectory. Under exact matching
   that shows up immediately as a shape mismatch; under a tolerance it
   would not show up at all.

The sweep is run across `MatchplayMode(2)` and `MatchplayMode(3)` (2pt
exercises Crawford and dead-cube suppression far more), `outcome_mix`
0.0 and 0.3 (a viewer-list bug is invisible at mix=0), and `met=None`
vs a real MatchEquityTable (`_eval_new_game` takes a completely
different branch, which also changes the number of forwards).

WHAT IS *NOT* CLAIMED
`TestWorker::test_lockstep_batch_size_does_not_change_results` compares
the full worker at B=1 against B=8. Those are NOT bit-equal by
construction: in a fused batch a row sits at an
arbitrary offset, and torch's reductions are batch-composition sensitive
at the ~1e-7 level. Rows are therefore sorted and compared with a
tolerance. Likewise a CUDA driver net is only distributionally
equivalent to the CPU serial path, never bitwise -- `TestCuda` asserts
sanity, not equality.

The B-dependence is MEASURED, not hand-waved: <= 6.0e-8 on the 16x8
fixture net used here, <= 3.0e-7 (CPU driver) and <= 4.2e-7 (CUDA
driver) on the production 512/512/256/256 net, with encodings, row
order and row count exactly equal in every case. `atol` is therefore
1e-6 -- one decade of headroom -- rather than the 1e-5 this file
originally used, which had two orders of magnitude of slack.

That tolerance is not the aliasing argument, though it was once billed
as one. While a generator is suspended at a yield, other games' move-gen
runs underneath it over the same shared C buffers, so anything retained
as a view rather than a copy shows up as a B-dependence -- but so does
harmless fp32, and a tolerance cannot tell them apart.
`TestBatchInvariance` separates them by construction: driven by a
batch-composition-INVARIANT network, B=1 and B=4 must agree BIT FOR
BIT, so any surviving difference is necessarily state. (Confirmed to
have teeth: injecting the hazard -- `bg_fast` returning views of one
shared buffer instead of copies -- breaks it by 2.0e-2.)
"""

import os
import random

import numpy as np
import pytest
import torch

from agents import CubeOffer
from backgammon_engine import BoardState, switch_turn
from encoding import CubePerspective
from met import DEFAULT_MET, MatchEquityTable
from model import TDNetwork
from modes import MatchState, MatchplayMode
from td_agent import TDAgent
import trainer as T
import vectorized_matchplay as VM
from vectorized_matchplay import (
    collect_episode_matchplay_gen,
    drive_episode,
    vectorized_matchplay_worker,
)


# ── fixtures / helpers ──────────────────────────────────────────────


def _net(encoder_name="matchplay_perspective196", hidden=(16, 8), seed=0):
    torch.manual_seed(seed)
    n = TDNetwork(
        hidden_sizes=list(hidden), encoder_name=encoder_name,
        output_mode="probability",
    )
    n.eval()
    return n


def _agent(net, met=None):
    return TDAgent(net, met=met)


_HAVE_MET = os.path.exists(DEFAULT_MET)
_met_table = None


def _met():
    """Real Kazaross-XG2 table, parsed once."""
    global _met_table
    if _met_table is None:
        _met_table = MatchEquityTable.from_xml(DEFAULT_MET)
    return _met_table


def _gen_episode(agent, net, mode, seed, met=None, oneply=True,
                 max_episode_plies=0, initial_ms=None):
    """Drive ONE generator to completion against `net` on CPU."""
    g = collect_episode_matchplay_gen(
        agent, mode, random.Random(seed),
        initial_ms if initial_ms is not None else mode.initial_match_state(),
        oneply=oneply, max_episode_plies=max_episode_plies,
    )
    return drive_episode(g, net)


def _serial_episode(agent, mode, seed, oneply=True, initial_ms=None):
    return T._collect_episode_matchplay(
        agent, mode, random.Random(seed),
        initial_ms if initial_ms is not None else mode.initial_match_state(),
        oneply=oneply,
    )


def _cube_col(encoder):
    """Feature column carrying `is_cube_action`, derived from the
    encoder rather than hard-coded: the three matchplay encoders have
    204 / 207 / 220 features and different block layouts."""
    a = encoder.match_extras(CubePerspective.CENTERED, 1, 2, 2, False, False)
    b = encoder.match_extras(CubePerspective.CENTERED, 1, 2, 2, False, True)
    diff = np.nonzero(a != b)[0]
    assert len(diff) == 1, "is_cube_action must be a single column"
    return encoder._base.num_features + int(diff[0])


def _worker_args(mode, num_episodes, seed, B, device="cpu",
                 max_episode_plies=0, net=None, met_path=None,
                 encoder_name="matchplay_perspective196", hidden=(16, 8)):
    net = net if net is not None else _net(encoder_name, hidden)
    state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    return (
        state_dict, list(net.hidden_sizes), net.input_size,
        net.output_mode, net.encoder_name,
        mode, num_episodes, seed,
        True,    # oneply
        False,   # oneply_acting
        0.0,     # boltzmann_temp
        False,   # bf16_inference
        met_path,
        B, device, max_episode_plies,
    )


def _count_games(monkeypatch):
    """Patch `opening_roll` in the vectorized module to count games:
    it is called exactly once per game of a match."""
    calls = []
    real = VM.opening_roll

    def counting(rng):
        calls.append(1)
        return real(rng)

    monkeypatch.setattr(VM, "opening_roll", counting)
    return calls


# A rigged cube offer: should_double is True (min(v_dt, v_dp) = 0.5 >
# v_nd = 0.0) and the responder passes (take iff v_dp > v_dt, and
# 0.5 > 1.0 is False). An untrained fixture net essentially never
# produces a GREEDY double/pass on its own, so the drop branch -- with
# its cube-row boundary sample and x3 upweight -- would otherwise go
# untested.
_RIGGED_CACHE = {"v_nd": 0.0, "v_dt": 1.0, "v_dp": 0.5}


def _rigged_offer(match_state, player):
    if not match_state.can_offer(player):
        return CubeOffer(should_double=False, _cache=None)
    return CubeOffer(should_double=True, _cache=dict(_RIGGED_CACHE))


def _force_greedy_drop(monkeypatch):
    """Make every cube-eligible turn a greedy double/pass, in BOTH the
    generator and the serial collector, with exploration off."""
    def rigged_gen(agent, state, match_state):
        return _rigged_offer(match_state, state.turn)
        yield  # unreachable: makes this a generator function

    monkeypatch.setattr(VM, "_offer_double_matchplay_gen", rigged_gen)
    monkeypatch.setattr(
        TDAgent, "offer_double_matchplay",
        lambda self, state, ms: _rigged_offer(ms, state.turn),
    )
    monkeypatch.setattr(VM, "_CUBE_EXPLORATION_EPS", 0.0)
    monkeypatch.setattr(T, "_CUBE_EXPLORATION_EPS", 0.0)


def _runs(enc, tgt):
    """Maximal runs of consecutive byte-identical (row, target) pairs.
    Returns a list of run lengths."""
    out = []
    i = 0
    n = len(tgt)
    while i < n:
        j = i + 1
        while (j < n and tgt[j] == tgt[i]
               and np.array_equal(enc[j], enc[i])):
            j += 1
        out.append(j - i)
        i = j
    return out


def _run_starts(enc, tgt):
    """(start_index, run_length) for every maximal run."""
    out = []
    i = 0
    for ln in _runs(enc, tgt):
        out.append((i, ln))
        i += ln
    return out


# ── 1. bitwise equivalence vs the serial collector ──────────────────


class TestBitwiseEquivalence:
    """Generator vs `trainer._collect_episode_matchplay`, zero
    tolerance. See the module docstring for why exactness is the right
    bar here rather than a numeric tolerance."""

    @pytest.mark.parametrize("match_length", [2, 3])
    @pytest.mark.parametrize("outcome_mix", [0.0, 0.3])
    def test_bitwise(self, match_length, outcome_mix):
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(match_length, outcome_mix=outcome_mix)
        compared = 0
        for seed in range(8):
            ev, tv, disc = _gen_episode(agent, net, mode, seed)
            er, tr = _serial_episode(agent, mode, seed)
            assert disc is False
            assert ev.shape == er.shape, (
                f"seed {seed}: sample COUNT differs "
                f"{ev.shape} vs {er.shape} -- the trajectory diverged, "
                f"which means an rng draw or a branch moved"
            )
            assert np.array_equal(ev, er), f"seed {seed}: encodings differ"
            assert np.array_equal(tv, tr), f"seed {seed}: targets differ"
            assert tv.dtype == np.float32
            compared += len(tv) > 0
        assert compared >= 6, "fixture produced (nearly) empty episodes"

    @pytest.mark.skipif(not _HAVE_MET, reason="no gnubg MET installed")
    def test_bitwise_with_met(self):
        """MET grounding ON: `_eval_new_game` becomes an exact table
        lookup instead of two memoised forwards, so BOTH the values and
        the number of forwards change. Parity must survive that."""
        net = _net()
        agent = _agent(net, met=_met())
        mode = MatchplayMode(3)
        for seed in range(5):
            ev, tv, _ = _gen_episode(agent, net, mode, seed)
            er, tr = _serial_episode(agent, mode, seed)
            assert ev.shape == er.shape
            assert np.array_equal(ev, er)
            assert np.array_equal(tv, tr)

    @pytest.mark.skipif(not _HAVE_MET, reason="no gnubg MET installed")
    def test_met_actually_changes_the_data(self):
        """Guard against a vacuous MET test: with the same seed and net,
        met=None and met=table must produce DIFFERENT targets (otherwise
        `test_bitwise_with_met` proves nothing about the MET branch)."""
        net = _net()
        mode = MatchplayMode(3)
        a_none, a_met = _agent(net), _agent(net, met=_met())
        _, t0, _ = _gen_episode(a_none, net, mode, 0)
        _, t1, _ = _gen_episode(a_met, net, mode, 0)
        assert not (t0.shape == t1.shape and np.array_equal(t0, t1))

    def test_zero_ply_targets_also_match(self):
        """oneply=False is not the production path but the branch
        exists; keep it honest."""
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(3)
        for seed in range(4):
            ev, tv, _ = _gen_episode(agent, net, mode, seed, oneply=False)
            er, tr = _serial_episode(agent, mode, seed, oneply=False)
            assert np.array_equal(ev, er) and np.array_equal(tv, tr)

    def test_targets_are_probabilities(self):
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(3)
        for seed in range(4):
            _, tv, _ = _gen_episode(agent, net, mode, seed)
            assert np.all(tv >= 0.0) and np.all(tv <= 1.0)


# ── 2. sample ORDER and COUNT structure ─────────────────────────────


class TestSampleStructure:
    """Order/count assertions that do not lean on the serial diff."""

    def test_cube_sample_precedes_its_checker_sample(self, monkeypatch):
        """Within a turn the cube sample (is_cube_action=1) is emitted
        BEFORE the checker sample (is_cube_action=0) for the SAME board
        -- the take path re-encodes the same board with the post-take
        cube block, so the first 196 base columns must agree.

        Exploration is switched off so the only other shape a cube row
        can take is a greedy drop, which is a run of exactly three
        identical rows ending its game (see
        `test_greedy_drop_emits_a_tripled_cube_sample`).
        """
        monkeypatch.setattr(VM, "_CUBE_EXPLORATION_EPS", 0.0)
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(3)
        col = _cube_col(agent.encoder)
        n_inline = n_drop = 0
        for seed in range(6):
            enc, tgt, _ = _gen_episode(agent, net, mode, seed)
            flags = enc[:, col]
            assert set(np.unique(flags)).issubset({0.0, 1.0})
            i = 0
            for ln in _runs(enc, tgt):
                end = i + ln - 1
                if flags[i] == 1.0:
                    if ln == 1:
                        assert end + 1 < len(flags), (
                            "a non-drop cube sample cannot end the episode"
                        )
                        assert flags[end + 1] == 0.0
                        assert np.array_equal(
                            enc[end + 1][:196], enc[i][:196],
                        ), (
                            "the checker sample must follow its cube "
                            "sample on the same board"
                        )
                        n_inline += 1
                    else:
                        assert ln == 3, "a drop boundary is tripled"
                        n_drop += 1
                i += ln
        assert n_inline > 0, "fixture never reached a cube decision"

    def test_greedy_drop_emits_a_tripled_cube_sample(self, monkeypatch):
        """Rig every cube decision into a greedy double/pass. Each game
        must then end on a run of exactly three identical rows, and
        some of those runs must be CUBE rows -- the drop path's own
        boundary sample and its x3 upweight, which nothing else can
        produce (a game that ends on a winning MOVE duplicates the
        checker row instead, and both kinds occur here because Crawford
        and dead-cube games never reach a cube decision). The whole
        episode must also stay bit-identical to the serial collector
        under the same rig."""
        monkeypatch.setattr(VM, "_BOUNDARY_UPWEIGHT", 3)
        monkeypatch.setattr(T, "_BOUNDARY_UPWEIGHT", 3)
        _force_greedy_drop(monkeypatch)
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(3)
        col = _cube_col(agent.encoder)
        cube_triples = 0
        for seed in range(6):
            calls = _count_games(monkeypatch)
            enc, tgt, _ = _gen_episode(agent, net, mode, seed)
            er, tr = _serial_episode(agent, mode, seed)
            assert np.array_equal(enc, er) and np.array_equal(tgt, tr)
            n_games = len(calls)
            # Not >= 3: the score curriculum can start the match part
            # way through, and a gammon on a checker move scores 2-3.
            assert n_games >= 1
            triples = [i for i, ln in _run_starts(enc, tgt) if ln >= 3]
            assert len(triples) == n_games, (
                "exactly one tripled boundary sample per game"
            )
            cube_triples += sum(1 for i in triples if enc[i, col] == 1.0)
        assert cube_triples > 0, (
            "no drop boundary duplicated the CUBE encoding -- only the "
            "drop branch can put an is_cube_action=1 row at a game "
            "boundary, so this is the branch's signature"
        )

    def test_counts_match_serial_exactly(self):
        """Row count is a structural fingerprint: it changes if an
        upweight is dropped, doubled, or applied on the wrong branch."""
        net = _net()
        agent = _agent(net)
        for ml in (2, 5):
            mode = MatchplayMode(ml)
            for seed in range(4):
                ev, tv, _ = _gen_episode(agent, net, mode, seed)
                er, tr = _serial_episode(agent, mode, seed)
                assert len(ev) == len(er) == len(tv) == len(tr)


# ── 3. the cube-decision sample ─────────────────────────────────────


class TestCubeSample:
    """The cube-phase sample and its max(v_nd, min(v_dt, v_dp)) target,
    re-derived independently of the serial collector."""

    def _record_offers(self, monkeypatch):
        recorded = []
        real = VM._offer_double_matchplay_gen

        def spy(agent, state, match_state):
            offer = yield from real(agent, state, match_state)
            c = offer._cache
            recorded.append(
                max(c["v_nd"], min(c["v_dt"], c["v_dp"]))
            )
            return offer

        monkeypatch.setattr(VM, "_offer_double_matchplay_gen", spy)
        return recorded

    def test_target_is_optimal_backup(self, monkeypatch):
        """Every cube sample's target is EXACTLY
        max(v_nd, min(v_dt, v_dp)) -- the optimal-action backup --
        whatever action was actually executed.

        This holds even on a greedy drop, where the collector writes
        `_eval_new_game(...)` (or 1.0) instead: should_double implies
        min(v_dt,v_dp) > v_nd and a pass implies v_dp <= v_dt, so the
        backup IS v_dp, and v_dp is that same `_eval_new_game` call.

        `_BOUNDARY_UPWEIGHT` is patched to 1 here purely so the cube
        rows line up 1:1 with the cube decisions; the x3 duplication has
        its own test.
        """
        monkeypatch.setattr(VM, "_BOUNDARY_UPWEIGHT", 1)
        monkeypatch.setattr(VM, "_CUBE_EXPLORATION_EPS", 0.0)
        recorded = self._record_offers(monkeypatch)
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(3)
        col = _cube_col(agent.encoder)
        total = 0
        for seed in range(6):
            recorded.clear()
            enc, tgt, _ = _gen_episode(agent, net, mode, seed)
            cube_idx = np.nonzero(enc[:, col] == 1.0)[0]
            assert len(cube_idx) == len(recorded), (
                "one cube sample per cube decision"
            )
            for i, v in zip(cube_idx, recorded):
                assert tgt[i] == np.float32(v)
            total += len(recorded)
        assert total > 0, "fixture never reached a cube decision"

    def test_exploration_keeps_the_optimal_backup(self, monkeypatch):
        """With eps=1.0 EVERY cube action is sampled off-policy (and the
        take/pass response too). The executed action is then random, but
        the target must still be the optimal backup -- including on an
        explored drop, where the collector must NOT substitute the
        grounded new-game value and must NOT upweight."""
        monkeypatch.setattr(VM, "_CUBE_EXPLORATION_EPS", 1.0)
        recorded = self._record_offers(monkeypatch)
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(3)
        col = _cube_col(agent.encoder)
        explored_drops = 0
        for seed in range(8):
            recorded.clear()
            enc, tgt, _ = _gen_episode(agent, net, mode, seed)
            cube_idx = np.nonzero(enc[:, col] == 1.0)[0]
            # Explored drops do not upweight, so cube rows stay 1:1.
            assert len(cube_idx) == len(recorded)
            for i, v in zip(cube_idx, recorded):
                assert tgt[i] == np.float32(v)
            # A cube row that is the LAST row of its game is a drop.
            for i in cube_idx:
                if i + 1 >= len(tgt) or not np.array_equal(
                    enc[i + 1][:196], enc[i][:196]
                ):
                    explored_drops += 1
        assert explored_drops > 0, (
            "no explored drop occurred; the branch that keeps "
            "v_cube_opt instead of the grounded outcome is untested"
        )

    def test_exploration_coins_gate_the_rng_identically(self):
        """Exploration is where the rng stream is most easily
        desynchronised (the first coin is drawn on EVERY cube-eligible
        turn, including no-double turns). Bitwise parity at eps=1.0 --
        the maximum-exploration regime -- pins that."""
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(3)
        old = T._CUBE_EXPLORATION_EPS
        try:
            T._CUBE_EXPLORATION_EPS = 1.0
            VM._CUBE_EXPLORATION_EPS = 1.0
            for seed in range(4):
                ev, tv, _ = _gen_episode(agent, net, mode, seed)
                er, tr = _serial_episode(agent, mode, seed)
                assert np.array_equal(ev, er) and np.array_equal(tv, tr)
        finally:
            T._CUBE_EXPLORATION_EPS = old
            VM._CUBE_EXPLORATION_EPS = old


# ── 4. game-boundary x3 upweighting ─────────────────────────────────


class TestBoundaryUpweight:
    def test_default_weight_is_the_unbiased_one(self, monkeypatch):
        """The x3 upweight triple-counted the winner's
        game-ending branch and biased the weighted fixed point (root
        cause of a TD degradation seen in training). The DEFAULT is now
        the unbiased 1: no duplicated boundary rows unless the
        historical env override is set."""
        assert VM._BOUNDARY_UPWEIGHT == 1
        monkeypatch.setattr(VM, "_CUBE_EXPLORATION_EPS", 0.0)
        net = _net()
        agent = _agent(net)
        for seed in range(4):
            enc, tgt, _ = _gen_episode(agent, net, MatchplayMode(3), seed)
            assert all(l < 3 for l in _runs(enc, tgt))

    def test_every_game_boundary_is_tripled(self, monkeypatch):
        """With exploration off, each game of the match contributes
        exactly one boundary sample repeated `_BOUNDARY_UPWEIGHT` = 3
        times (same encoding, same target). Counted against the number
        of games, which is read off `opening_roll` call count."""
        monkeypatch.setattr(VM, "_BOUNDARY_UPWEIGHT", 3)
        monkeypatch.setattr(VM, "_CUBE_EXPLORATION_EPS", 0.0)
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(3)
        for seed in range(5):
            calls = _count_games(monkeypatch)
            enc, tgt, _ = _gen_episode(agent, net, mode, seed)
            n_games = len(calls)
            assert n_games >= 1
            lens = _runs(enc, tgt)
            assert sorted(l for l in lens if l >= 3) == [3] * n_games, (
                f"seed {seed}: expected exactly {n_games} triples, "
                f"got run lengths {sorted(lens)[-5:]}"
            )

    def test_upweight_removes_exactly_two_rows_per_game(self, monkeypatch):
        """Cross-check on the same seeds with `_BOUNDARY_UPWEIGHT`
        patched to 1: the row count must drop by exactly 2 per game.
        This is what a dropped or mis-branched duplication looks like."""
        monkeypatch.setattr(VM, "_BOUNDARY_UPWEIGHT", 3)
        monkeypatch.setattr(VM, "_CUBE_EXPLORATION_EPS", 0.0)
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(3)
        for seed in range(4):
            calls = _count_games(monkeypatch)
            enc3, _, _ = _gen_episode(agent, net, mode, seed)
            n_games = len(calls)
            monkeypatch.setattr(VM, "_BOUNDARY_UPWEIGHT", 1)
            enc1, _, _ = _gen_episode(agent, net, mode, seed)
            monkeypatch.setattr(VM, "_BOUNDARY_UPWEIGHT", 3)
            assert len(enc3) - len(enc1) == 2 * n_games

    def test_boundary_row_duplicates_the_checker_encoding(self, monkeypatch):
        """A game that ends on a winning MOVE duplicates the CHECKER
        row (is_cube_action=0). The drop path's mirror-image assertion
        (duplicate the CUBE row) lives in
        `TestSampleStructure::test_greedy_drop_emits_a_tripled_cube_sample`;
        duplicating the wrong one is invisible in row counts."""
        monkeypatch.setattr(VM, "_BOUNDARY_UPWEIGHT", 3)
        monkeypatch.setattr(VM, "_CUBE_EXPLORATION_EPS", 0.0)
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(3)
        col = _cube_col(agent.encoder)
        saw = 0
        for seed in range(6):
            enc, tgt, _ = _gen_episode(agent, net, mode, seed)
            for i, ln in _run_starts(enc, tgt):
                if ln >= 3:
                    assert enc[i][col] == 0.0
                    saw += 1
        assert saw > 0, "no game ended on a checker move"


# ── 5. outcome-mix blending ─────────────────────────────────────────


class TestOutcomeMix:
    def test_blend_is_affine_in_a_binary_outcome(self):
        """mix does not touch the trajectory, only the targets: the
        encodings must be identical to the mix=0 run, and solving
        t_mix = (1-mix)*t0 + mix*z must recover z in {0, 1}. Both values
        must appear (each viewer sees the opposite outcome), which is
        what a corrupted `viewers` list would break."""
        net = _net()
        agent = _agent(net)
        mix = 0.25
        for seed in range(5):
            e0, t0, _ = _gen_episode(
                agent, net, MatchplayMode(3, outcome_mix=0.0), seed,
            )
            em, tm, _ = _gen_episode(
                agent, net, MatchplayMode(3, outcome_mix=mix), seed,
            )
            assert np.array_equal(e0, em)
            z = (tm.astype(np.float64) - (1 - mix) * t0.astype(np.float64)) / mix
            assert np.all(np.abs(z - np.round(z)) < 1e-4)
            assert set(np.round(z).astype(int)) == {0, 1}, (
                "both viewer perspectives must be represented"
            )

    def test_boundary_triples_share_one_z(self, monkeypatch):
        """All three copies of a boundary sample carry the same viewer,
        so the blend must not split them."""
        monkeypatch.setattr(VM, "_BOUNDARY_UPWEIGHT", 3)
        net = _net()
        agent = _agent(net)
        mix = 0.25
        for seed in range(4):
            enc, tgt, _ = _gen_episode(
                agent, net, MatchplayMode(3, outcome_mix=mix), seed,
            )
            # `_runs` requires the TARGETS to be equal too, so a blend
            # that split a triple (wrong viewer on one copy) would turn
            # every boundary triple into singletons.
            assert any(l >= 3 for l in _runs(enc, tgt)), (
                "boundary triples did not survive the outcome-mix blend"
            )


# ── 6. Crawford / post-Crawford / dead cube ─────────────────────────


def _no_curriculum(monkeypatch):
    """Pin the episode's starting score (the curriculum would otherwise
    randomise it) in BOTH collectors."""
    monkeypatch.setattr(VM, "_curriculum_start", lambda ims, rng: ims)
    monkeypatch.setattr(T, "_curriculum_start", lambda ims, rng: ims)


class TestCrawfordAndDeadCube:
    def test_crawford_game_emits_no_cube_sample(self):
        """A 1-point match is a single Crawford game (start_game flags
        Crawford as soon as either player is 1-away and the latch is
        unset), and `can_offer` is False throughout, so no cube sample
        may be emitted."""
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(1)
        col = _cube_col(agent.encoder)
        for seed in range(4):
            enc, tgt, _ = _gen_episode(agent, net, mode, seed)
            assert len(enc) > 0
            assert enc[:, col].sum() == 0.0, "cube sample during Crawford"
            # Crawford flag must be set on every row.
            assert np.array_equal(
                _serial_episode(agent, mode, seed)[0], enc,
            )

    def test_post_crawford_dead_cube_emits_no_cube_sample(self, monkeypatch):
        """(1,1) post-Crawford: cube_value 1 >= both away scores, so
        `can_offer` is False for both players (dead cube) even though
        Crawford is over. One DMP game decides the match."""
        _no_curriculum(monkeypatch)
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(2)
        ims = MatchState(
            jacoby=False, match_length=2, my_away=1, opp_away=1,
            crawford_done=True,
        )
        col = _cube_col(agent.encoder)
        for seed in range(4):
            calls = _count_games(monkeypatch)
            enc, tgt, _ = _gen_episode(
                agent, net, mode, seed, initial_ms=ims,
            )
            assert len(calls) == 1, "a DMP game ends the match"
            assert enc[:, col].sum() == 0.0, "dead cube must suppress"
            er, tr = _serial_episode(agent, mode, seed, initial_ms=ims)
            assert np.array_equal(enc, er) and np.array_equal(tgt, tr)

    def test_asymmetric_dead_cube(self, monkeypatch):
        """(1,3) post-Crawford: WHITE (1-away) may not double, BLACK
        (3-away) may. Cube samples must exist, and every one of them
        must be encoded with my_away > 1 -- i.e. from the player who is
        actually allowed to offer."""
        _no_curriculum(monkeypatch)
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(3)
        ims = MatchState(
            jacoby=False, match_length=3, my_away=1, opp_away=3,
            crawford_done=True,
        )
        col = _cube_col(agent.encoder)
        nb = agent.encoder._base.num_features  # MatchplayEncoder block
        cube_value_col, my_away_col, craw_col = nb + 4, nb + 5, nb + 7
        n_cube = 0
        for seed in range(6):
            enc, tgt, _ = _gen_episode(agent, net, mode, seed, initial_ms=ims)
            er, tr = _serial_episode(agent, mode, seed, initial_ms=ims)
            assert np.array_equal(enc, er) and np.array_equal(tgt, tr)
            for i in np.nonzero(enc[:, col] == 1.0)[0]:
                n_cube += 1
                # can_offer's dead-cube rule, read straight off the row:
                # cube_value < the doubler's own away score, never in
                # the Crawford game.
                assert enc[i, cube_value_col] < enc[i, my_away_col], (
                    "cube sample emitted on a DEAD cube "
                    f"(value {enc[i, cube_value_col]} >= away "
                    f"{enc[i, my_away_col]})"
                )
                assert enc[i, craw_col] == 0.0
        assert n_cube > 0, "fixture never reached a legal cube decision"

    def test_crawford_appears_in_a_2pt_match(self):
        """Sanity that the 2pt sweep really exercises Crawford: the
        crawford feature column must be set on some rows."""
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(2)
        craw_col = agent.encoder._base.num_features + 7
        hits = 0
        for seed in range(8):
            enc, _, _ = _gen_episode(agent, net, mode, seed)
            hits += int((enc[:, craw_col] == 1.0).any())
        assert hits > 0


# ── 7. forced pass ──────────────────────────────────────────────────


def _dance_state():
    """WHITE has a fully closed home board; BLACK has a checker on the
    bar. `switch_turn` of this is BLACK on roll with NO legal move for
    any of the 21 dice."""
    pts = [0] * 24
    for i in range(6):
        pts[i] = 2
    pts[12] = 2
    pts[23] = 1
    pts[18] = -5
    pts[16] = -5
    pts[11] = -4
    return BoardState(points=pts, bar=[0, 1], off=[0, 0], turn=0)


class TestForcedPass:
    def test_selection_returns_none_without_yielding(self):
        """A forced pass must consume no forward at all -- the mirror
        returns before its yield, exactly like the serial method."""
        net = _net()
        agent = _agent(net)
        ms = MatchplayMode(3).initial_match_state().start_game()
        barred = switch_turn(_dance_state())
        g = VM._choose_checker_action_matchplay_gen(agent, barred, (1, 2), ms)
        with pytest.raises(StopIteration) as ei:
            next(g)
        assert ei.value.value is None
        assert agent.choose_checker_action_matchplay(barred, (1, 2), ms) is None

    def test_oneply_all_dice_forced(self):
        """Every dice a forced pass: the 1-ply mirror must return the
        UNCLAMPED forced value, bit-identical to the serial method."""
        net = _net()
        agent = _agent(net)
        ms = MatchplayMode(3).initial_match_state().start_game()
        barred = switch_turn(_dance_state())
        got = drive_episode(
            _ValueGen(VM._value_oneply_checker_matchplay_gen(
                agent, barred, ms,
            )), net,
        )
        want = agent.value_oneply_checker_matchplay(barred, ms)
        assert got == want

    def test_oneply_some_dice_forced(self):
        """Mixed case: the forced-pass row is computed lazily ONCE and
        reused, and it is yielded from inside the 21-dice loop while
        other games' move-gen could be clobbering the shared C buffers.
        Bit-identical to serial is the assertion."""
        net = _net()
        agent = _agent(net)
        ms = MatchplayMode(3).initial_match_state().start_game()
        # WHITE on roll with a checker on the bar against a mostly
        # closed BLACK home board: some dice enter, some dance.
        pts = [0] * 24
        for i in range(18, 23):
            pts[i] = -2
        pts[5] = -5
        pts[10] = 8
        pts[2] = 6
        st = BoardState(points=pts, bar=[1, 0], off=[0, 0], turn=0)
        got = drive_episode(
            _ValueGen(VM._value_oneply_checker_matchplay_gen(agent, st, ms)),
            net,
        )
        want = agent.value_oneply_checker_matchplay(st, ms)
        assert got == want

    def test_forced_pass_target_inside_an_episode(self, monkeypatch):
        """End to end: episodes containing at least one forced pass are
        still bit-identical. Instrument the mirror to prove a forced
        pass actually occurred (otherwise the assertion is vacuous)."""
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(3)
        passes = []
        real = VM._choose_checker_action_matchplay_gen

        def spy(agent_, state, dice, ms, rng=None):
            r = yield from real(agent_, state, dice, ms, rng=rng)
            if r is None:
                passes.append(1)
            return r

        monkeypatch.setattr(VM, "_choose_checker_action_matchplay_gen", spy)
        for seed in range(10):
            ev, tv, _ = _gen_episode(agent, net, mode, seed)
            er, tr = _serial_episode(agent, mode, seed)
            assert np.array_equal(ev, er) and np.array_equal(tv, tr)
        assert passes, "no forced pass occurred in 10 episodes"


class _ValueGen:
    """Adapter so `drive_episode` can run a value generator whose
    StopIteration value is a bare float."""

    def __init__(self, g):
        self._g = g

    def __next__(self):
        return next(self._g)

    def send(self, v):
        return self._g.send(v)


# ── 8. match chaining and match end ─────────────────────────────────


class TestMatchChaining:
    def test_episode_spans_multiple_games(self, monkeypatch):
        """One episode = one full MATCH. Longer matches must chain."""
        net = _net()
        agent = _agent(net)
        multi = 0
        for seed in range(6):
            calls = _count_games(monkeypatch)
            _gen_episode(agent, net, MatchplayMode(5), seed)
            assert len(calls) >= 1
            multi += len(calls) > 1
        assert multi > 0, "no match ever chained a second game"

    def test_match_ends_with_target_one(self, monkeypatch):
        """The episode terminates only when `match_over()`, and the
        winning boundary sample is encoded from the winner's view, so
        with exploration off and 0-PLY targets the final targets are
        exactly 1.0. Under 1-ply the final row keeps the UNCONDITIONAL
        enumeration target (the realized 1.0 is conditioned on the
        winning roll), so it must NOT be pinned to 1.0."""
        monkeypatch.setattr(VM, "_CUBE_EXPLORATION_EPS", 0.0)
        net = _net()
        agent = _agent(net)
        for seed in range(5):
            _, tgt, _ = _gen_episode(agent, net, MatchplayMode(3), seed,
                                     oneply=False)
            assert tgt[-1] == np.float32(1.0)
        monkeypatch.setattr(VM, "_BOUNDARY_UPWEIGHT", 3)
        for seed in range(2):
            _, tgt, _ = _gen_episode(agent, net, MatchplayMode(3), seed,
                                     oneply=False)
            assert np.array_equal(tgt[-3:], np.float32([1.0, 1.0, 1.0]))
        monkeypatch.setattr(VM, "_BOUNDARY_UPWEIGHT", 1)
        finals = []
        for seed in range(5):
            _, tgt, _ = _gen_episode(agent, net, MatchplayMode(3), seed)
            finals.append(float(tgt[-1]))
        assert any(f < 1.0 for f in finals), (
            "1-ply final targets all exactly 1.0 -- the conditioned "
            f"boundary override is back: {finals}")

    def test_match_length_one_is_a_single_game(self, monkeypatch):
        net = _net()
        agent = _agent(net)
        for seed in range(5):
            calls = _count_games(monkeypatch)
            _gen_episode(agent, net, MatchplayMode(1), seed)
            assert len(calls) == 1


# ── 9. the per-game ply guard ───────────────────────────────────────


class TestPlyGuard:
    """The guard is NEW behaviour on this branch (no serial collector
    here has any episode bound). It must default to OFF, discard the
    WHOLE episode, be counted, and not depend on the lockstep batch."""

    def test_disabled_by_default_equals_huge_cap(self):
        mode = MatchplayMode(3)
        a = vectorized_matchplay_worker(_worker_args(mode, 4, 7, 4))
        b = vectorized_matchplay_worker(
            _worker_args(mode, 4, 7, 4, max_episode_plies=100000)
        )
        assert a[2] == b[2] == 0
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    @pytest.mark.parametrize("B", [1, 8])
    def test_cap_one_discards_everything_and_returns(self, B):
        mode = MatchplayMode(3)
        enc, tgt, disc = vectorized_matchplay_worker(
            _worker_args(mode, 6, 3, B, max_episode_plies=1)
        )
        assert disc == 6
        assert len(enc) == 0 and len(tgt) == 0

    def test_quota_is_honoured_when_num_episodes_exceeds_batch(self):
        """The initial fill must keep STARTING games until B slots are
        live, not merely attempt B times. With cap=1 every episode dies
        after one forward, so a naive fill would strand the quota."""
        mode = MatchplayMode(3)
        _, _, disc = vectorized_matchplay_worker(
            _worker_args(mode, 100, 11, 32, max_episode_plies=1)
        )
        assert disc == 100

    def test_discard_count_is_independent_of_batch(self):
        mode = MatchplayMode(3)
        partial = 0
        prev = None
        # 80 and 200 straddle the game-length distribution for this
        # fixture, so the sweep passes through a PARTIAL discard rather
        # than jumping from all to none.
        for cap in (1, 80, 200, 0):
            counts = set()
            for B in (1, 5):
                _, _, d = vectorized_matchplay_worker(
                    _worker_args(mode, 6, 5, B, max_episode_plies=cap)
                )
                counts.add(d)
            assert len(counts) == 1, (
                f"cap={cap}: discard count depends on batch size {counts}"
            )
            d = counts.pop()
            if cap != 0:
                if prev is not None:
                    assert d <= prev, "discards must fall as the cap rises"
                prev = d
                if 0 < d < 6:
                    partial += 1
            else:
                assert d == 0
        assert partial > 0, (
            "no cap produced a PARTIAL discard; the sweep is vacuous"
        )

    def test_rows_stay_aligned_under_discards(self):
        mode = MatchplayMode(3)
        enc, tgt, disc = vectorized_matchplay_worker(
            _worker_args(mode, 6, 5, 4, max_episode_plies=40)
        )
        assert len(enc) == len(tgt)
        assert 0 <= disc <= 6
        if len(tgt):
            assert np.all(tgt >= 0.0) and np.all(tgt <= 1.0)

    def test_guard_is_per_game_not_per_match(self):
        """The counter resets at every `start_game()`. A cap larger than
        any single game but smaller than a whole match must therefore
        discard nothing."""
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(5)
        for seed in range(4):
            e0, t0, d0 = _gen_episode(agent, net, mode, seed)
            e1, t1, d1 = _gen_episode(
                agent, net, mode, seed, max_episode_plies=400,
            )
            assert d0 is False and d1 is False
            assert np.array_equal(e0, e1) and np.array_equal(t0, t1)


# ── 10. the worker: contract, lockstep, aliasing ────────────────────


class TestWorker:
    def test_returns_three_tuple(self):
        out = vectorized_matchplay_worker(
            _worker_args(MatchplayMode(2), 2, 1, 2)
        )
        assert isinstance(out, tuple) and len(out) == 3
        enc, tgt, disc = out
        assert enc.ndim == 2 and tgt.ndim == 1 and isinstance(disc, int)
        assert len(enc) == len(tgt)

    def test_worker_at_batch_one_equals_the_serial_collector(self):
        """END-TO-END exactness. The worker derives a per-episode rng as
        `Random(base.randint(0, 2**31-1))`; replaying that derivation
        against `trainer._collect_episode_matchplay` must reproduce the
        worker's output BYTE FOR BYTE at B=1 on CPU (at B=1 the driver's
        fused batch is literally the generator's own matrix)."""
        net = _net()
        mode = MatchplayMode(3)
        seed = 4242
        enc, tgt, disc = vectorized_matchplay_worker(
            _worker_args(mode, 4, seed, 1, net=net)
        )
        assert disc == 0
        agent = _agent(net)
        base = random.Random(seed)
        e_chunks, t_chunks = [], []
        for _ in range(4):
            r = random.Random(base.randint(0, 2**31 - 1))
            e, t = T._collect_episode_matchplay(
                agent, mode, r, mode.initial_match_state(), oneply=True,
            )
            if len(e):
                e_chunks.append(e)
                t_chunks.append(t)
        e_ref = np.concatenate(e_chunks, axis=0)
        t_ref = np.concatenate(t_chunks, axis=0)
        assert enc.shape == e_ref.shape
        assert np.array_equal(enc, e_ref)
        assert np.array_equal(tgt, t_ref)

    def test_lockstep_batch_size_does_not_change_results(self):
        """B=1 vs B=8 on the same seed. This is the ALIASING test: while
        a generator is suspended at a yield, other games' move-gen runs
        over the same shared C buffers, so a retained view instead of a
        copy shows up here as a B-dependence. Games finish in a
        different order, so rows are sorted; targets are compared at
        atol=1e-6 because a row's position inside a fused batch changes
        its value in the last ~1e-7 (see the module docstring). The
        tolerance is one decade above the MEASURED delta, not five:
        `TestBatchInvariance` removes the fp noise entirely and demands
        exact equality, and this bound stops the residual from growing
        unnoticed."""
        mode = MatchplayMode(3)
        net = _net()
        e1, t1, d1 = vectorized_matchplay_worker(
            _worker_args(mode, 6, 99, 1, net=net)
        )
        e8, t8, d8 = vectorized_matchplay_worker(
            _worker_args(mode, 6, 99, 8, net=net)
        )
        assert d1 == d8 == 0
        assert e1.shape == e8.shape
        o1 = np.lexsort(e1.T[::-1])
        o8 = np.lexsort(e8.T[::-1])
        assert np.array_equal(e1[o1], e8[o8])
        np.testing.assert_allclose(t1[o1], t8[o8], rtol=0, atol=1e-6)

    def test_refuses_oneply_acting(self):
        args = list(_worker_args(MatchplayMode(2), 1, 0, 2))
        args[9] = True
        with pytest.raises(AssertionError, match="oneply-acting"):
            vectorized_matchplay_worker(tuple(args))

    def test_refuses_zero_ply_targets(self):
        args = list(_worker_args(MatchplayMode(2), 1, 0, 2))
        args[8] = False
        with pytest.raises(AssertionError, match="oneply"):
            vectorized_matchplay_worker(tuple(args))

    def test_boltzmann_output_independent_of_batch_size(self):
        """The property the old `boltzmann_temp == 0` refusal protected:
        collected data must not depend on B. Sampling draws from each
        episode's own rng, so the same seeds give the same rows at any
        batch size. (Previously this raised AssertionError instead.)"""
        out = {}
        for B in (1, 2, 4):
            args = list(_worker_args(MatchplayMode(2), 4, 7, B))
            args[10] = 0.01
            enc, tgt = vectorized_matchplay_worker(tuple(args))[:2]
            out[B] = (enc, tgt)
        def _rows(enc):
            return np.sort(enc.view([("", enc.dtype)] * enc.shape[1]).ravel())
        for B in (2, 4):
            assert out[B][0].shape == out[1][0].shape, f"B={B} row count differs"
            # Row ORDER varies with B at every temperature (episodes finish in
            # a different sequence) and always has -- rows go into a shuffled
            # replay pool, so only the multiset is meaningful. What must not
            # vary is WHICH states were visited, which is exactly what
            # global-rng sampling would have broken.
            assert np.array_equal(_rows(out[B][0]), _rows(out[1][0])), \
                f"B={B} visited different states"
            # Targets carry float32 batching noise (different batch
            # composition -> different reduction order), so compare sorted
            # values within tolerance rather than bit-exactly.
            assert np.allclose(np.sort(out[B][1]), np.sort(out[1][1]),
                               atol=1e-6), f"B={B} targets differ"

    def test_refuses_non_matchplay_mode(self):
        from modes import CubefulMoneyMode
        args = list(_worker_args(MatchplayMode(2), 1, 0, 2))
        args[5] = CubefulMoneyMode()
        with pytest.raises(AssertionError, match="matchplay"):
            vectorized_matchplay_worker(tuple(args))

    @pytest.mark.skipif(not _HAVE_MET, reason="no gnubg MET installed")
    def test_met_path_reaches_the_worker_agent(self):
        """Forgetting `met_path` in the args tuple would silently fall
        back to `_eval_new_game_net`, changing v_dp, take points, gammon
        pricing AND the boundary targets. Assert the data differs."""
        mode = MatchplayMode(3)
        net = _net()
        a = vectorized_matchplay_worker(_worker_args(mode, 4, 8, 2, net=net))
        b = vectorized_matchplay_worker(
            _worker_args(mode, 4, 8, 2, net=net, met_path=DEFAULT_MET)
        )
        assert not (a[1].shape == b[1].shape and np.array_equal(a[1], b[1]))


# ── 11. CUDA availability / trainer wiring ──────────────────────────


class TestCuda:
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="no CUDA on this host",
    )
    def test_cuda_driver_runs_and_is_sane(self):
        """A CUDA driver net is only DISTRIBUTIONALLY equivalent to the
        CPU serial path -- fp32 differs in the last ulp, and a cube
        min/max or an argmin tie can turn that into a different
        trajectory. So: assert it runs, returns aligned arrays and
        produces probabilities, not equality."""
        mode = MatchplayMode(3)
        enc, tgt, disc = vectorized_matchplay_worker(
            _worker_args(mode, 4, 21, 4, device="cuda")
        )
        assert disc == 0
        assert len(enc) == len(tgt) > 0
        assert np.all(tgt >= 0.0) and np.all(tgt <= 1.0)

    def test_trainer_falls_back_cleanly(self, capsys):
        """`--vectorized-collect` with unmet preconditions must WARN,
        name every reason, and run the ordinary collector -- never
        crash and never silently no-op."""
        net = _net()
        agent = _agent(net)
        tr = T.Trainer(agent, lr=1e-3)
        mode = MatchplayMode(2)
        tr.train(
            mode, num_episodes=1, episodes_per_round=1, batch_size=8,
            seed=0, workers=1, oneply=False, log_every=0,
            vectorized_collect=True, vectorized_device="cpu",
        )
        out = capsys.readouterr().out
        assert "INACTIVE" in out
        assert "--oneply is required" in out
        assert "--workers must be > 1" in out

    def test_trainer_default_path_untouched(self):
        """Sanity: with the flag off nothing about `train` changes."""
        net = _net()
        agent = _agent(net)
        tr = T.Trainer(agent, lr=1e-3)
        losses = tr.train(
            MatchplayMode(2), num_episodes=2, episodes_per_round=2,
            batch_size=8, seed=0, workers=1, log_every=0,
        )
        assert isinstance(losses, list)


# ── 12. Regressions from the adversarial review ─────────────────────
#
# Every test below reproduces a CONFIRMED divergence found by review of
# the first version of this collector. Each one fails on that version.


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _RowWiseNet(TDNetwork):
    """A network whose output is provably INDEPENDENT of batch
    composition: every row is pushed through the real forward on its
    own, so a row's value cannot depend on which other rows shared its
    GEMM. Used to separate "fp32 batch composition" (expected, ~1e-7)
    from "state leaked between lockstep slots" (a bug, unbounded)."""

    def forward(self, x):
        if x.dim() == 2 and x.shape[0] > 1:
            return torch.cat([
                TDNetwork.forward(self, x[i:i + 1]).reshape(1)
                for i in range(x.shape[0])
            ])
        return TDNetwork.forward(self, x)


class TestBatchInvariance:
    """REVIEW FINDING 1: `test_lockstep_batch_size_does_not_change_
    results` compares B=1 with B=8 at atol=1e-5, but the real fp32
    batch-composition delta is ~30x smaller (measured <= 3.0e-7 CPU /
    4.2e-7 CUDA on the production net, 6.0e-8 on this fixture net), so
    the old tolerance had ~2 orders of magnitude of slack in which a
    genuine cross-slot bug could hide. Two tests replace that slack:
    this one removes the fp noise by construction and demands EXACT
    equality, and the tightened tolerance in `TestWorker` pins the
    residual magnitude."""

    def test_batch_size_is_exactly_irrelevant_to_a_batch_invariant_net(
        self, monkeypatch,
    ):
        """With a batch-invariant network, B=1 and B=4 must agree BIT
        FOR BIT -- encodings AND targets. Any difference is then
        necessarily state, not arithmetic.

        Verified to have teeth: re-running this with `bg_fast` handing
        back views of one process-wide buffer instead of copies (the
        exact aliasing hazard lockstep creates) breaks it, with targets
        off by 2.0e-2."""
        monkeypatch.setattr(VM, "TDNetwork", _RowWiseNet)
        mode = MatchplayMode(2)
        net = _net()
        e1, t1, d1 = vectorized_matchplay_worker(
            _worker_args(mode, 4, 7, 1, net=net)
        )
        e4, t4, d4 = vectorized_matchplay_worker(
            _worker_args(mode, 4, 7, 4, net=net)
        )
        assert d1 == d4 == 0
        assert len(e1) > 0
        assert e1.shape == e4.shape
        o1 = np.lexsort(e1.T[::-1])
        o4 = np.lexsort(e4.T[::-1])
        assert np.array_equal(e1[o1], e4[o4])
        assert np.array_equal(t1[o1], t4[o4]), (
            "targets are B-dependent under a batch-INVARIANT net: this "
            "is cross-slot state, not fp32"
        )

    def test_the_fp32_delta_is_the_size_we_claim(self):
        """The documented magnitude, asserted rather than asserted-about.
        Fixture net (16x8): measured 6.0e-8. Keep a decade of headroom
        and no more -- the point of the number is that it is small
        enough to be arithmetic."""
        mode = MatchplayMode(3)
        net = _net()
        e1, t1, _ = vectorized_matchplay_worker(
            _worker_args(mode, 6, 99, 1, net=net)
        )
        e8, t8, _ = vectorized_matchplay_worker(
            _worker_args(mode, 6, 99, 8, net=net)
        )
        o1 = np.lexsort(e1.T[::-1])
        o8 = np.lexsort(e8.T[::-1])
        assert np.array_equal(e1[o1], e8[o8])
        assert np.abs(t1[o1] - t8[o8]).max() < 1e-6


class TestBatchSizeIsValidated:
    """REVIEW FINDING 2: `--vectorized-batch <= 0` made every worker
    return empty -- no error, no warning, zero discards -- so the run
    "completed" having trained on nothing while still reporting a normal
    episodes/s."""

    @pytest.mark.parametrize("B", [0, -1])
    def test_worker_refuses_a_batch_below_one(self, B):
        with pytest.raises(AssertionError, match="vectorized-batch"):
            vectorized_matchplay_worker(
                _worker_args(MatchplayMode(2), 10, 1, B)
            )

    def test_batch_one_still_collects(self):
        """Anti-vacuity: the bound is `< 1`, not `< 2`."""
        enc, tgt, disc = vectorized_matchplay_worker(
            _worker_args(MatchplayMode(2), 1, 1, 1)
        )
        assert len(enc) == len(tgt) > 0

    def test_trainer_names_it_and_falls_back(self, capsys):
        """The trainer must refuse the config and TRAIN ANYWAY, not
        print 'ON' and silently collect nothing."""
        net = _net()
        tr = T.Trainer(_agent(net), lr=1e-3)
        losses = tr.train(
            MatchplayMode(2), num_episodes=2, episodes_per_round=2,
            batch_size=8, seed=0, workers=2, oneply=True, log_every=0,
            vectorized_collect=True, vectorized_batch=0,
            vectorized_device="cpu",
        )
        out = capsys.readouterr().out
        assert "INACTIVE" in out
        assert "--vectorized-batch must be >= 1" in out
        assert "Vectorized lockstep collection ON" not in out
        assert len(losses) > 0, "fell back but trained on nothing"

    def test_cli_rejects_it(self):
        """Fail at argument parse time, before any weights are loaded."""
        import subprocess
        import sys as _sys
        r = subprocess.run(
            [_sys.executable, "train_batch.py", "--game-mode", "matchplay",
             "--num-episodes", "1", "--vectorized-batch", "0"],
            cwd=_REPO, capture_output=True, text=True,
        )
        assert r.returncode == 2
        assert "--vectorized-batch must be >= 1" in r.stderr


class TestFallbackAnnouncesTheDroppedPlyBound:
    """REVIEW FINDING 3: `--max-episode-plies` lives only in the
    vectorized collector, so ONE unmet precondition swings the data
    between "every episode kept" and "every episode discarded" -- and
    the INACTIVE warning named only the collector swap."""

    def test_inactive_warning_says_the_bound_is_dropped(self, capsys):
        net = _net()
        tr = T.Trainer(_agent(net), lr=1e-3)
        losses = tr.train(
            MatchplayMode(2), num_episodes=1, episodes_per_round=1,
            batch_size=8, seed=0, workers=1, oneply=True, log_every=0,
            vectorized_collect=True, vectorized_device="cpu",
            max_episode_plies=5,
        )
        out = capsys.readouterr().out
        assert "INACTIVE" in out
        assert "--max-episode-plies 5" in out
        assert "NOT" in out and "discarded" in out
        assert len(losses) > 0

    def test_no_such_note_when_the_bound_is_off(self, capsys):
        """Anti-vacuity: the note is about the bound, not boilerplate."""
        net = _net()
        tr = T.Trainer(_agent(net), lr=1e-3)
        tr.train(
            MatchplayMode(2), num_episodes=1, episodes_per_round=1,
            batch_size=8, seed=0, workers=1, oneply=True, log_every=0,
            vectorized_collect=True, vectorized_device="cpu",
        )
        out = capsys.readouterr().out
        assert "INACTIVE" in out
        assert "max-episode-plies" not in out


class TestSelectionValueShape:
    """REVIEW FINDING 4: the generator materialised the move-selection
    values as (n, 1) where the serial method has (n,) -- inert at
    temp 0 (argmin flattens), but `_boltzmann_select` at temp > 0 does
    `values.squeeze()` then softmax over dim 0, where the two shapes
    sample from different distributions."""

    def test_shape_matches_the_serial_collector(self, monkeypatch):
        net = _net()
        agent = _agent(net)
        mode = MatchplayMode(2)
        seen = []
        real = TDAgent._boltzmann_select

        def recording(self, values, rng=None):
            seen.append(tuple(values.shape))
            return real(self, values, rng=rng)

        monkeypatch.setattr(TDAgent, "_boltzmann_select", recording)
        _serial_episode(agent, mode, 5)
        serial_shapes = list(seen)
        seen.clear()
        _gen_episode(agent, net, mode, 5)
        gen_shapes = list(seen)

        assert len(serial_shapes) > 0
        assert all(len(s) == 1 for s in serial_shapes), (
            "TDNetwork.forward ends in squeeze(-1); serial values are 1-D"
        )
        assert gen_shapes == serial_shapes


class TestDeviceAdmission:
    """REVIEW FINDING 5: a worker-side CUDA failure killed the whole run
    from inside `pool.map` with no checkpoint, while every other unmet
    precondition falls back gracefully. The master checked only
    `torch.cuda.is_available()` -- not the device string, not the
    ordinal, and not whether `workers` CUDA contexts fit the card (the
    documented 48 workers x the default B=32 wants ~22 GB on a 12 GB
    card, a guaranteed round-1 abort)."""

    def test_cpu_and_plain_cuda_string_are_accepted_or_named(self):
        assert VM.device_reason("cpu") is None
        r = VM.device_reason("cuda")
        assert (r is None) == torch.cuda.is_available()

    def test_garbage_device_is_named(self):
        assert VM.device_reason("banana") is not None
        assert VM.device_reason("mps") is not None  # unsupported, not cuda

    def test_bad_ordinal_is_named(self):
        r = VM.device_reason("cuda:99")
        assert r is not None and "cuda:99" in r
        if torch.cuda.is_available():
            assert "ordinal" in r

    def test_vram_reason_is_arithmetic_on_free_memory(self, monkeypatch):
        monkeypatch.setattr(VM, "_free_vram_mb", lambda d: 12000.0)
        monkeypatch.setattr(VM, "vram_mb_per_driver", lambda: 500.0)
        assert VM.vram_reason("cuda", 8) is None
        r = VM.vram_reason("cuda", 48)
        assert r is not None
        assert "48" in r and "24000" in r
        assert "<= 21" in r  # the largest worker count that does fit

    def test_vram_check_is_skippable_and_silent_off_gpu(self, monkeypatch):
        monkeypatch.setattr(VM, "_free_vram_mb", lambda d: 12000.0)
        monkeypatch.setenv("VEC_SKIP_VRAM_CHECK", "1")
        assert VM.vram_reason("cuda", 4096) is None
        monkeypatch.delenv("VEC_SKIP_VRAM_CHECK")
        monkeypatch.setattr(VM, "_free_vram_mb", lambda d: None)
        assert VM.vram_reason("cuda", 4096) is None

    def test_trainer_falls_back_on_a_bad_device_instead_of_crashing(
        self, capsys,
    ):
        """Before: on a CUDA host this reached `pool.map` and died with
        'invalid device ordinal', losing the round and the run."""
        net = _net()
        tr = T.Trainer(_agent(net), lr=1e-3)
        losses = tr.train(
            MatchplayMode(2), num_episodes=2, episodes_per_round=2,
            batch_size=8, seed=0, workers=2, oneply=True, log_every=0,
            vectorized_collect=True, vectorized_device="cuda:99",
        )
        out = capsys.readouterr().out
        assert "INACTIVE" in out and "cuda:99" in out
        assert len(losses) > 0

    def test_trainer_consults_the_vram_check(self, capsys, monkeypatch):
        monkeypatch.setattr(
            VM, "vram_reason", lambda dev, w: "NOT ENOUGH VRAM (test)",
        )
        net = _net()
        tr = T.Trainer(_agent(net), lr=1e-3)
        losses = tr.train(
            MatchplayMode(2), num_episodes=2, episodes_per_round=2,
            batch_size=8, seed=0, workers=2, oneply=True, log_every=0,
            vectorized_collect=True, vectorized_device="cpu",
        )
        out = capsys.readouterr().out
        assert "INACTIVE" in out and "NOT ENOUGH VRAM (test)" in out
        assert len(losses) > 0


class TestWorkerSurvivesCudaFailure:
    """REVIEW FINDING 5, backstop: what the master's admission check
    cannot foresee (another process taking the card mid-round, a dead
    context) must degrade the WORKER, not the run."""

    def test_worker_retries_the_whole_call_on_cpu(self, capsys, monkeypatch):
        mode = MatchplayMode(2)
        net = _net()
        ref = vectorized_matchplay_worker(
            _worker_args(mode, 2, 3, 2, net=net, device="cpu")
        )
        real = VM._run_lockstep
        calls = []

        def flaky(a, device):
            calls.append(device)
            if device != "cpu":
                raise RuntimeError("CUDA error: out of memory")
            return real(a, device)

        monkeypatch.setattr(VM, "_run_lockstep", flaky)
        enc, tgt, disc = vectorized_matchplay_worker(
            _worker_args(mode, 2, 3, 2, net=net, device="cuda")
        )
        assert calls == ["cuda", "cpu"]
        out = capsys.readouterr().out
        assert "WARNING" in out and "CPU driver" in out
        # The retry restarts from the same seed with fresh state, so it
        # is the CPU run, not a partial one.
        assert np.array_equal(enc, ref[0])
        assert np.array_equal(tgt, ref[1])
        assert disc == ref[2]
        assert len(enc) > 0

    def test_a_non_cuda_error_is_not_swallowed(self, monkeypatch):
        def boom(a, device):
            raise RuntimeError("something else entirely")

        monkeypatch.setattr(VM, "_run_lockstep", boom)
        with pytest.raises(RuntimeError, match="something else entirely"):
            vectorized_matchplay_worker(
                _worker_args(MatchplayMode(2), 1, 0, 2, device="cuda")
            )

    def test_cpu_device_gets_no_retry_wrapper(self, monkeypatch):
        """A CPU run that fails is a real bug; do not paper over it."""
        def boom(a, device):
            raise RuntimeError("CUDA error: out of memory")

        monkeypatch.setattr(VM, "_run_lockstep", boom)
        with pytest.raises(RuntimeError, match="CUDA"):
            vectorized_matchplay_worker(
                _worker_args(MatchplayMode(2), 1, 0, 2, device="cpu")
            )


class TestRoundSizeNote:
    """REVIEW FINDING 7: the benchmarked ep/s needs each worker to get
    at least B episodes per `pool.map`. At the trainer's default
    `--episodes-per-round 100` with 12 workers each gets ~8, the fused
    batches never reach B, and throughput falls 2-3x. Nothing said so."""

    def test_note_when_the_round_is_smaller_than_workers_times_batch(
        self, capsys,
    ):
        net = _net()
        tr = T.Trainer(_agent(net), lr=1e-3)
        tr.train(
            MatchplayMode(2), num_episodes=2, episodes_per_round=2,
            batch_size=8, seed=0, workers=2, oneply=True, log_every=0,
            vectorized_collect=True, vectorized_device="cpu",
            vectorized_batch=32,
        )
        out = capsys.readouterr().out
        assert "Vectorized lockstep collection ON" in out
        assert "--episodes-per-round 2" in out
        assert "64" in out  # workers x batch

    def test_no_note_when_the_round_is_big_enough(self, capsys):
        """Anti-vacuity: the note is conditional, not always printed."""
        net = _net()
        tr = T.Trainer(_agent(net), lr=1e-3)
        tr.train(
            MatchplayMode(2), num_episodes=2, episodes_per_round=2,
            batch_size=8, seed=0, workers=2, oneply=True, log_every=0,
            vectorized_collect=True, vectorized_device="cpu",
            vectorized_batch=1,
        )
        out = capsys.readouterr().out
        assert "Vectorized lockstep collection ON" in out
        assert "episodes-per-round" not in out


# ── Boltzmann in the vectorized path ────────────────────────────────


def test_boltzmann_parity_serial_vs_generator():
    """temp > 0: the generator collector must still reproduce the serial
    collector byte-for-byte from a shared seed. This is what the old
    `boltzmann_temp == 0` refusal existed to protect -- sampling now draws
    from the episode rng, so parity holds instead of being asserted away."""
    net = _net()
    agent = _agent(net)
    agent.boltzmann_temp = 0.004
    mode = MatchplayMode(match_length=3)
    s_enc, s_t = _serial_episode(agent, mode, seed=1234)
    g_enc, g_t, _ = _gen_episode(agent, net, mode, seed=1234)
    assert s_enc.shape == g_enc.shape, (s_enc.shape, g_enc.shape)
    assert np.array_equal(s_enc, g_enc)
    assert np.allclose(s_t, g_t, atol=1e-6)


def test_boltzmann_actually_samples():
    """Sanity: at a high temperature two different seeds must diverge, so
    the parity test above is not passing because sampling is inert."""
    net = _net()
    agent = _agent(net)
    agent.boltzmann_temp = 0.05
    mode = MatchplayMode(match_length=3)
    a_enc, _, _ = _gen_episode(agent, net, mode, seed=1)
    b_enc, _, _ = _gen_episode(agent, net, mode, seed=2)
    assert not (a_enc.shape == b_enc.shape and np.array_equal(a_enc, b_enc))


def test_boltzmann_temp_zero_consumes_no_rng_draw():
    """Greedy runs must be bit-identical to before this change: temp 0
    takes the argmin branch and never touches the episode rng."""
    net = _net()
    agent = _agent(net)
    agent.boltzmann_temp = 0.0
    mode = MatchplayMode(match_length=3)
    a = _serial_episode(agent, mode, seed=99)
    b = _gen_episode(agent, net, mode, seed=99)
    assert np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1], atol=1e-6)
