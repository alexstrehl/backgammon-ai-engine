"""Bit-exact parity for the fast prob5 1-ply target.

`train_prob5._oneply_target_vec` dispatches to
`_oneply_target_vec_expand21` when the C engine is in use, else to
`_oneply_target_vec_perdice`, the reference. The fast path must match
the reference BIT FOR BIT. A 1-ulp equity difference can flip a per-dice
argmax and change a whole target, so every parity assertion is
`np.array_equal`, never a tolerance.

  * **The equity reduction.** `_prob5_equity_np` must add in torch's
    fp32 order, `(((p0+p4)+p1)+p2)+p3`. Plain left-to-right, `np.sum`,
    `@` and `einsum` all differ from it by 1 ulp (`TestEquityMirror`).

  * **Row order into the forward.** A CPU forward is not bit-invariant
    under a row permutation, so the fast path gathers forced-pass rows
    to the front as the reference does (`TestTargetParity`).

  * **Terminal replies.** `expand_21`'s `game_result_c` must agree with
    `BoardState.game_result()`, and its replies must be in
    `_DICE_OUTCOMES` order (`TestExpand21Contract`).

Corpus size: `PROB5_PARITY_POSITIONS` (default 700).
"""

import os
import random

import numpy as np
import pytest
import torch

import train_prob5 as T
from backgammon_engine import (
    BoardState, WHITE, BLACK, get_legal_plays, opening_roll, switch_turn,
)
from encoding import OPP_OFF_INDEX, TERMINAL_OFF_THRESHOLD
from model import ProbNetwork, prob5_to_equity

pytestmark = pytest.mark.skipif(
    not T._BG_FAST_AVAILABLE,
    reason="prob5 1-ply fast path needs the C engine (bg_fast)",
)

N_POSITIONS = int(os.environ.get("PROB5_PARITY_POSITIONS", "300"))


# ── fixtures ─────────────────────────────────────────────────────────


def _net(raw_logits=False, hidden=(64, 32), seed=0):
    torch.manual_seed(seed)
    net = ProbNetwork(
        hidden_sizes=list(hidden), input_size=196, activation="relu",
        encoder_name="perspective196", raw_logits=raw_logits,
    )
    net.eval()
    return net


def _fns(use_expand21=True):
    """Fresh (encode_fn, gpe_fn); `_resolve_engine_fns` builds a new
    closure each call, so the two arms can carry different flags."""
    encode_fn, gpe_fn = T._resolve_engine_fns()
    assert gpe_fn.use_expand21, "C engine expected for these tests"
    gpe_fn.use_expand21 = use_expand21
    return encode_fn, gpe_fn


def _playout_states(n, seed):
    """Random-playout states (no net). They reach bar/blocked positions,
    deep bear-offs and races that a strong net avoids."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        state, dice = opening_roll(rng)
        steps = 0
        while not state.is_game_over() and steps < 200:
            steps += 1
            out.append(state)
            if len(out) >= n:
                return out
            plays = get_legal_plays(state, dice)
            state = rng.choice(plays)[1] if plays else switch_turn(state)
            dice = (rng.randint(1, 6), rng.randint(1, 6))
    return out


def _dance_state():
    """BLACK on roll with a checker on the bar against WHITE's fully
    closed home board. All 21 dice are forced passes, the maximal
    row-order divergence between the two batch layouts."""
    pts = [0] * 24
    for i in range(6):
        pts[i] = 2          # WHITE home board closed
    pts[12] = 2
    pts[23] = 1             # 15 WHITE
    pts[18] = -5
    pts[16] = -5
    pts[11] = -4            # 14 BLACK on board, 1 on the bar
    return BoardState(points=pts, bar=[0, 1], off=[0, 0], turn=BLACK)


def _bearoff_states(n, seed):
    """Deep bear-off / near-terminal races: the only reliable source of
    terminal replies, including gammons and backgammons."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        w_off = rng.randint(10, 14)
        b_off = rng.choice([0, 0, 0, rng.randint(0, 13)])
        pts = [0] * 24
        for _ in range(15 - w_off):
            pts[rng.randint(0, 5)] += 1
        placed = 0
        while placed < 15 - b_off:
            i = rng.randint(0, 23)
            if pts[i] > 0:
                continue
            pts[i] -= 1
            placed += 1
        s = BoardState(points=pts, bar=[0, 0], off=[w_off, b_off],
                       turn=rng.randint(0, 1))
        if s.is_game_over():
            continue
        out.append(s)
    return out


def _corpus():
    """Positions + a per-category census so coverage is auditable."""
    states = (
        _playout_states(N_POSITIONS, seed=1234)
        + _bearoff_states(max(120, N_POSITIONS // 5), seed=99)
        + [_dance_state(), switch_turn(_dance_state())]
    )
    return [s for s in states if not s.is_game_over()]


@pytest.fixture(scope="module")
def corpus():
    return _corpus()


@pytest.fixture(scope="module")
def census(corpus):
    """(n_forced_pass_positions, n_forced_pass_dice, n_terminal_positions,
    n_terminal_replies, {gr: count}, n_bearoff_shape)."""
    _enc, gpe = _fns()
    n_pass_pos = n_pass_dice = 0
    n_term_pos = n_term_replies = 0
    n_bearoff = 0
    gr_hist = {1: 0, 2: 0, 3: 0}
    for s in corpus:
        npass = 0
        nterm = 0
        for dice, _p in T._DICE_OUTCOMES:
            feats, nxt = gpe(s, dice)
            if len(nxt) == 0:
                npass += 1
                continue
            mask = feats[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
            for i in np.flatnonzero(mask):
                nterm += 1
                gr_hist[nxt[int(i)].game_result()] += 1
        n_pass_dice += npass
        n_term_replies += nterm
        n_pass_pos += bool(npass)
        n_term_pos += bool(nterm)
        pts = s.points
        if s.turn == WHITE:
            home = not any(pts[i] > 0 for i in range(6, 24)) and s.bar[0] == 0
        else:
            home = not any(pts[i] < 0 for i in range(0, 18)) and s.bar[1] == 0
        n_bearoff += bool(home)
    return (n_pass_pos, n_pass_dice, n_term_pos, n_term_replies,
            gr_hist, n_bearoff)


# ── the equity mirror ────────────────────────────────────────────────


class TestEquityMirror:
    """`_prob5_equity_np` must be BITWISE `prob5_to_equity`."""

    @pytest.mark.parametrize("style", ["uniform", "rounded", "tiny",
                                       "terminal", "saturated"])
    def test_bitwise_against_torch(self, style):
        rng = np.random.default_rng(20260805)
        total = bad = 0
        for _ in range(60):
            n = int(rng.integers(1, 400))
            m = rng.random((n, 5)).astype(np.float32)
            if style == "rounded":
                m = np.round(m, 3).astype(np.float32)
            elif style == "tiny":
                m = (m * 1e-5).astype(np.float32)
            elif style == "terminal":
                m[:, 0] = 1.0
                m[:, 3:] = 0.0
            elif style == "saturated":
                m = np.clip(m * 2.0 - 0.5, 0.0, 1.0).astype(np.float32)
            ref = prob5_to_equity(torch.from_numpy(m)).numpy()
            got = T._prob5_equity_np(m)
            assert got.dtype == np.float32
            bad += int(np.count_nonzero(got != ref))
            total += n
        assert bad == 0, (
            f"{bad}/{total} rows differ from prob5_to_equity. The fp32 "
            f"reduction order in torch's sum() has changed; re-derive "
            f"_prob5_equity_np (see its docstring) -- do NOT relax this "
            f"to a tolerance, a 1-ulp equity change can flip an argmax.")

    def test_matches_the_scalar_form_used_by_the_reference(self):
        """Same call shape the per-dice path uses: a stacked (k,5) block
        of `_flip_vec` rows and `_terminal_target` rows."""
        rng = np.random.default_rng(5)
        for _ in range(200):
            k = int(rng.integers(1, 30))
            rows = []
            for _ in range(k):
                if rng.random() < 0.2:
                    r = int(rng.integers(1, 4))
                    rows.append(np.array(
                        [1.0, float(r >= 2), float(r >= 3), 0.0, 0.0],
                        dtype=np.float32))
                else:
                    rows.append(T._flip_vec(
                        rng.random(5).astype(np.float32)))
            m = np.stack(rows)
            ref = prob5_to_equity(torch.from_numpy(m)).numpy()
            assert np.array_equal(T._prob5_equity_np(m), ref)
            # and the argmax, which feeds the target
            assert int(np.argmax(T._prob5_equity_np(m))) == int(np.argmax(ref))

    def test_flip_rows_matches_flip_vec(self):
        rng = np.random.default_rng(11)
        for _ in range(200):
            n = int(rng.integers(1, 60))
            v = rng.random((n, 5)).astype(np.float32)
            got = T._flip_rows(v)
            ref = np.stack([T._flip_vec(v[i]) for i in range(n)])
            assert got.dtype == ref.dtype == np.float32
            assert np.array_equal(got, ref)

    def test_flip_rows_on_empty_block(self):
        out = T._flip_rows(np.empty((0, 5), dtype=np.float32))
        assert out.shape == (0, 5) and out.dtype == np.float32

    def test_dice_weighted_sum_matches_the_reference_loop(self):
        """The fast path's `(chosen * probs[:, None]).sum(axis=0)` must
        equal the reference's in-order `target += prob * vec` loop bit
        for bit (fp32 addition is not associative)."""
        rng = np.random.default_rng(31)
        for _ in range(3000):
            chosen = rng.random((21, 5)).astype(np.float32)
            got = (chosen * T._ONEPLY_PROBS32[:, None]).sum(axis=0)
            ref = np.zeros(5, dtype=np.float32)
            for d, (_dice, prob) in enumerate(T._DICE_OUTCOMES):
                ref += prob * chosen[d]
            assert np.array_equal(got, ref)

    def test_oneply_probs_vector_matches_dice_outcomes(self):
        """`_ONEPLY_PROBS32` must be float32, in `_DICE_OUTCOMES` order,
        and multiply bit-identically to the loop's Python float."""
        assert T._ONEPLY_PROBS32.dtype == np.float32
        assert len(T._ONEPLY_PROBS32) == len(T._DICE_OUTCOMES) == 21
        for i, (_d, p) in enumerate(T._DICE_OUTCOMES):
            assert T._ONEPLY_PROBS32[i] == np.float32(p)
            v = np.random.default_rng(i).random(5).astype(np.float32)
            assert np.array_equal(p * v, T._ONEPLY_PROBS32[i] * v)
        assert abs(float(T._ONEPLY_PROBS32.astype(np.float64).sum()) - 1.0) < 1e-6


# ── the expand_21 contract ───────────────────────────────────────────


class TestExpand21Contract:
    """Every fact `_oneply_target_vec_expand21` relies on, checked
    against the per-dice engine on real positions."""

    def test_dice_order_counts_and_features(self, corpus):
        _enc, gpe = _fns()
        checked = 0
        for s in corpus[:200]:
            exp = T._bg_fast.expand_21(s)
            assert exp is not None, "unexpected expand_21 overflow"
            feats, counts, gr, n_rows = exp
            assert counts.shape == (21,)
            assert (counts >= 1).all(), "every dice bucket must be non-empty"
            assert int(counts.sum()) == gr.shape[0]
            assert int(np.count_nonzero(gr == 0)) == n_rows
            rep = 0
            row = 0
            for d_idx, (dice, _p) in enumerate(T._DICE_OUTCOMES):
                pf, nxt = gpe(s, dice)
                n = len(nxt)
                if n == 0:
                    # forced pass: one non-terminal row, the switch_turn
                    # encoding
                    assert counts[d_idx] == 1
                    assert gr[rep] == 0
                    assert np.array_equal(
                        feats[row], T._bg_fast.encode_state(switch_turn(s)))
                    rep += 1
                    row += 1
                    continue
                assert counts[d_idx] == n, (
                    f"bucket {d_idx} ({dice}) has {counts[d_idx]} replies "
                    f"but the per-dice engine found {n} -- dice ORDER or "
                    f"count mismatch")
                mask = pf[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
                for i in range(n):
                    if mask[i]:
                        assert gr[rep] == nxt[i].game_result(), (
                            "game_result_c disagrees with "
                            "BoardState.game_result")
                    else:
                        assert gr[rep] == 0, (
                            "expand_21 called a reply terminal that the "
                            "OPP_OFF feature mask did not")
                        assert np.array_equal(feats[row], pf[i])
                        row += 1
                    rep += 1
            assert rep == gr.shape[0] and row == n_rows
            checked += 1
        assert checked >= 100

    def test_terminal_results_cover_all_three_outcomes(self, census):
        _pp, _pd, _tp, n_term, gr_hist, _bo = census
        assert n_term > 0, "corpus contains no terminal replies"
        assert gr_hist[1] > 0, "no single-win terminal replies"
        assert gr_hist[2] > 0, "no gammon terminal replies"

    def test_dance_state_is_21_forced_passes(self):
        _enc, gpe = _fns()
        s = _dance_state()
        feats, counts, gr, n_rows = T._bg_fast.expand_21(s)
        assert list(counts) == [1] * 21
        assert n_rows == 21 and (gr == 0).all()
        pe = T._bg_fast.encode_state(switch_turn(s))
        assert np.array_equal(feats, np.broadcast_to(pe, (21, 196)))
        for dice, _p in T._DICE_OUTCOMES:
            assert len(gpe(s, dice)[1]) == 0

    def test_forward_is_not_row_permutation_invariant(self):
        """The forward is not row-permutation invariant, which is why
        the fast path gathers. Skips if this BLAS is invariant."""
        net = _net(hidden=(512, 256), seed=3)
        rng = np.random.default_rng(4)
        differed = False
        for _ in range(60):
            n = int(rng.integers(50, 600))
            x = (rng.random((n, 196)) < 0.3).astype(np.float32)
            perm = rng.permutation(n)
            with torch.no_grad():
                a = net(torch.from_numpy(x)).numpy()
                b = net(torch.from_numpy(
                    np.ascontiguousarray(x[perm]))).numpy()
            if not np.array_equal(a[perm], b):
                differed = True
                break
        if not differed:
            pytest.skip("this BLAS happens to be row-permutation invariant; "
                        "the fast path's gather is then merely redundant")


# ── bitwise target parity ────────────────────────────────────────────


class TestTargetParity:

    @pytest.mark.parametrize("raw_logits", [False, True])
    @pytest.mark.parametrize("hidden", [
        (64, 32), pytest.param((512, 512, 256, 128), marks=pytest.mark.slow)])
    def test_bitwise_over_corpus(self, corpus, raw_logits, hidden):
        net = _net(raw_logits=raw_logits, hidden=hidden, seed=7)
        enc, gpe_fast = _fns(True)
        _e2, gpe_ref = _fns(False)
        worst = 0.0
        bad = []
        for i, s in enumerate(corpus):
            ref = T._oneply_target_vec_perdice(s, net, enc, gpe_ref, "cpu")
            fast = T._oneply_target_vec(s, net, enc, gpe_fast, "cpu")
            assert fast.dtype == ref.dtype == np.float32
            assert fast.shape == (5,)
            if not np.array_equal(fast, ref):
                bad.append(i)
                worst = max(worst, float(np.max(np.abs(fast - ref))))
        assert not bad, (
            f"{len(bad)}/{len(corpus)} targets differ "
            f"(max|delta| = {worst:.3e}); first offenders {bad[:8]}")

    @pytest.mark.parametrize("raw_logits", [False, True])
    def test_bitwise_on_all_dance_state(self, raw_logits):
        """All 21 buckets are forced passes. Both batches hold the same
        21 rows, and after the gather the same order."""
        net = _net(raw_logits=raw_logits, hidden=(512, 256), seed=8)
        enc, gpe_fast = _fns(True)
        _e2, gpe_ref = _fns(False)
        for s in (_dance_state(), switch_turn(_dance_state())):
            ref = T._oneply_target_vec_perdice(s, net, enc, gpe_ref, "cpu")
            fast = T._oneply_target_vec(s, net, enc, gpe_fast, "cpu")
            assert np.array_equal(fast, ref)

    def test_mixed_pass_and_play_positions_specifically(self, corpus):
        """Positions with BOTH forced-pass and playable dice, where the
        two batch layouts interleave differently."""
        net = _net(hidden=(512, 256), seed=9)
        enc, gpe_fast = _fns(True)
        _e2, gpe_ref = _fns(False)
        mixed = 0
        for s in corpus:
            npass = sum(1 for d, _p in T._DICE_OUTCOMES
                        if len(gpe_ref(s, d)[1]) == 0)
            if not (0 < npass < 21):
                continue
            mixed += 1
            ref = T._oneply_target_vec_perdice(s, net, enc, gpe_ref, "cpu")
            fast = T._oneply_target_vec(s, net, enc, gpe_fast, "cpu")
            assert np.array_equal(fast, ref), f"mixed-pass position {npass=}"
            if mixed >= 250:
                break
        assert mixed > 0, "corpus has no mixed pass/play positions"

    def test_corpus_coverage_is_not_vacuous(self, corpus, census):
        n_pass_pos, n_pass_dice, n_term_pos, n_term, gr_hist, n_bearoff = census
        print(
            f"\ncorpus: {len(corpus)} positions | "
            f"forced-pass positions {n_pass_pos} "
            f"({n_pass_dice} dice buckets) | "
            f"terminal-reply positions {n_term_pos} ({n_term} replies: "
            f"single {gr_hist[1]} gammon {gr_hist[2]} bg {gr_hist[3]}) | "
            f"bear-off-shape positions {n_bearoff} | "
            f"doubles buckets {6 * len(corpus)}"
        )
        assert len(corpus) >= 350
        assert n_pass_pos >= 10, "no forced-pass coverage"
        assert n_term_pos >= 10, "no terminal-reply coverage"
        assert n_bearoff >= 10, "no bear-off-shape coverage"

    @pytest.mark.parametrize("raw_logits", [False, True])
    def test_argmax_selection_agrees_move_for_move(self, corpus, raw_logits):
        """The per-dice best move must be the same INDEX, not just the
        same target vector."""
        net = _net(raw_logits=raw_logits, hidden=(256, 128), seed=12)
        enc, gpe = _fns(False)
        checked = 0
        for s in corpus[:150]:
            exp = T._bg_fast.expand_21(s)
            feats, counts, gr, n_rows = exp
            # fast-path per-dice choice
            fast_choice = _fast_dice_choices(s, net, enc, exp)
            ref_choice = _ref_dice_choices(s, net, enc, gpe)
            assert fast_choice == ref_choice, (
                f"per-dice argmax differs: {fast_choice} vs {ref_choice}")
            checked += 1
        assert checked >= 100


def _ref_dice_choices(state, net, encode_fn, gpe_fn):
    """Per-dice chosen move index, computed the reference way."""
    out = []
    for dice, _p in T._DICE_OUTCOMES:
        feats, nxt = gpe_fn(state, dice)
        if len(nxt) == 0:
            out.append(0)
            continue
        rows = []
        mask = feats[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
        with torch.no_grad():
            keep = np.flatnonzero(~mask)
            if keep.size:
                x = torch.from_numpy(np.ascontiguousarray(feats[keep]))
                o = net(x)
                if net.raw_logits:
                    o = torch.sigmoid(o)
                from model import prob5_postprocess
                o = prob5_postprocess(o).numpy()
            else:
                o = np.empty((0, 5), dtype=np.float32)
        k = 0
        for i in range(len(nxt)):
            if mask[i]:
                rows.append(T._terminal_target(nxt[i]))
            else:
                rows.append(T._flip_vec(o[k]))
                k += 1
        m = np.stack(rows)
        out.append(int(np.argmax(prob5_to_equity(torch.from_numpy(m)).numpy())))
    return out


def _fast_dice_choices(state, net, encode_fn, exp):
    """Per-dice chosen move index via the fast path's own reduction."""
    from model import prob5_postprocess
    feats, counts, gr, n_rows = exp
    nonterm = gr == 0
    with torch.no_grad():
        if n_rows:
            o = net(torch.from_numpy(feats))
            if net.raw_logits:
                o = torch.sigmoid(o)
            o = prob5_postprocess(o).numpy()
        else:
            o = np.empty((0, 5), dtype=np.float32)
    moves = np.empty((gr.shape[0], 5), dtype=np.float32)
    moves[nonterm] = T._flip_rows(o)
    tg = gr[~nonterm]
    if tg.size:
        t = np.zeros((tg.size, 5), dtype=np.float32)
        t[:, 0] = 1.0
        t[:, 1] = tg >= 2
        t[:, 2] = tg >= 3
        moves[~nonterm] = t
    eqs = T._prob5_equity_np(moves)
    seg = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int64)
    out = []
    for d in range(21):
        lo = int(seg[d])
        hi = lo + int(counts[d])
        out.append(int(np.argmax(eqs[lo:hi])))
    return out


# ── end-to-end: episode collection ───────────────────────────────────


class TestEpisodeParity:

    @pytest.mark.parametrize("raw_logits", [False, True])
    def test_collect_one_episode_bitwise(self, raw_logits):
        """Whole episodes, same rng seed, fast vs reference: encodings
        and targets must match exactly AND stay in lockstep."""
        net = _net(raw_logits=raw_logits, hidden=(128, 64), seed=13)
        enc, gpe_fast = _fns(True)
        _e2, gpe_ref = _fns(False)
        compared = 0
        for seed in range(4):
            a_enc, a_tgt = T._collect_one_episode(
                net, random.Random(seed), enc, gpe_ref,
                oneply=True, device="cpu")
            b_enc, b_tgt = T._collect_one_episode(
                net, random.Random(seed), enc, gpe_fast,
                oneply=True, device="cpu")
            assert a_enc.shape == b_enc.shape, f"seed {seed}: enc shape"
            assert a_tgt.shape == b_tgt.shape, f"seed {seed}: tgt shape"
            assert a_enc.shape[0] == a_tgt.shape[0], (
                f"seed {seed}: enc/tgt out of lockstep on the reference")
            assert b_enc.shape[0] == b_tgt.shape[0], (
                f"seed {seed}: enc/tgt out of lockstep on the fast path")
            assert np.array_equal(a_enc, b_enc), f"seed {seed}: encodings"
            assert np.array_equal(a_tgt, b_tgt), f"seed {seed}: targets"
            compared += a_enc.shape[0]
        assert compared >= 200, f"only {compared} plies compared"

    def test_same_seed_twice_is_identical(self):
        net = _net(hidden=(128, 64), seed=14)
        enc, gpe = _fns(True)
        for seed in (3, 21):
            e1, t1 = T._collect_one_episode(
                net, random.Random(seed), enc, gpe,
                oneply=True, device="cpu")
            e2, t2 = T._collect_one_episode(
                net, random.Random(seed), enc, gpe,
                oneply=True, device="cpu")
            assert np.array_equal(e1, e2) and np.array_equal(t1, t2)

    def test_worker_output_is_deterministic(self):
        """`_collect_worker` end to end (the thing the pool actually
        runs), same seed twice."""
        net = _net(hidden=(64, 32), seed=15)
        sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        args = (sd, net.hidden_sizes, net.input_size, net.activation,
                net.encoder_name, net.raw_logits, 3, 4242, True, "cpu",
                T.DEFAULT_MAX_EPISODE_PLIES)
        e1, t1, _ = T._collect_worker(args)
        e2, t2, _ = T._collect_worker(args)
        assert e1.shape[0] == t1.shape[0] > 0
        assert np.array_equal(e1, e2) and np.array_equal(t1, t2)

    def test_targets_are_valid_probabilities(self):
        net = _net(hidden=(64, 32), seed=16)
        enc, gpe = _fns(True)
        e, t = T._collect_one_episode(
            net, random.Random(5), enc, gpe, oneply=True, device="cpu")
        assert np.isfinite(t).all()
        assert (t >= 0.0).all() and (t <= 1.0).all()


# ── dispatch / fallback ──────────────────────────────────────────────


class TestDispatch:

    def test_flag_is_set_for_the_c_perspective196_path(self):
        _enc, gpe = T._resolve_engine_fns()
        assert gpe.use_expand21 is True

    def test_flag_is_off_for_a_non_c_encoder(self):
        """bg_fast implements only `perspective196`. The cubeful wrapper
        must route to the Python engine and the per-dice target."""
        _enc, gpe = T._resolve_engine_fns("cubeful_perspective196")
        assert gpe.use_expand21 is False

    def test_dispatcher_uses_the_reference_when_flag_is_off(self, monkeypatch):
        net = _net(hidden=(64, 32), seed=17)
        enc, gpe = _fns(False)
        called = {"n": 0}
        real = T._oneply_target_vec_expand21

        def spy(*a, **k):
            called["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(T, "_oneply_target_vec_expand21", spy)
        s = _playout_states(1, seed=2)[0]
        T._oneply_target_vec(s, net, enc, gpe, "cpu")
        assert called["n"] == 0

    def test_non_finite_outputs_raise_instead_of_poisoning_the_target(
            self, corpus):
        """The one intentional difference from the reference. On a
        NaN-producing net the reference returns a NaN target and the
        fast path raises."""
        net = _net(hidden=(64, 32), seed=19)
        with torch.no_grad():
            list(net.parameters())[0][0, 0] = float("nan")
        enc, gpe_fast = _fns(True)
        _e2, gpe_ref = _fns(False)
        s = corpus[0]
        ref = T._oneply_target_vec_perdice(s, net, enc, gpe_ref, "cpu")
        assert np.isnan(ref).any(), (
            "fixture failed to make the reference produce a NaN target")
        with pytest.raises(ValueError, match="non-finite"):
            T._oneply_target_vec(s, net, enc, gpe_fast, "cpu")

    def test_dispatcher_falls_back_when_expand21_returns_none(
            self, monkeypatch, corpus):
        """Simulates a buffer overflow: the dispatcher must silently use
        the per-dice path and still produce the reference target."""
        net = _net(hidden=(128, 64), seed=18)
        enc, gpe_fast = _fns(True)
        _e2, gpe_ref = _fns(False)
        s = corpus[3]
        want = T._oneply_target_vec_perdice(s, net, enc, gpe_ref, "cpu")
        monkeypatch.setattr(T._bg_fast, "expand_21", lambda _s: None)
        got = T._oneply_target_vec(s, net, enc, gpe_fast, "cpu")
        assert np.array_equal(got, want)


# ── against the real production checkpoint ───────────────────────────


_PROD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "best_models", "cubeless_prob5_512_512_256_256.pt")


@pytest.mark.skipif(not os.path.exists(_PROD),
                    reason=f"{_PROD} not present")
class TestProductionCheckpoint:
    """Parity on trained weights, whose outputs cluster tightly enough
    for near-equal moves to flip an argmax."""

    @pytest.mark.parametrize("raw_logits", [False, True])
    def test_bitwise_on_real_weights(self, corpus, raw_logits):
        net = ProbNetwork.load(_PROD)
        net.raw_logits = raw_logits
        net.eval()
        enc, gpe_fast = _fns(True)
        _e2, gpe_ref = _fns(False)
        n = min(len(corpus), 400)
        for s in corpus[:n]:
            ref = T._oneply_target_vec_perdice(s, net, enc, gpe_ref, "cpu")
            fast = T._oneply_target_vec(s, net, enc, gpe_fast, "cpu")
            assert np.array_equal(fast, ref)
