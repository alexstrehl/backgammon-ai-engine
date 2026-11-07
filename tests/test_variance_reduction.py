"""Dice control variates: centering, unbiasedness, and unchanged play."""
import json

import numpy as np
import pytest
import torch

from backgammon_engine import BoardState, switch_turn
from c_api import puretd_py as P
from experiments.head_to_head import variance as V
from model import ProbNetwork, TDNetwork

DICE = [(a, b) for a in range(1, 7) for b in range(a, 7)]
WEIGHTS = np.array([1 if a == b else 2 for a, b in DICE]) / 36


@pytest.fixture(scope="module")
def engines():
    P.ensure_lib("simd")
    models = []
    for seed in (3, 9):
        torch.manual_seed(seed)
        models.append(ProbNetwork(hidden_sizes=[12, 8]))
    torch.manual_seed(17)
    models.append(TDNetwork(hidden_sizes=[12], output_mode="equity"))
    es = [P.Engine(m, "simd") for m in models]
    yield es
    for e in es:
        e.close()


def bearoff(n=3, turn=0):
    points = [0] * 24
    points[0], points[23] = n, -n
    return BoardState(points, [0, 0], [15-n, 15-n], turn)


def pass_state():
    points = [0] * 24
    points[18:24] = [-2] * 6
    return BoardState(points, [1, 0], [14, 3], 0)


@pytest.mark.parametrize("which", [0, 2])
def test_dice_values_match_independent_roll_evaluations(engines, which):
    e = engines[which]
    # Includes both movers, all-terminal, mixed terminal/nonterminal,
    # and a fully blocked bar (all dice force a pass).
    for state in (BoardState.initial(), bearoff(3), bearoff(3, 1),
                  bearoff(1), pass_state()):
        q = e.dice_values(state)
        expected = []
        for dice in DICE:
            idx, _, value = e.best_move(state, dice)
            expected.append(value if idx >= 0 else -e.value(switch_turn(state)))
        np.testing.assert_allclose(q, expected, atol=2e-7, rtol=1e-6)
        assert abs(WEIGHTS @ (q - WEIGHTS @ q)) < 2e-15
    q = e.dice_values(pass_state())
    assert np.ptp(q) == 0
    np.testing.assert_array_equal(e.dice_values(bearoff(1)), np.ones(21))


def test_exact_small_endgame_preserves_mean_with_inaccurate_reference(engines):
    a, b, reference = engines

    def moments(state):
        """Enumerate the whole chance tree, returning E[Y], E[Z], E[C]."""
        if state.is_game_over():
            y = (1 if state.winner() == 0 else -1) * state.game_result()
            return np.array([y, y, 0.0])
        q = reference.dice_values(state)
        center = WEIGHTS @ q
        answer = np.zeros(3)
        actor = (a, b)[state.turn]
        for p, dice, v in zip(WEIGHTS, DICE, q):
            _, nxt, _ = actor.best_move(state, dice)
            if nxt is None:
                nxt = switch_turn(state)
            result = moments(nxt)
            correction = (1 if state.turn == 0 else -1) * (v - center)
            result[1] -= correction
            result[2] += correction
            answer += p * result
        return answer

    raw, adjusted, correction = moments(bearoff(3))
    # White wins immediately on doubles, otherwise wins iff Black's
    # next roll is not doubles: P(win)=1/6 + (5/6)*(5/6)=31/36.
    assert raw == pytest.approx(2 * 31 / 36 - 1, abs=1e-12)
    assert adjusted == pytest.approx(raw, abs=1e-12)
    assert correction == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("every", [1, 4])
@pytest.mark.parametrize("reference_index", [0, 2])
def test_correction_matches_record_replay_and_does_not_change_game(engines, every, reference_index):
    a, b, _ = engines
    ref = engines[reference_index]
    opts = P.game_opts(cubeful=False)
    r0, r1 = P.Rng(41, "simd"), P.Rng(41, "simd")
    base, records0 = P.play_game(a, b, opts, r0, record=True)
    vr, records1 = P.play_game(a, b, opts, r1, record=True, reference=ref, vr_every=every)
    assert len(records0) == len(records1)
    for r, s in zip(records0, records1):
        assert {k: v for k, v in r.items() if k != "value"} == {
            k: v for k, v in s.items() if k != "value"}
        np.testing.assert_equal(r["value"], s["value"])  # passes carry NaN
    assert r0.getstate() == r1.getstate()
    assert all(vr[k] == v for k, v in base.items())
    assert vr["n_corrected"] == len(records1[1::every])
    luck = 0.0
    for record in records1[1::every]:
        q = ref.dice_values(record["state"])
        k = DICE.index(tuple(sorted(record["dice"])))
        sign = 1 if record["state"].turn == 0 else -1
        luck += sign * (q[k] - WEIGHTS @ q)
    assert vr["luck_white"] == pytest.approx(luck, abs=1e-12)


def test_selfplay_pairs_cancel_raw_and_adjusted(engines):
    a, _, ref = engines
    # With identical actors both legs are the same board game, and side1
    # occupies opposite seats, so the signed scores cancel exactly.
    for seed in (12, 19):
        legs = [P.play_game(a, a, P.game_opts(cubeful=False), seed,
                            reference=ref)[0] for _ in range(2)]
        assert legs[0] == legs[1]
        assert legs[0]["luck_white"] - legs[1]["luck_white"] == 0


def test_invalid_modes_and_intervals_fail_before_play(engines):
    a, b, ref = engines
    rng = P.Rng(9, "simd")
    before = rng.getstate()
    with pytest.raises(ValueError, match="cubeless money"):
        P.play_game(a, b, P.game_opts(cubeful=True), rng, reference=ref)
    for every in (0, -1, 1.5, 2**31):
        with pytest.raises(ValueError, match="positive C int"):
            P.play_game(a, b, P.game_opts(cubeful=False), rng,
                        reference=ref, vr_every=every)
    assert rng.getstate() == before
    terminal = bearoff(1)
    terminal.off[0] = 15
    with pytest.raises(ValueError, match="nonterminal"):
        ref.dice_values(terminal)


def fake_data(n=100):
    rng = np.random.default_rng(19)
    noise = rng.normal(size=(n, 2))
    return dict(raw=0.01 + noise, luck=noise,
                turns=np.full((n, 2), 4), corrected=np.full((n, 2), 3),
                seconds=np.tile([1.0, 3.0], (n, 1)),
                cpu_seconds=np.tile([1.0, 3.0], (n, 1)))


def test_summary_uses_pairs_and_fixed_coefficient():
    d = fake_data()
    summary = V.summarize(d, beta=0.5)
    pair_means = (d["raw"] - 0.5 * d["luck"]).mean(axis=1)
    assert summary["adjusted"]["se_mEq"] == pytest.approx(
        1000 * pair_means.std(ddof=1) / np.sqrt(len(pair_means)))
    assert summary["variance_factor"] == pytest.approx(4)
    assert summary["time_to_precision_gain"] == pytest.approx(4/3)
    assert summary["runtime_factor"] == 3
    c = summary["luck_centering"]
    leg = d["luck"].mean(0) / (d["luck"].std(0, ddof=1) / np.sqrt(len(d["luck"])))
    np.testing.assert_allclose(c["leg_z"], leg)
    assert c["pair_z"] == pytest.approx(
        summary["correction"]["mean_mEq"] / summary["correction"]["se_mEq"])
    with pytest.raises(ValueError, match="finite"):
        V.summarize(d, float("nan"))


def test_pooling_refuses_duplicate_seed_pairs_and_different_reference(tmp_path):
    meta = dict(schema=V.SCHEMA, side1_sha256="a", side2_sha256="b",
                reference_sha256="r", backend="simd", backend_description="test",
                every=1, opening_correction=False, game_mode="cubeless-money",
                plies=[0, 0], engine_sha256="engine", bf16="0", expand_dedup="1", seed=12)

    def save(name, metadata, start):
        p = tmp_path / name
        np.savez(p, **fake_data(4), metadata=json.dumps(metadata), pair_id=np.arange(start, start+4))
        return p

    p1, p2 = save("one.npz", meta, 0), save("two.npz", meta, 4)
    data, _ = V.load_runs([p1, p2])
    assert len(data["raw"]) == 8
    with pytest.raises(ValueError, match="duplicate seed/pair"):
        V.load_runs([p1, p1])
    p3 = save("three.npz", {**meta, "reference_sha256": "other"}, 8)
    with pytest.raises(ValueError, match="different models"):
        V.load_runs([p1, p3])


def test_worker_chunking_does_not_change_games(engines):
    a, b, ref = engines
    V._ENGINES = (a, b, ref, 2, False, True, (0, 0))
    _, together = V._chunk((8, 4, 2))
    chunks = [V._chunk((8, i, 1))[1] for i in (4, 5)]
    for key in ("raw", "luck", "turns", "corrected"):
        np.testing.assert_array_equal(together[key], np.concatenate([d[key] for d in chunks]))


def test_capped_target_is_clipped_before_subtracting_luck():
    d = fake_data(4)
    d["raw"] = np.array([[256., -1.], [-512., 4.], [1., 2.], [256., -256.]])
    d["luck"] = np.array([[200., 0.], [-400., 0.], [0., 0.], [0., -300.]])
    s = V.summarize(d, payoff_cap=128)
    want = (np.clip(d["raw"], -128, 128) - d["luck"]).mean()
    wrong = np.clip(d["raw"] - d["luck"], -128, 128).mean()
    assert want != wrong
    assert s["adjusted"]["mean_mEq"] == pytest.approx(1000 * want)
    assert s["tails"]["games_over_128"] == 4


@pytest.fixture(scope="module")
def cubeful_engines():
    P.ensure_lib("simd")
    torch.manual_seed(170)
    ref = P.Engine(TDNetwork(hidden_sizes=[12], encoder_name="cubeful_perspective196",
                             output_mode="equity"), "simd")
    a = P.Engine(str(V.DEFAULT_CF_4L), "simd")
    b = P.Engine(str(V.DEFAULT_CF_3L), "simd")
    yield a, b, ref
    for e in (a, b, ref):
        e.close()


@pytest.mark.parametrize("owner", [0, 1, 2])
@pytest.mark.parametrize("jacoby", [False, True])
def test_cubeful_dice_values_match_independent_moves(cubeful_engines, owner, jacoby):
    from modes import MatchState, CubeOwner
    a, _, _ = cubeful_engines
    ms = MatchState(CubeOwner(owner), 1 if owner == 0 else 8, jacoby)
    for state in (BoardState.initial(), bearoff(3), bearoff(3, 1), pass_state()):
        q = a.dice_values(state, ms)
        want = []
        for dice in DICE:
            idx, _, value = a.best_move(state, dice, ms)
            want.append(value if idx >= 0 else -a.value(
                switch_turn(state), ms, is_cube_action=ms.can_offer(1-state.turn)))
        np.testing.assert_allclose(q, want, atol=2e-7, rtol=1e-6)
        assert abs(WEIGHTS @ (q - WEIGHTS @ q)) < 2e-15
        if owner:
            # The primitive returns per-unit equity. The game loop (tested
            # below) supplies the current cube multiplier.
            np.testing.assert_array_equal(q, a.dice_values(
                state, MatchState(CubeOwner(owner), 2, jacoby)))


@pytest.mark.parametrize("owner", [0, 1, 2])
def test_cubeful_dice_values_route_bearoff_like_best_move(owner):
    # bgc_dice_values' e->bo branch: an attached table routes the 21
    # successors exactly as the engine's own move choice does.
    from modes import MatchState, CubeOwner
    try:
        e = P.Engine(str(V.DEFAULT_CF_4L), "simd", bearoff=True)
    except FileNotFoundError as exc:
        pytest.skip(f"bearoff tables not built: {exc}")
    plain = P.Engine(str(V.DEFAULT_CF_4L), "simd")
    try:
        points = [0] * 24
        points[2], points[4], points[7] = 3, 4, 2       # White: 2 checkers outside home
        points[19], points[20], points[21] = -4, -2, -3  # Black: all home
        race = BoardState(points, [0, 0], [6, 6], 0)
        ms = MatchState(CubeOwner(owner), 1 if owner == 0 else 2, True)
        for state in (race, bearoff(3), bearoff(3, 1)):
            q = e.dice_values(state, ms)
            want = [e.best_move(state, dice, ms)[2] for dice in DICE]
            np.testing.assert_allclose(q, want, atol=2e-7, rtol=1e-6)
        # The table, not the net, supplied these values (the 4L net is within ~5e-5 of exact here).
        assert np.abs(e.dice_values(bearoff(3, 1), ms)
                      - plain.dice_values(bearoff(3, 1), ms)).max() > 1e-5
    finally:
        e.close()
        plain.close()


@pytest.mark.parametrize("gammon", [2, 3])
@pytest.mark.parametrize("turn", [0, 1])
def test_cubeful_terminal_values_respect_jacoby(cubeful_engines, gammon, turn):
    from modes import MatchState, CubeOwner
    ref = cubeful_engines[2]
    points = [0] * 24
    points[0] = 1
    points[18 if gammon == 2 else 1] = -15
    state = BoardState(points, [0, 0], [14, 0], 0)
    if turn:
        state = BoardState([-x for x in reversed(points)], [0, 0], [0, 14], 1)
    for owner, jacoby, expected in ((0, True, 1), (0, False, gammon),
                                    (1, True, gammon), (2, True, gammon)):
        ms = MatchState(CubeOwner(owner), 1 if not owner else 4, jacoby)
        np.testing.assert_array_equal(ref.dice_values(state, ms), np.full(21, expected))


@pytest.mark.parametrize("jacoby", [False, True])
@pytest.mark.parametrize("every", [1, 4])
def test_cubeful_game_phase_units_and_full_replay(cubeful_engines, jacoby, every):
    from modes import MatchState, CubeOwner
    a, b, ref = cubeful_engines
    found_take = found_drop = found_pass = False
    # Deterministic seed bank exercises actual production cube policies.
    for seed in range(12):
        opts = P.game_opts(cubeful=True, jacoby=jacoby)
        r0, r1 = P.Rng(seed, "simd"), P.Rng(seed, "simd")
        base, records0 = P.play_game(a, b, opts, r0, record=True)
        vr, records = P.play_game(a, b, opts, r1, record=True, reference=ref, vr_every=every)
        assert r0.getstate() == r1.getstate()
        assert all(vr[k] == v for k, v in base.items())
        assert len(records0) == len(records)
        for old, new in zip(records0, records):
            assert {k:v for k,v in old.items() if k != "value"} == {
                k:v for k,v in new.items() if k != "value"}
            np.testing.assert_equal(old["value"], new["value"])
        rolls = [r for r in records if r["kind"] == 0]
        assert vr["n_corrected"] == len(rolls[1::every])
        expected = 0.0
        for r in rolls[1::every]:
            owner, cube, jac = r["cube"]
            ms = MatchState(CubeOwner(owner), cube, bool(jac))
            q = ref.dice_values(r["state"], ms)
            sign = 1 if r["state"].turn == 0 else -1
            k = DICE.index(tuple(sorted(r["dice"])))
            expected += sign * cube * (q[k] - WEIGHTS @ q)
        assert vr["luck_white"] == pytest.approx(expected, abs=1e-11)
        found_pass |= any(r["decision"] < 0 for r in rolls)
        for i, r in enumerate(records):
            if r["kind"] == 2 and r["decision"] == 1:
                found_take = True
                next_roll = records[i+1]
                assert next_roll["kind"] == 0
                assert next_roll["cube"][1] == 2 * r["cube"][1]
                assert next_roll["cube"][0] == 2 - r["state"].turn
        if base["ended_by_drop"]:
            found_drop = True
            assert records[-1]["kind"] == 2 and records[-1]["decision"] == 0
            assert base["result"] == 1
    assert found_take and found_drop and found_pass


def test_cubeful_invalid_context_rejected(engines, cubeful_engines):
    from modes import MatchState, CubeOwner
    a, b, ref = cubeful_engines
    for ms in (None, MatchState(CubeOwner.WHITE, 0), MatchState(CubeOwner.WHITE, 3)):
        with pytest.raises(ValueError, match="valid matching cube"):
            ref.dice_values(BoardState.initial(), ms)
    with pytest.raises(ValueError, match="matching cubeful"):
        P.play_game(a, b, P.game_opts(cubeful=True), 1, reference=engines[0])
    with pytest.raises(ValueError, match="matching cubeful"):
        P.play_game(a, b, P.game_opts(cubeful=False), 1, reference=ref)


def test_cubeful_disabled_cube_retains_centered_jacoby(cubeful_engines):
    a, b, ref = cubeful_engines
    result, records = P.play_game(a, b, P.game_opts(cubeful=True, cube_enabled=False),
                                  19, reference=ref, record=True)
    assert result["cube_value"] == 1 and result["result"] == 1
    assert not result["ended_by_drop"]
    assert all(r["cube"] == (0, 1, 1) for r in records)


def test_exact_cubeful_endgames_with_real_cube_decisions(cubeful_engines):
    """Enumerate complete small races with the production double/take/drop
    policies and verify the adjusted expectation under a poor reference.
    Memoization collapses rolls reaching the same state. Nothing is sampled.
    """
    from functools import lru_cache
    from modes import MatchState, CubeOwner
    a, b, ref = cubeful_engines
    actions = dict(takes=0, drops=0, rolls=0)

    @lru_cache(None)
    def moments(nw, nb, turn, owner, cube):
        if nw == 0 or nb == 0:
            y = cube * (1 if nw == 0 else -1)
            return y, y, 0.0
        points = [0] * 24
        points[0], points[23] = nw, -nb
        state = BoardState(points, [0, 0], [15-nw, 15-nb], turn)
        ms = MatchState(CubeOwner(owner), cube, True)
        actor, opponent = ((a, b) if turn == 0 else (b, a))
        if ms.can_offer(turn) and actor.offer_double(state, ms)[0]:
            if not opponent.respond_to_double(state, ms):
                actions["drops"] += 1
                y = cube * (1 if turn == 0 else -1)
                return y, y, 0.0  # no roll, no correction
            actions["takes"] += 1
            ms = ms.after_take(turn)
        actions["rolls"] += 1
        q = ref.dice_values(state, ms)
        center = WEIGHTS @ q
        out = np.zeros(3)
        for p, dice, value in zip(WEIGHTS, DICE, q):
            _, nxt, _ = actor.best_move(state, dice, ms)
            assert nxt is not None
            r = np.array(moments(15-nxt.off[0], 15-nxt.off[1], nxt.turn,
                                 int(ms.cube_owner), ms.cube_value))
            correction = (1 if turn == 0 else -1) * ms.cube_value * (value-center)
            r[1] -= correction
            r[2] += correction
            out += p * r
        return tuple(out)

    for n in range(3, 10):
        for owner in (0, 1, 2):
            y, z, c = moments(n, n, 0, owner, 1 if owner == 0 else 8)
            assert z == pytest.approx(y, abs=2e-12)
            assert c == pytest.approx(0, abs=2e-12)
    assert all(v > 0 for v in actions.values()), actions


@pytest.mark.parametrize("cubeful,jacoby", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize("every", [1, 4])
def test_python_roll_hook_matches_native_game(engines, cubeful_engines, cubeful, jacoby, every):
    from experiments.head_to_head.variance_external import observed_game
    a, b, ref = cubeful_engines if cubeful else engines
    aw, ab = P.CAgent(None, engine=a), P.CAgent(None, engine=b)
    for seed in (7, 19, 41):
        r0, r1, rc = [P.Rng(seed, "simd") for _ in range(3)]
        base, base_obs = observed_game(aw, ab, r0, cubeful, jacoby, record_moves=True)
        rec, obs = observed_game(aw, ab, r1, cubeful, jacoby, ref, every, record_moves=True)
        native, _ = P.play_game(a, b, P.game_opts(cubeful=cubeful, jacoby=jacoby),
                                rc, reference=ref, vr_every=every)
        assert rec == base
        assert obs.trace.digest() == base_obs.trace.digest()
        assert r0.getstate() == r1.getstate() == rc.getstate()
        assert (rec.winner, rec.result, rec.cube_value, rec.ended_by_drop) == (
            native['winner'], native['result'], native['cube_value'], native['ended_by_drop'])
        assert obs.turns == native['n_turns']
        assert obs.corrected == native['n_corrected']
        assert obs.luck == pytest.approx(native['luck_white'], abs=1e-10)


@pytest.mark.parametrize("field,value", [("plies", [0, 2]),
                                         ("external", {"version": "other"}),
                                         ("cpu_accounting", "different")])
def test_pooling_rejects_different_external_configuration(tmp_path, field, value):
    meta = dict(schema=V.SCHEMA, seed=12, plies=[0, 0], external=None,
                cpu_accounting="worker-process")
    paths = []
    for i, m in enumerate((meta, {**meta, field: value})):
        p = tmp_path / f'{i}.npz'
        np.savez(p, **fake_data(4), metadata=json.dumps(m), pair_id=np.arange(i*4,i*4+4))
        paths.append(p)
    with pytest.raises(ValueError, match="different models"):
        V.load_runs(paths)


@pytest.mark.parametrize("cubeful", [False, True])
def test_live_gnubg_two_ply_variance_chunk(cubeful):
    import time
    from third_party_opponents.gnubg import gnubg_eval as G
    from experiments.head_to_head.variance_external import ExternalRunner, preflight
    if G._gnubg_version(G.GNUBG_CMD) != G.GNUBG_REQUIRED_VERSION:
        pytest.skip('live GNU Backgammon variance test requires pinned 1.08.003')
    provenance = preflight(2)
    assert provenance['version'] == '1.08.003'
    model = V.DEFAULT_CF_4L if cubeful else V.DEFAULT_4L
    path = P.export_bin(str(model))
    runner = ExternalRunner([path, 'gnubg-ext', path], 'simd', 1, cubeful, True, 2)
    try:
        c0 = time.process_time()
        _, d = runner.chunk((20261031, 0, 1), V.pair_seed)
        own_cpu = time.process_time()-c0
        np.testing.assert_array_equal(d['corrected'], d['turns']-1)
        assert np.isfinite(d['luck']).all()
        # CPU statistics must include the live GNU process, not just Python.
        assert d['cpu_seconds'].sum() > own_cpu
        # Each mode has its own GNU process: no cross-mode cache reuse.
        pids = [a._server.proc.pid for agents in runner.agents for a in agents
                if getattr(a, '_server', None) is not None]
        assert len(pids) == len(set(pids)) == 2
        assert d['gnubg_cube_clamps'].sum() == 0
    finally:
        runner.close()


@pytest.mark.parametrize("cubeful", [False, True], ids=["cubeless", "cubeful"])
@pytest.mark.parametrize("plies", [(1, 0), (0, 1), (1, 1)])
def test_correction_uses_only_pre_roll_state_at_one_ply(engines, cubeful_engines, cubeful, plies):
    """With 1-ply players each roll's correction is still the reference's
    value at the recorded pre-roll position, cube and roll, and the game
    is unchanged by the correction."""
    from modes import MatchState, CubeOwner
    a, b, ref = cubeful_engines if cubeful else engines
    opts = P.game_opts(cubeful=cubeful, plies_white=plies[0], plies_black=plies[1])
    for seed in (5, 23):
        r0, r1 = P.Rng(seed, "simd"), P.Rng(seed, "simd")
        base, records0 = P.play_game(a, b, opts, r0, record=True)
        vr, records = P.play_game(a, b, opts, r1, record=True, reference=ref)
        assert r0.getstate() == r1.getstate()
        assert all(vr[k] == v for k, v in base.items())
        assert [r["state"] for r in records0] == [r["state"] for r in records]
        expected = 0.0
        rolls = [r for r in records if r["kind"] == 0]
        for r in rolls[1:]:
            ms = MatchState(CubeOwner(r["cube"][0]), r["cube"][1], bool(r["cube"][2])) \
                if cubeful else None
            q = ref.dice_values(r["state"], ms)
            k = DICE.index(tuple(sorted(r["dice"])))
            sign = 1 if r["state"].turn == 0 else -1
            expected += sign * (ms.cube_value if ms else 1) * (q[k] - WEIGHTS @ q)
        assert vr["n_corrected"] == len(rolls) - 1
        assert vr["luck_white"] == pytest.approx(expected, abs=1e-11)


def test_chunk_plays_native_sides_at_their_ply(engines):
    """variance.py's chunk at native plies (1, 0): side1 searches 1-ply in
    both seats, and the luck is bgc_play_game_vr's at those plies."""
    a, b, ref = engines
    V._ENGINES = (a, b, ref, 1, False, True, (1, 0))
    _, d = V._chunk((8, 0, 2))
    for i, pid in enumerate(range(2)):
        for leg, (w, bl, pw, pb) in enumerate(((a, b, 1, 0), (b, a, 0, 1))):
            res, _ = P.play_game(w, bl, P.game_opts(cubeful=False, plies_white=pw,
                                                    plies_black=pb),
                                 P.Rng(V.pair_seed(8, pid), "simd"), reference=ref)
            sign = 1 if leg == 0 else -1
            assert d["luck"][i, leg] == pytest.approx(sign * res["luck_white"], abs=1e-12)
            assert d["turns"][i, leg] == res["n_turns"]
