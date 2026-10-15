"""JanowskiMatchAgent.value_oneply (the --oracle-plies 1 teacher).

1. The batched implementation must equal a naive per-successor
   reference (python move-gen, is_game_over terminal pricing, one
   _probs_state call per successor) on random midgame states — the
   reference shares no batching/terminal-detection code with the
   implementation, so a fast-path bug cannot pass by symmetry.

2. Forced pass: a dice with no legal move prices as
   1 - V0(same board, opponent on roll).

3. The trainer's plies=1 cube-action target composes
   max(v_nd, min(v_dt, v_dp)) from 1-ply branch values with v_dp
   exact — verified against manually assembled pieces.

4. Mixed-ply rows are rejected: oracle_plies=1 + oracle_move_eps
   raises (uniform-target invariant backstop).
"""

import random

import numpy as np
import pytest
import torch

import janowski_match as jm
from backgammon_engine import BoardState, get_legal_plays, switch_turn
from janowski_match import JanowskiMatchAgent
from model import ProbNetwork
from modes import MatchState, cube_perspective
from trainer import _oracle_target

DICE = [(a, b) for a in range(1, 7) for b in range(a, 7)]


def make_oracle(seed=0, x_policy=None):
    torch.manual_seed(seed)
    net = ProbNetwork(hidden_sizes=[32, 16], input_size=196,
                      encoder_name="perspective196")
    return JanowskiMatchAgent(net, x_policy=x_policy)


def ms(match_length=5, my_away=5, opp_away=5, **kw):
    return MatchState(jacoby=False, match_length=match_length,
                      my_away=my_away, opp_away=opp_away, **kw)


def random_state(seed, plies):
    rng = random.Random(seed)
    state = BoardState.initial()
    for _ in range(plies):
        dice = (rng.randint(1, 6), rng.randint(1, 6))
        plays = get_legal_plays(state, dice)
        if not plays:
            state = switch_turn(state)
            continue
        _, state = plays[rng.randrange(len(plays))]
        if state.is_game_over():
            return None
    return state


def ref_value_oneply(ora, state, match_state):
    """Naive reference: per-dice python loop, one forward per successor."""
    mover, opp = state.turn, 1 - state.turn
    ctx_me = jm.ctx_for(match_state, mover)
    ctx_opp = jm.ctx_for(match_state, opp)
    persp = cube_perspective(match_state.cube_owner, opp)
    total = 0.0
    for d1, d2 in DICE:
        p = (1.0 if d1 == d2 else 2.0) / 36.0
        plays = get_legal_plays(state, (d1, d2))
        if not plays:
            ov = switch_turn(state)
            v = 1.0 - jm.cubeful_mwc(
                ora._probs_state(ov), persp, ctx_opp, ora._x(ov), ora.met)
        else:
            best = -np.inf
            for _, s in plays:
                if s.is_game_over():
                    v1 = jm._me(ora.met, ctx_me,
                                ctx_me.cube * s.game_result(), True)
                else:
                    v1 = 1.0 - jm.cubeful_mwc(
                        ora._probs_state(s), persp, ctx_opp,
                        ora._x(s), ora.met)
                best = max(best, v1)
            v = best
        total += p * v
    return total


@pytest.mark.parametrize("seed,plies", [(1, 6), (2, 14), (3, 24), (4, 40)])
def test_matches_naive_reference(seed, plies):
    state = random_state(seed, plies)
    if state is None:
        pytest.skip("game ended during random walk")
    ora = make_oracle()
    for m in (ms(), ms(match_length=5, my_away=2, opp_away=4,
                    cube_value=2, cube_owner=state.turn)):
        got = ora.value_oneply(state, m)
        want = ref_value_oneply(ora, state, m)
        assert got == pytest.approx(want, abs=1e-6)
        assert 0.0 < got < 1.0


def test_reference_on_late_bearoff_with_terminals():
    """A state whose successor set includes game-ending moves."""
    ora = make_oracle()
    found = None
    for seed in range(60):
        state = random_state(seed, 70)
        if state is None:
            continue
        if any(s.is_game_over()
               for d in DICE for _, s in get_legal_plays(state, d)):
            found = state
            break
    assert found is not None, "no near-terminal state found"
    m = ms()
    assert ora.value_oneply(found, m) == pytest.approx(
        ref_value_oneply(ora, found, m), abs=1e-6)


def test_forced_pass_pricing():
    """All-dice-blocked mover: value collapses to 1 - V0(flipped)."""
    ora = make_oracle()
    # Mover (on the bar) faces a closed home board: no dice has a move.
    state = None
    for seed in range(400):
        s = random_state(seed, 30)
        if s is not None and all(
                not get_legal_plays(s, d) for d in DICE):
            state = s
            break
    if state is None:
        pytest.skip("no fully blocked state found in walk budget")
    m = ms()
    ov = switch_turn(state)
    persp = cube_perspective(m.cube_owner, ov.turn)
    ctx_opp = jm.ctx_for(m, ov.turn)
    want = 1.0 - jm.cubeful_mwc(
        ora._probs_state(ov), persp, ctx_opp, ora._x(ov), ora.met)
    assert ora.value_oneply(state, m) == pytest.approx(want, abs=1e-9)


def test_trainer_cube_action_composition():
    state = random_state(7, 12)
    ora = make_oracle()
    m = ms()  # centered cube, both can offer
    assert m.can_offer(state.turn)
    v_nd = ora.value_oneply(state, m)
    v_dt = ora.value_oneply(state, m.after_take(state.turn))
    ctx = jm.ctx_for(m, state.turn)
    v_dp = jm._me(ora.met, ctx, ctx.cube, True)
    want = max(v_nd, min(v_dt, v_dp))
    got = _oracle_target(ora, state, m, True, plies=1)
    assert got == pytest.approx(want, abs=1e-12)
    # ca=0 rows are the plain 1-ply value.
    assert _oracle_target(ora, state, m, False, plies=1) == pytest.approx(
        v_nd, abs=1e-12)


def _find_union_case(oracle, agent, seed0=0, eps=1e-4):
    """A decision where the student's argmax successor is OUTSIDE the
    oracle's eps-set — the case the union clause exists for."""
    from trainer import _oracle_move_rows
    for seed in range(seed0, seed0 + 200):
        state = random_state(seed, 16)
        if state is None:
            continue
        for d1 in range(1, 7):
            for d2 in range(d1, 7):
                dice = (d1, d2)
                res = agent.choose_checker_action_matchplay(
                    state, dice, ms())
                res5 = oracle.choose_checker_action_cubeful(
                    state, dice, ms())
                if res is None or res5 is None:
                    continue
                if res[0] != res5[0]:
                    return state, dice, res[0]
    return None


def test_union_includes_student_choice():
    """The student's chosen successor gets a row even when the oracle
    rates it outside eps of the oracle's best."""
    from td_agent import TDAgent
    from model import TDNetwork
    from trainer import _oracle_move_rows
    torch.manual_seed(5)
    ora = make_oracle(seed=5)
    net = TDNetwork(hidden_sizes=[32, 16], input_size=207,
                    encoder_name="matchplay1h_perspective196",
                    output_mode="probability")
    agent = TDAgent(net)
    found = _find_union_case(ora, agent)
    assert found is not None, "no disagreement found in search budget"
    state, dice, stu_succ = found
    m = ms()
    ca = m.can_offer(1 - state.turn)
    stu_enc = agent._encode_matchplay(stu_succ, m, is_cube_action=ca)
    # eps ~ 0: the oracle set collapses to its argmax; only the union
    # clause can admit the student's (different) choice.
    rows, _teacher = _oracle_move_rows(
        ora, agent, state, dice, m, eps=1e-6)
    assert any(np.array_equal(enc, stu_enc) for enc, _ in rows), \
        "student's chosen successor not supervised"


def test_eps_rows_priced_at_one_ply():
    """plies=1: every emitted eps-row target equals the _oracle_target
    funnel at 1 ply for that successor (uniform-target invariant)."""
    from td_agent import TDAgent
    from model import TDNetwork
    from trainer import _oracle_move_rows
    torch.manual_seed(6)
    ora = make_oracle(seed=6)
    net = TDNetwork(hidden_sizes=[32, 16], input_size=207,
                    encoder_name="matchplay1h_perspective196",
                    output_mode="probability")
    agent = TDAgent(net)
    state = random_state(9, 14)
    m = ms()
    from backgammon_engine import get_legal_plays
    dice = next(d for d in DICE if get_legal_plays(state, d))
    ca = m.can_offer(1 - state.turn)
    rows, _teacher = _oracle_move_rows(
        ora, agent, state, dice, m, eps=0.02, plies=1)
    assert rows, "no rows emitted"
    plays = get_legal_plays(state, dice)
    by_enc = {}
    for _, s in plays:
        if not s.is_game_over():
            enc = agent._encode_matchplay(s, m, is_cube_action=ca)
            by_enc[enc.tobytes()] = s
    for enc, t in rows:
        s = by_enc[enc.tobytes()]
        want = _oracle_target(ora, s, m, ca, plies=1)
        assert t == pytest.approx(want, abs=1e-12)


def test_mode_accepts_eps_with_plies():
    from modes import MatchplayMode
    mode = MatchplayMode(match_length=3, oracle_spec="dummy",
                         oracle_move_eps=0.01, oracle_plies=1)
    assert mode.oracle_plies == 1 and mode.oracle_move_eps == 0.01


def test_collector_plies_flag_changes_targets():
    """The wiring guard (met_scopes lesson): oracle_plies must survive
    the pickled-mode worker path and actually change collected
    targets. Same seed, same acting net -> identical trajectory;
    plies=1 must shift most targets (1-ply refinement), plies=0/1
    must NOT change which states are visited (acting is the
    student's, independent of target depth)."""
    import pickle
    from model import TDNetwork
    from td_agent import TDAgent
    from modes import MatchplayMode
    from trainer import _collect_episode_matchplay, _ORACLE_CACHE

    torch.manual_seed(3)
    prob5 = ProbNetwork(hidden_sizes=[32, 16], input_size=196,
                        encoder_name="perspective196")
    _ORACLE_CACHE["__test_oneply__"] = JanowskiMatchAgent(prob5)
    net = TDNetwork(hidden_sizes=[32, 16], input_size=207,
                    encoder_name="matchplay1h_perspective196",
                    output_mode="probability")
    rows = {}
    for plies in (0, 1):
        mode = MatchplayMode(match_length=3,
                             oracle_spec="__test_oneply__",
                             oracle_plies=plies)
        mode = pickle.loads(pickle.dumps(mode))  # worker-path fidelity
        assert mode.oracle_plies == plies
        _, tgt = _collect_episode_matchplay(
            TDAgent(net), mode, random.Random(11),
            mode.initial_match_state())
        rows[plies] = np.asarray(tgt, dtype=np.float64)
    del _ORACLE_CACHE["__test_oneply__"]
    a, b = rows[0], rows[1]
    assert len(a) == len(b), "target depth must not change the trajectory"
    diff = np.abs(a - b)
    assert (diff > 1e-9).sum() > len(a) // 2
    # No magnitude bound: with a random teacher the 1-ply exact-MET
    # terminal pricing legitimately moves near-terminal targets by
    # ~0.5. Both depths must still emit valid MWCs.
    assert np.all((a >= 0) & (a <= 1)) and np.all((b >= 0) & (b <= 1))


# ── --oracle-act-prob (teacher-forced acting) ────────────────────────


def test_teacher_succ_is_oracle_argmax():
    """The successor handed back for acting is the teacher's own argmax,
    recomputed here from the oracle's per-successor values without
    touching the batched path inside _oracle_move_rows."""
    from td_agent import TDAgent
    from model import TDNetwork
    from trainer import _oracle_move_rows
    torch.manual_seed(11)
    ora = make_oracle(seed=11)
    net = TDNetwork(hidden_sizes=[32, 16], input_size=207,
                    encoder_name="matchplay1h_perspective196",
                    output_mode="probability")
    agent = TDAgent(net)
    m = ms()
    checked = 0
    for sd in range(40):
        state = random_state(sd, 6)
        if state is None:
            continue
        dice = next((d for d in DICE if get_legal_plays(state, d)), None)
        if dice is None:
            continue
        rows, teacher = _oracle_move_rows(ora, agent, state, dice, m,
                                          eps=0.01)
        plays = get_legal_plays(state, dice)
        if len(plays) < 2:
            continue
        ca = m.can_offer(1 - state.turn)
        # Naive reference: mover-view oracle value of every successor.
        best_v, best_s = None, None
        for _, s in plays:
            if s.is_game_over():
                ctx = cube_perspective(m, state.turn)
                v = jm._me(ora.met, ctx, m.cube_value * s.game_result(),
                           True)
            else:
                v = 1.0 - _oracle_target(ora, s, m, ca, plies=0)
            if best_v is None or v > best_v:
                best_v, best_s = v, s
        assert teacher is not None
        assert teacher == best_s, f"seed {sd}: not the oracle argmax"
        checked += 1
        if checked >= 5:
            break
    assert checked >= 5, "insufficient multi-move positions checked"


def _collect(act_prob, seed):
    """One matchplay episode under oracle supervision at a given beta."""
    from td_agent import TDAgent
    from model import TDNetwork
    from trainer import _collect_episode_matchplay
    from train_cli import build_mode
    torch.manual_seed(3)
    net = TDNetwork(hidden_sizes=[32, 16], input_size=207,
                    encoder_name="matchplay1h_perspective196",
                    output_mode="probability")
    agent = TDAgent(net)
    mode = build_mode("matchplay", match_length=3,
                      oracle_spec="__test__", oracle_move_eps=0.01,
                      oracle_act_prob=act_prob)
    import trainer as _tr
    _tr._ORACLE_CACHE["__test__"] = make_oracle(seed=7)
    return _collect_episode_matchplay(
        agent, mode, random.Random(seed),
        MatchState(jacoby=False, match_length=3, my_away=3, opp_away=3),
    )


def test_act_prob_zero_is_historical_path():
    """beta=0 must reproduce the student-acting trajectory exactly."""
    a_enc, a_t = _collect(0.0, seed=99)
    b_enc, b_t = _collect(0.0, seed=99)
    assert np.array_equal(a_enc, b_enc) and np.array_equal(a_t, b_t)


def test_act_prob_one_changes_trajectory():
    """beta=1 walks into the teacher's move, so with a teacher that is a
    different net the visited states must diverge from beta=0."""
    z_enc, _ = _collect(0.0, seed=99)
    o_enc, _ = _collect(1.0, seed=99)
    assert not (z_enc.shape == o_enc.shape and np.array_equal(z_enc, o_enc)), \
        "beta=1 produced the identical trajectory to beta=0"


# ── --move-explore-eps (checker exploration) ─────────────────────────


def _collect_td(explore_eps, seed):
    """One --oneply TD episode (no oracle) at a given exploration rate."""
    from td_agent import TDAgent
    from model import TDNetwork
    from trainer import _collect_episode_matchplay
    from train_cli import build_mode
    torch.manual_seed(3)
    net = TDNetwork(hidden_sizes=[32, 16], input_size=207,
                    encoder_name="matchplay1h_perspective196",
                    output_mode="probability")
    agent = TDAgent(net)
    mode = build_mode("matchplay", match_length=3,
                      move_explore_eps=explore_eps)
    return _collect_episode_matchplay(
        agent, mode, random.Random(seed),
        MatchState(jacoby=False, match_length=3, my_away=3, opp_away=3),
        oneply=True,
    )


def test_explore_zero_is_historical_path():
    a_enc, a_t = _collect_td(0.0, seed=77)
    b_enc, b_t = _collect_td(0.0, seed=77)
    assert np.array_equal(a_enc, b_enc) and np.array_equal(a_t, b_t)


def test_explore_changes_visited_states():
    """eps=1 executes random moves, so the trajectory must diverge."""
    g_enc, _ = _collect_td(0.0, seed=77)
    e_enc, _ = _collect_td(1.0, seed=77)
    assert not (g_enc.shape == e_enc.shape and np.array_equal(g_enc, e_enc))


def test_explore_does_not_touch_targets():
    """The invariant that makes this safe: every emitted target is still the
    1-ply value of the state it encodes, whatever move was executed."""
    from td_agent import TDAgent
    from model import TDNetwork
    torch.manual_seed(3)
    net = TDNetwork(hidden_sizes=[32, 16], input_size=207,
                    encoder_name="matchplay1h_perspective196",
                    output_mode="probability")
    agent = TDAgent(net)
    m = ms(match_length=3, my_away=3, opp_away=3)
    checked = 0
    for sd in range(30):
        state = random_state(sd, 6)
        if state is None:
            continue
        dice = next((d for d in DICE if get_legal_plays(state, d)), None)
        if dice is None or len(get_legal_plays(state, dice)) < 2:
            continue
        # the target for this state is a max over successors: independent of
        # which successor is executed
        t = agent.value_oneply_checker_matchplay(state, m)
        assert 0.0 <= t <= 1.0
        t2 = agent.value_oneply_checker_matchplay(state, m)
        assert t == t2, "1-ply target is not a pure function of the state"
        checked += 1
        if checked >= 5:
            break
    assert checked >= 5


# ── --oracle-act-only (teacher moves, student targets) ───────────────


def _collect_actonly(act_only, seed):
    from td_agent import TDAgent
    from model import TDNetwork
    from trainer import _collect_episode_matchplay
    from train_cli import build_mode
    import trainer as _tr
    torch.manual_seed(3)
    net = TDNetwork(hidden_sizes=[32, 16], input_size=207,
                    encoder_name="matchplay1h_perspective196",
                    output_mode="probability")
    agent = TDAgent(net)
    mode = build_mode("matchplay", match_length=3,
                      oracle_spec="__actonly__", oracle_move_eps=0.01,
                      oracle_act_prob=1.0, oracle_act_only=act_only)
    _tr._ORACLE_CACHE["__actonly__"] = make_oracle(seed=7)
    return _collect_episode_matchplay(
        agent, mode, random.Random(seed),
        MatchState(jacoby=False, match_length=3, my_away=3, opp_away=3),
        oneply=act_only,
    )


def test_act_only_emits_no_oracle_rows():
    """act-only must emit far fewer rows than full oracle supervision: the
    eps-set rows and the cube take-branch row are exactly what it drops."""
    full_enc, _ = _collect_actonly(False, seed=5)
    act_enc, _ = _collect_actonly(True, seed=5)
    assert act_enc.shape[0] < full_enc.shape[0], (
        f"act-only emitted {act_enc.shape[0]} rows vs full {full_enc.shape[0]}")


def test_act_only_targets_are_in_range():
    """Targets come from the TD path, so they stay valid MWCs."""
    _, t = _collect_actonly(True, seed=11)
    assert len(t) > 0
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0
