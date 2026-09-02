"""Tests for matchplay: MatchState match rules, MatchplayEncoder,
TDAgent matchplay methods, the matchplay collector, and the
DMP -> matchplay warm start.

The perspective and state-transition rules here encode the pitfalls
found and fixed during the original matchplay prototype (score
auto-flip for BLACK, Crawford one-shot latch, live-cube score updates,
terminal target = 1.0 from the mover's perspective, no money logic in
targets).
"""

import random

import numpy as np
import pytest
import torch

from backgammon_engine import BoardState, switch_turn
from encoding import CubefulEncoder, CubePerspective, MatchplayEncoder, get_encoder
from model import TDNetwork
from modes import CubeOwner, MatchState, MatchplayMode
from td_agent import TDAgent
from trainer import collect_episode


def _matchplay_net(hidden=(16,), seed=0):
    torch.manual_seed(seed)
    return TDNetwork(
        hidden_sizes=list(hidden),
        encoder_name="matchplay_perspective196",
        output_mode="probability",
    )


def _agent(seed=0):
    return TDAgent(_matchplay_net(seed=seed))


# ── MatchState match rules ──────────────────────────────────────────────


class TestMatchStateRules:
    def test_crawford_fires_once(self):
        ms = MatchplayMode(2).initial_match_state()
        g1 = ms.start_game()
        assert not g1.is_crawford and not g1.crawford_done
        ms = g1.after_game(1, winner_is_me=True)  # -> (1, 2)
        assert (ms.my_away, ms.opp_away) == (1, 2)
        g2 = ms.start_game()
        assert g2.is_crawford and g2.crawford_done
        assert not g2.can_offer(0) and not g2.can_offer(1)
        ms = g2.after_game(1, winner_is_me=False)  # -> (1, 1)
        g3 = ms.start_game()
        assert not g3.is_crawford, "Crawford must fire exactly once"
        # And it stays off for any later game.
        g4 = g3.after_game(0 or 1, winner_is_me=True).start_game()
        assert not g4.is_crawford

    def test_dead_cube_rule(self):
        # (1,1) post-Crawford: cube 1 >= 1-away for both -> dead.
        ms = MatchState(jacoby=False, match_length=2, my_away=1,
                        opp_away=1, crawford_done=True)
        assert not ms.can_offer(0) and not ms.can_offer(1)
        # (2,1): WHITE 2-away may double at cube 1; BLACK 1-away may not.
        ms = MatchState(jacoby=False, match_length=2, my_away=2,
                        opp_away=1, crawford_done=True)
        assert ms.can_offer(0)
        assert not ms.can_offer(1)
        # After a take at cube 2, nobody 2-away may redouble.
        ms = MatchState(jacoby=False, match_length=3, my_away=2,
                        opp_away=2, crawford_done=False)
        taken = ms.after_take(0)  # cube 2, BLACK owns
        assert not taken.can_offer(1), "redouble at cube >= away is dead"

    def test_after_game_uses_live_cube(self):
        # Doubled gammon at (3,3) must award 4 points, ending the match.
        ms = MatchState(jacoby=False, match_length=3, my_away=3, opp_away=3)
        g = ms.start_game().after_take(0)  # WHITE doubled, BLACK took
        assert g.cube_value == 2
        done = g.after_game(2, winner_is_me=True)  # gammon x cube 2 = 4
        assert done.my_away == 0 and done.match_over()

    def test_money_match_state_unaffected(self):
        ms = MatchState(jacoby=True)  # money: match_length=0
        assert not ms.is_matchplay
        assert ms.can_offer(0) and ms.can_offer(1)  # no dead-cube rule


# ── Encoder ─────────────────────────────────────────────────────────────


class TestMatchplayEncoder:
    def test_layout(self):
        enc = get_encoder("matchplay_perspective196")
        assert isinstance(enc, MatchplayEncoder)
        assert enc.num_features == 204
        s = BoardState.initial()
        x = enc.encode(s, CubePerspective.MINE, cube_value=2,
                       my_away=2, opp_away=1, is_crawford=True,
                       is_cube_action=True)
        assert x.shape == (204,)
        assert x[196 + 1] == 1.0 and x[196] == 0.0 and x[196 + 2] == 0.0
        assert x[199] == 1.0   # is_cube_action
        assert x[200] == 2.0   # cube_value
        assert x[201] == 2.0   # my_away
        assert x[202] == 1.0   # opp_away
        assert x[203] == 1.0   # is_crawford

    def test_first_200_match_cubeful_layout(self):
        """Bit-for-bit compatibility with CubefulEncoder on the shared
        prefix — keeps money-model warm starts possible later."""
        mp_enc = MatchplayEncoder()
        cf_enc = CubefulEncoder()
        s = BoardState.initial()
        for persp in CubePerspective:
            for action in (False, True):
                a = mp_enc.encode(s, persp, 1, 2, 2, False,
                                  is_cube_action=action)
                b = cf_enc.encode(s, persp, is_cube_action=action)
                assert np.array_equal(a[:200], b)

    def test_not_cubeful_instance(self):
        assert not isinstance(MatchplayEncoder(), CubefulEncoder)


# ── Agent perspective handling ──────────────────────────────────────────


class TestAgentPerspective:
    def test_away_scores_flip_for_black(self):
        agent = _agent()
        ms = MatchState(jacoby=False, match_length=3, my_away=3, opp_away=1,
                        cube_owner=CubeOwner.WHITE, cube_value=2,
                        crawford_done=True)
        s_white = BoardState.initial()           # WHITE on roll
        s_black = switch_turn(BoardState.initial())  # BLACK on roll
        x_w = agent._encode_matchplay(s_white, ms)
        x_b = agent._encode_matchplay(s_black, ms)
        # WHITE viewer: my=3, opp=1; cube MINE.
        assert x_w[201] == 3.0 and x_w[202] == 1.0
        assert x_w[196 + int(CubePerspective.MINE)] == 1.0
        # BLACK viewer: my=1, opp=3; cube THEIRS.
        assert x_b[201] == 1.0 and x_b[202] == 3.0
        assert x_b[196 + int(CubePerspective.THEIRS)] == 1.0

    def test_eval_new_game_player_symmetry(self):
        agent = _agent()
        ms = MatchState(jacoby=False, match_length=2, my_away=2, opp_away=1,
                        crawford_done=True)
        p_white = agent._eval_new_game(ms, 0)
        p_black = agent._eval_new_game(ms, 1)
        assert abs((p_white + p_black) - 1.0) < 1e-6
        # Mirrored scores from the other player's seat give the same number.
        ms_flip = MatchState(jacoby=False, match_length=2, my_away=1,
                             opp_away=2, crawford_done=True)
        assert abs(agent._eval_new_game(ms_flip, 1) - p_white) < 1e-6

    def test_symmetric_score_is_half_for_symmetric_net(self):
        """At a symmetric score the averaged new-game eval of any net
        is exactly 0.5 + (its own asymmetry); for a net with zeroed
        match columns the two on-roll evals are equal so the average
        is the on-roll-bias-free value — and P_white must equal 0.5
        by the symmetry of the construction (V + (1-V))/2 ... = 0.5
        only when V(white view) == V(black view) of the mirrored
        initial position. The initial position IS mirror-symmetric, so
        any perspective encoder gives identical features and the
        average is exactly 0.5."""
        agent = _agent(seed=123)
        ms = MatchState(jacoby=False, match_length=2, my_away=2, opp_away=2)
        assert abs(agent._eval_new_game(ms, 0) - 0.5) < 1e-6


# ── Cube decision logic (scripted values) ───────────────────────────────


class _ScriptedAgent(TDAgent):
    """Overrides evaluations with scripted values to pin down the
    decision rules independent of network weights."""

    def __init__(self, v_nd, v_dt, v_new_game):
        super().__init__(_matchplay_net())
        self._v_nd = v_nd
        self._v_dt = v_dt
        self._v_new = v_new_game

    def evaluate_matchplay(self, state, match_state, is_cube_action=False):
        # after_take transfers the cube to the opponent of the doubler.
        if match_state.cube_owner != CubeOwner.CENTERED and \
                match_state.cube_value > 1:
            return self._v_dt
        return self._v_nd

    def _eval_new_game(self, match_state, player, scope="cube"):
        # `scope` names the calling consumer (see TDAgent._eval_new_game
        # and --met-scope). The scripted value is scope-independent.
        return self._v_new


class TestCubeDecisions:
    def _ms(self):
        return MatchState(jacoby=False, match_length=3, my_away=3,
                          opp_away=3)

    def test_double_when_both_responses_better(self):
        # min(v_dt=0.75, v_dp=0.70) = 0.70 > v_nd=0.60 -> double.
        a = _ScriptedAgent(0.60, 0.75, 0.70)
        offer = a.offer_double_matchplay(BoardState.initial(), self._ms())
        assert offer.should_double
        # Responder: take iff v_dp > v_dt -> 0.70 > 0.75 is False -> pass.
        assert a.respond_to_double_matchplay(
            BoardState.initial(), self._ms(), hint=offer) is False

    def test_no_double_when_take_punishes(self):
        # min(v_dt=0.55, v_dp=0.70) = 0.55 < v_nd=0.60 -> no double.
        a = _ScriptedAgent(0.60, 0.55, 0.70)
        offer = a.offer_double_matchplay(BoardState.initial(), self._ms())
        assert not offer.should_double

    def test_responder_takes_when_take_better(self):
        # v_dp=0.80 > v_dt=0.65 -> taking is better for the responder.
        a = _ScriptedAgent(0.60, 0.65, 0.80)
        offer = a.offer_double_matchplay(BoardState.initial(), self._ms())
        assert offer.should_double  # min(0.65, 0.80) = 0.65 > 0.60
        assert a.respond_to_double_matchplay(
            BoardState.initial(), self._ms(), hint=offer) is True

    def test_recompute_matches_hint_path(self):
        a = _ScriptedAgent(0.60, 0.65, 0.80)
        ms = self._ms()
        offer = a.offer_double_matchplay(BoardState.initial(), ms)
        with_hint = a.respond_to_double_matchplay(
            BoardState.initial(), ms, hint=offer)
        without_hint = a.respond_to_double_matchplay(
            BoardState.initial(), ms, hint=None)
        assert with_hint == without_hint

    def test_no_offer_when_ineligible(self):
        a = _ScriptedAgent(0.10, 0.90, 0.90)
        ms = MatchState(jacoby=False, match_length=2, my_away=1,
                        opp_away=2, is_crawford=True, crawford_done=True)
        offer = a.offer_double_matchplay(BoardState.initial(), ms)
        assert not offer.should_double and offer._cache is None


# ── Checker selection: terminal override ────────────────────────────────


class TestCheckerTerminal:
    def test_match_winning_gammon_is_chosen_with_target_1(self):
        """WHITE bears off the last checker for a gammon at (2,2):
        2 points -> match over -> the move's value is exactly 1.0."""
        agent = _agent()
        pts = [0] * 24
        pts[0] = 1       # WHITE's last checker on the 1-point
        pts[23] = -15    # BLACK hasn't borne off (gammon)
        s = BoardState(points=pts, bar=[0, 0], off=[14, 0], turn=0)
        ms = MatchState(jacoby=False, match_length=2, my_away=2, opp_away=2)
        result = agent.choose_checker_action_matchplay(s, (1, 2), ms)
        assert result is not None
        next_state, bootstrap = result
        assert next_state.is_game_over()
        assert next_state.game_result() >= 2  # gammon
        assert bootstrap == 1.0

    def test_oneply_value_all_dice_gammon_at_2away(self):
        """Every dice bears off WHITE's last checker for a gammon at
        (2,2) -> match over on every branch -> 1-ply value is exactly
        1.0 regardless of the network."""
        agent = _agent()
        pts = [0] * 24
        pts[0] = 1       # WHITE's last checker on the 1-point
        pts[23] = -15    # BLACK hasn't borne off (gammon)
        s = BoardState(points=pts, bar=[0, 0], off=[14, 0], turn=0)
        ms = MatchState(jacoby=False, match_length=2, my_away=2, opp_away=2)
        v = agent.value_oneply_checker_matchplay(s, ms)
        assert v == 1.0

    def test_oneply_value_plain_win_prices_new_game_mwc(self):
        """Every dice wins a PLAIN game (opponent has borne off) at
        (3,3) cube 1 -> score becomes (2,3) -> the 1-ply value equals
        the agent's own new-game MWC at (2,3) exactly."""
        agent = _agent()
        pts = [0] * 24
        pts[0] = 1       # WHITE's last checker on the 1-point
        pts[23] = -14    # BLACK has borne off one checker: plain loss
        s = BoardState(points=pts, bar=[0, 0], off=[14, 1], turn=0)
        ms = MatchState(jacoby=False, match_length=3, my_away=3, opp_away=3)
        v = agent.value_oneply_checker_matchplay(s, ms)
        new_ms = ms.after_game(1, winner_is_me=True)
        assert abs(v - agent._eval_new_game(new_ms, 0)) < 1e-6

    def test_oneply_value_range_initial(self):
        agent = _agent()
        from backgammon_engine import opening_roll
        state, _ = opening_roll(random.Random(3))
        ms = MatchState(jacoby=False, match_length=3, my_away=3, opp_away=3)
        v = agent.value_oneply_checker_matchplay(state, ms)
        assert 0.0 <= v <= 1.0


# ── Collector ───────────────────────────────────────────────────────────


class TestMatchplayCollector:
    def test_episode_shapes_and_target_range(self):
        agent = _agent()
        mode = MatchplayMode(2)
        mode.validate_agent(agent)
        rng = random.Random(11)
        for _ in range(3):
            enc, targets = collect_episode(agent, mode, rng)
            assert enc.shape[1] == 204
            assert enc.shape[0] == targets.shape[0] > 0
            assert np.all(targets >= 0.0) and np.all(targets <= 1.0)

    def test_oneply_accepted(self):
        agent = _agent()
        mode = MatchplayMode(2)
        enc, targets = collect_episode(
            agent, mode, random.Random(1), oneply=True,
        )
        assert enc.shape[0] == targets.shape[0] > 0
        assert np.all(targets >= 0.0) and np.all(targets <= 1.0)

    def test_validate_rejects_money_agent(self):
        torch.manual_seed(0)
        money_net = TDNetwork(hidden_sizes=[8],
                              encoder_name="cubeful_perspective196",
                              output_mode="equity")
        with pytest.raises(ValueError):
            MatchplayMode(2).validate_agent(TDAgent(money_net))

    def test_equity_matchplay_agent_rejected(self):
        # The agent-level check rejects an equity net paired with a
        # MatchplayEncoder (P(win match) must be a probability).
        torch.manual_seed(0)
        net = TDNetwork(hidden_sizes=[8], encoder_name="perspective196",
                        output_mode="equity")
        with pytest.raises(ValueError):
            TDAgent(net, encoder=MatchplayEncoder())


# ── Warm start ──────────────────────────────────────────────────────────


class TestWarmStart:
    def test_function_preserved_on_base_features(self, tmp_path):
        from train_cli import _warm_start_matchplay
        torch.manual_seed(7)
        src = TDNetwork(hidden_sizes=[32, 16],
                        encoder_name="perspective196",
                        output_mode="probability")
        src_path = str(tmp_path / "dmp_src.pt")
        src.save(src_path)

        net = _warm_start_matchplay(src_path)
        assert net.encoder_name == "matchplay_perspective196"
        assert net.output_mode == "probability"
        agent = TDAgent(net)
        src_agent = TDAgent(src)

        rng = random.Random(5)
        from backgammon_engine import opening_roll, get_legal_plays
        s, dice = opening_roll(rng)
        for _ in range(10):
            plays = get_legal_plays(s, dice)
            if not plays:
                s = switch_turn(s)
            else:
                s = rng.choice(plays)[1]
            dice = (rng.randint(1, 6), rng.randint(1, 6))
            if s.is_game_over():
                break
            # Any cube/score context must give the same value as the
            # source's P(win game) — the new columns are zero.
            for ms in (
                MatchState(jacoby=False, match_length=2, my_away=2, opp_away=2),
                MatchState(jacoby=False, match_length=2, my_away=1, opp_away=2,
                           is_crawford=True, crawford_done=True),
                MatchState(jacoby=False, match_length=5, my_away=4, opp_away=2,
                           cube_owner=CubeOwner.BLACK, cube_value=2,
                           crawford_done=False),
            ):
                v_mp = agent.evaluate_matchplay(s, ms, is_cube_action=True)
                v_src = src_agent.evaluate(s)
                assert abs(v_mp - v_src) < 1e-6

    def test_matchplay1h_warm_start_function_preserving(self, tmp_path):
        # Cube one-hot conversion must reproduce the source net's
        # outputs exactly at every reachable cube level (w*2^k init).
        import numpy as np
        from train_cli import _warm_start_matchplay1h
        from encoding import get_encoder
        torch.manual_seed(3)
        src_net = TDNetwork(hidden_sizes=[16],
                            input_size=204,
                            encoder_name="matchplay_perspective196",
                            output_mode="probability")
        p = str(tmp_path / "mp.pt")
        src_net.save(p)
        new_net = _warm_start_matchplay1h(p)
        assert new_net.encoder_name == "matchplay1h_perspective196"
        enc_old = get_encoder("matchplay_perspective196")
        enc_new = get_encoder("matchplay1h_perspective196")
        state = BoardState.initial()
        for cube in (1, 2, 4, 8):
            for ca in (False, True):
                x0 = enc_old.encode(state, CubePerspective.MINE, cube,
                                    2, 3, False, ca)
                x1 = enc_new.encode(state, CubePerspective.MINE, cube,
                                    2, 3, False, ca)
                with torch.no_grad():
                    v0 = src_net(torch.from_numpy(x0).unsqueeze(0)).item()
                    v1 = new_net(torch.from_numpy(x1).unsqueeze(0)).item()
                assert abs(v0 - v1) < 1e-6

    def test_accepts_equity_source_via_sigmoid(self, tmp_path):
        # Equity sources (e.g. cubeless money) are accepted: the output
        # layer is copied as-is and reinterpreted through the
        # probability head's sigmoid, so the init is sigmoid(equity).
        import numpy as np
        import torch
        from train_cli import _warm_start_matchplay
        from encoding import get_encoder, CubePerspective
        src = TDNetwork(hidden_sizes=[8], encoder_name="perspective196",
                        output_mode="equity")
        p = str(tmp_path / "eq.pt")
        src.save(p)
        net = _warm_start_matchplay(p)
        assert net.output_mode == "probability"
        state = BoardState.initial()
        base = get_encoder("perspective196").encode(state)
        x_src = torch.from_numpy(base).unsqueeze(0)
        enc = get_encoder("matchplay_perspective196")
        x_mp = torch.from_numpy(enc.encode_with_base(
            base, CubePerspective.CENTERED, 1, 2, 2, False, False,
        )).unsqueeze(0)
        with torch.no_grad():
            v_eq = src(x_src).item()        # raw equity (linear head)
            v_mp = net(x_mp).item()         # sigmoid(same logit)
        assert abs(v_mp - torch.sigmoid(torch.tensor(v_eq)).item()) < 1e-6
