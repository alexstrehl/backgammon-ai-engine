"""bearoff_td.py -- cubeful TDAgent + exact bearoff routing.

The Janowski comparison gives the exact bearoff tables to both sides.
BearoffTDAgent overrides exactly the decisions the exact tiers cover
(bearoff-class cube decisions, bearoff-class checker successors) and
leaves everything else to the stock TDAgent. 0-ply and 1-ply. At 1-ply
a bearoff-class root shortcuts the cube decision to the exact values
and a bearoff-class successor keeps its exact value unexpanded.

Side spec: ``bo:<cubeful checkpoint>``.
"""

from __future__ import annotations

import numpy as np
import torch

from agents import CubeOffer
from backgammon_engine import BoardState, get_legal_plays
from encoding import CubePerspective
from td_agent import TDAgent, _ONEPLY_PROBS


class BearoffTDAgent(TDAgent):
    """TDAgent whose bearoff-class decisions route to the exact tables."""

    def __init__(self, network, *args, device="cpu", **kwargs):
        from prob_agent import require_cpu
        require_cpu("BearoffTDAgent", network, device)
        super().__init__(network, *args, device=device, **kwargs)
        if not self.is_cubeful:
            raise ValueError("BearoffTDAgent requires a cubeful network")
        if self.twoply_k:
            raise ValueError("BearoffTDAgent supports 0-ply and 1-ply only")

    # ── cube decisions ───────────────────────────────────────────────

    def offer_double(self, state: BoardState, match_state) -> CubeOffer:
        import bearoff
        from modes import cube_perspective
        persp = cube_perspective(match_state.cube_owner, state.turn)
        if persp in (CubePerspective.CENTERED, CubePerspective.MINE):
            vals = bearoff.race_cube_values(state, persp)
            if vals is not None:
                v_nd, v_dt = vals
                should = TDAgent._money_cube_decision(
                    v_nd, v_dt, persp, match_state.jacoby)
                key = "1ply" if self.oneply else "0ply"
                return CubeOffer(
                    should_double=should,
                    _cache={f"v_double_take_{key}": v_dt,
                            f"v_no_double_{key}": v_nd},
                )
        return super().offer_double(state, match_state)

    def respond_to_double(self, state, match_state, hint=None) -> bool:
        import bearoff
        from modes import cube_perspective
        persp = cube_perspective(match_state.cube_owner, state.turn)
        vals = bearoff.race_cube_values(state, persp)
        if vals is not None:
            return vals[1] <= 1.0
        return super().respond_to_double(state, match_state, hint=hint)

    # ── checker play ─────────────────────────────────────────────────

    def choose_checker_action_cubeful(self, state, dice, match_state):
        import bearoff
        from modes import CubeOwner, cube_perspective

        # Positions that cannot reach the bearoff class within the
        # search keep the parent's C fast path.
        if not bearoff.may_reach_bearoff(state, 2 if self.oneply else 1):
            return super().choose_checker_action_cubeful(
                state, dice, match_state)

        plays = get_legal_plays(state, dice)
        if not plays:
            return None
        next_states = [s for _, s in plays]

        # Net values, encoded as in the parent's 0-ply path.
        base = self.encoder._base
        base_feats = np.stack([base.encode(s) for s in next_states])
        opponent = 1 - state.turn
        opp_persp = cube_perspective(match_state.cube_owner, opponent)
        extra = np.zeros((len(next_states), self.encoder.cube_features),
                         dtype=np.float32)
        extra[:, int(opp_persp)] = 1.0
        if match_state.can_offer(opponent):
            extra[:, 3] = 1.0
        t = self._to_device(torch.from_numpy(
            np.concatenate([base_feats, extra], axis=1)))
        with torch.no_grad():
            values = self.network(t).detach().cpu().numpy().ravel()
        values = values.astype(np.float64)      # exact values stay exact

        # Exact values for bearoff-class successors.
        exact = np.zeros(len(next_states), dtype=bool)
        for j, s in enumerate(next_states):
            v = bearoff.race_equity(s, opp_persp, with_own_fold=True)
            if v is not None:
                values[j] = v
                exact[j] = True

        # Terminals as in the parent (Jacoby collapse).
        gammons_count = (
            not match_state.jacoby
            or match_state.cube_owner != CubeOwner.CENTERED
        )
        for j, s in enumerate(next_states):
            if s.is_game_over():
                gr = float(s.game_result())
                values[j] = -gr if gammons_count else -1.0
                exact[j] = True

        if not self.oneply:
            idx = int(np.argmin(values))
            return next_states[idx], -float(values[idx])

        # 1-ply: exact and terminal candidates keep their value. The rest
        # are expanded, with bearoff-class leaves routed when reachable.
        # All are cubeful per-unit equities with the opponent's cube
        # rights folded in.
        best_state, best_val = None, float("-inf")  # mover's view
        net_plays = []
        for j, s in enumerate(next_states):
            if exact[j]:
                v = -float(values[j])
            elif bearoff.may_reach_bearoff(s, 1):
                v = self._candidate_value_oneply(s, match_state)
            else:
                net_plays.append(plays[j])
                continue
            if v > best_val:
                best_val, best_state = v, s
        if net_plays:
            s1, v1 = self._choose_checker_oneply_cubeful(
                state, net_plays, match_state)
            if v1 > best_val:
                best_val, best_state = v1, s1
        return best_state, best_val

    def _candidate_value_oneply(self, next_state, match_state) -> float:
        """_choose_checker_oneply_cubeful's per-candidate value (mover's
        view) through value_oneply_checker_cubeful, which routes leaves."""
        from modes import cube_perspective
        opp = next_state.turn
        u_a = -self.value_oneply_checker_cubeful(next_state, match_state)
        if not match_state.can_offer(opp):
            return u_a
        u_b = -self.value_oneply_checker_cubeful(
            next_state, match_state.after_take(opp))
        nd_opp, dt_opp = -u_a, -2.0 * u_b
        if TDAgent._money_cube_decision(
                nd_opp, dt_opp, cube_perspective(match_state.cube_owner, opp),
                match_state.jacoby):
            return 2.0 * u_b if dt_opp <= 1.0 else -1.0
        return u_a

    def value_oneply_checker_cubeful(self, state, match_state) -> float:
        """Parent's 1-ply value with bearoff-class leaves at their exact
        folded value."""
        import bearoff
        if not bearoff.may_reach_bearoff(state, 1):
            return super().value_oneply_checker_cubeful(state, match_state)
        from modes import CubeOwner, cube_perspective
        assert not state.is_game_over()
        opponent = 1 - state.turn
        opp_persp = cube_perspective(match_state.cube_owner, opponent)
        gammons_count = (
            not match_state.jacoby
            or match_state.cube_owner != CubeOwner.CENTERED
        )
        results, nonterm, nexts = self._expand_cubeful_level1(
            state, opp_persp, match_state.can_offer(opponent),
            gammons_count, track_next_states=True)
        total = 0.0
        for di, prob in enumerate(_ONEPLY_PROBS.tolist()):
            eqs = results[di]
            idxs = () if nonterm[di] is None else nonterm[di]
            for i, s in zip(idxs, nexts[di] or ()):
                v = bearoff.race_equity(s, opp_persp, with_own_fold=True)
                if v is not None:
                    eqs[int(i)] = -v
            total += prob * max(eqs)
        return total
