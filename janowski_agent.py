"""janowski_agent.py -- cubeful money player: prob5 net + Janowski cube layer.

Wraps a cubeless ProbNetwork in janowski.py's conversion at a fixed
cube-life index x (janowski.DEFAULT_X unless `x` is given) and implements
the cubeful Agent surface `gnubg_eval.play_and_record_cubeful` drives:
`offer_double`, `respond_to_double`, `choose_checker_action_cubeful`,
at 0-ply and 1-ply:

  * a successor is valued for the player to move next, with that
    player's own cube decision folded in (janowski_equity_folded);
    checker play takes the best value, float32 ties going to the better
    cubeless money equity;
  * cube actions follow janowski.money_cube_action (TDAgent's rule);
  * 1-ply mirrors TDAgent's cubeful 1-ply: every candidate is expanded
    (no move filter) and per roll each cube branch takes its own best
    reply, chosen by 0-ply checker play under that branch's cube;
  * bearoff class: gammon-free positions from the exact tables
    (bearoff.py, unexpanded at 1-ply), gammonish ones from the exact
    probabilities through the conversion. Tables must be built first
    (bearoff.build_table, bearoff.build_ts_table).

`cube_fold=False` drops every fold, `fold_0ply=False` only the 0-ply
checker-play fold.
"""

from __future__ import annotations

import numpy as np
import torch

from agents import CubeOffer
from backgammon_engine import BoardState, switch_turn
from encoding import CubePerspective, OPP_OFF_INDEX, TERMINAL_OFF_THRESHOLD
from janowski import (DEFAULT_X, cubeless_equity, fold_cube_decision,
                      janowski_equity, janowski_equity_folded,
                      money_cube_action)
from model import ProbNetwork
from prob_agent import ProbAgent, _DICE_OUTCOMES

# Recorded in saved H-H runs with a janowski:/bo: side; runs of different
# revisions refuse to pool. Bump whenever these agents' play changes.
JANOWSKI_REV = "2026-10-13"


def _feature_counts(feats: np.ndarray, offset: int) -> np.ndarray:
    t = feats[:, offset:offset + 96].reshape(-1, 24, 4)
    return np.rint(t[..., 0] + t[..., 1] + t[..., 2] + 2.0 * t[..., 3]).astype(np.int64)


def _bearoff_class_features(feats: np.ndarray) -> np.ndarray:
    """bearoff.is_bearoff_class per row of perspective196 features,
    without building BoardStates."""
    feats = np.asarray(feats, dtype=np.float64)
    me = _feature_counts(feats, 0)
    opp = _feature_counts(feats, 98)          # same frame as `me`
    no_bar = (np.rint(feats[:, 96] * 2) == 0) & (np.rint(feats[:, 194] * 2) == 0)
    return no_bar & (me[:, 6:].sum(axis=1) == 0) & (opp[:, :18].sum(axis=1) == 0)


def _pick(vals_opp, sec_opp) -> int:
    """Best successor: lowest opponent-view value, compared in float32;
    ties go to the lowest opponent-view cubeless money equity, then to
    the first."""
    v = np.asarray(vals_opp, dtype=np.float32)
    cand = np.flatnonzero(v == v.min())
    if cand.size == 1:
        return int(cand[0])
    sec = np.asarray(sec_opp, dtype=np.float32)[cand]
    return int(cand[int(np.argmin(sec))])


class JanowskiAgent(ProbAgent):
    """prob5 + Janowski cube layer = cubeful money agent. The prob5
    outputs get :func:`prob_agent.prob5_sanity` (``board_sanity``, on
    by default)."""

    def __init__(
        self,
        network: ProbNetwork,
        device: str = "cpu",
        bf16_inference: bool = False,
        oneply: bool = False,
        x: float = DEFAULT_X,
        cube_fold: bool = True,
        fold_0ply: bool = True,
        cubeless_checker: bool = False,
        bearoff_exact: bool = True,
        use_fast_engine: bool = True,
        board_sanity: bool = True,
    ):
        # ProbAgent's plies stay 0: its cubeless choosers are unused here
        # (cubeful games route through choose_checker_action_cubeful).
        super().__init__(
            network, device=device, bf16_inference=bf16_inference,
            plies=0, use_fast_engine=use_fast_engine,
            board_sanity=board_sanity,
        )
        self.is_cubeful = True
        self.output_mode = "equity"
        self.oneply = bool(oneply)
        self.x = float(x)
        self.cube_fold = bool(cube_fold)
        self.fold_0ply = self.cube_fold and bool(fold_0ply)
        # Ablation: rank checker moves by plain cubeless equity, keep the
        # Janowski cube decisions. 0-ply only.
        self.cubeless_checker = bool(cubeless_checker)
        if self.cubeless_checker and self.oneply:
            raise ValueError("cubeless_checker ablation is 0-ply only")
        self.bearoff_exact = bool(bearoff_exact)

    def _probs_np(self, t: torch.Tensor) -> np.ndarray:
        return self._probs(t).detach().cpu().numpy().astype(np.float64)

    def _probs_state(self, state: BoardState) -> np.ndarray:
        return self._probs_np(self._encode_batch([state]))[0]

    @staticmethod
    def _terminal_mag(state: BoardState) -> float:  # noqa: D401 (override)
        return float(state.game_result())

    @staticmethod
    def _gammons_count(match_state, cube_owner=None) -> bool:
        from modes import CubeOwner
        owner = match_state.cube_owner if cube_owner is None else cube_owner
        return not (match_state.jacoby and owner == CubeOwner.CENTERED)

    @staticmethod
    def _bearoff_probs(state: BoardState) -> np.ndarray:
        """Cubeless probs of a gammon-free bearoff: exact P(win)."""
        import bearoff
        p = bearoff.ts_win_prob(state)
        if p is None:
            p = float(bearoff.exact_probs5(state)[0])
        return np.array([p, 0.0, 0.0, 0.0, 0.0])

    def _succ_meta(self, feats: np.ndarray, state_at) -> dict:
        """Class data of successor rows. Materializes the rows the exact
        tiers need (terminal, bearoff class) now, before any further
        _bg_fast call invalidates `state_at`."""
        term = feats[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
        if self.bearoff_exact:
            bo = _bearoff_class_features(feats) & ~term
        else:
            bo = np.zeros_like(term)
        rows = {int(i): state_at(int(i)) for i in np.flatnonzero(term | bo)}
        return {"term": term, "bo": bo, "rows": rows}

    def _succ_values(self, probs: np.ndarray, meta: dict, branches: list,
                     fold: bool, exact_fold: bool):
        """0-ply values of successor rows for the player on roll there,
        one array per (persp, match_state) of `branches` (on-roll view,
        per unit cube). `fold` folds that player's own cube decision into
        the Janowski values, `exact_fold` into the exact table values.

        Returns (values, sec, probs, exact): sec the cubeless money
        equity with gammons counted (on-roll view), probs the outcome
        probabilities used (exact on bearoff and terminal rows), exact the
        rows that stay unexpanded at 1-ply (terminal, gammon-free table)."""
        import bearoff
        probs = probs.copy()
        exact = meta["term"].copy()
        table = {}                              # row -> [value per branch]
        for i in np.flatnonzero(meta["bo"]):
            s = meta["rows"][int(i)]
            vals = [bearoff.race_equity(s, persp, with_own_fold=exact_fold)
                    for persp, _ in branches]
            if vals[0] is not None:
                table[int(i)] = vals
                exact[i] = True
                probs[i] = self._bearoff_probs(s)
            else:
                probs[i] = bearoff.exact_probs5(s)
        raw = {int(i): self._terminal_mag(meta["rows"][int(i)])
               for i in np.flatnonzero(meta["term"])}
        for i, r in raw.items():                # the on-roll player lost
            probs[i] = (0.0, 0.0, 0.0, float(r >= 2), float(r >= 3))
        conv = janowski_equity_folded if fold else janowski_equity
        values = []
        for b, (persp, ms) in enumerate(branches):
            v = np.array(conv(probs, persp, self.x, ms.jacoby),
                         dtype=np.float64).reshape(-1)
            for i, vals in table.items():
                v[i] = vals[b]
            gammons = self._gammons_count(ms)
            for i, r in raw.items():
                v[i] = -(r if gammons else 1.0)
            values.append(v)
        return values, cubeless_equity(probs), probs, exact

    def _screen_0ply(self, features, succ, opp_persp, match_state):
        """0-ply opponent-view scores of the candidate successors (lower
        is better for the mover), with the opponent's cube decision folded
        in when fold_0ply (exact table values always fold). Returns
        (eq_opp, exact_mask, sec_opp)."""
        meta = self._succ_meta(features, succ.__getitem__)
        probs = self._probs_np(torch.from_numpy(features))
        values, sec, _, exact = self._succ_values(
            probs, meta, [(opp_persp, match_state)],
            fold=self.fold_0ply, exact_fold=True)
        return values[0], exact, sec

    def _screen_scores_0ply(self, features, succ, opp_persp, match_state):
        """(eq_opp, exact_mask) of _screen_0ply."""
        eq_opp, exact_mask, _ = self._screen_0ply(
            features, succ, opp_persp, match_state)
        return eq_opp, exact_mask

    def _oneply(self, state: BoardState, branch_ms: list):
        """1-ply at `state` (pre-roll, roller = state.turn), roller's
        view: the per-unit equity per root cube state of `branch_ms`
        (branch_ms[0] = the actual cube) and the averaged cubeless probs.
        Per roll and branch the roller's reply is the best by 0-ply
        checker play under that branch's cube; probs follow branch_ms[0]'s
        reply. The leaf player's own cube decision is folded when
        cube_fold. A forced pass leaves the board to the opponent."""
        from modes import cube_perspective
        leaf_player = 1 - state.turn
        branches = [(cube_perspective(ms.cube_owner, leaf_player), ms)
                    for ms in branch_ms]
        chunks, feats_all = [], []
        for d1, d2 in _DICE_OUTCOMES:
            w = 1.0 if d1 == d2 else 2.0
            feats, nxt = self._get_legal_plays_encoded(state, (d1, d2))
            if len(nxt) == 0:
                passed = switch_turn(state)
                feats = self.encode_state(passed)[None, :]
                meta = self._succ_meta(feats, lambda i, s=passed: s)
            else:
                meta = self._succ_meta(feats, nxt.__getitem__)
            chunks.append((w, len(feats), meta))
            feats_all.append(feats)
        probs_all = self._probs_np(torch.from_numpy(np.concatenate(feats_all)))
        acc = np.zeros(len(branch_ms))
        pacc = np.zeros(5)
        off = 0
        for w, n, meta in chunks:
            values, sec, probs, _ = self._succ_values(
                probs_all[off:off + n], meta, branches,
                fold=self.cube_fold, exact_fold=self.cube_fold)
            off += n
            for b in range(len(branch_ms)):
                j = _pick(values[b], sec)
                acc[b] += w * values[b][j]
                if b == 0:
                    pacc += w * probs[j]
        p = pacc / 36.0
        roller_probs = np.array([1.0 - p[0], p[3], p[4], p[1], p[2]])
        return -acc / 36.0, roller_probs

    def _value_oneply(self, state: BoardState, match_state, fold: bool):
        """(1-ply value, 1-ply cubeless equity) of `state`, state.turn's
        view; `fold` folds the roller's own cube decision."""
        roller = state.turn
        if fold and match_state.can_offer(roller):
            v, probs = self._oneply(
                state, [match_state, match_state.after_take(roller)])
            return (float(fold_cube_decision(v[0], 2.0 * v[1])),
                    float(cubeless_equity(probs)))
        v, probs = self._oneply(state, [match_state])
        return float(v[0]), float(cubeless_equity(probs))

    def _value_oneply_cubeful(self, state: BoardState, match_state,
                              fold: bool) -> float:
        return self._value_oneply(state, match_state, fold)[0]

    def _static_cube(self, state: BoardState, match_state):
        """0-ply (v_no_double, v_double_take, probs) at `state` (pre-roll,
        state.turn deciding). Bearoff-class positions use the exact
        tables, or exact probabilities while gammons are possible."""
        from modes import cube_perspective
        persp = cube_perspective(match_state.cube_owner, state.turn)
        probs = None
        if self.bearoff_exact:
            import bearoff
            if bearoff.is_bearoff_class(state):
                vals = bearoff.race_cube_values(state, persp)
                if vals is not None:
                    return vals[0], vals[1], self._bearoff_probs(state)
                probs = bearoff.exact_probs5(state)
        if probs is None:
            probs = self._probs_state(state)
        v_nd = float(janowski_equity(probs, persp, self.x, match_state.jacoby))
        # After double-take the cube is the opponent's: THEIRS view.
        v_dt = 2.0 * float(janowski_equity(
            probs, CubePerspective.THEIRS, self.x, match_state.jacoby))
        return v_nd, v_dt, probs

    def _static_nd_dt(self, state: BoardState, match_state):
        """0-ply (v_no_double, v_double_take)."""
        return self._static_cube(state, match_state)[:2]

    def _cube_values(self, state: BoardState, match_state):
        """(v_no_double, v_double_take, probs) at this agent's depth. A
        gammon-free bearoff-class root takes the exact tables at 1-ply
        too."""
        if not self.oneply:
            return self._static_cube(state, match_state)
        from modes import cube_perspective
        persp = cube_perspective(match_state.cube_owner, state.turn)
        if self.bearoff_exact:
            import bearoff
            if bearoff.is_bearoff_class(state):
                vals = bearoff.race_cube_values(state, persp)
                if vals is not None:
                    return vals[0], vals[1], self._bearoff_probs(state)
        v, probs = self._oneply(
            state, [match_state, match_state.after_take(state.turn)])
        return float(v[0]), 2.0 * float(v[1]), probs

    def offer_double(self, state: BoardState, match_state) -> CubeOffer:
        from modes import cube_perspective
        persp = cube_perspective(match_state.cube_owner, state.turn)
        if persp not in (CubePerspective.CENTERED, CubePerspective.MINE):
            return CubeOffer(should_double=False, _cache=None)
        key = "1ply" if self.oneply else "0ply"
        v_nd, v_dt, _ = self._cube_values(state, match_state)
        should, take = money_cube_action(v_nd, v_dt, persp, match_state.jacoby)
        return CubeOffer(
            should_double=should,
            _cache={f"v_double_take_{key}": v_dt, f"v_no_double_{key}": v_nd,
                    f"take_{key}": take},
        )

    def respond_to_double(
        self, state: BoardState, match_state, hint: CubeOffer = None,
    ) -> bool:
        """Take/pass at the doubler's position (`state.turn` is the
        doubler, pre-roll), per janowski.money_cube_action."""
        from modes import cube_perspective
        key = "take_1ply" if self.oneply else "take_0ply"
        if hint is not None and hint._cache is not None and key in hint._cache:
            return hint._cache[key]
        v_nd, v_dt, _ = self._cube_values(state, match_state)
        persp = cube_perspective(match_state.cube_owner, state.turn)
        return money_cube_action(v_nd, v_dt, persp, match_state.jacoby)[1]

    def choose_checker_action_cubeful(
        self, state: BoardState, dice, match_state,
    ):
        """Cube-aware move selection: (next_state, None), or None on a
        forced pass (the caller switches the turn)."""
        opponent = 1 - state.turn
        from modes import cube_perspective
        opp_persp = cube_perspective(match_state.cube_owner, opponent)

        features, next_states = self._get_legal_plays_encoded(state, dice)
        n = len(next_states)
        if n == 0:
            return None
        # Materialize now: the C-buffer views go stale on the next
        # _bg_fast call.
        succ = [next_states[i] for i in range(n)]

        if self.cubeless_checker:
            # ProbAgent._choose_0ply's selection with raw terminal values.
            probs = self._probs_np(torch.from_numpy(features))
            eq_opp = cubeless_equity(probs)
            for j in np.flatnonzero(
                features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
            ):
                eq_opp[int(j)] = -self._terminal_mag(succ[int(j)])
            return succ[int(np.argmin(eq_opp))], None

        eq_opp, exact, sec = self._screen_0ply(
            features, succ, opp_persp, match_state)
        if not self.oneply or n == 1:
            return succ[_pick(eq_opp, sec)], None

        # 1-ply: every candidate; exact candidates keep their 0-ply value.
        scores = np.empty(n)
        sec1 = np.empty(n)
        for j in range(n):
            if exact[j]:
                scores[j], sec1[j] = eq_opp[j], sec[j]
            else:
                scores[j], sec1[j] = self._value_oneply(
                    succ[j], match_state, fold=self.cube_fold)
        return succ[_pick(scores, sec1)], None
