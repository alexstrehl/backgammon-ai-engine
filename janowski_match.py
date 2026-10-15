"""janowski_match.py -- match-play Janowski cube layer for prob5.

Turns the cubeless prob5 net (5 nested outcome probabilities, no cube
policy, no score awareness) into a full N-point MATCH player:

  * checker play maximizes cubeful MWC at the current score/cube
    (score-aware gammon pricing falls out of the MET weighting);
  * cube decisions compare cubeful MWC of ND / DT / DP, exactly the
    comparison the trained matchplay TDAgent makes -- so the two are
    drop-in comparable and this agent doubles as a reference
    instrument for debugging the learned cubeful match player.

The cubeless->cubeful MWC conversion is a faithful port of gnubg
1.08.003 -- the de-facto standard all three engines surveyed implement
(gnubg natively; Open Sage ports the same `GetPoints` recursion; Hedgehog
is money-only):

    eval.c:   Cl2CfMatchCentered / Cl2CfMatchOwned /
              Cl2CfMatchUnavailable  (piecewise-live + dead, blended
              by the cube-life index x), fDoCubeful (dead-cube gate)
    matchequity.c: GetPoints (live cash points, recursive up the cube
              ladder through the MET), GetCubePrimeValue (the taker's
              automatic redouble when the doubled cube is dead for
              the doubler), getMEMultiple (outcome-MWC lookups; our
              met.mwc_after plays that role, already verified against
              MatchState.after_game on every reachable transition)

Structure of the conversion, from the evaluating player's view with
game-win probability p and conditional gammon/backgammon rates held
fixed (Janowski's model):

    MWC_cubeful = (1 - x) * MWC_dead  +  x * MWC_live

  MWC_dead  = sum over the six outcomes of P(outcome) * MWC(outcome)
              at the current cube value (identical to gnubg's
              eq2mwc(Utility(.)) -- the match gammon prices are
              defined to make that composition equal this mix).
  MWC_live  = piecewise-linear in p. Kinks sit at the LIVE cash
              points from GetPoints; kink values are plain
              win/lose-nCube MWCs (a cash is always a single-stake
              win); the p=0 / p=1 endpoints carry the position's
              actual gammon/backgammon mix.

GetPoints recursion (per side k at cube level v, all MWCs from k's
view):
  ladder top (someone's away <= 2v -- the doubled cube would be dead;
  the taker auto-redoubles when the cube is dead for the DOUBLER
  only, so stakes use the prime value pv = 2v in that case):
      CP_k = (DTL - DP) / (DTL - DTW)
      DTL/DTW at 2pv with the winner's gammon mix, DP = win v plain
  below (both alive):
      CP_k(v) = 1 - CP_!k(2v) * (DP - DTW) / (RDP - DTW)
      RDP = lose 2v plain (k passes the taker's efficient redouble)

Deliberate gnubg-isms kept (documented, not accidents):
  * fDoCubeful treats Crawford, cube>=both aways, AND the score
    2-away/2-away as fully dead (the 2a2a special case is gnubg's:
    any cube turn there is instantly dead, and the resulting
    dead-value ND/DT/DP comparison reproduces the classic "double on
    any clear advantage at 2a2a" doctrine -- see tests).
  * In the THEIRS curve the below-kink segment runs to the PLAIN
    lose-nCube MWC at the kink even when the opponent would never
    redouble; matching Cl2CfMatchUnavailable exactly.
  * Take points inherit the position's gammon rates through the whole
    ladder (gnubg carries arG/arBG into every level).

The x (cube-life index) policy is shared with the money layer
(janowski.cube_x): gnubg's shallow-ply class constants -- contact
0.76, race clamp(0.60 + 0.0025 * roller_pips, 0.60, 0.80) -- or a
fixed override.

C paths (c_api/README.md "p5jan / Janowski match play"). This module is
the reference; two opt-ins move work into C without changing a decision
beyond near-ties:
  * the net forward only: BG_FAST_FORWARD=<backend> at construction (or
    `agent._fast_ff = FastForward(net, backend)`): c_api/fast_forward.py
    runs the prob5 forward, the conversion and every decision rule stay
    here (~80% of a decision).
  * the whole decision: `agent.enable_fast_decisions(backend)` --
    offer_double, respond_to_double, choose_checker_action_cubeful and
    value_oneply run in a c_api match engine on the prob5 net
    (BGC_KIND_MATCH_JANOWSKI: c_api/puretd_janowski.c ports cubeful_mwc /
    get_points / dead_mwc / do_cubeful / cube_decision / cube_x operation
    for operation, bitwise this module's given the same probabilities;
    puretd_match.c's evaluator seam reuses the TD match core's move
    generation, batches and memos). Bitwise the net-only path on the same
    backend's plain forward, within ~4e-7 of the torch path on the
    production simd forward (tests/test_janowski_c.py).
    eval_matchplay.py selects it for a p5jan side with --fast-forward /
    `p5jan:ff-<backend>:`; `p5jan:ffnet-<backend>:` keeps the net-only path.
    The constructor never enables it (the trainer's p5jan oracle keeps its
    path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from agents import Agent, CubeOffer
from backgammon_engine import BoardState, get_legal_plays
from encoding import (
    CubePerspective,
    OPP_OFF_INDEX,
    TERMINAL_OFF_THRESHOLD,
    get_encoder,
)
from janowski import cube_x
from met import MatchEquityTable, default_met
from model import ProbNetwork, prob5_postprocess

try:
    import c_engine.bg_fast as _bg_fast
    _HAS_FAST = True
except Exception:                                   # pragma: no cover
    _bg_fast = None
    _HAS_FAST = False

_EPS = 1e-7


# ── score/cube context ───────────────────────────────────────────────


@dataclass(frozen=True)
class MatchCtx:
    """Score + cube context from the EVALUATING player's own view.

    `modes.MatchState` stores away scores in WHITE's view; build this
    through `ctx_for(match_state, viewer, ...)` so the flip happens in
    exactly one place.
    """
    my_away: int
    opp_away: int
    crawford_done: bool
    is_crawford: bool
    cube: int

    def flip(self, cube: Optional[int] = None) -> "MatchCtx":
        return MatchCtx(
            self.opp_away, self.my_away,
            self.crawford_done, self.is_crawford,
            self.cube if cube is None else cube,
        )

    def with_cube(self, cube: int) -> "MatchCtx":
        return MatchCtx(
            self.my_away, self.opp_away,
            self.crawford_done, self.is_crawford, cube,
        )


def ctx_for(match_state, viewer: int, cube: Optional[int] = None) -> MatchCtx:
    """MatchCtx from `viewer`'s perspective (away scores auto-flip for
    BLACK -- same convention as TDAgent._match_features)."""
    if viewer == 0:
        my_away, opp_away = match_state.my_away, match_state.opp_away
    else:
        my_away, opp_away = match_state.opp_away, match_state.my_away
    return MatchCtx(
        my_away, opp_away,
        match_state.crawford_done, match_state.is_crawford,
        match_state.cube_value if cube is None else cube,
    )


def _me(met: MatchEquityTable, ctx: MatchCtx, points: int, i_win: bool) -> float:
    """MWC (my view) after someone scores `points` from this game."""
    return met.mwc_after(
        ctx.my_away, ctx.opp_away, ctx.crawford_done, ctx.is_crawford,
        points, i_win,
    )


# ── gnubg fDoCubeful ─────────────────────────────────────────────────


def do_cubeful(ctx: MatchCtx) -> bool:
    """gnubg eval.c fDoCubeful: False -> the conversion is pure dead."""
    if ctx.is_crawford:
        return False
    if ctx.cube >= ctx.my_away and ctx.cube >= ctx.opp_away:
        return False
    if ctx.my_away == 2 and ctx.opp_away == 2:
        return False
    return True


# ── (rates, dead MWC) from nested probs ──────────────────────────────


def _rates(probs: np.ndarray) -> Tuple[float, float, float, float]:
    """(g_me, bg_me, g_opp, bg_opp): conditional gammon/backgammon
    rates, gnubg's guarded ratios of the nested unconditional probs."""
    p = float(probs[0])
    if p > 0.0:
        g_me = (float(probs[1]) - float(probs[2])) / p
        bg_me = float(probs[2]) / p
    else:
        g_me = bg_me = 0.0
    if p < 1.0:
        g_opp = (float(probs[3]) - float(probs[4])) / (1.0 - p)
        bg_opp = float(probs[4]) / (1.0 - p)
    else:
        g_opp = bg_opp = 0.0
    return g_me, bg_me, g_opp, bg_opp


def dead_mwc(probs: np.ndarray, ctx: MatchCtx, met: MatchEquityTable) -> float:
    """Cubeless (dead-cube) MWC at the current cube value: the six-way
    outcome mix. Equals gnubg's eq2mwc(Utility(arOutput, pci))."""
    v = ctx.cube
    p = float(probs[0])
    w1 = _me(met, ctx, v, True)
    w2 = _me(met, ctx, 2 * v, True)
    w3 = _me(met, ctx, 3 * v, True)
    l1 = _me(met, ctx, v, False)
    l2 = _me(met, ctx, 2 * v, False)
    l3 = _me(met, ctx, 3 * v, False)
    return (
        p * w1
        + float(probs[1]) * (w2 - w1)
        + float(probs[2]) * (w3 - w2)
        + (1.0 - p) * l1
        + float(probs[3]) * (l2 - l1)
        + float(probs[4]) * (l3 - l2)
    )


# ── GetPoints (matchequity.c) ────────────────────────────────────────


def get_points(
    rates: Tuple[float, float, float, float],
    ctx: MatchCtx,
    met: MatchEquityTable,
) -> Tuple[float, float]:
    """Live cash points (cp_me, cp_opp), each in that side's own
    game-win-probability space. Port of matchequity.c GetPoints.

    rates = (g_me, bg_me, g_opp, bg_opp), conditional.
    """
    away = (ctx.my_away, ctx.opp_away)
    ctxs = (ctx, ctx.flip())
    r_g = (rates[0], rates[2])
    r_bg = (rates[1], rates[3])

    # Ladder: climb while the doubled cube would still be alive for
    # both (gnubg: i >= 2*nDead && j >= 2*nDead with i = away-1).
    n_dead, n_max = ctx.cube, 0
    while away[0] - 1 >= 2 * n_dead and away[1] - 1 >= 2 * n_dead:
        n_max += 1
        n_dead *= 2

    cp_live = [[0.0] * (n_max + 2) for _ in range(2)]
    cv = n_dead
    for n in range(n_max, -1, -1):
        for k in (0, 1):
            c = ctxs[k]
            g_win, bg_win = r_g[k], r_bg[k]
            g_lose, bg_lose = r_g[1 - k], r_bg[1 - k]
            if away[0] - 1 < 2 * cv or away[1] - 1 < 2 * cv:
                # Doubled cube would be dead. Automatic redouble: when
                # it is dead for the DOUBLER only, the taker redoubles
                # for free -> stakes at the prime value 2cv.
                if away[k] - 1 < 2 * cv and away[1 - k] - 1 >= 2 * cv:
                    pv = 2 * cv
                else:
                    pv = cv
                dtl = (
                    (1.0 - g_lose - bg_lose) * _me(met, c, 2 * pv, False)
                    + g_lose * _me(met, c, 4 * pv, False)
                    + bg_lose * _me(met, c, 6 * pv, False)
                )
                dp = _me(met, c, cv, True)
                dtw = (
                    (1.0 - g_win - bg_win) * _me(met, c, 2 * pv, True)
                    + g_win * _me(met, c, 4 * pv, True)
                    + bg_win * _me(met, c, 6 * pv, True)
                )
                denom = dtl - dtw
                cp_live[k][n] = (dtl - dp) / denom if abs(denom) > _EPS else 1.0
            else:
                # Doubled cube alive: recurse through the taker's
                # next-level cash point.
                rdp = _me(met, c, 2 * cv, False)
                dp = _me(met, c, cv, True)
                dtw = (
                    (1.0 - g_win - bg_win) * _me(met, c, 2 * cv, True)
                    + g_win * _me(met, c, 4 * cv, True)
                    + bg_win * _me(met, c, 6 * cv, True)
                )
                denom = rdp - dtw
                frac = (dp - dtw) / denom if abs(denom) > _EPS else 0.0
                cp_live[k][n] = 1.0 - cp_live[1 - k][n + 1] * frac
        cv //= 2
    return cp_live[0][0], cp_live[1][0]


# ── Cl2CfMatch (eval.c) ──────────────────────────────────────────────


def cubeful_mwc(
    probs: np.ndarray,
    persp: CubePerspective,
    ctx: MatchCtx,
    x: float,
    met: Optional[MatchEquityTable] = None,
) -> float:
    """Cubeful MWC for the evaluating player. Port of Cl2CfMatch.

    `probs` are the nested 5 outcome probabilities from the evaluating
    player's view; `persp` is their cube perspective; `ctx` their
    score view; `x` the cube-life index.
    """
    if met is None:
        met = default_met()
    p = float(probs[0])
    dead = dead_mwc(probs, ctx, met)
    if not do_cubeful(ctx) or p <= _EPS or p >= 1.0 - _EPS:
        return dead

    g_me, bg_me, g_opp, bg_opp = _rates(probs)
    cp_me, cp_opp = get_points((g_me, bg_me, g_opp, bg_opp), ctx, met)

    v = ctx.cube
    w1 = _me(met, ctx, v, True)          # cash: plain win of the cube
    l1 = _me(met, ctx, v, False)         # cashed out: plain loss
    mwc_win = (
        (1.0 - g_me - bg_me) * w1
        + g_me * _me(met, ctx, 2 * v, True)
        + bg_me * _me(met, ctx, 3 * v, True)
    )
    mwc_lose = (
        (1.0 - g_opp - bg_opp) * l1
        + g_opp * _me(met, ctx, 2 * v, False)
        + bg_opp * _me(met, ctx, 3 * v, False)
    )

    if persp == CubePerspective.CENTERED:
        opp_tg = 1.0 - cp_opp
        tg = cp_me
        if p <= opp_tg:
            live = (mwc_lose + (l1 - mwc_lose) * p / opp_tg
                    if opp_tg > 0.0 else mwc_lose)
        elif p < tg:
            live = l1 + (w1 - l1) * (p - opp_tg) / (tg - opp_tg)
        else:
            live = (w1 + (mwc_win - w1) * (p - tg) / (1.0 - tg)
                    if tg < 1.0 else mwc_win)
    elif persp == CubePerspective.MINE:
        tg = cp_me
        if p <= tg:
            live = (mwc_lose + (w1 - mwc_lose) * p / tg
                    if tg > 0.0 else mwc_lose)
        else:
            live = (w1 + (mwc_win - w1) * (p - tg) / (1.0 - tg)
                    if tg < 1.0 else mwc_win)
    else:  # THEIRS -- cube unavailable to us
        opp_tg = 1.0 - cp_opp
        if p <= opp_tg:
            live = (mwc_lose + (l1 - mwc_lose) * p / opp_tg
                    if opp_tg > 0.0 else mwc_lose)
        else:
            live = l1 + (mwc_win - l1) * (p - opp_tg) / (1.0 - opp_tg)

    return (1.0 - x) * dead + x * live


# ── cube decision (the TDAgent-matchplay comparison, MWC space) ─────


def cube_decision(
    probs: np.ndarray,
    match_state,
    player: int,
    x: float,
    met: Optional[MatchEquityTable] = None,
) -> dict:
    """Full ND/DT/DP breakdown for `player` on roll (pre-roll).

    Returns {"v_nd", "v_dt", "v_dp", "should_double", "take"} -- all
    MWCs from the DOUBLER's view, matching TDAgent's matchplay
    convention (double iff min(DT, DP) > ND; responder takes iff
    DP > DT). Also the reference instrument for auditing the learned
    cubeful match player's cube decisions.
    """
    from modes import cube_perspective
    if met is None:
        met = default_met()
    persp = cube_perspective(match_state.cube_owner, player)
    ctx = ctx_for(match_state, player)
    v_nd = cubeful_mwc(probs, persp, ctx, x, met)
    v_dt = cubeful_mwc(
        probs, CubePerspective.THEIRS, ctx.with_cube(2 * ctx.cube), x, met,
    )
    v_dp = _me(met, ctx, ctx.cube, True)
    return {
        "v_nd": v_nd, "v_dt": v_dt, "v_dp": v_dp,
        "should_double": min(v_dt, v_dp) > v_nd,
        "take": v_dp > v_dt,
    }


# ── the agent ────────────────────────────────────────────────────────


class JanowskiMatchAgent(Agent):
    """prob5 net + gnubg-style match cube layer = N-point match agent.

    Implements the three methods eval_matchplay.play_match drives:
    offer_double / respond_to_double / choose_checker_action_cubeful.
    0-ply: every decision is one batched net forward + analytic
    conversion. The checker chooser ranks successors by cubeful MWC at
    the current score and cube (score-aware move selection -- at DMP
    scores this reduces to pure P(win); gammon value at e.g. 2-away
    enters through the MET weighting).
    """

    is_cubeful = True
    is_matchplay = True

    def __init__(
        self,
        network: ProbNetwork,
        met: Optional[MatchEquityTable] = None,
        x_policy: Optional[float] = None,
        device: str = "cpu",
    ):
        self.network = network.to(device)
        self.network.eval()
        self.device = device
        self.met = met if met is not None else default_met()
        self.x_policy = None if x_policy is None else float(x_policy)
        self.encoder = get_encoder(network.encoder_name)
        self._fast = _HAS_FAST and network.encoder_name == "perspective196"
        # c_api forward backend (BG_FAST_FORWARD, see c_api/fast_forward.py);
        # None keeps the torch forward.
        self._fast_ff = None
        from c_api.fast_forward import FastForward, backend_from_env, supports
        _ff_backend = backend_from_env()
        if _ff_backend is not None and device == "cpu" and supports(self.network):
            self._fast_ff = FastForward(self.network, _ff_backend)
        # C decisions (enable_fast_decisions); None: the Python methods.
        self._fast_dec = None

    def enable_fast_decisions(self, backend: Optional[str]) -> None:
        """Run this agent's decisions in C on a c_api backend (None: back
        to Python): offer_double, respond_to_double,
        choose_checker_action_cubeful and value_oneply go through a
        BGC_KIND_MATCH_JANOWSKI engine on this agent's net, MET and x
        policy -- the prob5 forward, the conversion and the selection in
        one C call. CPU only. The engine holds a copy of the weights,
        re-opened when the net's parameters change in place (tensor
        version counters, as FastForward) or the MET / x policy is
        replaced. Independent of the net-only `_fast_ff`, which keeps
        serving what stays in Python (_probs_state,
        choose_checker_action_oneply's 0-ply pass)."""
        if backend is None:
            self._fast_dec = None
            return
        from c_api.fast_forward import supports
        if self.device != "cpu":
            raise ValueError("C decisions run on the CPU (device='cpu')")
        if not supports(self.network):
            raise ValueError(f"C decisions: {type(self.network).__name__} is not a "
                             "plain ProbNetwork forward")
        self._fast_dec = _CDecisions(self, backend)

    # ── net plumbing ────────────────────────────────────────────────

    # Optional remote evaluator (gpu_eval_server.RemoteEvaluator):
    # when attached, batched forwards route to the shared GPU server
    # (one CUDA context for all workers) and only the post-processing
    # runs locally. The server returns raw net(t) outputs, so the
    # sigmoid/postprocess below mirrors the local path exactly.
    _remote = None

    # Below this row count the IPC round-trip costs more than a local
    # CPU forward; single-state reads and eps-membership batches stay
    # local, the ~350-row value_oneply batches route.
    _REMOTE_MIN_ROWS = 64

    def _forward(self, feats: np.ndarray) -> np.ndarray:
        if (self._remote is not None
                and feats.shape[0] >= self._REMOTE_MIN_ROWS):
            out_np = self._remote.forward(
                np.ascontiguousarray(feats, dtype=np.float32))
            if out_np is not None:
                with torch.no_grad():
                    out = torch.from_numpy(out_np)
                    if self.network.raw_logits:
                        out = torch.sigmoid(out)
                    out = prob5_postprocess(out)
                return out.numpy().astype(np.float64)
        if self._fast_ff is not None:
            # The backend's prob5 output is sigmoid(raw) = the branch below.
            with torch.no_grad():
                out = prob5_postprocess(torch.from_numpy(self._fast_ff(feats)))
            return out.numpy().astype(np.float64)
        t = torch.from_numpy(feats)
        if self.device != "cpu":
            t = t.to(self.device)
        with torch.no_grad():
            out = self.network(t)
            if self.network.raw_logits:
                out = torch.sigmoid(out)
            out = prob5_postprocess(out)
        return out.cpu().numpy().astype(np.float64)

    def _probs_state(self, state: BoardState) -> np.ndarray:
        """Nested probs from state.turn's (on-roll) perspective."""
        if self._fast:
            feats = _bg_fast.encode_state(state)[None, :]
        else:
            feats = self.encoder.encode(state)[None, :].astype(np.float32)
        return self._forward(feats)[0]

    def _x(self, state: BoardState) -> float:
        return self.x_policy if self.x_policy is not None else cube_x(state)

    # ── cube decisions ──────────────────────────────────────────────

    def offer_double(self, state: BoardState, match_state) -> CubeOffer:
        """Pre-roll cube decision for state.turn."""
        player = state.turn
        if not match_state.can_offer(player):
            return CubeOffer(should_double=False, _cache=None)
        if self._fast_dec is not None:
            dbl, c = self._fast_dec.engine().match_offer(state, match_state)
            return CubeOffer(should_double=dbl,
                             _cache={"v_nd": c.v_nd, "v_dt": c.v_dt,
                                     "v_dp": c.v_dp})
        probs = self._probs_state(state)
        d = cube_decision(probs, match_state, player,
                          self._x(state), self.met)
        return CubeOffer(
            should_double=d["should_double"],
            _cache={"v_nd": d["v_nd"], "v_dt": d["v_dt"],
                    "v_dp": d["v_dp"]},
        )

    def respond_to_double(
        self, state: BoardState, match_state,
        hint: Optional[CubeOffer] = None,
    ) -> bool:
        """Take/pass, True = take. `state.turn` is the DOUBLER and all
        values are the doubler's MWC (TDAgent convention): take iff
        v_dp > v_dt."""
        if hint is not None and hint._cache and "v_dt" in hint._cache:
            return hint._cache["v_dp"] > hint._cache["v_dt"]
        doubler = state.turn
        if (self._fast_dec is not None
                and match_state.can_offer(doubler)):     # else C has no triple
            _, c = self._fast_dec.engine().match_offer(state, match_state)
            return c.v_dp > c.v_dt
        probs = self._probs_state(state)
        d = cube_decision(probs, match_state, doubler,
                          self._x(state), self.met)
        return d["take"]

    # ── checker play ────────────────────────────────────────────────

    def choose_checker_action_cubeful(
        self, state: BoardState, dice, match_state,
    ):
        """Score/cube-aware 0-ply move selection: argmax over
        successors of the mover's cubeful MWC. Returns
        (next_state, None) or None on a forced pass."""
        from modes import cube_perspective
        if self._fast_dec is not None:
            idx, nxt, _ = self._fast_dec.engine().match_best_move(
                state, dice, match_state)
            return None if idx < 0 else (nxt, None)
        mover = state.turn
        opponent = 1 - mover

        if self._fast:
            feats, lazy = _bg_fast.get_legal_plays_encoded(state, dice)
            n = len(lazy)
            if n == 0:
                return None
            # Materialize: the lazy views go stale on the next C call.
            succ = [lazy[i] for i in range(n)]
        else:
            plays = get_legal_plays(state, dice)
            if not plays:
                return None
            succ = [s for _, s in plays]
            n = len(succ)
            feats = np.stack(
                [self.encoder.encode(s) for s in succ]
            ).astype(np.float32)

        probs = self._forward(feats)          # opponent-on-roll view
        opp_persp = cube_perspective(match_state.cube_owner, opponent)
        ctx_opp = ctx_for(match_state, opponent)
        ctx_me = ctx_for(match_state, mover)

        terminal = feats[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
        best_i, best_v = -1, -np.inf
        for i in range(n):
            if terminal[i]:
                # Mover just won gr * cube points: exact MWC via MET.
                gr = succ[i].game_result()
                v_me = _me(self.met, ctx_me, ctx_me.cube * gr, True)
            else:
                x = (self.x_policy if self.x_policy is not None
                     else cube_x(succ[i]))
                v_me = 1.0 - cubeful_mwc(
                    probs[i], opp_persp, ctx_opp, x, self.met,
                )
            if v_me > best_v:
                best_v, best_i = v_me, i
        return succ[best_i], None

    # ── 1-ply value (distillation teacher) ──────────────────────────

    def choose_checker_action_oneply(self, state: BoardState, dice,
                                     match_state, topk: int = 0):
        """1-ply move selection with this agent's own evaluator: score each
        successor by value_oneply and take the mover's best. The mirror of
        TDAgent's oneply_moves on the teacher side.

        Exists to SIZE the teacher. --oracle-plies 1 distils this policy's
        values, so how much headroom distillation has depends on how much
        stronger this actually is than the 0-ply teacher -- which had been
        assumed rather than measured.

        `topk` > 0 re-prices only the K best successors by the cheap 0-ply
        read (same cheap-filters-expensive cascade as --oracle-1ply-eps).
        """
        from modes import cube_perspective
        mover = state.turn
        opponent = 1 - mover
        plays = get_legal_plays(state, dice)
        if not plays:
            return None
        succ = [s for _, s in plays]
        feats = np.stack([self.encoder.encode(x) for x in succ]).astype(np.float32)
        probs = self._forward(feats)
        opp_persp = cube_perspective(match_state.cube_owner, opponent)
        ctx_opp = ctx_for(match_state, opponent)
        ctx_me = ctx_for(match_state, mover)
        terminal = feats[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD

        shallow = []
        for i, sx in enumerate(succ):
            if terminal[i]:
                shallow.append(_me(self.met, ctx_me,
                                   ctx_me.cube * sx.game_result(), True))
            else:
                x = (self.x_policy if self.x_policy is not None
                     else cube_x(sx))
                shallow.append(1.0 - cubeful_mwc(probs[i], opp_persp,
                                                 ctx_opp, x, self.met))
        order = sorted(range(len(succ)), key=lambda i: -shallow[i])
        deep = order if topk <= 0 else order[:topk]
        vals = list(shallow)                      # mover-view MWC
        for i in deep:
            if not terminal[i]:
                # value_oneply prices the OPPONENT on roll at succ[i]
                vals[i] = 1.0 - self.value_oneply(succ[i], match_state)
        best = max(range(len(succ)), key=lambda i: vals[i])
        return succ[best], None

    def value_oneply(self, state: BoardState, match_state) -> float:
        """1-ply cubeful MWC for `state.turn` about to roll: average
        over the 21 dice outcomes of the mover's best successor by
        0-ply cubeful MWC. The teacher-side mirror of
        TDAgent.value_oneply_checker_matchplay (same tree shape, this
        agent's prob5+Janowski evaluator at the leaves), built as the
        --oracle-plies 1 distillation target.

        The cube is frozen for the ply; successor pricing is the
        standard live-cube Janowski composition, so cube leverage
        enters analytically rather than by branch enumeration.
        Terminal successors are priced exactly via the MET; a dice
        with no legal move is a pass, priced 1 - V0(same board,
        opponent on roll). One batched net forward covers every
        non-terminal successor of every dice.
        """
        if self._fast_dec is not None:
            return self._fast_dec.engine().match_value_1ply(state, match_state)
        from backgammon_engine import switch_turn
        from modes import cube_perspective
        mover = state.turn
        opponent = 1 - mover
        ctx_me = ctx_for(match_state, mover)
        ctx_opp = ctx_for(match_state, opponent)
        opp_persp = cube_perspective(match_state.cube_owner, opponent)

        feat_chunks = []
        batch_succ = []      # non-terminal successors, batch order
        per_dice = []        # (prob, n_batched, floor_value)
        pass_value = None    # dice-independent
        for d1 in range(1, 7):
            for d2 in range(d1, 7):
                prob = (1.0 if d1 == d2 else 2.0) / 36.0
                if self._fast:
                    feats, lazy = _bg_fast.get_legal_plays_encoded(
                        state, (d1, d2))
                    n = 0 if feats is None else feats.shape[0]
                    # Materialize: lazy views go stale on the next C
                    # call (the following dice iteration).
                    succ = [lazy[i] for i in range(n)]
                else:
                    plays = get_legal_plays(state, (d1, d2))
                    succ = [s for _, s in plays]
                    n = len(succ)
                    if n:
                        feats = np.stack(
                            [self.encoder.encode(s) for s in succ]
                        ).astype(np.float32)
                if n == 0:
                    if pass_value is None:
                        opp_view = switch_turn(state)
                        probs = self._probs_state(opp_view)
                        pass_value = 1.0 - cubeful_mwc(
                            probs, opp_persp, ctx_opp,
                            self._x(opp_view), self.met)
                    per_dice.append((prob, 0, pass_value))
                    continue
                terminal = feats[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
                best_term = -np.inf
                keep = []
                for i in range(n):
                    if terminal[i]:
                        gr = succ[i].game_result()
                        v = _me(self.met, ctx_me, ctx_me.cube * gr, True)
                        if v > best_term:
                            best_term = v
                    else:
                        keep.append(i)
                if keep:
                    feat_chunks.append(feats[keep])
                    batch_succ.extend(succ[i] for i in keep)
                per_dice.append((prob, len(keep), best_term))

        vals = None
        if batch_succ:
            probs_all = self._forward(
                np.concatenate(feat_chunks).astype(np.float32))
            vals = np.empty(len(batch_succ))
            for i, s in enumerate(batch_succ):
                x = (self.x_policy if self.x_policy is not None
                     else cube_x(s))
                vals[i] = 1.0 - cubeful_mwc(
                    probs_all[i], opp_persp, ctx_opp, x, self.met)
        total, pos = 0.0, 0
        for prob, n_batched, floor_v in per_dice:
            if n_batched:
                v = vals[pos:pos + n_batched].max()
                pos += n_batched
                if floor_v > v:
                    v = floor_v
            else:
                v = floor_v
            total += prob * v
        return float(total)


class _CDecisions:
    """The C engine behind JanowskiMatchAgent.enable_fast_decisions: a
    c_api.puretd_py.Engine(net, backend, janowski=True) with the agent's
    MET and x policy, re-opened when any of the three changes."""

    def __init__(self, agent: JanowskiMatchAgent, backend: str):
        from c_api import puretd_py
        puretd_py.ensure_lib(backend)    # built on demand (missing / stale ABI), as the C collector
        self._P = puretd_py
        self.agent = agent
        self.backend = backend
        self._open()

    def _version(self):
        return sum(p._version for p in self._params)

    def _open(self):
        a = self.agent
        self._params = list(a.network.parameters())
        self._ver = self._version()
        eng = self._P.Engine(a.network, self.backend, janowski=True)
        eng.set_met(a.met)
        eng.set_jan_x(a.x_policy)
        self._eng, self._met, self._x = eng, a.met, a.x_policy

    def engine(self):
        a = self.agent
        if (self._version() != self._ver or a.met is not self._met
                or a.x_policy != self._x):
            self._open()
        return self._eng
