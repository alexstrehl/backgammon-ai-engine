"""Agent wrapper for a :class:`model.ProbNetwork` (5-output prob5 net).

Exposes the :class:`agents.Agent` surface ``play_models.play_game``
needs. Cubeless-money only: prob5 nets have no cube policy, so cube
calls raise. 0-ply and 1-ply move selection only.
"""

import os
from typing import List, Sequence, Tuple

import numpy as np
import torch

from agents import Agent
from backgammon_engine import (
    BoardState, Play, switch_turn, get_legal_plays,
    get_legal_plays_encoded as _py_get_legal_plays_encoded,
)
from encoding import get_encoder, OPP_OFF_INDEX, TERMINAL_OFF_THRESHOLD
from model import ProbNetwork, prob5_postprocess, prob5_to_equity

try:
    import sys as _sys
    _c_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_engine")
    if _c_path not in _sys.path:
        _sys.path.insert(0, _c_path)
    import bg_fast as _bg_fast  # noqa: E402
    _BG_FAST_AVAILABLE = True
except Exception:
    _bg_fast = None
    _BG_FAST_AVAILABLE = False


PROB5_DEPTH_ERROR = "prob5 nets support 0-ply and 1-ply only (no 2-ply)"

# The 21 distinct dice outcomes (1/36 for doubles, 2/36 otherwise).
_DICE_OUTCOMES = [(d1, d2) for d1 in range(1, 7) for d2 in range(d1, 7)]


def _sanity_matrix() -> np.ndarray:
    """(196, 8) map from perspective196 features to gnubg SanityCheck's
    per-side counts: (ac, anCross, anGammonCross - 1) for my side, the
    same for the opponent, then my and the opponent's occupancy bitmask
    (bit i = gnubg point i >= 1 occupied, bar adds >= 2^24; exact in
    float32).
    gnubg side 1 (on roll) = our "my" block, side 0 = the opp block;
    gnubg point i (owner's point - 1, 24 = bar) is my block i, opp
    block 23 - i."""
    m = np.zeros((196, 8), dtype=np.float32)
    thermo = np.array([1.0, 1.0, 1.0, 2.0], dtype=np.float32)
    for side, base, bar in ((0, 0, 96), (1, 98, 194)):
        for k in range(24):
            i = k if side == 0 else 23 - k
            rows = slice(base + 4 * k, base + 4 * k + 4)
            m[rows, 3 * side] = thermo
            m[rows, 3 * side + 1] = thermo * (1 + i // 6)
            m[rows, 3 * side + 2] = thermo * (i // 6)
            if i:
                m[base + 4 * k, 6 + side] = 2.0 ** i   # first unit: 1 iff >= 1
        m[bar, 3 * side: 3 * side + 3] = (2.0, 10.0, 8.0)   # bar/2 * (1, 5, 4)
        m[bar, 6 + side] = 2.0 ** 25
    return m


_SANITY_M = _sanity_matrix()


def _prob5_impossible_np(x: np.ndarray):
    """gnubg 1.08 SanityCheck's "impossible" rules (eval.c) on (N, 196)
    perspective196 features, outputs from the on-roll player's view:
      opp has borne off -> P(wg) = P(wbg) = 0; I have -> P(lg) = P(lbg) = 0;
      else, no contact (anBack sum < 24): P(wbg) = 0 if opp anBack < 18,
      P(lbg) = 0 if my anBack < 18, P(wg) = 0 if my anCross >
      8 * opp anGammonCross, P(lg) = 0 if opp anCross > 8 * my
      anGammonCross - 4.
    Returns an (N, 4) bool mask over (wg, wbg, lg, lbg), or None when
    nothing is impossible."""
    s = x @ _SANITY_M
    # anBack: highest occupied point, 0 if none; >= 24 if on the bar.
    back = np.frexp(s[:, 6:] + 1.0)[1] - 1
    race = back.sum(axis=1) < 24
    off = s[:, 0:6:3] < 14.5
    if not (race.any() or off.any()):
        return None
    off_my, off_opp = off[:, 0], off[:, 1]
    return np.stack((
        off_opp | (race & (s[:, 1] > 8 * (s[:, 5] + 1))),
        off_opp | (race & (back[:, 1] < 18)),
        off_my | (race & (s[:, 4] > 8 * (s[:, 2] + 1) - 4)),
        off_my | (race & (back[:, 0] < 18)),
    ), axis=1)


def prob5_impossible(x: np.ndarray):
    """:func:`_prob5_impossible_np`, in C (bg_fast) when available."""
    if (_BG_FAST_AVAILABLE and x.dtype == np.float32
            and x.flags.c_contiguous and x.shape[1:] == (196,)):
        return _bg_fast.prob5_impossible(x)
    return _prob5_impossible_np(x)


def prob5_sanity(probs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """:func:`prob5_impossible` zeroing, then the
    :func:`model.prob5_postprocess` clamps (in numpy on CPU, same values).
    Certain-win/gammon and the contact noise cut are not applied."""
    mask = prob5_impossible(x.detach().cpu().numpy())
    if probs.device.type != "cpu" or probs.dtype != torch.float32:
        if mask is not None:
            probs = probs.masked_fill(torch.nn.functional.pad(
                torch.from_numpy(mask), (1, 0)).to(probs.device), 0.0)
        return prob5_postprocess(probs)
    p = probs.detach().numpy().copy()
    if mask is not None:
        p[:, 1:][mask] = 0.0
    np.minimum(p[:, 0], p[:, 1], out=p[:, 1])
    np.minimum(1.0 - p[:, 0], p[:, 3], out=p[:, 3])
    np.minimum(p[:, 1], p[:, 2], out=p[:, 2])
    np.minimum(p[:, 3], p[:, 4], out=p[:, 4])
    return torch.from_numpy(p)


def board_sanity_from_env(var: str, default: bool, environ=None) -> bool:
    """board_sanity from env ``var``: "1" on, "0" off, unset/empty
    ``default``. Raises ValueError on any other value."""
    env = os.environ if environ is None else environ
    v = env.get(var, "").strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"{var} must be 0 or 1, got {v!r}")
    return default if v == "" else v == "1"


def require_cpu(name: str, network, device) -> None:
    """Play-time prob5 / Prob5Jan / bo: agents run on CPU only."""
    on_gpu = torch.device(device).type != "cpu" or any(
        p.device.type != "cpu" for p in network.parameters())
    if on_gpu:
        raise ValueError(f"{name} runs on CPU only (got device={device!r}, "
                         "or a network not on the CPU)")


class ProbAgent(Agent):
    """Play-time wrapper for a trained :class:`ProbNetwork`.

    Values are money equity from the 5 outputs after the
    :func:`model.prob5_postprocess` clamps, or, with ``board_sanity``,
    after :func:`prob5_sanity` (gnubg's impossible-gammon zeroing +
    clamps; on for Prob5Jan, off by default for plain prob5). Move
    selection minimises opponent-view equity at ``plies`` 0 or 1.
    Training primitives are not implemented: prob5 training lives in
    ``train_prob5.py``.
    """

    def __init__(
        self,
        network: ProbNetwork,
        device: str = "cpu",
        bf16_inference: bool = False,
        plies: int = 0,
        use_fast_engine: bool = True,
        board_sanity: bool = False,
    ):
        if not isinstance(network, ProbNetwork):
            raise TypeError(
                f"ProbAgent expects a ProbNetwork, got {type(network).__name__}"
            )
        require_cpu(type(self).__name__, network, device)
        self.board_sanity = bool(board_sanity)
        if self.board_sanity and network.encoder_name != "perspective196":
            raise ValueError("board_sanity (prob5_sanity) reads perspective196 "
                             f"features (got {network.encoder_name!r})")
        if plies not in (0, 1):
            raise ValueError(PROB5_DEPTH_ERROR + f" (got plies={plies})")
        self.network = network.to(device).eval()
        self.device = device
        self._device_is_cpu = (
            isinstance(device, str) and device == "cpu"
        ) or (
            isinstance(device, torch.device) and device.type == "cpu"
        )
        self.encoder = get_encoder(network.encoder_name)
        self.output_mode = "prob5"
        self.is_cubeful = False
        self.plies = plies
        # Optional bf16 inference (faster matmuls, ~3-digit precision).
        self.bf16_inference = bool(bf16_inference)
        if self.bf16_inference:
            import copy
            self._infer_net = copy.deepcopy(self.network).to(torch.bfloat16)
            self._infer_dtype = torch.bfloat16
        else:
            self._infer_net = self.network
            self._infer_dtype = torch.float32

        # C fast path (move-gen + encode in one call), as in TDAgent.
        # perspective196 encoder on CPU only.
        self._fast_engine = (
            use_fast_engine
            and _BG_FAST_AVAILABLE
            and network.encoder_name == "perspective196"
            and self._device_is_cpu
        )
        _impl = (
            _bg_fast.get_legal_plays_encoded
            if self._fast_engine
            else _py_get_legal_plays_encoded
        )
        _enc = self.encoder

        def _bound(state, dice, _impl=_impl, _enc=_enc):
            return _impl(state, dice, _enc)

        self._get_legal_plays_encoded = _bound

    # ── encoding ──────────────────────────────────────────────────────

    def encode_state(self, state: BoardState) -> np.ndarray:
        if self._fast_engine:
            return _bg_fast.encode_state(state)
        return self.encoder.encode(state)

    def _encode_batch(self, states: Sequence[BoardState]) -> torch.Tensor:
        arr = np.stack([self.encode_state(s) for s in states])
        t = torch.from_numpy(arr)
        if not self._device_is_cpu:
            t = t.to(self.device, dtype=torch.float32)
        return t

    # ── value primitives ─────────────────────────────────────────────

    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.bf16_inference:
                p = self._infer_net(x.to(torch.bfloat16)).to(torch.float32)
            else:
                p = self.network(x)
            if self.network.raw_logits:
                p = torch.sigmoid(p)
            if self.board_sanity:
                return prob5_sanity(p, x)
        return prob5_postprocess(p)

    def _score(self, probs: torch.Tensor) -> torch.Tensor:
        """Cubeless money equity from the on-roll player's view:
        ``2*P(w) + P(wg) + P(wbg) - P(lg) - P(lbg) - 1``."""
        return prob5_to_equity(probs)

    def _terminal_mag(self, state: BoardState) -> float:
        """Magnitude of a terminal result (1/2/3)."""
        return float(state.game_result())

    def evaluate(self, state: BoardState) -> float:
        t = self._encode_batch([state])
        eq = self._score(self._probs(t))
        return float(eq.item())

    def evaluate_batch(self, states: Sequence[BoardState]) -> np.ndarray:
        t = self._encode_batch(states)
        eq = self._score(self._probs(t))
        return eq.detach().cpu().numpy()

    # ── 1-ply value primitive ────────────────────────────────────────

    def value_oneply_checker(self, state: BoardState) -> float:
        """1-ply equity at ``state`` from the current player's view: over
        all 21 dice outcomes, the best 0-ply response (minimum
        opponent-view equity), averaged; a game-ending successor gets
        its exact game_result(). Mirrors
        :meth:`td_agent.TDAgent.value_oneply_checker` (equity mode).
        """
        if state.is_game_over():
            # The current player just lost. Their view is -result.
            return -self._terminal_mag(state)

        oneply_sum = 0.0
        # ONE batched forward over every dice outcome's successors;
        # per-dice min and terminal correction in numpy afterwards.
        feat_chunks: List[np.ndarray] = []
        chunk_meta: List[tuple] = []   # (prob, start, end, term_overrides)
        pass_prob = 0.0
        offset = 0
        for d1, d2 in _DICE_OUTCOMES:
            prob = (1.0 / 36.0) if d1 == d2 else (2.0 / 36.0)
            features, next_states = self._get_legal_plays_encoded(state, (d1, d2))
            if len(next_states) == 0:
                # Forced pass: the same position for every passing dice,
                # evaluated once below.
                pass_prob += prob
                continue
            n = len(next_states)
            feat_chunks.append(features)
            # Capture terminal magnitudes NOW: the C fast path reuses its
            # successor objects on the next call. (`features` is a copy.)
            term = {
                int(j): -self._terminal_mag(next_states[int(j)])
                for j in np.flatnonzero(
                    features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
                )
            }
            chunk_meta.append((prob, offset, offset + n, term))
            offset += n

        # Forced-pass dice: opponent's 0-ply view at switch_turn(state), negated.
        if pass_prob > 0.0:
            oneply_sum += pass_prob * (-self.evaluate(switch_turn(state)))

        if feat_chunks:
            all_features = (
                feat_chunks[0] if len(feat_chunks) == 1
                else np.concatenate(feat_chunks, axis=0)
            )
            t = torch.from_numpy(all_features)
            if not self._device_is_cpu:
                t = t.to(self.device, dtype=torch.float32)
            opp_views_all = self._score(self._probs(t)).detach().cpu().numpy()
            for prob, s, e, term in chunk_meta:
                opp_views = opp_views_all[s:e]
                for j, tv in term.items():
                    opp_views[j] = tv
                # The current player picks the move minimising opp view.
                oneply_sum += prob * (-float(np.min(opp_views)))
        return oneply_sum

    # ── action selection ─────────────────────────────────────────────

    def choose_checker_action(
        self,
        state: BoardState,
        dice: Tuple[int, int],
        actions: List[Tuple[Play, BoardState]],
    ) -> Tuple[Play, BoardState]:
        """Pick the action whose resulting state minimises opponent
        equity: 0-ply argmin over network evaluations, or 1-ply
        lookahead at each candidate's resulting state.
        """
        if self.plies == 0:
            return self._choose_0ply(actions)
        return self._choose_1ply(actions)

    def _choose_0ply(
        self, actions: List[Tuple[Play, BoardState]],
    ) -> Tuple[Play, BoardState]:
        next_states = [s for _, s in actions]
        opp_values = self.evaluate_batch(next_states)
        # Terminal correction: a game-ending move gets its exact result,
        # else a winning move can be skipped.
        for j, ns in enumerate(next_states):
            if ns.is_game_over():
                opp_values[j] = -self._terminal_mag(ns)
        idx = int(np.argmin(opp_values))
        return actions[idx]

    def choose_next_state(self, state: BoardState, dice: Tuple[int, int]):
        """Return the successor BoardState of the selected move, or None
        when there is no legal play (caller applies ``switch_turn``).
        Selection is identical to ``choose_checker_action``. At 0-ply
        the C fast path fuses move generation + encoding into one call.
        """
        if self.plies != 0 or not self._fast_engine:
            plays = get_legal_plays(state, dice)
            if not plays:
                return None
            return self.choose_checker_action(state, dice, plays)[1]
        features, next_states = self._get_legal_plays_encoded(state, dice)
        if len(next_states) == 0:
            return None
        opp_values = (
            self._score(self._probs(torch.from_numpy(features)))
            .detach().cpu().numpy()
        )
        # Terminal correction from the encoding's mask; next_states is
        # still valid (no _bg_fast call since the fused call).
        for j in np.flatnonzero(
            features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
        ):
            opp_values[int(j)] = -self._terminal_mag(next_states[int(j)])
        idx = int(np.argmin(opp_values))
        return next_states[idx]  # __getitem__ materialises off the C buffer

    def _choose_1ply(
        self, actions: List[Tuple[Play, BoardState]],
    ) -> Tuple[Play, BoardState]:
        """1-ply: evaluate each candidate's resulting state at 1-ply
        (opp's view) and pick the minimum."""
        scores = np.empty(len(actions), dtype=np.float64)
        for i, (_, s) in enumerate(actions):
            if s.is_game_over():
                # I just won. Opp view = -result.
                scores[i] = -self._terminal_mag(s)
            else:
                scores[i] = self.value_oneply_checker(s)  # opp on roll, opp's view
        idx = int(np.argmin(scores))
        return actions[idx]

    # ── cube primitives (unsupported) ────────────────────────────────

    def offer_double(self, state, match_state):
        raise NotImplementedError(
            "ProbAgent does not support cube decisions — use "
            "--game-mode cubeless-money."
        )

    def respond_to_double(self, state, match_state, hint=None):
        raise NotImplementedError(
            "ProbAgent does not support cube decisions — use "
            "--game-mode cubeless-money."
        )
