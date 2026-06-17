"""prob_agent.py -- Agent wrapper for a :class:`model.ProbNetwork`.

Exposes the minimal :class:`agents.Agent` surface ``play_models.play_game``
needs so a 5-output prob5 model can play H2H. Cubeless-money only — prob5
nets have no cube policy, so cube calls raise. Supports 0/1/2-ply action
selection; 2-ply pre-screens candidates within ``prune_threshold`` equity
(gnubg-style move filter, default 0.16) before the expensive re-evaluation.
"""

import os
from typing import List, Sequence, Tuple

import numpy as np
import torch

from agents import Agent
from backgammon_engine import (
    BoardState, Play, switch_turn,
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


# All 21 distinct dice outcomes with implicit probabilities (1/36 for
# doubles, 2/36 for non-doubles).
_DICE_OUTCOMES = [(d1, d2) for d1 in range(1, 7) for d2 in range(d1, 7)]


class ProbAgent(Agent):
    """Play-time wrapper for a trained :class:`ProbNetwork`.

    Implements the pieces of the :class:`Agent` interface that
    ``play_models.play_game`` actually calls at runtime:

    * :meth:`evaluate`, :meth:`evaluate_batch` — scalar money equity
      derived from the 5 probability outputs (postprocessed to satisfy
      the nested-event inequalities).
    * :meth:`choose_checker_action` — picks the action with minimum
      *opponent-view* equity at 0-ply, or applies 1-ply / 2-ply
      lookahead when ``plies > 0``. 2-ply uses a 0-ply pre-screen
      with ``prune_threshold`` to limit the move set before the
      expensive re-evaluation.

    Training primitives (bootstrap_target, terminal_target, predict,
    loss, stack_targets) are intentionally **not** implemented —
    prob5 training lives in ``train_prob5.py`` and has its own targets /
    loss outside the generic :class:`trainer.Trainer` path.
    """

    def __init__(
        self,
        network: ProbNetwork,
        device: str = "cpu",
        bf16_inference: bool = False,
        plies: int = 0,
        prune_threshold: float = 0.16,
        dmp: bool = False,
        use_fast_engine: bool = True,
    ):
        if not isinstance(network, ProbNetwork):
            raise TypeError(
                f"ProbAgent expects a ProbNetwork, got {type(network).__name__}"
            )
        if plies not in (0, 1, 2):
            raise ValueError(f"plies must be 0, 1, or 2 (got {plies})")
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
        self.prune_threshold = float(prune_threshold)
        # DMP scoring: state value is 2*P(win)-1 (in [-1,+1]); terminal
        # stakes collapse to ±1 (gammons/backgammons don't matter).
        self.dmp = bool(dmp)
        # Optional bf16 inference (~1.5-2x faster matmuls, ~3-digit precision).
        self.bf16_inference = bool(bf16_inference)
        if self.bf16_inference:
            import copy
            self._infer_net = copy.deepcopy(self.network).to(torch.bfloat16)
            self._infer_dtype = torch.bfloat16
        else:
            self._infer_net = self.network
            self._infer_dtype = torch.float32

        # C fast path (move-gen + perspective196 encode in one C call),
        # mirroring TDAgent. Safe only for the perspective196 encoder on
        # CPU. Used for 1-ply successor expansion in value_oneply_checker.
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
        return prob5_postprocess(p)

    def _score(self, probs: torch.Tensor) -> torch.Tensor:
        """State value from on-roll player's perspective.

        DMP: ``2*P(win) - 1`` (gammons/backgammons ignored).
        Otherwise: cubeless money equity ``2*P(w) + P(wg) + P(wbg) - P(lg) - P(lbg) - 1``.
        """
        if self.dmp:
            return 2.0 * probs[..., 0] - 1.0
        return prob5_to_equity(probs)

    def _terminal_mag(self, state: BoardState) -> float:
        """Magnitude of terminal-state result. ±1 in DMP, 1/2/3 otherwise."""
        return 1.0 if self.dmp else float(state.game_result())

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
        """1-ply equity at ``state`` from the current player's view.

        Enumerates all 21 dice outcomes; for each, finds the current
        player's best 0-ply response (the move minimising *opponent*-view
        equity at the resulting state), and averages. Mirrors
        :meth:`td_agent.TDAgent.value_oneply_checker` for the equity
        output mode, with terminal-state correction (a successor that
        ends the game gets its exact game_result() rather than the
        network's arbitrary output).
        """
        if state.is_game_over():
            # Edge case: state is itself terminal. Current player just
            # received the loss; their view is -result.
            return -self._terminal_mag(state)

        oneply_sum = 0.0
        # Gather every dice outcome's successor encodings into one buffer and
        # run a SINGLE batched forward (vs 21 small ones). Each play-dice chunk
        # records (prob, start, end, features, next_states) into the buffer; the
        # per-chunk min/terminal-correction is done in numpy afterwards.
        feat_chunks: List[np.ndarray] = []
        chunk_meta: List[tuple] = []   # (prob, start, end, features, next_states)
        pass_prob = 0.0
        offset = 0
        for d1, d2 in _DICE_OUTCOMES:
            prob = (1.0 / 36.0) if d1 == d2 else (2.0 / 36.0)
            features, next_states = self._get_legal_plays_encoded(state, (d1, d2))
            if len(next_states) == 0:
                # Forced pass: dice goes to opponent on the *same* board, so
                # the resulting position is identical for every passing dice —
                # accumulate the prob and evaluate it once below.
                pass_prob += prob
                continue
            n = len(next_states)
            feat_chunks.append(features)
            chunk_meta.append((prob, offset, offset + n, features, next_states))
            offset += n

        # Forced-pass dice (deduped): opponent's 0-ply view at switch_turn(state);
        # my view = -that.
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
            for prob, s, e, features, next_states in chunk_meta:
                opp_views = opp_views_all[s:e]
                # Terminal correction via the encoding (OPP_OFF feature ~1 means
                # the mover just bore off all 15 → opp, on roll here, lost),
                # avoiding materialising every successor. Opp view = -result.
                for j in np.flatnonzero(
                    features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
                ):
                    opp_views[j] = -self._terminal_mag(next_states[int(j)])
                # Current player picks the move minimising opp view.
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
        equity (equivalently: maximises mover equity).

        Dispatches on ``self.plies``:
          * 0 (default): 0-ply, direct argmin over network evaluations.
          * 1: 1-ply lookahead at each candidate's resulting state.
          * 2: 0-ply pre-screen of candidates with ``prune_threshold``,
            then 2-ply (= 1-ply for opponent) evaluation of survivors.
        """
        if self.plies == 0:
            return self._choose_0ply(actions)
        if self.plies == 1:
            return self._choose_1ply(actions)
        return self._choose_2ply(actions)

    def _choose_0ply(
        self, actions: List[Tuple[Play, BoardState]],
    ) -> Tuple[Play, BoardState]:
        next_states = [s for _, s in actions]
        opp_values = self.evaluate_batch(next_states)
        # Terminal correction: a move that ends the game (mover won) gets
        # its exact result, not the network's arbitrary output on the
        # terminal encoding — otherwise a winning move can be skipped.
        for j, ns in enumerate(next_states):
            if ns.is_game_over():
                opp_values[j] = -self._terminal_mag(ns)
        idx = int(np.argmin(opp_values))
        return actions[idx]

    def _choose_1ply(
        self, actions: List[Tuple[Play, BoardState]],
    ) -> Tuple[Play, BoardState]:
        """1-ply: for each candidate, evaluate the resulting state at
        1-ply (opp's view) and pick the move minimising it."""
        scores = np.empty(len(actions), dtype=np.float64)
        for i, (_, s) in enumerate(actions):
            if s.is_game_over():
                # Game over after my move = I just won. Opp view = -result.
                scores[i] = -self._terminal_mag(s)
            else:
                scores[i] = self.value_oneply_checker(s)  # opp on roll, opp's view
        idx = int(np.argmin(scores))
        return actions[idx]

    def _choose_2ply(
        self, actions: List[Tuple[Play, BoardState]],
    ) -> Tuple[Play, BoardState]:
        """2-ply with 0-ply pre-screen.

        Mirrors gnubg's move-filter approach: at the top level, screen
        all candidates at 0-ply and keep only those within
        ``prune_threshold`` (in equity units, default 0.16) of the best
        0-ply candidate. Then re-evaluate survivors at 2-ply (= 1-ply
        for opponent at next state) and pick the best.

        Pruning is only applied at this outer level: the 1-ply sub-call
        leaves are already at 0-ply (the network), so no deeper pruning
        is meaningful at depth 2.
        """
        next_states = [s for _, s in actions]
        opp_views_0ply = self.evaluate_batch(next_states)
        # Apply terminal correction at depth 1 too, so a winning move
        # never gets pruned by a noisy network output.
        for j, ns in enumerate(next_states):
            if ns.is_game_over():
                opp_views_0ply[j] = -self._terminal_mag(ns)
        # Move-filter: keep only candidates within prune_threshold of best
        # (best = lowest opp view = highest current-player view).
        best_opp = float(np.min(opp_views_0ply))
        keep_mask = opp_views_0ply <= best_opp + self.prune_threshold
        kept_idx = np.flatnonzero(keep_mask)

        if kept_idx.size == 1:
            return actions[int(kept_idx[0])]

        # Re-evaluate survivors at 2-ply.
        scores_2ply = np.empty(kept_idx.size, dtype=np.float64)
        for k, i in enumerate(kept_idx):
            s = next_states[int(i)]
            if s.is_game_over():
                scores_2ply[k] = -self._terminal_mag(s)
            else:
                scores_2ply[k] = self.value_oneply_checker(s)
        best_in_kept = int(np.argmin(scores_2ply))
        return actions[int(kept_idx[best_in_kept])]

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
