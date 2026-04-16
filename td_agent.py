"""
td_agent.py -- Torch-based learning agents for backgammon.

TDAgent is a scalar agent built on TDNetwork. Two output modes
(inherited from the network):

  - "probability": sigmoid → P(win) ∈ [0, 1].
    Bootstrap target = 1 - V(opp). Terminal target = 1.0.
  - "equity":      linear → ±3 expected payoff.
    Bootstrap target = -V(opp). Terminal target = game_result (1/2/3).

See `modes.py` for the Mode × output_mode pairing matrix.

Fast path: at construction the agent imports `bg_fast` (from
`c_engine/`) when available and uses it for combined move-gen +
encoding inside `choose_checker_action`. Falls back to
`backgammon_engine` when the C library is missing or the encoder is
unsupported.
"""

from dataclasses import dataclass
import warnings
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents import Agent, CubeOffer
from backgammon_engine import (
    BoardState,
    get_legal_plays,
    get_legal_plays_encoded as _py_get_legal_plays_encoded,
    switch_turn,
)
from encoding import (
    CubefulEncoder, CubePerspective, get_encoder,
    OPP_OFF_INDEX, TERMINAL_OFF_THRESHOLD,
)
from model import TDNetwork


# All 21 distinct dice outcomes with probabilities (used by value_oneply_checker).
_DICE_OUTCOMES = [(d1, d2) for d1 in range(1, 7) for d2 in range(d1, 7)]

# Optional C engine fast path. Loaded lazily so a missing/unbuildable
# library doesn't prevent TDAgent from being importable.
try:
    import sys
    import os
    _here = os.path.dirname(os.path.abspath(__file__))
    _c_engine_path = os.path.join(_here, "c_engine")
    if _c_engine_path not in sys.path:
        sys.path.insert(0, _c_engine_path)
    import bg_fast as _bg_fast  # noqa: E402
    _BG_FAST_AVAILABLE = True
except Exception:
    _bg_fast = None
    _BG_FAST_AVAILABLE = False


def td_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """0.5 * MSE — TD/Tesauro convention.

    Plain `(pred - target)^2` has gradient `2 * (pred - target)`; the
    classical TD update wants  `(pred - target)`. The 0.5 absorbs factor of 2.
    Adam normalises by gradient moments so this is invisible to batch training.
    """
    return 0.5 * F.mse_loss(predictions, targets)


@dataclass
class TerminalOutcome:
    """Game-end outcome from the winner's perspective.
    """
    won_gammon: bool = False
    won_backgammon: bool = False
    cube_value: int = 1

    @property
    def game_result(self) -> int:
        """Point value of the win: 1 (single), 2 (gammon), 3 (backgammon)."""
        if self.won_backgammon:
            return 3
        if self.won_gammon:
            return 2
        return 1


class TDAgent(Agent):
    """Scalar agent built on TDNetwork. Supports probability and
    equity output modes (inherited from the network).
    """

    def __init__(
        self,
        network: TDNetwork,
        encoder=None,
        device: str = "cpu",
        loss_fn: Optional[Callable] = None,
        use_fast_engine: bool = True,
        oneply: bool = False,
    ):
        """
        Args:
            network: the neural network TDNetwork to wrap.
            encoder: state encoder. Defaults to the network's
                `encoder_name`.
            device: torch device.
            loss_fn: loss module. Defaults to `td_mse_loss` (0.5*MSE).
            use_fast_engine: when True (default) the C `bg_fast`
                fast path is used in `choose_checker_action` if it
                is loadable AND the encoder is `perspective196` AND
                the device is CPU. Set False to use python engine.
            oneply: when True, use 1-ply lookahead for cube decisions
                (offer_double / respond_to_double).
        """
        self.network = network.to(device)
        # Pin train mode once — TDNetwork has no Dropout/BatchNorm so
        # train vs eval is functionally identical, and the recursive
        # toggles cost ~13% of online TD step time when called per
        # iteration. torch.no_grad() handles gradient avoidance.
        self.network.train()
        self.device = device
        # Cache the CPU check so we can skip a no-op .to() per encode.
        self._device_is_cpu = (
            isinstance(device, str) and device == "cpu"
        ) or (
            isinstance(device, torch.device) and device.type == "cpu"
        )
        self.output_mode = getattr(network, "output_mode", "probability")
        if encoder is None:
            encoder_name = getattr(network, "encoder_name", "perspective196")
            encoder = get_encoder(encoder_name)
        self.encoder = encoder
        self.is_cubeful = isinstance(encoder, CubefulEncoder)
        if self.is_cubeful and self.output_mode != "equity":
            raise ValueError(
                "CubefulEncoder requires output_mode='equity'; "
                f"got {self.output_mode!r}."
            )
        if oneply and not self.is_cubeful:
            warnings.warn(
                "oneply=True ignored: requires a CubefulEncoder agent.",
                stacklevel=2,
            )
        self.oneply = oneply and self.is_cubeful
        self._loss_fn = loss_fn if loss_fn is not None else td_mse_loss
        # Pick the fused move-gen + encode impl. C and Python versions
        # share signature `(state, dice, encoder)` so the call site
        # has no branching. C is safe when its hard-coded encoding
        # matches our encoder.
        encoder_name = getattr(self.encoder, "name", "")
        self._fast_engine = (
            use_fast_engine
            and _BG_FAST_AVAILABLE
            and encoder_name == "perspective196"
            and self._device_is_cpu
        )
        # C base (perspective196) encode is usable for any agent
        # whose BASE encoder is perspective196 — including cubeful,
        # which wraps perspective196 and appends a cube one-hot in
        # Python after the C call.
        self._c_base_available = (
            use_fast_engine
            and _BG_FAST_AVAILABLE
            and self._device_is_cpu
            and (
                encoder_name == "perspective196"
                or (
                    self.is_cubeful
                    and self.encoder.base_name == "perspective196"
                )
            )
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

        # Single-state encoder: C when available, Python otherwise.
        # Used by collect_episode to cache the current-state encoding
        # so train_step doesn't have to re-encode during the training
        # pass.
        if self._fast_engine:
            self.encode_state = _bg_fast.encode_state
        else:
            _enc_python = self.encoder.encode

            def _encode_state(state, _enc=_enc_python):
                return _enc(state)

            self.encode_state = _encode_state

    # ── device / encoding helpers ────────────────────────────────────────

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        """Move tensor to the agent's device if not already on CPU."""
        if self._device_is_cpu:
            return t
        return t.to(self.device, dtype=torch.float32)

    def _encode(self, state: BoardState) -> np.ndarray:
        return self.encoder.encode(state)

    def _encode_batch(self, states: Sequence[BoardState]) -> torch.Tensor:
        arr = np.stack([self._encode(s) for s in states])
        t = torch.from_numpy(arr)
        t = self._to_device(t)
        return t

    # ── value primitive (no grad, scalar) ──────────────────────────────

    def evaluate(self, state: BoardState) -> float:
        t = torch.from_numpy(self._encode(state))
        t = self._to_device(t)
        x = t.unsqueeze(0)
        with torch.no_grad():
            v = self.network(x).item()
        return v

    def evaluate_batch(self, states: Sequence[BoardState]) -> np.ndarray:
        """Batched evaluate, returning a CPU numpy array. Training goes
        through `predict` which keeps tensors on device.
        """
        x = self._encode_batch(states)
        with torch.no_grad():
            v = self.network(x)
        return v.detach().cpu().numpy()

    # ── action selection ──────────────────────────────────────────────

    def _flip_value(self, opp_value: float) -> float:
        """Flip an opponent's value to the mover's perspective.
        Equity: -V. Probability: 1-V."""
        if self.output_mode == "equity":
            return -opp_value
        if self.output_mode == "probability":
            return 1.0 - opp_value
        raise ValueError(f"Unknown output_mode: {self.output_mode}")

    def choose_checker_action(self, state, dice, actions=None, with_target=False):
        """Pick one (Play_or_None, next_state) from the legal checker
        actions, or None on a forced pass.

        With `with_target=True` returns a 3-tuple
        `(Play_or_None, next_state, bootstrap_target)` — the target
        is read from the same forward pass used for argmin, so it's
        effectively free.

        - `actions=None`: agent generates moves + encoded features in
          one shot via `self._get_legal_plays_encoded` (C fast path
          if available). Play is returned as None — the trainer
          doesn't use it.
        - `actions=[(Play, BoardState), ...]`: explicit list, used by
          eval scripts that need the Play for logging. The agent
          encodes the supplied successors itself.
        """
        if actions is None:
            features, next_states = self._get_legal_plays_encoded(state, dice)
            if not next_states:
                return None
            t = torch.from_numpy(features)
            t = self._to_device(t)
            with torch.no_grad():
                values = self.network(t)
            idx = int(torch.argmin(values).item())
            chosen_state = next_states[idx]
            if with_target:
                target = self._flip_value(float(values[idx].item()))
                return None, chosen_state, target
            return None, chosen_state

        if not actions:
            return None
        features = np.stack([self.encoder.encode(s) for _, s in actions])
        t = torch.from_numpy(features)
        t = self._to_device(t)
        with torch.no_grad():
            values = self.network(t)
        idx = int(torch.argmin(values).item())
        chosen = actions[idx]
        if with_target:
            target = self._flip_value(float(values[idx].item()))
            return chosen[0], chosen[1], target
        return chosen

    # ── training-side primitives (called by Trainer) ──────────────────

    def bootstrap_target(self, next_state: BoardState) -> float:
        """Probability mode: 1 - V(next_state). Equity mode: -V(next_state)."""
        return self._flip_value(self.evaluate(next_state))

    def terminal_target(self, outcome: TerminalOutcome) -> float:
        """Money / DMP terminal target, in PER-UNIT equity (the
        network is trained at cube_value=1; cube_value is not folded
        into the target).

        Probability mode: 1.0 (cube and gammons ignored).
        Equity mode: game_result (1 single, 2 gammon, 3 backgammon).

        Matchplay subclasses override this with a match-equity table
        lookup.
        """
        if self.output_mode == "equity":
            return float(outcome.game_result)
        elif self.output_mode == "probability":
            return 1.0
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

    def predict(self, states: Sequence[BoardState]) -> torch.Tensor:
        """Forward pass for training: encode + network forward. Returns
        a 1-D tensor (one output per state, with grad)."""
        x = self._encode_batch(states)
        return self.network(x)

    def forward_encoded(self, features: np.ndarray) -> torch.Tensor:
        """Forward pass on a pre-encoded (N, num_features) numpy
        array. Skips the per-state encode loop — use this when the
        trainer has cached encodings from collection time.
        """
        t = torch.from_numpy(features)
        t = self._to_device(t)
        return self.network(t)

    def stack_targets(self, targets: Sequence[float]) -> torch.Tensor:
        """Float list → 1-D tensor on the agent's device."""
        return torch.as_tensor(targets, dtype=torch.float32, device=self.device)

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss_fn(predictions, targets)

    # ── 1-ply value lookahead ────────────────────────────────────────

    def value_oneply_checker(self, state: BoardState) -> float:
        """1-ply estimate of V(state) for a CHECKER decision point.

        Enumerates all 21 dice outcomes; for each, picks the best
        successor (argmin over the opponent's view) and accumulates
        the probability-weighted result. The flip rule (1-V or -V)
        depends on output_mode, same as bootstrap_target.

        Equity mode: terminal successors get the actual game_result
        (1/2/3 for single/gammon/backgammon), inspected on the
        materialised Python BoardState.

        Probability mode: terminal successors get value 0 (opp loses
        with certainty).
        """
        assert not state.is_game_over(), \
            "value_oneply_checker called on a terminal state"

        if self.output_mode == "probability":
            def flip(opp_value: float) -> float:
                return 1.0 - opp_value

            def terminal_opp_value(next_state: BoardState) -> float:
                return 0.0
        elif self.output_mode == "equity":
            def flip(opp_value: float) -> float:
                return -opp_value

            def terminal_opp_value(next_state: BoardState) -> float:
                # next_state.turn == opponent (engine switched), but
                # the WINNER is the previous mover. game_result()
                # returns 1/2/3 for single/gammon/backgammon based on
                # the board, independent of perspective.
                return -float(next_state.game_result())
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

        oneply_sum = 0.0
        for d1, d2 in _DICE_OUTCOMES:
            prob = (1.0 / 36.0) if d1 == d2 else (2.0 / 36.0)
            features, next_states = self._get_legal_plays_encoded(state, (d1, d2))

            if len(next_states) == 0:
                # Forced pass: opponent gets the dice on the same
                # position. Evaluate from opponent's view.
                v_opp = self.evaluate(switch_turn(state))
                oneply_sum += prob * flip(v_opp)
                continue

            t = torch.from_numpy(features)
            t = self._to_device(t)
            with torch.no_grad():
                values = self.network(t)

            # Terminal detection: feature[OPP_OFF_INDEX] is the MOVER's off/15
            # in the encoded successor. = 1 means that successor ends
            # the game with the MOVER winning; for each terminal
            # index we materialise the Python BoardState and read the
            # exact game result.
            terminal_mask = t[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
            if terminal_mask.any():
                # Clone to avoid in-place edits to the network output buffer.
                values = values.clone()
                for i in torch.nonzero(terminal_mask).flatten().tolist():
                    values[i] = terminal_opp_value(next_states[i])

            best_for_next = torch.min(values).item()
            oneply_sum += prob * flip(best_for_next)

        return oneply_sum

    # ── cubeful primitives ────────────────────────────────────────────
    #
    # These exist only when self.is_cubeful is True. They mirror the
    # cubeless methods but take a CubePerspective so the cube one-hot
    # is appended to every encoding.
    #
    # Scope: money-game cube actions only. Matchplay cube decisions
    # will live on a MatchplayTDAgent subclass that overrides
    # offer_double / respond_to_double / terminal_target.
    #
    # Value convention: the network learns the NORMALIZED per-unit
    # equity of a position (cube_value = 1). Actual equity at the
    # current stake is V × cube_value; the factor of 2 in offer_double
    # reflects the stake doubling after a take.

    def _encode_cubeful(
        self, state: BoardState, cube_state: CubePerspective,
    ) -> np.ndarray:
        """Encode `state` with the given cube perspective. Uses the C
        fast path for the base features when available; the cube
        one-hot append is delegated to CubefulEncoder.
        """
        if self._c_base_available:
            base = _bg_fast.encode_state(state)
        else:
            base = self.encoder._base.encode(state)
        return self.encoder.encode_with_base(base, cube_state)

    def evaluate_cubeful(
        self, state: BoardState, cube_state: CubePerspective,
    ) -> float:
        """Per-unit equity (cube_value=1) at `state` with this cube
        perspective. Multiply by actual cube_value externally."""
        x = self._encode_cubeful(state, cube_state)
        t = torch.from_numpy(x)
        t = self._to_device(t)
        with torch.no_grad():
            return self.network(t.unsqueeze(0)).item()

    @staticmethod
    def _money_cube_decision(
        v_pre_double: float,
        v_cube_theirs: float,
        pre_persp: CubePerspective,
        jacoby: bool,
    ) -> bool:
        """Money-game double decision given pre-computed equities.
        Shared by 0-ply and 1-ply offer_double paths."""
        v_double_take = 2.0 * v_cube_theirs
        opponent_will_pass = (v_double_take > 1.0)
        if opponent_will_pass:
            if jacoby and pre_persp == CubePerspective.CENTERED:
                return True  # Jacoby: cash, gammons don't count
            return 1.0 >= v_pre_double  # too good check
        return v_double_take > v_pre_double

    def offer_double(self, state: BoardState, match_state) -> CubeOffer:
        """Money-game pre-roll cube decision.

        Algorithm: per-unit equity at the current cube perspective vs.
        2× per-unit equity if doubled. Honours the Jacoby rule on
        centered cubes (collapses gammons → no "too good" branch).

        Matchplay agents override with match-equity-table logic
        (score-dependent take points, free drops, etc.).

        Caches v_cube_theirs in the returned CubeOffer so a self-play
        respond_to_double() call can skip the redundant forward pass.
        """
        if not self.is_cubeful:
            return super().offer_double(state, match_state)
        if self.oneply:
            return self._offer_double_oneply(state, match_state)
        from modes import cube_perspective  # local: modes imports td_agent
        player = state.turn
        no_double_persp = cube_perspective(match_state.cube_owner, player)
        if no_double_persp not in (CubePerspective.CENTERED, CubePerspective.MINE):
            return CubeOffer(should_double=False, _cache=None)

        v_pre_double = self.evaluate_cubeful(state, no_double_persp)
        v_cube_theirs = self.evaluate_cubeful(state, CubePerspective.THEIRS)
        should_double = self._money_cube_decision(
            v_pre_double, v_cube_theirs, no_double_persp, match_state.jacoby,
        )

        return CubeOffer(
            should_double=should_double,
            _cache={"v_cube_theirs_0ply": v_cube_theirs},
        )

    def respond_to_double(
        self,
        state: BoardState,
        match_state,
        hint: CubeOffer = None,
    ) -> bool:
        """Money-game take/pass response. Take iff 2·v_cube_theirs ≤ 1
        (equivalently, the responder's continuation equity beats the
        −1 they'd get by passing).

        Matchplay agents override this for score-aware take points.
        """
        if not self.is_cubeful:
            return super().respond_to_double(state, match_state, hint=hint)
        # Read cached v_cube_theirs at the matching resolution,
        # or recompute if not available.
        key = "v_cube_theirs_1ply" if self.oneply else "v_cube_theirs_0ply"
        if hint is not None and hint._cache is not None and key in hint._cache:
            v_cube_theirs = hint._cache[key]
        elif self.oneply:
            from dataclasses import replace
            from modes import CubeOwner
            opponent = 1 - state.turn
            theirs_ms = replace(
                match_state, cube_owner=CubeOwner(opponent + 1),
            )
            v_cube_theirs = self.value_oneply_checker_cubeful(state, theirs_ms)
        else:
            v_cube_theirs = self.evaluate_cubeful(state, CubePerspective.THEIRS)
        return (2.0 * v_cube_theirs) <= 1.0

    def _offer_double_oneply(self, state: BoardState, match_state) -> CubeOffer:
        """1-ply cube decision: v_pre_double and v_cube_theirs from
        value_oneply_checker_cubeful, then standard money logic."""
        from dataclasses import replace
        from modes import CubeOwner, cube_perspective

        player = state.turn
        pre_persp = cube_perspective(match_state.cube_owner, player)
        if pre_persp not in (CubePerspective.CENTERED, CubePerspective.MINE):
            return CubeOffer(should_double=False, _cache=None)

        v_pre_double = self.value_oneply_checker_cubeful(state, match_state)
        opponent = 1 - player
        theirs_ms = replace(
            match_state, cube_owner=CubeOwner(opponent + 1),
        )
        v_cube_theirs = self.value_oneply_checker_cubeful(state, theirs_ms)
        should_double = self._money_cube_decision(
            v_pre_double, v_cube_theirs, pre_persp, match_state.jacoby,
        )

        return CubeOffer(
            should_double=should_double,
            _cache={"v_cube_theirs_1ply": v_cube_theirs},
        )

    def choose_checker_action_cubeful(
        self, state: BoardState, dice, match_state,
    ):
        """Cubeful move selection. Returns `(next_state, bootstrap)`
        or None on forced pass.

        With self.oneply: 1-ply lookahead — for each candidate move,
        averages over opponent's 21 dice and their best reply, all
        batched in a single forward pass.

        Without: 0-ply argmin over opponent's network evaluation.
        """
        if not self.is_cubeful:
            return super().choose_checker_action_cubeful(
                state, dice, match_state,
            )

        plays = get_legal_plays(state, dice)
        if not plays:
            return None

        if self.oneply:
            return self._choose_checker_oneply_cubeful(
                state, plays, match_state,
            )

        from modes import cube_perspective
        opponent = 1 - state.turn
        opp_persp = cube_perspective(match_state.cube_owner, opponent)

        next_states = [s for _, s in plays]
        if self._c_base_available:
            base_feats = np.stack([
                _bg_fast.encode_state(s) for s in next_states
            ])
        else:
            base = self.encoder._base
            base_feats = np.stack([base.encode(s) for s in next_states])
        cube_col = np.zeros((len(next_states), 3), dtype=np.float32)
        cube_col[:, int(opp_persp)] = 1.0
        features = np.concatenate([base_feats, cube_col], axis=1)

        t = torch.from_numpy(features)
        t = self._to_device(t)
        with torch.no_grad():
            values = self.network(t)
        idx = int(torch.argmin(values).item())
        chosen = next_states[idx]
        bootstrap = -float(values[idx].item())
        return chosen, bootstrap

    def _choose_checker_oneply_cubeful(
        self, state: BoardState, plays, match_state,
    ):
        """1-ply cubeful move selection. For each candidate move,
        looks one ply ahead: enumerates opponent's 21 dice outcomes ×
        their legal replies, batches ALL resulting positions into a
        single forward pass, then picks the candidate whose expected
        equity (after opponent's best reply per dice) is highest.
        """
        from modes import CubeOwner, cube_perspective

        mover = state.turn
        our_persp = cube_perspective(match_state.cube_owner, mover)
        opp_col = int(our_persp)
        gammons_count = (
            not match_state.jacoby
            or match_state.cube_owner != CubeOwner.CENTERED
        )
        base = self.encoder._base
        n_feat = self.encoder.num_features
        n_base = base.num_features

        # Accumulate ALL encodings across candidates × opp_dice × opp_moves
        # into a flat list, then ONE forward pass.
        all_encs: List[np.ndarray] = []
        # Per-candidate: either a terminal equity, or a dict mapping
        # dice_idx → list of (enc_index | None, terminal_val | None).
        cand_terminal: List[Optional[float]] = [None] * len(plays)
        cand_dice: List[Optional[dict]] = [None] * len(plays)

        for m_idx, (_, next_state) in enumerate(plays):
            if next_state.is_game_over():
                if gammons_count:
                    cand_terminal[m_idx] = float(next_state.game_result())
                else:
                    cand_terminal[m_idx] = 1.0
                continue

            # Opponent is now on roll at next_state.
            dice_data: dict = {}
            for dice_idx, (d1, d2) in enumerate(_DICE_OUTCOMES):
                opp_plays = get_legal_plays(next_state, (d1, d2))
                resolved: List[Tuple[Optional[int], float]] = []
                if opp_plays:
                    for _, after_opp in opp_plays:
                        if after_opp.is_game_over():
                            # Opponent won — negative equity for us.
                            if gammons_count:
                                resolved.append(
                                    (None, -float(after_opp.game_result()))
                                )
                            else:
                                resolved.append((None, -1.0))
                        else:
                            # Back to us. Encode from our perspective.
                            if self._c_base_available:
                                b = _bg_fast.encode_state(after_opp)
                            else:
                                b = base.encode(after_opp)
                            enc = np.zeros(n_feat, dtype=np.float32)
                            enc[:n_base] = b
                            enc[n_base + opp_col] = 1.0
                            resolved.append((len(all_encs), None))
                            all_encs.append(enc)
                else:
                    # Forced pass — opponent can't move, play returns
                    # to us. Encode from OUR perspective (switch_turn
                    # back to us).
                    back_to_us = switch_turn(next_state)
                    if self._c_base_available:
                        b = _bg_fast.encode_state(back_to_us)
                    else:
                        b = base.encode(back_to_us)
                    enc = np.zeros(n_feat, dtype=np.float32)
                    enc[:n_base] = b
                    enc[n_base + opp_col] = 1.0
                    resolved.append((len(all_encs), None))
                    all_encs.append(enc)
                dice_data[dice_idx] = resolved
            cand_dice[m_idx] = dice_data

        # ONE forward pass.
        if all_encs:
            batch = np.stack(all_encs)
            t = torch.from_numpy(batch)
            t = self._to_device(t)
            with torch.no_grad():
                net_values = self.network(t).cpu().numpy().ravel()
        else:
            net_values = np.empty(0)

        probs = [(1.0 / 36.0) if d1 == d2 else (2.0 / 36.0)
                 for d1, d2 in _DICE_OUTCOMES]

        best_idx = 0
        best_val = float("-inf")
        for m_idx in range(len(plays)):
            if cand_terminal[m_idx] is not None:
                val = cand_terminal[m_idx]
            else:
                val = 0.0
                for dice_idx, resolved in cand_dice[m_idx].items():
                    # Opponent picks their best = min of our equity.
                    min_eq = min(
                        net_values[idx] if idx is not None else term
                        for idx, term in resolved
                    )
                    val += probs[dice_idx] * min_eq
            if val > best_val:
                best_val = val
                best_idx = m_idx

        chosen = plays[best_idx][1]
        return chosen, best_val

    def value_oneply_checker_cubeful(
        self, state: BoardState, match_state,
    ) -> float:
        """1-ply per-unit equity at `state` under the cube ownership
        in `match_state`, from the on-roll player's perspective.

        Enumerates all 21 dice outcomes; for each, picks the mover's
        best legal move (max over per-unit equity = −V(opp view) for
        non-terminal successors, or +game_result for terminals under
        Jacoby). All non-terminal encodings across all 21 outcomes
        are accumulated into a single batched forward pass.

        Intended for cubeful money training only. Matchplay needs a
        score-aware variant.
        """
        assert self.is_cubeful, \
            "value_oneply_checker_cubeful requires a CubefulEncoder agent"
        assert not state.is_game_over(), \
            "value_oneply_checker_cubeful called on a terminal state"

        from modes import CubeOwner, cube_perspective

        player = state.turn
        opponent = 1 - player
        opp_persp = cube_perspective(match_state.cube_owner, opponent)
        base = self.encoder._base
        n_feat = self.encoder.num_features
        n_base = base.num_features

        # Jacoby: gammons/bgs count iff the cube has been turned.
        gammons_count = (
            not match_state.jacoby
            or match_state.cube_owner != CubeOwner.CENTERED
        )

        # Per-dice results (list of equities, one per legal move) plus
        # per-dice chunks of non-terminal encodings. All chunks are
        # concatenated ONCE at the end and fed through the network in
        # a single forward pass.
        dice_results: List[Optional[List[float]]] = [None] * len(_DICE_OUTCOMES)
        dice_nonterm_indices: List[Optional[np.ndarray]] = [None] * len(_DICE_OUTCOMES)
        enc_chunks: List[np.ndarray] = []
        chunk_dice_idx: List[int] = []  # dice_idx per chunk (for fill-in)
        probs: List[float] = []
        opp_col = int(opp_persp)

        for dice_idx, (d1, d2) in enumerate(_DICE_OUTCOMES):
            probs.append((1.0 / 36.0) if d1 == d2 else (2.0 / 36.0))

            if self._c_base_available:
                # Fused C call: move-gen + 196-feature encode in one shot.
                base_feats, next_states_view = (
                    _bg_fast.get_legal_plays_encoded(state, (d1, d2))
                )
                n = len(next_states_view)
                if n == 0:
                    # Forced pass: encode the opponent-on-roll position
                    # once and store as a single-element chunk.
                    opp_view = switch_turn(state)
                    base_enc = _bg_fast.encode_state(opp_view)
                    enc = self.encoder.encode_with_base(base_enc, opp_persp)
                    dice_results[dice_idx] = [0.0]
                    dice_nonterm_indices[dice_idx] = np.array([0], dtype=np.int64)
                    enc_chunks.append(enc.reshape(1, -1))
                    chunk_dice_idx.append(dice_idx)
                    continue

                # Terminal detection: feature[OPP_OFF_INDEX] is mover's off/15;
                # ≥1 means the successor terminates with mover winning.
                terminal_mask = base_feats[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
                results_list: List[float] = [0.0] * n
                if terminal_mask.any():
                    if gammons_count:
                        for i in np.nonzero(terminal_mask)[0]:
                            results_list[int(i)] = float(
                                next_states_view[int(i)].game_result()
                            )
                    else:
                        for i in np.nonzero(terminal_mask)[0]:
                            results_list[int(i)] = 1.0
                    non_term_idx = np.nonzero(~terminal_mask)[0]
                else:
                    non_term_idx = np.arange(n, dtype=np.int64)

                dice_results[dice_idx] = results_list
                dice_nonterm_indices[dice_idx] = non_term_idx
                if len(non_term_idx) > 0:
                    # Build the (n_nonterm, 199) chunk: base feats +
                    # cube one-hot column, vectorised.
                    chunk = np.zeros(
                        (len(non_term_idx), n_feat),
                        dtype=np.float32,
                    )
                    chunk[:, :n_base] = base_feats[non_term_idx]
                    chunk[:, n_base + opp_col] = 1.0
                    enc_chunks.append(chunk)
                    chunk_dice_idx.append(dice_idx)
            else:
                plays = get_legal_plays(state, (d1, d2))
                if not plays:
                    opp_view = switch_turn(state)
                    base_enc = base.encode(opp_view)
                    enc = self.encoder.encode_with_base(base_enc, opp_persp)
                    dice_results[dice_idx] = [0.0]
                    dice_nonterm_indices[dice_idx] = np.array([0], dtype=np.int64)
                    enc_chunks.append(enc.reshape(1, -1))
                    chunk_dice_idx.append(dice_idx)
                    continue
                next_states = [s for _, s in plays]
                results_list = [0.0] * len(next_states)
                non_term_list: List[int] = []
                for i, ns in enumerate(next_states):
                    if ns.is_game_over():
                        if gammons_count:
                            results_list[i] = float(ns.game_result())
                        else:
                            results_list[i] = 1.0
                    else:
                        non_term_list.append(i)
                dice_results[dice_idx] = results_list
                if non_term_list:
                    chunk = np.stack([
                        self.encoder.encode_with_base(
                            base.encode(next_states[i]), opp_persp,
                        )
                        for i in non_term_list
                    ])
                    dice_nonterm_indices[dice_idx] = np.asarray(
                        non_term_list, dtype=np.int64,
                    )
                    enc_chunks.append(chunk)
                    chunk_dice_idx.append(dice_idx)

        # ONE batched forward pass for every non-terminal / pass encoding.
        if enc_chunks:
            batch = np.concatenate(enc_chunks, axis=0)
            t = torch.from_numpy(batch)
            t = self._to_device(t)
            with torch.no_grad():
                opp_values = self.network(t).cpu().numpy()
            offset = 0
            for chunk, di in zip(enc_chunks, chunk_dice_idx):
                n_rows = chunk.shape[0]
                idxs = dice_nonterm_indices[di]
                results_list = dice_results[di]
                for k in range(n_rows):
                    results_list[int(idxs[k])] = -float(opp_values[offset + k])
                offset += n_rows

        oneply_sum = 0.0
        for di, prob in enumerate(probs):
            equities = dice_results[di]
            if equities:
                oneply_sum += prob * max(equities)

        return oneply_sum

    # ── fast online TD update (bypass loss + optimizer machinery) ─────

    def td_update(self, state: BoardState, target: float, lr: float) -> float:
        """Classical online TD(0) update for a single state:

            v = V(state); v.backward()
            for p in params: p += lr * (target - v) * dV/dp

        Bypasses MSE, the optimizer wrapper, and train_step dispatch
        — about 17% faster than going through Trainer.train_step for
        the batch-of-1 case. Returns the squared TD error.
        """
        return self.td_update_encoded(self._encode(state), target, lr)

    def td_update_encoded(
        self, features: np.ndarray, target: float, lr: float,
    ) -> float:
        """Online TD(0) update given pre-encoded features. Used by the
        cubeful online trainer where encodings include a cube-state
        one-hot that the raw `_encode` method doesn't know about.
        """
        t = torch.from_numpy(features)
        t = self._to_device(t)
        x = t.unsqueeze(0)
        self.network.zero_grad()
        v = self.network(x)
        v.backward()
        td_error = target - v.item()
        with torch.no_grad():
            for p in self.network.parameters():
                if p.grad is not None:
                    p += lr * td_error * p.grad
        return td_error * td_error
