"""
td_agent.py -- TDAgent: the scalar (TDNetwork) playing/learning agent.

The agent encodes positions, runs the net, and picks chequer moves and
cube actions at 0-, 1- or 2-ply. Two output modes, inherited from the
network:

  - "probability": sigmoid → P(win) ∈ [0, 1].
    Bootstrap target = 1 - V(opp). Terminal target = 1.0.
  - "equity":      linear → ±3 expected payoff.
    Bootstrap target = -V(opp). Terminal target = game_result (1/2/3).

See `modes.py` for the Mode × output_mode pairing matrix.

Move generation + encoding uses the C engine (`c_engine/bg_fast`) when
it is loadable and the encoder is supported; otherwise
`backgammon_engine`.
"""

import os
from dataclasses import dataclass
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


# The 21 distinct dice outcomes.
_DICE_OUTCOMES = [(d1, d2) for d1 in range(1, 7) for d2 in range(d1, 7)]

# TD_STRUCT_CHECKS=1 enables debug asserts in the cubeful 1-ply selector.
_STRUCT_CHECKS = os.environ.get("TD_STRUCT_CHECKS") == "1"

# Probability of each _DICE_OUTCOMES entry (doubles 1/36, others 2/36).
_ONEPLY_PROBS = np.array(
    [(1.0 / 36.0) if d1 == d2 else (2.0 / 36.0) for d1, d2 in _DICE_OUTCOMES],
    dtype=np.float64,
)


def cpu_supports_fast_bf16() -> bool:
    """True if the CPU has bf16 matmul instructions (AVX512-BF16 or
    AMX-BF16). Without them torch emulates bf16, far slower than fp32,
    so bf16 inference must stay off.
    """
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
        return "avx512_bf16" in flags or "amx_bf16" in flags
    except OSError:
        return False


_BF16_UNSUPPORTED_WARNED = False

# Optional C engine; TDAgent stays importable without it.
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
    """0.5 * MSE, so the gradient is `(pred - target)` as in the
    classical TD update. Invisible under Adam."""
    return 0.5 * F.mse_loss(predictions, targets)


@dataclass
class TerminalOutcome:
    """Game-end outcome from the winner's perspective."""
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
    """Scalar agent built on TDNetwork (probability or equity output
    mode, inherited from the network)."""

    def __init__(
        self,
        network: TDNetwork,
        encoder=None,
        device: str = "cpu",
        loss_fn: Optional[Callable] = None,
        use_fast_engine: bool = True,
        oneply: bool = False,
        twoply_k: int = 0,
        boltzmann_temp: float = 0.0,
        bf16_inference: bool = False,
    ):
        """
        Args:
            network: the TDNetwork to wrap.
            encoder: state encoder. Defaults to the network's
                `encoder_name`.
            device: torch device.
            loss_fn: loss module. Defaults to `td_mse_loss`.
            use_fast_engine: use the C `bg_fast` engine when it is
                loadable, the encoder is `perspective196` and the
                device is CPU. False = Python engine.
            oneply: 1-ply lookahead for chequer moves and cube
                decisions.
            twoply_k: > 0 = 2-ply chequer selection (gnubg's
                convention: two dice+move expansions before the static
                eval), cubeful or cubeless. At the depth-2 expansion
                only the top k candidates (ranked by 0-ply value) are
                deepened. Root candidates are always fully expanded.
                Takes precedence over `oneply` for chequer choice.
            boltzmann_temp: softmax exploration temperature for
                chequer play. 0.0 = greedy argmin.
            bf16_inference: run 1-ply value forwards on a bf16 copy of
                the net. Training keeps fp32 weights; meant for
                collection workers.
        """
        self.network = network.to(device)
        # Pinned in train mode: TDNetwork has no Dropout/BatchNorm, and
        # per-call train()/eval() toggles are measurably slow.
        self.network.train()
        self.device = device
        # Cached so encode paths can skip a no-op .to().
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
        self.oneply = oneply
        self.twoply_k = int(twoply_k)
        self.boltzmann_temp = boltzmann_temp
        self._loss_fn = loss_fn if loss_fn is not None else td_mse_loss
        # Fused move-gen + encode: C only when its hard-coded encoding
        # (perspective196) matches ours.
        encoder_name = getattr(self.encoder, "name", "")
        self._fast_engine = (
            use_fast_engine
            and _BG_FAST_AVAILABLE
            and encoder_name == "perspective196"
            and self._device_is_cpu
        )
        # C base encode also serves cubeful agents whose base encoder
        # is perspective196 (the cube one-hot is appended in Python).
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

        # Single-state encoder (C or Python); collect_episode caches
        # its output so train_step need not re-encode.
        if self._fast_engine:
            self.encode_state = _bg_fast.encode_state
        else:
            _enc_python = self.encoder.encode

            def _encode_state(state, _enc=_enc_python):
                return _enc(state)

            self.encode_state = _encode_state

        # bf16 inference copy for 1-ply values. Built from the current
        # fp32 weights; call `refresh_bf16_inference()` after a reload.
        self.bf16_inference = bool(bf16_inference)
        if self.bf16_inference and self._device_is_cpu \
                and not cpu_supports_fast_bf16():
            global _BF16_UNSUPPORTED_WARNED
            if not _BF16_UNSUPPORTED_WARNED:
                print(
                    "[td_agent] bf16 inference requested but this CPU has no "
                    "bf16 ISA (avx512_bf16/amx_bf16); torch emulation is "
                    "~20x slower than fp32 — disabling bf16 inference.",
                    flush=True,
                )
                _BF16_UNSUPPORTED_WARNED = True
            self.bf16_inference = False
        self._bf16_net: Optional[nn.Module] = None
        if self.bf16_inference:
            self._build_bf16_net()

        # Transposition table for 2-ply cubeless search: board key ->
        # 1-ply value. Exact because the net is frozen during play.
        self._tt_oneply: Optional[dict] = {} if self.twoply_k > 0 else None
        self._tt_max = 2_000_000  # entry cap; cleared wholesale if exceeded

    def _build_bf16_net(self) -> None:
        """(Re)build the bf16 inference copy from self.network."""
        import copy
        net = copy.deepcopy(self.network)
        net.eval()
        net = net.to(torch.bfloat16)
        self._bf16_net = net

    def refresh_bf16_inference(self) -> None:
        """Rebuild the bf16 copy after the fp32 weights change. No-op
        when `bf16_inference` is False."""
        if self.bf16_inference:
            self._build_bf16_net()

    # ── device / encoding helpers ────────────────────────────────────────

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the agent's device (no-op on CPU)."""
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
        """Batched evaluate as a CPU numpy array (no grad). Training
        uses `predict`, which keeps tensors on device."""
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
        """Pick one (Play_or_None, next_state) from the legal chequer
        actions, or None on a forced pass.

        `with_target=True` returns `(Play_or_None, next_state,
        bootstrap_target)`. The target comes from the same forward pass.

        - `actions=None`: moves + encodings come from
          `self._get_legal_plays_encoded`. Play is returned as None.
        - `actions=[(Play, BoardState), ...]`: explicit list (eval
          scripts that need the Play); the agent encodes the successors.
        """
        if self.twoply_k > 0 and not self.is_cubeful:
            return self._choose_checker_twoply_cubeless(
                state, dice, actions=actions,
            )
        if self.oneply and not self.is_cubeful:
            return self._choose_checker_oneply_cubeless(
                state, dice, actions=actions, with_target=with_target,
            )
        if actions is None and self.is_cubeful:
            # The fused path emits base features only. Cubeful agents
            # need the cube columns, so use the explicit-actions path.
            actions = get_legal_plays(state, dice)
            if not actions:
                return None
        if actions is None:
            features, next_states = self._get_legal_plays_encoded(state, dice)
            if not next_states:
                return None
            t = torch.from_numpy(features)
            t = self._to_device(t)
            with torch.no_grad():
                values = self.network(t)
            values = self._apply_terminal_mask(values, t, next_states)
            idx = self._boltzmann_select(values)
            chosen_state = next_states[idx]
            if with_target:
                target = self._flip_value(float(values[idx].item()))
                return None, chosen_state, target
            return None, chosen_state

        if not actions:
            return None
        if self.is_cubeful:
            # Chequer decision without cube context (eval scripts):
            # encode with a centered cube.
            features = np.stack([
                self.encoder.encode(s, CubePerspective.CENTERED)
                for _, s in actions
            ])
        else:
            features = np.stack([self.encoder.encode(s) for _, s in actions])
        t = torch.from_numpy(features)
        t = self._to_device(t)
        with torch.no_grad():
            values = self.network(t)
        next_states = [s for _, s in actions]
        values = self._apply_terminal_mask(values, t, next_states)
        idx = self._boltzmann_select(values)
        chosen = actions[idx]
        if with_target:
            target = self._flip_value(float(values[idx].item()))
            return chosen[0], chosen[1], target
        return chosen

    def _boltzmann_select(self, values: torch.Tensor) -> int:
        """Index sampled by softmax over negated opponent-view values
        (lower is better for us). Argmin when temp=0."""
        if self.boltzmann_temp <= 0.0 or len(values) == 1:
            return int(torch.argmin(values).item())
        logits = -values.squeeze() / self.boltzmann_temp
        probs = torch.softmax(logits, dim=0)
        return int(torch.multinomial(probs, 1).item())

    def _terminal_opp_value(self, next_state: BoardState) -> float:
        """Opponent-view value of a terminal next_state (the mover just
        won; next_state.turn is the loser). Replaces the net's
        arbitrary output on terminal encodings."""
        if self.output_mode == "probability":
            return 0.0
        return -float(next_state.game_result())

    def _apply_terminal_mask(self, values, features_tensor, next_states):
        """Overwrite values[i] with the true opp-view terminal value
        where the encoding says the game is over (OPP_OFF feature >= 1).
        Unmasked, argmin can skip a winning move."""
        terminal_mask = features_tensor[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
        if not terminal_mask.any():
            return values
        values = values.clone()
        for i in torch.nonzero(terminal_mask).flatten().tolist():
            values[i] = self._terminal_opp_value(next_states[i])
        return values

    # ── training-side primitives (called by Trainer) ──────────────────

    def bootstrap_target(self, next_state: BoardState) -> float:
        """Probability mode: 1 - V(next_state). Equity mode: -V(next_state)."""
        return self._flip_value(self.evaluate(next_state))

    def terminal_target(self, outcome: TerminalOutcome) -> float:
        """Money / DMP terminal target in per-unit equity (cube_value
        is not folded in). Probability mode: 1.0. Equity mode:
        game_result (1/2/3)."""
        if self.output_mode == "equity":
            return float(outcome.game_result)
        elif self.output_mode == "probability":
            return 1.0
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

    def predict(self, states: Sequence[BoardState]) -> torch.Tensor:
        """Encode + forward with grad; 1-D tensor, one value per state."""
        x = self._encode_batch(states)
        return self.network(x)

    def forward_encoded(self, features: np.ndarray) -> torch.Tensor:
        """Forward with grad on a pre-encoded (N, num_features) array
        (encodings cached at collection time)."""
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
        """1-ply estimate of V(state) at a chequer decision: for each
        of the 21 dice, the best successor (argmin over the opponent's
        view, flipped as in bootstrap_target), probability-weighted.
        Terminal successors (the mover just won) score game_result
        (equity) or 1 (probability) for the mover.

        Kept separate from _value_oneply_checker_batched, whose
        transposition table is only valid with a frozen net: this is
        the training-target path (trainer.collect_episode).
        """
        assert not state.is_game_over(), \
            "value_oneply_checker called on a terminal state"

        oneply_sum = 0.0
        for idx, (d1, d2) in enumerate(_DICE_OUTCOMES):
            prob = float(_ONEPLY_PROBS[idx])
            features, next_states = self._get_legal_plays_encoded(state, (d1, d2))

            if len(next_states) == 0:
                # Forced pass: opponent rolls on the same position.
                v_opp = self.evaluate(switch_turn(state))
                oneply_sum += prob * self._flip_value(v_opp)
                continue

            t = torch.from_numpy(features)
            t = self._to_device(t)
            with torch.no_grad():
                values = self.network(t)

            # feature[OPP_OFF_INDEX] is the mover's off/15 in the
            # successor. 1 means the mover just won.
            terminal_mask = t[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
            if terminal_mask.any():
                values = values.clone()
                for i in torch.nonzero(terminal_mask).flatten().tolist():
                    values[i] = self._terminal_opp_value(next_states[i])

            best_for_next = torch.min(values).item()
            oneply_sum += prob * self._flip_value(best_for_next)

        return oneply_sum

    def _choose_checker_oneply_cubeless(
        self, state, dice, actions=None, with_target=False,
    ):
        """Cubeless 1-ply move selection in one batched forward.

        Per candidate s_next: terminal (we won) scores our terminal
        value. Otherwise every opp reply over the 21 dice is encoded
        (our view, since the engine swaps turn) and the candidate scores
            E_dice[ min over opp replies of our-view value ].
        Pick the max.
        """
        if actions is None:
            features, next_states = self._get_legal_plays_encoded(state, dice)
            if not next_states:
                return None
            plays = [(None, ns) for ns in next_states]
        else:
            plays = actions
            if not plays:
                return None

        is_equity = self.output_mode == "equity"

        def _our_terminal_value(s: BoardState) -> float:
            # Magnitude only (game_result 1/2/3, or 1 in probability
            # mode); the caller applies the sign.
            if is_equity:
                return float(s.game_result())
            return 1.0

        all_encs: List[np.ndarray] = []
        total_rows = 0
        cand_terminal: List[Optional[float]] = [None] * len(plays)
        # Per candidate: dict[dice_idx] -> list of (row_idx_or_None,
        # terminal_or_None); min over the list is opp's best reply.
        cand_dice: List[Optional[dict]] = [None] * len(plays)

        for m_idx, (_, next_state) in enumerate(plays):
            if next_state.is_game_over():
                # We just won.
                cand_terminal[m_idx] = _our_terminal_value(next_state)
                continue

            dice_data: dict = {}
            for dice_idx, (d1, d2) in enumerate(_DICE_OUTCOMES):
                if self._c_base_available:
                    base_feats, opp_successors = (
                        _bg_fast.get_legal_plays_encoded(next_state, (d1, d2))
                    )
                else:
                    opp_plays_py = get_legal_plays(next_state, (d1, d2))
                    opp_successors = [s for _, s in opp_plays_py]
                    base_feats = None
                n = len(opp_successors)

                resolved: List[Tuple[Optional[int], Optional[float]]] = []

                if n > 0:
                    if base_feats is not None:
                        terminal_mask_np = (
                            base_feats[:, OPP_OFF_INDEX]
                            >= TERMINAL_OFF_THRESHOLD
                        )
                        non_term_idx = np.nonzero(~terminal_mask_np)[0]
                    else:
                        term_bools = [
                            opp_successors[i].is_game_over()
                            for i in range(n)
                        ]
                        non_term_idx = np.array(
                            [i for i, t in enumerate(term_bools) if not t],
                            dtype=np.int64,
                        )

                    non_term_slots: List[int] = []
                    for i in range(n):
                        if base_feats is not None:
                            is_terminal = bool(terminal_mask_np[i])
                        else:
                            is_terminal = term_bools[i]
                        if is_terminal:
                            # Opp won: our value is negative.
                            our_term_val = (
                                -float(opp_successors[i].game_result())
                                if is_equity else 0.0
                            )
                            resolved.append((None, our_term_val))
                        else:
                            non_term_slots.append(len(resolved))
                            resolved.append((None, None))

                    n_nt = len(non_term_idx)
                    if n_nt > 0:
                        # Leaves are our view (engine swapped turn), so
                        # the net output is our value directly.
                        n_feat = self.encoder.num_features
                        chunk = np.zeros((n_nt, n_feat), dtype=np.float32)
                        if base_feats is not None:
                            chunk[:] = base_feats[non_term_idx]
                        else:
                            for k, i in enumerate(non_term_idx):
                                chunk[k] = self.encoder.encode(
                                    opp_successors[int(i)]
                                )
                        start = total_rows
                        total_rows += n_nt
                        all_encs.append(chunk)
                        for k, slot in enumerate(non_term_slots):
                            resolved[slot] = (start + k, None)
                else:
                    # Forced pass: we roll at switch_turn(next_state).
                    # One bucket entry, so min() returns itself.
                    back_to_us = switch_turn(next_state)
                    if self._c_base_available:
                        b = _bg_fast.encode_state(back_to_us)
                    else:
                        b = self.encoder.encode(back_to_us)
                    n_feat = self.encoder.num_features
                    enc = np.zeros((1, n_feat), dtype=np.float32)
                    enc[0] = b
                    resolved.append((total_rows, None))
                    total_rows += 1
                    all_encs.append(enc)

                dice_data[dice_idx] = resolved
            cand_dice[m_idx] = dice_data

        if all_encs:
            batch = np.concatenate(all_encs, axis=0)
            t = torch.from_numpy(batch)
            t = self._to_device(t)
            with torch.no_grad():
                net_values = self.network(t).cpu().numpy().ravel()
        else:
            net_values = np.empty(0)

        probs = _ONEPLY_PROBS.tolist()

        best_idx = 0
        best_val = float("-inf")
        for m_idx in range(len(plays)):
            if cand_terminal[m_idx] is not None:
                val = cand_terminal[m_idx]
            else:
                exp_val = 0.0
                for dice_idx, resolved in cand_dice[m_idx].items():
                    # Opp picks the reply worst for us.
                    min_our = min(
                        net_values[idx] if idx is not None else term
                        for idx, term in resolved
                    )
                    exp_val += probs[dice_idx] * min_our
                val = exp_val
            if val > best_val:
                best_val = val
                best_idx = m_idx

        chosen_play, chosen_state = plays[best_idx]
        if with_target:
            # best_val is our 1-ply value at `state`.
            return chosen_play, chosen_state, best_val
        return chosen_play, chosen_state

    def _forward_inference(self, t: torch.Tensor) -> torch.Tensor:
        """No-grad forward, in bf16 when the bf16 copy exists, else
        fp32. Always returns fp32."""
        with torch.no_grad():
            if self.bf16_inference and self._bf16_net is not None:
                return self._bf16_net(t.to(torch.bfloat16)).to(torch.float32)
            return self.network(t)

    def _value_oneply_checker_batched(self, state: BoardState) -> float:
        """value_oneply_checker with one batched forward across all 21
        dice, bf16 when available, and a transposition-table memo.
        Used by value_twoply_checker."""
        assert not state.is_game_over(), \
            "_value_oneply_checker_batched called on a terminal state"

        tt = self._tt_oneply
        if tt is not None:
            tt_key = state._key()
            hit = tt.get(tt_key)
            if hit is not None:
                return hit

        # Collect the features of every dice outcome into one stack.
        # Forced passes take the 0-ply opp value of the swapped state.
        feat_chunks: List[np.ndarray] = []
        dice_slices: List[
            Tuple[Optional[int], Optional[int], float, Optional[float]]
        ] = []
        term_overrides: List[Tuple[int, float]] = []  # (global_idx, value)
        pos = 0
        for idx, (d1, d2) in enumerate(_DICE_OUTCOMES):
            prob = float(_ONEPLY_PROBS[idx])
            features, next_states_view = self._get_legal_plays_encoded(
                state, (d1, d2),
            )

            if len(next_states_view) == 0:
                v_opp = self.evaluate(switch_turn(state))
                dice_slices.append((None, None, prob, self._flip_value(v_opp)))
                continue

            n = len(next_states_view)
            # Terminals need the materialised state for game_result().
            terminal_mask_np = (
                features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
            )
            if terminal_mask_np.any():
                for i in np.nonzero(terminal_mask_np)[0]:
                    term_overrides.append(
                        (pos + int(i),
                         self._terminal_opp_value(next_states_view[int(i)])),
                    )

            feat_chunks.append(features)
            end = pos + n
            dice_slices.append((pos, end, prob, None))
            pos = end

        # Every dice a forced pass: no forward needed.
        if pos == 0:
            total = 0.0
            for _, _, prob, forced in dice_slices:
                if forced is not None:
                    total += prob * forced
            if tt is not None:
                if len(tt) >= self._tt_max:
                    tt.clear()
                tt[tt_key] = total
            return total

        big_feat = (
            feat_chunks[0] if len(feat_chunks) == 1
            else np.concatenate(feat_chunks, axis=0)
        )
        t = torch.from_numpy(big_feat)
        t = self._to_device(t)
        values = self._forward_inference(t)

        if term_overrides:
            values = values.clone()
            for idx, v in term_overrides:
                values[idx] = v

        oneply_sum = 0.0
        for start, end, prob, forced in dice_slices:
            if forced is not None:
                oneply_sum += prob * forced
                continue
            best_for_next = torch.min(values[start:end]).item()
            oneply_sum += prob * self._flip_value(best_for_next)

        if tt is not None:
            if len(tt) >= self._tt_max:
                tt.clear()
            tt[tt_key] = oneply_sum
        return oneply_sum

    def value_twoply_checker(
        self, state: BoardState, k: int = 3,
    ) -> float:
        """2-ply estimate of V(state) at a cubeless chequer decision
        (gnubg's convention: two dice+move expansions before the static
        eval). From the mover's view:
            E_{D1}[ max_{m1} ( flip( E_{D2}[ max_{m2}( flip( V(s2) ) ) ] ) ) ]

        Depth-1 candidates are ranked by 0-ply value and only the top-K
        non-terminal ones are deepened to a 1-ply call (same filter as
        value_twoply_checker_cubeful_filtered). Cost: one batched 0-ply
        forward plus up to 21*K 1-ply calls.
        """
        assert not state.is_game_over(), \
            "value_twoply_checker called on a terminal state"
        assert k >= 1, f"k must be >= 1, got {k}"

        # Phase 1: all outer-dice candidates in one batched forward for
        # the 0-ply ranking.
        feat_chunks: List[np.ndarray] = []
        per_dice: List[
            Tuple[Optional[int], Optional[int], float, Optional[float], list]
        ] = []  # (start, end, prob, forced_val_or_None, materialized_next_states)
        term_overrides: List[Tuple[int, float]] = []  # (global_idx, value)
        pos = 0
        for idx, (d1, d2) in enumerate(_DICE_OUTCOMES):
            prob = float(_ONEPLY_PROBS[idx])
            features, next_states_view = self._get_legal_plays_encoded(
                state, (d1, d2),
            )

            if len(next_states_view) == 0:
                # Forced pass: resolved with a 1-ply call in phase 2.
                per_dice.append((None, None, prob, None, []))  # sentinel "forced"
                continue

            # Materialise now: the next _bg_fast call invalidates the
            # shared C buffer.
            n = len(next_states_view)
            next_states = [next_states_view[i] for i in range(n)]

            terminal_mask_np = (
                features[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
            )
            if terminal_mask_np.any():
                for i in np.nonzero(terminal_mask_np)[0]:
                    term_overrides.append(
                        (pos + int(i),
                         self._terminal_opp_value(next_states[int(i)])),
                    )

            feat_chunks.append(features)
            end = pos + n
            per_dice.append((pos, end, prob, None, next_states))
            pos = end

        # Forced-pass value, computed once (same for every such dice).
        forced_resolved: Optional[float] = None
        if pos > 0:
            big_feat = (
                feat_chunks[0] if len(feat_chunks) == 1
                else np.concatenate(feat_chunks, axis=0)
            )
            t = torch.from_numpy(big_feat)
            t = self._to_device(t)
            all_values = self._forward_inference(t).clone()
            if term_overrides:
                for idx, v in term_overrides:
                    all_values[idx] = v
        else:
            all_values = None

        # Phase 2: per dice, best opp-view value (mover wants min);
        # deepen the top-K non-terminal candidates to 1-ply.
        twoply_sum = 0.0
        for start, end, prob, forced, next_states in per_dice:
            if forced is None and start is None:
                if forced_resolved is None:
                    forced_resolved = self._value_oneply_checker_batched(
                        switch_turn(state),
                    )
                twoply_sum += prob * self._flip_value(forced_resolved)
                continue
            if forced is not None:
                twoply_sum += prob * forced
                continue

            slc = all_values[start:end]
            best_opp_view = float("inf")

            # Terminal rows of this slice, from term_overrides.
            slice_terms = [
                (idx - start, v) for idx, v in term_overrides
                if start <= idx < end
            ]
            terminal_local = {i for i, _ in slice_terms}

            if slice_terms:
                term_min = min(v for _, v in slice_terms)
                if term_min < best_opp_view:
                    best_opp_view = term_min

            non_term_idx = [
                i for i in range(end - start) if i not in terminal_local
            ]
            if non_term_idx:
                if len(non_term_idx) <= k:
                    top_k_local = non_term_idx
                else:
                    nt_pairs = [(float(slc[i].item()), i) for i in non_term_idx]
                    nt_pairs.sort(key=lambda x: x[0])
                    top_k_local = [i for _, i in nt_pairs[:k]]

                for li in top_k_local:
                    cand = next_states[li]
                    opp_oneply = self._value_oneply_checker_batched(cand)
                    if opp_oneply < best_opp_view:
                        best_opp_view = opp_oneply

            twoply_sum += prob * self._flip_value(best_opp_view)

        return twoply_sum

    def _choose_checker_twoply_cubeless(
        self, state, dice, actions=None,
    ):
        """Cubeless 2-ply chequer selection: every root candidate is
        scored by flipping opp's value_twoply_checker at next_state
        (terminals by the mover's terminal value). Pick the max.
        `twoply_k` filters only the depth-2 expansion, never the root.

        Returns `(Play_or_None, next_state)`; no with_target.
        """
        if actions is None:
            features, next_states = self._get_legal_plays_encoded(state, dice)
            if not next_states:
                return None
            plays = [(None, ns) for ns in next_states]
        else:
            plays = actions
            if not plays:
                return None

        is_equity = self.output_mode == "equity"

        def our_terminal_value(s: BoardState) -> float:
            if is_equity:
                return float(s.game_result())
            return 1.0

        k = max(self.twoply_k, 1)

        best_value = float("-inf")
        best_idx = 0
        for m_idx, (_, next_state) in enumerate(plays):
            if next_state.is_game_over():
                mover_eq = our_terminal_value(next_state)
            else:
                opp_value = self.value_twoply_checker(next_state, k=k)
                mover_eq = self._flip_value(opp_value)

            if mover_eq > best_value:
                best_value = mover_eq
                best_idx = m_idx

        chosen = plays[best_idx]
        return chosen[0], chosen[1]

    # ── cubeful primitives ────────────────────────────────────────────
    #
    # Only for self.is_cubeful agents. Money-game cube actions only.
    #
    # Value convention: the net learns per-unit equity (cube_value = 1).
    # Actual equity is V x cube_value. The factor 2 in offer_double is
    # the stake doubling after a take.

    def _encode_cubeful(
        self, state: BoardState, cube_state: CubePerspective,
        is_cube_action: bool = False,
    ) -> np.ndarray:
        """Encode `state` with the given cube perspective (C base
        features when available)."""
        if self._c_base_available:
            base = _bg_fast.encode_state(state)
        else:
            base = self.encoder._base.encode(state)
        return self.encoder.encode_with_base(base, cube_state, is_cube_action)

    def evaluate_cubeful(
        self, state: BoardState, cube_state: CubePerspective,
        is_cube_action: bool,
    ) -> float:
        """Per-unit equity at `state` with this cube perspective."""
        x = self._encode_cubeful(state, cube_state, is_cube_action)
        t = torch.from_numpy(x)
        t = self._to_device(t)
        with torch.no_grad():
            return self.network(t.unsqueeze(0)).item()

    @staticmethod
    def _money_cube_decision(
        v_no_double: float,
        v_double_take: float,
        pre_persp: CubePerspective,
        jacoby: bool,
    ) -> bool:
        """Money-game double decision, shared by all ply depths.

        v_no_double:   equity if we play on (is_cube_action=0).
        v_double_take: 2 x V(theirs, is_cube_action=0), equity if we
                       double and opp takes.
        """
        opponent_will_pass = (v_double_take > 1.0)
        if opponent_will_pass:
            if jacoby and pre_persp == CubePerspective.CENTERED:
                return True  # Jacoby: cash, gammons don't count
            return 1.0 >= v_no_double  # too good check
        return v_double_take > v_no_double

    def offer_double(self, state: BoardState, match_state) -> CubeOffer:
        """Money-game pre-roll cube decision: per-unit equity at the
        current cube vs 2x per-unit equity if doubled, with the Jacoby
        rule on centered cubes. The returned CubeOffer caches
        v_double_take so respond_to_double() can skip a forward.
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

        if self.twoply_k > 0:
            # 2-ply cube: same depth as the chequer selector.
            from dataclasses import replace
            from modes import CubeOwner
            opponent = 1 - player
            theirs_ms = replace(
                match_state, cube_owner=CubeOwner(opponent + 1),
            )
            v_no_double = self.value_twoply_checker_cubeful_filtered(
                state, match_state, k=self.twoply_k,
            )
            v_double_take = 2.0 * self.value_twoply_checker_cubeful_filtered(
                state, theirs_ms, k=self.twoply_k,
            )
            should_double = self._money_cube_decision(
                v_no_double, v_double_take, no_double_persp, match_state.jacoby,
            )
            return CubeOffer(
                should_double=should_double,
                _cache={
                    "v_double_take_2ply": v_double_take,
                    "v_no_double_2ply": v_no_double,
                },
            )

        # Both evaluations in one forward.
        x_nd = self._encode_cubeful(state, no_double_persp, is_cube_action=False)
        x_dt = self._encode_cubeful(state, CubePerspective.THEIRS, is_cube_action=False)
        t = torch.from_numpy(np.stack([x_nd, x_dt]))
        t = self._to_device(t)
        with torch.no_grad():
            values = self.network(t).flatten()
        v_no_double = float(values[0].item())
        v_double_take = 2.0 * float(values[1].item())
        should_double = self._money_cube_decision(
            v_no_double, v_double_take, no_double_persp, match_state.jacoby,
        )

        return CubeOffer(
            should_double=should_double,
            _cache={
                "v_double_take_0ply": v_double_take,
                "v_no_double_0ply": v_no_double,
            },
        )

    def respond_to_double(
        self,
        state: BoardState,
        match_state,
        hint: CubeOffer = None,
    ) -> bool:
        """Money-game take/pass: take iff v_double_take <= 1 (the
        responder's continuation beats the -1 of passing)."""
        if not self.is_cubeful:
            return super().respond_to_double(state, match_state, hint=hint)
        # Use the cached v_double_take at the matching ply, else recompute.
        if self.twoply_k > 0:
            key = "v_double_take_2ply"
        elif self.oneply:
            key = "v_double_take_1ply"
        else:
            key = "v_double_take_0ply"
        if hint is not None and hint._cache is not None and key in hint._cache:
            v_double_take = hint._cache[key]
        elif self.twoply_k > 0:
            from dataclasses import replace
            from modes import CubeOwner
            opponent = 1 - state.turn
            theirs_ms = replace(
                match_state, cube_owner=CubeOwner(opponent + 1),
            )
            v_double_take = 2.0 * self.value_twoply_checker_cubeful_filtered(
                state, theirs_ms, k=self.twoply_k,
            )
        elif self.oneply:
            from dataclasses import replace
            from modes import CubeOwner
            opponent = 1 - state.turn
            theirs_ms = replace(
                match_state, cube_owner=CubeOwner(opponent + 1),
            )
            v_double_take = 2.0 * self.value_oneply_checker_cubeful(state, theirs_ms)
        else:
            v_double_take = 2.0 * self.evaluate_cubeful(state, CubePerspective.THEIRS, is_cube_action=False)
        return v_double_take <= 1.0

    def _offer_double_oneply(self, state: BoardState, match_state) -> CubeOffer:
        """1-ply cube decision: both equities from
        value_oneply_checker_cubeful, then the money logic."""
        from dataclasses import replace
        from modes import CubeOwner, cube_perspective

        player = state.turn
        pre_persp = cube_perspective(match_state.cube_owner, player)
        if pre_persp not in (CubePerspective.CENTERED, CubePerspective.MINE):
            return CubeOffer(should_double=False, _cache=None)

        v_no_double = self.value_oneply_checker_cubeful(state, match_state)
        opponent = 1 - player
        theirs_ms = replace(
            match_state, cube_owner=CubeOwner(opponent + 1),
        )
        v_double_take = 2.0 * self.value_oneply_checker_cubeful(state, theirs_ms)
        should_double = self._money_cube_decision(
            v_no_double, v_double_take, pre_persp, match_state.jacoby,
        )

        return CubeOffer(
            should_double=should_double,
            _cache={
                "v_double_take_1ply": v_double_take,
                "v_no_double_1ply": v_no_double,
            },
        )

    def choose_checker_action_cubeful(
        self, state: BoardState, dice, match_state,
    ):
        """Cubeful move selection at 0-, 1- or 2-ply. Returns
        `(next_state, bootstrap)` or None on a forced pass."""
        if not self.is_cubeful:
            return super().choose_checker_action_cubeful(
                state, dice, match_state,
            )

        # 2-ply needs full Play objects, so no C fast path.
        if self.twoply_k > 0:
            plays = get_legal_plays(state, dice)
            if not plays:
                return None
            return self._choose_checker_twoply_cubeful(
                state, plays, match_state, k=self.twoply_k,
            )

        # C fast path: move-gen + encode in one call. The 1-ply selector
        # only reads next_state from each tuple, so (None, next_state)
        # is fine.
        if self._c_base_available:
            base_feats, _lazy = _bg_fast.get_legal_plays_encoded(state, dice)
            n_plays = len(_lazy)
            if n_plays == 0:
                return None
            if self.oneply:
                # Materialise now: the next bg_fast call invalidates the
                # shared C buffer.
                plays = [(None, _lazy[i]) for i in range(n_plays)]
                return self._choose_checker_oneply_cubeful(
                    state, plays, match_state,
                )
            # 0-ply: no bg_fast call before the lazy reads below, so only
            # terminal rows and the chosen row are materialised.
            next_states = _lazy
        else:
            plays = get_legal_plays(state, dice)
            if not plays:
                return None

            if self.oneply:
                return self._choose_checker_oneply_cubeful(
                    state, plays, match_state,
                )

            next_states = [s for _, s in plays]
            base = self.encoder._base
            base_feats = np.stack([base.encode(s) for s in next_states])

        from modes import CubeOwner, cube_perspective
        opponent = 1 - state.turn
        opp_persp = cube_perspective(match_state.cube_owner, opponent)
        opp_cube_action = match_state.can_offer(opponent)
        # Encoder's cube block: 3-hot perspective + is_cube_action bit.
        extra = np.zeros(
            (len(next_states), self.encoder.cube_features),
            dtype=np.float32,
        )
        extra[:, int(opp_persp)] = 1.0
        if opp_cube_action:
            extra[:, 3] = 1.0
        features = np.concatenate([base_feats, extra], axis=1)

        t = torch.from_numpy(features)
        t = self._to_device(t)
        with torch.no_grad():
            values = self.network(t)
        # Terminals (mover just won): true opp-view value instead of the
        # net's output; -1.0 when Jacoby voids gammons.
        terminal_mask = t[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
        if terminal_mask.any():
            gammons_count = (
                not match_state.jacoby
                or match_state.cube_owner != CubeOwner.CENTERED
            )
            values = values.clone()
            for i in torch.nonzero(terminal_mask).flatten().tolist():
                gr = float(next_states[i].game_result())
                values[i] = -gr if gammons_count else -1.0
        idx = self._boltzmann_select(values)
        chosen = next_states[idx]
        bootstrap = -float(values[idx].item())
        return chosen, bootstrap

    def _choose_checker_oneply_cubeful(
        self, state: BoardState, plays, match_state,
    ):
        """1-ply cubeful move selection, one batched forward. Each
        candidate is scored over opp's 21 dice and best reply, with
        opp's cube decision at next_state simulated when opp can offer.

        Two per-unit expectations per candidate:
          U_A = our equity if opp does not double (cube unchanged)
          U_B = our equity if opp doubles and we take (cube ours;
                x2 for the actual stake)
        Opp decides from its side (V_no_double_opp = -U_A,
        V_double_take_opp = -2*U_B). If opp doubles we take iff
        V_double_take_opp <= 1, realising 2*U_B on take or -1 on pass.
        Otherwise the candidate is worth U_A.
        """
        from modes import CubeOwner, cube_perspective

        mover = state.turn
        opponent = 1 - mover
        opp_can_offer = match_state.can_offer(opponent)
        opp_persp_at_next = cube_perspective(match_state.cube_owner, opponent)


        # Scenario A: opp doesn't double.
        our_persp_A = cube_perspective(match_state.cube_owner, mover)
        after_opp_is_cube_action_A = match_state.can_offer(mover)
        gammons_count_A = (
            not match_state.jacoby
            or match_state.cube_owner != CubeOwner.CENTERED
        )

        # Scenario B: opp doubles and we take. The cube is ours, so we
        # can always offer, and gammons always count.
        our_persp_B = CubePerspective.MINE

        base = self.encoder._base
        n_feat = self.encoder.num_features
        n_base = base.num_features
        opp_col_A = int(our_persp_A)
        opp_col_B = int(our_persp_B)

        # All encodings (candidates x dice x replies x scenarios) as
        # chunks, concatenated once for the batched forward; resolved_A/B
        # entries index rows by `total_rows`.
        all_encs: List[np.ndarray] = []
        total_rows = 0
        cand_terminal: List[Optional[float]] = [None] * len(plays)
        cand_dice_A: List[Optional[dict]] = [None] * len(plays)
        cand_dice_B: List[Optional[dict]] = [None] * len(plays)

        for m_idx, (_, next_state) in enumerate(plays):
            if next_state.is_game_over():
                # We won; no opp cube decision.
                if gammons_count_A:
                    cand_terminal[m_idx] = float(next_state.game_result())
                else:
                    cand_terminal[m_idx] = 1.0
                continue

            dice_data_A: dict = {}
            dice_data_B: Optional[dict] = {} if opp_can_offer else None

            for dice_idx, (d1, d2) in enumerate(_DICE_OUTCOMES):
                if self._c_base_available:
                    base_feats, opp_successors = (
                        _bg_fast.get_legal_plays_encoded(next_state, (d1, d2))
                    )
                else:
                    opp_plays_py = get_legal_plays(next_state, (d1, d2))
                    opp_successors = [s for _, s in opp_plays_py]
                    base_feats = None
                n = len(opp_successors)

                resolved_A: List[Tuple[Optional[int], float]] = []
                resolved_B: Optional[List[Tuple[Optional[int], float]]] = (
                    [] if opp_can_offer else None
                )

                if n > 0:
                    if base_feats is not None:
                        terminal_mask_np = (
                            base_feats[:, OPP_OFF_INDEX]
                            >= TERMINAL_OFF_THRESHOLD
                        )
                        non_term_idx = np.nonzero(~terminal_mask_np)[0]
                    else:
                        term_bools = [
                            opp_successors[i].is_game_over()
                            for i in range(n)
                        ]
                        non_term_idx = np.array(
                            [i for i, t in enumerate(term_bools) if not t],
                            dtype=np.int64,
                        )

                    # resolved_A/B in opp-move order; non-terminal slots
                    # are placeholders filled once their row index is known.
                    non_term_slots_A: List[int] = []
                    non_term_slots_B: List[int] = []
                    for i in range(n):
                        if base_feats is not None:
                            is_terminal = bool(terminal_mask_np[i])
                        else:
                            is_terminal = term_bools[i]
                        if is_terminal:
                            gr = float(opp_successors[i].game_result())
                            resolved_A.append(
                                (None, -gr if gammons_count_A else -1.0)
                            )
                            if opp_can_offer:
                                # Gammons always count in scenario B.
                                resolved_B.append((None, -gr))
                        else:
                            non_term_slots_A.append(len(resolved_A))
                            resolved_A.append(None)  # filled below
                            if opp_can_offer:
                                non_term_slots_B.append(len(resolved_B))
                                resolved_B.append(None)

                    n_nt = len(non_term_idx)
                    if n_nt > 0:
                        chunk_A = np.zeros((n_nt, n_feat), dtype=np.float32)
                        if base_feats is not None:
                            chunk_A[:, :n_base] = base_feats[non_term_idx]
                        else:
                            for k, i in enumerate(non_term_idx):
                                chunk_A[k, :n_base] = base.encode(
                                    opp_successors[int(i)]
                                )
                        chunk_A[:, n_base + opp_col_A] = 1.0
                        if after_opp_is_cube_action_A:
                            chunk_A[:, n_base + 3] = 1.0
                        start_A = total_rows
                        total_rows += n_nt
                        all_encs.append(chunk_A)
                        for k, slot in enumerate(non_term_slots_A):
                            resolved_A[slot] = (start_A + k, None)

                        if opp_can_offer:
                            chunk_B = np.zeros((n_nt, n_feat), dtype=np.float32)
                            if base_feats is not None:
                                chunk_B[:, :n_base] = base_feats[non_term_idx]
                            else:
                                chunk_B[:, :n_base] = chunk_A[:, :n_base]
                            chunk_B[:, n_base + opp_col_B] = 1.0
                            chunk_B[:, n_base + 3] = 1.0
                            start_B = total_rows
                            total_rows += n_nt
                            all_encs.append(chunk_B)
                            for k, slot in enumerate(non_term_slots_B):
                                resolved_B[slot] = (start_B + k, None)
                else:
                    # Forced pass: play returns to us.
                    back_to_us = switch_turn(next_state)
                    if self._c_base_available:
                        b = _bg_fast.encode_state(back_to_us)
                    else:
                        b = base.encode(back_to_us)

                    enc_A = np.zeros((1, n_feat), dtype=np.float32)
                    enc_A[0, :n_base] = b
                    enc_A[0, n_base + opp_col_A] = 1.0
                    if after_opp_is_cube_action_A:
                        enc_A[0, n_base + 3] = 1.0
                    resolved_A.append((total_rows, None))
                    total_rows += 1
                    all_encs.append(enc_A)

                    if opp_can_offer:
                        enc_B = np.zeros((1, n_feat), dtype=np.float32)
                        enc_B[0, :n_base] = b
                        enc_B[0, n_base + opp_col_B] = 1.0
                        enc_B[0, n_base + 3] = 1.0
                        resolved_B.append((total_rows, None))
                        total_rows += 1
                        all_encs.append(enc_B)

                dice_data_A[dice_idx] = resolved_A
                if opp_can_offer:
                    dice_data_B[dice_idx] = resolved_B

            cand_dice_A[m_idx] = dice_data_A
            if opp_can_offer:
                cand_dice_B[m_idx] = dice_data_B

        # One batched forward over both scenarios.
        if all_encs:
            batch = np.concatenate(all_encs, axis=0)
            t = torch.from_numpy(batch)
            t = self._to_device(t)
            with torch.no_grad():
                net_values = self.network(t).cpu().numpy().ravel()
        else:
            net_values = np.empty(0)

        probs = _ONEPLY_PROBS.tolist()
        assert abs(sum(probs) - 1.0) < 1e-9, \
            "dice probabilities must sum to 1"

        # Debug (TD_STRUCT_CHECKS=1): every non-terminal candidate has
        # 21 dice buckets per scenario, each with >= 1 resolution.
        if _STRUCT_CHECKS:
            for _m in range(len(plays)):
                if cand_terminal[_m] is None:
                    assert cand_dice_A[_m] is not None \
                        and len(cand_dice_A[_m]) == 21, \
                        f"cand {_m}: scenario A missing dice buckets"
                    for _r in cand_dice_A[_m].values():
                        assert len(_r) >= 1, \
                            f"cand {_m}: empty resolution list in scenario A"
                    if opp_can_offer:
                        assert cand_dice_B[_m] is not None \
                            and len(cand_dice_B[_m]) == 21, \
                            f"cand {_m}: scenario B missing dice buckets"
                        for _r in cand_dice_B[_m].values():
                            assert len(_r) >= 1, \
                                f"cand {_m}: empty resolution list in scenario B"

        best_idx = 0
        best_val = float("-inf")
        for m_idx in range(len(plays)):
            if cand_terminal[m_idx] is not None:
                val = cand_terminal[m_idx]
            else:
                U_A = 0.0
                for dice_idx, resolved in cand_dice_A[m_idx].items():
                    min_eq = min(
                        net_values[idx] if idx is not None else term
                        for idx, term in resolved
                    )
                    U_A += probs[dice_idx] * min_eq

                if opp_can_offer:
                    U_B = 0.0
                    for dice_idx, resolved in cand_dice_B[m_idx].items():
                        min_eq = min(
                            net_values[idx] if idx is not None else term
                            for idx, term in resolved
                        )
                        U_B += probs[dice_idx] * min_eq

                    # Opp's cube decision, from opp's side.
                    V_no_double_opp = -U_A
                    V_double_take_opp = -2.0 * U_B
                    opp_doubles = self._money_cube_decision(
                        V_no_double_opp, V_double_take_opp,
                        opp_persp_at_next, match_state.jacoby,
                    )
                    if opp_doubles:
                        # We take iff take value >= pass value (-1).
                        takes = V_double_take_opp <= 1.0
                        val = 2.0 * U_B if takes else -1.0
                    else:
                        val = U_A
                else:
                    val = U_A

            if val > best_val:
                best_val = val
                best_idx = m_idx

        assert best_val > float("-inf"), \
            "no candidate was evaluated (empty plays?)"
        # +-3 (backgammon) times 2 (opp's simulated double).
        assert -6.0 - 1e-6 <= best_val <= 6.0 + 1e-6, \
            f"best_val out of [-6, 6] bounds: {best_val}"
        chosen = plays[best_idx][1]
        return chosen, best_val

    def _expand_cubeful_level1(
        self, state, opp_persp, opp_cube_action, gammons_count,
        track_next_states,
    ):
        """Level-1 dice expansion shared by the cubeful 1-ply and 2-ply
        value methods: per dice, the mover's per-unit equity of every
        successor (terminals from game_result, or +1 when gammons don't
        count; the rest -V(opp) from one batched forward with the
        opponent's cube perspective).

        Returns `(dice_results, dice_nonterm_indices, dice_next_states)`,
        each indexed by dice outcome. `dice_next_states` is None unless
        `track_next_states` (the 2-ply path deepens them).
        """
        base = self.encoder._base
        n_feat = self.encoder.num_features
        n_base = base.num_features
        opp_col = int(opp_persp)

        dice_results: List[Optional[List[float]]] = [None] * len(_DICE_OUTCOMES)
        dice_nonterm_indices: List[Optional[np.ndarray]] = [None] * len(_DICE_OUTCOMES)
        dice_next_states: Optional[List[Optional[list]]] = (
            [None] * len(_DICE_OUTCOMES) if track_next_states else None
        )
        enc_chunks: List[np.ndarray] = []
        chunk_dice_idx: List[int] = []

        for dice_idx, (d1, d2) in enumerate(_DICE_OUTCOMES):
            if self._c_base_available:
                base_feats, next_states_view = (
                    _bg_fast.get_legal_plays_encoded(state, (d1, d2))
                )
                n = len(next_states_view)
                if n == 0:
                    # Forced pass: one-row chunk of the opp-on-roll position.
                    opp_view = switch_turn(state)
                    base_enc = _bg_fast.encode_state(opp_view)
                    enc = self.encoder.encode_with_base(base_enc, opp_persp, opp_cube_action)
                    dice_results[dice_idx] = [0.0]
                    dice_nonterm_indices[dice_idx] = np.array([0], dtype=np.int64)
                    if track_next_states:
                        dice_next_states[dice_idx] = [opp_view]
                    enc_chunks.append(enc.reshape(1, -1))
                    chunk_dice_idx.append(dice_idx)
                    continue

                # feature[OPP_OFF_INDEX] >= 1: the mover just won.
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
                if track_next_states:
                    # Materialise: the C path returns a view.
                    dice_next_states[dice_idx] = [
                        next_states_view[int(i)] for i in non_term_idx
                    ]
                if len(non_term_idx) > 0:
                    # Base feats + cube columns.
                    chunk = np.zeros(
                        (len(non_term_idx), n_feat),
                        dtype=np.float32,
                    )
                    chunk[:, :n_base] = base_feats[non_term_idx]
                    chunk[:, n_base + opp_col] = 1.0
                    if opp_cube_action:
                        chunk[:, n_base + 3] = 1.0
                    enc_chunks.append(chunk)
                    chunk_dice_idx.append(dice_idx)
            else:
                plays = get_legal_plays(state, (d1, d2))
                if not plays:
                    opp_view = switch_turn(state)
                    base_enc = base.encode(opp_view)
                    enc = self.encoder.encode_with_base(base_enc, opp_persp, opp_cube_action)
                    dice_results[dice_idx] = [0.0]
                    dice_nonterm_indices[dice_idx] = np.array([0], dtype=np.int64)
                    if track_next_states:
                        dice_next_states[dice_idx] = [opp_view]
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
                if track_next_states:
                    dice_next_states[dice_idx] = [next_states[i] for i in non_term_list]
                if non_term_list:
                    chunk = np.stack([
                        self.encoder.encode_with_base(
                            base.encode(next_states[i]), opp_persp,
                            opp_cube_action,
                        )
                        for i in non_term_list
                    ])
                    dice_nonterm_indices[dice_idx] = np.asarray(
                        non_term_list, dtype=np.int64,
                    )
                    enc_chunks.append(chunk)
                    chunk_dice_idx.append(dice_idx)

        # One batched forward over every encoding.
        if enc_chunks:
            batch = np.concatenate(enc_chunks, axis=0)
            t = torch.from_numpy(batch)
            t = self._to_device(t)
            with torch.no_grad():
                if self.bf16_inference and self._bf16_net is not None:
                    opp_values = (
                        self._bf16_net(t.to(torch.bfloat16))
                        .to(torch.float32).cpu().numpy()
                    )
                else:
                    opp_values = self.network(t).cpu().numpy()
            offset = 0
            for chunk, di in zip(enc_chunks, chunk_dice_idx):
                n_rows = chunk.shape[0]
                idxs = dice_nonterm_indices[di]
                results_list = dice_results[di]
                for j in range(n_rows):
                    results_list[int(idxs[j])] = -float(opp_values[offset + j])
                offset += n_rows

        return dice_results, dice_nonterm_indices, dice_next_states

    def value_oneply_checker_cubeful(
        self, state: BoardState, match_state,
    ) -> float:
        """1-ply per-unit equity at `state` for the player on roll,
        under the cube in `match_state`: per dice, the mover's best move
        (max of -V(opp) for non-terminals, game_result for terminals,
        Jacoby applied), in one batched forward. Money game only.
        """
        assert self.is_cubeful, \
            "value_oneply_checker_cubeful requires a CubefulEncoder agent"
        assert not state.is_game_over(), \
            "value_oneply_checker_cubeful called on a terminal state"

        from modes import CubeOwner, cube_perspective

        player = state.turn
        opponent = 1 - player
        opp_persp = cube_perspective(match_state.cube_owner, opponent)
        opp_cube_action = match_state.can_offer(opponent)
        base = self.encoder._base
        n_feat = self.encoder.num_features
        n_base = base.num_features

        # Jacoby: gammons count iff the cube has been turned.
        gammons_count = (
            not match_state.jacoby
            or match_state.cube_owner != CubeOwner.CENTERED
        )

        # Fast path: one C 21-dice expansion + vectorised per-dice max.
        # Falls through when the expansion is unavailable or overflows.
        if self._c_base_available:
            exp = _bg_fast.expand_21(state)
            if exp is not None:
                base_feats, counts, gr, n_rows = exp
                opp_col = int(opp_persp)
                if n_rows:
                    feat = np.zeros((n_rows, n_feat), dtype=np.float32)
                    feat[:, :n_base] = base_feats
                    feat[:, n_base + opp_col] = 1.0
                    if opp_cube_action:
                        feat[:, n_base + 3] = 1.0
                    V = self._forward_inference(
                        self._to_device(torch.from_numpy(feat))
                    ).cpu().numpy().ravel()
                else:
                    V = np.empty(0, dtype=np.float32)
                nonterm = gr == 0
                # Mover equity per successor: -V for non-terminals,
                # game_result (or 1) for terminals; max per dice.
                vmover = gr.astype(np.float64)
                if not gammons_count:
                    vmover[~nonterm] = 1.0
                vmover[nonterm] = -V
                seg = np.empty(counts.shape[0], dtype=np.int64)
                seg[0] = 0
                np.cumsum(counts[:-1], out=seg[1:])
                dice_max = np.maximum.reduceat(vmover, seg)
                return float(_ONEPLY_PROBS @ dice_max)

        dice_results, _, _ = self._expand_cubeful_level1(
            state, opp_persp, opp_cube_action, gammons_count,
            track_next_states=False,
        )

        probs = _ONEPLY_PROBS.tolist()
        oneply_sum = 0.0
        for di, prob in enumerate(probs):
            equities = dice_results[di]
            if equities:
                oneply_sum += prob * max(equities)

        return oneply_sum

    def _choose_checker_twoply_cubeful(
        self, state: BoardState, plays, match_state, k: int = 1,
    ):
        """2-ply cubeful chequer selection: each candidate is scored by
        -value_twoply_checker_cubeful_filtered(next_state) (which
        includes opp's cube decision at level 2). Pick the max.
        """
        from modes import CubeOwner

        if not plays:
            return None

        gammons_count = (
            not match_state.jacoby
            or match_state.cube_owner != CubeOwner.CENTERED
        )

        best_value = float("-inf")
        best_idx = 0
        for m_idx, (_, next_state) in enumerate(plays):
            if next_state.is_game_over():
                mover_eq = (
                    float(next_state.game_result()) if gammons_count else 1.0
                )
            else:
                opp_value = self.value_twoply_checker_cubeful_filtered(
                    next_state, match_state, k=k,
                )
                mover_eq = -opp_value

            if mover_eq > best_value:
                best_value = mover_eq
                best_idx = m_idx

        chosen = plays[best_idx][1]
        return chosen, best_value

    def value_twoply_checker_cubeful_filtered(
        self, state: BoardState, match_state, k: int = 3,
    ) -> float:
        """2-ply per-unit equity at `state` with a top-K move
        filter. Per dice: rank the legal moves by 0-ply value (one
        batched forward), deepen the top-K non-terminal ones to opp's
        optimal 1-ply response (chequer play plus cube decision), and
        take the max over terminals and deepened candidates. Money game
        only. K = all candidates is exact 2-ply.
        """
        assert self.is_cubeful, \
            "value_twoply_checker_cubeful_filtered requires a CubefulEncoder agent"
        assert not state.is_game_over(), \
            "value_twoply_checker_cubeful_filtered called on a terminal state"
        assert k >= 1, f"k must be >= 1, got {k}"

        from modes import CubeOwner, cube_perspective

        player = state.turn
        opponent = 1 - player
        opp_persp = cube_perspective(match_state.cube_owner, opponent)
        opp_cube_action = match_state.can_offer(opponent)

        gammons_count = (
            not match_state.jacoby
            or match_state.cube_owner != CubeOwner.CENTERED
        )

        dice_results, dice_nonterm_indices, dice_next_states = (
            self._expand_cubeful_level1(
                state, opp_persp, opp_cube_action, gammons_count,
                track_next_states=True,
            )
        )
        probs = _ONEPLY_PROBS.tolist()

        # Level 2: per dice, deepen the top-K non-terminal candidates to
        # opp's 1-ply response. Best = max over terminals and deepened.
        twoply_sum = 0.0
        for di, prob in enumerate(probs):
            equities = dice_results[di]
            if not equities:
                continue
            non_term_idx = dice_nonterm_indices[di]
            non_term_states = dice_next_states[di]

            if non_term_idx is None or len(non_term_idx) == 0:
                # All terminal: same as 1-ply.
                twoply_sum += prob * max(equities)
                continue

            non_term_idx_set = set(int(i) for i in non_term_idx)
            terminal_values = [
                v for i, v in enumerate(equities) if i not in non_term_idx_set
            ]

            ranked = sorted(
                range(len(non_term_idx)),
                key=lambda j: -equities[int(non_term_idx[j])],  # descending
            )
            top_k = ranked[: min(k, len(ranked))]

            # Opp's cube decision MUST be enumerated here:
            # value_oneply_checker_cubeful scores chequer play only.
            level2_values = []
            for j in top_k:
                opp_next_state = non_term_states[j]
                opp_at_next = opp_next_state.turn  # = opponent

                v_no_double = self.value_oneply_checker_cubeful(
                    opp_next_state, match_state,
                )

                if match_state.can_offer(opp_at_next):
                    after_take_ms = match_state.after_take(opp_at_next)
                    v_after_take = self.value_oneply_checker_cubeful(
                        opp_next_state, after_take_ms,
                    )
                    # Per old cube unit: take gives opp 2*v_after_take,
                    # pass gives opp +1. I choose the min.
                    v_double_response = min(2.0 * v_after_take, 1.0)
                    v_opp_optimal = max(v_no_double, v_double_response)
                else:
                    v_opp_optimal = v_no_double

                level2_values.append(-v_opp_optimal)

            best = max(terminal_values + level2_values) if (
                terminal_values or level2_values
            ) else 0.0
            twoply_sum += prob * best

        return twoply_sum

    # ── online TD update (no loss/optimizer machinery) ────────────────

    def td_update(self, state: BoardState, target: float, lr: float) -> float:
        """Online TD(0) update for one state:

            v = V(state); v.backward()
            for p in params: p += lr * (target - v) * dV/dp

        Returns the squared TD error.
        """
        return self.td_update_encoded(self._encode(state), target, lr)

    def td_update_encoded(
        self, features: np.ndarray, target: float, lr: float,
    ) -> float:
        """td_update on pre-encoded features (the cubeful online trainer
        adds cube columns that `_encode` doesn't)."""
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
