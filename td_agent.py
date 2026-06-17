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

# Per-dice probabilities (doubles 1/36, non-doubles 2/36); aligned with
# _DICE_OUTCOMES order. Used by the expand_21 fast path's vectorized reduction.
_ONEPLY_PROBS = np.array(
    [(1.0 / 36.0) if d1 == d2 else (2.0 / 36.0) for d1, d2 in _DICE_OUTCOMES],
    dtype=np.float64,
)


def cpu_supports_fast_bf16() -> bool:
    """True if the CPU has bf16 matmul ISA support (AVX512-BF16 or
    AMX-BF16). Without it torch emulates bf16 in software — measured
    ~20x SLOWER than fp32 (e.g. on Zen 2) — so bf16 inference must
    not be enabled on such hosts.
    """
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
        return "avx512_bf16" in flags or "amx_bf16" in flags
    except OSError:
        return False


_BF16_UNSUPPORTED_WARNED = False

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
        twoply_k: int = 0,
        boltzmann_temp: float = 0.0,
        bf16_inference: bool = False,
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
            twoply_k: when > 0, use 2-ply chequer move selection with
                top-K 1-ply filter. Works for both cubeful and
                cubeless (DMP) modes. "2-ply" matches gnubg's
                convention: two dice+move expansions before static
                eval. The k value is the inner top-K 1-ply filter at
                the depth-2 expansion; root candidates are always
                fully expanded. Mutually exclusive with oneply for
                move selection — twoply_k > 0 takes precedence over
                oneply on chequer choice.
            boltzmann_temp: temperature for Boltzmann (softmax)
                exploration in checker play. 0.0 (default) = greedy
                argmin. Higher values = more exploration.
            bf16_inference: when True, 1-ply target computation uses
                a bf16 copy of the network for the forward pass
                (~1.9× speedup on the MLP matmul with ~1e-3 numerical
                drift). Master training still uses fp32 weights; this
                is intended for collection workers only.
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
        self.oneply = oneply
        self.twoply_k = int(twoply_k)
        self.boltzmann_temp = boltzmann_temp
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

        # Optional bf16 inference copy for 1-ply target computation.
        # Built once from the current fp32 weights; re-build via
        # `refresh_bf16_inference()` after any state_dict reload.
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

        # Transposition table for 2-ply cubeless search. Memoizes the
        # 1-ply value of a position keyed by its canonical board key.
        # The network is frozen during play, so a position's 1-ply
        # value is a pure function of the board — caching it is EXACT
        # (no fidelity loss) and skips a full batched forward on every
        # hit. Only allocated when 2-ply move selection is active.
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
        """Refresh the bf16 copy after fp32 weights change.
        No-op when `bf16_inference` is False.
        """
        if self.bf16_inference:
            self._build_bf16_net()

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
        if self.twoply_k > 0 and not self.is_cubeful:
            return self._choose_checker_twoply_cubeless(
                state, dice, actions=actions,
            )
        if self.oneply and not self.is_cubeful:
            return self._choose_checker_oneply_cubeless(
                state, dice, actions=actions, with_target=with_target,
            )
        if actions is None and self.is_cubeful:
            # The fused move-gen+encode path emits base features only;
            # cubeful agents need the cube columns appended, so take the
            # explicit-actions path below.
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
            # Checker decision in a cube-less context (e.g. eval scripts):
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
        """Select an index using Boltzmann (softmax) sampling over
        opponent-view values. Lower values are better for us, so we
        negate before softmax. Returns argmin when temp=0."""
        if self.boltzmann_temp <= 0.0 or len(values) == 1:
            return int(torch.argmin(values).item())
        # Negate: lower opp value = better for us = higher probability
        logits = -values.squeeze() / self.boltzmann_temp
        probs = torch.softmax(logits, dim=0)
        return int(torch.multinomial(probs, 1).item())

    def _terminal_opp_value(self, next_state: BoardState) -> float:
        """Value a terminal next_state from the opponent's view (i.e.
        AFTER the mover's winning move — the engine has switched turn,
        so next_state.turn == opponent, the LOSER). Used to override
        the network's arbitrary output on terminal encodings."""
        if self.output_mode == "probability":
            return 0.0
        return -float(next_state.game_result())

    def _apply_terminal_mask(self, values, features_tensor, next_states):
        """Overwrite values[i] with the true opp-view terminal value for
        any i whose encoding indicates the game is over (OPP_OFF feature
        ≥ 1). The network's output on terminal encodings is arbitrary
        — leaving it unmasked can cause argmin to skip a winning move.
        """
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

        oneply_sum = 0.0
        for idx, (d1, d2) in enumerate(_DICE_OUTCOMES):
            prob = float(_ONEPLY_PROBS[idx])
            features, next_states = self._get_legal_plays_encoded(state, (d1, d2))

            if len(next_states) == 0:
                # Forced pass: opponent gets the dice on the same
                # position. Evaluate from opponent's view.
                v_opp = self.evaluate(switch_turn(state))
                oneply_sum += prob * self._flip_value(v_opp)
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
                    values[i] = self._terminal_opp_value(next_states[i])

            best_for_next = torch.min(values).item()
            oneply_sum += prob * self._flip_value(best_for_next)

        return oneply_sum

    def _choose_checker_oneply_cubeless(
        self, state, dice, actions=None, with_target=False,
    ):
        """Cubeless 1-ply move selection, batched.

        For each candidate move (a, s_next):
          - if s_next is terminal (we won): assign our terminal value
          - else: enumerate opp's 21 dice, generate opp-successors at
            s_next; encode (us-view, since engine swaps turn after opp
            moves) and accumulate into one batched forward.
        After the single forward pass, per candidate compute
            E_dice[ min_over_opp_leaves( our-view value ) ]
        and pick the candidate with the highest expectation.
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
            # s is terminal AFTER a move. Engine has switched turn, so
            # s.turn = the LOSER. game_result() returns 1/2/3 for the
            # winner. Caller knows whose move produced the terminal —
            # this helper just returns a magnitude; caller flips sign.
            if is_equity:
                return float(s.game_result())
            return 1.0  # probability mode: P(mover-just-won) = 1

        all_encs: List[np.ndarray] = []
        total_rows = 0
        cand_terminal: List[Optional[float]] = [None] * len(plays)
        # Per candidate: dict[dice_idx] -> list of (row_idx_or_None,
        # terminal_or_None). Min over our-view values picks opp's worst-
        # for-us reply.
        cand_dice: List[Optional[dict]] = [None] * len(plays)

        for m_idx, (_, next_state) in enumerate(plays):
            if next_state.is_game_over():
                # We just won. Our terminal value = +game_result (equity)
                # or +1 (probability).
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
                            # Opp's move ended the game → opp won. Our
                            # terminal value (us-view) is negative.
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
                        # Encoded leaves are FROM US-VIEW (engine swapped
                        # turn after opp's move), so network output =
                        # our value directly.
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
                    # Opp can't move → forced pass; dice goes to us at
                    # switch_turn(next_state). Encode from our view —
                    # network output = our value of that position. Add
                    # as a single bucket entry so min() over it returns
                    # itself.
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
                    # Opp picks min over their leaves (worst-for-us).
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
            # Target is OUR 1-ply value at `state` — same as best_val,
            # which is OUR view of the chosen action's expected value.
            return chosen_play, chosen_state, best_val
        return chosen_play, chosen_state

    def _forward_inference(self, t: torch.Tensor) -> torch.Tensor:
        """Run a forward in bf16 if a bf16 inference net exists,
        else fp32. Returns an fp32 tensor either way so callers
        don't have to think about dtype downstream. ~3-4x faster
        than fp32 at large batches with ~1e-3 drift, which doesn't
        change argmin selections in practice. Always wrapped in
        no_grad — only meant for value queries.
        """
        with torch.no_grad():
            if self.bf16_inference and self._bf16_net is not None:
                return self._bf16_net(t.to(torch.bfloat16)).to(torch.float32)
            return self.network(t)

    def _value_oneply_checker_batched(self, state: BoardState) -> float:
        """Same semantics as value_oneply_checker but issues ONE
        batched forward across all 21 inner dice, instead of 21
        per-dice forwards. Used by value_twoply_checker to cut
        dispatch overhead. Uses bf16 when available.
        """
        assert not state.is_game_over(), \
            "_value_oneply_checker_batched called on a terminal state"

        # Transposition lookup: exact memo of this position's 1-ply
        # value (network is frozen during play). On a hit we skip the
        # entire enumeration + batched forward below.
        tt = self._tt_oneply
        if tt is not None:
            tt_key = state._key()
            hit = tt.get(tt_key)
            if hit is not None:
                return hit

        # Phase 1: enumerate per-dice; collect features for non-pass
        # outcomes into one big stack. Forced passes are resolved
        # against the 0-ply opp value (we already burned one ply by
        # being here; the swap_turn eval is the static estimate).
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
                # Forced pass: opp's static value at swapped state.
                v_opp = self.evaluate(switch_turn(state))
                dice_slices.append((None, None, prob, self._flip_value(v_opp)))
                continue

            n = len(next_states_view)
            # Mark terminals — need the materialized state to read
            # game_result(), so we materialize lazily only for those.
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

        # If every dice was a forced pass (extremely rare but
        # possible), bail out without a forward.
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
        """2-ply estimate of V(state) for a cubeless CHECKER decision point.

        "2-ply" here matches gnubg's convention: TWO dice+move
        expansions before static evaluation. From mover's view at
        `state`:
            E_{D1}[ max_{m1} ( flip( E_{D2}[ max_{m2}( flip( V(s2) ) ) ] ) ) ]
        where m1 is mover's choice after dice D1, and m2 is opp's
        choice after dice D2 at state s1 = result(state, D1, m1).

        Inner expansion uses a top-K 1-ply filter (matching
        value_twoply_checker_cubeful_filtered): rank the mover-side
        candidates at depth 1 by 0-ply static value, then deepen
        only the top-K to a 1-ply call. Terminal candidates are
        scored from game_result() and never deepened.

        Cost per call ≈ 21 × (1 batched 0-ply + K calls to
        value_oneply_checker). K=1 is fastest and vulnerable to
        0-ply ranking errors; K=3 is a reasonable safety margin.
        """
        assert not state.is_game_over(), \
            "value_twoply_checker called on a terminal state"
        assert k >= 1, f"k must be >= 1, got {k}"

        # Phase 1: gather all outer-dice candidates into one big
        # tensor, do ONE batched forward for 0-ply ranking. Saves 21
        # per-dice dispatch overheads at the cost of a single
        # bigger matmul (which is per-element faster, especially
        # with bf16).
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
                # Forced pass: opp gets the dice on the same position.
                # Resolve later with a batched 1-ply call (still bf16).
                # Defer to a post-pass to avoid breaking the big-batch
                # flow — pass a sentinel and resolve after the forward.
                per_dice.append((None, None, prob, None, []))  # sentinel "forced"
                continue

            # Materialize next_states immediately — the shared C
            # buffer is invalidated by any subsequent _bg_fast call.
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

        # Resolve forced-pass dice with a batched 1-ply call AFTER
        # the main forward (so the big-batch flow isn't broken).
        # Reuse the same swap_turn(state) result across all
        # forced-pass dice — the underlying value is identical
        # for each.
        forced_resolved: Optional[float] = None
        # Big-batch forward for the 0-ply rank phase.
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

        # Phase 2: per dice, compute best opp-view value (mover wants
        # min); deepen the top-K non-terminal candidates to 1-ply.
        twoply_sum = 0.0
        for start, end, prob, forced, next_states in per_dice:
            if forced is None and start is None:
                # Sentinel forced pass.
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
            # Recompute terminal mask cheaply from the original
            # features array — but we don't have direct access here
            # without indexing back. Use a global per-row flag: any
            # entry in term_overrides for this slice is terminal.
            # Build a fast lookup once per dice.
            #
            # In practice terminal rows are rare; a quick scan over
            # term_overrides is fine.
            best_opp_view = float("inf")

            # Build terminal mask for this slice if needed.
            slice_terms = [
                (idx - start, v) for idx, v in term_overrides
                if start <= idx < end
            ]
            terminal_local = {i for i, _ in slice_terms}

            # Min over terminals (already overwritten in all_values).
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
        """Cubeless 2-ply chequer move selection.

        For each root candidate move m → next_state:
          - if next_state is terminal: mover already won, use the
            terminal mover-value;
          - else: compute opp's 2-ply value at next_state via
            value_twoply_checker. Flip to mover's frame.
        Pick m maximizing mover's value.

        `twoply_k` (from the TDAgent ctor) is forwarded as the inner
        top-K filter at the depth-2 expansion (NOT a filter on root
        candidates — root is always fully expanded).

        Returns `(Play_or_None, next_state)` matching the contract
        of `choose_checker_action` (no with_target — 2-ply skips
        target collection).
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
            # s is terminal after our move. game_result() returns
            # 1/2/3 for single/gammon/backgammon.
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
                # Opp on-roll at next_state. Compute opp's 2-ply
                # value (= opp dice + opp move + my dice + my move +
                # static), then flip to mover's frame.
                opp_value = self.value_twoply_checker(next_state, k=k)
                mover_eq = self._flip_value(opp_value)

            if mover_eq > best_value:
                best_value = mover_eq
                best_idx = m_idx

        chosen = plays[best_idx]
        return chosen[0], chosen[1]

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
        is_cube_action: bool = False,
    ) -> np.ndarray:
        """Encode `state` with the given cube perspective. Uses the C
        fast path for the base features when available; the cube
        feature append is delegated to CubefulEncoder.
        """
        if self._c_base_available:
            base = _bg_fast.encode_state(state)
        else:
            base = self.encoder._base.encode(state)
        return self.encoder.encode_with_base(base, cube_state, is_cube_action)

    def evaluate_cubeful(
        self, state: BoardState, cube_state: CubePerspective,
        is_cube_action: bool,
    ) -> float:
        """Per-unit equity (cube_value=1) at `state` with this cube
        perspective. Multiply by actual cube_value externally."""
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
        """Money-game double decision given pre-computed equities.
        Shared by 0-ply and 1-ply offer_double paths.

        v_no_double:   checker-phase equity if we play on (is_cube_action=0).
        v_double_take: 2 × V(theirs, is_cube_action=0) — equity if we double
                       and opponent takes (stake doubles).
        """
        opponent_will_pass = (v_double_take > 1.0)
        if opponent_will_pass:
            if jacoby and pre_persp == CubePerspective.CENTERED:
                return True  # Jacoby: cash, gammons don't count
            return 1.0 >= v_no_double  # too good check
        return v_double_take > v_no_double

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

        if self.twoply_k > 0:
            # 2-ply cube: use the 2-ply chequer expectimax value with the
            # current vs theirs match_state. Matches the depth of the
            # chequer move selector and removes the 0-ply-cube/2-ply-chequer
            # depth-asymmetry vs gnubg-nn (which uses 2-ply for both).
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

        # Batch the two needed evaluations into one forward (batch=2)
        # — saves a kernel launch / Python dispatch per cube offer.
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
        """Money-game take/pass response. Take iff v_double_take ≤ 1
        (equivalently, the responder's continuation equity beats the
        −1 they'd get by passing).

        Matchplay agents override this for score-aware take points.
        """
        if not self.is_cubeful:
            return super().respond_to_double(state, match_state, hint=hint)
        # Read cached v_double_take at the matching resolution,
        # or recompute if not available.
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
        """1-ply cube decision: v_no_double and v_double_take from
        value_oneply_checker_cubeful, then standard money logic."""
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

        # 2-ply: needs the full Play objects, can't use the C fast path
        # below (which hands the 1-ply consumer (None, next_state)).
        if self.twoply_k > 0:
            plays = get_legal_plays(state, dice)
            if not plays:
                return None
            return self._choose_checker_twoply_cubeful(
                state, plays, match_state, k=self.twoply_k,
            )

        # Fast path: use the C move-gen + encode in one call, eagerly
        # materializing successor states. Saves the Python _generate_plays
        # / encode_state loop (~50% of cubeful self-play time per profile).
        # The 1-ply downstream consumer only reads `next_state` from each
        # (play_obj, next_state) tuple, so handing it (None, next_state)
        # is safe.
        if self._c_base_available:
            base_feats, _lazy = _bg_fast.get_legal_plays_encoded(state, dice)
            n_plays = len(_lazy)
            if n_plays == 0:
                return None
            # Materialize immediately — the shared C buffer is invalidated
            # by the next bg_fast call (incl. the opponent's turn or any
            # downstream 1-ply inner enumeration).
            next_states = [_lazy[i] for i in range(n_plays)]
            if self.oneply:
                plays = [(None, ns) for ns in next_states]
                return self._choose_checker_oneply_cubeful(
                    state, plays, match_state,
                )
            # 0-ply continues below with base_feats + next_states already set
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
        # Append cube one-hot (3) + is_cube_action bit (1)
        from encoding import CUBE_FEATURES
        extra = np.zeros((len(next_states), CUBE_FEATURES), dtype=np.float32)
        extra[:, int(opp_persp)] = 1.0
        if opp_cube_action:
            extra[:, 3] = 1.0
        features = np.concatenate([base_feats, extra], axis=1)

        t = torch.from_numpy(features)
        t = self._to_device(t)
        with torch.no_grad():
            values = self.network(t)
        # Terminal mask: for any next_state that ends the game (mover
        # just won), replace the network's arbitrary output with the
        # true opp-view value. Under Jacoby with a centered cube,
        # gammons don't count — collapse to -1.0.
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
        """1-ply cubeful move selection. For each candidate move,
        looks one ply ahead (opp's 21 dice × best reply per dice) and
        also simulates opp's cube decision at next_state when opp is
        eligible to offer.

        Per candidate we compute two 1-ply expectations:
          U_A = our per-unit equity if opp does NOT double
                (cube unchanged at after_opp)
          U_B = our per-unit equity if opp doubles and we take
                (cube transferred to us at after_opp; still per-unit
                at stake=1 — multiply by 2 for actual stake-2 equity)
        Opp's money cube decision is applied from opp's perspective
        (V_no_double_opp = -U_A, V_double_take_opp = -2·U_B). If opp
        doubles, we take iff 2·U_B > -1 (equiv: V_double_take_opp ≤ 1),
        realizing 2·U_B on take or -1 on pass. Otherwise we get U_A.
        """
        from modes import CubeOwner, cube_perspective

        mover = state.turn
        opponent = 1 - mover
        opp_can_offer = match_state.can_offer(opponent)
        opp_persp_at_next = cube_perspective(match_state.cube_owner, opponent)


        # Scenario A: opp doesn't double → cube unchanged at after_opp.
        our_persp_A = cube_perspective(match_state.cube_owner, mover)
        after_opp_is_cube_action_A = match_state.can_offer(mover)
        gammons_count_A = (
            not match_state.jacoby
            or match_state.cube_owner != CubeOwner.CENTERED
        )

        # Scenario B: opp doubles and we take → cube transfers to us, so
        # our_persp at after_opp is MINE and we can always offer there.
        # Cube is owned (not centered) → gammons always count.
        our_persp_B = CubePerspective.MINE

        base = self.encoder._base
        n_feat = self.encoder.num_features
        n_base = base.num_features
        opp_col_A = int(our_persp_A)
        opp_col_B = int(our_persp_B)

        # Accumulate ALL encodings across candidates × opp_dice × opp_moves
        # × {scenario A, scenario B if opp_can_offer} as 2D chunks; one
        # np.concatenate at the end gives the batched forward input.
        # `total_rows` tracks the row index resolved_A/B entries point at.
        all_encs: List[np.ndarray] = []
        total_rows = 0
        cand_terminal: List[Optional[float]] = [None] * len(plays)
        cand_dice_A: List[Optional[dict]] = [None] * len(plays)
        cand_dice_B: List[Optional[dict]] = [None] * len(plays)

        for m_idx, (_, next_state) in enumerate(plays):
            if next_state.is_game_over():
                # We won on this move — no opp cube decision possible.
                if gammons_count_A:
                    cand_terminal[m_idx] = float(next_state.game_result())
                else:
                    cand_terminal[m_idx] = 1.0
                continue

            # Opponent is now on roll at next_state.
            dice_data_A: dict = {}
            dice_data_B: Optional[dict] = {} if opp_can_offer else None

            for dice_idx, (d1, d2) in enumerate(_DICE_OUTCOMES):
                # Fused C call: move-gen + 196-feature encode in one shot.
                # Falls back to Python move-gen + per-state encode when
                # the C engine isn't usable.
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
                    # Terminal detection: vectorized when base_feats present.
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

                    # First pass: emit resolved_A/B entries in original
                    # opp-move order. For non-terminals we reserve the
                    # slot with a placeholder and fill it after we know
                    # its row index in the scenario-A / scenario-B chunk.
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
                                # gammons_count_B=True always
                                resolved_B.append((None, -gr))
                        else:
                            non_term_slots_A.append(len(resolved_A))
                            resolved_A.append(None)  # filled below
                            if opp_can_offer:
                                non_term_slots_B.append(len(resolved_B))
                                resolved_B.append(None)

                    n_nt = len(non_term_idx)
                    if n_nt > 0:
                        # Vectorized chunk for scenario A
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
                    # Forced pass — opp can't move, play returns to us.
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

        # ONE batched forward pass over scenario A + scenario B encodings.
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

        # Structural check: every non-terminal candidate must have a
        # full 21-bucket dice dict in scenario A (and B if opp_can_offer),
        # with at least one resolution per bucket.
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
                # U_A: our expected equity if opp doesn't double.
                U_A = 0.0
                for dice_idx, resolved in cand_dice_A[m_idx].items():
                    min_eq = min(
                        net_values[idx] if idx is not None else term
                        for idx, term in resolved
                    )
                    U_A += probs[dice_idx] * min_eq

                if opp_can_offer:
                    # U_B: our per-unit equity if opp doubles & we take.
                    U_B = 0.0
                    for dice_idx, resolved in cand_dice_B[m_idx].items():
                        min_eq = min(
                            net_values[idx] if idx is not None else term
                            for idx, term in resolved
                        )
                        U_B += probs[dice_idx] * min_eq

                    # Opp's money cube decision (from opp's perspective).
                    V_no_double_opp = -U_A
                    V_double_take_opp = -2.0 * U_B
                    opp_doubles = self._money_cube_decision(
                        V_no_double_opp, V_double_take_opp,
                        opp_persp_at_next, match_state.jacoby,
                    )
                    if opp_doubles:
                        # Our response: take iff V_double_take_opp <= 1
                        # (equiv: 2·U_B >= -1, our take value ≥ pass value)
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
        # Per-unit equity bound — backgammon (±3) at current stake,
        # but opponent cube simulation can double the stake (×2).
        assert -6.0 - 1e-6 <= best_val <= 6.0 + 1e-6, \
            f"best_val out of [-6, 6] bounds: {best_val}"
        chosen = plays[best_idx][1]
        return chosen, best_val

    def _expand_cubeful_level1(
        self, state, opp_persp, opp_cube_action, gammons_count,
        track_next_states,
    ):
        """Shared level-1 dice expansion for the cubeful 1-ply and 2-ply
        value methods.

        Enumerates all 21 dice outcomes; for each builds a per-dice list
        of mover per-unit equities (terminal successors scored by
        game_result, or +1 when gammons don't count), encodes every
        non-terminal/forced-pass successor with the opponent cube
        perspective, runs ONE batched forward across all of them, and
        fills in the non-terminal equities as -V(opp).

        Returns `(dice_results, dice_nonterm_indices, dice_next_states)`,
        each a list indexed by dice outcome. `dice_next_states` is
        populated only when `track_next_states` is True (the 2-ply path
        needs the successors for top-K deepening; the 1-ply path does
        not), otherwise it is None.
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
                    enc = self.encoder.encode_with_base(base_enc, opp_persp, opp_cube_action)
                    dice_results[dice_idx] = [0.0]
                    dice_nonterm_indices[dice_idx] = np.array([0], dtype=np.int64)
                    if track_next_states:
                        dice_next_states[dice_idx] = [opp_view]
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
                if track_next_states:
                    # Materialise the non-terminal next_states (the C fast
                    # path returns a view; copy refs so they survive past
                    # the function-local scope).
                    dice_next_states[dice_idx] = [
                        next_states_view[int(i)] for i in non_term_idx
                    ]
                if len(non_term_idx) > 0:
                    # Build the (n_nonterm, 199) chunk: base feats +
                    # cube one-hot column, vectorised.
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

        # ONE batched forward pass for every non-terminal / pass encoding.
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
        opp_cube_action = match_state.can_offer(opponent)
        base = self.encoder._base
        n_feat = self.encoder.num_features
        n_base = base.num_features

        # Jacoby: gammons/bgs count iff the cube has been turned.
        gammons_count = (
            not match_state.jacoby
            or match_state.cube_owner != CubeOwner.CENTERED
        )

        # Fast path: one C 21-dice expansion of `state` + vectorized
        # per-dice max reduction. Single cube scenario (opp on roll
        # after the mover's move). Falls through to the per-dice path
        # below if the C expansion is unavailable / overflows.
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
                # Mover's per-unit equity per reply: non-terminal -> -V
                # (flip opp view); terminal -> +game_result (or +1 when
                # gammons don't count). Mover picks the max per dice.
                vmover = gr.astype(np.float64)
                if not gammons_count:
                    vmover[~nonterm] = 1.0
                vmover[nonterm] = -V
                seg = np.empty(counts.shape[0], dtype=np.int64)
                seg[0] = 0
                np.cumsum(counts[:-1], out=seg[1:])
                dice_max = np.maximum.reduceat(vmover, seg)
                return float(_ONEPLY_PROBS @ dice_max)

        # Per-dice 0-ply expansion + single batched forward (shared with
        # the 2-ply method). The 1-ply path doesn't need the successors,
        # so track_next_states=False.
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
        """2-ply K=k cubeful chequer move selection.

        For each candidate move m, compute opp's value at the resulting
        state using value_twoply_checker_cubeful_filtered (which itself
        includes opp's cube decision at level 2 per the post-fix
        implementation). Pick the move maximizing mover equity.

        Simpler than the 1-ply chequer selector (no per-candidate batching
        across opp dice) — each candidate just calls the 2-ply value
        function, which does its own internal batching. Cost per call is
        higher: roughly k+1 calls to value_oneply_checker_cubeful inside.

        With k=1 on prod 4L on the test machine: ~42 ms per evaluation
        of one candidate's resulting state. Self-play games with this
        move selector are ~24-30x slower than 0-ply self-play.
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
                # Mover wins this move outright.
                mover_eq = (
                    float(next_state.game_result()) if gammons_count else 1.0
                )
            else:
                # Opp is on roll at next_state. The 2-ply value function
                # returns opp's per-unit equity (now including opp's
                # cube decision in the level-2 expansion).
                import os as _os
                if _os.environ.get("TWOPLY_IMMEDIATE_CUBE", "0") == "1":
                    opp_value = self.value_twoply_with_immediate_cube(
                        next_state, match_state, k=k,
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

    def value_twoply_with_immediate_cube(
        self, state: BoardState, match_state, k: int = 3,
    ) -> float:
        """value_twoply_checker_cubeful_filtered + on-roll player's own
        immediate cube decision at this state.

        The base function enumerates the OPPONENT's cube decision at
        level-2 (after on-roll player's move). It does NOT enumerate
        the on-roll player's cube decision RIGHT NOW (before rolling).
        This wrapper adds that level-1 enumeration via the standard
        max(no_double, min(2·v_after_take, 1)) analytical formula.

        Cost: two calls to the base function (vs one). Roughly 2x slower
        when the on-roll player is cube-eligible; same cost when not.
        """
        v_no_double = self.value_twoply_checker_cubeful_filtered(
            state, match_state, k=k,
        )
        if not match_state.can_offer(state.turn):
            return v_no_double
        # On-roll doubles → opp takes/passes. On take: cube=2, opp owns.
        # Recompute 2-ply value at same state under new cube state.
        after_take_ms = match_state.after_take(state.turn)
        v_after_take = self.value_twoply_checker_cubeful_filtered(
            state, after_take_ms, k=k,
        )
        # Opp picks min for on-roll: payoff after doubling = min(2·v, 1)
        v_double_response = min(2.0 * v_after_take, 1.0)
        return max(v_no_double, v_double_response)

    def value_twoply_checker_cubeful_filtered(
        self, state: BoardState, match_state, k: int = 3,
    ) -> float:
        """2-ply per-unit equity at `state` with 1-ply top-K move filter.

        For each of the 21 dice outcomes:
          1. Enumerate all legal moves (level-1 candidates)
          2. Compute 0-ply value for each candidate via batched forward
             (terminals scored by game_result + Jacoby; non-terminals
             by -V(opp))
          3. Filter to the top-K non-terminal candidates by 0-ply
             mover equity (= the 1-ply move filter)
          4. For each top-K candidate, compute 1-ply value of its
             successor (= opponent's optimal 1-ply response)
          5. Mover's best per dice = max over (terminal candidates
             level-1 values, top-K candidates level-2 values)
        Then weighted sum across dice.

        Cost: each state evaluation does 1 level-1 batched forward
        (cheap, same as 1-ply) + K calls to value_oneply_checker_cubeful
        per non-terminal-rich dice outcome. Roughly K+1 = 4x slower
        than 1-ply at K=3.

        K=1: pick best 0-ply move, evaluate at 1-ply. Cheap but
        vulnerable to 0-ply ranking errors.
        K = max_candidates: exact 2-ply (no filter); much slower.
        K=3 is a reasonable safety margin — the true best is almost
        always in the 0-ply top-3 for a trained network.

        Intended for cubeful money training only. Matchplay needs a
        score-aware variant.
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

        # Per-dice 0-ply expansion + single batched forward (shared with
        # the 1-ply method). track_next_states=True keeps the non-terminal
        # successors for the level-2 top-K deepening below.
        dice_results, dice_nonterm_indices, dice_next_states = (
            self._expand_cubeful_level1(
                state, opp_persp, opp_cube_action, gammons_count,
                track_next_states=True,
            )
        )
        probs = _ONEPLY_PROBS.tolist()

        # Level-2: for each dice, take the top-K non-terminal candidates
        # by level-1 mover equity, compute their 1-ply opponent response,
        # then pick the best across {terminal candidates, top-K level-2}.
        twoply_sum = 0.0
        for di, prob in enumerate(probs):
            equities = dice_results[di]
            if not equities:
                continue
            non_term_idx = dice_nonterm_indices[di]
            non_term_states = dice_next_states[di]

            # Identify terminal-vs-non-terminal candidates so we know
            # which ones can be re-evaluated at level-2.
            if non_term_idx is None or len(non_term_idx) == 0:
                # All terminal — same answer as 1-ply (no deeper search).
                twoply_sum += prob * max(equities)
                continue

            non_term_idx_set = set(int(i) for i in non_term_idx)
            terminal_values = [
                v for i, v in enumerate(equities) if i not in non_term_idx_set
            ]

            # Sort non-terminal candidates by 0-ply mover equity, take top-K.
            ranked = sorted(
                range(len(non_term_idx)),
                key=lambda j: -equities[int(non_term_idx[j])],  # descending
            )
            top_k = ranked[: min(k, len(ranked))]

            # Level-2 evaluation: for each top-K candidate, the opponent
            # is on roll at `non_term_states[j]`. Simulate opp's optimal
            # cube decision (if cube-eligible) plus their 1-ply chequer
            # value; mover equity = -opp_optimal_value.
            #
            # Critical: value_oneply_checker_cubeful alone evaluates ONLY
            # the chequer play — it does NOT enumerate opp's cube decision
            # at the intermediate node. The 1-ply target avoids this by
            # using V_theta(next, cube_action=True), which is trained to
            # include opp's cube decision. At level 2 we must enumerate
            # the cube decision explicitly, otherwise we systematically
            # undervalue opp (and overvalue our move), biasing targets
            # upward. Per-unit equities throughout (cube_value=1
            # normalization).
            level2_values = []
            for j in top_k:
                opp_next_state = non_term_states[j]
                opp_at_next = opp_next_state.turn  # = opponent

                # Opp's no-double value (chequer-only 1-ply lookahead).
                v_no_double = self.value_oneply_checker_cubeful(
                    opp_next_state, match_state,
                )

                if match_state.can_offer(opp_at_next):
                    # Opp may double; I respond optimally.
                    after_take_ms = match_state.after_take(opp_at_next)
                    v_after_take = self.value_oneply_checker_cubeful(
                        opp_next_state, after_take_ms,
                    )
                    # Per old cube unit: take -> 2 * v_after_take to opp;
                    # pass -> +1 to opp. I pick min for opp.
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
