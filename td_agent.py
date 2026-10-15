"""
td_agent.py -- TDAgent: the scalar (TDNetwork) playing/learning agent.

The agent encodes positions, runs the net, and picks chequer moves and
cube actions at 0- or 1-ply. Two output modes, inherited from the
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
    CubefulEncoder, CubePerspective, MatchplayEncoder, get_encoder,
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


# Cube composition for oneply_moves ranking. The composed form
# max(v_nd, min(v_dt, v_dp)) is the CORRECT quantity -- the opponent's MWC
# at a successor where their cube window is live includes the option --
# but correct and stronger are different claims, and it changed 6 of 128
# rankings on the confirmed-error set. What is MEASURED is the uncomposed
# form: +2.00pp over 0-ply across 12,000 matches (z=4.38).
# Default therefore stays on the measured variant; set
# TDAGENT_ONEPLY_MOVES_CUBE=1 to compose, and screen the two before
# changing this default.
_ONEPLY_MOVES_CUBE = os.environ.get(
    "TDAGENT_ONEPLY_MOVES_CUBE", "0") not in ("0", "", "false", "False")


class TDAgent(Agent):
    """Scalar agent built on TDNetwork (probability or equity output
    mode, inherited from the network)."""

    # ── GPU eval server hookup (opt-in, off by default) ──────────────
    # Set by `trainer._collect_worker` when `--gpu-eval-server` is
    # active: a `gpu_eval_server.RemoteEvaluator` that ships matchplay
    # feature matrices to the single GPU server process instead of
    # running the forward locally. Class attributes so the default path
    # costs one attribute lookup and `__init__` is untouched.
    _remote_eval = None          # None => every forward stays local
    # c_api forward backend (c_api/fast_forward.py), opt-in through
    # BG_FAST_FORWARD=<torch|simd|cblas|ref> for matchplay agents on CPU.
    # Set, it replaces the torch forward in the matchplay value paths
    # (evaluate_matchplay, _matchplay_values_np, 0-ply move selection),
    # and 0-ply move selection also takes the C move generator. None
    # (the default) keeps every path exactly as before.
    _fast_ff = None
    _remote_eval_moves = False   # also route choose_checker_action_matchplay

    def __init__(
        self,
        network: TDNetwork,
        encoder=None,
        device: str = "cpu",
        loss_fn: Optional[Callable] = None,
        use_fast_engine: bool = True,
        oneply: bool = False,
        boltzmann_temp: float = 0.0,
        bf16_inference: bool = False,
        met=None,
        met_scopes=None,
        oneply_moves: bool = False,
        oneply_moves_k: int = 0,
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
            boltzmann_temp: softmax exploration temperature for
                chequer play. 0.0 = greedy argmin.
            bf16_inference: run 1-ply value forwards on a bf16 copy of
                the net. Training keeps fp32 weights; meant for
                collection workers.
            met, met_scopes: match play only. A MatchEquityTable for
                the new-game MWC, and the consumers it grounds
                ("cube", "terminal", "boundary"; default all). See
                `_eval_new_game`.
            oneply_moves, oneply_moves_k: match play only. Rank 0-ply
                move candidates by their own 1-ply value, for the top
                k by the 0-ply read (0 = all).
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
        self.is_matchplay = isinstance(encoder, MatchplayEncoder)
        if self.is_cubeful and self.output_mode != "equity":
            raise ValueError(
                "CubefulEncoder requires output_mode='equity'; "
                f"got {self.output_mode!r}."
            )
        if self.is_matchplay and self.output_mode != "probability":
            raise ValueError(
                "MatchplayEncoder requires output_mode='probability' "
                f"(P(win match)); got {self.output_mode!r}."
            )
        self.oneply = oneply
        self.oneply_moves = bool(oneply_moves)
        # Top-K filter for oneply_moves: re-price only the K best successors
        # by the cheap 0-ply read, leaving the rest at it. Full 1-ply
        # selection costs 234x a 0-ply decision (measured), which is fine for
        # play but not for generating training data; K=3-4 keeps nearly all
        # of the benefit because the argmin is decided among the top few.
        # 0 = no filter (re-price every successor).
        self.oneply_moves_k = int(oneply_moves_k)
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
                    (self.is_cubeful or self.is_matchplay)
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

        # New-game MWC memo for matchplay (see _eval_new_game).
        self._new_game_memo = {}
        # Optional MatchEquityTable. When set, new-game MWC comes from
        # the table instead of the network's own evaluation — see
        # _eval_new_game for why that matters, and for what each scope
        # covers. Default grounds every consumer.
        self.met = met
        self.met_scopes = frozenset(
            met_scopes if met_scopes is not None
            else ("cube", "terminal", "boundary")
        )
        unknown = self.met_scopes - {"cube", "terminal", "boundary"}
        if unknown:
            raise ValueError(f"unknown met scope(s): {sorted(unknown)}")
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

        from c_api.fast_forward import backend_from_env
        from c_api.fast_forward import supports as _ff_supports
        _ff_backend = backend_from_env()
        if (_ff_backend is not None and self.is_matchplay
                and self._c_base_available and not self.bf16_inference
                and _ff_supports(self.network)):
            self.enable_fast_forward(_ff_backend)

    def enable_fast_forward(self, backend: Optional[str]) -> None:
        """Route this matchplay agent's forwards through a c_api backend
        (None: back to torch). CPU only, and needs the C base encoder,
        because 0-ply move selection then also uses the C move generator.
        The backend holds a copy of the weights, re-copied automatically
        when the net's parameters change in place."""
        if backend is None:
            self._fast_ff = None
            return
        if not self.is_matchplay:
            raise ValueError("fast forward is wired for matchplay agents only")
        if not self._c_base_available:
            raise ValueError("fast forward needs the CPU C engine (bg_fast)")
        if self.bf16_inference:
            raise ValueError("fast forward and bf16 inference are exclusive")
        from c_api.fast_forward import FastForward
        self._fast_ff = FastForward(self.network, backend)

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

    def _boltzmann_select(self, values: torch.Tensor, rng=None) -> int:
        """Index sampled by softmax over negated opponent-view values
        (lower is better for us). Argmin when temp=0.

        `rng` (a `random.Random`) draws the sample from the episode's
        own stream instead of torch's global one, so episodes stay
        reproducible when the vectorized collector interleaves many in
        one process: one draw per sampled decision, consumed in the same
        order by the serial and generator collectors. temp == 0 consumes
        no draw.
        """
        if self.boltzmann_temp <= 0.0 or len(values) == 1:
            return int(torch.argmin(values).item())
        logits = -values.squeeze() / self.boltzmann_temp
        probs = torch.softmax(logits, dim=0)
        if rng is None:
            return int(torch.multinomial(probs, 1).item())
        u = rng.random()
        acc = 0.0
        pl = probs.tolist()
        for i, pi in enumerate(pl):
            acc += pi
            if u < acc:
                return i
        return len(pl) - 1

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

    def _matchplay_values_np(self, features: np.ndarray) -> np.ndarray:
        """(N, n_feat) fp32 features -> (N,) fp32 values.

        Routes through the GPU eval server when one is attached
        (`--gpu-eval-server`), else runs the forward locally. The local
        branch is byte-for-byte the expression it replaced, so with no
        server attached -- the default -- this is not a behaviour change.

        The server returns None when the request exceeds its shared
        buffer, which is why the local forward is not an `else`: it is
        also the overflow fallback.
        """
        remote = self._remote_eval
        if remote is not None:
            out = remote.forward(features)
            if out is not None:
                return out
        if self._fast_ff is not None:
            return self._fast_ff(features)
        t = self._to_device(torch.from_numpy(features))
        with torch.no_grad():
            return self.network(t).cpu().numpy().ravel()

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

    # ── Matchplay value / cube / checker methods ────────────────────────
    #
    # These exist only when self.is_matchplay is True. The network
    # output is P(on-roll player wins the MATCH) ∈ [0, 1]; all values
    # below are match-winning chances (MWC), never money equity, and
    # cube_value NEVER multiplies a target — it only drives score
    # transitions via MatchState.after_game.
    #
    # Perspective rules (the historical source of most matchplay bugs):
    #   - MatchState stores away scores in WHITE's view, always.
    #   - _match_features flips them to the viewer's view; the viewer
    #     is state.turn for board encodings.
    #   - Every value is P(<viewer> wins match); flip with 1 - V.

    def _match_features(self, match_state, viewer: int):
        """(cube_persp, cube_value, my_away, opp_away, crawford) from
        `viewer`'s perspective. Away scores auto-flip for BLACK."""
        from modes import cube_perspective  # local: modes imports td_agent
        persp = cube_perspective(match_state.cube_owner, viewer)
        if viewer == 0:  # WHITE
            my_away, opp_away = match_state.my_away, match_state.opp_away
        else:
            my_away, opp_away = match_state.opp_away, match_state.my_away
        return (
            persp, match_state.cube_value, my_away, opp_away,
            match_state.is_crawford,
        )

    def _encode_matchplay(
        self, state: BoardState, match_state, is_cube_action: bool = False,
    ) -> np.ndarray:
        """Encode `state` from the on-roll player's perspective with
        match features (cube + away scores + crawford) appended. Uses
        the C fast path for base features when available."""
        persp, cube_value, my_away, opp_away, crawford = (
            self._match_features(match_state, state.turn)
        )
        if self._c_base_available:
            base = _bg_fast.encode_state(state)
        else:
            base = self.encoder._base.encode(state)
        return self.encoder.encode_with_base(
            base, persp, cube_value, my_away, opp_away, crawford,
            is_cube_action,
        )

    def evaluate_matchplay(
        self, state: BoardState, match_state, is_cube_action: bool = False,
    ) -> float:
        """P(on-roll player at `state` wins the match)."""
        x = self._encode_matchplay(state, match_state, is_cube_action)
        if self._fast_ff is not None:
            return float(self._fast_ff(x)[0])
        t = torch.from_numpy(x)
        t = self._to_device(t)
        with torch.no_grad():
            return self.network(t.unsqueeze(0)).item()

    def _eval_new_game(self, match_state, player: int,
                       scope: str = "cube") -> float:
        """P(`player` wins the match) standing at the START of a new
        game at `match_state`'s score, before the opening roll.

        This is the single point through which ALL cross-game equity
        enters matchplay: `v_dp` in the cube decision, the responder's
        take point, terminal-successor pricing in move selection (i.e.
        what a gammon is worth at this score), and the collector's
        game-boundary grounding target.

        With `self.met` set it is an exact table lookup. Without it the
        network is its own match-equity table, the "grounding hole":
        every one of the above becomes a bootstrap off the net's own
        values, so the only truly grounded signal in a match is its
        final outcome, and pure TD(0) admits self-consistent wrong fixed
        points at cube decisions. The recorded symptom was v_dp at 2-away/2-away
        drifting 0.680 -> 0.639 and taking the cube passive with it.

        `scope` names the CALLER, so grounding can be applied per
        consumer via `met_scopes`:
          "cube"     — v_dp in offer/respond_to_double. This one is a
                       PURE table lookup: being dropped scores
                       cube_value points, and what that is worth is a
                       MET entry with no board dependence at all.
          "terminal" — gammon/backgammon pricing of terminal successors
                       during move selection and 1-ply targets.
          "boundary" — the collector's game-boundary training target.
                       Unlike the other two this one trains the net's
                       OWN scale toward the table.
        v_nd and v_dt are deliberately absent: they are board-dependent
        (they need this position's win/gammon/backgammon rates) and no
        table can supply them. Grounding those needs a different output
        representation, not a bigger table.

        `_eval_new_game_net` keeps the learned version reachable — it
        is the implicit-MET diagnostic (`eval_matchplay.py --met-only`),
        and with a table pinned it becomes a direct measure of how far
        the net's internal beliefs have drifted from ground truth.
        """
        if self.met is None or scope not in self.met_scopes:
            return self._eval_new_game_net(match_state, player)
        ms = match_state.start_game()
        # MatchState stores away scores in WHITE's view, and player 0
        # is WHITE — the same convention met.mwc() takes.
        p_white = self.met.mwc(
            ms.my_away, ms.opp_away, ms.crawford_done, ms.is_crawford,
        )
        return p_white if player == 0 else 1.0 - p_white

    def _eval_new_game_net(self, match_state, player: int) -> float:
        """The network's own new-game MWC (the pre-MET behaviour).

        The network serves as its own implicit match-equity table:
        evaluate the initial position once with WHITE on roll and once
        with BLACK on roll, and average — who wins the opening roll is
        worth real equity, and averaging removes that bias. Callers
        must check match_over() first.

        Memoized per weight-version: the score lattice is tiny (a few
        (my_away, opp_away, crawford) combinations) while bear-off
        moves can hit this dozens of times per turn via terminal
        successors. The memo MUST be cleared whenever the network
        weights change (see clear_matchplay_memo); collection workers
        rebuild the agent per round so they are always consistent."""
        key = (
            match_state.my_away, match_state.opp_away,
            match_state.crawford_done, match_state.match_length,
        )
        hit = self._new_game_memo.get(key)
        if hit is None:
            ms = match_state.start_game()
            initial = BoardState.initial()
            v_white_on_roll = self.evaluate_matchplay(initial, ms)
            v_black_on_roll = self.evaluate_matchplay(switch_turn(initial), ms)
            hit = 0.5 * (v_white_on_roll + (1.0 - v_black_on_roll))
            self._new_game_memo[key] = hit
        return hit if player == 0 else 1.0 - hit

    def clear_matchplay_memo(self) -> None:
        """Invalidate the new-game MWC memo. Must be called after any
        in-place weight change when the same agent keeps collecting
        (the master's single-process path); worker processes rebuild
        their agent every round and never need it."""
        self._new_game_memo.clear()

    def offer_double_matchplay(self, state: BoardState, match_state) -> CubeOffer:
        """Matchplay pre-roll cube decision for `state.turn`.

        Compares MWC of playing on vs doubling:
          v_nd: P(win match) playing on at the current cube.
          v_dt: P(win match) if we double and opponent takes (cube
                transfers, value doubles; we are still on roll).
          v_dp: P(win match) if we double and opponent passes — we
                score cube_value points and a new game starts (1.0 if
                that wins the match).
        The opponent picks the response worse for us, so:
          double iff min(v_dt, v_dp) > v_nd.
        """
        player = state.turn
        if not match_state.can_offer(player):
            return CubeOffer(should_double=False, _cache=None)

        v_nd = self.evaluate_matchplay(state, match_state, is_cube_action=False)
        ms_take = match_state.after_take(player)
        v_dt = self.evaluate_matchplay(state, ms_take, is_cube_action=False)
        ms_pass = match_state.after_game(1, winner_is_me=(player == 0))
        if ms_pass.match_over():
            v_dp = 1.0
        else:
            v_dp = self._eval_new_game(ms_pass, player, scope="cube")

        should_double = min(v_dt, v_dp) > v_nd
        return CubeOffer(
            should_double=should_double,
            _cache={"v_nd": v_nd, "v_dt": v_dt, "v_dp": v_dp},
        )

    def respond_to_double_matchplay(
        self,
        state: BoardState,
        match_state,
        hint: Optional[CubeOffer] = None,
    ) -> bool:
        """Take/pass from the player who was just doubled. True = take.

        Convention: `state.turn` is the DOUBLER (cube decisions happen
        pre-roll, before the doubler's dice). All cached/recomputed
        values are P(doubler wins match); zero-sum gives the responder
        rule directly: take iff 1 - v_dt > 1 - v_dp, i.e. v_dp > v_dt.
        """
        if hint is not None and hint._cache and "v_dt" in hint._cache:
            return hint._cache["v_dp"] > hint._cache["v_dt"]
        doubler = state.turn
        ms_take = match_state.after_take(doubler)
        v_dt = self.evaluate_matchplay(state, ms_take, is_cube_action=False)
        ms_pass = match_state.after_game(1, winner_is_me=(doubler == 0))
        if ms_pass.match_over():
            v_dp = 1.0
        else:
            v_dp = self._eval_new_game(ms_pass, doubler, scope="cube")
        return v_dp > v_dt

    def choose_checker_action_matchplay(
        self, state: BoardState, dice, match_state, rng=None,
    ):
        """Matchplay 0-ply move selection. Returns `(next_state,
        bootstrap)` or None on forced pass. `bootstrap` is P(mover wins
        match) of the chosen successor.

        Successors are encoded from the opponent's perspective (the
        engine switches turn), so we argmin over P(opp wins match).
        Terminal successors (mover just won the game) are overridden
        with the true value: 1.0 if the resulting score wins the match,
        else the new-game evaluation at the updated score — this is how
        gammon value at the current score (e.g. 2-away) enters move
        selection.
        """
        fast = self._fast_ff
        c_gen = False
        if fast is not None:
            # C move generation + base encode in one call: the same plays
            # in the same order as get_legal_plays, except that C stops at
            # bg_fast.MAX_PLAYS (4096) successors -- never reached in play
            # (most seen ~1200) -- where we fall back (tests/test_fast_forward.py).
            base_feats, next_states = _bg_fast.get_legal_plays_encoded(
                state, dice)
            c_gen = len(next_states) < _bg_fast.MAX_PLAYS
            if c_gen and not next_states:
                return None
            if c_gen and self.oneply_moves:
                # The lazy view aliases bg_fast's play buffer, which the
                # 1-ply expansion below reuses.
                next_states = [next_states[i]
                               for i in range(len(next_states))]
        if not c_gen:
            plays = get_legal_plays(state, dice)
            if not plays:
                return None
            next_states = [s for _, s in plays]
        mover = state.turn
        opponent = 1 - mover

        if c_gen:
            pass  # base_feats came with the C move generation
        elif self._c_base_available:
            base_feats = np.stack([
                _bg_fast.encode_state(s) for s in next_states
            ])
        else:
            base = self.encoder._base
            base_feats = np.stack([base.encode(s) for s in next_states])

        persp, cube_value, my_away, opp_away, crawford = (
            self._match_features(match_state, opponent)
        )
        opp_cube_action = match_state.can_offer(opponent)
        extra_row = self.encoder.match_extras(
            persp, cube_value, my_away, opp_away, crawford,
            opp_cube_action,
        )
        extra = np.tile(extra_row, (len(next_states), 1))
        features = np.concatenate([base_feats, extra], axis=1)

        t = torch.from_numpy(features)
        t = self._to_device(t)
        # P(opp wins match) per successor. Only 4% of a worker's rows
        # (mean batch ~18) for 50% of its forwards, so routing this is a
        # separate opt-in: it doubles the server request rate to capture
        # a twentieth of the FLOPs. `t` is still built either way -- it
        # is a zero-copy view on CPU -- so the terminal-mask and
        # selection code below is untouched.
        values = None
        if self._remote_eval is not None and self._remote_eval_moves:
            _vals = self._remote_eval.forward(features)
            if _vals is not None:
                values = torch.from_numpy(_vals)
        if values is None and fast is not None:
            values = torch.from_numpy(fast(features))
        if values is None:
            with torch.no_grad():
                values = self.network(t)

        terminal_mask = t[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
        if terminal_mask.any():
            values = values.clone()
            for i in torch.nonzero(terminal_mask).flatten().tolist():
                gr = next_states[i].game_result()
                new_ms = match_state.after_game(
                    gr, winner_is_me=(mover == 0),
                )
                if new_ms.match_over():
                    mover_val = 1.0
                else:
                    mover_val = self._eval_new_game(
                        new_ms, mover, scope="terminal")
                values[i] = 1.0 - mover_val
        if self.oneply_moves:
            # Rank successors by their OWN 1-ply value rather than the raw
            # network read. Measured over 128 positions where
            # gnubg agrees at 2 and 4 ply that the model's move was worse:
            # the raw read overvalues the successor it picks (rollout - V
            # = -14.2 mMWC) and undervalues the one it skips (+12.7) --
            # the signature of maximisation bias, argmin over noisy
            # estimates selecting the favourably-erred ones. The 21-dice
            # average is statistically unbiased against rollout truth on
            # the skipped move (+0.4 +/- 2.3), so ranking by it removes
            # the selection error rather than re-fitting the weights.
            # Terminal successors keep their exact override; they are
            # priced by MET, not by the net, so a 1-ply read is strictly
            # worse there.
            if not terminal_mask.any():
                values = values.clone()
            # Which successors are worth the deep read. `values` holds
            # P(opp wins), so the mover's best candidates are the SMALLEST.
            if self.oneply_moves_k > 0 and len(next_states) > self.oneply_moves_k:
                _flat = values.reshape(-1)
                _cand = set(torch.topk(_flat, self.oneply_moves_k,
                                       largest=False).indices.tolist())
            else:
                _cand = None
            for i in range(len(next_states)):
                if _cand is not None and i not in _cand:
                    continue
                if bool(terminal_mask[i]):
                    continue
                v = self.value_oneply_checker_matchplay(
                    next_states[i], match_state,
                )
                if opp_cube_action and _ONEPLY_MOVES_CUBE:
                    # `values` holds P(opp wins match) and the opponent is
                    # on roll at the successor, so when their cube window
                    # is live their MWC includes the cube option. The raw
                    # read this replaces came from the ca=1 channel, which
                    # carries that option; the 1-ply enumeration is the
                    # no-double value alone, so composing it back is what
                    # keeps the two comparable. Same composition the cube
                    # rows and _td_move_rows use.
                    c = self.offer_double_matchplay(
                        next_states[i], match_state)._cache
                    if c is not None:
                        v = max(v, min(c["v_dt"], c["v_dp"]))
                values[i] = v
        idx = self._boltzmann_select(values, rng=rng)
        chosen = next_states[idx]
        bootstrap = 1.0 - float(values[idx].item())
        return chosen, bootstrap

    def value_oneply_checker_matchplay(self, state: BoardState, match_state,
                                       cube_fold: bool = False) -> float:
        """1-ply MWC target for `state.turn` at a checker decision.

        Enumerates all 21 dice outcomes, picks the mover's best move
        per dice by 0-ply matchplay evaluation, and averages over dice
        probabilities. Terminal successors are priced exactly via
        after_game -> 1.0 / new-game MWC at the updated score — gammon
        value at the current score enters every dice branch here,
        which is the main signal the 0-ply chosen-move bootstrap
        only sees when a terminal move is actually on the board.

        Mirrors choose_checker_action_matchplay's encoding: successors
        are evaluated from the opponent's perspective with the
        opponent's cube-phase eligibility flag set.

        `cube_fold` (gnubg MakeCubePos/GetECF3 semantics): when the
        opponent has a live cube window at the successors, price it
        EXPLICITLY as the optimal-action composition the cube samples
        train with —
            V_opp(s') = max(v_nd', min(v_dt', v_dp'))
            v_nd' = V(s' | current cube, is_cube_action=0)
            v_dt' = V(s' | doubled cube, owner flipped, ca=0)
            v_dp' = new-game MWC after the opponent scores cube_value
        — instead of reading the net's is_cube_action=1 channel. An
        oracle probe measured that channel to carry an ownership-odd ~±10 mMWC
        miscalibration which the plain 1-ply target transports into
        every checker value; the fold replaces the aliased read with
        the same backup formula used everywhere else.
        """
        mover = state.turn
        opponent = 1 - mover
        persp, cube_value, my_away, opp_away, crawford = (
            self._match_features(match_state, opponent)
        )
        opp_cube_action = match_state.can_offer(opponent)

        feat_chunks: List[np.ndarray] = []
        per_dice: List[Tuple[Optional[int], Optional[int], float, Optional[float]]] = []
        term_overrides: List[Tuple[int, float]] = []  # (global idx, opp-view value)
        forced_value: Optional[float] = None  # dice-independent
        pos = 0
        for d1, d2 in _DICE_OUTCOMES:
            prob = (1.0 / 36.0) if d1 == d2 else (2.0 / 36.0)
            if self._c_base_available:
                # C fast path: move-gen + base encode in one call. The
                # lazy states view is invalidated by the NEXT _bg_fast
                # call, so terminal states are materialized within this
                # iteration; the features array is owned and persists.
                base_feats, states_view = _bg_fast.get_legal_plays_encoded(
                    state, (d1, d2),
                )
                n_moves = base_feats.shape[0] if base_feats is not None else 0
            else:
                plays = get_legal_plays(state, (d1, d2))
                n_moves = len(plays)
                if n_moves:
                    states_view = [s for _, s in plays]
                    base = self.encoder._base
                    base_feats = np.stack([
                        base.encode(s) for s in states_view
                    ])
            if not n_moves:
                if forced_value is None:
                    opp_view = switch_turn(state)
                    if cube_fold and match_state.can_offer(opp_view.turn):
                        v_nd = self.evaluate_matchplay(
                            opp_view, match_state, is_cube_action=False)
                        ms_take = match_state.after_take(opp_view.turn)
                        v_dt = self.evaluate_matchplay(
                            opp_view, ms_take, is_cube_action=False)
                        ms_pass = match_state.after_game(
                            1, winner_is_me=(opp_view.turn == 0))
                        v_dp = (1.0 if ms_pass.match_over()
                                else self._eval_new_game(
                                    ms_pass, opp_view.turn, scope="cube"))
                        v_opp = max(v_nd, min(v_dt, v_dp))
                    else:
                        v_opp = self.evaluate_matchplay(
                            opp_view, match_state,
                            is_cube_action=match_state.can_offer(
                                opp_view.turn),
                        )
                    forced_value = 1.0 - v_opp
                per_dice.append((None, None, prob, forced_value))
                continue
            terminal_mask = base_feats[:, OPP_OFF_INDEX] >= TERMINAL_OFF_THRESHOLD
            if terminal_mask.any():
                for i in np.nonzero(terminal_mask)[0]:
                    gr = states_view[int(i)].game_result()
                    new_ms = match_state.after_game(
                        gr, winner_is_me=(mover == 0),
                    )
                    if new_ms.match_over():
                        mover_val = 1.0
                    else:
                        mover_val = self._eval_new_game(
                        new_ms, mover, scope="terminal")
                    term_overrides.append((pos + int(i), 1.0 - mover_val))
            feat_chunks.append(base_feats)
            per_dice.append((pos, pos + n_moves, prob, None))
            pos += n_moves

        if pos == 0:
            return forced_value  # every dice is a forced pass

        big_base = (
            feat_chunks[0] if len(feat_chunks) == 1
            else np.concatenate(feat_chunks, axis=0)
        )
        if cube_fold and opp_cube_action:
            # Explicit GetECF3 fold at every successor: two ca=0 reads
            # (current cube; doubled cube, owner flipped to the mover)
            # composed with the score-only drop value. v_dp is
            # successor-independent: it prices the opponent scoring
            # cube_value points from the CURRENT score.
            ms_take = match_state.after_take(1 - mover)
            persp_t, cube_t, my_t, opp_t, craw_t = (
                self._match_features(ms_take, opponent)
            )
            row_nd = self.encoder.match_extras(
                persp, cube_value, my_away, opp_away, crawford, False,
            )
            row_dt = self.encoder.match_extras(
                persp_t, cube_t, my_t, opp_t, craw_t, False,
            )
            ms_pass = match_state.after_game(
                1, winner_is_me=(opponent == 0))
            v_dp = (1.0 if ms_pass.match_over()
                    else self._eval_new_game(ms_pass, opponent,
                                             scope="cube"))
            f_nd = np.concatenate(
                [big_base, np.tile(row_nd, (big_base.shape[0], 1))],
                axis=1)
            f_dt = np.concatenate(
                [big_base, np.tile(row_dt, (big_base.shape[0], 1))],
                axis=1)
            v_nd = self._matchplay_values_np(f_nd)
            v_dt = self._matchplay_values_np(f_dt)
            values = np.maximum(v_nd, np.minimum(v_dt, v_dp))
        else:
            extra_row = self.encoder.match_extras(
                persp, cube_value, my_away, opp_away, crawford,
                opp_cube_action,
            )
            extra = np.tile(extra_row, (big_base.shape[0], 1))
            features = np.concatenate([big_base, extra], axis=1)

            # P(opp wins match) per successor. Routed to the GPU eval
            # server when one is attached; this is 96% of the rows a
            # matchplay 1-ply worker pushes through the net.
            values = self._matchplay_values_np(features)
        for idx, v in term_overrides:
            values[idx] = v

        total = 0.0
        for start, end, prob, forced in per_dice:
            if start is None:
                total += prob * forced
            else:
                # Mover picks the move minimizing the opponent's MWC.
                total += prob * (1.0 - float(values[start:end].min()))
        # Dice probabilities sum to 1 only up to float rounding; keep
        # the MWC target inside [0, 1].
        return min(1.0, max(0.0, total))

    def offer_double(self, state: BoardState, match_state) -> CubeOffer:
        """Money-game pre-roll cube decision: per-unit equity at the
        current cube vs 2x per-unit equity if doubled, with the Jacoby
        rule on centered cubes. The returned CubeOffer caches
        v_double_take so respond_to_double() can skip a forward.
        """
        if self.is_matchplay:
            return self.offer_double_matchplay(state, match_state)
        if not self.is_cubeful:
            return super().offer_double(state, match_state)
        if self.oneply:
            return self._offer_double_oneply(state, match_state)
        from modes import cube_perspective  # local: modes imports td_agent
        player = state.turn
        no_double_persp = cube_perspective(match_state.cube_owner, player)
        if no_double_persp not in (CubePerspective.CENTERED, CubePerspective.MINE):
            return CubeOffer(should_double=False, _cache=None)

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
        if self.is_matchplay:
            return self.respond_to_double_matchplay(
                state, match_state, hint=hint,
            )
        if not self.is_cubeful:
            return super().respond_to_double(state, match_state, hint=hint)
        # Use the cached v_double_take at the matching ply, else recompute.
        if self.oneply:
            key = "v_double_take_1ply"
        else:
            key = "v_double_take_0ply"
        if hint is not None and hint._cache is not None and key in hint._cache:
            v_double_take = hint._cache[key]
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
        """Cubeful move selection at 0- or 1-ply. Returns
        `(next_state, bootstrap)` or None on a forced pass."""
        if self.is_matchplay:
            return self.choose_checker_action_matchplay(
                state, dice, match_state,
            )
        if not self.is_cubeful:
            return super().choose_checker_action_cubeful(
                state, dice, match_state,
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
    ):
        """Level-1 dice expansion of the cubeful 1-ply value
        (value_oneply_checker_cubeful without the C 21-dice expansion):
        per dice, the mover's per-unit equity of every successor
        (terminals from game_result, or +1 when gammons don't count; the
        rest -V(opp) from one batched forward with the opponent's cube
        perspective).

        Returns `dice_results`, indexed by dice outcome.
        """
        base = self.encoder._base
        n_feat = self.encoder.num_features
        n_base = base.num_features
        opp_col = int(opp_persp)

        dice_results: List[Optional[List[float]]] = [None] * len(_DICE_OUTCOMES)
        dice_nonterm_indices: List[Optional[np.ndarray]] = [None] * len(_DICE_OUTCOMES)
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

        return dice_results

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

        dice_results = self._expand_cubeful_level1(
            state, opp_persp, opp_cube_action, gammons_count,
        )

        probs = _ONEPLY_PROBS.tolist()
        oneply_sum = 0.0
        for di, prob in enumerate(probs):
            equities = dice_results[di]
            if equities:
                oneply_sum += prob * max(equities)

        return oneply_sum

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
