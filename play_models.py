"""
play_models.py -- Play two sides against each other for N games and
report the head-to-head.

Either side is a checkpoint path or an engine keyword (gnubg-nn,
gnubg-ext*, opensage*, random). Colors are assigned randomly each game.
Reports win counts with a Wilson CI and, in money modes, bootstrap
equity CIs. Raw per-game results can be saved (--save-hh) and pooled
later (combine_head_to_head).

Usage examples:
    python play_models.py --model1 model_a.pt --model2 model_b.pt --games 1000
    python play_models.py --model1 model.pt --model2 random --games 500
"""

import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import random
import signal
import sys
import time
import math
import multiprocessing as mp
from typing import Tuple

from backgammon_engine import (
    WHITE,
    get_legal_plays, switch_turn, opening_roll,
)
from agents import Agent, RandomAgent
from td_agent import TDAgent
from model import ProbNetwork, load_model
from prob_agent import ProbAgent, PROB5_DEPTH_ERROR, board_sanity_from_env


def _load_agent(path: str, oneply: bool = False, twoply_k: int = 0,
                bf16: bool = False, board_sanity: bool = False) -> Agent:
    """Build the agent for a checkpoint: prob5 -> ProbAgent, else TDAgent.

    twoply_k>0 is TDAgent-only (prob5 raises). bf16 and board_sanity
    apply to ProbAgent only. ProbAgent has no DMP switch: game_mode dmp
    changes only the scoring.
    """
    net = load_model(path)
    if isinstance(net, ProbNetwork):
        if twoply_k > 0:
            raise ValueError(f"{path}: {PROB5_DEPTH_ERROR}")
        return ProbAgent(net, plies=1 if oneply else 0, bf16_inference=bf16,
                         board_sanity=board_sanity)
    return TDAgent(net, oneply=oneply, twoply_k=twoply_k)


def _load_c_agent(path: str, oneply: bool = False, twoply_k: int = 0,
                  backend: str = None, bf16: bool = False,
                  board_sanity: bool = False) -> Agent:
    """The C API (c_api/, libpuretd) agent for a checkpoint: 0- or 1-ply,
    fp32, no prob5_sanity (the options it lacks raise)."""
    if twoply_k > 0:
        raise ValueError("the C API plays 0- and 1-ply sides only (2-ply: "
                         "--twoply-k)")
    from c_api.puretd_py import CAgent, KIND_PROB5, play_backend
    agent = CAgent(path, plies=1 if oneply else 0, backend=play_backend(backend))
    if agent.kind == KIND_PROB5 and bf16:
        agent.engine.close()
        raise ValueError("the C API is fp32 only (--bf16)")
    if agent.kind == KIND_PROB5 and board_sanity:
        agent.engine.close()
        raise ValueError("the C API has no prob5_sanity (PROB5_BOARD_SANITY=1)")
    return agent


_JAN_C_UNSUPPORTED = ("JANOWSKI_NO_FOLD", "JANOWSKI_NO_FOLD_0PLY",
                      "JANOWSKI_CUBELESS_CHECKER")


def _load_c_janowski(path: str, oneply: bool = False, backend: str = None,
                     bf16: bool = False) -> Agent:
    """The C API Prob5Jan agent (janowski-c:, or janowski: on the C path):
    JANOWSKI_X, JANOWSKI_BOARD_SANITY and JANOWSKI_NO_BEAROFF honoured; the
    ablations and bf16 it lacks raise."""
    for k in _JAN_C_UNSUPPORTED:
        if os.environ.get(k, "0") == "1":
            raise ValueError(f"the C Prob5Jan has no {k}=1")
    if bf16:
        raise ValueError("the C Prob5Jan has no bf16 (drop --bf16)")
    if not _is_prob5_checkpoint(path):
        raise ValueError(f"janowski-c: requires a prob5 checkpoint (got {path})")
    from c_api.puretd_py import CAgent, play_backend
    x_env = os.environ.get("JANOWSKI_X", "")
    return CAgent(path, plies=1 if oneply else 0, backend=play_backend(backend),
                  janowski=True,
                  x=float(x_env) if x_env else None,
                  board_sanity=board_sanity_from_env("JANOWSKI_BOARD_SANITY", True),
                  bearoff=os.environ.get("JANOWSKI_NO_BEAROFF", "0") != "1")


def _c_api_game(agent_white, agent_black, cubeful, jacoby, dice_rng):
    """Play one game in C (bgc_play_game) when both sides are C API agents
    on one backend and the dice source exposes random.Random state (the
    global ``random`` module, a ``random.Random`` or a DiceStream). Returns
    (winner, stake, cube_value, ended_by_drop), or None to use the Python
    loop. Same decisions and dice as the Python loop; the dice source's
    state (and a DiceStream's n_dice) advance as the Python loop's would.
    """
    try:
        from c_api import puretd_py as P
    except Exception:
        return None
    if not (isinstance(agent_white, P.CAgent) and isinstance(agent_black, P.CAgent)):
        return None
    ew, eb = agent_white.engine, agent_black.engine
    if ew.lib is not eb.lib:
        return None
    src = dice_rng._rng if isinstance(dice_rng, DiceStream) else \
        (random if dice_rng is None else dice_rng)
    if not (hasattr(src, "getstate") and hasattr(src, "setstate")):
        return None
    state = src.getstate()
    rng = P.Rng.from_state(state, ew.backend)
    opts = P.game_opts(cubeful=cubeful, jacoby=jacoby, cube_enabled=True,
                       plies_white=agent_white.plies,
                       plies_black=agent_black.plies)
    res, _ = P.play_game(ew, eb, opts, rng)
    src.setstate(rng.getstate(state[2]))
    if isinstance(dice_rng, DiceStream):
        dice_rng.n_dice += res["n_dice"]
    if not cubeful:
        return res["winner"], res["result"], 1, False
    stake = res["cube_value"] if res["ended_by_drop"] \
        else res["cube_value"] * res["result"]
    return res["winner"], stake, res["cube_value"], bool(res["ended_by_drop"])


class DiceStream:
    """A dedicated dice source for ONE game (paired-dice mode).

    Wraps its own ``random.Random(seed)``: the dice of a game are a pure
    function of ``seed`` and of how many dice the game has consumed. Roll
    k goes to whoever is on move at turn k (opening roll included).
    Nothing else may draw from it; colour choice, tie-breaks and the
    worker's own RNG use the global ``random`` module, so two games
    seeded alike see identical dice. ``n_dice`` counts the single dice
    drawn (2 per roll, opening rerolls included) for diagnostics.
    """
    __slots__ = ("_rng", "n_dice")

    def __init__(self, seed):
        self._rng = random.Random(seed)
        self.n_dice = 0

    def randint(self, a, b):
        self.n_dice += 1
        return self._rng.randint(a, b)


def pair_seed(seed_base: str, pair_index: int) -> int:
    """Seed of pair ``pair_index`` (0-based, global across workers) under
    the run's ``seed_base`` (32 hex chars of OS entropy, recorded in the
    npz). SHA-256(base || index) -> 256-bit int, so pairs are distinct
    within a run and runs are distinct across chunks/boxes."""
    import hashlib
    h = hashlib.sha256(bytes.fromhex(seed_base)
                       + int(pair_index).to_bytes(8, "little")).digest()
    return int.from_bytes(h, "little")


def play_game(
    agent_white: Agent,
    agent_black: Agent,
    record: bool = False,
    cubeful: bool = False,
    jacoby: bool = True,
    dice_rng=None,
    stats: dict = None,
    cube_cap: int = None,
):
    """Play one full game. Returns `(winner, stake)`, plus the
    GameRecord when `record=True`.

    Cubeless: stake is 1/2/3 (single / gammon / backgammon).
    Cubeful: stake is the points won (cube_value × result, or
    cube_value on a drop). ``cube_cap=N``: at cube N neither side is
    offered a double.

    ``dice_rng`` (optional): a dedicated dice source with ``randint``
    (see :class:`DiceStream`); every die of the game comes from it.
    ``None`` (default) rolls from the global ``random`` module.
    ``stats`` (optional dict) receives ``cube_value`` / ``ended_by_drop``
    for the game (1 / False when cubeless).

    Two C API agents (the default play path) on one backend play without
    a record through bgc_play_game (:func:`_c_api_game`): same decisions,
    dice and return values as the Python loop. A cube cap keeps the Python loop.
    """
    if not record and not (cubeful and cube_cap):
        fast = _c_api_game(agent_white, agent_black, cubeful, jacoby, dice_rng)
        if fast is not None:
            winner, stake, cube_value, drop = fast
            if stats is not None:
                stats["cube_value"] = cube_value
                stats["ended_by_drop"] = drop
            return winner, stake
    if cubeful:
        from third_party_opponents.gnubg.gnubg_eval import play_and_record_cubeful
        # Per-move Play bookkeeping costs a get_legal_plays call per
        # turn. Skip it unless the caller wants the record.
        game_record = play_and_record_cubeful(
            agent_white, agent_black, jacoby=jacoby,
            record_moves=record, rng=dice_rng, cube_cap=cube_cap,
        )
        winner = game_record.winner
        if game_record.ended_by_drop:
            stake = game_record.cube_value
        else:
            stake = game_record.cube_value * game_record.result
        if stats is not None:
            stats["cube_value"] = game_record.cube_value
            stats["ended_by_drop"] = game_record.ended_by_drop
        if record:
            return winner, stake, game_record
        return winner, stake

    if stats is not None:
        stats["cube_value"] = 1
        stats["ended_by_drop"] = False
    if record:
        # gnubg_eval.play_and_record is the one recording game loop.
        from third_party_opponents.gnubg.gnubg_eval import play_and_record
        game_record = play_and_record(agent_white, agent_black, rng=dice_rng)
        return game_record.winner, game_record.result, game_record

    if dice_rng is None:
        dice_rng = random
    choose_white = _fused_chooser(agent_white)
    choose_black = _fused_chooser(agent_black)
    # Opening roll: no doublet. Later turns roll normally.
    state, opening_dice = opening_roll(dice_rng)
    is_opening = True

    while not state.is_game_over():
        if is_opening:
            d1, d2 = opening_dice
            is_opening = False
        else:
            d1, d2 = dice_rng.randint(1, 6), dice_rng.randint(1, 6)
        chooser = choose_white if state.turn == WHITE else choose_black
        if chooser is not None:
            next_state = chooser(state, (d1, d2))
            state = switch_turn(state) if next_state is None else next_state
            continue
        plays = get_legal_plays(state, (d1, d2))

        if plays:
            agent = agent_white if state.turn == WHITE else agent_black
            play, next_state = agent.choose_checker_action(state, (d1, d2), plays)
            # next_state already has turn switched (engine convention)
            state = next_state
        else:
            state = switch_turn(state)

    return state.winner(), state.game_result()


def _fused_chooser(agent):
    """``(state, dice) -> next_state``, or None on a forced pass, with move
    generation and perspective196 encoding in one C call (bg_fast), for
    the cubeless 0-ply ProbAgent / TDAgent (exact types). Same choice as
    ``choose_checker_action`` over the Python engine's plays: identical
    successor order and features. Returns None for any other agent (the
    C API path's CAgent included)."""
    if not getattr(agent, "_fast_engine", False):
        return None
    if type(agent) is ProbAgent and agent.plies == 0:
        return agent.choose_next_state
    if (type(agent) is TDAgent and not agent.is_cubeful
            and not agent.oneply and agent.twoply_k == 0):
        def choose(state, dice, _choose=agent.choose_checker_action):
            r = _choose(state, dice)
            return None if r is None else r[1]
        return choose
    return None


def _play_one_signed(agent1, agent2, cubeful, jacoby, record=False,
                     a1_is_white=None, dice_rng=None, stats=None,
                     cube_cap=None):
    """Play one game. Colors are random unless ``a1_is_white`` is given
    (still fair: the opening roll decides who moves first). Returns
    (signed_equity, game_record_or_None); signed_equity > 0 when
    agent1 wins. ``dice_rng`` / ``stats`` (paired-dice mode) and
    ``cube_cap`` are passed through to :func:`play_game`."""
    if a1_is_white is None:
        a1_is_white = random.random() < 0.5
    first = agent1 if a1_is_white else agent2
    second = agent2 if a1_is_white else agent1
    result = play_game(first, second, record=record,
                       cubeful=cubeful, jacoby=jacoby,
                       dice_rng=dice_rng, stats=stats, cube_cap=cube_cap)
    if record:
        winner, stake, game_record = result
    else:
        winner, stake = result
        game_record = None
    white_equity = stake if winner == WHITE else -stake
    signed = white_equity if a1_is_white else -white_equity
    return signed, game_record


def _sage_filter_kw():
    """SAGE_UNPRUNED=1 disables Open Sage's root move filter (cubeless
    agent and cubeful analyzer alike). The env var reaches spawned
    H-H workers."""
    if os.environ.get("SAGE_UNPRUNED", "0") == "1":
        return {"filter_max_moves": 1000, "filter_threshold": 10.0}
    return {}


def _is_prob5_checkpoint(path) -> bool:
    import torch
    if not os.path.isfile(path):
        return False
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt.get("model_type") == "prob5"


def _check_side_depth(spec, spec_type, twoply_k):
    if twoply_k and spec_type in ("janowski", "janowski-c", "bo"):
        raise ValueError(f"{spec}: {spec_type}: sides support 0-ply and "
                         f"1-ply only (got twoply_k={twoply_k})")


def preflight_side(spec, spec_type, *, twoply_k=0):
    """Reject, before any game, a side that cannot be played as asked:
    an unsupported depth, janowski:/janowski-c:/bo: without the bearoff
    tables, a gnubg-ext* side on a gnubg other than the pinned one, or an
    opensage* side on another bgsage than SAGE_VERSION (RuntimeError)."""
    _check_side_depth(spec, spec_type, twoply_k)
    if twoply_k and spec_type == "model" and _is_prob5_checkpoint(spec):
        raise ValueError(f"{spec}: {PROB5_DEPTH_ERROR}")
    if spec_type == "model":
        board_sanity_from_env("PROB5_BOARD_SANITY", False)
    if spec_type in ("janowski", "janowski-c"):
        board_sanity_from_env("JANOWSKI_BOARD_SANITY", True)
    if spec_type.startswith("gnubg-ext"):
        from third_party_opponents.gnubg.gnubg_eval import assert_gnubg_bearoff
        assert_gnubg_bearoff()
    if spec_type in ("opensage", "opensage-analyzer", "opensage-native"):
        from third_party_opponents.opensage.opensage_agents import assert_sage_version
        assert_sage_version()
    if spec_type == "bo" or (
            spec_type in ("janowski", "janowski-c")
            and os.environ.get("JANOWSKI_NO_BEAROFF", "0") != "1"):
        import bearoff
        bearoff.preflight()


# Side types the C API plays (CAgent) unless --python-api.
_C_SIDE_TYPES = ("model", "janowski")


def resolve_c_api(c_api, sides, *, cubeful, bf16, backend=None, log=print):
    """The play path of checkpoint and janowski: sides: (use_c, backend).

    `c_api`: None = the default (the C API when it can serve every such
    side, else the Python path with one NOTE naming why), True = --c-api
    (an error instead of the fallback), False = --python-api. `sides`:
    (spec, spec_type, oneply, twoply_k) per side. `backend`:
    puretd_py.play_backend's (simd unless named). The library is built on
    demand, or rebuilt when its ABI is stale (default path only; --c-api
    raises). A janowski-c: side always plays in C; PURETD_ROOT_FILTER > 0
    makes the C API required."""
    from c_api import puretd_py as P
    backend = P.play_backend(backend)
    jan_c = any(t == "janowski-c" for _, t, _, _ in sides)
    c_sides = [s for s in sides if s[1] in _C_SIDE_TYPES]
    if c_api is False:
        if jan_c:
            raise ValueError("janowski-c: is the C engine; with --python-api "
                             "use janowski:")
        if c_sides:
            log("Play path: Python (--python-api)")
        return False, backend
    if c_api is None and c_sides and P.root_filter_from_env() > 0:
        c_api = True                        # the Python path has no root filter
    if not (c_sides or jan_c):
        return False, backend
    why = P.ready_lib(backend, rebuild_stale=c_api is None)
    if why and (jan_c or c_api):
        raise RuntimeError(why)
    reasons = [why] if why else []
    for spec, t, oneply, twoply_k in c_sides if not why else ():
        if t == "janowski" and not cubeful:
            continue                        # refused on either path
        try:
            agent = _build_side(spec, t, plies=0, sage_plies=1,
                                cubeless_money=False, cubeful=cubeful,
                                oneply=oneply, twoply_k=twoply_k, bf16=bf16,
                                c_api=True, c_api_backend=backend)
            agent.engine.close()
        except Exception as e:              # noqa: BLE001 -- the default never errors
            reasons.append(f"{spec}: {e}")
    if not reasons:
        if c_sides:
            log(f"Play path: C API ({backend})")
        return bool(c_sides), backend
    if c_api:
        raise ValueError("--c-api: " + "; ".join(reasons))
    log(f"NOTE: the C API (default play path) cannot serve this run: "
        f"{'; '.join(reasons)}. Checkpoint sides play on the Python path "
        f"(--python-api selects it without this check).")
    return False, backend


def _build_side(spec, spec_type, *, plies, sage_plies, cubeless_money,
                cubeful, oneply, twoply_k, bf16, c_api=False, c_api_backend=None):
    """Build one side of a head-to-head.

    `spec_type` is "model" (`spec` is a checkpoint path), "random",
    "hedgehog" (`spec` is ``hedgehog:<model.ogxf>``), "janowski"
    (``janowski:<prob5.pt>``; with `c_api` the C engine), "janowski-c"
    (``janowski-c:<prob5.pt>``, the C engine), "bo" (``bo:<cubeful.pt>``), or an engine
    keyword: "gnubg-nn", "gnubg-ext", "gnubg-ext-full",
    "gnubg-ext-rootfull", "opensage", "opensage-analyzer". `plies` is
    gnubg's depth; `sage_plies` is Open Sage's ply (1 = 0-ply,
    2 = 1-ply); a hedgehog side takes `oneply` as depth 1. `c_api`: a
    checkpoint side is a C API agent (CAgent).
    """
    _reject_legacy_gnubg(spec_type)
    if spec_type == "random":
        return RandomAgent()
    if spec_type == "gnubg-nn":
        from third_party_opponents.gnubg.gnubg_nn_agent import GnubgNNAgent
        return GnubgNNAgent(plies=plies, cubeless_money=cubeless_money,
                            cubeful=cubeful)
    if spec_type == "gnubg-ext":
        # Real gnubg 1.08 (third_party_opponents/gnubg/gnubg_external.py):
        # production pruning nets + Normal filter, native money cube + Jacoby.
        from third_party_opponents.gnubg.gnubg_external import GnubgExternalAgent
        return GnubgExternalAgent(plies=plies, cubeless=cubeless_money)
    if spec_type == "gnubg-ext-full":
        # No pruning nets, keep-all move filter: every legal move
        # deep-evaluated.
        from third_party_opponents.gnubg.gnubg_external import GnubgExternalAgent
        return GnubgExternalAgent(plies=plies, cubeless=cubeless_money,
                                  prune=False, movefilter="all")
    if spec_type == "gnubg-ext-rootfull":
        # Keep-all filter at the root, gnubg's default pruning for the
        # replies: the paper's protocol for matched 2-ply comparisons.
        from third_party_opponents.gnubg.gnubg_external import GnubgExternalAgent
        return GnubgExternalAgent(plies=plies, cubeless=cubeless_money,
                                  prune=True, movefilter="all")
    if spec_type == "opensage":
        if cubeful:
            raise ValueError(
                "opensage is cubeless/DMP only; for cubeful-money use "
                "'opensage-analyzer' (Sage's self-play-identical, cube-aware "
                "config).")
        from third_party_opponents.opensage.opensage_agents import OpenSageAgent
        return OpenSageAgent(
            sage_plies=sage_plies,
            mode="cubeless-money" if cubeless_money else "dmp",
            **_sage_filter_kw(),
        )
    if spec_type == "opensage-analyzer":
        # Open Sage's self-play config (BgBotAnalyzer: cube-aware checker
        # play + native cube_action). Cubeful-money only.
        if not cubeful:
            raise ValueError("opensage-analyzer is cubeful-money only; "
                             "use 'opensage' for cubeless/DMP")
        from third_party_opponents.opensage.opensage_agents import OpenSageAnalyzerAgent
        return OpenSageAnalyzerAgent(sage_plies=sage_plies, **_sage_filter_kw())
    if spec_type == "opensage-native":
        # Same decisions as opensage-analyzer at production settings, through
        # Open Sage's native C API (nets loaded once). Cubeful money, 0/1-ply.
        if not cubeful:
            raise ValueError("opensage-native is cubeful-money only; "
                             "use 'opensage' for cubeless/DMP")
        if _sage_filter_kw():
            raise ValueError("opensage-native: cubeful decisions do not "
                             "depend on SAGE_UNPRUNED; unset it")
        from third_party_opponents.opensage.opensage_mobile import OpenSageMobileAgent
        return OpenSageMobileAgent(sage_plies=sage_plies)
    if spec_type == "hedgehog":
        # HedgeHog engine behind our Agent interface; needs libhhagent.so
        # (third_party_opponents/hedgehog/README.md). Money-only: no DMP
        # criterion.
        if not (cubeful or cubeless_money):
            raise ValueError(
                "hedgehog sides are money-only (cubeful-money or "
                "cubeless-money); the engine has no DMP criterion")
        from third_party_opponents.hedgehog.hedgehog_agent import HedgehogAgent
        return HedgehogAgent(spec.split(":", 1)[1],
                             depth=2 if twoply_k else (1 if oneply else 0))
    _check_side_depth(spec, spec_type, twoply_k)
    if spec_type == "bo":
        # Cubeful TDAgent + exact bearoff routing, the Janowski
        # comparison's PureTD side (experiments/janowski/README.md).
        if not cubeful:
            raise ValueError("bo: sides are cubeful-money only")
        from bearoff_td import BearoffTDAgent
        return BearoffTDAgent(load_model(spec.split(":", 1)[1]), oneply=oneply)
    if spec_type == "janowski-c" or (spec_type == "janowski" and c_api):
        if not cubeful:
            raise ValueError(f"{spec_type}: sides are cubeful-money only")
        return _load_c_janowski(spec.split(":", 1)[1], oneply=oneply,
                                backend=c_api_backend, bf16=bf16)
    if spec_type == "janowski":
        # prob5 checkpoint + analytic Janowski cube layer. Env knobs,
        # recorded in the npz: JANOWSKI_X (cube-life index; unset =
        # janowski.DEFAULT_X), JANOWSKI_NO_FOLD=1 (no opponent-cube
        # fold in checker play), JANOWSKI_NO_FOLD_0PLY=1 (no 0-ply fold),
        # JANOWSKI_CUBELESS_CHECKER=1, JANOWSKI_NO_BEAROFF=1,
        # JANOWSKI_BOARD_SANITY=0 (no prob5_sanity).
        if not cubeful:
            raise ValueError(
                "janowski: sides are cubeful-money only; use the bare "
                "prob5 checkpoint path for cubeless modes")
        from janowski import DEFAULT_X
        from janowski_agent import JanowskiAgent
        net = load_model(spec.split(":", 1)[1])
        if not isinstance(net, ProbNetwork):
            raise ValueError("janowski: requires a prob5 checkpoint")
        x_env = os.environ.get("JANOWSKI_X", "")
        return JanowskiAgent(
            net, oneply=oneply, bf16_inference=bf16,
            x=float(x_env) if x_env else DEFAULT_X,
            cube_fold=os.environ.get("JANOWSKI_NO_FOLD", "0") != "1",
            fold_0ply=os.environ.get("JANOWSKI_NO_FOLD_0PLY", "0") != "1",
            cubeless_checker=os.environ.get(
                "JANOWSKI_CUBELESS_CHECKER", "0") == "1",
            bearoff_exact=os.environ.get("JANOWSKI_NO_BEAROFF", "0") != "1",
            board_sanity=board_sanity_from_env("JANOWSKI_BOARD_SANITY", True),
        )
    if c_api:
        return _load_c_agent(spec, oneply=oneply, twoply_k=twoply_k,
                             backend=c_api_backend, bf16=bf16,
                             board_sanity=board_sanity_from_env(
                                 "PROB5_BOARD_SANITY", False))
    # Plain prob5: PROB5_BOARD_SANITY=1 opts in to prob5_sanity (npz-recorded).
    return _load_agent(spec, oneply=oneply, twoply_k=twoply_k, bf16=bf16,
                       board_sanity=board_sanity_from_env(
                           "PROB5_BOARD_SANITY", False))


def _die_with_parent():
    """Ask the kernel to SIGKILL this worker when its parent dies
    (spawned workers sit in their own process group, so killing the
    parent never reaches them)."""
    if not sys.platform.startswith("linux"):
        return                      # pdeathsig is Linux-only
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)
    except Exception:
        pass                        # best-effort


def _seed_worker_rng():
    """Pool initializer: seed this worker's RNG from OS entropy, so dice
    streams stay independent under 'fork' or a deterministic upstream
    seed (combine_head_to_head pooling relies on it)."""
    _die_with_parent()
    random.seed(int.from_bytes(os.urandom(16), "little"))


def _play_matches_worker(args):
    """Pool worker: build both sides and play `num_games` games."""
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch as _torch
    _torch.set_num_threads(1)
    (model1_path, model2_path, num_games, model1_type, model2_type, plies,
     sage_plies, cubeless_money, cubeful, jacoby, oneply1, oneply2,
     twoply_k1, twoply_k2, extreme_threshold, bf16, paired,
     cube_cap, c_api, c_api_backend) = args

    # Each worker opens its own C engines (never shared across processes).
    agent1 = _build_side(
        model1_path, model1_type, plies=plies, sage_plies=sage_plies,
        cubeless_money=cubeless_money, cubeful=cubeful,
        oneply=oneply1, twoply_k=twoply_k1, bf16=bf16,
        c_api=c_api, c_api_backend=c_api_backend,
    )
    agent2 = _build_side(
        model2_path, model2_type, plies=plies, sage_plies=sage_plies,
        cubeless_money=cubeless_money, cubeful=cubeful,
        oneply=oneply2, twoply_k=twoply_k2, bf16=bf16,
        c_api=c_api, c_api_backend=c_api_backend,
    )

    a1_wins = 0
    a2_wins = 0
    a1_equity = 0
    per_game = []
    extremes = []
    capture_extremes = extreme_threshold > 0 and cubeful
    if paired is None:
        for _ in range(num_games):
            signed, rec = _play_one_signed(
                agent1, agent2, cubeful, jacoby, record=capture_extremes,
                cube_cap=cube_cap,
            )
            if signed > 0:
                a1_wins += 1
            else:
                a2_wins += 1
            a1_equity += signed
            per_game.append(signed)
            if capture_extremes and abs(signed) >= extreme_threshold:
                extremes.append((signed, rec))
        return a1_wins, a2_wins, a1_equity, per_game, extremes, None

    # Paired dice: games come in pairs (2p, 2p+1) that share one dice
    # stream (pair seed = f(run base, global pair index)). agent1 sits
    # WHITE in the first game and BLACK in the second, so the opening
    # roll and every later roll k go to the other net in the mirror
    # game. Each game is a fair game on its own.
    seed_base, pair_start = paired
    assert num_games % 2 == 0, num_games
    diag = {"pair_id": [], "a1_is_white": [], "n_rolls": [],
            "cube_value": [], "ended_by_drop": []}
    for p in range(num_games // 2):
        pid = pair_start + p
        seed = pair_seed(seed_base, pid)
        for a1_white in (True, False):
            stream = DiceStream(seed)
            st = {}
            signed, rec = _play_one_signed(
                agent1, agent2, cubeful, jacoby, record=capture_extremes,
                a1_is_white=a1_white, dice_rng=stream, stats=st,
                cube_cap=cube_cap,
            )
            if signed > 0:
                a1_wins += 1
            else:
                a2_wins += 1
            a1_equity += signed
            per_game.append(signed)
            diag["pair_id"].append(pid)
            diag["a1_is_white"].append(a1_white)
            diag["n_rolls"].append(stream.n_dice // 2)
            diag["cube_value"].append(st["cube_value"])
            diag["ended_by_drop"].append(st["ended_by_drop"])
            if capture_extremes and abs(signed) >= extreme_threshold:
                extremes.append((signed, rec))
    return a1_wins, a2_wins, a1_equity, per_game, extremes, diag


def play_matches(
    agent1: Agent,
    agent2: Agent,
    num_games: int,
    print_progress: bool = True,
    record: bool = False,
    cubeful: bool = False,
    jacoby: bool = True,
    a1_is_white: bool = None,
):
    """Play agent1 vs agent2 for num_games in this process. Colors are
    random unless `a1_is_white` is given.

    Returns (agent1_wins, agent2_wins, agent1_equity, per_game), plus
    `records` when record=True. `agent1_equity` is the signed sum of
    per-game stakes from agent1's side:
      - cubeless: stake = result (1/2/3).
      - cubeful:  stake = cube_value × result, or cube_value on a drop.
    """
    agent1_wins = 0
    agent2_wins = 0
    agent1_equity = 0
    per_game = []
    records = [] if record else None
    start_time = time.time()
    progress_interval = max(100, num_games // 10)

    for i in range(1, num_games + 1):
        signed, game_record = _play_one_signed(
            agent1, agent2, cubeful, jacoby, record=record,
            a1_is_white=a1_is_white,
        )
        if record and game_record is not None:
            records.append(game_record)
        if signed > 0:
            agent1_wins += 1
        else:
            agent2_wins += 1
        agent1_equity += signed
        per_game.append(signed)

        if print_progress and i % progress_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"  Game {i:>5d}/{num_games}: "
                f"Model1 {agent1_wins:>4d}  Model2 {agent2_wins:>4d}  "
                f"({elapsed:.1f}s)",
                flush=True,  # visible in piped logs
            )

    if record:
        return agent1_wins, agent2_wins, agent1_equity, per_game, records
    return agent1_wins, agent2_wins, agent1_equity, per_game


def play_matches_parallel(
    model1_path: str,
    model2_path: str,
    num_games: int,
    workers: int,
    model2_type: str = "model",
    model1_type: str = "model",
    plies: int = 0,
    sage_plies: int = 1,
    cubeless_money: bool = False,
    cubeful: bool = False,
    jacoby: bool = True,
    oneply1: bool = False,
    oneply2: bool = False,
    twoply_k1: int = 0,
    twoply_k2: int = 0,
    extreme_threshold: float = 0.0,
    bf16: bool = False,
    cube_cap: int = None,
    c_api: bool = False,
    c_api_backend: str = None,
):
    """Play games across `workers` spawned processes.

    Returns (a1_wins, a2_wins, eq_sum, per_game, extremes); `extremes`
    is a list of (signed_equity, GameRecord) for cubeful games with
    |equity| >= extreme_threshold (empty when the threshold is 0).

    Either side may be an engine: `model1_type` / `model2_type` as in
    _build_side. `plies` is gnubg's depth; `sage_plies` is Open Sage's
    ply (1 = 0-ply, 2 = 1-ply). `cube_cap`: see :func:`play_game`.
    """
    return _play_matches_parallel_ex(
        model1_path, model2_path, num_games, workers,
        model2_type=model2_type, model1_type=model1_type, plies=plies,
        sage_plies=sage_plies, cubeless_money=cubeless_money,
        cubeful=cubeful, jacoby=jacoby, oneply1=oneply1, oneply2=oneply2,
        twoply_k1=twoply_k1, twoply_k2=twoply_k2,
        extreme_threshold=extreme_threshold, bf16=bf16, cube_cap=cube_cap,
        c_api=c_api, c_api_backend=c_api_backend,
    )[:5]


def _play_matches_parallel_ex(
    model1_path, model2_path, num_games, workers, *,
    model2_type="model", model1_type="model", plies=0, sage_plies=1,
    cubeless_money=False, cubeful=False, jacoby=True, oneply1=False,
    oneply2=False, twoply_k1=0, twoply_k2=0, extreme_threshold=0.0,
    bf16=False, paired_dice=False, cube_cap=None, c_api=False,
    c_api_backend=None,
):
    """:func:`play_matches_parallel` plus a 6th element: ``None`` for
    independent games, or, with ``paired_dice``, a dict of per-game
    pairing diagnostics (``seed_base``, ``pair_id``, ``a1_is_white``,
    ``n_rolls``, ``cube_value``, ``ended_by_drop``) aligned with
    ``per_game``, where games ``2p`` and ``2p+1`` form pair ``p``.

    Paired dice needs an even ``num_games``. Pairs are split across
    workers (each worker gets a contiguous range of global pair indices)
    and every worker's games are appended in pair order.
    """
    if paired_dice:
        if num_games % 2:
            raise ValueError(
                f"--paired-dice plays whole pairs: --games must be even "
                f"(got {num_games})")
        # High-entropy per-run base: distinct pairs across chunks, workers
        # and machines even when the same command is replayed everywhere.
        seed_base = os.urandom(16).hex()
        n_pairs = num_games // 2
        base_p = n_pairs // workers
        rem_p = n_pairs % workers
        pair_counts = [base_p + (1 if i < rem_p else 0) for i in range(workers)]
        game_counts = [2 * n for n in pair_counts]
        starts, acc = [], 0
        for n in pair_counts:
            starts.append(acc)
            acc += n
        paired_args = [(seed_base, st) for st in starts]
    else:
        # Split games across workers
        base = num_games // workers
        remainder = num_games % workers
        game_counts = [base + (1 if i < remainder else 0) for i in range(workers)]
        paired_args = [None] * workers

    args_list = [
        (model1_path, model2_path, n, model1_type, model2_type, plies,
         sage_plies, cubeless_money, cubeful, jacoby, oneply1, oneply2,
         twoply_k1, twoply_k2, extreme_threshold, bf16, pa, cube_cap,
         c_api, c_api_backend)
        for n, pa in zip(game_counts, paired_args)
    ]

    if cubeful and "gnubg-nn" in (model1_type, model2_type):
        from third_party_opponents.gnubg.gnubg_nn_agent import warn_gnubg_nn_cubeful
        warn_gnubg_nn_cubeful()     # here, not once per worker
    ctx = mp.get_context('spawn')
    start_time = time.time()
    with ctx.Pool(workers, initializer=_seed_worker_rng) as pool:
        results = pool.map(_play_matches_worker, args_list)
        # close+join so workers exit normally and atexit runs (gnubg
        # close). The with-block alone would terminate() and skip it.
        pool.close()
        pool.join()

    elapsed = time.time() - start_time
    total_a1 = sum(r[0] for r in results)
    total_a2 = sum(r[1] for r in results)
    total_eq = sum(r[2] for r in results)

    all_per_game = []
    for r in results:
        all_per_game.extend(r[3])
    extremes = []
    for r in results:
        extremes.extend(r[4])
    print(f"  {num_games} games completed in {elapsed:.1f}s "
          f"({num_games/elapsed:.0f} games/sec, {workers} workers"
          f"{', paired dice' if paired_dice else ''})")

    diag = None
    if paired_dice:
        diag = {"seed_base": seed_base}
        for key in ("pair_id", "a1_is_white", "n_rolls", "cube_value",
                    "ended_by_drop"):
            diag[key] = [v for r in results for v in r[5][key]]
    return total_a1, total_a2, total_eq, all_per_game, extremes, diag


def compute_binomial_pvalue(wins: int, trials: int) -> float:
    """Two-sided binomial p-value against p = 0.5. Uses
    scipy.stats.binomtest, else the normal approximation."""
    try:
        from scipy.stats import binomtest
        result = binomtest(wins, trials, 0.5, alternative='two-sided')
        return result.pvalue
    except (ImportError, AttributeError):
        # z = (wins/n - 0.5) / sqrt(0.25/n)
        p_hat = wins / trials
        z = (p_hat - 0.5) / math.sqrt(0.25 / trials)

        from math import erfc
        pvalue = erfc(abs(z) / math.sqrt(2))
        return pvalue


def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for the win rate: (lower, upper) as
    proportions in [0, 1]. z = 1.96 for 95%."""
    p_hat = wins / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    half_width = (z / denom) * math.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))
    return center - half_width, center + half_width


def get_verdict(wins: int, total: int) -> str:
    """Verdict from the Wilson 95% CI: "significantly stronger" when the
    CI excludes 50%; "effectively tied" when it lies within [49%, 51%]
    and is narrower than 2%; otherwise "Inconclusive"."""
    lower, upper = wilson_ci(wins, total, z=1.96)
    ci_width = upper - lower

    if lower > 0.50:
        return "Model1 is significantly stronger (p<0.05)"

    if upper < 0.50:
        return "Model2 is significantly stronger (p<0.05)"

    # CI includes 50%: "tied" only when tight and centered.
    if ci_width < 0.02 and lower >= 0.49 and upper <= 0.51:
        return "Models are effectively tied (difference < 1%)"

    return "Inconclusive — need more games to determine winner"


class _OnPairs:
    """Picklable adapter for the pair (cluster) bootstrap. The resampler
    draws whole rows of an (n_pairs, 2) array; the wrapped per-game
    statistic then sees the games of the drawn pairs as a flat vector."""

    def __init__(self, statistic):
        self.statistic = statistic

    def __call__(self, rows):
        import numpy as np
        return self.statistic(np.asarray(rows).ravel())


def _paired_win_shift(rows):
    """Win-rate statistic for the pair bootstrap, shifted by the 0.5 null
    so the generic bootstrap p-value (relative to 0) applies."""
    import numpy as np
    return float(np.asarray(rows).ravel().mean()) - 0.5


def paired_variance_ratio(pairs, transform=None):
    """How many independent games one dice-pair is worth, for the mean of
    ``transform(per_game)`` (identity by default).

    Returns ``(games_per_pair, factor)`` where ``games_per_pair`` =
    Var(game) / Var(pair mean) (a pair of independent games gives 2) and
    ``factor`` = Var(mean of 2 independent games) / Var(pair mean) =
    ``games_per_pair / 2`` (1.0 = no gain). Both from sample variances
    (ddof=1). The marginal per-game variance is unbiased for the
    independent-game variance: every game is a fair game on its own and
    only the within-pair correlation differs.
    """
    import numpy as np
    x = np.asarray(pairs, dtype=np.float64)
    if transform is not None:
        x = transform(x)
    if len(x) < 2:
        return float("nan"), float("nan")
    s2_game = float(x.ravel().var(ddof=1))
    s2_pair = float(x.mean(axis=1).var(ddof=1))
    if s2_pair <= 0:
        return float("inf"), float("inf")
    gpp = s2_game / s2_pair
    return gpp, gpp / 2.0


def report_head_to_head(
    agent1_wins, agent2_wins, agent1_equity, per_game,
    *, cubeless_money=False, cubeful=False, n_bootstrap=10000, do_print=True,
    pairs=None,
):
    """Summarize a head-to-head: returns a dict of every reported
    statistic and prints the RESULTS block when ``do_print``.

    The one H-H reporter: both ``main()`` and ``run_head_to_head()``
    call it.

    ``pairs`` (paired-dice runs): ``per_game`` reshaped to ``(n_pairs, 2)``.
    Every CI and p-value is then a bootstrap over PAIRS (the games of a
    pair are correlated, so the game-level Wilson / binomial numbers
    would be wrong), and the result gains ``paired=True``, ``n_pairs``
    and ``paired_gain`` = {statistic: (games_per_pair, factor)} from
    :func:`paired_variance_ratio`.
    """
    import numpy as np
    from functools import partial
    from stats import bootstrap_ci_statistic
    total_games = agent1_wins + agent2_wins
    agent1_pct = 100 * agent1_wins / total_games
    agent2_pct = 100 * agent2_wins / total_games

    paired = pairs is not None
    if paired:
        pairs = np.asarray(pairs, dtype=np.float64)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.size != total_games:
            raise ValueError(f"pairs must be (n_pairs, 2) over all "
                             f"{total_games} games; got {pairs.shape}")

    if paired:
        # Pair bootstrap on the win indicator; verdict from the CI.
        w_lo, w_hi, pvalue = bootstrap_ci_statistic(
            (pairs > 0).astype(np.float64), _paired_win_shift,
            n_boot=n_bootstrap, seed=42,
        )
        lower_ci, upper_ci = w_lo + 0.5, w_hi + 0.5
        if lower_ci > 0.50:
            verdict = "Model1 is significantly stronger (p<0.05)"
        elif upper_ci < 0.50:
            verdict = "Model2 is significantly stronger (p<0.05)"
        elif upper_ci - lower_ci < 0.02 and lower_ci >= 0.49 and upper_ci <= 0.51:
            verdict = "Models are effectively tied (difference < 1%)"
        else:
            verdict = "Inconclusive — need more games to determine winner"
    else:
        pvalue = compute_binomial_pvalue(agent1_wins, total_games)
        lower_ci, upper_ci = wilson_ci(agent1_wins, total_games, z=1.96)
        verdict = get_verdict(agent1_wins, total_games)

    result = {
        "agent1_wins": agent1_wins, "agent2_wins": agent2_wins,
        "total_games": total_games,
        "agent1_pct": agent1_pct, "agent2_pct": agent2_pct,
        "win_ci_pct": (lower_ci * 100, upper_ci * 100),
        "win_pvalue": pvalue, "verdict": verdict,
        "cubeless_money": cubeless_money, "cubeful": cubeful,
        "paired": paired,
    }
    if paired:
        result["n_pairs"] = len(pairs)
        result["paired_gain"] = {
            "win_rate": paired_variance_ratio(pairs, lambda x: (x > 0) * 1.0),
        }

    if do_print:
        print()
        print("=" * 60)
        print("RESULTS" + (f"  (paired dice: {len(pairs)} pairs, "
                           f"CIs bootstrap over pairs)" if paired else ""))
        print("=" * 60)
        print(f"Model1: {agent1_wins} wins ({agent1_pct:.1f}%)")
        print(f"Model2: {agent2_wins} wins ({agent2_pct:.1f}%)")
        print()
        ci_kind = "pair bootstrap" if paired else "Wilson"
        print(f"Model1 win rate 95% CI ({ci_kind}): "
              f"[{lower_ci*100:.1f}%, {upper_ci*100:.1f}%]")
        print(f"p-value: {pvalue:.4f}")
        print(f"Verdict: {verdict}")

    if cubeless_money or cubeful:
        from concurrent.futures import ProcessPoolExecutor
        from stats import (
            bootstrap_ci, trimmed_mean, capped_mean,
        )
        eq_per_game = agent1_equity / total_games
        tag = "Cubeful money" if cubeful else "Cubeless money"
        result["eq_per_game_pts"] = eq_per_game
        result["eq_mEq_per_game"] = eq_per_game * 1000
        if do_print:
            print()
            # This line is the UNCAPPED mean. Capped/trimmed print below.
            print(f"{tag} equity (Model1), UNCAPPED mean: "
                  f"{eq_per_game:+.4f} pts/game ({eq_per_game*1000:+.1f} mEq/game)")

        if paired:
            # Cluster bootstrap: resample pairs, evaluate the per-game
            # statistic on the games of the drawn pairs.
            def _boot(stat, executor):
                return bootstrap_ci_statistic(
                    pairs, _OnPairs(stat), n_boot=n_bootstrap,
                    executor=executor)
        else:
            def _boot(stat, executor):
                return bootstrap_ci_statistic(
                    per_game, stat, n_boot=n_bootstrap, executor=executor)

        # One ProcessPool for all the bootstrap calls.
        n_jobs = min(os.cpu_count() or 1, 32)
        with ProcessPoolExecutor(max_workers=n_jobs) as boot_pool:
            ci_lo, ci_hi, pval = _boot(np.mean, boot_pool)
            ci_half = (ci_hi - ci_lo) / 2
            if ci_lo > 0:
                eq_verdict = "Model1 significantly stronger (p<0.05)"
            elif ci_hi < 0:
                eq_verdict = "Model2 significantly stronger (p<0.05)"
            else:
                eq_verdict = "No significant equity difference"
            result["equity_ci_mEq"] = (ci_lo * 1000, ci_hi * 1000)
            result["equity_pvalue"] = pval
            result["equity_verdict"] = eq_verdict
            if do_print:
                print(f"  95% CI (bootstrap): [{ci_lo*1000:+.1f}, {ci_hi*1000:+.1f}] mEq/game "
                      f"(±{ci_half*1000:.1f})")
                print(f"  Equity p-value (bootstrap): {pval:.4f}")
                print(f"  Equity verdict: {eq_verdict}")

            # Robust estimators for the heavy tails cube escalation creates.
            per_game_arr = np.asarray(per_game, dtype=np.float64)

            med = float(np.median(per_game_arr))
            m_lo, m_hi, m_p = _boot(np.median, boot_pool)
            result["median_mEq"] = med * 1000
            result["median_ci_mEq"] = (m_lo * 1000, m_hi * 1000)
            result["median_pvalue"] = m_p
            if do_print:
                print(f"  Median equity:   {med*1000:+.1f} mEq/game  "
                      f"[{m_lo*1000:+.1f}, {m_hi*1000:+.1f}]  p={m_p:.4f}")

            tmean = trimmed_mean(per_game_arr, trim=0.01)
            t_lo, t_hi, t_p = _boot(partial(trimmed_mean, trim=0.01), boot_pool)
            result["trimmed_mean_mEq"] = tmean * 1000
            result["trimmed_ci_mEq"] = (t_lo * 1000, t_hi * 1000)
            result["trimmed_pvalue"] = t_p
            if do_print:
                print(f"  Trimmed mean (1%): {tmean*1000:+.1f} mEq/game  "
                      f"[{t_lo*1000:+.1f}, {t_hi*1000:+.1f}]  p={t_p:.4f}")

            if cubeful:
                # Capped (Winsorized) mean: equity under a per-game point cap.
                cap_pts = 128.0
                cmean = capped_mean(per_game_arr, cap=cap_pts)
                c_lo, c_hi, c_p = _boot(partial(capped_mean, cap=cap_pts), boot_pool)
                result["capped_mean_mEq"] = cmean * 1000
                result["capped_ci_mEq"] = (c_lo * 1000, c_hi * 1000)
                result["capped_pvalue"] = c_p
                if do_print:
                    print(f"  Capped mean (±{int(cap_pts)}): {cmean*1000:+.1f} mEq/game  "
                          f"[{c_lo*1000:+.1f}, {c_hi*1000:+.1f}]  p={c_p:.4f}")

        if paired:
            # Diagnostics: how much the pairing bought vs independent games.
            gain = result["paired_gain"]
            gain["equity"] = paired_variance_ratio(pairs)
            if cubeful:
                gain["capped_mean"] = paired_variance_ratio(
                    pairs, partial(np.clip, a_min=-cap_pts, a_max=cap_pts))
            if do_print:
                print()
                print("  Paired-dice gain (sample variances; 1.00 = no gain):")
                for name, (gpp, factor) in gain.items():
                    print(f"    {name:<12s} one pair ~ {gpp:5.2f} independent "
                          f"games  (variance factor {factor:.2f})")
    if do_print:
        print("=" * 60)
    return result


def _reject_legacy_gnubg(spec):
    """Reject the bare spec 'gnubg': the caller must name the engine
    (gnubg-nn results must never be labeled 'gnubg')."""
    if spec == "gnubg":
        raise ValueError(
            "spec 'gnubg' is retired: it silently selected gnubg-nn, the "
            "fast CUBE-BLIND approximation of gnubg. Say which one you "
            "mean:\n"
            "  'gnubg-nn'  -- the gnubg-nn approximation (H-H sanity "
            "checks ONLY; never label its results 'gnubg')\n"
            "  'gnubg-ext' / 'gnubg-ext-full' / 'gnubg-ext-rootfull' -- "
            "real gnubg 1.08 (native money cube, Jacoby)")


def _detect_side_type(spec):
    """Side spec -> type. ``hedgehog:<model.ogxf>`` is "hedgehog",
    ``janowski:<prob5.pt>`` "janowski", ``janowski-c:<prob5.pt>``
    "janowski-c", ``bo:<cubeful.pt>`` "bo".
    Otherwise an existing file is always "model" (even one named like an
    engine). Engine keywords map to themselves. Anything else is
    "model". Pass model*_type explicitly to override. Bare 'gnubg' is
    rejected (_reject_legacy_gnubg)."""
    if spec.startswith("hedgehog:"):
        return "hedgehog"
    if spec.startswith("janowski:"):
        return "janowski"
    if spec.startswith("janowski-c:"):
        return "janowski-c"
    if spec.startswith("bo:"):
        return "bo"
    if os.path.exists(spec):
        return "model"
    _reject_legacy_gnubg(spec)
    return {
        "random": "random", "gnubg-nn": "gnubg-nn", "opensage": "opensage",
        "gnubg-ext": "gnubg-ext", "gnubg-ext-full": "gnubg-ext-full",
        "gnubg-ext-rootfull": "gnubg-ext-rootfull",
        "opensage-analyzer": "opensage-analyzer",
        "opensage-native": "opensage-native",
    }.get(spec, "model")


def side_model_md5(spec) -> str:
    """md5 of the checkpoint behind a side spec, or ``""`` for non-file
    sides (promotions overwrite checkpoints in place). ``<kind>:<path>``
    specs resolve to the path after the colon."""
    import hashlib
    if not isinstance(spec, str):
        return ""
    path = spec
    if not os.path.isfile(path) and ":" in path:
        path = path.split(":", 1)[1]
    if not os.path.isfile(path):
        return ""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _side_depth_desc(spec_type, *, oneply, twoply_k, plies, sage_plies,
                     c_api=False):
    """Search-depth identity of one side, for the pooling guard (the side
    string does not encode depth, so 0-ply and 1-ply runs of one side
    would pool silently). A 1-ply C API checkpoint side
    with PURETD_ROOT_FILTER set reads "1ply root-filter T"."""
    if spec_type in ("gnubg", "gnubg-nn", "gnubg-ext", "gnubg-ext-full",
                     "gnubg-ext-rootfull"):
        return f"plies{plies}"
    if spec_type in ("opensage", "opensage-analyzer", "opensage-native"):
        return f"sage{sage_plies}"
    if spec_type == "hedgehog":
        return f"depth{2 if twoply_k else (1 if oneply else 0)}"
    if spec_type == "model" and c_api and oneply and not twoply_k:
        from c_api.puretd_py import root_filter_from_env, search_desc
        return " ".join(filter(None, ("1ply", search_desc(root_filter_from_env()))))
    if spec_type in ("model", "janowski", "janowski-c", "bo"):
        return (f"twoply-k{twoply_k}" if twoply_k
                else ("1ply" if oneply else "0ply"))
    return ""


def save_head_to_head(out_dir, per_game, *, game_mode, side1, side2, n_games,
                      paired=None, side1_depth="", side2_depth="",
                      cube_cap=None, jacoby=None, bf16=None, c_api_prov=None):
    """Save one run's raw per-game equities and metadata to ``out_dir``
    as a unique ``.npz``, so runs (from any machine) can be pooled later
    by combine_head_to_head. ``per_game[i] > 0`` means side1 won game i.
    The magnitude is the stake.

    ``paired`` (the diag dict from :func:`_play_matches_parallel_ex`) marks a
    paired-dice run: adds ``paired_dice=True``, the run's ``pair_seed_base``
    and per-game ``pair_id`` / ``a1_is_white`` / ``n_rolls`` / ``cube_value``
    / ``ended_by_drop`` arrays (games 2p and 2p+1 form pair p). Unpaired runs
    write only the base fields.

    ``side*_depth`` (:func:`_side_depth_desc`), ``cube_cap`` ("" =
    uncapped), ``jacoby``, ``bf16`` and, with a ``janowski:``/``bo:`` side,
    the JANOWSKI_* settings and ``janowski_rev`` are pooling identity:
    combine refuses runs that differ. ``prob5_board_sanity`` ("1" only
    when a plain prob5 side ran with PROB5_BOARD_SANITY=1) and, with a
    ``janowski:`` side, ``janowski_board_sanity`` are pooling identity
    too. ``c_api_prov`` (:func:`_c_api_provenance`): ``c_api`` and
    ``c_api_backend`` are pooling identity, ``c_api_describe`` (host ISA)
    is not.
    """
    import platform
    import sys
    import time
    import uuid
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"hh_{int(time.time())}_{uuid.uuid4().hex[:8]}.npz")
    extra = {}
    if paired is not None:
        n = len(per_game)
        pid = np.asarray(paired["pair_id"], dtype=np.int64)
        if n % 2 or len(pid) != n or not np.array_equal(pid[0::2], pid[1::2]):
            raise ValueError("paired-dice per_game is not in (2p, 2p+1) pair layout")
        extra = dict(
            paired_dice=True,
            pair_seed_base=str(paired["seed_base"]),
            pair_id=pid,
            a1_is_white=np.asarray(paired["a1_is_white"], dtype=bool),
            n_rolls=np.asarray(paired["n_rolls"], dtype=np.int32),
            cube_value=np.asarray(paired["cube_value"], dtype=np.int32),
            ended_by_drop=np.asarray(paired["ended_by_drop"], dtype=bool),
        )
    np.savez(
        path,
        per_game=np.asarray(per_game, dtype=np.float64),
        game_mode=str(game_mode), side1=str(side1), side2=str(side2),
        n_games=int(n_games),
        # Provenance.
        argv=" ".join(sys.argv),
        sage_unpruned=os.environ.get("SAGE_UNPRUNED", "0"),
        sage_version=_sage_version(side1, side2),
        **_sage_path(side1, side2, game_mode),
        janowski_x=os.environ.get("JANOWSKI_X", ""),
        janowski_no_fold=os.environ.get("JANOWSKI_NO_FOLD", "0"),
        janowski_no_fold_0ply=os.environ.get("JANOWSKI_NO_FOLD_0PLY", "0"),
        janowski_cubeless_checker=os.environ.get(
            "JANOWSKI_CUBELESS_CHECKER", "0"),
        janowski_no_bearoff=os.environ.get("JANOWSKI_NO_BEAROFF", "0"),
        # Model identity (promotions overwrite paths in place);
        # combine_head_to_head refuses mismatched md5s.
        side1_md5=side_model_md5(side1),
        side2_md5=side_model_md5(side2),
        side1_depth=str(side1_depth),
        side2_depth=str(side2_depth),
        cube_cap=str(cube_cap) if cube_cap else "",
        jacoby="" if jacoby is None else str(bool(jacoby)),
        bf16="" if bf16 is None else str(bool(bf16)),
        janowski_rev=_janowski_rev(side1, side2),
        **_board_sanity_provenance(side1, side2),
        # gnubg-ext raises GnubgReplyError rather than falling back: a run
        # with a gnubg-ext side has no fallbacks.
        **({"gnubg_fallbacks": "0"}
           if any(str(x).startswith("gnubg-ext") for x in (side1, side2))
           else {}),
        hostname=platform.node(),
        **_hh_shim_provenance(side1, side2),
        **(c_api_prov or {}),
        **extra,
    )
    return path


def _hh_shim_provenance(*sides) -> dict:
    """{"hh_shim_md5": md5 of the resolved libhhagent.so} with a hedgehog:
    side ("" when the shim is missing), else {}. Provenance only."""
    if not any(str(s).startswith("hedgehog:") for s in sides):
        return {}
    from third_party_opponents.hedgehog.hh_lib import lib_path
    p = lib_path()
    return {"hh_shim_md5": side_model_md5(p) if p else ""}


def _c_api_provenance(c_api, backend, *sides) -> dict:
    """npz fields of the play path: ``c_api`` "1"/"0"; with it or a janowski-c:
    side, the resolved backend and the engine's describe string (ISA) for
    the first checkpoint side (janowski sides: the path after the colon).
    ``sides``: (spec, spec_type) pairs."""
    if not c_api and not any(t == "janowski-c" for _, t in sides):
        return {"c_api": "0"}
    from c_api.puretd_py import Engine, play_backend
    b = play_backend(backend)
    model = next((s.split(":", 1)[1] if t != "model" else s for s, t in sides
                  if t in ("model", "janowski", "janowski-c")), None)
    return {"c_api": "1" if c_api else "0", "c_api_backend": b,
            "c_api_describe": Engine(model, b).describe if model else ""}


def _sage_version(*sides) -> str:
    """Installed bgsage version with an opensage* side, else ""."""
    if not any(str(s).startswith("opensage") for s in sides):
        return ""
    from importlib.metadata import version
    return version("bgsage")


def _sage_path(side1, side2, game_mode) -> dict:
    """{"sage_path": ...} with an ``opensage`` side, else {}: "analyzer"
    (cubeless money, BgBotAnalyzer) or "best_move_index" (DMP)."""
    if "opensage" not in (str(side1), str(side2)):
        return {}
    if game_mode == "cubeless-money":
        from third_party_opponents.opensage.opensage_agents import SAGE_CUBELESS_PATH
        return {"sage_path": SAGE_CUBELESS_PATH}
    return {"sage_path": "best_move_index"}


def _janowski_rev(*sides) -> str:
    """Play revision of the janowski:/bo: agents ("" for other sides)."""
    if not any(str(s).startswith(("janowski:", "janowski-c:", "bo:")) for s in sides):
        return ""
    from janowski_agent import JANOWSKI_REV
    return JANOWSKI_REV


def _board_sanity_provenance(*sides) -> dict:
    """prob5_board_sanity: "1" iff a plain prob5 side ran with
    PROB5_BOARD_SANITY=1; janowski_board_sanity with a janowski: side."""
    on = (board_sanity_from_env("PROB5_BOARD_SANITY", False)
          and any(_is_prob5_checkpoint(str(s)) for s in sides))
    out = {"prob5_board_sanity": "1" if on else "0"}
    if any(str(s).startswith(("janowski:", "janowski-c:")) for s in sides):
        out["janowski_board_sanity"] = "1" if board_sanity_from_env(
            "JANOWSKI_BOARD_SANITY", True) else "0"
    return out


# janowski_variant: recorded by runs below _JANOWSKI_1PLY_REV_END only.
_JANOWSKI_ENV = ("janowski_x", "janowski_no_fold", "janowski_no_fold_0ply",
                 "janowski_variant", "janowski_cubeless_checker",
                 "janowski_no_bearoff")

# 1-ply settings recorded by janowski: runs at JANOWSKI_REV below
# _JANOWSKI_1PLY_REV_END; an absent field reads as its default.
_JANOWSKI_1PLY_DEFAULT = {"janowski_1ply_simp": "1",
                          "janowski_1ply_filter": "8,0.16"}
_JANOWSKI_1PLY_REV_END = "2026-10-13"


def _config_identity(d, s1, s2) -> dict:
    """Settings that must match for runs to pool. An absent jacoby/bf16
    pools freely. For janowski:/bo: matchups the JANOWSKI_* settings and
    janowski_rev count even when absent. An absent prob5_board_sanity
    reads "0", an absent janowski_board_sanity "1"."""
    out = {}
    for k in ("jacoby", "bf16"):
        v = str(d[k]) if k in d else ""
        if v:
            out[k] = v
    out["prob5_board_sanity"] = (str(d["prob5_board_sanity"])
                                 if "prob5_board_sanity" in d else "0")
    # An absent c_api reads "0" (Python agents).
    out["c_api"] = str(d["c_api"]) if "c_api" in d else "0"
    out["c_api_backend"] = str(d["c_api_backend"]) if "c_api_backend" in d else ""
    # Open Sage: an absent sage_version reads 1.3.20260723.
    if any(x.startswith("opensage") for x in (s1, s2)):
        out["sage_version"] = (str(d["sage_version"]) if "sage_version" in d
                               else "1.3.20260723")
        out["sage_unpruned"] = (str(d["sage_unpruned"])
                                if "sage_unpruned" in d else "0")
    # An absent sage_path reads best_move_index.
    if "opensage" in (s1, s2):
        out["sage_path"] = (str(d["sage_path"]) if "sage_path" in d
                            else "best_move_index")
    if any(x.startswith(("janowski:", "janowski-c:", "bo:")) for x in (s1, s2)):
        out["janowski_rev"] = str(d["janowski_rev"]) if "janowski_rev" in d else ""
    if any(x.startswith(("janowski:", "janowski-c:")) for x in (s1, s2)):
        for k in _JANOWSKI_ENV:
            out[k] = str(d[k]) if k in d else ""
        out["janowski_board_sanity"] = (str(d["janowski_board_sanity"])
                                        if "janowski_board_sanity" in d else "1")
    # 1-ply-only settings: 0-ply runs pool whatever they record.
    if any(x.startswith(("janowski:", "janowski-c:")) and k in d and str(d[k]) == "1ply"
           for x, k in ((s1, "side1_depth"), (s2, "side2_depth"))) \
            and out["janowski_rev"] < _JANOWSKI_1PLY_REV_END:
        for k, dflt in _JANOWSKI_1PLY_DEFAULT.items():
            out[k] = str(d[k]) if k in d else dflt
    return out


def _matchup_key(spec):
    """Matchup identity: engine prefix + checkpoint FILENAME, not path,
    so legs from several hosts pool. The md5 guard catches a wrong
    checkpoint.
    """
    prefix, sep, rest = spec.partition(":")
    if sep and "/" in rest:
        return f"{prefix}:{os.path.basename(rest)}"
    return os.path.basename(spec) if "/" in spec else spec


def combine_head_to_head(out_dir, *, n_bootstrap=10000, do_print=True):
    """Pool every ``.npz`` in ``out_dir`` (from :func:`save_head_to_head`)
    and report the combined head-to-head.

    All runs must share game mode, matchup, model md5 per side, search
    depth per side and cube cap (runs with an empty md5 / depth pool
    freely; an absent cube_cap reads as uncapped). Returns the
    :func:`report_head_to_head` dict.
    """
    import glob
    import numpy as np
    files = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no saved H-H runs (*.npz) in {out_dir!r}")
    parts, modes, s1s, s2s = [], set(), set(), set()
    md5s = {"side1": {}, "side2": {}}  # md5 -> first file that carried it
    depths = {"side1": {}, "side2": {}}
    caps = {}                           # cube_cap ("" = uncapped) -> file
    configs = {}                        # setting -> {value: first file}
    seen_hashes = {}
    paired_flags = {}                   # file -> paired_dice
    seen_bases = {}                     # pair_seed_base -> file
    for f in files:
        d = np.load(f, allow_pickle=False)
        parts.append(d["per_game"])
        modes.add(str(d["game_mode"]))
        is_paired = bool(d["paired_dice"]) if "paired_dice" in d else False
        paired_flags[f] = is_paired
        if is_paired:
            pid = d["pair_id"]
            pg = d["per_game"]
            if len(pg) % 2 or len(pid) != len(pg) or \
                    not np.array_equal(pid[0::2], pid[1::2]):
                raise ValueError(f"{f}: paired_dice run not in pair layout")
            b = str(d["pair_seed_base"])
            if b in seen_bases:
                raise ValueError(
                    f"duplicate pair seed base: {f} replays the dice of "
                    f"{seen_bases[b]} — remove one before pooling.")
            seen_bases[b] = f
        s1s.add(_matchup_key(str(d["side1"])))
        s2s.add(_matchup_key(str(d["side2"])))
        caps.setdefault(str(d["cube_cap"]) if "cube_cap" in d else "", f)
        for k, v in _config_identity(d, str(d["side1"]), str(d["side2"])).items():
            configs.setdefault(k, {}).setdefault(v, f)
        for side in ("side1", "side2"):
            key = f"{side}_md5"
            m = str(d[key]) if key in d else ""
            if m:
                md5s[side].setdefault(m, f)
            dkey = f"{side}_depth"
            dv = str(d[dkey]) if dkey in d else ""
            if dv:
                depths[side].setdefault(dv, f)
        # Identical per-game sequences mean a shared RNG seed. Pooling
        # them would double-count games. Refuse.
        import hashlib
        h = hashlib.sha256(
            np.ascontiguousarray(d["per_game"]).tobytes()).hexdigest()
        if h in seen_hashes:
            raise ValueError(
                f"duplicate game stream: {f} is identical to "
                f"{seen_hashes[h]} — same RNG seed? Remove one before "
                f"pooling.")
        seen_hashes[h] = f
    if len(modes) > 1:
        raise ValueError(f"mixed game modes in {out_dir}: {sorted(modes)}")
    if len(set(paired_flags.values())) > 1:
        pf = sorted(os.path.basename(f) for f, v in paired_flags.items() if v)
        uf = sorted(os.path.basename(f) for f, v in paired_flags.items() if not v)
        raise ValueError(
            f"mixed paired-dice and independent runs in {out_dir}: the "
            f"bootstrap unit differs (pair vs game) — pool them separately."
            f" paired={pf[:3]}{'...' if len(pf) > 3 else ''} "
            f"independent={uf[:3]}{'...' if len(uf) > 3 else ''}")
    paired = any(paired_flags.values())
    if len(s1s) > 1 or len(s2s) > 1:
        raise ValueError(
            f"mixed matchups in {out_dir}: side1={sorted(s1s)} side2={sorted(s2s)}")
    for side in ("side1", "side2"):
        if len(md5s[side]) > 1:
            detail = "; ".join(
                f"{m[:8]} ({os.path.basename(src)})"
                for m, src in sorted(md5s[side].items())
            )
            raise ValueError(
                f"mixed {side} MODELS in {out_dir}: same path, different "
                f"checkpoints — {detail}. A promotion overwrote the model "
                f"in place; pool these separately (move one generation's "
                f".npz files to their own directory).")
    for side in ("side1", "side2"):
        if len(depths[side]) > 1:
            detail = "; ".join(
                f"{v} ({os.path.basename(src)})"
                for v, src in sorted(depths[side].items()))
            raise ValueError(
                f"mixed {side} SEARCH DEPTHS in {out_dir}: {detail}. "
                f"The side string does not encode depth — pool these "
                f"separately.")
    for k, vals in sorted(configs.items()):
        if len(vals) > 1:
            detail = "; ".join(f"{v or 'absent'} ({os.path.basename(src)})"
                               for v, src in sorted(vals.items()))
            raise ValueError(
                f"mixed {k} in {out_dir}: {detail} — different play "
                f"settings; pool these separately.")
    if len(caps) > 1:
        detail = "; ".join(
            f"{v or 'uncapped'} ({os.path.basename(src)})"
            for v, src in sorted(caps.items()))
        raise ValueError(
            f"mixed CUBE CAPS in {out_dir}: {detail}. Capped and uncapped "
            f"games are different experiments — pool these separately.")
    game_mode = modes.pop()
    pooled = np.concatenate(parts)
    cubeless_money = game_mode == "cubeless-money"
    cubeful = game_mode == "cubeful-money"
    agent1_wins = int((pooled > 0).sum())
    agent2_wins = int((pooled <= 0).sum())
    agent1_equity = float(pooled.sum())
    # Each file is in (2p, 2p+1) layout and has an even length, so the
    # concatenation keeps pairs adjacent: reshape gives one row per pair.
    pairs = pooled.reshape(-1, 2) if paired else None
    if do_print:
        print(f"Pooled {len(files)} run(s) -> {len(pooled)} games "
              f"({s1s.pop()} vs {s2s.pop()}, {game_mode}"
              f"{f', paired dice: {len(pairs)} pairs' if paired else ''})")
    return report_head_to_head(
        agent1_wins, agent2_wins, agent1_equity, pooled,
        cubeless_money=cubeless_money, cubeful=cubeful,
        n_bootstrap=n_bootstrap, do_print=do_print, pairs=pairs,
    )


def run_head_to_head(
    model1, model2, *, num_games, workers=1, game_mode="dmp", plies=0,
    sage_plies=1, jacoby=True, oneply1=False, oneply2=False,
    twoply_k1=0, twoply_k2=0, bf16=False, n_bootstrap=10000,
    model1_type=None, model2_type=None, save=None, do_print=True,
    runaway_threshold=0.0, paired_dice=False, cube_cap=None, c_api=None,
    c_api_backend=None,
):
    """Run a head-to-head between two sides and return the result dict.

    :func:`play_matches_parallel` + :func:`report_head_to_head`, with
    agents built as the CLI builds them. Either side may be a checkpoint
    path or an engine keyword, auto-detected unless ``model1_type`` /
    ``model2_type`` are given. ``game_mode`` is ``"dmp"``,
    ``"cubeless-money"`` or ``"cubeful-money"``; ``plies`` is gnubg's
    depth; ``sage_plies`` is Open Sage's ply (1 = 0-ply, 2 = 1-ply).

    ``paired_dice`` (opt-in): play the games as pairs sharing one dice
    stream with the seats swapped (variance reduction; see
    :func:`_play_matches_parallel_ex`). The report then bootstraps over
    pairs and the saved npz records the pairing, so ``--combine`` pools
    such runs by pair (and never together with independent-game runs).

    ``cube_cap=N`` (cubeful): neither side is offered a double once the
    cube reaches N. Recorded in the npz; capped and uncapped runs refuse
    to pool.

    ``c_api``: the play path of checkpoint and janowski: sides
    (:func:`resolve_c_api`): None (default) the C API (c_api/, libpuretd)
    with a Python fallback, True the C API or an error, False Python.
    ``c_api_backend``: default simd (puretd_py.play_backend).
    """
    cubeless_money = game_mode == "cubeless-money"
    cubeful = game_mode == "cubeful-money"
    if model1_type is None:
        model1_type = _detect_side_type(model1)
    if model2_type is None:
        model2_type = _detect_side_type(model2)
    preflight_side(model1, model1_type, twoply_k=twoply_k1)
    preflight_side(model2, model2_type, twoply_k=twoply_k2)
    c_api, c_api_backend = resolve_c_api(
        c_api, [(model1, model1_type, oneply1, twoply_k1),
                (model2, model2_type, oneply2, twoply_k2)],
        cubeful=cubeful, bf16=bf16, backend=c_api_backend,
        log=print if do_print else (lambda *a: None))
    (agent1_wins, agent2_wins, agent1_equity, per_game,
     extremes, diag) = _play_matches_parallel_ex(
        model1, model2, num_games, max(1, workers),
        model1_type=model1_type, model2_type=model2_type,
        plies=plies, sage_plies=sage_plies,
        cubeless_money=cubeless_money, cubeful=cubeful, jacoby=jacoby,
        oneply1=oneply1, oneply2=oneply2,
        twoply_k1=twoply_k1, twoply_k2=twoply_k2, bf16=bf16,
        extreme_threshold=runaway_threshold, paired_dice=paired_dice,
        cube_cap=cube_cap,
        c_api=c_api, c_api_backend=c_api_backend,
    )
    import numpy as np
    pairs = np.asarray(per_game, dtype=np.float64).reshape(-1, 2) \
        if paired_dice else None
    result = report_head_to_head(
        agent1_wins, agent2_wins, agent1_equity, per_game,
        cubeless_money=cubeless_money, cubeful=cubeful,
        n_bootstrap=n_bootstrap, do_print=do_print, pairs=pairs,
    )
    if save:
        path = save_head_to_head(
            save, per_game, game_mode=game_mode,
            side1=model1, side2=model2, n_games=num_games, paired=diag,
            side1_depth=_side_depth_desc(
                model1_type, oneply=oneply1, twoply_k=twoply_k1,
                plies=plies, sage_plies=sage_plies, c_api=c_api),
            side2_depth=_side_depth_desc(
                model2_type, oneply=oneply2, twoply_k=twoply_k2,
                plies=plies, sage_plies=sage_plies, c_api=c_api),
            cube_cap=cube_cap, jacoby=jacoby if cubeful else None, bf16=bf16,
            c_api_prov=_c_api_provenance(c_api, c_api_backend,
                                         (model1, model1_type),
                                         (model2, model2_type)),
        )
        if do_print:
            print(f"Saved {len(per_game)} raw per-game results -> {path}")
    if extremes:
        # Write runaway games as .mat for inspection in gnubg/XG.
        from third_party_opponents.gnubg.gnubg_eval import export_mat
        out_dir = save if save else "."
        rec_dir = os.path.join(out_dir, "runaway_games")
        os.makedirs(rec_dir, exist_ok=True)
        extremes.sort(key=lambda sr: -abs(sr[0]))
        for k, (signed, rec) in enumerate(extremes, 1):
            fn = os.path.join(
                rec_dir,
                f"runaway_{k:03d}_cube{rec.cube_value}_stake{int(abs(signed))}.mat")
            with open(fn, "w") as f:
                f.write(export_mat(rec, game_id=k, money_game=True))
        if do_print:
            print(f"Logged {len(extremes)} runaway game(s) "
                  f"(|stake| >= {runaway_threshold:g}) -> {rec_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Play two backgammon models against each other"
    )
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Path to first model (.pt file) or a side spec (as --model2)",
    )
    parser.add_argument(
        "--model2",
        type=str,
        default=None,
        help="Path to second model (.pt file), a side spec (janowski:<prob5.pt>, "
             "janowski-c:<prob5.pt> = Prob5Jan in the C API, bo:<cubeful.pt>, "
             "hedgehog:<model.ogxf>), an engine keyword, or 'random' "
             "(optional if --random is set)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to play (default 100)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Replace model2 with RandomAgent (shorthand for --model2 random)",
    )
    parser.add_argument(
        "--gnubg-nn",
        action="store_true",
        help="Replace model2 with GnubgNNAgent, the CUBE-BLIND approximation "
             "of gnubg (H-H sanity checks only; for real gnubg use a "
             "gnubg-ext side). Shorthand for --model2 gnubg-nn.",
    )
    parser.add_argument(
        "--gnubg",
        action="store_true",
        help=argparse.SUPPRESS,  # retired; errors with guidance below
    )
    parser.add_argument(
        "--plies",
        type=int,
        default=0,
        help="Gnubg evaluation depth when using --gnubg-nn or a gnubg-ext "
             "side (0=fastest, 2=strongest, default 0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )
    parser.add_argument(
        "--save-games",
        type=str,
        default=None,
        help="Directory to save games as .mat files (Jellyfish format for gnubg import)",
    )
    parser.add_argument(
        "--save-hh",
        type=str,
        default=None,
        help="Directory to save the raw per-game equity vector as .npz, so the "
             "exact bootstrap CI can be recreated or pooled across runs later "
             "(see combine_head_to_head).",
    )
    parser.add_argument(
        "--save-extreme-games",
        type=str,
        default=None,
        help="Directory to save only games whose |equity| >= --extreme-threshold "
             "(works with --workers > 1; cubeful only).",
    )
    parser.add_argument(
        "--extreme-threshold",
        type=float,
        default=128.0,
        help="Equity magnitude threshold for --save-extreme-games (default: 128 pts).",
    )
    parser.add_argument(
        "--game-mode",
        choices=["dmp", "cubeless-money", "cubeful-money"],
        default="dmp",
        help="dmp: score by win/loss. "
             "cubeless-money: money equity scoring with gammon/backgammon "
             "weighting but no doubling cube. "
             "cubeful-money: money play with doubling cube; models must "
             "be cubeful (cubeful_perspective196 encoder). Stakes are "
             "cube_value × result (or cube_value on a drop).",
    )
    parser.add_argument(
        "--jacoby",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Jacoby rule for cubeful-money. Ignored otherwise.",
    )
    parser.add_argument(
        "--oneply1",
        action="store_true",
        help="1-ply lookahead for model1 checker and cube decisions.",
    )
    parser.add_argument(
        "--oneply2",
        action="store_true",
        help="1-ply lookahead for model2 checker and cube decisions "
             "(ignored when model2 is a gnubg, Open Sage or random side).",
    )
    parser.add_argument(
        "--oneply",
        action="store_true",
        help="1-ply lookahead for BOTH models (shorthand for --oneply1 --oneply2).",
    )
    parser.add_argument(
        "--twoply-k1", type=int, default=0,
        help="2-ply checker/cube decisions for model1 with top-K 1-ply "
             "filter at the depth-2 expansion (0 = off). Works for both "
             "cubeful-money and dmp/cubeless modes. Takes precedence over "
             "--oneply1 for move selection.",
    )
    parser.add_argument(
        "--twoply-k2", type=int, default=0,
        help="2-ply checker/cube decisions for model2 with top-K 1-ply "
             "filter (0 = off; ignored when model2 is a gnubg, Open Sage or random side).",
    )
    parser.add_argument(
        "--sage-plies", type=int, default=1,
        help="Open Sage ply when --model2 opensage (1 = 0-ply/1P, "
             "2 = 1-ply/2P; default 1).",
    )
    parser.add_argument(
        "--model1-white", action="store_true",
        help="Pin model1 to WHITE every game instead of randomizing "
             "colors (for XG imports that bind one agent per color). "
             "First mover is still decided by the opening roll, so this "
             "introduces no first-move bias. Sequential play only.",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap resamples for equity CI (default: 10000).",
    )
    parser.add_argument(
        "--paired-dice", action="store_true",
        help="variance reduction: play games in pairs that share one dice "
             "stream with the seats swapped (roll k goes to whoever is on "
             "move at turn k, opening roll included). Each game is still "
             "fair; CIs bootstrap over pairs; the --save-hh npz records the "
             "pairing (paired and independent runs never pool). --games "
             "must be even. Not with --save-games / --model1-white.",
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="Use bf16 inference for prob5 (ProbAgent) models (ignored for "
             "scalar models).",
    )
    parser.add_argument(
        "--c-api", action="store_true",
        help="Require the C API play path (the default; this flag makes a "
             "configuration it cannot serve an error instead of a Python "
             "fallback).",
    )
    parser.add_argument(
        "--python-api", action="store_true",
        help="Checkpoint and janowski: sides play on the Python path "
             "(TDAgent / ProbAgent / JanowskiAgent). Default: the C API "
             "(c_api/, libpuretd, built on demand; 0/1-ply, fp32), with a "
             "NOTE and the Python path for 2-ply sides, --bf16 prob5 sides "
             "and anything else it cannot serve. C-vs-C games without a "
             "record run in C (bgc_play_game).",
    )
    parser.add_argument(
        "--c-api-backend", choices=["simd", "cblas", "torch", "ref"], default=None,
        help="Forward backend of the C API path (default: PURETD_BACKEND, "
             "else simd).",
    )
    parser.add_argument(
        "--c-api-root-filter", type=float, default=None, metavar="T",
        help="C API 1-ply sides: root move filter threshold (sets "
             "PURETD_ROOT_FILTER; 0 = full width, the default; > 0 requires "
             "the C API). Saved H-H runs record it in the side depth, so "
             "pruned and full-width runs never pool.",
    )
    args = parser.parse_args()
    if args.c_api and args.python_api:
        parser.error("--c-api and --python-api are mutually exclusive")
    if args.python_api and (args.c_api_backend or args.c_api_root_filter is not None):
        parser.error("--c-api-backend / --c-api-root-filter need the C API "
                     "(drop --python-api)")
    if args.c_api_root_filter is not None:
        if args.c_api_root_filter < 0:
            parser.error("--c-api-root-filter needs a value >= 0")
        os.environ["PURETD_ROOT_FILTER"] = repr(args.c_api_root_filter)
    if args.paired_dice:
        if args.games % 2:
            parser.error(f"--paired-dice plays whole pairs: --games must be "
                         f"even (got {args.games})")
        if args.save_games is not None or args.model1_white:
            parser.error("--paired-dice assigns the seats itself: not with "
                         "--save-games / --model1-white")
    if args.oneply:
        args.oneply1 = True
        args.oneply2 = True
    cubeless_money = (args.game_mode == "cubeless-money")
    cubeful = (args.game_mode == "cubeful-money")

    # Override model2 if shorthand flags are set
    if args.gnubg:
        parser.error(
            "--gnubg is retired (it silently selected gnubg-nn, the "
            "cube-blind approximation). Use --gnubg-nn for the "
            "approximation, or --model2 gnubg-ext for real gnubg.")
    if args.random:
        args.model2 = "random"
    elif args.gnubg_nn:
        args.model2 = "gnubg-nn"
    elif args.model2 is None:
        parser.error("--model2 is required unless --random or --gnubg-nn "
                     "is specified")
    if args.model2 == "opensage" and args.sage_plies < 1:
        parser.error("--sage-plies must be >= 1 (1 = 0-ply/1P convention)")

    # model1: same dispatch as model2, so an engine / hedgehog: side works
    # here as it does in run_head_to_head.
    model1_type = _detect_side_type(args.model1)
    model2_type = _detect_side_type(args.model2)
    try:
        preflight_side(args.model1, model1_type, twoply_k=args.twoply_k1)
        preflight_side(args.model2, model2_type, twoply_k=args.twoply_k2)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        parser.error(str(e))
    try:
        args.c_api, args.c_api_backend = resolve_c_api(
            True if args.c_api else (False if args.python_api else None),
            [(args.model1, model1_type, args.oneply1, args.twoply_k1),
             (args.model2, model2_type, args.oneply2, args.twoply_k2)],
            cubeful=cubeful, bf16=args.bf16, backend=args.c_api_backend)
    except (ValueError, RuntimeError) as e:
        parser.error(str(e))
    if model1_type == "model":
        print(f"Loading model1: {args.model1}")
        try:
            if args.c_api:
                agent1 = _load_c_agent(args.model1, oneply=args.oneply1,
                                       twoply_k=args.twoply_k1,
                                       backend=args.c_api_backend,
                                       bf16=args.bf16)
            else:
                agent1 = _load_agent(args.model1, oneply=args.oneply1,
                                     twoply_k=args.twoply_k1, bf16=args.bf16)
        except ValueError as e:
            parser.error(str(e))
    else:
        try:
            agent1 = _build_side(
                args.model1, model1_type, plies=args.plies,
                sage_plies=args.sage_plies, cubeless_money=cubeless_money,
                cubeful=cubeful, oneply=args.oneply1,
                twoply_k=args.twoply_k1, bf16=args.bf16,
                c_api=args.c_api, c_api_backend=args.c_api_backend,
            )
        except ValueError as e:
            parser.error(str(e))
    if args.twoply_k1 > 0:
        print(f"  Model1 using 2-ply lookahead (K={args.twoply_k1})")
    elif args.oneply1:
        print("  Model1 using 1-ply lookahead")

    # model2: same dispatch as the parallel workers.
    if model2_type == "model":
        print(f"Loading model2: {args.model2}")
    try:
        agent2 = _build_side(
            args.model2, model2_type, plies=args.plies,
            sage_plies=args.sage_plies, cubeless_money=cubeless_money,
            cubeful=cubeful, oneply=args.oneply2, twoply_k=args.twoply_k2,
            bf16=args.bf16, c_api=args.c_api, c_api_backend=args.c_api_backend,
        )
    except ValueError as e:
        parser.error(str(e))
    mode_tag = ", cubeless-money" if cubeless_money else ""
    model2_label = {
        "random": "RandomAgent",
        "hedgehog": (f"hedgehog ({args.model2.split(':', 1)[-1]}, depth "
                     f"{2 if args.twoply_k2 else (1 if args.oneply2 else 0)})"),
        "gnubg-nn": f"gnubg-nn ({args.plies}-ply{mode_tag}, cube-blind approx)",
        "gnubg-ext": f"gnubg-ext ({args.plies}-ply, real gnubg 1.08)",
        "gnubg-ext-full": f"gnubg-ext-full ({args.plies}-ply, unpruned)",
        "gnubg-ext-rootfull": (f"gnubg-ext-rootfull ({args.plies}-ply, "
                               "keep-all root, default reply pruning)"),
        "opensage": f"OpenSage (sage_plies={args.sage_plies})",
        "opensage-analyzer": f"OpenSage-analyzer (sage_plies={args.sage_plies})",
        "opensage-native": f"OpenSage-native (sage_plies={args.sage_plies})",
    }.get(model2_type, args.model2)
    if model2_type == "model":
        if args.twoply_k2 > 0:
            print(f"  Model2 using 2-ply lookahead (K={args.twoply_k2})")
        elif args.oneply2:
            print("  Model2 using 1-ply lookahead")

    # A janowski: side is a ProbAgent with a cube policy (is_cubeful).
    _has_prob5 = any((isinstance(a, ProbAgent) and not a.is_cubeful)
                     or getattr(a, "output_mode", None) == "prob5"
                     for a in (agent1, agent2))
    if cubeful and _has_prob5:
        parser.error("prob5 (ProbAgent) is cubeless-only and has no cube policy; "
                     "drop cubeful mode for prob5 models.")

    print(f"\nPlaying {args.games} games...")
    print(f"  Model1: {args.model1}")
    print(f"  Model2: {model2_label}")
    if args.workers > 1:
        print(f"  Workers: {args.workers}")
    print()

    # Play all games
    save_games = args.save_games is not None
    diag = None
    if args.workers > 1 or args.paired_dice:
        if save_games:
            print("WARNING: --save-games not supported with --workers > 1, "
                  "falling back to sequential play")
            result = play_matches(
                agent1, agent2, args.games, record=True,
                cubeful=cubeful, jacoby=args.jacoby,
                a1_is_white=(True if args.model1_white else None),
            )
            agent1_wins, agent2_wins, agent1_equity, per_game, records = result
        else:
            extreme_threshold = args.extreme_threshold if args.save_extreme_games else 0.0
            (agent1_wins, agent2_wins, agent1_equity,
             per_game, extremes, diag) = _play_matches_parallel_ex(
                args.model1, args.model2, args.games, max(1, args.workers),
                model1_type=model1_type,
                model2_type=model2_type, plies=args.plies,
                sage_plies=args.sage_plies,
                cubeless_money=cubeless_money,
                cubeful=cubeful, jacoby=args.jacoby,
                oneply1=args.oneply1, oneply2=args.oneply2,
                twoply_k1=args.twoply_k1, twoply_k2=args.twoply_k2,
                extreme_threshold=extreme_threshold, bf16=args.bf16,
                paired_dice=args.paired_dice,
                c_api=args.c_api, c_api_backend=args.c_api_backend,
            )

            if args.save_extreme_games and extremes:
                from third_party_opponents.gnubg.gnubg_eval import export_mat
                os.makedirs(args.save_extreme_games, exist_ok=True)
                extremes.sort(key=lambda x: -abs(x[0]))
                for i, (signed, rec) in enumerate(extremes):
                    mat = export_mat(rec, game_id=i + 1, money_game=cubeful)
                    name = f"extreme_{i+1:03d}_eq{int(signed):+d}.mat"
                    with open(os.path.join(args.save_extreme_games, name), "w") as f:
                        f.write(mat)
                print(f"Saved {len(extremes)} extreme games "
                      f"(|eq| >= {extreme_threshold}) to {args.save_extreme_games}/")
    else:
        if save_games:
            result = play_matches(
                agent1, agent2, args.games, record=True,
                cubeful=cubeful, jacoby=args.jacoby,
                a1_is_white=(True if args.model1_white else None),
            )
            agent1_wins, agent2_wins, agent1_equity, per_game, records = result
        else:
            agent1_wins, agent2_wins, agent1_equity, per_game = play_matches(
                agent1, agent2, args.games,
                cubeful=cubeful, jacoby=args.jacoby,
                a1_is_white=(True if args.model1_white else None),
            )

    # Save games as .mat files if requested
    if save_games:
        from third_party_opponents.gnubg.gnubg_eval import export_mat
        os.makedirs(args.save_games, exist_ok=True)
        for i, rec in enumerate(records):
            mat_content = export_mat(
                rec, game_id=i + 1, money_game=cubeful,
            )
            mat_path = os.path.join(args.save_games, f"game_{i+1}.mat")
            with open(mat_path, "w") as f:
                f.write(mat_content)
        print(f"Saved {len(records)} games to {args.save_games}/")

    import numpy as np
    report_head_to_head(
        agent1_wins, agent2_wins, agent1_equity, per_game,
        cubeless_money=cubeless_money, cubeful=cubeful,
        n_bootstrap=args.n_bootstrap,
        pairs=(np.asarray(per_game, dtype=np.float64).reshape(-1, 2)
               if args.paired_dice else None),
    )

    if args.save_hh:
        path = save_head_to_head(
            args.save_hh, per_game, game_mode=args.game_mode,
            side1=args.model1, side2=model2_label, n_games=len(per_game),
            paired=diag,
            side1_depth=_side_depth_desc(
                model1_type, oneply=args.oneply1, twoply_k=args.twoply_k1,
                plies=args.plies, sage_plies=args.sage_plies, c_api=args.c_api),
            side2_depth=_side_depth_desc(
                model2_type, oneply=args.oneply2, twoply_k=args.twoply_k2,
                plies=args.plies, sage_plies=args.sage_plies, c_api=args.c_api),
            jacoby=args.jacoby if cubeful else None, bf16=args.bf16,
            c_api_prov=_c_api_provenance(args.c_api, args.c_api_backend,
                                         (args.model1, model1_type),
                                         (args.model2, model2_type)),
        )
        print(f"Saved {len(per_game)} raw per-game equities -> {path}")


if __name__ == "__main__":
    main()
