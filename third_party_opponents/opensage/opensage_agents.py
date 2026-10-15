"""Open Sage (``bgsage``) opponents for head-to-head evaluation.

OPTIONAL module: everything here needs ``pip install bgsage==2.0.20260907``
(production model ``stage11``). bgsage is imported LAZILY inside the classes,
and nothing in the training codebase imports this module.

Agents (all use Sage's XG-numbered plies: ``sage_plies`` 1 = 0-ply,
2 = 1-ply):

* :class:`OpenSageAgent`: cubeless money via Sage's cubeless
  ``BgBotAnalyzer``; DMP via ``MultiPlyStrategy`` win probability.
* :class:`OpenSageAnalyzerAgent`: cubeful money via Sage's own
  ``BgBotAnalyzer``, byte-identical to its self-play generator (cube-AWARE
  checker play + native cube_action); spec keyword ``opensage-analyzer``.
"""

import os

from agents import Agent, CubeOffer
from backgammon_engine import BoardState, WHITE, BLACK

SAGE_VERSION = "2.0.20260907"
SAGE_VERSION_OVERRIDE = "OPENSAGE_ALLOW_VERSION_MISMATCH"


def installed_sage_version():
    """The installed bgsage distribution's version, or None."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("bgsage")
    except PackageNotFoundError:
        return None


def assert_sage_version() -> None:
    """Refuse an installed bgsage other than SAGE_VERSION: another version
    plays different moves while appearing to work. Override:
    OPENSAGE_ALLOW_VERSION_MISMATCH=1."""
    if os.environ.get(SAGE_VERSION_OVERRIDE) == "1":
        return
    v = installed_sage_version()
    if v is None:
        raise RuntimeError(
            f"Open Sage (bgsage) is not installed: "
            f"pip install bgsage=={SAGE_VERSION}")
    if v != SAGE_VERSION:
        raise RuntimeError(
            f"bgsage {v} is installed, but Open Sage results are pinned to "
            f"{SAGE_VERSION}: pip install bgsage=={SAGE_VERSION}, or set "
            f"{SAGE_VERSION_OVERRIDE}=1 to run another version ON PURPOSE "
            f"(results will not be comparable).")


def sage_multipy(n_plies: int, filter_max_moves: int = 5,
                 filter_threshold: float = 0.08):
    """Single-threaded MultiPlyStrategy on Open Sage's production model.

    ``n_plies`` is XG-numbered (1 = 0-ply). No bearoff database is attached
    (the analyzer attaches one; this cubeless path never has).
    """
    assert_sage_version()
    import bgbot_cpp
    from bgsage.weights import WeightConfig
    w = WeightConfig.default()
    return bgbot_cpp.create_multipy(
        w.strategy_type, w.weight_paths_list, w.hidden_sizes_list,
        n_plies=n_plies,
        filter_max_moves=filter_max_moves,
        filter_threshold=filter_threshold,
        parallel_evaluate=False,
    )


SAGE_CUBELESS_PATH = "analyzer"   # npz sage_path of cubeless-money runs


class _KeepCache:
    """The analyzer's N-ply strategy without its per-decision cache flush.

    The analyzer memsets its ~35 MB position cache after every checker
    play: ~35 ms per decision with 60 workers. Entries are
    keyed on board and ply, so a kept cache returns the values a flushed
    one would recompute; picks are unchanged.
    """

    def __init__(self, strategy):
        self._s = strategy

    def evaluate_board(self, board, pre):
        return self._s.evaluate_board(board, pre)

    def clear_cache(self):
        pass


class OpenSageAgent(Agent):
    """Open Sage (``bgsage``) as a cubeless opponent.

    ``sage_plies`` follows Open Sage's XG ply convention: ``1`` = 0-ply
    (``1P``, raw NN), ``2`` = 1-ply (``2P``, one level of lookahead).

    ``mode``:
      * ``"cubeless-money"``: Open Sage's cubeless analyzer,
        ``BgBotAnalyzer(eval_level, cubeful=False)`` with its bearoff
        database; plays the analyzer's best-ranked legal move.
      * ``"dmp"``: max win probability from ``MultiPlyStrategy`` (no bearoff
        database).
    """

    def __init__(self, sage_plies: int = 1, mode: str = "cubeless-money",
                 filter_max_moves: int = 5, filter_threshold: float = 0.08):
        if mode not in ("cubeless-money", "dmp"):
            raise ValueError(
                f"OpenSageAgent mode must be 'cubeless-money' or 'dmp', got {mode!r}")
        if sage_plies not in (1, 2):
            raise ValueError(f"sage_plies must be 1 (0-ply) or 2 (1-ply), got {sage_plies}")
        # filter_max_moves/filter_threshold default to Open Sage's production
        # move filter (5 / 0.08); 1000 / 10.0 disables root move pruning.
        assert_sage_version()
        self.sage_plies = sage_plies
        self.mode = mode
        if mode == "dmp":
            self._strat = sage_multipy(sage_plies, filter_max_moves,
                                       filter_threshold)
            return
        from bgsage import BgBotAnalyzer
        # parallel_threads=1 becomes 2 at "2ply", but the C++ search runs
        # threads only above 2 plies: one thread per agent.
        self._an = BgBotAnalyzer(
            eval_level={1: "1ply", 2: "2ply"}[sage_plies], cubeful=False,
            parallel_threads=1)
        inner = self._an._analyzer
        if hasattr(inner, "FILTER_MAX_MOVES"):
            # The multi-ply analyzer ignores the constructor's filter kwargs.
            inner.FILTER_MAX_MOVES = filter_max_moves
            inner.FILTER_THRESHOLD = filter_threshold
            inner._strategy_nply = _KeepCache(inner._strategy_nply)

    def analyzer_pick(self, pre, dice, legal):
        """Key of ``legal`` (sage board tuple -> anything) that the cubeless
        analyzer ranks highest. ``cube_owner=None`` keeps the root filter on
        cubeless equity (a cube owner switches it to Janowski equity)."""
        r = self._an.checker_play(pre, dice[0], dice[1], cube_owner=None)
        for m in r.moves:
            key = tuple(m.board)
            if key in legal:
                return key
        raise RuntimeError(
            f"OpenSage move-gen mismatch (cubeless analyzer): none of "
            f"checker_play's {len(r.moves)} ranked boards matched a legal "
            f"play (dice={tuple(dice)}, {len(legal)} legal plays)")

    @staticmethod
    def _to_sage_board(state: BoardState, mover: int) -> list:
        """BoardState -> Open Sage 26-int board from ``mover``'s perspective:
        ``[opp_bar] + 24 points (mover positive) + [mover_bar]``. This is the
        same convention ``bgsage.possible_moves`` / ``best_move_index`` use
        (post-move boards stay in the mover's frame, not flipped)."""
        pts, bar = state.points, state.bar
        if mover == WHITE:
            body = list(pts)
            mbar, obar = bar[WHITE], bar[BLACK]
        else:
            body = [-pts[23 - i] for i in range(24)]
            mbar, obar = bar[BLACK], bar[WHITE]
        return [obar] + body + [mbar]

    def choose_checker_action(self, state, dice, actions):
        mover = state.turn
        pre = self._to_sage_board(state, mover)
        if self.mode == "cubeless-money":
            if len(actions) == 1:
                return actions[0]
            # The analyzer ranks bgsage.possible_moves, which can include
            # rule-illegal plays (e.g. smaller-die-only when the larger is
            # forced): play its best-ranked move among OUR legal actions.
            legal = {}
            for a in actions:
                legal.setdefault(tuple(self._to_sage_board(a[1], mover)), a)
            return legal[self.analyzer_pick(pre, dice, legal)]
        # DMP: maximise the mover's win probability.
        best_i, best_val = 0, -1.0
        for i, (play, ns) in enumerate(actions):
            if ns.is_game_over():
                val = 1.0  # mover bore off -> wins this move
            else:
                val = float(self._strat.evaluate_board(
                    self._to_sage_board(ns, mover), pre)["probs"][0])
            if val > best_val:
                best_val, best_i = val, i
        return actions[best_i]


class OpenSageAnalyzerAgent(Agent):
    """Open Sage via its own ``BgBotAnalyzer``, in the SAME configuration its
    self-play generator (``scripts/run_sage_vs_sage_games.py``) uses.

    Checker play is CUBE-AWARE (``analyzer.checker_play`` receives
    cube_value / cube_owner). Cube decisions go through
    ``analyzer.cube_action`` (the self-play entry point).

    ``sage_plies``: 1 = 0-ply/1P (analyzer eval_level "1ply"),
    2 = 1-ply/2P ("2ply"), the same XG-numbered convention as the other Sage
    agents. ``beaver`` defaults False to match this arena (no beaver rule).
    The self-play generator itself runs beaver=True.
    """

    def __init__(self, sage_plies: int = 1, beaver: bool = False,
                 filter_max_moves: int = 5, filter_threshold: float = 0.08):
        if sage_plies not in (1, 2, 3):
            raise ValueError(
                f"sage_plies must be 1 (0-ply), 2 (1-ply) or 3 (2-ply), "
                f"got {sage_plies}")
        assert_sage_version()
        from bgsage import BgBotAnalyzer
        # Filter defaults = Sage production (5 / 0.08). 1000 / 10.0 disables
        # root move pruning. BgBotAnalyzer SILENTLY DROPS these kwargs for
        # the multi-ply eval levels (_MultiPlyAnalyzer hardwires them as
        # class attributes), so also shadow them on the inner analyzer.
        self._an = BgBotAnalyzer(
            eval_level={1: "1ply", 2: "2ply", 3: "3ply"}[sage_plies],
            cubeful=True, parallel_threads=1,
            filter_max_moves=filter_max_moves,
            filter_threshold=filter_threshold,
        )
        inner = self._an._analyzer
        inner = getattr(inner, "_inner", inner)  # unwrap _CubefulAnalyzer
        if hasattr(inner, "FILTER_MAX_MOVES"):
            inner.FILTER_MAX_MOVES = filter_max_moves
            inner.FILTER_THRESHOLD = filter_threshold
        self.sage_plies = sage_plies
        self.beaver = beaver
        self.is_cubeful = True

    @staticmethod
    def _owner_str(state, match_state) -> str:
        from modes import cube_perspective, CubePerspective
        p = cube_perspective(match_state.cube_owner, state.turn)
        return {CubePerspective.CENTERED: "centered",
                CubePerspective.MINE: "player",
                CubePerspective.THEIRS: "opponent"}[p]

    def choose_checker_action_cubeful(self, state, dice, match_state):
        """Cube-aware Open Sage checker move as ``(next_state, 0.0)``;
        ``None`` on a forced pass."""
        from backgammon_engine import get_legal_plays
        plays = get_legal_plays(state, dice)
        if not plays:
            return None
        mover = state.turn
        pre = OpenSageAgent._to_sage_board(state, mover)
        r = self._an.checker_play(
            pre, dice[0], dice[1],
            # Clamp: money decisions are per unit; the raw value overflows the
            # 32-bit binding in runaway redouble wars.
            cube_value=min(int(match_state.cube_value), 4096),
            cube_owner=self._owner_str(state, match_state),
            jacoby=match_state.jacoby, beaver=self.beaver,
        )
        # Play the analyzer's best-ranked LEGAL move. bgsage can emit
        # rule-illegal moves (e.g. smaller-die-only when the larger is
        # forced).
        legal = {tuple(OpenSageAgent._to_sage_board(ns, mover)): ns
                 for _play, ns in plays}
        for m in r.moves:
            ns = legal.get(tuple(m.board))
            if ns is not None:
                return ns, 0.0
        raise RuntimeError(
            f"OpenSage move-gen mismatch (analyzer): none of checker_play's "
            f"{len(r.moves)} ranked boards matched a legal play "
            f"(dice={tuple(dice)}, {len(plays)} legal plays)")

    def _cube_action(self, state, match_state):
        board = OpenSageAgent._to_sage_board(state, state.turn)
        return self._an.cube_action(
            board, cube_value=min(int(match_state.cube_value), 4096),
            cube_owner=self._owner_str(state, match_state),
            jacoby=match_state.jacoby, beaver=self.beaver,
        )

    def offer_double(self, state, match_state) -> CubeOffer:
        r = self._cube_action(state, match_state)
        return CubeOffer(
            should_double=bool(r.should_double),
            _cache={"should_take": bool(r.should_take)},
        )

    def respond_to_double(self, state, match_state, hint=None) -> bool:
        if hint is not None and hint._cache and "should_take" in hint._cache:
            return bool(hint._cache["should_take"])
        return bool(self._cube_action(state, match_state).should_take)
