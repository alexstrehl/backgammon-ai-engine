"""Open Sage (``bgsage``) opponents for head-to-head evaluation.

OPTIONAL module: everything here needs ``pip install bgsage``
(pinned: v0.8.20260611, commit e09bf9c) — but bgsage is imported LAZILY inside
the classes, and nothing in the training codebase imports this module. Open
Sage is a baseline opponent for reproducing the H-H tables, not a dependency
of training.

Agents (all use Sage's XG-numbered plies: ``sage_plies`` 1 = 0-ply/1P,
2 = 1-ply/2P):

* :class:`OpenSageAgent` — cubeless (money or DMP), Sage's native move pick.
* :class:`OpenSageAnalyzerAgent` — cubeful money via Sage's own
  ``BgBotAnalyzer``, byte-identical to its self-play generator (cube-AWARE
  checker play + native cube_action); spec keyword ``opensage-analyzer``.
"""

from agents import Agent, CubeOffer
from backgammon_engine import BoardState, WHITE, BLACK


class OpenSageAgent(Agent):
    """Open Sage (``bgsage``) as an opponent, using its NATIVE move selection.

    ``sage_plies`` follows Open Sage's XG ply convention: ``1`` = 0-ply
    (``1P``, raw NN), ``2`` = 1-ply (``2P``, one level of lookahead).

    ``mode``:
      * ``"cubeless-money"``: select moves by Open Sage's money equity via its
        native ``best_move_index`` (gammon/backgammon weighted, production move
        filter on) — i.e. the move Open Sage would actually play in money.
      * ``"dmp"``: select by win probability (gammons irrelevant).

    ``bgsage`` is imported lazily so this module loads without it installed.
    Requires ``pip install bgsage``.
    """

    def __init__(self, sage_plies: int = 1, mode: str = "cubeless-money",
                 filter_max_moves: int = 5, filter_threshold: float = 0.08):
        if mode not in ("cubeless-money", "dmp"):
            raise ValueError(
                f"OpenSageAgent mode must be 'cubeless-money' or 'dmp', got {mode!r}")
        if sage_plies not in (1, 2):
            raise ValueError(f"sage_plies must be 1 (0-ply) or 2 (1-ply), got {sage_plies}")
        import bgbot_cpp
        from bgsage.weights import WeightConfig
        w = WeightConfig.default()
        # Single-thread by construction. filter_max_moves/filter_threshold
        # default to Open Sage's production move filter (5 / 0.08); pass
        # e.g. 1000 / 10.0 to disable root move pruning entirely
        # (full_depth_opponent stays at the C++ default).
        self._strat = bgbot_cpp.create_multipy_stage9(
            w.weight_paths_list, w.hidden_sizes_list,
            n_plies=sage_plies,
            filter_max_moves=filter_max_moves,
            filter_threshold=filter_threshold,
            parallel_evaluate=False,
        )
        self.sage_plies = sage_plies
        self.mode = mode

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
            # Open Sage's native move = best_move_index (money equity), but
            # selected over OUR rule-legal actions' boards rather than
            # bgsage.possible_moves. The two candidate sets agree except that
            # bgsage can emit rule-ILLEGAL moves (e.g. playing only the
            # smaller die when exactly one die is playable — the rules force
            # the larger), and with the move filter widened Sage's eval can
            # prefer one, which used to trip the mismatch guard here
            # (2026-07-12, unpruned run, dice=(1,2)). Whenever Sage's native
            # pick is legal the argmax over either set is identical
            # (replay-verified: 0 divergences in 1639 decisions under the
            # production filter).
            cands = [self._to_sage_board(ns, mover) for _, ns in actions]
            return actions[self._strat.best_move_index(cands, pre)]
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
    """Open Sage via its own ``BgBotAnalyzer`` — the SAME configuration its
    self-play generator (``scripts/run_sage_vs_sage_games.py``) uses.

    Checker play is CUBE-AWARE (``analyzer.checker_play`` receives
    cube_value / cube_owner); cube decisions go through
    ``analyzer.cube_action`` (the self-play entry point).

    ``sage_plies``: 1 = 0-ply/1P (analyzer eval_level "1ply"),
    2 = 1-ply/2P ("2ply") — same XG-numbered convention as the other Sage
    agents. ``beaver`` defaults False to match this arena (no beaver rule);
    the self-play generator itself runs beaver=True.

    Cost: analyzer checker moves are ~10x Sage's lean cubeless move path
    (1.4ms vs 0.13ms at 1P) but cube decisions dominate Sage's per-game
    cost, so game throughput is within ~10-15% of the lean path.
    """

    def __init__(self, sage_plies: int = 1, beaver: bool = False,
                 filter_max_moves: int = 5, filter_threshold: float = 0.08):
        if sage_plies not in (1, 2, 3):
            raise ValueError(
                f"sage_plies must be 1 (0-ply), 2 (1-ply) or 3 (2-ply), "
                f"got {sage_plies}")
        from bgsage import BgBotAnalyzer
        # Filter defaults = Sage production (5 / 0.08); 1000 / 10.0
        # disables root move pruning. NOTE: BgBotAnalyzer accepts these
        # kwargs but SILENTLY DROPS them for the multi-ply eval levels —
        # _MultiPlyAnalyzer hardwires FILTER_MAX_MOVES=5 /
        # FILTER_THRESHOLD=0.08 as class attributes (verified 2026-07-12:
        # 53/53 checker choices identical under the ctor kwargs alone).
        # So we also shadow them with instance attributes on the actual
        # inner analyzer, which checker_play_analytics reads via self.
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
            # clamp: money decisions are per-unit; raw value overflows the
            # 32-bit binding in runaway redouble wars (2**31 seen 2026-07-23)
            cube_value=min(int(match_state.cube_value), 4096),
            cube_owner=self._owner_str(state, match_state),
            jacoby=match_state.jacoby, beaver=self.beaver,
        )
        # Walk the analyzer's ranked moves and play the best LEGAL one:
        # bgsage's candidate generation can emit rule-illegal moves (e.g.
        # smaller-die-only when the larger is forced), and with the move
        # filter widened its top pick can be one of them. Rank order
        # preserves Sage's own preference among legal moves.
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
