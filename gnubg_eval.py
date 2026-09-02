"""
gnubg_eval.py -- Evaluate backgammon self-play games using GNU Backgammon.

Workflow:
    1. Play evaluation self-play games (with the model, or random agents).
    2. Record dice rolls and moves for each game.
    3. Export the games as .mat files (Jellyfish format that gnubg imports).
    4. Run gnubg once in batch mode to analyze all games.
    5. Parse the mEMG error rates from gnubg's output.

The mEMG (millipoints Error per Move in Equivalent Money Game) is a
standard metric used by the backgammon community to measure play quality.

Platform support:
    Set GNUBG_CMD to the path of your gnubg CLI executable.
    - Windows:  typically r'C:\\Program Files (x86)\\gnubg\\gnubg-cli.exe'
                or wherever you installed it.
    - Linux:    typically '/usr/games/gnubg' (from apt-get install gnubg)

Usage:
    python gnubg_eval.py --model td_model_final.pt --games 10
"""

import multiprocessing as mp
import os
import platform
import random
import subprocess
import sys
import time as _time
import tempfile
import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from agents import RandomAgent
from backgammon_engine import (
    WHITE, BLACK, BAR, OFF,
    Play, Move, get_legal_plays, switch_turn, opening_roll,
)

# ── Platform configuration ───────────────────────────────────────────────────

# Auto-detect platform and set default gnubg path.
# Override GNUBG_CMD if your installation is in a different location.
if platform.system() == "Windows":
    # Common Windows install locations (try in order)
    _WINDOWS_PATHS = [
        r"C:\Program Files (x86)\gnubg\gnubg-cli.exe",
        r"C:\Program Files\gnubg\gnubg-cli.exe",
        r"C:\Program Files (x86)\GNU Backgammon\gnubg-cli.exe",
        r"C:\Program Files\GNU Backgammon\gnubg-cli.exe",
    ]
    GNUBG_CMD = "gnubg-cli.exe"  # fallback: assume on PATH
    for _p in _WINDOWS_PATHS:
        if os.path.isfile(_p):
            GNUBG_CMD = _p
            break
else:
    GNUBG_CMD = "/usr/games/gnubg"

# GNUBG_CMD overrides the platform default, so a non-standard gnubg path
# works for the whole pipeline (self-play + analysis).
GNUBG_CMD = os.environ.get("GNUBG_CMD", GNUBG_CMD)


# ── Judge preflight: pinned version/weights + bearoff database ───────────────

_BEAROFF_OK: dict = {}
_VERSION_OK: dict = {}

# Every published row is judged by this exact binary + net. A different
# version (older Debian/Ubuntu packages are 1.06) or a different gnubg.wd
# (gnubg >= 1.08 ships retrained nets) produces non-comparable numbers
# while everything appears to work.
GNUBG_REQUIRED_VERSION = "1.08.003"
GNUBG_REQUIRED_WEIGHTS_MD5 = "14184acc9c60ef67be0fad88548fd51a"


def _gnubg_version(cmd: str):
    """Version string of the binary at `cmd`, or None if it will not run."""
    try:
        out = subprocess.run([cmd, "--version"], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"GNU Backgammon\s+(\S+)", out or "")
    return m.group(1) if m else None


def _gnubg_weights_md5(cmd: str):
    """md5 of the gnubg.wd the binary at `cmd` will load, or None.

    gnubg will not report its pkgdatadir, so derive it the standard
    autotools way (<prefix>/bin/gnubg -> <prefix>/share/gnubg/gnubg.wd).
    None (file absent / non-standard layout) degrades to the version
    check alone rather than refusing to run."""
    prefix = os.path.dirname(os.path.dirname(os.path.realpath(cmd)))
    wd = os.path.join(prefix, "share", "gnubg", "gnubg.wd")
    if not os.path.isfile(wd):
        return None
    h = hashlib.md5()
    with open(wd, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_gnubg_pinned_judge(cmd: str) -> None:
    """Refuse to run on a gnubg other than the pinned version + weights.

    A wrong binary passes every other preflight (it exists, it runs, it
    may even have the bearoff DBs) and then produces silently different
    equities. GNUBG_ALLOW_VERSION_MISMATCH=1 opts out deliberately."""
    if cmd in _VERSION_OK:
        return
    if os.environ.get("GNUBG_ALLOW_VERSION_MISMATCH") == "1":
        _VERSION_OK[cmd] = True
        return
    v = _gnubg_version(cmd)
    if v != GNUBG_REQUIRED_VERSION:
        raise RuntimeError(
            f"gnubg at '{cmd}' is version {v!r}, but results are pinned to "
            f"{GNUBG_REQUIRED_VERSION}. Package-manager versions vary "
            f"(recent Debian/Ubuntu ship 1.08.003, older ship 1.06) -- "
            f"install or build 1.08.003 and set GNUBG_CMD, or set "
            f"GNUBG_ALLOW_VERSION_MISMATCH=1 to run on a different gnubg "
            f"ON PURPOSE (results will not be comparable)."
        )
    w = _gnubg_weights_md5(cmd)
    if w is None:
        # Non-autotools layout: the weights pin cannot be checked. The
        # pin is the point of this preflight, so refuse rather than fail
        # open on the version check alone.
        raise RuntimeError(
            f"cannot locate gnubg.wd for '{cmd}' (expected "
            f"<prefix>/share/gnubg/gnubg.wd), so the weights pin "
            f"{GNUBG_REQUIRED_WEIGHTS_MD5} cannot be verified. Set "
            f"GNUBG_ALLOW_VERSION_MISMATCH=1 to run unverified ON PURPOSE."
        )
    if w != GNUBG_REQUIRED_WEIGHTS_MD5:
        raise RuntimeError(
            f"gnubg.wd for '{cmd}' has md5 {w}, but results are pinned to "
            f"{GNUBG_REQUIRED_WEIGHTS_MD5} (gnubg >= 1.08 ships retrained "
            f"nets, so the version alone does not pin the judge). Set "
            f"GNUBG_ALLOW_VERSION_MISMATCH=1 to override ON PURPOSE."
        )
    _VERSION_OK[cmd] = True


def assert_gnubg_bearoff(gnubg_cmd: str = None) -> None:
    """Refuse to run unless gnubg has its TWO-SIDED bearoff database
    (gnubg_ts0.bd). Without it gnubg silently substitutes a net estimate
    for exact cubeful bearoff equities -- wrong cube play, fabricated cube
    errors when judging -- and the only signal is a startup warning that
    the launch sites discard. Install location varies (Debian:
    /var/lib/gnubg/), so ask gnubg itself; cached per binary (~0.2s once).
    GNUBG_ALLOW_ONESIDED_BEAROFF=1 opts out deliberately."""
    cmd = gnubg_cmd or GNUBG_CMD
    if cmd in _BEAROFF_OK:
        return
    _assert_gnubg_pinned_judge(cmd)
    if os.environ.get("GNUBG_ALLOW_ONESIDED_BEAROFF") == "1":
        _BEAROFF_OK[cmd] = True
        return
    try:
        out = subprocess.run(
            [cmd, "-t", "-q", "-c", os.devnull],
            capture_output=True, text=True, timeout=120,
            env=dict(os.environ, LC_ALL="C"),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"cannot probe gnubg at '{cmd}': {e}")
    blob = (out.stdout or "") + (out.stderr or "")
    if "will not use the two-sided bearoff" in blob:
        raise RuntimeError(
            "gnubg at '%s' is MISSING its two-sided bearoff database "
            "(gnubg_ts0.bd).\n"
            "Cubeful bearoff equities would fall back to a net estimate, so "
            "this host's results are NOT comparable with a correctly "
            "configured one.\n"
            "Fix: reinstall gnubg, or copy gnubg_ts0.bd (md5 "
            "44b6040b49b46cb9dd2ce8caa947044d) into the directory holding "
            "gnubg.wd, or regenerate it with "
            "`makebearoff -t 6x6 -f gnubg_ts0.bd`.\n"
            "To run one-sided ON PURPOSE set GNUBG_ALLOW_ONESIDED_BEAROFF=1."
            % cmd
        )
    _BEAROFF_OK[cmd] = True


# ── Game recording ───────────────────────────────────────────────────────────

@dataclass
class MoveRecord:
    """One player's turn: dice roll and chosen play."""
    player: int                # WHITE or BLACK
    dice: Tuple[int, int]      # e.g. (3, 1)
    play: Play                 # tuple of (src, dst) moves; () if no legal moves


@dataclass
class CubeRecord:
    """A cube action (double / take / drop)."""
    player: int                # who acted
    action: str                # "double", "take", or "drop"
    new_cube_value: int = 2    # cube value after doubling (for "double")


@dataclass
class GameRecord:
    """A complete recorded game. `moves` may interleave MoveRecord
    and CubeRecord entries when the game is cubeful.
    """
    moves: List = field(default_factory=list)
    winner: Optional[int] = None
    result: Optional[int] = None   # 1=normal, 2=gammon, 3=backgammon
    cube_value: int = 1            # final cube value (cubeful only)
    ended_by_drop: bool = False    # True if game ended on a drop


def assert_cubeful_supported(model_path: str, cubeful: bool) -> None:
    """Reject prob5 + cubeful early (in the parent, before workers spawn).

    Raising inside a multiprocessing pool worker surfaces as an opaque
    pool exception (or a hang); callers validate once up front instead.
    """
    if not cubeful:
        return
    from model import load_model, ProbNetwork
    if isinstance(load_model(model_path), ProbNetwork):
        raise ValueError(
            f"{model_path}: prob5 is cubeless-only and has no cube policy; "
            "cubeful gnubg analysis is not supported.")


def _load_eval_agent(model_path: str, oneply: bool = False, cubeful: bool = False):
    """Load a checkpoint and wrap it in the right agent for analysis:
    prob5 -> ProbAgent (cubeless money only), scalar -> TDAgent."""
    from model import load_model, ProbNetwork
    net = load_model(model_path)
    if isinstance(net, ProbNetwork):
        if cubeful:  # backstop; parent should validate before spawning workers
            raise ValueError(
                f"{model_path}: prob5 is cubeless-only and has no cube policy; "
                "cubeful gnubg analysis is not supported.")
        from prob_agent import ProbAgent
        return ProbAgent(net, plies=1 if oneply else 0)
    from td_agent import TDAgent
    return TDAgent(net, oneply=oneply)


def _avg_rates(rates):
    """Split a list of (white, black) error-rate tuples into per-side
    lists plus the combined mean. Returns (white_list, black_list, mean)
    where mean is None when there are no rates."""
    w = [er[0] for er in rates]
    b = [er[1] for er in rates]
    a = w + b
    return w, b, sum(a) / len(a) if a else None


def play_and_record(agent_white, agent_black) -> GameRecord:
    """Play a full cubeless game, recording every dice roll and move.

    Both agents should implement choose_checker_action(state, dice, plays).
    """
    # Standard opening: each player rolls one die, higher goes first, and
    # doublets are rerolled (a game cannot start with a doublet).
    state, opening_dice = opening_roll()
    record = GameRecord()
    first_roll = True

    while not state.is_game_over():
        if first_roll:
            d1, d2 = opening_dice
            first_roll = False
        else:
            d1, d2 = random.randint(1, 6), random.randint(1, 6)
        plays = get_legal_plays(state, (d1, d2))

        if plays:
            agent = agent_white if state.turn == WHITE else agent_black
            play, next_state = agent.choose_checker_action(state, (d1, d2), plays)
            if play is None:
                # Some agents (e.g. gnubg-ext) return only the successor;
                # recover the move by matching it among the legal plays
                # (as play_and_record_cubeful does) so export_mat can
                # render it — otherwise the .mat has dice but blank moves
                # and the gnubg judge scores 0 error.
                for _p, _s in plays:
                    if _s == next_state:
                        play = _p
                        break
            record.moves.append(MoveRecord(
                player=state.turn,
                dice=(d1, d2),
                play=play,
            ))
            # next_state already has turn switched (engine convention)
            state = next_state
        else:
            # No legal moves -- record the dice roll with empty play
            record.moves.append(MoveRecord(
                player=state.turn,
                dice=(d1, d2),
                play=(),
            ))
            state = switch_turn(state)

    record.winner = state.winner()
    record.result = state.game_result()
    return record


def play_and_record_cubeful(
    agent_white, agent_black, jacoby: bool = True,
    record_moves: bool = True,
) -> GameRecord:
    """Play a full cubeful money game, recording dice, moves, and
    cube actions. Both agents must be cubeful TDAgents supporting
    offer_double / respond_to_double / choose_checker_action_cubeful.

    When `record_moves=False`, skips the per-move Play recording —
    saves a `get_legal_plays` call per turn (the agent already
    enumerated and chose internally; re-enumerating to find the
    matching Play object is the dominant remaining cost in cubeful
    self-play). Use False for stat-only runs (the cube actions and
    final result are still recorded so stake computation works).

    The cube is intentionally UNCAPPED — redouble/take loops are left to
    run to completion (the +/-128 capped equity handles the statistics;
    runaway games are logged in full for inspection, not truncated).
    """
    from modes import MatchState, CubeOwner  # local

    state, opening_dice = opening_roll()
    record = GameRecord()
    match_state = MatchState(jacoby=jacoby)
    is_opening = True

    while not state.is_game_over():
        player = state.turn
        agent = agent_white if player == WHITE else agent_black
        opp_agent = agent_black if player == WHITE else agent_white

        # ── Cube decision (not allowed on opening roll) ──────
        if not is_opening and match_state.can_offer(player):
            offer = agent.offer_double(state, match_state)
            if offer.should_double:
                new_cube_value = match_state.cube_value * 2
                record.moves.append(CubeRecord(
                    player=player, action="double",
                    new_cube_value=new_cube_value,
                ))
                # Opponent responds; self-play threads the cache through,
                # inter-agent play gives no hint (opponent recomputes).
                hint = offer if opp_agent is agent else None
                takes = opp_agent.respond_to_double(
                    state, match_state, hint=hint,
                )
                opponent = 1 - player
                if takes:
                    record.moves.append(CubeRecord(
                        player=opponent, action="take",
                        new_cube_value=new_cube_value,
                    ))
                    match_state = match_state.after_take(player)
                else:
                    record.moves.append(CubeRecord(
                        player=opponent, action="drop",
                        new_cube_value=new_cube_value,
                    ))
                    record.winner = player
                    record.result = 1
                    record.cube_value = match_state.cube_value  # pre-double
                    record.ended_by_drop = True
                    return record

        # ── Checker play ──────────────────────────────────────
        if is_opening:
            d1, d2 = opening_dice
        else:
            d1, d2 = random.randint(1, 6), random.randint(1, 6)
        result = agent.choose_checker_action_cubeful(
            state, (d1, d2), match_state,
        )
        if result is not None:
            next_state, _bootstrap = result
            if record_moves:
                # Find the matching Play object (engine gives us both).
                plays = get_legal_plays(state, (d1, d2))
                play = ()
                for p, s in plays:
                    if s == next_state:
                        play = p
                        break
                record.moves.append(MoveRecord(
                    player=player, dice=(d1, d2), play=play,
                ))
            state = next_state
        else:
            if record_moves:
                record.moves.append(MoveRecord(
                    player=player, dice=(d1, d2), play=(),
                ))
            state = switch_turn(state)
        is_opening = False

    record.winner = state.winner()
    raw_result = state.game_result()
    # Jacoby: gammons/bgs only count if cube has been turned.
    if jacoby and match_state.cube_owner == CubeOwner.CENTERED:
        raw_result = 1
    record.result = raw_result
    record.cube_value = match_state.cube_value
    return record


# ── .mat export (Jellyfish format) ───────────────────────────────────────────

def _move_notation(move: Move, player: int) -> str:
    """Convert a single (src, dst) move to gnubg .mat notation.

    gnubg convention: bar = point 25, bear-off = point 0.
    White's notation uses points 1-24 (our index + 1).
    Black's notation mirrors: our index i -> Black's point (24 - i).
    """
    src, dst = move

    if player == WHITE:
        src_str = "25" if src == BAR else str(src + 1)
        dst_str = "0" if dst == OFF else str(dst + 1)
    else:
        src_str = "25" if src == BAR else str(24 - src)
        dst_str = "0" if dst == OFF else str(24 - dst)

    return f"{src_str}/{dst_str}"


def _play_notation(play: Play, player: int) -> str:
    """Convert a full play to notation string, e.g. '8/5 6/5'."""
    if not play:
        return ""
    return " ".join(_move_notation(m, player) for m in play)


def _dice_str(dice: Tuple[int, int]) -> str:
    """Format dice as two digits, e.g. (3,1) -> '31'."""
    return f"{dice[0]}{dice[1]}"


# .mat play-row layout: "%3d)" + space, then the white field padded to
# _MAT_W_WIDTH, then black. XG reads the winner off the column its token sits
# in, so the winner line must use these same offsets.
_MAT_W_WIDTH = 33
_MAT_COL_W = 5
_MAT_COL_B = _MAT_COL_W + _MAT_W_WIDTH

# A roll with no legal play, written as "DD:" with nothing after it.
_FORCED_PASS = re.compile(r"^\s*[1-6][1-6]:\s*$")


def export_mat(
    record: GameRecord, game_id: int = 1, money_game: bool = False,
    white: str = "white", black: str = "black",
) -> str:
    """Export a GameRecord to Jellyfish .mat format string.

    Cubeless: pairs white/black moves per line. `money_game=False`
    writes a "1 point match" header.

    Cubeful: same pairing but cube actions (Doubles/Takes/Drops) are
    interleaved into the appropriate columns following gnubg's
    left-to-right reading order. `money_game=True` writes a "0 point
    match" header (= money game).

    `white` / `black` name the players; XG shows them as profile names.
    """
    lines = []

    # Money games need the XG rules block: without it XG assumes Beavers
    # and misreads the cube sequence. Jacoby "On" matches the jacoby=True
    # money path; match exports skip the block.
    if money_game:
        lines.append('; [Jacoby "On"]')
        lines.append('; [Beaver "Off"]')
        lines.append('; [CubeLimit "1024"]')
        lines.append('; [Unrated "On"]')

    lines.append("")
    lines.append(" 0 point match" if money_game else " 1 point match")
    lines.append("")
    lines.append(f" Game {game_id}")
    # XG surfaces these as the profile names in a stats export, so naming them
    # after the engine/checkpoint makes an export self-identifying.
    lines.append(f" {white} : 0                      {black} : 0")

    # Rows are [left_col, right_col] (left=WHITE, right=BLACK).
    rows: List[List[str]] = []

    def _ensure_row(idx: int) -> None:
        while len(rows) <= idx:
            rows.append(["", ""])

    w_next = 0  # next row index where WHITE can write to the left col
    b_next = 0  # next row index where BLACK can write to the right col

    def _clear_of_cube(row: int, other_col: int) -> int:
        """First row at or after `row` whose `other_col` holds no cube token.

        The mirror of _clear_of_pass: that one keeps a cube action off a row
        whose other column is already a forced pass, but a forced pass written
        LATER could still land beside a cube token, which is the same
        ambiguity from the other direction.
        """
        while row < len(rows) and rows[row][other_col].strip().startswith(
                ("Doubles", "Takes", "Drops")):
            row += 1
        _ensure_row(row)
        return row

    def _clear_of_pass(row: int, other_col: int) -> int:
        """First row at or after `row` whose `other_col` is not a forced pass.

        A cube token sharing a row with a bare "DD:" (roll, no legal play) is
        where XG has been observed to lose the cube: it reported a 2-point win
        for a game the record scores at 8 (cube 4, gammon). Giving the cube
        action its own row removes the ambiguity.
        """
        while row < len(rows) and _FORCED_PASS.match(rows[row][other_col]):
            row += 1
        _ensure_row(row)
        return row

    i = 0
    entries = record.moves
    while i < len(entries):
        entry = entries[i]

        if isinstance(entry, CubeRecord) and entry.action == "double":
            dbl_str = f" Doubles => {entry.new_cube_value}"
            # Consume the following response (take/drop) entry, if any.
            resp_str = ""
            if i + 1 < len(entries) and isinstance(entries[i + 1], CubeRecord):
                resp = entries[i + 1]
                resp_str = " Takes" if resp.action == "take" else " Drops"
                i += 1

            if entry.player == WHITE:
                # WHITE doubles (left col) + response (right col) on SAME row.
                _ensure_row(w_next)
                row = _clear_of_pass(w_next, 1)
                rows[row][0] = dbl_str
                rows[row][1] = resp_str
                w_next = row + 1
                b_next = max(b_next, row + 1)
            else:
                # BLACK doubles: double on right col of current row, response
                # on left col of NEXT row (gnubg reads left-to-right).
                _ensure_row(b_next)
                row = _clear_of_pass(b_next, 0)
                rows[row][1] = dbl_str
                b_next = row + 1
                resp_row = _clear_of_pass(max(w_next, b_next), 1)
                rows[resp_row][0] = resp_str
                w_next = resp_row + 1
                b_next = max(b_next, resp_row)

        elif isinstance(entry, MoveRecord):
            notation = _play_notation(entry.play, entry.player)
            dice = _dice_str(entry.dice)
            s = f"{dice}: {notation}" if notation else f"{dice}: "

            forced_pass = not notation
            if entry.player == WHITE:
                _ensure_row(w_next)
                row = _clear_of_cube(w_next, 1) if forced_pass else w_next
                rows[row][0] = s
                w_next = row + 1
            else:
                _ensure_row(b_next)
                row = _clear_of_cube(b_next, 0) if forced_pass else b_next
                rows[row][1] = s
                b_next = row + 1
                w_next = max(w_next, b_next)

        i += 1

    # Winner detection
    winner = record.winner
    if winner is None:
        for entry in reversed(record.moves):
            if isinstance(entry, MoveRecord):
                winner = entry.player
                break
        if winner is None:
            winner = WHITE

    # Points awarded
    cube = record.cube_value
    result_mult = record.result if record.result else 1
    points = cube if record.ended_by_drop else cube * result_mult
    point_word = "point" if points == 1 else "points"
    wins_str = f"Wins {points} {point_word}"

    # For drops, place the "Wins" text alongside the drop marker in the same
    # row so gnubg reads it correctly -- but ONLY in the WINNER's column. The
    # winner of a dropped double is the DOUBLER, not the dropper, so appending
    # it to whichever column happens to hold " Drops" silently credits the game
    # to the loser -- readers take the winner from the column the token sits
    # in. When the winner's column on the last row is already taken
    # (white doubled there), fall back to a separate winner-column line.
    winner_col = 0 if winner == WHITE else 1
    drop_wins_separate = False
    if record.ended_by_drop and rows:
        last = rows[-1]
        if not last[winner_col]:
            last[winner_col] = wins_str
        else:
            drop_wins_separate = True

    for idx, (w, b) in enumerate(rows):
        num = f"{idx + 1:>3d})"
        if b:
            lines.append(f"{num} {w:<{_MAT_W_WIDTH}s}{b}")
        else:
            lines.append(f"{num} {w}")

    # Same columns as the play rows above, so a standalone winner line sits in
    # the same column as that player's moves.
    if not record.ended_by_drop or drop_wins_separate:
        pad = _MAT_COL_B if winner == BLACK else _MAT_COL_W
        lines.append(" " * pad + wins_str)
    lines.append("")

    return "\n".join(lines)


# ── gnubg analysis ───────────────────────────────────────────────────────────

def _write_gnubg_script(mat_files: List[str], script_path: str, plies: int = 3,
                        extra_commands: Optional[List[str]] = None):
    """Write a Python script for gnubg's embedded interpreter.

    The script imports each .mat file, analyzes it, and prints statistics.
    We look for mEMG in the output. ``extra_commands`` are issued first
    (e.g. "set threads 1").
    """
    lines = ["import gnubg", ""]
    for c in extra_commands or []:
        lines.append(f"gnubg.command('{c}')")
    lines += [
        f"# {plies}-ply analysis for both chequer and cube decisions",
        f"gnubg.command('set analysis chequerplay evaluation plies {plies}')",
    ]
    # Cubeless-money scoring: gnubg's default chequer eval is CUBEFUL, which
    # scores moves by cube-leveraged equity. Set GNUBG_CUBELESS_EVAL=1 to score
    # by static cubeless money equity (gammons/backgammons count, no cube) —
    # the objective our cubeless models train on. IMPORTANT: gnubg still scores
    # MISSED doubles in a 0-point session even when no cube action occurred, so
    # cube analysis must be turned OFF too — otherwise the never-doubling cubeless
    # agents rack up huge phantom cube errors and 'overall' becomes ~85% artifact
    # (it is NOT chequer+cube of real play). With cube off, overall == chequer.
    if os.environ.get("GNUBG_CUBELESS_EVAL") == "1":
        lines.append("gnubg.command('set analysis chequerplay evaluation cubeful off')")
        lines.append("gnubg.command('set analysis cube off')")
    else:
        lines.append(f"gnubg.command('set analysis cubedecision evaluation plies {plies}')")
    lines.append("")

    for mat_file in mat_files:
        # Normalize path separators for the platform
        mat_path = mat_file.replace("\\", "/")
        lines.append(f'gnubg.command(\'import mat "{mat_path}"\')')
        lines.append("gnubg.command('analyze match')")
        lines.append("gnubg.command('show statistics match')")
        lines.append("")

    with open(script_path, "w") as f:
        f.write("\n".join(lines))


def run_gnubg_analysis(
    mat_files: List[str],
    work_dir: str,
    gnubg_cmd: Optional[str] = None,
    verbose: bool = True,
    plies: int = 3,
    extra_commands: Optional[List[str]] = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]],
           List[Tuple[float, float]]]:
    """Run gnubg in batch mode on a list of .mat files.

    Returns (chequer_rates, cube_rates, overall_rates), where each is a
    list of (white_mEMG, black_mEMG) tuples, one per game.
    mEMG = millipoints Error per Move in Equivalent Money Game.
    """
    if gnubg_cmd is None:
        gnubg_cmd = GNUBG_CMD

    # Every analysis path funnels through here, so this is the choke point
    # that guarantees no result is ever produced by a gnubg missing its
    # two-sided bearoff database. Cached per binary, so the probe costs one
    # ~0.2 s gnubg start per worker process, not per chunk.
    assert_gnubg_bearoff(gnubg_cmd)

    # Use absolute paths so gnubg can find everything regardless of cwd.
    abs_work_dir = os.path.abspath(work_dir)
    abs_mat_files = [os.path.abspath(f) for f in mat_files]
    script_path = os.path.join(abs_work_dir, "_gnubg_analyze.py")
    _write_gnubg_script(abs_mat_files, script_path, plies=plies,
                        extra_commands=extra_commands)

    # Run gnubg with:
    #   -t : text-only (no GUI)
    #   -q : quiet (suppress sounds/splash)
    #   -p : run Python script
    cmd = [gnubg_cmd, "-t", "-q", "-p", script_path]

    if verbose:
        print(f"  gnubg command: {' '.join(cmd)}")
        print(f"  working dir:   {work_dir}")

    # No timeout: dropping slow games would bias mEMG against long/complex
    # games — every game must be scored; a hang here means a code bug,
    # which we want surfaced, not silently skipped.
    # Instead, chunks slower than a reference budget are LOGGED with
    # elapsed time + per-game move counts so we can tell
    # "long game" (many moves) from "stuck gnubg" (few moves, huge time).
    slow_after = 120 + len(mat_files) * 60 * (4 ** max(0, plies - 2))
    _t0 = _time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"gnubg not found at '{gnubg_cmd}'. "
            f"Install gnubg or set GNUBG_CMD in gnubg_eval.py."
        )
    _elapsed = _time.monotonic() - _t0
    if _elapsed > slow_after:
        _det = []
        for _mf in mat_files:
            try:
                with open(_mf) as _fh:
                    _txt = _fh.read()
                # count half-moves: lines like " 12) 63: 24/18 13/10 ..."
                _nmoves = sum(_ln.count(":") for _ln in _txt.splitlines()
                              if _ln.strip()[:1].isdigit() and ")" in _ln)
            except OSError:
                _nmoves = -1
            _det.append(f"{os.path.basename(_mf)}={_nmoves}mv")
        print(f"  [slow-chunk] {int(_elapsed)}s (budget ref {int(slow_after)}s) "
              f"plies={plies} games={len(mat_files)} moves: {' '.join(_det)} "
              f"-- high move count => genuinely long game; low count + huge "
              f"time => investigate for a bug", flush=True)

    # Dump raw output for debugging
    output = result.stdout + result.stderr
    if verbose:
        # Show all mEMG lines so we can see what gnubg actually prints
        print("  --- gnubg mEMG lines ---")
        for line in output.splitlines():
            if "mEMG" in line:
                print(f"  {line.rstrip()}")
        print("  --- end mEMG lines ---")

    # Parse section-by-section. Per game, gnubg's `show statistics match`
    # prints three blocks in order -- "Chequerplay statistics", "Cube
    # statistics", "Overall statistics" -- each containing exactly one
    # "Error rate ... mEMG ..." line with a (white, black) pair. We key each
    # error line off the section header that precedes it rather than grouping
    # positionally in 3s.
    #
    # Why not positional 3-grouping: a game in which a player has 0 unforced
    # moves (a very short game) prints "n/a" for the chequer error rate, so the
    # value-regex matches nothing on that line and the game emits only 2 parsed
    # error lines instead of 3. Positional grouping would then drop one line and
    # misalign EVERY subsequent game, scattering a few large junk values into
    # the wrong component (observed: a 1-point/DMP match has no cube decisions,
    # so overall must equal chequer, yet ~5/1000 games disagreed by 20-44 mEMG).
    # Section keying buckets each game's components correctly; a game missing any
    # component (e.g. chequer = n/a) is dropped whole so the series stay aligned.
    #
    # Header spelling varies by gnubg version: older builds print "Chequerplay
    # statistics", gnubg 1.08 prints "Checkerplay statistics". Match both.
    # Numbers may carry an exponent (gnubg's degenerate rates, e.g. -1.0e+32).
    # Without the exponent alternative the regex would backtrack and match "32"
    # out of "e+32" — a plausible value that sneaks past the CAP filter.
    val_re = re.compile(r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
                        r"\s*\(\s*[-+]?\d+\.\d+(?:[eE][-+]?\d+)?%?\s*\)")
    games = []          # list of {"chequer","cube","overall"} dicts
    cur = None
    section = None
    for line in output.splitlines():
        if "Chequerplay statistics" in line or "Checkerplay statistics" in line:
            if cur is not None:
                games.append(cur)
            cur = {}
            section = "chequer"
            continue
        if "Cube statistics" in line:
            section = "cube"
            continue
        if "Overall statistics" in line:
            section = "overall"
            continue
        if cur is None or section is None:
            continue
        if "mEMG" not in line or "Error rate" not in line:
            continue
        if section in cur:        # take the first error-rate line per section
            continue
        # Match-play (MWC): "-201.4 (-10.069%)" -> 201.4; Cubeful money
        # (Points): "-4.0 ( -0.008)" -> 4.0. "n/a" yields no match -> None.
        m = val_re.findall(line)
        cur[section] = (abs(float(m[0])), abs(float(m[1]))) if len(m) >= 2 else None
    if cur is not None:
        games.append(cur)

    # Drop degenerate or incomplete games: gnubg occasionally emits an absurd
    # MWC error rate (e.g. 1e32) when a near-perfect player has ~0 marked
    # decisions and normalization divides by ~0; and a game missing any
    # component (n/a) cannot be averaged. Real mEMG is at most a few hundred.
    CAP = 1.0e4
    # With GNUBG_CUBELESS_EVAL=1 the script issues 'set analysis cube off', so
    # gnubg prints NO Cube statistics section at all — a missing cube component
    # is expected (zero cube error), not an incomplete game to drop.
    cubeless = os.environ.get("GNUBG_CUBELESS_EVAL") == "1"
    chequer_rates, cube_rates, overall_rates = [], [], []
    dropped = 0
    for g in games:
        chk, cub, ovr = g.get("chequer"), g.get("cube"), g.get("overall")
        if cubeless and cub is None:
            cub = (0.0, 0.0)
        if chk is None or cub is None or ovr is None:
            dropped += 1
            continue
        vals = [chk[0], chk[1], cub[0], cub[1], ovr[0], ovr[1]]
        if any((v != v) or v > CAP for v in vals):  # v!=v catches NaN; inf>CAP
            dropped += 1
            continue
        chequer_rates.append(chk)
        cube_rates.append(cub)
        overall_rates.append(ovr)
    if dropped:
        print(f"  WARNING: dropped {dropped}/{len(games)} game(s) with degenerate "
              f"or incomplete gnubg mEMG (>{CAP:g}, non-finite, or n/a)", flush=True)

    return chequer_rates, cube_rates, overall_rates


# ── Parallel game generation ──────────────────────────────────────────────────

def _play_games_worker(args):
    """Worker function for parallel game generation.
    Each worker loads the model once and plays many games."""
    model_path, num_games, work_dir, start_idx, cubeful, jacoby, oneply = args

    os.environ["OMP_NUM_THREADS"] = "1"
    # model_path is None => random self-play baseline (the --random path).
    agent = (
        RandomAgent() if model_path is None
        else _load_eval_agent(model_path, oneply=oneply, cubeful=cubeful)
    )

    mat_files = []
    for i in range(num_games):
        game_id = start_idx + i
        if cubeful:
            record = play_and_record_cubeful(agent, agent, jacoby=jacoby)
        else:
            record = play_and_record(agent, agent)
        mat_content = export_mat(record, game_id=game_id, money_game=cubeful)
        mat_path = os.path.join(work_dir, f"game_{game_id}.mat")
        with open(mat_path, "w") as f:
            f.write(mat_content)
        mat_files.append(mat_path)
    return mat_files


def _gnubg_worker(args):
    """Worker function for parallel gnubg analysis on a chunk of .mat files."""
    mat_files, work_dir, gnubg_cmd, chunk_id, plies = args
    chequer, cube, overall = run_gnubg_analysis(
        mat_files, work_dir, gnubg_cmd=gnubg_cmd, verbose=False, plies=plies,
    )
    if len(overall) != len(mat_files):
        print(f"  WARNING: chunk {chunk_id}: gnubg returned {len(overall)}/{len(mat_files)} results",
              flush=True)
    return chequer, cube, overall


# ── High-level evaluation ────────────────────────────────────────────────────

def evaluate_with_gnubg_parallel(
    model_path: str,
    num_games: int = 100,
    gnubg_cmd: Optional[str] = None,
    work_dir: Optional[str] = None,
    keep_files: bool = False,
    verbose: bool = True,
    workers: int = 1,
    gnubg_workers: int = 4,
    gnubg_chunk_size: int = 50,
    cubeful: bool = False,
    jacoby: bool = True,
    oneply: bool = False,
    plies: int = 3,
) -> dict:
    """Parallel version: play games with multiple workers, analyze with multiple gnubg instances.

    Args:
        model_path:      Path to saved model (.pt file).
        num_games:        Number of evaluation games.
        gnubg_cmd:        Path to gnubg executable.
        work_dir:         Directory for .mat files.
        keep_files:       Keep .mat files after analysis.
        verbose:          Print progress.
        workers:          Number of parallel game-playing workers.
        gnubg_workers:    Number of parallel gnubg analysis processes.
        gnubg_chunk_size: Games per gnubg analysis chunk.
    """
    # model_path is None => random self-play baseline (no model to validate).
    if model_path is not None:
        assert_cubeful_supported(model_path, cubeful)
    if gnubg_cmd is None:
        gnubg_cmd = GNUBG_CMD
    # Preflight the judge BEFORE generating games: a wrong binary would
    # otherwise be discovered only after the whole generation phase.
    assert_gnubg_bearoff(gnubg_cmd)

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="gnubg_eval_")
    os.makedirs(work_dir, exist_ok=True)

    # ── Phase 1: Parallel game generation ──
    if verbose:
        print(f"Playing {num_games} games with {workers} workers...")

    import time
    t0 = time.perf_counter()

    if workers > 1:
        # Split games across workers — each worker loads model once, plays many games
        games_per_worker = []
        start_idx = 1
        for i in range(workers):
            n = num_games // workers + (1 if i < num_games % workers else 0)
            if n > 0:
                games_per_worker.append(
                    (model_path, n, work_dir, start_idx, cubeful, jacoby, oneply)
                )
                start_idx += n

        ctx = mp.get_context('spawn')
        with ctx.Pool(len(games_per_worker)) as pool:
            results = pool.map(_play_games_worker, games_per_worker)
        mat_files = [f for chunk in results for f in chunk]
    else:
        mat_files = _play_games_worker(
            (model_path, num_games, work_dir, 1, cubeful, jacoby, oneply))

    t_play = time.perf_counter() - t0
    if verbose:
        print(f"  Generated {len(mat_files)} games in {t_play:.1f}s "
              f"({len(mat_files)/t_play:.0f} games/sec)")

    # ── Phase 2: Parallel gnubg analysis ──
    if verbose:
        print(f"Analyzing with gnubg ({gnubg_workers} workers, {gnubg_chunk_size} games/chunk)...")

    t1 = time.perf_counter()

    # Split mat files into chunks for parallel gnubg analysis
    chunks = []
    for i in range(0, len(mat_files), gnubg_chunk_size):
        chunk = mat_files[i:i + gnubg_chunk_size]
        chunk_dir = os.path.join(work_dir, f"chunk_{len(chunks)}")
        os.makedirs(chunk_dir, exist_ok=True)
        chunks.append((chunk, chunk_dir, gnubg_cmd, len(chunks), plies))

    if gnubg_workers > 1 and len(chunks) > 1:
        ctx = mp.get_context('fork')  # fork is fine for gnubg subprocess calls
        with ctx.Pool(min(gnubg_workers, len(chunks))) as pool:
            chunk_results = pool.map(_gnubg_worker, chunks)
        all_chequer = [er for chk, _, _ in chunk_results for er in chk]
        all_cube = [er for _, cub, _ in chunk_results for er in cub]
        all_overall = [er for _, _, ovr in chunk_results for er in ovr]
    else:
        all_chequer, all_cube, all_overall = [], [], []
        for chunk_args in chunks:
            chk, cub, ovr = _gnubg_worker(chunk_args)
            all_chequer.extend(chk)
            all_cube.extend(cub)
            all_overall.extend(ovr)

    t_analyze = time.perf_counter() - t1

    # ── Summarize ──
    chk_w, chk_b, chk_avg = _avg_rates(all_chequer)
    cub_w, cub_b, cub_avg = _avg_rates(all_cube)
    ovr_w, ovr_b, ovr_avg = _avg_rates(all_overall)

    # Bootstrap CIs for mEMG. Share one ProcessPool across all 3 calls.
    from concurrent.futures import ProcessPoolExecutor
    from stats import bootstrap_ci
    ovr_per_game = [(w + b) / 2 for w, b in all_overall] if all_overall else []
    chk_per_game = [(w + b) / 2 for w, b in all_chequer] if all_chequer else []
    cub_per_game = [(w + b) / 2 for w, b in all_cube] if all_cube else []

    n_jobs = min(os.cpu_count() or 1, 32)
    with ProcessPoolExecutor(max_workers=n_jobs) as boot_pool:
        ovr_ci = (bootstrap_ci(ovr_per_game, executor=boot_pool)
                  if ovr_per_game else (0, 0, 1))
        chk_ci = (bootstrap_ci(chk_per_game, executor=boot_pool)
                  if chk_per_game else (0, 0, 1))
        cub_ci = (bootstrap_ci(cub_per_game, executor=boot_pool)
                  if cub_per_game else (0, 0, 1))

    results = {
        "chequer_mEMG": chk_avg,
        "chequer_ci": (chk_ci[0], chk_ci[1]),
        "cube_mEMG": cub_avg,
        "cube_ci": (cub_ci[0], cub_ci[1]),
        "avg_mEMG": ovr_avg,
        "avg_ci": (ovr_ci[0], ovr_ci[1]),
        "num_games": len(all_overall),
        "num_games_requested": num_games,
        "work_dir": work_dir,
    }

    if verbose:
        if all_overall:
            n_analyzed = len(all_overall)
            n_total = len(mat_files)
            print(f"  Analysis: {n_analyzed}/{n_total} games in {t_analyze:.1f}s "
                  f"({n_analyzed/t_analyze:.1f} games/sec)")
            if n_analyzed < n_total:
                print(f"  WARNING: {n_total - n_analyzed} games failed gnubg analysis")
            print(f"\n{'='*50}")
            print(f"  gnubg analysis: {n_analyzed} games")
            print(f"{'='*50}")
            print(f"  Chequer mEMG:   {chk_avg:.1f}  [{chk_ci[0]:.1f}, {chk_ci[1]:.1f}]  (W={sum(chk_w)/len(chk_w):.1f}  B={sum(chk_b)/len(chk_b):.1f})")
            print(f"  Cube mEMG:      {cub_avg:.1f}  [{cub_ci[0]:.1f}, {cub_ci[1]:.1f}]  (W={sum(cub_w)/len(cub_w):.1f}  B={sum(cub_b)/len(cub_b):.1f})")
            print(f"  Overall mEMG:   {ovr_avg:.1f}  [{ovr_ci[0]:.1f}, {ovr_ci[1]:.1f}]  (W={sum(ovr_w)/len(ovr_w):.1f}  B={sum(ovr_b)/len(ovr_b):.1f})")
            print(f"{'='*50}")
        else:
            print("  WARNING: no error rates returned from gnubg.")

    # Cleanup
    if not keep_files:
        for f in mat_files:
            try:
                os.remove(f)
            except OSError:
                pass
        for chunk_args in chunks:
            chunk_dir = chunk_args[1]
            try:
                script = os.path.join(chunk_dir, "_gnubg_analyze.py")
                os.remove(script)
                os.rmdir(chunk_dir)
            except OSError:
                pass
        try:
            os.rmdir(work_dir)
        except OSError:
            pass

    return results


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a backgammon model using gnubg error analysis"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to saved TD model (.pt)")
    parser.add_argument("--random", action="store_true",
                        help="Use random agents (no model needed, no torch needed)")
    parser.add_argument("--games", type=int, default=10,
                        help="Number of games to analyze")
    parser.add_argument("--gnubg", type=str, default=None,
                        help="Path to gnubg executable")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Directory for temp .mat files")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep .mat files after analysis")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel game-playing workers (default: 1)")
    parser.add_argument("--gnubg-workers", type=int, default=None,
                        help="Parallel gnubg analysis workers (default: CPU count)")
    parser.add_argument("--gnubg-chunk-size", type=int, default=50,
                        help="Games per gnubg analysis chunk (default: 50)")
    parser.add_argument("--cubeful", action="store_true",
                        help="Cubeful money game. Model must be a cubeful "
                             "agent (cubeful_perspective196 encoder).")
    parser.add_argument("--jacoby", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Jacoby rule for cubeful (default: on).")
    parser.add_argument("--oneply", action="store_true",
                        help="Use 1-ply lookahead during self-play game generation.")
    parser.add_argument("--plies", type=int, default=3,
                        help="gnubg analysis depth in plies (default: 3).")
    args = parser.parse_args()

    if not args.model and not args.random:
        parser.error("Either --model or --random is required")
    if args.model and args.random:
        parser.error("--model and --random are mutually exclusive")
    if args.random and args.cubeful:
        parser.error("--cubeful requires --model (random agents have no cube policy)")

    # Self-play only. model_path=None => random self-play baseline.
    model_path = None if args.random else args.model
    if args.random:
        print("Mode: random vs random (no model)")
    else:
        print(f"Loading model: {args.model}")
        print(f"Mode: self-play (model vs itself){' [1-ply]' if args.oneply else ''}")

    evaluate_with_gnubg_parallel(
        model_path=model_path,
        num_games=args.games,
        gnubg_cmd=args.gnubg,
        work_dir=args.work_dir,
        keep_files=args.keep_files,
        workers=args.workers,
        gnubg_workers=args.gnubg_workers or os.cpu_count(),
        gnubg_chunk_size=args.gnubg_chunk_size,
        cubeful=args.cubeful,
        jacoby=args.jacoby,
        oneply=args.oneply,
        plies=args.plies,
    )
