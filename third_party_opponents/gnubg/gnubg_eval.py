"""
gnubg_eval.py -- Game recording, .mat export and the gnubg binary preflight.

play_and_record / play_and_record_cubeful play one game and record every
roll, move and cube action; export_mat writes it as a Jellyfish .mat file
(XG PR kits, saved games). assert_gnubg_bearoff checks that the gnubg
binary is the pinned 1.08.003 with the stock weights and the two-sided
bearoff database. Override the gnubg binary with GNUBG_CMD.
"""

import os
import platform
import random
import subprocess
import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from backgammon_engine import (
    WHITE, BLACK, BAR, OFF,
    Play, Move, get_legal_plays, switch_turn, opening_roll,
)

# ── Platform configuration ───────────────────────────────────────────────────

# Default gnubg path per platform.
if platform.system() == "Windows":
    # Common install locations, tried in order.
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

# The GNUBG_CMD env var overrides the platform default.
GNUBG_CMD = os.environ.get("GNUBG_CMD", GNUBG_CMD)


# ── gnubg preflight: pinned version/weights + bearoff database ───────────────

_BEAROFF_OK: dict = {}
_VERSION_OK: dict = {}

# Pinned binary: a different version or gnubg.wd gives non-comparable
# numbers while appearing to work.
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
    """md5 of the gnubg.wd the binary at `cmd` will load, or None if it
    is not at the autotools location <prefix>/share/gnubg/gnubg.wd
    (gnubg does not report its pkgdatadir)."""
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
    Override: GNUBG_ALLOW_VERSION_MISMATCH=1."""
    if cmd in _VERSION_OK:
        return
    if os.environ.get("GNUBG_ALLOW_VERSION_MISMATCH") == "1":
        _VERSION_OK[cmd] = True
        return
    v = _gnubg_version(cmd)
    if v is None:
        raise RuntimeError(
            f"cannot run gnubg at '{cmd}' (missing, or `--version` failed): "
            f"install gnubg {GNUBG_REQUIRED_VERSION}, or point GNUBG_CMD at "
            f"its binary.")
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
        # Weights pin unverifiable: refuse rather than fail open.
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
            f"nets, so the version alone does not pin the engine). Set "
            f"GNUBG_ALLOW_VERSION_MISMATCH=1 to override ON PURPOSE."
        )
    _VERSION_OK[cmd] = True


def assert_gnubg_bearoff(gnubg_cmd: str = None) -> None:
    """Refuse to run unless gnubg has its two-sided bearoff database
    (gnubg_ts0.bd). Without it gnubg silently uses a net estimate for
    cubeful bearoff equities. Cached per binary; also runs the
    version/weights pin. Override: GNUBG_ALLOW_ONESIDED_BEAROFF=1."""
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
    jacoby: bool = True            # rule the game was played under (cubeful only)


def play_and_record(agent_white, agent_black, rng=None) -> GameRecord:
    """Play a full cubeless game, recording every dice roll and move.

    Both agents should implement choose_checker_action(state, dice, plays).

    ``rng`` (optional): a dedicated dice source with ``randint(a, b)``
    (e.g. a ``random.Random``); every die of this game, opening roll
    included, is drawn from it and nothing else touches it. ``None``
    (default) draws from the global ``random`` module.
    """
    if rng is None:
        rng = random
    # Opening roll: higher die moves first, no doublet.
    state, opening_dice = opening_roll(rng)
    record = GameRecord()
    first_roll = True

    while not state.is_game_over():
        if first_roll:
            d1, d2 = opening_dice
            first_roll = False
        else:
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        plays = get_legal_plays(state, (d1, d2))

        if plays:
            agent = agent_white if state.turn == WHITE else agent_black
            play, next_state = agent.choose_checker_action(state, (d1, d2), plays)
            if play is None:
                # Successor-only agents (e.g. gnubg-ext) return no Play:
                # recover it, since a blank move scores 0 error.
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
            # No legal moves: record the dice roll with an empty play.
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
    record_moves: bool = True, rng=None, cube_cap: int = None,
) -> GameRecord:
    """Play a full cubeful money game, recording dice, moves, and cube
    actions. Both agents must support offer_double / respond_to_double /
    choose_checker_action_cubeful.

    `record_moves=False` skips per-move Play recording. Cube actions and
    the result are still recorded. The cube is uncapped unless
    ``cube_cap=N``: once the cube reaches N neither side is offered a
    double (HedgeHog's CUBE_MAX_VALUE rule, applied to both sides).

    ``rng`` (optional): a dedicated dice source with ``randint(a, b)``;
    every die of this game, opening roll included, comes from it. ``None``
    (default) draws from the global ``random`` module.
    """
    from modes import MatchState, CubeOwner

    if rng is None:
        rng = random
    state, opening_dice = opening_roll(rng)
    record = GameRecord(jacoby=jacoby)
    match_state = MatchState(jacoby=jacoby)
    is_opening = True

    while not state.is_game_over():
        player = state.turn
        agent = agent_white if player == WHITE else agent_black
        opp_agent = agent_black if player == WHITE else agent_white

        # ── Cube decision (not allowed on opening roll) ──────
        if (not is_opening
                and (cube_cap is None or match_state.cube_value < cube_cap)
                and match_state.can_offer(player)):
            offer = agent.offer_double(state, match_state)
            if offer.should_double:
                new_cube_value = match_state.cube_value * 2
                record.moves.append(CubeRecord(
                    player=player, action="double",
                    new_cube_value=new_cube_value,
                ))
                # Self-play passes the offer as a cache hint. Inter-agent
                # play recomputes.
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
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        result = agent.choose_checker_action_cubeful(
            state, (d1, d2), match_state,
        )
        if result is not None:
            next_state, _bootstrap = result
            if record_moves:
                # Recover the Play for the record.
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


# .mat play-row layout: "%3d)" + space, white field padded to
# _MAT_W_WIDTH, then black. XG reads the winner off the column its token
# sits in, so the winner line uses the same offsets.
_MAT_W_WIDTH = 33
_MAT_COL_W = 5
_MAT_COL_B = _MAT_COL_W + _MAT_W_WIDTH

# A roll with no legal play, written as "DD:" with nothing after it.
_FORCED_PASS = re.compile(r"^\s*[1-6][1-6]:\s*$")


def export_mat(
    record: GameRecord, game_id: int = 1, money_game: bool = False,
    white: str = "white", black: str = "black",
) -> str:
    """Render a GameRecord as a Jellyfish .mat string.

    Cubeless: white/black moves paired per line under a "1 point match"
    header. Cubeful (`money_game=True`): "0 point match" (= money)
    header, cube actions interleaved in gnubg's left-to-right reading
    order. `white` / `black` are the player names (XG shows them as
    profile names).
    """
    lines = []

    # XG rules block: without it XG assumes Beavers and misreads the
    # cube sequence. Jacoby follows the rule the game was played under.
    if money_game:
        lines.append(f'; [Jacoby "{"On" if record.jacoby else "Off"}"]')
        lines.append('; [Beaver "Off"]')
        lines.append('; [CubeLimit "1024"]')
        lines.append('; [Unrated "On"]')

    lines.append("")
    lines.append(" 0 point match" if money_game else " 1 point match")
    lines.append("")
    lines.append(f" Game {game_id}")
    lines.append(f" {white} : 0                      {black} : 0")

    # Rows are [left_col, right_col] (left=WHITE, right=BLACK).
    rows: List[List[str]] = []

    def _ensure_row(idx: int) -> None:
        while len(rows) <= idx:
            rows.append(["", ""])

    w_next = 0  # next row index where WHITE can write to the left col
    b_next = 0  # next row index where BLACK can write to the right col

    def _clear_of_cube(row: int, other_col: int) -> int:
        """First row at or after `row` whose `other_col` holds no cube
        token: the mirror of _clear_of_pass, for a forced pass written
        after the cube action."""
        while row < len(rows) and rows[row][other_col].strip().startswith(
                ("Doubles", "Takes", "Drops")):
            row += 1
        _ensure_row(row)
        return row

    def _clear_of_pass(row: int, other_col: int) -> int:
        """First row at or after `row` whose `other_col` is not a forced
        pass: XG loses the cube when a cube token shares a row with a
        bare "DD:"."""
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

    # Drops: put "Wins" on the last row, in the WINNER's column (readers
    # take the winner from the column, and the doubler wins a drop, not
    # the dropper). If that column is taken, use a separate winner line.
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

    # Standalone winner line, in that player's column.
    if not record.ended_by_drop or drop_wins_separate:
        pad = _MAT_COL_B if winner == BLACK else _MAT_COL_W
        lines.append(" " * pad + wins_str)
    lines.append("")

    return "\n".join(lines)
