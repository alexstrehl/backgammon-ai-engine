"""
gnubg_eval.py -- Evaluate backgammon self-play games using GNU Backgammon.

Workflow:
    1. Play evaluation games with the model (self-play or vs random).
    2. Record dice rolls and moves for each game.
    3. Export the games as .mat files (Jellyfish format that gnubg imports).
    4. Run gnubg once in batch mode to analyze all games.
    5. Parse the mEMG error rates from gnubg's output.

The mEMG (millipoints Error per Move in Equivalent Money Game) is the
standard metric used by the backgammon community to measure play quality.
Lower is better:
    mEMG < 3    : world class
    mEMG 3-5    : expert
    mEMG 5-8    : advanced
    mEMG 8-12   : intermediate
    mEMG > 12   : beginner

Platform support:
    Set GNUBG_CMD to the path of your gnubg CLI executable.
    - Windows:  typically r'C:\\Program Files (x86)\\gnubg\\gnubg-cli.exe'
                or wherever you installed it.
    - Linux:    typically '/usr/games/gnubg' (from apt-get install gnubg)

Usage:
    python gnubg_eval.py --model td_model_final.pt --games 10
"""

import math
import multiprocessing as mp
import os
import platform
import random
import subprocess
import tempfile
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from backgammon_engine import (
    BoardState, WHITE, BLACK, BAR, OFF,
    Play, Move, get_legal_plays, switch_turn,
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


def play_and_record(agent_white, agent_black) -> GameRecord:
    """Play a full cubeless game, recording every dice roll and move.

    Both agents should implement choose_checker_action(state, dice, plays).
    """
    state = BoardState.initial()
    record = GameRecord()

    while not state.is_game_over():
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        plays = get_legal_plays(state, (d1, d2))

        if plays:
            agent = agent_white if state.turn == WHITE else agent_black
            play, next_state = agent.choose_checker_action(state, (d1, d2), plays)
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
) -> GameRecord:
    """Play a full cubeful money game, recording dice, moves, and
    cube actions. Both agents must be cubeful TDAgents supporting
    offer_double / respond_to_double / choose_checker_action_cubeful.
    """
    from modes import MatchState, CubeOwner, cube_perspective  # local

    state = BoardState.initial()
    record = GameRecord()
    match_state = MatchState(jacoby=jacoby)

    while not state.is_game_over():
        player = state.turn
        agent = agent_white if player == WHITE else agent_black
        opp_agent = agent_black if player == WHITE else agent_white

        # ── Cube decision ─────────────────────────────────────
        if match_state.can_offer(player):
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
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        result = agent.choose_checker_action_cubeful(
            state, (d1, d2), match_state,
        )
        if result is not None:
            next_state, _bootstrap = result
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
            record.moves.append(MoveRecord(
                player=player, dice=(d1, d2), play=(),
            ))
            state = switch_turn(state)

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


def export_mat(
    record: GameRecord, game_id: int = 1, money_game: bool = False,
) -> str:
    """Export a GameRecord to Jellyfish .mat format string.

    Cubeless: pairs white/black moves per line. `money_game=False`
    writes a "1 point match" header.

    Cubeful: same pairing but cube actions (Doubles/Takes/Drops) are
    interleaved into the appropriate columns following gnubg's
    left-to-right reading order. `money_game=True` writes a "0 point
    match" header (= money game).
    """
    lines = []

    # Header
    lines.append("")
    lines.append(" 0 point match" if money_game else " 1 point match")
    lines.append("")
    lines.append(f" Game {game_id}")
    lines.append(" white : 0                      black : 0")

    # Rows are [left_col, right_col] (left=WHITE, right=BLACK).
    rows: List[List[str]] = []

    def _ensure_row(idx: int) -> None:
        while len(rows) <= idx:
            rows.append(["", ""])

    w_next = 0  # next row index where WHITE can write to the left col
    b_next = 0  # next row index where BLACK can write to the right col

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
                row = w_next
                _ensure_row(row)
                rows[row][0] = dbl_str
                rows[row][1] = resp_str
                w_next = row + 1
                b_next = max(b_next, row + 1)
            else:
                # BLACK doubles: double on right col of current row, response
                # on left col of NEXT row (gnubg reads left-to-right).
                row = b_next
                _ensure_row(row)
                rows[row][1] = dbl_str
                b_next = row + 1
                resp_row = max(w_next, b_next)
                _ensure_row(resp_row)
                rows[resp_row][0] = resp_str
                w_next = resp_row + 1
                b_next = max(b_next, resp_row)

        elif isinstance(entry, MoveRecord):
            notation = _play_notation(entry.play, entry.player)
            dice = _dice_str(entry.dice)
            s = f"{dice}: {notation}" if notation else f"{dice}: "

            if entry.player == WHITE:
                _ensure_row(w_next)
                rows[w_next][0] = s
                w_next += 1
            else:
                _ensure_row(b_next)
                rows[b_next][1] = s
                b_next += 1
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

    # For drops, place the "Wins" text alongside the drop marker in
    # the same row so gnubg reads it correctly.
    drop_wins_separate = False
    if record.ended_by_drop and rows:
        last = rows[-1]
        if last[0] and not last[1]:
            last[1] = f" {wins_str}"
        elif not last[0] and last[1]:
            last[0] = f" {wins_str}"
        elif last[0] and last[1]:
            last[1] = last[1] + f"  {wins_str}"
        else:
            drop_wins_separate = True

    for idx, (w, b) in enumerate(rows):
        num = f"{idx + 1:>3d})"
        if b:
            lines.append(f"{num} {w:<33s}{b}")
        else:
            lines.append(f"{num} {w}")

    if not record.ended_by_drop or drop_wins_separate:
        if winner == BLACK:
            lines.append(f"                                  {wins_str}")
        else:
            lines.append(f"      {wins_str}")
    lines.append("")

    return "\n".join(lines)


# ── gnubg analysis ───────────────────────────────────────────────────────────

def _write_gnubg_script(mat_files: List[str], script_path: str):
    """Write a Python script for gnubg's embedded interpreter.

    The script imports each .mat file, analyzes it, and prints statistics.
    We look for mEMG in the output.
    """
    lines = [
        "import gnubg",
        "",
        "# Use 2-ply for accurate analysis (0 = fast but shallow)",
        "gnubg.command('set analysis chequerplay evaluation plies 2')",
        "",
    ]

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
) -> List[Tuple[float, float]]:
    """Run gnubg in batch mode on a list of .mat files.

    Returns a list of (white_mEMG, black_mEMG) tuples, one per game.
    mEMG = millipoints Error per Move in Equivalent Money Game.
    """
    if gnubg_cmd is None:
        gnubg_cmd = GNUBG_CMD

    # Use absolute paths so gnubg can find everything regardless of cwd.
    abs_work_dir = os.path.abspath(work_dir)
    abs_mat_files = [os.path.abspath(f) for f in mat_files]
    script_path = os.path.join(abs_work_dir, "_gnubg_analyze.py")
    _write_gnubg_script(abs_mat_files, script_path)

    # Run gnubg with:
    #   -t : text-only (no GUI)
    #   -q : quiet (suppress sounds/splash)
    #   -p : run Python script
    cmd = [gnubg_cmd, "-t", "-q", "-p", script_path]

    if verbose:
        print(f"  gnubg command: {' '.join(cmd)}")
        print(f"  working dir:   {work_dir}")

    # Scale timeout with number of games (~3s per game + 120s base for gnubg startup)
    timeout_secs = 120 + len(mat_files) * 3
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_secs,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"gnubg not found at '{gnubg_cmd}'. "
            f"Install gnubg or set GNUBG_CMD in gnubg_eval.py."
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"gnubg timed out after {timeout_secs}s analyzing "
            f"{len(mat_files)} games."
        )

    # Dump raw output for debugging
    output = result.stdout + result.stderr
    if verbose:
        # Show all mEMG lines so we can see what gnubg actually prints
        print("  --- gnubg mEMG lines ---")
        for line in output.splitlines():
            if "mEMG" in line:
                print(f"  {line.rstrip()}")
        print("  --- end mEMG lines ---")

    # Parse mEMG lines from output.
    #
    # Per game, gnubg prints 3 "Error rate mEMG" lines (all with both players):
    #   1) Chequer play:  Error rate mEMG (MWC)  -201.4  (-10.069%)  -183.1  (-9.153%)
    #   2) Cube decisions: Error rate mEMG (MWC)   -0.0  ( -0.000%)    -0.0  (-0.000%)
    #   3) Overall total: Error rate mEMG (MWC)  -201.4  (-10.069%)  -183.1  (-9.153%)
    #
    # Since we play without the doubling cube, lines 2 and 3 are redundant.
    # We collect all 3, then take every 3rd (the overall) per game.
    all_error_lines = []

    for line in output.splitlines():
        if "mEMG" not in line or "Error rate" not in line:
            continue

        # Extract the two mEMG values: the numbers immediately
        # BEFORE each parenthesized companion value.
        # Match-play (MWC) format: "-201.4   (-10.069%)" -> 201.4
        # Cubeful money (Points) format: "-4.0   ( -0.008)" -> 4.0
        mEMG_values = re.findall(
            r"([-+]?\d+\.?\d*)\s*\(\s*[-+]?\d+\.\d+%?\s*\)", line
        )
        if len(mEMG_values) >= 2:
            white_err = abs(float(mEMG_values[0]))
            black_err = abs(float(mEMG_values[1]))
            all_error_lines.append((white_err, black_err))

    # Take every 3rd line (index 0, 3, 6, ...) = the chequer play line per game.
    # (Could also use index 2, 5, 8, ... for overall, but they're identical
    # since cube errors are always 0 in our games.)
    error_rates = [all_error_lines[i] for i in range(0, len(all_error_lines), 3)]

    return error_rates


# ── Parallel game generation ──────────────────────────────────────────────────

def _play_games_worker(args):
    """Worker function for parallel game generation.
    Each worker loads the model once and plays many games."""
    if len(args) == 4:
        model_path, num_games, work_dir, start_idx = args
        cubeful = False
        jacoby = True
    else:
        model_path, num_games, work_dir, start_idx, cubeful, jacoby = args

    os.environ["OMP_NUM_THREADS"] = "1"
    from model import TDNetwork
    from td_agent import TDAgent

    net = TDNetwork.load(model_path)
    agent = TDAgent(net)

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
    mat_files, work_dir, gnubg_cmd, chunk_id = args
    error_rates = run_gnubg_analysis(
        mat_files, work_dir, gnubg_cmd=gnubg_cmd, verbose=False,
    )
    if len(error_rates) != len(mat_files):
        print(f"  WARNING: chunk {chunk_id}: gnubg returned {len(error_rates)}/{len(mat_files)} results",
              flush=True)
    return error_rates


# ── High-level evaluation ────────────────────────────────────────────────────

def evaluate_with_gnubg(
    agent_white,
    agent_black,
    num_games: int = 10,
    gnubg_cmd: Optional[str] = None,
    work_dir: Optional[str] = None,
    keep_files: bool = False,
    verbose: bool = True,
    cubeful: bool = False,
    jacoby: bool = True,
) -> dict:
    """Play games, export to .mat, analyze with gnubg, return error rates.

    Args:
        agent_white:  Agent playing WHITE.
        agent_black:  Agent playing BLACK.
        num_games:    Number of evaluation games to play.
        gnubg_cmd:    Path to gnubg executable (None = use default).
        work_dir:     Directory for temporary .mat files (None = auto).
        keep_files:   If True, don't delete .mat files after analysis.
        verbose:      Print progress.

    Returns:
        dict with keys:
            'white_mEMG':  list of per-game mEMG for white
            'black_mEMG':  list of per-game mEMG for black
            'avg_mEMG':    overall average mEMG across both players
            'num_games':   number of games analyzed
    """
    # Create work directory
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="gnubg_eval_")
    os.makedirs(work_dir, exist_ok=True)

    if verbose:
        print(f"Playing {num_games} evaluation games...")

    # Play and record games
    records = []
    mat_files = []
    for i in range(num_games):
        if cubeful:
            record = play_and_record_cubeful(
                agent_white, agent_black, jacoby=jacoby,
            )
        else:
            record = play_and_record(agent_white, agent_black)
        records.append(record)

        # Export to .mat
        mat_content = export_mat(record, game_id=i + 1, money_game=cubeful)
        mat_path = os.path.join(work_dir, f"game_{i+1}.mat")
        with open(mat_path, "w") as f:
            f.write(mat_content)
        mat_files.append(mat_path)

        if verbose and (i + 1) % max(1, num_games // 5) == 0:
            print(f"  Recorded game {i+1}/{num_games}")

    if verbose:
        print(f"Exported {num_games} .mat files to {work_dir}")
        print("Running gnubg analysis...")

    # Run gnubg
    error_rates = run_gnubg_analysis(
        mat_files, work_dir, gnubg_cmd=gnubg_cmd, verbose=verbose,
    )

    # Summarize results
    white_errors = [er[0] for er in error_rates]
    black_errors = [er[1] for er in error_rates]
    all_errors = white_errors + black_errors

    results = {
        "white_mEMG": white_errors,
        "black_mEMG": black_errors,
        "avg_mEMG": sum(all_errors) / len(all_errors) if all_errors else None,
        "num_games": len(error_rates),
        "num_games_requested": num_games,
        "work_dir": work_dir,
    }

    if verbose:
        if error_rates:
            print(f"\n{'='*50}")
            print(f"  gnubg analysis: {len(error_rates)} games")
            print(f"{'='*50}")
            print(f"  White avg mEMG: {sum(white_errors)/len(white_errors):.1f}")
            print(f"  Black avg mEMG: {sum(black_errors)/len(black_errors):.1f}")
            print(f"  Overall avg:    {results['avg_mEMG']:.1f}")
            print(f"{'='*50}")
        else:
            print("  WARNING: no error rates returned from gnubg.")
            print("  Check that gnubg is installed and GNUBG_CMD is correct.")

    # Cleanup temp files
    if not keep_files:
        for f in mat_files:
            try:
                os.remove(f)
            except OSError:
                pass
        script_path = os.path.join(work_dir, "_gnubg_analyze.py")
        try:
            os.remove(script_path)
        except OSError:
            pass
        try:
            os.rmdir(work_dir)
        except OSError:
            pass  # directory not empty or other issue

    return results


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
    if gnubg_cmd is None:
        gnubg_cmd = GNUBG_CMD

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
                    (model_path, n, work_dir, start_idx, cubeful, jacoby)
                )
                start_idx += n

        ctx = mp.get_context('spawn')
        with ctx.Pool(len(games_per_worker)) as pool:
            results = pool.map(_play_games_worker, games_per_worker)
        mat_files = [f for chunk in results for f in chunk]
    else:
        from model import TDNetwork
        from td_agent import TDAgent
        net = TDNetwork.load(model_path)
        agent = TDAgent(net)

        mat_files = []
        for i in range(num_games):
            if cubeful:
                record = play_and_record_cubeful(agent, agent, jacoby=jacoby)
            else:
                record = play_and_record(agent, agent)
            mat_content = export_mat(record, game_id=i + 1, money_game=cubeful)
            mat_path = os.path.join(work_dir, f"game_{i+1}.mat")
            with open(mat_path, "w") as f:
                f.write(mat_content)
            mat_files.append(mat_path)

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
        chunks.append((chunk, chunk_dir, gnubg_cmd, len(chunks)))

    if gnubg_workers > 1 and len(chunks) > 1:
        ctx = mp.get_context('fork')  # fork is fine for gnubg subprocess calls
        with ctx.Pool(min(gnubg_workers, len(chunks))) as pool:
            chunk_results = pool.map(_gnubg_worker, chunks)
        all_error_rates = [er for chunk in chunk_results for er in chunk]
    else:
        all_error_rates = []
        for chunk_args in chunks:
            all_error_rates.extend(_gnubg_worker(chunk_args))

    t_analyze = time.perf_counter() - t1

    # ── Summarize ──
    white_errors = [er[0] for er in all_error_rates]
    black_errors = [er[1] for er in all_error_rates]
    all_errors = white_errors + black_errors

    results = {
        "white_mEMG": white_errors,
        "black_mEMG": black_errors,
        "avg_mEMG": sum(all_errors) / len(all_errors) if all_errors else None,
        "num_games": len(all_error_rates),
        "num_games_requested": num_games,
        "work_dir": work_dir,
    }

    if verbose:
        if all_error_rates:
            n_analyzed = len(all_error_rates)
            n_total = len(mat_files)
            print(f"  Analysis: {n_analyzed}/{n_total} games in {t_analyze:.1f}s "
                  f"({n_analyzed/t_analyze:.1f} games/sec)")
            if n_analyzed < n_total:
                print(f"  WARNING: {n_total - n_analyzed} games failed gnubg analysis")
            print(f"\n{'='*50}")
            print(f"  gnubg analysis: {n_analyzed} games")
            print(f"{'='*50}")
            print(f"  White avg mEMG: {sum(white_errors)/len(white_errors):.1f}")
            print(f"  Black avg mEMG: {sum(black_errors)/len(black_errors):.1f}")
            print(f"  Overall avg:    {results['avg_mEMG']:.1f}")
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


# ── Lightweight random agent (no torch dependency) ──────────────────────────

class _RandomAgent:
    """Picks a random legal play.  No torch dependency."""
    def choose_checker_action(self, state, dice, plays):
        return random.choice(plays)


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
    parser.add_argument("--export-only", action="store_true",
                        help="Only export .mat files, don't run gnubg")
    parser.add_argument("--self-play", action="store_true",
                        help="Self-play (model vs itself) — this is the default when --model is used")
    parser.add_argument("--vs-random", action="store_true",
                        help="Play model vs random opponent instead of self-play")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel game-playing workers (default: 1)")
    parser.add_argument("--gnubg-workers", type=int, default=4,
                        help="Parallel gnubg analysis workers (default: 4)")
    parser.add_argument("--gnubg-chunk-size", type=int, default=50,
                        help="Games per gnubg analysis chunk (default: 50)")
    parser.add_argument("--cubeful", action="store_true",
                        help="Cubeful money game. Model must be a cubeful "
                             "agent (cubeful_perspective196 encoder).")
    parser.add_argument("--jacoby", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Jacoby rule for cubeful (default: on).")
    args = parser.parse_args()

    if args.random:
        # No torch needed — pure random agents
        agent_white = _RandomAgent()
        agent_black = _RandomAgent()
        print("Mode: random vs random (no model)")
    elif args.model:
        from td_agent import TDAgent
        from model import TDNetwork
        print(f"Loading model: {args.model}")
        net = TDNetwork.load(args.model)
        agent_white = TDAgent(net)

        if args.vs_random:
            agent_black = _RandomAgent()
            print("Mode: model (WHITE) vs random (BLACK)")
        else:
            agent_black = TDAgent(net)
            print("Mode: self-play (model vs itself)")
    else:
        parser.error("Either --model or --random is required")

    if args.export_only:
        # Just export .mat files without running gnubg
        work_dir = args.work_dir or "gnubg_games"
        os.makedirs(work_dir, exist_ok=True)
        print(f"Playing and exporting {args.games} games to {work_dir}/")

        for i in range(args.games):
            record = play_and_record(agent_white, agent_black)
            mat_content = export_mat(record, game_id=i + 1)
            mat_path = os.path.join(work_dir, f"game_{i+1}.mat")
            with open(mat_path, "w") as f:
                f.write(mat_content)
            print(f"  Exported game {i+1}: "
                  f"{'WHITE' if record.winner == WHITE else 'BLACK'} wins "
                  f"in {len(record.moves)} moves")

        print(f"\nDone. Import these into gnubg manually with:")
        print(f'  gnubg -> File -> Import -> .mat match')
    elif args.workers > 1 or args.self_play:
        # Use parallel version for self-play with workers
        if not args.model:
            parser.error("--workers requires --model")
        results = evaluate_with_gnubg_parallel(
            model_path=args.model,
            num_games=args.games,
            gnubg_cmd=args.gnubg,
            work_dir=args.work_dir,
            keep_files=args.keep_files,
            workers=args.workers,
            gnubg_workers=args.gnubg_workers,
            gnubg_chunk_size=args.gnubg_chunk_size,
            cubeful=args.cubeful,
            jacoby=args.jacoby,
        )
    else:
        results = evaluate_with_gnubg(
            agent_white, agent_black,
            num_games=args.games,
            gnubg_cmd=args.gnubg,
            work_dir=args.work_dir,
            keep_files=args.keep_files,
            cubeful=args.cubeful,
            jacoby=args.jacoby,
        )
