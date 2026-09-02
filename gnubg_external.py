"""True gnubg 1.08 money-play opponent via the external-player interface.

Each agent owns a `/usr/games/gnubg -t` subprocess running `external
localhost:<port>` and queries it with FIBS boardstates over TCP loopback.
This is the DESKTOP program: mainline nets, native money cube decisions
(`GeneralCubeDecision`/`FindBestMove`), native Jacoby — none of the
gnubg-nn caveats (21pt-match cube approximation, Jacoby override,
cube_plies>=1 clamp). Money is selected with match length 9999; rules are
sent per-request (`jacobyrule on|off crawfordrule off beavers off`).

N-POINT MATCH PLAY is supported: pass a MatchState with match_length > 0
and the board line carries real match length, points-scored, and the
post-Crawford flag, with `crawfordrule on`. Unlike the gnubg-nn wrapper
this gives SCORE-AWARE CHEQUER play (verified: initial position, dice
6-2, money plays `24/18 13/11` but DMP plays `24/18 18/16`) and gnubg's
native match cube handling. `met_path` pins the match equity table so
the referee and the agent under test share one; gnubg's default is
already Kazaross-XG2.

Search configuration is composable via two orthogonal knobs:

* ``prune`` (default True): gnubg's pruning neural nets (the small nets
  that prune opponent-reply enumeration inside deep evaluations).
* ``movefilter`` (default None = gnubg's shipped Normal filter, keep 0
  forced + up to 8 more within 0.16): a GUI preset name from
  ``_MOVEFILTERS`` (``tiny``/``narrow``/``normal``/``large``/``huge``)
  or ``"all"`` for keep-all (every legal move deep-evaluated).

The three named H-H specs map onto these:

* ``gnubg-ext`` (production): ``prune=True, movefilter=None`` — gnubg
  exactly as shipped.
* ``gnubg-ext-full``: ``prune=False, movefilter="all"`` — "totally
  unpruned", the full-width search of the historical gnubg-nn wrapper;
  gnubg's most thorough (and ~4x slowest) 1-ply.
* ``gnubg-ext-rootfull``: ``prune=True, movefilter="all"`` — every legal
  move evaluated at the root, shipped pruning for the replies; the
  paper's stated matched-2-ply protocol.

``full=True`` / ``root_full=True`` remain as convenience aliases for the
last two and translate onto the composable pair.

Cubeless sessions also send `set evaluation chequer evaluation cubeful
off`: gnubg's "Cubeful evaluations" flag survives `set cube use off`,
and without this the opponent ranks moves by cubeful money equity in
cubelessly-scored games — a mis-specified objective measured at ~11
mEq/game (regression-tested in tests/test_gnubg_external.py).

Protocol/field layout follows gnubg 1.08.003 (gnubg git external.c /
external_y.y). GNUBG_CMD overrides the binary path.
"""
import atexit
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

from agents import Agent, CubeOffer
from backgammon_engine import BoardState, get_legal_plays, WHITE, BLACK
from modes import CubeOwner, cube_perspective, CubePerspective

GNUBG_CMD = os.environ.get(
    "GNUBG_CMD",
    "/usr/local/bin/gnubg" if os.path.exists("/usr/local/bin/gnubg")
    else "/usr/games/gnubg",
)

# PR_SET_PDEATHSIG: kernel kills gnubg whenever its owning worker dies.
# Needed because Pool.terminate() SIGTERMs workers and atexit does not run
# on SIGTERM, which orphaned one gnubg per worker. libc resolved at import
# because preexec_fn runs between fork and exec (no fresh allocation there).
if sys.platform.startswith("linux"):
    import ctypes

    _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
    _PR_SET_PDEATHSIG = 1

    def _kill_with_parent():
        _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
else:  # non-Linux: no pdeathsig; atexit remains the only cleanup
    _kill_with_parent = None


def _mover_arrays(state: BoardState, mover: int):
    """(mover_pts[1..24], mover_bar, opp_pts[1..24], opp_bar) with each
    side's points numbered from its own bear-off direction (1 = home)."""
    mine = [0] * 25   # index 1..24
    theirs = [0] * 25
    for i in range(24):
        v = state.points[i]
        if v == 0:
            continue
        owner = WHITE if v > 0 else BLACK
        cnt = abs(v)
        # White's point number is i+1; Black's is 24-i.
        wpt, bpt = i + 1, 24 - i
        if owner == mover:
            mine[wpt if mover == WHITE else bpt] += cnt
        else:
            theirs[wpt if owner == WHITE else bpt] += cnt
    return mine, state.bar[mover], theirs, state.bar[1 - mover]


def fibs_board_ints(state: BoardState, mover: int):
    """26 ints, FIBS layout for direction=-1/home=0/bar=25: index p holds
    the mover's checkers (+) on the mover's point p, or the opponent's
    (-) on the opponent's point 25-p; [25]=mover bar (+), [0]=opp bar (-)."""
    mine, my_bar, theirs, opp_bar = _mover_arrays(state, mover)
    b = [0] * 26
    for p in range(1, 25):
        if mine[p]:
            b[p] = mine[p]
        opp_p = 25 - p
        if theirs[opp_p]:
            b[p] = -theirs[opp_p]
    b[25] = my_bar
    b[0] = -opp_bar
    return b


# gnubg GUI move-filter presets, level-0 row: (extra moves, equity window).
# All presets keep 0 forced moves and add up to N within the window.
# "all" = keep-all (every legal move deep-evaluated; wrapper-style
# full-width search).
_MOVEFILTERS = {
    "tiny": (5, 0.08), "narrow": (8, 0.12), "normal": (8, 0.16),
    "large": (16, 0.32), "huge": (20, 0.44),
}


def _legacy_search_kwargs(prune, movefilter, full, root_full):
    """Translate the legacy ``full``/``root_full`` booleans onto the
    composable (prune, movefilter) pair. The legacy flags are pure
    sugar: they may not be combined with each other or with non-default
    composable values."""
    if full or root_full:
        if full and root_full:
            raise ValueError("full and root_full are mutually exclusive")
        if prune is not True or movefilter is not None:
            raise ValueError(
                "full/root_full are aliases; don't combine them with "
                "prune/movefilter")
        if full:
            return False, "all"        # unpruned + keep-all
        return True, "all"             # rootfull: keep-all, nets stay on
    if movefilter is not None and movefilter != "all" \
            and movefilter not in _MOVEFILTERS:
        raise ValueError(
            f"unknown movefilter {movefilter!r}; expected 'all' or one of "
            f"{sorted(_MOVEFILTERS)}")
    return prune, movefilter

_REAPER_RAN = False


def reap_orphaned_gnubg(min_age_s: int = 600) -> int:
    """Kill orphaned gnubg externals (ppid==1, older than min_age_s).

    A live server always has its owner as parent, so ppid==1 is unambiguous;
    the age guard avoids racing pdeathsig on a just-orphaned process. Runs
    once per process, at first server construction. Returns kills."""
    global _REAPER_RAN
    if _REAPER_RAN:
        return 0
    _REAPER_RAN = True
    import re
    killed = 0
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,ppid,etimes,args"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return 0
    for line in out.splitlines()[1:]:
        m = re.match(r"\s*(\d+)\s+1\s+(\d+)\s+(\S+)\s+-t\s+-q\s*$", line)
        if not m:
            continue
        pid, age, cmd = int(m.group(1)), int(m.group(2)), m.group(3)
        if cmd.endswith("gnubg") and age >= min_age_s:
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except OSError:
                pass
    if killed:
        print(f"[gnubg_external] reaped {killed} orphaned gnubg "
              f"process(es)", flush=True)
    return killed


class _ExtServer:
    """One gnubg subprocess serving `external localhost:<port>`."""

    def __init__(self, plies: int, gnubg_cmd: str = GNUBG_CMD,
                 cubeless: bool = False, prune: bool = True,
                 movefilter: str = None, met_path: str = None):
        reap_orphaned_gnubg()
        self.plies = plies
        self.cubeless = cubeless
        self.prune = prune
        self.movefilter = movefilter
        self.met_path = met_path
        self.proc = None
        self.sock = None
        self.fh = None
        extra = ""
        if met_path:
            # Two silent-failure modes make a pre-flight mandatory. A
            # nonexistent path makes gnubg fall back to its compiled-in
            # Zadeh table (NOT met/zadeh.xml — they differ), and a file
            # that exists but is malformed SIGSEGVs gnubg, costing ~100s
            # in connect retries before the real error surfaces. gnubg's
            # warning goes to stderr, which we send to DEVNULL. Parsing it
            # ourselves fails fast and cheaply.
            met_path = os.path.abspath(met_path)
            from met import validate_met_file
            # Structural check only: gnubg accepts tables met.py itself
            # refuses to read (per-player post-Crawford rows, parameterised
            # types), and refusing them here would needlessly narrow the
            # choice of referee.
            validate_met_file(met_path)
            self.met_path = met_path
            extra += f"set matchequity {met_path}\n"
        if not prune:
            extra += ("set evaluation chequer evaluation prune off\n"
                      "set evaluation cubedecision evaluation prune off\n")
        if movefilter is not None:
            # Level-0 row per depth; deeper sub-level rows keep gnubg
            # defaults ("all" makes them irrelevant; with prune=True this
            # is the paper's rootfull protocol).
            for ply in (1, 2, 3, 4):
                if movefilter == "all":
                    extra += f"set evaluation movefilter {ply} 0 1000 0 10.0\n"
                else:
                    n, tol = _MOVEFILTERS[movefilter]
                    extra += f"set evaluation movefilter {ply} 0 0 {n} {tol}\n"
        # This Popen discards gnubg's output, so its startup warnings can
        # never be seen -- check the bearoff databases BEFORE spawning.
        # Independent of the port, so it belongs outside the retry loop
        # (inside, a failing probe also leaked the mkstemp script below).
        from gnubg_eval import assert_gnubg_bearoff
        assert_gnubg_bearoff(gnubg_cmd)
        last_err = None
        for _attempt in range(5):
            port = self._free_port()
            cmds = (
                "set threads 1\n"
                f"set evaluation chequer evaluation plies {plies}\n"
                f"set evaluation cubedecision evaluation plies {plies}\n"
                + extra
                # Cubeless: kill the cube AND switch the move criterion to
                # cubeless equity — "Cubeful evaluations" stays on through
                # `set cube use off` and would otherwise pick moves by
                # cubeful money equity in cubelessly-scored games
                # (~11 mEq/game; see module docstring).
                + ("set cube use off\n"
                   "set evaluation chequer evaluation cubeful off\n"
                   if cubeless else "")
                + "set beavers 0\n"
                f"external localhost:{port}\n"
            )
            fd, script = tempfile.mkstemp(suffix=".gnubg", text=True)
            with os.fdopen(fd, "w") as f:
                f.write(cmds)
            self.proc = subprocess.Popen(
                [gnubg_cmd, "-t", "-q"], stdin=open(script),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=dict(os.environ, LC_ALL="C", OMP_NUM_THREADS="1"),
                preexec_fn=_kill_with_parent,
            )
            os.unlink(script)
            try:
                self.sock = self._connect(port)
                self.fh = self.sock.makefile("rw")
                atexit.register(self.close)
                return
            except OSError as e:
                last_err = e
                self.proc.kill()
                self.proc.wait()      # reap; we retry in a loop
        raise RuntimeError(f"could not start gnubg external server: {last_err}")

    @staticmethod
    def _free_port() -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def _connect(self, port, tries=150, delay=0.1):
        for _ in range(tries):
            if self.proc.poll() is not None:
                raise OSError("gnubg exited during startup")
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(("127.0.0.1", port))
                return s
            except OSError:
                time.sleep(delay)
        raise OSError(f"gnubg external port {port} never came up")

    def request(self, line: str) -> str:
        self.fh.write(line + "\r\n")
        self.fh.flush()
        reply = self.fh.readline()
        if reply == "":
            raise RuntimeError("gnubg external connection closed")
        reply = reply.strip()
        if reply.startswith("Error"):
            # Without this a malformed board reads as "no double" in
            # offer_double and falls through to plays[0][1] in
            # _match_successor — an encoding bug would surface as
            # slightly worse play rather than a failure.
            raise RuntimeError(f"gnubg rejected board line: {reply}\n{line}")
        return reply

    def close(self):
        try:
            if self.fh:
                self.fh.write("exit\r\n")
                self.fh.flush()
        except Exception:
            pass
        for closer in (self.fh, self.sock):
            try:
                closer.close()
            except Exception:
                pass
        if self.proc:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()          # reap before dropping the handle
        self.fh = self.sock = self.proc = None


class GnubgExternalAgent(Agent):
    """gnubg 1.08 (external interface) as a faithful money opponent.

    plies: gnubg evaluation plies for BOTH chequer and cube decisions.
    cubeless: money play with the cube disabled; move selection by
        cubeless equity (see module docstring re the criterion fix).
    prune: gnubg's pruning neural nets (True = production). False drops
        them (full-width deep evaluation inside the search).
    movefilter: None = gnubg's shipped Normal filter; a preset name from
        ``_MOVEFILTERS`` (tiny/narrow/normal/large/huge); or "all" =
        keep-all (every legal root move deep-evaluated).
    full / root_full: legacy aliases. full=True == (prune=False,
        movefilter="all"), the totally-unpruned full-width config
        (~4x slower at 1-ply); root_full=True == (prune=True,
        movefilter="all"), the paper's stated matched-2-ply protocol
        (full-width root, shipped pruning for the replies). They may not
        be combined with each other or with explicit prune/movefilter.
    met_path: pin gnubg's match equity table (match play; see module
        docstring). None keeps gnubg's default (Kazaross-XG2).
    """

    def __init__(self, plies: int = 0, gnubg_cmd: str = GNUBG_CMD,
                 cubeless: bool = False, prune: bool = True,
                 movefilter: str = None, full: bool = False,
                 root_full: bool = False, met_path: str = None):
        self.plies = plies
        self.gnubg_cmd = gnubg_cmd
        self.cubeless = cubeless
        self.prune, self.movefilter = _legacy_search_kwargs(
            prune, movefilter, full, root_full)
        self.met_path = met_path
        self._server = None  # lazy: one subprocess per worker process

    # ── plumbing ─────────────────────────────────────────────────────
    def _srv(self) -> _ExtServer:
        if self._server is None:
            self._server = _ExtServer(self.plies, self.gnubg_cmd,
                                      self.cubeless, self.prune,
                                      self.movefilter, self.met_path)
        return self._server

    @staticmethod
    def _rules(match_state) -> str:
        jac = "on" if getattr(match_state, "jacoby", True) else "off"
        # crawfordrule MUST be on for match play: with it off gnubg forces
        # fCrawford = FALSE regardless of the board fields and never models
        # the Crawford game at all (same line replies "double" with the
        # rule off, "roll" with it on). Kept off for money so money lines
        # stay byte-identical. jacobyrule is inert at a match score
        # (SetCubeInfoMatch clears fJacoby) but stays live for money.
        crawford = "on" if getattr(match_state, "match_length", 0) > 0 else "off"
        return f"jacobyrule {jac} crawfordrule {crawford} beavers off"

    def _board_line(self, state, mover, match_state, dice, doubled):
        b = fibs_board_ints(state, mover)
        persp = cube_perspective(match_state.cube_owner, mover)
        may_i = 1 if persp in (CubePerspective.CENTERED, CubePerspective.MINE) else 0
        may_o = 1 if persp in (CubePerspective.CENTERED, CubePerspective.THEIRS) else 0
        d1, d2 = dice if dice else (0, 0)
        # FIBS 'turn': 1 = gnubg (the mover we encode for) is on turn;
        # -1 = the opponent is. A double RESPONSE happens on the
        # doubler's turn, so doubled requests must send -1 — with 1,
        # gnubg evaluates the take from the wrong side and takes
        # hopeless positions.
        turn = -1 if doubled else 1

        if getattr(match_state, "match_length", 0) > 0:
            L = match_state.match_length
            # gnubg rejects score >= match length with "badly formed
            # board"; away >= 1 makes that unreachable, so assert it.
            assert 1 <= match_state.my_away <= L and 1 <= match_state.opp_away <= L, (
                f"away scores out of range for a {L}-point match: "
                f"{match_state.my_away}/{match_state.opp_away}"
            )
            # gnubg wants POINTS SCORED, not away scores. MatchState
            # stores away in WHITE's view, so index by absolute player.
            pts = (L - match_state.my_away, L - match_state.opp_away)
            # The two score fields are REVERSED relative to their FIBS
            # names: field 5 carries the score of whoever is ON ROLL in
            # the real game, field 4 the other player's. On an undoubled
            # line that is the encoded mover; on a doubled line the board
            # is encoded from the RESPONDER, so the on-roll player is the
            # doubler. Getting this backwards yields plausible-but-wrong
            # play with no error, so it is verified in the test suite at
            # string level for both movers and both line shapes.
            on_roll = (1 - mover) if doubled else mover
            f_len, f4, f5 = str(L), str(pts[1 - on_roll]), str(pts[on_roll])
            # Slot 51 is fPostCrawford. gnubg INFERS the Crawford game
            # itself: crawfordrule on + this slot 0 + either side 1-away
            # => Crawford, cube dead. So 0 is right both pre-Crawford and
            # during Crawford; 1 only once it has been played. Same
            # predicate as met.py's post-Crawford test.
            f_post_crawford = (
                "1" if (match_state.crawford_done
                        and not match_state.is_crawford) else "0"
            )
        else:
            f_len, f4, f5, f_post_crawford = "9999", "0", "0", "0"

        fields = (
            ["board", "gnubg", "opp", f_len, f4, f5]
            + [str(x) for x in b]
            + [str(turn), str(d1), str(d2), "0", "0",
               # clamp: gnubg parses this with a 32-bit atoi; runaway
               # redouble wars can exceed 2**31 (money decisions are
               # per-unit, so the clamp cannot change any decision, and
               # +/-128-capped scoring saturates far below 4096).
               # Unreachable in match play, where cube <= match length.
               str(min(int(match_state.cube_value), 4096)),
               # NEVER emit may_i == may_o == 0: gnubg reinterprets an
               # undoubled line with both flags clear as a TAKE decision
               # (a move request comes back "take"). Dead-cube suppression
               # is gnubg's job given the right score — do not express it
               # here. cube_perspective can never produce 0/0.
               str(may_i), str(may_o),
               str(1 if doubled else 0),
               "1", "-1", "0", "25", "0", "0", "0", "0", "2", "0",
               f_post_crawford, "0"]
        )
        return ":".join(fields) + " " + self._rules(match_state)

    # ── move-string application ──────────────────────────────────────
    @staticmethod
    def _parse_token(tok):
        """'bar/22*', '13/7(2)', '6/off' -> (hops [(src,dst),...], repeat)."""
        rep = 1
        if tok.endswith(")") and "(" in tok:
            tok, n = tok[:-1].rsplit("(", 1)
            rep = int(n)
        parts = [p.rstrip("*") for p in tok.split("/")]
        def conv(p):
            if p == "bar":
                return 25
            if p == "off":
                return 0
            return int(p)
        pts = [conv(p) for p in parts]
        return list(zip(pts[:-1], pts[1:])), rep

    def _apply_move(self, board26, move_str):
        """Apply gnubg's move string to a copy of the FIBS array; returns
        the resulting array (mover perspective) or None on parse failure."""
        b = list(board26)
        try:
            for tok in move_str.split():
                hops, rep = self._parse_token(tok)
                for _ in range(rep):
                    for src, dst in hops:
                        b[src] -= 1
                        if dst == 0:
                            continue  # borne off
                        if b[dst] == -1:      # hit a blot
                            b[dst] = 0
                            b[0] -= 1         # opp checker to opp bar
                        b[dst] += 1
        except (ValueError, IndexError):
            return None
        return b

    def _match_successor(self, state, dice, move_str):
        """Map gnubg's move string to one of our legal successors by
        comparing final FIBS arrays (mover perspective)."""
        mover = state.turn
        plays = get_legal_plays(state, dice)
        if not plays:
            return None
        target = self._apply_move(fibs_board_ints(state, mover), move_str)
        if target is not None:
            for _play, nxt in plays:
                if fibs_board_ints(nxt, mover) == target:
                    return nxt
        # Fallback (never expected): first legal successor.
        return plays[0][1]

    # ── Agent interface ──────────────────────────────────────────────
    def choose_checker_action(self, state, dice, plays):
        reply = self._srv().request(
            self._board_line(state, state.turn, _MoneyCubeless(), dice, False))
        if not reply:
            return None, plays[0][1]
        nxt = self._match_successor(state, dice, reply)
        return None, nxt

    def choose_checker_action_cubeful(self, state, dice, match_state):
        reply = self._srv().request(
            self._board_line(state, state.turn, match_state, dice, False))
        if not reply:
            return None  # dance / no legal move
        nxt = self._match_successor(state, dice, reply)
        return (nxt, 0.0) if nxt is not None else None

    def offer_double(self, state, match_state) -> CubeOffer:
        reply = self._srv().request(
            self._board_line(state, state.turn, match_state, None, False))
        return CubeOffer(should_double=(reply.lower() == "double"))

    def respond_to_double(self, state, match_state, hint=None) -> bool:
        # state.turn is the DOUBLER; gnubg responds from the other side.
        responder = 1 - state.turn
        reply = self._srv().request(
            self._board_line(state, responder, match_state, None, True))
        # 'beaver' can't occur (beavers off) but map it to take defensively.
        return reply.lower() != "drop"


class _MoneyCubeless:
    """Stand-in match_state for cubeless checker calls (server runs
    `set cube use off`, so flags here are inert)."""
    cube_owner = CubeOwner.CENTERED
    cube_value = 1
    jacoby = False
    match_length = 0        # money: selects the 9999 / 0:0 line shape
    my_away = 0
    opp_away = 0
    is_crawford = False
    crawford_done = False
