"""Real gnubg 1.08 as a money-play opponent, via its external-player
interface.

Each agent owns a `gnubg -t` subprocess running `external
localhost:<port>` and sends it FIBS boardstates over TCP loopback. Money
play = match length 9999. Rules go with every request (`jacobyrule
on|off crawfordrule off beavers off`). GNUBG_CMD overrides the binary
path.

Two search knobs:

* ``prune`` (default True): gnubg's pruning nets for opponent replies.
* ``movefilter`` (default None = gnubg's shipped Normal filter): a
  preset name from ``_MOVEFILTERS`` or ``"all"`` (every legal move
  deep-evaluated).

The named H-H specs:

* ``gnubg-ext``: ``prune=True, movefilter=None``: gnubg as shipped.
* ``gnubg-ext-full``: ``prune=False, movefilter="all"``: fully
  unpruned.
* ``gnubg-ext-rootfull``: ``prune=True, movefilter="all"``: every
  root move evaluated, shipped pruning for the replies.

``full=True`` / ``root_full=True`` are aliases for the last two.

Cubeless sessions also send `set evaluation chequer evaluation cubeful
off`: the "Cubeful evaluations" flag survives `set cube use off`, and
without it moves are ranked by cubeful equity in cubeless games.

Field layout follows gnubg 1.08.003 (external.c / external_y.y).

A reply the agent cannot use (a move string matching no legal
successor, an empty move with legal plays, a cube reply other than
double/roll or take/drop) raises :class:`GnubgReplyError` with the
board line and reply. There is no fallback move or default decision.
So does a reply that takes longer than GNUBG_EXT_TIMEOUT seconds
(default 300). Cubes above 4096 are sent as 4096 (gnubg's 32-bit
parse; money decisions are per unit): each agent counts them in
``cube_clamps`` and logs each to stderr.
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

GNUBG_CMD = os.environ.get("GNUBG_CMD", "/usr/games/gnubg")
GNUBG_EXT_TIMEOUT = float(os.environ.get("GNUBG_EXT_TIMEOUT", "300"))

# PR_SET_PDEATHSIG: the kernel kills gnubg when its owning worker dies.
# libc is resolved at import: preexec_fn runs between fork and exec.
if sys.platform.startswith("linux"):
    import ctypes

    _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
    _PR_SET_PDEATHSIG = 1

    def _kill_with_parent():
        _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
else:  # non-Linux: no pdeathsig; atexit remains the only cleanup
    _kill_with_parent = None


def _mover_arrays(state: BoardState, mover: int):
    """(mover_pts[1..24], mover_bar, opp_pts[1..24], opp_bar), each side
    in its own point numbering (1 = deepest home point)."""
    mine = [0] * 25   # index 1..24
    theirs = [0] * 25
    for i in range(24):
        v = state.points[i]
        if v == 0:
            continue
        owner = WHITE if v > 0 else BLACK
        cnt = abs(v)
        # White's point number is i+1. Black's is 24-i.
        wpt, bpt = i + 1, 24 - i
        if owner == mover:
            mine[wpt if mover == WHITE else bpt] += cnt
        else:
            theirs[wpt if owner == WHITE else bpt] += cnt
    return mine, state.bar[mover], theirs, state.bar[1 - mover]


def fibs_board_ints(state: BoardState, mover: int):
    """26 ints in FIBS layout (direction=-1, home=0, bar=25): [p] is the
    mover's checkers (+) on the mover's point p or the opponent's (-) on
    the opponent's point 25-p; [25] = mover bar (+), [0] = opp bar (-)."""
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


# gnubg GUI move-filter presets, level-0 row: (extra moves, equity
# window). Each keeps up to N moves within the window. "all" = keep-all.
_MOVEFILTERS = {
    "tiny": (5, 0.08), "narrow": (8, 0.12), "normal": (8, 0.16),
    "large": (16, 0.32), "huge": (20, 0.44),
}


def _legacy_search_kwargs(prune, movefilter, full, root_full):
    """Translate the ``full``/``root_full`` aliases to (prune, movefilter).
    They may not be combined with each other or with explicit values."""
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
    """Kill orphaned gnubg externals (ppid==1, older than min_age_s; the
    age guard avoids racing pdeathsig). Runs once per process, at first
    server construction. Returns the number killed."""
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


class GnubgReplyError(RuntimeError):
    """gnubg's external reply cannot be applied. The message holds the
    request line (replayable against `gnubg -t` + `external`), the
    position, the dice and the reply."""


class _ExtServer:
    """One gnubg subprocess serving `external localhost:<port>`."""

    def __init__(self, plies: int, gnubg_cmd: str = GNUBG_CMD,
                 cubeless: bool = False, prune: bool = True,
                 movefilter: str = None, timeout: float = None):
        reap_orphaned_gnubg()
        self.timeout = GNUBG_EXT_TIMEOUT if timeout is None else timeout
        self.plies = plies
        self.cubeless = cubeless
        self.prune = prune
        self.movefilter = movefilter
        self.proc = None
        self.sock = None
        self.fh = None
        extra = ""
        if not prune:
            extra += ("set evaluation chequer evaluation prune off\n"
                      "set evaluation cubedecision evaluation prune off\n")
        if movefilter is not None:
            # Level-0 row per depth; deeper rows keep gnubg's defaults.
            for ply in (1, 2, 3, 4):
                if movefilter == "all":
                    extra += f"set evaluation movefilter {ply} 0 1000 0 10.0\n"
                else:
                    n, tol = _MOVEFILTERS[movefilter]
                    extra += f"set evaluation movefilter {ply} 0 0 {n} {tol}\n"
        # gnubg's output is discarded below, so check the bearoff
        # databases before spawning (and outside the retry loop).
        from third_party_opponents.gnubg.gnubg_eval import assert_gnubg_bearoff
        assert_gnubg_bearoff(gnubg_cmd)
        last_err = None
        for _attempt in range(5):
            port = self._free_port()
            cmds = (
                "set threads 1\n"
                f"set evaluation chequer evaluation plies {plies}\n"
                f"set evaluation cubedecision evaluation plies {plies}\n"
                + extra
                # Cubeless: cube off and cubeful move criterion off.
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
                self.sock.settimeout(self.timeout)
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
        try:
            reply = self.fh.readline()
        except socket.timeout:
            raise GnubgReplyError(
                f"gnubg-ext: no reply within {self.timeout:g} s "
                f"(GNUBG_EXT_TIMEOUT), request {line!r}") from None
        if reply == "":
            raise RuntimeError("gnubg external connection closed")
        return reply.strip()

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
    """gnubg 1.08 (external interface) as a money opponent.

    plies: evaluation plies for both chequer and cube decisions.
    cubeless: cube disabled; moves ranked by cubeless equity.
    prune: gnubg's pruning nets (True = as shipped).
    movefilter: None = shipped Normal filter; a ``_MOVEFILTERS`` preset
        name; or "all" = every legal root move deep-evaluated.
    full / root_full: aliases for (prune=False, movefilter="all") and
        (prune=True, movefilter="all"); see the module docstring.
    """

    def __init__(self, plies: int = 0, gnubg_cmd: str = GNUBG_CMD,
                 cubeless: bool = False, prune: bool = True,
                 movefilter: str = None, full: bool = False,
                 root_full: bool = False):
        self.plies = plies
        self.gnubg_cmd = gnubg_cmd
        self.cubeless = cubeless
        self.prune, self.movefilter = _legacy_search_kwargs(
            prune, movefilter, full, root_full)
        self._server = None  # lazy: one subprocess per worker process
        self.cube_clamps = 0

    # ── plumbing ─────────────────────────────────────────────────────
    def _srv(self) -> _ExtServer:
        if self._server is None:
            self._server = _ExtServer(self.plies, self.gnubg_cmd,
                                      self.cubeless, self.prune,
                                      self.movefilter)
        return self._server

    @staticmethod
    def _rules(match_state) -> str:
        jac = "on" if getattr(match_state, "jacoby", True) else "off"
        return f"jacobyrule {jac} crawfordrule off beavers off"

    def _board_line(self, state, mover, match_state, dice, doubled):
        b = fibs_board_ints(state, mover)
        persp = cube_perspective(match_state.cube_owner, mover)
        may_i = 1 if persp in (CubePerspective.CENTERED, CubePerspective.MINE) else 0
        may_o = 1 if persp in (CubePerspective.CENTERED, CubePerspective.THEIRS) else 0
        d1, d2 = dice if dice else (0, 0)
        cube = int(match_state.cube_value)
        if cube > 4096:
            self.cube_clamps += 1
            print(f"[gnubg_external] cube {cube} sent as 4096 (clamp "
                  f"#{self.cube_clamps} in pid {os.getpid()}), dice {dice}, "
                  f"points={list(state.points)} bar={list(state.bar)} "
                  f"off={list(state.off)} turn={state.turn}",
                  file=sys.stderr, flush=True)
        # FIBS 'turn': 1 = the side we encode for is on turn, -1 = the
        # opponent. A double RESPONSE happens on the doubler's turn, so
        # doubled requests must send -1.
        turn = -1 if doubled else 1
        fields = (
            ["board", "gnubg", "opp", "9999", "0", "0"]
            + [str(x) for x in b]
            + [str(turn), str(d1), str(d2), "0", "0",
               # 4096 clamp: see the module docstring.
               str(min(cube, 4096)), str(may_i), str(may_o),
               str(1 if doubled else 0),
               "1", "-1", "0", "25", "0", "0", "0", "0", "2", "0", "0", "0"]
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
        """Apply gnubg's move string to a copy of the FIBS array. Returns
        the result (mover perspective) or None on parse failure."""
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

    @staticmethod
    def _bad_reply(what, line, state, dice, reply):
        raise GnubgReplyError(
            f"gnubg-ext: {what}; reply {reply!r}, dice {dice}, position "
            f"points={list(state.points)} bar={list(state.bar)} "
            f"off={list(state.off)} turn={state.turn}, request {line!r}")

    def _match_successor(self, state, dice, move_str, line=""):
        """Map gnubg's move string to one of our legal successors by
        comparing final FIBS arrays. Empty string with no legal play:
        None. Raises GnubgReplyError on anything else that does not
        match exactly one successor."""
        mover = state.turn
        plays = get_legal_plays(state, dice)
        if not plays:
            if move_str:
                self._bad_reply("move with no legal play", line, state,
                                dice, move_str)
            return None
        if not move_str:
            self._bad_reply("no move with legal plays", line, state, dice,
                            move_str)
        target = self._apply_move(fibs_board_ints(state, mover), move_str)
        if target is not None:
            for _play, nxt in plays:
                if fibs_board_ints(nxt, mover) == target:
                    return nxt
        self._bad_reply("move matches no legal successor", line, state,
                        dice, move_str)

    # ── Agent interface ──────────────────────────────────────────────
    def choose_checker_action(self, state, dice, plays):
        line = self._board_line(state, state.turn, _MoneyCubeless(), dice, False)
        reply = self._srv().request(line)
        nxt = self._match_successor(state, dice, reply, line)
        if nxt is None:
            self._bad_reply("caller passed plays but none are legal", line,
                            state, dice, reply)
        return None, nxt

    def choose_checker_action_cubeful(self, state, dice, match_state):
        line = self._board_line(state, state.turn, match_state, dice, False)
        reply = self._srv().request(line)
        nxt = self._match_successor(state, dice, reply, line)
        return (nxt, 0.0) if nxt is not None else None   # None = dance

    def offer_double(self, state, match_state) -> CubeOffer:
        line = self._board_line(state, state.turn, match_state, None, False)
        reply = self._srv().request(line)
        if reply.lower() not in ("double", "roll"):
            self._bad_reply("cube offer reply is not double/roll", line,
                            state, None, reply)
        return CubeOffer(should_double=(reply.lower() == "double"))

    def respond_to_double(self, state, match_state, hint=None) -> bool:
        # state.turn is the DOUBLER. gnubg responds from the other side.
        responder = 1 - state.turn
        line = self._board_line(state, responder, match_state, None, True)
        reply = self._srv().request(line)
        # 'beaver' can't occur (beavers off).
        if reply.lower() not in ("take", "drop"):
            self._bad_reply("double response is not take/drop", line,
                            state, None, reply)
        return reply.lower() == "take"


class _MoneyCubeless:
    """Stand-in match_state for cubeless checker calls; the server has
    `set cube use off`, so these values are inert."""
    cube_owner = CubeOwner.CENTERED
    cube_value = 1
    jacoby = False
