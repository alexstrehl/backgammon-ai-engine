"""True gnubg 1.08 money-play opponent via the external-player interface.

Each agent owns a `/usr/games/gnubg -t` subprocess running `external
localhost:<port>` and queries it with FIBS boardstates over TCP loopback.
This is the DESKTOP program: mainline nets, native money cube decisions
(`GeneralCubeDecision`/`FindBestMove`), native Jacoby — none of the
gnubg-nn caveats (21pt-match cube approximation, Jacoby override,
cube_plies>=1 clamp). Money is selected with match length 9999; rules are
sent per-request (`jacobyrule on|off crawfordrule off beavers off`).

Two supported configurations (H-H specs `gnubg-ext` / `gnubg-ext-full`):

* production (default): gnubg exactly as shipped — pruning neural nets
  ON, Normal move filter (keep 0 forced + up to 8 more within 0.16).
* full (`full=True`): "totally unpruned" — pruning nets OFF and a
  keep-all move filter, so EVERY legal move gets the deep evaluation.
  This reproduces the full-width search of the historical gnubg-nn
  wrapper (`gnubg` spec) and is gnubg's most thorough
  (and ~4x slowest) 1-ply. Head-to-head the two are statistically tied
  at 1-ply cubeless (direct duel: -0.4 mEq [-1.8, +0.9] at 4M).

Cubeless sessions also send `set evaluation chequer evaluation cubeful
off`: gnubg's "Cubeful evaluations" flag survives `set cube use off`,
and without this the opponent ranks moves by cubeful money equity in
cubelessly-scored games — a mis-specified objective measured at ~11
mEq/game (found 2026-07-20; regression-tested in
tests/test_gnubg_external.py).

Protocol/field layout verified live 2026-07-20 against gnubg 1.08.003
(gnubg git external.c / external_y.y); throughput per process: ~3.4k
moves/s + 12k cube/s at 0-ply, ~390 + 880 at 1-ply (production).
GNUBG_CMD overrides the binary path.
"""
import atexit
import os
import socket
import subprocess
import tempfile
import time

from agents import Agent, CubeOffer
from backgammon_engine import BoardState, get_legal_plays, WHITE, BLACK
from modes import CubeOwner, cube_perspective, CubePerspective

GNUBG_CMD = os.environ.get("GNUBG_CMD", "/usr/games/gnubg")


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


class _ExtServer:
    """One gnubg subprocess serving `external localhost:<port>`."""

    def __init__(self, plies: int, gnubg_cmd: str = GNUBG_CMD,
                 cubeless: bool = False, full: bool = False):
        self.plies = plies
        self.cubeless = cubeless
        self.full = full
        self.proc = None
        self.sock = None
        self.fh = None
        extra = ""
        if full:
            # "Totally unpruned": no pruning nets + keep-all move filter
            # (level-0 row per depth; deeper sub-level rows are irrelevant
            # once every root move is kept).
            extra += ("set evaluation chequer evaluation prune off\n"
                      "set evaluation cubedecision evaluation prune off\n")
            for ply in (1, 2, 3, 4):
                extra += f"set evaluation movefilter {ply} 0 1000 0 10.0\n"
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
                # cubeful money equity in cubelessly-scored games (found
                # 2026-07-20: this mis-specification cost ~11 mEq/game).
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
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.fh = self.sock = self.proc = None


class GnubgExternalAgent(Agent):
    """gnubg 1.08 (external interface) as a faithful money opponent.

    plies: gnubg evaluation plies for BOTH chequer and cube decisions.
    cubeless: money play with the cube disabled; move selection by
        cubeless equity (see module docstring re the criterion fix).
    full: False = production config (pruning nets + Normal move filter,
        gnubg as shipped); True = totally unpruned (no pruning nets,
        keep-all filter — full-width search, ~4x slower at 1-ply).
    """

    def __init__(self, plies: int = 0, gnubg_cmd: str = GNUBG_CMD,
                 cubeless: bool = False, full: bool = False):
        self.plies = plies
        self.gnubg_cmd = gnubg_cmd
        self.cubeless = cubeless
        self.full = full
        self._server = None  # lazy: one subprocess per worker process

    # ── plumbing ─────────────────────────────────────────────────────
    def _srv(self) -> _ExtServer:
        if self._server is None:
            self._server = _ExtServer(self.plies, self.gnubg_cmd,
                                      self.cubeless, self.full)
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
        # FIBS 'turn': 1 = gnubg (the mover we encode for) is on turn;
        # -1 = the opponent is. A double RESPONSE happens on the
        # doubler's turn, so doubled requests must send -1 — with 1,
        # gnubg evaluates the take from the wrong side (found the hard
        # way: 0 drops in 400 games; hopeless positions "taken").
        turn = -1 if doubled else 1
        fields = (
            ["board", "gnubg", "opp", "9999", "0", "0"]
            + [str(x) for x in b]
            + [str(turn), str(d1), str(d2), "0", "0",
               # clamp: gnubg parses this with a 32-bit atoi; runaway
               # redouble wars can exceed 2**31 (money decisions are
               # per-unit, so the clamp cannot change any decision, and
               # +/-128-capped scoring saturates far below 4096).
               str(min(int(match_state.cube_value), 4096)), str(may_i), str(may_o),
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
