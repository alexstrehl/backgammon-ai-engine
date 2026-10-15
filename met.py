"""Match equity tables (MET), parsed from gnubg's XML format.

Why this exists: the matchplay network was originally its own implicit
MET (`TDAgent._eval_new_game`), which made `v_dp` — a quantity with an
exact known value — a *learned* quantity free to drift. It did: in
training, v_dp at 2-away/2-away decayed 0.680 → 0.639 and took the cube
passive with it. A real table pins it.

The tables are gnubg's own XML files (the default one bundled as
met_tables/Kazaross-XG2.xml, the others from gnubg's /usr/share/gnubg/met/),
so the values here are bit-identical to what gnubg scores us against —
provided the evaluator is pinned to the same table. See
`GNUBG_SET_COMMAND`.

Indexing (verified against `gnubg show matchequitytable`):
  pre[i][j]  MWC for the player who is (i+1)-away vs an opponent who
             is (j+1)-away, Crawford game included (the Crawford game
             uses the pre-Crawford table with the cube disabled).
  post[j]    MWC for the player who is (j+1)-away when the opponent is
             1-away and the Crawford game has already been played.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np

MET_DIR = "/usr/share/gnubg/met"

# Prefer the copy bundled with this repo so a fresh box needs no gnubg
# install; fall back to gnubg's system table. The bundled file is a
# byte-identical copy of gnubg-data 1.08.003's table (md5 a436fd95). It
# carries its own all-permissive copying notice -- keep it intact.
_BUNDLED_MET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "met_tables", "Kazaross-XG2.xml",
)
_SYSTEM_MET = os.path.join(MET_DIR, "Kazaross-XG2.xml")
DEFAULT_MET = _BUNDLED_MET if os.path.isfile(_BUNDLED_MET) else _SYSTEM_MET

# Pin the evaluator to the same table. Anything else means training
# against one MET and being scored against another.
GNUBG_SET_COMMAND = "set matchequity {path}"

# Layout of the appended MET feature block. Outcomes are ordered
# (win, lose) x (normal, gammon, backgammon) at a given cube level.
N_MET_FEATURES = 13
MET_FEATURE_NAMES = (
    "nd_win_n", "nd_win_g", "nd_win_bg",
    "nd_lose_n", "nd_lose_g", "nd_lose_bg",
    "dt_win_n", "dt_win_g", "dt_win_bg",
    "dt_lose_n", "dt_lose_g", "dt_lose_bg",
    "dp",
)

# The 13 above plus the six outcome prices at 4 x the current cube: the
# level reached after a double and an immediate recube-and-take, which the
# 13-feature block cannot express. Strict superset, appended at the end, so
# a 350-input net expands to 356 by zero-padding (see surgery_1h64met4.py).
N_MET4_FEATURES = 19

# met6: met4 with the two EXACT DUPLICATES removed. x4_win_n and x4_lose_n
# price 4C, which dt_win_g and dt_lose_g already price -- the block is a
# function of POINTS STAKED, not cube level, so a gammon at 2C and a single
# at 4C are the same MET lookup. Verified identical over 1,694 reachable
# (away, away, crawford, cube) states. Dropping them costs no rank (the
# 19-wide block is rank 13; the 17-wide one is rank 13 too) and removes two
# columns whose gradient is by construction a copy of another column's.
# The two bg columns are CONSTANT at 11pt and are kept deliberately: at
# longer match lengths 12 points stops deciding the match and they separate.
N_MET6_FEATURES = 17
MET6_FEATURE_NAMES = MET_FEATURE_NAMES + (
    "x4_win_g", "x4_win_bg", "x4_lose_g", "x4_lose_bg",
)
MET4_FEATURE_NAMES = MET_FEATURE_NAMES + (
    "x4_win_n", "x4_win_g", "x4_win_bg",
    "x4_lose_n", "x4_lose_g", "x4_lose_bg",
)


class MatchEquityTable:
    """A parsed MET with MWC lookups and cube-branch feature blocks."""

    def __init__(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        name: str,
        length: int,
        path: str,
    ) -> None:
        self._pre = pre
        self._post = post
        self.name = name
        self.length = length
        self.path = path
        # (my_away, opp_away, crawford_done, is_crawford, cube) -> block.
        # The reachable key space is tiny (3-point matches touch a few
        # dozen entries), and this sits in the encode hot path.
        self._block_cache: Dict[Tuple[int, int, bool, bool, int], np.ndarray] = {}
        self._block4_cache: Dict[Tuple[int, int, bool, bool, int], np.ndarray] = {}
        self._block6_cache: Dict[Tuple[int, int, bool, bool, int], np.ndarray] = {}
        self._mwc_cache: Dict[Tuple[int, int, bool, bool], float] = {}

    # ── construction ────────────────────────────────────────────────

    @classmethod
    def from_xml(cls, path: str = DEFAULT_MET) -> "MatchEquityTable":
        root = ET.parse(path).getroot()

        info = root.find("info")
        name = info.findtext("name", default=os.path.basename(path)).strip()
        length = int(info.findtext("length", default="25"))

        pre_el = root.find("pre-crawford-table")
        if pre_el is None or pre_el.get("type") != "explicit":
            raise NotImplementedError(
                f"{path}: only type='explicit' pre-Crawford tables are "
                f"supported (got {None if pre_el is None else pre_el.get('type')!r}). "
                f"Parameterised METs (mec.xml, jacobs.xml) would need the "
                f"generating formula implemented."
            )
        pre = np.array(
            [[float(me.text) for me in row.findall("me")]
             for row in pre_el.findall("row")],
            dtype=np.float64,
        )

        post_els = root.findall("post-crawford-table")
        if not post_els:
            raise ValueError(f"{path}: no post-Crawford table")
        posts = []
        for el in post_els:
            if el.get("type") != "explicit":
                raise NotImplementedError(
                    f"{path}: only type='explicit' post-Crawford tables "
                    f"are supported (got {el.get('type')!r})."
                )
            rows = el.findall("row")
            if len(rows) != 1:
                raise ValueError(
                    f"{path}: post-Crawford table has {len(rows)} rows, "
                    f"expected 1"
                )
            posts.append(
                np.array([float(me.text) for me in rows[0].findall("me")],
                         dtype=np.float64)
            )
        # gnubg permits a separate post-Crawford table per player (free-drop
        # asymmetry). Our encoding is perspective-relative and has no notion
        # of a physical player, so an asymmetric table cannot be represented
        # faithfully. Refuse rather than silently pick one.
        for other in posts[1:]:
            if not np.allclose(posts[0], other):
                raise NotImplementedError(
                    f"{path}: per-player post-Crawford tables differ; this "
                    f"module assumes a symmetric (player='both') table. "
                    f"Use Kazaross-XG2.xml or another symmetric MET."
                )
        return cls(pre, posts[0], name, length, path)

    # ── lookups ─────────────────────────────────────────────────────

    def mwc(
        self,
        my_away: int,
        opp_away: int,
        crawford_done: bool = False,
        is_crawford: bool = False,
    ) -> float:
        """P(the 'my' player wins the match) at the start of a game.

        `crawford_done` latches once the Crawford game has been
        scheduled (mirroring MatchState), so post-Crawford is
        `crawford_done and not is_crawford`.
        """
        if my_away <= 0:
            return 1.0
        if opp_away <= 0:
            return 0.0
        key = (my_away, opp_away, crawford_done, is_crawford)
        hit = self._mwc_cache.get(key)
        if hit is not None:
            return hit
        value = self._mwc_uncached(my_away, opp_away, crawford_done, is_crawford)
        self._mwc_cache[key] = value
        return value

    def _mwc_uncached(
        self, my_away: int, opp_away: int,
        crawford_done: bool, is_crawford: bool,
    ) -> float:
        self._check_range(my_away, opp_away)
        post_crawford = (
            crawford_done and not is_crawford
            and (my_away == 1 or opp_away == 1)
        )
        if post_crawford:
            if opp_away == 1:
                return float(self._post[my_away - 1])
            return 1.0 - float(self._post[opp_away - 1])
        # The Crawford game itself uses the pre-Crawford table; the cube
        # being off is handled by MatchState.can_offer, not by the table.
        return float(self._pre[my_away - 1][opp_away - 1])

    def _check_range(self, my_away: int, opp_away: int) -> None:
        n = self._pre.shape[0]
        if my_away > n or opp_away > n:
            raise ValueError(
                f"away score ({my_away}, {opp_away}) exceeds the "
                f"{n}-point {self.name}"
            )

    def mwc_after(
        self,
        my_away: int,
        opp_away: int,
        crawford_done: bool,
        is_crawford: bool,
        points: int,
        winner_is_me: bool,
    ) -> float:
        """MWC at the start of the NEXT game after someone scores
        `points`. Mirrors MatchState.after_game() + start_game() —
        `tests/test_met.py` asserts the two agree on every reachable
        transition.
        """
        if winner_is_me:
            new_my, new_opp = my_away - points, opp_away
        else:
            new_my, new_opp = my_away, opp_away - points
        if new_my <= 0:
            return 1.0
        if new_opp <= 0:
            return 0.0
        # crawford_done is latched by start_game, but tolerate a state
        # built by hand with is_crawford set and the latch not yet on.
        was_done = crawford_done or is_crawford
        new_crawford = (not was_done) and (new_my == 1 or new_opp == 1)
        return self.mwc(new_my, new_opp, was_done or new_crawford, new_crawford)

    # ── cube-branch feature block ───────────────────────────────────

    def outcome_block(
        self,
        my_away: int,
        opp_away: int,
        crawford_done: bool,
        is_crawford: bool,
        cube_value: int,
    ) -> np.ndarray:
        """The 13 MET features, from the 'my' player's perspective.

          [0:6]   no-double: MWC after each outcome at `cube_value`
          [6:12]  double-take: same at 2 x `cube_value`
          [12]    double-drop: MWC after winning `cube_value` points

        Returned array is cached and MUST NOT be mutated by callers;
        encoders copy it into their own buffer.
        """
        key = (my_away, opp_away, crawford_done, is_crawford, cube_value)
        hit = self._block_cache.get(key)
        if hit is not None:
            return hit

        out = np.empty(N_MET_FEATURES, dtype=np.float32)
        for slot, cube in ((0, cube_value), (6, 2 * cube_value)):
            for i, mult in enumerate((1, 2, 3)):
                out[slot + i] = self.mwc_after(
                    my_away, opp_away, crawford_done, is_crawford,
                    cube * mult, True,
                )
                out[slot + 3 + i] = self.mwc_after(
                    my_away, opp_away, crawford_done, is_crawford,
                    cube * mult, False,
                )
        # Drop: the doubler collects the PRE-double cube value.
        out[12] = self.mwc_after(
            my_away, opp_away, crawford_done, is_crawford, cube_value, True,
        )
        out.flags.writeable = False
        self._block_cache[key] = out
        return out

    def outcome_block_x6(
        self,
        my_away: int,
        opp_away: int,
        crawford_done: bool,
        is_crawford: bool,
        cube_value: int,
    ) -> np.ndarray:
        """The 17 MET features: outcome_block()'s 13, then the four 4x-cube
        prices that are NOT already present.

          [0:13]  exactly outcome_block(...)
          [13]    win  gammon      at 4x cube  ( 8C)
          [14]    win  backgammon  at 4x cube  (12C)
          [15]    lose gammon      at 4x cube  ( 8C)
          [16]    lose backgammon  at 4x cube  (12C)

        The 4x single prices (4C) are omitted: dt_win_g and dt_lose_g are the
        same MET lookup, so including them adds a column whose gradient is a
        copy of another column's and whose information content is nil.

        Returned array is cached and MUST NOT be mutated by callers.
        """
        key = (my_away, opp_away, crawford_done, is_crawford, cube_value)
        hit = self._block6_cache.get(key)
        if hit is not None:
            return hit

        out = np.empty(N_MET6_FEATURES, dtype=np.float32)
        out[:N_MET_FEATURES] = self.outcome_block(
            my_away, opp_away, crawford_done, is_crawford, cube_value,
        )
        cube = 4 * cube_value
        # mult 2 = gammon (8C), mult 3 = backgammon (12C); mult 1 (4C) omitted.
        out[13] = self.mwc_after(my_away, opp_away, crawford_done, is_crawford, cube * 2, True)
        out[14] = self.mwc_after(my_away, opp_away, crawford_done, is_crawford, cube * 3, True)
        out[15] = self.mwc_after(my_away, opp_away, crawford_done, is_crawford, cube * 2, False)
        out[16] = self.mwc_after(my_away, opp_away, crawford_done, is_crawford, cube * 3, False)
        out.flags.writeable = False
        self._block6_cache[key] = out
        return out

    def outcome_block_x4(
        self,
        my_away: int,
        opp_away: int,
        crawford_done: bool,
        is_crawford: bool,
        cube_value: int,
    ) -> np.ndarray:
        """The 19 MET features: outcome_block()'s 13, then six more.

          [0:13]  exactly outcome_block(...) -- same values, same order
          [13:19] MWC after win/lose x normal/gammon/backgammon at
                  4 x `cube_value`

        The 4x level is exactly where a double-and-recube lands. The
        13-feature block stops at 2x, so the price of the recube branch
        was previously only reachable through the 7-hot cube-level input,
        which carries no equity information at all.

        Returned array is cached and MUST NOT be mutated by callers.
        """
        key = (my_away, opp_away, crawford_done, is_crawford, cube_value)
        hit = self._block4_cache.get(key)
        if hit is not None:
            return hit

        out = np.empty(N_MET4_FEATURES, dtype=np.float32)
        out[:N_MET_FEATURES] = self.outcome_block(
            my_away, opp_away, crawford_done, is_crawford, cube_value,
        )
        cube = 4 * cube_value
        for i, mult in enumerate((1, 2, 3)):
            out[N_MET_FEATURES + i] = self.mwc_after(
                my_away, opp_away, crawford_done, is_crawford,
                cube * mult, True,
            )
            out[N_MET_FEATURES + 3 + i] = self.mwc_after(
                my_away, opp_away, crawford_done, is_crawford,
                cube * mult, False,
            )
        out.flags.writeable = False
        self._block4_cache[key] = out
        return out


_DEFAULT: Optional[MatchEquityTable] = None


def default_met() -> MatchEquityTable:
    """Process-wide MET. Override with $BG_MET_PATH."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MatchEquityTable.from_xml(
            os.environ.get("BG_MET_PATH", DEFAULT_MET)
        )
    return _DEFAULT


def set_default_met(met: MatchEquityTable) -> None:
    global _DEFAULT
    _DEFAULT = met


def validate_met_file(path: str) -> None:
    """Cheap structural check that `path` is a MET gnubg can load.

    Deliberately WEAKER than from_xml(): gnubg happily uses tables this
    module refuses to read (per-player post-Crawford tables, parameterised
    types). The only job here is to catch a file that would make gnubg
    fall back silently or crash — a nonexistent path leaves it on its
    compiled-in Zadeh table with the warning on a DEVNULLed stderr, and a
    malformed file SIGSEGVs it.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MET not found: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ValueError(f"{path}: not parseable as XML ({e})") from None
    if root.tag != "met":
        raise ValueError(f"{path}: root element is {root.tag!r}, expected 'met'")
    for required in ("info", "pre-crawford-table", "post-crawford-table"):
        if root.find(required) is None:
            raise ValueError(f"{path}: missing <{required}>")
    pre = root.find("pre-crawford-table")
    if pre.get("type") == "explicit":
        rows = pre.findall("row")
        if not rows or not rows[0].findall("me"):
            raise ValueError(f"{path}: pre-Crawford table has no entries")
        try:
            float(rows[0].findall("me")[0].text)
        except (TypeError, ValueError):
            raise ValueError(f"{path}: pre-Crawford entries are not numeric") from None


def resolve_met_path(spec: Optional[str]) -> Optional[str]:
    """Turn a --met argument into an absolute XML path.

    Accepts 'auto' (the bundled Kazaross-XG2 table, below), a bare table
    name such as 'snowie' (looked up in MET_DIR, gnubg's system table
    directory), or a path. Returns None for a falsey spec, meaning "keep
    the network's implicit MET".

    'auto' resolves to DEFAULT_MET: the copy bundled at
    met_tables/Kazaross-XG2.xml, falling back to gnubg's system table;
    it raises if neither file exists. Content is not validated here --
    call validate_met_file() if you need a guarantee.
    """
    if not spec:
        return None
    if spec in ("auto", "default"):
        # Check existence: silently returning a missing path yields
        # wrong equities rather than an error.
        if not os.path.isfile(DEFAULT_MET):
            raise ValueError(
                f"--met {spec!r}: no Kazaross-XG2 table found (looked for "
                f"{_BUNDLED_MET!r} and {_SYSTEM_MET!r}). Restore the "
                f"bundled file or pass --met <path> explicitly."
            )
        return DEFAULT_MET
    if os.path.isfile(spec):
        return os.path.abspath(spec)
    stem = spec if spec.endswith(".xml") else spec + ".xml"
    candidate = os.path.join(MET_DIR, stem)
    if os.path.isfile(candidate):
        return candidate
    names = ", ".join(
        os.path.basename(p)[:-4] for p in available_mets()
    ) or "(none installed)"
    raise ValueError(
        f"--met {spec!r} is neither a file nor a table in {MET_DIR}. "
        f"Available: {names}"
    )


def available_mets() -> List[str]:
    if not os.path.isdir(MET_DIR):
        return []
    return sorted(
        os.path.join(MET_DIR, f)
        for f in os.listdir(MET_DIR)
        if f.endswith(".xml") and f not in ("catalog.xml",)
    )
