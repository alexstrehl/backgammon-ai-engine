"""Match equity tables (MET), parsed from gnubg's XML format.

Pins quantities with exact known values (v_dp, game-boundary MWC) to a
real table instead of the network's implicit, drift-prone MET.

The tables are gnubg's own XML files (/usr/share/gnubg/met/), so the
values here are bit-identical to what gnubg scores us against —
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
# install; fall back to gnubg's system table. Both are byte-identical
# (md5 0f0f4684). The bundled file carries its own all-permissive
# copying notice -- keep it intact.
_BUNDLED_MET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "met_tables", "Kazaross-XG2.xml",
)
DEFAULT_MET = (_BUNDLED_MET if os.path.isfile(_BUNDLED_MET)
               else os.path.join(MET_DIR, "Kazaross-XG2.xml"))

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

    Accepts 'auto' (gnubg's default table), a bare table name such as
    'snowie', or a path. Returns None for a falsey spec, meaning "keep
    the network's implicit MET".

    'auto' resolves to DEFAULT_MET: the copy bundled at
    met_tables/Kazaross-XG2.xml, falling back to gnubg's system table;
    it raises if the resolved file does not exist. Content is not
    validated here — call validate_met_file() if you need a guarantee.
    """
    if not spec:
        return None
    if spec in ("auto", "default"):
        # Check existence: silently returning a missing path yields
        # wrong equities rather than an error.
        if not os.path.isfile(DEFAULT_MET):
            raise ValueError(
                f"--met {spec!r} resolved to {DEFAULT_MET!r}, which does "
                f"not exist. Pass --met <path> explicitly, or use the "
                f"bundled met_tables/Kazaross-XG2.xml."
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
