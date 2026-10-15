"""The C API's match-play core (c_api/puretd_match.c, BGC_KIND_MATCH).

Against the Python oracle (td_agent.TDAgent's matchplay methods, cpu,
boltzmann_temp 0, a MET grounding every scope; modes.MatchState;
met.MatchEquityTable):
  - bgc_rng_random == random.Random(seed).random() for 1e6 draws x 5 seeds;
    randint(1, n) for n = 1..64 and the worker-seed draw randint(0, 2**31-1)
  - bgc_match_new_game_mwc == TDAgent._eval_new_game (exact double) over
    every (my, opp, crawford_done, is_crawford, cube, result, winner,
    player) with aways 1..25, and over the states reachable by playing
    matches of lengths 1..25 out (tests/test_met_encoder.py's frontier)
  - the match block == Matchplay1H64MET[6]Encoder.match_extras (bit-exact
    float32, so the MET block == outcome_block[_x6] - 0.5) over every
    (my, opp 1..25, is_crawford, cube 1..128, perspective, is_cube_action),
    both players on roll, both encoders
  - bgc_match_encode == _encode_matchplay bitwise on every record of the
    matchplay_parity.py corpus, both is_cube_action, both encoders
  - EXACT (zero difference, zero flips) against the same agent with
    BG_FAST_FORWARD=<backend> (c_api/fast_forward.py) when C runs on that
    backend with the plain forward (simd: BGC_SIMD_DELTA0=0; ref): the
    0-ply values, every successor value, the 0-ply choice and bootstrap,
    offer / take (v_nd, v_dt, v_dp), the 1-ply value, value_1ply_play's
    choice and value; on corpus records and on the hand cases (all-21
    dance, match-winning and non-winning gammon / backgammon replies,
    Crawford, post-Crawford, dead cube, 2-away/2-away, DMP, owned cube 16
    at 25pt)
  - against the torch agent (no fast forward): within 1e-5 (0-ply values)
    / 1e-4 (1-ply), decisions equal or near-ties (C torch backend asserted;
    C simd with the production delta forward reported)
  - expand21's row dedup and the delta forward: dedup on/off bitwise; delta
    on within the 0-ply bound of delta off
  - errors: no MET, money calls on a MATCH engine and match calls on a
    money engine, a bad MET, encodings Python refuses
The corpus parity at full size is `matchplay_parity.py check-capi`.

Backends: ref and simd (CMATCH_BACKENDS / CAPI_BACKENDS override), built
and skipped as in tests/test_c_api.py. The corpus: $MATCHPLAY_CORPUS
(written by matchplay_parity.py gen); its tests skip without it. Sizes:
CMATCH_FULL=1 runs every corpus record in the exact and torch comparisons
(default: a spread of ~2,000 on simd, ~150 on ref; 1-ply on a quarter of
them).
"""
import functools
import os
import pickle
import random
import shutil
import sys

import numpy as np
import pytest

from test_c_api import _build, close, CAPI, REPO, TOL0, TOL1  # noqa: E402

from backgammon_engine import BoardState, BLACK, WHITE, get_legal_plays  # noqa: E402
from encoding import CubePerspective, get_encoder  # noqa: E402
from met import DEFAULT_MET, MatchEquityTable  # noqa: E402
from modes import CubeOwner, MatchState  # noqa: E402
from c_api import puretd_py as P  # noqa: E402
from c_api import matchplay_parity as MP  # noqa: E402

_env_b = (os.environ.get("CMATCH_BACKENDS") or os.environ.get("CAPI_BACKENDS")
          or "ref,simd").strip()
BACKENDS = [b for b in _env_b.split(",")
            if b and os.path.isfile(os.path.join(CAPI, f"backend_{b}.build.sh"))]
FULL = os.environ.get("CMATCH_FULL") == "1"

M350 = os.path.join(REPO, "best_models", "matchplay_1h64met_512_512_256_256.pt")
NETS = {"met": M350}
ENC = {"met": "matchplay1h64met_perspective196", "met6": "matchplay1h64met6_perspective196"}
CORPUS = os.environ.get("MATCHPLAY_CORPUS", "")   # unset: the corpus tests skip


@pytest.fixture(scope="session", params=BACKENDS)
def backend(request):
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    b = request.param
    ok, log = _build(b)
    if not ok:
        if b == "ref":
            pytest.fail(f"ref backend failed to build:\n{log[-3000:]}")
        pytest.skip(f"{b} backend not built:\n{log[-500:]}")
    return b


@pytest.fixture(params=sorted(NETS))
def net(request):
    if not os.path.isfile(NETS[request.param]):
        pytest.skip(f"{NETS[request.param]} not found")
    return request.param


@functools.lru_cache(maxsize=None)
def met():
    return MatchEquityTable.from_xml(DEFAULT_MET)


@functools.lru_cache(maxsize=None)
def engine(backend, net, delta=False):
    """MATCH engine, MET set, the 1-ply memo off (every call computes).
    delta=False: the plain forward (simd with BGC_SIMD_DELTA0=0), the
    exact-parity setting; True: the production default."""
    return MP.c_engine(NETS[net], "simd0" if backend == "simd" and not delta else backend, met())


@functools.lru_cache(maxsize=None)
def agent(net, ff):
    """TDAgent as the collector builds it; ff None = the torch path."""
    return MP.py_agent(NETS[net], ff, met())


@functools.lru_cache(maxsize=None)
def corpus():
    if not os.path.isfile(CORPUS):
        return None
    with open(CORPUS, "rb") as f:
        return pickle.load(f)


def need_corpus():
    c = corpus()
    if c is None:
        pytest.skip(f"no corpus at {CORPUS} (matchplay_parity.py gen; MATCHPLAY_CORPUS)")
    return c


def spread(seq, n):
    if n is None or n >= len(seq):
        return list(seq)
    step = len(seq) / n
    return [seq[int(i * step)] for i in range(n)]


def ms_of(L, my, opp, owner=0, cube=1, ic=False, cd=None):
    if cd is None:
        cd = ic or my == 1 or opp == 1
    return MatchState(cube_owner=CubeOwner(owner), cube_value=cube, jacoby=False,
                      match_length=L, my_away=my, opp_away=opp, is_crawford=ic,
                      crawford_done=cd)


# ── RNG ─────────────────────────────────────────────────────────────

SEEDS = (0, 1, 20261011, 2**31 - 1, 2**40 + 7)


@pytest.mark.parametrize("seed", SEEDS)
def test_rng_random(backend, seed):
    """random(): 1e6 draws, bitwise the CPython stream."""
    r, p = P.Rng(seed, backend), random.Random(seed)
    n = 10**6
    got = np.fromiter((r.random() for _ in range(n)), dtype=np.float64, count=n)
    want = np.fromiter((p.random() for _ in range(n)), dtype=np.float64, count=n)
    assert np.array_equal(got, want)
    # the stream stays aligned with the integer draws after it
    assert [r.randint(1, 6) for _ in range(100)] == [p.randint(1, 6) for _ in range(100)]


def test_rng_randint(backend):
    """randint(1, n) for n = 1..64 (the curriculum draws) and the worker /
    per-episode seed draw randint(0, 2**31 - 1), interleaved with random()."""
    for seed in SEEDS:
        r, p = P.Rng(seed, backend), random.Random(seed)
        for n in range(1, 65):
            assert [r.randint(1, n) for _ in range(300)] == [p.randint(1, n) for _ in range(300)]
            assert r.random() == p.random()
        assert ([r.randint(0, 2**31 - 1) for _ in range(5000)]
                == [p.randint(0, 2**31 - 1) for _ in range(5000)])


# ── MET: new-game MWC and the match block ───────────────────────────

def test_new_game_mwc_grid(backend):
    """bgc_match_new_game_mwc == TDAgent._eval_new_game (scope boundary;
    every scope is the same table) exactly, after every after_game of every
    (my, opp 1..25, crawford_done, is_crawford, cube 1..64, result, winner),
    for both players."""
    e = engine(backend, "met")
    ag = agent("met", "torch")
    n = 0
    for my in range(1, 26):
        for opp in range(1, 26):
            for cd in (False, True):
                for ic in (False, True):
                    for cube in (1, 2, 4, 8, 16, 32, 64):
                        ms = ms_of(25, my, opp, 0, cube, ic, cd)
                        for gr in (1, 2, 3):
                            for w in (True, False):
                                a = ms.after_game(gr, winner_is_me=w)
                                for pl in (0, 1):
                                    c = e.match_new_game_mwc(a, pl)
                                    assert c == ag._eval_new_game(a, pl, scope="boundary"), \
                                        (my, opp, cd, ic, cube, gr, w, pl)
                                    n += 1
    assert n == 25 * 25 * 2 * 2 * 7 * 3 * 2 * 2


def _reachable(L):
    """(my, opp, crawford_done, is_crawford) at game starts of an L-point
    match played out (test_met_encoder's frontier), with every cube level
    up to the match length."""
    frontier = [MatchState(match_length=L, my_away=L, opp_away=L, jacoby=False).start_game()]
    seen = set()
    while frontier:
        ms = frontier.pop()
        key = (ms.my_away, ms.opp_away, ms.crawford_done, ms.is_crawford)
        if key in seen:
            continue
        seen.add(key)
        for cube in (1, 2, 4, 8, 16, 32):
            for result in (1, 2, 3):
                for me in (True, False):
                    nxt = ms_of(L, ms.my_away, ms.opp_away, 0, cube, ms.is_crawford,
                                ms.crawford_done).after_game(result, me)
                    if not nxt.match_over():
                        frontier.append(nxt.start_game())
    return seen


def test_new_game_mwc_reachable(backend):
    """The same on the states reachable at lengths 1..25."""
    e = engine(backend, "met")
    ag = agent("met", "torch")
    n = 0
    for L in range(1, 26):
        for my, opp, cd, ic in _reachable(L):
            for cube in (1, 2, 4, 8, 16, 32, 64):
                ms = ms_of(L, my, opp, 0, cube, ic, cd)
                for gr in (1, 2, 3):
                    for w in (True, False):
                        a = ms.after_game(gr, winner_is_me=w)
                        for pl in (0, 1):
                            assert e.match_new_game_mwc(a, pl) == \
                                ag._eval_new_game(a, pl, scope="terminal")
                            n += 1
    assert n > 10000


def test_match_block_grid(backend, net):
    """bgc_match_encode == _encode_matchplay bitwise (the 141 one-hot / flag
    columns and the MET block, float32) over every (my, opp 1..25,
    is_crawford, cube 1..128, cube owner, is_cube_action) with WHITE and
    with BLACK on roll; the block itself == outcome_block[_x6] - 0.5."""
    e = engine(backend, net)
    ag = agent(net, "torch")
    enc = ag.encoder
    nb = 196
    rng = random.Random(5)
    boards = [BoardState.initial()]
    s = BoardState.initial()
    s.turn = BLACK
    boards.append(s)
    n = 0
    for my in range(1, 26):
        for opp in range(1, 26):
            for ic in (False, True):
                for cube in (1, 2, 4, 8, 16, 32, 64, 128):
                    for owner in (0, 1, 2):
                        ms = ms_of(25, my, opp, owner, cube, ic)
                        b = boards[rng.randrange(2)]
                        for ica in (False, True):
                            a = e.match_encode(b, ms, ica)
                            want = ag._encode_matchplay(b, ms, ica)
                            assert a.dtype == want.dtype == np.float32
                            assert np.array_equal(a, want), (my, opp, ic, cube, owner, ica, b.turn,
                                                             np.nonzero(a != want)[0])
                            n += 1
                        v = b.turn
                        mv, ov = (my, opp) if v == 0 else (opp, my)
                        blk = (met().outcome_block if net == "met" else met().outcome_block_x6)(
                            mv, ov, mv == 1 or ov == 1, ic, cube)
                        assert np.array_equal(a[nb + 141:], blk - np.float32(0.5))
    assert enc.num_features == e.input_size
    print(f"\n[cmatch] {net}: {n} match blocks bitwise")


def test_encode_corpus(backend, net):
    """bgc_match_encode == _encode_matchplay on every corpus record, both
    is_cube_action."""
    recs = need_corpus()
    e = engine(backend, net)
    ag = agent(net, "torch")
    n = 0
    for kind, s, d, mst, src in recs:
        ms = MP._ms_from(mst)
        if ms.match_over():
            continue
        for ica in (False, True):
            a = e.match_encode(s, ms, ica)
            b = ag._encode_matchplay(s, ms, ica)
            assert np.array_equal(a, b), (kind, mst, ica)
            n += 1
    print(f"\n[cmatch] {net}: {n} corpus rows bitwise")


# ── values and decisions on the corpus ─────────────────────────────

class Tally:
    def __init__(self):
        self.n = self.n1 = self.succ = 0
        self.max0 = self.max1 = self.maxc = 0.0
        self.flips = []

    def add(self, r):
        self.n += 1
        self.succ += r.get("n_moves", 0)
        self.max0 = max(self.max0, r["v0_err"], r.get("succ_err", 0.0))
        self.maxc = max(self.maxc, r.get("cube_err", 0.0))
        if "oneply_err" in r:
            self.n1 += 1
            self.max1 = max(self.max1, r["oneply_err"])
        if r.get("flip"):
            self.flips.append(("checker", r["flip_margin_c"]))
        if r.get("offer_flip"):
            self.flips.append(("offer", r["offer_margin_c"]))
        if r.get("take_flip"):
            self.flips.append(("take", r["take_margin_c"]))

    def line(self, what):
        return (f"[cmatch] {what}: {self.n} records, {self.succ} successors, {self.n1} 1-ply; "
                f"max|d| 0-ply {self.max0:.3g}, cube {self.maxc:.3g}, 1-ply {self.max1:.3g}; "
                f"{len(self.flips)} flips {sorted(self.flips, key=lambda x: -x[1])[:5]}")


def assert_exact(r, where):
    """A capi_compare result with zero differences."""
    assert r["enc_ok"], where
    assert r["v0_err"] == 0.0, (where, r)
    if "n_moves" in r:
        assert r["succ_states_ok"], where
        if r["n_moves"]:
            assert r["succ_bitwise"] and not r["flip"] and r["choice_value_ok"], (where, r)
        else:
            assert r["pass_ok"], where
    if "cube_err" in r:
        assert r["cube_bitwise"] and not r["offer_flip"] and not r["take_flip"], (where, r)
    else:
        assert r["no_offer_ok"], where
    if "oneply_err" in r:
        assert r["oneply_bitwise"], (where, r)
        assert r.get("v1play_ok", True), (where, r)


def assert_bounds(r, where):
    """Values within the bounds, decisions equal or near-ties
    (the C values of the two choices within 1e-5)."""
    assert r["enc_ok"], where
    assert r["v0_err"] <= TOL0 and r.get("succ_err", 0.0) <= TOL0, (where, r)
    assert r.get("cube_err", 0.0) <= TOL0, (where, r)
    assert r.get("oneply_err", 0.0) <= TOL1, (where, r)
    if r.get("flip"):
        assert r["flip_margin_c"] <= TOL0, (where, r)
    if r.get("offer_flip"):
        assert r["offer_margin_c"] <= TOL0, (where, r)
    if r.get("take_flip"):
        assert r["take_margin_c"] <= TOL0, (where, r)
    assert r.get("v1play_ok", True), (where, r)


def corpus_points(backend):
    recs = [r for r in need_corpus() if not MP._ms_from(r[3]).match_over()]
    n = None if FULL else (150 if backend == "ref" else 2000)
    return spread(recs, n)


def test_corpus_exact_vs_fast_forward(backend, net):
    """The corpus, exact half: C on `backend` (plain forward) against the
    agent with BG_FAST_FORWARD=`backend`: zero differences, zero flips."""
    e = engine(backend, net)
    ag = agent(net, backend)
    t = Tally()
    for i, (kind, s, d, mst, src) in enumerate(corpus_points(backend)):
        r = MP.capi_compare(e, ag, s, d, MP._ms_from(mst), oneply=(i % 4 == 0))
        assert_exact(r, (i, kind, mst))
        t.add(r)
    print("\n" + t.line(f"{net} C {backend} vs py {backend}"))
    assert t.max0 == t.max1 == t.maxc == 0.0 and not t.flips


def test_corpus_vs_torch(backend, net):
    """The corpus against the torch agent: C on the plain forward and
    (simd) on the production delta forward, within the bounds."""
    ag = agent(net, None)
    pts = corpus_points(backend)
    for delta in ((False, True) if backend == "simd" else (False,)):
        e = engine(backend, net, delta)
        t = Tally()
        for i, (kind, s, d, mst, src) in enumerate(pts):
            r = MP.capi_compare(e, ag, s, d, MP._ms_from(mst), oneply=(i % 4 == 0))
            t.add(r)
            assert_bounds(r, (i, kind, mst))
        print("\n" + t.line(f"{net} C {backend}{' delta' if delta else ''} vs py torch"))


def test_c_torch_backend_vs_torch(net):
    """The bounds' reference recipe: C on the torch backend (bitwise torch per
    call) against the torch agent, within the bounds (batch composition
    moves ~1e-7)."""
    ok, log = _build("torch")
    if not ok:
        pytest.skip(f"torch backend not built:\n{log[-500:]}")
    e = engine("torch", net)
    ag = agent(net, None)
    t = Tally()
    for i, (kind, s, d, mst, src) in enumerate(corpus_points("simd")[:None if FULL else 600]):
        r = MP.capi_compare(e, ag, s, d, MP._ms_from(mst), oneply=(i % 4 == 0))
        assert_bounds(r, (i, kind, mst))
        t.add(r)
    print("\n" + t.line(f"{net} C torch vs py torch"))


# ── hand cases ──────────────────────────────────────────────────────

def board(points, bar=(0, 0), off=(0, 0), turn=WHITE):
    p = [0] * 24
    for i, v in points.items():
        p[i] = v
    return BoardState(points=p, bar=list(bar), off=list(off), turn=turn)


def dance_all():
    """WHITE on the bar against BLACK's closed home board: all 21 rolls dance."""
    return board({**{i: -2 for i in range(18, 24)}, 0: -3, 5: 10, 7: 4}, bar=(1, 0))


def gammon_finish():
    """WHITE's last two checkers on its 1-point: every roll bears off with a
    gammon (BLACK has nothing off and nothing in WHITE's home)."""
    return board({0: 2, 12: -15}, off=(13, 0))


def backgammon_mix():
    """WHITE bears off; some rolls end the game with a backgammon (a BLACK
    checker in WHITE's home), others do not."""
    return board({0: 1, 3: 1, 5: 1, 2: -1, 13: -14}, off=(12, 0))


def single_mix():
    """Both sides have borne off: some rolls win a single game."""
    return board({0: 1, 4: 1, 23: -2, 20: -1}, off=(13, 12))


def midgame():
    s = BoardState.initial()
    s = get_legal_plays(s, (3, 1))[0][1]
    return get_legal_plays(s, (6, 4))[2][1]


HAND = [
    # (name, state, dice, match state)
    ("dance-all 5pt", dance_all(), (6, 5), ms_of(5, 3, 4)),
    ("dance-all DMP", dance_all(), (2, 2), ms_of(1, 1, 1)),
    ("gammon wins match", gammon_finish(), (2, 1), ms_of(5, 2, 4)),
    ("gammon non-winning", gammon_finish(), (6, 6), ms_of(7, 5, 3)),
    ("gammon cube 2 wins", gammon_finish(), (3, 1), ms_of(7, 4, 6, owner=2, cube=2)),
    ("backgammon wins match", backgammon_mix(), (6, 6), ms_of(5, 3, 2)),
    ("backgammon non-winning", backgammon_mix(), (4, 4), ms_of(11, 7, 9, owner=1, cube=1)),
    ("backgammon BLACK on roll wins match", board({23: -2, 20: 1, 10: 14}, off=(0, 13),
                                                  turn=BLACK), (6, 3), ms_of(5, 2, 3)),
    ("single race", single_mix(), (5, 1), ms_of(3, 2, 2)),
    ("crawford", midgame(), (5, 2), ms_of(5, 1, 3, ic=True, cd=True)),
    ("crawford trailer on roll", midgame(), (5, 2), ms_of(5, 3, 1, ic=True, cd=True)),
    ("post-crawford trailer", midgame(), (4, 3), ms_of(5, 3, 1, ic=False, cd=True)),
    ("post-crawford leader (dead cube)", midgame(), (4, 3), ms_of(5, 1, 3, ic=False, cd=True)),
    ("dead cube owned", midgame(), (2, 1), ms_of(7, 2, 5, owner=1, cube=2)),
    ("2-away/2-away", midgame(), (6, 1), ms_of(7, 2, 2)),
    ("2-away/2-away owned", midgame(), (6, 1), ms_of(7, 2, 2, owner=2, cube=2)),
    ("DMP", midgame(), (3, 3), ms_of(3, 1, 1, ic=False, cd=True)),
    ("owned cube 16 at 25pt", midgame(), (6, 5), ms_of(25, 20, 18, owner=1, cube=16)),
    ("cube 32 at 25pt theirs", midgame(), (1, 2), ms_of(25, 25, 24, owner=2, cube=32)),
    ("1pt match", midgame(), (5, 5), ms_of(1, 1, 1, ic=False, cd=False)),
]


@pytest.mark.parametrize("name,state,dice,ms", HAND, ids=[h[0] for h in HAND])
def test_hand_cases(backend, net, name, state, dice, ms):
    """Each API on the hand positions: exact against the fast-forward agent,
    within the bounds against torch; and the case's own property."""
    e = engine(backend, net)
    r = MP.capi_compare(e, agent(net, backend), state, dice, ms)
    assert_exact(r, name)
    assert_bounds(MP.capi_compare(e, agent(net, None), state, dice, ms), name)
    mover = state.turn
    if name.startswith("dance-all"):
        assert all(not get_legal_plays(state, (a, b)) for a in range(1, 7) for b in range(a, 7))
        from backgammon_engine import switch_turn
        sw = switch_turn(state)
        forced = 1.0 - float(e.match_value(sw, ms, ms.can_offer(sw.turn)))
        assert e.match_value_1ply(state, ms) == forced        # unclamped, not the sum
    if "gammon" in name or "backgammon" in name or "single" in name:
        vals, succ = e.match_score_moves(state, dice, ms)
        term = [i for i, s in enumerate(succ) if s.is_game_over()]
        assert term, name
        for i in term:
            gr = succ[i].game_result()
            nm = ms.after_game(gr, winner_is_me=(mover == 0))
            want = np.float32(1.0 - (1.0 if nm.match_over() else
                                     agent(net, None)._eval_new_game(nm, mover, scope="terminal")))
            assert vals[i] == want, (name, i, gr)
        if "wins match" in name:
            assert any(ms.after_game(succ[i].game_result(), winner_is_me=(mover == 0)).match_over()
                       for i in term)
        if "non-winning" in name:
            assert not any(ms.after_game(succ[i].game_result(),
                                         winner_is_me=(mover == 0)).match_over() for i in term)
    if not ms.can_offer(mover):
        d, c = e.match_offer(state, ms)
        assert not d and np.isnan(c.v_nd) and np.isnan(c.v_dt) and np.isnan(c.v_dp)


# ── dedup, delta, the one-expansion play ────────────────────────────

def test_dedup_bitwise(backend, net, monkeypatch):
    """expand21's cross-roll row dedup changes no bit of the 1-ply value or
    value_1ply_play's result."""
    on = engine(backend, net)
    with monkeypatch.context() as m:
        m.setenv("BGC_EXPAND_DEDUP", "0")
        off = MP.c_engine(NETS[net], "simd0" if backend == "simd" else backend, met())
    pts = [p for p in corpus_points(backend)][::4] + [(None, h[1], h[2], MP._ms_tuple(h[3]), None)
                                                      for h in HAND]
    for kind, s, d, mst, src in pts:
        ms = MP._ms_from(mst)
        assert on.match_value_1ply(s, ms) == off.match_value_1ply(s, ms)
        if d is not None:
            assert on.match_value_1ply_play(s, d, ms) == off.match_value_1ply_play(s, d, ms)
    off.close()


def test_delta_within_bound(backend, net):
    """The production delta forward (simd) against the plain forward: values
    within the 0-ply bound, decisions equal or near-ties; the maximum is
    reported (<= 7.2e-7 on the money nets). Other backends ignore
    the parent: bitwise."""
    on, off = engine(backend, net, True), engine(backend, net, False)
    w0 = w1 = 0.0
    near = 0
    for i, (kind, s, d, mst, src) in enumerate(corpus_points(backend)):
        ms = MP._ms_from(mst)
        if d is not None:
            a, sa = on.match_score_moves(s, d, ms)
            b, sb = off.match_score_moves(s, d, ms)
            assert sa == sb
            if len(a):
                w0 = max(w0, float(np.max(np.abs(a.astype(np.float64) - b))))
                ia, ib = int(np.argmin(a)), int(np.argmin(b))
                if ia != ib:
                    assert abs(float(a[ia]) - float(a[ib])) <= TOL0
                    near += 1
        if i % 4 == 0:
            v1, v0 = on.match_value_1ply(s, ms), off.match_value_1ply(s, ms)
            assert close(v1, v0, TOL0), (v1, v0)
            w1 = max(w1, abs(v1 - v0))
    assert w0 <= TOL0
    if backend != "simd":
        assert w0 == w1 == 0.0
    print(f"\n[cmatch] {net} {backend} delta vs plain: max|d| 0-ply {w0:.3g}, "
          f"1-ply {w1:.3g}, {near} near-tie choice changes")


# ── the encoder table, kinds, errors ───────────────────────────────

def test_encoder_table(backend):
    for name, n_in, nmet in (("matchplay1h64met_perspective196", 350, 13),
                             ("matchplay1h64met6_perspective196", 354, 17)):
        info = P.encoder_info(name, backend)
        enc = get_encoder(name)
        assert info is not None and info["match"] and info["nmet"] == nmet
        assert info["input_size"] == enc.num_features == n_in
        assert (info["n_base"], info["ncube"]) == (196, n_in - 196)
    for name in ("matchplay1h64met4_perspective196", "matchplay1h64met5_perspective196",
                 "matchplay1h64_perspective196", "matchplay_perspective196",
                 "matchplay1h64met_fancy196", "matchplay1h64met_"):
        assert P.encoder_info(name, backend) is None, name
    assert not P.encoder_info("perspective196", backend)["match"]


def test_kind_and_open(backend, net):
    e = engine(backend, net)
    assert e.kind == P.KIND_MATCH and e.encoder == ENC[net]
    assert e.input_size == (350 if net == "met" else 354)


def test_errors(backend, tmp_path):
    from test_c_api import PROD_CUBEFUL
    e = P.Engine(M350, backend)
    s, ms = midgame(), ms_of(5, 3, 4)
    # no MET yet
    assert np.isnan(e.match_value(s, ms))
    assert np.isnan(e.match_value_1ply(s, ms))
    assert np.isnan(e.match_new_game_mwc(ms, 0))
    with pytest.raises(RuntimeError):
        e.match_best_move(s, (3, 1), ms)
    with pytest.raises(RuntimeError):
        e.match_offer(s, ms)
    with pytest.raises(ValueError):
        e.match_encode(s, ms)
    # a bad MET
    bad = MatchEquityTable(met()._pre.copy(), met()._post.copy(), "x", 25, "x")
    bad._pre[3, 4] = np.nan
    with pytest.raises(ValueError):
        e.set_met(bad)
    e.set_met(met())
    # money entry points refuse a MATCH engine
    assert np.isnan(e.value(s, None))
    assert np.isnan(e.value_1ply(s, None))
    with pytest.raises(RuntimeError):
        e.best_move(s, (3, 1), None, 0)
    with pytest.raises(RuntimeError):
        e.score_moves(s, (3, 1), None, 0)
    with pytest.raises(RuntimeError):
        e.value_1ply_play(s, (3, 1), None)
    with pytest.raises(RuntimeError):
        e.offer_double(s, MatchState(), 0)
    with pytest.raises(RuntimeError):
        e.respond_to_double(s, MatchState(), 0)
    with pytest.raises(RuntimeError):
        e.collect(P.collect_opts(), 1, 1)
    with pytest.raises(RuntimeError):
        P.play_game(e, e, P.game_opts(), 1)
    # match entry points refuse a money engine
    m = P.Engine(PROD_CUBEFUL, backend)
    with pytest.raises(ValueError):
        m.set_met(met())
    assert np.isnan(m.match_value(s, ms))
    with pytest.raises(RuntimeError):
        m.match_best_move(s, (3, 1), ms)
    # encodings Python refuses: a finished match, cube 3, an away above the MET
    ag = agent("met", "torch")
    for bad_ms in (ms_of(5, 0, 3), ms_of(5, 3, 4, cube=3), ms_of(30, 26, 3)):
        with pytest.raises(Exception):
            ag._encode_matchplay(s, bad_ms)
        with pytest.raises(ValueError):
            e.match_encode(s, bad_ms)
        assert np.isnan(e.match_value(s, bad_ms))
    # ... but one the MET saturates (every outcome decides the match) encodes
    sat = ms_of(30, 26, 5, owner=0, cube=32)
    assert np.array_equal(e.match_encode(s, sat), ag._encode_matchplay(s, sat))
    # bad arguments
    assert e.lib.bgc_match_best_move(e.h, None, 3, 1, None, None, None) == -2
    with pytest.raises(RuntimeError):
        e.match_best_move(s, (0, 1), ms)
    e.close()
    m.close()


def test_memo_bitwise(backend, net):
    """The 1-ply memo (on by default; BGC_MATCH_MEMO=0 off): a repeated
    (board, match) returns the double the engine computed for it, so
    match_value_1ply and match_value_1ply_play are bitwise the memo-off
    engine's, first call or not; set_met clears it; counters."""
    on = MP.c_engine(NETS[net], "simd0" if backend == "simd" else backend, met(), memo=True)
    off = engine(backend, net)
    assert on.match_memo_stats()["on"] and not off.match_memo_stats()["on"]
    pts = corpus_points(backend)[:None if FULL else (40 if backend == "ref" else 400)]
    for rnd in range(2):
        for kind, s, d, mst, src in pts:
            ms = MP._ms_from(mst)
            assert on.match_value_1ply(s, ms) == off.match_value_1ply(s, ms)
            if d is not None:
                assert on.match_value_1ply_play(s, d, ms) == off.match_value_1ply_play(s, d, ms)
    st = on.match_memo_stats()
    # the key: the board (turn included) and the match state less its length
    distinct = len({(MP._key(p[1]),) + tuple(p[3][i] for i in (0, 1, 4, 5, 6, 7)) for p in pts})
    assert st["entries"] == distinct, (st, distinct)
    assert st["hits"] >= len(pts)                  # every second-round value_1ply hits
    on.set_met(met())
    assert on.match_memo_stats()["entries"] == 0
    from test_c_api import PROD_CUBEFUL
    m = P.Engine(PROD_CUBEFUL, backend)
    assert m.lib.bgc_match_memo_stats(m.h, None, None, None) == -1
    m.close()
    on.close()


# ── the fast move generator (c_api/puretd_movegen.c) ───────────────

def _selfplay_boards(rng, games, plies=200):
    """Boards of random-play games from the start position (both players
    opening), as Python BoardStates."""
    from backgammon_engine import BoardState as BS
    out = []
    for _ in range(games):
        s = BS()
        s.turn = rng.randrange(2)
        for _t in range(plies):
            out.append(s)
            plays = get_legal_plays(s, (rng.randint(1, 6), rng.randint(1, 6)))
            if not plays:
                s = BS(points=list(s.points), bar=list(s.bar), off=list(s.off), turn=1 - s.turn)
                continue
            s = rng.choice(plays).resulting_state
            if s.is_game_over():
                break
    return out


def test_fast_movegen_identical(backend):
    """bgc_i_legal_plays (BGC_FAST_MOVEGEN, the MATCH kind's default) ==
    get_legal_plays: the same play list (count, order, num_moves, the used
    moves, resulting boards) on random boards (movegen_fuzz.rand_board:
    contact, bar-heavy, bear-off) and random-play boards, all 36 ordered
    rolls."""
    import ctypes
    from bg_fast import _CBoardState, _CPlay, _python_to_c, MAX_PLAYS as BMAX
    sys.path.insert(0, CAPI)
    import movegen_fuzz
    lib = P.load_lib(backend)
    f = lib.bgc_test_legal_plays
    f.argtypes = [ctypes.POINTER(_CBoardState), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                  ctypes.POINTER(_CPlay)]
    f.restype = ctypes.c_int
    a, b = (_CPlay * BMAX)(), (_CPlay * BMAX)()
    ctypes.memset(b, 0xAB, ctypes.sizeof(b))      # stale bytes differ between the buffers
    rng = random.Random(20261011)
    n_boards = 4000 if FULL else (300 if backend == "ref" else 1500)
    boards = [movegen_fuzz.rand_board(rng) for _ in range(n_boards)]
    boards += _selfplay_boards(rng, n_boards // 100)
    calls = plays = 0
    for bs in boards:
        cs = _python_to_c(bs)
        for d1 in range(1, 7):
            for d2 in range(1, 7):
                na = f(ctypes.byref(cs), d1, d2, 0, a)
                nb = f(ctypes.byref(cs), d1, d2, 1, b)
                assert na == nb, (bs, d1, d2, na, nb)
                for i in range(na):
                    pa, pb = a[i], b[i]
                    assert pa.num_moves == pb.num_moves, (bs, d1, d2, i)
                    assert all(pa.moves[j].src == pb.moves[j].src
                               and pa.moves[j].dst == pb.moves[j].dst
                               for j in range(pa.num_moves)), (bs, d1, d2, i)
                    assert bytes(pa.resulting_state) == bytes(pb.resulting_state), (bs, d1, d2, i)
                calls += 1
                plays += na
    assert calls == 36 * len(boards) and plays > 5 * calls


def test_fast_movegen_bitwise(backend, net, monkeypatch):
    """BGC_FAST_MOVEGEN=0 (get_legal_plays) and the default give the same
    bits: 0-ply choice and scores, the 1-ply value, value_1ply_play."""
    on = engine(backend, net)
    with monkeypatch.context() as m:
        m.setenv("BGC_FAST_MOVEGEN", "0")
        off = MP.c_engine(NETS[net], "simd0" if backend == "simd" else backend, met())
    pts = [p for p in corpus_points(backend)][::4] + [(None, h[1], h[2], MP._ms_tuple(h[3]), None)
                                                      for h in HAND]
    for kind, s, d, mst, src in pts:
        ms = MP._ms_from(mst)
        assert on.match_value_1ply(s, ms) == off.match_value_1ply(s, ms)
        if d is not None:
            assert on.match_value_1ply_play(s, d, ms) == off.match_value_1ply_play(s, d, ms)
            bo, bf = on.match_best_move(s, d, ms), off.match_best_move(s, d, ms)
            assert bo[:2] == bf[:2] and (bo[2] == bf[2] or (np.isnan(bo[2]) and np.isnan(bf[2])))
    off.close()


# ── the shared 1-ply memo (BGC_MATCH_SHM) ──────────────────────────

def _shm_names():
    try:
        return {n for n in os.listdir("/dev/shm") if n.startswith(f"bgc_mm_{os.getuid()}_")}
    except OSError:
        return set()


def _shm_engine(binpath, backend, memo=True, shm=True):
    old = {k: os.environ.get(k) for k in ("BGC_MATCH_MEMO", "BGC_MATCH_SHM")}
    os.environ["BGC_MATCH_MEMO"] = "1" if memo else "0"
    os.environ["BGC_MATCH_SHM"] = "1" if shm else "0"
    try:
        e = P.Engine(binpath, backend)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return e


def _shm_worker(args):
    root, binpath, backend, seed, n, barrier = args
    sys.path[:0] = [root, os.path.join(root, "c_engine")]
    from c_api import puretd_py as P2
    from met import DEFAULT_MET as DM, MatchEquityTable as MET
    barrier.wait()
    os.environ.pop("BGC_MATCH_SHM", None)
    e = P2.Engine(binpath, backend)
    e.set_met(MET.from_xml(DM))
    o = P2.match_collect_opts(match_length=3, oneply=True, rng_mode=1)
    enc, tgt, nd = e.collect_match(o, seed, n)
    st = e.match_shm_stats()
    e.close()
    return enc, tgt, nd, st


@pytest.mark.parametrize("bk", ["ref", "simd"])
def test_shm_memo_bitwise(bk, net):
    """The shared memo (default on with the memo, row-exact backends):
    engines on the same weights file contents and MET share one table; the
    values another engine put there are bitwise the ones this engine
    computes (collect_match rows / targets / discards == a memo-off
    engine's), in one process and across concurrent spawn processes;
    BGC_MATCH_SHM=0 or the memo off: not attached; the name is gone once
    the engines are closed. The one-process part runs on a MET of its own
    (one entry a 3pt match never reads moved by a random 1e-15 multiple),
    so its table is this test's alone even while other processes of this
    user (a concurrent suite) run engines on the same net; the cleanup is
    asserted when the table it created is attributable (the only name that
    appeared while it attached)."""
    if bk not in BACKENDS:
        pytest.skip(f"{bk} not built")
    _build(bk)
    binpath = P.export_bin(NETS[net])
    o = P.match_collect_opts(match_length=3, oneply=True, rng_mode=1)
    n = 1 if bk == "ref" else 24
    mu = MatchEquityTable.from_xml(DEFAULT_MET)
    mu._pre = np.array(mu._pre, dtype=np.float64)
    mu._pre[5, 6] += random.SystemRandom().randint(1, 10**6) * 1e-15
    off = _shm_engine(binpath, bk, memo=False)
    off.set_met(mu)
    assert not off.match_shm_stats()["on"]
    a = _shm_engine(binpath, bk)
    names0 = _shm_names()
    a.set_met(mu)
    created = _shm_names() - names0
    assert a.match_shm_stats()["on"]
    b = _shm_engine(binpath, bk)
    b.set_met(mu)
    want = {s: off.collect_match(o, s, n) for s in (11, 12)}
    ga = a.collect_match(o, 11, n)
    assert a.match_shm_stats()["puts"] > 0
    gb = b.collect_match(o, 11, n)                       # a's values, from the table
    sb = b.match_shm_stats()
    assert sb["hits"] > 0 and sb["hits"] >= sb["lookups"] // 2, sb
    gb2 = b.collect_match(o, 12, n)
    for got, s in ((ga, 11), (gb, 11), (gb2, 12)):
        assert np.array_equal(got[0], want[s][0]) and np.array_equal(got[1], want[s][1])
        assert got[2] == want[s][2]
    nos = _shm_engine(binpath, bk, shm=False)
    nos.set_met(mu)
    assert not nos.match_shm_stats()["on"] and nos.match_memo_stats()["on"]
    nos.close()
    a.close()
    b.close()
    if len(created) == 1:                                # a's table: unlinked at the first close
        assert not (created & _shm_names()), created
    if bk == "simd":                                     # concurrent processes
        import multiprocessing as mpc
        ctx = mpc.get_context("spawn")
        mgr = ctx.Manager()
        bar = mgr.Barrier(3)
        with ctx.Pool(3) as pool:
            rs = pool.map(_shm_worker, [(REPO, binpath, bk, 20 + i, 16, bar) for i in range(3)])
        mgr.shutdown()
        for i, (enc, tgt, nd, st) in enumerate(rs):
            assert st["on"], st
            w = off.collect_match(o, 20 + i, 16)
            assert np.array_equal(enc, w[0]) and np.array_equal(tgt, w[1]) and nd == w[2]
        assert sum(st["hits"] for *_, st in rs) > 0
    off.close()


def test_shm_memo_met_scoped(net):
    """A MET change re-attaches: set_met with another MET gives an engine
    another table (no hits from the first MET's values)."""
    if "simd" not in BACKENDS:
        pytest.skip("simd not built")
    _build("simd")
    binpath = P.export_bin(NETS[net])
    o = P.match_collect_opts(match_length=3, oneply=True, rng_mode=1)
    a = _shm_engine(binpath, "simd")
    a.set_met(met())
    a.collect_match(o, 31, 8)
    m2 = MatchEquityTable.from_xml(DEFAULT_MET)
    m2._pre = np.array(m2._pre, dtype=np.float64)
    m2._pre[5, 6] += 1e-12                                # 6-away / 7-away: unused at 3pt
    b = _shm_engine(binpath, "simd")
    b.set_met(m2)
    b.collect_match(o, 31, 8)
    assert b.match_shm_stats()["on"] and b.match_shm_stats()["hits"] == 0
    b.set_met(met())                                      # back: a's table
    b.collect_match(o, 31, 8)
    assert b.match_shm_stats()["hits"] > 0
    a.close()
    b.close()
