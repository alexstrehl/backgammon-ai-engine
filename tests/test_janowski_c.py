"""p5jan in C against janowski_match.py, the reference.

The conversion alone (c_api/puretd_janowski.c, no net):
  - every match state reachable in 3, 5, 7 and 11 point matches (game-start
    scores of a match played out, every cube value / owner the doubles
    reach in that game: 25 / 99 / 231 / 705 states), both players, x few
    hundred nested prob5 vectors (uniform, net-like float32 outputs clamped
    by model.prob5_postprocess, and the _EPS / 0 / 1 / all-gammon edges) x
    both x policies (the constant 0.68 and a gnubg position-class value):
    cube_decision's v_nd / v_dt / v_dp within 1e-9 and the same double /
    take verdicts; cubeful_mwc at every perspective, dead_mwc, get_points,
    do_cubeful on a subset
  - the loose grid (every my / opp 1..11, the Crawford variants, cube
    1..64, every owner: unreachable states included)
  - janowski.cube_x on self-play boards (contact, race, bar, both turns)
  - errors: an away above the MET (Python raises)
The numbers are printed (pytest -s); the bound is 1e-9, the port is
expected bitwise.
"""
import random

import numpy as np
import pytest
import torch

import janowski_match as JM
from backgammon_engine import BoardState, get_legal_plays, switch_turn
from encoding import CubePerspective
from janowski import cube_x
from janowski_impl import c_impl
from janowski_match import MatchCtx, ctx_for
from met import default_met
from model import prob5_postprocess
from modes import CubeOwner, MatchState

MET = default_met()
TOL = 1e-9
X_CONST = 0.68
PERSPS = (CubePerspective.CENTERED, CubePerspective.MINE, CubePerspective.THEIRS)


# ── match states ─────────────────────────────────────────────────────

def cube_states(ms):
    """Every (cube owner, value) the doubles reach from game start `ms`."""
    out, todo, seen = [], [ms], set()
    while todo:
        c = todo.pop()
        k = (c.cube_owner, c.cube_value)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        for p in (0, 1):
            if c.can_offer(p):
                todo.append(c.after_take(p))
    return out


def reachable(L):
    """Every decision state of an L-point match: the game starts of a match
    played out (each game ending 1 / 2 / 3 x any reached cube, either
    winner) and the cube states of each."""
    frontier = [MatchState(jacoby=False, match_length=L, my_away=L,
                           opp_away=L).start_game()]
    seen, starts = set(), []
    while frontier:
        ms = frontier.pop()
        key = (ms.my_away, ms.opp_away, ms.crawford_done, ms.is_crawford)
        if key in seen:
            continue
        seen.add(key)
        starts.append(ms)
        for c in cube_states(ms):
            for gr in (1, 2, 3):
                for w in (True, False):
                    nxt = c.after_game(gr, w)
                    if not nxt.match_over():
                        frontier.append(nxt.start_game())
    return [c for s in sorted(starts, key=lambda m: (m.my_away, m.opp_away,
                                                     m.is_crawford))
            for c in cube_states(s)]


N_REACHABLE = {3: 25, 5: 99, 7: 231, 11: 705}


# ── probability vectors ──────────────────────────────────────────────

def _class_x(rng):
    """A value of the gnubg position-class policy (janowski.cube_x)."""
    if rng.random() < 0.5:
        return 0.76
    return min(0.80, max(0.60, 0.60 + 0.0025 * rng.randint(0, 200)))


def prob_vectors(n=300, seed=0):
    """[(probs float64 (5,), class x)]: uniform nested, net-like, edges."""
    rng = random.Random(seed)
    out = []
    eps = JM._EPS
    for p in (0.0, 1.0, eps, np.nextafter(eps, 1.0), 1.0 - eps,
              np.nextafter(1.0 - eps, 0.0), 0.5, 1e-4, 1.0 - 1e-4):
        out.append(np.array([p, 0.0, 0.0, 0.0, 0.0]))
        out.append(np.array([p, p, p, 1.0 - p, 1.0 - p]))            # all gammons
        out.append(np.array([p, 0.5 * p, 0.1 * p, 0.3 * (1 - p), 0.0]))
    g = torch.Generator().manual_seed(seed)
    raw = torch.sigmoid(3.0 * torch.randn(80, 5, generator=g))
    out += list(prob5_postprocess(raw).numpy().astype(np.float64))
    while len(out) < n:
        p = rng.uniform(0.0, 1.0)
        wg = rng.uniform(0.0, p)
        wbg = rng.uniform(0.0, wg)
        lg = rng.uniform(0.0, 1.0 - p)
        lbg = rng.uniform(0.0, lg)
        out.append(np.array([p, wg, wbg, lg, lbg]))
    return [(v, _class_x(rng)) for v in out]


VECS = prob_vectors()


def _rates(probs):
    return JM._rates(probs)


class Stats:
    def __init__(self, name):
        self.name, self.n, self.max, self.exact = name, 0, 0.0, 0

    def add(self, a, b):
        d = abs(a - b)
        if not d <= TOL:                       # NaN fails too
            raise AssertionError(f"{self.name}: |{a!r} - {b!r}| = {d:.3e}")
        self.n += 1
        self.exact += a == b
        self.max = max(self.max, d)

    def report(self):
        print(f"\n{self.name}: {self.n} values, max |C - Python| = "
              f"{self.max:.3e}, bitwise equal {self.exact}/{self.n}")


# ── the conversion ───────────────────────────────────────────────────

@pytest.mark.parametrize("L", sorted(N_REACHABLE))
def test_reachable_state_count(L):
    states = reachable(L)
    assert len(states) == N_REACHABLE[L]
    keys = {(m.my_away, m.opp_away, m.crawford_done, m.is_crawford,
             m.cube_value, int(m.cube_owner)) for m in states}
    assert len(keys) == len(states)


@pytest.mark.parametrize("L", sorted(N_REACHABLE))
def test_cube_decision_every_reachable_state(L):
    """cube_decision (v_nd, v_dt at the doubled cube THEIRS, v_dp, the
    verdicts) for every reachable state, both players, every vector, both
    x policies."""
    J = c_impl()
    st = Stats(f"{L}pt cube_decision")
    n_dbl = n_take = 0
    for ms in reachable(L):
        for player in (0, 1):
            for probs, xc in VECS:
                for x in (X_CONST, xc):
                    a = JM.cube_decision(probs, ms, player, x, MET)
                    b = J.cube_decision(probs, ms, player, x, MET)
                    for k in ("v_nd", "v_dt", "v_dp"):
                        st.add(a[k], b[k])
                    assert a["should_double"] == b["should_double"]
                    assert a["take"] == b["take"]
                    n_dbl += a["should_double"]
                    n_take += a["take"]
    st.report()
    print(f"  states {len(reachable(L))} x 2 players x {len(VECS)} vectors x 2 x: "
          f"{n_dbl} doubles, {n_take} takes, all verdicts equal")
    assert n_dbl and n_take


@pytest.mark.parametrize("L", sorted(N_REACHABLE))
def test_conversion_pieces_every_reachable_state(L):
    """cubeful_mwc at all three perspectives, dead_mwc, get_points and
    do_cubeful from both players' views, on a subset of the vectors."""
    J = c_impl()
    sc, sd, sp = (Stats(f"{L}pt {n}") for n in ("cubeful_mwc", "dead_mwc", "get_points"))
    sub = VECS[::6]
    for ms in reachable(L):
        for player in (0, 1):
            ctx = ctx_for(ms, player)
            assert JM.do_cubeful(ctx) == J.do_cubeful(ctx)
            for probs, xc in sub:
                sd.add(JM.dead_mwc(probs, ctx, MET), J.dead_mwc(probs, ctx, MET))
                for persp in PERSPS:
                    for x in (X_CONST, xc):
                        sc.add(JM.cubeful_mwc(probs, persp, ctx, x, MET),
                               J.cubeful_mwc(probs, persp, ctx, x, MET))
                if 0.0 < probs[0] < 1.0:
                    r = _rates(probs)
                    for a, b in zip(JM.get_points(r, ctx, MET), J.get_points(r, ctx, MET)):
                        sp.add(a, b)
    for s in (sc, sd, sp):
        s.report()


def test_loose_grid_11pt():
    """Every (my, opp) of 1..11 x {no Crawford, the Crawford game,
    post-Crawford} where they apply x cube 1..64 x every owner, reachable
    or not, both players, 12 vectors x both x."""
    J = c_impl()
    st = Stats("11pt loose grid cube_decision")
    states = 0
    for my in range(1, 12):
        for opp in range(1, 12):
            one = my == 1 or opp == 1
            flags = [(True, True), (True, False)] if one else [(False, False)]
            for cd, ic in flags:
                for cube in (1, 2, 4, 8, 16, 32, 64):
                    for owner in (CubeOwner.CENTERED, CubeOwner.WHITE, CubeOwner.BLACK):
                        ms = MatchState(cube_owner=owner, cube_value=cube, jacoby=False,
                                        match_length=11, my_away=my, opp_away=opp,
                                        is_crawford=ic, crawford_done=cd)
                        states += 1
                        for player in (0, 1):
                            for probs, xc in VECS[::25]:
                                for x in (X_CONST, xc):
                                    a = JM.cube_decision(probs, ms, player, x, MET)
                                    b = J.cube_decision(probs, ms, player, x, MET)
                                    for k in ("v_nd", "v_dt", "v_dp"):
                                        st.add(a[k], b[k])
                                    assert a["should_double"] == b["should_double"]
                                    assert a["take"] == b["take"]
    st.report()
    print(f"  {states} states")


def _boards(n_games=40, seed=3):
    rng = random.Random(seed)
    out = []
    for _ in range(n_games):
        s = BoardState.initial()
        for _ in range(400):
            out.append(s)
            out.append(switch_turn(s))
            plays = get_legal_plays(s, (rng.randint(1, 6), rng.randint(1, 6)))
            if not plays:
                s = switch_turn(s)
                continue
            s = plays[rng.randrange(len(plays))][1]
            if s.is_game_over():
                break
    return out


def test_cube_x_matches_python():
    J = c_impl()
    boards = _boards()
    race = 0
    for s in boards:
        a, b = cube_x(s), J.cube_x(s)
        assert a == b, (s, a, b)
        race += a != 0.76
    print(f"\ncube_x: {len(boards)} boards, {race} races, all equal")
    assert race > 1000


def test_errors():
    """An away above the table: Python raises, C gives NaN / ValueError."""
    J = c_impl()
    probs = np.array([0.6, 0.2, 0.01, 0.1, 0.01])
    ctx = MatchCtx(MET._pre.shape[0] + 1, 3, False, False, 1)
    with pytest.raises(ValueError):
        JM.cubeful_mwc(probs, CubePerspective.CENTERED, ctx, 0.7, MET)
    assert np.isnan(J.cubeful_mwc(probs, CubePerspective.CENTERED, ctx, 0.7, MET))
    assert np.isnan(J.dead_mwc(probs, ctx, MET))
    ms = MatchState(jacoby=False, match_length=40, my_away=40, opp_away=3)
    with pytest.raises(ValueError):
        JM.cube_decision(probs, ms, 0, 0.7, MET)
    with pytest.raises(ValueError):
        J.cube_decision(probs, ms, 0, 0.7, MET)


# ══ the match engine on a prob5 net (BGC_KIND_MATCH_JANOWSKI) ═════════
#
# Positions: every checker decision of p5jan (torch) vs the TD matchplay net
# (torch) in 3, 7 and 11 point matches (the fixture style of
# tests/test_fast_forward.py); the cube calls where the mover can offer.
# Against the Python agent:
#   exact    C on <b> with the plain forward (simd: BGC_SIMD_DELTA0=0; ref) vs
#            the agent with that backend's net-only forward (FastForward):
#            every choice, every candidate's fp32 value, the cube triple, the
#            0-ply value, the 1-ply value and value_1ply_play bitwise
#   torch    C torch vs the torch agent: the 0-ply choices and cube triples
#            bitwise; 1-ply within 1e-6 (its batch differs)
#   bounds   production C simd (delta forward) vs the torch agent: choices
#            equal or near-ties (both choices within 1e-4 on the agent),
#            cube triples within 5e-5 (a verdict flips only within 1e-4 of
#            its threshold), 1-ply within 1e-5
# P5JAN_NETS=a.pt,b.pt adds nets (e.g. a 5-layer prob5 net).

import functools  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402

import bg_fast  # noqa: E402
from c_api import puretd_py as P  # noqa: E402
from c_api.fast_forward import FastForward  # noqa: E402
from janowski_match import JanowskiMatchAgent, _me  # noqa: E402
from model import ProbNetwork  # noqa: E402
from modes import cube_perspective  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P5 = os.path.join(REPO, "best_models", "cubeless_prob5_512_512_256_128.pt")
TD = os.path.join(REPO, "best_models", "matchplay_1h64met_512_512_256_256.pt")
NETS = [P5] + [p for p in os.environ.get("P5JAN_NETS", "").split(",") if p]
ENGINE_BACKENDS = [b for b in (os.environ.get("CJAN_BACKENDS") or "ref,simd,torch").split(",")
                   if b]
TIE = 1e-4


@pytest.fixture(autouse=True)
def _one_thread():
    torch.set_num_threads(1)


@functools.lru_cache(maxsize=None)
def _built(backend):
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    from test_c_api import _build
    ok, log = _build(backend)
    if not ok:
        if backend == "ref":
            pytest.fail(f"ref backend failed to build:\n{log[-3000:]}")
        pytest.skip(f"{backend} backend not built:\n{log[-500:]}")
    return backend


@functools.lru_cache(maxsize=None)
def jan_engine(net, backend, delta=False, memo=False, x_policy=None):
    """MATCH_JANOWSKI engine on `backend`, MET set; delta False: the plain
    forward (simd with BGC_SIMD_DELTA0=0); memo off unless asked."""
    _built(backend)
    env = {"BGC_MATCH_MEMO": "1" if memo else "0"}
    if backend == "simd" and not delta:
        env["BGC_SIMD_DELTA0"] = "0"
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        e = P.Engine(net, backend, janowski=True)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    e.set_met(MET)
    e.set_jan_x(x_policy)
    return e


@functools.lru_cache(maxsize=None)
def py_agent(net, ff=None, x_policy=None):
    """JanowskiMatchAgent; ff None = the torch forward, else that backend's
    net-only forward (the pre-C fast path)."""
    ag = JanowskiMatchAgent(ProbNetwork.load(net), met=MET, x_policy=x_policy)
    ag._fast_ff = None if ff is None else FastForward(ag.network, ff)
    return ag


def py_scores(ag, state, dice, ms):
    """The mover-view values choose_checker_action_cubeful ranks, with its
    own pieces in its own batching (bg_fast move generation, one forward);
    (values, successors), or None on a pass."""
    feats, lazy = bg_fast.get_legal_plays_encoded(state, dice)
    n = len(lazy)
    if n == 0:
        return None
    succ = [lazy[i] for i in range(n)]
    probs = ag._forward(feats)
    mover, opp = state.turn, 1 - state.turn
    opp_persp = cube_perspective(ms.cube_owner, opp)
    ctx_opp, ctx_me = ctx_for(ms, opp), ctx_for(ms, mover)
    vals = []
    for i, s in enumerate(succ):
        if s.is_game_over():
            vals.append(_me(MET, ctx_me, ctx_me.cube * s.game_result(), True))
        else:
            vals.append(1.0 - JM.cubeful_mwc(probs[i], opp_persp, ctx_opp, ag._x(s), MET))
    return np.array(vals), succ


@functools.lru_cache(maxsize=None)
def match_positions():
    """(state, dice, match_state) at every checker decision of p5jan vs the
    TD net, both torch: 3 x 3pt, 2 x 7pt, 1 x 11pt."""
    import eval_matchplay as em
    from model import TDNetwork
    from td_agent import TDAgent
    td = TDAgent(TDNetwork.load(TD), met=MET)
    jan = py_agent(P5)
    log = []

    class Rec:
        def __init__(self, a):
            self.a = a

        def offer_double(self, s, ms):
            return self.a.offer_double(s, ms)

        def respond_to_double(self, s, ms, hint=None):
            return self.a.respond_to_double(s, ms, hint=hint)

        def choose_checker_action_cubeful(self, s, d, ms):
            log.append((s.copy(), tuple(d), ms))
            return self.a.choose_checker_action_cubeful(s, d, ms)

    rng = random.Random(23)
    for length, n in ((3, 3), (7, 2), (11, 1)):
        for i in range(n):
            a, b = (Rec(jan), Rec(td)) if i % 2 == 0 else (Rec(td), Rec(jan))
            em.play_match(a, b, length, rng)
    return log


def test_positions_cover_the_cases():
    pos = match_positions()
    n_off = sum(ms.can_offer(s.turn) for s, _, ms in pos)
    n_owned = sum(ms.cube_owner != 0 for _, _, ms in pos)
    n_craw = sum(bool(ms.is_crawford) for _, _, ms in pos)
    n_post = sum(bool(ms.crawford_done and not ms.is_crawford) for _, _, ms in pos)
    n_race = sum(cube_x(s) != 0.76 for s, _, _ in pos)
    print(f"\n{len(pos)} positions: {n_off} can offer, {n_owned} owned cube, "
          f"{n_craw} Crawford, {n_post} post-Crawford, {n_race} race")
    assert len(pos) > 500 and n_off > 100 and n_owned > 20 and n_race > 20


def _check(eng, ag, s, d, ms, exact, st):
    """One position through the C engine vs the agent. exact: bitwise
    (choices, every candidate value, cube, 0-ply value); else the bounds.
    st: counters."""
    ref = py_scores(ag, s, d, ms)
    r = ag.choose_checker_action_cubeful(s, d, ms)
    idx, ns, _ = eng.match_best_move(s, d, ms)
    if ref is None:
        assert r is None and idx == -1
        st["pass"] += 1
    else:
        vals, succ = ref
        i_py = int(np.argmax(vals))        # numpy: the first maximum, as the agent
        assert r[0]._key() == succ[i_py]._key()
        cv, cs = eng.match_score_moves(s, d, ms)
        assert [x._key() for x in cs] == [x._key() for x in succ]
        assert succ[idx]._key() == ns._key()
        st["checker"] += 1
        if exact:
            assert idx == i_py
            assert np.array_equal(cv, (1.0 - vals).astype(np.float32))
        elif idx != i_py:
            gap = abs(vals[idx] - vals[i_py])
            assert gap < TIE, (s, d, ms, gap)
            st["near_tie"] += 1
            st["tie_max"] = max(st["tie_max"], gap)
        else:
            st["val_max"] = max(st["val_max"], float(np.abs(cv - (1.0 - vals)).max()))
    if ms.can_offer(s.turn):
        o = ag.offer_double(s, ms)
        dd, c = eng.match_offer(s, ms)
        st["cube"] += 1
        got = {k: getattr(c, k) for k in ("v_nd", "v_dt", "v_dp")}
        if exact:
            assert got == o._cache and dd == o.should_double
        else:
            for k in got:
                assert abs(got[k] - o._cache[k]) < 5e-5
                st["cube_max"] = max(st["cube_max"], abs(got[k] - o._cache[k]))
            q = o._cache
            if dd != o.should_double:
                assert abs(min(q["v_dt"], q["v_dp"]) - q["v_nd"]) < TIE
                st["cube_flip"] += 1
            if (c.v_dp > c.v_dt) != (q["v_dp"] > q["v_dt"]):
                assert abs(q["v_dp"] - q["v_dt"]) < TIE
                st["take_flip"] += 1
        v0 = eng.match_value(s, ms)
        assert v0 == np.float32(got["v_nd"])
    else:
        dd, c = eng.match_offer(s, ms)
        assert not dd and np.isnan(c.v_nd) and np.isnan(c.v_dt) and np.isnan(c.v_dp)


def _stats():
    return {"checker": 0, "pass": 0, "near_tie": 0, "tie_max": 0.0, "val_max": 0.0,
            "cube": 0, "cube_max": 0.0, "cube_flip": 0, "take_flip": 0, "oneply": 0,
            "oneply_max": 0.0}


def _oneply(eng, ag, s, d, ms, tol, st):
    a = ag.value_oneply(s, ms)
    b = eng.match_value_1ply(s, ms)
    idx, ns, v1 = eng.match_value_1ply_play(s, d, ms)
    i0, n0, _ = eng.match_best_move(s, d, ms)
    assert v1 == b and idx == i0 and (ns is None) == (n0 is None)
    if ns is not None:
        assert ns._key() == n0._key()
    if tol == 0:
        assert a == b, (a, b)
    else:
        assert abs(a - b) <= tol, (a, b)
    st["oneply"] += 1
    st["oneply_max"] = max(st["oneply_max"], abs(a - b))


def _report(name, st):
    print(f"\n{name}: {st['checker']} checker decisions ({st['pass']} passes), "
          f"{st['near_tie']} near-tie flips (max gap {st['tie_max']:.2e}), max fp32 value "
          f"diff {st['val_max']:.2e}; {st['cube']} cube triples (max diff "
          f"{st['cube_max']:.2e}, {st['cube_flip']} double / {st['take_flip']} take "
          f"flips); {st['oneply']} 1-ply values (max diff {st['oneply_max']:.2e})")


@pytest.mark.parametrize("net", NETS, ids=os.path.basename)
@pytest.mark.parametrize("backend", [b for b in ENGINE_BACKENDS if b in ("ref", "simd")])
def test_exact_vs_net_only_agent(net, backend):
    """C with the plain forward == the agent on the same backend's net-only
    forward, bit for bit (choices, values, cube, 1-ply)."""
    eng = jan_engine(net, backend)
    ag = py_agent(net, backend)
    pos = match_positions()
    st = _stats()
    for s, d, ms in pos:
        _check(eng, ag, s, d, ms, True, st)
    for s, d, ms in pos[:: 8 if backend == "simd" else 40]:
        _oneply(eng, ag, s, d, ms, 0, st)
    _report(f"exact {os.path.basename(net)} C {backend} plain vs agent ff-{backend}", st)


@pytest.mark.parametrize("net", NETS, ids=os.path.basename)
def test_torch_backend_vs_torch_agent(net):
    """C torch (ATen addmm, bitwise the torch forward per batch) vs the torch
    agent: 0-ply choices and cube triples bitwise (the same batches); 1-ply
    within 1e-6 (its batch differs: the pass row is in it, ATen's blocking
    depends on the shape)."""
    if "torch" not in ENGINE_BACKENDS:
        pytest.skip("CJAN_BACKENDS")
    eng = jan_engine(net, "torch")
    ag = py_agent(net)
    pos = match_positions()
    st = _stats()
    for s, d, ms in pos:
        _check(eng, ag, s, d, ms, True, st)
    for s, d, ms in pos[::40]:
        _oneply(eng, ag, s, d, ms, 1e-6, st)
    _report(f"torch {os.path.basename(net)} C torch vs torch agent", st)


@pytest.mark.parametrize("net", NETS, ids=os.path.basename)
def test_production_simd_vs_torch_agent(net):
    """The production path (simd, delta forward) vs the torch agent: the
    parity bounds."""
    if "simd" not in ENGINE_BACKENDS:
        pytest.skip("CJAN_BACKENDS")
    eng = jan_engine(net, "simd", delta=True)
    ag = py_agent(net)
    pos = match_positions()
    st = _stats()
    for s, d, ms in pos:
        _check(eng, ag, s, d, ms, False, st)
    for s, d, ms in pos[::8]:
        _oneply(eng, ag, s, d, ms, 1e-5, st)
    _report(f"bounds {os.path.basename(net)} C simd (delta) vs torch agent", st)


def test_constant_x_policy():
    """x_policy = 0.68: bitwise the agent with that policy; bgc_set_jan_x
    clears the memo (the values change with x)."""
    eng = jan_engine(P5, "simd", memo=True)
    ag68 = py_agent(P5, "simd", 0.68)
    ag = py_agent(P5, "simd")
    pos = match_positions()[::5]
    st = _stats()
    try:
        s0, _, ms0 = pos[3]
        v_class = eng.match_value_1ply(s0, ms0)
        assert v_class == ag.value_oneply(s0, ms0)
        eng.set_jan_x(0.68)
        for s, d, ms in pos:
            _check(eng, ag68, s, d, ms, True, st)
        v68 = eng.match_value_1ply(s0, ms0)
        assert v68 == ag68.value_oneply(s0, ms0) and v68 != v_class
    finally:
        eng.set_jan_x(None)
    assert eng.match_value_1ply(s0, ms0) == v_class
    _report("x = 0.68 C simd plain vs agent ff-simd", st)


def test_memo_bitwise():
    """The 1-ply memo (on by default) returns the computed value."""
    eng = jan_engine(P5, "simd", memo=True)
    ref = jan_engine(P5, "simd")
    pos = match_positions()[::20]
    for _ in range(2):
        for s, d, ms in pos:
            assert eng.match_value_1ply(s, ms) == ref.match_value_1ply(s, ms)
            assert eng.match_value_1ply_play(s, d, ms) == ref.match_value_1ply_play(s, d, ms)
    m = eng.match_memo_stats()
    assert m["on"] and m["hits"] >= len(pos)


# ── hand cases (tests/test_c_match.py's positions) ───────────────────

def _hand():
    from test_c_match import HAND
    return HAND


@pytest.mark.parametrize("case", range(20))
def test_hand_cases(case):
    """All-21 dance, match-winning / non-winning gammons and backgammons,
    Crawford, post-Crawford, dead cube, 2-away/2-away, DMP, owned cubes at
    25pt: exact on simd (plain) and ref, the bounds on production simd; the
    case's own property."""
    hand = _hand()
    if case >= len(hand):
        pytest.skip("fewer hand cases")
    name, s, d, ms = hand[case]
    if max(ms.my_away, ms.opp_away) > MET._pre.shape[0]:
        pytest.skip("above the MET")
    st = _stats()
    for b in ("simd", "ref"):
        eng, ag = jan_engine(P5, b), py_agent(P5, b)
        _check(eng, ag, s, d, ms, True, st)
        if not s.is_game_over():
            _oneply(eng, ag, s, d, ms, 0, st)
    _check(jan_engine(P5, "simd", delta=True), py_agent(P5), s, d, ms, False, st)
    eng, ag = jan_engine(P5, "simd"), py_agent(P5, "simd")
    if name.startswith("dance-all"):
        # every roll a pass: the 21 terms of p * pass_value, NOT the pass value
        sw = switch_turn(s)
        pv = 1.0 - JM.cubeful_mwc(ag._probs_state(sw), cube_perspective(ms.cube_owner, sw.turn),
                                  ctx_for(ms, sw.turn), ag._x(sw), MET)
        total = 0.0
        for d1 in range(1, 7):
            for d2 in range(d1, 7):
                total += (1.0 if d1 == d2 else 2.0) / 36.0 * pv
        assert eng.match_value_1ply(s, ms) == total
    if "gammon" in name or "backgammon" in name or "single" in name:
        vals, succ = eng.match_score_moves(s, d, ms)
        term = [i for i, x in enumerate(succ) if x.is_game_over()]
        assert term
        ctx_me = ctx_for(ms, s.turn)
        for i in term:
            want = _me(MET, ctx_me, ctx_me.cube * succ[i].game_result(), True)
            assert vals[i] == np.float32(1.0 - want)


# ── errors and refusals ──────────────────────────────────────────────

def test_open_and_refusals(tmp_path):
    eng = jan_engine(P5, "ref")
    assert eng.kind == P.KIND_MATCH_JANOWSKI and eng.encoder == "perspective196"
    s = BoardState.initial()
    ms = MatchState(jacoby=False, match_length=5, my_away=5, opp_away=5)
    with pytest.raises(RuntimeError):
        eng.best_move(s, (3, 1))                         # money entry points refuse it
    with pytest.raises(Exception):
        eng.collect_match(P.match_collect_opts(match_length=5), 1, 1)
    probs, _ = eng.value_prob5(s)
    assert np.array_equal(probs, py_agent(P5, "ref")._probs_state(s).astype(np.float32))
    row = eng.match_encode(s, ms)
    assert np.array_equal(row, bg_fast.encode_state(s))
    with pytest.raises(RuntimeError, match="not a prob5 export"):
        P.Engine(TD, "ref", janowski=True)
    td = P.Engine(TD, "ref")
    with pytest.raises(ValueError):
        td.set_jan_x(0.7)
    assert td.lib.bgc_set_jan_x(td.h, 1, 0.7) == -1
    with pytest.raises(ValueError):
        eng.set_jan_x(float("nan"))
    fresh = P.Engine(P5, "ref", janowski=True)          # no MET yet
    assert np.isnan(fresh.match_value_1ply(s, ms))
    big = MatchState(jacoby=False, match_length=40, my_away=40, opp_away=3)
    with pytest.raises(RuntimeError):
        eng.match_offer(s, big)                          # an away above the MET
